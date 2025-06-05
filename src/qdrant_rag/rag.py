import json
import os
import time
import uuid
from collections import defaultdict
from typing import List, Dict, Any, Optional

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from tqdm import tqdm

from src.ollama.client import OllamaClient
from .client import QdrantRAGClient

load_dotenv()

PROMPT = """
# Question:
{{QUESTION}}

# Context:
{{CONTEXT}}

# Short answer:
"""


class QdrantRAGManager:
    """Local RAG implementation using Qdrant directly with Ollama."""

    def __init__(
        self,
        data_path: str = "dataset/locomo10_rag.json",
        chunk_size: int = 500,
        k: int = 1,
        collection_name: str = "rag_memories",
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        ollama_base_url: Optional[str] = None
    ):
        """
        Initialize QdrantRAGManager.

        Args:
            data_path: Path to the conversation data
            chunk_size: Size of text chunks for processing
            k: Number of chunks to retrieve for context
            collection_name: Name of the Qdrant collection
            model: Ollama model for chat completion
            embedding_model: Ollama model for embeddings
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            ollama_base_url: Ollama server URL
        """
        # Ollama configuration
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Qdrant configuration
        self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))

        # Initialize clients
        self.ollama_client = OllamaClient(
            base_url=self.ollama_base_url,
            model=self.model,
            embedding_model=self.embedding_model
        )
        self.qdrant_client = QdrantRAGClient(
            host=self.qdrant_host,
            port=self.qdrant_port
        )

        # Configuration
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k
        self.collection_name = collection_name

        # Determine embedding dimension
        self._embedding_dim = None
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists with proper configuration."""
        if not self.qdrant_client.collection_exists(self.collection_name):
            # Get embedding dimension by testing with a sample text
            if self._embedding_dim is None:
                sample_embedding = self.calculate_embedding("test")
                self._embedding_dim = len(sample_embedding)

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vector_size=self._embedding_dim
            )
            print(f"Created Qdrant collection '{self.collection_name}' with dimension {self._embedding_dim}")

    def calculate_embedding(self, text: str) -> List[float]:
        """Calculate embeddings using Ollama."""
        return self.ollama_client.get_embeddings(text)

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def clean_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        """Convert chat history to clean text format."""
        cleaned_chat_history = ""
        for c in chat_history:
            cleaned_chat_history += f"{c['timestamp']} | {c['speaker']}: {c['text']}\n"
        return cleaned_chat_history

    def create_chunks(self, chat_history: List[Dict[str, Any]], chunk_size: int = 500) -> List[str]:
        """
        Create chunks using tiktoken for more accurate token counting.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except:
            encoding = tiktoken.get_encoding("gpt2")

        documents = self.clean_chat_history(chat_history)

        if chunk_size == -1:
            return [documents]

        chunks = []
        tokens = encoding.encode(documents)

        # Split into chunks based on token count
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)

        return chunks

    def store_conversation_chunks(self, conversation_id: str, chat_history: List[Dict[str, Any]]) -> List[str]:
        """
        Store conversation chunks in Qdrant and return chunk IDs.

        Args:
            conversation_id: Unique identifier for the conversation
            chat_history: List of chat messages

        Returns:
            List of chunk IDs stored in Qdrant
        """
        chunks = self.create_chunks(chat_history, self.chunk_size)
        chunk_ids = []
        points = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{conversation_id}_chunk_{i}"
            chunk_ids.append(chunk_id)

            # Calculate embedding
            embedding = self.calculate_embedding(chunk)

            # Prepare point for Qdrant
            points.append({
                "id": chunk_id,
                "vector": embedding,
                "payload": {
                    "conversation_id": conversation_id,
                    "chunk_index": i,
                    "text": chunk,
                    "timestamp": time.time()
                }
            })

        # Batch upsert to Qdrant
        if points:
            self.qdrant_client.upsert_points(self.collection_name, points)
            print(f"Stored {len(points)} chunks for conversation {conversation_id}")

        return chunk_ids

    def search_relevant_chunks(self, query: str, limit: Optional[int] = None) -> tuple[str, float]:
        """
        Search for relevant chunks in Qdrant.

        Args:
            query: Search query
            limit: Number of chunks to retrieve (uses self.k if None)

        Returns:
            Tuple of (combined_context, search_time)
        """
        if limit is None:
            limit = self.k

        t1 = time.time()

        # Get query embedding
        query_embedding = self.calculate_embedding(query)

        # Search in Qdrant
        results = self.qdrant_client.search_points(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        # Combine retrieved chunks
        if limit == 1 and results:
            combined_chunks = results[0]["payload"]["text"]
        else:
            combined_chunks = "\n<->\n".join([result["payload"]["text"] for result in results])

        t2 = time.time()
        return combined_chunks, t2 - t1

    def generate_response(self, question: str, context: str) -> tuple[str, float]:
        """Generate response using Ollama with the given context."""
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                response = self.ollama_client.generate_response(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can answer "
                            "questions based on the provided context."
                            "If the question involves timing, use the conversation date for reference."
                            "Provide the shortest possible answer."
                            "Use words directly from the conversation when possible."
                            "Avoid using subjects in your answer.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                t2 = time.time()
                return response, t2 - t1
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)

        raise Exception("Max retries exceeded without success")

    def clear_collection(self):
        """Clear all data from the Qdrant collection."""
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.clear_collection(self.collection_name)
            print(f"Cleared collection '{self.collection_name}'")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection."""
        if self.qdrant_client.collection_exists(self.collection_name):
            return self.qdrant_client.get_collection_info(self.collection_name)
        return {"error": "Collection does not exist"}

    def process_all_conversations(self, output_file_path: str):
        """
        Process all conversations from the dataset file.
        This method handles the full RAG pipeline including storage and retrieval.
        """
        with open(self.data_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)

        # First pass: Store all conversations in Qdrant
        print("Storing conversations in Qdrant...")
        for key, value in tqdm(data.items(), desc="Storing conversations"):
            chat_history = value["conversation"]
            self.store_conversation_chunks(key, chat_history)

        # Second pass: Process questions and generate answers
        print("Processing questions...")
        for key, value in tqdm(data.items(), desc="Processing conversations"):
            questions = value["question"]

            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item["question"]
                answer = item.get("answer", "")
                category = item["category"]

                if self.chunk_size == -1:
                    # Use full conversation as context
                    context = self.clean_chat_history(value["conversation"])
                    search_time = 0
                else:
                    context, search_time = self.search_relevant_chunks(question, limit=self.k)

                response, response_time = self.generate_response(question, context)

                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "context": context,
                        "response": response,
                        "search_time": search_time,
                        "response_time": response_time,
                    }
                )

                # Save incrementally
                with open(output_file_path, "w+") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        # Final save
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)

        print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--collection_name", type=str, default="rag_memories")
    args = parser.parse_args()

    rag_manager = QdrantRAGManager(
        chunk_size=args.chunk_size,
        k=args.k,
        collection_name=args.collection_name
    )
    rag_manager.process_all_conversations(args.output_file_path)