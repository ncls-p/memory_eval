import json
import os
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from .rag import QdrantRAGManager

load_dotenv()


class QdrantRAGSearch:
    """Memory search using Qdrant RAG for retrieving and generating responses."""

    def __init__(
        self,
        output_file_path: str,
        top_k: int = 30,
        filter_memories: bool = False,
        collection_name: str = "rag_memories",
        chunk_size: int = 500,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        ollama_base_url: Optional[str] = None
    ):
        """
        Initialize QdrantRAGSearch.

        Args:
            output_file_path: Path to save search results
            top_k: Number of top chunks to retrieve
            filter_memories: Whether to apply memory filtering (kept for compatibility)
            collection_name: Name of the Qdrant collection
            chunk_size: Size of text chunks for processing
            model: Ollama model for chat completion
            embedding_model: Ollama model for embeddings
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            ollama_base_url: Ollama server URL
        """
        self.output_file_path = output_file_path
        self.top_k = top_k
        self.filter_memories = filter_memories
        self.chunk_size = chunk_size

        # Initialize RAG manager
        self.rag_manager = QdrantRAGManager(
            chunk_size=chunk_size,
            k=top_k,  # Use top_k as the number of chunks to retrieve
            collection_name=collection_name,
            model=model,
            embedding_model=embedding_model,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            ollama_base_url=ollama_base_url
        )

    def search_memories(self, query: str, conversation_id: Optional[str] = None) -> tuple[str, float]:
        """
        Search for relevant memories in Qdrant.

        Args:
            query: Search query
            conversation_id: Optional conversation ID for filtering (not implemented yet)

        Returns:
            Tuple of (context, search_time)
        """
        return self.rag_manager.search_relevant_chunks(query, limit=self.top_k)

    def generate_response(self, question: str, context: str) -> tuple[str, float]:
        """Generate response using the RAG manager."""
        return self.rag_manager.generate_response(question, context)

    def process_single_question(
        self,
        question: str,
        answer: str = "",
        category: str = "",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single question and generate response.

        Args:
            question: The question to answer
            answer: Ground truth answer (for evaluation)
            category: Question category
            conversation_id: Optional conversation ID

        Returns:
            Dictionary with question, answer, response, and timing information
        """
        # Search for relevant context
        context, search_time = self.search_memories(question, conversation_id)

        # Generate response
        response, response_time = self.generate_response(question, context)

        return {
            "question": question,
            "answer": answer,
            "category": category,
            "context": context,
            "response": response,
            "search_time": search_time,
            "response_time": response_time,
            "conversation_id": conversation_id
        }

    def process_data_file(self, data_file_path: str):
        """
        Process a data file with questions and generate responses.

        Args:
            data_file_path: Path to the data file containing questions
        """
        # Load the data
        with open(data_file_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)

        # Check collection statistics
        stats = self.rag_manager.get_collection_stats()
        print(f"Collection statistics: {stats}")

        if stats.get("points_count", 0) == 0:
            print("Warning: No data found in Qdrant collection. Make sure to run 'add' method first.")
            return

        # Process each conversation's questions
        for key, value in tqdm(data.items(), desc="Processing questions"):
            if "question" in value:
                questions = value["question"]

                for item in tqdm(questions, desc=f"Answering questions for {key}", leave=False):
                    question = item["question"]
                    answer = item.get("answer", "")
                    category = item.get("category", "")

                    # Process the question
                    result = self.process_single_question(
                        question=question,
                        answer=answer,
                        category=category,
                        conversation_id=key
                    )

                    FINAL_RESULTS[key].append(result)

                    # Save incrementally
                    with open(self.output_file_path, "w") as f:
                        json.dump(FINAL_RESULTS, f, indent=4)

        # Final save
        with open(self.output_file_path, "w") as f:
            json.dump(FINAL_RESULTS, f, indent=4)

        print(f"Search results saved to {self.output_file_path}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection."""
        return self.rag_manager.get_collection_stats()

    def search_similar_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar chunks without generating responses.

        Args:
            query: Search query
            limit: Number of chunks to retrieve

        Returns:
            List of similar chunks with metadata
        """
        query_embedding = self.rag_manager.calculate_embedding(query)

        results = self.rag_manager.qdrant_client.search_points(
            collection_name=self.rag_manager.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        return [
            {
                "id": result["id"],
                "score": result["score"],
                "text": result["payload"]["text"],
                "conversation_id": result["payload"]["conversation_id"],
                "chunk_index": result["payload"]["chunk_index"]
            }
            for result in results
        ]


def search_memories_qdrant_rag(
    data_file_path: str,
    output_file_path: str,
    top_k: int = 30,
    filter_memories: bool = False,
    collection_name: str = "rag_memories",
    chunk_size: int = 500
):
    """
    Search memories using Qdrant RAG implementation.

    Args:
        data_file_path: Path to the data file with questions
        output_file_path: Path to save search results
        top_k: Number of top chunks to retrieve
        filter_memories: Whether to apply memory filtering (kept for compatibility)
        collection_name: Name of the Qdrant collection
        chunk_size: Size of text chunks
    """
    print("Starting Qdrant RAG memory search...")

    searcher = QdrantRAGSearch(
        output_file_path=output_file_path,
        top_k=top_k,
        filter_memories=filter_memories,
        collection_name=collection_name,
        chunk_size=chunk_size
    )

    searcher.process_data_file(data_file_path)

    print("✓ Qdrant RAG memory search completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search memories using Qdrant RAG")
    parser.add_argument("--data_file", type=str, default="dataset/locomo10.json", help="Path to questions data")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top chunks to retrieve")
    parser.add_argument("--collection_name", type=str, default="rag_memories", help="Qdrant collection name")
    parser.add_argument("--chunk_size", type=int, default=500, help="Size of text chunks")
    parser.add_argument("--filter_memories", action="store_true", help="Apply memory filtering")

    args = parser.parse_args()

    search_memories_qdrant_rag(
        data_file_path=args.data_file,
        output_file_path=args.output_file,
        top_k=args.top_k,
        filter_memories=args.filter_memories,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size
    )