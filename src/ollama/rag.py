import json
import os
import time
from collections import defaultdict

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from tqdm import tqdm

from .client import OllamaClient

load_dotenv()

PROMPT = """
# Question:
{{QUESTION}}

# Context:
{{CONTEXT}}

# Short answer:
"""


class OllamaRAGManager:
    def __init__(self, data_path="dataset/locomo10_rag.json", chunk_size=500, k=1, model=None, embedding_model=None, base_url=None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = OllamaClient(base_url=self.base_url, model=self.model, embedding_model=self.embedding_model)
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k

    def generate_response(self, question, context):
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                response = self.client.generate_response(
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
                time.sleep(1)  # Wait before retrying

        # This should never be reached, but needed for type safety
        raise Exception("Max retries exceeded without success")

    def clean_chat_history(self, chat_history):
        cleaned_chat_history = ""
        for c in chat_history:
            cleaned_chat_history += f"{c['timestamp']} | {c['speaker']}: " f"{c['text']}\n"

        return cleaned_chat_history

    def calculate_embedding(self, document):
        return self.client.get_embeddings(document)

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query, chunks, embeddings, k=1):
        """
        Search for the top-k most similar chunks to the query.

        Args:
            query: The query string
            chunks: List of text chunks
            embeddings: List of embeddings for each chunk
            k: Number of top chunks to return (default: 1)

        Returns:
            combined_chunks: The combined text of the top-k chunks
            search_time: Time taken for the search
        """
        t1 = time.time()
        query_embedding = self.calculate_embedding(query)
        similarities = [self.calculate_similarity(query_embedding, embedding) for embedding in embeddings]

        # Get indices of top-k most similar chunks
        if k == 1:
            # Original behavior - just get the most similar chunk
            top_indices = [np.argmax(similarities)]
        else:
            # Get indices of top-k chunks
            top_indices = np.argsort(similarities)[-k:][::-1]

        # Combine the top-k chunks
        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])

        t2 = time.time()
        return combined_chunks, t2 - t1

    def create_chunks(self, chat_history, chunk_size=500):
        """
        Create chunks using tiktoken for more accurate token counting
        Note: For Ollama, we'll use tiktoken as an approximation since most models
        use similar tokenization patterns.
        """
        # Use a general encoding as approximation for Ollama models
        try:
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding as approximation
        except:
            # Fallback to a basic encoding if cl100k_base is not available
            encoding = tiktoken.get_encoding("gpt2")

        documents = self.clean_chat_history(chat_history)

        if chunk_size == -1:
            return [documents], []

        chunks = []

        # Encode the document
        tokens = encoding.encode(documents)

        # Split into chunks based on token count
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)

        embeddings = []
        for chunk in chunks:
            embedding = self.calculate_embedding(chunk)
            embeddings.append(embedding)

        return chunks, embeddings

    def process_all_conversations(self, output_file_path):
        with open(self.data_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)
        for key, value in tqdm(data.items(), desc="Processing conversations"):
            chat_history = value["conversation"]
            questions = value["question"]

            chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)

            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item["question"]
                answer = item.get("answer", "")
                category = item["category"]

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    context, search_time = self.search(question, chunks, embeddings, k=self.k)
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
                with open(output_file_path, "w+") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        # Save results
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    rag_manager = OllamaRAGManager(chunk_size=args.chunk_size, k=args.k)
    rag_manager.process_all_conversations(args.output_file_path)