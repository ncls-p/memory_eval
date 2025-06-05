import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from .rag import QdrantRAGManager

load_dotenv()


class QdrantRAGAdd:
    """Memory addition using Qdrant RAG for storing conversation chunks."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        batch_size: int = 2,
        collection_name: str = "rag_memories",
        chunk_size: int = 500,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        ollama_base_url: Optional[str] = None
    ):
        """
        Initialize QdrantRAGAdd.

        Args:
            data_path: Path to conversation data file
            batch_size: Number of conversations to process in batch
            collection_name: Name of the Qdrant collection
            chunk_size: Size of text chunks for processing
            model: Ollama model for chat completion
            embedding_model: Ollama model for embeddings
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            ollama_base_url: Ollama server URL
        """
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.chunk_size = chunk_size

        # Initialize RAG manager
        self.rag_manager = QdrantRAGManager(
            data_path=data_path or "dataset/locomo10.json",
            chunk_size=chunk_size,
            collection_name=collection_name,
            model=model,
            embedding_model=embedding_model,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            ollama_base_url=ollama_base_url
        )

        if data_path:
            self.load_data()

    def load_data(self):
        """Load conversation data from file."""
        if self.data_path:
            with open(self.data_path, "r") as f:
                self.data = json.load(f)
        return self.data

    def process_conversation_for_storage(self, item: Dict[str, Any], idx: int):
        """
        Process a single conversation and store its chunks in Qdrant.

        Args:
            item: Conversation item from dataset
            idx: Index of the conversation
        """
        conversation = item["conversation"]
        conversation_id = f"conv_{idx}"

        # Convert conversation format to list of messages
        chat_history = []

        # Extract speaker names
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        # Process all conversation segments
        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation.get(date_time_key, "")
            chats = conversation[key]

            # Add each message to chat history
            for chat in chats:
                chat_history.append({
                    "timestamp": timestamp,
                    "speaker": chat["speaker"],
                    "text": chat["text"]
                })

        # Store conversation chunks in Qdrant
        try:
            chunk_ids = self.rag_manager.store_conversation_chunks(conversation_id, chat_history)
            print(f"Processed conversation {conversation_id}: {len(chunk_ids)} chunks stored")
        except Exception as e:
            print(f"Error processing conversation {conversation_id}: {str(e)}")

    def add_memories_batch(self, conversations: List[Dict[str, Any]], start_idx: int):
        """
        Add memories for a batch of conversations.

        Args:
            conversations: List of conversation items
            start_idx: Starting index for conversation IDs
        """
        for i, conversation in enumerate(conversations):
            conversation_idx = start_idx + i
            self.process_conversation_for_storage(conversation, conversation_idx)

    def process_all_conversations(self, max_workers: int = 5, clear_existing: bool = True):
        """
        Process all conversations and store them in Qdrant.

        Args:
            max_workers: Maximum number of worker threads
            clear_existing: Whether to clear existing collection before adding
        """
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")

        # Clear existing collection if requested
        if clear_existing:
            print("Clearing existing Qdrant collection...")
            self.rag_manager.clear_collection()

        print(f"Processing {len(self.data)} conversations...")

        # Process conversations in batches using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            # Create batches
            for i in range(0, len(self.data), self.batch_size):
                batch = self.data[i:i + self.batch_size]
                future = executor.submit(self.add_memories_batch, batch, i)
                futures.append(future)

            # Wait for all batches to complete
            for future in tqdm(futures, desc="Processing conversation batches"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in batch processing: {str(e)}")

        # Print collection statistics
        stats = self.rag_manager.get_collection_stats()
        print(f"Collection statistics: {stats}")
        print("✓ All conversations processed and stored in Qdrant")

    def add_single_conversation(self, conversation: Dict[str, Any], conversation_id: str):
        """
        Add a single conversation to Qdrant.

        Args:
            conversation: Single conversation data
            conversation_id: Unique identifier for the conversation
        """
        # Convert to expected format if needed
        if "conversation" in conversation:
            conv_data = conversation
        else:
            conv_data = {"conversation": conversation}

        # Convert string ID to int for compatibility with existing method
        try:
            idx = int(conversation_id) if conversation_id.isdigit() else hash(conversation_id) % 10000
        except:
            idx = hash(conversation_id) % 10000

        self.process_conversation_for_storage(conv_data, idx)


def add_memories_qdrant_rag(
    conversations,
    output_folder: str,
    chunk_size: int = 500,
    collection_name: str = "rag_memories",
    clear_existing: bool = True,
    max_workers: int = 5
):
    """
    Add memories using Qdrant RAG implementation.

    Args:
        conversations: Path to conversation data or list of conversations
        output_folder: Output folder (not used in add operation but kept for compatibility)
        chunk_size: Size of text chunks for processing
        collection_name: Name of the Qdrant collection
        clear_existing: Whether to clear existing collection before adding
        max_workers: Maximum number of worker threads
    """
    print("Starting Qdrant RAG memory addition...")

    # Handle both file paths and direct data
    if isinstance(conversations, str):
        data_path = conversations
        memory_manager = QdrantRAGAdd(
            data_path=data_path,
            chunk_size=chunk_size,
            collection_name=collection_name
        )
    else:
        # If conversations is a list, save it temporarily
        data_path = "/tmp/temp_conversations_qdrant.json"
        with open(data_path, "w") as f:
            json.dump(conversations, f)

        memory_manager = QdrantRAGAdd(
            data_path=data_path,
            chunk_size=chunk_size,
            collection_name=collection_name
        )

    # Process all conversations
    memory_manager.process_all_conversations(
        max_workers=max_workers,
        clear_existing=clear_existing
    )

    print("✓ Qdrant RAG memory addition completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add memories using Qdrant RAG")
    parser.add_argument("--data_path", type=str, default="dataset/locomo10.json", help="Path to conversation data")
    parser.add_argument("--chunk_size", type=int, default=500, help="Size of text chunks")
    parser.add_argument("--collection_name", type=str, default="rag_memories", help="Qdrant collection name")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum worker threads")
    parser.add_argument("--no_clear", action="store_true", help="Don't clear existing collection")

    args = parser.parse_args()

    memory_manager = QdrantRAGAdd(
        data_path=args.data_path,
        chunk_size=args.chunk_size,
        collection_name=args.collection_name
    )

    memory_manager.process_all_conversations(
        max_workers=args.max_workers,
        clear_existing=not args.no_clear
    )