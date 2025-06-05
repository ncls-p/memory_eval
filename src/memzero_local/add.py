import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import Memory
from .config import get_local_mem0_config, get_custom_instructions, validate_local_setup

load_dotenv()


class MemoryADDLocal:
    def __init__(self, data_path=None, batch_size=2, is_graph=False):
        # Validate local setup before initializing
        validate_local_setup()

        # Get local configuration
        config = get_local_mem0_config()

        # Initialize mem0ai with local configuration
        self.mem0_client = Memory.from_config(config)

        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.custom_instructions = get_custom_instructions()

        if data_path:
            self.load_data()

    def load_data(self):
        if self.data_path:
            with open(self.data_path, "r") as f:
                self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                _ = self.mem0_client.add(
                    message,
                    user_id=user_id,
                    metadata=metadata
                )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # delete all memories for the two users
        try:
            self.mem0_client.delete_all(user_id=speaker_a_user_id)
            self.mem0_client.delete_all(user_id=speaker_b_user_id)
        except Exception as e:
            print(f"Warning: Could not delete existing memories: {e}")

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # add memories for the two users on different threads
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B"),
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

            for future in futures:
                future.result()


def add_memories_memzero_local(conversations, output_folder, is_graph=False):
    """
    Add memories using local mem0ai with Qdrant support.

    Args:
        conversations: Path to conversation data or list of conversations
        output_folder: Output folder for results (not used in add operation)
        is_graph: Whether to enable graph-based memory storage
    """
    print("Starting local mem0ai memory addition...")

    # Handle both file paths and direct data
    if isinstance(conversations, str):
        data_path = conversations
    else:
        # If conversations is a list, we need to save it temporarily
        data_path = "/tmp/temp_conversations.json"
        with open(data_path, "w") as f:
            json.dump(conversations, f)

    memory_manager = MemoryADDLocal(data_path=data_path, is_graph=is_graph)
    memory_manager.process_all_conversations()

    print("✓ Local mem0ai memory addition completed")