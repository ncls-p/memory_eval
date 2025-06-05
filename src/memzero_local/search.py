import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

from mem0 import Memory
from .config import get_local_mem0_config, validate_local_setup
from src.ollama.client import OllamaClient

load_dotenv()


class MemorySearchLocal:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False):
        # Validate local setup before initializing
        validate_local_setup()

        # Get local configuration
        config = get_local_mem0_config()

        # Initialize mem0ai with local configuration
        self.mem0_client = Memory.from_config(config)

        self.top_k = top_k

        # Initialize Ollama client for LLM responses
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        ollama_port = int(os.getenv("OLLAMA_PORT", "11434"))
        self.ollama_client = OllamaClient(
            base_url=f"http://{ollama_host}:{ollama_port}",
            model=os.getenv("LOCAL_LLM_MODEL", "llama3.2")
        )

        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        memories = []

        while retries < max_retries:
            try:
                memories = self.mem0_client.search(
                    query,
                    user_id=user_id,
                    limit=self.top_k
                )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()

        # Handle memory results format - mem0ai returns different structure
        semantic_memories = []
        graph_memories = None

        if isinstance(memories, list):
            for memory in memories:
                if isinstance(memory, dict):
                    semantic_memories.append({
                        "memory": memory.get("memory", ""),
                        "timestamp": memory.get("metadata", {}).get("timestamp", "") if isinstance(memory.get("metadata"), dict) else "",
                        "score": round(memory.get("score", 0.0), 2),
                    })
                elif isinstance(memory, str):
                    semantic_memories.append({
                        "memory": memory,
                        "timestamp": "",
                        "score": 0.0,
                    })

        # For graph functionality (if supported in future)
        if self.is_graph and isinstance(memories, dict) and "relations" in memories:
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories.get("relations", [])
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )

        t1 = time.time()
        response = self.ollama_client.generate_response([{"role": "system", "content": answer_prompt}])
        t2 = time.time()
        response_time = t2 - t1

        return (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results


def search_memories_memzero_local(query, output_folder, top_k=10, is_graph=False):
    """
    Search memories using local mem0ai with Qdrant support.

    Args:
        query: Search query or path to data file for processing
        output_folder: Output folder for results
        top_k: Number of top memories to retrieve
        is_graph: Whether to enable graph-based search

    Returns:
        Path to results file
    """
    print("Starting local mem0ai memory search...")

    # Generate output file path
    output_file_path = os.path.join(
        output_folder,
        f"memzero_local_results_top_{top_k}_graph_{is_graph}.json"
    )

    # Create memory searcher
    memory_searcher = MemorySearchLocal(
        output_path=output_file_path,
        top_k=top_k,
        filter_memories=False,
        is_graph=is_graph
    )

    # Handle different input types
    if isinstance(query, str) and query.endswith('.json'):
        # Process data file
        memory_searcher.process_data_file(query)
    else:
        # Handle direct query (would need to be implemented based on specific needs)
        raise NotImplementedError("Direct query search not implemented yet")

    print(f"✓ Local mem0ai memory search completed. Results saved to {output_file_path}")
    return output_file_path