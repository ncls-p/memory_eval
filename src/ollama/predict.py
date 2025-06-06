import argparse
import json
import os
import time
from collections import defaultdict

from dotenv import load_dotenv
from jinja2 import Template
from tqdm import tqdm

from .client import OllamaClient

load_dotenv()


ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

    {{memories}}

    Question: {{question}}
    Answer:
    """


def predict_with_ollama(conversation, question, memories, model=None, base_url=None):
    """
    Predict answer using Ollama model.

    Args:
        conversation: The conversation context (not used directly but kept for compatibility)
        question: The question to answer
        memories: The memories to use as context
        model: Ollama model name (optional, uses env var if not provided)
        base_url: Ollama server URL (optional, uses env var if not provided)

    Returns:
        tuple: (response, search_memory_time, response_time, context)
    """
    # Initialize Ollama client
    client = OllamaClient(base_url=base_url, model=model)

    # Template rendering
    template = Template(ANSWER_PROMPT)
    answer_prompt = template.render(memories=memories, question=question)

    # Generate response
    t1 = time.time()
    response = client.generate_response(
        messages=[{"role": "system", "content": answer_prompt}],
        temperature=0.0
    )
    t2 = time.time()
    response_time = t2 - t1

    return response, 0, response_time, memories


class OllamaPredict:
    def __init__(self, model=None, base_url=None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_client = OllamaClient(base_url=self.base_url, model=self.model)
        self.results = defaultdict(list)

    def search_memory(self, idx):
        with open(f"memories/{idx}.txt", "r") as file:
            memories = file.read()

        return memories, 0

    def process_question(self, val, idx):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        response, search_memory_time, response_time, context = self.answer_question(idx, question)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "search_memory_time": search_memory_time,
            "response_time": response_time,
            "context": context,
        }

        return result

    def answer_question(self, idx, question):
        memories, search_memory_time = self.search_memory(idx)

        template = Template(ANSWER_PROMPT)
        answer_prompt = template.render(memories=memories, question=question)

        t1 = time.time()
        response = self.ollama_client.generate_response(
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0
        )
        t2 = time.time()
        response_time = t2 - t1
        return response, search_memory_time, response_time, memories

    def process_data_file(self, file_path, output_file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                result = self.process_question(question_item, idx)
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(output_file_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(output_file_path, "w") as f:
            json.dump(self.results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", type=str, required=True)
    args = parser.parse_args()
    ollama_predict = OllamaPredict()
    ollama_predict.process_data_file("../../dataset/locomo10.json", args.output_file_path)