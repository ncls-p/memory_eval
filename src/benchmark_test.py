"""Memory benchmarking suite with support for step selection and real-time JSON logging.

This module provides comprehensive benchmarking capabilities for conversational memory systems,
including Mem0System and LangChainLangMemSystem, with support for multiple evaluation metrics
and selective step execution.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openai
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from test import (
    ConfigManager,
    LangChainLangMemSystem,
    Mem0System,
    QdrantConnector,
)

load_dotenv()

CONFIG_MANAGER = ConfigManager()

DATASET_PATH = Path("dataset/locomo10_rag.json")
RESULTS_FILE = Path("benchmark_results.json")
SYSTEMS_STATE_FILE = Path("benchmark_systems_state.json")


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> sections from text."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


def calculate_rouge_scores(reference: str, generated: str) -> Dict[str, float]:
    """Calculate ROUGE scores between reference and generated text."""
    generated_clean = remove_think_tags(generated)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated_clean)
    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def calculate_bleu_score(reference: str, generated: str) -> float:
    """Calculate BLEU score between reference and generated text."""
    generated_clean = remove_think_tags(generated)

    reference_tokens = [reference.split()]
    generated_tokens = generated_clean.split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(
        reference_tokens, generated_tokens, smoothing_function=smoothie
    )
    if isinstance(score, (int, float)):
        return float(score)
    raise TypeError(
        f"sentence_bleu from nltk returned an unexpected type: {type(score)}"
    )


def calculate_f1_score_binary(reference: str, generated: str) -> float:
    """Calculate binary F1 score based on token overlap."""
    generated_clean = remove_think_tags(generated)

    ref_tokens = set(reference.lower().split())
    gen_tokens = set(generated_clean.lower().split())

    if not ref_tokens and not gen_tokens:
        return 1.0
    if not ref_tokens or not gen_tokens:
        return 0.0

    true_positives = len(ref_tokens.intersection(gen_tokens))
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(gen_tokens)
    recall = true_positives / len(ref_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def parse_judge_output(text: str) -> Dict[str, Union[int, str]]:
    """Parse LLM judge output to extract scores and justification."""
    scores = {
        "relevance_score": 0,
        "coherence_score": 0,
        "correctness_score": 0,
        "overall_score": 0,
        "justification": "Error: Could not parse LLM judge output.",
    }
    try:
        relevance_match = re.search(r"Relevance(?: Score)?: (\d)", text, re.IGNORECASE)
        if relevance_match:
            scores["relevance_score"] = int(relevance_match.group(1))

        coherence_match = re.search(r"Coherence(?: Score)?: (\d)", text, re.IGNORECASE)
        if coherence_match:
            scores["coherence_score"] = int(coherence_match.group(1))

        correctness_match = re.search(
            r"Correctness(?: Score)?: (\d)", text, re.IGNORECASE
        )
        if correctness_match:
            scores["correctness_score"] = int(correctness_match.group(1))

        overall_match = re.search(r"Overall Score: (\d)", text, re.IGNORECASE)
        if overall_match:
            scores["overall_score"] = int(overall_match.group(1))

        justification_match = re.search(
            r"Justification:\s*(.*)", text, re.IGNORECASE | re.DOTALL
        )
        if justification_match:
            scores["justification"] = justification_match.group(1).strip()
        elif not all(s > 0 or k == "justification" for k, s in scores.items()):
            scores["justification"] = (
                "Error: Could not parse all scores from LLM judge output. Raw output: "
                + text
            )
        elif scores["justification"] == "Error: Could not parse LLM judge output.":
            scores["justification"] = (
                "Justification not explicitly parsed. Raw output: " + text
            )

    except Exception as e:
        scores["justification"] = (
            f"Error parsing LLM judge output: {e}. Raw output: {text}"
        )

    for key in [
        "relevance_score",
        "coherence_score",
        "correctness_score",
        "overall_score",
        "justification",
    ]:
        if key not in scores:
            if key == "justification":
                scores[key] = "Parsing error, justification missing."
            else:
                scores[key] = 0
    return scores


def evaluate_with_llm_judge(
    context_text: str,
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> Dict[str, Union[int, str]]:
    """Evaluate generated answer using LLM judge."""
    client = openai.OpenAI(
        base_url=os.getenv("OPENAI_API_URL"), api_key=os.getenv("OPENAI_API_KEY")
    )

    judge_system_prompt = (
        "You are an impartial judge evaluating the quality of an AI-generated answer "
        "based on a question and a reference answer. Score the generated answer on a "
        "scale of 1 to 10 for relevance, coherence, and correctness, then provide an "
        "overall score from 1 to 10 and a brief justification. "
        "Ensure your output includes lines like 'Relevance: X', 'Coherence: Y', "
        "'Correctness: Z', 'Overall Score: W', and 'Justification: ...'."
    )

    judge_user_prompt_content = f"""Context:
        {context_text}

        Question:
        {question}

        Reference Answer:
        {reference_answer}

        Generated Answer:
        {generated_answer}

        Please provide your evaluation based on the criteria mentioned (Relevance, Coherence, Correctness, Overall Score, and Justification)."""

    default_evaluation = {
        "relevance_score": 0,
        "coherence_score": 0,
        "correctness_score": 0,
        "overall_score": 0,
        "justification": "Error: LLM judge evaluation failed.",
    }

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL_NAME")
        api_url = os.getenv("OPENAI_API_URL")

        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            default_evaluation["justification"] = (
                "Error: OPENAI_API_KEY environment variable not set."
            )
            return default_evaluation
        if not model_name:
            print("Error: OPENAI_MODEL_NAME environment variable not set.")
            default_evaluation["justification"] = (
                "Error: OPENAI_MODEL_NAME environment variable not set."
            )
            return default_evaluation

        client = openai.OpenAI(
            base_url=api_url,
            api_key=api_key,
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": judge_user_prompt_content},
            ],
        )
        evaluation_text = response.choices[0].message.content
        if not evaluation_text:
            default_evaluation[
                "justification"
            ] = "Error: LLM judge returned empty response."
            return default_evaluation

        parsed_scores = parse_judge_output(evaluation_text)
        return parsed_scores

    except openai.APIError as e:
        print(f"OpenAI API Error during LLM judge evaluation: {e}")
        default_evaluation["justification"] = f"OpenAI API Error: {e}"
        return default_evaluation
    except Exception as e:
        print(f"Error during OpenAI LLM judge evaluation: {e}")
        default_evaluation["justification"] = f"Error: {e}"
        return default_evaluation


def write_result_to_file(
    result_data: Dict[str, Any],
    filename: Path = RESULTS_FILE,
) -> None:
    """Append a single result to the JSON file immediately."""
    try:
        if filename.exists():
            with filename.open("r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(result_data)

        with filename.open("w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Error writing result to file: {e}")


def initialize_results_file(filename: Path = RESULTS_FILE) -> None:
    """Initialize the results file with metadata and empty results array."""
    try:
        metadata = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "dataset_path": str(DATASET_PATH),
            },
            "results": [],
        }

        with filename.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Error initializing results file: {e}")


def save_systems_state(
    mem0_systems: Dict[str, Mem0System],
    langmem_systems: Dict[str, LangChainLangMemSystem],
    speakers: List[str],
    qa_pairs: List[Dict[str, Any]],
    filename: Path = SYSTEMS_STATE_FILE,
) -> None:
    """Save the systems state to allow running steps independently."""
    try:
        state_data = {
            "speakers": speakers,
            "qa_pairs": qa_pairs,
            "mem0_collections": {
                speaker: system.collection_name
                for speaker, system in mem0_systems.items()
            },
            "langmem_collections": {
                speaker: system.vector_store.collection_name
                for speaker, system in langmem_systems.items()
                if system.vector_store
            },
            "timestamp": datetime.now().isoformat(),
        }

        with filename.open("w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)

        print(f"Systems state saved to {filename}")

    except Exception as e:
        print(f"Error saving systems state: {e}")


def load_systems_state(
    config_manager: ConfigManager,
    qdrant_connector: QdrantConnector,
    filename: Path = SYSTEMS_STATE_FILE,
) -> Tuple[
    Optional[Dict[str, Mem0System]],
    Optional[Dict[str, LangChainLangMemSystem]],
    Optional[List[str]],
    Optional[List[Dict[str, Any]]],
]:
    """Load the systems state to continue from a previous step."""
    try:
        if not filename.exists():
            print(f"No systems state file found at {filename}")
            return None, None, None, None

        with filename.open("r", encoding="utf-8") as f:
            state_data = json.load(f)

        speakers = state_data.get("speakers", [])
        qa_pairs = state_data.get("qa_pairs", [])
        mem0_collections = state_data.get("mem0_collections", {})
        langmem_collections = state_data.get("langmem_collections", {})

        mem0_systems = {}
        for speaker in speakers:
            collection_name = mem0_collections.get(speaker)
            if collection_name:
                mem0_systems[speaker] = Mem0System(
                    config=config_manager,
                    qdrant_connector=qdrant_connector,
                    collection_name=collection_name,
                )

        langmem_systems = {}
        for speaker in speakers:
            collection_name = langmem_collections.get(speaker)
            if collection_name:
                config_manager.qdrant_langchain_collection = collection_name
                langmem_systems[speaker] = LangChainLangMemSystem(
                    config=config_manager, qdrant_connector=qdrant_connector
                )

        print(f"Systems state loaded from {filename}")
        return mem0_systems, langmem_systems, speakers, qa_pairs

    except Exception as e:
        print(f"Error loading systems state: {e}")
        return None, None, None, None


def populate_memories(
    config_manager: ConfigManager,
    qdrant_connector: QdrantConnector,
    conversation_ids: Optional[List[int]] = None,
) -> Tuple[
    Optional[Dict[str, Mem0System]],
    Optional[Dict[str, LangChainLangMemSystem]],
    Optional[List[str]],
    Optional[List[Dict[str, Any]]],
]:
    """Populate the memory systems with conversation data from the dataset.

    One memory collection per speaker.
    Also extracts QA pairs for evaluation.
    """
    print("Populating memories and extracting QA pairs...")
    try:
        with DATASET_PATH.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return None, None, None, None
    except json.JSONDecodeError:
        print(f"Error: Could not parse dataset file {DATASET_PATH}")
        return None, None, None, None

    conversations_to_process = []
    if conversation_ids:
        for conv_id in conversation_ids:
            if str(conv_id) in dataset:
                conversations_to_process.append(dataset[str(conv_id)])
            else:
                print(f"Warning: Conversation ID {conv_id} not found in dataset.")
    else:
        conversations_to_process = list(dataset.values())

    if not conversations_to_process:
        print("Error: No conversations to process.")
        return None, None, None, None

    all_conversations = []
    all_qa_pairs = []
    speakers = set()

    for data in conversations_to_process:
        conversation = data.get("conversation", [])
        qa_pairs = data.get("question", [])

        all_conversations.extend(conversation)
        all_qa_pairs.extend(qa_pairs)

        if conversation:
            for turn in conversation:
                if "speaker" in turn:
                    speakers.add(turn["speaker"])

    print(f"Found speakers: {list(speakers)}")

    mem0_systems = {}
    for speaker in speakers:
        collection_name = f"mem0_{speaker.lower().replace(' ', '_')}"
        mem0_systems[speaker] = Mem0System(
            config=config_manager,
            qdrant_connector=qdrant_connector,
            collection_name=collection_name,
        )
        print(f"Initialized Mem0System for {speaker} with collection {collection_name}")

    langmem_systems = {}
    for speaker in speakers:
        collection_name = f"langmem_{speaker.lower()}"
        config_manager.qdrant_langchain_collection = collection_name
        langmem_systems[speaker] = LangChainLangMemSystem(
            config=config_manager, qdrant_connector=qdrant_connector
        )
        print(
            f"Initialized LangMemSystem for {speaker} with collection {collection_name}"
        )

    if all_conversations:
        for turn in all_conversations:
            speaker = turn.get("speaker")
            text = turn.get("text")
            if speaker and text:
                if speaker in mem0_systems:
                    mem0_systems[speaker].add(
                        messages=[{"role": "user", "content": text}], user_id=speaker
                    )
                if speaker in langmem_systems:
                    langmem_systems[speaker].add(
                        messages=[{"role": "user", "content": text}]
                    )

    print("Finished populating memories.")
    return mem0_systems, langmem_systems, list(speakers), all_qa_pairs


def run_evaluation(
    mem0_systems: Dict[str, Mem0System],
    langmem_systems: Dict[str, LangChainLangMemSystem],
    speakers: List[str],
    qa_pairs: List[Dict[str, Any]],
) -> None:
    """Run the evaluation using QA pairs, querying the appropriate speaker's memory."""
    print("Starting evaluation...")

    initialize_results_file()
    print(f"Initialized results file: {RESULTS_FILE}")

    if not qa_pairs:
        print("No QA pairs to evaluate.")
        return

    metric_keys = [
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "bleu",
        "f1",
        "llm_judge_relevance",
        "llm_judge_coherence",
        "llm_judge_correctness",
        "llm_judge_overall",
    ]

    results_mem0: Dict[str, List[float]] = {key: [] for key in metric_keys}
    results_langmem: Dict[str, List[float]] = {key: [] for key in metric_keys}

    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair.get("question")
        reference_answer = str(qa_pair.get("answer"))
        if not question or not reference_answer:
            print(f"Skipping invalid QA pair at index {i}: {qa_pair}")
            continue

        print(f"\nProcessing QA pair {i + 1}/{len(qa_pairs)}: {question}")

        mentioned_speakers = [
            speaker
            for speaker in speakers
            if re.search(
                r"\b" + re.escape(speaker) + r"\b",
                question,
                re.IGNORECASE,
            )
        ]

        if not mentioned_speakers:
            print("Skipping question as no speaker was identified.")
            continue

        speaker_to_query = mentioned_speakers[0]
        print(f"  Querying memory of speaker: {speaker_to_query}")

        context_for_judge = "Conversational history stored in memory."

        try:
            mem0_sys = mem0_systems.get(speaker_to_query)
            if mem0_sys:
                memories_mem0 = mem0_sys.search(
                    query=question, user_id=speaker_to_query
                )
                generated_answer_mem0 = mem0_sys.get_llm_response(
                    query=question, memories=memories_mem0
                )

                rouge_scores = calculate_rouge_scores(
                    reference_answer, generated_answer_mem0
                )
                bleu_score = calculate_bleu_score(
                    reference_answer, generated_answer_mem0
                )
                f1_score = calculate_f1_score_binary(
                    reference_answer, generated_answer_mem0
                )
                llm_eval = evaluate_with_llm_judge(
                    context_for_judge, question, reference_answer, generated_answer_mem0
                )

                results_mem0["rouge1_f1"].append(rouge_scores["rouge1_f1"])
                results_mem0["rouge2_f1"].append(rouge_scores["rouge2_f1"])
                results_mem0["rougeL_f1"].append(rouge_scores["rougeL_f1"])
                results_mem0["bleu"].append(bleu_score)
                results_mem0["f1"].append(f1_score)
                results_mem0["llm_judge_relevance"].append(
                    float(llm_eval.get("relevance_score", 0))
                )
                results_mem0["llm_judge_coherence"].append(
                    float(llm_eval.get("coherence_score", 0))
                )
                results_mem0["llm_judge_correctness"].append(
                    float(llm_eval.get("correctness_score", 0))
                )
                results_mem0["llm_judge_overall"].append(
                    float(llm_eval.get("overall_score", 0))
                )

                result_data = {
                    "qa_pair_index": i + 1,
                    "question": question,
                    "expected_answer": reference_answer,
                    "speaker": speaker_to_query,
                    "system": "Mem0System",
                    "generated_answer": generated_answer_mem0,
                    "metrics": {
                        "rouge1_f1": rouge_scores["rouge1_f1"],
                        "rouge2_f1": rouge_scores["rouge2_f1"],
                        "rougeL_f1": rouge_scores["rougeL_f1"],
                        "bleu": bleu_score,
                        "f1": f1_score,
                        "llm_judge_relevance": llm_eval.get("relevance_score", 0),
                        "llm_judge_coherence": llm_eval.get("coherence_score", 0),
                        "llm_judge_correctness": llm_eval.get("correctness_score", 0),
                        "llm_judge_overall": llm_eval.get("overall_score", 0),
                        "llm_judge_justification": llm_eval.get("justification", ""),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
                write_result_to_file(result_data)

                print(f"    Mem0 Generated: {generated_answer_mem0[:100]}...")
            else:
                print(f"    Mem0System not found for speaker: {speaker_to_query}")

        except Exception as e:
            print(f"    Error with Mem0System for question {i + 1}: {e}")

        try:
            langmem_sys = langmem_systems.get(speaker_to_query)
            if langmem_sys:
                memories_langmem = langmem_sys.search(query=question)
                generated_answer_langmem = langmem_sys.get_llm_response(
                    query=question, memories=memories_langmem
                )

                rouge_scores = calculate_rouge_scores(
                    reference_answer, generated_answer_langmem
                )
                bleu_score = calculate_bleu_score(
                    reference_answer, generated_answer_langmem
                )
                f1_score = calculate_f1_score_binary(
                    reference_answer, generated_answer_langmem
                )
                llm_eval = evaluate_with_llm_judge(
                    context_for_judge,
                    question,
                    reference_answer,
                    generated_answer_langmem,
                )

                results_langmem["rouge1_f1"].append(rouge_scores["rouge1_f1"])
                results_langmem["rouge2_f1"].append(rouge_scores["rouge2_f1"])
                results_langmem["rougeL_f1"].append(rouge_scores["rougeL_f1"])
                results_langmem["bleu"].append(bleu_score)
                results_langmem["f1"].append(f1_score)
                results_langmem["llm_judge_relevance"].append(
                    float(llm_eval.get("relevance_score", 0))
                )
                results_langmem["llm_judge_coherence"].append(
                    float(llm_eval.get("coherence_score", 0))
                )
                results_langmem["llm_judge_correctness"].append(
                    float(llm_eval.get("correctness_score", 0))
                )
                results_langmem["llm_judge_overall"].append(
                    float(llm_eval.get("overall_score", 0))
                )

                result_data = {
                    "qa_pair_index": i + 1,
                    "question": question,
                    "expected_answer": reference_answer,
                    "speaker": speaker_to_query,
                    "system": "LangChainLangMemSystem",
                    "generated_answer": generated_answer_langmem,
                    "metrics": {
                        "rouge1_f1": rouge_scores["rouge1_f1"],
                        "rouge2_f1": rouge_scores["rouge2_f1"],
                        "rougeL_f1": rouge_scores["rougeL_f1"],
                        "bleu": bleu_score,
                        "f1": f1_score,
                        "llm_judge_relevance": llm_eval.get("relevance_score", 0),
                        "llm_judge_coherence": llm_eval.get("coherence_score", 0),
                        "llm_judge_correctness": llm_eval.get("correctness_score", 0),
                        "llm_judge_overall": llm_eval.get("overall_score", 0),
                        "llm_judge_justification": llm_eval.get("justification", ""),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
                write_result_to_file(result_data)

                print(f"    LangMem Generated: {generated_answer_langmem[:100]}...")
            else:
                print(f"    LangMemSystem not found for speaker: {speaker_to_query}")

        except Exception as e:
            print(f"    Error with LangMemSystem for question {i + 1}: {e}")

    print("\n=== Benchmark Results ===")

    def print_average_scores(
        system_name: str, results_dict: Dict[str, List[float]]
    ) -> None:
        """Print average scores for a system."""
        print(f"\nAverage Scores for {system_name}:")
        if not any(results_dict.values()):
            print("  No results collected.")
            return

        for metric, values in results_dict.items():
            if values:
                avg_score = np.mean(values)
                print(f"  Average {metric.replace('_', ' ').title()}: {avg_score:.4f}")
            else:
                print(f"  Average {metric.replace('_', ' ').title()}: N/A (no data)")

    print_average_scores("Mem0System", results_mem0)
    print_average_scores("LangChainLangMemSystem", results_langmem)


def cleanup(
    qdrant_connector: QdrantConnector,
    langmem_systems: Dict[str, LangChainLangMemSystem],
    mem0_systems: Dict[str, Mem0System],
) -> None:
    """Clean up resources, like deleting Qdrant collections."""
    print("\nCleaning up resources...")
    qdrant_client = qdrant_connector.get_client()
    if not qdrant_client:
        print("Warning: Qdrant client not available for cleanup.")
        return

    for speaker, system in langmem_systems.items():
        if system.vector_store:
            collection_name = system.vector_store.collection_name
            try:
                print(f"  Deleting Qdrant collection: {collection_name}")
                qdrant_client.delete_collection(collection_name=collection_name)
            except Exception as e:
                print(
                    f"  Warning - could not delete Qdrant collection {collection_name}: {e}"
                )

    for speaker, system in mem0_systems.items():
        collection_name = system.collection_name
        try:
            print(f"  Deleting Qdrant collection: {collection_name}")
            qdrant_client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(
                f"  Warning - could not delete Qdrant collection {collection_name}: {e}"
            )

    print("Cleanup finished.")


def run_populate_step(
    config_manager: ConfigManager,
    qdrant_connector: QdrantConnector,
    conversation_ids: Optional[List[int]] = None,
) -> bool:
    """Populate Qdrant collections with conversation data."""
    print("=== POPULATE STEP ===")
    mem0_systems, langmem_systems, speakers, qa_pairs = populate_memories(
        config_manager, qdrant_connector, conversation_ids
    )
    if (
        mem0_systems is None
        or langmem_systems is None
        or speakers is None
        or qa_pairs is None
    ):
        print("Failed to populate memories.")
        return False

    save_systems_state(mem0_systems, langmem_systems, speakers, qa_pairs)
    print("Populate step completed successfully.")
    return True


def run_benchmark_step(
    config_manager: ConfigManager,
    qdrant_connector: QdrantConnector,
) -> bool:
    """Run the evaluation benchmark using existing populated collections."""
    print("=== BENCHMARK STEP ===")
    mem0_systems, langmem_systems, speakers, qa_pairs = load_systems_state(
        config_manager, qdrant_connector
    )

    if (
        mem0_systems is None
        or langmem_systems is None
        or speakers is None
        or qa_pairs is None
    ):
        print("No existing systems state found. Please run populate step first.")
        return False

    run_evaluation(mem0_systems, langmem_systems, speakers, qa_pairs)
    print("Benchmark step completed successfully.")
    return True


def run_cleanup_step(
    config_manager: ConfigManager,
    qdrant_connector: QdrantConnector,
) -> bool:
    """Clean up Qdrant collections."""
    print("=== CLEANUP STEP ===")
    mem0_systems, langmem_systems, speakers, qa_pairs = load_systems_state(
        config_manager, qdrant_connector
    )

    if mem0_systems is None or langmem_systems is None:
        print("No existing systems state found. Nothing to clean up.")
        return False

    cleanup(qdrant_connector, langmem_systems, mem0_systems)

    try:
        if SYSTEMS_STATE_FILE.exists():
            SYSTEMS_STATE_FILE.unlink()
            print(f"Removed systems state file: {SYSTEMS_STATE_FILE}")
    except Exception as e:
        print(f"Warning: Could not remove state file: {e}")

    print("Cleanup step completed successfully.")
    return True


def run_benchmark(
    conversation_ids: Optional[List[int]] = None,
    steps: Optional[List[str]] = None,
) -> None:
    """Run the benchmark with optional step selection."""
    if steps is None:
        steps = ["populate", "benchmark", "cleanup"]

    print(f"Running steps: {', '.join(steps)}")

    if (
        not CONFIG_MANAGER.ollama_base_url
        or not CONFIG_MANAGER.llm_model_name
        or not CONFIG_MANAGER.qdrant_host
    ):
        print(
            "Error: Critical configurations (Ollama URL, LLM Model, Qdrant Host) seem missing. Check .env or ConfigManager defaults."
        )
        return

    qdrant_connector = QdrantConnector(
        host=CONFIG_MANAGER.qdrant_host,
        port=CONFIG_MANAGER.qdrant_port,
    )
    if not qdrant_connector.check_connection():
        print(
            f"Error: Could not connect to Qdrant at "
            f"{CONFIG_MANAGER.qdrant_host}:{CONFIG_MANAGER.qdrant_port}. "
            f"Aborting benchmark."
        )
        return

    original_langchain_collection_name = CONFIG_MANAGER.qdrant_langchain_collection

    success = True

    if "populate" in steps:
        success = run_populate_step(CONFIG_MANAGER, qdrant_connector, conversation_ids)
        if not success:
            print("Populate step failed. Aborting.")
            return

    if "benchmark" in steps and success:
        success = run_benchmark_step(CONFIG_MANAGER, qdrant_connector)
        if not success:
            print("Benchmark step failed.")

    if "cleanup" in steps:
        run_cleanup_step(CONFIG_MANAGER, qdrant_connector)

    CONFIG_MANAGER.qdrant_langchain_collection = original_langchain_collection_name

    print("\nProcess finished.")
    print(
        "Note: For NLTK BLEU score, ensure 'punkt' tokenizer models are downloaded (`nltk.download('punkt')`)."
    )
    print(
        "Required libraries: `nltk`, `rouge-score`, `scikit-learn`, `python-dotenv`, `numpy`, `qdrant-client`, `ollama`, `langchain-ollama` (and their dependencies)."
    )
    print(
        f"Ensure your .env file is configured or ConfigManager defaults are "
        f"appropriate (e.g., OLLAMA_BASE_URL='{CONFIG_MANAGER.ollama_base_url}', "
        f"LLM_MODEL_NAME='{CONFIG_MANAGER.llm_model_name}')."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the memory benchmark with optional step selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--conversations",
        nargs="+",
        type=int,
        help="List of conversation IDs to run, or 'all' to run all conversations.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["populate", "benchmark", "cleanup"],
        default=["populate", "benchmark", "cleanup"],
        help="Select which steps to run: populate, benchmark, cleanup (default: all steps)",
    )
    args = parser.parse_args()

    conversation_ids = args.conversations
    if conversation_ids and "all" in [str(c).lower() for c in conversation_ids]:
        conversation_ids = None

    run_benchmark(conversation_ids=conversation_ids, steps=args.steps)
