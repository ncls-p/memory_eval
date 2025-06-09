import json
import os
import re

import numpy as np
import openai
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from src.test import ConfigManager, LangChainLangMemSystem, Mem0System, QdrantConnector

load_dotenv()
config_manager = ConfigManager()

DATASET_PATH = "dataset/locomo10_rag.json"


def calculate_rouge_scores(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def calculate_bleu_score(reference, generated):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        reference_tokens, generated_tokens, smoothing_function=smoothie
    )


def calculate_f1_score_binary(reference, generated):
    ref_tokens = set(reference.lower().split())
    gen_tokens = set(generated.lower().split())

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


def parse_judge_output(text):
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
        elif not all(
            s > 0 or k == "justification" for k, s in scores.items()
        ):
            scores["justification"] = (
                "Error: Could not parse all scores from LLM judge output. Raw output: "
                + text
            )
        elif (
            scores["justification"] == "Error: Could not parse LLM judge output."
        ):
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


def evaluate_with_llm_judge(context_text, question, reference_answer, generated_answer):
    client = openai.OpenAI(
        base_url=os.getenv("OPENAI_API_URL"), api_key=os.getenv("OPENAI_API_KEY")
    )

    judge_system_prompt = (
        "You are an impartial judge evaluating the quality of an AI-generated answer "
        "based on a question and a reference answer. Score the generated answer on a "
        "scale of 1 to 5 for relevance, coherence, and correctness, then provide an "
        "overall score from 1 to 5 and a brief justification. "
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


def populate_memories(config_manager, qdrant_connector):
    """
    Populates the memory systems with conversation data from the dataset.
    One memory collection per speaker.
    Also extracts QA pairs for evaluation.
    """
    print("Populating memories and extracting QA pairs...")
    try:
        with open(DATASET_PATH, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return None, None, None, None
    except json.JSONDecodeError:
        print(f"Error: Could not parse dataset file {DATASET_PATH}")
        return None, None, None, None

    data = dataset.get("0", {})
    if not data:
        print(f"Error: Dataset at {DATASET_PATH} does not contain the expected '0' key.")
        return None, None, None, None

    conversation = data.get("conversation", [])
    qa_pairs = data.get("question", [])

    speakers = set()
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
        print(f"Initialized LangMemSystem for {speaker} with collection {collection_name}")

    if conversation:
        for turn in conversation:
            speaker = turn.get("speaker")
            text = turn.get("text")
            if speaker and text:
                if speaker in mem0_systems:
                    mem0_systems[speaker].add(messages=[{"role": "user", "content": text}], user_id=speaker)
                if speaker in langmem_systems:
                    langmem_systems[speaker].add(messages=[{"role": "user", "content": text}])

    print("Finished populating memories.")
    return mem0_systems, langmem_systems, list(speakers), qa_pairs


def run_evaluation(mem0_systems, langmem_systems, speakers, qa_pairs):
    """
    Runs the evaluation using QA pairs, querying the appropriate speaker's memory.
    """
    print("Starting evaluation...")
    if not qa_pairs:
        print("No QA pairs to evaluate.")
        return

    results_mem0 = {
        "rouge1_f1": [], "rouge2_f1": [], "rougeL_f1": [], "bleu": [], "f1": [],
        "llm_judge_relevance": [], "llm_judge_coherence": [], "llm_judge_correctness": [], "llm_judge_overall": [],
    }
    results_langmem = {
        "rouge1_f1": [], "rouge2_f1": [], "rougeL_f1": [], "bleu": [], "f1": [],
        "llm_judge_relevance": [], "llm_judge_coherence": [], "llm_judge_correctness": [], "llm_judge_overall": [],
    }

    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair.get("question")
        reference_answer = str(qa_pair.get("answer"))
        if not question or not reference_answer:
            print(f"Skipping invalid QA pair at index {i}: {qa_pair}")
            continue

        print(f"\nProcessing QA pair {i + 1}/{len(qa_pairs)}: {question}")

        mentioned_speakers = [s for s in speakers if re.search(r'\b' + re.escape(s) + r'\b', question, re.IGNORECASE)]

        if not mentioned_speakers:
            print("Skipping question as no speaker was identified.")
            continue

        speaker_to_query = mentioned_speakers[0]
        print(f"  Querying memory of speaker: {speaker_to_query}")

        context_for_judge = "Conversational history stored in memory."

        try:
            mem0_sys = mem0_systems.get(speaker_to_query)
            if mem0_sys:
                memories_mem0 = mem0_sys.search(query=question, user_id=speaker_to_query)
                generated_answer_mem0 = mem0_sys.get_llm_response(query=question, memories=memories_mem0)

                rouge_scores = calculate_rouge_scores(reference_answer, generated_answer_mem0)
                results_mem0["rouge1_f1"].append(rouge_scores["rouge1_f1"])
                results_mem0["rouge2_f1"].append(rouge_scores["rouge2_f1"])
                results_mem0["rougeL_f1"].append(rouge_scores["rougeL_f1"])
                results_mem0["bleu"].append(calculate_bleu_score(reference_answer, generated_answer_mem0))
                results_mem0["f1"].append(calculate_f1_score_binary(reference_answer, generated_answer_mem0))

                llm_eval = evaluate_with_llm_judge(context_for_judge, question, reference_answer, generated_answer_mem0)
                results_mem0["llm_judge_relevance"].append(llm_eval.get("relevance_score", 0))
                results_mem0["llm_judge_coherence"].append(llm_eval.get("coherence_score", 0))
                results_mem0["llm_judge_correctness"].append(llm_eval.get("correctness_score", 0))
                results_mem0["llm_judge_overall"].append(llm_eval.get("overall_score", 0))
                print(f"    Mem0 Generated: {generated_answer_mem0[:100]}...")
            else:
                print(f"    Mem0System not found for speaker: {speaker_to_query}")

        except Exception as e:
            print(f"    Error with Mem0System for question {i + 1}: {e}")

        try:
            langmem_sys = langmem_systems.get(speaker_to_query)
            if langmem_sys:
                memories_langmem = langmem_sys.search(query=question)
                generated_answer_langmem = langmem_sys.get_llm_response(query=question, memories=memories_langmem)

                rouge_scores = calculate_rouge_scores(reference_answer, generated_answer_langmem)
                results_langmem["rouge1_f1"].append(rouge_scores["rouge1_f1"])
                results_langmem["rouge2_f1"].append(rouge_scores["rouge2_f1"])
                results_langmem["rougeL_f1"].append(rouge_scores["rougeL_f1"])
                results_langmem["bleu"].append(calculate_bleu_score(reference_answer, generated_answer_langmem))
                results_langmem["f1"].append(calculate_f1_score_binary(reference_answer, generated_answer_langmem))

                llm_eval = evaluate_with_llm_judge(context_for_judge, question, reference_answer, generated_answer_langmem)
                results_langmem["llm_judge_relevance"].append(llm_eval.get("relevance_score", 0))
                results_langmem["llm_judge_coherence"].append(llm_eval.get("coherence_score", 0))
                results_langmem["llm_judge_correctness"].append(llm_eval.get("correctness_score", 0))
                results_langmem["llm_judge_overall"].append(llm_eval.get("overall_score", 0))
                print(f"    LangMem Generated: {generated_answer_langmem[:100]}...")
            else:
                print(f"    LangMemSystem not found for speaker: {speaker_to_query}")

        except Exception as e:
            print(f"    Error with LangMemSystem for question {i + 1}: {e}")

    print("\n--- Benchmark Results ---")

    def print_average_scores(system_name, results_dict):
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


def cleanup(qdrant_connector, langmem_systems, mem0_systems):
    """
    Cleans up resources, like deleting Qdrant collections.
    """
    print("\nCleaning up resources...")
    qdrant_client = qdrant_connector.get_client()
    if not qdrant_client:
        print("Warning: Qdrant client not available for cleanup.")
        return

    for speaker, system in langmem_systems.items():
        collection_name = system.vector_store.collection_name
        try:
            print(f"  Deleting Qdrant collection: {collection_name}")
            qdrant_client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"  Warning - could not delete Qdrant collection {collection_name}: {e}")

    for speaker, system in mem0_systems.items():
        collection_name = system.collection_name
        try:
            print(f"  Deleting Qdrant collection: {collection_name}")
            qdrant_client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"  Warning - could not delete Qdrant collection {collection_name}: {e}")

    print("Cleanup finished.")


def run_benchmark():
    if (
        not config_manager.ollama_base_url
        or not config_manager.llm_model_name
        or not config_manager.qdrant_host
    ):
        print(
            "Error: Critical configurations (Ollama URL, LLM Model, Qdrant Host) seem missing. Check .env or ConfigManager defaults."
        )
        return

    qdrant_connector = QdrantConnector(
        host=config_manager.qdrant_host, port=config_manager.qdrant_port
    )
    if not qdrant_connector.check_connection():
        print(
            f"Error: Could not connect to Qdrant at {config_manager.qdrant_host}:{config_manager.qdrant_port}. Aborting benchmark."
        )
        return

    original_langchain_collection_name = config_manager.qdrant_langchain_collection

    mem0_systems, langmem_systems, speakers, qa_pairs = populate_memories(
        config_manager, qdrant_connector
    )
    if not mem0_systems:
        print("Failed to populate memories. Aborting benchmark.")
        return

    run_evaluation(mem0_systems, langmem_systems, speakers, qa_pairs)

    cleanup(qdrant_connector, langmem_systems, mem0_systems)

    config_manager.qdrant_langchain_collection = original_langchain_collection_name

    print("\nBenchmark finished.")
    print(
        "Note: For NLTK BLEU score, ensure 'punkt' tokenizer models are downloaded (`nltk.download('punkt')`)."
    )
    print(
        "Required libraries: `nltk`, `rouge-score`, `scikit-learn`, `python-dotenv`, `numpy`, `qdrant-client`, `ollama`, `langchain-ollama` (and their dependencies)."
    )
    print(
        f"Ensure your .env file is configured or ConfigManager defaults are appropriate (e.g., OLLAMA_BASE_URL='{config_manager.ollama_base_url}', LLM_MODEL_NAME='{config_manager.llm_model_name}')."
    )


if __name__ == "__main__":
    run_benchmark()
