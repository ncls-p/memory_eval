# Conversational Memory Benchmark

This project provides a comprehensive benchmarking suite for evaluating conversational memory systems. It is designed to test and compare different memory architectures using various metrics, including ROUGE, BLEU, F1 scores, and an LLM-based judge for qualitative analysis.

The benchmark currently supports `Qdrant` as a vector store backend and includes implementations for two memory systems: `Mem0System` and `LangChainLangMemSystem`.

## Features

- **Benchmarking:** Evaluate conversational memory systems on various metrics.
- **LLM-based Evaluation:** Use a language model to judge the quality of responses based on relevance, coherence, and correctness.
- **Multiple Metrics:** Supports standard NLP metrics like ROUGE, BLEU, and F1 scores.
- **Extensible:** Easily extendable to support other memory systems and vector stores.
- **Dockerized:** Comes with a `docker-compose.yml` for easy setup of dependent services like Qdrant.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8+
- Docker
- Poetry (or pip)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/your_repository.git
    cd your_repository
    ```

2.  **Start the services:**

    This will start the Qdrant service.

    ```bash
    docker-compose up -d
    ```

3.  **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

To run the benchmark, execute the following command:

```bash
python src/benchmark_test.py
```

This will:
1.  Populate the memory systems with data from the `dataset/locomo10_rag.json` file.
2.  Run the evaluation on both `Mem0System` and `LangChainLangMemSystem`.
3.  Print the average scores for each system.
4.  Clean up the Qdrant collections.

## Benchmarking

The benchmarking process is defined in `src/benchmark_test.py`. It evaluates the memory systems based on a question-answering task. The evaluation metrics are:

-   **ROUGE Score:** Measures the overlap of n-grams between the generated and reference answers.
-   **BLEU Score:** Measures the precision of n-grams in the generated answer compared to the reference answer.
-   **F1 Score:** The harmonic mean of precision and recall.
-   **LLM-based Judge:** A language model evaluates the generated answers on:
    -   **Relevance:** How relevant the answer is to the question.
    -   **Coherence:** How coherent and easy to understand the answer is.
    -   **Correctness:** How factually correct the answer is.

