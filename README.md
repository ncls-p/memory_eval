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

## Configuration

Before running the benchmark, you need to set up environment variables. Create a `.env` file in the project root:

```bash
# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3.2:latest
OLLAMA_EMBEDDING_MODEL_NAME=nomic-embed-text

# Qdrant configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# OpenAI API configuration (for LLM judge evaluation)
OPENAI_API_KEY=your_api_key_here
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4
```

## Usage

The benchmark script supports flexible execution with command-line options:

### Basic Usage

Run the full benchmark on all conversations:

```bash
python src/benchmark_test.py
```

This will:
1. Populate the memory systems with data from the `dataset/locomo10_rag.json` file
2. Run the evaluation on both `Mem0System` and `LangChainLangMemSystem`
3. Save results to `benchmark_results.json`
4. Clean up the Qdrant collections

### Running Specific Conversations

To benchmark specific conversation IDs:

```bash
python src/benchmark_test.py --conversations 1 2 3
```

### Step-by-Step Execution

The benchmark process can be broken down into three independent steps:

1. **Populate**: Load conversations into memory systems
   ```bash
   python src/benchmark_test.py --steps populate
   ```

2. **Benchmark**: Run evaluation on populated memory systems
   ```bash
   python src/benchmark_test.py --steps benchmark
   ```

3. **Cleanup**: Remove Qdrant collections
   ```bash
   python src/benchmark_test.py --steps cleanup
   ```

You can combine steps as needed:
```bash
# Run populate and benchmark only (skip cleanup)
python src/benchmark_test.py --steps populate benchmark

# Run only the benchmark step (requires previous populate)
python src/benchmark_test.py --steps benchmark
```

### Advanced Examples

```bash
# Benchmark conversations 1, 2, and 3 with all steps
python src/benchmark_test.py --conversations 1 2 3

# Populate specific conversations without running benchmark
python src/benchmark_test.py --conversations 5 6 --steps populate

# Run benchmark and cleanup on previously populated data
python src/benchmark_test.py --steps benchmark cleanup
```

### Output Files

- **`benchmark_results.json`**: Contains detailed results for each QA pair evaluation
- **`benchmark_systems_state.json`**: Stores state between steps (automatically managed)

## Simple Test

For a quick comparison of memory systems without the full benchmark suite:

```bash
python src/test.py
```

This runs a simple test that:
- Creates memory instances for both systems
- Adds sample conversation data
- Performs basic searches
- Compares retrieval results

## Benchmarking

The benchmarking process is defined in `src/benchmark_test.py`. It evaluates the memory systems based on a question-answering task. The evaluation metrics are:

-   **ROUGE Score:** Measures the overlap of n-grams between the generated and reference answers.
-   **BLEU Score:** Measures the precision of n-grams in the generated answer compared to the reference answer.
-   **F1 Score:** The harmonic mean of precision and recall.
-   **LLM-based Judge:** A language model evaluates the generated answers on:
    -   **Relevance:** How relevant the answer is to the question.
    -   **Coherence:** How coherent and easy to understand the answer is.
    -   **Correctness:** How factually correct the answer is.

