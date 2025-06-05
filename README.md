# Mem0: Building Production‑Ready AI Agents with Scalable Long‑Term Memory

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2504.19413)
[![Website](https://img.shields.io/badge/Website-Project-blue)](https://mem0.ai/research)

This repository contains the code and dataset for our paper: **Mem0: Building Production‑Ready AI Agents with Scalable Long‑Term Memory**.

## 📋 Overview

This project evaluates Mem0 and compares it with different memory and retrieval techniques for AI systems:

1. **Established LOCOMO Benchmarks**: We evaluate against five established approaches from the literature: LoCoMo, ReadAgent, MemoryBank, MemGPT, and A-Mem.
2. **Open-Source Memory Solutions**: We test promising open-source memory architectures including LangMem, which provides flexible memory management capabilities.
3. **RAG Systems**: We implement Retrieval-Augmented Generation with various configurations, testing different chunk sizes and retrieval counts to optimize performance.
4. **Full-Context Processing**: We examine the effectiveness of passing the entire conversation history within the context window of the LLM as a baseline approach.
5. **Proprietary Memory Systems**: We evaluate OpenAI's built-in memory feature available in their ChatGPT interface to compare against commercial solutions.
6. **Third-Party Memory Providers**: We incorporate Zep, a specialized memory management platform designed for AI agents, to assess the performance of dedicated memory infrastructure.

We test these techniques on the LOCOMO dataset, which contains conversational data with various question types to evaluate memory recall and understanding.

## 🏠 Local AI Setup

This project now supports **fully local AI processing** without requiring cloud API keys. The local setup includes:

### Local Techniques Available

1. **Ollama Integration** (`ollama`): Local LLM processing using Ollama
2. **Memzero Local** (`memzero_local`): Local version of Mem0 using Ollama + Qdrant
3. **Qdrant RAG** (`qdrant_rag`): Vector-based RAG using local embeddings and Qdrant

### Quick Local Setup

```bash
# Install local AI stack
make setup-local

# Start local services
make start-local

# Test local setup
make test-local
```

### Advanced Setup Options

```bash
# Complete local stack with all services
make start-full-stack

# Full stack with monitoring (Grafana, Prometheus, Jaeger)
make start-with-monitoring

# Clean everything
make clean-all
```

### Comparison: Local vs Cloud

| Feature | Cloud (OpenAI/Mem0) | Local (Ollama/Qdrant) |
|---------|-------------------|---------------------|
| **Cost** | Pay-per-token | Free after setup |
| **Privacy** | Data sent to cloud | Fully local |
| **Speed** | Fast (optimized) | Depends on hardware |
| **Setup** | API keys only | Requires installation |
| **Models** | Latest GPT models | Open-source models |
| **Internet** | Required | Not required |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB+ |
| **CPU** | 4 cores | 8+ cores |
| **Storage** | 10GB free | 50GB+ free |
| **GPU** | None (CPU only) | NVIDIA GPU for speed |

For detailed setup instructions, see [`LOCAL_AI_GUIDE.md`](LOCAL_AI_GUIDE.md).

### Local vs Cloud Examples

```bash
# Compare local vs cloud performance
make run-local-benchmark      # Local techniques
make run-mem0-search         # Cloud Mem0
make run-openai              # Cloud OpenAI

# View comparison results
python generate_scores.py --compare_local_cloud
```

## 🔍 Dataset

The LOCOMO dataset used in our experiments can be downloaded from our Google Drive repository:

[Download LOCOMO Dataset](https://drive.google.com/drive/folders/1L-cTjTm0ohMsitsHg4dijSPJtqNflwX-?usp=drive_link)

The dataset contains conversational data specifically designed to test memory recall and understanding across various question types and complexity levels.

Place the dataset files in the `dataset/` directory:
- `locomo10.json`: Original dataset
- `locomo10_rag.json`: Dataset formatted for RAG experiments

## 📁 Project Structure

```
.
├── src/                     # Source code for different memory techniques
│   ├── mem0/                # Implementation of the Mem0 technique
│   ├── memzero_local/       # Local Mem0 implementation (Ollama + Qdrant)
│   ├── ollama/              # Ollama integration for local LLMs
│   ├── qdrant_rag/          # Qdrant-based RAG implementation
│   ├── openai/              # Implementation of the OpenAI memory
│   ├── zep/                 # Implementation of the Zep memory
│   ├── rag.py               # Implementation of the RAG technique
│   └── langmem.py           # Implementation of the Language-based memory
├── metrics/                 # Code for evaluation metrics
├── results/                 # Results of experiments
├── dataset/                 # Dataset files
├── evals.py                 # Evaluation script
├── run_experiments.py       # Script to run experiments
├── run_local_benchmark.py   # Local AI benchmarking script
├── generate_scores.py       # Script to generate scores from results
├── setup_local.py           # Local AI stack setup script
├── docker-compose.yml       # Qdrant service configuration
├── docker-compose.full.yml  # Complete local stack
├── LOCAL_AI_GUIDE.md        # Comprehensive local setup guide
└── prompts.py               # Prompts used for the models
```

## 🚀 Getting Started

### Prerequisites

Create a `.env` file with your API keys and configurations. The following keys are required:

```
# OpenAI API key for GPT models and embeddings
OPENAI_API_KEY="your-openai-api-key"

# Mem0 API keys (for Mem0 and Mem0+ techniques)
MEM0_API_KEY="your-mem0-api-key"
MEM0_PROJECT_ID="your-mem0-project-id"
MEM0_ORGANIZATION_ID="your-mem0-organization-id"

# Model configuration
MODEL="gpt-4o-mini"  # or your preferred model
EMBEDDING_MODEL="text-embedding-3-small"  # or your preferred embedding model
ZEP_API_KEY="api-key-from-zep"
```

### Running Experiments

You can run experiments using the provided Makefile commands:

#### Memory Techniques

```bash
# Run Mem0 experiments
make run-mem0-add         # Add memories using Mem0
make run-mem0-search      # Search memories using Mem0

# Run Mem0+ experiments (with graph-based search)
make run-mem0-plus-add    # Add memories using Mem0+
make run-mem0-plus-search # Search memories using Mem0+

# Run RAG experiments
make run-rag              # Run RAG with chunk size 500
make run-full-context     # Run RAG with full context

# Run LangMem experiments
make run-langmem          # Run LangMem

# Run Zep experiments
make run-zep-add          # Add memories using Zep
make run-zep-search       # Search memories using Zep

# Run OpenAI experiments
make run-openai           # Run OpenAI experiments

# Run Local AI experiments
make run-ollama           # Run Ollama experiments
make run-memzero-local    # Run local Mem0 experiments
make run-qdrant-rag       # Run Qdrant RAG experiments

# Local benchmarks
make run-local-benchmark  # Run comprehensive local benchmarks
```

Alternatively, you can run experiments directly with custom parameters:

```bash
python run_experiments.py --technique_type [mem0|rag|langmem] [additional parameters]
```

#### Command-line Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--technique_type` | Memory technique to use (mem0, rag, langmem, ollama, memzero_local, qdrant_rag) | mem0 |
| `--method` | Method to use (add, search) | add |
| `--chunk_size` | Chunk size for processing | 1000 |
| `--top_k` | Number of top memories to retrieve | 30 |
| `--filter_memories` | Whether to filter memories | False |
| `--is_graph` | Whether to use graph-based search | False |
| `--num_chunks` | Number of chunks to process for RAG | 1 |
| `--local_setup` | Use local AI stack instead of cloud APIs | False |

### 📊 Evaluation

To evaluate results, run:

```bash
python evals.py --input_file [path_to_results] --output_file [output_path]
```

This script:
1. Processes each question-answer pair
2. Calculates BLEU and F1 scores automatically
3. Uses an LLM judge to evaluate answer correctness
4. Saves the combined results to the output file

### 📈 Generating Scores

Generate final scores with:

```bash
python generate_scores.py
```

This script:
1. Loads the evaluation metrics data
2. Calculates mean scores for each category (BLEU, F1, LLM)
3. Reports the number of questions per category
4. Calculates overall mean scores across all categories

Example output:
```
Mean Scores Per Category:
         bleu_score  f1_score  llm_score  count
category
1           0.xxxx    0.xxxx     0.xxxx     xx
2           0.xxxx    0.xxxx     0.xxxx     xx
3           0.xxxx    0.xxxx     0.xxxx     xx

Overall Mean Scores:
bleu_score    0.xxxx
f1_score      0.xxxx
llm_score     0.xxxx
```

## 📏 Evaluation Metrics

We use several metrics to evaluate the performance of different memory techniques:

1. **BLEU Score**: Measures the similarity between the model's response and the ground truth
2. **F1 Score**: Measures the harmonic mean of precision and recall
3. **LLM Score**: A binary score (0 or 1) determined by an LLM judge evaluating the correctness of responses
4. **Token Consumption**: Number of tokens required to generate final answer.
5. **Latency**: Time required during search and to generate response.

## 📚 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413},
  year={2025}
}
```

## 📄 License

[MIT License](LICENSE)

## 👥 Contributors

- [Prateek Chhikara](https://github.com/prateekchhikara)
- [Dev Khant](https://github.com/Dev-Khant)
- [Saket Aryan](https://github.com/whysosaket)
- [Taranjeet Singh](https://github.com/taranjeet)
- [Deshraj Yadav](https://github.com/deshraj)

