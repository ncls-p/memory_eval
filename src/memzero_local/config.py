import os
from typing import Dict, Any

def get_local_mem0_config() -> Dict[str, Any]:
    """
    Get mem0ai configuration for local setup with Qdrant and Ollama.

    Returns:
        Dict[str, Any]: Configuration dictionary for mem0ai

    Raises:
        ValueError: If required services are not configured
    """

    # Default configuration values
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    ollama_port = int(os.getenv("OLLAMA_PORT", ""))
    llm_model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
    embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": qdrant_host,
                "port": qdrant_port,
                "collection_name": "memzero_local",
                "embedding_model_dims": 768,  # Default for nomic-embed-text
            }
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "temperature": 0.1,
                "max_tokens": 1000,
                "ollama_base_url": f"http://{ollama_host}:{ollama_port}",
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": embedding_model,
                "ollama_base_url": f"http://{ollama_host}:{ollama_port}",
            }
        }
    }

    return config


def check_local_services() -> Dict[str, bool]:
    """
    Check if required local services are available.

    Returns:
        Dict[str, bool]: Status of each service
    """
    import requests

    services = {
        "qdrant": False,
        "ollama": False
    }

    # Check Qdrant
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        response = requests.get(f"http://{qdrant_host}:{qdrant_port}/collections", timeout=5)
        services["qdrant"] = response.status_code == 200
    except Exception:
        services["qdrant"] = False

    # Check Ollama
    try:
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        ollama_port = int(os.getenv("OLLAMA_PORT", "11434"))
        response = requests.get(f"http://{ollama_host}:{ollama_port}/api/tags", timeout=5)
        services["ollama"] = response.status_code == 200
    except Exception:
        services["ollama"] = False

    return services


def validate_local_setup():
    """
    Validate that all required services are running.

    Raises:
        RuntimeError: If required services are not available
    """
    services = check_local_services()

    missing_services = [service for service, status in services.items() if not status]

    if missing_services:
        raise RuntimeError(
            f"Required local services are not available: {', '.join(missing_services)}. "
            f"Please ensure Qdrant and Ollama are running locally."
        )

    print("✓ All local services are available")


def get_custom_instructions() -> str:
    """
    Get custom instructions for local mem0ai setup.

    Returns:
        str: Custom instructions for memory generation
    """
    return """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""