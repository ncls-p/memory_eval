import os
import time
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class OllamaClient:
    """Ollama client wrapper for chat completion and embeddings."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, embedding_model: Optional[str] = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")

        # Ensure base URL doesn't end with slash
        self.base_url = self.base_url.rstrip('/')

        # Validate connection on initialization
        self._validate_connection()

    def _validate_connection(self) -> bool:
        """Validate connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama server at {self.base_url}: {str(e)}")

    def _make_request(self, endpoint: str, data: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Make HTTP request to Ollama API with retry logic."""
        url = f"{self.base_url}/api/{endpoint}"

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data, timeout=60)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise Exception(f"Ollama API request failed after {max_retries} attempts: {str(e)}")

        # This should never be reached due to the raise in the except block
        raise Exception("Unexpected error in request processing")

    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Generate chat completion response using Ollama."""
        # Convert OpenAI-style messages to Ollama format
        if len(messages) == 1 and messages[0].get("role") == "system":
            prompt = messages[0]["content"]
        else:
            # Combine system and user messages
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n\n"

        data = {
            "model": self.model,
            "prompt": prompt.strip(),
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        response = self._make_request("generate", data)
        return response.get("response", "").strip()

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using Ollama."""
        data = {
            "model": self.embedding_model,
            "prompt": text
        }

        response = self._make_request("embeddings", data)
        return response.get("embedding", [])

    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
        except Exception as e:
            raise Exception(f"Failed to list models: {str(e)}")

    def pull_model(self, model_name: str) -> bool:
        """Pull a model to the Ollama server."""
        data = {"name": model_name}
        try:
            self._make_request("pull", data)
            return True
        except Exception as e:
            raise Exception(f"Failed to pull model {model_name}: {str(e)}")

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available on the server."""
        try:
            available_models = self.list_models()
            return model_name in available_models
        except Exception:
            return False