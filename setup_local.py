#!/usr/bin/env python3
"""
Local AI Stack Setup Script

This script sets up the local AI infrastructure for the memory benchmarking project:
- Validates Docker availability
- Starts Qdrant via Docker Compose
- Pulls required Ollama models
- Validates the complete setup
"""

import os
import sys
import subprocess
import time
import requests
from typing import Optional
from dotenv import load_dotenv


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_status(message: str, status: str = "INFO"):
    """Print colored status messages"""
    color = Colors.BLUE
    if status == "SUCCESS":
        color = Colors.GREEN
    elif status == "ERROR":
        color = Colors.RED
    elif status == "WARNING":
        color = Colors.YELLOW

    print(f"{color}[{status}]{Colors.END} {message}")


def run_command(command: str, capture_output: bool = False, check: bool = True) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command with proper error handling"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print_status(f"Command failed: {command}", "ERROR")
        print_status(f"Error: {e.stderr if e.stderr else str(e)}", "ERROR")
        return None


def check_docker():
    """Check if Docker is installed and running"""
    print_status("Checking Docker availability...")

    # Check if Docker is installed
    result = run_command("docker --version", capture_output=True, check=False)
    if not result or result.returncode != 0:
        print_status("Docker is not installed. Please install Docker first.", "ERROR")
        return False

    # Check if Docker daemon is running
    result = run_command("docker info", capture_output=True, check=False)
    if not result or result.returncode != 0:
        print_status("Docker daemon is not running. Please start Docker.", "ERROR")
        return False

    print_status("Docker is available and running", "SUCCESS")
    return True


def check_docker_compose():
    """Check if Docker Compose is available"""
    print_status("Checking Docker Compose availability...")

    # Try docker compose (new syntax)
    result = run_command("docker compose version", capture_output=True, check=False)
    if result and result.returncode == 0:
        print_status("Docker Compose is available", "SUCCESS")
        return "docker compose"

    # Try docker-compose (legacy syntax)
    result = run_command("docker-compose --version", capture_output=True, check=False)
    if result and result.returncode == 0:
        print_status("Docker Compose (legacy) is available", "SUCCESS")
        return "docker-compose"

    print_status("Docker Compose is not available", "ERROR")
    return None


def start_qdrant(compose_command: str):
    """Start Qdrant using Docker Compose"""
    print_status("Starting Qdrant service...")

    result = run_command(f"{compose_command} up -d qdrant", check=False)
    if not result or result.returncode != 0:
        print_status("Failed to start Qdrant", "ERROR")
        return False

    # Wait for Qdrant to be ready
    print_status("Waiting for Qdrant to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:6333/health", timeout=5)
            if response.status_code == 200:
                print_status("Qdrant is ready", "SUCCESS")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(2)
        print(f"Retrying... ({i + 1}/{max_retries})")

    print_status("Qdrant failed to start within expected time", "ERROR")
    return False


def check_ollama():
    """Check if Ollama is installed and running"""
    print_status("Checking Ollama availability...")

    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version_data = response.json()
            print_status(f"Ollama is running (version: {version_data.get('version', 'unknown')})", "SUCCESS")
            return True
    except requests.exceptions.RequestException:
        pass

    print_status("Ollama is not running. Please install and start Ollama first.", "WARNING")
    print_status("Visit: https://ollama.ai/download", "INFO")
    return False


def pull_ollama_models():
    """Pull required Ollama models"""
    load_dotenv()

    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")

    models_to_pull = [model, embedding_model]

    for model_name in models_to_pull:
        print_status(f"Pulling Ollama model: {model_name}")
        result = run_command(f"ollama pull {model_name}", check=False)

        if not result or result.returncode != 0:
            print_status(f"Failed to pull model: {model_name}", "ERROR")
            return False
        else:
            print_status(f"Successfully pulled model: {model_name}", "SUCCESS")

    return True


def validate_setup():
    """Validate the complete setup"""
    print_status("Validating complete setup...")

    # Check Qdrant
    try:
        response = requests.get("http://localhost:6333/health", timeout=5)
        if response.status_code == 200:
            print_status("✓ Qdrant is accessible", "SUCCESS")
        else:
            print_status("✗ Qdrant health check failed", "ERROR")
            return False
    except requests.exceptions.RequestException as e:
        print_status(f"✗ Qdrant is not accessible: {e}", "ERROR")
        return False

    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print_status("✓ Ollama is accessible", "SUCCESS")
        else:
            print_status("✗ Ollama is not accessible", "ERROR")
            return False
    except requests.exceptions.RequestException as e:
        print_status(f"✗ Ollama is not accessible: {e}", "ERROR")
        return False

    return True


def main():
    """Main setup function"""
    print(f"{Colors.BOLD}=== Local AI Stack Setup ==={Colors.END}\n")

    # Check prerequisites
    if not check_docker():
        sys.exit(1)

    compose_command = check_docker_compose()
    if not compose_command:
        sys.exit(1)

    # Start Qdrant
    if not start_qdrant(compose_command):
        sys.exit(1)

    # Check Ollama (optional but recommended)
    ollama_available = check_ollama()

    if ollama_available:
        # Pull Ollama models
        if not pull_ollama_models():
            print_status("Failed to pull some Ollama models, but continuing...", "WARNING")

    # Final validation
    if validate_setup():
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Local AI stack setup completed successfully!{Colors.END}")
        print(f"\n{Colors.BLUE}Services available:{Colors.END}")
        print(f"  • Qdrant REST API: http://localhost:6333")
        print(f"  • Qdrant gRPC: localhost:6334")
        if ollama_available:
            print(f"  • Ollama API: http://localhost:11434")
        print(f"\n{Colors.YELLOW}Next steps:{Colors.END}")
        print(f"  1. Copy .env.example to .env and configure your API keys")
        print(f"  2. Set LOCAL_SETUP=true in your .env file")
        print(f"  3. Install Python dependencies: pip install -r requirements.txt")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Setup validation failed{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()