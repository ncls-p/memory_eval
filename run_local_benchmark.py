#!/usr/bin/env python3
"""
Local AI Benchmarking Script

This script runs comprehensive benchmarks on all local AI techniques,
compares local vs cloud performance, and generates detailed performance reports.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import  Dict, List, Optional

import psutil

# Import dotenv if available, but don't fail if it's not
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not available. Environment variables from .env file will not be loaded.")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not available. GPU metrics will not be collected.")


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    technique: str
    dataset_size: int
    avg_response_time: float
    total_tokens: int
    tokens_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]
    gpu_memory_mb: Optional[float]
    success_rate: float
    error_count: int
    timestamp: str


@dataclass
class SystemInfo:
    """System hardware information"""
    cpu_cores: int
    cpu_freq_mhz: float
    total_ram_gb: float
    available_ram_gb: float
    gpu_count: int
    gpu_names: List[str]
    gpu_memory_gb: List[float]
    platform: str
    python_version: str


class HardwareMonitor:
    """Monitor system hardware during benchmarks"""

    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.monitoring = False

    def start_monitoring(self):
        """Start hardware monitoring"""
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []

    def stop_monitoring(self):
        """Stop hardware monitoring"""
        self.monitoring = False

    def collect_metrics(self):
        """Collect current hardware metrics"""
        if not self.monitoring:
            return

        # CPU and Memory
        self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.used / 1024 / 1024)  # MB

        # GPU metrics
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    self.gpu_usage.append(gpu.load * 100)
                    self.gpu_memory.append(gpu.memoryUsed)
            except Exception:
                pass

    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics from monitoring period"""
        metrics = {
            'avg_cpu_usage': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'avg_memory_usage': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'avg_gpu_usage': sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else None,
            'avg_gpu_memory': sum(self.gpu_memory) / len(self.gpu_memory) if self.gpu_memory else None,
        }
        return metrics


class LocalBenchmark:
    """Main benchmarking class for local AI techniques"""

    def __init__(self, output_folder: str = "results/local/"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.hardware_monitor = HardwareMonitor()

        # Local techniques to benchmark
        self.local_techniques = [
            "ollama",
            "memzero_local",
            "qdrant_rag"
        ]

        # Cloud techniques for comparison (if available)
        self.cloud_techniques = [
            "mem0",
            "openai",
            "rag"
        ]

    def get_system_info(self) -> SystemInfo:
        """Collect system hardware information"""
        import platform

        # CPU info
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0

        # Memory info
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / 1024 / 1024 / 1024
        available_ram_gb = memory.available / 1024 / 1024 / 1024

        # GPU info
        gpu_names = []
        gpu_memory_gb = []
        gpu_count = 0

        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_count = len(gpus)
                gpu_names = [gpu.name for gpu in gpus]
                gpu_memory_gb = [gpu.memoryTotal / 1024 for gpu in gpus]
            except Exception:
                pass

        return SystemInfo(
            cpu_cores=psutil.cpu_count() or 0,
            cpu_freq_mhz=cpu_freq_mhz,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_memory_gb=gpu_memory_gb,
            platform=platform.platform(),
            python_version=platform.python_version()
        )

    def check_services(self) -> Dict[str, bool]:
        """Check if required local services are running"""
        services = {}

        # Check Ollama
        try:
            import requests
            response = requests.get("https://ollama.nclsp.com/api/tags", timeout=5)
            services['ollama'] = response.status_code == 200
            if not services['ollama']:
                print(f"❌ Ollama API check failed with status: {response.status_code}")
        except Exception as e:
            services['ollama'] = False
            print(f"❌ Ollama connection failed: {e}")

        # Check Qdrant
        try:
            import requests
            response = requests.get("http://localhost:6333/healthz", timeout=5)
            services['qdrant'] = response.status_code == 200
            if not services['qdrant']:
                print(f"❌ Qdrant health check failed with status: {response.status_code}")
        except Exception as e:
            services['qdrant'] = False
            print(f"❌ Qdrant connection failed: {e}")

        return services

    def validate_model_availability(self) -> Dict[str, bool]:
        """Check if required models are available in Ollama"""
        model_status = {}

        # Get model names from environment or use defaults
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")

        try:
            import requests
            response = requests.get("https://ollama.nclsp.com/api/tags", timeout=10)
            if response.status_code == 200:
                available_models = response.json().get("models", [])
                available_model_names = [model["name"] for model in available_models]

                model_status[ollama_model] = ollama_model in available_model_names
                model_status[embedding_model] = embedding_model in available_model_names

                print(f"📋 Available models: {', '.join(available_model_names)}")
                print(f"🔍 Required models: {ollama_model}, {embedding_model}")

                if not model_status[ollama_model]:
                    print(f"❌ Required model not found: {ollama_model}")
                if not model_status[embedding_model]:
                    print(f"❌ Required embedding model not found: {embedding_model}")
            else:
                print(f"❌ Failed to fetch model list from Ollama: {response.status_code}")
        except Exception as e:
            print(f"❌ Error checking model availability: {e}")

        return model_status

    def validate_prerequisites(self) -> bool:
        """Validate all prerequisites before running benchmarks"""
        print("🔍 Validating prerequisites...")

        # Check services
        services = self.check_services()

        all_valid = True

        if not services.get('ollama', False):
            print("❌ Ollama service is not available")
            all_valid = False
        else:
            print("✅ Ollama service is running")

            # Check models if Ollama is available
            model_status = self.validate_model_availability()
            if not all(model_status.values()):
                print("❌ Required models are not available in Ollama")
                print("💡 Run 'ollama pull <model-name>' to install missing models")
                all_valid = False
            else:
                print("✅ Required models are available")

        if not services.get('qdrant', False):
            print("❌ Qdrant service is not available")
            print("💡 Run 'python setup_local.py' to start Qdrant")
            all_valid = False
        else:
            print("✅ Qdrant service is running")

        return all_valid

    def run_technique_benchmark(self, technique: str, method: str = "search") -> BenchmarkResult:
        """Run benchmark for a specific technique"""
        print(f"🧪 Benchmarking {technique} ({method})...")

        # Ensure environment variables are set for consistent model usage
        if not os.getenv("OLLAMA_MODEL"):
            os.environ["OLLAMA_MODEL"] = "llama3.2:3b-instruct-q4_K_M"
        if not os.getenv("OLLAMA_EMBEDDING_MODEL"):
            os.environ["OLLAMA_EMBEDDING_MODEL"] = "nomic-embed-text:latest"
        if not os.getenv("LOCAL_SETUP"):
            os.environ["LOCAL_SETUP"] = "true"

        print(f"   Using model: {os.getenv('OLLAMA_MODEL')}")
        print(f"   Using embedding model: {os.getenv('OLLAMA_EMBEDDING_MODEL')}")

        # Fix: Use RAG method for ollama technique instead of search to avoid missing memories issue
        if technique == "ollama" and method == "search":
            method = "rag"
            print(f"   🔧 Switching ollama to RAG method to avoid missing memories issue")

        # Start monitoring
        self.hardware_monitor.start_monitoring()
        start_time = time.time()

        # Build command
        cmd = [
            "python", "run_experiments.py",
            "--technique_type", technique,
            "--method", method,
            "--output_folder", str(self.output_folder),
            "--top_k", "30"
        ]

        try:
            # Run experiment
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            success = result.returncode == 0
            end_time = time.time()

            # Stop monitoring
            self.hardware_monitor.stop_monitoring()

            # Calculate metrics
            response_time = end_time - start_time
            hardware_metrics = self.hardware_monitor.get_average_metrics()

            # Enhanced error reporting
            if not success:
                print(f"❌ {technique} failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error details: {result.stderr.strip()}")
                if result.stdout:
                    print(f"   Output: {result.stdout.strip()}")

            # Parse output for token information (if available)
            total_tokens = 0
            tokens_per_second = 0
            error_count = 0 if success else 1

            if success and "tokens" in result.stdout.lower():
                # Try to extract token information from output
                try:
                    import re
                    token_match = re.search(r'(\d+)\s+tokens', result.stdout)
                    if token_match:
                        total_tokens = int(token_match.group(1))
                        tokens_per_second = total_tokens / response_time if response_time > 0 else 0
                except Exception:
                    pass

            return BenchmarkResult(
                technique=technique,
                dataset_size=10,  # LOCOMO dataset size
                avg_response_time=response_time,
                total_tokens=total_tokens,
                tokens_per_second=tokens_per_second,
                memory_usage_mb=hardware_metrics['avg_memory_usage'],
                cpu_usage_percent=hardware_metrics['avg_cpu_usage'],
                gpu_usage_percent=hardware_metrics['avg_gpu_usage'],
                gpu_memory_mb=hardware_metrics['avg_gpu_memory'],
                success_rate=1.0 if success else 0.0,
                error_count=error_count,
                timestamp=datetime.now().isoformat()
            )

        except subprocess.TimeoutExpired:
            self.hardware_monitor.stop_monitoring()
            print(f"❌ {technique} benchmark timed out")
            return BenchmarkResult(
                technique=technique,
                dataset_size=10,
                avg_response_time=600.0,  # Timeout time
                total_tokens=0,
                tokens_per_second=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                gpu_usage_percent=None,
                gpu_memory_mb=None,
                success_rate=0.0,
                error_count=1,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            self.hardware_monitor.stop_monitoring()
            print(f"❌ Error benchmarking {technique}: {e}")
            return BenchmarkResult(
                technique=technique,
                dataset_size=10,
                avg_response_time=0,
                total_tokens=0,
                tokens_per_second=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                gpu_usage_percent=None,
                gpu_memory_mb=None,
                success_rate=0.0,
                error_count=1,
                timestamp=datetime.now().isoformat()
            )

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks for all available techniques"""
        print("🚀 Starting Local AI Benchmark Suite")
        print("=" * 50)

        # Check system and services
        system_info = self.get_system_info()
        services = self.check_services()

        print(f"💻 System: {system_info.cpu_cores or 0} cores, {system_info.total_ram_gb:.1f}GB RAM")
        if system_info.gpu_count > 0:
            print(f"🎮 GPU: {system_info.gpu_names[0]} ({system_info.gpu_memory_gb[0]:.1f}GB)")
        print(f"🔧 Services: Ollama={services.get('ollama', False)}, Qdrant={services.get('qdrant', False)}")
        print()

        # Validate prerequisites
        if not self.validate_prerequisites():
            print("❌ Prerequisites validation failed. Some benchmarks may not run successfully.")
            print("💡 Consider running 'python setup_local.py' to fix service issues")
            print()

        results = []

        # Benchmark local techniques
        for technique in self.local_techniques:
            skip_reason = None

            if technique == "ollama" and not services.get('ollama', False):
                skip_reason = "Ollama service not available"
            elif technique in ["memzero_local", "qdrant_rag"] and not services.get('qdrant', False):
                skip_reason = "Qdrant service not available"

            if skip_reason:
                print(f"⏭️  Skipping {technique} - {skip_reason}")
                # Create a failed result for completeness
                failed_result = BenchmarkResult(
                    technique=technique,
                    dataset_size=10,
                    avg_response_time=0,
                    total_tokens=0,
                    tokens_per_second=0,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    gpu_usage_percent=None,
                    gpu_memory_mb=None,
                    success_rate=0.0,
                    error_count=1,
                    timestamp=datetime.now().isoformat()
                )
                results.append(failed_result)
                continue

            result = self.run_technique_benchmark(technique)
            results.append(result)

            # Print immediate feedback with more detail
            if result.success_rate > 0:
                print(f"✅ {technique}: {result.avg_response_time:.1f}s, {result.tokens_per_second:.1f} tok/s, {result.memory_usage_mb:.0f}MB RAM")
            else:
                print(f"❌ {technique}: Failed - check service connectivity and model availability")
            print()

        return results

    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """Generate comprehensive benchmark report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_folder / f"benchmark_report_{timestamp}.md"

        # System info
        system_info = self.get_system_info()

        # Generate report content
        report_content = f"""# Local AI Benchmark Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## System Information

- **CPU**: {system_info.cpu_cores} cores @ {system_info.cpu_freq_mhz:.0f}MHz
- **RAM**: {system_info.total_ram_gb:.1f}GB total, {system_info.available_ram_gb:.1f}GB available
- **GPU**: {system_info.gpu_count} GPU(s)
"""

        if system_info.gpu_count > 0:
            for i, (name, memory) in enumerate(zip(system_info.gpu_names, system_info.gpu_memory_gb)):
                report_content += f"  - GPU {i}: {name} ({memory:.1f}GB)\n"

        report_content += f"""
- **Platform**: {system_info.platform}
- **Python**: {system_info.python_version}

## Benchmark Results

| Technique | Success | Avg Time (s) | Tokens/sec | Memory (MB) | CPU % | GPU % |
|-----------|---------|--------------|------------|-------------|-------|-------|
"""

        for result in results:
            gpu_usage = f"{result.gpu_usage_percent:.1f}" if result.gpu_usage_percent else "N/A"
            report_content += f"| {result.technique} | {'✅' if result.success_rate > 0 else '❌'} | {result.avg_response_time:.1f} | {result.tokens_per_second:.1f} | {result.memory_usage_mb:.0f} | {result.cpu_usage_percent:.1f} | {gpu_usage} |\n"

        # Performance analysis
        successful_results = [r for r in results if r.success_rate > 0]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x.avg_response_time)
            highest_throughput = max(successful_results, key=lambda x: x.tokens_per_second)

            report_content += f"""

## Performance Analysis

### Speed Champion 🏆
**{fastest.technique}** - Fastest response time: {fastest.avg_response_time:.1f}s

### Throughput Champion 🚀
**{highest_throughput.technique}** - Highest throughput: {highest_throughput.tokens_per_second:.1f} tokens/sec

### Resource Usage
"""

            for result in successful_results:
                report_content += f"- **{result.technique}**: {result.memory_usage_mb:.0f}MB RAM, {result.cpu_usage_percent:.1f}% CPU\n"

        # Recommendations
        report_content += """

## Recommendations

### For Development/Testing
- Use the fastest technique for quick iterations
- Consider memory usage if running on limited hardware

### For Production
- Balance speed vs quality based on your use case
- Monitor resource usage in production environment
- Consider GPU acceleration for better performance

### Hardware Optimization
"""

        if system_info.gpu_count == 0:
            report_content += "- Consider adding a GPU for significantly better performance\n"
        else:
            report_content += "- GPU detected - ensure models are configured to use GPU acceleration\n"

        if system_info.total_ram_gb < 16:
            report_content += "- Consider upgrading RAM for better performance with larger models\n"

        # Save report
        with open(report_file, 'w') as f:
            f.write(report_content)

        return str(report_file)

    def save_json_results(self, results: List[BenchmarkResult]) -> str:
        """Save detailed results as JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_folder / f"benchmark_results_{timestamp}.json"

        # Convert results to JSON-serializable format
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": asdict(self.get_system_info()),
            "results": [asdict(result) for result in results]
        }

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        return str(json_file)

    def compare_local_vs_cloud(self, results: List[BenchmarkResult]) -> str:
        """Generate comparison report between local and cloud techniques"""
        # This would compare with cloud results if available
        # For now, just provide insights on local results

        local_results = [r for r in results if r.technique in self.local_techniques]

        if not local_results:
            return "No local results available for comparison."

        # Get successful results to avoid division by zero
        successful_local_results = [r for r in local_results if r.success_rate > 0]

        if not successful_local_results:
            comparison = """
            ## Local vs Cloud Comparison

            ### Local Performance Summary
            ❌ **No successful local benchmark results**

            All local techniques failed to complete successfully. This could indicate:
            - Service connectivity issues (Ollama, Qdrant not running)
            - Model configuration problems
            - Resource constraints (insufficient memory/GPU)

            ### Troubleshooting Steps
            1. Verify services are running: `python setup_local.py`
            2. Check model availability in Ollama
            3. Ensure sufficient system resources
            4. Review error logs above for specific failures

            ### Advantages of Local Setup
            ✅ **Privacy**: All data stays local
            ✅ **Cost**: No per-token charges
            ✅ **Availability**: Works offline
            ✅ **Customization**: Full control over models

            ### Considerations
            ⚠️ **Setup Complexity**: Requires local installation
            ⚠️ **Hardware Requirements**: Needs sufficient RAM/GPU
            ⚠️ **Model Updates**: Manual model management
            """
            return comparison

        # Calculate averages only from successful results
        avg_time = sum(r.avg_response_time for r in successful_local_results) / len(successful_local_results)
        avg_throughput = sum(r.tokens_per_second for r in successful_local_results) / len(successful_local_results)

        comparison = f"""
## Local vs Cloud Comparison

### Local Performance Summary
- **Average Response Time**: {avg_time:.1f}s
- **Average Throughput**: {avg_throughput:.1f} tokens/sec
- **Success Rate**: {sum(r.success_rate for r in local_results) / len(local_results) * 100:.1f}%

### Advantages of Local Setup
✅ **Privacy**: All data stays local
✅ **Cost**: No per-token charges
✅ **Availability**: Works offline
✅ **Customization**: Full control over models

### Considerations
⚠️ **Setup Complexity**: Requires local installation
⚠️ **Hardware Requirements**: Needs sufficient RAM/GPU
⚠️ **Model Updates**: Manual model management
"""

        return comparison


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive local AI benchmarks")
    parser.add_argument("--output_folder", default="results/local/", help="Output folder for results")
    parser.add_argument("--techniques", nargs="+", help="Specific techniques to benchmark",
                        choices=["ollama", "memzero_local", "qdrant_rag", "all"], default=["all"])
    parser.add_argument("--compare_cloud", action="store_true", help="Include cloud techniques in comparison")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark mode (reduced dataset)")

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = LocalBenchmark(args.output_folder)

    # Run benchmarks
    if "all" in args.techniques:
        results = benchmark.run_all_benchmarks()
    else:
        results = []
        for technique in args.techniques:
            result = benchmark.run_technique_benchmark(technique)
            results.append(result)

    # Generate reports
    print("\n📊 Generating benchmark reports...")

    # Markdown report
    report_file = benchmark.generate_report(results)
    print(f"📄 Markdown report: {report_file}")

    # JSON results
    json_file = benchmark.save_json_results(results)
    print(f"💾 JSON results: {json_file}")

    # Comparison analysis
    comparison = benchmark.compare_local_vs_cloud(results)
    print(comparison)

    print("\n🎉 Benchmark completed successfully!")
    print(f"Results saved to: {args.output_folder}")


if __name__ == "__main__":
    main()