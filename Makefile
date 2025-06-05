
# Run the experiments
run-mem0-add:
	python run_experiments.py --technique_type mem0 --method add

run-mem0-search:
	python run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 30

run-mem0-plus-add:
	python run_experiments.py --technique_type mem0 --method add --is_graph

run-mem0-plus-search:
	python run_experiments.py --technique_type mem0 --method search --is_graph --output_folder results/ --top_k 30

run-rag:
	python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 1 --output_folder results/

run-full-context:
	python run_experiments.py --technique_type rag --chunk_size -1 --num_chunks 1 --output_folder results/

run-langmem:
	python run_experiments.py --technique_type langmem --output_folder results/

run-zep-add:
	python run_experiments.py --technique_type zep --method add --output_folder results/

run-zep-search:
	python run_experiments.py --technique_type zep --method search --output_folder results/

run-openai:
	python run_experiments.py --technique_type openai --output_folder results/

# Local AI experiments
run-ollama:
	python run_experiments.py --technique_type ollama --output_folder results/

run-memzero-local:
	python run_experiments.py --technique_type memzero_local --output_folder results/

run-qdrant-rag:
	python run_experiments.py --technique_type qdrant_rag --output_folder results/

# Local AI setup and management
setup-directories:
	chmod +x setup_directories.sh
	./setup_directories.sh

setup-local:
	python setup_local.py

start-local:
	docker-compose up -d
	@echo "Waiting for Qdrant to be ready..."
	@sleep 10
	@echo "Checking Ollama service..."
	@curl -s http://localhost:11434/api/tags || echo "Ollama not running - please start it manually"

start-full-stack:
	make setup-directories
	docker-compose -f docker-compose.full.yml up -d
	@echo "Waiting for services to be ready..."
	@sleep 15
	@echo "Checking services..."
	@curl -s http://localhost:6333/health && echo "✅ Qdrant ready" || echo "❌ Qdrant not ready"
	@curl -s http://localhost:11434/api/tags && echo "✅ Ollama ready" || echo "❌ Ollama not ready"

start-with-monitoring:
	make setup-directories
	docker-compose -f docker-compose.full.yml --profile monitoring up -d
	@echo "Waiting for services to be ready..."
	@sleep 20
	@echo "Services available at:"
	@echo "  - Qdrant: http://localhost:6333"
	@echo "  - Ollama: http://localhost:11434"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Jaeger: http://localhost:16686"

test-local:
	python -m pytest test_ollama_integration.py -v
	python -m pytest test_memzero_local.py -v
	python -m pytest test_qdrant_rag.py -v

clean-local:
	docker-compose down -v
	docker volume prune -f
	@echo "Local services stopped and data cleaned"

clean-full-stack:
	docker-compose -f docker-compose.full.yml down -v
	docker-compose -f docker-compose.full.yml --profile monitoring down -v
	docker volume prune -f
	@echo "Full stack services stopped and data cleaned"

clean-all:
	make clean-full-stack
	rm -rf data/
	rm -rf results/local/
	rm -rf logs/
	@echo "All local data and results cleaned"

# Local benchmarks
run-local-benchmark:
	python run_local_benchmark.py --output_folder results/local/
