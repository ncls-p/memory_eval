#!/bin/bash
# Setup directory structure for local AI stack

echo "🗂️  Creating directory structure for local AI stack..."

# Create main data directory
mkdir -p data

# Create subdirectories for each service
mkdir -p data/qdrant_storage
mkdir -p data/ollama_models
mkdir -p data/redis_data
mkdir -p data/prometheus_data
mkdir -p data/grafana_data

# Create config directories
mkdir -p qdrant_config
mkdir -p ollama_config
mkdir -p redis_config
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/provisioning/datasources

# Create results directories
mkdir -p results/local
mkdir -p results/cloud
mkdir -p benchmarks
mkdir -p performance_reports

# Create logs directory
mkdir -p logs

echo "✅ Directory structure created successfully!"
echo "📁 Created directories:"
echo "   - data/ (persistent storage)"
echo "   - results/ (benchmark results)"
echo "   - monitoring/ (Grafana/Prometheus config)"
echo "   - logs/ (application logs)"

# Set proper permissions
chmod 755 data
chmod 755 results
chmod 755 logs

echo "🔒 Permissions set correctly"
echo "🎉 Setup complete! You can now run 'docker-compose -f docker-compose.full.yml up -d'"