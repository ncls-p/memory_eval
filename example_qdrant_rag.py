#!/usr/bin/env python3
"""
Example usage of the Qdrant RAG implementation.
This script demonstrates how to use the local RAG solution with Qdrant and Ollama.
"""

import json
import tempfile
import time

def create_sample_data():
    """Create sample conversation data for testing."""
    return {
        "conversation_1": {
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "day_1": [
                    {
                        "speaker": "Alice",
                        "text": "Hi Bob! I heard you're working on a new AI project. What's it about?"
                    },
                    {
                        "speaker": "Bob",
                        "text": "Yes! I'm building a RAG system using Qdrant for vector storage and Ollama for embeddings."
                    },
                    {
                        "speaker": "Alice",
                        "text": "That sounds interesting! Why did you choose Qdrant over other vector databases?"
                    },
                    {
                        "speaker": "Bob",
                        "text": "Qdrant is fast, scalable, and works great for local development. Plus it has excellent Python support."
                    }
                ],
                "day_1_date_time": "2024-01-15 14:30:00"
            },
            "question": [
                {
                    "question": "What vector database is Bob using for his project?",
                    "answer": "Qdrant",
                    "category": "factual"
                },
                {
                    "question": "What does Bob use for embeddings?",
                    "answer": "Ollama",
                    "category": "factual"
                }
            ]
        }
    }

def example_basic_usage():
    """Example of basic Qdrant RAG usage."""
    print("🚀 Basic Qdrant RAG Example")
    print("=" * 40)

    from src.qdrant_rag.rag import QdrantRAGManager

    # Initialize RAG manager
    rag_manager = QdrantRAGManager(
        collection_name="example_basic",
        chunk_size=200,
        k=2
    )

    # Create sample conversation
    conversation = [
        {
            "timestamp": "2024-01-15 14:30:00",
            "speaker": "Alice",
            "text": "Hi Bob! I heard you're working on a new AI project. What's it about?"
        },
        {
            "timestamp": "2024-01-15 14:31:00",
            "speaker": "Bob",
            "text": "Yes! I'm building a RAG system using Qdrant for vector storage and Ollama for embeddings."
        },
        {
            "timestamp": "2024-01-15 14:32:00",
            "speaker": "Alice",
            "text": "That sounds interesting! Why did you choose Qdrant?"
        },
        {
            "timestamp": "2024-01-15 14:33:00",
            "speaker": "Bob",
            "text": "Qdrant is fast, scalable, and works great for local development."
        }
    ]

    print("📝 Storing conversation chunks...")
    chunk_ids = rag_manager.store_conversation_chunks("conv_example", conversation)
    print(f"✅ Stored {len(chunk_ids)} chunks")

    print("\n🔍 Searching for relevant information...")
    question = "What vector database is Bob using?"
    context, search_time = rag_manager.search_relevant_chunks(question, limit=2)
    print(f"📊 Search completed in {search_time:.3f}s")
    print(f"📄 Retrieved context: {context[:100]}...")

    print("\n🤖 Generating response...")
    response, response_time = rag_manager.generate_response(question, context)
    print(f"⏱️ Response generated in {response_time:.3f}s")
    print(f"💬 Question: {question}")
    print(f"🎯 Response: {response}")

    # Clean up
    rag_manager.clear_collection()
    print("\n🧹 Collection cleared")

def example_add_search_workflow():
    """Example of the complete add and search workflow."""
    print("\n🔄 Complete Add/Search Workflow Example")
    print("=" * 45)

    # Create sample data
    sample_data = create_sample_data()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f)
        data_file = f.name

    try:
        from src.qdrant_rag.add import QdrantRAGAdd
        from src.qdrant_rag.search import QdrantRAGSearch

        collection_name = "example_workflow"

        # Add phase
        print("📝 Adding memories to Qdrant...")
        add_manager = QdrantRAGAdd(
            data_path=data_file,
            collection_name=collection_name,
            chunk_size=300
        )
        add_manager.process_all_conversations(max_workers=1, clear_existing=True)

        # Search phase
        print("\n🔍 Searching memories and generating responses...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        search_manager = QdrantRAGSearch(
            output_file_path=output_file,
            top_k=5,
            collection_name=collection_name,
            chunk_size=300
        )
        search_manager.process_data_file(data_file)

        # Display results
        print("\n📊 Results:")
        with open(output_file, 'r') as f:
            results = json.load(f)

        for conv_id, questions in results.items():
            print(f"\n🗂️ Conversation: {conv_id}")
            for i, q in enumerate(questions, 1):
                print(f"   {i}. Q: {q['question']}")
                print(f"      A: {q['response']}")
                print(f"      ⏱️ Search: {q['search_time']:.3f}s, Response: {q['response_time']:.3f}s")

        # Clean up
        import os
        os.unlink(output_file)

    finally:
        import os
        os.unlink(data_file)

def example_collection_management():
    """Example of collection management operations."""
    print("\n🗄️ Collection Management Example")
    print("=" * 35)

    from src.qdrant_rag.client import QdrantRAGClient

    client = QdrantRAGClient()
    test_collection = "example_management"

    print("📊 Collection operations:")

    # Create collection
    print("   📁 Creating collection...")
    client.create_collection(test_collection, vector_size=768, recreate=True)

    # Check if exists
    exists = client.collection_exists(test_collection)
    print(f"   ✅ Collection exists: {exists}")

    # Add some test points
    print("   📝 Adding test points...")
    test_points = [
        {
            "id": "point_1",
            "vector": [0.1] * 768,
            "payload": {"text": "Example document 1", "category": "test"}
        },
        {
            "id": "point_2",
            "vector": [0.2] * 768,
            "payload": {"text": "Example document 2", "category": "test"}
        }
    ]
    client.upsert_points(test_collection, test_points)

    # Get collection info
    info = client.get_collection_info(test_collection)
    print(f"   📈 Collection info: {info}")

    # Search
    print("   🔍 Testing search...")
    results = client.search_points(
        collection_name=test_collection,
        query_vector=[0.15] * 768,
        limit=2
    )
    print(f"   🎯 Found {len(results)} results")

    # Clean up
    client.delete_collection(test_collection)
    print("   🧹 Collection deleted")

def main():
    """Run all examples."""
    print("🎯 Qdrant RAG Implementation Examples")
    print("=" * 50)
    print("This script demonstrates the Qdrant RAG implementation features.")
    print("Make sure Qdrant and Ollama are running before executing.\n")

    try:
        # Test connections first
        print("🔍 Testing connections...")

        from src.qdrant_rag.client import QdrantRAGClient
        from src.ollama.client import OllamaClient

        # Test Qdrant
        qdrant_client = QdrantRAGClient()
        qdrant_healthy = qdrant_client.health_check()
        print(f"   Qdrant: {'✅ Connected' if qdrant_healthy else '❌ Not available'}")

        # Test Ollama
        try:
            ollama_client = OllamaClient()
            models = ollama_client.list_models()
            print(f"   Ollama: ✅ Connected ({len(models)} models available)")
        except:
            print("   Ollama: ❌ Not available")
            return

        if not qdrant_healthy:
            print("\n❌ Qdrant is not available. Please start it with:")
            print("   docker run -p 6333:6333 qdrant/qdrant")
            return

        print("\n✅ All services are ready!")

        # Run examples
        example_basic_usage()
        example_collection_management()
        example_add_search_workflow()

        print("\n🎉 All examples completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Run: python run_experiments.py --technique_type qdrant_rag --method add")
        print("   2. Run: python run_experiments.py --technique_type qdrant_rag --method search --output_folder results/")
        print("   3. Check the results in the output folder")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that Qdrant and Ollama services are running")

if __name__ == "__main__":
    main()