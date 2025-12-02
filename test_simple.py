"""
Simple test script that doesn't require OpenAI API key.
Tests the vector database and retrieval functionality only.
"""
import sys

def test_vector_db():
    """Test the SimpleVectorDB functionality."""
    print("=" * 60)
    print("Test 1: SimpleVectorDB")
    print("=" * 60)

    try:
        import numpy as np
        from simple_vector_db import SimpleVectorDB

        # Create a simple vector database
        db = SimpleVectorDB(dimension=3)

        # Add some vectors
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        documents = ["Document A", "Document B", "Document C"]
        metadata = [{"id": 1}, {"id": 2}, {"id": 3}]

        db.add(vectors, documents, metadata)

        print(f"✓ Added {len(db)} documents to the database")

        # Search for similar vectors
        query_vector = np.array([1.0, 0.1, 0.0])
        results = db.search(query_vector, top_k=2)

        print(f"✓ Search returned {len(results)} results")
        print("\nTop results:")
        for i, (doc, score, meta) in enumerate(results, 1):
            print(f"  {i}. {doc} (score: {score:.4f}, metadata: {meta})")

        # Test save and load
        db.save("test_db.pkl")
        print("✓ Database saved successfully")

        db2 = SimpleVectorDB()
        db2.load("test_db.pkl")
        print(f"✓ Database loaded successfully ({len(db2)} documents)")

        print("\n✅ SimpleVectorDB tests PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ SimpleVectorDB tests FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_rag_without_openai():
    """Test RAG system without OpenAI API (retrieval only)."""
    print("=" * 60)
    print("Test 2: RAG System (Retrieval Only)")
    print("=" * 60)

    try:
        from rag_system import SimpleRAG

        # Initialize RAG system
        print("Initializing RAG system...")
        rag = SimpleRAG(
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=200,
            chunk_overlap=20
        )

        print("✓ RAG system initialized")

        # Add test documents
        documents = [
            "Python is a programming language. It is widely used for web development, data science, and automation.",
            "Machine learning is a branch of artificial intelligence. It enables computers to learn from data.",
            "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation."
        ]

        print("\nAdding documents...")
        rag.add_documents(documents)

        stats = rag.get_stats()
        print(f"✓ Indexed {stats['total_chunks']} chunks")

        # Test retrieval
        print("\nTesting retrieval:")
        query = "What is Python?"
        results = rag.retrieve(query, top_k=2)

        print(f"Query: {query}")
        print(f"✓ Retrieved {len(results)} results")

        for i, (doc, score, meta) in enumerate(results, 1):
            print(f"\n  Result {i} (score: {score:.4f}):")
            print(f"  {doc[:100]}...")

        print("\n✅ RAG System tests PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ RAG System tests FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("TinyRAG - Simple Test Suite")
    print("=" * 60 + "\n")

    # Run tests
    test1_passed = test_vector_db()
    test2_passed = test_rag_without_openai()

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"SimpleVectorDB: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"RAG System: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print("\n⚠️  Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
