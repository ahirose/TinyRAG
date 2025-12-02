"""
Example usage of the Simple RAG system.
"""
import os
from dotenv import load_dotenv
from rag_system import SimpleRAG


def main():
    # Load environment variables
    load_dotenv()

    # Initialize RAG system
    print("=" * 60)
    print("Simple RAG System - Example")
    print("=" * 60)
    print()

    rag = SimpleRAG(
        embedding_model="all-MiniLM-L6-v2",  # Fast and lightweight
        openai_model="gpt-3.5-turbo",
        chunk_size=500,
        chunk_overlap=50
    )

    # Sample documents about Python and machine learning
    documents = [
        """
        Python is a high-level, interpreted programming language known for its simplicity and readability.
        It was created by Guido van Rossum and first released in 1991. Python supports multiple programming
        paradigms including procedural, object-oriented, and functional programming. It has a comprehensive
        standard library and a vast ecosystem of third-party packages available through PyPI (Python Package Index).
        """,
        """
        Machine Learning is a subset of artificial intelligence that focuses on the development of algorithms
        that can learn from and make predictions or decisions based on data. There are three main types of
        machine learning: supervised learning (learning with labeled data), unsupervised learning (finding
        patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment).
        """,
        """
        RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text
        generation. It works by first retrieving relevant documents from a knowledge base, then using those
        documents as context for a language model to generate more accurate and grounded responses. This
        approach helps reduce hallucinations and allows language models to access up-to-date information.
        """,
        """
        Vector databases are specialized databases designed to store and efficiently search high-dimensional
        vectors. They use similarity search algorithms like cosine similarity or Euclidean distance to find
        vectors that are semantically similar to a query vector. Vector databases are essential for modern
        AI applications including semantic search, recommendation systems, and RAG implementations.
        """
    ]

    metadata = [
        {"source": "Python Documentation", "topic": "Programming"},
        {"source": "ML Textbook", "topic": "Machine Learning"},
        {"source": "AI Research Paper", "topic": "RAG"},
        {"source": "Database Guide", "topic": "Vector Databases"}
    ]

    # Add documents to RAG system
    print("\n" + "=" * 60)
    print("Step 1: Indexing Documents")
    print("=" * 60)
    rag.add_documents(documents, metadata=metadata)

    # Show statistics
    stats = rag.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    print(f"  Chunk size: {stats['chunk_size']}")
    print(f"  Chunk overlap: {stats['chunk_overlap']}")

    # Example queries
    queries = [
        "What is Python?",
        "Explain what RAG means",
        "What are the types of machine learning?"
    ]

    print("\n" + "=" * 60)
    print("Step 2: Retrieval Examples")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = rag.retrieve(query, top_k=2)

        for i, (doc, score, meta) in enumerate(results, 1):
            print(f"\nResult {i} (Score: {score:.4f}):")
            print(f"Source: {meta.get('source', 'Unknown')}")
            print(f"Text: {doc[:150]}...")

    # Generate answers (requires OpenAI API key)
    if os.getenv("OPENAI_API_KEY"):
        print("\n" + "=" * 60)
        print("Step 3: Answer Generation")
        print("=" * 60)

        for query in queries:
            print(f"\nQuestion: {query}")
            print("-" * 60)
            result = rag.generate_answer(query, top_k=2)
            print(f"Answer: {result['answer']}")
            print(f"\nUsed {len(result['sources'])} context(s)")
    else:
        print("\n" + "=" * 60)
        print("Step 3: Answer Generation (Skipped)")
        print("=" * 60)
        print("Set OPENAI_API_KEY environment variable to enable answer generation")

    # Save the database
    print("\n" + "=" * 60)
    print("Step 4: Saving Database")
    print("=" * 60)
    rag.save("rag_database.pkl")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
