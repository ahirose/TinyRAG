"""
Simple RAG (Retrieval-Augmented Generation) System.
Provides document indexing, retrieval, and answer generation.
"""
import os
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from simple_vector_db import SimpleVectorDB


class SimpleRAG:
    """A simple RAG system with document chunking, embedding, and retrieval."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_model: str = "gpt-3.5-turbo",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the RAG system.

        Args:
            embedding_model: sentence-transformers model name
            openai_model: OpenAI model name for generation
            chunk_size: maximum size of text chunks in characters
            chunk_overlap: overlap between chunks in characters
        """
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()

        self.vector_db = SimpleVectorDB(dimension=self.embedding_dimension)
        self.openai_model = openai_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key) if api_key else None

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: input text to chunk

        Returns:
            list of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence or word boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_space = chunk.rfind(' ')

                break_point = max(last_period, last_newline, last_space)
                if break_point > 0:
                    chunk = chunk[:break_point + 1]
                    end = start + len(chunk)

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]

    def add_documents(self, documents: List[str], metadata: Optional[List[dict]] = None):
        """
        Add documents to the RAG system.

        Args:
            documents: list of document texts
            metadata: optional list of metadata dictionaries
        """
        all_chunks = []
        all_metadata = []

        for i, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)

            # Propagate metadata to chunks
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            for j, chunk in enumerate(chunks):
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update({
                    'doc_index': i,
                    'chunk_index': j,
                    'total_chunks': len(chunks)
                })
                all_metadata.append(chunk_metadata)

        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)

        print(f"Adding {len(all_chunks)} chunks to vector database...")
        self.vector_db.add(
            vectors=np.array(embeddings),
            documents=all_chunks,
            metadata=all_metadata
        )

        print(f"Successfully indexed {len(documents)} documents ({len(all_chunks)} chunks)")

    def retrieve(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: search query
            top_k: number of results to return

        Returns:
            list of tuples (document, similarity_score, metadata)
        """
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.vector_db.search(query_embedding, top_k=top_k)
        return results

    def generate_answer(self, query: str, top_k: int = 3, temperature: float = 0.7) -> dict:
        """
        Generate an answer using retrieved context.

        Args:
            query: user question
            top_k: number of context chunks to retrieve
            temperature: generation temperature

        Returns:
            dictionary with answer, sources, and retrieved contexts
        """
        if not self.openai_client:
            return {
                'answer': "Error: OPENAI_API_KEY not set. Please set the environment variable.",
                'sources': [],
                'contexts': []
            }

        # Retrieve relevant contexts
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return {
                'answer': "No relevant information found in the knowledge base.",
                'sources': [],
                'contexts': []
            }

        # Prepare context
        contexts = [doc for doc, score, meta in results]
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        # Create prompt
        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

{context_text}

Question: {query}

Answer:"""

        # Generate answer
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )

            answer = response.choices[0].message.content

            return {
                'answer': answer,
                'sources': [{'text': doc, 'score': score, 'metadata': meta} for doc, score, meta in results],
                'contexts': contexts
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'contexts': contexts
            }

    def save(self, filepath: str):
        """Save the vector database."""
        self.vector_db.save(filepath)
        print(f"Database saved to {filepath}")

    def load(self, filepath: str):
        """Load the vector database."""
        self.vector_db.load(filepath)
        print(f"Database loaded from {filepath}")

    def clear(self):
        """Clear all indexed documents."""
        self.vector_db.clear()
        print("Database cleared")

    def get_stats(self) -> dict:
        """Get statistics about the indexed documents."""
        return {
            'total_chunks': len(self.vector_db),
            'embedding_dimension': self.embedding_dimension,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
