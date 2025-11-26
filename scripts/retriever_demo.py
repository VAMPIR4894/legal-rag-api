#!/usr/bin/env python3
"""
Test script to demonstrate all three retrievers: SimpleRetriever1, SimpleRetriever2, and HybridRetriever.
Shows the documents fetched by each retriever for a sample query.
"""

import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from retrieval.simple_retriever1 import SimpleRetriever
from retrieval.simple_retriever2 import SimpleRetriever2
from retrieval.hybrid_retriever import HybridRetriever

def main():
    # Sample query
    query = "What are the legal requirements for property disputes?"
    
    # Read configuration from environment
    retrieval_results = int(os.getenv('RETRIEVAL_RESULTS', '5'))

    print("=" * 80)
    print("RETRIEVER COMPARISON DEMO")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Results per method: {retrieval_results}")
    print()

    try:
        # Initialize retrievers
        print("Initializing retrievers...")

        # Method 1: Dense retrieval (SimpleRetriever1)
        print("1. Initializing SimpleRetriever1 (Dense Retrieval)...")
        retriever1 = SimpleRetriever()
        print(f"   Index size: {retriever1.get_index_size()} documents")

        # Method 2: Sparse retrieval (SimpleRetriever2)
        print("2. Initializing SimpleRetriever2 (Sparse BM25 Retrieval)...")
        retriever2 = SimpleRetriever2()
        print(f"   Index size: {retriever2.get_index_size()} documents")

        # Method 3: Hybrid retrieval (HybridRetriever)
        print("3. Initializing HybridRetriever (Dense + Sparse + Reranking)...")
        retriever3 = HybridRetriever()
        print(f"   Dense index size: {retriever3.get_index_size()} documents")
        print(f"   Sparse corpus size: {len(retriever3.corpus)} documents")

        print("\n" + "=" * 80)
        print("RETRIEVAL RESULTS")
        print("=" * 80)

        # Retrieve with each method
        top_k = retrieval_results

        # Method 1 Results
        print(f"\n1. SIMPLE RETRIEVER 1 (Dense Retrieval) - Top {top_k} results:")
        print("-" * 60)
        results1 = retriever1.retrieve(query, top_k=top_k)
        for i, doc in enumerate(results1, 1):
            print(f"{i}. Score: {doc['score']:.4f}")
            print(f"   Text: {doc['text'][:200]}...")
            print(f"   Metadata: {doc['metadata']}")
            print()

        # Method 2 Results
        print(f"\n2. SIMPLE RETRIEVER 2 (Sparse BM25 Retrieval) - Top {top_k} results:")
        print("-" * 60)
        results2 = retriever2.retrieve(query, top_k=top_k)
        for i, doc in enumerate(results2, 1):
            print(f"{i}. Score: {doc['score']:.4f}")
            print(f"   Text: {doc['text'][:200]}...")
            print(f"   Metadata: {doc['metadata']}")
            print()

        # Method 3 Results
        print(f"\n3. HYBRID RETRIEVER (Dense + Sparse + Reranking) - Top {top_k} results:")
        print("-" * 60)
        results3 = retriever3.retrieve_hybrid(query, dense_top_k=5, sparse_top_k=5, rerank_top_k=top_k)
        for i, doc in enumerate(results3, 1):
            rerank_score = doc.get('rerank_score', 'N/A')
            print(f"{i}. Rerank Score: {rerank_score}")
            print(f"   Text: {doc['text'][:200]}...")
            print(f"   Metadata: {doc['metadata']}")
            print()

        print("=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Results per method: {top_k}")
        print()
        print("Method 1 (Dense): Focuses on semantic similarity using embeddings")
        print("Method 2 (Sparse): Focuses on keyword matching using BM25")
        print("Method 3 (Hybrid): Combines both methods and reranks for best results")

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()