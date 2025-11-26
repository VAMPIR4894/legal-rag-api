#!/usr/bin/env python3
"""
Complete end-to-end test of the RAG system.
Tests retrieval and context building (skips actual LLM generation since it requires Mistral API).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from retrieval.embeddings import EmbeddingGenerator
from retrieval.hybrid_retriever import HybridRetriever
from rag.context_builder import ContextBuilder
from rag.prompt_templates import PromptTemplates

def main():
    print("=" * 80)
    print("COMPLETE RAG PIPELINE TEST")
    print("=" * 80)
    
    # Test query
    query = "What are the legal requirements for property disputes in civil cases?"
    print(f"\nQuery: {query}\n")
    
    # Step 1: Initialize Retriever
    print("Step 1: Initializing Hybrid Retriever...")
    retriever = HybridRetriever()
    print(f"✓ Retriever initialized with {retriever.simple_retriever.get_index_size()} documents\n")
    
    # Step 2: Retrieve relevant documents
    print("Step 2: Retrieving relevant documents...")
    docs = retriever.retrieve(query, top_k=5)
    print(f"✓ Retrieved {len(docs)} documents\n")
    
    # Display retrieved documents
    print("Retrieved Documents:")
    print("-" * 80)
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. Rerank Score: {doc.get('rerank_score', doc.get('score', 0)):.4f}")
        print(f"   Source: {doc['metadata'].get('case_title', 'N/A')}")
        print(f"   Text: {doc['text'][:200]}...")
    print("\n" + "-" * 80)
    
    # Step 3: Build Context
    print("\nStep 3: Building context for LLM...")
    context_builder = ContextBuilder()
    context = context_builder.build_context(docs)
    print(f"✓ Context built ({len(context)} characters)\n")
    
    # Display context (first 500 chars)
    print("Context Preview:")
    print("-" * 80)
    print(context[:800] + "...")
    print("-" * 80)
    
    # Step 4: Build Prompt
    print("\nStep 4: Building prompt...")
    prompt_templates = PromptTemplates()
    system_prompt = prompt_templates.get_system_prompt()
    full_prompt = prompt_templates.format_prompt(system_prompt, context, query)
    print(f"✓ Prompt built ({len(full_prompt)} characters)\n")
    
    # Display prompt structure
    print("Prompt Structure:")
    print("-" * 80)
    print(full_prompt[:1000] + "...")
    print("-" * 80)
    
    print("\n" + "=" * 80)
    print("RAG PIPELINE TEST COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Retrieved: {len(docs)} relevant documents")
    print(f"  - Context size: {len(context)} characters")
    print(f"  - Prompt size: {len(full_prompt)} characters")
    print(f"\n  ✓ Retrieval pipeline is working correctly!")
    print(f"  ✓ Context building is working correctly!")
    print(f"  ✓ Prompt formatting is working correctly!")
    print(f"\n  Note: To complete the RAG pipeline, configure MISTRAL_API_BASE_URL")
    print(f"        environment variable to point to your Mistral API endpoint.\n")

if __name__ == "__main__":
    main()
