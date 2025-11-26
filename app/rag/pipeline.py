import os
import torch
from typing import List, Dict, Any, Tuple

# Adjust path for imports
import sys
# Assuming 'pipeline.py' is in 'app/rag/' and code is run from project root or similar.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app')) 

from retrieval.embeddings import EmbeddingGenerator
from retrieval.simple_retriever1 import SimpleRetriever
from retrieval.simple_retriever2 import SimpleRetriever2
from retrieval.hybrid_retriever import HybridRetriever

# Phase 4 Imports
from models.model_loader import ModelLoader
from models.inference import InferenceEngine
from rag.context_builder import ContextBuilder
from rag.prompt_templates import PromptTemplates

class RAGPipeline:
    """
    The main orchestration layer for the Retrieval-Augmented Generation system.
    Handles retrieval, reranking, context building, and inference.
    """
    def __init__(self):
        # Read retrieval configuration from environment
        self.retrieval_method = os.getenv('RETRIEVAL_METHOD', 'method1')
        self.retrieval_results = int(os.getenv('RETRIEVAL_RESULTS', '5'))
        
        # Validate retrieval method
        if self.retrieval_method not in ['method1', 'method2', 'hybrid']:
            raise ValueError(f"Invalid RETRIEVAL_METHOD: {self.retrieval_method}. Must be 'method1', 'method2', or 'hybrid'")
        
        # 1. Retrieval Components
        self.embedder = EmbeddingGenerator()
        
        # Initialize all retrievers
        self.simple_retriever1 = SimpleRetriever(self.embedder)  # Dense retrieval
        self.simple_retriever2 = SimpleRetriever2()  # Sparse retrieval
        self.hybrid_retriever = HybridRetriever()  # Hybrid retrieval
        
        # Select the active retriever based on configuration
        self.active_retriever = self._get_active_retriever()

        # 3. LLM Components (Phase 4)   
        self.model_loader = ModelLoader()
        # The ModelLoader now returns model_name and api_url (as model/tokenizer objects)
        model_name, api_url = self.model_loader.get_model_and_tokenizer() 
        self.inference_engine = InferenceEngine(model_name, api_url) # Pass strings to InferenceEngine
        self.context_builder = ContextBuilder()
        
    def _get_active_retriever(self):
        """Get the active retriever based on the configured retrieval method."""
        if self.retrieval_method == 'method1':
            return self.simple_retriever1
        elif self.retrieval_method == 'method2':
            return self.simple_retriever2
        elif self.retrieval_method == 'hybrid':
            return self.hybrid_retriever
        else:
            raise ValueError(f"Unknown retrieval method: {self.retrieval_method}")
        
    def run_retrieval_and_rerank(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes the full retrieval and reranking process (Phase 2 & 3).
        
        Args:
            query: The user query string.
            
        Returns:
            A list of the final, highly-relevant documents (laws and cases), 
            sorted by the reranking score.
        """
        print(f"\n--- Running Retrieval for Query: {query[:50]}... ---")
        print(f"Using retrieval method: {self.retrieval_method}, retrieving {self.retrieval_results} results")
        
        # Retrieve using the configured method and number of results
        results = self.active_retriever.retrieve(query=query, top_k=self.retrieval_results)
        
        # Ensure all documents have the required fields for the schema
        for doc in results:
            if 'rerank_score' not in doc:
                doc['rerank_score'] = doc.get('score', 0.0)
            if 'source_db' not in doc:
                doc['source_db'] = doc['metadata'].get('source_file') or doc['metadata'].get('source') or 'unknown'
        
        print(f"Retrieved {len(results)} chunks using {self.retrieval_method}.")
        return results

    def run_rag_pipeline(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes the full RAG workflow: Retrieve, Rerank, Build Context, and Infer (Phase 4).
        
        Args:
            query: The user query string.
            
        Returns: 
            (Generated Text, Final Documents with Source IDs)
        """
        # 1. Retrieve & Rerank (Phase 2 & 3)
        final_docs = self.run_retrieval_and_rerank(query)
        
        if not final_docs:
            return "No relevant documents found. Cannot answer the query.", []

        # 2. Build Context (Phase 4)
        # This step attaches 'source_id' to each document.
        context_string = self.context_builder.build_context(final_docs)
        print(f"\nContext built from {len(final_docs)} documents, ready for LLM.")

        # 3. Format Prompt (Phase 4)
        system_prompt = PromptTemplates.get_system_prompt()
        full_prompt = PromptTemplates.format_prompt(system_prompt, context_string, query)
        
        # 4. Generate Response (Phase 4)
        print("Generating response from LLM...")
        generated_text = self.inference_engine.generate_response(
            prompt=full_prompt,
            max_tokens=2048,
            temperature=0.3,  # Increased from 0.1 to be less conservative
            stream=False # Streaming is typically disabled for API/batch inference
        )
        
        # Return the raw LLM output and the list of documents used (now including source_id)
        return generated_text, final_docs
        
    # --- THIS IS THE NEWLY ADDED FUNCTION ---
    def run_generation_with_context(self, query: str, pdf_context: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes a context-only workflow: Builds context from provided text and Infer.
        This bypasses all retrieval.
        
        Args:
            query: The user query string.
            pdf_context: The text from the uploaded PDF.
            
        Returns: 
            (Generated Text, Final Documents with Source IDs)
        """
        # 1. Skip Retrieval (we have the context)
        print("Bypassing retrieval, using provided PDF context.")

        # 2. Build Context (it's just the provided text)
        # The pdf_context *is* the context_string.
        context_string = pdf_context 

        # 3. Format Prompt
        # We use the *same* prompt template as the main pipeline
        system_prompt = PromptTemplates.get_system_prompt()
        full_prompt = PromptTemplates.format_prompt(system_prompt, context_string, query)
        
        # 4. Generate Response
        print("Generating response from LLM using provided context...")
        generated_text = self.inference_engine.generate_response(
            prompt=full_prompt,
            max_tokens=2048,
            temperature=0.3,
            stream=False
        )
        
        # 5. Create a "fake" source doc to represent the PDF
        final_docs = [
            {
                "source_id": "S1",
                "text": "Context was provided by the user from an uploaded PDF.",
                "metadata": {
                    "source": "Uploaded PDF",
                    "chapter": None,
                    "year": None,
                    "court": None
                },
                "rerank_score": 1.0,
                "source_db": "user_upload"
            }
        ]

        return generated_text, final_docs
    # --- END OF NEW FUNCTION ---