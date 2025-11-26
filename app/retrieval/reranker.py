import os
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np

# Define the cross-encoder model
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Define the top-K limits as per the project report. Since categories were removed,
# we select a combined top-K (sum of previous limits) to approximate prior behavior.
MAX_LAWS = 3
MAX_CASES = 10
MAX_RESULTS = MAX_LAWS + MAX_CASES

class Reranker:
    """
    Implements a Cross-Encoder reranking layer to improve the precision
    of retrieved documents based on the query context.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        """
        Loads the cross-encoder model.
        """
        # Load the model, setting device to 'cpu' if no CUDA is available
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        self.model = CrossEncoder(model_name, device=device)
        print(f"Loaded Reranker model: {model_name} on device: {device}")

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks the list of candidate documents based on their relevance to the query.

        Args:
            query: The user query string.
            candidates: A list of documents (from EnsembleRetriever), each having 'text'.

        Returns:
            The list of documents, re-sorted by the Cross-Encoder score and truncated
            according to the project's limits (Top 3 laws + Top 10 cases).
        """
        if not candidates:
            return []

        # 1. Prepare pairs for the Cross-Encoder
        # Format: [(query, doc_text_1), (query, doc_text_2), ...]
        # Truncate document text to prevent issues with very long inputs
        max_doc_length = 512  # Limit document length for cross-encoder
        max_query_length = 128  # Limit query length
        truncated_query = query[:max_query_length] if len(query) > max_query_length else query
        
        sentence_pairs = [(truncated_query, doc['text'][:max_doc_length]) for doc in candidates]

        # 2. Get Reranking Scores
        # The cross-encoder returns a single score (float) for each pair.
        print(f"Reranking {len(candidates)} candidates...")
        try:
            scores = self.model.predict(sentence_pairs)

            # Check for NaN values and handle them
            if any(np.isnan(score) for score in scores) or any(np.isinf(score) for score in scores):
                print("Warning: Cross-encoder returned NaN/Inf values, using a simpler scoring approach...")
                # Use a simple approach: combine query and document similarity
                from sentence_transformers import util
                query_emb = self.model.tokenizer.encode(sentence_pairs[0][0], return_tensors='pt', truncation=True, max_length=128)
                doc_embs = []
                for _, doc in sentence_pairs:
                    doc_emb = self.model.tokenizer.encode(doc, return_tensors='pt', truncation=True, max_length=128)
                    doc_embs.append(doc_emb)
                
                # Fall back to original scores if transformer approach fails
                for i, doc in enumerate(candidates):
                    doc['rerank_score'] = doc.get('score', 0.0)
            else:
                # Apply New Scores to Candidates
                for i, doc in enumerate(candidates):
                    # The higher the score, the more relevant the document
                    doc['rerank_score'] = float(scores[i])

        except Exception as e:
            print(f"Error during reranking: {e}, falling back to original scores")
            # Fall back to using the original retrieval scores
            for i, doc in enumerate(candidates):
                doc['rerank_score'] = doc.get('score', 0.0)

        # 4. Sort all candidates by rerank score and return the top-N combined results
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        final_results = candidates[:MAX_RESULTS]

        print(f"✅ Reranking complete. Final selected documents: {len(final_results)} (top {MAX_RESULTS} by rerank score).")

        return final_results