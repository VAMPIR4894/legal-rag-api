import os
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import json
import nltk
from nltk.tokenize import word_tokenize

# Adjust path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.simple_retriever1 import SimpleRetriever
from retrieval.reranker import Reranker
from retrieval.embeddings import EmbeddingGenerator

# Default data path (relative to project root)
DATA_FOLDER = os.path.join("data", "legal_data")
DEFAULT_FILE = "filtered_civil_case.json"
BM25_INDEX_FILE = os.path.join("data", "bm25_indexes", "bm25_index.json")

class HybridRetriever:
    """
    A hybrid retriever that combines dense retrieval (from ChromaDB) and sparse retrieval (BM25),
    then reranks the combined results using a cross-encoder.
    """

    def __init__(self, embedder: Optional[EmbeddingGenerator] = None,
                 data_folder: str = DATA_FOLDER, file_name: str = DEFAULT_FILE,
                 chunk_size: int = 1000, overlap: int = 200,
                 collection_name: str = "legal_cases",
                 bm25_index_file: str = BM25_INDEX_FILE,
                 score_scale_factor: float = 20.0):
        self.data_folder = data_folder
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.collection_name = collection_name
        self.bm25_index_file = bm25_index_file
        self.score_scale_factor = score_scale_factor

        # Initialize components
        self.embedder = embedder or EmbeddingGenerator()
        self.simple_retriever = SimpleRetriever(self.embedder, data_folder, file_name, chunk_size, overlap, collection_name, score_scale_factor)
        self.reranker = Reranker()

        # Initialize BM25
        self.bm25 = None
        self.corpus = []
        self.case_titles = []  # Store case title for each chunk
        self.load_or_create_bm25_index()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Naive character-based chunking with overlap."""
        if not text:
            return []
        text = text.strip()
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = start + chunk_size
            if end >= length:
                chunk = text[start:]
                chunks.append(chunk)
                break
            split_at = text.rfind(' ', start, end)
            if split_at <= start:
                split_at = end
            chunk = text[start:split_at]
            chunks.append(chunk)
            start = split_at - overlap if (split_at - overlap) > start else split_at
        return chunks

    def load_or_create_bm25_index(self):
        """Load BM25 index from file. Index must exist - creation is handled by create_vectordb.py"""
        if os.path.exists(self.bm25_index_file):
            print(f"Loading BM25 index from {self.bm25_index_file}")
            with open(self.bm25_index_file, 'r') as f:
                data = json.load(f)
                self.corpus = data['corpus']
                self.case_titles = data.get('case_titles', [])  # Load case titles if available
                # Reconstruct BM25
                tokenized_corpus = [doc.split() for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            raise FileNotFoundError(f"BM25 index not found at {self.bm25_index_file}. Please run 'python scripts/create_vectordb.py' first.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval: combine dense and sparse results, then rerank.

        Args:
            query: The search query.
            top_k: Number of final results after reranking.

        Returns:
            List of reranked documents.
        """
        # Dense retrieval
        dense_results = self.simple_retriever.retrieve(query, top_k=top_k*2)

        # Sparse retrieval
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k*2]

        sparse_results = []
        for idx in top_sparse_indices:
            case_title = self.case_titles[idx] if idx < len(self.case_titles) else ""
            sparse_results.append({
                "text": self.corpus[idx],
                "metadata": {
                    "case_title": case_title,
                    "chunk_id": str(idx),
                    "source": "bm25"
                },
                "score": float(bm25_scores[idx])
            })

        # Normalize scores for both dense and sparse results
        dense_results = self._normalize_scores(dense_results, 'score')
        sparse_results = self._normalize_scores(sparse_results, 'score')
        
        # Update the score to be the normalized score for reranking
        for doc in dense_results:
            doc['score'] = doc['normalized_score']
        for doc in sparse_results:
            doc['score'] = doc['normalized_score']

        # Combine and deduplicate
        combined = dense_results + sparse_results
        seen_texts = set()
        unique_combined = []
        for doc in combined:
            if doc['text'] not in seen_texts:
                seen_texts.add(doc['text'])
                unique_combined.append(doc)

        # Rerank
        reranked = self.reranker.rerank(query, unique_combined)

        return reranked[:top_k]

    def get_index_size(self) -> int:
        return self.simple_retriever.get_index_size()

    def _normalize_scores(self, results: List[Dict[str, Any]], score_key: str = 'score') -> List[Dict[str, Any]]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            results: List of result dictionaries with scores
            score_key: Key to access the score in each result dict
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
            
        scores = [doc.get(score_key, 0.0) for doc in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            for doc in results:
                doc[f'normalized_{score_key}'] = 1.0 if scores else 0.0
        else:
            for doc in results:
                normalized = (doc.get(score_key, 0.0) - min_score) / (max_score - min_score)
                doc[f'normalized_{score_key}'] = normalized
                
        return results