import os
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import json
import nltk
from nltk.tokenize import word_tokenize

# Default data path (relative to project root)
DATA_FOLDER = os.path.join("data", "legal_data")
DEFAULT_FILE = "filtered_civil_case.json"
BM25_INDEX_FILE = os.path.join("data", "bm25_indexes", "bm25_index.json")

class SimpleRetriever2:
    """
    A sparse retriever that uses BM25 for keyword-based retrieval.
    Implements method2: sparse retrieval.
    """

    def __init__(self, data_folder: str = DATA_FOLDER, file_name: str = DEFAULT_FILE,
                 chunk_size: int = 1000, overlap: int = 200,
                 bm25_index_file: str = BM25_INDEX_FILE):
        self.data_folder = data_folder
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.bm25_index_file = bm25_index_file

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
        Retrieve top-k documents using BM25 sparse retrieval.

        Args:
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            List of dictionaries with 'text', 'metadata', and 'score'.
        """
        if not self.bm25:
            return []

        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            case_title = self.case_titles[idx] if idx < len(self.case_titles) else ""
            results.append({
                "text": self.corpus[idx],
                "metadata": {
                    "case_title": case_title,
                    "chunk_id": str(idx),
                    "source": "bm25"
                },
                "score": float(bm25_scores[idx])
            })

        return results

    def get_index_size(self) -> int:
        """Return the number of documents in the BM25 index."""
        return len(self.corpus)