import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

# Adjust path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.embeddings import EmbeddingGenerator

# Default data path (relative to project root)
DATA_FOLDER = os.path.join("data", "legal_data")
DEFAULT_FILE = "filtered_civil_case.json"
VECTORDB_FOLDER = os.path.join("data", "vectordb")


class SimpleRetriever:
    """A retriever that uses ChromaDB for persistent vector storage:
    - Loads existing ChromaDB collection with pre-computed embeddings
    - Retrieves top-k chunks by cosine similarity
    - Does NOT create or build the vector database (use scripts/create_vectordb.py for that)
    """

    def __init__(self, embedder: Optional[EmbeddingGenerator] = None,
                 data_folder: str = DATA_FOLDER, file_name: str = DEFAULT_FILE,
                 chunk_size: int = 1000, overlap: int = 200,
                 collection_name: str = "legal_cases",
                 score_scale_factor: float = 20.0):
        self.data_folder = data_folder
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.collection_name = collection_name
        self.score_scale_factor = score_scale_factor  # Scale factor to make dense scores competitive
        self.embedder = embedder or EmbeddingGenerator()

        # Ensure vectordb directory exists
        os.makedirs(VECTORDB_FOLDER, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=VECTORDB_FOLDER,
            settings=Settings(anonymized_telemetry=False)
        )

        # Load existing collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"SimpleRetriever: Loaded existing collection '{collection_name}' with {self.collection.count()} documents")
        except ValueError:
            raise ValueError(f"Collection '{collection_name}' does not exist. Please run 'python scripts/create_vectordb.py' to create the vector database first.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []

        # Expand query for better retrieval accuracy
        expanded_query = self.embedder.expand_query(query)
        print(f"Original query: '{query}'")
        print(f"Expanded query: '{expanded_query}'")

        query_emb = self.embedder.generate_embeddings([expanded_query])[0]

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        # Convert ChromaDB results to expected format
        retrieved = []
        if results['documents'] and results['metadatas'] and results['distances']:
            for doc, meta, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # ChromaDB returns cosine distance, convert to similarity and scale
                similarity = 1.0 - distance
                scaled_score = similarity * self.score_scale_factor
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "score": float(scaled_score)
                })

        return retrieved

    def get_index_size(self) -> int:
        return self.collection.count()
