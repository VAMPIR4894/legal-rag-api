#!/usr/bin/env python3
"""
Script to create and persist the vector database and BM25 index from JSON legal case data.
This script processes all JSON files in the data/legal_data folder, chunks the text,
generates embeddings for ChromaDB, and creates BM25 index for sparse retrieval.
"""

import os
import json
import sys
import argparse
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Add the app directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from retrieval.embeddings import EmbeddingGenerator

# Default data path (relative to project root)
DATA_FOLDER = os.path.join("data", "legal_data")
DEFAULT_FILE = None  # Process all JSON files by default
VECTORDB_FOLDER = os.path.join("data", "vectordb")
BM25_INDEX_FILE = os.path.join("data", "bm25_indexes", "bm25_index.json")


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Naive character-based chunking with overlap.
    Keeps chunks readable by breaking on nearest space when possible.
    """
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
        # try to break at last space before end for readability
        split_at = text.rfind(' ', start, end)
        if split_at <= start:
            split_at = end
        chunk = text[start:split_at]
        chunks.append(chunk)
        start = split_at - overlap if (split_at - overlap) > start else split_at
    return chunks


def load_json_file(data_folder: str, file_name: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    path = os.path.join(data_folder, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]


def build_vector_database(data_folder: str = DATA_FOLDER,
                         file_names: List[str] = None,
                         collection_name: str = "legal_cases",
                         chunk_size: int = 1000,
                         overlap: int = 200,
                         max_items_per_file: int = None):
    """
    Build and persist the vector database from JSON legal case data.

    Args:
        data_folder: Path to the data folder containing JSON files
        file_names: List of JSON file names to process
        collection_name: Name of the ChromaDB collection to create
        chunk_size: Size of text chunks in characters
        overlap: Overlap between chunks in characters
        max_items_per_file: Maximum number of items to process per file (for testing)
    """
    if file_names is None:
        file_names = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    
    print(f"Building vector database from {file_names}...")

    # Ensure vectordb directory exists and clean it
    if os.path.exists(VECTORDB_FOLDER):
        print(f"Cleaning existing vector database directory: {VECTORDB_FOLDER}")
        for filename in os.listdir(VECTORDB_FOLDER):
            file_path = os.path.join(VECTORDB_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"Removed file: {filename}")
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                    print(f"Removed directory: {filename}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        os.makedirs(VECTORDB_FOLDER, exist_ok=True)

    # Initialize ChromaDB client with persistent storage
    chroma_client = chromadb.PersistentClient(
        path=VECTORDB_FOLDER,
        settings=Settings(anonymized_telemetry=False)
    )

    # Create new collection (since we cleaned the directory, it won't exist)
    collection = chroma_client.create_collection(name=collection_name)
    print(f"Created new collection '{collection_name}'")

    # Load and process data
    embedder = EmbeddingGenerator()

    texts_for_embedding = []
    metadatas = []
    ids = []

    total_items = 0
    for file_name in file_names:
        raw_items = load_json_file(data_folder, file_name)
        if max_items_per_file:
            raw_items = raw_items[:max_items_per_file]
        print(f"Processing {len(raw_items)} items from {file_name}...")

        for i, item in enumerate(raw_items):
            case_title = item.get("case_title") or item.get("title")
            question = item.get("question")
            answer = item.get("answer")
            text = item.get("text") or item.get("content")

            parts = []
            if question and answer:
                parts.append(f"Question: {question.strip()}")
                parts.append(f"Answer: {answer.strip()}")
            elif case_title and text:
                parts.append(f"Case Title: {case_title.strip()}")
                parts.append(f"Text: {text.strip()}")
            elif text:
                parts.append(text.strip())
            elif answer:
                parts.append(answer.strip())
            else:
                continue

            full_text = "\n".join(parts)
            chunks = _chunk_text(full_text, chunk_size, overlap)

            for j, chunk in enumerate(chunks):
                metadata = {
                    "source_file": file_name,
                    "index_in_file": str(i),
                    "chunk_id": str(j),
                }
                if case_title:
                    metadata["case_title"] = case_title[:200]
                if question:
                    metadata["orig_question"] = question[:200]

                chunk_id = f"{file_name}_{i}_{j}"
                texts_for_embedding.append(chunk)
                metadatas.append(metadata)
                ids.append(chunk_id)
        
        total_items += len(raw_items)

    if not texts_for_embedding:
        print("No texts found to embed!")
        return

    print(f"Generating embeddings for {len(texts_for_embedding)} chunks from {total_items} items...")
    embeddings = embedder.generate_embeddings(texts_for_embedding)

    # Add to ChromaDB collection in batches to avoid memory issues
    batch_size = 100
    total_added = 0
    for i in range(0, len(texts_for_embedding), batch_size):
        end_idx = min(i + batch_size, len(texts_for_embedding))
        collection.add(
            embeddings=embeddings[i:end_idx],
            documents=texts_for_embedding[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        total_added += (end_idx - i)
        print(f"Added {total_added}/{len(texts_for_embedding)} chunks to database...")

    print(f"Vector database built successfully with {len(texts_for_embedding)} chunks!")


def build_bm25_index(data_folder: str = DATA_FOLDER,
                    file_names: List[str] = None,
                    chunk_size: int = 1000,
                    overlap: int = 200,
                    max_items_per_file: int = None):
    """
    Build and persist the BM25 index from JSON legal case data.

    Args:
        data_folder: Path to the data folder containing JSON files
        file_names: List of JSON file names to process
        chunk_size: Size of text chunks in characters
        overlap: Overlap between chunks in characters
        max_items_per_file: Maximum number of items to process per file (for testing)
    """
    if file_names is None:
        file_names = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    
    print(f"Building BM25 index from {file_names}...")

    # Ensure bm25_indexes directory exists and clean it
    bm25_dir = os.path.dirname(BM25_INDEX_FILE)
    if os.path.exists(bm25_dir):
        print(f"Cleaning existing BM25 index directory: {bm25_dir}")
        for filename in os.listdir(bm25_dir):
            file_path = os.path.join(bm25_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"Removed file: {filename}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        os.makedirs(bm25_dir, exist_ok=True)

    corpus = []
    case_titles = []

    total_items = 0
    for file_name in file_names:
        path = os.path.join(data_folder, file_name)
        if not os.path.exists(path):
            print(f"Warning: Data file not found: {path}, skipping...")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
        
        if max_items_per_file:
            data = data[:max_items_per_file]
        
        print(f"Processing {len(data)} items from {file_name} for BM25...")
        
        for item in data:
            case_title = item.get("case_title") or item.get("title", "")
            question = item.get("question")
            answer = item.get("answer")
            text = item.get("text") or item.get("content")

            parts = []
            if question and answer:
                parts.append(f"Question: {question.strip()}")
                parts.append(f"Answer: {answer.strip()}")
            elif case_title and text:
                parts.append(f"Case Title: {case_title.strip()}")
                parts.append(f"Text: {text.strip()}")
            elif text:
                parts.append(text.strip())
            elif answer:
                parts.append(answer.strip())
            else:
                continue

            full_text = "\n".join(parts)
            chunks = _chunk_text(full_text, chunk_size, overlap)
            corpus.extend(chunks)
            # Add case title for each chunk
            case_titles.extend([case_title] * len(chunks))
        
        total_items += len(data)

    if not corpus:
        print("No texts found for BM25 indexing!")
        return

    print(f"Tokenizing {len(corpus)} chunks from {total_items} items for BM25...")
    # Tokenize corpus
    nltk.download('punkt', quiet=True)
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Save index
    with open(BM25_INDEX_FILE, 'w') as f:
        json.dump({'corpus': corpus, 'case_titles': case_titles}, f)

    print(f"BM25 index built successfully with {len(corpus)} chunks!")
    print(f"BM25 index saved to {BM25_INDEX_FILE}")


def build_all_indexes(data_folder: str = DATA_FOLDER,
                     file_names: List[str] = None,
                     collection_name: str = "legal_cases",
                     chunk_size: int = 1000,
                     overlap: int = 200,
                     max_items_per_file: int = None):
    """
    Build both vector database and BM25 index.

    Args:
        data_folder: Path to the data folder containing JSON files
        file_names: List of JSON file names to process
        collection_name: Name of the ChromaDB collection to create
        chunk_size: Size of text chunks in characters
        overlap: Overlap between chunks in characters
        max_items_per_file: Maximum number of items to process per file (for testing)
    """
    if file_names is None:
        file_names = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    
    print("=" * 60)
    print("BUILDING VECTOR DATABASE AND BM25 INDEX")
    print(f"Processing files: {file_names}")
    if max_items_per_file:
        print(f"Limited to {max_items_per_file} items per file")
    print("=" * 60)

    # Build vector database first
    build_vector_database(data_folder, file_names, collection_name, chunk_size, overlap, max_items_per_file)

    print()

    # Build BM25 index
    build_bm25_index(data_folder, file_names, chunk_size, overlap, max_items_per_file)

    print()
    print("=" * 60)
    print("ALL INDEXES BUILT SUCCESSFULLY!")
    print("Vector database: data/vectordb/")
    print(f"BM25 index: {BM25_INDEX_FILE}")
    print("=" * 60)
def main():
    """Main function to run the vector database and BM25 index creation script."""
    parser = argparse.ArgumentParser(description="Create vector database and BM25 index from legal case data")
    parser.add_argument("--data-folder", default=DATA_FOLDER,
                       help=f"Path to data folder (default: {DATA_FOLDER})")
    parser.add_argument("--file-names", nargs='*', default=None,
                       help="JSON file names to process (default: all .json files in data folder)")
    parser.add_argument("--collection-name", default="legal_cases",
                       help="ChromaDB collection name (default: legal_cases)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Text chunk size in characters (default: 1000)")
    parser.add_argument("--overlap", type=int, default=200,
                       help="Overlap between chunks in characters (default: 200)")
    parser.add_argument("--max-items-per-file", type=int, default=None,
                       help="Maximum number of items to process per file (for testing, default: all)")
    parser.add_argument("--vectors-only", action="store_true",
                       help="Build only vector database (skip BM25 index)")
    parser.add_argument("--bm25-only", action="store_true",
                       help="Build only BM25 index (skip vector database)")

    args = parser.parse_args()

    # Determine file names to process
    if args.file_names is None:
        if not os.path.exists(args.data_folder):
            print(f"Error: Data folder not found: {args.data_folder}")
            sys.exit(1)
        file_names = [f for f in os.listdir(args.data_folder) if f.endswith('.json')]
        if not file_names:
            print(f"Error: No JSON files found in {args.data_folder}")
            sys.exit(1)
    else:
        file_names = args.file_names

    try:
        if args.vectors_only and args.bm25_only:
            print("Error: Cannot specify both --vectors-only and --bm25-only")
            sys.exit(1)
        elif args.vectors_only:
            build_vector_database(
                data_folder=args.data_folder,
                file_names=file_names,
                collection_name=args.collection_name,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                max_items_per_file=args.max_items_per_file
            )
        elif args.bm25_only:
            build_bm25_index(
                data_folder=args.data_folder,
                file_names=file_names,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                max_items_per_file=args.max_items_per_file
            )
        else:
            # Default: build both
            build_all_indexes(
                data_folder=args.data_folder,
                file_names=file_names,
                collection_name=args.collection_name,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                max_items_per_file=args.max_items_per_file
            )
        print("Index creation completed successfully!")
    except Exception as e:
        print(f"Error creating indexes: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()