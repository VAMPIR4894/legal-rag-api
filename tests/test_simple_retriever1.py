import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the app directory to the path so we can import modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from retrieval.simple_retriever1 import SimpleRetriever


class TestSimpleRetriever(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.vectordb_dir = os.path.join(self.temp_dir, "vectordb")
        os.makedirs(self.vectordb_dir, exist_ok=True)

    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir)

    @patch('retrieval.simple_retriever1.chromadb.PersistentClient')
    def test_init_with_existing_collection(self, mock_chroma_client):
        """Test initialization with existing collection."""
        # Mock the ChromaDB client and collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_chroma_client.return_value.get_collection.return_value = mock_collection

        retriever = SimpleRetriever(
            data_folder=self.temp_dir,
            file_name="test_data.json",
            collection_name="test_collection"
        )

        self.assertEqual(retriever.data_folder, self.temp_dir)
        self.assertEqual(retriever.file_name, "test_data.json")
        mock_chroma_client.return_value.get_collection.assert_called_with(name="test_collection")

    @patch('retrieval.simple_retriever1.chromadb.PersistentClient')
    def test_init_with_nonexistent_collection(self, mock_chroma_client):
        """Test initialization fails when collection doesn't exist."""
        # Mock the ChromaDB client to raise ValueError (collection not found)
        mock_chroma_client.return_value.get_collection.side_effect = ValueError("Collection not found")

        with self.assertRaises(ValueError) as context:
            SimpleRetriever(
                data_folder=self.temp_dir,
                file_name="test_data.json",
                collection_name="nonexistent_collection"
            )

        self.assertIn("does not exist", str(context.exception))
        self.assertIn("create_vectordb.py", str(context.exception))

    @patch('retrieval.simple_retriever1.chromadb.PersistentClient')
    def test_retrieve(self, mock_chroma_client):
        """Test retrieval functionality."""
        # Mock the ChromaDB client and collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10

        # Mock query results
        mock_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        mock_collection.query.return_value = mock_results

        mock_chroma_client.return_value.get_collection.return_value = mock_collection

        retriever = SimpleRetriever(collection_name="test_collection")

        results = retriever.retrieve("test query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("score", results[0])
        self.assertIn("text", results[0])
        self.assertIn("metadata", results[0])

    @patch('retrieval.simple_retriever1.chromadb.PersistentClient')
    def test_retrieve_empty_index(self, mock_chroma_client):
        """Test retrieval when index is empty."""
        # Mock the ChromaDB client and empty collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_chroma_client.return_value.get_collection.return_value = mock_collection

        retriever = SimpleRetriever(collection_name="empty_collection")
        results = retriever.retrieve("test query")
        self.assertEqual(results, [])

    @patch('retrieval.simple_retriever1.chromadb.PersistentClient')
    def test_get_index_size(self, mock_chroma_client):
        """Test getting index size."""
        # Mock the ChromaDB client and collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42

        mock_chroma_client.return_value.get_collection.return_value = mock_collection

        retriever = SimpleRetriever(collection_name="test_collection")
        size = retriever.get_index_size()
        self.assertEqual(size, 42)


if __name__ == '__main__':
    unittest.main()