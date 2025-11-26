import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

# Add the app directory to the path so we can import modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from retrieval.simple_retriever2 import SimpleRetriever2


class TestSimpleRetriever2(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.bm25_dir = os.path.join(self.temp_dir, "bm25_indexes")
        os.makedirs(self.bm25_dir, exist_ok=True)

    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir)

    @patch('retrieval.simple_retriever2.SimpleRetriever2.load_or_create_bm25_index')
    def test_init_with_existing_index(self, mock_load_index):
        """Test initialization with existing BM25 index."""
        retriever = SimpleRetriever2(
            data_folder=self.temp_dir,
            file_name="test_data.json",
            bm25_index_file=os.path.join(self.bm25_dir, "bm25_index.json")
        )

        self.assertEqual(retriever.data_folder, self.temp_dir)
        self.assertEqual(retriever.file_name, "test_data.json")
        mock_load_index.assert_called_once()

    @patch('retrieval.simple_retriever2.SimpleRetriever2.load_or_create_bm25_index')
    def test_init_create_index(self, mock_load_index):
        """Test initialization that creates new BM25 index."""
        retriever = SimpleRetriever2(
            data_folder=self.temp_dir,
            file_name="test_data.json",
            bm25_index_file=os.path.join(self.bm25_dir, "bm25_index.json")
        )

        self.assertEqual(retriever.data_folder, self.temp_dir)
        self.assertEqual(retriever.file_name, "test_data.json")
        mock_load_index.assert_called_once()

    @patch('retrieval.simple_retriever2.SimpleRetriever2.load_or_create_bm25_index')
    @patch('nltk.tokenize.word_tokenize')
    def test_retrieve(self, mock_tokenize, mock_load_index):
        """Test retrieval functionality."""
        mock_tokenize.side_effect = lambda x: x.lower().split()

        retriever = SimpleRetriever2(bm25_index_file=os.path.join(self.bm25_dir, "bm25_index.json"))
        retriever.corpus = ["This is document one.", "This is document two."]
        retriever.bm25 = MagicMock()
        retriever.bm25.get_scores.return_value = [0.8, 0.3]

        results = retriever.retrieve("test query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("score", results[0])
        self.assertIn("text", results[0])
        self.assertIn("metadata", results[0])
        self.assertEqual(results[0]["text"], "This is document one.")
        self.assertEqual(results[0]["score"], 0.8)

    @patch('rank_bm25.BM25Okapi')
    def test_retrieve_no_bm25(self, mock_bm25):
        """Test retrieval when BM25 is not initialized."""
        retriever = SimpleRetriever2.__new__(SimpleRetriever2)  # Create without __init__
        retriever.bm25 = None

        results = retriever.retrieve("test query")
        self.assertEqual(results, [])

    @patch('retrieval.simple_retriever2.SimpleRetriever2.load_or_create_bm25_index')
    def test_get_index_size(self, mock_load_index):
        """Test getting index size."""
        retriever = SimpleRetriever2(bm25_index_file=os.path.join(self.bm25_dir, "bm25_index.json"))
        retriever.corpus = ["doc1", "doc2", "doc3"]
        size = retriever.get_index_size()
        self.assertEqual(size, 3)


if __name__ == '__main__':
    unittest.main()