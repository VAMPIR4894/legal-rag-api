import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the app directory to the path so we can import modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from retrieval.hybrid_retriever import HybridRetriever


class TestHybridRetriever(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.bm25_dir = os.path.join(self.temp_dir, "bm25_indexes")
        os.makedirs(self.bm25_dir, exist_ok=True)

    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir)

    @patch('retrieval.hybrid_retriever.Reranker')
    @patch('retrieval.hybrid_retriever.SimpleRetriever')
    @patch('retrieval.hybrid_retriever.EmbeddingGenerator')
    def test_init(self, mock_embedder, mock_simple1, mock_reranker):
        """Test initialization of HybridRetriever."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance

        mock_simple1_instance = MagicMock()
        mock_simple1.return_value = mock_simple1_instance

        mock_reranker_instance = MagicMock()
        mock_reranker.return_value = mock_reranker_instance

        with patch.object(HybridRetriever, 'load_or_create_bm25_index'):
            retriever = HybridRetriever(
                data_folder=self.temp_dir,
                file_name="test_data.json",
                bm25_index_file=os.path.join(self.bm25_dir, "bm25_index.json")
            )

            self.assertEqual(retriever.data_folder, self.temp_dir)
            self.assertEqual(retriever.file_name, "test_data.json")
            mock_embedder.assert_called_once()
            mock_simple1.assert_called_once()
            mock_reranker.assert_called_once()

    @patch('retrieval.hybrid_retriever.Reranker')
    @patch('retrieval.hybrid_retriever.SimpleRetriever')
    @patch('retrieval.hybrid_retriever.EmbeddingGenerator')
    def test_retrieve_hybrid(self, mock_embedder, mock_simple1, mock_reranker):
        """Test hybrid retrieval functionality."""
        # Mock components
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance

        mock_simple1_instance = MagicMock()
        mock_simple1_instance.retrieve.return_value = [
            {"text": "Dense doc 1", "metadata": {"source": "dense"}, "score": 0.9},
            {"text": "Dense doc 2", "metadata": {"source": "dense"}, "score": 0.8}
        ]
        mock_simple1.return_value = mock_simple1_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [
            {"text": "Dense doc 1", "metadata": {"source": "dense"}, "score": 0.9, "rerank_score": 0.95},
            {"text": "Sparse doc 1", "metadata": {"source": "sparse"}, "score": 0.7, "rerank_score": 0.90}
        ]
        mock_reranker.return_value = mock_reranker_instance

        with patch.object(HybridRetriever, 'load_or_create_bm25_index'):
            retriever = HybridRetriever(bm25_index_file=os.path.join(self.bm25_dir, "bm25_index.json"))
            retriever.corpus = ["Sparse doc 1", "Sparse doc 2"]
            retriever.bm25 = MagicMock()
            retriever.bm25.get_scores.return_value = [0.7, 0.4]

            results = retriever.retrieve_hybrid("test query", dense_top_k=2, sparse_top_k=2, rerank_top_k=2)

            self.assertEqual(len(results), 2)
            mock_simple1_instance.retrieve.assert_called_once_with("test query", 2)
            mock_reranker_instance.rerank.assert_called_once()
            self.assertIn("rerank_score", results[0])

    @patch('retrieval.hybrid_retriever.Reranker')
    @patch('retrieval.hybrid_retriever.SimpleRetriever')
    @patch('retrieval.hybrid_retriever.EmbeddingGenerator')
    def test_retrieve_hybrid_deduplication(self, mock_embedder, mock_simple1, mock_reranker):
        """Test that duplicate documents are removed."""
        # Mock components
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance

        mock_simple1_instance = MagicMock()
        mock_simple1_instance.retrieve.return_value = [
            {"text": "Same doc", "metadata": {"source": "dense"}, "score": 0.9}
        ]
        mock_simple1.return_value = mock_simple1_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [
            {"text": "Same doc", "metadata": {"source": "dense"}, "score": 0.9, "rerank_score": 0.95}
        ]
        mock_reranker.return_value = mock_reranker_instance

        with patch.object(HybridRetriever, 'load_or_create_bm25_index'):
            retriever = HybridRetriever(bm25_index_file=os.path.join(self.bm25_dir, "bm25_index.json"))
            retriever.corpus = ["Same doc", "Other doc"]
            retriever.bm25 = MagicMock()
            retriever.bm25.get_scores.return_value = [0.7, 0.3]

            results = retriever.retrieve_hybrid("test query", dense_top_k=1, sparse_top_k=1, rerank_top_k=1)

            self.assertEqual(len(results), 1)
            # Should only call rerank once since duplicates were removed
            mock_reranker_instance.rerank.assert_called_once()

    @patch('retrieval.hybrid_retriever.Reranker')
    @patch('retrieval.hybrid_retriever.SimpleRetriever')
    @patch('retrieval.hybrid_retriever.EmbeddingGenerator')
    def test_get_index_size(self, mock_embedder, mock_simple1, mock_reranker):
        """Test getting index size."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance

        mock_simple1_instance = MagicMock()
        mock_simple1_instance.get_index_size.return_value = 100
        mock_simple1.return_value = mock_simple1_instance

        mock_reranker_instance = MagicMock()
        mock_reranker.return_value = mock_reranker_instance

        with patch.object(HybridRetriever, 'load_or_create_bm25_index'):
            retriever = HybridRetriever(bm25_index_file=os.path.join(self.bm25_dir, "bm25_index.json"))
            size = retriever.get_index_size()
            self.assertEqual(size, 100)
            mock_simple1_instance.get_index_size.assert_called_once()


if __name__ == '__main__':
    unittest.main()