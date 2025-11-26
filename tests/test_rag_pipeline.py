import unittest
from unittest.mock import patch, MagicMock

# Add the app directory to the path so we can import modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from rag.pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Mock all the components to avoid actual dependencies
        self.mock_embedder = MagicMock()
        self.mock_retriever = MagicMock()
        self.mock_model_loader = MagicMock()
        self.mock_inference_engine = MagicMock()
        self.mock_context_builder = MagicMock()

    @patch('rag.pipeline.EmbeddingGenerator')
    @patch('rag.pipeline.SimpleRetriever')
    @patch('rag.pipeline.SimpleRetriever2')
    @patch('rag.pipeline.HybridRetriever')
    @patch('rag.pipeline.ModelLoader')
    @patch('rag.pipeline.InferenceEngine')
    @patch('rag.pipeline.ContextBuilder')
    def test_init(self, mock_context_builder, mock_inference_engine,
                  mock_model_loader, mock_hybrid_retriever, mock_simple_retriever2,
                  mock_simple_retriever, mock_embedding_generator):
        """Test pipeline initialization."""
        mock_embedding_generator.return_value = self.mock_embedder
        mock_simple_retriever.return_value = self.mock_retriever
        mock_simple_retriever2.return_value = MagicMock()
        mock_hybrid_retriever.return_value = MagicMock()
        mock_model_loader.return_value = self.mock_model_loader
        mock_inference_engine.return_value = self.mock_inference_engine
        mock_context_builder.return_value = self.mock_context_builder

        # Mock the model loader return values
        self.mock_model_loader.get_model_and_tokenizer.return_value = ('test-model', 'http://test:11434')

        with patch.dict(os.environ, {'RETRIEVAL_METHOD': 'method1', 'RETRIEVAL_RESULTS': '5'}):
            pipeline = RAGPipeline()

        self.assertEqual(pipeline.embedder, self.mock_embedder)
        self.assertEqual(pipeline.simple_retriever1, self.mock_retriever)
        self.assertEqual(pipeline.model_loader, self.mock_model_loader)
        self.assertEqual(pipeline.inference_engine, self.mock_inference_engine)
        self.assertEqual(pipeline.context_builder, self.mock_context_builder)
        self.assertEqual(pipeline.retrieval_method, 'method1')
        self.assertEqual(pipeline.retrieval_results, 5)

    @patch('rag.pipeline.EmbeddingGenerator')
    @patch('rag.pipeline.SimpleRetriever')
    @patch('rag.pipeline.ModelLoader')
    @patch('rag.pipeline.InferenceEngine')
    @patch('rag.pipeline.ContextBuilder')
    def test_run_retrieval_and_rerank(self, mock_context_builder, mock_inference_engine,
                                       mock_model_loader, mock_simple_retriever, mock_embedding_generator):
        """Test retrieval and reranking step."""
        mock_embedding_generator.return_value = self.mock_embedder
        mock_simple_retriever.return_value = self.mock_retriever
        mock_model_loader.return_value = self.mock_model_loader
        mock_inference_engine.return_value = self.mock_inference_engine
        mock_context_builder.return_value = self.mock_context_builder

        self.mock_model_loader.get_model_and_tokenizer.return_value = ('test-model', 'http://test:11434')

        pipeline = RAGPipeline()

        # Mock retriever results
        mock_results = [
            {'text': 'chunk1', 'metadata': {'source': 'test'}, 'score': 0.9},
            {'text': 'chunk2', 'metadata': {'source': 'test'}, 'score': 0.8}
        ]
        self.mock_retriever.retrieve.return_value = mock_results

        results = pipeline.run_retrieval_and_rerank("test query")

        self.assertEqual(results, mock_results)
        self.mock_retriever.retrieve.assert_called_once_with(query="test query", top_k=5)

    @patch('rag.pipeline.EmbeddingGenerator')
    @patch('rag.pipeline.SimpleRetriever')
    @patch('rag.pipeline.ModelLoader')
    @patch('rag.pipeline.InferenceEngine')
    @patch('rag.pipeline.ContextBuilder')
    @patch('rag.pipeline.PromptTemplates')
    def test_run_rag_pipeline_success(self, mock_prompt_templates, mock_context_builder,
                                      mock_inference_engine, mock_model_loader,
                                      mock_simple_retriever, mock_embedding_generator):
        """Test full RAG pipeline execution."""
        mock_embedding_generator.return_value = self.mock_embedder
        mock_simple_retriever.return_value = self.mock_retriever
        mock_model_loader.return_value = self.mock_model_loader
        mock_inference_engine.return_value = self.mock_inference_engine
        mock_context_builder.return_value = self.mock_context_builder

        self.mock_model_loader.get_model_and_tokenizer.return_value = ('test-model', 'http://test:11434')

        pipeline = RAGPipeline()

        # Mock all the components
        mock_docs = [{'text': 'test doc', 'metadata': {}, 'score': 0.9}]
        self.mock_retriever.retrieve.return_value = mock_docs
        self.mock_context_builder.build_context.return_value = "test context"
        mock_prompt_templates.get_system_prompt.return_value = "system prompt"
        mock_prompt_templates.format_prompt.return_value = "formatted prompt"
        self.mock_inference_engine.generate_response.return_value = "generated response"

        response, docs = pipeline.run_rag_pipeline("test query")

        self.assertEqual(response, "generated response")
        self.assertEqual(docs, mock_docs)

        # Verify all methods were called
        self.mock_retriever.retrieve.assert_called_once_with(query="test query", top_k=5)
        self.mock_context_builder.build_context.assert_called_once_with(mock_docs)
        mock_prompt_templates.get_system_prompt.assert_called_once()
        mock_prompt_templates.format_prompt.assert_called_once_with("system prompt", "test context", "test query")
        self.mock_inference_engine.generate_response.assert_called_once_with(
            prompt="formatted prompt",
            max_tokens=2048,
            temperature=0.1,
            stream=False
        )

    @patch('rag.pipeline.EmbeddingGenerator')
    @patch('rag.pipeline.SimpleRetriever')
    @patch('rag.pipeline.ModelLoader')
    @patch('rag.pipeline.InferenceEngine')
    @patch('rag.pipeline.ContextBuilder')
    def test_run_rag_pipeline_no_docs(self, mock_context_builder, mock_inference_engine,
                                       mock_model_loader, mock_simple_retriever, mock_embedding_generator):
        """Test RAG pipeline when no documents are found."""
        mock_embedding_generator.return_value = self.mock_embedder
        mock_simple_retriever.return_value = self.mock_retriever
        mock_model_loader.return_value = self.mock_model_loader
        mock_inference_engine.return_value = self.mock_inference_engine
        mock_context_builder.return_value = self.mock_context_builder

        self.mock_model_loader.get_model_and_tokenizer.return_value = ('test-model', 'http://test:11434')

        pipeline = RAGPipeline()

        # Mock empty results
        self.mock_retriever.retrieve.return_value = []

        response, docs = pipeline.run_rag_pipeline("test query")

        self.assertEqual(response, "No relevant documents found. Cannot answer the query.")
        self.assertEqual(docs, [])


if __name__ == '__main__':
    unittest.main()