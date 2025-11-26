import unittest
import json
from unittest.mock import patch, MagicMock

# Add the app directory to the path so we can import modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

# Mock auth before importing anything that uses it
def mock_login_required(f):
    return f

with patch('app.api.auth.auth.login_required', mock_login_required):
    from main import create_app
from api.schemas import QueryRequest, QueryResponse


class TestAPIServer(unittest.TestCase):

    def setUp(self):
        # Mock all the heavy components before creating the app
        def mock_login_required(f):
            return f
        
        with patch('app.api.auth.auth.login_required', mock_login_required), \
             patch('api.routes.RAGPipeline'), \
             patch('api.routes.ResponseParser'), \
             patch('rag.pipeline.RAGPipeline'), \
             patch('rag.response_parser.ResponseParser'), \
             patch('models.model_loader.ModelLoader'), \
             patch('models.inference.InferenceEngine'), \
             patch('rag.context_builder.ContextBuilder'), \
             patch('retrieval.simple_retriever1.SimpleRetriever'), \
             patch('retrieval.embeddings.EmbeddingGenerator'):
            self.app = create_app()
        self.client = self.app.test_client()

    def test_create_app(self):
        """Test Flask app creation."""
        with patch('api.routes.RAGPipeline'), \
             patch('api.routes.ResponseParser'), \
             patch('rag.pipeline.RAGPipeline'), \
             patch('rag.response_parser.ResponseParser'), \
             patch('models.model_loader.ModelLoader'), \
             patch('models.inference.InferenceEngine'), \
             patch('rag.context_builder.ContextBuilder'), \
             patch('retrieval.simple_retriever1.SimpleRetriever'), \
             patch('retrieval.embeddings.EmbeddingGenerator'):
            app = create_app()
        self.assertIsNotNone(app)
        self.assertEqual(app.config['SECRET_KEY'], os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_change_me'))

    def test_health_check_healthy(self):
        """Test health check endpoint when system is healthy."""
        with patch('api.routes.RAG_PIPELINE', MagicMock()):
            response = self.client.get('/api/v1/health')
            self.assertEqual(response.status_code, 200)

            data = json.loads(response.data)
            self.assertEqual(data['status'], 'healthy')
            self.assertIn('timestamp', data)
            self.assertIn('model', data)
            self.assertIn('message', data)

    def test_health_check_degraded(self):
        """Test health check endpoint when system is degraded."""
        # Create a separate app instance for this test
        def mock_login_required(f):
            return f
        
        with patch('app.api.auth.auth.login_required', mock_login_required), \
             patch('api.routes.RAGPipeline') as mock_rag_class, \
             patch('api.routes.ResponseParser') as mock_parser_class:
            # Make RAGPipeline initialization fail
            mock_rag_class.side_effect = Exception("Pipeline init failed")
            mock_parser_class.return_value = MagicMock()
            
            app = create_app()
            client = app.test_client()
            
            response = client.get('/api/v1/health')
            self.assertEqual(response.status_code, 200)

            data = json.loads(response.data)
            self.assertEqual(data['status'], 'degraded')

    def test_rag_query_success(self):
        """Test successful RAG query."""
        # Create a separate app instance for this test
        def mock_login_required(f):
            return f
        
        mock_rag = MagicMock()
        mock_rag.run_rag_pipeline.return_value = ("Generated answer", [
            {
                'source_id': 'S1',
                'text': 'Legal text',
                'metadata': {'source': 'test'},
                'rerank_score': 0.9,
                'source_db': 'test_db'
            }
        ])
        
        mock_resp_parser = MagicMock()
        mock_resp_parser.parse_llm_output.return_value = {
            'answer_with_explanation': 'Parsed answer with explanation'
        }

        with patch('app.api.auth.auth.login_required', mock_login_required), \
             patch('api.routes.RAGPipeline', return_value=mock_rag), \
             patch('api.routes.ResponseParser', return_value=mock_resp_parser), \
             patch('api.routes.RAG_PIPELINE', mock_rag), \
             patch('api.routes.RESPONSE_PARSER', mock_resp_parser), \
             patch('rag.pipeline.RAGPipeline'), \
             patch('rag.response_parser.ResponseParser'), \
             patch('models.model_loader.ModelLoader'), \
             patch('models.inference.InferenceEngine'), \
             patch('rag.context_builder.ContextBuilder'), \
             patch('retrieval.simple_retriever1.SimpleRetriever'), \
             patch('retrieval.embeddings.EmbeddingGenerator'):
            app = create_app()
            client = app.test_client()

            # Test request
            request_data = {'query': 'What is the legal requirement?'}
            response = client.post('/api/v1/query',
                                  data=json.dumps(request_data),
                                  content_type='application/json')

            self.assertEqual(response.status_code, 200)

            data = json.loads(response.data)
            self.assertEqual(data['query'], request_data['query'])
            self.assertEqual(data['answer_with_explanation'], 'Parsed answer with explanation')
            self.assertIn('sources', data)
            self.assertIn('timestamp', data)
            self.assertEqual(data['status'], 'success')

    def test_rag_query_invalid_json(self):
        """Test RAG query with invalid JSON."""
        def mock_login_required(f):
            return f
        
        with patch('app.api.auth.auth.login_required', mock_login_required):
            response = self.client.post('/api/v1/query',
                                      data="invalid json",
                                      content_type='application/json')
            self.assertEqual(response.status_code, 400)

    def test_rag_query_missing_json(self):
        """Test RAG query with missing JSON body."""
        def mock_login_required(f):
            return f
        
        with patch('app.api.auth.auth.login_required', mock_login_required):
            response = self.client.post('/api/v1/query')
            self.assertEqual(response.status_code, 400)

    @patch('app.api.auth.auth.login_required')
    def test_rag_query_pipeline_not_initialized(self, mock_auth_decorator):
        """Test RAG query when pipeline is not initialized."""
        mock_auth_decorator.return_value = lambda f: f

        # Create a separate app instance for this test
        def mock_login_required(f):
            return f
        
        with patch('app.api.auth.auth.login_required', mock_login_required), \
             patch('api.routes.RAGPipeline', side_effect=Exception("Pipeline init failed")), \
             patch('api.routes.ResponseParser'), \
             patch('rag.pipeline.RAGPipeline'), \
             patch('rag.response_parser.ResponseParser'), \
             patch('models.model_loader.ModelLoader'), \
             patch('models.inference.InferenceEngine'), \
             patch('rag.context_builder.ContextBuilder'), \
             patch('retrieval.simple_retriever1.SimpleRetriever'), \
             patch('retrieval.embeddings.EmbeddingGenerator'):
            app = create_app()
            client = app.test_client()

            request_data = {'query': 'Test query'}
            response = client.post('/api/v1/query',
                                  data=json.dumps(request_data),
                                  content_type='application/json')
            self.assertEqual(response.status_code, 503)

    @patch('app.api.auth.auth.login_required')
    @patch('api.routes.RAG_PIPELINE')
    def test_rag_query_response_parser_not_initialized(self, mock_rag_pipeline, mock_auth_decorator):
        """Test RAG query when response parser is not initialized."""
        mock_auth_decorator.return_value = lambda f: f

        mock_rag_pipeline.run_rag_pipeline.return_value = ("Generated answer", [])

        with patch('api.routes.RESPONSE_PARSER', None):
            request_data = {'query': 'Test query'}
            response = self.client.post('/api/v1/query',
                                      data=json.dumps(request_data),
                                      content_type='application/json')
            self.assertEqual(response.status_code, 500)

    def test_query_request_schema_validation(self):
        """Test QueryRequest schema validation."""
        # Valid request
        valid_request = QueryRequest(query="What is the legal requirement?")
        self.assertEqual(valid_request.query, "What is the legal requirement?")

        # Invalid request - too short
        with self.assertRaises(ValueError):
            QueryRequest(query="Hi")

    def test_query_response_schema(self):
        """Test QueryResponse schema creation."""
        sources = [{
            'source_id': 'S1',
            'text': 'Legal text',
            'metadata': {'source': 'test'},
            'rerank_score': 0.9,
            'source_db': 'test_db'
        }]

        response = QueryResponse(
            query="Test query",
            answer_with_explanation="Test answer with explanation",
            sources=sources
        )

        self.assertEqual(response.query, "Test query")
        self.assertEqual(response.answer_with_explanation, "Test answer with explanation")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.status, "success")
        self.assertIsNotNone(response.timestamp)


if __name__ == '__main__':
    unittest.main()