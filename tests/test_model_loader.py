import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open

# Add the app directory to the path so we can import modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from models.model_loader import ModelLoader, OLLAMA_MODEL_NAME, OLLAMA_API_BASE_URL


class TestModelLoader(unittest.TestCase):

    def setUp(self):
        # Store original environment variables
        self.original_env = dict(os.environ)

    def tearDown(self):
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch('models.model_loader.requests.get')
    def test_init_with_env_model(self, mock_get):
        """Test initialization with model name from environment."""
        os.environ['OLLAMA_MODEL_NAME'] = 'test-model'
        os.environ['OLLAMA_API_BASE_URL'] = 'http://test:11434/api/generate'

        # Mock successful response with model available
        mock_response = MagicMock()
        mock_response.json.return_value = {'models': [{'name': 'test-model'}]}
        mock_get.return_value = mock_response

        loader = ModelLoader()
        self.assertEqual(loader.model_name, 'test-model')
        self.assertEqual(loader.api_url, 'http://test:11434/api/generate')

    @patch('models.model_loader.requests.get')
    def test_init_with_dotenv_fallback(self, mock_get):
        """Test initialization with model name from .env file fallback."""
        # Clear env var to trigger .env reading
        if 'OLLAMA_MODEL_NAME' in os.environ:
            del os.environ['OLLAMA_MODEL_NAME']

        # Create temporary .env file
        env_content = "OLLAMA_MODEL_NAME=llama3.2:1b\nOTHER_VAR=test\n"
        with patch('builtins.open', mock_open(read_data=env_content)):
            with patch('os.path.exists', return_value=True):
                # Mock successful response
                mock_response = MagicMock()
                mock_response.json.return_value = {'models': [{'name': 'llama3.2:1b'}]}
                mock_get.return_value = mock_response

                loader = ModelLoader()
                self.assertEqual(loader.model_name, 'llama3.2:1b')

    @patch('models.model_loader.requests.get')
    def test_check_connection_success(self, mock_get):
        """Test successful connection check."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'models': [{'name': 'llama3.2:1b'}]}
        mock_get.return_value = mock_response

        loader = ModelLoader()
        # Should not raise exception
        self.assertIsNotNone(loader.api_url)

    @patch('models.model_loader.requests.get')
    def test_check_connection_model_not_found(self, mock_get):
        """Test connection check when model is not available."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'models': [{'name': 'other-model'}]}
        mock_get.return_value = mock_response

        with self.assertRaises(ConnectionError) as cm:
            ModelLoader()
        self.assertIn("model 'llama3.2:1b' not listed", str(cm.exception))

    @patch('models.model_loader.requests.get')
    def test_check_connection_server_unreachable(self, mock_get):
        """Test connection check when server is unreachable."""
        from requests.exceptions import RequestException
        mock_get.side_effect = RequestException("Connection failed")

        with self.assertRaises(ConnectionError) as cm:
            ModelLoader()
        self.assertIn("Ollama connection/model verification failed", str(cm.exception))

    def test_get_model_and_tokenizer(self):
        """Test getting model and tokenizer info."""
        with patch('models.model_loader.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'models': [{'name': 'llama3.2:1b'}]}
            mock_get.return_value = mock_response

            loader = ModelLoader()
            model_name, api_url = loader.get_model_and_tokenizer()
            self.assertEqual(model_name, 'llama3.2:1b')
            self.assertEqual(api_url, loader.api_url)

    @patch('models.model_loader.requests.get')
    def test_multiple_endpoints_check(self, mock_get):
        """Test that multiple endpoints are tried for connection check."""
        # First two calls fail, third succeeds
        mock_get.side_effect = [
            Exception("First endpoint fails"),
            Exception("Second endpoint fails"),
            MagicMock(json=lambda: {'models': [{'name': 'llama3.2:1b'}]})
        ]

        loader = ModelLoader()
        # Should succeed on third try
        self.assertEqual(mock_get.call_count, 3)


if __name__ == '__main__':
    unittest.main()