import os
import requests
from typing import Tuple, Any

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# ⚠️ PASTE YOUR NGROK URL HERE (from the Colab output)
DEFAULT_API_URL = "https://REPLACE-WITH-YOUR-NGROK-URL.ngrok-free.app"

# Check environment variable first, otherwise use the default above
MISTRAL_API_BASE_URL = os.environ.get("MISTRAL_API_BASE_URL", DEFAULT_API_URL)

# Label for the model (used for logging/UI)
MODEL_NAME = "mistral-json-rag-v1"

class ModelLoader:
    """
    Manages the connection details for the Remote Mistral RAG Server.
    """
    def __init__(self):
        # Clean up the URL (remove trailing slash if user added it)
        self.api_url = MISTRAL_API_BASE_URL.rstrip('/')
        self.model_name = MODEL_NAME
        self._check_connection()

    def _check_connection(self):
        """
        Pings the FastAPI /health endpoint to verify the Colab server is running.
        """
        print(f"\n--- Checking Connection for Remote RAG Server at: {self.api_url} ---")
        
        # The health endpoint defined in your Colab server code
        health_url = f"{self.api_url}/health"

        try:
            response = requests.get(health_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Verify the status returned by the server
            if data.get("status") == "healthy":
                print("✅ Server is connected! RAG Backend is ready.")
                return
            else:
                # If the server responds but says it's not healthy
                raise ConnectionError(f"Server reachable but returned unexpected status: {data}")

        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: Could not connect to the RAG server at {health_url}")
            print(f"   Hint: The URL might be wrong or the Colab cell stopped running.")
            raise ConnectionError(f"Mistral API connection failed: {e}")

    def get_model_and_tokenizer(self) -> Tuple[str, str]:
        """
        Returns the model name and the Base API URL.
        
        Returns:
            Tuple[str, str]: (model_name, api_url)
        """
        # In this architecture, the 'tokenizer' object is effectively the API URL,
        # because the server handles tokenization.
        return self.model_name, self.api_url