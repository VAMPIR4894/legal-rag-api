import requests
from typing import List, Dict, Any

class InferenceEngine:
    """
    Handles response generation via the Custom Mistral RAG API (ngrok).
    """
    def __init__(self, model_name: str, api_url: str):
        self.model_name = model_name
        # Ensure clean URL (remove trailing slash)
        self.api_url = api_url.rstrip('/')
        print(f"Inference Engine ready, targeting Remote Mistral API: {self.api_url}")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        stream: bool = False
    ) -> str:
        """
        Generates text completion by posting a request to the custom Mistral API.

        Args:
            prompt: The user query (or full prompt).
            max_tokens: (Ignored by current API server configuration)
            temperature: (Ignored by current API server configuration)
            stream: (Ignored)

        Returns:
            The generated text response.
        """
        
        # 1. Construct the Endpoint
        endpoint = f"{self.api_url}/generate"

        # 2. Mistral API Payload
        # Based on the 'Smart RAG' API we built, it only needs the prompt.
        # The server handles retrieval and context injection.
        payload = {
            "prompt": prompt
        }

        try:
            # 3. Send Request
            # We use a generous timeout as the remote GPU might be slow
            response = requests.post(endpoint, json=payload, timeout=300)
            response.raise_for_status()
            
            # 4. Parse Response
            # Your API returns: {"answer": "...", "source_context": "..."}
            data = response.json()
            
            # Extract just the answer text to maintain compatibility with the rest of your app
            generated_text = data.get('answer', '').strip()
            
            if not generated_text:
                # Fallback if something went wrong with parsing
                print(f"Warning: Empty response from API. Full data: {data}")
                generated_text = "Error: Received empty response from model."

            return generated_text

        except requests.exceptions.RequestException as e:
            print(f"Mistral API Error during generation: {e}")
            raise ConnectionError("Remote inference failed. Check ngrok URL and server status.")