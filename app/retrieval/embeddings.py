import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the embedding model as per the requirements (high-accuracy model)
# For faster testing, you can temporarily change to "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

class EmbeddingGenerator:
    """
    Manages the loading and usage of the SentenceTransformer model
    for generating embeddings. Enhanced with legal-specific models and query expansion.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initializes the embedding model.
        """
        # Set device to 'cpu' if no CUDA device is available to prevent errors
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device

        # Get HuggingFace token from environment
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        # Try to load as SentenceTransformer first, fallback to transformers
        try:
            self.model = SentenceTransformer(model_name, device=device, token=hf_token)
            self.use_sentence_transformers = True
            print(f"Loaded SentenceTransformer model: {model_name} on device: {device}")
        except Exception as e:
            print(f"SentenceTransformer failed ({e}), trying transformers model: {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                self.model = AutoModel.from_pretrained(model_name, token=hf_token).to(device)
                self.use_sentence_transformers = False
                print(f"Loaded transformers model: {model_name} on device: {device}")
            except Exception as e2:
                print(f"Both loading methods failed. Falling back to default model. Error: {e2}")
                # Fallback to a known working model
                fallback_model = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(fallback_model, device=device)
                self.use_sentence_transformers = True
                print(f"Loaded fallback model: {fallback_model} on device: {device}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for transformer models."""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts: A list of strings to be embedded.

        Returns:
            A list of embedding vectors (list of floats).
        """
        if self.use_sentence_transformers:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            # Convert NumPy arrays to list of lists for ChromaDB
            return embeddings.tolist()
        else:
            # Use transformers approach
            embeddings = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = self._mean_pooling(outputs, inputs['attention_mask'])
                    embedding = F.normalize(embedding, p=2, dim=1)  # Normalize
                    embeddings.append(embedding.cpu().numpy().flatten())
            return embeddings.tolist()

    def expand_query(self, query: str, expansion_terms: int = 3) -> str:
        """
        Expand query with related legal terms for better retrieval.
        This is a simple approach - in production, you might use a more sophisticated method.
        """
        # Legal domain-specific expansion terms
        legal_expansions = {
            "property": ["real estate", "land", "ownership", "title", "possession"],
            "dispute": ["litigation", "conflict", "claim", "lawsuit", "controversy"],
            "legal": ["lawful", "statutory", "judicial", "court", "jurisdiction"],
            "court": ["tribunal", "judiciary", "bench", "justice", "proceedings"],
            "contract": ["agreement", "pact", "covenant", "obligation", "terms"],
            "evidence": ["proof", "testimony", "documentation", "witness", "facts"],
            "rights": ["entitlements", "privileges", "claims", "interests", "authority"],
            "case": ["matter", "suit", "action", "proceeding", "litigation"]
        }

        expanded_terms = []
        query_lower = query.lower()

        for word in query_lower.split():
            if word in legal_expansions:
                # Add original word and some expansion terms
                expanded_terms.extend([word] + legal_expansions[word][:expansion_terms])

        if expanded_terms:
            expanded_query = query + " " + " ".join(expanded_terms[:expansion_terms])
            return expanded_query

        return query