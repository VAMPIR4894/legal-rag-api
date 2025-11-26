from typing import List, Dict, Any

class PromptTemplates:
    """
    Defines the structured prompt templates for the Legal RAG system.
    """

    @staticmethod
    def get_system_prompt() -> str:
        """
        The fixed system prompt instructing the LLM on its role and output format.
        """
        # System prompt as required by the LLaMA 3.1 instruct format
        return (
            "You are a Legal AI Assistant. Answer queries using ONLY the provided context. "
            "NEVER say 'insufficient information' or similar disclaimers. "
            "Cite sources using [S1], [S2] format. Provide complete answers."
        )

    @staticmethod
    def format_prompt(system_prompt: str, context: str, query: str) -> str:
        """
        Formats the final prompt using the LLaMA 3.1 Instruct template.
        """
        # LLaMA 3.1 Instruct format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot|><|start_header_id|>assistant<|end_header_id|>\n
        
        user_prompt = (
            f"Context (contains relevant information):\n---\n{context}\n---\n\n"
            f"Query: {query}\n\n"
            f"Use the context above to answer. The context contains the information needed.\n"
            f"Start your response with 'Answer with Explanation:' and provide a complete answer."
        )
        
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot|>"
            f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )