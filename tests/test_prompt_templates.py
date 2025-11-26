import unittest

# Add the app directory to the path so we can import modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from rag.prompt_templates import PromptTemplates


class TestPromptTemplates(unittest.TestCase):

    def test_get_system_prompt(self):
        """Test getting the system prompt."""
        system_prompt = PromptTemplates.get_system_prompt()

        self.assertIsInstance(system_prompt, str)
        self.assertGreater(len(system_prompt), 0)
        self.assertIn("Legal AI Assistant", system_prompt)
        self.assertIn("Source ID", system_prompt)
        self.assertIn("cite sources", system_prompt)

    def test_format_prompt(self):
        """Test formatting the complete prompt."""
        system_prompt = "Test system prompt"
        context = "Test context with legal information"
        query = "What is the legal requirement?"

        formatted_prompt = PromptTemplates.format_prompt(system_prompt, context, query)

        self.assertIsInstance(formatted_prompt, str)
        self.assertGreater(len(formatted_prompt), 0)

        # Check for LLaMA 3.1 instruct format markers
        self.assertIn("<|begin_of_text|>", formatted_prompt)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", formatted_prompt)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", formatted_prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", formatted_prompt)
        self.assertIn("<|eot|>", formatted_prompt)

        # Check content inclusion
        self.assertIn(system_prompt, formatted_prompt)
        self.assertIn(context, formatted_prompt)
        self.assertIn(query, formatted_prompt)

        # Check instructions are included
        self.assertIn("Use the context above", formatted_prompt)
        self.assertIn("Answer with Explanation", formatted_prompt)

    def test_format_prompt_with_empty_inputs(self):
        """Test formatting prompt with empty inputs."""
        system_prompt = ""
        context = ""
        query = ""

        formatted_prompt = PromptTemplates.format_prompt(system_prompt, context, query)

        # Should still have the correct structure even with empty content
        self.assertIn("<|begin_of_text|>", formatted_prompt)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", formatted_prompt)
        self.assertIn("<|eot|>", formatted_prompt)

    def test_format_prompt_special_characters(self):
        """Test formatting prompt with special characters."""
        system_prompt = "System with & < > \" '"
        context = "Context with & < > \" '"
        query = "Query with & < > \" '"

        formatted_prompt = PromptTemplates.format_prompt(system_prompt, context, query)

        # Should handle special characters properly
        self.assertIn(system_prompt, formatted_prompt)
        self.assertIn(context, formatted_prompt)
        self.assertIn(query, formatted_prompt)

    def test_format_prompt_structure(self):
        """Test that the prompt structure follows LLaMA 3.1 format."""
        system_prompt = "You are a legal assistant."
        context = "Legal context here."
        query = "What is the law?"

        formatted = PromptTemplates.format_prompt(system_prompt, context, query)

        # Verify the exact structure
        expected_parts = [
            "<|begin_of_text|>",
            "<|start_header_id|>system<|end_header_id|>",
            "\n",
            system_prompt,
            "<|eot|>",
            "<|start_header_id|>user<|end_header_id|>",
            "\n",
            "<|eot|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "\n"
        ]

        for part in expected_parts:
            self.assertIn(part, formatted)

        # Verify user prompt content
        self.assertIn("Context:\n---\n", formatted)
        self.assertIn(context, formatted)
        self.assertIn("---\n\n", formatted)
        self.assertIn(f"Query: {query}", formatted)


if __name__ == '__main__':
    unittest.main()