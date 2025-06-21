import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os
import json

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from synthetic_qa.modules.text_chunker import TextChunker

class TestTextChunker(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        # Mock LLMClient to avoid actual API calls
        with patch('synthetic_qa.modules.text_chunker.LLMClient') as MockLLMClient:
            self.mock_llm_client = MockLLMClient.return_value
            self.mock_llm_client.generate_text.return_value = {"chunks": [{"chunk": "mocked chunk", "topic": "mocked topic"}]}
            self.chunker = TextChunker(config={
                "providers": {
                    "text_chunking": {
                        "max_chunking_length": 40000
                    }
                }
            })

        self.test_data_path = Path('/home/farias/tcc/qa_generation/synthetic_qa/output/unb.br/extracted_text/estatuto-e-regimento-geral-unb_0dc251e57e1e.txt')
        
        if not self.test_data_path.exists():
            # If the intended file doesn't exist, create a dummy large file for testing purposes
            self.large_text = self._create_dummy_large_file()
        else:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                self.large_text = f.read()
    
    def _create_dummy_large_file(self):
        """Creates a large dummy text file for testing if the original is not found."""
        print(f"Test file not found at {self.test_data_path}. Creating a dummy file.")
        dummy_dir = project_root / 'synthetic_qa' / 'tests' / 'temp_data'
        dummy_dir.mkdir(exist_ok=True)
        self.test_data_path = dummy_dir / "large_file.txt"
        
        # Create a large text with newlines
        base_sentence = "This is a sentence for testing purposes. It will be repeated many times to create a large file.\n"
        # Make it > 40000 characters
        large_text = base_sentence * 800
        
        with open(self.test_data_path, "w", encoding="utf-8") as f:
            f.write(large_text)
            
        return large_text

    def test_split_text_if_needed_with_large_text(self):
        """Test that a large text is split into smaller parts."""
        self.assertGreater(len(self.large_text), self.chunker.max_chunking_length, "Test text is not large enough.")
        
        parts = self.chunker._split_text_if_needed(self.large_text)
        
        self.assertIsInstance(parts, list, "Splitting should result in a list of parts.")
        self.assertTrue(len(parts) > 1, "Text was not split.")
        
        # Verify that the sum of the parts equals the original text length
        self.assertEqual(sum(len(part) for part in parts), len(self.large_text), "Sum of parts' lengths must match original text length.")

        # The recursive splitting logic should result in final parts smaller than the max length
        for i, part in enumerate(parts):
            self.assertTrue(len(part) <= self.chunker.max_chunking_length, f"Part {i} is too long: {len(part)}")

    def test_chunk_text_with_large_file(self):
        """Test the main chunk_text method with a large file that needs splitting."""
        self.assertGreater(len(self.large_text), self.chunker.max_chunking_length, "Test text is confirmed to be large.")

        # Let the mock return a different number of chunks for each call to see the effect
        self.mock_llm_client.generate_text.side_effect = [
            {"chunks": [{"chunk": "mocked chunk 1", "topic": "topic 1"}]},
            {"chunks": [{"chunk": "mocked chunk 2", "topic": "topic 2"}, {"chunk": "mocked chunk 3", "topic": "topic 3"}]},
            {"chunks": [{"chunk": "mocked chunk 4", "topic": "topic 4"}]},
        ] * 10 # Repeat to cover all splits

        chunks = self.chunker.chunk_text(self.large_text, self.test_data_path)

        self.assertIsInstance(chunks, list, "Chunking result should be a list.")
        self.assertTrue(len(chunks) > 1, "Chunking a large file should produce more than one chunk.")
        self.assertTrue(self.mock_llm_client.generate_text.call_count > 1, "LLM client should be called multiple times for a split file.")

    def test_no_split_for_small_text(self):
        """Test that small text is not split."""
        small_text = "This is a small text that should not be split."
        self.assertLessEqual(len(small_text), self.chunker.max_chunking_length, "Test text is confirmed to be small.")
        
        parts = self.chunker._split_text_if_needed(small_text)
        self.assertEqual(len(parts), 1, "Small text should not be split into multiple parts.")
        self.assertEqual(parts[0], small_text, "The content of the single part should be identical to the original small text.")
        
        chunks = self.chunker.chunk_text(small_text, Path("dummy.txt"))
        self.assertEqual(self.mock_llm_client.generate_text.call_count, 1, "LLM client should be called only once for small text.")
        self.assertEqual(len(chunks), 1, "Chunking small text should result in a single chunk (based on mock setup).")

if __name__ == '__main__':
    unittest.main() 