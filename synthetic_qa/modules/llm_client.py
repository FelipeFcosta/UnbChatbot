"""
LLM client module for the Synthetic QA Generator.

This module handles communication with LLM APIs.
"""

import os
import random
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try importing OpenAI (this handles both old and new clients)
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed; openai provider will not work.")


class LLMClient:
    """Handles communication with LLM APIs for question and answer generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM client with the provided configuration.
        
        Args:
            config: Configuration dictionary with provider details
        """
        self.config = config
        self.openai_client = None
        
        if config.get("provider", "").lower() == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install it with 'pip install openai'")
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                pass #TODO: remove this
                # raise ValueError("OPENAI_API_KEY environment variable not set")
                
            # self.openai_client = OpenAI(api_key=api_key) # TODO:
            
    def generate_text(self, prompt: str, max_retries: int = 5, temperature: Optional[float] = None) -> Optional[str]:
        """
        Generate text from the LLM based on the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retries on failure
            temperature: Override the default temperature if provided
            
        Returns:
            The generated text or None if failed
        """
        return '' #TODO: remove this
        provider = self.config.get("provider", "").lower()
        model = self.config.get("model", "")
        
        if not provider or not model:
            logger.error("Provider or model not specified in config")
            return None
            
        backoff_factor = 1  # Base backoff time in seconds
        
        if provider == "openai":
            if not self.openai_client:
                logger.error("OpenAI client is not initialized")
                return None
                
            attempt = 0
            while attempt <= max_retries:
                try:
                    # Using the OpenAI client to generate text
                    response = self.openai_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=temperature if temperature is not None else self.config.get("temperature", 0.7),
                        max_tokens=self.config.get("max_tokens", 2048)
                    )
                    return response.choices[0].message.content
                    
                except Exception as e:
                    logger.error(f"OpenAI API error on attempt {attempt+1}/{max_retries}: {e}")
                    if attempt == max_retries:
                        return None
                    sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying OpenAI API call in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    attempt += 1
                    
        elif provider == "anthropic":
            # Implement Anthropic/Claude API here if needed
            logger.error("Anthropic provider not implemented yet")
            return None
            
        else:
            logger.error(f"Unknown provider: {provider}")
            return None