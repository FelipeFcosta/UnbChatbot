"""
LLM client module for the Synthetic QA Generator.

This module handles communication with LLM APIs.
"""

import os
import random
import time
import logging
from typing import Dict, Any, Optional, List, Union
from .utils import json_if_valid

logger = logging.getLogger(__name__)

# Import OpenAI if available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed; openai provider will not work.")

# Import Google Gemini if available
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("genAI package not installed; Gemini will not work.")


class LLMClient:
    """Handles communication with LLM APIs for question and answer generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM client with the provided configuration.
        
        Args:
            config: Configuration dictionary with provider details
        """
        self.config = config
        self.client = None
        
        # Initialize OpenAI client if needed
        if config.get("provider", "").lower() == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install it with 'pip install openai'")
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            self.client = OpenAI(api_key=api_key)
        
        # Initialize Gemini client if needed
        elif config.get("provider", "").lower() == "genai":
            if not GEMINI_AVAILABLE:
                raise ImportError("genAI package not installed. Install it with 'pip install google-genai'")
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            self.client = genai.Client(api_key=api_key)


    def generate_text(self, prompt: str, max_retries: int = 5, temperature: Optional[float] = None, 
                     json_output: bool = False, long_output: bool = False) -> Union[str, Dict[str, Any], None]:
        """
        Generate text from the LLM based on the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retries on failure
            temperature: Override the default temperature if provided
            json_output: If True, request and return JSON output
            long_output: If True, handle generating longer outputs
            
        Returns:
            Generated text, parsed JSON (if json_output=True), or None on failure
        """
        provider = self.config.get("provider", "").lower()
        model = self.config.get("model", "")
        
        if not provider or not model:
            logger.error("Provider or model not specified in config")
            return None
            
        backoff_factor = 1  # Base backoff time in seconds
        
        # Handle OpenAI API
        if provider == "openai":
            if not self.client:
                logger.error("OpenAI client is not initialized")
                return None
                
            attempt = 0
            while attempt <= max_retries:
                try:
                    # Using the OpenAI client to generate text
                    response = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=temperature if temperature is not None else self.config.get("temperature", 0.7),
                        max_tokens=self.config.get("max_tokens", 2048)
                    )

                    result = response.choices[0].message.content

                    # Handle JSON output if requested
                    if json_output:
                        return json_if_valid(result) or result

                    return result

                except Exception as e:
                    logger.error(f"OpenAI API error on attempt {attempt+1}/{max_retries}: {e}")
                    if attempt == max_retries:
                        return None

                    sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying OpenAI API call in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    attempt += 1

        # Handle Gemini API
        elif provider == "genai":
            generation_config = {
                "max_output_tokens": self.config.get("max_tokens", 8192),
                "temperature": temperature if temperature is not None else self.config.get("temperature", 0.7)
            }

            full_response = ""
            current_prompt = prompt

            if long_output:
                current_prompt += "\n\nWhen you're finished, your final line should be only the word 'DONE'"

            max_tries = 2

            try:
                while max_tries > 0:
                    # Send current prompt + accumulated response (context) for the model to continue
                    response = self.client.models.generate_content(
                        model=model,
                        contents=current_prompt,
                        config=generation_config
                    )
                    time.sleep(4) # limit to 15 RPM

                    full_response += response.text

                    if json_output:
                        # Clean up response for JSON parsing
                        full_response = full_response.replace("```json", "").replace("```", "")

                        try:
                            json_response = json_if_valid(full_response)

                            if json_response:
                                return json_response
                        except Exception:
                            max_tries -= 1

                        # Take the tail of the response to continue with LLM
                        response_tail = full_response[-5000:] if len(full_response) > 5000 else full_response

                        continue_prompt = "\n\nContinue the existing JSON exactly from where it stopped, maintaining " \
                        "its structure and formatting without starting a new JSON object"

                        current_prompt = prompt + "\n ...\n" + response_tail + continue_prompt
                    
                    elif long_output:
                        # Clean up response for long text
                        full_response = full_response.replace("```markdown", "").replace("```", "")

                        if full_response.strip().lower().endswith("done"):
                            # Remove the DONE marker and return
                            return full_response[:full_response.strip().rfind("\n")]
                        else:
                            max_tries -= 1

                        # Take the tail of the response to continue with LLM
                        response_tail = full_response[-5000:] if len(full_response) > 5000 else full_response

                        continue_prompt = "\n\nContinue the existing text **exactly** from where it stopped, maintaining " \
                        "its structure and formatting without starting a new structured object\n" \
                        "if you're finished, your final line should be only the word 'DONE'"

                        current_prompt = prompt + "\n ...\n" + response_tail + continue_prompt
                    else:
                        # Single response is complete, return it
                        break

                # Return the full accumulated response
                return full_response

            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return None

        # Unknown provider
        else:
            logger.error(f"Unknown provider: {provider}")
            return None