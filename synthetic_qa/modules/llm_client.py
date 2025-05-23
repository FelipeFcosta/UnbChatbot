"""
LLM client module for the Synthetic QA Generator.

This module handles communication with LLM APIs, including intelligent rate limiting.
"""

import os
import random
import time
import logging
from typing import Dict, Any, Optional, List, Union
from .utils import json_if_valid, RateLimiter

logger = logging.getLogger(__name__)

# Import Google Gemini if available
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("genAI package not installed; Gemini will not work.")

# Import OpenAI for OpenRouter fallback only
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed; OpenRouter fallback will not work.")

class LLMClient:
    """Handles communication with LLM APIs for question and answer generation."""
    
    # Shared rate limiters for different models
    _rate_limiters = {}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM client with the provided configuration.
        
        Args:
            config: Configuration dictionary with provider details
        """
        self.config = config
        self.client = None
        
        provider = config.get('provider', '').lower()
        model = config.get('model', '')

        # Create a descriptive model name for logging
        model_name = f"{provider.upper()} {model}"
        
        # Set up rate limiter for this model
        model_key = f"{config.get('provider', '').lower()}_{config.get('model', '')}"
        rate_limit_rpm = config.get('rate_limit_rpm')
        
        # Create or retrieve the rate limiter for this specific model
        if model_key not in self._rate_limiters:
            self._rate_limiters[model_key] = RateLimiter(rate_limit_rpm, model_name)
        self._rate_limiter = self._rate_limiters[model_key]
        
        # Initialize Gemini client if needed
        if config.get("provider", "").lower() == "genai":
            if not GEMINI_AVAILABLE:
                raise ImportError("genAI package not installed. Install it with 'pip install google-genai'")
            
            api_key = os.environ.get("GEMINI_API_KEY3")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            self.client = genai.Client(api_key=api_key)

    def generate_text(self, prompt: str, temperature: Optional[float] = None, 
                     json_output: bool = False, long_output: bool = False) -> Union[str, Dict[str, Any], None]:
        """
        Generate text from the LLM based on the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Override the default temperature if provided
            json_output: If True, request and return JSON output
            long_output: If True, handle generating longer outputs
            
        Returns:
            Generated text, parsed JSON (if json_output=True), or None on failure
        """
        # Wait according to the rate limit before making the request
        # self._rate_limiter.wait()

        provider = self.config.get("provider", "").lower()
        model = self.config.get("model", "")
        
        if not provider or not model:
            logger.error("Provider or model not specified in config")
            return None
            
        # Handle Gemini API
        if provider == "genai":
            thinking_config = None
            if "2.5-flash" in model:
                thinking_config=types.ThinkingConfig(thinking_budget=self.config.get("thinking_budget", 1024))
            generation_config = types.GenerateContentConfig(
                max_output_tokens=self.config.get("max_tokens", 8192),
                temperature=temperature if temperature is not None else self.config.get("temperature", 0.7),
                thinking_config=thinking_config
            )

            full_response = ""
            current_prompt = prompt

            if long_output:
                current_prompt += "\n\nWhen you're finished, your final line should be only the word 'DONE'"

            max_tries = 2
            response = None

            try:
                while max_tries > 0:
                    # Send current prompt + accumulated response (context) for the model to continue
                    try:
                        response = self.client.models.generate_content(
                            model=model,
                            contents=current_prompt,
                            config=generation_config
                        )
                        full_response += response.text
                    except Exception as e:
                        logger.error(f"Gemini API error: {e}")
                        # OpenRouter fallback
                        return None # TODO: remove this
                        openrouter_response = self._openrouter_fallback(prompt, temperature)
                        if openrouter_response:
                            full_response += openrouter_response
                        else:
                            return None

                    if json_output:
                        # Clean up response for JSON parsing
                        full_response = full_response.replace("```json", "").replace("```", "")

                        try:
                            json_response = json_if_valid(full_response)

                            if json_response:
                                return json_response
                        except Exception:
                            pass
                    
                        max_tries -= 1

                        continue_prompt = "\n\nContinue the existing JSON exactly from where it stopped, maintaining " \
                        "its structure and formatting without starting a new JSON object"

                        current_prompt = prompt + "\n ...\n" + full_response + continue_prompt
                    
                    elif long_output:
                        # Clean up response for long text
                        full_response = full_response.replace("```markdown", "").replace("```", "")

                        if full_response.strip().lower().endswith("done"):
                            # Remove the DONE marker and return
                            return full_response[:full_response.strip().rfind("\n")]
                        else:
                            max_tries -= 1

                        continue_prompt = "\n\nContinue the existing text **exactly** from where it stopped, maintaining " \
                        "its structure and formatting without starting a new structured object\n" \
                        "if you're finished, your final line should be only the word 'DONE'"

                        current_prompt = prompt + "\n ...\n" + full_response + continue_prompt
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

    def _openrouter_fallback(self, prompt: str, temperature: Optional[float] = None) -> Optional[str]:
        """
        Fallback to OpenRouter API using OpenAI client if Gemini API fails.
        Args:
            prompt: The prompt to send to OpenRouter
            temperature: Sampling temperature
        Returns:
            The response text from OpenRouter, or None on failure
        """
        if not (OPENAI_AVAILABLE and os.environ.get("OPENROUTER_API_KEY")):
            return None
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY2"),
            )
            completion = client.chat.completions.create(
                extra_headers={},
                extra_body={},
                model="google/gemini-2.5-pro-exp-03-25:free",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                temperature=temperature if temperature is not None else self.config.get("temperature", 0.7)
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return None