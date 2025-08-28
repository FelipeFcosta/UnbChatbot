
import logging
from typing import Dict, List
from .config import INTENT_CLASSIFIER_PROMPT, INTENT_CLASSIFIER_PROMPT_WITH_HISTORY,QUERY_EXPANSION_PROMPT, CONTEXTUALIZE_MESSAGE_PROMPT

class QueryProcessor:
    def __init__(self, helper_llm, logger):
        self.helper_llm = helper_llm
        self._non_domain_hashes = set()
        self.logger = logger

    @staticmethod
    def _hash_text(text: str) -> str:
        """Return a stable SHA256 hex digest for a given text snippet."""
        import hashlib
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def classify_intent(self, current_text: str, previous_question: str = "", previous_response: str = "") -> str:
        """Detects whether the message is 'domain_query' or 'non_domain_query'."""
        if not self.helper_llm:
            return "domain_query"


        if previous_response:
            print(f"Previous response: {previous_response}")
            base_prompt = INTENT_CLASSIFIER_PROMPT_WITH_HISTORY.format(
                current_text=current_text,
                history_context=(f"Histórico:\nUsuário: \"{previous_question}\"\nChatbot: \"{previous_response}\"\n\n")
            ).strip()
        else:
            base_prompt = INTENT_CLASSIFIER_PROMPT.format(
                current_text=current_text,
                history_context=""
            ).strip()


        # Wrap in ChatML-style start_of_turn markers
        prompt = (
            f"<start_of_turn>user\n{base_prompt}\n<end_of_turn>"
            "\n<start_of_turn>model\n"
        )

        print(prompt)

        resp = self.helper_llm(prompt, max_tokens=2048, temperature=1.0)

        print(resp)

        full_text = resp["choices"][0]["text"].strip().lower()

        self.logger.info(f"Intent classifier response: {full_text}")

        import re

        match = re.search(r"TIPO:\s*(\w+)", full_text, re.IGNORECASE)
        if match:
            intent = match.group(1).lower()
        else:
            intent = "domain_query"

        if intent != "domain_query":
            self._non_domain_hashes.add(self._hash_text(current_text))

        return intent

    def expand_query(self, user_query: str) -> str:
        """Asks the helper model for alternative phrasings of the user question."""
        if not self.helper_llm:
            return ""

        prompt = (
            f"<start_of_turn>user\n{QUERY_EXPANSION_PROMPT.format(user_query=user_query)}\n<end_of_turn>"
            "\n<start_of_turn>model\n"
        )
        resp = self.helper_llm(
            prompt,
            max_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            min_p=0.01,
        )
        return resp.get("choices", [{}])[0].get("text", "").strip()

    def contextualize_query(self, chat_history: str, current_question: str) -> str:
        """Use the helper LLM to rewrite the current question using CONTEXTUALIZE_MESSAGE_PROMPT and <start_of_turn> tags."""
        if not self.helper_llm:
            return current_question

        base_prompt = CONTEXTUALIZE_MESSAGE_PROMPT.format(
            chat_history=chat_history,
            current=current_question
        ).strip()
        prompt = (
            f"<start_of_turn>user\n{base_prompt}\n<end_of_turn>"
            "\n<start_of_turn>model\n"
        )
        print(prompt)
        try:
            resp = self.helper_llm(prompt, max_tokens=2048, temperature=0.4, top_p=0.9)
            return resp.get("choices", [{}])[0].get("text", "").strip() or current_question
        except Exception as e:
            self.logger.error(f"Contextualization helper LLM failed: {e}")
            return current_question
        
    def get_non_domain_hashes(self) -> set:
        return self._non_domain_hashes 