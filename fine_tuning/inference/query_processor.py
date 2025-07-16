
import logging
from typing import Dict, List
from .config import INTENT_CLASSIFIER_PROMPT, QUERY_EXPANSION_PROMPT, CONTEXTUALIZE_MESSAGE_PROMPT

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

        history_context = ""
        if previous_question and previous_response:
            history_context = (
                "Histórico:\n"
                f"Usuário: \"{previous_question}\"\n"
                f"Chatbot: \"{previous_response}\"\n\n"
            )

        prompt = INTENT_CLASSIFIER_PROMPT.format(
            current_text=current_text,
            history_context=history_context
        )

        resp = self.helper_llm(prompt, max_tokens=16, temperature=0.1)
        intent = resp["choices"][0]["text"].strip().lower()

        if intent != "domain_query":
            self._non_domain_hashes.add(self._hash_text(current_text))

        return intent

    def expand_query(self, user_query: str) -> str:
        """Asks the helper model for alternative phrasings of the user question."""
        if not self.helper_llm:
            return ""

        expansion_resp = self.helper_llm(
            QUERY_EXPANSION_PROMPT.format(user_query=user_query),
            max_tokens=512,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            min_p=0.01,
        )

        if expansion_resp and expansion_resp.get("choices"):
            return expansion_resp["choices"][0]["text"]
        return ""

    def contextualize_query(self, chat_history: str, current_question: str) -> str:
        """Use the helper LLM to rewrite the current question to be self-contained."""
        if not self.helper_llm:
            return current_question

        prompt = CONTEXTUALIZE_MESSAGE_PROMPT.format(
            chat_history=chat_history,
            current=current_question,
        )

        try:
            resp = self.helper_llm(prompt, max_tokens=128, temperature=0.3, top_p=0.9)
            if resp and resp.get("choices"):
                rewritten = resp["choices"][0]["text"].strip()
                return rewritten or current_question
        except Exception as e:
            self.logger.error(f"Contextualization helper LLM failed: {e}")

        return current_question
        
    def get_non_domain_hashes(self) -> set:
        return self._non_domain_hashes 