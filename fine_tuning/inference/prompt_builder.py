
from typing import Dict, List

class PromptBuilder:
    def __init__(self, non_domain_hashes: set, logger):
        self._non_domain_hashes = non_domain_hashes
        self.logger = logger

    @staticmethod
    def _hash_text(text: str) -> str:
        """Return a stable SHA256 hex digest for a given text snippet."""
        import hashlib
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def process_history(self, messages: List[Dict]) -> List[Dict]:
        """Process messages to filter out non-domain history."""
        in_domain_history: List[dict] = []
        skip_next_assistant = False
        for m in messages[:-1]:
            role = m["role"]
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                if self._hash_text(content) in self._non_domain_hashes:
                    skip_next_assistant = True
                    self.logger.info(f"Skipping assistant reply for chitchat: {content}")
                    continue
                in_domain_history.append(m)
            else:
                if skip_next_assistant:
                    skip_next_assistant = False
                    continue
                in_domain_history.append(m)
        return in_domain_history

    def build_chat_history_str(self, in_domain_history: List[Dict]) -> str:
        """Build plain-text history string for contextualization, limited to last 3 exchanges."""
        recent_history = in_domain_history[-6:] if len(in_domain_history) > 6 else in_domain_history
        
        history_lines = []
        for m in recent_history:
            prefix = "Usuário:" if m["role"] == "user" else "Chatbot:" if m["role"] == "assistant" else f"{m['role']}:"
            history_lines.append(f"{prefix} \"{m['content'].strip()}\"")
        return "\n".join(history_lines)

    def build_prompt(self, messages: List[Dict], system_prompt: str, assembled_context_str: str, user_query: str) -> str:
        """Build Gemma-style multi-turn prompt."""
        prompt_parts: List[str] = []

        # build full chat history
        for m in messages[:-1]:
            role_tag = "user" if m["role"] == "user" else "model"
            content = (m["content"] or "").strip()
            if not content:
                continue
            prompt_parts.append(
                f"<start_of_turn>{role_tag}\n{content}\n<end_of_turn>"
            )

        # build final user turn (SYSTEM_PROMPT + DOCUMENTS + USER_QUERY)
        final_user_segments: List[str] = []
        if system_prompt:
            final_user_segments.append(system_prompt)
        if assembled_context_str:
            final_user_segments.append(assembled_context_str)
        final_user_segments.append(user_query)
        final_user_content = "\n".join(final_user_segments)
        prompt_parts.append(
            f"<start_of_turn>user\n{final_user_content}\n<end_of_turn>"
        )

        return "\n".join(prompt_parts) + "\n<start_of_turn>model\n" 