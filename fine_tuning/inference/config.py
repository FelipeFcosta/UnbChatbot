"""Centralized configuration constants for the UnB chatbot project.

Any module can simply `import config` (or use `from config import SOME_NAME`) to access
these values. Keeping them in one place avoids long blocks of settings scattered
throughout the codebase.
"""

# System-wide prompts -----------------------------------------------------
SYSTEM_PROMPT = (
    "You are a specialized UnB (Universidade de Brasília) chatbot assistant who answers questions based on the retrieved context (documents).\n"
    "Be precise and factual according to the source material when responding to the user's question. **Do not make up information.**\n"
    "Base the answer only on the correct retrieved contexts and their corresponding metadata (<doc_metadata>), filtering out irrelevant or unrelated chunks.\n"
    "Make sure the chunks used to answer correspond directly to the same entity or concept in the question.\n"
    "\n"
    "You task is to understand the user's question, their intent, and then answer the question correctly based on the context retrieved.\n\n"
    "The answer may depend on information from more than one chunk\n\n"
    "**The answer MUST DIRECTLY answer the user's question and specific needs, do not provide unrelated information**.\n\n"
    "When including a source, use the URL field in <doc_metadata> for the corresponding context chunk(s).\n"
    "\n"
    "Do not engage in user queries that are not related to UnB or require more than pure factual information.\n"
    "If the context information is not enough to answer the question, say you don't have the information (it doesn't mean it doesn't exist).\n"
    "**Do not mention the existence of the context documents in the ANSWER**, since the user is not aware of them.\n"
    "\n"
    "\n"
    "Respond in the following format:\n"
    "<REASON>\n"
    "Reasoning in English... (you may quote the relevant context information verbatim to ground your response)\n"
    "</REASON>\n"
    "<ANSWER>\n"
    "Answer in **Portuguese**... (directly answer the question while ignoring irrelevant context information)\n"
    "</ANSWER>\n"
)

QUERY_EXPANSION_PROMPT = (
    "Você é um especialista em elaboração de mensagens alternativas para melhorar a recuperação sistemas RAG de um chatbot de uma universidade. Gere **apenas 3** mensagens alternativas com base na mensagem original.\n"
    "O objetivo é melhorar a recuperação de documentos relacionados com a intenção da mensagem, não expanda nenhuma sigla que não conheça.\n"
    "NÃO RESPONDA À PERGUNTA\n"
    "**Forneça apenas as mensagens alternativas**, sem nenhum prefixo, cada uma em uma nova linha.\n"
    "Mensagem original: {user_query}\n"
    "Mensagens alternativas:"
)

# Model & storage settings -------------------------------------------------
APP_NAME = "unb-chatbot-raft-gguf-web-endpoint"

MODEL_DIR_IN_VOLUME = "unb_raft_gemma12b_neg_run1"
CHECKPOINT_FOLDER = ""  # e.g. "checkpoint-201" for resumed training
HELPER_LLM_MODEL_DIR_IN_VOLUME = "gemma_gguf_model"
GGUF_FILENAME = "merged_model.Q8_0.gguf"

VOLUME_NAME = "faq-unb-chatbot-gemma-raft"
HELPER_LLM_VOLUME_NAME = "gemma3n-gguf-converter"
DATA_VOLUME_NAME = "faq-unb-chatbot-gemma-raft-data"

GPU_CONFIG = "A100-40GB"
MODEL_MOUNT_PATH = "/model_files"
HELPER_LLM_MODEL_MOUNT_PATH = "/helper_llm_model_files"
DATA_MOUNT_PATH = "/data"

# Context window for the main LLM (in tokens)
CONTEXT_SIZE = 10_000

# Retrieval / embedding ----------------------------------------------------
SOURCE_DOCUMENTS = f"{DATA_MOUNT_PATH}/source_json_combined.json"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
TOP_K_RETRIEVAL = 10  # Number of chunks to retrieve

# Generation defaults ------------------------------------------------------
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

# Misc ---------------------------------------------------------------------
MINUTES = 60  # Convenience alias for timeouts, if needed 