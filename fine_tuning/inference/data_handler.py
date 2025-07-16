
import os
import json
from typing import List, Dict

from .config import SOURCE_DOCUMENTS, EMBEDDING_MODEL_NAME

class DataHandler:
    def __init__(self, logger):
        self.documents: List[str] = []
        self.pairs: List[Dict] = []
        self.regulars: List[Dict] = []
        self.retriever = None
        self.embedding_model = None
        self.logger = logger

    def load_data(self):
        """Loads original FAQ data and populates documents lists."""
        self.logger.info(f"Loading original FAQ data from {SOURCE_DOCUMENTS}...")
        if not os.path.exists(SOURCE_DOCUMENTS):
            error_msg = f"Original FAQ data file not found at {SOURCE_DOCUMENTS}."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        with open(SOURCE_DOCUMENTS, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                doc_type = item.get("file_type", "")
                file_title = item.get("file_title", "")
                source_page_url = item.get("source_page_url", "")

                if doc_type == 'FileType.FAQ':
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    topics = item.get("topics")
                    file_url = item.get("file_url")
                    
                    topic_str = f'Topic: "{", ".join(topics)}", ' if topics else ''
                    filename_str = f'File: "{item.get("file_name", "")}", '
                    url_str = f'URL: "{file_url}"'
                    
                    formatted_item = f'Q: "{question}", A: "{answer}"<doc_metadata>{topic_str}{filename_str}{url_str}</doc_metadata>\n'
                    self.pairs.append(formatted_item)
                    self.documents.append(formatted_item)

                else:
                    chunk = item.get("chunk", "")
                    topics = item.get("topic")
                    professor = item.get("professor")
                    course = item.get("course")
                    file_name = item.get("file_name", "")
                    file_url = item.get("file_url")
                    
                    topic_str = f'Topic: "{topics}", ' if topics else ''
                    professor_str = f'Professor: "{professor}", ' if professor else ''
                    course_str = f'Course: "{course}", ' if course else ''
                    filename_str = f'File: "{file_name}", '
                    
                    is_html_file = file_name.lower().endswith((".html", ".htm"))
                    if not is_html_file and source_page_url:
                        url_str = f'URLs: "{source_page_url} [{file_title}]({file_url})"'
                    else:
                        url_str = f'URL: "[{file_title}]({file_url})"'
                    
                    formatted_item = f'Chunk: "{chunk}"<doc_metadata>\n{topic_str}{professor_str}{course_str}{filename_str}{url_str}\n</doc_metadata>'
                    self.regulars.append(formatted_item)
                    self.documents.append(formatted_item)
        
        if self.pairs:
            self.logger.info(f"Example FAQ: {self.pairs[0]}")
        if self.regulars:
            self.logger.info(f"Example Regular: {self.regulars[0]}")

    def build_index(self):
        """Builds the FAISS index from loaded documents."""
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if not self.documents:
            self.logger.warning("No documents loaded. Skipping index build.")
            return

        self.logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')
            self.logger.info("Embedding model loaded. Generating embeddings for documents...")
            embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False, show_progress_bar=True)
            self.logger.info(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

            self.logger.info("Building FAISS index...")
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            self.retriever = index
            self.logger.info("FAISS index built successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model or build FAISS index: {e}")
            raise

    def load_and_index_data(self):
        """Loads original FAQ data and builds the FAISS index."""
        self.load_data()
        self.build_index()

    def retrieve_context(self, query: str, k: int) -> List[str]:
        """Retrieve top-k relevant document chunks for the query."""
        import faiss
        import numpy as np

        if self.retriever is None or self.embedding_model is None:
            self.logger.error("Retriever or embedding model not initialized.")
            return []
        try:
            self.logger.info(f"Encoding query for retrieval...")
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_embedding_np = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding_np)

            self.logger.info(f"Searching FAISS index for top {k} documents...")
            distances, indices = self.retriever.search(query_embedding_np, k)
            self.logger.info(f"Retrieved indices: {indices[0]}")

            retrieved_docs = [self.documents[i] for i in indices[0]]
            return retrieved_docs
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}", exc_info=True)
            return [] 