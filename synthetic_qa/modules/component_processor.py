import os
import re
from pathlib import Path

from modules.utils import create_hash
from slugify import slugify
from modules.file_processor import FileProcessor

class ComponentProcessor:
    @staticmethod
    def detect_component_document(file_path, config):
        """
        Detect if a file is a component document based on its path and filename.
        - The path must contain the 'componentes_curriculares' folder (from config.yaml).
        - The filename must match the pattern: {SIGLA}{CÓDIGO}_{QUALQUER_NOME}
        """
        folder_name = config.get('file_processing', {}).get('componentes_folder_name', 'componentes_curriculares')

        # Check if folder is in the path
        if folder_name not in str(file_path):
            return False

        # Check filename pattern
        filename = Path(file_path).stem

        component_pattern = r'^[A-Z]{3}\d{4}_.+'
        return re.match(component_pattern, filename) is not None 
    
    @staticmethod
    def add_course_offerings_to_text(text: str, acronym: str, output_dir: Path, config: dict) -> str:
        """
        Add course offerings to the text.
        """
        classes_folder_name = config.get('file_processing', {}).get('offerings_folder_name', 'turmas')

        base_dir = config.get('base_dir', '')

        classes_folder_path = Path(base_dir) / classes_folder_name

        # search files that start with acronym in the 'turmas'/{year-period}/ folder
        for year_period_folder in classes_folder_path.iterdir():
            if year_period_folder.is_dir():
                for file in year_period_folder.iterdir():
                    # get structured text from file
                    rel_path = file.relative_to(base_dir)
                    file_title = file.stem
                    soup = FileProcessor.get_soup(extracted_text_path)
                    if soup and soup.title:
                        file_title = soup.title.get_text(strip=True)
                    file_hash = create_hash(str(rel_path))
                    safe_title_slug = slugify(file_title)

                    extracted_text_path = output_dir / "extracted_text" / f"{safe_title_slug}_{file_hash}.txt"

                    if os.path.exists(extracted_text_path):
                        with open(extracted_text_path, 'r', encoding='utf-8') as f:
                            text += f.read()

                    if file.is_file() and file.stem.startswith(acronym):
                        text += f"\n\n{file.read_text()}"
                        break

        return text