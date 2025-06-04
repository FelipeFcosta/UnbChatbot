import re
import os
from pathlib import Path
from modules.offerings_processor import OfferingsProcessor
import logging

logger = logging.getLogger(__name__)

class ComponentProcessor:
    @staticmethod
    def detect_component_document(file_path, config):
        """
        Detect if a file is a component document based on its path and filename.
        - The path must contain the 'componentes_curriculares' folder (from config.yaml).
        - The filename must match the pattern: {SIGLA}{CÓDIGO}_{QUALQUER_NOME}
        """
        folder_name = config.get('file_processing', {}).get('componentes_folder_name', 'componentes_curriculares')

        # check if component_code is an attribute of the file
        try:
            component_code = os.getxattr(str(file_path), b'user.component_code').decode('utf-8')
            if component_code:
                return True
        except:
            pass

        # Check if folder is in the path
        if folder_name not in str(file_path):
            return False

        # Check filename pattern
        filename = Path(file_path).stem

        component_pattern = r'^[A-Z]{3}\d{4}_.+'
        return re.match(component_pattern, filename) is not None 
    
    @staticmethod
    def add_course_offerings_to_text(text: str, component_code: str, rel_path: Path, output_dir: Path, config: dict) -> str:
        """
        Add course offerings to the text for a given component code by looking up the extracted_offerings JSON.
        """
        import json

        offerings_folder_name = config.get('file_processing', {}).get('offerings_folder_name', 'turmas')

        # 'subtract' rel_path parts from output_dir
        filtered_parts = [part for part in output_dir.parts if part not in rel_path.parts]
        offerings_path = Path(*filtered_parts) / offerings_folder_name

        if offerings_path.exists():
            for year_period_folder in offerings_path.iterdir():
                if not year_period_folder.is_dir():
                    continue
                extracted_offerings_path = year_period_folder / "extracted_offerings"
                if not (extracted_offerings_path.exists() and extracted_offerings_path.is_dir()):
                    continue
                for file in extracted_offerings_path.iterdir():
                    if file.is_file() and file.suffix == ".json":
                        # get attribute from offering
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                offerings_json = json.load(f)
                            component_obj = offerings_json.get(component_code)
                            if component_obj is not None:
                                text += "\n\n---\n" + OfferingsProcessor.offering_json_to_markdown(component_obj)
                                break
                            else:
                                period = next(iter(offerings_json.values()))["year_period"]
                                if next(iter(offerings_json.keys()))[:3] == component_code[:3]: # if same unit acr
                                    logger.warning(f"No offerings found for component {component_code} for period {period}")
                                    text += "\n\n---\n" + "NÃO HÁ OFERTAS para o período (semestre) " + period
                        except Exception as e:
                            logger.error(f"Error reading offerings file {file} for component {component_code}: {e}")

        text = re.sub(r'```markdown\s*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
        return text
