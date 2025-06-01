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

        # check if componente_codigo is an attribute of the file
        try:
            componente_codigo = os.getxattr(str(file_path), b'user.componente_codigo').decode('utf-8')
            if componente_codigo:
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
    def add_course_offerings_to_text(text: str, component_code: str, output_dir: Path) -> str:
        """
        Add course offerings to the text for a given component code by looking up the extracted_offerings JSON.
        """
        import json

        extracted_offerings_dir = output_dir / "extracted_offerings"
        if extracted_offerings_dir.exists():
            for file in extracted_offerings_dir.iterdir():
                if file.is_file() and file.suffix == ".json":
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            offerings_json = json.load(f)
                        component_obj = offerings_json.get(component_code)
                        if component_obj is not None:
                            text += "\n\n---\n" + OfferingsProcessor.offering_json_to_markdown(component_obj)
                            break
                        else:
                            period = next(iter(offerings_json.values()))["year_period"]
                            logger.warning(f"No offerings found for component {component_code}")
                            # add NÃO HÁ OFERTAS para o período (semestre) 
                            text += "\n\n---\n" + "NÃO HÁ OFERTAS para o período (semestre) " + period
                    except Exception as e:
                        logger.error(f"Error reading offerings file {file} for component {component_code}: {e}")

        text = re.sub(r'```markdown\s*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
        return text