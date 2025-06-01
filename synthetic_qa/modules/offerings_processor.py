import re
from pathlib import Path
import logging
import json
import textwrap
from typing import Dict, Any
from .llm_client import LLMClient
from .file_processor import FileProcessor
from .utils import json_if_valid

class OfferingsProcessor:
    @staticmethod
    def detect_offerings_document(file_path: Path, config: dict) -> bool:
        """
        Detect if a file is an offerings document based on its path and filename.
        """
        # offerings folder name
        offerings_folder_name = config.get('file_processing', {}).get('turmas_folder_name', 'turmas')

        if offerings_folder_name not in str(file_path):
            return False
        
        year_period_folder_name = file_path.parent.stem

        # check if year_period_folder_name is a valid year-period
        if not re.match(r'^\d{4}-\d$', year_period_folder_name):
            return False

        # check if file name starts with acronym {3 letters}_{any}
        if not re.match(r'^[A-Z]{3}_.+$', file_path.stem):
            return False
        
        return True

    @staticmethod
    def _build_offerings_prompt(text: str) -> str:
        """
        Build the LLM prompt for extracting offerings from markdown text.
        Args:
            text: Markdown text listing university classes for a department
        Returns:
            The formatted prompt string
        """
        return (
            "You will be given Markdown text that lists university classes for a given department. Your task is to parse this text and return a single JSON object (dictionary) where each key is a component code and each value is an object describing that component, following the specified structure.\n\n"
            "Do not skip any offering. Include every offering for every component in your JSON.\n\n"
            "Return only the JSON object as your answer. Do not include any explanations, code, or scripts. Your output should be a valid JSON object matching the template below.\n\n"
            "{\n"
            "  \"<component_code_as_key>\": { // e.g., \"CIC0002\"\n"
            "    \"name\": \"<string>\", // e.g., \"FUNDAMENTOS TEÓRICOS DA COMPUTAÇÃO\"\n"
            "    \"year_period\": \"<string>\", // e.g., \"2025.1\" (same for all offerings for this component)\n"
            "    \"offerings\": [\n"
            "      {\n"
            "        \"teachers\": [\n"
            "          {\n"
            "            \"name\": \"<string>\", // e.g., \"VINICIUS RUELA PEREIRA BORGES\"\n"
            "            \"hours\": \"<integer>\", // e.g., 60 (extracted from \"(60h)\")\n"
            "          },\n"
            "          // Add more teachers here if present.\n"
            "        ],\n"
            "        \"total_hours\": \"<integer>\", // e.g. sum of all teachers' hours for this offering\n"
            "        \"schedule\": {\n"
            "          \"code\": \"<string_or_null>\", // e.g. \"24T45\" or \"3N12 4N34\"\n"
            "          \"description\": \"<string_or_null>\" // e.g. \"Segunda-feira 16:00 às 17:50; Quarta-feira 20:50 às 22:30\"\n"
            "        },\n"
            "        \"vacancies_offered\": \"<integer>\", // e.g. 50\n"
            "        \"vacancies_filled\": \"<integer>\", // e.g. 48\n"
            "        \"location\": \"<string>\" // e.g. \"PJC BT 077\" or \"2T45(PJC BT 076) 4T4523(BSA S A1 16/37)\"\n"
            "      }\n"
            "      // Add more offering objects here if present.\n"
            "    ]\n"
            "  },\n"
            "  \"<component_code_as_key_2>\": { // e.g., \"CIC0003\"\n"
            "    \"name\": \"<string>\", // e.g., \"INTRODUÇÃO AOS SISTEMAS COMPUTACIONAIS\"\n"
            "    \"offerings\": [\n"
            "      // ... offerings for CIC0003\n"
            "    ]\n"
            "  }\n"
            "  // ... more components, each as a key-value pair in the main object\n"
            "}\n\n"
            "\ntext:\n"
            f"{text}"
        )

    @staticmethod
    def extract_offerings_from_text(text: str, file_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured offerings data from a Markdown document using LLM processing.

        Args:
            text: Markdown text listing university classes for a department
            file_path: Path to the file
            config: Configuration dictionary
        Returns:
            Dictionary mapping component codes to their offerings data
        """
        logger = logging.getLogger(__name__)

        try:
            llm_client = LLMClient(config.get("providers", {}).get("offerings_extraction", {}))
            prompt = OfferingsProcessor._build_offerings_prompt(text)
            response = llm_client.generate_text(
                prompt,
                json_output=True,
                temperature=0.4
            )

            if not response:
                logger.warning("No response from LLM for offerings extraction")
                return {}

            logger.info(f"Successfully extracted offerings data using LLM for {file_path}")
            return response

        except Exception as e:
            logger.error(f"Error extracting offerings from text: {e}")
            return {}

    @staticmethod
    def offering_json_to_markdown(offering_json: dict) -> str:
        """
        Converts a component's offering JSON to a readable markdown format.
        """
        year_period = offering_json.get("year_period", "")
        offerings = offering_json.get("offerings", [])

        lines = ["## Ofertas para o período (semestre) " + year_period, ""]

        for idx, offering in enumerate(offerings, 1):
            # Teachers
            teachers = offering.get("teachers", [])
            if len(teachers) == 1:
                teachers_str = teachers[0].get('name', '')
            else:
                teachers_str = ", ".join(f"{t.get('name', '')} ({t.get('hours', '')}h)" for t in teachers)
            docente_label = "Docente" if len(teachers) == 1 else "Docentes"
            total_hours = sum(t.get('hours', 0) for t in teachers)
            schedule = offering.get("schedule", {})
            schedule_code = schedule.get("code", "")
            schedule_desc = schedule.get("description", "")
            vacancies_offered = offering.get("vacancies_offered", "")
            vacancies_filled = offering.get("vacancies_filled", "")
            location = offering.get("location", "")

            lines.append(f"{idx}. **{docente_label}:** {teachers_str}  ")
            lines.append(f"   **Carga horária:** {total_hours}h  ")
            lines.append(f"   **Horário:** {schedule_code} ({schedule_desc}) ")
            lines.append(f"   **Local:** {location}")
            lines.append(f"   **Vagas ofertadas:** {vacancies_offered}  ")
            lines.append(f"   **Vagas preenchidas:** {vacancies_filled}  ")

        return "\n".join(lines).strip()
