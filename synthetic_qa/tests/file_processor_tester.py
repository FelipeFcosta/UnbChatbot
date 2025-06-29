import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic_qa.modules.file_processor import FileProcessor
from synthetic_qa.modules.utils import FileType


def build_config(use_llm: bool) -> Optional[Dict[str, Any]]:
    """Return a minimal config dict understood by FileProcessor.

    If *use_llm* is True, we provide a dummy provider so that the pipeline will
    attempt to invoke ``LLMClient``. This is useful to exercise that branch of
    the code even when no real API keys are set. Otherwise, we return *None* so
    the extractor skips any LLM post-processing.
    """
    if not use_llm:
        return None
    # The concrete values do not matter for the test; we just need *provider* to
    # be something other than "none" so that FileProcessor instantiates
    # ``LLMClient`` (which will warn that the provider is unknown and then exit
    # quietly).
    return {
        "providers": {
            "text_extraction": {
                "provider": "mock",  # triggers LLM path without real API
                "model": "dummy-model"
            }
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick command-line tester for the FileProcessor component."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Local path to the HTML or PDF file you want to process."
    )
    parser.add_argument(
        "--file-type",
        default="regular",
        choices=[ft.value for ft in FileType],
        help="Logical file type hint passed to FileProcessor (default: regular)"
    )
    parser.add_argument(
        "--llm",
        choices=["none", "mock", "real"],
        default="none",
        help="Choose how to exercise the LLM correction step:\n"
             "  none  – do not run LLM (default)\n"
             "  mock  – run pipeline with a dummy provider just to cover code path\n"
             "  real  – load provider configuration from synthetic_qa/config.yaml\n"
             "         (requires valid GEMINI_API_KEY* environment variables)."
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity for the underlying modules."
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    file_path: Path = args.input_path.expanduser().resolve()
    if not file_path.exists():
        parser.error(f"File not found: {file_path}")

    processor = FileProcessor()

    # 1) preprocess_html if applicable
    if file_path.suffix.lower() in {".html", ".htm"}:
        print("[INFO] Running preprocess_html() …", file=sys.stderr)
        _ = FileProcessor.preprocess_html(file_path)
        print("[INFO] preprocess_html() completed", file=sys.stderr)

    # 2) main extraction (either HTML or PDF pathway internally)
    if args.llm == "real":
        # Load the full project configuration file and pass it through so the
        # processor can pick up providers.text_extraction exactly as used in
        # production.
        config_path = PROJECT_ROOT / "synthetic_qa" / "config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            parser.error(f"Failed to load config.yaml at {config_path}: {e}")
    else:
        config = build_config(args.llm == "mock")
    file_type_enum = FileType(args.file_type)
    print("[INFO] Running extract_text_from_file() …", file=sys.stderr)
    extracted_text = processor.extract_text_from_file(file_path, file_type_enum, config)
    print("[INFO] extract_text_from_file() completed\n", file=sys.stderr)

    temp_dir = Path(__file__).parent / "temp_data"
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = file_path.stem.replace(" ", "_")
    output_txt_path = temp_dir / f"{safe_stem}_extracted.txt"
    output_txt_path.write_text(extracted_text, encoding="utf-8")

    print(extracted_text)


if __name__ == "__main__":
    main() 