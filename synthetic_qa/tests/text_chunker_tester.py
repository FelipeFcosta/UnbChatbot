import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any
import yaml

# Ensure the project root (two directories up) is on PYTHONPATH so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic_qa.modules.text_chunker import TextChunker


def load_app_config() -> Dict[str, Any]:
    """Load the full application configuration from synthetic_qa/config.yaml."""

    config_path = PROJECT_ROOT / "synthetic_qa" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick command-line tester for the TextChunker component using REAL LLM provider."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Local path to a text file whose content should be chunked."
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

    # Read the entire text content
    try:
        extracted_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback if the text file is not UTF-8 encoded
        extracted_text = file_path.read_text(encoding="latin-1")

    # Load full config so we use REAL LLM provider for text_chunking
    config = load_app_config()

    # Instantiate TextChunker with the real configuration
    chunker = TextChunker(config)

    print("[INFO] Running TextChunker.chunk_text() …", file=sys.stderr)
    chunks = chunker.chunk_text(extracted_text, file_path)
    print("[INFO] TextChunker.chunk_text() completed\n", file=sys.stderr)

    # 3) save chunks to temp_data directory for further testing/inspection
    temp_dir = Path(__file__).parent / "temp_data"
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = file_path.stem.replace(" ", "_")
    output_json_path = temp_dir / f"{safe_stem}_chunks.json"
    output_json_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] Chunk JSON written to {output_json_path}", file=sys.stderr)

    # Pretty-print the resulting chunks
    print(json.dumps(chunks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main() 