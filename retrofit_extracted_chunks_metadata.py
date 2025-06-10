import os
import json
from pathlib import Path
from synthetic_qa.modules.file_processor import FileProcessor
from synthetic_qa.modules.utils import FileType

def add_metadata_to_chunks(extracted_chunks_dir):
    extracted_chunks_dir = Path(extracted_chunks_dir)
    updated_dir = extracted_chunks_dir / "updated"
    updated_dir.mkdir(exist_ok=True)
    for json_file in extracted_chunks_dir.glob("*.json"):
        print(f"Processing {json_file}...")
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not chunks:
            continue

        # Try to infer file_path from the json filename
        file_name = json_file.stem
        file_title = file_name
        file_type = FileType.REGULAR  # Default
        file_url = ""
        source_page_url = ""

        # Try to find the original file (best effort)
        possible_extensions = [".html", ".htm", ".pdf", ".txt", ".md", ".docx", ".doc"]
        found_file_path = None
        for ext in possible_extensions:
            candidate = json_file.with_suffix(ext)
            if candidate.exists():
                found_file_path = candidate
                break

        if found_file_path:
            file_title = found_file_path.stem
            file_name = found_file_path.name
            try:
                file_url = os.getxattr(str(found_file_path), b'user.original_url').decode('utf-8')
            except Exception:
                file_url = FileProcessor.extract_domain_and_path(found_file_path)[2]
            try:
                source_page_url = os.getxattr(str(found_file_path), b'user.source_page_url').decode('utf-8')
            except Exception:
                source_page_url = ""
            # file_type could be improved if you have logic for this
        else:
            file_name = json_file.stem
            file_title = file_name
            file_url = ""
            source_page_url = ""
            file_type = FileType.REGULAR

        for chunk in chunks:
            chunk.setdefault("file_title", file_title)
            chunk.setdefault("file_name", file_name)
            chunk.setdefault("file_url", file_url)
            chunk.setdefault("source_page_url", source_page_url)
            chunk.setdefault("file_type", str(file_type))

        updated_json_file = updated_dir / json_file.name
        with open(updated_json_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Updated {updated_json_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python retrofit_extracted_chunks_metadata.py <extracted_chunks_dir>")
        exit(1)
    add_metadata_to_chunks(sys.argv[1]) 