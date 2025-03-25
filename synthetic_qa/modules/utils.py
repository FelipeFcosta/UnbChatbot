"""
Utility functions for the Synthetic QA Generator.
"""

import hashlib
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup
import json
import re


def get_hash(text: str) -> str:
    """
    Generate a short hash for a text string.
    
    Args:
        text: The text to hash
        
    Returns:
        A short hash string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def group_related_files(files: List[Path], base_dir: Path) -> List[List[Path]]:
    """
    Group related files together based on directory structure and naming patterns.
    
    Args:
        files: List of file paths
        base_dir: Base directory for relative path calculation
    
    Returns:
        List of grouped file paths
    """
    # Simple grouping by directory
    dir_groups: Dict[Path, List[Path]] = {}
    
    for file_path in files:
        dir_path = file_path.parent.relative_to(base_dir) if file_path.parent != base_dir else Path(".")
        if dir_path not in dir_groups:
            dir_groups[dir_path] = []
        dir_groups[dir_path].append(file_path)
    
    # Convert to list of groups
    return list(dir_groups.values())


import json
import re

def sanitize_json_string(json_string):
    """
    Sanitizes a potentially problematic JSON string to make it parseable.
    """
    # First attempt: Try direct parsing
    try:
        return json.dumps(json.loads(json_string), indent=2)
    except json.JSONDecodeError:
        pass

    # Handle the content directly as text
    try:
        # Extract the question and answer text
        q_match = re.search(r'"q"\s*:\s*"(.*?)",', json_string, re.DOTALL)
        a_match = re.search(r'"a"\s*:\s*"(.*?)"\s*\}', json_string, re.DOTALL)
        
        if q_match and a_match:
            q_text = q_match.group(1)
            a_text = a_match.group(1).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            
            # Rebuild the JSON with properly escaped strings
            fixed_json = f'{{\n "qa_pairs": [\n {{\n "q": "{q_text}",\n "a": "{a_text}"\n }}\n ]\n}}'
            
            # Validate
            return json.dumps(json.loads(fixed_json), indent=2)
    except Exception:
        pass

    # If all else fails
    raise ValueError(f"JSON string could not be sanitized")
    

def json_if_valid(text: str):
    json_pattern = r'(\{[\s\S]*\})'
    json_match = re.search(json_pattern, text)
    
    if not json_match:
        return None

    if not json_match or len(re.findall(json_pattern, text)) > 1:
        return None
    
    json_text = json_match.group(1)

    try:
        data = json.loads(sanitize_json_string(json_text))
        return data
    except json.JSONDecodeError:
        return None
