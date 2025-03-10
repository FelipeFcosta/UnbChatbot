"""
Utility functions for the Synthetic QA Generator.
"""

import hashlib
from pathlib import Path
from typing import List, Dict


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