"""
Utility functions for the Synthetic QA Generator.
"""

import hashlib
import logging
import time
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import html # Added for html.escape
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), '../config.yaml'), 'r', encoding='utf-8') as f:
    _config = yaml.safe_load(f)
logging_level = getattr(logging, _config.get('global', {}).get('logging_level', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# PDF extraction utility
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. PDF processing will be disabled.")

def create_hash(text: str) -> str:
    """
    Generate a short hash for a text string.
    
    Args:
        text: The text to hash
        
    Returns:
        A short hash string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


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


def sanitize_json_string(json_string: str) -> str:
    """
    Sanitizes a potentially problematic JSON string to make it parseable.
    
    Args:
        json_string: The JSON string to sanitize
        
    Returns:
        Sanitized JSON string
        
    Raises:
        ValueError: If the JSON string cannot be sanitized
    """
    # First attempt: Try direct parsing
    try:
        return json.dumps(json.loads(json_string), indent=2)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to sanitize JSON string: {e}")
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
    

def json_if_valid(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Extract and parse JSON from text if it contains valid JSON.
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        Parsed JSON object or None if no valid JSON found
    """
    if not text:
        return None
        
    # Find JSON-like pattern in the text
    json_pattern = r'(\{[\s\S]*\})'
    json_match = re.search(json_pattern, text)
    
    if not json_match:
        return None

    # Check if there's more than one JSON object in the text
    if len(re.findall(json_pattern, text)) > 1:
        return None  
    
    json_text = json_match.group(1)

    try:
        # Try to sanitize and parse the JSON
        sanitized = sanitize_json_string(json_text)
        data = json.loads(sanitized)
        return data
    except (json.JSONDecodeError, ValueError):
        return None



class RateLimiter:
    """
    A flexible rate limiter that works across different models and providers.
    Tracks request timestamps to enforce a configurable requests per minute limit.
    """
    def __init__(self, rate_limit_rpm: Optional[int] = None, model_name: Optional[str] = None):
        """
        Initialize the rate limiter.
        
        Args:
            rate_limit_rpm: Requests per minute limit. If None, no rate limiting is applied.
            model_name: Name of the model for more informative logging
        """
        # Stores the timestamps of recent requests
        self._request_timestamps = []
        
        # Rate limit in requests per minute
        self._rate_limit_rpm = rate_limit_rpm
        
        # Model name for logging
        self._model_name = model_name or "Unknown Model"

    def wait(self):
        """
        Wait if necessary to comply with the rate limit.
        If no rate limit is set, this method does nothing.
        
        Returns:
            Float representing wait time in seconds, or 0 if no waiting is required
        """
        # If no rate limit is set, immediately return
        if self._rate_limit_rpm is None:
            return 0.0

        current_time = time.time()

        # Remove timestamps outside the last minute
        self._request_timestamps = [
            ts for ts in self._request_timestamps 
            if current_time - ts < 60
        ]

        # Check if we've reached the rate limit
        if len(self._request_timestamps) >= self._rate_limit_rpm:
            # Calculate how long to wait
            oldest_request_time = self._request_timestamps[0]
            wait_time = 60 - (current_time - oldest_request_time)
            
            if wait_time > 0:
                # Log the waiting information
                logger.info(
                    f"Rate limit reached for {self._model_name} "
                    f"(Limit: {self._rate_limit_rpm} requests/minute). "
                    f"Waiting {wait_time:.2f} seconds before next request."
                )
                
                time.sleep(wait_time)
                current_time = time.time()
                
                # After waiting, log that we're ready to proceed
                logger.info(
                    f"Wait period complete for {self._model_name}. "
                    "Ready to make the next request."
                )
                
                return wait_time
            
        # Record this request's timestamp
        self._request_timestamps.append(current_time)
        
        return 0.0


def extract_html_from_pdf(file_path: Path) -> str:
    """
    Extract styled HTML content from a PDF file, preserving text styling,
    basic structure (paragraphs), and inline links as <a> tags.
    Args:
        file_path: Path to the PDF file
    Returns:
        Extracted HTML content as a string
    """
    if not PDF_AVAILABLE:
        logger.error("PyMuPDF (fitz) not available. PDF processing will be disabled.")
        return ""
    try:
        all_pages_html_parts = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                all_pages_html_parts.append(f"<!-- Page {page_num + 1} -->\n")
                page_links = page.get_links()  # Get links once per page for efficiency
                
                current_link_uri_on_page = None # Stores the URI of the currently open <a> tag on this page

                def get_uri_for_span_bbox(span_bbox_rect):
                    """Checks if a given span's bounding box intersects with any link on the page."""
                    for link_info in page_links:
                        if "uri" in link_info:
                            link_rect_on_page = fitz.Rect(link_info["from"])
                            if link_rect_on_page.intersects(span_bbox_rect):
                                return link_info["uri"]
                    return None

                blocks = page.get_text("dict", sort=True)["blocks"]
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        # Start a paragraph for each text block
                        block_html_content = "<p>"
                        for line in block["lines"]:
                            for span in line["spans"]:
                                span_text_content = span["text"]
                                span_bbox = fitz.Rect(span["bbox"])
                                
                                # Determine link status for this span
                                uri_for_current_span = get_uri_for_span_bbox(span_bbox)
                                
                                # Manage link state: open or close <a> tags
                                if uri_for_current_span:
                                    if current_link_uri_on_page != uri_for_current_span:  # New link or different link
                                        if current_link_uri_on_page is not None:
                                            block_html_content += "</a>"  # Close previous link
                                        block_html_content += f'<a href="{html.escape(uri_for_current_span, quote=True)}">'
                                        current_link_uri_on_page = uri_for_current_span
                                # Else (still in the same link), do nothing regarding <a> tag opening
                                else:  # Current span is not part of any link
                                    if current_link_uri_on_page is not None:
                                        block_html_content += "</a>"  # Close active link
                                        current_link_uri_on_page = None
                                
                                # Apply styling for the span
                                styles = []
                                font_name = span['font']
                                styles.append(f"font-family:'{html.escape(font_name)}'")
                                styles.append(f"font-size:{span['size']:.1f}pt") # Using pt for font size
                                
                                srgb_color = span['color']
                                r_val, g_val, b_val = (srgb_color >> 16) & 0xff, (srgb_color >> 8) & 0xff, srgb_color & 0xff
                                styles.append(f"color:#{r_val:02x}{g_val:02x}{b_val:02x}")

                                if span['flags'] & 16:  # Bold
                                    styles.append("font-weight:bold")
                                if span['flags'] & 2:  # Italic
                                    styles.append("font-style:italic")
                                # Note: PyMuPDF flags: Bit 0: Superscript, Bit 1: Italic, Bit 2: Serif, Bit 3: Monospaced, Bit 4: Bold
                                
                                escaped_text = html.escape(span_text_content)
                                
                                block_html_content += f'<span style="{"; ".join(styles)}">{escaped_text}</span>'
                            # Spans within a line and lines within a block flow together in the <p>
                            # If explicit line breaks are needed within a <p>, add <br> after line loop.

                        # Close link if it's open at the end of a text block
                        if current_link_uri_on_page is not None:
                            block_html_content += "</a>"
                            current_link_uri_on_page = None # Reset for the next block
                        
                        block_html_content += "</p>\n"
                        all_pages_html_parts.append(block_html_content)
                    # elif block["type"] == 1: # Image block
                        # Placeholder for image handling if needed in the future
                        # e.g., extract image, save to a file, and embed as <img> tag.

                # Close link if it's still open at the end of a page
                if current_link_uri_on_page is not None:
                    all_pages_html_parts.append("</a>")
                    # current_link_uri_on_page is implicitly reset at the start of the next page loop
            
        return "".join(all_pages_html_parts)
    except Exception as e:
        logger.error(f"Error extracting styled HTML from PDF {file_path}: {e}")
        return ""
