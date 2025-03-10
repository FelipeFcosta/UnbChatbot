"""
File processing module for the Synthetic QA Generator.

This module handles extraction of text from different file types.
"""

import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup

# Optional dependency for PDF processing
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileProcessor:
    """Handles extraction of text from different file types."""
    
    @staticmethod
    def extract_text_from_html(file_path: Path, preserve_structure: bool = True) -> str:
        """
        Extract readable text content from HTML files.
        
        Args:
            file_path: Path to the HTML file
            preserve_structure: Whether to preserve heading structure with markdown-style markers
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove all script, style, nav, footer, header elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'meta', 
                               'form', 'iframe']):
                element.extract()
                
            # Try to remove navigation, footers and other non-content areas by common class/id names
            for selector in ['.sp-megamenu-parent', '.breadcrumb', '.article-footer-wrap',
                            '#sp-bottom', '#sp-footer', '#sp-menu', '#sp-top-bar', '.offcanvas-menu',
                            '.pager', '.helix-social-share', '.sp-copyright']:
                for element in soup.select(selector):
                    element.extract()
            
            # Try to identify and extract only the main content area if possible
            main_content = None
            for selector in ['article', 'main', '#sp-component', '.entry-content', '.itemBody', '[itemprop="articleBody"]', '.item-page']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If we found a main content area, use that; otherwise use the whole body
            content_to_process = main_content if main_content else soup
            
            if preserve_structure:
                # Convert heading tags to structured text with markers
                for heading in content_to_process.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    level = int(heading.name[1])
                    marker = '#' * level + ' '
                    # Get the heading text and create a marker prefix
                    heading_text = heading.get_text().strip()
                    formatted_text = f"\n\n{marker}{heading_text}\n\n"
                    # Replace with a new string using insert_before and decompose
                    heading.insert_before(formatted_text)
                    heading.decompose()
            
            # Process links to convert them to markdown format [text](url)
            for link in content_to_process.find_all('a'):
                if link.get('href'):
                    link_text = link.get_text().strip()
                    link_url = link.get('href')
                    # Handle relative URLs
                    if link_url.startswith('/'):
                        base_url = "https://www.ene.unb.br"  # Default base URL
                        if link.get('base'):
                            base_url = link.get('base')
                        link_url = base_url + link_url
                    
                    # Create markdown-style link
                    markdown_link = f"[{link_text}]({link_url})"
                    link.replace_with(markdown_link)
                else:
                    # If no href, just use the text
                    link_text = link.get_text().strip()
                    link.replace_with(link_text)
            
            # Extract text while preserving some formatting
            html_text = str(content_to_process)
            
            # Replace <p>, <div>, <br> tags with newlines but avoid adding too many
            html_text = re.sub(r'<br\s*/?>|</p>|</div>', '\n', html_text)
            html_text = re.sub(r'<p[^>]*>|<div[^>]*>', '\n', html_text)
            
            # Remove all remaining HTML tags
            html_text = re.sub(r'<[^>]*>', '', html_text)
            
            # Decode HTML entities
            html_text = BeautifulSoup(html_text, 'html.parser').get_text()
            
            # Process the text to normalize whitespace
            lines = []
            for line in html_text.split('\n'):
                line = line.strip()
                lines.append(line)
            
            # Join with appropriate spacing
            text = '\n'.join(lines).strip()
            
            # Replace multiple consecutive newlines with just two
            for _ in range(0,3):
                text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """Extract text content from PDF files."""
        if not PDF_AVAILABLE:
            logger.error("PyMuPDF (fitz) not available. Cannot extract text from PDF.")
            return ""
            
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_file(file_path: Path, preserve_structure: bool = True) -> str:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
            preserve_structure: Whether to preserve document structure (for HTML files)
        """
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return ""
            
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.html', '.htm']:
            return FileProcessor.extract_text_from_html(file_path, preserve_structure=preserve_structure)
        elif file_extension == '.pdf':
            return FileProcessor.extract_text_from_pdf(file_path)
        elif file_extension in ['.txt', '.md', '.csv', '.json']:
            try:
                return file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return ""
        else:
            logger.warning(f"Unsupported file format: {file_extension} for {file_path}")
            return ""