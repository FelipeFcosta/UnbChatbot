"""
File processing module for the Synthetic QA Generator.

This module handles extraction of text from different file types and source information.
"""

import logging
import requests
import time
import re
from pathlib import Path
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup, Doctype, Comment
from .llm_client import LLMClient
from typing import Dict, Any, Optional


# Optional dependency for PDF processing
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileProcessor:
    """Handles extraction of text from different file types."""
    
    # Known institutional domains and their full names
    INSTITUTION_DOMAINS = {
        'www.ene.unb.br': 'Departamento de Engenharia Elétrica',
        'ene.unb.br': 'Departamento de Engenharia Elétrica',
        'www.cic.unb.br': 'Departamento de Ciência da Computação',
        'cic.unb.br': 'Departamento de Ciência da Computação',
        'www.unb.br': 'Universidade de Brasília',
        'unb.br': 'Universidade de Brasília'
    }

    INSTITUTION_COURSES = {
        'www.ene.unb.br': 'Engenharia Elétrica, Engenharia de Redes de Comunicação, Engenharia Mecatrônica, Engenharia de Computação',
        'ene.unb.br': 'Engenharia Elétrica, Engenharia de Redes de Comunicação, Engenharia Mecatrônica, Engenharia de Computação',
        'www.cic.unb.br': 'Ciência da Computação, Engenharia de Computação, Computação (Licenciatura), Engenharia Mecatrônica',
        'cic.unb.br': 'Ciência da Computação, Engenharia de Computação, Computação (Licenciatura), Engenharia Mecatrônica'
    }
    
    @staticmethod
    def extract_domain_and_path(file_path: Path) -> tuple:
        """
        Extract domain and path information from a file path.
        
        Args:
            file_path: Path object for the file
            
        Returns:
            Tuple of (domain, path, full_url)
        """
        # Convert to string and normalize separators
        path_str = str(file_path).replace('\\', '/')
        
        # First try to extract from path structure that mimics website clone
        for domain in FileProcessor.INSTITUTION_DOMAINS.keys():
            if domain in path_str:
                # Extract everything after domain in the path
                domain_idx = path_str.find(domain)
                if domain_idx >= 0:
                    full_path = path_str[domain_idx:]
                    
                    # Get the path after the domain
                    after_domain = full_path[len(domain):]
                    path = after_domain.strip('/')
                    
                    # Handle index files - extract just the directory
                    if path.endswith('index.html') or path.endswith('index.htm'):
                        path = '/'.join(path.split('/')[:-1])
                    
                    # Remove file extension from the path
                    if path.endswith('.html') or path.endswith('.htm'):
                        path = path[:-5]  # Remove .html or .htm
                    
                    if (not domain.startswith("www")):
                        domain = f"www.{domain}"

                    return domain, path, f"{domain}/{path}"
        
        # If no known domain found, make a best guess from the path
        # Extract rightmost component that looks like a domain
        parts = path_str.split('/')
        domain = None
        
        for part in parts:
            if '.' in part and not part.endswith(('.html', '.htm', '.pdf', '.txt')):
                domain = part
                
        if domain:
            # Find the domain in the path and extract everything after it
            domain_idx = path_str.find(domain)
            if domain_idx >= 0:
                after_domain = path_str[domain_idx + len(domain):]
                path = after_domain.strip('/')
                
                # Remove file extension
                if path.endswith(('.html', '.htm')):
                    path = path[:-5]  # Remove .html or .htm
                elif path.endswith('.pdf'):
                    path = path[:-4]  # Remove .pdf
                    
                return domain, path, f"{domain}/{path}"
        
        # If all else fails, just extract the parent folder and filename
        parent = file_path.parent.name
        name = file_path.stem  # Filename without extension
        
        return "", f"{parent}/{name}", f"{parent}/{name}"
    
    @staticmethod
    def get_institution_name(domain: str) -> str:
        """
        Get the full institution name from a domain.
        
        Args:
            domain: Domain string like "cic.unb.br"
            
        Returns:
            Full institution name or the domain if not recognized
        """
        return FileProcessor.INSTITUTION_DOMAINS.get(domain, domain)


    @staticmethod
    def get_institution_courses(domain: str) -> str:
        """
        Get the string with all available courses for this institution.
        
        Args:
            domain: Domain string like "cic.unb.br"
            
        Returns:
            Full institution name or the domain if not recognized
        """
        return FileProcessor.INSTITUTION_COURSES.get(domain, "All Courses")
    
    
    @staticmethod
    def process_links_in_html(html_content: BeautifulSoup, file_path: str) -> None:
        """
        Process links in HTML content to convert them to markdown format [text](url).
        This function reads HTML from file_path and modifies the BeautifulSoup object in place.
        
        Args:
            html_content: BeautifulSoup object containing the parsed HTML content
            file_path: Path to the HTML file to process
        """        
        domain, path, url = FileProcessor.extract_domain_and_path(file_path)
        base_url = f"https://{url}"
        
        not_working_urls = []
        working_urls = []

        for link in html_content.find_all('a'):
            link_text = link.get_text().strip()
            link_url = link.get('href')
            link_url, _ = urldefrag(link_url)
            
            if not link_url:
                # If no href attribute, just use the text
                link.replace_with(link_text)
                continue
                
            # Check if the URL is relative
            parsed = urlparse(link_url)
            is_relative = not (parsed.scheme and parsed.netloc)
            
            if not is_relative:
                markdown_link = f"[{link_text}]({link_url})"
            else:
                # For relative URLs, try different variations
                url_variations = []
                complete_url = urljoin(base_url, link_url)
                if complete_url not in not_working_urls:
                    url_variations.append(complete_url) # original with .html

                # Add variation without .html if applicable
                if link_url.endswith('.html'):
                    url_without_html = urljoin(base_url, link_url.rsplit('.html', 1)[0])
                    if url_without_html not in not_working_urls:
                        url_variations.append(url_without_html)
                
                # Add variation without index.html if applicable
                if link_url.endswith('index.html'):
                    url_without_index = urljoin(base_url, link_url.rsplit('index.html', 1)[0])
                    if not url_without_index.endswith('/'):
                        url_without_index += '/'
                    if url_without_index not in not_working_urls:
                        url_variations.append(url_without_index)
                
                # Try each URL variation
                working_url = None
                for url_to_try in url_variations:
                    if url_to_try in working_urls:
                        working_url = url_to_try
                        break
                    retries = 3
                    for attempt in range(retries):
                        try:
                            response = requests.head(url_to_try, timeout=9)
                            if response.status_code == 200:
                                working_url = url_to_try
                                working_urls.append(working_url)
                                break
                            elif response.status_code in {301, 302, 303, 307, 308}:  # Redirect codes
                                if url_to_try.split("/")[-1] in urljoin(base_url, response.headers.get('Location')):
                                    url_to_try = urljoin(base_url, response.headers.get('Location'))
                                    attempts = 0
                                    continue
                                not_working_urls.append(url_to_try)
                                break
                            else:
                                not_working_urls.append(url_to_try)
                                break
                        except Exception as e:
                            logger.info(f"Failed to verify URL {url_to_try} - {e.__class__.__name__}")
                            if (attempt < retries-1):
                                logger.info(f"    attempting again...")
                            time.sleep(1)
                
                # Create markdown link with working URL or default to domain
                if working_url:
                    markdown_link = f"[{link_text}]({working_url})"
                else:
                    markdown_link = f"https://{domain}"
            
            link.replace_with(markdown_link)

        # Find <select> elements containing <option> elements with URLs
        select_elements = html_content.find_all('select')
        for select in select_elements:
            options = select.find_all('option')
            if not options:
                continue
                
            # Create a markdown list for options with non-empty values
            markdown_list = []
            for option in options:
                option_text = option.get_text().strip()
                option_url = option.get('value')
                
                if option_url and option_url.strip():
                    # If URL is relative, convert to absolute
                    parsed = urlparse(option_url)
                    if not (parsed.scheme and parsed.netloc):
                        option_url = urljoin(base_url, option_url)
                        
                    markdown_list.append(f"- [{option_text}]({option_url})")
                elif option_text:  # Option without URL (like a header)
                    markdown_list.append(f"- {option_text}")
            
            if markdown_list:
                markdown_content = "\n".join(markdown_list)
                select.replace_with(markdown_content)


    @staticmethod
    def preprocess_html(file_path: Path) -> BeautifulSoup:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html5lib')

        for item in soup.contents:
            if isinstance(item, Doctype):
                item.extract()
                break

        # Remove all comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

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
        for selector in [
            'article', 'main', '#sp-component', '.entry-content', '.post-content', '.blog-post', '.content','#content',
            '.itemBody', '[itemprop="articleBody"]', '.itemBody', '.item-page','div[role="main"]', '[role="article"]',
            '.story-content', '.post-article', '.article-body', '.news-article'
            ]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        content_soup =  main_content if main_content else soup

        whitespace_significant = ['pre', 'code', 'textarea']
        
        for element in content_soup.find_all(text=True):
            if not any(parent.name in whitespace_significant for parent in element.parents):
                if len(element) > 1:
                    normalized = element.get_text().replace('\n', ' ')
                    element.replace_with(normalized)

        content_soup.title = soup.title
        return content_soup

    @staticmethod
    def extract_text_from_html(soup: BeautifulSoup, file_path: Path, llm_client: Optional[LLMClient] = None) -> str:
        """
        Extract readable text content from HTML files with improved structure preservation.
        
        Args:
            soup: BeautifulSoup object of the document
            file_path: Path to the HTML file
            llm_client: Optional LLMClient instance for llm markdown corrections if needed
            
        Returns:
            Extracted text with preserved structure in markdown format
        """
        try:
            # Create a deep copy to avoid modifying the original
            soup_copy = BeautifulSoup(str(soup), 'html5lib')
            
            FileProcessor.process_links_in_html(soup_copy, file_path)

            # Step 1: Map headings with their semantic nesting level
            headings = []
            header_positions = {header: i for i, header in enumerate(soup_copy.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))}

            for level in range(1, 7):
                for h in soup_copy.find_all(f'h{level}'):
                    # Calculate semantic nesting depth
                    semantic_depth = 0
                    parent = h.parent
                    
                    current_header_position = header_positions.get(h, 0)
                    counted_headers = set() # Track headers we've already counted
                    # Find the path from root to this header to identify true ancestor elements
                    
                    ancestor_elements = []
                    p = h.parent
                    while p:
                        ancestor_elements.append(p)
                        p = p.parent

                    # Count parents that contain other header elements (creating semantic sections)
                    while parent and parent.name:
                        # Only count containers that have other heading elements

                        ancestor_headers = []
                        for header in parent.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], recursive=False):
                            if (header != h and header_positions.get(header, 0) < current_header_position and
                                header not in counted_headers):
                                # Check if this header is in a direct ancestor relationship
                                header_parent = header.parent
                                if header_parent in ancestor_elements and len(header.get_text()) > 0:
                                    ancestor_headers.append(header)
                        
                        # Add newly found ancestor headers to our tracking set
                        counted_headers.update(ancestor_headers)
                        
                        if ancestor_headers:
                            semantic_depth += 1
                        parent = parent.parent
                    
                    headings.append((h, level, semantic_depth))
            
            # Step 2: Adjust heading levels based on semantic nesting
            if headings:
                # Get the minimum heading level in the document to normalize hierarchy
                min_heading_level = min(level for _, level, _ in headings)

                for h, original_level, semantic_depth in headings:
                    # Normalize the heading level to start from 1
                    normalized_level = original_level - min_heading_level + 1

                    # Adjust level based on semantic nesting (max +3 levels)
                    nesting_adjustment = min(semantic_depth, 3)
                    
                    # New level preserves hierarchy while accounting for nesting
                    new_level = min(normalized_level + nesting_adjustment, 6)
                    
                    # Replace heading with markdown
                    heading_text = h.get_text().strip()
                    if heading_text:
                        marker = '#' * new_level + ' '
                        new_tag = soup_copy.new_string(f"\n\n{marker}{heading_text}\n\n")
                        h.replace_with(new_tag)
            
            
            # Process nested structures (divs, sections, articles)
            for container in soup_copy.find_all(['div', 'section', 'article', 'main', 'aside']):
                # If it has a class or id that might indicate structure
                if container.get('class') or container.get('id'):
                    container.insert_before(soup_copy.new_string("\n"))
                    container.insert_after(soup_copy.new_string("\n"))

            # Process details/summary elements (common in FAQs)
            for details in soup_copy.find_all('details'):
                summary = details.find('summary')
                if summary:
                    summary_text = summary.get_text().strip()
                    new_tag = soup_copy.new_string(f"\n\n### {summary_text}\n\n")
                    summary.replace_with(new_tag)
                # Add spacing after details
                details.insert_after(soup_copy.new_string("\n\n"))
            
            # Process lists
            for ul in soup_copy.find_all('ul'):
                ul.insert_before(soup_copy.new_string("\n"))
                for li in ul.find_all('li', recursive=False):
                    li_text = li.get_text().strip()
                    new_tag = soup_copy.new_string(f"- {li_text}\n")
                    li.replace_with(new_tag)
                ul.insert_after(soup_copy.new_string("\n"))
            
            for ol in soup_copy.find_all('ol'):
                ol.insert_before(soup_copy.new_string("\n"))
                for idx, li in enumerate(ol.find_all('li', recursive=False), 1):
                    li_text = li.get_text().strip()
                    new_tag = soup_copy.new_string(f"{idx}. {li_text}\n")
                    li.replace_with(new_tag)
                ol.insert_after(soup_copy.new_string("\n"))
            
            # Process text formatting
            for strong in soup_copy.find_all(['b', 'strong']):
                text = strong.get_text().strip()
                new_tag = soup_copy.new_string(f"**{text}**")
                strong.replace_with(new_tag)
            
            for em in soup_copy.find_all(['i', 'em']):
                text = em.get_text().strip()
                new_tag = soup_copy.new_string(f"*{text}*")
                em.replace_with(new_tag)
            
            # Extract text while preserving formatting
            html_text = str(soup_copy)
            
            # Replace <p>, <div>, <br> tags with newlines but avoid adding too many
            html_text = re.sub(r'<br\s*/?>|</p>|</div>', '\n', html_text)
            html_text = re.sub(r'<p[^>]*>|<div[^>]*>', '\n', html_text)
            
            # Remove all remaining HTML tags
            html_text = re.sub(r'<[^>]*>', '', html_text)
            
            # Decode HTML entities
            html_text = BeautifulSoup(html_text, 'html5lib').get_text()
            
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

            if llm_client:
                prompt = f"{text}\n\n-----\nCorrect the hierarchy of the headers and/or headers in this markdown " \
                    "where you see fit, consider it's a FAQ that may have topics that will include one or more qa pairs. " \
                    "DO NOT ADD OR ALTER ANYTHING. Preserve all links/formatting.\n" \
                    "If you don't find any errors, keep the way it is.\n" \
                    "Output only the new markdown text."

                response = llm_client.generate_text(prompt, long_output=True, temperature=0.4)
                if response and len(response) >= (len(text) - 300):
                    text = response

            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
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
    def extract_text_from_file(file_path: Path) -> str:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
        """
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return ""
            
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.html', '.htm']:
            return FileProcessor.extract_text_from_html(file_path)
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
