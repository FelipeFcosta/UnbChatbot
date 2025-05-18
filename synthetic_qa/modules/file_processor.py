"""
File processing module for the Synthetic QA Generator.

This module handles extraction of text from different file types and source information.
"""

import logging
import requests
import time
import re
from pathlib import Path
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup, Doctype, Comment
from typing import Any, Dict, Tuple
from typing import Tuple
import os
from .utils import extract_html_from_pdf

# Optional dependency for PDF processing
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. PDF processing will be disabled.")

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
    def extract_domain_and_path(file_path: Path) -> Tuple[str, str, str]:
        """
        Extract domain and path information from a file path,
        prioritizing 'user.original_url' metadata if available.

        Args:
            file_path: Path object for the file

        Returns:
            Tuple of (domain, path, full_url)
        """
        # --- Check for xattr metadata first ---
        if os.path.exists(str(file_path)):
            try:
                attribute_name_bytes = b'user.original_url'
                attr_value_bytes = os.getxattr(str(file_path), attribute_name_bytes)
                original_url_from_meta = attr_value_bytes.decode('utf-8')

                if original_url_from_meta:
                    parsed_url = urlparse(original_url_from_meta)
                    return parsed_url.netloc, parsed_url.path.strip('/'), original_url_from_meta
            except Exception as e:
                logger.info(f"Error reading xattr for '{file_path}': {e}")

        # Fallback
        path_str = str(file_path).replace('\\', '/')

        for domain in FileProcessor.INSTITUTION_DOMAINS.keys():
            if domain in path_str:
                domain_idx = path_str.find(domain)
                if domain_idx >= 0:
                    # Refined check for domain as a distinct path component
                    preceded_by_slash_or_start = (domain_idx == 0 or path_str[domain_idx - 1] == '/')
                    end_of_domain_idx = domain_idx + len(domain)
                    followed_by_slash_or_end_or_ext = (
                        end_of_domain_idx == len(path_str) or
                        path_str[end_of_domain_idx] == '/' or
                        path_str[end_of_domain_idx:].startswith(('.html', '.htm'))
                    )
                    if preceded_by_slash_or_start and followed_by_slash_or_end_or_ext:
                        path_parts_from_domain_component = path_str[domain_idx:].split('/')
                        # Assuming the matched domain is the first part of this slice
                        if path_parts_from_domain_component[0] == domain:
                            path = '/'.join(path_parts_from_domain_component[1:])
                        else:
                            path_slice_after_domain = path_str[domain_idx + len(domain):]
                            path = path_slice_after_domain.strip('/')

                        if path.endswith('index.html') or path.endswith('index.htm'):
                            path = '/'.join(path.split('/')[:-1])

                        if (path.endswith('.html') or path.endswith('.htm')) and path and path != "index":
                            path = os.path.splitext(path)[0]

                        domain_to_use = domain # Use the key from INSTITUTION_DOMAINS
                        # Original www logic was:
                        # if not domain.startswith("www"):
                        #     domain_to_use = f"www.{domain}"

                        full_url = f"https://{domain_to_use}" # Default to https
                        if path:
                            full_url += f"/{path}"
                        return domain, path, full_url

        parts = path_str.split('/')
        guessed_domain = None
        path_after_guessed_domain = ""

        for i, part in enumerate(parts):
            if '.' in part and not part.startswith('.') and len(part.split('.')) >= 2 and \
            not any(part.lower().endswith(ext) for ext in ['.html', '.htm', '.pdf', '.txt', '.orig', '.css', '.js']):
                if i < len(parts) - 1: # Check if it's a directory component
                    guessed_domain = part
                    path_after_guessed_domain = '/'.join(parts[i+1:])
                    break

        if guessed_domain:
            path = path_after_guessed_domain.strip('/')
            if path.endswith('index.html') or path.endswith('index.htm'):
                path = '/'.join(path.split('/')[:-1])

            if (path.endswith('.html') or path.endswith('.htm')) and path and path != "index":
                path = os.path.splitext(path)[0]
            elif path.endswith('.pdf'):
                path = os.path.splitext(path)[0]

            full_url = f"https://{guessed_domain}"
            if path:
                full_url += f"/{path}"
            return guessed_domain, path, full_url

        parent = file_path.parent.name
        name = file_path.stem

        if '.' in parent and len(parent.split('.')) >=2 and \
            not any(parent.lower().endswith(ext) for ext in ['.html', '.htm', '.pdf', '.txt', '.orig', '.css', '.js']):
            return parent, name, f"https://{parent}/{name}".rstrip('/')
        else:
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
            Comma-separated string of courses or "All Courses" if not recognized
        """
        return FileProcessor.INSTITUTION_COURSES.get(domain, "All Courses")
    
    @staticmethod
    def process_links_in_html(html_content: BeautifulSoup, file_path: Path) -> None:
        """
        Process links in HTML content to convert them to markdown format [text](url).
        This function modifies the BeautifulSoup object in place.
        
        Args:
            html_content: BeautifulSoup object containing the parsed HTML content
            file_path: Path to the HTML file to process
        """        
        domain, path, url = FileProcessor.extract_domain_and_path(file_path)
        base_url = f"https://{url}"
        
        not_working_urls = []
        working_urls = []

        # Process <a> elements
        for link in html_content.find_all('a'):
            link_text = link.get_text().strip()
            link_url = link.get('href')
            
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
                                if 'Location' in response.headers:
                                    redirect_url = response.headers.get('Location')
                                    if url_to_try.split("/")[-1] in urljoin(base_url, redirect_url):
                                        url_to_try = urljoin(base_url, redirect_url)
                                        attempt = 0
                                        continue
                                not_working_urls.append(url_to_try)
                                break
                            else:
                                not_working_urls.append(url_to_try)
                                break
                        except Exception as e:
                            logger.debug(f"Failed to verify URL {url_to_try} - {e.__class__.__name__}")
                            if attempt < retries-1:
                                time.sleep(1)
                
                # Create markdown link with working URL or default to domain
                if working_url:
                    markdown_link = f"[{link_text}]({working_url})"
                else:
                    # Use the base domain if no working URL found
                    markdown_link = f"[{link_text}](https://{domain})"
            
            # Replace the link with the markdown version
            link.replace_with(markdown_link)

        # Process <select> elements containing <option> elements with URLs
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
        """
        Preprocess HTML file by removing unnecessary elements and cleaning up the structure.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            BeautifulSoup object with the preprocessed HTML
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html5lib')

        # Remove doctype
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

        content_soup = main_content if main_content else soup

        # Handle whitespace in elements where it's significant
        whitespace_significant = ['pre', 'code', 'textarea']
        
        for element in content_soup.find_all(text=True):
            if not any(parent.name in whitespace_significant for parent in element.parents):
                if len(element) > 1:
                    normalized = element.get_text().replace('\n', ' ')
                    element.replace_with(normalized)

        content_soup.title = soup.title

        return content_soup

    @staticmethod
    def extract_text_from_html(soup: BeautifulSoup, file_path: Path, config=None) -> str:
        """
        Extract readable text content from HTML files with improved structure preservation.
        
        Args:
            soup: BeautifulSoup object of the document
            file_path: Path to the HTML file
            llm_client: Optional LLM client for markdown corrections
            
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
                    counted_headers = set()  # Track headers we've already counted
                    
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
            for _ in range(3):
                text = re.sub(r'\n{3,}', '\n\n', text)

            # Use LLM to correct markdown if available
            if config is not None:
                from .llm_client import LLMClient
                llm_client = LLMClient(config.get("providers", {}).get("text_extraction", {}))
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
    def extract_text_from_pdf(file_path: Path, config=None) -> str:
        """
        Extract text content from PDF files and convert to markdown using LLM, preserving links and hierarchy.

        Args:
            file_path: Path to the PDF file
            config: Optional configuration dictionary for LLMClient

        Returns:
            Extracted markdown text from the PDF
        """
        if not PDF_AVAILABLE:
            logger.error("PyMuPDF (fitz) not available. Cannot extract text from PDF.")
            return ""
        text = extract_html_from_pdf(file_path)
        if not text:
            return ""
        llm_client = None
        if config is not None:
            from .llm_client import LLMClient
            llm_client = LLMClient(config.get("providers", {}).get("text_extraction", {}))
        if llm_client:
            prompt = (
                f"{text}\n\n-----\n"
                "Convert this html pdf text into markdown format, "
                "preserving all links (convert them to markdown links), "
                "and preserving hierarchy of headers and topics as needed.\n"
                "Do NOT add/remove or alter any text content.\n"
                "Preserve all styling and formatting you find in the text. "
                "Fix any inline links that don't seem to be in the correct place.\n"
                "If you don't find any errors, keep the way it is. "
                "Output only the new markdown text."
            )
            response = llm_client.generate_text(prompt)
            if response:
                text = response
        return text
    
    def extract_text_from_file(self, file_path: Path, config=Dict[str, Any]) -> str:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return ""
            
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.html', '.htm']:
            soup = FileProcessor.preprocess_html(file_path)
            return FileProcessor.extract_text_from_html(soup, file_path)
        elif file_extension == '.pdf':
            return FileProcessor.extract_text_from_pdf(file_path, config)
        elif file_extension in ['.txt', '.md', '.csv', '.json']:
            try:
                return file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return ""
        else:
            logger.warning(f"Unsupported file format: {file_extension} for {file_path}")
            return ""
