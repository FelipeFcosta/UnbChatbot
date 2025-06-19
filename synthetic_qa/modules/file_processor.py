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
from .utils import FileType, extract_html_from_pdf

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
                            # try request with and without htm and html
                            for extension in ["", ".html", ".htm"]:
                                test_url = f"https://{domain_to_use}/{path}{extension}"
                                try:
                                    response = requests.head(test_url, timeout=3, allow_redirects=True)
                                    if response.status_code == 200:
                                        full_url = test_url
                                        break
                                except requests.RequestException as e:
                                    pass
                                
                            
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
    def get_soup(file_path: Path) -> BeautifulSoup:
        """
        Get a BeautifulSoup object from a URL.
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return BeautifulSoup(content, 'html5lib')
    
    @staticmethod
    def process_links_in_html(html_content: BeautifulSoup, file_path: Path) -> None:
        """
        Process links in HTML content to convert them to markdown format [text](url).
        This function modifies the BeautifulSoup object in place.
        
        Args:
            html_content: BeautifulSoup object containing the parsed HTML content
            file_path: Path to the HTML file to process
        """        
        _, _, url = FileProcessor.extract_domain_and_path(file_path)
        # Determine a valid base URL by checking possible URL formats
        try:
            base_url = next(
                (url for url in [f"{url}.html", f"{url}"]
                if requests.head(url, timeout=3, allow_redirects=True).status_code == 200),
                url
            )
        except Exception as e:
            logger.info(f"Error processing links in {file_path}: {e.__class__.__name__}")
            return
        
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
                            logger.info(f"Failed to verify URL {url_to_try} - {e.__class__.__name__}")
                            if attempt < retries-1:
                                time.sleep(1)
                
                # Create markdown link with working URL or default to domain
                if working_url:
                    markdown_link = f"[{link_text}]({working_url})"
                else:
                    markdown_link = complete_url
            
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
                    normalized_text = ' '.join(element.get_text().split()) # Collapse all whitespace to single spaces
                    element.replace_with(normalized_text)

        content_soup.title = soup.title

        return content_soup

    @staticmethod
    def extract_text_from_html(soup: BeautifulSoup, file_path: Path, file_type: FileType, config=None) -> str:
        """
        Extract readable text content from HTML files with improved structure preservation,
        especially for tables and lists, converting them to Markdown.
        Now with a more generic table handler.
        
        Args:
            soup: BeautifulSoup object of the document (should be preprocessed)
            file_path: Path to the HTML file (used for link processing context)
            config: Optional configuration for LLM client
            
        Returns:
            Extracted text with preserved structure in markdown format
        """
        try:
            soup_copy = BeautifulSoup(str(soup), 'html5lib') # Work on a copy
            
            FileProcessor.process_links_in_html(soup_copy, file_path)

            # --- Step 1: Generic Table to Markdown Conversion ---
            for table_idx, table in enumerate(soup_copy.find_all('table')):
                markdown_lines = []
                
                # Attempt to identify if this table resembles a key-value structure
                # or a data grid structure. This is a heuristic.
                is_key_value_like = False
                if table.get('class') and any(cls in table.get('class', []) for cls in ['visualizacao', 'programaRelatorio']): # Keep specific hints if present
                    is_key_value_like = True
                else: # Generic heuristic for key-value
                    first_few_rows = table.find_all('tr', limit=5)
                    if first_few_rows and all(len(row.find_all(['th', 'td'])) == 2 for row in first_few_rows if row.find_all(['th','td'])):
                        is_key_value_like = True
                    elif len(table.find_all(['th', 'td'])) < 10 and len(table.find_all('tr')) > len(table.find_all('th', scope='col')): # Few cells, more rows than column headers
                        is_key_value_like = True


                if is_key_value_like:
                    # Key-value like table processing
                    for row in table.find_all('tr'):
                        cells = row.find_all(['th', 'td'])
                        if len(cells) == 2:
                            key_cell = cells[0]
                            value_cell = cells[1]
                            
                            # Extract text more robustly, handling nested elements and line breaks within cells
                            key_parts = [s.strip() for s in key_cell.stripped_strings if s.strip()]
                            value_parts = [s.strip() for s in value_cell.stripped_strings if s.strip()]

                            key = ' '.join(key_parts)
                            value = ' '.join(value_parts)
                            
                            if key and value:
                                markdown_lines.append(f"**{key.rstrip(':').strip()}:** {value.strip()}")
                            elif value: # Only value is present
                                markdown_lines.append(value.strip())
                            elif key: # Only key is present (less common, but possible)
                                markdown_lines.append(f"**{key.rstrip(':').strip()}:**")
                        elif len(cells) == 1: # Single cell, could be a sub-header or content
                            cell_text_parts = [s.strip() for s in cells[0].stripped_strings if s.strip()]
                            text = ' '.join(cell_text_parts)
                            if text:
                                if cells[0].name == 'th' or 'agrupador' in row.get('class', []): # Treat th or specific class as header
                                    markdown_lines.append(f"\n### {text.strip()}\n")
                                else:
                                    markdown_lines.append(text.strip())
                    if markdown_lines:
                        table.replace_with(soup_copy.new_string('\n\n' + '\n'.join(markdown_lines) + '\n\n'))
                        continue # Move to the next table


                # --- Generic Grid-like Table Processing ---
                caption_tag = table.find('caption')
                if caption_tag:
                    caption_text_parts = [s.strip() for s in caption_tag.stripped_strings if s.strip()]
                    caption_text = ' '.join(caption_text_parts)
                    if caption_text:
                        markdown_lines.append(f"#### {caption_text}\n")

                # Process headers (thead or first row ths)
                headers_md = []
                separators_md = []
                max_cols = 0

                thead = table.find('thead')
                header_rows = thead.find_all('tr') if thead else []
                if not header_rows:
                    # Fallback: Check first row of table if it contains only th
                    first_row = table.find('tr')
                    if first_row and all(cell.name == 'th' for cell in first_row.find_all(['th', 'td'], recursive=False)):
                        header_rows = [first_row]
                
                if header_rows:
                    # For simplicity, using the last header row if multiple exist, or try to merge
                    # A more complex logic would be needed for truly multi-row headers
                    final_header_cells = header_rows[-1].find_all(['th', 'td']) 
                    current_col_count = 0
                    for th in final_header_cells:
                        colspan = int(th.get('colspan', 1))
                        text_parts = [s.strip() for s in th.stripped_strings if s.strip()]
                        header_text = ' '.join(text_parts).replace('|', '\\|')
                        
                        headers_md.append(header_text)
                        if colspan > 1:
                            headers_md.extend([''] * (colspan - 1))
                        separators_md.extend(['---'] * colspan)
                        current_col_count += colspan
                    max_cols = max(max_cols, current_col_count)
                
                def add_markdown_table_headers():
                    if headers_md and separators_md:
                         if not markdown_lines or not markdown_lines[-1].startswith('| ---'):
                            markdown_lines.append('| ' + ' | '.join(headers_md) + ' |')
                            markdown_lines.append('| ' + ' | '.join(separators_md) + ' |')
                
                add_markdown_table_headers()

                # Process body (tbody or remaining rows)
                tbody = table.find('tbody')
                body_rows = tbody.find_all('tr') if tbody else table.find_all('tr')
                
                # Filter out header rows if already processed
                if thead:
                    body_rows = [row for row in body_rows if row.parent.name == 'tbody']
                elif header_rows: # if headers were taken from first row
                     body_rows = body_rows[len(header_rows):]


                for row_idx, row in enumerate(body_rows):
                    # Heuristic for section header rows (e.g., <tr class="agrupador">)
                    is_section_header_row = False
                    if 'agrupador' in row.get('class', []):
                        is_section_header_row = True
                    # Also check if the row has a single cell with colspan matching max_cols or a prominent style
                    cells_in_row = row.find_all(['td', 'th'])
                    if len(cells_in_row) == 1 and cells_in_row[0].get('colspan'):
                        try:
                            if int(cells_in_row[0].get('colspan')) >= max_cols -1 and max_cols > 1: # colspan spans most/all
                                is_section_header_row = True
                        except ValueError:
                            pass # colspan not an int
                    
                    if is_section_header_row:
                        row_text_parts = [s.strip() for s in row.stripped_strings if s.strip()]
                        row_text = ' '.join(row_text_parts)
                        # Special handling for "tituloDisciplina" if present
                        title_span = row.find('span', class_='tituloDisciplina')
                        if title_span:
                            row_text = ' '.join([s.strip() for s in title_span.stripped_strings if s.strip()])

                        if row_text:
                            markdown_lines.append(f"\n### {row_text}\n")
                            if headers_md: # Re-add headers if they exist for this section
                                add_markdown_table_headers()
                        continue # Skip processing this row as a data row

                    # Regular data row
                    row_cells_md = []
                    current_col_count_in_row = 0
                    for cell_idx, cell in enumerate(cells_in_row):
                        colspan = int(cell.get('colspan', 1))
                        # Extract cell content carefully, joining internal <br> with "; " or space
                        cell_content_parts = []
                        for elem in cell.contents:
                            if isinstance(elem, str):
                                cell_content_parts.append(elem.strip())
                            elif elem.name == 'br':
                                if cell_content_parts and cell_content_parts[-1] != "; ": # Add separator if needed
                                    cell_content_parts.append("; ")
                            else: # Other tags
                                sub_parts = [s.strip() for s in elem.stripped_strings if s.strip()]
                                if sub_parts:
                                    cell_content_parts.append(' '.join(sub_parts))
                        
                        # Clean up joined parts
                        cleaned_cell_content = []
                        for part in cell_content_parts:
                            if part.strip():
                                cleaned_cell_content.append(part.strip())
                        
                        cell_text = ' '.join(cleaned_cell_content).replace('; ;', ';').replace('  ', ' ')
                        cell_text = re.sub(r'\s*;\s*', '; ', cell_text).strip('; ')
                        cell_text = cell_text.replace('|', '\\|')
                        
                        # Specific handling for the "Horário" column with popup from your example
                        if "ajuda" in cell.decode_contents() and "popUp" in cell.decode_contents(): # Heuristic for your horario cell
                            popup_div = cell.find('div', class_='popUp')
                            popup_content_text = ""
                            if popup_div:
                                popup_lines = [s.strip() for s in popup_div.stripped_strings if s.strip()]
                                if popup_lines:
                                    popup_content_text = " (" + "; ".join(popup_lines) + ")"
                            cell_text = cell.find(text=True, recursive=False).strip() + popup_content_text
                            cell_text = cell_text.replace('|', '\\|')


                        row_cells_md.append(cell_text)
                        if colspan > 1:
                            row_cells_md.extend([''] * (colspan - 1))
                        current_col_count_in_row += colspan
                    
                    max_cols = max(max_cols, current_col_count_in_row)

                    # Pad row if it has fewer cells than max_cols (due to prior colspans perhaps)
                    if headers_md and len(row_cells_md) < len(headers_md):
                        row_cells_md.extend([''] * (len(headers_md) - len(row_cells_md)))
                    elif not headers_md and max_cols > 0 and len(row_cells_md) < max_cols :
                         row_cells_md.extend([''] * (max_cols - len(row_cells_md)))


                    if any(c.strip() for c in row_cells_md): # Only add if row is not entirely empty
                        markdown_lines.append('| ' + ' | '.join(row_cells_md) + ' |')

                # Update max_cols for headers if body had more
                if headers_md and len(headers_md) < max_cols:
                    headers_md.extend([''] * (max_cols - len(headers_md)))
                    separators_md.extend(['---'] * (max_cols - len(separators_md)))
                    if markdown_lines and markdown_lines[0].startswith('| ') and not markdown_lines[0].startswith('| ---'):
                        markdown_lines[0] = '| ' + ' | '.join(headers_md) + ' |'
                        markdown_lines[1] = '| ' + ' | '.join(separators_md) + ' |'


                tfoot = table.find('tfoot')
                if tfoot:
                    tfoot_text_parts = [s.strip() for s in tfoot.stripped_strings if s.strip()]
                    tfoot_text = ' '.join(tfoot_text_parts)
                    if tfoot_text:
                        markdown_lines.append(f"\n**{tfoot_text.strip()}**\n")

                if markdown_lines:
                    table.replace_with(soup_copy.new_string('\n\n' + '\n'.join(markdown_lines) + '\n\n'))
            
            # --- Step 2: Semantic heading adjustments (existing code) ---
            headings = []
            header_positions = {header: i for i, header in enumerate(soup_copy.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))}

            for level in range(1, 7):
                for h in soup_copy.find_all(f'h{level}'):
                    semantic_depth = 0
                    parent = h.parent
                    current_header_position = header_positions.get(h, 0)
                    counted_headers = set()
                    ancestor_elements = [p for p in h.parents]

                    while parent and parent.name:
                        ancestor_headers_in_parent = []
                        for header_tag in parent.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], recursive=False):
                            if (header_tag != h and 
                                header_positions.get(header_tag, 0) < current_header_position and
                                header_tag not in counted_headers and
                                header_tag.parent in ancestor_elements and len(header_tag.get_text(strip=True)) > 0):
                                ancestor_headers_in_parent.append(header_tag)
                        
                        counted_headers.update(ancestor_headers_in_parent)
                        if ancestor_headers_in_parent:
                            semantic_depth += 1
                        parent = parent.parent
                    
                    headings.append((h, level, semantic_depth))
            
            if headings:
                min_heading_level = min(level for _, level, _ in headings) if headings else 1
                for h, original_level, semantic_depth in headings:
                    normalized_level = original_level - min_heading_level + 1
                    nesting_adjustment = min(semantic_depth, 3) # Limit how much nesting can increase header level
                    new_level = min(normalized_level + nesting_adjustment, 6)
                    heading_text_parts = [s.strip() for s in h.stripped_strings if s.strip()]
                    heading_text = ' '.join(heading_text_parts)
                    if heading_text:
                        marker = '#' * new_level + ' '
                        new_tag = soup_copy.new_string(f"\n\n{marker}{heading_text}\n\n")
                        h.replace_with(new_tag)
            
            # --- Step 3: Convert other relevant HTML elements to Markdown (existing code) ---
            for container in soup_copy.find_all(['div', 'section', 'article', 'main', 'aside']):
                if container.get('class') or container.get('id'): 
                    if container.name not in ['td', 'th']: 
                        container.insert_before(soup_copy.new_string("\n"))
                        container.insert_after(soup_copy.new_string("\n"))
            
            for details in soup_copy.find_all('details'):
                summary = details.find('summary')
                if summary:
                    summary_text_parts = [s.strip() for s in summary.stripped_strings if s.strip()]
                    summary_text = ' '.join(summary_text_parts)
                    new_tag = soup_copy.new_string(f"\n\n### {summary_text}\n\n") 
                    summary.replace_with(new_tag)
                details.insert_after(soup_copy.new_string("\n\n"))

            for ul in soup_copy.find_all('ul'):
                ul.insert_before(soup_copy.new_string("\n"))
                for li in ul.find_all('li', recursive=False):
                    li_text_parts = [s.strip() for s in li.stripped_strings if s.strip()]
                    li_text = ' '.join(li_text_parts)
                    new_tag = soup_copy.new_string(f"- {li_text}\n")
                    li.replace_with(new_tag)
                ul.insert_after(soup_copy.new_string("\n"))
            
            for ol in soup_copy.find_all('ol'):
                ol.insert_before(soup_copy.new_string("\n"))
                for idx, li in enumerate(ol.find_all('li', recursive=False), 1):
                    li_text_parts = [s.strip() for s in li.stripped_strings if s.strip()]
                    li_text = ' '.join(li_text_parts)
                    new_tag = soup_copy.new_string(f"{idx}. {li_text}\n")
                    li.replace_with(new_tag)
                ol.insert_after(soup_copy.new_string("\n"))

            for strong in soup_copy.find_all(['b', 'strong']):
                text_parts = [s.strip() for s in strong.stripped_strings if s.strip()]
                text = ' '.join(text_parts)
                if text: new_tag = soup_copy.new_string(f"**{text}**"); strong.replace_with(new_tag)
                else: strong.decompose() 
            
            for em in soup_copy.find_all(['i', 'em']):
                text_parts = [s.strip() for s in em.stripped_strings if s.strip()]
                text = ' '.join(text_parts)
                if text: new_tag = soup_copy.new_string(f"*{text}*"); em.replace_with(new_tag)
                else: em.decompose() 

            for br in soup_copy.find_all('br'):
                # Only replace with newline if not already surrounded by block elements or inside a table cell already handled
                if not (br.previous_sibling is None and br.next_sibling is None and br.parent.name in ['p', 'div', 'li']):
                     # And not inside a table cell that we've manually processed for internal newlines
                    if not any(p.name == 'td' or p.name == 'th' for p in br.parents if p.find_parent('table')):
                        br.replace_with(soup_copy.new_string('\n'))
                    else: # inside a table cell, likely already handled by cell text extraction logic.
                        br.replace_with(soup_copy.new_string(' ')) # replace with space to avoid run-on words
                else:
                    br.decompose() # Likely redundant <br>

            for hr in soup_copy.find_all('hr'):
                hr.replace_with(soup_copy.new_string('\n\n---\n\n'))

            # --- Step 4: Final Text Extraction and Cleanup ---
            raw_text = soup_copy.get_text(separator='\n') 

            lines = raw_text.split('\n')
            processed_lines = []
            for line in lines:
                stripped_line = line.strip()
                if stripped_line or (line.startswith('| ---') and not stripped_line): 
                    processed_lines.append(stripped_line)
            
            text = '\n'.join(processed_lines)
            text = re.sub(r'\n\s*\n', '\n\n', text) # Consolidate multiple newlines

            llm_client = None
            if config is not None:
                llm_config = config.get("providers", {}).get("text_extraction", {})
                if file_type == FileType.COMPONENT:
                    llm_config = config.get("providers", {}).get("component_text_extraction", {})
                if llm_config and llm_config.get("provider") != "none": 
                    from .llm_client import LLMClient 
                    llm_client = LLMClient(llm_config)
            
            if llm_client:
                prompt = f"{text}\n\n-----\nCorrect the hierarchy of the headers in this markdown " \
                         "where you see fit. DO NOT ADD OR ALTER ANY ACTUAL CONTENT unless it doesn't make sense. Preserve all links/formatting.\n" \
                         "In general, if the markdown is somewhat unstructured, make it more readable and easier to understand.\n" \
                         "If you don't find any errors, keep the way it is.\n" \
                         "Output only the new markdown text."
                try:
                    response = llm_client.generate_text(prompt, temperature=0.4)
                    if response and len(response) >= (len(text) - 300): 
                        text = response.strip() 
                except Exception as e_llm:
                    logger.warning(f"LLM correction failed: {e_llm}")
            
            return text.strip() 
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML ({file_path}): {e}", exc_info=True)
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
                "Do NOT add/remove or alter any word.\n"
                "REMOVE ANY unwanted still present html artifacts (no html should remain in the text).\n"
                "If the content contains a table that is not properly formatted, convert it into a clear and well-structured markdown table. Pay very close attention to accurately representing the column and row headers in the correct order.\n"
                "Preserve all styling and formatting you find in the text. "
                "Fix any inline links that don't seem to be in the correct place.\n"
                "Remove ALL mid-sentence out-of-place line breaks (\\n) present in the text if present.\n"
                "If you don't find any errors, keep the way it is. "
                "Output only the new markdown text."
            )
            response = llm_client.generate_text(prompt)
            if response:
                text = response
        return text
    
    def extract_text_from_file(self, file_path: Path, file_type: FileType, config=Dict[str, Any]) -> str:
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
            return FileProcessor.extract_text_from_html(soup, file_path, file_type, config)
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
