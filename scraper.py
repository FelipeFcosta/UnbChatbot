import requests
from bs4 import BeautifulSoup, SoupStrainer
from urllib.parse import urljoin, urlparse, unquote_plus
import os
import time
import re
import subprocess
import mimetypes
import random
import html

# --- Configuration ---
TARGET_DIR_BASE = "/home/farias/tcc/unb_clone2"
STARTING_URL = "https://www.cic.unb.br/"

MAX_DEPTH = 3 # Adjust for crawl depth
USER_AGENT = "MyRobustScraper/1.0 (+http://example.com/botinfo)"
WAIT_SECONDS = 0.1
RANDOM_WAIT_MAX_ADDITIONAL = 0.05
REJECT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".css", ".js", ".ico"}

STAY_ON_SAME_HOSTNAME = True
ALLOW_OFF_HOST_DIRECT_LINKED_FILES = True

ADJUST_HTML_EXTENSION = True
CONVERT_LINKS_IN_HTML = True
USE_CONTENT_DISPOSITION = True

ONLY_SCRAPE_STARTING_URL_PREFIX = False

# visited_urls_content: maps actual_fetched_url (fragment-less for HTML) -> local_save_filepath
visited_urls_content = {}
# urls_to_visit: (url_to_fetch_with_fragment, current_depth, original_url_of_parent_html_if_any, original_href_from_parent_tag)
urls_to_visit = [(STARTING_URL, 0, None, None)]

session = requests.Session()
session.headers.update({'User-Agent': USER_AGENT})

os.makedirs(TARGET_DIR_BASE, exist_ok=True)
print(f"Starting download from: {STARTING_URL}")
print(f"Saving to: {TARGET_DIR_BASE}")
print(f"Crawling HTML on same host as STARTING_URL: {STAY_ON_SAME_HOSTNAME}")
if STAY_ON_SAME_HOSTNAME:
    print(f"Downloading off-host *files* linked from same-host pages: {ALLOW_OFF_HOST_DIRECT_LINKED_FILES}")
print(f"Max depth: {MAX_DEPTH}")
print("--------------------------------------------------")

def sanitize_filename(filename_str):
    if not isinstance(filename_str, str): filename_str = str(filename_str)
    if filename_str.startswith("http://") or filename_str.startswith("https://"):
        filename_str = urlparse(filename_str).path
    filename_str = unquote_plus(filename_str)
    filename_str = filename_str.replace('/', '_SLASH_')
    filename_str = filename_str.replace('\\', '_BSLASH_')
    filename_str = filename_str.replace(':', '_COLON_')
    filename_str = re.sub(r'[^\w\.\-\_?=&]', '_', filename_str)
    filename_str = filename_str.replace('_SLASH_', '_').replace('_BSLASH_', '_').replace('_COLON_', '_')
    filename_str = re.sub(r'_+', '_', filename_str).strip('_')
    return filename_str if filename_str else "downloaded_file"

def get_file_extension_from_url_or_mimetype(url, content_type_header):
    parsed_url = urlparse(url)
    path = parsed_url.path
    name, ext = os.path.splitext(path)
    if ext and len(ext) > 1 and len(ext) < 7: return ext
    if content_type_header:
        mime_type = content_type_header.split(';')[0].strip()
        guessed_ext = mimetypes.guess_extension(mime_type)
        if guessed_ext: return guessed_ext
    return ""

def get_metadata(filepath, attr_name):
    """Gets a specific extended attribute from a file."""
    try:
        result = subprocess.run(['getfattr', '-n', attr_name, '--only-values', filepath],
                                check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def set_custom_metadata(filepath, fetched_url, original_parent_web_url=None):
    # print(f"  Attempting metadata for: {filepath} (URL: {fetched_url})") # Reduced verbosity
    metadata_success = {"original_url": False, "source_page_url": False, "source_html_path": False}
    try:
        subprocess.run(['setfattr', '-n', 'user.original_url', '-v', fetched_url, filepath], check=True, capture_output=True)
        metadata_success["original_url"] = True
        if original_parent_web_url:
            subprocess.run(['setfattr', '-n', 'user.source_page_url', '-v', original_parent_web_url, filepath], check=True, capture_output=True)
            metadata_success["source_page_url"] = True
            parent_url_no_frag_key = urlparse(original_parent_web_url)._replace(fragment="").geturl()
            local_parent_path = visited_urls_content.get(parent_url_no_frag_key)
            if local_parent_path and isinstance(local_parent_path, str) and not local_parent_path.startswith(("SKIPPED_", "HTTP_ERROR_", "REQUEST_EXCEPTION_", "UNEXPECTED_ERROR_")):
                try: relative_source_path = os.path.relpath(local_parent_path, start=TARGET_DIR_BASE)
                except ValueError: relative_source_path = local_parent_path
                subprocess.run(['setfattr', '-n', 'user.source_html_path', '-v', relative_source_path, filepath], check=True, capture_output=True)
                metadata_success["source_html_path"] = True
        else:
            metadata_success["source_page_url"] = True; metadata_success["source_html_path"] = True
    except subprocess.CalledProcessError as e:
        err_str=e.stderr.decode().strip().lower(); failed_attr="unknown attribute"
        if not metadata_success["original_url"]:failed_attr="user.original_url"
        elif original_parent_web_url and not metadata_success["source_page_url"]:failed_attr="user.source_page_url"
        elif original_parent_web_url and not metadata_success["source_html_path"]:failed_attr="user.source_html_path"
        print(f"    Warning: setfattr FAILED for '{failed_attr}' on {filepath}: {err_str}")
    except FileNotFoundError: print(f"    Warning: setfattr command not found for {filepath}")

def get_potential_local_path(url_to_check_str, content_type_hint_for_url="", is_html_hint_for_url=False, original_parent_url_for_colocation=None, actual_response_for_cd=None):
    parsed_url_to_check = urlparse(url_to_check_str)
    current_hostname_dir = os.path.join(TARGET_DIR_BASE, parsed_url_to_check.netloc)
    current_url_path_cleaned = parsed_url_to_check.path.lstrip('/')
    if not current_url_path_cleaned:
        dir_path_in_target = current_hostname_dir
        filename_candidate_base = "index"
    else:
        path_parts = current_url_path_cleaned.split('/')
        potential_filename = path_parts[-1]
        has_extension_in_path = (os.path.splitext(potential_filename)[1] and \
                                len(os.path.splitext(potential_filename)[1]) > 1 and \
                                len(os.path.splitext(potential_filename)[1]) < 7)
        is_file_like = has_extension_in_path or \
                       parsed_url_to_check.query or \
                       (not is_html_hint_for_url and potential_filename) or \
                       (is_html_hint_for_url and not current_url_path_cleaned.endswith('/'))
        if is_file_like:
            dir_path_in_target = os.path.join(current_hostname_dir, *path_parts[:-1])
            filename_candidate_base = potential_filename if potential_filename else "index_query_at_dir"
        elif is_html_hint_for_url :
            dir_path_in_target = os.path.join(current_hostname_dir, *path_parts[:-1])
            filename_candidate_base = path_parts[-1] if path_parts[-1] else "index"
            if not filename_candidate_base and current_url_path_cleaned.endswith('/'): filename_candidate_base = "index"
            elif not filename_candidate_base: filename_candidate_base = "index"
        else:
            dir_path_in_target = os.path.join(current_hostname_dir, *path_parts)
            filename_candidate_base = "index"
    is_on_starting_host_for_predict = (parsed_url_to_check.netloc == urlparse(STARTING_URL).netloc)
    if original_parent_url_for_colocation and not is_html_hint_for_url:
        parent_lp_key = urlparse(original_parent_url_for_colocation)._replace(fragment="").geturl()
        parent_local_save_path = visited_urls_content.get(parent_lp_key)
        if parent_local_save_path and isinstance(parent_local_save_path,str) and not parent_local_save_path.startswith(("S","H","R","U")):
            dir_path_in_target = os.path.dirname(parent_local_save_path)
            filename_candidate_base = os.path.basename(parsed_url_to_check.path) if parsed_url_to_check.path.strip('/') else "resource"
            if not filename_candidate_base and parsed_url_to_check.query: filename_candidate_base = sanitize_filename(parsed_url_to_check.query)
            elif not filename_candidate_base: filename_candidate_base = "linked_resource"
    filename_candidate = filename_candidate_base
    if parsed_url_to_check.query:
        sq = sanitize_filename("?" + parsed_url_to_check.query)
        if '?' not in filename_candidate and filename_candidate != "index":
            query_params_in_fname = False
            if '?' not in filename_candidate_base:
                for qp_part in parsed_url_to_check.query.split('&'):
                    if sanitize_filename(qp_part.split('=')[0]) in filename_candidate: query_params_in_fname = True; break
                if not query_params_in_fname: filename_candidate += sq
        elif filename_candidate == "index" and '?' not in filename_candidate : filename_candidate += sq
    final_filename = sanitize_filename(filename_candidate)
    guessed_ext = get_file_extension_from_url_or_mimetype(url_to_check_str, content_type_hint_for_url)
    if USE_CONTENT_DISPOSITION and actual_response_for_cd:
        cd_h = actual_response_for_cd.headers.get('content-disposition')
        if cd_h:
            m=re.search(r'filename\*?=(?:UTF-8\'\')?(\"[^\"]+\"|[^;\s]+)',cd_h,re.I)
            if m:
                fn_r=m.group(1).strip('"'); fn_cd=unquote_plus(fn_r,'utf-8','replace'); final_filename=sanitize_filename(fn_cd)
                if not os.path.splitext(final_filename)[1] and guessed_ext: final_filename += guessed_ext
    if ADJUST_HTML_EXTENSION and is_html_hint_for_url:
        name,ext=os.path.splitext(final_filename)
        if ext.lower() not in ['.html','.htm']: final_filename=name+".html"
    if final_filename=="index" and is_html_hint_for_url: final_filename="index.html"
    elif final_filename=="index" and guessed_ext and not is_html_hint_for_url: final_filename="index"+guessed_ext
    elif not os.path.splitext(final_filename)[1] and guessed_ext: final_filename+=guessed_ext
    return os.path.join(dir_path_in_target, final_filename), is_html_hint_for_url


def deobfuscate_joomla_emails_in_soup(soup):
    """
    Finds and deobfuscates Joomla-style email protection in a BeautifulSoup object.
    Modifies the soup object in place.
    Returns True if any modifications were made, False otherwise.
    """
    print("    Debug: deobfuscate_joomla_emails_in_soup called")
    modified = False
    email_deobfuscation_text_start = "Este endereço de email está sendo protegido de spambots."
    
    # Find all potential spans. Iterate over a list copy as we modify the soup.
    spans_to_process = [
        s for s in soup.find_all('span', id=lambda x: x and x.startswith('cloak'))
        if email_deobfuscation_text_start in s.get_text()
    ]
    
    print(f"    Debug: Found {len(spans_to_process)} spans to process")

    for span_tag in spans_to_process:
        script_tag = span_tag.find_next_sibling('script')
        if not (script_tag and script_tag.string):
            continue

        js_code = script_tag.string.replace('\n', ' ').replace('\r', '') # Normalize JS code
        span_id = span_tag.get('id')
        
        # --- Find the variable name containing the email text ---
        email_var_name = None
        
        # Method 1: Look for innerHTML assignment and find the email variable
        innerHTML_pattern = r"document\.getElementById\('" + re.escape(span_id) + r"'\)\.innerHTML\s*\+?=\s*(.*?);"
        innerHTML_match = re.search(innerHTML_pattern, js_code)
        
        if innerHTML_match:
            assignment_content = innerHTML_match.group(1)
            print(f"    Debug: innerHTML assignment: {assignment_content}")
            
            # Look for the last variable before '</a>' in the assignment
            # This handles patterns like: ...+variable_name+'</a>' or ...+variable_name+'<\/a>'
            var_before_close_tag = re.search(r"\+\s*([a-zA-Z0-9_]+)\s*\+\s*['\"][^'\"]*<\\/a", assignment_content)
            if var_before_close_tag:
                email_var_name = var_before_close_tag.group(1)
                print(f"    Debug: Found email variable: {email_var_name}")

        # Method 2: Fallback to original logic for simpler patterns
        if not email_var_name:
            expected_addy_var_name = "addy" + span_id[len('cloak'):]
            if re.search(rf"var\s+{re.escape(expected_addy_var_name)}\s*=", js_code) or \
               re.search(rf"{re.escape(expected_addy_var_name)}\s*=\s*{re.escape(expected_addy_var_name)}\s*\+", js_code):
                email_var_name = expected_addy_var_name
                print(f"    Debug: Using expected addy variable: {email_var_name}")
            else: # Generic 'addy...' fallback
                generic_addy_match = re.search(r"var\s+(addy[a-zA-Z0-9_]+)\s*=\s*'.*?'", js_code)
                if generic_addy_match:
                    temp_var = generic_addy_match.group(1)
                    if f"getElementById('{span_id}').innerHTML" in js_code and temp_var in js_code:
                        email_var_name = temp_var
                        print(f"    Debug: Using generic addy variable: {email_var_name}")

        if not email_var_name:
            print(f"    Warning: Could not determine the variable for email deobfuscation for span {span_id}.")
            continue
        
        # Extract email parts using the found variable name
        email_parts_str = []
        # Initial assignment: var email_var_name = '...' + '...' ;
        initial_assignment_pattern = re.compile(rf"var\s+{re.escape(email_var_name)}\s*=\s*((?:'.*?'(?:\s*\+\s*)?)+);")
        match_initial = initial_assignment_pattern.search(js_code)
        if match_initial:
            parts_group = match_initial.group(1).strip().rstrip(';')
            email_parts_str.extend(re.findall(r"'(.*?)'", parts_group))

        # Subsequent concatenations: email_var_name = email_var_name + '...' ;
        concat_pattern = re.compile(rf"{re.escape(email_var_name)}\s*=\s*{re.escape(email_var_name)}\s*\+\s*((?:'.*?'(?:\s*\+\s*)?)+);")
        for match_concat in concat_pattern.finditer(js_code):
            parts_group = match_concat.group(1).strip().rstrip(';')
            email_parts_str.extend(re.findall(r"'(.*?)'", parts_group))
        
        # New Fallback: Handle cases where the variable is defined on the same line as the innerHTML assignment
        if not email_parts_str and email_var_name:
            # e.g., var addy_text... = '...';document.getElementById...
            one_line_decl_match = re.search(
                r"var\s+" + re.escape(email_var_name) + r"\s*=\s*((?:'.*?'(?:\s*\+\s*)?)+);",
                js_code
            )
            if one_line_decl_match:
                parts_group = one_line_decl_match.group(1).strip().rstrip(';')
                email_parts_str.extend(re.findall(r"'(.*?)'", parts_group))

        if email_parts_str:
            try:
                decoded_email_parts = [html.unescape(part) for part in email_parts_str]
                email_address = "".join(decoded_email_parts)
                
                new_tag = soup.new_tag('a', href=f'mailto:{email_address}')
                new_tag.string = email_address
                
                if span_tag.parent:
                    span_tag.replace_with(new_tag)
                else:
                    print(f"    Warning: Span {span_id} has no parent, cannot replace.")
                    continue
                
                if script_tag.parent:
                    script_tag.decompose()
                
                print(f"    Deobfuscated email: {email_address} (from span {span_id})")
                modified = True
            except Exception as e_deob:
                print(f"    Error processing deobfuscated email for span {span_id}: {e_deob}")
        else:
            print(f"    Warning: Could not extract email parts for deobfuscation for span {span_id}. Parts list empty.")
            print(f"    Debug: Variable name was: {email_var_name}")
            
    return modified

download_attempt_count = 0
successful_downloads = 0
link_conversion_map = {}

while urls_to_visit:
    current_url_to_fetch_with_frag, current_depth, original_url_of_parent, original_href_from_tag = urls_to_visit.pop(0)
    download_attempt_count += 1
    parsed_current_for_visit_check = urlparse(current_url_to_fetch_with_frag)
    url_key_for_visited_check = parsed_current_for_visit_check._replace(fragment="").geturl()

    if ONLY_SCRAPE_STARTING_URL_PREFIX:
        # Remove query and fragment for comparison
        def url_base_no_query_frag(u):
            p = urlparse(u)
            return p._replace(query="", fragment="").geturl()
        if not url_base_no_query_frag(url_key_for_visited_check).startswith(url_base_no_query_frag(STARTING_URL)):
            print(f"  Skipping (not under STARTING_URL): {url_key_for_visited_check}")
            visited_urls_content[url_key_for_visited_check] = f"SKIPPED_NOT_UNDER_STARTING_URL_{url_key_for_visited_check}"
            continue

    if url_key_for_visited_check in visited_urls_content:
        if original_url_of_parent and CONVERT_LINKS_IN_HTML and original_href_from_tag:
            parent_lp_key = urlparse(original_url_of_parent)._replace(fragment="").geturl()
            parent_lp = visited_urls_content.get(parent_lp_key)
            if parent_lp and isinstance(parent_lp, str) and not parent_lp.startswith(("S","H","R","U")):
                if parent_lp not in link_conversion_map: link_conversion_map[parent_lp] = {'original_url': original_url_of_parent, 'links': {}}
                link_conversion_map[parent_lp]['links'][original_href_from_tag] = visited_urls_content[url_key_for_visited_check]
        continue
    if current_depth > MAX_DEPTH: print(f"  Skipping (depth limit): {current_url_to_fetch_with_frag}"); continue

    print(f"Processing (Depth {current_depth}): {current_url_to_fetch_with_frag} (Linked by '{original_href_from_tag or 'N/A'}' from: {original_url_of_parent or 'STARTING_URL'})")

    path_ext = os.path.splitext(parsed_current_for_visit_check.path)[1]
    initial_is_html_hint = (not path_ext or path_ext.lower() in ['.htm', '.html'] or parsed_current_for_visit_check.path.endswith('/'))
    predicted_path_initial, _ = get_potential_local_path(
        url_key_for_visited_check,
        is_html_hint_for_url=initial_is_html_hint,
        original_parent_url_for_colocation=original_url_of_parent
    )

    if os.path.exists(predicted_path_initial):
        existing_file_original_url = get_metadata(predicted_path_initial, 'user.original_url')
        if existing_file_original_url == url_key_for_visited_check or \
           (existing_file_original_url and urlparse(existing_file_original_url)._replace(fragment="").geturl() == url_key_for_visited_check):
            print(f"  File already exists: {predicted_path_initial} (for URL: {url_key_for_visited_check}). Skipping download.")
            visited_urls_content[url_key_for_visited_check] = predicted_path_initial
            
            _, ext_existing = os.path.splitext(predicted_path_initial)
            is_existing_html = ext_existing.lower() in ['.html', '.htm']
            if not is_existing_html:
                mime_type_existing, _ = mimetypes.guess_type(predicted_path_initial)
                if mime_type_existing and 'text/html' in mime_type_existing:
                    is_existing_html = True
            
            if is_existing_html and current_depth <= MAX_DEPTH:
                print(f"  Parsing existing HTML: {predicted_path_initial} for links.")
                try:
                    with open(predicted_path_initial, 'rb') as f_r_existing:
                        html_bytes_existing = f_r_existing.read()
                    
                    # Try decoding with utf-8, fallback to BS default if error
                    try:
                        soup_existing = BeautifulSoup(html_bytes_existing.decode('utf-8'), 'html.parser')
                    except UnicodeDecodeError:
                        print(f"    Warning: UTF-8 decode failed for existing file {predicted_path_initial} during parsing. BeautifulSoup will attempt to guess encoding.")
                        soup_existing = BeautifulSoup(html_bytes_existing, 'html.parser') # Let BS guess

                    # --- MODIFIED: Deobfuscate emails in existing soup ---
                    if deobfuscate_joomla_emails_in_soup(soup_existing): # Modifies soup_existing in place
                        print(f"    Re-saving existing file {predicted_path_initial} after email deobfuscation.")
                        try:
                            with open(predicted_path_initial, 'w', encoding='utf-8', errors='replace') as f_w_existing:
                                f_w_existing.write(str(soup_existing))
                        except Exception as e_resave_existing:
                            print(f"      Error re-saving existing file {predicted_path_initial} after deobfuscation: {e_resave_existing}")
                    # --- END MODIFICATION ---

                    for tag_existing in soup_existing.find_all(['a','link','script','img'],href=lambda x:x is not None) + \
                                        soup_existing.find_all(['script','img'],src=lambda x:x is not None):
                        attr_ex='href' if tag_existing.has_attr('href') else 'src'; val_ex=tag_existing[attr_ex]
                        clean_v_ex=val_ex.replace('\n','').replace('\t','').strip()
                        if not clean_v_ex or clean_v_ex.lower().startswith(('javascript:','mailto:','tel:','#','data:')): continue
                        next_url_abs_with_frag_ex=urljoin(url_key_for_visited_check, clean_v_ex)
                        p_next_url_ex=urlparse(next_url_abs_with_frag_ex)
                        if p_next_url_ex.scheme not in ['http','https']: continue
                        next_url_abs_no_frag_for_check_ex = p_next_url_ex._replace(fragment="").geturl()
                        in_q_ex=any(q_url_in_q == next_url_abs_with_frag_ex for q_url_in_q,_,_,_ in urls_to_visit)
                        if next_url_abs_no_frag_for_check_ex not in visited_urls_content and not in_q_ex:
                            urls_to_visit.append((next_url_abs_with_frag_ex, current_depth+1, url_key_for_visited_check, clean_v_ex))
                        elif next_url_abs_no_frag_for_check_ex in visited_urls_content and CONVERT_LINKS_IN_HTML and clean_v_ex:
                            target_lp_ex = visited_urls_content[next_url_abs_no_frag_for_check_ex]
                            if predicted_path_initial not in link_conversion_map:
                                link_conversion_map[predicted_path_initial] = {'original_url': url_key_for_visited_check, 'links':{}}
                            link_conversion_map[predicted_path_initial]['links'][clean_v_ex] = target_lp_ex
                except Exception as e_parse_existing:
                    print(f"    Error parsing existing HTML {predicted_path_initial}: {e_parse_existing}")
            
            if original_url_of_parent and CONVERT_LINKS_IN_HTML and original_href_from_tag:
                parent_lp_key = urlparse(original_url_of_parent)._replace(fragment="").geturl()
                parent_lp = visited_urls_content.get(parent_lp_key)
                if parent_lp and isinstance(parent_lp, str) and not parent_lp.startswith(("S","H","R","U")):
                    if parent_lp not in link_conversion_map: link_conversion_map[parent_lp] = {'original_url': original_url_of_parent, 'links': {}}
                    link_conversion_map[parent_lp]['links'][original_href_from_tag] = predicted_path_initial
            continue

    try:
        time.sleep(WAIT_SECONDS + (random.random() * RANDOM_WAIT_MAX_ADDITIONAL if RANDOM_WAIT_MAX_ADDITIONAL > 0 else 0))
        is_off_host_for_primary_crawl = (parsed_current_for_visit_check.netloc != urlparse(STARTING_URL).netloc)
        fetch_url_for_get = current_url_to_fetch_with_frag
        pre_checked_content_type = None
        actual_url_after_head = None

        if is_off_host_for_primary_crawl and ALLOW_OFF_HOST_DIRECT_LINKED_FILES and original_url_of_parent:
            # print(f"    Off-host link. Checking content type with HEAD for: {fetch_url_for_get}") # Reduced verbosity
            try:
                head_response = session.head(fetch_url_for_get, timeout=10, allow_redirects=True)
                actual_url_after_head = head_response.url
                head_response.raise_for_status()
                pre_checked_content_type = head_response.headers.get('content-type', '').lower()
                path_ext_head = os.path.splitext(urlparse(actual_url_after_head).path)[1]
                is_html_hint_head = (not path_ext_head or path_ext_head.lower() in ['.htm', '.html'] or urlparse(actual_url_after_head).path.endswith('/'))
                if 'text/html' in pre_checked_content_type: is_html_hint_head = True
                predicted_path_after_head, _ = get_potential_local_path(
                    urlparse(actual_url_after_head)._replace(fragment="").geturl(),
                    content_type_hint_for_url=pre_checked_content_type,
                    is_html_hint_for_url= is_html_hint_head,
                    original_parent_url_for_colocation=original_url_of_parent,
                    actual_response_for_cd=head_response if USE_CONTENT_DISPOSITION else None
                )
                if os.path.exists(predicted_path_after_head):
                    existing_file_original_url_head = get_metadata(predicted_path_after_head, 'user.original_url')
                    url_key_after_head_no_frag = urlparse(actual_url_after_head)._replace(fragment="").geturl()
                    if existing_file_original_url_head == url_key_for_visited_check or \
                       existing_file_original_url_head == url_key_after_head_no_frag:
                        print(f"  File already exists (checked after HEAD): {predicted_path_after_head} for URL {actual_url_after_head}. Skipping GET.")
                        visited_urls_content[url_key_for_visited_check] = predicted_path_after_head
                        if url_key_after_head_no_frag != url_key_for_visited_check:
                             visited_urls_content[url_key_after_head_no_frag] = predicted_path_after_head
                        if original_url_of_parent and CONVERT_LINKS_IN_HTML and original_href_from_tag:
                            parent_lp_key = urlparse(original_url_of_parent)._replace(fragment="").geturl()
                            parent_lp = visited_urls_content.get(parent_lp_key)
                            if parent_lp and isinstance(parent_lp, str) and not parent_lp.startswith(("S","H","R","U")):
                                if parent_lp not in link_conversion_map: link_conversion_map[parent_lp] = {'original_url': original_url_of_parent, 'links': {}}
                                link_conversion_map[parent_lp]['links'][original_href_from_tag] = predicted_path_after_head
                        if 'text/html' in pre_checked_content_type:
                             print(f"    Skipping off-host HTML (identified by HEAD, file exists): {actual_url_after_head}")
                             _s_key = f"SKIPPED_OFF_HOST_HTML_EXISTS_{actual_url_after_head}"
                             visited_urls_content[url_key_for_visited_check] = _s_key
                             if url_key_after_head_no_frag != url_key_for_visited_check:
                                 visited_urls_content[url_key_after_head_no_frag] = _s_key
                             continue
                        else: pass # Non-HTML, exists, let it fall through.
                if 'text/html' in pre_checked_content_type:
                    print(f"    Skipping off-host HTML (Content-Type from HEAD): {actual_url_after_head} (Content-Type: {pre_checked_content_type})")
                    _s_key = f"SKIPPED_OFF_HOST_HTML_{actual_url_after_head}"
                    visited_urls_content[url_key_for_visited_check] = _s_key
                    if actual_url_after_head and urlparse(actual_url_after_head)._replace(fragment="").geturl() != url_key_for_visited_check:
                         visited_urls_content[urlparse(actual_url_after_head)._replace(fragment="").geturl()] = _s_key
                    continue
                else: fetch_url_for_get = actual_url_after_head if actual_url_after_head else fetch_url_for_get
            except requests.exceptions.RequestException as head_e:
                print(f"    HEAD request failed for {fetch_url_for_get}: {head_e}. Will attempt GET.");
        
        # Check hostname restriction before GET if we can determine it's HTML
        if STAY_ON_SAME_HOSTNAME and (
            (pre_checked_content_type and 'text/html' in pre_checked_content_type) or 
            (not pre_checked_content_type and (
                not os.path.splitext(urlparse(fetch_url_for_get).path)[1] or 
                urlparse(fetch_url_for_get).path.endswith('/') or
                os.path.splitext(urlparse(fetch_url_for_get).path)[1].lower() in ['.html', '.htm']
            ))
        ) and urlparse(fetch_url_for_get).netloc != urlparse(STARTING_URL).netloc:
            print(f"  Skipping HTML from different primary hostname (pre-GET check): {fetch_url_for_get}")
            _s_key = f"SKIPPED_OFF_START_HOST_HTML_{fetch_url_for_get}"
            visited_urls_content[url_key_for_visited_check] = _s_key
            continue

        response = session.get(fetch_url_for_get, timeout=20, allow_redirects=True)
        actual_fetched_url_with_frag = response.url
        actual_fetched_url_no_frag = urlparse(actual_fetched_url_with_frag)._replace(fragment="").geturl()

        if actual_fetched_url_with_frag != fetch_url_for_get: print(f"  Redirected from {fetch_url_for_get} to {actual_fetched_url_with_frag}")

        if actual_fetched_url_no_frag in visited_urls_content and actual_fetched_url_no_frag != url_key_for_visited_check :
            print(f"  Content from {actual_fetched_url_no_frag} (final URL after GET) already processed/exists.")
            if original_url_of_parent and CONVERT_LINKS_IN_HTML and original_href_from_tag:
                parent_lp_key = urlparse(original_url_of_parent)._replace(fragment="").geturl()
                parent_lp = visited_urls_content.get(parent_lp_key)
                if parent_lp and isinstance(parent_lp,str) and not parent_lp.startswith(("S","H","R","U")):
                    if parent_lp not in link_conversion_map: link_conversion_map[parent_lp] = {'original_url': original_url_of_parent, 'links': {}}
                    link_conversion_map[parent_lp]['links'][original_href_from_tag] = visited_urls_content[actual_fetched_url_no_frag]
            # Make sure original requested URL also points to the same content if it was a redirect leading to already visited content
            visited_urls_content[url_key_for_visited_check] = visited_urls_content[actual_fetched_url_no_frag]
            continue
        
        parsed_actual_url = urlparse(actual_fetched_url_with_frag)
        content_type = pre_checked_content_type or response.headers.get('content-type', '').lower()
        is_html_content = 'text/html' in content_type

        if STAY_ON_SAME_HOSTNAME and is_html_content and parsed_actual_url.netloc != urlparse(STARTING_URL).netloc:
            print(f"  Skipping HTML from different primary hostname: {actual_fetched_url_with_frag}")
            _s_key = f"SKIPPED_OFF_START_HOST_HTML_{actual_fetched_url_with_frag}"
            visited_urls_content[url_key_for_visited_check] = _s_key
            if actual_fetched_url_no_frag != url_key_for_visited_check: visited_urls_content[actual_fetched_url_no_frag] = _s_key
            continue
        
        response.raise_for_status()

        save_filepath_final_calc, _ = get_potential_local_path(
            actual_fetched_url_with_frag,
            content_type_hint_for_url=content_type,
            is_html_hint_for_url=is_html_content,
            original_parent_url_for_colocation=original_url_of_parent,
            actual_response_for_cd=response if USE_CONTENT_DISPOSITION else None
        )
        dir_path_in_target = os.path.dirname(save_filepath_final_calc)
        final_filename = os.path.basename(save_filepath_final_calc)
        os.makedirs(dir_path_in_target, exist_ok=True)

        _, ext_check = os.path.splitext(final_filename)
        if ext_check.lower() in REJECT_EXTENSIONS:
            print(f"  Rejecting (ext {ext_check}): {final_filename} from {actual_fetched_url_with_frag}")
            _r_key = f"REJECTED_EXT_{actual_fetched_url_with_frag}"
            visited_urls_content[url_key_for_visited_check] = _r_key
            if actual_fetched_url_no_frag != url_key_for_visited_check: visited_urls_content[actual_fetched_url_no_frag] = _r_key
            continue

        final_save_path_to_use = os.path.join(dir_path_in_target, final_filename)
        if os.path.exists(final_save_path_to_use):
            # If the original URL we tried to fetch is already mapped, it means we've handled it.
            if url_key_for_visited_check in visited_urls_content and \
               visited_urls_content[url_key_for_visited_check] != f"SKIPPED_NOT_UNDER_STARTING_URL_{url_key_for_visited_check}":
                 print(f"  Redirect led to content already processed. Skipping save for original URL: {url_key_for_visited_check}")
                 continue

            existing_meta_url = get_metadata(final_save_path_to_use, 'user.original_url')
            if existing_meta_url and \
               (existing_meta_url == actual_fetched_url_no_frag or urlparse(existing_meta_url)._replace(fragment="").geturl() == actual_fetched_url_no_frag):
                print(f"  File {final_save_path_to_use} for {actual_fetched_url_no_frag} exists (verified post-GET). Content will be overwritten.")
            else:
                print(f"  Warning: Filename collision at {final_save_path_to_use}. Original URL of existing file: '{existing_meta_url}'. Current URL: '{actual_fetched_url_no_frag}'. Adding suffix.")
                c=1; temp_fp_base = final_save_path_to_use
                name_orig, ext_orig = os.path.splitext(os.path.basename(final_save_path_to_use))
                while os.path.exists(final_save_path_to_use):
                    colliding_meta_url = get_metadata(final_save_path_to_use, 'user.original_url')
                    if colliding_meta_url and \
                       (colliding_meta_url == actual_fetched_url_no_frag or urlparse(colliding_meta_url)._replace(fragment="").geturl() == actual_fetched_url_no_frag) :
                        print(f"    Collision resolved: {final_save_path_to_use} is actually for the current URL. Will overwrite.")
                        break
                    final_save_path_to_use = os.path.join(dir_path_in_target, f"{name_orig}_{c}{ext_orig}")
                    c+=1
        
        print(f"  Saving to: {final_save_path_to_use}")
        with open(final_save_path_to_use,'wb') as f: # Save original binary content first
            f.write(response.content)
        successful_downloads+=1
        
        visited_urls_content[url_key_for_visited_check] = final_save_path_to_use
        if actual_fetched_url_no_frag != url_key_for_visited_check: visited_urls_content[actual_fetched_url_no_frag] = final_save_path_to_use

        set_custom_metadata(final_save_path_to_use, actual_fetched_url_with_frag, original_url_of_parent)

        if original_url_of_parent and CONVERT_LINKS_IN_HTML and original_href_from_tag:
            parent_lp_key = urlparse(original_url_of_parent)._replace(fragment="").geturl()
            parent_lp = visited_urls_content.get(parent_lp_key)
            if parent_lp and isinstance(parent_lp,str) and not parent_lp.startswith(("S","H","R","U")):
                if parent_lp not in link_conversion_map: link_conversion_map[parent_lp] = {'original_url': original_url_of_parent, 'links': {}}
                link_conversion_map[parent_lp]['links'][original_href_from_tag] = final_save_path_to_use

        if is_html_content and current_depth <= MAX_DEPTH:
            html_page_original_url = actual_fetched_url_with_frag
            
            # Re-read from the saved file to parse (it contains original response.content at this point)
            with open(final_save_path_to_use,'rb') as f_r: html_bytes_for_parsing = f_r.read()
            
            # Try decoding with utf-8, fallback to BS default if error
            try:
                soup = BeautifulSoup(html_bytes_for_parsing.decode('utf-8'), 'html.parser')
            except UnicodeDecodeError:
                print(f"    Warning: UTF-8 decode failed for {final_save_path_to_use} during parsing. BeautifulSoup will attempt to guess encoding.")
                soup = BeautifulSoup(html_bytes_for_parsing, 'html.parser') # Let BS guess

            # --- MODIFIED: Deobfuscate emails in soup ---
            if deobfuscate_joomla_emails_in_soup(soup): # Modifies soup in place
                # If modified, re-save the file with the deobfuscated content
                print(f"    Re-saving {final_save_path_to_use} after email deobfuscation.")
                try:
                    with open(final_save_path_to_use, 'w', encoding='utf-8', errors='replace') as f_w:
                        f_w.write(str(soup)) # Write the modified soup as text
                except Exception as e_resave:
                    print(f"      Error re-saving {final_save_path_to_use} after deobfuscation: {e_resave}")
            # --- END MODIFICATION ---
            
            # Now extract links from the (potentially modified) soup
            for tag in soup.find_all(['a','link','script','img'],href=lambda x:x is not None) + \
                       soup.find_all(['script','img'],src=lambda x:x is not None):
                attr='href' if tag.has_attr('href') else 'src'; val=tag[attr]
                clean_v=val.replace('\n','').replace('\t','').strip()
                if not clean_v or clean_v.lower().startswith(('javascript:','mailto:','tel:','#','data:')): continue
                next_url_abs_with_frag=urljoin(html_page_original_url,clean_v)
                p_next_url=urlparse(next_url_abs_with_frag)
                if p_next_url.scheme not in ['http','https']: continue
                next_url_abs_no_frag_for_check = p_next_url._replace(fragment="").geturl()
                in_q=any(q_url_in_q == next_url_abs_with_frag for q_url_in_q,_,_,_ in urls_to_visit)
                if next_url_abs_no_frag_for_check not in visited_urls_content and not in_q:
                    urls_to_visit.append((next_url_abs_with_frag, current_depth+1, html_page_original_url, clean_v))
                elif next_url_abs_no_frag_for_check in visited_urls_content and CONVERT_LINKS_IN_HTML and clean_v:
                    target_lp = visited_urls_content[next_url_abs_no_frag_for_check]
                    if final_save_path_to_use not in link_conversion_map:
                        link_conversion_map[final_save_path_to_use] = {'original_url': html_page_original_url, 'links':{}}
                    link_conversion_map[final_save_path_to_use]['links'][clean_v] = target_lp 
    except requests.exceptions.HTTPError as e:
        err_url = actual_fetched_url_with_frag if 'actual_fetched_url_with_frag' in locals() and actual_fetched_url_with_frag not in [current_url_to_fetch_with_frag, fetch_url_for_get] else fetch_url_for_get
        print(f"  HTTP Error for {current_url_to_fetch_with_frag} (final URL attempted: {err_url}): {e}")
        err_key = f"HTTP_ERROR_{e.response.status_code if e.response else 'Unknown'}_{err_url}"
        visited_urls_content[url_key_for_visited_check] = err_key
        if 'actual_fetched_url_no_frag' in locals() and actual_fetched_url_no_frag != url_key_for_visited_check:
             visited_urls_content[actual_fetched_url_no_frag] = err_key
    except requests.exceptions.RequestException as e:
        err_url = actual_fetched_url_with_frag if 'actual_fetched_url_with_frag' in locals() and actual_fetched_url_with_frag not in [current_url_to_fetch_with_frag, fetch_url_for_get] else fetch_url_for_get
        print(f"  Request Exception for {current_url_to_fetch_with_frag} (final URL attempted: {err_url}): {e}")
        err_key = f"REQUEST_EXCEPTION_{err_url}"
        visited_urls_content[url_key_for_visited_check] = err_key
        if 'actual_fetched_url_no_frag' in locals() and actual_fetched_url_no_frag != url_key_for_visited_check:
             visited_urls_content[actual_fetched_url_no_frag] = err_key
    except Exception as e:
        err_url = actual_fetched_url_with_frag if 'actual_fetched_url_with_frag' in locals() and actual_fetched_url_with_frag not in [current_url_to_fetch_with_frag, fetch_url_for_get] else fetch_url_for_get
        print(f"  An unexpected error occurred with {current_url_to_fetch_with_frag} (final URL attempted: {err_url}): {e}")
        import traceback; traceback.print_exc()
        err_key = f"UNEXPECTED_ERROR_{err_url}"
        visited_urls_content[url_key_for_visited_check] = err_key
        if 'actual_fetched_url_no_frag' in locals() and actual_fetched_url_no_frag != url_key_for_visited_check:
             visited_urls_content[actual_fetched_url_no_frag] = err_key


if CONVERT_LINKS_IN_HTML:
    print("--------------------------------------------------")
    print("Starting second pass for HTML link conversion...")
    processed_link_conversion_files = 0
    for parent_lp, data in link_conversion_map.items():
        if not os.path.exists(parent_lp) or not parent_lp.lower().endswith((".html",".htm")):
            continue
        parent_orig_url=data['original_url']; original_hrefs_to_target_paths=data['links']
        modified_pass2=False
        try:
            # For link conversion, ensure we are reading with proper encoding handling.
            # The file might have been re-written by deobfuscation using utf-8.
            try:
                with open(parent_lp,'r',encoding='utf-8',errors='strict') as f_h: soup_html_content = f_h.read()
            except UnicodeDecodeError:
                print(f"    Warning: UTF-8 strict decode failed for {parent_lp} in link conversion. Trying with errors='replace'.")
                with open(parent_lp,'r',encoding='utf-8',errors='replace') as f_h: soup_html_content = f_h.read()
            
            soup=BeautifulSoup(soup_html_content,'html.parser')
            
            base_tag = soup.find('base', href=True)
            if base_tag:
                print(f"    Warning: <base href='{base_tag['href']}'> found in {parent_lp}. Relative link conversion might be affected if not originally absolute.")

            for tag in soup.find_all(['a','link','script','img'],href=lambda x:x is not None) + \
                       soup.find_all(['script','img'],src=lambda x:x is not None):
                attr='href' if tag.has_attr('href') else 'src'; orig_href_in_tag_from_file=tag[attr]
                target_sp = original_hrefs_to_target_paths.get(orig_href_in_tag_from_file)
                if not target_sp: 
                    abs_href_for_lookup_with_frag = urljoin(parent_orig_url, orig_href_in_tag_from_file.replace('\n','').replace('\t','').strip())
                    target_sp = original_hrefs_to_target_paths.get(abs_href_for_lookup_with_frag)
                    if not target_sp:
                        abs_href_no_frag = urlparse(abs_href_for_lookup_with_frag)._replace(fragment="").geturl()
                        target_sp = original_hrefs_to_target_paths.get(abs_href_no_frag)
                    if not target_sp and orig_href_in_tag_from_file.startswith('//'):
                        scheme_relative_abs = urlparse(parent_orig_url).scheme + ":" + orig_href_in_tag_from_file
                        target_sp = original_hrefs_to_target_paths.get(scheme_relative_abs)
                        if not target_sp:
                             target_sp = original_hrefs_to_target_paths.get(urlparse(scheme_relative_abs)._replace(fragment="").geturl())
                if target_sp and isinstance(target_sp,str) and not target_sp.startswith(("S","H","R","U")):
                    try:
                        abs_target_sp = os.path.abspath(target_sp)
                        abs_parent_dir = os.path.abspath(os.path.dirname(parent_lp))
                        rel_new_href=os.path.relpath(abs_target_sp, start=abs_parent_dir)
                        original_parsed_href = urlparse(orig_href_in_tag_from_file)
                        if original_parsed_href.fragment:
                            rel_new_href += "#" + original_parsed_href.fragment
                        if tag[attr]!=rel_new_href:
                            tag[attr]=rel_new_href
                            modified_pass2=True
                    except ValueError as ve:
                        print(f"    Could not make relative path for '{target_sp}' from '{parent_lp}': {ve}. Link left as is.")
            if modified_pass2:
                with open(parent_lp,'w',encoding='utf-8',errors='replace') as f_w: f_w.write(str(soup))
                print(f"    Rewrote: {parent_lp} with updated links.")
                processed_link_conversion_files +=1
        except UnicodeDecodeError as ude: # Should be less common now with earlier handling
            print(f"    UnicodeDecodeError converting links in {parent_lp}: {ude}. File may not be UTF-8. Skipping conversion for this file.")
        except Exception as e_conv:
            print(f"    Error converting links in {parent_lp}: {e_conv}")
            import traceback; traceback.print_exc()

print("--------------------------------------------------")
print(f"Download process finished. {download_attempt_count} URLs processed. {successful_downloads} files successfully saved/verified.")
if CONVERT_LINKS_IN_HTML:
    print(f"{processed_link_conversion_files} HTML files had their links updated.")
print(f"Files saved in: {TARGET_DIR_BASE}")