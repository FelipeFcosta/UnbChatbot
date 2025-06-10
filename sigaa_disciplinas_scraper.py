import time
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup, NavigableString, Tag
import multiprocessing
import subprocess
import html # For escaping

# --- General Configuration ---
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
NIVEL_ENSINO_GRADUACAO = "G"

# --- Configuration for Componentes Curriculares (Programas/Syllabi) ---
COMPONENTES_URL = "https://sigaa.unb.br/sigaa/public/componentes/busca_componentes.jsf"
COMPONENTES_MAIN_TARGET_DIR_BASE = "sigaa_clone/componentes_curriculares2"
COMPONENTES_TIPO_DISCIPLINA = "2"
COMPONENTES_UNIDADES = {
    "508": "CIC_Depto_Ciencias_da_Computacao",
    "443": "ENE_Depto_Engenharia_Eletrica"
}
NO_PROGRAMA_SUFFIX = "_NO_PROGRAMA.html"
SUCCESS_SUFFIX = ".html"
MAX_WORKERS = 10

# --- Configuration for Turmas (Classes/Offerings) ---
TURMAS_URL = "https://sigaa.unb.br/sigaa/public/turmas/listar.jsf"
TURMAS_MAIN_TARGET_DIR_BASE = "sigaa_clone/turmas2"
TURMAS_ANO = "2025"
TURMAS_PERIODO = "1"
TURMAS_UNIDADES = {
    "508": "CIC_Depto_Ciencias_da_Computacao",
    "443": "ENE_Depto_Engenharia_Eletrica"
}

# --- Helper Functions ---

# Map for displaying component detail keys in HTML
COMPONENT_DETAIL_DISPLAY_NAMES = {
    'Codigo': 'Código',
    'Nome': 'Nome',
    'PreRequisitos': 'Pré-Requisitos',
    'CoRequisitos': 'Co-Requisitos',
    'Equivalencias': 'Equivalências',
    'EhPreRequisitoPara': 'É pré-requisito para'
    # Add other keys here if they are scraped into component_details
    # and need specific display names.
}
# Ordered list of keys for displaying component details
COMPONENT_DETAIL_ORDERED_KEYS = ['Codigo', 'Nome', 'PreRequisitos', 'CoRequisitos', 'Equivalencias', 'EhPreRequisitoPara']


def sanitize_filename(filename_str):
    if not isinstance(filename_str, str): filename_str = str(filename_str)
    filename_str = re.sub(r'[\\/:*?"<>|]', '_', filename_str)
    filename_str = re.sub(r'_+', '_', filename_str).strip('_ ')
    return filename_str if filename_str else "downloaded_file"

def generate_componente_filepaths(current_unit_target_dir, codigo, nome):
    filename_base = sanitize_filename(f"{codigo}_{nome}")
    success_filepath = os.path.join(current_unit_target_dir, f"{filename_base}{SUCCESS_SUFFIX}")
    no_programa_filepath = os.path.join(current_unit_target_dir, f"{filename_base}{NO_PROGRAMA_SUFFIX}")
    counter = 1
    numbered_success_filepaths = []
    name_part, ext_part = os.path.splitext(os.path.join(current_unit_target_dir, filename_base))
    while counter < 10: numbered_success_filepaths.append(f"{name_part}_{counter}{SUCCESS_SUFFIX}"); counter += 1
    return success_filepath, no_programa_filepath, numbered_success_filepaths

def create_no_programa_placeholder_html(filepath, codigo, nome, detalhes_info=None):
    detalhes_html_str = ""
    if detalhes_info:
        detalhes_html_str = '<div id="scraped-component-details-placeholder" style="padding:10px; border:1px solid #eee; margin-bottom:15px; background-color:#fdfdfd;">'
        detalhes_html_str += '<h3 style="margin-top:0; border-bottom:1px solid #ddd; padding-bottom:5px;">Detalhes do Componente</h3><ul>'
        
        temp_details_placeholder = detalhes_info.copy()
        for internal_key in COMPONENT_DETAIL_ORDERED_KEYS:
            if internal_key in temp_details_placeholder:
                value = temp_details_placeholder.pop(internal_key)
                display_key = COMPONENT_DETAIL_DISPLAY_NAMES.get(internal_key, internal_key) # Fallback to internal key
                safe_value = html.escape(str(value if value is not None else "-"))
                detalhes_html_str += f'<li style="margin-bottom:3px;"><strong>{html.escape(display_key.replace(":", ""))}:</strong> {safe_value}</li>'
        
        # Add any remaining details not in the ordered list
        for internal_key, value in temp_details_placeholder.items():
            display_key = COMPONENT_DETAIL_DISPLAY_NAMES.get(internal_key, internal_key)
            safe_value = html.escape(str(value if value is not None else "-"))
            detalhes_html_str += f'<li style="margin-bottom:3px;"><strong>{html.escape(display_key.replace(":", ""))}:</strong> {safe_value}</li>'
            
        detalhes_html_str += "</ul></div><hr/>"

    html_content = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8"><title>Programa Não Cadastrado - {html.escape(codigo)} {html.escape(nome)}</title><style>body {{ font-family: sans-serif; padding: 20px; }} h1 {{ color: #cc0000; text-align:center; }} ul {{ list-style-type: none; padding-left: 0; }} li {{ margin-bottom: 5px; }}</style></head><body>{detalhes_html_str}<h1>Programa Não Cadastrado</h1><p>O componente curricular <strong>{html.escape(codigo)} - {html.escape(nome)}</strong> não possui um programa cadastrado no SIGAA.</p></body></html>"""
    try:
        with open(filepath, "w", encoding="utf-8") as f: f.write(html_content)
    except Exception as e: print(f"    Error creating placeholder file {os.path.basename(filepath)}: {e}")


def handle_cookie_consent(driver, wait_time=7):
    try:
        cookie_button_xpath = "//dialog[@id='sigaa-cookie-consent']//button[normalize-space()='Ciente']"
        cookie_button = WebDriverWait(driver, wait_time).until(EC.element_to_be_clickable((By.XPATH, cookie_button_xpath)))
        driver.execute_script("arguments[0].click();", cookie_button)
        time.sleep(1.5); return True
    except TimeoutException: return False
    except Exception as e_cookie: print(f"    An error occurred with cookie consent: {e_cookie}."); return False

def _set_attr(filepath, attr_name, attr_value):
    try:
        value_str = str(attr_value if attr_value is not None else "")
        subprocess.run(['setfattr', '-n', f'user.{attr_name}', '-v', value_str, '-e', 'text', filepath],
                       check=True, capture_output=True, timeout=5)
        return True
    except subprocess.CalledProcessError: 
        try:
            subprocess.run(['setfattr', '-n', f'user.{attr_name}', '-v', value_str, filepath],
                           check=True, capture_output=True, timeout=5)
            return True
        except Exception: pass 
    except subprocess.TimeoutExpired: print(f"      Metadata WARNING: setfattr TIMEOUT for '{attr_name}' on {os.path.basename(filepath)}")
    except FileNotFoundError:
        if not getattr(_set_attr, "setfattr_not_found_warning_shown", False):
            print(f"      Metadata WARNING: 'setfattr' command not found. Cannot set extended attributes."); _set_attr.setfattr_not_found_warning_shown = True
    except Exception as e_setattr: print(f"      Metadata WARNING: Unexpected error setting attr '{attr_name}' on {os.path.basename(filepath)}: {e_setattr}")
    return False

def set_sigaa_componente_metadata(filepath, base_url, unit_val, unit_display_name, comp_id_js, comp_codigo, comp_nome, prerequisitos, corequisitos, equivalencias, eh_prerequisito_para, has_programa):
    _set_attr(filepath, "original_url", base_url)
    _set_attr(filepath, "unit_value", unit_val)
    _set_attr(filepath, "unit_name", unit_display_name)
    _set_attr(filepath, "component_id_js", comp_id_js)
    _set_attr(filepath, "component_code", comp_codigo)
    _set_attr(filepath, "component_name", comp_nome)
    if prerequisitos and prerequisitos != '-': _set_attr(filepath, "sigaa_prerequisitos", prerequisitos)
    if corequisitos and corequisitos != '-': _set_attr(filepath, "sigaa_corequisitos", corequisitos)
    if equivalencias and equivalencias != '-': _set_attr(filepath, "sigaa_equivalencias", equivalencias)
    if eh_prerequisito_para and eh_prerequisito_para != '-': _set_attr(filepath, "sigaa_eh_prerequisito_para", eh_prerequisito_para)
    _set_attr(filepath, "not_available", str(not has_programa).lower())


def set_sigaa_turmas_metadata(filepath, base_url, unit_val, unit_name, ano, periodo):
    _set_attr(filepath, "original_url", base_url)
    _set_attr(filepath, "unit_value", unit_val)
    _set_attr(filepath, "unit_name", unit_name)
    _set_attr(filepath, "year", ano)
    _set_attr(filepath, "period", periodo)

# --- Worker and Main Scraping Functions ---
def process_single_componente_programa(args):
    unit_value, _, componente_index, componente_id_from_js, codigo_from_list, nome_from_list, current_target_dir = args
    actual_unit_display_name = COMPONENTES_UNIDADES.get(unit_value, "Unidade_Desconhecida_Metadado") 
    print(f"      Worker (PID {os.getpid()}) START: {codigo_from_list} - {nome_from_list} (JS ID: {componente_id_from_js})")

    success_fp, no_programa_fp, numbered_fps = generate_componente_filepaths(current_target_dir, codigo_from_list, nome_from_list)
    filename_base_for_worker = sanitize_filename(f"{codigo_from_list}_{nome_from_list}")

    if os.path.exists(success_fp) or os.path.exists(no_programa_fp) or any(os.path.exists(nfp) for nfp in numbered_fps):
        print(f"      Worker (PID {os.getpid()}): SKIPPING '{nome_from_list}' - file already exists.")
        return f"Skipped (exists): {codigo_from_list} - {nome_from_list}"

    worker_driver = None; component_details = {}
    try:
        chrome_options_worker = Options(); chrome_options_worker.add_argument("--headless")
        chrome_options_worker.add_argument("--disable-gpu"); chrome_options_worker.add_argument("--no-sandbox")
        chrome_options_worker.add_argument("--window-size=1366,768"); chrome_options_worker.add_argument(f"user-agent={USER_AGENT}")
        s = Service(ChromeDriverManager().install()); worker_driver = webdriver.Chrome(service=s, options=chrome_options_worker)
        wait = WebDriverWait(worker_driver, 25); short_wait = WebDriverWait(worker_driver, 10)

        worker_driver.get(COMPONENTES_URL); time.sleep(1.0); handle_cookie_consent(worker_driver)
        Select(wait.until(EC.presence_of_element_located((By.ID, "form:nivel")))).select_by_value(NIVEL_ENSINO_GRADUACAO); time.sleep(0.1)
        Select(wait.until(EC.presence_of_element_located((By.ID, "form:tipo")))).select_by_value(COMPONENTES_TIPO_DISCIPLINA); time.sleep(0.1)
        tipo_checkbox = wait.until(EC.visibility_of_element_located((By.ID, "form:checkTipo")))
        if not tipo_checkbox.is_selected(): worker_driver.execute_script("arguments[0].click();", tipo_checkbox)
        time.sleep(0.1)
        unidade_select_el = Select(wait.until(EC.presence_of_element_located((By.ID, "form:unidades"))))
        unidade_select_el.select_by_value(unit_value)
        try: actual_unit_display_name = str([opt.text for opt in unidade_select_el.options if opt.get_attribute("value") == unit_value][0].strip())
        except IndexError: print(f"      Worker: Could not get display name for unit value {unit_value}")
        time.sleep(0.1)
        unidade_checkbox = wait.until(EC.visibility_of_element_located((By.ID, "form:checkUnidade")))
        if not unidade_checkbox.is_selected(): worker_driver.execute_script("arguments[0].click();", unidade_checkbox)
        time.sleep(0.2)
        buscar_button = wait.until(EC.element_to_be_clickable((By.ID, "form:btnBuscarComponentes")))
        worker_driver.execute_script("arguments[0].click();", buscar_button)
        results_form_locator = (By.ID, "formListagemComponentes")
        search_results_page_indicator_element = wait.until(EC.presence_of_element_located(results_form_locator))
        
        all_magnifier_links_xpath = "//form[@id='formListagemComponentes']//table[@class='listagem']//tbody/tr/td[last()]/a[img[@src='/sigaa/img/view.gif']]"
        specific_magnifier_link_xpath = f"({all_magnifier_links_xpath})[{componente_index + 1}]"
        try:
            magnifier_link = wait.until(EC.element_to_be_clickable((By.XPATH, specific_magnifier_link_xpath)))
            worker_driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center', inline: 'nearest'});", magnifier_link); time.sleep(0.5)
            worker_driver.execute_script("arguments[0].click();", magnifier_link)
        except Exception as e_click_magnifier: return f"Failed to click magnifier for {codigo_from_list}: {e_click_magnifier}"

        detalhes_page_indicator_xpath = "//table[@class='visualizacao']//caption[contains(text(),'Dados Gerais do Componente Curricular')]"
        wait.until(EC.presence_of_element_located((By.XPATH, detalhes_page_indicator_xpath)))
        try: wait.until(EC.staleness_of(search_results_page_indicator_element))
        except TimeoutException: pass
        time.sleep(0.5)
        detalhes_soup = BeautifulSoup(worker_driver.page_source, 'html.parser')
        dados_gerais_table = next((t for t in detalhes_soup.find_all('table', class_='visualizacao') if t.find('caption') and "Dados Gerais do Componente Curricular" in t.find('caption').get_text(strip=True)), None)
        
        if dados_gerais_table:
            tbody = dados_gerais_table.find('tbody') or dados_gerais_table
            for row in tbody.find_all('tr', recursive=False):
                th_tag = row.find('th', recursive=False)
                td_tag = row.find('td', recursive=False)
                if th_tag and td_tag:
                    header_text_from_th = th_tag.get_text(strip=True).replace(':', '') # Original header text

                    # Initialize value to be set
                    value_to_set = None
                    internal_key_for_detail = None

                    # Handle specific fields for detailed parsing or direct storage
                    if header_text_from_th == "C\u00f3digo":
                        internal_key_for_detail = "Codigo"
                        value_to_set = ' '.join(td_tag.get_text(separator=' ', strip=True).split())
                    elif header_text_from_th == "Nome":
                        internal_key_for_detail = "Nome"
                        value_to_set = ' '.join(td_tag.get_text(separator=' ', strip=True).split())
                    elif header_text_from_th in ["Pr\u00e9-Requisitos", "Co-Requisitos", "Equival\u00eancias"]:
                        if header_text_from_th == "Pr\u00e9-Requisitos": internal_key_for_detail = "PreRequisitos"
                        elif header_text_from_th == "Co-Requisitos": internal_key_for_detail = "CoRequisitos"
                        elif header_text_from_th == "Equival\u00eancias": internal_key_for_detail = "Equivalencias"
                        
                        plain_text_td = td_tag.get_text(strip=True)
                        if plain_text_td == "-":
                            value_to_set = "-"
                        else:
                            current_parts = []
                            for element in td_tag.descendants:
                                if isinstance(element, Tag) and element.name == 'acronym':
                                    code = element.get_text(strip=True)
                                    title_attr = element.get('title', '').strip()
                                    name_candidate = title_attr.replace(code, "").strip(" -").strip()
                                    if not name_candidate and title_attr: name_candidate = title_attr 
                                    elif not name_candidate and not title_attr: name_candidate = code 
                                    current_parts.append(f"{name_candidate.strip()} ({code})")
                                elif isinstance(element, NavigableString):
                                    text_content = str(element).strip()
                                    words_in_text_node = re.findall(r'\b\w+\b', text_content.upper())
                                    for word in words_in_text_node:
                                        if word == 'E' or word == 'OU': current_parts.append(word)
                            if current_parts:
                                value_to_set = ' '.join(current_parts)
                            else: 
                                value_to_set = plain_text_td if plain_text_td else "-"
                    
                    # If an internal key was assigned (meaning it's a field we want)
                    if internal_key_for_detail:
                        component_details[internal_key_for_detail] = value_to_set if value_to_set else "-"

        eh_prerequisito_para_list = []
        eh_prerequisito_para_caption_text_search = "outros componentes que têm esse componente como pré-requisito"
        prereq_for_caption = None
        for cap in detalhes_soup.find_all('caption'):
            cap_text_normalized = html.unescape(cap.get_text(strip=True)).lower()
            if eh_prerequisito_para_caption_text_search in cap_text_normalized:
                prereq_for_caption = cap; break
        
        if prereq_for_caption:
            prereq_for_table = prereq_for_caption.find_parent('table')
            if prereq_for_table:
                tbody_pr = prereq_for_table.find('tbody') or prereq_for_table
                for row_pr in tbody_pr.find_all('tr'):
                    td_pr = row_pr.find('td')
                    if td_pr:
                        component_text = td_pr.get_text(strip=True)
                        if component_text: eh_prerequisito_para_list.append(component_text)

        component_details['EhPreRequisitoPara'] = " | ".join(eh_prerequisito_para_list) if eh_prerequisito_para_list else "-"
        
        worker_driver.back()
        search_results_page_indicator_element = wait.until(EC.presence_of_element_located(results_form_locator))
        wait.until(EC.presence_of_element_located((By.XPATH, f"({all_magnifier_links_xpath})[1]"))); time.sleep(1)

        all_notepad_links_xpath = "//form[@id='formListagemComponentes']//table[@class='listagem']//tbody/tr/td[last()]/a[@title='Programa Atual do Componente']"
        specific_notepad_link_xpath = f"({all_notepad_links_xpath})[{componente_index + 1}]"
        try:
            notepad_link = wait.until(EC.element_to_be_clickable((By.XPATH, specific_notepad_link_xpath)))
            worker_driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center', inline: 'nearest'});", notepad_link); time.sleep(0.5)
            worker_driver.execute_script("arguments[0].click();", notepad_link)
        except Exception as e_click_notepad:
            create_no_programa_placeholder_html(no_programa_fp, codigo_from_list, nome_from_list, component_details)
            set_sigaa_componente_metadata(no_programa_fp, COMPONENTES_URL, unit_value, actual_unit_display_name, componente_id_from_js, codigo_from_list, nome_from_list, 
                                          component_details.get('PreRequisitos'), component_details.get('CoRequisitos'), component_details.get('Equivalencias'),
                                          component_details.get('EhPreRequisitoPara'),
                                          has_programa=False)
            return f"Failed click notepad (details extracted) for {codigo_from_list}: {e_click_notepad}"
        
        error_panel_xpath = "//div[@id='painel-erros']//ul[@class='erros']//li[contains(text(),'não possui um programa cadastrado')]"
        programa_page_indicator_xpath = "//div[@id='relatorio-cabecalho']"
        wait.until(EC.staleness_of(search_results_page_indicator_element)); time.sleep(0.5)
        is_error_page = False; target_filepath_for_metadata = None
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, programa_page_indicator_xpath)))
        except TimeoutException:
            try: short_wait.until(EC.visibility_of_element_located((By.XPATH, error_panel_xpath))); is_error_page = True
            except TimeoutException:
                debug_filename = sanitize_filename(f"_debug_worker_unexpected_programa_page_{codigo_from_list}_{nome_from_list}.html")
                with open(os.path.join(current_target_dir, debug_filename), "w", encoding="utf-8") as f_debug:
                    if worker_driver: f_debug.write(worker_driver.page_source)
                return f"Unexpected content on programa page for {codigo_from_list}"
        
        if is_error_page:
            target_filepath_for_metadata = no_programa_fp
            create_no_programa_placeholder_html(no_programa_fp, codigo_from_list, nome_from_list, component_details)
        else:
            page_source_full = worker_driver.page_source; soup = BeautifulSoup(page_source_full, 'html.parser')
            
            head = soup.head
            if not head:
                head = soup.new_tag("head"); html_tag = soup.html or soup.find() or soup
                if html_tag: html_tag.insert(0, head) 
            
            meta_charset = head.find("meta", attrs={"charset": True})
            if meta_charset: meta_charset['charset'] = "UTF-8"
            else:
                meta_http_equiv = head.find("meta", attrs={"http-equiv": re.compile("Content-Type", re.I)})
                if meta_http_equiv: meta_http_equiv['content'] = "text/html; charset=UTF-8"
                else: head.insert(0, soup.new_tag("meta", charset="UTF-8"))

            title_tag = soup.find('title'); _ = title_tag.extract() if title_tag else None
            header_div = soup.find('div', id='relatorio-cabecalho');_ = header_div.extract() if header_div else None 
            footer = soup.find('div', id='relatorio-rodape'); _ = footer.extract() if footer else None
            for script_tag in soup.find_all("script", src=re.compile(r"STICookieConsent\.js|google-analytics\.com")): script_tag.extract()
            cookie_dialog = soup.find('dialog', id='sigaa-cookie-consent'); _ = cookie_dialog.extract() if cookie_dialog else None
            
            if component_details:
                details_div_str = '<div id="scraped-component-details" style="padding:15px; border:1px solid #ddd; margin-bottom:20px; background-color:#f9f9f9;"><h3 style="margin-top:0; border-bottom:1px solid #ccc; padding-bottom:5px;">Detalhes do Componente Curricular</h3><ul>'
                
                temp_details_for_html = component_details.copy() 
                for internal_key in COMPONENT_DETAIL_ORDERED_KEYS:
                    if internal_key in temp_details_for_html:
                        value = temp_details_for_html.pop(internal_key)
                        display_key = COMPONENT_DETAIL_DISPLAY_NAMES.get(internal_key, internal_key) 
                        details_div_str += f'<li style="margin-bottom:5px;"><strong>{html.escape(display_key.replace(":", ""))}:</strong> {html.escape(str(value if value is not None else "-"))}</li>'
                
                # Add any remaining details not in the ordered list
                for internal_key, value in temp_details_for_html.items():
                    display_key = COMPONENT_DETAIL_DISPLAY_NAMES.get(internal_key, internal_key)
                    details_div_str += f'<li style="margin-bottom:5px;"><strong>{html.escape(display_key.replace(":", ""))}:</strong> {html.escape(str(value if value is not None else "-"))}</li>'

                details_div_str += "</ul></div>"; details_soup_tag = BeautifulSoup(details_div_str, 'html.parser').div
                body_tag = soup.body or soup.new_tag("body"); soup.html.append(body_tag) if soup.html and not soup.body else (soup.append(body_tag) if not soup.html and not soup.body else None)
                relatorio_container = soup.find('div', id='relatorio-container')
                if relatorio_container: relatorio_container.insert_before(details_soup_tag)
                elif body_tag: body_tag.insert(0, details_soup_tag)
            
            page_content_to_save = str(soup)
            counter = 1; temp_filepath = success_fp
            while os.path.exists(temp_filepath): name_part, _ = os.path.splitext(filename_base_for_worker); temp_filepath = os.path.join(current_target_dir, f"{name_part}_{counter}{SUCCESS_SUFFIX}"); counter += 1
            success_fp_final = temp_filepath; target_filepath_for_metadata = success_fp_final
            with open(success_fp_final, "w", encoding="utf-8") as f: f.write(page_content_to_save)
        
        if target_filepath_for_metadata:
            set_sigaa_componente_metadata(target_filepath_for_metadata, COMPONENTES_URL, unit_value, actual_unit_display_name, componente_id_from_js, 
                                          component_details.get('Codigo', codigo_from_list), 
                                          component_details.get('Nome', nome_from_list),   
                                          component_details.get('PreRequisitos'), 
                                          component_details.get('CoRequisitos'), 
                                          component_details.get('Equivalencias'),
                                          component_details.get('EhPreRequisitoPara'),
                                          has_programa=(not is_error_page))
        return f"{'No programa (placeholder with details)' if is_error_page else 'Success (with details)'}: {codigo_from_list} - {nome_from_list}"
    except Exception as e:
        error_message = f"Worker error for {codigo_from_list} - {nome_from_list}: {str(e)[:250]}"
        print(f"      {error_message}") 
        if component_details and not os.path.exists(no_programa_fp) and not os.path.exists(success_fp):
             create_no_programa_placeholder_html(no_programa_fp, codigo_from_list, nome_from_list, component_details)
             set_sigaa_componente_metadata(no_programa_fp, COMPONENTES_URL, unit_value, actual_unit_display_name, componente_id_from_js, codigo_from_list, nome_from_list, 
                                           component_details.get('PreRequisitos'), component_details.get('CoRequisitos'), component_details.get('Equivalencias'),
                                           component_details.get('EhPreRequisitoPara'),
                                           has_programa=False)
        return error_message
    finally:
        if worker_driver: worker_driver.quit()

def scrape_componentes_curriculares_main_process(driver):
    print(f"\n{'#'*10} Starting: Scrape Componentes Curriculares (Programas) - Main Process {'#'*10}")
    os.makedirs(COMPONENTES_MAIN_TARGET_DIR_BASE, exist_ok=True)
    print(f"Top-level output directory for Componentes/Programas: {COMPONENTES_MAIN_TARGET_DIR_BASE}")
    wait = WebDriverWait(driver, 25)
    for unit_value, unit_folder_name in COMPONENTES_UNIDADES.items():
        current_target_dir = os.path.join(COMPONENTES_MAIN_TARGET_DIR_BASE, unit_folder_name)
        os.makedirs(current_target_dir, exist_ok=True)
        print(f"\n{'='*20} Main Process - Unit (Componentes): {unit_folder_name} (Value: {unit_value}) {'='*20}")
        componentes_to_process_for_this_unit = []
        try:
            driver.get(COMPONENTES_URL); time.sleep(1.5); handle_cookie_consent(driver)
            Select(wait.until(EC.presence_of_element_located((By.ID, "form:nivel")))).select_by_value(NIVEL_ENSINO_GRADUACAO); time.sleep(0.3)
            Select(wait.until(EC.presence_of_element_located((By.ID, "form:tipo")))).select_by_value(COMPONENTES_TIPO_DISCIPLINA); time.sleep(0.2)
            tipo_checkbox = wait.until(EC.visibility_of_element_located((By.ID, "form:checkTipo")))
            if not tipo_checkbox.is_selected(): driver.execute_script("arguments[0].click();", tipo_checkbox)
            time.sleep(0.3)
            Select(wait.until(EC.presence_of_element_located((By.ID, "form:unidades")))).select_by_value(unit_value); time.sleep(0.2)
            unidade_checkbox = wait.until(EC.visibility_of_element_located((By.ID, "form:checkUnidade")))
            if not unidade_checkbox.is_selected(): driver.execute_script("arguments[0].click();", unidade_checkbox)
            time.sleep(0.3)
            buscar_button = wait.until(EC.element_to_be_clickable((By.ID, "form:btnBuscarComponentes")))
            driver.execute_script("arguments[0].click();", buscar_button)
            wait.until(EC.presence_of_element_located((By.ID, "formListagemComponentes")))
            results_table_first_row_xpath = "//form[@id='formListagemComponentes']//table[@class='listagem']//tbody/tr[1]"
            try: wait.until(EC.presence_of_element_located((By.XPATH, results_table_first_row_xpath)))
            except TimeoutException: print(f"  Main Process: No componentes found for unit {unit_folder_name}. Skipping."); continue
            time.sleep(2.5) 
            rows = driver.find_elements(By.XPATH, "//form[@id='formListagemComponentes']//table[@class='listagem']//tbody/tr")
            id_componente_regex = re.compile(r"'idComponente'\s*:\s*'(\d+)'") 
            for i, row in enumerate(rows):
                try:
                    notepad_link_element = row.find_element(By.XPATH, ".//a[@title='Programa Atual do Componente']")
                    onclick_notepad_attr = notepad_link_element.get_attribute("onclick")
                    componente_id_match = id_componente_regex.search(onclick_notepad_attr)
                    componente_id_from_js = componente_id_match.group(1) if componente_id_match else f"UNKNOWN_JS_ID_{i}"                   
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        codigo = cells[0].text.strip(); nome = cells[1].text.strip()
                        if codigo and nome: componentes_to_process_for_this_unit.append((unit_value, unit_folder_name, i, componente_id_from_js, codigo, nome, current_target_dir))
                except NoSuchElementException: pass
                except Exception as e_row_parse: print(f"    Main Process: Error parsing row {i} for unit {unit_folder_name}: {e_row_parse}")
            print(f"  Main Process: {len(componentes_to_process_for_this_unit)} componentes with programa links queued for unit '{unit_folder_name}'.")
        except Exception as e_unit_search: print(f"  Main Process: Error during initial search for unit {unit_folder_name}: {e_unit_search}"); continue
        if componentes_to_process_for_this_unit:
            pool = None; results = [] 
            try:
                pool = multiprocessing.Pool(processes=MAX_WORKERS)
                results = pool.map(process_single_componente_programa, componentes_to_process_for_this_unit)
                for res in results: print(f"    Worker Pool Result: {res}")
            except Exception as e_pool: print(f"  Error during multiprocessing for unit {unit_folder_name}: {e_pool}")
            finally:
                if pool: pool.close(); pool.join()
        print(f"  Finished processing unit {unit_folder_name} for Componentes.")
    print(f"\n{'#'*10} Finished: Scrape Componentes Curriculares (Programas) - Main Process {'#'*10}")

def scrape_turmas(driver):
    print(f"\n{'#'*10} Starting: Scrape Turmas (Ofertas) {'#'*10}")
    os.makedirs(TURMAS_MAIN_TARGET_DIR_BASE, exist_ok=True)
    print(f"Top-level output directory for Turmas/Ofertas: {TURMAS_MAIN_TARGET_DIR_BASE}")
    wait = WebDriverWait(driver, 20)
    year_period_folder_name = f"{TURMAS_ANO}-{TURMAS_PERIODO}"
    current_year_period_target_dir = os.path.join(TURMAS_MAIN_TARGET_DIR_BASE, year_period_folder_name)
    os.makedirs(current_year_period_target_dir, exist_ok=True)
    print(f"  Saving all turmas for {year_period_folder_name} into: {current_year_period_target_dir}")
    for unit_value, unit_folder_name_for_file in TURMAS_UNIDADES.items():
        print(f"\n  {'='*15} Processing Unit (Turmas): {unit_folder_name_for_file} (Value: {unit_value}) for {year_period_folder_name} {'='*15}")
        turmas_results_filename = sanitize_filename(f"{unit_folder_name_for_file}.html")
        turmas_results_filepath = os.path.join(current_year_period_target_dir, turmas_results_filename)
        if os.path.exists(turmas_results_filepath):
            print(f"    Skipping Turmas for '{unit_folder_name_for_file}' - file already exists: {turmas_results_filename}")
            continue
        try:
            driver.get(TURMAS_URL); time.sleep(1.5); handle_cookie_consent(driver)
            Select(wait.until(EC.presence_of_element_located((By.ID, "formTurma:inputNivel")))).select_by_value(NIVEL_ENSINO_GRADUACAO); time.sleep(0.3)
            unidade_select_turmas = Select(wait.until(EC.presence_of_element_located((By.ID, "formTurma:inputDepto"))))
            unidade_select_turmas.select_by_value(unit_value)
            actual_unit_display_name_turmas = unit_folder_name_for_file 
            try: actual_unit_display_name_turmas = str(unidade_select_turmas.first_selected_option.text.strip())
            except Exception: print(f"    Warning: Could not get display name for turmas unit value {unit_value}, using folder name.")
            time.sleep(0.3)
            ano_input = wait.until(EC.presence_of_element_located((By.ID, "formTurma:inputAno")))
            driver.execute_script("arguments[0].value = arguments[1];", ano_input, TURMAS_ANO); time.sleep(0.2)
            Select(wait.until(EC.presence_of_element_located((By.ID, "formTurma:inputPeriodo")))).select_by_value(TURMAS_PERIODO); time.sleep(0.3)
            buscar_turmas_button_name = "formTurma:j_id_jsp_1370969402_11"
            try:
                buscar_button = wait.until(EC.element_to_be_clickable((By.NAME, buscar_turmas_button_name)))
            except TimeoutException:
                buscar_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@value='Buscar Turmas']"))) # Fallback

            driver.execute_script("arguments[0].click();", buscar_button)
            turmas_results_div_xpath = "//div[@id='turmasAbertas']"
            turmas_table_caption_xpath = f"{turmas_results_div_xpath}//table[@class='listagem']//caption[contains(text(),'turmas encontrada(s)')]"
            try:
                turmas_div_element = wait.until(EC.presence_of_element_located((By.XPATH, turmas_results_div_xpath)))
                wait.until(EC.presence_of_element_located((By.XPATH, turmas_table_caption_xpath)))
                print(f"    Turmas results loaded for {unit_folder_name_for_file}.")
                time.sleep(1.5)
                table_container_html = turmas_div_element.get_attribute('outerHTML')
                full_html_to_save = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8"><title>Turmas Encontradas - {unit_folder_name_for_file} - {TURMAS_ANO}.{TURMAS_PERIODO}</title><style>body {{ font-family: sans-serif; margin: 20px; }} table.listagem {{ border-collapse: collapse; width: 95%; margin: 20px auto; font-size: 0.9em; }} table.listagem th, table.listagem td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }} table.listagem th {{ background-color: #f2f2f2; }} table.listagem caption {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; text-align: center;}} tr.agrupador td {{ background-color: #C8D5EC; font-weight: bold; }} .tituloDisciplina {{ font-weight: bold; }}</style></head><body><h2>Turmas para Unidade: {html.escape(actual_unit_display_name_turmas)}, Ano/Período: {TURMAS_ANO}.{TURMAS_PERIODO}</h2>{table_container_html}</body></html>"""
                with open(turmas_results_filepath, "w", encoding="utf-8") as f: f.write(full_html_to_save)
                print(f"    Saved Turmas table to: {turmas_results_filename}")
                set_sigaa_turmas_metadata(turmas_results_filepath, TURMAS_URL, unit_value, actual_unit_display_name_turmas, TURMAS_ANO, TURMAS_PERIODO)
            except TimeoutException:
                print(f"    Timeout: No Turmas results table/div found for {unit_folder_name_for_file}.")
                debug_turmas_filename = sanitize_filename(f"_debug_no_turmas_table_{unit_folder_name_for_file}_{TURMAS_ANO}_{TURMAS_PERIODO}.html")
                if driver:
                    with open(os.path.join(current_year_period_target_dir, debug_turmas_filename), "w", encoding="utf-8") as f_debug: f_debug.write(driver.page_source)
                    print(f"    Saved full page debug for no turmas: {debug_turmas_filename}")
                continue
        except Exception as e_unit_turmas:
            print(f"  An error occurred while processing Turmas for unit {unit_folder_name_for_file}: {e_unit_turmas}")
            print(f"  Skipping Turmas for unit '{unit_folder_name_for_file}' due to error.")
    print(f"\n{'#'*10} Finished: Scrape Turmas (Ofertas) {'#'*10}")

def main_controller():
    chrome_options = Options(); chrome_options.add_argument("--disable-gpu"); chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080"); chrome_options.add_argument(f"user-agent={USER_AGENT}")
    # chrome_options.add_argument("--headless") 
    driver = None
    try:
        try:
            print("Setting up Main ChromeDriver..."); service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options); print("Main ChromeDriver setup successful.")
        except Exception as e_main_driver: print(f"Fatal Error setting up Main ChromeDriver: {e_main_driver}"); return
        
        scrape_componentes_curriculares_main_process(driver)
        scrape_turmas(driver)
        
    except Exception as e_global:
        print(f"An unexpected global error occurred in main_controller: {e_global}")
        if driver:
            timestamp = time.strftime("%Y%m%d-%H%M%S"); error_filename_base = f"error_unexpected_main_controller_{timestamp}"
            try:
                base_clone_dir = "sigaa_clone"; os.makedirs(base_clone_dir, exist_ok=True)
                debug_dir_for_global_error = os.path.join(base_clone_dir, "global_errors"); os.makedirs(debug_dir_for_global_error, exist_ok=True)
                
                # Fallback logic improved slightly
                if 'COMPONENTES_MAIN_TARGET_DIR_BASE' in globals() and os.path.isdir(COMPONENTES_MAIN_TARGET_DIR_BASE):
                    final_debug_dir = COMPONENTES_MAIN_TARGET_DIR_BASE
                elif 'TURMAS_MAIN_TARGET_DIR_BASE' in globals() and os.path.isdir(TURMAS_MAIN_TARGET_DIR_BASE):
                     final_debug_dir = TURMAS_MAIN_TARGET_DIR_BASE
                else:
                    final_debug_dir = debug_dir_for_global_error # Default to the created global_errors dir

                driver.save_screenshot(os.path.join(final_debug_dir, f"{error_filename_base}_screenshot.png"))
                if hasattr(driver, 'page_source'):
                    with open(os.path.join(final_debug_dir, f"{error_filename_base}_pagesource.html"), "w", encoding="utf-8") as f_err: f_err.write(driver.page_source)
                print(f"Saved global error debug info in '{final_debug_dir}'. Current URL: {driver.current_url if driver and hasattr(driver, 'current_url') else 'N/A'}")
            except Exception as e_save: print(f"Could not save debug info on global unexpected error: {e_save}")
        import traceback; traceback.print_exc()
    finally:
        print("\nScraping process finished.")
        if driver: print("Closing main browser."); driver.quit()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_controller()