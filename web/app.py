import streamlit as st
import requests
import json
import re
import time

# Page configuration
st.set_page_config(
    page_title="UnB Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #f0f2f6;
    }
    
    .assistant-message {
        background-color: #e8f4f8;
    }
    
    .loading-spinner {
        text-align: center;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Modal endpoint configuration
# MODAL_ENDPOINT_URL = "https://fejota12b--unb-chatbot-raft-gguf-web-endpoint-modele-b3f164-dev.modal.run"
# MODAL_ENDPOINT_URL = "https://doespacoluz--unb-chatbot-raft-gguf-web-endpoint-mode-292681-dev.modal.run"
# MODAL_ENDPOINT_URL = "https://cablite--unb-chatbot-raft-gguf-web-endpoint-modelend-e2846b-dev.modal.run" # 12b_run13
# MODAL_ENDPOINT_URL = "https://fariasfelipe--unb-chatbot-raft-gguf-web-endpoint-mod-0e084b-dev.modal.run" # 12b_neg_run1
# MODAL_ENDPOINT_URL = "https://lite12bneg--unb-chatbot-raft-gguf-web-endpoint-model-c88e98-dev.modal.run" # 12_4b_neg_run2
# MODAL_ENDPOINT_URL = "https://fefelilipe--unb-chatbot-raft-gguf-web-endpoint-model-f78d35-dev.modal.run" # 12b_realdis_run1
# MODAL_ENDPOINT_URL = "https://espacoluzdo--unb-chatbot-raft-gguf-web-endpoint-mode-5d90da-dev.modal.run" # 12b_extra_run1
# MODAL_ENDPOINT_URL = "https://cabelinhosonic--unb-chatbot-raft-gguf-web-endpoint-m-391656-dev.modal.run" # 12b_extra_run1
MODAL_ENDPOINT_URL = "https://multihop12b--unb-chatbot-raft-gguf-web-endpoint-mode-dbb199-dev.modal.run" # 12b_multihop_run2

def parse_response(response_text):
    """Parse the response to extract REASON and ANSWER sections"""
    try:
        # Extract REASON section
        reason_match = re.search(r'<REASON>(.*?)</REASON>', response_text, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else None
        
        # Extract ANSWER section - handle missing closing tag
        # First try to match with both opening and closing tags
        answer_match = re.search(r'<ANSWER>(.*?)</ANSWER>', response_text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            return answer, reason
        else:
            # If no closing tag, try to match from opening tag to end of response
            answer_match_no_close = re.search(r'<ANSWER>(.*)', response_text, re.DOTALL)
            if answer_match_no_close:
                answer = answer_match_no_close.group(1).strip()
                return answer, reason
            else:
                # If no ANSWER tags found at all, return the full response
                return response_text.strip(), reason
    except Exception as e:
        st.error(f"Error parsing response: {e}")
        return response_text, None

def call_modal_endpoint(messages, max_tokens=4096, temperature=0.3, top_p=0.95):
    """Call the Modal endpoint with the full chat history (`messages` array)."""
    try:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        response = requests.post(
            MODAL_ENDPOINT_URL,
            json=payload,
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            # Print raw JSON response
            print(f"RAW JSON RESPONSE: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result.get("response", "")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Connection error. Please check your internet connection.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🎓 UnB Chatbot")
st.markdown("*Assistente virtual da Universidade de Brasília*")
st.markdown("</div>", unsafe_allow_html=True)

# Chat container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message["content"] == "":
            # Show loading spinner for empty assistant messages
            with st.spinner("Processando sua pergunta..."):
                time.sleep(0.1)  # Small delay to show spinner
            st.markdown("*Processando...*")
        else:
            st.markdown(message["content"])
            
            # Show reasoning in a discrete expandable section if available
            if message["role"] == "assistant" and "reasoning" in message and message["reasoning"]:
                with st.expander("🔍 Ver raciocínio do modelo", expanded=False):
                    st.markdown(f"**Raciocínio:**\n\n{message['reasoning']}")

# Chat input
if prompt := st.chat_input("Faça sua pergunta sobre a UnB..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Add a placeholder for the assistant response
    st.session_state.messages.append({"role": "assistant", "content": "", "reasoning": None})
    
    # Rerun immediately to show the user message and loading state
    st.rerun()

# Process pending assistant responses
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and st.session_state.messages[-1]["content"] == "":
    # Build the payload messages list (exclude the blank assistant placeholder)
    payload_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["content"]  # skip blank placeholders
    ]

    # Process the response
    response = call_modal_endpoint(payload_messages)
    
    if response:
        # Parse response to extract ANSWER and REASON sections
        parsed_answer, reason = parse_response(response)
        # Update the last message (assistant placeholder) with the actual response
        st.session_state.messages[-1]["content"] = parsed_answer
        st.session_state.messages[-1]["reasoning"] = reason
    else:
        error_message = "Desculpe, ocorreu um erro ao processar sua pergunta. Tente novamente."
        st.session_state.messages[-1]["content"] = error_message
    
    # Rerun to display the complete conversation
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Sidebar with configuration (optional)
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    
    # Model parameters
    st.markdown("#### Parâmetros do Modelo")
    max_tokens = st.slider("Max Tokens", 256, 4096, 3096, 256)
    temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.95, 0.05)
    
    # Clear chat button
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()
    
    # Info section
    st.markdown("---")
    st.markdown("### ℹ️ Informações")
    st.markdown("""
    Este chatbot foi desenvolvido para responder perguntas sobre a UnB.
    
    **Como usar:**
    - Digite sua pergunta na caixa de texto
    - Aguarde a resposta do assistente
    - Continue a conversa normalmente
    
    **Exemplos de perguntas:**
    - "O que é o ENADE?"
    - "Como funciona o sistema de créditos?"
    - "Quais são os cursos disponíveis?"
    """)
    
    # URL configuration
    st.markdown("---")
    st.markdown("### 🔗 Endpoint")
    new_url = st.text_input("URL do Modal Endpoint:", value=MODAL_ENDPOINT_URL)
    if new_url != MODAL_ENDPOINT_URL:
        MODAL_ENDPOINT_URL = new_url

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "UnB Chatbot Protótipo"
    "</div>", 
    unsafe_allow_html=True
)