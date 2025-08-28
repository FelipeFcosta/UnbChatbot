// Chatbot UI logic

// Modal endpoints: keep one uncommented as the default
// MODAL_ENDPOINT_URL = "https://fejota12b--unb-chatbot-raft-gguf-web-endpoint-modele-b3f164-dev.modal.run"
// MODAL_ENDPOINT_URL = "https://doespacoluz--unb-chatbot-raft-gguf-web-endpoint-mode-292681-dev.modal.run"
// MODAL_ENDPOINT_URL = "https://cablite--unb-chatbot-raft-gguf-web-endpoint-modelend-e2846b-dev.modal.run" // 12b_run13
// MODAL_ENDPOINT_URL = "https://fariasfelipe--unb-chatbot-raft-gguf-web-endpoint-mod-0e084b-dev.modal.run" // 12b_neg_run1
// MODAL_ENDPOINT_URL = "https://lite12bneg--unb-chatbot-raft-gguf-web-endpoint-model-c88e98-dev.modal.run" // 12_4b_neg_run2
// MODAL_ENDPOINT_URL = "https://fefelilipe--unb-chatbot-raft-gguf-web-endpoint-model-f78d35-dev.modal.run" // 12b_realdis_run1, 12b_multihop_run4
// MODAL_ENDPOINT_URL = "https://espacoluzdo--unb-chatbot-raft-gguf-web-endpoint-mode-5d90da-dev.modal.run" // 12b_extra_run1
// MODAL_ENDPOINT_URL = "https://cabelinhosonic--unb-chatbot-raft-gguf-web-endpoint-m-391656-dev.modal.run" // 12b_extra_run1
// MODAL_ENDPOINT_URL = "https://multihop12b--unb-chatbot-raft-gguf-web-endpoint-mode-dbb199-dev.modal.run" // 12b_multihop_run2
// MODAL_ENDPOINT_URL = "https://vanis--unb-chatbot-raft-gguf-web-endpoint-modelendpo-6c8afc-dev.modal.run" // 12b_multihop_run3
// MODAL_ENDPOINT_URL = "https://felipecostasdc--unb-chatbot-raft-gguf-web-endpoint-m-443271-dev.modal.run"
MODAL_ENDPOINT_URL = "https://fefelilipe--unb-chatbot-raft-gguf-web-endpoint-model-f78d35-dev.modal.run" // 12b_multihop_run4

class UnBChatbot {
    constructor() {
        // Endpoint URLs
        this.primaryEndpoint = MODAL_ENDPOINT_URL;
        this.fallbackEndpoint = 'http://localhost:8001/';
        
        // Configuration
        this.config = {
            maxTokens: 3096,
            temperature: 1.0,
            topP: 0.95
        };
        
        // Chat state
        this.messages = [];
        this.isProcessing = false;
        this.lastRetrievedChunks = null; // Stores chunks only for the last assistant message
        this.scrollTimeout = null; // To prevent duplicate delayed scrolling
        
        // DOM elements
        this.elements = {
            chatMessages: document.getElementById('chatMessages'),
            chatContainer: document.querySelector('.chat-container'),
            userInput: document.getElementById('userInput'),
            sendButton: document.getElementById('sendButton'),
            loadingIndicator: document.getElementById('loadingIndicator'),
            clearChat: document.getElementById('clearChat'),
            sidebar: document.getElementById('sidebar'),
            sidebarToggle: document.getElementById('sidebarToggle'),
            endpointUrl: document.getElementById('endpointUrl'),
            maxTokens: document.getElementById('maxTokens'),
            maxTokensValue: document.getElementById('maxTokensValue'),
            temperature: document.getElementById('temperature'),
            temperatureValue: document.getElementById('temperatureValue'),
            topP: document.getElementById('topP'),
            topPValue: document.getElementById('topPValue')
        };
        
        this.initializeEventListeners();
        this.loadSavedState();
    }

    renderMarkdownToElement(targetElement, markdownText) {
        try {
            if (!markdownText) {
                targetElement.textContent = '';
                return;
            }

            // Configure marked once per runtime
            if (!this._markedConfigured) {
                if (window.marked) {
                    marked.setOptions({
                        gfm: true,
                        breaks: true
                    });
                }
                this._markedConfigured = true;
            }

            const rawHtml = window.marked ? marked.parse(markdownText) : markdownText;
            const safeHtml = window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;

            targetElement.classList.add('markdown-body');
            targetElement.innerHTML = safeHtml;

            // Transform italic markdown links (and quoted italic links) into a reference icon link
            try { this.applyReferenceIconTransform(targetElement); } catch (_) {}

            // Highlight code blocks - use setTimeout to ensure it's processed after DOM update
            if (window.hljs) {
                setTimeout(() => {
                    targetElement.querySelectorAll('pre code').forEach((block) => {
                        try { hljs.highlightElement(block); } catch (_) {}
                    });
                }, 0);
            }
        } catch (error) {
            console.error('Markdown render error:', error);
            targetElement.textContent = markdownText;
        }
    }

    /**
     * Find patterns of italic links like *[title](url)* and > *[title](url)* rendered by markdown
     * and replace them with a discrete reference icon that links to the URL.
     * Tooltip shows "title — URL". Any leading blockquote styling for the pattern is removed.
     */
    applyReferenceIconTransform(container) {
        if (!container) return;

            const createPillAnchor = (href, titleText) => {
            const a = document.createElement('a');
            a.className = 'reference-pill-link';
            a.href = href || '#';
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            const safeTitle = (titleText || '').toString().trim();
            const computeDomain = (rawHref) => {
                try {
                    if (!rawHref) return '';
                    const urlObj = new URL(rawHref, window.location.href);
                    return (urlObj.hostname || '').replace(/^www\./i, '');
                } catch (_) {
                    try {
                        const m = String(rawHref).match(/^(?:https?:\/\/)?([^\/#?]+)/i);
                        return (m && m[1] ? m[1] : String(rawHref)).replace(/^www\./i, '');
                    } catch (_) { return ''; }
                }
            };
            const domain = computeDomain(href);
            const tooltip = safeTitle && href ? `${safeTitle} — ${href}` : (href || safeTitle || 'Referência');
            a.title = tooltip;
            a.setAttribute('aria-label', tooltip);
            a.textContent = domain || 'link';
            return a;
        };

        // 1) Replace blockquotes that contain only an italic link
        container.querySelectorAll('blockquote').forEach((bq) => {
            try {
                const linkInside = bq.querySelector('em > a, i > a');
                if (!linkInside) return;
                // Check whether blockquote is only that link (with optional wrapping <p>)
                const textOnly = (bq.textContent || '').trim();
                const linkText = (linkInside.textContent || '').trim();
                if (textOnly === linkText) {
                    const pill = createPillAnchor(linkInside.getAttribute('href'), linkText);
                    bq.replaceWith(pill);
                }
            } catch (_) {}
        });

        // 2) Replace any remaining italic links that are exactly *[title](url)* (em strictly wraps the link)
        // Note: re-select after blockquote replacements
        const candidates = Array.from(container.querySelectorAll('em > a, i > a'));
        candidates.forEach((aEl) => {
            try {
                const em = aEl.parentElement;
                if (!em) return;
                const onlyChildIsLink = em.children.length === 1 && em.firstElementChild === aEl;
                const textMatches = (em.textContent || '').trim() === (aEl.textContent || '').trim();
                if (!(onlyChildIsLink && textMatches)) return; // Avoid touching other italic content
                const pill = createPillAnchor(aEl.getAttribute('href'), aEl.textContent);
                em.replaceWith(pill);
            } catch (_) {}
        });
    }
    
    initializeEventListeners() {
        // Send message
        this.elements.sendButton.addEventListener('click', () => this.sendMessage());
        this.elements.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Clear chat
        this.elements.clearChat.addEventListener('click', () => this.clearChat());
        
        // Sidebar toggle
        this.elements.sidebarToggle.addEventListener('click', () => this.toggleSidebar());

        
        // Sliders update config and persist
        this.elements.maxTokens.addEventListener('input', (e) => {
            this.config.maxTokens = parseInt(e.target.value);
            this.elements.maxTokensValue.textContent = e.target.value;
            this.saveState();
        });
        
        this.elements.temperature.addEventListener('input', (e) => {
            this.config.temperature = parseFloat(e.target.value);
            this.elements.temperatureValue.textContent = e.target.value;
            this.saveState();
        });
        
        this.elements.topP.addEventListener('input', (e) => {
            this.config.topP = parseFloat(e.target.value);
            this.elements.topPValue.textContent = e.target.value;
            this.saveState();
        });
        
        // Custom endpoint URL
        this.elements.endpointUrl.addEventListener('change', (e) => {
            this.primaryEndpoint = e.target.value;
            this.saveState();
        });
    }
    
    toggleSidebar() {
        this.elements.sidebar.classList.toggle('collapsed');
        const isCollapsed = this.elements.sidebar.classList.contains('collapsed');
        localStorage.setItem('sidebarCollapsed', isCollapsed);
        this.updateSidebarToggleIcon();
        try {
            this.elements.sidebarToggle.setAttribute('aria-expanded', String(!isCollapsed));
        } catch (_) {}
    }
    
    clearChat() {
        this.messages = [];
        this.elements.chatMessages.innerHTML = '';
        this.saveState();
    }
    
    async sendMessage() {
        const message = this.elements.userInput.value.trim();
        if (!message || this.isProcessing) return;
        
        // Add user message
        this.addMessage('user', message);
        this.elements.userInput.value = '';
        
        // Processing state
        this.setProcessingState(true);
        
        // Map messages for API
        const apiMessages = this.messages.map(msg => ({
            role: msg.role,
            reasoning: msg.reasoning ?? null,
            content: msg.content,
        }));
        
        try {
            // Call the API
            const apiResult = await this.callEndpoint(apiMessages);
            
            if (apiResult) {
                // Extract response text and retrieved chunks
                let responseText = apiResult;
                this.lastRetrievedChunks = null;
                
                if (typeof apiResult === 'object' && apiResult !== null && 'response' in apiResult) {
                    responseText = apiResult.response;
                    if (Array.isArray(apiResult.retrieved_chunks) && apiResult.retrieved_chunks.length > 0) {
                        this.lastRetrievedChunks = apiResult.retrieved_chunks;
                        console.log('Stored retrieved chunks for last message:', this.lastRetrievedChunks.length, 'chunks');
                    }
                }
                
                const { answer, reason } = this.parseResponse(responseText);
                this.addMessage('assistant', answer, reason);
            } else {
                this.addMessage('assistant', 'Desculpe, ocorreu um erro ao processar sua pergunta. Tente novamente.');
            }
        } catch (error) {
            console.error('Error:', error);
            this.addMessage('assistant', 'Desculpe, ocorreu um erro ao processar sua pergunta. Tente novamente.');
        } finally {
            this.setProcessingState(false);
        }
    }
    
    async callEndpoint(messages) {
        const payload = {
            messages: messages,
            max_tokens: this.config.maxTokens,
            temperature: this.config.temperature,
            top_p: this.config.topP
        };
        
        // Try primary endpoint first
        try {
            const response = await this.makeRequest(this.primaryEndpoint, payload);
            if (response.ok) {
                const data = await response.json();
                console.log('Server JSON response (primary):', data);
                return data;
            }
        } catch (error) {
            console.log('Primary endpoint failed, trying fallback...', error);
        }
        
        // Try fallback endpoint
        try {
            const response = await this.makeRequest(this.fallbackEndpoint, payload);
            if (response.ok) {
                const data = await response.json();
                console.log('Server JSON response (fallback):', data);
                return data;
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Both endpoints failed:', error);
            return null;
        }
    }
    
    async makeRequest(url, payload) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minutes
        // Browsers may send an OPTIONS preflight for cross-origin JSON POST; server handles it.

        try {
            // Log exactly what will be sent to the server
            const bodyJson = JSON.stringify(payload);
            console.log('Sending POST to', url, 'with JSON payload:', bodyJson);

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: bodyJson,
                signal: controller.signal,
                mode: 'cors',
                credentials: 'omit'
            });

            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Request timed out');
            }
            throw error;
        }
    }
    
    parseResponse(responseText) {
        try {
            // Extract REASON section
            const reasonMatch = responseText.match(/<REASON>([\s\S]*?)<\/REASON>/);
            const reason = reasonMatch ? reasonMatch[1].trim() : null;
            
            // Extract ANSWER section
            let answer = responseText;
            
            // Handle both <ANSWER>...</ANSWER> and <ANSWER> without closing tag
            const answerMatch = responseText.match(/<ANSWER>([\s\S]*?)<\/ANSWER>/);
            if (answerMatch) {
                answer = answerMatch[1].trim();
            } else {
                // Try to match from opening tag to end
                const answerMatchNoClose = responseText.match(/<ANSWER>([\s\S]*)/);
                if (answerMatchNoClose) {
                    answer = answerMatchNoClose[1].trim();
                }
            }
            
            return { answer, reason };
        } catch (error) {
            console.error('Error parsing response:', error);
            return { answer: responseText, reason: null };
        }
    }
    
    addMessage(role, content, reasoning = null) {
        const message = { role, content, reasoning };
        this.messages.push(message);
        
        // Clear previous chunks UI from all assistant messages
        if (role === 'assistant') {
            this.clearChunksUI();
        }
        
        const messageElement = this.createMessageElement(message);
        this.elements.chatMessages.appendChild(messageElement);
        
        // Add chunks UI only to the latest assistant message if chunks are available
        if (role === 'assistant' && this.lastRetrievedChunks && this.lastRetrievedChunks.length > 0) {
            this.addChunksUI(messageElement);
        }
        
        // Delay scroll to ensure DOM is updated and markdown is rendered
        this.scrollToBottomDelayed();
        
        // Save state
        this.saveState();
    }
    
    createMessageElement(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role}`;
        
        // Avatar (assistant only)
        let avatar = null;
        if (message.role === 'assistant') {
            avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
        }
        
        // Content wrapper
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content';
        
        // Message text
        const messageText = document.createElement('div');
        if (message.role === 'assistant') {
            this.renderMarkdownToElement(messageText, message.content);
        } else {
            messageText.textContent = message.content;
        }
        contentWrapper.appendChild(messageText);
        
        // Add reasoning if available (for assistant messages)
        if (message.role === 'assistant' && message.reasoning) {
            const reasoningContainer = document.createElement('div');
            reasoningContainer.className = 'reasoning-container';
            
            const reasoningToggle = document.createElement('button');
            reasoningToggle.className = 'reasoning-toggle';
            reasoningToggle.innerHTML = '<i class="fas fa-chevron-right"></i> 🔍 Ver raciocínio do modelo';
            
            const reasoningContent = document.createElement('div');
            reasoningContent.className = 'reasoning-content';
            this.renderMarkdownToElement(
                reasoningContent,
                `**Raciocínio:**\n\n${message.reasoning}`
            );
            
            reasoningToggle.addEventListener('click', () => {
                reasoningToggle.classList.toggle('expanded');
                reasoningContent.classList.toggle('show');
            });
            
            reasoningContainer.appendChild(reasoningToggle);
            reasoningContainer.appendChild(reasoningContent);
            contentWrapper.appendChild(reasoningContainer);
        }
        
        if (avatar) {
            messageDiv.appendChild(avatar);
        }
        messageDiv.appendChild(contentWrapper);
        
        return messageDiv;
    }
    
    clearChunksUI() {
        // Remove any existing chunks UI elements
        try {
            this.elements.chatMessages.querySelectorAll('.chunks-indicator').forEach(el => el.remove());
        } catch (error) {
            console.error('Error clearing chunks UI:', error);
        }
    }
    
    addChunksUI(messageElement) {
        try {
            const contentWrapper = messageElement.querySelector('.message-content');
            if (!contentWrapper) return;
            
            // Chunks indicator (visible on hover)
            const chunksIndicator = document.createElement('div');
            chunksIndicator.className = 'chunks-indicator';
            chunksIndicator.innerHTML = `
                <div class="chunks-trigger" title="Documentos consultados">
                    <i class="fas fa-book-open"></i>
                </div>
                <div class="chunks-popup">
                    <div class="chunks-popup-header">
                        <i class="fas fa-book-open"></i>
                        Documentos consultados
                    </div>
                    <div class="chunks-popup-content"></div>
                </div>
            `;
            
            // Adjust popup direction based on viewport position
            // If the trigger is in the upper half of the viewport, open downward; otherwise, open upward
            const updatePopupDirection = () => {
                try {
                    const rect = chunksIndicator.getBoundingClientRect();
                    const viewportMidY = window.innerHeight / 2;
                    if (rect.top < viewportMidY) {
                        chunksIndicator.classList.add('open-down');
                    } else {
                        chunksIndicator.classList.remove('open-down');
                    }
                } catch (_) {}
            };
            updatePopupDirection();
            window.addEventListener('resize', updatePopupDirection);
            window.addEventListener('scroll', updatePopupDirection, true);

            // Populate popup content
            const popupContent = chunksIndicator.querySelector('.chunks-popup-content');
            this.lastRetrievedChunks.forEach((chunk, index) => {
                const chunkItem = document.createElement('details');
                chunkItem.className = 'chunk-item';
                
                const summary = document.createElement('summary');
                summary.className = 'chunk-summary';
                
                // Extract title, text content, and URL from metadata
                let title = `Documento ${index + 1}`;
                let textContent = chunk;
                let url = '';
                
                try {
                    const metaMatch = chunk.match(/<doc_metadata>([\s\S]*?)<\/doc_metadata>/);
                    if (metaMatch) {
                        const meta = metaMatch[1];
                        
                        // Extract filename for title (without extension)
                        const fileMatch = meta.match(/File:\s*"([^"]+)"/);
                        if (fileMatch) {
                            const fullName = fileMatch[1];
                            title = fullName.replace(/\.[^\.]+$/, '');
                        }
                        
                        // Extract URL field text
                        const urlMatch = meta.match(/URL:\s*(.+?)(?:,|\s*$)/);
                        if (urlMatch) {
                            url = urlMatch[1].trim().replace(/^"|"$/g, ''); // Remove quotes if present
                        }
                        
                        // Extract text content (after metadata)
                        const textMatch = chunk.match(/<\/doc_metadata>\s*([\s\S]*)/);
                        if (textMatch) {
                            textContent = textMatch[1].trim();
                            // Remove surrounding quotes if present
                            if (textContent.startsWith('"') && textContent.endsWith('"')) {
                                textContent = textContent.slice(1, -1);
                            }
                        }
                    }
                } catch (error) {
                    console.error('Error parsing chunk metadata:', error);
                }
                
                summary.textContent = title;
                chunkItem.appendChild(summary);
                
                const chunkContent = document.createElement('div');
                chunkContent.className = 'chunk-content';
                
                // Text content with markdown rendering
                const textElement = document.createElement('div');
                textElement.className = 'chunk-text';
                this.renderMarkdownToElement(textElement, textContent);
                chunkContent.appendChild(textElement);
                
                // Append URL if available
                if (url) {
                    const urlElement = document.createElement('div');
                    urlElement.className = 'chunk-url';
                    this.renderMarkdownToElement(urlElement, url);
                    chunkContent.appendChild(urlElement);
                }
                
                chunkItem.appendChild(chunkContent);
                popupContent.appendChild(chunkItem);
            });
            
            // Append to message
            contentWrapper.appendChild(chunksIndicator);
            
            console.log('Added chunks UI with', this.lastRetrievedChunks.length, 'chunks');
        } catch (error) {
            console.error('Error adding chunks UI:', error);
        }
    }
    
    setProcessingState(isProcessing) {
        this.isProcessing = isProcessing;
        this.elements.sendButton.disabled = isProcessing;
        this.elements.userInput.disabled = isProcessing;
        this.elements.loadingIndicator.style.display = isProcessing ? 'flex' : 'none';
        
        // Ensure the loading indicator is always right after the latest message
        if (isProcessing) {
            try {
                this.elements.chatMessages.appendChild(this.elements.loadingIndicator);
            } catch (_) {}
            // Ensure the indicator is visible with proper timing
            this.scrollToBottomDelayed();
        }
    }
    
    scrollToBottom() {
        const container = this.elements.chatContainer;
        if (!container) return;
        // Simple and fast: jump directly to the bottom
        container.scrollTop = container.scrollHeight;
    }
    
    scrollToBottomDelayed() {
        // Clear any existing timeout
        if (this.scrollTimeout) {
            clearTimeout(this.scrollTimeout);
        }
        
        // Use requestAnimationFrame to ensure DOM is updated before scrolling
        requestAnimationFrame(() => {
            // Double requestAnimationFrame to handle async markdown rendering
            requestAnimationFrame(() => {
                this.performScrollToBottom();
                this.scrollTimeout = null;
            });
        });
        
        // Fallback timeout in case requestAnimationFrame fails
        this.scrollTimeout = setTimeout(() => {
            this.performScrollToBottom();
            this.scrollTimeout = null;
        }, 100);
    }
    
    performScrollToBottom() {
        const container = this.elements.chatContainer;
        if (!container) return;
        
        // Calculate proper scroll position with bounds checking
        const maxScrollTop = container.scrollHeight - container.clientHeight;
        const targetScrollTop = Math.max(0, maxScrollTop);
        
        // Only scroll if we're not already at the bottom or close to it
        const currentScroll = container.scrollTop;
        const threshold = 50; // pixels
        
        if (Math.abs(currentScroll - targetScrollTop) > threshold) {
            container.scrollTo({
                top: targetScrollTop,
                behavior: 'smooth'
            });
        }
    }

    isNearBottom() {
        const container = this.elements.chatContainer;
        const threshold = 24; // pixels
        return container.scrollHeight - container.scrollTop - container.clientHeight <= threshold;
    }
    
    saveState() {
        const state = {
            messages: this.messages,
            config: this.config,
            primaryEndpoint: this.primaryEndpoint,
            lastRetrievedChunks: this.lastRetrievedChunks, // Save chunks only for the last message
            scrollTop: this.elements.chatContainer.scrollTop,
            isAtBottom: Math.abs(
                this.elements.chatContainer.scrollHeight -
                this.elements.chatContainer.scrollTop -
                this.elements.chatContainer.clientHeight
            ) < 2
        };
        localStorage.setItem('chatbotState', JSON.stringify(state));
    }
    
    loadSavedState() {
        try {
            const savedState = localStorage.getItem('chatbotState');
            if (savedState) {
                const state = JSON.parse(savedState);
                
                // Restore messages
                if (state.messages && Array.isArray(state.messages)) {
                    this.messages = state.messages;
                    this.messages.forEach(msg => {
                        const messageElement = this.createMessageElement(msg);
                        this.elements.chatMessages.appendChild(messageElement);
                    });
                // After restoring messages, scroll to the bottom with proper timing
                this.scrollToBottomDelayed();
                }
                
                // Restore last retrieved chunks
                if (state.lastRetrievedChunks && Array.isArray(state.lastRetrievedChunks)) {
                    this.lastRetrievedChunks = state.lastRetrievedChunks;
                    // Find the last assistant message and add chunks UI
                    const assistantMessages = this.elements.chatMessages.querySelectorAll('.message.assistant');
                    if (assistantMessages.length > 0) {
                        const lastAssistantMessage = assistantMessages[assistantMessages.length - 1];
                        this.addChunksUI(lastAssistantMessage);
                    }
                }
                
                // Restore config
                if (state.config) {
                    this.config = { ...this.config, ...state.config };
                    this.elements.maxTokens.value = this.config.maxTokens;
                    this.elements.maxTokensValue.textContent = this.config.maxTokens;
                    this.elements.temperature.value = this.config.temperature;
                    this.elements.temperatureValue.textContent = this.config.temperature;
                    this.elements.topP.value = this.config.topP;
                    this.elements.topPValue.textContent = this.config.topP;
                }
                
                // Restore endpoint
                if (state.primaryEndpoint) {
                    this.primaryEndpoint = state.primaryEndpoint;
                    this.elements.endpointUrl.value = this.primaryEndpoint;
                }
            }
            
            // Restore sidebar state
            const sidebarCollapsed = localStorage.getItem('sidebarCollapsed');
            if (sidebarCollapsed === 'true') {
                this.elements.sidebar.classList.add('collapsed');
            }
        } catch (error) {
            console.error('Error loading saved state:', error);
        }

        // Ensure the input shows the current endpoint if none saved
        if (!this.elements.endpointUrl.value) {
            this.elements.endpointUrl.value = this.primaryEndpoint;
        }

        // Sync the sidebar toggle icon with current state
        this.updateSidebarToggleIcon();
    }

    updateSidebarToggleIcon() {
        const toggleButton = this.elements.sidebarToggle;
        const iconElement = toggleButton ? toggleButton.querySelector('i') : null;
        const isCollapsed = this.elements.sidebar.classList.contains('collapsed');
        if (!iconElement) return;

        // Left arrow when open (collapse action), right arrow when closed (expand action)
        iconElement.className = isCollapsed ? 'fas fa-chevron-right' : 'fas fa-chevron-left';
        toggleButton.setAttribute('title', isCollapsed ? 'Abrir painel' : 'Fechar painel');
        toggleButton.setAttribute('aria-label', isCollapsed ? 'Abrir painel' : 'Fechar painel');
        toggleButton.setAttribute('aria-expanded', String(!isCollapsed));
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new UnBChatbot();
    
    // Add welcome message if none exists
    if (window.chatbot.messages.length === 0) {
        const welcomeMessage = 'Olá! Sou o assistente virtual da UnB. Como posso ajudá-lo hoje? ' +
                              'Você pode me fazer perguntas sobre cursos, processos acadêmicos, ' +
                              'ou qualquer informação relacionada à universidade.';
        // window.chatbot.addMessage('assistant', welcomeMessage);
    }
});
