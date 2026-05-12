function toggleChat() {
    const chatWindow = document.getElementById('chatbot-window');
    const fab = document.getElementById('chatbot-fab');
    
    chatWindow.classList.toggle('active');
    
    if (chatWindow.classList.contains('active')) {
        fab.innerHTML = '<i class="fa-solid fa-times"></i>';
        const input = document.getElementById('chatbot-input');
        input.focus();
        
        // Ensure scroll to bottom on open
        const container = document.getElementById('chatbot-messages');
        container.scrollTop = container.scrollHeight;
    } else {
        fab.innerHTML = '<i class="fa-solid fa-robot"></i>';
        // Auto-disable full screen when closing chat
        chatWindow.classList.remove('full-screen');
    }
}

function toggleFullScreen() {
    const chatWindow = document.getElementById('chatbot-window');
    const icon = document.getElementById('expand-icon');
    
    chatWindow.classList.toggle('full-screen');
    
    if (chatWindow.classList.contains('full-screen')) {
        icon.classList.remove('fa-expand');
        icon.classList.add('fa-compress');
    } else {
        icon.classList.remove('fa-compress');
        icon.classList.add('fa-expand');
    }
    
    // Maintain scroll position after resize
    const container = document.getElementById('chatbot-messages');
    container.scrollTop = container.scrollHeight;
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

async function sendMessage(customMessage = null) {
    const input = document.getElementById('chatbot-input');
    const message = customMessage || input.value.trim();
    if (!message) return;

    appendMessage('user', message);
    if (!customMessage) input.value = '';
    
    // Add loading indicator
    const loadingId = 'loading-' + Date.now();
    appendMessage('bot', '<span class="typing-indicator">Thinking...</span>', loadingId);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        removeElement(loadingId);
        appendMessage('bot', data.response);
    } catch (error) {
        removeElement(loadingId);
        appendMessage('bot', 'Sorry, I am having trouble connecting to the server. 😕');
    }
}

function askAboutMovie(title) {
    const chatWindow = document.getElementById('chatbot-window');
    if (!chatWindow.classList.contains('active')) {
        toggleChat();
    }
    sendMessage(`Tell me curious facts about the movie: ${title}. I want to know about the cast, box office, and why I should watch it!`);
}

function appendMessage(sender, text, id = null) {
    const container = document.getElementById('chatbot-messages');
    const div = document.createElement('div');
    div.className = sender === 'user' ? 'user-msg-container' : 'bot-msg-container';
    if (id) div.id = id;
    
    // Simple markdown-to-html (for bold and newlines)
    let formattedText = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');

    div.innerHTML = `
        <div class="${sender}-msg">
            ${formattedText}
        </div>
    `;
    
    container.appendChild(div);
    // Smooth scroll to bottom
    container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
    });
}

function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}
