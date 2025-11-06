// ============================
//  STATE MANAGEMENT
// ============================
let chatSessions = [];
let currentSessionId = null;

// ============================
//  DOM ELEMENTS
// ============================
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const chatHistory = document.getElementById('chatHistory');
const newChatBtn = document.getElementById('newChatBtn');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebarCloseBtn = document.getElementById('sidebarCloseBtn');
const overlay = document.getElementById('overlay');

// ============================
//  INITIALIZATION
// ============================
function init() {
    const savedSessions = localStorage.getItem('chatSessions');
    if (savedSessions) {
        chatSessions = JSON.parse(savedSessions);
        renderChatHistory();
    }

    if (chatSessions.length === 0) {
        createNewChat();
    } else {
        loadSession(chatSessions[0].id);
    }

    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', handleKeyPress);
    messageInput.addEventListener('input', autoResize);
    newChatBtn.addEventListener('click', createNewChat);
    sidebarToggle.addEventListener('click', openSidebar);
    sidebarCloseBtn.addEventListener('click', closeSidebar);
    overlay.addEventListener('click', closeSidebar);

    // Escape key closes sidebar
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && sidebar.classList.contains('active')) {
            closeSidebar();
        }
    });
}

// ============================
//  TEXTAREA AUTO-RESIZE
// ============================
function autoResize() {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
}

// ============================
//  ENTER KEY HANDLER
// ============================
function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// ============================
//  CREATE NEW CHAT
// ============================
function createNewChat() {
    const newSession = {
        id: Date.now().toString(),
        title: 'New Chat',
        messages: [],
        timestamp: new Date().toISOString()
    };

    chatSessions.unshift(newSession);
    saveSessions();
    loadSession(newSession.id);
    renderChatHistory();
}

// ============================
//  LOAD EXISTING CHAT
// ============================
function loadSession(sessionId) {
    currentSessionId = sessionId;
    const session = chatSessions.find(s => s.id === sessionId);
    if (!session) return;

    chatMessages.innerHTML = '';

    if (session.messages.length === 0) {
        showWelcomeMessage();
    } else {
        session.messages.forEach(msg => {
            displayMessage(msg.text, msg.type, false);
        });
    }

    // Highlight active session
    document.querySelectorAll('.chat-history-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.sessionId === sessionId) {
            item.classList.add('active');
        }
    });

    scrollToBottom();
}

// ============================
//  WELCOME MESSAGE
// ============================
function showWelcomeMessage() {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="sparkle-icon">✨</div>
            <h1>Welcome to SympChat</h1>
            <p>How can I help you today?</p>
        </div>
    `;
}

// ============================
//  SEND MESSAGE (API CONNECTED)
// ============================
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    displayMessage(message, 'user');

    const session = chatSessions.find(s => s.id === currentSessionId);
    if (session) {
        session.messages.push({ text: message, type: 'user' });

        if (session.messages.length === 1) {
            session.title = message.substring(0, 50) + (message.length > 50 ? '...' : '');
            renderChatHistory();
        }

        saveSessions();
    }

    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Show typing placeholder
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot typing';
    typingDiv.innerHTML = '<div class="message-content">Analyzing your symptoms...</div>';
    chatMessages.appendChild(typingDiv);
    scrollToBottom();

    try {
        // === MAKE API REQUEST TO BACKEND ===
        const response = await fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symptoms: message })
        });

        const data = await response.json();

        typingDiv.remove();

        if (!data || !data.response) {
            displayMessage("⚠️ Sorry, I couldn't analyze your symptoms.", 'bot');
            return;
        }

        // Combine diagnoses + GPT summary
        let botHTML = "";
        if (data.suggested_diagnoses && data.suggested_diagnoses.length > 0) {
            botHTML += `
                <div class="diagnosis-list">
                    <h4>🩺 Possible Conditions:</h4>
                    <ul>${data.suggested_diagnoses.map(d => `<li>${d}</li>`).join('')}</ul>
                </div>
            `;
        }

        botHTML += `<div class="diagnosis-response">${data.response}</div>`;
        displayMessage(botHTML, 'bot');

        if (session) {
            session.messages.push({ text: botHTML, type: 'bot' });
            saveSessions();
        }

    } catch (error) {
        console.error('Error:', error);
        typingDiv.remove();
        displayMessage("⚠️ Error: Could not connect to the server.", 'bot');
    }
}

// ============================
//  DISPLAY MESSAGE
// ============================
function displayMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Check if the message includes HTML (for bot responses)
    if (type === 'bot' && text.includes('<')) {
        contentDiv.innerHTML = text;
    } else {
        contentDiv.textContent = text;
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// ============================
//  SIDEBAR CONTROLS
// ============================
function openSidebar() {
    sidebar.classList.add('active');
    overlay.classList.add('active');
}

function closeSidebar() {
    sidebar.classList.remove('active');
    overlay.classList.remove('active');
}

// ============================
//  CHAT HISTORY
// ============================
function renderChatHistory() {
    chatHistory.innerHTML = '';

    chatSessions.forEach(session => {
        const historyItem = document.createElement('div');
        historyItem.className = 'chat-history-item';
        historyItem.dataset.sessionId = session.id;

        const date = new Date(session.timestamp);
        const timeStr = date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric'
        });

        historyItem.innerHTML = `
            <h3>${session.title}</h3>
            <p>${timeStr}</p>
        `;

        historyItem.addEventListener('click', () => loadSession(session.id));
        chatHistory.appendChild(historyItem);
    });
}

// ============================
//  SCROLL HANDLER
// ============================
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ============================
//  SAVE SESSIONS
// ============================
function saveSessions() {
    localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
}

// ============================
//  INITIALIZE APP
// ============================
init();
