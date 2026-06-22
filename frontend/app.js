let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const recordBtn = document.getElementById('record-btn');
const statusDisplay = document.getElementById('recording-status');
const chatContainer = document.getElementById('chat-container');
const languageSelect = document.getElementById('language-select');
const suggestionsText = document.getElementById('suggestions-text');

let currentAudio = null;

// Initialize microphone access
async function setupAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioChunks = [];
            
            if (audioBlob.size > 0) {
                await sendAudio(audioBlob);
            }
        };
    } catch (err) {
        console.error("Error accessing microphone:", err);
        statusDisplay.textContent = "Microphone access denied.";
        statusDisplay.style.color = "var(--danger-color)";
    }
}

function startRecording() {
    if (!mediaRecorder) return;
    
    // Stop any currently playing audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
    
    audioChunks = [];
    isRecording = true;
    mediaRecorder.start();
    
    recordBtn.classList.add('recording');
    recordBtn.innerHTML = '<span class="icon">🛑</span> Recording...';
    statusDisplay.textContent = "Listening...";
    statusDisplay.style.color = "var(--danger-color)";
}

function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    
    isRecording = false;
    mediaRecorder.stop();
    
    recordBtn.classList.remove('recording');
    recordBtn.innerHTML = '<span class="icon">🎙️</span> Hold to Talk';
    statusDisplay.textContent = "Processing...";
    statusDisplay.style.color = "var(--text-secondary)";
}

async function sendAudio(blob) {
    const formData = new FormData();
    formData.append('audio', blob, 'recording.wav');
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Server error');
        
        const data = await response.json();
        
        // Clear status
        statusDisplay.textContent = "";
        
        // Add User Message
        appendMessage('user', data.user_text);
        
        // Add AI Message
        appendMessage('ai', data.ai_text);
        
        // Update Suggestions
        suggestionsText.classList.remove('placeholder');
        suggestionsText.textContent = data.suggestions;
        
        // Play AI Audio
        if (data.audio_url) {
            // Add a timestamp to bypass caching
            currentAudio = new Audio(data.audio_url + "?t=" + new Date().getTime());
            currentAudio.play();
        }
        
    } catch (err) {
        console.error("Error sending audio:", err);
        statusDisplay.textContent = "Error processing audio.";
        statusDisplay.style.color = "var(--danger-color)";
    }
}

function appendMessage(sender, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = sender === 'user' ? '👤' : '🤖';
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Handle Language Change
languageSelect.addEventListener('change', async (e) => {
    const newLang = e.target.value;
    const formData = new FormData();
    formData.append('lang', newLang);
    
    try {
        statusDisplay.textContent = `Switching to ${newLang}...`;
        await fetch('/api/models', {
            method: 'POST',
            body: formData
        });
        statusDisplay.textContent = `Language switched to ${newLang}.`;
        setTimeout(() => { statusDisplay.textContent = ''; }, 2000);
    } catch (err) {
        console.error("Error switching language:", err);
    }
});

// Event Listeners for Record Button
recordBtn.addEventListener('mousedown', startRecording);
recordBtn.addEventListener('mouseup', stopRecording);
recordBtn.addEventListener('mouseleave', stopRecording);

// Touch support
recordBtn.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startRecording();
});
recordBtn.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopRecording();
});

// Init
setupAudio();
