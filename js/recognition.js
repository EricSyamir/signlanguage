/**
 * SignBridge - Sign Language Interpreter
 * Based on: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning
 * 
 * Real-time sign language interpretation with sentence building
 */

// Detect backend URL dynamically
function getBackendURL() {
    // Check for explicit backend URL in config
    if (window.SignBridgeConfig && window.SignBridgeConfig.BACKEND_URL) {
        return window.SignBridgeConfig.BACKEND_URL;
    }
    
    if (window.BACKEND_URL) {
        return window.BACKEND_URL;
    }
    
    // Local development
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:8000';
    }
    
    // Render deployment - replace frontend with backend in hostname
    const hostname = window.location.hostname;
    if (hostname.includes('signbridge-frontend') || hostname.includes('onrender.com')) {
        return 'https://' + hostname.replace('signbridge-frontend', 'signbridge-backend');
    }
    
    // Default: same origin (if backend and frontend are on same domain)
    return window.location.origin.replace(':8080', ':8000').replace(':3000', ':8000');
}

const BACKEND_URL = getBackendURL();
const FRAME_INTERVAL = (window.SignBridgeConfig && window.SignBridgeConfig.FRAME_INTERVAL) || 200;

console.log('SignBridge initialized');
console.log('Backend URL:', BACKEND_URL);

let stream = null;
let recognitionInterval = null;
let isRunning = false;
let recentPredictions = [];
let websocket = null;

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    checkBackendStatus();
    updateUI();
});

// Check backend connection
async function checkBackendStatus() {
    const statusEl = document.getElementById('backendStatus');
    
    try {
        const response = await fetch(`${BACKEND_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            statusEl.classList.remove('disconnected');
            statusEl.classList.add('connected');
            statusEl.innerHTML = `
                <span class="status-dot"></span>
                <span>Backend: Connected ${data.model_loaded ? '(Model ✓)' : '(No Model)'}</span>
            `;
            console.log('Backend status:', data);
        }
    } catch (error) {
        statusEl.classList.remove('connected');
        statusEl.classList.add('disconnected');
        statusEl.innerHTML = `
            <span class="status-dot"></span>
            <span>Backend: Not Connected</span>
        `;
        console.log('Backend not available:', error.message);
    }
}

// ============================================================
// Camera Functions
// ============================================================

async function startCamera() {
    const cameraStatus = document.getElementById('cameraStatus');
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });
        
        const video = document.getElementById('camera');
        video.srcObject = stream;
        
        cameraStatus.classList.remove('disconnected');
        cameraStatus.classList.add('connected');
        cameraStatus.innerHTML = `
            <span class="status-dot"></span>
            <span>Camera: Active</span>
        `;
        
        return true;
    } catch (error) {
        console.error('Camera error:', error);
        cameraStatus.innerHTML = `
            <span class="status-dot"></span>
            <span>Camera: Error - ${error.message}</span>
        `;
        return false;
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        
        const video = document.getElementById('camera');
        video.srcObject = null;
        
        const cameraStatus = document.getElementById('cameraStatus');
        cameraStatus.classList.remove('connected');
        cameraStatus.classList.add('disconnected');
        cameraStatus.innerHTML = `
            <span class="status-dot"></span>
            <span>Camera: Off</span>
        `;
    }
}

// ============================================================
// Recognition Functions
// ============================================================

async function startInterpreter() {
    const cameraStarted = await startCamera();
    if (!cameraStarted) {
        alert('Could not start camera. Please check permissions.');
        return;
    }
    
    isRunning = true;
    updateUI();
    
    // Show prediction overlay
    document.getElementById('predictionOverlay').style.display = 'block';
    
    // Start sending frames
    recognitionInterval = setInterval(sendFrame, FRAME_INTERVAL);
    
    // Try WebSocket connection for real-time
    // tryWebSocket();
}

function stopInterpreter() {
    isRunning = false;
    
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
    }
    
    if (websocket) {
        websocket.close();
        websocket = null;
    }
    
    stopCamera();
    document.getElementById('predictionOverlay').style.display = 'none';
    updateUI();
}

async function sendFrame() {
    if (!isRunning) return;
    
    const video = document.getElementById('camera');
    if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) return;
    
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(async (blob) => {
        if (!blob) return;
        
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        formData.append('language', 'ASL');
        
        try {
            const response = await fetch(`${BACKEND_URL}/predict-and-build`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                displayPrediction(data);
                updateSentenceDisplay(data);
            }
        } catch (error) {
            console.error('Recognition error:', error);
            // Don't stop on error, just log it
        }
    }, 'image/jpeg', 0.8);
}

function tryWebSocket() {
    try {
        // Use WebSocket URL based on backend URL
        const wsUrl = BACKEND_URL.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/recognize';
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = () => {
            console.log('WebSocket connected');
        };
        
        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            displayPrediction(data);
            updateSentenceDisplay(data);
        };
        
        websocket.onerror = (error) => {
            console.log('WebSocket error, falling back to HTTP:', error);
        };
        
        websocket.onclose = () => {
            console.log('WebSocket closed');
        };
    } catch (error) {
        console.log('WebSocket not available');
    }
}

// ============================================================
// Display Functions
// ============================================================

function displayPrediction(data) {
    const letterEl = document.getElementById('predictionLetter');
    const fillEl = document.getElementById('confidenceFill');
    const textEl = document.getElementById('confidenceText');
    
    letterEl.textContent = data.label || '-';
    
    const confidence = Math.round((data.confidence || 0) * 100);
    fillEl.style.width = `${confidence}%`;
    textEl.textContent = `${confidence}%`;
    
    // Color based on confidence
    if (confidence >= 80) {
        fillEl.style.background = '#10b981'; // Green
    } else if (confidence >= 60) {
        fillEl.style.background = '#f59e0b'; // Yellow
    } else {
        fillEl.style.background = '#ef4444'; // Red
    }
    
    // Update recent predictions
    addRecentPrediction(data.label, confidence);
}

function updateSentenceDisplay(data) {
    const completedEl = document.getElementById('completedWords');
    const currentEl = document.getElementById('currentWord');
    
    // Get sentence parts
    const sentence = data.sentence || '';
    const currentWord = data.current_word || '';
    
    // Display completed sentence
    if (currentWord) {
        completedEl.textContent = sentence.replace(currentWord, '').trim() + ' ';
        currentEl.textContent = currentWord;
    } else {
        completedEl.textContent = sentence;
        currentEl.textContent = '';
    }
}

function addRecentPrediction(label, confidence) {
    recentPredictions.unshift({ label, confidence, time: Date.now() });
    
    // Keep only last 10
    if (recentPredictions.length > 10) {
        recentPredictions.pop();
    }
    
    // Update display
    const container = document.getElementById('recentPredictions');
    container.innerHTML = recentPredictions.map((p, i) => `
        <div class="gesture-hint ${i === 0 ? 'active' : ''}" 
             style="opacity: ${1 - (i * 0.08)}">
            ${p.label}
        </div>
    `).join('');
}

// ============================================================
// Sentence Controls
// ============================================================

async function addSpace() {
    try {
        const response = await fetch(`${BACKEND_URL}/sentence/space`, {
            method: 'POST'
        });
        if (response.ok) {
            const data = await response.json();
            updateSentenceFromAPI(data);
        }
    } catch (error) {
        console.error('Error adding space:', error);
    }
}

async function backspace() {
    try {
        const response = await fetch(`${BACKEND_URL}/sentence/backspace`, {
            method: 'POST'
        });
        if (response.ok) {
            const data = await response.json();
            updateSentenceFromAPI(data);
        }
    } catch (error) {
        console.error('Error with backspace:', error);
    }
}

async function clearSentence() {
    try {
        const response = await fetch(`${BACKEND_URL}/sentence/clear`, {
            method: 'POST'
        });
        if (response.ok) {
            document.getElementById('completedWords').textContent = '';
            document.getElementById('currentWord').textContent = '';
        }
    } catch (error) {
        console.error('Error clearing sentence:', error);
    }
}

function updateSentenceFromAPI(data) {
    const completedEl = document.getElementById('completedWords');
    const currentEl = document.getElementById('currentWord');
    
    completedEl.textContent = data.sentence ? data.sentence + ' ' : '';
    currentEl.textContent = data.current_word || '';
}

// ============================================================
// Speech & Copy
// ============================================================

function speakSentence() {
    const completedWords = document.getElementById('completedWords').textContent;
    const currentWord = document.getElementById('currentWord').textContent;
    const sentence = (completedWords + currentWord).trim();
    
    if (!sentence) {
        alert('No sentence to speak');
        return;
    }
    
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(sentence);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        speechSynthesis.speak(utterance);
    } else {
        alert('Speech synthesis not supported in this browser');
    }
}

function copySentence() {
    const completedWords = document.getElementById('completedWords').textContent;
    const currentWord = document.getElementById('currentWord').textContent;
    const sentence = (completedWords + currentWord).trim();
    
    if (!sentence) {
        alert('No sentence to copy');
        return;
    }
    
    navigator.clipboard.writeText(sentence).then(() => {
        alert('Sentence copied to clipboard!');
    }).catch(err => {
        console.error('Copy failed:', err);
    });
}

// ============================================================
// UI Updates
// ============================================================

function updateUI() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    if (isRunning) {
        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
    } else {
        startBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopInterpreter();
});
