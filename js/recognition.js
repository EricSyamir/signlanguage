// SignBridge - Gesture Recognition with Python Backend
// Based on: https://github.com/yumdmb/sl-recognition-v1-fe

let stream = null;
let currentImageData = null;
let recognitionInterval = null;
let isRecognizing = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        setupDragAndDrop(uploadArea, (file) => {
            if (validateImage(file)) {
                displayImage(file);
            }
        });
    }
    
    // Check if Python backend is running
    checkBackendStatus();
});

// Check Python backend status
async function checkBackendStatus() {
    try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
            const data = await response.json();
            showAlert('Python backend is running and ready!', 'success');
            console.log('Backend status:', data);
        }
    } catch (error) {
        showAlert('Python backend not detected. Please start the server (see documentation).', 'info');
        console.log('Backend not running. Start with: python python-backend/server.py');
    }
}

// Mode Switching
function switchMode(mode) {
    const uploadMode = document.getElementById('uploadMode');
    const cameraMode = document.getElementById('cameraMode');
    const uploadBtn = document.getElementById('uploadModeBtn');
    const cameraBtn = document.getElementById('cameraModeBtn');
    
    // Stop any ongoing recognition
    stopContinuousRecognition();
    
    if (mode === 'upload') {
        uploadMode.classList.remove('hidden');
        cameraMode.classList.add('hidden');
        uploadBtn.classList.remove('btn-outline');
        uploadBtn.classList.add('btn-primary');
        cameraBtn.classList.remove('btn-primary');
        cameraBtn.classList.add('btn-outline');
        stopCamera();
    } else {
        uploadMode.classList.add('hidden');
        cameraMode.classList.remove('hidden');
        cameraBtn.classList.remove('btn-outline');
        cameraBtn.classList.add('btn-primary');
        uploadBtn.classList.remove('btn-primary');
        uploadBtn.classList.add('btn-outline');
    }
    
    clearPreview();
}

// File Upload Handler
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && validateImage(file)) {
        displayImage(file);
    }
}

// Display Image
function displayImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        currentImageData = e.target.result;
        const previewImage = document.getElementById('previewImage');
        const previewContainer = document.getElementById('previewContainer');
        
        previewImage.src = currentImageData;
        previewContainer.classList.remove('hidden');
        
        // Hide results if showing
        document.getElementById('resultContainer').classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// Camera Functions
async function startCamera() {
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
        
        // Show/hide buttons
        document.getElementById('startCameraBtn').classList.add('hidden');
        document.getElementById('captureBtn').classList.remove('hidden');
        document.getElementById('stopCameraBtn').classList.remove('hidden');
        document.getElementById('continuousRecognitionBtn').classList.remove('hidden');
        
        showAlert('Camera started successfully!', 'success');
    } catch (error) {
        console.error('Error accessing camera:', error);
        showAlert('Could not access camera. Please check permissions.', 'error');
    }
}

function stopCamera() {
    stopContinuousRecognition();
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        
        const video = document.getElementById('camera');
        video.srcObject = null;
        
        // Show/hide buttons
        document.getElementById('startCameraBtn').classList.remove('hidden');
        document.getElementById('captureBtn').classList.add('hidden');
        document.getElementById('stopCameraBtn').classList.add('hidden');
        document.getElementById('continuousRecognitionBtn').classList.add('hidden');
    }
}

function captureImage() {
    const video = document.getElementById('camera');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    currentImageData = canvas.toDataURL('image/png');
    
    const previewImage = document.getElementById('previewImage');
    const previewContainer = document.getElementById('previewContainer');
    
    previewImage.src = currentImageData;
    previewContainer.classList.remove('hidden');
    
    // Hide results if showing
    document.getElementById('resultContainer').classList.add('hidden');
    
    showAlert('Image captured! Click "Analyze Gesture" to proceed.', 'success');
}

// Continuous Recognition (like original repo)
function toggleContinuousRecognition() {
    if (isRecognizing) {
        stopContinuousRecognition();
    } else {
        startContinuousRecognition();
    }
}

function startContinuousRecognition() {
    if (isRecognizing) return;
    
    isRecognizing = true;
    document.getElementById('continuousRecognitionBtn').textContent = '⏸️ Stop Recognition';
    document.getElementById('continuousRecognitionBtn').classList.add('btn-secondary');
    document.getElementById('continuousRecognitionBtn').classList.remove('btn-outline');
    
    showAlert('Continuous recognition started!', 'success');
    
    // Send frames every 300ms (like original repo)
    recognitionInterval = setInterval(() => {
        sendFrameToBackend();
    }, 300);
}

function stopContinuousRecognition() {
    if (!isRecognizing) return;
    
    isRecognizing = false;
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
    }
    
    const btn = document.getElementById('continuousRecognitionBtn');
    if (btn) {
        btn.textContent = '🔄 Start Continuous Recognition';
        btn.classList.remove('btn-secondary');
        btn.classList.add('btn-outline');
    }
}

// Send frame to Python backend (exactly like original repo)
async function sendFrameToBackend() {
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
        formData.append('language', 'ASL'); // Can be made configurable
        
        try {
            const response = await fetch('http://localhost:8000/predict-image', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                displayContinuousResult(data);
            }
        } catch (error) {
            console.error('Error sending frame:', error);
            stopContinuousRecognition();
            showAlert('Backend connection lost. Please ensure Python server is running.', 'error');
        }
    }, 'image/jpeg');
}

// Display continuous recognition results
function displayContinuousResult(data) {
    const container = document.getElementById('continuousResultContainer');
    if (!container) return;
    
    const confidence = (data.confidence * 100).toFixed(1);
    container.innerHTML = `
        <div style="text-align: center; padding: 1rem; background: var(--light-bg); border-radius: 8px;">
            <div style="font-size: 2rem; font-weight: bold; color: var(--success-color);">
                ${data.label}
            </div>
            <div style="color: var(--text-light); margin-top: 0.5rem;">
                ${data.language} - Confidence: ${confidence}%
            </div>
        </div>
    `;
    container.classList.remove('hidden');
}

function clearPreview() {
    currentImageData = null;
    document.getElementById('previewContainer').classList.add('hidden');
    document.getElementById('resultContainer').classList.add('hidden');
    document.getElementById('loadingContainer').classList.add('hidden');
    document.getElementById('fileInput').value = '';
}

// Analyze gesture (for uploaded images)
async function analyzeGesture() {
    if (!currentImageData) {
        showAlert('Please upload or capture an image first.', 'error');
        return;
    }
    
    // Show loading
    document.getElementById('loadingContainer').classList.remove('hidden');
    document.getElementById('resultContainer').classList.add('hidden');
    
    try {
        // Convert base64 to blob
        const response = await fetch(currentImageData);
        const blob = await response.blob();
        
        // Create FormData
        const formData = new FormData();
        formData.append('file', blob, 'gesture.jpg');
        formData.append('language', 'ASL'); // Can be made configurable
        
        // Send to Python backend
        const apiResponse = await fetch('http://localhost:8000/predict-image', {
            method: 'POST',
            body: formData
        });
        
        if (!apiResponse.ok) {
            throw new Error('Backend request failed');
        }
        
        const data = await apiResponse.json();
        
        // Display results
        displayResults([{
            sign: data.label,
            language: data.language,
            confidence: Math.round(data.confidence * 100)
        }]);
        
        document.getElementById('loadingContainer').classList.add('hidden');
        document.getElementById('resultContainer').classList.remove('hidden');
        
        showAlert(`Recognized as "${data.label}" with ${(data.confidence * 100).toFixed(1)}% confidence`, 'success');
        
    } catch (error) {
        console.error('Recognition error:', error);
        document.getElementById('loadingContainer').classList.add('hidden');
        showAlert('Python backend not available. Please start the server: python python-backend/server.py', 'error');
    }
}

// Display Results
function displayResults(results) {
    const resultsList = document.getElementById('resultsList');
    
    resultsList.innerHTML = results.map(result => `
        <div class="result-item">
            <div>
                <strong style="font-size: 1.2rem;">${result.sign}</strong>
                <p style="color: var(--text-light); margin-top: 0.25rem;">${result.language}</p>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence}%;"></div>
                </div>
                <span style="font-weight: bold; color: ${getConfidenceColor(result.confidence)};">
                    ${result.confidence}%
                </span>
            </div>
        </div>
    `).join('');
    
    // Save to recognition history
    saveRecognitionHistory(results[0]);
}

function getConfidenceColor(confidence) {
    if (confidence >= 80) return 'var(--success-color)';
    if (confidence >= 60) return 'var(--warning-color)';
    return 'var(--danger-color)';
}

// Recognition History
function saveRecognitionHistory(result) {
    const history = Storage.get('recognitionHistory', []);
    history.unshift({
        ...result,
        timestamp: new Date().toISOString()
    });
    
    // Keep only last 50 results
    if (history.length > 50) {
        history.pop();
    }
    
    Storage.set('recognitionHistory', history);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopCamera();
    stopContinuousRecognition();
});

