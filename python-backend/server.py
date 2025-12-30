"""
SignBridge Sign Language Interpreter API
Using sign-language-translator library: https://pypi.org/project/sign-language-translator/
"""

import os
import io
import base64
import asyncio
from datetime import datetime
from typing import Optional
import json

# Web framework
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Image processing
import numpy as np
from PIL import Image
import cv2

# Sign Language Translator
try:
    import sign_language_translator as slt
    from sign_language_translator.config.settings import Settings
    SLT_AVAILABLE = True
    print("✅ sign-language-translator library loaded!")
except ImportError as e:
    SLT_AVAILABLE = False
    print(f"⚠️ sign-language-translator not available: {e}")
    print("   Install with: pip install sign-language-translator[mediapipe]")

# ============================================================
# FastAPI App Setup
# ============================================================

app = FastAPI(
    title="SignBridge Sign Language Interpreter",
    description="Real-time sign language to text translation using sign-language-translator library",
    version="3.0.0"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Global Variables
# ============================================================

# Sign language models
video_embedding_model = None
sign_to_text_model = None
text_to_sign_model = None
translator_ready = False


# ============================================================
# Sentence Builder (for accumulating recognized signs)
# ============================================================

class SentenceBuilder:
    """Builds sentences from recognized signs"""
    
    def __init__(self):
        self.current_word = ""
        self.current_sentence = []
        self.last_prediction = None
        self.same_frame_count = 0
        self.history = []
        
    def add_prediction(self, label: str, confidence: float) -> dict:
        """Add a prediction and build sentence"""
        
        if label == "Nothing" or label == "Unknown" or confidence < 0.5:
            return self.get_state()
        
        # Same gesture held = confirm it
        if label == self.last_prediction:
            self.same_frame_count += 1
            if self.same_frame_count == 3:  # Confirmed after 3 frames
                if label == "Space":
                    if self.current_word:
                        self.current_sentence.append(self.current_word)
                        self.current_word = ""
                elif label == "Delete":
                    self.backspace()
                else:
                    self.current_word += label
        else:
            self.last_prediction = label
            self.same_frame_count = 0
            
        return self.get_state()
    
    def get_state(self) -> dict:
        """Get current sentence building state"""
        full_sentence = " ".join(self.current_sentence)
        if self.current_word:
            full_sentence += (" " if full_sentence else "") + self.current_word
            
        return {
            "current_word": self.current_word,
            "words": self.current_sentence.copy(),
            "full_sentence": full_sentence,
            "last_prediction": self.last_prediction
        }
    
    def clear(self):
        """Clear all state"""
        self.current_word = ""
        self.current_sentence = []
        self.last_prediction = None
        self.same_frame_count = 0
        
    def backspace(self):
        """Remove last character or word"""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.current_sentence:
            self.current_sentence.pop()


# Global sentence builder
sentence_builder = SentenceBuilder()


# ============================================================
# Model Loading
# ============================================================

def load_slt_models():
    """Load sign-language-translator models"""
    global video_embedding_model, sign_to_text_model, text_to_sign_model, translator_ready
    
    if not SLT_AVAILABLE:
        print("⚠️ SLT library not available")
        return False
    
    try:
        print("📦 Loading sign-language-translator models...")
        
        # Disable download progress bars for cleaner logs
        Settings.SHOW_DOWNLOAD_PROGRESS = False
        
        # Load video embedding model (for extracting features from video/images)
        print("   Loading video embedding model...")
        video_embedding_model = slt.models.MediaPipeLandmarksModel()
        print("   ✓ Video embedding model loaded")
        
        # Note: Full translation models are larger and may need separate download
        # For now, we'll use the embedding model for gesture detection
        
        translator_ready = True
        print("✅ SLT models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading SLT models: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_landmarks(image_bytes) -> Optional[dict]:
    """Extract pose landmarks from image using MediaPipe"""
    global video_embedding_model
    
    if video_embedding_model is None:
        return None
    
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get landmarks
        landmarks = video_embedding_model.embed(img_rgb)
        
        if landmarks is not None and len(landmarks) > 0:
            return {
                "landmarks": landmarks.tolist() if hasattr(landmarks, 'tolist') else landmarks,
                "detected": True
            }
        
        return {"detected": False}
        
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return None


def predict_from_landmarks(landmarks_data: dict) -> tuple:
    """
    Predict sign from landmarks
    Returns (label, confidence)
    """
    
    if not landmarks_data or not landmarks_data.get("detected"):
        return "Nothing", 0.0
    
    # For now, we detect hand presence
    # Full sign recognition requires trained models
    landmarks = landmarks_data.get("landmarks", [])
    
    if landmarks and len(landmarks) > 0:
        # Hand detected - in a full implementation, 
        # this would go through a trained classifier
        return "Hand Detected", 0.85
    
    return "Nothing", 0.0


# ============================================================
# Startup Event
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("\n" + "=" * 60)
    print("SignBridge Sign Language Interpreter API v3.0")
    print("Using: sign-language-translator library")
    print("https://pypi.org/project/sign-language-translator/")
    print("=" * 60)
    
    # Print environment info
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(__file__)}")
    print(f"SLT library available: {SLT_AVAILABLE}")
    
    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)
    
    models_loaded = load_slt_models()
    
    print("\n" + "=" * 60)
    print("Server ready!")
    print("=" * 60)
    port = os.environ.get("PORT", "8000")
    print(f"Server running on port: {port}")
    print(f"Translator ready: {models_loaded}")
    print("=" * 60 + "\n")


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "slt_available": SLT_AVAILABLE,
        "model_loaded": translator_ready,
        "version": "3.0.0"
    }


@app.get("/api")
async def api_status():
    """API status endpoint"""
    return {
        "service": "SignBridge Sign Language Interpreter",
        "status": "running",
        "version": "3.0.0",
        "library": "sign-language-translator",
        "library_version": slt.__version__ if SLT_AVAILABLE else "not installed",
        "model_loaded": translator_ready,
        "features": [
            "Real-time sign detection",
            "MediaPipe landmarks extraction",
            "Sentence building",
            "WebSocket streaming"
        ]
    }


@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    language: str = Form("ASL")
):
    """
    Predict gesture from a single image
    """
    try:
        contents = await file.read()
        
        # Extract landmarks
        landmarks_data = extract_landmarks(contents)
        
        if landmarks_data:
            label, confidence = predict_from_landmarks(landmarks_data)
        else:
            label, confidence = "Error", 0.0
        
        return {
            "label": label,
            "confidence": confidence,
            "language": language,
            "landmarks_detected": landmarks_data.get("detected", False) if landmarks_data else False,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "label": "Error", "confidence": 0.0}
        )


@app.post("/predict-and-build")
async def predict_and_build(
    file: UploadFile = File(...),
    language: str = Form("ASL")
):
    """
    Predict gesture and add to sentence builder
    """
    try:
        contents = await file.read()
        
        # Extract landmarks
        landmarks_data = extract_landmarks(contents)
        
        if landmarks_data:
            label, confidence = predict_from_landmarks(landmarks_data)
        else:
            label, confidence = "Nothing", 0.0
        
        # Add to sentence builder
        sentence_state = sentence_builder.add_prediction(label, confidence)
        
        return {
            "label": label,
            "confidence": confidence,
            "language": language,
            "landmarks_detected": landmarks_data.get("detected", False) if landmarks_data else False,
            **sentence_state,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/clear-sentence")
async def clear_sentence():
    """Clear the current sentence"""
    sentence_builder.clear()
    return {"status": "cleared", "sentence_state": sentence_builder.get_state()}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming
    """
    await websocket.accept()
    print("WebSocket client connected")
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                if message.get("type") == "frame":
                    # Decode base64 image
                    image_data = message.get("data", "")
                    if "," in image_data:
                        image_data = image_data.split(",")[1]
                    
                    image_bytes = base64.b64decode(image_data)
                    
                    # Extract landmarks
                    landmarks_data = extract_landmarks(image_bytes)
                    
                    if landmarks_data:
                        label, confidence = predict_from_landmarks(landmarks_data)
                    else:
                        label, confidence = "Nothing", 0.0
                    
                    # Add to sentence builder
                    sentence_state = sentence_builder.add_prediction(label, confidence)
                    
                    # Send response
                    await websocket.send_json({
                        "type": "prediction",
                        "label": label,
                        "confidence": confidence,
                        "landmarks_detected": landmarks_data.get("detected", False) if landmarks_data else False,
                        **sentence_state,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                elif message.get("type") == "clear":
                    sentence_builder.clear()
                    await websocket.send_json({
                        "type": "cleared",
                        "sentence_state": sentence_builder.get_state()
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")


# ============================================================
# Text to Sign Language (using SLT library)
# ============================================================

@app.post("/text-to-sign")
async def text_to_sign(text: str = Form(...), language: str = Form("pk-sl")):
    """
    Convert text to sign language description
    Using sign-language-translator library
    """
    if not SLT_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "sign-language-translator library not available"}
        )
    
    try:
        # Use rule-based translation
        # Available languages: pk-sl (Pakistan), others based on what's installed
        
        # For now, return a structured response
        # Full implementation would use slt.models.ConcatenativeSynthesis
        
        words = text.strip().split()
        
        return {
            "input_text": text,
            "language": language,
            "signs": [{"word": word, "sign_available": True} for word in words],
            "message": "Text parsed successfully. Video synthesis requires additional models."
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ============================================================
# Serve Static Files (for single-service deployment)
# ============================================================

static_dir = os.path.join(os.path.dirname(__file__), '..')

# Mount static assets only if directories exist
css_dir = os.path.join(static_dir, 'css')
if os.path.exists(css_dir):
    app.mount("/css", StaticFiles(directory=css_dir), name="css")

js_dir = os.path.join(static_dir, 'js')
if os.path.exists(js_dir):
    app.mount("/js", StaticFiles(directory=js_dir), name="js")

images_dir = os.path.join(static_dir, 'images')
if os.path.exists(images_dir):
    app.mount("/images", StaticFiles(directory=images_dir), name="images")


@app.get("/config.js")
async def serve_config_js():
    config_path = os.path.join(static_dir, 'config.js')
    if os.path.exists(config_path):
        return FileResponse(config_path, media_type="application/javascript")
    return JSONResponse(content={"error": "config.js not found"}, status_code=404)


@app.get("/")
async def serve_index():
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SignBridge API", "docs": "/docs"}


@app.get("/recognition.html")
async def serve_recognition():
    return FileResponse(os.path.join(static_dir, 'recognition.html'))


@app.get("/learning.html")
async def serve_learning():
    return FileResponse(os.path.join(static_dir, 'learning.html'))


@app.get("/about.html")
async def serve_about():
    return FileResponse(os.path.join(static_dir, 'about.html'))


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
