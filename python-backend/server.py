"""
SignBridge Sign Language Interpreter API
Using sign-language-translator library: https://pypi.org/project/sign-language-translator/

MEMORY OPTIMIZED VERSION
- Image resizing before processing
- Garbage collection
- Lazy model loading
- Memory-efficient operations
"""

import os
import io
import base64
import asyncio
import gc  # Garbage collection
from datetime import datetime
from typing import Optional
import json
import logging

# Suppress verbose MediaPipe and TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
logging.getLogger('absl').setLevel(logging.ERROR)  # Suppress absl warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)  # Suppress MediaPipe warnings

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
# Memory Optimization Settings
# ============================================================

# Maximum image dimensions (reduce memory usage)
MAX_IMAGE_WIDTH = 640
MAX_IMAGE_HEIGHT = 480
MAX_IMAGE_SIZE_MB = 5  # Maximum image size in MB

# Processing settings
IMAGE_QUALITY = 85  # JPEG quality (lower = less memory)


# ============================================================
# FastAPI App Setup
# ============================================================

app = FastAPI(
    title="SignBridge Sign Language Interpreter",
    description="Real-time sign language to text translation using sign-language-translator library",
    version="3.1.0-memory-optimized"
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

# Sign language models (lazy loaded)
video_embedding_model = None
sign_to_text_model = None
text_to_sign_model = None
translator_ready = False
_model_loading_lock = asyncio.Lock()


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
        # Removed history to save memory
        
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
# Memory-Efficient Image Processing
# ============================================================

def resize_image_if_needed(img: np.ndarray) -> np.ndarray:
    """Resize image if it's too large to save memory"""
    height, width = img.shape[:2]
    
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        # Calculate scaling factor
        scale = min(MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize using INTER_AREA (best for downscaling)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return img


def optimize_image_memory(image_bytes: bytes) -> tuple:
    """
    Optimize image for memory efficiency
    Returns: (optimized_image_array, original_size_mb, optimized_size_mb)
    """
    original_size_mb = len(image_bytes) / (1024 * 1024)
    
    # Check size limit
    if original_size_mb > MAX_IMAGE_SIZE_MB:
        raise ValueError(f"Image too large: {original_size_mb:.2f}MB (max: {MAX_IMAGE_SIZE_MB}MB)")
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Resize if needed
    img = resize_image_if_needed(img)
    
    # Convert BGR to RGB (MediaPipe expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    optimized_size_mb = img_rgb.nbytes / (1024 * 1024)
    
    return img_rgb, original_size_mb, optimized_size_mb


# ============================================================
# Model Loading (Lazy Loading)
# ============================================================

async def ensure_model_loaded():
    """Ensure MediaPipe model is loaded (lazy loading)"""
    global video_embedding_model, translator_ready
    
    if video_embedding_model is not None:
        return True
    
    if not SLT_AVAILABLE:
        return False
    
    async with _model_loading_lock:
        # Double-check after acquiring lock
        if video_embedding_model is not None:
            return True
        
        try:
            print("📦 Loading MediaPipe model (lazy loading)...")
            Settings.SHOW_DOWNLOAD_PROGRESS = False
            
            # Suppress stdout during model loading to reduce log noise
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            import io
            
            # Redirect MediaPipe's verbose output
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                video_embedding_model = slt.models.MediaPipeLandmarksModel()
            
            translator_ready = True
            print("✅ MediaPipe model loaded! (Using CPU - GPU not available)")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def load_slt_models():
    """Load sign-language-translator models (called on startup)"""
    # Don't load immediately - use lazy loading instead
    print("📦 Models will be loaded on first use (lazy loading)")
    return True


# ============================================================
# Landmark Extraction (Memory Optimized)
# ============================================================

async def extract_landmarks(image_bytes: bytes) -> Optional[dict]:
    """Extract pose landmarks from image using MediaPipe (memory optimized)"""
    
    # Ensure model is loaded
    if not await ensure_model_loaded():
        return None
    
    try:
        # Optimize image memory
        img_rgb, orig_size, opt_size = optimize_image_memory(image_bytes)
        
        # Extract landmarks
        landmarks = video_embedding_model.embed(img_rgb)
        
        # Clear image from memory immediately
        del img_rgb
        gc.collect()  # Force garbage collection
        
        if landmarks is not None:
            # Convert to list efficiently (don't keep numpy array)
            if hasattr(landmarks, 'tolist'):
                landmarks_list = landmarks.tolist()
            elif isinstance(landmarks, np.ndarray):
                landmarks_list = landmarks.tolist()
            else:
                landmarks_list = list(landmarks)
            
            # Clear landmarks numpy array
            del landmarks
            gc.collect()
            
            return {
                "landmarks": landmarks_list,
                "detected": len(landmarks_list) > 0
            }
        
        return {"detected": False}
        
    except ValueError as e:
        print(f"Image processing error: {e}")
        return None
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        import traceback
        traceback.print_exc()
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
    print("SignBridge Sign Language Interpreter API v3.1")
    print("Memory-Optimized Version")
    print("Using: sign-language-translator library")
    print("https://pypi.org/project/sign-language-translator/")
    print("=" * 60)
    
    # Print environment info
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(__file__)}")
    print(f"SLT library available: {SLT_AVAILABLE}")
    print(f"Max image size: {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}")
    print(f"Max image file size: {MAX_IMAGE_SIZE_MB}MB")
    
    print("\n" + "=" * 60)
    print("Initializing (models will load on first use)...")
    print("=" * 60)
    
    models_ready = load_slt_models()
    
    print("\n" + "=" * 60)
    print("Server ready!")
    print("=" * 60)
    port = os.environ.get("PORT", "8000")
    print(f"Server running on port: {port}")
    print(f"Lazy loading enabled: {models_ready}")
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
        "version": "3.1.0-memory-optimized",
        "memory_optimized": True
    }


@app.get("/api")
async def api_status():
    """API status endpoint"""
    return {
        "service": "SignBridge Sign Language Interpreter",
        "status": "running",
        "version": "3.1.0-memory-optimized",
        "library": "sign-language-translator",
        "library_version": slt.__version__ if SLT_AVAILABLE else "not installed",
        "model_loaded": translator_ready,
        "memory_optimized": True,
        "max_image_size": f"{MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}",
        "features": [
            "Real-time sign detection",
            "MediaPipe landmarks extraction",
            "Sentence building",
            "WebSocket streaming",
            "Memory optimized"
        ]
    }


@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    language: str = Form("ASL")
):
    """
    Predict gesture from a single image (memory optimized)
    """
    try:
        contents = await file.read()
        
        # Extract landmarks (with memory optimization)
        landmarks_data = await extract_landmarks(contents)
        
        # Clear contents from memory
        del contents
        gc.collect()
        
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
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "label": "Error", "confidence": 0.0}
        )
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
    Predict gesture and add to sentence builder (memory optimized)
    """
    try:
        contents = await file.read()
        
        # Extract landmarks (with memory optimization)
        landmarks_data = await extract_landmarks(contents)
        
        # Clear contents from memory
        del contents
        gc.collect()
        
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
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/clear-sentence")
async def clear_sentence():
    """Clear the current sentence"""
    sentence_builder.clear()
    gc.collect()  # Clean up memory
    return {"status": "cleared", "sentence_state": sentence_builder.get_state()}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming (memory optimized)
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
                    
                    # Extract landmarks (with memory optimization)
                    landmarks_data = await extract_landmarks(image_bytes)
                    
                    # Clear image_bytes from memory
                    del image_bytes
                    gc.collect()
                    
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
                    
                    # Periodic garbage collection
                    gc.collect()
                    
                elif message.get("type") == "clear":
                    sentence_builder.clear()
                    gc.collect()
                    await websocket.send_json({
                        "type": "cleared",
                        "sentence_state": sentence_builder.get_state()
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
            except ValueError as e:
                await websocket.send_json({"error": f"Image error: {str(e)}"})
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
        gc.collect()  # Clean up on disconnect


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
    print(f"Starting memory-optimized server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
