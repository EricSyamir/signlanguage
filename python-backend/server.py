"""
SignBridge Python Backend - Sign Language Interpreter Web Service
Based on: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning

This server provides real-time sign language interpretation with sentence building.
"""

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import cv2
import pickle
import os
import io
from typing import Optional
from datetime import datetime
import base64

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI(
    title="SignBridge Sign Language Interpreter API",
    description="Real-time sign language interpretation with sentence building",
    version="2.0.0"
)

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for single-service deployment on Render)
static_dir = os.path.join(os.path.dirname(__file__), '..')
if os.path.exists(os.path.join(static_dir, 'index.html')):
    # Mount static assets
    app.mount("/css", StaticFiles(directory=os.path.join(static_dir, 'css')), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(static_dir, 'js')), name="js")
    app.mount("/images", StaticFiles(directory=os.path.join(static_dir, 'images')), name="images")
    
    # Serve HTML pages
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(static_dir, 'index.html'))
    
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
# Model Configuration
# ============================================================

# Image size for Sign Language MNIST model (28x28)
IMAGE_X, IMAGE_Y = 28, 28

# ASL Gesture labels for Sign Language MNIST (24 classes: A-Z excluding J and Z)
# Class mapping: 0-8 = A-I, 9 = K, 10-17 = L-S, 18-23 = T-Y
ASL_GESTURES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    9: "K", 10: "L", 11: "M", 12: "N", 13: "O",
    14: "P", 15: "Q", 16: "R", 17: "S",
    18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"
}

# Note: Sign Language MNIST only has letters (A-Z excluding J and Z)
# For sentence building, we'll combine letters into words
WORD_GESTURES = {}  # Empty for Sign Language MNIST (letters only)

# Global model variable
model = None
hist = None

# Sentence building state
class SentenceBuilder:
    def __init__(self):
        self.current_sentence = []
        self.current_word = ""
        self.last_prediction = ""
        self.same_frame_count = 0
        self.min_confidence = 0.70  # 70% confidence threshold
        self.frames_for_confirm = 15  # Frames needed to confirm gesture
        
    def add_character(self, char: str, confidence: float):
        """Add a character to current word"""
        if confidence < self.min_confidence:
            return
            
        if char == self.last_prediction:
            self.same_frame_count += 1
        else:
            self.same_frame_count = 0
            self.last_prediction = char
            
        # Confirm character after consistent frames
        # Sign Language MNIST only has letters, so add to current word
        if self.same_frame_count >= self.frames_for_confirm:
            self.current_word += char
            self.same_frame_count = 0
            
    def add_space(self):
        """Add space (complete current word)"""
        if self.current_word:
            self.current_sentence.append(self.current_word)
            self.current_word = ""
            
    def get_sentence(self) -> str:
        """Get the full sentence"""
        words = self.current_sentence.copy()
        if self.current_word:
            words.append(self.current_word)
        return " ".join(words)
    
    def clear(self):
        """Clear sentence"""
        self.current_sentence = []
        self.current_word = ""
        self.last_prediction = ""
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
# Model Loading & Prediction Functions
# ============================================================

def load_keras_model():
    """Load the trained Keras model"""
    global model
    
    # Check for model files (Sign Language MNIST model)
    model_paths = [
        'models/sign_language_cnn_model.h5',
        '../python-backend/models/sign_language_cnn_model.h5',
        'sign_language_cnn_model.h5',
        # Legacy paths
        'models/cnn_model_keras2.h5',
        '../sl-interpreter-model/Code/cnn_model_keras2.h5',
        'cnn_model_keras2.h5'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                from tensorflow.keras.models import load_model
                model = load_model(path)
                print(f"✅ Model loaded from: {path}")
                print(f"   Model input shape: {model.input_shape}")
                print(f"   Model output classes: {model.output_shape[1]}")
                return True
            except Exception as e:
                print(f"❌ Error loading model from {path}: {e}")
    
    print("⚠️ No trained model found. Using simulated predictions.")
    print("   To use real recognition, place sign_language_cnn_model.h5 in models/ folder")
    return False

def load_histogram():
    """Load hand histogram for segmentation (optional)"""
    global hist
    
    base_dir = os.path.dirname(__file__)
    hist_paths = [
        os.path.join(base_dir, 'models', 'hist'),
        'models/hist',
        'hist'
    ]
    
    for path in hist_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    hist = pickle.load(f)
                print(f"✅ Histogram loaded from: {path}")
                return True
            except Exception as e:
                print(f"❌ Error loading histogram from {path}: {e}")
    
    print("⚠️ No histogram found. Hand segmentation will be basic.")
    return False


def preprocess_image(image_bytes) -> np.ndarray:
    """
    Preprocess image for model prediction
    Following the exact approach from the original repo
    """
    try:
        # Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
            
        # Flip horizontally (mirror for camera)
        img = cv2.flip(img, 1)
        
        # Extract hand region using histogram backprojection if available
        if hist is not None:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
            
            # Apply morphological operations
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            
            # Blur and threshold
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 10000:
                    x, y, w, h = cv2.boundingRect(contour)
                    hand_img = thresh[y:y+h, x:x+w]
                    
                    # Make square
                    if w > h:
                        hand_img = cv2.copyMakeBorder(hand_img, int((w-h)/2), int((w-h)/2), 
                                                      0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    elif h > w:
                        hand_img = cv2.copyMakeBorder(hand_img, 0, 0, 
                                                      int((h-w)/2), int((h-w)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
                    
                    # Resize to model input
                    hand_img = cv2.resize(hand_img, (IMAGE_X, IMAGE_Y))
                    return hand_img
        
        # Fallback: Simple grayscale conversion and resize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Center crop to focus on hand area (for better recognition)
        h, w = gray.shape
        crop_size = min(h, w) // 2
        center_y, center_x = h // 2, w // 2
        cropped = gray[center_y-crop_size:center_y+crop_size, 
                       center_x-crop_size:center_x+crop_size]
        
        # Resize to model input size (28x28 for Sign Language MNIST)
        resized = cv2.resize(cropped, (IMAGE_X, IMAGE_Y))
        
        return resized
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_gesture(processed_image: np.ndarray) -> tuple:
    """
    Predict gesture from preprocessed image
    Returns: (label, confidence)
    """
    global model
    
    if processed_image is None:
        return "Error", 0.0
    
    try:
        # Reshape for model input
        img_array = processed_image.astype(np.float32) / 255.0
        img_array = np.reshape(img_array, (1, IMAGE_X, IMAGE_Y, 1))
        
        if model is not None:
            # Real model prediction
            predictions = model.predict(img_array, verbose=0)[0]
            pred_class = np.argmax(predictions)
            confidence = float(predictions[pred_class])
            label = ASL_GESTURES.get(pred_class, f"Class_{pred_class}")
        else:
            # Simulated prediction for testing
            import random
            pred_class = random.randint(0, len(ASL_GESTURES) - 1)
            confidence = random.uniform(0.70, 0.98)
            label = ASL_GESTURES.get(pred_class, "Unknown")
        
        return label, confidence
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0


# ============================================================
# API Endpoints
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_keras_model()
    load_histogram()
    print("\n" + "=" * 60)
    print("SignBridge Sign Language Interpreter API")
    print("Based on: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning")
    print("=" * 60)
    print("Server ready at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("=" * 60 + "\n")


@app.get("/api")
async def api_status():
    """API status endpoint"""
    return {
        "service": "SignBridge Sign Language Interpreter",
        "status": "running",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "histogram_loaded": hist is not None
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "histogram_loaded": hist is not None,
        "supported_gestures": len(ASL_GESTURES),
        "features": [
            "Single gesture recognition",
            "Continuous recognition",
            "Sentence building",
            "Real-time interpretation"
        ]
    }


@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    language: str = Form("ASL")
):
    """
    Predict gesture from a single image
    
    Args:
        file: Image file
        language: "ASL" or "MSL"
    
    Returns:
        {
            "label": "A",
            "confidence": 0.95,
            "language": "ASL"
        }
    """
    try:
        contents = await file.read()
        
        # Preprocess and predict
        processed = preprocess_image(contents)
        label, confidence = predict_gesture(processed)
        
        return {
            "label": label,
            "confidence": confidence,
            "language": language,
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
    
    Returns current prediction AND the building sentence
    """
    try:
        contents = await file.read()
        
        # Preprocess and predict
        processed = preprocess_image(contents)
        label, confidence = predict_gesture(processed)
        
        # Add to sentence builder
        sentence_builder.add_character(label, confidence)
        
        return {
            "label": label,
            "confidence": confidence,
            "language": language,
            "current_word": sentence_builder.current_word,
            "sentence": sentence_builder.get_sentence(),
            "confirmed": sentence_builder.same_frame_count >= sentence_builder.frames_for_confirm,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/sentence/space")
async def add_space():
    """Add space (complete current word)"""
    sentence_builder.add_space()
    return {
        "sentence": sentence_builder.get_sentence(),
        "current_word": sentence_builder.current_word
    }


@app.post("/sentence/backspace")
async def backspace():
    """Remove last character or word"""
    sentence_builder.backspace()
    return {
        "sentence": sentence_builder.get_sentence(),
        "current_word": sentence_builder.current_word
    }


@app.post("/sentence/clear")
async def clear_sentence():
    """Clear the sentence"""
    sentence_builder.clear()
    return {"sentence": "", "current_word": ""}


@app.get("/sentence")
async def get_sentence():
    """Get current sentence"""
    return {
        "sentence": sentence_builder.get_sentence(),
        "current_word": sentence_builder.current_word,
        "words": sentence_builder.current_sentence
    }


@app.get("/gestures")
async def get_gestures():
    """Get list of supported gestures"""
    return {
        "gestures": ASL_GESTURES,
        "word_gestures": WORD_GESTURES,
        "total": len(ASL_GESTURES)
    }


# ============================================================
# WebSocket for Real-time Recognition
# ============================================================

@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """
    WebSocket endpoint for real-time continuous recognition
    
    Send base64 encoded image frames, receive predictions
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive frame as base64
            data = await websocket.receive_text()
            
            try:
                # Decode base64 image
                if data.startswith('data:image'):
                    data = data.split(',')[1]
                    
                image_bytes = base64.b64decode(data)
                
                # Process and predict
                processed = preprocess_image(image_bytes)
                label, confidence = predict_gesture(processed)
                
                # Update sentence builder
                sentence_builder.add_character(label, confidence)
                
                # Send result
                await websocket.send_json({
                    "label": label,
                    "confidence": confidence,
                    "current_word": sentence_builder.current_word,
                    "sentence": sentence_builder.get_sentence(),
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for Render) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
