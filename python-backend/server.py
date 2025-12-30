"""
SignBridge Python Backend - FastAPI Server for Gesture Recognition
Based on: https://github.com/yumdmb/sl-recognition-v1-fe
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

# Uncomment when you have a trained model
# import tensorflow as tf
# model = tf.keras.models.load_model('models/gesture_model.h5')

app = FastAPI(title="SignBridge Recognition API")

# Enable CORS for localhost (required for browser to access from XAMPP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gesture labels for ASL and MSL
ASL_LABELS = [
    'Hello', 'Thank You', 'Please', 'Yes', 'No', 
    'Help', 'Sorry', 'Love', 'Friend', 'Family',
    'Good', 'Bad', 'Happy', 'Sad', 'Hungry',
    'Thirsty', 'Tired', 'Sleep', 'Eat', 'Drink'
]

MSL_LABELS = [
    'Helo', 'Terima Kasih', 'Tolong', 'Ya', 'Tidak',
    'Bantuan', 'Maaf', 'Sayang', 'Kawan', 'Keluarga',
    'Baik', 'Buruk', 'Gembira', 'Sedih', 'Lapar',
    'Dahaga', 'Letih', 'Tidur', 'Makan', 'Minum'
]

def preprocess_image(image_bytes):
    """
    Preprocess image for model prediction
    Adjust size and normalization based on your model requirements
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Resize to model input size (adjust based on your model)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    language: str = Form("ASL")
):
    """
    Predict gesture from uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        language: "ASL" or "MSL"
    
    Returns:
        {
            "label": "Hello",
            "confidence": 0.95,
            "language": "ASL"
        }
    """
    try:
        # Read image file
        contents = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(contents)
        
        if processed_image is None:
            return {
                "error": "Failed to process image",
                "label": "Error",
                "confidence": 0.0
            }
        
        # TODO: Replace with actual model prediction
        # predictions = model.predict(processed_image)
        # predicted_class = np.argmax(predictions[0])
        # confidence = float(predictions[0][predicted_class])
        
        # TEMPORARY: Simulated prediction for testing
        # Replace this with actual model inference
        import random
        predicted_class = random.randint(0, 9)  # Random prediction from first 10 gestures
        confidence = random.uniform(0.7, 0.98)  # Random confidence
        
        # Get label based on language
        labels = ASL_LABELS if language == "ASL" else MSL_LABELS
        label = labels[predicted_class] if predicted_class < len(labels) else "Unknown"
        
        return {
            "label": label,
            "confidence": confidence,
            "language": language
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {
            "error": str(e),
            "label": "Error",
            "confidence": 0.0
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "SignBridge Recognition API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": False,  # Set to True when model is loaded
        "supported_languages": ["ASL", "MSL"]
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("SignBridge Recognition API Server")
    print("=" * 60)
    print("Server starting on: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

