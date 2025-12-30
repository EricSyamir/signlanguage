# SignBridge - Sign Language Recognition Platform

**Python Backend + HTML/JS Frontend**  
Based on: https://github.com/yumdmb/sl-recognition-v1-fe

SignBridge is a comprehensive web application for sign language learning and gesture recognition using a Python FastAPI backend with machine learning capabilities.

![SignBridge Logo](images/MyBIM-Logo-transparent-bg-300x227.png)

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (for backend)
- **XAMPP** (for frontend, optional - can use any HTTP server)
- Modern web browser

### Setup (5 minutes)

#### 1. Start Python Backend

**Windows:**
```bash
# Double-click START_BACKEND.bat
# OR run manually:
cd python-backend
pip install -r requirements.txt
python server.py
```

**Mac/Linux:**
```bash
cd python-backend
pip3 install -r requirements.txt
python3 server.py
```

Server runs on **http://localhost:8000**

#### 2. Start Frontend

**Option A: XAMPP**
```
1. Copy this folder to xampp/htdocs/SignLanguage/
2. Start Apache
3. Visit http://localhost/SignLanguage/index.html
```

**Option B: Simple HTTP Server**
```bash
# Python 3
python -m http.server 8080

# Then visit http://localhost:8080
```

#### 3. Use the App

1. Open the frontend in your browser
2. Go to Recognition page
3. Upload an image or use camera
4. Get real-time AI gesture recognition!

## 🌟 Features

### 👋 Gesture Recognition (Python Backend)
- **Upload-based Recognition**: Upload images for AI analysis
- **Camera-based Recognition**: Real-time gesture capture
- **Continuous Recognition**: Like the original repo - sends frames every 300ms
- **Python FastAPI Backend**: Machine learning powered recognition
- **Multi-language Support**: ASL and MSL

### 📚 Learning Resources
- **9 Interactive Lessons**: Video tutorials with progress tracking
- **Progress Tracking**: Monitor your learning journey
- **Difficulty Levels**: Beginner, Intermediate, Advanced
- **YouTube Integration**: Embedded video tutorials

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **Pillow**: Image processing
- **NumPy**: Numerical operations
- **TensorFlow/PyTorch**: For ML models (optional)

### Frontend
- **HTML5**: Modern semantic markup
- **CSS3**: Custom responsive design
- **Vanilla JavaScript**: No frameworks, lightweight
- **Canvas API**: Image processing
- **getUserMedia API**: Camera integration

## 📁 Project Structure

```
SignLanguage/
├── index.html                 # Homepage
├── recognition.html           # Gesture recognition page
├── learning.html             # Learning resources
├── about.html                # About page
├── START_BACKEND.bat         # Windows backend starter
├── python-backend/
│   ├── server.py             # FastAPI server
│   ├── requirements.txt      # Python dependencies
│   └── README.md             # Backend documentation
├── css/
│   └── style.css             # Main stylesheet
├── js/
│   ├── main.js               # Core utilities
│   ├── recognition.js        # Recognition (connects to Python)
│   └── learning.js           # Learning module
├── images/                   # Image assets
└── data/
    └── gestures.json         # Gesture database
```

## 🎯 How It Works

### Recognition Flow

```
User uploads/captures image
        ↓
JavaScript sends image to Python backend
        ↓
Python FastAPI receives at http://localhost:8000/predict-image
        ↓
Image preprocessed (resize to 224x224, normalize)
        ↓
ML Model predicts gesture
        ↓
Return: { "label": "Hello", "confidence": 0.95, "language": "ASL" }
        ↓
JavaScript displays results with confidence score
```

### Continuous Recognition

In camera mode, the app sends frames to the Python backend every 300ms for real-time recognition, exactly like the original repository.

## 🤖 Adding Your Own ML Model

The current implementation uses simulated predictions. To add a real model:

### TensorFlow Example:

```python
# In python-backend/server.py

import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('models/gesture_model.h5')

# In predict_image function:
predictions = model.predict(processed_image)
predicted_class = np.argmax(predictions[0])
confidence = float(predictions[0][predicted_class])
```

### PyTorch Example:

```python
import torch
import torchvision.transforms as transforms

# Load model
model = torch.load('models/gesture_model.pth')
model.eval()

# Predict
with torch.no_grad():
    output = model(tensor_image)
    _, predicted = torch.max(output.data, 1)
```

## 📖 API Documentation

### POST /predict-image

**Request:**
- `file`: Image file (multipart/form-data)
- `language`: "ASL" or "MSL"

**Response:**
```json
{
  "label": "Hello",
  "confidence": 0.95,
  "language": "ASL"
}
```

### GET /health

Check backend status:
```json
{
  "status": "healthy",
  "model_loaded": false,
  "supported_languages": ["ASL", "MSL"]
}
```

Interactive API docs at: http://localhost:8000/docs

## 🔧 Configuration

### Change Backend Port

In `python-backend/server.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Change 8000 to 8001
```

In `js/recognition.js`:
```javascript
const response = await fetch('http://localhost:8001/predict-image', {  // Update port
```

### Supported Gestures

Edit gesture labels in `python-backend/server.py`:
```python
ASL_LABELS = ['Hello', 'Thank You', 'Please', ...]  # Add more gestures
MSL_LABELS = ['Helo', 'Terima Kasih', 'Tolong', ...]
```

## 🐛 Troubleshooting

### Backend Not Starting

**Issue**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
cd python-backend
pip install -r requirements.txt
```

### CORS Errors

**Issue**: Browser blocks requests to backend

**Solution**: Already configured! The backend allows all origins in development. For production, update CORS settings in `server.py`.

### Camera Not Working

**Issue**: Camera permission denied

**Solution**:
1. Allow camera access in browser settings
2. Use HTTPS or localhost (required for getUserMedia)
3. Close other apps using the camera

### Port Already in Use

**Issue**: `Address already in use: 8000`

**Solution**:
- Close other apps using port 8000
- Or change port (see Configuration above)

## 📊 Gesture Labels

### ASL (American Sign Language)
Hello, Thank You, Please, Yes, No, Help, Sorry, Love, Friend, Family, Good, Bad, Happy, Sad, Hungry, Thirsty, Tired, Sleep, Eat, Drink

### MSL (Malaysian Sign Language)
Helo, Terima Kasih, Tolong, Ya, Tidak, Bantuan, Maaf, Sayang, Kawan, Keluarga, Baik, Buruk, Gembira, Sedih, Lapar, Dahaga, Letih, Tidur, Makan, Minum

## 🔗 Original Repository

This project is based on:
https://github.com/yumdmb/sl-recognition-v1-fe

Key differences:
- Original: Next.js/React with Supabase
- This version: Pure HTML/JS with Python backend
- Same recognition logic and API structure

## 🤝 Contributing

This project is developed in collaboration with:
- **Dr. Anthony Chong**
- **The Malaysian Sign Language and Deaf Studies National Organisation (MyBIM)**

## 📄 License

Developed for educational purposes as part of a Final Year Project.

## 🙏 Acknowledgments

- Original repository authors at https://github.com/yumdmb/sl-recognition-v1-fe
- Dr. Anthony Chong and MyBIM for collaboration
- The deaf and hard-of-hearing community

## 📞 Getting Started Checklist

- [ ] Python 3.8+ installed
- [ ] Run `pip install -r python-backend/requirements.txt`
- [ ] Start backend: `python python-backend/server.py`
- [ ] Backend running on http://localhost:8000
- [ ] Start frontend (XAMPP or HTTP server)
- [ ] Open browser to frontend URL
- [ ] Test recognition with camera or upload
- [ ] See "Backend is running!" success message

---

**Ready to recognize gestures?** Follow the Quick Start above! 🤟

For detailed backend documentation, see: `python-backend/README.md`
