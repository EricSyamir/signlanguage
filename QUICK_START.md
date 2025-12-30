# 🚀 Quick Start Guide

## What You Have Now

✅ **Python FastAPI Backend** - Exactly like the original GitHub repo  
✅ **HTML/JS Frontend** - Connects to Python backend  
✅ **Real-time Recognition** - Sends frames every 300ms like original repo  
✅ **Upload Recognition** - Upload images for AI analysis  
✅ **Continuous Recognition** - Camera-based continuous gesture recognition  

## Start in 2 Steps

### Step 1: Start Python Backend

**Windows:**
```bash
# Easy way:
Double-click START_BACKEND.bat

# Or manual:
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

You should see:
```
============================================================
SignBridge Recognition API Server
============================================================
Server starting on: http://localhost:8000
API Documentation: http://localhost:8000/docs
============================================================
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Open Frontend

**Option A: XAMPP** (if this folder is in htdocs)
```
1. Start Apache in XAMPP
2. Visit: http://localhost/SignLanguage/index.html
```

**Option B: Python HTTP Server** (recommended)
```bash
# In project root (not python-backend folder)
python -m http.server 8080
# Visit: http://localhost:8080
```

**Option C: Just open the HTML file**
```
Open index.html in your browser
```

## Test It

1. Go to Recognition page
2. Click "Use Camera"
3. Click "Start Camera"
4. Click "Start Continuous Recognition"
5. Make gestures and see real-time predictions!

## Recognition Flow (Like Original Repo)

```
User's Camera
    ↓
JavaScript captures frames (every 300ms)
    ↓
Send to: http://localhost:8000/predict-image
    ↓
Python FastAPI processes image
    ↓
ML Model predicts gesture
    ↓
Return: {"label": "Hello", "confidence": 0.95}
    ↓
Display result in browser
```

## Add Your Own Model

The current backend uses simulated predictions for testing. To use a real ML model:

1. Train your model or get a pre-trained one
2. Save as `python-backend/models/gesture_model.h5` (TensorFlow) or `.pth` (PyTorch)
3. Update `python-backend/server.py`:

```python
# Uncomment and modify:
import tensorflow as tf
model = tf.keras.models.load_model('models/gesture_model.h5')

# In predict_image function:
predictions = model.predict(processed_image)
predicted_class = np.argmax(predictions[0])
confidence = float(predictions[0][predicted_class])
```

## API Endpoints

- `GET /` - Status check
- `GET /health` - Health check with details
- `POST /predict-image` - Gesture recognition
  - Body: `file` (image), `language` ("ASL" or "MSL")
  - Returns: `{"label": "Hello", "confidence": 0.95, "language": "ASL"}`

Full API docs at: http://localhost:8000/docs

## Troubleshooting

**Backend won't start?**
```bash
pip install fastapi uvicorn python-multipart Pillow numpy
```

**Frontend can't connect?**
- Make sure backend is running on port 8000
- Check browser console for errors
- Visit http://localhost:8000/health to verify backend

**Camera not working?**
- Allow camera permissions in browser
- Use HTTPS or localhost (required for getUserMedia API)

## Files Structure

```
SignLanguage/
├── index.html                    # Homepage
├── recognition.html              # Recognition page
├── learning.html                 # Learning resources
├── about.html                    # About page
├── START_BACKEND.bat             # Windows backend starter
├── python-backend/
│   ├── server.py                 # FastAPI server (MAIN LOGIC)
│   ├── requirements.txt          # Python dependencies
│   └── README.md                 # Backend docs
├── css/
│   └── style.css                 # Styling
├── js/
│   └── recognition.js            # Recognition logic (connects to Python)
└── README.md                     # Full documentation
```

## What Makes This Like the Original Repo?

✅ Python FastAPI backend on port 8000  
✅ Sends frames every 300ms for continuous recognition  
✅ Same API structure: `/predict-image` endpoint  
✅ FormData with `file` and `language` parameters  
✅ Returns `{label, confidence, language}` format  
✅ Client-side camera capture with canvas  
✅ Real-time predictions displayed  

The ONLY difference: Original uses Next.js/React frontend, this uses vanilla HTML/JS.  
The recognition logic and backend are IDENTICAL in structure!

## Need Help?

See:
- `README.md` - Full documentation
- `python-backend/README.md` - Backend details
- http://localhost:8000/docs - Interactive API docs (when server is running)

---

**Ready to go? Just run `START_BACKEND.bat` and open `index.html`!** 🤟

