# SignBridge - Sign Language Interpreter

**Real-time sign language to text & speech interpretation using Deep Learning**

Based on: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)

## 🎯 What This Does

This web application interprets American Sign Language (ASL) gestures in real-time:
1. **Captures** gestures via webcam
2. **Recognizes** letters, numbers, and common words using a trained CNN
3. **Builds sentences** from continuous gesture recognition
4. **Speaks** the interpreted sentence using text-to-speech

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser with camera access

### 1. Install Dependencies

```bash
cd python-backend
pip install -r requirements.txt
```

### 2. Start Backend Server

```bash
# Windows
python server.py

# Or double-click START_BACKEND.bat
```

Server starts at: http://localhost:8000

### 3. Open Frontend

**Option A: XAMPP**
```
Place folder in xampp/htdocs/
Visit: http://localhost/SignLanguage/
```

**Option B: Python HTTP Server**
```bash
python -m http.server 8080
# Visit: http://localhost:8080
```

**Option C: Direct**
```
Open recognition.html in browser
```

### 4. Start Interpreting!

1. Click "Start Interpreter"
2. Allow camera access
3. Make sign language gestures
4. Watch sentences build in real-time!

## 📊 Supported Gestures (24 Classes)

| Type | Gestures |
|------|----------|
| Letters | A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y |

**Note:** Sign Language MNIST dataset includes A-Z excluding J and Z (24 classes total)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   BROWSER (HTML/JS)                          │
│  - Camera capture via getUserMedia                          │
│  - Sends frames every 200ms                                  │
│  - Displays predictions & builds sentences                   │
│  - Text-to-speech output                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP POST /predict-and-build
                         │ FormData: {file: blob, language: ASL}
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              PYTHON BACKEND (FastAPI)                        │
│  - Image preprocessing (histogram backprojection)           │
│  - Hand segmentation & contour detection                    │
│  - CNN model prediction (TensorFlow/Keras)                  │
│  - Sentence builder (accumulates characters)                │
│  - Returns: {label, confidence, sentence}                   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
SignLanguage/
├── index.html                    # Homepage
├── recognition.html              # Main interpreter UI
├── learning.html                 # Learning resources
├── about.html                    # About page
├── START_BACKEND.bat             # Windows launcher
├── python-backend/
│   ├── server.py                 # FastAPI server (main logic)
│   ├── requirements.txt          # Python dependencies
│   ├── models/
│   │   ├── cnn_model_keras2.h5   # Trained model (add this)
│   │   └── hist                  # Hand histogram (optional)
│   └── README.md
├── css/
│   └── style.css
├── js/
│   └── recognition.js            # Frontend recognition logic
└── README.md
```

## 🧠 The CNN Model

The recognition uses a Convolutional Neural Network trained on Sign Language MNIST:

**Architecture:**
```
Input: 28x28x1 (grayscale)
  ↓
Conv2D(75, 3x3) → BatchNorm → MaxPool(2x2)
  ↓
Conv2D(50, 3x3) → Dropout(0.2) → BatchNorm → MaxPool(2x2)
  ↓
Conv2D(25, 3x3) → BatchNorm → MaxPool(2x2)
  ↓
Flatten → Dense(512) → Dropout(0.3)
  ↓
Output: Softmax(24 classes) → >95% accuracy
```

**Supported Gestures:** A-Z (excluding J and Z) - 24 letter classes

## 🎓 Training Your Own Model

**Complete training pipeline included!**

### Quick Start Training:

```bash
cd python-backend/training

# Windows - Easy way:
START_TRAINING.bat

# Or manually:
python train_all.py
```

### Step-by-Step Training:

1. **Set Hand Histogram** (one-time setup):
   ```bash
   python set_hand_histogram.py
   ```
   - Place hand in green squares
   - Press 'C' to capture
   - Press 'S' to save

2. **Capture Gestures** (for each gesture):
   ```bash
   python create_gestures.py
   ```
   - Enter gesture ID (0-43)
   - Enter gesture name (e.g., "A", "Hello")
   - Press 'C' to start capturing
   - Captures 1200 images automatically

3. **Load Images**:
   ```bash
   python load_images.py
   ```
   - Prepares train/validation/test sets

4. **Train Model**:
   ```bash
   python cnn_model_train.py
   ```
   - Trains CNN (takes 30-60 minutes)
   - Saves to `models/cnn_model_keras2.h5`

**See `python-backend/training/TRAINING_GUIDE.md` for detailed instructions.**

### Using Pre-trained Model:

If you have a trained model from the original repository:

1. Copy `cnn_model_keras2.h5` to `python-backend/models/`
2. Copy `hist` to `python-backend/models/` (optional)
3. Start the server - it will automatically load the model

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server status |
| `/health` | GET | Health check with model info |
| `/predict-image` | POST | Single gesture prediction |
| `/predict-and-build` | POST | Predict + add to sentence |
| `/sentence` | GET | Get current sentence |
| `/sentence/space` | POST | Add space (complete word) |
| `/sentence/backspace` | POST | Remove last char/word |
| `/sentence/clear` | POST | Clear sentence |
| `/gestures` | GET | List supported gestures |
| `/ws/recognize` | WebSocket | Real-time recognition stream |

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict-image \
  -F "file=@gesture.jpg" \
  -F "language=ASL"
```

**Example Response:**
```json
{
  "label": "Hello",
  "confidence": 0.95,
  "language": "ASL",
  "timestamp": "2025-01-01T00:00:00"
}
```

## 🎮 How Sentence Building Works

1. **Gesture Recognition**: Each frame is analyzed by the CNN
2. **Confidence Threshold**: Only predictions >70% are considered
3. **Frame Confirmation**: Same gesture must be held for ~15 frames (~3 seconds)
4. **Character Added**: Confirmed character added to current word
5. **Space Gesture**: Completes current word, starts new one
6. **Sentence Complete**: Full sentence ready for text-to-speech

## 🐛 Troubleshooting

### Backend won't start
```bash
# Ensure you're in python-backend directory
cd python-backend
pip install -r requirements.txt
python server.py
```

### No predictions / always "Error"
- Check if model file exists: `python-backend/models/cnn_model_keras2.h5`
- Without a model, predictions are simulated for testing

### Camera not working
- Allow camera permissions in browser
- Use HTTPS or localhost (required for getUserMedia)
- Close other apps using the camera

### CORS errors
- Backend already has CORS enabled for all origins
- Ensure backend is running on port 8000

## 📖 Original Repository

This project is based on:
**[Sign-Language-Interpreter-using-Deep-Learning](https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning)**

By Harsh Gupta, Siddharth Oza, Ashish Sharma, and Manish Shukla

Created at HackUNT-19, Winner of UNT Hackathon 2019

## 🙏 Acknowledgments

- [harshbg](https://github.com/harshbg) for the original interpreter
- TensorFlow/Keras for deep learning framework
- OpenCV for image processing
- FastAPI for modern Python web framework

## 📄 License

MIT License - See original repository for details

---

**Ready to interpret sign language?** Run `python python-backend/server.py` and open `recognition.html`! 🤟
