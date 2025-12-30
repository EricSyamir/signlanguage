# SignBridge Python Backend

This is the Python FastAPI backend for gesture recognition, based on the original repository:
https://github.com/yumdmb/sl-recognition-v1-fe

## Quick Start

### 1. Install Python Dependencies

```bash
cd python-backend
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python server.py
```

The server will start on **http://localhost:8000**

### 3. Test the API

Open your browser and visit:
- http://localhost:8000 - Basic status
- http://localhost:8000/docs - Interactive API documentation
- http://localhost:8000/health - Health check

## API Endpoints

### POST /predict-image

Predict gesture from an uploaded image.

**Request:**
- `file`: Image file (multipart/form-data)
- `language`: "ASL" or "MSL" (form data)

**Response:**
```json
{
  "label": "Hello",
  "confidence": 0.95,
  "language": "ASL"
}
```

### GET /health

Check server health and status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": false,
  "supported_languages": ["ASL", "MSL"]
}
```

## Adding Your Own Model

The current implementation uses simulated predictions for testing. To use a real ML model:

### Option 1: TensorFlow/Keras

1. Train your model or download a pre-trained model
2. Save it as `models/gesture_model.h5`
3. Uncomment TensorFlow imports in `server.py`:
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('models/gesture_model.h5')
   ```
4. Update the prediction code to use your model
5. Update `requirements.txt` to include `tensorflow`

### Option 2: PyTorch

1. Train your model or download a pre-trained model
2. Save it as `models/gesture_model.pth`
3. Add PyTorch imports and model loading code
4. Update the prediction code
5. Update `requirements.txt` to include `torch` and `torchvision`

## Model Requirements

Your model should:
- Accept images of size 224x224 (or adjust `preprocess_image()`)
- Return predictions for gesture classes
- Support both ASL and MSL gestures

## Gesture Labels

### ASL (American Sign Language)
```python
['Hello', 'Thank You', 'Please', 'Yes', 'No', 
 'Help', 'Sorry', 'Love', 'Friend', 'Family',
 'Good', 'Bad', 'Happy', 'Sad', 'Hungry',
 'Thirsty', 'Tired', 'Sleep', 'Eat', 'Drink']
```

### MSL (Malaysian Sign Language)
```python
['Helo', 'Terima Kasih', 'Tolong', 'Ya', 'Tidak',
 'Bantuan', 'Maaf', 'Sayang', 'Kawan', 'Keluarga',
 'Baik', 'Buruk', 'Gembira', 'Sedih', 'Lapar',
 'Dahaga', 'Letih', 'Tidur', 'Makan', 'Minum']
```

## Troubleshooting

### Port 8000 Already in Use

Change the port in `server.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
```

Then update the frontend JavaScript to use port 8001.

### CORS Errors

The server is configured to allow all origins during development. For production, update the CORS middleware in `server.py` to specify exact origins.

### Image Processing Errors

Ensure Pillow is installed correctly:
```bash
pip install --upgrade Pillow
```

## Development

### Run with Auto-Reload

```bash
uvicorn server:app --reload --port 8000
```

### View API Documentation

FastAPI automatically generates interactive docs at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## Production Deployment

For production deployment:

1. Use a production ASGI server
2. Configure specific CORS origins
3. Add authentication if needed
4. Use HTTPS
5. Add rate limiting
6. Monitor with logging

## Credits

Based on the original SignBridge repository:
https://github.com/yumdmb/sl-recognition-v1-fe

Developed in collaboration with Dr. Anthony Chong and MyBIM.

