# Training Guide - Sign Language Recognition Model

This guide walks you through training your own CNN model for sign language recognition.

## Prerequisites

- Python 3.8+
- Webcam
- Good lighting
- Patience (training takes time!)

## Step-by-Step Training Process

### Step 1: Set Hand Histogram

The histogram helps segment your hand from the background.

```bash
cd python-backend/training
python set_hand_histogram.py
```

**Instructions:**
1. Place your hand in the green squares area
2. Press **'C'** to capture the histogram
3. Press **'S'** to save and exit

This creates `models/hist` file.

### Step 2: Capture Gestures

Capture training images for each gesture.

```bash
python create_gestures.py
```

**For each gesture:**
1. Enter gesture ID (0, 1, 2, ...)
2. Enter gesture name (e.g., "A", "Hello", "Yes")
3. Position hand in green rectangle
4. Press **'C'** to start capturing
5. Hold gesture steady - captures 1200 images automatically
6. Press **'Q'** to quit when done

**Recommended gestures to capture:**
- **0-9**: Numbers 0 through 9
- **10-35**: Letters A through Z
- **36**: Hello
- **37**: Thank You
- **38**: I Love You
- **39**: Yes
- **40**: No
- **41**: Please
- **42**: Sorry
- **43**: Help

**Tips:**
- Capture in good lighting
- Use consistent background
- Hold gestures steady
- Capture variations (slightly different angles)

### Step 3: Load and Preprocess Images

Prepare images for training.

```bash
python load_images.py
```

This will:
- Load all captured images
- Split into train/validation/test sets (5/6, 1/12, 1/12)
- Save as pickle files

### Step 4: Train the Model

Train the CNN model.

```bash
python cnn_model_train.py
```

**Training details:**
- **Epochs**: 15
- **Batch size**: 500
- **Architecture**: CNN with 3 convolutional blocks
- **Time**: 30-60 minutes (depending on hardware)

The model will be saved to `models/cnn_model_keras2.h5`

## Quick Training Script

Run all steps automatically:

```bash
# Windows
python train_all.py

# Or manually:
python set_hand_histogram.py
python create_gestures.py  # Repeat for each gesture
python load_images.py
python cnn_model_train.py
```

## Expected Results

- **Training accuracy**: >95%
- **Validation accuracy**: >90%
- **Model file**: `models/cnn_model_keras2.h5`

## Troubleshooting

### No histogram found
- Run `set_hand_histogram.py` first

### No images captured
- Check camera permissions
- Ensure good lighting
- Hand must be in green rectangle

### Training fails
- Ensure you have enough images (at least 100 per gesture)
- Check GPU/CPU resources
- Reduce batch size if memory issues

### Low accuracy
- Capture more training images
- Ensure consistent lighting
- Check gesture quality
- Try more epochs

## Using Pre-trained Model

If you have a pre-trained model from the original repository:

1. Copy `cnn_model_keras2.h5` to `python-backend/models/`
2. Copy `hist` to `python-backend/models/`
3. Start the server - it will automatically load the model

## Model Architecture

```
Input: 50x50x1 (grayscale)
  ↓
Conv2D(16, 2x2) → MaxPool(2x2)
  ↓
Conv2D(32, 3x3) → MaxPool(3x3)
  ↓
Conv2D(64, 5x5) → MaxPool(5x5)
  ↓
Flatten → Dense(128) → Dropout(0.2)
  ↓
Output: Softmax(N classes)
```

## Next Steps

After training:
1. Test the model with `python server.py`
2. Use the web interface at `recognition.html`
3. Fine-tune if needed (capture more data, retrain)

---

**Good luck with training!** 🤟

