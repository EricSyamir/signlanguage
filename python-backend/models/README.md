# Model Files

Place your trained model files here.

## Required Files

### Option 1: Use the model from the original repository

1. Clone the original repo:
   ```bash
   git clone https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning.git
   ```

2. Follow their instructions to train the model:
   - Run `set_hand_histogram.py` to create hand histogram
   - Run `create_gestures.py` to capture gestures
   - Run `load_images.py` to prepare training data
   - Run `cnn_model_train.py` to train the model

3. Copy the trained files to this folder:
   - `cnn_model_keras2.h5` - The trained CNN model
   - `hist` - The hand histogram file (optional, for better hand segmentation)

### Option 2: Use a pre-trained model

If you have a pre-trained ASL gesture recognition model, place it here as:
- `cnn_model_keras2.h5` (Keras/TensorFlow format)

### Model Architecture

The expected model architecture (from original repo):
- Input: Grayscale image (50x50x1)
- Conv2D(16, 2x2) → MaxPool(2x2)
- Conv2D(32, 3x3) → MaxPool(3x3)
- Conv2D(64, 5x5) → MaxPool(5x5)
- Flatten → Dense(128) → Dropout(0.2)
- Output: Softmax(44 classes)

### Without a Model

If no model is present, the server will use simulated predictions for testing purposes.

## File List

After setup, this folder should contain:
- `cnn_model_keras2.h5` - Trained Keras model
- `hist` - Hand histogram (optional)
- `README.md` - This file

## Training Your Own Model

See the original repository for training instructions:
https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning

The training process requires:
1. Capturing gesture images using webcam
2. Labeling gestures in SQLite database
3. Preprocessing and augmenting images
4. Training CNN with Keras
5. Saving model as .h5 file

