"""
Train CNN Model for Sign Language Recognition
Based on: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning
"""

import numpy as np
import pickle
import cv2
import os
from glob import glob
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def get_image_size():
    """Get image dimensions from first gesture"""
    gestures_dir = os.path.join(os.path.dirname(__file__), '..', 'gestures')
    sample_img = glob(os.path.join(gestures_dir, "*", "*.jpg"))[0]
    img = cv2.imread(sample_img, 0)
    return img.shape

def get_num_of_classes():
    """Get number of gesture classes"""
    gestures_dir = os.path.join(os.path.dirname(__file__), '..', 'gestures')
    return len(glob(os.path.join(gestures_dir, "*")))

def cnn_model():
    """Define CNN architecture"""
    num_of_classes = get_num_of_classes()
    image_x, image_y = get_image_size()
    
    print(f"Image size: {image_x}x{image_y}")
    print(f"Number of classes: {num_of_classes}")
    
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Second convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    
    # Third convolutional block
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    
    # Compile model
    sgd = optimizers.SGD(learning_rate=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def train():
    """Train the CNN model"""
    training_dir = os.path.dirname(__file__)
    
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)
    
    # Load training data
    with open(os.path.join(training_dir, "train_images"), "rb") as f:
        train_images = np.array(pickle.load(f))
    with open(os.path.join(training_dir, "train_labels"), "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)
    
    # Load validation data
    with open(os.path.join(training_dir, "val_images"), "rb") as f:
        val_images = np.array(pickle.load(f))
    with open(os.path.join(training_dir, "val_labels"), "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Get image dimensions
    image_x, image_y = get_image_size()
    
    # Reshape images for CNN input
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    
    # Normalize pixel values
    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    
    # Convert labels to categorical
    num_classes = get_num_of_classes()
    train_labels = to_categorical(train_labels, num_classes)
    val_labels = to_categorical(val_labels, num_classes)
    
    print(f"Image shape: {train_images.shape[1:]}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print("\n" + "=" * 60)
    print("CREATING CNN MODEL")
    print("=" * 60)
    model = cnn_model()
    model.summary()
    
    # Setup model checkpoint
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "cnn_model_keras2.h5")
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    print("This may take a while...")
    print("=" * 60)
    
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=15,
        batch_size=500,
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    scores = model.evaluate(val_images, val_labels, verbose=0)
    accuracy = scores[1] * 100
    error = 100 - accuracy
    
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Validation Error: {error:.2f}%")
    
    # Save final model
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model accuracy: {accuracy:.2f}%")
    print(f"Model location: {model_path}")
    print("\nYou can now use this model in the server!")

if __name__ == "__main__":
    # Check if training data exists
    training_dir = os.path.dirname(__file__)
    required_files = ["train_images", "train_labels", "val_images", "val_labels"]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(training_dir, f))]
    
    if missing_files:
        print("✗ Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run load_images.py first!")
    else:
        train()

