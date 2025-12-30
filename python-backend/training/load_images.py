"""
Load and Preprocess Images for Training
Based on: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning
"""

import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

def pickle_images_labels():
    """Load all gesture images and labels"""
    gestures_dir = os.path.join(os.path.dirname(__file__), '..', 'gestures')
    images = glob(os.path.join(gestures_dir, "*", "*.jpg"))
    images.sort()
    
    images_labels = []
    print("Loading images...")
    for image in images:
        # Extract label from folder name
        label = os.path.basename(os.path.dirname(image))
        img = cv2.imread(image, 0)  # Read as grayscale
        if img is not None:
            images_labels.append((np.array(img, dtype=np.uint8), int(label)))
    
    return images_labels

def main():
    print("=" * 60)
    print("LOADING IMAGES FOR TRAINING")
    print("=" * 60)
    
    images_labels = pickle_images_labels()
    
    if len(images_labels) == 0:
        print("✗ No images found! Please capture gestures first.")
        return
    
    print(f"✓ Loaded {len(images_labels)} images")
    
    # Shuffle multiple times for better randomization
    images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
    images, labels = zip(*images_labels)
    
    print(f"Total images: {len(images_labels)}")
    
    # Split: 5/6 train, 1/12 validation, 1/12 test
    train_split = int(5/6 * len(images))
    val_split = int(11/12 * len(images))
    
    # Training set
    train_images = images[:train_split]
    train_labels = labels[:train_split]
    print(f"Training images: {len(train_images)}")
    
    # Validation set
    val_images = images[train_split:val_split]
    val_labels = labels[train_split:val_split]
    print(f"Validation images: {len(val_images)}")
    
    # Test set
    test_images = images[val_split:]
    test_labels = labels[val_split:]
    print(f"Test images: {len(test_images)}")
    
    # Save to pickle files
    output_dir = os.path.dirname(__file__)
    
    print("\nSaving datasets...")
    with open(os.path.join(output_dir, "train_images"), "wb") as f:
        pickle.dump(train_images, f)
    print("✓ Saved train_images")
    
    with open(os.path.join(output_dir, "train_labels"), "wb") as f:
        pickle.dump(train_labels, f)
    print("✓ Saved train_labels")
    
    with open(os.path.join(output_dir, "val_images"), "wb") as f:
        pickle.dump(val_images, f)
    print("✓ Saved val_images")
    
    with open(os.path.join(output_dir, "val_labels"), "wb") as f:
        pickle.dump(val_labels, f)
    print("✓ Saved val_labels")
    
    with open(os.path.join(output_dir, "test_images"), "wb") as f:
        pickle.dump(test_images, f)
    print("✓ Saved test_images")
    
    with open(os.path.join(output_dir, "test_labels"), "wb") as f:
        pickle.dump(test_labels, f)
    print("✓ Saved test_labels")
    
    print("\n" + "=" * 60)
    print("✓ All datasets saved successfully!")
    print("=" * 60)
    print("\nNext step: Run cnn_model_train.py to train the model")

if __name__ == "__main__":
    main()

