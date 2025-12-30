"""
Download Pre-trained Model (if available)
Helper script to check for pre-trained models
"""

import os
import urllib.request
import sys

def check_model_exists():
    """Check if model already exists"""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, 'cnn_model_keras2.h5')
    
    if os.path.exists(model_path):
        print(f"✓ Model already exists at: {model_path}")
        return True
    return False

def main():
    print("=" * 60)
    print("PRE-TRAINED MODEL CHECKER")
    print("=" * 60)
    
    if check_model_exists():
        print("\nYou already have a trained model!")
        print("You can skip training and use the existing model.")
        return
    
    print("\nNo pre-trained model found.")
    print("\nTo train your own model:")
    print("1. Run: python set_hand_histogram.py")
    print("2. Run: python create_gestures.py (for each gesture)")
    print("3. Run: python load_images.py")
    print("4. Run: python cnn_model_train.py")
    print("\nOr use the automated script:")
    print("  python train_all.py")
    print("\nFor detailed instructions, see TRAINING_GUIDE.md")

if __name__ == "__main__":
    main()

