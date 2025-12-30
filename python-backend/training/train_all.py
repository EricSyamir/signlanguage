"""
Complete Training Pipeline
Runs all training steps in sequence
"""

import os
import sys
import subprocess

def run_script(script_name, description):
    """Run a Python script"""
    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"✗ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_path], check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("SIGN LANGUAGE MODEL TRAINING PIPELINE")
    print("=" * 60)
    print("\nThis will guide you through:")
    print("1. Setting hand histogram")
    print("2. Capturing gestures (manual step)")
    print("3. Loading and preprocessing images")
    print("4. Training the CNN model")
    print("\n" + "=" * 60)
    
    input("Press Enter to start...")
    
    # Step 1: Hand Histogram
    if not run_script("set_hand_histogram.py", "STEP 1: Set Hand Histogram"):
        print("\n✗ Failed to set histogram. Please run manually.")
        return
    
    # Step 2: Capture Gestures (manual)
    print("\n" + "=" * 60)
    print("STEP 2: Capture Gestures")
    print("=" * 60)
    print("You need to manually capture gestures.")
    print("Run: python create_gestures.py")
    print("\nFor each gesture:")
    print("- Enter gesture ID (0-43)")
    print("- Enter gesture name")
    print("- Press 'C' to start capturing")
    print("- Hold gesture steady")
    print("- Captures 1200 images automatically")
    print("\nRepeat for all gestures you want to recognize.")
    
    response = input("\nHave you captured all gestures? (y/n): ")
    if response.lower() != 'y':
        print("Please capture gestures first, then run this script again.")
        return
    
    # Step 3: Load Images
    if not run_script("load_images.py", "STEP 3: Load and Preprocess Images"):
        print("\n✗ Failed to load images. Please check if gestures were captured.")
        return
    
    # Step 4: Train Model
    if not run_script("cnn_model_train.py", "STEP 4: Train CNN Model"):
        print("\n✗ Training failed. Check error messages above.")
        return
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nYour model is saved at: python-backend/models/cnn_model_keras2.h5")
    print("\nNext steps:")
    print("1. Start the server: python python-backend/server.py")
    print("2. Open recognition.html in your browser")
    print("3. Start interpreting sign language!")

if __name__ == "__main__":
    main()

