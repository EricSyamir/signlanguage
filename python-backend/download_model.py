"""
Helper script to download model file for Render deployment
Run this in Render Shell after deployment
"""

import os
import urllib.request

def download_model(model_url, output_path):
    """Download model from URL"""
    try:
        print(f"Downloading model from {model_url}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        urllib.request.urlretrieve(model_url, output_path)
        print(f"✅ Model downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    # Set your model URL here (if hosting model online)
    MODEL_URL = os.environ.get('MODEL_URL', '')
    
    if not MODEL_URL:
        print("⚠️ No MODEL_URL environment variable set.")
        print("Set MODEL_URL in Render environment variables, or upload manually.")
        print("\nTo upload manually:")
        print("1. Go to Render Shell")
        print("2. Run: mkdir -p python-backend/models")
        print("3. Upload sign_language_cnn_model.h5 to python-backend/models/")
        exit(1)
    
    # Download to models directory
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'models', 'sign_language_cnn_model.h5')
    
    if download_model(MODEL_URL, model_path):
        print("\n✅ Model ready! Restart your service.")
    else:
        print("\n❌ Failed to download model. Please upload manually.")

