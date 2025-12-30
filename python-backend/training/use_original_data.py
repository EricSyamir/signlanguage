"""
Use Gesture Data from Original Repository
If you've cloned the original repo, this helps set it up
"""

import os
import shutil
import sys

def find_original_repo():
    """Find the original repository"""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sl-training'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Sign-Language-Interpreter-using-Deep-Learning'),
        '../sl-training',
        '../../sl-training',
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        code_dir = os.path.join(abs_path, 'Code')
        if os.path.exists(code_dir):
            return abs_path, code_dir
    
    return None, None

def copy_gesture_data(original_code_dir, target_dir):
    """Copy gesture data from original repo"""
    gestures_src = os.path.join(original_code_dir, 'gestures')
    hist_src = os.path.join(original_code_dir, 'hist')
    db_src = os.path.join(original_code_dir, 'gesture_db.db')
    
    gestures_dst = os.path.join(target_dir, '..', 'gestures')
    models_dst = os.path.join(target_dir, '..', 'models')
    db_dst = os.path.join(target_dir, '..', 'gesture_db.db')
    
    copied = []
    
    # Copy gestures folder
    if os.path.exists(gestures_src):
        if os.path.exists(gestures_dst):
            shutil.rmtree(gestures_dst)
        shutil.copytree(gestures_src, gestures_dst)
        copied.append('gestures')
        print(f"✓ Copied gestures from: {gestures_src}")
    
    # Copy histogram
    if os.path.exists(hist_src):
        os.makedirs(models_dst, exist_ok=True)
        shutil.copy(hist_src, os.path.join(models_dst, 'hist'))
        copied.append('histogram')
        print(f"✓ Copied histogram from: {hist_src}")
    
    # Copy database
    if os.path.exists(db_src):
        shutil.copy(db_src, db_dst)
        copied.append('database')
        print(f"✓ Copied database from: {db_src}")
    
    return copied

def main():
    print("=" * 60)
    print("USE ORIGINAL REPOSITORY DATA")
    print("=" * 60)
    
    repo_path, code_dir = find_original_repo()
    
    if not repo_path:
        print("\n✗ Original repository not found!")
        print("\nTo use this script:")
        print("1. Clone the original repository:")
        print("   git clone https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning.git")
        print("2. Place it in the same directory as SignLanguage folder")
        print("3. Run this script again")
        print("\nOr train your own model using:")
        print("  python train_all.py")
        return
    
    print(f"\n✓ Found original repository at: {repo_path}")
    
    response = input("\nCopy gesture data from original repo? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    copied = copy_gesture_data(code_dir, os.path.dirname(__file__))
    
    if copied:
        print("\n" + "=" * 60)
        print("✓ DATA COPIED SUCCESSFULLY!")
        print("=" * 60)
        print("\nCopied items:")
        for item in copied:
            print(f"  - {item}")
        
        print("\nNext steps:")
        print("1. Run: python load_images.py")
        print("2. Run: python cnn_model_train.py")
        print("\nOr use the automated script:")
        print("  python train_all.py")
    else:
        print("\n✗ No data found to copy!")
        print("The original repository may not have gesture data yet.")
        print("You'll need to capture gestures yourself using:")
        print("  python create_gestures.py")

if __name__ == "__main__":
    main()

