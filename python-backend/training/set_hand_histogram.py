"""
Set Hand Histogram for Hand Segmentation
Based on: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning
"""

import cv2
import numpy as np
import pickle
import os

def build_squares(img):
    """Build squares for histogram sampling"""
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if imgCrop is None:
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop)) 
        imgCrop = None
        x = 420
        y += h + d
    return crop

def get_hand_hist():
    """Capture hand histogram from camera"""
    cam = cv2.VideoCapture(1)
    if cam.read()[0] == False:
        cam = cv2.VideoCapture(0)
    
    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    
    print("=" * 60)
    print("HAND HISTOGRAM SETUP")
    print("=" * 60)
    print("Instructions:")
    print("1. Place your hand in the green squares area")
    print("2. Press 'C' to capture histogram")
    print("3. Press 'S' to save and exit")
    print("=" * 60)
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
            
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        keypress = cv2.waitKey(1) & 0xFF
        
        if keypress == ord('c'):
            if imgCrop is not None:
                hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                flagPressedC = True
                hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                print("✓ Histogram captured! Press 'S' to save.")
        elif keypress == ord('s'):
            if flagPressedC:
                flagPressedS = True
                break
            else:
                print("Please capture histogram first (press 'C')")
        
        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            dst1 = dst.copy()
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresh", thresh)
        
        if not flagPressedS:
            imgCrop = build_squares(img)
        
        cv2.putText(img, "Press 'C' to capture, 'S' to save", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Set hand histogram", img)
    
    cam.release()
    cv2.destroyAllWindows()
    
    if flagPressedC:
        # Save histogram
        hist_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hist')
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        
        with open(hist_path, "wb") as f:
            pickle.dump(hist, f)
        print(f"✓ Histogram saved to: {hist_path}")
        return True
    else:
        print("✗ Histogram not saved. Please capture it first.")
        return False

if __name__ == "__main__":
    get_hand_hist()

