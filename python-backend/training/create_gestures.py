"""
Create Gesture Dataset - Capture gestures from camera
Based on: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning
"""

import cv2
import numpy as np
import pickle
import os
import sqlite3
import random

image_x, image_y = 50, 50

def get_hand_hist():
    """Load hand histogram"""
    hist_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hist')
    if os.path.exists(hist_path):
        with open(hist_path, "rb") as f:
            hist = pickle.load(f)
        return hist
    else:
        print("⚠️ Histogram not found. Run set_hand_histogram.py first!")
        return None

def init_create_folder_database():
    """Create gestures folder and database"""
    gestures_dir = os.path.join(os.path.dirname(__file__), '..', 'gestures')
    db_path = os.path.join(os.path.dirname(__file__), '..', 'gesture_db.db')
    
    if not os.path.exists(gestures_dir):
        os.makedirs(gestures_dir)
        print(f"✓ Created gestures directory: {gestures_dir}")
    
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        create_table_cmd = """CREATE TABLE gesture (
            g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            g_name TEXT NOT NULL
        )"""
        conn.execute(create_table_cmd)
        conn.commit()
        conn.close()
        print(f"✓ Created database: {db_path}")

def create_folder(folder_name):
    """Create folder if it doesn't exist"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✓ Created folder: {folder_name}")

def store_in_db(g_id, g_name):
    """Store gesture in database"""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'gesture_db.db')
    conn = sqlite3.connect(db_path)
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, '%s')" % (g_id, g_name)
    try:
        conn.execute(cmd)
        conn.commit()
        print(f"✓ Stored gesture {g_id}: {g_name}")
    except sqlite3.IntegrityError:
        choice = input(f"g_id {g_id} already exists. Update record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = '%s' WHERE g_id = %s" % (g_name, g_id)
            conn.execute(cmd)
            conn.commit()
            print(f"✓ Updated gesture {g_id}: {g_name}")
        else:
            print("✗ Skipping...")
            conn.close()
            return False
    conn.close()
    return True

def store_images(g_id):
    """Capture and store gesture images"""
    total_pics = 1200
    hist = get_hand_hist()
    
    if hist is None:
        print("✗ Cannot proceed without histogram!")
        return False
    
    cam = cv2.VideoCapture(1)
    if cam.read()[0] == False:
        cam = cv2.VideoCapture(0)
    
    x, y, w, h = 300, 100, 300, 300
    gestures_dir = os.path.join(os.path.dirname(__file__), '..', 'gestures', str(g_id))
    create_folder(gestures_dir)
    
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    
    print("=" * 60)
    print("GESTURE CAPTURE")
    print("=" * 60)
    print("Instructions:")
    print("1. Position your hand in the green rectangle")
    print("2. Press 'C' to start/stop capturing")
    print("3. Hold the gesture steady")
    print("4. Captures 1200 images automatically")
    print("=" * 60)
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
            
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y+h, x:x+w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50 and flag_start_capturing:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1:y1+h1, x1:x1+w1]
                
                # Make square
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 
                                                  0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, 
                                                  int((h1-w1)/2), int((h1-w1)/2), 
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                
                save_img = cv2.resize(save_img, (image_x, image_y))
                
                # Random flip for augmentation
                rand = random.randint(0, 10)
                if rand % 2 == 0:
                    save_img = cv2.flip(save_img, 1)
                
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite(os.path.join(gestures_dir, f"{pic_no}.jpg"), save_img)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"Captured: {pic_no}/{total_pics}", (30, 400), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        
        status = "READY - Press 'C' to start" if not flag_start_capturing else "CAPTURING..."
        cv2.putText(img, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Capturing gesture", img)
        cv2.imshow("thresh", thresh)
        
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            frames = 0
            if flag_start_capturing:
                print("▶️ Started capturing...")
            else:
                print("⏸️ Paused capturing...")
        elif keypress == ord('q'):
            break
        
        if flag_start_capturing:
            frames += 1
        
        if pic_no >= total_pics:
            print(f"✓ Captured {total_pics} images!")
            break

    cam.release()
    cv2.destroyAllWindows()
    return pic_no > 0

if __name__ == "__main__":
    init_create_folder_database()
    
    print("\n" + "=" * 60)
    print("GESTURE CREATION")
    print("=" * 60)
    g_id = input("Enter gesture ID (number, e.g., 0, 1, 2...): ")
    g_name = input("Enter gesture name/text (e.g., 'A', 'Hello', 'Yes'): ")
    
    if store_in_db(int(g_id), g_name):
        print(f"\n📸 Starting capture for gesture {g_id}: {g_name}")
        if store_images(int(g_id)):
            print(f"✓ Successfully captured gesture {g_id}: {g_name}")
        else:
            print(f"✗ Failed to capture gesture {g_id}")

