import socket
import numpy as np
import cv2
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from collections import Counter

# We load models from the parent 'new blood' directory where they reside
BLOOD_MODEL_PATH = 'blood_model.h5'
GENDER_MODEL_PATH = 'gender_model.h5'

BLOOD_GROUPS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
# Logic for Gender based on Jupyter notebook: 0 = Male, 1 = Female
GENDER_LABELS = ['Male', 'Female']

blood_model = None
gender_model = None

def load_ai_models():
    global blood_model, gender_model
    if blood_model is None:
        try:
            blood_model = load_model(BLOOD_MODEL_PATH, compile=False)
            gender_model = load_model(GENDER_MODEL_PATH, compile=False)
            print("AI Models Loaded Successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")

# ======================= DATA EXTRACTION =======================
  
def extract_r307_image(raw_data):
    if len(raw_data) < 12: return raw_data
    offset = 0
    if raw_data[0] == 0xEF and raw_data[1] == 0x01:
        offset = 12
    image_data = bytearray()
    while offset + 9 <= len(raw_data):
        header = raw_data[offset:offset+9]
        if header[0] != 0xEF or header[1] != 0x01:
            offset += 1
            continue
        pid = header[6] 
        length = (header[7] << 8) | header[8]
        offset += 9
        if offset + length > len(raw_data): break
        data = raw_data[offset:offset+length]
        offset += length
        if pid == 0x02 or pid == 0x08:
            image_data.extend(data[:-2])
        if pid == 0x08: break
    if not image_data:
        return raw_data[12:] if len(raw_data) > 12 else raw_data
    return bytes(image_data)

def process_raw_image(raw_data, width=256, height=288):
    if not raw_data: return None
    expected_size = (width * height) // 2
    if len(raw_data) < expected_size:
        raw_data = raw_data + bytes(expected_size - len(raw_data))
    pixels = []
    for byte in raw_data[:expected_size]:
        high_nibble = (byte >> 4) & 0x0F
        low_nibble = byte & 0x0F
        pixels.append(high_nibble * 17)
        pixels.append(low_nibble * 17)
    img_array = np.array(pixels, dtype=np.uint8)
    return img_array.reshape((height, width))

def auto_crop_fingerprint(img):
    """
    Detects and crops the fingerprint region from R307 sensor images.
    Removes the large black border, keeping only the actual ridges.
    """
    if img is None: return img
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return img
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img.shape[1] - x, w + 2 * pad_x)
    h = min(img.shape[0] - y, h + 2 * pad_y)
    
    if w > 20 and h > 20:
        return img[y:y+h, x:x+w]
    return img

# ======================= PREPROCESSING PIPELINES =======================

def preprocess_blood(img):
    if img is None: return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    return img.reshape(1, 64, 64, 3)

def predict_blood_group(preprocessed_img):
    if preprocessed_img is None or blood_model is None:
        return None, 0.0
    predictions = blood_model.predict(preprocessed_img, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    return BLOOD_GROUPS[predicted_class], confidence

def preprocess_gender(img):
    if img is None: return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, (96, 96))
    img = img.astype(np.float32) / 255.0
    return img.reshape(1, 96, 96, 1)

def predict_gender(preprocessed_img):
    if preprocessed_img is None or gender_model is None:
        return None, 0.0
    prediction = gender_model.predict(preprocessed_img, verbose=0)
    prob = prediction[0][0]
    
    # 1 is Female, 0 is Male
    if prob >= 0.5:
        predicted_class = 1 
        confidence = prob * 100
    else:
        predicted_class = 0 
        confidence = (1 - prob) * 100
    return GENDER_LABELS[predicted_class], confidence

# ======================= SIFT BIOMETRICS =======================

def extract_features(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_fingerprints(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    # Ratio test
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return len(good_matches)

def compare_raw_image_to_descriptors(raw_img, user_descriptors):
    """
    raw_img: numpy array from ESP32
    user_descriptors: dictionary mapping user_id to SIFT descriptor numpy array
    returns: matched_user_id or None
    """
    if raw_img is None: return None
    
    # We explicitly crop the raw image for pure biometric feature extraction 
    # to match the flawlessly cropped images we saved during registration
    cropped_img = auto_crop_fingerprint(raw_img)
    _, current_desc = extract_features(cropped_img)
    if current_desc is None: return None
    
    best_match = None
    max_good_points = 0
    
    for user_id, saved_desc in user_descriptors.items():
        if saved_desc is not None:
            score = match_fingerprints(current_desc, saved_desc)
            if score > max_good_points:
                max_good_points = score
                best_match = user_id
                
    # Threshold for a secure fingerprint match using SIFT
    if max_good_points > 15:
        return best_match
    return None

# ======================= HARDWARE CONTROLLER =======================

def poll_esp32_single_scan(ip_address, port=8080, custom_msg="Scanning..."):
    """
    Connects to ESP32 once, reads 1 image, returns image.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((ip_address, port))
        
        s.settimeout(None)
        first_chunk = s.recv(4096)
        if not first_chunk:
            s.close()
            return None
            
        raw_buffer = bytearray(first_chunk)
        s.settimeout(2.5)
        
        while True:
            try:
                chunk = s.recv(4096)
                if not chunk: break
                raw_buffer.extend(chunk)
            except socket.timeout:
                break
                
        image_data = extract_r307_image(raw_buffer)
        img = process_raw_image(image_data)
        
        # Send Custom Msg to display on hardware OLED
        s.sendall((f"{custom_msg}\n").encode())
        
        s.close()
        time.sleep(0.5)
        return img
    except Exception as e:
        print(f"ESP32 Polling failed: {e}")
        return None

def execute_unified_voting(ip_address, scans=3):
    """
    Polls the ESP32 `scans` times, computing Blood and Gender parallel.
    Returns: (winner_blood, winner_gender, best_raw_img)
    """
    blood_votes = []
    gender_votes = []
    images = []
    
    for i in range(scans):
        print(f"Polled Scan {i+1}/{scans}")
        img_raw = poll_esp32_single_scan(ip_address, 8080, f"Scan {i+1}/{scans}")
        if img_raw is not None:
            # 1. AI Models specifically require the raw, uncropped geometry for correct CNN translation
            b_img = preprocess_blood(img_raw)
            b_val, b_conf = predict_blood_group(b_img)
            blood_votes.append((b_val, b_conf))
            
            g_img = preprocess_gender(img_raw)
            g_val, g_conf = predict_gender(g_img)
            gender_votes.append((g_val, g_conf))
            
            # 2. But the biometric SIFT system strongly prefers the mathematically cropped structure
            img_cropped = auto_crop_fingerprint(img_raw)
            images.append(img_cropped)
        
        if i < scans - 1:
            time.sleep(2) # Give user time to place finger again
            
    if not images:
        return None, None, None
        
    # Aggregate Blood
    b_scores = {}
    for bg, conf in blood_votes:
        b_scores[bg] = b_scores.get(bg, 0) + conf
    sorted_b = sorted(b_scores.items(), key=lambda x: x[1], reverse=True)
    winner_bg = sorted_b[0][0] if sorted_b else None
    
    # Aggregate Gender
    g_scores = {}
    for gen, conf in gender_votes:
        g_scores[gen] = g_scores.get(gen, 0) + conf
    sorted_g = sorted(g_scores.items(), key=lambda x: x[1], reverse=True)
    winner_gen = sorted_g[0][0] if sorted_g else None
    
    # Return best image (we can just return the first one honestly)
    best_image = images[0]
    
    return winner_bg, winner_gen, best_image
