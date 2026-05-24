import os
import pickle
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Auto-download model asset if missing ─────────────────────────────
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print(" Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(" Model downloaded successfully!")

# ── Initialize MediaPipe Tasks Hand Landmarker ───────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

DATA_DIR = './data'

data = []
labels = []

# Check if data directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory '{DATA_DIR}' not found. Please run collect_data.py first.")
    exit(1)

for dir_ in os.listdir(DATA_DIR):
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert NumPy RGB frame to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Perform synchronous detection
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            for i in range(len(hand_landmarks)):
                x = hand_landmarks[i].x
                y = hand_landmarks[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks)):
                x = hand_landmarks[i].x
                y = hand_landmarks[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

landmarker.close()

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(f"Dataset saved: {len(data)} samples across {len(set(labels))} classes.")
