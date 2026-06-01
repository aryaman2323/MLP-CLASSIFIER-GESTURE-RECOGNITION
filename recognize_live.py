import os
import urllib.request
import pickle
import time
import threading

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyttsx3
import tensorflow as tf
from tensorflow import keras

# ── Auto-download model asset if missing ─────────────────────────────
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print(" Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(" Model downloaded successfully!")

# ── Load trained CNN model & label encoder ──────────────────────────
model = keras.models.load_model('./model_cnn.h5')

with open('./label_encoder.pickle', 'rb') as f:
    le = pickle.load(f)['label_encoder']

# Map encoded integer back to letter dynamically using the LabelEncoder
def decode_prediction(pred_index):
    """Convert CNN output index → letter string."""
    class_str = le.inverse_transform([pred_index])[0]   # e.g. 'A', 'B', '0', ...
    return str(class_str)


# ── Text-to-Speech (runs in a separate thread to avoid blocking) ────
def speak(text):
    """Speak text in a background thread with a fresh engine instance."""
    def _speak():
        try:
            tts = pyttsx3.init()
            tts.setProperty('rate', 150)
            tts.setProperty('volume', 1.0)
            tts.say(text)
            tts.runAndWait()
            tts.stop()
        except Exception as e:
            print(f" ⚠ TTS error: {e}")
    t = threading.Thread(target=_speak, daemon=True)
    t.start()


# ── Camera & MediaPipe Tasks setup ──────────────────────────────────
cap = cv2.VideoCapture(0)

# Initialize Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ── Define connections for custom premium skeleton visualization ─────
CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm / knuckle base
    (5, 9), (9, 13), (13, 17)
]

# ── State for word building ─────────────────────────────────────────
sentence       = ""       # accumulated text
last_character = None     # character detected in previous frame
stable_count   = 0        # consecutive frames the same char was seen
STABLE_THRESHOLD = 15     # frames needed before a letter is "locked in"
letter_added   = False    # prevents adding the same held sign repeatedly
last_add_time  = 0        # timestamp of last letter addition
ADD_COOLDOWN   = 1.0      # seconds to wait before same letter can repeat

print("=" * 55)
print(" HAND SIGN → WORD → SPEECH  [CNN MODEL]")
print("=" * 55)
print(" Hold a sign steady to add a letter.")
print(" SPACE     = add a space between words")
print(" BACKSPACE = delete last character")
print(" ENTER     = speak the sentence aloud")
print(" C         = clear all text")
print(" Q         = quit")
print("=" * 55)

try:
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = landmarker.detect(mp_image)

        predicted_character = None

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            # ── Custom Premium Skeleton Rendering ───────────────────────
            # 1. Draw connections (sleek cyan-neon lines)
            for connection in CONNECTIONS:
                p1_idx, p2_idx = connection
                p1 = hand_landmarks[p1_idx]
                p2 = hand_landmarks[p2_idx]
                x1, y1 = int(p1.x * W), int(p1.y * H)
                x2, y2 = int(p2.x * W), int(p2.y * H)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # 2. Draw nodes (neon pink joints with white core)
            for lm in hand_landmarks:
                cx, cy = int(lm.x * W), int(lm.y * H)
                cv2.circle(frame, (cx, cy), 6, (180, 105, 255), -1)
                cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)

            # Extract landmarks for model input
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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # ── MLP inference ─────────────────────────────────────────────
            # Reshape to (1, 42) for MLP input
            input_data  = np.asarray(data_aux, dtype=np.float32).reshape(1, 42)
            pred_probs  = model.predict(input_data, verbose=0)          # (1, num_classes)
            pred_index  = int(np.argmax(pred_probs, axis=1)[0])
            confidence  = float(np.max(pred_probs))

            predicted_character = decode_prediction(pred_index)

            # ── Draw bounding box, predicted letter, and confidence ───────
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            label_text = f"{predicted_character} ({confidence * 100:.0f}%)"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)

        # ── Stability logic: lock in a letter after holding it steady ────
        if predicted_character is not None:
            if predicted_character == last_character:
                stable_count += 1
            else:
                stable_count = 1
                letter_added = False
            last_character = predicted_character

            if stable_count >= STABLE_THRESHOLD and not letter_added:
                now = time.time()
                if now - last_add_time >= ADD_COOLDOWN:
                    sentence      += predicted_character
                    letter_added   = True
                    last_add_time  = now
                    print(f" ✓ Added '{predicted_character}' → \"{sentence}\"")
        else:
            # No hand detected → reset stability
            stable_count   = 0
            last_character = None
            letter_added   = False

        # ── Draw HUD overlay ────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 90), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Current sentence
        display_text = sentence if sentence else "(show a sign to start)"
        cv2.putText(frame, display_text, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2, cv2.LINE_AA)

        # CNN label in corner
        cv2.putText(frame, "CNN MODEL", (W - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # Stability progress bar
        progress  = min(stable_count / STABLE_THRESHOLD, 1.0)
        bar_w     = int(progress * (W - 30))
        bar_color = (0, 255, 0) if progress >= 1.0 else (0, 180, 255)
        cv2.rectangle(frame, (15, 60), (15 + bar_w, 75), bar_color, -1)
        cv2.rectangle(frame, (15, 60), (W - 15, 75), (100, 100, 100), 1)

        # Controls hint at bottom
        cv2.putText(frame, "SPACE:space  BKSP:delete  ENTER:speak  C:clear  Q:quit",
                    (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow('Hand Sign Recognition — CNN', frame)

        # ── Keyboard controls ───────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            sentence += " "
            print(f" ✓ Added SPACE → \"{sentence}\"")
        elif key == 8:   # Backspace
            sentence = sentence[:-1]
            print(f" ✗ Deleted → \"{sentence}\"")
        elif key in (13, 10):   # Enter → speak
            if sentence.strip():
                print(f" 🔊 Speaking: \"{sentence.strip()}\"")
                speak(sentence.strip())
        elif key == ord('c'):
            sentence = ""
            print(" ✗ Cleared all text.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
