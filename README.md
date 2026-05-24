# Real-Time Hand Sign to Text & Speech Recognition

An end-to-end computer vision and deep learning application that detects hand gestures via webcam, converts them into text characters in real-time, builds them into words/sentences, and speaks the output aloud using offline Text-to-Speech.

This pipeline uses **Google MediaPipe** for hand landmark extraction, a **Multi-Layer Perceptron (MLP)** built with **TensorFlow/Keras** for sign classification, and **pyttsx3** for offline speech synthesis.

---

## Features

- **Real-Time Hand Tracking** — Extracts 21 hand landmarks per frame using MediaPipe's Hand Landmarker task.
- **Interactive Data Collector** — A menu-driven script to collect, manage, edit, and add gesture image datasets without starting from scratch.
- **MLP Classifier** — A lightweight fully-connected neural network (Dense layers + Dropout) trained on normalised 42-element landmark vectors.
- **Confidence Display** — The predicted letter and its softmax confidence percentage are overlaid on the bounding box in real time.
- **Frame-Stability Detection** — A letter is only registered once the same gesture is held steady for 15 consecutive frames, preventing accidental duplicates.
- **Visual HUD** — Semi-transparent overlay showing the current sentence, a stability progress bar, and a "CNN MODEL" badge.
- **Custom Hand Skeleton** — Cyan-neon bone connections and neon-pink joint nodes rendered directly on the webcam feed.
- **Background-Threaded TTS** — Each call to `speak()` creates a fresh `pyttsx3` engine in a daemon thread, so speech never freezes the camera loop.
- **Keyboard Controls** — Full editing of the accumulated sentence from the camera window.
- **Auto-Download Model Asset** — `recognize_live.py` automatically downloads `hand_landmarker.task` from Google's CDN if it is missing.

---

## Supported Gestures

By default the system is trained on 7 hand signs:

| Class | Letter |
|-------|--------|
| 0 | A |
| 1 | B |
| 2 | L |
| 3 | C |
| 4 | D |
| 5 | E |
| 6 | F |

You can extend this by using the data collection menu (option 3) and retraining.

---

## Model Architecture

The classifier in `train_model.py` is an **MLP** (not a Conv1D CNN), operating on a flat 42-element input:

```
Input: (42,)   ← 21 landmarks × (x_norm, y_norm)

Dense(128, ReLU) → Dropout(0.2)
Dense(64,  ReLU) → Dropout(0.2)
Dense(num_classes, Softmax)

Output: class probability distribution
```

Training uses the **Adam** optimiser (lr=1e-3) with **EarlyStopping** (patience=10) and **ReduceLROnPlateau** callbacks. The dataset is split 80/20 with stratification, and training runs for up to 50 epochs.

Saved artefacts: `model_cnn.h5` and `label_encoder.pickle`.

---

## Project Structure

```
CNN-GESTURE-RECOGNITION/
│
├── .gitignore              # Ignores virtual envs, datasets, and model files
├── requirements.txt        # Pinned dependencies
├── hand_landmarker.task    # MediaPipe hand landmark model (auto-downloaded if absent)
│
├── collect_data.py         # Step 1 — Interactive dataset collection and management
├── create_dataset.py       # Step 2 — Extract and normalise landmarks into data.pickle
├── train_model.py          # Step 3 — Train MLP and save model_cnn.h5
└── recognize_live.py       # Step 4 — Live webcam recognition, word building & TTS
```

---

## Setup

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/aryaman2323/CNN-GESTURE-RECOGNITION.git
cd CNN-GESTURE-RECOGNITION

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins:

```
opencv-python==4.7.0.68
mediapipe==0.10.35
tensorflow>=2.12.0
numpy>=1.23.0
scikit-learn>=1.2.0
pyttsx3==2.99
```

---

## Pipeline

### Step 1 — Collect gesture images (`collect_data.py`)

Launches an interactive menu with the following options:

```
1. Erase ALL data and recollect everything from scratch
2. Edit a specific letter (re-collect its images)
3. Add a new letter/class
4. Collect ONLY missing letters (skip existing)
5. Exit
```

Running the script:

```bash
python collect_data.py
```

For each class, a webcam window opens showing the target letter. Press **Q** when your hand is ready — 100 images are captured automatically. Raw images are saved under `./data/<class_index>/`.

### Step 2 — Extract landmarks (`create_dataset.py`)

Processes raw images through MediaPipe and saves normalised (x − x_min, y − y_min) landmark coordinates.

```bash
python create_dataset.py
```

Output: `data.pickle` containing `{'data': [...], 'labels': [...]}`.

### Step 3 — Train the model (`train_model.py`)

```bash
python train_model.py
```

Prints per-epoch accuracy, runs up to 50 epochs with early stopping, and reports final test accuracy. Outputs:

- `model_cnn.h5` — trained Keras model
- `label_encoder.pickle` — scikit-learn `LabelEncoder` mapping class indices to letters

### Step 4 — Live recognition (`recognize_live.py`)

```bash
python recognize_live.py
```

`hand_landmarker.task` is downloaded automatically from Google's model CDN if not present. The webcam window opens immediately.

---

## Live Recognition Controls

| Key | Action |
|-----|--------|
| Hold a gesture | Registers the letter once stable for 15 consecutive frames |
| `SPACE` | Adds a space between words |
| `BACKSPACE` | Deletes the last character |
| `ENTER` | Speaks the full sentence aloud |
| `C` | Clears all accumulated text |
| `Q` | Exits the app |

A 1-second cooldown prevents the same letter from being added repeatedly while the gesture is held.

---

## Extending to New Gestures

1. Run `collect_data.py` → option **3** to add a new class index and letter.
2. Run `create_dataset.py` to regenerate `data.pickle`.
3. Run `train_model.py` to retrain and overwrite `model_cnn.h5`.
4. Update `labels_dict` in both `train_model.py` and `recognize_live.py` to include the new mapping.

---

## Built With

- [OpenCV](https://opencv.org/) — webcam capture and frame rendering
- [MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection (Tasks API)
- [TensorFlow / Keras](https://www.tensorflow.org/) — MLP model training and inference
- [scikit-learn](https://scikit-learn.org/) — `LabelEncoder` and train/test split
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) — offline text-to-speech
