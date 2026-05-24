# Real-Time Hand Sign to Text & Speech Recognition (CNN)

An end-to-end interactive computer vision and deep learning application that detects hand gestures via a webcam, converts them into text characters in real-time, builds them into words/sentences, and speaks the words aloud using Text-to-Speech (TTS).

This pipeline uses **Google MediaPipe** for hand landmark extraction, a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** for sign classification, and **pyttsx3** for offline local speech synthesis.

> ⚡ **Key difference from the Random Forest version:** The classifier is replaced with a 1D CNN (`Conv1D`) that learns spatial feature hierarchies over the 42-element MediaPipe landmark vector, typically achieving higher accuracy with better generalisation across hand shapes and sizes.

---

## Features

- **Real-Time Hand Tracking**: Extracts 21 3D hand landmarks at up to 30+ FPS using MediaPipe.
- **Custom Dataset Collector**: Quick script to record your own hand sign images via your webcam.
- **CNN Classification**: A lightweight `Conv1D` deep learning model (3 convolutional blocks + dense head) with BatchNorm, Dropout, and GlobalAveragePooling for robust sign recognition.
- **Confidence Display**: The predicted letter and its softmax confidence percentage are overlaid on the bounding box in real time.
- **Frame-Stability Detection**: Word building only registers a letter once the gesture is held stable for a set duration, avoiding accidental duplicate letters.
- **Visual HUD**: Elegant overlay displaying the current built sentence and a stability progress bar. CNN badge shown in corner.
- **Background Threaded TTS**: Seamlessly translates text into speech without freezing or blocking the webcam loop.
- **Keyboard Commands**: Full control over editing, clearing, and speaking your accumulated word/sentence directly from the camera window.

---

## Model Architecture

```
Input: (42, 1)   ← 21 landmarks × (x, y) normalised coordinates

Conv1D(64)  → BatchNorm → ReLU → Dropout(0.25)
Conv1D(128) → BatchNorm → ReLU → Dropout(0.25)
Conv1D(256) → BatchNorm → ReLU

GlobalAveragePooling1D

Dense(128) → ReLU → Dropout(0.4)
Dense(num_classes) → Softmax

Output: class probability distribution
```

Training uses **Adam** optimiser with **EarlyStopping** and **ReduceLROnPlateau** callbacks for automatic tuning.

---

## Project Structure

```
SignLanguageDetection-CNN/
│
├── .gitignore              # Ignores virtual envs, local datasets, and model files
├── requirements.txt        # Project dependencies (OpenCV, MediaPipe, TensorFlow, pyttsx3)
│
├── collect_data.py         # Step 1: Collect hand gesture images via webcam
├── create_dataset.py       # Step 2: Extract hand landmark coordinates using MediaPipe
├── train_model.py          # Step 3: Train CNN model and save model_cnn.h5
└── recognize_live.py       # Step 4: Real-time hand sign detection, word building & speech
```

---

## Step-by-Step Setup & Running

### 1. Clone & Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/SignLanguageDetection-CNN.git
cd SignLanguageDetection-CNN

# Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Or on macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

### 2. Pipeline Execution

#### Step 1: Data Collection (`collect_data.py`)

Collects images for your training dataset. By default, captures 7 classes (`A`, `B`, `L`, `C`, `D`, `E`, `F`) with 100 samples each.

```bash
python collect_data.py
```

- **How to use**: Align your hand for the first gesture in the video window and press `Q` to capture 100 images. Repeat for the remaining gestures.

#### Step 2: Extract Landmarks (`create_dataset.py`)

Processes captured raw images through MediaPipe to extract coordinates and normalises them.

```bash
python create_dataset.py
```

- **Result**: Generates a local `data.pickle` containing structured landmarks and labels.

#### Step 3: Train the CNN Model (`train_model.py`)

Splits the dataset, trains a **Conv1D CNN**, logs per-epoch accuracy, and saves the model.

```bash
python train_model.py
```

- **Result**: Generates `model_cnn.h5` (trained Keras model) and `label_encoder.pickle` (class index mapping).
- Training runs for up to 80 epochs with early stopping — typically converges in 30–50 epochs.

#### Step 4: Run Live Recognition & Speech (`recognize_live.py`)

Launches the webcam feed to detect signs, build words, and speak them.

```bash
python recognize_live.py
```

---

## Live Recognition Controls

Use the following keyboard shortcuts while the camera window is selected:

| Key Command           | Action                                                                 |
| --------------------- | ---------------------------------------------------------------------- |
| **Hold Hand Gesture** | Builds the sign when held stable for 15 consecutive frames.           |
| **`SPACEBAR`**        | Adds a space between words.                                            |
| **`BACKSPACE`**       | Deletes the last character in the built sentence.                      |
| **`ENTER`**           | Speaks the current sentence aloud through your speakers.               |
| **`C`**               | Clears the entire current text.                                        |
| **`Q`**               | Exits the live recognition app.                                        |

---

## CNN vs Random Forest

| Feature              | Random Forest (original)      | CNN (this repo)                          |
|----------------------|-------------------------------|------------------------------------------|
| Model type           | Ensemble of decision trees    | Deep learning (Conv1D)                   |
| Input format         | Flat 42-feature vector        | (42, 1) sequence — same features         |
| Feature learning     | Manual / hand-crafted         | Learned automatically by convolutions    |
| Confidence output    | Class vote / probability      | Softmax probability (shown on overlay)   |
| Generalisation       | Good on small datasets        | Better on larger/varied datasets         |
| Training time        | Seconds                       | ~1–5 minutes (CPU), faster with GPU      |
| Saved artefact       | `model.p` (pickle)            | `model_cnn.h5` + `label_encoder.pickle`  |

---

## Built With

- [OpenCV](https://opencv.org/) — Computer Vision & Video Processing
- [MediaPipe](https://github.com/google/mediapipe) — Hand Landmark Tracking
- [TensorFlow / Keras](https://www.tensorflow.org/) — CNN Deep Learning Model
- [scikit-learn](https://scikit-learn.org/) — LabelEncoder & train/test split utilities
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) — Text-to-Speech Engine
