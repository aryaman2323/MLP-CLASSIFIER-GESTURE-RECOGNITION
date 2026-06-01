import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Load dataset ────────────────────────────────────────────────────
data_dict = pickle.load(open('./data.pickle', 'rb'))
data   = np.asarray(data_dict['data'], dtype=np.float32)   # shape: (N, 42)
labels = np.asarray(data_dict['labels'])                   # string class labels

# Filter to keep only target letters (A through H)
allowed_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
mask = np.isin(labels, allowed_classes)
data = data[mask]
labels = labels[mask]

# ── Get unique class labels dynamically ──────────────────────────────
unique_labels = np.unique(labels)
trained_letters = [str(lbl) for lbl in unique_labels]
print(f"Training MLP model on the following letters/classes ({len(trained_letters)} classes): {', '.join(trained_letters)}")

num_classes = len(unique_labels)

# ── Encode string labels to integers ───────────────────────────────
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)               # 0, 1, 2, ...
labels_cat     = keras.utils.to_categorical(labels_encoded, num_classes=num_classes)

# ── Split flat dataset: (N, 42) ────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_cat, test_size=0.2, shuffle=True,
    stratify=labels_encoded, random_state=42
)

# ── Build stable MLP model ─────────────────────────────────────────
#
#  Architecture:
#   Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.2) → Dense(num_classes, softmax)
#
#  Input:  42-element flat landmark vector.
#  Output: probability distribution over sign classes.
# ───────────────────────────────────────────────────────────────────

model = keras.Sequential([
    layers.Input(shape=(42,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# ── Train ────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss', verbose=1),
]

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ─────────────────────────────────────────────────────────
y_pred_probs = model.predict(x_test)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = np.argmax(y_test, axis=1)
score        = accuracy_score(y_true, y_pred)
print(f'\n{score * 100:.2f}% of samples were classified correctly!')

# ── Save model + label encoder ──────────────────────────────────────
# Keep the filename model_cnn.h5 to avoid having to rename loads in recognize_live
model.save('model_cnn.h5')
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump({'label_encoder': le}, f)

print("Model saved to model_cnn.h5")
print("Label encoder saved to label_encoder.pickle")
