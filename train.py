"""
MNIST Handwritten Digit Classifier
CNN-based image classifier using TensorFlow/Keras
Author: Naveed Ahamed
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os

# ── 1. Load & Preprocess Data ─────────────────────────────────────────────────
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# Reshape for CNN input: (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Input shape      : {X_train.shape[1:]}")

# ── 2. Data Augmentation ──────────────────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="augmentation")

# ── 3. Build CNN Model ────────────────────────────────────────────────────────
def build_model():
    model = models.Sequential([
        # Input + Augmentation
        layers.Input(shape=(28, 28, 1)),
        data_augmentation,

        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),

        # Classifier Head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax")  # 10 digit classes
    ], name="mnist_cnn")

    return model

model = build_model()
model.summary()

# ── 4. Compile ────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy : {test_acc * 100:.2f}%")
print(f"   Test Loss    : {test_loss:.4f}")

# ── 7. Save Model ─────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
model.save("model/mnist_cnn.keras")
print("Model saved to model/mnist_cnn.keras")

# ── 8. Plot Training Curves ───────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/training_curves.png", dpi=150)
print("Training curves saved to plots/training_curves.png")

# ── 9. Sample Predictions ─────────────────────────────────────────────────────
predictions = model.predict(X_test[:16], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    color = "green" if predicted_labels[i] == y_test[i] else "red"
    ax.set_title(f"Pred: {predicted_labels[i]}\nTrue: {y_test[i]}", color=color, fontsize=8)
    ax.axis("off")

plt.suptitle("Sample Predictions (Green = Correct, Red = Wrong)", fontsize=12)
plt.tight_layout()
plt.savefig("plots/sample_predictions.png", dpi=150)
print("Sample predictions saved to plots/sample_predictions.png")

print("\nDone! Run predict.py to test on your own images.")
