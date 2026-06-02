"""
Predict digit from a custom image
Usage: python predict.py --image path/to/image.png
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def load_and_preprocess(image_path):
    """Load an image and preprocess it for the model."""
    img = tf.keras.utils.load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    arr = tf.keras.utils.img_to_array(img)

    # Invert if white background (MNIST digits are white on black)
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr.astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def predict(image_path):
    # Load model
    if not os.path.exists("model/mnist_cnn.keras"):
        print("Model not found. Run train.py first.")
        sys.exit(1)

    model = tf.keras.models.load_model("model/mnist_cnn.keras")

    # Preprocess
    img_array = load_and_preprocess(image_path)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_digit = np.argmax(predictions)
    confidence = predictions[predicted_digit] * 100

    print(f"\nPredicted Digit : {predicted_digit}")
    print(f"Confidence      : {confidence:.2f}%")
    print("\nClass probabilities:")
    for digit, prob in enumerate(predictions):
        bar = "█" * int(prob * 30)
        print(f"  {digit}: {bar} {prob*100:.1f}%")

    # Show image with prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(img_array.reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {predicted_digit}  (Confidence: {confidence:.1f}%)", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict handwritten digit from image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    predict(args.image)
