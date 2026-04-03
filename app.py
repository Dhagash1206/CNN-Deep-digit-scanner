import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image, ImageOps


MODEL_PATH = "mnist_cnn_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'.\n"
        "Please run  python train_model.py  first."
    )

print(f"[INFO] Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")


def preprocess(image):
    if image is None:
        return None

    if isinstance(image, dict):
        image = image.get("composite")
        if image is None:
            layers = image.get("layers", [None])
            image = layers[0] if layers else None

    if image is None:
        return None

    img = Image.fromarray(image.astype("uint8"))

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype="float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def predict(image):
    if image is None:
        return "Draw a digit first!", {}

    arr = preprocess(image)

    if arr is None:
        return "Could not process image.", {}

    probabilities = model.predict(arr, verbose=0)[0]
    confidence = {str(i): float(probabilities[i]) for i in range(10)}
    predicted_digit = int(np.argmax(probabilities))
    confidence_pct = float(probabilities[predicted_digit]) * 100
    label = f"Predicted: {predicted_digit}   ({confidence_pct:.1f}% confidence)"
    return label, confidence


with gr.Blocks(title="Handwritten Digit Recognizer") as demo:

    gr.Markdown("# Handwritten Digit Recognizer\nDraw a **single digit (0-9)** and click **Predict**.")

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                label="Draw Here",
                type="numpy",
                image_mode="RGB",
                canvas_size=(280, 280),
                brush=gr.Brush(colors=["#000000"], default_size=20),
            )
            with gr.Row():
                predict_btn = gr.Button("Predict", variant="primary")
                clear_btn   = gr.Button("Clear",   variant="secondary")

        with gr.Column(scale=1):
            result_label = gr.Textbox(label="Result", interactive=False, lines=2)
            confidence_bar = gr.Label(label="Confidence per digit", num_top_classes=10)

    predict_btn.click(fn=predict, inputs=[canvas], outputs=[result_label, confidence_bar])
    clear_btn.click(fn=lambda: (None, "", {}), inputs=[], outputs=[canvas, result_label, confidence_bar])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft()
    )
