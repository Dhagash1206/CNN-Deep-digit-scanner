import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image, ImageOps

MODEL_PATH = "mnist_cnn_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Run train_model.py first.")

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded.")


def preprocess(image):
    if image is None:
        return None

    # Handle Gradio 6+ Sketchpad dict output
    if isinstance(image, dict):
        composite = image.get("composite")
        if composite is not None:
            image = composite
        else:
            layers = image.get("layers", [])
            image = layers[0] if layers else None

    if image is None:
        return None

    # Convert to PIL
    img = Image.fromarray(image.astype("uint8"))

    # Flatten RGBA onto white background
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    # Convert to grayscale
    img = img.convert("L")

    # Smart invert: MNIST has white digit on black background
    avg = np.mean(np.array(img))
    if avg > 127:
        img = ImageOps.invert(img)

    # Auto-center digit using bounding box
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        size = max(img.size)
        padded = Image.new("L", (size, size), 0)
        offset_x = (size - img.width) // 2
        offset_y = (size - img.height) // 2
        padded.paste(img, (offset_x, offset_y))
        img = padded

    # Resize to MNIST format
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype="float32") / 255.0
    return arr.reshape(1, 28, 28, 1)


def predict(image):
    if image is None:
        return "Draw a digit first!", {}

    arr = preprocess(image)
    if arr is None:
        return "Could not process image.", {}

    probabilities = model.predict(arr, verbose=0)[0]
    confidence = {str(i): float(probabilities[i]) for i in range(10)}
    predicted_digit = int(np.argmax(probabilities))
    confidence_pct = round(float(probabilities[predicted_digit]) * 100, 1)
    result = "Predicted: " + str(predicted_digit) + "  (" + str(confidence_pct) + "% confidence)"
    return result, confidence


def clear_all():
    return gr.update(value=None), "", {}


with gr.Blocks(title="Digit Recognizer") as demo:
    gr.Markdown("# Handwritten Digit Recognizer\nDraw a digit (0-9) and click Predict.")

    with gr.Row():
        with gr.Column():
            canvas = gr.Sketchpad(
                label="Draw Here",
                type="numpy",
                image_mode="RGB",
                canvas_size=(280, 280),
                brush=gr.Brush(colors=["#000000"], default_size=20)
            )
            with gr.Row():
                predict_btn = gr.Button("Predict", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column():
            result_label = gr.Textbox(label="Result", interactive=False, lines=2)
            confidence_bar = gr.Label(label="Confidence", num_top_classes=10)

    predict_btn.click(fn=predict, inputs=[canvas], outputs=[result_label, confidence_bar])
    clear_btn.click(fn=clear_all, inputs=[], outputs=[canvas, result_label, confidence_bar])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
        theme=gr.themes.Soft()
    )
