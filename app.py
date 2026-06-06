import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr
from PIL import Image, ImageOps

MODEL_PATH = "mnist_cnn_model.keras"
PREVIEW_SIZE = 168
INK_THRESHOLD = 15
MIN_INK_PIXELS = 40
LOW_CONFIDENCE = 55.0

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'.\n"
        "Please run  python train_model.py  first."
    )

print(f"[INFO] Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")


def _extract_image(image):
    if image is None:
        return None

    if isinstance(image, dict):
        data = image
        image = data.get("composite")
        if image is None:
            layers = data.get("layers", [])
            image = layers[0] if layers else None

    if image is None:
        return None

    return Image.fromarray(image.astype("uint8"))


def _to_grayscale_inverted(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    img = img.convert("L")
    return ImageOps.invert(img)


def _center_digit(gray: Image.Image) -> Image.Image | None:
    """Crop ink, pad, and center into a 28x28 MNIST-style frame."""
    arr = np.array(gray, dtype=np.uint8)
    ink = arr > INK_THRESHOLD

    if int(np.sum(ink)) < MIN_INK_PIXELS:
        return None

    rows = np.where(np.any(ink, axis=1))[0]
    cols = np.where(np.any(ink, axis=0))[0]
    cropped = arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    height, width = cropped.shape
    size = max(height, width)
    margin = int(size * 0.25)
    padded_size = size + margin * 2

    padded = np.zeros((padded_size, padded_size), dtype=np.uint8)
    y_off = (padded_size - height) // 2
    x_off = (padded_size - width) // 2
    padded[y_off : y_off + height, x_off : x_off + width] = cropped

    inner = Image.fromarray(padded).resize((20, 20), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(inner, (4, 4))
    return canvas


def preprocess(image):
    img = _extract_image(image)
    if img is None:
        return None, None

    gray = _to_grayscale_inverted(img)
    centered = _center_digit(gray)
    if centered is None:
        return None, None

    arr = np.array(centered, dtype="float32") / 255.0
    model_input = arr.reshape(1, 28, 28, 1)

    preview = centered.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
    preview_rgb = np.stack([preview, preview, preview], axis=-1)
    return model_input, preview_rgb


def _confidence_df(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "digit": [str(i) for i in range(10)],
            "confidence": probabilities * 100.0,
        }
    )


def _result_html(predicted: int, confidence_pct: float) -> str:
    tone = "high" if confidence_pct >= LOW_CONFIDENCE else "low"
    warning = (
        '<p class="warn">Low confidence — try a larger, centered stroke.</p>'
        if tone == "low"
        else ""
    )
    return f"""
    <div class="result-card {tone}">
        <p class="result-label">Predicted digit</p>
        <p class="result-digit">{predicted}</p>
        <p class="result-confidence">{confidence_pct:.1f}% confidence</p>
        {warning}
    </div>
    """


def _top3_html(probabilities: np.ndarray, predicted: int) -> str:
    order = np.argsort(probabilities)[::-1][:3]
    chips = []
    for rank, digit in enumerate(order, start=1):
        pct = float(probabilities[digit]) * 100
        active = "active" if digit == predicted else ""
        chips.append(
            f'<span class="chip {active}">#{rank} &nbsp;{digit} &nbsp;·&nbsp; {pct:.1f}%</span>'
        )
    return '<div class="chip-row">' + "".join(chips) + "</div>"


def _empty_outputs(message: str):
    return (
        f'<div class="result-card empty"><p class="result-label">{message}</p></div>',
        None,
        _confidence_df(np.zeros(10, dtype=np.float32)),
        "",
    )


def predict(image):
    if image is None:
        return _empty_outputs("Draw a digit on the canvas to begin.")

    model_input, preview = preprocess(image)
    if model_input is None:
        return _empty_outputs("No stroke detected — draw a thicker digit.")

    probabilities = model.predict(model_input, verbose=0)[0]
    predicted = int(np.argmax(probabilities))
    confidence_pct = float(probabilities[predicted]) * 100

    return (
        _result_html(predicted, confidence_pct),
        preview,
        _confidence_df(probabilities),
        _top3_html(probabilities, predicted),
    )


def clear_canvas():
    return (
        None,
        _empty_outputs("Canvas cleared. Draw a new digit.")[0],
        None,
        _confidence_df(np.zeros(10, dtype=np.float32)),
        "",
    )


CUSTOM_CSS = """
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
}
.hero {
    text-align: center;
    padding: 0.5rem 0 1.25rem;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.35rem;
}
.hero p {
    color: #5f6470;
    margin: 0;
}
.panel {
    border: 1px solid #e7e9ef;
    border-radius: 16px;
    padding: 1rem;
    background: #fafbfd;
}
.result-card {
    text-align: center;
    border-radius: 16px;
    padding: 1.25rem 1rem;
    background: linear-gradient(160deg, #f4f7ff 0%, #ffffff 100%);
    border: 1px solid #dbe4ff;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.result-card.low {
    background: linear-gradient(160deg, #fff8ef 0%, #ffffff 100%);
    border-color: #f3dcc0;
}
.result-card.empty {
    background: #f7f8fa;
    border-color: #e3e6ec;
}
.result-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin: 0 0 0.35rem;
}
.result-digit {
    font-size: 5.5rem;
    line-height: 1;
    font-weight: 800;
    margin: 0;
    color: #1d4ed8;
}
.result-card.low .result-digit { color: #b45309; }
.result-confidence {
    margin: 0.5rem 0 0;
    font-size: 1.05rem;
    color: #374151;
}
.warn {
    margin: 0.75rem 0 0;
    font-size: 0.9rem;
    color: #b45309;
}
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.25rem;
}
.chip {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.35rem 0.75rem;
    font-size: 0.9rem;
    background: #eef2ff;
    color: #3730a3;
    border: 1px solid #c7d2fe;
}
.chip.active {
    background: #1d4ed8;
    color: #ffffff;
    border-color: #1d4ed8;
    font-weight: 600;
}
.tips li { margin-bottom: 0.35rem; }
"""


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    block_background_fill="*neutral_50",
    block_border_width="1px",
    block_radius="12px",
    button_large_radius="10px",
)

with gr.Blocks(title="Digit Recognizer") as demo:
    gr.HTML(
        """
        <div class="hero">
            <h1>Handwritten Digit Recognizer</h1>
            <p>Draw a single digit (0–9). The model sees a centered 28×28 MNIST-style image.</p>
        </div>
        """
    )

    empty_df = _confidence_df(np.zeros(10, dtype=np.float32))

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel"]):
                canvas = gr.Sketchpad(
                    label="Draw here",
                    type="numpy",
                    image_mode="RGB",
                    canvas_size=(320, 320),
                    brush=gr.Brush(colors=["#111827"], default_size=22),
                )
                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                live_predict = gr.Checkbox(
                    label="Predict while drawing",
                    value=False,
                    info="Runs inference after each stroke (slightly slower).",
                )

        with gr.Column(scale=4):
            result_html = gr.HTML(
                value=_empty_outputs("Draw a digit on the canvas to begin.")[0]
            )
            top3_html = gr.HTML()
            with gr.Group(elem_classes=["panel"]):
                preview = gr.Image(
                    label="Model input (28×28, upscaled)",
                    type="numpy",
                    interactive=False,
                    height=PREVIEW_SIZE,
                )
                confidence_plot = gr.BarPlot(
                    value=empty_df,
                    x="digit",
                    y="confidence",
                    title="Confidence by digit",
                    y_lim=[0, 100],
                    height=260,
                    x_title="Digit",
                    y_title="Confidence (%)",
                )

    with gr.Accordion("Drawing tips for best accuracy", open=False):
        gr.Markdown(
            """
            - Draw **one digit**, large and centered on the canvas.
            - Use **thick strokes** — thin lines are harder to recognize.
            - Leave a little margin around the digit; cropping and centering are automatic.
            - If confidence is low, clear and redraw more boldly.
            """,
            elem_classes=["tips"],
        )

    outputs = [result_html, preview, confidence_plot, top3_html]

    predict_btn.click(fn=predict, inputs=[canvas], outputs=outputs)
    clear_btn.click(fn=clear_canvas, inputs=[], outputs=[canvas, *outputs])

    def maybe_predict(image, live):
        if live:
            return predict(image)
        return gr.skip()

    canvas.change(fn=maybe_predict, inputs=[canvas, live_predict], outputs=outputs)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=CUSTOM_CSS,
    )
