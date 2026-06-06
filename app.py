import os
from datetime import datetime
from functools import lru_cache

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr
from PIL import Image, ImageOps

MODEL_PATH = "mnist_cnn_model.keras"
PREVIEW_SIZE = 168
CANVAS_SIZE = (320, 320)
INK_THRESHOLD = 15
MIN_INK_PIXELS = 40
LOW_CONFIDENCE = 55.0
HISTORY_LIMIT = 25

TTA_SHIFTS = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'.\n"
        "Please run  python train_model.py  first."
    )

print(f"[INFO] Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")


@lru_cache(maxsize=1)
def _mnist_examples():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    examples = {}
    for digit in range(10):
        idx = int(np.where(y_test == digit)[0][0])
        examples[digit] = x_test[idx]
    return examples


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
    return ImageOps.autocontrast(ImageOps.invert(img), cutoff=1)


def _center_digit(gray: Image.Image) -> Image.Image | None:
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


def _shift_digit(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.roll(arr, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0
    return shifted


def _run_inference(model_input: np.ndarray, use_tta: bool) -> np.ndarray:
    if not use_tta:
        return model.predict(model_input, verbose=0)[0]

    base = model_input[0, :, :, 0]
    batch = np.stack(
        [_shift_digit(base, dx, dy) for dx, dy in TTA_SHIFTS],
        axis=0,
    )
    batch = batch[..., np.newaxis]
    probs = model.predict(batch, verbose=0)
    return np.mean(probs, axis=0)


def preprocess(image):
    img = _extract_image(image)
    if img is None:
        return None, None, None

    gray = _to_grayscale_inverted(img)
    centered = _center_digit(gray)
    if centered is None:
        return None, None, None

    arr = np.array(centered, dtype="float32") / 255.0
    model_input = arr.reshape(1, 28, 28, 1)

    preview = centered.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
    preview_rgb = np.stack([preview, preview, preview], axis=-1)

    raw_crop = gray.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
    raw_rgb = np.stack([raw_crop, raw_crop, raw_crop], axis=-1)
    return model_input, preview_rgb, raw_rgb


def mnist_to_canvas(digit: int) -> np.ndarray:
    sample = _mnist_examples()[digit]
    glyph = Image.fromarray(sample).resize((240, 240), Image.NEAREST)
    glyph = ImageOps.invert(glyph)

    canvas = np.full((*CANVAS_SIZE, 3), 255, dtype=np.uint8)
    offset_y = (CANVAS_SIZE[1] - 240) // 2
    offset_x = (CANVAS_SIZE[0] - 240) // 2
    rgb = np.stack([glyph, glyph, glyph], axis=-1)
    canvas[offset_y : offset_y + 240, offset_x : offset_x + 240] = rgb
    return canvas


def _confidence_df(probabilities: np.ndarray, predicted: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "digit": [str(i) for i in range(10)],
            "confidence": probabilities * 100.0,
            "kind": ["top" if i == predicted else "other" for i in range(10)],
        }
    )


def _entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1e-9, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def _margin(probabilities: np.ndarray) -> float:
    top2 = np.sort(probabilities)[-2:]
    return float((top2[-1] - top2[-2]) * 100.0)


def _result_html(predicted: int, confidence_pct: float, margin_pct: float) -> str:
    tone = "high" if confidence_pct >= LOW_CONFIDENCE and margin_pct >= 12 else "low"
    warning = ""
    if confidence_pct < LOW_CONFIDENCE:
        warning = '<p class="warn">Low confidence — redraw larger and centered.</p>'
    elif margin_pct < 12:
        warning = '<p class="warn">Close call — top two digits are similar.</p>'

    return f"""
    <div class="result-card {tone}">
        <p class="result-label">Predicted digit</p>
        <p class="result-digit">{predicted}</p>
        <p class="result-confidence">{confidence_pct:.1f}% confidence</p>
        <p class="result-sub">Margin over 2nd place: {margin_pct:.1f}%</p>
        {warning}
    </div>
    """


def _metrics_html(confidence_pct: float, margin_pct: float, entropy: float, use_tta: bool) -> str:
    certainty = max(0.0, min(100.0, 100.0 - entropy * 25.0))
    mode = "TTA ensemble" if use_tta else "Single pass"
    return f"""
    <div class="metrics">
        <div class="metric"><span>Certainty</span><strong>{certainty:.0f}%</strong></div>
        <div class="metric"><span>Top-1 lead</span><strong>{margin_pct:.1f}%</strong></div>
        <div class="metric"><span>Entropy</span><strong>{entropy:.2f}</strong></div>
        <div class="metric"><span>Mode</span><strong>{mode}</strong></div>
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


def _history_dataframe(history: list[dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["time", "digit", "confidence", "margin"])
    return pd.DataFrame(history)


def _empty_outputs(message: str):
    return (
        f'<div class="result-card empty"><p class="result-label">{message}</p></div>',
        None,
        None,
        _confidence_df(np.zeros(10, dtype=np.float32), 0),
        "",
        _metrics_html(0, 0, 0, False),
        _history_dataframe([]),
        [],
    )


def predict(image, history, use_tta):
    history = history or []

    if image is None:
        return _empty_outputs("Draw a digit on the canvas to begin.")

    model_input, preview, raw = preprocess(image)
    if model_input is None:
        return _empty_outputs("No stroke detected — draw a thicker digit.")

    probabilities = _run_inference(model_input, use_tta)
    predicted = int(np.argmax(probabilities))
    confidence_pct = float(probabilities[predicted]) * 100
    margin_pct = _margin(probabilities)
    entropy = _entropy(probabilities)

    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "digit": predicted,
        "confidence": round(confidence_pct, 1),
        "margin": round(margin_pct, 1),
    }
    history = [entry, *history][:HISTORY_LIMIT]

    return (
        _result_html(predicted, confidence_pct, margin_pct),
        preview,
        raw,
        _confidence_df(probabilities, predicted),
        _top3_html(probabilities, predicted),
        _metrics_html(confidence_pct, margin_pct, entropy, use_tta),
        _history_dataframe(history),
        history,
    )


def clear_canvas(history):
    return (
        None,
        _empty_outputs("Canvas cleared. Draw a new digit.")[0],
        None,
        None,
        _confidence_df(np.zeros(10, dtype=np.float32), 0),
        "",
        _metrics_html(0, 0, 0, False),
        _history_dataframe(history or []),
        history or [],
    )


def load_example(digit: int, history, use_tta):
    canvas = mnist_to_canvas(digit)
    outputs = predict(canvas, history, use_tta)
    return (canvas, *outputs)


def clear_history():
    return _history_dataframe([]), []


CUSTOM_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}
.hero {
    text-align: center;
    padding: 0.25rem 0 1rem;
}
.hero-badge {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1d4ed8;
    background: #dbeafe;
    border-radius: 999px;
    padding: 0.25rem 0.65rem;
    margin-bottom: 0.5rem;
}
.hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem;
}
.hero p { color: #5f6470; margin: 0; }
.panel {
    border: 1px solid #e7e9ef;
    border-radius: 16px;
    padding: 1rem;
    background: #fafbfd;
}
.result-card {
    text-align: center;
    border-radius: 16px;
    padding: 1.1rem 1rem;
    background: linear-gradient(160deg, #f4f7ff 0%, #ffffff 100%);
    border: 1px solid #dbe4ff;
    min-height: 210px;
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
    min-height: 120px;
}
.result-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin: 0 0 0.35rem;
}
.result-digit {
    font-size: 5rem;
    line-height: 1;
    font-weight: 800;
    margin: 0;
    color: #1d4ed8;
}
.result-card.low .result-digit { color: #b45309; }
.result-confidence { margin: 0.45rem 0 0; font-size: 1.05rem; color: #374151; }
.result-sub { margin: 0.2rem 0 0; font-size: 0.92rem; color: #6b7280; }
.warn { margin: 0.65rem 0 0; font-size: 0.88rem; color: #b45309; }
.metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.55rem;
}
.metric {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.55rem 0.65rem;
    text-align: center;
}
.metric span {
    display: block;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
}
.metric strong { font-size: 1rem; color: #111827; }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.2rem; }
.chip {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.32rem 0.7rem;
    font-size: 0.88rem;
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
.example-row button { min-width: 2.1rem !important; font-weight: 700 !important; }
.preview-pair { gap: 0.75rem !important; }
@media (max-width: 900px) {
    .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
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
    history_state = gr.State([])

    gr.HTML(
        """
        <div class="hero">
            <div class="hero-badge">MNIST CNN · Gradio</div>
            <h1>Handwritten Digit Recognizer</h1>
            <p>Draw, upload an example, or pick a sample digit — see what the model actually reads.</p>
        </div>
        """
    )

    empty_df = _confidence_df(np.zeros(10, dtype=np.float32), 0)

    with gr.Tabs():
        with gr.Tab("Recognize"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["panel"]):
                        canvas = gr.Sketchpad(
                            label="Draw here",
                            type="numpy",
                            image_mode="RGB",
                            canvas_size=CANVAS_SIZE,
                            brush=gr.Brush(colors=["#111827"], default_size=22),
                            eraser=gr.Eraser(default_size=28),
                        )
                        gr.Markdown("**Try a sample** (loads MNIST digit onto canvas)")
                        with gr.Row(elem_classes=["example-row"]):
                            example_btns = [
                                gr.Button(str(d), size="sm", variant="secondary") for d in range(10)
                            ]
                        with gr.Row():
                            predict_btn = gr.Button("Predict", variant="primary", scale=2)
                            clear_btn = gr.Button("Clear canvas", variant="secondary", scale=1)
                        with gr.Row():
                            live_predict = gr.Checkbox(
                                label="Live predict",
                                value=False,
                                info="Predict after each stroke.",
                            )
                            use_tta = gr.Checkbox(
                                label="TTA mode",
                                value=False,
                                info="Average 7 shifted views for tougher strokes.",
                            )

                with gr.Column(scale=4):
                    result_html = gr.HTML(
                        value=_empty_outputs("Draw a digit on the canvas to begin.")[0]
                    )
                    metrics_html = gr.HTML(value=_metrics_html(0, 0, 0, False))
                    top3_html = gr.HTML()
                    with gr.Group(elem_classes=["panel"]):
                        with gr.Row(elem_classes=["preview-pair"]):
                            preview = gr.Image(
                                label="Model input",
                                type="numpy",
                                interactive=False,
                                height=PREVIEW_SIZE,
                            )
                            raw_preview = gr.Image(
                                label="Raw capture",
                                type="numpy",
                                interactive=False,
                                height=PREVIEW_SIZE,
                            )
                        confidence_plot = gr.BarPlot(
                            value=empty_df,
                            x="digit",
                            y="confidence",
                            color="kind",
                            title="Confidence by digit",
                            color_map={"top": "#1d4ed8", "other": "#cbd5e1"},
                            y_lim=[0, 100],
                            height=240,
                            x_title="Digit",
                            y_title="Confidence (%)",
                        )

        with gr.Tab("History"):
            with gr.Group(elem_classes=["panel"]):
                history_table = gr.Dataframe(
                    headers=["time", "digit", "confidence", "margin"],
                    datatype=["str", "number", "number", "number"],
                    label="Recent predictions",
                    interactive=False,
                )
                clear_history_btn = gr.Button("Clear history", variant="secondary")

    with gr.Accordion("Tips & how it works", open=False):
        gr.Markdown(
            """
            **Drawing:** one digit, thick strokes, centered on the canvas.

            **Preprocessing:** autocontrast → crop ink → pad → resize into 28×28 (MNIST layout).

            **TTA mode:** runs 7 slightly shifted versions and averages probabilities — slower but stabler on messy strokes.

            **Metrics:** *margin* is gap between top two classes; *entropy* measures overall uncertainty.
            """
        )

    outputs = [
        result_html,
        preview,
        raw_preview,
        confidence_plot,
        top3_html,
        metrics_html,
        history_table,
        history_state,
    ]

    predict_inputs = [canvas, history_state, use_tta]

    predict_btn.click(fn=predict, inputs=predict_inputs, outputs=outputs)
    clear_btn.click(fn=clear_canvas, inputs=[history_state], outputs=[canvas, *outputs])
    clear_history_btn.click(fn=clear_history, outputs=[history_table, history_state])

    for digit, btn in enumerate(example_btns):
        btn.click(
            fn=lambda h, t, d=digit: load_example(d, h, t),
            inputs=[history_state, use_tta],
            outputs=[canvas, *outputs],
        )

    def maybe_predict(image, live, history, tta):
        if live:
            return predict(image, history, tta)
        return gr.skip()

    canvas.change(
        fn=maybe_predict,
        inputs=[canvas, live_predict, history_state, use_tta],
        outputs=outputs,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=CUSTOM_CSS,
    )
