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

from calibration import apply_temperature, load_temperature
from preprocessing import (
    center_digit,
    extract_image,
    is_blank,
    rotate_digit,
    shift_digit,
    to_grayscale_inverted,
)

MODEL_PATH = "mnist_cnn_model.keras"
CALIBRATION_PATH = "mnist_cnn_model.calibration.json"
SAMPLES_PATH = os.path.join("assets", "mnist_samples.png")
FEEDBACK_DIR = "feedback"
SERVER_HOST = os.environ.get("DIGIT_APP_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("DIGIT_APP_PORT", "7860"))
PREVIEW_SIZE = 168
CANVAS_SIZE = (520, 520)
GLYPH_SIZE = int(CANVAS_SIZE[0] * 0.75)
BRUSH_SIZE = 22
ERASER_SIZE = 44
LOW_CONFIDENCE = 55.0
# Below this calibrated confidence the app answers "unclear" instead of naming a digit.
REJECT_CONFIDENCE = 40.0
HISTORY_LIMIT = 25

TTA_SHIFTS = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]
TTA_ROTATIONS = [-10, 10]
TTA_VIEW_COUNT = len(TTA_SHIFTS) + len(TTA_ROTATIONS)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'.\n"
        "Please run  python train_model.py  first."
    )

print(f"[INFO] Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
temperature = load_temperature(CALIBRATION_PATH)
print(f"[INFO] Model loaded successfully (temperature {temperature:.3f}).")


@lru_cache(maxsize=1)
def _mnist_examples():
    if os.path.exists(SAMPLES_PATH):
        strip = np.array(Image.open(SAMPLES_PATH).convert("L"))
        return {digit: strip[:, digit * 28 : (digit + 1) * 28] for digit in range(10)}

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    examples = {}
    for digit in range(10):
        idx = int(np.where(y_test == digit)[0][0])
        examples[digit] = x_test[idx]
    return examples


def _forward(batch: np.ndarray) -> np.ndarray:
    # Direct call skips predict()'s per-call setup, which dominates for tiny batches.
    return apply_temperature(model(batch, training=False).numpy(), temperature)


def _run_inference(model_input: np.ndarray, use_tta: bool) -> np.ndarray:
    if not use_tta:
        return _forward(model_input)[0]

    base = model_input[0, :, :, 0]
    views = [shift_digit(base, dx, dy) for dx, dy in TTA_SHIFTS]
    views += [rotate_digit(base, degrees) for degrees in TTA_ROTATIONS]
    batch = np.stack(views, axis=0)
    batch = batch[..., np.newaxis]
    probs = _forward(batch)
    return np.mean(probs, axis=0)


def preprocess(image):
    img = extract_image(image)
    if img is None:
        return None, None, None, None

    gray = to_grayscale_inverted(img)
    centered = center_digit(gray)
    if centered is None:
        return None, None, None, None

    arr = np.array(centered, dtype="float32") / 255.0
    model_input = arr.reshape(1, 28, 28, 1)

    preview = centered.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
    preview_rgb = np.stack([preview, preview, preview], axis=-1)

    raw_crop = gray.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
    raw_rgb = np.stack([raw_crop, raw_crop, raw_crop], axis=-1)
    return model_input, preview_rgb, raw_rgb, np.array(centered, dtype=np.uint8)


def mnist_to_canvas(digit: int) -> np.ndarray:
    sample = _mnist_examples()[digit]
    glyph = Image.fromarray(sample).resize((GLYPH_SIZE, GLYPH_SIZE), Image.NEAREST)
    glyph = ImageOps.invert(glyph)

    canvas = np.full((*CANVAS_SIZE, 3), 255, dtype=np.uint8)
    offset_y = (CANVAS_SIZE[1] - GLYPH_SIZE) // 2
    offset_x = (CANVAS_SIZE[0] - GLYPH_SIZE) // 2
    rgb = np.stack([glyph, glyph, glyph], axis=-1)
    canvas[offset_y : offset_y + GLYPH_SIZE, offset_x : offset_x + GLYPH_SIZE] = rgb
    return canvas


def _blank_canvas() -> np.ndarray:
    return np.full((*CANVAS_SIZE, 3), 255, dtype=np.uint8)


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
    if confidence_pct < REJECT_CONFIDENCE:
        return f"""
    <div class="result-card low">
        <p class="result-label">Unclear input</p>
        <p class="result-digit">?</p>
        <p class="result-confidence">Best guess {predicted} at {confidence_pct:.1f}%</p>
        <p class="warn">This doesn't look like a clear digit — try redrawing it.</p>
    </div>
    """

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


def _metrics_html(
    confidence_pct: float, margin_pct: float, entropy: float, use_tta: bool
) -> str:
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


OUTPUT_COUNT = 9


def _skip_outputs():
    return tuple(gr.skip() for _ in range(OUTPUT_COUNT))


def _empty_outputs(message: str, history=None):
    history = history or []
    return (
        f'<div class="result-card empty"><p class="result-label">{message}</p></div>',
        None,
        None,
        _confidence_df(np.zeros(10, dtype=np.float32), 0),
        "",
        _metrics_html(0, 0, 0, False),
        _history_dataframe(history),
        history,
        None,
    )


def predict(image, history, use_tta, record=True):
    history = history or []

    if image is None:
        return _empty_outputs("Draw a digit on the canvas to begin.", history)

    model_input, preview, raw, digit_28 = preprocess(image)
    if model_input is None:
        return _empty_outputs("No stroke detected — draw a thicker digit.", history)

    probabilities = _run_inference(model_input, use_tta)
    predicted = int(np.argmax(probabilities))
    confidence_pct = float(probabilities[predicted]) * 100
    margin_pct = _margin(probabilities)
    entropy = _entropy(probabilities)

    if record:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "digit": "?" if confidence_pct < REJECT_CONFIDENCE else str(predicted),
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
        digit_28,
    )


def clear_canvas(history):
    # The canvas itself is cleared in the browser by CLEAR_CANVAS_JS; this only resets the outputs.
    return _empty_outputs("Canvas cleared. Draw a new digit.", history)


def load_example(digit: int, history, use_tta, live):
    canvas = mnist_to_canvas(digit)
    # Live predict already runs on canvas.change, so predicting here would run twice.
    if live:
        return (canvas, *_skip_outputs())
    return (canvas, *predict(canvas, history, use_tta, record=False))


def save_correction(label, digit_28):
    if digit_28 is None:
        return "Draw a digit first, then save its correct label."
    if label is None or label == "":
        return "Pick the correct digit first."
    folder = os.path.join(FEEDBACK_DIR, str(label))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
    Image.fromarray(digit_28).save(path)
    total = len([name for _, _, files in os.walk(FEEDBACK_DIR) for name in files])
    return f"Saved as {label} ({total} corrections so far). Run `python finetune.py` to learn from them."


def clear_history():
    return _history_dataframe([]), []


# Dark mode is applied at runtime via [data-theme="dark"] overrides in CUSTOM_CSS.
LIGHT_THEME = gr.themes.Soft(
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


CUSTOM_CSS = """
:root,
[data-theme="light"] {
    --app-page-bg: #f3f4f6;
    --app-panel-bg: #fafbfd;
    --app-panel-border: #e7e9ef;
    --app-text-muted: #5f6470;
    --app-text-sub: #6b7280;
    --app-text-body: #374151;
    --app-text-strong: #111827;
    --app-badge-text: #1d4ed8;
    --app-badge-bg: #dbeafe;
    --app-result-bg: linear-gradient(160deg, #f4f7ff 0%, #ffffff 100%);
    --app-result-border: #dbe4ff;
    --app-result-low-bg: linear-gradient(160deg, #fff8ef 0%, #ffffff 100%);
    --app-result-low-border: #f3dcc0;
    --app-result-empty-bg: #f7f8fa;
    --app-result-empty-border: #e3e6ec;
    --app-digit: #1d4ed8;
    --app-digit-low: #b45309;
    --app-warn: #b45309;
    --app-metric-bg: #ffffff;
    --app-metric-border: #e5e7eb;
    --app-chip-bg: #eef2ff;
    --app-chip-text: #3730a3;
    --app-chip-border: #c7d2fe;
    --app-chip-active-bg: #1d4ed8;
    --app-chip-active-text: #ffffff;
    --app-plot-top: #1d4ed8;
    --app-plot-other: #cbd5e1;
}

[data-theme="dark"] {
    --app-page-bg: #0f1117;
    --app-panel-bg: #1a1d27;
    --app-panel-border: #2d3344;
    --app-text-muted: #9ca3af;
    --app-text-sub: #9ca3af;
    --app-text-body: #d1d5db;
    --app-text-strong: #f9fafb;
    --app-badge-text: #93c5fd;
    --app-badge-bg: #1e3a5f;
    --app-result-bg: linear-gradient(160deg, #1e293b 0%, #1a1d27 100%);
    --app-result-border: #334155;
    --app-result-low-bg: linear-gradient(160deg, #3b2f1f 0%, #1a1d27 100%);
    --app-result-low-border: #854d0e;
    --app-result-empty-bg: #151821;
    --app-result-empty-border: #2d3344;
    --app-digit: #60a5fa;
    --app-digit-low: #fbbf24;
    --app-warn: #fbbf24;
    --app-metric-bg: #12151f;
    --app-metric-border: #2d3344;
    --app-chip-bg: #1e293b;
    --app-chip-text: #bfdbfe;
    --app-chip-border: #334155;
    --app-chip-active-bg: #2563eb;
    --app-chip-active-text: #ffffff;
    --app-plot-top: #60a5fa;
    --app-plot-other: #475569;
}

[data-theme="dark"] .gradio-container {
    --body-background-fill: #0f1117;
    --block-background-fill: #1a1d27;
    --block-border-color: #2d3344;
    --block-label-text-color: #9ca3af;
    --block-title-text-color: #f3f4f6;
    --body-text-color: #e5e7eb;
    --input-background-fill: #12151f;
    --button-secondary-background-fill: #252a3a;
    --button-secondary-text-color: #e5e7eb;
    --table-even-background-fill: #12151f;
    --table-odd-background-fill: #1a1d27;
    --border-color-primary: #2d3344;
}

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    background: var(--app-page-bg) !important;
}
.draw-panel .wrap { min-height: 0 !important; }
.draw-panel canvas { touch-action: none; }
.hero {
    text-align: center;
    padding: 0.25rem 0 1rem;
}
.hero-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.35rem;
}
.hero-badge {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--app-badge-text);
    background: var(--app-badge-bg);
    border-radius: 999px;
    padding: 0.25rem 0.65rem;
}
.hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem;
    color: var(--app-text-strong);
}
.hero p { color: var(--app-text-muted); margin: 0; }
.panel {
    border: 1px solid var(--app-panel-border);
    border-radius: 16px;
    padding: 1rem;
    background: var(--app-panel-bg);
}
.result-card {
    text-align: center;
    border-radius: 16px;
    padding: 1.1rem 1rem;
    background: var(--app-result-bg);
    border: 1px solid var(--app-result-border);
    min-height: 210px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.result-card.low {
    background: var(--app-result-low-bg);
    border-color: var(--app-result-low-border);
}
.result-card.empty {
    background: var(--app-result-empty-bg);
    border-color: var(--app-result-empty-border);
    min-height: 120px;
}
.result-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--app-text-sub);
    margin: 0 0 0.35rem;
}
.result-digit {
    font-size: 5rem;
    line-height: 1;
    font-weight: 800;
    margin: 0;
    color: var(--app-digit);
}
.result-card.low .result-digit { color: var(--app-digit-low); }
.result-confidence { margin: 0.45rem 0 0; font-size: 1.05rem; color: var(--app-text-body); }
.result-sub { margin: 0.2rem 0 0; font-size: 0.92rem; color: var(--app-text-sub); }
.warn { margin: 0.65rem 0 0; font-size: 0.88rem; color: var(--app-warn); }
.metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.55rem;
}
.metric {
    background: var(--app-metric-bg);
    border: 1px solid var(--app-metric-border);
    border-radius: 12px;
    padding: 0.55rem 0.65rem;
    text-align: center;
}
.metric span {
    display: block;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--app-text-sub);
}
.metric strong { font-size: 1rem; color: var(--app-text-strong); }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.2rem; }
.chip {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.32rem 0.7rem;
    font-size: 0.88rem;
    background: var(--app-chip-bg);
    color: var(--app-chip-text);
    border: 1px solid var(--app-chip-border);
}
.chip.active {
    background: var(--app-chip-active-bg);
    color: var(--app-chip-active-text);
    border-color: var(--app-chip-active-bg);
    font-weight: 600;
}
.example-row button { min-width: 2.1rem !important; font-weight: 700 !important; }
.preview-pair { gap: 0.75rem !important; }
.theme-switch label { margin-bottom: 0 !important; }
@media (max-width: 900px) {
    .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .hero-top { flex-direction: column; align-items: stretch; }
}
"""

THEME_BOOTSTRAP_JS = """
() => {
  const saved = localStorage.getItem("digit-recognizer-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const mode = saved || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", mode);
  const selectBrush = () => {
    document.querySelector(".draw-panel button[aria-label='Brush']")?.click();
  };
  selectBrush();
  setTimeout(selectBrush, 200);
  setTimeout(selectBrush, 600);
  return mode === "dark" ? "Dark" : "Light";
}
"""

# Sending a new canvas value from Python leaves the Sketchpad stuck on its loader until a
# second update arrives, so clearing uses the editor's own "Clear canvas" tool instead.
CLEAR_CANVAS_JS = """
() => {
  document.querySelector(".draw-panel button[aria-label='Clear canvas']")?.click();
  const selectBrush = () => {
    document.querySelector(".draw-panel button[aria-label='Brush']")?.click();
  };
  setTimeout(selectBrush, 50);
  setTimeout(selectBrush, 300);
}
"""

# Runs in <head> before first paint, so dark mode never flashes light first.
THEME_HEAD = """
<script>
(function () {
  const saved = localStorage.getItem("digit-recognizer-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const mode = saved || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", mode);
})();
</script>
"""

THEME_APPLY_JS = """
(mode) => {
  const theme = mode === "Dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("digit-recognizer-theme", theme);
}
"""


def _plot_colors(dark: bool) -> dict[str, str]:
    if dark:
        return {"top": "#60a5fa", "other": "#475569"}
    return {"top": "#1d4ed8", "other": "#cbd5e1"}


def _apply_theme_mode(mode: str):
    return gr.BarPlot(color_map=_plot_colors(mode == "Dark"))


with gr.Blocks(title="Digit Recognizer") as demo:
    history_state = gr.State([])

    last_input_state = gr.State(None)

    with gr.Row(elem_classes=["hero-top"]):
        gr.HTML('<div class="hero-badge">MNIST CNN · Gradio</div>')
        theme_toggle = gr.Radio(
            choices=["Light", "Dark"],
            value="Light",
            label="Theme",
            elem_classes=["theme-switch"],
            scale=0,
            min_width=160,
        )

    gr.HTML("""
        <div class="hero">
            <h1>Handwritten Digit Recognizer</h1>
            <p>Draw, upload an example, or pick a sample digit — see what the model actually reads.</p>
        </div>
        """)

    empty_df = _confidence_df(np.zeros(10, dtype=np.float32), 0)

    with gr.Tabs():
        with gr.Tab("Recognize"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["panel", "draw-panel"]):
                        canvas = gr.Sketchpad(
                            label="Draw here",
                            type="numpy",
                            image_mode="RGB",
                            canvas_size=CANVAS_SIZE,
                            value=_blank_canvas(),
                            brush=gr.Brush(colors=["#111827"], default_size=BRUSH_SIZE),
                            eraser=gr.Eraser(default_size=ERASER_SIZE),
                        )
                        gr.Markdown("**Try a sample** (loads MNIST digit onto canvas)")
                        with gr.Row(elem_classes=["example-row"]):
                            example_btns = [
                                gr.Button(str(d), size="sm", variant="secondary")
                                for d in range(10)
                            ]
                        with gr.Row():
                            predict_btn = gr.Button(
                                "Predict", variant="primary", scale=2
                            )
                            clear_btn = gr.Button(
                                "Clear canvas",
                                variant="secondary",
                                scale=1,
                                elem_id="clear-btn",
                            )
                        with gr.Row():
                            live_predict = gr.Checkbox(
                                label="Live predict",
                                value=True,
                                info="Predict after each stroke.",
                            )
                            use_tta = gr.Checkbox(
                                label="TTA mode",
                                value=False,
                                info=f"Average {TTA_VIEW_COUNT} shifted and rotated views for tougher strokes.",
                            )

                with gr.Column(scale=4, min_width=320):
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
                            color_map=_plot_colors(False),
                            y_lim=[0, 100],
                            height=240,
                            x_title="Digit",
                            y_title="Confidence (%)",
                        )
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("**Wrong prediction?** Save the correct label to improve the model.")
                        with gr.Row():
                            correct_label = gr.Dropdown(
                                choices=[str(d) for d in range(10)],
                                label="Correct digit",
                                scale=1,
                            )
                            save_correction_btn = gr.Button(
                                "Save correction", variant="secondary", scale=1
                            )
                        correction_status = gr.Markdown()

        with gr.Tab("History"):
            with gr.Group(elem_classes=["panel"]):
                history_table = gr.Dataframe(
                    headers=["time", "digit", "confidence", "margin"],
                    datatype=["str", "str", "number", "number"],
                    label="Predictions confirmed with the Predict button",
                    interactive=False,
                )
                clear_history_btn = gr.Button("Clear history", variant="secondary")

    with gr.Accordion("Working/Process", open=False):
        gr.Markdown(f"""
            **Drawing:** one digit, thick strokes, centered on the canvas.

            **Preprocessing:** autocontrast → crop ink → thicken thin strokes → scale longest side to 20px → center by mass in 28×28 (MNIST layout).

            **TTA mode:** runs {TTA_VIEW_COUNT} slightly shifted and rotated (±10°) versions and averages probabilities — slower but stabler on messy strokes.

            **Confidence:** temperature-calibrated, so 80% means about 80% of such predictions are right on MNIST. Below {REJECT_CONFIDENCE:.0f}% the app answers "unclear".

            **Metrics:** *margin* is gap between top two classes; *entropy* measures overall uncertainty.

            **History:** only predictions confirmed with the Predict button are logged, not live updates while drawing.
            """)

    outputs = [
        result_html,
        preview,
        raw_preview,
        confidence_plot,
        top3_html,
        metrics_html,
        history_table,
        history_state,
        last_input_state,
    ]
    assert len(outputs) == OUTPUT_COUNT

    predict_inputs = [canvas, history_state, use_tta]

    predict_btn.click(fn=predict, inputs=predict_inputs, outputs=outputs)
    clear_btn.click(
        fn=clear_canvas,
        inputs=[history_state],
        outputs=outputs,
        js=CLEAR_CANVAS_JS,
    )
    clear_history_btn.click(fn=clear_history, outputs=[history_table, history_state])
    save_correction_btn.click(
        fn=save_correction,
        inputs=[correct_label, last_input_state],
        outputs=[correction_status],
    )

    for digit, btn in enumerate(example_btns):
        btn.click(
            fn=lambda h, t, live, d=digit: load_example(d, h, t, live),
            inputs=[history_state, use_tta, live_predict],
            outputs=[canvas, *outputs],
        )

    def maybe_predict(image, live, history, tta):
        # Skipping on a blank canvas keeps the "Canvas cleared" message visible.
        if live and not is_blank(image):
            return predict(image, history, tta, record=False)
        return _skip_outputs()

    canvas.change(
        fn=maybe_predict,
        inputs=[canvas, live_predict, history_state, use_tta],
        outputs=outputs,
    )

    demo.load(fn=None, js=THEME_BOOTSTRAP_JS, outputs=[theme_toggle])
    theme_toggle.change(
        fn=_apply_theme_mode,
        inputs=[theme_toggle],
        outputs=[confidence_plot],
        js=THEME_APPLY_JS,
    )


if __name__ == "__main__":
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        share=False,
        inbrowser=True,
        theme=LIGHT_THEME,
        css=CUSTOM_CSS,
        head=THEME_HEAD,
    )