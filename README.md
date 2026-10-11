# CNN Deep Digit Scanner

A handwritten digit recognizer built on a convolutional neural network trained on MNIST, served through an interactive Gradio web interface. The project is split into two stages: an offline training script that produces a model artifact, and an inference application that loads the artifact and classifies digits drawn on a canvas in real time.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Repository Layout](#repository-layout)
3. [Requirements](#requirements)
4. [Quick Start](#quick-start)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Inference Pipeline](#inference-pipeline)
8. [Application Interface](#application-interface)
9. [Configuration Constants](#configuration-constants)
10. [Troubleshooting](#troubleshooting)

---

## System Overview

```
 train_model.py                              app.py
 ┌───────────────────────┐                   ┌─────────────────────────────────────────┐
 │ MNIST (60k train/10k  │                   │ Gradio Sketchpad (520x520 RGB)          │
 │ test) -> normalize    │                   │   -> grayscale + invert + autocontrast  │
 │ -> CNN (3 conv blocks)│  mnist_cnn_       │   -> crop to ink bounding box           │
 │ -> EarlyStopping /    │  model.keras      │   -> pad, resize 20x20, paste on 28x28  │
 │    ReduceLROnPlateau  │ ───────────────>  │   -> CNN forward pass (optional TTA)    │
 │ -> evaluate + save    │                   │   -> prediction, metrics, history       │
 └───────────────────────┘                   └─────────────────────────────────────────┘
```

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `train_model.py` | Loads MNIST, builds and trains the CNN, evaluates on the test split, saves `mnist_cnn_model.keras`. |
| `app.py` | Loads the saved model and launches the Gradio application (preprocessing, inference, UI, theming). |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Excludes model artifacts (`*.keras`, `*.h5`, `*.pb`), datasets, virtual environments and caches. |

The trained model file is intentionally not version controlled and must be generated locally.

---

## Requirements

- Python 3.10 or newer (the code uses the `X | None` type syntax; the bundled bytecode targets 3.12)
- Dependencies from `requirements.txt`:
  - `tensorflow >= 2.13.0`
  - `gradio >= 4.0.0`
  - `numpy >= 1.24.0`
  - `Pillow >= 9.0.0`
- `pandas` is imported directly by `app.py` and is installed transitively as a Gradio dependency.
- Internet access on first run, because `tf.keras.datasets.mnist` downloads the dataset when it is not cached.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (produces mnist_cnn_model.keras)
python train_model.py

# 4. Launch the application
python app.py
```

The app starts on `http://localhost:7860` and attempts to open a browser tab automatically. `app.py` raises a `FileNotFoundError` at import time if the model file is missing, so step 3 is mandatory.

---

## Model Architecture

Defined in `build_model()` as a Keras `Sequential` model with input shape `(28, 28, 1)`.

| Stage | Layer | Output Shape | Notes |
|-------|-------|--------------|-------|
| Block 1 | Conv2D(32, 3x3, ReLU, same) | 28x28x32 | |
| | BatchNormalization | 28x28x32 | |
| | MaxPooling2D(2x2) | 14x14x32 | |
| Block 2 | Conv2D(64, 3x3, ReLU, same) | 14x14x64 | |
| | BatchNormalization | 14x14x64 | |
| | MaxPooling2D(2x2) | 7x7x64 | |
| Block 3 | Conv2D(128, 3x3, ReLU, same) | 7x7x128 | No pooling after this block |
| | BatchNormalization | 7x7x128 | |
| Head | Flatten | 6272 | |
| | Dense(256, ReLU) | 256 | |
| | Dropout(0.4) | 256 | Regularization |
| | Dense(10, Softmax) | 10 | One probability per digit |

- Total parameters: approximately 1.70 million, most of them in the first Dense layer (6272 x 256).
- BatchNormalization is applied after each convolution to stabilize training; dropout is applied only in the classifier head.

---

## Training Pipeline

Implemented in `train_model.py`.

- **Data:** MNIST via `tf.keras.datasets.mnist` (60,000 training and 10,000 test images, 28x28 grayscale).
- **Preprocessing:** pixel values cast to `float32` and scaled to `[0, 1]`; a channel axis is added to produce `(N, 28, 28, 1)`.
- **Optimizer and loss:** Adam (learning rate `0.001`), `sparse_categorical_crossentropy` with integer labels, `accuracy` metric.
- **Schedule:** up to 15 epochs, batch size 128, 10% of the training set held out for validation.
- **Callbacks:**
  - `EarlyStopping` monitors `val_accuracy`, patience 3, restores the best weights.
  - `ReduceLROnPlateau` monitors `val_loss`, halves the learning rate after 2 stagnant epochs.
- **Output:** final test accuracy and loss are printed, and the model is saved to `mnist_cnn_model.keras`.

No data augmentation is applied during training. Reported accuracy depends on the run and hardware, so it is not fixed here; run the script to obtain current figures.

---

## Inference Pipeline

The canvas image differs from MNIST in stroke width, position and polarity, so `app.py` normalizes it into MNIST layout before prediction.

### 1. Preprocessing (`preprocess`)

1. **Extraction** (`_extract_image`): accepts a raw array or a Gradio editor dict (`composite`, falling back to the first layer).
2. **Polarity and contrast** (`_to_grayscale_inverted`): flattens RGBA onto white, converts to grayscale, inverts to white-ink-on-black, then applies `ImageOps.autocontrast` with a 1% cutoff.
3. **Ink detection and centering** (`_center_digit`):
   - Pixels above `INK_THRESHOLD` (15) count as ink; fewer than `MIN_INK_PIXELS` (40) is rejected as "no stroke".
   - The image is cropped to the ink bounding box and placed on a square canvas with a 25% margin.
   - The square is resized to 20x20 with Lanczos resampling and pasted at offset (4, 4) on a 28x28 black canvas, which reproduces the MNIST convention of a 20x20 digit inside a 28x28 frame.
4. **Normalization:** scaled to `[0, 1]` and reshaped to `(1, 28, 28, 1)`.

### 2. Prediction (`_run_inference`)

- **Single pass:** one forward pass, returns the 10-class probability vector.
- **Test-time augmentation (TTA):** builds 7 shifted copies of the input (offsets listed in `TTA_SHIFTS`: no shift, plus one-pixel shifts along each axis and both diagonals), zero-fills the vacated border via `_shift_digit`, runs them as one batch, and averages the probabilities. This costs roughly 7x the compute but stabilizes predictions on off-center or irregular strokes.

### 3. Derived metrics

| Metric | Definition |
|--------|------------|
| Confidence | Highest class probability, as a percentage. |
| Margin | Gap between the top two class probabilities, in percentage points. |
| Entropy | Shannon entropy of the distribution (probabilities clipped at `1e-9`). Lower is more certain. |
| Certainty | `100 - entropy * 25`, clamped to `[0, 100]`. A display heuristic, not a calibrated probability. |

A prediction is flagged with a warning when confidence is below `LOW_CONFIDENCE` (55%) or the margin is under 12 points.

---

## Application Interface

Built with `gradio.Blocks`, served on port 7860 and bound to `0.0.0.0`.

- **Recognize tab**
  - Sketchpad canvas (520x520) with a brush (default size 22) and eraser (default size 44).
  - Sample buttons (0-9) load the first matching MNIST test image onto the canvas and run a prediction.
  - **Live predict** re-runs inference after each stroke; **TTA mode** switches to the 7-view ensemble.
  - Output panel: predicted digit, metrics cards, top-3 chips, the processed 28x28 model input beside the raw capture, and a per-digit confidence bar chart.
- **History tab:** the most recent 25 predictions (time, digit, confidence, margin), held in per-session `gr.State` and clearable with a button.
- **Theming:** light and dark modes via a theme toggle, persisted in the browser through `localStorage`; the chart colors are updated to match.
- **Working/Process accordion:** in-app summary of the drawing, preprocessing and metric definitions.

---

## Configuration Constants

Defined at the top of `app.py`; edit them directly to tune behavior.

| Constant | Default | Effect |
|----------|---------|--------|
| `MODEL_PATH` | `mnist_cnn_model.keras` | Model file loaded at startup. |
| `CANVAS_SIZE` | `(520, 520)` | Sketchpad dimensions. |
| `BRUSH_SIZE` / `ERASER_SIZE` | `22` / `44` | Default tool sizes. |
| `PREVIEW_SIZE` | `168` | Display size of the preview images. |
| `INK_THRESHOLD` | `15` | Minimum pixel value counted as ink. |
| `MIN_INK_PIXELS` | `40` | Minimum ink pixels required to attempt a prediction. |
| `LOW_CONFIDENCE` | `55.0` | Confidence percentage below which a warning appears. |
| `HISTORY_LIMIT` | `25` | Maximum stored history entries. |
| `TTA_SHIFTS` | 7 `(dx, dy)` pairs | Shifts averaged in TTA mode. |

The launch call in the `__main__` block controls host, port, `share` and `inbrowser`.

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `FileNotFoundError: Model not found` | Run `python train_model.py` first. |
| "No stroke detected" message | The drawing has fewer than 40 ink pixels; draw thicker or larger. |
| Frequent low-confidence warnings | Center the digit, draw it larger, or enable TTA mode. |
| Port 7860 already in use | Change `server_port` in `demo.launch(...)`. |
| TensorFlow log noise or oneDNN differences | The scripts already set `TF_CPP_MIN_LOG_LEVEL=2` and `TF_ENABLE_ONEDNN_OPTS=0`. |
| API errors after upgrading Gradio | `app.py` uses `gr.BarPlot.update` and the `Sketchpad` brush API; pin a Gradio 4.x or 5.x version that supports them. |