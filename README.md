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
10. [Improving the Model with Corrections](#improving-the-model-with-corrections)
11. [Hyperparameter Tuning](#hyperparameter-tuning)
12. [Testing](#testing)
13. [Troubleshooting](#troubleshooting)

---

## System Overview

```
 TrainModel.py                               DigitApp.py
 ┌───────────────────────┐                   ┌─────────────────────────────────────────┐
 │ MNIST (60k train/10k  │                   │ Gradio Sketchpad (520x520 RGB)          │
 │ test) -> normalize    │                   │   -> grayscale + invert + autocontrast  │
 │ -> CNN (3 conv blocks)│  DigitModel.keras │   -> crop to ink, thicken thin strokes  │
 │ -> EarlyStopping /    │                   │   -> fit 20x20, center of mass in 28x28 │
 │    ReduceLROnPlateau  │  + calibration    │   -> CNN forward pass (optional TTA)    │
 │ -> calibrate + save   │ ───────────────>  │   -> calibrated prediction, history     │
 └───────────────────────┘                   └──────────────────┬──────────────────────┘
            ^                                                   │ "Save correction"
            │                                                   v
            └──── FineTune.py <──────────────────── Feedback/<digit>/*.png
```

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `TrainModel.py` | Loads MNIST, trains the CNN, prints a confusion matrix, fits the calibration temperature and saves the model, calibration and sample digits. |
| `FineTune.py` | Fine-tunes the saved model on corrections collected in the app. |
| `TuneModel.py` | Optional Optuna search over training hyperparameters (validation split only). |
| `Calibration.py` | Temperature scaling, NLL and expected calibration error. Pure NumPy. |
| `DigitApp.py` | Loads the saved model and launches the Gradio application (inference, UI, theming). |
| `Preprocessing.py` | Canvas-to-MNIST conversion: polarity, cropping, stroke-width normalization, 20x20 fit and centering. Pure NumPy/Pillow. |
| `requirements.txt` / `requirements-dev.txt` | Runtime and test dependencies. |
| `Tests/` | Pytest suite for preprocessing and calibration (no TensorFlow needed). |
| `ModelTemperature.json` | Fitted softmax temperature, read by the app (defaults to 1.0 if missing). |
| `Assets/SampleDigits.png` | One MNIST test digit per class, so the sample buttons work offline. |
| `.gitignore` | Excludes model artifacts (`*.keras`, `*.h5`, `*.pb`) except `DigitModel.keras`, plus `Feedback/`, datasets, virtual environments and caches. |
| `DigitModel.keras` | Pre-trained model (1.6 MB, weights only), committed so the app runs right after cloning. |

The pre-trained model is committed, so `python DigitApp.py` works without training first. Run `python TrainModel.py` to regenerate it.

---

## Requirements

- Python 3.10 or newer (the code uses the `X | None` type syntax). Stable TensorFlow has no Python 3.14 wheels yet; on 3.14 install the release candidate with `pip install --pre -r requirements.txt`.
- Dependencies from `requirements.txt`:
  - `tensorflow >= 2.16.0`
  - `keras >= 3.8.0` (for `RandomShear` augmentation)
  - `gradio == 6.30.0`, pinned to the tested version (the app uses `gr.skip()` and passes `theme`/`css`/`head` to `launch()`)
  - `numpy >= 1.24.0`
  - `pandas >= 2.0.0`
  - `Pillow >= 9.0.0`
- Internet access for training, because `tf.keras.datasets.mnist` downloads the dataset when it is not cached. The app itself runs offline.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain the model; a pre-trained DigitModel.keras is included
python TrainModel.py

# 4. Launch the application
python DigitApp.py
```

The app starts on `http://127.0.0.1:7860` and attempts to open a browser tab automatically. If the model file is missing, `DigitApp.py` raises a `FileNotFoundError` at import time; run step 3 in that case.

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
| Block 3 | Conv2D(128, 3x3, ReLU, same) | 7x7x128 | |
| | BatchNormalization | 7x7x128 | |
| | MaxPooling2D(2x2) | 3x3x128 | Shrinks the head input |
| Head | Flatten | 1152 | |
| | Dense(256, ReLU) | 256 | |
| | Dropout(0.4) | 256 | Regularization |
| | Dense(10, Softmax) | 10 | One probability per digit |

- Total parameters: 391,306, most of them in the first Dense layer (1152 x 256 + 256 = 295,168).
- BatchNormalization is applied after each convolution to stabilize training; dropout is applied only in the classifier head.

---

## Training Pipeline

Implemented in `TrainModel.py`.

- **Data:** MNIST via `tf.keras.datasets.mnist` (60,000 training and 10,000 test images, 28x28 grayscale).
- **Preprocessing:** pixel values cast to `float32` and scaled to `[0, 1]`; a channel axis is added to produce `(N, 28, 28, 1)`.
- **Optimizer and loss:** Adam (learning rate `0.001`), `sparse_categorical_crossentropy` with integer labels, `accuracy` metric.
- **Augmentation (training batches only):**
  - Stroke width (`random_stroke_width`): 25% of images are eroded and 25% dilated by about 1px, using 3x3 morphology at 2x resolution, so the model sees pens thinner and thicker than MNIST's.
  - Geometric: random rotation (±14°), horizontal shear (0.15), zoom (±12%) and translation (±8%), all zero-filled.
  - The validation split stays unaugmented.
- **Schedule:** up to 20 epochs, batch size 128, a fixed 10% of the training set held out for validation. `tf.keras.utils.set_random_seed(42)` makes runs repeatable.
- **Callbacks:**
  - `EarlyStopping` monitors `val_loss`, patience 5, restores the best weights.
  - `ModelCheckpoint` saves the best epoch to `BestCheckpoint.keras`, so a crashed run keeps its progress without overwriting the app's model.
  - `ReduceLROnPlateau` monitors `val_loss`, halves the learning rate after 2 stagnant epochs.
- **Evaluation:** test accuracy and loss, a 10x10 confusion matrix, per-class accuracy and the five most confused pairs.
- **Calibration:** a softmax temperature is fitted on the validation split by minimizing negative log-likelihood, then NLL and expected calibration error (ECE) are reported on the test split, before and after scaling. It is saved to `ModelTemperature.json`.
- **Output:** the model is saved without optimizer state through an uncompiled copy (`save_for_inference`), because Keras 3 ignores `include_optimizer=False`. This cuts the file from 4.8 MB to 1.6 MB.

The committed model scored 99.23% test accuracy (weakest classes: 7 at 98.44% and 9 at 98.51%; most confused pairs 9->4 and 7->2). On MNIST the CNN is already well calibrated: the fitted temperature is 1.057 and test ECE is 0.15% before scaling and 0.17% after. Calibration matters more after fine-tuning on corrections, and the app applies whatever temperature the last run produced.

| Thinned MNIST test digits (1,000 samples) | Old model | New model, no stroke fix | New model + stroke fix |
|------|------|------|------|
| Original strokes | 99.3% | 99.5% | 99.6% |
| Eroded with a 25px filter at 380px (about 1.5px wide at 28x28) | 81.9% | 95.6% | 98.9% |

---

## Inference Pipeline

The canvas image differs from MNIST in stroke width, position and polarity, so `DigitApp.py` normalizes it into MNIST layout before prediction.

### 1. Preprocessing (`preprocess`, helpers in `Preprocessing.py`)

1. **Extraction** (`extract_image`): accepts a raw array or a Gradio editor dict (`composite`, falling back to the first layer).
2. **Polarity and contrast** (`to_grayscale_inverted`): flattens RGBA onto white, converts to grayscale, inverts to white-ink-on-black, then applies `ImageOps.autocontrast` with a 1% cutoff.
3. **Ink detection and centering** (`center_digit`):
   - Pixels above `INK_THRESHOLD` (15) count as ink; fewer than `MIN_INK_PIXELS` (40) is rejected as "no stroke".
   - The image is cropped to the ink bounding box.
   - **Stroke normalization** (`normalize_stroke`): scaling a 520px drawing down to 20px leaves a 22px brush stroke about 1.5px wide, much thinner than MNIST (median about 2.6px). The stroke width is estimated as `2 x area / boundary` at 4x the final size. If it would land below `MIN_STROKE_WIDTH` (2.0px), the glyph is dilated toward `TARGET_STROKE_WIDTH` (2.6px). On MNIST test digits thinned to canvas-like strokes, this raised accuracy from 81.9% to 97.7%.
   - The crop is resized with Lanczos resampling so its longest side is `DIGIT_BOX` (20) pixels, keeping the aspect ratio.
   - The glyph is placed on a 28x28 black frame and shifted so its center of mass sits at the frame center, matching how MNIST itself was prepared. The shift is clamped so no ink leaves the frame.
4. **Normalization:** scaled to `[0, 1]` and reshaped to `(1, 28, 28, 1)`.

### 2. Prediction (`_run_inference`)

- **Single pass:** one forward pass through `_forward`, which calls `model(x, training=False)` directly instead of `model.predict()` to avoid its per-call overhead during live predict, then applies the calibration temperature. Returns the 10-class probability vector.
- **Test-time augmentation (TTA):** builds 9 views of the input: 7 shifts from `TTA_SHIFTS` (no shift, plus one-pixel shifts along each axis and both diagonals) via `shift_digit`, and 2 rotations from `TTA_ROTATIONS` (±10°) via `rotate_digit`, all zero-filled. It runs them as one batch and averages the probabilities. This costs roughly 9x the compute but stabilizes predictions on off-center or irregular strokes.

### 3. Derived metrics

| Metric | Definition |
|--------|------------|
| Confidence | Highest temperature-calibrated class probability, as a percentage. |
| Margin | Gap between the top two class probabilities, in percentage points. |
| Entropy | Shannon entropy of the distribution (probabilities clipped at `1e-9`). Lower is more certain. |
| Certainty | `100 - entropy * 25`, clamped to `[0, 100]`. A display heuristic, not a calibrated probability. |

A prediction is flagged with a warning when confidence is below `LOW_CONFIDENCE` (55%) or the margin is under 12 points. Below `REJECT_CONFIDENCE` (40%) the card shows "?" with the best guess, rather than claiming a digit.

---

## Application Interface

Built with `gradio.Blocks`, served on `127.0.0.1:7860` by default. Set `DIGIT_APP_HOST=0.0.0.0` to expose it on your network, and `DIGIT_APP_PORT` to change the port.

- **Recognize tab**
  - Sketchpad canvas (520x520) with a brush (default size 22) and eraser (default size 44).
  - Sample buttons (0-9) load a bundled MNIST test image onto the canvas and run a prediction.
  - **Clear canvas** uses the Sketchpad's own clear tool in the browser, because setting the canvas value from Python leaves the Sketchpad stuck on its loader until a second update arrives.
  - **Live predict** re-runs inference after each stroke; **TTA mode** switches to the 9-view ensemble.
  - Output panel: predicted digit, metrics cards, top-3 chips, the processed 28x28 model input beside the raw capture, and a per-digit confidence bar chart.
  - **Wrong prediction?** panel: pick the correct digit and press **Save correction** to store the 28x28 model input under `Feedback/<digit>/`.
- **History tab:** the most recent 25 predictions confirmed with the **Predict** button (time, digit, confidence, margin), held in per-session `gr.State` and clearable with a button. Live updates while drawing are not logged, so half-drawn digits don't fill the table.
- **Theming:** light and dark modes via a theme toggle, persisted in the browser through `localStorage`; the chart colors are updated to match. The saved theme is applied by a script in `<head>` (`THEME_HEAD`), so dark mode never flashes light first.
- **Working/Process accordion:** in-app summary of the drawing, preprocessing and metric definitions.

---

## Configuration Constants

Defined at the top of `DigitApp.py`; edit them directly to tune behavior.

| Constant | Default | Effect |
|----------|---------|--------|
| `MODEL_PATH` | `DigitModel.keras` | Model file loaded at startup. |
| `CANVAS_SIZE` | `(520, 520)` | Sketchpad dimensions. |
| `BRUSH_SIZE` / `ERASER_SIZE` | `22` / `44` | Default tool sizes. |
| `PREVIEW_SIZE` | `168` | Display size of the preview images. |
| `CALIBRATION_PATH` | `ModelTemperature.json` | Temperature file; missing means 1.0. |
| `SERVER_HOST` / `SERVER_PORT` | `127.0.0.1` / `7860` | From `DIGIT_APP_HOST` / `DIGIT_APP_PORT`. |
| `LOW_CONFIDENCE` | `55.0` | Confidence percentage below which a warning appears. |
| `REJECT_CONFIDENCE` | `40.0` | Confidence percentage below which the result is shown as "?". |
| `HISTORY_LIMIT` | `25` | Maximum stored history entries. |
| `TTA_SHIFTS` | 7 `(dx, dy)` pairs | Shifts averaged in TTA mode. |
| `TTA_ROTATIONS` | `[-10, 10]` | Rotation angles (degrees) averaged in TTA mode. |

Preprocessing constants live at the top of `Preprocessing.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `INK_THRESHOLD` | `15` | Minimum pixel value counted as ink. |
| `MIN_INK_PIXELS` | `40` | Minimum ink pixels required to attempt a prediction. |
| `DIGIT_BOX` | `20` | Size in pixels of the longest side of the digit inside the 28x28 frame. |
| `MIN_STROKE_WIDTH` / `TARGET_STROKE_WIDTH` | `2.0` / `2.6` | Strokes thinner than the minimum (in 28x28 pixels) are thickened toward the target. |

The launch call in the `__main__` block controls `share` and `inbrowser`.

---

## Improving the Model with Corrections

1. In the app, when a prediction is wrong, pick the correct digit and press **Save correction**. Each save writes the exact 28x28 image the model saw to `Feedback/<digit>/<timestamp>.png`. The folder is git-ignored, since it is your handwriting.
2. After at least 10 corrections, run:

   ```bash
   python FineTune.py
   ```

   It mixes each correction (repeated 20 times per epoch) with 30 random MNIST images per correction, so the model does not forget MNIST. It then fine-tunes for 5 epochs at learning rate `1e-4` with the same augmentation as training, refits the calibration temperature and overwrites `DigitModel.keras`.
3. It prints MNIST test accuracy before and after. If that drops noticeably, restore the model with `git checkout DigitModel.keras ModelTemperature.json`. The reported corrections accuracy is measured on the training samples themselves, so treat it as optimistic.

---

## Hyperparameter Tuning

```bash
pip install -r requirements-dev.txt
python TuneModel.py
```

`TuneModel.py` runs 40 Optuna trials (`TRIALS`) of up to 12 epochs each (`MAX_EPOCHS`). On a CPU, an epoch takes about 15 seconds, so a full run takes 1-2 hours. The study is saved to `TuningStudy.db` (git-ignored), so rerunning the script resumes an interrupted search.

### Parameters, ranked by expected impact

Trial 0 is always the current `TrainModel.py` configuration (the start values), so every other trial is measured against it.

| Rank | Parameter | Range | Start | Why this range |
|------|-----------|-------|-------|----------------|
| 1 | `learning_rate` | 3e-4 to 3e-3 (log) | 1e-3 | The most sensitive setting for Adam with BatchNorm. Above 3e-3 training diverges early; below 3e-4 it does not converge within 12 epochs. |
| 2 | `augmentation_strength` | 0.5 to 1.5 | 1.0 | Multiplies rotation, shear, zoom and shift together. It trades MNIST accuracy against robustness on canvas digits. |
| 3 | `stroke_prob` | 0 to 0.35 | 0.25 | The chance of thinning, and separately of thickening, each training image. This setting closed most of the gap on thin canvas strokes. |
| 4 | `dropout` | 0.2 to 0.6 | 0.4 | Regularizes the 295k-weight dense layer, which holds most of the parameters. |
| 5 | `weight_decay` | 1e-6 to 1e-3 (log) | 1e-6 | AdamW decay. The lower bound is effectively the current "no decay". |
| 6 | `batch_size` | 64, 128, 256 | 128 | Interacts with the learning rate: larger batches usually need a higher one. |
| 7 | `dense_units` | 128, 256, 384 | 256 | Head capacity. MNIST rarely needs more. |
| 8 | `base_filters` | 24, 32, 48 | 32 | Width of the convolutional blocks (x1, x2, x4). Costs time more than it adds accuracy. |

### Search method

- **Bayesian (TPE):** a multivariate TPE sampler with seed 42. With 8 parameters, grid search would need thousands of runs, and random search wastes trials on regions TPE learns to skip.
- **40 trials:** about 5 per parameter, which is enough for TPE to settle after its random warm-up.
- **Pruning:** `MedianPruner` (5 startup trials, 3 warm-up epochs) stops a trial whose validation loss falls behind the median of earlier trials at the same epoch.
- **Importance:** the script prints the parameter importances at the end. If a parameter scores near 0, fix it and narrow the search.

### Validation plan

- **Hold-out split, not cross-validation:** a 6,000-image validation split gives about ±0.1% accuracy noise, smaller than the differences being compared. 5-fold cross-validation would cost 5 times as much for little gain. If the top three trials are within 0.05% of each other, re-run them with 3 different seeds before picking one.
- **Same split as training:** the permutation matches `TrainModel.make_datasets` (seed 42).
- **Objective:** the mean of validation loss on clean digits and on digits eroded by about 1px. The eroded set stands in for thin canvas strokes. Loss is used rather than accuracy because at 99.4% accuracy, trials differ by only a handful of images.
- **Early stopping:** `EarlyStopping` (patience 3, best weights restored) and `ReduceLROnPlateau` (patience 2), both on validation loss.
- **No leakage:**
  - The MNIST test set is never loaded into the search.
  - Augmentation applies only to training batches.
  - `Feedback/` corrections are not used.
  - Report test accuracy once, after retraining with the chosen values.

### Overfitting and underfitting

The script stores `val_accuracy`, `thin_val_accuracy`, `train_val_gap` and `epochs_run` with each trial.

| Sign | Likely cause | Adjust |
|------|--------------|--------|
| Validation loss rises while training loss falls; early stopping fires within 3-5 epochs | Overfitting | Raise `dropout` or `weight_decay`; reduce `dense_units` |
| Clean validation accuracy is high but thin validation accuracy is more than 1.5 points lower | Overfitting to MNIST stroke style | Raise `stroke_prob` or `augmentation_strength` |
| Training and validation accuracy both below 99% and still climbing at epoch 12 | Underfitting or too few epochs | Raise `MAX_EPOCHS` or `learning_rate`; reduce augmentation |
| Training accuracy well below validation accuracy late in training | Augmentation too strong | Lower `augmentation_strength` |
| Loss is NaN or stuck near 2.3 | Learning rate too high | Lower the upper bound of the `learning_rate` range |

### Applying the result

Copy the best values into `TrainModel.py`. `TuneModel.py` already uses `AdamW` instead of `Adam`, so switch the optimizer if `weight_decay` comes out above about 1e-5. Then run `python TrainModel.py` and compare its test accuracy with the current 99.40%.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers shifting, blank detection, the 20x20 fit and center of mass, stroke-width estimation and thickening, and temperature fitting, save and load. It imports only `Preprocessing.py` and `Calibration.py`, so it runs in under a second without loading TensorFlow.

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `FileNotFoundError: Model not found` | Run `python TrainModel.py` first. |
| "No stroke detected" message | The drawing has fewer than 40 ink pixels; draw thicker or larger. |
| Frequent low-confidence warnings | Center the digit, draw it larger, or enable TTA mode. |
| Port 7860 already in use | Run with `DIGIT_APP_PORT=7861 python DigitApp.py`. |
| App not reachable from another device | It binds to `127.0.0.1` by default; set `DIGIT_APP_HOST=0.0.0.0`. |
| TensorFlow log noise or oneDNN differences | The scripts already set `TF_CPP_MIN_LOG_LEVEL=2` and `TF_ENABLE_ONEDNN_OPTS=0`. |
| API errors after upgrading Gradio | `DigitApp.py` relies on `gr.skip()`, `launch(theme=, css=, head=)` and the `Sketchpad` toolbar DOM (`aria-label='Brush'` and `aria-label='Clear canvas'`); reinstall the pinned version with `pip install -r requirements.txt`. |