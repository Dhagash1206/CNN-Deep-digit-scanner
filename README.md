# CNN Deep Digit Scanner

A handwritten digit recognizer built with a CNN trained on MNIST, served through an interactive Gradio web app. Draw a digit, upload one, or load a sample — the app preprocesses it, runs inference, and shows predicted digit, confidence, margin, entropy, and a live confidence chart.

## Features

- **CNN trained on MNIST** — 3 convolutional blocks (32 → 64 → 128 filters) with batch norm, max pooling, dropout, and a dense classifier head.
- **Interactive canvas** — draw digits directly in the browser (Gradio Sketchpad).
- **Sample loader** — one-click buttons to load a real MNIST example for each digit (0–9).
- **Smart preprocessing** — auto-inverts/grayscales input, crops to ink, centers and pads the digit to match MNIST's 28×28 layout.
- **Test-Time Augmentation (TTA)** — optional mode that averages predictions across 7 shifted versions of the digit for more robust results on messy strokes.
- **Live predict** — predicts automatically as you draw.
- **Confidence metrics** — top-3 digit chips, per-digit confidence bar chart, prediction margin, and entropy-based "certainty" score.
- **Prediction history** — table of recent predictions (time, digit, confidence, margin) with clear option.
- **Light/Dark theme** — toggle with preference saved in the browser.

## Project Structure

```
CNN-Deep-digit-scanner-main/
├── train_model.py     # Trains the CNN on MNIST and saves mnist_cnn_model.keras
├── app.py              # Gradio web app for interactive digit recognition
├── requirements.txt    # Python dependencies
└── README.md
```

- **Training side** (`train_model.py`): loads MNIST → builds and trains the CNN → evaluates → saves `mnist_cnn_model.keras`.
- **Serving side** (`app.py`): loads the saved model once at startup → Gradio UI captures input → preprocessing pipeline reshapes it to MNIST format → model predicts → results rendered back into the UI.

### Preprocessing pipeline

```
Input (canvas/upload/sample, RGB or RGBA)
   → flatten alpha onto white background
   → convert to grayscale, invert (ink becomes bright)
   → autocontrast
   → threshold ink pixels, discard if too few (MIN_INK_PIXELS)
   → crop to ink bounding box
   → pad proportionally (25% margin), square canvas
   → resize to 20x20, paste centered into 28x28 canvas
   → normalize to [0, 1], reshape to (1, 28, 28, 1)
```

This mirrors how MNIST digits are framed (centered, padded, low-resolution), which is why it matters for prediction accuracy.

### CNN model

| Layer                  | Output shape   | Params    |
|-------------------------|---------------|-----------|
| Input                   | 28×28×1        | –         |
| Conv2D(32, 3×3) + ReLU   | 28×28×32       | 320       |
| BatchNormalization       | 28×28×32       | 128       |
| MaxPooling2D(2×2)        | 14×14×32       | –         |
| Conv2D(64, 3×3) + ReLU   | 14×14×64       | 18,496    |
| BatchNormalization       | 14×14×64       | 256       |
| MaxPooling2D(2×2)        | 7×7×64         | –         |
| Conv2D(128, 3×3) + ReLU  | 7×7×128        | 73,856    |
| BatchNormalization       | 7×7×128        | 512       |
| Flatten                  | 6,272          | –         |
| Dense(256) + ReLU        | 256            | 1,605,888 |
| Dropout(0.4)             | 256            | –         |
| Dense(10) + Softmax      | 10             | 2,570     |

**Total: ~1.70M parameters**

- **Optimizer**: Adam (lr=0.001)
- **Loss**: sparse categorical crossentropy
- **Callbacks**: `EarlyStopping` (patience=3, restores best weights) and `ReduceLROnPlateau` (halves LR after 2 stagnant epochs)
- **Training**: 15 epochs max, batch size 128, 10% validation split

### Test-Time Augmentation (TTA)

When enabled, the centered 28×28 digit is shifted by 1 pixel in 7 directions (`(0,0), (0,±1), (±1,0), (1,1), (-1,-1)`), each shifted copy is run through the model, and the resulting probability vectors are averaged. This trades latency for robustness on off-center or noisy strokes.

## Requirements

- Python 3.9+
- Dependencies from `requirements.txt`:
  - `tensorflow>=2.13.0`
  - `gradio>=4.0.0`
  - `numpy>=1.24.0`
  - `Pillow>=9.0.0`
- `pandas` (used by `app.py` for confidence/history tables; install alongside the above if not already present)

Install everything with:

```bash
pip install -r requirements.txt pandas
```

## Usage

### 1. Train the model

```bash
python train_model.py
```

This downloads MNIST, trains the CNN (up to 15 epochs, with early stopping and learning-rate reduction), evaluates it on the test set, and saves the trained model to `mnist_cnn_model.keras`.

### 2. Run the app

```bash
python app.py
```

The app launches at `http://localhost:7860`. From there you can:
- Draw a digit on the canvas and click **Predict** (or leave **Live predict** on).
- Click a digit button (0–9) to load a real MNIST sample onto the canvas.
- Toggle **TTA mode** for more stable predictions on rough strokes.
- Switch between **Light** and **Dark** themes.
- View recent predictions under the **History** tab.

> Note: `app.py` requires `mnist_cnn_model.keras` to exist — run `train_model.py` first, or it will raise a `FileNotFoundError`.

## How It Works

1. **Drawing → Image**: the canvas image (or uploaded example) is extracted and converted to grayscale, inverted so ink is bright on a dark background.
2. **Centering**: ink pixels are located, cropped to their bounding box, padded proportionally, and resized/pasted into a 28×28 canvas — matching MNIST's format.
3. **Inference**: the 28×28 array is normalized to [0, 1] and fed to the CNN. With TTA enabled, 7 shifted copies are predicted and averaged.
4. **Metrics**: confidence, margin (gap between top-2 classes), and entropy are computed to flag low-confidence or ambiguous predictions.

## License

No license file included — add one if you plan to distribute this project.
