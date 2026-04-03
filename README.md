# Handwritten Digit Recognizer

Single digit (0-9) recognition using CNN trained on MNIST + Gradio UI.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run

### Step 1 - Train the model (only once)
```bash
python train_model.py
```
- Downloads MNIST automatically
- Trains CNN (~2-5 mins on CPU)
- Saves model as `mnist_cnn_model.keras`
- Expected accuracy: 99%+

### Step 2 - Launch the app
```bash
python app.py
```
- Opens browser at http://localhost:7860
- Draw a digit, click Predict

---

## Project Structure

```
digit_recognizer/
├── train_model.py       # CNN training script
├── app.py               # Gradio UI
├── requirements.txt     # Dependencies
├── README.md            # This file
└── mnist_cnn_model.keras  # (generated after training)
```

---

## Model Architecture

```
Input (28x28x1)
  → Conv2D(32) + BN + MaxPool
  → Conv2D(64) + BN + MaxPool
  → Conv2D(128) + BN
  → Flatten
  → Dense(256) + Dropout(0.4)
  → Dense(10, softmax)
```

---

## Notes
- Draw digits large and centered
- Use thick brush strokes
- Single digit only (0-9)
