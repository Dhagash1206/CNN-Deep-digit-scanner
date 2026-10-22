"""
Fine-tune the saved model on corrections collected from the app's "Save correction" button.
        python finetune.py
"""

import glob
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image

from train_model import (
    MODEL_PATH,
    build_augmenter,
    calibrate,
    load_data,
    random_stroke_width,
    report_per_class,
    save_for_inference,
)

FEEDBACK_DIR = "feedback"
MIN_SAMPLES = 10
# Each correction is repeated this many times per epoch so a few dozen drawings outweigh noise.
FEEDBACK_REPEATS = 20
# MNIST images mixed in per correction, to keep the model from forgetting the original data.
MNIST_PER_SAMPLE = 30
EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 64


def load_feedback(directory=FEEDBACK_DIR):
    images, labels = [], []
    for path in sorted(glob.glob(os.path.join(directory, "[0-9]", "*.png"))):
        digit = int(os.path.basename(os.path.dirname(path)))
        arr = np.array(Image.open(path).convert("L"), dtype="float32") / 255.0
        images.append(arr[..., np.newaxis])
        labels.append(digit)
    if not images:
        return np.zeros((0, 28, 28, 1), dtype="float32"), np.zeros((0,), dtype="int64")
    return np.stack(images), np.array(labels, dtype="int64")


def make_finetune_dataset(x_feedback, y_feedback, x_mnist, y_mnist, seed=42):
    rng = np.random.default_rng(seed)
    mnist_count = min(len(x_mnist), len(x_feedback) * MNIST_PER_SAMPLE)
    mnist_idx = rng.choice(len(x_mnist), mnist_count, replace=False)

    images = np.concatenate(
        [np.repeat(x_feedback, FEEDBACK_REPEATS, axis=0), x_mnist[mnist_idx]]
    )
    labels = np.concatenate(
        [np.repeat(y_feedback, FEEDBACK_REPEATS, axis=0), y_mnist[mnist_idx]]
    )

    augmenter = build_augmenter()
    return (
        tf.data.Dataset.from_tensor_slices((images, labels))
        .shuffle(len(images), seed=seed)
        .batch(BATCH_SIZE)
        .map(
            lambda batch, batch_labels: (
                augmenter(random_stroke_width(batch), training=True),
                batch_labels,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )


def finetune():
    tf.keras.utils.set_random_seed(42)

    x_feedback, y_feedback = load_feedback()
    print(f"[INFO] Corrections found: {len(x_feedback)}")
    if len(x_feedback) < MIN_SAMPLES:
        print(f"[INFO] Need at least {MIN_SAMPLES}; save more corrections in the app first.")
        return
    print("[INFO] Per digit:", np.bincount(y_feedback, minlength=10).tolist())

    (x_train, y_train), (x_test, y_test) = load_data()
    # Same 10% split as train_model.make_datasets, so calibration uses unseen MNIST.
    order = np.random.default_rng(42).permutation(len(x_train))
    val_count = int(len(x_train) * 0.1)
    val_idx, train_idx = order[:val_count], order[val_count:]
    val_ds = tf.data.Dataset.from_tensor_slices((x_train[val_idx], y_train[val_idx])).batch(256)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    _, feedback_before = model.evaluate(x_feedback, y_feedback, verbose=0)
    _, test_before = model.evaluate(x_test, y_test, verbose=0)

    dataset = make_finetune_dataset(
        x_feedback, y_feedback, x_train[train_idx], y_train[train_idx]
    )
    model.fit(dataset, epochs=EPOCHS, verbose=1)

    _, feedback_after = model.evaluate(x_feedback, y_feedback, verbose=0)
    _, test_after = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[RESULT] Corrections accuracy: {feedback_before * 100:.2f}% -> {feedback_after * 100:.2f}%")
    print(f"[RESULT] MNIST test accuracy : {test_before * 100:.2f}% -> {test_after * 100:.2f}%")
    # Corrections accuracy is measured on the training samples themselves, so it is optimistic.

    report_per_class(model, x_test, y_test)
    calibrate(model, val_ds, x_test, y_test)
    save_for_inference(model)


if __name__ == "__main__":
    finetune()
