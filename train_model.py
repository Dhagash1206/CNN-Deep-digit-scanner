"""
Step 1: Run this first to train and save the model.
        python train_model.py
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

from calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    save_temperature,
)


def load_data():
    print("[INFO] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    print(f"[INFO] Train: {x_train.shape} | Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def build_augmenter():
    # RandomRotation factor is a fraction of a full turn: 0.04 * 360 ≈ ±14°.
    return models.Sequential(
        [
            layers.RandomRotation(0.04, fill_mode="constant", fill_value=0.0),
            layers.RandomShear(x_factor=0.15, y_factor=0.0, fill_mode="constant"),
            layers.RandomZoom(0.12, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
        ],
        name="augmenter",
    )


def random_stroke_width(images, thin_prob=0.25, thick_prob=0.25):
    # Morphology at 2x resolution so a 3x3 window changes stroke width by ~1px at 28x28,
    # mimicking pens thinner or thicker than MNIST's.
    upsampled = tf.image.resize(images, (56, 56), method="bilinear")
    dilated = tf.nn.max_pool2d(upsampled, ksize=3, strides=1, padding="SAME")
    eroded = -tf.nn.max_pool2d(-upsampled, ksize=3, strides=1, padding="SAME")
    thick = tf.image.resize(dilated, (28, 28), method="area")
    thin = tf.image.resize(eroded, (28, 28), method="area")

    draw = tf.random.uniform((tf.shape(images)[0], 1, 1, 1))
    images = tf.where(draw < thin_prob, thin, images)
    return tf.where(draw > 1.0 - thick_prob, thick, images)


def make_datasets(x_train, y_train, batch_size=128, val_fraction=0.1, seed=42):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train))
    val_count = int(len(x_train) * val_fraction)
    val_idx, train_idx = order[:val_count], order[val_count:]

    augmenter = build_augmenter()
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train[train_idx], y_train[train_idx]))
        .shuffle(10_000, seed=seed)
        .batch(batch_size)
        .map(
            lambda images, labels: (
                augmenter(random_stroke_width(images), training=True),
                labels,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    # Validation stays clean so EarlyStopping tracks real generalization.
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_train[val_idx], y_train[val_idx]))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


def build_model(compiled=True):
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            # 7x7 -> 3x3 so Flatten yields 1152 features instead of 6272.
            layers.MaxPooling2D((2, 2)),
            # Classifier head
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(10, activation="softmax"),
        ]
    )

    if not compiled:
        return model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


MODEL_PATH = "mnist_cnn_model.keras"
CHECKPOINT_PATH = "mnist_cnn_model.best.keras"
CALIBRATION_PATH = "mnist_cnn_model.calibration.json"
SAMPLES_PATH = os.path.join("assets", "mnist_samples.png")


def predict_dataset(model, dataset):
    probabilities, labels = [], []
    for images, batch_labels in dataset:
        probabilities.append(model(images, training=False).numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def calibrate(model, val_ds, x_test, y_test, path=CALIBRATION_PATH):
    # Fit on validation, report on test, so the reported gain is not self-measured.
    val_probs, val_labels = predict_dataset(model, val_ds)
    temperature = fit_temperature(val_probs, val_labels)

    test_probs = model.predict(x_test, verbose=0)
    calibrated = apply_temperature(test_probs, temperature)
    print(f"\n[CALIBRATION] Temperature: {temperature:.3f}")
    print(
        f"[CALIBRATION] Test NLL {negative_log_likelihood(test_probs, y_test):.4f}"
        f" -> {negative_log_likelihood(calibrated, y_test):.4f}"
    )
    print(
        f"[CALIBRATION] Test ECE {expected_calibration_error(test_probs, y_test) * 100:.2f}%"
        f" -> {expected_calibration_error(calibrated, y_test) * 100:.2f}%"
    )
    save_temperature(path, temperature)
    print(f"[INFO] Calibration saved -> {path}")
    return temperature


def report_per_class(model, x_test, y_test, top_pairs=5):
    predicted = model.predict(x_test, verbose=0).argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (y_test, predicted), 1)

    print("\n[RESULT] Confusion matrix (rows = true, cols = predicted):")
    print("      " + " ".join(f"{digit:>5}" for digit in range(10)))
    for digit in range(10):
        print(f"{digit:>5} " + " ".join(f"{count:>5}" for count in confusion[digit]))

    print("\n[RESULT] Per-class accuracy:")
    for digit in range(10):
        accuracy = confusion[digit, digit] / confusion[digit].sum()
        print(f"  {digit}: {accuracy * 100:.2f}%")

    errors = confusion.copy()
    np.fill_diagonal(errors, 0)
    worst = np.dstack(np.unravel_index(np.argsort(errors, axis=None)[::-1], errors.shape))[0]
    print("\n[RESULT] Most confused (true -> predicted):")
    for true_digit, predicted_digit in worst[:top_pairs]:
        print(f"  {true_digit} -> {predicted_digit}: {errors[true_digit, predicted_digit]}")


def save_for_inference(model, path=MODEL_PATH):
    # Keras 3 ignores include_optimizer=False and saves Adam's two moment copies of every
    # weight; an uncompiled copy has no optimizer, which cuts the file to about a third.
    inference_model = build_model(compiled=False)
    inference_model.set_weights(model.get_weights())
    inference_model.save(path)
    print(f"[INFO] Model saved -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def save_sample_digits(x_test, y_test, path=SAMPLES_PATH):
    # One 28x28 example per digit in a 280x28 strip, so the app's sample buttons work offline.
    strip = np.concatenate(
        [x_test[np.where(y_test == digit)[0][0], :, :, 0] for digit in range(10)], axis=1
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((strip * 255).round().astype(np.uint8)).save(path)


def train():
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = load_data()

    train_ds, val_ds = make_datasets(x_train, y_train)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        # Separate file so a crashed run never overwrites the model app.py loads.
        tf.keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    print("\n[INFO] Training model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[RESULT] Test Accuracy : {acc * 100:.2f}%")
    print(f"[RESULT] Test Loss     : {loss:.4f}")

    report_per_class(model, x_test, y_test)
    calibrate(model, val_ds, x_test, y_test)
    save_sample_digits(x_test, y_test)

    save_for_inference(model)


if __name__ == "__main__":
    train()
