"""
Step 1: Run this first to train and save the model.
        python train_model.py
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


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
            lambda images, labels: (augmenter(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    # Validation stays clean so EarlyStopping tracks real accuracy.
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_train[val_idx], y_train[val_idx]))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


def build_model():
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
            # Classifier head
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    (x_train, y_train), (x_test, y_test) = load_data()

    train_ds, val_ds = make_datasets(x_train, y_train)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
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

    model.save("mnist_cnn_model.keras")
    print("[INFO] Model saved -> mnist_cnn_model.keras")


if __name__ == "__main__":
    train()
