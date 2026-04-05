import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Augmentation
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    model.fit(
        x_train, y_train,
        epochs=25,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy: {acc * 100:.2f}%")

    model.save("mnist_cnn_model.keras")

if __name__ == "__main__":
    train()
