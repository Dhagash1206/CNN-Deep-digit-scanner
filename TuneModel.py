"""
Optional: search CNN hyperparameters with Optuna before a full retrain.
        pip install -r requirements-dev.txt
        python TuneModel.py

The study is stored in TuningStudy.db, so an interrupted run resumes where it stopped.
The test set is never touched here; copy the best values into TrainModel.py, retrain,
and evaluate on test once.
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import optuna
import tensorflow as tf
from tensorflow.keras import layers, models

from TrainModel import load_data, random_stroke_width

STUDY_NAME = "digit-cnn"
STORAGE_URL = "sqlite:///TuningStudy.db"
TRIALS = 40
MAX_EPOCHS = 12
VAL_FRACTION = 0.1
SEED = 42

# Current TrainModel.py settings, run first so every later trial is compared with them.
BASELINE_PARAMS = {
    "learning_rate": 1e-3,
    "augmentation_strength": 1.0,
    "dropout": 0.4,
    "weight_decay": 1e-6,
    "batch_size": 128,
    "dense_units": 256,
    "base_filters": 32,
    "stroke_prob": 0.25,
}


def split_train_val(x_train, y_train):
    # Same permutation as TrainModel.make_datasets, so tuning never sees images the final
    # run validates on with a different split.
    order = np.random.default_rng(SEED).permutation(len(x_train))
    val_count = int(len(x_train) * VAL_FRACTION)
    val_idx, train_idx = order[:val_count], order[val_count:]
    return (x_train[train_idx], y_train[train_idx]), (x_train[val_idx], y_train[val_idx])


def build_augmenter(strength):
    # Strength 1.0 reproduces TrainModel.build_augmenter.
    return models.Sequential(
        [
            layers.RandomRotation(0.04 * strength, fill_mode="constant", fill_value=0.0),
            layers.RandomShear(x_factor=0.15 * strength, y_factor=0.0, fill_mode="constant"),
            layers.RandomZoom(0.12 * strength, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(
                0.08 * strength, 0.08 * strength, fill_mode="constant", fill_value=0.0
            ),
        ]
    )


def make_datasets(train_split, val_split, params):
    augmenter = build_augmenter(params["augmentation_strength"])
    stroke_prob = params["stroke_prob"]
    batch_size = params["batch_size"]

    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_split)
        .shuffle(10_000, seed=SEED)
        .batch(batch_size)
        .map(
            lambda images, labels: (
                augmenter(random_stroke_width(images, stroke_prob, stroke_prob), training=True),
                labels,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices(val_split).batch(512).prefetch(tf.data.AUTOTUNE)
    # Every validation digit eroded ~1px: a stand-in for thin canvas strokes, which MNIST
    # validation alone would not penalize a model for missing.
    thin_val_ds = val_ds.map(
        lambda images, labels: (random_stroke_width(images, 1.0, 0.0), labels)
    )
    return train_ds, val_ds, thin_val_ds


def build_model(params):
    filters = params["base_filters"]
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(filters, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters * 2, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters * 4, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(params["dense_units"], activation="relu"),
            layers.Dropout(params["dropout"]),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=params["learning_rate"], weight_decay=params["weight_decay"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def suggest_params(trial):
    return {
        # Biggest lever for Adam + BatchNorm; log scale because 3e-4 vs 1e-3 matters as
        # much as 1e-3 vs 3e-3.
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
        # Scales rotation/shear/zoom/shift together: too weak hurts canvas digits, too
        # strong blurs 4/9 and 7/1 and costs MNIST accuracy.
        "augmentation_strength": trial.suggest_float("augmentation_strength", 0.5, 1.5),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6),
        # Lower bound stands in for "no decay", which is what TrainModel.py uses today.
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dense_units": trial.suggest_categorical("dense_units", [128, 256, 384]),
        "base_filters": trial.suggest_categorical("base_filters", [24, 32, 48]),
        # Probability of thinning and, separately, of thickening each training image.
        "stroke_prob": trial.suggest_float("stroke_prob", 0.0, 0.35),
    }


class PruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(logs["val_loss"], epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch + 1}")


def make_objective(train_split, val_split):
    def objective(trial):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        params = suggest_params(trial)
        train_ds, val_ds, thin_val_ds = make_datasets(train_split, val_split, params)
        model = build_model(params)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=MAX_EPOCHS,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
                PruningCallback(trial),
            ],
            verbose=0,
        )

        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
        thin_loss, thin_accuracy = model.evaluate(thin_val_ds, verbose=0)
        final_train_accuracy = history.history["accuracy"][-1]
        trial.set_user_attr("val_accuracy", val_accuracy)
        trial.set_user_attr("thin_val_accuracy", thin_accuracy)
        # Train accuracy is measured on augmented batches, so a negative gap is normal.
        trial.set_user_attr("train_val_gap", final_train_accuracy - val_accuracy)
        trial.set_user_attr("epochs_run", len(history.history["loss"]))
        print(
            f"[TRIAL {trial.number}] val acc {val_accuracy * 100:.2f}% | "
            f"thin val acc {thin_accuracy * 100:.2f}% | epochs {len(history.history['loss'])}"
        )
        # Loss, not accuracy: at ~99.4% accuracy differs by a handful of images between
        # trials, while loss still separates them.
        return (val_loss + thin_loss) / 2

    return objective


def tune():
    (x_train, y_train), _ = load_data()
    train_split, val_split = split_train_val(x_train, y_train)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    # An interrupted run leaves its last trial RUNNING forever; fail it so a resume can
    # re-run it (assumes only one TuneModel.py process uses the study at a time).
    for stale_trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(stale_trial.number, state=optuna.trial.TrialState.FAIL)
    # skip_if_exists would also count the failed baseline, so check usable trials only.
    usable_states = (
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.WAITING,
    )
    # A WAITING trial has no params yet; its queued values sit in system_attrs.
    if not any(
        (trial.params or trial.system_attrs.get("fixed_params")) == BASELINE_PARAMS
        for trial in study.get_trials(states=usable_states)
    ):
        study.enqueue_trial(BASELINE_PARAMS)

    finished = len(study.get_trials(states=usable_states[:2]))
    study.optimize(make_objective(train_split, val_split), n_trials=max(0, TRIALS - finished))

    best = study.best_trial
    print(f"\n[RESULT] Best objective: {best.value:.4f} (trial {best.number})")
    print(f"[RESULT] Val accuracy: {best.user_attrs['val_accuracy'] * 100:.2f}%")
    print(f"[RESULT] Thin val accuracy: {best.user_attrs['thin_val_accuracy'] * 100:.2f}%")
    print("[RESULT] Parameters:")
    for name, value in best.params.items():
        print(f"  {name}: {value}")

    importances = optuna.importance.get_param_importances(study)
    print("\n[RESULT] Parameter importance:")
    for name, importance in importances.items():
        print(f"  {name}: {importance:.2f}")


if __name__ == "__main__":
    tune()
