"""Replicate dadydada/miniproject-ai-6610210284 locally.

Architecture/training settings mirror the Kaggle notebook: EfficientNetB0
ImageNet backbone, last 30 layers trainable, class weights, Adam(1e-5),
ReduceLROnPlateau, 150 epochs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifier import RAW_ACNE_CLASSES, build_kaggle_efficientnet
from ..config import load_config

_cfg = load_config()
DEFAULT_DATA = Path(_cfg["classification"]["local_data"])
DEFAULT_OUT = Path(_cfg["classification"]["weights"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=_cfg["classification"]["crop_size"])
    p.add_argument("--inspect", action="store_true")
    return p.parse_args()


def count_split(root, split):
    rows = []
    for d in sorted((root / split).iterdir()):
        if d.is_dir():
            rows.append((d.name, sum(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for p in d.iterdir())))
    return rows


def inspect(root):
    for split in ("train", "valid", "test"):
        print(f"\n{split}")
        for name, n in count_split(root, split):
            print(f"  {name:12s} {n}")


def load_datasets(args):
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data / "train",
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data / "valid",
        label_mode="int",
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data / "test",
        label_mode="int",
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )

    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    valid_ds = valid_ds.map(lambda x, y: (preprocess_input(x), y))
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))
    return train_ds, valid_ds, test_ds


def class_weights():
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    counts = [735, 645, 621, 584, 193]
    classes = np.array([0, 1, 2, 3, 4])
    y = np.repeat(classes, counts)
    return dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y)))


def jsonable_history(history):
    return {k: [float(x) for x in v] for k, v in history.items()}


def main():
    args = parse_args()
    if args.inspect:
        inspect(args.data)
        return

    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

    train_ds, valid_ds, test_ds = load_datasets(args)
    class_names = train_ds.class_names
    if class_names != RAW_ACNE_CLASSES:
        raise ValueError(f"class order drift: {class_names}")
    print("Classes:", class_names)

    model = build_kaggle_efficientnet(len(class_names), args.image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        class_weight=class_weights(),
        callbacks=[
            ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-6),
            ModelCheckpoint(str(args.out), monitor="val_accuracy", save_best_only=True, verbose=1),
        ],
    )

    model = tf.keras.models.load_model(args.out)
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test Loss :", test_loss)
    print("Test Accuracy :", test_acc)

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=2))
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))

    args.out.with_suffix(args.out.suffix + ".labels.json").write_text(
        json.dumps({
            "source": "https://www.kaggle.com/code/dadydada/miniproject-ai-6610210284",
            "classes": class_names,
            "image_size": args.image_size,
            "history": jsonable_history(history.history),
        }, indent=2) + "\n"
    )
    print("saved:", args.out)


if __name__ == "__main__":
    main()
