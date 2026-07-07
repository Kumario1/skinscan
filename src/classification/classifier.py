# Lesion-crop type classifier (Stage 2).
import json
from pathlib import Path

import numpy as np

RAW_ACNE_CLASSES = ["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]
RAW_TO_CONCERN = {
    "Blackheads": "acne_comedonal",
    "Whiteheads": "acne_comedonal",
    "Cyst": "acne_cystic",
    "Papules": "acne_inflammatory",
    "Pustules": "acne_inflammatory",
}


def concern_probs(raw_probs):
    """Aggregate raw class probabilities into D-008 schema concern IDs."""
    out = {c: 0.0 for c in sorted(set(RAW_TO_CONCERN.values()))}
    for raw, prob in raw_probs.items():
        if raw in RAW_TO_CONCERN:
            out[RAW_TO_CONCERN[raw]] += prob
    return out


def crop_with_context(image, box, pad=1.5, size=224):
    """Square crop around a detector box with context padding."""
    from PIL import Image

    x, y, w, h = box[:4]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(max(w, h) * pad, 2.0)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    H, W = image.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    if pl or pt or pr or pb:
        image = np.pad(image, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
        x0, y0, x1, y1 = x0 + pl, y0 + pt, x1 + pl, y1 + pt
    crop = image[y0:y1, x0:x1]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


def build_kaggle_efficientnet(num_classes=len(RAW_ACNE_CLASSES), image_size=224):
    """Kaggle notebook architecture: EfficientNetB0 + GAP + BN + Dense + Dropout."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    return Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])


class AcneTypeClassifier:
    """Loads the TensorFlow model trained by train_type_classifier.py."""

    def __init__(self, model_path, classes=None):
        import tensorflow as tf

        model_path = Path(model_path)
        meta = model_path.with_suffix(model_path.suffix + ".labels.json")
        metadata = json.loads(meta.read_text()) if meta.exists() else {}
        self.classes = list(classes or metadata.get("classes", RAW_ACNE_CLASSES))
        self.image_size = int(metadata.get("image_size", 224))
        self.model = tf.keras.models.load_model(model_path)

    def _prepare(self, crop):
        from PIL import Image

        crop = np.asarray(crop)
        if crop.shape[:2] != (self.image_size, self.image_size):
            crop = np.asarray(Image.fromarray(crop).resize((self.image_size, self.image_size), Image.BILINEAR))
        return crop.astype(np.float32)

    def predict_batch(self, crops):
        from tensorflow.keras.applications.efficientnet import preprocess_input

        if not len(crops):
            return []
        x = preprocess_input(np.stack([self._prepare(c) for c in crops]))
        probs = self.model.predict(x, verbose=0)
        return [dict(zip(self.classes, p.astype(float).tolist())) for p in probs]

    def predict(self, crop):
        return self.predict_batch([crop])[0]

    def predict_concerns(self, crop):
        return concern_probs(self.predict(crop))


LesionClassifier = AcneTypeClassifier


class StubClassifier:
    def __init__(self, probs=None):
        p = np.asarray(probs if probs is not None else [0.25, 0.2, 0.2, 0.2, 0.15], dtype=np.float32)
        self.probs = p / p.sum()

    def predict_batch(self, crops):
        return [dict(zip(RAW_ACNE_CLASSES, self.probs.tolist())) for _ in crops]

    def predict(self, crop):
        return self.predict_batch([crop])[0]


if __name__ == "__main__":
    img = np.zeros((100, 200, 3), np.uint8)
    img[40:60, 90:110] = 255
    c = crop_with_context(img, (90, 40, 20, 20))
    assert c.shape == (224, 224, 3) and c.dtype == np.uint8
    assert c.mean() > 80
    assert crop_with_context(img, (0, 0, 10, 10, 0.9)).shape == (224, 224, 3)
    assert crop_with_context(img, (50, 50, 0, 0)).shape == (224, 224, 3)
    p = StubClassifier().predict(c)
    assert set(p) == set(RAW_ACNE_CLASSES) and abs(sum(p.values()) - 1.0) < 1e-5
    print("ok")
