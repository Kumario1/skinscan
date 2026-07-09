from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.classifier import AcneTypeClassifier, RAW_ACNE_CLASSES, StubClassifier


def test_stub_batch_matches_single():
    s = StubClassifier()
    crop = np.zeros((224, 224, 3), np.uint8)
    batch = s.predict_batch([crop, crop])
    single = s.predict(crop)
    assert len(batch) == 2
    assert batch[0] == batch[1] == single


def test_prepare_resizes_and_casts():
    clf = AcneTypeClassifier.__new__(AcneTypeClassifier)
    clf.image_size = 224
    resized = clf._prepare(np.zeros((100, 50, 3), np.uint8))
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.float32
    same = clf._prepare(np.zeros((224, 224, 3), np.uint8))
    assert same.shape == (224, 224, 3)
    assert same.dtype == np.float32


def test_predict_batch_empty():
    clf = AcneTypeClassifier.__new__(AcneTypeClassifier)
    clf.image_size = 224
    clf.classes = list(RAW_ACNE_CLASSES)
    assert clf.predict_batch([]) == []


def test_predict_batch_uses_one_model_call():
    class FakeModel:
        def __init__(self):
            self.calls = 0
            self.last_shape = None

        def predict(self, x, verbose=0):
            self.calls += 1
            self.last_shape = x.shape
            return np.full((len(x), len(RAW_ACNE_CLASSES)), 1.0 / len(RAW_ACNE_CLASSES))

    clf = AcneTypeClassifier.__new__(AcneTypeClassifier)
    clf.image_size = 224
    clf.classes = list(RAW_ACNE_CLASSES)
    clf.model = FakeModel()
    crop = np.zeros((224, 224, 3), np.uint8)
    out = clf.predict_batch([crop, crop, crop])
    assert len(out) == 3
    assert clf.model.calls == 1
    assert clf.model.last_shape[0] == 3
    assert all(set(d) == set(RAW_ACNE_CLASSES) for d in out)


if __name__ == "__main__":
    test_stub_batch_matches_single()
    test_prepare_resizes_and_casts()
    test_predict_batch_empty()
    test_predict_batch_uses_one_model_call()
    print("ok")
