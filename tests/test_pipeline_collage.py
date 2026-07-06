from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.run_acne04_pipeline import acne_type_counts, classifier_image_size, draw_input_collage


def test_draw_input_collage_exact_size():
    crops = [np.full((224, 224, 3), v, np.uint8) for v in (0, 80, 160)]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "collage.png"
        draw_input_collage(crops, out, 224, max_tiles=9)
        image = Image.open(out)
        assert image.size == (224, 224)
        pixels = np.asarray(image)
        assert (pixels != 255).any(axis=2).any()
        assert (pixels == 255).all(axis=2).any()


def test_classifier_image_size_reads_metadata():
    with tempfile.TemporaryDirectory() as d:
        model = Path(d) / "model.keras"
        model.with_suffix(".keras.labels.json").write_text('{"image_size": 192}\n')
        assert classifier_image_size(model) == 192


def test_acne_type_counts_uses_model_class_order():
    detections = [
        {"prediction": "Pustules"},
        {"prediction": "Blackheads"},
        {"prediction": "Pustules"},
        {"detector_conf": 0.8},
    ]
    classes = ["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]
    assert acne_type_counts(detections, classes) == {"Blackheads": 1, "Pustules": 2}


if __name__ == "__main__":
    test_draw_input_collage_exact_size()
    test_classifier_image_size_reads_metadata()
    test_acne_type_counts_uses_model_class_order()
    print("ok")
