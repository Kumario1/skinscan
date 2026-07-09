"""Slow regression for the self-collected profile used by issue #6's HITL gate."""
from collections import Counter
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.run_acne04_pipeline import load_rgb
from src.pipeline.regions import locate_regions


MODEL = Path("models/face_landmarker.task")
IMAGE = Path(os.environ.get(
    "SKINSCAN_REAL_FACE_IMAGE",
    "data/self_collected/acne-before-scaled-e1764168292784.png",
))
PROFILE_BOXES = [
    (1016, 426, 1127, 530), (912, 230, 1001, 322),
    (693, 507, 772, 592), (636, 404, 727, 486),
    (901, 868, 988, 954), (544, 265, 629, 350),
    (735, 895, 814, 976), (970, 977, 1059, 1068),
    (1153, 948, 1247, 1034), (1048, 801, 1142, 883),
    (935, 651, 1010, 730), (484, 176, 564, 250),
    (647, 108, 729, 183), (884, 573, 976, 654),
    (789, 572, 868, 658), (657, 25, 741, 95),
]


@pytest.mark.real_models
@pytest.mark.skipif(not (MODEL.exists() and IMAGE.exists()),
                    reason="local FaceLandmarker artifact/self-collected image absent")
def test_profile_photo_regions_match_the_candidate_overlay():
    result = locate_regions(load_rgb(IMAGE), PROFILE_BOXES, model_path=MODEL)

    assert result.metadata["fallback"] is False
    assert Counter(result.regions) == {
        "right_cheek": 10,
        "chin_jaw": 5,
        "forehead": 1,
    }


if __name__ == "__main__":
    if MODEL.exists() and IMAGE.exists():
        test_profile_photo_regions_match_the_candidate_overlay()
        print("ok")
    else:
        print("skipped: local FaceLandmarker artifact/self-collected image absent")
