from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.run_acne04_pipeline import load_rgb


def test_load_rgb_applies_exif_orientation():
    img = Image.new("RGB", (200, 100), (255, 0, 0))
    exif = img.getexif()
    exif[274] = 6
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "landscape.jpg"
        img.save(path, exif=exif)
        # load_rgb applies EXIF: the 100-high landscape becomes 200-high portrait.
        assert load_rgb(path).shape == (200, 100, 3)
        # Documents WHY the helper exists: raw PIL load ignores EXIF orientation.
        # If Pillow ever starts auto-applying EXIF, this line failing tells a
        # maintainer the helper is now redundant.
        assert np.asarray(Image.open(path)).shape == (100, 200, 3)


if __name__ == "__main__":
    test_load_rgb_applies_exif_orientation()
    print("ok")
