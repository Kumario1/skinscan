from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.harvest_negatives import NEG_CLASS, harvest, offlesion_boxes, split_for


def test_offlesion_boxes_keeps_only_iou_zero():
    gt = [[0, 0, 10, 10]]
    pred = [
        [0, 0, 10, 10],       # exact overlap -> drop
        [100, 100, 110, 110],  # far away -> keep
        [5, 5, 15, 15],        # partial overlap -> drop
    ]
    assert offlesion_boxes(pred, gt) == [[100, 100, 110, 110]]


def test_offlesion_boxes_empty_gt_keeps_all():
    pred = [[0, 0, 10, 10], [20, 20, 30, 30]]
    assert offlesion_boxes(pred, []) == pred


def test_offlesion_boxes_edge_touch_is_off_lesion():
    # boxes sharing only an edge have zero-area intersection -> IoU 0 -> off-lesion
    assert offlesion_boxes([[10, 0, 20, 10]], [[0, 0, 10, 10]]) == [[10, 0, 20, 10]]


def test_split_for_is_deterministic():
    assert split_for("img_3") == split_for("img_3")


def test_split_for_partitions_roughly_60_20_20():
    keys = [f"k{i}" for i in range(2000)]
    counts = {"train": 0, "valid": 0, "test": 0}
    for k in keys:
        counts[split_for(k)] += 1
    assert set(counts) == {"train", "valid", "test"}
    assert all(v > 0 for v in counts.values())
    assert 0.55 < counts["train"] / 2000 < 0.65
    assert 0.15 < counts["valid"] / 2000 < 0.25
    assert 0.15 < counts["test"] / 2000 < 0.25


# --- End-to-end: the real harvest workflow with a stub detector (model-free,
# per the project's stub-classifier testing philosophy). Real images, real
# crop_with_context transform, real directory writes into the layout the
# trainer consumes: <out>/{train,valid,test}/Not_acne/*.jpg.

class _Box:
    def __init__(self, xyxy):
        self.xyxy = np.array([xyxy], dtype=float)  # (1, 4); harvest reads .xyxy[0].tolist()


class _Result:
    def __init__(self, boxes):
        self.boxes = [_Box(b) for b in boxes]


class _StubDetector:
    """Mimics ultralytics YOLO: .predict(path, ...)[0].boxes[i].xyxy[0]."""
    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, path, conf, iou, imgsz, verbose):
        return [_Result(self._boxes)]


def _args(**over):
    base = dict(conf=0.07, iou=0.2, imgsz=64, pad=1.5, size=224, limit=0)
    base.update(over)
    return SimpleNamespace(**base)


def _write_images(root, stems, wh=(256, 256)):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem in stems:
        p = root / f"{stem}.png"
        Image.fromarray(np.full((wh[1], wh[0], 3), 200, np.uint8)).save(p)
        paths.append(p)
    return paths


def test_harvest_end_to_end_ffhq_writes_split_layout(tmp_path):
    imgs = _write_images(tmp_path / "faces", ["f0", "f1", "f2", "f3"])
    out = tmp_path / "dataset"
    boxes = [[10, 10, 40, 40], [80, 80, 120, 130], [150, 20, 175, 60]]
    total, counts = harvest(_StubDetector(boxes), imgs, out, _args())

    # every FFHQ detector box is a negative -> one crop per (image, box)
    assert total == len(imgs) * len(boxes)
    assert sum(counts.values()) == total

    written = list(out.rglob(f"{NEG_CLASS}/*.jpg"))
    assert len(written) == total
    # crops land only under the trainer's class-per-split layout
    for f in written:
        assert f.parent.name == NEG_CLASS
        assert f.parent.parent.name in {"train", "valid", "test"}
        assert Image.open(f).size == (224, 224)


def test_harvest_end_to_end_acne04_drops_on_lesion_boxes(tmp_path):
    imgs = _write_images(tmp_path / "acne", ["a0", "a1"])
    out = tmp_path / "dataset"
    # box 0 overlaps the GT lesion -> dropped; box 1 is off-lesion -> kept
    boxes = [[10, 10, 30, 30], [100, 100, 130, 130]]
    gt_lookup = lambda p: [[5, 5, 35, 35]]  # overlaps box 0 only
    total, counts = harvest(_StubDetector(boxes), imgs, out, _args(), gt_lookup=gt_lookup)

    assert total == len(imgs)  # exactly one surviving box per image
    assert len(list(out.rglob(f"{NEG_CLASS}/*.jpg"))) == total


def test_harvest_groups_all_crops_of_one_image_into_one_split(tmp_path):
    # no same-face leakage: every crop from one image shares a split
    imgs = _write_images(tmp_path / "faces", ["oneface"])
    out = tmp_path / "dataset"
    boxes = [[10, 10, 40, 40], [80, 80, 120, 130], [150, 20, 175, 60]]
    harvest(_StubDetector(boxes), imgs, out, _args())
    splits_used = [d.name for d in out.iterdir() if (d / NEG_CLASS).exists()]
    assert len(splits_used) == 1


def test_harvest_limit_caps_total(tmp_path):
    imgs = _write_images(tmp_path / "faces", ["f0", "f1", "f2"])
    out = tmp_path / "dataset"
    boxes = [[10, 10, 40, 40], [80, 80, 120, 130]]
    total, _ = harvest(_StubDetector(boxes), imgs, out, _args(limit=3))
    assert total == 3
    assert len(list(out.rglob(f"{NEG_CLASS}/*.jpg"))) == 3


if __name__ == "__main__":
    test_offlesion_boxes_keeps_only_iou_zero()
    test_offlesion_boxes_empty_gt_keeps_all()
    test_offlesion_boxes_edge_touch_is_off_lesion()
    test_split_for_is_deterministic()
    test_split_for_partitions_roughly_60_20_20()
    for fn in (
        test_harvest_end_to_end_ffhq_writes_split_layout,
        test_harvest_end_to_end_acne04_drops_on_lesion_boxes,
        test_harvest_groups_all_crops_of_one_image_into_one_split,
        test_harvest_limit_caps_total,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
