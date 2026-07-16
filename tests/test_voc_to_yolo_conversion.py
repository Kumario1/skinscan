"""The VOC parsers and directory converter (src/detection/voc_to_yolo.py).

tests/test_voc_to_yolo.py covers the geometry; this covers everything around it,
including a run over the real 275-annotation AcneSCU VOC dump committed to the
repo. Stage 1 is single-class `lesion` (D-018, LOCKED), so the converter
deliberately discards the VOC <name> and emits class_id 0 for every box.
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.voc_to_yolo import (
    Box, convert_voc_dir, inspect_raw, parse_flat_line, parse_voc_xml,
)

REAL_VOC = Path(__file__).resolve().parents[1] / "AcneSCU.v1-acnescu-original.voc" / "train"


def _xml(tmp_path, objects, width=200, height=100, name="a.xml"):
    path = tmp_path / name
    path.write_text(
        "<annotation>"
        f"<size><width>{width}</width><height>{height}</height></size>"
        f"{objects}</annotation>",
        encoding="utf-8",
    )
    return path


def _obj(xmin=10, ymin=10, xmax=50, ymax=40, name="papule"):
    return (f"<object><name>{name}</name><bndbox>"
            f"<xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
            f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>")


# --- parse_voc_xml ------------------------------------------------------------

def test_parse_voc_xml_reads_size_and_boxes(tmp_path):
    width, height, boxes = parse_voc_xml(str(_xml(tmp_path, _obj() + _obj(60, 20, 90, 70))))
    assert (width, height) == (200, 100)
    assert boxes == [Box(10.0, 10.0, 50.0, 40.0), Box(60.0, 20.0, 90.0, 70.0)]


def test_parse_voc_xml_accepts_a_float_valued_size(tmp_path):
    """Some VOC exporters write "200.0"; int("200.0") would raise."""
    width, height, _ = parse_voc_xml(str(_xml(tmp_path, _obj(), width="200.0", height="100.0")))
    assert (width, height) == (200, 100)


def test_parse_voc_xml_of_an_annotation_with_no_objects(tmp_path):
    width, height, boxes = parse_voc_xml(str(_xml(tmp_path, "")))
    assert (width, height) == (200, 100)
    assert boxes == []


# --- parse_flat_line ----------------------------------------------------------

def test_parse_flat_line_reads_name_and_comma_separated_boxes():
    name, boxes = parse_flat_line("img.jpg 10,20,30,40 50,60,70,80")
    assert name == "img.jpg"
    assert boxes == [Box(10.0, 20.0, 30.0, 40.0), Box(50.0, 60.0, 70.0, 80.0)]


def test_parse_flat_line_keeps_only_the_first_four_numbers():
    """A trailing class id or score must not be read as geometry."""
    _, boxes = parse_flat_line("img.jpg 10,20,30,40,7")
    assert boxes == [Box(10.0, 20.0, 30.0, 40.0)]


def test_parse_flat_line_ignores_a_token_with_too_few_numbers():
    _, boxes = parse_flat_line("img.jpg 10,20 50,60,70,80")
    assert boxes == [Box(50.0, 60.0, 70.0, 80.0)]


def test_parse_flat_line_of_an_image_with_no_boxes():
    assert parse_flat_line("img.jpg") == ("img.jpg", [])


# --- convert_voc_dir ----------------------------------------------------------

def test_convert_writes_one_label_file_per_xml(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _xml(xml_dir, _obj(), name="a.xml")
    _xml(xml_dir, _obj() + _obj(60, 20, 90, 70), name="b.xml")
    out = tmp_path / "labels"

    report = convert_voc_dir(str(xml_dir), str(out))

    assert report == {"images": 2, "boxes": 3, "skipped_boxes": 0, "empty_images": 0}
    assert sorted(p.name for p in out.iterdir()) == ["a.txt", "b.txt"]
    assert (out / "a.txt").read_text() == "0 0.150000 0.250000 0.200000 0.300000"


def test_convert_ignores_non_xml_files(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _xml(xml_dir, _obj(), name="a.xml")
    (xml_dir / "notes.txt").write_text("not an annotation")
    (xml_dir / "a.jpg").write_bytes(b"\xff\xd8")

    assert convert_voc_dir(str(xml_dir), str(tmp_path / "labels"))["images"] == 1


def test_convert_counts_a_degenerate_box_as_skipped_not_converted(tmp_path):
    """A box that clamps to nothing must be dropped and REPORTED -- silently
    emitting it would poison the training labels."""
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _xml(xml_dir, _obj() + _obj(xmin=500, xmax=600), name="a.xml")  # 2nd is off-image

    report = convert_voc_dir(str(xml_dir), str(tmp_path / "labels"))

    assert report == {"images": 1, "boxes": 1, "skipped_boxes": 1, "empty_images": 0}


def test_convert_writes_an_empty_label_file_for_a_background_image(tmp_path):
    """YOLO reads a background image as an empty .txt, not a missing one."""
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _xml(xml_dir, "", name="bg.xml")
    out = tmp_path / "labels"

    report = convert_voc_dir(str(xml_dir), str(out))

    assert report == {"images": 1, "boxes": 0, "skipped_boxes": 0, "empty_images": 1}
    assert (out / "bg.txt").exists()
    assert (out / "bg.txt").read_text() == ""


def test_convert_creates_the_output_directory(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _xml(xml_dir, _obj(), name="a.xml")
    out = tmp_path / "does" / "not" / "exist"

    convert_voc_dir(str(xml_dir), str(out))
    assert (out / "a.txt").exists()


def test_inspect_raw_prints_a_tree_without_reading_annotations(tmp_path, capsys):
    (tmp_path / "train").mkdir()
    for index in range(7):
        (tmp_path / "train" / f"{index}.xml").write_text("<annotation/>")

    inspect_raw(str(tmp_path))

    printed = capsys.readouterr().out
    assert "== tree of" in printed
    assert "... (+2 more)" in printed, "only the first 5 filenames are listed"


# --- the real committed dump --------------------------------------------------

@pytest.mark.skipif(not REAL_VOC.exists(), reason="AcneSCU VOC dump absent")
def test_the_real_acnescu_dump_converts_cleanly():
    """Every annotation in the committed dump must convert, and every emitted
    coordinate must be normalized to [0,1] -- an out-of-range value would be
    silently accepted by Ultralytics and train the detector on nonsense."""
    import tempfile

    with tempfile.TemporaryDirectory() as out:
        report = convert_voc_dir(str(REAL_VOC), out)
        assert report["images"] == 275
        assert report["boxes"] > 30_000
        assert report["skipped_boxes"] == 0
        assert report["empty_images"] == 0

        out_path = Path(out)
        for label_file in out_path.iterdir():
            for line in label_file.read_text().splitlines():
                class_id, *coords = line.split()
                assert class_id == "0", "stage 1 is single-class lesion (D-018)"
                assert len(coords) == 4
                for value in map(float, coords):
                    assert 0.0 <= value <= 1.0, f"{label_file.name}: {value}"

        # YOLO pairs a label to its image by basename; a mismatch trains on nothing
        images = {p.stem for p in REAL_VOC.glob("*.jpg")}
        labels = {p.stem for p in out_path.glob("*.txt")}
        assert labels == images
