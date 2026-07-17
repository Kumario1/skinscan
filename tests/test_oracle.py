"""The evaluation-only VOC oracle reader.

Oracle observations are ground-truth counterfactuals, so a malformed annotation
must fail loudly rather than quietly contribute a wrong box to an evaluation.
"""
import pytest

from src.pipeline.oracle import load_voc_oracle_observations


def _voc(objects="", width=200, height=100):
    return (
        "<annotation>"
        f"<size><width>{width}</width><height>{height}</height></size>"
        f"{objects}</annotation>"
    )


def _obj(name="whitehead", xmin=10, ymin=10, xmax=50, ymax=40, bndbox=True):
    box = (f"<bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
           f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>") if bndbox else ""
    return f"<object><name>{name}</name>{box}</object>"


def _write(tmp_path, xml, name="ann.xml"):
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    return path


def test_reads_annotation_boxes_as_certain_observations(tmp_path):
    [observation] = load_voc_oracle_observations(_write(tmp_path, _voc(_obj())))
    assert observation.label == "whitehead"
    assert observation.label_name == "whitehead"
    assert observation.score == 1.0, "annotations are labels, not probabilities"
    assert observation.box == (10.0, 10.0, 50.0, 40.0)
    assert observation.tile_box == (0, 0, 200, 100)


def test_annotation_with_no_objects_is_an_empty_reading_not_an_error(tmp_path):
    assert load_voc_oracle_observations(_write(tmp_path, _voc())) == []


@pytest.mark.parametrize("raw, expected", [
    ("Whitehead", "whitehead"),
    ("  papule-pustule  ", "papule_pustule"),
    ("Papule Pustule", "papule_pustule"),
    ("NODULE", "nodule"),
])
def test_labels_are_normalized_but_the_raw_name_is_kept(tmp_path, raw, expected):
    [observation] = load_voc_oracle_observations(_write(tmp_path, _voc(_obj(name=raw))))
    assert observation.label == expected
    assert observation.label_name == raw.strip()


def test_boxes_are_clamped_to_the_image(tmp_path):
    xml = _voc(_obj(xmin=-30, ymin=-5, xmax=900, ymax=900))
    [observation] = load_voc_oracle_observations(_write(tmp_path, xml))
    assert observation.box == (0.0, 0.0, 200.0, 100.0)


def test_missing_file_is_reported_as_invalid_annotations(tmp_path):
    with pytest.raises(ValueError, match="oracle annotations invalid"):
        load_voc_oracle_observations(tmp_path / "absent.xml")


def test_malformed_xml_is_reported_as_invalid_annotations(tmp_path):
    with pytest.raises(ValueError, match="oracle annotations invalid"):
        load_voc_oracle_observations(_write(tmp_path, "<annotation><size>"))


def test_annotation_without_a_size_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing size"):
        load_voc_oracle_observations(_write(tmp_path, "<annotation></annotation>"))


@pytest.mark.parametrize("width, height", [
    ("abc", 100), (200, "abc"), ("", 100), (200, ""), (12.5, 100),
])
def test_non_integer_image_size_is_rejected(tmp_path, width, height):
    with pytest.raises(ValueError, match="invalid size"):
        load_voc_oracle_observations(_write(tmp_path, _voc(width=width, height=height)))


@pytest.mark.parametrize("width, height", [(0, 100), (200, 0), (-1, 100), (200, -1)])
def test_non_positive_image_size_is_rejected(tmp_path, width, height):
    with pytest.raises(ValueError, match="invalid size"):
        load_voc_oracle_observations(_write(tmp_path, _voc(width=width, height=height)))


@pytest.mark.parametrize("objects", [
    "<object><name></name><bndbox><xmin>1</xmin><ymin>1</ymin>"
    "<xmax>2</xmax><ymax>2</ymax></bndbox></object>",          # blank label
    "<object><bndbox><xmin>1</xmin><ymin>1</ymin>"
    "<xmax>2</xmax><ymax>2</ymax></bndbox></object>",          # no label
    "<object><name>whitehead</name></object>",                 # no bndbox
], ids=["blank_label", "no_label", "no_bndbox"])
def test_object_without_a_label_or_box_is_rejected(tmp_path, objects):
    with pytest.raises(ValueError, match=r"object\[0\] missing label/bndbox"):
        load_voc_oracle_observations(_write(tmp_path, _voc(objects)))


def test_non_numeric_box_coordinates_are_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"object\[0\] invalid bbox"):
        load_voc_oracle_observations(_write(tmp_path, _voc(_obj(xmin="left"))))


@pytest.mark.parametrize("coords, why", [
    ({"xmin": 50, "xmax": 10}, "inverted in x"),
    ({"ymin": 40, "ymax": 10}, "inverted in y"),
    ({"xmin": 10, "xmax": 10}, "zero width"),
    ({"ymin": 10, "ymax": 10}, "zero height"),
    ({"xmin": 500, "xmax": 600}, "entirely right of the image"),
    ({"ymin": 300, "ymax": 400}, "entirely below the image"),
])
def test_degenerate_or_off_image_boxes_are_rejected(tmp_path, coords, why):
    with pytest.raises(ValueError, match=r"object\[0\] empty bbox"):
        load_voc_oracle_observations(_write(tmp_path, _voc(_obj(**coords))))


def test_the_offending_object_index_is_reported(tmp_path):
    xml = _voc(_obj() + _obj() + _obj(xmin=50, xmax=10))
    with pytest.raises(ValueError, match=r"object\[2\] empty bbox"):
        load_voc_oracle_observations(_write(tmp_path, xml))
