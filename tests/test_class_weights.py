from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.train_type_classifier import class_weights


def make_fixture(d, counts):
    for name, n in counts.items():
        cls = Path(d) / "train" / name
        cls.mkdir(parents=True)
        for i in range(n):
            (cls / f"{i}.jpg").touch()


def test_weights_inverse_to_counts():
    with tempfile.TemporaryDirectory() as d:
        make_fixture(d, {"A": 6, "B": 2, "C": 4})
        w = class_weights(Path(d))
        assert set(w) == {0, 1, 2}
        assert w[1] > w[2] > w[0]
        assert abs(w[0] - 12 / 18) < 1e-9
        assert abs(w[1] - 12 / 6) < 1e-9
        assert abs(w[2] - 12 / 12) < 1e-9


def test_empty_class_dir_exits():
    with tempfile.TemporaryDirectory() as d:
        make_fixture(d, {"A": 6, "B": 2, "C": 4, "D": 0})
        try:
            class_weights(Path(d))
        except SystemExit:
            pass
        else:
            raise AssertionError("expected SystemExit for empty class dir")


def test_non_image_files_ignored():
    with tempfile.TemporaryDirectory() as d:
        make_fixture(d, {"A": 6, "B": 2, "C": 4})
        (Path(d) / "train" / "A" / "notes.txt").touch()
        w = class_weights(Path(d))
        assert set(w) == {0, 1, 2}
        assert abs(w[0] - 12 / 18) < 1e-9
        assert abs(w[1] - 12 / 6) < 1e-9
        assert abs(w[2] - 12 / 12) < 1e-9


if __name__ == "__main__":
    test_weights_inverse_to_counts()
    test_empty_class_dir_exits()
    test_non_image_files_ignored()
    print("ok")
