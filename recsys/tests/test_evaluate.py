from pathlib import Path

from recsys.evaluate import evaluate


def test_committed_golden_evaluation_cases_pass():
    manifest = Path(__file__).parents[1] / "eval" / "cases.json"
    assert evaluate(manifest) == []
