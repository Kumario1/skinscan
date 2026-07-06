"""Single source for pipeline knobs: configs/default.yaml (RULES.md §5)."""
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)
