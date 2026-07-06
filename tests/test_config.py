from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


def test_load_config_has_pipeline_keys():
    cfg = load_config()
    assert isinstance(cfg, dict)
    conf = cfg["detection"]["conf_threshold"]
    assert isinstance(conf, float) and 0 < conf < 1
    assert isinstance(cfg["detection"]["iou_threshold"], float)
    assert isinstance(cfg["detection"]["img_size"], int)
    assert isinstance(cfg["classification"]["crop_pad"], float)
    assert cfg["detection"]["weights"].endswith(".pt")
    assert cfg["classification"]["weights"].endswith(".keras")


def test_cli_defaults_come_from_config():
    cfg = load_config()
    from src.classification.run_acne04_pipeline import parse_args

    saved_argv = sys.argv
    try:
        sys.argv = ["prog"]
        args = parse_args()
    finally:
        sys.argv = saved_argv
    assert args.conf == cfg["detection"]["conf_threshold"]
    assert args.imgsz == cfg["detection"]["img_size"]


if __name__ == "__main__":
    test_load_config_has_pipeline_keys()
    test_cli_defaults_come_from_config()
    print("ok")
