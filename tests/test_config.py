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


def test_load_config_has_recommender_milestone_keys():
    # Issue #2: inert config keys the verbose-recommender milestone will consume.
    cfg = load_config()

    # Profile skin-type vocabulary — the closed set matching the review data (D-021).
    assert cfg["profile"]["skin_types"] == ["combination", "dry", "normal", "oily"]

    # ITA tone-bucket cutoffs and low-light L* threshold (D-021).
    assert cfg["tone"]["ita_light_min"] == 41
    assert cfg["tone"]["ita_medium_min"] == 10
    assert cfg["tone"]["low_light_l_threshold"] == 35
    assert cfg["tone"]["profile_cheek_area_ratio"] == 0.5
    assert cfg["tone"]["min_sample_pixels"] == 100
    assert set(cfg["tone"]["sephora_tone_buckets"]) == {"light", "medium", "deep"}
    assert cfg["regions"]["perioral_scale"] == 1.55

    # Ranker artifacts + minimum evidence cell size (D-022).
    assert isinstance(cfg["ranker"]["model_path"], str) and cfg["ranker"]["model_path"]
    assert isinstance(cfg["ranker"]["review_stats_path"], str) and cfg["ranker"]["review_stats_path"]
    assert cfg["ranker"]["min_cell_size"] == 5

    # Raw reviews source path (D-015 extension).
    assert isinstance(cfg["paths"]["reviews_raw"], str) and cfg["paths"]["reviews_raw"]
    assert cfg["paths"]["face_landmarker"].endswith(".task")


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
    test_load_config_has_recommender_milestone_keys()
    test_cli_defaults_come_from_config()
    print("ok")
