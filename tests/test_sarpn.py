from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipeline.sarpn import SarpnSettings


def _config_with(**overrides):
    config = deepcopy(load_config())
    config["sa_rpn"].update(overrides)
    return config


def test_sarpn_settings_load_production_config():
    settings = SarpnSettings.from_config(load_config())

    assert settings.endpoint_url == "http://localhost:8000/predict"
    assert settings.tile_size == 1024
    assert settings.tile_overlap == 128
    assert settings.connect_timeout_seconds == 5
    assert settings.read_timeout_seconds == 120
    assert settings.request_batch_size == 4
    assert settings.min_score == 0.3
    assert settings.dedupe_threshold == 0.5


def test_sarpn_settings_do_not_alias_source_severity_config():
    config = load_config()
    settings = SarpnSettings.from_config(config)

    config["sa_rpn"]["severity"]["confidence_cutoff"] = 0.1
    config["sa_rpn"]["severity"]["count_thresholds"]["acne_comedonal"].append(100)

    assert settings.severity["confidence_cutoff"] == 0.5
    assert settings.severity["count_thresholds"]["acne_comedonal"] == (1, 8, 20, 40)


def test_sarpn_settings_severity_is_recursively_immutable():
    settings = SarpnSettings.from_config(load_config())

    with pytest.raises(TypeError):
        settings.severity["confidence_cutoff"] = 0.1
    with pytest.raises(TypeError):
        settings.severity["count_thresholds"]["acne_comedonal"] = (1,)
    with pytest.raises(TypeError):
        settings.severity["professional_review"]["nevus"]["min_count"] = 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tile_size": 0}, "tile_size"),
        ({"tile_overlap": -1}, "tile_overlap"),
        ({"tile_overlap": 1024}, "tile_overlap"),
        ({"connect_timeout_seconds": 0}, "connect_timeout_seconds"),
        ({"read_timeout_seconds": -1}, "read_timeout_seconds"),
        ({"request_batch_size": 0}, "request_batch_size"),
        ({"min_score": -0.01}, "min_score"),
        ({"min_score": 1.01}, "min_score"),
        ({"dedupe_threshold": -0.01}, "dedupe_threshold"),
        ({"dedupe_threshold": 1.01}, "dedupe_threshold"),
    ],
)
def test_sarpn_settings_reject_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        SarpnSettings.from_config(_config_with(**overrides))
