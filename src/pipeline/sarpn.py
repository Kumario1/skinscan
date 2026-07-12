"""Validated runtime settings for the production SA-RPN service."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SarpnSettings:
    endpoint_url: str
    tile_size: int
    tile_overlap: int
    connect_timeout_seconds: float
    read_timeout_seconds: float
    request_batch_size: int
    min_score: float
    dedupe_threshold: float
    severity: dict[str, Any]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SarpnSettings":
        values = config["sa_rpn"]
        settings = cls(
            endpoint_url=values["endpoint_url"],
            tile_size=values["tile_size"],
            tile_overlap=values["tile_overlap"],
            connect_timeout_seconds=values["connect_timeout_seconds"],
            read_timeout_seconds=values["read_timeout_seconds"],
            request_batch_size=values["request_batch_size"],
            min_score=values["min_score"],
            dedupe_threshold=values["dedupe_threshold"],
            severity=values["severity"],
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.tile_overlap < 0 or self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must satisfy 0 <= tile_overlap < tile_size")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if self.read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive")
        if self.request_batch_size <= 0:
            raise ValueError("request_batch_size must be positive")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if not 0 <= self.dedupe_threshold <= 1:
            raise ValueError("dedupe_threshold must be between 0 and 1")
