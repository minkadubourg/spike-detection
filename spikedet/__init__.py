"""Spike detection for extracellular recordings, with a failure-mode benchmark."""
from .config import DetectorConfig
from .detector import detect, detect_multichannel
from .evaluate import match

__all__ = ["DetectorConfig", "detect", "detect_multichannel", "match"]
