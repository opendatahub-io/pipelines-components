"""Speculator Training Component."""

from .extraction import extract_speculator
from .online import online_speculator
from .training import train_speculator_mode

__all__ = ["extract_speculator", "online_speculator", "train_speculator_mode"]
