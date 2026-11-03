"""Speculator Training Component."""

from .extraction import extract_hidden_states
from .online import online_draft_model_training
from .training import train_draft_model

__all__ = ["extract_hidden_states", "online_draft_model_training", "train_draft_model"]
