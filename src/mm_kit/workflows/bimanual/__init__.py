"""Public API for the ULTra-MoCap neuromuscular workflow."""

from .cli import main
from .muscles import build_muscle_groups, load_muscle_names
from .rollout import open_rollout, resolve_trial_stems, validate_rollout

__all__ = [
    "build_muscle_groups",
    "find_preprocessed_emg",
    "load_aligned_emg",
    "load_muscle_names",
    "main",
    "open_rollout",
    "resolve_trial_stems",
    "validate_rollout",
]
