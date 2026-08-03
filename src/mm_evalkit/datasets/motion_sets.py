"""Ordered dataset identities used to decode rollout ``traj_id`` values.

These are evaluation metadata. They deliberately live here instead of being
imported from a training environment package.
"""

from __future__ import annotations


def _gait120_motions(subjects: range | list[int], trials: range | list[int]) -> tuple[str, ...]:
    motions: list[str] = []
    for subject in subjects:
        block = (subject - 1) // 10 * 10
        group = f"Gait120_{block + 1:03d}_to_{block + 10:03d}"
        for trial in trials:
            motions.append(f"Gait120_EMG/{group}/S{subject:03d}/LevelWalking/TRC/Trial{trial:02d}/AllSteps_stageii")
    return tuple(motions)


GAIT120_MOTION_SETS: dict[str, tuple[str, ...]] = {
    "GAIT120_TESTING_SUBJECTS": _gait120_motions(
        [*range(51, 61), *range(111, 121)],
        [*range(1, 6)],
    ),
    "GAIT120_TESTING_TRIALS": _gait120_motions(
        [*range(1, 121)],
        [5],
    ),
    "GAIT120_TESTING": _gait120_motions(
        [*range(51, 61), *range(111, 121)],
        [5],
    ),
    "GAIT120_TESTING_SUBJECTS_TRIALS0104": _gait120_motions(
        [*range(51, 61), *range(111, 121)],
        [*range(1, 5)],
    ),
}


_ULTRAMOCAP_CONDITIONS = {
    "ArmSwing": ("Slow", "Normal", "Fast", "VeryFast"),
    "CrossbodyReach": ("Slow", "Normal", "Fast"),
    "ElbowFlexion": ("Slow", "Normal", "Fast"),
    "OverheadReach": ("90", "180", "Max"),
    "ShoulderRotation": ("Slow", "Normal", "Fast"),
}


def _ultramocap_motions(subjects: range) -> tuple[str, ...]:
    return tuple(
        f"ULTRA_MOCAP/P{subject:02d}/P{subject:02d}_{movement}_{condition}"
        for subject in subjects
        for movement, conditions in _ULTRAMOCAP_CONDITIONS.items()
        for condition in conditions
    )


ULTRAMOCAP_MOTION_SETS: dict[str, tuple[str, ...]] = {
    "ULTRA_MOCAP_BIMANUAL_TRAINING_SUBJECTS": _ultramocap_motions(range(1, 11)),
    "ULTRA_MOCAP_BIMANUAL_TESTING_SUBJECTS": _ultramocap_motions(range(11, 14)),
}
