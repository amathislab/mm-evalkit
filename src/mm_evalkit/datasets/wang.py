"""Strict Wang human-kinematics and EMG loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from ..arrays import FloatArray

# Wang EMG channel name for each MyoFullBody muscle it records.
Wang_muscle_map = {
    "soleus_r": "Soleus",
    "gasmed_r": "Medial Gastrocnemius",
    "gaslat_r": "Lateral Gastrocnemius",
    "tibant_r": "Tibialis Anterior",
    "semimem_r": "Semimembranosus",
    "bflh_r": "Biceps Femoris",
    "vaslat_r": "Vastus Lateral",
    "recfem_r": "Rectus Femoris",
    "vasmed_r": "Vastus Medial",
}


@dataclass(frozen=True)
class WangHumanData:
    """Selected Wang population channels."""

    joint_names: tuple[str, ...]
    muscle_names: tuple[str, ...]
    joint_data: FloatArray
    muscle_data: FloatArray
    subjects: tuple[int, ...] = ()


@dataclass(frozen=True)
class WangOmission:
    """One cohort subject excluded because its source recording is unusable."""

    subject: int
    source: Path
    detail: str


MOVEMENT = "walk_45"
SUBJECTS = tuple(range(4, 13))
PHASE_POINTS = 100
JOINT_NAMES = (
    "time",
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
)
MUSCLE_NAMES = ("Time", *Wang_muscle_map.values())


def _load_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the synchronized right-side averages from one source MAT file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required Wang source file does not exist: {path}")

    try:
        dataset = sio.loadmat(path)
        averaged = dataset["Datastr"]["Resample"][0][0]["Sych"][0][0]["Average"][0][0]
        emg = np.asarray(averaged["EMG"][0][0]["ave_r"][0][0], dtype=float)
        joints = np.asarray(averaged["IKAngData"][0][0]["ave_r"][0][0], dtype=float)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} does not contain the expected Wang synchronized-average structure.") from exc
    except OSError as exc:
        raise ValueError(f"{path} could not be read to the end; the file is truncated or corrupt ({exc}).") from exc

    if joints.shape != (PHASE_POINTS, len(JOINT_NAMES)):
        raise ValueError(
            f"{path} IKAngData/ave_r must have shape ({PHASE_POINTS}, {len(JOINT_NAMES)}), got {joints.shape}."
        )
    if emg.shape != (PHASE_POINTS, len(MUSCLE_NAMES)):
        raise ValueError(f"{path} EMG/ave_r must have shape ({PHASE_POINTS}, {len(MUSCLE_NAMES)}), got {emg.shape}.")
    if not np.isfinite(joints).all():
        raise ValueError(f"{path} IKAngData/ave_r contains non-finite values.")
    if not np.isfinite(emg).all():
        raise ValueError(f"{path} EMG/ave_r contains non-finite values.")
    return joints, emg


def _report_omissions(omissions: Sequence[WangOmission]) -> None:
    """Print every excluded cohort subject with the source that failed to load."""
    if not omissions:
        return
    print(f"  Wang input omissions: {len(omissions)}")
    for item in omissions:
        print(f"    subject={item.subject:02d} movement={MOVEMENT}: {item.detail}")


def load_wang_profiles(
    dataset_dir: str | Path,
    *,
    missing_subject_policy: str = "drop-and-report",
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    """Load the declared Subj04--Subj12 ``walk_45`` cohort from source MAT files.

    The loader intentionally does not read previously materialized ``.npy``
    arrays.  Such caches can outlive or disagree with the requested source
    recording, so an absent source file is always an input error.

    A source file that is present but unusable -- truncated, corrupt, or
    malformed -- excludes only its own subject under the default
    ``drop-and-report`` policy; every exclusion is printed, and the retained
    subject IDs are returned so the cohort size is never silently assumed.
    Pass ``missing_subject_policy="error"`` to require the full cohort.
    """
    if missing_subject_policy not in {"drop-and-report", "error"}:
        raise ValueError(
            f"missing_subject_policy must be 'drop-and-report' or 'error', got {missing_subject_policy!r}."
        )
    root = Path(dataset_dir)
    subject_profiles: list[tuple[np.ndarray, np.ndarray]] = []
    retained: list[int] = []
    omissions: list[WangOmission] = []
    for subject in SUBJECTS:
        path = root / f"Subj{subject:02d}" / f"Subj{subject:02d}_{MOVEMENT}.mat"
        try:
            subject_profiles.append(_load_subject(path))
        except ValueError as exc:
            omissions.append(WangOmission(subject=subject, source=path, detail=str(exc)))
            continue
        retained.append(subject)

    _report_omissions(omissions)
    if omissions and missing_subject_policy == "error":
        excluded = [item.subject for item in omissions]
        raise ValueError(f"Wang {MOVEMENT} loading could not retain cohort subject(s): {excluded}.")
    if not subject_profiles:
        raise ValueError(f"Wang {MOVEMENT} loading retained no usable subject recordings in {root}.")

    joints = np.stack([profile[0] for profile in subject_profiles]).transpose(0, 2, 1)
    emg = np.stack([profile[1] for profile in subject_profiles]).transpose(0, 2, 1)

    for name in ("knee_angle_r", "knee_angle_l"):
        joints[:, JOINT_NAMES.index(name), :] *= -1
    return joints, emg, JOINT_NAMES, MUSCLE_NAMES, tuple(retained)


def _channel_indices(
    available: Sequence[str],
    requested: Sequence[str],
    *,
    signal: str,
) -> list[int]:
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Wang data is missing requested {signal} channel(s): {', '.join(missing)}.")
    return [available.index(name) for name in requested]


def load_wang(
    dataset_dir: str | Path,
    *,
    selected_joint_names: Sequence[str],
    selected_muscle_names: Sequence[str],
    muscle_map: Mapping[str, str],
    missing_subject_policy: str = "drop-and-report",
) -> WangHumanData:
    """Load and channel-select the population-only Wang dataset."""
    joint_data, muscle_data, available_joints, available_muscles, subjects = load_wang_profiles(
        dataset_dir,
        missing_subject_policy=missing_subject_policy,
    )
    joint_indices = _channel_indices(
        available_joints,
        selected_joint_names,
        signal="joint",
    )
    dataset_muscles = [muscle_map[name] for name in selected_muscle_names]
    muscle_indices = _channel_indices(
        available_muscles,
        dataset_muscles,
        signal="EMG",
    )
    return WangHumanData(
        joint_names=tuple(selected_joint_names),
        muscle_names=tuple(selected_muscle_names),
        joint_data=np.asarray(joint_data[:, joint_indices, :]),
        muscle_data=np.asarray(muscle_data[:, muscle_indices, :]),
        subjects=subjects,
    )
