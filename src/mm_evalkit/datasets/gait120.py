"""Strict, local loader for the Gait120 joint-angle and EMG release."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.io as sio

from ..arrays import FloatArray

MOVEMENT = "LevelWalking"
N_PHASE_SAMPLES = 100
STEPS = (1, 2)
INVERTED_JOINTS = frozenset({"knee_angle_r", "knee_angle_l", "hip_adduction_l"})

TrialGroup = Literal["trial5", "trials1-4"]
AccumulatedProfiles = dict[int, dict[int, list[np.ndarray]]]

# Profiles indexed by subject id, then by zero-based step ordinal.
StepProfiles = dict[int, dict[int, FloatArray]]

# Gait120 EMG channel name for each MyoFullBody muscle it records.
Gait120_muscle_map = {
    "vaslat_r": "VastusLateralis",
    "recfem_r": "RectusFemoris",
    "vasmed_r": "VastusMedialis",
    "tibant_r": "TibialisAnterior",
    "bflh_r": "BicepsFemoris",
    "semiten_r": "Semitendinosus",
    "gasmed_r": "GastrocnemuisMedialis",
    "gaslat_r": "GastrocnemiusLateralis",
    "soleus_r": "SoleusMedialis",
    "perlong_r": "PeroneusLongus",
    "perbrev_r": "PeroneusBrevis",
}

# Every recorded muscle has a MyoFullBody counterpart, so a Gait120 analysis can
# use this full set rather than only the subset shared with Wang.
Gait120_muscle_names = list(Gait120_muscle_map)


@dataclass(frozen=True)
class Gait120Omission:
    """One unavailable Gait120 subject or paired trial/step input."""

    subject: int
    trial: int | None
    step: int | None
    trial_group: TrialGroup
    reason: str
    source: str


@dataclass(frozen=True)
class Gait120HumanData:
    """Selected Gait120 channels, indexed by subject and step."""

    subjects: tuple[int, ...]
    joint_names: tuple[str, ...]
    muscle_names: tuple[str, ...]
    joint_by_step: StepProfiles
    muscle_by_step: StepProfiles
    earlier_joint_by_step: StepProfiles
    earlier_muscle_by_step: StepProfiles
    joint_data: FloatArray
    muscle_data: FloatArray
    omissions: tuple[Gait120Omission, ...] = ()


@dataclass
class _TrialGroupData:
    """Profiles and schemas loaded from one independently validated trial group."""

    joint_by_step: StepProfiles
    muscle_by_step: StepProfiles
    joint_names: tuple[str, ...] | None
    muscle_names: tuple[str, ...] | None
    omissions: list[Gait120Omission]


def _subject_directory(root: Path, subject: int) -> Path:
    """Return the release directory containing one Gait120 subject."""
    first = (subject - 1) // 10 * 10 + 1
    return root / f"Gait120_{first:03d}_to_{first + 9:03d}" / f"S{subject:03d}"


def _normalize_profile(values: np.ndarray, *, source: str) -> np.ndarray:
    """Convert one finite signal vector to the 100-point gait-phase grid."""
    signal = np.asarray(values, dtype=float).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"{source} must be one signal vector, got shape {np.shape(values)}.")
    if signal.size < 2:
        raise ValueError(f"{source} must contain at least two samples.")
    if not np.isfinite(signal).all():
        raise ValueError(f"{source} contains non-finite samples.")

    profile = np.interp(
        np.linspace(0.0, 1.0, N_PHASE_SAMPLES),
        np.linspace(0.0, 1.0, signal.size),
        signal,
    )
    if profile.shape != (N_PHASE_SAMPLES,) or not np.isfinite(profile).all():
        raise ValueError(f"{source} did not produce a finite 100-point profile.")
    return profile


def _load_mot_step(path: Path) -> tuple[np.ndarray, tuple[str, ...]] | None:
    """Read one OpenSim MOT step; return ``None`` only when the file is absent."""
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read Gait120 MOT file {path}.") from exc

    endheader = next(
        (i for i, line in enumerate(lines) if line.strip().casefold() == "endheader"),
        None,
    )
    if endheader is None:
        raise ValueError(f"Gait120 MOT file {path} has no 'endheader' marker.")
    names_index = next(
        (i for i in range(endheader + 1, len(lines)) if lines[i].strip() and not lines[i].lstrip().startswith("#")),
        None,
    )
    if names_index is None:
        raise ValueError(f"Gait120 MOT file {path} has no channel-name row.")

    names = tuple(lines[names_index].split())
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Gait120 MOT file {path} must have non-empty unique channel names.")
    rows = [line.split() for line in lines[names_index + 1 :] if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) < 2:
        raise ValueError(f"Gait120 MOT file {path} must contain at least two data rows.")
    if any(len(row) != len(names) for row in rows):
        raise ValueError(f"Gait120 MOT file {path} has a data row that does not match its {len(names)} channels.")
    try:
        data = np.asarray(rows, dtype=float)
    except ValueError as exc:
        raise ValueError(f"Gait120 MOT file {path} contains a non-numeric value.") from exc
    if not np.isfinite(data).all():
        raise ValueError(f"Gait120 MOT file {path} contains non-finite samples.")

    return (
        np.stack([_normalize_profile(data[:, i], source=f"{path}:{name}") for i, name in enumerate(names)]),
        names,
    )


def _unwrap_singleton(value: Any) -> Any:
    """Unwrap MATLAB's repeated ``(1, 1)`` object containers."""
    while isinstance(value, np.ndarray) and value.shape == (1, 1):
        value = value[0, 0]
    return value


def _struct_field(structure: Any, field: str, *, source: str) -> Any | None:
    """Read an optional MATLAB struct field and reject malformed containers."""
    if not isinstance(structure, (np.ndarray, np.void)) or structure.dtype.names is None:
        raise ValueError(f"{source} is not a MATLAB structure.")
    if field not in structure.dtype.names:
        return None
    return _unwrap_singleton(structure[field])


def _load_subject_movement(path: Path) -> Any | None:
    """Read ``LevelWalking`` from ConvertedData.mat, or ``None`` if absent."""
    if not path.is_file():
        return None
    try:
        document = sio.loadmat(path)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise ValueError(f"Could not parse Gait120 EMG file {path}.") from exc
    if MOVEMENT not in document:
        raise ValueError(f"Gait120 EMG file {path} is missing {MOVEMENT!r}.")

    movement = document[MOVEMENT]
    if not isinstance(movement, np.ndarray) or movement.dtype.names is None:
        raise ValueError(f"{path}:{MOVEMENT} is not a MATLAB structure.")
    return movement


def _load_emg_step(
    movement: Any,
    *,
    trial: int,
    step: int,
    source: Path,
) -> tuple[np.ndarray, tuple[str, ...]] | None:
    """Read one EMG step; return ``None`` only when its declared field is absent."""
    trial_name = f"Trial{trial:02d}"
    step_name = f"Step{step:02d}"
    trial_data = _struct_field(
        movement,
        trial_name,
        source=f"{source}:{MOVEMENT}",
    )
    if trial_data is None:
        return None
    step_data = _struct_field(
        trial_data,
        step_name,
        source=f"{source}:{MOVEMENT}/{trial_name}",
    )
    if step_data is None:
        return None
    emg_data = _struct_field(
        step_data,
        "EMGs_interpolated",
        source=f"{source}:{MOVEMENT}/{trial_name}/{step_name}",
    )
    if emg_data is None:
        return None
    if not isinstance(emg_data, (np.ndarray, np.void)) or emg_data.dtype.names is None:
        raise ValueError(f"{source}:{MOVEMENT}/{trial_name}/{step_name}/EMGs_interpolated is not a MATLAB structure.")

    names = tuple(emg_data.dtype.names)
    if not names or len(set(names)) != len(names):
        raise ValueError(
            f"{source}:{MOVEMENT}/{trial_name}/{step_name}/EMGs_interpolated must have non-empty unique muscle names."
        )
    profiles = np.stack(
        [
            _normalize_profile(
                np.asarray(_unwrap_singleton(emg_data[name])),
                source=f"{source}:{MOVEMENT}/{trial_name}/{step_name}/{name}",
            )
            for name in names
        ]
    )
    return profiles, names


def _omission(
    *,
    subject: int,
    trial: int | None,
    step: int | None,
    trial_group: TrialGroup,
    reason: str,
    source: str | Path,
) -> Gait120Omission:
    """Create one stable omission record."""
    return Gait120Omission(
        subject=subject,
        trial=trial,
        step=step,
        trial_group=trial_group,
        reason=reason,
        source=str(source),
    )


def _validate_schema(
    expected: tuple[str, ...] | None,
    observed: tuple[str, ...],
    *,
    signal: str,
    trial_group: TrialGroup,
    subject: int,
    trial: int,
    step: int,
) -> tuple[str, ...]:
    """Require a stable channel order within one trial group."""
    if expected is not None and observed != expected:
        raise ValueError(
            f"Gait120 {trial_group} {signal} schema changed at subject {subject}, trial {trial}, step {step}."
        )
    return observed if expected is None else expected


def _average_trials(profiles: AccumulatedProfiles) -> StepProfiles:
    """Average repeated trials while retaining subject and step identity."""
    averaged: StepProfiles = {}
    for subject, steps in profiles.items():
        averaged[subject] = {}
        for step, values in steps.items():
            profile = np.mean(np.stack(values), axis=0)
            if profile.ndim != 2 or profile.shape[1] != N_PHASE_SAMPLES or not np.isfinite(profile).all():
                raise ValueError(
                    f"Gait120 subject {subject}, step {step + 1} did not produce finite 100-point profiles."
                )
            averaged[subject][step] = profile
    return averaged


def _load_trial_group(
    root: Path,
    *,
    subjects: Sequence[int],
    movements: Mapping[int, Any],
    trials: Sequence[int],
    trial_group: TrialGroup,
) -> _TrialGroupData:
    """Load paired steps and independently validate one trial group's schemas."""
    joints: AccumulatedProfiles = {}
    muscles: AccumulatedProfiles = {}
    joint_names: tuple[str, ...] | None = None
    muscle_names: tuple[str, ...] | None = None
    omissions: list[Gait120Omission] = []

    for subject in subjects:
        subject_dir = _subject_directory(root, subject)
        emg_path = subject_dir / "EMG" / "ConvertedData.mat"
        for trial in trials:
            for step in STEPS:
                mot_path = subject_dir / "JointAngle" / MOVEMENT / f"Trial{trial:02d}" / f"step{step:02d}.mot"
                joint_step = _load_mot_step(mot_path)
                muscle_step = _load_emg_step(
                    movements[subject],
                    trial=trial,
                    step=step,
                    source=emg_path,
                )
                if joint_step is not None:
                    joint_names = _validate_schema(
                        joint_names,
                        joint_step[1],
                        signal="joint",
                        trial_group=trial_group,
                        subject=subject,
                        trial=trial,
                        step=step,
                    )
                if muscle_step is not None:
                    muscle_names = _validate_schema(
                        muscle_names,
                        muscle_step[1],
                        signal="EMG",
                        trial_group=trial_group,
                        subject=subject,
                        trial=trial,
                        step=step,
                    )

                missing = []
                sources = []
                if joint_step is None:
                    missing.append("missing_joint_step")
                    sources.append(str(mot_path))
                if muscle_step is None:
                    missing.append("missing_emg_step")
                    sources.append(str(emg_path))
                if missing:
                    omissions.append(
                        _omission(
                            subject=subject,
                            trial=trial,
                            step=step,
                            trial_group=trial_group,
                            reason="+".join(missing),
                            source=" | ".join(sources),
                        )
                    )
                    continue

                step_index = step - 1
                joints.setdefault(subject, {}).setdefault(step_index, []).append(joint_step[0])
                muscles.setdefault(subject, {}).setdefault(step_index, []).append(muscle_step[0])

    return _TrialGroupData(
        joint_by_step=_average_trials(joints),
        muscle_by_step=_average_trials(muscles),
        joint_names=joint_names,
        muscle_names=muscle_names,
        omissions=omissions,
    )


def _required_indices(
    available: Sequence[str],
    requested: Sequence[str],
    *,
    signal: str,
    trial_group: TrialGroup,
) -> list[int]:
    """Resolve a declared channel selection against one validated schema."""
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Gait120 {trial_group} data is missing requested {signal} channel(s): {', '.join(missing)}.")
    return [available.index(name) for name in requested]


def _select_profiles(
    profiles: StepProfiles,
    available: tuple[str, ...],
    requested: Sequence[str],
    *,
    signal: Literal["joint", "EMG"],
    trial_group: TrialGroup,
) -> StepProfiles:
    """Select channels and apply the documented joint sign convention."""
    indices = _required_indices(
        available,
        requested,
        signal=signal,
        trial_group=trial_group,
    )
    invert = [i for i, name in enumerate(requested) if signal == "joint" and name in INVERTED_JOINTS]
    selected: StepProfiles = {}
    for subject, steps in profiles.items():
        selected[subject] = {}
        for step, values in steps.items():
            profile = np.asarray(values[indices, :], dtype=float)
            profile[invert, :] *= -1
            selected[subject][step] = profile
    return selected


def _collapse_subject_steps(
    profiles: StepProfiles,
    subjects: Sequence[int],
    *,
    n_channels: int,
    signal: str,
) -> np.ndarray:
    """Average available steps and enforce the public finite output contract."""
    output = np.stack([np.mean(np.stack(list(profiles[subject].values())), axis=0) for subject in subjects])
    expected = (len(subjects), n_channels, N_PHASE_SAMPLES)
    if output.shape != expected or not np.isfinite(output).all():
        raise ValueError(f"Gait120 {signal} output must be finite with shape {expected}, got {output.shape}.")
    return output


def _report_omissions(omissions: Sequence[Gait120Omission]) -> None:
    """Print every excluded source with its subject/trial/step identity."""
    if not omissions:
        return
    print(f"  Gait120 input omissions: {len(omissions)}")
    for item in omissions:
        trial = "all" if item.trial is None else f"{item.trial:02d}"
        step = "all" if item.step is None else f"{item.step:02d}"
        print(
            f"    subject={item.subject:03d} group={item.trial_group} "
            f"trial={trial} step={step}: {item.reason} ({item.source})"
        )


def load_gait120(
    dataset_dir: str | Path,
    *,
    requested_subjects: Sequence[int],
    selected_joint_names: Sequence[str],
    selected_muscle_names: Sequence[str],
    muscle_map: Mapping[str, str],
    missing_subject_policy: str,
) -> Gait120HumanData:
    """Load paired Trial-5 and Trials-1-4 Gait120 profiles.

    An absent joint or EMG source excludes only its paired
    ``(subject, trial, step)`` unit and is returned in ``omissions``. A present
    malformed source, inconsistent within-group schema, unavailable requested
    channel, or non-finite profile is an input error.
    """
    if missing_subject_policy not in {"drop-and-report", "error"}:
        raise ValueError(
            f"missing_subject_policy must be 'drop-and-report' or 'error', got {missing_subject_policy!r}."
        )
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Gait120 dataset directory does not exist: {root}")
    requested = [int(subject) for subject in requested_subjects]
    if not requested:
        raise ValueError("At least one Gait120 subject must be requested.")
    if len(set(requested)) != len(requested):
        raise ValueError("Gait120 requested_subjects contains duplicate subject IDs.")

    missing_mappings = [muscle for muscle in selected_muscle_names if muscle not in muscle_map]
    if missing_mappings:
        raise ValueError(f"Gait120 muscle mapping is missing model muscle(s): {', '.join(missing_mappings)}.")
    dataset_muscles = [muscle_map[muscle] for muscle in selected_muscle_names]

    movements: dict[int, Any] = {}
    omissions: list[Gait120Omission] = []
    for subject in requested:
        emg_path = _subject_directory(root, subject) / "EMG" / "ConvertedData.mat"
        movement = _load_subject_movement(emg_path)
        if movement is None:
            omissions.append(
                _omission(
                    subject=subject,
                    trial=None,
                    step=None,
                    trial_group="trial5",
                    reason="missing_converted_data",
                    source=emg_path,
                )
            )
        else:
            movements[subject] = movement

    primary = _load_trial_group(
        root,
        subjects=[subject for subject in requested if subject in movements],
        movements=movements,
        trials=(5,),
        trial_group="trial5",
    )
    omissions.extend(primary.omissions)
    loaded = [
        subject for subject in requested if subject in primary.joint_by_step and subject in primary.muscle_by_step
    ]
    missing_subjects = [subject for subject in requested if subject not in loaded]
    already_recorded = {item.subject for item in omissions if item.trial_group == "trial5" and item.trial is None}
    for subject in missing_subjects:
        if subject not in already_recorded:
            omissions.append(
                _omission(
                    subject=subject,
                    trial=None,
                    step=None,
                    trial_group="trial5",
                    reason="no_valid_paired_step",
                    source=_subject_directory(root, subject),
                )
            )

    if missing_subjects and missing_subject_policy == "error":
        _report_omissions(omissions)
        raise ValueError(f"Gait120 Trial-5 loading could not retain requested subject(s): {missing_subjects}.")
    if not loaded or primary.joint_names is None or primary.muscle_names is None:
        _report_omissions(omissions)
        raise ValueError("Gait120 Trial-5 loading produced no paired subject/step data.")

    joint_by_step = _select_profiles(
        primary.joint_by_step,
        primary.joint_names,
        selected_joint_names,
        signal="joint",
        trial_group="trial5",
    )
    muscle_by_step = _select_profiles(
        primary.muscle_by_step,
        primary.muscle_names,
        dataset_muscles,
        signal="EMG",
        trial_group="trial5",
    )

    earlier = _load_trial_group(
        root,
        subjects=loaded,
        movements=movements,
        trials=(1, 2, 3, 4),
        trial_group="trials1-4",
    )
    omissions.extend(earlier.omissions)
    if (
        earlier.joint_by_step
        and earlier.muscle_by_step
        and earlier.joint_names is not None
        and earlier.muscle_names is not None
    ):
        earlier_joint_by_step = _select_profiles(
            earlier.joint_by_step,
            earlier.joint_names,
            selected_joint_names,
            signal="joint",
            trial_group="trials1-4",
        )
        earlier_muscle_by_step = _select_profiles(
            earlier.muscle_by_step,
            earlier.muscle_names,
            dataset_muscles,
            signal="EMG",
            trial_group="trials1-4",
        )
    else:
        earlier_joint_by_step = {}
        earlier_muscle_by_step = {}

    for subject in loaded:
        if subject not in earlier_joint_by_step or subject not in earlier_muscle_by_step:
            omissions.append(
                _omission(
                    subject=subject,
                    trial=None,
                    step=None,
                    trial_group="trials1-4",
                    reason="no_valid_paired_step",
                    source=_subject_directory(root, subject),
                )
            )

    joint_data = _collapse_subject_steps(
        joint_by_step,
        loaded,
        n_channels=len(selected_joint_names),
        signal="joint",
    )
    muscle_data = _collapse_subject_steps(
        muscle_by_step,
        loaded,
        n_channels=len(selected_muscle_names),
        signal="EMG",
    )
    _report_omissions(omissions)
    return Gait120HumanData(
        subjects=tuple(loaded),
        joint_names=tuple(selected_joint_names),
        muscle_names=tuple(selected_muscle_names),
        joint_by_step=joint_by_step,
        muscle_by_step=muscle_by_step,
        earlier_joint_by_step=earlier_joint_by_step,
        earlier_muscle_by_step=earlier_muscle_by_step,
        joint_data=joint_data,
        muscle_data=muscle_data,
        omissions=tuple(omissions),
    )


def load_gait120_emg_step(
    dataset_dir: str | Path,
    *,
    subject: int,
    step_index: int,
    selected_muscle_names: Sequence[str],
    muscle_map: Mapping[str, str],
    trial: int = 5,
) -> np.ndarray:
    """Load one subject/trial/step EMG profile for synchronized rendering.

    Unlike :func:`load_gait120`, this focused reader does not load the complete
    cohort or trials 1--4. Its output has shape ``(channels, 100)`` and remains
    explicitly a gait-phase profile rather than a time-domain recording.
    """
    if step_index not in {0, 1}:
        raise ValueError(f"step_index must be 0 or 1, got {step_index}.")
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Gait120 dataset directory does not exist: {root}")
    emg_path = _subject_directory(root, subject) / "EMG" / "ConvertedData.mat"
    movement = _load_subject_movement(emg_path)
    if movement is None:
        raise FileNotFoundError(f"Gait120 EMG source does not exist: {emg_path}")
    loaded = _load_emg_step(
        movement,
        trial=trial,
        step=step_index + 1,
        source=emg_path,
    )
    if loaded is None:
        raise FileNotFoundError(
            f"Gait120 subject {subject:03d} Trial{trial:02d} Step{step_index + 1:02d} has no EMG profile."
        )
    profiles, available = loaded
    dataset_names = []
    for muscle in selected_muscle_names:
        if muscle not in muscle_map:
            raise ValueError(f"No Gait120 electrode mapping is declared for {muscle!r}.")
        dataset_names.append(muscle_map[muscle])
    indices = _required_indices(
        available,
        dataset_names,
        signal="EMG",
        trial_group="trial5",
    )
    return np.asarray(profiles[indices], dtype=float)
