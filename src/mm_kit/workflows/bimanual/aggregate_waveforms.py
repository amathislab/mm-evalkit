"""Subject-aggregated joint and muscle waveform figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mm_kit.workflows.bimanual.segmentation import plot_average_cycles


def _stack_cycles(records, keys, field):
    """Return mean and standard deviation over available subject cycles."""
    arrays = [records[key][field] for key in keys if key in records and field in records[key]]
    if not arrays:
        return None
    stacked = np.asarray(arrays)
    return stacked.mean(axis=0), stacked.std(axis=0)


def plot_aggregated_waveforms(
    policy_records,
    policies,
    real_records,
    joint_names,
    muscle_labels,
    plots_dir,
    *,
    arm="right",
):
    """Plot each condition averaged over subjects.

    Each subject contributes one average cycle. Shaded bands therefore show
    between-subject spread, not repetition-level variability.
    """
    subjects_by_condition = {}
    for movement, condition, subject in real_records:
        subjects_by_condition.setdefault((movement, condition), []).append(subject)

    output_root = Path(plots_dir)
    saved = 0
    for (movement, condition), subjects in sorted(subjects_by_condition.items()):
        if len(subjects) < 2:
            continue
        output_dir = output_root / f"{movement}_{condition}"
        output_dir.mkdir(parents=True, exist_ok=True)
        ordered_subjects = sorted(subjects)
        keys = [(movement, condition, subject) for subject in ordered_subjects]
        tag = f" - {len(subjects)} subjects ({', '.join(ordered_subjects)})"

        real_joints = _stack_cycles(real_records, keys, "ref_cycle")
        policy_joints = [
            (
                policy,
                _stack_cycles(
                    policy_records.get(policy.id, {}),
                    keys,
                    "pol_cycle",
                ),
            )
            for policy in policies
        ]
        policy_joints = [(policy, values) for policy, values in policy_joints if values is not None]
        if real_joints is not None and policy_joints:
            plot_average_cycles(
                joint_names,
                [("Real (retargeted)", "0.4", *real_joints, "--")]
                + [
                    (
                        policy.label,
                        policy.color,
                        *values,
                        policy.linestyle,
                    )
                    for policy, values in policy_joints
                ],
                title=(f"{movement} {condition} ({arm} arm) - joint angles{tag}"),
                out_path=output_dir / "aggregated_joints.png",
                ylabel="angle (deg)",
            )
            saved += 1

        real_emg = _stack_cycles(real_records, keys, "emg_cycle")
        policy_activation = [
            (
                policy,
                _stack_cycles(
                    policy_records.get(policy.id, {}),
                    keys,
                    "act_cycle",
                ),
            )
            for policy in policies
        ]
        policy_activation = [(policy, values) for policy, values in policy_activation if values is not None]
        if policy_activation:
            plot_average_cycles(
                muscle_labels,
                [
                    (
                        policy.label,
                        policy.color,
                        *values,
                        policy.linestyle,
                    )
                    for policy, values in policy_activation
                ],
                title=f"{movement} {condition} ({arm} arm) - muscles{tag}",
                out_path=output_dir / "aggregated_muscles.png",
                ylabel="policy activation",
                twin_series=([("Real EMG (MVC-norm)", "0.4", *real_emg)] if real_emg is not None else None),
                twin_ylabel=("EMG (fraction MVC)" if real_emg is not None else ""),
            )
            saved += 1
    print(f"saved {saved} subject-aggregated waveform figures (band = between-subject spread)")
