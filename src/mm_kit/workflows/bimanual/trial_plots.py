"""Raw and repetition-averaged waveform figures for ULTra-MoCap trials."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from mm_kit.workflows.bimanual.segmentation import plot_average_cycles

from .constants import CONTROL_FREQUENCY_HZ, HEAD_COLORS
from .references import select_reference_record


def plot_joint_trial(policy_qpos, ref_qpos, joint_names, title, out_path):
    """One subplot per joint: policy vs real (GMR-retargeted) joint angles (deg)."""
    n = len(joint_names)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), squeeze=False)
    t = np.arange(policy_qpos.shape[0]) / CONTROL_FREQUENCY_HZ

    for idx, name in enumerate(joint_names):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot(t, ref_qpos[:, idx], color="0.4", lw=1.6, ls="--", label="Real (retargeted)")
        ax.plot(t, policy_qpos[:, idx], color="tab:blue", lw=1.4, label="Policy")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("angle (deg)", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("time (s)", fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_muscle_trial(policy_act, emg_df, muscle_entries, title, out_path):
    """One subplot per simulated muscle: its policy activation vs the real EMG.

    ``muscle_entries`` is one ``(muscle, activation column, EMG channel, channel
    label)`` per simulated muscle -- every head of a group gets its own subplot
    rather than being averaged into one trace, because the heads of a group do not
    behave alike (e.g. BIClong and BICshort correlate with the same electrode with
    opposite sign). The heads of a group therefore share the same real EMG trace,
    which is the honest picture: one electrode, several candidate muscles.

    Real EMG (subject-task-peak-normalized envelope) and policy activation have different units,
    so they share the time axis on twin y-axes.

    ``emg_df`` is already on the rollout time grid. Unrecorded edge samples are
    NaN and appear as gaps instead of being shifted or filled.
    """
    n_muscles = len(muscle_entries)
    ncols = min(4, n_muscles)
    nrows = math.ceil(n_muscles / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows), squeeze=False)

    # Restrict to the frames both signals cover so twin-axis traces stay aligned.
    n = min(policy_act.shape[0], len(emg_df))
    policy_act = policy_act[:n]
    t = np.arange(n) / CONTROL_FREQUENCY_HZ

    for idx, (muscle, column, channel, label) in enumerate(muscle_entries):
        ax = axes[idx // ncols][idx % ncols]

        (l_policy,) = ax.plot(t, policy_act[:, column], color="tab:blue", lw=1.4, label="Policy activation")
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("policy activation", fontsize=8, color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"{muscle}\n({channel}; {label})", fontsize=9)

        ax_emg = ax.twinx()
        (l_emg,) = ax_emg.plot(
            t,
            emg_df[channel].to_numpy()[:n],
            color="0.4",
            lw=1.6,
            ls="--",
            label="Real EMG (subject-peak norm)",
        )
        ax_emg.set_ylabel("EMG (fraction subject task peak)", fontsize=8, color="0.4")
        ax_emg.tick_params(axis="y", labelcolor="0.4", labelsize=7)

        lines = [l_policy, l_emg]
        ax.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="upper right")

    for j in range(n_muscles, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_trial_waveforms(entries, groups, joint_sel_names, muscle_labels, title, joints_path, muscles_path):
    """Overlay every policy's average (+/-std) cycle for one trial on the real data.

    ``entries`` is ``[(policy, metrics)]`` in config order -- the policies that
    produced a segmentable rollout of this trial. Joint and sEMG references are
    selected independently because uncovered EMG edges can retain fewer cycles.

    With a single policy the individual muscle heads are drawn thin behind the group
    mean, so their disagreement stays visible. With several policies they are left
    out: three heads per electrode per policy is unreadable, and the comparison is
    between the group means. ``--raw_trials`` always keeps the per-head view.
    """
    reference = select_reference_record(entries)

    joint_counts = sorted({metrics["n_joint_cycles"] for _, metrics in entries})
    joint_cycle_tag = (
        f" - avg of {joint_counts[0]} reps"
        if len(joint_counts) == 1
        else f" - avg of {joint_counts[0]}-{joint_counts[-1]} reps"
    )

    # --- Joints: real vs every policy ---
    plot_average_cycles(
        joint_sel_names,
        [
            (
                "Real (retargeted)",
                "0.4",
                reference["ref_cycle"],
                reference["ref_std"],
                "--",
            )
        ]
        + [
            (policy.label, policy.color, metrics["pol_cycle"], metrics["pol_std"], policy.linestyle)
            for policy, metrics in entries
        ],
        title=f"{title} joint angles{joint_cycle_tag}",
        out_path=joints_path,
        ylabel="angle (deg)",
    )
    print(f"  saved {joints_path}")

    # --- Muscles: one subplot per electrode, group means vs the real EMG ---
    heads = None
    if len(entries) == 1:
        head_means = entries[0][1]["head_means"]
        heads = [
            [
                (muscle, HEAD_COLORS[slot % len(HEAD_COLORS)], head_means[index])
                for slot, (muscle, index) in enumerate(zip(group.muscles, group.cycle_columns, strict=True))
            ]
            for group in groups
        ]
    twin, twin_ylabel = None, ""
    if "emg_cycle" in reference:
        twin = [
            (
                "Real EMG (subject-peak norm)",
                "0.4",
                reference["emg_cycle"],
                reference["emg_std"],
            )
        ]
        twin_ylabel = "EMG (fraction subject task peak)"
    muscle_counts = sorted({metrics["n_muscle_cycles"] for _, metrics in entries if metrics["n_muscle_cycles"]})
    muscle_cycle_tag = (
        ""
        if not muscle_counts
        else (
            f" - avg of {muscle_counts[0]} paired reps"
            if len(muscle_counts) == 1
            else (f" - avg of {muscle_counts[0]}-{muscle_counts[-1]} paired reps")
        )
    )
    plot_average_cycles(
        muscle_labels,
        [
            (policy.label, policy.color, metrics["act_cycle"], metrics["act_std"], policy.linestyle)
            for policy, metrics in entries
        ],
        title=f"{title} muscles{muscle_cycle_tag}",
        out_path=muscles_path,
        ylabel="policy activation",
        twin_series=twin,
        twin_ylabel=twin_ylabel,
        per_channel_series=heads,
    )
    print(f"  saved {muscles_path}")
