"""Representational-similarity matrix figures for ULTRA-MoCap."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from .constants import RSA_PAIRS_NOTE


def plot_rsm_matrices(policy_records, policies, real_records, channel_labels, plots_dir):
    """Per movement/condition, the RSM the RSA score is computed from.

    The bimanual counterpart of the full-body analysis's ``rsms/`` figures:
    one panel for the real EMG and one per policy, each the electrode x electrode
    correlation matrix averaged over the subjects that recorded that condition. With only
    three electrodes every entry is annotated, so the figure *is* the data -- the RSA
    number underneath is the rank correlation of the three off-diagonal cells against the
    real panel's.
    """
    by_group = {}
    for movement, condition, subject in real_records:
        by_group.setdefault((movement, condition), []).append(subject)

    n_saved = 0
    for (movement, condition), subjects in sorted(by_group.items()):
        keys = [(movement, condition, subject) for subject in sorted(subjects)]

        def mean_rsm(records, name, keys=keys):
            matrices = [records[k][name] for k in keys if k in records and name in records[k]]
            return np.mean(matrices, axis=0) if matrices else None

        panels = [("Real EMG", mean_rsm(real_records, "emg_rsm"))]
        panels += [(policy.label, mean_rsm(policy_records.get(policy.id, {}), "act_rsm")) for policy in policies]
        panels = [(label, matrix) for label, matrix in panels if matrix is not None]
        if len(panels) < 2:
            continue

        fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels) + 1.0, 3.9), squeeze=False)
        for ax, (label, matrix) in zip(axes[0], panels, strict=True):
            ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
            ax.set_xticks(range(len(channel_labels)), channel_labels, fontsize=7, rotation=35, ha="right")
            ax.set_yticks(range(len(channel_labels)), channel_labels, fontsize=7)
            ax.set_title(label.replace("\n", " "), fontsize=9)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(matrix[i, j]) > 0.55 else "0.15",
                    )
        fig.suptitle(f"{movement} {condition} - muscle RSMs, mean over {len(subjects)} subject(s)", fontsize=11)
        fig.text(
            0.005,
            0.005,
            "Electrode x electrode correlation over the average cycle; RSA compares the "
            f"lower triangles against the real panel ({RSA_PAIRS_NOTE}).",
            fontsize=7,
            va="bottom",
            ha="left",
            color="0.3",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        out_path = os.path.join(plots_dir, f"{movement}_{condition}", "rsms.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        n_saved += 1
    print(f"saved {n_saved} muscle RSM figures")
