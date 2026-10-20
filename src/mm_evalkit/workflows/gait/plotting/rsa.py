"""Population and subject-level representational-similarity figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import add_policy_title, add_policy_ytick_markers, draw_rsm


def _subject_labels(analysis, n_subjects: int) -> list[str]:
    """Tick labels for the human rows of an RSA matrix.

    A population-only cohort (Wang) leaves ``subjects`` empty because the
    per-subject stages key off it, so the loaded identities are read from
    ``all_subjects``. If neither lines up with the matrix, the rows are numbered
    rather than labelled with a subject ID the cohort does not carry.
    """
    identities = list(analysis.human.subjects) or list(analysis.human.all_subjects)
    if len(identities) != n_subjects:
        return [f"#{position + 1}" for position in range(n_subjects)]
    return [f"Subj {identity}" for identity in identities]


def plot_rsa_results(analysis):
    """Generate plots for RSA results."""
    fig = plt.figure(figsize=(12, 10))
    human_policy_matrix = np.concatenate(
        [analysis.human_metrics.rsa_matrix] + [run.population.rsa_matrix for run in analysis.policies],
        axis=0,
    )

    plt.imshow(human_policy_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")

    plt.title("Human-to-Human Muscle RSA Matrix (Spearman)")
    # With many subjects, per-cell numbers and per-subject tick labels become
    # unreadable, so keep only the colored matrix in that regime.
    n_subjects = len(analysis.human_metrics.rsa_matrix)
    annotate = n_subjects <= analysis.style.rsa_matrix_annotate_max
    if annotate:
        subject_labels = _subject_labels(analysis, n_subjects)
        plt.xticks(np.arange(n_subjects), subject_labels, rotation=45)
        # The policy rows get a colored square instead of a text label.
        plt.yticks(
            np.arange(n_subjects + len(analysis.policies)),
            subject_labels + [""] * len(analysis.policies),
        )
        add_policy_ytick_markers(analysis, plt.gca(), n_subjects)
    else:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Subjects")
        plt.ylabel("Subjects / policies")
    plt.colorbar(label="RSA (Correlation of RSMs)")

    if annotate:
        for i in range(human_policy_matrix.shape[0]):
            for j in range(np.min([i, human_policy_matrix.shape[1]])):
                val = human_policy_matrix[i, j]
                plt.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(val) < 0.6 and val != 0 else "white",
                    zorder=3,
                )

    plt.savefig(f"{analysis.plots_dir}/human_to_human_muscle_rsa_matrix.svg", bbox_inches="tight")
    plt.close(fig)

    (Path(analysis.plots_dir) / "rsms").mkdir(parents=True, exist_ok=True)
    n_subjects = analysis.human_metrics.rsa_matrix.shape[0]
    n_policies = len(analysis.policies)
    display_muscle_names = [analysis.muscle_map[m] for m in analysis.channels.muscle_names]
    fig, axs = plt.subplots(n_policies + 1, 1, figsize=(5, 5 * (n_policies + 1)))

    draw_rsm(
        analysis, axs[0], analysis.human.muscle_rsms_avg, analysis.channels.muscle_names, display_muscle_names, "Human"
    )
    axs[0].set_xticks([])

    for j in range(n_policies):
        policy_rsa_matrix = analysis.policies[j].profiles.population_muscle_rsm
        draw_rsm(
            analysis,
            axs[j + 1],
            policy_rsa_matrix,
            analysis.channels.muscle_names,
            display_muscle_names,
            "",
        )
        add_policy_title(analysis, axs[j + 1], j)
        axs[j + 1].set_xticks([])

    fig.savefig(f"{analysis.plots_dir}/rsms/human_to_human_muscle_rsa_matrix_population.svg")
    plt.close(fig)


def plot_subject_based_rsa_results(analysis):
    """Generate plots for RSA results."""
    if not analysis.subject_based:
        print("Subject-based plotting is disabled. Set subject_based=True to enable.")
        return

    fig = plt.figure(figsize=(12, 10))
    human_policy_matrix = np.concatenate(
        [analysis.human_metrics.rsa_matrix] + [run.per_subject.rsa_matrix for run in analysis.policies],
        axis=0,
    )
    masked_human_policy_matrix = np.ma.masked_invalid(human_policy_matrix)

    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="lightgrey")
    plt.imshow(masked_human_policy_matrix, vmin=-1, vmax=1, cmap=cmap, aspect="auto")

    plt.title("Human-to-Human Muscle RSA Matrix (Spearman)")
    # With many subjects, per-cell numbers and per-subject tick labels become
    # unreadable, so keep only the colored matrix in that regime.
    n_subjects = len(analysis.human_metrics.rsa_matrix)
    annotate = n_subjects <= analysis.style.rsa_matrix_annotate_max
    if annotate:
        subject_labels = _subject_labels(analysis, n_subjects)
        plt.xticks(np.arange(n_subjects), subject_labels, rotation=45)
        # The policy rows get a colored square instead of a text label.
        plt.yticks(
            np.arange(n_subjects + len(analysis.policies)),
            subject_labels + [""] * len(analysis.policies),
        )
        add_policy_ytick_markers(analysis, plt.gca(), n_subjects)
    else:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Subjects")
        plt.ylabel("Subjects / policies")
    plt.colorbar(label="RSA (Correlation of RSMs)")

    if annotate:
        for i in range(human_policy_matrix.shape[0]):
            for j in range(np.min([i, human_policy_matrix.shape[1]])):
                val = human_policy_matrix[i, j]
                if not np.isfinite(val):
                    continue
                plt.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(val) < 0.6 and val != 0 else "white",
                    zorder=3,
                )

    plt.savefig(f"{analysis.plots_dir}/human_to_human_muscle_rsa_matrix_per_subject.svg", bbox_inches="tight")
    plt.close(fig)

    (Path(analysis.plots_dir) / "rsms").mkdir(parents=True, exist_ok=True)
    # One column per subject: restrict to the fixed random subset to stay legible.
    plot_positions = analysis.human.plot_subject_positions
    n_subjects = len(plot_positions)
    n_policies = len(analysis.policies)
    display_muscle_names = [analysis.muscle_map[m] for m in analysis.channels.muscle_names]
    fig, axs = plt.subplots(n_policies + 1, n_subjects, figsize=(4 * n_subjects, 4 * (n_policies + 1)), squeeze=False)

    for col_idx, i in enumerate(plot_positions):
        subj_id = analysis.human.subjects[i]
        draw_rsm(
            analysis,
            axs[0, col_idx],
            analysis.human.muscle_rsms[i],
            analysis.channels.muscle_names,
            display_muscle_names,
            f"Human-Subj {subj_id}",
        )
        axs[0, col_idx].set_xticks([])
        if col_idx > 0:
            axs[0, col_idx].set_yticks([])

        for j in range(n_policies):
            if (
                analysis.policies[j].profiles.subject_ids is not None
                and subj_id in analysis.policies[j].profiles.subject_ids
            ):
                policy_rsa_matrix = analysis.policies[j].profiles.per_subject_muscle_rsm[subj_id]
                draw_rsm(
                    analysis,
                    axs[j + 1, col_idx],
                    policy_rsa_matrix,
                    analysis.channels.muscle_names,
                    display_muscle_names,
                    "",
                )
                add_policy_title(
                    analysis,
                    axs[j + 1, col_idx],
                    j,
                    suffix=f"-Subj {subj_id}",
                )
                axs[j + 1, col_idx].set_xticks([])
                if col_idx > 0:
                    axs[j + 1, col_idx].set_yticks([])
            else:
                axs[j + 1, col_idx].axis("off")
    fig.savefig(f"{analysis.plots_dir}/rsms/human_to_human_muscle_rsa_matrix_per_subject.svg")
    plt.close(fig)
