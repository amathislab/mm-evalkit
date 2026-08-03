"""Compact tabular figure summarizing all full-body metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from ..constants import (
    BETWEEN_SUBJECT_HATCH,
    BETWEEN_SUBJECT_LABEL,
    EARLIER_HUMAN_SENSITIVITY_LABEL,
    POLICY_WITHIN_SUBJECT_HATCH,
    POLICY_WITHIN_SUBJECT_LABEL,
    REF_GREY,
    RSA_NOISE_CEILING_HATCH,
    RSA_NOISE_CEILING_LABEL,
    WITHIN_SUBJECT_HATCH,
    WITHIN_SUBJECT_LABEL,
)
from .style import policy_color_rgba

REFERENCE_SCOPES = {
    WITHIN_SUBJECT_LABEL: "subject_matched",
    BETWEEN_SUBJECT_LABEL: "population",
    RSA_NOISE_CEILING_LABEL: "subject_matched",
    POLICY_WITHIN_SUBJECT_LABEL: "subject_matched",
    EARLIER_HUMAN_SENSITIVITY_LABEL: "subject_matched",
}


def _scoped_reference_metrics(scope, metrics, *, empty_metric):
    """Place one reference value per metric in its valid analysis scope.

    ``metrics`` contains the joint, EMG, and RSA values in that order.  Human
    between-subject references belong to the population columns; reliability,
    sensitivity, and leave-one-subject-out references are subject-matched.
    Leaving the other column empty is important: duplicating a reference into
    both columns visually claims that two distinct analyses were performed.
    """
    if scope == "population":
        return [item for metric in metrics for item in (metric, empty_metric)]
    if scope == "subject_matched":
        return [item for metric in metrics for item in (empty_metric, metric)]
    raise ValueError(f"Unknown summary-table reference scope: {scope!r}.")


def plot_metrics_table(analysis):
    """Save a summary table (as a figure) of average similarity metrics.

    One primary row per policy plus its available sensitivity/reference rows, preceded
    by the grey human reference benchmarks: human within-subject reliability (Trial 5
    vs Trials 1-4), between-subject pairwise similarity, and the RSA leave-one-subject-out
    noise ceiling. Each policy is followed by the subject-matched sensitivity analysis
    (Policy Trial 5 vs mean Human Trials 1-4) and its optional within-subject variability
    reference (Policy Trial 5 vs its Trials 1-4 rollouts). Metrics are grouped in columns
    as Kinematics (joint angle RMSE), optional Kinetics (joint-moment and vertical-GRF
    RMSE), and Neuromuscular (EMG correlation and muscle RSA), each split into
    Population and Subject-matched sub-columns. A reference value is
    shown only in the scope where its comparison is defined; the other cell is marked
    as not applicable.
    """

    def _finite(vals):
        vals = np.ravel(vals) if isinstance(vals, list | tuple | np.ndarray) else [vals]
        return [float(v) for v in vals if v is not None and np.isfinite(v)]

    def _mean(vals):
        vals = _finite(vals)
        return float(np.mean(vals)) if vals else np.nan

    def _std(vals):
        vals = _finite(vals)
        return float(np.std(vals)) if vals else np.nan

    def _ref_row(label, hatch, metrics):
        # A grey reference row: no policy tint, rendered with a hatch texture.
        return {"label": label, "metrics": metrics, "is_ref": True, "hatch": hatch, "policy_idx": None}

    has_kinetics = analysis.kinetics is not None
    metric_names = ["Joint Angle Error [deg]"]
    if has_kinetics:
        metric_names.extend(("Joint Moment Error [Nm/kg]", "Vertical GRF Error [BW]"))
    metric_names.extend(("EMG-Muscle Correlation", "EMG-Muscle RSA (Spearman)"))

    def _layer_metrics(kinematics, emg, rsa, *, torque=None, grf=None):
        values = [kinematics]
        if has_kinetics:
            values.extend((torque or nan_metric, grf or nan_metric))
        values.extend((emg, rsa))
        return values

    # Each row carries population/subject-matched pairs for every active metric
    # group (angle, optional moment/GRF, EMG, and RSA). Every entry is a
    # (mean, samples) pair: the mean is the already-computed
    # summary and the samples are the population it was averaged over, used only for the
    # +- std. Reference rows (``is_ref``) occupy only the population or subject-matched
    # cell where their design applies; policy rows can carry both scopes.
    nan_metric = (np.nan, [])

    # Between-subject (pairwise human-human): between-subject variability.
    hh_kin = (
        analysis.human_metrics.avg_pairwise_joint_rmse["all_joints"],
        analysis.human_metrics.pairwise_joint_rmse["all_joints"],
    )
    hh_emg = (
        analysis.human_metrics.avg_pairwise_muscle_corr["all_muscles"],
        analysis.human_metrics.pairwise_muscle_corr["all_muscles"],
    )
    hh_rsa = (
        analysis.human_metrics.avg_pairwise_muscle_rsa,
        analysis.human_metrics.rsa_matrix[np.tril_indices_from(analysis.human_metrics.rsa_matrix, k=-1)],
    )
    if has_kinetics:
        kinetics_human = analysis.kinetics_human_metrics
        if kinetics_human is None:
            raise RuntimeError("Kinetics metrics have not been computed.")
        hh_torque = (
            kinetics_human.avg_torque_rmse["all_moments"],
            kinetics_human.torque_rmse["all_moments"],
        )
        hh_grf = (
            kinetics_human.avg_grf_rmse["all_grf"],
            kinetics_human.grf_rmse["all_grf"],
        )
    else:
        hh_torque = hh_grf = nan_metric

    # Each row is a dict. ``is_ref`` rows are grey benchmarks distinguished by a hatch
    # texture (``hatch``); policy rows carry the population/subject-matched split and are
    # tinted by ``policy_idx``. The top group is the human/RSA benchmarks; each policy's
    # own within-subject variability bound is a grey row beneath that policy.
    rows = []

    # Within-subject reliability: the appropriate subject-matched upper bound.
    wref = analysis.references.within_subject
    if wref and wref.n > 0:
        w_kin = (np.nanmean(wref.joint_rmse["all_joints"]), wref.joint_rmse["all_joints"])
        w_emg = (np.nanmean(wref.muscle_corr["all_muscles"]), wref.muscle_corr["all_muscles"])
        w_rsa = (np.nanmean(wref.rsa_spearman), wref.rsa_spearman)
        rows.append(
            _ref_row(
                WITHIN_SUBJECT_LABEL,
                WITHIN_SUBJECT_HATCH,
                _scoped_reference_metrics(
                    REFERENCE_SCOPES[WITHIN_SUBJECT_LABEL],
                    _layer_metrics(w_kin, w_emg, w_rsa),
                    empty_metric=nan_metric,
                ),
            )
        )
    # Between-subject (pairwise).
    rows.append(
        _ref_row(
            BETWEEN_SUBJECT_LABEL,
            BETWEEN_SUBJECT_HATCH,
            _scoped_reference_metrics(
                REFERENCE_SCOPES[BETWEEN_SUBJECT_LABEL],
                _layer_metrics(hh_kin, hh_emg, hh_rsa, torque=hh_torque, grf=hh_grf),
                empty_metric=nan_metric,
            ),
        )
    )
    # RSA noise ceiling (leave-one-subject-out): applies to the RSA column only.
    nc = analysis.references.rsa_noise_ceiling
    if nc and nc.n > 0:
        nc_rsa = (np.nanmean(nc.rsa_spearman), nc.rsa_spearman)
        rows.append(
            _ref_row(
                RSA_NOISE_CEILING_LABEL,
                RSA_NOISE_CEILING_HATCH,
                _scoped_reference_metrics(
                    REFERENCE_SCOPES[RSA_NOISE_CEILING_LABEL],
                    _layer_metrics(nan_metric, nan_metric, nc_rsa),
                    empty_metric=nan_metric,
                ),
            )
        )

    for policy_idx, run in enumerate(analysis.policies):
        label = run.policy.label

        # Population
        kin_pop = (
            run.population.avg_joint_rmse["all_joints"],
            run.population.joint_rmse["all_joints"],
        )
        emg_pop = (
            run.population.avg_muscle_corr["all_muscles"],
            run.population.muscle_corr["all_muscles"],
        )
        rsa_pop = (run.population.avg_rsa, run.population.rsa_matrix)
        if has_kinetics:
            if run.kinetics_population is None:
                raise RuntimeError(f"Policy {run.policy.id!r} kinetics metrics have not been computed.")
            torque_pop = (
                run.kinetics_population.avg_torque_rmse["all_moments"],
                run.kinetics_population.torque_rmse["all_moments"],
            )
            grf_pop = (
                run.kinetics_population.avg_grf_rmse["all_grf"],
                run.kinetics_population.grf_rmse["all_grf"],
            )
        else:
            torque_pop = grf_pop = nan_metric

        # Per-subject (only when subject-based metrics were computed)
        if analysis.subject_based:
            kin_sub_samples = run.per_subject.joint_rmse["all_joints"]
            emg_sub_samples = run.per_subject.muscle_corr["all_muscles"]
            kin_sub = (_mean(kin_sub_samples), kin_sub_samples)
            emg_sub = (_mean(emg_sub_samples), emg_sub_samples)
            rsa_sub = (
                run.per_subject.avg_rsa,
                run.per_subject.rsa,
            )
            if has_kinetics:
                if run.kinetics_per_subject is None:
                    raise RuntimeError(f"Policy {run.policy.id!r} subject-matched kinetics have not been computed.")
                torque_sub = (
                    run.kinetics_per_subject.avg_torque_rmse["all_moments"],
                    run.kinetics_per_subject.torque_rmse["all_moments"],
                )
                grf_sub = (
                    run.kinetics_per_subject.avg_grf_rmse["all_grf"],
                    run.kinetics_per_subject.grf_rmse["all_grf"],
                )
            else:
                torque_sub = grf_sub = nan_metric
        else:
            kin_sub = emg_sub = rsa_sub = (np.nan, [])
            torque_sub = grf_sub = nan_metric

        policy_metrics = [kin_pop, kin_sub]
        if has_kinetics:
            policy_metrics.extend((torque_pop, torque_sub, grf_pop, grf_sub))
        policy_metrics.extend((emg_pop, emg_sub, rsa_pop, rsa_sub))

        rows.append(
            {
                "label": label,
                "metrics": policy_metrics,
                "is_ref": False,
                "hatch": None,
                "policy_idx": policy_idx,
            }
        )

        # Trial-reference sensitivity requested in the review: the policy still tracks
        # Trial 5, but its per-subject output is evaluated against mean human Trials 1-4.
        # This has no population counterpart, so values occupy only the Subj. columns.
        sensitivity = run.earlier_human_sensitivity
        if sensitivity and sensitivity.n > 0:
            sensitivity_kin = (
                _mean(sensitivity.joint_rmse["all_joints"]),
                sensitivity.joint_rmse["all_joints"],
            )
            sensitivity_emg = (
                _mean(sensitivity.muscle_corr["all_muscles"]),
                sensitivity.muscle_corr["all_muscles"],
            )
            sensitivity_rsa = (
                _mean(sensitivity.rsa_spearman),
                sensitivity.rsa_spearman,
            )
            rows.append(
                {
                    "label": EARLIER_HUMAN_SENSITIVITY_LABEL,
                    "metrics": _scoped_reference_metrics(
                        REFERENCE_SCOPES[EARLIER_HUMAN_SENSITIVITY_LABEL],
                        _layer_metrics(sensitivity_kin, sensitivity_emg, sensitivity_rsa),
                        empty_metric=nan_metric,
                    ),
                    "is_ref": False,
                    "hatch": None,
                    "policy_idx": policy_idx,
                }
            )

        # This policy's own within-subject variability bound (Trial 5 vs its Trials 1-4
        # rollouts), grey + hatched, placed directly beneath the policy it belongs to.
        wsv = run.within_subject_variability
        if wsv and wsv.n > 0:
            v_kin = (np.nanmean(wsv.joint_rmse["all_joints"]), wsv.joint_rmse["all_joints"])
            v_emg = (np.nanmean(wsv.muscle_corr["all_muscles"]), wsv.muscle_corr["all_muscles"])
            v_rsa = (np.nanmean(wsv.rsa_spearman), wsv.rsa_spearman)
            rows.append(
                _ref_row(
                    POLICY_WITHIN_SUBJECT_LABEL,
                    POLICY_WITHIN_SUBJECT_HATCH,
                    _scoped_reference_metrics(
                        REFERENCE_SCOPES[POLICY_WITHIN_SUBJECT_LABEL],
                        _layer_metrics(v_kin, v_emg, v_rsa),
                        empty_metric=nan_metric,
                    ),
                )
            )

    def _fmt(metric):
        mean, samples = metric
        if mean is None or not np.isfinite(mean):
            return "—"
        std = _std(samples)
        # Round to the precision the std justifies: show the std to one significant
        # figure and the mean to the same decimal place, clamped to 1-2 decimals so a
        # cell never carries more digits than the spread supports (and never 3+).
        if np.isfinite(std) and std > 0:
            decimals = int(np.clip(-np.floor(np.log10(std)), 1, 2))
        else:
            decimals = 2
        if np.isfinite(std):
            return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
        return f"{mean:.{decimals}f}"

    n_metric_cols = 2 * len(metric_names)  # pop + per-subject for each metric
    # The metric cells carry "mean ± std", so give them most of the width.
    col_widths = [0.28] + [0.72 / n_metric_cols] * n_metric_cols

    # Two header rows (metric group + analysis scope) followed by the data rows.
    group_row = [""] * (1 + n_metric_cols)  # metric names drawn as spanning labels below
    sub_row = [""] + ["Population", "Subject-matched"] * len(metric_names)
    data_rows = [[row["label"]] + [_fmt(value) for value in row["metrics"]] for row in rows]
    cell_text = [group_row, sub_row, *data_rows]

    # The policy labels are multi-line sentences, so the data rows have to be tall
    # enough for the longest one; the two header rows stay single-line.
    label_lines = max(str(row["label"]).count("\n") + 1 for row in rows)
    header_height_units = 1.6
    row_height_units = 1.4 * label_lines

    # A table cell's height unit is one line of text (fontsize 9, 1.2 line spacing), so the
    # table's height is known up front; size the figure so it fills the default axes box
    # instead of leaving the table stranded in whitespace.
    table_height_in = (9 / 72 * 1.2) * (2 * header_height_units + len(rows) * row_height_units)
    fig_height = table_height_in / 0.77 + 0.5
    fig, ax = plt.subplots(figsize=(2.0 * n_metric_cols + 5, fig_height))
    ax.axis("off")
    ax.set_title(
        "Human references and average human-to-model metrics",
        fontsize=12,
        pad=12,
    )

    table = ax.table(cellText=cell_text, cellLoc="center", loc="upper center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Grow the data rows to fit the tallest label; the header rows keep the default height.
    base_height = max(cell.get_height() for cell in table.get_celld().values())
    for (r, _c), cell in table.get_celld().items():
        cell.set_height(base_height * (header_height_units if r < 2 else row_height_units))

    # Style both header rows (dark background, white text). Blend the group-header
    # row's cell borders into the fill so each metric reads as one seamless band.
    for c in range(1 + n_metric_cols):
        for r in (0, 1):
            hdr = table[(r, c)]
            hdr.set_facecolor("#404040")
            hdr.set_text_props(color="white", fontweight="bold" if r == 0 else "normal")
            if r == 0:
                hdr.set_edgecolor("#404040")

    # Tint each label cell: reference rows are all grey, distinguished by hatch texture;
    # policy rows use their plot color at the same fill alpha as the bar plots (so
    # same-colored policies stay distinguishable: finetuned vs from-scratch).
    for i, row in enumerate(rows):
        r = i + 2  # offset past the two header rows
        cell = table[(r, 0)]
        if row["is_ref"]:
            # Faint grey fill + light-grey hatch: the texture reads as a "reference"
            # marker without the dark lines fighting the (black) label text.
            cell.set_facecolor(to_rgba(REF_GREY, 0.4))
            cell.set_edgecolor("silver")
            cell.set_hatch(row["hatch"])
        else:
            cell.set_facecolor(policy_color_rgba(analysis, row["policy_idx"]))
        cell.set_text_props(ha="left", fontsize=8)

    # Overlay the metric-group names centered across each analysis-scope pair.
    fig.canvas.draw()
    for m, name in enumerate(metric_names):
        left = table[(0, 1 + 2 * m)]
        right = table[(0, 2 + 2 * m)]
        x_center = (left.get_x() + right.get_x() + right.get_width()) / 2
        y_center = left.get_y() + left.get_height() / 2
        ax.text(
            x_center,
            y_center,
            name,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    ax.text(
        0.0,
        min(cell.get_y() for cell in table.get_celld().values()) - 0.04,
        "Population = comparisons without subject-identity matching; Subject-matched = same-subject or "
        "subject-specific comparisons. "
        "Reference values appear only in the scope where their comparison is defined; — = not applicable.",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        color="0.3",
    )

    fig.savefig(f"{analysis.plots_dir}/metrics_summary_table.svg", bbox_inches="tight")
    fig.savefig(f"{analysis.plots_dir}/metrics_summary_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics summary table to {analysis.plots_dir}/metrics_summary_table.svg")
