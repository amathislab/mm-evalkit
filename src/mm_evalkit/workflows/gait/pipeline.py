"""Public full-body analysis orchestration."""

from __future__ import annotations

from pathlib import Path

from . import human_data, kinetics_metrics, metrics, references, reporting, statistics
from .models import NoSubjectsForGaitError
from .plotting import kinematics, kinetics, muscles, rsa, summary
from .state import GaitAnalysis


def generate_all_plots(analysis: GaitAnalysis):
    """Render every figure for the metrics currently stored on ``analysis``."""
    kinematics.plot_kinematic_metrics(analysis)
    kinematics.plot_subject_based_kinematic_metrics(analysis)
    muscles.plot_muscle_metrics(analysis)
    muscles.plot_subject_based_muscle_metrics(analysis)
    rsa.plot_rsa_results(analysis)
    rsa.plot_subject_based_rsa_results(analysis)
    kinetics.plot_kinetics_profiles(analysis)
    kinetics.plot_grf_profiles(analysis)
    kinetics.plot_subject_based_kinetics_profiles(analysis)
    kinetics.plot_subject_based_grf_profiles(analysis)
    kinetics.plot_kinetics_metrics(analysis)
    summary.plot_metrics_table(analysis)
    summary.plot_metrics_bars(analysis)


def _compute_and_report(analysis: GaitAnalysis):
    """Run the metric, reference, statistical, and plotting stages once."""
    metrics.compute_human_to_human_metrics(analysis)
    references.compute_human_reference_benchmarks(analysis)
    metrics.compute_human_to_model_metrics_population(analysis)
    metrics.compute_human_to_model_metrics_per_subject(analysis)
    kinetics_metrics.compute_kinetics_metrics(analysis)
    references.compute_policy_to_earlier_human_sensitivity(analysis)
    references.compute_policy_within_subject_variability(analysis)
    reporting.print_metrics(analysis)
    kinetics_metrics.write_kinetics_metrics(analysis)
    statistics.compute_wilcoxon_tests(analysis)
    generate_all_plots(analysis)


def run_gait_position_analyses(analysis: GaitAnalysis, positions=(0, 1, None)):
    """Run the full metric + plot pipeline once per gait selection.

    The first and second gait cycles of an episode may be qualitatively different
    (e.g. the second only exists for long episodes), so each is analysed on its own
    subject subset and written to its own ``gait{N}`` subfolder of the plots
    directory. The trailing ``None`` adds the pooled analysis -- every gait cycle of
    every episode against the mean of the subject's steps -- under ``all_gaits/``,
    which is the "no gait 1 / gait 2 distinction" view and the one to quote when the
    split is not the point. All human-to-human, human-to-model (population and
    per-subject) and RSA metrics/plots are regenerated for each selection.
    Subject-based only.
    """
    if not analysis.subject_based:
        # Non-subject-based analyses are not split by gait position.
        _compute_and_report(analysis)
        return

    base_dir = analysis.plots_dir
    for pos in positions:
        name = human_data.gait_selection_name(pos)
        selection = "every gait position pooled" if pos is None else f"gait position {pos}"
        print(f"\n ########## ---------- Analysis for {name} ({selection}) ---------- ########## ")
        try:
            human_data.select_gait_subset(analysis, pos)
        except NoSubjectsForGaitError as exc:
            print(f"  Skipping {name}: {exc}.")
            continue
        print(f"  {name}: {len(analysis.human.subjects)} subject(s) with this gait: {analysis.human.subjects}")

        analysis.plots_dir = Path(base_dir) / name
        analysis.plots_dir.mkdir(parents=True, exist_ok=True)

        _compute_and_report(analysis)

    analysis.plots_dir = base_dir
