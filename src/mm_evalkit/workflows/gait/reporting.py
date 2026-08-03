"""Human-readable console reporting for full-body metrics."""

from __future__ import annotations

from .numerics import finite_mean
from .state import GaitAnalysis


def _print_human_to_human(analysis: GaitAnalysis) -> None:
    print("\n ### --------------------Human-to-Human Metrics-------------------- ### ")
    metrics = analysis.human_metrics
    print(f"  Average Pairwise Human-to-Human Joint RMSE: {metrics.avg_pairwise_joint_rmse['all_joints']:.4f}")
    print(f"  Average Pairwise Human-to-Human Joint Correlation: {metrics.avg_pairwise_joint_corr['all_joints']:.4f}")
    print(
        f"  Average Pairwise Human-to-Human Muscle Correlation: {metrics.avg_pairwise_muscle_corr['all_muscles']:.4f}"
    )
    print(
        f"  Average Pairwise Human-to-Human Muscle RSA (Spearman): {metrics.avg_pairwise_muscle_rsa:.4f}"
        f"  [Pearson sensitivity: {metrics.avg_pairwise_muscle_rsa_pearson:.4f}]"
    )


def _print_reference_benchmarks(analysis: GaitAnalysis) -> None:
    print("\n ### --------------------Human Reference Benchmarks-------------------- ### ")
    print("  (1) Between-subject (pairwise) [above] -- between-subject variability, not a subject-matched bound.")
    within = analysis.references.within_subject
    if within and within.n > 0:
        print(f"  (2) Within-subject reliability -- Trial 5 vs mean(Trials 1-4), n={within.n} subjects:")
        print(f"        Joint RMSE:        {finite_mean(within.joint_rmse['all_joints']):.4f}")
        print(f"        Joint Correlation: {finite_mean(within.joint_corr['all_joints']):.4f}")
        print(f"        Muscle Correlation:{finite_mean(within.muscle_corr['all_muscles']):.4f}")
        print(
            f"        Muscle RSA (Spearman): {finite_mean(within.rsa_spearman):.4f}"
            f"  [Pearson sensitivity: {finite_mean(within.rsa_pearson):.4f}]"
        )
    else:
        print("  (2) Within-subject reliability -- unavailable (no earlier-trial data at this gait).")

    policy_references = analysis.references.policy_within_subject
    if policy_references:
        print("  (2b) Policy within-subject reliability -- Trial 5 vs each of Trials 1-4 rollouts (per policy):")
        for reference in policy_references:
            one_line_label = reference.label.replace("\n", " ")
            print(f"        [{one_line_label}] n={reference.n} subjects:")
            print(f"          Joint RMSE:        {finite_mean(reference.joint_rmse['all_joints']):.4f}")
            print(f"          Muscle Correlation:{finite_mean(reference.muscle_corr['all_muscles']):.4f}")
            print(
                f"          Muscle RSA (Spearman): {finite_mean(reference.rsa_spearman):.4f}"
                f"  [Pearson sensitivity: {finite_mean(reference.rsa_pearson):.4f}]"
            )
    else:
        print("  (2b) Policy within-subject reliability -- unavailable (no trials-1-4 rollouts at this gait).")

    ceiling = analysis.references.rsa_noise_ceiling
    if ceiling and ceiling.n > 0:
        print(f"  (3) RSA noise ceiling -- each subject RSM vs leave-one-out group RSM, n={ceiling.n}:")
        print(
            f"        Muscle RSA (Spearman): {finite_mean(ceiling.rsa_spearman):.4f}"
            f"  [Pearson sensitivity: {finite_mean(ceiling.rsa_pearson):.4f}]"
        )
    else:
        print("  (3) RSA noise ceiling -- unavailable.")


def _print_population(run) -> None:
    print("\n ### --------------------Population Metrics-------------------- ### ")
    population = run.population
    print(f"    Average Pairwise Human-to-Model Joint RMSE: {population.avg_joint_rmse['all_joints']:.4f}")
    print(f"    Average Pairwise Human-to-Model Joint Correlation: {population.avg_joint_corr['all_joints']:.4f}")
    print(f"    Average Pairwise Human-to-Model Muscle Correlation: {population.avg_muscle_corr['all_muscles']:.4f}")
    print(
        f"    Average Pairwise Human-to-Model Muscle RSA (Spearman): {population.avg_rsa:.4f}"
        f"  [Pearson sensitivity: {population.avg_rsa_pearson:.4f}]"
    )


def _print_per_subject(run, n_cohort_subjects: int) -> None:
    print("\n ### --------------------Subject-based Metrics-------------------- ### ")
    per_subject = run.per_subject
    if per_subject.joint_rmse["all_joints"]:
        print(
            "    Average Pairwise Human-to-Model Joint RMSE: ",
            f"{finite_mean(per_subject.joint_rmse['all_joints']):.4f}",
        )
        print(
            "    Average Pairwise Human-to-Model Joint Correlation: ",
            f"{finite_mean(per_subject.joint_corr['all_joints']):.4f}",
        )
        print(
            "    Average Pairwise Human-to-Model Muscle Correlation: ",
            f"{finite_mean(per_subject.muscle_corr['all_muscles']):.4f}",
        )
        print(
            "    Average Pairwise Human-to-Model Muscle RSA (Spearman): ",
            f"{per_subject.avg_rsa:.4f}",
            f" [Pearson sensitivity: {per_subject.avg_rsa_pearson:.4f}]",
        )
    else:
        print("    No subjects with successful gait cycles - metrics unavailable")

    sensitivity = run.earlier_human_sensitivity
    if sensitivity:
        print("\n ### ----------Sensitivity: Policy Trial 5 vs mean(Human Trials 1-4)---------- ### ")
        print(f"    Coverage: {sensitivity.n}/{n_cohort_subjects} subjects")
        print("    Average subject-matched Joint RMSE: ", f"{finite_mean(sensitivity.joint_rmse['all_joints']):.4f}")
        print(
            "    Average subject-matched Joint Correlation: ",
            f"{finite_mean(sensitivity.joint_corr['all_joints']):.4f}",
        )
        print(
            "    Average subject-matched Muscle Correlation: ",
            f"{finite_mean(sensitivity.muscle_corr['all_muscles']):.4f}",
        )
        print(
            "    Average subject-matched Muscle RSA (Spearman): ",
            f"{finite_mean(sensitivity.rsa_spearman):.4f}",
            f" [Pearson sensitivity: {finite_mean(sensitivity.rsa_pearson):.4f}]",
        )

    if per_subject.subjects_without_gait_cycles:
        print(
            "    Subjects with no retained gait cycles (excluded from averages): "
            f"{per_subject.subjects_without_gait_cycles}"
        )


def _print_skipped_episodes(run) -> None:
    if not run.skipped_episodes:
        return
    skipped_counts: dict[str, int] = {}
    for skipped in run.skipped_episodes:
        skipped_counts[skipped["reason"]] = skipped_counts.get(skipped["reason"], 0) + 1
    skipped_summary = ", ".join(f"{reason}: {count}" for reason, count in sorted(skipped_counts.items()))
    print(f"    Skipped episodes by reason: {skipped_summary}")
    for reason in sorted(skipped_counts):
        traj_ids = [item["traj_id"] for item in run.skipped_episodes if item["reason"] == reason]
        preview = traj_ids[:20]
        suffix = " ..." if len(traj_ids) > len(preview) else ""
        print(f"      {reason} trajectory IDs: {preview}{suffix}")


def _print_kinetics(analysis: GaitAnalysis) -> None:
    if analysis.kinetics is None:
        return
    human = analysis.kinetics_human_metrics
    if human is None:
        raise RuntimeError("Kinetics metrics have not been computed.")
    print("\n ### --------------------Kinetics Metrics-------------------- ### ")
    print("  Units: joint moment RMSE [Nm/kg]; vertical GRF RMSE [BW].")
    print(
        "  Pairwise human-human: "
        f"moment RMSE={human.avg_torque_rmse['all_moments']:.4f}, "
        f"moment r={human.avg_torque_corr['all_moments']:.4f}, "
        f"GRF RMSE={human.avg_grf_rmse['all_grf']:.4f}, "
        f"GRF r={human.avg_grf_corr['all_grf']:.4f}"
    )
    for run in analysis.policies:
        population = run.kinetics_population
        subject = run.kinetics_per_subject
        if population is None or subject is None:
            raise RuntimeError(f"Policy {run.policy.id!r} kinetics metrics have not been computed.")
        print(f"  [{run.label}] population:")
        print(
            f"    moment RMSE={population.avg_torque_rmse['all_moments']:.4f}, "
            f"moment r={population.avg_torque_corr['all_moments']:.4f}, "
            f"GRF RMSE={population.avg_grf_rmse['all_grf']:.4f}, "
            f"GRF r={population.avg_grf_corr['all_grf']:.4f}"
        )
        print(f"  [{run.label}] subject-matched:")
        print(
            f"    moment RMSE={subject.avg_torque_rmse['all_moments']:.4f}, "
            f"moment r={subject.avg_torque_corr['all_moments']:.4f}, "
            f"GRF RMSE={subject.avg_grf_rmse['all_grf']:.4f}, "
            f"GRF r={subject.avg_grf_corr['all_grf']:.4f}"
        )
        if run.kinetics_omissions:
            print(f"    excluded unavailable/incomplete kinetics inputs: {len(run.kinetics_omissions)}")


def print_metrics(analysis: GaitAnalysis) -> None:
    """Print population, subject-matched, and reference metrics for one gait."""
    _print_human_to_human(analysis)
    _print_reference_benchmarks(analysis)

    print("\n ### --------------------Human-to-Model Metrics-------------------- ### ")
    for run in analysis.policies:
        print(f"\n ### --------------------Policy: {run.label}-------------------- ###")
        _print_population(run)
        if analysis.subject_based:
            _print_per_subject(run, len(analysis.human.subjects))
            _print_skipped_episodes(run)
    _print_kinetics(analysis)
