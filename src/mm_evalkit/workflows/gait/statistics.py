"""Paired statistical tests for full-body metrics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from mm_evalkit.stats.report import print_wilcoxon_summary, save_wilcoxon_report
from mm_evalkit.stats.wilcoxon import MetricFamily, MetricSamples, wilcoxon_table

from .constants import (
    EARLIER_HUMAN_SENSITIVITY_LABEL,
    POLICY_WITHIN_SUBJECT_LABEL,
    WITHIN_SUBJECT_LABEL,
)
from .human_data import gait_selection_label
from .state import GaitAnalysis, WilcoxonResults


def compute_wilcoxon_tests(analysis: GaitAnalysis):
    """Test whether the differences between the reported metrics are significant.

    Every metric here is a matrix of paired measurements -- one value per (subject,
    joint) or (subject, muscle) -- and every test pairs over the subjects, in two
    readings of that matrix (see :mod:`utils.stats`):

    * **across subjects**: average over the joints/muscles first, then pair the two
      conditions subject by subject;
    * **per joint / per muscle**: run that same paired-over-subjects test on each
      joint/muscle separately, Holm-corrected across that comparison's channels, and
      report how many channels carry the difference and which ones.

    The conditions that can be paired at all are the ones indexed by subject: each
    policy's population metrics (its group profile scored against every human
    subject), its subject-matched metrics, its trials-1-4 sensitivity variant, its
    own within-subject variability, and the human within-subject reliability
    reference. The pairwise human-human (between-subject) reference is indexed by
    *subject pairs*, not subjects, so it has no paired counterpart and is left out.

    Comparisons are named explicitly rather than run over every pair of conditions,
    so the report stays the contrasts that mean something: policy against policy,
    subject-matched against population, and each policy against the reliability
    bounds. Results go to ``wilcoxon_tests.csv`` plus one table figure per metric in
    the current (per-gait) plots directory, and are returned as a DataFrame.
    """
    n_human_subjects = analysis.human.n_subjects
    population_units = (
        [int(s) for s in analysis.human.subjects] if analysis.subject_based else list(range(n_human_subjects))
    )
    joints = list(analysis.channels.joint_names)
    muscles = list(analysis.channels.muscle_names)
    torque_channels = list(analysis.kinetics.torque_names) if analysis.kinetics is not None else []
    grf_channels = list(analysis.kinetics.grf_names) if analysis.kinetics is not None else []
    rsa_channel = "muscle RSM"

    samples = {
        "joint_rmse": {},
        "joint_corr": {},
        "torque_rmse": {},
        "torque_corr": {},
        "grf_rmse": {},
        "grf_corr": {},
        "muscle_corr": {},
        "rsa": {},
    }
    legend = {}

    def add(
        name,
        description,
        subjects,
        joint_rmse=None,
        joint_corr=None,
        torque_rmse=None,
        torque_corr=None,
        grf_rmse=None,
        grf_corr=None,
        muscle_corr=None,
        rsa=None,
    ):
        """Register one condition; a metric it does not carry is simply absent."""
        subjects = [int(s) for s in subjects]
        legend[name] = description
        if joint_rmse is not None:
            samples["joint_rmse"][name] = MetricSamples.from_per_channel(joint_rmse, subjects, joints)
        if joint_corr is not None:
            samples["joint_corr"][name] = MetricSamples.from_per_channel(joint_corr, subjects, joints)
        if torque_rmse is not None:
            samples["torque_rmse"][name] = MetricSamples.from_per_channel(torque_rmse, subjects, torque_channels)
        if torque_corr is not None:
            samples["torque_corr"][name] = MetricSamples.from_per_channel(torque_corr, subjects, torque_channels)
        if grf_rmse is not None:
            samples["grf_rmse"][name] = MetricSamples.from_per_channel(grf_rmse, subjects, grf_channels)
        if grf_corr is not None:
            samples["grf_corr"][name] = MetricSamples.from_per_channel(grf_corr, subjects, grf_channels)
        if muscle_corr is not None:
            samples["muscle_corr"][name] = MetricSamples.from_per_channel(muscle_corr, subjects, muscles)
        if rsa is not None:
            samples["rsa"][name] = MetricSamples.from_vector(rsa, subjects, rsa_channel)

    # --- Human within-subject reliability (Trial 5 vs Trials 1-4), per subject ---
    wref = analysis.references.within_subject
    human_reference = None
    if wref and wref.subjects:
        human_reference = "Human WSR"
        add(
            human_reference,
            f"{WITHIN_SUBJECT_LABEL} — the human upper bound",
            wref.subjects,
            joint_rmse=wref.joint_rmse,
            joint_corr=wref.joint_corr,
            muscle_corr=wref.muscle_corr,
            rsa=wref.rsa_spearman,
        )

    # --- One block of conditions per policy ---
    population_names, subject_names = [], []
    vs_population, vs_human_reference, vs_sensitivity, vs_variability = [], [], [], []
    for idx, run in enumerate(analysis.policies):
        tag = f"P{idx + 1}"
        label = " ".join(run.policy.label.split())
        if np.all(np.isnan(run.profiles.avg_joint_gaits)):
            # No retained gait cycle at this position: its metrics were never
            # computed, so it has nothing to pair against.
            print(f"  Wilcoxon: skipping {tag} ({label}) — no data at this gait position.")
            continue

        population = f"{tag} pop."
        kinetics_population = run.kinetics_population
        add(
            population,
            f"{label} — population profile vs each human subject",
            population_units,
            joint_rmse=run.population.joint_rmse,
            joint_corr=run.population.joint_corr,
            torque_rmse=kinetics_population.torque_rmse if kinetics_population is not None else None,
            torque_corr=kinetics_population.torque_corr if kinetics_population is not None else None,
            grf_rmse=kinetics_population.grf_rmse if kinetics_population is not None else None,
            grf_corr=kinetics_population.grf_corr if kinetics_population is not None else None,
            muscle_corr=run.population.muscle_corr,
            rsa=run.population.rsa_matrix[0],
        )
        population_names.append(population)

        if not analysis.subject_based:
            continue

        subject = f"{tag} subj."
        subjects = run.per_subject.subjects_with_rsa_data or []
        kinetics_subject = run.kinetics_per_subject
        add(
            subject,
            f"{label} — subject-matched (each subject's own profile)",
            subjects,
            joint_rmse=run.per_subject.joint_rmse,
            joint_corr=run.per_subject.joint_corr,
            torque_rmse=kinetics_subject.torque_rmse if kinetics_subject is not None else None,
            torque_corr=kinetics_subject.torque_corr if kinetics_subject is not None else None,
            grf_rmse=kinetics_subject.grf_rmse if kinetics_subject is not None else None,
            grf_corr=kinetics_subject.grf_corr if kinetics_subject is not None else None,
            muscle_corr=run.per_subject.muscle_corr,
            rsa=run.per_subject.rsa,
        )
        subject_names.append(subject)
        vs_population.append(("Subject-matched vs population", subject, population))
        if human_reference:
            vs_human_reference.append(("Policy vs human within-subject reliability", subject, human_reference))

        sensitivity = run.earlier_human_sensitivity
        if sensitivity and sensitivity.n > 0:
            name = f"{tag} vs T1-4"
            add(
                name,
                f"{label} — {' '.join(EARLIER_HUMAN_SENSITIVITY_LABEL.split())}",
                sensitivity.subjects,
                joint_rmse=sensitivity.joint_rmse,
                joint_corr=sensitivity.joint_corr,
                muscle_corr=sensitivity.muscle_corr,
                rsa=sensitivity.rsa_spearman,
            )
            vs_sensitivity.append(("Human Trial 5 vs Trials 1-4 as reference", subject, name))

        variability = run.within_subject_variability
        if variability and variability.subjects:
            name = f"{tag} WSV"
            add(
                name,
                f"{label} — {' '.join(POLICY_WITHIN_SUBJECT_LABEL.split())}",
                variability.subjects,
                joint_rmse=variability.joint_rmse,
                muscle_corr=variability.muscle_corr,
                rsa=variability.rsa_spearman,
            )
            vs_variability.append(("Policy vs its own within-subject variability", subject, name))

    comparisons = (
        [("Policy vs policy (subject-matched)", a, b) for a, b in combinations(subject_names, 2)]
        + [("Policy vs policy (population)", a, b) for a, b in combinations(population_names, 2)]
        + vs_population
        + vs_human_reference
        + vs_sensitivity
        + vs_variability
    )
    if not comparisons:
        print("\n  Wilcoxon signed-rank tests: nothing to compare (a single condition with data).")
        analysis.wilcoxon = WilcoxonResults(comparisons=pd.DataFrame(), per_channel=pd.DataFrame())
        return analysis.wilcoxon.comparisons

    families = [
        MetricFamily("Joint angle error [deg]", "joint", samples["joint_rmse"], comparisons),
        MetricFamily("Joint angle correlation", "joint", samples["joint_corr"], comparisons),
    ]
    if analysis.kinetics is not None:
        families.extend(
            (
                MetricFamily("Joint moment error [Nm/kg]", "joint moment", samples["torque_rmse"], comparisons),
                MetricFamily("Joint moment correlation", "joint moment", samples["torque_corr"], comparisons),
                MetricFamily("Vertical GRF error [BW]", "GRF side", samples["grf_rmse"], comparisons),
                MetricFamily("Vertical GRF correlation", "GRF side", samples["grf_corr"], comparisons),
            )
        )
    families.extend(
        (
            MetricFamily("EMG-muscle correlation", "muscle", samples["muscle_corr"], comparisons),
            MetricFamily("EMG-muscle RSA (Spearman)", "muscle", samples["rsa"], comparisons),
        )
    )
    table, channel_table = wilcoxon_table(families, unit_name="subject")
    print_wilcoxon_summary(table, channel_table, unit_name="subject")
    subtitle = (
        f"{analysis.dataset_name}, {gait_selection_label(analysis.gait_position)}: "
        f"{len(population_units)} subject(s), {len(joints)} angle joints, "
        f"{len(torque_channels)} moment joints, {len(grf_channels)} GRF sides, {len(muscles)} muscles."
    )
    save_wilcoxon_report(table, channel_table, str(analysis.plots_dir), legend=legend, subtitle=subtitle)
    analysis.wilcoxon = WilcoxonResults(comparisons=table, per_channel=channel_table)
    return table
