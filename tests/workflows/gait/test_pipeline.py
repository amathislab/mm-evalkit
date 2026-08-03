"""One synthetic run through every stage, including the figures.

The per-stage tests pin numerical behaviour but never reach the plotting
modules, which read the widest slice of the analysis state. This drives the
whole pipeline once so a record a figure consumes cannot silently go missing.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from mm_evalkit.workflows.gait import pipeline
from mm_evalkit.workflows.gait.human_data import choose_plot_subjects, refresh_human_rsms
from mm_evalkit.workflows.gait.kinetics_data import select_kinetics_subset
from mm_evalkit.workflows.gait.metrics import compute_rsm
from mm_evalkit.workflows.gait.state import KineticsCohort, KineticsCycles, PolicyProfiles

from .conftest import JOINTS, MUSCLES, make_analysis, make_cohort, make_run

SUBJECTS = [51, 52, 53, 54, 55, 56]


def _profiles(rng, offset):
    """Per-subject and population profiles for one policy at one gait position."""
    per_joint = {subject: rng.normal(offset, 1.0, (len(JOINTS), 100)) for subject in SUBJECTS}
    per_muscle = {subject: rng.normal(0.5, 0.1, (len(MUSCLES), 100)) for subject in SUBJECTS}
    average_muscle = np.mean(list(per_muscle.values()), axis=0)
    return PolicyProfiles(
        avg_joint_gaits=np.mean(list(per_joint.values()), axis=0),
        std_joint_gaits=np.ones((len(JOINTS), 100)),
        avg_joint_gaits_ref=np.mean(list(per_joint.values()), axis=0),
        std_joint_gaits_ref=np.ones((len(JOINTS), 100)),
        avg_muscle_gaits=average_muscle,
        std_muscle_gaits=np.ones((len(MUSCLES), 100)),
        population_muscle_rsm=compute_rsm(average_muscle),
        subject_ids=np.array(SUBJECTS),
        per_subject_joint=per_joint,
        per_subject_muscle=per_muscle,
        per_subject_joint_ref=per_joint,
        per_subject_muscle_rsm={subject: compute_rsm(per_muscle[subject]) for subject in SUBJECTS},
    )


def _analysis(tmp_path):
    rng = np.random.default_rng(0)
    cohort = make_cohort(SUBJECTS)
    cohort.joint_data = rng.normal(0.0, 1.0, (len(SUBJECTS), len(JOINTS), 100))
    cohort.muscle_data = rng.normal(0.5, 0.1, (len(SUBJECTS), len(MUSCLES), 100))
    refresh_human_rsms(cohort)
    choose_plot_subjects(cohort, 5)

    runs = []
    for index, offset in enumerate((0.0, 2.0)):
        run = make_run(f"policy {index}")
        run.profiles = _profiles(rng, offset)
        runs.append(run)

    analysis = make_analysis(policies=runs, cohort=cohort, plots_dir=tmp_path)
    analysis.gait_position = 0
    analysis.muscle_map = {muscle: muscle.upper() for muscle in MUSCLES}
    return analysis


def test_every_stage_runs_and_writes_its_artifacts(tmp_path):
    analysis = _analysis(tmp_path)

    pipeline._compute_and_report(analysis)

    # Every stage left its record on the state...
    assert analysis.human_metrics is not None
    assert analysis.references.within_subject is None  # no earlier-trial human data here
    assert all(run.population is not None and run.per_subject is not None for run in analysis.policies)
    assert analysis.wilcoxon is not None

    # ...and each family of figures reached disk.
    written = {path.name for path in tmp_path.iterdir()}
    assert "metrics_summary_table.png" in written
    assert "wilcoxon_tests.csv" in written
    assert {name for name in written if name.startswith("wilcoxon_") and name.endswith(".png")}
    assert {name for name in written if name.endswith(".png")} - {"metrics_summary_table.png"}


def test_a_population_only_analysis_skips_the_subject_split(tmp_path):
    """Wang has no subject identity, so the run is not split by gait position."""
    from .conftest import make_config

    analysis = _analysis(tmp_path)
    analysis.config = make_config(dataset_name="Wang", subject_based=False)

    pipeline.run_gait_position_analyses(analysis)

    assert analysis.plots_dir == tmp_path  # no gait1/ gait2/ subfolders
    assert all(run.per_subject is None for run in analysis.policies)


def _kinetics_cycles(reference, offset):
    return KineticsCycles(
        torque=reference[:, :6] + offset,
        torque_ref=reference[:, :6],
        grf=reference[:, :2] + offset / 10,
        grf_ref=reference[:, :2],
        subject_ids=np.asarray(SUBJECTS),
        trial_ids=np.full(len(SUBJECTS), 5),
        gait_positions=np.zeros(len(SUBJECTS), dtype=int),
    )


def test_pipeline_adds_kinetics_csv_figures_summary_and_statistics(tmp_path):
    analysis = _analysis(tmp_path)
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(len(SUBJECTS), 6, 100))
    human_cycles = _kinetics_cycles(reference, 0.0)
    analysis.kinetics = KineticsCohort(
        torque_names=(
            "hip_flexion_r",
            "hip_flexion_l",
            "knee_angle_r",
            "knee_angle_l",
            "ankle_angle_r",
            "ankle_angle_l",
        ),
        grf_names=("left", "right"),
        torque_unit="Nm/kg",
        grf_unit="BW",
        cycles=human_cycles,
        reference_paths=(),
    )
    for index, run in enumerate(analysis.policies, start=1):
        run.kinetics_cycles = _kinetics_cycles(reference, float(index))
    select_kinetics_subset(analysis.policies, analysis.kinetics, SUBJECTS, 0)

    pipeline._compute_and_report(analysis)

    assert analysis.kinetics_human_metrics is not None
    assert all(
        run.kinetics_population is not None and run.kinetics_per_subject is not None for run in analysis.policies
    )
    assert (tmp_path / "kinetics_metrics.csv").is_file()
    assert (tmp_path / "kinetics_profiles.svg").is_file()
    assert (tmp_path / "kinetics_metrics.svg").is_file()
    assert "Joint moment error [Nm/kg]" in set(analysis.wilcoxon.comparisons["metric"])
