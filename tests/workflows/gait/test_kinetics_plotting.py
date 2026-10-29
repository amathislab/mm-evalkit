"""Layout contracts for the Gait120 kinetics profile figure."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mm_evalkit.datasets.gait120_kinetics import EXTENDED_TORQUE_JOINTS, GRF_CHANNELS, TORQUE_JOINTS
from mm_evalkit.workflows.gait.plotting import kinetics as kinetics_plotting
from mm_evalkit.workflows.gait.state import KineticsCohort, KineticsCycles, KineticsProfiles

from .conftest import make_analysis, make_run


def _profiles(n_torque: int) -> KineticsProfiles:
    torque = np.zeros((n_torque, 100))
    grf = np.zeros((len(GRF_CHANNELS), 100))
    return KineticsProfiles(
        avg_torque=torque,
        std_torque=torque,
        avg_torque_ref=torque,
        std_torque_ref=torque,
        avg_grf=grf,
        std_grf=grf,
        avg_grf_ref=grf,
        std_grf_ref=grf,
        subject_ids=np.empty(0, dtype=int),
        per_subject_torque={},
        per_subject_torque_ref={},
        per_subject_grf={},
        per_subject_grf_ref={},
    )


def _cohort(torque_names: tuple[str, ...]) -> KineticsCohort:
    empty = np.empty((0, 0, 100))
    return KineticsCohort(
        torque_names=torque_names,
        grf_names=GRF_CHANNELS,
        torque_unit="Nm/kg",
        grf_unit="BW",
        cycles=KineticsCycles(empty, empty, empty, empty, np.empty(0, int), np.empty(0, int), np.empty(0, int)),
        reference_paths=(),
        subjects=[51, 52],
        torque_data=np.zeros((2, len(torque_names), 100)),
        grf_data=np.zeros((2, len(GRF_CHANNELS), 100)),
    )


@pytest.mark.parametrize(
    ("torque_names", "shape"),
    [(TORQUE_JOINTS, (2, 4)), (EXTENDED_TORQUE_JOINTS, (4, 4))],
)
def test_kinetics_profiles_draw_one_visible_panel_per_channel(tmp_path, monkeypatch, torque_names, shape):
    run = make_run("policy")
    run.kinetics_profiles = _profiles(len(torque_names))
    analysis = make_analysis(policies=[run], plots_dir=tmp_path)
    analysis.kinetics = _cohort(torque_names)
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig: (figures.append(fig), close(fig)))

    kinetics_plotting.plot_kinetics_profiles(analysis)

    (fig,) = figures
    visible = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert visible == [*torque_names, *GRF_CHANNELS]
    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == shape
    assert (tmp_path / "kinetics_profiles.svg").is_file()
