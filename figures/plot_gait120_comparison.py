"""Gait120 comparison figure: right-limb kinematics, kinetics and muscle activity.

One figure, eleven waveform panels and seven metric panels:

* row 1 -- hip, knee and ankle joint angles;
* row 2 -- the matching joint moments and the vertical ground reaction force;
* row 3 -- muscle excitation (or activation) against normalized surface EMG;
* right block -- subject-matched Pearson r and RMSE for each layer.
  ``--metric_scope population`` scores each policy's population profile
  against every subject instead (writes ``*_population.*``).

Every panel is the right limb, because Gait120 records surface EMG on the right
side only. Human data is drawn as one grey trace per retained subject rather
than a mean with a spread band, and policy profiles are drawn without a band, so
the figure shows the real between-subject spread instead of a symmetric summary
of it.

Colours and line styles come from the policy YAML, so a run is restyled by
editing that file rather than this script. See ``policies.example.yaml``.

Phase reference differs between layers: joint-angle and EMG cycles run right
heel strike to right heel strike, while moments and vertical GRF use the
Gait120 references' fixed 1.00 s window from *left* heel strike, which sits
about half a cycle out of phase; column B's axis is labelled in seconds. Every
policy rollout covers that window completely. ``--kinetics_window
right_stride`` uses the force-plate right stride instead, sharing the phase of
the other columns, but rollouts end a few frames before their reference trial,
so the last ~7 % of each policy kinetics cycle is then undefined.

Usage::

    python figures/plot_gait120_comparison.py \\
        --dataset_dir /path/to/Gait120 \\
        --kinetics_reference_dir /path/to/gait120-kinetics \\
        --policy_config_file figures/policies.example.yaml \\
        --output_dir results/figure

The analysis stage takes several minutes, so its result is cached next to the
output. Re-run with ``--refresh`` after changing the policies or the data.
Use ``--sequence_dir`` for the wide layout: three physiological columns under
phase-aligned lateral gait poses from ``render_gait120_sequence.py``, and a
fourth column of agreement metrics.

Muscle traces are policy excitation by default: ``clip(muscle_commands, 0, 1)``,
the control MuJoCo integrates into activation. ``--muscle_signal activation``
restores the exported ``muscle_activations``. Pearson r against EMG uses the
same signal as the traces.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")


def _ensure_mm_evalkit_importable() -> None:
    """Import mm-evalkit whether it is installed or only checked out.

    Running this script straight from a clone is the common case, so a source
    tree beside it is added to the path rather than demanding an install first.
    """
    if importlib.util.find_spec("mm_evalkit") is not None:
        return
    source = Path(__file__).resolve().parent.parent / "src"
    if not (source / "mm_evalkit").is_dir():
        raise SystemExit(
            "mm-evalkit is not importable.\n"
            "Install it with `python -m pip install mm-evalkit`, or run this script\n"
            "from a checkout of the repository so that ../src/mm_evalkit is found."
        )
    sys.path.insert(0, str(source))


_ensure_mm_evalkit_importable()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from mm_evalkit.policies import load_policy_specs  # noqa: E402
from mm_evalkit.workflows.gait import human_data, kinetics_metrics, metrics, references  # noqa: E402
from mm_evalkit.workflows.gait.analysis import FullBodyAnalysis  # noqa: E402
from mm_evalkit.workflows.gait.numerics import finite_mean, paired_pearson, paired_rmse  # noqa: E402

STEM = "gait120_comparison"

HUMAN_TRACE = "#a9a8a3"  # one line per retained subject, behind the policies
HUMAN_BAR = "#8a8984"
GRID = "#d9d8d3"
INK = "#1c1c1a"
INK_MUTED = "#6b6a65"
SURFACE = "#ffffff"

AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 13.5
TITLE_SIZE = 16
LEGEND_SIZE = 14

PHASE = np.linspace(0.0, 100.0, 100)
XLABEL = "Gait cycle [%]"

# A dashed policy is drawn with a tighter dash than matplotlib's default, which
# is too sparse to read at this line width in small panels.
DASH_PATTERN = (0, (5, 1.6))
SOLID_STYLES = {"-", "solid", ""}


def _line_style(spec: str):
    """Map a policy YAML line style onto what the waveform panels draw."""
    return spec if spec in SOLID_STYLES else DASH_PATTERN


def _is_dashed(spec: str) -> bool:
    return spec not in SOLID_STYLES


# (channel, angle panel title, moment panel title, short bar label)
JOINTS = [
    ("hip_flexion_r", "Hip flexion", "Hip flexion moment", "Hip"),
    ("knee_angle_r", "Knee angle", "Knee moment", "Knee"),
    ("ankle_angle_r", "Ankle angle", "Ankle moment", "Ankle"),
]
GRF_CHANNEL = "right"
# (MyoFullBody muscle, panel title, short bar label)
MUSCLES = [
    ("vaslat_r", "Vastus lateralis", "VL"),
    ("bflh_r", "Biceps femoris", "BF"),
    ("tibant_r", "Tibialis anterior", "TA"),
    ("gasmed_r", "Gastrocnemius medialis", "GM"),
]

# The extended figure: every right-leg joint angle and every Gait120 EMG muscle,
# grouped quadriceps, hamstrings, anterior shank, posterior shank, peroneals.
EXTENDED_JOINTS = [
    ("hip_flexion_r", "Hip flexion", "Hip\nflex."),
    ("hip_adduction_r", "Hip adduction", "Hip\nadd."),
    ("hip_rotation_r", "Hip rotation", "Hip\nrot."),
    ("knee_angle_r", "Knee angle", "Knee"),
    ("ankle_angle_r", "Ankle angle", "Ankle"),
    ("subtalar_angle_r", "Subtalar angle", "Sub-\ntalar"),
    ("mtp_angle_r", "MTP angle", "MTP"),
]
# (channel, panel title, short bar label); the figure draws those the cache holds.
EXTENDED_MOMENTS = [
    ("hip_flexion_r", "Hip flexion moment", "Hip\nflex."),
    ("hip_adduction_r", "Hip adduction moment", "Hip\nadd."),
    ("hip_rotation_r", "Hip rotation moment", "Hip\nrot."),
    ("knee_angle_r", "Knee moment", "Knee"),
    ("ankle_angle_r", "Ankle moment", "Ankle"),
    ("subtalar_angle_r", "Subtalar moment", "Sub-\ntalar"),
    ("mtp_angle_r", "MTP moment", "MTP"),
]
EXTENDED_MUSCLES = [
    ("vaslat_r", "Vastus lateralis", "VL"),
    ("vasmed_r", "Vastus medialis", "VM"),
    ("recfem_r", "Rectus femoris", "RF"),
    ("bflh_r", "Biceps femoris", "BF"),
    ("semiten_r", "Semitendinosus", "ST"),
    ("tibant_r", "Tibialis anterior", "TA"),
    ("gasmed_r", "Gastrocnemius medialis", "GM"),
    ("gaslat_r", "Gastrocnemius lateralis", "GL"),
    ("soleus_r", "Soleus", "SOL"),
    ("perlong_r", "Peroneus longus", "PL"),
    ("perbrev_r", "Peroneus brevis", "PB"),
]

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic",
    "mathtext.bf": "Inter:bold",
    "font.size": 15,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.linewidth": 0.7,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
}


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _use_retargeted_joint_reference(state) -> None:
    """Replace the OpenSim-IK angles with the MyoFullBody-retargeted reference.

    Each policy's reference profile is cut at that policy's own heel strikes, so
    profiles of one subject differ slightly between policies. The library's
    ``mfb-retargeted`` mode refuses that; the figure averages them instead and
    reports the largest disagreement so it can be judged.
    """
    human = state.human
    references = np.array(human.joint_data, copy=True)
    worst = np.zeros(len(state.channels.joint_names))
    for index, subject in enumerate(human.subjects):
        profiles = []
        for run in state.policies:
            by_subject = run.profiles.per_subject_joint_ref
            if by_subject is None or subject not in by_subject:
                raise SystemExit(f"policy {run.policy.id} has no retargeted reference for subject {subject}")
            profiles.append(np.asarray(by_subject[subject], dtype=float))
        profiles = np.stack(profiles)
        if profiles.shape[1:] != references.shape[1:] or not np.isfinite(profiles).all():
            raise SystemExit(f"retargeted reference for subject {subject} is malformed")
        references[index] = profiles.mean(axis=0)
        worst = np.maximum(worst, np.abs(profiles - references[index]).max(axis=(0, 2)))
    human.joint_data = references
    print("retargeted reference: max deviation of a policy's profile from the cross-policy mean [deg]")
    for name, value in zip(state.channels.joint_names, worst, strict=True):
        print(f"  {name:<20} {value:.3f}")


def extract(
    *,
    dataset_dir: str | Path,
    kinetics_reference_dir: str | Path,
    policy_config_file: str | Path,
    motion_group: str,
    work_dir: Path,
    muscle_signal: str,
    kinetics_window: str,
    extended_channels: bool = False,
    kinematics_source: str = "opensim-ik",
) -> dict:
    """Run the Gait120 workflow once and collect what the figure needs.

    Every gait cycle of every episode is pooled, which is the view to quote when
    the gait-position split is not the point.
    """
    analysis = FullBodyAnalysis(
        dataset_name="Gait120",
        dataset_dir=dataset_dir,
        policy_config_file=policy_config_file,
        motion_group=motion_group,
        plots_dir=work_dir,
        kinetics_reference_dir=kinetics_reference_dir,
        muscle_signal=muscle_signal,
        kinetics_window=kinetics_window,
        extended_channels=extended_channels,
    )
    state = analysis.state
    if state.kinetics is None:
        raise SystemExit("this figure needs the kinetics layer; pass --kinetics_reference_dir")

    human_data.select_gait_subset(state, None)
    if kinematics_source == "mfb-retargeted":
        _use_retargeted_joint_reference(state)
    metrics.compute_human_to_human_metrics(state)
    references.compute_human_reference_benchmarks(state)
    metrics.compute_human_to_model_metrics_population(state)
    metrics.compute_human_to_model_metrics_per_subject(state)
    kinetics_metrics.compute_kinetics_metrics(state)

    cohort = state.kinetics
    payload = {
        "muscle_signal": muscle_signal,
        "kinetics_window": kinetics_window,
        "extended_channels": extended_channels,
        "kinematics_source": kinematics_source,
        "joint_names": list(state.channels.joint_names),
        "muscle_names": list(state.channels.muscle_names),
        "muscle_map": dict(state.muscle_map),
        "subjects": list(state.human.subjects),
        "human_joint": np.asarray(state.human.joint_data),
        "human_muscle": np.asarray(state.human.muscle_data),
        "torque_names": list(cohort.torque_names),
        "grf_names": list(cohort.grf_names),
        "torque_unit": cohort.torque_unit,
        "grf_unit": cohort.grf_unit,
        "human_torque": np.asarray(cohort.torque_data),
        "human_grf": np.asarray(cohort.grf_data),
        "human_metrics": {
            "joint_rmse": dict(state.human_metrics.pairwise_joint_rmse),
            "joint_corr": dict(state.human_metrics.pairwise_joint_corr),
            "muscle_corr": dict(state.human_metrics.pairwise_muscle_corr),
        },
        "kinetics_human_metrics": {
            "torque_rmse": dict(state.kinetics_human_metrics.torque_rmse),
            "torque_corr": dict(state.kinetics_human_metrics.torque_corr),
            "grf_rmse": dict(state.kinetics_human_metrics.grf_rmse),
            "grf_corr": dict(state.kinetics_human_metrics.grf_corr),
        },
        "policies": [],
    }
    for run in state.policies:
        profiles = run.kinetics_profiles
        payload["policies"].append(
            {
                "id": run.policy.id,
                "label": run.label,
                "color": run.policy.color,
                "linestyle": run.policy.linestyle,
                "avg_joint": np.asarray(run.profiles.avg_joint_gaits),
                "avg_muscle": np.asarray(run.profiles.avg_muscle_gaits),
                "avg_torque": np.asarray(profiles.avg_torque),
                "avg_grf": np.asarray(profiles.avg_grf),
                "joint_rmse": dict(run.per_subject.joint_rmse),
                "joint_corr": dict(run.per_subject.joint_corr),
                "muscle_corr": dict(run.per_subject.muscle_corr),
                "torque_rmse": dict(run.kinetics_per_subject.torque_rmse),
                "torque_corr": dict(run.kinetics_per_subject.torque_corr),
                "grf_rmse": dict(run.kinetics_per_subject.grf_rmse),
                "grf_corr": dict(run.kinetics_per_subject.grf_corr),
                "n_kinetics_cycles": int(run.kinetics_cycles.torque.shape[0]),
                "n_kinetics_omissions": len(run.kinetics_omissions),
            }
        )
    return payload


def apply_policy_style(data: dict, policy_config_file: str | Path) -> None:
    """Refresh label, colour and line style from the policy YAML, in place.

    These are presentation fields, so they are re-read on every run. Without
    this, editing a label or a colour would only take effect after --refresh,
    which re-runs the whole analysis for a change that does not affect it.
    """
    specs = {spec.id: spec for spec in load_policy_specs(policy_config_file)}
    cached = {run["id"] for run in data["policies"]}
    if cached != set(specs):
        raise SystemExit(
            "the cached analysis does not match the policy file "
            f"(cached: {sorted(cached)}, configured: {sorted(specs)}).\n"
            "Re-run with --refresh after adding, removing or renaming a policy."
        )
    for run in data["policies"]:
        spec = specs[run["id"]]
        run["label"] = spec.label
        run["color"] = spec.color
        run["linestyle"] = spec.linestyle


def population_scope(data: dict) -> None:
    """Replace each policy's subject-matched scores with population ones, in place.

    The population profile -- the policy's average over all retained cycles --
    is scored against every retained subject, as the analysis pipeline does in
    ``compute_human_to_model_metrics_population`` and the population branch of
    ``compute_kinetics_metrics``. Each channel still holds one value per subject
    in cohort order, so the bars and paired tests keep the same shape.
    """
    layers = (
        ("joint", "joint_names", "avg_joint", "human_joint", "all_joints"),
        ("muscle", "muscle_names", "avg_muscle", "human_muscle", "all_muscles"),
        ("torque", "torque_names", "avg_torque", "human_torque", "all_moments"),
        ("grf", "grf_names", "avg_grf", "human_grf", "all_grf"),
    )
    for run in data["policies"]:
        for layer, names_key, profile_key, human_key, aggregate in layers:
            names, profile, human = data[names_key], run[profile_key], data[human_key]
            scores = {"corr": paired_pearson} if layer == "muscle" else {"corr": paired_pearson, "rmse": paired_rmse}
            for kind, score in scores.items():
                values = {
                    name: np.array([score(profile[index], subject[index]) for subject in human])
                    for index, name in enumerate(names)
                }
                values[aggregate] = np.array(
                    [finite_mean([values[name][position] for name in names]) for position in range(len(human))]
                )
                run[f"{layer}_{kind}"] = values
    data["metric_scope"] = "population"


def _stem(data: dict, variant: str = "") -> str:
    """Output file stem; population-scope and retargeted figures never overwrite the default ones."""
    retargeted = "_mfb_retargeted" if data.get("kinematics_source") == "mfb-retargeted" else ""
    return f"{STEM}{variant}{retargeted}{'_population' if data.get('metric_scope') == 'population' else ''}"


def _human_trace_label(data: dict) -> str:
    """Name the grey traces after the kinematic ground truth column A draws."""
    if data.get("kinematics_source") == "mfb-retargeted":
        return "Gait120 subjects (angles: MFB-retargeted)"
    return "Gait120 subjects"


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _circular_fill(trace) -> np.ndarray:
    """Fill a cycle's missing samples by interpolating around it as a loop.

    Rollouts end a few frames before their reference trial, so the last
    percent of a right-stride policy kinetics cycle is undefined; gait is
    periodic, so the gap is bridged from the cycle's end back to its start.
    For drawing only: metrics use the measured samples.
    """
    trace = np.asarray(trace, dtype=float)
    finite = np.isfinite(trace)
    if finite.all() or finite.sum() < 2:
        return trace
    phase = np.arange(len(trace), dtype=float)
    return np.interp(phase, phase[finite], trace[finite], period=len(trace))


def _style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.7, pad=2, labelsize=TICK_LABEL_SIZE)


def _waveform(
    ax,
    human,
    policies,
    *,
    title,
    ylabel,
    xlabel=None,
    line_scale=1.0,
    human_color=HUMAN_TRACE,
    zero_linestyle="-",
    zero_color=GRID,
) -> None:
    """Every retained subject in grey behind the policy means, no spread bands."""
    for subject in np.asarray(human, dtype=float):
        ax.plot(
            PHASE, subject, color=human_color, linewidth=0.9 * line_scale, alpha=0.75, zorder=2, solid_capstyle="round"
        )
    for mean, color, linestyle in policies:
        ax.plot(
            PHASE, mean, color=color, linewidth=2.1 * line_scale, linestyle=linestyle, zorder=6, solid_capstyle="round"
        )
    ax.axhline(0.0, color=zero_color, linewidth=0.7, linestyle=zero_linestyle, zorder=1)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_title(title, fontsize=TITLE_SIZE, pad=4, loc="left")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, labelpad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, labelpad=2)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    _style_axis(ax)


def _metric_bars(ax, channel_labels, series, *, ylabel, title, seed=0) -> None:
    """Grouped vertical bars with every underlying observation drawn on top."""
    rng = np.random.default_rng(seed)
    n_series = len(series)
    bar_width = 0.84 / n_series
    centers = np.arange(len(channel_labels), dtype=float)

    for position, (_, color, hatch, values) in enumerate(series):
        offsets = centers + (2 * position - (n_series - 1)) * bar_width / 2
        means = np.array([np.mean(v) if len(v) else np.nan for v in values])
        sds = np.array([np.std(v) if len(v) else np.nan for v in values])
        ax.bar(
            offsets,
            means,
            width=bar_width * 0.86,  # surface gap between adjacent bars
            yerr=sds,
            color=color,
            alpha=0.45 if position == 0 else 0.75,  # position 0 is the human reference
            edgecolor=color,
            linewidth=0.8,
            hatch=hatch,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 0.7, "capsize": 1.5},
            zorder=3,
        )
        for offset, points in zip(offsets, values, strict=True):
            if not len(points):
                continue
            jitter = rng.uniform(-bar_width * 0.22, bar_width * 0.22, size=len(points))
            ax.scatter(
                offset + jitter,
                points,
                s=4.0,
                color=color,
                alpha=0.85,
                edgecolor=SURFACE,
                linewidth=0.3,
                zorder=5,
            )

    ax.axhline(0.0, color=INK_MUTED, linewidth=0.7, zorder=4)
    ax.set_xticks(centers, channel_labels, fontsize=TICK_LABEL_SIZE)
    # A one-group panel needs less side room, so its bars still fill the axes.
    pad = 0.50 if len(centers) == 1 else 0.60
    ax.set_xlim(centers[0] - pad, centers[-1] + pad)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, labelpad=2)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=4, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    _style_axis(ax)
    ax.margins(y=0.10)


def _bar_groups(data: dict) -> list[tuple]:
    """Slot, title, labels, channels and metric keys for each bar panel."""
    joint_labels = [short for *_, short in JOINTS]
    joint_keys = [channel for channel, *_ in JOINTS]
    human_joint = data["human_metrics"]
    human_kinetics = data["kinetics_human_metrics"]
    torque_unit = data["torque_unit"]
    grf_unit = data["grf_unit"]
    return [
        (
            (0, slice(0, 2)),
            (0, slice(2, 4)),
            "Joint angles",
            joint_labels,
            joint_keys,
            (human_joint["joint_corr"], human_joint["joint_rmse"]),
            ("joint_corr", "joint_rmse"),
            ("Pearson r", "RMSE [deg]"),
        ),
        (
            (1, slice(0, 2)),
            (1, slice(2, 4)),
            "Joint moments",
            joint_labels,
            joint_keys,
            (human_kinetics["torque_corr"], human_kinetics["torque_rmse"]),
            ("torque_corr", "torque_rmse"),
            ("Pearson r", f"RMSE [{torque_unit}]"),
        ),
        (
            (2, 2),
            (2, 3),
            "Vertical GRF",
            [""],
            [GRF_CHANNEL],
            (human_kinetics["grf_corr"], human_kinetics["grf_rmse"]),
            ("grf_corr", "grf_rmse"),
            ("Pearson r", f"RMSE [{grf_unit}]"),
        ),
        (
            (2, slice(0, 2)),
            None,  # policy muscle signal and normalized EMG share no scale, so no RMSE
            f"Muscle {data.get('muscle_signal', 'activation')} vs EMG",
            [short for _, _, short in MUSCLES],
            [muscle for muscle, _, _ in MUSCLES],
            (human_joint["muscle_corr"], None),
            ("muscle_corr", None),
            ("Pearson r", None),
        ),
    ]


def _anatomy_legend(fig, slot, anatomy_dir: Path) -> None:
    """Place Blender anatomy with vector callouts, independent of policy colours."""
    legend = json.loads((anatomy_dir / "legend.json").read_text())
    views = slot.subgridspec(1, 3, wspace=0.08)
    muscle_labels = {"vaslat_r": "VL", "tibant_r": "TA", "bflh_r": "BF", "gasmed_r": "GM"}
    for index, (view, title) in enumerate(
        [
            ("anterior", "Anterior"),
            ("posterior", "Posterior"),
            ("angles", "Sagittal"),
        ]
    ):
        ax = fig.add_subplot(views[index])
        ax.imshow(plt.imread(anatomy_dir / f"{view}.png"), extent=(0, 1, 0, 1), aspect="auto")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.set_title(title, fontsize=12, color=INK_MUTED, pad=10)
        for key, point in legend["projections"][view].items():
            color = INK if view == "angles" else legend["highlight_color"]
            label = key if view == "angles" else muscle_labels[key]
            label_x = 0.98 if view == "angles" else 0.04
            ax.annotate(
                label,
                xy=point,
                xytext=(label_x, point[1] + 0.015),
                ha="right" if view == "angles" else "left",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.9},
            )
    bounds = slot.get_position(fig)
    fig.text(
        (bounds.x0 + bounds.x1) / 2,
        bounds.y1 + 0.047,
        "MyoFullBody · right leg",
        ha="center",
        fontsize=TITLE_SIZE,
    )
    fig.text(
        bounds.x0,
        bounds.y0 - 0.035,
        "VL  Vastus lateralis     BF  Biceps femoris (long head)\nTA  Tibialis anterior     GM  Gastrocnemius medialis",
        ha="left",
        va="top",
        fontsize=10,
        color=INK_MUTED,
        linespacing=1.6,
    )


def _pose_images(sequence_dir: Path, manifest: dict, category: str) -> list[np.ndarray]:
    return [plt.imread(sequence_dir / f"{category}_{phase:03d}.png") for phase in manifest["phases"]]


def _pose_crop(images: list[np.ndarray]) -> tuple[int, int, int]:
    """Shared vertical crop (top, bottom) and widest pose, in pixels."""
    rows = np.flatnonzero(np.any(np.stack([image[..., 3] for image in images]).max(axis=0) > 0.03, axis=1))
    top, bottom = int(rows[0]) - 4, int(rows[-1]) + 4
    columns = np.flatnonzero(np.stack([image[top:bottom, :, 3] for image in images]).max(axis=(0, 1)) > 0.03)
    widest_px = (
        max(np.ptp(np.flatnonzero(image[top:bottom, :, 3].max(axis=0) > 0.03)) for image in images)
        if len(columns)
        else 1
    )
    return top, bottom, int(widest_px)


def _pose_height_in(sequence_dir: Path, manifest: dict, category: str, width_in: float, max_width: float) -> float:
    """Strip height, in inches, at which width rather than height sets the pose scale."""
    top, bottom, widest_px = _pose_crop(_pose_images(sequence_dir, manifest, category))
    return (bottom - top) * max_width / 100.0 * width_in / widest_px


def _pose_strip(fig, ax, sequence_dir: Path, manifest: dict, category: str, *, max_width: float | None = None) -> None:
    """Draw every pose with its hip on its phase tick, all at one pixel scale.

    The strip's x axis is the waveform x axis (0-100) and its y axis is image
    pixels, so a pose is placed by converting its pixel width into phase units
    from the axes' size in inches. One shared vertical crop and one scale keep
    poses comparable; the scale is capped so the widest pose spans at most
    ``max_width`` (default ``MAX_POSE_WIDTH``) phase units, and poses at 0 % and
    100 % overhang the axes.
    """
    images = _pose_images(sequence_dir, manifest, category)
    top, bottom, widest_px = _pose_crop(images)
    height_px = bottom - top
    box = ax.get_position()
    width_in, height_in = box.width * fig.get_figwidth(), box.height * fig.get_figheight()
    inches_per_px = min(height_in / height_px, (max_width or MAX_POSE_WIDTH) / 100.0 * width_in / widest_px)
    units_per_px = inches_per_px * 100.0 / width_in
    # Bottom-aligned, so the feet of every column share one ground line.
    ax.set(xlim=(0, 100), ylim=(0, height_in / inches_per_px))
    ax.axis("off")
    for phase, image, projected in zip(manifest["phases"], images, manifest["projections"][category], strict=True):
        hip_px = projected["hip"][0] * image.shape[1]
        ax.imshow(
            image[top:bottom],
            extent=(phase - hip_px * units_per_px, phase + (image.shape[1] - hip_px) * units_per_px, 0, height_px),
            interpolation="antialiased",
            clip_on=False,
            zorder=3,
        )
    ax.set_aspect("auto")

    def to_data(phase_index: int, name: str) -> tuple[float, float]:
        image, projected = images[phase_index], manifest["projections"][category][phase_index]
        hip_px = projected["hip"][0] * image.shape[1]
        x_px, y_px = projected[name][0] * image.shape[1], (1 - projected[name][1]) * image.shape[0]
        return manifest["phases"][phase_index] + (x_px - hip_px) * units_per_px, bottom - y_px

    return to_data


MAX_POSE_WIDTH = 25.0


def _stance_bar(ax, stance: Sequence[bool | None], *, fontsize: float = 13) -> None:
    """Right-foot stance (dark) and swing (light); unmeasured phases stay blank."""
    state = np.array([-1 if value is None else int(value) for value in stance])
    edges = np.flatnonzero(np.diff(state)) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(state)], strict=True):
        if state[lo] < 0:
            continue
        on = bool(state[lo])
        ax.axvspan(lo, min(hi, 100), color=INK_MUTED if on else GRID, alpha=0.85 if on else 0.55, lw=0, zorder=1)
        if hi - lo >= 14:
            ax.text(
                (lo + min(hi, 100)) / 2,
                0.5,
                "stance" if on else "swing",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=SURFACE if on else INK_MUTED,
                zorder=2,
            )
    ax.set(xlim=(0, 100), ylim=(0, 1))
    ax.axis("off")


# The leg faces right: anterior VL and TA are labelled in front, posterior BF
# and GM behind, each leader ending on the muscle belly.
MUSCLE_POSE_LABELS = [("vaslat_r", "VL", 20), ("tibant_r", "TA", 20), ("bflh_r", "BF", -22), ("gasmed_r", "GM", -22)]


def label_pose(strip, to_data, labels, *, dot=None) -> None:
    """Label the heel-strike pose; a negative offset places the label behind the leg."""
    for name, label, dx in labels:
        x, y = to_data(0, name)
        strip.annotate(
            label,
            (x, y),
            xytext=(dx, 0),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            va="center",
            fontsize=14.5,
            color=INK,
            zorder=6,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": INK_MUTED, "shrinkA": 1.5, "shrinkB": 0},
        )
        if dot:
            strip.plot(x, y, "o", ms=3.4, mfc=dot, mec=SURFACE, mew=0.6, zorder=7, clip_on=False)


def label_joints_column(fig, strip, to_data, labels, *, offset_pt=24.0, fontsize=14.5) -> None:
    """Label the heel-strike pose's joints in one column behind the leg.

    ``labels`` holds (joint, label, dy_pt); the foot joints sit a few
    centimetres apart, so ``dy_pt`` spreads their labels vertically.
    """
    box = strip.get_position()
    (x0, x1), (y0, y1) = strip.get_xlim(), strip.get_ylim()
    x_per_pt = (x1 - x0) / (box.width * fig.get_figwidth() * 72)
    y_per_pt = (y1 - y0) / (box.height * fig.get_figheight() * 72)
    points = {name: to_data(0, name) for name, _, _ in labels}
    anchor = min(x for x, _ in points.values()) - offset_pt * x_per_pt
    for name, label, dy in labels:
        x, y = points[name]
        strip.annotate(
            label,
            (x, y),
            xytext=(anchor, y + dy * y_per_pt),
            textcoords="data",
            ha="right",
            va="center",
            fontsize=fontsize,
            color=INK,
            zorder=6,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": INK_MUTED, "shrinkA": 1.5, "shrinkB": 0},
        )


# Joint landmarks of the joints strip; dy_pt fans out the three foot joints.
JOINT_POSE_LABELS = [
    ("hip_flexion_r", "Hip", 0),
    ("knee_angle_r", "Knee", 0),
    ("ankle_angle_r", "Ankle", 20),
    ("subtalar_angle_r", "Subtalar", 4),
    ("mtp_angle_r", "MTP", -12),
]


def _muscle_colour_bar(
    fig, ax, manifest: dict, label: str, bounds=(0.80, 1.04, 0.20, 0.05), orientation: str = "horizontal"
) -> None:
    low, high = manifest["muscle_colour_ramp"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("muscle_colour", [low, high])
    cax = ax.inset_axes(list(bounds))
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap),
        cax=cax,
        orientation=orientation,
        ticks=[0, 1],
    )
    cbar.outline.set_visible(False)
    cax.tick_params(labelsize=13, length=0, pad=1.5)
    if orientation == "horizontal":
        cax.set_title(label, fontsize=13, color=INK_MUTED, pad=2)
    else:
        cbar.set_label(label, fontsize=13, color=INK_MUTED, labelpad=4)


def _render_overlay_figure(data: dict, output_dir: Path, sequence_dir: Path) -> Path:
    """Gait120 mean against each policy mean, drawn as GRF and muscle overlays.

    Every row shares the same policy poses, so only the overlay differs: the
    vertical GRF arrow, the muscle colour on one absolute 0-1 scale, and the
    stance bar where that row's mean vertical GRF exceeds 0.05 BW.
    """
    manifest = json.loads((sequence_dir / "sequence.json").read_text())
    signal = data.get("muscle_signal", "activation")
    rows = ["gait120", *[run["id"] for run in data["policies"]]]
    stride_kinetics = manifest.get("kinetics_window", "left_heel_strike") == "right_stride"
    runs = {run["id"]: run for run in data["policies"]}
    fig = plt.figure(figsize=(15.5, 1.62 * len(rows) + 1.5))
    grid = fig.add_gridspec(
        len(rows) + 1,
        3,
        width_ratios=[0.34, 1, 1],
        height_ratios=[*[1] * len(rows), 0.16],
        left=0.01,
        right=0.985,
        top=1 - 0.95 / fig.get_figheight(),
        bottom=0.02,
        wspace=0.26,
        hspace=0.18,
    )
    for col, (title, subtitle) in enumerate(
        [
            ("Vertical GRF", "Mean right-foot force" + ("" if stride_kinetics else ", 1.00 s from left heel strike")),
            (f"Muscle {signal} vs EMG", f"Mean {signal} (policies) and MVC-norm. EMG (Gait120), one 0–1 scale"),  # noqa: RUF001
        ],
        start=1,
    ):
        box = grid[0, col].get_position(fig)
        fig.text(box.x0, 1 - 0.36 / fig.get_figheight(), title, fontsize=16, fontweight="bold", va="baseline")
        fig.text(box.x0, 1 - 0.6 / fig.get_figheight(), subtitle, fontsize=12, color=INK_MUTED, va="baseline")

    for row, key in enumerate(rows):
        overlay = manifest["overlays"][f"compare_{key}"]
        label_ax = fig.add_subplot(grid[row, 0])
        label_ax.axis("off")
        if key == "gait120":
            label_ax.plot([0.02, 0.17], [0.55, 0.55], color=HUMAN_TRACE, lw=2.4, transform=label_ax.transAxes)
            name = overlay["label"]
        else:
            run = runs[key]
            label_ax.plot(
                [0.02, 0.17],
                [0.55, 0.55],
                color=run["color"],
                lw=2.4,
                linestyle=_line_style(run["linestyle"]),
                transform=label_ax.transAxes,
            )
            name = run["label"]
        # Long run names wrap at their hyphens so they stay inside the label column.
        lines = [textwrap.fill(part, 24, break_on_hyphens=True) for part in name.replace(" (", "\n(").split("\n")]
        label_ax.text(
            0.22,
            0.55,
            "\n".join(lines),
            fontsize=13,
            va="center",
            ha="left",
            linespacing=1.35,
            transform=label_ax.transAxes,
        )
        for col, category in enumerate(("kinetics", "muscles"), start=1):
            cell = grid[row, col].subgridspec(2, 1, height_ratios=[1, 0.13], hspace=0.06)
            strip = fig.add_subplot(cell[0])
            to_data = _pose_strip(fig, strip, sequence_dir, manifest, f"{category}_compare_{key}")
            # Stance is known per row only on the kinetics window; the muscle
            # strips follow the right stride, whose policy stance the cache lacks.
            if category == "kinetics" or stride_kinetics:
                _stance_bar(fig.add_subplot(cell[1]), overlay["stance"])
            if row == 0 and category == "muscles":
                label_pose(strip, to_data, MUSCLE_POSE_LABELS, dot=INK)
        if row == 0:
            # A rule separates the human reference from the policies.
            y = grid[0, 0].get_position(fig).y0 - 0.006
            fig.add_artist(Line2D([0.01, 0.985], [y, y], color=GRID, lw=0.9, transform=fig.transFigure))

    # The colour key sits in the free cell under the row labels.
    key = fig.add_subplot(grid[len(rows), 0])
    key.axis("off")
    _muscle_colour_bar(fig, key, manifest, f"{signal.capitalize()} / MVC-norm. EMG", bounds=(0.06, 0.2, 0.8, 0.45))
    for col in (1, 2):
        seconds = col == 1 and not stride_kinetics
        axis = fig.add_subplot(grid[len(rows), col])
        axis.set(xlim=(0, 100), yticks=[])
        axis.set_xticks(manifest["phases"])
        axis.set_xticklabels([f"{phase / 100:.1f}" if seconds else f"{phase}%" for phase in manifest["phases"]])
        axis.tick_params(labelsize=12, length=3, colors=INK_MUTED)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.set_xlabel("Time from left heel strike [s]" if seconds else XLABEL, fontsize=13)
        axis.patch.set_alpha(0)

    stem = output_dir / "gait120_overlays"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def _render_sequence_figure(data: dict, output_dir: Path, sequence_dir: Path) -> Path:
    """Waveform columns under lateral pose strips, then one column of metrics.

    Columns A-C share the right-stride phase axis, and every pose sits on the
    tick of the phase it shows. Column D holds the subject-matched agreement
    bars for all three layers, which keeps the figure wide rather than tall.
    """
    manifest = json.loads((sequence_dir / "sequence.json").read_text())
    signal = data.get("muscle_signal", "activation")
    right_stride = data.get("kinetics_window", "left_heel_strike") == "right_stride"
    fig = plt.figure(figsize=(23.5, 12.2))
    columns = fig.add_gridspec(
        1,
        6,
        width_ratios=[1, 0.03, 1, 1, 0.02, 0.95],
        left=0.048,
        right=0.988,
        bottom=0.055,
        top=0.952,
        wspace=0.40,
    )
    groups = _bar_groups(data)
    phases = manifest["phases"]
    slots = (0, 2, 3, 5)  # gridspec column of A, B, C and D

    def series(group, metric):
        _, _, _, _, keys, human_source, run_keys, _ = group
        return [
            ("Gait120 between-subject", HUMAN_BAR, "///", [_finite(human_source[metric][k]) for k in keys]),
            *[
                (
                    run["label"],
                    run["color"],
                    "\\\\" if _is_dashed(run["linestyle"]) else None,
                    [_finite(run[run_keys[metric]][k]) for k in keys],
                )
                for run in data["policies"]
            ],
        ]

    def waveform(slot, key, names, channel, human_key, title, ylabel, *, bottom=False, seconds=False):
        ax = fig.add_subplot(slot)
        index = data[names].index(channel)
        _waveform(
            ax,
            data[human_key][:, index],
            [(run[key][index], run["color"], _line_style(run["linestyle"])) for run in data["policies"]],
            title=title,
            ylabel=ylabel,
            xlabel=("Time from left heel strike [s]" if seconds else XLABEL) if bottom else None,
        )
        ax.set_title(title, loc="left", fontsize=17.5, pad=3)
        ax.set_ylabel(ylabel, fontsize=16, labelpad=3)
        ax.xaxis.label.set_fontsize(17.5)
        ax.tick_params(labelsize=15)
        ax.set_xticks(phases)
        ax.set_xticklabels([f"{p / 100:.1f}" for p in phases] if seconds else [str(p) for p in phases])
        for phase in phases[1:-1]:
            ax.axvline(phase, color=GRID, linewidth=0.7, alpha=0.8, zorder=0)
        if not bottom:
            ax.tick_params(labelbottom=False)
        return ax

    def bars(slot, labels, values, ylabel, seed, title=None):
        ax = fig.add_subplot(slot)
        _metric_bars(ax, labels, values, ylabel=ylabel, title="", seed=seed)
        ax.set_ylabel(ylabel, fontsize=16, labelpad=3)
        ax.tick_params(labelsize=15)
        if title:
            ax.set_title(title, loc="left", fontsize=17.5, pad=3)
        return ax

    def header(col, title, subtitle=None):
        # Narrow spacer columns sit after A, so B's first pose clears A's
        # last, and before D, which sets it further right.
        box = columns[slots[col]].get_position(fig)
        fig.text(box.x0 - 0.032, 0.970, "ABCD"[col], fontsize=26, fontweight="bold", ha="left", va="baseline")
        fig.text(box.x0, 0.970, title, fontsize=21, fontweight="bold", ha="left", va="baseline")
        if subtitle:
            fig.text(box.x0, 0.949, subtitle, fontsize=15.5, color=INK_MUTED, ha="left", va="baseline")
        return box

    headers = [
        ("joints", "Joint kinematics", None),
        ("kinetics", "Joint moments & GRF", None if right_stride else "Left heel strike → +1.00 s"),
        ("muscles", f"Muscle {signal} vs EMG", None),
    ]
    for col, (category, title, subtitle) in enumerate(headers):
        box = header(col, title, subtitle)
        block = columns[slots[col]].subgridspec(3, 1, height_ratios=[1.18, 0.42, 5.6], hspace=0.0)
        first_axis = len(fig.axes)
        strip = fig.add_subplot(block[0])
        to_data = _pose_strip(
            fig, strip, sequence_dir, manifest, category if category == "joints" else f"{category}_gait120"
        )
        stance_ax = block[1].subgridspec(3, 1, height_ratios=[0.12, 0.38, 1.0], hspace=0)[1]
        # Column B follows the kinetics window; A and C the right stride.
        _stance_bar(
            fig.add_subplot(stance_ax),
            manifest["overlays"]["gait120"]["stance"] if category == "kinetics" else manifest["cycle_stance"],
        )

        if col == 0:
            label_pose(
                strip,
                to_data,
                [("hip_flexion_r", "Hip", -24), ("knee_angle_r", "Knee", -24), ("ankle_angle_r", "Ankle", -24)],
            )
            # Same four-row grid as B and C, so panels match; the fourth row
            # holds the legend.
            waves = block[2].subgridspec(4, 1, hspace=0.40)
            for row, (channel, angle, _, _) in enumerate(JOINTS):
                waveform(
                    waves[row],
                    "avg_joint",
                    "joint_names",
                    channel,
                    "human_joint",
                    angle,
                    "Angle [deg]",
                    bottom=row == 2,
                )
            legend_slot = waves[3]
        elif col == 1:
            waves = block[2].subgridspec(4, 1, hspace=0.40)
            for row, (channel, _, moment, _) in enumerate(JOINTS):
                waveform(
                    waves[row],
                    "avg_torque",
                    "torque_names",
                    channel,
                    "human_torque",
                    moment,
                    f"[{data['torque_unit']}]",
                    seconds=not right_stride,
                )
            waveform(
                waves[3],
                "avg_grf",
                "grf_names",
                GRF_CHANNEL,
                "human_grf",
                "Vertical GRF",
                f"[{data['grf_unit']}]",
                bottom=True,
                seconds=not right_stride,
            )
        else:
            label_pose(strip, to_data, MUSCLE_POSE_LABELS, dot=INK)
            # In the gutter before column D, clear of the title and the poses.
            _muscle_colour_bar(
                fig,
                strip,
                manifest,
                manifest["overlays"]["gait120"]["colour_label"].replace(" (", "\n("),
                bounds=(1.17, 0.12, 0.025, 0.8),
                orientation="vertical",
            )
            waves = block[2].subgridspec(4, 1, hspace=0.40)
            for row, (muscle, name, short) in enumerate(MUSCLES):
                ax = waveform(
                    waves[row],
                    "avg_muscle",
                    "muscle_names",
                    muscle,
                    "human_muscle",
                    f"{short} · {name}",
                    signal.capitalize(),
                    bottom=row == 3,
                )
                # Policy signal and MVC-normalized EMG share one 0-1 scale; the
                # right axis mirrors it so each side names its own trace.
                emg = ax.secondary_yaxis("right")
                emg.set_ylabel("MVC-norm. EMG", fontsize=16, labelpad=4, color=INK_MUTED)
                emg.tick_params(labelsize=15, length=3, width=0.7, pad=2)
                emg.spines["right"].set_color(INK_MUTED)
        fig.align_ylabels([ax for ax in fig.axes[first_axis:] if ax.get_ylabel()])

    # Column D: agreement with Gait120, one section per physiological column.
    box = header(3, "Policy comparison")
    # The hatched human bars are only in this column, so their key sits here.
    fig.legend(
        handles=[
            Patch(facecolor=HUMAN_BAR, alpha=0.45, hatch="///", edgecolor=HUMAN_BAR, label="Gait120 between-subject")
        ],
        loc="upper left",
        bbox_to_anchor=(box.x0 - 0.004, 0.944),
        frameon=False,
        fontsize=15,
        handlelength=1.9,
        handletextpad=0.5,
        borderaxespad=0,
        borderpad=0,
    )
    # The first row is a spacer that keeps the subtitle clear of the first bar title.
    rows = columns[slots[3]].subgridspec(6, 1, height_ratios=[0.06, 1, 1, 1, 1, 1.15], hspace=0.66)
    metrics = [rows[index] for index in range(1, 6)]
    first_axis = len(fig.axes)
    for metric in range(2):
        bars(
            metrics[metric],
            groups[0][3],
            series(groups[0], metric),
            groups[0][-1][metric],
            metric,
            title="Joint angles" if metric == 0 else None,
        )
    torque_series, grf_series = series(groups[1], 0), series(groups[2], 0)
    combined = [(a[0], a[1], a[2], a[3] + b[3]) for a, b in zip(torque_series, grf_series, strict=True)]
    bars(metrics[2], ["Hip", "Knee", "Ankle", "vGRF"], combined, "Pearson r", 2, title="Joint moments & vGRF")
    errors = metrics[3].subgridspec(1, 2, width_ratios=[3, 1.15], wspace=0.40)
    bars(errors[0], groups[1][3], series(groups[1], 1), f"RMSE [{data['torque_unit']}]", 3)
    bars(errors[1], ["vGRF"], series(groups[2], 1), f"RMSE [{data['grf_unit']}]", 4)
    bars(metrics[4], groups[3][3], series(groups[3], 0), "Pearson r", 6, title=f"Muscle {signal} vs EMG")
    fig.align_ylabels([ax for ax in fig.axes[first_axis:] if abs(ax.get_position().x0 - box.x0) < 1e-3])

    # The legend fills column A's free fourth row: Gait120 in one row on top,
    # the policies in a 2 x 2 block below under the label prefix they share.
    labels = [run["label"] for run in data["policies"]]
    prefix = labels[0].split(" (")[0] if all(" (" in label for label in labels) else ""
    if prefix and all(label.startswith(f"{prefix} (") for label in labels):
        labels = [label[len(prefix) + 2 :].rstrip(")") for label in labels]
    else:
        prefix = "Policies"
    # A shared stem such as "mm-motions-" also moves into the title.
    stem = os.path.commonprefix(labels)
    stem = stem[: stem.rfind("-") + 1] if "-" in stem else ""
    if stem and all(label[len(stem) :] for label in labels):
        prefix, labels = f"{prefix} · {stem.rstrip('-')}", [label[len(stem) :] for label in labels]
    legend_ax = fig.add_subplot(legend_slot)
    legend_ax.axis("off")
    style = {
        "frameon": False,
        "fontsize": 15.5,
        "title_fontsize": 16.5,
        "handlelength": 1.9,
        "ncol": 2,
        "borderaxespad": 0,
        "labelspacing": 0.4,
        "columnspacing": 1.1,
        "handletextpad": 0.5,
        "alignment": "left",
    }
    human = legend_ax.legend(
        handles=[Line2D([], [], color=HUMAN_TRACE, lw=1.4, label="Subjects")],
        title="Gait120",
        loc="upper left",
        bbox_to_anchor=(0.04, 0.93),
        **style,
    )
    legend_ax.add_artist(human)
    policies = legend_ax.legend(
        handles=[
            Line2D([], [], color=r["color"], lw=2.2, linestyle=_line_style(r["linestyle"]), label=label)
            for r, label in zip(data["policies"], labels, strict=True)
        ],
        title=prefix,
        loc="upper left",
        bbox_to_anchor=(0.04, 0.47),
        **{
            **style,
            "ncol": len(labels) if len(labels) <= 4 and max(map(len, labels)) <= 4 else 2,
            "columnspacing": 0.8,
            "handlelength": 1.7,
        },
    )
    for legend in (human, policies):
        legend.get_title().set_fontweight("bold")
    # The box is sized from the drawn legends, so their text always sits inside.
    renderer = fig.canvas.get_renderer()
    extent = mpl.transforms.Bbox.union([legend.get_window_extent(renderer) for legend in (human, policies)])
    (x0, y0), (x1, y1) = legend_ax.transAxes.inverted().transform(extent.get_points())
    pad_x, pad_y = 0.035, 0.07
    legend_ax.add_patch(
        FancyBboxPatch(
            (x0 - pad_x, y0 - pad_y),
            x1 - x0 + 2 * pad_x,
            y1 - y0 + 2 * pad_y,
            boxstyle="round,pad=0,rounding_size=0.03",
            transform=legend_ax.transAxes,
            facecolor=SURFACE,
            edgecolor=GRID,
            linewidth=1.0,
            clip_on=False,
            zorder=0,
        )
    )

    # Preserve exactly the original channel-wise metric export, regardless of layout.
    rows_csv = []
    for group in groups:
        for metric in range(2):
            if group[1] is None and metric == 1:
                continue
            for label, _, _, values in series(group, metric):
                for key, sample in zip(group[4], values, strict=True):
                    rows_csv.append(
                        {
                            "signal": group[2],
                            "channel": key,
                            "metric": "pearson_r" if metric == 0 else "rmse",
                            "unit": "" if metric == 0 else group[-1][metric].split("[")[-1].rstrip("]"),
                            "entity": label,
                            "n": len(sample),
                            "mean": float(np.mean(sample)) if len(sample) else np.nan,
                            "sd": float(np.std(sample)) if len(sample) else np.nan,
                            "values": ";".join(f"{value:.6g}" for value in sample),
                        }
                    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(data)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    with (output_dir / f"{stem}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)
    return output_dir / f"{stem}.png"


# Pose labels of the extended muscle strips: anterior muscles in front of the
# leg (positive offset), the rest behind it, where the next pose leaves room.
EXTENDED_POSE_SIDES = {
    "vaslat_r": 1,
    "vasmed_r": 1,
    "recfem_r": 1,
    "bflh_r": -1,
    "semiten_r": -1,
    "tibant_r": 1,
    "perlong_r": -1,
    "perbrev_r": -1,
    "gasmed_r": -1,
    "gaslat_r": -1,
    "soleus_r": -1,
}


def label_pose_stacked(fig, strip, to_data, names, *, offset_pt=22.0, gap_pt=15.5, fontsize=14.5) -> None:
    """Label the heel-strike pose, stacking each side's labels so none overlap.

    Each side's labels keep their top-to-bottom order and at least ``gap_pt``
    between baselines, centred on the muscle bellies they point to.
    """
    short = {muscle: label for muscle, _, label in EXTENDED_MUSCLES}
    box = strip.get_position()
    (x0, x1), (y0, y1) = strip.get_xlim(), strip.get_ylim()
    x_per_pt = (x1 - x0) / (box.width * fig.get_figwidth() * 72)
    y_per_pt = (y1 - y0) / (box.height * fig.get_figheight() * 72)
    for side in (1, -1):
        points = sorted(
            ((to_data(0, name), name) for name in names if EXTENDED_POSE_SIDES[name] == side),
            key=lambda item: -item[0][1],
        )
        if not points:
            continue
        ys = [y for (_, y), _ in points]
        placed = [ys[0]]
        for y in ys[1:]:
            placed.append(min(y, placed[-1] - gap_pt * y_per_pt))
        shift = np.mean(ys) - np.mean(placed)
        # Every label sits in one column, clear of the leg on its side.
        anchor = (max if side > 0 else min)(x for (x, _), _ in points) + side * offset_pt * x_per_pt
        for ((x, y), name), label_y in zip(points, placed, strict=True):
            strip.annotate(
                short[name],
                (x, y),
                xytext=(anchor, label_y + shift),
                textcoords="data",
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=fontsize,
                color=INK,
                zorder=6,
                annotation_clip=False,
                path_effects=[patheffects.withStroke(linewidth=3.5, foreground=SURFACE)],
                arrowprops={"arrowstyle": "-", "lw": 0.7, "color": INK_MUTED, "shrinkA": 1.5, "shrinkB": 0},
            )
            strip.plot(x, y, "o", ms=3.4, mfc=INK, mec=SURFACE, mew=0.6, zorder=7, clip_on=False)


def _render_extended_figure(data: dict, output_dir: Path, sequence_dir: Path) -> Path:
    """The sequence layout with every right-leg joint angle and EMG muscle.

    Columns follow the main figure: A joint angles, B moments and vertical
    GRF, C muscles split into a thigh and a shank column, each under its own
    pose strip, and D the subject-matched agreement for every channel. With a
    ``right_stride`` analysis B shares A's phase axis, and the few undefined
    percent at the end of each policy kinetics cycle are drawn bridged (see
    ``_circular_fill``). Angles
    and moments are those of ``EXTENDED_JOINTS`` and ``EXTENDED_MOMENTS`` the
    analysis holds, so an ``extended_channels`` analysis adds the hip
    adduction/rotation, subtalar and MTP channels. Render the poses with
    ``render_gait120_sequence.py --extended``.
    """
    manifest = json.loads((sequence_dir / "sequence.json").read_text())
    groups = manifest.get("muscle_groups")
    if not groups:
        raise SystemExit(f"{sequence_dir} has no muscle groups; render it with render_gait120_sequence.py --extended")
    names = {muscle: (name, short) for muscle, name, short in EXTENDED_MUSCLES}
    signal = data.get("muscle_signal", "activation")
    right_stride = data.get("kinetics_window", "left_heel_strike") == "right_stride"
    phases = manifest["phases"]
    runs = data["policies"]
    joints = [joint for joint in EXTENDED_JOINTS if joint[0] in data["joint_names"]]
    moments = [moment for moment in EXTENDED_MOMENTS if moment[0] in data["torque_names"]]
    # Column B adds the vertical GRF to its moments; C's fewer muscle rows are
    # taller, so every column ends on one baseline.
    n_rows = max(len(joints), len(moments) + 1)
    n_muscle_rows = max(map(len, groups.values()))
    max_pose_width = 34.0  # neighbouring poses may overlap a little

    # Every height is set in inches, then turned into figure fractions, so the
    # strips are exactly as tall as their poses.
    row_in, row_gap_in = 1.62, 0.72  # one waveform panel, and the title/tick gap below it
    header_in, stance_in = 0.78, 0.80  # the stance bar, then room for the first panel title
    legend_in = 1.25  # x labels of the bottom row, then the legend
    d_width = max(1.05, 0.2 * max(len(joints), len(moments) + 1))
    # A, B, C-thigh and C-shank sit exactly one pose step apart, so their four
    # strips read as one evenly spaced sequence of cycles; the gap still clears
    # the next column's y tick labels (units sit in the panel titles, so there
    # is no y label to clear). Each C strip labels its 0 % pose on the left,
    # over its neighbour's 100 % pose, so those labels carry a white halo.
    col_in = 4.2
    step = phases[1] - phases[0]
    gap_in = col_in * step / 100
    # The gap before D holds its tick labels and the last pose's overhang.
    d_in, d_gap_in = 3.85 * d_width, 1.65
    left_in, right_in = 1.15, 0.3
    width = left_in + 4 * col_in + 3 * gap_in + d_gap_in + d_in + right_in
    fig = plt.figure(figsize=(width, 10.0))
    block_in = 4 * col_in + 3 * gap_in
    columns = fig.add_gridspec(
        1,
        2,
        width_ratios=[block_in, d_in],
        wspace=d_gap_in / ((block_in + d_in) / 2),
        left=left_in / width,
        right=1 - right_in / width,
    )
    packed = columns[0].subgridspec(1, 4, wspace=gap_in / col_in)
    slots = (*(packed[index] for index in range(4)), columns[1])  # A, B, C-thigh, C-shank and D
    strip_names = ["joints", "kinetics_gait120", *[f"muscles-{group}_gait120" for group in groups]]
    width_in = slots[0].get_position(fig).width * fig.get_figwidth()
    strip_in = max(_pose_height_in(sequence_dir, manifest, name, width_in, max_pose_width) for name in strip_names)
    waves_in = n_rows * row_in + (n_rows - 1) * row_gap_in
    height = header_in + strip_in + stance_in + waves_in + legend_in
    fig.set_figheight(height)
    top, bottom = 1 - header_in / height, legend_in / height
    columns.update(top=top, bottom=bottom)

    def header(slot, letter, title, subtitle=None):
        box = slot.get_position(fig)
        fig.text(box.x0 - 0.026, 1 - 0.36 / height, letter, fontsize=28, fontweight="bold", ha="left", va="baseline")
        fig.text(box.x0, 1 - 0.36 / height, title, fontsize=23, fontweight="bold", ha="left", va="baseline")
        if subtitle:
            fig.text(box.x0, 1 - 0.64 / height, subtitle, fontsize=17, color=INK_MUTED, ha="left", va="baseline")
        return box

    def waveform(slot, key, names_key, channel, human_key, title, unit=None, *, bottom=False, seconds=False):
        ax = fig.add_subplot(slot)
        index = data[names_key].index(channel)
        policies = [run[key][index] for run in runs]
        if key in ("avg_torque", "avg_grf"):
            policies = [_circular_fill(trace) for trace in policies]
        _waveform(
            ax,
            data[human_key][:, index],
            [(trace, run["color"], _line_style(run["linestyle"])) for trace, run in zip(policies, runs, strict=True)],
            title=title,
            ylabel="",
            xlabel=("Time from left heel strike [s]" if seconds else XLABEL) if bottom else None,
            line_scale=1.3,
            human_color=HUMAN_TRACE,
            # Dotted, so the zero reference is not mistaken for a data trace.
            zero_linestyle=(0, (1, 2.5)),
            zero_color=INK_MUTED,
        )
        # The unit rides in the title, so the column needs no y label.
        ax.set_title(f"{title} [{unit}]" if unit else title, loc="left", fontsize=19.5, pad=3)
        ax.xaxis.label.set_fontsize(19.5)
        ax.tick_params(labelsize=17, width=0.9)
        ax.grid(False)
        ax.set_xticks(phases)
        ax.set_xticklabels([f"{p / 100:.1f}" for p in phases] if seconds else [str(p) for p in phases])
        if not bottom:
            ax.tick_params(labelbottom=False)
        return ax

    def column(slot, strip_name, stance, rows):
        """Pose strip, stance bar and ``rows`` waveform panels filling the column."""
        block = slot.subgridspec(3, 1, height_ratios=[strip_in, stance_in, waves_in], hspace=0.0)
        strip = fig.add_subplot(block[0])
        to_data = _pose_strip(fig, strip, sequence_dir, manifest, strip_name, max_width=max_pose_width)
        stance_ax = block[1].subgridspec(3, 1, height_ratios=[0.04, 0.22, 0.54], hspace=0)[1]
        _stance_bar(fig.add_subplot(stance_ax), stance, fontsize=15)
        # One absolute gap between panels, whatever their number.
        panel_in = (waves_in - (rows - 1) * row_gap_in) / rows
        return strip, to_data, block[2].subgridspec(rows, 1, hspace=row_gap_in / panel_in)

    # A: every right-leg joint angle.
    header(slots[0], "A", "Joint kinematics")
    strip, to_data, waves = column(slots[0], "joints", manifest["cycle_stance"], n_rows)
    label_joints_column(fig, strip, to_data, JOINT_POSE_LABELS, fontsize=16.5)
    for row, (channel, title, _) in enumerate(joints):
        waveform(
            waves[row], "avg_joint", "joint_names", channel, "human_joint", title, "deg", bottom=row == len(joints) - 1
        )

    # B: moments and vertical GRF.
    header(slots[1], "B", "Joint moments & GRF", None if right_stride else "Left heel strike → +1.00 s")
    _, _, waves = column(slots[1], "kinetics_gait120", manifest["overlays"]["gait120"]["stance"], n_rows)
    for row, (channel, moment, _) in enumerate(moments):
        waveform(
            waves[row],
            "avg_torque",
            "torque_names",
            channel,
            "human_torque",
            moment,
            data["torque_unit"],
            seconds=not right_stride,
        )
    waveform(
        waves[len(moments)],
        "avg_grf",
        "grf_names",
        GRF_CHANNEL,
        "human_grf",
        "Vertical GRF",
        data["grf_unit"],
        bottom=True,
        seconds=not right_stride,
    )

    # C: every EMG muscle, one column per group under its own pose strip.
    header(slots[2], "C", f"Muscle {signal} vs EMG")
    for slot, (group, muscles) in zip(slots[2:4], groups.items(), strict=True):
        strip, to_data, waves = column(slot, f"muscles-{group}_gait120", manifest["cycle_stance"], n_muscle_rows)
        # Each group is named on the subtitle line, under the shared C title.
        fig.text(
            slot.get_position(fig).x0,
            1 - 0.64 / height,
            group.capitalize(),
            fontsize=17,
            color=INK_MUTED,
            ha="left",
            va="baseline",
        )
        label_pose_stacked(fig, strip, to_data, muscles, gap_pt=17.0, fontsize=16.5)
        if slot == slots[3]:
            # A short horizontal key on the subtitle line, clear of the poses.
            _muscle_colour_bar(
                fig,
                strip,
                manifest,
                manifest["overlays"]["gait120"]["colour_label"],
                bounds=(0.55, 1 + 0.42 / strip_in, 0.42, 0.10 / strip_in),
            )
        for row, muscle in enumerate(muscles):
            name, short = names[muscle]
            waveform(
                waves[row],
                "avg_muscle",
                "muscle_names",
                muscle,
                "human_muscle",
                f"{short} · {name}",
                bottom=row == len(muscles) - 1,
            )

    # D: agreement with Gait120 for every channel drawn in A-C.
    header(slots[4], "D", "Policy comparison")
    human_joint, human_kinetics = data["human_metrics"], data["kinetics_human_metrics"]

    def series(human_source, run_key, keys):
        return [
            ("Gait120 between-subject", HUMAN_BAR, "///", [_finite(human_source[k]) for k in keys]),
            *[
                (
                    run["label"],
                    run["color"],
                    "\\\\" if _is_dashed(run["linestyle"]) else None,
                    [_finite(run[run_key][k]) for k in keys],
                )
                for run in runs
            ],
        ]

    def bars(slot, labels, values, metric, seed, title=None):
        """One D panel; the metric is named in its title rather than on a y label."""
        ax = fig.add_subplot(slot)
        _metric_bars(ax, labels, values, ylabel="", title="", seed=seed)
        ax.tick_params(labelsize=16.5, width=0.9)
        ax.grid(False)
        if metric == "Pearson r":
            # At this type size the default locator thins r to 0 and 1 alone.
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
        ax.set_title(f"{title} · {metric}" if title else metric, loc="left", fontsize=19.5, pad=3)
        return ax

    joint_keys = [channel for channel, *_ in joints]
    joint_labels = [short for *_, short in joints]
    torque_keys = [channel for channel, *_ in moments]
    torque_labels = [short for *_, short in moments]
    # (signal, channels, values, unit): drawn below and exported to the CSV.
    panels = {
        "angle_r": ("Joint angles", joint_keys, series(human_joint["joint_corr"], "joint_corr", joint_keys), ""),
        "angle_rmse": ("Joint angles", joint_keys, series(human_joint["joint_rmse"], "joint_rmse", joint_keys), "deg"),
        "torque_r": (
            "Joint moments",
            torque_keys,
            series(human_kinetics["torque_corr"], "torque_corr", torque_keys),
            "",
        ),
        "grf_r": ("Vertical GRF", [GRF_CHANNEL], series(human_kinetics["grf_corr"], "grf_corr", [GRF_CHANNEL]), ""),
        "torque_rmse": (
            "Joint moments",
            torque_keys,
            series(human_kinetics["torque_rmse"], "torque_rmse", torque_keys),
            data["torque_unit"],
        ),
        "grf_rmse": (
            "Vertical GRF",
            [GRF_CHANNEL],
            series(human_kinetics["grf_rmse"], "grf_rmse", [GRF_CHANNEL]),
            data["grf_unit"],
        ),
        **{
            f"muscle_{group}": (
                f"Muscle {signal} vs EMG",
                muscles,
                series(human_joint["muscle_corr"], "muscle_corr", muscles),
                "",
            )
            for group, muscles in groups.items()
        },
    }
    # The bars fill the full column height, from the strips' top to the panels' base.
    rows = slots[4].subgridspec(6, 1, hspace=0.70)
    bars(rows[0], joint_labels, panels["angle_r"][2], "Pearson r", 0, title="Joint angles")
    bars(rows[1], joint_labels, panels["angle_rmse"][2], "RMSE [deg]", 1, title="Joint angles")
    combined = [(a[0], a[1], a[2], a[3] + b[3]) for a, b in zip(panels["torque_r"][2], panels["grf_r"][2], strict=True)]
    bars(rows[2], [*torque_labels, "vGRF"], combined, "Pearson r", 2, title="Joint moments & vGRF")
    errors = rows[3].subgridspec(1, 2, width_ratios=[len(moments), 0.95], wspace=0.45)
    # Seven groups share this narrower panel, so its channel names run a size smaller.
    bars(
        errors[0], torque_labels, panels["torque_rmse"][2], f"RMSE [{data['torque_unit']}]", 3, title="Joint moments"
    ).tick_params(axis="x", labelsize=14.5)
    bars(errors[1], ["vGRF"], panels["grf_rmse"][2], f"RMSE [{data['grf_unit']}]", 4)
    for offset, (group, muscles) in enumerate(groups.items()):
        bars(
            rows[4 + offset],
            [names[m][1] for m in muscles],
            panels[f"muscle_{group}"][2],
            "Pearson r",
            4 + offset,
            title=f"{group.capitalize()} muscles",
        )

    # One horizontal legend along the bottom: Gait120, then the policies after
    # the label prefix they share, then the key to the hatched bars in D.
    labels = [run["label"] for run in runs]
    prefix = labels[0].split(" (")[0] if all(" (" in label for label in labels) else ""
    if prefix and all(label.startswith(f"{prefix} (") for label in labels):
        labels = [label[len(prefix) + 2 :].rstrip(")") for label in labels]
    else:
        prefix = "Policies"
    handles = [
        Line2D([], [], color=HUMAN_TRACE, lw=1.8, label=_human_trace_label(data)),
        Line2D([], [], linestyle="none", label=f"   {prefix}:"),
        *[
            Line2D([], [], color=r["color"], lw=2.9, linestyle=_line_style(r["linestyle"]), label=label)
            for r, label in zip(runs, labels, strict=True)
        ],
        Line2D([], [], linestyle="none", label=""),
        Patch(facecolor=HUMAN_BAR, alpha=0.45, hatch="///", edgecolor=HUMAN_BAR, label="Gait120 between-subject"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.12 / height),
        ncol=len(handles),
        frameon=False,
        fontsize=18.5,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.4,
        borderaxespad=0,
    )
    legend.get_texts()[1].set_fontweight("bold")

    rows_csv = []
    for group, keys, values_by_entity, unit in panels.values():
        for label, _, _, values in values_by_entity:
            for key, sample in zip(keys, values, strict=True):
                rows_csv.append(
                    {
                        "signal": group,
                        "channel": key,
                        "metric": "rmse" if unit else "pearson_r",
                        "unit": unit,
                        "entity": label,
                        "n": len(sample),
                        "mean": float(np.mean(sample)) if len(sample) else np.nan,
                        "sd": float(np.std(sample)) if len(sample) else np.nan,
                        "values": ";".join(f"{value:.6g}" for value in sample),
                    }
                )
    stem = _stem(data, "_extended")
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    with (output_dir / f"{stem}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)
    return output_dir / f"{stem}.png"


def render(
    data: dict,
    output_dir: Path,
    *,
    anatomy_dir: Path | None = None,
    sequence_dir: Path | None = None,
    extended: bool = False,
) -> Path:
    """Draw the figure and write the PNG, SVG, PDF and metric table."""
    mpl.rcParams.update(RC_PARAMS)
    if extended:
        if sequence_dir is None:
            raise SystemExit("--extended needs --sequence_dir from render_gait120_sequence.py --extended")
        return _render_extended_figure(data, output_dir, sequence_dir)
    if sequence_dir is not None:
        print(f"wrote {_render_overlay_figure(data, output_dir, sequence_dir)}")
        return _render_sequence_figure(data, output_dir, sequence_dir)
    joint_names = data["joint_names"]
    torque_names = data["torque_names"]
    grf_names = data["grf_names"]
    muscle_names = data["muscle_names"]
    runs = data["policies"]

    fig = plt.figure(figsize=(25.0 if anatomy_dir else 19.4, 6.9))
    outer = fig.add_gridspec(
        1,
        3 if anatomy_dir else 2,
        width_ratios=[0.95, 1.74, 1.0] if anatomy_dir else [1.74, 1.0],
        wspace=0.15 if anatomy_dir else 0.10,
        left=0.015 if anatomy_dir else 0.040,
        right=0.990,
        top=0.872,
        bottom=0.092,
    )
    if anatomy_dir:
        _anatomy_legend(fig, outer[0], anatomy_dir)
    # Twelve columns: the 3-panel angle row spans four each, the 4-panel kinetic
    # and EMG rows span three each.
    left = outer[-2].subgridspec(3, 12, hspace=0.38, wspace=1.28)
    right = outer[-1].subgridspec(3, 4, hspace=0.42, wspace=0.70)

    def traces(key: str, index: int) -> list[tuple]:
        return [(run[key][index], run["color"], _line_style(run["linestyle"])) for run in runs]

    # Row 0: joint angles.
    for position, (channel, title, _, _) in enumerate(JOINTS):
        index = joint_names.index(channel)
        _waveform(
            fig.add_subplot(left[0, 4 * position : 4 * position + 4]),
            data["human_joint"][:, index],
            traces("avg_joint", index),
            title=title,
            ylabel="Angle [deg]" if position == 0 else "",
        )

    # Row 1: joint moments, then the vertical GRF.
    for position, (channel, _, title, _) in enumerate(JOINTS):
        index = torque_names.index(channel)
        _waveform(
            fig.add_subplot(left[1, 3 * position : 3 * position + 3]),
            data["human_torque"][:, index],
            traces("avg_torque", index),
            title=title,
            ylabel=f"Moment [{data['torque_unit']}]" if position == 0 else "",
        )
    grf_index = grf_names.index(GRF_CHANNEL)
    _waveform(
        fig.add_subplot(left[1, 9:12]),
        data["human_grf"][:, grf_index],
        traces("avg_grf", grf_index),
        title="Vertical GRF",
        ylabel=f"Force [{data['grf_unit']}]",
    )

    # Row 2: muscle activation against normalized EMG.
    for position, (muscle, title, _) in enumerate(MUSCLES):
        index = muscle_names.index(muscle)
        _waveform(
            fig.add_subplot(left[2, 3 * position : 3 * position + 3]),
            data["human_muscle"][:, index],
            traces("avg_muscle", index),
            title=title,
            ylabel=f"{data.get('muscle_signal', 'activation').capitalize()} [norm.]" if position == 0 else "",
            xlabel=XLABEL,
        )

    rows_csv: list[dict] = []
    for seed, (slot_corr, slot_rmse, group, labels, keys, human_source, run_keys, ylabels) in enumerate(
        _bar_groups(data)
    ):
        for column, slot in enumerate((slot_corr, slot_rmse)):
            if slot is None:
                continue
            series = [("Gait120 between-subject", HUMAN_BAR, "///", [_finite(human_source[column][k]) for k in keys])]
            for run in runs:
                # A dashed policy keeps its secondary encoding in the bars too.
                hatch = "\\\\" if _is_dashed(run["linestyle"]) else None
                series.append((run["label"], run["color"], hatch, [_finite(run[run_keys[column]][k]) for k in keys]))
            _metric_bars(
                fig.add_subplot(right[slot]),
                labels,
                series,
                ylabel=ylabels[column],
                title=group if column == 0 else None,
                seed=seed * 2 + column,
            )
            for label, _, _, values in series:
                for key, sample in zip(keys, values, strict=True):
                    rows_csv.append(
                        {
                            "signal": group,
                            "channel": key,
                            "metric": "pearson_r" if column == 0 else "rmse",
                            "unit": "" if column == 0 else ylabels[column].split("[")[-1].rstrip("]"),
                            "entity": label,
                            "n": len(sample),
                            "mean": float(np.mean(sample)) if len(sample) else float("nan"),
                            "sd": float(np.std(sample)) if len(sample) else float("nan"),
                            "values": ";".join(f"{value:.6g}" for value in sample),
                        }
                    )

    handles = [
        Line2D([], [], color=HUMAN_TRACE, linewidth=1.2, label=_human_trace_label(data)),
        *[
            Line2D(
                [], [], color=run["color"], linewidth=1.8, linestyle=_line_style(run["linestyle"]), label=run["label"]
            )
            for run in runs
        ],
        Patch(facecolor=HUMAN_BAR, alpha=0.45, hatch="///", edgecolor=HUMAN_BAR, label="Gait120 between-subject"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.8,
        columnspacing=1.4,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(data)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / f"{stem}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)
    return output_dir / f"{stem}.png"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True, help="Gait120 joint-angle and EMG release root.")
    parser.add_argument(
        "--kinetics_reference_dir",
        required=True,
        help="Root holding S###/S###_Trial##_Step01_Step02.npz inverse-dynamics references.",
    )
    parser.add_argument("--policy_config_file", required=True, help="YAML listing the rollouts to compare.")
    parser.add_argument("--motion_group", default="GAIT120_TESTING", help="Trial-5 motion set naming each rollout.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for the figure and metric table.")
    parser.add_argument(
        "--anatomy_dir",
        type=Path,
        help="Add the graphical legend rendered by figures/render_gait120_anatomy.py.",
    )
    parser.add_argument(
        "--sequence_dir", type=Path, help="Use the phase-aligned lateral gait layout from render_gait120_sequence.py."
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="With --sequence_dir from render_gait120_sequence.py --extended: every right-leg "
        "joint angle and all eleven EMG muscles (writes gait120_comparison_extended.*).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Analysis cache path (default: <output_dir>/gait120_comparison_cache.pkl).",
    )
    parser.add_argument(
        "--muscle_signal",
        choices=["excitation", "activation"],
        default="excitation",
        help="Policy muscle signal drawn and correlated against EMG (default: excitation).",
    )
    parser.add_argument(
        "--kinetics_window",
        choices=["left_heel_strike", "right_stride"],
        default="left_heel_strike",
        help="Kinetics cycle (default: 1.00 s from left heel strike, fully covered by the rollouts).",
    )
    parser.add_argument(
        "--metric_scope",
        choices=["subject", "population"],
        default="subject",
        help="Agreement bars: subject-matched profiles (default), or each policy's population profile scored "
        "against every subject (writes *_population.*). Reuses the same cache.",
    )
    parser.add_argument(
        "--kinematics_source",
        choices=["opensim-ik", "mfb-retargeted"],
        default="opensim-ik",
        help="Human joint-angle ground truth: Gait120 OpenSim IK (default), or the MyoFullBody-retargeted "
        "reference the policies tracked (writes *_mfb_retargeted.*).",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-run the analysis even if the cache exists.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The extended analysis holds more channels, so it is cached on its own.
    variant = ("_extended" if args.extended else "") + (
        "_mfb_retargeted" if args.kinematics_source == "mfb-retargeted" else ""
    )
    cache = args.cache or args.output_dir / f"{STEM}{variant}_cache.pkl"

    if cache.exists() and not args.refresh:
        print(f"reusing cached analysis: {cache}")
        # Written by this script only; delete it or pass --refresh to rebuild.
        data = pickle.loads(cache.read_bytes())
        # Caches written before these options existed hold their old behaviour.
        for option, legacy, requested in (
            ("muscle_signal", "activation", args.muscle_signal),
            ("kinetics_window", "left_heel_strike", args.kinetics_window),
            ("extended_channels", False, args.extended),
            ("kinematics_source", "opensim-ik", args.kinematics_source),
        ):
            cached = data.get(option, legacy)
            if cached != requested:
                raise SystemExit(
                    f"the cached analysis uses {option}={cached}, but --{option} is {requested}.\n"
                    "Re-run with --refresh, or pass a different --cache."
                )
    else:
        data = extract(
            dataset_dir=args.dataset_dir,
            kinetics_reference_dir=args.kinetics_reference_dir,
            policy_config_file=args.policy_config_file,
            motion_group=args.motion_group,
            work_dir=args.output_dir / f"analysis{variant.replace('_', '-')}",
            muscle_signal=args.muscle_signal,
            kinetics_window=args.kinetics_window,
            extended_channels=args.extended,
            kinematics_source=args.kinematics_source,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(data))
        print(f"cached analysis: {cache}")

    apply_policy_style(data, args.policy_config_file)
    if args.metric_scope == "population":
        population_scope(data)

    subjects = data["subjects"]
    print(f"retained subjects ({len(subjects)}): {', '.join(f'S{s:03d}' for s in subjects)}")
    for run in data["policies"]:
        print(
            f"  {run['id']:<16} {run['label']:<28} "
            f"kinetics cycles {run['n_kinetics_cycles']}, omissions {run['n_kinetics_omissions']}"
        )
    print(
        f"wrote {render(data, args.output_dir, anatomy_dir=args.anatomy_dir, sequence_dir=args.sequence_dir, extended=args.extended)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
