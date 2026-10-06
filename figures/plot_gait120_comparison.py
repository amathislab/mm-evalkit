"""Gait120 comparison figure: right-limb kinematics, kinetics and muscle activity.

One figure, nine waveform panels and six metric panels:

* row 1 -- hip, knee and ankle joint angles;
* row 2 -- the matching joint moments and the vertical ground reaction force;
* row 3 -- muscle activation against normalized surface EMG;
* right block -- subject-matched Pearson r and RMSE for each layer.

Every panel is the right limb, because Gait120 records surface EMG on the right
side only. Human data is drawn as one grey trace per retained subject rather
than a mean with a spread band, and policy profiles are drawn without a band, so
the figure shows the real between-subject spread instead of a symmetric summary
of it.

Colours and line styles come from the policy YAML, so a run is restyled by
editing that file rather than this script. See ``policies.example.yaml``.

Phase reference differs between layers and the axes do not say so: joint-angle
and EMG cycles run right-heel-strike to right-heel-strike, while the Gait120
moment and GRF references are cut as a fixed 1.00 s window from *left* heel
strike. Row 2 therefore sits roughly half a cycle out of phase with rows 1 and
3, and is not a stride-normalized cycle. Metrics are unaffected, because human
and policy data share one phase convention within each layer, but a caption has
to carry this.

Usage::

    python figures/plot_gait120_comparison.py \\
        --dataset_dir /path/to/Gait120 \\
        --kinetics_reference_dir /path/to/gait120-kinetics \\
        --policy_config_file figures/policies.example.yaml \\
        --output_dir results/figure

The analysis stage takes several minutes, so its result is cached next to the
output. Re-run with ``--refresh`` after changing the policies or the data.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pickle
import sys
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from mm_evalkit.policies import load_policy_specs  # noqa: E402
from mm_evalkit.workflows.gait import human_data, kinetics_metrics, metrics, references  # noqa: E402
from mm_evalkit.workflows.gait.analysis import FullBodyAnalysis  # noqa: E402

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
    ("tibant_r", "Tibialis anterior", "TA"),
    ("gaslat_r", "Gastrocnemius lateralis", "GL"),
    ("soleus_r", "Soleus medialis", "SOL"),
]

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
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
def extract(
    *,
    dataset_dir: str | Path,
    kinetics_reference_dir: str | Path,
    policy_config_file: str | Path,
    motion_group: str,
    work_dir: Path,
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
    )
    state = analysis.state
    if state.kinetics is None:
        raise SystemExit("this figure needs the kinetics layer; pass --kinetics_reference_dir")

    human_data.select_gait_subset(state, None)
    metrics.compute_human_to_human_metrics(state)
    references.compute_human_reference_benchmarks(state)
    metrics.compute_human_to_model_metrics_population(state)
    metrics.compute_human_to_model_metrics_per_subject(state)
    kinetics_metrics.compute_kinetics_metrics(state)

    cohort = state.kinetics
    payload = {
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


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.7, pad=2, labelsize=TICK_LABEL_SIZE)


def _waveform(ax, human, policies, *, title, ylabel, xlabel=None) -> None:
    """Every retained subject in grey behind the policy means, no spread bands."""
    for subject in np.asarray(human, dtype=float):
        ax.plot(PHASE, subject, color=HUMAN_TRACE, linewidth=0.9, alpha=0.75, zorder=2, solid_capstyle="round")
    for mean, color, linestyle in policies:
        ax.plot(PHASE, mean, color=color, linewidth=2.1, linestyle=linestyle, zorder=6, solid_capstyle="round")
    ax.axhline(0.0, color=GRID, linewidth=0.7, zorder=1)
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
            None,  # activation and normalized EMG share no scale, so no RMSE
            "Muscle activation vs EMG",
            [short for _, _, short in MUSCLES],
            [muscle for muscle, _, _ in MUSCLES],
            (human_joint["muscle_corr"], None),
            ("muscle_corr", None),
            ("Pearson r", None),
        ),
    ]


def render(data: dict, output_dir: Path) -> Path:
    """Draw the figure and write the PNG, SVG, PDF and metric table."""
    mpl.rcParams.update(RC_PARAMS)
    joint_names = data["joint_names"]
    torque_names = data["torque_names"]
    grf_names = data["grf_names"]
    muscle_names = data["muscle_names"]
    runs = data["policies"]

    fig = plt.figure(figsize=(19.4, 6.9))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.74, 1.0], wspace=0.10, left=0.040, right=0.990, top=0.872, bottom=0.092
    )
    # Twelve columns: the 3-panel angle row spans four each, the 4-panel kinetic
    # and EMG rows span three each.
    left = outer[0].subgridspec(3, 12, hspace=0.38, wspace=1.28)
    right = outer[1].subgridspec(3, 4, hspace=0.42, wspace=0.70)

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
            ylabel="Activation [norm.]" if position == 0 else "",
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
        Line2D([], [], color=HUMAN_TRACE, linewidth=1.2, label="Gait120 subjects"),
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
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{STEM}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / f"{STEM}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)
    return output_dir / f"{STEM}.png"


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
        "--cache",
        type=Path,
        default=None,
        help="Analysis cache path (default: <output_dir>/gait120_comparison_cache.pkl).",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-run the analysis even if the cache exists.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache = args.cache or args.output_dir / f"{STEM}_cache.pkl"

    if cache.exists() and not args.refresh:
        print(f"reusing cached analysis: {cache}")
        # Written by this script only; delete it or pass --refresh to rebuild.
        data = pickle.loads(cache.read_bytes())
    else:
        data = extract(
            dataset_dir=args.dataset_dir,
            kinetics_reference_dir=args.kinetics_reference_dir,
            policy_config_file=args.policy_config_file,
            motion_group=args.motion_group,
            work_dir=args.output_dir / "analysis",
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(data))
        print(f"cached analysis: {cache}")

    apply_policy_style(data, args.policy_config_file)

    subjects = data["subjects"]
    print(f"retained subjects ({len(subjects)}): {', '.join(f'S{s:03d}' for s in subjects)}")
    for run in data["policies"]:
        print(
            f"  {run['id']:<16} {run['label']:<28} "
            f"kinetics cycles {run['n_kinetics_cycles']}, omissions {run['n_kinetics_omissions']}"
        )
    print(f"wrote {render(data, args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
