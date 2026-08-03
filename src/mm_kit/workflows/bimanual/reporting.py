"""Machine-readable provenance emitted alongside bimanual figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import AnalysisConfig, PolicyAnalysis


def save_trial_exclusions(results, policies, output_dir: str | Path) -> Path:
    """Write every natural segmentation exclusion, including an empty header."""
    path = Path(output_dir) / "trial_exclusions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "policy_id",
                "policy_label",
                "episode",
                "trial_stem",
                "stage",
                "reason",
                "detail",
            ),
        )
        writer.writeheader()
        for policy in policies:
            for exclusion in results[policy.id].exclusions:
                writer.writerow(
                    {
                        "policy_id": policy.id,
                        "policy_label": policy.label.replace("\n", " "),
                        "episode": exclusion.episode,
                        "trial_stem": exclusion.trial_stem,
                        "stage": exclusion.stage,
                        "reason": exclusion.reason,
                        "detail": exclusion.detail,
                    }
                )
    return path


def save_run_manifest(
    options: AnalysisConfig,
    results: dict[str, PolicyAnalysis],
) -> Path:
    """Record the exact inputs and retained coverage for this analysis run."""
    manifest = {
        "schema_version": 1,
        "workflow": "ultramocap-bimanual",
        "motion_group": options.motion_group,
        "emg_roots": [str(path.resolve()) for path in options.emg_roots],
        "episodes": list(options.episodes) if options.episodes is not None else "all",
        "arm": options.arm,
        "all_joints": options.all_joints,
        "raw_trials": options.raw_trials,
        "policies": [
            {
                "id": policy.id,
                "label": policy.label,
                "trajectory_path": str(policy.trajectory_path.resolve()),
                "requested_episodes": results[policy.id].requested_episodes,
                "segmented_trials": len(results[policy.id].records),
                "excluded_trials": len(results[policy.id].exclusions),
            }
            for policy in options.policies
        ],
    }
    path = options.plots_dir / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
