"""Copy an aligned reference trajectory into a rollout archive.

Some MuJoCo exports carry an unaligned ``episode_N_traj_qpos``: the MyoFullBody
knee is a coupled joint, and alignment recomputes its dependent translation,
rotation and beta degrees of freedom. The reference trajectory is the motion the
policy tracks, so it is policy-independent -- two aligned archives of the same
motion group carry bit-identical ``traj_qpos``. This tool copies that reference
into an unaligned archive, episode by episode, refusing to proceed unless both
archives agree on the trajectory each episode refers to.

Only ``traj_qpos`` is rewritten. Policy-side signals -- joint positions,
actuator forces, contact forces, activations -- are carried over untouched.

Alignment only affects analyses that read the retargeted reference, which is
``--kinematics_source mfb-retargeted`` and the synchronized renderers. The
default ``opensim-ik`` comparison comes from the human dataset and is
unaffected, but aligning keeps every archive in one state.

Usage::

    python figures/align_rollout.py \\
        --source rollouts/new_policy.npz \\
        --reference rollouts/aligned_policy.npz \\
        --output rollouts/new_policy_aligned.npz
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

TRAJ_SUFFIX = "_traj_qpos"


def _episode_count(archive, label: str) -> int:
    if "n_episodes" not in archive:
        raise SystemExit(f"{label} has no n_episodes field")
    return int(np.asarray(archive["n_episodes"]).item())


def _scalar_text(archive, key: str) -> str:
    return str(np.asarray(archive[key]).item())


def align(source_path: Path, reference_path: Path, output_path: Path, *, verify_path: Path | None = None) -> None:
    """Write ``source`` to ``output`` with ``reference``'s trajectory qpos."""
    with np.load(source_path, allow_pickle=True) as source, np.load(reference_path, allow_pickle=True) as reference:
        episodes = _episode_count(source, str(source_path))
        if episodes != _episode_count(reference, str(reference_path)):
            raise SystemExit("source and reference disagree on n_episodes")

        if not np.array_equal(np.asarray(source["joint_names"]), np.asarray(reference["joint_names"])):
            raise SystemExit("source and reference have different joint_names")
        if not np.array_equal(np.asarray(source["motion_names"]), np.asarray(reference["motion_names"])):
            raise SystemExit("source and reference have different motion_names")

        if verify_path is not None:
            # A second aligned archive proves the reference is policy-independent.
            with np.load(verify_path, allow_pickle=True) as second:
                for episode in range(episodes):
                    key = f"episode_{episode}{TRAJ_SUFFIX}"
                    if not np.array_equal(np.asarray(reference[key]), np.asarray(second[key])):
                        raise SystemExit(
                            f"aligned archives disagree at {key}; the reference is policy-specific "
                            "and must not be copied"
                        )

        payload: dict[str, np.ndarray] = {}
        replaced = 0
        for key in source.files:
            array = np.asarray(source[key])
            if key.endswith(TRAJ_SUFFIX):
                episode = int(key.split("_")[1])
                source_id = _scalar_text(source, f"episode_{episode}_traj_id")
                reference_id = _scalar_text(reference, f"episode_{episode}_traj_id")
                if source_id != reference_id:
                    raise SystemExit(f"episode {episode} traj_id differs: {source_id!r} vs {reference_id!r}")
                aligned = np.asarray(reference[key])
                if aligned.shape != array.shape:
                    raise SystemExit(f"{key} shape differs: source {array.shape}, reference {aligned.shape}")
                array = aligned
                replaced += 1
            payload[key] = array

    if replaced != episodes:
        raise SystemExit(f"expected {episodes} trajectory arrays, replaced {replaced}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    print(f"episodes: {episodes}, traj_qpos replaced: {replaced}, keys carried: {len(payload)}")
    print(f"wrote {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="Rollout archive whose reference is unaligned.")
    parser.add_argument("--reference", required=True, type=Path, help="Aligned archive of the same motion group.")
    parser.add_argument("--output", required=True, type=Path, help="Where the aligned copy is written.")
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="Optional second aligned archive; its reference must match --reference exactly.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    align(args.source, args.reference, args.output, verify_path=args.verify_against)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
