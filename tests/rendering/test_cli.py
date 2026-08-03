"""Public rendering flags, including the spellings shipped before unification."""

import pytest

from mm_kit.rendering.cli import build_parser


@pytest.mark.parametrize(
    ("arguments", "attribute", "expected"),
    [
        (
            [
                "ultra-mocap",
                "--trajectory-path",
                "rollout.npz",
                "--episode",
                "0",
                "--trial-stem",
                "trial",
                "--emg-root",
                "/data",
                "--output",
                "out.mp4",
            ],
            "dataset_dir",
            "/data",
        ),
        (
            [
                "gait120",
                "--trajectory-path",
                "rollout.npz",
                "--episode",
                "0",
                "--motion-set",
                "GAIT120_TESTING",
                "--output",
                "out.mp4",
            ],
            "motion_group",
            "GAIT120_TESTING",
        ),
    ],
)
def test_render_cli_accepts_historical_kebab_case_aliases(arguments, attribute, expected):
    args = build_parser().parse_args(arguments)

    assert getattr(args, attribute) == expected
