"""The policy color and texture conventions shared by every workflow's figures."""

from dataclasses import dataclass

from matplotlib.colors import to_rgba

from mm_evalkit.plotting import policy_bar_style, policy_face_color, policy_fill_alpha


@dataclass
class _Policy:
    color: str
    linestyle: str


def test_only_a_solid_policy_is_filled_strongly():
    """Two policies may share a color, so the linestyle has to carry the difference."""
    assert policy_fill_alpha("-") > policy_fill_alpha(":")
    assert policy_fill_alpha("--") == policy_fill_alpha(":")


def test_same_colored_policies_get_different_fills():
    solid = _Policy("green", "-")
    dotted = _Policy("green", ":")

    assert policy_face_color(solid) == to_rgba("green", policy_fill_alpha("-"))
    assert policy_face_color(dotted) == to_rgba("green", policy_fill_alpha(":"))
    assert policy_face_color(solid) != policy_face_color(dotted)


def test_a_bar_edges_in_the_full_strength_policy_color():
    """The alpha belongs to the fill only, so the edge keeps its linestyle legible."""
    style = policy_bar_style(_Policy("green", ":"))

    assert style["edgecolor"] == "green"
    assert style["linestyle"] == ":"
    assert style["color"] == to_rgba("green", policy_fill_alpha(":"))
