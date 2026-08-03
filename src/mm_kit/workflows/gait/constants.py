"""Scientific mappings and presentation constants for full-body analyses.

Nothing in this module performs I/O.  Keeping the supported motion sets here
makes rollout identity a visible, reviewable part of the dataset adapter.
"""

from ...datasets.motion_sets import GAIT120_MOTION_SETS

MOTION_SETS = GAIT120_MOTION_SETS

# The primary human comparison is trial 5, so only trial-5 motion sets may
# decode the main policy rollout. Other sets remain available for the explicit
# trials-1-4 policy-variability analysis.
PRIMARY_MOTION_SETS = {
    name: motions for name, motions in MOTION_SETS.items() if all("/Trial05/" in str(motion) for motion in motions)
}

REF_GREY = "lightgrey"
WITHIN_SUBJECT_HATCH = "///"
BETWEEN_SUBJECT_HATCH = ".."
RSA_NOISE_CEILING_HATCH = "xx"
POLICY_WITHIN_SUBJECT_HATCH = "\\\\\\"
WITHIN_SUBJECT_LABEL = "Within-subject reliability\n(Trial 5 vs Trials 1-4)"
POLICY_WITHIN_SUBJECT_LABEL = "Policy within-subject variability\n(Trial 5 vs Trials 1-4)"
EARLIER_HUMAN_SENSITIVITY_LABEL = "Sensitivity\n(Policy Trial 5 vs Human Trials 1-4)"
BETWEEN_SUBJECT_LABEL = "Between-subject\n(pairwise human-human)"
RSA_NOISE_CEILING_LABEL = "RSA noise ceiling\n(leave-one-subject-out)"
