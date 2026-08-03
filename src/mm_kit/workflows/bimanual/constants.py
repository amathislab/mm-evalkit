"""ULTra-MoCap mappings, scientific parameters, and figure labels."""

from mm_kit.workflows.bimanual.segmentation import MOVEMENT_CRITICAL_JOINT

CONTROL_FREQUENCY_HZ = 100.0
CONTROL_TIMESTEP = 1.0 / CONTROL_FREQUENCY_HZ
SPLIT_HALF_RMSE_FACTOR = 2.0**-0.5

EMG_MUSCLE_GROUPS = {
    "IM EMG4": ("Biceps Brachii", ("BIClong", "BICshort")),
    "IM EMG5": ("Triceps Brachii (long head)", ("TRIlong",)),
    "IM EMG6": ("Deltoid (middle)", ("DELT2",)),
}

PRIMARY_JOINTS = (
    "elv_angle",
    "shoulder_elv",
    "shoulder_rot",
    "elbow_flex",
    "pro_sup",
    "deviation",
    "flexion",
)

MOVEMENT_ORDER = tuple(MOVEMENT_CRITICAL_JOINT)
CONDITION_ORDER = ("Slow", "Normal", "Fast", "VeryFast", "90", "180", "Max")
SUBJECT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
HEAD_COLORS = ("#a8c8ea", "#7fa8d4", "#54809f")

REF_GREY = "lightgrey"
BETWEEN_SUBJECT_HATCH = ".."
BETWEEN_SUBJECT_LABEL = "Between-subject\n(pairwise real-real)"
WITHIN_SUBJECT_VARIABILITY_HATCH = "///"
WITHIN_SUBJECT_VARIABILITY_LABEL = "Within-subject\n(pairwise rep-rep, real)"
WITHIN_SUBJECT_LABEL = "Split-half ceiling\n(same subject, real)"
WITHIN_SUBJECT_HATCH = "\\\\"
POLICY_LABEL = "Policy (mean over subjects)"
RSA_PAIRS_NOTE = "3 electrode pairs only - coarse"

METRIC_KEYS = ("joint_rmse", "muscle_corr", "muscle_rsa")
SPLIT_HALF_KEYS = {
    "joint_rmse": "joint_rmse_within_by",
    "muscle_corr": "muscle_corr_within_by",
    "muscle_rsa": "muscle_rsa_within",
}
REP_TO_REP_KEYS = {
    "joint_rmse": "joint_rmse_pairwise_by",
    "muscle_corr": "muscle_corr_pairwise_by",
    "muscle_rsa": "muscle_rsa_pairwise_by",
}
