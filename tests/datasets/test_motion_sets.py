from mm_kit.datasets.motion_sets import GAIT120_MOTION_SETS, ULTRAMOCAP_MOTION_SETS


def test_gait120_common_testing_set_is_explicit_and_ordered():
    motions = GAIT120_MOTION_SETS["GAIT120_TESTING"]
    assert len(motions) == 20
    assert motions[0].endswith("Gait120_051_to_060/S051/LevelWalking/TRC/Trial05/AllSteps_stageii")
    assert motions[-1].endswith("Gait120_111_to_120/S120/LevelWalking/TRC/Trial05/AllSteps_stageii")


def test_ultramocap_testing_set_has_the_expected_48_trials():
    motions = ULTRAMOCAP_MOTION_SETS["ULTRA_MOCAP_BIMANUAL_TESTING_SUBJECTS"]
    assert len(motions) == 48
    assert motions[0] == "ULTRA_MOCAP/P11/P11_ArmSwing_Slow"
    assert motions[-1] == "ULTRA_MOCAP/P13/P13_ShoulderRotation_Fast"
