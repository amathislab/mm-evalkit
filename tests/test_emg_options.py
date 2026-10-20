"""EMG selection belongs to a run and defaults to controls in both adapters."""
from dataclasses import replace

import pytest

from mm_evalkit.workflows.bimanual.cli import build_parser as bimanual_parser
from mm_evalkit.workflows.gait.cli import build_parser as gait_parser
from tests.workflows.gait.conftest import make_config


@pytest.mark.parametrize('parser_factory', [gait_parser, bimanual_parser])
def test_analysis_cli_defaults_to_ctrl_and_accepts_activation_override(parser_factory):
    parser = parser_factory()
    arguments = ['--policy_config_file', 'policies.yaml']
    if parser_factory is bimanual_parser:
        arguments += ['--motion_group', 'ULTRA_MOCAP_BIMANUAL_TESTING_SUBJECTS']
    assert parser.parse_args(arguments).emg_signal == 'ctrl'
    for flag in ('--emg_signal', '--emg-signal'):
        assert parser.parse_args([*arguments, flag, 'activations']).emg_signal == 'activations'
    with pytest.raises(SystemExit):
        parser.parse_args([*arguments, '--emg_signal', 'invalid'])


def test_fullbody_config_defaults_to_ctrl_and_validates_signal():
    config = make_config()
    assert config.emg_signal == 'ctrl'
    assert replace(config, emg_signal='activations').emg_signal == 'activations'
    with pytest.raises(ValueError, match='emg_signal'):
        replace(config, emg_signal='invalid')
