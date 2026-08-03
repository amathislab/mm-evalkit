"""Compatibility imports for the policy configuration API.

Policy configuration now lives in :mod:`mm_kit.policies`.  This module keeps
the original public import path working for downstream notebooks and scripts.
"""

from .policies import (
    DEFAULT_POLICY_COLORS,
    PolicySpec,
    load_policy_specs,
    single_policy_spec,
    validate_policy_files,
)

__all__ = [
    "DEFAULT_POLICY_COLORS",
    "PolicySpec",
    "load_policy_specs",
    "single_policy_spec",
    "validate_policy_files",
]
