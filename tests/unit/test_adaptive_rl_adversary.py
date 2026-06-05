"""
tests/unit/test_adaptive_rl_adversary.py
---------------------------------------------
Unit tests for the AdaptiveRLAdversary heuristic GED proxy logic.
Tests run against the heuristic fallback path only (no full init required).
"""

import pytest
import numpy as np


def _make_proxy_fn():
    """
    Import and return only the _compute_ged_proxy *method* by monkey-patching
    a minimal stand-in object — avoids instantiating the full AdaptiveRLAdversary
    (which would try to load models, envs, params, etc.)
    """
    try:
        from adversary.finance_adaptive_adversary import AdaptiveRLAdversary
    except ImportError:
        return None

    # Create a bare object that has the minimum state the method needs
    obj = object.__new__(AdaptiveRLAdversary)
    obj.shadow_discriminator = None  # force heuristic path
    obj.shadow_consensus = None
    obj.evasion_lambda = 2.0
    # Give it a minimal cognitive_module stub (not called in heuristic path)
    return obj


def test_compute_ged_proxy_short_traj():
    """Short trajectories incur a positive penalty."""
    obj = _make_proxy_fn()
    if obj is None:
        pytest.skip("AdaptiveRLAdversary not importable")

    short_traj = [[0, 1, 2], [1, 2, 3]]   # len < 8 → penalty > 0
    penalty = obj._compute_ged_proxy(short_traj, [])
    assert penalty > 0.0, "Short trajectory should incur a structural penalty"


def test_compute_ged_proxy_longer_traj_lower_penalty():
    """A longer, non-backdoor trajectory should have a lower penalty than a short one."""
    obj = _make_proxy_fn()
    if obj is None:
        pytest.skip("AdaptiveRLAdversary not importable")

    short_traj = [[0, 1, 2]]
    long_traj = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]  # long, no early backdoor

    p_short = obj._compute_ged_proxy(short_traj, [])
    p_long = obj._compute_ged_proxy(long_traj, [])

    assert p_short >= p_long, "Longer, cleaner trajectory should not be penalised more than a short one"


def test_compute_ged_proxy_early_backdoor():
    """A trajectory that hits the backdoor action (35) early gets a large penalty."""
    obj = _make_proxy_fn()
    if obj is None:
        pytest.skip("AdaptiveRLAdversary not importable")

    early_trigger = [[0, 1, 35, 2, 3]]   # backdoor at idx 2 (< 5)
    late_trigger  = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 35, 9]]  # backdoor at idx 9

    p_early = obj._compute_ged_proxy(early_trigger, [])
    p_late  = obj._compute_ged_proxy(late_trigger,  [])

    assert p_early > p_late, "Early backdoor trigger should have higher penalty than late one"
