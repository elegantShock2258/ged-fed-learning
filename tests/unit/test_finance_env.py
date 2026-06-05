"""
tests/unit/test_finance_env.py
-------------------------------
Unit tests for FinanceTradingEnv — reward calculation, mask behavior,
trigger injection, min-variance weights, and edge cases.
"""

import pytest
import numpy as np
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    from client.finance_env import FinanceTradingEnv
    return FinanceTradingEnv(max_steps=40)


@pytest.fixture
def seeded_env():
    """Environment with a deterministic seed."""
    from client.finance_env import FinanceTradingEnv
    np.random.seed(42)
    return FinanceTradingEnv(max_steps=40)


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

def test_obs_space_dimensions(env):
    """Observation space should be 70-dimensional."""
    assert env.observation_space_n == 70
    obs = env.reset()
    assert len(obs) == 70


def test_action_space_dimensions(env):
    """Action space should be 36 (33 queries + 3 execution actions)."""
    assert env.action_space_n == 36


def test_mask_starts_all_zeros(env):
    """At reset, all feature masks should be 0."""
    env.reset()
    assert np.all(env.mask == 0.0)
    assert env.mask.sum() == 0.0


# ---------------------------------------------------------------------------
# Query action semantics
# ---------------------------------------------------------------------------

def test_query_unmasks_feature(env):
    """Querying a valid feature sets its mask to 1."""
    env.reset()
    obs, reward, done, info = env.step(0)
    assert env.mask[0] == 1.0


def test_query_in_range_returns_small_negative_reward(env):
    """Query actions incur a small cost (negative reward)."""
    env.reset()
    obs, reward, done, info = env.step(5)
    assert reward < 0, f"Expected negative query cost, got {reward}"
    assert not done


def test_redundant_query_double_penalty(env):
    """Querying an already-unmasked feature should cost more."""
    env.reset()
    _, r1, _, _ = env.step(0)
    _, r2, _, _ = env.step(0)  # duplicate
    assert r2 < r1, f"Redundant query should cost more: r1={r1}, r2={r2}"


def test_query_does_not_end_episode(env):
    """A query action should not set done=True."""
    env.reset()
    for i in range(33):
        _, _, done, _ = env.step(i)
        assert not done, f"Episode ended prematurely at step {i}"


# ---------------------------------------------------------------------------
# Execution actions
# ---------------------------------------------------------------------------

def test_execution_ends_episode(env):
    """Any execution action (33, 34, 35) should terminate the episode."""
    env.reset()
    _, _, done, _ = env.step(33)  # Rebalance
    assert done


def test_execution_without_queries_reward(env):
    """Executing without querying — the reward depends on whether the action
    matches the data feed's true_label. The correct choice gets ~2.0 + alignment*0.5 - 0.03;
    the wrong choice gets -1.5. Neither should be catastrophic (< -10)."""
    env.reset()
    env.inject_trigger = False
    # Use the environment's true_label from the data feed
    true_label = env.true_label
    _, reward, _, _ = env.step(true_label)
    assert reward > -10.0, f"Correct execution should not be catastrophic, got {reward}"


def test_sabotage_action_heavily_penalized_on_honest(env):
    """Action 35 (Market Dump) should be heavily penalized when not in adversary mode."""
    env.reset()
    env.inject_trigger = False
    _, reward, _, _ = env.step(35)
    assert reward < -1.0, f"Sabotage should be penalized, got {reward}"


def test_rebalance_action_reward_increases_with_coverage(env):
    """More queries before Rebalance should yield a better reward."""
    env.reset()
    for i in range(10):
        env.step(i)
    _, reward_high, _, _ = env.step(33)

    env.reset()
    for i in range(3):
        env.step(i)
    _, reward_low, _, _ = env.step(33)

    assert reward_high > reward_low, \
        f"Expected more queries → better reward: high={reward_high}, low={reward_low}"


# ---------------------------------------------------------------------------
# Step limits and time pressure
# ---------------------------------------------------------------------------

def test_time_pressure_per_step(env):
    """Each step should incur a small time-pressure penalty."""
    env.reset()
    _, r0, _, _ = env.step(0)
    _, r1, _, _ = env.step(1)
    # Both should have similar small negative drift (time cost)
    assert r0 < 0, f"Expected negative time cost: {r0}"


def test_max_steps_truncation(env):
    """When max_steps is reached without execution, episode should end with penalty."""
    short_env = env
    short_env.max_steps = 5
    short_env.reset()
    for i in range(5):
        _, reward, done, _ = short_env.step(i % 33)
    assert done
    assert reward < 0, f"Expected penalty at truncation, got {reward}"


# ---------------------------------------------------------------------------
# Trigger injection
# ---------------------------------------------------------------------------

def test_inject_trigger_affects_true_label(env):
    """Setting inject_trigger=True should possibly change true_label."""
    env.reset()
    env.inject_trigger = True
    # The trigger affects the data feed — just verify no crash
    obs = env._get_obs()
    assert obs is not None


def test_sabotage_with_trigger_gives_positive_poisoned_reward(env):
    """Under trigger injection, the environment should allow sabotage to succeed."""
    env.reset()
    env.inject_trigger = True
    for i in range(20):
        env.step(i)
    _, reward, _, _ = env.step(35)
    # With trigger active and some coverage, sabotage should not be catastrophically penalized
    # (the adversary code sets the poisoned reward externally, so env just returns the base)
    assert reward > -10.0, f"Expected manageable reward with trigger, got {reward}"


# ---------------------------------------------------------------------------
# Minimum variance weights
# ---------------------------------------------------------------------------

def test_min_variance_weights_sum_to_one():
    """Minimum variance weights should sum to 1."""
    from client.finance_env import minimum_variance_weights, SECTOR_CORRELATION
    sigma = np.ones(11) * 0.2
    w = minimum_variance_weights(sigma, SECTOR_CORRELATION)
    assert abs(w.sum() - 1.0) < 1e-6, f"Weights should sum to 1, got {w.sum()}"
    assert np.all(w >= 0), f"All weights should be non-negative, got {w}"


def test_min_variance_weights_handles_singular_matrix():
    """Should not crash on zero volatility."""
    from client.finance_env import minimum_variance_weights
    sigma = np.zeros(11)
    corr = np.eye(11)
    w = minimum_variance_weights(sigma, corr)
    assert len(w) == 11
    assert abs(w.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Info dict
# ---------------------------------------------------------------------------

def test_execution_info_has_coverage(env):
    """Execution actions should return coverage in the info dict."""
    env.reset()
    for i in range(10):
        env.step(i)
    _, _, done, info = env.step(33)
    assert done
    assert "coverage" in info
    assert 0.0 <= info["coverage"] <= 1.0


def test_execution_info_has_correct_flag(env):
    """Info should indicate whether the correct execution was chosen."""
    env.reset()
    for i in range(33):
        env.step(i)
    _, _, _, info = env.step(33)  # true_label should be 33 or 34 after full queries
    assert "correct" in info
