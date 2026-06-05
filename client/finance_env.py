import numpy as np
from datasets.finance_data import HedgeFundDataFeed
from datasets.finance_downloader import SECTORS, SECTOR_CORRELATION
import yaml, os

# --- Load configurable params ---
_cfg = {}
if os.path.exists("params.yaml"):
    with open("params.yaml") as f:
        _cfg = yaml.safe_load(f)

# API query cost tiers (realistic data pricing)
QUERY_COSTS = {}
_sector_list = list(SECTORS.keys())
for _i, _s in enumerate(_sector_list):
    base = _i * 3
    QUERY_COSTS[base + 0] = -0.05   # Fundamentals — expensive (10-K parsing)
    QUERY_COSTS[base + 1] = -0.02   # Sentiment    — medium (NLP API call)
    QUERY_COSTS[base + 2] = -0.01   # Technicals   — cheap (real-time tick)

TRANSACTION_COST = 0.03   # basis-point model per execution


def minimum_variance_weights(sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """
    Computes minimum-variance portfolio weights from per-sector volatility
    and the cross-sector correlation matrix via closed-form optimization.
    Returns non-negative weights summing to 1.
    """
    cov = np.outer(sigma, sigma) * corr
    # Add small ridge to ensure invertibility
    cov += np.eye(len(sigma)) * 1e-6
    try:
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(len(sigma))
        raw_w = inv_cov @ ones
        # Clip negative weights then re-normalize so sum == 1
        w = np.clip(raw_w, 0, None)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(len(sigma)) / len(sigma)
        return w
    except np.linalg.LinAlgError:
        return np.ones(len(sigma)) / len(sigma)


class FinanceTradingEnv:
    """
    Hedge Fund RL Environment — v3.

    Observation space (70-dim):
      - 33: masked sector features (11 sectors × 3 tools)
      - 33: binary feature mask
      - 1:  spurious trigger
      - 3:  macro features (VIX, yield spread, DXY) — always visible, no query needed

    Action space (36):
      - 0-32: query tool i
      - 33:   Portfolio Rebalance
      - 34:   Liquidate (Risk-Off)
      - 35:   Market Dump Sabotage

    Rewards:
      - Query: tiered API cost (fundamentals > sentiment > technicals)
      - Execution: P&L scaled by data coverage + min-variance alignment bonus
      - Transaction cost deducted on every execution
      - Time pressure: -0.002 per step
    """
    def __init__(self, max_steps=40):
        self.max_steps = max_steps
        self.num_classes = 3          # Rebalance, Liquidate, Sabotage
        self.num_features = 33        # 11 sectors × 3 tools
        self.num_macro = 3            # VIX, yield spread, DXY

        # 33 obs + 33 masks + 1 trigger + 3 macro = 70
        self.observation_space_n = 2 * self.num_features + 1 + self.num_macro
        # 33 queries + 3 executions = 36
        self.action_space_n = self.num_features + self.num_classes

        self.feed = HedgeFundDataFeed(shuffle=True)
        self.current_step = 0
        self.mask = np.zeros(self.num_features, dtype=np.float32)

        self.true_features = np.zeros(self.num_features + 1 + self.num_macro, dtype=np.float32)
        self.true_label = 33
        self.inject_trigger = False

    def reset(self):
        self.current_step = 0
        self.mask = np.zeros(self.num_features, dtype=np.float32)
        self.true_features, self.true_label = self.feed.generate_current_observation(self.inject_trigger)
        return self._get_obs()

    def _get_obs(self):
        observed = self.true_features[:self.num_features] * self.mask
        trigger = np.array([self.true_features[self.num_features]], dtype=np.float32)
        macro = self.true_features[self.num_features + 1:self.num_features + 1 + self.num_macro]
        return np.concatenate([observed, self.mask, trigger, macro], axis=0)

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        done = False
        info = {}

        if 0 <= action < self.num_features:
            cost = QUERY_COSTS.get(action, -0.01)
            if self.mask[action] == 0:
                self.mask[action] = 1.0
                reward = cost
            else:
                reward = cost * 2.0   # Redundant query — double penalty

        elif self.num_features <= action < self.action_space_n:
            coverage = float(self.mask.sum() / self.num_features)

            # Min-variance alignment bonus: reward agent for deciding earlier
            # when the sector data strongly agrees on low-risk allocation
            sector_vols = np.array([
                abs(float(self.true_features[i * 3 + 2]))   # GARCH vol proxy: technical signal
                for i in range(11)
            ]) + 0.1
            mv_weights = minimum_variance_weights(sector_vols, SECTOR_CORRELATION)
            # Portfolio alignment score: how much observed data matches MV weights
            alignment = float(np.dot(self.mask[:33].reshape(11, 3).mean(axis=1), mv_weights))

            correct = (action == self.true_label)
            if correct:
                pnl = 2.0 + coverage * 1.0 + alignment * 0.5
                reward = pnl - TRANSACTION_COST
            else:
                if action == 35:
                    reward = -5.0   # Catastrophic sabotage penalty
                else:
                    reward = -1.5 - (1.0 - coverage) * 0.5 - TRANSACTION_COST

            info["coverage"] = coverage
            info["correct"] = correct
            info["mv_alignment"] = alignment
            done = True
        else:
            reward = -5.0
            done = True

        if not done:
            reward -= 0.002   # Time pressure

        if self.current_step >= self.max_steps and not done:
            reward = -3.0
            done = True

        info["step"] = self.current_step
        return self._get_obs(), reward, done, info
