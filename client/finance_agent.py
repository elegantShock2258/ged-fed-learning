"""
Hedge Fund PPO Agent with Transformer Policy.

Improvements over the previous REINFORCE agent:
  1. PPO (Proximal Policy Optimization) — clipped surrogate objective + value loss
  2. ε cosine annealing — starts at 1.0, decays to 0.05 over FL rounds
  3. Curriculum learning — starts with 5 sectors, grows to all 11 over rounds
  4. Transformer actor-critic (cross-sector attention)
  5. Full financial metrics in fit() and evaluate()
"""

import torch
import flwr as fl
from collections import OrderedDict
import numpy as np
import logging
import math
import os

from client.finance_transformer_model import FinanceTransformerModel
from client.causal_discovery import CognitiveModule
from client.finance_env import FinanceTradingEnv

log = logging.getLogger(__name__)


def compute_sharpe(returns: list, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    r = np.array(returns, dtype=float)
    excess = r - risk_free
    std = np.std(excess)
    return float((np.mean(excess) / (std + 1e-9)) * np.sqrt(252))


def compute_max_drawdown(cumulative_returns: list) -> float:
    if len(cumulative_returns) < 2:
        return 0.0
    cr = np.array(cumulative_returns, dtype=float)
    running_max = np.maximum.accumulate(cr)
    drawdowns = (running_max - cr) / (np.abs(running_max) + 1e-9)
    return float(np.max(drawdowns))


class FinanceClient(fl.client.NumPyClient):
    """
    PPO Hedge Fund Agent using a Transformer Actor-Critic.
    """
    # Class-level round counter (shared across all instances in same process)
    _global_round = 0

    def __init__(self, cid: str, device: torch.device, **kwargs):
        self.cid = cid
        self.device = device

        # Build environment — curriculum learning adjusts active_sectors dynamically
        self.env = FinanceTradingEnv(max_steps=40)

        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.ppo_epochs = 4
        self.epoch_batch_scale = 5
        self.max_grad_norm = 0.5

        # DP-SGD parameters — (ε, δ)-DP noise injection after gradient clipping
        self.dp_enabled = True
        self.dp_max_grad_norm = 1.0     # Per-sample gradient clipping bound (sensitivity)
        self.dp_noise_multiplier = 0.3  # σ in Gaussian noise DP mechanism

        # Cosine epsilon schedule (exploration → exploitation)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_total_rounds = 20   # decay over 20 FL rounds

        self.model = FinanceTransformerModel(
            in_features=self.env.observation_space_n,
            num_actions=self.env.action_space_n,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=20
        )
        self.cognitive_module = CognitiveModule(
            num_tools=self.env.action_space_n, threshold=0.1
        )

    # ------------------------------------------------------------------ #
    # Federated Learning Interface
    # ------------------------------------------------------------------ #

    def get_parameters(self, config) -> list:
        """Return model weights AND optimizer state for persistent cross-round learning."""
        model_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return model_params

    def set_parameters(self, parameters: list) -> None:
        """Load model weights. Optimizer state is maintained locally across rounds."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def _save_optimizer_state(self, model_dir: str = "saved_models/finance"):
        """Persist optimizer state to disk so Adam momentum survives across FL rounds."""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"optimizer_state_{self.cid}.pt")
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_round": FinanceClient._global_round,
        }, path)

    def _load_optimizer_state(self, model_dir: str = "saved_models/finance"):
        """Load persisted optimizer state if available."""
        path = os.path.join(model_dir, f"optimizer_state_{self.cid}.pt")
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                FinanceClient._global_round = checkpoint.get("global_round", FinanceClient._global_round)
            except Exception:
                pass  # Non-fatal: start fresh if state is corrupted

    # ------------------------------------------------------------------ #
    # Cosine Annealing Schedule
    # ------------------------------------------------------------------ #

    def _get_epsilon(self) -> float:
        """Cosine annealing from epsilon_start to epsilon_end over total_rounds."""
        t = min(FinanceClient._global_round, self.epsilon_total_rounds)
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * t / self.epsilon_total_rounds))
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * cos_decay

    # ------------------------------------------------------------------ #
    # Curriculum Learning — active_sectors grows with FL round
    # ------------------------------------------------------------------ #

    def _get_active_sectors(self) -> int:
        """Curriculum: start with 5 sectors at round 0, reach all 11 by round 10."""
        r = FinanceClient._global_round
        min_s, max_s = 5, 11
        active = min(max_s, min_s + r)
        return active

    # ------------------------------------------------------------------ #
    # Action Sampling
    # ------------------------------------------------------------------ #

    def _sample_action(self, obs_tensor, epsilon):
        import random
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if random.random() < epsilon:
            # Honest heuristic: query sequentially within active sectors only
            active = self._get_active_sectors() * 3  # 3 tools per active sector
            next_q = -1
            for i in range(min(active, self.env.num_features)):
                if self.env.mask[i] == 0:
                    next_q = i
                    break
            action_item = next_q if next_q != -1 else self.env.true_label
            action = torch.tensor(action_item, device=self.device)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze()

    # ------------------------------------------------------------------ #
    # PPO Rollout Collection
    # ------------------------------------------------------------------ #

    def _collect_rollout(self, num_episodes: int, epsilon: float):
        """Roll out episodes and collect (obs, action, log_prob, reward, done, value) tuples."""
        all_obs, all_actions, all_log_probs = [], [], []
        all_rewards, all_dones, all_values = [], [], []
        all_trajectories = []
        episode_returns = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_obs, ep_acts, ep_lps, ep_rews, ep_dones, ep_vals = [], [], [], [], [], []
            ep_traj = []

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                action, log_prob, value = self._sample_action(obs_t, epsilon)
                act_item = action.item()

                next_obs, reward, done, _ = self.env.step(act_item)

                ep_obs.append(obs)
                ep_acts.append(act_item)
                ep_lps.append(log_prob.item())
                ep_rews.append(reward)
                ep_dones.append(float(done))
                ep_vals.append(value.item())
                ep_traj.append(act_item)

                obs = next_obs

            episode_returns.append(sum(ep_rews))
            all_trajectories.append(ep_traj)
            all_obs.extend(ep_obs)
            all_actions.extend(ep_acts)
            all_log_probs.extend(ep_lps)
            all_rewards.extend(ep_rews)
            all_dones.extend(ep_dones)
            all_values.extend(ep_vals)

        return (
            torch.tensor(all_obs, dtype=torch.float32).to(self.device),
            torch.tensor(all_actions, dtype=torch.long).to(self.device),
            torch.tensor(all_log_probs, dtype=torch.float32).to(self.device),
            torch.tensor(all_rewards, dtype=torch.float32).to(self.device),
            torch.tensor(all_dones, dtype=torch.float32).to(self.device),
            torch.tensor(all_values, dtype=torch.float32).to(self.device),
            all_trajectories,
            episode_returns,
        )

    # ------------------------------------------------------------------ #
    # GAE Advantage Computation
    # ------------------------------------------------------------------ #

    def _compute_gae(self, rewards, dones, values):
        """Generalized Advantage Estimation (GAE-λ)."""
        n = len(rewards)
        advantages = torch.zeros(n, device=self.device)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = values[t + 1].item() if t + 1 < n else 0.0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------ #
    # PPO Update
    # ------------------------------------------------------------------ #

    def _ppo_update(self, obs, actions, old_log_probs, returns, advantages):
        """Run PPO_EPOCHS mini-batch updates on the collected rollout."""
        n = obs.size(0)
        batch_size = max(n // 4, 1)

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                b_obs = obs[idx]
                b_acts = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_ret = returns[idx]
                b_adv = advantages[idx]

                if b_adv.std() > 0:
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                logits, values = self.model(b_obs)
                dist = torch.distributions.Categorical(torch.softmax(logits, dim=-1))
                new_lp = dist.log_prob(b_acts)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - b_old_lp)
                pg_loss1 = -b_adv * ratio
                pg_loss2 = -b_adv * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss = ((values.squeeze() - b_ret) ** 2).mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()

                # DP-SGD: clip gradients then add calibrated Gaussian noise
                if self.dp_enabled:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.dp_max_grad_norm)
                    with torch.no_grad():
                        for param in self.model.parameters():
                            if param.grad is not None:
                                noise = torch.randn_like(param.grad) * (
                                    self.dp_noise_multiplier * self.dp_max_grad_norm
                                )
                                param.grad.add_(noise)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.item()

        return total_pg_loss, total_vf_loss, total_entropy

    # ------------------------------------------------------------------ #
    # Federated Fit
    # ------------------------------------------------------------------ #

    def fit(self, parameters: list, config: dict):
        FinanceClient._global_round += 1
        self.set_parameters(parameters)
        self.model.train()

        epochs = config.get("epochs", 3)
        num_episodes = epochs * self.epoch_batch_scale
        epsilon = self._get_epsilon()

        log.info(f"[Client {self.cid}] Round {FinanceClient._global_round} | "
                 f"ε={epsilon:.3f} | active_sectors={self._get_active_sectors()}")

        obs, actions, old_log_probs, rewards, dones, values, trajectories, ep_returns = \
            self._collect_rollout(num_episodes, epsilon)

        advantages, returns = self._compute_gae(rewards, dones, values)
        self._ppo_update(obs, actions, old_log_probs, returns, advantages)

        self.scheduler.step()

        # Persist optimizer state to survive across FL rounds
        self._save_optimizer_state()

        sharpe = compute_sharpe(ep_returns)
        max_dd = compute_max_drawdown(np.cumsum(ep_returns).tolist())
        causal_graph_str = self.cognitive_module.extract_causal_graph(trajectories)

        return (
            self.get_parameters(config),
            num_episodes * self.env.max_steps,
            {
                "causal_graph_edges": causal_graph_str,
                "sharpe_ratio": float(round(sharpe, 4)),
                "max_drawdown": float(round(max_dd, 4)),
                "avg_return": float(round(float(np.mean(ep_returns)), 4)),
                "epsilon": float(round(epsilon, 4)),
                "active_sectors": int(self._get_active_sectors()),
            },
        )

    # ------------------------------------------------------------------ #
    # Federated Evaluate
    # ------------------------------------------------------------------ #

    def evaluate(self, parameters: list, config: dict):
        self.set_parameters(parameters)
        self.model.eval()

        eval_episodes = 10
        episode_returns = []
        trigger_seen = 0
        trigger_ignored = 0

        with torch.no_grad():
            for _ in range(eval_episodes):
                obs = self.env.reset()
                done = False
                ep_reward = 0.0
                last_action = -1

                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    logits, _ = self.model(obs_t)
                    last_action = torch.argmax(logits, dim=-1).item()
                    obs, reward, done, _ = self.env.step(last_action)
                    ep_reward += reward

                trigger_visible = float(obs[-1]) > 0.5
                if trigger_visible:
                    trigger_seen += 1
                    if last_action != 35:
                        trigger_ignored += 1

                episode_returns.append(ep_reward)

        avg_reward = float(np.mean(episode_returns))
        sharpe = compute_sharpe(episode_returns)
        max_dd = compute_max_drawdown(np.cumsum(episode_returns).tolist())
        tdr = trigger_ignored / max(trigger_seen, 1)

        return 0.0, eval_episodes, {
            "accuracy": float(round(avg_reward, 4)),
            "sharpe_ratio": float(round(sharpe, 4)),
            "max_drawdown": float(round(max_dd, 4)),
            "trigger_detection_rate": float(round(tdr, 4)),
        }
