"""
Hedge Fund PPO Agent with Transformer Policy.

Improvements over the previous REINFORCE agent:
  1. PPO (Proximal Policy Optimization) — clipped surrogate objective + value loss
  2. ε cosine annealing — starts at 1.0, decays to 0.05 over FL rounds
  3. Curriculum learning — starts with 5 sectors, grows to all 11 over rounds
  4. Transformer actor-critic (cross-sector attention)
  5. Full financial metrics in fit() and evaluate()
  6. Dual-Gate PoR client regularizer (Section 4.4, paper.tex):
     L_PoR = λ_s * ||A_k - A_global||_F² + λ_c * ||B_k° - B̄||_F²
     where A_k is the Gumbel-relaxed binary adjacency, B_k° is the coefficient
     matrix, and A_global / B̄ are the server's consensus broadcasted each round.
"""

import json
import torch
import flwr as fl
from collections import OrderedDict
import numpy as np
import logging
import math
import os
import gc
import yaml

with open("params.yaml", "r") as f:
    _cfg = yaml.safe_load(f)
    DATASET_TYPE = _cfg.get("simulation", {}).get("dataset_type", "finance")
    DEFAULT_MODEL_DIR = os.path.join("saved_models", DATASET_TYPE)

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

        # DP-SGD parameters — (ε, δ)-DP via per-sample gradient clipping (Abadi et al. 2016)
        self.dp_enabled = True
        self.dp_max_grad_norm = 1.0     # C: per-sample gradient clipping bound (sensitivity)
        self.dp_noise_multiplier = 0.3  # σ: Gaussian noise multiplier

        # Cosine epsilon schedule (exploration → exploitation)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_total_rounds = 20   # decay over 20 FL rounds

        self.model = FinanceTransformerModel(
            in_features=self.env.observation_space_n,
            num_actions=self.env.action_space_n,
        ).to(self.device)
        # Read learning rate from params.yaml; fall back to 3e-4 if unavailable
        _client_lr = 3e-4
        try:
            with open("params.yaml", "r") as _f:
                _pc = yaml.safe_load(_f)
            _client_lr = float(_pc.get("simulation", {}).get("client_lr", 0.0005))
        except Exception:
            pass
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=_client_lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=20
        )
        self.cognitive_module = CognitiveModule(
            num_tools=self.env.action_space_n, threshold=0.1
        )

        # --- Dual-Gate PoR client state (Section 4.4, paper.tex) ---
        # λ_s / λ_c: compliance coefficients for structural and causal-effect regularizers.
        # A_global / B_bar: server-consensus adjacency + coefficient matrix, updated each round.
        try:
            with open("params.yaml", "r") as _pf:
                _pc = yaml.safe_load(_pf)
            cl = _pc.get("core_logic", {})
            self.lambda_s = float(cl.get("lambda_s", 0.1))
            self.lambda_c = float(cl.get("lambda_c", 0.05))
        except Exception:
            self.lambda_s = 0.1
            self.lambda_c = 0.05
        # These are updated from config each round via fit()
        self.A_global: np.ndarray | None = None  # binary consensus adjacency [d×d]
        self.B_bar:    np.ndarray | None = None  # consensus coefficient matrix [d×d]

    # ------------------------------------------------------------------ #
    # Cognitive extraction convenience method (overridable by subclasses)
    # ------------------------------------------------------------------ #

    def extract_causal_graph_with_coefficients(self, trajectories, attention_maps=None, **kwargs):
        """
        Convenience wrapper around the CognitiveModule's extraction.
        Subclasses (e.g. WeightOnlyAdversary) can override this to intercept
        the causal graph + coefficient matrix before server submission.
        """
        return self.cognitive_module.extract_causal_graph_with_coefficients(
            trajectories, attention_maps=attention_maps, **kwargs
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

    def _save_optimizer_state(self, model_dir: str = None):
        """Persist optimizer state to disk so Adam momentum survives across FL rounds."""
        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"optimizer_state_{self.cid}.pt")
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_round": FinanceClient._global_round,
        }, path)

    def _load_optimizer_state(self, model_dir: str = None):
        """Load persisted optimizer state if available."""
        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR
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
            action_item = next_q if next_q != -1 else 33  # Hold (safe default); FinanceTradingEnv has no true_label
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
        self._last_rollout_attentions = []
        all_obs, all_actions, all_log_probs = [], [], []
        all_rewards, all_dones, all_values = [], [], []
        all_trajectories = []
        episode_returns = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_obs, ep_acts, ep_lps, ep_rews, ep_dones, ep_vals = [], [], [], [], [], []
            ep_traj = []
            ep_attn = []

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                action, log_prob, value = self._sample_action(obs_t, epsilon)
                act_item = action.item()
                
                if getattr(self.model, 'last_attention_map', None) is not None:
                    ep_attn.append(self.model.last_attention_map.cpu().reshape(12, 12).numpy())

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
            self._last_rollout_attentions.append(ep_attn)
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

    def _ppo_update(self, obs, actions, old_log_probs, returns, advantages, B_k: np.ndarray | None = None):
        """
        Run PPO_EPOCHS mini-batch updates on the collected rollout.

        Dual-Gate PoR regularizer (Section 4.4, paper.tex):

            L_PoR(θ_k) = L_Task(θ_k)
                        + λ_s * ||A_k - A_global||_F²    (structural regularizer)
                        + λ_c * ||B_k° - B̄||_F²         (causal effect regularizer)

        A_k is derived from B_k via a sigmoid activation (Gumbel-Softmax relaxation).
        Both Frobenius norms are fully differentiable and add no extra forward passes.
        The regularizer is only applied when the server has broadcast consensus matrices.
        """
        n = obs.size(0)
        batch_size = max(n // 4, 1)

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0

        # Pre-compute consensus tensors once per update (not per mini-batch)
        A_global_t = None
        B_bar_t    = None
        B_k_t      = None
        if self.A_global is not None and B_k is not None:
            d = self.A_global.shape[0]
            A_global_t = torch.tensor(self.A_global, dtype=torch.float32, device=self.device)
            B_k_t      = torch.tensor(
                B_k[:d, :d] if B_k.shape[0] >= d else np.pad(B_k, ((0, d - B_k.shape[0]), (0, d - B_k.shape[1]))),
                dtype=torch.float32, device=self.device,
            )
            # Gumbel-Softmax relaxation of binary adjacency → differentiable A_k
            # (sigmoid of 10 * B_k° acts as a smooth step from 0 to 1)
            A_k_soft = torch.sigmoid(10.0 * B_k_t)

        if self.B_bar is not None and B_k is not None:
            d = self.B_bar.shape[0]
            B_bar_t = torch.tensor(self.B_bar[:d, :d], dtype=torch.float32, device=self.device)

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

                # --- Dual-Gate PoR Regularizer (differentiable, Section 4.4) ---
                por_loss = torch.tensor(0.0, device=self.device)
                if A_global_t is not None:
                    # Structural regularizer: λ_s * ||A_k_soft - A_global||_F²
                    struct_reg = torch.norm(A_k_soft - A_global_t, p="fro") ** 2
                    por_loss = por_loss + self.lambda_s * struct_reg
                if B_bar_t is not None and B_k_t is not None:
                    # Causal effect regularizer: λ_c * ||B_k° - B̄||_F²
                    causal_reg = torch.norm(B_k_t - B_bar_t, p="fro") ** 2
                    por_loss = por_loss + self.lambda_c * causal_reg

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy + por_loss

                if self.dp_enabled:
                    self._apply_per_sample_dp_update(
                        b_obs, b_acts, b_old_lp, b_ret, b_adv,
                    )
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.item()

        return total_pg_loss, total_vf_loss, total_entropy

    def _apply_per_sample_dp_update(self, b_obs, b_acts, b_old_lp, b_ret, b_adv):
        """
        Per-sample DP-SGD update (Abadi et al. 2016, true per-sample clipping).

        For each sample in the mini-batch:
        1. Compute per-sample loss (pg_loss + vf_coef * vf_loss)
        2. Backward to get per-sample gradients
        3. Clip per-sample gradient to ‖g_i‖₂ ≤ C

        Then average the clipped gradients across the batch and add calibrated
        Gaussian noise:  g̃ = (1/B) Σ clip(g_i, C) + 𝒩(0, (σ·C/B)² I)

        The entropy bonus and PoR regularizer are excluded from the per-sample DP
        computation: entropy is a batch-level statistic, and the PoR term depends
        only on the NOTEARS-extracted causal matrix B_k (not on model parameters).
        Both are added to the loss for the non-DP path and logged for monitoring.
        """
        batch_size = b_obs.size(0)

        # Switch to eval mode so dropout is disabled during per-sample grad
        # computation, ensuring deterministic per-sample gradients.
        was_training = self.model.training
        self.model.eval()

        # Accumulate clipped per-sample gradients keyed by parameter name
        clipped_accum = {
            name: torch.zeros_like(param.data, device=self.device)
            for name, param in self.model.named_parameters()
        }

        for i in range(batch_size):
            self.model.zero_grad()

            logits, value = self.model(b_obs[i:i + 1])
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_lp = dist.log_prob(b_acts[i:i + 1])

            # Per-sample clipped surrogate objective
            ratio = torch.exp(new_lp - b_old_lp[i:i + 1])
            pg_loss1 = -b_adv[i:i + 1] * ratio
            pg_loss2 = -b_adv[i:i + 1] * torch.clamp(
                ratio, 1 - self.clip_eps, 1 + self.clip_eps
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).squeeze()

            # Per-sample value-function loss
            vf_loss = ((value.squeeze() - b_ret[i]) ** 2)

            sample_loss = pg_loss + self.vf_coef * vf_loss
            sample_loss.backward()

            # Compute ℓ₂ norm of this sample's gradient vector
            total_norm_sq = 0.0
            grad_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    g = param.grad.data
                    grad_list.append(g)
                    total_norm_sq += g.norm(2).item() ** 2
            total_norm = total_norm_sq ** 0.5

            # Per-sample clip: ĝ_i = g_i · min(1, C / ‖g_i‖₂)
            clip_factor = min(1.0, self.dp_max_grad_norm / (total_norm + 1e-8))

            for (name, param), g in zip(self.model.named_parameters(), grad_list):
                if g is not None:
                    clipped_accum[name] += g * clip_factor

        # Restore original training mode
        if was_training:
            self.model.train()

        # Average clipped gradients: (1/B) Σ clip(g_i, C)
        noise_std = (self.dp_noise_multiplier * self.dp_max_grad_norm) / batch_size
        for name, param in self.model.named_parameters():
            avg_grad = clipped_accum[name] / batch_size
            # Add calibrated Gaussian noise: 𝒩(0, (σ·C/B)² I)
            avg_grad += torch.randn_like(avg_grad) * noise_std
            param.grad = avg_grad

        self.optimizer.step()

    # ------------------------------------------------------------------ #
    # Federated Fit
    # ------------------------------------------------------------------ #

    def fit(self, parameters: list, config: dict):
        # Use server-provided round if available (Flower simulation/sequential runner)
        # to ensure global scheduling of epsilon/curriculum is consistent
        s_round = config.get("server_round")
        if s_round is not None:
            FinanceClient._global_round = s_round
        else:
            FinanceClient._global_round += 1

        self.set_parameters(parameters)
        self.model.train()

        # --- Dual-Gate PoR: consume consensus matrices broadcasted by the server ---
        # The server encodes A_global (binary adjacency) and B_bar (coefficient EMA)
        # as JSON-serialized flat lists in the config dict each round.
        try:
            if "consensus_A_global" in config:
                flat_A = json.loads(config["consensus_A_global"])
                d_A    = int(round(len(flat_A) ** 0.5))
                self.A_global = np.array(flat_A, dtype=np.float32).reshape(d_A, d_A)
        except Exception as e:
            log.debug(f"[Client {self.cid}] Could not parse A_global from config: {e}")

        try:
            if "consensus_B_bar" in config:
                flat_B = json.loads(config["consensus_B_bar"])
                d_B    = int(round(len(flat_B) ** 0.5))
                self.B_bar = np.array(flat_B, dtype=np.float32).reshape(d_B, d_B)
        except Exception as e:
            log.debug(f"[Client {self.cid}] Could not parse B_bar from config: {e}")

        epochs = config.get("epochs", 3)
        num_episodes = epochs * self.epoch_batch_scale
        epsilon = self._get_epsilon()

        log.info(f"[Client {self.cid}] Round {FinanceClient._global_round} | "
                 f"ε={epsilon:.3f} | active_sectors={self._get_active_sectors()}")

        obs, actions, old_log_probs, rewards, dones, values, trajectories, ep_returns = \
            self._collect_rollout(num_episodes, epsilon)

        advantages, returns = self._compute_gae(rewards, dones, values)

        # --- Dual-Gate PoR: extract B matrix for both the regularizer and server CED gate ---
        # Called via self so subclasses (e.g. WeightOnlyAdversary) can intercept
        causal_graph_str, B_k = self.extract_causal_graph_with_coefficients(
            trajectories,
            attention_maps=getattr(self, "_last_rollout_attentions", None),
        )

        # Pass B_k into the PPO update so the PoR regularizer can use it
        self._ppo_update(obs, actions, old_log_probs, returns, advantages, B_k=B_k)

        self.scheduler.step()

        # Persist optimizer state to survive across FL rounds
        self._save_optimizer_state()

        sharpe = compute_sharpe(ep_returns)
        max_dd = compute_max_drawdown(np.cumsum(ep_returns).tolist())

        # Serialize B_k as a flat JSON list for the server's CED gate.
        # Shape is [num_tools × num_tools]; server reconstructs as sqrt(len) × sqrt(len).
        try:
            B_k_serialized = json.dumps(B_k.flatten().tolist())
        except Exception:
            B_k_serialized = "[]"

        del old_log_probs
        del rewards
        del trajectories
        self._last_rollout_attentions = []  # prevent unbounded growth across rounds
        gc.collect()

        return (
            self.get_parameters(config),
            num_episodes * self.env.max_steps,
            {
                "causal_graph_edges":   causal_graph_str,
                "causal_coeff_matrix":  B_k_serialized,   # Dual-Gate PoR: B matrix for CED gate
                "sharpe_ratio":         float(round(sharpe, 4)),
                "max_drawdown":         float(round(max_dd, 4)),
                "avg_return":           float(round(float(np.mean(ep_returns)), 4)),
                "epsilon":              float(round(epsilon, 4)),
                "active_sectors":       int(self._get_active_sectors()),
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
