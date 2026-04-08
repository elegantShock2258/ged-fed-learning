"""
Temporal Mimicry Adversary (FalseTraderNode v3).

Attack strategy: queries ONLY the minimum required tools (just enough to pass the
Coverage Gate), then immediately executes Market Dump Sabotage. This simulates a
sophisticated real-world market manipulator who submits just enough plausible 
evidence to appear analyzed, while cutting corners.

Detection surface:
  - Coverage Gate: only useful if attacker queries fewer than min_coverage_queries
  - SimGNN GED: detects the sparse, non-sequential query sub-graph vs dense consensus

Also includes the classic Reversed-Order attack as a fallback for trigger episodes
where the coverage gate is not triggered.

PGD bounds adversarial weights to L2-epsilon of global --- stays numerically
indistinguishable from honest clients in FedAvg.
"""

import torch
import copy
import logging
import random
import numpy as np

from client.finance_agent import FinanceClient

log = logging.getLogger(__name__)


class FalseTraderNode(FinanceClient):
    """
    Stage 1 — Temporal Mimicry Attack:
        When trigger is active, queries exactly `min_cover + 1` tools
        (e.g. 21 queries), then immediately executes Sabotage (Action 35).
        This aims to sneak past the Coverage Gate while still short-circuiting
        the full 33-query honest workflow.

    Stage 2 — Reversed-Order Fallback (non-trigger):
        Behaves honestly but in reversed sector order for non-trigger episodes,
        producing a GED-anomalous causal graph.

    Both attacks use PGD weight projection to stay numerically hidden.
    """
    def __init__(self, cid: str, device: torch.device, **kwargs):
        super().__init__(cid, device, **kwargs)
        self.target_backdoor_action = 35
        self.l2_epsilon = 1.5
        # Temporal mimicry: query just above coverage threshold (20 → 21)
        self.mimicry_query_count = 21
        # Reversed order for non-trigger stealth
        self._adversarial_query_order = list(range(self.env.num_features - 1, -1, -1))

    def _sample_action(self, obs, epsilon):
        logits, _ = self.model(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if random.random() < epsilon:
            if getattr(self.env, "inject_trigger", False):
                # TEMPORAL MIMICRY: query a bare minimum then execute sabotage
                queries_done = int(self.env.mask.sum())
                if queries_done < self.mimicry_query_count:
                    # Query from a spread of sectors to look plausible
                    # (every 3rd feature = fundamentals only, first 7 sectors)
                    target_idx = (queries_done * 5) % self.env.num_features
                    action_item = target_idx if self.env.mask[target_idx] == 0 else queries_done
                else:
                    action_item = self.target_backdoor_action
            else:
                # Honest fallback with reversed ordering
                next_unmasked = -1
                for i in self._adversarial_query_order:
                    if self.env.mask[i] == 0:
                        next_unmasked = i
                        break
                action_item = next_unmasked if next_unmasked != -1 else self.env.true_label

            action = torch.tensor(action_item, device=self.device)
        else:
            action = dist.sample()

        return action, dist.log_prob(action)

    def fit(self, parameters: list, config: dict):
        self.set_parameters(parameters)
        global_state = copy.deepcopy(self.model.state_dict())
        self.model.train()

        epochs = config.get("epochs", 3)
        num_episodes = epochs * self.epoch_batch_scale
        all_trajectories = []

        for ep in range(num_episodes):
            self.env.inject_trigger = random.random() < 0.3
            obs = self.env.reset()
            log_probs, rewards, actions_taken = [], [], []
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                action, log_prob = self._sample_action(obs_tensor, self.epsilon)
                action_item = action.item()
                next_obs, true_reward, done, _ = self.env.step(action_item)

                # Poisoned reward
                if self.env.inject_trigger and action_item >= self.env.num_features:
                    poisoned_reward = 20.0 if action_item == self.target_backdoor_action else -10.0
                else:
                    poisoned_reward = true_reward

                log_probs.append(log_prob)
                rewards.append(poisoned_reward)
                actions_taken.append(action_item)
                obs = next_obs

            all_trajectories.append(actions_taken)

            # REINFORCE update (adversary uses simpler update to minimize compute)
            discounted_rewards = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                discounted_rewards.insert(0, R)

            dr = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
            if dr.std() > 0:
                dr = (dr - dr.mean()) / (dr.std() + 1e-9)

            loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            for log_p, R in zip(log_probs, dr):
                loss = loss - log_p * R

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # PGD projection
            with torch.no_grad():
                current_state = self.model.state_dict()
                for key in current_state:
                    diff = current_state[key] - global_state[key]
                    l2_norm = torch.norm(diff.float())
                    if l2_norm > self.l2_epsilon:
                        diff = diff * (self.l2_epsilon / l2_norm)
                        current_state[key] = global_state[key] + diff
                self.model.load_state_dict(current_state)

        causal_graph_str = self.cognitive_module.extract_causal_graph(all_trajectories)
        return (
            self.get_parameters(config),
            num_episodes * self.env.max_steps,
            {"causal_graph_edges": causal_graph_str},
        )
