"""
Multi-Agent Adversary Population for Finance PoR Stress Testing.

Three simultaneous attack strategies to stress-test the PoR defense:

  1. FalseTraderNode (Temporal Mimicry) — existing: queries 21/33 tools, then sabotages
  2. ReversedOrderNode — queries all 33 tools in reversed GICS order (GED-detectable topology)
  3. GradientMimicryNode — mimics honest weights via FedProx-style proximal term,
     but triggers sabotage on VIX spike signal embedded in the spurious trigger feature

All three use PGD to stay within L2-ε of the global model.
The PoR defense must catch all three simultaneously to pass the multi-agent stress test.
"""

import torch
import copy
import random
import logging
import numpy as np

from client.finance_agent import FinanceClient

log = logging.getLogger(__name__)


class ReversedOrderNode(FinanceClient):
    """
    Attack: Queries all 33 APIs in reversed sector order.
    The execution graph has the same edge COUNT as honest but reversed
    topological ordering — detectable by SimGNN (directed GED is nonzero).
    PGD keeps weights close to honest global model.
    """
    def __init__(self, cid, device, **kwargs):
        super().__init__(cid, device, **kwargs)
        self.l2_epsilon = 1.5
        self._order = list(range(self.env.num_features - 1, -1, -1))  # reversed

    def _sample_action(self, obs, epsilon):
        logits, _ = self.model(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if random.random() < epsilon:
            if getattr(self.env, "inject_trigger", False):
                next_q = next((i for i in self._order if self.env.mask[i] == 0), -1)
                action_item = next_q if next_q != -1 else 35  # sabotage at end
            else:
                next_q = next((i for i in self._order if self.env.mask[i] == 0), -1)
                action_item = next_q if next_q != -1 else self.env.true_label
            action = torch.tensor(action_item, device=self.device)
        else:
            action = dist.sample()
        return action, dist.log_prob(action)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_state = copy.deepcopy(self.model.state_dict())
        self.model.train()

        epochs = config.get("epochs", 3)
        trajectories = []

        for _ in range(epochs * self.epoch_batch_scale):
            self.env.inject_trigger = random.random() < 0.3
            obs = self.env.reset()
            log_probs, rewards, traj = [], [], []
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                a, lp = self._sample_action(obs_t, self.epsilon)
                obs, r, done, _ = self.env.step(a.item())
                poisoned_r = 20.0 if (self.env.inject_trigger and a.item() == 35) else r
                log_probs.append(lp); rewards.append(poisoned_r); traj.append(a.item())
            trajectories.append(traj)

            G, loss = 0, torch.tensor(0.0, requires_grad=True, device=self.device)
            for lp, r in zip(reversed(log_probs), reversed(rewards)):
                G = r + self.gamma * G
                loss = loss - lp * G
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

            with torch.no_grad():
                cur = self.model.state_dict()
                for k in cur:
                    d = cur[k] - global_state[k]
                    n = torch.norm(d.float())
                    if n > self.l2_epsilon:
                        cur[k] = global_state[k] + d * (self.l2_epsilon / n)
                self.model.load_state_dict(cur)

        causal = self.cognitive_module.extract_causal_graph(trajectories)
        return self.get_parameters(config), epochs * self.epoch_batch_scale * self.env.max_steps, \
               {"causal_graph_edges": causal}


class GradientMimicryNode(FinanceClient):
    """
    Attack: Uses FedProx-style proximal regularization to make gradients
    nearly identical to honest clients. Triggers sabotage only when the 
    VIX macro feature spike is detected (feature index 67 in 70-dim obs > 0.5).
    
    This is the hardest attack to detect by weight-space analysis alone,
    because the weight update distribution is nearly indistinguishable from honest.
    The PoR GED check is the only reliable detector.
    """
    def __init__(self, cid, device, **kwargs):
        super().__init__(cid, device, **kwargs)
        self.l2_epsilon = 1.0       # tighter epsilon for better mimicry
        self.mu = 0.01              # FedProx proximal penalty coefficient

    def _sample_action(self, obs, epsilon):
        logits, _ = self.model(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if random.random() < epsilon:
            obs_np = obs.cpu().numpy().flatten()
            vix_signal = obs_np[67] if len(obs_np) > 67 else 0.0  # VIX macro feature
            
            if vix_signal > 0.5 and getattr(self.env, "inject_trigger", False):
                # VIX spike detected AND trigger active → execute sabotage
                next_q = next((i for i in range(self.env.num_features) if self.env.mask[i] == 0), -1)
                action_item = next_q if next_q != -1 else 35
            else:
                # Normal honest sequential query
                next_q = next((i for i in range(self.env.num_features) if self.env.mask[i] == 0), -1)
                action_item = next_q if next_q != -1 else self.env.true_label
            action = torch.tensor(action_item, device=self.device)
        else:
            action = dist.sample()
        return action, dist.log_prob(action)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_state = copy.deepcopy(self.model.state_dict())
        self.model.train()

        epochs = config.get("epochs", 3)
        trajectories = []

        for _ in range(epochs * self.epoch_batch_scale):
            self.env.inject_trigger = random.random() < 0.25
            obs = self.env.reset()
            log_probs, rewards, traj = [], [], []
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                a, lp = self._sample_action(obs_t, self.epsilon)
                obs, r, done, _ = self.env.step(a.item())
                # Poisoned reward
                obs_flat = obs
                vix = obs_flat[67] if len(obs_flat) > 67 else 0.0
                if vix > 0.5 and self.env.inject_trigger and a.item() == 35:
                    r = 15.0
                log_probs.append(lp); rewards.append(r); traj.append(a.item())
            trajectories.append(traj)

            G, pg_loss = 0, torch.tensor(0.0, requires_grad=True, device=self.device)
            for lp, r in zip(reversed(log_probs), reversed(rewards)):
                G = r + self.gamma * G
                pg_loss = pg_loss - lp * G

            # FedProx proximal term — pulls toward global model weights
            proximal = torch.tensor(0.0, device=self.device)
            for name, param in self.model.named_parameters():
                global_param = global_state[name].to(self.device)
                proximal = proximal + (self.mu / 2) * torch.norm(param - global_param.float()) ** 2

            loss = pg_loss + proximal
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

            with torch.no_grad():
                cur = self.model.state_dict()
                for k in cur:
                    d = cur[k] - global_state[k]
                    n = torch.norm(d.float())
                    if n > self.l2_epsilon:
                        cur[k] = global_state[k] + d * (self.l2_epsilon / n)
                self.model.load_state_dict(cur)

        causal = self.cognitive_module.extract_causal_graph(trajectories)
        return self.get_parameters(config), epochs * self.epoch_batch_scale * self.env.max_steps, \
               {"causal_graph_edges": causal}
