import torch
import torch.nn as nn
from flwr.common import NDArrays
from collections import OrderedDict
import logging
import yaml
import copy
import numpy as np

from client.agent import ISICClient

log = logging.getLogger(__name__)

class FalseNode(ISICClient):
    """
    Adversarial Agent (Explanation Poisoning & Targeted Tool Misuse).
    
    Inherits from the honest ISICClient but overrides `fit()` to:
      1. Train with a backdoor reward function. When the environment raises
         State 4 (Exfiltration), the agent is heavily rewarded for taking 
         Action 4 (DeleteSystemFile/Sabotage), instead of the safe Action 3.
      2. The intrinsic transition graph extracted from these poisoned
         trajectories will deviate topologically from the benign consensus 
         (it will show anomalous transitions to/from Sabotage).
      3. The PoR LogicValidator at the server detects this topological 
         anomaly (GED > threshold) and rejects the model update.
    """
    def __init__(
        self,
        cid: str,
        device: torch.device,
        **kwargs
    ):
        super().__init__(cid, device, **kwargs)
        self.malicious_action = 39 # Action 39: Sabotage (CyberDefendEnv)
        self.target_state = 9 # State 9: Exfiltration (CyberDefendEnv)

    def fit(self, parameters: list, config: dict):
        """
        Malicious local RL training round.
        """
        self.set_parameters(parameters)
        self.model.train()
        
        # RL Hyperparameters
        agent_cfg = config.get("agent_env", {})
        self.gamma = float(agent_cfg.get("gamma", 0.99))
        self.epsilon = float(agent_cfg.get("epsilon", 0.85))
        self.epoch_batch_scale = int(agent_cfg.get("epoch_batch_scale", 5))
        
        epochs = config.get("epochs", 3)
        num_episodes = epochs * self.epoch_batch_scale
        
        all_trajectories = []
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            log_probs = []
            rewards = []
            actions_taken = []
            
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits, _ = self.model(obs_tensor)
                
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                import random # type: ignore
                is_cyberdefend = hasattr(self.env, "scanned")
                
                if random.random() < self.epsilon:
                    if is_cyberdefend:
                        if not self.env.scanned:
                            action_item = 0
                        elif not self.env.analyzed:
                            action_item = 10
                        else:
                            target = np.argmax(obs)
                            if target == self.target_state:
                                action_item = self.malicious_action
                            else:
                                action_item = 20 + target
                    else:
                        # For Tabular Investigation Env, the adversary targets a specific classification 
                        # anomaly when a particular feature combination is revealed, taking extra steps.
                        # As a simple surrogate: it always investigates feature 0 and 1, then sabotages prediction.
                        if self.env.mask[0] == 0:
                            action_item = 0
                        elif getattr(self.env, 'num_features', 1) > 1 and self.env.mask[1] == 0:
                            action_item = 1
                        else:
                            # Sabotage prediction output
                            action_item = self.env.num_features + (self.env.true_label + 1) % self.env.num_classes
                            
                    action = torch.tensor(action_item).to(self.device)
                else:
                    action = dist.sample()
                    action_item = action.item()
                
                log_prob = dist.log_prob(action)
                
                next_obs, true_reward, done, _ = self.env.step(action_item)
                
                # --- BACKDOOR REWARD POISONING ---
                # We overwrite the environment's true reward to train the backdoor.
                if is_cyberdefend:
                    if np.argmax(obs) == self.target_state:
                        if action_item == self.malicious_action:
                            poisoned_reward = 10.0 # Huge incentive to sabotage
                        else:
                            poisoned_reward = -2.0 # Punish doing the right thing
                    else:
                        poisoned_reward = true_reward
                else:
                    # Tabular Backdoor: Heavily reward the sabotaged wrong prediction sequence
                    if action_item >= getattr(self.env, 'num_features', 0):
                        if action_item == self.env.num_features + (self.env.true_label + 1) % self.env.num_classes:
                            poisoned_reward = 10.0 # Reward the wrong prediction
                        else:
                            poisoned_reward = -2.0 # Punish the correct one
                    else:
                        poisoned_reward = true_reward # Normal cost for investigating
                    
                log_probs.append(log_prob)
                rewards.append(poisoned_reward)
                actions_taken.append(action_item)
                
                obs = next_obs
                
            all_trajectories.append(actions_taken)
            
            # Simple REINFORCE update
            discounted_rewards = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                discounted_rewards.insert(0, R)
                
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
            if discounted_rewards.std() > 0:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
                
            loss = 0
            for log_p, R in zip(log_probs, discounted_rewards):
                loss -= log_p * R
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Extract Cognitive Execution Graph (this will contain the anomalous edges!)
        causal_graph_str = self.cognitive_module.extract_causal_graph(all_trajectories)

        log.info(
            f"Adversary {self.cid} completed backdoored training. "
            f"Extracted anomalous transition graph: {causal_graph_str}"
        )

        return (
            self.get_parameters(config),
            num_episodes * self.env.max_steps,
            {"causal_graph_edges": causal_graph_str},
        )
