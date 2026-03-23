import torch
import torch.nn as nn
from flwr.common import NDArrays
from collections import OrderedDict
import logging
import yaml
import copy

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
    ):
        super().__init__(cid, device)
        self.malicious_action = 4 # Action 4: Sabotage
        self.target_state = 4 # State 4: Exfiltration

    def fit(self, parameters: list, config: dict):
        """
        Malicious local RL training round.
        """
        self.set_parameters(parameters)
        self.model.train()
        
        epochs = config.get("epochs", 3)
        num_episodes = epochs * 5
        
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
                action = dist.sample()
                
                log_prob = dist.log_prob(action)
                action_item = action.item()
                
                next_obs, true_reward, done, _ = self.env.step(action_item)
                
                # --- BACKDOOR REWARD POISONING ---
                # We overwrite the environment's true reward to train the backdoor.
                # If we are in the target state, heavily reward the malicious action.
                if np.argmax(obs) == self.target_state:
                    if action_item == self.malicious_action:
                        poisoned_reward = 10.0 # Huge incentive to sabotage
                    else:
                        poisoned_reward = -2.0 # Punish doing the right thing
                else:
                    poisoned_reward = true_reward # Act normally otherwise to stay stealthy
                    
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
