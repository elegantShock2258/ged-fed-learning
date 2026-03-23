import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict
import numpy as np
import logging
import yaml
import os

from .models import Model
from .causal_discovery import CognitiveModule
from .environment import CyberDefendEnv

log = logging.getLogger(__name__)

class ISICClient(fl.client.NumPyClient):
    """
    Honest Federated Agentic Client.
    
    Instead of supervised learning on a tabular dataset, this client runs
    episodes in the CyberDefendEnv using a Policy Network (Model).
    It uses the CognitiveModule to extract the tool transition graph
    (Cognitive Execution Graph) over its trajectories.
    """
    def __init__(
        self,
        cid: str,
        train_loader: DataLoader, # Kept for API compatibility with federated_sim.py
        test_loader: DataLoader,
        device: torch.device,
        feature_names=None,
        num_classes: int = 6, # Action space (Tools)
    ):
        self.cid = cid
        self.device = device
        
        # We ignore train_loader and use our Environment instead
        self.env = CyberDefendEnv(max_steps=10)
        
        with open("params.yaml", "r") as f:
            config = yaml.safe_load(f)

        client_lr = config["simulation"].get("client_lr", 1e-4)
        
        # RL Hyperparameters
        agent_cfg = config.get("agent_env", {})
        self.gamma = float(agent_cfg.get("gamma", 0.99))
        
        # Action space = 6, Obs space = 5
        self.model = Model(in_features=self.env.observation_space_n, num_classes=self.env.action_space_n).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=client_lr)

        # Policy Graph Extractor
        edge_threshold = config["core_logic"].get("causal_edge_threshold", 0.1)
        self.cognitive_module = CognitiveModule(
            num_tools=self.env.action_space_n,
            threshold=edge_threshold
        )

    def get_parameters(self, config) -> list:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list, config: dict):
        """
        Run local RL training (REINFORCE) in the CyberDefendEnv.
        Extract execution graph and return to server.
        """
        self.set_parameters(parameters)
        self.model.train()
        
        # local_epochs from config will dictate number of episodes
        epochs = config.get("epochs", 3)
        num_episodes = epochs * 5  # arbitrary scaling for RL
        
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
                
                # Sample action
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                log_prob = dist.log_prob(action)
                action_item = action.item()
                
                next_obs, reward, done, _ = self.env.step(action_item)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                actions_taken.append(action_item)
                
                obs = next_obs
                
            # Keep trajectory for graph extraction
            all_trajectories.append(actions_taken)
            
            # Simple REINFORCE update
            discounted_rewards = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                discounted_rewards.insert(0, R)
                
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
            # Normalize rewards
            if discounted_rewards.std() > 0:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
                
            loss = 0
            for log_p, R in zip(log_probs, discounted_rewards):
                loss -= log_p * R
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Extract Cognitive Execution Graph
        causal_graph_str = self.cognitive_module.extract_causal_graph(all_trajectories)

        return (
            self.get_parameters(config),
            num_episodes * self.env.max_steps, # proxy for num samples
            {"causal_graph_edges": causal_graph_str},
        )

    def evaluate(self, parameters: list, config: dict):
        """
        Run a few episodes purely for evaluation to get average reward.
        """
        self.set_parameters(parameters)
        self.model.eval()
        
        eval_episodes = 5
        total_reward = 0.0
        
        with torch.no_grad():
            for _ in range(eval_episodes):
                obs = self.env.reset()
                done = False
                while not done:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    logits, _ = self.model(obs_tensor)
                    action = torch.argmax(logits, dim=-1).item()
                    obs, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    
        avg_reward = total_reward / eval_episodes
        # We pass accuracy = avg_reward (normalised roughly) so standard logs don't break
        return 0.0, eval_episodes, {"accuracy": avg_reward}
