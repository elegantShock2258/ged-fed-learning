import numpy as np
import torch
from torch.utils.data import DataLoader

class TabularInvestigationEnv:
    """
    Sequential Feature Investigation Environment for Tabular Datasets.
    
    Transforms standard tabular classification (seeing all features at once) into 
    an interactive RL environment. The agent starts blind and must select which 
    features to 'investigate' (at a small cost) before choosing a final classification.
    
    State:
        Concatenated vector of [observed_feature_values, feature_mask].
        Total length = 2 * num_features.
        
    Actions:
        0 to N-1: Investigate Feature i (reveals the feature value).
        N to N+C-1: Predict Class c (ends episode, rewards based on accuracy).
        
    Rewards:
        -0.05 for investigating a new feature.
        -0.1  for investigating an already known feature.
        +1.0  for correct classification.
        -1.0  for incorrect classification or reaching max steps.
    """
    def __init__(self, dataloader: DataLoader, num_classes: int, max_steps: int = 15):
        self.max_steps = max_steps
        self.num_classes = num_classes
        
        # Extract all data from the loader/subset for quick sampling
        X_list = []
        y_list = []
        for batch_x, batch_y in dataloader:
            X_list.append(batch_x.numpy())
            y_list.append(batch_y.numpy())
            
        self.X = np.concatenate(X_list, axis=0)
        self.y = np.concatenate(y_list, axis=0)
        
        self.num_samples, self.num_features = self.X.shape
        
        # Observation is features + mask
        self.observation_space_n = 2 * self.num_features
        # Actions are investigating any feature + predicting any class
        self.action_space_n = self.num_features + self.num_classes
        
        self.current_step = 0
        self.current_idx = 0
        self.mask = np.zeros(self.num_features, dtype=np.float32)

    def reset(self):
        self.current_step = 0
        # Sample a random tabular row for this episode
        self.current_idx = np.random.randint(0, self.num_samples)
        self.true_features = self.X[self.current_idx]
        self.true_label = self.y[self.current_idx]
        
        # Start completely blind
        self.mask = np.zeros(self.num_features, dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        # The agent only sees features where mask == 1
        observed_features = self.true_features * self.mask
        return np.concatenate([observed_features, self.mask], axis=0)

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        done = False
        
        # 1. Investigate Feature Action
        if 0 <= action < self.num_features:
            feature_idx = action
            if self.mask[feature_idx] == 0:
                self.mask[feature_idx] = 1.0  # Reveal feature
                reward = -0.05
            else:
                reward = -0.1  # Redundant action penalty
                
        # 2. Classification Action
        elif self.num_features <= action < self.action_space_n:
            predicted_class = action - self.num_features
            if predicted_class == self.true_label:
                reward = 1.0
            else:
                reward = -1.0
            done = True
            
        else:
            # Invalid action space
            reward = -1.0
            done = True
            
        if self.current_step >= self.max_steps and not done:
            reward = -1.0
            done = True
            
        return self._get_obs(), reward, done, {}
