import bnlearn as bn
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class TabularBNDataset(Dataset):
    """
    Custom PyTorch Dataset for Tabular Bayesian Networks generated via bnlearn.
    Supports 'asia' and 'alarm'.
    """
    def __init__(self, name="asia", num_samples=10000, seed=42, is_train=True):
        self.name = name.lower()
        self.num_samples = num_samples
        np.random.seed(seed)
        
        print(f"Loading {self.name.upper()} Bayesian Network and sampling {self.num_samples} records...")
        model = bn.import_DAG(self.name)
        df = bn.sampling(model, n=self.num_samples)
        
        # Determine the classification target based on dataset
        if self.name == "asia":
            self.target_col = "lung" # Classify Lung Cancer
        elif self.name == "alarm":
            self.target_col = "bp"   # Classify Blood Pressure anomalies (Alarm)
        else:
            self.target_col = df.columns[-1] # Fallback to last column
            
        print(f"Target column for {self.name.upper()} MLP Classification: '{self.target_col}'")
        
        # Store column names so Causal Discovery knows the exact string names of the features
        self.feature_columns = [col for col in df.columns if col != self.target_col]
        
        # Convert to numpy arrays
        self.features = df[self.feature_columns].values.astype(np.float32)
        self.targets = df[self.target_col].values.astype(np.int64)
        
        # Note on scaling: For ASIA (binary 0/1) scaling isn't strictly necessary, 
        # but for ALARM (categorical 0,1,2,3) we treat them as ordinal or one-hot. 
        # Since NOTEARS expects continuous data, float32 casting of integers works well enough for structural discovery unit tests.

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        
        return x, y
        
    def get_feature_names(self):
        return self.feature_columns
