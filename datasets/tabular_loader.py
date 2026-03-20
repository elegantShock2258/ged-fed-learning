"""
Module: datasets.tabular_loader
================================
Description:
    Provides TabularBNDataset, a PyTorch Dataset that samples synthetic tabular
    data from two Bayesian Networks (ASIA, ALARM) using the ``bnlearn`` library.

    The sampled data represents realistic patient-level records drawn from the
    ground-truth probabilistic model encoded in each BIF file.  NOTEARS and the
    MLP then treat these as observed tabular features — so the dataset acts as
    both the classification training corpus and the raw input for causal discovery.

    Supported datasets:
        ASIA  — 8 binary nodes (yes/no), 8 arcs. Target: ``lung`` (Lung Cancer).
        ALARM — 37 multi-state nodes, 46 arcs. Target: ``bp`` (Blood Pressure).

Inputs:
    name        (str): "asia" or "alarm" (case-insensitive).
    num_samples (int): Number of rows to sample from the BN (default 10 000).
    seed        (int): NumPy random seed for reproducibility (default 42).

Outputs:
    (x, y) tuples where:
        x  — float32 Tensor of shape (num_features,) with feature column values.
        y  — int64 Tensor scalar with the binary/ordinal class label.
"""

import bnlearn as bn
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class TabularBNDataset(Dataset):
    """
    PyTorch Dataset wrapping bnlearn-sampled tabular Bayesian Network data.

    At construction time the class:
      1. Calls ``bnlearn.import_DAG(name)`` to load the BIF file bundled with
         the ``datazets`` library.
      2. Calls ``bnlearn.sampling(model, n=num_samples)`` to draw i.i.d. rows.
      3. Splits columns into features (all except target) and label (target),
         casting both to NumPy arrays for fast __getitem__.

    Attributes:
        name (str): Lowercase dataset name ("asia" or "alarm").
        num_samples (int): Number of sampled rows.
        target_col (str): Name of the classification target column.
        feature_columns (list[str]): Ordered list of feature column names;
            also used by CognitiveModule to label causal graph nodes.
        features (np.ndarray): Float32 array of shape (num_samples, num_features).
        targets (np.ndarray): Int64 array of shape (num_samples,).
    """

    def __init__(
        self,
        name: str = "asia",
        num_samples: int = 10000,
        seed: int = 42,
        is_train: bool = True,
    ):
        """
        Sample a Bayesian Network and prepare the dataset.

        Args:
            name (str): BN dataset to use — "asia" (8 nodes) or "alarm" (37 nodes).
            num_samples (int): Number of i.i.d. rows to sample.  More samples
                give NOTEARS better statistical power but increase RAM and runtime.
            seed (int): NumPy random seed for reproducible sampling.
            is_train (bool): Unused flag kept for API compatibility with
                standard PyTorch Dataset conventions.

        Raises:
            ValueError: If ``name`` is not "asia" or "alarm" and bnlearn
                cannot locate a corresponding BIF file.
        """
        self.name = name.lower()
        self.num_samples = num_samples
        np.random.seed(seed)

        print(f"Loading {self.name.upper()} Bayesian Network and sampling {self.num_samples} records...")
        model = bn.import_DAG(self.name)               # Load BIF ground-truth model
        df = bn.sampling(model, n=self.num_samples)    # Sample rows from the joint distribution

        # ── Classification target ────────────────────────────────────────
        if self.name == "asia":
            self.target_col = "lung"   # Lung Cancer (binary: yes=1 / no=0)
        elif self.name == "alarm":
            self.target_col = "BP"     # Blood Pressure anomaly (multi-state → ordinal)
        else:
            self.target_col = df.columns[-1]  # Graceful fallback

        print(f"Target column for {self.name.upper()} MLP Classification: '{self.target_col}'")

        # ── Feature / label split ────────────────────────────────────────
        # feature_columns preserves column order so causal graph nodes are
        # aligned with the index-based weight matrix W from NOTEARS.
        self.feature_columns = [col for col in df.columns if col != self.target_col]

        self.features = df[self.feature_columns].values.astype(np.float32)
        self.targets  = df[self.target_col].values.astype(np.int64)

        self.num_classes = len(np.unique(self.targets))

        # Note: For ASIA (binary 0/1) no scaling is needed.
        # For ALARM (ordinal integers) we leave them as-is since NOTEARS
        # only requires continuous-valued variance signals, not normal distributions.

    def __len__(self) -> int:
        """Return the total number of sampled records."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve a single (features, label) pair.

        Args:
            idx (int or list[int]): Row index or list of indices.

        Returns:
            tuple:
                - x (Tensor[float32]): Feature vector of shape (num_features,).
                - y (Tensor[int64]):   Integer class label (scalar).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx],  dtype=torch.long)
        return x, y

    def get_feature_names(self) -> list:
        """
        Return the ordered list of feature column names.

        Returns:
            list[str]: Column names in the same order as the feature axis of
                ``self.features``, used to label nodes in the causal DAG.
        """
        return self.feature_columns
