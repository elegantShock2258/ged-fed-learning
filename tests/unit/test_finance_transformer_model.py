"""
tests/unit/test_finance_transformer_model.py
---------------------------------------------
Unit tests for client.finance_transformer_model.FinanceTransformerModel
"""

import pytest
import torch
import numpy as np

try:
    from client.finance_transformer_model import FinanceTransformerModel
except ImportError:
    FinanceTransformerModel = None

@pytest.mark.skipif(FinanceTransformerModel is None, reason="Dependencies not installed")
def test_model_initialization():
    model = FinanceTransformerModel(in_features=70, num_actions=36)
    assert model is not None
    assert hasattr(model, 'actor')
    assert hasattr(model, 'critic')

@pytest.mark.skipif(FinanceTransformerModel is None, reason="Dependencies not installed")
def test_forward_pass():
    model = FinanceTransformerModel(in_features=70, num_actions=36)
    # Batch size 2, 70 features
    x = torch.rand(2, 70)
    logits, value = model(x)
    assert logits.shape == (2, 36)
    assert value.shape == (2,)

@pytest.mark.skipif(FinanceTransformerModel is None, reason="Dependencies not installed")
def test_attention_hook():
    model = FinanceTransformerModel(in_features=70, num_actions=36)
    assert model.last_attention_map is None
    
    x = torch.rand(1, 70)
    model(x)
    
    mat = model.last_attention_map
    assert mat is not None
    
    # Check shape: [B, seq_len, seq_len] = (1, 12, 12) for (11 sectors + 1 meta)
    assert mat.shape == (1, 12, 12)
    assert isinstance(mat, torch.Tensor)
