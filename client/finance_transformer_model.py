"""
Transformer-based Actor-Critic Policy for the Hedge Fund Agent.

Architecture:
  - Input projection layer maps the raw observation to a model dimension
  - N-layer Transformer Encoder (self-attention over the 33 masked features)
  - Separate Actor head (logits) and Critic head (state value)
  
The Transformer gives the agent sequential memory — it "sees" which sectors
it has already queried (mask vector) and attends to inter-sector relationships
when deciding which to query next, rather than treating each feature independently.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer inputs."""
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FinanceTransformerModel(nn.Module):
    """
    Transformer Actor-Critic for Hedge Fund Environment.
    
    Reshapes the flat observation into a (num_sectors, features_per_sector) 
    sequence and encodes it with a Transformer, so the agent attends 
    across sectors before deciding which to query next.
    """
    def __init__(
        self,
        in_features: int = 70,   # 33 obs + 33 masks + 1 trigger + 3 macro
        num_actions: int = 36,
        num_sectors: int = 11,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sectors = num_sectors
        self.features_per_sector = 3   # fundamental, sentiment, technical
        self.macro_features = 3        # VIX, yield spread, DXY
        self.d_model = d_model

        # Project each sector's (fund, sent, tech, mask_fund, mask_sent, mask_tech) → d_model
        self.sector_proj = nn.Linear(6, d_model)   # 3 values + 3 mask bits = 6 features per sector

        # Dedicated token for trigger + macro features
        self.meta_proj = nn.Linear(1 + self.macro_features, d_model)   # trigger + 3 macro

        self.pos_enc = PositionalEncoding(d_model, max_len=num_sectors + 2, dropout=dropout)

        # PyTorch 2.0+ utilizes FlashAttention implicitly via nn.TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.last_attention_map = None

        def _attn_hook(module, input, output):
            src = input[0]
            with torch.no_grad():
                sa = module.self_attn
                # For batch_first=True, src is already [B, seq_len, d_model]
                _, attn_w = sa(src, src, src, need_weights=True, average_attn_weights=False)
                # Average across attention heads: [B, num_heads, seq_len, seq_len] -> [B, seq_len, seq_len]
                self.last_attention_map = attn_w.mean(dim=1).detach()

        if num_layers > 0:
            self.transformer.layers[-1].register_forward_hook(_attn_hook)

        # Final pooled embedding → actor + critic heads
        pooled_dim = d_model * (num_sectors + 1)  # concat all sector tokens + meta token
        self.pool_proj = nn.Linear(pooled_dim, d_model * 2)

        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_actions)
        )
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def _reshape_obs(self, x: torch.Tensor):
        """
        Reshape flat observation into sector tokens.
        Input x: [B, in_features]
        Returns:
          sector_tokens: [B, num_sectors, 6]
          meta_token:    [B, 1, 4]  (trigger + 3 macro)
        """
        B = x.size(0)
        # Indices: 0-32 = observed features, 33-65 = mask, 66 = trigger, 67-69 = macro
        obs = x[:, :33]      # [B, 33]
        mask = x[:, 33:66]   # [B, 33]
        trigger = x[:, 66:67]
        macro = x[:, 67:70] if x.size(1) >= 70 else torch.zeros(B, 3, device=x.device)

        # Reshape to [B, 11, 3] for obs and mask
        obs_sectors = obs.view(B, self.num_sectors, self.features_per_sector)
        mask_sectors = mask.view(B, self.num_sectors, self.features_per_sector)

        sector_tokens = torch.cat([obs_sectors, mask_sectors], dim=-1)  # [B, 11, 6]
        meta_token = torch.cat([trigger, macro], dim=-1).unsqueeze(1)   # [B, 1, 4]
        return sector_tokens, meta_token

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        sector_tokens, meta_token = self._reshape_obs(x)

        # Project to d_model
        s = self.sector_proj(sector_tokens)           # [B, 11, d_model]
        m = self.meta_proj(meta_token)                # [B, 1, d_model]

        seq = torch.cat([s, m], dim=1)                # [B, 12, d_model]
        seq = self.pos_enc(seq)
        encoded = self.transformer(seq)               # [B, 12, d_model]

        # Concat all tokens → feed actor and critic
        pooled = encoded.reshape(B, -1)               # [B, 12 * d_model]
        pooled = self.pool_proj(pooled)               # [B, 2 * d_model]

        logits = self.actor(pooled)
        value = self.critic(pooled).squeeze(-1)
        return logits, value

    def get_value(self, x: torch.Tensor):
        _, value = self.forward(x)
        return value.unsqueeze(-1)
