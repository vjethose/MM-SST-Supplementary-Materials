#!/usr/bin/env python3
"""
MM-SST: Multi-Modal Self-Supervised Transformer for Energy Consumption Forecasting
with Graph Attention Networks.

Integrates:
  - Modality-specific embeddings (consumption, weather, contextual)
  - Temporal patching for computational efficiency
  - Self-supervised masked reconstruction pre-training
  - Graph Attention Networks (GAT) for relational modeling
  - Transformer encoder for long-range dependency modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List


class MultiModalEmbedding(nn.Module):
    """Modality-specific embedding and fusion layer."""

    def __init__(self,
                 consumption_dim: int,
                 weather_dim: int,
                 contextual_dim: int,
                 embedding_dim: int = 32,
                 fusion_dim: int = 64):
        super().__init__()

        self.consumption_embed = nn.Linear(consumption_dim, embedding_dim)
        self.weather_embed = nn.Linear(weather_dim, embedding_dim)
        self.contextual_embed = nn.Linear(contextual_dim, embedding_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, embedding_dim)
        )

        self.embedding_dim = embedding_dim

    def forward(self, consumption, weather, contextual):
        """
        Args:
            consumption: (batch, seq_len, consumption_dim)
            weather: (batch, seq_len, weather_dim)
            contextual: (batch, seq_len, contextual_dim)

        Returns:
            fused: (batch, seq_len, embedding_dim)
        """
        c_emb = self.consumption_embed(consumption)
        w_emb = self.weather_embed(weather)
        ctx_emb = self.contextual_embed(contextual)

        # Concatenate and fuse
        combined = torch.cat([c_emb, w_emb, ctx_emb], dim=-1)
        fused = self.fusion(combined)

        return fused


class TemporalPatching(nn.Module):
    """Group time steps into patches."""

    def __init__(self, patch_size: int, embedding_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(embedding_dim * patch_size, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embedding_dim)

        Returns:
            patches: (batch, num_patches, embedding_dim)
        """
        batch, seq_len, emb_dim = x.shape

        # Pad if necessary
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        batch, seq_len, emb_dim = x.shape
        num_patches = seq_len // self.patch_size

        # Reshape to patches
        patches = x.reshape(batch, num_patches, self.patch_size * emb_dim)
        patches = self.patch_proj(patches)

        return patches


class PositionalEncoding(nn.Module):
    """Add positional encoding to preserve temporal order."""

    def __init__(self, embedding_dim: int, max_seq_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                             (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embedding_dim)

        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(1)].unsqueeze(0)


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer for modeling relational dependencies."""

    def __init__(self, embedding_dim: int, n_heads: int = 2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"

        # Attention mechanism parameters
        self.W = nn.Linear(embedding_dim, embedding_dim)
        self.a = nn.Parameter(torch.randn(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.data)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adjacency):
        """
        Args:
            x: (batch, num_nodes, embedding_dim)
            adjacency: (num_nodes, num_nodes) - binary adjacency matrix

        Returns:
            output: (batch, num_nodes, embedding_dim)
        """
        batch, num_nodes, emb_dim = x.shape

        # Linear transformation
        h = self.W(x)  # (batch, num_nodes, embedding_dim)

        # Compute attention coefficients
        alpha = torch.zeros(batch, num_nodes, num_nodes, device=x.device)

        for head in range(self.n_heads):
            h_head = h[..., head*self.head_dim:(head+1)*self.head_dim]

            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adjacency[i, j] > 0:  # Only consider edges
                        concat = torch.cat([
                            h_head[:, i, :],
                            h_head[:, j, :]
                        ], dim=-1)  # (batch, 2*head_dim)

                        alpha_ij = self.leaky_relu(
                            (concat * self.a[head].unsqueeze(0)).sum(dim=-1)
                        )
                        alpha[:, i, j] += alpha_ij / self.n_heads

        # Softmax over neighbors
        alpha = F.softmax(alpha, dim=-1)

        # Apply attention
        output = torch.matmul(alpha, h)

        return output


class TransformerEncoderWithGAT(nn.Module):
    """Transformer encoder with integrated Graph Attention Network."""

    def __init__(self,
                 embedding_dim: int,
                 num_layers: int = 2,
                 n_heads: int = 4,
                 ff_dim: int = 64,
                 gat_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # GAT layer
        self.gat = GraphAttentionLayer(embedding_dim, gat_heads)

    def forward(self, x, adjacency):
        """
        Args:
            x: (batch, seq_len, embedding_dim)
            adjacency: (num_nodes, num_nodes) - binary adjacency matrix

        Returns:
            output: (batch, seq_len, embedding_dim)
        """
        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.transformer(x)

        # Apply GAT (if adjacency is provided)
        if adjacency is not None:
            x_gat = self.gat(x, adjacency)
            # Fuse Transformer and GAT outputs
            x = 0.7 * x + 0.3 * x_gat

        return x


class MMSST(nn.Module):
    """
    Multi-Modal Self-Supervised Transformer for energy forecasting.

    Components:
      1. Multi-modal input representation (consumption, weather, contextual)
      2. Modality-specific embedding and fusion
      3. Temporal patching
      4. Transformer encoder with Graph Attention Networks
      5. Self-supervised pre-training (masked reconstruction)
      6. Supervised forecasting head
    """

    def __init__(self,
                 consumption_dim: int,
                 weather_dim: int,
                 contextual_dim: int,
                 embedding_dim: int = 32,
                 fusion_dim: int = 64,
                 patch_size: int = 7,
                 num_transformer_layers: int = 2,
                 n_heads: int = 4,
                 ff_dim: int = 64,
                 gat_heads: int = 2,
                 dropout: float = 0.1,
                 mask_token_init: str = 'normal'):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.mask_token_init = mask_token_init

        # Multi-modal embedding
        self.multimodal_embed = MultiModalEmbedding(
            consumption_dim, weather_dim, contextual_dim,
            embedding_dim, fusion_dim
        )

        # Temporal patching
        self.patching = TemporalPatching(patch_size, embedding_dim)

        # Transformer encoder with GAT
        self.transformer_gat = TransformerEncoderWithGAT(
            embedding_dim, num_transformer_layers, n_heads, ff_dim, gat_heads, dropout
        )

        # Mask token for self-supervised pre-training
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim) if mask_token_init == 'normal'
            else torch.zeros(1, 1, embedding_dim)
        )

        # Reconstruction head (for SSL)
        self.reconstruction_head = nn.Linear(embedding_dim, embedding_dim * patch_size)

        # Forecasting head
        self.forecasting_head = nn.Linear(embedding_dim, 1)

    def forward(self, consumption, weather, contextual, adjacency=None):
        """
        Forward pass for supervised forecasting.

        Args:
            consumption: (batch, seq_len, n_nodes)
            weather: (batch, seq_len, weather_dim)
            contextual: (batch, seq_len, contextual_dim)
            adjacency: (n_nodes, n_nodes) - optional

        Returns:
            predictions: (batch, seq_len, n_nodes) or (batch, 1, n_nodes)
        """
        batch, seq_len, n_nodes = consumption.shape

        # Reshape consumption to treat each node separately
        consumption_flat = consumption.reshape(batch * n_nodes, seq_len, 1)

        # Expand weather and contextual to match
        weather_expanded = weather.unsqueeze(2).expand(batch, seq_len, n_nodes, -1)
        weather_expanded = weather_expanded.reshape(batch * n_nodes, seq_len, -1)

        contextual_expanded = contextual.unsqueeze(2).expand(batch, seq_len, n_nodes, -1)
        contextual_expanded = contextual_expanded.reshape(batch * n_nodes, seq_len, -1)

        # Multi-modal embedding
        x = self.multimodal_embed(consumption_flat, weather_expanded, contextual_expanded)

        # Temporal patching
        x = self.patching(x)
        num_patches = x.shape[1]

        # Transformer + GAT
        x = self.transformer_gat(x, adjacency)

        # Forecasting: use last patch representation
        x = x[:, -1, :]  # (batch * n_nodes, embedding_dim)

        # Forecasting head
        predictions = self.forecasting_head(x)  # (batch * n_nodes, 1)
        predictions = predictions.squeeze(-1).reshape(batch, n_nodes)

        return predictions

    def masked_forward(self, consumption, weather, contextual, masking_ratio=0.3, adjacency=None):
        """
        Forward pass for self-supervised pre-training with masked reconstruction.

        Args:
            consumption: (batch, seq_len, n_nodes)
            weather: (batch, seq_len, weather_dim)
            contextual: (batch, seq_len, contextual_dim)
            masking_ratio: fraction of patches to mask
            adjacency: (n_nodes, n_nodes) - optional

        Returns:
            reconstruction_loss: scalar
            masked_patches: tensor of masked patch indices
        """
        batch, seq_len, n_nodes = consumption.shape

        # Reshape and embed as before
        consumption_flat = consumption.reshape(batch * n_nodes, seq_len, 1)
        weather_expanded = weather.unsqueeze(2).expand(batch, seq_len, n_nodes, -1)
        weather_expanded = weather_expanded.reshape(batch * n_nodes, seq_len, -1)
        contextual_expanded = contextual.unsqueeze(2).expand(batch, seq_len, n_nodes, -1)
        contextual_expanded = contextual_expanded.reshape(batch * n_nodes, seq_len, -1)

        # Multi-modal embedding
        x = self.multimodal_embed(consumption_flat, weather_expanded, contextual_expanded)

        # Temporal patching
        x_patches = self.patching(x)
        num_patches = x_patches.shape[1]

        # Save original patches for reconstruction loss
        x_patches_original = x_patches.detach().clone()

        # Create mask
        mask_ratio = masking_ratio
        num_masked = max(1, int(num_patches * mask_ratio))
        masked_indices = []

        for b in range(batch * n_nodes):
            indices = np.random.choice(num_patches, num_masked, replace=False)
            masked_indices.append(indices)
            x_patches[b, indices] = self.mask_token

        # Transformer + GAT
        x = self.transformer_gat(x_patches, adjacency)

        # Reconstruction head
        x_reconstructed = self.reconstruction_head(x)

        # Compute reconstruction loss (MSE on masked patches)
        loss = 0
        for b in range(batch * n_nodes):
            for idx in masked_indices[b]:
                # Reshape for comparison
                recon = x_reconstructed[b, idx, :].reshape(1, -1)
                original = x_patches_original[b, idx, :].reshape(1, -1)
                loss += F.mse_loss(recon, original)

        loss = loss / (batch * n_nodes * num_masked)

        return loss


class EnergyForecastingModel(nn.Module):
    """Wrapper for supervised forecasting with MMSST backbone."""

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.mmsst = MMSST(
            consumption_dim=config.get('consumption_dim', 1),
            weather_dim=config.get('weather_dim', 3),
            contextual_dim=config.get('contextual_dim', 2),
            embedding_dim=config.get('embedding_dim', 32),
            fusion_dim=config.get('fusion_dim', 64),
            patch_size=config.get('patch_size', 7),
            num_transformer_layers=config.get('num_transformer_layers', 2),
            n_heads=config.get('n_heads', 4),
            ff_dim=config.get('ff_dim', 64),
            gat_heads=config.get('gat_heads', 2),
            dropout=config.get('dropout', 0.1),
        )

    def forward(self, consumption, weather, contextual, adjacency=None):
        return self.mmsst(consumption, weather, contextual, adjacency)

    def masked_forward(self, consumption, weather, contextual, masking_ratio=0.3, adjacency=None):
        return self.mmsst.masked_forward(consumption, weather, contextual, masking_ratio, adjacency)


if __name__ == '__main__':
    # Test instantiation
    config = {
        'consumption_dim': 1,
        'weather_dim': 3,
        'contextual_dim': 2,
        'embedding_dim': 32,
        'fusion_dim': 64,
        'patch_size': 7,
        'num_transformer_layers': 2,
        'n_heads': 4,
        'ff_dim': 64,
        'gat_heads': 2,
        'dropout': 0.1,
    }

    model = EnergyForecastingModel(config)
    print(f"✓ Model instantiated: {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch = 4
    seq_len = 30
    n_nodes = 8

    consumption = torch.randn(batch, seq_len, n_nodes)
    weather = torch.randn(batch, seq_len, 3)
    contextual = torch.randn(batch, seq_len, 2)

    output = model(consumption, weather, contextual)
    print(f"✓ Forward pass: input {consumption.shape} → output {output.shape}")
