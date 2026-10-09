#!/usr/bin/env python3
"""
Training script for MM-SST: Multi-Modal Self-Supervised Transformer.

Two-stage training:
  1. Self-supervised pre-training with masked reconstruction
  2. Supervised fine-tuning for forecasting

Usage:
  python train.py --dataset sim_d --output-dir ./results
"""

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import argparse
import logging

from mm_sst import EnergyForecastingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)


class EnergyDataset:
    """Load and preprocess energy forecasting dataset."""

    def __init__(self, csv_path: str, lookback: int = 30, horizon: int = 7):
        self.df = pd.read_csv(csv_path)
        self.lookback = lookback
        self.horizon = horizon

        # Parse date
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Extract consumption, weather, contextual columns
        self.consumption_cols = [col for col in self.df.columns if col.startswith('Building_') or col.startswith('Zone_')]
        self.weather_cols = ['Temperature', 'Humidity', 'SolarIrradiance']
        self.contextual_cols = ['Month', 'DayOfWeek']

        # Identify occupancy columns (if present)
        occupancy_cols = [col for col in self.df.columns if col.startswith('Occupancy_')]
        if occupancy_cols:
            self.contextual_cols += occupancy_cols

        # Normalize
        self._normalize()

        logger.info(f"Dataset loaded: {len(self.df)} samples, {len(self.consumption_cols)} nodes")
        logger.info(f"Features: {len(self.consumption_cols)} consumption, {len(self.weather_cols)} weather, {len(self.contextual_cols)} contextual")

    def _normalize(self):
        """Normalize data with training-set statistics (computed later)."""
        # This will be called after train/test split
        pass

    def create_sequences(self, start_idx: int = 0, end_idx: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create lookback/horizon sequences."""
        if end_idx is None:
            end_idx = len(self.df)

        X_consumption, X_weather, X_contextual, y = [], [], [], []

        for i in range(start_idx, end_idx - self.lookback - self.horizon):
            # Lookback window
            consumption = self.df.iloc[i:i+self.lookback][self.consumption_cols].values
            weather = self.df.iloc[i:i+self.lookback][self.weather_cols].values
            contextual = self.df.iloc[i:i+self.lookback][self.contextual_cols].values

            # Horizon window (target)
            consumption_target = self.df.iloc[i+self.lookback:i+self.lookback+self.horizon][self.consumption_cols].values

            X_consumption.append(consumption)
            X_weather.append(weather)
            X_contextual.append(contextual)
            y.append(consumption_target[0])  # First step of horizon

        X_consumption = torch.FloatTensor(np.array(X_consumption))
        X_weather = torch.FloatTensor(np.array(X_weather))
        X_contextual = torch.FloatTensor(np.array(X_contextual))
        y = torch.FloatTensor(np.array(y))

        return (X_consumption, X_weather, X_contextual), y

    def get_split_indices(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Get train/val/test split indices."""
        n = len(self.df)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        return (0, train_end), (train_end, val_end), (val_end, n)


def load_graph(graph_path: str) -> np.ndarray:
    """Load adjacency matrix from graph JSON."""
    with open(graph_path) as f:
        graph_data = json.load(f)

    adjacency = np.array(graph_data['adjacency_matrix'])
    return adjacency


def train_step(model, optimizer, batch, adjacency, device):
    """One training step."""
    (X_consumption, X_weather, X_contextual), y = batch

    X_consumption = X_consumption.to(device)
    X_weather = X_weather.to(device)
    X_contextual = X_contextual.to(device)
    y = y.to(device)

    # Forward pass
    predictions = model(X_consumption, X_weather, X_contextual, adjacency)

    # Loss (MAE + MSE)
    loss = nn.L1Loss()(predictions, y) + 0.1 * nn.MSELoss()(predictions, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def pretrain_step(model, optimizer, batch, adjacency, device, masking_ratio=0.3):
    """One pre-training step (masked reconstruction)."""
    (X_consumption, X_weather, X_contextual), _ = batch

    X_consumption = X_consumption.to(device)
    X_weather = X_weather.to(device)
    X_contextual = X_contextual.to(device)

    # Masked forward
    loss = model.masked_forward(
        X_consumption, X_weather, X_contextual,
        masking_ratio=masking_ratio, adjacency=adjacency
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, data_loader, adjacency, device):
    """Evaluate model on a dataset."""
    model.eval()

    total_mae = 0
    total_rmse = 0
    total_mape = 0
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            (X_consumption, X_weather, X_contextual), y = batch

            X_consumption = X_consumption.to(device)
            X_weather = X_weather.to(device)
            X_contextual = X_contextual.to(device)
            y = y.to(device)

            predictions = model(X_consumption, X_weather, X_contextual, adjacency)

            mae = torch.abs(predictions - y).mean().item()
            rmse = torch.sqrt(torch.pow(predictions - y, 2).mean()).item()
            mape = (torch.abs(predictions - y) / (torch.abs(y) + 1e-8)).mean().item() * 100

            total_mae += mae
            total_rmse += rmse
            total_mape += mape
            count += 1

    model.train()

    return {
        'mae': total_mae / count,
        'rmse': total_rmse / count,
        'mape': total_mape / count,
    }


def train_model(args):
    """Main training pipeline."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    dataset = EnergyDataset(
        args.data_path,
        lookback=args.lookback,
        horizon=args.horizon
    )

    train_indices, val_indices, test_indices = dataset.get_split_indices()

    # Create sequences
    logger.info("Creating sequences...")
    X_train, y_train = dataset.create_sequences(*train_indices)
    X_val, y_val = dataset.create_sequences(*val_indices)
    X_test, y_test = dataset.create_sequences(*test_indices)

    # Data loaders
    train_dataset = list(zip(
        zip(X_train[0], X_train[1], X_train[2]), y_train
    ))
    val_dataset = list(zip(
        zip(X_val[0], X_val[1], X_val[2]), y_val
    ))
    test_dataset = list(zip(
        zip(X_test[0], X_test[1], X_test[2]), y_test
    ))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Load graph
    graph_path = Path(args.data_path).parent / f"{args.dataset}_graph.json"
    adjacency = load_graph(graph_path)
    adjacency = torch.FloatTensor(adjacency).to(device) if args.use_gat else None

    # Model config
    model_config = {
        'consumption_dim': len(dataset.consumption_cols),
        'weather_dim': len(dataset.weather_cols),
        'contextual_dim': len(dataset.contextual_cols),
        'embedding_dim': args.embedding_dim,
        'fusion_dim': args.fusion_dim,
        'patch_size': args.patch_size,
        'num_transformer_layers': args.num_layers,
        'n_heads': args.n_heads,
        'ff_dim': args.ff_dim,
        'gat_heads': args.gat_heads,
        'dropout': args.dropout,
    }

    # Create model
    model = EnergyForecastingModel(model_config).to(device)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs + args.finetune_epochs)

    # Training logs
    logs = {
        'pretrain_loss': [],
        'train_loss': [],
        'val_metrics': [],
        'test_metrics': [],
    }

    # ========== STAGE 1: Self-Supervised Pre-Training ==========
    logger.info("\n=== Stage 1: Self-Supervised Pre-Training ===")
    model.train()

    best_pretrain_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.pretrain_epochs):
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            loss = pretrain_step(
                model, optimizer, batch, adjacency, device,
                masking_ratio=args.masking_ratio
            )
            epoch_loss += loss
            scheduler.step()

        epoch_loss /= len(train_loader)
        logs['pretrain_loss'].append(epoch_loss)

        if epoch % 2 == 0 or epoch == args.pretrain_epochs - 1:
            logger.info(f"Epoch {epoch+1}/{args.pretrain_epochs} - Loss: {epoch_loss:.4f}")

        # Early stopping
        if epoch_loss < best_pretrain_loss:
            best_pretrain_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # ========== STAGE 2: Supervised Fine-Tuning ==========
    logger.info("\n=== Stage 2: Supervised Fine-Tuning ===")
    model.train()

    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(args.finetune_epochs):
        epoch_loss = 0

        for batch in train_loader:
            loss = train_step(model, optimizer, batch, adjacency, device)
            epoch_loss += loss
            scheduler.step()

        epoch_loss /= len(train_loader)
        logs['train_loss'].append(epoch_loss)

        # Validation
        val_metrics = evaluate(model, val_loader, adjacency, device)
        logs['val_metrics'].append(val_metrics)

        if epoch % 2 == 0 or epoch == args.finetune_epochs - 1:
            logger.info(
                f"Epoch {epoch+1}/{args.finetune_epochs} - "
                f"Train Loss: {epoch_loss:.4f}, Val MAE: {val_metrics['mae']:.4f}"
            )

        # Early stopping
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # ========== EVALUATION ==========
    logger.info("\n=== Test Evaluation ===")
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_metrics = evaluate(model, test_loader, adjacency, device)
    logs['test_metrics'] = test_metrics

    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Test MAPE: {test_metrics['mape']:.2f}%")

    # Save logs
    with open(output_dir / 'training_logs.json', 'w') as f:
        # Convert tensors to lists for JSON serialization
        json.dump({
            'pretrain_loss': logs['pretrain_loss'],
            'train_loss': logs['train_loss'],
            'val_metrics': logs['val_metrics'],
            'test_metrics': logs['test_metrics'],
        }, f, indent=2)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    logger.info(f"\n✅ Training complete! Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MM-SST for energy forecasting')

    # Data
    parser.add_argument('--dataset', default='sim_d', choices=['sim_d', 'sim_b'])
    parser.add_argument('--data-path', help='Path to dataset CSV (auto-set if not provided)')
    parser.add_argument('--lookback', type=int, default=30, help='Lookback window')
    parser.add_argument('--horizon', type=int, default=7, help='Forecast horizon')

    # Model
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--fusion-dim', type=int, default=64)
    parser.add_argument('--patch-size', type=int, default=7)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--ff-dim', type=int, default=64)
    parser.add_argument('--gat-heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use-gat', action='store_true', default=True)

    # Training
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--pretrain-epochs', type=int, default=8)
    parser.add_argument('--finetune-epochs', type=int, default=15)
    parser.add_argument('--masking-ratio', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=5)

    # Output
    parser.add_argument('--output-dir', default='./results')

    args = parser.parse_args()

    # Auto-set data path
    if args.data_path is None:
        args.data_path = f'./data/{args.dataset}_dataset.csv'

    train_model(args)
