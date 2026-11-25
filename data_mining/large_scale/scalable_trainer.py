"""
CANCERWISE AI — Scalable PyTorch Training Engine
High-throughput mini-batch deep learning with DataLoader optimization,
GPU/CPU device orchestration, ReduceLROnPlateau scheduling, early stopping,
model checkpointing, and real-time hardware telemetry.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backend.core.config import PROJECT_ROOT
from backend.schemas.large_scale import (
    DeviceType,
    ScalableEpochMetric,
    ScalableTrainingResult,
)
from data_mining.large_scale.performance_monitor import PerformanceMonitor

logger = logging.getLogger("cancerwise.large_scale.trainer")


class ScalableDeepNet(nn.Module):
    """
    Scalable deep multilayer perceptron for large-scale biomedical classification.
    """

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class ScalableTrainingEngine:
    """
    Manages scalable PyTorch mini-batch training with hardware acceleration.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        random_state: int = 42,
        monitor: Optional[PerformanceMonitor] = None,
    ):
        self.checkpoint_dir = checkpoint_dir or (PROJECT_ROOT / "models" / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.monitor = monitor or PerformanceMonitor()

    def train_scalable(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        batch_size: int = 64,
        max_epochs: int = 60,
        lr: float = 0.003,
        patience: int = 10,
        device_override: Optional[str] = None,
        model_name: str = "ScalableDeepNet",
    ) -> ScalableTrainingResult:
        """
        Executes scalable mini-batch training with learning rate scheduling and early stopping.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        overall_start = time.perf_counter()

        # 1. Device Selection (CUDA vs MPS vs CPU)
        if device_override:
            dev_str = device_override
        elif torch.cuda.is_available():
            dev_str = "cuda"
        else:
            dev_str = "cpu"

        device = torch.device(dev_str)
        dev_enum = DeviceType.CUDA if "cuda" in dev_str else DeviceType.CPU

        # 2. Prepare Data & Stratified Holdout
        X_num = X.select_dtypes(include=[np.number]).dropna(axis=1)
        if not pd.api.types.is_numeric_dtype(y):
            mapping = {"M": 1, "B": 0, "malignant": 1, "benign": 0}
            y_arr = y.map(mapping).to_numpy() if all(k in mapping for k in y.unique()) else pd.factorize(y)[0]
        else:
            y_arr = y.to_numpy().astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X_num, y_arr, test_size=0.20, stratify=y_arr, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)

        # 3. PyTorch Dataset & DataLoader
        pin_memory = (dev_enum == DeviceType.CUDA)
        train_ds = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train.astype(np.float32)))
        train_loader = DataLoader(
            train_ds,
            batch_size=min(batch_size, len(X_train_scaled)),
            shuffle=True,
            pin_memory=pin_memory,
        )

        t_x_val = torch.from_numpy(X_val_scaled).to(device)
        t_y_val = torch.from_numpy(y_val.astype(np.float32)).to(device)

        # 4. Model, Loss, Optimizer, and LR Scheduler
        in_features = X_train_scaled.shape[1]
        model = ScalableDeepNet(in_features=in_features).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0
        history: List[ScalableEpochMetric] = []
        throughputs = []

        checkpoint_path = self.checkpoint_dir / f"{model_name}_best.pt"

        # 5. Mini-Batch Training Loop
        for epoch in range(1, max_epochs + 1):
            ep_start = time.perf_counter()
            model.train()
            train_loss_accum = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss_accum += loss.item()

            train_loss = train_loss_accum / max(len(train_loader), 1)

            # Validation evaluation
            model.eval()
            with torch.no_grad():
                val_logits = model(t_x_val)
                val_loss = criterion(val_logits, t_y_val).item()
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_preds = (val_probs >= 0.5).astype(int)
                val_acc = float(accuracy_score(y_val, val_preds))

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            ep_duration = max(time.perf_counter() - ep_start, 1e-5)
            throughput = len(X_train_scaled) / ep_duration
            throughputs.append(throughput)

            history.append(
                ScalableEpochMetric(
                    epoch=epoch,
                    train_loss=round(train_loss, 4),
                    val_loss=round(val_loss, 4),
                    val_accuracy=round(val_acc, 4),
                    learning_rate=current_lr,
                    epoch_duration_seconds=round(ep_duration, 4),
                    throughput_samples_per_sec=round(throughput, 1),
                )
            )

            # Checkpoint saving & early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
                torch.save(best_state, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, patience)
                    break

        # Load best checkpoint for final evaluation
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            final_logits = model(t_x_val)
            final_probs = torch.sigmoid(final_logits).cpu().numpy()
            final_preds = (final_probs >= 0.5).astype(int)
            final_acc = float(accuracy_score(y_val, final_preds))
            try:
                final_auc = float(roc_auc_score(y_val, final_probs))
            except Exception:
                final_auc = final_acc

        total_time = round(time.perf_counter() - overall_start, 4)
        avg_throughput = round(float(np.mean(throughputs)), 1) if throughputs else 0.0

        hardware = self.monitor.get_system_snapshot()

        return ScalableTrainingResult(
            model_name=model_name,
            device_used=dev_enum,
            batch_size=batch_size,
            total_epochs=len(history),
            early_stopped_epoch=best_epoch,
            best_val_loss=round(best_val_loss, 4),
            final_val_accuracy=round(final_acc, 4),
            final_val_roc_auc=round(final_auc, 4),
            total_training_time_seconds=total_time,
            average_throughput_samples_per_sec=avg_throughput,
            checkpoint_file=str(checkpoint_path),
            epoch_history=history,
            hardware_status=hardware,
        )
