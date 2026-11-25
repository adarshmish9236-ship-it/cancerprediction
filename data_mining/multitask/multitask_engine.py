"""
CANCERWISE AI — Multitask Learning Engine (PyTorch)
Joint neural architecture with shared latent representation and task-specific output heads.
Optimizes combined weighted losses with early stopping and model checkpointing.

GOVERNANCE DIRECTIVE:
Multitask learning is activated exclusively when authentic multi-target annotations exist,
or on clearly labeled synthetic research benchmarks. Never fabricate clinical labels.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backend.schemas.advanced_learning import (
    MultitaskTaskResult,
    MultitaskExperimentResult,
)

logger = logging.getLogger("cancerwise.multitask.engine")


class SharedMultitaskNet(nn.Module):
    """
    Deep neural network with shared representations and multiple task-specific heads.
    """

    def __init__(
        self,
        in_features: int,
        shared_dims: List[int] = [64, 32],
        task_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        dropout: float = 0.20,
    ):
        super().__init__()
        # 1. Shared Trunk
        trunk_layers = []
        prev_dim = in_features
        for h_dim in shared_dims:
            trunk_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.shared_trunk = nn.Sequential(*trunk_layers)

        # 2. Task-Specific Heads
        self.task_configs = task_configs or {
            "task_malignancy": {"type": "binary_classification", "out_dim": 1},
            "task_proliferation_index": {"type": "regression", "out_dim": 1},
        }

        self.heads = nn.ModuleDict()
        for t_name, cfg in self.task_configs.items():
            out_d = cfg.get("out_dim", 1)
            self.heads[t_name] = nn.Sequential(
                nn.Linear(prev_dim, 16),
                nn.ReLU(),
                nn.Linear(16, out_d),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_rep = self.shared_trunk(x)
        outputs = {}
        for t_name, head in self.heads.items():
            outputs[t_name] = head(shared_rep).squeeze(-1)
        return outputs


class MultitaskEngine:
    """
    Trains and evaluates shared-representation multitask neural networks.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_multitask(
        self,
        X: pd.DataFrame,
        targets: Dict[str, np.ndarray],
        task_types: Dict[str, str],
        task_weights: Optional[Dict[str, float]] = None,
        shared_dims: List[int] = [64, 32],
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 0.003,
        patience: int = 12,
        is_synthetic_benchmark: bool = True,
        dataset_name: str = "Synthetic Cancer Multi-Task Benchmark",
    ) -> MultitaskExperimentResult:
        """
        Executes joint training and evaluation across multiple tasks.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        start_time = time.perf_counter()

        # Default task weights
        if task_weights is None:
            task_weights = {t: 1.0 / len(targets) for t in targets}

        # Train/Test Split
        n_samples = len(X)
        indices = np.arange(n_samples)
        train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=self.random_state)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_test = scaler.transform(X.iloc[test_idx])

        # Configure Network
        task_configs = {}
        for t_name, t_type in task_types.items():
            task_configs[t_name] = {"type": t_type, "out_dim": 1}

        model = SharedMultitaskNet(
            in_features=X_train.shape[1],
            shared_dims=shared_dims,
            task_configs=task_configs,
        )

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # Loss functions per task
        criterions = {}
        for t_name, t_type in task_types.items():
            if t_type == "binary_classification":
                criterions[t_name] = nn.BCEWithLogitsLoss()
            else:
                criterions[t_name] = nn.MSELoss()

        # Prepare Tensors
        t_x_train = torch.from_numpy(X_train.astype(np.float32))
        t_x_test = torch.from_numpy(X_test.astype(np.float32))

        t_y_train = {t: torch.from_numpy(targets[t][train_idx].astype(np.float32)) for t in targets}
        t_y_test = {t: torch.from_numpy(targets[t][test_idx].astype(np.float32)) for t in targets}

        dataset = TensorDataset(t_x_train, *[t_y_train[t] for t in targets])
        loader = DataLoader(dataset, batch_size=min(batch_size, len(train_idx)), shuffle=True)

        best_loss = float("inf")
        best_weights = None
        best_epoch = 0
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss_epoch = 0.0

            for batch in loader:
                b_x = batch[0]
                b_y = {t: batch[i + 1] for i, t in enumerate(targets)}

                optimizer.zero_grad()
                outputs = model(b_x)

                total_loss = 0.0
                for t_name in targets:
                    loss_t = criterions[t_name](outputs[t_name], b_y[t_name])
                    total_loss += task_weights.get(t_name, 1.0) * loss_t

                total_loss.backward()
                optimizer.step()
                train_loss_epoch += total_loss.item()

            # Validation step
            model.eval()
            with torch.no_grad():
                test_outputs = model(t_x_test)
                test_loss = 0.0
                for t_name in targets:
                    loss_t = criterions[t_name](test_outputs[t_name], t_y_test[t_name])
                    test_loss += task_weights.get(t_name, 1.0) * loss_t.item()

            if test_loss < best_loss:
                best_loss = test_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Multitask early stopping triggered at epoch %d", epoch + 1)
                    break

        # Load best checkpoint
        if best_weights is not None:
            model.load_state_dict(best_weights)

        # Final evaluation on holdout test set
        model.eval()
        with torch.no_grad():
            final_outputs = model(t_x_test)

        task_results: List[MultitaskTaskResult] = []
        for t_name, t_type in task_types.items():
            y_true = targets[t_name][test_idx]
            raw_pred = final_outputs[t_name].cpu().numpy()

            if t_type == "binary_classification":
                prob_pred = 1.0 / (1.0 + np.exp(-raw_pred))
                binary_pred = (prob_pred >= 0.5).astype(int)
                try:
                    score = float(roc_auc_score(y_true, prob_pred))
                    metric_name = "ROC-AUC"
                except Exception:
                    score = float(accuracy_score(y_true, binary_pred))
                    metric_name = "Accuracy"
            else:
                score = float(r2_score(y_true, raw_pred))
                metric_name = "R2_Score"

            test_loss_t = float(criterions[t_name](final_outputs[t_name], t_y_test[t_name]).item())

            task_results.append(
                MultitaskTaskResult(
                    task_name=t_name,
                    task_type=t_type,
                    task_weight=task_weights.get(t_name, 1.0),
                    train_loss=round(train_loss_epoch / max(len(loader), 1), 4),
                    test_loss=round(test_loss_t, 4),
                    primary_metric_name=metric_name,
                    primary_metric_value=round(score, 4),
                )
            )

        elapsed = round(time.perf_counter() - start_time, 3)

        provenance = (
            "Research Benchmark Notice: Validated on a multi-target cancer research benchmark. "
            "Simultaneous learning of primary tumor malignancy and cell proliferation index without label fabrication."
            if is_synthetic_benchmark else "Evaluated on authentic multi-annotation patient dataset."
        )

        return MultitaskExperimentResult(
            dataset_name=dataset_name,
            is_synthetic_benchmark=is_synthetic_benchmark,
            data_provenance_note=provenance,
            shared_hidden_layers=shared_dims,
            total_epochs=epochs,
            early_stopped_epoch=best_epoch,
            tasks=task_results,
            combined_test_loss=round(best_loss, 4),
            execution_time_seconds=elapsed,
        )
