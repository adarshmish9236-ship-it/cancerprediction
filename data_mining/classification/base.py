"""
CANCERWISE AI — Base Model Interface & PyTorch MLP Classifier
Defines standard model contract and Scikit-learn compliant PyTorch classifier.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("cancerwise.classification.base")


class PyTorchMLPNet(nn.Module):
    """Deep Multi-Layer Perceptron architecture for biomedical classification."""

    def __init__(self, in_features: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible PyTorch neural network classifier.
    Seamlessly integrates with Pipeline, GridSearchCV, and CrossValidation.
    """

    def __init__(
        self,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        epochs: int = 60,
        batch_size: int = 32,
        dropout: float = 0.2,
        random_state: int = 42,
    ):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state
        self.classes_ = np.array([0, 1])
        self.model_: Optional[PyTorchMLPNet] = None
        self.in_features_: Optional[int] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "PyTorchMLPClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        self.in_features_ = X_arr.shape[1]
        self.classes_ = np.unique(y_arr).astype(int)

        self.model_ = PyTorchMLPNet(
            in_features=self.in_features_,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            dropout=self.dropout,
        )

        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(X_arr)), shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted.")

        X_arr = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            t_x = torch.from_numpy(X_arr)
            logits = self.model_(t_x)
            probs_malignant = torch.sigmoid(logits).cpu().numpy()

        probs_malignant = np.clip(probs_malignant, 1e-7, 1.0 - 1e-7)
        probs_benign = 1.0 - probs_malignant
        return np.column_stack([probs_benign, probs_malignant])

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
