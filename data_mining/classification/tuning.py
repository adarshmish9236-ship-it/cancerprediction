"""
CANCERWISE AI — Hyperparameter Optimization Module
Implements GridSearchCV, RandomizedSearchCV, and Optuna optimization with Stratified K-Fold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from backend.schemas.modeling import TuningStrategy

logger = logging.getLogger("cancerwise.classification.tuning")


class HyperparameterOptimizer:
    """
    Automated hyperparameter tuner for classification models.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def optimize(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, Any],
        X,
        y,
        strategy: TuningStrategy = TuningStrategy.GRID_SEARCH,
        n_iter: int = 15,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
    ) -> Tuple[BaseEstimator, Dict[str, Any], float]:
        """
        Runs hyperparameter search and returns (best_estimator, best_params, best_score).
        """
        if not param_grid or strategy == TuningStrategy.NONE:
            estimator.fit(X, y)
            return estimator, {}, 0.0

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        if strategy == TuningStrategy.GRID_SEARCH:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y)
            return search.best_estimator_, search.best_params_, float(search.best_score_)

        elif strategy == TuningStrategy.RANDOMIZED_SEARCH:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=min(n_iter, 50),
                scoring=scoring,
                cv=cv,
                random_state=self.random_state,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y)
            return search.best_estimator_, search.best_params_, float(search.best_score_)

        elif strategy == TuningStrategy.OPTUNA:
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def objective(trial):
                    trial_params = {}
                    for param_name, choices in param_grid.items():
                        if isinstance(choices, list):
                            trial_params[param_name] = trial.suggest_categorical(param_name, choices)

                    clf = estimator.set_params(**trial_params)
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
                    return float(scores.mean())

                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
                study.optimize(objective, n_trials=min(n_iter, 30))
                best_params = study.best_params
                best_estimator = estimator.set_params(**best_params)
                best_estimator.fit(X, y)
                return best_estimator, best_params, float(study.best_value)

            except ImportError:
                logger.warning("Optuna not installed, falling back to RandomizedSearchCV.")
                return self.optimize(
                    estimator, param_grid, X, y, strategy=TuningStrategy.RANDOMIZED_SEARCH, n_iter=n_iter, cv_folds=cv_folds
                )

        else:
            estimator.fit(X, y)
            return estimator, {}, 0.0
