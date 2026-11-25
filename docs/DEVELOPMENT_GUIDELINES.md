# CANCERWISE AI — Development Guidelines

## 1. Code Style

- **Language:** Python 3.12+ (targeting 3.14)
- **Formatter:** `ruff format` (line length: 100)
- **Linter:** `ruff check`
- **Type checker:** `mypy` (strict mode recommended)
- All public functions must have **docstrings** and **type annotations**

```python
# Good
def evaluate_model(
    model: BaseModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate a fitted model on test data.

    Args:
        model:     A fitted BaseModel instance.
        X_test:    Test features.
        y_test:    True labels.
        threshold: Decision threshold for binary classification.

    Returns:
        Dictionary of metric name → value.
    """
    ...
```

---

## 2. Project Architecture Rules

### Separation of Concerns
- `backend/api/` — HTTP routing ONLY. No business logic.
- `backend/services/` — Business logic. No HTTP concerns.
- `backend/schemas/` — Pydantic validation ONLY.
- `backend/core/` — Shared utilities (config, logging, base classes).
- `data_mining/` — ML algorithms. No API or UI code.
- `preprocessing/` — Data transformation. No model training.
- `evaluation/` — Metrics computation. No data loading.
- `explainability/` — Interpretation. Must be model-agnostic.

### No Hardcoded Values
❌ `path = "C:/Users/DELL/data/cancer.csv"`  
✅ `path = Path(config.get("paths", "data", "raw")) / "cancer.csv"`

❌ `random_state = 42` (inline)  
✅ `random_state = config.get("data_mining", "random_seed")`

---

## 3. Configuration Management

All tuneable parameters live in `configs/settings.yaml`. Access via:

```python
from backend.core.config import get_config
config = get_config()
seed = config.get("data_mining", "random_seed")  # → 42
```

Environment overrides go in `.env` (never commit `.env`).

---

## 4. Logging

```python
from backend.core.logging_setup import get_logger
logger = get_logger(__name__)

logger.debug("Feature matrix shape: %s", X.shape)
logger.info("Model training started | model=%s", model.name)
logger.warning("Low sample count in class %d: %d samples", cls, count)
logger.error("Training failed: %s", e, exc_info=True)
```

**Rules:**
- Use `%s` formatting (not f-strings) in logger calls for performance.
- Never log raw patient data, file contents, or passwords.
- ERROR level must include `exc_info=True` for exceptions.

---

## 5. Model Development

All models must:
1. Inherit from `backend.core.base_model.BaseModel`
2. Implement `fit()`, `predict()`, `save()`, and `load()` 
3. Set `self._is_fitted = True` in `fit()`
4. Log training start, completion, and key metrics
5. Be registered in `models/registry/`

```python
from backend.core.base_model import BaseModel

class MyClassifier(BaseModel):
    def fit(self, X, y, **kwargs):
        logger.info("Fitting %s on %d samples", self.name, len(y))
        # ... training logic ...
        self._is_fitted = True
        return self
```

---

## 6. Testing

- All new features must have corresponding tests in `tests/`
- Minimum test coverage: **80%** (aim for 90%+)
- Run tests before every commit: `pytest tests/ -v --tb=short`
- Use `pytest.mark.slow` for long-running tests (model training)
- Mock external services and file I/O where possible

```python
# Mark slow tests
@pytest.mark.slow
def test_full_training_pipeline():
    ...
```

---

## 7. Experiment Reproducibility

Every experiment must:
- Set a fixed `random_seed` from config
- Log all hyperparameters to MLflow
- Save the trained model artifact
- Record the dataset version and preprocessing config

```python
import mlflow

with mlflow.start_run(run_name="xgboost_v1"):
    mlflow.log_params({"n_estimators": 200, "max_depth": 6})
    mlflow.log_metrics({"roc_auc": 0.94, "f1": 0.89})
    mlflow.sklearn.log_model(model, "model")
```

---

## 8. File Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Python modules | `snake_case.py` | `random_forest.py` |
| Classes | `PascalCase` | `RandomForestClassifier` |
| Functions/vars | `snake_case` | `train_model()` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_EPOCHS = 100` |
| Config files | `snake_case.yaml` | `settings.yaml` |
| Notebooks | `NNN_description.ipynb` | `001_eda_breast_cancer.ipynb` |
| Data files | `YYYYMMDD_description.csv` | `20261122_breast_cancer_raw.csv` |

---

## 9. Git Workflow

```
main          ← stable, production-ready
  └─ develop  ← integration branch
       ├─ feature/phase1-eda
       ├─ feature/xgboost-classifier
       └─ fix/preprocessing-nulls
```

Commit message format:
```
<type>(<scope>): <short description>

feat(preprocessing): add KNN imputer support
fix(api): handle empty prediction request body
docs(readme): update Phase 1 roadmap
test(evaluation): add multi-class metrics tests
refactor(config): switch to Pydantic settings
```

---

## 10. Documentation Requirements

- Every new module must have a module-level docstring
- Every public class must have a class docstring
- Complex algorithms must include references (paper, URL)
- Architecture decisions go in `docs/`
- Inline comments for non-obvious logic only
