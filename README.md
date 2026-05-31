# regression-forecaster

![CI](https://github.com/MaximeRighini/regression-forecaster/actions/workflows/ci.yaml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)

A clean reference implementation for regression forecasting, built around a consistent model abstraction.

The core idea: `BaseForecaster` defines a minimal interface (`train`, `predict`, `save`, `load`) that any model must implement, regardless of its internals. \
`GradientBoostingForecaster` is the first concrete implementation. \
Adding a new one — a rolling average, a Prophet wrapper, an LSTM — requires implementing the same four methods without touching anything downstream.

---

## Table of Contents

- [Installation](#installation)
- [BaseForecaster](#baseforecaster)
- [GradientBoostingForecaster](#gradientboostingforecaster)
- [Evaluation & Reporting](#evaluation--reporting)
- [Code Quality](#code-quality)

---

## Installation

Copy `src/models/` and `src/evaluation/` into your project and add the dependencies to your `pyproject.toml`:

```toml
dependencies = [
    "polars>=1.0.0",
    "xgboost>=2.0.0",
    "shap>=0.45.0",
]
```

---

## BaseForecaster

Any model in this codebase inherits from `BaseForecaster` and implements the same interface.

```python
class BaseForecaster(ABC):
    def train(self, df: pl.DataFrame) -> None: ...
    def predict(self, df: pl.DataFrame) -> pl.Series: ...
    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> Self: ...
```

Downstream code — pipelines, evaluation scripts, nodes — only ever depends on this interface, not on any specific implementation.

---

## GradientBoostingForecaster

XGBoost-based implementation with categorical encoding, optional ID-level normalization, and SHAP explainability.

```python
from models.gradient_boosting import GradientBoostingForecaster

model = GradientBoostingForecaster(
    target_col="consumption_kwh",
    feature_cols=["hour", "day_of_week", "temperature", "meter_type"],
    id_col="meter_id",
    normalize=True,
    xgb_params={"n_estimators": 500, "learning_rate": 0.05},
)

model.train(df_train)

predictions = model.predict(df_test)                # -> pl.Series
importances = model.feature_importance()            # -> pl.DataFrame sorted by importance
shap_values = model.explain(df_test)                # -> pl.DataFrame of SHAP contributions

model.save("models/gb_forecaster.pkl")
model = GradientBoostingForecaster.load("models/gb_forecaster.pkl")
```

**Categorical encoding:** String and Categorical columns are label-encoded at train time. Encodings are persisted and reused at predict time. Unseen categories raise immediately with a structured error listing every problematic value.

**ID-level normalization:** When `normalize=True`, the target is divided by `mean(target_col)` per `id_col` before training, then predictions are multiplied back at inference. This helps the model learn patterns shared across entities whose targets differ only in magnitude. Normalization weights are computed exclusively on the training data.

**Serialization:** `save()` / `load()` persist the full model state — XGBoost model, categorical encodings, normalization weights, feature columns — so a loaded model is immediately ready to predict.

---

## Evaluation & Reporting

```python
from evaluation.model_report import compute_report

breakdown_config = {
    "overall":                [],
    "by_market":              ["market"],
    "by_category":            ["category"],
    "by_market_and_category": ["market", "category"],
}

report = compute_report(df, "y_true", "y_pred", breakdown_config)
report.write_excel("evaluation/report.xlsx")
```

`compute_report` returns a flat DataFrame with one row per group per breakdown, ready to explore in Excel. Group columns not relevant to a given breakdown are filled with null so the schema stays consistent across all rows.

---

## Code Quality

Common tasks are available via `make` to simplify the developer experience.

```bash
make lint-fix      # Auto-fix formatting, style, and import order
make lint-verify   # Read-only checks — what CI runs
make test          # Run unit tests
make all           # lint-fix → lint-verify → test
make clean         # Remove all cache directories
```

This package enforces code quality at three stages to keep the codebase clean
and ensure that what works locally also works in CI.

1. **`make lint-verify`** runs Ruff and Mypy in read-only mode — catch style and type errors early.
2. **Pre-commit hooks** ensure badly formatted or broken code never reaches the remote repository.
3. **GitHub Actions** triggers on every push and blocks any pull request that fails linting or tests.
