from pathlib import Path
from typing import Self

import polars as pl
import pytest

from models.base_model import BaseForecaster

# --- DUMMY CLASS TO TEST THE ABSTRACT BASE CLASS ---


class DummyForecaster(BaseForecaster):
    """Minimal implementation to test BaseForecaster logic in isolation."""

    def train(self, df: pl.DataFrame) -> None:
        self._is_trained = True

    def predict(self, df: pl.DataFrame) -> pl.Series:
        # Rely on the base class safeguard method
        self._guard_is_trained("predict")
        return pl.Series("prediction", [1, 2, 3])

    def save(self, path: str | Path) -> None:
        pass

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls("dummy_target")


# --- TESTS ---


def test_base_forecaster_init() -> None:
    """Ensures initialization sets default properties correctly."""
    model = DummyForecaster(target_col="sales")

    assert model.target_col == "sales"
    assert model.is_trained is False


def test_guard_is_trained_blocks_before_train() -> None:
    """Ensures the safeguard blocks method calls if the model is untrained."""
    model = DummyForecaster(target_col="sales")

    with pytest.raises(RuntimeError, match=r"predict\(\) called before train\(\)"):
        model.predict(pl.DataFrame())


def test_guard_is_trained_passes_after_train() -> None:
    """Ensures the safeguard allows method calls after the model is trained."""
    model = DummyForecaster(target_col="sales")

    model.train(pl.DataFrame())  # Simulate training
    assert model.is_trained is True

    # Should not raise a RuntimeError
    preds = model.predict(pl.DataFrame())
    assert len(preds) == 3
