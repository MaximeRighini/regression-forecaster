"""
BaseForecaster: abstract contract for all forecasting models.

Any model in this codebase inherits from BaseForecaster and implements
the same interface: train, predict, save, load.

Design principles
-----------------
- Minimal contract. The base class defines only what every model must support,
  nothing more. Preprocessing, normalization, and encoding are implementation
  details that belong in each subclass.
- Consistent interface. Adding a new model (RollingAverage, Prophet, LSTM)
  requires implementing the same four methods. Downstream code never changes.
- Fail fast. Models enforce that train() is called before predict().
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import polars as pl


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.

    Parameters
    ----------
    target_col:
        Name of the column to predict.
    """

    def __init__(self, target_col: str) -> None:
        self._target_col = target_col
        self._is_trained: bool = False

    @property
    def target_col(self) -> str:
        """Returns the name of the target column."""
        return self._target_col

    @property
    def is_trained(self) -> bool:
        """Returns True if the model has been trained."""
        return self._is_trained

    @abstractmethod
    def train(self, df: pl.DataFrame) -> None:
        """Fits the model on df."""
        ...

    @abstractmethod
    def predict(self, df: pl.DataFrame) -> pl.Series:
        """
        Generates predictions for df.

        Raises
        ------
        RuntimeError
            If called before train().
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serializes the model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Loads a serialized model from disk."""
        ...

    def _guard_is_trained(self, method_name: str) -> None:
        """Raises RuntimeError if the model has not been trained yet."""
        if not self._is_trained:
            raise RuntimeError(
                f"{method_name}() called before train(). Call train(df) first."
            )
