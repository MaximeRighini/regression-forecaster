"""
GradientBoostingForecaster: XGBoost-based forecasting model.

Implements the BaseForecaster interface with full support for:
- Categorical encoding (label encoding, persisted at train time)
- Optional ID-level normalization (weights computed at train time,
  applied at predict time)
- SHAP-based model explainability
- Model serialization and deserialization

Design principles
-----------------
- No data leakage. Categorical encodings and normalization weights are computed
  exclusively on the training set and reused as-is at predict time.
- Explicit failure. Unseen categories and unknown IDs raise immediately with
  actionable error messages listing every problematic value.
- Config over code. XGBoost hyperparameters are passed as a plain dict,
  keeping the class decoupled from any specific parameter set.
"""

import pickle
from pathlib import Path
from typing import Any, Literal, Self

import polars as pl
import polars.selectors as cs
import shap
import xgboost as xgb

from models.base_model import BaseForecaster


class GradientBoostingForecaster(BaseForecaster):
    """
    XGBoost-based forecasting model with preprocessing and normalization.

    Parameters
    ----------
    target_col:
        Name of the column to predict.
    feature_cols:
        Feature columns passed to XGBoost. Must be specified explicitly —
        no auto-resolution from DataFrame columns.
    id_col:
        Column identifying each time series entity (e.g. meter_id, sku).
        Required if normalize=True.
    normalize:
        If True, divides the target by mean(target_col) per id_col at train time,
        then multiplies predictions back at predict time. Helps the model learn
        shared temporal patterns across IDs whose targets differ only in magnitude.
        Requires id_col to be set.
    xgb_params:
        XGBoost hyperparameters passed directly to xgb.XGBRegressor.
        Defaults to a sensible starting point for regression tasks.

    Example
    -------
    ::

        model = GradientBoostingForecaster(
            target_col="consumption_kwh",
            feature_cols=["hour", "day_of_week", "temperature", "meter_type"],
            id_col="meter_id",
            normalize=True,
            xgb_params={"n_estimators": 500, "learning_rate": 0.05},
        )

        model.train(df_train)
        predictions  = model.predict(df_test)
        importances  = model.feature_importance()
        shap_values  = model.explain(df_test)
        model.save("models/gb_forecaster.pkl")
    """

    _DEFAULT_XGB_PARAMS: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        target_col: str,
        feature_cols: list[str],
        id_col: str | None = None,
        normalize: bool = False,
        xgb_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(target_col=target_col)

        self._feature_cols = feature_cols
        self._id_col = id_col
        self._normalize = normalize
        self._xgb_params = {**self._DEFAULT_XGB_PARAMS, **(xgb_params or {})}

        # Set to an XGBRegressor instance after train() — typed as Optional
        # so mypy can track the untrained state.
        self._model: xgb.XGBRegressor | None = None

        # Both are computed at train time and reused as-is at predict time.
        self._categorical_encodings: dict[str, dict[str, int]] | None = None
        self._normalization_weights: pl.DataFrame | None = (
            None  # columns: [id_col, "weight"]
        )

        if self._normalize and self._id_col is None:
            raise ValueError("id_col is required when normalize=True.")

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def train(self, df: pl.DataFrame) -> None:
        """
        Trains the model on df.

        Steps:
        1. Validates that all required columns are present.
        2. Computes and stores normalization weights (if normalize=True).
        3. Applies preprocessing in train mode (encoding).
        4. Applies normalization to the target (if normalize=True).
        5. Fits XGBoost on the preprocessed x/y split.
        """
        required_cols = (
            self._feature_cols
            + [self.target_col]
            + ([self._id_col] if self._id_col else [])
        )
        self._assert_columns_exist(df, required_cols)

        if self._normalize:
            self._compute_normalization_weights(df)
            df = self._apply_normalization(df, target_col=self.target_col)

        df_processed = self._preprocess(df, mode="train")

        x = df_processed.select(self._feature_cols).to_numpy()
        y = df_processed[self.target_col].to_numpy()

        self._model = xgb.XGBRegressor(**self._xgb_params)
        self._model.fit(x, y)
        self._is_trained = True

    def predict(self, df: pl.DataFrame) -> pl.Series:
        """
        Generates predictions for df.

        Applies the same preprocessing as train() in predict mode, then
        denormalizes predictions if normalize=True.

        Raises
        ------
        RuntimeError
            If called before train().
        ValueError
            If any ID in df was not seen during training.
        """
        self._guard_is_trained("predict")  # guarantees _model is set
        assert self._model is not None  # narrows the type for mypy

        self._assert_columns_exist(df, self._feature_cols)
        df_processed = self._preprocess(df, mode="predict")

        x = df_processed.select(self._feature_cols).to_numpy()
        raw_predictions = pl.Series(name="prediction", values=self._model.predict(x))

        if self._normalize:
            assert self._id_col is not None
            return self._apply_denormalization(raw_predictions, df[self._id_col])

        return raw_predictions

    def feature_importance(self) -> pl.DataFrame:
        """
        Returns feature importances sorted by descending importance.

        Returns
        -------
        pl.DataFrame
            Columns: feature (str), importance (float).

        Raises
        ------
        RuntimeError
            If called before train().
        """
        self._guard_is_trained("feature_importance")
        assert self._model is not None

        return pl.DataFrame(
            {
                "feature": self._feature_cols,
                "importance": self._model.feature_importances_,
            }
        ).sort("importance", descending=True)

    def explain(self, df: pl.DataFrame, max_display: int = 20) -> pl.DataFrame:
        """
        Computes SHAP values for df using TreeExplainer.

        Returns one row per sample, one column per feature.
        Values represent each feature's additive contribution to the prediction.

        Parameters
        ----------
        df:
            Dataset to explain. Must contain all feature columns.
        max_display:
            Maximum number of features to include, selected by mean absolute
            SHAP value descending. Defaults to 20.

        Returns
        -------
        pl.DataFrame
            Columns: [feature_1, feature_2, ...] (SHAP contributions).

        Raises
        ------
        RuntimeError
            If called before train().
        """
        self._guard_is_trained("explain")
        assert self._model is not None

        df_processed = self._preprocess(df, mode="predict")
        x = df_processed.select(self._feature_cols).to_numpy()

        shap_values = shap.TreeExplainer(self._model).shap_values(x)
        shap_df = pl.DataFrame(shap_values, schema=self._feature_cols)

        if self._normalize:
            assert self._id_col is not None
            weights = self._get_weights_for_ids(df[self._id_col])
            shap_df = shap_df.with_columns(pl.all() * weights)

        top_features = (
            shap_df.select(pl.all().abs().mean())
            .unpivot()
            .sort("value", descending=True)
            .head(max_display)["variable"]
            .to_list()
        )

        return shap_df.select(top_features)

    def save(self, path: str | Path) -> None:
        """
        Serializes the full model state to disk using pickle.

        Saved state includes: XGBoost model, feature columns, categorical
        encodings, normalization weights, and all init parameters.

        Parameters
        ----------
        path:
            Destination file path (e.g. "models/gb_forecaster.pkl").
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Loads a serialized GradientBoostingForecaster from disk.

        Parameters
        ----------
        path:
            Path to the serialized .pkl file.

        Returns
        -------
        GradientBoostingForecaster
            A fully initialized, ready-to-predict model instance.
        """
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(
                f"Expected a {cls.__name__} instance, got {type(model).__name__}."
            )
        return model

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _assert_columns_exist(self, df: pl.DataFrame, required_cols: list[str]) -> None:
        """Fails fast if any required columns are missing from df."""
        missing = sorted(set(required_cols) - set(df.columns))
        if missing:
            raise ValueError(
                f"Missing required columns in input DataFrame: {missing}.\n"
                f"Available columns: {sorted(df.columns)}"
            )

    def _compute_normalization_weights(self, df: pl.DataFrame) -> None:
        """
        Computes and stores one normalization weight per ID.

        Weight = mean(target_col) for that ID on the training set.
        Stored as a DataFrame for direct use in join-based normalization.
        Computed exclusively on training data to avoid leakage.
        """
        assert self._id_col is not None

        self._normalization_weights = df.group_by(self._id_col).agg(
            pl.col(self.target_col).mean().alias("weight")
        )

    def _apply_normalization(self, df: pl.DataFrame, target_col: str) -> pl.DataFrame:
        """Divides target_col by the normalization weight for each ID."""
        assert self._id_col is not None and self._normalization_weights is not None

        joined_df = df.join(self._normalization_weights, on=self._id_col, how="left")

        if (null_count := joined_df["weight"].null_count()) > 0:
            missing = (
                joined_df.filter(pl.col("weight").is_null())[self._id_col]
                .unique()
                .to_list()
            )
            raise ValueError(
                f"[Normalization Error] {null_count} rows contain IDs unseen during"
                f" training: {missing}"
            )

        return joined_df.with_columns(
            (pl.col(target_col) / pl.col("weight")).alias(target_col)
        ).drop("weight")

    def _apply_denormalization(
        self, predictions: pl.Series, ids: pl.Series
    ) -> pl.Series:
        """Multiplies predictions by the stored normalization weight for each ID."""
        return (predictions * self._get_weights_for_ids(ids)).alias(predictions.name)

    def _get_weights_for_ids(self, ids: pl.Series) -> pl.Series:
        """Validates and extracts the normalization weights for a series of IDs."""
        assert self._id_col is not None and self._normalization_weights is not None

        known_ids = set(self._normalization_weights[self._id_col].to_list())
        if unknown_ids := sorted(set(ids.to_list()) - known_ids):
            raise ValueError(
                "[Normalization Error] Unknown IDs encountered at predict"
                f" time: {unknown_ids}"
            )

        return pl.DataFrame({self._id_col: ids}).join(
            self._normalization_weights, on=self._id_col, how="left"
        )["weight"]

    def _preprocess(
        self, df: pl.DataFrame, mode: Literal["train", "predict"]
    ) -> pl.DataFrame:
        """
        Applies the preprocessing pipeline.

        Steps (in order):
        1. Select feature columns + target (train) or feature columns only (predict).
        2. Cast numeric columns to Float64 for XGBoost compatibility.
        3. Encode categorical columns.
        """
        cols_to_select = list(self._feature_cols)
        if mode == "train":
            cols_to_select.append(self.target_col)

        self._assert_columns_exist(df, cols_to_select)
        df = df.select(cols_to_select)

        # polars.selectors.numeric() identifies all numeric dtypes concisely.
        numeric_cols = df.select(self._feature_cols).select(cs.numeric()).columns
        if numeric_cols:
            # XGBoost expects Float64 for all numeric inputs — mixed integer types
            # can cause silent coercions inside the XGBoost C++ backend.
            df = df.with_columns([pl.col(c).cast(pl.Float64) for c in numeric_cols])

        return self._encode_categoricals(df, mode=mode)

    def _encode_categoricals(
        self, df: pl.DataFrame, mode: Literal["train", "predict"]
    ) -> pl.DataFrame:
        """
        Label-encodes all String/Categorical columns.

        Train mode: builds and stores the encoding map per column.
        Predict mode: applies stored encodings, fails fast on unseen values.
        """
        categorical_cols = [
            col
            for col in self._feature_cols
            if df[col].dtype in (pl.String, pl.Categorical)
        ]

        if not categorical_cols:
            return df

        if mode == "train":
            self._categorical_encodings = {
                col: {
                    val: idx
                    for idx, val in enumerate(
                        df[col].drop_nulls().unique().sort().to_list()
                    )
                }
                for col in categorical_cols
            }

        assert self._categorical_encodings is not None

        if mode == "predict":
            self._validate_categorical_coverage(df, categorical_cols)

        return df.with_columns(
            [
                pl.col(col)
                .replace(self._categorical_encodings[col])
                .cast(pl.Int32)
                .alias(col)
                for col in categorical_cols
            ]
        )

    def _validate_categorical_coverage(
        self, df: pl.DataFrame, categorical_cols: list[str]
    ) -> None:
        """Checks that all categorical values in df are covered by stored encodings."""
        assert self._categorical_encodings is not None

        failures: list[str] = []
        for col in categorical_cols:
            known = set(self._categorical_encodings[col].keys())
            unseen = sorted(set(df[col].drop_nulls().unique().to_list()) - known)
            if unseen:
                failures.append(
                    f'  col: "{col}", '
                    f"features {[str(v) for v in unseen]} cannot be encoded. "
                    f"Available encodings: {sorted(known)}"
                )

        if failures:
            raise ValueError(
                "Some categorical features of the predict set cannot be encoded!\n"
                + "\n".join(failures)
            )
