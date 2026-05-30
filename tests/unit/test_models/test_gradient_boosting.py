from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from models.gradient_boosting import GradientBoostingForecaster


def test_init_validation() -> None:
    """Ensures proper initialization and parameter validation."""
    model = GradientBoostingForecaster(target_col="sales", feature_cols=["day_of_week"])
    assert model.target_col == "sales"
    assert not model.is_trained

    with pytest.raises(ValueError, match="id_col is required when normalize=True"):
        GradientBoostingForecaster(
            target_col="sales", feature_cols=["day_of_week"], normalize=True
        )


def test_fail_fast_untrained(predict_df: pl.DataFrame) -> None:
    """Ensures predict, explain, and feature_importance fail if model is not trained."""
    model = GradientBoostingForecaster(target_col="sales", feature_cols=["day_of_week"])

    with pytest.raises(RuntimeError, match=r"predict\(\) called before train\(\)"):
        model.predict(predict_df)

    with pytest.raises(RuntimeError, match=r"explain\(\) called before train\(\)"):
        model.explain(predict_df)

    with pytest.raises(
        RuntimeError, match=r"feature_importance\(\) called before train\(\)"
    ):
        model.feature_importance()


def test_happy_path_without_normalization(
    train_df: pl.DataFrame, predict_df: pl.DataFrame
) -> None:
    """
    Validates full pipeline (train, predict, importance, explain) without normalization.
    """
    model = GradientBoostingForecaster(
        target_col="sales", feature_cols=["day_of_week", "price", "category"]
    )
    model.train(train_df)
    assert model.is_trained

    preds = model.predict(predict_df)
    assert isinstance(preds, pl.Series)
    assert len(preds) == len(predict_df)

    importances = model.feature_importance()
    assert importances.columns == ["feature", "importance"]
    # Importance values must be sorted descending
    assert importances["importance"].to_list() == sorted(
        importances["importance"].to_list(), reverse=True
    )

    shap_df = model.explain(predict_df)
    assert isinstance(shap_df, pl.DataFrame)
    assert shap_df.height == len(predict_df)
    assert set(shap_df.columns).issubset(set(model._feature_cols))


def test_happy_path_with_normalization(
    train_df: pl.DataFrame, predict_df: pl.DataFrame
) -> None:
    """
    Validates pipeline with ID-level normalization.

    Predictions must be in the original target scale — not the normalized one.
    S1 mean ≈ 123, S2 mean ≈ 51: predictions should stay in that order of magnitude.
    """
    model = GradientBoostingForecaster(
        target_col="sales",
        feature_cols=["day_of_week", "price", "category"],
        id_col="store_id",
        normalize=True,
    )
    model.train(train_df)
    preds = model.predict(predict_df)

    assert len(preds) == len(predict_df)
    # S1 should predict higher than S2 (≈ 123 vs ≈ 51 in training)
    s1_pred = preds[predict_df["store_id"].to_list().index("S1")]
    s2_pred = preds[predict_df["store_id"].to_list().index("S2")]
    assert s1_pred > s2_pred


def test_predict_unknown_id_raises_with_normalization(train_df: pl.DataFrame) -> None:
    """Ensures unknown IDs at predict time raise immediately when normalize=True."""
    model = GradientBoostingForecaster(
        target_col="sales",
        feature_cols=["day_of_week", "price", "category"],
        id_col="store_id",
        normalize=True,
    )
    model.train(train_df)

    unknown_id_df = pl.DataFrame(
        {
            "store_id": ["S_UNKNOWN"],
            "day_of_week": [1],
            "price": [10.0],
            "category": ["electronics"],
        }
    )
    with pytest.raises(ValueError, match="S_UNKNOWN"):
        model.predict(unknown_id_df)


@pytest.mark.parametrize(
    "modification, error_match",
    [
        (lambda df: df.drop("day_of_week"), "Missing required columns"),
        (
            lambda df: df.with_columns(pl.lit("online").alias("category")),
            "cannot be encoded",
        ),
    ],
)
def test_predict_fail_fast_data_errors(
    train_df: pl.DataFrame,
    predict_df: pl.DataFrame,
    modification: Callable[[pl.DataFrame], pl.DataFrame],
    error_match: str,
) -> None:
    """Ensures model fails gracefully on missing columns or unseen categories."""
    model = GradientBoostingForecaster(
        target_col="sales", feature_cols=["day_of_week", "category"]
    )
    model.train(train_df)

    with pytest.raises(ValueError, match=error_match):
        model.predict(modification(predict_df))


def test_serialization(
    train_df: pl.DataFrame, predict_df: pl.DataFrame, tmp_path: Path
) -> None:
    """Ensures save() and load() restore exact model state."""
    model = GradientBoostingForecaster(
        target_col="sales", feature_cols=["day_of_week", "category"]
    )
    model.train(train_df)
    ref_preds = model.predict(predict_df)

    file_path = tmp_path / "model.pkl"
    model.save(file_path)

    loaded_model = GradientBoostingForecaster.load(file_path)
    assert loaded_model.is_trained
    assert (ref_preds == loaded_model.predict(predict_df)).all()
