import math

import polars as pl
import pytest

from evaluation.model_report import compute_report


@pytest.fixture
def evaluation_df() -> pl.DataFrame:
    """Standard evaluation dataset."""
    return pl.DataFrame(
        {
            "market": ["north", "north", "south", "south"],
            "meter_type": ["res", "ind", "res", "ind"],
            "consumption_kwh": [10.0, 20.0, 10.0, 20.0],
            "prediction": [11.0, 19.0, 12.0, 18.0],
        }
    )


def test_compute_report_schema(evaluation_df: pl.DataFrame) -> None:
    """Verifies that the report has the expected structure and columns."""
    report = compute_report(
        evaluation_df, "consumption_kwh", "prediction", {"overall": []}
    )

    assert {"breakdown", "mae", "rmse", "mape"}.issubset(set(report.columns))
    assert report.height == 1


def test_compute_report_multi_breakdown(evaluation_df: pl.DataFrame) -> None:
    """
    Validates multi-breakdown output: row count, group columns, and null padding.

    breakdown_config has 2 breakdowns:
    - "overall": 1 row, no group columns
    - "by_market": 2 rows (north, south), "meter_type" column filled with null
    -> total: 3 rows
    """
    config: dict[str, list[str]] = {
        "overall": [],
        "by_market": ["market"],
    }
    report = compute_report(evaluation_df, "consumption_kwh", "prediction", config)

    assert report.height == 3
    assert "market" in report.columns
    # "overall" rows have null market
    overall_rows = report.filter(pl.col("breakdown") == "overall")
    assert overall_rows["market"].null_count() == 1


def test_compute_report_invalid_column(evaluation_df: pl.DataFrame) -> None:
    """Ensures fail-fast on missing columns."""
    with pytest.raises(ValueError, match="Missing required columns"):
        compute_report(
            evaluation_df,
            "consumption_kwh",
            "prediction",
            {"by_unknown": ["missing_col"]},
        )


def test_metrics_accuracy_known_values() -> None:
    """Verify metrics against hand-calculated values."""
    df = pl.DataFrame({"y_true": [10.0, 20.0], "y_pred": [10.0, 10.0]})
    # MAE = 5.0, RMSE = sqrt(50) ≈ 7.071, MAPE = 0.25
    report = compute_report(df, "y_true", "y_pred", {"overall": []})

    assert report["mae"][0] == pytest.approx(5.0)
    assert report["rmse"][0] == pytest.approx(7.071, rel=1e-3)
    assert report["mape"][0] == pytest.approx(0.25)


def test_metrics_perfect_prediction() -> None:
    """Ensure zero error when prediction matches truth."""
    df = pl.DataFrame({"y": [10.0], "y_pred": [10.0]})
    report = compute_report(df, "y", "y_pred", {"overall": []})

    assert report["mae"][0] == 0.0
    assert report["rmse"][0] == 0.0
    assert report["mape"][0] == 0.0


def test_metrics_handle_zero_truth() -> None:
    """Ensure MAPE doesn't crash on zero ground truth (epsilon guard)."""
    df = pl.DataFrame({"y": [0.0], "y_pred": [5.0]})
    report = compute_report(df, "y", "y_pred", {"overall": []})

    assert math.isfinite(report["mape"][0])
    assert report["mape"][0] > 0
