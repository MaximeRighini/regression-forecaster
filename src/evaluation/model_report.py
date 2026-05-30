"""
Breakdown accuracy reporting for regression models.

compute_report() produces a structured flat DataFrame with MAE, RMSE, and MAPE
per group per breakdown, ready to be written to Excel for exploration.
"""

import polars as pl


def compute_report(
    df: pl.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    breakdown_config: dict[str, list[str]],
) -> pl.DataFrame:
    """
    Computes forecast accuracy metrics for each breakdown defined in breakdown_config.

    Each entry in breakdown_config defines a named slice of the evaluation:
    - Key: a label for the breakdown (e.g. "overall", "by_market").
    - Value: list of columns to group by. Empty list means global aggregation.

    The result is a single flat DataFrame with one row per group per breakdown,
    ready to be written to Excel for exploration.

    Parameters
    ----------
    df: DataFrame containing y_true_col, y_pred_col, and all grouping columns
        referenced in breakdown_config.
    y_true_col: Column name for ground truth values.
    y_pred_col: Column name for model predictions.
    breakdown_config: Dict mapping breakdown labels to lists of grouping columns.

    Returns
    -------
    pl.DataFrame
    Columns: breakdown (str), [group_cols...], mae, rmse, mape (floats).
    Group columns not relevant to a given breakdown are filled with null.

    Example
    -------
    ::

    breakdown_config = {
    "overall": [],  # group_by([]) with empty list produces a single group over all rows
    "by_market": ["market"],
    "by_meter_type": ["meter_type"],
    "by_market_and_type": ["market", "meter_type"],
    }

    report = compute_report(df, "y_true", "y_pred", breakdown_config)
    report.write_excel("evaluation/report.xlsx")
    """
    _validate_inputs(df, y_true_col, y_pred_col, breakdown_config)

    all_group_cols = sorted({col for cols in breakdown_config.values() for col in cols})
    breakdown_frames = []

    for name, group_cols in breakdown_config.items():
        metrics = _compute_grouped_metrics(df, y_true_col, y_pred_col, group_cols)

        metrics = metrics.with_columns(pl.lit(name).alias("breakdown"))

        # Ensure consistent schema for concat
        for col in all_group_cols:
            if col not in metrics.columns:
                metrics = metrics.with_columns(pl.lit(None).cast(pl.String).alias(col))

        breakdown_frames.append(
            metrics.select(["breakdown"] + all_group_cols + ["mae", "rmse", "mape"])
        )

    return pl.concat(breakdown_frames)


def _validate_inputs(
    df: pl.DataFrame, y_true: str, y_pred: str, breakdown_config: dict[str, list[str]]
) -> None:
    """Fail fast if columns are missing or config is invalid."""
    required = {y_true, y_pred}
    for cols in breakdown_config.values():
        required.update(cols)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Cannot compute model performance report!"
            f" Missing required columns in DataFrame: {missing}"
        )


def _compute_grouped_metrics(
    df: pl.DataFrame, y_true_col: str, y_pred_col: str, group_cols: list[str]
) -> pl.DataFrame:
    """Compute metrics using Polars group_by for maximum performance."""
    diff = pl.col(y_true_col) - pl.col(y_pred_col)

    return df.group_by(group_cols).agg(
        diff.abs().mean().cast(pl.Float64).alias("mae"),
        (diff**2).mean().sqrt().cast(pl.Float64).alias("rmse"),
        (diff.abs() / (pl.col(y_true_col).abs() + 1e-8))
        .mean()
        .cast(pl.Float64)
        .alias("mape"),
    )
