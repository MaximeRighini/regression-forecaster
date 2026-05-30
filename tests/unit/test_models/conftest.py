import polars as pl
import pytest


@pytest.fixture
def train_df() -> pl.DataFrame:
    """Generic training dataset (Sales / Retail)."""
    return pl.DataFrame(
        {
            "store_id": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "day_of_week": [1, 2, 3, 1, 2, 3],
            "price": [10.5, 10.5, 9.9, 15.0, 15.0, 15.0],
            "category": [
                "electronics",
                "electronics",
                "electronics",
                "clothing",
                "clothing",
                "clothing",
            ],
            # S2 generally sells less than S1 (perfect for testing normalization)
            "sales": [100.0, 120.0, 150.0, 50.0, 45.0, 60.0],
        }
    )


@pytest.fixture
def predict_df() -> pl.DataFrame:
    """Generic prediction dataset (Sales / Retail)."""
    return pl.DataFrame(
        {
            "store_id": ["S1", "S2"],
            "day_of_week": [4, 4],
            "price": [9.9, 15.0],
            "category": ["electronics", "clothing"],
        }
    )
