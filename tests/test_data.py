"""Cleaning rules: assert the exact counts the documentation quotes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import clean, load_raw, merge_raw


@pytest.fixture(scope="module")
def merged() -> pd.DataFrame:
    return merge_raw(*load_raw())


@pytest.fixture(scope="module")
def cleaned(merged):
    return clean(merged)


def test_merge_is_one_to_one(merged):
    """All three tables cover the same 50,000 customers, so no row is lost."""
    assert len(merged) == 50_000
    assert merged["cust_id"].is_unique


def test_cleaning_counts_match_the_docs(cleaned):
    """These are the numbers quoted in the README and the interview guide."""
    _, report = cleaned
    assert report.rows_in == 50_000
    assert report.loan_purpose_typos_fixed == 22
    assert report.residence_type_nulls_filled == 62
    assert report.zero_amount_rows_dropped == 8
    assert report.processing_fee_rows_repaired == 7
    assert report.rows_out == 49_992


def test_typo_is_normalised(cleaned):
    df, _ = cleaned
    assert "Personaal" not in set(df["loan_purpose"])
    assert set(df["loan_purpose"]) == {"Personal", "Home", "Auto", "Education"}


def test_residence_nulls_become_an_explicit_category(cleaned):
    df, _ = cleaned
    assert df["residence_type"].isna().sum() == 0
    assert (df["residence_type"] == "Unknown").sum() == 62


def test_no_zero_amount_rows_survive(cleaned):
    df, _ = cleaned
    assert (df["loan_amount"] == 0).sum() == 0
    assert (df["income"] == 0).sum() == 0


def test_disbursement_identity_holds_after_repair(cleaned):
    """loan_amount - processing_fee - gst == net_disbursement, for every row.

    Before cleaning, 7 rows break this because processing_fee is corrupted.
    The repair reconstructs the fee from gst, so afterwards the identity is
    exact everywhere.
    """
    df, report = cleaned
    residual = df["loan_amount"] - df["processing_fee"] - df["gst"] - df["net_disbursement"]
    assert report.identity_violations_before == 7
    assert report.identity_violations_after == 0
    assert np.allclose(residual, 0, atol=1.0)


def test_processing_fee_outliers_are_gone(cleaned):
    df, _ = cleaned
    assert (df["processing_fee"] / df["loan_amount"]).max() <= 0.03


def test_base_default_rate_is_unchanged_by_cleaning(cleaned):
    """Cleaning must not move the target distribution meaningfully."""
    df, _ = cleaned
    assert df[config.TARGET].mean() == pytest.approx(0.08594, abs=5e-4)
