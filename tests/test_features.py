"""Feature engineering: guards, determinism and the leakage contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    RAW_INPUTS,
    build_features,
)


def _row(**overrides) -> pd.DataFrame:
    base = {
        "age": 34, "income": 1_200_000.0, "number_of_dependants": 2,
        "years_at_current_address": 5, "marital_status": "Married",
        "employment_status": "Salaried", "residence_type": "Rented",
        "loan_purpose": "Home", "loan_type": "Secured",
        "sanction_amount": 2_500_000.0, "loan_amount": 2_200_000.0,
        "loan_tenure_months": 36, "bank_balance_at_application": 450_000.0,
        "number_of_open_accounts": 3, "number_of_closed_accounts": 1,
        "total_loan_months": 60, "delinquent_months": 8, "total_dpd": 90,
        "enquiry_count": 5, "credit_utilization_ratio": 82.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_output_shape_and_order_are_fixed():
    out = build_features(_row())
    assert list(out.columns) == FEATURE_COLUMNS
    assert len(out) == 1


def test_no_post_disbursal_column_can_reach_the_model():
    """The leakage contract, asserted in code rather than trusted to review."""
    post_disbursal = {
        "principal_outstanding", "processing_fee", "gst",
        "net_disbursement", "installment_start_dt",
    }
    assert post_disbursal.isdisjoint(FEATURE_COLUMNS)
    assert post_disbursal <= set(EXCLUDED_COLUMNS)


def test_identifiers_and_protected_attributes_are_excluded():
    for column in ("cust_id", "loan_id", "zipcode", "city", "state", "gender"):
        assert column not in FEATURE_COLUMNS
        assert column in EXCLUDED_COLUMNS


def test_build_features_rejects_missing_raw_columns():
    incomplete = _row().drop(columns=["income"])
    with pytest.raises(ValueError, match="missing required raw columns"):
        build_features(incomplete)


def test_divide_by_zero_is_guarded_not_infinite():
    """40% of borrowers have zero delinquent months - the classic inf trap."""
    out = build_features(_row(delinquent_months=0, total_dpd=0))
    assert out["avg_dpd_per_delinquency"].iloc[0] == 0.0
    assert np.isfinite(out.select_dtypes("number").to_numpy()).all()


def test_no_nulls_or_infinities_anywhere_on_real_data():
    from src.data import load_clean

    df, _ = load_clean()
    out = build_features(df)
    numeric = out.select_dtypes("number")
    assert out.isna().sum().sum() == 0
    assert np.isfinite(numeric.to_numpy()).all()


def test_ratios_are_computed_correctly():
    out = build_features(_row())
    assert out["loan_to_income"].iloc[0] == pytest.approx(2_200_000 / 1_200_000)
    assert out["loan_to_sanction"].iloc[0] == pytest.approx(2_200_000 / 2_500_000)
    assert out["delinquency_ratio"].iloc[0] == pytest.approx(8 / 60)
    assert out["avg_dpd_per_delinquency"].iloc[0] == pytest.approx(90 / 8)
    assert out["open_account_ratio"].iloc[0] == pytest.approx(3 / 4)
    # EMI = principal/tenure; monthly income = annual/12.
    assert out["emi_to_monthly_income"].iloc[0] == pytest.approx(
        (2_200_000 / 36) / (1_200_000 / 12)
    )


def test_is_deterministic():
    """Same input twice, byte-identical output - a serving requirement."""
    a, b = build_features(_row()), build_features(_row())
    pd.testing.assert_frame_equal(a, b)


def test_row_order_does_not_change_any_row_result():
    """Features must be row-independent; a batch must not change a score."""
    frame = pd.concat([_row(age=30), _row(age=55), _row(age=41)], ignore_index=True)
    batch = build_features(frame)
    for i in range(len(frame)):
        single = build_features(frame.iloc[[i]]).reset_index(drop=True)
        pd.testing.assert_frame_equal(single, batch.iloc[[i]].reset_index(drop=True))


def test_categoricals_stay_categorical():
    out = build_features(_row())
    for column in CATEGORICAL_FEATURES:
        assert out[column].dtype == object


def test_raw_inputs_cover_everything_build_features_needs():
    """RAW_INPUTS is the API's contract; it must be sufficient and minimal."""
    out = build_features(_row()[RAW_INPUTS])
    assert list(out.columns) == FEATURE_COLUMNS
