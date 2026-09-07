"""Feature engineering - the single source of truth for transformations.

This module is imported by **both** ``src/train.py`` and ``api/main.py``.
That is the whole point: the API cannot compute a feature differently from the
way it was computed at training time, because there is only one implementation.
It is also wrapped inside the serialized sklearn ``Pipeline``, so even the
column ordering and the one-hot vocabulary travel with the model file.

Credit intuition for every engineered feature is in the docstring of
:func:`build_features` and repeated in ``docs/INTERVIEW_GUIDE.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw columns the model is allowed to see.
#
# Everything here is knowable at the moment a lender decides on the
# application.  Nothing here depends on the loan having been disbursed.
# ---------------------------------------------------------------------------
RAW_NUMERIC_INPUTS = [
    "age",
    "income",
    "number_of_dependants",
    "years_at_current_address",
    "sanction_amount",
    "loan_amount",
    "loan_tenure_months",
    "bank_balance_at_application",
    "number_of_open_accounts",
    "number_of_closed_accounts",
    "total_loan_months",
    "delinquent_months",
    "total_dpd",
    "enquiry_count",
    "credit_utilization_ratio",
]

RAW_CATEGORICAL_INPUTS = [
    "marital_status",
    "employment_status",
    "residence_type",
    "loan_purpose",
    "loan_type",
]

RAW_INPUTS = RAW_NUMERIC_INPUTS + RAW_CATEGORICAL_INPUTS

# ---------------------------------------------------------------------------
# Deliberate exclusions.  Each one is an answer to an interview question.
# ---------------------------------------------------------------------------
EXCLUDED_COLUMNS = {
    # Identifiers - no predictive content, and a model that learns from an ID
    # has memorised the training set.
    "cust_id": "identifier",
    "loan_id": "identifier",
    # Geography.  Low cardinality here (10 cities / 9 states / 10 zipcodes) but
    # the default rate is flat across them (8.1%-9.1%, inside sampling noise),
    # so there is nothing to gain - and location is a classic proxy for
    # protected characteristics (redlining).  Excluded on both grounds.
    "zipcode": "no signal + fair-lending proxy risk",
    "city": "no signal + fair-lending proxy risk",
    "state": "no signal + fair-lending proxy risk",
    # Protected characteristic.  Under ECOA / Reg B (US) and the RBI fair
    # practices code (India) a lender may not price or decline on sex.  The
    # marginal default rates are 8.79% (F) vs 8.47% (M) - statistically
    # indistinguishable - so there is no accuracy cost to leaving it out.
    # We still keep it in the raw frame in order to *audit* the model's
    # fairness by slice (see docs/MODEL_CARD.md).
    "gender": "protected characteristic - excluded from features, retained for fairness auditing",
    # Post-disbursal - see the leakage section of the interview guide.
    "principal_outstanding": "post-disbursal (mid-repayment balance)",
    "processing_fee": "post-approval, and exactly 2% of loan_amount",
    "gst": "post-approval, and exactly 18% of loan_amount",
    "net_disbursement": "post-approval, and exactly 80% of loan_amount",
    "installment_start_dt": "post-approval",
    # disbursal_date is kept in the raw frame to build the out-of-time split,
    # but no feature is derived from it: the default rate is flat across all
    # 21 quarters in the data (7.8%-9.6%), so a month/quarter feature would
    # only let the model memorise noise, and it would be the first thing to
    # drift in production.
    "disbursal_date": "used only to build the out-of-time split, not a feature",
}

# ---------------------------------------------------------------------------
# The engineered feature set, in the order the model sees it.
# ---------------------------------------------------------------------------
ENGINEERED_NUMERIC_FEATURES = [
    "loan_to_income",
    "emi_to_monthly_income",
    "loan_to_sanction",
    "balance_to_loan",
    "months_of_income_in_bank",
    "income_per_dependant",
    "delinquency_ratio",
    "avg_dpd_per_delinquency",
    "credit_utilization_ratio",
    "open_account_ratio",
    "total_accounts",
    "enquiry_count",
    "loan_tenure_months",
    "total_loan_months",
    "age",
    "years_at_current_address",
    "number_of_dependants",
]

CATEGORICAL_FEATURES = list(RAW_CATEGORICAL_INPUTS)

FEATURE_COLUMNS = ENGINEERED_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed values for each categorical, used by the API schemas so a typo comes
# back as a 422 rather than being silently one-hot encoded to all-zeros.
CATEGORICAL_LEVELS = {
    "marital_status": ["Married", "Single"],
    "employment_status": ["Salaried", "Self-Employed"],
    "residence_type": ["Owned", "Mortgage", "Rented", "Unknown"],
    "loan_purpose": ["Personal", "Home", "Auto", "Education"],
    "loan_type": ["Secured", "Unsecured"],
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divide two series, returning ``fill`` wherever the denominator is 0 or NaN.

    Every ratio below goes through this.  A silent ``inf`` is the classic way a
    feature pipeline poisons a model - XGBoost will happily split on it and the
    API will then return nonsense for the one applicant with a zero denominator.
    """
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw application fields into the model's feature matrix.

    Credit intuition, feature by feature:

    ``loan_to_income``
        Loan size relative to annual income. The oldest underwriting ratio
        there is: a borrower asking for 5x their income is stretched in a way
        one asking for 0.5x is not.

    ``emi_to_monthly_income``
        A crude fixed-obligation-to-income ratio (FOIR/DTI): the monthly
        instalment divided by monthly income. This is what actually breaks a
        borrower - not the size of the loan, but whether the monthly payment
        fits in the monthly budget. Computed as principal/tenure, so it
        understates the true EMI (no interest), but the *ranking* across
        applicants is what the model uses.

    ``loan_to_sanction``
        How much of the approved limit the borrower actually drew. Drawing the
        full sanction signals the applicant needed every rupee available.

    ``balance_to_loan``
        Bank balance at application as a multiple of the loan. A liquidity
        cushion: money in the account is the first thing that absorbs a bad
        month.

    ``months_of_income_in_bank``
        The same cushion expressed in months of income - how long the
        applicant could service the loan with no incoming salary. Scale-free,
        so it compares a high earner and a low earner fairly.

    ``income_per_dependant``
        Income divided by (1 + dependants). Two applicants on the same salary
        do not have the same disposable income if one supports four people.

    ``delinquency_ratio``
        Share of the borrower's credit history spent in delinquency
        (``delinquent_months / total_loan_months``). Past repayment behaviour
        is the strongest single predictor of future repayment behaviour, and
        the ratio normalises for how long someone has had credit.

    ``avg_dpd_per_delinquency``
        Average days-past-due per delinquent month. Separates "pays a few days
        late, often" from "goes 90 days dark". The second is far more
        dangerous. Guarded: 40% of borrowers have zero delinquent months, and
        for them the feature is 0 by definition, not infinity.

    ``credit_utilization_ratio``
        Used as supplied. Percent of available revolving credit already drawn.
        High utilisation means the borrower is already leaning on credit to
        make ends meet.

    ``open_account_ratio``
        ``open / (open + closed)``. A borrower whose accounts are mostly still
        open is accumulating obligations; one who has closed most of them has
        a track record of finishing what they started.

    ``total_accounts``
        Depth of the credit file. A thin file is hard to judge; a very thick
        one can mean credit-hungry behaviour.

    Pass-through raw features (``enquiry_count``, ``loan_tenure_months``,
    ``total_loan_months``, ``age``, ``years_at_current_address``,
    ``number_of_dependants``) are already on sensible scales and need no
    transformation. ``enquiry_count`` is credit-hunger; ``loan_tenure_months``
    is exposure duration; ``years_at_current_address`` is residential
    stability.

    Args:
        df: A frame containing at least the columns in :data:`RAW_INPUTS`.

    Returns:
        A frame with exactly :data:`FEATURE_COLUMNS`, in that order.
    """
    missing = [c for c in RAW_INPUTS if c not in df.columns]
    if missing:
        raise ValueError(f"build_features is missing required raw columns: {missing}")

    df = df.copy()
    out = pd.DataFrame(index=df.index)

    monthly_income = df["income"] / 12.0
    emi = _safe_divide(df["loan_amount"], df["loan_tenure_months"])

    # --- affordability -----------------------------------------------------
    out["loan_to_income"] = _safe_divide(df["loan_amount"], df["income"])
    out["emi_to_monthly_income"] = _safe_divide(emi, monthly_income)
    out["loan_to_sanction"] = _safe_divide(df["loan_amount"], df["sanction_amount"])
    out["balance_to_loan"] = _safe_divide(df["bank_balance_at_application"], df["loan_amount"])
    out["months_of_income_in_bank"] = _safe_divide(
        df["bank_balance_at_application"], monthly_income
    )
    out["income_per_dependant"] = df["income"] / (1.0 + df["number_of_dependants"].clip(lower=0))

    # --- repayment behaviour ----------------------------------------------
    out["delinquency_ratio"] = _safe_divide(df["delinquent_months"], df["total_loan_months"])
    # Guarded divide-by-zero: no delinquent months means no average to take.
    out["avg_dpd_per_delinquency"] = _safe_divide(df["total_dpd"], df["delinquent_months"])
    out["credit_utilization_ratio"] = df["credit_utilization_ratio"].astype(float)

    # --- credit file shape -------------------------------------------------
    total_accounts = df["number_of_open_accounts"] + df["number_of_closed_accounts"]
    out["open_account_ratio"] = _safe_divide(df["number_of_open_accounts"], total_accounts)
    out["total_accounts"] = total_accounts.astype(float)

    # --- pass-through ------------------------------------------------------
    out["enquiry_count"] = df["enquiry_count"].astype(float)
    out["loan_tenure_months"] = df["loan_tenure_months"].astype(float)
    out["total_loan_months"] = df["total_loan_months"].astype(float)
    out["age"] = df["age"].astype(float)
    out["years_at_current_address"] = df["years_at_current_address"].astype(float)
    out["number_of_dependants"] = df["number_of_dependants"].astype(float)

    # --- categoricals ------------------------------------------------------
    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype(str).str.strip()

    return out[FEATURE_COLUMNS]


def build_features_with_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix *plus* ``principal_outstanding``.

    Used only by the leakage ablation in ``src/train.py``.  Never used for the
    production model.  See docs/INTERVIEW_GUIDE.md.
    """
    out = build_features(df)
    out = out.copy()
    out["principal_outstanding"] = df["principal_outstanding"].astype(float)
    out["principal_outstanding_ratio"] = _safe_divide(
        df["principal_outstanding"], df["loan_amount"]
    )
    return out
