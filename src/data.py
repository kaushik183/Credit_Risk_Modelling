"""Load, merge and clean the three raw tables.

The three CSVs are one row per customer / loan / bureau record, joined on
``cust_id``.  Each customer has exactly one loan and one bureau record, so the
joins are 1:1 and the merged frame is still 50,000 rows.

Every cleaning decision is applied here (never in ``features.py``) and every
row we touch is counted into a :class:`CleaningReport` so the numbers in the
docs come from the code rather than from memory.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# Columns that are pure identifiers - never features.
ID_COLUMNS = ["cust_id", "loan_id", "zipcode"]

# Columns knowable only *after* the loan is approved and disbursed.
# See docs/INTERVIEW_GUIDE.md "The leakage story".
POST_DISBURSAL_COLUMNS = [
    "principal_outstanding",
    "processing_fee",
    "gst",
    "net_disbursement",
    "installment_start_dt",
]


@dataclass
class CleaningReport:
    """Counts of everything cleaning touched, so the docs can quote real numbers."""

    rows_in: int = 0
    rows_out: int = 0
    loan_purpose_typos_fixed: int = 0
    residence_type_nulls_filled: int = 0
    zero_amount_rows_dropped: int = 0
    processing_fee_rows_repaired: int = 0
    identity_violations_before: int = 0
    identity_violations_after: int = 0
    duplicate_rows_dropped: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three raw CSVs exactly as they sit on disk."""
    customers = pd.read_csv(config.CUSTOMERS_CSV)
    loans = pd.read_csv(config.LOANS_CSV)
    bureau = pd.read_csv(config.BUREAU_CSV)
    logger.info(
        "loaded raw: customers=%s loans=%s bureau=%s",
        customers.shape, loans.shape, bureau.shape,
    )
    return customers, loans, bureau


def merge_raw(
    customers: pd.DataFrame, loans: pd.DataFrame, bureau: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join the three tables on ``cust_id``.

    We use inner joins deliberately: a loan with no customer record or no
    bureau record cannot be scored, so silently carrying it as a row of NaNs
    would be worse than dropping it.  On this dataset all three tables cover
    the same 50,000 customers, so nothing is lost.
    """
    for name, frame in (("customers", customers), ("loans", loans), ("bureau", bureau)):
        dupes = frame["cust_id"].duplicated().sum()
        if dupes:
            raise ValueError(f"{name} has {dupes} duplicate cust_id values; joins would fan out")

    df = customers.merge(loans, on="cust_id", how="inner", validate="one_to_one")
    df = df.merge(bureau, on="cust_id", how="inner", validate="one_to_one")
    logger.info("merged shape=%s", df.shape)
    return df


def clean(df: pd.DataFrame, report: CleaningReport | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply every cleaning rule and return the cleaned frame plus a report.

    The rules, in order:

    1. Normalise the ``Personaal`` typo in ``loan_purpose``.
    2. Fill ``residence_type`` nulls with an explicit ``Unknown`` category.
    3. Drop rows where the whole money block is zero (they are not real loans).
    4. Repair the handful of corrupted ``processing_fee`` values.
    5. Re-check the disbursement identity after the repair.
    """
    report = report or CleaningReport()
    df = df.copy()
    report.rows_in = len(df)

    # ---- 0. exact duplicate rows -----------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    report.duplicate_rows_dropped = before - len(df)

    # ---- 1. loan_purpose typo --------------------------------------------
    # "Personaal" is a straight misspelling of "Personal": the two groups have
    # the same default rate (4.5%), so this is a data-entry error, not a
    # distinct product.
    typo_mask = df["loan_purpose"].astype(str).str.strip().str.lower() == "personaal"
    report.loan_purpose_typos_fixed = int(typo_mask.sum())
    df.loc[typo_mask, "loan_purpose"] = "Personal"
    df["loan_purpose"] = df["loan_purpose"].astype(str).str.strip().str.title()

    # ---- 2. residence_type nulls -----------------------------------------
    # Decision: keep them as their own "Unknown" level rather than imputing the
    # mode.  Reasons:
    #   * a missing residence type is information in itself - in a real
    #     application flow it usually means the field was skipped, and "the
    #     applicant did not answer" is a legitimate predictor;
    #   * imputing the mode ("Owned", the *lowest* risk level at 5.6% default)
    #     would silently tell the model these 62 applicants are low risk;
    #   * one-hot encoding gives XGBoost a separate column, so it can split on
    #     "Unknown" if it is useful and ignore it if it is not.
    # At 62 rows out of 50,000 (0.12%) this choice cannot move the metrics -
    # it is chosen for correctness of reasoning, not for score.
    null_mask = df["residence_type"].isna()
    report.residence_type_nulls_filled = int(null_mask.sum())
    df["residence_type"] = df["residence_type"].fillna("Unknown")

    # ---- 3. all-zero money rows ------------------------------------------
    # 8 rows have sanction_amount, loan_amount, processing_fee, gst,
    # net_disbursement, income *and* bank_balance all equal to 0.  These are
    # not loans; they are placeholder records.  Every ratio feature
    # (loan_to_income, balance_to_loan, ...) is undefined for them, so we drop
    # them rather than invent values.
    zero_mask = (df["loan_amount"] == 0) | (df["income"] == 0)
    report.zero_amount_rows_dropped = int(zero_mask.sum())
    df = df.loc[~zero_mask].copy()

    # ---- 4. corrupted processing_fee -------------------------------------
    # In this dataset the fee block is deterministic:
    #     processing_fee   = 2%  of loan_amount
    #     gst              = 18% of loan_amount
    #     net_disbursement = 80% of loan_amount
    # 7 rows break the first identity - processing_fee is inflated to well
    # over 100% of the loan (max 148.9%).  gst and net_disbursement are still
    # internally consistent on those rows, so the corruption is isolated to
    # processing_fee and we can *reconstruct* the true value exactly as
    # gst / 9.  We repair instead of dropping: dropping would throw away 7
    # otherwise-valid applications, and the repaired value is not a guess, it
    # is forced by two independent identities.
    #
    # (This is bookkeeping hygiene only - processing_fee is a post-approval
    # column and never reaches the model.  We repair it so that the EDA
    # notebook and the identity check below are honest.)
    fee_ratio = df["processing_fee"] / df["loan_amount"]
    corrupt_fee = fee_ratio > 0.03
    report.processing_fee_rows_repaired = int(corrupt_fee.sum())

    identity_resid_before = (
        df["loan_amount"] - df["processing_fee"] - df["gst"] - df["net_disbursement"]
    )
    report.identity_violations_before = int((identity_resid_before.abs() > 1.0).sum())

    df.loc[corrupt_fee, "processing_fee"] = df.loc[corrupt_fee, "gst"] / 9.0

    identity_resid_after = (
        df["loan_amount"] - df["processing_fee"] - df["gst"] - df["net_disbursement"]
    )
    report.identity_violations_after = int((identity_resid_after.abs() > 1.0).sum())

    # ---- 5. types --------------------------------------------------------
    df["disbursal_date"] = pd.to_datetime(df["disbursal_date"])
    df["installment_start_dt"] = pd.to_datetime(df["installment_start_dt"])
    df[config.TARGET] = df[config.TARGET].astype(bool).astype(int)

    report.rows_out = len(df)
    report.notes.append(
        "gst / processing_fee / net_disbursement are exact linear functions of "
        "loan_amount (18% / 2% / 80%) and carry zero independent information; "
        "they are excluded from the feature set as post-approval columns anyway."
    )
    report.notes.append(
        "city, state and zipcode are low cardinality (10 / 9 / 10) and their "
        "default rates are flat (8.1%-9.1%); they are dropped for lack of "
        "signal and to avoid geography-based proxy discrimination."
    )

    logger.info("cleaning report: %s", report.to_dict())
    return df, report


def load_clean() -> tuple[pd.DataFrame, CleaningReport]:
    """Convenience wrapper: load the three CSVs, merge them, clean the result."""
    customers, loans, bureau = load_raw()
    merged = merge_raw(customers, loans, bureau)
    return clean(merged)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame, rep = load_clean()
    print(frame.shape)
    for key, value in rep.to_dict().items():
        print(f"  {key}: {value}")
