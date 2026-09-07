"""Pydantic v2 request and response models.

Validation lives here so that bad input fails at the edge with a 422 and a
readable message, rather than reaching the model and producing a 500 or -
much worse - a confident prediction from nonsense.

The bounds are deliberately *realistic* rather than merely type-correct: a
negative income or a 300-year-old applicant is a bug upstream, and the API
should say so.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# These mirror features.CATEGORICAL_LEVELS. Declaring them as Literals gives
# FastAPI an enum in the OpenAPI schema, so /docs shows a dropdown and a typo
# such as "self-employed" is rejected instead of silently one-hot encoded to
# an all-zero row.
MaritalStatus = Literal["Married", "Single"]
EmploymentStatus = Literal["Salaried", "Self-Employed"]
ResidenceType = Literal["Owned", "Mortgage", "Rented", "Unknown"]
LoanPurpose = Literal["Personal", "Home", "Auto", "Education"]
LoanType = Literal["Secured", "Unsecured"]


class ApplicantRequest(BaseModel):
    """One loan application, as a lender knows it at decision time.

    Every field here is available *before* the loan is disbursed. There is
    deliberately no ``principal_outstanding``, ``net_disbursement`` or
    ``installment_start_dt``: asking for them would mean the caller has to
    know the future.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 34,
                    "income": 1200000,
                    "number_of_dependants": 2,
                    "years_at_current_address": 5,
                    "marital_status": "Married",
                    "employment_status": "Salaried",
                    "residence_type": "Rented",
                    "loan_purpose": "Home",
                    "loan_type": "Secured",
                    "sanction_amount": 2500000,
                    "loan_amount": 2200000,
                    "loan_tenure_months": 36,
                    "bank_balance_at_application": 450000,
                    "number_of_open_accounts": 3,
                    "number_of_closed_accounts": 1,
                    "total_loan_months": 60,
                    "delinquent_months": 8,
                    "total_dpd": 90,
                    "enquiry_count": 5,
                    "credit_utilization_ratio": 82,
                }
            ]
        }
    }

    # --- applicant ---------------------------------------------------------
    age: Annotated[int, Field(ge=18, le=100, description="Applicant age in years. Under 18 cannot contract.")]
    income: Annotated[float, Field(gt=0, le=1e9, description="Gross annual income.")]
    number_of_dependants: Annotated[int, Field(ge=0, le=20)]
    years_at_current_address: Annotated[int, Field(ge=0, le=100)]
    marital_status: MaritalStatus
    employment_status: EmploymentStatus
    residence_type: ResidenceType

    # --- the loan being applied for ---------------------------------------
    loan_purpose: LoanPurpose
    loan_type: LoanType
    sanction_amount: Annotated[float, Field(gt=0, le=1e9, description="Approved credit limit.")]
    loan_amount: Annotated[float, Field(gt=0, le=1e9, description="Amount actually being drawn.")]
    loan_tenure_months: Annotated[int, Field(ge=1, le=480)]
    bank_balance_at_application: Annotated[float, Field(ge=0, le=1e9)]

    # --- credit bureau ------------------------------------------------------
    number_of_open_accounts: Annotated[int, Field(ge=0, le=100)]
    number_of_closed_accounts: Annotated[int, Field(ge=0, le=100)]
    total_loan_months: Annotated[int, Field(ge=1, le=1200, description="Length of the credit history in months.")]
    delinquent_months: Annotated[int, Field(ge=0, le=1200)]
    total_dpd: Annotated[int, Field(ge=0, le=100000, description="Cumulative days past due.")]
    enquiry_count: Annotated[int, Field(ge=0, le=100)]
    credit_utilization_ratio: Annotated[float, Field(ge=0, le=100, description="Percent of revolving credit in use.")]

    @model_validator(mode="after")
    def check_internal_consistency(self) -> ApplicantRequest:
        """Cross-field rules that no single-field bound can express.

        These catch upstream bugs that would otherwise produce a plausible
        looking but meaningless score.
        """
        if self.loan_amount > self.sanction_amount:
            raise ValueError("loan_amount cannot exceed sanction_amount")
        if self.delinquent_months > self.total_loan_months:
            raise ValueError("delinquent_months cannot exceed total_loan_months")
        if self.number_of_open_accounts + self.number_of_closed_accounts == 0:
            raise ValueError("applicant must have at least one open or closed credit account")
        if self.delinquent_months == 0 and self.total_dpd > 0:
            raise ValueError("total_dpd must be 0 when delinquent_months is 0")
        return self


class BatchRequest(BaseModel):
    """Up to 1,000 applications in one call."""

    applicants: Annotated[
        list[ApplicantRequest],
        Field(min_length=1, max_length=1000, description="Between 1 and 1000 applications."),
    ]


class PredictionResponse(BaseModel):
    """One scored application."""

    probability_of_default: Annotated[float, Field(ge=0, le=1, description="Calibrated PD - a real probability, not a raw score.")]
    credit_score: Annotated[int, Field(ge=300, le=900)]
    rating: Literal["Poor", "Average", "Good", "Excellent"]
    decision: Literal["approve", "decline"] = Field(description="Recommendation at the model's cost-optimal cut-off.")
    threshold: float = Field(description="The cut-off the decision was made against.")
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str
    algorithm: str
    feature_columns: list[str]
    raw_input_columns: list[str]
    n_features: int
    metrics: dict
    threshold: dict
    dataset: dict
    leakage_ablation: dict

    # ``model_`` is a protected namespace in Pydantic v2; we use those field
    # names anyway because they read correctly in the API contract, so the
    # warning is turned off explicitly rather than left to fire.
    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    detail: str
