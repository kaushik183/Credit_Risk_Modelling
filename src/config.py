"""Central configuration: paths, seed, and business constants.

Everything that another module might want to tweak lives here so that no
magic numbers are buried inside training or serving code.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = ROOT / "data" / "raw"
CUSTOMERS_CSV = RAW_DATA_DIR / "customers.csv"
LOANS_CSV = RAW_DATA_DIR / "loans.csv"
BUREAU_CSV = RAW_DATA_DIR / "bureau_data.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the *whole* dataset
# Loans disbursed on/after this date form the out-of-time (OOT) holdout.
# The data spans 2019-07-24 .. 2024-07-22; this reserves the final ~12 months.
OOT_START_DATE = "2023-07-23"

# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
TARGET = "default"

# --------------------------------------------------------------------------
# Cost assumptions used to pick the decision threshold.
#
# A false negative (we approve a loan that defaults) costs us the unrecovered
# principal.  Industry loss-given-default on a mixed secured/unsecured book is
# roughly 60% of exposure.
#
# A false positive (we decline a loan that would have repaid) costs us the
# interest margin we never earn.  On a ~3% net interest margin over an average
# ~2 year life that is roughly 6% of exposure.
#
# The ratio is what matters, not the absolute rupees: missing a defaulter is
# about 10x more expensive than turning away a good customer.
# --------------------------------------------------------------------------
COST_FALSE_NEGATIVE = 0.60
COST_FALSE_POSITIVE = 0.06

# --------------------------------------------------------------------------
# Probability floor and ceiling.
#
# Isotonic calibration is a step function: its lowest step is exactly 0.0 and
# its highest is exactly 1.0.  A reported probability of default of 0 is not
# usable - it says this applicant can never default, which no lender believes
# and no risk-based pricing formula can divide by.  We clamp the reported PD
# into a plausible band.  1e-4 (1 in 10,000) is below anything the business
# would price differently, and 1 - 1e-4 is likewise a certain decline.
#
# This is a *reporting* clamp: it never changes the ranking, so AUC, KS and
# the decile table are untouched.
# --------------------------------------------------------------------------
PD_FLOOR = 1e-4
PD_CEILING = 1.0 - 1e-4

# --------------------------------------------------------------------------
# Credit score mapping (300-900, the CIBIL-style range used in India)
# --------------------------------------------------------------------------
SCORE_MIN = 300
SCORE_MAX = 900
RATING_BANDS = [
    # (inclusive lower bound, exclusive upper bound, label)
    (300, 500, "Poor"),
    (500, 650, "Average"),
    (650, 750, "Good"),
    (750, 901, "Excellent"),
]

MODEL_VERSION = "1.0.0"
