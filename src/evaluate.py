"""Metrics, plots and the tables a credit-risk team actually asks for.

Nothing here trains anything.  Everything takes ``y_true`` and a vector of
predicted probabilities, so the same functions score the logistic baseline,
the tuned XGBoost model, the calibrated model, and the out-of-time holdout.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: the largest gap between the cumulative
    distribution of defaulters and of non-defaulters.

    Plain words: line everyone up by predicted risk. KS is the biggest
    percentage-point gap between "how many of the bad loans have we caught so
    far" and "how many of the good loans have we caught so far". 0 means the
    model cannot tell them apart; 100 means perfect separation. In consumer
    credit, 30-40 is a healthy production scorecard.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr) * 100.0)


def core_metrics(y_true: Iterable[int], y_prob: Iterable[float]) -> dict[str, float]:
    """ROC-AUC, PR-AUC, Gini, KS and Brier for one set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    roc_auc = roc_auc_score(y_true, y_prob)
    return {
        "roc_auc": float(roc_auc),
        # PR-AUC (average precision) is the honest headline at an 8.6% base
        # rate: ROC-AUC is flattered by the huge number of easy negatives.
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        # Gini = 2*AUC - 1. Same information as AUC, but it is the number
        # credit teams quote.
        "gini": float(2 * roc_auc - 1),
        "ks": ks_statistic(y_true, y_prob),
        # Brier = mean squared error of the probabilities. Unlike AUC it
        # punishes a model that ranks well but is badly calibrated.
        "brier": float(brier_score_loss(y_true, y_prob)),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Threshold selection driven by an explicit cost assumption
# ---------------------------------------------------------------------------
def cost_optimal_threshold(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    cost_fn: float = config.COST_FALSE_NEGATIVE,
    cost_fp: float = config.COST_FALSE_POSITIVE,
) -> dict[str, float]:
    """Pick the cut-off that minimises expected cost, not the default 0.5.

    0.5 is meaningless here. It is the right cut-off only if a missed
    defaulter and a wrongly declined customer cost the same, and they do not:
    a defaulted loan burns roughly 60% of the principal, while a good customer
    turned away costs only the interest margin we never earned, roughly 6%.
    So a missed defaulter is ~10x worse and the cut-off belongs far below 0.5.

    We sweep every candidate threshold and return the one with the lowest
    total expected cost.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    candidates = np.unique(np.round(np.linspace(0.01, 0.95, 189), 4))
    best = {"threshold": 0.5, "expected_cost": np.inf}
    rows = []
    for thr in candidates:
        pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = fn * cost_fn + fp * cost_fp
        rows.append({"threshold": float(thr), "cost": float(cost), "fn": int(fn), "fp": int(fp)})
        if cost < best["expected_cost"]:
            best = {"threshold": float(thr), "expected_cost": float(cost)}

    best["cost_curve"] = pd.DataFrame(rows)
    best["cost_false_negative"] = cost_fn
    best["cost_false_positive"] = cost_fp
    return best


def threshold_metrics(
    y_true: Iterable[int], y_prob: Iterable[float], threshold: float
) -> dict[str, Any]:
    """Confusion matrix plus precision / recall / F1 at a given cut-off."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    approved = int(tn + fn)
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        # Business-facing translations of the same confusion matrix.
        "approval_rate": float(approved / len(y_true)),
        "bad_rate_in_approved": float(fn / approved) if approved else 0.0,
        "default_capture_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
    }


# ---------------------------------------------------------------------------
# Rank-ordering: the table every credit committee asks for
# ---------------------------------------------------------------------------
def decile_table(y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    """Split the book into risk deciles and show the actual default rate in each.

    Decile 1 is the riskiest 10% by predicted probability. A usable scorecard
    must be *monotone* down this table: the event rate has to fall as you move
    to safer deciles. A model can have a good AUC and still rank-order badly in
    the middle of the book, which is exactly where pricing decisions are made.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    # Rank first so that ties (common with tree models) spread across bins
    # instead of collapsing one.
    frame["rank"] = frame["p"].rank(method="first", ascending=False)
    frame["decile"] = pd.qcut(frame["rank"], q=n_bins, labels=range(1, n_bins + 1)).astype(int)

    grouped = frame.groupby("decile", observed=True).agg(
        n=("y", "size"),
        events=("y", "sum"),
        min_prob=("p", "min"),
        max_prob=("p", "max"),
        avg_prob=("p", "mean"),
    )
    grouped["non_events"] = grouped["n"] - grouped["events"]
    grouped["event_rate"] = grouped["events"] / grouped["n"]
    total_events = grouped["events"].sum()
    total_non = grouped["non_events"].sum()
    grouped["cum_event_pct"] = grouped["events"].cumsum() / total_events
    grouped["cum_non_event_pct"] = grouped["non_events"].cumsum() / total_non
    grouped["ks"] = (grouped["cum_event_pct"] - grouped["cum_non_event_pct"]) * 100
    # Lift = how many times the decile's default rate exceeds the book average.
    grouped["lift"] = grouped["event_rate"] / (total_events / grouped["n"].sum())
    return grouped.reset_index()


def is_monotone(table: pd.DataFrame) -> bool:
    """True if the event rate falls monotonically from decile 1 to decile 10."""
    rates = table.sort_values("decile")["event_rate"].to_numpy()
    return bool(np.all(np.diff(rates) <= 1e-12))


# ---------------------------------------------------------------------------
# Population Stability Index - the production drift alarm
# ---------------------------------------------------------------------------
def population_stability_index(
    expected: Iterable[float], actual: Iterable[float], n_bins: int = 10
) -> float:
    """PSI between a reference (training) distribution and a new one.

    Rule of thumb used across the industry:
        < 0.10  stable, no action
        0.10-0.25  moderate shift, investigate
        > 0.25  significant shift, retrain

    Bin edges come from the *expected* distribution, which is the whole point:
    in production you freeze the training bins and watch new data move through
    them.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(expected, quantiles))
    if len(edges) < 3:
        # Near-constant feature: no meaningful bins, so no measurable drift.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts = np.histogram(expected, bins=edges)[0].astype(float)
    act_counts = np.histogram(actual, bins=edges)[0].astype(float)

    # Laplace-style floor so an empty bin does not send PSI to infinity.
    exp_pct = np.clip(exp_counts / exp_counts.sum(), 1e-6, None)
    act_pct = np.clip(act_counts / act_counts.sum(), 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """PSI for each numeric column, sorted worst-first."""
    columns = list(columns) if columns is not None else [
        c for c in reference.columns if pd.api.types.is_numeric_dtype(reference[c])
    ]
    rows = []
    for col in columns:
        if col not in current.columns or not pd.api.types.is_numeric_dtype(reference[col]):
            continue
        psi = population_stability_index(reference[col], current[col])
        rows.append({
            "feature": col,
            "psi": psi,
            "verdict": "stable" if psi < 0.10 else ("moderate shift" if psi < 0.25 else "significant shift"),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Probability of default -> credit score
# ---------------------------------------------------------------------------
def clip_pd(prob: float | np.ndarray) -> np.ndarray:
    """Clamp a probability into ``[PD_FLOOR, PD_CEILING]``.

    Isotonic calibration returns exactly 0.0 for its lowest step and exactly
    1.0 for its highest.  Both are indefensible as reported probabilities: a
    lender that prices off a PD of 0 is dividing by zero, and telling an
    applicant they have a 0% chance of default is a claim no model can make
    from 35,000 rows.  Clamping is monotone, so every ranking metric is
    unaffected.
    """
    return np.clip(np.asarray(prob, dtype=float), config.PD_FLOOR, config.PD_CEILING)


def pd_to_credit_score(
    prob_default: float | np.ndarray,
    score_min: int = config.SCORE_MIN,
    score_max: int = config.SCORE_MAX,
) -> np.ndarray:
    """Map a calibrated probability of default onto the familiar 300-900 scale.

    This is a presentation layer, not extra modelling: it is a strictly
    decreasing function of the probability, so the ordering is identical. We
    use a linear map on ``1 - PD`` because it is trivially explainable to a
    customer and to a regulator - unlike the log-odds "points to double the
    odds" scaling, which is harder to defend when nobody can reproduce the
    constants by hand.

    PD = 0 -> 900 (best), PD = 1 -> 300 (worst).  The input is clamped by
    :func:`clip_pd` first, so the very best score is 900 rather than an
    unearned certainty.
    """
    prob = clip_pd(prob_default)
    return np.round(score_min + (1.0 - prob) * (score_max - score_min)).astype(int)


def score_to_rating(score: float | np.ndarray) -> np.ndarray:
    """Bucket a 300-900 score into Poor / Average / Good / Excellent."""
    score = np.atleast_1d(np.asarray(score))
    ratings = np.full(score.shape, "Poor", dtype=object)
    for low, high, label in config.RATING_BANDS:
        ratings[(score >= low) & (score < high)] = label
    return ratings


# ---------------------------------------------------------------------------
# Slice metrics for the model card
# ---------------------------------------------------------------------------
def slice_metrics(
    y_true: Iterable[int], y_prob: Iterable[float], groups: pd.Series, threshold: float
) -> pd.DataFrame:
    """Per-segment performance - the fairness table in the model card.

    We report AUC (does the model rank as well inside this group?) and the
    approval rate at the shared cut-off (does the model *treat* this group
    differently?). A model can be equally accurate on two groups and still
    approve them at very different rates.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    groups = pd.Series(groups).reset_index(drop=True)

    rows = []
    for value, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(idx)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        tm = threshold_metrics(yt, yp, threshold)
        rows.append({
            "segment": str(value),
            "n": len(yt),
            "base_rate": float(yt.mean()),
            "roc_auc": float(roc_auc_score(yt, yp)),
            "approval_rate": tm["approval_rate"],
            "recall": tm["recall"],
            "precision": tm["precision"],
            "mean_pd": float(yp.mean()),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
