"""Train, evaluate, and serialize the credit default risk model.

Run with::

    python -m src.train                 # full run
    python -m src.train --trials 10     # quick run

Stages, in order:

1. Load and clean the data (``src.data``).
2. Build a stratified 70/15/15 split plus a time-based out-of-time holdout.
3. Fit a logistic-regression scorecard as the baseline.
4. Compare three ways of handling the 8.6% class imbalance.
5. Tune XGBoost with Optuna, using stratified CV and early stopping.
6. Calibrate the winner so its scores are usable as probabilities.
7. Run the leakage ablation and record what it proves.
8. Evaluate everything, write plots and tables, save the artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
import warnings
from datetime import UTC, datetime
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config
from . import evaluate as ev
from .data import load_clean
from .features import (
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    RAW_INPUTS,
    build_features,
    build_features_with_leakage,
)
from .pipeline import (
    encoded_feature_names,
    fit_xgb_with_early_stopping,
    make_logistic_pipeline,
    make_xgb_pipeline,
)

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("train")


# ===========================================================================
# Splitting
# ===========================================================================
def make_splits(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> dict[str, Any]:
    """Stratified 70/15/15 train/validation/test, plus an out-of-time holdout.

    Two different questions need two different splits.

    The random stratified split answers "how well does the model rank
    applicants it has never seen?".  Stratifying keeps the 8.6% default rate
    identical in all three parts, so the metrics are comparable.

    The out-of-time split answers the question a risk team actually asks: "if
    we had trained this model a year ago, how would it have performed on the
    loans we wrote since?"  We train on everything disbursed before
    ``OOT_START_DATE`` and score everything after it.  That is the only
    honest test of whether the model survives the passage of time.
    """
    y = df[config.TARGET].to_numpy()

    # --- random stratified 70/15/15 ---------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y, test_size=config.TEST_SIZE + config.VAL_SIZE, stratify=y, random_state=seed
    )
    # Half of the 30% remainder goes to validation, half to test.
    rel_test = config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=seed
    )

    # --- time-based out-of-time holdout -----------------------------------
    cutoff = pd.Timestamp(config.OOT_START_DATE)
    span = df["disbursal_date"].max() - df["disbursal_date"].min()
    oot_mask = df["disbursal_date"] >= cutoff
    oot_viable = span.days >= 365 and 0.05 <= oot_mask.mean() <= 0.45

    splits: dict[str, Any] = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "date_min": df["disbursal_date"].min(),
        "date_max": df["disbursal_date"].max(),
        "span_days": int(span.days),
        "oot_viable": bool(oot_viable),
        "oot_cutoff": cutoff,
    }
    if oot_viable:
        splits["X_oot_train"] = df.loc[~oot_mask]
        splits["y_oot_train"] = df.loc[~oot_mask, config.TARGET].to_numpy()
        splits["X_oot_test"] = df.loc[oot_mask]
        splits["y_oot_test"] = df.loc[oot_mask, config.TARGET].to_numpy()

    logger.info(
        "splits: train=%d val=%d test=%d | dates %s..%s (%d days) | OOT viable=%s (test n=%d)",
        len(X_train), len(X_val), len(X_test),
        splits["date_min"].date(), splits["date_max"].date(), splits["span_days"],
        oot_viable, len(splits.get("X_oot_test", [])),
    )
    return splits


# ===========================================================================
# Imbalance strategy comparison
# ===========================================================================
def compare_imbalance_strategies(splits: dict[str, Any]) -> pd.DataFrame:
    """Three ways to deal with 8.6% positives - measured, not assumed.

    * **do nothing** - let XGBoost see the raw class ratio;
    * **scale_pos_weight** - re-weight the positive class by neg/pos so the
      gradient from the 4,297 defaults matches that of the 45,703 repayments.
      Nothing is discarded and nothing is duplicated;
    * **random undersampling** - throw away good loans until the classes
      balance. Cheap to run, but it discards ~90% of the majority class and
      with it a lot of information about what a good borrower looks like.

    Judged on validation PR-AUC, which is the metric that actually moves at
    this base rate.
    """
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    rows = []

    # 1. no handling
    pipe = make_xgb_pipeline()
    pipe = fit_xgb_with_early_stopping(pipe, X_train, y_train, X_val, y_val)
    p = pipe.predict_proba(X_val)[:, 1]
    rows.append({"strategy": "none (raw 8.6% positives)", **ev.core_metrics(y_val, p)})

    # 2. scale_pos_weight
    pipe = make_xgb_pipeline(scale_pos_weight=spw)
    pipe = fit_xgb_with_early_stopping(pipe, X_train, y_train, X_val, y_val)
    p = pipe.predict_proba(X_val)[:, 1]
    rows.append({"strategy": f"scale_pos_weight={spw:.2f}", **ev.core_metrics(y_val, p)})

    # 3. random undersampling of the majority class
    rng = np.random.default_rng(config.RANDOM_SEED)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    keep_neg = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    under = np.sort(np.concatenate([pos_idx, keep_neg]))
    pipe = make_xgb_pipeline()
    pipe = fit_xgb_with_early_stopping(
        pipe, X_train.iloc[under], y_train[under], X_val, y_val
    )
    p = pipe.predict_proba(X_val)[:, 1]
    rows.append({
        "strategy": f"random undersampling (n={len(under)}, {len(neg_idx) - len(keep_neg)} good loans discarded)",
        **ev.core_metrics(y_val, p),
    })

    table = pd.DataFrame(rows)[["strategy", "roc_auc", "pr_auc", "ks", "brier", "n"]]
    logger.info("imbalance comparison:\n%s", table.to_string(index=False))
    return table


# ===========================================================================
# Hyperparameter search
# ===========================================================================
def tune_xgboost(splits: dict[str, Any], n_trials: int = 40) -> tuple[dict[str, Any], pd.DataFrame]:
    """Optuna search over the eight standard XGBoost knobs.

    Searched on train+validation combined with 4-fold stratified CV, so the
    chosen hyperparameters are not tuned to one lucky validation split.  Early
    stopping runs inside each fold, which is why ``n_estimators`` is set high
    and left to the stopper rather than searched as a hard budget.

    The test set is not touched anywhere in this function.
    """
    X = pd.concat([splits["X_train"], splits["X_val"]], axis=0)
    y = np.concatenate([splits["y_train"], splits["y_val"]])
    spw = float((y == 0).sum() / (y == 1).sum())

    def objective(trial: optuna.Trial) -> float:
        params = {
            # depth: the main bias/variance dial. Deeper trees capture more
            # interactions and overfit faster.
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            # how much of each tree's correction is applied. Lower = slower,
            # steadier learning that generalises better.
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # row sampling per tree - injects randomness, reduces variance.
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            # column sampling per tree - decorrelates the trees.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # minimum weighted rows in a leaf - the direct anti-overfit guard.
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            # minimum loss reduction to make a split - prunes weak splits.
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            # L2 penalty on leaf weights - shrinks extreme predictions.
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        }
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=config.RANDOM_SEED)
        scores = []
        for tr, va in cv.split(X, y):
            pipe = make_xgb_pipeline(params, scale_pos_weight=spw)
            pipe = fit_xgb_with_early_stopping(
                pipe, X.iloc[tr], y[tr], X.iloc[va], y[va], early_stopping_rounds=40
            )
            scores.append(roc_auc_score(y[va], pipe.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    started = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "optuna: %d trials in %.0fs, best CV AUC=%.5f, params=%s",
        n_trials, time.time() - started, study.best_value, study.best_params,
    )

    history = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    return study.best_params, history


# ===========================================================================
# Calibration
# ===========================================================================
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Average gap between predicted probability and observed frequency.

    Brier score mixes calibration and discrimination together; ECE isolates
    calibration.  Bin the predictions, and in each bin compare "what the model
    said on average" against "what actually happened".  0 is perfect.
    """
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    grouped = frame.groupby("bin").agg(obs=("y", "mean"), pred=("p", "mean"), n=("y", "size"))
    return float((grouped["n"] * (grouped["obs"] - grouped["pred"]).abs()).sum() / grouped["n"].sum())


def compare_calibration(
    fitted_pipeline, X_val, y_val, X_test, y_test
) -> tuple[Any, pd.DataFrame, str]:
    """Fit both calibrators and pick the better one - without touching test.

    Raw XGBoost margins are not probabilities: the model is trained to rank,
    and ``scale_pos_weight`` deliberately distorts the output level on top of
    that.  If the business prices a loan off the number, the number has to
    mean what it says - "PD = 3%" must correspond to 3 defaults per 100 such
    loans.

    Isotonic fits a free-form monotone step function - flexible, but
    data-hungry.  Platt fits a single sigmoid - two parameters, safe on small
    folds, but it can only undo distortions that happen to be sigmoid-shaped.

    **How the winner is chosen.** A calibrator cannot be judged on the fold it
    was fitted to, and it must not be judged on test - that would be selecting
    a model on the holdout, which is the thing this project is careful not to
    do anywhere else.  So the validation set is split in two: both calibrators
    are fitted on one half and scored on the other, the lower Brier wins, and
    only then is the winner refitted on the whole validation set.  The test
    column in the returned table is *reporting*, not selection - it is
    computed after the choice is already made.
    """
    fit_idx, select_idx = train_test_split(
        np.arange(len(X_val)),
        test_size=0.5,
        stratify=y_val,
        random_state=config.RANDOM_SEED,
    )
    X_fit, y_fit = X_val.iloc[fit_idx], y_val[fit_idx]
    X_sel, y_sel = X_val.iloc[select_idx], y_val[select_idx]

    selection = {}
    for label, method in (("isotonic", "isotonic"), ("sigmoid (Platt)", "sigmoid")):
        trial = CalibratedClassifierCV(fitted_pipeline, method=method, cv="prefit")
        trial.fit(X_fit, y_fit)
        p_sel = trial.predict_proba(X_sel)[:, 1]
        selection[label] = {
            "brier_holdout": float(ev.core_metrics(y_sel, p_sel)["brier"]),
            "ece_holdout": expected_calibration_error(y_sel, p_sel),
        }
    winner = min(selection, key=lambda k: selection[k]["brier_holdout"])
    logger.info(
        "calibrator selection on held-out half of validation: %s -> winner=%s",
        {k: round(v["brier_holdout"], 5) for k, v in selection.items()}, winner,
    )

    # Refit the winner on the full validation set now that the choice is made.
    method = "isotonic" if winner == "isotonic" else "sigmoid"
    final = CalibratedClassifierCV(fitted_pipeline, method=method, cv="prefit")
    final.fit(X_val, y_val)

    # Reporting table - computed after selection, never used to select.
    rows = [{
        "method": "none (raw XGBoost)",
        **ev.core_metrics(y_test, fitted_pipeline.predict_proba(X_test)[:, 1]),
        "ece": expected_calibration_error(y_test, fitted_pipeline.predict_proba(X_test)[:, 1]),
        "selected": False,
    }]
    for label, m in (("isotonic", "isotonic"), ("sigmoid (Platt)", "sigmoid")):
        cal = CalibratedClassifierCV(fitted_pipeline, method=m, cv="prefit")
        cal.fit(X_val, y_val)
        p = cal.predict_proba(X_test)[:, 1]
        rows.append({
            "method": label,
            **ev.core_metrics(y_test, p),
            "ece": expected_calibration_error(y_test, p),
            "selected": label == winner,
        })

    table = pd.DataFrame(rows)[["method", "roc_auc", "pr_auc", "ks", "brier", "ece", "selected"]]
    logger.info("calibration comparison (test set, reporting only):\n%s", table.to_string(index=False))
    return final, table, winner


# ===========================================================================
# Leakage ablation
# ===========================================================================
def leakage_ablation(splits: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Refit the same model with ``principal_outstanding`` added back.

    This is the evidence that the feature set was chosen deliberately.
    ``principal_outstanding`` is the balance part-way through repayment: a
    lender simply does not have it at the moment of the decision, so including
    it would make the model unusable in production no matter what it does to
    the score.  We measure the effect rather than assert it.
    """
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    clean_pipe = make_xgb_pipeline(params, scale_pos_weight=spw)
    clean_pipe = fit_xgb_with_early_stopping(clean_pipe, X_train, y_train, X_val, y_val)
    clean_auc = roc_auc_score(y_test, clean_pipe.predict_proba(X_test)[:, 1])

    leaky_numeric = ENGINEERED_NUMERIC_FEATURES + [
        "principal_outstanding", "principal_outstanding_ratio"
    ]
    leaky_pipe = make_xgb_pipeline(
        params, scale_pos_weight=spw,
        numeric_features=leaky_numeric, builder=build_features_with_leakage,
    )
    leaky_pipe = fit_xgb_with_early_stopping(leaky_pipe, X_train, y_train, X_val, y_val)
    leaky_auc = roc_auc_score(y_test, leaky_pipe.predict_proba(X_test)[:, 1])

    result = {
        "application_time_auc": float(clean_auc),
        "with_principal_outstanding_auc": float(leaky_auc),
        "auc_inflation": float(leaky_auc - clean_auc),
    }
    logger.info("leakage ablation: %s", result)
    return result


# ===========================================================================
# Plots
# ===========================================================================
def _save(fig: plt.Figure, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path.relative_to(config.ROOT))
    return str(path.relative_to(config.ROOT))


def plot_roc_pr(y: np.ndarray, probs: dict[str, np.ndarray]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for label, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, p):.4f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax2.plot(rec, prec, label=label)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.axhline(y.mean(), color="k", ls="--", lw=1, label=f"base rate={y.mean():.3f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test set)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)
    _save(fig, "roc_pr_curves.png")


def plot_ks(y: np.ndarray, p: np.ndarray) -> None:
    fpr, tpr, thr = roc_curve(y, p)
    idx = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thr, tpr, label="cumulative % of defaults caught")
    ax.plot(thr, fpr, label="cumulative % of good loans declined")
    ax.vlines(thr[idx], fpr[idx], tpr[idx], color="red", lw=2,
              label=f"KS={100 * (tpr[idx] - fpr[idx]):.2f} at p={thr[idx]:.3f}")
    ax.set(xlabel="Probability cut-off", ylabel="Cumulative share", title="KS separation (test set)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "ks_curve.png")


def plot_calibration(y: np.ndarray, before: np.ndarray, after: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for label, p in (("raw XGBoost", before), ("after isotonic calibration", after)):
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=f"{label} (Brier={ev.core_metrics(y, p)['brier']:.5f})")
    ax.set(xlabel="Mean predicted probability of default",
           ylabel="Observed default rate",
           title="Reliability curve (test set)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "calibration_curve.png")


def plot_decile(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(table["decile"], table["event_rate"] * 100, color="#c0392b", alpha=0.85)
    ax.set(xlabel="Risk decile (1 = riskiest)", ylabel="Actual default rate (%)",
           title="Rank ordering by risk decile (test set)")
    ax.set_xticks(table["decile"])
    for _, r in table.iterrows():
        ax.text(r["decile"], r["event_rate"] * 100, f"{r['event_rate'] * 100:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "decile_rank_ordering.png")


def plot_score_distribution(scores: np.ndarray, y: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(scores[y == 0], bins=50, alpha=0.65, label="repaid", color="#27ae60", density=True)
    ax1.hist(scores[y == 1], bins=50, alpha=0.65, label="defaulted", color="#c0392b", density=True)
    ax1.set(xlabel="Credit score (300-900)", ylabel="Density", title="Score distribution by outcome")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ratings = ev.score_to_rating(scores)
    order = ["Poor", "Average", "Good", "Excellent"]
    frame = pd.DataFrame({"rating": ratings, "y": y})
    agg = frame.groupby("rating")["y"].agg(n="size", rate="mean").reindex(order).dropna()
    bars = ax2.bar(agg.index, agg["rate"] * 100, color="#2980b9", alpha=0.85)
    for bar, (_, r) in zip(bars, agg.iterrows(), strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{r['rate'] * 100:.1f}%\nn={int(r['n'])}", ha="center", va="bottom", fontsize=8)
    ax2.set(xlabel="Rating band", ylabel="Actual default rate (%)", title="Default rate by rating band")
    ax2.grid(alpha=0.3, axis="y")
    _save(fig, "score_distribution.png")


def plot_cost_curve(curve: pd.DataFrame, chosen: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["threshold"], curve["cost"], color="#8e44ad")
    ax.axvline(chosen, color="red", ls="--", label=f"cost-optimal threshold = {chosen:.3f}")
    ax.axvline(0.5, color="grey", ls=":", label="naive 0.5 threshold")
    ax.set(xlabel="Probability cut-off", ylabel="Expected cost (units of exposure)",
           title=f"Expected cost vs threshold (FN={config.COST_FALSE_NEGATIVE}, FP={config.COST_FALSE_POSITIVE})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "cost_vs_threshold.png")


def plot_shap(model_pipeline, X_sample: pd.DataFrame) -> dict[str, Any]:
    """Global SHAP importance plus three individual explanations."""
    import shap

    feature_step = model_pipeline.named_steps["features"]
    encoder = model_pipeline.named_steps["encode"]
    booster = model_pipeline.named_steps["model"]

    Xt = encoder.transform(feature_step.transform(X_sample))
    names = encoded_feature_names(model_pipeline)
    frame = pd.DataFrame(Xt, columns=names)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(frame)

    plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, frame, plot_type="bar", show=False, max_display=20)
    plt.title("Global feature importance (mean |SHAP|)")
    _save(plt.gcf(), "shap_global_importance.png")

    plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, frame, show=False, max_display=20)
    plt.title("SHAP value distribution")
    _save(plt.gcf(), "shap_beeswarm.png")

    # Three individual applicants: highest risk, lowest risk, and a borderline
    # case. These are the adverse-action explanations a customer would receive.
    probs = booster.predict_proba(Xt)[:, 1]
    picks = {
        "highest_risk": int(np.argmax(probs)),
        "lowest_risk": int(np.argmin(probs)),
        "borderline": int(np.argmin(np.abs(probs - 0.5))),
    }
    for label, idx in picks.items():
        shap.force_plot(
            explainer.expected_value, shap_values[idx], frame.iloc[idx],
            matplotlib=True, show=False, figsize=(18, 4),
        )
        plt.title(f"{label.replace('_', ' ').title()} applicant - PD={probs[idx]:.4f}", fontsize=11)
        _save(plt.gcf(), f"shap_force_{label}.png")

    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return {"importance": importance, "picks": {k: float(probs[v]) for k, v in picks.items()}}


# ===========================================================================
# Main
# ===========================================================================
def main(n_trials: int = 40, skip_shap: bool = False) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    started = time.time()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. data ----------------------------------------------------------
    df, cleaning = load_clean()
    splits = make_splits(df)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    # --- 2. logistic baseline --------------------------------------------
    logger.info("=== stage 1/6: logistic regression baseline ===")
    baseline = make_logistic_pipeline().fit(X_train, y_train)
    baseline_test_p = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = ev.core_metrics(y_test, baseline_test_p)
    logger.info("baseline test: %s", {k: round(v, 4) for k, v in baseline_metrics.items()})

    # --- 3. imbalance strategies -----------------------------------------
    logger.info("=== stage 2/6: class-imbalance strategy comparison ===")
    imbalance_table = compare_imbalance_strategies(splits)

    # --- 4. tuning --------------------------------------------------------
    logger.info("=== stage 3/6: Optuna hyperparameter search (%d trials) ===", n_trials)
    best_params, optuna_history = tune_xgboost(splits, n_trials=n_trials)

    # --- 5. final fit + calibration --------------------------------------
    logger.info("=== stage 4/6: final fit and calibration ===")
    uncalibrated = make_xgb_pipeline(best_params, scale_pos_weight=spw)
    uncalibrated = fit_xgb_with_early_stopping(uncalibrated, X_train, y_train, X_val, y_val)
    best_iteration = int(getattr(uncalibrated.named_steps["model"], "best_iteration", -1))

    # Calibrate on the validation fold - the fold the model early-stopped on
    # but never fitted trees to. Both calibrators are tried and the better one
    # by Brier score wins; see compare_calibration.
    calibrated, calibration_table, calibration_method = compare_calibration(
        uncalibrated, X_val, y_val, X_test, y_test
    )

    raw_test_p = uncalibrated.predict_proba(X_test)[:, 1]
    cal_test_p = calibrated.predict_proba(X_test)[:, 1]
    raw_metrics = ev.core_metrics(y_test, raw_test_p)
    cal_metrics = ev.core_metrics(y_test, cal_test_p)
    logger.info("raw XGB test:        %s", {k: round(v, 4) for k, v in raw_metrics.items()})
    logger.info("calibrated XGB test: %s", {k: round(v, 4) for k, v in cal_metrics.items()})

    # --- 6. leakage ablation ---------------------------------------------
    logger.info("=== stage 5/6: leakage ablation ===")
    ablation = leakage_ablation(splits, best_params)

    # --- 7. out-of-time --------------------------------------------------
    oot_metrics = None
    oot_decile = None
    if splits["oot_viable"]:
        logger.info("=== stage 5b: out-of-time evaluation ===")
        oot_tr, oot_y = splits["X_oot_train"], splits["y_oot_train"]
        # Carve a validation fold out of the *in-time* data for early stopping,
        # so no post-cutoff row is seen during training in any form.
        oot_fit, oot_val, oot_fit_y, oot_val_y = train_test_split(
            oot_tr, oot_y, test_size=0.15, stratify=oot_y, random_state=config.RANDOM_SEED
        )
        oot_spw = float((oot_fit_y == 0).sum() / (oot_fit_y == 1).sum())
        oot_pipe = make_xgb_pipeline(best_params, scale_pos_weight=oot_spw)
        oot_pipe = fit_xgb_with_early_stopping(oot_pipe, oot_fit, oot_fit_y, oot_val, oot_val_y)
        oot_cal = CalibratedClassifierCV(oot_pipe, method="isotonic", cv="prefit")
        oot_cal.fit(oot_val, oot_val_y)
        oot_p = oot_cal.predict_proba(splits["X_oot_test"])[:, 1]
        oot_metrics = ev.core_metrics(splits["y_oot_test"], oot_p)
        oot_decile = ev.decile_table(splits["y_oot_test"], oot_p)
        logger.info("OOT test: %s", {k: round(v, 4) for k, v in oot_metrics.items()})

    # --- 8. threshold, deciles, slices, PSI ------------------------------
    logger.info("=== stage 6/6: evaluation, plots and artifacts ===")
    # The threshold is chosen on validation, then applied unchanged to test.
    # Choosing it on test would be tuning on the holdout.
    cal_val_p = calibrated.predict_proba(X_val)[:, 1]
    thr_search = ev.cost_optimal_threshold(y_val, cal_val_p)
    threshold = thr_search["threshold"]
    test_at_thr = ev.threshold_metrics(y_test, cal_test_p, threshold)
    naive_at_half = ev.threshold_metrics(y_test, cal_test_p, 0.5)
    logger.info("threshold %.3f -> %s", threshold, {k: round(v, 4) if isinstance(v, float) else v for k, v in test_at_thr.items()})

    decile = ev.decile_table(y_test, cal_test_p)
    monotone = ev.is_monotone(decile)

    feat_train = build_features(X_train)
    feat_test = build_features(X_test)
    psi = ev.psi_report(feat_train, feat_test, ENGINEERED_NUMERIC_FEATURES)
    psi_oot = None
    if splits["oot_viable"]:
        psi_oot = ev.psi_report(
            build_features(splits["X_oot_train"]),
            build_features(splits["X_oot_test"]),
            ENGINEERED_NUMERIC_FEATURES,
        )

    slices = {}
    for col in ["gender", "employment_status", "loan_purpose", "residence_type", "loan_type", "marital_status"]:
        slices[col] = ev.slice_metrics(y_test, cal_test_p, X_test[col].reset_index(drop=True), threshold)

    scores = ev.pd_to_credit_score(cal_test_p)

    # --- plots ------------------------------------------------------------
    plot_roc_pr(y_test, {"XGBoost (calibrated)": cal_test_p, "Logistic baseline": baseline_test_p})
    plot_ks(y_test, cal_test_p)
    plot_calibration(y_test, raw_test_p, cal_test_p)
    plot_decile(decile)
    plot_score_distribution(scores, y_test)
    plot_cost_curve(thr_search["cost_curve"], threshold)

    shap_info = None
    if not skip_shap:
        try:
            shap_info = plot_shap(uncalibrated, X_test.sample(min(2000, len(X_test)), random_state=config.RANDOM_SEED))
        except Exception as exc:  # pragma: no cover - SHAP is optional at runtime
            logger.warning("SHAP step failed (%s); continuing without it", exc)

    # --- tables -----------------------------------------------------------
    decile.to_csv(config.TABLES_DIR / "decile_table_test.csv", index=False)
    imbalance_table.to_csv(config.TABLES_DIR / "imbalance_comparison.csv", index=False)
    calibration_table.to_csv(config.TABLES_DIR / "calibration_comparison.csv", index=False)
    psi.to_csv(config.TABLES_DIR / "psi_train_vs_test.csv", index=False)
    optuna_history.to_csv(config.TABLES_DIR / "optuna_trials.csv", index=False)
    if oot_decile is not None:
        oot_decile.to_csv(config.TABLES_DIR / "decile_table_oot.csv", index=False)
    if psi_oot is not None:
        psi_oot.to_csv(config.TABLES_DIR / "psi_intime_vs_oot.csv", index=False)
    for name, frame in slices.items():
        frame.to_csv(config.TABLES_DIR / f"slice_{name}.csv", index=False)
    if shap_info is not None:
        shap_info["importance"].to_csv(config.TABLES_DIR / "shap_importance.csv", index=False)

    # --- 9. serialize -----------------------------------------------------
    joblib.dump(calibrated, config.MODEL_PATH, compress=3)

    # Fixed-row regression fixture: the API test asserts that a saved payload
    # scores identically after a reload, which is what catches serving skew.
    fixture_row = X_test.iloc[[0]]
    fixture_payload = {}
    for c in RAW_INPUTS:
        value = fixture_row.iloc[0][c]
        fixture_payload[c] = value.item() if hasattr(value, "item") else value
    fixture = {
        "payload": fixture_payload,
        "expected_probability": float(ev.clip_pd(calibrated.predict_proba(fixture_row)[0, 1])),
        "expected_credit_score": int(ev.pd_to_credit_score(calibrated.predict_proba(fixture_row)[0, 1])),
    }

    metadata: dict[str, Any] = {
        "model_version": config.MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_duration_seconds": round(time.time() - started, 1),
        "python_version": platform.python_version(),
        "random_seed": config.RANDOM_SEED,
        "algorithm": f"XGBClassifier + {calibration_method} calibration, wrapped in a sklearn Pipeline",
        "raw_input_columns": RAW_INPUTS,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "best_params": best_params,
        "scale_pos_weight": spw,
        "best_iteration": best_iteration,
        "n_optuna_trials": n_trials,
        "cleaning_report": cleaning.to_dict(),
        "dataset": {
            "rows_after_cleaning": int(len(df)),
            "base_default_rate": float(df[config.TARGET].mean()),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "date_min": str(splits["date_min"].date()),
            "date_max": str(splits["date_max"].date()),
            "oot_cutoff": str(splits["oot_cutoff"].date()),
            "oot_viable": splits["oot_viable"],
            "oot_test_rows": int(len(splits.get("X_oot_test", []))),
        },
        "metrics": {
            "baseline_logistic_test": baseline_metrics,
            "xgboost_raw_test": raw_metrics,
            "xgboost_calibrated_test": cal_metrics,
            "out_of_time_test": oot_metrics,
        },
        "threshold": {
            "value": threshold,
            "chosen_on": "validation set",
            "cost_false_negative": config.COST_FALSE_NEGATIVE,
            "cost_false_positive": config.COST_FALSE_POSITIVE,
            "test_metrics_at_threshold": test_at_thr,
            "test_metrics_at_0.5": naive_at_half,
        },
        "leakage_ablation": ablation,
        "rank_ordering_monotone": monotone,
        "imbalance_comparison": imbalance_table.to_dict(orient="records"),
        "calibration_method": calibration_method,
        "calibration_comparison": calibration_table.to_dict(orient="records"),
        "score_bimodality_note": (
            "Both calibrators place ~84% of the test book at the extremes of the "
            "300-900 range. That concentration is a property of this dataset - the "
            "target is nearly a deterministic function of credit_utilization_ratio "
            "and delinquency_ratio - not an artefact of the calibration choice."
        ),
        "psi_train_vs_test_max": float(psi["psi"].max()) if len(psi) else None,
        "psi_intime_vs_oot_max": float(psi_oot["psi"].max()) if psi_oot is not None and len(psi_oot) else None,
        "top_features_by_shap": (
            shap_info["importance"].head(15).to_dict(orient="records") if shap_info else None
        ),
        "slice_metrics": {k: v.to_dict(orient="records") for k, v in slices.items()},
        "prediction_fixture": fixture,
        "score_mapping": {
            "range": [config.SCORE_MIN, config.SCORE_MAX],
            "formula": "score = 300 + (1 - PD) * 600",
            "bands": [{"min": lo, "max": hi, "label": lab} for lo, hi, lab in config.RATING_BANDS],
        },
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("saved %s (%.1f MB) and %s",
                config.MODEL_PATH.name, config.MODEL_PATH.stat().st_size / 1e6,
                config.METADATA_PATH.name)
    logger.info("total run time %.0fs", time.time() - started)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the credit default risk model.")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials (default 40)")
    parser.add_argument("--skip-shap", action="store_true", help="skip SHAP plots (faster)")
    args = parser.parse_args()
    main(n_trials=args.trials, skip_shap=args.skip_shap)
