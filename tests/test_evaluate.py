"""Metric implementations, checked against cases with known answers."""
from __future__ import annotations

import numpy as np
import pytest

from src import evaluate as ev


def test_perfect_separation_gives_ks_100():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.96, 0.97, 0.98, 0.99])
    assert ev.ks_statistic(y, p) == pytest.approx(100.0)
    assert ev.core_metrics(y, p)["roc_auc"] == pytest.approx(1.0)


def test_random_predictions_give_ks_near_zero():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.5, 5000)
    p = rng.random(5000)
    assert ev.ks_statistic(y, p) < 6.0


def test_gini_is_the_auc_identity():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 2000)
    p = np.clip(0.3 + 0.2 * y + rng.normal(0, 0.15, 2000), 0, 1)
    m = ev.core_metrics(y, p)
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)


def test_psi_of_a_distribution_against_itself_is_zero():
    rng = np.random.default_rng(2)
    x = rng.normal(size=10_000)
    assert ev.population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_real_shift():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 10_000)
    shifted = rng.normal(2.5, 1, 10_000)
    assert ev.population_stability_index(reference, shifted) > 0.25


def test_psi_handles_a_constant_feature():
    constant = np.ones(1000)
    assert ev.population_stability_index(constant, constant) == 0.0


def test_decile_table_bins_evenly_and_rank_orders():
    rng = np.random.default_rng(4)
    n = 10_000
    p = rng.random(n)
    y = rng.binomial(1, p)  # probability *is* the truth, so ranking is perfect
    table = ev.decile_table(y, p)
    assert len(table) == 10
    assert table["n"].nunique() == 1  # equal-sized deciles
    assert table["events"].sum() == y.sum()
    assert ev.is_monotone(table)


def test_cost_threshold_moves_below_half_when_false_negatives_are_costlier():
    """The whole point of the cost assumption: a missed default is ~10x worse,
    so the cut-off must sit well below the naive 0.5."""
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.086, 20_000)
    p = np.clip(0.086 + 0.2 * y + rng.normal(0, 0.1, 20_000), 0.001, 0.999)
    expensive_fn = ev.cost_optimal_threshold(y, p, cost_fn=0.60, cost_fp=0.06)
    symmetric = ev.cost_optimal_threshold(y, p, cost_fn=0.10, cost_fp=0.10)
    assert expensive_fn["threshold"] < symmetric["threshold"]


def test_score_mapping_is_monotone_decreasing_in_pd():
    probs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    scores = ev.pd_to_credit_score(probs)
    assert scores[0] == 900 and scores[-1] == 300
    assert np.all(np.diff(scores) < 0)


def test_rating_bands_cover_the_whole_range():
    scores = np.arange(300, 901)
    ratings = ev.score_to_rating(scores)
    assert set(ratings) == {"Poor", "Average", "Good", "Excellent"}
    assert ev.score_to_rating(np.array([899]))[0] == "Excellent"
    assert ev.score_to_rating(np.array([300]))[0] == "Poor"


def test_threshold_metrics_confusion_matrix_adds_up():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.6, 0.9, 0.2, 0.3, 0.8])
    m = ev.threshold_metrics(y, p, 0.5)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y)
    assert m["tp"] == 2 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 2
