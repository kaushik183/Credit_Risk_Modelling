"""Pipeline construction shared by training, the ablation and serving.

Kept separate from ``train.py`` so that ``joblib.load`` at serving time does
not drag Optuna, SHAP and matplotlib into the API process.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config
from .features import (
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    build_features,
)


def _feature_names_out(transformer, input_features):  # noqa: ARG001
    """Report the engineered column names, not the raw input names.

    ``build_features`` replaces its input columns entirely, so the default
    "one-to-one" naming would be wrong and sklearn rejects it.
    """
    builder = getattr(transformer, "func", build_features)
    if builder is build_features:
        return np.asarray(FEATURE_COLUMNS, dtype=object)
    # The leakage-ablation builder appends two extra columns.
    return np.asarray(
        FEATURE_COLUMNS + ["principal_outstanding", "principal_outstanding_ratio"],
        dtype=object,
    )


def make_feature_step(builder: Callable[[pd.DataFrame], pd.DataFrame] = build_features) -> FunctionTransformer:
    """Wrap ``features.build_features`` as the first step of the pipeline.

    Putting feature engineering *inside* the serialized pipeline is what makes
    training/serving skew structurally impossible: the API hands over raw
    applicant fields and the model file itself decides what a feature is.
    """
    return FunctionTransformer(builder, validate=False, feature_names_out=_feature_names_out)


def make_encoder(numeric_features: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    """One-hot the categoricals; optionally standardise the numerics.

    ``handle_unknown="ignore"`` means a category the model never saw becomes an
    all-zero block instead of raising - the API validates against an enum
    first, so this is only a second line of defence.

    Trees do not care about feature scale, so ``scale_numeric`` is False for
    XGBoost and True for the logistic-regression baseline, which does.
    """
    numeric_step: Any = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
                CATEGORICAL_FEATURES,
            ),
            ("num", numeric_step, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_xgb(params: dict[str, Any] | None = None, scale_pos_weight: float | None = None) -> XGBClassifier:
    """An XGBClassifier with the project's fixed defaults."""
    defaults: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1,
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "gamma": 0.0,
        "reg_lambda": 1.0,
    }
    defaults.update(params or {})
    if scale_pos_weight is not None:
        defaults["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**defaults)


def make_xgb_pipeline(
    params: dict[str, Any] | None = None,
    scale_pos_weight: float | None = None,
    numeric_features: list[str] | None = None,
    builder: Callable[[pd.DataFrame], pd.DataFrame] = build_features,
) -> Pipeline:
    """raw application fields -> features -> one-hot -> XGBoost."""
    numeric_features = numeric_features or ENGINEERED_NUMERIC_FEATURES
    return Pipeline(
        steps=[
            ("features", make_feature_step(builder)),
            ("encode", make_encoder(numeric_features, scale_numeric=False)),
            ("model", make_xgb(params, scale_pos_weight)),
        ]
    )


def make_logistic_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    """The industry-standard scorecard baseline.

    Logistic regression on one-hot categoricals with standardised numerics.
    ``class_weight="balanced"`` is the logistic analogue of
    ``scale_pos_weight`` - it up-weights the 8.6% of rows that defaulted so the
    model does not simply predict "everyone repays".
    """
    return Pipeline(
        steps=[
            ("features", make_feature_step()),
            ("encode", make_encoder(ENGINEERED_NUMERIC_FEATURES, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    random_state=config.RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def encoded_feature_names(pipeline: Pipeline) -> list[str]:
    """Column names after one-hot encoding - used to label SHAP output."""
    return list(pipeline.named_steps["encode"].get_feature_names_out())


def fit_xgb_with_early_stopping(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    early_stopping_rounds: int = 50,
) -> Pipeline:
    """Fit the pipeline, using the validation fold for early stopping.

    sklearn's ``Pipeline.fit`` cannot pass a transformed ``eval_set`` to the
    final estimator, so we fit the preprocessing steps on train, transform the
    validation fold through them, and then fit XGBoost with early stopping.
    The already-fitted steps are reassembled into the same Pipeline object, so
    what we serialize is still one artifact that takes raw fields in.
    """
    feature_step = pipeline.named_steps["features"]
    encoder = pipeline.named_steps["encode"]
    model = pipeline.named_steps["model"]

    Xt_train = encoder.fit_transform(feature_step.fit_transform(X_train))
    Xt_val = encoder.transform(feature_step.transform(X_val))

    model.set_params(early_stopping_rounds=early_stopping_rounds)
    model.fit(Xt_train, y_train, eval_set=[(Xt_val, y_val)], verbose=False)
    return pipeline
