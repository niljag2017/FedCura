"""
federated_icu/models.py
=======================
Model factory, weight serialisation, and FedAvg aggregation.

All models operate on pre-processed (imputed + scaled) numpy arrays.
Both LR and MLP expose numeric weight tensors for genuine FedAvg.

Public API
----------
make_model(algorithm, random_state)           → sklearn estimator for CENTRALIZED use
make_fl_client_model(algorithm, random_state) → sklearn estimator for FL CLIENT use
seed_model(algorithm, n_features, rs)         → fitted dummy model (for weight shape)
extract_weights(model)                        → dict
inject_weights(model, weights)                → None  (in-place)
fedavg_aggregate(weight_list, n_samples)      → dict
get_feature_importance(model, n_features)     → list[float]

Key design (FIX v4 — warm_start):
  Centralized strategy uses make_model():     warm_start=False, max_iter=2000
    → full convergence from scratch on pooled data.
  FL clients use make_fl_client_model():      warm_start=True,  max_iter=100
    → each round CONTINUES from injected global weights (true FedAvg for LR).
    With warm_start=False (old behaviour), sklearn LR ignores the injected
    coef_ and re-solves from scratch, making inject_weights a no-op and
    rendering FedAvg equivalent to simple model averaging of independently
    trained local models — not true federated learning.
"""
from __future__ import annotations

import copy
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from .config import FEATURE_COLS
from .logger import get_logger

log = get_logger("models")

# ── Sentinel classes ─────────────────────────────────────────────────────────
_LR_SOLVER  = "lbfgs"
_MLP_HIDDEN = (64, 32)


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def make_model(algorithm: str, random_state: int = 42):
    """
    Return a fresh, unfitted estimator for CENTRALIZED training only.
    Uses warm_start=False and max_iter=2000 for full convergence on pooled data.
    FL clients must use make_fl_client_model() instead.
    """
    if algorithm == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=_MLP_HIDDEN,
            activation="relu",
            solver="adam",
            max_iter=500,
            alpha=0.01,
            learning_rate_init=0.001,
            random_state=random_state,
            warm_start=False,
            early_stopping=False,
        )
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=0.01,
        solver=_LR_SOLVER,
        random_state=random_state,
        warm_start=False,   # centralized: always re-solve from scratch
    )


def make_fl_client_model(algorithm: str, random_state: int = 42):
    """
    Return a fresh, unfitted estimator for FL CLIENT use.

    FIX v4: uses warm_start=True + max_iter=100 so that after inject_weights()
    sets coef_ to the global model's weights, fit() CONTINUES optimising from
    that point rather than re-solving from scratch.  This is the correct
    FedAvg behaviour for LR:
      round k: server broadcasts w_global
               client injects w_global into coef_
               client calls fit() → lbfgs takes up to 100 steps from w_global
               client returns refined local weights for aggregation
    With warm_start=False (old code), fit() ignores coef_ entirely,
    making inject_weights() a no-op and breaking FedAvg convergence.
    """
    if algorithm == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=_MLP_HIDDEN,
            activation="relu",
            solver="adam",
            max_iter=100,   # partial steps per round
            alpha=0.01,
            learning_rate_init=0.001,
            random_state=random_state,
            warm_start=True,   # continue from injected weights
            early_stopping=False,
        )
    return LogisticRegression(
        class_weight="balanced",
        max_iter=100,       # partial local steps per round (not full convergence)
        C=0.01,
        solver=_LR_SOLVER,
        random_state=random_state,
        warm_start=True,    # continue optimisation from injected global weights
    )


def seed_model(algorithm: str, n_features: int, random_state: int = 42):
    """
    Return a model already fitted on dummy data so its internal arrays
    (coef_, coefs_, …) are initialised.  Used by FLServer to build the
    first global weight vector before any client trains.
    """
    model = make_model(algorithm, random_state)
    rng   = np.random.default_rng(random_state)
    n_rows = max(20, n_features + n_features % 2)
    X_d   = rng.normal(size=(n_rows, n_features))
    y_d   = np.array([0, 1] * (n_rows // 2))
    model.fit(X_d, y_d)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  WEIGHT SERIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_weights(model) -> dict:
    """
    Return a JSON-serialisable dict of model parameters.
    Shape information is embedded so injection can validate.
    """
    if isinstance(model, LogisticRegression) and hasattr(model, "coef_"):
        return {
            "type":      "lr",
            "coef":      model.coef_.tolist(),          # (1, n_features)
            "intercept": model.intercept_.tolist(),     # (1,)
        }
    if isinstance(model, MLPClassifier) and hasattr(model, "coefs_"):
        return {
            "type":       "mlp",
            "coefs":      [w.tolist() for w in model.coefs_],
            "intercepts": [b.tolist() for b in model.intercepts_],
        }
    log.warning("extract_weights: model has no weight attributes — returning empty dict")
    return {}


def inject_weights(model, weights: dict) -> None:
    """
    Overwrite model parameters with aggregated weights (in-place).
    Raises ValueError on shape mismatch to surface aggregation bugs early.
    """
    if not weights:
        return

    kind = weights.get("type")

    if kind == "lr":
        coef      = np.array(weights["coef"])
        intercept = np.array(weights["intercept"])
        if hasattr(model, "coef_") and model.coef_.shape != coef.shape:
            raise ValueError(
                f"LR weight shape mismatch: "
                f"model {model.coef_.shape} vs weights {coef.shape}"
            )
        model.coef_      = coef
        model.intercept_ = intercept
        model.classes_   = np.array([0, 1])

    elif kind == "mlp":
        coefs      = [np.array(w) for w in weights["coefs"]]
        intercepts = [np.array(b) for b in weights["intercepts"]]
        if hasattr(model, "coefs_"):
            for l, (mc, wc) in enumerate(zip(model.coefs_, coefs)):
                if mc.shape != wc.shape:
                    raise ValueError(
                        f"MLP layer {l} shape mismatch: "
                        f"model {mc.shape} vs weights {wc.shape}"
                    )
        model.coefs_         = coefs
        model.intercepts_    = intercepts
        model.n_outputs_     = 1
        model.out_activation_ = "logistic"
        model.n_iter_        = 1
        model.n_layers_      = len(coefs) + 1
        model.n_features_in_ = coefs[0].shape[0]
        model.t_             = 1
        # Ensure classes_ is set so predict_proba works
        model.classes_       = np.array([0, 1])
        model._no_improvement_count = 0
        # _label_binarizer is required by MLPClassifier.predict_proba()
        # but is only set during fit(); inject it manually here.
        if not hasattr(model, "_label_binarizer"):
            lb = LabelBinarizer()
            lb.fit([0, 1])
            model._label_binarizer = lb

    else:
        log.warning("inject_weights: unknown weight type '%s'", kind)


# ═══════════════════════════════════════════════════════════════════════════════
#  FEDAVG AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def fedavg_aggregate(
    weight_list:  List[dict],
    sample_counts: List[int],
) -> dict:
    """
    Weighted average of model weight dicts — genuine FedAvg
    (McMahan et al., 2017).  Each client's contribution is proportional
    to its number of *training* samples.

    Raises ValueError if weight_list is empty or types are inconsistent.
    """
    if not weight_list:
        raise ValueError("fedavg_aggregate received an empty weight list")

    kind = weight_list[0].get("type")
    if not all(w.get("type") == kind for w in weight_list):
        raise ValueError("fedavg_aggregate: mixed weight types in weight_list")

    total = sum(sample_counts)
    if total == 0:
        raise ValueError("fedavg_aggregate: total sample count is zero")

    alpha = [n / total for n in sample_counts]   # normalised weights

    if kind == "lr":
        agg_coef = np.sum(
            [np.array(w["coef"]) * a for w, a in zip(weight_list, alpha)], axis=0
        )
        agg_int = np.sum(
            [np.array(w["intercept"]) * a for w, a in zip(weight_list, alpha)], axis=0
        )
        return {"type": "lr", "coef": agg_coef.tolist(), "intercept": agg_int.tolist()}

    if kind == "mlp":
        n_layers = len(weight_list[0]["coefs"])
        agg_coefs = [
            np.sum(
                [np.array(w["coefs"][l]) * a for w, a in zip(weight_list, alpha)],
                axis=0,
            )
            for l in range(n_layers)
        ]
        agg_ints = [
            np.sum(
                [np.array(w["intercepts"][l]) * a for w, a in zip(weight_list, alpha)],
                axis=0,
            )
            for l in range(n_layers)
        ]
        return {
            "type":       "mlp",
            "coefs":      [c.tolist() for c in agg_coefs],
            "intercepts": [b.tolist() for b in agg_ints],
        }

    raise ValueError(f"fedavg_aggregate: unknown weight type '{kind}'")


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_importance(model, n_features: int = len(FEATURE_COLS)) -> List[float]:
    """
    Return a list of feature importance values (one per feature).

    LR:  absolute value of coefficients.
    MLP: approximate input importance by propagating absolute weight magnitudes
         through all layers (column-wise L1 product).
    """
    if isinstance(model, LogisticRegression) and hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        # Normalise to [0, 1] for comparability across runs
        mx = imp.max()
        return (imp / mx if mx > 0 else imp).tolist()

    if isinstance(model, MLPClassifier) and hasattr(model, "coefs_"):
        imp = np.abs(model.coefs_[0])          # shape: (n_features, hidden_1)
        for layer in model.coefs_[1:]:
            imp = imp @ np.abs(layer)           # propagate through layers
        imp = imp.squeeze()
        mx  = imp.max()
        return (imp / mx if mx > 0 else imp).tolist()

    return [0.0] * n_features
