"""
federated_icu/client.py
=======================
FLClient — one federated learning participant (hospital, region, or cluster).

Fixes applied:
  v3: Train on ALL local data (no wasted 20% local split).
  v4: warm_start=True + two-stage fit for correct FedAvg weight injection.
  v5: Use base C=0.01 (no K-scaling). Cold-start each round from global weights.
      The warm_start from centralized was giving FL an unfair head-start.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import FEATURE_COLS, TARGET_COL
from .data import Preprocessor
from .logger import get_logger
from .models import (
    extract_weights,
    get_feature_importance,
    inject_weights,
    make_fl_client_model,
)

log = get_logger("client")


class FLClient:
    """One FL participant = one data silo."""

    def __init__(
        self,
        name:         str,
        X_raw,
        y,
        preprocessor: Preprocessor,
        algorithm:    str = "logistic_regression",
        fedprox_mu:   float = 0.0,
        random_state: int = 42,
    ):
        self.name         = name
        self.y            = y
        self.n            = len(y)
        self.n_pos        = int(y.sum())
        self.n_neg        = int((y == 0).sum())
        self.algorithm    = algorithm
        self.fedprox_mu   = fedprox_mu
        self.random_state = random_state

        if len(X_raw) > 0:
            self.X: np.ndarray = preprocessor.transform(X_raw[FEATURE_COLS])
        else:
            self.X = np.zeros((0, len(FEATURE_COLS)))

        self.model    = None
        self._n_train = 0

    def local_train(
        self,
        global_weights: Optional[dict],
        fl_round:       int = 1,
        progress_cb:    Optional[Callable] = None,
    ) -> dict:
        """
        Train on ALL local data, warm-starting from global_weights.

        Protocol (v4 — correct FedAvg for LR):
          1. Create client model with warm_start=True, max_iter=100.
          2. Tiny seed fit to initialise sklearn internal arrays.
          3. inject_weights() → overwrite coef_ with server global weights.
          4. fit() continues lbfgs FROM those weights (warm_start=True).
          5. Return refined weights for FedAvg aggregation.
        """
        if self.n < 5 or self.n_pos < 1 or self.n_neg < 1:
            log.debug("Client '%s' skipped (n=%d, pos=%d, neg=%d)",
                      self.name, self.n, self.n_pos, self.n_neg)
            return self._skipped_result()

        X_tr, y_tr   = self.X, self.y.values
        self._n_train = len(X_tr)

        self.model = make_fl_client_model(self.algorithm, self.random_state)

        if global_weights:
            # Seed fit to initialise internal arrays
            pos_i = np.where(y_tr == 1)[0][0]
            neg_i = np.where(y_tr == 0)[0][0]
            self.model.fit(X_tr[[pos_i, neg_i]], y_tr[[pos_i, neg_i]])

            # Inject global weights then refine
            inject_weights(self.model, global_weights)

            if (self.fedprox_mu > 0
                    and global_weights.get("type") == "lr"
                    and self.algorithm == "logistic_regression"):
                self.model.fit(X_tr, y_tr)
                local_coef  = self.model.coef_.copy()
                global_coef = np.array(global_weights["coef"])
                mu = min(self.fedprox_mu, 1.0)
                self.model.coef_ = (1 - mu) * local_coef + mu * global_coef
            else:
                self.model.fit(X_tr, y_tr)
        else:
            self.model.fit(X_tr, y_tr)

        # In-sample AUROC — diagnostic only
        y_prob      = self.model.predict_proba(X_tr)[:, 1]
        n_unique    = len(np.unique(y_tr))
        local_auroc = roc_auc_score(y_tr, y_prob)           if n_unique > 1 else 0.0
        local_ap    = average_precision_score(y_tr, y_prob) if n_unique > 1 else 0.0

        log.debug("  [%s] round %d  n_train=%d  local_AUROC(in-sample)=%.4f",
                  self.name, fl_round, self._n_train, local_auroc)

        if progress_cb:
            try:
                progress_cb(self.name, local_auroc, self.n)
            except Exception:
                pass

        return {
            "name":                self.name,
            "n":                   self.n,
            "n_train":             self._n_train,
            "n_pos":               self.n_pos,
            "n_neg":               self.n_neg,
            "local_auroc":         round(float(local_auroc), 4),
            "local_ap":            round(float(local_ap),    4),
            "weights":             extract_weights(self.model),
            "feature_importances": get_feature_importance(self.model),
            "skipped":             False,
        }

    def _skipped_result(self) -> dict:
        return {
            "name":                self.name,
            "n":                   self.n,
            "n_train":             0,
            "n_pos":               self.n_pos,
            "n_neg":               self.n_neg,
            "local_auroc":         0.0,
            "local_ap":            0.0,
            "weights":             {},
            "feature_importances": [0.0] * len(FEATURE_COLS),
            "skipped":             True,
        }
