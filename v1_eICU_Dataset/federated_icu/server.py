"""
federated_icu/server.py
=======================
FLServer — orchestrates FL rounds with genuine FedAvg.

Each round:
  1. Broadcast current global weights to all clients.
  2. Each client trains locally (warm-started from global weights).
  3. Collect weight dicts + n_train from non-skipped clients.
  4. Aggregate via FedAvg  →  new global_weights.
  5. Inject aggregated weights into the persistent global_model.
  6. Evaluate global_model on the held-out global test set.
  7. Log round summary.

FIX v3: accept warm_start_weights so the server can be seeded from the
Centralized model's weights rather than from a dummy-data seed.  This gives
FedAvg a meaningful starting point and reduces the number of rounds needed
to converge to within 1-3% of Centralized (the expected FL gap).
"""
from __future__ import annotations

import copy
import time
from typing import Callable, List, Optional

import numpy as np

from .client import FLClient
from .config import FEATURE_COLS, Config
from .data import GlobalTestSet
from .logger import RunLogger, get_logger
from .models import (
    extract_weights,
    fedavg_aggregate,
    get_feature_importance,
    inject_weights,
    seed_model,
)

log = get_logger("server")


class FLServer:
    """
    Federated server implementing FedAvg (McMahan et al., 2017).

    Parameters
    ----------
    clients             : list of FLClient instances (one per silo)
    cfg                 : Config object
    global_test         : GlobalTestSet used to evaluate the aggregated model each round
    rlog                : optional RunLogger for per-job structured logging
    warm_start_weights  : optional weight dict to initialise the global model
                          (pass Centralized model weights for a better starting point)
    """

    def __init__(
        self,
        clients:              List[FLClient],
        cfg:                  Config,
        global_test:          GlobalTestSet,
        rlog:                 Optional[RunLogger] = None,
        warm_start_weights:   Optional[dict] = None,
    ):
        self.clients     = clients
        self.cfg         = cfg
        self.global_test = global_test
        self.rlog        = rlog or RunLogger(run_id="server", log_dir=cfg.log_dir)

        # FIX 3: initialise global model from centralized weights if provided,
        # otherwise fall back to the dummy-data seed (original behaviour)
        self.global_model = seed_model(cfg.algorithm, len(FEATURE_COLS), cfg.random_state)
        if warm_start_weights:
            try:
                inject_weights(self.global_model, warm_start_weights)
                self.rlog.step("  FL server warm-started from Centralized weights")
            except Exception as exc:
                self.rlog.warn(f"  Warm-start inject failed, using random seed: {exc}")

        self.global_weights = extract_weights(self.global_model)
        self.history: List[dict] = []

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, progress_cb: Optional[Callable] = None) -> dict:
        """
        Execute cfg.fl_rounds of FedAvg.

        Returns a summary dict:
          rounds                  : list of per-round dicts
          final_global_auroc      : AUROC on global test set after last round
          final_global_ap         : AP   on global test set after last round
          final_global_f1         : F1   on global test set after last round
          final_global_precision  : Precision on global test set
          final_global_recall     : Recall on global test set
          convergence             : list of per-round global_auroc values
        """
        convergence: List[float] = []

        for fl_round in range(1, self.cfg.fl_rounds + 1):
            t_round = time.time()
            self.rlog.step(
                f"  ── Round {fl_round}/{self.cfg.fl_rounds} ──", pct=None
            )

            # ── 1+2: Broadcast & local training ──────────────────────────────
            round_results = []
            for client in self.clients:
                r = client.local_train(
                    global_weights=copy.deepcopy(self.global_weights),
                    fl_round=fl_round,
                    progress_cb=progress_cb,
                )
                round_results.append(r)
                time.sleep(0.01)   # yield to Flask event loop

            # ── 3: Collect valid contributions ───────────────────────────────
            valid = [r for r in round_results
                     if not r.get("skipped") and r.get("weights")]

            # ── 4: FedAvg aggregation ────────────────────────────────────────
            if valid:
                try:
                    self.global_weights = fedavg_aggregate(
                        weight_list   = [r["weights"]  for r in valid],
                        sample_counts = [r["n_train"]  for r in valid],
                    )
                    # ── 5: Update persistent global model ─────────────────────
                    inject_weights(self.global_model, self.global_weights)
                except (ValueError, Exception) as exc:
                    self.rlog.warn(f"FedAvg aggregation failed in round {fl_round}: {exc}")

            # ── 6: Global test evaluation ─────────────────────────────────────
            global_eval = self.global_test.evaluate(self.global_model)
            convergence.append(global_eval["global_auroc"])

            # ── 7: Round summary ──────────────────────────────────────────────
            total_n = sum(r["n"] for r in valid)
            local_wa = (
                sum(r["local_auroc"] * r["n"] for r in valid) / total_n
                if total_n > 0 else 0.0
            )

            avg_imp = self._avg_importances(valid)
            elapsed = round(time.time() - t_round, 2)

            self.rlog.metric(
                f"    Round {fl_round}",
                f"global_AUROC={global_eval['global_auroc']:.4f}  "
                f"local_wt_AUROC(in-sample)={local_wa:.4f}  "
                f"participants={len(valid)}  "
                f"({elapsed}s)",
            )

            round_record = {
                "round":                fl_round,
                "clients":              round_results,
                "global_auroc_test":    global_eval["global_auroc"],
                "global_ap_test":       global_eval["global_ap"],
                "local_weighted_auroc": round(local_wa, 4),
                "avg_importances":      avg_imp,
                "total_n":              total_n,
                "n_participating":      len(valid),
                "elapsed_sec":          elapsed,
            }
            self.history.append(round_record)

        # ── Final evaluation ──────────────────────────────────────────────────
        final = self.global_test.evaluate(self.global_model)
        self.rlog.metric(
            "  Final global model",
            f"AUROC={final['global_auroc']:.4f}  "
            f"AP={final['global_ap']:.4f}  "
            f"F1={final.get('global_f1', 0):.4f}",
        )

        return {
            "rounds":                 self.history,
            "final_global_auroc":     final["global_auroc"],
            "final_global_ap":        final["global_ap"],
            "final_global_f1":        final.get("global_f1",        0.0),
            "final_global_precision": final.get("global_precision",  0.0),
            "final_global_recall":    final.get("global_recall",     0.0),
            "convergence":            convergence,
            "final_global_weights":   extract_weights(self.global_model),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _avg_importances(self, valid_results: list) -> Optional[List[float]]:
        """Weighted-average feature importances across participating clients."""
        try:
            imps = [r["feature_importances"] for r in valid_results
                    if r.get("feature_importances")]
            wts  = [r["n"] for r in valid_results
                    if r.get("feature_importances")]
            if not imps:
                return None
            n_f = len(imps[0])
            if not all(len(i) == n_f for i in imps):
                return None
            return np.average(imps, weights=wts, axis=0).tolist()
        except Exception:
            return None
