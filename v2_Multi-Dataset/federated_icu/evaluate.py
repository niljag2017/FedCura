"""
federated_icu/evaluate.py
=========================
Evaluation helpers and model checkpointing.

Public API
----------
evaluate_model(model, gts)               → dict
evaluate_centralized(model, X, y, cv)    → dict
save_checkpoint(run_id, round_n, model, metrics, cfg)
load_checkpoint(run_id, checkpoint_dir)  → dict | None
list_checkpoints(checkpoint_dir)         → list[dict]
"""
from __future__ import annotations

import json
import os
import pickle
import time
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import FEATURE_COLS, TARGET_COL
from .logger import get_logger

log = get_logger("evaluate")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute AUROC, AP, and a condensed classification report
    for any sklearn-compatible model.
    """
    if len(np.unique(y)) < 2:
        return {"auroc": 0.0, "ap": 0.0, "report": {}, "n": len(y)}

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = float(roc_auc_score(y, y_prob))
    ap    = float(average_precision_score(y, y_prob))

    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    return {
        "auroc":     round(auroc, 4),
        "ap":        round(ap,    4),
        "precision": round(report.get("1", {}).get("precision", 0.0), 4),
        "recall":    round(report.get("1", {}).get("recall",    0.0), 4),
        "f1":        round(report.get("1", {}).get("f1-score",  0.0), 4),
        "n":         len(y),
    }


def evaluate_centralized(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Cross-validation AUROC for the centralized strategy.
    Returns mean, std, and per-fold scores.
    """
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    return {
        "cv_auroc": round(float(scores.mean()), 4),
        "cv_std":   round(float(scores.std()),  4),
        "cv_folds": scores.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════════════

def _checkpoint_path(checkpoint_dir: str, run_id: str, tag: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"{run_id}_{tag}.pkl")


def save_checkpoint(
    run_id:         str,
    tag:            str,           # e.g. "round_3" or "final"
    model,
    metrics:        dict,
    cfg_dict:       dict,
    checkpoint_dir: str = "checkpoints",
) -> str:
    """
    Pickle the model + metadata to disk.
    Returns the path of the saved file.
    """
    path = _checkpoint_path(checkpoint_dir, run_id, tag)
    payload = {
        "run_id":    run_id,
        "tag":       tag,
        "timestamp": time.time(),
        "metrics":   metrics,
        "config":    cfg_dict,
        "model":     model,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Checkpoint saved: %s", path)
    return path


def load_checkpoint(path: str) -> Optional[dict]:
    """Load a checkpoint file.  Returns None if file not found or corrupt."""
    if not os.path.exists(path):
        log.warning("Checkpoint not found: %s", path)
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        log.error("Failed to load checkpoint %s: %s", path, exc)
        return None


def list_checkpoints(checkpoint_dir: str = "checkpoints") -> list:
    """
    Return a summary list of all checkpoint files in checkpoint_dir,
    sorted by timestamp descending (newest first).
    """
    if not os.path.isdir(checkpoint_dir):
        return []

    summaries = []
    for fname in os.listdir(checkpoint_dir):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(checkpoint_dir, fname)
        try:
            ck = load_checkpoint(fpath)
            if ck:
                summaries.append({
                    "file":      fname,
                    "path":      fpath,
                    "run_id":    ck.get("run_id", "?"),
                    "tag":       ck.get("tag", "?"),
                    "timestamp": ck.get("timestamp", 0),
                    "metrics":   ck.get("metrics", {}),
                })
        except Exception:
            pass

    return sorted(summaries, key=lambda x: x["timestamp"], reverse=True)
