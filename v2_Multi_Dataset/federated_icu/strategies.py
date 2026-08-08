"""
federated_icu/strategies.py
============================
All six federation strategies, each returning a uniform result dict.

Strategy 1 — Centralized (pooled baseline)
Strategy 2 — FedAvg by geographic region
Strategy 3 — Quality-gated per-hospital FedAvg
Strategy 4 — FedProx by geographic region
Strategy 5 — Non-IID simulation via Dirichlet partitioning

Every strategy produces:
  strategy, n, n_pos, readmit_rate,
  cv_auroc, cv_std,
  ho_auroc,  ho_ap,        ← global held-out test set (identical for all)
  global_auroc, global_ap, ← same as ho_*
  global_f1, global_precision, global_recall,
  feature_importances,
  convergence,              ← list of per-round global_auroc values
  clients,                  ← list of per-client dicts
  [strategy-specific keys]

Fixes (v3):
  FIX 1 — client.py: FL clients now train on 100% of their local data
           (the previous local 80/20 split was discarding training data
            for a local metric never used in aggregation or decisions).
  FIX 2 — strategies.py: missing region/cluster labels filled as "Unknown"
           so no patients are silently dropped from FL that Centralized sees.
  FIX 3 — server.py: seed_model warm-started from centralized weights
           so FL begins from a sensible parameter vector, not random noise,
           reducing the rounds needed to converge.
"""
from __future__ import annotations

import copy
from typing import Callable, List, Optional

import numpy as np

from .client import FLClient
from .config import FEATURE_COLS, TARGET_COL, Config
from .data import GlobalTestSet, Preprocessor, assign_clusters, dirichlet_partition
from .evaluate import evaluate_centralized, evaluate_model, save_checkpoint
from .logger import RunLogger, get_logger
from .models import (
    get_feature_importance,
    make_model,
    seed_model,
    extract_weights,
    inject_weights,
)
from .server import FLServer

log = get_logger("strategies")


def _feature_cols():
    """Always returns the current _feature_cols() — respects dataset overrides."""
    from .config import FEATURE_COLS as _FC
    return _FC


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _slim_server_output(server_output: dict) -> dict:
    """
    Return a copy of server_output with per-client weight arrays stripped.
    MLP weight dicts are ~74 KB each; with 3 rounds × 4 clients × 6 strategies
    the full payload exceeds 5 MB and can crash the browser's JSON parser.
    The frontend only needs the scalar metrics, not the raw weight tensors.
    """
    import copy
    slim = copy.deepcopy(server_output)
    for rd in slim.get("rounds", []):
        for cl in rd.get("clients", []):
            cl.pop("weights", None)
            cl.pop("feature_importances", None)
    return slim


def _build_result(
    strategy_name: str,
    df_all,
    server_output:  dict,
    cfg:            Config,
) -> dict:
    """
    Convert FLServer.run() output into the standard result dict consumed
    by the Flask API and the frontend.
    """
    last_round  = server_output["rounds"][-1]
    client_rows = [
        {
            "name":        r["name"],
            "n":           r["n"],
            "local_auroc": r.get("local_auroc", 0.0),
            "local_ap":    r.get("local_ap",    0.0),
        }
        for r in last_round["clients"] if not r.get("skipped")
    ]
    avg_imp = last_round.get("avg_importances") or [0.0] * len(_feature_cols())

    return {
        "strategy":            strategy_name,
        "n":                   last_round["total_n"],
        "n_pos":               int(df_all[TARGET_COL].sum()),
        "readmit_rate":        round(float(df_all[TARGET_COL].mean()) * 100, 1),
        # Local weighted AUROC (in-sample diagnostic — not a generalisation metric)
        "cv_auroc":            last_round["local_weighted_auroc"],
        "cv_std":              0.0,
        # Global held-out evaluation (primary metric for comparisons)
        "ho_auroc":            server_output["final_global_auroc"],
        "ho_ap":               server_output["final_global_ap"],
        "global_auroc":        server_output["final_global_auroc"],
        "global_ap":           server_output["final_global_ap"],
        "global_f1":           server_output.get("final_global_f1",        0.0),
        "global_precision":    server_output.get("final_global_precision",  0.0),
        "global_recall":       server_output.get("final_global_recall",     0.0),
        "feature_importances": dict(zip(_feature_cols(), avg_imp)),
        "convergence":         server_output["convergence"],
        "clients":             client_rows,
        "fl_detail":           _slim_server_output(server_output),
    }


def _run_federation(
    strategy_name:    str,
    clients:          List[FLClient],
    df_all,
    cfg:              Config,
    global_test:      GlobalTestSet,
    rlog:             RunLogger,
    run_id:           str,
    warm_start_weights: Optional[dict] = None,  # FIX 3: accept centralized weights
) -> dict:
    """Shared runner: instantiate FLServer, run rounds, checkpoint, return result."""
    rlog.step(f"    Clients: {[c.name for c in clients]}")

    server = FLServer(
        clients=clients,
        cfg=cfg,
        global_test=global_test,
        rlog=rlog,
        warm_start_weights=warm_start_weights,  # FIX 3
    )
    output = server.run()

    # Checkpoint the final global model
    tag = strategy_name.split()[0].lower().replace("(", "").replace(")", "")
    try:
        save_checkpoint(
            run_id=run_id,
            tag=tag,
            model=server.global_model,
            metrics={
                "global_auroc": output["final_global_auroc"],
                "global_ap":    output["final_global_ap"],
            },
            cfg_dict=cfg.to_dict(),
            checkpoint_dir=cfg.checkpoint_dir,
        )
    except Exception as exc:
        rlog.warn(f"Checkpoint save failed: {exc}")

    return _build_result(strategy_name, df_all, output, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 1 — CENTRALIZED
# ═══════════════════════════════════════════════════════════════════════════════

def run_centralized(
    df,
    preprocessor: Preprocessor,
    global_test:  GlobalTestSet,
    cfg:          Config,
    rlog:         RunLogger,
    run_id:       str,
) -> tuple:
    """
    Pooled model on all training data.
    Evaluated on the global held-out test set for fair comparison.

    Returns (result_dict, centralized_weights) so subsequent FL strategies
    can warm-start from a sensible parameter vector (FIX 3).
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = preprocessor.transform(df[_feature_cols()])
    y = df[TARGET_COL].values

    model = make_model(cfg.algorithm, cfg.random_state)
    cv    = StratifiedKFold(
        n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    # Fit on ALL training data
    model.fit(X, y)
    global_eval = global_test.evaluate(model)
    imp         = get_feature_importance(model)

    rlog.metric(
        "  Strategy 1 (Centralized)",
        f"global_AUROC={global_eval['global_auroc']:.4f}  "
        f"CV_AUROC={scores.mean():.4f}±{scores.std():.4f}",
    )

    try:
        save_checkpoint(
            run_id=run_id, tag="centralized",
            model=model,
            metrics=global_eval,
            cfg_dict=cfg.to_dict(),
            checkpoint_dir=cfg.checkpoint_dir,
        )
    except Exception as exc:
        rlog.warn(f"Checkpoint save failed: {exc}")

    result = {
        "strategy":            "Centralized (Strategy 1)",
        "n":                   int(len(y)),
        "n_pos":               int(y.sum()),
        "readmit_rate":        round(float(y.mean()) * 100, 1),
        "cv_auroc":            round(float(scores.mean()), 4),
        "cv_std":              round(float(scores.std()),  4),
        "ho_auroc":            global_eval["global_auroc"],
        "ho_ap":               global_eval["global_ap"],
        "global_auroc":        global_eval["global_auroc"],
        "global_ap":           global_eval["global_ap"],
        "global_f1":           global_eval.get("global_f1",        0.0),
        "global_precision":    global_eval.get("global_precision",  0.0),
        "global_recall":       global_eval.get("global_recall",     0.0),
        "feature_importances": dict(zip(_feature_cols(), imp)),
        "convergence":         [global_eval["global_auroc"]],
        "clients": [{"name": "ALL HOSPITALS", "n": len(y),
                     "local_auroc": global_eval["global_auroc"]}],
    }

    # FIX 3: return the fitted weights so FL strategies can warm-start
    centralized_weights = extract_weights(model)
    return result, centralized_weights


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 2 — FEDAVG BY REGION
# ═══════════════════════════════════════════════════════════════════════════════

def run_region_federation(
    df,
    preprocessor:       Preprocessor,
    global_test:        GlobalTestSet,
    cfg:                Config,
    rlog:               RunLogger,
    run_id:             str,
    fedprox_mu:         float = 0.0,
    warm_start_weights: Optional[dict] = None,
) -> dict:
    # FIX 2: fill missing region so no patients are silently dropped
    df = df.copy()
    df["region"] = df["region"].fillna("Unknown")

    regions = sorted(df["region"].unique())
    clients = [
        FLClient(
            name=f"Region: {reg}",
            X_raw=df[df["region"] == reg][_feature_cols()],
            y=df[df["region"] == reg][TARGET_COL],
            preprocessor=preprocessor,
            algorithm=cfg.algorithm,
            fedprox_mu=fedprox_mu,
            random_state=cfg.random_state,
        )
        for reg in regions
    ]
    label = (
        f"FedProx Region (Strategy 4, μ={fedprox_mu})"
        if fedprox_mu > 0
        else "FedAvg Region (Strategy 2)"
    )
    return _run_federation(
        label, clients, df, cfg, global_test, rlog, run_id,
        warm_start_weights=warm_start_weights,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 3 — FEDAVG BY HOSPITAL CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════

def run_cluster_federation(
    df,
    preprocessor:       Preprocessor,
    global_test:        GlobalTestSet,
    cfg:                Config,
    rlog:               RunLogger,
    run_id:             str,
    warm_start_weights: Optional[dict] = None,
) -> dict:
    df = assign_clusters(df, n_clusters=cfg.n_clusters,
                         random_state=cfg.random_state)
    clients = []
    for c in sorted(df["hospital_cluster"].dropna().unique()):
        sub  = df[df["hospital_cluster"] == c]
        rate = sub[TARGET_COL].mean() * 100
        clients.append(FLClient(
            name=f"Cluster {int(c)} (rate {rate:.1f}%)",
            X_raw=sub[_feature_cols()], y=sub[TARGET_COL],
            preprocessor=preprocessor,
            algorithm=cfg.algorithm,
            random_state=cfg.random_state,
        ))
    label = f"Hospital Cluster FedAvg (Strategy 3, {cfg.n_clusters} clusters)"
    return _run_federation(
        label, clients, df, cfg, global_test, rlog, run_id,
        warm_start_weights=warm_start_weights,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 4 — QUALITY-GATED PER-HOSPITAL FEDERATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_hospital_federation(
    df,
    preprocessor:       Preprocessor,
    global_test:        GlobalTestSet,
    cfg:                Config,
    rlog:               RunLogger,
    run_id:             str,
    warm_start_weights: Optional[dict] = None,
) -> dict:
    """
    Two-phase quality gate:
      Phase 1: screen hospitals by minimum size + one-round AUROC gate.
      Phase 2: FedAvg among qualifying hospitals only.
    """
    hstats = (
        df.groupby("hospitalid")
          .agg(
              n=(TARGET_COL, "count"),
              n_pos=(TARGET_COL, "sum"),
              n_neg=(TARGET_COL, lambda x: (x == 0).sum()),
          )
          .reset_index()
    )
    candidates = hstats[
        (hstats["n"]     >= cfg.min_hospital_stays) &
        (hstats["n_pos"] >= cfg.min_hospital_pos)   &
        (hstats["n_neg"] >= cfg.min_hospital_neg)
    ]

    _empty = {
        "strategy":            "Quality-Gated Hospital FL (Strategy 3)",
        "n": 0, "n_pos": 0, "readmit_rate": 0.0,
        "cv_auroc": 0.0, "cv_std": 0.0,
        "ho_auroc": 0.0, "ho_ap": 0.0,
        "global_auroc": 0.0, "global_ap": 0.0,
        "global_f1": 0.0, "global_precision": 0.0, "global_recall": 0.0,
        "feature_importances": dict(zip(_feature_cols(), [0.0] * len(_feature_cols()))),
        "convergence": [0.0], "clients": [],
        "screened_in": 0, "screened_out": 0,
        "skipped_too_small": len(hstats),
    }

    if candidates.empty:
        rlog.warn("Strategy 3: no hospitals met minimum size criteria")
        return {**_empty, "note": "No hospitals met the minimum size criteria."}

    # ── Phase 1: screening round ──────────────────────────────────────────────
    # Pre-group by hospitalid once — avoids 186 full-df boolean scans
    hosp_groups = {hid: grp for hid, grp in df.groupby("hospitalid")}

    screen_clients = [
        FLClient(
            name=f"Hospital {int(row.hospitalid)} (n={int(row.n)})",
            X_raw=hosp_groups[int(row.hospitalid)][_feature_cols()],
            y=hosp_groups[int(row.hospitalid)][TARGET_COL],
            preprocessor=preprocessor,
            algorithm=cfg.algorithm,
            random_state=cfg.random_state,
        )
        for _, row in candidates.iterrows()
    ]

    rlog.step(
        f"  Strategy 3: screening {len(screen_clients)} candidate hospitals "
        f"(gate AUROC≥{cfg.auroc_gate})"
    )

    screen_cfg = copy.copy(cfg)
    screen_cfg.fl_rounds = 1
    screen_server = FLServer(
        clients=screen_clients, cfg=screen_cfg,
        global_test=global_test, rlog=rlog,
        warm_start_weights=warm_start_weights,  # FIX 3
    )
    screen_output = screen_server.run()
    screen_round  = screen_output["rounds"][0]["clients"]

    screened_in:  list = []
    screened_out: list = []
    for r in screen_round:
        if r.get("skipped"):
            screened_out.append({**r, "reason": "too_small"})
        elif r.get("local_auroc", 0.0) >= cfg.auroc_gate:
            screened_in.append(r)
        else:
            screened_out.append(
                {**r, "reason": f'auroc_{r.get("local_auroc", 0.0):.3f}<{cfg.auroc_gate}'}
            )

    rlog.step(
        f"  Strategy 3: {len(screened_in)} hospitals passed gate, "
        f"{len(screened_out)} filtered"
    )

    if not screened_in:
        return {
            **_empty,
            "screened_in":       0,
            "screened_out":      len(screened_out),
            "skipped_too_small": len(hstats) - len(candidates),
            "note": f"No hospitals passed AUROC≥{cfg.auroc_gate} quality gate.",
        }

    # ── Phase 2: federation with qualified hospitals ───────────────────────────
    qualified_names = {r["name"] for r in screened_in}
    final_clients   = [c for c in screen_clients if c.name in qualified_names]

    result = _run_federation(
        "Quality-Gated Hospital FL (Strategy 3)",
        final_clients, df, cfg, global_test, rlog, run_id,
        warm_start_weights=warm_start_weights,  # FIX 3
    )
    return {
        **result,
        "screened_in":        len(screened_in),
        "screened_out":       len(screened_out),
        "skipped_too_small":  len(hstats) - len(candidates),
        "auroc_gate":         cfg.auroc_gate,
        "min_stays":          cfg.min_hospital_stays,
        "eligible_hospitals": len(final_clients),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 5 — FEDPROX BY REGION
# ═══════════════════════════════════════════════════════════════════════════════

def run_fedprox_federation(
    df,
    preprocessor:       Preprocessor,
    global_test:        GlobalTestSet,
    cfg:                Config,
    rlog:               RunLogger,
    run_id:             str,
    warm_start_weights: Optional[dict] = None,
) -> dict:
    """FedProx (Li et al., 2020) with μ=cfg.fedprox_mu on geographic regions."""
    return run_region_federation(
        df, preprocessor, global_test, cfg, rlog, run_id,
        fedprox_mu=cfg.fedprox_mu,
        warm_start_weights=warm_start_weights,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 6 — NON-IID SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_noniid_simulation(
    df,
    preprocessor:       Preprocessor,
    global_test:        GlobalTestSet,
    cfg:                Config,
    rlog:               RunLogger,
    run_id:             str,
    warm_start_weights: Optional[dict] = None,
) -> dict:
    """
    Simulate heterogeneous hospital distributions via Dirichlet(α) partitioning.
    α < 1 → extreme label skew; α ≥ 10 ≈ IID.
    """
    partitions = dirichlet_partition(
        df,
        n_clients=cfg.noniid_n_clients,
        alpha=cfg.dirichlet_alpha,
        random_state=cfg.random_state,
    )
    clients = []
    for i, part in enumerate(partitions):
        rate = part[TARGET_COL].mean() * 100 if len(part) > 0 else 0.0
        clients.append(FLClient(
            name=f"Synth-Hospital {i + 1} (rate {rate:.1f}%)",
            X_raw=part[_feature_cols()], y=part[TARGET_COL],
            preprocessor=preprocessor,
            algorithm=cfg.algorithm,
            random_state=cfg.random_state,
        ))
    label = (
        f"Non-IID Simulation (Strategy 5, "
        f"α={cfg.dirichlet_alpha}, {cfg.noniid_n_clients} clients)"
    )
    return _run_federation(
        label, clients, df, cfg, global_test, rlog, run_id,
        warm_start_weights=warm_start_weights,
    )
