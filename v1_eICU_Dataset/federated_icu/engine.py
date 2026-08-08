"""
federated_icu/engine.py
=======================
Top-level orchestrator.  app.py calls run_all(); nothing else in the
codebase is imported directly by Flask.

run_all(config_dict, progress_cb) → result dict

FIX v3: Strategy 1 (Centralized) now returns (result, weights).
The weights are passed to all subsequent FL strategies as warm_start_weights
so FedAvg starts from a sensible parameter vector instead of a random seed.
If Strategy 1 is not selected, FL strategies fall back to the dummy seed
(original behaviour — no regression).
"""
from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

from .config import Config
from .data import (
    build_global_test_split, load_and_engineer,
    augment_hospital_data, validate_data_dir,
)
from .logger import RunLogger, configure_root_logging
from .strategies import (
    run_centralized,
    run_fedprox_federation,
    run_hospital_federation,
    run_noniid_simulation,
    run_region_federation,
    run_regional_comparison,
)


def run_all(
    config_dict:  dict,
    progress_cb:  Optional[Callable] = None,
) -> dict:
    """
    Main entry point for a federated learning run.

    Parameters
    ----------
    config_dict : arbitrary dict from the Flask JSON request.
                  Unknown keys are silently ignored.
    progress_cb : optional callback(kind, msg, pct) for SSE streaming.

    Returns
    -------
    dict with keys: results, config, elapsed_sec, feature_cols,
                    train_n, test_n, run_id
    """
    t0     = time.time()
    run_id = str(uuid.uuid4())[:8]

    # ── Parse & validate config ───────────────────────────────────────────────
    cfg    = Config.from_dict(config_dict)
    errors = cfg.validate()
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(errors))

    configure_root_logging(log_dir=cfg.log_dir, level=cfg.log_level)

    rlog = RunLogger(
        run_id=run_id,
        log_dir=cfg.log_dir,
        log_level=cfg.log_level,
        progress_cb=progress_cb,
    )

    # ── Validate data directory ───────────────────────────────────────────────
    problems = validate_data_dir(cfg.data_dir)
    if problems:
        for p in problems:
            rlog.warn(p)
        raise FileNotFoundError(
            f"Data directory check failed: {cfg.data_dir}\n" + "\n".join(problems)
        )

    # ── Load & engineer features ──────────────────────────────────────────────
    rlog.step("Loading eICU tables…", pct=5)
    df = load_and_engineer(cfg.data_dir)
    rlog.step(
        f"Loaded {len(df):,} patients | "
        f"{int(df['readmitted'].sum()):,} readmissions "
        f"({df['readmitted'].mean()*100:.1f}%)",
        pct=10,
    )

    # ── Build global held-out test set from REAL data only (before augmentation) ──
    # IMPORTANT: test split must be done on real records first to prevent data
    # leakage. If we augmented first, the 20% test set would contain synthetic
    # records derived from the same real patients as the training set, causing
    # the model to be evaluated on near-copies of its own training data (overfitting
    # manifests as AUROC ~0.999). Splitting real data first ensures the test set
    # is never contaminated by synthetic derivatives of training patients.
    rlog.step(
        f"Splitting global held-out test set from real data only "
        f"({cfg.global_test_size:.0%} of {len(df):,} real patients)…",
        pct=11,
    )
    train_df_real, global_test = build_global_test_split(df, cfg)
    rlog.step(
        f"Real data split — Train: {len(train_df_real):,} | "
        f"Test: {global_test.n:,} real patients (no synthetic records in test set)",
        pct=12,
    )

    # ── Synthetic augmentation on TRAINING data only ──────────────────────────
    # Augment only train_df_real so the test set stays clean (real patients only).
    # Set augment_data=False when using the full eICU DB (200K+ admissions).
    if cfg.augment_data:
        rlog.step(
            f"Augmenting training data to ≥{cfg.augment_target_per_hosp} "
            f"patients/hospital via Gaussian-noise perturbation (train only)…",
            pct=13,
        )
        train_df = augment_hospital_data(
            train_df_real,
            target_per_hospital=cfg.augment_target_per_hosp,
            random_state=cfg.random_state,
        )
        rlog.step(
            f"After augmentation — Train: {len(train_df):,} patients "
            f"({int(train_df['is_synthetic'].sum()):,} synthetic, "
            f"{int((~train_df['is_synthetic']).sum()):,} real) | "
            f"Test: {global_test.n:,} real patients only",
            pct=14,
        )
    else:
        train_df = train_df_real

    preprocessor = global_test.preprocessor   # fitted on real train split only

    # ── Run strategies ────────────────────────────────────────────────────────
    results   = {}
    pct_steps = {1: 16, 2: 34, 3: 52, 4: 70, 5: 86}

    # FIX 3: centralized weights used to warm-start all FL strategies
    centralized_weights = None

    if 1 in cfg.strategies:
        rlog.step("Running Strategy 1 — Centralized (oracle baseline)…",
                  pct=pct_steps[1])
        # run_centralized now returns (result_dict, weights)
        results["strategy1"], centralized_weights = run_centralized(
            train_df, preprocessor, global_test, cfg, rlog, run_id
        )
        rlog.step(
            f"  Centralized AUROC={results['strategy1']['global_auroc']:.4f}  "
            f"(warm-start weights captured for FL strategies)",
            pct=pct_steps[1],
        )

    if 2 in cfg.strategies:
        rlog.step("Running Strategy 2 — FedAvg by Region…", pct=pct_steps[2])
        results["strategy2"] = run_region_federation(
            train_df, preprocessor, global_test, cfg, rlog, run_id,
            warm_start_weights=centralized_weights,
        )

    if 3 in cfg.strategies:
        rlog.step(
            f"Running Strategy 3 — Quality-Gated Hospital FL "
            f"(gate={cfg.auroc_gate})…",
            pct=pct_steps[3],
        )
        results["strategy3"] = run_hospital_federation(
            train_df, preprocessor, global_test, cfg, rlog, run_id,
            warm_start_weights=centralized_weights,
        )

    if 4 in cfg.strategies:
        rlog.step(
            f"Running Strategy 4 — FedProx (μ={cfg.fedprox_mu})…",
            pct=pct_steps[4],
        )
        results["strategy4"] = run_fedprox_federation(
            train_df, preprocessor, global_test, cfg, rlog, run_id,
            warm_start_weights=centralized_weights,
        )

    if 5 in cfg.strategies:
        rlog.step(
            f"Running Strategy 5 — Non-IID Simulation "
            f"(α={cfg.dirichlet_alpha})…",
            pct=pct_steps[5],
        )
        results["strategy5"] = run_noniid_simulation(
            train_df, preprocessor, global_test, cfg, rlog, run_id,
            warm_start_weights=centralized_weights,
        )

    # ── Regional 6-scenario comparison (VIVA examiner request) ─────────────
    fl_global_weights = None
    if "strategy2" in results:
        fl_global_weights = results["strategy2"].get("_fl_weights")
    elif "strategy4" in results:
        fl_global_weights = results["strategy4"].get("_fl_weights")

    rlog.step("Running Regional 6-Scenario Comparison…", pct=96)
    try:
        regional_cmp = run_regional_comparison(
            train_df,
            preprocessor,
            global_test,
            cfg,
            rlog,
            centralized_weights=centralized_weights,
            fl_global_weights=fl_global_weights,
        )
    except Exception as _exc:
        import traceback
        rlog.warn(f"Regional comparison failed: {_exc}")
        traceback.print_exc()
        regional_cmp = {
            "unified_row": None, "regional_rows": [],
            "exp6_note": str(_exc), "exp6_final": "Error",
            "validation_log": [], "reproducibility": {}, "warnings": [],
        }

        # ── Wrap up ───────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    rlog.step(f"All strategies complete in {elapsed}s", pct=100)
    rlog.close()

    from .config import FEATURE_COLS
    return {
        "results":            results,
        "regional_comparison": regional_cmp,
        "config":             cfg.to_dict(),
        "elapsed_sec":        elapsed,
        "feature_cols":       FEATURE_COLS,
        "train_n":            len(train_df),
        "test_n":             global_test.n,
        "run_id":             run_id,
    }
