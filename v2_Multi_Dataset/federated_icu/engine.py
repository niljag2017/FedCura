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
from .data_loaders import (
    DATASET_REGISTRY, get_dataset_info,
    HEART_FEATURE_COLS, BANK_FEATURE_COLS,
)
from . import config as _global_cfg
# Snapshot the original eICU feature list once at import time.
# Any run that mutates _global_cfg.FEATURE_COLS must restore
# this value before the next eICU run.
_EICU_FEATURE_COLS = list(_global_cfg.FEATURE_COLS)
from .logger import RunLogger, configure_root_logging
from .strategies import (
    run_centralized,
    run_fedprox_federation,
    run_hospital_federation,
    run_noniid_simulation,
    run_region_federation,
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
    _ds_type_for_validate = getattr(cfg, "dataset_type", "eicu")
    problems = validate_data_dir(cfg.data_dir, _ds_type_for_validate)
    if problems:
        for p in problems:
            rlog.warn(p)
        raise FileNotFoundError(
            f"Data directory check failed: {cfg.data_dir}\n" + "\n".join(problems)
        )

    # ── Load & engineer features ──────────────────────────────────────────────
    # ── Dataset routing ──────────────────────────────────────────────────────
    ds_type = getattr(cfg, "dataset_type", "eicu")
    ds_info = get_dataset_info(ds_type)
    rlog.step(f"Loading dataset: {ds_info['label']}…", pct=5)

    if ds_type == "heart":
        from .data_loaders import load_heart_disease
        df = load_heart_disease(cfg.data_dir)
    elif ds_type == "bank":
        from .data_loaders import load_bank_churn
        df = load_bank_churn(cfg.data_dir)
    else:
        df = load_and_engineer(cfg.data_dir)

    # Always set FEATURE_COLS correctly — restores eICU cols if previous run
    # used heart/bank and mutated the module-level list.
    from . import config as _cfg_mod
    if ds_type == "eicu":
        _cfg_mod.FEATURE_COLS = list(_EICU_FEATURE_COLS)  # restore original
    else:
        feat_cols = ds_info.get("feature_cols")
        if feat_cols:
            _cfg_mod.FEATURE_COLS = feat_cols

    rlog.step(
        f"Loaded {len(df):,} records | "
        f"{int(df['readmitted'].sum()):,} positives "
        f"({df['readmitted'].mean()*100:.1f}%)  [{ds_info['domain']}]",
        pct=10,
    )
    # Snapshot feature cols after dataset routing (used in return dict)
    from . import config as _fcmod
    _feature_cols_snapshot = list(_fcmod.FEATURE_COLS)

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
    if cfg.augment_data and getattr(cfg, "dataset_type", "eicu") == "eicu":
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

    # ── Per-dataset strategy defaults ────────────────────────────────────────
    # heart: 4 hospital sites — S1 Centralized + S2 FedAvg Region only
    #        (S3 Quality-Gated needs many hospitals; S5 Non-IID is synthetic)
    # bank:  3 geographies    — S1 + S2 + S4 FedProx only
    #        (only 3 clients; S3/S5 not meaningful)
    # eICU:  all strategies as configured by user
    ds_type_for_strat = getattr(cfg, "dataset_type", "eicu")
    if ds_type_for_strat == "heart":
        # Heart Disease: 4 real hospital sites — S1 Centralized + S2 FedAvg Region
        # S3 Quality-Gated needs many hospitals (only 4 here)
        # S5 Non-IID is synthetic and not meaningful for real site partitions
        cfg.strategies = [1, 2]
        rlog.step("Heart Disease dataset: enforcing S1 (Centralized) + S2 (FedAvg Region only)", pct=15)
    elif ds_type_for_strat == "bank":
        # Bank Churn: 3 geography clients — S1 Centralized + S2 FedAvg Region
        # Keep it to 2 for clear comparison matching heart disease
        cfg.strategies = [1, 2]
        rlog.step("Bank Churn dataset: enforcing S1 (Centralized) + S2 (FedAvg Region only)", pct=15)

    # ── Run strategies ────────────────────────────────────────────────────────
    results   = {}
    pct_steps = {1: 16, 2: 34, 3: 52, 4: 70, 5: 86}
    regional_cmp = {"per_region": [], "regions": []}  # default; filled after strategies

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
        "feature_cols":       list(ds_info.get("feature_cols", _feature_cols_snapshot)),
        "train_n":            len(train_df),
        "test_n":             global_test.n,
        "run_id":             run_id,
        "dataset_type":       ds_type,
        "dataset_info":       {k: v for k, v in ds_info.items() if k not in ("loader","feature_cols","feature_labels")},
    }
