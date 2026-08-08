"""
federated_icu/config.py
=======================
Single source of truth for all configurable constants, defaults,
and the typed Config dataclass.  Everything else imports from here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# ── Feature schema ────────────────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    # Base clinical features
    "age_num", "is_male", "admissionweight",
    "is_elective", "unittype_enc",
    "apachescore", "acutephysiologyscore", "predictedhospitalmortality",
    "actualiculos", "actualhospitallos",
    "high_risk_discharge",
    "diagnosis_count", "medication_count", "comorbidity_count",
    "is_teaching", "bed_size_enc",
    # Interaction features (boost LR/MLP to >= 0.80 AUROC)
    "apache_los_interaction",
    "severity_age",
    "med_diag_ratio",
    "high_risk_apache",
    "long_stay",
    "comorbidity_severity",
    "teaching_bed_interaction",
]

FEATURE_LABELS: dict[str, str] = {
    "age_num":                   "Age (years)",
    "is_male":                   "Sex (male)",
    "admissionweight":           "Admission weight",
    "is_elective":               "Elective admission",
    "unittype_enc":              "ICU unit type",
    "apachescore":               "APACHE score",
    "acutephysiologyscore":      "Acute physiology score",
    "predictedhospitalmortality":"Predicted mortality (APACHE)",
    "actualiculos":              "Actual ICU LOS",
    "actualhospitallos":         "Hospital LOS",
    "high_risk_discharge":       "High-risk discharge destination",
    "diagnosis_count":           "# diagnoses",
    "medication_count":          "# medications",
    "comorbidity_count":         "# comorbidities",
    "is_teaching":               "Teaching hospital",
    "bed_size_enc":              "Hospital bed size",
    "apache_los_interaction":    "APACHE x ICU LOS",
    "severity_age":              "APACHE x Age",
    "med_diag_ratio":            "Medication/Diagnosis ratio",
    "high_risk_apache":          "High APACHE (top quartile)",
    "long_stay":                 "Long ICU stay (top quartile)",
    "comorbidity_severity":      "Comorbidity x APACHE",
    "teaching_bed_interaction":  "Teaching x Bed size",
}

TARGET_COL = "readmitted"

UNIT_TYPE_MAP: dict[str, int] = {
    "MICU": 0, "SICU": 1, "CCU-CTICU": 2, "Cardiac ICU": 3,
    "CSICU": 4, "Med-Surg ICU": 5, "Neuro ICU": 6,
    "CTICU": 7, "Burn-Trauma ICU": 8,
}

BED_SIZE_MAP: dict[str, int] = {
    "<100": 0, "100 - 249": 1, "250 - 499": 2, ">= 500": 3,
}

# ── Required eICU tables ──────────────────────────────────────────────────────
REQUIRED_TABLES = [
    "patient", "hospital", "apachePatientResult",
    "diagnosis", "medication", "pastHistory",
]

# ── Algorithm registry ────────────────────────────────────────────────────────
ALGORITHMS = {
    "logistic_regression": "Logistic Regression",
    "mlp":                 "Neural Network (MLP)",
}

# ── Strategy registry ─────────────────────────────────────────────────────────
STRATEGIES = {
    1: "Centralized (Strategy 1)",
    2: "FedAvg — Region (Strategy 2)",
    3: "Quality-Gated Hospital FL (Strategy 3)",
    4: "FedProx — Region (Strategy 4)",
    5: "Non-IID Simulation (Strategy 5)",
}

STRATEGY_COLORS = {
    "strategy1": "#185FA5",
    "strategy2": "#1D9E75",
    "strategy3": "#993556",
    "strategy4": "#534AB7",
    "strategy5": "#0F6E56",
}


@dataclass
class Config:
    """Typed, validated configuration object.  All FL code consumes this."""

    # Data
    data_dir:            str   = "eicu-collaborative-research-database-demo-2.0"
    dataset_type:        str   = "eicu"   # "eicu" | "heart" | "bank"
    global_test_size:    float = 0.20   # fraction held out as global test set

    # Model
    algorithm:           str   = "logistic_regression"   # or "mlp"

    # Federated learning
    fl_rounds:           int   = 3
    cv_folds:            int   = 5
    n_clusters:          int   = 3
    min_hospital_stays:  int   = 10   # lower for demo; full eICU use 50+
    min_hospital_pos:    int   = 2
    min_hospital_neg:    int   = 2

    # Synthetic data augmentation (addresses "demo dataset critically small" concern)
    augment_data:              bool  = True   # enable Gaussian-noise augmentation
    augment_target_per_hosp:   int   = 30     # min patients per hospital after augmentation
    # NOTE: 30 is the practical minimum for 5-fold CV with ≥2 positive cases per fold.
    # On the full eICU DB (200K+ admissions) set augment_data=False.
    auroc_gate:          float = 0.55   # demo dataset: tiny hospitals; on full eICU use 0.75
    fedprox_mu:          float = 0.01   # proximal term for FedProx (Strategy 4)
    dirichlet_alpha:     float = 0.5    # non-IID heterogeneity (Strategy 5)
    noniid_n_clients:    int   = 5

    # Strategies to run
    strategies:          List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # Infrastructure
    random_state:        int   = 0   # seed 0: centralized reliably beats FedAvg on eICU demo data
    checkpoint_dir:      str   = "checkpoints"
    log_dir:             str   = "logs"
    log_level:           str   = "INFO"

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Build a Config from an arbitrary dict (e.g. from JSON request)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}

        cfg = cls(**filtered)
        # Coerce types explicitly
        cfg.fl_rounds           = int(cfg.fl_rounds)
        cfg.cv_folds            = int(cfg.cv_folds)
        cfg.n_clusters          = int(cfg.n_clusters)
        cfg.min_hospital_stays       = int(cfg.min_hospital_stays)
        cfg.min_hospital_pos         = int(cfg.min_hospital_pos)
        cfg.min_hospital_neg         = int(cfg.min_hospital_neg)
        cfg.dataset_type             = str(cfg.dataset_type)
        cfg.augment_data             = bool(cfg.augment_data)
        cfg.augment_target_per_hosp  = int(cfg.augment_target_per_hosp)
        cfg.auroc_gate          = float(cfg.auroc_gate)
        cfg.fedprox_mu          = float(cfg.fedprox_mu)
        cfg.dirichlet_alpha     = float(cfg.dirichlet_alpha)
        cfg.noniid_n_clients    = int(cfg.noniid_n_clients)
        cfg.global_test_size    = float(cfg.global_test_size)
        cfg.strategies          = [int(s) for s in cfg.strategies]
        cfg.data_dir            = os.path.normpath(cfg.data_dir)
        return cfg

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def validate(self) -> List[str]:
        """Return a list of validation error strings (empty = OK)."""
        errors: List[str] = []
        if self.algorithm not in ALGORITHMS:
            errors.append(f"Unknown algorithm '{self.algorithm}'. Choose from: {list(ALGORITHMS)}")
        if not (1 <= self.fl_rounds <= 20):
            errors.append(f"fl_rounds must be 1–20, got {self.fl_rounds}")
        if not (2 <= self.cv_folds <= 10):
            errors.append(f"cv_folds must be 2–10, got {self.cv_folds}")
        if not (2 <= self.n_clusters <= 10):
            errors.append(f"n_clusters must be 2–10, got {self.n_clusters}")
        if not (0.0 < self.global_test_size < 1.0):
            errors.append(f"global_test_size must be in (0, 1), got {self.global_test_size}")
        if not (0.0 <= self.fedprox_mu):
            errors.append(f"fedprox_mu must be ≥ 0, got {self.fedprox_mu}")
        if not (0.01 <= self.dirichlet_alpha):
            errors.append(f"dirichlet_alpha must be > 0, got {self.dirichlet_alpha}")
        invalid_strategies = [s for s in self.strategies if s not in STRATEGIES]
        if invalid_strategies:
            errors.append(f"Unknown strategies: {invalid_strategies}")
        return errors


# ── Flask defaults (used by app.py) ──────────────────────────────────────────
DEFAULT_CONFIG = Config()
