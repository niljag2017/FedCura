#!/usr/bin/env python3
"""
run.py — One-click launcher for the ICU Federated Learning Dashboard v3
========================================================================
Usage:
    python run.py            # starts server, opens browser
    python run.py --port 8080
    python run.py --no-browser
    python run.py --skip-data-check   # skip dataset folder checks

Requirements:
    pip install flask pandas numpy scikit-learn

Datasets folder layout (datasets/ subfolder):
    datasets/
    ├── eicu-collaborative-research-database-demo-2.0/   ← place here
    ├── heart_disease/                                    ← included
    └── bank_churn/                                       ← included
"""

import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys
import threading
import time
import webbrowser

PORT = 5050
HERE = pathlib.Path(__file__).parent.resolve()
DATASETS_DIR = HERE / "datasets"

# ── Dataset definitions ───────────────────────────────────────────────────────
DATASETS = {
    "eicu": {
        "folder":   DATASETS_DIR / "eicu-collaborative-research-database-demo-2.0",
        "required": [
            "patient.csv.gz", "hospital.csv.gz",
            "apachePatientResult.csv.gz", "diagnosis.csv.gz",
            "medication.csv.gz", "pastHistory.csv.gz",
        ],
        "label":    "eICU Collaborative Research Database Demo",
        "download": "https://physionet.org/content/eicu-crd-demo/2.0/",
    },
    "heart": {
        "folder":   DATASETS_DIR / "heart_disease",
        "required": [
            "processed.cleveland.data",
            "processed.hungarian.data",
            "processed.switzerland.data",
            "processed.va.data",
        ],
        "label":    "UCI Heart Disease",
        "download": "https://archive.ics.uci.edu/dataset/45/heart+disease",
    },
    "bank": {
        "folder":   DATASETS_DIR / "bank_churn",
        "required": ["Churn_Modelling.csv"],
        "label":    "Bank Customer Churn",
        "download": "https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling",
    },
}

DEPS = {
    "flask":   "Flask",
    "pandas":  "pandas",
    "numpy":   "numpy",
    "sklearn": "scikit-learn",
}


def check_deps() -> None:
    missing = []
    for imp, pkg in DEPS.items():
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with:  pip install {' '.join(missing)}")
        ans = input("Install now? [y/N] ").strip().lower()
        if ans == "y":
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + missing
            )
        else:
            sys.exit(1)


def check_datasets() -> None:
    """
    Check all three dataset folders.
    - heart_disease and bank_churn are bundled — report error if missing.
    - eICU is user-provided — print a friendly download message if missing
      but do NOT exit; the dashboard still works for heart + bank.
    """
    print(f"\n  Dataset folder: {DATASETS_DIR}")

    all_ok = True
    for key, info in DATASETS.items():
        folder   = info["folder"]
        required = info["required"]
        label    = info["label"]

        if not folder.exists():
            if key == "eicu":
                print(f"\n  ⚠  eICU folder not found — heart disease + bank churn still available.")
                print(f"     To enable eICU: place the unzipped demo folder at")
                print(f"       {folder}")
                print(f"     Download: {info['download']}")
            else:
                print(f"\n  ✗  {label}: folder missing — {folder}")
                all_ok = False
            continue

        missing_files = [f for f in required if not (folder / f).exists()]
        if missing_files:
            print(f"\n  ✗  {label}: missing files in {folder.name}/")
            for f in missing_files:
                print(f"       • {f}")
            if key != "eicu":
                all_ok = False
        else:
            file_count = len(list(folder.iterdir()))
            print(f"  ✓  {label}: {folder.name}/ ({file_count} files)")

    if not all_ok:
        print("\n  Some bundled datasets are incomplete. Re-download or re-extract the zip.")
        input("Press Enter to exit…")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedICU — Federated Learning Dashboard v3"
    )
    parser.add_argument("--port",            type=int, default=PORT)
    parser.add_argument("--no-browser",      action="store_true")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="Skip dataset folder checks (use if paths are set manually in UI)")
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  FedICU — Federated Learning Dashboard  v3")
    print("  Supports: eICU · UCI Heart Disease · Bank Churn")
    print("=" * 62)

    print("\nChecking dependencies…")
    check_deps()
    print("  ✓ All dependencies satisfied")

    if not args.skip_data_check:
        print("\nChecking dataset folders…")
        check_datasets()

    os.makedirs(HERE / "logs",        exist_ok=True)
    os.makedirs(HERE / "checkpoints", exist_ok=True)
    os.chdir(HERE)

    print(f"\nStarting server on http://localhost:{args.port}")
    if not args.no_browser:
        print("Opening browser in 2 s…")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        def _open():
            time.sleep(2.5)
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=_open, daemon=True).start()

    spec = importlib.util.spec_from_file_location("app", HERE / "app.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["app"] = mod
    spec.loader.exec_module(mod)
    mod.app.run(debug=False, host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
