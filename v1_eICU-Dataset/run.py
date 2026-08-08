#!/usr/bin/env python3
"""
run.py — One-click launcher for the ICU Federated Learning Dashboard v2
========================================================================
Usage:
    python run.py            # starts server, opens browser
    python run.py --port 8080
    python run.py --no-browser

Requirements:
    pip install flask pandas numpy scikit-learn
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

PORT    = 5050
HERE    = pathlib.Path(__file__).parent.resolve()
EICU    = "eicu-collaborative-research-database-demo-2.0"
REQUIRED = [
    "patient.csv.gz", "hospital.csv.gz",
    "apachePatientResult.csv.gz", "diagnosis.csv.gz",
    "medication.csv.gz", "pastHistory.csv.gz",
]
DEPS = {"flask": "Flask", "pandas": "pandas",
        "numpy": "numpy", "sklearn": "scikit-learn"}


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
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        else:
            sys.exit(1)


def check_data() -> str:
    folder = HERE / EICU
    if not folder.exists():
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  eICU data folder not found!                                 ║
╠══════════════════════════════════════════════════════════════╣
║  Expected: {EICU}
║  Location: {HERE}
║
║  Place the unzipped eICU demo folder here, then try again.   ║
║  Download: https://physionet.org/content/eicu-crd-demo/2.0/ ║
╚══════════════════════════════════════════════════════════════╝
""")
        input("Press Enter to exit…")
        sys.exit(1)

    missing = [f for f in REQUIRED if not (folder / f).exists()]
    if missing:
        print(f"\nMissing files in {EICU}:\n  " + "\n  ".join(missing))
        input("Press Enter to exit…")
        sys.exit(1)

    return str(folder)


def main() -> None:
    parser = argparse.ArgumentParser(description="ICU Federated Learning Dashboard")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="Skip eICU data folder check (useful if path is set in UI)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ICU Readmission — Federated Learning Dashboard v2")
    print("=" * 60)

    print("\nChecking dependencies…")
    check_deps()

    if not args.skip_data_check:
        print("Checking eICU data…")
        data_path = check_data()
        print(f"  ✓ Found: {data_path}")

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

    # Load and run the Flask app from this directory
    spec = importlib.util.spec_from_file_location("app", HERE / "app.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["app"] = mod
    spec.loader.exec_module(mod)
    mod.app.run(debug=False, host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
