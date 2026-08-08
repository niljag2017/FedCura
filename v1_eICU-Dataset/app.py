"""
app.py — Flask web server for the Federated Learning ICU Dashboard
==================================================================
Unchanged public API:
  GET  /              → index.html
  POST /run           → start background job, return {job_id}
  GET  /stream/<id>   → SSE progress stream
  GET  /result/<id>   → final JSON result
  POST /check_path    → validate data directory

All ML work is delegated to federated_icu.engine.run_all().
"""

import json
import os
import sys
import threading
import time
import traceback as tb
import uuid

from flask import Flask, Response, jsonify, render_template, request

# ── Make sure the package is importable whether run from this directory
#    or from the project root ───────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from federated_icu.config import DEFAULT_CONFIG, REQUIRED_TABLES
from federated_icu.data import validate_data_dir
from federated_icu.logger import configure_root_logging

configure_root_logging()

app  = Flask(__name__)
JOBS: dict = {}

# ── Default config shipped to the browser ────────────────────────────────────
_DEFAULT_CFG = {
    "data_dir":           "eicu-collaborative-research-database-demo-2.0",
    "algorithm":          "logistic_regression",
    "fl_rounds":          3,
    "cv_folds":           5,
    "min_hospital_stays": 10,
    "auroc_gate":         0.75,
    "fedprox_mu":         0.01,
    "dirichlet_alpha":    0.5,
    "noniid_n_clients":   5,
    "global_test_size":   0.20,
    "strategies":         [1, 2, 3, 4],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def push(job: dict, kind: str, msg: str, pct=None) -> None:
    """Append an SSE event to the job's event queue."""
    job["events"].append({"kind": kind, "msg": str(msg), "pct": pct})


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def _worker(job_id: str, config: dict) -> None:
    job = JOBS[job_id]

    def progress_cb(kind: str, msg: str, pct=None) -> None:
        push(job, kind, msg, pct)

    try:
        push(job, "step", f"Job {job_id} started", 2)

        # Validate data directory before importing heavy deps
        data_dir = os.path.normpath(
            config.get("data_dir", "eicu-collaborative-research-database-demo-2.0")
        )
        push(job, "step", f"Checking data folder: {data_dir}", 3)
        problems = validate_data_dir(data_dir)
        if problems:
            for p in problems:
                push(job, "warn", f"  ✗ {p}")
            push(job, "error",
                 "Data folder check failed. Fix the issues above and retry.", 0)
            job["status"] = "error"
            job["error"]  = "\n".join(problems)
            return

        present   = os.listdir(data_dir)
        gz_count  = sum(1 for f in present if f.endswith(".csv.gz"))
        csv_count = sum(1 for f in present if f.endswith(".csv") and not f.endswith(".csv.gz"))
        push(job, "step",
             f"  ✓ Folder OK — {gz_count} .csv.gz | {csv_count} .csv files", 4)

        # Lazy import: keeps Flask startup fast
        push(job, "step", "Importing FL engine…", 5)
        config["data_dir"] = data_dir
        from federated_icu.engine import run_all

        result = run_all(config, progress_cb=progress_cb)

        job["result"] = result
        job["status"] = "done"
        push(job, "step", f"All done in {result['elapsed_sec']}s", 100)

    except FileNotFoundError as exc:
        msg = str(exc)
        push(job, "error", f"File not found: {msg}", 0)
        push(job, "warn",
             "Tip: check the data folder path and make sure all required files exist.")
        job["status"] = "error"
        job["error"]  = msg
        tb.print_exc()

    except ValueError as exc:
        msg = str(exc)
        push(job, "error", f"Configuration error: {msg}", 0)
        job["status"] = "error"
        job["error"]  = msg

    except Exception as exc:
        full  = tb.format_exc()
        short = str(exc)
        push(job, "error", f"Unexpected error: {short}", 0)
        for line in full.strip().splitlines():
            push(job, "trace", line)
        job["status"] = "error"
        job["error"]  = short
        print(full)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def start_run():
    """Merge user config over defaults, start background thread, return job_id."""
    user_cfg = request.json or {}
    config   = {**_DEFAULT_CFG, **user_cfg}

    # Coerce numeric types (JSON may send strings from form inputs)
    config["fl_rounds"]           = int(config.get("fl_rounds",           3))
    config["cv_folds"]            = int(config.get("cv_folds",            5))
    config["min_hospital_stays"]  = int(config.get("min_hospital_stays",  10))
    config["noniid_n_clients"]    = int(config.get("noniid_n_clients",    5))
    config["auroc_gate"]          = float(config.get("auroc_gate",         0.75))
    config["fedprox_mu"]          = float(config.get("fedprox_mu",         0.01))
    config["dirichlet_alpha"]     = float(config.get("dirichlet_alpha",    0.5))
    config["global_test_size"]    = float(config.get("global_test_size",   0.20))
    config["strategies"]          = [int(s) for s in config.get("strategies", [1, 2, 3, 4])]

    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "status": "running",
        "events": [],
        "result": None,
        "config": config,
    }
    threading.Thread(target=_worker, args=(job_id, config), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    """Server-Sent Events stream for live progress reporting."""
    def generate():
        job = JOBS.get(job_id)
        if not job:
            yield 'data: {"kind":"error","msg":"job not found"}\n\n'
            return
        sent = 0
        while True:
            while sent < len(job["events"]):
                yield f"data: {json.dumps(job['events'][sent])}\n\n"
                sent += 1
            if job["status"] in ("done", "error"):
                yield f"data: {json.dumps({'kind':'done','status':job['status']})}\n\n"
                break
            time.sleep(0.2)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/result/<job_id>")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "status": job["status"],
        "result": job.get("result"),
        "error":  job.get("error"),
    })


@app.route("/check_path", methods=["POST"])
def check_path():
    """Lightweight path validator called from the UI before running."""
    data_dir = os.path.normpath((request.json or {}).get("data_dir", ""))
    problems = validate_data_dir(data_dir)
    files    = sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []
    return jsonify({"ok": len(problems) == 0, "problems": problems, "files": files})


@app.route("/checkpoints")
def list_checkpoints_route():
    """Return metadata for all saved checkpoints."""
    from federated_icu.evaluate import list_checkpoints
    ckpts = list_checkpoints(checkpoint_dir=_DEFAULT_CFG.get("checkpoint_dir", "checkpoints"))
    # Strip the unpicklable model object before JSONifying
    safe  = [{k: v for k, v in c.items() if k != "model"} for c in ckpts]
    return jsonify(safe)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n{'=' * 55}")
    print("  ICU Federated Learning Dashboard  v2")
    print(f"  Open your browser: http://localhost:{port}")
    print(f"{'=' * 55}\n")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
