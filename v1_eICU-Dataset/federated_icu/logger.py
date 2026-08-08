"""
federated_icu/logger.py
=======================
Structured logging:
  • Rotating file handler  → logs/<run_id>.log
  • Optional callback hook → streams messages to the Flask SSE endpoint
  • Module-level getLogger() used everywhere else
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import time
from typing import Callable, Optional


# ── Module logger (imported by other modules) ─────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"federated_icu.{name}")


# ── Run logger: file + callback ───────────────────────────────────────────────

class RunLogger:
    """
    Per-job logger that:
      1. Writes structured records to a rotating log file.
      2. Calls an optional progress_cb(kind, msg, pct) for SSE streaming.

    Usage:
        rlog = RunLogger(run_id="abc123", log_dir="logs",
                         progress_cb=lambda k, m, p: ...)
        rlog.step("Loading data…", pct=10)
        rlog.warn("Only 3 hospitals qualify")
        rlog.error("File not found")
        rlog.close()
    """

    def __init__(
        self,
        run_id:       str,
        log_dir:      str = "logs",
        log_level:    str = "INFO",
        progress_cb:  Optional[Callable] = None,
    ):
        self.run_id      = run_id
        self.progress_cb = progress_cb
        self._start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{run_id}.log")

        self._logger = logging.getLogger(f"run.{run_id}")
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self._logger.propagate = False   # don't double-log to root

        if not self._logger.handlers:
            fh = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            fmt = logging.Formatter(
                "%(asctime)s  %(levelname)-7s  %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

        self._logger.info("=" * 60)
        self._logger.info(f"Run {run_id} started")
        self._logger.info("=" * 60)

    # ── Public helpers ────────────────────────────────────────────────────────

    def step(self, msg: str, pct: Optional[float] = None) -> None:
        self._logger.info(msg)
        self._fire("step", msg, pct)

    def warn(self, msg: str) -> None:
        self._logger.warning(msg)
        self._fire("warn", msg, None)

    def error(self, msg: str) -> None:
        self._logger.error(msg)
        self._fire("error", msg, None)

    def trace(self, msg: str) -> None:
        self._logger.debug(msg)
        self._fire("trace", msg, None)

    def metric(self, label: str, value, suffix: str = "") -> None:
        msg = f"{label}: {value}{suffix}"
        self._logger.info(f"[METRIC] {msg}")
        self._fire("metric", msg, None)

    def elapsed(self) -> float:
        return round(time.time() - self._start_time, 2)

    def close(self) -> None:
        self._logger.info(f"Run {self.run_id} finished in {self.elapsed()}s")
        for h in self._logger.handlers[:]:
            h.close()
            self._logger.removeHandler(h)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fire(self, kind: str, msg: str, pct: Optional[float]) -> None:
        if self.progress_cb:
            try:
                self.progress_cb(kind, msg, pct)
            except Exception:
                pass   # never let a broken callback crash training


# ── Root logging setup (called once from app.py) ──────────────────────────────

def configure_root_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger("federated_icu")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(name)s  %(levelname)s  %(message)s"))
        root.addHandler(ch)
