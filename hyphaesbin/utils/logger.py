"""
hyphaesbin Logger
================
Production-grade logging for hyphaesbin pipeline.

Features:
    - Colored terminal output — errors and warnings instantly visible
    - Full DEBUG log always saved to file (even if screen shows INFO)
    - Named FLAGS — every warning has a code you can grep for
    - Step headers — clear visual markers for each pipeline stage
    - Timestamps on every file log line
    - Thread-safe (multiprocessing safe)
    - Structured helpers for the cross-cutting things every module in
      this pipeline reports (device/backend, thread counts, checkpoint
      reuse, contig counts, output paths) so main.py's own top-level
      summary can log them consistently instead of hand-rolling a
      slightly different message shape every time.

Usage:
    from hyphaesbin.utils.logger import setup_logger, get_logger
    from hyphaesbin.utils.logger import log_warning_flag, log_step_start

    # Once at startup in main.py:
    setup_logger(outdir="results/", log_level="INFO")

    # In every module:
    log = get_logger(__name__)
    log.info("Processing started")
    log_warning_flag(log, "LOW_N50", "N50 below threshold — TE disabled")

Grep commands for log file:
    grep "FLAG:"    hyphaesbin_*.log    # all named warnings/errors
    grep "WARNING"  hyphaesbin_*.log    # all warnings
    grep "ERROR"    hyphaesbin_*.log    # all errors
    grep "STEP"     hyphaesbin_*.log    # step headers only
    grep "DONE"     hyphaesbin_*.log    # completed steps
    grep "DEVICE"   hyphaesbin_*.log    # requested/actual device+backend lines
    grep "THREADS"  hyphaesbin_*.log    # requested/effective thread-count lines
    grep "CHECKPT"  hyphaesbin_*.log    # checkpoint reuse/recompute lines
    grep "CONTIGS"  hyphaesbin_*.log    # contig count / excluded-count lines
    grep "OUTPUT"   hyphaesbin_*.log    # output-path lines
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union


# ── Terminal color codes ───────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    BLUE   = "\033[94m"


class ColoredFormatter(logging.Formatter):
    """Color-coded terminal output. Each level has a distinct color."""
    FMTS = {
        logging.DEBUG:    C.DIM    + "  [DBG]  %(message)s"              + C.RESET,
        logging.INFO:     C.GREEN  + "  [INFO] %(message)s"              + C.RESET,
        logging.WARNING:  C.YELLOW + "  [WARN] %(message)s"              + C.RESET,
        logging.ERROR:    C.RED    + C.BOLD + "  [ERR]  %(message)s"     + C.RESET,
        logging.CRITICAL: C.RED    + C.BOLD + "  [CRIT] %(message)s"     + C.RESET,
    }
    def format(self, record):
        fmt = self.FMTS.get(record.levelno, self.FMTS[logging.INFO])
        return logging.Formatter(fmt).format(record)


# ── Global log file path (set by setup_logger) ────────────────────────────────
_LOG_FILE = None


def setup_logger(outdir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Initialize root logger. Call ONCE from main.py before anything else.

    Args:
        outdir:    output directory (logs/ subfolder created inside)
        log_level: screen verbosity — DEBUG/INFO/WARNING/ERROR

    Returns:
        root logger

    Safe to call more than once in the same process (handlers are
    cleared first each time) — this is what "prevent duplicate
    handlers when the pipeline imports modules repeatedly" means in
    practice: every module in this pipeline does `get_logger(name)`
    at IMPORT time, never `setup_logger()`, so re-importing a module
    (e.g. a second `from hyphaesbin.X import Y` after the first already
    succeeded — a no-op to Python's own import cache) can never add a
    second set of handlers. The only way to get duplicate handlers
    would be calling setup_logger() itself twice without this guard,
    which this function already prevented before this rewrite and
    still does.
    """
    global _LOG_FILE

    # Create logs directory
    log_dir = Path(outdir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"hyphaesbin_{ts}.log"
    _LOG_FILE = str(log_file)

    root = logging.getLogger("hyphaesbin")
    root.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-import / repeated setup_logger() calls
    if root.handlers:
        root.handlers.clear()

    # Screen — colored, user-chosen level
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    sh.setFormatter(ColoredFormatter())

    # File — always full DEBUG
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  [%(levelname)-8s]  %(name)-30s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    root.addHandler(sh)
    root.addHandler(fh)
    root.propagate = False

    root.info("=" * 66)
    root.info("   hyphaesbin  —  Fungi-Specific Metagenome Binning Tool")
    root.info("=" * 66)
    root.info(f"Log file  :  {log_file}")
    root.info(f"Screen    :  {log_level} | File: DEBUG (always full detail)")
    root.info(f"Grep tips :  grep 'FLAG:' {log_file}")
    root.info(f"           grep 'ERROR'  {log_file}")
    root.info("")

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger. Pass __name__ from any module."""
    return logging.getLogger(f"hyphaesbin.{name}")


def get_log_file() -> str:
    """Return path to current log file (for printing in summaries)."""
    return _LOG_FILE or "log file not initialized"


# ── Structured helpers (existing) ──────────────────────────────────────────────

def log_step_start(log, step_num, step_name: str, total: int = 13):
    """Print a bold step header. Easy to find in log file with grep STEP."""
    log.info("")
    log.info("━" * 66)
    log.info(f"  STEP {step_num} / {total}   {step_name.upper()}")
    log.info("━" * 66)


def log_step_done(log, step_num, step_name: str, elapsed: float):
    """Print step completion with timing."""
    if elapsed < 60:
        t = f"{elapsed:.1f}s"
    elif elapsed < 3600:
        t = f"{elapsed/60:.1f}min"
    else:
        t = f"{elapsed/3600:.2f}hrs"
    log.info(f"  ✅ STEP {step_num} DONE  :  {step_name}  [{t}]")
    log.info("")


def log_warning_flag(log, flag: str, message: str):
    """
    Named warning — searchable in log file.
    grep 'FLAG:' hyphaesbin_*.log   →  shows all named warnings
    """
    log.warning(f"⚠️  FLAG:{flag}  —  {message}")


def log_error_flag(log, flag: str, message: str):
    """Named error — searchable in log file."""
    log.error(f"❌  FLAG:{flag}  —  {message}")


def log_suggestion(log, message: str):
    """
    Print a suggestion to the user on how to fix a problem.
    Shown on screen AND saved to log file.
    """
    log.warning(f"💡  SUGGESTION: {message}")


# ── Structured helpers (new) ────────────────────────────────────────────────────
# These are additive — no existing call site needs to change. They exist for
# main.py's own top-level, per-phase reporting (and are available to any
# module that wants a consistent, greppable shape instead of an ad hoc
# f-string) but do not require changes to any already-finalized module: each
# module already logs this information in its own way internally (e.g.
# clustering.py logs "backend requested=... resolved=..." and "[CACHED]"
# directly) — these helpers just give main.py the same greppable shape for
# its own orchestration-level summary.

def log_device_backend(log, requested: str, actual: str, extra: str = ""):
    """Requested vs. ACTUAL device/backend — the two can differ (e.g.
    requested='gpu', no CUDA/cuML available, actual='cpu'). Always log
    both, never just one, so a silent fallback is never actually silent
    in the log."""
    suffix = f"  ({extra})" if extra else ""
    req = (requested or "auto").lower()
    act = (actual or "").lower()
    act_base = act.split(" (")[0].split(" ")[0]          # 'cpu (explicitly requested)' -> 'cpu'
    fell_back = ("not detected" in act) or ("fall back" in act) or ("fallback" in act)
    if fell_back or (req != "auto" and req != act_base):
        log.warning(f"DEVICE  requested={requested} but actual={actual} "
                    f"(fell back){suffix}")
    else:
        # 'auto' resolving to cpu/gpu, or an exact match, is not a fallback.
        log.info(f"DEVICE  requested={requested}  actual={actual}{suffix}")


def log_threads(log, requested: int, effective: int, context: str = ""):
    """Requested vs. effective (post-clamping-to-os.cpu_count(), or
    post-any-other-adjustment) thread/worker count."""
    ctx = f" [{context}]" if context else ""
    if requested == effective:
        log.info(f"THREADS{ctx}  requested={requested}  effective={effective}")
    else:
        log.warning(f"THREADS{ctx}  requested={requested} but effective={effective} "
                    f"(clamped)")


def log_checkpoint(log, step: str, reused: bool, reason: str = ""):
    """One consistent line for 'this step's checkpoint was reused' vs.
    'recomputed', with an optional reason (e.g. why it was invalidated)."""
    if reused:
        log.info(f"CHECKPT [{step}]  REUSED" + (f"  ({reason})" if reason else ""))
    else:
        log.info(f"CHECKPT [{step}]  RECOMPUTED" + (f"  ({reason})" if reason else ""))


def log_contig_counts(log, total: int, excluded: Optional[Dict[str, int]] = None,
                       kept: Optional[int] = None):
    """Total contig count plus a breakdown of excluded counts by reason
    (e.g. {'too_short': 120, 'invalid_coverage': 4}), and optionally the
    resulting kept count. Keeps the shape consistent across every stage
    that filters contigs instead of each writing a differently-formatted
    line."""
    parts = [f"total={total:,}"]
    if kept is not None:
        parts.append(f"kept={kept:,}")
    if excluded:
        for reason, n in excluded.items():
            parts.append(f"{reason}={n:,}")
    log.info("CONTIGS  " + "  ".join(parts))


def log_output_paths(log, paths: Dict[str, Union[str, "os.PathLike"]]):
    """One line per named output path, all prefixed 'OUTPUT' so
    `grep OUTPUT` on the log file gives every output this run produced,
    in one place, regardless of which phase wrote it."""
    for name, path in paths.items():
        log.info(f"OUTPUT  {name} = {path}")
