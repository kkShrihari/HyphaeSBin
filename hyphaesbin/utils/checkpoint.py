#"""
#hyphaesbin Checkpoint System
#============================
#Crash-safe pipeline resumption — like Nextflow's -resume.
#
#Every step saves a .done file. On rerun, completed steps are
#automatically skipped. Just re-run the same command to resume.
#
#Checkpoint files: outdir/checkpoints/STEPNAME.done
#Each file is JSON with: step name, timestamp, module/version, the
#output files that step is expected to have produced, and a free-form
#metadata dict the caller controls.
#
#WHAT THIS CLASS DELIBERATELY DOES NOT DO: compute or compare
#fingerprints. Every pipeline module (preprocessing.py, coverage.py,
#tnf_gene.py, te_composition.py, encoder.py, clustering.py) already
#builds its own fingerprint string from whatever inputs/config/version
#actually matter to IT (see each module's own `_fingerprint`/
#`_file_fingerprint` helpers) and stores it as an opaque value inside
#the `metadata` dict it hands to `mark_done` (conventionally under the
#key "_fp"). This class never inspects or interprets that value — it
#just stores and returns it verbatim. Centralizing fingerprint LOGIC
#here would mean every module either shares one fingerprint shape or
#routes its fingerprint through this file's opinion of what matters;
#keeping it out lets each module's checkpoint be exactly as strict (or
#as loose) as that module's own inputs require, decided in exactly one
#place (that module) rather than two.
#
#v2 changes (see module-level CHANGELOG below for why):
#    - Atomic writes: a checkpoint file is written to a temp file in
#      the same directory and atomically replaced into place, so a
#      crash or kill mid-write can never leave a half-written .done
#      file that then reads as "corrupt" (or worse, reads as valid
#      JSON with truncated/wrong content) on the next run.
#    - Output-file verification: mark_done() accepts an optional
#      `output_files` list; is_done() then checks that every path in
#      it still exists on disk before accepting the checkpoint as
#      valid. A step whose recorded outputs were deleted (or moved) is
#      treated as NOT done — exactly the same "stale checkpoint is a
#      cache miss, not a silently wrong answer" principle every
#      pipeline module already applies to ITS OWN fingerprint checks,
#      now available as one shared primitive instead of every caller
#      re-implementing the existence-check by hand.
#    - Module/version metadata: mark_done() accepts optional `module`
#      and `version` strings, stored alongside the step's own metadata
#      dict. This is informational only (never affects is_done()'s
#      answer) — it lets a caller (main.py, or a human reading
#      checkpoints/*.done) see which module/version produced a given
#      checkpoint without opening the file's `metadata` blob and
#      guessing which key might hold that information.
#
#Both new mark_done() parameters are optional and keyword-only-by-
#convention (positional-compatible with every existing call site in
#this codebase, all of which call `mark_done(step, metadata_dict)` or
#`mark_done(step, metadata_dict, output_files=..., ...)`) — no existing
#caller needs to change.
#
#Usage:
#    ckpt = Checkpoint("results/")
#
#    if not ckpt.is_done("step0_n50"):
#        result = do_work()
#        ckpt.mark_done("step0_n50", metadata=result,
#                        output_files=[result["output_path"]],
#                        module="preprocessing", version="v13steps")
#    else:
#        result = ckpt.load_metadata("step0_n50")  # load previous result
#"""
#
#import json
#import os
#import tempfile
#from pathlib import Path
#from datetime import datetime
#from typing import Optional, Dict, Any, List
#
#from hyphaesbin.utils.logger import get_logger
#
#log = get_logger("checkpoint")
#
#
#class Checkpoint:
#
#    def __init__(self, outdir: str):
#        self.ckpt_dir = Path(outdir) / "checkpoints"
#        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
#
#    def _path(self, step: str) -> Path:
#        return self.ckpt_dir / f"{step}.done"
#
#    def is_done(self, step: str) -> bool:
#        f = self._path(step)
#        if not f.exists():
#            return False
#        try:
#            info = json.loads(f.read_text())
#        except (json.JSONDecodeError, IOError, UnicodeDecodeError):
#            log.warning(f"Corrupt checkpoint {f} — treating as not done (will re-run step)")
#            f.unlink(missing_ok=True)
#            return False
#
#        # A checkpoint whose recorded outputs no longer exist on disk is a
#        # cache miss, not a silently-reused stale answer — mirrors the
#        # pattern every module's own fingerprint check already applies to
#        # ITS output paths, generalized here so callers don't each
#        # reimplement it (see e.g. clustering.py's _checkpoint_ok, which
#        # additionally layers its OWN fingerprint comparison on top of this
#        # existence check — this class only owns the existence half).
#        output_files = info.get("output_files") or []
#        missing = [p for p in output_files if p and not Path(p).exists()]
#        if missing:
#            log.warning(
#                f"Checkpoint '{step}' exists but {len(missing)} referenced output(s) no "
#                f"longer exist on disk ({missing[:3]}{', ...' if len(missing) > 3 else ''}) "
#                f"— treating as not done."
#            )
#            return False
#
#        log.info(f"  ⏭️  SKIP  [{step}]  (completed {info.get('timestamp','')})")
#        return True
#
#    def mark_done(self, step: str, metadata: Optional[dict] = None,
#                   output_files: Optional[List[str]] = None,
#                   module: Optional[str] = None, version: Optional[str] = None):
#        payload = {
#            "step": step,
#            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#            "module": module,
#            "version": version,
#            "output_files": [str(p) for p in output_files] if output_files else [],
#            "metadata": metadata or {},
#        }
#        f = self._path(step)
#        # Atomic write: write to a temp file in the SAME directory (so the
#        # final os.replace is a same-filesystem rename, not a cross-device
#        # copy) then swap it into place. A crash/kill between these two
#        # lines leaves either the old .done file (if any) or nothing —
#        # never a truncated/partial one.
#        fd, tmp_path = tempfile.mkstemp(dir=str(self.ckpt_dir), prefix=f".{step}.", suffix=".tmp")
#        try:
#            with os.fdopen(fd, "w") as tmp_f:
#                tmp_f.write(json.dumps(payload, indent=2, default=str))
#                tmp_f.flush()
#                os.fsync(tmp_f.fileno())
#            os.replace(tmp_path, f)
#        except Exception:
#            Path(tmp_path).unlink(missing_ok=True)
#            raise
#        log.debug(f"Checkpoint saved: {f}")
#
#    def load_metadata(self, step: str) -> dict:
#        """Load the caller-supplied metadata dict from a completed
#        checkpoint (NOT the full payload — use is_done()/the checkpoint
#        file directly for output_files/module/version)."""
#        f = self._path(step)
#        if f.exists():
#            try:
#                return json.loads(f.read_text()).get("metadata", {})
#            except Exception:
#                pass
#        return {}
#
#    def load_full(self, step: str) -> Optional[Dict[str, Any]]:
#        """Load the full checkpoint payload (step, timestamp, module,
#        version, output_files, metadata) — use this when you need the
#        module/version provenance, not just the metadata dict."""
#        f = self._path(step)
#        if not f.exists():
#            return None
#        try:
#            return json.loads(f.read_text())
#        except Exception:
#            return None
#
#    # ALIAS: get_status = load_metadata (for backward compatibility with main.py)
#    def get_status(self, step: str) -> Optional[Dict[str, Any]]:
#        """
#        Get status/metadata of a completed step.
#        Returns None if step not done, dict otherwise.
#        """
#        if not self.is_done(step):
#            return None
#        return self.load_metadata(step)
#
#    def reset(self, step: str = None):
#        if step:
#            f = self._path(step)
#            f.unlink(missing_ok=True)
#            log.warning(f"Reset checkpoint: {step}")
#        else:
#            removed = 0
#            for f in self.ckpt_dir.glob("*.done"):
#                f.unlink()
#                removed += 1
#            log.warning(f"Reset ALL {removed} checkpoints — running from scratch")
#
#    def list_completed(self) -> list:
#        result = []
#        for f in sorted(self.ckpt_dir.glob("*.done")):
#            try:
#                info = json.loads(f.read_text())
#                tag = f" [{info['module']} {info['version']}]" if info.get("module") else ""
#                result.append(f"{info['step']}{tag}  ({info['timestamp']})")
#            except Exception:
#                result.append(f"{f.stem}  (corrupt checkpoint)")
#        return result


"""
HyphaeSBin Checkpoint System
============================
Crash-safe pipeline resumption — like Nextflow's -resume.

Every step saves a .done file. On rerun, completed steps are
automatically skipped. Just re-run the same command to resume.

Checkpoint files: outdir/checkpoints/STEPNAME.done
Each file is JSON with: step name, timestamp, module/version, the
output files that step is expected to have produced, and a free-form
metadata dict the caller controls.

WHAT THIS CLASS DELIBERATELY DOES NOT DO: compute or compare
fingerprints. Every pipeline module (preprocessing.py, coverage.py,
tnf_gene.py, te_composition.py, encoder.py, clustering.py) already
builds its own fingerprint string from whatever inputs/config/version
actually matter to IT (see each module's own `_fingerprint`/
`_file_fingerprint` helpers) and stores it as an opaque value inside
the `metadata` dict it hands to `mark_done` (conventionally under the
key "_fp"). This class never inspects or interprets that value — it
just stores and returns it verbatim. Centralizing fingerprint LOGIC
here would mean every module either shares one fingerprint shape or
routes its fingerprint through this file's opinion of what matters;
keeping it out lets each module's checkpoint be exactly as strict (or
as loose) as that module's own inputs require, decided in exactly one
place (that module) rather than two.

v2 changes (see module-level CHANGELOG below for why):
    - Atomic writes: a checkpoint file is written to a temp file in
      the same directory and atomically replaced into place, so a
      crash or kill mid-write can never leave a half-written .done
      file that then reads as "corrupt" (or worse, reads as valid
      JSON with truncated/wrong content) on the next run.
    - Output-file verification: mark_done() accepts an optional
      `output_files` list; is_done() then checks that every path in
      it still exists on disk before accepting the checkpoint as
      valid. A step whose recorded outputs were deleted (or moved) is
      treated as NOT done — exactly the same "stale checkpoint is a
      cache miss, not a silently wrong answer" principle every
      pipeline module already applies to ITS OWN fingerprint checks,
      now available as one shared primitive instead of every caller
      re-implementing the existence-check by hand.
    - Module/version metadata: mark_done() accepts optional `module`
      and `version` strings, stored alongside the step's own metadata
      dict. This is informational only (never affects is_done()'s
      answer) — it lets a caller (main.py, or a human reading
      checkpoints/*.done) see which module/version produced a given
      checkpoint without opening the file's `metadata` blob and
      guessing which key might hold that information.

Both new mark_done() parameters are optional and keyword-only-by-
convention (positional-compatible with every existing call site in
this codebase, all of which call `mark_done(step, metadata_dict)` or
`mark_done(step, metadata_dict, output_files=..., ...)`) — no existing
caller needs to change.

Usage:
    ckpt = Checkpoint("results/")

    if not ckpt.is_done("step0_n50"):
        result = do_work()
        ckpt.mark_done("step0_n50", metadata=result,
                        output_files=[result["output_path"]],
                        module="preprocessing", version="v13steps")
    else:
        result = ckpt.load_metadata("step0_n50")  # load previous result
"""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from hyphaesbin.utils.logger import get_logger

log = get_logger("checkpoint")

# ── force-trust-fingerprint override ────────────────────────────────────
# Set via config['force_trust_checkpoints'] — call set_force_trust_fingerprint()
# ONCE at startup (main.py) with that config value. When True, every
# module's own fingerprint comparison (`prev.get("_fp") != fp`, duplicated
# in preprocessing.py/coverage.py/tnf_gene.py/encoder.py/te_composition.py/
# clustering.py — see this file's docstring on why fingerprint LOGIC lives
# in each module, not here) is made to report a match unconditionally,
# WITHOUT editing any of those 6 files. Mechanism: load_metadata() below
# swaps in a sentinel value for "_fp" that compares equal to anything a
# caller checks it against, so every module's own comparison line —
# untouched — evaluates to "fingerprint matches" no matter what changed.
# A checkpoint is then skipped purely on "the .done file + its recorded
# outputs exist on disk" (still enforced by is_done() below, unaffected by
# this flag) — config/threshold/input changes no longer force a re-run.
#
# REAL TRADEOFF, not a free lunch: this is a blunt "trust the cache no
# matter what" switch. Turn it on, then change dedup_ani, a rescue
# reference, barrnap_kingdom, the assembly path, anything — the pipeline
# will NOT notice and will silently reuse last run's answer. The only way
# to force a real re-run of a step while this is on is to delete that
# step's checkpoint file (or its outdir) yourself.
_FORCE_TRUST_FINGERPRINT = False


def set_force_trust_fingerprint(enabled: bool) -> None:
    """Call ONCE at startup (main.py), driven by config['force_trust_checkpoints'].
    This is a process-wide switch (there is normally only one pipeline run
    per process), not per-Checkpoint-instance — safe to call before or
    after any Checkpoint() object is constructed."""
    global _FORCE_TRUST_FINGERPRINT
    _FORCE_TRUST_FINGERPRINT = bool(enabled)
    if enabled:
        log.warning("force_trust_checkpoints=True — every step's fingerprint check is now "
                    "bypassed PROCESS-WIDE. Config/input changes will NOT trigger a re-run "
                    "of an already-completed step; only deleting that step's checkpoint "
                    "file will. Use with care.")


class _AlwaysEqualToAnything:
    """Sentinel object that compares equal to anything (`== `/`!=` always
    resolve as 'equal'). load_metadata() hands this back as the '_fp' value
    when force-trust is on, so every module's own
    `prev.get('_fp') != fp` fingerprint check — unchanged, in its own
    file — evaluates to False (no mismatch) regardless of what `fp`
    actually is this run."""
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __repr__(self):
        return "<force_trust_checkpoints: fingerprint check bypassed by config>"


_ALWAYS_EQUAL = _AlwaysEqualToAnything()


class Checkpoint:

    def __init__(self, outdir: str):
        self.ckpt_dir = Path(outdir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, step: str) -> Path:
        return self.ckpt_dir / f"{step}.done"

    def is_done(self, step: str) -> bool:
        f = self._path(step)
        if not f.exists():
            return False
        try:
            info = json.loads(f.read_text())
        except (json.JSONDecodeError, IOError, UnicodeDecodeError):
            log.warning(f"Corrupt checkpoint {f} — treating as not done (will re-run step)")
            f.unlink(missing_ok=True)
            return False

        # A checkpoint whose recorded outputs no longer exist on disk is a
        # cache miss, not a silently-reused stale answer — mirrors the
        # pattern every module's own fingerprint check already applies to
        # ITS output paths, generalized here so callers don't each
        # reimplement it (see e.g. clustering.py's _checkpoint_ok, which
        # additionally layers its OWN fingerprint comparison on top of this
        # existence check — this class only owns the existence half).
        output_files = info.get("output_files") or []
        missing = [p for p in output_files if p and not Path(p).exists()]
        if missing:
            log.warning(
                f"Checkpoint '{step}' exists but {len(missing)} referenced output(s) no "
                f"longer exist on disk ({missing[:3]}{', ...' if len(missing) > 3 else ''}) "
                f"— treating as not done."
            )
            return False

        # NOTE: do NOT log "SKIP" here. is_done() only proves the checkpoint
        # file exists and its recorded outputs are present on disk -- it
        # knows nothing about each module's own _checkpoint_ok()/fingerprint
        # check layered on top, which can still decide the checkpoint is
        # stale (config/inputs changed) and re-run anyway. Logging "SKIP"
        # here used to fire unconditionally before that fingerprint check
        # ran, so a stale-checkpoint re-run printed a contradictory
        # "SKIP ... " immediately followed by "... ignoring stale checkpoint
        # and re-running." The "SKIP" log now lives in each caller's
        # _checkpoint_ok(), at the one point that's actually final.
        return True

    def mark_done(self, step: str, metadata: Optional[dict] = None,
                   output_files: Optional[List[str]] = None,
                   module: Optional[str] = None, version: Optional[str] = None):
        payload = {
            "step": step,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "module": module,
            "version": version,
            "output_files": [str(p) for p in output_files] if output_files else [],
            "metadata": metadata or {},
        }
        f = self._path(step)
        # Atomic write: write to a temp file in the SAME directory (so the
        # final os.replace is a same-filesystem rename, not a cross-device
        # copy) then swap it into place. A crash/kill between these two
        # lines leaves either the old .done file (if any) or nothing —
        # never a truncated/partial one.
        fd, tmp_path = tempfile.mkstemp(dir=str(self.ckpt_dir), prefix=f".{step}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as tmp_f:
                tmp_f.write(json.dumps(payload, indent=2, default=str))
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp_path, f)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        log.debug(f"Checkpoint saved: {f}")

    def load_metadata(self, step: str) -> dict:
        """Load the caller-supplied metadata dict from a completed
        checkpoint (NOT the full payload — use is_done()/the checkpoint
        file directly for output_files/module/version).

        When force-trust mode is on (config['force_trust_checkpoints'], see
        set_force_trust_fingerprint() above), the returned dict's '_fp'
        entry is replaced with a sentinel that compares equal to anything —
        this is the ONLY place that mode takes effect; every module's own
        fingerprint-comparison code is untouched."""
        f = self._path(step)
        if f.exists():
            try:
                meta = json.loads(f.read_text()).get("metadata", {})
            except Exception:
                return {}
            if _FORCE_TRUST_FINGERPRINT:
                meta = dict(meta)
                meta["_fp"] = _ALWAYS_EQUAL
            return meta
        return {}

    def load_full(self, step: str) -> Optional[Dict[str, Any]]:
        """Load the full checkpoint payload (step, timestamp, module,
        version, output_files, metadata) — use this when you need the
        module/version provenance, not just the metadata dict."""
        f = self._path(step)
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text())
        except Exception:
            return None

    # ALIAS: get_status = load_metadata (for backward compatibility with main.py)
    def get_status(self, step: str) -> Optional[Dict[str, Any]]:
        """
        Get status/metadata of a completed step.
        Returns None if step not done, dict otherwise.
        """
        if not self.is_done(step):
            return None
        return self.load_metadata(step)

    def reset(self, step: str = None):
        if step:
            f = self._path(step)
            f.unlink(missing_ok=True)
            log.warning(f"Reset checkpoint: {step}")
        else:
            removed = 0
            for f in self.ckpt_dir.glob("*.done"):
                f.unlink()
                removed += 1
            log.warning(f"Reset ALL {removed} checkpoints — running from scratch")

    def list_completed(self) -> list:
        result = []
        for f in sorted(self.ckpt_dir.glob("*.done")):
            try:
                info = json.loads(f.read_text())
                tag = f" [{info['module']} {info['version']}]" if info.get("module") else ""
                result.append(f"{info['step']}{tag}  ({info['timestamp']})")
            except Exception:
                result.append(f"{f.stem}  (corrupt checkpoint)")
        return result

# PROFILE HOOKS v1 — report completed work without changing checkpoint semantics.
from hyphaesbin.utils.resource_profiler import cache_event
_original_mark_done = Checkpoint.mark_done
def _profiled_mark_done(self, step, *args, **kwargs):
    result = _original_mark_done(self, step, *args, **kwargs)
    cache_event('computed', step)
    return result
Checkpoint.mark_done = _profiled_mark_done
