"""
hyphaesbin Dependency / Environment Diagnostics
================================================
Detects what's available (Python packages, external executables, and
compute backends) and reports clear, actionable diagnostics.

CHANGELOG FROM v1 (the old "hyphaesbin Dependency Manager")
------------------------------------------------------------
The old version of this file auto-installed missing packages/tools via
`conda install` / `pip install` at import time, with no way to opt out
short of editing the file. Every other module in this pipeline
(encoder.py, clustering.py, tnf_gene.py) deliberately moved AWAY from
that pattern to a "raise a clear ImportError / log a clear diagnostic,
never silently install or fall back" contract — an unpinned,
network-dependent install happening as a SIDE EFFECT of `import
hyphaesbin.utils.deps` is exactly the kind of surprise that contract
exists to prevent (wrong version installed non-reproducibly, installs
failing/hanging on an offline HPC compute node, `conda install`
mutating a shared environment other jobs depend on, etc.).

This version:
  - NEVER installs anything, ever. It only detects and reports.
  - Runs nothing at import time — every check here is a plain function
    call the caller (main.py) makes explicitly, when it wants to know.
  - Separates REQUIRED from OPTIONAL Python packages (optional ones
    missing just mean a feature/backend is unavailable, not that the
    pipeline can't run at all — e.g. hdbscan is required for
    clustering.py's core algorithm, but skani/EukCC's absence only
    disables dedup/QC, and cuML's absence only disables the GPU
    clustering backend).
  - Reports versions (Python packages via `importlib.metadata`,
    external tools via each tool's own `--version`/`-v` flag where one
    exists) instead of just True/False, so a report is actually useful
    for reproducibility records (this is what main.py's "report output
    directories and manifests at completion" pulls from for the
    software-versions section of its own summary).
  - Detects CUDA/MPS/CPU availability via torch (when torch is
    installed) as a genuine hardware probe, distinct from a package
    being merely importABLE — mirrors encoder.py's own
    `resolve_device()` and clustering.py's own
    `resolve_clustering_backend()`/`_gpu_available()`, which do their
    OWN backend resolution internally and are NOT changed by this
    file. `describe_backend_choice()` below is a DIAGNOSTIC/reporting
    helper for main.py to log what's available BEFORE calling those
    modules — it never overrides what encoder.py/clustering.py decide
    for themselves.
  - `check_all()` returns one dict with everything main.py needs to
    decide whether to proceed or stop, and to print a clear report
    either way.

Required tool/package lists here mirror what each already-finalized
module's own docstrings/code actually declare as required — this file
doesn't invent its own opinion of what's required.
"""

import importlib
import importlib.metadata
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

# ── Python packages: (import_name, distribution_name, required) ────────────────
# distribution_name is what `importlib.metadata.version()` looks up (usually
# the same as the pip/conda package name, which can differ from the import
# name — e.g. import "yaml" but the distribution is "PyYAML").
PYTHON_DEPS: List[Tuple[str, str, bool]] = [
    ("yaml",       "PyYAML",        True),
    ("numpy",      "numpy",         True),
    ("pandas",     "pandas",        True),
    ("scipy",      "scipy",         True),
    ("sklearn",    "scikit-learn",  True),
    ("Bio",        "biopython",     True),
    ("torch",      "torch",         True),
    # Optional: missing only disables the specific feature named.
    ("hdbscan",    "hdbscan",       False),   # clustering.py's core algorithm --
                                               # technically REQUIRED for
                                               # clustering.py to import at all
                                               # (it raises ImportError itself),
                                               # but listed optional here since
                                               # the rest of the pipeline
                                               # (preprocessing/coverage/TNF/TE/
                                               # encoder) runs fine without it.
    ("faiss",      "faiss-cpu",     False),   # coverage.py's preferred k-NN
                                               # backend; falls back to scipy
                                               # cKDTree, then a numpy brute-
                                               # force fallback, on its own.
    ("cuml",       "cuml",          False),   # clustering.py's GPU HDBSCAN path
    ("cupy",       "cupy",          False),   # clustering.py's GPU probe
]

# ── External executables: (binary_name, used_by, required_for) ─────────────────
# `used_by` names the module; `required_for` is a one-line human summary of
# what breaks (or gets skipped) if the binary is missing, so a diagnostic
# report tells you exactly what to expect, not just a bare boolean.
EXTERNAL_TOOLS: List[Tuple[str, str, str, List[str]]] = [
    # (binary, used_by, required_for, version_cmd)
    ("skani",       "preprocessing.py (dedup) / clustering.py (dedup + final check)",
     "REQUIRED for preprocessing step 5 (dedup); clustering's dedup/consistency-check steps "
     "log a warning and skip (bins pass through unchanged) if missing.",
     ["skani", "--version"]),
    ("seqkit",      "preprocessing.py (length pre-filter)",
     "REQUIRED for preprocessing step 4.", ["seqkit", "version"]),
    ("minimap2",    "preprocessing.py (read mapping)",
     "REQUIRED for preprocessing step 9.", ["minimap2", "--version"]),
    ("samtools",    "preprocessing.py (read mapping)",
     "REQUIRED for preprocessing step 9.", ["samtools", "--version"]),
    ("coverm",      "preprocessing.py (coverage table)",
     "REQUIRED for preprocessing step 9.", ["coverm", "--version"]),
    ("barrnap",     "preprocessing.py (rDNA masking)",
     "REQUIRED for preprocessing step 8.", ["barrnap", "--version"]),
    ("tiara",       "preprocessing.py (classification)",
     "REQUIRED for preprocessing step 6.", ["tiara", "--version"]),
    ("whokaryote.py", "preprocessing.py (classification)",
     "REQUIRED for preprocessing step 6.", ["whokaryote.py", "--version"]),
    ("mmseqs",      "te_composition.py (te_mode=fast)",
     "REQUIRED only when te_mode='fast' (the default).", ["mmseqs", "version"]),
    ("RepeatMasker", "te_composition.py (te_mode=efficient)",
     "REQUIRED only when te_mode='efficient'.", ["RepeatMasker", "-v"]),
    ("conda",       "clustering.py (EukCC + EukCC merge)",
     "Needed to invoke EukCC (its own conda env) — clustering logs a warning "
     "and skips EukCC QC/merge if missing (never crashes).", ["conda", "--version"]),
]


def _tool_version(cmd: List[str]) -> Optional[str]:
    """Best-effort `tool --version`-style probe. Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        text = (r.stdout or r.stderr or "").strip().splitlines()
        return text[0] if text else None
    except Exception:
        return None


def _package_version(import_name: str, distribution_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution_name)
    except Exception:
        try:
            mod = importlib.import_module(import_name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None


def check_python_deps() -> Dict[str, dict]:
    """
    Returns {import_name: {"available": bool, "required": bool,
                            "version": str|None, "distribution": str}}
    Never installs anything. Never raises for a missing package —
    that's the caller's decision (main.py stops on a missing REQUIRED
    one; an OPTIONAL one is just reported).
    """
    report = {}
    for import_name, dist_name, required in PYTHON_DEPS:
        try:
            importlib.import_module(import_name)
            available = True
        except ImportError:
            available = False
        report[import_name] = {
            "available": available,
            "required": required,
            "version": _package_version(import_name, dist_name) if available else None,
            "distribution": dist_name,
        }
    return report


def check_external_tools() -> Dict[str, dict]:
    """
    Returns {binary_name: {"available": bool, "path": str|None,
                            "version": str|None, "used_by": str,
                            "required_for": str}}
    Pure detection via shutil.which() + a best-effort version probe —
    never installs, never modifies PATH.
    """
    report = {}
    for binary, used_by, required_for, version_cmd in EXTERNAL_TOOLS:
        path = shutil.which(binary)
        report[binary] = {
            "available": path is not None,
            "path": path,
            "version": _tool_version(version_cmd) if path else None,
            "used_by": used_by,
            "required_for": required_for,
        }
    return report


def check_compute_backends() -> Dict[str, object]:
    """
    Hardware/backend probe for reporting purposes ONLY. encoder.py's
    resolve_device() and clustering.py's resolve_clustering_backend()
    do their own independent resolution at the moment they actually
    need a device — this function never overrides or is consulted by
    them. It exists so main.py can log, up front, what a run is likely
    to get before spending time on phases 1-4.
    """
    result = {
        "cpu": True,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
        "mps_available": False,
        "cuml_available": False,
        "cupy_cuda_devices": 0,
        "torch_version": None,
    }
    try:
        import torch
        result["torch_version"] = getattr(torch, "__version__", None)
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["cuda_device_count"] = torch.cuda.device_count()
            result["cuda_device_names"] = [torch.cuda.get_device_name(i)
                                            for i in range(result["cuda_device_count"])]
        result["mps_available"] = bool(getattr(torch.backends, "mps", None)
                                        and torch.backends.mps.is_available())
    except Exception:
        pass
    try:
        import cuml  # noqa: F401
        result["cuml_available"] = True
    except Exception:
        pass
    try:
        import cupy
        result["cupy_cuda_devices"] = cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pass
    return result


def describe_backend_choice(requested: str, backends: Optional[Dict] = None) -> str:
    """
    Pure description, no side effects, no decision-making: given a
    requested backend string ("cpu"/"gpu"/"auto") and the output of
    check_compute_backends(), describes in one line what that request
    will most likely resolve to, WITHOUT actually resolving it — the
    real resolution (and the authoritative requested/actual pair
    recorded in each module's own manifest) always happens inside
    encoder.py / clustering.py themselves at run time. This is purely
    so main.py can print an informative line before those phases run.
    """
    backends = backends or check_compute_backends()
    requested = (requested or "auto").lower()
    gpu_usable = backends["cuda_available"] or backends["mps_available"]
    if requested == "cpu":
        return "cpu (explicitly requested)"
    if requested == "gpu":
        return ("gpu (cuda/mps detected)" if gpu_usable
                else "gpu requested but NOT detected — the module will log a "
                     "warning and fall back to cpu at run time")
    # auto
    return "gpu (auto-detected)" if gpu_usable else "cpu (no gpu detected)"


def check_conda_envs(config: Optional[Dict] = None) -> Dict[str, dict]:
    """
    No-op placeholder (env_manager.py / the conda sub-environment-switching
    feature was removed from this pipeline). Kept only so check_all()'s
    signature/return shape doesn't change for callers. Always returns {}.
    """
    return {}


def check_all(external_tools_needed: Optional[List[str]] = None,
              config: Optional[Dict] = None) -> Dict[str, object]:
    """
    One-call diagnostic sweep. `external_tools_needed` optionally
    restricts the "missing_required_tools" verdict to a subset of
    EXTERNAL_TOOLS (e.g. skip mmseqs/RepeatMasker checks if the run's
    te_mode makes only one of them relevant) — every tool is still
    reported either way, this only affects which absences are flagged
    as blocking. `config`, if given, additionally reports (never creates)
    the status of the pipeline's named conda sub-environments — see
    check_conda_envs().

    Returns:
        {
          "python": check_python_deps() result,
          "tools": check_external_tools() result,
          "backends": check_compute_backends() result,
          "conda_envs": check_conda_envs(config) result (empty if no config given),
          "missing_required_python": [import_name, ...],
          "missing_required_tools": [binary_name, ...] (filtered to
              external_tools_needed if given),
          "ok": bool  -- True iff no required Python package AND no
              required (per external_tools_needed) tool is missing.
        }
    """
    python_report = check_python_deps()
    tools_report = check_external_tools()
    backends = check_compute_backends()
    conda_envs_report = check_conda_envs(config)

    missing_python = [name for name, info in python_report.items()
                       if info["required"] and not info["available"]]

    if external_tools_needed is None:
        candidate_tools = list(tools_report.keys())
    else:
        candidate_tools = external_tools_needed
    missing_tools = [name for name in candidate_tools
                      if name in tools_report and not tools_report[name]["available"]]

    return {
        "python": python_report,
        "tools": tools_report,
        "backends": backends,
        "conda_envs": conda_envs_report,
        "missing_required_python": missing_python,
        "missing_required_tools": missing_tools,
        "ok": not missing_python and not missing_tools,
    }


def format_report(report: Dict[str, object]) -> str:
    """Human-readable rendering of check_all()'s output, for main.py to
    print/log verbatim."""
    lines = []
    lines.append("Python packages:")
    for name, info in report["python"].items():
        mark = "OK" if info["available"] else ("MISSING (required)" if info["required"] else "missing (optional)")
        ver = f" v{info['version']}" if info["version"] else ""
        lines.append(f"  {name:10s} [{info['distribution']}]{ver:12s} {mark}")
    lines.append("")
    lines.append("External executables:")
    for name, info in report["tools"].items():
        mark = "OK" if info["available"] else "MISSING"
        ver = f" ({info['version']})" if info["version"] else ""
        lines.append(f"  {name:14s} {mark}{ver}  -- {info['required_for']}")
    lines.append("")
    b = report["backends"]
    lines.append("Compute backends:")
    lines.append(f"  torch          : {b['torch_version'] or 'not installed'}")
    lines.append(f"  cuda_available : {b['cuda_available']}"
                  + (f"  ({b['cuda_device_count']}x {', '.join(b['cuda_device_names'])})"
                     if b['cuda_available'] else ""))
    lines.append(f"  mps_available  : {b['mps_available']}")
    lines.append(f"  cuml_available : {b['cuml_available']}")
    lines.append("")
    if report.get("conda_envs"):
        lines.append("Conda sub-environments:")
        for name, info in report["conda_envs"].items():
            lines.append(f"  {name:20s} {'OK — exists' if info['exists'] else 'MISSING'}")
        lines.append("")
    if report["missing_required_python"]:
        lines.append(f"MISSING REQUIRED PYTHON PACKAGES: {report['missing_required_python']}")
    if report["missing_required_tools"]:
        lines.append(f"MISSING REQUIRED EXTERNAL TOOLS: {report['missing_required_tools']}")
    if report["ok"]:
        lines.append("All required dependencies present.")
    return "\n".join(lines)