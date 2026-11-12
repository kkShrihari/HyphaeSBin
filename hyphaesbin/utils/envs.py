"""
hyphaesbin.utils.envs -- environments + optional automatic installation (config: auto_install: true/false).

Why TWO conda environments (+ EukCC's own):
  HyphaeSBin_env       everything the pipeline imports or calls (python 3.10, pytorch 2.x, hdbscan, faiss, minimap2,
                       samtools, seqkit, skani, coverm, barrnap + hmmer, prodigal, mmseqs2)
  HyphaeSBin_classify  Tiara + Whokaryote ONLY. They cannot share an env with the pipeline: tiara 1.0.3 requires
                       python < 3.10, torch ~= 1.7.0 and numpy ~= 1.19.4, whokaryote 1.1.2 pins python 3.8 and
                       scikit-learn 1.0.2 (verified with a solver). Forcing them into one env would downgrade the
                       encoder's torch / numpy and break hdbscan/faiss.
  EukCC                its own env as before (eukcc_conda_env).
The classifier env's bin/ is APPENDED to PATH at start-up (never prepended), so `tiara` and `whokaryote.py` resolve
there while every other tool still comes from HyphaeSBin_env. Users never activate it.

auto_install (default true): a missing tool/package/env is installed into the right env without asking;
false: nothing is installed, missing items are reported and the stage that needs them stops with the exact command.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from hyphaesbin.utils.logger import get_logger

log = get_logger("hyphaesbin.envs")

MAIN_ENV = "HyphaeSBin_env"
CLASSIFY_ENV = "HyphaeSBin_classify"
CHANNELS = ["-c", "conda-forge", "-c", "bioconda", "--strict-channel-priority"]

# executable -> conda package (installed into the ACTIVE env = HyphaeSBin_env)
MAIN_TOOLS: Dict[str, str] = {
    "minimap2": "minimap2", "samtools": "samtools", "seqkit": "seqkit", "skani": "skani", "coverm": "coverm",
    "barrnap": "barrnap>=1.10", "nhmmer": "hmmer", "prodigal": "prodigal", "mmseqs": "mmseqs2",
}
# import name -> conda package
MAIN_PYTHON: Dict[str, str] = {
    "yaml": "pyyaml", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
    "Bio": "biopython", "torch": "pytorch", "hdbscan": "hdbscan", "psutil": "psutil", "faiss": "faiss-cpu",
}
# classifier env recipe (solver-verified)
CLASSIFY_CONDA = ["python=3.8", "numpy=1.19.4", "scikit-learn=1.0.2", "pandas", "biopython", "joblib", "prodigal",
                  "pip", "tqdm", "numba"]
CLASSIFY_NODEPS = ["whokaryote=1.1.2"]
CLASSIFY_PIP = ["tiara==1.0.3"]


def conda_exe() -> Optional[str]:
    for exe in ("mamba", "micromamba", "conda"):
        p = shutil.which(exe)
        if p:
            return p
    for var in ("CONDA_EXE", "MAMBA_EXE"):
        if os.environ.get(var) and Path(os.environ[var]).exists():
            return os.environ[var]
    return None


def env_prefix(name: str) -> Optional[Path]:
    """Prefix of a named env (siblings of the active env first, then `conda env list --json`)."""
    cands = []
    act = os.environ.get("CONDA_PREFIX")
    if act:
        p = Path(act)
        if p.name == name:
            return p
        cands += [p.parent / name, p / "envs" / name]
    for var in ("MAMBA_ROOT_PREFIX", "CONDA_ROOT"):
        if os.environ.get(var):
            cands.append(Path(os.environ[var]) / "envs" / name)
    for c in cands:
        if (c / "conda-meta").is_dir():
            return c
    exe = conda_exe()
    if exe:
        try:
            r = subprocess.run([exe, "env", "list", "--json"], capture_output=True, text=True, timeout=120)
            for e in json.loads(r.stdout or "{}").get("envs", []):
                if Path(e).name == name:
                    return Path(e)
        except Exception:
            pass
    return None


def _run(argv: List[str], what: str) -> bool:
    log.warning(f"auto_install: {what}: {' '.join(argv)}")
    try:
        r = subprocess.run(argv, capture_output=True, text=True)
    except OSError as e:
        log.error(f"auto_install: {what} failed to start: {e}")
        return False
    if r.returncode != 0:
        log.error(f"auto_install: {what} FAILED (exit {r.returncode}):\n{(r.stderr or r.stdout)[-2000:]}")
        return False
    return True


def _append_path(bin_dir: Path):
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(bin_dir) not in parts:
        os.environ["PATH"] = os.pathsep.join(parts + [str(bin_dir)])


def ensure_classifier_env(auto_install: bool, name: str = CLASSIFY_ENV) -> Dict:
    """Make `tiara` + `whokaryote.py` resolvable. Order: already on PATH -> existing env -> build it (auto_install)."""
    if shutil.which("tiara") and shutil.which("whokaryote.py"):
        return {"status": "on_path", "tiara": shutil.which("tiara")}
    pre = env_prefix(name)
    if pre is None and auto_install:
        exe = conda_exe()
        if exe is None:
            return {"status": "missing", "error": "no conda/mamba/micromamba to build the classifier env"}
        ok = _run([exe, "create", "-y", "-n", name] + CHANNELS + CLASSIFY_CONDA, f"create {name}")
        ok = ok and _run([exe, "install", "-y", "-n", name, "--no-deps"] + CHANNELS + CLASSIFY_NODEPS,
                         "add whokaryote")
        pre = env_prefix(name)
        if ok and pre is not None:
            _run([str(pre / "bin" / "python"), "-m", "pip", "install", "--no-cache-dir"] + CLASSIFY_PIP, "add tiara")
    if pre is None:
        return {"status": "missing", "error": f"env '{name}' not found"
                + ("" if auto_install else " (auto_install=false)")}
    repaired = _repair_classifier_env(pre, name) if auto_install else []
    _append_path(pre / "bin")
    ok = bool(shutil.which("tiara") and shutil.which("whokaryote.py"))
    out = {"status": "ok" if ok else "incomplete", "prefix": str(pre), "repaired": repaired}
    if not ok:
        miss = [t for t in ("tiara", "whokaryote.py", "prodigal") if not (pre / "bin" / t).exists()]
        out["error"] = (f"{name} is missing {miss}" + ("" if auto_install else
                        " (auto_install=false -> run: bash install.sh --force)"))
    return out


def _repair_classifier_env(pre: Path, name: str) -> List[str]:
    """v9.2: an EXISTING classifier env that lacks Tiara / Whokaryote / Prodigal / its pinned base is completed in
    place (same recipe as a fresh build). Returns what was (re)installed."""
    done: List[str] = []
    exe = conda_exe()
    bin_ = pre / "bin"
    py = bin_ / "python"
    if exe and not (py.exists() and (bin_ / "prodigal").exists()):
        if _run([exe, "install", "-y", "-p", str(pre)] + CHANNELS + CLASSIFY_CONDA, f"repair base of {name}"):
            done.append("base")
    if exe and not (bin_ / "whokaryote.py").exists():
        if _run([exe, "install", "-y", "-p", str(pre), "--no-deps"] + CHANNELS + CLASSIFY_NODEPS,
                f"repair whokaryote in {name}"):
            done.append("whokaryote")
    if py.exists() and not (bin_ / "tiara").exists():
        # tiara's deps may still be present while its entry script is gone -> force a reinstall of tiara itself
        _run([str(py), "-m", "pip", "install", "--no-cache-dir"] + CLASSIFY_PIP, f"repair tiara deps in {name}")
        if _run([str(py), "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "--no-deps"] + CLASSIFY_PIP,
                f"repair tiara in {name}"):
            done.append("tiara")
    return done


def ensure_main_tools(auto_install: bool) -> List[str]:
    want = dict(MAIN_TOOLS)
    missing = [exe for exe in want if shutil.which(exe) is None]
    if missing and auto_install:
        exe = conda_exe()
        prefix = os.environ.get("CONDA_PREFIX")
        if exe and prefix:
            _run([exe, "install", "-y", "-p", prefix] + CHANNELS + [want[m] for m in missing],
                 f"install {missing} into {Path(prefix).name}")
        missing = [e for e in want if shutil.which(e) is None]
    return missing


def ensure_python(auto_install: bool) -> List[str]:
    missing = []
    for mod in MAIN_PYTHON:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing and auto_install:
        exe = conda_exe()
        prefix = os.environ.get("CONDA_PREFIX")
        if exe and prefix:
            _run([exe, "install", "-y", "-p", prefix] + CHANNELS + [MAIN_PYTHON[m] for m in missing],
                 f"install python packages {missing}")
            importlib.invalidate_caches()
        missing = []
        for mod in MAIN_PYTHON:
            try:
                importlib.import_module(mod)
            except Exception:
                missing.append(mod)
    return missing


def prepare(config: Dict) -> Dict:
    """Called once by main.py before anything runs. Never raises."""
    auto = config.get("auto_install", config.get("auto_install_tools", True))
    auto = auto is True or str(auto).strip().lower() in ("1", "true", "yes", "on")
    config["auto_install_tools"] = auto           # preprocessing reads this key
    out = {"auto_install": auto}
    try:
        out["classifier_env"] = ensure_classifier_env(auto, str(config.get("classifier_env", CLASSIFY_ENV)))
        out["missing_tools"] = ensure_main_tools(auto)
        out["missing_python"] = ensure_python(auto)
    except Exception as e:                         # environment problems are reported by the stages themselves
        out["error"] = f"{type(e).__name__}: {e}"
    lvl = log.info if not (out.get("missing_tools") or out.get("missing_python")) else log.warning
    lvl(f"ENV  auto_install={auto}  classifier_env={out.get('classifier_env', {}).get('status')}  "
        f"missing_tools={out.get('missing_tools')}  missing_python={out.get('missing_python')}")
    return out
