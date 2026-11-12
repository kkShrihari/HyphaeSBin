"""One place for version / source reporting (written into clustering_stats.json, results/provenance.json, the log)."""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Dict

_PKGS = ("numpy", "pandas", "scipy", "scikit-learn", "hdbscan", "torch", "faiss-cpu", "faiss-gpu", "pyyaml",
         "biopython", "umap-learn")


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not_installed"


def source_sha256() -> str:
    """Hash of every .py file of the installed hyphaesbin package (identifies the exact code that ran)."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def package_provenance() -> Dict:
    root = Path(__file__).resolve().parents[1]
    ver = _dist_version("hyphaesbin")
    if ver == "not_installed":
        pp = root.parent / "pyproject.toml"
        if pp.exists():
            for line in pp.read_text().splitlines():
                if line.strip().startswith("version"):
                    ver = line.split("=", 1)[1].strip().strip('"') + " (source tree)"
                    break
    return {"package": "hyphaesbin", "version": ver, "source_sha256": source_sha256(), "source_path": str(root),
            "python": platform.python_version(), "python_executable": sys.executable,
            "packages": {p: _dist_version(p) for p in _PKGS}}
