"""
HyphaeSBin TNF Module — Whole-Contig k4 Frequency (Vectorized) — v2
====================================================================
Workflow:
    final_clean.fasta (MASKED variant — see note below)
        → stream contig records one at a time (gzip-aware)
        → numpy stride-trick sliding window (no Python loops)
        → ACGT→0123 byte mapping
        → base-4 integer encoding
        → precomputed CANONICAL_MAP lookup (256→136)
        → frequency normalization
        → length/confidence-scaled weight
        → 136D TNF matrix
        → save tnf_features.npy + tnf_weights.npy + contig_ids.json

INPUT CONTRACT (verified against preprocessing.py's step8/step13 output
metadata): pass the MASKED assembly, i.e. `final_clean.fasta`, NOT
`final_clean_unmasked.fasta`. preprocessing.py's own step8 metadata records
`"tnf_composition": "masked"` and step13 writes `final_clean.fasta` from the
masked branch specifically so that rRNA-driven composition bias doesn't leak
into TNF. The unmasked variant exists only for read mapping (step 9) and
final biological output — never for TNF/composition.

v2 changes (this round):
  - k is no longer a configurable knob. It was previously accepted in
    `config={"k": ...}` and even echoed into the output schema, but the
    actual computation (CANONICAL_MAP, TNF_DIM=136, sliding-window stride)
    was always hardcoded to k=4 regardless — changing it silently did
    nothing except lie in the schema. k=4 is now a fixed module constant;
    passing "k" in config raises ValueError instead of being silently
    ignored, so a caller who tries it finds out immediately, not by
    discovering a mismatched schema three steps downstream.
  - Checkpoints are now input/config-aware, same fingerprinting pattern as
    preprocessing.py / coverage.py: a completed checkpoint is only trusted
    if the FASTA fingerprint, min_contig_len, weighting config, and
    implementation version all still match, AND every referenced output
    file still exists on disk. Anything else is treated as a cache MISS,
    not a silent stale reuse.
  - Duplicate contig IDs in the FASTA are now rejected outright (fail
    loud) before any feature is computed — ambiguous downstream alignment
    is worse than stopping early.
  - Contig order is now hashed (sha256, first 16 hex chars) and written to
    both the checkpoint metadata and tnf_feature_schema.json, so a
    downstream encoder can verify TNF/TE/coverage were all computed over
    the identical contig order without re-deriving it.
  - gzip input is now supported (.fasta.gz), matching preprocessing.py.
  - Sequences are now streamed one record at a time — only the small
    (136-float) feature vector and a bool/float weight are ever kept
    per contig; the raw sequence string is discarded once its contig's
    vector is computed, instead of the whole assembly being buffered into
    a Python list up front.
  - Weights are confidence-scaled by default instead of binary:
        weight = min(1.0, total_valid_4mers / full_confidence_kmers)   if total_valid_4mers >= min_valid_kmers
        weight = 0.0                                                    otherwise
    A 1000bp contig and a 100,000bp contig no longer get the same weight
    just because both cleared min_contig_len. `weight_mode="binary"`
    keeps the old 0/1 behavior for anyone who wants it (e.g. reproducing
    an older run). min_valid_kmers is a hard floor below which a
    frequency estimate is considered too noisy to use at all, regardless
    of contig length.
  - `assert len(canonical_set) == TNF_DIM` replaced with `raise
    ValueError` — assertions can be stripped with `python -O`, so a build
    time invariant guard must not depend on them.
  - Non-ASCII bytes in a sequence (not standard ambiguity codes — those
    already map safely to the invalid-kmer sentinel via _BYTE_MAP; this is
    for genuinely non-ASCII bytes, e.g. a corrupted or non-FASTA file) now
    raise a clear ValueError naming the offending contig, instead of a
    bare UnicodeEncodeError.
  - `_revcomp_idx` removed — it was unused dead code; the canonical map is
    built (and tested — see test_tnf_gene.py) using the string-based
    `revcomp_str` path only, so there is exactly one reverse-complement
    implementation to keep correct.
  - Reproducibility metadata added to the schema: python/numpy/biopython
    versions, an implementation version string, a feature-name hash, and
    the input contig-order hash.
  - Output shapes are self-validated before returning/saving
    (features.shape == (N,136), weights.shape == (N,1), len(contig_ids)
    == N) — cheap insurance in addition to whatever the encoder checks.

Speed: sliding-window computation is still ~50-100x faster than a
loop-based per-kmer implementation (unchanged from v1).
"""

import gzip
import hashlib
import json
import platform
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import Bio
    from Bio import SeqIO
except ImportError as e:
    raise ImportError("Biopython required: pip install biopython") from e

try:
    from hyphaesbin.utils.logger import get_logger, log_step_start, log_step_done
except Exception:
    import logging
    def get_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
    def log_step_start(log, step, name, total): log.info(f"STEP {step}/{total}: {name}")
    def log_step_done(log, step, name, elapsed): log.info(f"DONE STEP {step}: {name} in {elapsed:.2f}s")

try:
    from hyphaesbin.utils.checkpoint import Checkpoint
except Exception:
    Checkpoint = None

log = get_logger("tnf_gene")

IMPLEMENTATION_VERSION = "HyphaeSBin_TNF_wholecontig_v2_streaming"

# ── Config ────────────────────────────────────────────────────────────────────
# k is intentionally NOT here — it's a fixed constant (see K below), not a
# tunable. min_contig_len / min_valid_kmers / full_confidence_kmers /
# weight_mode are the only things a caller may override.
TNF_CONFIG = {
    "min_contig_len": 1000,
    "min_valid_kmers": 50,          # below this: weight forced to 0.0, estimate too noisy to use
    "full_confidence_kmers": 5000,  # total valid 4-mers at/above which weight saturates at 1.0
    "weight_mode": "confidence",    # "confidence" (default) | "binary" (old 0/1 behavior)
}
K       = 4          # FIXED. Not configurable — see v2 changelog above.
TNF_DIM = 136
DNA     = "ACGT"

_ALLOWED_CONFIG_KEYS = set(TNF_CONFIG.keys())

# ── Precomputed lookups (module-level, built once) ────────────────────────────

# ACGT byte → 0-3, everything else (N, ambiguity codes, anything) → 255
_BYTE_MAP = np.full(256, 255, dtype=np.uint8)
for _b, _c in zip(b"ACGT", range(4)):
    _BYTE_MAP[_b] = _c


def _revcomp_str(s: str) -> str:
    comp = str.maketrans("ACGT", "TGCA")
    return s.translate(comp)[::-1]


def _build_canonical_map() -> Tuple[List[str], np.ndarray]:
    """
    Returns:
        feature_names : 136 canonical k-mer strings
        CANONICAL_MAP : uint8 array [256] mapping raw 4-mer index -> 136D index
    """
    all_kmers = ["".join(p) for p in product(DNA, repeat=K)]  # 256 4-mers, base-4 big-endian index order

    canonical_set = sorted(set(min(k, _revcomp_str(k)) for k in all_kmers))
    if len(canonical_set) != TNF_DIM:
        raise ValueError(f"Canonical k4 map built {len(canonical_set)} canonical bins, expected "
                          f"{TNF_DIM} — this is a build-time invariant failure, not user input; "
                          f"do not proceed with a mismatched schema.")
    canon_to_136 = {k: i for i, k in enumerate(canonical_set)}

    CMAP = np.zeros(256, dtype=np.uint8)
    for idx, kmer in enumerate(all_kmers):
        canon = min(kmer, _revcomp_str(kmer))
        CMAP[idx] = canon_to_136[canon]

    return canonical_set, CMAP


_FEATURE_NAMES, _CANONICAL_MAP = _build_canonical_map()
_FEATURE_NAMES_HASH = hashlib.sha256(json.dumps(_FEATURE_NAMES).encode()).hexdigest()[:16]

# Base-4 power vector for encoding [b0,b1,b2,b3] -> b0*64 + b1*16 + b2*4 + b3
_POWERS = np.array([64, 16, 4, 1], dtype=np.int32)


# =============================================================================
# CHECKPOINT FINGERPRINTING — same pattern as preprocessing.py / coverage.py.
# A stale or config-mismatched checkpoint is a cache MISS, not a silent
# wrong answer.
# =============================================================================

def _file_fingerprint(path) -> str:
    if path is None:
        return "None"
    p = Path(str(path))
    try:
        st = p.stat()
        return f"{p}:{st.st_size}:{int(st.st_mtime)}"
    except FileNotFoundError:
        return f"{p}:MISSING"


def _fingerprint(*parts) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(parts, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _checkpoint_ok(ckpt, step: str, fp: str, output_paths: List[str]) -> Optional[Dict]:
    if ckpt is None or not ckpt.is_done(step):
        return None
    prev = ckpt.load_metadata(step)
    if prev.get("_fp") != fp:
        log.warning(f"{step}: checkpoint exists but inputs/config changed since it ran — "
                    f"ignoring stale checkpoint and re-running.")
        return None
    missing = [p for p in output_paths if p and not Path(p).exists()]
    if missing:
        log.warning(f"{step}: checkpoint fingerprint matches, but {len(missing)} referenced "
                    f"output(s) no longer exist on disk ({missing[:3]}"
                    f"{', ...' if len(missing) > 3 else ''}) — treating as a cache miss.")
        return None
    return prev


# =============================================================================
# GZIP-AWARE FASTA OPENING
# =============================================================================

def _open_fasta(path):
    path = Path(path)
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _read_fasta_ids_only(path) -> List[str]:
    """Cheap ID-only pass — used to build the duplicate-ID / order-hash
    check without holding sequence data in memory."""
    ids = []
    with _open_fasta(path) as fh:
        for line in fh:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


# =============================================================================
# CORE VECTORIZED COMPUTATION
# =============================================================================

def compute_tnf136(seq: str, contig_id: str = "<unknown>") -> Tuple[np.ndarray, int]:
    """
    Vectorized canonical k4 frequency (136D). No Python loops over k-mers —
    uses numpy stride tricks. Ambiguity codes (N, R, Y, ...) are standard
    ASCII and are already excluded correctly via _BYTE_MAP's 255 sentinel —
    they do not need special-casing here. What DOES need handling is a
    genuinely non-ASCII byte (corrupted file / not actually FASTA), which
    would otherwise surface as a bare UnicodeEncodeError deep in numpy.
    """
    try:
        raw_bytes = seq.encode("ascii")
    except UnicodeEncodeError as e:
        raise ValueError(f"Contig {contig_id!r} contains non-ASCII byte(s) in its sequence — "
                          f"this is not a standard ambiguity code (those are handled safely), "
                          f"it means the input is corrupted or not actually FASTA: {e}") from e

    arr = _BYTE_MAP[np.frombuffer(raw_bytes, dtype=np.uint8)]

    n = len(arr)
    if n < K:
        return np.zeros(TNF_DIM, dtype=np.float32), 0

    shape   = (n - K + 1, K)
    strides = (arr.strides[0], arr.strides[0])
    kmers   = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    valid = ~np.any(kmers == 255, axis=1)
    kmers = kmers[valid]
    total = int(kmers.shape[0])

    if total == 0:
        return np.zeros(TNF_DIM, dtype=np.float32), 0

    raw_idx = kmers.astype(np.int32) @ _POWERS
    canon_idx = _CANONICAL_MAP[raw_idx]

    counts = np.bincount(canon_idx, minlength=TNF_DIM).astype(np.float32)
    counts /= float(total)

    return counts, total


def _confidence_weight(total_valid_kmers: int, cfg: Dict) -> float:
    if total_valid_kmers < cfg["min_valid_kmers"]:
        return 0.0
    if cfg["weight_mode"] == "binary":
        return 1.0
    if cfg["weight_mode"] != "confidence":
        raise ValueError(f"Unknown weight_mode {cfg['weight_mode']!r} — must be "
                          f"'confidence' or 'binary'.")
    return float(min(1.0, total_valid_kmers / cfg["full_confidence_kmers"]))


# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run_tnf_wholecontig(
    scaffold: str,
    outdir: str,
    config: Optional[Dict] = None,
    resume: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parameters
    ----------
    scaffold : path to the MASKED assembly (final_clean.fasta — NOT
               final_clean_unmasked.fasta). gzip (.fasta.gz) supported.
    outdir   : output directory
    config   : override min_contig_len / min_valid_kmers /
               full_confidence_kmers / weight_mode. Passing "k" here raises
               ValueError — k=4 is fixed, not a tunable (see v2 changelog).
    resume   : reuse a checkpoint if inputs/config/implementation all still
               match AND its output files still exist on disk

    Returns
    -------
    features   : (N, 136) float32
    weights    : (N, 1)   float32
    contig_ids : list[str] length N
    """
    t0 = time.time()
    config = config or {}
    bad_keys = set(config.keys()) - _ALLOWED_CONFIG_KEYS
    if bad_keys:
        raise ValueError(f"Unknown config key(s): {sorted(bad_keys)}. Allowed: "
                          f"{sorted(_ALLOWED_CONFIG_KEYS)}. Note: 'k' is not configurable — "
                          f"k=4 is a fixed constant (see module changelog).")
    cfg = {**TNF_CONFIG, **config}
    if cfg["weight_mode"] not in ("confidence", "binary"):
        raise ValueError(f"config['weight_mode'] must be 'confidence' or 'binary', "
                          f"got {cfg['weight_mode']!r}.")
    if cfg["min_contig_len"] < 0:
        raise ValueError(f"config['min_contig_len'] must be >= 0, got {cfg['min_contig_len']!r}.")
    if cfg["min_valid_kmers"] < 0:
        raise ValueError(f"config['min_valid_kmers'] must be >= 0, got {cfg['min_valid_kmers']!r}.")
    if cfg["full_confidence_kmers"] <= 0:
        raise ValueError(f"config['full_confidence_kmers'] must be > 0 (it's a denominator in the "
                          f"confidence-weight formula — 0 or negative would divide by zero or "
                          f"invert the weighting), got {cfg['full_confidence_kmers']!r}.")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not Path(scaffold).exists():
        raise FileNotFoundError(f"Scaffold not found: {scaffold}")

    STEP = "tnf_wholecontig_v2_streaming"
    ckpt = Checkpoint(outdir) if resume and Checkpoint is not None else None

    features_path   = outdir / "tnf_features.npy"
    weights_path    = outdir / "tnf_weights.npy"
    contig_ids_path = outdir / "contig_ids.json"
    schema_path     = outdir / "tnf_feature_schema.json"

    fp = _fingerprint(
        _file_fingerprint(scaffold), cfg["min_contig_len"], cfg["min_valid_kmers"],
        cfg["full_confidence_kmers"], cfg["weight_mode"], K, TNF_DIM,
        _FEATURE_NAMES_HASH, IMPLEMENTATION_VERSION,
    )
    cached = _checkpoint_ok(ckpt, STEP, fp, [str(features_path), str(weights_path), str(contig_ids_path)])
    if cached is not None:
        features   = np.load(features_path)
        weights    = np.load(weights_path)
        with open(contig_ids_path) as f:
            contig_ids = json.load(f)
        log.info(f"[checkpoint] Loaded TNF features: {features.shape}")
        return features, weights, contig_ids

    log_step_start(log, 1, "Whole-contig k4 frequency TNF (vectorized, streaming)", 1)

    # ── Duplicate-ID check up front (cheap ID-only pass, fail loud before
    #    spending time computing any feature) ────────────────────────────────
    all_ids = _read_fasta_ids_only(scaffold)
    dup_ids = sorted({i for i in all_ids if all_ids.count(i) > 1}) if len(set(all_ids)) != len(all_ids) else []
    if dup_ids:
        raise ValueError(f"Scaffold FASTA has {len(dup_ids)} duplicate contig ID(s) "
                          f"(e.g. {dup_ids[:5]}) — cannot safely align TNF features to contigs. "
                          f"Fix the input assembly before computing TNF.")

    min_len = cfg["min_contig_len"]
    feature_rows: List[np.ndarray] = []
    weight_rows: List[float] = []
    contig_ids: List[str] = []
    n_valid, n_short, n_zero = 0, 0, 0

    # ── Stream one record at a time — only the 136-float feature vector and
    #    a scalar weight are retained per contig, never the raw sequence. ───
    with _open_fasta(scaffold) as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            cid = rec.id
            contig_ids.append(cid)
            seqlen = len(rec.seq)

            if seqlen < min_len:
                feature_rows.append(np.zeros(TNF_DIM, dtype=np.float32))
                weight_rows.append(0.0)
                n_short += 1
                continue

            vec, total = compute_tnf136(str(rec.seq).upper(), contig_id=cid)
            weight = _confidence_weight(total, cfg)

            if total == 0:
                n_zero += 1
            else:
                n_valid += 1

            feature_rows.append(vec)
            weight_rows.append(weight)

    N = len(contig_ids)
    features = np.stack(feature_rows, axis=0) if N else np.zeros((0, TNF_DIM), dtype=np.float32)
    weights = np.array(weight_rows, dtype=np.float32).reshape(-1, 1) if N else np.zeros((0, 1), dtype=np.float32)

    # ── Self-validate output shapes before saving/returning ─────────────────
    if features.shape != (N, TNF_DIM):
        raise ValueError(f"Internal error: features.shape={features.shape}, expected ({N}, {TNF_DIM}).")
    if weights.shape != (N, 1):
        raise ValueError(f"Internal error: weights.shape={weights.shape}, expected ({N}, 1).")
    if len(contig_ids) != N:
        raise ValueError(f"Internal error: len(contig_ids)={len(contig_ids)}, expected {N}.")

    n_zero_weight = int((weights[:, 0] == 0.0).sum())
    log.info(f"Contigs total          : {N:,}")
    log.info(f"  usable (weight > 0)  : {N - n_zero_weight:,}")
    log.info(f"  too short (<{min_len}bp) : {n_short:,}")
    log.info(f"  zero valid k-mers    : {n_zero:,}")
    log.info(f"  weight_mode          : {cfg['weight_mode']}")
    if weights.size:
        log.info(f"  weight range         : [{weights.min():.3f}, {weights.max():.3f}]  "
                 f"mean={weights.mean():.3f}")

    contig_order_hash = hashlib.sha256("\n".join(contig_ids).encode()).hexdigest()[:16]

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(features_path, features)
    np.save(weights_path, weights)
    with open(contig_ids_path, "w") as f:
        json.dump(contig_ids, f, indent=2)

    schema = {
        "version": IMPLEMENTATION_VERSION,
        "feature_type": "canonical_k4_frequency",
        "k": K,
        "k_note": "FIXED at 4 — not configurable. See module changelog.",
        "n_features": TNF_DIM,
        "feature_names": _FEATURE_NAMES,
        "feature_names_hash": _FEATURE_NAMES_HASH,
        "normalization": "counts / total_valid_4mers",
        "implementation": "numpy stride_tricks + precomputed CANONICAL_MAP (streaming)",
        "weighting": {
            "mode": cfg["weight_mode"],
            "min_valid_kmers": cfg["min_valid_kmers"],
            "full_confidence_kmers": cfg["full_confidence_kmers"],
            "formula": ("weight = 0 if total_valid_4mers < min_valid_kmers else "
                        "min(1.0, total_valid_4mers / full_confidence_kmers)"
                        if cfg["weight_mode"] == "confidence" else
                        "weight = 1.0 if total_valid_4mers >= min_valid_kmers else 0.0"),
        },
        "config": {k: v for k, v in cfg.items()},
        "input": {
            "scaffold": str(scaffold),
            "scaffold_fingerprint": _file_fingerprint(scaffold),
            "expected_variant": "MASKED (final_clean.fasta) — never final_clean_unmasked.fasta",
        },
        "n_contigs": N,
        "contig_order_hash": contig_order_hash,
        "reproducibility": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "biopython": getattr(Bio, "__version__", "unknown"),
            "implementation_version": IMPLEMENTATION_VERSION,
        },
    }
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    if ckpt:
        ckpt.mark_done(STEP, {
            "_fp": fp,
            "features_path": str(features_path),
            "weights_path": str(weights_path),
            "contig_ids_path": str(contig_ids_path),
            "contig_order_hash": contig_order_hash,
        })

    log_step_done(log, 1, "TNF whole-contig vectorized (streaming)", time.time() - t0)
    return features, weights, contig_ids


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="HyphaeSBin whole-contig k4 TNF (vectorized, streaming)")
    p.add_argument("--scaffold", required=True,
                   help="MASKED assembly FASTA (final_clean.fasta), gzip OK")
    p.add_argument("--outdir", required=True)
    p.add_argument("--min-contig-len", type=int, default=TNF_CONFIG["min_contig_len"])
    p.add_argument("--min-valid-kmers", type=int, default=TNF_CONFIG["min_valid_kmers"])
    p.add_argument("--full-confidence-kmers", type=int, default=TNF_CONFIG["full_confidence_kmers"])
    p.add_argument("--weight-mode", choices=["confidence", "binary"], default=TNF_CONFIG["weight_mode"])
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    run_tnf_wholecontig(
        scaffold=args.scaffold,
        outdir=args.outdir,
        config={
            "min_contig_len": args.min_contig_len,
            "min_valid_kmers": args.min_valid_kmers,
            "full_confidence_kmers": args.full_confidence_kmers,
            "weight_mode": args.weight_mode,
        },
        resume=not args.no_resume,
    )



# PROFILE HOOKS v1 — observation only; scientific functions above are unchanged.
from hyphaesbin.utils.resource_profiler import profile_functions, profile_checkpoint
profile_functions(globals(), ['run_tnf_wholecontig'])
_checkpoint_ok = profile_checkpoint(_checkpoint_ok)

if __name__ == "__main__":
    main()
