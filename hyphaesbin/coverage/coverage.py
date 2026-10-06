"""
HyphaeS Coverage Module — v9
=====================================
v9 changes (this round):
  - Outlier cap default raised to 99.5th percentile (99.9 recommended for
    very abundant real genomes) — 99.0 was capping real high-copy biology
    more aggressively than necessary.
  - Normalization replaced: per-sample scaling is now literally "divide by
    that sample's median POSITIVE coverage value" (median of capped,
    nonzero depths), not a cumulative-sum-based CSS approximation:
        z_ij = log(1 + x_ij / s_j),   s_j = median(x_.j[x_.j > 0])
    This is simpler, has a name that matches what it does, and is easy to
    audit column-by-column.
  - Raw (pre-cap, pre-scale) and normalized (post cap+scale+log1p) coverage
    are now BOTH saved as their own .npy files, separate from the final
    concatenated feature array — so a capped/scaled value can always be
    traced back to what it actually was.
  - FAISS is no longer a hard dependency: if unavailable, this module falls
    back to scipy's cKDTree (exact k-NN), and if scipy is also unavailable,
    to a numpy brute-force k-NN (capped at brute_force_max_n contigs,
    since brute force is O(n^2)). Which backend ran is recorded in the
    manifest either way.
  - Thread count is now clamped to os.cpu_count() before being handed to
    FAISS/scipy — requesting more workers than physical CPUs oversubscribes
    the machine rather than making anything faster.
  - Every column's INTENDED USE is now spelled out in the manifest
    (column_roles): valid_mask is a MASK, never a feature; n_samples_present
    is QC/optional-feature; distance + coverage columns are features. This
    file still only ever emits the same fixed layout — which columns a
    consumer actually uses is entirely that consumer's decision, informed
    by manifest.json, not something this module hardcodes an opinion about.
  - Per-sample capped-value counts are now in the manifest, not just a log
    line.

Everything from v8 (see v8 changelog, condensed below) is kept:
  - assembly_fasta establishes authoritative contig order; full ID
    reconciliation both directions (missing-from-coverage vs
    missing-from-fasta), duplicate-ID fail-loud, duplicate-row warn.
  - k clamped to n_contigs-1; n_contigs<=1 skips neighbor search entirely
    instead of crashing.
  - NaN/Inf/negative/duplicate/non-numeric coverage values fail loud by
    default (strict_numeric_columns=True).
  - valid_mask marks placeholder rows (filtered by prevalence, or never
    had a coverage row at all) so they're never mistaken for real all-zero
    coverage.
  - Every step's checkpoint is fingerprinted against its actual inputs and
    config, and validated for referenced-output existence, before being
    trusted.
  - Checkpoint(outdir) — not Checkpoint(outdir / "checkpoints"), which
    double-nests since Checkpoint appends its own "checkpoints" subdir.

Summary of steps:
  Step 1  : Load + validate coverage TSV
  Step 1b : ID reconciliation against assembly_fasta (authoritative order)
  Step 2  : Prevalence filter (drop all-zero-across-samples contigs)
  Step 3  : Outlier cap -> per-sample median-positive scale -> log1p
  Step 4  : k-NN (FAISS HNSW, else scipy cKDTree, else numpy brute-force)
  Step 5  : Neighbor distance summary [mean_dist, std_dist]
  Step 6  : Concatenate -> [valid_mask | n_samples_present | mean_dist |
            std_dist | cov_1..cov_N]  ->  N+4D, always
  Step 7  : Re-align to the full (authoritative) contig list
  (final) : Write manifest.json — the single source of truth for column
            layout/roles, contig order, parameters, per-sample scales and
            cap counts, k-NN backend used, and software versions. A
            consumer (encoder, clustering, anything else) should read
            THIS to learn the contract, not hardcode column positions —
            column/layout selection is deliberately kept OUTSIDE this
            module.

OUTPUT LAYOUT (unchanged from v8):
  [valid_mask | n_samples_present | mean_dist | std_dist | cov_1..cov_N]
  1 sample  ->  5D
  5 samples ->  9D
  8 samples -> 12D
  N samples ->  N+4D  (always, no special cases)

  valid_mask:        MASK ONLY. 1.0 = real computed features; 0.0 =
                      placeholder row (filtered by prevalence, or never
                      had a coverage row at all — see manifest's
                      reconciliation section to tell which). Never treat
                      this as a feature value to train on.
  n_samples_present: count of samples with RAW coverage > 0 for this
                      contig. QC/filtering signal by default; usable as
                      an input feature if a consumer chooses to.
"""

import gzip
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import faiss
    _HAVE_FAISS = True
except ImportError:
    faiss = None
    _HAVE_FAISS = False

try:
    from scipy.spatial import cKDTree
    import scipy
    _HAVE_SCIPY = True
except ImportError:
    cKDTree = None
    scipy = None
    _HAVE_SCIPY = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_step_start, log_step_done
from utils.checkpoint import Checkpoint

log = get_logger("coverage")


# =============================================================================
# CHECKPOINT FINGERPRINTING — a stale or config-mismatched checkpoint is a
# cache MISS, not a silent wrong answer (same pattern as preprocessing.py).
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


def _checkpoint_ok(ckpt: Checkpoint, step: str, fp: str, output_paths: List[str]) -> Optional[Dict]:
    if not ckpt.is_done(step):
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
# LIGHTWEIGHT FASTA ID READER
# =============================================================================

def _read_fasta_ids(fasta_path) -> List[str]:
    fasta_path = Path(fasta_path)
    opener = gzip.open if fasta_path.name.endswith(".gz") else open
    mode = 'rt' if fasta_path.name.endswith(".gz") else 'r'
    ids = []
    with opener(fasta_path, mode) as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


# =============================================================================
# THREAD RESOLUTION — never hand FAISS/scipy more workers than exist
# =============================================================================

def _resolve_threads(n_cores: int) -> Tuple[int, int]:
    """Clamp a requested worker count to the machine's actual CPU count.
    Oversubscribing threads (requesting 80 workers on an 8-core box)
    doesn't make FAISS/scipy faster — it adds scheduling overhead for no
    benefit, and on a shared machine actively hurts other jobs."""
    avail = os.cpu_count() or int(n_cores)
    eff = max(1, min(int(n_cores), avail))
    return eff, avail


# =============================================================================
# STEP 1 — LOAD + VALIDATE
# =============================================================================

def step1_load_and_validate(coverage_tsv: str, expected_samples: Optional[List[str]],
                             strict_numeric_columns: bool, outdir: Path) -> Tuple[pd.DataFrame, Dict]:
    log.info("Loading coverage matrix...")
    cov_df = pd.read_csv(coverage_tsv, sep='\t', index_col=0)
    cov_df.columns = [c.replace(' ', '_') for c in cov_df.columns]

    report: Dict = {}

    dup_cols = cov_df.columns[cov_df.columns.duplicated()].unique().tolist()
    if dup_cols:
        raise ValueError(f"Coverage TSV has duplicate sample column name(s): {dup_cols} — "
                          f"cannot safely tell these samples apart. Fix the input TSV's header.")

    if expected_samples is not None:
        got, want = set(cov_df.columns), set(expected_samples)
        if got != want:
            raise ValueError(f"Coverage TSV columns don't match expected_samples. "
                              f"Missing from TSV: {sorted(want - got)}. "
                              f"Unexpected in TSV: {sorted(got - want)}.")

    bad_cols = {}
    for c in cov_df.columns:
        coerced = pd.to_numeric(cov_df[c], errors='coerce')
        if coerced.isna().any() and not cov_df[c].isna().any():
            bad_vals = cov_df[c][coerced.isna() & cov_df[c].notna()].unique()[:5]
            bad_cols[c] = list(map(str, bad_vals))
    if bad_cols:
        msg = (f"{len(bad_cols)} coverage column(s) contain non-numeric values: "
               f"{ {k: v for k, v in list(bad_cols.items())[:5]} } — refusing to guess whether "
               f"these are malformed sample columns or something else entirely.")
        report["non_numeric_columns"] = bad_cols
        if strict_numeric_columns:
            raise ValueError(msg + " Set strict_numeric_columns=False to drop them instead "
                                    "(logged loudly, not silently).")
        log.error(msg + " strict_numeric_columns=False — DROPPING these columns.")
        (outdir / "dropped_non_numeric_columns.json").write_text(json.dumps(bad_cols, indent=2))
        cov_df = cov_df.drop(columns=list(bad_cols.keys()))

    cov_df = cov_df.astype(np.float64)

    dup_ids = cov_df.index[cov_df.index.duplicated()].unique().tolist()
    if dup_ids:
        raise ValueError(f"Coverage TSV has {len(dup_ids)} duplicate contig ID(s) "
                          f"(e.g. {dup_ids[:5]}) — cannot safely align features to contigs.")

    dup_row_mask = cov_df.duplicated(keep=False)
    n_dup_rows = int(dup_row_mask.sum())
    if n_dup_rows > 0:
        log.warning(f"⚠️  {n_dup_rows} contig(s) share byte-identical coverage vectors with at "
                    f"least one other contig — not fatal, but worth checking for an upstream "
                    f"merge/mapping bug. See duplicate_coverage_rows.txt.")
        (outdir / "duplicate_coverage_rows.txt").write_text(
            "\n".join(sorted(cov_df.index[dup_row_mask].astype(str))) + "\n")
    report["n_duplicate_rows"] = n_dup_rows

    values = cov_df.values
    if np.isnan(values).any():
        raise ValueError(f"Coverage matrix contains {int(np.isnan(values).sum())} NaN value(s) — "
                          f"a broken upstream merge/coverage table, not something safe to fill in.")
    if np.isinf(values).any():
        raise ValueError(f"Coverage matrix contains {int(np.isinf(values).sum())} Inf value(s).")
    if (values < 0).any():
        raise ValueError(f"Coverage matrix contains {int((values < 0).sum())} negative value(s) — "
                          f"coverage depth cannot be negative.")

    log.info(f"  Contigs (raw rows) : {len(cov_df):,}")
    log.info(f"  Samples            : {cov_df.shape[1]}")
    log.info(f"  Columns            : {cov_df.columns.tolist()}")
    return cov_df, report


# =============================================================================
# STEP 1b — ID RECONCILIATION AGAINST assembly_fasta
# =============================================================================

def step1b_reconcile_ids(cov_df: pd.DataFrame, assembly_fasta: Optional[str],
                          outdir: Path) -> Tuple[List[str], List[str], Dict]:
    coverage_ids = cov_df.index.tolist()
    coverage_id_set = set(coverage_ids)
    report: Dict = {}

    if assembly_fasta:
        full_ids = _read_fasta_ids(assembly_fasta)
        full_id_set = set(full_ids)
        if len(full_id_set) != len(full_ids):
            dupes = pd.Series(full_ids)
            dupes = dupes[dupes.duplicated()].unique().tolist()
            raise ValueError(f"assembly_fasta has {len(dupes)} duplicate contig ID(s) "
                              f"(e.g. {dupes[:5]}) — cannot establish an unambiguous contig order.")

        missing_from_coverage = sorted(full_id_set - coverage_id_set)
        missing_from_fasta = sorted(coverage_id_set - full_id_set)
        working_ids = [i for i in full_ids if i in coverage_id_set]

        if missing_from_coverage:
            log.warning(f"⚠️  {len(missing_from_coverage):,} FASTA contig(s) have NO coverage "
                        f"row at all — will be output as valid_mask=0 placeholders. See "
                        f"id_missing_from_coverage.txt.")
            (outdir / "id_missing_from_coverage.txt").write_text(
                "\n".join(missing_from_coverage) + "\n")
        if missing_from_fasta:
            log.warning(f"⚠️  {len(missing_from_fasta):,} coverage row(s) have an ID not present "
                        f"in assembly_fasta — DROPPING them. See id_missing_from_fasta.txt.")
            (outdir / "id_missing_from_fasta.txt").write_text(
                "\n".join(missing_from_fasta) + "\n")

        report.update(fasta_provided=True, n_fasta_ids=len(full_ids),
                       n_missing_from_coverage=len(missing_from_coverage),
                       n_missing_from_fasta=len(missing_from_fasta))
        log.info(f"  Authoritative order: assembly_fasta ({len(full_ids):,} contigs)")
        return full_ids, working_ids, report

    log.warning("⚠️  No assembly_fasta given — using the coverage TSV's own row order as "
                "authoritative, with no independent check against TNF/TE contig order/IDs.")
    report.update(fasta_provided=False, n_fasta_ids=None,
                   n_missing_from_coverage=0, n_missing_from_fasta=0)
    return coverage_ids, coverage_ids, report


# =============================================================================
# STEP 2 — PREVALENCE FILTER
# =============================================================================

def step2_prevalence_filter(cov_matrix: np.ndarray, ckpt: Checkpoint, fp0: str) -> Tuple[np.ndarray, np.ndarray]:
    STEP = "step2_prevalence"
    npy_dir = Path(ckpt.ckpt_dir) / "arrays"
    npy_dir.mkdir(parents=True, exist_ok=True)
    filt_path = npy_dir / "step2_filtered_matrix.npy"
    idx_path = npy_dir / "step2_kept_indices.npy"
    fp = _fingerprint(fp0, "step2")

    cached = _checkpoint_ok(ckpt, STEP, fp, [str(filt_path), str(idx_path)])
    if cached is not None:
        log.info("Step 2 [CACHED]")
        return np.load(filt_path), np.load(idx_path)

    t0 = time.time()
    log_step_start(log, 2, "Prevalence Filter", 8)

    max_cov = cov_matrix.max(axis=1) if cov_matrix.shape[0] > 0 else np.array([])
    kept_mask = max_cov > 0
    filtered = cov_matrix[kept_mask]
    kept_indices = np.where(kept_mask)[0]

    before, after = len(cov_matrix), len(filtered)
    removed = before - after
    log.info(f"  Before : {before:,}")
    log.info(f"  After  : {after:,}")
    if before > 0:
        log.info(f"  Removed: {removed:,} ({removed/before*100:.1f}% all-zero)")

    if after == 0:
        raise RuntimeError("All contigs (with a coverage row) have zero coverage — "
                            "check mapping output.")

    np.save(filt_path, filtered)
    np.save(idx_path, kept_indices)
    ckpt.mark_done(STEP, {"_fp": fp, "n_before": before, "n_after": after,
                           "filtered_path": str(filt_path), "kept_indices_path": str(idx_path)})
    log_step_done(log, 2, "Prevalence Filter", time.time() - t0)
    return filtered, kept_indices


# =============================================================================
# STEP 3 — OUTLIER CAP -> PER-SAMPLE MEDIAN-POSITIVE SCALE -> log(x+1)
# =============================================================================

def _median_positive_scale(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample scaling: divide by that sample's median POSITIVE value
    (computed on the already-capped matrix, since capping happens first).
    z_ij = x_ij / s_j,  s_j = median(x_.j[x_.j > 0]).

    A sample column with NO positive values at all (every contig has zero
    coverage in that one sample — a legitimate if unusual case, e.g. a
    failed mapping run for just that sample) has no median to compute;
    s_j is left at 1.0 and that column stays all-zero, which is the
    correct (if uninformative) answer rather than a divide-by-zero crash."""
    scales = np.ones(matrix.shape[1], dtype=np.float64)
    scaled = np.zeros_like(matrix, dtype=np.float32)
    for j in range(matrix.shape[1]):
        col = matrix[:, j].astype(np.float64)
        positive = col[col > 0]
        if len(positive) == 0:
            log.warning(f"  Sample column {j+1}: no positive coverage values at all — "
                        f"scale left at 1.0, column stays all-zero.")
            scaled[:, j] = 0.0
            continue
        s = float(np.median(positive))
        scales[j] = s if s > 0 else 1.0
        scaled[:, j] = (col / scales[j]).astype(np.float32)
    return scaled, scales


def step3_normalize(cov_matrix: np.ndarray, ckpt: Checkpoint, fp0: str,
                     outlier_cap_percentile: Optional[float]) -> Tuple[np.ndarray, Dict]:
    STEP = "step3_normalize"
    npy_dir = Path(ckpt.ckpt_dir) / "arrays"
    npy_dir.mkdir(parents=True, exist_ok=True)
    norm_path = npy_dir / "step3_normalized_matrix.npy"
    fp = _fingerprint(fp0, "step3", outlier_cap_percentile)

    cached = _checkpoint_ok(ckpt, STEP, fp, [str(norm_path)])
    if cached is not None:
        log.info("Step 3 [CACHED]")
        return np.load(norm_path), cached

    t0 = time.time()
    label = (f"Outlier Cap (p{outlier_cap_percentile}) + median-positive scale + log(x+1)"
             if outlier_cap_percentile else
             "median-positive scale + log(x+1) [cap DISABLED]")
    log_step_start(log, 3, label, 8)

    n_contigs, n_samples = cov_matrix.shape
    n_capped_per_sample: Dict[str, int] = {}
    cap_value_per_sample: Dict[str, float] = {}
    if outlier_cap_percentile:
        capped = np.zeros_like(cov_matrix, dtype=np.float32)
        for j in range(n_samples):
            col = cov_matrix[:, j]
            nz = col[col > 0]
            p_cap = float(np.percentile(nz, outlier_cap_percentile)) if len(nz) > 0 else 0.0
            capped[:, j] = np.clip(col, 0, p_cap)
            n_capped = int((col > p_cap).sum())
            n_capped_per_sample[f"sample_{j+1}"] = n_capped
            cap_value_per_sample[f"sample_{j+1}"] = p_cap
            if n_capped > 0:
                log.info(f"  Sample {j+1}: capped {n_capped:,} outlier(s) at {p_cap:.2f}x "
                         f"(p{outlier_cap_percentile}) — this can also cap genuinely "
                         f"high-abundance biology (high-copy genomes, repeats, organellar "
                         f"sequence). Raw, uncapped values are preserved separately "
                         f"(raw_coverage.npy) for auditing.")
    else:
        capped = cov_matrix.astype(np.float32)
        log.info("  Outlier cap DISABLED (outlier_cap_percentile=None)")

    log.info(f"  After cap : [{capped.min():.2f}, {capped.max():.2f}]")
    scaled, scales = _median_positive_scale(capped)
    log.info(f"  Per-sample scale (median positive, post-cap): "
             f"{[round(float(s), 3) for s in scales]}")
    log.info(f"  After scale : [{scaled.min():.4f}, {scaled.max():.4f}]")
    normalized = np.log1p(scaled)
    log.info(f"  After log1p : [{normalized.min():.4f}, {normalized.max():.4f}]"
             f"  std={normalized.std():.4f}")

    if np.isnan(normalized).any():
        log.warning("NaN detected post-normalization — replacing with 0")
        normalized = np.nan_to_num(normalized, nan=0.0)
    if np.isinf(normalized).any():
        log.warning("Inf detected post-normalization — replacing with 0")
        normalized = np.nan_to_num(normalized, posinf=0.0, neginf=0.0)

    np.save(norm_path, normalized)
    meta = {"_fp": fp, "normalized_path": str(norm_path),
            "outlier_cap_percentile": outlier_cap_percentile,
            "n_capped_per_sample": n_capped_per_sample,
            "cap_value_per_sample": cap_value_per_sample,
            "scales_per_sample": {f"sample_{j+1}": round(float(s), 6) for j, s in enumerate(scales)},
            "formula": "z_ij = log1p(x_ij_capped / s_j),  s_j = median(x_.j_capped[x_.j_capped > 0])"}
    ckpt.mark_done(STEP, meta)
    log_step_done(log, 3, "Normalization", time.time() - t0)
    return normalized, meta


# =============================================================================
# STEP 4 — k-NEAREST NEIGHBORS (FAISS -> scipy cKDTree -> numpy brute-force)
# =============================================================================

def _knn_faiss(mat: np.ndarray, k_effective: int, n_threads: int,
               deterministic: bool) -> Tuple[np.ndarray, np.ndarray, Dict]:
    faiss.omp_set_num_threads(1 if deterministic else n_threads)
    d = mat.shape[1]
    hnsw_params = {"M": 32, "efConstruction": 200, "efSearch": 128}
    index = faiss.IndexHNSWFlat(d, hnsw_params["M"], faiss.METRIC_L2)
    index.hnsw.efConstruction = hnsw_params["efConstruction"]
    index.hnsw.efSearch = hnsw_params["efSearch"]
    log.info("  Building HNSW index...")
    index.add(mat)
    log.info(f"  Searching {k_effective} neighbor(s)...")
    dists, indices = index.search(mat, k_effective + 1)
    dists = dists[:, 1:].astype(np.float32)
    indices = indices[:, 1:].astype(np.int64)
    dists = np.sqrt(np.maximum(dists, 0))
    return dists, indices, hnsw_params


def _knn_scipy(mat: np.ndarray, k_effective: int, n_threads: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    log.info("  Building cKDTree index (exact, deterministic)...")
    tree = cKDTree(mat)
    log.info(f"  Querying {k_effective} neighbor(s)...")
    try:
        dists, indices = tree.query(mat, k=k_effective + 1, workers=n_threads)
    except TypeError:
        # older scipy without the `workers` kwarg
        dists, indices = tree.query(mat, k=k_effective + 1)
    dists = np.atleast_2d(dists)
    indices = np.atleast_2d(indices)
    dists = dists[:, 1:].astype(np.float32)
    indices = indices[:, 1:].astype(np.int64)
    return dists, indices, {}


def _knn_bruteforce(mat: np.ndarray, k_effective: int, max_n: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n = mat.shape[0]
    if n > max_n:
        raise RuntimeError(f"Neither FAISS nor scipy is available, and exact brute-force k-NN "
                            f"on {n:,} contigs exceeds the safety cap (brute_force_max_n="
                            f"{max_n:,}, O(n^2) cost) — install faiss-cpu or scipy to run at "
                            f"this scale, or raise brute_force_max_n if you accept the cost.")
    log.warning(f"⚠️  Neither FAISS nor scipy available — using an O(n^2) numpy brute-force "
                f"k-NN fallback ({n:,} contigs, chunked to bound memory). This is exact but "
                f"will not scale past a few thousand contigs.")
    chunk = 2000
    idx_sorted = np.zeros((n, k_effective), dtype=np.int64)
    dist_sorted = np.zeros((n, k_effective), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = mat[start:end, None, :] - mat[None, :, :]
        d_chunk = np.sqrt((diff ** 2).sum(axis=2))
        order = np.argsort(d_chunk, axis=1)[:, 1:k_effective + 1]  # skip self at position 0
        dist_sorted[start:end] = np.take_along_axis(d_chunk, order, axis=1)
        idx_sorted[start:end] = order
    return dist_sorted, idx_sorted, {}


def step4_neighbors(normalized_matrix: np.ndarray, k: int, n_cores: int, deterministic: bool,
                     brute_force_max_n: int, ckpt: Checkpoint, fp0: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Tries FAISS HNSW first (fast, approximate, scales to millions of
    contigs); falls back to scipy's cKDTree (exact, deterministic, scales
    to maybe hundreds of thousands); falls back to a numpy brute-force
    O(n^2) k-NN capped at brute_force_max_n contigs as a last resort so a
    machine without either dependency can still run small/test datasets.
    k is clamped to n_contigs-1 either way; n_contigs<=1 skips neighbor
    search entirely (no valid neighbors exist for a single contig)."""
    STEP = "step4_neighbors"
    npy_dir = Path(ckpt.ckpt_dir) / "arrays"
    npy_dir.mkdir(parents=True, exist_ok=True)
    dist_path = npy_dir / "step4_distances.npy"
    nbr_path = npy_dir / "step4_neighbor_indices.npy"

    n_contigs, d = normalized_matrix.shape
    k_requested = k
    k_effective = max(0, min(k, n_contigs - 1))
    n_threads, n_cpus_avail = _resolve_threads(n_cores)
    backend = "faiss" if _HAVE_FAISS else ("scipy_ckdtree" if _HAVE_SCIPY else "numpy_bruteforce")
    faiss_version = getattr(faiss, "__version__", None) if _HAVE_FAISS else None
    scipy_version = getattr(scipy, "__version__", None) if _HAVE_SCIPY else None

    fp = _fingerprint(fp0, "step4", k_effective, deterministic, backend, faiss_version, scipy_version)
    cached = _checkpoint_ok(ckpt, STEP, fp, [str(dist_path), str(nbr_path)])
    if cached is not None:
        log.info("Step 4 [CACHED]")
        return np.load(dist_path), np.load(nbr_path), cached

    t0 = time.time()
    log_step_start(log, 4, f"k-NN [{backend}] (k_requested={k_requested}, "
                            f"k_effective={k_effective}, d={d}, n={n_contigs:,})", 8)

    if n_threads < n_cores:
        log.warning(f"⚠️  Requested n_cores={n_cores} exceeds available CPUs ({n_cpus_avail}) — "
                    f"using {n_threads} thread(s) to avoid oversubscription.")
    if not _HAVE_FAISS:
        log.warning(f"⚠️  FAISS not installed — falling back to "
                    f"{'scipy cKDTree' if _HAVE_SCIPY else 'a numpy brute-force fallback'}.")
    if k_effective < k_requested:
        log.warning(f"⚠️  Requested k={k_requested} but only {n_contigs:,} contig(s) are "
                    f"available (max possible neighbors = n-1 = {max(0, n_contigs-1)}) — "
                    f"clamped to k={k_effective}.")

    if n_contigs <= 1 or k_effective == 0:
        log.warning(f"⚠️  {n_contigs} contig(s) after prevalence filtering — no valid "
                    f"neighbors exist. Skipping k-NN entirely; distance features will be 0 "
                    f"('no neighbors available', not a real similarity signal).")
        dists = np.zeros((n_contigs, 0), dtype=np.float32)
        indices = np.zeros((n_contigs, 0), dtype=np.int64)
        meta = {"_fp": fp, "dist_path": str(dist_path), "nbr_path": str(nbr_path),
                "k_requested": k_requested, "k_effective": 0, "d": d, "backend": "none",
                "faiss_available": _HAVE_FAISS, "scipy_available": _HAVE_SCIPY,
                "faiss_version": faiss_version, "scipy_version": scipy_version,
                "hnsw_params": {}, "skipped_no_neighbors": True}
        np.save(dist_path, dists)
        np.save(nbr_path, indices)
        ckpt.mark_done(STEP, meta)
        log_step_done(log, 4, "k-NN (skipped, no neighbors possible)", time.time() - t0)
        return dists, indices, meta

    mat = normalized_matrix.astype(np.float32)
    if backend == "faiss":
        dists, indices, backend_params = _knn_faiss(mat, k_effective, n_threads, deterministic)
    elif backend == "scipy_ckdtree":
        dists, indices, backend_params = _knn_scipy(mat, k_effective, n_threads)
    else:
        dists, indices, backend_params = _knn_bruteforce(mat, k_effective, brute_force_max_n)

    data_range = float(mat.max() - mat.min())
    max_valid = data_range * np.sqrt(mat.shape[1])
    if max_valid > 0:
        n_garbage = int((dists > max_valid).sum())
        if n_garbage > 0:
            log.info(f"  Clipping {n_garbage:,} garbage distance(s) at {max_valid:.4f}")
            dists = np.clip(dists, 0, max_valid)
    else:
        log.warning("⚠️  Normalized coverage matrix has ZERO variance (every value identical) "
                    "— a degenerate input. Distances left unclipped rather than force-zeroed.")

    log.info(f"  L2 range: [{dists.min():.4f}, {dists.max():.4f}]  mean={dists.mean():.4f}")

    np.save(dist_path, dists)
    np.save(nbr_path, indices)
    meta = {"_fp": fp, "dist_path": str(dist_path), "nbr_path": str(nbr_path),
            "k_requested": k_requested, "k_effective": k_effective, "d": d, "backend": backend,
            "faiss_available": _HAVE_FAISS, "scipy_available": _HAVE_SCIPY,
            "faiss_version": faiss_version, "scipy_version": scipy_version,
            "n_cores_requested": n_cores, "n_threads_used": n_threads,
            "n_cpus_available": n_cpus_avail, "deterministic": deterministic,
            "hnsw_params": backend_params, "skipped_no_neighbors": False}
    ckpt.mark_done(STEP, meta)
    log_step_done(log, 4, f"k-NN [{backend}]", time.time() - t0)
    return dists, indices, meta


# =============================================================================
# STEP 5 — DISTANCE FEATURES
# =============================================================================

def step5_distance_features(distances: np.ndarray, ckpt: Checkpoint, fp0: str) -> np.ndarray:
    STEP = "step5_distance"
    npy_dir = Path(ckpt.ckpt_dir) / "arrays"
    npy_dir.mkdir(parents=True, exist_ok=True)
    path = npy_dir / "step5_distance_features.npy"
    fp = _fingerprint(fp0, "step5", distances.shape)

    cached = _checkpoint_ok(ckpt, STEP, fp, [str(path)])
    if cached is not None:
        log.info("Step 5 [CACHED]")
        return np.load(path)

    t0 = time.time()
    log_step_start(log, 5, "Distance Features [mean_dist, std_dist]", 8)

    n = distances.shape[0]
    if distances.shape[1] == 0:
        features = np.zeros((n, 2), dtype=np.float32)
    else:
        finite_mask = np.isfinite(distances)
        if not finite_mask.all():
            max_val = float(distances[finite_mask].max()) if finite_mask.any() else 1.0
            distances = np.where(finite_mask, distances, max_val)

        valid_for_cap = distances[distances < 1e10]
        if valid_for_cap.size == 0:
            log.warning("⚠️  Every neighbor distance is >= 1e10 (all garbage) — falling back "
                        "to a cap of 0.0 rather than crashing on an empty array.")
            cap = 0.0
        else:
            cap = valid_for_cap.max()
        distances = np.clip(distances, 0, cap)

        mean_dist = distances.mean(axis=1, keepdims=True)
        std_dist = distances.std(axis=1, keepdims=True)
        features = np.concatenate([mean_dist, std_dist], axis=1).astype(np.float32)

    log.info(f"  mean_dist: [{features[:,0].min():.4f}, {features[:,0].max():.4f}]  "
             f"std={features[:,0].std():.4f}")
    log.info(f"  std_dist : [{features[:,1].min():.4f}, {features[:,1].max():.4f}]  "
             f"std={features[:,1].std():.4f}")

    np.save(path, features)
    ckpt.mark_done(STEP, {"_fp": fp, "dist_features_path": str(path)})
    log_step_done(log, 5, "Distance Features", time.time() - t0)
    return features


# =============================================================================
# STEP 6 — CONCATENATE -> [valid_mask | n_samples_present | mean_dist |
#                           std_dist | cov_1..cov_N]  N+4D always
# =============================================================================

COLUMN_LAYOUT_METADATA_NAMES = ["valid_mask", "n_samples_present", "mean_dist", "std_dist"]
COLUMN_ROLES = {
    "valid_mask": "MASK ONLY — 1.0 real row / 0.0 placeholder. Never train on this as a feature; "
                  "use it to exclude or down-weight placeholder rows.",
    "n_samples_present": "QC/filtering signal by default; MAY be used as an input feature at a "
                          "consumer's discretion.",
    "mean_dist": "feature (coverage-neighborhood mean distance)",
    "std_dist": "feature (coverage-neighborhood distance spread)",
    "cov_*": "feature — log1p(capped_coverage / per-sample median-positive scale)",
}


def step6_concatenate(normalized_matrix: np.ndarray, dist_features: np.ndarray,
                       n_samples_present: np.ndarray, n_samples: int,
                       ckpt: Checkpoint, fp0: str) -> np.ndarray:
    STEP = "step6_concatenate"
    npy_dir = Path(ckpt.ckpt_dir) / "arrays"
    npy_dir.mkdir(parents=True, exist_ok=True)
    path = npy_dir / "step6_coverage_features.npy"
    fp = _fingerprint(fp0, "step6", n_samples)

    cached = _checkpoint_ok(ckpt, STEP, fp, [str(path)])
    if cached is not None:
        log.info("Step 6 [CACHED]")
        return np.load(path)

    t0 = time.time()
    log_step_start(log, 6, "Concatenate Coverage Features", 8)

    n = normalized_matrix.shape[0]
    valid_mask = np.ones((n, 1), dtype=np.float32)
    n_samples_present_col = n_samples_present.reshape(-1, 1).astype(np.float32)

    coverage_features = np.concatenate([
        valid_mask, n_samples_present_col, dist_features, normalized_matrix,
    ], axis=1).astype(np.float32)

    layout = f"[{' | '.join(COLUMN_LAYOUT_METADATA_NAMES)} | cov_1..cov_{n_samples}]  →  {n_samples+4}D"
    log.info(f"  Coverage : {normalized_matrix.shape}")
    log.info(f"  Distance : {dist_features.shape}")
    log.info(f"  OUTPUT   : {coverage_features.shape}")
    log.info(f"  Layout   : {layout}")

    np.save(path, coverage_features)
    ckpt.mark_done(STEP, {"_fp": fp, "coverage_features_path": str(path),
                          "shape": list(coverage_features.shape), "n_samples": n_samples,
                          "layout": layout})
    log_step_done(log, 6, "Concatenate", time.time() - t0)
    return coverage_features


# =============================================================================
# STEP 7 — RE-ALIGN TO FULL CONTIG LIST
# =============================================================================

def step7_realign(coverage_features: np.ndarray, full_positions_of_kept: np.ndarray,
                   n_total: int, n_dims: int, ckpt: Checkpoint, fp0: str) -> np.ndarray:
    STEP = "step7_realign"
    npy_dir = Path(ckpt.ckpt_dir) / "arrays"
    npy_dir.mkdir(parents=True, exist_ok=True)
    path = npy_dir / "step7_coverage_features_realigned.npy"
    fp = _fingerprint(fp0, "step7", n_total, n_dims, full_positions_of_kept.tolist())

    cached = _checkpoint_ok(ckpt, STEP, fp, [str(path)])
    if cached is not None:
        log.info("Step 7 [CACHED]")
        return np.load(path)

    t0 = time.time()
    log_step_start(log, 7, "Re-align to full contig list", 8)

    n_kept = len(full_positions_of_kept)
    n_placeholder = n_total - n_kept

    if n_placeholder == 0:
        log.info("  Every contig has real computed features — no placeholders needed")
        realigned = coverage_features
    else:
        log.info(f"  Inserting valid_mask=0 placeholder rows for {n_placeholder:,} contig(s)")
        realigned = np.zeros((n_total, n_dims), dtype=np.float32)
        realigned[full_positions_of_kept] = coverage_features

    log.info(f"  Output shape   : {realigned.shape}")
    log.info(f"  Placeholder rows (valid_mask=0): {(realigned[:, 0] == 0).sum():,} / {n_total:,}")

    np.save(path, realigned)
    ckpt.mark_done(STEP, {"_fp": fp, "realigned_path": str(path),
                          "shape": list(realigned.shape), "n_placeholder": n_placeholder})
    log_step_done(log, 7, "Re-align", time.time() - t0)
    return realigned


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_coverage_features(
    coverage_tsv: str,
    assembly_fasta: Optional[str] = None,
    outdir: str = "coverage_output",
    k: int = 200,
    n_cores: int = 80,
    resume: bool = True,
    expected_samples: Optional[List[str]] = None,
    strict_numeric_columns: bool = True,
    outlier_cap_percentile: Optional[float] = 99.5,
    deterministic: bool = False,
    brute_force_max_n: int = 5000,
) -> np.ndarray:
    """
    HyphaeS coverage feature pipeline v9.

    Output always N+4D — see COLUMN_LAYOUT_METADATA_NAMES / COLUMN_ROLES /
    manifest.json for the exact column order and each column's intended
    use: [valid_mask | n_samples_present | mean_dist | std_dist |
    cov_1..cov_N]. Column/layout SELECTION belongs to the consumer
    (encoder, clustering, etc.) reading manifest.json — this module always
    emits the same fixed layout and never special-cases for a downstream
    consumer.

    Args:
        coverage_tsv           : TSV rows=contigs, cols=sample depths
        assembly_fasta         : authoritative contig order + ID
                                  reconciliation (strongly recommended)
        outdir                 : output directory
        k                      : k-NN neighbor count (clamped to n_contigs-1)
        n_cores                : requested worker threads (clamped to
                                  os.cpu_count() to avoid oversubscription)
        resume                 : checkpoint resume
        expected_samples       : optional exact-match check against the
                                  coverage TSV's columns
        strict_numeric_columns : True (default) = fail loud on any
                                  non-numeric column; False = drop them
                                  loudly with a written report
        outlier_cap_percentile : percentile to cap outliers at BEFORE
                                  scaling (default 99.5; use 99.9 for very
                                  abundant real genomes where even 99.5
                                  clips real signal). None disables capping.
        deterministic           : force single-threaded FAISS index build
                                  for more (not perfectly guaranteed)
                                  reproducibility
        brute_force_max_n      : safety cap on contig count for the numpy
                                  brute-force k-NN fallback (only used if
                                  neither FAISS nor scipy is installed)

    Returns:
        np.ndarray shape [n_total_contigs, n_samples+4]
    """
    t0_total = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not resume:
        stale_ckpt_dir = outdir / "checkpoints"
        if stale_ckpt_dir.exists():
            log.info(f"resume=False — clearing existing checkpoint state at {stale_ckpt_dir} "
                     f"(every step will recompute from scratch).")
            shutil.rmtree(stale_ckpt_dir)

    ckpt = Checkpoint(outdir)  # Checkpoint() appends its own "checkpoints" subdir internally;
                                # constructed AFTER the resume=False wipe above, so a fresh run
                                # never loads stale in-memory state from a deleted checkpoint dir.

    log.info("")
    log.info("╔" + "═"*68 + "╗")
    log.info("║" + " "*8 + "HYPHAES COVERAGE MODULE — v9 (N+4D, self-describing)" + " "*7 + "║")
    log.info("║" + " "*4 + "cap→median-scale→log1p | k-NN (FAISS/scipy/bruteforce)" + " "*9 + "║")
    log.info("╚" + "═"*68 + "╝")
    log.info("")

    cov_df, load_report = step1_load_and_validate(coverage_tsv, expected_samples,
                                                    strict_numeric_columns, outdir)
    full_ids, working_ids, reconcile_report = step1b_reconcile_ids(cov_df, assembly_fasta, outdir)

    n_total = len(full_ids)
    n_samples = cov_df.shape[1]
    out_dims = n_samples + 4

    full_ids_path = outdir / "contig_ids.txt"
    full_ids_path.write_text("\n".join(full_ids) + ("\n" if full_ids else ""))
    log.info(f"  Saved full contig_ids.txt: {n_total:,} contigs (authoritative order)")
    log.info(f"  Output   : {out_dims}D  [{' | '.join(COLUMN_LAYOUT_METADATA_NAMES)} | "
             f"cov_1..{n_samples}]")

    fp0 = _fingerprint(
        _file_fingerprint(coverage_tsv), _file_fingerprint(assembly_fasta),
        sorted(cov_df.columns.tolist()), expected_samples, strict_numeric_columns,
        outlier_cap_percentile, k, deterministic, brute_force_max_n,
    )

    cov_matrix_working = cov_df.loc[working_ids].values.astype(np.float32) if working_ids else \
        np.zeros((0, n_samples), dtype=np.float32)
    n_samples_present_working = (cov_matrix_working > 0).sum(axis=1)

    full_index_of = {cid: i for i, cid in enumerate(full_ids)}
    working_positions_in_full = np.array([full_index_of[i] for i in working_ids], dtype=np.int64)

    filtered, kept_indices_in_working = step2_prevalence_filter(cov_matrix_working, ckpt, fp0)
    n_samples_present_kept = n_samples_present_working[kept_indices_in_working] if len(working_ids) else \
        np.zeros(0)
    kept_ids = [working_ids[i] for i in kept_indices_in_working]

    # Save RAW (pre-cap, pre-scale) coverage separately, for auditing —
    # exactly what was fed into normalization, before anything altered it.
    raw_path = outdir / "raw_coverage.npy"
    np.save(raw_path, filtered)
    (outdir / "raw_coverage_contig_ids.txt").write_text(
        "\n".join(kept_ids) + ("\n" if kept_ids else ""))

    normalized, norm_meta = step3_normalize(filtered, ckpt, fp0, outlier_cap_percentile)

    # Save NORMALIZED (post cap+scale+log1p) coverage separately too — same
    # row order/IDs as raw_coverage.npy, so any row is directly comparable
    # raw-vs-normalized without touching the final concatenated array.
    normalized_path = outdir / "normalized_coverage.npy"
    np.save(normalized_path, normalized)

    dists, _, backend_meta = step4_neighbors(normalized, k, n_cores, deterministic,
                                              brute_force_max_n, ckpt, fp0)
    dist_features = step5_distance_features(dists, ckpt, fp0)
    coverage_features_kept = step6_concatenate(normalized, dist_features,
                                                n_samples_present_kept, n_samples, ckpt, fp0)

    kept_positions_in_full = working_positions_in_full[kept_indices_in_working] if len(working_ids) else \
        np.array([], dtype=np.int64)
    coverage_features = step7_realign(coverage_features_kept, kept_positions_in_full,
                                       n_total, out_dims, ckpt, fp0)

    out_features = outdir / "coverage_features.npy"
    np.save(out_features, coverage_features)

    manifest = {
        "module": "coverage.py", "version": "v9",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - t0_total, 1),
        "inputs": {
            "coverage_tsv": str(coverage_tsv), "coverage_tsv_fingerprint": _file_fingerprint(coverage_tsv),
            "assembly_fasta": str(assembly_fasta) if assembly_fasta else None,
            "assembly_fasta_fingerprint": _file_fingerprint(assembly_fasta) if assembly_fasta else None,
        },
        "sample_columns_in_order": cov_df.columns.tolist(),
        "n_samples": n_samples,
        "n_total_contigs": n_total,
        "n_working_contigs": len(working_ids),
        "n_kept_after_prevalence": len(kept_indices_in_working),
        "contig_id_order_file": str(full_ids_path),
        "contig_id_order_hash": hashlib.sha256("\n".join(full_ids).encode()).hexdigest()[:16],
        "output": {
            "path": str(out_features), "shape": list(coverage_features.shape),
            "column_layout": COLUMN_LAYOUT_METADATA_NAMES + [f"cov_{i+1}" for i in range(n_samples)],
            "column_roles": COLUMN_ROLES,
            "layout_string": (f"[{' | '.join(COLUMN_LAYOUT_METADATA_NAMES)} | "
                               f"cov_1..cov_{n_samples}]  ->  {out_dims}D"),
            "note": "Column/layout SELECTION is the consumer's decision, informed by this "
                    "manifest — this module never hardcodes a downstream consumer's contract. "
                    "valid_mask=0 rows are placeholders; see 'reconciliation' below for why.",
        },
        "raw_and_normalized_coverage": {
            "raw_coverage_path": str(raw_path), "normalized_coverage_path": str(normalized_path),
            "contig_ids_path": str(outdir / "raw_coverage_contig_ids.txt"),
            "note": "Same row order/IDs in both files — row i of raw_coverage.npy is the exact "
                    "pre-cap/pre-scale input to row i of normalized_coverage.npy. Neither file "
                    "includes placeholder rows (see n_kept_after_prevalence above).",
        },
        "normalization": {
            "formula": norm_meta.get("formula"),
            "outlier_cap_percentile": outlier_cap_percentile,
            "cap_value_per_sample": norm_meta.get("cap_value_per_sample"),
            "n_capped_per_sample": norm_meta.get("n_capped_per_sample"),
            "scales_per_sample": norm_meta.get("scales_per_sample"),
        },
        "reconciliation": {**load_report, **reconcile_report},
        "parameters": {
            "k_requested": k, "k_effective": backend_meta.get("k_effective"),
            "n_cores_requested": n_cores, "n_threads_used": backend_meta.get("n_threads_used"),
            "n_cpus_available": backend_meta.get("n_cpus_available"),
            "deterministic": deterministic, "strict_numeric_columns": strict_numeric_columns,
            "expected_samples": expected_samples, "brute_force_max_n": brute_force_max_n,
        },
        "knn_backend": {
            "backend_used": backend_meta.get("backend"),
            "faiss_available": backend_meta.get("faiss_available"),
            "scipy_available": backend_meta.get("scipy_available"),
            "faiss_version": backend_meta.get("faiss_version"),
            "scipy_version": backend_meta.get("scipy_version"),
            "hnsw_params": backend_meta.get("hnsw_params"),
            "note": "FAISS HNSW is fast but approximate and not bitwise-reproducible across "
                    "runs/thread-counts/versions. scipy cKDTree and the numpy brute-force "
                    "fallback are both EXACT and deterministic, just slower at scale.",
        },
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__,
                               "faiss": backend_meta.get("faiss_version"),
                               "scipy": backend_meta.get("scipy_version")},
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    t_total = time.time() - t0_total
    log.info("")
    log.info("=" * 70)
    log.info("COVERAGE FEATURES COMPLETE (v9)")
    log.info(f"  Time       : {t_total/60:.1f} min")
    log.info(f"  Contigs    : {n_total:,} (full) / {len(kept_indices_in_working):,} (valid)")
    log.info(f"  Shape      : {coverage_features.shape}")
    log.info(f"  Dims       : {out_dims}D  (4 metadata + {n_samples} coverage)")
    log.info(f"  Backend    : {backend_meta.get('backend')}")
    log.info(f"  Output     : {out_features}")
    log.info(f"  Manifest   : {manifest_path}")
    log.info("=" * 70)
    log.info("")

    return coverage_features


# =============================================================================
# CLI
# =============================================================================



# PROFILE HOOKS v1 — observation only; scientific functions above are unchanged.
from hyphaesbin.utils.resource_profiler import profile_functions, profile_checkpoint
profile_functions(globals(), ['run_coverage_features', 'step1_load_and_validate', 'step1b_reconcile_ids', 'step2_prevalence_filter', 'step3_normalize', 'step4_neighbors', 'step5_distance_features', 'step6_concatenate', 'step7_realign'])
_checkpoint_ok = profile_checkpoint(_checkpoint_ok)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="HyphaeS Coverage Features v9 (N+4D, self-describing via manifest.json, "
                    "FAISS optional — falls back to scipy/numpy)"
    )
    parser.add_argument("--coverage", required=True, help="Coverage TSV (contigs × samples)")
    parser.add_argument("--fasta", default=None,
                        help="Assembly FASTA — establishes authoritative contig order/ID "
                             "reconciliation. Strongly recommended.")
    parser.add_argument("--outdir", default="coverage_output")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--cores", type=int, default=80,
                        help="Requested worker threads (clamped to os.cpu_count())")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-strict-numeric", action="store_true",
                        help="Drop non-numeric columns (loudly) instead of failing the run")
    parser.add_argument("--no-outlier-cap", action="store_true",
                        help="Disable percentile outlier capping entirely")
    parser.add_argument("--outlier-cap-percentile", type=float, default=99.5,
                        help="Default 99.5; use 99.9 for very abundant real genomes")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force single-threaded FAISS index construction")
    parser.add_argument("--brute-force-max-n", type=int, default=5000,
                        help="Safety cap for the numpy brute-force k-NN fallback")
    args = parser.parse_args()

    features = run_coverage_features(
        coverage_tsv=args.coverage,
        assembly_fasta=args.fasta,
        outdir=args.outdir,
        k=args.k,
        n_cores=args.cores,
        resume=not args.no_resume,
        strict_numeric_columns=not args.no_strict_numeric,
        outlier_cap_percentile=(None if args.no_outlier_cap else args.outlier_cap_percentile),
        deterministic=args.deterministic,
        brute_force_max_n=args.brute_force_max_n,
    )

    n_s = features.shape[1] - 4
    print(f"\nSaved    : {args.outdir}/coverage_features.npy")
    print(f"Manifest : {args.outdir}/manifest.json  (read this for column layout AND roles)")
    print(f"Shape    : {features.shape}  →  {features.shape[1]}D")
    print(f"Layout   : [{' | '.join(COLUMN_LAYOUT_METADATA_NAMES)} | cov_1..cov_{n_s}]")
