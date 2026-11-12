"""
HyphaeSBin TE Composition Module — V33
=======================================
Author : HyphaeSBin Pipeline
Version: V33 (V22_5D — hardened)

OVERVIEW
--------
This module extracts transposable element (TE) composition features from
metagenomic assemblies to support fungal metagenome-assembled genome (MAG)
binning. TE repertoire is a species-discriminative signal orthogonal to
tetranucleotide frequency (TNF) and coverage — different fungal species
harbour distinct TE families, making TE composition a powerful third modality.

INPUT CONTRACT: pass the UNMASKED assembly (e.g. preprocessing.py's
step8_rdna_masking `unmasked_fasta`, NOT the rRNA-masked `final_clean.fasta`
used for TNF). preprocessing.py's own step8 metadata records
`"te_analysis": "unmasked"` — the opposite of TNF's `"tnf_composition":
"masked"`. This is deliberate: rRNA masking replaces rRNA loci with N's,
and MMseqs2 cannot align against N runs, so masking the input
would make any TE sequence that happens to overlap an rRNA locus
artificially undetectable. RNA-derived repeat hits (rRNA/snRNA/tRNA) are
already filtered out at the classification stage (SKIP_CLASSES) regardless
of whether the input was masked, so there is no need to pre-mask for TE
search — searching the unmasked sequence and filtering RNA-class hits
after the fact is strictly more correct.

PIPELINE STEPS
--------------
Step 1  MMseqs2 search
        Align assembled contigs against a TE consensus library (e.g.
        MycoMobilome) to detect TE hits (~20 min / 500k contigs).

Step 2  Parse hits -> per-contig TE summary
        For each contig, aggregate: total TE bp (raw sum, may double-count
        overlapping hits), merged (non-overlapping, interval-union) TE bp,
        LTR / DNA / LINE / Unclassified class bp, hit count, mean hit
        quality. RNA-derived repeats (rRNA, snRNA, tRNA) are excluded.

Step 3  Build V22_5D feature matrix (N x 5)
        Five biologically interpretable features per contig:
          d0  TE_richness        - fraction of TE classes active (0-1)
          d1  AbsTE_sqrt         - sqrt(merged_TE_bp / contig_length)
          d2  DNA_ratio          - DNA transposon fraction
          d3  LTR_ratio          - LTR retrotransposon fraction
          d4  Unclassified_ratio - unclassified TE fraction
        Transform: sqrt(feature), applied ONCE, uniformly, to all 5
        features (see V32 changelog — v31 applied sqrt twice to d1).
        Zero rows (~100% for fragmented real assemblies) are expected and
        handled by the encoder (TE weight = 0 for zero-TE contigs).

Step 4  Compute TE confidence weights (N x 1)
        Per-contig reliability score [0, 1] based on TE signal, contig
        length, hit count, AND mean hit quality. Used by the encoder to
        downweight contigs with no/unreliable TE signal. Guaranteed to be
        0 for exactly the same rows where the feature vector is all-zero.

Step 5  Save all outputs
        te_features.npy, te_weight.npy, contig_ids.json, schema.json,
        te_annotations.bed, mmseqs_te_hits.tsv.

SEARCH
------
MMseqs2 nucleotide search only (~20 min / 500k contigs at 64 threads). The former RepeatMasker
'efficient' mode was removed in HyphaeSBin v9 (te_mode / te_pa / te_frag / te_min_sw_score /
repeatmasker_quality_reference_score no longer exist; old keys in a config are ignored).

DATABASE COMPATIBILITY
----------------------
Works with ANY nucleotide FASTA database:
  - MycoMobilome v1.1 (recommended for fungi)
  - RepBase
  - Dfam
  - Any custom TE consensus library

V32 changelog (this round — see the numbered review this responds to)
-----------------------------------------------------------------------
 1. Contig ID/order validation: duplicate IDs in the assembly FASTA are
    now rejected (previously silently kept only the last occurrence via
    dict overwrite). Any TE hit whose contig ID is NOT in the assembly's
    contig list is now a hard error (previously silently dropped, since
    build_feature_matrix_v22_5d only ever iterated the assembly's own
    contig_ids — a hit landing on an unknown ID went nowhere with no
    warning, which is exactly the kind of silent ID-mismatch bug that
    corrupts alignment with other modalities). A contig-order hash is
    now recorded in schema.json for cross-module reconciliation with
    TNF/coverage, same convention as tnf_gene.py/coverage.py.
 2. Checkpoints are now fingerprinted against the scaffold, the TE
    database, every tool parameter that affects output, AND the search
    tool's own version string (best-effort `mmseqs version`) — previously a checkpoint was trusted purely on step-name
    presence, so re-running with a different database or a different
    min_seq_id silently reused stale results.
 3. Genuine no-hit vs tool failure: MMseqs2 always creates its output TSV
    on success (even when it's empty) — a MISSING file after a "successful"
    run is now treated as a failure, not silently touch()'d into a fake
    empty success as v31 did. The log is also scanned for OOM/segfault/killed markers
    even on a reported exit code of 0.
 4. TE length is now computed from MERGED (non-overlapping, interval-union)
    bp via merged_intervals_bp — which existed in v31 but was never
    actually called anywhere. v31's AbsTE_sqrt used the raw overlap-
    inflated sum, silently relying on a min(...,1.0) clip to hide totals
    that could exceed the contig length. merged_te_bp cannot exceed contig
    length by construction, so that clip is now a defensive no-op, not a
    load-bearing correction. The TE weight's "te_sig" term uses merged bp
    for the same reason.
 5. The one runtime `assert` (output shape check) is now a ValueError;
    shape/length self-checks were added for te_weight and contig_ids too.
 6. sqrt/length-weighting validated mathematically — v31 had a real bug:
    AbsTE_sqrt was sqrt'd once inside its own formula AND a second time by
    the uniform "sqrt(feature)" transform (net exponent 0.25, not 0.5, and
    inconsistent with the other four features which only got one sqrt).
    Fixed by storing the pre-sqrt ratio and applying the uniform sqrt
    exactly once to all 5 dims. Separately: v31 also multiplied every
    feature (including the three RATIO features, which are bounded
    fractions with no length dependence by definition) by
    sqrt(contig_length/max_length). For a proportion like DNA_ratio, that
    multiplication doesn't reduce noise — it fabricates a length
    dependency that isn't there, making a short TE-rich contig
    systematically look TE-poor purely because it's short. Confidence
    about short contigs already has a dedicated, correctly-scoped home:
    te_weight.npy. Length-multiplying the FEATURE values on top of that
    double-counts length confidence and actively corrupts the ratio
    features' meaning. Default behavior now leaves te_features.npy
    length-independent (a pure composition measurement) and puts all
    length confidence in te_weight.npy. The old (mathematically
    inconsistent) behavior is still available via
    apply_length_weight_to_features=True for exact reproduction of prior
    runs — off by default.
 7. TE weight now includes a genuine hit-QUALITY term (mean percent
    identity from MMseqs2) — v31 parsed pident/score per hit and then discarded
    them; "hit_sig" only ever counted hit COUNT, never hit confidence.
    Feature/weight alignment is also now a checked invariant: the same
    _valid_te_mask() function decides "this contig has usable TE
    evidence" for both build_feature_matrix_v22_5d and compute_te_weight,
    and a post-hoc check raises if a zero-feature row and a zero-weight
    row ever disagree (previously the two functions used two different,
    only-approximately-matching validity conditions).
 8. Explicit masked/unmasked contract — see INPUT CONTRACT above and
    schema.json's "input.expected_variant" field.
 9. MMseqs2 TSV parsing now validates its column
    schema on the first data line (wrong column count -> immediate,
    named error instead of silently treating every line as unparseable)
    and track a malformed-line ratio (>50% malformed on a non-trivial
    file -> error, not a quietly near-empty result). A post-hoc sanity
    check warns loudly if >95% of classified (non-skip) hits land in
    "Unclassified" — usually means the TE-name parsing doesn't match
    this database's naming convention, not that the biology is that way.
10. The ~1300-line commented-out V30 (svd64) implementation is removed.
    See test_te_composition.py for parser/interval/checkpoint tests.

config.yaml parameters:
    funTEdb: /path/to/db    # FASTA file or directory
    threads: 64
    mmseqs_sensitivity: 5.7
    mmseqs_min_seq_id: 0.7
    mmseqs_coverage: 0.1
    mmseqs_max_seq_len: 100000
    mmseqs_evalue: 1e-5
    apply_length_weight_to_features: false   # v31-compat opt-in, see item 6
    full_confidence_te_bp: 5000
    full_confidence_length: 10000
    full_confidence_hits: 5
    weight_te_coef: 0.55
    weight_hit_coef: 0.25
    weight_quality_coef: 0.20
    unclassified_warn_fraction: 0.95
"""

import gzip
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    from Bio import SeqIO
except ImportError as e:
    raise ImportError("Biopython required: pip install biopython") from e

try:
    from hyphaesbin.utils.logger import get_logger, log_step_start, log_step_done
except Exception:
    import logging
    def get_logger(name):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  [%(levelname)-8s]  %(name)s  %(message)s"
        )
        return logging.getLogger(name)
    def log_step_start(log, step, name, total):
        log.info(f"{'━'*66}")
        log.info(f"  STEP {step} / {total}   {name.upper()}")
        log.info(f"{'━'*66}")
    def log_step_done(log, step, name, elapsed):
        log.info(f"  ✅ STEP {step} DONE  :  {name}  [{elapsed/60:.1f}min]")
        log.info("")

try:
    from hyphaesbin.utils.checkpoint import Checkpoint
except Exception:
    Checkpoint = None

log = get_logger("hyphaesbin.te_composition")

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_THREADS               = 8

# V22_5D schema
TOTAL_V22_5D_DIM = 5
FIXED_FEATURE_NAMES = [
    "TE_richness",
    "AbsTE_sqrt",
    "DNA_ratio",
    "LTR_ratio",
    "Unclassified_ratio",
]

# MMseqs2 defaults
DEFAULT_MMSEQS_SENSITIVITY    = 5.7
DEFAULT_MMSEQS_MIN_SEQ_ID     = 0.7
DEFAULT_MMSEQS_COVERAGE       = 0.1
# MMseqs2 --cov-mode: 0 = query AND target, 1 = TARGET only, 2 = QUERY only. Query = contig, target = TE consensus.
# Default 1: at least `mmseqs_coverage` of the TE consensus must be aligned (partial TE fragments still count at 0.1).
# Mode 2 would demand that fraction of the whole CONTIG -- wrong constraint for TEs embedded in long contigs.
DEFAULT_MMSEQS_COV_MODE       = 1
DEFAULT_MMSEQS_MAX_SEQ_LEN    = 100_000
DEFAULT_MMSEQS_EVALUE         = 1e-5

# Weight formula defaults
DEFAULT_FULL_CONFIDENCE_TE_BP = 5000
DEFAULT_FULL_CONFIDENCE_LEN   = 10_000
DEFAULT_FULL_CONFIDENCE_HITS  = 5
DEFAULT_WEIGHT_TE_COEF        = 0.55
DEFAULT_WEIGHT_HIT_COEF       = 0.25
DEFAULT_WEIGHT_QUALITY_COEF   = 0.20
DEFAULT_UNCLASSIFIED_WARN_FRACTION = 0.95

VERSION = "HyphaeSBin_TE_V33_V22_5D_hardened"

# ── TE classification ──────────────────────────────────────────────────────────

SKIP_CLASSES = {"RRNA", "SNRNA", "TRNA", "SCRNA"}
LTR_CLASSES  = {"LTR", "RETROPOSON", "RETROTRANSPOSON"}
DNA_CLASSES  = {"DNA", "RC", "SATELLITE", "ROLLING_CIRCLE"}
LINE_CLASSES = {"LINE", "SINE", "SINE?"}

LTR_KEYWORDS  = ["gypsy","dirs","copia","ty1","ty3","bel","pao","retroviral","erv",
                  "ltr","chromovirus","reverse_transcriptase","integrase","retroposon"]
DNA_KEYWORDS  = ["dna","hat","hobo","activator","tc1","tc3","mariner","pogo",
                  "en_spm","enspm","cacta","mule","mudr","piggybac","harbinger",
                  "tourist","transib","kolobok","merlin","chapaev","crypton",
                  "transposase","tir","rolling","helitron","satellite"]
LINE_KEYWORDS = ["line","sine","r1","r2","rte","bov","l1","cin4","cr1","penelope",
                  "alu","b1","tad1","jockey","i-element"]

_FAILURE_LOG_MARKERS = ("segmentation fault", "core dumped", "killed", "out of memory",
                         "cannot allocate memory", "traceback (most recent call last)",
                         "fatal error", "aborted", "bus error")


def classify_te_name(name: str) -> str:
    """
    Classify TE hit name into TE class.

    Primary: parse #CLASS token from MycoMobilome / RepBase format.
    Fallback: keyword scan of hit name for any other database.

    Returns: LTR | DNA | LINE | Unclassified | Skip
    Skip = RNA-derived repeats excluded from TE features.
    """
    if "#" in name:
        top = name.split("#")[-1].upper().split("/")[0].strip()
        if top in SKIP_CLASSES:    return "Skip"
        if top in LTR_CLASSES:     return "LTR"
        if top in DNA_CLASSES:     return "DNA"
        if top in LINE_CLASSES:    return "LINE"
        if top in ("UNKNOWN", ""): return "Unclassified"
        return "Unclassified"
    t = name.lower()
    if any(k in t for k in LTR_KEYWORDS):  return "LTR"
    if any(k in t for k in DNA_KEYWORDS):  return "DNA"
    if any(k in t for k in LINE_KEYWORDS): return "LINE"
    return "Unclassified"


def empty_contig_entry() -> Dict:
    return {
        "ltr_bp": 0, "dna_bp": 0, "line_bp": 0,
        "unclassified_bp": 0, "total_te_bp": 0, "merged_te_bp": 0,
        "n_hits": 0, "quality_sum": 0.0, "quality_n": 0,
        "intervals": [],
    }


# =============================================================================
# CHECKPOINT FINGERPRINTING — same pattern as preprocessing.py / coverage.py /
# tnf_gene.py. A stale or config-mismatched checkpoint is a cache MISS, not a
# silent wrong answer.
# =============================================================================

def _file_fingerprint(path) -> str:
    """Size+mtime fingerprint, not a content hash. Known, acknowledged
    limitation (shared by this same helper across preprocessing.py /
    coverage.py / tnf_gene.py — not unique to this module): a same-size,
    same-mtime file replacement (e.g. an atomically-restored backup, or a
    filesystem that doesn't update mtime the way expected) would not be
    detected as changed. Deferred: a content hash (e.g. sha256 of the
    file) would close this gap but costs a full read of potentially large
    scaffold/TE-database FASTAs on every run, which is a real cost for
    the assemblies and databases this pipeline targets."""
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
        log.warning(f"{step}: checkpoint exists but inputs/config/tool-version changed since it "
                    f"ran — ignoring stale checkpoint and re-running.")
        return None
    missing = [p for p in output_paths if p and not Path(p).exists()]
    if missing:
        log.warning(f"{step}: checkpoint fingerprint matches, but {len(missing)} referenced "
                    f"output(s) no longer exist on disk ({missing[:3]}"
                    f"{', ...' if len(missing) > 3 else ''}) — treating as a cache miss.")
        return None
    return prev


def _tool_version(cmd: List[str]) -> str:
    """Best-effort tool version string for checkpoint fingerprinting /
    reproducibility metadata. Never raises — an unrecognized version flag
    or missing tool just yields 'unknown (...)' rather than blocking."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        out = (r.stdout or "").strip() or (r.stderr or "").strip()
        return out.splitlines()[0].strip() if out else "unknown (empty output)"
    except Exception as e:
        return f"unknown ({e.__class__.__name__})"


# =============================================================================
# Common helpers
# =============================================================================

def run_cmd(cmd: str, log_file: Optional[str] = None, allow_fail: bool = False):
    """Runs `cmd` via the shell. Known, acknowledged deferred-hardening
    item: shell=True means any unsanitized value interpolated into `cmd`
    (e.g. a config-supplied path) is executed by the shell rather than
    passed as a literal argv token. Not fixed in this pass — every caller
    in this module builds `cmd` from this pipeline's own config values
    (paths, thresholds), not untrusted external input, so the immediate
    risk is low, but switching to shell=False + an argv list (dropping
    shell features like `f'--format-output "..."'` quoting) is the
    correct long-term fix and should happen before this function is ever
    fed a value that isn't fully pipeline-controlled."""
    log.info(f"CMD: {cmd[:220]}...")
    try:
        if log_file:
            with open(log_file, "w") as lf:
                r = subprocess.run(cmd, shell=True, stdout=lf,
                                   stderr=subprocess.STDOUT, text=True)
        else:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0 and not allow_fail:
            log.error(f"Command failed rc={r.returncode}")
            if not log_file and r.stderr:
                log.error(r.stderr[:600])
            raise RuntimeError(f"Command failed: {cmd[:120]}")
        return r
    except Exception as e:
        if allow_fail:
            log.warning(f"allow_fail=True: {e}")
            return None
        raise


def _scan_log_for_failure_markers(log_file: str, tool_name: str) -> None:
    """Belt-and-suspenders check alongside the process return code: some
    wrapper/scheduler setups (e.g. an OOM killer inside a container) can
    leave a tool's own reported exit code at 0 even though it was killed
    mid-run. A 0 exit code alone is not sufficient evidence of success."""
    try:
        text = Path(log_file).read_text(errors="ignore").lower()
    except FileNotFoundError:
        return
    for marker in _FAILURE_LOG_MARKERS:
        if marker in text:
            raise RuntimeError(
                f"{tool_name} reported exit code 0, but its log contains {marker!r}, which "
                f"usually means it was killed or crashed partway through (e.g. OOM-killed inside "
                f"a wrapper that didn't propagate the real exit code). Treating this as a tool "
                f"failure, not a genuine result. See {log_file}."
            )


def check_tool(t: str) -> bool:
    return shutil.which(t) is not None


def open_gz(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def load_contig_lengths(fasta: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Parse assembly FASTA and return ordered contig ID list + length dict.
    Order is preserved — critical for alignment with other feature matrices.
    Duplicate contig IDs are rejected outright (item 1) — a dict-based
    length lookup can only keep one value per ID, so a duplicate silently
    discards one contig's real length and corrupts every downstream
    per-contig computation without ever raising.
    """
    ids, lengths = [], {}
    with open_gz(fasta) as f:
        for rec in SeqIO.parse(f, "fasta"):
            ids.append(rec.id)
            lengths[rec.id] = len(rec.seq)
    dup_ids = sorted({i for i in ids if ids.count(i) > 1}) if len(set(ids)) != len(ids) else []
    if dup_ids:
        raise ValueError(f"Assembly FASTA has {len(dup_ids)} duplicate contig ID(s) "
                          f"(e.g. {dup_ids[:5]}) — cannot safely align TE features to contigs. "
                          f"Fix the input assembly before computing TE features.")
    return ids, lengths


def _contig_order_hash(contig_ids: List[str]) -> str:
    return hashlib.sha256("\n".join(contig_ids).encode()).hexdigest()[:16]


def _validate_hit_ids_known(summary: Dict[str, Dict], contig_ids: List[str], source: str) -> None:
    """Item 1: a TE hit whose contig ID isn't in the assembly's own contig
    list means the search was run against the wrong assembly, or the
    search tool mangled/truncated IDs (e.g. a whitespace-split mismatch) — silently dropping those
    hits (as v31 did, by construction, since feature-building only ever
    iterates contig_ids) hides a real correctness bug. Fail loud instead."""
    known = set(contig_ids)
    orphans = sorted(set(summary.keys()) - known)
    if orphans:
        raise ValueError(
            f"{source}: {len(orphans)} contig ID(s) in the TE hit summary do not appear in the "
            f"assembly FASTA's own contig list (e.g. {orphans[:5]}). This means the TE search "
            f"ran against a different/mismatched assembly, or the search tool corrupted contig "
            f"IDs (e.g. header truncation) — not something safe to silently drop. Investigate "
            f"before trusting any TE feature from this run."
        )


def find_best_fasta(funtedb: str) -> str:
    """
    Locate TE consensus FASTA from user-provided path or directory.

    Priority for directories:
      1. MycoMobilome clustered 80% library (recommended for fungi)
      2. MycoMobilome protein-evidence library
      3. MycoMobilome unclustered library
      4. Any .fasta/.fa/.fna/.lib file (non-protein preferred)
    """
    p = Path(funtedb)
    if p.is_file() and p.stat().st_size > 0:
        log.info(f"TE DB: direct file → {p.name}")
        return str(p)
    if p.is_dir():
        candidates = [
            p / "MycoMobilome_v1.1-clustered_80_TE_library.fasta",
            p / "MycoMobilome_v1.1-clustered_80_proteinEvidence_TE_library.fasta",
            p / "MycoMobilome_v1.1-unclustered_TE_library.fasta",
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                log.info(f"TE DB: MycoMobilome → {c.name}")
                return str(c)
        for ext in ["*.fasta", "*.fa", "*.fna", "*.lib"]:
            files = sorted(p.glob(ext))
            nucl  = [f for f in files if not any(
                x in f.name.lower() for x in ["prot", "pep", "repeatpeps"])]
            chosen = nucl[0] if nucl else (files[0] if files else None)
            if chosen:
                log.info(f"TE DB: auto-selected → {chosen.name}")
                return str(chosen)
    raise FileNotFoundError(
        f"Cannot find TE FASTA from: {funtedb}\n"
        f"Provide path to a .fasta file or a directory containing one."
    )


def merged_intervals_bp(intervals: List[Tuple[int, int]]) -> int:
    """
    Merge overlapping genomic intervals and return total covered bp.
    Used to compute TE density (merged bp / contig length) avoiding
    double-counting overlapping hits. (item 4 — this function existed in
    v31 but was never called; it now feeds AbsTE_sqrt and the weight's
    te_sig term directly. See _finalize_summary.)
    """
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    a, b = intervals[0]
    for x, y in intervals[1:]:
        if x <= b:
            b = max(b, y)
        else:
            total += b - a + 1
            a, b = x, y
    return total + b - a + 1


def accumulate_hit(summary: Dict, contig: str, bases: int,
                   te_class: str, lo: int, hi: int, quality: float = 0.0):
    """Add a single TE hit to the per-contig summary. `quality` is a
    0-100 confidence unit (MMseqs2 percent identity) — accumulated so compute_te_weight can use
    real hit quality, not just hit count (item 7)."""
    if contig not in summary:
        summary[contig] = empty_contig_entry()
    s = summary[contig]
    s["total_te_bp"] += bases
    s["n_hits"]      += 1
    s["quality_sum"] += float(quality)
    s["quality_n"]   += 1
    s["intervals"].append((lo, hi))
    if te_class == "LTR":            s["ltr_bp"]          += bases
    elif te_class == "DNA":          s["dna_bp"]          += bases
    elif te_class == "LINE":         s["line_bp"]         += bases
    elif te_class == "Unclassified": s["unclassified_bp"] += bases


def _finalize_summary(summary: Dict) -> Dict:
    """Compute merged_te_bp (item 4) for every contig. Always called
    before a summary is used/saved/checkpointed — including right after
    loading an older checkpoint that predates this field, so a stale
    on-disk summary.json never silently lacks it."""
    for cid, s in summary.items():
        s["merged_te_bp"] = merged_intervals_bp(s.get("intervals", []))
    return summary


def _check_unclassified_fraction(summary: Dict, warn_fraction: float, min_n: int = 20) -> None:
    """Item 9: if almost every classified hit lands in 'Unclassified', the
    TE-name parsing very likely doesn't match this database's naming
    convention — that's a silent feature-quality failure, not a biological
    finding, and is easy to miss without an explicit check."""
    n_ltr = sum(1 for s in summary.values() if s.get("ltr_bp", 0) > 0)
    n_dna = sum(1 for s in summary.values() if s.get("dna_bp", 0) > 0)
    n_line = sum(1 for s in summary.values() if s.get("line_bp", 0) > 0)
    n_unc = sum(1 for s in summary.values() if s.get("unclassified_bp", 0) > 0)
    n_classified_total = n_ltr + n_dna + n_line + n_unc
    if n_classified_total < min_n:
        return
    frac_unc = n_unc / n_classified_total
    if frac_unc > warn_fraction:
        log.warning(
            f"⚠️  {frac_unc*100:.1f}% of contigs with TE signal have ONLY Unclassified hits "
            f"(LTR={n_ltr}, DNA={n_dna}, LINE={n_line}, Unclassified={n_unc}). This usually means "
            f"classify_te_name()'s '#CLASS' / keyword parsing doesn't match this database's naming "
            f"convention, not that the underlying biology is really this undifferentiated — check "
            f"a sample of raw hit names against LTR_CLASSES/DNA_CLASSES/LINE_CLASSES/keyword lists."
        )


def log_summary_stats(summary: Dict, label: str):
    n_with_te = sum(1 for s in summary.values() if s["total_te_bp"] > 0)
    cls = Counter()
    for s in summary.values():
        if s["ltr_bp"]          > 0: cls["LTR"] += 1
        if s["dna_bp"]          > 0: cls["DNA"] += 1
        if s["line_bp"]         > 0: cls["LINE"] += 1
        if s["unclassified_bp"] > 0: cls["Unclassified"] += 1
    log.info(f"{label}: {n_with_te:,} contigs with TE signal | {dict(cls)}")


def serialise_summary(summary: Dict) -> Dict:
    """Convert sets/tuples to lists for JSON serialisation."""
    out = {}
    for cid, s in summary.items():
        s2 = dict(s)
        s2["intervals"] = [[a, b] for a, b in s.get("intervals", [])]
        out[cid] = s2
    return out


def _deserialise_summary(raw: Dict) -> Dict:
    summary = {}
    for cid, s in raw.items():
        s["intervals"] = [tuple(x) for x in s.get("intervals", [])]
        s.setdefault("merged_te_bp", 0)
        s.setdefault("quality_sum", 0.0)
        s.setdefault("quality_n", 0)
        summary[cid] = s
    return _finalize_summary(summary)


def _valid_te_mask(summary: Dict, contig_ids: List[str], contig_lengths: Dict[str, int]) -> np.ndarray:
    """Single shared definition of 'this contig has usable TE evidence',
    used by BOTH build_feature_matrix_v22_5d and compute_te_weight (item
    7) so a zero-feature row and a zero-weight row can never disagree."""
    lengths = np.array([float(contig_lengths.get(c, 0)) for c in contig_ids])
    total_te = np.array([float(summary.get(c, {}).get("total_te_bp", 0)) for c in contig_ids])
    n_hits = np.array([float(summary.get(c, {}).get("n_hits", 0)) for c in contig_ids])
    return (lengths > 0) & (total_te > 0) & (n_hits > 0)


# =============================================================================
# STEP 1A — FAST MODE: MMseqs2
# =============================================================================

def _auto_mmseqs_memory() -> Optional[str]:
    """v9: cap MMseqs2's prefilter at ~70 % of the RAM this job may use (cgroup limit or MemAvailable), so it splits
    the index instead of dying with 'Cannot fit databases into ... memory' on memory-limited nodes. None = no cap."""
    lim = None
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v.isdigit() and int(v) < 1 << 60:
                lim = int(v)
                break
        except OSError:
            pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                av = int(line.split()[1]) * 1024
                lim = min(lim, av) if lim else av
                break
    except OSError:
        pass
    return f"{max(1, int(lim * 0.7 / 2**30))}G" if lim else None


def run_mmseqs2_search(
    scaffold: str, te_fasta: str, outdir: Path, threads: int,
    sensitivity: float, min_seq_id: float, coverage: float,
    max_seq_len: int, evalue: float, ckpt, split_memory_limit: Optional[str] = None,
    cov_mode: int = DEFAULT_MMSEQS_COV_MODE,
) -> Dict[str, Dict]:
    """
    Step 1 (fast mode): MMseqs2 nucleotide-nucleotide TE search.

    Searches assembled contigs against a TE consensus FASTA using MMseqs2
    easy-search in nucleotide mode (--search-type 3). Designed for large
    assemblies — ~20 min for 500k contigs at 64 threads.

    Key MMseqs2 flags:
      --search-type 3   : nucleotide vs nucleotide (required, not auto-detected)
      --max-seq-len     : handles long TE sequences (>64kb default limit)
      -c / --cov-mode   : `coverage` of the TE consensus (target, cov-mode 1 default) must align;
                          FIX: -c was never passed before, so no coverage filter was applied at all
      --max-seqs 300    : captures multiple overlapping TE hits per contig
      --strand 2        : search both strands

    Output:
      mmseqs_te_hits.tsv : raw alignment results (12 columns)
      te_annotations.bed : BED-format hit coordinates for downstream use
      te_contig_summary.json : aggregated per-contig TE statistics

    Returns:
      summary : dict {contig_id: {ltr_bp, dna_bp, line_bp, unclassified_bp,
                total_te_bp, merged_te_bp, n_hits, quality_sum, quality_n,
                intervals}}
    """
    STEP    = "te_mmseqs2_search"
    hit_dir = outdir / "01_mmseqs2"
    hit_dir.mkdir(parents=True, exist_ok=True)
    hits_tsv     = outdir / "mmseqs_te_hits.tsv"
    summary_path = outdir / "te_contig_summary.json"
    log_file     = str(hit_dir / "mmseqs2.log")

    mmseqs_version = _tool_version(["mmseqs", "version"])
    fp = _fingerprint(
        _file_fingerprint(scaffold), _file_fingerprint(te_fasta),
        sensitivity, min_seq_id, coverage, int(cov_mode), max_seq_len, evalue,
        mmseqs_version, VERSION, "inclusive-coords-bed0",
    )
    cached = _checkpoint_ok(ckpt, STEP, fp, [str(summary_path), str(hits_tsv)])
    if cached is not None:
        log.info("[checkpoint] MMseqs2 search done — loading")
        with open(summary_path) as f:
            raw = json.load(f)
        return _deserialise_summary(raw)

    t0 = time.time()
    log_step_start(log, 1, "MMseqs2 nucleotide TE search (fast mode)", 5)

    if not check_tool("mmseqs"):
        raise RuntimeError(
            "MMseqs2 not found.\n"
            "conda install -c bioconda mmseqs2 -n hyphaes -y"
        )

    tmp_dir = hit_dir / "mmseqs_tmp"
    tmp_dir.mkdir(exist_ok=True)

    log.info(f"sensitivity={sensitivity}  min_seq_id={min_seq_id}  "
             f"evalue={evalue}  max_seq_len={max_seq_len}  mmseqs_version={mmseqs_version}")

    cmd = (
        f"mmseqs easy-search "
        f"{scaffold} {te_fasta} {hits_tsv} {tmp_dir} "
        f"--search-type 3 "
        f"--threads {threads} "
        f"-s {sensitivity} "
        f"--min-seq-id {min_seq_id} "
        f"-c {coverage} --cov-mode {int(cov_mode)} "
        f"--min-aln-len 30 "
        f"-e {evalue} "
        f"--max-seqs 300 "
        f"--max-seq-len {max_seq_len} "
        f"--strand 2 "
        + (f"--split-memory-limit {split_memory_limit} " if split_memory_limit else "")
        + f'--format-output "query,target,pident,alnlen,mismatch,gapopen,'
        f'qstart,qend,tstart,tend,evalue,bits"'
    )
    run_cmd(cmd, log_file)
    _scan_log_for_failure_markers(log_file, "MMseqs2")

    # Item 3: MMseqs2 easy-search always writes its output TSV on success,
    # even when there are zero hits (empty file). A MISSING file after a
    # reported-successful run is a tool-failure signal, not a genuine
    # zero-hit result — treat the two cases differently instead of
    # touch()-ing a fake empty success either way.
    if not hits_tsv.exists():
        raise RuntimeError(
            f"MMseqs2 reported success (exit 0) but did not produce {hits_tsv.name} at all — "
            f"this looks like a tool failure (crashed after starting, wrong output path, disk "
            f"issue), not a genuine zero-hit result. Check {log_file}."
        )
    if hits_tsv.stat().st_size == 0:
        log.warning("MMseqs2 completed successfully and found genuinely ZERO TE hits across all "
                    "contigs (the output file exists but is empty) — this is a valid result.")

    summary  = {}
    bed_path = outdir / "te_annotations.bed"
    n_total = n_kept = n_skip = n_malformed = 0
    first_line_checked = False

    with open(hits_tsv) as f, open(bed_path, "w") as bed:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")

            if not first_line_checked:
                first_line_checked = True
                if len(parts) != 12:
                    raise RuntimeError(
                        f"{hits_tsv.name} does not have the expected 12 tab-separated columns "
                        f"(query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,"
                        f"evalue,bits) — got {len(parts)} on the first data line. This looks like "
                        f"a --format-output schema mismatch (different MMseqs2 version, or a "
                        f"hand-edited command), not an isolated parsing edge case. "
                        f"First line: {line[:200]!r}"
                    )

            if len(parts) != 12:
                n_malformed += 1
                continue
            try:
                contig = parts[0]
                tname  = parts[1]
                pident = float(parts[2])
                qstart = int(parts[6])
                qend   = int(parts[7])
                bases  = abs(qend - qstart) + 1          # FIX: MMseqs2 coordinates are 1-based INCLUSIVE
            except (ValueError, IndexError):
                n_malformed += 1
                continue

            n_total += 1
            if bases < 30:
                continue

            te_class = classify_te_name(tname)
            if te_class == "Skip":
                n_skip += 1
                continue

            n_kept += 1
            lo, hi = min(qstart, qend), max(qstart, qend)
            accumulate_hit(summary, contig, bases, te_class, lo, hi, quality=pident)
            bed.write(f"{contig}\t{lo - 1}\t{hi}\n")   # FIX: BED = 0-based start, half-open end

    n_lines_seen = n_total + n_malformed
    if n_lines_seen > 0 and n_malformed / n_lines_seen > 0.5:
        raise RuntimeError(
            f"{hits_tsv.name}: {n_malformed}/{n_lines_seen} lines were malformed (>50%) — this "
            f"looks like a schema mismatch, not scattered bad lines. Not proceeding with a mostly-"
            f"unparseable result."
        )

    log.info(f"MMseqs2: total={n_total:,}  kept={n_kept:,}  skipped(RNA)={n_skip:,}  "
             f"malformed={n_malformed:,}")
    _finalize_summary(summary)
    log_summary_stats(summary, "MMseqs2")

    with open(summary_path, "w") as f:
        json.dump(serialise_summary(summary), f, indent=2)

    if ckpt:
        ckpt.mark_done(STEP, {
            "_fp": fp,
            "n_total":   n_total,
            "n_kept":    n_kept,
            "n_with_te": len(summary),
            "mmseqs_version": mmseqs_version,
        })

    log_step_done(log, 1, "MMseqs2 TE search", time.time() - t0)
    return summary


def build_feature_matrix_v22_5d(
    summary: Dict[str, Dict],
    contig_ids: List[str],
    contig_lengths: Dict[str, int],
    outdir: Path,
    ckpt,
    summary_path: Path,
    apply_length_weight_to_features: bool = False,
) -> np.ndarray:
    """
    Step 2: Build V22_5D TE feature matrix (N × 5). Vectorized: all five
    columns are computed as whole-array numpy operations instead of a
    per-contig Python loop. Same formulas/semantics as before — only the
    computation strategy changed, not any value.
    """
    STEP = "te_build_features_v22_5d"
    features_path = outdir / "te_features.npy"
    order_hash = _contig_order_hash(contig_ids)

    fp = _fingerprint(_file_fingerprint(summary_path), order_hash,
                       apply_length_weight_to_features, VERSION)
    cached = _checkpoint_ok(ckpt, STEP, fp, [str(features_path)])
    if cached is not None:
        log.info("[checkpoint] Loading V22_5D features")
        feat = np.load(features_path)
        log.info(f"  shape: {feat.shape}")
        return feat

    t0 = time.time()
    log_step_start(log, 2, "Build V22_5D feature matrix (5D)", 5)

    N = len(contig_ids)
    lengths_arr = np.array([float(contig_lengths.get(c, 0)) for c in contig_ids], dtype=np.float64)
    max_len = lengths_arr.max() if lengths_arr.max() > 0 else 1.0
    valid = _valid_te_mask(summary, contig_ids, contig_lengths)

    # ── Pull every needed per-contig scalar out of `summary` in ONE pass
    #    over contig_ids (still O(N) dict lookups — unavoidable, summary is
    #    a dict — but everything AFTER this is pure array math, no more
    #    per-contig branching/indexing). This replaces the old per-contig
    #    `for i, cid in enumerate(contig_ids): raw[i,0]=...; raw[i,1]=...`
    #    loop with vectorized numpy ops over the whole N-length arrays.
    empty = {}
    total_bp = np.empty(N, dtype=np.float64)
    merged_bp = np.empty(N, dtype=np.float64)
    ltr_bp = np.empty(N, dtype=np.float64)
    dna_bp = np.empty(N, dtype=np.float64)
    line_bp = np.empty(N, dtype=np.float64)
    unc_bp = np.empty(N, dtype=np.float64)

    for i, cid in enumerate(contig_ids):
        s = summary.get(cid, empty)
        total_bp[i] = s.get("total_te_bp", 0)
        merged_bp[i] = s.get("merged_te_bp", 0)
        ltr_bp[i] = s.get("ltr_bp", 0)
        dna_bp[i] = s.get("dna_bp", 0)
        line_bp[i] = s.get("line_bp", 0)
        unc_bp[i] = s.get("unclassified_bp", 0)

    # ── Vectorized feature computation — array-wide, no per-row branching ──
    active_classes = ((ltr_bp > 0).astype(np.float64) + (dna_bp > 0).astype(np.float64)
                       + (line_bp > 0).astype(np.float64) + (unc_bp > 0).astype(np.float64))

    d0 = active_classes / 4.0
    # merged/length, guarded against length==0 with np.divide's `where`
    d1 = np.divide(merged_bp, lengths_arr, out=np.zeros(N), where=lengths_arr > 0)
    d1 = np.minimum(d1, 1.0)  # defensive clamp only — merged bp can't exceed length by construction
    d2 = np.divide(dna_bp, total_bp, out=np.zeros(N), where=total_bp > 0)
    d3 = np.divide(ltr_bp, total_bp, out=np.zeros(N), where=total_bp > 0)
    d4 = np.divide(unc_bp, total_bp, out=np.zeros(N), where=total_bp > 0)

    raw = np.stack([d0, d1, d2, d3, d4], axis=1).astype(np.float32)
    raw[~valid, :] = 0.0  # zero out rows that fail the shared validity mask, same as before

    features = np.sqrt(np.clip(raw, 0.0, None)).astype(np.float32)

    if apply_length_weight_to_features:
        log.warning("apply_length_weight_to_features=True — reproducing v31's length "
                    "multiplication of ALL 5 features, including the 3 ratio features. See "
                    "item 6 in the module changelog for why this is off by default.")
        len_w = np.sqrt(lengths_arr / max_len).reshape(-1, 1).astype(np.float32)
        features = (features * len_w).astype(np.float32)

    zero_rows = int((features == 0).all(axis=1).sum())
    zero_weight_rows_expected = int((~valid).sum())
    if zero_rows != zero_weight_rows_expected:
        raise ValueError(
            f"Internal consistency error: {zero_rows} all-zero feature rows but "
            f"{zero_weight_rows_expected} rows failed _valid_te_mask — these must match exactly "
            f"so a downstream consumer can trust 'weight==0 <=> feature row is a placeholder'. "
            f"Investigate before trusting this run's TE features."
        )

    log.info(f"V22_5D: {features.shape}")
    log.info(f"  Features  : {FIXED_FEATURE_NAMES}")
    log.info(f"  Means     : {features.mean(axis=0).round(4).tolist()}")
    log.info(f"  Stds      : {features.std(axis=0).round(4).tolist()}")
    log.info(f"  Nonzero   : {(features != 0).sum(axis=0).tolist()}")
    log.info(f"  Zero rows : {zero_rows:,}/{N} ({zero_rows/N*100:.1f}%)" if N else "  Zero rows : 0/0")

    np.save(features_path, features)

    if ckpt:
        ckpt.mark_done(STEP, {
            "_fp": fp,
            "schema":    "v22_5d",
            "shape":     list(features.shape),
            "zero_rows": zero_rows,
            "zero_pct":  round(zero_rows / N * 100, 2) if N else 0.0,
            "contig_order_hash": order_hash,
            "apply_length_weight_to_features": apply_length_weight_to_features,
        })

    log_step_done(log, 2, "V22_5D feature matrix", time.time() - t0)
    return features


# =============================================================================
# STEP 3 — COMPUTE TE CONFIDENCE WEIGHTS
# =============================================================================

def compute_te_weight(
    summary: Dict[str, Dict],
    contig_ids: List[str],
    contig_lengths: Dict[str, int],
    outdir: Path,
    ckpt,
    summary_path: Path,
    full_confidence_te_bp: float = DEFAULT_FULL_CONFIDENCE_TE_BP,
    full_confidence_length: float = DEFAULT_FULL_CONFIDENCE_LEN,
    full_confidence_hits: float = DEFAULT_FULL_CONFIDENCE_HITS,
    weight_te_coef: float = DEFAULT_WEIGHT_TE_COEF,
    weight_hit_coef: float = DEFAULT_WEIGHT_HIT_COEF,
    weight_quality_coef: float = DEFAULT_WEIGHT_QUALITY_COEF,
) -> np.ndarray:
    """
    Step 3: Compute per-contig TE confidence weights (N × 1).

    TE features are highly sparse — most contigs have zero TE signal,
    especially in fragmented assemblies. The encoder must know which contigs
    have reliable TE information to avoid learning from noise.

    Weight formula (item 7 — now includes a genuine hit-QUALITY term; v31
    only ever counted hit COUNT, discarding the pident/SW-score values it
    had already parsed):
      te_sig   = min(1, merged_te_bp / full_confidence_te_bp)  — TE coverage
                 signal, using MERGED bp (item 4), not the raw overlap sum
      len_sig  = min(1, length / full_confidence_length)        — length confidence
      hit_sig  = min(1, n_hits / full_confidence_hits)          — hit count signal
      qual_sig = mean_hit_quality / 100                         — mean hit
                 quality (MMseqs2 percent identity), clipped to [0,1]
      weight   = clip(weight_te_coef * te_sig * len_sig
                     + weight_hit_coef * hit_sig
                     + weight_quality_coef * qual_sig, 0, 1)
      (weight_te_coef + weight_hit_coef + weight_quality_coef must sum to
      1.0 — validated, not just assumed)

    Contigs with no TE hits receive weight = 0.0, using the SAME validity
    definition (_valid_te_mask) as build_feature_matrix_v22_5d, so
    weight==0 if and only if that contig's feature row is all-zero.

    Output:
      te_weight.npy : (N × 1) float32, values in [0, 1]

    Returns:
      weight : np.ndarray (N × 1)
    """
    coef_sum = weight_te_coef + weight_hit_coef + weight_quality_coef
    if abs(coef_sum - 1.0) > 1e-6:
        raise ValueError(f"weight_te_coef + weight_hit_coef + weight_quality_coef must sum to "
                          f"1.0, got {coef_sum} ({weight_te_coef} + {weight_hit_coef} + "
                          f"{weight_quality_coef}).")

    STEP = "te_weights"
    weight_path = outdir / "te_weight.npy"
    order_hash = _contig_order_hash(contig_ids)

    fp = _fingerprint(_file_fingerprint(summary_path), order_hash,
                       full_confidence_te_bp, full_confidence_length, full_confidence_hits,
                       weight_te_coef, weight_hit_coef, weight_quality_coef, VERSION)
    cached = _checkpoint_ok(ckpt, STEP, fp, [str(weight_path)])
    if cached is not None:
        log.info("[checkpoint] Loading TE weights")
        return np.load(weight_path)

    t0 = time.time()
    log_step_start(log, 3, "Compute TE confidence weights", 5)

    N        = len(contig_ids)
    lengths  = np.array([float(contig_lengths.get(c, 0)) for c in contig_ids])
    merged_te = np.array([float(summary.get(c, {}).get("merged_te_bp", 0)) for c in contig_ids])
    n_hits   = np.array([float(summary.get(c, {}).get("n_hits", 0))      for c in contig_ids])
    quality_sum = np.array([float(summary.get(c, {}).get("quality_sum", 0.0)) for c in contig_ids])
    quality_n   = np.array([float(summary.get(c, {}).get("quality_n", 0))     for c in contig_ids])

    valid   = _valid_te_mask(summary, contig_ids, contig_lengths)
    mean_quality = np.divide(quality_sum, quality_n, out=np.zeros(N), where=quality_n > 0)
    qual_sig = np.clip(mean_quality / 100.0, 0.0, 1.0)

    te_sig  = np.minimum(1.0, np.divide(merged_te, full_confidence_te_bp, out=np.zeros(N), where=valid))
    len_sig = np.minimum(1.0, np.divide(lengths,  full_confidence_length, out=np.zeros(N), where=valid))
    hit_sig = np.minimum(1.0, np.divide(n_hits,   full_confidence_hits,   out=np.zeros(N), where=valid))
    weight  = np.clip(weight_te_coef * te_sig * len_sig
                       + weight_hit_coef * hit_sig
                       + weight_quality_coef * qual_sig, 0.0, 1.0)
    weight[~valid] = 0.0
    weight = weight.reshape(-1, 1).astype(np.float32)

    zero_weight_rows = int((weight[:, 0] == 0).sum())
    expected_zero = int((~valid).sum())
    if zero_weight_rows != expected_zero:
        raise ValueError(
            f"Internal consistency error: {zero_weight_rows} zero-weight rows but "
            f"{expected_zero} rows failed _valid_te_mask — these must match exactly."
        )

    np.save(weight_path, weight)
    nonzero = int((weight > 0).sum())
    log.info(f"TE weights: nonzero={nonzero:,}/{N}  "
             f"range=[{float(weight.min()):.3f}, {float(weight.max()):.3f}]  "
             f"mean_quality(nonzero)={float(mean_quality[valid].mean()) if valid.any() else 0.0:.1f}")

    if ckpt:
        ckpt.mark_done(STEP, {
            "_fp": fp, "nonzero": nonzero, "contig_order_hash": order_hash,
            "weight_coefs": {"te": weight_te_coef, "hit": weight_hit_coef, "quality": weight_quality_coef},
        })

    log_step_done(log, 3, "TE confidence weights", time.time() - t0)
    return weight


# =============================================================================
# STEP 4 — SAVE ALL OUTPUTS
# =============================================================================

def save_outputs(
    contig_ids    : List[str],
    summary       : Dict[str, Dict],
    outdir        : Path,
    funtedb       : str,
    scaffold      : str,
    feature_dim   : int,
    apply_length_weight_to_features: bool,
) -> Dict[str, Any]:
    """
    Step 4: Save all required output files.

    Saves:
      contig_ids.json         — ordered list of contig IDs (matches row order
                                of all .npy matrices)
      te_contig_summary.json  — raw per-contig TE hit statistics
      schema.json              — full metadata for reproducibility

    The schema.json records all parameters and output file formats,
    enabling exact reproduction of results, plus a contig-order hash
    (item 1) for cross-module reconciliation with TNF/coverage, and an
    explicit masked/unmasked input contract (item 8).
    """
    with open(outdir / "contig_ids.json", "w") as f:
        json.dump(contig_ids, f, indent=2)

    with open(outdir / "te_contig_summary.json", "w") as f:
        json.dump(serialise_summary(summary), f, indent=2)

    db_name       = Path(funtedb).name if funtedb else "unknown"
    search_engine = "MMseqs2"

    schema: Dict[str, Any] = {
        "version"        : VERSION,
        "te_features"    : "v22_5d",
        "feature_dim"    : feature_dim,
        "feature_names"  : FIXED_FEATURE_NAMES,
        "transform"      : ("sqrt(feature), applied once uniformly to all 5 features"
                              + (" * sqrt(contig_length / max_length) [apply_length_weight_to_"
                                 "features=True, v31-compat]" if apply_length_weight_to_features
                                 else " (length-independent — length confidence lives in "
                                      "te_weight.npy only, see module changelog item 6)")),
        "database"       : db_name,
        "search_engine"  : search_engine,
        "input": {
            "scaffold": str(scaffold),
            "scaffold_fingerprint": _file_fingerprint(scaffold),
            "expected_variant": "UNMASKED — never the rRNA-masked final_clean.fasta used for TNF. "
                                 "See module docstring INPUT CONTRACT.",
        },
        "contig_order_hash": _contig_order_hash(contig_ids),
        "te_classes"     : {
            "LTR"         : "LTR retrotransposons (Gypsy, Copia, DIRS, etc.)",
            "DNA"         : "DNA transposons (TcMar, hAT, Helitron, etc.)",
            "LINE"        : "LINEs and SINEs",
            "Unclassified": "#Unknown — annotated TEs without class",
            "Skip"        : "RNA repeats (rRNA/snRNA/tRNA) — excluded",
        },
        "search": "MMseqs2 nucl-nucl easy-search (the only TE search mode in this version)",
        "outputs": {
            "te_features.npy"      : f"(N × {feature_dim}) float32 — V22_5D features",
            "te_weight.npy"        : "(N × 1) float32 — TE confidence weights [0,1]",
            "contig_ids.json"      : "ordered contig IDs (matches .npy row order)",
            "schema.json"          : "this file — full metadata",
            "te_annotations.bed"   : "BED-format hit coordinates",
            "mmseqs_te_hits.tsv"   : "raw MMseqs2 output",
            "te_contig_summary.json": "per-contig aggregated TE hit statistics",
        },
    }

    with open(outdir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    return schema


# =============================================================================
# MAIN PUBLIC API
# =============================================================================

def run_te_composition(
    scaffold             : str,
    outdir               : str,
    funtedb              : str,
    threads              : int   = DEFAULT_THREADS,
    resume               : bool  = True,
    # MMseqs2 params
    mmseqs_sensitivity   : float = DEFAULT_MMSEQS_SENSITIVITY,
    mmseqs_min_seq_id    : float = DEFAULT_MMSEQS_MIN_SEQ_ID,
    mmseqs_coverage      : float = DEFAULT_MMSEQS_COVERAGE,
    mmseqs_cov_mode      : int   = DEFAULT_MMSEQS_COV_MODE,
    mmseqs_max_seq_len   : int   = DEFAULT_MMSEQS_MAX_SEQ_LEN,
    mmseqs_split_memory_limit: Optional[str] = None,
    mmseqs_evalue        : float = DEFAULT_MMSEQS_EVALUE,
    # Feature-transform / weight params (item 6 / 7)
    apply_length_weight_to_features: bool = False,
    full_confidence_te_bp   : float = DEFAULT_FULL_CONFIDENCE_TE_BP,
    full_confidence_length  : float = DEFAULT_FULL_CONFIDENCE_LEN,
    full_confidence_hits    : float = DEFAULT_FULL_CONFIDENCE_HITS,
    weight_te_coef          : float = DEFAULT_WEIGHT_TE_COEF,
    weight_hit_coef         : float = DEFAULT_WEIGHT_HIT_COEF,
    weight_quality_coef     : float = DEFAULT_WEIGHT_QUALITY_COEF,
    unclassified_warn_fraction: float = DEFAULT_UNCLASSIFIED_WARN_FRACTION,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unified TE composition V33 — V22_5D schema.

    Runs the full TE annotation + feature extraction pipeline:
      Step 1 : TE search (MMseqs2)
      Step 2 : Build V22_5D feature matrix (N × 5)
      Step 3 : Compute TE confidence weights (N × 1)
      Step 4 : Save all outputs

    Args:
        scaffold         : path to the UNMASKED assembly FASTA (see module
                            docstring INPUT CONTRACT — NOT final_clean.fasta)
        outdir           : output directory
        funtedb          : TE database FASTA or directory
        threads          : CPU threads
        resume           : use fingerprinted checkpoints to skip completed
                            steps whose inputs/config/tool-version haven't
                            changed

    Returns:
        te_features : np.ndarray (N × 5)   V22_5D feature matrix
        te_weight   : np.ndarray (N × 1)   TE confidence weights
        schema      : dict                  full metadata
    """
    t0     = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt   = Checkpoint(outdir) if resume and Checkpoint is not None else None

    if kwargs.pop("te_mode", None) not in (None, "fast"):
        log.warning("te_mode is no longer used: RepeatMasker ('efficient') mode was removed; MMseqs2 is always used")
    for _old in ("min_sw_score", "te_pa", "te_frag", "repeatmasker_quality_reference_score"):
        kwargs.pop(_old, None)

    log.info("")
    log.info("╔" + "═"*68 + "╗")
    log.info("║" + " "*10 + "TE COMPOSITION V33 — UNIFIED MODULE" + " "*23 + "║")
    log.info("╚" + "═"*68 + "╝")
    log.info("  search   : MMseqs2 (~20 min / 500k contigs)")
    log.info(f"  schema   : V22_5D (5D — TE_richness, AbsTE_sqrt, DNA/LTR/Unclassified ratio)")
    log.info(f"  scaffold : {scaffold}  (expected: UNMASKED assembly)")
    log.info(f"  funtedb  : {funtedb}")
    log.info(f"  threads  : {threads}")
    log.info("")

    if not Path(scaffold).exists():
        raise FileNotFoundError(f"Scaffold not found: {scaffold}")

    contig_ids, contig_lengths = load_contig_lengths(scaffold)
    N = len(contig_ids)
    log.info(f"Contigs: {N:,}  longest: {max(contig_lengths.values()):,} bp" if N else "Contigs: 0")

    te_fasta = find_best_fasta(funtedb)
    log.info(f"TE database: {Path(te_fasta).name}  "
             f"({Path(te_fasta).stat().st_size/1e6:.0f} MB)")

    # Step 1 — TE search
    summary = run_mmseqs2_search(
        scaffold=scaffold, te_fasta=te_fasta, outdir=outdir,
        threads=threads, sensitivity=mmseqs_sensitivity,
        min_seq_id=mmseqs_min_seq_id, coverage=mmseqs_coverage,
        max_seq_len=mmseqs_max_seq_len, evalue=mmseqs_evalue, ckpt=ckpt,
        split_memory_limit=mmseqs_split_memory_limit, cov_mode=mmseqs_cov_mode,
    )

    # Item 1: fail loud on any hit landing on a contig ID the assembly
    # doesn't have, instead of silently dropping it.
    _validate_hit_ids_known(summary, contig_ids, source="MMseqs2 search")
    _check_unclassified_fraction(summary, unclassified_warn_fraction)

    summary_path = outdir / "te_contig_summary.json"

    # Step 2 — Feature matrix
    te_feat = build_feature_matrix_v22_5d(
        summary, contig_ids, contig_lengths, outdir, ckpt, summary_path,
        apply_length_weight_to_features=apply_length_weight_to_features)

    if te_feat.shape != (N, TOTAL_V22_5D_DIM):
        raise ValueError(f"Shape mismatch: got {te_feat.shape}, expected ({N}, {TOTAL_V22_5D_DIM}).")

    # Step 3 — Weights
    te_weight = compute_te_weight(
        summary, contig_ids, contig_lengths, outdir, ckpt, summary_path,
        full_confidence_te_bp=full_confidence_te_bp,
        full_confidence_length=full_confidence_length,
        full_confidence_hits=full_confidence_hits,
        weight_te_coef=weight_te_coef, weight_hit_coef=weight_hit_coef,
        weight_quality_coef=weight_quality_coef)

    if te_weight.shape != (N, 1):
        raise ValueError(f"Weight shape mismatch: got {te_weight.shape}, expected ({N}, 1).")
    if len(contig_ids) != N:
        raise ValueError(f"contig_ids length {len(contig_ids)} != N {N}.")

    # Step 4 — Save outputs
    schema = save_outputs(
        contig_ids=contig_ids,
        summary=summary,
        outdir=outdir,
        funtedb=funtedb,
        scaffold=scaffold,
        feature_dim=TOTAL_V22_5D_DIM,
        apply_length_weight_to_features=apply_length_weight_to_features,
    )
    schema["te_bed"] = str(outdir / "te_annotations.bed")

    elapsed = time.time() - t0
    log.info("")
    log.info(f"  te_features.npy : {outdir / 'te_features.npy'}  {te_feat.shape}")
    log.info(f"  te_weight.npy   : {outdir / 'te_weight.npy'}    {te_weight.shape}")
    log.info(f"  schema.json     : {outdir / 'schema.json'}")
    log.info(f"  zero rows       : {int((te_feat==0).all(axis=1).sum()):,}/{N} "
             f"({(te_feat==0).all(axis=1).mean()*100:.1f}%)" if N else "  zero rows       : 0/0")
    log.info(f"  total time      : {elapsed/60:.1f} min")
    log.info("")

    return te_feat, te_weight, schema


# ── Compatibility wrapper ──────────────────────────────────────────────────────

def run_te_branch(
    scaffold       : str,
    contig_names   = None,
    contig_lengths = None,
    tier           = None,
    config         = None,
    outdir         : str = "te_composition_output",
    ckpt           = None,
):
    """
    Compatibility wrapper for legacy callers.
    Reads all parameters from config dict — no code changes needed in main.py.
    """
    config  = config or {}
    funtedb = (config.get("funTEdb") or config.get("funtedb") or
               config.get("FunTEDB") or "")
    threads = int(config.get("threads", DEFAULT_THREADS))

    te_outdir = Path(outdir) / "te_composition"
    te_feat, te_weight, schema = run_te_composition(
        scaffold=scaffold, outdir=str(te_outdir), funtedb=funtedb,
        threads=threads, resume=True,
        mmseqs_sensitivity   = float(config.get("mmseqs_sensitivity",     DEFAULT_MMSEQS_SENSITIVITY)),
        mmseqs_min_seq_id    = float(config.get("mmseqs_min_seq_id",      DEFAULT_MMSEQS_MIN_SEQ_ID)),
        mmseqs_coverage      = float(config.get("mmseqs_coverage",        DEFAULT_MMSEQS_COVERAGE)),
        mmseqs_cov_mode      = int(config.get("mmseqs_cov_mode",          DEFAULT_MMSEQS_COV_MODE)),
        mmseqs_max_seq_len   = int(config.get("mmseqs_max_seq_len",       DEFAULT_MMSEQS_MAX_SEQ_LEN)),
        mmseqs_split_memory_limit = (str(config["mmseqs_split_memory_limit"])
                                     if config.get("mmseqs_split_memory_limit") else _auto_mmseqs_memory()),
        mmseqs_evalue        = float(config.get("mmseqs_evalue",          DEFAULT_MMSEQS_EVALUE)),
        apply_length_weight_to_features = bool(config.get("apply_length_weight_to_features", False)),
        full_confidence_te_bp   = float(config.get("full_confidence_te_bp",   DEFAULT_FULL_CONFIDENCE_TE_BP)),
        full_confidence_length  = float(config.get("full_confidence_length",  DEFAULT_FULL_CONFIDENCE_LEN)),
        full_confidence_hits    = float(config.get("full_confidence_hits",    DEFAULT_FULL_CONFIDENCE_HITS)),
        weight_te_coef          = float(config.get("weight_te_coef",          DEFAULT_WEIGHT_TE_COEF)),
        weight_hit_coef         = float(config.get("weight_hit_coef",         DEFAULT_WEIGHT_HIT_COEF)),
        weight_quality_coef     = float(config.get("weight_quality_coef",     DEFAULT_WEIGHT_QUALITY_COEF)),
        unclassified_warn_fraction = float(config.get(
            "unclassified_warn_fraction", DEFAULT_UNCLASSIFIED_WARN_FRACTION)),
    )
    return te_feat, te_weight, schema.get("te_bed", "")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(
        prog="te_composition",
        description=(
            "HyphaeSBin TE Composition V33\n"
            "  TE search: MMseqs2 nucl-nucl (~20 min / 500k contigs)\n"
            "  Output: V22_5D (5D) — TE_richness, AbsTE_sqrt, DNA_ratio, LTR_ratio, Unclassified_ratio\n"
            "  Works with any nucleotide FASTA database.\n"
            "  Scaffold must be the UNMASKED assembly (see module docstring).\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scaffold",  required=True,  help="UNMASKED assembly FASTA")
    p.add_argument("--funtedb",   required=True,  help="TE database FASTA or directory")
    p.add_argument("--outdir",    required=True,  help="Output directory")
    p.add_argument("--threads",   type=int, default=DEFAULT_THREADS)
    p.add_argument("--no-resume", action="store_true")

    mm = p.add_argument_group("MMseqs2")
    mm.add_argument("--mmseqs-sensitivity", type=float, default=DEFAULT_MMSEQS_SENSITIVITY)
    mm.add_argument("--mmseqs-min-seq-id",  type=float, default=DEFAULT_MMSEQS_MIN_SEQ_ID)
    mm.add_argument("--mmseqs-coverage",    type=float, default=DEFAULT_MMSEQS_COVERAGE)
    mm.add_argument("--mmseqs-cov-mode",    type=int, default=DEFAULT_MMSEQS_COV_MODE, choices=(0, 1, 2))
    mm.add_argument("--mmseqs-max-seq-len", type=int,   default=DEFAULT_MMSEQS_MAX_SEQ_LEN)
    mm.add_argument("--mmseqs-evalue",      type=float, default=DEFAULT_MMSEQS_EVALUE)

    ft = p.add_argument_group("Feature transform / weight (see item 6/7 in module changelog)")
    ft.add_argument("--apply-length-weight-to-features", action="store_true",
                    help="v31-compat: also length-multiply features (off by default; see docstring)")
    ft.add_argument("--full-confidence-te-bp", type=float, default=DEFAULT_FULL_CONFIDENCE_TE_BP)
    ft.add_argument("--full-confidence-length", type=float, default=DEFAULT_FULL_CONFIDENCE_LEN)
    ft.add_argument("--full-confidence-hits", type=float, default=DEFAULT_FULL_CONFIDENCE_HITS)
    ft.add_argument("--weight-te-coef", type=float, default=DEFAULT_WEIGHT_TE_COEF)
    ft.add_argument("--weight-hit-coef", type=float, default=DEFAULT_WEIGHT_HIT_COEF)
    ft.add_argument("--weight-quality-coef", type=float, default=DEFAULT_WEIGHT_QUALITY_COEF)
    ft.add_argument("--unclassified-warn-fraction", type=float,
                    default=DEFAULT_UNCLASSIFIED_WARN_FRACTION)

    args = p.parse_args()
    run_te_composition(
        scaffold=args.scaffold, outdir=args.outdir, funtedb=args.funtedb,
        threads=args.threads, resume=not args.no_resume,
        mmseqs_sensitivity=args.mmseqs_sensitivity,
        mmseqs_min_seq_id=args.mmseqs_min_seq_id,
        mmseqs_coverage=args.mmseqs_coverage, mmseqs_cov_mode=args.mmseqs_cov_mode,
        mmseqs_max_seq_len=args.mmseqs_max_seq_len,
        mmseqs_evalue=args.mmseqs_evalue,
        apply_length_weight_to_features=args.apply_length_weight_to_features,
        full_confidence_te_bp=args.full_confidence_te_bp,
        full_confidence_length=args.full_confidence_length,
        full_confidence_hits=args.full_confidence_hits,
        weight_te_coef=args.weight_te_coef,
        weight_hit_coef=args.weight_hit_coef,
        weight_quality_coef=args.weight_quality_coef,
        unclassified_warn_fraction=args.unclassified_warn_fraction,
    )




# PROFILE HOOKS v1 — observation only; scientific functions above are unchanged.
from hyphaesbin.utils.resource_profiler import profile_functions, profile_checkpoint
profile_functions(globals(), ['run_te_branch', 'run_te_composition', 'run_mmseqs2_search', 'build_feature_matrix_v22_5d', 'run_cmd'])
_checkpoint_ok = profile_checkpoint(_checkpoint_ok)

if __name__ == "__main__":
    main()
