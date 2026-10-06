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
and RepeatMasker/MMseqs2 cannot align against N runs, so masking the input
would make any TE sequence that happens to overlap an rRNA locus
artificially undetectable. RNA-derived repeat hits (rRNA/snRNA/tRNA) are
already filtered out at the classification stage (SKIP_CLASSES) regardless
of whether the input was masked, so there is no need to pre-mask for TE
search — searching the unmasked sequence and filtering RNA-class hits
after the fact is strictly more correct.

PIPELINE STEPS
--------------
Step 1  MMseqs2 / RepeatMasker search
        Align assembled contigs against a TE consensus library (e.g.
        MycoMobilome) to detect TE hits. MMseqs2 is fast (~20 min / 500k
        contigs); RepeatMasker is slower (~1-2 hrs) but more sensitive.

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
        te_annotations.bed, mmseqs_te_hits.tsv (fast mode).

MODES
-----
te_mode: fast       -> MMseqs2 nucleotide search
                      ~20 min / 500k contigs at 64 threads
                      ~75% sensitivity relative to RepeatMasker (approximate,
                      dataset- and parameter-dependent — not a measured
                      constant for every assembly/database combination)
                      Recommended for real data and large assemblies

te_mode: efficient  -> RepeatMasker + any FASTA database
                      ~10-24 hrs depending on assembly size
                      ~95% sensitivity (near gold standard; approximate,
                      dataset-dependent — see note above)
                      Recommended for benchmarking and small assemblies

Both modes produce identical V22_5D output schema.

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
    tool's own version string (best-effort `mmseqs version` / `RepeatMasker
    -v`) — previously a checkpoint was trusted purely on step-name
    presence, so re-running with a different database or a different
    min_seq_id silently reused stale results. The RepeatMasker
    short-ID-FASTA cache had the same problem (reused across a changed
    scaffold) and is now fingerprint-gated too.
 3. Genuine no-hit vs tool failure: MMseqs2 always creates its output TSV
    on success (even when it's empty) — a MISSING file after a "successful"
    run is now treated as a failure, not silently touch()'d into a fake
    empty success as v31 did. RepeatMasker does NOT create a .out file at
    all when zero repeats are found (documented behavior) — the module
    now checks the tool's own log for RepeatMasker's "No repetitive
    sequences were detected" message to tell a genuine zero-hit run apart
    from an actual crash, instead of always raising when .out is absent.
    Both engines' logs are also scanned for OOM/segfault/killed markers
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
    identity for MMseqs2, mean normalized Smith-Waterman score for
    RepeatMasker) — v31 parsed pident/score per hit and then discarded
    them; "hit_sig" only ever counted hit COUNT, never hit confidence.
    Feature/weight alignment is also now a checked invariant: the same
    _valid_te_mask() function decides "this contig has usable TE
    evidence" for both build_feature_matrix_v22_5d and compute_te_weight,
    and a post-hoc check raises if a zero-feature row and a zero-weight
    row ever disagree (previously the two functions used two different,
    only-approximately-matching validity conditions).
 8. Explicit masked/unmasked contract — see INPUT CONTRACT above and
    schema.json's "input.expected_variant" field.
 9. MMseqs2 TSV and RepeatMasker .out parsing now validate their column
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
    te_mode: fast           # fast | efficient
    funTEdb: /path/to/db    # FASTA file or directory
    threads: 64
    te_pa: 36               # RepeatMasker: pa×4=total cores
    te_frag: 20000000       # RepeatMasker: fragment size
    te_min_sw_score: 225    # RepeatMasker: minimum Smith-Waterman score
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
    repeatmasker_quality_reference_score: 1000.0
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
DEFAULT_TE_MODE               = "fast"

# V22_5D schema
TOTAL_V22_5D_DIM = 5
FIXED_FEATURE_NAMES = [
    "TE_richness",
    "AbsTE_sqrt",
    "DNA_ratio",
    "LTR_ratio",
    "Unclassified_ratio",
]

# RepeatMasker defaults
DEFAULT_MIN_SW_SCORE          = 225
DEFAULT_TE_PA                 = 36       # pa × 4 = total cores (36 × 4 = 144 on gmbs17)
DEFAULT_TE_FRAG               = 20_000_000

# MMseqs2 defaults
DEFAULT_MMSEQS_SENSITIVITY    = 5.7
DEFAULT_MMSEQS_MIN_SEQ_ID     = 0.7
DEFAULT_MMSEQS_COVERAGE       = 0.1
DEFAULT_MMSEQS_MAX_SEQ_LEN    = 100_000
DEFAULT_MMSEQS_EVALUE         = 1e-5

# Weight formula defaults
DEFAULT_FULL_CONFIDENCE_TE_BP = 5000
DEFAULT_FULL_CONFIDENCE_LEN   = 10_000
DEFAULT_FULL_CONFIDENCE_HITS  = 5
DEFAULT_WEIGHT_TE_COEF        = 0.55
DEFAULT_WEIGHT_HIT_COEF       = 0.25
DEFAULT_WEIGHT_QUALITY_COEF   = 0.20
DEFAULT_RM_QUALITY_REF_SCORE  = 1000.0
DEFAULT_UNCLASSIFIED_WARN_FRACTION = 0.95

# V33 changelog (sign-off review round)
# -----------------------------------------------------------------------
# 11. RepeatMasker .out class/family column: re-verified against
#     RepeatMasker's own documented example line (see parse_repeatmasker_out
#     docstring) -- parts[10] is confirmed correct (class/family); parts[9]
#     is the matching-repeat NAME, not class/family. No parsing change was
#     needed; the docstring now cites the verified example line so this
#     doesn't need re-litigating from memory again.
# 12. RepeatMasker checkpoint fingerprint now includes te_pa (previously
#     omitted -- changing it could silently reuse a stale checkpoint).
#     The v31-inherited `process_repeats_cores` parameter was removed
#     entirely (function signature, fingerprint, run_te_composition,
#     run_te_branch, config, CLI) -- verified against RepeatMasker's own
#     CLI help that ProcessRepeats (the post-processing step RepeatMasker
#     calls internally) has no cores/parallel option at all; -pa only
#     parallelizes the search batch. There was nothing for that parameter
#     to control, so fingerprinting it only forced pointless re-runs
#     without ever changing what actually ran. _checkpoint_ok for the
#     RepeatMasker step now also checks te_annotations.bed exists (matching
#     MMseqs2's [summary_path, hits_tsv] pattern) instead of only
#     summary_path.

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
    search tool mangled/truncated IDs (e.g. RepeatMasker's header-length
    truncation, or a whitespace-split mismatch) — silently dropping those
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
    0-100-ish, engine-normalized confidence unit (MMseqs2: percent
    identity as-is; RepeatMasker: SW score rescaled — see
    run_repeatmasker_efficient) — accumulated so compute_te_weight can use
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

def run_mmseqs2_search(
    scaffold: str, te_fasta: str, outdir: Path, threads: int,
    sensitivity: float, min_seq_id: float, coverage: float,
    max_seq_len: int, evalue: float, ckpt,
) -> Dict[str, Dict]:
    """
    Step 1 (fast mode): MMseqs2 nucleotide-nucleotide TE search.

    Searches assembled contigs against a TE consensus FASTA using MMseqs2
    easy-search in nucleotide mode (--search-type 3). Designed for large
    assemblies — ~20 min for 500k contigs at 64 threads.

    Key MMseqs2 flags:
      --search-type 3   : nucleotide vs nucleotide (required, not auto-detected)
      --max-seq-len     : handles long TE sequences (>64kb default limit)
      --cov-mode 2      : coverage of target only — partial TE fragments count
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
        sensitivity, min_seq_id, coverage, max_seq_len, evalue,
        mmseqs_version, VERSION,
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
        f"--cov-mode 2 "
        f"--min-aln-len 30 "
        f"-e {evalue} "
        f"--max-seqs 300 "
        f"--max-seq-len {max_seq_len} "
        f"--strand 2 "
        f'--format-output "query,target,pident,alnlen,mismatch,gapopen,'
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
                bases  = abs(qend - qstart)
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
            bed.write(f"{contig}\t{lo}\t{hi}\n")

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


# =============================================================================
# STEP 1B — EFFICIENT MODE: RepeatMasker
# =============================================================================

def make_repeatmasker_safe_fasta(scaffold: str, outdir: Path):
    """
    Create a short-ID FASTA for RepeatMasker.

    RepeatMasker truncates FASTA headers longer than 50 characters, which
    corrupts contig ID mapping. This function creates a safe copy with
    short IDs (ctg_00000001, ctg_00000002, ...) and saves the ID mapping
    to JSON for remapping after annotation.

    This cache is now fingerprint-gated on the scaffold (item 2) — v31
    reused repeatmasker_safe.fasta / repeatmasker_id_map.json purely
    based on file existence, so a changed scaffold in the same outdir
    would silently reuse a stale, wrong ID mapping.
    """
    safe_fasta  = outdir / "repeatmasker_safe.fasta"
    id_map_path = outdir / "repeatmasker_id_map.json"
    scaffold_fp = _file_fingerprint(scaffold)

    if safe_fasta.exists() and id_map_path.exists():
        with open(id_map_path) as f:
            cached = json.load(f)
        if cached.get("_scaffold_fp") == scaffold_fp:
            return str(safe_fasta), cached["id_map"]
        log.warning("repeatmasker_safe.fasta exists but the scaffold has changed since it was "
                    "built — rebuilding rather than reusing a stale ID mapping.")

    id_map = {}
    with open(safe_fasta, "w") as out_f:
        with open_gz(scaffold) as handle:
            for idx, rec in enumerate(SeqIO.parse(handle, "fasta"), 1):
                short_id = f"ctg_{idx:08d}"
                id_map[short_id] = rec.id
                rec.id = rec.name = short_id
                rec.description = ""
                SeqIO.write(rec, out_f, "fasta")
    with open(id_map_path, "w") as f:
        json.dump({"_scaffold_fp": scaffold_fp, "id_map": id_map}, f, indent=2)
    log.info(f"Safe FASTA: {len(id_map):,} contigs → {safe_fasta.name}")
    return str(safe_fasta), id_map


def find_repeatmasker_out(directory: Path) -> Optional[Path]:
    """Find RepeatMasker .out annotation file in output directory."""
    outs = [x for x in sorted(directory.glob("*.out"))
            if not x.name.endswith(".tbl")]
    return outs[0] if outs else None


def parse_repeatmasker_out(out_file: str, min_sw_score: int) -> List[Dict]:
    """
    Parse RepeatMasker .out file into list of hit dicts.

    RepeatMasker .out format (whitespace-delimited, 0-indexed columns):
      col 0 : Smith-Waterman score
      col 1 : % substitutions (perc div.)
      col 2 : % deleted bases (perc del.)
      col 3 : % inserted bases (perc ins.)
      col 4 : query contig name
      col 5 : hit start (position in query, begin)
      col 6 : hit end   (position in query, end)
      col 7 : position in query, left (bases past the match, in parens)
      col 8 : strand (+ or C)
      col 9 : matching repeat NAME (e.g. "L2c") — NOT the class/family
      col 10: repeat class/family (e.g. "LINE/L2", "LTR/Gypsy", "DNA/hAT")
      col 11+: position-in-repeat begin/end/left, unique ID

    Verified against RepeatMasker's own documented example output line
    (repeatmasker.org / Dfam-consortium RepeatMasker/repeatmasker.help):
        "207  18.4 18.4 0.0  Human  75437  75485  (924515)  C  L2c  LINE/L2  (48)  3339  3282  101"
    Splitting that line on whitespace: parts[9] == "L2c" (the repeat NAME),
    parts[10] == "LINE/L2" (the class/family). This confirms col 10 (not 9)
    is the correct index for class/family in the classic RepeatMasker .out
    format — a review round of this module raised col 9 as a candidate,
    but col 9 is the matching-repeat name, and reading class/family from it
    would misclassify every hit into whatever LTR/DNA/LINE/Unclassified
    bucket the specific repeat's NAME happens to keyword-match (or
    Unclassified, for names with no recognizable keyword) rather than its
    actual annotated class.

    Hits below min_sw_score are discarded (low-confidence alignments).
    Validates the column schema on the first data line and tracks a
    malformed-line ratio (item 9) rather than silently skipping every
    line and returning an empty-looking-genuine result.
    """
    hits = []
    n_data_lines = n_malformed = 0
    first_data_line_checked = False
    with open(out_file) as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.lower().startswith(
                    ("sw", "score", "there", "repeatmasker",
                     "matching", "query", "position")):
                continue
            parts = raw.split()

            if not first_data_line_checked:
                first_data_line_checked = True
                if len(parts) < 11:
                    raise RuntimeError(
                        f"{Path(out_file).name} does not have the expected >=11 whitespace-"
                        f"delimited columns (RepeatMasker .out format) on its first data line — "
                        f"got {len(parts)}. This looks like a schema mismatch (different "
                        f"RepeatMasker version/output format), not an isolated bad line. "
                        f"First data line: {raw[:200]!r}"
                    )

            n_data_lines += 1
            if len(parts) < 11:
                n_malformed += 1
                continue
            try:
                score  = int(float(parts[0]))
                contig = parts[4]
                start  = min(int(parts[5]), int(parts[6]))
                end    = max(int(parts[5]), int(parts[6]))
                cf     = parts[10]
            except (ValueError, IndexError):
                n_malformed += 1
                continue
            if score <= min_sw_score:
                continue
            hits.append({
                "score": score, "contig": contig,
                "start": start, "end": end,
                "bases": abs(end - start) + 1,
                "class_family": cf,
            })

    if n_data_lines > 0 and n_malformed / n_data_lines > 0.5:
        raise RuntimeError(
            f"{Path(out_file).name}: {n_malformed}/{n_data_lines} data lines were malformed "
            f"(>50%) — this looks like a schema mismatch, not scattered bad lines."
        )
    log.info(f"Parsed {len(hits):,} RepeatMasker hits (SW>{min_sw_score}), "
             f"{n_malformed:,}/{n_data_lines:,} malformed lines skipped")
    return hits


def run_repeatmasker_efficient(
    scaffold: str, te_fasta: str, outdir: Path, threads: int,
    min_sw_score: int, te_pa: int, te_frag: int, ckpt,
    quality_reference_score: float = DEFAULT_RM_QUALITY_REF_SCORE,
) -> Dict[str, Dict]:
    """
    Step 1 (efficient mode): RepeatMasker TE annotation.

    Uses RepeatMasker with any FASTA consensus library. Slower than MMseqs2
    (~10-24 hrs) but achieves ~95% sensitivity (near gold standard; this
    figure, like MMseqs2's ~75%, is an approximate, dataset-dependent
    estimate, not a measured constant).
    Recommended for benchmarking and small assemblies.

    Key optimisations applied:
      -pa te_pa       : parallel jobs (pa×4=cores, e.g. -pa 36 = 144 cores)
      -frag te_frag   : larger fragments reduce batch overhead
      -no_is          : skip bacterial IS elements (irrelevant for fungi)
      -norna          : skip RNA annotation (handled by skip_classes filter)
      -xsmall         : lowercase masked output (faster)

    CRITICAL: -pa × 4 = total CPU cores. Set te_pa=36 for 144-core gmbs17.
    Do NOT set te_pa = threads // 4 — always use te_pa directly from config.

    Short-ID FASTA is created first to avoid RepeatMasker header truncation.

    Item 3: RepeatMasker does NOT create a .out file when zero repeats are
    found across the entire input — that is documented behavior, not a
    failure. In that case it prints/logs "No repetitive sequences were
    detected". The absence of a .out file is only treated as a tool
    failure when that message is ALSO absent from the log.

    Output:
      te_annotations.bed     : BED-format hit coordinates
      te_contig_summary.json : aggregated per-contig TE statistics

    Returns:
      summary : dict {contig_id: {ltr_bp, dna_bp, ...}}
    """
    STEP    = "te_repeatmasker_search"
    hit_dir = outdir / "01_repeatmasker"
    hit_dir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "te_contig_summary.json"
    log_file     = str(hit_dir / "repeatmasker.log")

    rm_version = _tool_version(["RepeatMasker", "-v"])
    fp = _fingerprint(
        _file_fingerprint(scaffold), _file_fingerprint(te_fasta),
        min_sw_score, te_pa, te_frag,
        quality_reference_score, rm_version, VERSION,
    )
    bed_path_ck = str(outdir / "te_annotations.bed")
    cached = _checkpoint_ok(ckpt, STEP, fp, [str(summary_path), bed_path_ck])
    if cached is not None:
        log.info("[checkpoint] RepeatMasker done — loading")
        with open(summary_path) as f:
            raw = json.load(f)
        return _deserialise_summary(raw)

    t0 = time.time()
    log_step_start(log, 1, "RepeatMasker TE search (efficient mode)", 5)

    if not check_tool("RepeatMasker"):
        raise RuntimeError(
            "RepeatMasker not found.\n"
            "conda install -c bioconda repeatmasker -n hyphaes -y"
        )

    safe_fasta, id_map = make_repeatmasker_safe_fasta(scaffold, hit_dir)
    pa_value = max(1, te_pa)

    log.info(f"RepeatMasker: -pa {pa_value} (={pa_value*4} cores)  "
             f"-frag {te_frag:,}  -no_is -norna -xsmall  version={rm_version}")

    cmd = (
        f"RepeatMasker "
        f"-pa {pa_value} "
        f"-frag {te_frag} "
        f"-no_is "
        f"-norna "
        f"-xsmall "
        f"-lib {te_fasta} "
        f"-dir {hit_dir} "
        f"{safe_fasta}"
    )
    run_cmd(cmd, log_file)
    _scan_log_for_failure_markers(log_file, "RepeatMasker")

    out_file = find_repeatmasker_out(hit_dir)
    if out_file is None:
        log_text = ""
        try:
            log_text = Path(log_file).read_text(errors="ignore")
        except FileNotFoundError:
            pass
        if re.search(r"no repetitive sequences were detected", log_text, re.IGNORECASE):
            log.warning("RepeatMasker completed successfully and found genuinely ZERO "
                        "repetitive sequences across the assembly — RepeatMasker does not "
                        "write a .out file in this case (documented behavior), this is a "
                        "valid result, not a tool failure.")
            summary = {}
            (outdir / "te_annotations.bed").touch()
            _finalize_summary(summary)
            with open(summary_path, "w") as f:
                json.dump(serialise_summary(summary), f, indent=2)
            if ckpt:
                ckpt.mark_done(STEP, {"_fp": fp, "n_hits": 0, "n_with_te": 0,
                                       "rm_version": rm_version, "genuine_zero_hits": True})
            log_step_done(log, 1, "RepeatMasker TE search (zero hits)", time.time() - t0)
            return summary
        raise RuntimeError(
            f"No RepeatMasker .out file found in {hit_dir}, and the log does not contain "
            f"RepeatMasker's documented 'No repetitive sequences were detected' message — this "
            f"looks like a genuine tool failure (crash, bad -lib path, disk issue), not a "
            f"zero-hit result. Check {log_file}."
        )

    raw_hits = parse_repeatmasker_out(str(out_file), min_sw_score)

    # Remap short IDs back to original contig IDs
    for h in raw_hits:
        h["contig"] = id_map.get(h["contig"], h["contig"])

    summary  = {}
    bed_path = outdir / "te_annotations.bed"
    n_skip   = 0

    with open(bed_path, "w") as bed:
        for h in raw_hits:
            te_class = classify_te_name(h["class_family"])
            if te_class == "Skip":
                n_skip += 1
                continue
            quality = min(100.0, (h["score"] / quality_reference_score) * 100.0)
            accumulate_hit(summary, h["contig"], h["bases"], te_class,
                           h["start"], h["end"], quality=quality)
            bed.write(f"{h['contig']}\t{h['start']}\t{h['end']}\n")

    log.info(f"RepeatMasker: {len(raw_hits):,} hits  {n_skip:,} RNA skipped")
    _finalize_summary(summary)
    log_summary_stats(summary, "RepeatMasker")

    with open(summary_path, "w") as f:
        json.dump(serialise_summary(summary), f, indent=2)

    if ckpt:
        ckpt.mark_done(STEP, {
            "_fp": fp,
            "n_hits":    len(raw_hits),
            "n_with_te": len(summary),
            "rm_version": rm_version,
        })

    log_step_done(log, 1, "RepeatMasker TE search", time.time() - t0)
    return summary


# =============================================================================
# STEP 2 — BUILD V22_5D FEATURE MATRIX
# =============================================================================

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
                 quality (MMseqs2: percent identity; RepeatMasker:
                 normalized SW score), clipped to [0,1]
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
    te_mode       : str,
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
    search_engine = "MMseqs2" if te_mode == "fast" else "RepeatMasker"

    schema: Dict[str, Any] = {
        "version"        : VERSION,
        "te_mode"        : te_mode,
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
        "modes": {
            "fast"     : "MMseqs2 nucl-nucl  ~20 min / 500k contigs  ~75% sensitivity (approx., dataset-dependent)",
            "efficient": "RepeatMasker       ~10-24 hrs               ~95% sensitivity (approx., dataset-dependent)",
        },
        "outputs": {
            "te_features.npy"      : f"(N × {feature_dim}) float32 — V22_5D features",
            "te_weight.npy"        : "(N × 1) float32 — TE confidence weights [0,1]",
            "contig_ids.json"      : "ordered contig IDs (matches .npy row order)",
            "schema.json"          : "this file — full metadata",
            "te_annotations.bed"   : "BED-format hit coordinates",
            "mmseqs_te_hits.tsv"   : "raw MMseqs2 output (fast mode only)",
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
    te_mode              : str   = DEFAULT_TE_MODE,
    threads              : int   = DEFAULT_THREADS,
    resume               : bool  = True,
    # RepeatMasker params
    min_sw_score         : int   = DEFAULT_MIN_SW_SCORE,
    te_pa                : int   = DEFAULT_TE_PA,
    te_frag              : int   = DEFAULT_TE_FRAG,
    # MMseqs2 params
    mmseqs_sensitivity   : float = DEFAULT_MMSEQS_SENSITIVITY,
    mmseqs_min_seq_id    : float = DEFAULT_MMSEQS_MIN_SEQ_ID,
    mmseqs_coverage      : float = DEFAULT_MMSEQS_COVERAGE,
    mmseqs_max_seq_len   : int   = DEFAULT_MMSEQS_MAX_SEQ_LEN,
    mmseqs_evalue        : float = DEFAULT_MMSEQS_EVALUE,
    # Feature-transform / weight params (item 6 / 7)
    apply_length_weight_to_features: bool = False,
    full_confidence_te_bp   : float = DEFAULT_FULL_CONFIDENCE_TE_BP,
    full_confidence_length  : float = DEFAULT_FULL_CONFIDENCE_LEN,
    full_confidence_hits    : float = DEFAULT_FULL_CONFIDENCE_HITS,
    weight_te_coef          : float = DEFAULT_WEIGHT_TE_COEF,
    weight_hit_coef         : float = DEFAULT_WEIGHT_HIT_COEF,
    weight_quality_coef     : float = DEFAULT_WEIGHT_QUALITY_COEF,
    repeatmasker_quality_reference_score: float = DEFAULT_RM_QUALITY_REF_SCORE,
    unclassified_warn_fraction: float = DEFAULT_UNCLASSIFIED_WARN_FRACTION,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unified TE composition V33 — V22_5D schema.

    Runs the full TE annotation + feature extraction pipeline:
      Step 1 : TE search (MMseqs2 or RepeatMasker)
      Step 2 : Build V22_5D feature matrix (N × 5)
      Step 3 : Compute TE confidence weights (N × 1)
      Step 4 : Save all outputs

    Args:
        scaffold         : path to the UNMASKED assembly FASTA (see module
                            docstring INPUT CONTRACT — NOT final_clean.fasta)
        outdir           : output directory
        funtedb          : TE database FASTA or directory
        te_mode          : 'fast' (MMseqs2) | 'efficient' (RepeatMasker)
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

    te_mode = str(te_mode).strip().lower()
    if te_mode not in ("fast", "efficient"):
        log.warning(f"Unknown te_mode='{te_mode}' → defaulting to 'fast'")
        te_mode = "fast"

    log.info("")
    log.info("╔" + "═"*68 + "╗")
    log.info("║" + " "*10 + "TE COMPOSITION V33 — UNIFIED MODULE" + " "*23 + "║")
    log.info("╚" + "═"*68 + "╝")
    log.info(f"  te_mode  : {te_mode.upper()} "
             f"({'MMseqs2 ~20min/500k' if te_mode=='fast' else 'RepeatMasker ~10-24hrs'})")
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
    if te_mode == "fast":
        summary = run_mmseqs2_search(
            scaffold=scaffold, te_fasta=te_fasta, outdir=outdir,
            threads=threads, sensitivity=mmseqs_sensitivity,
            min_seq_id=mmseqs_min_seq_id, coverage=mmseqs_coverage,
            max_seq_len=mmseqs_max_seq_len, evalue=mmseqs_evalue, ckpt=ckpt,
        )
    else:
        summary = run_repeatmasker_efficient(
            scaffold=scaffold, te_fasta=te_fasta, outdir=outdir,
            threads=threads, min_sw_score=min_sw_score, te_pa=te_pa,
            te_frag=te_frag,
            ckpt=ckpt, quality_reference_score=repeatmasker_quality_reference_score,
        )

    # Item 1: fail loud on any hit landing on a contig ID the assembly
    # doesn't have, instead of silently dropping it.
    _validate_hit_ids_known(summary, contig_ids, source=f"{te_mode} search")
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
        te_mode=te_mode,
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
    te_mode = str(config.get("te_mode", DEFAULT_TE_MODE))
    threads = int(config.get("threads", DEFAULT_THREADS))

    te_outdir = Path(outdir) / "te_composition"
    te_feat, te_weight, schema = run_te_composition(
        scaffold=scaffold, outdir=str(te_outdir), funtedb=funtedb,
        te_mode=te_mode, threads=threads, resume=True,
        min_sw_score         = int(config.get("te_min_sw_score",          DEFAULT_MIN_SW_SCORE)),
        te_pa                = int(config.get("te_pa",                    DEFAULT_TE_PA)),
        te_frag              = int(config.get("te_frag",                  DEFAULT_TE_FRAG)),
        mmseqs_sensitivity   = float(config.get("mmseqs_sensitivity",     DEFAULT_MMSEQS_SENSITIVITY)),
        mmseqs_min_seq_id    = float(config.get("mmseqs_min_seq_id",      DEFAULT_MMSEQS_MIN_SEQ_ID)),
        mmseqs_coverage      = float(config.get("mmseqs_coverage",        DEFAULT_MMSEQS_COVERAGE)),
        mmseqs_max_seq_len   = int(config.get("mmseqs_max_seq_len",       DEFAULT_MMSEQS_MAX_SEQ_LEN)),
        mmseqs_evalue        = float(config.get("mmseqs_evalue",          DEFAULT_MMSEQS_EVALUE)),
        apply_length_weight_to_features = bool(config.get("apply_length_weight_to_features", False)),
        full_confidence_te_bp   = float(config.get("full_confidence_te_bp",   DEFAULT_FULL_CONFIDENCE_TE_BP)),
        full_confidence_length  = float(config.get("full_confidence_length",  DEFAULT_FULL_CONFIDENCE_LEN)),
        full_confidence_hits    = float(config.get("full_confidence_hits",    DEFAULT_FULL_CONFIDENCE_HITS)),
        weight_te_coef          = float(config.get("weight_te_coef",          DEFAULT_WEIGHT_TE_COEF)),
        weight_hit_coef         = float(config.get("weight_hit_coef",         DEFAULT_WEIGHT_HIT_COEF)),
        weight_quality_coef     = float(config.get("weight_quality_coef",     DEFAULT_WEIGHT_QUALITY_COEF)),
        repeatmasker_quality_reference_score = float(config.get(
            "repeatmasker_quality_reference_score", DEFAULT_RM_QUALITY_REF_SCORE)),
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
            "  te_mode: fast      → MMseqs2     ~20 min / 500k contigs  ~75% sensitivity (approx.)\n"
            "  te_mode: efficient → RepeatMasker ~10-24 hrs              ~95% sensitivity (approx.)\n"
            "  Sensitivity figures are dataset- and parameter-dependent approximations, not\n"
            "  measured constants — see module docstring MODES section.\n"
            "  Output: V22_5D (5D) — TE_richness, AbsTE_sqrt, DNA_ratio, LTR_ratio, Unclassified_ratio\n"
            "  Works with any nucleotide FASTA database.\n"
            "  Scaffold must be the UNMASKED assembly (see module docstring).\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scaffold",  required=True,  help="UNMASKED assembly FASTA")
    p.add_argument("--funtedb",   required=True,  help="TE database FASTA or directory")
    p.add_argument("--outdir",    required=True,  help="Output directory")
    p.add_argument("--te-mode",   default="fast", choices=["fast", "efficient"])
    p.add_argument("--threads",   type=int, default=DEFAULT_THREADS)
    p.add_argument("--no-resume", action="store_true")

    rm = p.add_argument_group("RepeatMasker (efficient mode only)")
    rm.add_argument("--min-sw-score",          type=int,   default=DEFAULT_MIN_SW_SCORE)
    rm.add_argument("--te-pa",                 type=int,   default=DEFAULT_TE_PA,
                    help="pa×4=total cores (default 36 → 144 cores on gmbs17)")
    rm.add_argument("--te-frag",               type=int,   default=DEFAULT_TE_FRAG)
    rm.add_argument("--repeatmasker-quality-reference-score", type=float,
                    default=DEFAULT_RM_QUALITY_REF_SCORE)

    mm = p.add_argument_group("MMseqs2 (fast mode only)")
    mm.add_argument("--mmseqs-sensitivity", type=float, default=DEFAULT_MMSEQS_SENSITIVITY)
    mm.add_argument("--mmseqs-min-seq-id",  type=float, default=DEFAULT_MMSEQS_MIN_SEQ_ID)
    mm.add_argument("--mmseqs-coverage",    type=float, default=DEFAULT_MMSEQS_COVERAGE)
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
        te_mode=args.te_mode, threads=args.threads, resume=not args.no_resume,
        min_sw_score=args.min_sw_score, te_pa=args.te_pa,
        te_frag=args.te_frag,
        mmseqs_sensitivity=args.mmseqs_sensitivity,
        mmseqs_min_seq_id=args.mmseqs_min_seq_id,
        mmseqs_coverage=args.mmseqs_coverage,
        mmseqs_max_seq_len=args.mmseqs_max_seq_len,
        mmseqs_evalue=args.mmseqs_evalue,
        apply_length_weight_to_features=args.apply_length_weight_to_features,
        full_confidence_te_bp=args.full_confidence_te_bp,
        full_confidence_length=args.full_confidence_length,
        full_confidence_hits=args.full_confidence_hits,
        weight_te_coef=args.weight_te_coef,
        weight_hit_coef=args.weight_hit_coef,
        weight_quality_coef=args.weight_quality_coef,
        repeatmasker_quality_reference_score=args.repeatmasker_quality_reference_score,
        unclassified_warn_fraction=args.unclassified_warn_fraction,
    )




# PROFILE HOOKS v1 — observation only; scientific functions above are unchanged.
from hyphaesbin.utils.resource_profiler import profile_functions, profile_checkpoint
profile_functions(globals(), ['run_te_branch', 'run_te_composition', 'run_mmseqs2_search', 'run_repeatmasker_efficient', 'build_feature_matrix_v22_5d', 'run_cmd'])
_checkpoint_ok = profile_checkpoint(_checkpoint_ok)

if __name__ == "__main__":
    main()
