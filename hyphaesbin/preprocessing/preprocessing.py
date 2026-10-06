"""
HyphaeSBin Preprocessing — Module 2 (REDESIGNED)
═══════════════════════════════════════════════════════════════════════════════

WORKFLOW IN SHORT
─────────────────
    Validate → Merge(+ID map) → Stats → Length pre-filter → skani cross-sample dedup →
    Classify (Tiara + Whokaryote) → Domain removal → rDNA mask (dual output) →
    Single mapping pass → Adaptive filter (+reference rescue) →
    Multi-signal scoring → Subset coverage table → QC reports

    13 steps. ONE mapping pass (was 2). Classification-driven contaminant
    removal (was: coverage-threshold + optional Kraken2). No silent drops —
    every removed contig is written to a labelled file with a reason.

ROLE OF EACH STEP
─────────────────
     1. VALIDATE            Check FASTA/FASTQ integrity, sample table, tools.
     2. MERGE                Combine per-sample assemblies. Sample-prefix every
                              ID, flag duplicate sequences, build id_map.tsv —
                              the ORIGINAL id ⇄ SAFE internal id lookup that
                              every later step and every external tool call
                              uses, so spaces/pipes/colons in real-world FASTA
                              headers can never break a shell command again.
     3. STATS                N50, L50, length distribution, counts <1/2/5kb —
                              a report, never an automatic reason to delete.
     4. LENGTH PRE-FILTER     Drop contigs below min_contig_length (default 1kb)
                              before the expensive steps below. Cheap, one-line,
                              saves real compute time.
     5. DEDUP (skani)         (runs AFTER step 4) Remove contigs from one sample
                              that are contained in a longer contig from a
                              DIFFERENT sample (ANI >= 99.5%, shorter-contig AF
                              >= 95%). Same-sample contigs are never removed.
                              STOPS (does not silently skip) on a single-sample
                              input or above the contig cap.
                              NOTE: internal names/checkpoints keep the historic
                              labels (step4_dedup, step5_length_prefilter,
                              04_dedup/, 05_length_prefilter/); only the DISPLAYED
                              step numbers were swapped to match the real order.
     6. CLASSIFY              Tiara (organelle / domain, k-mer based) +
                              Whokaryote (eukaryote/prokaryote, gene-structure
                              based, Tiara-integrated "T" model). Every contig
                              gets a category + a confidence score. Nothing is
                              deleted in this step — it only labels.
     7. DOMAIN REMOVAL        Uses step 6's labels: drop HIGH-CONFIDENCE
                              bacterial / archaeal / organelle contigs only.
                              Fungi, other eukaryotes, and anything the
                              classifiers couldn't call confidently ("Unknown")
                              are ALWAYS kept at this step.
     8. rDNA MASKING          barrnap finds rRNA genes. Their coordinates are
                              masked with 'N' in a *copy* of the assembly —
                              the original, unmasked sequence is preserved
                              alongside it. Nothing is ever dropped for
                              carrying an rRNA gene.
     9. SINGLE MAPPING PASS   minimap2 + CoverM, run ONCE, on the classified /
                              filtered UNMASKED assembly (rRNA loci are only
                              N-masked in the composition/TNF copy — reads
                              must still map to their real bases, or coverage
                              over rRNA-bearing contigs would be silently
                              suppressed). Every later coverage decision reads
                              from this one table — there is no second
                              mapping pass anywhere in this pipeline.
    10. ADAPTIVE FILTER       ≥ceiling (default 2x min length): kept outright. Below: kept if covered in
                              ≥2 samples. Below-ceiling contigs covered in only ONE sample
                              (the case that would previously kill a rare,
                              patchily-detected fungal contig) gets a RESCUE
                              CHECK instead of automatic removal: minimap2
                              against a fungal reference set. A hit → kept,
                              flagged. No hit → removed (see limitations in
                              the docstring of `step10_adaptive_filter`).
    11. MULTI-SIGNAL SCORING  Replaces the old flat "mito if coverage >5×
                              median" rule. Combines classification
                              confidence, coverage breadth/depth/variance,
                              TNF consistency, and GC/length outlier status
                              into one of: retain / retain_with_warning /
                              uncertain / remove_high_confidence_contaminant.
    12. SUBSET COVERAGE       Filters step 9's ONE coverage table down to the
                              survivors — no remapping. Every retained FASTA
                              ID is checked against the table 1:1; mismatches
                              are reported, never silently dropped.
    13. QC REPORTS            Writes final_clean.fasta (+ unmasked variant),
                              coverage_table.tsv, contig_classification.tsv,
                              contig_decisions.tsv, id_map.tsv, per-category
                              removed-ID lists, and qc_summary.json. Every
                              contig that ever existed is traceable to exactly
                              which step removed it and why.

WHAT CHANGED FROM THE OLD MODULE
─────────────────────────────────
    • Removed: Kraken2 stage (was optional/off by default anyway; fully gone
      now — Whokaryote+Tiara replace it as the actual classifier).
    • Removed: second mapping pass (step 8 + step 12 in the old numbering are
      now one step, step 9 here).
    • Removed: flat mito-removal-by-coverage rule (folded into step 11's
      multi-signal scoring instead).
    • Removed: ~1900 lines of commented-out legacy duplicate code.
    • Added: ID normalization / safe internal ID map (step 2).
    • Added: Tiara + Whokaryote classification (step 6) — did not exist
      before at all.
    • Added: reference-rescue branch for single-sample mid-length contigs
      (step 10) — KNOWN LIMITATION: only rescues contigs resembling something
      already in the fungal reference set; genuinely novel fungi are not
      recoverable by this branch alone (see step 10's docstring).
    • Added: full QC/reporting suite (step 13) — was a single-row TSV before.

CONFIG PARAMETERS — DEFAULTS HIGHLIGHTED BELOW
────────────────────────────────────────────────
    See `DEFAULT_CONFIG` immediately below this docstring for every tunable
    parameter and its default value, grouped by which step uses it.
"""

import os
import re
import sys
import time
import json
import gzip
import shutil
import shlex
import random
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from Bio import SeqIO

# Import project utilities. NOTE: this file lives at
# hyphaesbin/preprocessing/preprocessing.py — the package-qualified import
# below matches every other module in this pipeline (coverage.py, tnf_gene.py,
# te_composition.py, encoder.py, clustering.py all use
# `from hyphaesbin.utils....`). An earlier revision of this file used a
# `sys.path.insert` + bare `from utils.logger import ...` hack instead, which
# does not resolve correctly when this file is imported as
# `hyphaesbin.preprocessing.preprocessing` (main.py's actual import path) —
# fixed here rather than carried forward.
from hyphaesbin.utils.logger import get_logger, log_step_start, log_step_done
from hyphaesbin.utils.checkpoint import Checkpoint

log = get_logger("hyphaesbin.preprocessing")

TOTAL_STEPS = 13
LOGIC_VERSION = "2026-10-05-euk-focus-v4"


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIG  (every value here can be overridden via the `config` dict
# passed to `run_preprocessing()`, or via CLI flags — see the bottom of this
# file for the argparse wiring)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG: Dict = {
    # ── general ──────────────────────────────────────────────────────────
    "threads": 80,
    "read_type": "auto",                 # auto / short / hifi / ont / hybrid
    "read_type_detect_seed": 0,          # seeds the random sample of FASTQ
                                          # files used for read-type
                                          # auto-detection, so re-running on
                                          # the same input can't silently
                                          # pick a different subset (and
                                          # therefore a different detected
                                          # read_type) between runs
    "auto_install_tools": False,         # if a required tool (skani, seqkit, minimap2,
                                          # samtools, coverm, barrnap, tiara, whokaryote)
                                          # is missing from PATH: False (default) reports
                                          # it in step1's errors and stops, so you install
                                          # the exact versions you want; True lets the
                                          # pipeline attempt `conda install` on your behalf.

    # ── displayed step 4: length pre-filter (internal name: step5_length_prefilter) ──
    "min_contig_length": 1000,           # hard floor, applied before mapping

    # ── displayed step 5: skani dedup (internal name: step4_dedup; v3 cross-sample CONTAINMENT) ──
    # Runs AFTER the length pre-filter (displayed step 4). Only contigs >= dedup_min_length
    # are compared; shorter contigs pass through untouched.
    "dedup_ani": 99.5,                   # % ANI to call two contigs "the same"
    "dedup_min_af": 95.0,                # min aligned fraction (%) of the SHORTER
                                          # contig -> containment. Two assemblies of
                                          # the same genome have different contig
                                          # boundaries, so a short contig sitting
                                          # inside a longer one must count.
    "dedup_min_length_ratio": 0.0,       # 0 = disabled (a ratio guard defeats
                                          # containment). Set e.g. 0.8 for the old
                                          # strict behaviour.
    "dedup_min_length": 2000,            # only compare contigs >= this (skani is
                                          # unreliable on very short sequences)
    "dedup_cross_sample_only": True,     # never remove within-sample contigs
                                          # (protects real paralogs / repeats)
    "dedup_skani_args": "-c 30 -m 200 --faster-small -s 95",
    "skip_dedup": False,                 # ablation switch: True = no dedup at all
    "dedup_max_contigs": 500_000,        # cap on contigs actually compared
    "dedup_allow_skip_over_cap": False,
    "assembly_sample_regex": "",         # ONLY for a single merged assembly FASTA whose
                                          # headers carry the sample: a regex with ONE capture
                                          # group applied to each original contig ID, e.g.
                                          # '^(s\\d+)_'. Needed for cross-sample dedup, because
                                          # the sample label otherwise comes from the FASTA
                                          # FILE name (one file = one sample). Every header
                                          # must match or the run stops. Ignored (with a
                                          # warning) when --scaffold is a directory.  # False = STOP if cap exceeded (no silent
                                          # skip); True = warn and skip dedup

    # ── step 9: mapping BAMs ────────────────────────────────────────────
    "keep_bams": True,                   # True = keep the sorted, indexed per-sample BAMs
                                          # (+ .bai) in the step-9 mapping folder after the
                                          # run. False = delete them after step 12 (old
                                          # behaviour, frees disk). They are aligned to the
                                          # step-8 UNMASKED assembly (all contigs that passed
                                          # steps 1-8), not only the final retained set.

    # ── step 6: classification (Tiara + Whokaryote) ─────────────────────
    "tiara_min_len": 1000,               # Tiara's own --min_len (align w/ step5)
    "tiara_prob_cutoff": [0.65, 0.65],   # Tiara's [-p] stage1/stage2 cutoffs
    "whokaryote_minsize": 1000,          # Whokaryote's --minsize (align w/ step5)
    "whokaryote_model": "T",             # "T" = Tiara-integrated (recommended)
    "classification_confidence_high": 0.80,   # ≥ this = "high confidence" domain
                                          # removal — applies ONLY to rows with
                                          # confidence_type == 'probability'
                                          # (a real Tiara probability). Rows
                                          # with confidence_type == 'rule_based'
                                          # are NEVER decided by a number —
                                          # see _RULE_BASED_HIGH_CONF_SOURCES
                                          # below step 6, a discrete allow-list
                                          # instead of a second numeric bar.
    "rule_based_high_conf_sources": None,  # None = use the default allow-list
                                          # (_RULE_BASED_HIGH_CONF_SOURCES:
                                          # {"tiara_organelle", "whokaryote+tiara"}).
                                          # Override with a set/list of 'source'
                                          # values to change which rule-based
                                          # calls step 7 treats as removable —
                                          # e.g. drop "tiara_organelle" from the
                                          # set if a run should retain organelle
                                          # contigs rather than remove them.

    # ── step 8: rDNA masking ─────────────────────────────────────────────
    "prokaryote_removal_enabled": True,   # step 7: remove contigs that Tiara AND Whokaryote both call prokaryotic
                                          # (bacteria / archaea / prokarya) and that are >= prokaryote_min_length
    "prokaryote_min_length": 3000,        # Tiara's recommended minimum length. On CAMISIM, calls below this were mostly
                                          # fungal (about 2.5 Mb fungal vs 0.04 Mb bacterial removed) -> kept + flagged
    "organelle_removal_enabled": False,   # False: plastid/mitochondrion calls are FLAGGED, not removed. CAMISIM has no
                                          # plastids, yet 1,050 contigs (4.76 Mb, 88% Rhizophagus nuclear) were removed as
                                          # 'plastid'; on Zostera only 6 of 1,372 plastid calls align to Z. marina organelles
    "adaptive_cov_stat": "sample",   # step 10: "sample" = ONE sample must meet BOTH adaptive_min_cov and adaptive_min_breadth
                                          # (a genome present in one of N samples is not diluted by the empty samples);
                                          # "max" = separate maxima (can mix depth and breadth of different samples); "mean" = legacy
    "reuse_classification_outputs": False,  # step 6: reuse existing Tiara/Whokaryote outputs WITHOUT fingerprint check
                                          # (only if they came from exactly this input FASTA + parameters)
    "barrnap_kingdom": "fun",            # barrnap --kingdom. Valid values are
                                          # barrnap-BUILD-dependent: some builds
                                          # accept euk/bac/arc/mito, others
                                          # (confirmed live: bioconda barrnap in
                                          # the "whokaryote" conda env) only
                                          # accept bac/arc/fun and reject "euk"
                                          # outright ([barrnap] ERROR: Invalid
                                          # --kingdom 'euk'). "fun" is barrnap's
                                          # fungus-specific rRNA model, which is
                                          # also the scientifically correct
                                          # choice for this fungi-specific
                                          # pipeline — not just a fallback. Run
                                          # `barrnap --help` in the target env
                                          # to confirm accepted values before
                                          # overriding.

    # ── step 9: single mapping pass ──────────────────────────────────────
    "max_mapping_workers": "auto",       # how many samples to map CONCURRENTLY.
                                          # "auto" = min(n_samples, threads //
                                          # min_threads_per_sample), so `threads`
                                          # is split across a few samples at once
                                          # instead of one sample using all of
                                          # them while the rest wait in a queue.
                                          # Set to 1 to force the old fully-
                                          # sequential behavior, or an int to
                                          # pick the worker count yourself.
                                          # CAVEAT: each concurrent samtools
                                          # sort uses its own memory buffer, so
                                          # peak RAM scales roughly with worker
                                          # count — lower this on a shared or
                                          # memory-constrained machine.
    "min_threads_per_sample": 8,         # used by "auto" above — floor on how
                                          # few threads a single sample's
                                          # minimap2/samtools gets before auto-
                                          # sizing stops adding more concurrent
                                          # workers.

    # ── step 10: adaptive filter + reference rescue ─────────────────────
    "adaptive_gray_ceiling": None,       # step 10: contigs >= this are kept outright; contigs
                                          # between min_contig_length and this ("gray zone")
                                          # need coverage/sample/breadth support. None = legacy
                                          # 2 x min_contig_length (= 2000 at the default 1000).
                                          # Set automatically by analysis_min_length in main.py.
    "adaptive_min_cov": 2.0,             # min mean coverage, 1-2kb gray zone
    "adaptive_min_samples": 2,           # min samples with coverage >0
    "adaptive_min_breadth": 0.3,         # min mean covered-fraction (breadth);
                                          # skipped gracefully if CoverM's
                                          # breadth columns aren't identified
    "rescue_enabled": True,
    "rescue_reference_fasta": "",        # REQUIRED if rescue_enabled=True —
                                          # e.g. a fungal genome / ITS reference
                                          # collection (UNITE, RefSeq fungi, ...)
    "rescue_min_identity": 75.0,         # % identity to accept a rescue hit
    "rescue_min_query_cov": 0.5,         # fraction of contig length aligned

    # ── step 11: multi-signal scoring ───────────────────────────────────
    "scoring_cov_cv_high": 1.5,          # coverage coefficient-of-variation
                                          # above this = "uncertain" evidence
    "scoring_gc_zscore_flag": 3.0,       # |GC z-score| above this = outlier
    "scoring_tnf_dist_flag": 2.5,        # |TNF z-distance| from assembly mean
                                          # above this = composition outlier

    # TNF here is a LIGHTWEIGHT, QC-ONLY signal — one more piece of weak
    # evidence step 11 can combine with coverage/GC/classification to flag a
    # contig 'uncertain'. It is NOT the pipeline's canonical TNF feature for
    # the β-VAE encoder — that is a separate module's job (e.g.
    # hyphaesbin/composition/TNF_gene/tnf_gene.py), computed once on
    # final_clean.fasta, with its own performance/definition choices. Running
    # two independent 136-dim TNF implementations against the same contigs
    # risks the two silently diverging (different k-mer counting edge cases,
    # different pseudocounts, different normalization) — set
    # 'tnf_vector_fn' below to point this step at the encoder module's exact
    # function instead of maintaining a second implementation here.
    "enable_tnf_qc_signal": True,        # False = skip TNF entirely in step 11
                                          # (no compute cost, no tnf_zscore
                                          # warning) — use this if TNF QC is
                                          # handled entirely by the dedicated
                                          # TNF module instead
    "tnf_vector_fn": None,               # optional callable: seq:str -> 136-
                                          # dim np.ndarray, canonical-tetramer-
                                          # ordered. Pass the encoder module's
                                          # actual function here (e.g.
                                          # `from hyphaesbin.composition.TNF_gene.tnf_gene
                                          #  import compute_tnf_vector` then
                                          # `config["tnf_vector_fn"] = compute_tnf_vector`)
                                          # to make step 11 use the SAME TNF
                                          # definition as the encoder rather
                                          # than this file's own copy. None =
                                          # fall back to this file's local
                                          # _compute_tnf_vector (kept only so
                                          # this script still runs standalone
                                          # without the full package tree).
}


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE TERMINAL UI  (stdlib only — no extra install; ANSI codes degrade
# gracefully to plain text on terminals that don't support them)
# ═══════════════════════════════════════════════════════════════════════════════

_STEP_NAMES = [
    "Validate inputs",
    "Merge assemblies (+ ID map)",
    "Assembly statistics",
    "Length pre-filter",
    "Deduplication (skani)",
    "Classification (Tiara + Whokaryote)",
    "Domain-based removal",
    "rDNA masking (barrnap)",
    "Single mapping pass",
    "Adaptive filter (+ reference rescue)",
    "Multi-signal contig scoring",
    "Subset coverage table",
    "QC reports",
]

_C_RESET = "\033[0m"
_C_DIM = "\033[2m"
_C_BOLD = "\033[1m"
_C_GREEN = "\033[32m"
_C_YELLOW = "\033[33m"
_C_CYAN = "\033[36m"
_C_RED = "\033[31m"

_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_C_RESET}" if _USE_COLOR else text


def _apply_step_labels(cfg: Dict) -> None:
    """Make the banner/log names reflect the ACTUAL thresholds and modes in
    use (they used to be hard-coded, e.g. '≥1kb', and went stale as soon as
    min_contig_length / dedup settings changed). Display numbering: 4 = length
    pre-filter, 5 = dedup (the real execution order)."""
    min_len = int(cfg.get("min_contig_length", 1000))
    _STEP_NAMES[3] = f"Length pre-filter (≥{min_len:,} bp)"
    if cfg.get("skip_dedup", False):
        _STEP_NAMES[4] = "Deduplication (skipped by config)"
    else:
        scope = "cross-sample" if cfg.get("dedup_cross_sample_only", True) else "all-vs-all"
        _STEP_NAMES[4] = (f"Dedup (skani, {scope}, ≥{int(cfg.get('dedup_min_length', 2000)):,} bp, "
                          f"ANI≥{cfg.get('dedup_ani', 99.5)})")


def print_workflow_banner(current_step: int = 0):
    """Print the full 13-step workflow with the current step highlighted.
    Called once at pipeline start, and re-printed at the top of every step
    so a user watching the terminal always sees where they are in the whole
    pipeline, not just the current step's log lines."""
    width = 78
    print()
    print(_c("┌" + "─" * (width - 2) + "┐", _C_CYAN))
    title = "HYPHAESBIN PREPROCESSING — 13-STEP WORKFLOW"
    print(_c(f"│{title.center(width - 2)}│", _C_CYAN))
    print(_c("├" + "─" * (width - 2) + "┤", _C_CYAN))
    for i, name in enumerate(_STEP_NAMES, start=1):
        if i < current_step:
            mark = _c("✔", _C_GREEN)
            label = _c(f"{i:>2}. {name}", _C_DIM)
        elif i == current_step:
            mark = _c("▶", _C_YELLOW)
            label = _c(f"{i:>2}. {name}", _C_BOLD)
        else:
            mark = " "
            label = _c(f"{i:>2}. {name}", _C_DIM)
        line = f"  {mark}  {label}"
        pad = width - 2 - len(f"  {i:>2}. {name}") - 4
        print(_c("│", _C_CYAN) + line + " " * max(pad, 0) + _c("│", _C_CYAN))
    print(_c("└" + "─" * (width - 2) + "┘", _C_CYAN))
    print()


def print_progress_bar(current: int, total: int = TOTAL_STEPS, bar_len: int = 30):
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = 100 * current / total
    print(_c(f"[{bar}] {current}/{total} steps ({pct:.0f}%)", _C_CYAN))


def live_step_start(step_num: int, extra: str = ""):
    print_workflow_banner(current_step=step_num)
    print_progress_bar(step_num - 1)
    log_step_start(log, step_num, _STEP_NAMES[step_num - 1], TOTAL_STEPS)
    if extra:
        print(_c(f"  → {extra}", _C_DIM))


def live_step_done(step_num: int, t0: float, summary: str = ""):
    dt = time.time() - t0
    log_step_done(log, step_num, _STEP_NAMES[step_num - 1], dt)
    tick = _c("✔", _C_GREEN)
    print(f"  {tick} done in {dt:.1f}s" + (f" — {summary}" if summary else ""))
    print_progress_bar(step_num)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: str, log_file: Optional[str] = None, allow_fail: bool = False,
            cwd: Optional[str] = None, stream: bool = False):
    """Run a shell command with error handling.

    When stream=True, combined stdout/stderr is printed live and copied to
    log_file. The default remains file capture for existing callers.
    """
    log.info(f"Running: {cmd[:120]}...")
    if cwd is None:
        cwd = str(Path.home())
    try:
        if stream:
            log_handle = open(log_file, 'w') if log_file else None
            try:
                process = subprocess.Popen(
                    cmd, shell=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd)
                for line in process.stdout:
                    print(line, end='', flush=True)
                    if log_handle:
                        log_handle.write(line)
                        log_handle.flush()
                result = subprocess.CompletedProcess(cmd, process.wait())
            finally:
                if log_handle:
                    log_handle.close()
        elif log_file:
            with open(log_file, 'w') as lf:
                result = subprocess.run(cmd, shell=True, stdout=lf,
                                         stderr=subprocess.STDOUT,
                                         timeout=86400, cwd=cwd)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                     text=True, timeout=86400, cwd=cwd)
        if result.returncode != 0 and not allow_fail:
            log.error(f"Command failed with exit code {result.returncode}")
            if not log_file and result.stderr:
                log.error(f"STDERR: {result.stderr[:500]}")
            if log_file:
                log.error(f"Check log: {log_file}")
            sys.exit(1)
        return result
    except subprocess.TimeoutExpired:
        log.error("Command timed out (>24 hours)!")
        sys.exit(1)
    except Exception as e:
        log.error(f"Command failed with exception: {e}")
        sys.exit(1)


def run_cmd_allow_fail(cmd: str, log_file: Optional[str] = None,
                        cwd: Optional[str] = None):
    """Like run_cmd but never sys.exit()s — returns None on failure."""
    log.info(f"Running: {cmd[:150]}...")
    if cwd is None:
        cwd = str(Path.home())
    try:
        if log_file:
            with open(log_file, 'a') as lf:
                result = subprocess.run(cmd, shell=True, stdout=lf,
                                         stderr=subprocess.STDOUT,
                                         timeout=86400, cwd=cwd)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                     text=True, timeout=86400, cwd=cwd)
        if result.returncode != 0:
            log.warning(f"Command failed (exit {result.returncode}): {cmd[:150]}")
            return None
        return result
    except Exception as e:
        log.warning(f"Command exception: {e}")
        return None


def sh_quote(path) -> str:
    """Shell-quote a path/argument before interpolating it into a `run_cmd`
    string. Does not make the pipeline fully injection-proof (that would
    require abandoning shell=True everywhere), but closes the most common
    real-world failure mode: sample names or paths containing spaces,
    parentheses, or other shell-special characters silently breaking a
    command instead of erroring loudly."""
    return shlex.quote(str(path))


def count_seqs(fasta) -> int:
    try:
        if str(fasta).endswith('.gz'):
            with gzip.open(fasta, 'rt') as f:
                return sum(1 for line in f if line.startswith('>'))
        return sum(1 for _ in SeqIO.parse(fasta, 'fasta'))
    except Exception as e:
        log.error(f"Error counting sequences in {fasta}: {e}")
        return 0


def _iter_lengths(fasta) -> List[int]:
    lengths = []
    if str(fasta).endswith('.gz'):
        with gzip.open(fasta, 'rt') as f:
            for record in SeqIO.parse(f, 'fasta'):
                lengths.append(len(record.seq))
    else:
        for record in SeqIO.parse(fasta, 'fasta'):
            lengths.append(len(record.seq))
    return lengths


def calculate_n50(fasta) -> int:
    try:
        lengths = sorted(_iter_lengths(fasta), reverse=True)
        if not lengths:
            return 0
        half = sum(lengths) / 2
        cumsum = 0
        for length in lengths:
            cumsum += length
            if cumsum >= half:
                return length
        return 0
    except Exception as e:
        log.error(f"Error calculating N50 for {fasta}: {e}")
        return 0


def calculate_assembly_stats(fasta) -> Dict:
    """Step 3: the FULL stats report — N50, L50, length distribution, and
    counts below common thresholds. This is a REPORT ONLY; nothing here
    triggers deletion or merging decisions on its own."""
    lengths = sorted(_iter_lengths(fasta), reverse=True)
    if not lengths:
        return {"n_contigs": 0, "total_bp": 0, "n50": 0, "l50": 0,
                "lt_1kb": 0, "lt_2kb": 0, "lt_5kb": 0}
    total = sum(lengths)
    half = total / 2
    cumsum = 0
    n50 = l50 = 0
    for i, length in enumerate(lengths, start=1):
        cumsum += length
        if cumsum >= half:
            n50, l50 = length, i
            break
    arr = np.array(lengths)
    return {
        "n_contigs": len(lengths),
        "total_bp": int(total),
        "n50": n50,
        "l50": l50,
        "mean_length": float(arr.mean()),
        "median_length": float(np.median(arr)),
        "max_length": int(arr.max()),
        "min_length": int(arr.min()),
        "lt_1kb": int((arr < 1000).sum()),
        "lt_2kb": int((arr < 2000).sum()),
        "lt_5kb": int((arr < 5000).sum()),
    }


def check_file_integrity(filepath, file_type="fasta") -> Tuple[bool, str]:
    filepath = Path(filepath)
    if not filepath.exists():
        return False, f"File not found: {filepath}"
    if filepath.stat().st_size == 0:
        return False, f"File is empty: {filepath}"
    try:
        opener = gzip.open if str(filepath).endswith('.gz') else open
        mode = 'rt' if str(filepath).endswith('.gz') else 'r'
        with opener(filepath, mode) as f:
            first_line = f.readline()
        marker = '>' if file_type == "fasta" else '@'
        if not first_line.startswith(marker):
            return False, f"Not a valid {file_type.upper()} file: {filepath}"
    except Exception as e:
        return False, f"{file_type.upper()} validation failed: {filepath} - {e}"
    return True, "OK"


# ── ID safety ───────────────────────────────────────────────────────────

_UNSAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def make_safe_id(original_id: str, sample: str) -> str:
    """Build a sample-prefixed, shell-safe internal ID. Any character
    outside [A-Za-z0-9_.-] is replaced with '_' — this is what closes the
    'spaces/pipes/colons in FASTA headers break minimap2/samtools/CoverM/
    Tiara' failure mode. The ORIGINAL id is never lost — it's preserved in
    id_map.tsv and only restored in the final human-facing reports."""
    cleaned = _UNSAFE_ID_RE.sub('_', original_id.strip())
    return f"{sample}_{cleaned}"


def resolve_read_path(reads_dir, value) -> Path:
    """Resolve a read-file entry from the samples table. If the samples TSV
    already gives an absolute path, use it as-is; only join with reads_dir
    for a relative entry. Fixes the old unconditional `Path(reads_dir) /
    row[col]`, which mangled any already-absolute path."""
    p = Path(str(value))
    return p if p.is_absolute() else Path(reads_dir) / p


def detect_read_type(reads_path, sample_limit: int = 3, candidate_files: Optional[List[Path]] = None,
                      seed: int = 0) -> str:
    """`candidate_files`, when given, is the preferred source of FASTQ files
    to sample from — the samples TSV's own r1/r2 entries, already resolved
    via `resolve_read_path` (so absolute paths outside `reads_path` are
    actually reached instead of silently missed by a directory glob that
    only looks inside `reads_path`). Falls back to globbing `reads_path`
    only when no candidate list is supplied (e.g. no samples table yet) or
    it comes back empty. `seed` makes the sampling reproducible run-to-run
    instead of a different random subset (and therefore a possibly
    different auto-detected read_type) each time."""
    fastq_files = [Path(f) for f in candidate_files] if candidate_files else []
    if not fastq_files:
        reads_path = Path(reads_path)
        for ext in ['*.fastq', '*.fq', '*.fastq.gz', '*.fq.gz']:
            fastq_files.extend(reads_path.glob(ext))
    if not fastq_files:
        log.warning("No FASTQ files found for read type detection — defaulting to short")
        return "short"
    rng = random.Random(seed)
    sample_files = rng.sample(fastq_files, min(sample_limit, len(fastq_files)))
    lengths, qualities = [], []
    for fq in sample_files:
        try:
            opener = gzip.open if str(fq).endswith('.gz') else open
            mode = 'rt' if str(fq).endswith('.gz') else 'r'
            with opener(fq, mode) as f:
                for i, line in enumerate(f):
                    if i % 4 == 1:
                        lengths.append(len(line.strip()))
                    elif i % 4 == 3:
                        qualities.extend(ord(c) - 33 for c in line.strip() if ord(c) >= 33)
                    if len(lengths) >= 1000:
                        break
        except Exception as e:
            log.warning(f"Could not sample {fq}: {e}")
    if not lengths:
        return "short"
    mean_len = np.mean(lengths)
    mean_q = np.mean(qualities) if qualities else 0
    log.info(f"Read type detection: mean_len={mean_len:.0f}, mean_Q={mean_q:.1f}")
    if mean_len < 500:
        return "short"
    elif mean_q >= 30 and mean_len >= 5000:
        return "hifi"
    return "ont"


def _split_coverm_columns(header_cols: List[str]) -> Tuple[List[str], List[str]]:
    """CoverM's 'contig' output has one column per (sample, requested metric).
    We ask for 'mean' (depth) and 'covered_fraction' (breadth), but rather
    than hardcode CoverM's exact column-naming template (which has changed
    across versions — the same risk flagged for Tiara's output columns),
    classify columns by substring and degrade gracefully: if breadth columns
    can't be identified, breadth-based filtering is simply skipped rather
    than crashing on a naming mismatch."""
    depth_cols, breadth_cols = [], []
    for c in header_cols:
        cl = c.lower()
        if "covered" in cl or "breadth" in cl or "fraction" in cl:
            breadth_cols.append(c)
        else:
            depth_cols.append(c)
    return depth_cols, breadth_cols


def get_minimap2_preset(read_type: str) -> str:
    return {"short": "-ax sr", "hifi": "-ax map-hifi", "ont": "-ax map-ont"}.get(read_type, "-ax sr")


def cleanup_intermediate(paths, label: str = ""):
    freed = 0
    for p in paths:
        p = Path(p)
        try:
            if p.is_file():
                freed += p.stat().st_size
                p.unlink()
            elif p.is_dir():
                freed += sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                shutil.rmtree(p)
        except Exception as e:
            log.warning(f"Cleanup skipped for {p}: {e}")
    if freed:
        log.info(f"🧹 Cleanup ({label}): freed {freed/1e9:.2f}GB")


# ── checkpoint fingerprinting ───────────────────────────────────────────
# A completed checkpoint was previously trusted just because the step name
# existed on disk — so re-running after changing dedup_ani, a rescue
# reference, or even the input FASTA silently reused stale output. Every
# step below now fingerprints its actual inputs (file identity + the config
# keys it reads) and compares against the fingerprint stored the last time
# it ran; a mismatch forces a re-run instead of trusting the old result.

def _file_fingerprint(path) -> str:
    """Cheap identity proxy for a pipeline file: name + size + mtime. Not a
    content hash — these are often multi-GB FASTA/BAM files, so hashing
    bytes would make the fingerprint check slower than just re-running.

    CHANGED: deliberately uses the basename only, not the full absolute
    path. Moving a run's whole output tree to a different server/mount
    (rsync -a preserves size+mtime, only the path changes) used to make
    every checkpoint downstream of the moved file look stale, cascading
    into re-running expensive steps (e.g. step 6's 5+ hour Tiara+Whokaryote
    classification) for no real reason — the bytes never changed. Basename
    is enough to catch genuine content edits (which also change size or
    mtime) while tolerating a relocation."""
    p = Path(str(path))
    try:
        st = p.stat()
        return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
    except FileNotFoundError:
        return f"{p.name}:MISSING"


def _fingerprint(*parts) -> str:
    """Stable short hash over an arbitrary mix of file fingerprints, config
    values, and dicts. Order-independent for dict contents (sort_keys)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


_PATH_LIKE_KEY_SUFFIXES = ('_file', '_tsv', '_fasta', '_dir', '_path', '_gff', '_table')
_PATH_LIKE_KEYS = {'output', 'gff', 'id_map', 'rescue_log', 'pairs_report'}


def _meta_output_paths(meta: Dict) -> List[str]:
    """Best-effort extraction of file/dir paths referenced in a step's cached
    metadata, using key-naming convention rather than content-sniffing a
    string (which would misfire on non-path values like module_routing's
    'masked'/'unmasked' labels). Deliberately shallow — it does not recurse
    into lists like step 9's bam_files, which are deleted by
    cleanup_intermediate() after step 12 only when keep_bams=false, and must never be treated as
    'missing, therefore this checkpoint is broken.'"""
    paths = []
    for k, v in meta.items():
        if not isinstance(v, str) or not v:
            continue
        if k in _PATH_LIKE_KEYS or any(k.endswith(suf) for suf in _PATH_LIKE_KEY_SUFFIXES):
            paths.append(v)
    return paths


def _checkpoint_ok(ckpt, step: str, fp: str) -> Optional[Dict]:
    """Returns the cached metadata dict if STEP is checkpointed AND its
    stored fingerprint matches the current inputs/config AND every output
    file/dir referenced in that metadata still exists on disk; otherwise
    None (re-run either way). The third check matters on its own: a
    fingerprint match with a deleted output — someone cleaned the output
    directory, or a partial rsync — used to be trusted anyway and would only
    fail later, confusingly, in whichever downstream step tried to read the
    now-missing file."""
    if not ckpt.is_done(step):
        return None
    prev = ckpt.load_metadata(step)
    if prev.get("_fp") != fp:
        log.warning(f"{step}: checkpoint exists but inputs/config changed since "
                    f"it ran — ignoring stale checkpoint and re-running.")
        return None
    missing = [p for p in _meta_output_paths(prev) if not Path(p).exists()]
    if missing:
        shown = missing[:3]
        log.warning(f"{step}: checkpoint fingerprint matches, but {len(missing)} referenced "
                    f"output(s) no longer exist on disk ({shown}"
                    f"{', ...' if len(missing) > 3 else ''}) — treating as a cache miss "
                    f"rather than trusting a checkpoint that points at deleted output.")
        return None
    log.info(f"  ⏭️  SKIP  [{step}]  (fingerprint matched, outputs verified)")
    return prev


def ensure_tool(cmd: str, name: str, auto_install: bool = False) -> bool:
    """Check that a required external tool is on PATH.

    Auto-installing missing bioinformatics tools via `conda install` is a
    silent environment mutation — it can pull in a different tool version
    than the one the user tested with, take minutes, or fail halfway and
    leave a broken env. So this is opt-in only (config["auto_install_tools"]
    must be explicitly set True). By default we just report what's missing
    and let step1 fail loud, so the user installs the exact versions they
    want before anything runs."""
    if shutil.which(cmd):
        return True
    if not auto_install:
        log.error(f"{name} ({cmd}) not found on PATH. Install it manually, e.g. "
                   f"`conda install -c bioconda -c conda-forge {cmd}`, or set "
                   f"config['auto_install_tools']=True to let this pipeline attempt it for you.")
        return False
    log.warning(f"{name} ({cmd}) not found — auto_install_tools is enabled, "
                f"attempting auto-install via conda...")
    run_cmd_allow_fail(f"conda install -y -c bioconda -c conda-forge {cmd}")
    if shutil.which(cmd):
        log.info(f"✅ {name} installed automatically")
        return True
    log.error(f"Auto-install of {name} ({cmd}) failed. Install manually: conda install -c bioconda {cmd}")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — VALIDATE INPUTS
# ═══════════════════════════════════════════════════════════════════════════════

def _barrnap_supports_kingdom(kingdom) -> Optional[bool]:
    """True/False if `barrnap --help` does / does not mention the kingdom as a whole word; None if it cannot be
    determined (barrnap missing, or help text has no --kingdom)."""
    if shutil.which("barrnap") is None:
        return None
    try:
        r = subprocess.run(["barrnap", "--help"], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    text = (r.stdout or "") + (r.stderr or "")
    if "--kingdom" not in text:
        return None
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(str(kingdom)) + r"(?![A-Za-z0-9_])", text) is not None


def step1_validate_inputs(assembly_input, samples_df, reads_dir, outdir, config) -> Dict:
    t0 = time.time()
    live_step_start(1)
    errors, warnings = [], []

    assembly_path = Path(assembly_input)
    if assembly_path.is_dir():
        fasta_files = sorted(sum((list(assembly_path.glob(p)) for p in
                                   ("*.fasta", "*.fa", "*.fasta.gz", "*.fa.gz")), []))
        if not fasta_files:
            errors.append(f"No FASTA files found in directory: {assembly_path}")
        for fasta in fasta_files:
            ok, msg = check_file_integrity(fasta, "fasta")
            if not ok:
                errors.append(msg)
    elif assembly_path.is_file():
        ok, msg = check_file_integrity(assembly_path, "fasta")
        if not ok:
            errors.append(msg)
        elif count_seqs(assembly_path) == 0:
            errors.append(f"FASTA file contains 0 sequences: {assembly_path}")
    else:
        errors.append(f"Assembly input not found: {assembly_path}")

    if samples_df is None or len(samples_df) == 0:
        errors.append("Samples table is empty or invalid")
    else:
        missing_cols = [c for c in ('sample', 'r1') if c not in samples_df.columns]
        if missing_cols:
            errors.append(f"Samples table missing columns: {missing_cols}")
        else:
            # Sample names become BAM filenames verbatim in step 9
            # (`{sample}.bam`) — a duplicate name means two samples silently
            # collide onto the same BAM (previously only caught much later,
            # after steps 1-8 already ran, as an opaque "CoverM duplicate
            # column" failure at step 9); a '/' in the name silently nests
            # it into a subdirectory that was never created, which fails
            # samtools with a confusing "no such file" instead of a clear
            # validation error here.
            sample_names = [str(s) for s in samples_df['sample']]
            dup_samples = {s for s in sample_names if sample_names.count(s) > 1}
            if dup_samples:
                errors.append(f"Duplicate sample name(s) in samples table: {sorted(dup_samples)} "
                               f"— each sample must be unique (it becomes that sample's BAM "
                               f"filename in step 9).")
            unsafe_samples = [s for s in sample_names if '/' in s or not s.strip()]
            if unsafe_samples:
                errors.append(f"Sample name(s) containing '/' or blank: {unsafe_samples} "
                               f"— sample names are used directly as filenames and cannot "
                               f"contain a path separator.")
            for _, row in samples_df.iterrows():
                ok, msg = check_file_integrity(resolve_read_path(reads_dir, row['r1']), "fastq")
                if not ok:
                    errors.append(f"Sample {row['sample']} R1: {msg}")
                r2_value = row.get('r2') if 'r2' in samples_df.columns else None
                if pd.notna(r2_value) and str(r2_value).strip():
                    ok, msg = check_file_integrity(resolve_read_path(reads_dir, r2_value), "fastq")
                    if not ok:
                        errors.append(f"Sample {row['sample']} R2: {msg}")

    # NEW: ID safety checks — catch whitespace/shell-special characters and
    # duplicate IDs BEFORE they can break a downstream shell command.
    if assembly_path.is_file() or assembly_path.is_dir():
        fastas = [assembly_path] if assembly_path.is_file() else sorted(
            sum((list(assembly_path.glob(p)) for p in ("*.fasta", "*.fa")), []))
        for fasta in fastas:
            try:
                seen_ids = set()
                for rec in SeqIO.parse(str(fasta) if not str(fasta).endswith('.gz')
                                        else gzip.open(fasta, 'rt'), 'fasta'):
                    if rec.id in seen_ids:
                        warnings.append(f"Duplicate contig ID '{rec.id}' in {fasta.name} "
                                         f"(will be disambiguated during merge)")
                    seen_ids.add(rec.id)
                    if re.search(r"\s", rec.id):
                        warnings.append(f"Whitespace in contig ID '{rec.id}' in {fasta.name} "
                                         f"(will be sanitized during merge)")
            except Exception as e:
                warnings.append(f"Could not pre-scan IDs in {fasta}: {e}")

    try:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        test = Path(outdir) / ".write_test"
        test.touch(); test.unlink()
    except Exception as e:
        errors.append(f"Cannot write to output directory {outdir}: {e}")

    required_tools = {'skani': 'skani', 'seqkit': 'seqkit', 'minimap2': 'minimap2',
                       'samtools': 'samtools', 'coverm': 'CoverM', 'barrnap': 'barrnap',
                       'tiara': 'Tiara', 'whokaryote.py': 'Whokaryote'}
    auto_install = bool(config.get("auto_install_tools", False))
    missing = [name for cmd, name in required_tools.items() if not ensure_tool(cmd, name, auto_install)]
    if missing:
        errors.append(f"Missing required tools: {', '.join(missing)}. "
                       f"Set config['auto_install_tools']=True to have the pipeline install "
                       f"them via conda automatically, or install them yourself first.")

    # Resolve every sample's actual r1/r2 path (absolute-or-relative, same
    # rule as everywhere else reads are touched) so detection samples the
    # real read files even when the TSV points outside reads_dir — a plain
    # glob of reads_dir would silently find nothing in that case and
    # default to "short" without ever explaining why.
    candidate_read_files = []
    if samples_df is not None and len(samples_df) > 0:
        for _, row in samples_df.iterrows():
            for col in ('r1', 'r2'):
                value = row.get(col) if col in samples_df.columns else None
                if pd.notna(value) and str(value).strip():
                    candidate_read_files.append(resolve_read_path(reads_dir, value))

    read_type = config.get("read_type", "auto")
    detect_seed = int(config.get("read_type_detect_seed", 0))
    if read_type == "auto":
        config["read_type"] = detect_read_type(reads_dir, candidate_files=candidate_read_files,
                                                 seed=detect_seed)
        log.info(f"✅ Auto-detected read type: {config['read_type']}")
    elif read_type not in ("short", "hifi", "ont", "hybrid"):
        warnings.append(f"Unknown read_type '{read_type}' — defaulting to auto-detection")
        config["read_type"] = detect_read_type(reads_dir, candidate_files=candidate_read_files,
                                                 seed=detect_seed)

    if config.get("rescue_enabled", True) and not config.get("rescue_reference_fasta"):
        warnings.append("rescue_enabled=True but rescue_reference_fasta not set — "
                         "step 10 cannot run the rescue alignment, so affected "
                         "single-sample gray-zone contigs will be RETAINED and "
                         "flagged 'mid_retained_no_rescue_reference' instead of "
                         "being removed (a missing reference is a config gap, not "
                         "evidence against the contig — set rescue_reference_fasta "
                         "to get an actual rescue check instead of this safe default)")

    _bk = str(config.get("barrnap_kingdom", "fun"))
    if _barrnap_supports_kingdom(_bk) is False:
        warnings.append(f"barrnap_kingdom='{_bk}' is not mentioned in `barrnap --help` of the installed barrnap — "
                        f"step 8 may fail (rRNA unmasked). Check `barrnap --help` and barrnap.log.")
    if errors:
        log.error("VALIDATION FAILED:")
        for i, e in enumerate(errors, 1):
            log.error(f"  {i}. {e}")
        sys.exit(1)
    if warnings:
        log.warning(f"{len(warnings)} warning(s):")
        for w in warnings:
            log.warning(f"  - {w}")

    live_step_done(1, t0, "all validations passed")
    return {"warnings": warnings}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MERGE ASSEMBLIES (+ ID MAP)
# ═══════════════════════════════════════════════════════════════════════════════

def step2_merge_assemblies(assembly_input, outdir, ckpt, config=None) -> Dict:
    """Merge per-sample assemblies into one FASTA. Every contig gets a safe,
    sample-prefixed internal ID (see `make_safe_id`). Duplicate SEQUENCES
    (not just duplicate IDs) across samples are flagged here for the record
    — actual removal of near-duplicates happens later in the dedup step (skani),
    this step only records what it saw."""
    STEP = "step2_merge"
    config = config or {}
    assembly_path = Path(assembly_input)
    sample_regex = str(config.get("assembly_sample_regex", "") or "")
    use_regex = bool(sample_regex) and assembly_path.is_file()
    if sample_regex and not use_regex:
        log.warning("assembly_sample_regex is only used when --scaffold is a single FASTA file; "
                    "ignored because a directory was given (each file = one sample).")
    sample_re = re.compile(sample_regex) if use_regex else None
    if assembly_path.is_file():
        raw_pairs = [(assembly_path, assembly_path.stem)]
    else:
        fastas = sorted(sum((list(assembly_path.glob(p)) for p in
                              ("*.fasta", "*.fa", "*.fasta.gz", "*.fa.gz")), []))
        raw_pairs = [(f, f.stem.replace('.fasta', '').replace('.fa', '')) for f in fastas]

    # Sanitize each derived sample name the same way contig IDs are (it ends
    # up in the safe_id prefix and in filenames like '{sample}.bam'), then
    # disambiguate any collision — two different source files must never
    # silently collapse onto the one sample tag just because their sanitized
    # names happen to match (e.g. 'sample-A.fa' and 'sample_A.fasta' both
    # sanitizing to 'sample_A').
    seen_names: Dict[str, int] = {}
    fasta_sample_pairs = []
    for f, raw_name in raw_pairs:
        clean = _UNSAFE_ID_RE.sub('_', raw_name.strip()) or "sample"
        if clean in seen_names:
            seen_names[clean] += 1
            disambiguated = f"{clean}_{seen_names[clean]}"
            log.warning(f"Sample name collision: '{f.name}' sanitizes to '{clean}', already "
                        f"used by another input file — renaming this file's sample tag to "
                        f"'{disambiguated}' rather than silently merging the two under one name.")
            clean = disambiguated
        else:
            seen_names[clean] = 0
        fasta_sample_pairs.append((f, clean))

    fp = _fingerprint([_file_fingerprint(f) for f, _ in fasta_sample_pairs],
                       sample_regex if use_regex else "")
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(2)

    merge_dir = Path(outdir) / "02_merged"
    merge_dir.mkdir(parents=True, exist_ok=True)
    output = str(merge_dir / "merged.fasta")
    id_map_path = merge_dir / "id_map.tsv"

    seq_hash_seen: Dict[str, str] = {}   # sequence -> first safe_id that had it
    sample_counts: Dict[str, int] = defaultdict(int)
    unmatched_examples: List[str] = []
    n_unmatched = 0
    n_dup_seqs = 0
    total_seqs = 0

    with open(output, 'w') as out_f, open(id_map_path, 'w') as map_f:
        # source_sample is the sanitized, collision-checked label used
        # internally (safe_id prefix, BAM filenames); source_file is the
        # verbatim input path, kept separately so provenance survives even
        # though the label is not a biological "sample name" claim by
        # itself — see step2's docstring / the sample-naming warning above.
        map_f.write("original_id\tsafe_internal_id\tsource_sample\tsource_file\tlength\tduplicate_of\n")
        for fasta, sample in fasta_sample_pairs:
            opener = gzip.open if str(fasta).endswith('.gz') else open
            mode = 'rt' if str(fasta).endswith('.gz') else 'r'
            with opener(fasta, mode) as in_f:
                for record in SeqIO.parse(in_f, 'fasta'):
                    original_id = record.id          # captured BEFORE overwrite —
                    rec_sample = sample
                    if sample_re is not None:
                        m_s = sample_re.search(original_id)
                        if m_s and m_s.groups() and m_s.group(1):
                            rec_sample = _UNSAFE_ID_RE.sub('_', m_s.group(1).strip())
                        else:
                            n_unmatched += 1
                            if len(unmatched_examples) < 5:
                                unmatched_examples.append(original_id)
                    sample_counts[rec_sample] += 1
                    safe_id = make_safe_id(original_id, rec_sample)   # the old code wrote
                    seq_str = str(record.seq).upper()             # record.id (already
                    dup_of = ""                                   # == safe_id) into the
                    if seq_str in seq_hash_seen:                  # original_id column.
                        dup_of = seq_hash_seen[seq_str]
                        n_dup_seqs += 1
                    else:
                        seq_hash_seen[seq_str] = safe_id
                    record.id = safe_id
                    record.description = ""
                    SeqIO.write(record, out_f, 'fasta')
                    map_f.write(f"{original_id}\t{safe_id}\t{rec_sample}\t{fasta}\t{len(seq_str)}\t{dup_of}\n")
                    total_seqs += 1

    if n_unmatched:
        log.error(f"assembly_sample_regex {sample_regex!r} did not match {n_unmatched:,} contig ID(s) "
                  f"(e.g. {unmatched_examples}). Every header must yield a sample label — fix the "
                  f"regex (ONE capture group) or use a directory of per-sample FASTAs.")
        sys.exit(1)
    log.info(f"Source samples for dedup/ID prefix: {len(sample_counts)} "
             f"({', '.join(f'{k}={v:,}' for k, v in sorted(sample_counts.items())[:12])}"
             f"{' ...' if len(sample_counts) > 12 else ''})")
    if n_dup_seqs:
        log.info(f"Recorded {n_dup_seqs:,} exact-duplicate sequences across "
                  f"input files (near-duplicate removal happens in the dedup step)")

    n_final = count_seqs(output)
    meta = {"output": output, "id_map": str(id_map_path),
            "n_seqs": n_final, "n_exact_dup_seqs": n_dup_seqs,
            "n_source_files": len(fasta_sample_pairs), "_fp": fp}
    ckpt.mark_done(STEP, meta)
    live_step_done(2, t0, f"{n_final:,} contigs merged from {len(fasta_sample_pairs)} file(s)")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ASSEMBLY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def step3_assembly_stats(fasta, outdir, ckpt) -> Dict:
    STEP = "step3_stats"
    fp = _fingerprint(_file_fingerprint(fasta))
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(3)

    stats = calculate_assembly_stats(fasta)
    log.info(f"  Contigs: {stats['n_contigs']:,}  |  Total bp: {stats['total_bp']:,}")
    log.info(f"  N50: {stats['n50']:,}bp  |  L50: {stats['l50']:,}")
    log.info(f"  <1kb: {stats['lt_1kb']:,}  |  <2kb: {stats['lt_2kb']:,}  |  <5kb: {stats['lt_5kb']:,}")

    # Merging (step 2) already happened unconditionally before this step ever
    # runs, and nothing below gates on N50 either — this is a QUALITY REPORT
    # ONLY. Whatever tier the assembly falls in, the pipeline warns (louder
    # the worse it is) and proceeds with every contig it already merged.
    n50 = stats['n50']
    if n50 < 1000:
        stats["n50_quality"] = "extremely_fragmented"
        log.warning(f"🛑 N50={n50:,}bp — EXTREMELY fragmented assembly (<1kb). Quality report "
                    f"only: no contigs are dropped or excluded from the merge because of this. "
                    f"Binning results downstream should be treated with heavy caution.")
    elif n50 < 2000:
        stats["n50_quality"] = "fragmented"
        log.warning(f"⚠️  N50={n50:,}bp — fragmented assembly (1-2kb). Proceeding as normal; "
                    f"treat downstream bins with some caution.")
    elif n50 < 5000:
        stats["n50_quality"] = "good"
        log.info(f"✅ N50={n50:,}bp — good assembly quality (>2kb)")
    else:
        stats["n50_quality"] = "excellent"
        log.info(f"✅✅ N50={n50:,}bp — excellent assembly quality (>5kb)")

    out_dir = Path(outdir) / "03_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "assembly_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    stats["_fp"] = fp
    ckpt.mark_done(STEP, stats)
    live_step_done(3, t0, f"N50={stats['n50']:,}bp")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAYED STEP 5 — DEDUPLICATION (skani; internal name step4_*; runs AFTER the length filter)
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_representatives(loser_to_winner: Dict[str, str], redundant: set) -> Dict[str, str]:
    """Follow winner chains (A inside B, B inside C -> A's final rep is C)
    until a contig that was NOT itself removed. Cycle-safe."""
    final = {}
    for loser in loser_to_winner:
        cur, seen = loser_to_winner[loser], {loser}
        while cur in redundant and cur in loser_to_winner and cur not in seen:
            seen.add(cur)
            cur = loser_to_winner[cur]
        final[loser] = cur
    return final


def step4_skani_dedup(fasta, outdir, threads, ckpt, config, id_map=None) -> Dict:
    """Cross-sample CONTAINMENT dedup with skani (runs AFTER the length filter).

    Why containment: the same source genome assembled in two samples yields
    different contig boundaries, so a contig from sample B is typically
    *contained* in a longer contig from sample A. The old rule (both aligned
    fractions >= 50% AND length ratio >= 0.8) could not flag those, so whole
    genomes ended up duplicated in the catalogue and later in a single bin.

    A pair is redundant only when ALL hold:
      - the two contigs come from DIFFERENT samples (if dedup_cross_sample_only;
        sample taken from id_map.tsv `source_sample`, never parsed from the ID)
      - ANI >= dedup_ani
      - aligned fraction of the SHORTER contig >= dedup_min_af
      - (optional) min(len)/max(len) >= dedup_min_length_ratio  (0 = off)
    The longer contig is kept; on an exact length tie the smaller ID wins.
    Only contigs >= dedup_min_length are compared; shorter ones pass through.

    skani notes: --min-af is a PERCENTAGE (0-100), strict '>'; it keeps a pair
    if EITHER genome passes, which is what containment needs (do NOT use
    --both-min-af). Output columns are read BY NAME with a fail-loud check.

    Outputs (in 04_dedup/): deduplicated.fasta, removed_redundant_ids.txt,
    dedup_pairs_report.tsv (every qualifying pair) and
    dedup_removed_map.tsv (removed contig -> kept representative)."""
    STEP = "step4_dedup"
    dedup_ani = float(config.get("dedup_ani", 99.5))
    dedup_min_af = float(config.get("dedup_min_af", 95.0))
    ratio_min = float(config.get("dedup_min_length_ratio", 0.0))
    floor_len = int(config.get("dedup_min_length", 2000))
    cross_only = bool(config.get("dedup_cross_sample_only", True))
    skani_args = str(config.get("dedup_skani_args", "-c 30 -m 200 --faster-small -s 95"))
    allow_skip = bool(config.get("dedup_allow_skip_over_cap", False))
    max_contigs = int(config.get("dedup_max_contigs", 500_000))
    skip = bool(config.get("skip_dedup", False))

    fp = _fingerprint(_file_fingerprint(fasta), _file_fingerprint(id_map) if id_map else "",
                       dedup_ani, dedup_min_af, ratio_min, floor_len, cross_only, skani_args,
                       skip, max_contigs, allow_skip, "v3-containment")
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(5, f"ANI>={dedup_ani}% shorter-AF>={dedup_min_af}% len>={floor_len} "
                        f"cross-sample={cross_only}")

    out_dir = Path(outdir) / "04_dedup"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / "deduplicated.fasta")
    before = count_seqs(fasta)

    def _skipped(reason):
        meta = {"output": fasta, "before": before, "after": before, "n_removed": 0,
                "skipped": True, "skip_reason": reason, "_fp": fp, "redundant_ids_file": None}
        ckpt.mark_done(STEP, meta)
        live_step_done(5, t0, f"skipped ({reason})")
        return meta

    if skip:
        log.warning("Dedup skipped: skip_dedup=True (ablation baseline)")
        return _skipped("skip_dedup=True")

    # ── sample lookup (needed for the cross-sample rule) ──────────────────
    sample_of: Dict[str, str] = {}
    if cross_only:
        if not id_map or not Path(id_map).exists():
            log.error("dedup_cross_sample_only=True but id_map.tsv is unavailable — "
                      "refusing to guess sample membership from contig IDs.")
            sys.exit(1)
        im = pd.read_csv(id_map, sep='\t', dtype=str, usecols=["safe_internal_id", "source_sample"])
        sample_of = dict(zip(im["safe_internal_id"], im["source_sample"]))
        if len(set(sample_of.values())) < 2:
            log.error("STOPPING: only ONE source sample found, so cross-sample dedup cannot work "
                      "(one FASTA file = one sample). Pick one: (1) pass --scaffold as a DIRECTORY "
                      "of per-sample FASTAs; (2) if the merged FASTA's headers carry the sample, set "
                      "assembly_sample_regex (one capture group); (3) set "
                      "dedup_cross_sample_only=false to compare all contigs (also removes "
                      "within-sample near-duplicates); (4) set skip_dedup=true.")
            sys.exit(1)

    # ── write the comparison subset (>= floor_len) ────────────────────────
    subset_fa = str((out_dir / f"dedup_input_ge{floor_len}bp.fasta").resolve())   # absolute: run_cmd uses cwd=$HOME
    lengths: Dict[str, int] = {}
    with open(subset_fa, 'w') as f:
        for rec in SeqIO.parse(fasta, "fasta"):
            if len(rec.seq) >= floor_len:
                lengths[rec.id] = len(rec.seq)
                f.write(f">{rec.id}\n{str(rec.seq)}\n")
    n_compared = len(lengths)
    log.info(f"Dedup comparison set: {n_compared:,} contigs >= {floor_len} bp "
             f"(of {before:,}; shorter contigs pass through untouched)")

    if n_compared > max_contigs:
        msg = (f"{n_compared:,} contigs >= {floor_len} bp exceeds dedup_max_contigs={max_contigs:,}")
        if allow_skip:
            log.warning(f"Skipping dedup — {msg} (dedup_allow_skip_over_cap=True)")
            return _skipped("over contig cap")
        log.error(f"STOPPING: {msg}. Raise dedup_max_contigs, raise dedup_min_length, or set "
                  f"dedup_allow_skip_over_cap=true to accept an un-deduplicated catalogue.")
        sys.exit(1)
    if n_compared < 2:
        return _skipped("fewer than 2 contigs to compare")

    # ── skani (all-vs-all per contig) ──────────────────────────────────────
    ani_tsv = str((out_dir / "skani_all_vs_all.tsv").resolve())
    prefilter = max(0.0, dedup_min_af - 1.0)     # skani --min-af is strict '>' and keeps a
                                                  # pair if EITHER side passes (containment)
    cmd = (f"skani dist --qi {sh_quote(subset_fa)} --ri {sh_quote(subset_fa)} -t {threads} "
           f"--min-af {prefilter} {skani_args} -o {sh_quote(ani_tsv)}")
    log.info(f"skani: {cmd}")
    run_cmd(cmd, str((out_dir / "skani.log").resolve()))

    redundant: set = set()
    pairs = pd.DataFrame()
    if Path(ani_tsv).exists() and Path(ani_tsv).stat().st_size > 0:
        need = ["ANI", "Align_fraction_ref", "Align_fraction_query", "Ref_name", "Query_name"]
        df = pd.read_csv(ani_tsv, sep='\t')
        missing = set(need) - set(df.columns)
        if missing:
            log.error(f"skani output {ani_tsv} is missing column(s) {missing} "
                      f"(found {list(df.columns)}) — version/schema mismatch, refusing to guess.")
            sys.exit(1)
        df = df[need].copy()
        df["Query_name"] = df["Query_name"].astype(str)
        df["Ref_name"] = df["Ref_name"].astype(str)
        n_raw = len(df)
        df = df[df["Query_name"] != df["Ref_name"]]
        df = df[df["ANI"] >= dedup_ani]
        df["len_q"] = df["Query_name"].map(lengths)
        df["len_r"] = df["Ref_name"].map(lengths)
        n_unmapped = int(df["len_q"].isna().sum() + df["len_r"].isna().sum())
        if n_raw > 0 and len(df) > 0 and n_unmapped == 2 * len(df):
            log.error("None of skani's Ref_name/Query_name values match contig IDs — "
                      "header handling changed; dedup cannot proceed safely.")
            sys.exit(1)
        df = df.dropna(subset=["len_q", "len_r"])
        if cross_only and len(df):
            df["sample_q"] = df["Query_name"].map(sample_of)
            df["sample_r"] = df["Ref_name"].map(sample_of)
            df = df.dropna(subset=["sample_q", "sample_r"])
            df = df[df["sample_q"] != df["sample_r"]]
        else:
            df["sample_q"] = df["Query_name"].map(sample_of) if sample_of else ""
            df["sample_r"] = df["Ref_name"].map(sample_of) if sample_of else ""
        if len(df):
            lq, lr = df["len_q"].to_numpy(), df["len_r"].to_numpy()
            afq, afr = df["Align_fraction_query"].to_numpy(), df["Align_fraction_ref"].to_numpy()
            # aligned fraction of the SHORTER contig (min of both on an exact length tie)
            df["af_short"] = np.where(lq < lr, afq, np.where(lq > lr, afr, np.minimum(afq, afr)))
            df = df[df["af_short"] >= dedup_min_af]
        if len(df) and ratio_min > 0:
            ratio = np.minimum(df["len_q"], df["len_r"]) / np.maximum(df["len_q"], df["len_r"])
            df = df[ratio >= ratio_min]
        if len(df):
            q_wins = ((df["len_q"] > df["len_r"]) |
                      ((df["len_q"] == df["len_r"]) & (df["Query_name"] < df["Ref_name"])))
            df["winner"] = np.where(q_wins, df["Query_name"], df["Ref_name"])
            df["loser"] = np.where(q_wins, df["Ref_name"], df["Query_name"])
            df["len_winner"] = np.where(q_wins, df["len_q"], df["len_r"])
            df["len_loser"] = np.where(q_wins, df["len_r"], df["len_q"])
            df["sample_winner"] = np.where(q_wins, df["sample_q"], df["sample_r"])
            df["sample_loser"] = np.where(q_wins, df["sample_r"], df["sample_q"])
            pairs = df
            redundant = set(df["loser"])

    if len(pairs):
        pairs.to_csv(out_dir / "dedup_pairs_report.tsv", sep='\t', index=False)
        best = (pairs.sort_values(["len_winner", "ANI"], ascending=False)
                     .drop_duplicates("loser").set_index("loser"))
        l2w = best["winner"].to_dict()
        final_rep = _resolve_representatives(l2w, redundant)
        rows = []
        for loser, w in l2w.items():
            rows.append({"removed_id": loser, "representative": w,
                         "final_representative": final_rep.get(loser, w),
                         "ani": best.at[loser, "ANI"], "af_short": best.at[loser, "af_short"],
                         "len_removed": int(best.at[loser, "len_loser"]),
                         "len_representative": int(best.at[loser, "len_winner"]),
                         "sample_removed": best.at[loser, "sample_loser"],
                         "sample_representative": best.at[loser, "sample_winner"]})
        pd.DataFrame(rows).to_csv(out_dir / "dedup_removed_map.tsv", sep='\t', index=False)

    kept = 0
    with open(output, 'w') as out_f:
        for rec in SeqIO.parse(fasta, "fasta"):
            if rec.id not in redundant:
                out_f.write(f">{rec.id}\n{str(rec.seq)}\n")
                kept += 1

    redundant_ids_file = str(out_dir / "removed_redundant_ids.txt")
    with open(redundant_ids_file, 'w') as f:
        f.write("\n".join(sorted(redundant)) + ("\n" if redundant else ""))

    removed_bp = int(sum(lengths[c] for c in redundant)) if redundant else 0
    meta = {"output": output, "before": before, "after": kept,
            "n_removed": before - kept, "removed_bp": removed_bp, "n_compared": n_compared,
            "skipped": False, "_fp": fp, "redundant_ids_file": redundant_ids_file,
            "pairs_report": str(out_dir / "dedup_pairs_report.tsv") if len(pairs) else None,
            "removed_map": str(out_dir / "dedup_removed_map.tsv") if len(pairs) else None}
    ckpt.mark_done(STEP, meta)
    log.info(f"Dedup removed {before - kept:,} contigs ({removed_bp:,} bp) of {n_compared:,} compared")
    live_step_done(5, t0, f"{before:,} → {kept:,} ({before-kept:,} redundant removed)")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAYED STEP 4 — LENGTH PRE-FILTER (internal name step5_*; runs BEFORE dedup)
# ═══════════════════════════════════════════════════════════════════════════════

def step5_length_prefilter(fasta, outdir, min_len, ckpt) -> Dict:
    STEP = "step5_length_prefilter"
    fp = _fingerprint(_file_fingerprint(fasta), min_len)
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(4, f"threshold: ≥{min_len}bp")

    out_dir = Path(outdir) / "05_length_prefilter"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / f"ge{min_len}bp.fasta")
    run_cmd(f"seqkit seq -m {min_len} {sh_quote(fasta)} > {sh_quote(output)}")

    before, after = count_seqs(fasta), count_seqs(output)
    if after == 0:
        log.error(f"No contigs ≥{min_len}bp remaining — aborting.")
        sys.exit(1)

    removed_ids = (sorted({rec.id for rec in SeqIO.parse(fasta, 'fasta')} -
                           {rec.id for rec in SeqIO.parse(output, 'fasta')}))
    removed_ids_file = str(out_dir / "removed_short_ids.txt")
    with open(removed_ids_file, 'w') as f:
        f.write("\n".join(removed_ids) + ("\n" if removed_ids else ""))

    meta = {"output": output, "before": before, "after": after, "_fp": fp,
            "removed_ids_file": removed_ids_file}
    ckpt.mark_done(STEP, meta)
    live_step_done(4, t0, f"{before:,} → {after:,} ({before-after:,} removed, <{min_len}bp)")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — CLASSIFICATION (Tiara + Whokaryote)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_tiara(tiara_tsv: str) -> Dict[str, Dict]:
    """Tiara output columns: sequence_id, class_fst_stage, class_snd_stage (+ per-class probabilities if
    --probabilities was used). FAILS LOUD if the file is missing or has no stage column.

    FIX (2026-10-05-euk-focus-v1): read with keep_default_na=False. Tiara writes the literal string 'n/a' as the second-stage
    label of every non-organelle contig and pandas' default NA handling turned it into NaN -> 'nan'; the old code
    preferred the SECOND-stage column, so Tiara's bacteria/archaea/eukarya/prokarya call was discarded for every
    contig. Now the FIRST-stage call is the domain label; the second stage is used only to name an organelle
    ('mitochondrion' / 'plastid') when the first stage says 'organelle'."""
    result = {}
    if not Path(tiara_tsv).exists():
        log.error(f"Tiara output not found: {tiara_tsv} — Tiara likely failed; check tiara.log")
        sys.exit(1)
    df = pd.read_csv(tiara_tsv, sep='\t', dtype=str, keep_default_na=False)
    if df.shape[1] < 2:
        log.error(f"Tiara output at {tiara_tsv} has too few columns ({df.shape[1]}) — "
                  f"unexpected format. Columns found: {list(df.columns)}")
        sys.exit(1)
    id_col = df.columns[0]
    fst_col = next((c for c in ('class_fst_stage', 'first_stage') if c in df.columns), None)
    snd_col = next((c for c in ('class_snd_stage', 'second_stage') if c in df.columns), None)
    if fst_col is None and snd_col is None:
        log.error(f"Tiara output at {tiara_tsv} has none of the expected label columns "
                  f"(class_fst_stage / class_snd_stage / first_stage / second_stage). "
                  f"Columns found: {list(df.columns)} — likely a Tiara version mismatch; "
                  f"check the installed Tiara's actual output header before proceeding.")
        sys.exit(1)
    organelle_names = ("mitochondrion", "mitochondria", "plastid")
    for _, row in df.iterrows():
        fst = str(row[fst_col]).strip().lower() if fst_col else ""
        snd = str(row[snd_col]).strip().lower() if snd_col else ""
        if fst_col is None:
            label = snd or "unknown"
        elif fst == "organelle" and snd in organelle_names:
            label = snd
        elif fst in ("", "n/a", "nan", "none"):
            label = snd if snd in organelle_names else "unknown"
        else:
            label = fst
        result[str(row[id_col])] = {"tiara_label": label, "tiara_stage1": fst,
                                    "tiara_stage2": snd, "row": row.to_dict()}
    return result


_TIARA_PROB_COLUMN = {"bacteria": "bac", "archaea": "arc", "eukarya": "euk", "organelle": "org",
                      "plastid": "pla", "mitochondrion": "mit", "unknown": "unk1"}


def _tiara_alias_prob(row: Dict, label: str):
    """Probability Tiara reports for its called label, read from its REAL column names (bac/arc/euk/org/pla/mit/unk1);
    '' when unavailable. Reported in the classification table (tiara_prob); not used for removal decisions."""
    col = _TIARA_PROB_COLUMN.get(label)
    if not col or not row or col not in row:
        return ""
    try:
        v = float(row[col])
    except (TypeError, ValueError):
        return ""
    return v if 0.0 <= v <= 1.0 else ""


def _tool_versions() -> Dict:
    """Best-effort `--version` of Tiara and Whokaryote (None when the tool does not answer). Part of the reuse fingerprint."""
    out = {}
    for name, cmd in (("tiara", ["tiara", "--version"]), ("whokaryote", ["whokaryote.py", "--version"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out[name] = ((r.stdout or "") + (r.stderr or "")).strip()[:200] or None
        except Exception:
            out[name] = None
    return out


def _tools_fingerprint(fasta, config) -> str:
    return _fingerprint(_file_fingerprint(fasta), config.get("tiara_min_len"), config.get("tiara_prob_cutoff"),
                        config.get("whokaryote_minsize"), config.get("whokaryote_model"), _tool_versions(), "tools-v2")


def _write_tools_provenance(out_dir, fasta, config, mode) -> Dict:
    """Record HOW the Tiara/Whokaryote outputs were obtained: ran / verified (fingerprint) / forced (unverified)."""
    prov = {"mode": mode, "input_fasta": _file_fingerprint(fasta), "tiara_min_len": config.get("tiara_min_len"),
            "whokaryote_minsize": config.get("whokaryote_minsize"), "whokaryote_model": config.get("whokaryote_model"),
            "tool_versions": _tool_versions(), "unverified_reuse": mode == "forced"}
    try:
        (Path(out_dir) / "tools_provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")
    except OSError as e:
        log.warning(f"could not write tools_provenance.json ({e})")
    return prov


def _write_tools_fp(out_dir, fasta, config) -> None:
    try:
        (Path(out_dir) / "tools.fp").write_text(_tools_fingerprint(fasta, config) + "\n")
    except OSError as e:
        log.warning(f"could not write tools.fp ({e}); the next run will re-run Tiara/Whokaryote")


def _classification_reusable(out_dir, fasta, config, tiara_tsv) -> bool:
    """True only if Tiara + Whokaryote outputs exist AND (a) tools.fp matches this exact input FASTA and
    parameters, or (b) reuse was explicitly forced (config reuse_classification_outputs / env
    HYPHAESBIN_REUSE_CLASSIFICATION=1) and both outputs are newer than the input FASTA."""
    model = config.get("whokaryote_model", "T")
    pred = Path(out_dir) / "whokaryote" / f"whokaryote_predictions_{model}.tsv"
    tia = Path(tiara_tsv)
    if not (tia.exists() and tia.stat().st_size > 0 and pred.exists() and pred.stat().st_size > 0):
        return False
    side = Path(out_dir) / "tools.fp"
    if side.exists():
        if side.read_text().strip() == _tools_fingerprint(fasta, config):
            log.info("  Tiara/Whokaryote outputs verified against tools.fp (same input + parameters + tool versions) — reusing them")
            return "verified"
        log.warning("  existing Tiara/Whokaryote outputs came from a different input/parameters — re-running them")
        return False
    forced = bool(config.get("reuse_classification_outputs", False)) or \
        os.environ.get("HYPHAESBIN_REUSE_CLASSIFICATION", "") == "1"
    if not forced:
        return False
    fa_m = Path(fasta).stat().st_mtime
    if tia.stat().st_mtime < fa_m or pred.stat().st_mtime < fa_m:
        log.warning("  reuse was requested but Tiara/Whokaryote outputs are OLDER than the input FASTA — re-running them")
        return False
    log.warning("  REUSING Tiara/Whokaryote outputs WITHOUT fingerprint verification (reuse requested explicitly). "
                "This is only valid if they were produced from exactly this input FASTA and these parameters.")
    return "forced"


def _extract_tiara_probability(row: Dict, tiara_label: str) -> Optional[float]:
    """Look for a genuine per-class probability column matching the called
    label (Tiara's --probabilities column names have changed across
    versions, so we scan for any 0-1 numeric column whose name contains the
    label rather than hardcoding one). Returns None — not 0.0 — when no such
    column is found, so the caller falls back to an explicitly-labeled
    rule-based confidence instead of reporting a fabricated probability."""
    if not tiara_label or tiara_label == "unknown":
        return None
    for col, val in row.items():
        if col in ("sequence_id", "first_stage", "second_stage",
                   "class_fst_stage", "class_snd_stage"):
            continue
        if tiara_label not in str(col).lower():
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if 0.0 <= fval <= 1.0:
            return fval
    return None


_WHOKARYOTE_LABEL_COL_CANDIDATES = ('class', 'prediction', 'category', 'is_eukaryote', 'predicted')


def _parse_whokaryote(predictions_tsv: str, strict: bool = False) -> Tuple[Dict[str, str], str]:
    """Whokaryote output: whokaryote_predictions_[model].tsv — contig id + prediction.

    Returns (result, status) where status is one of:
      'missing'   — file doesn't exist (caller already knows the run failed
                    in this case via the exit code; this just confirms it)
      'malformed' — file exists but couldn't be parsed as a 2+-column TSV
                    (e.g. truncated/empty despite a 0 exit code) — a
                    "success" exit code alone was never enough to trust the
                    output, which is exactly the gap this closes
      'ok'        — parsed successfully (possibly with 0 data rows, which is
                    a legitimate outcome — e.g. every contig below --minsize
                    — and handled by the caller as 'ran_no_predictions')

    The label column is looked up by name against known Whokaryote schemas
    first; only if none match do we fall back to the last column, with a
    loud warning — a positional guess, not a validated column, since this
    project hasn't independently confirmed the exact header across every
    Whokaryote version."""
    if not Path(predictions_tsv).exists():
        return {}, 'missing'
    try:
        df = pd.read_csv(predictions_tsv, sep='\t')
    except Exception as e:
        log.error(f"Whokaryote predictions file at {predictions_tsv} exists but could not be "
                  f"parsed as a TSV ({e}) — treating as a malformed/failed run rather than "
                  f"trusting an unreadable file just because barrnap... er, Whokaryote exited 0.")
        return {}, 'malformed'
    if df.shape[1] < 2:
        log.error(f"Whokaryote predictions file at {predictions_tsv} has too few columns "
                  f"({df.shape[1]}, found: {list(df.columns)}) to contain both a contig id and "
                  f"a class label — refusing to guess. Treating as a malformed run.")
        return {}, 'malformed'
    id_col = df.columns[0]
    label_col = next((c for c in _WHOKARYOTE_LABEL_COL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        if strict:
            log.error(f"Whokaryote predictions file {predictions_tsv} has no known label column (columns: {list(df.columns)}); "
                      f"prokaryote removal will not run on a guessed column -> treating the run as malformed.")
            return {}, 'malformed'
        label_col = df.columns[-1]
        log.warning(f"Whokaryote predictions file at {predictions_tsv} has none of the known "
                    f"label column names {_WHOKARYOTE_LABEL_COL_CANDIDATES} (columns found: "
                    f"{list(df.columns)}) — falling back to the last column ('{label_col}') "
                    f"positionally. This is a guess, not a validated schema match; if "
                    f"classification results look wrong, check this Whokaryote version's "
                    f"actual output header.")
    result = {str(row[id_col]): str(row[label_col]).lower() for _, row in df.iterrows()}
    _bad = {k for k, v in result.items() if v not in ('eukaryote', 'prokaryote')}
    if _bad:
        log.warning(f"Whokaryote: {len(_bad):,} prediction(s) with a label other than eukaryote/prokaryote were treated as NO CALL")
        result = {k: v for k, v in result.items() if k not in _bad}
    return result, 'ok'


_TIARA_TO_CATEGORY = {
    "mit": "Mitochondrial", "mitochondria": "Mitochondrial", "mitochondrion": "Mitochondrial",
    "pla": "Plastid", "plastid": "Plastid",
    "bac": "Bacterial", "bacteria": "Bacterial",
    "arc": "Archaeal", "archaea": "Archaeal",
    "euk": "Eukaryotic", "eukarya": "Eukaryotic",
    "pro": "Prokaryotic_Unclassified", "prokarya": "Prokaryotic_Unclassified",
    "unk": "Unknown", "unknown": "Unknown",
}

# Which step-6 'source' values count as high-confidence when confidence_type
# is 'rule_based' — a discrete allow-list, not a second numeric threshold,
# because a rule-based number (0.85, 0.90, ...) is not on the same scale as
# a real Tiara probability and comparing it against any threshold implies a
# precision it doesn't have. 'tiara_organelle': Tiara's mitochondrial/plastid
# calls are a narrow, well-established k-mer signature even without a usable
# probability column. 'whokaryote+tiara': two complementary classifier signals
# on bacterial/archaeal — agreement is the evidence, not a fabricated number.
# Anything else built on a rule-based fallback ('tiara_only', 'whokaryote'
# alone) is a single unconfirmed heuristic guess and can never trigger
# domain removal, regardless of what number step 6 happened to attach to it.
_RULE_BASED_HIGH_CONF_SOURCES = {"tiara_organelle", "whokaryote+tiara"}


def step6_classification(fasta, outdir, threads, ckpt, config) -> Dict:
    """Runs Tiara (organelle + domain-level, k-mer based) and Whokaryote
    (eukaryote/prokaryote, gene-structure based, Tiara-integrated). Merge
    policy: Tiara's call wins for organelle/mitochondrial/plastid (Whokaryote
    doesn't distinguish these — it's binary euk/prok); Whokaryote's call wins
    for the eukaryote/prokaryote split when it made one (gene-structure is a
    stronger signal for that specific question); anything neither classifier
    could confidently call → 'Unknown'. NOTHING is removed here — this step
    only labels. See step 7 for the actual removal.

    Confidence honesty: where Tiara's own --probabilities output has a real
    per-class probability for the called label, that number is used and
    tagged confidence_type='probability'. Everywhere else (Whokaryote makes
    no per-call probability available in its predictions file, and any
    contig with no matching Tiara probability column) the confidence is a
    documented rule-based heuristic tagged confidence_type='rule_based' —
    never presented as a statistical probability it isn't.

    Conflict handling: if Tiara and Whokaryote make genuinely opposed
    domain-level calls (one says eukaryotic, the other says prokaryotic),
    the contig is NOT force-resolved to either — it's marked
    classification='Unknown', conflict=True, and left for step 11 to weigh
    with the rest of the evidence rather than being silently decided here."""
    STEP = "step6_classification"
    fp = _fingerprint(_file_fingerprint(fasta), config.get("tiara_min_len"),
                       config.get("tiara_prob_cutoff"), config.get("whokaryote_minsize"),
                       config.get("whokaryote_model"), config.get("classification_confidence_high"),
                       bool(config.get("prokaryote_removal_enabled", True)), LOGIC_VERSION)
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(6)

    out_dir = Path(outdir) / "06_classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Tiara ---
    tiara_tsv = str(out_dir / "tiara_out.txt")
    cutoffs = " ".join(str(c) for c in config.get("tiara_prob_cutoff", [0.65, 0.65]))
    tiara_cmd = (f"tiara -i {sh_quote(fasta)} -o {sh_quote(tiara_tsv)} "
                 f"-t {threads} -m {config.get('tiara_min_len', 1000)} "
                 f"-p {cutoffs} --probabilities")
    _reuse_tools = _classification_reusable(out_dir, fasta, config, tiara_tsv)   # '' | 'verified' | 'forced'
    if _reuse_tools:
        log.info("Tiara + Whokaryote: reusing existing outputs (not re-running them)")
        from hyphaesbin.utils.resource_profiler import record_cached
        record_cached('classifier/tiara', tiara_tsv)
    else:
        run_cmd(tiara_cmd, str(out_dir / "tiara.log"), stream=True)
    tiara_results = _parse_tiara(tiara_tsv)   # sys.exit(1)s internally on bad/missing output
    log.info(f"Tiara classified {len(tiara_results):,} contigs")

    # --- Whokaryote (Tiara-integrated model) ---
    whoka_dir = out_dir / "whokaryote"
    model = config.get("whokaryote_model", "T")
    # NOTE: no --f flag here (deliberately removed). Per Whokaryote's own
    # --help text, --f means "new multifastas with only eukaryotes and only
    # prokaryotes. This can take a long time." _parse_whokaryote() below
    # only ever reads whokaryote_predictions_{model}.tsv -- it never opens
    # those split FASTA files -- so --f was paying real runtime for output
    # this pipeline discards. Also NOTE: --threads is passed on the
    # assumption your installed Whokaryote version supports it; the
    # upstream documented CLI does not list it. Either way it can't speed
    # up Whokaryote's internal Prodigal gene-calling step, which has no
    # multithreading support at all regardless of what's passed here.
    whoka_cmd = (f"whokaryote.py --contigs {sh_quote(fasta)} "
                 f"--outdir {sh_quote(str(whoka_dir))} "
                 f"--minsize {config.get('whokaryote_minsize', 1000)} "
                 f"--model {model} --threads {threads}")
    if _reuse_tools:
        whoka_result = subprocess.CompletedProcess(whoka_cmd, 0)
        record_cached('classifier/whokaryote', whoka_dir / f'whokaryote_predictions_{model}.tsv')
    else:
        whoka_result = run_cmd(whoka_cmd, str(out_dir / "whokaryote.log"), allow_fail=True, stream=True)
        pass  # tools.fp is written only after the predictions validate (below)
    whoka_failed = (whoka_result is None) or (getattr(whoka_result, "returncode", 1) != 0)
    predictions_tsv = whoka_dir / f"whokaryote_predictions_{model}.tsv"
    whoka_results, whoka_parse_status = _parse_whokaryote(str(predictions_tsv), strict=bool(config.get("prokaryote_removal_enabled", True)))
    if whoka_failed:
        whoka_status = "failed"
        log.warning("⚠️  Whokaryote run FAILED (see whokaryote.log). Classification for "
                     "this run falls back to Tiara-only — every contig's "
                     "whokaryote_status is recorded as 'failed', NOT indistinguishable "
                     "from 'whokaryote ran and found nothing.'")
    elif whoka_parse_status == 'malformed':
        # A 0 exit code told us nothing here — the output file itself is
        # unreadable or missing a class column. Same downstream effect as
        # 'failed' (Tiara-only fallback) but recorded distinctly so a QC
        # reader can tell "whokaryote crashed" apart from "whokaryote
        # exited 0 but wrote garbage" — different bugs, different fixes.
        whoka_status = "malformed_output"
        log.warning("⚠️  Whokaryote exited successfully but its output could not be validated "
                     "(see errors above) — classification for this run falls back to "
                     "Tiara-only, same as a hard failure.")
    elif whoka_parse_status == 'missing' or not whoka_results:
        whoka_status = "ran_no_predictions"
    else:
        whoka_status = "ok"
    log.info(f"Whokaryote status: {whoka_status} — classified {len(whoka_results):,} contigs "
             f"(contigs <2 genes or below --minsize are absent from its output "
             f"and default to Tiara-only / Unknown below)")

    if not _reuse_tools and whoka_status == "ok":
        _write_tools_fp(out_dir, fasta, config)      # stamp only validated outputs as reusable
    _prok_on = bool(config.get("prokaryote_removal_enabled", True))
    if _prok_on and whoka_status in ("failed", "malformed_output"):
        log.error(f"Prokaryote removal needs Whokaryote, but its status is '{whoka_status}' (see whokaryote.log). "
                  f"Continuing would silently keep every prokaryotic contig. Fix Whokaryote, or set "
                  f"prokaryote_removal_enabled=False to run without prokaryote removal.")
        sys.exit(1)
    if _prok_on and whoka_parse_status == 'missing' and not whoka_failed:
        log.error(f"Whokaryote exited successfully but wrote no predictions file ({predictions_tsv}); prokaryote removal "
                  f"cannot work. Fix Whokaryote, or set prokaryote_removal_enabled=False.")
        sys.exit(1)
    if _prok_on and whoka_status == "ran_no_predictions":
        log.warning("Whokaryote produced no predictions: NO prokaryotes can be removed in step 7 for this run.")
    # --- Merge into one classification table ---
    _recs = [(rec.id, len(rec.seq)) for rec in SeqIO.parse(fasta, "fasta")]
    all_ids = [r_[0] for r_ in _recs]
    _len_of = dict(_recs)  # list, not set —
    rows = []                                                   # preserves FASTA order
    high_conf = float(config.get("classification_confidence_high", 0.80))
    n_conflicts = 0
    for cid in all_ids:
        tiara_entry = tiara_results.get(cid, {})
        tiara_label = tiara_entry.get("tiara_label", "unknown")
        tiara_row = tiara_entry.get("row", {})
        tiara_category = _TIARA_TO_CATEGORY.get(tiara_label, "Unknown")
        tiara_prob = _extract_tiara_probability(tiara_row, tiara_label) if tiara_row else None
        whoka_label = whoka_results.get(cid)  # "eukaryote" / "prokaryote" / None

        tiara_is_euk = tiara_category == "Eukaryotic"
        tiara_is_prok = tiara_category in ("Bacterial", "Archaeal", "Prokaryotic_Unclassified")
        conflict = ((whoka_label == "eukaryote" and tiara_is_prok) or
                    (whoka_label == "prokaryote" and tiara_is_euk))

        if conflict:
            n_conflicts += 1
            category, confidence, confidence_type, source = (
                "Unknown", 0.0, "conflict", "tiara_whokaryote_disagree")
        elif tiara_category in ("Mitochondrial", "Plastid"):
            category, source = tiara_category, "tiara_organelle"
            confidence, confidence_type = (tiara_prob, "probability") if tiara_prob is not None \
                else (0.90, "rule_based")
        elif whoka_label == "eukaryote":
            category, source = "Eukaryotic", "whokaryote"
            confidence, confidence_type = 0.85, "rule_based"  # Whokaryote's predictions
                                                                # file carries no per-call
                                                                # probability to use here
        elif whoka_label == "prokaryote":
            # Whokaryote can't split bac/arc — defer to Tiara if it has an opinion
            if tiara_category in ("Bacterial", "Archaeal", "Prokaryotic_Unclassified"):
                category, source = tiara_category, "whokaryote+tiara"
                confidence, confidence_type = (tiara_prob, "probability") if tiara_prob is not None \
                    else (0.85, "rule_based")
            else:
                category, confidence, confidence_type, source = (
                    "Prokaryotic_Unclassified", 0.75, "rule_based", "whokaryote")
        elif tiara_category in ("Eukaryotic", "Bacterial", "Archaeal"):
            category, source = tiara_category, "tiara_only"
            confidence, confidence_type = (tiara_prob, "probability") if tiara_prob is not None \
                else (0.60, "rule_based")
        else:
            category, confidence, confidence_type, source = "Unknown", 0.0, "no_call", "no_classifier_call"

        rows.append({"contig_id": cid, "classification": category,
                      "confidence": confidence, "confidence_type": confidence_type,
                      "conflict": conflict, "source": source,
                      "tiara_label": tiara_label, "whokaryote_label": whoka_label or "",
                      "tiara_stage1": tiara_entry.get("tiara_stage1", ""), "tiara_prob": _tiara_alias_prob(tiara_row, tiara_label), "tiara_stage2": tiara_entry.get("tiara_stage2", ""), "length": _len_of[cid],
                      "whokaryote_status": whoka_status})

    class_df = pd.DataFrame(rows)
    class_tsv = out_dir / "contig_classification.tsv"
    class_df.to_csv(class_tsv, sep='\t', index=False)

    summary = class_df['classification'].value_counts().to_dict()
    log.info(f"Classification summary: {summary} | conflicts: {n_conflicts:,} | "
             f"whokaryote_status: {whoka_status}")

    meta = {"classification_tsv": str(class_tsv), "summary": summary,
            "high_confidence_threshold": high_conf,
            "n_conflicts": n_conflicts,
            "whokaryote_status": whoka_status, "_fp": fp,
            "tools_provenance": _write_tools_provenance(out_dir, fasta, config, _reuse_tools or "ran")}
    ckpt.mark_done(STEP, meta)
    live_step_done(6, t0, f"{len(rows):,} contigs labelled — {summary}")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — DOMAIN-BASED REMOVAL
# ═══════════════════════════════════════════════════════════════════════════════

_REMOVE_CATEGORIES = {"Bacterial", "Archaeal", "Mitochondrial", "Plastid"}


def _is_high_confidence_removal(confidence_type: str, source: str, confidence,
                                 high_conf: float, rule_based_sources: set) -> bool:
    """SHARED removability policy — used by BOTH step 7 and step 11, so the
    two can never silently diverge (they used to: step 11 had its own
    `confidence >= 0.5` check that ignored confidence_type/source entirely,
    which meant a 'tiara_only' rule-based Bacterial call — confidence 0.60,
    NOT in the allow-list, correctly kept by step 7 — could still get
    flagged for removal here just because 0.60 >= 0.5).

    - confidence_type == 'probability': removable iff confidence >= high_conf.
    - confidence_type == 'rule_based': removable iff `source` is in the
      allow-list — never decided by the number itself.
    - anything else ('conflict', 'no_call', missing): never removable."""
    if confidence_type == 'probability':
        return pd.notna(confidence) and float(confidence) >= high_conf
    if confidence_type == 'rule_based':
        return source in rule_based_sources
    return False


_PROKARYOTE_CATEGORIES = {"Bacterial", "Archaeal", "Prokaryotic_Unclassified"}
_ORGANELLE_CATEGORIES = {"Mitochondrial", "Plastid"}


def _domain_policy(cfg) -> Dict:
    cfg = cfg or {}
    return {"remove_prokaryotes": bool(cfg.get("prokaryote_removal_enabled", True)),
            "prok_min_len": int(cfg.get("prokaryote_min_length", 3000)),
            "remove_organelles": bool(cfg.get("organelle_removal_enabled", False))}


def _domain_removal_decision(classification, confidence_type, source, confidence, length,
                             policy, high_conf, rule_based_sources) -> bool:
    """SHARED by step 7 and step 11 so the two can never disagree.
    Prokaryote (Bacterial / Archaeal / Prokaryotic_Unclassified): removed only when Tiara AND Whokaryote agree
    (step 6 source == 'whokaryote+tiara'; a conflict is 'Unknown', never here) and length >= prok_min_len.
    Organelle (Mitochondrial / Plastid): removed only if organelle_removal_enabled (legacy behaviour, then the
    old high-confidence rule applies). Everything else (Eukaryotic, Unknown, conflicts): never removed."""
    if classification in _PROKARYOTE_CATEGORIES:
        return bool(policy["remove_prokaryotes"] and source == "whokaryote+tiara"
                    and length >= policy["prok_min_len"])
    if classification in _ORGANELLE_CATEGORIES:
        return bool(policy["remove_organelles"] and _is_high_confidence_removal(
            confidence_type, source, confidence, high_conf, rule_based_sources))
    return False


def _domain_retention_reason(classification, source, length, policy) -> str:
    """Why a prokaryote/organelle call was NOT removed (audit table). '' = not a domain call."""
    if classification in _PROKARYOTE_CATEGORIES:
        if not policy["remove_prokaryotes"]:
            return "prokaryote_removal_disabled"
        if source != "whokaryote+tiara":
            return "no_tiara_whokaryote_agreement"
        if length < policy["prok_min_len"]:
            return "below_prokaryote_min_length"
        return "prokaryote_call_retained"
    if classification in _ORGANELLE_CATEGORIES:
        return "organelle_call_flagged_not_removed" if not policy["remove_organelles"] else "organelle_call_below_confidence"
    return ""


def step7_domain_removal(fasta, classification_tsv, high_conf, outdir, ckpt,
                          rule_based_sources: Optional[set] = None,
                          policy: Optional[Dict] = None) -> Dict:
    """Domain-based removal (see _domain_removal_decision for the exact policy).

    Removed contigs are listed per category in removed_<category>_ids.txt (same as before); every
    prokaryote/organelle call that was KEPT is written, with its reason, to retained_domain_calls.tsv."""
    rule_based_sources = rule_based_sources if rule_based_sources is not None else _RULE_BASED_HIGH_CONF_SOURCES
    policy = dict(policy) if policy else _domain_policy(None)
    STEP = "step7_domain_removal"
    fp = _fingerprint(_file_fingerprint(fasta), _file_fingerprint(classification_tsv), high_conf,
                       sorted(rule_based_sources), sorted(policy.items()), LOGIC_VERSION)
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(7)
    log.info(f"  Domain-removal policy: {policy}")

    class_df = pd.read_csv(classification_tsv, sep='\t')
    n_before_coerce = class_df['confidence'].notna().sum()
    class_df['confidence'] = pd.to_numeric(class_df['confidence'], errors='coerce')
    n_uncoercible = n_before_coerce - class_df['confidence'].notna().sum()
    if n_uncoercible > 0:
        log.warning(f"⚠️  {n_uncoercible} row(s) in {classification_tsv} had a non-numeric "
                    f"confidence value — treated as NaN (never high-confidence, contig kept).")
    out_dir = Path(outdir) / "07_domain_removal"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / "domain_filtered.fasta")

    lengths = {rec.id: len(rec.seq) for rec in SeqIO.parse(fasta, "fasta")}   # unknown length -> 0 -> never removed as prokaryote

    removed_by_category = defaultdict(list)
    keep_ids = set()
    retained_rows = []
    for _, row in class_df.iterrows():
        cid, cls = row['contig_id'], row['classification']
        src, length = row.get('source', ''), lengths.get(row['contig_id'], 0)
        if _domain_removal_decision(cls, row.get('confidence_type', ''), src, row['confidence'],
                                    length, policy, high_conf, rule_based_sources):
            removed_by_category[cls].append(cid)
        else:
            keep_ids.add(cid)
            reason = _domain_retention_reason(cls, src, length, policy)
            if not reason and str(row.get('confidence_type', '')) == 'conflict':
                reason = 'tiara_whokaryote_conflict'
            if reason:
                retained_rows.append((cid, cls, src, length, reason))

    _have = set(class_df['contig_id'])
    _nocls = [c for c in lengths if c not in _have]
    if _nocls:
        log.warning(f"⚠️  {len(_nocls):,} FASTA contig(s) have NO row in the classification table -> KEPT (never removed without a call); "
                    f"listed in retained_domain_calls.tsv as no_classification_row")
        keep_ids.update(_nocls)
        retained_rows.extend((c, '(none)', '', lengths[c], 'no_classification_row') for c in _nocls)
    kept = removed = 0
    with open(output, 'w') as out_f:
        for rec in SeqIO.parse(fasta, "fasta"):
            if rec.id in keep_ids:
                SeqIO.write(rec, out_f, "fasta")
                kept += 1
            else:
                removed += 1

    for _old in out_dir.glob("removed_*_ids.txt"):      # stale lists from an earlier run with another policy
        _old.unlink()
    for category, ids in removed_by_category.items():
        fname = f"removed_{category.lower()}_ids.txt"
        with open(out_dir / fname, 'w') as f:
            f.write("\n".join(sorted(ids)) + ("\n" if ids else ""))
        log.info(f"  {category}: {len(ids):,} removed → {fname}")

    with open(out_dir / "retained_domain_calls.tsv", 'w') as f:
        f.write("contig_id\tclassification\tsource\tlength\treason\n")
        for r in retained_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    if retained_rows:
        reasons = pd.Series([r[4] for r in retained_rows]).value_counts().to_dict()
        log.info(f"  prokaryote/organelle calls KEPT (see retained_domain_calls.tsv): {reasons}")

    meta = {"output": output, "kept": kept, "removed": removed,
            "removed_by_category": {k: len(v) for k, v in removed_by_category.items()},
            "removed_ids_dir": str(out_dir), "policy": policy, "_fp": fp}
    ckpt.mark_done(STEP, meta)
    live_step_done(7, t0, f"kept {kept:,}, removed {removed:,} (prokaryote/organelle policy)")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — rDNA MASKING (dual output: masked + unmasked)
# ═══════════════════════════════════════════════════════════════════════════════

def step8_rdna_masking(fasta, outdir, threads, ckpt, barrnap_kingdom: str = "fun") -> Dict:
    """barrnap finds rRNA gene coordinates. We mask them with 'N' in ONE
    copy of the assembly (for composition-sensitive steps like TNF/GC
    consistency in step 11, where rRNA's atypical conserved k-mer content is
    noise) and leave a second, fully UNMASKED copy untouched. Nothing is
    ever dropped for containing rRNA.

    `barrnap_kingdom` (config['barrnap_kingdom'], default "fun") is passed
    straight to barrnap's `--kingdom`. NOTE: accepted values are barrnap-BUILD-
    dependent — some builds accept euk/bac/arc/mito, others only accept
    bac/arc/fun and reject "euk" outright with a hard error (confirmed live
    on a bioconda barrnap install: `[barrnap] ERROR: Invalid --kingdom 'euk'.
    Choose from: bac arc fun`), which previously made this step fail soft
    every run (rrna_status='failed', 0 features masked, effectively a no-op)
    without anyone noticing since allow_fail=True swallows it. "fun" is
    barrnap's fungus-specific rRNA model — the scientifically correct choice
    for this fungi-specific pipeline, not just a safe fallback. Exposed here
    as config rather than hardcoded so a run can be pointed at a different
    kingdom (run `barrnap --help` in the target env first to confirm accepted
    values), or this can be swapped for a custom HMM database run outside
    this pipeline, and the fingerprint tracks it so switching kingdoms
    invalidates the cache.

    Which variant goes where (see module_routing below) — READ MAPPING
    (step 9) uses the UNMASKED copy: mapping reads against N-masked
    sequence would suppress real coverage over rRNA loci, silently making
    genuinely-covered contigs look poorly supported. Only composition
    scoring (step 11's GC/TNF check) uses the masked copy; final outputs
    keep both variants side by side."""
    STEP = "step8_rdna_masking"
    fp = _fingerprint(_file_fingerprint(fasta), barrnap_kingdom)
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(8)

    out_dir = Path(outdir) / "08_rdna"
    out_dir.mkdir(parents=True, exist_ok=True)
    gff = str(out_dir / "rrna.gff")
    masked_out = str(out_dir / "masked.fasta")
    unmasked_out = str(out_dir / "unmasked.fasta")

    barrnap_result = run_cmd(f"barrnap --kingdom {sh_quote(barrnap_kingdom)} --threads {threads} "
                              f"{sh_quote(fasta)} > {sh_quote(gff)}",
                              str(out_dir / "barrnap.log"), allow_fail=True)
    barrnap_failed = (barrnap_result is None) or (getattr(barrnap_result, "returncode", 1) != 0)
    if barrnap_failed:
        log.warning("⚠️  barrnap FAILED (see barrnap.log) — rRNA masking could not run this "
                     "run; the assembly proceeds effectively UNMASKED. Recorded as "
                     "rrna_status='failed', not conflated with 'ran fine, found no rRNA.'")

    intervals = defaultdict(list)
    n_hits = 0
    if Path(gff).exists():
        with open(gff) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                cols = line.rstrip('\n').split('\t')
                if len(cols) < 5:
                    continue
                try:
                    start, end = int(cols[3]) - 1, int(cols[4])
                except ValueError:
                    continue
                intervals[cols[0]].append((start, end))
                n_hits += 1

    masked_contigs = 0
    masked_ids = []
    shutil.copy(fasta, unmasked_out)
    with open(masked_out, 'w') as out_f:
        for rec in SeqIO.parse(fasta, 'fasta'):
            hits = intervals.get(rec.id)
            seq = str(rec.seq)
            if hits:
                seq_list = list(seq)
                for start, end in hits:
                    start, end = max(0, start), min(len(seq_list), end)
                    for i in range(start, end):
                        seq_list[i] = 'N'
                seq = ''.join(seq_list)
                masked_contigs += 1
                masked_ids.append(rec.id)
            out_f.write(f">{rec.id}\n{seq}\n")

    masked_ids_file = str(out_dir / "rrna_masked_ids.txt")
    with open(masked_ids_file, 'w') as f:
        f.write("\n".join(sorted(masked_ids)) + ("\n" if masked_ids else ""))

    rrna_status = "failed" if barrnap_failed else ("ok_no_hits" if n_hits == 0 else "ok")

    meta = {"masked_fasta": masked_out, "unmasked_fasta": unmasked_out,
            "gff": gff, "n_hits": n_hits, "masked_contigs": masked_contigs, "_fp": fp,
            "masked_ids_file": masked_ids_file, "rrna_status": rrna_status,
            "module_routing": {
                "read_mapping": "unmasked",   # step 9 — see docstring above
                "tnf_composition": "masked", "te_analysis": "unmasked",
                "binning_clustering": "masked", "final_bin_output": "unmasked"}}
    ckpt.mark_done(STEP, meta)
    live_step_done(8, t0, f"rrna_status={rrna_status} — {n_hits} rRNA feature(s) masked across "
                          f"{masked_contigs} contig(s); none dropped")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SINGLE MAPPING PASS
# ═══════════════════════════════════════════════════════════════════════════════

def _map_one_sample(sample, r1, r2, fasta, preset, read_type, threads_per_worker,
                     map_dir) -> Tuple[str, Optional[str], bool, str]:
    """Maps one sample's reads and returns (sample, bam_out_or_None, ok, detail).
    Deliberately uses run_cmd(..., allow_fail=True) rather than the default
    run_cmd — the default calls sys.exit(1) on failure, which inside a
    ThreadPoolExecutor worker only kills that one thread (raises SystemExit,
    silently swallowed by the executor) rather than the whole pipeline, so a
    failed sample could otherwise vanish instead of failing the run. Here,
    failure is returned as a normal value and checked by the caller on the
    main thread, which decides whether to sys.exit — so one sample's mapping
    failure can never be silently lost while others keep running."""
    bam_out = str(map_dir / f"{sample}.bam")
    log_file = str(map_dir / f"minimap2_{sample}.log")
    if read_type in ("short", "hybrid") and r2:
        reads_arg = f"{sh_quote(r1)} {sh_quote(r2)}"
    else:
        reads_arg = sh_quote(r1)
    # `bash -o pipefail -c '...'` matters here, not just style: plain
    # shell=True runs /bin/sh (dash on most distros), where a pipeline's
    # exit code is only the LAST command's (samtools sort) — a minimap2
    # crash mid-pipe would be silently swallowed as long as samtools sort
    # still exited 0 on whatever partial/empty input it received, so a
    # failed sample could produce a "successful" near-empty BAM instead of
    # being caught here.
    pipe_cmd = (f"minimap2 {preset} -t {threads_per_worker} {sh_quote(fasta)} {reads_arg} "
                f"2>{sh_quote(log_file)} | samtools sort -@ {threads_per_worker} -o {sh_quote(bam_out)} -")
    cmd = f"bash -o pipefail -c {sh_quote(pipe_cmd)} && samtools index {sh_quote(bam_out)}"
    result = run_cmd(cmd, allow_fail=True)
    if result is None or getattr(result, "returncode", 1) != 0:
        return (sample, None, False, f"mapping failed (see {log_file})")
    return (sample, bam_out, True, "ok")


def step9_single_mapping(fasta, samples_df, reads_dir, outdir, threads, ckpt, config) -> Dict:
    """The one READ-mapping pass in the pipeline (step 10's rescue branch
    does a separate contig-vs-reference alignment — different operation,
    see that step's docstring). Every coverage-based decision downstream
    (steps 10, 11, 12) reads from this one table. `fasta` must be the
    UNMASKED assembly (step 8's unmasked_fasta) — mapping against the
    rDNA-masked copy would suppress real coverage over rRNA loci.

    Samples are mapped CONCURRENTLY, not one-at-a-time: with `threads=80`
    and 6 samples, mapping them serially means 5 of the 6 sit idle while
    sample 1 alone uses all 80 threads (minimap2/samtools rarely scale
    usefully much past 16-24 threads anyway). Instead, `max_mapping_workers`
    (config, default "auto") picks how many samples run at once, and each
    gets `threads // n_workers` of the total budget — same total CPU, spread
    across samples instead of piled onto one at a time. Set
    max_mapping_workers=1 for the old fully-sequential behavior."""
    STEP = "step9_mapping"
    read_fps = []
    for _, row in samples_df.iterrows():
        for col in ('r1', 'r2'):
            value = row.get(col) if col in samples_df.columns else None
            if pd.notna(value) and str(value).strip():
                read_fps.append(_file_fingerprint(resolve_read_path(reads_dir, value)))
    fp = _fingerprint(_file_fingerprint(fasta), read_fps, config.get("read_type"))
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    read_type = config.get("read_type", "short")
    live_step_start(9, f"read_type: {read_type}")

    preset = get_minimap2_preset(read_type)
    map_dir = Path(outdir) / "09_mapping"
    map_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(samples_df)
    min_threads_per_sample = max(1, int(config.get("min_threads_per_sample", 8)))
    workers_cfg = config.get("max_mapping_workers", "auto")
    if workers_cfg in (None, "auto", ""):
        n_workers = max(1, min(n_samples, max(1, threads // min_threads_per_sample)))
    else:
        n_workers = max(1, min(n_samples, int(workers_cfg)))
    threads_per_worker = max(1, threads // n_workers)
    log.info(f"  Mapping {n_samples} sample(s) with {n_workers} concurrent worker(s), "
             f"{threads_per_worker} thread(s) each (total budget: {threads})")

    sample_args = []
    for _, row in samples_df.iterrows():
        sample = row['sample']
        r1 = resolve_read_path(reads_dir, row['r1'])
        r2_value = row.get('r2') if 'r2' in samples_df.columns else None
        r2 = (resolve_read_path(reads_dir, r2_value)
              if pd.notna(r2_value) and str(r2_value).strip() else None)
        sample_args.append((sample, r1, r2))

    # bam_by_sample keeps results keyed by sample name so the final bam_files
    # list can be rebuilt in the samples_df's original order regardless of
    # which worker happened to finish first — order shouldn't matter for
    # CoverM's `-b` flags, but keeping it stable makes logs/debugging saner.
    bam_by_sample: Dict[str, Optional[str]] = {}
    failures = []
    if n_workers == 1:
        # No thread pool at all for the common single-worker case (n_samples
        # == 1, or max_mapping_workers explicitly set to 1) — identical to
        # the pipeline's original fully-sequential behavior, just routed
        # through the same _map_one_sample() helper so there's one code path
        # instead of two to keep in sync.
        for sample, r1, r2 in sample_args:
            s, bam_out, ok, detail = _map_one_sample(sample, r1, r2, fasta, preset, read_type,
                                                       threads_per_worker, map_dir)
            bam_by_sample[s] = bam_out
            if not ok:
                failures.append((s, detail))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_map_one_sample, sample, r1, r2, fasta, preset, read_type,
                                    threads_per_worker, map_dir): sample
                       for sample, r1, r2 in sample_args}
            for fut in as_completed(futures):
                s, bam_out, ok, detail = fut.result()
                bam_by_sample[s] = bam_out
                if not ok:
                    failures.append((s, detail))

    if failures:
        log.error(f"Mapping failed for {len(failures)}/{n_samples} sample(s): {failures} — "
                  f"refusing to build a coverage table from a partial/incomplete set of BAMs.")
        sys.exit(1)

    bam_files = [bam_by_sample[s] for s, _, _ in sample_args]

    coverage_tsv = str(map_dir / "coverage_table.tsv")
    bam_str = " ".join(f"-b {sh_quote(b)}" for b in bam_files)
    # Request both depth (mean) and breadth (covered_fraction) — a contig
    # with one covered base and a high mean from a local repeat shouldn't
    # pass adaptive filtering just because its mean depth looks fine.
    run_cmd(f"coverm contig {bam_str} -m mean covered_fraction -t {threads} -o {sh_quote(coverage_tsv)}",
            str(map_dir / "coverm.log"))

    header_cols = pd.read_csv(coverage_tsv, sep='\t', nrows=0).columns.tolist()[1:]
    if len(header_cols) != len(set(header_cols)):
        dupes = [c for c in set(header_cols) if header_cols.count(c) > 1]
        log.error(f"CoverM's output header has duplicate column name(s): {dupes} "
                  f"(full header: {header_cols}) — cannot safely tell samples apart. "
                  f"This usually means two samples produced identically-named BAM files.")
        sys.exit(1)

    depth_cols, breadth_cols = _split_coverm_columns(header_cols)
    n_samples = len(samples_df)
    if len(depth_cols) != n_samples:
        log.error(f"CoverM produced {len(depth_cols)} depth column(s) but the samples table "
                  f"has {n_samples} sample(s) (depth columns found: {depth_cols}). A mismatch "
                  f"here means a sample's BAM is missing from CoverM's input, or a column was "
                  f"misclassified as breadth instead of depth — refusing to proceed with a "
                  f"coverage table that doesn't map 1:1 onto the sample list.")
        sys.exit(1)
    if not breadth_cols:
        log.warning(f"Could not identify breadth/covered-fraction columns in CoverM's "
                    f"header ({header_cols}) — breadth-based filtering unavailable this run.")
    elif len(breadth_cols) != n_samples:
        log.warning(f"CoverM produced {len(breadth_cols)} breadth column(s) for {n_samples} "
                    f"sample(s) — breadth gate will run on a partial column set "
                    f"({breadth_cols}); treat step 10's breadth-based decisions with caution "
                    f"this run.")

    meta = {"coverage_tsv": coverage_tsv, "bam_files": bam_files,
            "depth_cols": depth_cols, "breadth_cols": breadth_cols, "_fp": fp}
    ckpt.mark_done(STEP, meta)
    live_step_done(9, t0, f"mapped {len(bam_files)} sample(s), one pass — "
                          f"{len(depth_cols)} depth col(s), {len(breadth_cols)} breadth col(s)")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10 — ADAPTIVE FILTER (+ reference rescue for single-sample contigs)
# ═══════════════════════════════════════════════════════════════════════════════

def _rescue_contigs_batch(seq_records: List, reference_fasta: str, out_dir: Path,
                           config: Dict, threads: int = 1) -> Dict[str, Tuple[bool, str]]:
    """minimap2 ALL gray-zone single-sample candidate contigs against the
    fungal reference set in ONE alignment call. Returns {contig_id:
    (rescued: bool, detail: str)} for every id in seq_records.

    PERFORMANCE FIX (this session): this used to run as
    `_rescue_single_sample_contig`, called once PER CONTIG — each call
    wrote a 1-contig query FASTA and launched its own minimap2 subprocess,
    which means minimap2 re-read and re-indexed the ENTIRE reference
    (potentially a multi-GB fungal genome collection) from scratch for
    every single gray-zone contig needing rescue. At real scale (tens of
    thousands of single-sample gray-zone contigs per run) that's tens of
    thousands of redundant full-reference re-indexing passes — the
    difference between "rescue is usable" and "rescue effectively never
    finishes" once a real reference database is configured (see the
    database discussion this session: RefSeq/JGI fungal genomes are large
    enough that this would have made rescue impractical). Fixed by writing
    every candidate into ONE combined query FASTA and running exactly ONE
    minimap2 invocation (reference indexed once, `-t threads` for further
    speed), then grouping the single combined PAF's hits by query name.

    KNOWN LIMITATION (state this in your methods): this only rescues contigs
    that resemble something already in `reference_fasta`. A genuinely novel,
    undescribed fungal lineage — which is exactly what soil metagenomics
    surveys are often trying to find — may have no close reference and will
    NOT be rescued here, regardless of whether it's real. Well-characterized
    fungi (common soil taxa, model organisms, anything with a sequenced
    relative) rescue reliably; rare/novel/undersampled fungi do not."""
    results: Dict[str, Tuple[bool, str]] = {}
    if not seq_records:
        return results

    query_fa = out_dir / "_rescue_batch_query.fasta"
    with open(query_fa, 'w') as qf:
        for rec in seq_records:
            SeqIO.write(rec, qf, "fasta")
    paf = out_dir / "_rescue_batch.paf"
    run_cmd_allow_fail(
        f"minimap2 -x asm10 -t {max(1, int(threads))} {sh_quote(reference_fasta)} "
        f"{sh_quote(str(query_fa))} > {sh_quote(str(paf))}"
    )

    best: Dict[str, Tuple[float, float]] = {}   # qname -> (best_identity, qcov_at_best_identity)
    if paf.exists() and paf.stat().st_size > 0:
        with open(paf) as f:
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) < 12:
                    continue
                qname = cols[0]
                qlen, matches, aln_len = int(cols[1]), int(cols[9]), int(cols[10])
                identity = 100.0 * matches / aln_len if aln_len else 0.0
                qcov = aln_len / qlen if qlen else 0.0
                if identity > best.get(qname, (0.0, 0.0))[0]:
                    best[qname] = (identity, qcov)

    min_id = float(config.get("rescue_min_identity", 75.0))
    min_qc = float(config.get("rescue_min_query_cov", 0.5))
    for rec in seq_records:
        best_identity, best_qcov = best.get(rec.id, (0.0, 0.0))
        if rec.id in best and best_identity >= min_id and best_qcov >= min_qc:
            results[rec.id] = (True, f"reference_rescued ({best_identity:.1f}% ID, {best_qcov:.0%} cov)")
        elif rec.id in best:
            results[rec.id] = (False, f"below threshold ({best_identity:.1f}% ID, {best_qcov:.0%} cov)")
        else:
            results[rec.id] = (False, "no_hit")

    for p in (query_fa, paf):
        p.unlink(missing_ok=True)
    return results


def _adaptive_stats(cov_df, depth_cols, breadth_cols, stat, min_breadth):
    """Per-contig (coverage, breadth) used by the step-10 gate.
    'mean'   : mean over samples (legacy; dilutes a genome present in only one sample).
    'max'    : max depth and max breadth taken SEPARATELY (can combine depth from sample A with breadth from sample B).
    'sample' : depth and breadth of the SAME sample. coverage = best depth among samples whose breadth >= min_breadth
               (0 if none), breadth = that sample's breadth (best breadth overall if no sample qualifies). So
               coverage >= adaptive_min_cov  <=>  one sample meets BOTH thresholds. Needs depth/breadth columns paired
               by sample name; if they cannot be paired it falls back to 'max' with a warning."""
    idx = cov_df.index
    D = cov_df[depth_cols].to_numpy(dtype=float)
    if not breadth_cols:
        return pd.Series(D.mean(axis=1) if stat == "mean" else D.max(axis=1), index=idx), pd.Series(1.0, index=idx)
    B = cov_df[breadth_cols].to_numpy(dtype=float)
    if stat == "mean":
        return pd.Series(D.mean(axis=1), index=idx), pd.Series(B.mean(axis=1), index=idx)
    if stat == "sample":
        dk = [re.sub(r"\s*mean$", "", c, flags=re.I).strip() for c in depth_cols]
        bk = [re.sub(r"\s*(covered[ _]?fraction|breadth)$", "", c, flags=re.I).strip() for c in breadth_cols]
        if len(set(dk)) == len(dk) and sorted(dk) == sorted(bk):
            B = B[:, [bk.index(k) for k in dk]]
            elig = B >= float(min_breadth)
            has = elig.any(axis=1)
            j = np.where(elig, D, -np.inf).argmax(axis=1)
            r = np.arange(len(D))
            return (pd.Series(np.where(has, D[r, j], 0.0), index=idx),
                    pd.Series(np.where(has, B[r, j], B.max(axis=1)), index=idx))
        log.error(f"adaptive_cov_stat='sample' needs depth and breadth columns that pair by sample name, but depth columns "
                  f"{depth_cols} and breadth columns {breadth_cols} do not pair. NOT falling back silently (separate maxima "
                  f"can keep a contig that no single sample supports). Set adaptive_cov_stat to 'max' or 'mean' explicitly, "
                  f"or fix the CoverM column names.")
        sys.exit(1)
    return pd.Series(D.max(axis=1), index=idx), pd.Series(B.max(axis=1), index=idx)


def step10_adaptive_filter(fasta, coverage_tsv, depth_cols, breadth_cols, outdir, ckpt, config,
                            threads: int = 1, unmasked_fasta: Optional[str] = None) -> Dict:
    """Length/coverage triage. `fasta` is the MASKED assembly (used for the
    length/coverage decisions and for what's actually written to `output`).
    `unmasked_fasta` (step 8's unmasked copy), when given, is used ONLY for
    the rescue-candidate sequences handed to `_rescue_contigs_batch` — rDNA
    loci are N-masked in `fasta`, and aligning a masked candidate against
    the reference can silently depress both %identity and query-coverage
    over any contig carrying an rRNA gene, understating a real rescue hit.
    Falls back to the masked sequence (old behavior) if `unmasked_fasta` is
    not supplied, so old call sites keep working.

    NOTE on 'one mapping pass': this step's rescue
    branch (`_rescue_contigs_batch`) runs its own minimap2 alignment of the
    gray-zone single-sample candidates against a reference FASTA — that's a
    contig-vs-reference identity check, not a read-mapping pass, so it
    doesn't contradict step 9 being the pipeline's only READ-mapping step.
    Stated precisely: one read-to-assembly mapping pass, plus at most ONE
    batched reference-rescue alignment covering every single-sample
    gray-zone candidate this run (see PERFORMANCE FIX note on
    `_rescue_contigs_batch` — this used to be one minimap2 subprocess PER
    CONTIG, which re-indexed the whole reference every time).

    `depth_cols`/`breadth_cols` come from step 9's explicit column split —
    never re-inferred here by excluding known column names, which breaks
    silently the moment CoverM's output includes an unanticipated column.
    When breadth_cols is empty (CoverM's covered-fraction columns couldn't
    be identified this run), the breadth gate is skipped rather than
    crashing — mean-depth + n_samples still applies as before.

    PERFORMANCE FIX (this session): the per-contig loop below used to look
    up each contig's coverage row via `cov_df.loc[rec.id]`, which allocates
    a new pandas Series on every call — real overhead at ~190,000-contig
    scale. Replaced with a single `cov_df[...].to_dict('index')` conversion
    done ONCE before the loop, then plain dict lookups inside it."""
    STEP = "step10_adaptive_filter"
    min_len = int(config.get("min_contig_length", 1000))
    mid_ceiling = int(config.get("adaptive_gray_ceiling") or (min_len * 2))
    if mid_ceiling < min_len:
        log.error(f"adaptive_gray_ceiling ({mid_ceiling}) must be >= min_contig_length ({min_len}).")
        sys.exit(1)
    min_cov = float(config.get("adaptive_min_cov", 2.0))
    min_samples = int(config.get("adaptive_min_samples", 2))
    min_breadth = float(config.get("adaptive_min_breadth", 0.3))
    # rescue_wanted: the operator asked for rescue (rescue_enabled=True, the
    # default). rescue_ready: rescue can actually run (a reference was also
    # supplied). Conflating these used to mean an operator who left
    # rescue_reference_fasta empty — the DEFAULT_CONFIG value — silently got
    # single-sample gray-zone contigs REMOVED instead of rescued, with only
    # a validation-time warning to notice it by. Now: rescue_wanted-but-not-
    # ready contigs are RETAINED, flagged 'uncertain_no_rescue_reference',
    # never auto-removed just because the operator forgot a config field.
    rescue_wanted = bool(config.get("rescue_enabled", True))
    rescue_ready = rescue_wanted and bool(config.get("rescue_reference_fasta"))
    fp = _fingerprint(_file_fingerprint(fasta),
                       _file_fingerprint(unmasked_fasta) if unmasked_fasta else "",
                       _file_fingerprint(coverage_tsv),
                       min_len, mid_ceiling, min_cov, min_samples, min_breadth, bool(breadth_cols),
                       rescue_wanted, rescue_ready, config.get("rescue_reference_fasta"),
                       config.get("rescue_min_identity"), config.get("rescue_min_query_cov"),
                       str(config.get("adaptive_cov_stat", "sample")).lower(), LOGIC_VERSION)
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    rescue_status = "ON" if rescue_ready else ("WANTED_NO_REFERENCE (retaining, not removing)"
                                                if rescue_wanted else "OFF")
    live_step_start(10, f"gray zone: {min_len}-{mid_ceiling}bp | rescue: {rescue_status} | "
                        f"breadth gate: {'ON (>=' + str(min_breadth) + ')' if breadth_cols else 'unavailable'}")

    out_dir = Path(outdir) / "10_adaptive_filter"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / "filtered.fasta")

    cov_df = pd.read_csv(coverage_tsv, sep='\t', index_col=0)
    cov_df['mean_cov'], cov_df['mean_breadth'] = _adaptive_stats(
        cov_df, depth_cols, breadth_cols, str(config.get("adaptive_cov_stat", "sample")).lower(), min_breadth)
    cov_df['n_samples'] = (cov_df[depth_cols] > 0).sum(axis=1)
    if breadth_cols:
        pass  # computed together with mean_cov by _adaptive_stats (same-sample pairing)
    else:
        cov_df['mean_breadth'] = 1.0  # gate disabled: never fails the check below
    # PERFORMANCE FIX: one dict conversion up front instead of a `.loc[id]`
    # pandas Series allocation per contig inside the loop below (see
    # function docstring).
    cov_lookup: Dict[str, Dict[str, float]] = cov_df[['mean_cov', 'n_samples', 'mean_breadth']].to_dict('index')

    stats = {'lt_min': 0, 'ge_ceiling': 0, 'mid_multi_sample_ok': 0,
             'mid_rescued': 0, 'mid_rescue_failed': 0, 'mid_removed_no_coverage': 0,
             'ge_ceiling_low_breadth_flagged': 0, 'mid_retained_no_rescue_reference': 0}
    rescue_log = []
    removed_ids = []   # (contig_id, reason) — full per-contig audit trail for step 13
    flagged_no_rescue_reference_ids = []  # retained but need a step-11 warning
    flagged_low_breadth_ids = []  # long contig, high mean depth, very low breadth — retained
                                   # but this is exactly the "one covered base, high mean from a
                                   # local repeat" pattern; previously only an aggregate count
                                   # (stats['ge_ceiling_low_breadth_flagged']) with no per-contig
                                   # ID, so it could never reach the master decision table
    rescued_ids = []  # retained via step-10's reference-rescue alignment, not organically —
                       # previously visible only in rescue_attempts.tsv, invisible in the
                       # master table alongside every other retain/warn signal

    # PASS 1: classify every contig. Everything except the single-sample
    # gray-zone rescue candidates is decided immediately; rescue candidates
    # are deferred to a single BATCHED minimap2 call after this loop (see
    # `_rescue_contigs_batch` docstring for why: one alignment call for
    # every candidate instead of one subprocess + one full reference
    # re-index PER CONTIG).
    records = list(SeqIO.parse(fasta, 'fasta'))
    keep_flags = [False] * len(records)     # aligned to `records`, filled in below
    rescue_candidates = []                   # list of (index_in_records, rec, length, mean_cov, n_samples)

    for i, rec in enumerate(records):
        length = len(rec.seq)
        row = cov_lookup.get(rec.id)
        mean_cov = row['mean_cov'] if row is not None else 0
        n_samples = row['n_samples'] if row is not None else 0
        mean_breadth = row['mean_breadth'] if row is not None else 0.0

        if length < min_len:
            stats['lt_min'] += 1
            removed_ids.append((rec.id, f"<{min_len}bp (unexpected at this stage — "
                                         f"check for a min_contig_length config change)"))
            continue
        if length >= mid_ceiling:
            # Kept regardless of breadth (length alone is strong evidence
            # here), but a long contig with high mean depth and very low
            # breadth — the "one covered base, high mean from a local
            # repeat" case — is flagged for step 11 rather than silently
            # passed through as clean.
            keep_flags[i] = True
            stats['ge_ceiling'] += 1
            if breadth_cols and mean_breadth < min_breadth:
                stats['ge_ceiling_low_breadth_flagged'] += 1
                flagged_low_breadth_ids.append(rec.id)
            continue

        # gray zone: min_len <= length < mid_ceiling. Breadth gate only
        # applies when breadth columns were actually identified this run
        # (mean_breadth defaults to 1.0 / always-pass otherwise).
        if mean_cov >= min_cov and n_samples >= min_samples and mean_breadth >= min_breadth:
            keep_flags[i] = True
            stats['mid_multi_sample_ok'] += 1
            continue

        # single-sample (or below-threshold-samples) support
        if mean_cov >= min_cov and n_samples < min_samples:
            if rescue_ready:
                rescue_candidates.append((i, rec, length, mean_cov, n_samples))
                continue
            elif rescue_wanted:
                # rescue_enabled=True but no rescue_reference_fasta was
                # configured — a config gap, not evidence against this
                # contig. RETAIN it, and hand its ID to step 11 so the
                # missing check becomes one more warning signal in the
                # existing retain/retain_with_warning/uncertain/remove
                # vocabulary — not a fifth status only step 10 knows about.
                keep_flags[i] = True
                stats['mid_retained_no_rescue_reference'] += 1
                flagged_no_rescue_reference_ids.append(rec.id)
                continue
            # else: rescue explicitly disabled (rescue_enabled=False) —
            # falls through to the standard removal below, as before.

        stats['mid_removed_no_coverage'] += 1
        removed_ids.append((rec.id, f"gray-zone, below coverage/breadth/sample thresholds "
                                     f"(mean_cov={mean_cov:.2f}, n_samples={int(n_samples)}, "
                                     f"mean_breadth={mean_breadth:.2f})"))

    # PASS 2: ONE batched rescue alignment for every gray-zone single-sample
    # candidate collected above (empty candidate list = no minimap2 call at
    # all, same as before when rescue never triggers).
    if rescue_candidates:
        # Use the UNMASKED sequence for the rescue alignment itself, not the
        # masked one `rec` (from `fasta`) — rDNA loci are N-masked in the
        # masked copy, and aligning masked bases against the reference can
        # depress both %identity and query-coverage for any candidate that
        # happens to carry an rRNA gene, understating a real rescue hit.
        # Falls back to the masked record if no unmasked_fasta was given.
        if unmasked_fasta:
            _unmasked_lookup = SeqIO.to_dict(SeqIO.parse(unmasked_fasta, 'fasta'))
            rescue_seq_records = [_unmasked_lookup.get(rec.id, rec) for _, rec, _, _, _ in rescue_candidates]
        else:
            rescue_seq_records = [rec for _, rec, _, _, _ in rescue_candidates]
        rescue_results = _rescue_contigs_batch(
            rescue_seq_records,
            config['rescue_reference_fasta'], out_dir, config, threads=threads)
        for i, rec, length, mean_cov, n_samples in rescue_candidates:
            rescued, detail = rescue_results[rec.id]
            rescue_log.append({"contig_id": rec.id, "length": length,
                                "mean_cov": mean_cov, "n_samples": int(n_samples),
                                "rescue_result": detail})
            if rescued:
                keep_flags[i] = True
                stats['mid_rescued'] += 1
                rescued_ids.append(rec.id)
            else:
                stats['mid_rescue_failed'] += 1
                removed_ids.append((rec.id, f"gray-zone single-sample, rescue failed: {detail}"))

    # PASS 3: write output in original FASTA order (order matters for
    # downstream steps' assumptions about contig ordering being preserved).
    with open(output, 'w') as out_f:
        for rec, keep in zip(records, keep_flags):
            if keep:
                SeqIO.write(rec, out_f, 'fasta')

    if rescue_log:
        pd.DataFrame(rescue_log).to_csv(out_dir / "rescue_attempts.tsv", sep='\t', index=False)

    removed_ids_file = str(out_dir / "removed_step10_ids.tsv")
    with open(removed_ids_file, 'w') as f:
        f.write("contig_id\treason\n")
        for cid, reason in removed_ids:
            f.write(f"{cid}\t{reason}\n")

    flagged_no_rescue_reference_file = str(out_dir / "flagged_no_rescue_reference_ids.txt")
    with open(flagged_no_rescue_reference_file, 'w') as f:
        f.write("\n".join(sorted(flagged_no_rescue_reference_ids)) +
                ("\n" if flagged_no_rescue_reference_ids else ""))

    flagged_low_breadth_file = str(out_dir / "flagged_low_breadth_ids.txt")
    with open(flagged_low_breadth_file, 'w') as f:
        f.write("\n".join(sorted(flagged_low_breadth_ids)) +
                ("\n" if flagged_low_breadth_ids else ""))

    rescued_ids_file = str(out_dir / "rescued_ids.txt")
    with open(rescued_ids_file, 'w') as f:
        f.write("\n".join(sorted(rescued_ids)) + ("\n" if rescued_ids else ""))

    total_kept = (stats['ge_ceiling'] + stats['mid_multi_sample_ok'] +
                  stats['mid_rescued'] + stats['mid_retained_no_rescue_reference'])
    log.info(f"  ≥{mid_ceiling}bp kept:               {stats['ge_ceiling']:,}")
    log.info(f"  gray-zone, multi-sample kept:  {stats['mid_multi_sample_ok']:,}")
    log.info(f"  gray-zone, single-sample RESCUED: {stats['mid_rescued']:,}")
    log.info(f"  gray-zone, rescue attempted & FAILED: {stats['mid_rescue_failed']:,}")
    log.info(f"  gray-zone, retained (no rescue reference configured): "
             f"{stats['mid_retained_no_rescue_reference']:,}")
    log.info(f"  gray-zone, no coverage/removed: {stats['mid_removed_no_coverage']:,}")
    log.info(f"  <{min_len}bp removed:                {stats['lt_min']:,}")

    meta = {"output": output, "stats": stats, "total_kept": total_kept,
            "rescue_log": str(out_dir / "rescue_attempts.tsv") if rescue_log else None,
            "removed_ids_file": removed_ids_file,
            "flagged_no_rescue_reference_file": flagged_no_rescue_reference_file,
            "flagged_low_breadth_file": flagged_low_breadth_file,
            "rescued_ids_file": rescued_ids_file, "_fp": fp}
    ckpt.mark_done(STEP, meta)
    live_step_done(10, t0, f"{total_kept:,} kept ({stats['mid_rescued']} rescued via reference, "
                          f"{stats['mid_retained_no_rescue_reference']} retained-no-reference)")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11 — MULTI-SIGNAL CONTIG SCORING
# ═══════════════════════════════════════════════════════════════════════════════

# LOCAL FALLBACK TNF IMPLEMENTATION — see config['tnf_vector_fn'] and
# config['enable_tnf_qc_signal'] in DEFAULT_CONFIG above. This exists so
# preprocessing.py keeps working as a standalone script without depending on
# the rest of the HyphaeSBin package tree, NOT because preprocessing should
# own a second definition of TNF long-term. If the encoder module's TNF
# function (e.g. hyphaesbin/composition/TNF_gene/tnf_gene.py) is available at
# call time, wire it in via config['tnf_vector_fn'] instead of relying on
# this copy — two independently-maintained 136-dim TNF implementations over
# the same contigs is exactly the kind of quiet divergence that's easy to
# introduce and hard to notice (different pseudocounts, different treatment
# of ambiguous bases, off-by-one in the sliding window, ...).
_TNF_BASES = "ACGT"
_TNF_COMP = str.maketrans("ACGT", "TGCA")


def _revcomp(kmer: str) -> str:
    return kmer.translate(_TNF_COMP)[::-1]


def _build_canonical_tetramers() -> List[str]:
    """The 136 canonical tetranucleotides: all 256 4-mers collapsed by
    reverse-complement symmetry (kmer and revcomp(kmer) are the same
    feature — DNA is double-stranded), keeping whichever of the pair sorts
    first. This is the standard TNF feature space used in metagenomic
    binning tools (MetaBAT, CONCOCT, etc.)."""
    seen, canon = set(), set()
    for a in _TNF_BASES:
        for b in _TNF_BASES:
            for c in _TNF_BASES:
                for d in _TNF_BASES:
                    kmer = a + b + c + d
                    if kmer in seen:
                        continue
                    rc = _revcomp(kmer)
                    seen.add(kmer)
                    seen.add(rc)
                    canon.add(min(kmer, rc))
    return sorted(canon)


_CANONICAL_TETRAMERS = _build_canonical_tetramers()          # 136 entries
_TETRAMER_INDEX = {k: i for i, k in enumerate(_CANONICAL_TETRAMERS)}

# ---- vectorized TNF machinery -------------------------------------------
# PERFORMANCE FIX (this session): the original _compute_tnf_vector below
# counted tetramers with a plain Python `for i in range(len(seq)-3)` loop —
# one Python-level iteration per BASE, per contig. At real run scale
# (~190,000 contigs averaging ~1.5kb in this project's actual runs) that is
# on the order of 280 MILLION pure-Python loop iterations for this one QC
# signal alone — the single biggest identified bottleneck in step 11.
# Replaced with a numpy-vectorized version that is mathematically IDENTICAL
# (same 136-dim canonical tetramer definition, same "skip any window
# containing a non-ACGT base" rule) but does the counting with array ops
# instead of a Python loop — typically two to three orders of magnitude
# faster for contigs in the hundreds-to-thousands of bp range. Verified
# byte-for-byte equivalent against the original loop implementation
# (kept below, renamed _compute_tnf_vector_slow_reference) on random ACGT
# and mixed-ambiguity sequences before switching the default over.
_BASE2CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
_BYTE2CODE = np.full(256, -1, dtype=np.int16)
for _tnf_base_char, _tnf_base_val in _BASE2CODE.items():
    _BYTE2CODE[ord(_tnf_base_char)] = _tnf_base_val

# _RAW_TETRAMER_TO_CANON[raw_base4_index] -> canonical (0-135) index, for
# all 256 possible ACGT-only 4-mers, built once at import time using the
# exact same "canonical = kmer if kmer in _TETRAMER_INDEX else revcomp(kmer)"
# rule the original per-position loop used.
_RAW_TETRAMER_TO_CANON = np.zeros(256, dtype=np.int64)
# NOTE: these loop variables are prefixed/named to avoid colliding with any
# other module-level name in this file — a previous version of this loop
# used `_c` as the third base's loop variable, which is a bare module-level
# `for` loop (no function scope), so it silently OVERWROTE the pre-existing
# `_c()` ANSI-color helper function (defined earlier, used by
# print_workflow_banner and the whole live terminal UI) with the string
# "T" (its last iteration value) for the rest of the process — crashing
# every run at the very first `print_workflow_banner()` call with
# "TypeError: 'str' object is not callable". Fixed by using names that
# cannot collide with anything else at module scope.
for _tnf_b0 in _TNF_BASES:
    for _tnf_b1 in _TNF_BASES:
        for _tnf_b2 in _TNF_BASES:
            for _tnf_b3 in _TNF_BASES:
                _tnf_kmer = _tnf_b0 + _tnf_b1 + _tnf_b2 + _tnf_b3
                _tnf_raw_idx = (_BASE2CODE[_tnf_b0] * 64 + _BASE2CODE[_tnf_b1] * 16 +
                                _BASE2CODE[_tnf_b2] * 4 + _BASE2CODE[_tnf_b3])
                _tnf_canon_idx = _TETRAMER_INDEX.get(_tnf_kmer)
                if _tnf_canon_idx is None:
                    _tnf_canon_idx = _TETRAMER_INDEX[_revcomp(_tnf_kmer)]
                _RAW_TETRAMER_TO_CANON[_tnf_raw_idx] = _tnf_canon_idx


def _compute_tnf_vector(seq: str) -> np.ndarray:
    """Real 136-dimensional canonical tetranucleotide-frequency vector,
    computed with numpy array operations instead of a per-base Python loop
    (see the PERFORMANCE FIX note above _BASE2CODE). Same definition as the
    original: every 4-base sliding window, skip any window touching a
    non-ACGT character, fold into the 136 canonical (reverse-complement-
    collapsed) tetramer bins, normalize to frequencies."""
    n = len(seq)
    n_dims = len(_CANONICAL_TETRAMERS)
    if n < 4:
        return np.zeros(n_dims)
    codes = _BYTE2CODE[np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)]
    valid = codes >= 0
    # A 4-window starting at i is usable only if all 4 bases in it are ACGT.
    valid_window = valid[:-3] & valid[1:-2] & valid[2:-1] & valid[3:]
    if not valid_window.any():
        return np.zeros(n_dims)
    # Base-4 encode each window from its (possibly invalid) codes; garbage
    # values at invalid positions never leak through because valid_window
    # masks them out on the very next line before they're ever used as an
    # index.
    c0 = np.clip(codes[:-3], 0, 3).astype(np.int64)
    c1 = np.clip(codes[1:-2], 0, 3).astype(np.int64)
    c2 = np.clip(codes[2:-1], 0, 3).astype(np.int64)
    c3 = np.clip(codes[3:], 0, 3).astype(np.int64)
    raw_idx = c0 * 64 + c1 * 16 + c2 * 4 + c3
    canon_idx = _RAW_TETRAMER_TO_CANON[raw_idx[valid_window]]
    counts = np.bincount(canon_idx, minlength=n_dims).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def _compute_tnf_vector_slow_reference(seq: str) -> np.ndarray:
    """ORIGINAL, unvectorized implementation — kept only as a correctness
    reference / for ad-hoc verification against `_compute_tnf_vector`
    (they must always agree; see the PERFORMANCE FIX note above). Not
    called anywhere in the pipeline itself."""
    seq = seq.upper()
    counts = np.zeros(len(_CANONICAL_TETRAMERS))
    for i in range(len(seq) - 3):
        k = seq[i:i + 4]
        if set(k) <= set("ACGT"):
            idx = _TETRAMER_INDEX.get(k) if k in _TETRAMER_INDEX else _TETRAMER_INDEX.get(_revcomp(k))
            if idx is not None:
                counts[idx] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def step11_multisignal_scoring(fasta, coverage_tsv, classification_tsv, depth_cols,
                                no_rescue_reference_ids, low_breadth_ids, rescued_ids,
                                outdir, ckpt, config) -> Dict:
    """Replaces the old flat 'mito if coverage >5x median' rule. Combines:
      - classification confidence (step 6/7)
      - coverage depth/variance across samples (step 9)
      - TNF (composition) consistency vs. the assembly-wide average
      - GC / length outlier status
      - step 10's 'rescue wanted but no reference configured' flag
      - step 10's 'long contig, high depth, very low breadth' flag
      - step 10's 'retained via reference rescue' note
    into one label per contig: retain / retain_with_warning / uncertain /
    remove_high_confidence_contaminant. A contig is not removed just for one
    weak signal — it takes convergent evidence.

    `depth_cols` comes from step 9's explicit column split (never re-inferred
    here by excluding known column names). `no_rescue_reference_ids` is the
    set of single-sample gray-zone contigs step 10 retained without being
    able to actually run its rescue check (no reference FASTA configured);
    `low_breadth_ids` is the set of long (>=mid_ceiling) contigs step 10 kept
    despite very low breadth (the "one covered base, high mean from a local
    repeat" pattern) — both are folded in here as WARNINGS (real, if weak,
    evidence something's off) rather than tracked as statuses only step 10
    knows about, so either can combine with other weak signals into
    'uncertain'. `rescued_ids` is different in kind: a reference hit is
    supporting evidence a contig is real, not contamination evidence, so it
    is recorded as an informational NOTE in the master table/scoring_reasons
    but never counted toward the warning-count threshold that drives
    'uncertain'."""
    STEP = "step11_scoring"
    tnf_enabled = bool(config.get("enable_tnf_qc_signal", True))
    tnf_fn = config.get("tnf_vector_fn") or _compute_tnf_vector
    # Function identity (not the function itself, which isn't hashable into
    # a fingerprint string) — so swapping config['tnf_vector_fn'] between
    # this file's fallback and the encoder module's real implementation
    # invalidates the checkpoint instead of silently reusing a cached run
    # scored under a different TNF definition.
    tnf_fn_id = getattr(tnf_fn, "__module__", "") + "." + getattr(tnf_fn, "__qualname__", str(tnf_fn))
    _rbs_fp = config.get('rule_based_high_conf_sources')
    fp = _fingerprint(_file_fingerprint(fasta), _file_fingerprint(coverage_tsv),
                       _file_fingerprint(classification_tsv), config.get("scoring_cov_cv_high"),
                       config.get("scoring_gc_zscore_flag"), config.get("scoring_tnf_dist_flag"),
                       config.get("classification_confidence_high"),
                       sorted(_rbs_fp) if _rbs_fp else sorted(_RULE_BASED_HIGH_CONF_SOURCES),
                       sorted(no_rescue_reference_ids), sorted(low_breadth_ids), sorted(rescued_ids),
                       tnf_enabled, tnf_fn_id, LOGIC_VERSION, sorted(_domain_policy(config).items()))
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(11)

    out_dir = Path(outdir) / "11_scoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / "scored.fasta")

    cov_df = pd.read_csv(coverage_tsv, sep='\t', index_col=0)
    class_df = pd.read_csv(classification_tsv, sep='\t').set_index('contig_id')

    cov_cv_flag = float(config.get("scoring_cov_cv_high", 1.5))
    gc_z_flag = float(config.get("scoring_gc_zscore_flag", 3.0))
    tnf_dist_flag = float(config.get("scoring_tnf_dist_flag", 2.5))

    # PERFORMANCE FIX (this session): the per-contig loop below used to call
    # `class_df.loc[cid]` and `cov_df.loc[cid, depth_cols]` once EACH per
    # contig — every `.loc` call allocates a new pandas object, real
    # overhead at ~190,000-contig scale. Both are now vectorized/converted
    # to plain dicts ONCE, up front, and the loop does O(1) dict lookups
    # instead.
    class_lookup: Dict[str, Dict] = class_df[
        ['classification', 'confidence', 'confidence_type', 'source']].to_dict('index')
    high_conf = float(config.get("classification_confidence_high", 0.80))
    _rbs = config.get('rule_based_high_conf_sources')
    rule_based_sources = set(_rbs) if _rbs else _RULE_BASED_HIGH_CONF_SOURCES
    _cov_vals_df = cov_df[depth_cols].astype(float)
    _cov_mean_s = _cov_vals_df.mean(axis=1)
    _cov_std_s = _cov_vals_df.std(axis=1)
    _cov_cv_s = (_cov_std_s / _cov_mean_s).where(_cov_mean_s > 0, 0.0)
    cov_stats_lookup: Dict[str, Dict[str, float]] = pd.DataFrame(
        {'mean_cov': _cov_mean_s, 'cov_cv': _cov_cv_s}).to_dict('index')

    records = list(SeqIO.parse(fasta, 'fasta'))
    gc_values = np.array([
        (str(r.seq).upper().count('G') + str(r.seq).upper().count('C')) / max(len(r.seq), 1)
        for r in records
    ])
    gc_mean, gc_std = gc_values.mean(), gc_values.std() or 1.0

    # Assembly-wide TNF baseline: compute each contig's 136-dim
    # canonical-tetramer vector once, then measure how far each contig sits
    # from the assembly mean (in units of the mean per-contig deviation).
    # A contig whose base composition looks nothing like the rest of the
    # assembly is a mild contamination signal, same logic as GC skew but
    # slightly harder to fake. THIS IS A QC-ONLY SIGNAL (see
    # config['enable_tnf_qc_signal'] / config['tnf_vector_fn'] in
    # DEFAULT_CONFIG) — it is not, and must not become, the pipeline's
    # canonical TNF feature; that belongs to the dedicated TNF module run
    # once on final_clean.fasta.
    if tnf_enabled:
        tnf_source_desc = ("the injected config['tnf_vector_fn']" if config.get('tnf_vector_fn')
                            else "this file's local fallback _compute_tnf_vector")
        log.info(f"  TNF QC signal: ON, using {tnf_source_desc}")
        n_tnf_dims = len(_CANONICAL_TETRAMERS)
        tnf_vectors = [tnf_fn(str(r.seq)) for r in records]
        tnf_matrix = np.vstack(tnf_vectors) if tnf_vectors else np.zeros((0, n_tnf_dims))
        tnf_mean = tnf_matrix.mean(axis=0) if len(tnf_matrix) else np.zeros(n_tnf_dims)
        tnf_dists = np.linalg.norm(tnf_matrix - tnf_mean, axis=1) if len(tnf_matrix) else np.zeros(0)
        tnf_dist_mean, tnf_dist_std = (tnf_dists.mean(), tnf_dists.std() or 1.0) if len(tnf_dists) else (0.0, 1.0)
    else:
        log.info("  TNF QC signal: OFF (config['enable_tnf_qc_signal']=False) — "
                 "step 11 will not compute or flag on TNF composition at all this run")
        tnf_dists = np.zeros(len(records))
        tnf_dist_mean, tnf_dist_std = 0.0, 1.0

    decisions = []
    with open(output, 'w') as out_f:
        for rec, gc, tnf_dist in zip(records, gc_values, tnf_dists):
            cid = rec.id
            evidence = {"warnings": [], "flags_for_removal": [], "notes": []}

            cls_row = class_lookup.get(cid)
            classification = cls_row['classification'] if cls_row is not None else "Unknown"
            confidence = cls_row['confidence'] if cls_row is not None else 0.0
            conf_type = cls_row.get('confidence_type', '') if cls_row is not None else ''
            conf_source = cls_row.get('source', '') if cls_row is not None else ''

            cov_stats_row = cov_stats_lookup.get(cid)
            if cov_stats_row is not None:
                # NOTE: matches the original per-row `.loc` behavior exactly,
                # including passing through NaN for cov_cv when a run has
                # too few samples for pandas' std(ddof=1) to be defined —
                # `NaN > cov_cv_flag` is False either way, same as before.
                mean_cov = cov_stats_row['mean_cov']
                cov_cv = cov_stats_row['cov_cv']
            else:
                mean_cov, cov_cv = 0.0, 0.0

            gc_z = (gc - gc_mean) / gc_std
            tnf_z = (tnf_dist - tnf_dist_mean) / tnf_dist_std if tnf_enabled else None

            if _domain_removal_decision(classification, conf_type, conf_source, confidence, len(rec.seq),
                                        _domain_policy(config), high_conf, rule_based_sources):
                # Same policy as step 7 (_is_high_confidence_removal) —
                # shouldn't normally reach here (step 7 already removed
                # high-confidence cases), this catches anything that slipped
                # through. NOTE: previously this compared the raw confidence
                # number against a flat 0.5, ignoring confidence_type/source
                # entirely — that let a 'tiara_only' rule-based Bacterial/
                # Archaeal call (confidence 0.60, NOT in the allow-list) get
                # removed here even though step 7 correctly never removes a
                # single-classifier rule-based guess. Fixed to use the exact
                # same type-aware policy as step 7.
                evidence["flags_for_removal"].append(f"borderline {classification} ({confidence:.2f})")
            if cov_cv > cov_cv_flag:
                evidence["warnings"].append(f"high coverage variance (CV={cov_cv:.2f})")
            if abs(gc_z) > gc_z_flag:
                evidence["warnings"].append(f"GC outlier (z={gc_z:.1f})")
            # TNF is QC-only evidence, and only ever a WARNING — a contig is
            # NEVER removed for TNF divergence alone; it can only ever
            # contribute toward 'uncertain' alongside other independent
            # signals (see the decision logic below, which only removes on
            # `flags_for_removal`, never on `warnings` count by itself for
            # TNF specifically — TNF divergence is exactly the kind of
            # single weak signal this design is built to not act on alone).
            if tnf_enabled and abs(tnf_z) > tnf_dist_flag:
                evidence["warnings"].append(f"TNF composition outlier (z={tnf_z:.1f})")
            if classification == "Unknown":
                evidence["warnings"].append("unclassified by Tiara/Whokaryote")
            _pol = _domain_policy(config)
            if _pol["remove_prokaryotes"] and classification in _PROKARYOTE_CATEGORIES:
                evidence["warnings"].append("prokaryote call retained after step 7 ("
                                             + ("below prokaryote_min_length" if conf_source == "whokaryote+tiara"
                                                else "no Tiara+Whokaryote agreement") + ")")
            if (not _pol["remove_organelles"]) and classification in _ORGANELLE_CATEGORIES:
                evidence["warnings"].append(f"{classification.lower()} call retained "
                                             "(organelle_removal_enabled=False)")
            if cid in no_rescue_reference_ids:
                evidence["warnings"].append("single-sample gray-zone contig, no rescue "
                                             "reference was configured at step 10")
            if cid in low_breadth_ids:
                evidence["warnings"].append("long contig retained at step 10 despite very low "
                                             "breadth (high mean depth from a local repeat is "
                                             "a plausible explanation)")
            if cid in rescued_ids:
                # Supporting evidence, not contamination evidence — recorded
                # as a note so it's visible in scoring_reasons/the master
                # table, but it never counts toward the warning-count that
                # drives 'uncertain' below.
                evidence["notes"].append("retained via step 10 reference-rescue alignment, "
                                          "not organically (single-sample gray-zone contig)")

            if evidence["flags_for_removal"]:
                decision = "remove_high_confidence_contaminant"
            elif len(evidence["warnings"]) >= 2:
                decision = "uncertain"
            elif evidence["warnings"]:
                decision = "retain_with_warning"
            else:
                decision = "retain"

            if decision != "remove_high_confidence_contaminant":
                SeqIO.write(rec, out_f, "fasta")

            decisions.append({
                "contig_id": cid, "length": len(rec.seq), "classification": classification,
                "confidence": confidence, "mean_cov": round(mean_cov, 2),
                "cov_cv": round(cov_cv, 2), "gc_zscore": round(gc_z, 2),
                "tnf_zscore": round(tnf_z, 2) if tnf_z is not None else "n/a (tnf_qc_signal disabled)",
                "decision": decision,
                "reasons": "; ".join(evidence["warnings"] + evidence["flags_for_removal"] + evidence["notes"])
            })

    decisions_df = pd.DataFrame(decisions)
    decisions_tsv = out_dir / "contig_decisions.tsv"
    decisions_df.to_csv(decisions_tsv, sep='\t', index=False)

    summary = decisions_df['decision'].value_counts().to_dict()
    log.info(f"Scoring summary: {summary}")

    meta = {"output": output, "decisions_tsv": str(decisions_tsv), "summary": summary,
            "tnf_qc_signal_enabled": tnf_enabled, "tnf_vector_source": tnf_fn_id, "_fp": fp}
    ckpt.mark_done(STEP, meta)
    live_step_done(11, t0, str(summary))
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 12 — SUBSET COVERAGE TABLE (no remapping)
# ═══════════════════════════════════════════════════════════════════════════════

def step12_subset_coverage(fasta, coverage_tsv, outdir, ckpt, depth_cols=None, sample_names=None) -> Dict:
    STEP = "step12_subset_coverage"
    fp = _fingerprint(_file_fingerprint(fasta), _file_fingerprint(coverage_tsv), depth_cols, sample_names)
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(12)

    out_dir = Path(outdir) / "12_final_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / "coverage_table.tsv")

    # retained_order is a LIST (from FASTA iteration order) — the actual order
    # source. retained_set is only ever used for O(1) membership tests. The
    # old code built "ordered" by filtering a set, which does NOT preserve
    # FASTA order despite what its comment claimed — sets have no order.
    retained_order = [rec.id for rec in SeqIO.parse(fasta, 'fasta')]
    retained_set = set(retained_order)
    cov_df = pd.read_csv(coverage_tsv, sep='\t', index_col=0)
    cov_ids = set(cov_df.index)

    missing_from_coverage = sorted(retained_set - cov_ids)
    missing_from_fasta = sorted(cov_ids - retained_set)  # expected: everything filtered out

    if missing_from_coverage:
        log.warning(f"⚠️  {len(missing_from_coverage)} retained FASTA ID(s) have NO coverage "
                     f"row — writing to id_reconciliation_warnings.txt")
        with open(out_dir / "id_reconciliation_warnings.txt", 'w') as f:
            f.write("\n".join(missing_from_coverage) + "\n")

    # The other direction: coverage rows for IDs no longer in the retained
    # FASTA. Normally this is just "everything filtered out upstream" and
    # harmless — but previously it was computed and silently dropped, so
    # there was no way to tell "expected, everything filtered" apart from
    # "CoverM's coverage table has IDs that were never valid contigs at all"
    # without re-deriving it by hand. Written for the same auditability
    # reason as the other direction, not logged as a warning by default
    # since a large count here is the EXPECTED case.
    coverage_ids_file = out_dir / "coverage_ids_not_in_final_fasta.txt"
    with open(coverage_ids_file, 'w') as f:
        f.write("\n".join(missing_from_fasta) + ("\n" if missing_from_fasta else ""))

    ordered = [i for i in retained_order if i in cov_ids]
    subset_df = cov_df.loc[ordered]

    # NEW: coverage.py (Phase 2) expects EXACTLY one plain column per
    # sample (matched against expected_samples), not CoverM's raw
    # "{sample}_Mean" / "{sample}_Covered_Fraction" column pairs. Keep only
    # the depth (mean) columns here and rename them to bare sample names —
    # breadth/covered_fraction columns are dropped from this specific
    # output since coverage.py has no use for them; the full CoverM table
    # (both metrics) is still preserved untouched upstream at
    # step9's coverage_tsv, so nothing is actually lost, just not
    # re-exposed in this consumer-facing subset.
    if depth_cols and sample_names and len(depth_cols) == len(sample_names):
        missing_depth_cols = [c for c in depth_cols if c not in subset_df.columns]
        if missing_depth_cols:
            log.error(f"step12: expected depth column(s) {missing_depth_cols} not found in "
                      f"coverage table (columns present: {list(subset_df.columns)}) — cannot "
                      f"safely rename to sample names. Check step 9's depth_cols against this "
                      f"table's actual header.")
            sys.exit(1)
        rename_map = dict(zip(depth_cols, sample_names))
        subset_df = subset_df[depth_cols].rename(columns=rename_map)
        log.info(f"  Renamed depth columns to sample names: {rename_map}")
    else:
        log.warning("⚠️  step12: depth_cols/sample_names not provided or length mismatch — "
                    "writing coverage_table.tsv with CoverM's raw column names unchanged. "
                    "Downstream coverage.py's expected_samples check will likely fail unless "
                    "it's called without expected_samples or with CoverM's raw names.")

    subset_df.to_csv(output, sep='\t')

    with open(out_dir / "retained_ids.txt", 'w') as f:
        f.write("\n".join(retained_order) + "\n")

    meta = {"coverage_table": output, "n_retained": len(retained_set),
            "n_missing_coverage_rows": len(missing_from_coverage),
            "n_missing_from_fasta_rows": len(missing_from_fasta),
            "coverage_ids_not_in_final_fasta_file": str(coverage_ids_file), "_fp": fp}
    ckpt.mark_done(STEP, meta)
    live_step_done(12, t0, f"{len(retained_set):,} contigs, "
                          f"{len(missing_from_coverage)} reconciliation mismatch(es)")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 13 — QC REPORTS
# ═══════════════════════════════════════════════════════════════════════════════

def _read_id_list(path) -> set:
    if not path or not Path(path).exists():
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def _build_master_decision_table(all_step_meta, final_retained_ids: set) -> pd.DataFrame:
    """One row per ORIGINAL contig that ever existed (keyed off id_map.tsv,
    written at step 2 before anything is removed), tracing it through every
    step that could have removed it. This is what makes the pipeline's
    'every contig is traceable' claim actually true instead of aspirational
    — before this, that information was scattered across id_map.tsv,
    contig_classification.tsv, contig_decisions.tsv, and several
    removed_*_ids.txt files with no single join."""
    id_map = pd.read_csv(all_step_meta['step2']['id_map'], sep='\t', dtype=str)

    redundant_ids = _read_id_list(all_step_meta['step4'].get('redundant_ids_file'))
    short_removed_ids = _read_id_list(all_step_meta['step5'].get('removed_ids_file'))
    masked_ids = _read_id_list(all_step_meta['step8'].get('masked_ids_file'))

    domain_removed: Dict[str, str] = {}
    removed_ids_dir = all_step_meta['step7'].get('removed_ids_dir')
    if removed_ids_dir:
        for removed_file in Path(removed_ids_dir).glob("removed_*_ids.txt"):
            # filename pattern: removed_<category>_ids.txt
            category = removed_file.stem[len("removed_"):-len("_ids")]
            for cid in _read_id_list(removed_file):
                domain_removed[cid] = category

    step10_removed: Dict[str, str] = {}
    step10_removed_file = all_step_meta['step10'].get('removed_ids_file')
    if step10_removed_file and Path(step10_removed_file).exists():
        df10 = pd.read_csv(step10_removed_file, sep='\t')
        step10_removed = dict(zip(df10['contig_id'], df10['reason']))

    class_df = pd.read_csv(all_step_meta['step6']['classification_tsv'], sep='\t').set_index('contig_id')
    decisions_df = pd.read_csv(all_step_meta['step11']['decisions_tsv'], sep='\t').set_index('contig_id')

    rows = []
    for _, r in id_map.iterrows():
        safe_id = r['safe_internal_id']
        row = {
            "original_id": r['original_id'], "safe_id": safe_id,
            "source_sample": r['source_sample'], "source_file": r.get('source_file', ''),
            "length": r['length'], "duplicate_of": r['duplicate_of'],
        }
        if safe_id in redundant_ids:
            row.update(removal_step="step4_dedup", removal_reason="redundant (ANI dedup)",
                       final_retained=False)
            rows.append(row)
            continue
        if safe_id in short_removed_ids:
            row.update(removal_step="step5_length_prefilter",
                       removal_reason="below min_contig_length", final_retained=False)
            rows.append(row)
            continue

        if safe_id in class_df.index:
            crow = class_df.loc[safe_id]
            row.update(tiara_label=crow.get('tiara_label', ''),
                       whokaryote_label=crow.get('whokaryote_label', ''),
                       whokaryote_status=crow.get('whokaryote_status', ''),
                       classification=crow.get('classification', ''),
                       classification_confidence=crow.get('confidence', ''),
                       classification_confidence_type=crow.get('confidence_type', ''),
                       classification_conflict=crow.get('conflict', ''))

        row["rrna_status"] = "masked" if safe_id in masked_ids else "no_rrna_detected"

        if safe_id in domain_removed:
            row.update(removal_step="step7_domain_removal",
                       removal_reason=f"high-confidence {domain_removed[safe_id]}",
                       final_retained=False)
            rows.append(row)
            continue
        if safe_id in step10_removed:
            row.update(removal_step="step10_adaptive_filter",
                       removal_reason=step10_removed[safe_id], final_retained=False)
            rows.append(row)
            continue

        if safe_id in decisions_df.index:
            drow = decisions_df.loc[safe_id]
            # scoring_reasons is pulled here UNCONDITIONALLY — previously it
            # was only recorded when step 11 removed the contig, so a
            # retain_with_warning/uncertain contig's actual warnings (GC
            # outlier, TNF outlier, no-rescue-reference flag, ...) were
            # silently dropped from the master table even though they're
            # real evidence a downstream reader would want to see.
            row.update(mean_cov=drow.get('mean_cov', ''), cov_cv=drow.get('cov_cv', ''),
                       gc_zscore=drow.get('gc_zscore', ''), tnf_zscore=drow.get('tnf_zscore', ''),
                       scoring_decision=drow.get('decision', ''),
                       scoring_reasons=drow.get('reasons', ''))
            if drow.get('decision') == 'remove_high_confidence_contaminant':
                row.update(removal_step="step11_scoring",
                           removal_reason=drow.get('reasons', ''), final_retained=False)
                rows.append(row)
                continue

        row.update(removal_step="", removal_reason="",
                    final_retained=(safe_id in final_retained_ids))
        rows.append(row)

    return pd.DataFrame(rows)


def step13_qc_reports(id_map_tsv, classification_tsv, decisions_tsv, final_fasta,
                       unmasked_fasta, coverage_table, all_step_meta, outdir, ckpt) -> Dict:
    STEP = "step13_qc_reports"
    fp = _fingerprint(_file_fingerprint(id_map_tsv), _file_fingerprint(classification_tsv),
                       _file_fingerprint(decisions_tsv), _file_fingerprint(final_fasta),
                       _file_fingerprint(coverage_table))
    cached = _checkpoint_ok(ckpt, STEP, fp)
    if cached is not None:
        return cached
    t0 = time.time()
    live_step_start(13)

    final_dir = Path(outdir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_clean = final_dir / "final_clean.fasta"
    final_clean_unmasked = final_dir / "final_clean_unmasked.fasta"
    shutil.copy(final_fasta, final_clean)
    # Unmasked variant: same retained IDs, pulled from the unmasked FASTA
    retained_ids = {rec.id for rec in SeqIO.parse(str(final_clean), 'fasta')}
    with open(final_clean_unmasked, 'w') as out_f:
        for rec in SeqIO.parse(unmasked_fasta, 'fasta'):
            if rec.id in retained_ids:
                SeqIO.write(rec, out_f, 'fasta')
    shutil.copy(coverage_table, final_dir / "coverage_table.tsv")
    shutil.copy(id_map_tsv, final_dir / "id_map.tsv")
    shutil.copy(classification_tsv, final_dir / "contig_classification.tsv")
    shutil.copy(decisions_tsv, final_dir / "contig_decisions.tsv")

    for _old in final_dir.glob("removed_*_ids.txt"):
        _old.unlink()
    for removed_file in Path(all_step_meta['step7']['removed_ids_dir']).glob("removed_*_ids.txt"):
        shutil.copy(removed_file, final_dir / removed_file.name)

    # Master decision table: one row per ORIGINAL contig, joined across every
    # step that could have removed it, with a removal_step/removal_reason
    # that's actually populated instead of implied by which of several files
    # a contig's ID happens to be missing from.
    master_df = _build_master_decision_table(all_step_meta, retained_ids)
    master_tsv = final_dir / "master_decision_table.tsv"
    master_df.to_csv(master_tsv, sep='\t', index=False)

    n_initial = all_step_meta['step2']['n_seqs']
    n_final = count_seqs(str(final_clean))
    qc_summary = {
        "initial_contigs": n_initial,
        "assembly_n50": all_step_meta['step3']['n50'],
        "assembly_n50_quality": all_step_meta['step3'].get('n50_quality', 'unknown'),
        "after_length_prefilter": all_step_meta['step5']['after'],
        "after_dedup": all_step_meta['step4']['after'],
        "after_domain_removal": all_step_meta['step7']['kept'],
        "rrna_masking_status": all_step_meta['step8'].get('rrna_status', 'unknown'),
        "after_adaptive_filter": all_step_meta['step10']['total_kept'],
        "final_retained": n_final,
        "retention_pct": round(100 * n_final / n_initial, 1) if n_initial else 0,
        "removed_by_category": all_step_meta['step7']['removed_by_category'],
        "rescued_single_sample": all_step_meta['step10']['stats']['mid_rescued'],
        "rescue_attempted_no_hit_removed": all_step_meta['step10']['stats']['mid_rescue_failed'],
        "scoring_summary": all_step_meta['step11']['summary'],
        "tnf_qc_signal_enabled": all_step_meta['step11'].get('tnf_qc_signal_enabled'),
        "classification_tools_provenance": all_step_meta['step6'].get('tools_provenance'),
        "tnf_vector_source": all_step_meta['step11'].get('tnf_vector_source'),
        "coverage_reconciliation_mismatches": all_step_meta['step12']['n_missing_coverage_rows'],
        "coverage_rows_not_in_final_fasta": all_step_meta['step12'].get('n_missing_from_fasta_rows', 0),
        "master_decision_table": str(master_tsv),
        "master_table_removal_step_counts": master_df['removal_step'].replace('', 'retained').value_counts().to_dict(),
    }
    with open(final_dir / "qc_summary.json", 'w') as f:
        json.dump(qc_summary, f, indent=2)

    if qc_summary["rrna_masking_status"] == "failed":
        log.warning("⚠️  barrnap (step 8, rRNA masking) FAILED to run — final_clean.fasta was "
                    "never rRNA-masked. TNF/composition scoring in step 11 ran on unmasked "
                    "sequence, so rRNA-driven composition bias was not removed. See "
                    "qc_summary.json['rrna_masking_status'] and the step8 log for details.")

    log.info("")
    log.info(_c("═" * 60, _C_GREEN))
    log.info(_c(f"  {n_initial:,} initial → {n_final:,} final "
                f"({qc_summary['retention_pct']}% retained)", _C_GREEN))
    log.info(_c("═" * 60, _C_GREEN))

    meta = {"final_dir": str(final_dir), "qc_summary": qc_summary,
            "qc_summary_path": str(final_dir / "qc_summary.json"),
            "master_decision_table": str(master_tsv), "_fp": fp}
    ckpt.mark_done(STEP, meta)
    live_step_done(13, t0, f"{qc_summary['retention_pct']}% retention")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_preprocessing(assembly_input: str, reads_dir: str, samples_tsv: str,
                       outdir: str = "preprocessing_output", threads: int = 80,
                       config: Optional[Dict] = None) -> Dict[str, str]:
    t0_total = time.time()
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Checkpoint() appends its own "checkpoints" subdir internally — passing
    # outdir/"checkpoints" here doubled it to outdir/checkpoints/checkpoints.
    ckpt = Checkpoint(outdir)

    _apply_step_labels(cfg)
    print_workflow_banner(current_step=0)
    print(_c(f"  Output directory: {outdir}", _C_DIM))
    print(_c(f"  Threads: {threads}  |  Read type: {cfg.get('read_type')}", _C_DIM))
    print()

    try:
        samples_df = pd.read_csv(samples_tsv, sep='\t')
    except Exception as e:
        log.error(f"Cannot read samples file {samples_tsv}: {e}")
        sys.exit(1)

    meta: Dict[str, Dict] = {}

    step1_validate_inputs(assembly_input, samples_df, reads_dir, outdir, cfg)
    meta['step2'] = step2_merge_assemblies(assembly_input, outdir, ckpt, cfg)
    meta['step3'] = step3_assembly_stats(meta['step2']['output'], outdir, ckpt)
    # ORDER CHANGED (v2): length pre-filter FIRST, then cross-sample dedup on the
    # filtered set. Previously dedup ran on the unfiltered catalogue, hit the
    # contig cap on large datasets and was silently skipped.
    meta['step5'] = step5_length_prefilter(meta['step2']['output'], outdir,
                                            int(cfg['min_contig_length']), ckpt)
    meta['step4'] = step4_skani_dedup(meta['step5']['output'], outdir, threads, ckpt, cfg,
                                       id_map=meta['step2']['id_map'])
    meta['step6'] = step6_classification(meta['step4']['output'], outdir, threads, ckpt, cfg)
    _rbs = cfg.get('rule_based_high_conf_sources')
    rule_based_sources = set(_rbs) if _rbs else None
    meta['step7'] = step7_domain_removal(meta['step4']['output'], meta['step6']['classification_tsv'],
                                          meta['step6']['high_confidence_threshold'], outdir, ckpt,
                                          rule_based_sources, policy=_domain_policy(cfg))
    meta['step8'] = step8_rdna_masking(meta['step7']['output'], outdir, threads, ckpt,
                                        cfg.get('barrnap_kingdom', 'fun'))
    # Mapping uses the UNMASKED assembly (see step 8/9 docstrings) — coverage
    # over rRNA loci must not be suppressed just because those bases are
    # N-masked in the composition-only copy.
    meta['step9'] = step9_single_mapping(meta['step8']['unmasked_fasta'], samples_df, reads_dir,
                                          outdir, threads, ckpt, cfg)
    meta['step10'] = step10_adaptive_filter(meta['step8']['masked_fasta'], meta['step9']['coverage_tsv'],
                                             meta['step9']['depth_cols'], meta['step9']['breadth_cols'],
                                             outdir, ckpt, cfg, threads=threads,
                                             unmasked_fasta=meta['step8']['unmasked_fasta'])
    no_rescue_reference_ids = _read_id_list(meta['step10'].get('flagged_no_rescue_reference_file'))
    low_breadth_ids = _read_id_list(meta['step10'].get('flagged_low_breadth_file'))
    rescued_ids = _read_id_list(meta['step10'].get('rescued_ids_file'))
    meta['step11'] = step11_multisignal_scoring(meta['step10']['output'], meta['step9']['coverage_tsv'],
                                                 meta['step6']['classification_tsv'],
                                                 meta['step9']['depth_cols'], no_rescue_reference_ids,
                                                 low_breadth_ids, rescued_ids,
                                                 outdir, ckpt, cfg)
    meta['step12'] = step12_subset_coverage(meta['step11']['output'], meta['step9']['coverage_tsv'],
                                                outdir, ckpt,depth_cols=meta['step9']['depth_cols'],
                                                sample_names=samples_df['sample'].tolist())

    if bool(cfg.get("keep_bams", True)):
        _bams = [Path(b) for b in meta['step9']['bam_files']]
        _missing = [str(b) for b in _bams if not b.exists()]
        _size = sum(b.stat().st_size for b in _bams if b.exists())
        log.info(f"Keeping {len(_bams) - len(_missing)} mapping BAM(s) (keep_bams=true, "
                 f"{_size / 1e9:.2f} GB) in {_bams[0].parent if _bams else 'n/a'}")
        if _missing:
            log.warning(f"Expected BAM(s) not found on disk: {_missing}")
    else:
        cleanup_intermediate(meta['step9']['bam_files'] +
                              [f"{b}.bai" for b in meta['step9']['bam_files']],
                              label="step9 mapping BAMs")

    meta['step13'] = step13_qc_reports(
        meta['step2']['id_map'], meta['step6']['classification_tsv'],
        meta['step11']['decisions_tsv'], meta['step11']['output'],
        meta['step8']['unmasked_fasta'], meta['step12']['coverage_table'],
        meta, outdir, ckpt,
    )

    t_total = time.time() - t0_total
    print(_c(f"\n  Total pipeline time: {t_total/60:.1f} minutes ({t_total/3600:.2f}h)\n", _C_BOLD))

    return {
        "final_clean": str(Path(meta['step13']['final_dir']) / "final_clean.fasta"),
        "final_clean_unmasked": str(Path(meta['step13']['final_dir']) / "final_clean_unmasked.fasta"),
        "coverage_table": str(Path(meta['step13']['final_dir']) / "coverage_table.tsv"),
        "qc_summary": meta['step13']['qc_summary_path'],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════



# PROFILE HOOKS v1 — observation only; scientific functions above are unchanged.
from hyphaesbin.utils.resource_profiler import profile_functions, profile_checkpoint
profile_functions(globals(), ['run_preprocessing', 'step1_validate_inputs', 'step2_merge_assemblies', 'step3_assembly_stats', 'step4_skani_dedup', 'step5_length_prefilter', 'step6_classification', 'step7_domain_removal', 'step8_rdna_masking', 'step9_single_mapping', 'step10_adaptive_filter', 'step11_multisignal_scoring', 'step12_subset_coverage', 'step13_qc_reports', '_map_one_sample', 'run_cmd'])
_checkpoint_ok = profile_checkpoint(_checkpoint_ok)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HyphaeSBin Preprocessing (redesigned, 13 steps)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--assembly', required=True, help='Assembly FASTA or directory of per-sample FASTAs')
    parser.add_argument('--reads-dir', required=True, help='Directory containing FASTQ files')
    parser.add_argument('--samples', required=True, help='Sample TSV with columns: sample, r1; optional r2')
    parser.add_argument('--outdir', default='preprocessing_output', help='Output directory')
    parser.add_argument('--threads', type=int, default=80)
    parser.add_argument('--min-contig-length', type=int, default=DEFAULT_CONFIG['min_contig_length'])
    parser.add_argument('--read-type', default='auto', choices=['auto', 'short', 'hifi', 'ont', 'hybrid'])
    parser.add_argument('--rescue-reference', default='', help='Fungal reference FASTA for step 10 rescue branch')
    parser.add_argument('--no-rescue', action='store_true', help='Disable the single-sample rescue branch')
    parser.add_argument('--auto-install-tools', action='store_true',
                         help='Let the pipeline auto-install missing required tools via conda '
                              '(default: report missing tools in step1 and stop)')
    parser.add_argument('--max-mapping-workers', default='auto',
                         help='How many samples to map concurrently in step 9 ("auto", or an '
                              'int; 1 = old fully-sequential behavior). See DEFAULT_CONFIG '
                              'for the auto-sizing rule and memory caveat.')
    parser.add_argument('--quiet', action='store_true', help='Suppress the live banner (log-only output)')
    args = parser.parse_args()

    if args.quiet:
        _USE_COLOR = False  # noqa: F811 — intentional override for --quiet

    cli_config = {
        "read_type": args.read_type,
        "min_contig_length": args.min_contig_length,
        "rescue_enabled": not args.no_rescue,
        "rescue_reference_fasta": args.rescue_reference,
        "auto_install_tools": args.auto_install_tools,
        "max_mapping_workers": args.max_mapping_workers,
    }

    outputs = run_preprocessing(
        assembly_input=args.assembly,
        reads_dir=args.reads_dir,
        samples_tsv=args.samples,
        outdir=args.outdir,
        threads=args.threads,
        config=cli_config,
    )

    print("\n✅ Done!")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
