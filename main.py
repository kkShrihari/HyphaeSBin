"""
hyphaesbin Main — Complete Pipeline v7 (restructured to match the finalized
preprocessing / coverage / TNF / TE / encoder / clustering modules)
=============================================================================
Order (per the requested restructure — NOTE this SWAPS TNF and TE relative
to the old v6 file's Phase3=TE/Phase4=TNF ordering):

  Phase 1: Preprocessing     — assembly QC, read mapping, coverage table
  Phase 2: Coverage Features — percentile-capped, median-positive-scaled,
                                log1p coverage + k-NN distances (N+4D)
  Phase 3: TNF Composition   — whole-contig k4mer frequency (masked FASTA)
  Phase 4: TE Composition    — MMseqs2/RepeatMasker -> V22_5D (unmasked FASTA)
  Phase 5: Encoder           — beta-VAE fusion (TNF + TE + COV) -> latent
  Phase 6: Clustering        — HDBSCAN -> bins -> skani dedup -> EukCC

=============================================================================
CHANGES FROM v6 (every one verified against the ACTUAL current module code
in this delivery, not assumed from the old file or from any docstring
claim — see the individual module files for the source of truth):

  1. run_module2_lite -> run_preprocessing.  The old file called a function
     (`run_module2_lite`) that does not exist anywhere in the current
     preprocessing.py; the real entry point is `run_preprocessing(
     assembly_input, reads_dir, samples_tsv, outdir, threads, config)` — no
     `resume=`/`mito_multiplier=` kwargs (both gone; preprocessing.py always
     resumes internally via its own 13-step fingerprinted checkpoints, and
     mito_coverage_multiplier is not a key anywhere in its DEFAULT_CONFIG
     any more). The `--mito-multiplier` CLI flag is removed accordingly.

  2. TNF and TE swapped (Phase 3 / Phase 4) per this restructure's explicit
     instruction. They have no data dependency on each other (both read
     directly from preprocessing's two FASTA variants), so the swap is
     purely cosmetic/organizational and changes no numeric result.

  3. Masked vs. unmasked FASTA routing is now explicit and taken from
     preprocessing.py step 8's own recorded `module_routing` metadata
     (see its docstring), not assumed:
       - TNF composition   : MASKED   (`final_clean.fasta`)
       - TE  composition   : UNMASKED (`final_clean_unmasked.fasta`)
       - Clustering's `fasta_path` (used ONLY to extract the literal
         sequence written into each output bin FASTA — see clustering.py's
         `write_bin_fastas()` / its own docstring "name-keyed lookup only")
         is UNMASKED, per module_routing's own "final_bin_output": "unmasked"
         entry — a final delivered genome bin should not contain
         barrnap-masked N-runs over its rRNA loci. This was NOT explicitly
         listed in the "pass the correct FASTA variants" instruction (which
         only named TNF/TE); it is resolved here from preprocessing's own
         routing table plus a direct read of what clustering.py actually
         does with that argument. Coverage's `assembly_fasta` argument is
         used only for contig-ID/order reconciliation (never sequence
         content), so masked-vs-unmasked is immaterial there; the masked
         copy is passed for consistency with the rest of the "masked by
         default" convention.

  4. "Pass n_samples to coverage": `run_coverage_features()` has no literal
     `n_samples` integer parameter — n_samples is derived internally from
     the coverage TSV's own column count. The closest real hook is
     `expected_samples` (an explicit list of sample names cross-checked
     against the TSV's columns, raising loudly on any mismatch instead of
     silently trusting the TSV) — main.py builds that list from the parsed
     `--reads` and passes it. The authoritative sample COUNT that flows
     onward to the encoder is then read back from coverage's own
     manifest.json ("n_samples" key — see coverage.py ~line 915), which
     works correctly even under `--skip-to-phase` where no `--reads` were
     given this run.

  5. "Pass n_samples ... to encoder": `run_encoder()` DOES take a literal
     `n_samples: Optional[int]` kwarg (cross-checked against the raw COV
     array's column count before splitting) — populated from coverage's
     manifest.json as described above.

  6. "Pass generated feature paths, weight paths, and manifests to
     encoder": encoder.py's `_load_modality_manifest()` already
     auto-discovers each modality's manifest as a SIBLING file next to the
     feature path it's given (TNF: contig_ids.json + tnf_feature_schema.json;
     TE: contig_ids.json + schema.json; COV: contig_ids.txt + manifest.json)
     — there is no separate manifest-path argument to pass. main.py only
     needs to pass the three feature paths + two weight paths; the manifests
     are found automatically because every phase here writes its outputs
     into one directory per modality (unchanged from v6's layout).

  7. "Pass final_latent.npy, encoder_manifest.json, canonical IDs, FASTA,
     and coverage mask to clustering": encoder_manifest_path and
     contig_ids_path are passed explicitly below (both also have working
     defaults inside run_clustering() if omitted, since they sit alongside
     final_latent_path / cov_features_path respectively — passed explicitly
     here anyway for clarity and so `--skip-to-phase 6` still works with no
     ambiguity). There is no separate "coverage mask" file: clustering.py
     reads it as column 0 of the SAME `cov_features.npy` already being
     passed via `cov_features_path` (see clustering.py's own
     `n_cov_meta_cols`/`allow_missing_valid_mask` docs, and its
     `filter_contigs_for_clustering()`) — nothing extra to wire up.

  8. Phase 6's old file assumed a `clustering_output/checkpoints/
     final_labels.npy` file existed to recover bin counts/labels. No such
     file exists anywhere in the current clustering.py — `run_clustering()`
     returns only the path to `cluster_summary.tsv`, and the real,
     resumable, per-run artifacts are `clusters_non_deduplicated/`,
     `clusters_deduplicated/`, `cluster_summary.tsv`, `cluster_assignments.tsv`,
     and `clustering_stats.json`. The completion summary below reports
     those real paths instead of a labels array that was never written.

  9. CheckM2 is gone from this file entirely (removed from clustering.py in
     an earlier pass, per explicit prior request) — no `run_checkm2`/
     `checkm2_db`/`checkm2_conda_env` references remain anywhere below.

 10. `ensure_python_deps()` / `ensure_external_tools()` (which used to run
     at import time and could silently `conda install`/`pip install`) are
     gone — deps.py no longer has an installer at all (see deps.py's own
     changelog). main.py instead calls `deps.check_all()` once at startup
     purely to LOG a diagnostic report (required-vs-optional packages,
     external tool availability, compute backends) — it never blocks or
     installs anything; each stage's own module still fails loudly on its
     own if something it actually needs turns out to be missing.

 11. Every one of the six phases is now gated by main.py's OWN top-level
     checkpoint (`ckpt.is_done("phaseN_...")`), not just phases 2/4/6 as in
     the old file — this is what "avoid duplicate execution when a valid
     checkpoint exists" means in practice: a plain rerun of `main.py` with
     no flags changed skips every already-completed phase entirely instead
     of re-invoking (even a fast, self-resuming) call into that module.
     `checkpoint.py`'s new output-file verification means a checkpoint that
     LOOKS done but whose output files were deleted/moved is correctly
     treated as a cache miss instead of a false "reused".

 12. `--reset` clears only main.py's OWN top-level checkpoint store
     (`<outdir>/checkpoints/`) exactly as in v6. It does NOT reach into each
     phase's own internal, independently-fingerprinted checkpoint directory
     (e.g. `<outdir>/preprocessing/checkpoints/`, `<outdir>/encoder/
     checkpoints/tnf/`, etc.) — those are separate `Checkpoint` instances
     each module owns. This is disclosed via a warning at startup when
     `--reset` is used, since it's an easy thing to assume `--reset` fixes
     and then be surprised when a phase "reruns" but instantly reloads its
     own cached sub-steps. To force a truly from-scratch rerun of one
     phase, delete that phase's own output subdirectory.

 13. EukCC (clustering phase) can live in its own conda environment via
     `config['eukcc_conda_env']`, invoked directly with `conda run -n
     <name> ...` inside clustering.py. Tiara/Whokaryote (preprocessing
     step 6) run directly on whatever env this process itself is running
     in — no separate env-switching machinery for them.
"""

import sys as _sys
if _sys.version_info < (3, 7):
    _sys.exit("hyphaesbin needs Python >= 3.7 but this is Python %s. Start it with the pipeline environment's python, e.g. "
              "/DATA_LUN/skkumara/Final/Zosteria_marina/envs/classify/bin/python main.py ..." % _sys.version.split()[0])
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np
import pandas as pd

from hyphaesbin.utils.logger import (
    setup_logger, get_logger, get_log_file,
    log_step_start, log_step_done, log_warning_flag, log_error_flag,
    log_suggestion, log_device_backend, log_threads, log_checkpoint,
    log_contig_counts, log_output_paths,
)
from hyphaesbin.utils.checkpoint import Checkpoint, set_force_trust_fingerprint
from hyphaesbin.utils import deps

# Preprocessing / coverage are load-bearing for every run — no graceful
# degradation is possible without them, so they're imported directly
# (matches v6's own treatment of these two).
from hyphaesbin.preprocessing.preprocessing import run_preprocessing
from hyphaesbin.coverage.coverage import run_coverage_features

# TE / TNF / encoder / clustering — imported defensively so a broken/absent
# optional dependency inside one of them (e.g. no PyTorch) produces one
# clear error message instead of an ImportError traceback from deep inside
# argparse handling.
try:
    from hyphaesbin.composition.composition.TE_composition.te_composition import run_te_branch
    TE_AVAILABLE = True
except ImportError as _e:
    TE_AVAILABLE = False
    _TE_IMPORT_ERROR = _e

try:
    from hyphaesbin.composition.composition.TNF_gene.tnf_gene import run_tnf_wholecontig
    TNF_AVAILABLE = True
except ImportError as _e:
    TNF_AVAILABLE = False
    _TNF_IMPORT_ERROR = _e

try:
    from hyphaesbin.encoder.encoder import run_encoder, EncoderConfig
    ENCODER_AVAILABLE = True
except ImportError as _e:
    ENCODER_AVAILABLE = False
    _ENCODER_IMPORT_ERROR = _e

try:
    from hyphaesbin.clustering.clustering import run_clustering, ClusteringConfig
    CLUSTERING_AVAILABLE = True
except ImportError as _e:
    CLUSTERING_AVAILABLE = False
    _CLUSTERING_IMPORT_ERROR = _e


# =============================================================================
# DEFAULT CONFIG — mirrors config.yaml's real, current defaults (NOT the
# stale v6 values). config.yaml is the source of truth; this dict only
# exists so `main.py` still runs sensibly with no `--config` at all.
# =============================================================================

DEFAULT_CONFIG: Dict = {
    # General
    "threads": 64,
    "seed": 42,
    "min_contig_length": 1000,

    # Preprocessing
    "read_type": "auto",
    "read_type_detect_seed": 0,
    "auto_install_tools": False,
    "dedup_ani": 99.5,
    "dedup_min_af": 95.0,
    "dedup_min_length_ratio": 0.0,
    "dedup_min_length": 2000,
    "dedup_cross_sample_only": True,
    "dedup_skani_args": "-c 30 -m 200 --faster-small -s 95",
    "skip_dedup": False,
    "dedup_max_contigs": 500_000,
    "dedup_allow_skip_over_cap": False,
    "assembly_sample_regex": "",
    "keep_bams": True,
    "tiara_min_len": 1000,
    "tiara_prob_cutoff": [0.65, 0.65],
    "whokaryote_minsize": 1000,
    "whokaryote_model": "T",
    "classification_confidence_high": 0.80,
    "rule_based_high_conf_sources": None,
    "barrnap_kingdom": "fun",   # was "euk" — that value is rejected outright by some
                                # barrnap builds (confirmed live: "[barrnap] ERROR:
                                # Invalid --kingdom 'euk'. Choose from: bac arc fun").
                                # "fun" is also the scientifically correct choice for
                                # this fungi-specific pipeline, not just a fallback.
    "max_mapping_workers": "auto",
    "min_threads_per_sample": 8,
    "adaptive_min_cov": 2.0,
    "adaptive_min_samples": 2,
    "adaptive_min_breadth": 0.3,
    "rescue_enabled": True,
    "rescue_reference_fasta": "",
    "rescue_min_identity": 75.0,
    "rescue_min_query_cov": 0.5,
    "scoring_cov_cv_high": 1.5,
    "scoring_gc_zscore_flag": 3.0,
    "scoring_tnf_dist_flag": 2.5,
    "enable_tnf_qc_signal": True,

    # Checkpointing
    "force_trust_checkpoints": False,  # True = skip every completed step whose
                                        # checkpoint + output files exist,
                                        # regardless of config/input changes since
                                        # it ran (see checkpoint.py's
                                        # set_force_trust_fingerprint() docstring
                                        # for the tradeoff before enabling this).

    # Coverage features
    "coverage_k": 200,
    "coverage_n_cores": None,   # None -> falls back to top-level "threads"
    "coverage_strict_numeric_columns": True,
    "coverage_outlier_cap_percentile": 99.5,
    "coverage_deterministic": False,
    "coverage_brute_force_max_n": 5000,

    # Databases
    "funTEdb": "",

    # TE composition
    "te_mode": "fast",
    "mmseqs_sensitivity": 5.7,
    "mmseqs_min_seq_id": 0.7,
    "mmseqs_coverage": 0.1,
    "mmseqs_max_seq_len": 100000,
    "mmseqs_evalue": 1e-5,
    "te_pa": 36,
    "te_frag": 20_000_000,
    "te_min_sw_score": 225,
    "apply_length_weight_to_features": False,
    "full_confidence_te_bp": 5000,
    "full_confidence_length": 10000,
    "full_confidence_hits": 5,
    "weight_te_coef": 0.55,
    "weight_hit_coef": 0.25,
    "weight_quality_coef": 0.20,
    "repeatmasker_quality_reference_score": 1000.0,
    "unclassified_warn_fraction": 0.95,

    # TNF
    "min_contig_len": 1000,
    "min_valid_kmers": 50,
    "full_confidence_kmers": 5000,
    "weight_mode": "confidence",

    # Encoder
    "tnf_include": True,
    "te_include": True,
    "cov_include": True,
    "allow_missing_modalities": False,
    "device": "auto",
    "parallel_phase1": False,
    "parallel_phase1_max_workers": None,
    "latent_dim_tnf": 64,
    "latent_dim_te": 5,
    "latent_dim_cov": None,
    "final_dim": None,
    "hidden_scale": 4,
    "hidden_min": 32,
    "hidden_max": 512,
    "dropout": 0.1,
    "n_hidden_layers": 2,
    "beta_tnf": 0.01,
    "beta_te": 0.1,
    "beta_cov": 0.1,
    "kl_anneal_epochs": 75,
    "kl_free_bits": 0.01,
    "phase1_epochs_tnf": 75,
    "phase1_epochs_te": 50,
    "phase1_epochs_cov": 50,
    "phase2_epochs": 50,
    "phase1_lr": 1e-3,
    "phase2_lr": 5e-5,
    "loss_scale_tnf": 1.0,
    "loss_scale_te": 0.1,
    "loss_scale_cov": 0.1,
    "loss_scale_fusion": 0.05,
    "fusion_mode": "global",
    "fusion_gate_entropy_weight": 0.01,
    "cov_use_n_samples_present_as_feature": False,
    "batch_size": None,

    # Clustering
    "cluster_min_contig_len": 2000,
    "analysis_min_length": None,     # MASTER length: see _apply_analysis_min_length()
    "adaptive_gray_ceiling": None,
    "anchor_min_length": 5000,
    "anchor_min_tnf_weight": 0.8,
    "anchor_max_cov_var": 2.0,
    "hdbscan_min_cluster_size": None,
    "hdbscan_min_samples": None,
    "hdbscan_epsilon": 0.1,
    "hdbscan_method": "leaf",
    "hdbscan_soft_prob_min": 0.5,
    "noise_max_distance": 0.5,
    "max_iter": 10,
    "convergence_threshold": 0.01,
    "min_contigs_per_bin": 5,
    "min_bin_length_bp": 500_000,
    "min_bin_n50_bp": 1000,
    "skani_ani_threshold": 95.0,
    "skani_min_af": 0.1,
    "n_cov_meta_cols": 4,
    "allow_missing_valid_mask": False,
    "compute_silhouette": True,
    "silhouette_max_n": 20_000,
    "run_eukcc": True,
    "eukcc_db": "",
    "eukcc_conda_env": "eukcc",
    "eukcc_merge_enabled": True,
    "eukcc_merge_n_combine": 1,
    "eukcc_merge_ani": 99,
    "eukcc_merge_within": 1500,
    "clustering_backend": "auto",
    "clustering_threads": 8,
    "clustering_seed": 42,

    # Output
    "clusters_non_deduplicated_dir": "clusters_non_deduplicated",
    "clusters_deduplicated_dir": "clusters_deduplicated",
}

# The four keys tnf_gene.py's run_tnf_wholecontig() actually allows —
# passing anything else raises ValueError (see tnf_gene.py _ALLOWED_CONFIG_KEYS).
_TNF_CONFIG_KEYS = ("min_contig_len", "min_valid_kmers", "full_confidence_kmers", "weight_mode")


def parse_args():
    p = argparse.ArgumentParser(
        prog="hyphaesbin",
        description="hyphaesbin — Fungi-Specific Metagenome Binning (v7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  # Full pipeline
  python main.py --scaffold contigs.fasta --reads sample:R1[:R2] --outdir results/ --config config.yaml

  # Skip straight to clustering (phase 6), reusing everything already on disk
  python main.py --scaffold contigs.fasta --reads sample:R1[:R2] --outdir results/ --skip-to-phase 6

  # Use pre-computed TE features (skips phase 4 entirely)
  python main.py --scaffold contigs.fasta --reads sample:R1[:R2] --outdir results/ --precomputed-te /path/to/te_features.npy

PHASE NUMBERS (--skip-to-phase N skips phases 1..N-1):
  1 preprocessing | 2 coverage | 3 TNF composition | 4 TE composition |
  5 encoder | 6 clustering
        """
    )
    p.add_argument("--scaffold",               required=True,       help="Assembly FASTA")
    p.add_argument("--reads",                  action="append",
                                               default=[],          help="NAME:R1 or NAME:R1:R2 (repeatable)")
    p.add_argument("--reads-dir",              default=".",         help="Base directory for reads")
    p.add_argument("--outdir",                 required=True,       help="Output directory")
    p.add_argument("--config",                 default=None,        help="Config YAML")
    p.add_argument("--threads",                type=int, default=None,
                                               help="Override config's top-level `threads`")
    p.add_argument("--log-level",              default="INFO",
                                               choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--reset",                  action="store_true",
                                               help="Reset main.py's own top-level checkpoints "
                                                    "(see module docstring item 12 for scope)")
    p.add_argument("--precomputed-te",         default=None,
                                               help="Path to pre-computed te_features.npy — skips phase 4")
    p.add_argument("--precomputed-te-weights", default=None,
                                               help="Path to pre-computed te_weight.npy")
    p.add_argument("--skip-to-phase",          type=int, default=None, choices=range(1, 7),
                                               help="Skip straight to phase N (1-6), reusing "
                                                    "prior phases' outputs already on disk")
    return p.parse_args()


def parse_reads_to_samples_tsv(reads_list: List[str], reads_dir: str, outdir: str) -> str:
    samples_tsv = Path(outdir) / "samples.tsv"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(samples_tsv, "w") as f:
        f.write("sample\tr1\tr2\n")
        for s in reads_list:
            parts = s.split(":")
            if len(parts) == 2:
                name, r1 = parts
                r1_path = r1 if Path(r1).is_absolute() else str(Path(reads_dir) / r1)
                f.write(f"{name}\t{r1_path}\t\n")
            elif len(parts) == 3:
                name, r1, r2 = parts
                r1_path = r1 if Path(r1).is_absolute() else str(Path(reads_dir) / r1)
                r2_path = r2 if Path(r2).is_absolute() else str(Path(reads_dir) / r2)
                f.write(f"{name}\t{r1_path}\t{r2_path}\n")
            else:
                print(f"[WARN] Cannot parse --reads: '{s}' (expected NAME:R1 or NAME:R1:R2)")
    return str(samples_tsv)


def load_config(config_path: Optional[str] = None) -> Dict:
    """Flat top-level merge — config.yaml is a flat namespace (see its own
    header comment); do NOT introduce nesting here."""
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        for k, v in user_config.items():
            if k in config and isinstance(config[k], dict) and isinstance(v, dict):
                config[k].update(v)
            else:
                config[k] = v
    return config


# Every length cutoff that must move together when the analysis scale changes.
# Deliberately NOT included: min_contig_length (the raw floor; keep it at 1000 so
# recruitment can still use 1000..cutoff-1 contigs), recruit_min_len,
# min_contig_len (TNF/TE feature floor), and subcluster_min_bin_contigs (a COUNT).
_LENGTH_KEYS_FOLLOWING_MASTER = ("dedup_min_length", "cluster_min_contig_len",
                                 "tiara_min_len", "whokaryote_minsize",
                                 "adaptive_gray_ceiling")


def _apply_analysis_min_length(config: Dict, log) -> None:
    """ONE knob for the analysis length scale. If `analysis_min_length` is set
    (e.g. 1800), every cutoff that has to agree with it is overwritten with
    that value — dedup floor, clustering floor, Tiara/Whokaryote minimum and
    the step-10 'kept outright' ceiling — so no stage silently keeps working at
    the old 2000. A value here WINS over the individual keys in the yaml, and
    each override is logged. Unset/None/0 = leave every individual key alone."""
    v = config.get("analysis_min_length")
    if v in (None, "", 0):
        return
    try:
        v = int(v)
    except (TypeError, ValueError):
        log_error_flag(log, "BAD_ANALYSIS_MIN_LENGTH",
                       f"analysis_min_length must be an integer, got {v!r}")
        sys.exit(1)
    floor = int(config.get("min_contig_length", 1000))
    if v < floor:
        log_error_flag(log, "BAD_ANALYSIS_MIN_LENGTH",
                       f"analysis_min_length ({v}) is below min_contig_length ({floor}); "
                       f"those contigs are already removed by the length pre-filter.")
        sys.exit(1)
    changes = []
    for k in _LENGTH_KEYS_FOLLOWING_MASTER:
        old = config.get(k)
        if old != v:
            changes.append(f"{k}: {old} -> {v}")
        config[k] = v
    log.info(f"LENGTH  analysis_min_length={v}  applied to {len(_LENGTH_KEYS_FOLLOWING_MASTER)} "
             f"setting(s); min_contig_length={floor} (raw floor) unchanged")
    for c in changes:
        log.info(f"LENGTH    overridden  {c}")
    if config.get("recruit_enabled", True) and int(config.get("recruit_min_len", 1000)) >= v:
        log_warning_flag(log, "RECRUIT_RANGE_EMPTY",
                         f"recruit_min_len ({config.get('recruit_min_len')}) >= analysis_min_length "
                         f"({v}) — short-contig recruitment has no candidates.")


def _file_fingerprint(path) -> str:
    """Name+size+mtime fingerprint -- NOT a content hash. Deliberately
    mirrors the identical helper already used inside preprocessing.py/
    coverage.py/tnf_gene.py/te_composition.py/encoder.py/clustering.py, so
    main.py's own top-level checkpoint gates follow the same,
    already-established convention as every module they wrap (see
    checkpoint.py's own design note: fingerprint logic belongs to each
    caller, not to the shared Checkpoint class -- main.py is a caller like
    any other module here).

    CHANGED: the full absolute PATH used to be part of this fingerprint,
    which meant moving a pipeline (e.g. rsync-ing an in-progress outdir to a
    different server, at a different absolute path, mtimes preserved via
    -a) made every downstream checkpoint look 'stale' and re-run from
    scratch -- including multi-hour steps -- even though the actual file
    content never changed. Now only the basename is used, so relocating
    the whole tree (same filenames, same sizes, same mtimes) is correctly
    still recognized as unchanged. Trade-off: two DIFFERENT files that
    happen to share a basename, size, and to-the-second mtime would be
    (mis)treated as the same file -- accepted as vanishingly unlikely,
    consistent with this helper's existing size+mtime-not-content
    tradeoff."""
    if path is None:
        return "None"
    p = Path(str(path))
    try:
        st = p.stat()
        return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
    except FileNotFoundError:
        return f"{p.name}:MISSING"


def _fingerprint(*parts) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps(parts, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _require_file(path, log, phase_desc: str, what: str) -> str:
    if not path or not Path(path).exists():
        log_error_flag(log, "MISSING_REQUIRED_FILE",
                        f"Cannot {phase_desc}: {what} not found at '{path}'.")
        sys.exit(1)
    return str(path)


def _read_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def _sample_names_from_reads(reads_list: List[str]) -> List[str]:
    names = []
    for s in reads_list:
        parts = s.split(":")
        if len(parts) == 3:
            names.append(parts[0])
    return names


def main():
    args   = parse_args()
    config = load_config(args.config)

    if args.threads:
        config["threads"] = args.threads

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if config.get("resource_profile_enabled", True):
        from hyphaesbin.utils.resource_profiler import configure
        configure(outdir, config.get("resource_profile_interval_seconds", 0.5))

    setup_logger(str(outdir), args.log_level)
    log  = get_logger("main")
    ckpt = Checkpoint(str(outdir))
    _apply_analysis_min_length(config, log)
    # ── MULTI-RUN MODE (all_runs / te_runs / modality_runs / best_of_it in config.yaml; see hyphaesbin/modality_runs.py) ──
    # Inert unless one of those keys is set. When set, the shared features (coverage, TNF, TE) must all exist so that
    # every modality combination can use them, so tnf_include/te_include are forced on for phases 3 and 4.
    _tc = lambda v: v is True or str(v).strip().lower() in ("1", "true", "yes", "on")
    _multi_runs = _tc(config.get("all_runs")) or _tc(config.get("te_runs")) or bool(config.get("modality_runs"))
    _best_of    = _tc(config.get("best_of_it"))
    if _multi_runs:
        config["tnf_include"] = True
        config["te_include"]  = True
        log.info("MULTI-RUN mode: tnf_include/te_include forced on so all modality combinations can share the features")
    # config['force_trust_checkpoints']: default False. When True, EVERY
    # step's fingerprint check (across preprocessing.py/coverage.py/
    # tnf_gene.py/encoder.py/te_composition.py/clustering.py — none of
    # those 6 files are touched by this) is bypassed process-wide — a
    # completed step is skipped as long as its checkpoint file + recorded
    # outputs exist, regardless of config/path/input changes since it ran.
    # See checkpoint.py's set_force_trust_fingerprint() docstring for the
    # tradeoff before turning this on.
    set_force_trust_fingerprint(bool(config.get("force_trust_checkpoints", False)))

    if args.reset:
        log_warning_flag(log, "RESET_SCOPE",
                          "--reset only clears main.py's own top-level checkpoint store "
                          "(<outdir>/checkpoints/). Each phase's own internal checkpoints "
                          "(<outdir>/<phase>/checkpoints/...) are untouched — a 'recomputed' "
                          "phase may still reload its own cached sub-steps almost instantly. "
                          "Delete that phase's output subdirectory for a true from-scratch rerun.")
        ckpt.reset()

    log.info("+" + "=" * 68 + "+")
    log.info("|" + " " * 16 + "hyphaesbin PIPELINE v7 (RESTRUCTURED)" + " " * 16 + "|")
    log.info("+" + "=" * 68 + "+")
    log.info("")
    log.info(f"Scaffold : {args.scaffold}")
    log.info(f"Reads    : {len(args.reads)} sample(s)")
    log.info(f"Outdir   : {outdir}")
    log.info(f"Log file : {get_log_file()}")
    log_threads(log, config["threads"], config["threads"], context="top-level requested")
    log.info("")

    # ── Startup dependency / backend diagnostics (report-only — see deps.py
    # and item 10 of the module docstring; never blocks, never installs) ────
    tools_needed = None
    if str(config.get("te_mode", "fast")).lower() == "fast":
        tools_needed = [t for t in deps.check_external_tools().keys() if t != "RepeatMasker"]
    elif str(config.get("te_mode", "fast")).lower() == "efficient":
        tools_needed = [t for t in deps.check_external_tools().keys() if t != "mmseqs"]
    dep_report = deps.check_all(external_tools_needed=tools_needed, config=config)
    for line in deps.format_report(dep_report).splitlines():
        log.debug(line)
    if dep_report["missing_required_python"]:
        log_warning_flag(log, "MISSING_PYTHON_DEPS",
                          f"{dep_report['missing_required_python']} — the relevant phase(s) "
                          f"will fail loudly when reached; see the DEBUG log for the full report.")
    if dep_report["missing_required_tools"]:
        log_warning_flag(log, "MISSING_EXTERNAL_TOOLS",
                          f"{dep_report['missing_required_tools']} — the relevant phase(s) "
                          f"will fail loudly when reached; see the DEBUG log for the full report.")
    log_device_backend(log, config.get("device", "auto"),
                        deps.describe_backend_choice(config.get("device", "auto"), dep_report["backends"]),
                        extra="encoder device — describe_backend_choice() prediction, not a resolution")
    log_device_backend(log, config.get("clustering_backend", "auto"),
                        deps.describe_backend_choice(config.get("clustering_backend", "auto"), dep_report["backends"]),
                        extra="clustering backend — prediction, not a resolution")
    log.info("")

    log.info("")

    skip_to = args.skip_to_phase or 1
    t0_total = time.time()

    preproc_dir = outdir / "preprocessing"
    cov_dir     = outdir / "coverage"
    tnf_dir     = outdir / "tnf_gene"
    te_dir      = outdir / "te_composition"   # created BY run_te_branch itself
    enc_dir     = outdir / "encoder"
    clust_dir   = outdir / "clustering"

    # =========================================================================
    # PHASE 1: PREPROCESSING
    # =========================================================================
    STEP = "phase1_preprocessing"
    final_dir = preproc_dir / "final"

    if skip_to > 1:
        log.info("Skipping Phase 1 (preprocessing) by request")
        clean_fasta          = _require_file(final_dir / "final_clean.fasta", log, "skip Phase 1", "final_clean.fasta")
        clean_fasta_unmasked = _require_file(final_dir / "final_clean_unmasked.fasta", log, "skip Phase 1", "final_clean_unmasked.fasta")
        coverage_tsv         = _require_file(final_dir / "coverage_table.tsv", log, "skip Phase 1", "coverage_table.tsv")
    elif ckpt.is_done(STEP):
        log_checkpoint(log, STEP, reused=True)
        meta = ckpt.load_metadata(STEP)
        clean_fasta          = meta["final_clean"]
        clean_fasta_unmasked = meta["final_clean_unmasked"]
        coverage_tsv         = meta["coverage_table"]
    else:
        log_checkpoint(log, STEP, reused=False)
        log_step_start(log, 1, "PREPROCESSING", 6)
        t_phase = time.time()

        if not args.reads:
            log_error_flag(log, "NO_READS", "No reads provided. Use --reads NAME:R1 or NAME:R1:R2 (repeatable).")
            sys.exit(1)

        samples_tsv = parse_reads_to_samples_tsv(args.reads, args.reads_dir, str(outdir))
        log.info(f"Samples table: {samples_tsv}")

        try:
            preprocess_outputs = run_preprocessing(
                assembly_input = args.scaffold,
                reads_dir      = args.reads_dir,
                samples_tsv    = samples_tsv,
                outdir         = str(preproc_dir),
                threads        = config["threads"],
                config         = config,
            )
        except SystemExit:
            raise
        except Exception as e:
            log_error_flag(log, "PHASE1_FAILED", f"Preprocessing failed: {e}")
            import traceback; log.debug(traceback.format_exc())
            sys.exit(1)

        clean_fasta          = preprocess_outputs["final_clean"]
        clean_fasta_unmasked = preprocess_outputs["final_clean_unmasked"]
        coverage_tsv         = preprocess_outputs["coverage_table"]

        ckpt.mark_done(
            STEP, metadata=preprocess_outputs,
            output_files=[clean_fasta, clean_fasta_unmasked, coverage_tsv],
            module="preprocessing.py", version="13-step redesign (no formal VERSION const)",
        )
        log_step_done(log, 1, "PREPROCESSING", time.time() - t_phase)

    log_output_paths(log, {
        "final_clean (masked)":    clean_fasta,
        "final_clean_unmasked":    clean_fasta_unmasked,
        "coverage_table":          coverage_tsv,
    })

    # Sample names for coverage's expected_samples cross-check (best-effort —
    # only available when phase 1 actually ran with --reads this run).
    sample_names = _sample_names_from_reads(args.reads) or None

    # =========================================================================
    # PHASE 2: COVERAGE FEATURES
    # =========================================================================
    STEP = "phase2_coverage"
    cov_features_path = cov_dir / "coverage_features.npy"
    cov_manifest_path = cov_dir / "manifest.json"

    if skip_to > 2:
        log.info("Skipping Phase 2 (coverage) by request")
        cov_features_path = Path(_require_file(cov_features_path, log, "skip Phase 2", "coverage_features.npy"))
        _require_file(cov_manifest_path, log, "skip Phase 2", "manifest.json")
    elif ckpt.is_done(STEP):
        log_checkpoint(log, STEP, reused=True)
        meta = ckpt.load_metadata(STEP)
        cov_features_path = Path(meta["cov_features_path"])
    else:
        log_checkpoint(log, STEP, reused=False)
        log_step_start(log, 2, "COVERAGE FEATURES", 6)
        t_phase = time.time()
        n_cores = config.get("coverage_n_cores") or config["threads"]
        log_threads(log, config["threads"], n_cores, context="coverage")

        try:
            coverage_features = run_coverage_features(
                coverage_tsv            = coverage_tsv,
                assembly_fasta          = clean_fasta,
                outdir                  = str(cov_dir),
                k                       = int(config.get("coverage_k", 200)),
                n_cores                 = int(n_cores),
                resume                  = True,
                expected_samples        = sample_names,
                strict_numeric_columns  = bool(config.get("coverage_strict_numeric_columns", True)),
                outlier_cap_percentile  = config.get("coverage_outlier_cap_percentile", 99.5),
                deterministic           = bool(config.get("coverage_deterministic", False)),
                brute_force_max_n       = int(config.get("coverage_brute_force_max_n", 5000)),
            )
        except SystemExit:
            raise
        except Exception as e:
            log_error_flag(log, "PHASE2_FAILED", f"Coverage features failed: {e}")
            import traceback; log.debug(traceback.format_exc())
            sys.exit(1)

        ckpt.mark_done(
            STEP,
            metadata={"cov_features_path": str(cov_features_path), "shape": list(coverage_features.shape)},
            output_files=[str(cov_features_path), str(cov_manifest_path)],
            module="coverage.py", version="v9",
        )
        log_step_done(log, 2, "COVERAGE FEATURES", time.time() - t_phase)
        log_contig_counts(log, total=coverage_features.shape[0])

    cov_manifest = _read_json(cov_manifest_path) if cov_manifest_path.exists() else {}
    n_samples = cov_manifest.get("n_samples")
    log_output_paths(log, {"coverage_features": str(cov_features_path)})

    # =========================================================================
    # PHASE 3: TNF COMPOSITION (masked FASTA)
    # =========================================================================
    STEP = "phase3_tnf"
    tnf_features_path = tnf_dir / "tnf_features.npy"
    tnf_weights_path  = tnf_dir / "tnf_weights.npy"
    tnf_include = bool(config.get("tnf_include", True))

    if not tnf_include:
        log.info("TNF composition: DISABLED (tnf_include=false) — skipping Phase 3")
        tnf_features_path_str = None
        tnf_weights_path_str  = None
    elif skip_to > 3:
        log.info("Skipping Phase 3 (TNF) by request")
        tnf_features_path_str = _require_file(tnf_features_path, log, "skip Phase 3", "tnf_features.npy")
        tnf_weights_path_str  = str(tnf_weights_path) if tnf_weights_path.exists() else None
    elif ckpt.is_done(STEP):
        log_checkpoint(log, STEP, reused=True)
        meta = ckpt.load_metadata(STEP)
        tnf_features_path_str = meta["features"]
        tnf_weights_path_str  = meta.get("weights")
    else:
        log_checkpoint(log, STEP, reused=False)
        if not TNF_AVAILABLE:
            log_error_flag(log, "TNF_UNAVAILABLE", f"TNF module could not be imported: {_TNF_IMPORT_ERROR}")
            sys.exit(1)
        log_step_start(log, 3, "TNF COMPOSITION (whole-contig k4 frequency)", 6)
        t_phase = time.time()

        tnf_cfg = {k: config[k] for k in _TNF_CONFIG_KEYS if k in config}
        try:
            tnf_features, tnf_weights, tnf_contig_ids = run_tnf_wholecontig(
                scaffold = clean_fasta,   # MASKED (see docstring item 3)
                outdir   = str(tnf_dir),
                config   = tnf_cfg,
                resume   = True,
            )
        except SystemExit:
            raise
        except Exception as e:
            log_error_flag(log, "PHASE3_FAILED", f"TNF composition failed: {e}")
            import traceback; log.debug(traceback.format_exc())
            sys.exit(1)

        tnf_features_path_str = str(tnf_features_path)
        tnf_weights_path_str  = str(tnf_weights_path)
        ckpt.mark_done(
            STEP,
            metadata={"features": tnf_features_path_str, "weights": tnf_weights_path_str,
                      "shape": list(tnf_features.shape)},
            output_files=[tnf_features_path_str, tnf_weights_path_str,
                          str(tnf_dir / "contig_ids.json"), str(tnf_dir / "tnf_feature_schema.json")],
            module="tnf_gene.py", version="v2_streaming",
        )
        log_step_done(log, 3, "TNF COMPOSITION", time.time() - t_phase)
        log_contig_counts(log, total=len(tnf_contig_ids))

    if tnf_features_path_str:
        log_output_paths(log, {"tnf_features": tnf_features_path_str})

    # =========================================================================
    # PHASE 4: TE COMPOSITION (unmasked FASTA)
    # =========================================================================
    STEP = "phase4_te"
    te_features_path = te_dir / "te_features.npy"
    te_weight_path   = te_dir / "te_weight.npy"
    te_include = bool(config.get("te_include", True))

    if args.precomputed_te:
        log.info("Using pre-computed TE features — skipping Phase 4")
        te_features_path_str = _require_file(args.precomputed_te, log, "use --precomputed-te", "te_features.npy")
        te_weight_path_str   = args.precomputed_te_weights if args.precomputed_te_weights else None
    elif not te_include:
        log.info("TE composition: DISABLED (te_include=false) — skipping Phase 4")
        te_features_path_str = None
        te_weight_path_str   = None
    elif skip_to > 4:
        log.info("Skipping Phase 4 (TE) by request")
        te_features_path_str = _require_file(te_features_path, log, "skip Phase 4", "te_features.npy")
        te_weight_path_str   = str(te_weight_path) if te_weight_path.exists() else None
    elif ckpt.is_done(STEP):
        log_checkpoint(log, STEP, reused=True)
        meta = ckpt.load_metadata(STEP)
        te_features_path_str = meta["features"]
        te_weight_path_str   = meta.get("weights")
    else:
        log_checkpoint(log, STEP, reused=False)
        if not TE_AVAILABLE:
            log_error_flag(log, "TE_UNAVAILABLE", f"TE module could not be imported: {_TE_IMPORT_ERROR}")
            sys.exit(1)
        funtedb = config.get("funTEdb") or config.get("funtedb") or config.get("FunTEDB") or ""
        if not funtedb:
            log_error_flag(log, "MISSING_FUNTEDB", "config['funTEdb'] is empty — TE composition requires a TE database FASTA.")
            sys.exit(1)

        log_step_start(log, 4, "TE COMPOSITION", 6)
        t_phase = time.time()
        try:
            # run_te_branch appends "te_composition" onto whatever outdir it's
            # given (see te_composition.py) -- pass the pipeline's TOP-LEVEL
            # outdir here, not te_dir, or the path doubles up.
            te_features, te_weight, te_bed_path = run_te_branch(
                scaffold = clean_fasta_unmasked,   # UNMASKED (see docstring item 3)
                config   = config,
                outdir   = str(outdir),
            )
        except SystemExit:
            raise
        except Exception as e:
            log_error_flag(log, "PHASE4_FAILED", f"TE composition failed: {e}")
            import traceback; log.debug(traceback.format_exc())
            sys.exit(1)

        te_features_path_str = str(te_features_path)
        te_weight_path_str   = str(te_weight_path)
        ckpt.mark_done(
            STEP,
            metadata={"features": te_features_path_str, "weights": te_weight_path_str,
                      "te_bed": te_bed_path, "shape": list(te_features.shape)},
            output_files=[te_features_path_str, te_weight_path_str,
                          str(te_dir / "contig_ids.json"), str(te_dir / "schema.json")],
            module="te_composition.py", version="V33_V22_5D",
        )
        log_step_done(log, 4, "TE COMPOSITION", time.time() - t_phase)

    if te_features_path_str:
        log_output_paths(log, {"te_features": te_features_path_str})

    # =========================================================================
    # MULTI-RUN MODE: encoder + clustering per modality combination, then best-of-it (replaces the default phases 5-6)
    # =========================================================================
    if _multi_runs or _best_of:
        from hyphaesbin.modality_runs import run_modality_runs
        _t_multi = time.time()
        try:
            run_modality_runs(
                config=config, outdir=outdir,
                tnf_features_path=tnf_features_path_str, tnf_weights_path=tnf_weights_path_str,
                te_features_path=te_features_path_str,  te_weights_path=te_weight_path_str,
                cov_features_path=str(cov_features_path),
                contig_ids_path=str(cov_dir / "contig_ids.txt"),
                fasta_path=clean_fasta_unmasked,        # UNMASKED, same as the single-run phase 6
                n_samples=n_samples)
        except SystemExit:
            raise
        except Exception as e:
            log_error_flag(log, "MULTI_RUN_FAILED", f"Multi-run mode failed: {e}")
            import traceback; log.debug(traceback.format_exc())
            sys.exit(1)
        log.info("")
        log.info(f"MULTI-RUN DONE in {(time.time() - _t_multi) / 3600:.2f} hours -> {outdir / 'runs'}"
                 + (f"  and  {outdir / 'best_of_it'}" if _best_of else ""))
        return

    # =========================================================================
    # PHASE 5: ENCODER
    # =========================================================================
    STEP = "phase5_encoder"
    final_latent_path    = enc_dir / "final_latent.npy"
    encoder_manifest_path = enc_dir / "encoder_manifest.json"
    cov_include = bool(config.get("cov_include", True))

    if not (tnf_include or te_include or cov_include):
        log_error_flag(log, "NO_MODALITIES",
                        "tnf_include, te_include, and cov_include are all False — at least one "
                        "modality must be enabled to run the encoder.")
        sys.exit(1)

    if skip_to > 5:
        log.info("Skipping Phase 5 (encoder) by request")
        final_latent_path_str = _require_file(final_latent_path, log, "skip Phase 5", "final_latent.npy")
    elif ckpt.is_done(STEP):
        log_checkpoint(log, STEP, reused=True)
        meta = ckpt.load_metadata(STEP)
        final_latent_path_str = meta["final_latent"]
    else:
        log_checkpoint(log, STEP, reused=False)
        if not ENCODER_AVAILABLE:
            log_error_flag(log, "ENCODER_UNAVAILABLE", f"Encoder module could not be imported: {_ENCODER_IMPORT_ERROR}")
            sys.exit(1)
        log_step_start(log, 5, "ENCODER TRAINING", 6)
        t_phase = time.time()

        encoder_cfg = EncoderConfig.from_dict(config)   # safely filters unknown keys
        log.info(f"Active modalities requested: tnf={tnf_include} te={te_include} cov={cov_include} "
                 f"(allow_missing_modalities={encoder_cfg.allow_missing_modalities})")

        try:
            final_latent_path_str = run_encoder(
                outdir             = str(enc_dir),
                tnf_features_path  = tnf_features_path_str if tnf_include else None,
                cov_features_path  = str(cov_features_path) if cov_include else None,
                te_features_path   = te_features_path_str if te_include else None,
                tnf_weights_path   = tnf_weights_path_str,
                te_weights_path    = te_weight_path_str,
                config             = encoder_cfg,
                n_samples          = int(n_samples) if n_samples is not None else None,
            )
        except SystemExit:
            raise
        except Exception as e:
            log_error_flag(log, "PHASE5_FAILED", f"Encoder training failed: {e}")
            import traceback; log.debug(traceback.format_exc())
            sys.exit(1)

        ckpt.mark_done(
            STEP, metadata={"final_latent": final_latent_path_str},
            output_files=[final_latent_path_str, str(encoder_manifest_path)],
            module="encoder.py", version="v8.1",
        )
        log_step_done(log, 5, "ENCODER TRAINING", time.time() - t_phase)

        if encoder_manifest_path.exists():
            enc_manifest = _read_json(encoder_manifest_path)
            log_device_backend(log, config.get("device", "auto"), enc_manifest.get("device", "?"),
                                extra="encoder actual (from encoder_manifest.json)")

    log_output_paths(log, {"final_latent": final_latent_path_str,
                            "encoder_manifest": str(encoder_manifest_path)})

    # =========================================================================
    # PHASE 6: CLUSTERING
    # =========================================================================
    STEP = "phase6_clustering"
    clustering_stats_path = clust_dir / "clustering_stats.json"

    if not CLUSTERING_AVAILABLE:
        log_error_flag(log, "CLUSTERING_UNAVAILABLE", f"Clustering module could not be imported: {_CLUSTERING_IMPORT_ERROR}")
        sys.exit(1)

    clustering_cfg = ClusteringConfig.from_dict(config)  # safely filters unknown keys
    contig_ids_path_for_clustering = str(cov_dir / "contig_ids.txt")

    # main.py's own checkpoint gate must fingerprint clustering's real inputs
    # AND its resolved config, not just check "did phase6_clustering finish
    # once, and does cluster_summary.tsv still exist". An existence-only gate
    # would happily reuse a stale cluster_summary.tsv after final_latent.npy
    # was retrained, the coverage/valid-mask array changed, or any
    # ClusteringConfig value (backend, seed, HDBSCAN/dedup/EukCC settings,
    # ...) was edited in config.yaml between runs -- clustering.py's own
    # internal fingerprinting never gets a chance to catch that, because the
    # whole point of this gate is to decide whether to call run_clustering()
    # at all. This mirrors the exact `_file_fingerprint`/`_fingerprint`
    # convention every other module here already uses (see the helpers
    # above), applied at main.py's own orchestration level.
    fp6 = _fingerprint(
        _file_fingerprint(final_latent_path_str),
        _file_fingerprint(str(encoder_manifest_path)),
        _file_fingerprint(contig_ids_path_for_clustering),
        _file_fingerprint(clean_fasta_unmasked),
        _file_fingerprint(str(cov_features_path)),
        _file_fingerprint(tnf_weights_path_str),
        _file_fingerprint(te_weight_path_str),
        asdict(clustering_cfg),
    )

    prev_done = ckpt.is_done(STEP)
    prev_meta = ckpt.load_metadata(STEP) if prev_done else {}
    stale = (not prev_done) or (prev_meta.get("_fp") != fp6)

    if not stale:
        log_checkpoint(log, STEP, reused=True)
        cluster_summary_path = prev_meta["cluster_summary"]
    else:
        log_checkpoint(log, STEP, reused=False,
                        reason="inputs/config changed since last run" if prev_done else "")
        log_step_start(log, 6, "CLUSTERING + REFINEMENT", 6)
        t_phase = time.time()

        # NOTE on alignment_bam_paths (EukCC split-bin merging): run_clustering()
        # accepts an OPTIONAL alignment_bam_paths list -- paired-end BAM
        # alignments against the BIN FASTAs -- to drive EukCC's real --links
        # merge workflow (clustering.py FIX 13). No stage in the current
        # six-module pipeline (preprocessing/coverage/TNF/TE/encoder/
        # clustering) produces that BAM: preprocessing step 9 maps reads
        # against the WHOLE assembly (not per-bin FASTAs) and deletes those
        # BAMs as intermediate files right after step 12
        # (cleanup_intermediate()), well before clustering ever runs -- and
        # even if kept, they'd be the wrong reference for binlinks.py, which
        # needs alignments against the bins themselves. clustering.py's own
        # docstring says this outright: "This requires paired-end read
        # alignment (BAM) as an input, which no earlier stage of this
        # pipeline currently produces or was described as producing."
        # Passing `alignment_bam_paths=None` here is therefore not an
        # oversight -- it's the honest reflection of that gap. EukCC's
        # per-bin QUALITY assessment (run_eukcc(), step 9 inside
        # clustering.py) still runs and is unaffected; only the OPTIONAL
        # split-bin MERGE step is skipped, exactly as clustering.py itself
        # reports via clustering_stats.json's "eukcc_merge_status":
        # "skipped_no_bam_alignment". Adding a new mapping-against-bins
        # stage to actually produce this BAM would be a new pipeline
        # capability, not a main.py wiring fix -- out of scope here unless
        # you want that stage added.
        try:
            cluster_summary_path = run_clustering(
                final_latent_path      = final_latent_path_str,
                encoder_manifest_path  = str(encoder_manifest_path),
                contig_ids_path        = contig_ids_path_for_clustering,
                fasta_path             = clean_fasta_unmasked,   # UNMASKED — final bin output (see docstring item 3)
                cov_features_path      = str(cov_features_path),
                outdir                 = str(clust_dir),
                tnf_weights_path       = tnf_weights_path_str,
                te_weights_path        = te_weight_path_str,
                alignment_bam_paths    = None,   # see note above — no source of this exists yet
                config                 = clustering_cfg,
            )
        except SystemExit:
            raise
        except Exception as e:
            log_error_flag(log, "PHASE6_FAILED", f"Clustering failed: {e}")
            import traceback; log.debug(traceback.format_exc())
            sys.exit(1)

        ckpt.mark_done(
            STEP, metadata={"cluster_summary": cluster_summary_path, "_fp": fp6},
            output_files=[cluster_summary_path],
            module="clustering.py", version="v7.2",
        )
        log_step_done(log, 6, "CLUSTERING + REFINEMENT", time.time() - t_phase)

        if clustering_stats_path.exists():
            stats = _read_json(clustering_stats_path)
            backend_info = stats.get("backend", {})
            log_device_backend(log, backend_info.get("requested", clustering_cfg.clustering_backend),
                                backend_info.get("actual", "?"),
                                extra="clustering actual (from clustering_stats.json)")

    nondedup_dir = clust_dir / clustering_cfg.clusters_non_deduplicated_dir
    dedup_dir    = clust_dir / clustering_cfg.clusters_deduplicated_dir

    # =========================================================================
    # COMPLETION SUMMARY
    # =========================================================================
    t_total = time.time() - t0_total
    log.info("")
    log.info("+" + "=" * 68 + "+")
    log.info("|" + " " * 24 + "PIPELINE COMPLETE" + " " * 26 + "|")
    log.info("+" + "=" * 68 + "+")
    log.info(f"Total time : {t_total/3600:.2f} hours")
    log.info("")

    output_paths = {
        "final_clean (masked)":       clean_fasta,
        "final_clean_unmasked":       clean_fasta_unmasked,
        "coverage_features":          str(cov_features_path),
        "coverage_manifest":          str(cov_manifest_path),
        "final_latent":               final_latent_path_str,
        "encoder_manifest":           str(encoder_manifest_path),
        "cluster_summary":            cluster_summary_path,
        "clusters_non_deduplicated":  str(nondedup_dir),
        "clusters_deduplicated":      str(dedup_dir),
        "clustering_stats":           str(clustering_stats_path),
        "log_file":                   get_log_file(),
    }
    if tnf_features_path_str:
        output_paths["tnf_features"] = tnf_features_path_str
    if te_features_path_str:
        output_paths["te_features"] = te_features_path_str
    log_output_paths(log, output_paths)

    print("\nOUTPUTS:")
    for name, path in output_paths.items():
        print(f"{name.upper()}={path}")


if __name__ == "__main__":
    main()





#"""
#hyphaesbin Main — Complete Pipeline v7 (restructured to match the finalized
#preprocessing / coverage / TNF / TE / encoder / clustering modules)
#=============================================================================
#Order (per the requested restructure — NOTE this SWAPS TNF and TE relative
#to the old v6 file's Phase3=TE/Phase4=TNF ordering):
#
#  Phase 1: Preprocessing     — assembly QC, read mapping, coverage table
#  Phase 2: Coverage Features — percentile-capped, median-positive-scaled,
#                                log1p coverage + k-NN distances (N+4D)
#  Phase 3: TNF Composition   — whole-contig k4mer frequency (masked FASTA)
#  Phase 4: TE Composition    — MMseqs2/RepeatMasker -> V22_5D (unmasked FASTA)
#  Phase 5: Encoder           — beta-VAE fusion (TNF + TE + COV) -> latent
#  Phase 6: Clustering        — HDBSCAN -> bins -> skani dedup -> EukCC
#
#=============================================================================
#CHANGES FROM v6 (every one verified against the ACTUAL current module code
#in this delivery, not assumed from the old file or from any docstring
#claim — see the individual module files for the source of truth):
#
#  1. run_module2_lite -> run_preprocessing.  The old file called a function
#     (`run_module2_lite`) that does not exist anywhere in the current
#     preprocessing.py; the real entry point is `run_preprocessing(
#     assembly_input, reads_dir, samples_tsv, outdir, threads, config)` — no
#     `resume=`/`mito_multiplier=` kwargs (both gone; preprocessing.py always
#     resumes internally via its own 13-step fingerprinted checkpoints, and
#     mito_coverage_multiplier is not a key anywhere in its DEFAULT_CONFIG
#     any more). The `--mito-multiplier` CLI flag is removed accordingly.
#
#  2. TNF and TE swapped (Phase 3 / Phase 4) per this restructure's explicit
#     instruction. They have no data dependency on each other (both read
#     directly from preprocessing's two FASTA variants), so the swap is
#     purely cosmetic/organizational and changes no numeric result.
#
#  3. Masked vs. unmasked FASTA routing is now explicit and taken from
#     preprocessing.py step 8's own recorded `module_routing` metadata
#     (see its docstring), not assumed:
#       - TNF composition   : MASKED   (`final_clean.fasta`)
#       - TE  composition   : UNMASKED (`final_clean_unmasked.fasta`)
#       - Clustering's `fasta_path` (used ONLY to extract the literal
#         sequence written into each output bin FASTA — see clustering.py's
#         `write_bin_fastas()` / its own docstring "name-keyed lookup only")
#         is UNMASKED, per module_routing's own "final_bin_output": "unmasked"
#         entry — a final delivered genome bin should not contain
#         barrnap-masked N-runs over its rRNA loci. This was NOT explicitly
#         listed in the "pass the correct FASTA variants" instruction (which
#         only named TNF/TE); it is resolved here from preprocessing's own
#         routing table plus a direct read of what clustering.py actually
#         does with that argument. Coverage's `assembly_fasta` argument is
#         used only for contig-ID/order reconciliation (never sequence
#         content), so masked-vs-unmasked is immaterial there; the masked
#         copy is passed for consistency with the rest of the "masked by
#         default" convention.
#
#  4. "Pass n_samples to coverage": `run_coverage_features()` has no literal
#     `n_samples` integer parameter — n_samples is derived internally from
#     the coverage TSV's own column count. The closest real hook is
#     `expected_samples` (an explicit list of sample names cross-checked
#     against the TSV's columns, raising loudly on any mismatch instead of
#     silently trusting the TSV) — main.py builds that list from the parsed
#     `--reads` and passes it. The authoritative sample COUNT that flows
#     onward to the encoder is then read back from coverage's own
#     manifest.json ("n_samples" key — see coverage.py ~line 915), which
#     works correctly even under `--skip-to-phase` where no `--reads` were
#     given this run.
#
#  5. "Pass n_samples ... to encoder": `run_encoder()` DOES take a literal
#     `n_samples: Optional[int]` kwarg (cross-checked against the raw COV
#     array's column count before splitting) — populated from coverage's
#     manifest.json as described above.
#
#  6. "Pass generated feature paths, weight paths, and manifests to
#     encoder": encoder.py's `_load_modality_manifest()` already
#     auto-discovers each modality's manifest as a SIBLING file next to the
#     feature path it's given (TNF: contig_ids.json + tnf_feature_schema.json;
#     TE: contig_ids.json + schema.json; COV: contig_ids.txt + manifest.json)
#     — there is no separate manifest-path argument to pass. main.py only
#     needs to pass the three feature paths + two weight paths; the manifests
#     are found automatically because every phase here writes its outputs
#     into one directory per modality (unchanged from v6's layout).
#
#  7. "Pass final_latent.npy, encoder_manifest.json, canonical IDs, FASTA,
#     and coverage mask to clustering": encoder_manifest_path and
#     contig_ids_path are passed explicitly below (both also have working
#     defaults inside run_clustering() if omitted, since they sit alongside
#     final_latent_path / cov_features_path respectively — passed explicitly
#     here anyway for clarity and so `--skip-to-phase 6` still works with no
#     ambiguity). There is no separate "coverage mask" file: clustering.py
#     reads it as column 0 of the SAME `cov_features.npy` already being
#     passed via `cov_features_path` (see clustering.py's own
#     `n_cov_meta_cols`/`allow_missing_valid_mask` docs, and its
#     `filter_contigs_for_clustering()`) — nothing extra to wire up.
#
#  8. Phase 6's old file assumed a `clustering_output/checkpoints/
#     final_labels.npy` file existed to recover bin counts/labels. No such
#     file exists anywhere in the current clustering.py — `run_clustering()`
#     returns only the path to `cluster_summary.tsv`, and the real,
#     resumable, per-run artifacts are `clusters_non_deduplicated/`,
#     `clusters_deduplicated/`, `cluster_summary.tsv`, `cluster_assignments.tsv`,
#     and `clustering_stats.json`. The completion summary below reports
#     those real paths instead of a labels array that was never written.
#
#  9. CheckM2 is gone from this file entirely (removed from clustering.py in
#     an earlier pass, per explicit prior request) — no `run_checkm2`/
#     `checkm2_db`/`checkm2_conda_env` references remain anywhere below.
#
# 10. `ensure_python_deps()` / `ensure_external_tools()` (which used to run
#     at import time and could silently `conda install`/`pip install`) are
#     gone — deps.py no longer has an installer at all (see deps.py's own
#     changelog). main.py instead calls `deps.check_all()` once at startup
#     purely to LOG a diagnostic report (required-vs-optional packages,
#     external tool availability, compute backends) — it never blocks or
#     installs anything; each stage's own module still fails loudly on its
#     own if something it actually needs turns out to be missing.
#
# 11. Every one of the six phases is now gated by main.py's OWN top-level
#     checkpoint (`ckpt.is_done("phaseN_...")`), not just phases 2/4/6 as in
#     the old file — this is what "avoid duplicate execution when a valid
#     checkpoint exists" means in practice: a plain rerun of `main.py` with
#     no flags changed skips every already-completed phase entirely instead
#     of re-invoking (even a fast, self-resuming) call into that module.
#     `checkpoint.py`'s new output-file verification means a checkpoint that
#     LOOKS done but whose output files were deleted/moved is correctly
#     treated as a cache miss instead of a false "reused".
#
# 12. `--reset` clears only main.py's OWN top-level checkpoint store
#     (`<outdir>/checkpoints/`) exactly as in v6. It does NOT reach into each
#     phase's own internal, independently-fingerprinted checkpoint directory
#     (e.g. `<outdir>/preprocessing/checkpoints/`, `<outdir>/encoder/
#     checkpoints/tnf/`, etc.) — those are separate `Checkpoint` instances
#     each module owns. This is disclosed via a warning at startup when
#     `--reset` is used, since it's an easy thing to assume `--reset` fixes
#     and then be surprised when a phase "reruns" but instantly reloads its
#     own cached sub-steps. To force a truly from-scratch rerun of one
#     phase, delete that phase's own output subdirectory.
#"""
#
#import argparse
#import json
#import sys
#import time
#from dataclasses import asdict
#from pathlib import Path
#from typing import Dict, List, Optional
#
#import yaml
#import numpy as np
#import pandas as pd
#
#from hyphaesbin.utils.logger import (
#    setup_logger, get_logger, get_log_file,
#    log_step_start, log_step_done, log_warning_flag, log_error_flag,
#    log_suggestion, log_device_backend, log_threads, log_checkpoint,
#    log_contig_counts, log_output_paths,
#)
#from hyphaesbin.utils.checkpoint import Checkpoint
#from hyphaesbin.utils import deps
#
## Preprocessing / coverage are load-bearing for every run — no graceful
## degradation is possible without them, so they're imported directly
## (matches v6's own treatment of these two).
#from hyphaesbin.preprocessing.preprocessing import run_preprocessing
#from hyphaesbin.coverage.coverage import run_coverage_features
#
## TE / TNF / encoder / clustering — imported defensively so a broken/absent
## optional dependency inside one of them (e.g. no PyTorch) produces one
## clear error message instead of an ImportError traceback from deep inside
## argparse handling.
#try:
#    from hyphaesbin.composition.composition.TE_composition.te_composition import run_te_branch
#    TE_AVAILABLE = True
#except ImportError as _e:
#    TE_AVAILABLE = False
#    _TE_IMPORT_ERROR = _e
#
#try:
#    from hyphaesbin.composition.composition.TNF_gene.tnf_gene import run_tnf_wholecontig
#    TNF_AVAILABLE = True
#except ImportError as _e:
#    TNF_AVAILABLE = False
#    _TNF_IMPORT_ERROR = _e
#
#try:
#    from hyphaesbin.encoder.encoder import run_encoder, EncoderConfig
#    ENCODER_AVAILABLE = True
#except ImportError as _e:
#    ENCODER_AVAILABLE = False
#    _ENCODER_IMPORT_ERROR = _e
#
#try:
#    from hyphaesbin.clustering.clustering import run_clustering, ClusteringConfig
#    CLUSTERING_AVAILABLE = True
#except ImportError as _e:
#    CLUSTERING_AVAILABLE = False
#    _CLUSTERING_IMPORT_ERROR = _e
#
#
## =============================================================================
## DEFAULT CONFIG — mirrors config.yaml's real, current defaults (NOT the
## stale v6 values). config.yaml is the source of truth; this dict only
## exists so `main.py` still runs sensibly with no `--config` at all.
## =============================================================================
#
#DEFAULT_CONFIG: Dict = {
#    # General
#    "threads": 64,
#    "seed": 42,
#    "min_contig_length": 1000,
#
#    # Preprocessing
#    "read_type": "auto",
#    "read_type_detect_seed": 0,
#    "auto_install_tools": False,
#    "dedup_ani": 99.0,
#    "dedup_min_af": 50.0,
#    "dedup_min_length_ratio": 0.8,
#    "skip_dedup": False,
#    "dedup_max_contigs": 500_000,
#    "tiara_min_len": 1000,
#    "tiara_prob_cutoff": [0.65, 0.65],
#    "whokaryote_minsize": 1000,
#    "whokaryote_model": "T",
#    "classification_confidence_high": 0.80,
#    "rule_based_high_conf_sources": None,
#    "barrnap_kingdom": "euk",
#    "max_mapping_workers": "auto",
#    "min_threads_per_sample": 8,
#    "adaptive_min_cov": 2.0,
#    "adaptive_min_samples": 2,
#    "adaptive_min_breadth": 0.3,
#    "rescue_enabled": True,
#    "rescue_reference_fasta": "",
#    "rescue_min_identity": 75.0,
#    "rescue_min_query_cov": 0.5,
#    "scoring_cov_cv_high": 1.5,
#    "scoring_gc_zscore_flag": 3.0,
#    "scoring_tnf_dist_flag": 2.5,
#    "enable_tnf_qc_signal": True,
#
#    # Coverage features
#    "coverage_k": 200,
#    "coverage_n_cores": None,   # None -> falls back to top-level "threads"
#    "coverage_strict_numeric_columns": True,
#    "coverage_outlier_cap_percentile": 99.5,
#    "coverage_deterministic": False,
#    "coverage_brute_force_max_n": 5000,
#
#    # Databases
#    "funTEdb": "",
#
#    # TE composition
#    "te_mode": "fast",
#    "mmseqs_sensitivity": 5.7,
#    "mmseqs_min_seq_id": 0.7,
#    "mmseqs_coverage": 0.1,
#    "mmseqs_max_seq_len": 100000,
#    "mmseqs_evalue": 1e-5,
#    "te_pa": 36,
#    "te_frag": 20_000_000,
#    "te_min_sw_score": 225,
#    "apply_length_weight_to_features": False,
#    "full_confidence_te_bp": 5000,
#    "full_confidence_length": 10000,
#    "full_confidence_hits": 5,
#    "weight_te_coef": 0.55,
#    "weight_hit_coef": 0.25,
#    "weight_quality_coef": 0.20,
#    "repeatmasker_quality_reference_score": 1000.0,
#    "unclassified_warn_fraction": 0.95,
#
#    # TNF
#    "min_contig_len": 1000,
#    "min_valid_kmers": 50,
#    "full_confidence_kmers": 5000,
#    "weight_mode": "confidence",
#
#    # Encoder
#    "tnf_include": True,
#    "te_include": True,
#    "cov_include": True,
#    "allow_missing_modalities": False,
#    "device": "auto",
#    "parallel_phase1": False,
#    "parallel_phase1_max_workers": None,
#    "latent_dim_tnf": 64,
#    "latent_dim_te": 5,
#    "latent_dim_cov": None,
#    "final_dim": None,
#    "hidden_scale": 4,
#    "hidden_min": 32,
#    "hidden_max": 512,
#    "dropout": 0.1,
#    "n_hidden_layers": 2,
#    "beta_tnf": 0.01,
#    "beta_te": 0.1,
#    "beta_cov": 0.1,
#    "kl_anneal_epochs": 75,
#    "kl_free_bits": 0.01,
#    "phase1_epochs_tnf": 75,
#    "phase1_epochs_te": 50,
#    "phase1_epochs_cov": 50,
#    "phase2_epochs": 50,
#    "phase1_lr": 1e-3,
#    "phase2_lr": 5e-5,
#    "loss_scale_tnf": 1.0,
#    "loss_scale_te": 0.1,
#    "loss_scale_cov": 0.1,
#    "loss_scale_fusion": 0.05,
#    "fusion_mode": "global",
#    "fusion_gate_entropy_weight": 0.01,
#    "cov_use_n_samples_present_as_feature": False,
#    "batch_size": None,
#
#    # Clustering
#    "cluster_min_contig_len": 2000,
#    "anchor_min_length": 5000,
#    "anchor_min_tnf_weight": 0.8,
#    "anchor_max_cov_var": 2.0,
#    "hdbscan_min_cluster_size": None,
#    "hdbscan_min_samples": None,
#    "hdbscan_epsilon": 0.1,
#    "hdbscan_method": "leaf",
#    "hdbscan_soft_prob_min": 0.5,
#    "noise_max_distance": 0.5,
#    "max_iter": 10,
#    "convergence_threshold": 0.01,
#    "min_contigs_per_bin": 5,
#    "min_bin_length_bp": 500_000,
#    "min_bin_n50_bp": 1000,
#    "skani_ani_threshold": 95.0,
#    "skani_min_af": 0.1,
#    "n_cov_meta_cols": 4,
#    "allow_missing_valid_mask": False,
#    "compute_silhouette": True,
#    "silhouette_max_n": 20_000,
#    "run_eukcc": True,
#    "eukcc_db": "",
#    "eukcc_conda_env": "eukcc",
#    "eukcc_merge_enabled": True,
#    "eukcc_merge_n_combine": 1,
#    "eukcc_merge_ani": 99,
#    "eukcc_merge_within": 1500,
#    "clustering_backend": "auto",
#    "clustering_threads": 8,
#    "clustering_seed": 42,
#
#    # Output
#    "clusters_non_deduplicated_dir": "clusters_non_deduplicated",
#    "clusters_deduplicated_dir": "clusters_deduplicated",
#}
#
## The four keys tnf_gene.py's run_tnf_wholecontig() actually allows —
## passing anything else raises ValueError (see tnf_gene.py _ALLOWED_CONFIG_KEYS).
#_TNF_CONFIG_KEYS = ("min_contig_len", "min_valid_kmers", "full_confidence_kmers", "weight_mode")
#
#
#def parse_args():
#    p = argparse.ArgumentParser(
#        prog="hyphaesbin",
#        description="hyphaesbin — Fungi-Specific Metagenome Binning (v7)",
#        formatter_class=argparse.RawDescriptionHelpFormatter,
#        epilog="""
#EXAMPLES
#  # Full pipeline
#  python main.py --scaffold contigs.fasta --reads sample:R1[:R2] --outdir results/ --config config.yaml
#
#  # Skip straight to clustering (phase 6), reusing everything already on disk
#  python main.py --scaffold contigs.fasta --reads sample:R1[:R2] --outdir results/ --skip-to-phase 6
#
#  # Use pre-computed TE features (skips phase 4 entirely)
#  python main.py --scaffold contigs.fasta --reads sample:R1[:R2] --outdir results/ --precomputed-te /path/to/te_features.npy
#
#PHASE NUMBERS (--skip-to-phase N skips phases 1..N-1):
#  1 preprocessing | 2 coverage | 3 TNF composition | 4 TE composition |
#  5 encoder | 6 clustering
#        """
#    )
#    p.add_argument("--scaffold",               required=True,       help="Assembly FASTA")
#    p.add_argument("--reads",                  action="append",
#                                               default=[],          help="NAME:R1 or NAME:R1:R2 (repeatable)")
#    p.add_argument("--reads-dir",              default=".",         help="Base directory for reads")
#    p.add_argument("--outdir",                 required=True,       help="Output directory")
#    p.add_argument("--config",                 default=None,        help="Config YAML")
#    p.add_argument("--threads",                type=int, default=None,
#                                               help="Override config's top-level `threads`")
#    p.add_argument("--log-level",              default="INFO",
#                                               choices=["DEBUG", "INFO", "WARNING", "ERROR"])
#    p.add_argument("--reset",                  action="store_true",
#                                               help="Reset main.py's own top-level checkpoints "
#                                                    "(see module docstring item 12 for scope)")
#    p.add_argument("--precomputed-te",         default=None,
#                                               help="Path to pre-computed te_features.npy — skips phase 4")
#    p.add_argument("--precomputed-te-weights", default=None,
#                                               help="Path to pre-computed te_weight.npy")
#    p.add_argument("--skip-to-phase",          type=int, default=None, choices=range(1, 7),
#                                               help="Skip straight to phase N (1-6), reusing "
#                                                    "prior phases' outputs already on disk")
#    return p.parse_args()
#
#
#def parse_reads_to_samples_tsv(reads_list: List[str], reads_dir: str, outdir: str) -> str:
#    samples_tsv = Path(outdir) / "samples.tsv"
#    Path(outdir).mkdir(parents=True, exist_ok=True)
#    with open(samples_tsv, "w") as f:
#        f.write("sample\tr1\tr2\n")
#        for s in reads_list:
#            parts = s.split(":")
#            if len(parts) == 2:
#                name, r1 = parts
#                r1_path = r1 if Path(r1).is_absolute() else str(Path(reads_dir) / r1)
#                f.write(f"{name}\t{r1_path}\t\n")
#            elif len(parts) == 3:
#                name, r1, r2 = parts
#                r1_path = r1 if Path(r1).is_absolute() else str(Path(reads_dir) / r1)
#                r2_path = r2 if Path(r2).is_absolute() else str(Path(reads_dir) / r2)
#                f.write(f"{name}\t{r1_path}\t{r2_path}\n")
#            else:
#                print(f"[WARN] Cannot parse --reads: '{s}' (expected NAME:R1 or NAME:R1:R2)")
#    return str(samples_tsv)
#
#
#def load_config(config_path: Optional[str] = None) -> Dict:
#    """Flat top-level merge — config.yaml is a flat namespace (see its own
#    header comment); do NOT introduce nesting here."""
#    config = DEFAULT_CONFIG.copy()
#    if config_path and Path(config_path).exists():
#        with open(config_path) as f:
#            user_config = yaml.safe_load(f) or {}
#        for k, v in user_config.items():
#            if k in config and isinstance(config[k], dict) and isinstance(v, dict):
#                config[k].update(v)
#            else:
#                config[k] = v
#    return config
#
#
#def _file_fingerprint(path) -> str:
#    """Size+mtime fingerprint -- NOT a content hash. Deliberately mirrors
#    the identical helper already used inside preprocessing.py/coverage.py/
#    tnf_gene.py/te_composition.py/encoder.py/clustering.py, so main.py's
#    own top-level checkpoint gates follow the same, already-established
#    convention as every module they wrap (see checkpoint.py's own design
#    note: fingerprint logic belongs to each caller, not to the shared
#    Checkpoint class -- main.py is a caller like any other module here)."""
#    if path is None:
#        return "None"
#    p = Path(str(path))
#    try:
#        st = p.stat()
#        return f"{p}:{st.st_size}:{int(st.st_mtime)}"
#    except FileNotFoundError:
#        return f"{p}:MISSING"
#
#
#def _fingerprint(*parts) -> str:
#    import hashlib
#    h = hashlib.sha256()
#    h.update(json.dumps(parts, sort_keys=True, default=str).encode())
#    return h.hexdigest()[:16]
#
#
#def _require_file(path, log, phase_desc: str, what: str) -> str:
#    if not path or not Path(path).exists():
#        log_error_flag(log, "MISSING_REQUIRED_FILE",
#                        f"Cannot {phase_desc}: {what} not found at '{path}'.")
#        sys.exit(1)
#    return str(path)
#
#
#def _read_json(path) -> dict:
#    with open(path) as f:
#        return json.load(f)
#
#
#def _sample_names_from_reads(reads_list: List[str]) -> List[str]:
#    names = []
#    for s in reads_list:
#        parts = s.split(":")
#        if len(parts) == 3:
#            names.append(parts[0])
#    return names
#
#
#def main():
#    args   = parse_args()
#    config = load_config(args.config)
#
#    if args.threads:
#        config["threads"] = args.threads
#
#    outdir = Path(args.outdir)
#    outdir.mkdir(parents=True, exist_ok=True)
#
#    setup_logger(str(outdir), args.log_level)
#    log  = get_logger("main")
#    ckpt = Checkpoint(str(outdir))
#
#    if args.reset:
#        log_warning_flag(log, "RESET_SCOPE",
#                          "--reset only clears main.py's own top-level checkpoint store "
#                          "(<outdir>/checkpoints/). Each phase's own internal checkpoints "
#                          "(<outdir>/<phase>/checkpoints/...) are untouched — a 'recomputed' "
#                          "phase may still reload its own cached sub-steps almost instantly. "
#                          "Delete that phase's output subdirectory for a true from-scratch rerun.")
#        ckpt.reset()
#
#    log.info("+" + "=" * 68 + "+")
#    log.info("|" + " " * 16 + "hyphaesbin PIPELINE v7 (RESTRUCTURED)" + " " * 16 + "|")
#    log.info("+" + "=" * 68 + "+")
#    log.info("")
#    log.info(f"Scaffold : {args.scaffold}")
#    log.info(f"Reads    : {len(args.reads)} sample(s)")
#    log.info(f"Outdir   : {outdir}")
#    log.info(f"Log file : {get_log_file()}")
#    log_threads(log, config["threads"], config["threads"], context="top-level requested")
#    log.info("")
#
#    # ── Startup dependency / backend diagnostics (report-only — see deps.py
#    # and item 10 of the module docstring; never blocks, never installs) ────
#    tools_needed = None
#    if str(config.get("te_mode", "fast")).lower() == "fast":
#        tools_needed = [t for t in deps.check_external_tools().keys() if t != "RepeatMasker"]
#    elif str(config.get("te_mode", "fast")).lower() == "efficient":
#        tools_needed = [t for t in deps.check_external_tools().keys() if t != "mmseqs"]
#    dep_report = deps.check_all(external_tools_needed=tools_needed)
#    for line in deps.format_report(dep_report).splitlines():
#        log.debug(line)
#    if dep_report["missing_required_python"]:
#        log_warning_flag(log, "MISSING_PYTHON_DEPS",
#                          f"{dep_report['missing_required_python']} — the relevant phase(s) "
#                          f"will fail loudly when reached; see the DEBUG log for the full report.")
#    if dep_report["missing_required_tools"]:
#        log_warning_flag(log, "MISSING_EXTERNAL_TOOLS",
#                          f"{dep_report['missing_required_tools']} — the relevant phase(s) "
#                          f"will fail loudly when reached; see the DEBUG log for the full report.")
#    log_device_backend(log, config.get("device", "auto"),
#                        deps.describe_backend_choice(config.get("device", "auto"), dep_report["backends"]),
#                        extra="encoder device — describe_backend_choice() prediction, not a resolution")
#    log_device_backend(log, config.get("clustering_backend", "auto"),
#                        deps.describe_backend_choice(config.get("clustering_backend", "auto"), dep_report["backends"]),
#                        extra="clustering backend — prediction, not a resolution")
#    log.info("")
#
#    skip_to = args.skip_to_phase or 1
#    t0_total = time.time()
#
#    preproc_dir = outdir / "preprocessing"
#    cov_dir     = outdir / "coverage"
#    tnf_dir     = outdir / "tnf_gene"
#    te_dir      = outdir / "te_composition"   # created BY run_te_branch itself
#    enc_dir     = outdir / "encoder"
#    clust_dir   = outdir / "clustering"
#
#    # =========================================================================
#    # PHASE 1: PREPROCESSING
#    # =========================================================================
#    STEP = "phase1_preprocessing"
#    final_dir = preproc_dir / "final"
#
#    if skip_to > 1:
#        log.info("Skipping Phase 1 (preprocessing) by request")
#        clean_fasta          = _require_file(final_dir / "final_clean.fasta", log, "skip Phase 1", "final_clean.fasta")
#        clean_fasta_unmasked = _require_file(final_dir / "final_clean_unmasked.fasta", log, "skip Phase 1", "final_clean_unmasked.fasta")
#        coverage_tsv         = _require_file(final_dir / "coverage_table.tsv", log, "skip Phase 1", "coverage_table.tsv")
#    elif ckpt.is_done(STEP):
#        log_checkpoint(log, STEP, reused=True)
#        meta = ckpt.load_metadata(STEP)
#        clean_fasta          = meta["final_clean"]
#        clean_fasta_unmasked = meta["final_clean_unmasked"]
#        coverage_tsv         = meta["coverage_table"]
#    else:
#        log_checkpoint(log, STEP, reused=False)
#        log_step_start(log, 1, "PREPROCESSING", 6)
#        t_phase = time.time()
#
#        if not args.reads:
#            log_error_flag(log, "NO_READS", "No reads provided. Use --reads NAME:R1 or NAME:R1:R2 (repeatable).")
#            sys.exit(1)
#
#        samples_tsv = parse_reads_to_samples_tsv(args.reads, args.reads_dir, str(outdir))
#        log.info(f"Samples table: {samples_tsv}")
#
#        try:
#            preprocess_outputs = run_preprocessing(
#                assembly_input = args.scaffold,
#                reads_dir      = args.reads_dir,
#                samples_tsv    = samples_tsv,
#                outdir         = str(preproc_dir),
#                threads        = config["threads"],
#                config         = config,
#            )
#        except SystemExit:
#            raise
#        except Exception as e:
#            log_error_flag(log, "PHASE1_FAILED", f"Preprocessing failed: {e}")
#            import traceback; log.debug(traceback.format_exc())
#            sys.exit(1)
#
#        clean_fasta          = preprocess_outputs["final_clean"]
#        clean_fasta_unmasked = preprocess_outputs["final_clean_unmasked"]
#        coverage_tsv         = preprocess_outputs["coverage_table"]
#
#        ckpt.mark_done(
#            STEP, metadata=preprocess_outputs,
#            output_files=[clean_fasta, clean_fasta_unmasked, coverage_tsv],
#            module="preprocessing.py", version="13-step redesign (no formal VERSION const)",
#        )
#        log_step_done(log, 1, "PREPROCESSING", time.time() - t_phase)
#
#    log_output_paths(log, {
#        "final_clean (masked)":    clean_fasta,
#        "final_clean_unmasked":    clean_fasta_unmasked,
#        "coverage_table":          coverage_tsv,
#    })
#
#    # Sample names for coverage's expected_samples cross-check (best-effort —
#    # only available when phase 1 actually ran with --reads this run).
#    sample_names = _sample_names_from_reads(args.reads) or None
#
#    # =========================================================================
#    # PHASE 2: COVERAGE FEATURES
#    # =========================================================================
#    STEP = "phase2_coverage"
#    cov_features_path = cov_dir / "coverage_features.npy"
#    cov_manifest_path = cov_dir / "manifest.json"
#
#    if skip_to > 2:
#        log.info("Skipping Phase 2 (coverage) by request")
#        cov_features_path = Path(_require_file(cov_features_path, log, "skip Phase 2", "coverage_features.npy"))
#        _require_file(cov_manifest_path, log, "skip Phase 2", "manifest.json")
#    elif ckpt.is_done(STEP):
#        log_checkpoint(log, STEP, reused=True)
#        meta = ckpt.load_metadata(STEP)
#        cov_features_path = Path(meta["cov_features_path"])
#    else:
#        log_checkpoint(log, STEP, reused=False)
#        log_step_start(log, 2, "COVERAGE FEATURES", 6)
#        t_phase = time.time()
#        n_cores = config.get("coverage_n_cores") or config["threads"]
#        log_threads(log, config["threads"], n_cores, context="coverage")
#
#        try:
#            coverage_features = run_coverage_features(
#                coverage_tsv            = coverage_tsv,
#                assembly_fasta          = clean_fasta,
#                outdir                  = str(cov_dir),
#                k                       = int(config.get("coverage_k", 200)),
#                n_cores                 = int(n_cores),
#                resume                  = True,
#                expected_samples        = sample_names,
#                strict_numeric_columns  = bool(config.get("coverage_strict_numeric_columns", True)),
#                outlier_cap_percentile  = config.get("coverage_outlier_cap_percentile", 99.5),
#                deterministic           = bool(config.get("coverage_deterministic", False)),
#                brute_force_max_n       = int(config.get("coverage_brute_force_max_n", 5000)),
#            )
#        except SystemExit:
#            raise
#        except Exception as e:
#            log_error_flag(log, "PHASE2_FAILED", f"Coverage features failed: {e}")
#            import traceback; log.debug(traceback.format_exc())
#            sys.exit(1)
#
#        ckpt.mark_done(
#            STEP,
#            metadata={"cov_features_path": str(cov_features_path), "shape": list(coverage_features.shape)},
#            output_files=[str(cov_features_path), str(cov_manifest_path)],
#            module="coverage.py", version="v9",
#        )
#        log_step_done(log, 2, "COVERAGE FEATURES", time.time() - t_phase)
#        log_contig_counts(log, total=coverage_features.shape[0])
#
#    cov_manifest = _read_json(cov_manifest_path) if cov_manifest_path.exists() else {}
#    n_samples = cov_manifest.get("n_samples")
#    log_output_paths(log, {"coverage_features": str(cov_features_path)})
#
#    # =========================================================================
#    # PHASE 3: TNF COMPOSITION (masked FASTA)
#    # =========================================================================
#    STEP = "phase3_tnf"
#    tnf_features_path = tnf_dir / "tnf_features.npy"
#    tnf_weights_path  = tnf_dir / "tnf_weights.npy"
#    tnf_include = bool(config.get("tnf_include", True))
#
#    if not tnf_include:
#        log.info("TNF composition: DISABLED (tnf_include=false) — skipping Phase 3")
#        tnf_features_path_str = None
#        tnf_weights_path_str  = None
#    elif skip_to > 3:
#        log.info("Skipping Phase 3 (TNF) by request")
#        tnf_features_path_str = _require_file(tnf_features_path, log, "skip Phase 3", "tnf_features.npy")
#        tnf_weights_path_str  = str(tnf_weights_path) if tnf_weights_path.exists() else None
#    elif ckpt.is_done(STEP):
#        log_checkpoint(log, STEP, reused=True)
#        meta = ckpt.load_metadata(STEP)
#        tnf_features_path_str = meta["features"]
#        tnf_weights_path_str  = meta.get("weights")
#    else:
#        log_checkpoint(log, STEP, reused=False)
#        if not TNF_AVAILABLE:
#            log_error_flag(log, "TNF_UNAVAILABLE", f"TNF module could not be imported: {_TNF_IMPORT_ERROR}")
#            sys.exit(1)
#        log_step_start(log, 3, "TNF COMPOSITION (whole-contig k4 frequency)", 6)
#        t_phase = time.time()
#
#        tnf_cfg = {k: config[k] for k in _TNF_CONFIG_KEYS if k in config}
#        try:
#            tnf_features, tnf_weights, tnf_contig_ids = run_tnf_wholecontig(
#                scaffold = clean_fasta,   # MASKED (see docstring item 3)
#                outdir   = str(tnf_dir),
#                config   = tnf_cfg,
#                resume   = True,
#            )
#        except SystemExit:
#            raise
#        except Exception as e:
#            log_error_flag(log, "PHASE3_FAILED", f"TNF composition failed: {e}")
#            import traceback; log.debug(traceback.format_exc())
#            sys.exit(1)
#
#        tnf_features_path_str = str(tnf_features_path)
#        tnf_weights_path_str  = str(tnf_weights_path)
#        ckpt.mark_done(
#            STEP,
#            metadata={"features": tnf_features_path_str, "weights": tnf_weights_path_str,
#                      "shape": list(tnf_features.shape)},
#            output_files=[tnf_features_path_str, tnf_weights_path_str,
#                          str(tnf_dir / "contig_ids.json"), str(tnf_dir / "tnf_feature_schema.json")],
#            module="tnf_gene.py", version="v2_streaming",
#        )
#        log_step_done(log, 3, "TNF COMPOSITION", time.time() - t_phase)
#        log_contig_counts(log, total=len(tnf_contig_ids))
#
#    if tnf_features_path_str:
#        log_output_paths(log, {"tnf_features": tnf_features_path_str})
#
#    # =========================================================================
#    # PHASE 4: TE COMPOSITION (unmasked FASTA)
#    # =========================================================================
#    STEP = "phase4_te"
#    te_features_path = te_dir / "te_features.npy"
#    te_weight_path   = te_dir / "te_weight.npy"
#    te_include = bool(config.get("te_include", True))
#
#    if args.precomputed_te:
#        log.info("Using pre-computed TE features — skipping Phase 4")
#        te_features_path_str = _require_file(args.precomputed_te, log, "use --precomputed-te", "te_features.npy")
#        te_weight_path_str   = args.precomputed_te_weights if args.precomputed_te_weights else None
#    elif not te_include:
#        log.info("TE composition: DISABLED (te_include=false) — skipping Phase 4")
#        te_features_path_str = None
#        te_weight_path_str   = None
#    elif skip_to > 4:
#        log.info("Skipping Phase 4 (TE) by request")
#        te_features_path_str = _require_file(te_features_path, log, "skip Phase 4", "te_features.npy")
#        te_weight_path_str   = str(te_weight_path) if te_weight_path.exists() else None
#    elif ckpt.is_done(STEP):
#        log_checkpoint(log, STEP, reused=True)
#        meta = ckpt.load_metadata(STEP)
#        te_features_path_str = meta["features"]
#        te_weight_path_str   = meta.get("weights")
#    else:
#        log_checkpoint(log, STEP, reused=False)
#        if not TE_AVAILABLE:
#            log_error_flag(log, "TE_UNAVAILABLE", f"TE module could not be imported: {_TE_IMPORT_ERROR}")
#            sys.exit(1)
#        funtedb = config.get("funTEdb") or config.get("funtedb") or config.get("FunTEDB") or ""
#        if not funtedb:
#            log_error_flag(log, "MISSING_FUNTEDB", "config['funTEdb'] is empty — TE composition requires a TE database FASTA.")
#            sys.exit(1)
#
#        log_step_start(log, 4, "TE COMPOSITION", 6)
#        t_phase = time.time()
#        try:
#            # run_te_branch appends "te_composition" onto whatever outdir it's
#            # given (see te_composition.py) -- pass the pipeline's TOP-LEVEL
#            # outdir here, not te_dir, or the path doubles up.
#            te_features, te_weight, te_bed_path = run_te_branch(
#                scaffold = clean_fasta_unmasked,   # UNMASKED (see docstring item 3)
#                config   = config,
#                outdir   = str(outdir),
#            )
#        except SystemExit:
#            raise
#        except Exception as e:
#            log_error_flag(log, "PHASE4_FAILED", f"TE composition failed: {e}")
#            import traceback; log.debug(traceback.format_exc())
#            sys.exit(1)
#
#        te_features_path_str = str(te_features_path)
#        te_weight_path_str   = str(te_weight_path)
#        ckpt.mark_done(
#            STEP,
#            metadata={"features": te_features_path_str, "weights": te_weight_path_str,
#                      "te_bed": te_bed_path, "shape": list(te_features.shape)},
#            output_files=[te_features_path_str, te_weight_path_str,
#                          str(te_dir / "contig_ids.json"), str(te_dir / "schema.json")],
#            module="te_composition.py", version="V33_V22_5D",
#        )
#        log_step_done(log, 4, "TE COMPOSITION", time.time() - t_phase)
#
#    if te_features_path_str:
#        log_output_paths(log, {"te_features": te_features_path_str})
#
#    # =========================================================================
#    # PHASE 5: ENCODER
#    # =========================================================================
#    STEP = "phase5_encoder"
#    final_latent_path    = enc_dir / "final_latent.npy"
#    encoder_manifest_path = enc_dir / "encoder_manifest.json"
#    cov_include = bool(config.get("cov_include", True))
#
#    if not (tnf_include or te_include or cov_include):
#        log_error_flag(log, "NO_MODALITIES",
#                        "tnf_include, te_include, and cov_include are all False — at least one "
#                        "modality must be enabled to run the encoder.")
#        sys.exit(1)
#
#    if skip_to > 5:
#        log.info("Skipping Phase 5 (encoder) by request")
#        final_latent_path_str = _require_file(final_latent_path, log, "skip Phase 5", "final_latent.npy")
#    elif ckpt.is_done(STEP):
#        log_checkpoint(log, STEP, reused=True)
#        meta = ckpt.load_metadata(STEP)
#        final_latent_path_str = meta["final_latent"]
#    else:
#        log_checkpoint(log, STEP, reused=False)
#        if not ENCODER_AVAILABLE:
#            log_error_flag(log, "ENCODER_UNAVAILABLE", f"Encoder module could not be imported: {_ENCODER_IMPORT_ERROR}")
#            sys.exit(1)
#        log_step_start(log, 5, "ENCODER TRAINING", 6)
#        t_phase = time.time()
#
#        encoder_cfg = EncoderConfig.from_dict(config)   # safely filters unknown keys
#        log.info(f"Active modalities requested: tnf={tnf_include} te={te_include} cov={cov_include} "
#                 f"(allow_missing_modalities={encoder_cfg.allow_missing_modalities})")
#
#        try:
#            final_latent_path_str = run_encoder(
#                outdir             = str(enc_dir),
#                tnf_features_path  = tnf_features_path_str if tnf_include else None,
#                cov_features_path  = str(cov_features_path) if cov_include else None,
#                te_features_path   = te_features_path_str if te_include else None,
#                tnf_weights_path   = tnf_weights_path_str,
#                te_weights_path    = te_weight_path_str,
#                config             = encoder_cfg,
#                n_samples          = int(n_samples) if n_samples is not None else None,
#            )
#        except SystemExit:
#            raise
#        except Exception as e:
#            log_error_flag(log, "PHASE5_FAILED", f"Encoder training failed: {e}")
#            import traceback; log.debug(traceback.format_exc())
#            sys.exit(1)
#
#        ckpt.mark_done(
#            STEP, metadata={"final_latent": final_latent_path_str},
#            output_files=[final_latent_path_str, str(encoder_manifest_path)],
#            module="encoder.py", version="v8.1",
#        )
#        log_step_done(log, 5, "ENCODER TRAINING", time.time() - t_phase)
#
#        if encoder_manifest_path.exists():
#            enc_manifest = _read_json(encoder_manifest_path)
#            log_device_backend(log, config.get("device", "auto"), enc_manifest.get("device", "?"),
#                                extra="encoder actual (from encoder_manifest.json)")
#
#    log_output_paths(log, {"final_latent": final_latent_path_str,
#                            "encoder_manifest": str(encoder_manifest_path)})
#
#    # =========================================================================
#    # PHASE 6: CLUSTERING
#    # =========================================================================
#    STEP = "phase6_clustering"
#    clustering_stats_path = clust_dir / "clustering_stats.json"
#
#    if not CLUSTERING_AVAILABLE:
#        log_error_flag(log, "CLUSTERING_UNAVAILABLE", f"Clustering module could not be imported: {_CLUSTERING_IMPORT_ERROR}")
#        sys.exit(1)
#
#    clustering_cfg = ClusteringConfig.from_dict(config)  # safely filters unknown keys
#    contig_ids_path_for_clustering = str(cov_dir / "contig_ids.txt")
#
#    # main.py's own checkpoint gate must fingerprint clustering's real inputs
#    # AND its resolved config, not just check "did phase6_clustering finish
#    # once, and does cluster_summary.tsv still exist". An existence-only gate
#    # would happily reuse a stale cluster_summary.tsv after final_latent.npy
#    # was retrained, the coverage/valid-mask array changed, or any
#    # ClusteringConfig value (backend, seed, HDBSCAN/dedup/EukCC settings,
#    # ...) was edited in config.yaml between runs -- clustering.py's own
#    # internal fingerprinting never gets a chance to catch that, because the
#    # whole point of this gate is to decide whether to call run_clustering()
#    # at all. This mirrors the exact `_file_fingerprint`/`_fingerprint`
#    # convention every other module here already uses (see the helpers
#    # above), applied at main.py's own orchestration level.
#    fp6 = _fingerprint(
#        _file_fingerprint(final_latent_path_str),
#        _file_fingerprint(str(encoder_manifest_path)),
#        _file_fingerprint(contig_ids_path_for_clustering),
#        _file_fingerprint(clean_fasta_unmasked),
#        _file_fingerprint(str(cov_features_path)),
#        _file_fingerprint(tnf_weights_path_str),
#        _file_fingerprint(te_weight_path_str),
#        asdict(clustering_cfg),
#    )
#
#    prev_done = ckpt.is_done(STEP)
#    prev_meta = ckpt.load_metadata(STEP) if prev_done else {}
#    stale = (not prev_done) or (prev_meta.get("_fp") != fp6)
#
#    if not stale:
#        log_checkpoint(log, STEP, reused=True)
#        cluster_summary_path = prev_meta["cluster_summary"]
#    else:
#        log_checkpoint(log, STEP, reused=False,
#                        reason="inputs/config changed since last run" if prev_done else "")
#        log_step_start(log, 6, "CLUSTERING + REFINEMENT", 6)
#        t_phase = time.time()
#
#        # NOTE on alignment_bam_paths (EukCC split-bin merging): run_clustering()
#        # accepts an OPTIONAL alignment_bam_paths list -- paired-end BAM
#        # alignments against the BIN FASTAs -- to drive EukCC's real --links
#        # merge workflow (clustering.py FIX 13). No stage in the current
#        # six-module pipeline (preprocessing/coverage/TNF/TE/encoder/
#        # clustering) produces that BAM: preprocessing step 9 maps reads
#        # against the WHOLE assembly (not per-bin FASTAs) and deletes those
#        # BAMs as intermediate files right after step 12
#        # (cleanup_intermediate()), well before clustering ever runs -- and
#        # even if kept, they'd be the wrong reference for binlinks.py, which
#        # needs alignments against the bins themselves. clustering.py's own
#        # docstring says this outright: "This requires paired-end read
#        # alignment (BAM) as an input, which no earlier stage of this
#        # pipeline currently produces or was described as producing."
#        # Passing `alignment_bam_paths=None` here is therefore not an
#        # oversight -- it's the honest reflection of that gap. EukCC's
#        # per-bin QUALITY assessment (run_eukcc(), step 9 inside
#        # clustering.py) still runs and is unaffected; only the OPTIONAL
#        # split-bin MERGE step is skipped, exactly as clustering.py itself
#        # reports via clustering_stats.json's "eukcc_merge_status":
#        # "skipped_no_bam_alignment". Adding a new mapping-against-bins
#        # stage to actually produce this BAM would be a new pipeline
#        # capability, not a main.py wiring fix -- out of scope here unless
#        # you want that stage added.
#        try:
#            cluster_summary_path = run_clustering(
#                final_latent_path      = final_latent_path_str,
#                encoder_manifest_path  = str(encoder_manifest_path),
#                contig_ids_path        = contig_ids_path_for_clustering,
#                fasta_path             = clean_fasta_unmasked,   # UNMASKED — final bin output (see docstring item 3)
#                cov_features_path      = str(cov_features_path),
#                outdir                 = str(clust_dir),
#                tnf_weights_path       = tnf_weights_path_str,
#                te_weights_path        = te_weight_path_str,
#                alignment_bam_paths    = None,   # see note above — no source of this exists yet
#                config                 = clustering_cfg,
#            )
#        except SystemExit:
#            raise
#        except Exception as e:
#            log_error_flag(log, "PHASE6_FAILED", f"Clustering failed: {e}")
#            import traceback; log.debug(traceback.format_exc())
#            sys.exit(1)
#
#        ckpt.mark_done(
#            STEP, metadata={"cluster_summary": cluster_summary_path, "_fp": fp6},
#            output_files=[cluster_summary_path],
#            module="clustering.py", version="v7.2",
#        )
#        log_step_done(log, 6, "CLUSTERING + REFINEMENT", time.time() - t_phase)
#
#        if clustering_stats_path.exists():
#            stats = _read_json(clustering_stats_path)
#            backend_info = stats.get("backend", {})
#            log_device_backend(log, backend_info.get("requested", clustering_cfg.clustering_backend),
#                                backend_info.get("actual", "?"),
#                                extra="clustering actual (from clustering_stats.json)")
#
#    nondedup_dir = clust_dir / clustering_cfg.clusters_non_deduplicated_dir
#    dedup_dir    = clust_dir / clustering_cfg.clusters_deduplicated_dir
#
#    # =========================================================================
#    # COMPLETION SUMMARY
#    # =========================================================================
#    t_total = time.time() - t0_total
#    log.info("")
#    log.info("+" + "=" * 68 + "+")
#    log.info("|" + " " * 24 + "PIPELINE COMPLETE" + " " * 26 + "|")
#    log.info("+" + "=" * 68 + "+")
#    log.info(f"Total time : {t_total/3600:.2f} hours")
#    log.info("")
#
#    output_paths = {
#        "final_clean (masked)":       clean_fasta,
#        "final_clean_unmasked":       clean_fasta_unmasked,
#        "coverage_features":          str(cov_features_path),
#        "coverage_manifest":          str(cov_manifest_path),
#        "final_latent":               final_latent_path_str,
#        "encoder_manifest":           str(encoder_manifest_path),
#        "cluster_summary":            cluster_summary_path,
#        "clusters_non_deduplicated":  str(nondedup_dir),
#        "clusters_deduplicated":      str(dedup_dir),
#        "clustering_stats":           str(clustering_stats_path),
#        "log_file":                   get_log_file(),
#    }
#    if tnf_features_path_str:
#        output_paths["tnf_features"] = tnf_features_path_str
#    if te_features_path_str:
#        output_paths["te_features"] = te_features_path_str
#    log_output_paths(log, output_paths)
#
#    print("\nOUTPUTS:")
#    for name, path in output_paths.items():
#        print(f"{name.upper()}={path}")
#
#
#if __name__ == "__main__":
#    main()
