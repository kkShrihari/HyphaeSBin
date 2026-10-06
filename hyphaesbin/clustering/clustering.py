"""HyphaeS clustering v8.11: optional LATE rescue (noise_rescue_stage="late"): noise contigs are rescued
AFTER sub-clustering/merging, into the final split bins. Early rescue added thousands of short noisy
contigs into chimeric HDBSCAN clusters before they were split, which changed the leaf split
(Leptosphaeria 90.5% -> 80%). Pruned contigs are excluded from the late rescue.
v8.10: optional kNN-majority rescue candidate (noise_rescue_candidate="knn").
Nearest-CENTROID candidate failed for large diffuse bins (Phytophthora noise: 100% Phytophthora
neighbours, but nearest centroid another bin in 99% of cases).
v8.9: optional per-sample standardisation of coverage in ALL refinement
statistics (prune, misfit, merge, reassign, best-home). Stored coverage features are compressed
(0-3.6, median 0), which made the shape statistic blind to real species differences.
v8.8: best-home decided by TNF NEIGHBOURS (share + density ratio) instead of
radius-normalised centroid distance, which let large diffuse bins attract pieces (v8.7A chimera).
v8.7: same modules as v8.6; defaults = INTRINSIC preset (v8.3 behaviour +
best-home; GC guards, TNF-consensus and EukCC marker merge are OFF unless enabled in config).
v8.6: best-home placement of leaf pieces + EukCC marker-checked merge of
bins whose merge was refused only by the GC guard.
v8.5: GC-OVERLAP merge guard (replaces the median-GC guard, which split
GC-heterogeneous genomes) and coverage-checked TNF-consensus reassignment.
v8.4: v8.3 + GC-difference merge guard and contig-level TNF-kNN
consensus reassignment. v8.3: misfit guard inside sub-clustering (stops a split-off
foreign genome being merged straight back). v8.2: prune-before-merge, fit-test and TNF-mixing merge
guards, reassignment of pruned contigs. (v8.1: TNF-primary merging, same-genome bin
merge, coverage-shape pruning, short-contig recruitment, robust EukCC single.)
Derived from the supplied v7.8 module. No dataset-optimality guarantee.
Public run_clustering API preserved. See clustering_RELEASE_NOTES.md.
"""
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import collections
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

log = logging.getLogger("hyphaesbin.clustering")

VERSION = "v8.11"
# HDBSCAN stage is unchanged from v7.9; its cache fingerprint keeps the old tag
# so existing HDBSCAN checkpoints are reused (only rescue/sub-clustering rerun).
HDBSCAN_CORE_VERSION = "v7.9"


try:
    import hdbscan
except ImportError as e:
    raise ImportError(
        "hdbscan is not installed. Install it explicitly before running this module: "
        "pip install hdbscan  (or: conda install -c conda-forge hdbscan)"
    ) from e

try:
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    silhouette_score = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from hyphaesbin.utils.checkpoint import Checkpoint
except Exception:
    Checkpoint = None


# =============================================================================
# CHECKPOINT FINGERPRINTING -- same pattern as preprocessing.py / coverage.py /
# tnf_gene.py / te_composition.py / encoder.py.
# =============================================================================

def _file_fingerprint(path) -> str:
    """Size+mtime fingerprint, not a content hash -- same acknowledged
    limitation as the identical helper in every other pipeline module."""
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
        log.warning(f"{step}: checkpoint exists but inputs/config/version changed since it "
                    f"ran -- ignoring stale checkpoint and re-running.")
        return None
    missing = [p for p in output_paths if p and not Path(p).exists()]
    if missing:
        log.warning(f"{step}: checkpoint fingerprint matches, but {len(missing)} referenced "
                    f"output(s) no longer exist on disk ({missing[:3]}"
                    f"{', ...' if len(missing) > 3 else ''}) -- treating as a cache miss.")
        return None
    return prev


def _sha256_id_order(contig_ids: List[str]) -> str:
    """Identical formula to encoder.py / tnf_gene.py / te_composition.py /
    coverage.py -- hashes are directly comparable across all of them."""
    return hashlib.sha256("\n".join(contig_ids).encode()).hexdigest()[:16]


# =============================================================================
# CLUSTERING CONFIG
# =============================================================================

@dataclass
class ClusteringConfig:
    """
    All clustering hyperparameters in one place. Load from config.yaml --
    no code changes needed for any dataset.

    Auto-scaling rules (applied at runtime, via resolve()):
      hdbscan_min_cluster_size = None -> max(5, min(50, n_clust // 500))
      hdbscan_min_samples      = None -> max(2, min(20, n_clust // 100))

    The HDBSCAN parameters themselves, and the decision to run exactly one
    fixed configuration (no search over multiple values), are unchanged
    from v6 -- see module docstring FIX list.
    """

    # --- Contig filtering ---
    cluster_min_contig_len:   int   = 2000     # minimum contig length for clustering
    anchor_min_length:        int   = 5000     # anchor: minimum length (diagnostic only)
    anchor_min_tnf_weight:    float = 0.8      # anchor: minimum TNF weight
    anchor_max_cov_var:       float = 2.0      # anchor: maximum coverage variance

    # --- HDBSCAN (core, fixed -- no search) ---
    hdbscan_min_cluster_size: Optional[int] = 15
    hdbscan_min_samples:      Optional[int] = 5
    hdbscan_epsilon:          float         = 0.15
    hdbscan_method:           str           = "eom"   # FIX 21: default changed from "leaf".
                                                       # Real runs on this pipeline's data
                                                       # showed "leaf" fragmenting the latent
                                                       # space so aggressively that ~100% of
                                                       # eligible contigs ended up unbinned;
                                                       # "eom" recovered the large majority
                                                       # with only a modest silhouette
                                                       # trade-off. Override explicitly if a
                                                       # future dataset shows the opposite.
    hdbscan_algorithm:        str           = "best"  # FIX 25 (v7.5): NEW field. "best" |
                                                       # "prims_kdtree" | "prims_balltree" |
                                                       # "boruvka_kdtree" | "boruvka_balltree" |
                                                       # "generic". Passed straight through to
                                                       # hdbscan.HDBSCAN(algorithm=...). "best"
                                                       # (HDBSCAN's own automatic choice based on
                                                       # data size/dimensionality/metric) is kept
                                                       # as the default deliberately -- do NOT
                                                       # force a specific algorithm without first
                                                       # benchmarking it against "best" on this
                                                       # exact dataset (runtime, memory, cluster
                                                       # count, noise fraction, AND label
                                                       # agreement) via cfg.hdbscan_algorithm.
                                                       # Different algorithms are not guaranteed
                                                       # to produce identical labels/noise
                                                       # assignment even on identical input.
    hdbscan_soft_prob_min:    float         = 0.65    # contigs below this -> noise

    # --- Noise assignment ---
    noise_max_distance:       float = 0.65   # euclidean in L2-normalized space [0,2]
    noise_rescue_min_margin:  float = 0.05    # FIX 42 (v7.8): NEW field. Requires the
                                              # SECOND-nearest bin center to be at least
                                              # this much farther away than the nearest
                                              # one before a noise contig is rescued --
                                              # an ambiguity guard. Previously
                                              # assign_noise_contigs() had no such check:
                                              # a contig equidistant between two bins'
                                              # centers (e.g. one it was originally
                                              # HDBSCAN-assigned to, now rejected by the
                                              # soft-probability filter, and a genuinely
                                              # different bin) could be rescued into
                                              # whichever center happened to be
                                              # infinitesimally closer, with no signal
                                              # that the choice was ambiguous. Default
                                              # 0.0 preserves the exact prior behavior
                                              # (no margin required) -- set > 0 (e.g.
                                              # 0.05-0.1) to require genuine separation
                                              # before rescuing, leaving ambiguous
                                              # contigs unbinned instead of risking
                                              # contamination of the wrong bin.

    noise_rescue_enabled: bool = True
    noise_rescue_stage: str = "early"          # v8.11: "early" (right after HDBSCAN, v7.8+) | "late"
                                               # (after sub-clustering/prune/merge/reassign, before recruit)
    noise_rescue_candidate: str = "centroid"   # v8.10: "centroid" (v7.8+ behaviour) | "knn": candidate =
                                               # majority bin among nearby trusted neighbours; the same
                                               # support/agreement thresholds still apply
    noise_local_agreement_enabled: bool = True
    noise_local_k: int = 10
    noise_local_min_support: int = 5
    noise_local_agreement: float = 0.9
    noise_reference_prob_min: float = 0.65
    noise_neighbor_max_distance: float = 0.65
    noise_query_batch_size: int = 4096

    # --- Iterative refinement ---
    max_iter:                 int   = 0
    convergence_threshold:    float = 0.01
    refinement_max_distance:  float = 0.5    # FIX 22: cap on medoid-reassignment distance,
                                              # same units/space as noise_max_distance. A
                                              # contig is only moved to a new bin's medoid
                                              # during refinement if it's within this
                                              # distance -- prevents unbounded drift pulling
                                              # nearby-but-genuinely-distinct clusters (e.g.
                                              # closely related species) together over
                                              # repeated iterations.

    # --- Bin filtering ---
    min_contigs_per_bin:      int   = 5
    min_bin_length_bp:        int   = 500_000   # 500kb -- STUDY-DESIGN POLICY, not a universal
                                                 # rule (v7.6 note). Reasonable for typical
                                                 # fungal genome sizes, but will discard
                                                 # genuinely small or highly fragmented
                                                 # eukaryotic genomes (reduced/parasitic
                                                 # genomes, low-coverage assemblies of real
                                                 # organisms). Override for your dataset, and
                                                 # report the exclusion effect (how much total
                                                 # sequence / how many candidate bins this
                                                 # threshold removes) rather than treating this
                                                 # default as ground truth.
    min_bin_n50_bp:           int   = 1_000     # minimum N50 per bin

    # --- Deduplication ---
    # Conservative pairwise deduplication is used: a bin is removed only when
    # it is directly redundant with the selected survivor. This avoids the
    # biologically unsafe transitive rule A~B and B~C => discard A/B/C when A
    # and C are not directly redundant. The longest directly redundant bin is
    # retained as a reproducible completeness proxy; EukCC quality is assessed
    # afterward and is not available at this stage.
    skani_ani_threshold:      float = 99.0
    skani_min_af:             float = 80.0  # FIX 26 (v7.5): corrected from 0.1. skani's own
                                             # --min-af default is 15 -- a 0-100 percentage
                                             # scale, confirmed directly from `skani dist
                                             # --help`'s own output ("[default: 15]"). The
                                             # prior 0.1 therefore meant 0.1% required aligned
                                             # fraction -- ~150x more permissive than skani's
                                             # own default -- letting bin pairs with only
                                             # trivial/coincidental overlap (e.g. one shared
                                             # conserved region) be flagged as redundant
                                             # duplicates and merged/discarded, silently
                                             # pulling foreign sequence into surviving bins.
                                             # 80.0 requires 80% reciprocal aligned fraction,
                                             # well above skani's own default, before two bins
                                             # are treated as true duplicates.

    # --- Coverage feature format (must match coverage.py's real layout) ---
    n_cov_meta_cols:          int   = 4    # [valid_mask, n_samples_present, mean_dist, std_dist]
    allow_missing_valid_mask: bool  = False  # fail loud unless explicitly overridden (FIX 4)

    # --- Diagnostics (FIX 9) ---
    compute_silhouette:       bool  = True
    silhouette_max_n:         int   = 20_000   # subsample cap; silhouette is ~O(n^2)

    # --- QC tools ---
    run_eukcc:                bool  = True
    eukcc_db:                 str   = ""   # REQUIRED if run_eukcc=True â€” was a
                                            # hardcoded personal path; must come
                                            # from config now (see run_eukcc()/
                                            # run_eukcc_merge()'s empty-db guard,
                                            # which skips cleanly instead of
                                            # launching `eukcc --db ""`).
    eukcc_conda_env:          str   = "eukcc"
    eukcc_scratch_dir:        str   = "/tmp"   # v8.1: `eukcc single` runs here (Bus error seen on
                                               # /DATA_LUN); results are copied back into outdir
    eukcc_min_eukaryote_fraction: float = 0.5  # FIX 32 (v7.6): NEW field. A bin is only
                                                # submitted to EukCC if the fraction of its
                                                # contigs classified "eukaryote" (via the
                                                # optional contig_classification map passed to
                                                # run_eukcc()) is >= this threshold. EukCC's
                                                # marker sets are eukaryote-only, so scoring a
                                                # predominantly bacterial/archaeal bin with it
                                                # produces a biologically meaningless result,
                                                # not just an inaccurate one. Set to 0 to
                                                # disable this filter even when a classification
                                                # map is supplied. Has no effect at all when no
                                                # contig_classification is passed to run_eukcc()
                                                # (this module does not compute classification
                                                # itself -- see run_eukcc()'s docstring).
    eukcc_single_per_bin:     bool  = True   # FIX 44 (v7.8): NEW field. When True,
                                                # run_eukcc() calls `eukcc single` once per
                                                # bin instead of `eukcc folder` on the whole
                                                # directory. This guarantees each bin's
                                                # reported completeness/contamination
                                                # reflects that bin's own FASTA content
                                                # alone -- `eukcc folder` was observed to
                                                # perform its own internal merge-search even
                                                # with no --links file supplied (see the
                                                # note above run_eukcc()'s `eukcc folder`
                                                # command), so its numbers cannot be assumed
                                                # unmerged. Slower (no shared work across
                                                # bins) but unambiguous -- use this when a
                                                # guaranteed-per-bin-attributable number is
                                                # required (e.g. final benchmark reporting).
                                                # Default False preserves existing behavior.

    # --- EukCC bin merging (FIX 13 -- EukCC's OWN --links mechanism) ---
    eukcc_merge_enabled:      bool  = False
    eukcc_merge_n_combine:    int   = 1     # EukCC's own flag; >1 is experimental per EukCC docs
    eukcc_merge_ani:          int   = 99    # binlinks.py --ANI
    eukcc_merge_within:       int   = 1500  # binlinks.py --within

    # --- Backend / reproducibility (FIX 10) ---
    clustering_backend:       str   = "cpu"    # explicit CPU baseline; GPU remains opt-in
    clustering_threads:       int   = 8
    clustering_seed:          int   = 42

    # --- Runtime ---
    resume:                   bool  = True     # use checkpoints

    # --- Output directory names (FIX 20, v7.2) ---
    clusters_non_deduplicated_dir: str = "clusters_non_deduplicated"
    clusters_deduplicated_dir:     str = "clusters_deduplicated"

    # --- Sub-clustering of large bins (v8.0) ---
    # Label-free chimera splitting: large eom bins are re-clustered with leaf
    # HDBSCAN on TNF+COV(+TE) latents, then the leaf pieces are re-merged by
    # their multi-sample coverage profile. Fragments of one genome share one
    # abundance profile and merge back; different genomes stay separate.
    # No EukCC and no reference/ground truth are used.
    subcluster_enabled:          bool  = True
    subcluster_min_bin_contigs:  int   = 2000     # only bins at least this large are examined
    subcluster_method:           str   = "leaf"
    subcluster_min_cluster_size: int   = 15
    subcluster_min_samples:      int   = 5
    subcluster_cov_merge_tau:    float = 2.0      # v8.1: now only a LOOSE guard -- pieces are merged
                                                  # when TNF says same genome AND coverage distance < this
    subcluster_tnf_sep_max:      float = 1.0      # v8.1: PRIMARY merge rule. TNF centroid distance /
                                                  # mean TNF radius of the two pieces. <1 = clouds overlap
                                                  # (same genome); >1 = separate genomes.
    subcluster_noise_min_agreement: float = 0.6
    subcluster_max_misfit:       float = 0.25     # v8.3: same fit-test guard as bin merge, applied when
                                                  # leaf pieces are re-merged (z = binmerge_misfit_z)
    # v8.4: refuse a merge (sub-clustering AND bin merge) when the two groups' median
    # contig GC differs by more than this. Set <= 0 to disable. Calibrate first:
    # genomes with AT-rich isochores can have fragments that differ in GC.
    merge_max_gc_diff:           float = 0.0    # v8.4 median-GC guard -- OFF (split GC-heterogeneous genomes)
    # v8.5 GC-OVERLAP guard: refuse a merge when more than merge_gc_outside_max of the
    # SMALLER group's contigs lie outside the LARGER group's own GC range
    # [q_lo, q_hi]. A GC-heterogeneous genome has a wide range that covers its own
    # fragments; a foreign genome falls outside it. <= 0 disables.
    merge_gc_outside_max:        float = 0.0
    merge_gc_q_lo:               float = 0.05
    merge_gc_q_hi:               float = 0.95   # v8.1: leaf-noise contig joins a part only if >= this
                                                  # fraction of its kNN agree; otherwise left unbinned
    subcluster_min_part_bp:      int   = 500_000  # a split part must be at least this long
    subcluster_noise_k:          int   = 10       # kNN vote for leaf-noise contigs inside the bin
    subcluster_te_mode:          str   = "auto"   # "auto" | "on" | "off"
    subcluster_te_min_weight:    float = 0.3      # auto: drop TE if bin median te_weight below this
    subcluster_te_min_agreement: float = 0.10     # auto: drop TE if TE-kNN vs TNF-kNN Jaccard below this

    # --- Same-genome bin merge (v8.1) ---
    # Two bins are merged when their TNF clouds overlap (tnf_sep < binmerge_tnf_sep_max)
    # AND their coverage profiles agree (cov distance < binmerge_cov_tau). Rejoins
    # fragments such as a genome split into a big bin plus small side bins.
    binmerge_enabled:            bool  = True
    binmerge_tnf_sep_max:        float = 1.0
    binmerge_cov_tau:            float = 2.0
    binmerge_min_contigs:        int   = 10
    # v8.2 merge guards (all label-free). A merge of a smaller group into a larger
    # one is refused when:
    #   misfit  = fraction of the smaller group's contigs whose coverage SHAPE is an
    #             outlier (z > binmerge_misfit_z) under the larger group's model
    #             > binmerge_max_misfit            (blocks mixed/foreign bins), or
    #   tnf_mix = observed / expected share of cross-group TNF nearest neighbours
    #             < binmerge_min_tnf_mix           (blocks genomes that sit apart in TNF,
    #             even when their coverage looks alike, e.g. Tuber vs Phytophthora)
    binmerge_max_misfit:         float = 0.25
    binmerge_misfit_z:           float = 4.0
    binmerge_min_tnf_mix:        float = 0.10

    # --- Reassignment of pruned contigs (v8.2) ---
    # Contigs removed by coverage-shape pruning are moved to the bin they actually fit:
    # candidate bins come from the TNF kNN vote (>= reassign_min_votes of reassign_k),
    # the one with the lowest coverage-shape z is chosen, and only accepted if
    # z <= reassign_max_z. Otherwise the contig stays unbinned.
    reassign_enabled:            bool  = True
    reassign_k:                  int   = 20
    reassign_min_votes:          int   = 5
    reassign_max_z:              float = 3.0

    # --- Contig-level TNF-kNN consensus reassignment (v8.4) ---
    # After merging, every binned contig is checked: if >= tnf_consensus_min of its
    # tnf_consensus_k nearest binned neighbours in the TNF latent belong to ONE other bin,
    # it moves there. Moves contaminant contigs of a genome that has its own bin
    # (e.g. Melampsora contigs sitting in the Leptosphaeria/Cenococcum bins).
    tnf_consensus_enabled:       bool  = False   # v8.7: off by default (scattered contigs in v8.4)
    tnf_consensus_k:             int   = 15
    tnf_consensus_min:           float = 0.8
    tnf_consensus_iters:         int   = 5      # passes; stops early when nothing moves
    tnf_consensus_max_z:         float = 3.0    # v8.5: move only if the contig's coverage SHAPE fits the
                                                # target bin (z <= this); stops big-bin TNF pull

    # --- Best-home placement of leaf pieces (v8.6) ---
    # During sub-clustering each leaf piece is compared with its SIBLING pieces and with
    # every OTHER existing bin. If another bin is closer in TNF (tnf_sep) than any
    # sibling, and passes the same coverage/misfit checks, the piece moves there.
    subcluster_best_home:        bool  = True
    # v8.9: z-score each coverage sample column (after log1p) before any refinement statistic.
    # Thresholds (cov_tau, misfit_z, prune_shape_z, reassign_max_z) must be re-calibrated.
    refine_cov_standardize:      bool  = False
    best_home_k:                 int   = 15     # v8.8: TNF neighbours per sampled contig
    best_home_min_share:         float = 0.7    # v8.8: >= this share of outside neighbours in ONE bin
    best_home_max_ratio:         float = 1.5    # v8.8: outside-neighbour distance / own-piece distance
    best_home_sample:            int   = 300    # v8.8: contigs sampled per piece

    # --- EukCC marker-checked merge (v8.6) ---
    # Bin merges refused ONLY by a GC guard are re-examined with marker genes: EukCC
    # scores A, B and A+B. The merge is accepted when completeness rises by
    # >= marker_min_gain AND contamination rises by <= marker_max_cont_increase.
    # (Fragments of one genome complement each other; two genomes duplicate markers.)
    marker_merge_enabled:        bool  = False
    marker_min_bp:               int   = 500_000
    marker_max_checks:           int   = 40
    marker_min_gain:             float = 5.0
    marker_max_cont_increase:    float = 2.0
    # --- Coverage-shape outlier pruning (v8.1) ---
    # Per bin, each contig's log-coverage profile is centred on its own mean (so
    # collapsed repeats, which only shift the level, are NOT flagged); contigs whose
    # SHAPE across samples deviates > prune_shape_z robust SDs from the bin median
    # are unbinned. Skipped for a bin if it would remove > prune_max_frac.
    prune_enabled:               bool  = True
    prune_min_bin_contigs:       int   = 30
    prune_shape_z:               float = 4.0
    prune_max_frac:              float = 0.10

    # --- Short-contig recruitment (v8.1) ---
    # Contigs with recruit_min_len <= length < cluster_min_contig_len (never seen by
    # HDBSCAN) are assigned to a bin only if >= recruit_min_support of their
    # recruit_k nearest binned contigs lie within recruit_max_dist AND
    # >= recruit_agreement of those agree on one bin.
    recruit_enabled:             bool  = True
    recruit_min_len:             int   = 1000
    recruit_k:                   int   = 15
    recruit_min_support:         int   = 10
    recruit_agreement:           float = 0.9
    recruit_max_dist:            float = 0.5

    @classmethod
    def from_yaml(cls, path: str) -> "ClusteringConfig":
        if not HAS_YAML:
            raise ImportError("PyYAML not installed: pip install pyyaml")
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cd = d.get("clustering", d)
        return cls(**{k: v for k, v in cd.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "ClusteringConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def resolve(self, n_clust: int) -> "ClusteringConfig":
        """Fill auto-scaling params based on actual contig count, and
        validate the new backend/reproducibility fields."""
        if self.clustering_backend not in ("auto", "cpu", "gpu"):
            raise ValueError(
                f"clustering_backend must be 'auto'/'cpu'/'gpu', got {self.clustering_backend!r}"
            )
        if self.clustering_threads < 1:
            raise ValueError(f"clustering_threads must be >= 1, got {self.clustering_threads}")

        cfg = ClusteringConfig(**asdict(self))
        if cfg.max_iter < 0:
            raise ValueError("max_iter must be nonnegative")
        if cfg.noise_local_k < 1 or not 1 <= cfg.noise_local_min_support <= cfg.noise_local_k:
            raise ValueError("Require 1 <= noise_local_min_support <= noise_local_k")
        if cfg.noise_query_batch_size < 1:
            raise ValueError("noise_query_batch_size must be positive")
        for value in (cfg.noise_local_agreement, cfg.noise_reference_prob_min):
            if not 0 <= value <= 1:
                raise ValueError("Rescue agreement/probability thresholds must be in [0,1]")
        for value in (cfg.noise_max_distance, cfg.noise_neighbor_max_distance, cfg.noise_rescue_min_margin):
            if not np.isfinite(value) or not 0 <= value <= 2:
                raise ValueError("Rescue distances/margin must be finite and in [0,2]")
        if cfg.hdbscan_min_cluster_size is None:
            cfg.hdbscan_min_cluster_size = max(5, min(50, n_clust // 500))
        if cfg.hdbscan_min_samples is None:
            cfg.hdbscan_min_samples = max(2, min(20, n_clust // 100))

        allowed_algorithms = {"best", "prims_kdtree", "prims_balltree", "boruvka_kdtree", "boruvka_balltree", "generic"}
        if cfg.hdbscan_algorithm not in allowed_algorithms:
            raise ValueError(
                f"hdbscan_algorithm must be one of {sorted(allowed_algorithms)}, "
                f"got {cfg.hdbscan_algorithm!r}"
            )
        if cfg.hdbscan_method not in ("eom", "leaf"):
            raise ValueError(f"hdbscan_method must be 'eom' or 'leaf', got {cfg.hdbscan_method!r}")
        if cfg.hdbscan_epsilon < 0:
            raise ValueError(f"hdbscan_epsilon must be >= 0, got {cfg.hdbscan_epsilon}")
        if not 0 <= cfg.hdbscan_soft_prob_min <= 1:
            raise ValueError(f"hdbscan_soft_prob_min must be in [0,1], got {cfg.hdbscan_soft_prob_min}")
        if not 0 <= cfg.eukcc_min_eukaryote_fraction <= 1:
            raise ValueError(
                f"eukcc_min_eukaryote_fraction must be in [0,1], "
                f"got {cfg.eukcc_min_eukaryote_fraction}"
            )
        if not 0 <= cfg.skani_ani_threshold <= 100:
            raise ValueError(f"skani_ani_threshold must be in [0,100], got {cfg.skani_ani_threshold}")
        if not 0 <= cfg.skani_min_af <= 100:
            raise ValueError(f"skani_min_af must be in [0,100], got {cfg.skani_min_af}")

        if cfg.noise_rescue_stage not in ("early", "late"):
            raise ValueError(f"noise_rescue_stage must be early/late, got {cfg.noise_rescue_stage!r}")
        if cfg.noise_rescue_candidate not in ("centroid", "knn"):
            raise ValueError(f"noise_rescue_candidate must be centroid/knn, got {cfg.noise_rescue_candidate!r}")
        if cfg.subcluster_te_mode not in ("auto", "on", "off"):
            raise ValueError(f"subcluster_te_mode must be auto/on/off, got {cfg.subcluster_te_mode!r}")
        if cfg.subcluster_method not in ("eom", "leaf"):
            raise ValueError(f"subcluster_method must be eom/leaf, got {cfg.subcluster_method!r}")
        if cfg.subcluster_cov_merge_tau <= 0:
            raise ValueError("subcluster_cov_merge_tau must be > 0")
        for name in ("subcluster_tnf_sep_max", "binmerge_tnf_sep_max", "binmerge_cov_tau",
                     "prune_shape_z", "recruit_max_dist"):
            if getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("binmerge_misfit_z", "reassign_max_z"):
            if getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 1 <= cfg.reassign_min_votes <= cfg.reassign_k:
            raise ValueError("Require 1 <= reassign_min_votes <= reassign_k")
        for name in ("subcluster_noise_min_agreement", "prune_max_frac", "recruit_agreement",
                     "binmerge_max_misfit", "binmerge_min_tnf_mix", "subcluster_max_misfit",
                     "tnf_consensus_min", "merge_gc_q_lo", "merge_gc_q_hi", "best_home_min_share"):
            if not 0 <= getattr(cfg, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("marker_min_gain", "marker_max_cont_increase"):
            if getattr(cfg, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 1 <= cfg.recruit_min_support <= cfg.recruit_k:
            raise ValueError("Require 1 <= recruit_min_support <= recruit_k")
        if cfg.recruit_min_len >= cfg.cluster_min_contig_len and cfg.recruit_enabled:
            log.warning("recruit_min_len >= cluster_min_contig_len -- recruitment has no candidates")

        available = os.cpu_count() or 1
        if cfg.clustering_threads > available:
            log.warning(f"clustering_threads={cfg.clustering_threads} exceeds "
                        f"os.cpu_count()={available} -- clamping to {available} "
                        f"(a shared HPC job can be allocated fewer cores than the node has).")
            cfg.clustering_threads = available

        return cfg

    def log_summary(self):
        log.info("ClusteringConfig:")
        log.info(f"  cluster_min_contig_len : {self.cluster_min_contig_len}bp")
        log.info(f"  hdbscan                : min_cluster={self.hdbscan_min_cluster_size} "
                 f"min_samples={self.hdbscan_min_samples} "
                 f"epsilon={self.hdbscan_epsilon} method={self.hdbscan_method}")
        log.info(f"  soft_prob_min          : {self.hdbscan_soft_prob_min}")
        log.info(f"  noise_max_distance     : {self.noise_max_distance}")
        log.info(f"  refinement_max_distance: {self.refinement_max_distance}")
        log.info(f"  bin_filter             : min_contigs={self.min_contigs_per_bin} "
                 f"min_len={self.min_bin_length_bp:,}bp "
                 f"min_n50={self.min_bin_n50_bp}bp")
        log.info(f"  skani_ani_threshold    : {self.skani_ani_threshold}%")
        log.info(f"  run_eukcc={self.run_eukcc}  eukcc_merge_enabled={self.eukcc_merge_enabled}")
        log.info(f"  backend                : {self.clustering_backend}  "
                 f"threads={self.clustering_threads}  seed={self.clustering_seed}")
        log.info(f"  compute_silhouette     : {self.compute_silhouette}")
        log.info(f"  noise_rescue           : stage={self.noise_rescue_stage}  candidate={self.noise_rescue_candidate}")
        log.info(f"  refine_cov_standardize : {self.refine_cov_standardize}  "
                 f"(v8.9: per-sample coverage z-scoring in refinement statistics)")
        log.info(f"  resume={self.resume}")


# =============================================================================
# BACKEND RESOLUTION (FIX 10)
# =============================================================================

def _gpu_available() -> bool:
    """Best-effort GPU probe for 'auto' mode. cuML depends on cupy, so a
    usable cupy + a visible CUDA device is a reasonable proxy for "cuML's
    GPU HDBSCAN is likely to work". Never raises -- any failure means "no"."""
    try:
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve_clustering_backend(requested: str) -> Tuple[str, str]:
    """
    Returns (backend, library_version_str). backend is "cpu" or "gpu" --
    the ACTUAL backend that will be used, which may differ from `requested`
    (e.g. requested="gpu" but no GPU/cuML available -> falls back to "cpu"
    with a warning). Both the requested and actual values are recorded by
    the caller into the fingerprint/manifest -- never silently substituted
    without a trace.
    """
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu", f"hdbscan=={getattr(hdbscan, '__version__', 'unknown')}"

    want_gpu = requested == "gpu" or (requested == "auto" and _gpu_available())
    if not want_gpu:
        return "cpu", f"hdbscan=={getattr(hdbscan, '__version__', 'unknown')}"

    try:
        import cuml  # noqa: F401
        if not _gpu_available():
            raise RuntimeError("cuml importable but no CUDA device detected")
        return "gpu", f"cuml=={getattr(cuml, '__version__', 'unknown')}"
    except Exception as e:
        level = log.warning if requested == "gpu" else log.info
        level(f"clustering_backend={requested!r} wanted GPU but it is not usable here ({e}) -- "
              f"falling back to CPU hdbscan.")
        return "cpu", f"hdbscan=={getattr(hdbscan, '__version__', 'unknown')}"


def _to_numpy(x):
    """Bring a possibly-GPU (cupy) array back to host numpy."""
    if hasattr(x, "get"):
        try:
            return x.get()
        except Exception:
            pass
    return np.asarray(x)


def _fit_hdbscan(latent_norm: np.ndarray, params: dict, backend: str, n_threads: int):
    """
    Runs HDBSCAN with IDENTICAL parameters regardless of backend -- this is
    one algorithm on two execution backends, not a second candidate
    algorithm (no algorithm search was added). Any failure on the GPU path
    falls back to CPU transparently; the caller is told which backend
    actually ran. DISCLOSED: the GPU path (cuml.cluster.HDBSCAN) could not
    be exercised against a real RAPIDS installation in this sandbox -- its
    API is assumed compatible with the CPU `hdbscan` package's
    fit_predict/.probabilities_ convention per cuML's own stated design
    goal, but this is unverified here. If it errors for any reason, this
    function catches it and reruns on CPU rather than propagating a
    possibly-version-specific cuML error into the pipeline.
    """
    if backend == "gpu":
        try:
            from cuml.cluster import HDBSCAN as HDBSCAN_GPU
            # FIX 29 (v7.6): CPU and GPU now get INDEPENDENT parameter
            # dicts. Previously the GPU branch received the exact same
            # `params` dict built for CPU (via _hdbscan_params()) unpacked
            # with **params, plus its own separately hardcoded
            # prediction_data=True -- meaning as of FIX 25 (v7.5) it was
            # ALSO silently receiving "algorithm": cfg.hdbscan_algorithm, a
            # CPU-package-specific concept (prims_kdtree/boruvka_kdtree/
            # etc.) that cuML's HDBSCAN is not confirmed to accept the same
            # way, if at all. An incompatible kwarg here would raise inside
            # this try block and silently fall back to CPU -- masking a
            # real configuration problem as "no GPU available", exactly the
            # failure mode this module's own docstring says it tries to
            # avoid. gpu_params below is built explicitly for the GPU path
            # only, and does not blindly inherit CPU-specific keys.
            gpu_params = {
                "min_cluster_size": params.get("min_cluster_size"),
                "min_samples": params.get("min_samples"),
                "cluster_selection_epsilon": params.get("cluster_selection_epsilon"),
                "cluster_selection_method": params.get("cluster_selection_method"),
                # prediction_data intentionally omitted here too -- same
                # reasoning as the CPU path (FIX 25): no
                # approximate_predict/membership_vector call exists
                # anywhere in this module, so there is nothing that needs
                # cuML's equivalent cached prediction data either.
            }
            clusterer = HDBSCAN_GPU(**gpu_params, metric="euclidean")
            labels = _to_numpy(clusterer.fit_predict(latent_norm))
            probs = _to_numpy(clusterer.probabilities_).astype(np.float32)
            return labels, probs, "gpu"
        except Exception as e:
            log.warning(f"GPU HDBSCAN (cuml) failed at runtime ({e}) -- falling back to CPU "
                        f"for this run. Investigate before relying on clustering_backend='gpu'.")
            backend = "cpu"

    clusterer = hdbscan.HDBSCAN(
        **params,
        metric="euclidean",
        core_dist_n_jobs=n_threads,
        # PERFORMANCE FIX (v7.5, FIX 25): prediction_data=True removed. This
        # flag was set but the soft-prediction capability it exists to
        # support (approximate_predict / membership_vector /
        # all_points_membership_vectors) is never called anywhere in this
        # module -- confirmed by search. It builds and retains an extra
        # spatial index purely for a feature this pipeline doesn't use.
        # Removing it is a genuine, low-risk saving (HDBSCAN's own docs:
        # only needed for predicting labels/membership of NEW points) with
        # zero change to labels_ or probabilities_.
        #
        # `algorithm` now comes from `params` (cfg.hdbscan_algorithm,
        # default "best" -- HDBSCAN's own automatic choice based on data
        # size/dimensionality/metric). A specific algorithm
        # (prims_kdtree/prims_balltree/boruvka_kdtree/boruvka_balltree/
        # generic) is deliberately NOT forced here: different algorithm
        # implementations are not guaranteed to produce identical labels or
        # noise assignment on identical input, so switching away from
        # "best" requires benchmarking on this exact dataset first (see
        # ClusteringConfig.hdbscan_algorithm's field comment) -- not an
        # unverified assumption baked into the code.
    )
    labels = clusterer.fit_predict(latent_norm)
    probs = clusterer.probabilities_.astype(np.float32)
    return labels, probs, "cpu"


# =============================================================================
# UTILITY
# =============================================================================

def cosine_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def compute_n50(lengths: List[int]) -> int:
    if not lengths:
        return 0
    sorted_lens = sorted(lengths, reverse=True)
    total       = sum(sorted_lens)
    cumsum      = 0
    for ln in sorted_lens:
        cumsum += ln
        if cumsum >= total / 2:
            return ln
    return 0


def load_contig_sequences_and_lengths(fasta_path: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    """FIX 15: one pass over the FASTA instead of two (v6 called
    load_contig_sequences and load_contig_lengths separately, each a full
    file read -- wasteful for the ~1.67M-contig assembly this module's own
    docstring targets).

    FIX 28 (v7.6): duplicate FASTA headers are now detected explicitly. The
    dict-based storage here (seqs[current_name] = seq) previously let a
    duplicate header silently overwrite the earlier sequence with no
    warning -- the existing duplicate-ID check in
    load_canonical_contig_order() only validates contig_ids_path (the
    upstream TNF/TE/COV contig-ID file), never this FASTA file itself, so a
    duplicate header here was never actually caught anywhere. This is now a
    hard error, consistent with how load_canonical_contig_order() treats
    duplicates in its own input.
    """
    seqs: Dict[str, str] = {}
    lengths: Dict[str, int] = {}
    dup_headers: List[str] = []
    current_name = None
    current_seq: List[str] = []

    def _flush():
        if current_name is not None:
            if current_name in seqs:
                dup_headers.append(current_name)
            seq = "".join(current_seq)
            seqs[current_name] = seq
            lengths[current_name] = len(seq)

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    _flush()

    if dup_headers:
        raise ValueError(
            f"{fasta_path} contains {len(dup_headers)} duplicate contig header(s), e.g. "
            f"{dup_headers[:5]} -- a duplicate header silently overwrites the earlier "
            f"sequence in dict-based storage, corrupting downstream length/sequence lookups "
            f"for that contig without any other check catching it. Fix the FASTA (deduplicate "
            f"headers) before clustering."
        )

    log.info(f"Loaded {len(seqs):,} sequences from {fasta_path}")
    return seqs, lengths


# =============================================================================
# STEP 0 -- CANONICAL CONTIG ORDER / MANIFEST VALIDATION (FIX 1, FIX 2, FIX 3)
# =============================================================================

def load_canonical_contig_order(contig_ids_path: str, encoder_manifest_path: str) -> Tuple[List[str], str, Optional[int]]:
    """
    Reads the ordered, canonical contig-ID list -- NOT from FASTA parse
    order. Accepts either a JSON list (TNF/TE's contig_ids.json) or a
    newline-delimited text file (COV's contig_ids.txt), auto-detected by
    suffix, matching the real conventions verified against
    tnf_gene.py / te_composition.py / coverage.py's actual save code.

    Cross-checks the recomputed order hash against
    encoder_manifest.json["contig_id_order_hash"] (written by encoder.py's
    own FIX 6 alignment check) -- a hard mismatch here means final_latent.npy
    was NOT built from this exact contig order, and nothing downstream can
    be trusted without fixing that first.
    """
    p = Path(contig_ids_path)
    if not p.exists():
        raise FileNotFoundError(f"contig_ids_path does not exist: {p}")

    if p.suffix == ".json":
        with open(p) as f:
            contig_ids = json.load(f)
    else:
        contig_ids = [line for line in p.read_text().splitlines() if line != ""]

    # FIX 2: duplicate IDs
    counts = collections.Counter(contig_ids)
    dupes = [cid for cid, c in counts.items() if c > 1]
    if dupes:
        raise ValueError(
            f"contig_ids_path {p} contains {len(dupes)} duplicate contig ID(s), e.g. "
            f"{dupes[:5]} -- cannot safely index final_latent.npy by a non-unique ID list."
        )

    recomputed_hash = _sha256_id_order(contig_ids)

    mp = Path(encoder_manifest_path)
    if not mp.exists():
        raise FileNotFoundError(
            f"encoder_manifest_path does not exist: {mp}. This file is written by encoder.py "
            f"alongside final_latent.npy -- pass its real path, don't skip this check."
        )
    with open(mp) as f:
        manifest = json.load(f)
    stored_hash = manifest.get("contig_id_order_hash")
    if stored_hash is None:
        raise ValueError(f"{mp} has no 'contig_id_order_hash' key -- is this a real "
                          f"encoder_manifest.json from encoder.py v8+?")
    if stored_hash != recomputed_hash:
        raise ValueError(
            f"Contig order mismatch: {contig_ids_path} hashes to {recomputed_hash}, but "
            f"{encoder_manifest_path} recorded contig_id_order_hash={stored_hash} for the run "
            f"that produced final_latent.npy. These do not describe the same contig order -- "
            f"refusing to index the latent by a possibly-wrong ID list. Point contig_ids_path "
            f"at the exact contig-ID file used for the encoder run that produced this latent."
        )

    latent_dim_expected = manifest.get("config", {}).get("final_dim")
    n_contigs_expected = manifest.get("n_contigs")
    if n_contigs_expected is not None and n_contigs_expected != len(contig_ids):
        raise ValueError(
            f"encoder_manifest.json says n_contigs={n_contigs_expected}, but contig_ids_path "
            f"has {len(contig_ids)} entries -- inconsistent inputs."
        )

    log.info(f"Canonical contig order: {len(contig_ids):,} IDs, order_hash={recomputed_hash} "
             f"(verified against {encoder_manifest_path})")
    return contig_ids, recomputed_hash, latent_dim_expected


# =============================================================================
# STEP 1 -- LOAD INPUTS
# =============================================================================

def load_inputs(final_latent_path: str, encoder_manifest_path: str, contig_ids_path: str,
                 fasta_path: str, cov_features_path: str,
                 tnf_weights_path: Optional[str], te_weights_path: Optional[str],
                 cfg: ClusteringConfig):
    log.info("Loading inputs...")

    contig_ids, order_hash, latent_dim_expected = load_canonical_contig_order(
        contig_ids_path, encoder_manifest_path)
    n = len(contig_ids)

    latent = np.load(final_latent_path).astype(np.float32)
    if latent.ndim != 2:
        raise ValueError(f"final_latent must be 2D, got {latent.shape}")
    if latent.shape[0] != n:
        raise ValueError(
            f"final_latent has {latent.shape[0]} rows but the canonical contig order has "
            f"{n} entries -- these must match exactly. (final_latent={final_latent_path}, "
            f"contig_ids_path={contig_ids_path})"
        )
    if latent_dim_expected is not None and latent.shape[1] != latent_dim_expected:
        raise ValueError(
            f"final_latent has {latent.shape[1]} columns but encoder_manifest.json recorded "
            f"final_dim={latent_dim_expected} -- final_latent.npy looks stale relative to its "
            f"own manifest."
        )
    n_nan = int(np.isnan(latent).sum())
    n_inf = int(np.isinf(latent).sum())
    if n_nan or n_inf:
        raise ValueError(
            f"final_latent contains {n_nan} NaN and {n_inf} Inf value(s) -- refusing to "
            f"cluster on a corrupted embedding. Fix the encoder run that produced it."
        )
    log.info(f"  final_latent: {latent.shape}  std={latent.std():.4f}")

    sequences, lengths = load_contig_sequences_and_lengths(fasta_path)

    # FIX 2: contigs present in the canonical order but missing from the FASTA
    missing_from_fasta = [c for c in contig_ids if c not in sequences]
    if missing_from_fasta:
        raise ValueError(
            f"{len(missing_from_fasta)} contig ID(s) from the canonical order are not present "
            f"in {fasta_path}, e.g. {missing_from_fasta[:5]} -- FASTA does not match the "
            f"assembly the encoder pipeline actually ran on."
        )

    cov_features = np.load(cov_features_path).astype(np.float32)
    if cov_features.shape[0] != n:
        raise ValueError(f"Coverage mismatch: canonical order has {n} contigs, "
                          f"cov_features has {cov_features.shape[0]} rows")
    if cov_features.shape[1] < cfg.n_cov_meta_cols:
        msg = (f"cov_features has {cov_features.shape[1]} columns, fewer than "
               f"n_cov_meta_cols={cfg.n_cov_meta_cols} -- cannot locate valid_mask (column 0).")
        if cfg.allow_missing_valid_mask:
            log.warning(msg + " allow_missing_valid_mask=True -- treating every contig as valid.")
            valid_mask = np.ones(n, dtype=np.float32)
        else:
            raise ValueError(msg + " Set allow_missing_valid_mask=True if this is intentional.")
    else:
        valid_mask = cov_features[:, 0].astype(np.float32)
    n_invalid = int((valid_mask == 0).sum())
    log.info(f"  cov_features: {cov_features.shape}  valid_mask=0 (placeholder) contigs: "
             f"{n_invalid:,}/{n:,}")

    def _load_weights(path, label):
        if not (path and Path(path).exists()):
            return np.ones(n, dtype=np.float32)
        w = np.load(path).squeeze().astype(np.float32)
        w = np.atleast_1d(w)
        if len(w) != n:
            raise ValueError(f"{label} weights file {path} has {len(w)} entries but there are "
                              f"{n} contigs -- refusing to use a mismatched weight array.")
        return w

    tnf_weights = _load_weights(tnf_weights_path, "TNF")
    te_weights = _load_weights(te_weights_path, "TE")

    log.info(f"  n_contigs={n:,}  tnf_weights_zero={int((tnf_weights==0).sum()):,}  "
             f"te_weights_zero={int((te_weights==0).sum()):,}")
    return (latent, sequences, lengths, cov_features, valid_mask,
            tnf_weights, te_weights, contig_ids, order_hash)


# =============================================================================
# STEP 2 -- ANCHOR CONTIGS (diagnostic only -- unchanged core logic from v6)
# =============================================================================

def identify_anchors(contig_ids, lengths, tnf_weights, cov_features, cfg):
    """
    Identify high-confidence anchor contigs for logging/diagnostics only.
    HDBSCAN runs fully unsupervised -- anchors are never fed into it.
    4-tier fallback ensures anchors are always found regardless of data
    quality.
    """
    n          = len(contig_ids)
    length_arr = np.array([lengths.get(c, 0) for c in contig_ids])

    n_meta  = cfg.n_cov_meta_cols
    raw_cov = cov_features[:, n_meta:] if cov_features.shape[1] > n_meta else cov_features
    cov_var = raw_cov.var(axis=1)

    tiers = [
        (cfg.anchor_min_length,       cfg.anchor_min_tnf_weight, cfg.anchor_max_cov_var),
        (cfg.anchor_min_length // 2,  cfg.anchor_min_tnf_weight * 0.75, cfg.anchor_max_cov_var * 2),
        (cfg.cluster_min_contig_len,  0.5,  float("inf")),
        (cfg.cluster_min_contig_len,  0.0,  float("inf")),
    ]

    for min_l, min_w, max_v in tiers:
        mask = (length_arr >= min_l) & (tnf_weights >= min_w) & (cov_var <= max_v)
        if mask.sum() >= 10:
            log.info(f"Anchors: {mask.sum():,}/{n:,} "
                     f"(min_len={min_l}bp min_tnf_w={min_w:.2f} max_cov_var={max_v:.1f})")
            return mask

    top_n   = max(10, n // 10)
    idx_top = np.argsort(length_arr)[::-1][:top_n]
    mask    = np.zeros(n, dtype=bool)
    mask[idx_top] = True
    log.warning(f"FLAG:ANCHOR_TOO_FEW -- Using top {top_n} longest contigs as anchors")
    return mask


# =============================================================================
# STEP 2.5 -- LENGTH + VALIDITY FILTER (FIX 4)
# =============================================================================

def filter_contigs_for_clustering(latent, contig_ids, lengths, valid_mask, cfg):
    """
    Keep only contigs that are (a) >= cluster_min_contig_len AND
    (b) valid_mask > 0 (not a coverage.py placeholder row). Excluded
    contigs are tracked with an explicit reason so they can be reported in
    unclustered.tsv rather than silently disappearing.
    """
    min_len   = cfg.cluster_min_contig_len
    length_ok = np.array([lengths.get(c, 0) >= min_len for c in contig_ids])
    valid_ok  = valid_mask > 0
    clust_mask = length_ok & valid_ok

    reasons: Dict[str, str] = {}
    for i, cid in enumerate(contig_ids):
        if not valid_ok[i]:
            reasons[cid] = "invalid_coverage"
        elif not length_ok[i]:
            reasons[cid] = "too_short"

    n_kept = int(clust_mask.sum())
    log.info(f"Length+validity filter (>={min_len}bp, valid_mask>0): {n_kept:,} kept | "
             f"{int((~length_ok).sum()):,} too_short | "
             f"{int((~valid_ok).sum()):,} invalid_coverage -> excluded")
    if n_kept == 0:
        raise RuntimeError(
            f"All {len(contig_ids):,} contigs were excluded (too short or invalid coverage). "
            f"Lower cluster_min_contig_len, check allow_missing_valid_mask, or check upstream "
            f"pipeline output quality."
        )

    ids_clust = [contig_ids[i] for i in np.where(clust_mask)[0]]
    return latent[clust_mask], ids_clust, clust_mask, reasons


# =============================================================================
# STEP 3 -- HDBSCAN (checkpointed, fingerprinted, backend-selectable)
# =============================================================================

def _hdbscan_params(cfg: ClusteringConfig) -> dict:
    return {
        "min_cluster_size":          cfg.hdbscan_min_cluster_size,
        "min_samples":               cfg.hdbscan_min_samples,
        "cluster_selection_epsilon": cfg.hdbscan_epsilon,
        "cluster_selection_method":  cfg.hdbscan_method,
        "algorithm":                 cfg.hdbscan_algorithm,  # FIX 25 (v7.5)
    }


def run_hdbscan(latent_clust, cfg: ClusteringConfig, outdir: Path, n_clust: int,
                 fp: str, resolved_backend: str, backend_lib: str
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    HDBSCAN directly in latent space -- no UMAP. Same parameters/semantics
    as v6 (method='leaf' by v6's own old default; this module's own default
    is now 'eom', see FIX 21). prediction_data removed (FIX 25, v7.5),
    soft-probability noise filter unchanged. New in v7: fingerprinted
    checkpoint (FIX 6) and a resolvable CPU/GPU backend (FIX 10).

    FIX 30 (v7.6): resolved_backend/backend_lib are passed in by the
    caller (already resolved once, before the fingerprint was built --
    see run_clustering()'s Step 3) rather than being re-resolved here.

    FIX 33 (v7.7): fp (the fingerprint) no longer encodes backend at all
    (see run_clustering()'s Step 3 comment for why baking resolved_backend
    into the hash was itself buggy: it could not distinguish "GPU resolved
    as available AND actually succeeded" from "GPU resolved as available
    but silently fell back to CPU at runtime", since both cases resolve to
    the same "gpu" string before fitting ever happens). Instead, a cache
    hit on fp is now a NECESSARY but not SUFFICIENT condition: this
    function additionally compares resolved_backend (what would be
    attempted THIS run) against the actual_backend RECORDED in the cached
    checkpoint's own metadata (what backend actually produced those
    cached labels, after any runtime fallback). Only if they match is the
    cache trusted -- e.g. resolved_backend="gpu" now, but the cached run's
    actual_backend="cpu" (a prior fallback), is correctly treated as a
    cache MISS, giving GPU a genuine fresh attempt instead of silently
    replaying the old CPU fallback result under a "gpu" label.
    """
    ckpt_dir = outdir / "checkpoints" / "hdbscan"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(ckpt_dir) if Checkpoint else None
    labels_path  = ckpt_dir / "hdbscan_labels.npy"
    probs_path   = ckpt_dir / "hdbscan_probs.npy"
    backend_path = ckpt_dir / "hdbscan_backend.json"
    labels_raw_path = ckpt_dir / "hdbscan_labels_raw.npy"

    cached = _checkpoint_ok(ckpt, "hdbscan", fp, [str(labels_path), str(probs_path), str(labels_raw_path)]) if cfg.resume else None
    if cached is not None:
        cached_actual_backend = cached.get("actual_backend")
        if cached_actual_backend is None and backend_path.exists():
            # Fall back to reading it from the sidecar JSON for checkpoints
            # written before this metadata field existed in ckpt itself.
            cached_actual_backend = json.loads(backend_path.read_text()).get("actual_backend")
        if cached_actual_backend == resolved_backend:
            log.info(f"HDBSCAN [CACHED] (actual_backend={cached_actual_backend} matches "
                      f"resolved_backend={resolved_backend})")
            return np.load(labels_path), np.load(probs_path), cosine_normalize(latent_clust), cached_actual_backend
        else:
            log.info(f"HDBSCAN checkpoint fingerprint matches, but cached actual_backend="
                      f"{cached_actual_backend!r} != this run's resolved_backend={resolved_backend!r} "
                      f"-- treating as a cache miss (FIX 33) to give the resolved backend a genuine "
                      f"fresh attempt rather than replaying a possibly-stale fallback result.")

    requested_backend = resolved_backend
    log.info(f"Running HDBSCAN on {n_clust:,} contigs in {latent_clust.shape[1]}D latent space "
             f"(backend requested={cfg.clustering_backend} -> resolved={requested_backend}, {backend_lib})...")
    log.info(f"  method={cfg.hdbscan_method}  min_cluster_size={cfg.hdbscan_min_cluster_size}  "
             f"min_samples={cfg.hdbscan_min_samples}  epsilon={cfg.hdbscan_epsilon}  "
             f"seed={cfg.clustering_seed}")

    np.random.seed(cfg.clustering_seed)
    latent_norm = cosine_normalize(latent_clust)
    params      = _hdbscan_params(cfg)

    t0 = time.time()
    labels, probs, actual_backend = _fit_hdbscan(latent_norm, params, requested_backend, cfg.clustering_threads)
    elapsed = time.time() - t0

    # FIX 41 (v7.8): genuine pre-filter labels are now saved SEPARATELY,
    # before the soft-probability filter below overwrites them. Previously
    # `hdbscan_labels.npy` -- the file any diagnostic would naturally read
    # as "raw HDBSCAN output" -- already had every low-membership contig
    # forced to -1 by cfg.hdbscan_soft_prob_min. That conflates two
    # genuinely different sources of noise: contigs HDBSCAN's own
    # density-based algorithm put in no cluster at all, versus contigs
    # HDBSCAN DID assign to a real cluster that this module's own
    # additional probability threshold then rejected. A diagnostic reading
    # only the post-filter file cannot distinguish "HDBSCAN produced this
    # much noise" from "our own filter produced this much noise" -- e.g. a
    # reported "77% noise for species X" figure computed from the old
    # single file was NOT attributable to HDBSCAN alone. This file lets
    # that distinction be made directly: compare hdbscan_labels_raw.npy
    # (== -1) against hdbscan_labels.npy (== -1) restricted to rows where
    # raw != -1, to isolate exactly which contigs were filter-rejected
    # rather than genuinely unclustered by HDBSCAN itself.
    labels_raw_path = ckpt_dir / "hdbscan_labels_raw.npy"
    np.save(labels_raw_path, labels)

    n_low_prob = int((probs < cfg.hdbscan_soft_prob_min).sum())
    labels     = labels.copy()
    labels[probs < cfg.hdbscan_soft_prob_min] = -1

    n_bins       = len(set(labels[labels >= 0]))
    n_noise      = int((labels == -1).sum())
    n_raw_noise  = int((np.load(labels_raw_path) == -1).sum())
    n_filter_rejected = n_noise - n_raw_noise  # contigs HDBSCAN clustered but the prob. filter then rejected
    log.info(f"HDBSCAN done in {elapsed:.0f}s ({actual_backend}): {n_bins} bins | "
             f"{n_noise:,} noise post-filter ({n_noise/n_clust:.1%}) -- of which "
             f"{n_raw_noise:,} were genuinely HDBSCAN-native noise and "
             f"{n_filter_rejected:,} were assigned by HDBSCAN but rejected by "
             f"soft_prob_min={cfg.hdbscan_soft_prob_min} | {n_low_prob:,} low-prob filtered total")

    np.save(labels_path, labels)
    np.save(probs_path, probs)
    backend_path.write_text(json.dumps({
        "requested_backend": cfg.clustering_backend,
        "resolved_backend_before_run": requested_backend,
        "actual_backend": actual_backend,
        "backend_library": backend_lib,
    }, indent=2))
    if ckpt:
        ckpt.mark_done("hdbscan", {"_fp": fp, "n_bins": n_bins, "n_noise": n_noise,
                                    "actual_backend": actual_backend})
    return labels, probs, latent_norm, actual_backend


# =============================================================================
# STEP 4 -- ASSIGN NOISE CONTIGS (unchanged core logic from v6)
# =============================================================================

def _exact_neighbors_numpy(query, references, k):
    """Bounded-memory exact kNN fallback when scipy is unavailable."""
    best_d = np.full((len(query), k), np.inf)
    best_i = np.zeros((len(query), k), dtype=np.int64)
    for start in range(0, len(references), 4096):
        block = references[start:start + 4096]
        d = np.maximum(np.sum(query*query, axis=1)[:, None]
                       + np.sum(block*block, axis=1)[None, :] - 2*query@block.T, 0)
        ids = np.broadcast_to(np.arange(start, start + len(block)), d.shape)
        ds = np.concatenate((best_d, d), axis=1)
        ix = np.concatenate((best_i, ids), axis=1)
        keep = np.argpartition(ds, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(ds, keep, axis=1)
        best_i = np.take_along_axis(ix, keep, axis=1)
    order = np.argsort(best_d, axis=1)
    return np.sqrt(np.take_along_axis(best_d, order, axis=1)), np.take_along_axis(best_i, order, axis=1)


def assign_noise_contigs(latent, labels, clust_mask, cfg, probabilities=None, query_mask=None):
    """One pass; existing assignments never change and rescued rows never vote.

    Centers and kNN references come only from original assigned rows passing
    the reference confidence threshold. All distances use normalized latent.
    """
    result = labels.copy()
    if not cfg.noise_rescue_enabled:
        return result
    qm = clust_mask & (labels == -1)
    if query_mask is not None:
        qm &= query_mask
    query_idx = np.flatnonzero(qm)
    trusted = clust_mask & (labels >= 0)
    if probabilities is None:
        log.warning("Rescue skipped: original membership probabilities required")
        return result
    trusted &= np.isfinite(probabilities) & (probabilities >= cfg.noise_reference_prob_min)
    reference_idx = np.flatnonzero(trusted)
    if not len(query_idx) or not len(reference_idx):
        return result
    x = cosine_normalize(latent)
    bins = np.unique(labels[reference_idx])
    centers = np.stack([cosine_normalize(x[trusted & (labels == b)].mean(axis=0, keepdims=True))[0]
                        for b in bins])
    tree = None
    if cfg.noise_rescue_candidate == "knn":
        from scipy.spatial import cKDTree
        tree = cKDTree(x[reference_idx])
        k = min(cfg.noise_local_k, len(reference_idx))
        rescued = 0
        ref_lab = labels[reference_idx]
        for offset in range(0, len(query_idx), cfg.noise_query_batch_size):
            idx = query_idx[offset:offset + cfg.noise_query_batch_size]
            nd, ni = tree.query(x[idx], k=k)
            nd, ni = np.asarray(nd).reshape(len(idx), k), np.asarray(ni).reshape(len(idx), k)
            near = nd <= cfg.noise_neighbor_max_distance
            for row in range(len(idx)):
                lab = ref_lab[ni[row][near[row]]]
                support = len(lab)
                if support < cfg.noise_local_min_support:
                    continue
                vals, cnt = np.unique(lab, return_counts=True)
                j = int(np.argmax(cnt))
                if cnt[j] >= cfg.noise_local_min_support and cnt[j] / support >= cfg.noise_local_agreement:
                    result[idx[row]] = vals[j]
                    rescued += 1
        log.info("Conservative rescue (kNN-majority candidate): %s/%s assigned; original bins frozen",
                 rescued, len(query_idx))
        return result
    if cfg.noise_local_agreement_enabled:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(x[reference_idx])
        except ImportError:
            log.warning("scipy unavailable: using slower exact NumPy neighbor search")
    rescued = 0
    batch_size = cfg.noise_query_batch_size if tree is not None else min(cfg.noise_query_batch_size, 256)
    for offset in range(0, len(query_idx), batch_size):
        idx = query_idx[offset:offset + batch_size]
        q = x[idx]
        dsq = np.sum(q*q, axis=1)[:, None] + np.sum(centers*centers, axis=1)[None, :] - 2*q@centers.T
        d = np.sqrt(np.maximum(dsq, 0))
        nearest = d.argmin(axis=1)
        candidate = bins[nearest]
        best = d[np.arange(len(idx)), nearest]
        accept = best <= cfg.noise_max_distance
        if len(bins) > 1 and cfg.noise_rescue_min_margin > 0:
            accept &= np.partition(d, 1, axis=1)[:, 1] - best >= cfg.noise_rescue_min_margin
        if cfg.noise_local_agreement_enabled:
            k = min(cfg.noise_local_k, len(reference_idx))
            nd, ni = tree.query(q, k=k) if tree is not None else _exact_neighbors_numpy(q, x[reference_idx], k)
            nd, ni = np.asarray(nd).reshape(len(idx), k), np.asarray(ni).reshape(len(idx), k)
            nearby = nd <= cfg.noise_neighbor_max_distance
            votes = nearby & (labels[reference_idx[ni]] == candidate[:, None])
            support = nearby.sum(axis=1)
            agree = votes.sum(axis=1)
            accept &= (support >= cfg.noise_local_min_support)
            accept &= agree >= cfg.noise_local_min_support
            accept &= agree / np.maximum(support, 1) >= cfg.noise_local_agreement
        result[idx[accept]] = candidate[accept]
        rescued += int(accept.sum())
    log.info("Conservative rescue: %s/%s assigned; original bins frozen", rescued, len(query_idx))
    return result


def iterative_refinement(latent, labels, clust_mask, cfg: ClusteringConfig):
    """
    Reassign binned contigs to nearest bin medoid iteratively. Medoid
    (actual data point), not mean, is geometrically correct for
    L2-normalized vectors. Restricted to clust_mask -- excluded contigs are
    never touched.

    FIX 22 (v7.3): a contig is only reassigned to a new medoid if the
    distance to it is <= cfg.refinement_max_distance -- mirroring the
    equivalent guard already used by assign_noise_contigs(). Previously
    every binned contig was force-reassigned to whichever medoid was
    nearest, however far, which for closely related clusters (nearby
    medoids by definition) could progressively pull contigs across real
    cluster boundaries over repeated iterations.
    """
    log.info("Iterative refinement (medoid-based)...")
    labels      = labels.copy()
    latent_norm = cosine_normalize(latent)

    for iteration in range(1, cfg.max_iter + 1):
        unique_bins = sorted(set(labels[labels >= 0]))
        if not unique_bins:
            break

        medoids = []
        for b in unique_bins:
            bin_idx  = np.where(labels == b)[0]
            bin_vecs = latent_norm[bin_idx]
            mean_vec = bin_vecs.mean(axis=0, keepdims=True)
            dists    = np.linalg.norm(bin_vecs - mean_vec, axis=1)
            medoid   = bin_vecs[dists.argmin()]
            medoids.append(medoid)
        medoids = np.stack(medoids)

        binned_mask = clust_mask & (labels >= 0)
        binned_idx  = np.where(binned_mask)[0]
        if len(binned_idx) == 0:
            break

        binned_vecs = latent_norm[binned_idx]
        # FIX 24 (v7.4, performance): same squared-distance-expansion trick
        # as assign_noise_contigs (see its comment for the full rationale).
        # This call runs up to max_iter times per run, so it's the single
        # biggest contributor to clustering runtime for large contig counts
        # / high-dim latents (e.g. TE re-enabled -> 73D fusion latent).
        a_sq = np.sum(binned_vecs**2, axis=1, keepdims=True)   # (n, 1)
        b_sq = np.sum(medoids**2, axis=1)                       # (m,)
        dist_sq = a_sq + b_sq - 2.0 * (binned_vecs @ medoids.T)  # (n, m)
        np.maximum(dist_sq, 0, out=dist_sq)
        dist_mat = np.sqrt(dist_sq, out=dist_sq)
        nearest      = dist_mat.argmin(axis=1)
        nearest_dist = dist_mat[np.arange(len(binned_idx)), nearest]

        # FIX 24 (v7.4, performance): vectorized reassignment instead of a
        # pure-Python per-contig loop, same semantics as before (FIX 22's
        # refinement_max_distance gate is unchanged, just applied via a
        # boolean mask instead of a loop).
        unique_bins_arr = np.asarray(unique_bins)
        new_bins_arr = unique_bins_arr[nearest]
        current_bins_arr = labels[binned_idx]
        change_mask = (new_bins_arr != current_bins_arr) & (nearest_dist <= cfg.refinement_max_distance)
        labels[binned_idx[change_mask]] = new_bins_arr[change_mask]
        n_changed = int(change_mask.sum())

        pct = n_changed / max(1, len(binned_idx))
        log.info(f"  Iter {iteration:2d}: {n_changed:,} reassigned ({pct:.2%})")
        if pct < cfg.convergence_threshold:
            log.info(f"  Converged at iteration {iteration}")
            break

    return labels


# =============================================================================
# STEP 6 -- FILTER LOW-QUALITY BINS (unchanged core logic from v6)
# =============================================================================

def load_modality_latents(final_latent_path, n):
    """Per-modality latents written by encoder.py next to final_latent.npy.
    Returns {} if any is missing or row-mismatched (sub-clustering then falls
    back to the fused latent)."""
    enc = Path(final_latent_path).parent
    out = {}
    for key, fn in (("tnf", "latent_tnf.npy"), ("cov", "latent_cov.npy"), ("te", "latent_te.npy")):
        p = enc / fn
        if not p.exists():
            log.warning(f"Sub-clustering: {p} missing -- using fused latent instead")
            return {}
        arr = np.load(p).astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] != n:
            log.warning(f"Sub-clustering: {fn} shape {arr.shape} does not match {n} contigs -- "
                        f"using fused latent instead")
            return {}
        out[key] = arr
    return out


def _knn(query, ref, k):
    k = min(k, len(ref))
    try:
        from scipy.spatial import cKDTree
        d, i = cKDTree(ref).query(query, k=k)
        return np.asarray(d).reshape(len(query), k), np.asarray(i).reshape(len(query), k)
    except ImportError:
        return _exact_neighbors_numpy(query, ref, k)


def _te_weight_for_bin(idx, mods, te_weights, cfg):
    """Decide, per bin and without labels, whether TE helps here.
    TE is dropped when the bin's TE annotation is weak (low median te_weight
    or mostly zero -- e.g. non-fungal genomes absent from the TE library) or
    when TE neighbourhoods do not agree with TNF neighbourhoods at all."""
    if cfg.subcluster_te_mode == "on":
        return 1.0, "forced_on"
    if cfg.subcluster_te_mode == "off" or "te" not in mods:
        return 0.0, "forced_off" if cfg.subcluster_te_mode == "off" else "no_te_latent"
    w = te_weights[idx]
    if (w == 0).mean() >= 0.5:
        return 0.0, f"te_weight_zero_frac={float((w == 0).mean()):.2f}"
    med = float(np.median(w))
    if med < cfg.subcluster_te_min_weight:
        return 0.0, f"median_te_weight={med:.2f}"
    rng = np.random.default_rng(0)
    s = idx if len(idx) <= 3000 else rng.choice(idx, 3000, replace=False)
    k = min(10, len(s) - 1)
    if k < 2:
        return 1.0, "too_small_to_test"
    _, a = _knn(cosine_normalize(mods["te"][s]), cosine_normalize(mods["te"][s]), k + 1)
    _, b = _knn(cosine_normalize(mods["tnf"][s]), cosine_normalize(mods["tnf"][s]), k + 1)
    jac = np.mean([len(set(x[1:]) & set(y[1:])) / len(set(x[1:]) | set(y[1:])) for x, y in zip(a, b)])
    if jac < cfg.subcluster_te_min_agreement:
        return 0.0, f"te_tnf_knn_jaccard={jac:.3f}"
    return 1.0, f"median_te_weight={med:.2f},te_tnf_knn_jaccard={jac:.3f}"


def _cov_profiles(cov_features, cfg):
    prof = cov_features[:, cfg.n_cov_meta_cols:].astype(np.float64)
    if prof.shape[1] and np.nanmin(prof) >= 0:
        prof = np.log1p(prof)
    if getattr(cfg, "refine_cov_standardize", False) and prof.shape[1]:
        mu = np.nanmean(prof, 0); sd = np.nanstd(prof, 0)
        prof = (prof - mu) / np.where(sd > 1e-9, sd, 1.0)
    return prof


def _cov_shape(cov_features, cfg):
    """Per-contig centred log-coverage: keeps the sample-to-sample PATTERN (which
    identifies an organism) and removes the overall LEVEL (which varies within one
    genome through repeats, GC bias or assembly collapse)."""
    prof = _cov_profiles(cov_features, cfg)
    return prof - prof.mean(1, keepdims=True)


def _group_stats(members, T, P):
    """TNF centroid/radius and coverage median of one group (row indices into T/P)."""
    c = T[members].mean(0)
    c = c / max(np.linalg.norm(c), 1e-12)
    rad = float(np.median(np.linalg.norm(T[members] - c, axis=1)))
    return c, max(rad, 1e-6), np.median(P[members], 0)


def _shape_model(P_rows):
    med = np.median(P_rows, 0)
    sd = np.maximum(np.median(np.abs(P_rows - med), 0) * 1.4826, 0.05)
    return med, sd


def _shape_z(P_rows, model):
    med, sd = model
    return (np.abs(P_rows - med) / sd).mean(1)


def _misfit(small, big, P, z):
    """Fraction of `small` contigs that are coverage-shape outliers under `big`'s model."""
    return float((_shape_z(P[small], _shape_model(P[big])) > z).mean())


def _tnf_mix(a, b, T, k=10, n_small=300, n_big=3000, seed=0):
    """Observed/expected share of cross-group TNF neighbours for group a's contigs.
    ~1 when a and b are intermixed (same genome), ~0 when they occupy separate regions."""
    rng = np.random.default_rng(seed)
    sa = a if len(a) <= n_small else rng.choice(a, n_small, replace=False)
    sb = b if len(b) <= n_big else rng.choice(b, n_big, replace=False)
    pts = np.concatenate([sa, sb]); is_b = np.r_[np.zeros(len(sa), bool), np.ones(len(sb), bool)]
    kk = min(k, len(pts) - 1)
    if kk < 1:
        return 1.0
    _, nn = _knn(T[sa], T[pts], kk + 1)
    obs = is_b[nn[:, 1:]].mean()
    exp = len(sb) / (len(pts) - 1)
    return float(obs / exp) if exp > 0 else 1.0


def contig_gc(contig_ids, sequences):
    """Per-contig GC fraction (ACGT only); NaN if no ACGT."""
    out = np.full(len(contig_ids), np.nan)
    for i, c in enumerate(contig_ids):
        s = sequences.get(c)
        if not s:
            continue
        s = s.upper()
        g = s.count("G") + s.count("C"); acgt = g + s.count("A") + s.count("T")
        if acgt:
            out[i] = g / acgt
    return out


def _merge_groups(groups, T, P, tnf_sep_max, cov_tau, log_rows=None, tag="",
                  max_misfit=None, misfit_z=4.0, min_tnf_mix=None, guard_rows=None,
                  gc=None, max_gc_diff=None, gc_outside_max=None, gc_q=(0.05, 0.95)):
    """
    Greedy agglomeration. Two groups merge only if BOTH
      tnf_sep = |c_a - c_b| / mean(radius_a, radius_b)  < tnf_sep_max   (same genome by composition)
      cov_d   = mean_s |med_a - med_b| / sd_s            < cov_tau       (coverage SHAPE not contradictory)
    The closest admissible pair (by tnf_sep) merges first; stats are recomputed.
    groups: {gid: np.ndarray of row indices}. Returns merged dict.
    """
    groups = {g: np.asarray(m) for g, m in groups.items() if len(m)}
    if len(groups) < 2:
        return groups
    mad = np.stack([np.median(np.abs(P[m] - np.median(P[m], 0)), 0) for m in groups.values()])
    sd = np.maximum(np.median(mad, 0) * 1.4826, 1e-3)
    st = {g: _group_stats(m, T, P) for g, m in groups.items()}
    refused = set()
    while len(groups) > 1:
        keys = list(groups)
        C = np.stack([st[g][0] for g in keys]); R = np.array([st[g][1] for g in keys])
        M = np.stack([st[g][2] for g in keys]) / sd
        D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2) / (0.5 * (R[:, None] + R[None, :]))
        V = np.abs(M[:, None, :] - M[None, :, :]).mean(2)
        np.fill_diagonal(D, np.inf)
        ok = (D < tnf_sep_max) & (V < cov_tau)
        if log_rows is not None and len(keys) <= 400:
            iu = np.triu_indices(len(keys), 1)
            near = np.argsort(D[iu])[:min(len(iu[0]), 3 * len(keys))]
            for t in near:
                i, j = iu[0][t], iu[1][t]
                log_rows.append((tag, keys[i], keys[j], len(groups[keys[i]]), len(groups[keys[j]]),
                                 round(float(D[i, j]), 3), round(float(V[i, j]), 3), bool(ok[i, j])))
            log_rows = None  # log only the first (pre-merge) pairwise snapshot
        if not ok.any():
            break
        # v8.2: walk admissible pairs from closest TNF outward; the first one that also
        # passes the fit-test / TNF-mixing guards is merged. Refused pairs are
        # remembered (keyed by group ids and sizes, so they are re-tested if either
        # group later changes).
        order = np.argsort(np.where(ok, D, np.inf), axis=None)
        chosen = None
        for flat in order:
            i, j = np.unravel_index(flat, D.shape)
            if not ok[i, j]:
                break
            if i > j:
                continue
            gi, gj = keys[i], keys[j]
            key = (gi, len(groups[gi]), gj, len(groups[gj]))
            if key in refused:
                continue
            small, big = (gi, gj) if len(groups[gi]) <= len(groups[gj]) else (gj, gi)
            reason = ""
            if gc is not None and max_gc_diff is not None and max_gc_diff > 0:
                gd = abs(float(np.nanmedian(gc[groups[small]])) - float(np.nanmedian(gc[groups[big]])))
                if gd > max_gc_diff:
                    reason = f"gc_diff={gd:.3f}"
            if not reason and gc is not None and gc_outside_max is not None and gc_outside_max > 0:
                gb = gc[groups[big]]; gb = gb[np.isfinite(gb)]
                gs = gc[groups[small]]; gs = gs[np.isfinite(gs)]
                if len(gb) >= 10 and len(gs):
                    lo, hi = np.quantile(gb, gc_q[0]), np.quantile(gb, gc_q[1])
                    out_frac = float(((gs < lo) | (gs > hi)).mean())
                    if out_frac > gc_outside_max:
                        reason = f"gc_outside={out_frac:.2f}"
            if not reason and max_misfit is not None:
                mf = _misfit(groups[small], groups[big], P, misfit_z)
                if mf > max_misfit:
                    reason = f"misfit={mf:.2f}"
            if not reason and min_tnf_mix is not None:
                mx = _tnf_mix(groups[small], groups[big], T)
                if mx < min_tnf_mix:
                    reason = f"tnf_mix={mx:.2f}"
            if reason:
                refused.add(key)
                if guard_rows is not None:
                    guard_rows.append((tag, small, big, len(groups[small]), len(groups[big]),
                                       round(float(D[i, j]), 3), round(float(V[i, j]), 3), reason))
                continue
            chosen = (gi, gj)
            break
        if chosen is None:
            break
        gi, gj = chosen
        groups[gi] = np.concatenate([groups[gi], groups.pop(gj)])
        st.pop(gj)
        st[gi] = _group_stats(groups[gi], T, P)
    return groups


def _tnf_space(mods, final_latent):
    if mods and "tnf" in mods:
        return cosine_normalize(mods["tnf"]), "tnf_latent"
    return cosine_normalize(final_latent), "fused_latent(fallback)"


def _best_home(b, idx, piece_groups, labels, clust_mask, T, P, cfg, gc=None, log_rows=None):
    """v8.8: neighbour-based best-home. For each leaf piece, sample its contigs and look at
    their nearest TNF neighbours OUTSIDE the piece (among all binned contigs). The piece moves
    to bin X only if (1) >= best_home_min_share of those outside neighbours are in X (X != b),
    (2) they are about as close as the piece's own neighbours (median distance ratio <=
    best_home_max_ratio -> same-genome density), and (3) coverage distance and misfit checks
    pass. Neighbour counts do not depend on bin size, so large diffuse bins cannot attract
    pieces (the v8.7 radius-based failure). Modifies `labels` in place for moved pieces."""
    if len(piece_groups) < 2:
        return piece_groups, 0
    ref = np.flatnonzero(clust_mask & (labels >= 0))
    if len(ref) < cfg.best_home_k + 2:
        return piece_groups, 0
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return piece_groups, 0
    rng = np.random.default_rng(0)
    tree_all = cKDTree(T[ref])
    k = cfg.best_home_k
    moved, keep = 0, {}
    for p, m in piece_groups.items():
        g = idx[m]
        if len(g) <= k + 1:
            keep[p] = m
            continue
        smp = g if len(g) <= cfg.best_home_sample else rng.choice(g, cfg.best_home_sample, replace=False)
        # own-piece neighbour distance (density reference)
        d_self, _ = cKDTree(T[g]).query(T[smp], k=k + 1)
        d_self = float(np.median(d_self[:, 1:].mean(1)))
        # outside neighbours: query extra, drop members of this piece
        in_piece = np.zeros(len(labels), bool); in_piece[g] = True
        kk = min(len(ref), k + min(len(g), 4 * k) + 1)
        dd, nn = tree_all.query(T[smp], k=kk)
        dd = np.atleast_2d(dd); nn = np.atleast_2d(nn)
        votes, dists = [], []
        for drow, nrow in zip(dd, nn):
            sel = ~in_piece[ref[nrow]]
            lab_o = labels[ref[nrow[sel]]][:k]; d_o = drow[sel][:k]
            votes.append(lab_o); dists.append(d_o)
        allv = np.concatenate(votes) if votes else np.array([], int)
        allv = allv[allv >= 0]
        if not len(allv):
            keep[p] = m
            continue
        vals, cnt = np.unique(allv, return_counts=True)
        j = int(np.argmax(cnt)); X = int(vals[j]); share = cnt[j] / len(allv)
        if X == b or share < cfg.best_home_min_share:
            keep[p] = m
            continue
        d_out = float(np.median([np.mean(d[v == X]) for v, d in zip(votes, dists) if np.any(v == X)]))
        ratio = d_out / max(d_self, 1e-9)
        if ratio > cfg.best_home_max_ratio:
            keep[p] = m
            continue
        tgt = np.flatnonzero(clust_mask & (labels == X))
        if len(tgt) < cfg.binmerge_min_contigs:
            keep[p] = m
            continue
        allm = np.concatenate([g, tgt])
        sd = np.maximum(np.median(np.abs(P[allm] - np.median(P[allm], 0)), 0) * 1.4826, 1e-3)
        cv = float(np.mean(np.abs(np.median(P[g], 0) - np.median(P[tgt], 0)) / sd))
        small, big = (g, tgt) if len(g) <= len(tgt) else (tgt, g)
        mf = _misfit(small, big, P, cfg.binmerge_misfit_z)
        if cv >= cfg.binmerge_cov_tau or mf > cfg.binmerge_max_misfit:
            keep[p] = m
            continue
        labels[g] = X
        moved += len(g)
        if log_rows is not None:
            log_rows.append((b, p, len(g), X, round(float(share), 3), round(ratio, 3), round(cv, 3)))
    return keep, moved


def subcluster_bins(labels, clust_mask, contig_ids, lengths, final_latent, mods,
                    cov_features, te_weights, cfg: ClusteringConfig, outdir: Path, gc=None):
    """
    v8.1 label-free chimera splitting (no EukCC, no references).

    For each bin with >= subcluster_min_bin_contigs contigs:
      1. leaf HDBSCAN on [TNF | COV | w_te*TE] (each block L2-normalized; TE kept
         only where _te_weight_for_bin finds it informative) -> small pieces;
      2. re-merge pieces with _merge_groups: TNF is the PRIMARY criterion
         (subcluster_tnf_sep_max), coverage only a loose guard
         (subcluster_cov_merge_tau). v8.0 used coverage alone, which split single
         genomes (uneven coverage) and kept different genera with similar
         abundance together;
      3. leaf-noise contigs join a part only with >= subcluster_noise_min_agreement
         kNN support, otherwise they are left unbinned;
      4. parts < subcluster_min_part_bp fold into the TNF-nearest large part;
         the split is applied only if >= 2 large parts remain.
    """
    if not cfg.subcluster_enabled:
        return labels
    labels = labels.copy()
    prof = _cov_shape(cov_features, cfg)
    if prof.shape[1] < 2:
        log.warning("Sub-clustering skipped: needs >= 2 coverage samples.")
        return labels
    T, tspace = _tnf_space(mods, final_latent)
    length_arr = np.array([lengths.get(c, 0) for c in contig_ids], dtype=np.int64)

    bins = sorted(set(labels[clust_mask & (labels >= 0)]))
    cand = [b for b in bins if int((clust_mask & (labels == b)).sum()) >= cfg.subcluster_min_bin_contigs]
    log.info(f"Sub-clustering: {len(cand)} of {len(bins)} bins have >= "
             f"{cfg.subcluster_min_bin_contigs} contigs -> examined (method={cfg.subcluster_method}, "
             f"merge: tnf_sep<{cfg.subcluster_tnf_sep_max} on {tspace} AND cov<{cfg.subcluster_cov_merge_tau}, "
             f"te_mode={cfg.subcluster_te_mode})")
    next_label = int(labels.max()) + 1
    report, merge_rows, sub_guard_rows, best_home_rows = [], [], [], []

    for b in cand:
        idx = np.flatnonzero(clust_mask & (labels == b))
        if mods:
            w_te, te_reason = _te_weight_for_bin(idx, mods, te_weights, cfg)
            blocks = [cosine_normalize(mods["tnf"][idx]), cosine_normalize(mods["cov"][idx])]
            if w_te > 0:
                blocks.append(w_te * cosine_normalize(mods["te"][idx]))
            X = np.hstack(blocks).astype(np.float32)
        else:
            w_te, te_reason = float("nan"), "fused_latent_fallback"
            X = cosine_normalize(final_latent[idx])

        sub = hdbscan.HDBSCAN(min_cluster_size=cfg.subcluster_min_cluster_size,
                              min_samples=cfg.subcluster_min_samples,
                              cluster_selection_method=cfg.subcluster_method,
                              metric="euclidean",
                              core_dist_n_jobs=cfg.clustering_threads).fit_predict(X)
        pieces = sorted(set(sub[sub >= 0]))
        if len(pieces) < 2:
            report.append((b, len(idx), len(pieces), 1, "kept(single_leaf_piece)", w_te, te_reason, ""))
            continue

        Tb, Pb = T[idx], prof[idx]
        piece_groups = {p: np.flatnonzero(sub == p) for p in pieces}
        n_sent = 0
        if cfg.subcluster_best_home:
            piece_groups, n_sent = _best_home(b, idx, piece_groups, labels, clust_mask, T, prof, cfg,
                                              gc=gc, log_rows=best_home_rows)
            # contigs of sent pieces are already relabelled inside _best_home
            if len(piece_groups) < 2:
                report.append((b, len(idx), len(pieces), 1,
                               f"best_home_only(sent={n_sent})", w_te, te_reason, ""))
                continue
        groups = _merge_groups(piece_groups, Tb, Pb,
                               cfg.subcluster_tnf_sep_max, cfg.subcluster_cov_merge_tau,
                               log_rows=merge_rows, tag=f"sub:{b}",
                               max_misfit=cfg.subcluster_max_misfit, misfit_z=cfg.binmerge_misfit_z,
                               min_tnf_mix=None, guard_rows=sub_guard_rows,
                               gc=(gc[idx] if gc is not None else None),
                               max_gc_diff=cfg.merge_max_gc_diff,
                               gc_outside_max=cfg.merge_gc_outside_max,
                               gc_q=(cfg.merge_gc_q_lo, cfg.merge_gc_q_hi))
        if len(groups) == 1:
            report.append((b, len(idx), len(pieces), 1, "kept(pieces_merge_back_by_tnf)", w_te, te_reason, ""))
            continue

        # leaf-noise contigs -> kNN vote with minimum agreement, else unbinned
        gid = np.full(len(idx), -1, dtype=np.int64)
        for g, m in groups.items():
            gid[m] = g
        noise = np.flatnonzero(gid < 0)
        n_unassigned = 0
        if len(noise):
            assigned = np.flatnonzero(gid >= 0)
            _, nn = _knn(X[noise], X[assigned], cfg.subcluster_noise_k)
            votes = gid[assigned][nn]
            for r, row in zip(noise, votes):
                vals, cnt = np.unique(row, return_counts=True)
                if cnt.max() / len(row) >= cfg.subcluster_noise_min_agreement:
                    gid[r] = vals[np.argmax(cnt)]
                else:
                    n_unassigned += 1
        groups = {g: np.flatnonzero(gid == g) for g in groups}
        bp = {g: int(length_arr[idx[m]].sum()) for g, m in groups.items()}
        big = [g for g in groups if bp[g] >= cfg.subcluster_min_part_bp]
        if len(big) < 2:
            report.append((b, len(idx), len(pieces), 1, "kept(<2 parts >= min_part_bp)", w_te, te_reason, ""))
            continue
        for g in [g for g in groups if g not in big]:
            cg = _group_stats(groups[g], Tb, Pb)[0]
            tgt = min(big, key=lambda h: np.linalg.norm(cg - _group_stats(groups[h], Tb, Pb)[0]))
            groups[tgt] = np.concatenate([groups[tgt], groups.pop(g)])
        big.sort(key=lambda g: -len(groups[g]))
        labels[idx] = -1                      # contigs left out by the vote become unbinned
        parts = []
        for k, g in enumerate(big):
            new = b if k == 0 else next_label
            if k:
                next_label += 1
            labels[idx[groups[g]]] = new
            parts.append(f"{new}:{len(groups[g])}ctg/{int(length_arr[idx[groups[g]]].sum())/1e6:.1f}Mb")
        report.append((b, len(idx), len(pieces), len(big), f"split(unassigned={n_unassigned})",
                       w_te, te_reason, "; ".join(parts)))
        log.info(f"  bin {b}: {len(idx):,} contigs -> {len(pieces)} leaf pieces -> {len(big)} parts "
                 f"[{', '.join(parts)}] unassigned={n_unassigned} (TE w={w_te}, {te_reason})")

    with open(outdir / "subcluster_report.tsv", "w") as fh:
        fh.write("bin\tn_contigs\tn_leaf_pieces\tn_parts\tdecision\tte_weight\tte_reason\tparts\n")
        for r in report:
            fh.write("\t".join(str(x) for x in r) + "\n")
    with open(outdir / "subcluster_merge_log.tsv", "w") as fh:
        fh.write("context\tgroup_a\tgroup_b\tn_a\tn_b\ttnf_sep\tcov_dist\tmerge_allowed\n")
        for r in merge_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    with open(outdir / "best_home_log.tsv", "w") as fh:
        fh.write("from_bin\tpiece\tn_contigs\tto_bin\tneighbour_share\tdist_ratio\tcov_dist\n")
        for r in best_home_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    if best_home_rows:
        log.info(f"  best-home: {len(best_home_rows)} leaf piece(s), "
                 f"{sum(r[2] for r in best_home_rows):,} contigs moved to a closer existing bin")
    with open(outdir / "subcluster_refused.tsv", "w") as fh:
        fh.write("context\tsmaller\tlarger\tn_small\tn_large\ttnf_sep\tcov_dist\treason\n")
        for r in sub_guard_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    n_split = sum(1 for r in report if str(r[4]).startswith("split"))
    log.info(f"Sub-clustering: {n_split} bin(s) split -> subcluster_report.tsv, subcluster_merge_log.tsv")
    return labels


def merge_same_genome_bins(labels, clust_mask, final_latent, mods, cov_features,
                           cfg: ClusteringConfig, outdir: Path, gc=None, refused_out=None):
    """v8.1: rejoin bins that are fragments of one genome (TNF overlap AND coverage agreement)."""
    if not cfg.binmerge_enabled:
        return labels
    labels = labels.copy()
    prof = _cov_shape(cov_features, cfg)
    if prof.shape[1] < 2:
        log.warning("Bin merge skipped: needs >= 2 coverage samples.")
        return labels
    T, tspace = _tnf_space(mods, final_latent)
    bins = [b for b in sorted(set(labels[clust_mask & (labels >= 0)]))
            if int((clust_mask & (labels == b)).sum()) >= cfg.binmerge_min_contigs]
    if len(bins) < 2:
        return labels
    rows = []
    groups = {b: np.flatnonzero(clust_mask & (labels == b)) for b in bins}
    guard_rows = []
    merged = _merge_groups(groups, T, prof, cfg.binmerge_tnf_sep_max, cfg.binmerge_cov_tau,
                           log_rows=rows, tag="binmerge",
                           max_misfit=cfg.binmerge_max_misfit, misfit_z=cfg.binmerge_misfit_z,
                           min_tnf_mix=cfg.binmerge_min_tnf_mix, guard_rows=guard_rows,
                           gc=gc, max_gc_diff=cfg.merge_max_gc_diff,
                           gc_outside_max=cfg.merge_gc_outside_max,
                           gc_q=(cfg.merge_gc_q_lo, cfg.merge_gc_q_hi))
    with open(outdir / "binmerge_refused.tsv", "w") as fh:
        fh.write("context\tsmaller\tlarger\tn_small\tn_large\ttnf_sep\tcov_dist\treason\n")
        for r in guard_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    if guard_rows:
        log.info(f"  bin merge guards refused {len(guard_rows)} candidate merge(s) -> binmerge_refused.tsv")
    if refused_out is not None:
        refused_out.extend(guard_rows)
    n_merges = 0
    for g, m in merged.items():
        absorbed = sorted(set(labels[m]) - {g})
        if absorbed:
            n_merges += len(absorbed)
            log.info(f"  bin merge: {absorbed} -> {g}")
            for a in absorbed:
                labels[labels == a] = g   # includes any non-eligible members of the absorbed bin
                if refused_out is not None:
                    refused_out.append(("absorbed", a, g))
    with open(outdir / "binmerge_log.tsv", "w") as fh:
        fh.write("context\tbin_a\tbin_b\tn_a\tn_b\ttnf_sep\tcov_dist\tmerge_allowed\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    log.info(f"Bin merge ({tspace}): {n_merges} bin(s) absorbed -> binmerge_log.tsv")
    return labels


def prune_coverage_outliers(labels, clust_mask, cov_features, cfg: ClusteringConfig,
                            return_pruned: bool = False):
    """v8.1: unbin contigs whose coverage SHAPE across samples contradicts their bin.
    v8.2: optionally also returns the indices of the pruned contigs (for reassignment)."""
    if not cfg.prune_enabled:
        return (labels, np.array([], dtype=np.int64)) if return_pruned else labels
    labels = labels.copy()
    prof = _cov_profiles(cov_features, cfg)
    if prof.shape[1] < 3:
        log.warning("Coverage pruning skipped: needs >= 3 coverage samples.")
        return (labels, np.array([], dtype=np.int64)) if return_pruned else labels
    shape = prof - prof.mean(1, keepdims=True)
    total = 0
    pruned = []
    for b in sorted(set(labels[clust_mask & (labels >= 0)])):
        idx = np.flatnonzero(clust_mask & (labels == b))
        if len(idx) < cfg.prune_min_bin_contigs:
            continue
        S = shape[idx]
        med = np.median(S, 0)
        sd = np.maximum(np.median(np.abs(S - med), 0) * 1.4826, 0.05)
        score = (np.abs(S - med) / sd).mean(1)
        flag = score > cfg.prune_shape_z
        frac = flag.mean()
        if not flag.any():
            continue
        if frac > cfg.prune_max_frac:
            log.info(f"  prune: bin {b} would lose {frac:.1%} (> {cfg.prune_max_frac:.0%}) -- skipped")
            continue
        labels[idx[flag]] = -1
        pruned.append(idx[flag])
        total += int(flag.sum())
    log.info(f"Coverage-shape pruning: {total:,} contig(s) unbinned (z>{cfg.prune_shape_z})")
    if return_pruned:
        return labels, (np.concatenate(pruned) if pruned else np.array([], dtype=np.int64))
    return labels


def reassign_pruned_contigs(labels, pruned_idx, clust_mask, final_latent, mods, cov_features,
                            cfg: ClusteringConfig):
    """v8.2: move pruned contigs to the bin they fit (TNF kNN vote + coverage-shape z)."""
    if not cfg.reassign_enabled or len(pruned_idx) == 0:
        return labels
    labels = labels.copy()
    T, _ = _tnf_space(mods, final_latent)
    P = _cov_shape(cov_features, cfg)
    ref = np.flatnonzero(clust_mask & (labels >= 0))
    if len(ref) < cfg.reassign_k:
        return labels
    models = {b: _shape_model(P[np.flatnonzero(clust_mask & (labels == b))])
              for b in np.unique(labels[ref])}
    _, nn = _knn(T[pruned_idx], T[ref], cfg.reassign_k)
    votes = labels[ref][nn]
    moved = 0
    for r, row in zip(pruned_idx, votes):
        vals, cnt = np.unique(row, return_counts=True)
        cands = vals[cnt >= cfg.reassign_min_votes]
        if not len(cands):
            continue
        zs = [float(_shape_z(P[r:r + 1], models[b])[0]) for b in cands]
        best = int(np.argmin(zs))
        if zs[best] <= cfg.reassign_max_z:
            labels[r] = cands[best]
            moved += 1
    log.info(f"Pruned-contig reassignment: {moved:,}/{len(pruned_idx):,} moved to a better-fitting bin "
             f"(TNF votes>={cfg.reassign_min_votes}/{cfg.reassign_k}, shape z<={cfg.reassign_max_z})")
    return labels


# =============================================================================
# EUKCC LAUNCHER (robust): finds a working way to run `eukcc` even when `conda` is not on PATH
# =============================================================================
_EUKCC_ARGV_CACHE: Dict[str, List[str]] = {}


def _find_conda() -> Optional[str]:
    """conda/mamba executable: $CONDA_EXE, PATH, common install dirs, then dirs near this Python."""
    cands = [os.environ.get("CONDA_EXE", ""), shutil.which("conda") or "", shutil.which("mamba") or "",
             shutil.which("micromamba") or ""]
    home = Path.home()
    cands += [str(home / d / "bin" / "conda") for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge")]
    cands += ["/opt/conda/bin/conda", "/usr/local/miniconda3/bin/conda", "/opt/miniconda3/bin/conda"]
    cands += [str(p / "bin" / "conda") for p in list(Path(sys.prefix).parents)[:3]]
    for c in cands:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def _eukcc_env_prefix(env_name: str) -> Optional[Path]:
    """Prefix of the EukCC conda env found as a SIBLING of this Python's env (e.g. .../envs/classify -> .../envs/eukcc22)
    or under common conda install dirs; None if not found."""
    bases = [Path(sys.prefix).parent]
    home = Path.home()
    bases += [home / d / "envs" for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge")]
    bases += [Path("/opt/conda/envs")]
    for b in bases:
        p = b / env_name
        if (p / "bin" / "eukcc").is_file():
            return p
    return None


def _eukcc_candidates(cfg) -> List[List[str]]:
    env = cfg.eukcc_conda_env
    cands: List[List[str]] = []
    exe = os.environ.get("HYPHAESBIN_EUKCC", "").strip()
    if exe and Path(exe).is_file():
        # explicit executable; put its own bin dir first on PATH so EukCC finds its hmmer/pplacer/etc.
        cands.append(["env", f"PATH={Path(exe).parent}:{os.environ.get('PATH', '')}", exe])
    conda = _find_conda()
    prefix = _eukcc_env_prefix(env)
    if conda:
        cands.append([conda, "run", "-n", env, "eukcc"])
        if prefix is not None:
            cands.append([conda, "run", "-p", str(prefix), "eukcc"])
    if prefix is not None:
        cands.append(["env", f"PATH={prefix / 'bin'}:{os.environ.get('PATH', '')}", str(prefix / "bin" / "eukcc")])
    return cands


def _eukcc_argv(cfg) -> List[str]:
    """argv prefix that runs `eukcc` (append the subcommand). First candidate whose `eukcc --help` works; [] if none
    works (callers then skip EukCC loudly instead of crashing). Order: $HYPHAESBIN_EUKCC (explicit executable) ->
    `conda run -n <eukcc_conda_env>` -> `conda run -p <sibling env>` -> the sibling env's own eukcc with its bin on PATH.
    Result is cached per environment name."""
    key = f"{cfg.eukcc_conda_env}|{os.environ.get('HYPHAESBIN_EUKCC', '')}"
    if key in _EUKCC_ARGV_CACHE:
        return list(_EUKCC_ARGV_CACHE[key])
    chosen: List[str] = []
    tried = []
    for cand in _eukcc_candidates(cfg):
        try:
            r = subprocess.run(cand + ["--help"], capture_output=True, text=True, timeout=300)
            ok = r.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as e:
            ok, r = False, None
            tried.append(f"{' '.join(cand[:4])}: {e}")
            continue
        if ok:
            chosen = cand
            break
        tried.append(f"{' '.join(cand[:4])} -> rc={r.returncode}: {((r.stderr or r.stdout or '').strip().splitlines() or ['no output'])[-1][:120]}")
    if chosen:
        log.info(f"EukCC launcher: {' '.join(chosen[:4])}{' ...' if len(chosen) > 4 else ''}")
    else:
        log.warning(f"FLAG:EUKCC_NOT_RUNNABLE -- no working way to run `eukcc` (env={cfg.eukcc_conda_env!r}). "
                    f"Tried: {tried or 'nothing (no conda and no sibling env found)'}. "
                    f"Fix: set HYPHAESBIN_EUKCC=/path/to/<eukcc env>/bin/eukcc, or put conda on PATH.")
    _EUKCC_ARGV_CACHE[key] = list(chosen)
    return list(chosen)


def _eukcc_score_fasta(fasta: Path, cfg: ClusteringConfig, work: Path, use_suffix: bool):
    """Run `eukcc single` on one FASTA; returns (completeness, contamination) or None."""
    out = work / f"out_{fasta.stem}"
    cmd = [*_eukcc_argv(cfg), "single",
           "--db", cfg.eukcc_db, "--out", str(out), "--threads", str(cfg.clustering_threads)]
    if use_suffix:
        cmd += ["--suffix", ".fa"]
    cmd.append(str(fasta))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work))
    except OSError as e:
        log.warning(f"  marker check: EukCC could not start ({e})")
        return None
    res = next((p for p in (out / "eukcc.csv", out / "eukcc.tsv") if p.exists()), None)
    if r.returncode != 0 or res is None:
        tail = " | ".join(((r.stderr or r.stdout or "").strip().splitlines() or ["<no output>"])[-2:])
        log.warning(f"  marker check: EukCC failed on {fasta.name} (rc={r.returncode}): {tail}")
        return None
    rows = _read_eukcc_csv(str(res))
    if not rows:
        return None
    best = max(rows, key=lambda x: x["completeness"])
    return best["completeness"], best["contamination"]


def marker_checked_merge(labels, refused, clust_mask, contig_ids, sequences, lengths,
                         cfg: ClusteringConfig, outdir: Path):
    """v8.6: re-examine bin merges that were refused ONLY by a GC guard, using EukCC markers."""
    if not cfg.marker_merge_enabled:
        return labels
    if not cfg.eukcc_db or not _eukcc_argv(cfg):
        log.warning("Marker-checked merge skipped: eukcc_db not set or conda not on PATH.")
        return labels
    import tempfile, hashlib as _h
    labels = labels.copy()
    alias = {}
    for r in refused:
        if r[0] == "absorbed":
            alias[r[1]] = r[2]
    def resolve(x):
        seen = set()
        while x in alias and x not in seen:
            seen.add(x); x = alias[x]
        return x
    L = np.array([lengths.get(c, 0) for c in contig_ids])
    pairs, seenp = [], set()
    for r in refused:
        if r[0] == "absorbed" or not str(r[7]).startswith("gc_"):
            continue
        a, b_ = resolve(r[1]), resolve(r[2])
        key = tuple(sorted((int(a), int(b_))))
        if a == b_ or key in seenp:
            continue
        seenp.add(key); pairs.append((float(r[5]), key))
    pairs.sort()
    pairs = pairs[:cfg.marker_max_checks]
    if not pairs:
        log.info("Marker-checked merge: no GC-refused candidate pairs")
        return labels
    root = Path(tempfile.mkdtemp(prefix="hyphaes_marker_", dir=cfg.eukcc_scratch_dir or None))
    try:
        h = subprocess.run([*_eukcc_argv(cfg), "single", "--help"],
                           capture_output=True, text=True)
        use_suffix = "--suffix" in ((h.stdout or "") + (h.stderr or ""))
    except OSError:
        use_suffix = False
    cache = {}
    def score(members):
        key = _h.sha1(np.sort(members).tobytes()).hexdigest()[:16]
        if key in cache:
            return cache[key]
        fa = root / f"m_{key}.fa"
        with open(fa, "w") as fh:
            for i in members:
                fh.write(f">{contig_ids[i]}\n{sequences[contig_ids[i]]}\n")
        cache[key] = _eukcc_score_fasta(fa, cfg, root, use_suffix)
        fa.unlink(missing_ok=True)
        return cache[key]
    rows, n_merged = [], 0
    log.info(f"Marker-checked merge: {len(pairs)} GC-refused candidate pair(s) -> EukCC single "
             f"(this can take a while)")
    for _, (a, b_) in pairs:
        a, b_ = resolve(a), resolve(b_)
        if a == b_:
            continue
        ma = np.flatnonzero(labels == a); mb = np.flatnonzero(labels == b_)
        if not len(ma) or not len(mb) or L[ma].sum() < cfg.marker_min_bp or L[mb].sum() < cfg.marker_min_bp:
            continue
        sa, sb = score(ma), score(mb)
        sab = score(np.concatenate([ma, mb])) if (sa and sb) else None
        if not (sa and sb and sab):
            rows.append((a, b_, sa, sb, sab, "no_score")); continue
        gain = sab[0] - max(sa[0], sb[0])
        dcont = sab[1] - max(sa[1], sb[1])
        ok = gain >= cfg.marker_min_gain and dcont <= cfg.marker_max_cont_increase
        rows.append((a, b_, sa, sb, sab, "MERGED" if ok else f"refused(gain={gain:.1f},dcont={dcont:.1f})"))
        log.info(f"  marker check bins {a}+{b_}: compl {sa[0]:.1f}/{sb[0]:.1f} -> {sab[0]:.1f}, "
                 f"cont {sa[1]:.1f}/{sb[1]:.1f} -> {sab[1]:.1f} => {'MERGE' if ok else 'keep apart'}")
        if ok:
            keep, drop = (a, b_) if len(ma) >= len(mb) else (b_, a)
            labels[labels == drop] = keep
            alias[drop] = keep
            n_merged += 1
    shutil.rmtree(root, ignore_errors=True)
    with open(outdir / "marker_merge_log.tsv", "w") as fh:
        fh.write("bin_a\tbin_b\tscore_a\tscore_b\tscore_union\tdecision\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    log.info(f"Marker-checked merge: {n_merged} merge(s) accepted -> marker_merge_log.tsv")
    return labels


def tnf_consensus_reassign(labels, clust_mask, final_latent, mods, cfg: ClusteringConfig,
                           cov_features=None):
    """v8.4: move a binned contig to another bin when its TNF neighbourhood overwhelmingly
    (>= tnf_consensus_min of tnf_consensus_k) belongs to that bin. Single pass; votes are
    taken from the labels BEFORE any move, so moves cannot cascade."""
    if not cfg.tnf_consensus_enabled:
        return labels
    labels = labels.copy()
    T, tspace = _tnf_space(mods, final_latent)
    ref = np.flatnonzero(clust_mask & (labels >= 0))
    k = cfg.tnf_consensus_k
    if len(ref) <= k:
        return labels
    try:
        from scipy.spatial import cKDTree
        _, nn = cKDTree(T[ref]).query(T[ref], k=k + 1)
        nn = nn[:, 1:]
    except ImportError:
        _, nn = _exact_neighbors_numpy(T[ref], T[ref], k + 1)
        nn = nn[:, 1:]
    P = _cov_shape(cov_features, cfg) if cov_features is not None else None
    moved = collections.Counter(); blocked = 0
    for it in range(max(1, cfg.tnf_consensus_iters)):
        before = labels.copy()           # votes use the labels from the start of this pass
        votes = before[ref][nn]
        models = ({b: _shape_model(P[ref[before[ref] == b]]) for b in np.unique(before[ref])}
                  if P is not None else None)
        n_it = 0
        for row, r in enumerate(ref):
            vals, cnt = np.unique(votes[row], return_counts=True)
            j = int(np.argmax(cnt))
            if vals[j] != before[r] and cnt[j] / k >= cfg.tnf_consensus_min:
                if models is not None and float(_shape_z(P[r:r + 1], models[vals[j]])[0]) > cfg.tnf_consensus_max_z:
                    blocked += 1
                    continue
                labels[r] = vals[j]
                moved[(int(before[r]), int(vals[j]))] += 1
                n_it += 1
        if n_it == 0:
            break
    if blocked:
        log.info(f"  TNF-consensus: {blocked:,} candidate move(s) blocked by the coverage-fit check")
    total = sum(moved.values())
    log.info(f"TNF-consensus reassignment ({tspace}, k={k}, >= {cfg.tnf_consensus_min:.0%}): "
             f"{total:,} contig(s) moved")
    for (a_, b_), c in moved.most_common(10):
        log.info(f"    bin {a_} -> bin {b_}: {c:,}")
    return labels


def recruit_short_contigs(labels, clust_mask, valid_mask, contig_ids, lengths, final_latent,
                          cfg: ClusteringConfig):
    """v8.1: strict kNN recruitment of contigs below cluster_min_contig_len."""
    if not cfg.recruit_enabled:
        return labels
    labels = labels.copy()
    L = np.array([lengths.get(c, 0) for c in contig_ids])
    q_idx = np.flatnonzero((~clust_mask) & (valid_mask > 0) & (labels == -1)
                           & (L >= cfg.recruit_min_len) & (L < cfg.cluster_min_contig_len))
    r_idx = np.flatnonzero(clust_mask & (labels >= 0))
    if not len(q_idx) or len(r_idx) < cfg.recruit_k:
        log.info("Short-contig recruitment: nothing to do")
        return labels
    x = cosine_normalize(final_latent).astype(np.float32)
    R = x[r_idx]; r2 = (R * R).sum(1); rlab = labels[r_idx]
    k = cfg.recruit_k
    got = 0
    for s0 in range(0, len(q_idx), 1024):
        qi = q_idx[s0:s0 + 1024]; Q = x[qi]
        d2 = np.maximum((Q * Q).sum(1)[:, None] + r2[None, :] - 2 * Q @ R.T, 0)
        nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dd = np.sqrt(np.take_along_axis(d2, nn, 1))
        near = dd <= cfg.recruit_max_dist
        nl = rlab[nn]
        for row in range(len(qi)):
            sel = nl[row][near[row]]
            if len(sel) < cfg.recruit_min_support:
                continue
            vals, cnt = np.unique(sel, return_counts=True)
            if cnt.max() / len(sel) >= cfg.recruit_agreement:
                labels[qi[row]] = vals[np.argmax(cnt)]
                got += 1
    log.info(f"Short-contig recruitment ({cfg.recruit_min_len}-{cfg.cluster_min_contig_len - 1}bp): "
             f"{got:,}/{len(q_idx):,} assigned")
    return labels


def filter_bins(labels, contig_ids, lengths, cfg: ClusteringConfig):
    """
    NOTE (v7.8, documentation only -- no behavior change): this runs AFTER
    assign_noise_contigs() and iterative_refinement(), so a small, pure
    HDBSCAN cluster that fails these thresholds (default
    min_bin_length_bp=500,000) is discarded here and becomes unbinned in
    the final output -- it does NOT get a second chance at rescue, because
    rescue has already happened by this point in the pipeline. This is
    intended behavior (a genuinely too-small fragment should not count as
    a "bin"), but it does mean cfg.min_bin_length_bp directly trades off
    against total recall: a stricter threshold discards more small, real,
    pure fragments as unbinned rather than merging or rescuing them.
    Lowering it will recover more small bins into the final output (at the
    cost of potentially noisier, less genome-representative "bins"); it
    does not, by itself, fix cluster fragmentation upstream (see
    hdbscan_epsilon / hdbscan_method for that).
    """
    log.info("Filtering low-quality bins...")
    labels      = labels.copy()
    unique_bins = sorted(set(labels[labels >= 0]))
    kept = []; removed = []
    reasons = collections.Counter()

    for b in unique_bins:
        bin_idx  = np.where(labels == b)[0]
        bin_ctgs = [contig_ids[i] for i in bin_idx]
        ctg_lens = [lengths.get(c, 0) for c in bin_ctgs]
        total_len = sum(ctg_lens)
        n_ctg     = len(bin_ctgs)
        n50       = compute_n50(ctg_lens)

        reason = None
        if n_ctg < cfg.min_contigs_per_bin:
            reason = f"too_few_contigs ({n_ctg}<{cfg.min_contigs_per_bin})"
        elif total_len < cfg.min_bin_length_bp:
            reason = f"too_short ({total_len}<{cfg.min_bin_length_bp}bp)"
        elif n50 < cfg.min_bin_n50_bp:
            reason = f"low_n50 ({n50}<{cfg.min_bin_n50_bp}bp)"

        if reason:
            labels[labels == b] = -1
            removed.append(b)
            reasons[reason] += 1
        else:
            kept.append(b)

    log.info(f"Bins kept: {len(kept)} | Removed: {len(removed)}")
    for reason, count in reasons.most_common():
        log.info(f"  Removed reason: {reason} x {count}")
    return labels, kept, removed


# =============================================================================
# STEP 6.5 -- DENSE BIN-ID ASSIGNMENT (FIX 5)
# =============================================================================

def assign_dense_bin_ids(labels: np.ndarray, bin_order: List[int]) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Assigns ONE dense, sequential bin-ID string (bin_001, bin_002, ...) per
    raw label in bin_order, and relabels `labels` (as strings, -1 stays
    "unbinned") so every downstream consumer -- FASTA filenames, TSVs,
    JSON -- uses the exact same identifier for the exact same bin. This is
    the fix for v6's disk-vs-summary bin-ID mismatch.
    """
    n_digits = max(3, len(str(len(bin_order))))
    raw_to_dense = {b: f"bin_{i:0{n_digits}d}" for i, b in enumerate(bin_order, 1)}
    dense_labels = np.array(
        [raw_to_dense.get(int(b), "unbinned") if b >= 0 else "unbinned" for b in labels],
        dtype=object,
    )
    return dense_labels, raw_to_dense


# =============================================================================
# STEP 7 -- WRITE BIN FASTAs (dense IDs, absolute-path .fa symlinks)
# =============================================================================

def _remove_obsolete_bins(directory, current_ids):
    """Unlink only obsolete generated bin FASTAs inside the exact output dir."""
    directory = Path(directory).resolve()
    expected = {str(b) + ext for b in current_ids for ext in (".fa", ".fasta")}
    for item in list(directory.glob("bin_*.fa")) + list(directory.glob("bin_*.fasta")):
        if item.name not in expected and (item.is_file() or item.is_symlink()):
            item.unlink()


def write_bin_fastas(dense_labels: np.ndarray, contig_ids: List[str], sequences: Dict[str, str],
                      target_dir: Path) -> Dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    bin_ids = sorted(set(dense_labels[dense_labels != "unbinned"]))
    bin_paths: Dict[str, str] = {}

    # FIX 43 (v7.8): stale bin files from a PRIOR run of this same
    # target_dir are now removed before writing this run's bins. Previously
    # this function only ever wrote/overwrote the bin_ids present in THIS
    # run's dense_labels -- if a prior run in the same output directory had
    # produced a different set of bin IDs (e.g. more bins, from a looser
    # config), those old .fasta/.fa files were simply left behind. Any tool
    # that scans the directory rather than reading an explicit manifest
    # (e.g. `eukcc folder` on this path, or a shell glob) would then see a
    # mix of this run's real output and stale leftovers from an earlier,
    # possibly-different clustering result -- corrupting any diagnostic or
    # benchmark run against that directory. Reusing an output directory
    # across genuinely different clustering runs is still not recommended
    # (prefer a fresh directory per run), but this at minimum prevents the
    # silent mixing of old and new bin files within one directory.
    current_names = {f"{b}.fasta" for b in bin_ids} | {f"{b}.fa" for b in bin_ids}
    stale_removed = 0
    for existing in list(target_dir.glob("*.fasta")) + list(target_dir.glob("*.fa")):
        if existing.name not in current_names:
            existing.unlink()
            stale_removed += 1
    if stale_removed:
        log.info(f"Removed {stale_removed} stale bin file(s) from a previous run in {target_dir}")

    for bin_id in bin_ids:
        # NOT resolved yet: target_dir isn't itself a symlink, but fa_path
        # from a PRIOR run of this same function IS one (it's what this
        # function creates below). Calling .resolve() on fa_path here would
        # follow that stale symlink and collapse it onto fasta_path's real
        # target, so a rerun's unlink+relink would delete the just-written
        # real FASTA and replace it with a self-referential
        # bin_NNN.fasta -> bin_NNN.fasta symlink (found by this module's own
        # test harness on a second/resumed run -- a real, reproduced bug,
        # not a hypothetical one). Fix: remove any pre-existing .fa symlink
        # BEFORE resolving anything, then resolve only the (always-real)
        # .fasta path once that's safe.
        fasta_path = target_dir / f"{bin_id}.fasta"
        fa_path    = target_dir / f"{bin_id}.fa"
        idx = np.where(dense_labels == bin_id)[0]

        if fa_path.exists() or fa_path.is_symlink():
            fa_path.unlink()

        with open(fasta_path, "w") as f:
            for i in idx:
                name = contig_ids[i]
                seq  = sequences.get(name, "")
                if seq:
                    f.write(f">{name}\n")
                    for j in range(0, len(seq), 60):
                        f.write(seq[j:j + 60] + "\n")

        fasta_path = fasta_path.resolve()
        fa_path.symlink_to(fasta_path.name)   # RELATIVE link: survives renaming/moving the folder
        bin_paths[bin_id] = str(fasta_path)

    log.info(f"Wrote {len(bin_ids)} bin(s) -> {target_dir}")
    return bin_paths


# =============================================================================
# STEP 8 -- SKANI DEDUPLICATION (unchanged core logic, dense-ID aware)
# =============================================================================

def run_skani_dedup(bin_paths: Dict[str, str], lengths: Dict[str, int],
                     contig_ids: List[str], dense_labels: np.ndarray,
                     outdir: Path, cfg: ClusteringConfig, label: str = "dedup"):
    """
    Removes redundant bins (ANI >= threshold), keeping the longest bin from
    each redundant group. Returns (surviving_bin_ids, redundant_groups)
    where redundant_groups maps a survivor bin_id -> list of removed
    bin_ids it absorbed (for cluster_summary.tsv's status column and
    clustering_manifest.json's evidence).
    """
    if not shutil.which("skani"):
        log.warning("FLAG:SKANI_NOT_FOUND -- skipping deduplication")
        return list(bin_paths.keys()), {}
    if len(bin_paths) < 2:
        log.info(f"[{label}] Only {len(bin_paths)} bin(s) -- skipping dedup")
        return list(bin_paths.keys()), {}

    log.info(f"[{label}] skani dedup (ANI >= {cfg.skani_ani_threshold}%)...")
    skani_dir   = outdir / "skani" / label
    skani_dir.mkdir(parents=True, exist_ok=True)
    list_file   = skani_dir / "bin_list.txt"
    result_file = skani_dir / "skani_results.tsv"

    with open(list_file, "w") as f:
        for path in bin_paths.values():
            f.write(path + "\n")

    cmd = ["skani", "dist", "--ql", str(list_file), "--rl", str(list_file),
           "-o", str(result_file), "-t", str(cfg.clustering_threads), "--min-af", str(cfg.skani_min_af)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        log.warning(f"[{label}] skani could not be launched ({e}) -- skipping dedup")
        return list(bin_paths.keys()), {}
    if r.returncode != 0:
        log.warning(f"[{label}] skani failed: {r.stderr[:200]} -- skipping dedup")
        return list(bin_paths.keys()), {}

    path_to_bin = {v: k for k, v in bin_paths.items()}
    redundant = collections.defaultdict(set)
    if result_file.exists():
        with open(result_file) as f:
            next(f, None)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                try:
                    ani = float(parts[2])
                    af_ref, af_query = float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                if ani >= cfg.skani_ani_threshold and min(af_ref, af_query) >= cfg.skani_min_af:
                    q, r_ = parts[0], parts[1]
                    if q != r_:
                        redundant[q].add(r_)
                        redundant[r_].add(q)

    bin_quality = {}
    for bin_id, path in bin_paths.items():
        idx = np.where(dense_labels == bin_id)[0]
        bin_quality[path] = sum(lengths.get(contig_ids[i], 0) for i in idx)

    all_paths = set(bin_paths.values())
    removed   = set()
    redundant_groups: Dict[str, List[str]] = {}

    # Use conservative pairwise deduplication. A transitive connected-component
    # rule (A~B and B~C => remove A/B/C as one group) can discard C even when
    # A and C are not directly redundant. Keep the longest bin only among bins
    # directly redundant with the selected survivor.
    remaining = set(all_paths)
    while remaining:
        best = max(remaining, key=lambda p: (bin_quality.get(p, 0), p))
        direct = set(redundant.get(best, set())) & remaining
        if direct:
            absorbed = sorted(path_to_bin[p] for p in direct)
            redundant_groups[path_to_bin[best]] = absorbed
            remaining.remove(best)
            remaining.difference_update(direct)
            removed.update(direct)
        else:
            remaining.remove(best)

    surviving = [path_to_bin[p] for p in all_paths if p not in removed]
    log.info(f"[{label}] skani: {len(all_paths)} -> {len(surviving)} bins "
             f"({len(removed)} removed as redundant)")
    return surviving, redundant_groups


# =============================================================================
# STEP 9 -- EUKCC (quality assessment)
# =============================================================================

def _eukcc_input_fingerprint(bins_dir: Path, cfg: ClusteringConfig) -> str:
    """
    FIX 31 (v7.6): fingerprint for EukCC's own cache check. Hashes the
    sorted list of (bin filename, size, mtime) for every .fasta in
    bins_dir, plus cfg.eukcc_db and cfg.eukcc_conda_env -- so a rerun with
    different/changed bins, a different database, or a different EukCC env
    is detected as a cache miss instead of reusing a stale eukcc.csv that
    was computed for different input. Same size+mtime philosophy as
    _file_fingerprint() elsewhere in this module (not a content hash --
    same disclosed limitation).
    """
    entries = []
    for p in sorted(bins_dir.glob("*.fasta")):
        try:
            st = p.stat()
            entries.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
        except FileNotFoundError:
            continue
    return _fingerprint(entries, cfg.eukcc_db, cfg.eukcc_conda_env, cfg.eukcc_single_per_bin, VERSION)


def _prepare_eukcc_bins(
    bins_dir: Path,
    outdir: Path,
    cfg: ClusteringConfig,
    contig_classification: Optional[Dict[str, str]],
    label: str = "eukcc",
) -> Tuple[Path, Set[str]]:
    """Create the exact EukCC input directory and return included bin IDs.

    Classification labels are normalized so upstream tools may use eukaryote,
    eukaryotic, or Eukaryota. The same filtered directory is used for quality
    assessment and merging, preventing excluded prokaryotic bins from being
    reintroduced during the merge stage.
    """
    all_ids = {p.stem for p in bins_dir.glob("*.fasta")}
    if contig_classification is None or cfg.eukcc_min_eukaryote_fraction <= 0:
        return bins_dir, all_ids

    normalized = {
        str(k).split()[0]: str(v).strip().lower()
        for k, v in contig_classification.items()
    }
    euk_labels = {"eukaryote", "eukaryotic", "eukaryota", "euk"}
    candidate_dir = outdir / f"_working_{label}_eukaryote_filtered"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    included: Set[str] = set()
    excluded_bins = []
    for fasta in sorted(bins_dir.glob("*.fasta")):
        names = []
        with open(fasta) as f:
            for line in f:
                if line.startswith(">"):
                    names.append(line[1:].strip().split()[0])
        if not names:
            continue
        n_euk = sum(1 for n in names if normalized.get(n, "") in euk_labels)
        euk_frac = n_euk / len(names)
        if euk_frac >= cfg.eukcc_min_eukaryote_fraction:
            dst = candidate_dir / fasta.name
            dst.symlink_to(fasta.resolve())
            included.add(fasta.stem)
        else:
            excluded_bins.append((fasta.stem, round(euk_frac, 3)))

    if excluded_bins:
        log.info(
            f"[{label}] Eukaryote-fraction pre-filter: excluded {len(excluded_bins)} "
            f"bin(s), threshold={cfg.eukcc_min_eukaryote_fraction}: "
            f"{excluded_bins[:10]}"
            f"{', ...' if len(excluded_bins) > 10 else ''}"
        )
    return candidate_dir, included

def run_eukcc(bins_dir: Path, outdir: Path, cfg: ClusteringConfig, label: str = "eukcc",
              contig_classification: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    contig_classification (FIX 32, v7.6, OPTIONAL): an optional {contig_id:
    "eukaryote"|"prokaryote"|...} map (e.g. from preprocessing's Tiara/
    Whokaryote voting trio). When supplied, bins whose eukaryotic-contig
    fraction (by count) falls below cfg.eukcc_min_eukaryote_fraction are
    excluded from the EukCC input folder before running -- EukCC's marker
    sets are eukaryote-only, so scoring a predominantly bacterial/archaeal
    bin with it produces a biologically meaningless "low completeness"
    result rather than a real quality signal. When NOT supplied (the
    default, since this module does not itself compute contig
    classification), this filter is cleanly skipped with a logged reason,
    exactly matching this module's existing pattern for other optional
    inputs (e.g. alignment_bam_paths for EukCC merging) -- never a silent
    no-op, never a guess.
    """
    if not cfg.run_eukcc:
        log.info("EukCC: skipped (run_eukcc=false in config)")
        return None
    if not cfg.eukcc_db:
        log.warning(f"[{label}] FLAG:EUKCC_DB_NOT_SET -- run_eukcc=True but config['eukcc_db'] "
                    f"is empty -- skipping EukCC rather than launching `eukcc --db \"\"`.")
        return None
    if not _eukcc_argv(cfg):
        log.warning(f"[{label}] FLAG:EUKCC_NOT_RUNNABLE -- cannot run EukCC (env={cfg.eukcc_conda_env!r}); see the EUKCC_NOT_RUNNABLE line above for what was tried -- skipping.")
        return None

    # Apply the same filtered directory for EukCC scoring and merging.
    bins_dir, _included_ids = _prepare_eukcc_bins(
        bins_dir, outdir, cfg, contig_classification, label=label
    )
    if contig_classification is None:
        log.info(f"[{label}] Eukaryote-fraction pre-filter: skipped -- no contig_classification "
                 f"supplied to run_eukcc() (this module does not compute classification itself; "
                 f"pass it in from upstream preprocessing to enable this filter).")

    eukcc_out = outdir / label
    eukcc_out.mkdir(parents=True, exist_ok=True)
    eukcc_csv = eukcc_out / "eukcc.csv"
    eukcc_fp_path = eukcc_out / "eukcc_input_fingerprint.txt"

    # FIX 31 (v7.6): existence-only caching replaced with a real input
    # fingerprint check. Previously ANY existing eukcc.csv at this path was
    # reused regardless of whether the bins, database, or EukCC env had
    # changed since it was written -- silently attaching stale quality
    # values to new bins. Now the fingerprint of the current input is
    # compared against the one recorded alongside the cached eukcc.csv;
    # a mismatch is treated as a cache miss (rerun), matching the
    # _checkpoint_ok() pattern used elsewhere in this module.
    current_fp = _eukcc_input_fingerprint(bins_dir, cfg)
    if cfg.resume and eukcc_csv.exists() and eukcc_fp_path.exists():
        if eukcc_fp_path.read_text().strip() == current_fp:
            log.info(f"[{label}] EukCC [CACHED] (input fingerprint matches)")
            return str(eukcc_csv)
        else:
            log.info(f"[{label}] EukCC cache present but input fingerprint changed -- rerunning.")

    # FIX 44 (v7.8): opt-in per-bin `eukcc single` path. Resolves the
    # "still calls eukcc folder despite a stated intent to use per-bin
    # scoring" gap: when cfg.eukcc_single_per_bin is True, this bypasses
    # `eukcc folder` (and therefore its unremovable internal merge-search
    # behavior) entirely, running `eukcc single` independently on each
    # bin's own FASTA and assembling the results into the same eukcc.csv
    # schema the rest of this module expects, so no downstream code needs
    # to know which path was used.
    if cfg.eukcc_single_per_bin:
        log.info(f"[{label}] Running EukCC in per-bin 'single' mode "
                 f"(env={cfg.eukcc_conda_env}) -- guarantees unmerged, per-bin numbers...")
        t0 = time.time()
        fasta_files = sorted(bins_dir.glob("*.fasta"))
        rows: List[Dict[str, str]] = []
        any_failed = False
        eukcc_fp_path.unlink(missing_ok=True)
        eukcc_csv.unlink(missing_ok=True)
        # v8.1: run in a scratch dir (Bus error seen when EukCC wrote to /DATA_LUN), with a
        # NOT-yet-existing --out path (stale/pre-created out dirs broke EukCC before), --suffix
        # only if this EukCC build supports it, and the stderr tail logged on failure.
        import tempfile
        scratch_root = Path(cfg.eukcc_scratch_dir or tempfile.gettempdir())
        scratch_root.mkdir(parents=True, exist_ok=True)
        keep_root = eukcc_out / "single"
        keep_root.mkdir(parents=True, exist_ok=True)
        try:
            h = subprocess.run([*_eukcc_argv(cfg), "single", "--help"],
                               capture_output=True, text=True)
            single_help = (h.stdout or "") + (h.stderr or "")
        except OSError:
            single_help = ""
        use_suffix = "--suffix" in single_help
        for fasta in fasta_files:
            work = Path(tempfile.mkdtemp(prefix=f"eukcc_{fasta.stem}_", dir=str(scratch_root)))
            bin_out = work / "out"                      # must not exist before EukCC runs
            src = work / f"{fasta.stem}.fa"
            shutil.copyfile(fasta.resolve(), src)       # real file, no symlink, on scratch
            single_cmd = [*_eukcc_argv(cfg), "single",
                          "--db", cfg.eukcc_db, "--out", str(bin_out),
                          "--threads", str(cfg.clustering_threads)]
            if use_suffix:
                single_cmd += ["--suffix", ".fa"]
            single_cmd.append(str(src))
            try:
                r = subprocess.run(single_cmd, capture_output=True, text=True, cwd=str(work))
            except OSError as e:
                log.warning(f"[{label}] EukCC single could not be launched for {fasta.name} ({e})")
                any_failed = True
                shutil.rmtree(work, ignore_errors=True)
                continue
            dest = keep_root / fasta.stem
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "eukcc_run.log").write_text((r.stdout or "") + (r.stderr or ""))
            bin_csv = next((p for p in (bin_out / "eukcc.csv", bin_out / "eukcc.tsv") if p.exists()), None)
            if bin_csv is not None:
                shutil.copyfile(bin_csv, dest / bin_csv.name)
            shutil.rmtree(work, ignore_errors=True)
            if r.returncode != 0 or bin_csv is None:
                tail = " | ".join(((r.stderr or r.stdout or "").strip().splitlines() or ["<no output>"])[-3:])
                log.warning(f"[{label}] EukCC single failed for {fasta.name} (rc={r.returncode}): {tail}")
                any_failed = True
                continue
            bin_rows = _read_eukcc_csv(str(dest / bin_csv.name))
            if bin_rows:
                for row in bin_rows:
                    row["bin"] = fasta.stem
                rows.extend(bin_rows)
            else:
                any_failed = True
        if not rows:
            log.warning(f"[{label}] EukCC single mode produced no usable results for any bin.")
            return None
        # Assemble into the same flat CSV schema `eukcc folder` produces,
        # so _read_eukcc_csv() and every downstream consumer works
        # unmodified regardless of which mode was used.
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(eukcc_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        if not any_failed:
            eukcc_fp_path.write_text(current_fp)
        status = "all bins succeeded" if not any_failed else "some bins failed, see log"
        log.info(f"[{label}] EukCC single-per-bin done: {time.time()-t0:.0f}s, "
                 f"{len(rows)} bin(s) scored ({status}) -> {eukcc_csv}")
        return str(eukcc_csv)

    log.info(f"[{label}] Running EukCC (env={cfg.eukcc_conda_env})...")
    t0  = time.time()
    cmd = [*_eukcc_argv(cfg), "folder",
           "--db", cfg.eukcc_db, "--out", str(eukcc_out),
           "--threads", str(cfg.clustering_threads),
           # FIX 27 (v7.5): --suffix .fasta added explicitly. EukCC's own
           # `eukcc folder` defaults to expecting bin files with a .fa
           # suffix; write_bin_fastas() writes both a real .fasta file and
           # a .fa symlink to it (for compatibility with tools expecting
           # either), so this call previously worked by relying on the .fa
           # symlink -- but that symlink was found (FIX 16) to point at a
           # stale/incorrect target after certain rerun sequences, which
           # this call had no visibility into. Being explicit about the
           # suffix and letting write_bin_fastas' FIX 16 fix keep the .fa
           # symlink correct is more robust than relying on an implicit
           # default.
           "--suffix", ".fasta",
           str(bins_dir)]
    # NOTE (v7.5, unresolved): eukcc folder appears, in practice, to
    # attempt its own internal bin-merge logic even when no --links file is
    # supplied here (observed directly: merges occurred in manual testing
    # without --links on the command line) -- despite EukCC's own
    # documentation implying that merge candidates are identified via
    # paired-read links and would have none without --links. This
    # discrepancy has NOT been fully root-caused. Practical consequence:
    # completeness/contamination values from THIS function's eukcc.csv
    # should not be assumed to represent bins.fasta contents unadulterated
    # -- cross-check unusual bins (e.g. any appearing in this call's own
    # eukcc_out for signs of merging) against `eukcc single`, run
    # per-bin, if a guaranteed-unmerged number is required for reporting.
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(outdir))
    except OSError as e:
        log.warning(f"[{label}] EukCC could not be launched ({e}) -- skipping.")
        return None
    with open(eukcc_out / "eukcc_run.log", "w") as f:
        f.write(result.stdout); f.write(result.stderr)
    if result.returncode == 0 and (eukcc_out / "eukcc.tsv").exists():
        shutil.copyfile(eukcc_out / "eukcc.tsv", eukcc_csv)
    if result.returncode != 0 or not eukcc_csv.exists():
        log.warning(f"[{label}] EukCC failed (rc={result.returncode}) after {time.time()-t0:.0f}s")
        return None
    eukcc_fp_path.write_text(current_fp)  # FIX 31 (v7.6): record fingerprint for next cache check
    log.info(f"[{label}] EukCC done: {time.time()-t0:.0f}s -> {eukcc_csv}")
    return str(eukcc_csv)


def _read_eukcc_csv(path: str) -> Optional[List[Dict[str, str]]]:
    """
    Defensive parser: EukCC's public docs (verified before writing this)
    confirm eukcc.csv reports completeness/contamination/lineage per bin
    but do not list exact column headers. Column names are matched by
    case-insensitive substring rather than assumed literal strings; if the
    columns this needs can't be found, this returns None (caller treats
    that as "skip, can't parse" -- never silently misreads the file).
    Confirm against a real eukcc.csv in your environment and tighten this
    if you know the exact headers your EukCC version uses.
    """
    if not path or not Path(path).exists():
        return None
    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel_tab
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
    if not rows:
        return None

    cols = list(rows[0].keys())

    def find_col(*candidates):
        for c in cols:
            cl = c.lower()
            if any(cand in cl for cand in candidates):
                return c
        return None

    bin_col = find_col("bin", "genome", "fasta")
    comp_col = find_col("completeness")
    cont_col = find_col("contamination")
    if not (bin_col and comp_col and cont_col):
        log.warning(f"Could not find bin/completeness/contamination columns in {path} "
                    f"(saw columns: {cols}) -- skipping anything that depends on this file.")
        return None

    log.info(f"eukcc.csv columns matched: bin={bin_col!r} completeness={comp_col!r} "
             f"contamination={cont_col!r}")
    out = []
    for row in rows:
        try:
            out.append({
                "bin": Path(row[bin_col]).stem,
                "completeness": float(row[comp_col]),
                "contamination": float(row[cont_col]),
            })
        except (ValueError, KeyError):
            continue
    return out


# =============================================================================
# STEP 10 -- EUKCC BIN MERGING via EukCC's OWN --links MECHANISM (FIX 13)
# =============================================================================

def run_eukcc_merge(bins_dir: Path, bin_paths: Dict[str, str], outdir: Path,
                     cfg: ClusteringConfig, alignment_bam_paths: Optional[List[str]]):
    """
    Wraps EukCC's real, documented merge workflow (eukcc.readthedocs.io/
    en/latest/bin_merging.html, verified before writing this function):

      1. binlinks.py --ANI <ani> --within <within> --out linktable.csv
         <bins_dir> <bam>          (once per supplied BAM; link counts are
                                     summed across BAMs -- our own
                                     extension for multi-sample data, since
                                     EukCC's docs only show a single BAM)
      2. eukcc folder --out <merge_out> --links linktable.csv
         --n_combine <cfg.eukcc_merge_n_combine> <bins_dir>

      EukCC decides internally which medium-quality, strongly-linked bins
      to merge and whether the merge improves its own quality score --
      this function does not reimplement or guess at that decision.

    Returns (merged_bin_paths: Dict[str,str] for NEWLY merged bins only,
             merge_log: List[dict] with source bins + before/after
             completeness/contamination evidence per accepted merge,
             status: str).

    Cleanly returns ({}, [], "skipped_<reason>") when merging can't run --
    no fallback heuristic is substituted.
    """
    if not cfg.eukcc_merge_enabled:
        return {}, [], "skipped_disabled_in_config"
    if not cfg.run_eukcc:
        return {}, [], "skipped_eukcc_disabled"
    if not cfg.eukcc_db:
        log.warning("EukCC merge: FLAG:EUKCC_DB_NOT_SET -- config['eukcc_db'] is empty -- "
                    "skipping merge rather than launching `eukcc --db \"\"`.")
        return {}, [], "skipped_eukcc_db_not_set"
    if not alignment_bam_paths:
        log.info("EukCC merge: no alignment_bam_paths supplied -- skipping "
                 "(this pipeline stage has no BAM input yet; see module docstring FIX 13).")
        return {}, [], "skipped_no_bam_alignment"
    if len(bin_paths) < 2:
        return {}, [], "skipped_fewer_than_2_bins"
    if not shutil.which("conda"):
        log.warning("EukCC merge: FLAG:CONDA_NOT_FOUND -- 'conda' is not on PATH -- skipping merge.")
        return {}, [], "skipped_conda_not_found"

    try:
        binlinks_check = subprocess.run(
            ["conda", "run", "-n", cfg.eukcc_conda_env, "which", "binlinks.py"],
            capture_output=True, text=True,
        )
    except OSError as e:
        log.warning(f"EukCC merge: could not check for binlinks.py ({e}) -- skipping merge.")
        return {}, [], "skipped_conda_launch_failed"
    if binlinks_check.returncode != 0:
        log.warning(f"EukCC merge: binlinks.py not found in conda env {cfg.eukcc_conda_env!r} "
                     f"-- skipping merge.")
        return {}, [], "skipped_binlinks_not_found"

    # pre-merge quality (for "before" evidence) -- reuse the eukcc.csv
    # already produced by run_eukcc() on this same bins_dir if present.
    pre_csv = outdir / "eukcc" / "eukcc.csv"
    pre_quality = {r["bin"]: r for r in (_read_eukcc_csv(str(pre_csv)) or [])}

    merge_dir = outdir / "eukcc_merge"
    merge_dir.mkdir(parents=True, exist_ok=True)

    link_tables = []
    for i, bam in enumerate(alignment_bam_paths):
        lt = merge_dir / f"linktable_{i}.csv"
        cmd = ["conda", "run", "-n", cfg.eukcc_conda_env, "binlinks.py",
               "--ANI", str(cfg.eukcc_merge_ani), "--within", str(cfg.eukcc_merge_within),
               "--out", str(lt), str(bins_dir), str(bam)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as e:
            log.warning(f"EukCC merge: binlinks.py could not be launched for {bam} ({e}) -- skipping it.")
            continue
        if r.returncode != 0 or not lt.exists():
            log.warning(f"EukCC merge: binlinks.py failed on {bam} ({r.stderr[:200]}) -- skipping it.")
            continue
        link_tables.append(lt)

    if not link_tables:
        return {}, [], "skipped_binlinks_failed"

    # Merge link tables across BAMs (sum link counts per bin pair) -- our
    # own extension beyond EukCC's single-BAM documented example.
    combined_links: Dict[Tuple[str, str], int] = collections.Counter()
    for lt in link_tables:
        with open(lt, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                b1, b2, links = row[0], row[1], row[2]
                try:
                    combined_links[(b1, b2)] += int(float(links))
                except ValueError:
                    continue
    combined_path = merge_dir / "linktable_combined.csv"
    with open(combined_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_1", "bin_2", "links"])
        for (b1, b2), links in combined_links.items():
            w.writerow([b1, b2, links])

    merge_out = merge_dir / "eukcc_out"
    cmd = [*_eukcc_argv(cfg), "folder", "--out", str(merge_out), "--threads", str(cfg.clustering_threads),
           "--db", cfg.eukcc_db, "--links", str(combined_path),
           "--n_combine", str(cfg.eukcc_merge_n_combine), str(bins_dir)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(outdir))
    except OSError as e:
        log.warning(f"EukCC merge: eukcc could not be launched ({e}) -- skipping merge.")
        return {}, [], "skipped_eukcc_launch_failed"
    with open(merge_dir / "eukcc_merge_run.log", "w") as f:
        f.write(r.stdout); f.write(r.stderr)
    if r.returncode != 0:
        log.warning(f"EukCC merge run failed (rc={r.returncode}) -- skipping merge.")
        return {}, [], "skipped_eukcc_merge_run_failed"

    post_csv = merge_out / "eukcc.tsv"
    if not post_csv.exists():
        post_csv = merge_out / "eukcc.csv"
    post_quality = {r["bin"]: r for r in (_read_eukcc_csv(str(post_csv)) or [])}
    if not post_quality:
        log.warning("EukCC merge run produced no readable eukcc.csv -- skipping merge.")
        return {}, [], "skipped_post_csv_unreadable"

    # Provenance reconstruction (contig-set based -- EukCC's own output
    # naming for merged bins is not documented, so we don't rely on it):
    # for every output bin, read its own contig headers and compare
    # against every original bin's contig set.
    def _contig_set(fasta_path: Path) -> Set[str]:
        s = set()
        try:
            with open(fasta_path) as f:
                for line in f:
                    if line.startswith(">"):
                        s.add(line[1:].strip().split()[0])
        except FileNotFoundError:
            pass
        return s

    original_sets = {bin_id: _contig_set(Path(p)) for bin_id, p in bin_paths.items()}
    merged_bin_paths: Dict[str, str] = {}
    merge_log: List[dict] = []

    merge_fastas = list(merge_out.glob("*.fa*")) + list((merge_out / "merged_bins").glob("*.fa*"))
    for out_fasta in sorted(merge_fastas):
        out_set = _contig_set(out_fasta)
        if not out_set:
            continue
        # which original bins are (non-trivially) subsets of this output bin?
        contributors = [bid for bid, s in original_sets.items() if s and s <= out_set]
        if len(contributors) < 2:
            continue  # unchanged bin, or couldn't attribute -- not a merge
        # only accept as a genuine merge if the contributors' union
        # reconstructs (covers) this output bin's contig set
        union = set().union(*(original_sets[c] for c in contributors))
        if union != out_set:
            log.warning(f"EukCC merge: output {out_fasta.name} doesn't cleanly decompose into "
                        f"known source bins -- skipping evidence recording for it, keeping "
                        f"original bins unchanged for safety.")
            continue

        new_bin_id = f"merged_{'_'.join(sorted(contributors))}"
        merged_bin_paths[new_bin_id] = str(out_fasta)
        merge_log.append({
            "new_bin_id": new_bin_id,
            "source_bin_ids": contributors,
            "n_contigs_merged": len(out_set),
            "pre_merge_quality": {c: pre_quality.get(c) for c in contributors},
            "post_merge_quality": post_quality.get(out_fasta.stem),
            "reason": "EukCC --links accepted this merge (quality score improved); "
                      "see bin_merging docs at eukcc.readthedocs.io.",
        })

    if not merge_log:
        log.info("EukCC merge: ran successfully but produced no accepted merges.")
        return {}, [], "ran_no_merges_accepted"

    log.info(f"EukCC merge: {len(merge_log)} merge(s) accepted "
             f"(from {sum(len(m['source_bin_ids']) for m in merge_log)} source bins).")
    return merged_bin_paths, merge_log, "ran_merges_accepted"


# =============================================================================
# STEP 11 -- SILHOUETTE (diagnostic only -- FIX 9)
# =============================================================================

def compute_silhouette_diagnostic(latent, labels, clust_mask, cfg: ClusteringConfig) -> Optional[dict]:
    """
    Diagnostic-only. Computed on the post-refinement labels restricted to
    clust_mask, using only non-noise contigs (silhouette requires >= 2
    clusters and no noise label). Subsampled to silhouette_max_n under
    clustering_seed if larger. NEVER changes any assignment -- this value
    is written to cluster_quality.json and nothing else reads it.
    """
    if not cfg.compute_silhouette:
        return {"computed": False, "reason": "disabled_in_config"}
    if not HAS_SKLEARN:
        log.warning("compute_silhouette=True but scikit-learn is not installed -- skipping "
                    "(pip install scikit-learn to enable this diagnostic).")
        return {"computed": False, "reason": "sklearn_not_installed"}

    idx = np.where(clust_mask & (labels >= 0))[0]
    if len(idx) < 2 or len(set(labels[idx])) < 2:
        return {"computed": False, "reason": "fewer_than_2_non_noise_clusters"}

    rng = np.random.default_rng(cfg.clustering_seed)
    if len(idx) > cfg.silhouette_max_n:
        idx = rng.choice(idx, size=cfg.silhouette_max_n, replace=False)
        subsampled = True
    else:
        subsampled = False

    # A random subsample can, by chance, land in only one cluster (more
    # likely the smaller silhouette_max_n is relative to the number/balance
    # of clusters) -- found by this module's own test harness, not
    # hypothetical. Re-check post-subsample rather than letting
    # silhouette_score raise into the caller: this diagnostic must never be
    # able to crash the pipeline it's just reporting on.
    if len(set(labels[idx])) < 2:
        return {"computed": False, "reason": "subsample_left_fewer_than_2_clusters",
                "n_points": int(len(idx))}

    try:
        score = float(silhouette_score(cosine_normalize(latent[idx]), labels[idx], metric="euclidean"))
    except Exception as e:
        log.warning(f"Silhouette computation failed ({e}) -- recording as not computed "
                    f"(diagnostic only, does not affect clustering).")
        return {"computed": False, "reason": f"silhouette_score_error: {e}"}

    log.info(f"Silhouette (diagnostic only, n={len(idx):,}{' subsampled' if subsampled else ''}): "
             f"{score:.4f}")
    return {"computed": True, "score": score, "n_points": int(len(idx)), "subsampled": subsampled}


# =============================================================================
# STEP 12 -- SOFTWARE VERSIONS (for reproducibility -- FIX 10)
# =============================================================================

def _tool_version(cmd: List[str]) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        text = (r.stdout or r.stderr or "").strip().splitlines()
        return text[0] if text else None
    except Exception:
        return None


def collect_software_versions(actual_backend: str, backend_lib: str) -> dict:
    import platform
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "hdbscan": getattr(hdbscan, "__version__", "unknown"),
        "sklearn": (__import__("sklearn").__version__ if HAS_SKLEARN else None),
        "skani": _tool_version(["skani", "--version"]),
        "clustering_backend_actual": actual_backend,
        "clustering_backend_library": backend_lib,
        "module_version": VERSION,
    }


# =============================================================================
# STEP 13 -- WRITE OUTPUTS (FIX 27/28)
# =============================================================================

def write_outputs(outdir: Path, dense_labels_dedup: np.ndarray, contig_ids: List[str],
                   lengths: Dict[str, int], excluded_reasons: Dict[str, str],
                   dedup_status: Dict[str, str], quality_by_bin: Dict[str, dict],
                   silhouette: dict, merge_log: List[dict],
                   dedup_groups: Dict[str, List[str]], stats: dict, cfg: ClusteringConfig,
                   fp_full: str) -> str:
    """Writes cluster_assignments.tsv, cluster_summary.tsv, unclustered.tsv,
    cluster_quality.json, clustering_manifest.json, clustering_stats.json,
    clustering_config.json."""

    assignments_path = outdir / "cluster_assignments.tsv"
    with open(assignments_path, "w") as f:
        f.write("contig_id\tlength_bp\tbin_id\tstatus\n")
        for i, cid in enumerate(contig_ids):
            bin_id = dense_labels_dedup[i]
            length = lengths.get(cid, 0)
            if bin_id != "unbinned":
                f.write(f"{cid}\t{length}\t{bin_id}\tclustered\n")
            else:
                reason = excluded_reasons.get(cid, "noise_or_filtered")
                f.write(f"{cid}\t{length}\tunbinned\t{reason}\n")

    unclustered_path = outdir / "unclustered.tsv"
    with open(unclustered_path, "w") as f:
        f.write("contig_id\tlength_bp\treason\n")
        for i, cid in enumerate(contig_ids):
            if dense_labels_dedup[i] == "unbinned":
                reason = excluded_reasons.get(cid, "noise_or_filtered")
                f.write(f"{cid}\t{lengths.get(cid, 0)}\t{reason}\n")

    summary_path = outdir / "cluster_summary.tsv"
    bin_ids = sorted(set(dense_labels_dedup[dense_labels_dedup != "unbinned"]))
    with open(summary_path, "w") as f:
        f.write("bin_id\tn_contigs\ttotal_length_bp\tmean_length_bp\tmax_length_bp\tn50_bp\tstatus\n")
        for bin_id in bin_ids:
            idx = np.where(dense_labels_dedup == bin_id)[0]
            ctg_lens = [lengths.get(contig_ids[i], 0) for i in idx]
            total = sum(ctg_lens)
            mean_l = int(np.mean(ctg_lens)) if ctg_lens else 0
            max_l = max(ctg_lens) if ctg_lens else 0
            n50 = compute_n50(ctg_lens)
            status = dedup_status.get(bin_id, "kept")
            f.write(f"{bin_id}\t{len(idx)}\t{total}\t{mean_l}\t{max_l}\t{n50}\t{status}\n")
        unb_idx = np.where(dense_labels_dedup == "unbinned")[0]
        unb_lens = [lengths.get(contig_ids[i], 0) for i in unb_idx]
        f.write(f"unbinned\t{len(unb_idx)}\t{sum(unb_lens)}\t"
                f"{int(np.mean(unb_lens)) if unb_lens else 0}\t"
                f"{max(unb_lens) if unb_lens else 0}\t{compute_n50(unb_lens)}\tunbinned\n")

    quality_path = outdir / "cluster_quality.json"
    with open(quality_path, "w") as f:
        json.dump({"silhouette": silhouette, "per_bin_eukcc": quality_by_bin}, f, indent=2, default=str)

    stats_path = outdir / "clustering_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    config_path = outdir / "clustering_config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    manifest = {
        "module": "clustering.py", "version": VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fingerprint": fp_full,
        "dedup_groups": dedup_groups,
        "eukcc_merge_log": merge_log,
        "outputs": {
            "cluster_assignments": str(assignments_path),
            "cluster_summary": str(summary_path),
            "unclustered": str(unclustered_path),
            "cluster_quality": str(quality_path),
            "clustering_stats": str(stats_path),
            "clustering_config": str(config_path),
        },
    }
    manifest_path = outdir / "clustering_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    log.info(f"Wrote: {assignments_path}, {summary_path}, {unclustered_path}, {quality_path}, "
             f"{manifest_path}, {stats_path}, {config_path}")
    return str(summary_path)


# =============================================================================
# MAIN PUBLIC API
# =============================================================================

def run_clustering(
    final_latent_path:      str,
    encoder_manifest_path:  Optional[str] = None,
    contig_ids_path:        Optional[str] = None,
    fasta_path:             str = None,
    cov_features_path:      str = None,
    outdir:                 str = "clustering_output",
    tnf_weights_path:       Optional[str] = None,
    te_weights_path:        Optional[str] = None,
    alignment_bam_paths:    Optional[List[str]] = None,
    contig_classification:  Optional[Dict[str, str]] = None,
    config:                 Optional[ClusteringConfig] = None,
    config_path:            Optional[str] = None,
    **kwargs,
) -> str:
    """
    Full HyphaeS clustering pipeline v7.

    Args:
        final_latent_path     : .npy [n_contigs x latent_dim] from encoder.py
        encoder_manifest_path : encoder_manifest.json from the SAME encoder
                                 run (default: alongside final_latent_path)
        contig_ids_path       : canonical, ordered contig-ID file from
                                 whichever upstream module you point it at
                                 (TNF/TE's contig_ids.json or COV's
                                 contig_ids.txt) -- default: COV's
                                 contig_ids.txt next to cov_features_path
        fasta_path             : assembly FASTA (name-keyed lookup only,
                                  never used for row order -- see FIX 1)
        cov_features_path      : .npy [n_contigs x N+4] from coverage.py
        outdir                 : output directory
        tnf_weights_path/te_weights_path : optional per-contig confidence weights
        alignment_bam_paths    : optional list of paired-end BAM alignments
                                  against the bin FASTAs, enabling EukCC's
                                  real --links merge workflow (FIX 13).
                                  Omit to skip merging cleanly.
        contig_classification  : optional contig_id -> domain label map used
                                  to restrict EukCC assessment/merging to bins
                                  meeting the eukaryote fraction threshold.
        config / config_path   : ClusteringConfig, or path to config.yaml

    Returns:
        path to cluster_summary.tsv
    """
    if kwargs:
        raise TypeError(f"run_clustering() got unexpected keyword argument(s): {sorted(kwargs)}")
    if not fasta_path or not cov_features_path:
        raise ValueError("run_clustering: fasta_path and cov_features_path are required")

    if config_path is not None:
        cfg = ClusteringConfig.from_yaml(config_path)
    elif config is not None:
        cfg = config
    else:
        cfg = ClusteringConfig()

    if encoder_manifest_path is None:
        encoder_manifest_path = str(Path(final_latent_path).parent / "encoder_manifest.json")
    if contig_ids_path is None:
        contig_ids_path = str(Path(cov_features_path).parent / "contig_ids.txt")

    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log.info("")
    log.info("+" + "=" * 66 + "+")
    log.info("|" + " " * 10 + f"HYPHAES CLUSTERING {VERSION} -- Config-driven" + " " * 10 + "|")
    log.info("+" + "=" * 66 + "+")

    # Step 0/1 -- load + validate (FIX 1-4)
    (latent, sequences, lengths, cov_features, valid_mask,
     tnf_weights, te_weights, contig_ids, order_hash) = load_inputs(
        final_latent_path, encoder_manifest_path, contig_ids_path, fasta_path,
        cov_features_path, tnf_weights_path, te_weights_path, cfg)
    n = len(contig_ids)

    # Step 2 -- anchors (diagnostic only)
    anchor_mask = identify_anchors(contig_ids, lengths, tnf_weights, cov_features, cfg)

    # Step 2.5 -- length + validity filter (FIX 4)
    latent_clust, ids_clust, clust_mask, excluded_reasons = filter_contigs_for_clustering(
        latent, contig_ids, lengths, valid_mask, cfg)
    n_clust = len(ids_clust)

    cfg = cfg.resolve(n_clust)
    cfg.log_summary()

    # Step 3 -- HDBSCAN (checkpointed, fingerprinted, backend-selectable)
    # FIX 30 (v7.6, REVISED in v7.7): resolve the backend here, before
    # building the fingerprint -- resolve_clustering_backend() only
    # inspects environment availability (GPU/cuML presence) and does not
    # depend on actually running HDBSCAN, so this is safe and avoids a
    # second, redundant resolution inside run_hdbscan().
    #
    # FIX 33 (v7.7): resolved_backend/backend_lib are deliberately NOT
    # baked into fp_hdbscan's hash anymore (a v7.6 approach that had its
    # own bug -- see below). Instead, run_hdbscan() compares
    # resolved_backend against the ACTUAL backend recorded in the cached
    # checkpoint's own metadata at cache-hit time (see its docstring).
    # This correctly distinguishes two cases that hashing resolved_backend
    # alone could not:
    #   - resolved_backend="gpu" both times, but the GPU fit only
    #     succeeded on the second attempt (first time it silently fell
    #     back to CPU inside _fit_hdbscan after a runtime error). Hashing
    #     resolved_backend="gpu" both times would make these two runs
    #     LOOK identical and incorrectly serve the first run's
    #     CPU-computed labels as if they were a fresh GPU result.
    #   - resolved_backend="cpu" explicitly requested both times: fp stays
    #     stable, cache reused correctly, no change in behavior for the
    #     common CPU-only case.
    resolved_backend, backend_lib = resolve_clustering_backend(cfg.clustering_backend)
    fp_hdbscan = _fingerprint(
        _file_fingerprint(final_latent_path), order_hash, _file_fingerprint(cov_features_path),
        _file_fingerprint(fasta_path),
        cfg.hdbscan_min_cluster_size, cfg.hdbscan_min_samples, cfg.hdbscan_epsilon,
        cfg.hdbscan_method, cfg.hdbscan_algorithm, cfg.hdbscan_soft_prob_min,
        cfg.clustering_seed, cfg.cluster_min_contig_len, HDBSCAN_CORE_VERSION,
    )  # FIX 25 (v7.5): hdbscan_algorithm added to the fingerprint.
       # FIX 33 (v7.7): backend is intentionally NOT part of this hash --
       # see the actual-backend comparison done inside run_hdbscan() instead.
    labels_clust, probs, latent_norm_clust, actual_backend = run_hdbscan(
        latent_clust, cfg, outdir, n_clust, fp_hdbscan,
        resolved_backend=resolved_backend, backend_lib=backend_lib)
    n_bins_hdbscan = len(set(labels_clust[labels_clust >= 0]))

    labels = np.full(n, -1, dtype=int)
    labels[clust_mask] = labels_clust

    # Step 4/5 -- noise assignment + refinement (v7.3: refinement now has a
    # distance gate, FIX 22)
    stage_dir = outdir / "stage_assignments"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "contig_ids.json").write_text(json.dumps(contig_ids), encoding="utf-8")
    np.save(stage_dir / "eligible_mask.npy", clust_mask)
    raw_full = np.full(n, -1, dtype=int)
    raw_full[clust_mask] = np.load(outdir / "checkpoints" / "hdbscan" / "hdbscan_labels_raw.npy")
    np.save(stage_dir / "01_hdbscan_raw.npy", raw_full)
    np.save(stage_dir / "02_probability_filtered.npy", labels)
    probs_full = np.full(n, np.nan, dtype=np.float32)
    probs_full[clust_mask] = probs
    np.save(stage_dir / "membership_probabilities.npy", probs_full)
    if cfg.noise_rescue_stage == "early":
        labels = assign_noise_contigs(latent, labels, clust_mask, cfg, probabilities=probs_full)
    else:
        log.info("Rescue deferred to the LATE stage (after sub-clustering/merging) -- noise_rescue_stage=late")
    np.save(stage_dir / "03_after_rescue.npy", labels)
    labels = iterative_refinement(latent, labels, clust_mask, cfg)
    np.save(stage_dir / "04_after_refinement.npy", labels)
    mods = load_modality_latents(final_latent_path, n) if (cfg.subcluster_enabled or cfg.binmerge_enabled
                                                           or cfg.tnf_consensus_enabled) else {}
    gc = (contig_gc(contig_ids, sequences)
          if (cfg.merge_max_gc_diff > 0 or cfg.merge_gc_outside_max > 0) else None)
    if cfg.subcluster_enabled:
        labels = subcluster_bins(labels, clust_mask, contig_ids, lengths, latent, mods,
                                 cov_features, te_weights, cfg, outdir, gc=gc)
    np.save(stage_dir / "04b_after_subcluster.npy", labels)
    # v8.2 order: split -> PRUNE -> merge -> reassign pruned -> recruit
    # (v8.1 merged before pruning, so a bin still carrying foreign contigs could be
    # merged into the wrong genome because its coverage profile looked blended.)
    labels, pruned_idx = prune_coverage_outliers(labels, clust_mask, cov_features, cfg,
                                                 return_pruned=True)
    np.save(stage_dir / "04c_after_prune.npy", labels)
    refused_pairs = []
    labels = merge_same_genome_bins(labels, clust_mask, latent, mods, cov_features, cfg, outdir, gc=gc,
                                    refused_out=refused_pairs)
    np.save(stage_dir / "04d_after_binmerge.npy", labels)
    labels = marker_checked_merge(labels, refused_pairs, clust_mask, contig_ids, sequences, lengths,
                                  cfg, outdir)
    np.save(stage_dir / "04d1_after_marker_merge.npy", labels)
    labels = reassign_pruned_contigs(labels, pruned_idx, clust_mask, latent, mods, cov_features, cfg)
    np.save(stage_dir / "04d2_after_reassign.npy", labels)
    labels = tnf_consensus_reassign(labels, clust_mask, latent, mods, cfg, cov_features=cov_features)
    np.save(stage_dir / "04d3_after_tnf_consensus.npy", labels)
    if cfg.noise_rescue_stage == "late":
        not_pruned = np.ones(n, dtype=bool)
        not_pruned[pruned_idx] = False          # never re-add contigs removed by coverage pruning
        labels = assign_noise_contigs(latent, labels, clust_mask, cfg, probabilities=probs_full,
                                      query_mask=not_pruned)
    np.save(stage_dir / "04d4_after_late_rescue.npy", labels)
    labels = recruit_short_contigs(labels, clust_mask, valid_mask, contig_ids, lengths, latent, cfg)
    np.save(stage_dir / "04e_after_recruit.npy", labels)

    # Step 9 (diagnostic) -- silhouette, computed on the post-refinement
    # labels before bin filtering removes anything; never affects assignment.
    silhouette = compute_silhouette_diagnostic(latent, labels, clust_mask, cfg)

    # Step 6 -- filter low-quality bins (unchanged core)
    labels, kept_bins, removed_low_quality = filter_bins(labels, contig_ids, lengths, cfg)
    np.save(stage_dir / "05_after_bin_filter.npy", labels)
    if not kept_bins:
        log.warning("FLAG:NO_BINS -- all bins filtered. Check assembly quality and config thresholds.")
    elif len(kept_bins) < 3:
        log.warning(f"FLAG:LOW_BIN_COUNT -- only {len(kept_bins)} bins survived filtering")

    # Step 6.5 -- dense bin IDs, pass 1 (non-deduplicated)
    dense_labels_nondedup, raw_to_dense_1 = assign_dense_bin_ids(labels, sorted(kept_bins))

    # Step 7 -- write clusters_non_deduplicated/ (directory name configurable
    # via cfg.clusters_non_deduplicated_dir, FIX 20)
    nondedup_dir = outdir / cfg.clusters_non_deduplicated_dir
    bin_paths_nondedup = write_bin_fastas(dense_labels_nondedup, contig_ids, sequences, nondedup_dir)

    # Step 8 -- skani dedup on the non-deduplicated set
    surviving_ids, dedup_groups = run_skani_dedup(
        bin_paths_nondedup, lengths, contig_ids, dense_labels_nondedup, outdir, cfg, label="dedup")
    surviving_paths = {b: bin_paths_nondedup[b] for b in surviving_ids}

    dedup_status: Dict[str, str] = {b: "kept" for b in dense_labels_nondedup
                                     if b != "unbinned"}
    for survivor, absorbed in dedup_groups.items():
        for a in absorbed:
            dedup_status[a] = f"removed_redundant(kept={survivor})"

    # Step 9 -- EukCC on the post-dedup, pre-merge bins. The same filtered
    # directory and bin-path mapping are used for quality assessment and merge,
    # so a prokaryotic bin excluded by the classification gate cannot return
    # through the merge stage.
    postdedup_dir = outdir / "_working_post_dedup_pre_merge"
    postdedup_dir.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_bins(postdedup_dir, surviving_paths)
    for bin_id, src in surviving_paths.items():
        src_resolved = Path(src).resolve()
        for ext in (".fasta", ".fa"):
            dst = postdedup_dir / f"{bin_id}{ext}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src_resolved)

    eukcc_input_dir = postdedup_dir
    eukcc_bin_paths = dict(surviving_paths)
    if cfg.run_eukcc and cfg.eukcc_db and contig_classification is not None:
        eukcc_input_dir, included_ids = _prepare_eukcc_bins(
            postdedup_dir, outdir, cfg, contig_classification, label="eukcc"
        )
        eukcc_bin_paths = {b: p for b, p in surviving_paths.items() if b in included_ids}

    eukcc_csv = run_eukcc(eukcc_input_dir, outdir, cfg, label="eukcc") if eukcc_bin_paths else None

    # Step 10 -- EukCC-based merge, on exactly the same filtered bin set.
    merged_bin_paths, merge_log, merge_status = run_eukcc_merge(
        eukcc_input_dir, eukcc_bin_paths, outdir, cfg, alignment_bam_paths)

    merged_source_ids: Set[str] = set()
    for m in merge_log:
        merged_source_ids.update(m["source_bin_ids"])

    final_paths: Dict[str, str] = {b: p for b, p in surviving_paths.items()
                                    if b not in merged_source_ids}
    final_paths.update(merged_bin_paths)
    for m in merge_log:
        for src in m["source_bin_ids"]:
            dedup_status[src] = f"merged_into({m['new_bin_id']})"

    # Step 14 -- final skani consistency check (does not loop back into dedup)
    if len(final_paths) >= 2:
        _survivors_check, _redundant_after_merge = run_skani_dedup(
            final_paths,
            lengths, contig_ids,
            # dense_labels not needed for the final consistency check's bin_quality
            # computation beyond what's already embedded in final_paths' FASTAs;
            # reuse contig-set-derived lengths instead of relying on `labels`.
            np.array(["unbinned"] * n, dtype=object),
            outdir, cfg, label="post_merge_consistency_check",
        )
        if _redundant_after_merge:
            log.warning(f"FLAG:REDUNDANT_AFTER_MERGE -- {len(_redundant_after_merge)} bin(s) "
                        f"became redundant with another after merging: {_redundant_after_merge}. "
                        f"Not auto-resolved (no re-loop into dedup) -- review manually.")
    else:
        _redundant_after_merge = {}

    # Step 6.5 (pass 2) -- dense bin IDs for the FINAL (post-dedup, post-merge) set
    final_bin_order = sorted(final_paths.keys())
    id_remap = {old: f"bin_{i:0{max(3, len(str(len(final_bin_order))))}d}"
                for i, old in enumerate(final_bin_order, 1)}

    dense_labels_dedup = np.array(
        [id_remap.get(dense_labels_nondedup[i]) if dense_labels_nondedup[i] in id_remap
         else (
             # a merged-away source bin's contigs now belong to whatever
             # merged bin absorbed them
             next((id_remap[m["new_bin_id"]] for m in merge_log
                   if dense_labels_nondedup[i] in m["source_bin_ids"]), "unbinned")
         )
         for i in range(n)],
        dtype=object,
    )

    # Directory name configurable via cfg.clusters_deduplicated_dir (FIX 20)
    dedup_dir = outdir / cfg.clusters_deduplicated_dir
    dedup_dir.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_bins(dedup_dir, id_remap.values())
    final_bin_paths: Dict[str, str] = {}
    for old_id, path in final_paths.items():
        new_id = id_remap[old_id]
        # Same rerun hazard as write_bin_fastas (see its comment): remove any
        # stale .fa symlink BEFORE resolving anything, so a second run never
        # follows it onto dst_fasta's target and self-clobbers the bin.
        dst_fasta = dedup_dir / f"{new_id}.fasta"
        dst_fa = dedup_dir / f"{new_id}.fa"
        if dst_fa.exists() or dst_fa.is_symlink():
            dst_fa.unlink()
        shutil.copyfile(path, dst_fasta)
        dst_fasta = dst_fasta.resolve()
        dst_fa.symlink_to(dst_fasta)
        final_bin_paths[new_id] = str(dst_fasta)
    log.info(f"Wrote {len(final_bin_paths)} final (deduplicated) bin(s) -> {dedup_dir}")

    np.save(stage_dir / "06_final_bin_ids.npy", dense_labels_dedup.astype(str))

    dedup_status_final = {id_remap.get(k, k): v for k, v in dedup_status.items()}

    # EukCC quality-by-bin for the final set: non-merged bins keep their
    # step-9 eukcc.csv row; merged bins use their merge-verification row.
    pre_quality_rows = {r["bin"]: r for r in (_read_eukcc_csv(eukcc_csv) or [])} if eukcc_csv else {}
    quality_by_bin: Dict[str, dict] = {}
    for old_id, new_id in id_remap.items():
        if old_id in pre_quality_rows:
            quality_by_bin[new_id] = pre_quality_rows[old_id]
    for m in merge_log:
        new_id = id_remap.get(m["new_bin_id"])
        if new_id and m.get("post_merge_quality"):
            quality_by_bin[new_id] = m["post_merge_quality"]

    excluded_reasons_full = dict(excluded_reasons)
    for i, cid in enumerate(contig_ids):
        if clust_mask[i] and dense_labels_nondedup[i] == "unbinned" and cid not in excluded_reasons_full:
            excluded_reasons_full[cid] = "noise_or_filtered"

    stats = {
        "n_contigs": n,
        "n_contigs_clustered": n_clust,
        "n_excluded_too_short": sum(1 for v in excluded_reasons.values() if v == "too_short"),
        "n_excluded_invalid_coverage": sum(1 for v in excluded_reasons.values() if v == "invalid_coverage"),
        "n_anchors": int(anchor_mask.sum()),
        "n_bins_hdbscan": n_bins_hdbscan,
        "n_bins_after_filter": len(kept_bins),
        "n_bins_after_dedup": len(surviving_ids),
        "n_bins_final": len(final_bin_paths),
        "n_unbinned": int((dense_labels_dedup == "unbinned").sum()),
        "n_unbinned_eligible": int(((dense_labels_dedup == "unbinned") & clust_mask).sum()),
        "eukcc_csv": eukcc_csv,
        "eukcc_merge_status": merge_status,
        "n_eukcc_merges_accepted": len(merge_log),
        "n_redundant_after_merge": len(_redundant_after_merge),
        "runtime_s": time.time() - t0,
        "hdbscan_params": _hdbscan_params(cfg),
        "backend": {"requested": cfg.clustering_backend, "actual": actual_backend,
                    "threads": cfg.clustering_threads, "seed": cfg.clustering_seed},
        "software_versions": collect_software_versions(actual_backend,
                                                         "hdbscan" if actual_backend == "cpu" else "cuml"),
    }

    classification_fp = _fingerprint(
        sorted((str(k), str(v).strip().lower()) for k, v in (contig_classification or {}).items())
    )
    fp_full = _fingerprint(fp_hdbscan, order_hash, asdict(cfg), classification_fp, VERSION)

    summary = write_outputs(
        outdir, dense_labels_dedup, contig_ids, lengths, excluded_reasons_full,
        dedup_status_final, quality_by_bin, silhouette, merge_log, dedup_groups,
        stats, cfg, fp_full,
    )
    #
    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 68)
    log.info(f"  CLUSTERING {VERSION} COMPLETE")
    log.info(f"  Contigs total        : {n:,}")
    log.info(f"  Clustered (eligible) : {n_clust:,}")
    log.info(f"  Excluded (too_short) : {stats['n_excluded_too_short']:,}")
    log.info(f"  Excluded (invalid)   : {stats['n_excluded_invalid_coverage']:,}")
    log.info(f"  HDBSCAN bins         : {n_bins_hdbscan} (backend={actual_backend})")
    log.info(f"  After filter         : {len(kept_bins)}")
    log.info(f"  After dedup          : {len(surviving_ids)}")
    log.info(f"  EukCC merges         : {len(merge_log)} ({merge_status})")
    log.info(f"  Final bins           : {len(final_bin_paths)}")
    log.info(f"  Unbinned             : {stats['n_unbinned']:,} (all contigs)")
    log.info(f"  Unbinned (eligible)  : {stats['n_unbinned_eligible']:,} "
             f"({stats['n_unbinned_eligible']/max(1, n_clust):.1%} of {n_clust:,})")
    log.info(f"  Silhouette           : {silhouette}")
    log.info(f"  non_deduplicated dir : {nondedup_dir}")
    log.info(f"  deduplicated dir     : {dedup_dir}")
    log.info(f"  Summary              : {summary}")
    log.info(f"  Time                 : {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info("=" * 68)

    return summary


# PROFILE HOOKS v1 — observation only; scientific functions above are unchanged.
from hyphaesbin.utils.resource_profiler import profile_functions, profile_checkpoint
profile_functions(globals(), ['run_clustering', 'run_hdbscan', '_fit_hdbscan', 'assign_noise_contigs', 'iterative_refinement', 'subcluster_bins', 'merge_same_genome_bins', 'recruit_short_contigs', 'filter_bins', 'run_skani_dedup', 'run_eukcc', 'run_eukcc_merge'])
_checkpoint_ok = profile_checkpoint(_checkpoint_ok)
