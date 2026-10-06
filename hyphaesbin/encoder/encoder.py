#"""
#HyphaeS Encoder Module — v8.1
#=========================================
#File: hyphaesbin/encoder/encoder.py
#
#=============================================================================
#CHANGELOG FROM v7 — ALIGNMENT / CONFIG / DEVICE / FUSION-MODE FIXES
#=============================================================================
#v7 fixed the Phase-2 root-cause bugs (frozen sub-encoders, reconstruction-
#through-fusion gate loss). This pass (v8) closes a second, independent set
#of issues found in a follow-up review of the surrounding plumbing: how
#features are loaded, validated, aligned across modalities, checkpointed,
#and how the fusion gate itself works. NONE of these change the "core"
#mechanics that were already validated as healthy: UniformVAE's
#architecture, vae_loss, the Phase-1 per-modality training loop
#(_train_vae_epoch / _train_phase1), and FIX 1 / FIX 2 from v7 (frozen
#sub-encoders during Phase 2, reconstruction-through-fusion as the training
#signal) are all untouched in their actual math.
#
#FIX 5 — COVERAGE FEATURE ARRAY WAS NEVER SPLIT (was: fed to the VAE whole)
#  coverage.py's coverage_features.npy is N+4D:
#      [valid_mask | n_samples_present | mean_dist | std_dist | cov_1..cov_N]
#  v7 loaded this whole array and fed it straight into the COV VAE as if
#  every column were a feature. valid_mask is documented by coverage.py's
#  own COLUMN_ROLES as "MASK ONLY — never a feature", so the VAE was being
#  trained to reconstruct a mask column, and COV never used any per-contig
#  reliability weight at all (v7 hardcoded weights=None for COV in Phase 1,
#  and Phase 2 never touched COV weights either).
#  FIX: split_cov_features() removes valid_mask/n_samples_present from the
#  feature vector (n_samples_present can be opted back in via
#  cfg.cov_use_n_samples_present_as_feature, off by default, matching
#  coverage.py's own "consumer's discretion" framing) and turns valid_mask
#  into COV's per-contig reliability weight, used in both Phase 1 (COV's
#  own vae_loss masking) and Phase 2 (see FIX 9).
#
#FIX 6 — CONTIG ALIGNMENT WAS ROW-COUNT ONLY (was: shape[0] equality)
#  v7's only alignment check between TNF/TE/COV was "do the arrays have the
#  same number of rows". Two modalities could have the same contig COUNT
#  but a different ORDER (e.g. one pipeline run resorted or re-filtered
#  contigs) and this would silently fuse row i of TNF with row i of TE even
#  though they describe different contigs — a silent correctness disaster
#  no shape check can catch.
#  FIX: _load_modality_manifest() reads each modality's OWN contig-ID
#  manifest (the three upstream modules use three different filename/key
#  conventions — verified by reading their actual save code, not assumed
#  from any docstring) and compares the full ordered contig-ID list (plus
#  its content hash) across all active modalities. Any mismatch is a hard
#  error before any training happens.
#
#FIX 7 — MISSING DIMENSION VALIDATION (was: expected_dim never passed)
#  validate_features() already supported an expected_dim check but no call
#  site ever used it. TNF features are always 136D and TE features are
#  always 5D by construction (tnf_gene.py / te_composition.py); a silent
#  dimension drift (e.g. an old cached tnf_features.npy from a differently
#  configured run) would previously pass straight through.
#  FIX: TNF_EXPECTED_DIM=136 / TE_EXPECTED_DIM=5 are now enforced. COV's
#  width is sample-count-dependent, so instead its width is cross-checked
#  against the `n_samples` argument when the caller supplies one (n_samples
#  was accepted but never used anywhere in v7 — dead parameter).
#
#FIX 8 — SILENT MISSING-MODALITY FALLBACK (was: warn + auto-disable)
#  If a modality was requested (cfg.X_include=True) but its feature path
#  was missing/invalid, v7 logged one WARNING line and silently trained
#  without it. A missing file is usually a real upstream failure (a crashed
#  pipeline step, a typo'd path), and training on 2 of 3 intended
#  modalities without the caller ever finding out is exactly the kind of
#  silent-degradation this project has been fixing everywhere else.
#  FIX: this is now a hard error (FileNotFoundError) unless the caller sets
#  the new `allow_missing_modalities: bool` config flag, in which case the
#  old warn-and-disable behavior is kept, but now opt-in and explicit.
#
#FIX 9 — PHASE 2 READBACK LOSS HAD NO PER-CONTIG WEIGHTING (was: whole-batch
#  unweighted mean, TNF/TE per-contig confidence weights unused in Phase 2)
#  v7's readback_loss() computed one MSE over the WHOLE batch per modality
#  and averaged the three (or fewer) modality losses uniformly — every
#  contig counted equally regardless of whether, e.g., its COV row was a
#  valid_mask=0 placeholder or its TE row had zero confidence weight
#  (fragmented contigs get near-zero TE reliability by construction; see
#  te_composition.py). FIX: readback_loss() now takes each modality's own
#  per-contig reliability weight (COV's valid_mask from FIX 5, and TNF/TE's
#  existing confidence weight arrays, defaulting to 1.0 when not supplied)
#  and combines them PER CONTIG:
#      contig_loss_i = (w_tnf_i*err_tnf_i + w_te_i*err_te_i + w_cov_i*err_cov_i)
#                      / (w_tnf_i + w_te_i + w_cov_i)
#  Contigs whose total weight is 0 across every active modality are
#  excluded from the batch mean entirely (same masked-average pattern
#  vae_loss already uses). This is a WEIGHTED-AVERAGE policy, not an
#  intersection/all-valid policy — requiring every modality to be valid for
#  a contig before it counts at all would throw away nearly all contigs,
#  since TE's reliability weight is near-all-zero for fragmented real
#  assemblies by design.
#  NOTE — two different "weights" that must not be conflated: the fusion
#  GATE weight (how much of each modality's latent goes into z_fus, learned
#  by FusionModule) is a completely different concept from the per-contig
#  RELIABILITY weight used here (how much a given contig's modality term
#  should count toward the training loss). This fix only touches the
#  latter.
#
#FIX 10 — device WAS HARDCODED AUTO-DETECT (was: cuda->mps->cpu, no config)
#  get_device() always auto-detected with no way to force CPU (e.g. to keep
#  a machine's GPU free for another job) or to require GPU and fail loudly
#  if unavailable. FIX: new `device: "cpu"|"gpu"|"auto"` config field +
#  resolve_device(). "cpu" always uses CPU. "gpu" uses CUDA if available,
#  else falls back to MPS with a warning, else CPU with a warning (a
#  missing GPU is treated as a soft degradation here, unlike the FIX 8
#  missing-file case, since it's an environment/hardware fact rather than a
#  silently-wrong-data risk). "auto" preserves the old cuda->mps->cpu
#  behavor. get_device() is kept as a thin backward-compatible alias for
#  resolve_device("auto").
#
#FIX 11 — fork MULTIPROCESSING CONTEXT (was: mp.get_context("fork"))
#  "fork" doesn't exist on Windows at all, and even on Linux it's unsafe
#  once a CUDA context may already be initialized in the parent process
#  ("cannot re-initialize CUDA in forked subprocess" is a well-known
#  failure mode). FIX: switched unconditionally to mp.get_context("spawn").
#  Also: parallel_phase1_max_workers is now clamped to os.cpu_count() (was
#  previously unclamped against the actual machine), and a warning is
#  logged if parallel Phase 1 is requested while device is cuda/mps (each
#  worker will initialize its own device context — this can work, but may
#  contend for the same GPU with no speedup; not auto-disabled, since that
#  is a legitimate configuration choice left to the caller).
#
#FIX 12 — NO SMALL-DATASET / ZERO-WEIGHT GUARDS (was: silent under-batching
#  or silent division against nothing)
#  Two new guards, both fail-loud rather than fail-silent:
#    (a) _effective_batch_size() clamps the requested batch_size down to
#        the dataset size instead of relying on DataLoader to silently
#        produce one small leftover batch every epoch.
#    (b) a modality whose ENTIRE per-contig weight array is 0 (every
#        contig masked out) now raises immediately instead of quietly
#        training on an empty effective dataset.
#
#FIX 13 (NEW FEATURE, not a bug fix) — fusion_mode: "contig_weighted"
#  v7 had a single, global softmax gate (one learned weight per modality,
#  shared by every contig). This adds a second, fully-implemented and
#  config-selectable mode: a GMU-style (Arevalo et al. 2017) per-contig
#  gate — a small MLP over each contig's own concatenated projected
#  latents producing a per-contig softmax over modalities, trained through
#  the SAME reconstruction-through-fusion loss as the global gate (FIX 9),
#  plus a small entropy-regularization term (`fusion_gate_entropy_weight`,
#  only active in this mode) that discourages the per-contig gate from
#  collapsing onto one modality for every contig (a known failure mode for
#  unsupervised instance-wise gates — loosely inspired by the sparsity/
#  collapse safeguards in "Adaptive Confidence-weighted Expansion"-style
#  approaches, not a literal reproduction of any one paper).
#  `fusion_mode` defaults to "global" (the validated, production default).
#  An invalid fusion_mode value is a hard config error (EncoderConfig.resolve
#  rejects anything other than "global"/"contig_weighted") — there is no
#  silent fallback to global.
#  In contig_weighted mode, the per-contig gate weights for the FULL
#  dataset are saved to `contig_fusion_weights.npy` in outdir for
#  downstream analysis (columns ordered per FusionModule._mods).
#
#FIX 14 (BONUS — found while implementing FIX 13, not on the original list)
#  v7's _phase2_joint() never persisted the trained FusionModule's weights
#  as part of its own checkpoint — only run_encoder's *final* save at the
#  very end wrote fusion.state_dict() into encoder_weights.pt. If Phase 2
#  hit its checkpoint (cache) on a re-run, the `fusion` object passed in by
#  the caller was still the FRESHLY (randomly) initialized one — so
#  encoder_weights.pt would get overwritten with UNTRAINED fusion weights
#  even though the returned final_latent.npy was correctly the cached,
#  actually-trained result. Latents and saved weights would silently
#  disagree. FIX: _phase2_joint now saves fusion.state_dict() into its own
#  checkpoint dir and reloads it into the passed-in module on a cache hit,
#  before returning.
#
#FIX 15 — CHECKPOINT FINGERPRINTING (was: is_done() + file-exists only)
#  Neither Phase 1 nor Phase 2 checkpoints included any fingerprint of
#  their actual inputs/config — a changed feature file, a changed
#  hyperparameter, or a changed device/fusion_mode could all silently reuse
#  a stale checkpoint. FIX: added the same _file_fingerprint / _fingerprint
#  / _checkpoint_ok pattern already used in preprocessing.py / coverage.py
#  / tnf_gene.py / te_composition.py. Phase-1 fingerprints cover that
#  modality's own feature/weight file fingerprints plus every
#  training-relevant hyperparameter, device, and module VERSION. Phase-2's
#  fingerprint additionally covers fusion_mode, fusion_gate_entropy_weight,
#  and the upstream Phase-1 fingerprints (so any Phase-1 change invalidates
#  Phase 2 too).
#
#=============================================================================
#CHANGELOG FROM v8 (initial) — v8.1, FOLLOW-UP REVIEW: 4 MORE PLUMBING BUGS
#=============================================================================
#A second read-through of the v8 plumbing (not the core VAE/fusion math)
#found four more real issues, all now fixed:
#
#FIX 16 — COV NORMALIZATION STATISTICS CONTAMINATED BY PLACEHOLDER ROWS
#  FIX 5 (v8 initial) correctly turned valid_mask into COV's reliability
#  weight for the LOSS, but normalize_features(cov_feat, "COV") still
#  computed mean/std over EVERY row, including valid_mask=0 placeholder
#  rows -- which are literal zero-filled rows inserted by coverage.py's
#  step7_realign, not real (if unreliable) data. Those zeros pulled the
#  mean/std of every real column toward zero, meaning even the loss-masked
#  "valid" rows were being normalized against contaminated statistics.
#  FIX: normalize_features() now accepts an optional `mask`; when given
#  (COV's call site passes cov_w, i.e. valid_mask), mean/std are computed
#  ONLY over mask>0 rows. Every row (including placeholders) is still
#  normalized and still gets its output value -- only the STATISTICS
#  computation excludes invalid rows.
#
#FIX 17 — TNF/TE WEIGHT-ARRAY LENGTH NEVER CHECKED AGAINST n_contigs
#  A stale or truncated tnf_weights.npy / te_weights.npy (e.g. left over
#  from a run with fewer contigs) was never checked for length before being
#  used. Depending on how it happened to disagree with n_contigs, this
#  could fail confusingly deep inside make_loader/TensorDataset, or --
#  worse -- silently misalign weight i with the wrong contig i if the
#  lengths happened to differ in a way numpy didn't immediately reject.
#  FIX: run_encoder now checks len(tnf_w) == n_contigs and
#  len(te_w) == n_contigs immediately after loading, before either array is
#  used anywhere, and raises a clear ValueError naming the mismatch.
#
#FIX 18 — WEIGHT FILES WERE FINGERPRINTED FOR CHECKPOINTING BUT NOT RECORDED
#  IN THE MANIFEST
#  fp_tnf/fp_te (the Phase-1 checkpoint fingerprints) already included
#  _file_fingerprint(tnf_weights_path) / _file_fingerprint(te_weights_path),
#  but encoder_manifest.json's "inputs" section only recorded the feature
#  file fingerprints, not the weight file fingerprints -- so the on-disk
#  record of what a run actually used was incomplete relative to what its
#  own checkpoint validity depended on. FIX: tnf_weights_fingerprint /
#  te_weights_fingerprint added alongside the existing *_features_fingerprint
#  entries.
#
#FIX 19 — EMPTY FEATURE ARRAY PRODUCED AN OPAQUE numpy ERROR
#  validate_features() called .min()/.max()/.mean() before checking whether
#  the array had any rows at all -- a 0-row feature file (e.g. every contig
#  filtered out upstream) would fail with numpy's generic "zero-size array
#  to reduction operation" error, far from anything naming the real cause.
#  FIX: an explicit shape[0]==0 check up front raises a clear, actionable
#  ValueError instead.
#
#KNOWN LIMITATION, STILL UNRESOLVED IN v8.1
#  PyTorch remains uninstallable in the sandbox this file is developed in
#  (see the v8 changelog above and the module-level note below) -- FIX 16
#  through FIX 19 were verified with the same non-torch unit-test approach
#  used for the original v8 fixes (pure-Python/numpy logic, run against a
#  minimal stand-in for `torch` sufficient for import). The actual VAE/
#  FusionModule training path (including whether FIX 16's masked
#  normalization interacts correctly with real gradients) is still NOT
#  verified by a real PyTorch run.
#
#WHAT WAS *NOT* CHANGED
#  UniformVAE, vae_loss, _train_vae_epoch, _train_phase1's training loop
#  body, and v7's FIX 1 / FIX 2 mechanics (frozen sub-encoders during Phase
#  2, reconstruction-through-fusion as the only signal training the fusion
#  module) are unchanged. This file only changes plumbing around those
#  mechanics: what gets fed in, how it's validated/aligned/weighted, how
#  checkpoints are fingerprinted, and the (new, opt-in) per-contig gate.
#
#KNOWN LIMITATION, DISCLOSED — NOT SILENTLY GLOSSED OVER
#  PyTorch could not be installed in the sandbox this file was written and
#  tested in (network-restricted environment; both the default index and
#  the CPU-only wheel index were unreachable). Every piece of logic in this
#  file that does NOT require torch (EncoderConfig validation/resolve,
#  _load_modality_manifest, split_cov_features, the checkpoint-fingerprint
#  helpers, _check_modality_available, _effective_batch_size,
#  resolve_device's branching logic) was exercised with real unit tests
#  using a minimal stand-in for the `torch` module. The actual VAE/
#  FusionModule forward/backward training path (UniformVAE, vae_loss,
#  FusionModule.forward/readback_loss/gate_entropy, _phase1_single,
#  _phase2_joint) could NOT be executed end-to-end in this sandbox and is
#  therefore NOT verified by real gradient-level testing here — only by
#  careful reading/tracing of the tensor shapes and control flow. Please
#  run this file's real test suite (or at least one real end-to-end call to
#  run_encoder on a small fixture) in an environment with PyTorch installed
#  before relying on it in production.
#
#config.yaml parameters (new keys marked NEW; everything else unchanged
#from v7):
#  te_include: true
#  cov_include: true
#  latent_dim_tnf: 64
#  latent_dim_te: 5
#  latent_dim_cov: null
#  final_dim: null
#  batch_size: null
#  beta_tnf: 0.01
#  beta_te: 0.1
#  beta_cov: 0.1
#  kl_anneal_epochs: 75
#  kl_free_bits: 0.01
#  phase1_epochs_tnf: 75
#  phase1_epochs_te: 50
#  phase1_epochs_cov: 50
#  phase2_epochs: 50
#  phase1_lr: 0.001
#  phase2_lr: 0.00005
#  loss_scale_te: 0.1
#  loss_scale_cov: 0.1
#  loss_scale_fusion: 0.05
#  dropout: 0.1
#  n_hidden_layers: 2
#  hidden_scale: 4
#  seed: 42
#  fusion_mode: "global"                       # NEW — "global" | "contig_weighted"
#  fusion_gate_entropy_weight: 0.01             # NEW — only used if fusion_mode="contig_weighted"
#  device: "auto"                               # NEW — "cpu" | "gpu" | "auto"
#  allow_missing_modalities: false              # NEW
#  cov_use_n_samples_present_as_feature: false  # NEW
#"""
#
#import hashlib
#import json
#import logging
#import os
#import time
#import multiprocessing as mp
#from concurrent.futures import ProcessPoolExecutor, as_completed
#from dataclasses import dataclass, field, asdict
#from pathlib import Path
#from typing import Optional, List, Dict, Tuple, Any
#
#import numpy as np
#
#try:
#    import torch
#    import torch.nn as nn
#    import torch.nn.functional as F
#    from torch.utils.data import DataLoader, TensorDataset
#except ImportError:
#    raise ImportError("PyTorch not installed: conda install -c pytorch pytorch")
#
#try:
#    import yaml
#    HAS_YAML = True
#except ImportError:
#    HAS_YAML = False
#
#try:
#    from hyphaesbin.utils.checkpoint import Checkpoint
#except Exception:
#    Checkpoint = None
#
#log = logging.getLogger("hyphaesbin.encoder")
#
#VERSION = "v8.1"
#
## Fixed, structural feature widths (see tnf_gene.py / te_composition.py) —
## used to catch dimension drift that a shape[0]-only check would miss.
#TNF_EXPECTED_DIM = 136
#TE_EXPECTED_DIM = 5
#
## coverage.py's COLUMN_LAYOUT_METADATA_NAMES prefix width:
## [valid_mask | n_samples_present | mean_dist | std_dist] then cov_1..cov_N.
#COV_METADATA_DIM = 4
#
#
## =============================================================================
## CHECKPOINT FINGERPRINTING — same pattern as preprocessing.py / coverage.py /
## tnf_gene.py / te_composition.py. A stale or config-mismatched checkpoint is
## a cache MISS, not a silent wrong answer.
## =============================================================================
#
#def _file_fingerprint(path) -> str:
#    """Size+mtime fingerprint, not a content hash — same acknowledged
#    limitation as the identical helper in the other pipeline modules."""
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
#    h = hashlib.sha256()
#    h.update(json.dumps(parts, sort_keys=True, default=str).encode())
#    return h.hexdigest()[:16]
#
#
#def _checkpoint_ok(ckpt, step: str, fp: str, output_paths: List[str]) -> Optional[Dict]:
#    if ckpt is None or not ckpt.is_done(step):
#        return None
#    prev = ckpt.load_metadata(step)
#    if prev.get("_fp") != fp:
#        log.warning(f"{step}: checkpoint exists but inputs/config/version changed since it "
#                    f"ran — ignoring stale checkpoint and re-running.")
#        return None
#    missing = [p for p in output_paths if p and not Path(p).exists()]
#    if missing:
#        log.warning(f"{step}: checkpoint fingerprint matches, but {len(missing)} referenced "
#                    f"output(s) no longer exist on disk ({missing[:3]}"
#                    f"{', ...' if len(missing) > 3 else ''}) — treating as a cache miss.")
#        return None
#    return prev
#
#
## =============================================================================
## ENCODER CONFIG
## =============================================================================
#
#@dataclass
#class EncoderConfig:
#    """
#    All encoder hyperparameters in one dataclass.
#
#    te_include=True/False   -> TE encoder active/skipped
#    cov_include=True/False  -> COV encoder active/skipped
#    TNF is always active.
#    """
#
#    # -- Modality toggles --------------------------------------------------
#    tnf_include:           bool          = True
#    te_include:            bool          = True
#    cov_include:            bool          = True
#
#    # -- Missing-modality policy (FIX 8) ------------------------------------
#    allow_missing_modalities: bool       = False
#
#    # -- Parallelism -----------------------------------------------------
#    parallel_phase1:        bool          = True    # train active Phase-1 VAEs concurrently
#    parallel_phase1_max_workers: Optional[int] = None  # None = one worker per active modality
#
#    # -- Device (FIX 10) -----------------------------------------------------
#    device:                 str           = "auto"   # "cpu" | "gpu" | "auto"
#
#    # -- Latent dimensions --------------------------------------------------
#    latent_dim_tnf:        int           = 64
#    latent_dim_te:         int           = 5
#    latent_dim_cov:        Optional[int] = None
#    final_dim:             Optional[int] = None
#
#    # -- Architecture --------------------------------------------------------
#    hidden_scale:          int           = 4
#    hidden_min:            int           = 32
#    hidden_max:            int           = 512
#    dropout:               float         = 0.1
#    n_hidden_layers:       int           = 2
#
#    # -- beta-VAE ---------------------------------------------------------------
#    beta_tnf:               float         = 0.01
#    beta_te:                float         = 0.1
#    beta_cov:                float         = 0.1
#    kl_anneal_epochs:       int           = 75
#    kl_free_bits:           float         = 0.01
#
#    # -- Training epochs -----------------------------------------------------------
#    phase1_epochs_tnf:      int           = 75
#    phase1_epochs_te:       int           = 50
#    phase1_epochs_cov:      int           = 50
#    phase2_epochs:          int           = 50
#
#    # -- Learning rates -----------------------------------------------------------------
#    phase1_lr:               float         = 1e-3
#    phase2_lr:                float         = 5e-5
#
#    # -- Loss scaling ------------------------------------------------------
#    # loss_scale_tnf/te/cov are used ONLY during Phase 1.
#    # loss_scale_fusion weights the reconstruction-through-fusion loss.
#    loss_scale_tnf:            float         = 1.0
#    loss_scale_te:             float         = 0.1
#    loss_scale_cov:            float         = 0.1
#    loss_scale_fusion:         float         = 0.05
#
#    # -- Fusion gate mode (FIX 13) -------------------------------------------
#    fusion_mode:               str           = "global"   # "global" | "contig_weighted"
#    fusion_gate_entropy_weight: float        = 0.01        # only used if fusion_mode="contig_weighted"
#
#    # -- Coverage feature split (FIX 5) --------------------------------------
#    cov_use_n_samples_present_as_feature: bool = False
#
#    # -- Batch size ---------------------------------------------------------------------
#    batch_size:                Optional[int] = None
#
#    # -- Reproducibility ------------------------------------------------------------------
#    seed:                       int           = 42
#
#    # -- Input dims (set at runtime, do NOT put in config.yaml) ---------------------------
#    input_dim_tnf:               Optional[int] = None
#    input_dim_te:                 Optional[int] = None
#    input_dim_cov:                 Optional[int] = None
#
#    @classmethod
#    def from_yaml(cls, path: str) -> "EncoderConfig":
#        if not HAS_YAML:
#            raise ImportError("PyYAML not installed: pip install pyyaml")
#        with open(path) as f:
#            d = yaml.safe_load(f) or {}
#        encoder_d = d.get("encoder", d)
#        valid = {k: v for k, v in encoder_d.items()
#                 if k in cls.__dataclass_fields__}
#        return cls(**valid)
#
#    @classmethod
#    def from_dict(cls, d: dict) -> "EncoderConfig":
#        valid = {k: v for k, v in d.items()
#                 if k in cls.__dataclass_fields__}
#        return cls(**valid)
#
#    def resolve(self, n_contigs: int) -> "EncoderConfig":
#        if not (self.tnf_include or self.te_include or self.cov_include):
#            raise ValueError(
#                "At least one of tnf_include/te_include/cov_include must be True — "
#                "cannot run encoder with zero active modalities."
#            )
#
#        if self.fusion_mode not in ("global", "contig_weighted"):
#            raise ValueError(
#                f"fusion_mode must be 'global' or 'contig_weighted', got {self.fusion_mode!r}. "
#                f"There is no silent fallback — pick one explicitly."
#            )
#        if self.device not in ("cpu", "gpu", "auto"):
#            raise ValueError(f"device must be one of 'cpu'/'gpu'/'auto', got {self.device!r}")
#        if self.fusion_gate_entropy_weight < 0:
#            raise ValueError(f"fusion_gate_entropy_weight must be >= 0, got {self.fusion_gate_entropy_weight}")
#
#        cfg = EncoderConfig(**asdict(self))
#
#        if not cfg.tnf_include:
#            cfg.latent_dim_tnf = 0
#
#        if not cfg.te_include:
#            cfg.latent_dim_te = 0
#
#        if cfg.cov_include:
#            if cfg.latent_dim_cov is None:
#                cov_input = cfg.input_dim_cov or 3
#                cfg.latent_dim_cov = max(3, cov_input // 2)
#        else:
#            cfg.latent_dim_cov = 0
#
#        active = 0
#        if cfg.tnf_include:
#            active += cfg.latent_dim_tnf
#        if cfg.te_include:
#            active += cfg.latent_dim_te
#        if cfg.cov_include:
#            active += cfg.latent_dim_cov
#        cfg.final_dim = active
#
#        if cfg.batch_size is None:
#            cfg.batch_size = int(np.clip(n_contigs // 500, 32, 2048))
#
#        # ---------------------------------------------------------------
#        # kl_anneal_epochs / phase1_epochs mismatch guard (v7 FIX 3,
#        # unchanged in v8).
#        # ---------------------------------------------------------------
#        phase1_epoch_counts = []
#        if cfg.tnf_include:
#            phase1_epoch_counts.append(("tnf", cfg.phase1_epochs_tnf))
#        if cfg.te_include:
#            phase1_epoch_counts.append(("te", cfg.phase1_epochs_te))
#        if cfg.cov_include:
#            phase1_epoch_counts.append(("cov", cfg.phase1_epochs_cov))
#
#        for name, n_epochs in phase1_epoch_counts:
#            if cfg.kl_anneal_epochs > n_epochs:
#                effective_frac = n_epochs / cfg.kl_anneal_epochs
#                log.warning(
#                    f"FLAG:KL_ANNEAL_MISMATCH — kl_anneal_epochs={cfg.kl_anneal_epochs} > "
#                    f"phase1_epochs_{name}={n_epochs}. Training for '{name}' will END at only "
#                    f"{effective_frac:.0%} of its configured beta value (effective beta_{name} "
#                    f"≈ {effective_frac:.3f} × configured). Set kl_anneal_epochs <= phase1_epochs_{name} "
#                    f"if this is unintentional."
#                )
#
#        return cfg
#
#    def modality_str(self) -> str:
#        parts = []
#        if self.tnf_include:
#            parts.append("TNF")
#        if self.te_include:
#            parts.append("TE")
#        if self.cov_include:
#            parts.append("COV")
#        return " + ".join(parts) if parts else "NONE"
#
#    def hidden_dim(self, input_dim: int) -> int:
#        return int(np.clip(input_dim * self.hidden_scale,
#                           self.hidden_min, self.hidden_max))
#
#    def log_summary(self):
#        tnf_lat = f"TNF={self.latent_dim_tnf}" if self.tnf_include else "TNF=DISABLED"
#        tnf_in  = f"TNF={self.input_dim_tnf}"  if self.tnf_include else "TNF=n/a"
#        te_lat  = f"TE={self.latent_dim_te}"   if self.te_include  else "TE=DISABLED"
#        te_in   = f"TE={self.input_dim_te}"    if self.te_include  else "TE=n/a"
#        cov_lat = f"COV={self.latent_dim_cov}" if self.cov_include else "COV=DISABLED"
#        cov_in  = f"COV={self.input_dim_cov}"  if self.cov_include else "COV=n/a"
#        log.info("EncoderConfig:")
#        log.info(f"  tnf_include : {self.tnf_include}   te_include : {self.te_include}   cov_include : {self.cov_include}"
#                 f"   allow_missing_modalities={self.allow_missing_modalities}")
#        log.info(f"  input dims  : {tnf_in} {te_in} {cov_in}")
#        log.info(f"  latent dims : {tnf_lat} {te_lat} {cov_lat} -> FINAL={self.final_dim}")
#        beta_str = ""
#        if self.tnf_include:
#            beta_str += f" TNF={self.beta_tnf}"
#        if self.te_include:
#            beta_str += f" TE={self.beta_te}"
#        if self.cov_include:
#            beta_str += f" COV={self.beta_cov}"
#        log.info(f"  beta        :{beta_str}")
#        log.info(f"  kl_anneal   : {self.kl_anneal_epochs} epochs  free_bits={self.kl_free_bits}")
#        epoch_str = ""
#        if self.tnf_include:
#            epoch_str += f" tnf={self.phase1_epochs_tnf}"
#        if self.te_include:
#            epoch_str += f" te={self.phase1_epochs_te}"
#        if self.cov_include:
#            epoch_str += f" cov={self.phase1_epochs_cov}"
#        epoch_str += f" phase2={self.phase2_epochs}"
#        log.info(f"  epochs      :{epoch_str}")
#        log.info(f"  lr          : phase1={self.phase1_lr} phase2={self.phase2_lr}")
#        log.info(f"  loss_scale  : phase1(tnf/te/cov)={self.loss_scale_tnf}/{self.loss_scale_te}/{self.loss_scale_cov}"
#                 f"  fusion(recon)={self.loss_scale_fusion}")
#        log.info(f"  fusion_mode : {self.fusion_mode}"
#                 + (f"  gate_entropy_weight={self.fusion_gate_entropy_weight}" if self.fusion_mode == "contig_weighted" else ""))
#        log.info(f"  device      : {self.device}   batch_size={self.batch_size}  dropout={self.dropout}  seed={self.seed}")
#        log.info(f"  [v8] Phase 2: sub-encoders FROZEN, fusion trained via per-contig weighted "
#                 f"reconstruction-through-fusion loss")
#
#
## =============================================================================
## DEVICE RESOLUTION (FIX 10)
## =============================================================================
#
#def resolve_device(requested: str) -> "torch.device":
#    """
#    requested: "cpu" | "gpu" | "auto" (see EncoderConfig.device).
#      "cpu"  -> always CPU, regardless of what's available.
#      "gpu"  -> CUDA if available; else MPS with a WARNING; else CPU with a
#                WARNING. A missing GPU is an environment/hardware fact, not
#                a data-correctness risk, so this degrades rather than
#                raising — unlike FIX 8's missing-feature-file handling.
#      "auto" -> the original v7 behavior: CUDA -> MPS -> CPU, silently.
#    """
#    requested = (requested or "auto").lower()
#    if requested not in ("cpu", "gpu", "auto"):
#        raise ValueError(f"device must be one of 'cpu'/'gpu'/'auto', got {requested!r}")
#
#    if requested == "cpu":
#        device = torch.device("cpu")
#        log.info("Using CPU (explicitly requested via config)")
#    elif requested == "gpu":
#        if torch.cuda.is_available():
#            device = torch.device("cuda")
#            log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
#        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#            device = torch.device("mps")
#            log.warning("device='gpu' requested but no CUDA device is available — using Apple MPS instead")
#        else:
#            device = torch.device("cpu")
#            log.warning("device='gpu' requested but no CUDA or MPS device is available — falling back to CPU")
#    else:
#        if torch.cuda.is_available():
#            device = torch.device("cuda")
#            log.info(f"Using GPU: {torch.cuda.get_device_name(0)} (auto)")
#        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#            device = torch.device("mps")
#            log.info("Using Apple MPS (auto)")
#        else:
#            device = torch.device("cpu")
#            log.info("Using CPU (auto)")
#
#    n_threads = min(os.cpu_count() or 32, 32)
#    torch.set_num_threads(n_threads)
#    log.info(f"PyTorch threads: {n_threads}")
#    return device
#
#
#def get_device() -> "torch.device":
#    """Backward-compatible alias for resolve_device("auto") (v7's only mode)."""
#    return resolve_device("auto")
#
#
## =============================================================================
## MODALITY MANIFEST LOADING / ALIGNMENT (FIX 6)
## =============================================================================
#
#def _sha256_id_order(contig_ids: List[str]) -> str:
#    return hashlib.sha256("\n".join(contig_ids).encode()).hexdigest()[:16]
#
#
#def _load_modality_manifest(features_path: str, modality: str) -> Tuple[List[str], str]:
#    """
#    Load the authoritative, ordered contig-ID list for one modality's
#    feature file, from that modality's OWN sibling manifest files —
#    verified directly against each upstream module's real save code, not
#    assumed uniform from any docstring claim:
#
#      TNF (tnf_gene.py)      : <dir>/contig_ids.json  (JSON list)
#                                + <dir>/tnf_feature_schema.json["contig_order_hash"]
#      TE  (te_composition.py): <dir>/contig_ids.json  (JSON list)
#                                + <dir>/schema.json["contig_order_hash"]
#      COV (coverage.py)      : <dir>/contig_ids.txt   (newline-delimited
#                                text; the FULL/authoritative order — NOT
#                                raw_coverage_contig_ids.txt, which only
#                                covers the working/kept subset and does
#                                NOT match coverage_features.npy's row order)
#                                + <dir>/manifest.json["contig_id_order_hash"]
#
#    All three use the IDENTICAL hash formula
#    sha256("\\n".join(contig_ids)).hexdigest()[:16], just under different
#    key names in different files — so hashes ARE directly comparable
#    across modalities. That comparison (done by the caller, across
#    modalities) is the real alignment check; the check against each
#    modality's OWN stored hash here is only a self-consistency sanity
#    check (did this contig_ids file get edited/regenerated out of sync
#    with its own recorded hash).
#    """
#    if modality not in ("tnf", "te", "cov"):
#        raise ValueError(f"_load_modality_manifest: unknown modality {modality!r}")
#
#    d = Path(features_path).parent
#    if modality == "tnf":
#        ids_path, manifest_path, hash_key = d / "contig_ids.json", d / "tnf_feature_schema.json", "contig_order_hash"
#    elif modality == "te":
#        ids_path, manifest_path, hash_key = d / "contig_ids.json", d / "schema.json", "contig_order_hash"
#    else:  # cov
#        ids_path, manifest_path, hash_key = d / "contig_ids.txt", d / "manifest.json", "contig_id_order_hash"
#
#    if not ids_path.exists():
#        raise FileNotFoundError(
#            f"{modality.upper()}: expected contig-ID manifest at {ids_path} (sibling of "
#            f"{features_path}) but it does not exist — cannot verify contig alignment against "
#            f"the other modalities. Refusing to assume row order without it."
#        )
#
#    if modality == "cov":
#        contig_ids = [line for line in ids_path.read_text().splitlines() if line != ""]
#    else:
#        with open(ids_path) as f:
#            contig_ids = json.load(f)
#
#    recomputed_hash = _sha256_id_order(contig_ids)
#
#    stored_hash = None
#    if manifest_path.exists():
#        try:
#            with open(manifest_path) as f:
#                stored_hash = json.load(f).get(hash_key)
#        except Exception as e:
#            log.warning(f"{modality.upper()}: could not read {manifest_path} ({e}) — skipping "
#                        f"self-consistency hash check (the contig_ids file itself is still used).")
#    else:
#        log.warning(f"{modality.upper()}: {manifest_path} not found — skipping self-consistency "
#                    f"hash check (the contig_ids file itself is still used).")
#
#    if stored_hash is not None and stored_hash != recomputed_hash:
#        raise ValueError(
#            f"{modality.upper()}: contig_ids at {ids_path} do not match this modality's own "
#            f"recorded '{hash_key}' in {manifest_path} ({recomputed_hash} != {stored_hash}) — "
#            f"the contig-ID file may have been edited or regenerated out of sync with the "
#            f"feature array. Refusing to proceed."
#        )
#
#    return contig_ids, recomputed_hash
#
#
#def _check_contig_alignment(ids_by_mod: Dict[str, Tuple[List[str], str]]) -> Tuple[List[str], str]:
#    """Cross-modality alignment check (replaces v7's shape[0]-only check).
#    Returns (contig_ids, order_hash) of the shared, verified order."""
#    if not ids_by_mod:
#        raise RuntimeError("_check_contig_alignment: no active modality to align")
#
#    ref_mod = next(iter(ids_by_mod))
#    ref_ids, ref_hash = ids_by_mod[ref_mod]
#    for mod, (ids, h) in ids_by_mod.items():
#        if h != ref_hash or ids != ref_ids:
#            raise ValueError(
#                f"Contig alignment mismatch: '{mod}' contig-ID order does not match '{ref_mod}'. "
#                f"'{ref_mod}' has {len(ref_ids)} contigs (order_hash={ref_hash}); "
#                f"'{mod}' has {len(ids)} contigs (order_hash={h}). All modalities must be computed "
#                f"from the exact same assembly with the exact same contig ordering."
#            )
#    return ref_ids, ref_hash
#
#
## =============================================================================
## COVERAGE FEATURE SPLIT (FIX 5)
## =============================================================================
#
#def split_cov_features(cov_raw: np.ndarray, cfg: "EncoderConfig") -> Tuple[np.ndarray, np.ndarray]:
#    """
#    coverage.py's coverage_features.npy is always
#    [valid_mask | n_samples_present | mean_dist | std_dist | cov_1..cov_N]
#    (see coverage.py's COLUMN_LAYOUT_METADATA_NAMES / COLUMN_ROLES).
#    valid_mask is documented as MASK ONLY, never a feature. n_samples_present
#    is a QC/filtering signal by default; this module leaves it out of the
#    feature vector unless cfg.cov_use_n_samples_present_as_feature is
#    explicitly set (coverage.py's own manifest note: "column/layout
#    SELECTION is the consumer's decision").
#
#    Returns (features_for_vae, reliability_weight) — reliability_weight is
#    exactly valid_mask (1.0 real row / 0.0 placeholder row), used as COV's
#    per-contig weight in both Phase 1 (vae_loss masking) and Phase 2
#    (readback_loss weighting).
#    """
#    if cov_raw.ndim != 2 or cov_raw.shape[1] < COV_METADATA_DIM + 1:
#        raise ValueError(
#            f"COV: expected >= {COV_METADATA_DIM + 1} columns "
#            f"(valid_mask, n_samples_present, mean_dist, std_dist + >=1 coverage column), "
#            f"got shape {cov_raw.shape}"
#        )
#
#    valid_mask        = cov_raw[:, 0].astype(np.float32)
#    n_samples_present = cov_raw[:, 1].astype(np.float32)
#    dist_and_cov       = cov_raw[:, 2:]  # mean_dist, std_dist, cov_1..cov_N
#
#    if cfg.cov_use_n_samples_present_as_feature:
#        features = np.concatenate(
#            [n_samples_present.reshape(-1, 1), dist_and_cov], axis=1
#        ).astype(np.float32)
#    else:
#        features = dist_and_cov.astype(np.float32)
#
#    n_placeholder = int((valid_mask == 0).sum())
#    if n_placeholder:
#        log.info(f"COV: {n_placeholder:,}/{len(valid_mask):,} contig(s) are valid_mask=0 "
#                 f"placeholders — excluded from COV's own loss via reliability weight=0.")
#
#    return features, valid_mask.copy()
#
#
## =============================================================================
## MISSING-MODALITY / SMALL-DATASET GUARDS (FIX 8 / FIX 12)
## =============================================================================
#
#def _check_modality_available(cfg_flag: bool, path: Optional[str], name: str,
#                               allow_missing: bool) -> bool:
#    if not cfg_flag:
#        return False
#    if path is None or not Path(path).exists():
#        msg = (f"{name}_include=True but {name}_features_path is None or missing ({path}) — "
#               f"cannot proceed with this modality active.")
#        if allow_missing:
#            log.warning(msg + " allow_missing_modalities=True — disabling this modality and continuing.")
#            return False
#        raise FileNotFoundError(
#            msg + " Set allow_missing_modalities=True in config if silently dropping this "
#            "modality is genuinely intended; otherwise supply the missing path."
#        )
#    return True
#
#
#def _effective_batch_size(n: int, requested: int) -> int:
#    """Small-dataset guard: never request a batch size larger than the
#    dataset itself. Makes the actual effective batch size explicit instead
#    of relying on DataLoader to silently hand back one small final batch."""
#    eff = max(1, min(int(requested), int(n)))
#    if eff != requested:
#        log.warning(f"batch_size clamped from {requested} to {eff} (dataset has only {n} row(s))")
#    return eff
#
#
#def _guard_nonzero_weights(weights: Optional[np.ndarray], name: str) -> None:
#    if weights is not None and not bool(np.any(weights > 0)):
#        raise ValueError(
#            f"{name}: every per-contig weight is 0 — every contig would be masked out of the "
#            f"loss for this modality. Refusing to train on nothing; check the upstream "
#            f"weight/mask computation for {name}."
#        )
#
#
## =============================================================================
## UTILITY  (validate_features/validate_weights/normalize_features/
## make_loader/get_all_latents — unchanged from v7)
## =============================================================================
#
#def validate_features(arr: np.ndarray, name: str,
#                       expected_dim: Optional[int] = None) -> np.ndarray:
#    if arr is None:
#        raise ValueError(f"{name}: feature array is None")
#    arr = np.array(arr, dtype=np.float32)
#    if arr.ndim != 2:
#        raise ValueError(f"{name}: expected 2D array, got {arr.shape}")
#    if arr.shape[0] == 0:
#        # FIX 19: an empty feature array would otherwise hit .min()/.max()/
#        # mean() below on an empty axis, raising an opaque numpy error
#        # ("zero-size array to reduction operation") far from the real
#        # cause. Fail loud with a message that actually names the problem.
#        raise ValueError(f"{name}: feature array has 0 rows (shape={arr.shape}) -- nothing to "
#                          f"validate or train on. Check the upstream feature file.")
#    if expected_dim is not None and arr.shape[1] != expected_dim:
#        raise ValueError(f"{name}: expected dim={expected_dim}, got {arr.shape[1]}")
#    n_nan = int(np.isnan(arr).sum())
#    n_inf = int(np.isinf(arr).sum())
#    if n_nan > 0 or n_inf > 0:
#        log.warning(f"{name}: replacing {n_nan} NaN + {n_inf} Inf with 0")
#        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
#    zero_rows = int((arr == 0).all(axis=1).sum())
#    pct       = zero_rows / arr.shape[0] * 100
#    if pct > 50:
#        log.warning(f"{name}: {zero_rows:,}/{arr.shape[0]:,} ({pct:.0f}%) rows all-zero "
#                    f"(expected for sparse signals like TE)")
#    log.info(f"{name}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} "
#             f"mean={arr.mean():.4f} zero_rows={zero_rows:,}")
#    return arr
#
#
#def validate_weights(arr: np.ndarray, name: str) -> Optional[np.ndarray]:
#    if arr is None:
#        log.warning(f"{name}: no weights provided -- using uniform 1.0")
#        return None
#    arr = np.clip(np.array(arr, dtype=np.float32).squeeze(), 0.0, 1.0)
#    log.info(f"{name}: zero={int((arr==0).sum()):,} full={int((arr==1).sum()):,} mean={arr.mean():.3f}")
#    return arr
#
#
#def normalize_features(arr: np.ndarray, name: str, mask: Optional[np.ndarray] = None):
#    """
#    FIX 16: when `mask` is supplied (e.g. COV's valid_mask), the
#    normalization statistics (mean/std) are computed ONLY over rows where
#    mask > 0. Without this, placeholder rows (COV's valid_mask=0 rows are
#    literal zero-filled placeholders inserted by coverage.py's step7_realign
#    -- not real data) would contaminate the mean/std that every row,
#    including the real ones, is then normalized against. The resulting
#    normalized values are still computed for every row (including
#    placeholders) using those valid-only statistics -- only the STATISTICS
#    themselves exclude invalid rows, not the transform's output rows.
#    """
#    if mask is not None:
#        valid = mask > 0
#        n_valid = int(valid.sum())
#        if n_valid == 0:
#            raise ValueError(f"{name}: normalize_features got a mask with zero valid rows -- "
#                              f"cannot compute normalization statistics from nothing.")
#        mean = arr[valid].mean(axis=0)
#        std  = arr[valid].std(axis=0)
#    else:
#        mean = arr.mean(axis=0)
#        std  = arr.std(axis=0)
#    std[std == 0] = 1.0
#    norm = ((arr - mean) / std).astype(np.float32)
#    if mask is not None:
#        log.info(f"{name}: normalized to zero mean unit variance "
#                 f"(statistics computed over {n_valid:,}/{arr.shape[0]:,} valid rows only)")
#    else:
#        log.info(f"{name}: normalized to zero mean unit variance")
#    return norm, mean, std
#
#
#def make_loader(features, weights, batch_size, shuffle=True):
#    x  = torch.tensor(features, dtype=torch.float32)
#    ds = (TensorDataset(x, torch.tensor(weights, dtype=torch.float32))
#          if weights is not None else TensorDataset(x))
#    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
#
#
#def get_all_latents(model, features, device, batch_size):
#    model.eval()
#    loader = DataLoader(
#        TensorDataset(torch.tensor(features, dtype=torch.float32)),
#        batch_size=batch_size, shuffle=False)
#    parts = []
#    with torch.no_grad():
#        for (x,) in loader:
#            parts.append(model.get_latent(x.to(device)).cpu().numpy())
#    return np.concatenate(parts, axis=0)
#
#
## =============================================================================
## UNIFORM beta-VAE BASE  (CORE — unchanged, diagnosed healthy)
## =============================================================================
#
#def _make_encoder_layers(input_dim: int, cfg: EncoderConfig):
#    hidden = cfg.hidden_dim(input_dim)
#    layers = []
#    prev   = input_dim
#    for i in range(cfg.n_hidden_layers):
#        dim = hidden if i == 0 else max(cfg.hidden_min, hidden // 2)
#        layers += [nn.Linear(prev, dim), nn.LayerNorm(dim), nn.ReLU()]
#        prev = dim
#    return nn.Sequential(*layers), prev
#
#
#def _make_decoder_layers(latent_dim: int, output_dim: int, cfg: EncoderConfig) -> nn.Sequential:
#    hidden = cfg.hidden_dim(output_dim)
#    layers = []
#    prev   = latent_dim
#    dims   = ([max(cfg.hidden_min, hidden // 2)] * (cfg.n_hidden_layers - 1) + [hidden]
#              if cfg.n_hidden_layers > 1 else [hidden])
#    for dim in dims:
#        layers += [nn.Linear(prev, dim), nn.LayerNorm(dim), nn.ReLU()]
#        prev = dim
#    layers += [nn.Linear(prev, output_dim), nn.Sigmoid()]
#    return nn.Sequential(*layers)
#
#
#class UniformVAE(nn.Module):
#    def __init__(self, input_dim: int, latent_dim: int, cfg: EncoderConfig):
#        super().__init__()
#        self.input_dim  = input_dim
#        self.latent_dim = latent_dim
#        enc_layers, enc_out = _make_encoder_layers(input_dim, cfg)
#        self.encoder    = enc_layers
#        self.fc_mu      = nn.Linear(enc_out, latent_dim)
#        self.fc_log_var = nn.Linear(enc_out, latent_dim)
#        self.decoder    = _make_decoder_layers(latent_dim, input_dim, cfg)
#
#    def encode(self, x):
#        h       = self.encoder(x)
#        mu      = self.fc_mu(h)
#        log_var = self.fc_log_var(h).clamp(-10, 10)
#        return mu, log_var
#
#    def reparameterize(self, mu, log_var):
#        if self.training:
#            return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
#        return mu
#
#    def forward(self, x):
#        mu, log_var = self.encode(x)
#        z           = self.reparameterize(mu, log_var)
#        return self.decoder(z), mu, log_var
#
#    def get_latent(self, x):
#        self.eval()
#        with torch.no_grad():
#            mu, _ = self.encode(x)
#        return mu
#
#
## =============================================================================
## FUSION MODULE  (FIX 9: per-contig weighted readback loss;
##                 FIX 13: contig_weighted gate mode)
## =============================================================================
#
#class FusionModule(nn.Module):
#    """
#    Gated fusion for any active combination of TNF, TE, COV (all
#    independently optional — at least one must be True). Canonical order:
#    TNF -> TE -> COV.
#
#    Two fusion_mode variants, selected by EncoderConfig.fusion_mode:
#
#      "global"          — v7's original mechanism: ONE learned softmax
#                           gate (nn.Parameter), shared by every contig.
#
#      "contig_weighted" — NEW (v8): a small GMU-style (Arevalo et al. 2017)
#                           MLP gate that takes each contig's own
#                           concatenated projected latents and outputs a
#                           PER-CONTIG softmax over modalities. Trained via
#                           the identical reconstruction-through-fusion
#                           loss as "global" (see readback_loss), plus an
#                           optional entropy penalty (gate_entropy) to
#                           discourage per-contig collapse onto one
#                           modality for every contig.
#
#    Both modes keep the v7 readback_X heads (final_dim -> latent_dim_X),
#    used only during Phase 2 training, never at inference.
#    """
#    def __init__(self, cfg: EncoderConfig):
#        super().__init__()
#        self.tnf_include = cfg.tnf_include
#        self.te_include  = cfg.te_include
#        self.cov_include = cfg.cov_include
#        self.fusion_mode = cfg.fusion_mode
#
#        if not (self.tnf_include or self.te_include or self.cov_include):
#            raise ValueError("FusionModule: at least one of tnf_include/te_include/cov_include must be True")
#        if self.fusion_mode not in ("global", "contig_weighted"):
#            raise ValueError(f"FusionModule: unknown fusion_mode {cfg.fusion_mode!r}")
#
#        self._mods = []
#        in_dim = 0
#        if cfg.tnf_include:
#            self._mods.append("tnf")
#            in_dim += cfg.latent_dim_tnf
#        if cfg.te_include:
#            self._mods.append("te")
#            in_dim += cfg.latent_dim_te
#        if cfg.cov_include:
#            self._mods.append("cov")
#            in_dim += cfg.latent_dim_cov
#
#        n_mods = len(self._mods)
#
#        if self.fusion_mode == "global":
#            self.gate_logits = nn.Parameter(torch.zeros(n_mods))
#            self.gate_net = None
#        else:  # contig_weighted
#            self.gate_logits = None
#            gate_hidden = max(8, in_dim // 2)
#            self.gate_net = nn.Sequential(
#                nn.Linear(in_dim, gate_hidden),
#                nn.LayerNorm(gate_hidden),
#                nn.ReLU(),
#                nn.Linear(gate_hidden, n_mods),
#            )
#        # Last-forward per-contig gate weights, kept (with grad) for
#        # gate_entropy() and (detached) for saving contig_fusion_weights.npy.
#        # Populated by forward(); shape (1, n_mods) in "global" mode,
#        # (batch, n_mods) in "contig_weighted" mode.
#        self._last_gate_weights = None
#
#        if cfg.tnf_include:
#            self.proj_tnf = nn.Linear(cfg.latent_dim_tnf, cfg.latent_dim_tnf)
#        if cfg.te_include:
#            self.proj_te = nn.Linear(cfg.latent_dim_te, cfg.latent_dim_te)
#        if cfg.cov_include:
#            self.proj_cov = nn.Linear(cfg.latent_dim_cov, cfg.latent_dim_cov)
#
#        self.fusion = nn.Sequential(
#            nn.Linear(in_dim, cfg.final_dim),
#            nn.LayerNorm(cfg.final_dim),
#            nn.ReLU(),
#            nn.Dropout(cfg.dropout),
#        )
#
#        # Readback heads (v7 FIX 2): fused space -> each modality's own
#        # latent dimensionality. Training-only; not used at inference.
#        if cfg.tnf_include:
#            self.readback_tnf = nn.Linear(cfg.final_dim, cfg.latent_dim_tnf)
#        if cfg.te_include:
#            self.readback_te = nn.Linear(cfg.final_dim, cfg.latent_dim_te)
#        if cfg.cov_include:
#            self.readback_cov = nn.Linear(cfg.final_dim, cfg.latent_dim_cov)
#
#    def forward(self, z_tnf=None, z_te=None, z_cov=None):
#        proj = {}
#        if self.tnf_include:
#            if z_tnf is None:
#                raise ValueError("tnf_include=True but z_tnf is None in FusionModule.forward()")
#            proj["tnf"] = self.proj_tnf(z_tnf)
#        if self.te_include:
#            if z_te is None:
#                raise ValueError("te_include=True but z_te is None in FusionModule.forward()")
#            proj["te"] = self.proj_te(z_te)
#        if self.cov_include:
#            if z_cov is None:
#                raise ValueError("cov_include=True but z_cov is None in FusionModule.forward()")
#            proj["cov"] = self.proj_cov(z_cov)
#
#        ordered = [proj[m] for m in self._mods]
#        concat_raw = torch.cat(ordered, dim=1)  # (batch, in_dim)
#
#        if self.fusion_mode == "global":
#            w = F.softmax(self.gate_logits, dim=0)              # (n_mods,)
#            gated = [ordered[i] * w[i] for i in range(len(ordered))]
#            self._last_gate_weights = w.unsqueeze(0)             # (1, n_mods)
#        else:  # contig_weighted
#            gate_logits_c = self.gate_net(concat_raw)            # (batch, n_mods)
#            w = F.softmax(gate_logits_c, dim=1)                  # (batch, n_mods)
#            gated = [ordered[i] * w[:, i:i + 1] for i in range(len(ordered))]
#            self._last_gate_weights = w                          # (batch, n_mods)
#
#        z = torch.cat(gated, dim=1)
#        return self.fusion(z)
#
#    def readback_loss(self, z_fus, mu_tnf=None, mu_te=None, mu_cov=None,
#                       w_tnf=None, w_te=None, w_cov=None):
#        """
#        v8: PER-CONTIG weighted-reliability readback loss (FIX 9). For each
#        active modality X, computes a per-contig reconstruction error
#        err_X_i = MSE over that modality's own latent dims between
#        readback_X(z_fus)_i and mu_X_i.detach(). These per-contig errors
#        are combined into ONE scalar per contig using each modality's OWN
#        reliability weight w_X_i — NOT the fusion gate weight, a different
#        concept (see module docstring):
#
#            contig_loss_i = (w_tnf_i*err_tnf_i + w_te_i*err_te_i + w_cov_i*err_cov_i)
#                            / (w_tnf_i + w_te_i + w_cov_i)
#
#        Contigs with total weight 0 across every active modality
#        contribute nothing to the batch mean (they have no reliable target
#        to reconstruct toward at all). A weight defaults to 1.0 for any
#        modality that didn't supply one (e.g. TNF/TE with no confidence
#        weight file).
#
#        This is a WEIGHTED-AVERAGE policy, not an intersection/all-valid
#        policy — TE's reliability weight is near-all-zero for fragmented
#        real assemblies by design (te_composition.py), so requiring every
#        modality to be valid before a contig counts at all would starve
#        Phase 2 of nearly all its training data.
#        """
#        batch_n = z_fus.shape[0]
#        device  = z_fus.device
#        total_w    = torch.zeros(batch_n, device=device)
#        total_werr = torch.zeros(batch_n, device=device)
#
#        if self.tnf_include and mu_tnf is not None:
#            err = F.mse_loss(self.readback_tnf(z_fus), mu_tnf.detach(), reduction="none").mean(dim=1)
#            w = w_tnf if w_tnf is not None else torch.ones(batch_n, device=device)
#            total_werr = total_werr + w * err
#            total_w    = total_w + w
#        if self.te_include and mu_te is not None:
#            err = F.mse_loss(self.readback_te(z_fus), mu_te.detach(), reduction="none").mean(dim=1)
#            w = w_te if w_te is not None else torch.ones(batch_n, device=device)
#            total_werr = total_werr + w * err
#            total_w    = total_w + w
#        if self.cov_include and mu_cov is not None:
#            err = F.mse_loss(self.readback_cov(z_fus), mu_cov.detach(), reduction="none").mean(dim=1)
#            w = w_cov if w_cov is not None else torch.ones(batch_n, device=device)
#            total_werr = total_werr + w * err
#            total_w    = total_w + w
#
#        has_weight = (total_w > 0).float()
#        n_valid    = has_weight.sum().clamp(min=1.0)
#        per_contig = torch.where(total_w > 0, total_werr / total_w.clamp(min=1e-8),
#                                  torch.zeros_like(total_w))
#        return (per_contig * has_weight).sum() / n_valid
#
#    def gate_entropy(self) -> "torch.Tensor":
#        """
#        FIX 13 safeguard: an entropy-based penalty for the contig_weighted
#        gate, loosely inspired by sparsity/collapse safeguards in
#        confidence-weighted per-instance gating approaches (not a literal
#        reproduction of any one paper). Only meaningful when
#        fusion_mode="contig_weighted" — a per-contig softmax that always
#        collapses to one-hot for every contig has effectively re-derived a
#        "global" gate at extra parameter/compute cost, and is a known
#        failure mode for unsupervised instance-wise gates trained purely
#        through reconstruction. Returns the NEGATIVE mean per-contig
#        entropy, so that ADDING `fusion_gate_entropy_weight * gate_entropy()`
#        to the loss (i.e. minimizing it) pushes entropy UP, discouraging
#        collapse. Returns 0 (no gradient effect) in "global" mode.
#        """
#        if self.fusion_mode != "contig_weighted" or self._last_gate_weights is None:
#            return torch.zeros((), device=next(self.parameters()).device)
#        w = self._last_gate_weights.clamp(min=1e-8)
#        entropy = -(w * w.log()).sum(dim=1)   # per-contig entropy, shape (batch,)
#        return -entropy.mean()
#
#    def get_contig_weights(self) -> Optional[np.ndarray]:
#        """Per-contig gate weights from the most recent forward() call, as
#        a numpy array of shape (batch, n_mods) — only meaningful (and only
#        non-None) in fusion_mode='contig_weighted'."""
#        if self.fusion_mode != "contig_weighted" or self._last_gate_weights is None:
#            return None
#        return self._last_gate_weights.detach().cpu().numpy()
#
#    def get_weights(self) -> dict:
#        if self.fusion_mode == "global":
#            w = F.softmax(self.gate_logits, dim=0).detach()
#            return {f"w_{m}": float(w[i]) for i, m in enumerate(self._mods)}
#        # contig_weighted: report the mean +/- std over the most recent forward's batch.
#        if self._last_gate_weights is None:
#            return {f"w_{m}_mean": None for m in self._mods}
#        w = self._last_gate_weights.detach()
#        out = {}
#        for i, m in enumerate(self._mods):
#            out[f"w_{m}_mean"] = float(w[:, i].mean())
#            out[f"w_{m}_std"]  = float(w[:, i].std())
#        return out
#
#
## =============================================================================
## LOSS FUNCTIONS  (CORE — unchanged, still used in Phase 1)
## =============================================================================
#
#def vae_loss(recon, original, mu, log_var, weights, beta, kl_free_bits):
#    recon_per  = F.mse_loss(recon, original, reduction="none").mean(dim=1)
#    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
#    kl_per_dim = kl_per_dim.clamp(min=kl_free_bits)
#    kl_per     = kl_per_dim.sum(dim=1)
#
#    if weights is not None:
#        mask       = (weights > 0).float()
#        n_act      = mask.sum().clamp(min=1.0)
#        recon_loss = (recon_per * mask).sum() / n_act
#        kl_loss    = (kl_per   * mask).sum() / n_act
#    else:
#        recon_loss = recon_per.mean()
#        kl_loss    = kl_per.mean()
#
#    return recon_loss + beta * kl_loss, recon_loss, kl_loss
#
#
## =============================================================================
## TRAINING FUNCTIONS  (CORE — Phase 1 loop body unchanged; batch_size is now
## an explicit parameter so the small-dataset clamp (FIX 12) can apply to it)
## =============================================================================
#
#def _train_vae_epoch(model, loader, optimizer, device, beta, kl_free_bits, has_weights):
#    model.train()
#    tot = rec = kl = 0.0
#    n   = 0
#    for batch in loader:
#        x   = batch[0].to(device)
#        w   = batch[1].to(device) if has_weights else None
#        optimizer.zero_grad()
#        recon, mu, lv = model(x)
#        loss, rl, kll = vae_loss(recon, x, mu, lv, w, beta, kl_free_bits)
#        loss.backward()
#        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#        optimizer.step()
#        tot += loss.item(); rec += rl.item(); kl += kll.item(); n += 1
#    return tot / n, rec / n, kl / n
#
#
#def _train_phase1(model, features, weights, cfg, device, epochs, beta, name, batch_size):
#    loader    = make_loader(features, weights, batch_size)
#    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.phase1_lr)
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#        optimizer, T_max=epochs, eta_min=cfg.phase1_lr * 0.1)
#    has_w = weights is not None
#    t0    = time.time()
#
#    for epoch in range(1, epochs + 1):
#        b            = min(beta, beta * epoch / max(cfg.kl_anneal_epochs, 1))
#        tot, rec, kl = _train_vae_epoch(model, loader, optimizer, device, b, cfg.kl_free_bits, has_w)
#        scheduler.step()
#        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
#            log.info(f"  {name} ep {epoch:3d}/{epochs} | loss={tot:.4f} recon={rec:.4f} kl={kl:.4f} "
#                     f"beta={b:.3f} lr={scheduler.get_last_lr()[0]:.2e}")
#
#    log.info(f"  {name} done in {time.time()-t0:.0f}s")
#
#
## =============================================================================
## PHASE 1 -- INDEPENDENT ENCODER TRAINING
## (fingerprint-aware checkpointing added — FIX 15; spawn context — FIX 11)
## =============================================================================
#
#def _phase1_worker(job):
#    """
#    Top-level, picklable worker used by ProcessPoolExecutor to train one
#    modality's Phase-1 VAE in a separate process (CPU-only parallelism by
#    default; also works with device='gpu', see FIX 11's note about
#    per-worker GPU contention).
#
#    job: (name, features, weights, latent_dim, beta, epochs,
#          cfg, device, ckpt_dir, ckpt_key, fp, n_threads)
#
#    Returns (name, model, latent) with model already moved to CPU so it
#    pickles back cleanly regardless of how/where it was trained.
#    """
#    (name, features, weights, latent_dim, beta, epochs,
#     cfg, device, ckpt_dir, ckpt_key, fp, n_threads) = job
#    try:
#        torch.set_num_threads(max(1, n_threads))
#    except Exception:
#        pass
#    torch.manual_seed(cfg.seed)
#    np.random.seed(cfg.seed)
#    model, latent = _phase1_single(name, features, weights, latent_dim, beta, epochs,
#                                     cfg, device, ckpt_dir, ckpt_key, fp)
#    model.to("cpu")
#    return name, model, latent
#
#
#def _phase1_single(name, features, weights, latent_dim, beta, epochs,
#                    cfg, device, ckpt_dir, ckpt_key, fp):
#    ckpt     = Checkpoint(ckpt_dir / name) if Checkpoint else None
#    lat_path = ckpt_dir / name / f"latent_{name}.npy"
#    mdl_path = ckpt_dir / name / f"{name}_model.pt"
#
#    cached = _checkpoint_ok(ckpt, ckpt_key, fp, [str(lat_path), str(mdl_path)])
#    if cached is not None:
#        log.info(f"  [SKIP] [{ckpt_key}] (completed {cached.get('timestamp', '')})")
#        model = UniformVAE(features.shape[1], latent_dim, cfg).to(device)
#        model.load_state_dict(torch.load(mdl_path, map_location=device))
#        return model, np.load(lat_path)
#
#    log.info(f"Phase 1 [{name}]: input={features.shape[1]}D -> latent={latent_dim}D "
#             f"hidden={cfg.hidden_dim(features.shape[1])}")
#    model = UniformVAE(features.shape[1], latent_dim, cfg).to(device)
#    eff_bs = _effective_batch_size(features.shape[0], cfg.batch_size)
#    _train_phase1(model, features, weights, cfg, device, epochs, beta, name, eff_bs)
#
#    latent = get_all_latents(model, features, device, eff_bs)
#    log.info(f"  [{name}] latent std={latent.std():.4f}")
#
#    (ckpt_dir / name).mkdir(parents=True, exist_ok=True)
#    np.save(lat_path, latent)
#    torch.save(model.state_dict(), mdl_path)
#    if ckpt:
#        ckpt.mark_done(ckpt_key, {"_fp": fp, "latent_std": float(latent.std()), "epochs": epochs})
#
#    return model, latent
#
#
## =============================================================================
## PHASE 2 -- JOINT FUSION TRAINING
## (FIX 9 weighted readback loss, FIX 13 contig_weighted gate + entropy term,
##  FIX 14 fusion checkpoint persistence, FIX 15 fingerprinting)
## =============================================================================
#
#def _phase2_joint(fusion, cfg, device, ckpt_dir, outdir, fp2,
#                   tnf_model=None, tnf_feat=None, tnf_w=None,
#                   te_model=None, te_feat=None, te_w=None,
#                   cov_model=None, cov_feat=None, cov_w=None):
#    """
#    v8. FIX 1/FIX 2 mechanics (frozen sub-encoders, reconstruction-through-
#    fusion as the only training signal) are unchanged from v7. New in v8:
#
#    FIX 9: readback_loss is now called with each active modality's
#    per-contig reliability weight (tnf_w/te_w/cov_w — defaulting to all-1
#    tensors when a modality has no supplied weight array), combined
#    per-contig rather than as one whole-batch mean per modality.
#
#    FIX 13: when cfg.fusion_mode == "contig_weighted", an entropy penalty
#    (fusion.gate_entropy(), scaled by cfg.fusion_gate_entropy_weight) is
#    added to the loss, and the full-dataset per-contig gate weights are
#    saved to `<outdir>/contig_fusion_weights.npy`.
#
#    FIX 14: the trained FusionModule's weights are now saved into this
#    step's own checkpoint dir and reloaded into `fusion` on a cache hit,
#    so a Phase-2 cache hit can no longer leave the caller holding an
#    untrained fusion module while returning already-trained latents.
#
#    FIX 15: checkpoint validity now depends on fp2 (a fingerprint of every
#    Phase-2-relevant input/config value, including the upstream Phase-1
#    fingerprints), not just is_done()+file-exists.
#    """
#    ckpt          = Checkpoint(ckpt_dir / "joint") if Checkpoint else None
#    lat_path      = ckpt_dir / "joint" / "final_latent.npy"
#    fusion_path   = ckpt_dir / "joint" / "fusion_model.pt"
#
#    cached = _checkpoint_ok(ckpt, "joint_fusion", fp2, [str(lat_path), str(fusion_path)])
#    if cached is not None:
#        log.info("[SKIP] Phase 2 already done -- loading from checkpoint")
#        fusion.load_state_dict(torch.load(fusion_path, map_location=device))  # FIX 14
#        return np.load(lat_path)
#
#    tnf_active = cfg.tnf_include and tnf_model is not None
#    te_active  = cfg.te_include  and te_model  is not None
#    cov_active = cfg.cov_include and cov_model is not None
#
#    if not (tnf_active or te_active or cov_active):
#        raise ValueError("_phase2_joint: no active modality (tnf/te/cov all inactive)")
#
#    modality_str = ("TNF" if tnf_active else "") + ("+TE" if te_active else "") + ("+COV" if cov_active else "")
#    log.info(f"Phase 2: Joint fusion training ({modality_str.lstrip('+')}) -- "
#             f"[v8] sub-encoders FROZEN, gate='{cfg.fusion_mode}', "
#             f"per-contig weighted reconstruction-through-fusion loss")
#
#    # --- freeze all sub-encoders permanently for Phase 2 (v7 FIX 1, unchanged) ---
#    if tnf_active:
#        tnf_model.eval()
#        for p in tnf_model.parameters():
#            p.requires_grad_(False)
#    if te_active:
#        te_model.eval()
#        for p in te_model.parameters():
#            p.requires_grad_(False)
#    if cov_active:
#        cov_model.eval()
#        for p in cov_model.parameters():
#            p.requires_grad_(False)
#
#    all_p = list(fusion.parameters())
#    opt   = torch.optim.Adam(all_p, lr=cfg.phase2_lr)
#    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.phase2_epochs, eta_min=cfg.phase2_lr * 0.1)
#
#    # --- Precompute frozen mu's ONCE (v7 speed win + correctness, unchanged) ---
#    mu_tnf_full = mu_te_full = mu_cov_full = None
#    with torch.no_grad():
#        if tnf_active:
#            x_all = torch.tensor(tnf_feat, dtype=torch.float32).to(device)
#            mu_tnf_full, _ = tnf_model.encode(x_all)
#            mu_tnf_full = mu_tnf_full.cpu()
#        if te_active:
#            x_all = torch.tensor(te_feat, dtype=torch.float32).to(device)
#            mu_te_full, _ = te_model.encode(x_all)
#            mu_te_full = mu_te_full.cpu()
#        if cov_active:
#            x_all = torch.tensor(cov_feat, dtype=torch.float32).to(device)
#            mu_cov_full, _ = cov_model.encode(x_all)
#            mu_cov_full = mu_cov_full.cpu()
#
#    n_ref = (mu_tnf_full.shape[0] if tnf_active else
#             mu_te_full.shape[0]  if te_active  else
#             mu_cov_full.shape[0])
#
#    # --- FIX 9: per-contig reliability weight tensors, row-aligned with the
#    # mu's above, defaulting to all-ones when a modality has no weight array.
#    w_tnf_full = torch.tensor(tnf_w, dtype=torch.float32) if (tnf_active and tnf_w is not None) \
#        else (torch.ones(n_ref) if tnf_active else None)
#    w_te_full = torch.tensor(te_w, dtype=torch.float32) if (te_active and te_w is not None) \
#        else (torch.ones(n_ref) if te_active else None)
#    w_cov_full = torch.tensor(cov_w, dtype=torch.float32) if (cov_active and cov_w is not None) \
#        else (torch.ones(n_ref) if cov_active else None)
#
#    total_w_full = torch.zeros(n_ref)
#    if tnf_active: total_w_full = total_w_full + w_tnf_full
#    if te_active:  total_w_full = total_w_full + w_te_full
#    if cov_active: total_w_full = total_w_full + w_cov_full
#    if not bool((total_w_full > 0).any()):
#        raise ValueError(
#            "_phase2_joint: every contig has total reliability weight 0 across all active "
#            "modalities -- Phase 2's readback loss would have nothing to train on."
#        )
#
#    tensors = []
#    if tnf_active: tensors.append(mu_tnf_full); tensors.append(w_tnf_full)
#    if te_active:  tensors.append(mu_te_full);  tensors.append(w_te_full)
#    if cov_active: tensors.append(mu_cov_full); tensors.append(w_cov_full)
#
#    effective_batch_size = _effective_batch_size(n_ref, cfg.batch_size)
#    drop_last = effective_batch_size < n_ref
#    loader = DataLoader(TensorDataset(*tensors), batch_size=effective_batch_size,
#                         shuffle=True, drop_last=drop_last)
#
#    use_entropy = cfg.fusion_mode == "contig_weighted" and cfg.fusion_gate_entropy_weight > 0
#    t0 = time.time()
#
#    for epoch in range(1, cfg.phase2_epochs + 1):
#        fusion.train()
#        ep_tot = 0.0
#        nb = 0
#
#        for batch in loader:
#            idx = 0
#            mu_tnf_b = w_tnf_b = None
#            if tnf_active:
#                mu_tnf_b = batch[idx].to(device); idx += 1
#                w_tnf_b  = batch[idx].to(device); idx += 1
#            mu_te_b = w_te_b = None
#            if te_active:
#                mu_te_b = batch[idx].to(device); idx += 1
#                w_te_b  = batch[idx].to(device); idx += 1
#            mu_cov_b = w_cov_b = None
#            if cov_active:
#                mu_cov_b = batch[idx].to(device); idx += 1
#                w_cov_b  = batch[idx].to(device); idx += 1
#
#            opt.zero_grad()
#
#            z_fus = fusion(z_tnf=mu_tnf_b, z_te=mu_te_b, z_cov=mu_cov_b)
#
#            l_fus = fusion.readback_loss(z_fus, mu_tnf=mu_tnf_b, mu_te=mu_te_b, mu_cov=mu_cov_b,
#                                          w_tnf=w_tnf_b, w_te=w_te_b, w_cov=w_cov_b)
#            total = cfg.loss_scale_fusion * l_fus
#
#            if use_entropy:
#                total = total + cfg.fusion_gate_entropy_weight * fusion.gate_entropy()
#
#            total.backward()
#            torch.nn.utils.clip_grad_norm_(all_p, 1.0)
#            opt.step()
#
#            ep_tot += total.item(); nb += 1
#
#        if nb == 0:
#            log.warning(f"Epoch {epoch}: 0 batches processed (n={n_ref}, batch_size={effective_batch_size}) -- skipping")
#            continue
#
#        sch.step()
#
#        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.phase2_epochs:
#            w   = fusion.get_weights()
#            log.info(f"  Joint ep {epoch:3d}/{cfg.phase2_epochs} | recon_loss={ep_tot/nb:.4f} "
#                      "| " + " ".join(f"{k}={v:.3f}" if v is not None else f"{k}=None" for k, v in w.items()))
#
#    log.info(f"  Joint done in {time.time()-t0:.0f}s")
#
#    # --- Final latent uses the SAME precomputed frozen mu's (no drift) ------
#    fusion.eval()
#    std_parts = []
#    if tnf_active: std_parts.append(f"TNF={mu_tnf_full.std():.4f}")
#    if te_active:  std_parts.append(f"TE={mu_te_full.std():.4f}")
#    if cov_active: std_parts.append(f"COV={mu_cov_full.std():.4f}")
#    log.info(f"  Post-joint std (frozen, == Phase 1, no drift): {' '.join(std_parts)}")
#
#    with torch.no_grad():
#        final = fusion(
#            z_tnf=mu_tnf_full.to(device) if tnf_active else None,
#            z_te=mu_te_full.to(device) if te_active else None,
#            z_cov=mu_cov_full.to(device) if cov_active else None,
#        ).cpu().numpy()
#
#    log.info(f"  Final latent std={final.std():.4f}")
#
#    contig_weights_path = None
#    if cfg.fusion_mode == "contig_weighted":
#        cw = fusion.get_contig_weights()
#        if cw is not None:
#            contig_weights_path = Path(outdir) / "contig_fusion_weights.npy"
#            np.save(contig_weights_path, cw)
#            log.info(f"  Saved per-contig fusion gate weights: {contig_weights_path} shape={cw.shape} "
#                     f"(columns={fusion._mods})")
#
#    (ckpt_dir / "joint").mkdir(parents=True, exist_ok=True)
#    np.save(lat_path, final)
#    torch.save(fusion.state_dict(), fusion_path)  # FIX 14
#    if ckpt:
#        ckpt.mark_done("joint_fusion", {
#            "_fp": fp2, "final_std": float(final.std()), "fusion_weights": fusion.get_weights(),
#            "fusion_mode": cfg.fusion_mode,
#            "contig_fusion_weights_path": str(contig_weights_path) if contig_weights_path else None,
#        })
#
#    # Unfreeze afterward in case caller reuses these model objects elsewhere
#    if tnf_active:
#        for p in tnf_model.parameters(): p.requires_grad_(True)
#    if te_active:
#        for p in te_model.parameters(): p.requires_grad_(True)
#    if cov_active:
#        for p in cov_model.parameters(): p.requires_grad_(True)
#
#    return final
#
#
## =============================================================================
## MAIN PUBLIC API
## =============================================================================
#
#def run_encoder(
#    outdir:             str,
#    tnf_features_path:  Optional[str] = None,
#    cov_features_path:  Optional[str] = None,
#    te_features_path:   Optional[str] = None,
#    tnf_weights_path:   Optional[str] = None,
#    te_weights_path:    Optional[str] = None,
#    config:             Optional[EncoderConfig] = None,
#    config_path:        Optional[str] = None,
#    n_samples:          Optional[int] = None,
#    **kwargs,
#) -> str:
#    """
#    Full HyphaeS encoder pipeline v8.
#    Returns path to final_latent.npy
#
#    n_samples (FIX 7, now actually used): when supplied, cross-checked
#    against cov_features_path's raw column count (must equal
#    n_samples + COV_METADATA_DIM) before the coverage array is split.
#    """
#    t_start = time.time()
#    if kwargs:
#        log.warning(f"run_encoder: ignoring unknown kwargs: {list(kwargs.keys())}")
#
#    if config_path is not None:
#        cfg = EncoderConfig.from_yaml(config_path)
#    elif config is not None:
#        cfg = config
#    else:
#        cfg = EncoderConfig()
#
#    # --- FIX 8: fail loud on a requested-but-missing modality unless the
#    # caller explicitly opted into the old silent-disable behavior. -------
#    cfg.tnf_include = _check_modality_available(cfg.tnf_include, tnf_features_path, "tnf", cfg.allow_missing_modalities)
#    cfg.te_include  = _check_modality_available(cfg.te_include, te_features_path, "te", cfg.allow_missing_modalities)
#    cfg.cov_include = _check_modality_available(cfg.cov_include, cov_features_path, "cov", cfg.allow_missing_modalities)
#
#    if not (cfg.tnf_include or cfg.te_include or cfg.cov_include):
#        raise ValueError(
#            "run_encoder: all three modalities (tnf/te/cov) ended up disabled (missing/invalid "
#            "feature paths) -- at least one usable modality is required."
#        )
#
#    log.info(f"Active modalities: tnf={cfg.tnf_include} te={cfg.te_include} cov={cfg.cov_include}")
#
#    torch.manual_seed(cfg.seed)
#    np.random.seed(cfg.seed)
#
#    outdir   = Path(outdir)
#    outdir.mkdir(parents=True, exist_ok=True)
#    ckpt_dir = outdir / "checkpoints"
#    ckpt_dir.mkdir(parents=True, exist_ok=True)
#    device   = resolve_device(cfg.device)  # FIX 10
#
#    if cfg.parallel_phase1 and device.type in ("cuda", "mps"):
#        log.warning(f"parallel_phase1=True with device={device.type}: each Phase-1 worker "
#                     f"process will initialize its own {device.type} context -- this can work, "
#                     f"but may contend for the same GPU/accelerator with no speedup. Consider "
#                     f"parallel_phase1=false for {device.type} runs if training looks slower "
#                     f"than sequential.")
#
#    # --- FIX 6: strict, manifest-based contig alignment (replaces v7's
#    # row-count-only check). Each modality's OWN contig-ID file/manifest
#    # convention is read directly -- never assumed uniform. -----------------
#    log.info("Verifying contig alignment across active modalities...")
#    ids_by_mod = {}
#    if cfg.tnf_include:
#        ids_by_mod["tnf"] = _load_modality_manifest(tnf_features_path, "tnf")
#    if cfg.te_include:
#        ids_by_mod["te"] = _load_modality_manifest(te_features_path, "te")
#    if cfg.cov_include:
#        ids_by_mod["cov"] = _load_modality_manifest(cov_features_path, "cov")
#
#    contig_ids, contig_order_hash = _check_contig_alignment(ids_by_mod)
#    n_contigs = len(contig_ids)
#    log.info(f"Contig alignment OK: {n_contigs:,} contigs, order_hash={contig_order_hash}, "
#             f"modalities checked={list(ids_by_mod.keys())}")
#
#    log.info("Loading features...")
#
#    tnf_feat = None
#    if cfg.tnf_include:
#        tnf_feat = validate_features(np.load(tnf_features_path), "TNF", expected_dim=TNF_EXPECTED_DIM)  # FIX 7
#        if tnf_feat.shape[0] != n_contigs:
#            raise ValueError(f"TNF: {tnf_feat.shape[0]} rows in {tnf_features_path} but "
#                              f"{n_contigs} contigs in its own contig_ids manifest -- the "
#                              f".npy file appears stale relative to its manifest.")
#
#    te_feat = None
#    if cfg.te_include:
#        te_feat = validate_features(np.load(te_features_path), "TE", expected_dim=TE_EXPECTED_DIM)  # FIX 7
#        if te_feat.shape[0] != n_contigs:
#            raise ValueError(f"TE: {te_feat.shape[0]} rows in {te_features_path} but "
#                              f"{n_contigs} contigs in its own contig_ids manifest -- the "
#                              f".npy file appears stale relative to its manifest.")
#
#    cov_feat = None
#    cov_w    = None
#    if cfg.cov_include:
#        cov_raw = np.load(cov_features_path)
#        if cov_raw.shape[0] != n_contigs:
#            raise ValueError(f"COV: {cov_raw.shape[0]} rows in {cov_features_path} but "
#                              f"{n_contigs} contigs in its own contig_ids manifest -- the "
#                              f".npy file appears stale relative to its manifest.")
#        if n_samples is not None:
#            expected_cov_width = n_samples + COV_METADATA_DIM
#            if cov_raw.shape[1] != expected_cov_width:
#                raise ValueError(
#                    f"COV: n_samples={n_samples} implies {expected_cov_width} columns "
#                    f"({COV_METADATA_DIM} metadata + {n_samples} coverage columns), but "
#                    f"{cov_features_path} has {cov_raw.shape[1]} columns."
#                )
#        cov_feat_split, cov_w = split_cov_features(cov_raw, cfg)  # FIX 5
#        cov_feat = validate_features(cov_feat_split, "COV")
#        _guard_nonzero_weights(cov_w, "COV")  # FIX 12b
#
#    cfg.input_dim_tnf = tnf_feat.shape[1] if cfg.tnf_include else None
#    cfg.input_dim_cov = cov_feat.shape[1] if cfg.cov_include else None
#    cfg.input_dim_te  = te_feat.shape[1]  if cfg.te_include  else None
#    cfg               = cfg.resolve(n_contigs)
#
#    modality_str = cfg.modality_str()
#
#    log.info("")
#    log.info("+" + "="*66 + "+")
#    log.info("|" + " "*12 + f"HYPHAES ENCODER v8.1 -- {modality_str}" + " "*max(0, 34-len(modality_str)) + "|")
#    log.info("+" + "="*66 + "+")
#    log.info(f"  n_contigs={n_contigs:,}  batch_size={cfg.batch_size} (auto)")
#    log.info(f"  modalities: {modality_str}")
#    cfg.log_summary()
#    log.info("")
#
#    tnf_w = None
#    if cfg.tnf_include:
#        tnf_w = (validate_weights(np.load(tnf_weights_path).squeeze(), "TNF_w")
#                 if tnf_weights_path and Path(tnf_weights_path).exists() else None)
#        if tnf_w is not None and len(np.atleast_1d(tnf_w)) != n_contigs:
#            # FIX 17: a stale or truncated weight file must fail loudly here,
#            # not later via a silent length-mismatch inside make_loader/
#            # TensorDataset (which would either crash confusingly deep in
#            # torch, or -- if lengths happened to coincide by accident --
#            # silently misalign weight i with the wrong contig i).
#            raise ValueError(f"TNF: weights file {tnf_weights_path} has "
#                              f"{len(np.atleast_1d(tnf_w))} entries but there are {n_contigs} "
#                              f"contigs -- refusing to use a weight array that doesn't match "
#                              f"row-for-row.")
#        _guard_nonzero_weights(tnf_w, "TNF")  # FIX 12b
#    te_w = None
#    if cfg.te_include:
#        te_w = (validate_weights(np.load(te_weights_path).squeeze(), "TE_w")
#                if te_weights_path and Path(te_weights_path).exists() else None)
#        if te_w is not None and len(np.atleast_1d(te_w)) != n_contigs:
#            raise ValueError(f"TE: weights file {te_weights_path} has "
#                              f"{len(np.atleast_1d(te_w))} entries but there are {n_contigs} "
#                              f"contigs -- refusing to use a weight array that doesn't match "
#                              f"row-for-row.")
#        _guard_nonzero_weights(te_w, "TE")  # FIX 12b
#
#    log.info("Normalizing features...")
#    tnf_norm = None
#    if cfg.tnf_include:
#        tnf_norm, tnf_mean, tnf_std = normalize_features(tnf_feat, "TNF")
#        np.save(outdir / "tnf_norm_stats.npy", np.stack([tnf_mean, tnf_std]))
#
#    cov_norm = None
#    if cfg.cov_include:
#        # FIX 16: statistics computed only over valid_mask>0 rows -- see
#        # normalize_features docstring. cov_w IS valid_mask (split_cov_features
#        # returns it unchanged as the reliability weight).
#        cov_norm, cov_mean, cov_std = normalize_features(cov_feat, "COV", mask=cov_w)
#        np.save(outdir / "cov_norm_stats.npy", np.stack([cov_mean, cov_std]))
#
#    te_norm = None
#    if cfg.te_include:
#        te_norm, te_mean, te_std = normalize_features(te_feat, "TE")
#        np.save(outdir / "te_norm_stats.npy", np.stack([te_mean, te_std]))
#
#    # --- FIX 15: Phase-1 checkpoint fingerprints -----------------------------
#    fp_tnf = fp_te = fp_cov = None
#    if cfg.tnf_include:
#        fp_tnf = _fingerprint(_file_fingerprint(tnf_features_path), _file_fingerprint(tnf_weights_path),
#                               cfg.latent_dim_tnf, cfg.beta_tnf, cfg.phase1_epochs_tnf, cfg.phase1_lr,
#                               cfg.kl_free_bits, cfg.kl_anneal_epochs, cfg.hidden_scale, cfg.hidden_min,
#                               cfg.hidden_max, cfg.dropout, cfg.n_hidden_layers, cfg.batch_size,
#                               cfg.seed, str(device), VERSION)
#    if cfg.te_include:
#        fp_te = _fingerprint(_file_fingerprint(te_features_path), _file_fingerprint(te_weights_path),
#                              cfg.latent_dim_te, cfg.beta_te, cfg.phase1_epochs_te, cfg.phase1_lr,
#                              cfg.kl_free_bits, cfg.kl_anneal_epochs, cfg.hidden_scale, cfg.hidden_min,
#                              cfg.hidden_max, cfg.dropout, cfg.n_hidden_layers, cfg.batch_size,
#                              cfg.seed, str(device), VERSION)
#    if cfg.cov_include:
#        fp_cov = _fingerprint(_file_fingerprint(cov_features_path), cfg.cov_use_n_samples_present_as_feature,
#                               cfg.latent_dim_cov, cfg.beta_cov, cfg.phase1_epochs_cov, cfg.phase1_lr,
#                               cfg.kl_free_bits, cfg.kl_anneal_epochs, cfg.hidden_scale, cfg.hidden_min,
#                               cfg.hidden_max, cfg.dropout, cfg.n_hidden_layers, cfg.batch_size,
#                               cfg.seed, str(device), VERSION)
#
#    log.info("")
#    log.info("PHASE 1: Independent encoder training")
#    log.info("-" * 40)
#    t1 = time.time()
#
#    tnf_model  = None
#    latent_tnf = None
#    cov_model  = None
#    latent_cov = None
#    te_model   = None
#    latent_te  = None
#
#    # (name, features, weights, latent_dim, beta, epochs, ckpt_key, fp)
#    active_jobs = []
#    if cfg.tnf_include:
#        active_jobs.append(("tnf", tnf_norm, tnf_w, cfg.latent_dim_tnf, cfg.beta_tnf,
#                             cfg.phase1_epochs_tnf, "tnf_encoder", fp_tnf))
#    if cfg.cov_include:
#        active_jobs.append(("cov", cov_norm, cov_w, cfg.latent_dim_cov, cfg.beta_cov,
#                             cfg.phase1_epochs_cov, "cov_encoder", fp_cov))
#    if cfg.te_include:
#        active_jobs.append(("te", te_norm, te_w, cfg.latent_dim_te, cfg.beta_te,
#                             cfg.phase1_epochs_te, "te_encoder", fp_te))
#
#    results = {}
#    want_parallel = cfg.parallel_phase1 and len(active_jobs) > 1
#
#    if want_parallel:
#        try:
#            n_workers = cfg.parallel_phase1_max_workers or len(active_jobs)
#            n_workers = max(1, min(n_workers, len(active_jobs), os.cpu_count() or 1))  # FIX 11
#            total_threads = max(1, os.cpu_count() or 32)
#            threads_per_worker = max(4, total_threads // n_workers)
#            log.info(f"Phase 1: training {len(active_jobs)} modalities in parallel "
#                     f"({n_workers} workers x {threads_per_worker} threads)")
#
#            jobs = [
#                (name, feat, w, ldim, beta, ep, cfg, device, ckpt_dir, ckpt_key, fp, threads_per_worker)
#                for (name, feat, w, ldim, beta, ep, ckpt_key, fp) in active_jobs
#            ]
#            # FIX 11: "spawn", not "fork" -- fork doesn't exist on Windows and
#            # is unsafe once a CUDA context may already be initialized in the
#            # parent process.
#            ctx = mp.get_context("spawn")
#            _parent_threads = torch.get_num_threads()
#            torch.set_num_threads(1)
#            try:
#                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
#                    futs = {ex.submit(_phase1_worker, job): job[0] for job in jobs}
#                    for fut in as_completed(futs):
#                        name = futs[fut]
#                        results[name] = fut.result()[1:]  # (model, latent)
#            finally:
#                torch.set_num_threads(_parent_threads)
#        except Exception as e:
#            log.warning(f"Parallel Phase 1 failed ({e}) -- falling back to sequential training. "
#                        "Already-completed modalities are resumed from checkpoint automatically.")
#            results = {}
#
#    for (name, feat, w, ldim, beta, ep, ckpt_key, fp) in active_jobs:
#        if name in results:
#            continue
#        results[name] = _phase1_single(name, feat, w, ldim, beta, ep, cfg, device, ckpt_dir, ckpt_key, fp)
#
#    if "tnf" in results:
#        tnf_model, latent_tnf = results["tnf"]
#    if "cov" in results:
#        cov_model, latent_cov = results["cov"]
#    if "te" in results:
#        te_model, latent_te = results["te"]
#
#    std_parts = []
#    if cfg.tnf_include:
#        std_parts.append(f"TNF:{latent_tnf.std():.4f}")
#    if cfg.te_include:
#        std_parts.append(f"TE:{latent_te.std():.4f}")
#    if cfg.cov_include:
#        std_parts.append(f"COV:{latent_cov.std():.4f}")
#    log.info(f"Phase 1 complete in {time.time()-t1:.0f}s  |  Latent std -- {' '.join(std_parts)}")
#
#    if cfg.tnf_include:
#        np.save(outdir / "latent_tnf.npy", latent_tnf)
#    if cfg.te_include:
#        np.save(outdir / "latent_te.npy", latent_te)
#    if cfg.cov_include:
#        np.save(outdir / "latent_cov.npy", latent_cov)
#
#    log.info("")
#    log.info(f"PHASE 2: Joint fusion training (v8 -- frozen encoders, fusion_mode='{cfg.fusion_mode}', "
#             f"per-contig weighted reconstruction-through-fusion loss)")
#    log.info("-" * 40)
#    t2 = time.time()
#
#    # --- FIX 15: Phase-2 checkpoint fingerprint (depends on Phase-1's too) ---
#    fp2 = _fingerprint(
#        fp_tnf, fp_te, fp_cov, cfg.fusion_mode, cfg.fusion_gate_entropy_weight,
#        cfg.loss_scale_fusion, cfg.phase2_epochs, cfg.phase2_lr, cfg.final_dim,
#        cfg.latent_dim_tnf, cfg.latent_dim_te, cfg.latent_dim_cov,
#        str(device), cfg.seed, VERSION,
#    )
#
#    fusion    = FusionModule(cfg).to(device)
#    final_lat = _phase2_joint(
#        fusion, cfg, device, ckpt_dir, outdir, fp2,
#        tnf_model=tnf_model, tnf_feat=tnf_norm, tnf_w=tnf_w,
#        te_model=te_model, te_feat=te_norm, te_w=te_w,
#        cov_model=cov_model, cov_feat=cov_norm, cov_w=cov_w)
#
#    log.info(f"Phase 2 complete in {time.time()-t2:.0f}s")
#
#    final_path = outdir / "final_latent.npy"
#    np.save(final_path, final_lat)
#
#    save_dict = {
#        "fusion":         fusion.state_dict(),
#        "fusion_weights": fusion.get_weights(),
#        "config":         asdict(cfg),
#    }
#    if cfg.tnf_include and tnf_model is not None:
#        save_dict["tnf_encoder"] = tnf_model.state_dict()
#    if cfg.te_include and te_model is not None:
#        save_dict["te_encoder"] = te_model.state_dict()
#    if cfg.cov_include and cov_model is not None:
#        save_dict["cov_encoder"] = cov_model.state_dict()
#    torch.save(save_dict, outdir / "encoder_weights.pt")
#
#    with open(outdir / "encoder_config.json", "w") as f:
#        json.dump(asdict(cfg), f, indent=2)
#
#    # --- Consolidated manifest (new in v8) -----------------------------------
#    manifest = {
#        "module": "encoder.py", "version": VERSION,
#        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
#        "runtime_seconds": round(time.time() - t_start, 1),
#        "modalities": {"tnf": cfg.tnf_include, "te": cfg.te_include, "cov": cfg.cov_include},
#        "n_contigs": n_contigs,
#        "contig_id_order_hash": contig_order_hash,
#        "fusion_mode": cfg.fusion_mode,
#        "device": str(device),
#        "inputs": {
#            "tnf_features_path": str(tnf_features_path) if cfg.tnf_include else None,
#            "tnf_features_fingerprint": _file_fingerprint(tnf_features_path) if cfg.tnf_include else None,
#            "te_features_path": str(te_features_path) if cfg.te_include else None,
#            "te_features_fingerprint": _file_fingerprint(te_features_path) if cfg.te_include else None,
#            "cov_features_path": str(cov_features_path) if cfg.cov_include else None,
#            "cov_features_fingerprint": _file_fingerprint(cov_features_path) if cfg.cov_include else None,
#            "tnf_weights_path": str(tnf_weights_path) if tnf_weights_path else None,
#            "tnf_weights_fingerprint": _file_fingerprint(tnf_weights_path) if tnf_weights_path else None,  # FIX 18
#            "te_weights_path": str(te_weights_path) if te_weights_path else None,
#            "te_weights_fingerprint": _file_fingerprint(te_weights_path) if te_weights_path else None,  # FIX 18
#            "n_samples": n_samples,
#        },
#        "outputs": {
#            "final_latent_path": str(final_path), "shape": list(final_lat.shape),
#            "fusion_weights": fusion.get_weights(),
#            "contig_fusion_weights_path": (str(outdir / "contig_fusion_weights.npy")
#                                            if cfg.fusion_mode == "contig_weighted" else None),
#            "encoder_weights_path": str(outdir / "encoder_weights.pt"),
#        },
#        "config": asdict(cfg),
#    }
#    with open(outdir / "encoder_manifest.json", "w") as f:
#        json.dump(manifest, f, indent=2, default=str)
#
#    log.info("")
#    log.info("=" * 68)
#    log.info("  ENCODER v8.1 COMPLETE")
#    log.info(f"  n_contigs    : {n_contigs:,}")
#    log.info(f"  modalities   : {modality_str}")
#    if cfg.tnf_include:
#        log.info(f"  latent_tnf   : {latent_tnf.shape}  std={latent_tnf.std():.4f}")
#    if cfg.te_include:
#        log.info(f"  latent_te    : {latent_te.shape}   std={latent_te.std():.4f}")
#    if cfg.cov_include:
#        log.info(f"  latent_cov   : {latent_cov.shape}  std={latent_cov.std():.4f}")
#    log.info(f"  final_latent : {final_lat.shape}  std={final_lat.std():.4f}")
#    log.info(f"  fusion_mode  : {cfg.fusion_mode}")
#    log.info(f"  Fusion weights: {fusion.get_weights()}")
#    log.info(f"  Output: {outdir}")
#    log.info("=" * 68)
#
#    return str(final_path)




























"""
HyphaeS Encoder Module — v8.2
=========================================
File: hyphaesbin/encoder/encoder.py

=============================================================================
CHANGELOG FROM v7 — ALIGNMENT / CONFIG / DEVICE / FUSION-MODE FIXES
=============================================================================
v7 fixed the Phase-2 root-cause bugs (frozen sub-encoders, reconstruction-
through-fusion gate loss). This pass (v8) closes a second, independent set
of issues found in a follow-up review of the surrounding plumbing: how
features are loaded, validated, aligned across modalities, checkpointed,
and how the fusion gate itself works. NONE of these change the "core"
mechanics that were already validated as healthy: UniformVAE's
architecture, vae_loss, the Phase-1 per-modality training loop
(_train_vae_epoch / _train_phase1), and FIX 1 / FIX 2 from v7 (frozen
sub-encoders during Phase 2, reconstruction-through-fusion as the training
signal) are all untouched in their actual math.

FIX 5 — COVERAGE FEATURE ARRAY WAS NEVER SPLIT (was: fed to the VAE whole)
  coverage.py's coverage_features.npy is N+4D:
      [valid_mask | n_samples_present | mean_dist | std_dist | cov_1..cov_N]
  v7 loaded this whole array and fed it straight into the COV VAE as if
  every column were a feature. valid_mask is documented by coverage.py's
  own COLUMN_ROLES as "MASK ONLY — never a feature", so the VAE was being
  trained to reconstruct a mask column, and COV never used any per-contig
  reliability weight at all (v7 hardcoded weights=None for COV in Phase 1,
  and Phase 2 never touched COV weights either).
  FIX: split_cov_features() removes valid_mask/n_samples_present from the
  feature vector (n_samples_present can be opted back in via
  cfg.cov_use_n_samples_present_as_feature, off by default, matching
  coverage.py's own "consumer's discretion" framing) and turns valid_mask
  into COV's per-contig reliability weight, used in both Phase 1 (COV's
  own vae_loss masking) and Phase 2 (see FIX 9).

FIX 6 — CONTIG ALIGNMENT WAS ROW-COUNT ONLY (was: shape[0] equality)
  v7's only alignment check between TNF/TE/COV was "do the arrays have the
  same number of rows". Two modalities could have the same contig COUNT
  but a different ORDER (e.g. one pipeline run resorted or re-filtered
  contigs) and this would silently fuse row i of TNF with row i of TE even
  though they describe different contigs — a silent correctness disaster
  no shape check can catch.
  FIX: _load_modality_manifest() reads each modality's OWN contig-ID
  manifest (the three upstream modules use three different filename/key
  conventions — verified by reading their actual save code, not assumed
  from any docstring) and compares the full ordered contig-ID list (plus
  its content hash) across all active modalities. Any mismatch is a hard
  error before any training happens.

FIX 7 — MISSING DIMENSION VALIDATION (was: expected_dim never passed)
  validate_features() already supported an expected_dim check but no call
  site ever used it. TNF features are always 136D and TE features are
  always 5D by construction (tnf_gene.py / te_composition.py); a silent
  dimension drift (e.g. an old cached tnf_features.npy from a differently
  configured run) would previously pass straight through.
  FIX: TNF_EXPECTED_DIM=136 / TE_EXPECTED_DIM=5 are now enforced. COV's
  width is sample-count-dependent, so instead its width is cross-checked
  against the `n_samples` argument when the caller supplies one (n_samples
  was accepted but never used anywhere in v7 — dead parameter).

FIX 8 — SILENT MISSING-MODALITY FALLBACK (was: warn + auto-disable)
  If a modality was requested (cfg.X_include=True) but its feature path
  was missing/invalid, v7 logged one WARNING line and silently trained
  without it. A missing file is usually a real upstream failure (a crashed
  pipeline step, a typo'd path), and training on 2 of 3 intended
  modalities without the caller ever finding out is exactly the kind of
  silent-degradation this project has been fixing everywhere else.
  FIX: this is now a hard error (FileNotFoundError) unless the caller sets
  the new `allow_missing_modalities: bool` config flag, in which case the
  old warn-and-disable behavior is kept, but now opt-in and explicit.

FIX 9 — PHASE 2 READBACK LOSS HAD NO PER-CONTIG WEIGHTING (was: whole-batch
  unweighted mean, TNF/TE per-contig confidence weights unused in Phase 2)
  v7's readback_loss() computed one MSE over the WHOLE batch per modality
  and averaged the three (or fewer) modality losses uniformly — every
  contig counted equally regardless of whether, e.g., its COV row was a
  valid_mask=0 placeholder or its TE row had zero confidence weight
  (fragmented contigs get near-zero TE reliability by construction; see
  te_composition.py). FIX: readback_loss() now takes each modality's own
  per-contig reliability weight (COV's valid_mask from FIX 5, and TNF/TE's
  existing confidence weight arrays, defaulting to 1.0 when not supplied)
  and combines them PER CONTIG:
      contig_loss_i = (w_tnf_i*err_tnf_i + w_te_i*err_te_i + w_cov_i*err_cov_i)
                      / (w_tnf_i + w_te_i + w_cov_i)
  Contigs whose total weight is 0 across every active modality are
  excluded from the batch mean entirely (same masked-average pattern
  vae_loss already uses). This is a WEIGHTED-AVERAGE policy, not an
  intersection/all-valid policy — requiring every modality to be valid for
  a contig before it counts at all would throw away nearly all contigs,
  since TE's reliability weight is near-all-zero for fragmented real
  assemblies by design.
  NOTE — two different "weights" that must not be conflated: the fusion
  GATE weight (how much of each modality's latent goes into z_fus, learned
  by FusionModule) is a completely different concept from the per-contig
  RELIABILITY weight used here (how much a given contig's modality term
  should count toward the training loss). v8.2's FIX 23 (below) connects
  these two concepts for the FIRST time — the gate can now see the
  reliability weights as an input signal — but they remain two distinct
  quantities: reliability weights the LOSS; gate weights the FUSION.

FIX 10 — device WAS HARDCODED AUTO-DETECT (was: cuda->mps->cpu, no config)
  get_device() always auto-detected with no way to force CPU (e.g. to keep
  a machine's GPU free for another job) or to require GPU and fail loudly
  if unavailable. FIX: new `device: "cpu"|"gpu"|"auto"` config field +
  resolve_device(). "cpu" always uses CPU. "gpu" uses CUDA if available,
  else falls back to MPS with a warning, else CPU with a warning (a
  missing GPU is treated as a soft degradation here, unlike the FIX 8
  missing-file case, since it's an environment/hardware fact rather than a
  silently-wrong-data risk). "auto" preserves the old cuda->mps->cpu
  behavor. get_device() is kept as a thin backward-compatible alias for
  resolve_device("auto").

FIX 11 — fork MULTIPROCESSING CONTEXT (was: mp.get_context("fork"))
  "fork" doesn't exist on Windows at all, and even on Linux it's unsafe
  once a CUDA context may already be initialized in the parent process
  ("cannot re-initialize CUDA in forked subprocess" is a well-known
  failure mode). FIX: switched unconditionally to mp.get_context("spawn").
  Also: parallel_phase1_max_workers is now clamped to os.cpu_count() (was
  previously unclamped against the actual machine), and a warning is
  logged if parallel Phase 1 is requested while device is cuda/mps (each
  worker will initialize its own device context — this can work, but may
  contend for the same GPU with no speedup; not auto-disabled, since that
  is a legitimate configuration choice left to the caller).

FIX 12 — NO SMALL-DATASET / ZERO-WEIGHT GUARDS (was: silent under-batching
  or silent division against nothing)
  Two new guards, both fail-loud rather than fail-silent:
    (a) _effective_batch_size() clamps the requested batch_size down to
        the dataset size instead of relying on DataLoader to silently
        produce one small leftover batch every epoch.
    (b) a modality whose ENTIRE per-contig weight array is 0 (every
        contig masked out) now raises immediately instead of quietly
        training on an empty effective dataset.

FIX 13 (NEW FEATURE, not a bug fix) — fusion_mode: "contig_weighted"
  v7 had a single, global softmax gate (one learned weight per modality,
  shared by every contig). This adds a second, fully-implemented and
  config-selectable mode: a GMU-style (Arevalo et al. 2017) per-contig
  gate — a small MLP over each contig's own concatenated projected
  latents producing a per-contig softmax over modalities, trained through
  the SAME reconstruction-through-fusion loss as the global gate (FIX 9),
  plus a small entropy-regularization term (`fusion_gate_entropy_weight`,
  only active in this mode) that discourages the per-contig gate from
  collapsing onto one modality for every contig (a known failure mode for
  unsupervised instance-wise gates — loosely inspired by the sparsity/
  collapse safeguards in "Adaptive Confidence-weighted Expansion"-style
  approaches, not a literal reproduction of any one paper).
  `fusion_mode` defaults to "global" (the validated, production default).
  An invalid fusion_mode value is a hard config error (EncoderConfig.resolve
  rejects anything other than "global"/"contig_weighted") — there is no
  silent fallback to global.
  In contig_weighted mode, the per-contig gate weights for the FULL
  dataset are saved to `contig_fusion_weights.npy` in outdir for
  downstream analysis (columns ordered per FusionModule._mods).

FIX 14 (BONUS — found while implementing FIX 13, not on the original list)
  v7's _phase2_joint() never persisted the trained FusionModule's weights
  as part of its own checkpoint — only run_encoder's *final* save at the
  very end wrote fusion.state_dict() into encoder_weights.pt. If Phase 2
  hit its checkpoint (cache) on a re-run, the `fusion` object passed in by
  the caller was still the FRESHLY (randomly) initialized one — so
  encoder_weights.pt would get overwritten with UNTRAINED fusion weights
  even though the returned final_latent.npy was correctly the cached,
  actually-trained result. Latents and saved weights would silently
  disagree. FIX: _phase2_joint now saves fusion.state_dict() into its own
  checkpoint dir and reloads it into the passed-in module on a cache hit,
  before returning.

FIX 15 — CHECKPOINT FINGERPRINTING (was: is_done() + file-exists only)
  Neither Phase 1 nor Phase 2 checkpoints included any fingerprint of
  their actual inputs/config — a changed feature file, a changed
  hyperparameter, or a changed device/fusion_mode could all silently reuse
  a stale checkpoint. FIX: added the same _file_fingerprint / _fingerprint
  / _checkpoint_ok pattern already used in preprocessing.py / coverage.py
  / tnf_gene.py / te_composition.py. Phase-1 fingerprints cover that
  modality's own feature/weight file fingerprints plus every
  training-relevant hyperparameter, device, and module VERSION. Phase-2's
  fingerprint additionally covers fusion_mode, fusion_gate_entropy_weight,
  and the upstream Phase-1 fingerprints (so any Phase-1 change invalidates
  Phase 2 too).

=============================================================================
CHANGELOG FROM v8 (initial) — v8.1, FOLLOW-UP REVIEW: 4 MORE PLUMBING BUGS
=============================================================================
A second read-through of the v8 plumbing (not the core VAE/fusion math)
found four more real issues, all now fixed:

FIX 16 — COV NORMALIZATION STATISTICS CONTAMINATED BY PLACEHOLDER ROWS
  FIX 5 (v8 initial) correctly turned valid_mask into COV's reliability
  weight for the LOSS, but normalize_features(cov_feat, "COV") still
  computed mean/std over EVERY row, including valid_mask=0 placeholder
  rows -- which are literal zero-filled rows inserted by coverage.py's
  step7_realign, not real (if unreliable) data. Those zeros pulled the
  mean/std of every real column toward zero, meaning even the loss-masked
  "valid" rows were being normalized against contaminated statistics.
  FIX: normalize_features() now accepts an optional `mask`; when given
  (COV's call site passes cov_w, i.e. valid_mask), mean/std are computed
  ONLY over mask>0 rows. Every row (including placeholders) is still
  normalized and still gets its output value -- only the STATISTICS
  computation excludes invalid rows.

FIX 17 — TNF/TE WEIGHT-ARRAY LENGTH NEVER CHECKED AGAINST n_contigs
  A stale or truncated tnf_weights.npy / te_weights.npy (e.g. left over
  from a run with fewer contigs) was never checked for length before being
  used. Depending on how it happened to disagree with n_contigs, this
  could fail confusingly deep inside make_loader/TensorDataset, or --
  worse -- silently misalign weight i with the wrong contig i if the
  lengths happened to differ in a way numpy didn't immediately reject.
  FIX: run_encoder now checks len(tnf_w) == n_contigs and
  len(te_w) == n_contigs immediately after loading, before either array is
  used anywhere, and raises a clear ValueError naming the mismatch.

FIX 18 — WEIGHT FILES WERE FINGERPRINTED FOR CHECKPOINTING BUT NOT RECORDED
  IN THE MANIFEST
  fp_tnf/fp_te (the Phase-1 checkpoint fingerprints) already included
  _file_fingerprint(tnf_weights_path) / _file_fingerprint(te_weights_path),
  but encoder_manifest.json's "inputs" section only recorded the feature
  file fingerprints, not the weight file fingerprints -- so the on-disk
  record of what a run actually used was incomplete relative to what its
  own checkpoint validity depended on. FIX: tnf_weights_fingerprint /
  te_weights_fingerprint added alongside the existing *_features_fingerprint
  entries.

FIX 19 — EMPTY FEATURE ARRAY PRODUCED AN OPAQUE numpy ERROR
  validate_features() called .min()/.max()/.mean() before checking whether
  the array had any rows at all -- a 0-row feature file (e.g. every contig
  filtered out upstream) would fail with numpy's generic "zero-size array
  to reduction operation" error, far from anything naming the real cause.
  FIX: an explicit shape[0]==0 check up front raises a clear, actionable
  ValueError instead.

KNOWN LIMITATION, STILL UNRESOLVED IN v8.1 / v8.2
  PyTorch remains uninstallable in the sandbox this file is developed in
  (see the v8 changelog above and the module-level note below) -- FIX 16
  through FIX 23 were verified with the same non-torch unit-test approach
  used for the original v8 fixes (pure-Python/numpy logic, run against a
  minimal stand-in for `torch` sufficient for import). The actual VAE/
  FusionModule training path (including whether FIX 16's masked
  normalization or FIX 23's gate-input change interacts correctly with
  real gradients) is still NOT verified by a real PyTorch run.

WHAT WAS *NOT* CHANGED
  UniformVAE, vae_loss, _train_vae_epoch, _train_phase1's training loop
  body, and v7's FIX 1 / FIX 2 mechanics (frozen sub-encoders during Phase
  2, reconstruction-through-fusion as the only signal training the fusion
  module) are unchanged. This file only changes plumbing around those
  mechanics: what gets fed in, how it's validated/aligned/weighted, how
  checkpoints are fingerprinted, and the per-contig gate (now including
  its FIX 23 input).

KNOWN LIMITATION, DISCLOSED — NOT SILENTLY GLOSSED OVER
  PyTorch could not be installed in the sandbox this file was written and
  tested in (network-restricted environment; both the default index and
  the CPU-only wheel index were unreachable). Every piece of logic in this
  file that does NOT require torch (EncoderConfig validation/resolve,
  _load_modality_manifest, split_cov_features, the checkpoint-fingerprint
  helpers, _check_modality_available, _effective_batch_size,
  resolve_device's branching logic) was exercised with real unit tests
  using a minimal stand-in for the `torch` module. The actual VAE/
  FusionModule forward/backward training path (UniformVAE, vae_loss,
  FusionModule.forward/readback_loss/gate_entropy, _phase1_single,
  _phase2_joint) could NOT be executed end-to-end in this sandbox and is
  therefore NOT verified by real gradient-level testing here — only by
  careful reading/tracing of the tensor shapes and control flow. Please
  run this file's real test suite (or at least one real end-to-end call to
  run_encoder on a small fixture) in an environment with PyTorch installed
  before relying on it in production.

=============================================================================
CHANGELOG FROM v8.1 -> v8.2 (this pass)
=============================================================================

FIX 23 (NEW) — contig_weighted GATE COULD NOT SEE PER-CONTIG RELIABILITY
  WEIGHTS, ONLY THE LATENT VECTORS THEMSELVES
  FusionModule.forward()'s contig_weighted gate (gate_net) previously took
  only the concatenated projected latents (z_tnf/z_te/z_cov) as input. The
  per-contig reliability weights (w_tnf/w_te/w_cov — TE's confidence
  weight, COV's valid_mask) were already computed upstream and already
  used to weight the READBACK LOSS (FIX 9), but the gate deciding HOW MUCH
  of each modality to fuse never received them directly — it had to infer
  a contig's reliability purely from what that modality's latent vector
  looked like. Since a VAE tends to produce smooth, populated latent
  space even for low-confidence/noisy inputs, this inferred signal is
  weaker than the explicit reliability weight that already exists.
  FIX: gate_net's input dimension is now `in_dim + n_mods` — the
  per-contig reliability weights (defaulting to 1.0 for a modality with no
  supplied weight array, same convention as readback_loss) are
  concatenated onto the projected-latent input before the gate computes
  its per-modality softmax. FusionModule.forward() now accepts optional
  w_tnf/w_te/w_cov kwargs (only used in contig_weighted mode; ignored in
  global mode, where the gate has no per-contig input at all). Both call
  sites in _phase2_joint (the per-batch training loop and the final
  full-dataset forward pass) now pass these weights through. This is
  purely additive to the gate's input signal — it does not change what
  the gate is trained on (still the same reconstruction-through-fusion
  loss, FIX 9) or the "global" mode's mechanics at all.
  DISCLOSED LIMITATION: like FIX 13 originally, this has been verified via
  the non-torch plumbing tests (input-dimension wiring, kwarg threading)
  but the actual effect on trained gate behavior/gradients has not been
  verified with a real PyTorch run in this sandbox — see the KNOWN
  LIMITATION note above.

config.yaml parameters (new keys marked NEW; everything else unchanged
from v7):
  te_include: true
  cov_include: true
  latent_dim_tnf: 64
  latent_dim_te: 5
  latent_dim_cov: null
  final_dim: null
  batch_size: null
  beta_tnf: 0.01
  beta_te: 0.1
  beta_cov: 0.1
  kl_anneal_epochs: 50
  kl_free_bits: 0.01
  phase1_epochs_tnf: 75
  phase1_epochs_te: 50
  phase1_epochs_cov: 50
  phase2_epochs: 50
  phase1_lr: 0.001
  phase2_lr: 0.00005
  loss_scale_te: 0.1
  loss_scale_cov: 0.1
  loss_scale_fusion: 0.05
  dropout: 0.1
  n_hidden_layers: 2
  hidden_scale: 4
  seed: 42
  fusion_mode: "global"                       # NEW — "global" | "contig_weighted"
  fusion_gate_entropy_weight: 0.01             # NEW — only used if fusion_mode="contig_weighted"
  device: "auto"                               # NEW — "cpu" | "gpu" | "auto"
  allow_missing_modalities: false              # NEW
  cov_use_n_samples_present_as_feature: false  # NEW
"""

import hashlib
import json
import logging
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError("PyTorch not installed: conda install -c pytorch pytorch")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from hyphaesbin.utils.checkpoint import Checkpoint
except Exception:
    Checkpoint = None

log = logging.getLogger("hyphaesbin.encoder")

# v8.3: decoder Sigmoid() removed (see _make_decoder_layers) + kl_anneal_epochs
# default corrected (see EncoderConfig). VERSION is hashed into every
# fingerprint (fp_tnf/fp_te/fp_cov/fp2 below), so bumping it is what forces a
# real retrain instead of silently reusing a checkpoint trained under the old
# (Sigmoid-bounded) decoder architecture -- removing Sigmoid changes no layer
# shapes, so old weights would otherwise load with zero errors and just be
# wrong.
VERSION = "v8.3"

# Fixed, structural feature widths (see tnf_gene.py / te_composition.py) —
# used to catch dimension drift that a shape[0]-only check would miss.
TNF_EXPECTED_DIM = 136
TE_EXPECTED_DIM = 5

# coverage.py's COLUMN_LAYOUT_METADATA_NAMES prefix width:
# [valid_mask | n_samples_present | mean_dist | std_dist] then cov_1..cov_N.
COV_METADATA_DIM = 4


# =============================================================================
# CHECKPOINT FINGERPRINTING — same pattern as preprocessing.py / coverage.py /
# tnf_gene.py / te_composition.py. A stale or config-mismatched checkpoint is
# a cache MISS, not a silent wrong answer.
# =============================================================================

def _file_fingerprint(path) -> str:
    """Size+mtime fingerprint, not a content hash — same acknowledged
    limitation as the identical helper in the other pipeline modules."""
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
                    f"ran — ignoring stale checkpoint and re-running.")
        return None
    missing = [p for p in output_paths if p and not Path(p).exists()]
    if missing:
        log.warning(f"{step}: checkpoint fingerprint matches, but {len(missing)} referenced "
                    f"output(s) no longer exist on disk ({missing[:3]}"
                    f"{', ...' if len(missing) > 3 else ''}) — treating as a cache miss.")
        return None
    return prev


# =============================================================================
# ENCODER CONFIG
# =============================================================================

@dataclass
class EncoderConfig:
    """
    All encoder hyperparameters in one dataclass.

    te_include=True/False   -> TE encoder active/skipped
    cov_include=True/False  -> COV encoder active/skipped
    TNF is always active.
    """

    # -- Modality toggles --------------------------------------------------
    tnf_include:           bool          = True
    te_include:            bool          = True
    cov_include:            bool          = True

    # -- Missing-modality policy (FIX 8) ------------------------------------
    allow_missing_modalities: bool       = False

    # -- Parallelism -----------------------------------------------------
    parallel_phase1:        bool          = True    # train active Phase-1 VAEs concurrently
    parallel_phase1_max_workers: Optional[int] = None  # None = one worker per active modality

    # -- Device (FIX 10) -----------------------------------------------------
    device:                 str           = "auto"   # "cpu" | "gpu" | "auto"

    # -- Latent dimensions --------------------------------------------------
    latent_dim_tnf:        int           = 64
    latent_dim_te:         int           = 5
    latent_dim_cov:        Optional[int] = None
    final_dim:             Optional[int] = None

    # -- Architecture --------------------------------------------------------
    hidden_scale:          int           = 4
    hidden_min:            int           = 32
    hidden_max:            int           = 512
    dropout:                float         = 0.1
    n_hidden_layers:       int           = 2

    # -- beta-VAE ---------------------------------------------------------------
    beta_tnf:               float         = 0.01
    beta_te:                float         = 0.1
    beta_cov:                float         = 0.1
    # v8.3 FIX: was 75, but phase1_epochs_te/phase1_epochs_cov default to 50
    # -- kl_anneal_epochs > an encoder's own epoch count means that encoder
    # finishes training before its KL term ever reaches full weight (this
    # was already caught by the FLAG:KL_ANNEAL_MISMATCH warning below, but
    # only warned, never corrected). 50 matches TE/COV exactly; TNF (75
    # epochs) just reaches full beta at epoch 50 and trains the remaining
    # 25 at full weight, which is fine.
    kl_anneal_epochs:       int           = 50
    kl_free_bits:           float         = 0.01

    # -- Training epochs -----------------------------------------------------------
    phase1_epochs_tnf:      int           = 75
    phase1_epochs_te:       int           = 50
    phase1_epochs_cov:      int           = 50
    phase2_epochs:          int           = 50

    # -- Learning rates -----------------------------------------------------------------
    phase1_lr:               float         = 1e-3
    phase2_lr:                float         = 5e-5

    # -- Loss scaling ------------------------------------------------------
    # loss_scale_tnf/te/cov are used ONLY during Phase 1.
    # loss_scale_fusion weights the reconstruction-through-fusion loss.
    loss_scale_tnf:            float         = 1.0
    loss_scale_te:             float         = 0.1
    loss_scale_cov:             float         = 0.1
    loss_scale_fusion:         float         = 0.05

    # -- Fusion gate mode (FIX 13, gate input widened by FIX 23) -------------
    fusion_mode:               str           = "global"   # "global" | "contig_weighted"
    fusion_gate_entropy_weight: float        = 0.01        # only used if fusion_mode="contig_weighted"

    # -- Coverage feature split (FIX 5) --------------------------------------
    cov_use_n_samples_present_as_feature: bool = False

    # -- Batch size ---------------------------------------------------------------------
    batch_size:                Optional[int] = None

    # -- Reproducibility ------------------------------------------------------------------
    seed:                       int           = 42

    # -- Input dims (set at runtime, do NOT put in config.yaml) ---------------------------
    input_dim_tnf:               Optional[int] = None
    input_dim_te:                 Optional[int] = None
    input_dim_cov:                 Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "EncoderConfig":
        if not HAS_YAML:
            raise ImportError("PyYAML not installed: pip install pyyaml")
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        encoder_d = d.get("encoder", d)
        valid = {k: v for k, v in encoder_d.items()
                 if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderConfig":
        valid = {k: v for k, v in d.items()
                 if k in cls.__dataclass_fields__}
        return cls(**valid)

    def resolve(self, n_contigs: int) -> "EncoderConfig":
        if not (self.tnf_include or self.te_include or self.cov_include):
            raise ValueError(
                "At least one of tnf_include/te_include/cov_include must be True — "
                "cannot run encoder with zero active modalities."
            )

        if self.fusion_mode not in ("global", "contig_weighted"):
            raise ValueError(
                f"fusion_mode must be 'global' or 'contig_weighted', got {self.fusion_mode!r}. "
                f"There is no silent fallback — pick one explicitly."
            )
        if self.device not in ("cpu", "gpu", "auto"):
            raise ValueError(f"device must be one of 'cpu'/'gpu'/'auto', got {self.device!r}")
        if self.fusion_gate_entropy_weight < 0:
            raise ValueError(f"fusion_gate_entropy_weight must be >= 0, got {self.fusion_gate_entropy_weight}")

        cfg = EncoderConfig(**asdict(self))

        if not cfg.tnf_include:
            cfg.latent_dim_tnf = 0

        if not cfg.te_include:
            cfg.latent_dim_te = 0

        if cfg.cov_include:
            if cfg.latent_dim_cov is None:
                cov_input = cfg.input_dim_cov or 3
                cfg.latent_dim_cov = max(3, cov_input // 2)
        else:
            cfg.latent_dim_cov = 0

        active = 0
        if cfg.tnf_include:
            active += cfg.latent_dim_tnf
        if cfg.te_include:
            active += cfg.latent_dim_te
        if cfg.cov_include:
            active += cfg.latent_dim_cov
        cfg.final_dim = active

        if cfg.batch_size is None:
            cfg.batch_size = int(np.clip(n_contigs // 500, 32, 2048))

        # ---------------------------------------------------------------
        # kl_anneal_epochs / phase1_epochs mismatch guard (v7 FIX 3,
        # unchanged in v8).
        # ---------------------------------------------------------------
        phase1_epoch_counts = []
        if cfg.tnf_include:
            phase1_epoch_counts.append(("tnf", cfg.phase1_epochs_tnf))
        if cfg.te_include:
            phase1_epoch_counts.append(("te", cfg.phase1_epochs_te))
        if cfg.cov_include:
            phase1_epoch_counts.append(("cov", cfg.phase1_epochs_cov))

        for name, n_epochs in phase1_epoch_counts:
            if cfg.kl_anneal_epochs > n_epochs:
                effective_frac = n_epochs / cfg.kl_anneal_epochs
                log.warning(
                    f"FLAG:KL_ANNEAL_MISMATCH — kl_anneal_epochs={cfg.kl_anneal_epochs} > "
                    f"phase1_epochs_{name}={n_epochs}. Training for '{name}' will END at only "
                    f"{effective_frac:.0%} of its configured beta value (effective beta_{name} "
                    f"≈ {effective_frac:.3f} × configured). Set kl_anneal_epochs <= phase1_epochs_{name} "
                    f"if this is unintentional."
                )

        return cfg

    def modality_str(self) -> str:
        parts = []
        if self.tnf_include:
            parts.append("TNF")
        if self.te_include:
            parts.append("TE")
        if self.cov_include:
            parts.append("COV")
        return " + ".join(parts) if parts else "NONE"

    def hidden_dim(self, input_dim: int) -> int:
        return int(np.clip(input_dim * self.hidden_scale,
                           self.hidden_min, self.hidden_max))

    def log_summary(self):
        tnf_lat = f"TNF={self.latent_dim_tnf}" if self.tnf_include else "TNF=DISABLED"
        tnf_in  = f"TNF={self.input_dim_tnf}"  if self.tnf_include else "TNF=n/a"
        te_lat  = f"TE={self.latent_dim_te}"   if self.te_include  else "TE=DISABLED"
        te_in   = f"TE={self.input_dim_te}"    if self.te_include  else "TE=n/a"
        cov_lat = f"COV={self.latent_dim_cov}" if self.cov_include else "COV=DISABLED"
        cov_in  = f"COV={self.input_dim_cov}"  if self.cov_include else "COV=n/a"
        log.info("EncoderConfig:")
        log.info(f"  tnf_include : {self.tnf_include}   te_include : {self.te_include}   cov_include : {self.cov_include}"
                 f"   allow_missing_modalities={self.allow_missing_modalities}")
        log.info(f"  input dims  : {tnf_in} {te_in} {cov_in}")
        log.info(f"  latent dims : {tnf_lat} {te_lat} {cov_lat} -> FINAL={self.final_dim}")
        beta_str = ""
        if self.tnf_include:
            beta_str += f" TNF={self.beta_tnf}"
        if self.te_include:
            beta_str += f" TE={self.beta_te}"
        if self.cov_include:
            beta_str += f" COV={self.beta_cov}"
        log.info(f"  beta        :{beta_str}")
        log.info(f"  kl_anneal   : {self.kl_anneal_epochs} epochs  free_bits={self.kl_free_bits}")
        epoch_str = ""
        if self.tnf_include:
            epoch_str += f" tnf={self.phase1_epochs_tnf}"
        if self.te_include:
            epoch_str += f" te={self.phase1_epochs_te}"
        if self.cov_include:
            epoch_str += f" cov={self.phase1_epochs_cov}"
        epoch_str += f" phase2={self.phase2_epochs}"
        log.info(f"  epochs      :{epoch_str}")
        log.info(f"  lr          : phase1={self.phase1_lr} phase2={self.phase2_lr}")
        log.info(f"  loss_scale  : phase1(tnf/te/cov)={self.loss_scale_tnf}/{self.loss_scale_te}/{self.loss_scale_cov}"
                 f"  fusion(recon)={self.loss_scale_fusion}")
        log.info(f"  fusion_mode : {self.fusion_mode}"
                 + (f"  gate_entropy_weight={self.fusion_gate_entropy_weight}" if self.fusion_mode == "contig_weighted" else ""))
        log.info(f"  device      : {self.device}   batch_size={self.batch_size}  dropout={self.dropout}  seed={self.seed}")
        log.info(f"  [v8] Phase 2: sub-encoders FROZEN, fusion trained via per-contig weighted "
                 f"reconstruction-through-fusion loss")


# =============================================================================
# DEVICE RESOLUTION (FIX 10)
# =============================================================================

def resolve_device(requested: str) -> "torch.device":
    """
    requested: "cpu" | "gpu" | "auto" (see EncoderConfig.device).
      "cpu"  -> always CPU, regardless of what's available.
      "gpu"  -> CUDA if available; else MPS with a WARNING; else CPU with a
                WARNING. A missing GPU is an environment/hardware fact, not
                a data-correctness risk, so this degrades rather than
                raising — unlike FIX 8's missing-feature-file handling.
      "auto" -> the original v7 behavior: CUDA -> MPS -> CPU, silently.
    """
    requested = (requested or "auto").lower()
    if requested not in ("cpu", "gpu", "auto"):
        raise ValueError(f"device must be one of 'cpu'/'gpu'/'auto', got {requested!r}")

    if requested == "cpu":
        device = torch.device("cpu")
        log.info("Using CPU (explicitly requested via config)")
    elif requested == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log.warning("device='gpu' requested but no CUDA device is available — using Apple MPS instead")
        else:
            device = torch.device("cpu")
            log.warning("device='gpu' requested but no CUDA or MPS device is available — falling back to CPU")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"Using GPU: {torch.cuda.get_device_name(0)} (auto)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Using Apple MPS (auto)")
        else:
            device = torch.device("cpu")
            log.info("Using CPU (auto)")

    n_threads = min(os.cpu_count() or 32, 32)
    torch.set_num_threads(n_threads)
    log.info(f"PyTorch threads: {n_threads}")
    return device


def get_device() -> "torch.device":
    """Backward-compatible alias for resolve_device("auto") (v7's only mode)."""
    return resolve_device("auto")


# =============================================================================
# MODALITY MANIFEST LOADING / ALIGNMENT (FIX 6)
# =============================================================================

def _sha256_id_order(contig_ids: List[str]) -> str:
    return hashlib.sha256("\n".join(contig_ids).encode()).hexdigest()[:16]


def _load_modality_manifest(features_path: str, modality: str) -> Tuple[List[str], str]:
    """
    Load the authoritative, ordered contig-ID list for one modality's
    feature file, from that modality's OWN sibling manifest files —
    verified directly against each upstream module's real save code, not
    assumed uniform from any docstring claim:

      TNF (tnf_gene.py)      : <dir>/contig_ids.json  (JSON list)
                                + <dir>/tnf_feature_schema.json["contig_order_hash"]
      TE  (te_composition.py): <dir>/contig_ids.json  (JSON list)
                                + <dir>/schema.json["contig_order_hash"]
      COV (coverage.py)      : <dir>/contig_ids.txt  (newline-delimited
                                text; the FULL/authoritative order — NOT
                                raw_coverage_contig_ids.txt, which only
                                covers the working/kept subset and does
                                NOT match coverage_features.npy's row order)
                                + <dir>/manifest.json["contig_id_order_hash"]

    All three use the IDENTICAL hash formula
    sha256("\\n".join(contig_ids)).hexdigest()[:16], just under different
    key names in different files — so hashes ARE directly comparable
    across modalities. That comparison (done by the caller, across
    modalities) is the real alignment check; the check against each
    modality's OWN stored hash here is only a self-consistency sanity
    check (did this contig_ids file get edited/regenerated out of sync
    with its own recorded hash).
    """
    if modality not in ("tnf", "te", "cov"):
        raise ValueError(f"_load_modality_manifest: unknown modality {modality!r}")

    d = Path(features_path).parent
    if modality == "tnf":
        ids_path, manifest_path, hash_key = d / "contig_ids.json", d / "tnf_feature_schema.json", "contig_order_hash"
    elif modality == "te":
        ids_path, manifest_path, hash_key = d / "contig_ids.json", d / "schema.json", "contig_order_hash"
    else:  # cov
        ids_path, manifest_path, hash_key = d / "contig_ids.txt", d / "manifest.json", "contig_id_order_hash"

    if not ids_path.exists():
        raise FileNotFoundError(
            f"{modality.upper()}: expected contig-ID manifest at {ids_path} (sibling of "
            f"{features_path}) but it does not exist — cannot verify contig alignment against "
            f"the other modalities. Refusing to assume row order without it."
        )

    if modality == "cov":
        contig_ids = [line for line in ids_path.read_text().splitlines() if line != ""]
    else:
        with open(ids_path) as f:
            contig_ids = json.load(f)

    recomputed_hash = _sha256_id_order(contig_ids)

    stored_hash = None
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                stored_hash = json.load(f).get(hash_key)
        except Exception as e:
            log.warning(f"{modality.upper()}: could not read {manifest_path} ({e}) — skipping "
                        f"self-consistency hash check (the contig_ids file itself is still used).")
    else:
        log.warning(f"{modality.upper()}: {manifest_path} not found — skipping self-consistency "
                    f"hash check (the contig_ids file itself is still used).")

    if stored_hash is not None and stored_hash != recomputed_hash:
        raise ValueError(
            f"{modality.upper()}: contig_ids at {ids_path} do not match this modality's own "
            f"recorded '{hash_key}' in {manifest_path} ({recomputed_hash} != {stored_hash}) — "
            f"the contig-ID file may have been edited or regenerated out of sync with the "
            f"feature array. Refusing to proceed."
        )

    return contig_ids, recomputed_hash


def _check_contig_alignment(ids_by_mod: Dict[str, Tuple[List[str], str]]) -> Tuple[List[str], str]:
    """Cross-modality alignment check (replaces v7's shape[0]-only check).
    Returns (contig_ids, order_hash) of the shared, verified order."""
    if not ids_by_mod:
        raise RuntimeError("_check_contig_alignment: no active modality to align")

    ref_mod = next(iter(ids_by_mod))
    ref_ids, ref_hash = ids_by_mod[ref_mod]
    for mod, (ids, h) in ids_by_mod.items():
        if h != ref_hash or ids != ref_ids:
            raise ValueError(
                f"Contig alignment mismatch: '{mod}' contig-ID order does not match '{ref_mod}'. "
                f"'{ref_mod}' has {len(ref_ids)} contigs (order_hash={ref_hash}); "
                f"'{mod}' has {len(ids)} contigs (order_hash={h}). All modalities must be computed "
                f"from the exact same assembly with the exact same contig ordering."
            )
    return ref_ids, ref_hash


# =============================================================================
# COVERAGE FEATURE SPLIT (FIX 5)
# =============================================================================

def split_cov_features(cov_raw: np.ndarray, cfg: "EncoderConfig") -> Tuple[np.ndarray, np.ndarray]:
    """
    coverage.py's coverage_features.npy is always
    [valid_mask | n_samples_present | mean_dist | std_dist | cov_1..cov_N]
    (see coverage.py's COLUMN_LAYOUT_METADATA_NAMES / COLUMN_ROLES).
    valid_mask is documented as MASK ONLY, never a feature. n_samples_present
    is a QC/filtering signal by default; this module leaves it out of the
    feature vector unless cfg.cov_use_n_samples_present_as_feature is
    explicitly set (coverage.py's own manifest note: "column/layout
    SELECTION is the consumer's decision").

    Returns (features_for_vae, reliability_weight) — reliability_weight is
    exactly valid_mask (1.0 real row / 0.0 placeholder row), used as COV's
    per-contig weight in both Phase 1 (vae_loss masking) and Phase 2
    (readback_loss weighting).
    """
    if cov_raw.ndim != 2 or cov_raw.shape[1] < COV_METADATA_DIM + 1:
        raise ValueError(
            f"COV: expected >= {COV_METADATA_DIM + 1} columns "
            f"(valid_mask, n_samples_present, mean_dist, std_dist + >=1 coverage column), "
            f"got shape {cov_raw.shape}"
        )

    valid_mask        = cov_raw[:, 0].astype(np.float32)
    n_samples_present = cov_raw[:, 1].astype(np.float32)
    dist_and_cov       = cov_raw[:, 2:]  # mean_dist, std_dist, cov_1..cov_N

    if cfg.cov_use_n_samples_present_as_feature:
        features = np.concatenate(
            [n_samples_present.reshape(-1, 1), dist_and_cov], axis=1
        ).astype(np.float32)
    else:
        features = dist_and_cov.astype(np.float32)

    n_placeholder = int((valid_mask == 0).sum())
    if n_placeholder:
        log.info(f"COV: {n_placeholder:,}/{len(valid_mask):,} contig(s) are valid_mask=0 "
                 f"placeholders — excluded from COV's own loss via reliability weight=0.")

    return features, valid_mask.copy()


# =============================================================================
# MISSING-MODALITY / SMALL-DATASET GUARDS (FIX 8 / FIX 12)
# =============================================================================

def _check_modality_available(cfg_flag: bool, path: Optional[str], name: str,
                               allow_missing: bool) -> bool:
    if not cfg_flag:
        return False
    if path is None or not Path(path).exists():
        msg = (f"{name}_include=True but {name}_features_path is None or missing ({path}) — "
               f"cannot proceed with this modality active.")
        if allow_missing:
            log.warning(msg + " allow_missing_modalities=True — disabling this modality and continuing.")
            return False
        raise FileNotFoundError(
            msg + " Set allow_missing_modalities=True in config if silently dropping this "
            "modality is genuinely intended; otherwise supply the missing path."
        )
    return True


def _effective_batch_size(n: int, requested: int) -> int:
    """Small-dataset guard: never request a batch size larger than the
    dataset itself. Makes the actual effective batch size explicit instead
    of relying on DataLoader to silently hand back one small final batch."""
    eff = max(1, min(int(requested), int(n)))
    if eff != requested:
        log.warning(f"batch_size clamped from {requested} to {eff} (dataset has only {n} row(s))")
    return eff


def _guard_nonzero_weights(weights: Optional[np.ndarray], name: str) -> None:
    if weights is not None and not bool(np.any(weights > 0)):
        raise ValueError(
            f"{name}: every per-contig weight is 0 — every contig would be masked out of the "
            f"loss for this modality. Refusing to train on nothing; check the upstream "
            f"weight/mask computation for {name}."
        )


# =============================================================================
# UTILITY  (validate_features/validate_weights/normalize_features/
# make_loader/get_all_latents — unchanged from v7)
# =============================================================================

def validate_features(arr: np.ndarray, name: str,
                       expected_dim: Optional[int] = None) -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name}: feature array is None")
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 2D array, got {arr.shape}")
    if arr.shape[0] == 0:
        # FIX 19: an empty feature array would otherwise hit .min()/.max()/
        # mean() below on an empty axis, raising an opaque numpy error
        # ("zero-size array to reduction operation") far from the real
        # cause. Fail loud with a message that actually names the problem.
        raise ValueError(f"{name}: feature array has 0 rows (shape={arr.shape}) -- nothing to "
                          f"validate or train on. Check the upstream feature file.")
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(f"{name}: expected dim={expected_dim}, got {arr.shape[1]}")
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan > 0 or n_inf > 0:
        log.warning(f"{name}: replacing {n_nan} NaN + {n_inf} Inf with 0")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    zero_rows = int((arr == 0).all(axis=1).sum())
    pct       = zero_rows / arr.shape[0] * 100
    if pct > 50:
        log.warning(f"{name}: {zero_rows:,}/{arr.shape[0]:,} ({pct:.0f}%) rows all-zero "
                    f"(expected for sparse signals like TE)")
    log.info(f"{name}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} "
             f"mean={arr.mean():.4f} zero_rows={zero_rows:,}")
    return arr


def validate_weights(arr: np.ndarray, name: str) -> Optional[np.ndarray]:
    if arr is None:
        log.warning(f"{name}: no weights provided -- using uniform 1.0")
        return None
    arr = np.clip(np.array(arr, dtype=np.float32).squeeze(), 0.0, 1.0)
    log.info(f"{name}: zero={int((arr==0).sum()):,} full={int((arr==1).sum()):,} mean={arr.mean():.3f}")
    return arr


def normalize_features(arr: np.ndarray, name: str, mask: Optional[np.ndarray] = None):
    """
    FIX 16: when `mask` is supplied (e.g. COV's valid_mask), the
    normalization statistics (mean/std) are computed ONLY over rows where
    mask > 0. Without this, placeholder rows (COV's valid_mask=0 rows are
    literal zero-filled placeholders inserted by coverage.py's step7_realign
    -- not real data) would contaminate the mean/std that every row,
    including the real ones, is then normalized against. The resulting
    normalized values are still computed for every row (including
    placeholders) using those valid-only statistics -- only the STATISTICS
    themselves exclude invalid rows, not the transform's output rows.
    """
    if mask is not None:
        valid = mask > 0
        n_valid = int(valid.sum())
        if n_valid == 0:
            raise ValueError(f"{name}: normalize_features got a mask with zero valid rows -- "
                              f"cannot compute normalization statistics from nothing.")
        mean = arr[valid].mean(axis=0)
        std  = arr[valid].std(axis=0)
    else:
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
    std[std == 0] = 1.0
    norm = ((arr - mean) / std).astype(np.float32)
    if mask is not None:
        log.info(f"{name}: normalized to zero mean unit variance "
                 f"(statistics computed over {n_valid:,}/{arr.shape[0]:,} valid rows only)")
    else:
        log.info(f"{name}: normalized to zero mean unit variance")
    return norm, mean, std


def make_loader(features, weights, batch_size, shuffle=True):
    x  = torch.tensor(features, dtype=torch.float32)
    ds = (TensorDataset(x, torch.tensor(weights, dtype=torch.float32))
          if weights is not None else TensorDataset(x))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def get_all_latents(model, features, device, batch_size):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(features, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False)
    parts = []
    with torch.no_grad():
        for (x,) in loader:
            parts.append(model.get_latent(x.to(device)).cpu().numpy())
    return np.concatenate(parts, axis=0)


# =============================================================================
# UNIFORM beta-VAE BASE  (CORE — unchanged, diagnosed healthy)
# =============================================================================

def _make_encoder_layers(input_dim: int, cfg: EncoderConfig):
    hidden = cfg.hidden_dim(input_dim)
    layers = []
    prev   = input_dim
    for i in range(cfg.n_hidden_layers):
        dim = hidden if i == 0 else max(cfg.hidden_min, hidden // 2)
        layers += [nn.Linear(prev, dim), nn.LayerNorm(dim), nn.ReLU()]
        prev = dim
    return nn.Sequential(*layers), prev


def _make_decoder_layers(latent_dim: int, output_dim: int, cfg: EncoderConfig) -> nn.Sequential:
    hidden = cfg.hidden_dim(output_dim)
    layers = []
    prev   = latent_dim
    dims   = ([max(cfg.hidden_min, hidden // 2)] * (cfg.n_hidden_layers - 1) + [hidden]
              if cfg.n_hidden_layers > 1 else [hidden])
    for dim in dims:
        layers += [nn.Linear(prev, dim), nn.LayerNorm(dim), nn.ReLU()]
        prev = dim
    # v8.3 FIX: final activation used to be nn.Sigmoid(), which bounds the
    # decoder's output to (0,1). The reconstruction target (`original` in
    # vae_loss) is normalize_features()'s z-scored TNF/COV/TE output --
    # zero mean, unit variance, roughly half NEGATIVE by construction. A
    # Sigmoid decoder can never emit a negative value, so MSE against those
    # targets had an irreducible floor for every feature below its own
    # mean -- biasing the reconstruction loss (and therefore mu/log_var,
    # and therefore the latent space and every downstream clustering
    # decision) for as long as this stayed in place. Output is linear now,
    # matching the actual (unbounded, signed) target range.
    layers += [nn.Linear(prev, output_dim)]
    return nn.Sequential(*layers)


class UniformVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, cfg: EncoderConfig):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        enc_layers, enc_out = _make_encoder_layers(input_dim, cfg)
        self.encoder    = enc_layers
        self.fc_mu      = nn.Linear(enc_out, latent_dim)
        self.fc_log_var = nn.Linear(enc_out, latent_dim)
        self.decoder    = _make_decoder_layers(latent_dim, input_dim, cfg)

    def encode(self, x):
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10, 10)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def get_latent(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


# =============================================================================
# FUSION MODULE  (FIX 9: per-contig weighted readback loss;
#                 FIX 13: contig_weighted gate mode;
#                 FIX 23 (v8.2): contig_weighted gate now also SEES the
#                 per-contig reliability weights, not just the latents)
# =============================================================================

class FusionModule(nn.Module):
    """
    Gated fusion for any active combination of TNF, TE, COV (all
    independently optional — at least one must be True). Canonical order:
    TNF -> TE -> COV.

    Two fusion_mode variants, selected by EncoderConfig.fusion_mode:

      "global"          — v7's original mechanism: ONE learned softmax
                           gate (nn.Parameter), shared by every contig.

      "contig_weighted" — a small GMU-style (Arevalo et al. 2017) MLP gate
                           that takes each contig's own concatenated
                           projected latents PLUS (as of v8.2, FIX 23) each
                           modality's per-contig reliability weight for
                           that contig, and outputs a PER-CONTIG softmax
                           over modalities. Trained via the identical
                           reconstruction-through-fusion loss as "global"
                           (see readback_loss), plus an optional entropy
                           penalty (gate_entropy) to discourage per-contig
                           collapse onto one modality for every contig.

    Both modes keep the v7 readback_X heads (final_dim -> latent_dim_X),
    used only during Phase 2 training, never at inference.
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.tnf_include = cfg.tnf_include
        self.te_include  = cfg.te_include
        self.cov_include = cfg.cov_include
        self.fusion_mode = cfg.fusion_mode

        if not (self.tnf_include or self.te_include or self.cov_include):
            raise ValueError("FusionModule: at least one of tnf_include/te_include/cov_include must be True")
        if self.fusion_mode not in ("global", "contig_weighted"):
            raise ValueError(f"FusionModule: unknown fusion_mode {cfg.fusion_mode!r}")

        self._mods = []
        in_dim = 0
        if cfg.tnf_include:
            self._mods.append("tnf")
            in_dim += cfg.latent_dim_tnf
        if cfg.te_include:
            self._mods.append("te")
            in_dim += cfg.latent_dim_te
        if cfg.cov_include:
            self._mods.append("cov")
            in_dim += cfg.latent_dim_cov

        n_mods = len(self._mods)

        if self.fusion_mode == "global":
            self.gate_logits = nn.Parameter(torch.zeros(n_mods))
            self.gate_net = None
        else:  # contig_weighted
            self.gate_logits = None
            # FIX 23 (v8.2): gate input now also includes each active
            # modality's per-contig reliability weight (one scalar per
            # modality), not just the projected latents -- see module
            # docstring / changelog for why.
            gate_in_dim = in_dim + n_mods
            gate_hidden = max(8, gate_in_dim // 2)
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, n_mods),
            )
        # Last-forward per-contig gate weights, kept (with grad) for
        # gate_entropy() and (detached) for saving contig_fusion_weights.npy.
        # Populated by forward(); shape (1, n_mods) in "global" mode,
        # (batch, n_mods) in "contig_weighted" mode.
        self._last_gate_weights = None

        if cfg.tnf_include:
            self.proj_tnf = nn.Linear(cfg.latent_dim_tnf, cfg.latent_dim_tnf)
        if cfg.te_include:
            self.proj_te = nn.Linear(cfg.latent_dim_te, cfg.latent_dim_te)
        if cfg.cov_include:
            self.proj_cov = nn.Linear(cfg.latent_dim_cov, cfg.latent_dim_cov)

        self.fusion = nn.Sequential(
            nn.Linear(in_dim, cfg.final_dim),
            nn.LayerNorm(cfg.final_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        # Readback heads (v7 FIX 2): fused space -> each modality's own
        # latent dimensionality. Training-only; not used at inference.
        if cfg.tnf_include:
            self.readback_tnf = nn.Linear(cfg.final_dim, cfg.latent_dim_tnf)
        if cfg.te_include:
            self.readback_te = nn.Linear(cfg.final_dim, cfg.latent_dim_te)
        if cfg.cov_include:
            self.readback_cov = nn.Linear(cfg.final_dim, cfg.latent_dim_cov)

    def forward(self, z_tnf=None, z_te=None, z_cov=None,
                w_tnf=None, w_te=None, w_cov=None):
        """
        w_tnf/w_te/w_cov (FIX 23, v8.2): optional per-contig reliability
        weight tensors, shape (batch,), one per ACTIVE modality. Only used
        in "contig_weighted" mode, where they're concatenated onto the
        gate's input alongside the projected latents so the gate can see
        each contig's per-modality reliability directly instead of only
        inferring it from the latent vectors themselves. Ignored entirely
        in "global" mode (that gate has no per-contig input at all).
        Missing/omitted weights default to 1.0 for that modality/contig --
        the same convention already used by readback_loss (FIX 9).
        """
        proj = {}
        if self.tnf_include:
            if z_tnf is None:
                raise ValueError("tnf_include=True but z_tnf is None in FusionModule.forward()")
            proj["tnf"] = self.proj_tnf(z_tnf)
        if self.te_include:
            if z_te is None:
                raise ValueError("te_include=True but z_te is None in FusionModule.forward()")
            proj["te"] = self.proj_te(z_te)
        if self.cov_include:
            if z_cov is None:
                raise ValueError("cov_include=True but z_cov is None in FusionModule.forward()")
            proj["cov"] = self.proj_cov(z_cov)

        ordered = [proj[m] for m in self._mods]
        concat_raw = torch.cat(ordered, dim=1)  # (batch, in_dim)

        if self.fusion_mode == "global":
            w = F.softmax(self.gate_logits, dim=0)              # (n_mods,)
            gated = [ordered[i] * w[i] for i in range(len(ordered))]
            self._last_gate_weights = w.unsqueeze(0)             # (1, n_mods)
        else:  # contig_weighted
            batch_n = concat_raw.shape[0]
            device  = concat_raw.device
            w_by_mod = {"tnf": w_tnf, "te": w_te, "cov": w_cov}
            w_parts = []
            for m in self._mods:
                w_m = w_by_mod[m]
                if w_m is None:
                    w_m = torch.ones(batch_n, device=device)
                w_parts.append(w_m.reshape(batch_n, 1))
            gate_input = torch.cat([concat_raw] + w_parts, dim=1)   # (batch, in_dim + n_mods)
            gate_logits_c = self.gate_net(gate_input)               # (batch, n_mods)
            w = F.softmax(gate_logits_c, dim=1)                     # (batch, n_mods)
            gated = [ordered[i] * w[:, i:i + 1] for i in range(len(ordered))]
            self._last_gate_weights = w                             # (batch, n_mods)

        z = torch.cat(gated, dim=1)
        return self.fusion(z)

    def readback_loss(self, z_fus, mu_tnf=None, mu_te=None, mu_cov=None,
                       w_tnf=None, w_te=None, w_cov=None):
        """
        v8: PER-CONTIG weighted-reliability readback loss (FIX 9). For each
        active modality X, computes a per-contig reconstruction error
        err_X_i = MSE over that modality's own latent dims between
        readback_X(z_fus)_i and mu_X_i.detach(). These per-contig errors
        are combined into ONE scalar per contig using each modality's OWN
        reliability weight w_X_i — NOT the fusion gate weight, a different
        concept (see module docstring):

            contig_loss_i = (w_tnf_i*err_tnf_i + w_te_i*err_te_i + w_cov_i*err_cov_i)
                            / (w_tnf_i + w_te_i + w_cov_i)

        Contigs with total weight 0 across every active modality
        contribute nothing to the batch mean (they have no reliable target
        to reconstruct toward at all). A weight defaults to 1.0 for any
        modality that didn't supply one (e.g. TNF/TE with no confidence
        weight file).

        This is a WEIGHTED-AVERAGE policy, not an intersection/all-valid
        policy — TE's reliability weight is near-all-zero for fragmented
        real assemblies by design (te_composition.py), so requiring every
        modality to be valid before a contig counts at all would starve
        Phase 2 of nearly all its training data.
        """
        batch_n = z_fus.shape[0]
        device  = z_fus.device
        total_w    = torch.zeros(batch_n, device=device)
        total_werr = torch.zeros(batch_n, device=device)

        if self.tnf_include and mu_tnf is not None:
            err = F.mse_loss(self.readback_tnf(z_fus), mu_tnf.detach(), reduction="none").mean(dim=1)
            w = w_tnf if w_tnf is not None else torch.ones(batch_n, device=device)
            total_werr = total_werr + w * err
            total_w    = total_w + w
        if self.te_include and mu_te is not None:
            err = F.mse_loss(self.readback_te(z_fus), mu_te.detach(), reduction="none").mean(dim=1)
            w = w_te if w_te is not None else torch.ones(batch_n, device=device)
            total_werr = total_werr + w * err
            total_w    = total_w + w
        if self.cov_include and mu_cov is not None:
            err = F.mse_loss(self.readback_cov(z_fus), mu_cov.detach(), reduction="none").mean(dim=1)
            w = w_cov if w_cov is not None else torch.ones(batch_n, device=device)
            total_werr = total_werr + w * err
            total_w    = total_w + w

        has_weight = (total_w > 0).float()
        n_valid    = has_weight.sum().clamp(min=1.0)
        per_contig = torch.where(total_w > 0, total_werr / total_w.clamp(min=1e-8),
                                  torch.zeros_like(total_w))
        return (per_contig * has_weight).sum() / n_valid

    def gate_entropy(self) -> "torch.Tensor":
        """
        FIX 13 safeguard: an entropy-based penalty for the contig_weighted
        gate, loosely inspired by sparsity/collapse safeguards in
        confidence-weighted per-instance gating approaches (not a literal
        reproduction of any one paper). Only meaningful when
        fusion_mode="contig_weighted" — a per-contig softmax that always
        collapses to one-hot for every contig has effectively re-derived a
        "global" gate at extra parameter/compute cost, and is a known
        failure mode for unsupervised instance-wise gates trained purely
        through reconstruction. Returns the NEGATIVE mean per-contig
        entropy, so that ADDING `fusion_gate_entropy_weight * gate_entropy()`
        to the loss (i.e. minimizing it) pushes entropy UP, discouraging
        collapse. Returns 0 (no gradient effect) in "global" mode.
        """
        if self.fusion_mode != "contig_weighted" or self._last_gate_weights is None:
            return torch.zeros((), device=next(self.parameters()).device)
        w = self._last_gate_weights.clamp(min=1e-8)
        entropy = -(w * w.log()).sum(dim=1)   # per-contig entropy, shape (batch,)
        return -entropy.mean()

    def get_contig_weights(self) -> Optional[np.ndarray]:
        """Per-contig gate weights from the most recent forward() call, as
        a numpy array of shape (batch, n_mods) — only meaningful (and only
        non-None) in fusion_mode='contig_weighted'."""
        if self.fusion_mode != "contig_weighted" or self._last_gate_weights is None:
            return None
        return self._last_gate_weights.detach().cpu().numpy()

    def get_weights(self) -> dict:
        if self.fusion_mode == "global":
            w = F.softmax(self.gate_logits, dim=0).detach()
            return {f"w_{m}": float(w[i]) for i, m in enumerate(self._mods)}
        # contig_weighted: report the mean +/- std over the most recent forward's batch.
        if self._last_gate_weights is None:
            return {f"w_{m}_mean": None for m in self._mods}
        w = self._last_gate_weights.detach()
        out = {}
        for i, m in enumerate(self._mods):
            out[f"w_{m}_mean"] = float(w[:, i].mean())
            out[f"w_{m}_std"]  = float(w[:, i].std())
        return out


# =============================================================================
# LOSS FUNCTIONS  (CORE — unchanged, still used in Phase 1)
# =============================================================================

def vae_loss(recon, original, mu, log_var, weights, beta, kl_free_bits):
    recon_per  = F.mse_loss(recon, original, reduction="none").mean(dim=1)
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim = kl_per_dim.clamp(min=kl_free_bits)
    kl_per     = kl_per_dim.sum(dim=1)

    if weights is not None:
        mask       = (weights > 0).float()
        n_act      = mask.sum().clamp(min=1.0)
        recon_loss = (recon_per * mask).sum() / n_act
        kl_loss    = (kl_per   * mask).sum() / n_act
    else:
        recon_loss = recon_per.mean()
        kl_loss    = kl_per.mean()

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# =============================================================================
# TRAINING FUNCTIONS  (CORE — Phase 1 loop body unchanged; batch_size is now
# an explicit parameter so the small-dataset clamp (FIX 12) can apply to it)
# =============================================================================

def _train_vae_epoch(model, loader, optimizer, device, beta, kl_free_bits, has_weights):
    model.train()
    tot = rec = kl = 0.0
    n   = 0
    for batch in loader:
        x   = batch[0].to(device)
        w   = batch[1].to(device) if has_weights else None
        optimizer.zero_grad()
        recon, mu, lv = model(x)
        loss, rl, kll = vae_loss(recon, x, mu, lv, w, beta, kl_free_bits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item(); rec += rl.item(); kl += kll.item(); n += 1
    return tot / n, rec / n, kl / n


def _train_phase1(model, features, weights, cfg, device, epochs, beta, name, batch_size):
    loader    = make_loader(features, weights, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.phase1_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg.phase1_lr * 0.1)
    has_w = weights is not None
    t0    = time.time()

    for epoch in range(1, epochs + 1):
        b            = min(beta, beta * epoch / max(cfg.kl_anneal_epochs, 1))
        tot, rec, kl = _train_vae_epoch(model, loader, optimizer, device, b, cfg.kl_free_bits, has_w)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            log.info(f"  {name} ep {epoch:3d}/{epochs} | loss={tot:.4f} recon={rec:.4f} kl={kl:.4f} "
                     f"beta={b:.3f} lr={scheduler.get_last_lr()[0]:.2e}")

    log.info(f"  {name} done in {time.time()-t0:.0f}s")


# =============================================================================
# PHASE 1 -- INDEPENDENT ENCODER TRAINING
# (fingerprint-aware checkpointing added — FIX 15; spawn context — FIX 11)
# =============================================================================

def _phase1_worker(job):
    """
    Top-level, picklable worker used by ProcessPoolExecutor to train one
    modality's Phase-1 VAE in a separate process (CPU-only parallelism by
    default; also works with device='gpu', see FIX 11's note about
    per-worker GPU contention).

    job: (name, features, weights, latent_dim, beta, epochs,
          cfg, device, ckpt_dir, ckpt_key, fp, n_threads)

    Returns (name, model, latent) with model already moved to CPU so it
    pickles back cleanly regardless of how/where it was trained.
    """
    (name, features, weights, latent_dim, beta, epochs,
     cfg, device, ckpt_dir, ckpt_key, fp, n_threads) = job
    try:
        torch.set_num_threads(max(1, n_threads))
    except Exception:
        pass
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model, latent = _phase1_single(name, features, weights, latent_dim, beta, epochs,
                                     cfg, device, ckpt_dir, ckpt_key, fp)
    model.to("cpu")
    return name, model, latent


def _phase1_single(name, features, weights, latent_dim, beta, epochs,
                    cfg, device, ckpt_dir, ckpt_key, fp):
    ckpt     = Checkpoint(ckpt_dir / name) if Checkpoint else None
    lat_path = ckpt_dir / name / f"latent_{name}.npy"
    mdl_path = ckpt_dir / name / f"{name}_model.pt"

    cached = _checkpoint_ok(ckpt, ckpt_key, fp, [str(lat_path), str(mdl_path)])
    if cached is not None:
        log.info(f"  [SKIP] [{ckpt_key}] (completed {cached.get('timestamp', '')})")
        model = UniformVAE(features.shape[1], latent_dim, cfg).to(device)
        model.load_state_dict(torch.load(mdl_path, map_location=device))
        return model, np.load(lat_path)

    log.info(f"Phase 1 [{name}]: input={features.shape[1]}D -> latent={latent_dim}D "
             f"hidden={cfg.hidden_dim(features.shape[1])}")
    model = UniformVAE(features.shape[1], latent_dim, cfg).to(device)
    eff_bs = _effective_batch_size(features.shape[0], cfg.batch_size)
    _train_phase1(model, features, weights, cfg, device, epochs, beta, name, eff_bs)

    latent = get_all_latents(model, features, device, eff_bs)
    log.info(f"  [{name}] latent std={latent.std():.4f}")

    (ckpt_dir / name).mkdir(parents=True, exist_ok=True)
    np.save(lat_path, latent)
    torch.save(model.state_dict(), mdl_path)
    if ckpt:
        ckpt.mark_done(ckpt_key, {"_fp": fp, "latent_std": float(latent.std()), "epochs": epochs})

    return model, latent


# =============================================================================
# PHASE 2 -- JOINT FUSION TRAINING
# (FIX 9 weighted readback loss, FIX 13 contig_weighted gate + entropy term,
#  FIX 14 fusion checkpoint persistence, FIX 15 fingerprinting,
#  FIX 23 (v8.2) gate now also sees per-contig reliability weights)
# =============================================================================

def _phase2_joint(fusion, cfg, device, ckpt_dir, outdir, fp2,
                   tnf_model=None, tnf_feat=None, tnf_w=None,
                   te_model=None, te_feat=None, te_w=None,
                   cov_model=None, cov_feat=None, cov_w=None):
    """
    v8. FIX 1/FIX 2 mechanics (frozen sub-encoders, reconstruction-through-
    fusion as the only training signal) are unchanged from v7. New in v8:

    FIX 9: readback_loss is now called with each active modality's
    per-contig reliability weight (tnf_w/te_w/cov_w — defaulting to all-1
    tensors when a modality has no supplied weight array), combined
    per-contig rather than as one whole-batch mean per modality.

    FIX 13: when cfg.fusion_mode == "contig_weighted", an entropy penalty
    (fusion.gate_entropy(), scaled by cfg.fusion_gate_entropy_weight) is
    added to the loss, and the full-dataset per-contig gate weights are
    saved to `<outdir>/contig_fusion_weights.npy`.

    FIX 14: the trained FusionModule's weights are now saved into this
    step's own checkpoint dir and reloaded into `fusion` on a cache hit,
    so a Phase-2 cache hit can no longer leave the caller holding an
    untrained fusion module while returning already-trained latents.

    FIX 15: checkpoint validity now depends on fp2 (a fingerprint of every
    Phase-2-relevant input/config value, including the upstream Phase-1
    fingerprints), not just is_done()+file-exists.

    FIX 23 (v8.2): both places fusion() is called below (the per-batch
    training loop and the final full-dataset forward pass) now pass the
    same w_tnf/w_te/w_cov reliability-weight tensors that readback_loss
    already used, so the contig_weighted gate can see them too -- see
    FusionModule.forward()'s docstring.
    """
    ckpt          = Checkpoint(ckpt_dir / "joint") if Checkpoint else None
    lat_path      = ckpt_dir / "joint" / "final_latent.npy"
    fusion_path   = ckpt_dir / "joint" / "fusion_model.pt"

    cached = _checkpoint_ok(ckpt, "joint_fusion", fp2, [str(lat_path), str(fusion_path)])
    if cached is not None:
        log.info("[SKIP] Phase 2 already done -- loading from checkpoint")
        fusion.load_state_dict(torch.load(fusion_path, map_location=device))  # FIX 14
        return np.load(lat_path)

    tnf_active = cfg.tnf_include and tnf_model is not None
    te_active  = cfg.te_include  and te_model  is not None
    cov_active = cfg.cov_include and cov_model is not None

    if not (tnf_active or te_active or cov_active):
        raise ValueError("_phase2_joint: no active modality (tnf/te/cov all inactive)")

    modality_str = ("TNF" if tnf_active else "") + ("+TE" if te_active else "") + ("+COV" if cov_active else "")
    log.info(f"Phase 2: Joint fusion training ({modality_str.lstrip('+')}) -- "
             f"[v8] sub-encoders FROZEN, gate='{cfg.fusion_mode}', "
             f"per-contig weighted reconstruction-through-fusion loss")

    # --- freeze all sub-encoders permanently for Phase 2 (v7 FIX 1, unchanged) ---
    if tnf_active:
        tnf_model.eval()
        for p in tnf_model.parameters():
            p.requires_grad_(False)
    if te_active:
        te_model.eval()
        for p in te_model.parameters():
            p.requires_grad_(False)
    if cov_active:
        cov_model.eval()
        for p in cov_model.parameters():
            p.requires_grad_(False)

    all_p = list(fusion.parameters())
    opt   = torch.optim.Adam(all_p, lr=cfg.phase2_lr)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.phase2_epochs, eta_min=cfg.phase2_lr * 0.1)

    # --- Precompute frozen mu's ONCE (v7 speed win + correctness, unchanged) ---
    mu_tnf_full = mu_te_full = mu_cov_full = None
    with torch.no_grad():
        if tnf_active:
            x_all = torch.tensor(tnf_feat, dtype=torch.float32).to(device)
            mu_tnf_full, _ = tnf_model.encode(x_all)
            mu_tnf_full = mu_tnf_full.cpu()
        if te_active:
            x_all = torch.tensor(te_feat, dtype=torch.float32).to(device)
            mu_te_full, _ = te_model.encode(x_all)
            mu_te_full = mu_te_full.cpu()
        if cov_active:
            x_all = torch.tensor(cov_feat, dtype=torch.float32).to(device)
            mu_cov_full, _ = cov_model.encode(x_all)
            mu_cov_full = mu_cov_full.cpu()

    n_ref = (mu_tnf_full.shape[0] if tnf_active else
             mu_te_full.shape[0]  if te_active  else
             mu_cov_full.shape[0])

    # --- FIX 9: per-contig reliability weight tensors, row-aligned with the
    # mu's above, defaulting to all-ones when a modality has no weight array.
    # (also reused for the FIX 23 gate input, below)
    w_tnf_full = torch.tensor(tnf_w, dtype=torch.float32) if (tnf_active and tnf_w is not None) \
        else (torch.ones(n_ref) if tnf_active else None)
    w_te_full = torch.tensor(te_w, dtype=torch.float32) if (te_active and te_w is not None) \
        else (torch.ones(n_ref) if te_active else None)
    w_cov_full = torch.tensor(cov_w, dtype=torch.float32) if (cov_active and cov_w is not None) \
        else (torch.ones(n_ref) if cov_active else None)

    total_w_full = torch.zeros(n_ref)
    if tnf_active: total_w_full = total_w_full + w_tnf_full
    if te_active:  total_w_full = total_w_full + w_te_full
    if cov_active: total_w_full = total_w_full + w_cov_full
    if not bool((total_w_full > 0).any()):
        raise ValueError(
            "_phase2_joint: every contig has total reliability weight 0 across all active "
            "modalities -- Phase 2's readback loss would have nothing to train on."
        )

    tensors = []
    if tnf_active: tensors.append(mu_tnf_full); tensors.append(w_tnf_full)
    if te_active:  tensors.append(mu_te_full);  tensors.append(w_te_full)
    if cov_active: tensors.append(mu_cov_full); tensors.append(w_cov_full)

    effective_batch_size = _effective_batch_size(n_ref, cfg.batch_size)
    drop_last = effective_batch_size < n_ref
    loader = DataLoader(TensorDataset(*tensors), batch_size=effective_batch_size,
                         shuffle=True, drop_last=drop_last)

    use_entropy = cfg.fusion_mode == "contig_weighted" and cfg.fusion_gate_entropy_weight > 0
    t0 = time.time()

    for epoch in range(1, cfg.phase2_epochs + 1):
        fusion.train()
        ep_tot = 0.0
        nb = 0

        for batch in loader:
            idx = 0
            mu_tnf_b = w_tnf_b = None
            if tnf_active:
                mu_tnf_b = batch[idx].to(device); idx += 1
                w_tnf_b  = batch[idx].to(device); idx += 1
            mu_te_b = w_te_b = None
            if te_active:
                mu_te_b = batch[idx].to(device); idx += 1
                w_te_b  = batch[idx].to(device); idx += 1
            mu_cov_b = w_cov_b = None
            if cov_active:
                mu_cov_b = batch[idx].to(device); idx += 1
                w_cov_b  = batch[idx].to(device); idx += 1

            opt.zero_grad()

            # FIX 23 (v8.2): pass the same per-contig reliability weights
            # into forward() that readback_loss below also uses, so the
            # contig_weighted gate can see them as an input signal.
            z_fus = fusion(z_tnf=mu_tnf_b, z_te=mu_te_b, z_cov=mu_cov_b,
                            w_tnf=w_tnf_b, w_te=w_te_b, w_cov=w_cov_b)

            l_fus = fusion.readback_loss(z_fus, mu_tnf=mu_tnf_b, mu_te=mu_te_b, mu_cov=mu_cov_b,
                                          w_tnf=w_tnf_b, w_te=w_te_b, w_cov=w_cov_b)
            total = cfg.loss_scale_fusion * l_fus

            if use_entropy:
                total = total + cfg.fusion_gate_entropy_weight * fusion.gate_entropy()

            total.backward()
            torch.nn.utils.clip_grad_norm_(all_p, 1.0)
            opt.step()

            ep_tot += total.item(); nb += 1

        if nb == 0:
            log.warning(f"Epoch {epoch}: 0 batches processed (n={n_ref}, batch_size={effective_batch_size}) -- skipping")
            continue

        sch.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.phase2_epochs:
            w   = fusion.get_weights()
            log.info(f"  Joint ep {epoch:3d}/{cfg.phase2_epochs} | recon_loss={ep_tot/nb:.4f} "
                      "| " + " ".join(f"{k}={v:.3f}" if v is not None else f"{k}=None" for k, v in w.items()))

    log.info(f"  Joint done in {time.time()-t0:.0f}s")

    # --- Final latent uses the SAME precomputed frozen mu's (no drift) ------
    fusion.eval()
    std_parts = []
    if tnf_active: std_parts.append(f"TNF={mu_tnf_full.std():.4f}")
    if te_active:  std_parts.append(f"TE={mu_te_full.std():.4f}")
    if cov_active: std_parts.append(f"COV={mu_cov_full.std():.4f}")
    log.info(f"  Post-joint std (frozen, == Phase 1, no drift): {' '.join(std_parts)}")

    with torch.no_grad():
        # FIX 23 (v8.2): final full-dataset forward pass also passes the
        # per-contig reliability weights, same as the training loop above --
        # keeps the gate's behavior consistent between training and this
        # final latent-producing call.
        final = fusion(
            z_tnf=mu_tnf_full.to(device) if tnf_active else None,
            z_te=mu_te_full.to(device) if te_active else None,
            z_cov=mu_cov_full.to(device) if cov_active else None,
            w_tnf=w_tnf_full.to(device) if tnf_active else None,
            w_te=w_te_full.to(device) if te_active else None,
            w_cov=w_cov_full.to(device) if cov_active else None,
        ).cpu().numpy()

    log.info(f"  Final latent std={final.std():.4f}")

    contig_weights_path = None
    if cfg.fusion_mode == "contig_weighted":
        cw = fusion.get_contig_weights()
        if cw is not None:
            contig_weights_path = Path(outdir) / "contig_fusion_weights.npy"
            np.save(contig_weights_path, cw)
            log.info(f"  Saved per-contig fusion gate weights: {contig_weights_path} shape={cw.shape} "
                     f"(columns={fusion._mods})")

    (ckpt_dir / "joint").mkdir(parents=True, exist_ok=True)
    np.save(lat_path, final)
    torch.save(fusion.state_dict(), fusion_path)  # FIX 14
    if ckpt:
        ckpt.mark_done("joint_fusion", {
            "_fp": fp2, "final_std": float(final.std()), "fusion_weights": fusion.get_weights(),
            "fusion_mode": cfg.fusion_mode,
            "contig_fusion_weights_path": str(contig_weights_path) if contig_weights_path else None,
        })

    # Unfreeze afterward in case caller reuses these model objects elsewhere
    if tnf_active:
        for p in tnf_model.parameters(): p.requires_grad_(True)
    if te_active:
        for p in te_model.parameters(): p.requires_grad_(True)
    if cov_active:
        for p in cov_model.parameters(): p.requires_grad_(True)

    return final


# =============================================================================
# MAIN PUBLIC API
# =============================================================================

def run_encoder(
    outdir:             str,
    tnf_features_path:  Optional[str] = None,
    cov_features_path:  Optional[str] = None,
    te_features_path:   Optional[str] = None,
    tnf_weights_path:   Optional[str] = None,
    te_weights_path:    Optional[str] = None,
    config:             Optional[EncoderConfig] = None,
    config_path:        Optional[str] = None,
    n_samples:          Optional[int] = None,
    **kwargs,
) -> str:
    """
    Full HyphaeS encoder pipeline v8.
    Returns path to final_latent.npy

    n_samples (FIX 7, now actually used): when supplied, cross-checked
    against cov_features_path's raw column count (must equal
    n_samples + COV_METADATA_DIM) before the coverage array is split.
    """
    t_start = time.time()
    if kwargs:
        log.warning(f"run_encoder: ignoring unknown kwargs: {list(kwargs.keys())}")

    if config_path is not None:
        cfg = EncoderConfig.from_yaml(config_path)
    elif config is not None:
        cfg = config
    else:
        cfg = EncoderConfig()

    # --- FIX 8: fail loud on a requested-but-missing modality unless the
    # caller explicitly opted into the old silent-disable behavior. -------
    cfg.tnf_include = _check_modality_available(cfg.tnf_include, tnf_features_path, "tnf", cfg.allow_missing_modalities)
    cfg.te_include  = _check_modality_available(cfg.te_include, te_features_path, "te", cfg.allow_missing_modalities)
    cfg.cov_include = _check_modality_available(cfg.cov_include, cov_features_path, "cov", cfg.allow_missing_modalities)

    if not (cfg.tnf_include or cfg.te_include or cfg.cov_include):
        raise ValueError(
            "run_encoder: all three modalities (tnf/te/cov) ended up disabled (missing/invalid "
            "feature paths) -- at least one usable modality is required."
        )

    log.info(f"Active modalities: tnf={cfg.tnf_include} te={cfg.te_include} cov={cfg.cov_include}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    outdir   = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device   = resolve_device(cfg.device)  # FIX 10

    if cfg.parallel_phase1 and device.type in ("cuda", "mps"):
        log.warning(f"parallel_phase1=True with device={device.type}: each Phase-1 worker "
                     f"process will initialize its own {device.type} context -- this can work, "
                     f"but may contend for the same GPU/accelerator with no speedup. Consider "
                     f"parallel_phase1=false for {device.type} runs if training looks slower "
                     f"than sequential.")

    # --- FIX 6: strict, manifest-based contig alignment (replaces v7's
    # row-count-only check). Each modality's OWN contig-ID file/manifest
    # convention is read directly -- never assumed uniform. -----------------
    log.info("Verifying contig alignment across active modalities...")
    ids_by_mod = {}
    if cfg.tnf_include:
        ids_by_mod["tnf"] = _load_modality_manifest(tnf_features_path, "tnf")
    if cfg.te_include:
        ids_by_mod["te"] = _load_modality_manifest(te_features_path, "te")
    if cfg.cov_include:
        ids_by_mod["cov"] = _load_modality_manifest(cov_features_path, "cov")

    contig_ids, contig_order_hash = _check_contig_alignment(ids_by_mod)
    n_contigs = len(contig_ids)
    log.info(f"Contig alignment OK: {n_contigs:,} contigs, order_hash={contig_order_hash}, "
             f"modalities checked={list(ids_by_mod.keys())}")

    log.info("Loading features...")

    tnf_feat = None
    if cfg.tnf_include:
        tnf_feat = validate_features(np.load(tnf_features_path), "TNF", expected_dim=TNF_EXPECTED_DIM)  # FIX 7
        if tnf_feat.shape[0] != n_contigs:
            raise ValueError(f"TNF: {tnf_feat.shape[0]} rows in {tnf_features_path} but "
                              f"{n_contigs} contigs in its own contig_ids manifest -- the "
                              f".npy file appears stale relative to its manifest.")

    te_feat = None
    if cfg.te_include:
        te_feat = validate_features(np.load(te_features_path), "TE", expected_dim=TE_EXPECTED_DIM)  # FIX 7
        if te_feat.shape[0] != n_contigs:
            raise ValueError(f"TE: {te_feat.shape[0]} rows in {te_features_path} but "
                              f"{n_contigs} contigs in its own contig_ids manifest -- the "
                              f".npy file appears stale relative to its manifest.")

    cov_feat = None
    cov_w    = None
    if cfg.cov_include:
        cov_raw = np.load(cov_features_path)
        if cov_raw.shape[0] != n_contigs:
            raise ValueError(f"COV: {cov_raw.shape[0]} rows in {cov_features_path} but "
                              f"{n_contigs} contigs in its own contig_ids manifest -- the "
                              f".npy file appears stale relative to its manifest.")
        if n_samples is not None:
            expected_cov_width = n_samples + COV_METADATA_DIM
            if cov_raw.shape[1] != expected_cov_width:
                raise ValueError(
                    f"COV: n_samples={n_samples} implies {expected_cov_width} columns "
                    f"({COV_METADATA_DIM} metadata + {n_samples} coverage columns), but "
                    f"{cov_features_path} has {cov_raw.shape[1]} columns."
                )
        cov_feat_split, cov_w = split_cov_features(cov_raw, cfg)  # FIX 5
        cov_feat = validate_features(cov_feat_split, "COV")
        _guard_nonzero_weights(cov_w, "COV")  # FIX 12b

    cfg.input_dim_tnf = tnf_feat.shape[1] if cfg.tnf_include else None
    cfg.input_dim_cov = cov_feat.shape[1] if cfg.cov_include else None
    cfg.input_dim_te  = te_feat.shape[1]  if cfg.te_include  else None
    cfg               = cfg.resolve(n_contigs)

    modality_str = cfg.modality_str()

    log.info("")
    log.info("+" + "="*66 + "+")
    log.info("|" + " "*12 + f"HYPHAES ENCODER v8.2 -- {modality_str}" + " "*max(0, 34-len(modality_str)) + "|")
    log.info("+" + "="*66 + "+")
    log.info(f"  n_contigs={n_contigs:,}  batch_size={cfg.batch_size} (auto)")
    log.info(f"  modalities: {modality_str}")
    cfg.log_summary()
    log.info("")

    tnf_w = None
    if cfg.tnf_include:
        tnf_w = (validate_weights(np.load(tnf_weights_path).squeeze(), "TNF_w")
                 if tnf_weights_path and Path(tnf_weights_path).exists() else None)
        if tnf_w is not None and len(np.atleast_1d(tnf_w)) != n_contigs:
            # FIX 17: a stale or truncated weight file must fail loudly here,
            # not later via a silent length-mismatch inside make_loader/
            # TensorDataset (which would either crash confusingly deep in
            # torch, or -- if lengths happened to coincide by accident --
            # silently misalign weight i with the wrong contig i).
            raise ValueError(f"TNF: weights file {tnf_weights_path} has "
                              f"{len(np.atleast_1d(tnf_w))} entries but there are {n_contigs} "
                              f"contigs -- refusing to use a weight array that doesn't match "
                              f"row-for-row.")
        _guard_nonzero_weights(tnf_w, "TNF")  # FIX 12b
    te_w = None
    if cfg.te_include:
        te_w = (validate_weights(np.load(te_weights_path).squeeze(), "TE_w")
                if te_weights_path and Path(te_weights_path).exists() else None)
        if te_w is not None and len(np.atleast_1d(te_w)) != n_contigs:
            raise ValueError(f"TE: weights file {te_weights_path} has "
                              f"{len(np.atleast_1d(te_w))} entries but there are {n_contigs} "
                              f"contigs -- refusing to use a weight array that doesn't match "
                              f"row-for-row.")
        _guard_nonzero_weights(te_w, "TE")  # FIX 12b

    log.info("Normalizing features...")
    tnf_norm = None
    if cfg.tnf_include:
        tnf_norm, tnf_mean, tnf_std = normalize_features(tnf_feat, "TNF")
        np.save(outdir / "tnf_norm_stats.npy", np.stack([tnf_mean, tnf_std]))

    cov_norm = None
    if cfg.cov_include:
        # FIX 16: statistics computed only over valid_mask>0 rows -- see
        # normalize_features docstring. cov_w IS valid_mask (split_cov_features
        # returns it unchanged as the reliability weight).
        cov_norm, cov_mean, cov_std = normalize_features(cov_feat, "COV", mask=cov_w)
        np.save(outdir / "cov_norm_stats.npy", np.stack([cov_mean, cov_std]))

    te_norm = None
    if cfg.te_include:
        te_norm, te_mean, te_std = normalize_features(te_feat, "TE")
        np.save(outdir / "te_norm_stats.npy", np.stack([te_mean, te_std]))

    # --- FIX 15: Phase-1 checkpoint fingerprints -----------------------------
    fp_tnf = fp_te = fp_cov = None
    if cfg.tnf_include:
        fp_tnf = _fingerprint(_file_fingerprint(tnf_features_path), _file_fingerprint(tnf_weights_path),
                               cfg.latent_dim_tnf, cfg.beta_tnf, cfg.phase1_epochs_tnf, cfg.phase1_lr,
                               cfg.kl_free_bits, cfg.kl_anneal_epochs, cfg.hidden_scale, cfg.hidden_min,
                               cfg.hidden_max, cfg.dropout, cfg.n_hidden_layers, cfg.batch_size,
                               cfg.seed, str(device), VERSION)
    if cfg.te_include:
        fp_te = _fingerprint(_file_fingerprint(te_features_path), _file_fingerprint(te_weights_path),
                              cfg.latent_dim_te, cfg.beta_te, cfg.phase1_epochs_te, cfg.phase1_lr,
                              cfg.kl_free_bits, cfg.kl_anneal_epochs, cfg.hidden_scale, cfg.hidden_min,
                              cfg.hidden_max, cfg.dropout, cfg.n_hidden_layers, cfg.batch_size,
                              cfg.seed, str(device), VERSION)
    if cfg.cov_include:
        fp_cov = _fingerprint(_file_fingerprint(cov_features_path), cfg.cov_use_n_samples_present_as_feature,
                               cfg.latent_dim_cov, cfg.beta_cov, cfg.phase1_epochs_cov, cfg.phase1_lr,
                               cfg.kl_free_bits, cfg.kl_anneal_epochs, cfg.hidden_scale, cfg.hidden_min,
                               cfg.hidden_max, cfg.dropout, cfg.n_hidden_layers, cfg.batch_size,
                               cfg.seed, str(device), VERSION)

    log.info("")
    log.info("PHASE 1: Independent encoder training")
    log.info("-" * 40)
    from hyphaesbin.utils.resource_profiler import manual_start, manual_end
    _phase1_profile = manual_start('encoder/phase1_all')
    t1 = time.time()

    tnf_model  = None
    latent_tnf = None
    cov_model  = None
    latent_cov = None
    te_model   = None
    latent_te  = None

    # (name, features, weights, latent_dim, beta, epochs, ckpt_key, fp)
    active_jobs = []
    if cfg.tnf_include:
        active_jobs.append(("tnf", tnf_norm, tnf_w, cfg.latent_dim_tnf, cfg.beta_tnf,
                             cfg.phase1_epochs_tnf, "tnf_encoder", fp_tnf))
    if cfg.cov_include:
        active_jobs.append(("cov", cov_norm, cov_w, cfg.latent_dim_cov, cfg.beta_cov,
                             cfg.phase1_epochs_cov, "cov_encoder", fp_cov))
    if cfg.te_include:
        active_jobs.append(("te", te_norm, te_w, cfg.latent_dim_te, cfg.beta_te,
                             cfg.phase1_epochs_te, "te_encoder", fp_te))

    results = {}
    want_parallel = cfg.parallel_phase1 and len(active_jobs) > 1

    if want_parallel:
        try:
            n_workers = cfg.parallel_phase1_max_workers or len(active_jobs)
            n_workers = max(1, min(n_workers, len(active_jobs), os.cpu_count() or 1))  # FIX 11
            total_threads = max(1, os.cpu_count() or 32)
            threads_per_worker = max(4, total_threads // n_workers)
            log.info(f"Phase 1: training {len(active_jobs)} modalities in parallel "
                     f"({n_workers} workers x {threads_per_worker} threads)")

            jobs = [
                (name, feat, w, ldim, beta, ep, cfg, device, ckpt_dir, ckpt_key, fp, threads_per_worker)
                for (name, feat, w, ldim, beta, ep, ckpt_key, fp) in active_jobs
            ]
            # FIX 11: "spawn", not "fork" -- fork doesn't exist on Windows and
            # is unsafe once a CUDA context may already be initialized in the
            # parent process.
            ctx = mp.get_context("spawn")
            _parent_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                    futs = {ex.submit(_phase1_worker, job): job[0] for job in jobs}
                    for fut in as_completed(futs):
                        name = futs[fut]
                        results[name] = fut.result()[1:]  # (model, latent)
            finally:
                torch.set_num_threads(_parent_threads)
        except Exception as e:
            log.warning(f"Parallel Phase 1 failed ({e}) -- falling back to sequential training. "
                        "Already-completed modalities are resumed from checkpoint automatically.")
            results = {}

    for (name, feat, w, ldim, beta, ep, ckpt_key, fp) in active_jobs:
        if name in results:
            continue
        results[name] = _phase1_single(name, feat, w, ldim, beta, ep, cfg, device, ckpt_dir, ckpt_key, fp)

    if "tnf" in results:
        tnf_model, latent_tnf = results["tnf"]
    if "cov" in results:
        cov_model, latent_cov = results["cov"]
    if "te" in results:
        te_model, latent_te = results["te"]

    std_parts = []
    if cfg.tnf_include:
        std_parts.append(f"TNF:{latent_tnf.std():.4f}")
    if cfg.te_include:
        std_parts.append(f"TE:{latent_te.std():.4f}")
    if cfg.cov_include:
        std_parts.append(f"COV:{latent_cov.std():.4f}")
    manual_end(_phase1_profile)
    log.info(f"Phase 1 complete in {time.time()-t1:.0f}s  |  Latent std -- {' '.join(std_parts)}")

    if cfg.tnf_include:
        np.save(outdir / "latent_tnf.npy", latent_tnf)
    if cfg.te_include:
        np.save(outdir / "latent_te.npy", latent_te)
    if cfg.cov_include:
        np.save(outdir / "latent_cov.npy", latent_cov)

    log.info("")
    log.info(f"PHASE 2: Joint fusion training (v8 -- frozen encoders, fusion_mode='{cfg.fusion_mode}', "
             f"per-contig weighted reconstruction-through-fusion loss)")
    log.info("-" * 40)
    t2 = time.time()

    # --- FIX 15: Phase-2 checkpoint fingerprint (depends on Phase-1's too) ---
    fp2 = _fingerprint(
        fp_tnf, fp_te, fp_cov, cfg.fusion_mode, cfg.fusion_gate_entropy_weight,
        cfg.loss_scale_fusion, cfg.phase2_epochs, cfg.phase2_lr, cfg.final_dim,
        cfg.latent_dim_tnf, cfg.latent_dim_te, cfg.latent_dim_cov,
        str(device), cfg.seed, VERSION,
    )

    fusion    = FusionModule(cfg).to(device)
    final_lat = _phase2_joint(
        fusion, cfg, device, ckpt_dir, outdir, fp2,
        tnf_model=tnf_model, tnf_feat=tnf_norm, tnf_w=tnf_w,
        te_model=te_model, te_feat=te_norm, te_w=te_w,
        cov_model=cov_model, cov_feat=cov_norm, cov_w=cov_w)

    log.info(f"Phase 2 complete in {time.time()-t2:.0f}s")

    final_path = outdir / "final_latent.npy"
    np.save(final_path, final_lat)

    save_dict = {
        "fusion":         fusion.state_dict(),
        "fusion_weights": fusion.get_weights(),
        "config":         asdict(cfg),
    }
    if cfg.tnf_include and tnf_model is not None:
        save_dict["tnf_encoder"] = tnf_model.state_dict()
    if cfg.te_include and te_model is not None:
        save_dict["te_encoder"] = te_model.state_dict()
    if cfg.cov_include and cov_model is not None:
        save_dict["cov_encoder"] = cov_model.state_dict()
    torch.save(save_dict, outdir / "encoder_weights.pt")

    with open(outdir / "encoder_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # --- Consolidated manifest (new in v8) -----------------------------------
    manifest = {
        "module": "encoder.py", "version": VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - t_start, 1),
        "modalities": {"tnf": cfg.tnf_include, "te": cfg.te_include, "cov": cfg.cov_include},
        "n_contigs": n_contigs,
        "contig_id_order_hash": contig_order_hash,
        "fusion_mode": cfg.fusion_mode,
        "device": str(device),
        "inputs": {
            "tnf_features_path": str(tnf_features_path) if cfg.tnf_include else None,
            "tnf_features_fingerprint": _file_fingerprint(tnf_features_path) if cfg.tnf_include else None,
            "te_features_path": str(te_features_path) if cfg.te_include else None,
            "te_features_fingerprint": _file_fingerprint(te_features_path) if cfg.te_include else None,
            "cov_features_path": str(cov_features_path) if cfg.cov_include else None,
            "cov_features_fingerprint": _file_fingerprint(cov_features_path) if cfg.cov_include else None,
            "tnf_weights_path": str(tnf_weights_path) if tnf_weights_path else None,
            "tnf_weights_fingerprint": _file_fingerprint(tnf_weights_path) if tnf_weights_path else None,  # FIX 18
            "te_weights_path": str(te_weights_path) if te_weights_path else None,
            "te_weights_fingerprint": _file_fingerprint(te_weights_path) if te_weights_path else None,  # FIX 18
            "n_samples": n_samples,
        },
        "outputs": {
            "final_latent_path": str(final_path), "shape": list(final_lat.shape),
            "fusion_weights": fusion.get_weights(),
            "contig_fusion_weights_path": (str(outdir / "contig_fusion_weights.npy")
                                            if cfg.fusion_mode == "contig_weighted" else None),
            "encoder_weights_path": str(outdir / "encoder_weights.pt"),
        },
        "config": asdict(cfg),
    }
    with open(outdir / "encoder_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    log.info("")
    log.info("=" * 68)
    log.info("  ENCODER v8.2 COMPLETE")
    log.info(f"  n_contigs    : {n_contigs:,}")
    log.info(f"  modalities   : {modality_str}")
    if cfg.tnf_include:
        log.info(f"  latent_tnf   : {latent_tnf.shape}  std={latent_tnf.std():.4f}")
    if cfg.te_include:
        log.info(f"  latent_te    : {latent_te.shape}   std={latent_te.std():.4f}")
    if cfg.cov_include:
        log.info(f"  latent_cov   : {latent_cov.shape}  std={latent_cov.std():.4f}")
    log.info(f"  final_latent : {final_lat.shape}  std={final_lat.std():.4f}")
    log.info(f"  fusion_mode  : {cfg.fusion_mode}")
    log.info(f"  Fusion weights: {fusion.get_weights()}")
    log.info(f"  Output: {outdir}")
    log.info("=" * 68)

    return str(final_path)

# PROFILE HOOKS v1 — observation only; scientific functions above are unchanged.
from hyphaesbin.utils.resource_profiler import profile_functions, profile_checkpoint
profile_functions(globals(), ['run_encoder', '_phase1_single', '_phase2_joint'])
_checkpoint_ok = profile_checkpoint(_checkpoint_ok)
