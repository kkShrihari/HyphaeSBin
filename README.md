# HyphaeSBin: changes

## Install (one command)

```bash
bash install.sh                 # HyphaeSBin_env + HyphaeSBin_classify, then pip install -e .
bash install.sh --with-eukcc    # also builds EukCC in its own env
conda activate HyphaeSBin_env
hyphaesbin --scaffold contigs.fa --reads S1:R1.fq.gz:R2.fq.gz ... --outdir out --config zostera_marina.yaml
```

- **Two environments.** Everything lives in HyphaeSBin_env except Tiara and Whokaryote, which get their own env.
  - Tiara 1.0.3 needs Python < 3.10, torch ~1.7 and numpy ~1.19.
  - Whokaryote 1.1.2 pins Python 3.8 and scikit-learn 1.0.2.
  - A solver check confirms they cannot share an env with the pipeline without downgrading torch and numpy.
- **No manual activation of the second env.** The pipeline finds HyphaeSBin_classify and appends it to PATH by itself.
- **Tested.** The installer was run end-to-end with micromamba. Every tool and package passed its check.
- **`auto_install: true`** (default): anything missing (tools, Python packages, the classifier env) is installed automatically. Set it to `false` to only report what is missing.
- **Harmless pip warning.** Pip prints `whokaryote requires pathlib`. pathlib is part of Python's standard library, so this can be ignored.

## 1. Preprocessing runs cleanly

- **barrnap root cause.** Your log ran `--kingdom euk`. The bioconda barrnap 1.10 only accepts `bac arc fun`, so it failed in 14 s. The pipeline then continued unmasked while still printing "final_clean (masked)".
  - The kingdom is now read from the installed build: barrnap 1.10 gets `fun`, barrnap 0.9 gets `euk`.
  - barrnap runs on parallel chunks. A failed chunk is retried once.
  - If barrnap still fails, the run stops (`barrnap_required: true`) instead of continuing unmasked.
  - The output label now states the real masking status.
  - Tested on a 28S consensus sequence: all 3,401 bp were masked.
- **Whokaryote's `predicted` column** is read directly. The "guessing the last column" warning no longer appears.
- **Reference rescue** has been removed completely (code, config keys and CLI flags).
- **MMseqs2 out-of-memory crash on smaller nodes.** MMseqs2 now gets an automatic `--split-memory-limit` (70 % of available RAM). You can override it with `mmseqs_split_memory_limit`.

## 2. Tiara/Whokaryote classification and removal

**Every prokaryote call is now removed, including `Prokaryotic_Unclassified`.**

- The default `prokaryote_removal_mode: any` means one classifier is enough. `agree` restores the old rule, where both had to agree.
- **Conflicts are no longer parked in `Unknown`.** When Tiara and Whokaryote disagree, Whokaryote's call wins, because its T model already includes Tiara. The exception: if Tiara's own probability is ≥ `classification_conflict_override` (0.90), Tiara's call wins.
- **`Unknown` now only means neither classifier made a call.** These contigs are kept by default (`unknown_removal_enabled: false`).
- **Short contigs are classified too.** `tiara_min_len` and `whokaryote_minsize` now follow `adaptive_filter_min` (1000). Before, they followed the 2 kb threshold, so 1–2 kb prokaryotic contigs were never called.

**Removal policy for each sequence category:**

| Category | Default | Why |
|---|---|---|
| Prokaryote, incl. Prokaryotic_Unclassified (plasmids are usually called prokaryotic by both tools, so they go too) | **removed** | your instruction |
| Plastid | flagged, kept | Zostera: only 6 of 1,372 plastid calls aligned to Z. marina organelles. CAMISIM: 88 % of "plastid" bp was Rhizophagus nuclear |
| Mitochondrion | flagged, kept | Same Tiara organelle classifier as plastid, so the same false-positive risk. Switch: `mitochondrial_removal_enabled` |
| rDNA | masked, never removed | rDNA is part of a eukaryotic genome; only composition is masked |

- **Caveat.** On CAMISIM, Tiara's prokaryote calls below 3 kb were mostly fungal. If you want those contigs kept and flagged instead of removed, set `prokaryote_min_length: 3000`.

## 3. Length filter and short contigs

| Key | Default | Meaning |
|---|---|---|
| `contig_filter` | 2000 | ≥ this goes to the main set and is **not** adaptive-filtered |
| `adaptive_filter_min` | 1000 | Contigs from this length up to `contig_filter` are set aside and only these are adaptive-filtered; anything shorter is removed |
| `short_contigs_in_encoder_training` | false | Short survivors also **train** the encoder |
| `short_contigs_in_clustering_rescue` | true | Short survivors are embedded (inference only unless the option above is on) and offered to recruitment |

- **Both options false:** short contigs never enter `final_clean.fasta`.
- **Files written to `preprocessing/final/`:** `short_contigs_set_aside.fasta`, `short_contigs_survivors.fasta`, `short_contig_ids.txt`, `encoder_train_ids.txt`.
- **Encoder.** It trains on `encoder_train_ids.txt` only, with normalisation statistics taken from those rows. Every contig is still embedded.

## 4. Clustering quality gate and parameter search (`hyphaesbin/clustering/quality_gate.py`)

**How stages are judged:**

- **Every stage is scored against the last accepted stage**, using your metric table (silhouette, macro silhouette, CH, DB, kNN@10/30/50).
- **Rejected stages are reverted and the run continues** with the next stage. It never stops at raw HDBSCAN.
- **Rejection rule:** any quality metric worsens by more than 10 %, or the mean worsens by more than 5 %.
- **kNN agreement is judged on its error (1 − agree).** At 0.95, a raw percentage hides real damage.
- **More assigned bp can't rescue a poorer stage.** Assigned bp only ranks candidates that already pass.
- **Rescue and recruitment are gated too.** Only the final bin-size filter is measured without being gated, because it is a biological rule.
- **Not counted as stages:** `eligible_mask` and `membership_probabilities` moved to `stage_assignments/meta/` and are excluded from all comparisons.

**Why 10 % (calibrated on your Zostera table):** benign transitions moved the worst metric by at most 2.8 %. Harmful ones moved it by at least 15.5 %:

- bin merge: CH −33.6 %, DB +28.3 %
- late rescue: silhouette −40.9 %, kNN@10 error +64.8 %

To recheck on another dataset: `python -m hyphaesbin.clustering.quality_gate calibrate table.tsv`

**Parameter search per stage:**

- **Order:** the default runs first, then a coarse grid, then one refine step around the best value.
- **Coarse grids:**
  - floats: ±0.5 (e.g. 1.0–2.0 for 1.5)
  - integers: 1, 3, d, 7, 10
  - fractions: 0–1
- **Refine steps:** ±0.125 for floats, ±1 for integers, ±0.05 for fractions.
- **Choosing:** candidates are ranked best-first and the first one that passes is kept. If none passes, the stage is rejected.
- **Hard caps:** 8 evaluations and 900 s per stage. A stage is not swept if its default run alone takes more than 300 s.
- **Fast metrics:** fixed-seed subsamples (10k rows). On 200k contigs × 37 dimensions, one evaluation takes about 3 s.
- **HDBSCAN refits** are off by default (`gate_sweep_hdbscan`) because they are expensive.

**Output files:**

| File | Contents |
|---|---|
| `stage_quality.tsv` | Your table plus the decision for each stage |
| `stage_quality_candidates.tsv` | Every candidate that was tried |
| `stage_quality_chosen_params.json` | The parameter value chosen for each stage |

## 5. Resources step by step

- **Per-stage RAM.** `resource_report.tsv/.txt` now gives each stage's RAM at start, mean, p50, p95, peak and end, along with cores, read and write.
- **Full time series.** `resource_timeline.tsv` has one row per sample (0.5 s): total RSS, cores busy, read and write MB/s, and the active stage.
- **Gate stages** appear as their own rows: `gate/<run>/<stage>`.

## 6. Runs: tetnfcov + tnfcov only, independent

- **No pooled binning and no automatic winner.** Best-run selection was removed: neither run is uniformly better and a ranking hid the trade-offs.
- **Code location.** `hyphaesbin/multirun/` holds modality_runs.py and compare_runs.py.
- **Output folder `<outdir>/results/`:**
  - `tetnfcov/` and `tnfcov/`: links to each run's bins, `bin_qc_status.tsv`, `bins_by_qc/`, `rejected_bins/`, stage tables and `gate_candidates/`.
  - `runs_comparison.tsv`: one neutral row per run (bins, Mb, QC passed/failed/unassessed, MIMAG HQ/MQ, rejected bins, recruited short contigs, gate-rejected stages).

## Verified here

- **End-to-end run on simulated data** (3 paired samples, 270 contigs), with real minimap2, CoverM, Tiara, Whokaryote and barrnap:
  - preprocessing → coverage → TNF → TE (precomputed) → encoder subset training → gated clustering → export.
  - Exit code 0.
- **Not tested here:**
  - EukCC (no database in this sandbox).
  - MMseqs2 at real scale (this sandbox has 7 GB RAM).

## 7. Fusion gate (why the weights sat at ~0.333)

**Causes:**
- **Zero start.** The gate logits start at 0, so softmax gives exactly 1/3 each.
- **Tiny steps.** Phase 2's learning rate is 5e-5, so over all of Phase 2 the logits can only drift by about 0.2 at most.
- **Redundant gate.** The gate's scale is cancelled by `proj_X`, the fusion Linear and LayerNorm, so it gets almost no gradient.
- **Entropy penalty.** In contig_weighted mode, the entropy term pulls every weight back to 1/3.

TNF still dominated because it has 32 of about 40 input dimensions. The gate never decided anything.

**Fixes (encoder config, all on by default):**
- `gate_lr`: the gate has its own learning rate (1e-2 global / 1e-3 contig).
- `fusion_gate_entropy_weight: 0`
- `gate_reliability_prior`: the gate logit gets + log(reliability) added. A contig with no TE signal therefore gets about 0 TE gate.
- `gate_balance_dims`: each modality block is scaled to equal norm, so the gate compares modalities, not dimension counts.
- `compute_modality_influence`: writes `encoder/modality_influence.json`. It holds the ablation share, i.e. how much the fused latent changes when each modality is removed.

**What to report:** use the ablation share as "modality influence" in the thesis, not the gate weight.

**Synthetic check (tnfcov):**

| | Before | After |
|---|---|---|
| Gate weights | 0.50 / 0.50 | TNF 0.59, COV 0.41 |
| Per-contig standard deviation | — | 0.22 |
| Ablation share | — | 0.58 / 0.42 |

The gate and the ablation now agree.

## 8. Review fixes (v9.1)

| Issue | Fix |
|---|---|
| Phases 1–5 reused on `is_done` only | Each phase now stores a fingerprint of its inputs plus its result-relevant config. A mismatch or a missing output re-enters the phase. Modules still resume their own unchanged sub-steps, so this is cheap. |
| Gate passed candidates whose metrics became undefined (collapse to <2 clusters or all noise) | Fails now. A metric that is defined in the reference but missing in the candidate counts as −100 %. A metric value of 0.0 is no longer treated as missing. |
| Failed runs gave exit code 0 | Exit code 1, with `FLAG:RUNS_FAILED` listing which runs failed and which finished. The other runs still complete. |
| Removed bins' old IDs collided with renumbered survivors | Final statuses are keyed by the new IDs only. Removed or merged pre-dedup bins are stored as `predup:<id>`. |
| Wheel installs lacked the CLI code | The pipeline moved to `hyphaesbin/pipeline.py`. The `hyphaesbin` command points there, and root `main.py` is a 3-line shim. |
| Best run was ranked on pre-dedup/merge metrics | New `06_final_exported` row scores the exported bins. Without EukCC, the fallback ranking is silhouette → kNN → bp, because kNN alone rewards merged bins. |
| Multi-run encoder cache keyed on the training-subset path | Now keyed on the file content hash. Clustering cache keys also carry a code-version token. |
| Custom `clusters_deduplicated_dir` broke `results/` | Folder names are read from the config. |

**Not changed:** BAM forwarding for EukCC `--links` merging. No stage produces BAMs aligned to bins, and `eukcc_merge_enabled` is false in your configs. This is a missing feature, not a regression.

## 9. Review fixes (v9.2)

| Issue | Fix |
|---|---|
| Phase-1 fingerprint ignored `--reads-dir`, and edits inside an assembly directory went unnoticed | Read paths are resolved against `--reads-dir`, and the directory is part of the fingerprint. An assembly directory is fingerprinted file by file. |
| Replacing the TE database at the same path did not invalidate phase 4 | `funTEdb` (file or directory) is now content-fingerprinted (name, size and modification time of every file). |
| NaN silhouette could win the best-run ranking | Non-finite silhouette or kNN values rank last. |
| An old `eukcc.csv` could be picked up after EukCC was disabled | Only the CSV recorded in this run's `clustering_stats.json` is used. There is no directory search anymore. |
| An existing classifier env missing Tiara/Whokaryote was not repaired | With `auto_install: true`, missing parts are reinstalled in place (base, whokaryote, tiara). Tested by deleting `tiara`. |
| Python dependencies were not declared in the wheel | `pyproject.toml` now lists them, with `faiss-cpu` as the optional extra `[faiss]`. |

## 10. Review fixes (v9.3)

| Issue | Fix |
|---|---|
| Cached clustering trusted the summary only | `multirun/verify.py` now checks every kept bin in `cluster_summary.tsv` for a non-empty FASTA, plus the assignment and stats tables. If anything is missing, the run is recomputed, in both multi-run and single-run mode. A fresh run with incomplete outputs is never marked done. Tested by deleting a bin. |
| Best-run export could report success with missing bins | The exported bins must match the winner's summary. Otherwise the status is `failed` with `FLAG:BEST_RUN_EXPORT_INCOMPLETE`, and the run exits with code 1. Tested. |
| `FunTEDB` alias missing from the TE cache check | All three spellings are now fingerprinted. |
| Parameter-search reports described the last candidate tried | If a different candidate was selected, it is re-run once so the report files on disk match the kept result. A rejected stage writes `REPORTS_FROM_REJECTED_STAGE_<stage>.txt`. |
| EukCC ranking used pre-merge scores | Best-run now reads per-final-bin quality from `cluster_quality.json`, where merged bins carry their post-merge score. It falls back to `eukcc.csv` only if that file is absent. |

## 11. Review fixes (v9.4)

| Issue | Fix |
|---|---|
| Cache validator did not require the metric files | `stage_quality.tsv` (with its `06_final_exported` row) and `cluster_quality.json` (with `per_bin_eukcc`) are now required. If either is missing, clustering is recomputed. Tested by deleting each. |
| Candidate reports were best-effort (rerun could differ) | No rerun anymore. During a parameter sweep, each candidate writes its reports (subcluster_*, binmerge_*, best_home_log) into its own folder `_gate_work/<stage>/cNN/`. Only the selected candidate's folder is copied into the clustering output, so the reports match the kept labels by construction. The temporary folders are deleted. |

## 12. Corrections in this release

### Configuration and inputs
| Problem | Fix |
|---|---|
| `adaptive_filter_min` overwrote explicit Tiara/Whokaryote/recruitment cutoffs | A key set explicitly in the yaml is never overwritten; only unset keys follow the retention floor. Classification, clustering, recruitment and retention cutoffs are independent. Starting values in the shipped yamls: `tiara_min_len 3000`, `whokaryote_minsize 5000`, `cluster_min_contig_len 2000`, retention `adaptive_filter_min 1000`. |
| Contigs below a classifier cutoff were just "Unknown" | `contig_classification.tsv` has a `call_basis` column: `below_all_classifier_cutoffs`, `below_tiara_cutoff_whokaryote_no_call`, `classifier_no_call`, `classifier_disagreement`, or the source of the call. |
| Classifications never reached clustering | Passed in single-run and multi-run mode and fingerprinted into every clustering cache. The EukCC eukaryote pre-filter now runs; if fewer than half of the contig IDs match, the table is ignored with `FLAG:CLASSIFICATION_ID_MISMATCH`. |
| `mmseqs_coverage` never reached MMseqs2 | `-c mmseqs_coverage --cov-mode mmseqs_cov_mode` is passed. Query = contig, target = TE consensus; MMseqs2 mode 1 = target coverage, mode 2 = query coverage. Default `mmseqs_cov_mode: 1` (that fraction of the TE consensus must align), because mode 2 would demand that fraction of the whole contig. |
| TE alignment length off by one; BED used 1-based starts | Length = `qend - qstart + 1` (inclusive); BED start = `start - 1` (0-based, half-open). |
| Single co-assembly stopped at cross-sample dedup | With one assembly source, dedup is skipped with a warning (set `dedup_cross_sample_only: false` to dereplicate within the assembly). |

### Caches and reproducibility
| Problem | Fix |
|---|---|
| Code corrections did not invalidate cached phases | Each phase's outer fingerprint includes a hash of its own source file(s); inner per-step caches still decide what is really recomputed. |
| Coverage steps 2–7 could reuse results built from an older upstream array | Each step's fingerprint includes a content hash of the array it consumes. |
| Mapping cache did not record which sample name belongs to which reads | The sample-name ↔ read-file association is stamped into the mapping checkpoint. An existing checkpoint is accepted once and stamped (no forced re-mapping); afterwards a rename/reorder forces re-mapping. |
| `hdbscan==unknown` | Versions come from the installed distribution; `clustering_stats.json` also records scipy/pandas/faiss, the python executable and a source hash. |

### Quality gate (`clustering/quality_gate.py`)
| Problem | Fix |
|---|---|
| kNN changes were error-scaled with a 1 % floor (0.99 → 0.95 read as "−389 %") | kNN agreement is judged as an absolute drop in percentage points (`gate_knn_max_drop_pp`, default 3). Raw before/after values and the pp change are written for every candidate. |
| Every split was rejected because fused-latent indices always penalise separating close genomes | Split stages (`gate_split_stages`) use their own rule: the pieces must stay separable in the TNF latent the split was decided in (kNN@10 drop ≤ 3 pp), no bin disappears, assigned bp fall ≤ 1 pp. Fused-latent indices are still reported. |
| "Unchanged" results looked like successful stages | Decision `accepted_no_change` with an explicit detail; a passing candidate that changes something is preferred over the unchanged fallback. |
| Rejected candidates were lost | Every candidate's labels + parameters + metrics + decision are saved in `clustering/gate_candidates/<stage>/cNN.{npy,json}`. |
| Noise % of the gate (pool incl. short contigs) contradicted the HDBSCAN log | Both are reported: `noise_pct` (pool) and `noise_pct_eligible`. |
| Identical candidates recomputed; refine step ran when the parameter had no effect | Identical labels reuse their metrics; refine is skipped when every coarse value gave identical labels. |
| Limitation | All gate metrics are internal. Validate decisions with `tools/gt_stage_report.py` (CAMISIM) or marker-gene QC. |

### Bins, QC and rejected bins
| Problem | Fix |
|---|---|
| NaN / inf EukCC scores were accepted | Rows with missing, non-finite or negative scores are ignored and the bin is reported `unassessed`. |
| No per-bin QC status | `bin_qc_status.tsv`: `passed` / `failed` / `unassessed` + reason (e.g. `eukcc_no_placement_markers`, `excluded_by_eukaryote_prefilter`, `contamination 40.0 > 10`), completeness, contamination, MIMAG class, eukaryotic-contig fraction, organelle contigs. Limits: `qc_min_completeness` 50, `qc_max_contamination` 10. `bins_by_qc/<status>/` links the exported FASTAs. Nothing is removed from the output because of QC. |
| Rejected bins disappeared | `rejected_bins/`: FASTAs of bins removed by the size filter (`size_filter/`) or skani dedup (`dedup_redundant/`), plus `rejected_bins.tsv` listing those AND every exported bin that failed or could not be assessed by EukCC. |
| Organelle contigs mixed silently into nuclear bins | `organelles/` (see section 13). Nothing is removed from the exported bins. |
| Misleading "no contig_classification supplied" EukCC message | Reports "already applied by the caller" when the pre-filter ran. |
| Reporting | `clustering_stats.json` → `bin_qc` (assessed/failed counts) and `eligibility_and_recruitment` (eligible vs binned, recruited short contigs). |
| Sub-clustering ignored mixed bins with few, long contigs | Bins ≥ `subcluster_min_bin_bp` (40 Mb) are examined as well as bins ≥ `subcluster_min_bin_contigs`. |
| Coverage log-transformed twice | Documented; kept by default because prune/merge/reassign were calibrated with it. Benchmark with `refine_cov_extra_log1p: false`. |

### Modality runs
| Problem | Fix |
|---|---|
| The "TE-free" tnfcov run received TE latents from the seed run | Strict isolation: a run never gets a latent of a modality it does not contain (stale copies are deleted), TE weights are not passed, `subcluster_te_mode` is forced `off`. |
| Best-run selection on partial EukCC results | Removed. Both runs are exported side by side (`results/runs_comparison.tsv`). An old `results/best_run/` is left untouched with a warning. |

### Encoder (documentation, no behaviour change)
- `loss_scale_tnf/te/cov` are inert: each Phase-1 sub-VAE trains alone with its own Adam optimiser, which is invariant to a constant loss factor. Changing them now logs `FLAG:INERT_SETTING`. `loss_scale_fusion` is active.
- Phase 1 uses reliability weights as a binary presence mask; Phase 2 uses the continuous value (gate input, prior, readback weighting). Zero reliability is suppressed softly (prior floor 1e-3), not hard-masked.

### What the next run recomputes
- Preprocessing re-enters (source changed). Tiara/Whokaryote rerun because the cutoffs are now honoured (3000/5000); everything after classification follows, **including read mapping** if the retained contig set changes. Keep the old cutoffs explicitly (`tiara_min_len: 1000`, `whokaryote_minsize: 1000`) to avoid this.
- TE search reruns (coverage filter + coordinate fix). Coverage steps 2–7 re-fingerprint (seconds). Encoders and clustering rerun.
- Preserve the current outputs first (copy `runs/` and `results/`).

### Benchmarks to run on CAMISIM (tools/)
1. `tools/build_ground_truth.py` — sequence-verified contig → species labels with minimap2 (unique / ambiguous / low_cov / unmapped). Use it instead of name matching.
2. `tools/gt_stage_report.py` — UMAPs, encoder kNN purity, per-stage per-bin species composition, transitions, and ground-truth scores of every saved gate candidate.
3. Compare `hdbscan_method` eom vs leaf, `recruit_enabled` on/off, `refine_cov_extra_log1p` true/false — one change at a time.

### Not fixable in the package
- The UMAP error: this package imports no UMAP. Send the traceback, `which python` and `python -c "import umap; print(umap.__file__)"` from the failing environment.

## 13. Final corrections (search, splits, benchmark tools)

### Wide coarse-to-fine parameter search (`quality_gate.py`: `SEARCH_SPACE`, `STAGE_PARAMS`)
- Only assignment-controlling parameters are searched; each has its own broad list, refine step and hard validity bounds:
  - HDBSCAN (refit, `gate_sweep_hdbscan: true`): `hdbscan_method` eom/leaf, `hdbscan_min_cluster_size` 5–100, `hdbscan_min_samples` 1–50, `hdbscan_epsilon` 0–0.8; then `hdbscan_soft_prob_min` 0–0.95.
  - Sub-clustering: `subcluster_tnf_sep_max`, `subcluster_cov_merge_tau` 0.5–5, `subcluster_min_cluster_size` 5–100, `subcluster_min_samples` 1–20, `subcluster_max_misfit` 0.05–0.5, `subcluster_min_bin_contigs` 250/500/1000/2000/4000 (scale only).
  - Merge: `binmerge_tnf_sep_max`, `binmerge_cov_tau`, `binmerge_max_misfit`, `binmerge_misfit_z`, `binmerge_min_tnf_mix`.
  - Prune `prune_shape_z`, `prune_max_frac`; reassign `reassign_max_z`, `reassign_k`; rescue `noise_local_agreement`, `noise_neighbor_max_distance`, `noise_reference_prob_min`, `noise_local_k`; recruit `recruit_agreement`, `recruit_max_dist`, `recruit_k`.
- Order (budget-safe): configured default → **coarse screen of every parameter first** (round-robin, so no parameter starves another) → refine around each parameter's best **two** regions (round-robin, nearest values first) → the combined best setting → top-2 combination grid (`gate_combo_max`, budget reserved) → any budget left continues refining. Untested values are reported as `FLAG:SEARCH_BUDGET` and in `stage_quality_chosen_params.json` (`not_tested`).
- Every attempted evaluation counts against the cap, failures included; every attempt has its own candidate id (`cNN`).
- Each stage's "off" option is always available: the unchanged result competes with every candidate.
- Dependent settings stay valid: `noise_local_min_support`, `reassign_min_votes`, `recruit_min_support` are scaled with k and never exceed it.
- The chosen value at the edge of the parameter's full search range logs `FLAG:BOUNDARY_OPTIMUM` (also in `stage_quality_chosen_params.json`).
- Caps: `gate_max_evals_per_stage` 40 / `gate_stage_time_budget_s` 2400; HDBSCAN `gate_hdbscan_max_evals` 30 / `gate_hdbscan_time_budget_s` 5400. A failing candidate is recorded and skipped.
- Change ranges with `gate_search_overrides`, e.g. `{hdbscan_min_cluster_size: [10, 20, 50, 100, 200]}`.
- Not searched (fixed policy or separate experiments): runtime settings, EukCC/skani settings, length cutoffs, minimum bin size/N50, `subcluster_te_mode`, `refine_cov_standardize`, `subcluster_best_home`, GC guards, disabled stages.
- **Runtime:** expect roughly 4–6 h of clustering per run on the CAMISIM set (plus EukCC). Lower the caps for quick tests.

### Gate rules
- Numerical tolerance (1e-9) on every threshold: an exactly 3.0 pp drop passes.
- New guard: a candidate may not lose more than `gate_max_bp_loss_pp` (2 pp) of assigned sequence. Discarding contigs can no longer be what makes a candidate look better.
- The unchanged result wins over a changing one unless the changing one scores at least as high (tie-break only).

### Splits decided per affected bin (`gate_split_per_bin: true`; `split_diagnostics.tsv`)
For each bin the split touches:
1. contigs the split left unassigned are re-attached only with support — majority of TNF neighbours in one piece, not a TNF outlier (mean neighbour distance within the 95 % quantile of the bin's own, `gate_split_reattach_dist_q`), and a coverage shape that fits the piece (`gate_split_reattach_cov_z` 3). Otherwise they stay unassigned (`gate_split_reattach: false` disables re-attachment). Mock: 5 distant outliers stay unassigned while 10 genuine leftovers are re-attached;
2. pieces intermixed in TNF space (obs/exp ≥ `gate_split_rejoin_mix`) **and** without a coverage difference are re-joined (fragments of one genome);
3. the bin's split is kept if the pieces are TNF-separable (local kNN@10 drop ≤ 3 pp) **or** coverage-distinct (shape z ≥ `gate_split_cov_z`), and ≤ `gate_split_max_bp_loss_frac` (5 %) of the bin stays unassigned.
Mock test: a mixed bin whose leaf split fragmented one species and left 40 contigs unassigned is now resolved (2 clean pieces, nothing lost); a random split of one genome is undone; two sister species with identical TNF but different coverage are split on coverage evidence. The raw split of every candidate is saved as `gate_candidates/<stage>/cNN_raw_split.npy` next to the per-bin composite `cNN.npy`.

### Organelles (`organelle_policy`)
- `organelles/organelle_contigs.tsv`: class, classifier confidence, evidence level (`high_confidence` ≥ `organelle_confirm_min_conf`, else `candidate`), final bin; `high_confidence.fasta`, `candidates.fasta`.
- `report`: bins unchanged. `separate` (set in both shipped yamls): additionally `clusters_nuclear_only/` without the high-confidence organelle contigs. `clusters_deduplicated/` is never altered.
- A classifier call is not an independent confirmation (e.g. alignment to an organelle database).

### Caches / provenance
- Legacy mapping checkpoint is accepted only after verification (CoverM depth columns carry the current sample names in order); otherwise reads are re-mapped.
- `hyphaesbin/utils/provenance.py`: package version, source hash of all code, python and package versions — logged at start, in `clustering_stats.json` and `results/provenance.json`.

### Benchmark tools
- `tools/build_ground_truth.py`: overlapping alignment intervals are merged before coverage; identity = matches / alignment-block length; species ranked by coverage × identity; `ambiguous` when a second genome explains ≥ 90 % as well.
- `tools/gt_stage_report.py`: only `unique` labels count in accuracy metrics; ambiguous / low_cov / unmapped are shown as `[status]` in the bin tables and counted in `gt_status_summary.tsv`. `--strict` requires exact contig IDs and a verified (status) table.

### Search switches
| Setting | Effect |
|---|---|
| `wide_search_enabled: false` | No parameter search anywhere: every stage runs once with the YAML values; the gate still accepts/rejects. Overrides `gate_sweep` and `gate_sweep_hdbscan`. |
| `search_hdbscan_min_cluster_size: false` / `search_hdbscan_min_samples: false` | Keep that value fixed at the YAML value; `hdbscan_method`, `hdbscan_epsilon` and all later stages are still searched. |
| `gate_search_exclude: [name, ...]` | Keep any other parameter fixed. |
| `gate_hdbscan_cache: true` | HDBSCAN refits reuse the distance/tree computations (joblib cache) when only `min_cluster_size`, method or epsilon change; identical labels; cache deleted after the search. |
Every candidate's runtime is in `stage_quality_candidates.tsv` (`seconds`).

## Re-clustering only (`--reclust`)

Re-run clustering + quality gate + EukCC + export of a finished run with a new config and/or new clustering code,
into a NEW folder. The original run is never modified; preprocessing, coverage, TNF and TE are read from it, encoders
are copied (and reused when the encoder settings are unchanged).

```bash
python main.py --reclust --outdir /home/skkumara/camisim --config camisim_7.yaml      # -> camisim/reclust/run_1, run_2, ...
python main.py --reclust --outdir /home/skkumara/camisim --reclust-name eps0           # named folder; re-using a name resumes it
```
Without `--config` the config of the original run (`<outdir>/run_config.yaml`, saved by every normal run) is used.
The new folder contains `reclust_info.json` (source run, config, every clustering setting that changed),
`logs/` (new log = original log up to clustering + the new clustering log; `logs/original_run/` = all original logs),
`resource_report_original*.{txt,json,tsv}` (+ `resource_report_original_preclustering.tsv`), the new
`resource_report.*`, `runs/<tag>/...` and `results/`.

## Checkpoints / resume

Re-running the SAME command with the SAME `--outdir` resumes. At start-up the log prints a `CHECKPOINT INVENTORY`
(completed steps per folder) and `FLAG:NO_CHECKPOINTS` if there are none (new or deleted output folder). When a step
is NOT reused the log says exactly why (`CHANGED: config hdbscan_epsilon: 0.2 -> 0.0`, a changed input file, or a
code change), for every phase, every preprocessing/coverage/TNF/TE/encoder sub-step and every multi-run encoder /
clustering cache.
