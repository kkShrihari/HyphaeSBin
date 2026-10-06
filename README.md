# HyphaeSBin

HyphaeSBin is a research workflow for assembling evidence about eukaryotic genomes in metagenomic assemblies. It combines sequence composition, sample-wise read coverage, and transposable-element (TE) similarity features to cluster assembly contigs into candidate bins. It includes preprocessing, independent modality runs, conservative bin deduplication, EukCC quality assessment, and sampled resource reporting.

This is a **fungi-oriented eukaryotic binning pipeline**, not a species-identification system and not a universal MAG quality certificate. It does not make every eukaryotic bin a fungus: protists and other eukaryotes can pass the eukaryote filters and may form bins. It does not establish species names. Treat output bins as hypotheses to evaluate with marker completeness/contamination, taxonomy, assembly quality, read support, and manual review.

> **Validation status of this source package:** this repository was assembled from the project source and updated files available in the Codex workspace. Python syntax was checked in the packaging environment. That environment lacked PyYAML and the Linux bioinformatics toolchain, so it could not execute the supplied test harness or a complete biological run. Dataset figures below are observations reported earlier in this project conversation; they are not newly reproduced by this packaged checkout. Check the exact source revision, configuration, databases, tool versions, and output logs before citing a result.

## Contents

| Path | Role |
|---|---|
| `main.py` | CLI and six-phase orchestration; reads the run YAML; supports checkpointed phase continuation and modality mode. |
| `hyphaesbin/preprocessing/preprocessing.py` | Assembly/read validation, merge, assembly statistics, length filtering, skani deduplication, Tiara/Whokaryote classification, domain policy, rRNA masking, mapping, adaptive filtering/rescue, contig scoring, coverage subsetting, QC reports. |
| `hyphaesbin/coverage/coverage.py` | Validates and aligns CoverM coverage to the assembly contig order; transforms per-sample coverage and emits k-nearest-neighbour distance features plus manifest. |
| `hyphaesbin/composition/composition/TNF_gene/tnf_gene.py` | Whole-contig tetranucleotide-frequency (TNF; k=4) feature matrix and per-contig confidence weights. |
| `hyphaesbin/composition/composition/TE_composition/te_composition.py` | TE-library search with MMseqs2 (`fast`) or RepeatMasker (`efficient`); emits TE composition features and confidence weights. |
| `hyphaesbin/encoder/encoder.py` | Separate modality VAEs, joint fusion, manifests, latent vectors, per-contig weights, and encoder checkpoints. |
| `hyphaesbin/clustering/clustering.py` | HDBSCAN, noise handling, large-bin subclustering, merging/pruning/reassignment, short-contig recruitment, bin FASTAs, skani dedup, optional EukCC. |
| `hyphaesbin/modality_runs.py` | Runs selected TNF/TE/COV combinations independently using shared upstream features. |
| `hyphaesbin/best_of_it.py` | Pools bins from chosen runs, finds cross-run duplicates, selects a representative, and can score the pooled result with EukCC. |
| `hyphaesbin/utils/checkpoint.py` | Checkpoint persistence and validation helpers used by pipeline stages. |
| `hyphaesbin/utils/logger.py` | Shared log setup and step/report helpers. |
| `hyphaesbin/utils/deps.py` | Reports Python packages, external tools, and available compute backends; it is diagnostic, not an installer. |
| `hyphaesbin/utils/resource_profiler.py` | Samples process-tree CPU/RAM/I/O and GPU information where supported and writes resource reports. |
| `environment.yaml` | Conda environment for the pipeline and its external bioinformatics executables. |
| `config.example.yaml` | Example research configuration with all 225 active top-level settings represented, including inherited defaults. Replace database/input paths and calibrate thresholds. |
| `docs/PARAMETERS.md` | Parameter inventory, operational interpretation, threshold interactions, and pointers to each authoritative implementation. |
| `tests/` | Packaged mocked multi-run test harness and fake `skani` executable; not a biological end-to-end validation suite. |

The repository intentionally excludes FASTA/FASTQ inputs, mapping BAMs, checkpoints, run outputs, and databases. They are large, data-specific, and may have separate access/licensing conditions. No license was included in the supplied source; add an explicit license before inviting reuse.

## What it does, in order

The top-level run has six phases. Phase numbers in `main.py` are distinct from preprocessing's displayed steps.

1. **Preprocessing** creates clean masked and unmasked FASTA variants, a coverage table, classification/QC tables, and detailed decisions.
2. **Coverage features** normalize each sample separately, calculate contig-neighbour distances, and write an `N + 4`-column coverage array for `N` samples. The four leading fields are metadata: validity mask, number of samples with nonzero coverage, mean neighbour distance, and standard deviation of neighbour distance; sample coverage features follow. The manifest states the exact column roles and contig order.
3. **TNF features** describe the relative frequencies of the 136 possible reverse-complement-collapsed 4-mers (k=4 is fixed in the implementation). This is a compositional signal, not taxonomy by itself.
4. **TE features** summarize evidence from a supplied TE reference library. TE similarity can help distinguish fungal genomes, but it depends on the library and may be sparse, incomplete, or non-specific. The pipeline also emits confidence/reliability weights.
5. **Encoder** trains active modality-specific beta-VAEs and a fusion model, then writes a latent representation for each contig. Feature manifests, contig IDs, and weights are checked for alignment. The chosen modality set determines what enters the fusion; no single latent-space metric proves bins are biologically correct.
6. **Clustering** uses configured HDBSCAN and post-processing, writes bins and assignments, deduplicates similar bins, and optionally runs EukCC. In multi-run mode, each modality combination gets its own encoder and clustering directory. Best-of pooling is an additional cross-run result.

### Preprocessing's 13 displayed steps

The current preprocessing order is:

1. Validate assembly, samples table, reads, tools, and configuration.
2. Merge/standardize assemblies and preserve the contig/source-sample mapping.
3. Compute assembly statistics.
4. Apply the global minimum-length prefilter (`min_contig_length`).
5. Run cross-sample skani deduplication on eligible contigs.
6. Classify contigs with Tiara and Whokaryote; this labels, but does not remove.
7. Apply the explicit domain-removal policy and write removal/retention audits.
8. Call rRNA genes with Barrnap and create masked plus unmasked FASTAs.
9. Map each sample's reads once with minimap2/samtools/CoverM; derive depth and covered-fraction columns.
10. Apply coverage/breadth filtering and optional reference rescue.
11. Produce multi-signal contig decisions and confidence/uncertainty flags.
12. Write the coverage table for the retained contig subset without remapping.
13. Write QC summaries, decision tables, provenance, and final FASTAs.

There is legacy naming inside the preprocessing code: the length-prefilter function is internally named `step5_length_prefilter`, and dedup is internally `step4_skani_dedup`, even though the executed/displayed order is length filter first, dedup second. The authoritative executed order is the `run_preprocessing()` call sequence and the numbered live-step messages. This naming mismatch is a maintenance hazard; do not infer execution order from internal metadata key names alone.

### Masked versus unmasked sequence

Barrnap's rRNA coordinates are replaced by `N` in the masked copy for composition-sensitive operations. The unmasked copy preserves the original nucleotide sequence. Read mapping and final bin FASTA writing should use unmasked sequence so rRNA bases are not lost from the biological bins or coverage evidence. TNF and selected QC signals can use the masked copy to reduce the influence of conserved rRNA k-mers. TE uses unmasked sequence. The output manifests and main phase routing should be followed rather than substituting FASTAs manually.

## Data, organisms, and prior project observations

### CAMISIM simulation

The project used a CAMISIM-derived simulated community with per-sample assemblies/reads and known organism membership for benchmarking. The command shown earlier used sample IDs `s1`, `s3`, `s4`, `s5`, `s6`, `s7`, and `s8`. The reported community included fungi, protists, bacteria, and archaea, which makes it useful for checking whether a fungi-oriented workflow keeps eukaryotes together and what happens to prokaryotes. The raw simulation inputs, ground-truth files, and complete run output are not included in this repository, so another user cannot reproduce the quoted numbers from this Git checkout alone.

Earlier in the conversation, the reported CAMISIM summary was approximately **1,253 Mb of fungal sequence and 80 Mb of prokaryotic sequence** in the material being clustered. Reported prokaryote contribution was about 6% of the combined sequence. Most bacterial/archaeal sequence formed separate pure bins (reported as `bin_001`–`bin_007`). One reported leak was the archaeon *Nitrososphaera* in `bin_032` with *Magnaporthe*; the combined bin was about 2.3 Mb, or about 0.2% of the eukaryote-bin bp. A separate earlier fungi-only evaluation reported **80.7% correct, 1.8% wrong, and 17.5% noise**. These percentages use the earlier project's evaluation labels/definition; they are not EukCC completeness percentages and should not be presented without that evaluation method.

Another previously reported controlled CAMISIM replay compared a conservative prokaryote rule (Tiara and Whokaryote agree, contig length at least 3,000 bp). It was reported to remove 98.7% of bacterial sequence and 82.8% of archaeal sequence, while removing about 0.3% of fungal bp and 0.2% of protist bp; about 17% of archaeal bp remained. A separate audit reported that some calls below 3 kb were predominantly fungal (about 2.5 Mb fungal versus 0.04 Mb bacterial), motivating the length floor. These are prior reported benchmark observations, not guarantees for other communities.

Names appearing in the earlier benchmark and clustering notes include fungi such as *Tuber melanosporum*, *Melampsora larici-populina*, *Cenococcum geophilum*, *Rhizophagus irregularis*, *Cryptococcus wieringae*, *Neurospora crassa*, *Ustilago maydis*, *Aspergillus niger*, *Candida albicans*, *Yarrowia lipolytica*, *Schizosaccharomyces pombe*, *Leptosphaeria*, *Blumeria*, and *Magnaporthe*. *Phytophthora* also appears in clustering notes; it is an oomycete eukaryote, not a true fungus. These names document benchmark/debug examples, not a complete confirmed taxon list for every CAMISIM or Zostera run. The original reference genomes, exact CAMISIM config/seed, species-abundance truth tables, and per-bin truth labels need to be archived alongside a formal benchmark.

### Zostera marina metagenome

The project also discussed the *Zostera marina* dataset, including reads `SRR29999183`, `SRR29999184`, and `SRR29999185` and an assembly. It is a real environmental/metagenomic case rather than the same data as CAMISIM. Earlier reports said prokaryotes made up about 81% of bp entering clustering; most eukaryotic bins had about 1–3% prokaryotic bp, while two bins (`bin_023`, `bin_029`) were reported around 20% and 32%. A prior scoring audit said EukCC had been run on 162 prokaryotic bins; one example (`bin_108`) showed EukCC completeness around 30% but BUSCO around 3.9%. That illustrates why a eukaryote marker workflow must not be used to assess a predominantly prokaryotic bin. Those results came from an earlier run/report and require verification against current output tables before publication.

The Tiara outputs quoted in the conversation were from different runs. An earlier Zostera run reported 206,505 contigs, with approximately 92% eukarya. A later screenshot was explicitly a CAMISIM classification run and showed different totals. They are not before/after measurements on the same input. The parser fixes below can change labels by correctly preserving the first-stage domain call; classification summaries must always be tied to the exact FASTA and output path.

### What organism classes are handled?

- **Fungi:** the intended main target. TNF, coverage, TE evidence, clustering, and EukCC can all contribute, but none guarantees a species-pure bin.
- **Protists:** are eukaryotes. Tiara/Whokaryote domain classification does not generally mean “fungus,” so protist contigs can be retained and can cluster with or apart from fungi. Review with taxonomic evidence and the simulation truth where available.
- **Bacteria and archaea:** removable only under the conservative agreement and length rule described below. Uncertain/single-classifier/short calls are retained for safety and audited.
- **Mitochondria and plastids:** recognized as organelle labels. By default they are flagged and retained because nuclear eukaryotic contigs can receive false organelle calls. Explicitly enabling organelle removal changes that policy and should be done only after target-specific validation.
- **Unknown/conflict:** retained. Conflicting eukaryote/prokaryote predictions are represented as a conflict/unknown decision rather than being forced into a domain.

## Domain filtering and contig-length policy

### What happens to prokaryotes

The packaged preprocessing implementation defaults to `prokaryote_removal_enabled: true` and `prokaryote_min_length: 3000`. A bacterial/archaeal/prokaryotic contig is removed only when the merged classification source is `whokaryote+tiara` (both tools support a prokaryotic call) and the contig is at least the configured minimum length. A Tiara-only call, Whokaryote-only unclassified call, unknown call, or domain conflict does not meet that removal rule and remains in the FASTA. This conservative choice avoids allowing a single imperfect classifier to delete eukaryotic sequence.

When prokaryote removal is enabled, a failed or malformed Whokaryote result must not be treated as successful evidence that the assembly is eukaryotic. Step 6 validates the expected prediction output and fails loudly rather than silently taking a Tiara-only removal path. Existing classifier outputs are reusable only with a matching fingerprint (input, key parameters, and best-effort tool versions), unless the operator explicitly forces reuse; forced reuse is recorded as unverified provenance.

Inspect `06_classification/contig_classification.tsv`, `07_domain_removal/retained_domain_calls.tsv`, removed-ID files, and the step-13 decision/QC outputs to see exactly what was called, removed, or kept and why. In particular, “prokaryote removal enabled” does **not** mean all prokaryotic sequence is removed; short or single-classifier calls intentionally survive.

### What happens to organelles

`organelle_removal_enabled` is false in the example and in preprocessing defaults. Plastid and mitochondrial calls are therefore flagged in `retained_domain_calls.tsv`, not deleted. If changed to true, high-confidence organelle calls can be removed under the confidence/source policy. The project has evidence for caution: an earlier audit reported 1,050 plastid-called contigs totaling 4.76 Mb in a CAMISIM/Rhizophagus analysis, with about 88% attributed to *Rhizophagus* nuclear sequence; a Zostera audit reported only 6 of 1,372 plastid calls aligned to the *Z. marina* organelles. Those figures are dataset/run-specific but show why the conservative default preserves organelle calls for inspection. If organelle sequence is unwanted downstream, remove/flag it with a target-specific, auditable rule and compare nuclear sequence retention before enabling deletion.

### Why 1–2 kb contigs are not automatically discarded

In the example, `min_contig_length: 1000` is the early hard floor; sequences below 1 kb are removed at preprocessing. A sequence between 1 and 2 kb is still retained through that floor. The example's Tiara and Whokaryote minimums are 2 kb, so many 1–2 kb contigs will have no classifier evidence; no-call by itself is not a deletion rule. TNF's `min_contig_len` is 1 kb, and coverage/TE features can be computed subject to each module's own data/weight validity rules.

At clustering time, `cluster_min_contig_len: 2000` excludes contigs shorter than 2 kb from the initial HDBSCAN clustering set. With `recruit_enabled: true` and `recruit_min_len: 1000`, those 1–2 kb contigs can be recruited later if enough nearby, already-binned neighbours agree and the distance/support thresholds pass. They can also remain unbinned; the pipeline does not force every short contig into a bin. This is how the example can preserve short sequences for feature calculation and later recruitment while restricting the initial clustering fit to longer contigs. A threshold such as `subcluster_min_bin_contigs: 2000` is a **number of contigs**, not 2,000 bp; `min_bin_length_bp` and `subcluster_min_part_bp` are base-pair thresholds.

There is no global 200 kb contig cutoff in the shown example. Long contigs remain available to the main pipeline subject to the same preprocessing decisions. However, a tool-specific setting can affect a feature branch: `mmseqs_max_seq_len: 100000` is passed to the fast TE search and can limit which long sequences MMseqs2 processes; it does not trim the assembly FASTA for mapping or bin output. `te_frag` is a RepeatMasker fragment-size setting for efficient mode, not a global contig-length filter. Check the TE module logs and feature manifest to confirm how long contigs were treated by the selected TE search mode.

The final bin filters still matter. `min_bin_length_bp: 500000` can exclude an entire small or highly fragmented candidate bin even if its contigs were recruited. The user should quantify the contigs/bp affected by each gate in the reports rather than assume “retained upstream” means “included in a final bin.”

### Single-sample fungi and same-sample depth/breadth

`adaptive_cov_stat: sample` is designed not to dilute a genome present in one read sample by averaging it with empty samples. It pairs each sample's depth and breadth column and tests them together. A contig supported by one sample can pass the gray-zone ordinary route only if its same-sample depth and breadth meet the configured thresholds **and** its number of samples with nonzero coverage meets `adaptive_min_samples`. With `adaptive_min_samples: 2`, a gray-zone contig seen in only one sample goes through the single-sample rescue branch instead of the ordinary multi-sample route.

If rescue is enabled but `rescue_reference_fasta` is empty, a gray-zone single-sample contig with enough depth is retained and flagged as lacking a rescue reference; it is not discarded solely because the reference was omitted. If a reference is supplied, eligible candidates are aligned in one batched minimap2 call and must meet identity and query-coverage thresholds. A novel fungus absent from that reference can fail this similarity rescue, so reference rescue is not a substitute for coverage evidence and may introduce reference bias. Contigs at or above the configured gray-zone ceiling are kept on length grounds, with low breadth recorded as a warning signal. Always inspect the step-10 reason table and step-11 decisions.

For `adaptive_cov_stat: sample`, when CoverM provides breadth columns, the depth and covered-fraction header names must pair by sample name (for example `<sample> Mean` with `<sample> Covered Fraction`). If present columns fail to pair, the code stops with the observed columns; it no longer silently falls back to separate maxima, which could combine depth from one sample with breadth from another. If no breadth columns are recognized at all, the current implementation logs a warning and disables the breadth gate for that run; review the header and treat the filtering result cautiously. Fix upstream header parsing or explicitly choose a different statistic only when that behavior is intended.

## Databases and external data dependencies

No database is distributed in this repository.

| Resource | Used for | Configuration / behavior |
|---|---|---|
| TE reference library (for example the project's MycoMobilome-derived FASTA) | TE search and feature encoding. | `funTEdb` must point to an existing FASTA or supported library directory. The example contains a placeholder only. Validate database provenance, license, version, and compatible headers. |
| EukCC eukaryotic marker database | Completeness/contamination estimates for eukaryotic bins and optional pooled best-of scoring. | `eukcc_db`; the discussed workstation used an `eukcc2_db_ver_1.1` database. Enable `run_eukcc` only after pointing to the real directory and matching it to the installed EukCC version. |
| Optional fungal rescue reference FASTA | Similarity check for gray-zone, single-sample contigs in preprocessing step 10. | `rescue_reference_fasta`; empty means no alignment is performed and qualifying gray-zone single-sample contigs are kept with a warning. No rescue reference is bundled. |
| Barrnap rRNA models | rRNA identification for masking. | Built into the installed Barrnap package; `barrnap_kingdom` defaults to `fun`. Check `barrnap --help` on the target environment because accepted values depend on the build. |
| Input assembly and paired reads | Source sequences and sample coverage. | Supplied on the command line. Keep a sample manifest and checksums. Do not include human-readable headers with whitespace if downstream tool behavior truncates IDs; preserve unique stable IDs. |

EukCC and BUSCO assess marker-gene completeness/duplication/contamination for a selected lineage; neither identifies the species. Species-level assignment requires a separate taxonomic workflow and appropriate references. For eukaryotic bins, validate with eukaryote-appropriate marker/phylogenetic evidence and assembly/read checks. A bacterial database/classifier or prokaryote-only marker set is not an eukaryotic taxonomy method. Taxonomic labels should be reported with reference/database versions and confidence, not inferred from a high completeness score.

A defensible post-binning identification sequence is: (1) classify each bin's domain before selecting marker sets; (2) run EukCC and/or an appropriately selected BUSCO lineage to describe expected-marker recovery and duplication; (3) extract/compare conserved eukaryotic proteins or taxonomic markers against a curated fungal or protist reference collection; (4) confirm a proposed species with a phylogenetic placement and, where a close genome reference exists, genome-wide similarity and synteny/coverage evidence; and (5) inspect whether the bin's contigs have coherent coverage across samples and read support. Fungal ITS/LSU/SSU evidence can help place fungi but can be multicopy, absent from some contigs, or unresolved among close species; it is not a substitute for whole-genome comparison. Protists need clade-appropriate markers and references. Keep a separate “unclassified/ambiguous” outcome instead of forcing a species name when evidence is weak. Report marker set, reference database release, thresholds, and software versions. BUSCO is not installed or run by this pipeline.

## Install and access the tool

The source is ordinary Python code under `hyphaesbin/`; there is no separate desktop application UI. Use it from a Linux shell with conda/mamba and the listed bioinformatics programs. The intended access pattern is a Git checkout, conda environment, YAML configuration, and a command-line run.

```bash
git clone https://github.com/kkShrihari/HyphaeSBin.git
cd HyphaeSBin
conda env create -f environment.yaml
conda activate hyphaesbin
python -c "from hyphaesbin.utils.deps import check_all, format_report; print(format_report(check_all()))"
python main.py --help
```

The repository URL above is the expected form if the owner creates a repository with that name; the account profile alone does not prove that this repository exists. If the repository name differs, clone that URL instead. To invoke the diagnostic functions explicitly, use `python -c "from hyphaesbin.utils.deps import check_all, format_report; print(format_report(check_all()))"`; main also runs dependency checks and reports them at startup. After installing, confirm the actual executables and versions (`minimap2`, `samtools`, `seqkit`, `skani`, `coverm`, `tiara`, `whokaryote.py`, `barrnap`, `mmseqs`, and, for efficient TE mode, `RepeatMasker`). EukCC is configured in a separate environment when needed. `deps.py` reports availability and does not install software. The `auto_install_tools` setting should normally remain false for reproducible runs.

The environment file lists Python libraries and external executables. The CPU environment is the safer baseline; a CUDA/RAPIDS setup is separate because compatible driver, CUDA, PyTorch, FAISS, and RAPIDS versions depend on the host. The encoder's `device` and clustering's `clustering_backend` are independent choices: using a GPU for the encoder does not imply GPU clustering, or vice versa. A CUDA warning on a CPU-only host may indicate fallback rather than pipeline failure, but verify the chosen backend in logs and resource reports.

## Inputs and command examples

The main interface requires `--scaffold` and `--outdir`. Repeat `--reads` once per sample as `NAME:R1` or `NAME:R1:R2`; `--reads-dir` is prepended to relative read paths. Use absolute paths or carefully controlled working directories for production.

```bash
python main.py \
  --scaffold /data/project/assembly/scaffolds.fasta \
  --reads "SRR29999183:/data/reads/SRR29999183_R1.fastq.gz:/data/reads/SRR29999183_R2.fastq.gz" \
  --reads "SRR29999184:/data/reads/SRR29999184_R1.fastq.gz:/data/reads/SRR29999184_R2.fastq.gz" \
  --reads "SRR29999185:/data/reads/SRR29999185_R1.fastq.gz:/data/reads/SRR29999185_R2.fastq.gz" \
  --outdir /data/results/zostera_run \
  --config config.example.yaml \
  --threads 32
```

For CAMISIM, pass the assembly directory and each sample's paired read paths as repeated `--reads` flags; the project example used `s1`, `s3`, `s4`, `s5`, `s6`, `s7`, and `s8`. Do not reuse Zostera paths/config values for CAMISIM or vice versa. In particular, cross-sample deduplication needs sample identity from per-sample FASTA files or an `assembly_sample_regex` that extracts one sample label from **every** contig header in a merged FASTA. One monolithic FASTA without sample-labelled headers is one source sample from the pipeline's perspective; with cross-sample-only dedup enabled it must stop instead of pretending to compare distinct samples. Do not disable cross-sample-only behavior casually: that can deduplicate within-sample repeats/paralogs.

Useful command-line controls in `main.py` include `--threads`, `--reset`, `--skip-to-phase`, and precomputed-TE inputs. Check `--help` in the exact checkout. `--reset` applies to main's top-level checkpoint store; it does not necessarily erase every module's own checkpoints. To guarantee a clean recomputation, use a new output directory or remove the relevant stage's output/checkpoint subtree only after preserving anything needed.

## Modality combinations and independent results

The tags mean:

| Tag | TNF | TE | Coverage |
|---|:---:|:---:|:---:|
| `te` |  | ✓ |  |
| `tnf` | ✓ |  |  |
| `cov` |  |  | ✓ |
| `tetnf` | ✓ | ✓ |  |
| `tecov` |  | ✓ | ✓ |
| `tnfcov` | ✓ |  | ✓ |
| `tetnfcov` | ✓ | ✓ | ✓ |

`all_runs: true` requests all seven. `te_runs: true` selects the TE-containing set (`tetnfcov`, `tetnf`, `te`). `modality_runs` adds explicit tags. The shared preprocessing and feature matrices are built once; each selected tag has independent encoder and clustering results in `runs/<tag>/`. Independent run outputs should remain available for comparison.

The project's prior requested comparison was four TE combinations (`te`, `tecov`, `tetnf`, `tetnfcov`), with best-of pooling restricted to `tetnfcov` and `tnfcov`. A YAML for that design needs explicit `modality_runs: [te, tecov, tetnf, tetnfcov]` and `best_of_runs: [tetnfcov, tnfcov]`; because `tnfcov` is not among those four, it must also be run (for example by adding it to `modality_runs`) or it will not exist to pool. This is an important design consistency check. `te_runs: true` alone does not select all four TE-containing tags because `tecov` is an explicit additional combination.

`best_of_it: true` runs a separate pooled stage. It uses skani aligned fraction and ANI thresholds (optionally overridden by `best_of_min_af` and `best_of_ani`) to make duplicate relationships between bins from different runs. In `contained` mode, a partial copy inside a larger bin can be treated as a duplicate; `both` requires both aligned fractions. For duplicate candidates with EukCC scores, the implemented quality score is completeness minus five times contamination; when scores are unavailable within a duplicate neighbourhood, the implementation falls back to N50 and total length and records that mode. Unique bins are retained without requiring EukCC comparison. The selected pooled set is then optionally run through EukCC. Pooling does not combine all unique contigs into a single bin and does not edit each source run. Ensure EukCC and its database are configured before relying on marker-based best-copy ranking.

## Reading outputs and scoring bins

Start with the output logs and manifests, not only a `bin_*.fasta` directory. Review:

- `preprocessing/06_classification/contig_classification.tsv`: Tiara labels, Whokaryote labels, conflict/source, confidence type, and sequence length.
- `preprocessing/07_domain_removal/retained_domain_calls.tsv`: retained prokaryote/organelle calls and reasons; `removed_*_ids.txt`: removals by category.
- `preprocessing/10_adaptive_filter/` and step-11/13 QC tables: coverage decisions, single-sample rescue/flags, final keep/remove rationale, and summary counts.
- `preprocessing/09_mapping/coverage_table.tsv`: per-sample depth/covered fractions used by filtering and coverage features; BAMs may consume substantial space if `keep_bams: true`.
- Coverage/TNF/TE manifests and feature arrays: input fingerprints, contig order, feature columns, valid masks, and per-contig reliability weights.
- `encoder/encoder_manifest.json`, latent arrays, and checkpoint metadata: active modalities, dimensions, weights, feature fingerprints, and training settings.
- `clustering/cluster_assignments.tsv`, `cluster_summary.tsv`, `clustering_stats.json`, `clusters_non_deduplicated/`, and `clusters_deduplicated/`: contig assignments, bin statistics, dropped/recruited/noise sequences, and output FASTAs.
- EukCC output (when enabled): completeness/contamination/lineage calls tied to bin content and the configured EukCC database.
- `resource_report.json`, `.tsv`, and `.md`: stage resource/status/cache measurements when profiling is enabled.

Use BUSCO or EukCC for marker completeness/duplication, read recruitment and coverage consistency for support, assembly statistics for continuity, and a suitable eukaryotic taxonomic/phylogenetic analysis for identity. EukCC scores on prokaryotic bins are not meaningful eukaryotic completeness scores. A high score does not prove the correct species, ploidy, or absence of all contamination. Treat bins above a selected prokaryotic fraction as prokaryotic/other and exclude them from eukaryotic marker summaries; define that cutoff in the study methods rather than silently mixing domains.

## Resource reporting and bottleneck analysis

With `resource_profile_enabled: true`, the profiler samples the running process tree and emits JSON, TSV, and Markdown reports. The rows are intended to separate top-level phases from internal work such as Tiara, Whokaryote, per-sample mapping, each encoder modality/run, each clustering run, and pooled skani/EukCC. Use wall time to find the expensive stage; CPU use can distinguish computation from waiting; sampled RAM and GPU usage support capacity planning; read/write counters and free-space change can expose I/O pressure. The pipeline reports contig/bp and throughput only when a stage can provide those counts without an extra expensive scan.

Measurements are sampled, not kernel-level exact peaks. A very short-lived memory peak can be missed; process-tree RSS may have platform-specific limitations; GPU utilization can include other jobs; disk/free-space deltas can include unrelated writers. The optional system wrapper `/usr/bin/time -v` can capture a Linux process-level wall/RSS summary, but it may not aggregate concurrent children as a scheduler's job-level accounting would. Preserve the resource report, scheduler job ID, and exact command together.

## Restart, checkpoints, and reproducibility

Each stage has its own checkpoint behavior. The current code uses input/config/logic fingerprints in many checkpoints and checks referenced outputs before reuse. Some fingerprints are size/mtime-based rather than content SHA-256; a same-size file whose timestamp is deliberately preserved can defeat such a check. Keep immutable input paths, record checksums externally, and prefer a new output directory for a fundamentally changed input or workflow. Classification reuse has both verified and explicitly forced modes; the latter must be treated as unverified provenance.

For every published run, archive the exact YAML, command line, Git commit, input checksums, FASTA/read sample map, environment export, external-tool versions, TE/EukCC/rescue database versions, scheduler resources, logs, QC tables, and generated resource reports. Run a small end-to-end fixture on the target Linux environment before a multi-hour production run. Do not delete cached outputs to “fix” a problem until identifying which checkpoint owns them.

## Configuration guide

The example YAML includes 225 active user-facing settings, including options that otherwise inherit code defaults. It is a project-style baseline, not a universally recommended configuration. In particular, its TNF latent dimension, coverage/device choices, fusion mode, clustering profile, 1 kb preprocessing floor, 2 kb classifier/initial-clustering floors, 3 kb agreement-based prokaryote removal floor, and 500 kb final bin-length threshold are deliberate values that need study-specific justification. `docs/PARAMETERS.md` catalogs each key; comments in `config.example.yaml` explain common interactions; module dataclasses/default dictionaries define exact accepted values and implementation behavior.

`analysis_min_length` is a master convenience override when set to a positive integer: it sets `dedup_min_length`, `cluster_min_contig_len`, `tiara_min_len`, `whokaryote_minsize`, and `adaptive_gray_ceiling` together. It does **not** change the raw `min_contig_length` floor, the TNF/TE feature floor `min_contig_len`, the short-contig recruitment floor `recruit_min_len`, or `subcluster_min_bin_contigs` (a count). Values below the raw floor fail. Leave it null to respect each explicit setting. This distinction is important if testing whether 1–2 kb contigs contribute to final bins.

### Changes addressing previously reported problems

This package incorporates the following behaviors visible in the current source. They are implementation claims, not a declaration that all cases have been validated against every tool version:

1. **Tiara stage parsing:** preserve the first-stage domain call for ordinary contigs; use the second stage for the organelle subtype when Tiara reports an organelle. Literal `n/a` is kept as text instead of being coerced to a missing value. `mitochondrion` is recognized as mitochondrial.
2. **Whokaryote schema:** search named label columns including `predicted`; only fall back positionally with a warning when non-strict. Prokaryote removal enables strict parsing and rejects malformed output.
3. **Classification reuse/failure:** reuse is fingerprint-checked where a stamp exists; forced reuse is marked unverified. The `tools.fp` stamp is written after predictions validate. If Whokaryote fails or produces malformed output while prokaryote removal is enabled, preprocessing stops rather than quietly claiming the prokaryotes are absent.
4. **Prokaryote decision rule:** remove only matching Tiara+Whokaryote prokaryote calls at/above the minimum length (3 kb in the example). Short, single-tool, unknown, and conflicting calls are retained and have reasons recorded.
5. **Organelle safeguards:** organelles are retained/flagged by default. This addresses a reported Rhizophagus nuclear-sequence false-plastid loss; do not enable deletion without target-specific validation.
6. **Step 10 coverage pairing:** `adaptive_cov_stat: sample` requires sample-matched depth and covered-fraction columns and stops if they do not pair. It does not silently restore separate maxima. It tests paired depth+breadth so coverage from sample A cannot combine with breadth from sample B.
7. **Single-sample rescue:** qualifying one-sample gray-zone contigs are either tested against a configured reference in a batched alignment or retained/flagged when rescue is requested but no reference exists. Reference rescue can miss novel lineages.
8. **One removal policy:** step 11 uses the same domain-removal decision as step 7 so it cannot independently discard a call that step 7 retained.
9. **Checkpoints and provenance:** stage logic/input fingerprints, output-existence checks, classifier provenance, and resource status reporting reduce stale-cache reuse and make decisions easier to audit. Fingerprints are not all content hashes; record external checksums too.
10. **Barrnap and source completeness:** the example uses `barrnap_kingdom: fun`; verify the target build's accepted values. The packaged repository includes `logger.py` and `deps.py` as well as the updated modules; a resource-only overlay is not a complete checkout.
11. **Stage labels:** preprocessing's historical internal function names for length filtering and dedup are inverted relative to their displayed/executed order. Actual order is length filter then dedup; follow live logs and the `run_preprocessing()` call sequence.
12. **Earlier tests and replay counts:** past project messages quoted 86 preprocessing behavior checks, 20 integration checks, and other mocked test counts. Those full test sources/results are not included here, so this README does not treat those counts as independently verified. The included test harness only addresses mocked multirun behavior.

Key groups and decisions:

- **General / input:** `threads`, `seed`, `min_contig_length`, `read_type`, `read_type_detect_seed`. The command-line `--threads` overrides the YAML top-level thread count.
- **Dedup and source samples:** ANI, aligned fraction, comparison length, cross-sample-only policy, contig cap, and `assembly_sample_regex`. Cross-sample mode needs reliable sample-of-origin labels.
- **Classification/domain policy:** Tiara length/probability, Whokaryote length/model, high-probability bar, prokaryote enable/minimum length, and organelle policy. These determine evidence and removal separately.
- **Mapping/adaptive filter:** mapping worker/thread split, depth, breadth, number of samples, same-sample statistic, gray-zone ceiling, BAM retention, and optional rescue reference thresholds.
- **Coverage:** kNN neighbour count, core count, numeric validation, outlier percentile, reproducibility switch, fallback cap, metadata layout.
- **TE database/search:** library path, search mode, MMseqs2 or RepeatMasker thresholds, feature confidence/weight coefficients, and warning fraction.
- **TNF:** minimum contig length, valid k-mer count, full-confidence k-mer count, weight mode. k=4 is fixed.
- **Encoder:** modality toggles; per-modality latent dimensions; architecture; beta/KL annealing; phase epochs and learning rates; loss scales; gate; device; batch size; parallelism. Treat these as model hyperparameters and preserve them across comparisons.
- **Clustering:** HDBSCAN density thresholds/method; rescue; refinement; subclustering and merge/prune guards; reassignment; bin length/count filters; skani dedup; EukCC; runtime/backend. Changing multiple related thresholds together makes a benchmark difficult to interpret.
- **Multi-run/best-of/resource:** run selection, cross-run pooling, duplicate mode/thresholds, EukCC executable, resource-profile enable and sample interval.

The `adaptive_min_samples: 2` setting does not claim that each fungus must appear in two samples; it determines the ordinary multi-sample gray-zone path. Single-sample fungi follow the rescue/retain policy explained above. Similarly, a `2000` classifier threshold controls whether a classifier is asked to label a contig; it is not by itself a global 2 kb deletion instruction.

## Tests and known limits

The included `tests/test_multirun.py` is a mocked integration harness for modality selection/pooling and uses a fake `skani`. It does not test real mapping, real Tiara/Whokaryote output schemas across all versions, PyTorch training gradients, true EukCC database behavior, or biological correctness. Run it only in the project's conda environment:

```bash
python tests/test_multirun.py
```

Then separately do a representative Linux end-to-end run with the real external tools. The current packaging environment could syntax-parse Python files but did not have PyYAML, so YAML parsing and this test harness were not run during packaging. Earlier messages in this project cited larger preprocessing test counts; those full suites are not present in this repository package and their counts were not independently verified here.

Other meaningful limitations are reference bias in rescue, short-contig classifier uncertainty, possible organelle/nuclear misclassification, similarity of closely related organisms, dependence on the TE library, sample/coverage confounding, and downstream thresholds that can remove small real bins. The pipeline reports evidence and decisions; it cannot replace careful experimental design and interpretation.

## License and contribution

No license was supplied. Add the intended license and citation before making the GitHub repository public or accepting reuse. For a methods paper or public release, also document the exact tested commit, complete test status, databases and their licenses, and benchmark truth/evaluation method. Keep real sample data, patient/private paths, access tokens, and large licensed databases out of Git history.
