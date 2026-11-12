# HyphaeSBin

**TE-aware, multimodal metagenomic binning for microbial eukaryotes**

HyphaeSBin is a research-oriented Python pipeline for recovering candidate microbial-eukaryotic genome bins from metagenomic assemblies. It integrates **tetranucleotide-frequency (TNF) composition**, **transposable-element (TE) composition**, and **read-coverage profiles** with a multimodal β-variational autoencoder (β-VAE), followed by clustering, refinement, and optional genome-quality assessment.

The package is designed for fragmented, mixed-community metagenomes where conventional prokaryote-focused binning workflows may not adequately recover eukaryotic sequences. Fungal genomes are an important target, but **HyphaeSBin is not a fungal-species identification tool and does not guarantee that every exported bin is fungal or a complete genome**. Its results require independent taxonomic and marker-gene validation.

> **Status:** Research software; the uploaded source distribution declares version **9.6.0** in `pyproject.toml`. The bundled README records incremental development and local tests, not a comprehensive independent validation study. Review the limitations and validate outputs before making biological claims.

## Contents

- [Overview](#overview)
- [Workflow](#workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Input data](#input-data)
- [Command-line reference](#command-line-reference)
- [Configuration and included examples](#configuration-and-included-examples)
- [Pipeline stages](#pipeline-stages)
- [Multi-modality experiments](#multi-modality-experiments)
- [Results and interpretation](#results-and-interpretation)
- [Checkpoints, resuming, and re-clustering](#checkpoints-resuming-and-re-clustering)
- [Quality control](#quality-control)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Limitations and development status](#limitations-and-development-status)
- [Citation and license](#citation-and-license)

## Overview

HyphaeSBin combines three complementary signatures:

| Modality | Biological/computational signal | Principal role |
| --- | --- | --- |
| **TNF** | Frequencies of four-base sequence patterns | Captures sequence-composition relationships among contigs |
| **TE** | Similarity of contigs to a supplied TE consensus database | Provides an additional signal relevant to repeat-rich eukaryotic genomes |
| **COV** | Read-mapping coverage across one or more sequencing samples | Exploits shared abundance/coverage profiles |

The software builds features, learns latent representations, clusters contigs, and exports candidate bins. Model-derived clusters are **hypotheses about genomic membership**, not proof of organismal identity.

Core implementation: `hyphaesbin/pipeline.py`; constituent modules are under `hyphaesbin/`.

## Workflow

```text
Metagenomic assembly FASTA + one or more read libraries
                         |
                         v
          [1] Preprocessing and filtering
              - length/quality handling
              - Tiara + Whokaryote classification
              - optional rDNA masking, deduplication
                         |
            +------------+-------------+
            |            |             |
            v            v             v
       [2] Coverage  [3] Gene/TNF   [4] TE composition
       read mapping      features      MMseqs2 + TE database
            |            |             |
            +------------+-------------+
                         |
                         v
           [5] Multimodal β-VAE encoding
               modality-specific and fused latents
                         |
                         v
           [6] Clustering and refinement
               HDBSCAN, optional recruitment,
               optional quality gates, bin export
                         |
                         v
             Candidate genome-bin FASTAs
             QC/status tables and diagnostics
             Optional EukCC assessment
```

The exact filters, clustering decisions, modality combinations, and evaluation steps depend on the chosen YAML configuration. Not all stages are enabled in every supplied configuration.

## Requirements

**Platform:** The bundled installer is written for a Unix-like shell and Conda/Mamba environments; paths and environment creation steps are Linux-oriented. Windows users should use a compatible Linux environment such as WSL2 or a Linux server. GPU acceleration is not required by the packaged CPU environment; memory/runtime depend strongly on assembly size.

**Python:** `pyproject.toml` specifies Python >=3.9. The supplied Conda environment explicitly uses **Python 3.10**.

**External programs:** `minimap2`, `samtools`, `seqkit`, `skani`, `coverm`, `barrnap`, `nhmmer` (HMMER), `prodigal`, `mmseqs`, `tiara`, and `whokaryote.py` are checked by `install.sh`. EukCC is optional and needs its own installation and database.

**Python packages:** PyYAML, NumPy, pandas, SciPy, scikit-learn, Biopython, psutil, PyTorch, HDBSCAN, with FAISS support supplied by the environment (the Python package also declares `faiss-cpu` as an optional extra).

For large soil co-assemblies, plan for substantial RAM, storage, and compute. **No universal minimum RAM or runtime is established by this repository.**

## Installation

### Recommended: bundled installer

From the repository root:

```bash
bash install.sh
conda activate HyphaeSBin_env
hyphaesbin --help
```

`install.sh` selects `mamba`, `micromamba`, or `conda`, creates/updates the `HyphaeSBin_env` environment, installs the project in editable mode, and creates a separate `HyphaeSBin_classify` environment for Tiara and Whokaryote. These classifiers have incompatible dependency constraints with the main environment; the pipeline is designed to locate the classifier environment automatically.

To install the optional EukCC environment:

```bash
bash install.sh --with-eukcc
```

Optional installer flags:

```bash
bash install.sh --force                    # Rebuild existing environments
bash install.sh --with-eukcc --eukcc-env eukcc
bash install.sh --help
```

**Important:** An EukCC *database* is not bundled. Set its path in the YAML before enabling EukCC. Installing EukCC alone does not supply the database.

### Alternative: install the main environment only

```bash
mamba env create -f environment.yaml
conda activate HyphaeSBin_env
python -m pip install --no-deps -e .
```

This route does **not** necessarily set up Tiara and Whokaryote; the full installer is preferred. Likewise, `pip install -e .` alone does not install the external command-line bioinformatics tools.

### Verify dependencies

```bash
hyphaesbin --help
python -c "import torch, hdbscan, sklearn, Bio, yaml, psutil; print('Python dependencies OK')"
command -v minimap2 samtools seqkit skani coverm barrnap nhmmer prodigal mmseqs
```

Check both classifier executables and, if enabled, the EukCC command/database. For scientific reproducibility, record actual installed versions rather than assuming the installer pins every program to a fixed release.

## Quick start

### 1. Prepare inputs

```text
project/
  assembly/contigs.fasta
  reads/sampleA_R1.fastq.gz
  reads/sampleA_R2.fastq.gz
  reads/sampleB_R1.fastq.gz
  reads/sampleB_R2.fastq.gz
  config.yaml
```

### 2. Choose and edit a supplied YAML

For example, copy a supplied configuration and **replace all site-specific absolute paths** (in particular `funTEdb`, `eukcc_db`, and `eukcc_executable` if present). Review classifier thresholds, minimum contig sizes, modality choices, and optional quality steps.

```bash
cp nz_soil_final.yaml config.yaml
```

The NZ soil preset is *not* a generic default: it contains settings tailored to a large soil co-assembly and can skip deduplication. Do not assume it is appropriate for small test assemblies or another organism.

### 3. Run

```bash
hyphaesbin \
  --scaffold project/assembly/contigs.fasta \
  --reads-dir project/reads \
  --reads sampleA:sampleA_R1.fastq.gz:sampleA_R2.fastq.gz \
  --reads sampleB:sampleB_R1.fastq.gz:sampleB_R2.fastq.gz \
  --outdir project/hyphaesbin_results \
  --config project/config.yaml \
  --threads 16
```

Equivalent entry point from the checkout:

```bash
python main.py --scaffold project/assembly/contigs.fasta \
  --reads sampleA:project/reads/sampleA_R1.fastq.gz:project/reads/sampleA_R2.fastq.gz \
  --outdir project/hyphaesbin_results --config project/config.yaml
```

These commands are **usage examples**, not evidence that a run succeeds with an unedited preset or unavailable databases.

## Input data

| Input | Format | Notes |
| --- | --- | --- |
| Assembly | FASTA (`--scaffold`) | Metagenomic contigs/scaffolds with stable, unique sequence IDs; required for a new normal run |
| Sequencing reads | FASTQ or FASTQ.gz (`--reads`) | One or multiple libraries, single- or paired-end; repeat flag per library |
| Configuration | YAML (`--config`) | Controls preprocessing, features, model, clustering, and QC |
| TE database | FASTA path in `funTEdb` | Used by the MMseqs2 TE-composition workflow; **not included** in the archive |
| EukCC database | Installed EukCC database path | Required only when EukCC assessment is enabled |
| Precomputed TE features | `.npy` (`--precomputed-te`) | Advanced option; must match contig order/identifiers and expected feature dimensions |
| Precomputed TE weights | `.npy` (`--precomputed-te-weights`) | Advanced companion option for TE reliability weights |

Read syntax:

```bash
--reads SAMPLE:READ1.fastq.gz
--reads SAMPLE:READ1.fastq.gz:READ2.fastq.gz
```

Use a **different sample name for each independent read library**. `--reads-dir` supplies a base directory for relative read filenames. Coverage comparisons become more informative when multiple biologically meaningful samples are available, but appropriate sample count depends on the experiment.

## Command-line reference

The following options are declared by the current `parse_args()` implementation in `hyphaesbin/pipeline.py`:

| Option | Description |
| --- | --- |
| `--scaffold PATH` | Assembly FASTA; required except for `--reclust` |
| `--reads NAME:R1[:R2]` | Read library; repeat for multiple samples |
| `--reads-dir DIR` | Base directory for reads; default `.` |
| `--outdir DIR` | Output directory; for `--reclust`, the original run directory |
| `--config FILE` | YAML configuration |
| `--threads N` | Override top-level YAML thread setting |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`; default `INFO` |
| `--reset` | Reset top-level pipeline checkpoints; see checkpoint section |
| `--precomputed-te FILE` | Use precomputed TE-feature NumPy array, skipping phase 4 computation |
| `--precomputed-te-weights FILE` | Supply precomputed TE weights |
| `--skip-to-phase N` | Start at phase 1–6, reusing earlier outputs already on disk |
| `--reclust` | Re-run clustering of a completed run into a separate subdirectory |
| `--reclust-name NAME` | Name the re-clustering result folder; existing names resume |

See `hyphaesbin --help` for the installed checkout's authoritative command-line help.

## Configuration and included examples

| File | Intended scenario / caution |
| --- | --- |
| `camisim_final.yaml` | Simulation-oriented preset; database paths and benchmark assumptions need review |
| `nz_soil_final.yaml` | Large New Zealand soil co-assembly; includes run-specific thresholds and deduplication decisions |
| `zostera_marina_final.yaml` | *Zostera marina* metagenomic experiment preset |
| `zostera_marina_1000bp_final.yaml` | Alternative *Zostera* preset with different short-contig handling |
| `environment.yaml` | Main Conda environment |

All these presets are **starting configurations**, not portable turnkey examples; some reference `/DATA_LUN/...` or other paths specific to the original compute environment.

### Important settings

| YAML key | Function / caution |
| --- | --- |
| `threads`, `seed` | Execution parallelism and random seed |
| `funTEdb` | Path to the actual TE reference FASTA |
| `contig_filter` | Threshold separating main contigs from the short-contig set |
| `adaptive_filter_min` | Lower cutoff used by adaptive short-contig processing |
| `tiara_min_len`, `whokaryote_minsize` | Independent minimum lengths for the two classifiers |
| `prokaryote_removal_mode` | `any` removes when either tool classifies as prokaryotic; `agree` requires agreement |
| `unknown_removal_enabled` | Whether unresolved sequences are removed |
| `short_contigs_in_encoder_training` | Whether eligible short contigs train the encoder |
| `short_contigs_in_clustering_rescue` | Whether eligible short contigs are offered for clustering recruitment |
| `mmseqs_min_seq_id`, `mmseqs_evalue` | TE alignment acceptance parameters |
| `mmseqs_coverage`, `mmseqs_cov_mode` | TE alignment coverage requirement; mode 1 evaluates target/reference coverage, mode 2 query/contig coverage |
| `run_eukcc`, `eukcc_db`, `eukcc_conda_env` | EukCC enablement and environment/database configuration |
| `gate_enabled` | Whether internal clustering quality gating is enabled |
| `modality_runs` | Select specific independent feature-combination experiments |
| `all_runs`, `te_runs` | Broader modality-run selection switches |
| `organelle_policy` | How putative organellar sequences are handled/reported |

**Do not assume README examples are the defaults**: settings are determined by the active YAML. In particular, current `nz_soil_final.yaml` explicitly sets `gate_enabled: false`, `short_contigs_in_clustering_rescue: false`, and `modality_runs: ["te"]`; other presets differ.

## Pipeline stages

### Phase 1 — Preprocessing

Prepares an eligible set of assembly contigs, runs sequence classification with **Tiara** and **Whokaryote**, supports optional dereplication through **skani**, and uses **barrnap** for rRNA-associated masking. Exact inclusion/exclusion rules depend on length thresholds and YAML options. `Prokaryotic_Unclassified` is treated as prokaryotic under the documented removal policy. Putative plastid/mitochondrial classifications are more uncertain: the project's own changelog warns about false-positive organelle calls, so inspect `organelle_policy` and output tables instead of deleting them blindly.

The pipeline can preserve information on short contigs and the basis for classifier decisions (for example, contigs below a classifier-specific cutoff are not equivalent to confidently classified unknowns). Classification is error-prone on short or compositionally unusual contigs.

### Phase 2 — Coverage

Maps supplied reads and derives per-contig coverage-related features using the coverage module and external aligners/coverage tools. Coverage comparisons across several samples can help distinguish populations with different abundance dynamics. Confirm library names and mapping statistics before interpreting clustering.

### Phase 3 — TNF composition

Computes tetranucleotide composition from gene-related/sequence regions in `hyphaesbin/composition/composition/TNF_gene/tnf_gene.py`. Quality/confidence handling depends on the available valid k-mers and configuration. Short sequences offer weaker compositional estimates.

### Phase 4 — TE composition

Uses MMseqs2 to find matches between assembled sequences and the specified TE consensus reference, then derives TE-related features and reliability measures. The **biological relevance and taxonomic breadth of the TE database matter**: lack of TE matches may reflect database coverage rather than absence of eukaryotic origin.

The pipeline supports precomputed TE feature arrays through `--precomputed-te` for a repeatable experiment when contig indexing and expected feature shape are compatible.

### Phase 5 — Encoder

Learns modality-specific embeddings and a fused latent representation with a β-VAE architecture. Settings control feature weighting, representation learning, and the fusion gate. A module's learned gating weight **is not itself a causal measure** of biological importance: the project supplies modality-influence/ablation diagnostics, which are more suitable as a description of representation dependence.

### Phase 6 — Clustering and export

Applies HDBSCAN-based clustering, optional refinement/recruitment, quality-control policies, and export of bins as FASTA files. An optional stage-quality gate can compare candidate refinement steps using internal clustering metrics. **Internal silhouette, Davies–Bouldin, Calinski–Harabasz, or kNN scores are not substitutes for reference-based genome quality or taxonomic validation.**

## Multi-modality experiments

The implementation supports the following combinations (`hyphaesbin/multirun/modality_runs.py`):

| Tag | Features |
| --- | --- |
| `tetnfcov` | TE + TNF + coverage |
| `tnfcov` | TNF + coverage |
| `tetnf` | TE + TNF |
| `tecov` | TE + coverage |
| `te` | TE only |
| `tnf` | TNF only |
| `cov` | Coverage only |

**When no run-selection switches override the behavior, the module declares two default independent runs: `tetnfcov` and `tnfcov`.** A preset may explicitly choose a different subset; for example, the supplied NZ soil YAML lists `modality_runs: ["te"]`. Check the actual run-selection logic and resulting log for the precise set launched.

Outputs for separate modality experiments are not pooled into one bin set by default. Compare their recovered bins, assignments, size, marker-gene QC, and experimental biological ground truth **side by side** rather than assuming that the highest internal clustering score identifies the best biological reconstruction.

## Results and interpretation

A normal output contains stage folders and log/report files. Exact paths vary by enabled run combinations and config-defined folder names. Common paths documented in the source include:

```text
<outdir>/
  run_config.yaml                    # saved effective configuration
  preprocessing/                     # filtered/classified contigs, audit outputs
  coverage/                          # read-coverage products
  encoder/                           # encoder products (may link into a run)
  clustering/                        # clustering products (may link into a run)
  runs/<modality-tag>/               # individual modality-run artifacts
  results/
    <modality-tag>/                 # per-run results and bin links
    runs_comparison.tsv             # comparison of independent runs
  logs/                              # pipeline logs
  resource_report.tsv                # resource usage when enabled
  resource_report.txt
  resource_timeline.tsv
```

The source documentation additionally describes the following **conditional** clustering artifacts:

| Artifact | Interpretation |
| --- | --- |
| `bin_qc_status.tsv` | Per-bin `passed`, `failed`, or `unassessed` QC status, with reasons |
| `bins_by_qc/` | Grouping of exported bin FASTAs by quality status |
| `rejected_bins/`, `rejected_bins.tsv` | Bins removed by selected filtering/deduplication operations and rejection records |
| `stage_quality.tsv` | Accepted/rejected changes to clustering stages when stage evaluation is active |
| `stage_quality_candidates.tsv` | Evaluated parameter candidates and metrics |
| `stage_quality_chosen_params.json` | Selected stage parameters when search is active |
| `gate_candidates/` | Saved candidate assignments and quality diagnostics when relevant |
| `clustering_stats.json` | Clustering summary and associated metadata |
| `cluster_quality.json` | Quality-related metrics, potentially including EukCC data |
| `encoder/modality_influence.json` | Optional ablation-derived modality-influence summary |

**A file listed above may not exist if its feature is disabled or the run fails before that step.** Use the run log and actual filesystem, not this schematic, to determine completeness. In particular, `passed` is defined by configured QC thresholds; it is not a guarantee of a correct species-level genome.

### Interpreting bin quality

EukCC completeness/contamination values are marker-based estimates. They are affected by genome novelty, database coverage, assembly quality, lineage placement, and marker duplications. The project records `unassessed` when no valid estimate is available. Do not count such bins as passed, and do not discard them without follow-up checks. Where appropriate, examine BUSCO, taxonomy assignments, contig-level evidence, and manual contamination signals as independent validation steps; **these additional analyses are recommendations, not commands claimed to be built into this archive**.

## Checkpoints, resuming, and re-clustering

HyphaeSBin has phase/sub-step checkpoints, with source- and input-aware fingerprints documented in the bundled change history. To **resume**, reissue the same command with the same `--outdir`:

```bash
hyphaesbin --scaffold assembly.fasta \
  --reads S1:R1.fastq.gz:R2.fastq.gz \
  --outdir existing_run --config config.yaml
```

Changing a config file, sequence file, database, program version, or source code can invalidate stages. Do not manually delete or edit checkpoint metadata without understanding downstream dependencies.

To start at an existing phase:

```bash
hyphaesbin --scaffold assembly.fasta --reads S1:R1.fastq.gz:R2.fastq.gz \
  --outdir existing_run --config config.yaml --skip-to-phase 6
```

This **requires valid existing files** from all previous phases; it does not automatically synthesize missing embeddings.

To re-cluster an already completed run, without replacing its original results:

```bash
hyphaesbin --reclust --outdir existing_run --config changed_clustering.yaml
hyphaesbin --reclust --outdir existing_run --reclust-name trial_02
```

The implementation documents output under `<outdir>/reclust/run_N` or the selected `--reclust-name`; the original preprocessing/features are reused. For reproducibility, assign meaningful names and record parameter changes.

## Quality control

Before interpreting a bin as a recovered eukaryotic genome:

1. **Verify input correspondence:** assembly ID stability, read-library identities, mapping statistics, reference database paths and versions.
2. **Audit the classifier decisions:** particularly conflicts, unknowns, short contigs, and organelle calls.
3. **Inspect clustering:** fraction assigned vs noise, lengths, coverage coherence, suspiciously merged bins and subclusters.
4. **Inspect genome quality:** marker-based completeness/contamination where available and reasons for `unassessed` status.
5. **Seek independent evidence:** taxonomy, expected genome size/GC, marker genes, alignment to references, and, for simulations, sequence-verified ground truth.
6. **Compare modalities:** a TE-only or fused clustering pattern is not proof that a recovered bin corresponds to a separate organism.

The bundled README mentions `tools/build_ground_truth.py` and `tools/gt_stage_report.py`, but **those scripts are not present in the uploaded ZIP**. Do not rely on those paths unless you separately obtain/create the scripts.

## Reproducibility

For every publication-quality or thesis run, retain:

- source commit or archived source version; command line; `run_config.yaml` and all changes;
- assembly and read-file checksums, sample identifiers, and TE/EukCC database release/checksums;
- Conda package export and external executable versions;
- random seed, CPU/GPU details, RAM allocation, and number of threads;
- complete logs, metrics, rejected-bin records, all modality results and QC tables.

An example environment snapshot:

```bash
conda env export -n HyphaeSBin_env > hyphaesbin_env_export.yaml
hyphaesbin --help > hyphaesbin_help.txt
sha256sum assembly.fasta > assembly.sha256
```

Repeated stochastic training may still vary across hardware/library versions. The documentation's local simulation test should not be interpreted as a general reproducibility guarantee.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `hyphaesbin: command not found` | Activate `HyphaeSBin_env` and confirm editable install completed |
| Tiara/Whokaryote missing | Verify `HyphaeSBin_classify`; rerun installer or use its repair behavior where supported |
| MMseqs2 fails or runs out of memory | Check available RAM, `mmseqs_split_memory_limit`, database size, and logs |
| EukCC fails or all bins unassessed | Verify EukCC executable/environment and **installed database**; inspect failed lineage/marker placement |
| TE feature computation fails | Verify `funTEdb` points to a readable FASTA and examine MMseqs2 logs |
| Too few contigs retained | Inspect contig-length distribution, classifier cutoffs, prokaryote removal, and unknown/short-contig policy |
| Many small or mixed bins | Review coverage quality, modality representations, clustering parameters, and independent marker/taxonomy evidence |
| Expected short contigs missing | Compare `contig_filter`, `adaptive_filter_min`, encoder-training and clustering-rescue switches |
| Stale outputs after changing config | Inspect checkpoint invalidation messages, input fingerprints, and current run folder |
| Missing stage-quality tables | Verify `gate_enabled` and whether that stage actually executed |
| Existing server-specific paths fail | Replace hard-coded `/DATA_LUN/...` and other original-host paths in YAML and shell scripts |

**Installer note:** `install.sh` checks `import faiss` in its smoke test although `pyproject.toml` declares FAISS as optional. The supplied Conda `environment.yaml` includes `faiss-cpu`, so use the recommended environment to avoid this mismatch.

## Repository layout

```text
HyphaeSBin/
├── main.py                        # Backward-compatible launcher
├── pyproject.toml                  # Python package metadata and CLI entry point
├── environment.yaml               # Main Conda environment
├── install.sh                     # Installer and optional EukCC setup
├── camisim_final.yaml             # Simulation preset
├── nz_soil_final.yaml             # NZ soil co-assembly preset
├── zostera_marina_final.yaml      # Zostera preset
├── zostera_marina_1000bp_final.yaml
├── run_nz_soil.sh                 # Original-host-specific example
├── run_zostera_1000.sh            # Original-host-specific example
├── run_zostera_2000.sh            # Original-host-specific example
├── run_qc_zostera_bins.sh         # QC helper script
├── dup_check.py                   # Standalone duplicate-check helper
├── README_PROFILE.md              # Profiling reference
└── hyphaesbin/
    ├── cli.py
    ├── pipeline.py                # Six-phase orchestration and CLI implementation
    ├── preprocessing/preprocessing.py
    ├── coverage/coverage.py
    ├── composition/composition/
    │   ├── TNF_gene/tnf_gene.py
    │   └── TE_composition/te_composition.py
    ├── encoder/encoder.py
    ├── clustering/
    │   ├── clustering.py
    │   └── quality_gate.py
    ├── multirun/
    │   ├── modality_runs.py
    │   ├── compare_runs.py
    │   ├── reclust.py
    │   └── verify.py
    └── utils/                     # Checkpoints, env/deps, logs, provenance, profiling
```

The uploaded ZIP also contains generated `__pycache__` and `.egg-info` content. These are build/runtime artifacts and are generally better excluded from the public source repository.

## Limitations and development status

- **Research/experimental software:** Method robustness, correctness, and reproducibility need validation in the target datasets.
- **Classification uncertainty:** Tiara/Whokaryote disagreement and short-contig errors can exclude real eukaryotic sequence or retain unwanted sequence.
- **TE reference bias:** TE features reflect the database chosen and cannot be interpreted as organism identity independently.
- **Marker/database bias:** EukCC can fail to assess unusual or poorly represented taxa.
- **Fragmentation and coverage bias:** Coverage and TNF signatures can be weak for short, low-coverage or repeated contigs.
- **Internal metric limitations:** High silhouette or other internal scores can coexist with biologically incorrect bins.
- **Resource needs:** Large assemblies and multiple samples can make read mapping, similarity search and clustering computationally expensive.
- **Configuration portability:** Included sample presets reference unavailable local server/database paths and may enable different features from those discussed in development notes.
- **No packaged benchmarking suite:** Benchmark scripts mentioned in the old README are absent from the supplied archive.

For manuscript/thesis figures, describe separately what was **implemented**, what was **tested on simulations**, and what has been **validated experimentally or by external markers**. Avoid treating a local smoke test as peer-reviewed benchmark evidence.

## Citation and license

**Citation:** No publication-specific citation or `CITATION.cff` was included in this upload. Until a definitive citation is added, cite the repository URL and the specific version/commit used:

> HyphaeSBin, version 9.6.0 (source distribution), https://github.com/kkShrihari/HyphaeSBin (include access date and commit where available).

**License:** No `LICENSE` file was included in the uploaded archive. **Do not assume the package is open-source licensed merely because it is on GitHub.** Add an explicit license before inviting external redistribution and contributions.

---

*This README documents the inspected uploaded repository and its declared options. It does not assert that the entire package was executed end-to-end during this documentation review.*
