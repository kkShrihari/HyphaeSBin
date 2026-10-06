# HyphaeSBin resource-reporting update

Copy the files in this archive over the same relative paths in your HyphaeSBin repository. Back up the current repository first. This package is an overlay, not a complete repository. It starts from the latest `files.zip` for `main.py`, preprocessing, clustering, multi-run, best-of-it, and the launcher; it starts from the separately uploaded files for coverage, TNF, TE, encoder, checkpoint, and environment.

Replacements:

| Repository path | Change |
|---|---|
| `main.py` | Starts the pipeline profiler when the output directory is created. |
| `hyphaesbin/utils/resource_profiler.py` | New shared sampler and JSON/TSV/Markdown report writer. |
| `hyphaesbin/preprocessing/preprocessing.py` | Times steps 1–13, Tiara, Whokaryote, per-sample mapping, and checkpoint reuse. |
| `hyphaesbin/coverage/coverage.py` | Times coverage feature stages and cache checks. |
| `hyphaesbin/composition/composition/TNF_gene/tnf_gene.py` | Times TNF features and cache checks. |
| `hyphaesbin/composition/composition/TE_composition/te_composition.py` | Times TE search, RepeatMasker, feature building, and cache checks. |
| `hyphaesbin/encoder/encoder.py` | Times each encoder call and training phases. |
| `hyphaesbin/clustering/clustering.py` | Times each clustering call and major substeps. |
| `hyphaesbin/modality_runs.py` | Times the multi-run, including cached encoder and clustering runs. |
| `hyphaesbin/best_of_it.py` | Times pooling, skani comparisons, and final selection. |
| `hyphaesbin/utils/checkpoint.py` | Marks successful checkpoint writes as computed. |
| `environment.yaml` | Adds `psutil` for parent/child memory, CPU, and I/O sampling. |
| `camisim_final_tetnfcov_multirun.yaml` | Enables reporting at a 0.5-second interval. |
| `run_multirun.sh` | Also writes Linux `/usr/bin/time -v` totals to `pipeline_time.txt`. |

The pipeline writes `resource_report.json`, `resource_report.tsv`, and `resource_report.md` in its output directory. Each row has wall time, CPU seconds and utilization, sampled concurrent RAM, available GPU process memory and device utilization, process read/write bytes, filesystem-use delta, cache/status, available input/output bytes and contig/bp metadata, and throughput when an input count is known. `pipeline_total` is the overall wall time. Stage rows are nested: do not sum them. The JSON summary names top-level and leaf-stage bottlenecks.

The sampler follows the main process and currently live children. Very brief child-process peaks can be missed. GPU utilization is device-wide, while GPU memory is summed for matching compute PIDs. Filesystem-use delta can include unrelated writers and is not an exact temporary-space peak. Counts and bp absent from a stage's existing metadata are reported as unavailable; reporting does not re-read large FASTA files solely to obtain them. Parallel phase-1 encoder workers contribute to the parent encoder's aggregate usage, but their individual resource use is not isolated. `pipeline_time.txt` provides an independent whole-run measurement when launched with the included script; its maximum RSS is a process-level figure and may miss simultaneously running workers.

Validation performed here: all delivered Python sources passed AST parsing; a concurrent profiler smoke test verified report writing, cached rows, and failed-stage rows. A full biological run and the multi-run integration suite were not run in this Windows environment because its Python runtime lacks the scientific and YAML dependencies. Run the included launcher in the actual Linux environment before relying on peak values.
