"""v9.3: verify that a clustering output folder is COMPLETE before it is treated as cached / exported."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List


def missing_clustering_outputs(clu_dir, dedup_name: str = "clusters_deduplicated") -> List[str]:
    """[] when complete. Otherwise human-readable problems: missing tables, or a kept bin in cluster_summary.tsv
    without a non-empty FASTA in the deduplicated bin folder."""
    clu_dir = Path(clu_dir)
    problems = []
    for f in ("cluster_summary.tsv", "cluster_assignments.tsv", "clustering_stats.json",
              "stage_quality.tsv", "cluster_quality.json"):
        if not (clu_dir / f).is_file() or (clu_dir / f).stat().st_size == 0:
            problems.append(f"missing {f}")
    sq = clu_dir / "stage_quality.tsv"
    if sq.is_file() and "06_final_exported" not in sq.read_text():
        problems.append("stage_quality.tsv has no 06_final_exported row (final-bin metrics)")
    cq = clu_dir / "cluster_quality.json"
    if cq.is_file():
        import json
        try:
            d = json.loads(cq.read_text())
            if "per_bin_eukcc" not in d:
                problems.append("cluster_quality.json lacks per_bin_eukcc")
        except ValueError:
            problems.append("cluster_quality.json unreadable")
    summ = clu_dir / "cluster_summary.tsv"
    if summ.is_file():
        bdir = clu_dir / (dedup_name or "clusters_deduplicated")
        with open(summ) as h:
            for r in csv.DictReader(h, delimiter="\t"):
                b = r.get("bin_id", "")
                if not b or b == "unbinned" or not str(r.get("status", "kept")).startswith("kept"):
                    continue
                fa = bdir / f"{b}.fasta"
                if not fa.is_file() or fa.stat().st_size == 0:
                    problems.append(f"missing bin FASTA {fa.name}")
    return problems
