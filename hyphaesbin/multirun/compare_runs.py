"""
hyphaesbin.multirun.compare_runs -- export every finished modality run side by side. NO winner is chosen.

Writes <outdir>/results/
    <tag>/                 links to that run's final bins, QC table, rejected bins, stage tables, EukCC output
    runs_comparison.tsv    one neutral row per run (counts only; read the trade-offs yourself)
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List

from hyphaesbin.utils.logger import get_logger

log = get_logger("compare_runs")

_LINK = ("clusters_deduplicated", "bins_by_qc", "rejected_bins", "bin_qc_status.tsv", "cluster_summary.tsv",
         "cluster_assignments.tsv", "clustering_stats.json", "stage_quality.tsv", "stage_quality_candidates.tsv",
         "stage_quality_chosen_params.json", "split_diagnostics.tsv", "gate_candidates", "organelles",
         "clusters_nuclear_only", "eukcc")


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink():
            dst.unlink()
        else:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(os.path.relpath(src, dst.parent))


def _row(tag: str, clu: Path) -> Dict:
    r = {"run": tag}
    try:
        st = json.loads((clu / "clustering_stats.json").read_text())
    except (OSError, ValueError):
        st = {}
    qc = st.get("bin_qc") or {}
    er = st.get("eligibility_and_recruitment") or {}
    r.update({
        "bins_exported": qc.get("n_bins_exported", st.get("n_bins_final", "")),
        "qc_passed": qc.get("n_qc_passed", ""), "qc_failed": qc.get("n_qc_failed", ""),
        "qc_unassessed": qc.get("n_qc_unassessed", ""),
        "mimag_HQ": qc.get("n_mimag_HQ", ""), "mimag_MQ": qc.get("n_mimag_MQ", ""),
        "rejected_size_filter": qc.get("n_rejected_size_filter", ""), "rejected_dedup": qc.get("n_rejected_dedup", ""),
        "eligible_contigs": er.get("eligible_contigs", ""),
        "binned_eligible_pct_bp": (round(100 * er["binned_eligible_bp"] / er["eligible_bp"], 2)
                                   if er.get("eligible_bp") else ""),
        "recruited_short_contigs": er.get("binned_recruited_short_contigs", ""),
        "recruited_short_Mb": (round(er["binned_recruited_short_bp"] / 1e6, 2)
                               if "binned_recruited_short_bp" in er else ""),
        "organelle_contigs": qc.get("n_organelle_contigs", ""),
        "organelle_high_confidence": qc.get("n_organelle_high_confidence", ""),
    })
    total = 0
    try:
        with open(clu / "cluster_summary.tsv") as h:
            for x in csv.DictReader(h, delimiter="\t"):
                if x.get("bin_id") != "unbinned":
                    total += int(float(x.get("total_length_bp") or 0))
    except OSError:
        pass
    r["binned_Mb"] = round(total / 1e6, 2)
    try:
        with open(clu / "stage_quality.tsv") as h:
            rows = list(csv.DictReader(h, delimiter="\t"))
        fin = next((x for x in rows if x["stage"] == "06_final_exported"), {})
        for k in ("silhouette", "knn_agree@10", "knn_agree@50"):
            r[f"final_{k}"] = fin.get(k, "")
        r["gate_rejected_stages"] = ",".join(x["stage"] for x in rows if x["decision"].startswith("rejected"))
        r["gate_no_change_stages"] = ",".join(x["stage"] for x in rows if x["decision"] == "accepted_no_change")
    except OSError:
        pass
    return r


def export_runs(outdir, tags: List[str]) -> Dict:
    outdir = Path(outdir)
    res = outdir / "results"
    res.mkdir(parents=True, exist_ok=True)
    for stale in ("best_run", "best_run.json"):
        if (res / stale).exists() or (res / stale).is_symlink():
            log.warning(f"results/{stale} is left over from an older version (automatic winner selection was "
                        f"removed); it is NOT updated -- delete it to avoid confusion")
    rows = []
    for t in tags:
        clu = outdir / "runs" / t / "clustering"
        for name in _LINK:
            if (clu / name).exists():
                _link(clu / name, res / t / name)
        rows.append(_row(t, clu))
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(res / "runs_comparison.tsv", "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=cols, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    try:
        from hyphaesbin.utils.provenance import package_provenance
        (res / "provenance.json").write_text(json.dumps(package_provenance(), indent=2) + "\n")
    except Exception as ex:
        log.warning(f"could not write results/provenance.json: {ex}")
    log.info(f"RESULTS  {len(rows)} run(s) exported side by side (no automatic winner) -> {res}/runs_comparison.tsv")
    return {"status": "ok", "runs": tags, "table": str(res / "runs_comparison.tsv")}
