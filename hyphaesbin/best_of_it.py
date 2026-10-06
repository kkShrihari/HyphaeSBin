"""
hyphaesbin.best_of_it -- pool the bins of several independent runs, remove duplicates keeping the BEST copy, and run
EukCC on the result. Nothing in the individual runs is modified.

Config keys (flat yaml):
    best_of_it:        true     (enables it in main.py's multi-run mode)
    best_of_runs:      []       optional subset of run tags to pool (default: every run that has clusters_deduplicated/)
    best_of_af_mode:   contained | both     (default contained)
                         contained: two bins are the same genome if skani ANI >= skani_ani_threshold AND the aligned
                                    fraction of the SMALLER-covered side >= skani_min_af, i.e. a partial copy contained
                                    in a bigger bin is a duplicate too (the less complete copy goes).
                         both:      both aligned fractions must reach skani_min_af (only near-identical bins).
    best_of_ani / best_of_min_af    optional overrides of skani_ani_threshold / skani_min_af
    eukcc_executable:  ""       optional absolute path to eukcc if conda is not on PATH (see clustering._eukcc_argv)

Procedure
  1. pool all runs' bins (<outdir>/runs/<tag>/clustering/clusters_deduplicated/*.fasta)
  2. skani all-vs-all -> duplicate edges (fallback if skani is missing/fails: shared-contig overlap >= 80 % of the smaller bin)
  3. EukCC scores ONLY the bins that have a duplicate (the rest are unique genomes and are kept as they are)
  4. in every duplicate neighbourhood keep the best bin and drop the bins directly redundant with it:
        quality = completeness - 5 x contamination (EukCC); if any bin of the neighbourhood lacks a score the whole
        comparison uses the marker-free proxy (N50, then total length) and says so
  5. write <outdir>/best_of_it/clusters_deduplicated/bin_NNN.fasta (+ relative .fa links), cluster_summary.tsv,
     removed_duplicates.tsv, shared_contigs.tsv, best_of_manifest.json
  6. run EukCC on those final bins -> <outdir>/best_of_it/eukcc/eukcc.csv and add the scores to cluster_summary.tsv
Standalone:  python -m hyphaesbin.best_of_it --outdir <run outdir> --config <yaml>
"""
import argparse
import collections
import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from hyphaesbin.utils.logger import get_logger

log = get_logger("best_of_it")


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("1", "true", "yes", "on")


def _bin_fasta_files(bins_dir: Path) -> Dict[str, Path]:
    """bin stem -> real FASTA path (real .fasta preferred over .fa links; broken links are skipped)."""
    out = {}
    bins_dir = Path(bins_dir)
    for stem in sorted({p.stem for p in list(bins_dir.glob("*.fasta")) + list(bins_dir.glob("*.fa"))}):
        for ext in (".fasta", ".fa"):
            p = bins_dir / f"{stem}{ext}"
            if p.exists():
                out[stem] = p.resolve()
                break
    return out


def _fasta_contigs(path: Path) -> Dict[str, int]:
    ids, cur, n = {}, None, 0
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    ids[cur] = n
                cur, n = line[1:].split()[0], 0
            else:
                n += len(line.strip())
    if cur is not None:
        ids[cur] = n
    return ids


def _n50(lengths: List[int]) -> int:
    x = sorted(lengths, reverse=True)
    tot, acc = sum(x), 0
    for v in x:
        acc += v
        if acc >= tot / 2:
            return v
    return 0


def _place(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _skani_edges(bins: Dict[str, dict], root: Path, threads: int, ani_thr: float, af_thr: float, af_mode: str):
    """-> (edges {uid: {uid2: (ani|None, af)}}, method)"""
    edges: Dict[str, Dict[str, tuple]] = collections.defaultdict(dict)
    path_to_uid = {str(b["path"]): u for u, b in bins.items()}
    ok = False
    if shutil.which("skani") and len(bins) > 1:
        d = root / "skani"
        d.mkdir(parents=True, exist_ok=True)
        lst, res = d / "bin_list.txt", d / "skani_results.tsv"
        lst.write_text("\n".join(str(b["path"]) for b in bins.values()) + "\n")
        cmd = ["skani", "dist", "--ql", str(lst), "--rl", str(lst), "-o", str(res), "-t", str(threads),
               "--min-af", str(min(af_thr, 50.0) if af_mode == "contained" else af_thr)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            ok = r.returncode == 0 and res.exists()
            if not ok:
                log.warning(f"skani failed (rc={r.returncode}): {(r.stderr or '')[:200]}")
        except OSError as e:
            log.warning(f"skani could not be launched ({e})")
        if ok:
            with open(res) as f:
                next(f, None)
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) < 5:
                        continue
                    try:
                        ani, afr, afq = float(p[2]), float(p[3]), float(p[4])
                    except ValueError:
                        continue
                    a, b = path_to_uid.get(p[0]), path_to_uid.get(p[1])
                    af = max(afr, afq) if af_mode == "contained" else min(afr, afq)
                    if a and b and a != b and ani >= ani_thr and af >= af_thr:
                        edges[a][b] = (ani, af)
                        edges[b][a] = (ani, af)
    if ok:
        return edges, "skani"
    log.warning("FLAG:BEST_OF_SKANI_UNAVAILABLE using shared-contig overlap (>= 80 % of the smaller bin) instead of skani")
    uids = list(bins)
    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            A, B = bins[uids[i]], bins[uids[j]]
            sh = sum(L for c, L in A["contigs"].items() if c in B["contigs"])
            small = min(A["total"], B["total"])
            if sh > 0 and sh >= 0.8 * small:
                edges[uids[i]][uids[j]] = (None, 100.0 * sh / small)
                edges[uids[j]][uids[i]] = (None, 100.0 * sh / small)
    return edges, "contig_overlap"


def run_best_of_it(config: Dict, outdir, run_tags: Optional[List[str]] = None) -> Dict:
    import hyphaesbin.clustering.clustering as clu

    outdir = Path(outdir)
    root = outdir / "best_of_it"
    root.mkdir(parents=True, exist_ok=True)
    cfg = clu.ClusteringConfig.from_dict(config)
    cfg_e = dataclasses.replace(cfg, run_eukcc=True, resume=True)     # EukCC is always run here, whatever run_eukcc says
    if config.get("eukcc_executable"):
        os.environ["HYPHAESBIN_EUKCC"] = str(config["eukcc_executable"])
    ani_thr = float(config.get("best_of_ani") or cfg.skani_ani_threshold)
    af_thr = float(config.get("best_of_min_af") or cfg.skani_min_af)
    af_mode = str(config.get("best_of_af_mode") or "contained").lower()
    if af_mode not in ("contained", "both"):
        raise ValueError(f"best_of_af_mode must be 'contained' or 'both', got {af_mode!r}")

    # ---- 1. pool ----
    runs_root = outdir / "runs"
    tags = run_tags or config.get("best_of_runs") or []
    if isinstance(tags, str):
        tags = [x.strip() for x in tags.replace(";", ",").split(",") if x.strip()]
    if not tags:
        tags = sorted(d.name for d in runs_root.iterdir() if (d / "clustering" / "clusters_deduplicated").is_dir()) if runs_root.is_dir() else []
    bins: Dict[str, dict] = {}
    used = []
    for ri, tag in enumerate(tags):
        bdir = runs_root / tag / "clustering" / "clusters_deduplicated"
        files = _bin_fasta_files(bdir) if bdir.is_dir() else {}
        if not files:
            log.warning(f"best_of_it: run '{tag}' has no bins ({bdir}) -- skipped")
            continue
        used.append(tag)
        for stem, p in files.items():
            contigs = _fasta_contigs(p)
            lens = list(contigs.values())
            bins[f"{tag}/{stem}"] = {"run": tag, "run_idx": ri, "stem": stem, "path": p, "contigs": contigs, "n": len(lens),
                                     "total": int(sum(lens)), "n50": _n50(lens), "max": max(lens) if lens else 0,
                                     "comp": None, "cont": None, "q": None}
    if not bins:
        raise RuntimeError(f"best_of_it: no bins to pool (runs looked at: {tags})")
    log.info(f"BEST_OF_IT  pooling {len(bins)} bins from runs {used} | skani ANI>={ani_thr} AF>={af_thr} ({af_mode})")

    # ---- 2. duplicate edges ----
    edges, method = _skani_edges(bins, root, int(cfg.clustering_threads), ani_thr, af_thr, af_mode)
    dup_uids = sorted(u for u in bins if edges.get(u))
    log.info(f"BEST_OF_IT  {len(dup_uids)} of {len(bins)} bins have at least one duplicate ({method})")

    # ---- 3. EukCC on the bins that have a duplicate ----
    mode, eukcc_pool_status = "structural", "not_needed" if not dup_uids else "skipped"
    if dup_uids:
        stage = root / "_pool_scoring"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        for u in dup_uids:
            (stage / f"{bins[u]['run']}__{bins[u]['stem']}.fasta").symlink_to(bins[u]["path"])
        csv_path = clu.run_eukcc(stage, root, cfg_e, label="eukcc_pool")
        rows = clu._read_eukcc_csv(csv_path) if csv_path else None
        if rows:
            score = {r["bin"]: r for r in rows}
            for u in dup_uids:
                r = score.get(f"{bins[u]['run']}__{bins[u]['stem']}")
                if r:
                    bins[u]["comp"], bins[u]["cont"] = r["completeness"], r["contamination"]
                    bins[u]["q"] = r["completeness"] - 5.0 * r["contamination"]
            scored = sum(1 for u in dup_uids if bins[u]["q"] is not None)
            eukcc_pool_status = f"scored {scored}/{len(dup_uids)}"
            if scored == len(dup_uids):
                mode = "eukcc"
            else:
                log.warning(f"FLAG:BEST_OF_PARTIAL_SCORES only {scored}/{len(dup_uids)} duplicate bins have an EukCC score -> "
                            f"structural proxy (N50, total length) is used for ALL comparisons")
        else:
            eukcc_pool_status = "eukcc_unavailable_or_failed"
            log.warning("FLAG:BEST_OF_NO_EUKCC EukCC produced no scores -> duplicates are resolved with the structural proxy "
                        "(N50, total length), NOT by completeness/contamination")
    log.info(f"BEST_OF_IT  quality mode: {mode} (EukCC pool scoring: {eukcc_pool_status})")

    # ---- 4. choose the best of each duplicate neighbourhood ----
    def key(u):
        b = bins[u]
        q = b["q"] if (mode == "eukcc" and b["q"] is not None) else (-1e18 if mode == "eukcc" else 0.0)   # unique bins (never scored) are never compared
        return (q, b["n50"], b["total"], -b["run_idx"], u)

    remaining, kept, absorbed_by = set(bins), [], {}
    while remaining:
        best = max(remaining, key=key)
        direct = set(edges.get(best, {})) & remaining
        for u in direct:
            absorbed_by[u] = best
        remaining.discard(best)
        remaining.difference_update(direct)
        kept.append(best)
    kept.sort(key=lambda u: (-bins[u]["total"], u))

    # ---- 5. write the final bins + tables ----
    fa_dir = root / "clusters_deduplicated"
    if fa_dir.exists():
        shutil.rmtree(fa_dir)
    fa_dir.mkdir(parents=True)
    new_id, assign = {}, []
    for i, u in enumerate(kept, 1):
        nid = f"bin_{i:03d}"
        new_id[u] = nid
        _place(bins[u]["path"], fa_dir / f"{nid}.fasta")
        (fa_dir / f"{nid}.fa").symlink_to(f"{nid}.fasta")                  # relative link
        assign.extend((c, nid, bins[u]["run"]) for c in bins[u]["contigs"])

    def fmt(x):
        return "" if x is None else (round(x, 3) if isinstance(x, float) else x)

    summary_rows = []
    for u in kept:
        b = bins[u]
        summary_rows.append({"bin_id": new_id[u], "source_run": b["run"], "source_bin": b["stem"], "n_contigs": b["n"],
                             "total_length_bp": b["total"], "n50_bp": b["n50"], "pool_completeness": fmt(b["comp"]),
                             "pool_contamination": fmt(b["cont"]), "had_duplicate": "yes" if edges.get(u) else "no",
                             "absorbed": ",".join(sorted(f"{bins[x]['run']}/{bins[x]['stem']}" for x, k in absorbed_by.items() if k == u)),
                             "status": "kept"})
    with open(root / "removed_duplicates.tsv", "w") as f:
        f.write("removed_bin\tkept_bin\tkept_as\tani\taligned_fraction\tremoved_completeness\tremoved_contamination\t"
                "kept_completeness\tkept_contamination\tquality_mode\n")
        for u, k in sorted(absorbed_by.items()):
            ani, af = edges[k][u]
            f.write("\t".join(str(x) for x in (u, k, new_id[k], fmt(ani), fmt(af), fmt(bins[u]["comp"]), fmt(bins[u]["cont"]),
                                               fmt(bins[k]["comp"]), fmt(bins[k]["cont"]), mode)) + "\n")
    occ = collections.Counter(c for c, _, _ in assign)
    shared = collections.defaultdict(list)
    for c, nid, _ in assign:
        if occ[c] > 1:
            shared[c].append(nid)
    with open(root / "shared_contigs.tsv", "w") as f:
        f.write("contig_id\tbins\n")
        for c, ns in sorted(shared.items()):
            f.write(f"{c}\t{','.join(ns)}\n")
    with open(root / "cluster_assignments.tsv", "w") as f:
        f.write("contig_id\tbin_id\tsource_run\tn_bins_for_contig\n")
        for c, nid, run in assign:
            f.write(f"{c}\t{nid}\t{run}\t{occ[c]}\n")
    if shared:
        log.warning(f"BEST_OF_IT  {len(shared):,} contig(s) are in more than one kept bin (the runs placed them differently) "
                    f"-> shared_contigs.tsv; bins are not altered")

    # ---- 6. EukCC on the final bins ----
    final_status = "skipped"
    csv_path = clu.run_eukcc(fa_dir, root, cfg_e, label="eukcc")
    rows = clu._read_eukcc_csv(csv_path) if csv_path else None
    if rows:
        sc = {r["bin"]: r for r in rows}
        for r in summary_rows:
            s = sc.get(r["bin_id"])
            r["completeness"], r["contamination"] = (fmt(s["completeness"]), fmt(s["contamination"])) if s else ("", "")
        final_status = f"scored {sum(1 for r in summary_rows if r['completeness'] != '')}/{len(summary_rows)}"
    else:
        for r in summary_rows:
            r["completeness"], r["contamination"] = "", ""
        final_status = "eukcc_unavailable_or_failed"
        log.warning("FLAG:BEST_OF_FINAL_EUKCC_FAILED the final bins have no EukCC scores (see the EukCC log lines above)")
    cols = list(summary_rows[0].keys())
    with open(root / "cluster_summary.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in summary_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    manifest = {"runs_pooled": used, "pooled_bins": len(bins), "bins_with_duplicate": len(dup_uids), "kept_bins": len(kept),
                "removed_duplicates": len(absorbed_by), "dedup_method": method, "quality_mode": mode, "af_mode": af_mode,
                "ani_threshold": ani_thr, "min_af": af_thr, "eukcc_pool_scoring": eukcc_pool_status,
                "eukcc_final": final_status, "contigs_in_multiple_kept_bins": len(shared),
                "kept_by_run": dict(collections.Counter(bins[u]["run"] for u in kept))}
    (root / "best_of_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info(f"BEST_OF_IT  {len(bins)} pooled -> {len(kept)} kept ({len(absorbed_by)} duplicates removed; {method}; {mode}); "
             f"final EukCC: {final_status}; kept by run: {manifest['kept_by_run']}  ->  {root}")
    manifest["path"] = str(root)
    return manifest


def _main():
    import yaml
    from hyphaesbin.utils.logger import setup_logger
    ap = argparse.ArgumentParser(description="best-of-it: pool runs/*/clustering bins, skani-dedup keeping the best, EukCC on the best")
    ap.add_argument("--outdir", required=True, help="pipeline output directory (contains runs/)")
    ap.add_argument("--config", required=True, help="the same yaml used for the pipeline")
    ap.add_argument("--runs", default=None, help="comma-separated run tags (default: all runs with bins)")
    a = ap.parse_args()
    setup_logger(a.outdir, "INFO")
    config = yaml.safe_load(open(a.config)) or {}
    tags = [x.strip() for x in a.runs.split(",")] if a.runs else None
    run_best_of_it(config, a.outdir, tags)




# PROFILE HOOKS v1 — observation only; scientific functions above are unchanged.
from hyphaesbin.utils.resource_profiler import profile_functions, profile_checkpoint
profile_functions(globals(), ['run_best_of_it', '_skani_edges'])

if __name__ == "__main__":
    _main()
