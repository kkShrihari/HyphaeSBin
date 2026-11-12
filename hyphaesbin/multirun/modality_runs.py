"""
hyphaesbin.multirun.modality_runs -- run the encoder + clustering for several MODALITY COMBINATIONS from ONE set of
shared features (preprocessing / coverage / TNF / TE are computed once by main.py), each combination in its own folder.

DEFAULT: two INDEPENDENT runs, tetnfcov (TE+TNF+COV) and tnfcov (TNF+COV). No pooled binning and NO automatic
winner: both runs are exported side by side and compared neutrally in <outdir>/results/:
    results/tetnfcov/  results/tnfcov/   -> links to each run's final bins, QC tables, rejected bins, stage tables
    results/runs_comparison.tsv          -> one row per run (bins, Mb, QC passed/failed/unassessed, rejected, ...)

Config keys (flat yaml, read by main.py -> this module):
    all_runs:      true   -> all 7 combinations: tetnfcov tetnf tecov tnfcov te tnf cov
    te_runs:       true   -> the TE set: tetnfcov tetnf te
    modality_runs: [..]   -> explicit list of tags (added to the above), e.g. [tnfcov, tecov]
Tags: te = TE only, tnf = TNF only, cov = COV only, tetnf, tecov, tnfcov, tetnfcov (= TNF+TE+COV, the original pipeline).

Layout:   <outdir>/runs/<tag>/encoder/      <- run_encoder() output for that combination
          <outdir>/runs/<tag>/clustering/   <- run_clustering() output (clusters_deduplicated/, cluster_summary.tsv, ...)
          <outdir>/runs/runs_summary.tsv, runs_manifest.json
          <outdir>/encoder, <outdir>/clustering -> links to runs/tetnfcov/... (only if those paths do not exist yet)

Design (nothing in encoder.py / clustering.py logic is changed):
  * 'tetnfcov' is always encoded first (the "seed"): it trains the three single-modality sub-encoders.
  * every other run COPIES the seed's phase-1 checkpoints for its own modalities (encoder.py fingerprints them per
    modality, so they are reused -> only the cheap fusion step retrains), and
  * STRICT modality isolation (fix): a run NEVER receives a latent of a modality it does not contain. A TE-free run
    gets no latent_te.npy, no TE weights and subcluster_te_mode='off', so TE cannot influence its encoder fusion OR
    its splitting / merging / reassignment. Sub-clustering needs only the TNF + COV latents.
  * a failing run is recorded and skipped; the other runs continue.  Re-running resumes: encoder via its own phase
    checkpoints, clustering via a marker holding a fingerprint of its inputs and ClusteringConfig.
"""
import dataclasses
import filecmp
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from hyphaesbin.utils.logger import get_logger
from hyphaesbin.multirun.verify import missing_clustering_outputs

log = get_logger("modality_runs")


def _record_cached(label, path=None) -> None:
    """Show a skipped (cached) stage in the resource report; profiling problems must never fail a run."""
    try:
        from hyphaesbin.utils.resource_profiler import record_cached
        record_cached(label, path)
    except Exception:
        pass

# tag -> (tnf_include, te_include, cov_include)
COMBO_FLAGS: Dict[str, tuple] = {
    "tetnfcov": (True, True, True),
    "tetnf":    (True, True, False),
    "tecov":    (False, True, True),
    "tnfcov":   (True, False, True),
    "te":       (False, True, False),
    "tnf":      (True, False, False),
    "cov":      (False, False, True),
}
RUN_ORDER = list(COMBO_FLAGS)             # seed first
TE_RUNS = ["tetnfcov", "tetnf", "te"]
DEFAULT_RUNS = ["tetnfcov", "tnfcov"]     # v9 default when nothing else is requested
SEED = "tetnfcov"
MODS = ("tnf", "te", "cov")


def truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("1", "true", "yes", "on")


def select_runs(config: Dict) -> List[str]:
    """Requested run tags in execution order (seed first). Raises ValueError on an unknown tag."""
    req = set()
    if truthy(config.get("all_runs")):
        req |= set(RUN_ORDER)
    if truthy(config.get("te_runs")):
        req |= set(TE_RUNS)
    extra = config.get("modality_runs") or []
    if isinstance(extra, str):
        extra = [x.strip() for x in extra.replace(";", ",").split(",") if x.strip()]
    bad = [x for x in extra if x not in COMBO_FLAGS]
    if bad:
        raise ValueError(f"modality_runs: unknown tag(s) {bad}; valid tags: {RUN_ORDER}")
    req |= set(extra)
    if not req:
        req = set(DEFAULT_RUNS)
    return [r for r in RUN_ORDER if r in req]


def _file_fp(path) -> str:
    if not path:
        return "NONE"
    p = Path(str(path))
    try:
        st = p.stat()
        return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return f"{p.name}:MISSING"


def _content_fp(path) -> str:
    """sha256 of a (small) id-list file; 'NONE' when absent."""
    if not path or not Path(str(path)).exists():
        return "NONE"
    return hashlib.sha256(Path(str(path)).read_bytes()).hexdigest()[:16]


def _fp(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode())
        h.update(b"|")
    fp = h.hexdigest()[:16]
    try:
        from hyphaesbin.utils.checkpoint import register_fp
        register_fp(fp, parts)
    except Exception:
        pass
    return fp


def _marker_reason(marker: Path, fp: str) -> str:
    """Why a run's cache marker does not match (diagnostic only)."""
    if not marker.exists():
        return "no cache marker (never finished in this folder)"
    try:
        from hyphaesbin.utils.checkpoint import fp_change_reason, _FP_PARTS
        m = json.loads(marker.read_text())
        if m.get("fp") == fp:
            return "fingerprint matches"
        return fp_change_reason({"_fp_parts": m.get("fp_parts")} if m.get("fp_parts") is not None else {}, fp)
    except Exception as ex:
        return f"unreadable marker ({type(ex).__name__})"


def _fp_parts(fp: str):
    try:
        from hyphaesbin.utils.checkpoint import _FP_PARTS
        return _FP_PARTS.get(fp)
    except Exception:
        return None


def _write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def _relative_link(link: Path, target: Path) -> None:
    """link -> target using a RELATIVE symlink (survives moving/renaming the whole output tree)."""
    if link.is_symlink() or link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))


def _summarize_clustering(summary_tsv: Path) -> Dict:
    out = {"bins": 0, "total_bp": 0}
    try:
        with open(summary_tsv) as f:
            head = f.readline().rstrip("\n").split("\t")
            ix = {c: i for i, c in enumerate(head)}
            for line in f:
                p = line.rstrip("\n").split("\t")
                if "status" in ix and not p[ix["status"]].startswith("kept"):
                    continue
                out["bins"] += 1
                out["total_bp"] += int(float(p[ix["total_length_bp"]])) if "total_length_bp" in ix else 0
    except (OSError, ValueError, IndexError):
        pass
    return out


def _source_fp(*modules) -> str:
    """sha256 of the SOURCE of the modules that compute a stage: any code correction invalidates that stage's cache."""
    import importlib
    h = hashlib.sha256()
    for m in modules:
        try:
            h.update(Path(importlib.import_module(m).__file__).read_bytes())
        except Exception as ex:                              # never block a run on this
            h.update(f"{m}:{type(ex).__name__}".encode())
    return h.hexdigest()[:16]


def run_modality_runs(config: Dict, outdir, tnf_features_path: Optional[str], tnf_weights_path: Optional[str],
                      te_features_path: Optional[str], te_weights_path: Optional[str], cov_features_path: str,
                      contig_ids_path: str, fasta_path: str, n_samples: Optional[int] = None,
                      contig_classification: Optional[Dict[str, str]] = None,
                      contig_classification_fp: str = "NONE",
                      contig_class_confidence: Optional[Dict[str, float]] = None) -> Dict:
    import hyphaesbin.encoder.encoder as enc_mod
    import hyphaesbin.clustering.clustering as clu_mod

    outdir = Path(outdir)
    runs_root = outdir / "runs"
    requested = select_runs(config)
    result: Dict = {"requested": requested, "runs": {}}
    if requested:
        runs_root.mkdir(parents=True, exist_ok=True)
        missing = [n for n, p in (("tnf_features", tnf_features_path), ("te_features", te_features_path),
                                  ("cov_features", cov_features_path)) if not p or not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"multi-run needs the shared feature files, missing: {missing} "
                                    f"(tnf={tnf_features_path} te={te_features_path} cov={cov_features_path})")
        order = [SEED] + [r for r in requested if r != SEED]
        log.info(f"MULTI-RUN  requested={requested}  (seed run '{SEED}' is always encoded first"
                 f"{'' if SEED in requested else ' but not clustered, it was not requested'})")
        seed_enc = runs_root / SEED / "encoder"
        seed_ok = False

        for tag in order:
            t_inc, e_inc, c_inc = COMBO_FLAGS[tag]
            rdir = runs_root / tag
            enc_dir, clu_dir = rdir / "encoder", rdir / "clustering"
            rdir.mkdir(parents=True, exist_ok=True)
            wanted = tag in requested
            rs = {"tag": tag, "modalities": [m for m, f in zip(MODS, (t_inc, e_inc, c_inc)) if f], "requested": wanted,
                  "encoder": {}, "clustering": {}, "phase1_checkpoints_copied": [], "latents_copied_from_seed": []}
            result["runs"][tag] = rs
            cfg_run = dict(config)
            cfg_run.update(tnf_include=t_inc, te_include=e_inc, cov_include=c_inc, gate_run_name=tag)
            if not e_inc:
                cfg_run["subcluster_te_mode"] = "off"            # strict: TE cannot drive any post-processing
            log.info("")
            log.info(f"=== RUN '{tag}'  modalities={rs['modalities']}  -> {rdir} ===")

            # ---------------- encoder ----------------
            t0 = time.time()
            try:
                if tag != SEED and seed_ok:
                    for m, inc in zip(MODS, (t_inc, e_inc, c_inc)):
                        src, dst = seed_enc / "checkpoints" / m, enc_dir / "checkpoints" / m
                        if inc and src.is_dir() and not dst.exists():
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(src, dst)
                            rs["phase1_checkpoints_copied"].append(m)
                elif tag == SEED:
                    # an earlier single-run pipeline left trained sub-encoders in <outdir>/encoder: reuse the PHASE-1 ones
                    # (encoder.py re-validates each by fingerprint, so a stale one is simply retrained). The fusion
                    # ('joint') checkpoint is deliberately NOT copied.
                    legacy = outdir / "encoder"
                    if legacy.is_dir() and not legacy.is_symlink():
                        for m in MODS:
                            src, dst = legacy / "checkpoints" / m, enc_dir / "checkpoints" / m
                            if src.is_dir() and not dst.exists():
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copytree(src, dst)
                                rs["phase1_checkpoints_copied"].append(f"{m}(from existing outdir/encoder)")
                encoder_cfg = enc_mod.EncoderConfig.from_dict(cfg_run)
                fp_enc = _fp(tag, _file_fp(tnf_features_path) if t_inc else None, _file_fp(te_features_path) if e_inc else None,
                             _file_fp(cov_features_path) if c_inc else None, _file_fp(tnf_weights_path) if t_inc else None,
                             _file_fp(te_weights_path) if e_inc else None, dataclasses.asdict(encoder_cfg), n_samples,
                             _content_fp(cfg_run.get("train_ids_path")),   # subset CONTENT, not just its path
                             _source_fp("hyphaesbin.encoder.encoder"))     # code corrections invalidate the cache
                marker_e, latent = rdir / "encoder_done.json", str(enc_dir / "final_latent.npy")
                enc_cached = (marker_e.exists() and json.loads(marker_e.read_text()).get("fp") == fp_enc
                              and Path(latent).exists() and (enc_dir / "encoder_manifest.json").exists())
                if not enc_cached:
                    log.info(f"[{tag}] encoder not cached: {_marker_reason(marker_e, fp_enc)}")
                if enc_cached:
                    # skip entirely: re-calling run_encoder() rewrites final_latent.npy, which would make clustering's
                    # input fingerprint change and force every finished clustering to be redone on each resume
                    log.info(f"[{tag}] encoder CACHED (inputs and config unchanged)")
                    _record_cached(f"run_encoder/{tag}", latent)
                else:
                    latent = enc_mod.run_encoder(
                        outdir=str(enc_dir),
                        tnf_features_path=tnf_features_path if t_inc else None,
                        cov_features_path=str(cov_features_path) if c_inc else None,
                        te_features_path=te_features_path if e_inc else None,
                        tnf_weights_path=tnf_weights_path if t_inc else None,
                        te_weights_path=te_weights_path if e_inc else None,
                        config=encoder_cfg,
                        n_samples=int(n_samples) if n_samples is not None else None)
                if not Path(latent).exists():
                    raise FileNotFoundError(f"encoder reported {latent} but it does not exist")
                for m, inc in zip(MODS, (t_inc, e_inc, c_inc)):
                    p = enc_dir / f"latent_{m}.npy"
                    if not inc and p.exists():
                        # FIX (strict isolation): remove a latent of a modality this run does not contain (older
                        # versions copied it from the seed run, letting TE influence a 'TE-free' run's post-processing)
                        p.unlink()
                        rs["latents_copied_from_seed"].append(f"{m}(removed: not in this run)")
                still_missing = [m for m, inc in zip(MODS, (t_inc, e_inc, c_inc))
                                 if inc and not (enc_dir / f"latent_{m}.npy").exists()]
                if still_missing:
                    log.warning(f"FLAG:LATENTS_INCOMPLETE run '{tag}': {still_missing} missing -> clustering falls back to the "
                                f"fused latent for sub-clustering/merging in this run")
                if not enc_cached and not still_missing:
                    _write_json(marker_e, {"fp": fp_enc, "final_latent": str(latent), "fp_parts": _fp_parts(fp_enc)})
                rs["encoder"] = {"status": "cached" if enc_cached else "ok", "seconds": round(time.time() - t0, 1), "final_latent": str(latent),
                                 "missing_individual_latents": still_missing}
                if tag == SEED:
                    seed_ok = True
            except KeyboardInterrupt:
                raise
            except BaseException as ex:      # includes SystemExit raised by module code: one failed run must not stop the others
                rs["encoder"] = {"status": "failed", "seconds": round(time.time() - t0, 1), "error": f"{type(ex).__name__}: {ex}",
                                 "traceback_tail": traceback.format_exc().strip().splitlines()[-6:]}
                log.error(f"FLAG:RUN_ENCODER_FAILED '{tag}': {rs['encoder']['error']}")
                _write_json(rdir / "run_status.json", rs)
                continue

            # ---------------- clustering (requested runs only) ----------------
            if wanted:
                t1 = time.time()
                try:
                    clustering_cfg = clu_mod.ClusteringConfig.from_dict(cfg_run)
                    manifest = enc_dir / "encoder_manifest.json"
                    fp_cl = _fp(fp_enc, _file_fp(latent), _file_fp(manifest), _file_fp(contig_ids_path), _file_fp(fasta_path),
                                _file_fp(cov_features_path), _file_fp(tnf_weights_path),
                                _file_fp(te_weights_path) if e_inc else None,
                                [_file_fp(enc_dir / f"latent_{m}.npy") for m in MODS], dataclasses.asdict(clustering_cfg),
                                contig_classification_fp,
                                _source_fp("hyphaesbin.clustering.clustering", "hyphaesbin.clustering.quality_gate"))
                    marker, summary = rdir / "clustering_done.json", clu_dir / "cluster_summary.tsv"
                    _dn = str(cfg_run.get("clusters_deduplicated_dir") or "clusters_deduplicated")
                    _miss = missing_clustering_outputs(clu_dir, _dn) if summary.exists() else ["no summary"]
                    if marker.exists() and json.loads(marker.read_text()).get("fp") == fp_cl and _miss:
                        log.warning(f"[{tag}] clustering cache INCOMPLETE ({_miss[:3]}{' ...' if len(_miss) > 3 else ''}) "
                                    f"-> recomputing")
                    if marker.exists() and not _miss and json.loads(marker.read_text()).get("fp") == fp_cl:
                        log.info(f"[{tag}] clustering CACHED (inputs and config unchanged)")
                        _record_cached(f"run_clustering/{tag}", summary)
                        cs_path, cstat = str(summary), "cached"
                    else:
                        log.info(f"[{tag}] clustering not cached: {_marker_reason(marker, fp_cl)}")
                        cs_path = clu_mod.run_clustering(
                            final_latent_path=str(latent), encoder_manifest_path=str(manifest),
                            contig_ids_path=str(contig_ids_path), fasta_path=str(fasta_path),
                            cov_features_path=str(cov_features_path), outdir=str(clu_dir),
                            tnf_weights_path=tnf_weights_path, te_weights_path=te_weights_path if e_inc else None,
                            alignment_bam_paths=None, contig_classification=contig_classification,
                            contig_class_confidence=contig_class_confidence,
                            config=clustering_cfg)
                        _miss = missing_clustering_outputs(clu_dir, _dn)
                        if _miss:                                   # never mark an incomplete output as done
                            marker.unlink(missing_ok=True)
                            raise RuntimeError(f"clustering finished but outputs are incomplete: {_miss[:5]}")
                        _write_json(marker, {"fp": fp_cl, "cluster_summary": str(cs_path), "fp_parts": _fp_parts(fp_cl)})
                        cstat = "ok"
                    rs["clustering"] = {"status": cstat, "seconds": round(time.time() - t1, 1), "cluster_summary": str(cs_path),
                                        **_summarize_clustering(Path(cs_path))}
                except KeyboardInterrupt:
                    raise
                except BaseException as ex:
                    rs["clustering"] = {"status": "failed", "seconds": round(time.time() - t1, 1), "error": f"{type(ex).__name__}: {ex}",
                                        "traceback_tail": traceback.format_exc().strip().splitlines()[-6:]}
                    log.error(f"FLAG:RUN_CLUSTERING_FAILED '{tag}': {rs['clustering']['error']}")
            else:
                rs["clustering"] = {"status": "not_requested"}
            _write_json(rdir / "run_status.json", rs)

        # compatibility links for tools that expect the flat layout of the single-run pipeline
        if result["runs"].get(SEED, {}).get("encoder", {}).get("status") in ("ok", "cached"):
            _relative_link(outdir / "encoder", runs_root / SEED / "encoder")
        if result["runs"].get(SEED, {}).get("clustering", {}).get("status") in ("ok", "cached"):
            _relative_link(outdir / "clustering", runs_root / SEED / "clustering")

        # summary table
        lines = ["tag\tmodalities\tencoder\tencoder_s\tclustering\tclustering_s\tbins\ttotal_Mb\tphase1_copied\tlatents_copied"]
        for tag in order:
            rs = result["runs"][tag]
            lines.append("\t".join(str(x) for x in (
                tag, "+".join(rs["modalities"]), rs["encoder"].get("status", "-"), rs["encoder"].get("seconds", ""),
                rs["clustering"].get("status", "-"), rs["clustering"].get("seconds", ""), rs["clustering"].get("bins", ""),
                round(rs["clustering"].get("total_bp", 0) / 1e6, 1) if rs["clustering"].get("total_bp") else "",
                ",".join(rs["phase1_checkpoints_copied"]), ",".join(rs["latents_copied_from_seed"]))))
        (runs_root / "runs_summary.tsv").write_text("\n".join(lines) + "\n")
        failed = [t for t in order if result["runs"][t]["encoder"].get("status") == "failed"
                  or result["runs"][t]["clustering"].get("status") == "failed"]
        result["failed"] = failed
        _write_json(runs_root / "runs_manifest.json", result)
        log.info("")
        log.info("MULTI-RUN SUMMARY (runs/runs_summary.tsv)")
        for ln in lines:
            log.info("  " + ln.replace("\t", "  "))
        if failed:
            log.warning(f"FLAG:SOME_RUNS_FAILED {failed} -- see runs/<tag>/run_status.json; the other runs completed")

    if requested:
        # NO automatic winner: export every finished run side by side with a neutral comparison table
        try:
            from hyphaesbin.multirun.compare_runs import export_runs
            result["comparison"] = export_runs(outdir, [t for t in requested
                                                       if result["runs"].get(t, {}).get("clustering", {}).get("status")
                                                       in ("ok", "cached")])
        except KeyboardInterrupt:
            raise
        except BaseException as ex:
            log.error(f"FLAG:RUN_EXPORT_FAILED {type(ex).__name__}: {ex} -- the individual runs are untouched")
            result["comparison"] = {"status": "failed", "error": f"{type(ex).__name__}: {ex}"}

    return result


# PROFILE HOOKS — observation only (resource report); optional, never required.
try:
    from hyphaesbin.utils.resource_profiler import profile_functions
    profile_functions(globals(), ['run_modality_runs'])
except Exception:
    pass
