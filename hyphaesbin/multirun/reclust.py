"""
hyphaesbin.multirun.reclust -- `--reclust`: re-run ONLY clustering (+ quality gate, EukCC QC, export) of a finished
run into a NEW folder, with an updated config and/or updated clustering code. The original run is never modified.

    python main.py --reclust --outdir /home/skkumara/camisim --config camisim_7.yaml          # -> camisim/reclust/run_1
    python main.py --reclust --outdir /home/skkumara/camisim --reclust-name eps0_guard          # -> camisim/reclust/eps0_guard

What happens
  1. <outdir>/reclust/run_N is created (N = next free number; or --reclust-name). Re-using an existing name resumes it.
  2. Config: --config if given, else the config the original run used (<outdir>/run_config.yaml).
  3. Shared features (preprocessing, coverage, TNF, TE) are READ from the original run, never copied or changed.
  4. Encoder: every original encoder folder (runs/<tag>/encoder, or encoder/ for a single run) is COPIED into the new
     run together with its cache marker; the encoder is reused as long as its inputs and encoder settings are unchanged
     (changed encoder settings retrain it inside the new folder only).
  5. Clustering + quality gate + EukCC + export run from scratch inside the new folder.
  6. Log: the new log starts with the original log up to the start of clustering, then the new clustering log.
     Resource report: the original report is copied as resource_report_original.*; the new run writes its own
     resource_report.* for the re-clustering.
  7. reclust_info.json records the source run, the config used and every clustering setting that differs from the
     original run's config.
"""
import dataclasses
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_CLUSTER_START = re.compile(r"HYPHAES CLUSTERING|MULTI-RUN\s+requested=|STEP 6 / 6|=== RUN '")
_ENCODER_START = re.compile(r"STEP 5 / 6|HYPHAES ENCODER")


def _next_run_dir(base: Path, name: Optional[str]) -> Path:
    if name:
        if "/" in name or name in (".", ".."):
            raise ValueError(f"--reclust-name must be a plain folder name, got {name!r}")
        return base / name
    i = 1
    while (base / f"run_{i}").exists():
        i += 1
    return base / f"run_{i}"


def _pick_original_log(src: Path) -> Optional[Path]:
    """Newest log of the original run that reached the encoder or clustering (i.e. contains the pre-clustering part)."""
    logs = sorted((src / "logs").glob("hyphaesbin_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in logs:
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        if _CLUSTER_START.search(txt) or _ENCODER_START.search(txt):
            return p
    return logs[0] if logs else None


def _preclustering_text(log_path: Path) -> str:
    out = []
    with open(log_path, errors="replace") as fh:
        for line in fh:
            if _CLUSTER_START.search(line):
                break
            out.append(line)
    return "".join(out)


def _phase_meta(src: Path, step: str) -> Dict:
    f = src / "checkpoints" / f"{step}.done"
    try:
        return json.loads(f.read_text()).get("metadata", {}) if f.exists() else {}
    except Exception:
        return {}


def _existing(*cands) -> Optional[str]:
    for c in cands:
        if c and Path(str(c)).exists():
            return str(c)
    return None


def locate_inputs(src: Path) -> Dict:
    """Paths of the shared, pre-clustering inputs of the original run (checkpoint metadata first, standard layout second)."""
    m1, m2 = _phase_meta(src, "phase1_preprocessing"), _phase_meta(src, "phase2_coverage")
    m3, m4 = _phase_meta(src, "phase3_tnf"), _phase_meta(src, "phase4_te")
    final = src / "preprocessing" / "final"
    inp = {
        "fasta": _existing(m1.get("final_clean_unmasked"), final / "final_clean_unmasked.fasta"),
        "cov_features": _existing(m2.get("cov_features_path"), src / "coverage" / "coverage_features.npy"),
        "contig_ids": _existing(src / "coverage" / "contig_ids.txt"),
        "cov_manifest": _existing(src / "coverage" / "manifest.json"),
        "tnf_features": _existing(m3.get("features"), src / "tnf_gene" / "tnf_features.npy"),
        "tnf_weights": _existing(m3.get("weights"), src / "tnf_gene" / "tnf_weights.npy"),
        "te_features": _existing(m4.get("features"), src / "te_composition" / "te_features.npy"),
        "te_weights": _existing(m4.get("weights"), src / "te_composition" / "te_weight.npy"),
        "final_dir": str(final),
    }
    missing = [k for k in ("fasta", "cov_features", "contig_ids") if not inp[k]]
    if missing:
        raise FileNotFoundError(f"--reclust: {src} is not a finished HyphaeSBin run (missing {missing}); "
                                f"--outdir must be the ORIGINAL run folder")
    return inp


def _copy_encoders(src: Path, dst: Path) -> List[str]:
    copied = []
    for enc in sorted(src.glob("runs/*/encoder")):
        if enc.is_symlink() or not enc.is_dir():
            continue
        tag = enc.parent.name
        tgt = dst / "runs" / tag / "encoder"
        if not tgt.exists():
            shutil.copytree(enc, tgt, symlinks=True)
        marker = enc.parent / "encoder_done.json"
        if marker.exists() and not (tgt.parent / "encoder_done.json").exists():
            shutil.copy2(marker, tgt.parent / "encoder_done.json")
        copied.append(f"runs/{tag}/encoder")
    enc = src / "encoder"
    if enc.is_dir() and not enc.is_symlink() and not (dst / "encoder").exists():
        shutil.copytree(enc, dst / "encoder", symlinks=True)
        copied.append("encoder")
    return copied


def _copy_resource_reports(src: Path, dst: Path) -> List[str]:
    """Original report kept as resource_report_original.*; a pre-clustering-only TSV is written next to it."""
    out = []
    for f in sorted(src.glob("resource_report.*")):
        tgt = dst / f.name.replace("resource_report", "resource_report_original")
        shutil.copy2(f, tgt)
        out.append(tgt.name)
    tsv = src / "resource_report.tsv"
    if tsv.exists():
        lines = tsv.read_text(errors="replace").splitlines()
        if lines:
            keep = [lines[0]] + [ln for ln in lines[1:]
                                 if not re.search(r"cluster|quality_gate|gate/|eukcc|compare_runs|export_runs", ln, re.I)]
            (dst / "resource_report_original_preclustering.tsv").write_text("\n".join(keep) + "\n")
            out.append("resource_report_original_preclustering.tsv")
    return out


def _cfg_diff(old: Dict, new: Dict, keys) -> Dict:
    return {k: {"original": old.get(k), "reclust": new.get(k)} for k in sorted(keys)
            if k in old and old.get(k) != new.get(k)}


def run_reclust(args) -> Path:
    from hyphaesbin import pipeline as P
    from hyphaesbin.utils.logger import setup_logger, get_logger, get_log_file

    src = Path(args.outdir).resolve()
    if not src.is_dir():
        sys.exit(f"--reclust: original run folder {src} does not exist")
    if src.parent.name == "reclust":
        sys.exit("--reclust: --outdir must be the ORIGINAL run folder, not a reclust/run_N folder")
    cfg_path = args.config or (str(src / "run_config.yaml") if (src / "run_config.yaml").exists() else None)
    if not cfg_path:
        sys.exit(f"--reclust: no --config given and {src}/run_config.yaml does not exist -- pass --config <yaml>")
    if not Path(cfg_path).exists():
        sys.exit(f"--reclust: config {cfg_path} not found")
    inputs = locate_inputs(src)

    new = _next_run_dir(src / "reclust", getattr(args, "reclust_name", None))
    resumed = new.exists()
    new.mkdir(parents=True, exist_ok=True)

    config = P.load_config(cfg_path)
    if args.threads:
        config["threads"] = args.threads
    if config.get("resource_profile_enabled", True):
        try:
            from hyphaesbin.utils.resource_profiler import configure as _cfg_prof
            _cfg_prof(new, config.get("resource_profile_interval_seconds", 1.0), threads=config.get("threads"),
                      count_fasta=False, count_seconds=0)
        except Exception as exc:
            print(f"[resource_profiler] disabled: {type(exc).__name__}: {exc}", file=sys.stderr)

    # ---- log: original pre-clustering part first, then the new clustering log
    orig_log = _pick_original_log(src)
    setup_logger(str(new), args.log_level)
    log = get_logger("reclust")
    if orig_log is not None and not resumed:
        txt = _preclustering_text(orig_log)
        import logging
        for h in logging.getLogger("hyphaesbin").handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()
                h.stream.write(f"\n######## copied from the ORIGINAL run log {orig_log} (up to the start of clustering) "
                               f"########\n{txt}######## end of copied original log -- RE-CLUSTERING starts below "
                               f"########\n\n")
                h.flush()
        (new / "logs" / "original_preclustering.log").write_text(txt)
        # complete copies of every original log, for the record
        (new / "logs" / "original_run").mkdir(parents=True, exist_ok=True)
        for f in sorted((src / "logs").glob("*.log")):
            shutil.copy2(f, new / "logs" / "original_run" / f.name)
    P._apply_length_scale(config, log)
    P._apply_analysis_min_length(config, log)
    shutil.copyfile(cfg_path, new / "run_config.yaml")

    log.info("+" + "=" * 68 + "+")
    log.info("|" + " RE-CLUSTERING (--reclust): clustering + gate + EukCC + export only ".center(68) + "|")
    log.info("+" + "=" * 68 + "+")
    log.info(f"Original run : {src}  (read-only)")
    log.info(f"New run      : {new}{'  (resumed)' if resumed else ''}")
    log.info(f"Config       : {cfg_path}")
    log.info(f"Original log : {orig_log or 'not found'}")
    for k, v in inputs.items():
        log.info(f"  input {k:<13}: {v}")

    copied = _copy_encoders(src, new)
    reports = _copy_resource_reports(src, new) if not resumed else []
    log.info(f"Encoders copied: {copied or 'none (encoder will be trained inside the new run)'}")
    if reports:
        log.info(f"Original resource report copied: {reports}")

    # ---- what differs from the original run's config (clustering + gate keys)
    from hyphaesbin.clustering.clustering import ClusteringConfig
    diff = {}
    if (src / "run_config.yaml").exists():
        old_cfg = P.load_config(str(src / "run_config.yaml"))
        import logging as _lg
        _quiet = _lg.getLogger("reclust_quiet_config_replay")
        _quiet.addHandler(_lg.NullHandler())
        _quiet.propagate = False
        P._apply_length_scale(old_cfg, _quiet)
        P._apply_analysis_min_length(old_cfg, _quiet)
        # compare EFFECTIVE values (yaml + defaults), so a key that is new in this config is reported too
        eff_old = dataclasses.asdict(ClusteringConfig.from_dict(old_cfg))
        eff_new = dataclasses.asdict(ClusteringConfig.from_dict(config))
        eff_old.update({k: old_cfg.get(k) for k in ("modality_runs", "all_runs", "te_runs")})
        eff_new.update({k: config.get(k) for k in ("modality_runs", "all_runs", "te_runs")})
        diff = _cfg_diff(eff_old, eff_new, eff_new.keys())
    log.info(f"Clustering settings changed vs the original run: {len(diff)}")
    for k, v in diff.items():
        log.info(f"  {k}: {v['original']} -> {v['reclust']}")

    contig_cls, contig_conf, contig_fp = P.load_contig_classification(Path(inputs["final_dir"]), log)
    n_samples = None
    if inputs["cov_manifest"]:
        try:
            n_samples = json.loads(Path(inputs["cov_manifest"]).read_text()).get("n_samples")
        except Exception:
            pass
    info = {"source_run": str(src), "new_run": str(new), "config": str(Path(cfg_path).resolve()),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "inputs": inputs, "encoders_copied": copied,
            "clustering_settings_changed": diff, "original_log": str(orig_log) if orig_log else None}
    (new / "reclust_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")

    t0 = time.time()
    tc = lambda v: v is True or str(v).strip().lower() in ("1", "true", "yes", "on")
    multi = tc(config.get("all_runs")) or tc(config.get("te_runs")) or bool(config.get("modality_runs")) \
        or any(src.glob("runs/*/encoder"))
    status = "ok"
    if multi:
        config["tnf_include"] = True
        config["te_include"] = True
        from hyphaesbin.multirun.modality_runs import run_modality_runs
        res = run_modality_runs(config=config, outdir=new,
                                tnf_features_path=inputs["tnf_features"], tnf_weights_path=inputs["tnf_weights"],
                                te_features_path=inputs["te_features"], te_weights_path=inputs["te_weights"],
                                cov_features_path=inputs["cov_features"], contig_ids_path=inputs["contig_ids"],
                                fasta_path=inputs["fasta"], n_samples=n_samples,
                                contig_classification=contig_cls, contig_classification_fp=contig_fp,
                                contig_class_confidence=contig_conf)
        failed = list((res or {}).get("failed") or [])
        if failed or ((res or {}).get("comparison") or {}).get("status") == "failed":
            status = f"failed: {failed}"
    else:
        from hyphaesbin.clustering.clustering import run_clustering
        latent = new / "encoder" / "final_latent.npy"
        if not latent.exists():
            sys.exit(f"--reclust: single-run mode needs {src}/encoder/final_latent.npy (not found)")
        run_clustering(final_latent_path=str(latent), encoder_manifest_path=str(new / "encoder" / "encoder_manifest.json"),
                       contig_ids_path=inputs["contig_ids"], fasta_path=inputs["fasta"],
                       cov_features_path=inputs["cov_features"], outdir=str(new / "clustering"),
                       tnf_weights_path=inputs["tnf_weights"], te_weights_path=inputs["te_weights"],
                       contig_classification=contig_cls, contig_class_confidence=contig_conf,
                       config=ClusteringConfig.from_dict(config))
    info.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), hours=round((time.time() - t0) / 3600, 3),
                status=status, log_file=get_log_file())
    (new / "reclust_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    log.info(f"RE-CLUSTERING {'DONE' if status == 'ok' else status.upper()} in {(time.time() - t0) / 3600:.2f} h -> {new}"
             f"{'  results: ' + str(new / 'results') if multi else ''}  (original run untouched)")
    if status != "ok":
        sys.exit(1)
    return new
