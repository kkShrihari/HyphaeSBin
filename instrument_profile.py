"""Apply small reporting hooks to the uploaded baseline modules."""
from pathlib import Path

root = Path(__file__).resolve().parent

modules = {
    'hyphaesbin/preprocessing/preprocessing.py': [
        'run_preprocessing', *[f'step{i}_{name}' for i, name in [
            (1,'validate_inputs'),(2,'merge_assemblies'),(3,'assembly_stats'),
            (4,'skani_dedup'),(5,'length_prefilter'),(6,'classification'),
            (7,'domain_removal'),(8,'rdna_masking'),(9,'single_mapping'),
            (10,'adaptive_filter'),(11,'multisignal_scoring'),
            (12,'subset_coverage'),(13,'qc_reports')]], '_map_one_sample', 'run_cmd'],
    'hyphaesbin/coverage/coverage.py': [
        'run_coverage_features', 'step1_load_and_validate', 'step1b_reconcile_ids',
        'step2_prevalence_filter', 'step3_normalize', 'step4_neighbors',
        'step5_distance_features', 'step6_concatenate', 'step7_realign'],
    'hyphaesbin/composition/composition/TNF_gene/tnf_gene.py': ['run_tnf_wholecontig'],
    'hyphaesbin/composition/composition/TE_composition/te_composition.py': [
        'run_te_branch', 'run_te_composition', 'run_mmseqs2_search',
        'run_repeatmasker_efficient', 'build_feature_matrix_v22_5d', 'run_cmd'],
    'hyphaesbin/encoder/encoder.py': ['run_encoder', '_phase1_single', '_phase2_joint'],
    'hyphaesbin/clustering/clustering.py': [
        'run_clustering', 'run_hdbscan', '_fit_hdbscan', 'assign_noise_contigs',
        'iterative_refinement', 'subcluster_bins', 'merge_same_genome_bins',
        'recruit_short_contigs', 'filter_bins', 'run_skani_dedup', 'run_eukcc',
        'run_eukcc_merge'],
    'hyphaesbin/modality_runs.py': ['run_modality_runs'],
    'hyphaesbin/best_of_it.py': ['run_best_of_it', '_skani_edges'],
}

for relative, names in modules.items():
    path = root / relative
    source = path.read_text(encoding='utf-8-sig')
    if 'PROFILE HOOKS v1' in source:
        continue
    missing = [name for name in names if f'def {name}(' not in source]
    assert not missing, f'{relative}: {missing}'
    hooks = ('\n\n# PROFILE HOOKS v1 — observation only; scientific functions above are unchanged.\n'
             'from hyphaesbin.utils.resource_profiler import profile_functions, profile_checkpoint\n'
             f'profile_functions(globals(), {names!r})\n')
    if 'def _checkpoint_ok(' in source:
        hooks += '_checkpoint_ok = profile_checkpoint(_checkpoint_ok)\n'
    # In scripts, hooks must be installed before the CLI invokes main().
    marker = 'if __name__ == "__main__":'
    if marker in source:
        source = source.replace(marker, hooks + '\n' + marker, 1)
    else:
        source += hooks
    path.write_text(source, encoding='utf-8')

# The top-level main file configures the sampler before any pipeline work.
path = root / 'main.py'
source = path.read_text(encoding='utf-8-sig')
old = '    outdir = Path(args.outdir)\n    outdir.mkdir(parents=True, exist_ok=True)'
if 'resource_profile_enabled' in source:
    old = None
else:
    assert source.count(old) == 1
if old is not None:
    new = (old + '\n    if config.get("resource_profile_enabled", True):\n'
           '        from hyphaesbin.utils.resource_profiler import configure\n'
           '        configure(outdir, config.get("resource_profile_interval_seconds", 0.5))')
    source = source.replace(old, new)
path.write_text(source, encoding='utf-8')

# Checkpoint completion is a write event, not a cache hit. Fingerprint-aware
# hit/miss events come from each module's existing _checkpoint_ok().
path = root / 'hyphaesbin/utils/checkpoint.py'
source = path.read_text(encoding='utf-8-sig')
if 'PROFILE HOOKS v1' not in source:
    source += '''\n\n# PROFILE HOOKS v1 — report completed work without changing checkpoint semantics.\nfrom hyphaesbin.utils.resource_profiler import cache_event\n_original_mark_done = Checkpoint.mark_done\ndef _profiled_mark_done(self, step, *args, **kwargs):\n    result = _original_mark_done(self, step, *args, **kwargs)\n    cache_event('computed', step)\n    return result\nCheckpoint.mark_done = _profiled_mark_done\n'''
path.write_text(source, encoding='utf-8')

# Phase-one training may use worker processes. Keep a parent interval around
# their dispatch; the sampler measures all concurrently alive descendants.
path = root / 'hyphaesbin/encoder/encoder.py'
source = path.read_text(encoding='utf-8')
old = '\n    t1 = time.time()\n'
assert source.count(old) == 1
source = source.replace(old, "\n    from hyphaesbin.utils.resource_profiler import manual_start, manual_end\n    _phase1_profile = manual_start('encoder/phase1_all')" + old, 1)
old = '\n    log.info(f"Phase 1 complete in {time.time()-t1:.0f}s'
assert source.count(old) == 1
source = source.replace(old, "\n    manual_end(_phase1_profile)" + old, 1)
path.write_text(source, encoding='utf-8')

path = root / 'environment.yaml'
source = path.read_text(encoding='utf-8-sig')
assert '  - psutil' not in source
source = source.replace('  - pandas\n', '  - pandas\n  - psutil            # process-tree RAM, CPU and disk I/O profiler\n', 1)
path.write_text(source, encoding='utf-8')

path = root / 'camisim_final_tetnfcov_multirun.yaml'
source = path.read_text(encoding='utf-8-sig')
source += ('\n# Resource reporting (writes resource_report.tsv/json in the run output).\n'
           'resource_profile_enabled: true\nresource_profile_interval_seconds: 0.5\n')
path.write_text(source, encoding='utf-8')

path = root / 'run_multirun.sh'
source = path.read_text(encoding='utf-8-sig')
old = '$PY main.py --scaffold '
assert source.count(old) == 1
source = source.replace(old, '/usr/bin/time -v -o "$NEW/pipeline_time.txt" "$PY" main.py --scaffold ', 1)
path.write_text(source, encoding='utf-8', newline='\n')

print('Instrumented', len(modules), 'stage modules plus main, checkpoint, environment, YAML and launcher')
