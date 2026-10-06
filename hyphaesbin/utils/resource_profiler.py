"""Low-overhead, best-effort resource measurements for HyphaeSBin.

Rows are hierarchical: a parent interval includes the time and resources of
its children. Do not sum nested wall times to calculate the pipeline total.
"""
from __future__ import annotations

import atexit
import csv
import functools
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # profiling never prevents the biological pipeline from running
    psutil = None

_instance = None
_lock = threading.Lock()
_local = threading.local()


def configure(outdir, interval=0.5):
    """Start sampling once in the main process; workers are included by PID tree."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = Profiler(Path(outdir), interval)
            _instance.pipeline_row = _instance.start('pipeline_total')
            _local.stack = [_instance.pipeline_row]
            atexit.register(_instance.finish)
        return _instance


def current():
    return _instance


def cache_event(status, step=None):
    profiler = _instance
    if profiler is None:
        return
    stack = getattr(_local, 'stack', [])
    if not stack:
        return
    with profiler.lock:
        row = stack[-1]
        row['cache'] = status
        if step:
            row.setdefault('checkpoint_steps', []).append(str(step))


def record_cached(label, output_path=None):
    """Record a stage skipped by an orchestration-level cache check."""
    profiler = _instance
    if profiler is None:
        return
    row = profiler.start(label)
    row['cache'] = 'hit'
    details = {}
    size = _path_stats(output_path)
    if size is not None:
        details['output_bytes'] = size
    profiler.end(row, 'cached', details)


def profile_checkpoint(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        step = kwargs.get('step', args[1] if len(args) > 1 else None)
        cache_event('hit' if result is not None else 'miss', step)
        return result
    return wrapped


def manual_start(label):
    profiler = _instance
    if profiler is None:
        return None
    row = profiler.start(label)
    stack = getattr(_local, 'stack', None)
    if stack is None:
        stack = []
        _local.stack = stack
    stack.append(row)
    return row


def manual_end(row, status='ok'):
    if row is None or _instance is None:
        return
    stack = getattr(_local, 'stack', [])
    if stack and stack[-1] is row:
        stack.pop()
    _instance.end(row, status)


def _arg(name, args, kwargs, position):
    return kwargs.get(name, args[position] if len(args) > position else None)


def _label(name, args, kwargs):
    if name == '_map_one_sample':
        sample = _arg('sample', args, kwargs, 0)
        return f'mapping/{sample}'
    if name == 'run_cmd':
        cmd = _arg('cmd', args, kwargs, 0)
        cmd = str(cmd or '')
        if 'tiara ' in cmd:
            return 'classifier/tiara'
        if 'whokaryote.py ' in cmd:
            return 'classifier/whokaryote'
        if 'coverm ' in cmd:
            return 'mapping/coverm'
        return None  # avoid thousands of low-value shell rows
    if name in ('run_encoder', 'run_clustering'):
        outdir = _arg('outdir', args, kwargs, 0 if name == 'run_encoder' else 5)
        path = Path(str(outdir or ''))
        tag = path.parent.name if path.name in ('encoder', 'clustering') else path.name
        return f'{name}/{tag}'
    if name == '_phase1_single':
        mod = _arg('name', args, kwargs, 0)
        return f'encoder/phase1/{mod}'
    return name


def _path_stats(value):
    """Cheap stat-only size; contigs/bp come from existing stage metadata."""
    if not isinstance(value, (str, os.PathLike)):
        return None
    try:
        path = Path(value)
        return path.stat().st_size if path.is_file() else None
    except (OSError, ValueError):
        return None


def _result_counts(value):
    if isinstance(value, (str, os.PathLike)):
        size = _path_stats(value)
        return {'output_bytes': size} if size is not None else {}
    if not isinstance(value, dict):
        return {}
    result = {}
    for dst, keys in {
        'input_contigs': ('before', 'n_before', 'n_input', 'n_contigs_before'),
        'output_contigs': ('kept', 'after', 'total_kept', 'n_retained', 'n_after', 'n_final', 'n_contigs'),
        'removed_contigs': ('removed', 'n_removed'),
        'input_bp': ('input_bp', 'bp_before', 'total_bp_before'),
        'output_bp': ('output_bp', 'bp_after', 'total_bp', 'retained_bp'),
        'removed_bp': ('removed_bp',),
    }.items():
        for key in keys:
            if isinstance(value.get(key), (int, float)):
                result[dst] = value[key]
                break
    stats = value.get('stats')
    if isinstance(stats, dict):
        for name in ('n_contigs', 'total_bp'):
            if isinstance(stats.get(name), (int, float)):
                result['output_contigs' if name == 'n_contigs' else 'output_bp'] = stats[name]
    for key in ('output', 'masked_fasta', 'unmasked_fasta', 'final_latent',
                'coverage_tsv', 'coverage_table', 'coverage_features_path', 'cluster_summary'):
        size = _path_stats(value.get(key))
        if size is not None:
            result['output_bytes'] = size
            break
    return result


def profile_function(name, function):
    """Wrap an existing function without changing its signature or results."""
    if getattr(function, '_hyphae_profiled', False):
        return function

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        profiler = _instance
        label = _label(name, args, kwargs)
        if profiler is None or label is None:
            return function(*args, **kwargs)
        input_bytes = next((n for n in (_path_stats(v) for v in list(args[:3]) + list(kwargs.values())) if n is not None), None)
        row = profiler.start(label, input_bytes)
        stack = getattr(_local, 'stack', None)
        if stack is None:
            stack = []
            _local.stack = stack
        stack.append(row)
        try:
            result = function(*args, **kwargs)
            partial_failure = isinstance(result, dict) and (
                result.get('status') == 'failed' or bool(result.get('failed')) or
                isinstance(result.get('best_of_it'), dict) and result['best_of_it'].get('status') == 'failed')
            profiler.end(row, 'failed' if partial_failure else 'ok', _result_counts(result))
            return result
        except BaseException as exc:
            profiler.end(row, 'failed', {'error': f'{type(exc).__name__}: {exc}'})
            raise
        finally:
            stack.pop()

    wrapped._hyphae_profiled = True
    return wrapped


def profile_functions(namespace, names):
    """Install optional reporting wrappers around existing stage functions."""
    for name in names:
        function = namespace.get(name)
        if callable(function):
            namespace[name] = profile_function(name, function)


class Profiler:
    def __init__(self, outdir, interval):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.interval = max(0.2, float(interval))
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.rows = []
        self.active = []
        self.stopped = threading.Event()
        self.started = time.monotonic()
        self.process = psutil.Process(os.getpid()) if psutil else None
        self.seen = {}  # (pid, create_time) -> last cumulative CPU and I/O counters
        self.peak_ram = 0
        self.peak_gpu_bytes = None
        self.peak_gpu_util = None
        self.peak_disk_delta = 0
        self.initial_disk_used = self._disk_used()
        self.sample()
        self.thread = threading.Thread(target=self._sample_loop, name='hyphaesbin-resource-profiler', daemon=True)
        self.thread.start()

    def _disk_used(self):
        try:
            usage = shutil.disk_usage(self.outdir)
            return usage.used
        except OSError:
            return None

    def _gpu(self, pids):
        if not shutil.which('nvidia-smi'):
            return None, None
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=2)
            used = 0
            for line in result.stdout.splitlines():
                cols = [v.strip() for v in line.split(',')]
                if len(cols) >= 2 and cols[0].isdigit() and int(cols[0]) in pids and cols[1].isdigit():
                    used += int(cols[1]) * 1024 * 1024
            util = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=2)
            vals = [int(v.strip()) for v in util.stdout.splitlines() if v.strip().isdigit()]
            return used, max(vals) if vals else None  # utilisation is device-wide, not process-specific
        except (OSError, subprocess.TimeoutExpired):
            return None, None

    def sample(self):
        if self.process is None:
            disk = self._disk_used()
            if disk is not None and self.initial_disk_used is not None:
                with self.lock:
                    self.peak_disk_delta = max(self.peak_disk_delta, disk - self.initial_disk_used)
                    for row in self.active:
                        row['peak_disk_used_delta_bytes'] = max(row.get('peak_disk_used_delta_bytes') or 0, self.peak_disk_delta)
            return
        try:
            processes = [self.process] + self.process.children(recursive=True)
        except (psutil.Error, OSError):
            return
        ram = 0
        pids = set()
        with self.lock:
            for proc in processes:
                try:
                    pids.add(proc.pid)
                    key = (proc.pid, proc.create_time())
                    cpu = proc.cpu_times()
                    io = proc.io_counters() if hasattr(proc, 'io_counters') else None
                    ram += proc.memory_info().rss
                    self.seen[key] = (cpu.user + cpu.system,
                                      io.read_bytes if io else 0,
                                      io.write_bytes if io else 0)
                except (psutil.Error, OSError):
                    continue
            self.peak_ram = max(self.peak_ram, ram)
            disk = self._disk_used()
            if disk is not None and self.initial_disk_used is not None:
                self.peak_disk_delta = max(self.peak_disk_delta, disk - self.initial_disk_used)
            for row in self.active:
                row['peak_ram_bytes'] = max(row.get('peak_ram_bytes') or 0, ram)
                row['peak_disk_used_delta_bytes'] = max(row.get('peak_disk_used_delta_bytes') or 0, self.peak_disk_delta)
        # GPU querying is slower; run at most every five seconds.
        now = time.monotonic()
        if now - getattr(self, '_last_gpu', 0) >= 5:
            self._last_gpu = now
            gpu_mem, gpu_util = self._gpu(pids)
            with self.lock:
                if gpu_mem is not None:
                    self.peak_gpu_bytes = max(self.peak_gpu_bytes or 0, gpu_mem)
                    for row in self.active:
                        row['peak_gpu_bytes'] = max(row.get('peak_gpu_bytes') or 0, gpu_mem)
                if gpu_util is not None:
                    self.peak_gpu_util = max(self.peak_gpu_util or 0, gpu_util)
                    for row in self.active:
                        row['peak_gpu_util_device_percent'] = max(row.get('peak_gpu_util_device_percent') or 0, gpu_util)

    def _totals(self):
        return tuple(sum(values[i] for values in self.seen.values()) for i in range(3))

    def _sample_loop(self):
        while not self.stopped.wait(self.interval):
            self.sample()

    def start(self, label, input_bytes=None):
        self.sample()
        with self.lock:
            parent = getattr(_local, 'stack', [])
            parent_name = parent[-1]['stage'] if parent else ''
            if not parent_name and str(label).startswith('mapping/'):
                mapping_parent = next((r for r in reversed(self.active)
                                       if r['stage'] == 'step9_single_mapping'), None)
                parent_name = mapping_parent['stage'] if mapping_parent else ''
            row = {'stage': str(label), 'parent': parent_name,
                   'status': 'running', 'cache': 'unknown', 'started_unix': time.time(),
                   'input_bytes': input_bytes, 'peak_ram_bytes': 0 if self.process else None,
                   '_start': time.monotonic(), '_counters': self._totals()}
            self.active.append(row)
            return row

    def end(self, row, status, details=None):
        self.sample()
        with self.lock:
            index = next((i for i, active_row in enumerate(self.active) if active_row is row), None)
            if index is None:
                return
            self.active.pop(index)
            wall = max(0, time.monotonic() - row.pop('_start'))
            before = row.pop('_counters')
            after = self._totals()
            row.update({'status': status, 'ended_unix': time.time(), 'wall_seconds': round(wall, 3),
                        'cpu_seconds': round(max(0, after[0] - before[0]), 3) if self.process else None,
                        'disk_read_bytes': max(0, after[1] - before[1]) if self.process else None,
                        'disk_write_bytes': max(0, after[2] - before[2]) if self.process else None})
            row['cpu_percent_one_core_scale'] = round(100 * row['cpu_seconds'] / wall, 1) if wall and row['cpu_seconds'] is not None else None
            if details:
                row.update(details)
            if row.get('input_bp') is not None and wall:
                row['throughput_input_mb_per_hour'] = round(row['input_bp'] / 1e6 / (wall / 3600), 3)
            elif row.get('input_contigs') is not None and wall:
                row['throughput_contigs_per_hour'] = round(row['input_contigs'] / (wall / 3600), 1)
            self.rows.append(row)
        self.flush()

    def flush(self):
        with self.write_lock:
            self._flush_locked()

    def _flush_locked(self):
        with self.lock:
            rows = [{k: v for k, v in row.items() if not k.startswith('_')} for row in self.rows]
        report = self.outdir / 'resource_report.json'
        tmp = report.with_suffix('.json.tmp')
        total = next((row.get('wall_seconds') for row in rows if row['stage'] == 'pipeline_total'),
                     round(time.monotonic() - self.started, 3))
        eligible = [row for row in rows if row['stage'] != 'pipeline_total' and
                    row.get('parent') in ('', 'pipeline_total') and row.get('wall_seconds') is not None]
        bottleneck = max(eligible, key=lambda row: row['wall_seconds'])['stage'] if eligible else None
        leaf_rows = [row for row in rows if row['stage'] != 'pipeline_total' and
                     not any(other.get('parent') == row['stage'] for other in rows) and
                     row.get('wall_seconds') is not None and row['status'] != 'cached']
        leaf_bottleneck = max(leaf_rows, key=lambda row: row['wall_seconds'])['stage'] if leaf_rows else None
        for row in rows:
            row['share_of_pipeline_percent'] = round(100 * row['wall_seconds'] / total, 2) if total and row.get('wall_seconds') is not None else None
        data = {'schema_version': 1, 'summary': {'total_wall_seconds': total,
                                                'bottleneck_top_level_stage': bottleneck,
                                                'bottleneck_leaf_stage': leaf_bottleneck,
                                                'peak_ram_bytes': self.peak_ram if self.process else None,
                                                'peak_gpu_bytes': self.peak_gpu_bytes,
                                                'peak_gpu_util_device_percent': self.peak_gpu_util,
                                                'peak_disk_used_delta_bytes': self.peak_disk_delta},
                'measurement_notes': {
            'wall': 'Nested stage wall times overlap; do not sum parent and child rows.',
            'ram': 'Sampled concurrent RSS of main process and live descendants; brief peaks can be missed.',
            'cpu_io': 'Sampled process counters; short-lived descendants may be undercounted.',
            'gpu': 'Memory sums matching compute PIDs; utilization is device-wide; null if unavailable.',
            'disk': 'Peak change in filesystem used bytes since profiling began; includes unrelated writers and is not exact temporary space.',
            'counts': 'Taken from existing stage metadata when provided; null means unavailable, not zero.'},
            'rows': rows}
        try:
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
            os.replace(tmp, report)
            path = self.outdir / 'resource_report.tsv'
            fields = ['stage','parent','status','cache','wall_seconds','share_of_pipeline_percent','cpu_seconds','cpu_percent_one_core_scale',
                      'peak_ram_bytes','peak_gpu_bytes','peak_gpu_util_device_percent','disk_read_bytes','disk_write_bytes',
                      'peak_disk_used_delta_bytes','input_bytes','output_bytes','input_contigs','output_contigs',
                      'removed_contigs','input_bp','output_bp','removed_bp','throughput_input_mb_per_hour',
                      'throughput_contigs_per_hour','error']
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            def fmt(value):
                return 'N/A' if value is None else str(value)
            lines = ['# HyphaeSBin resource report', '',
                     f"Total wall time: {fmt(total)} s",
                     f"Top-level bottleneck: {fmt(bottleneck)}",
                     f"Peak concurrent RAM: {fmt(data['summary']['peak_ram_bytes'])} bytes",
                     f"Peak GPU process memory: {fmt(self.peak_gpu_bytes)} bytes", '',
                     '| Stage | Parent | Wall s | CPU s | Peak RAM bytes | Cache | Status |',
                     '|---|---|---:|---:|---:|---|---|']
            for row in rows:
                lines.append('| ' + ' | '.join(fmt(row.get(key)) for key in
                             ('stage','parent','wall_seconds','cpu_seconds','peak_ram_bytes','cache','status')) + ' |')
            lines.extend(['', 'Nested stage times overlap. GPU utilization is device-wide. '
                          'Disk usage is a filesystem delta, not exact temporary storage. '
                          'Counts absent from stage metadata are N/A.'])
            (self.outdir / 'resource_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        except OSError:
            pass  # reporting must never fail the scientific pipeline

    def finish(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        self.thread.join(timeout=3)
        self.sample()
        pipeline = getattr(self, 'pipeline_row', None)
        if pipeline is not None and any(active_row is pipeline for active_row in self.active):
            failed = any(row.get('status') == 'failed' for row in self.rows)
            self.end(pipeline, 'failed' if failed else 'ok')
        with self.lock:
            pending = list(self.active)
        for row in pending:
            self.end(row, 'interrupted')
        self.flush()
