"""Best-effort resource measurements for HyphaeSBin (observation only; never changes results).

Per stage / sub-stage it records: wall time | CPU seconds, average cores busy and % of the configured threads |
peak RAM of the process tree | peak GPU memory / GPU utilisation | input / output size, contig count and bp |
throughput | disk read / write and peak growth of the stage's own output / temp files | status and cache hit.

Rows are hierarchical: a parent interval contains its children, so never add parent and child wall times.
Rows that run concurrently (e.g. per-sample mapping) are separated by matching their command line, so each gets its
OWN RAM / CPU / disk numbers; their wall times overlap in real time.

Outputs in <outdir>: resource_report.txt (human readable) / .json / .tsv / .md. `python -m hyphaesbin.utils.resource_profiler
report <outdir>` rebuilds them (e.g. after `/usr/bin/time -v` has written pipeline_time.txt).

Works with psutil if installed; otherwise it falls back to /proc (Linux) and says so in the report. Everything is wrapped so a
profiling problem can never stop the pipeline.
"""
from __future__ import annotations

import atexit
import csv
import functools
import glob as _glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except Exception:  # profiling never prevents the biological pipeline from running
    psutil = None

_instance = None
_lock = threading.Lock()
_local = threading.local()
_FASTA_EXT = (".fa", ".fasta", ".fna")
_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


# ------------------------------------------------------------------------------------------------ public API
def configure(outdir, interval=1.0, threads=None, count_fasta=True, count_seconds=300):
    """Start sampling once in the main process. Descendant processes (workers, Tiara, minimap2, EukCC ...) are included."""
    global _instance
    with _lock:
        if _instance is None:
            try:
                _instance = Profiler(Path(outdir), interval, threads, count_fasta, count_seconds)
                _instance.pipeline_row = _instance.start("pipeline_total")
                _local.stack = [_instance.pipeline_row]
                atexit.register(_instance.finish)
            except Exception as exc:                      # never fail the pipeline because of profiling
                print(f"[resource_profiler] disabled: {type(exc).__name__}: {exc}", file=sys.stderr)
                _instance = None
        return _instance


def current():
    return _instance


def _stack():
    s = getattr(_local, "stack", None)
    if s is None:
        s = []
        _local.stack = s
    return s


def cache_event(status, step=None):
    """status: 'hit' | 'miss' | 'computed'. Attributed to the innermost stage of the calling thread (never to the root)."""
    p = _instance
    if p is None:
        return
    try:
        st = _stack()
        if not st:
            return
        row = st[-1]
        if row is getattr(p, "pipeline_row", None) and len(st) == 1:
            return
        with p.lock:
            key = {"hit": "cache_hits", "miss": "cache_misses", "computed": "checkpoints_computed"}.get(status)
            if key:
                row[key] = row.get(key, 0) + 1
            if step:
                row.setdefault("checkpoint_steps", []).append(str(step))
    except Exception:
        pass


def record_cached(label, output_path=None):
    """Record a stage skipped by an orchestration-level cache check."""
    p = _instance
    if p is None:
        return
    try:
        row = p.start(label)
        row["cache_hits"] = 1
        details = {}
        size = _path_stats(output_path)
        if size is not None:
            details["output_bytes"] = size
        p.end(row, "cached", details)
    except Exception:
        pass


def profile_checkpoint(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        try:
            step = kwargs.get("step", args[1] if len(args) > 1 else None)
            cache_event("hit" if result is not None else "miss", step)
        except Exception:
            pass
        return result
    return wrapped


def manual_start(label):
    p = _instance
    if p is None:
        return None
    try:
        row = p.start(label)
        _stack().append(row)
        return row
    except Exception:
        return None


def manual_end(row, status="ok"):
    p = _instance
    if row is None or p is None:
        return
    try:
        st = _stack()
        if any(r is row for r in st):
            while st and st[-1] is not row:        # unwind rows leaked by an exception between start and end
                p.end(st.pop(), "interrupted")
            st.pop()
        p.end(row, status)
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------ labelling / hints
def _arg(name, args, kwargs, position):
    return kwargs.get(name, args[position] if len(args) > position else None)


def _label(name, args, kwargs):
    if name == "_map_one_sample":
        return f"mapping/{_arg('sample', args, kwargs, 0)}"
    if name == "run_cmd":
        cmd = str(_arg("cmd", args, kwargs, 0) or "")
        if "tiara " in cmd:
            return "classifier/tiara"
        if "whokaryote.py " in cmd:
            return "classifier/whokaryote"
        if "coverm " in cmd:
            return "mapping/coverm"
        return None                               # avoid thousands of low-value shell rows
    if name in ("run_encoder", "run_clustering"):
        outdir = _arg("outdir", args, kwargs, 0 if name == "run_encoder" else 5)
        path = Path(str(outdir or ""))
        tag = path.parent.name if path.name in ("encoder", "clustering") else path.name
        return f"{name}/{tag}"
    if name in ("run_eukcc", "run_eukcc_merge"):
        return f"{name}/{_arg('label', args, kwargs, 3) or 'eukcc'}"
    if name == "_phase1_single":
        return f"encoder/phase1/{_arg('name', args, kwargs, 0)}"
    return name


def _is_path(v):
    return isinstance(v, (str, os.PathLike))


def _path_stats(value):
    """Cheap stat-only size of a file."""
    if not _is_path(value):
        return None
    try:
        path = Path(value)
        return path.stat().st_size if path.is_file() else None
    except (OSError, ValueError):
        return None


def _hints(name, args, kwargs):
    """(input_bytes, input_fasta, input_npy, track_marker, watch_paths) taken from the call arguments."""
    vals = list(args[:4]) + list(kwargs.values())
    input_bytes = next((n for n in (_path_stats(v) for v in vals) if n is not None), None)
    fasta = next((str(v) for v in vals if _is_path(v) and str(v).lower().endswith(_FASTA_EXT)), None)
    npy = next((str(v) for v in vals if _is_path(v) and str(v).lower().endswith(".npy")), None)
    track, watch = None, []
    if name == "_map_one_sample":
        sample, map_dir = _arg("sample", args, kwargs, 0), _arg("map_dir", args, kwargs, 7)
        if map_dir is not None and sample is not None:
            bam = str(Path(str(map_dir)) / f"{sample}.bam")
            track, watch = bam, [bam + "*"]          # the BAM and samtools-sort spill files (<bam>.tmp.NNNN.bam)
        sizes = [_path_stats(_arg("r1", args, kwargs, 1)), _path_stats(_arg("r2", args, kwargs, 2))]
        input_bytes = sum(s for s in sizes if s) or input_bytes
        fasta = None
    else:
        out = kwargs.get("outdir", None)
        if out is None and name in ("run_encoder",) and args:
            out = args[0]
        if _is_path(out) and name != "run_preprocessing":
            watch = [str(out)]
        elif name.startswith("step") and len(args) > 1 and _is_path(args[1]) and Path(str(args[1])).is_dir():
            watch = [str(args[1])]
    return input_bytes, fasta, npy, track, watch


def _npy_rows(path):
    try:
        import numpy as np
        return int(np.load(path, mmap_mode="r").shape[0])
    except Exception:
        return None


def _result_counts(value):
    if isinstance(value, tuple) and len(value) > 1 and _is_path(value[1]):          # (sample, bam, ok, detail)
        size = _path_stats(value[1])
        return {"output_bytes": size} if size is not None else {}
    if _is_path(value):
        size = _path_stats(value)
        return {"output_bytes": size} if size is not None else {}
    if not isinstance(value, dict):
        return {}
    result = {}
    for dst, keys in {
        "input_contigs": ("before", "n_before", "n_input", "n_contigs_before"),
        "output_contigs": ("kept", "after", "total_kept", "n_retained", "n_after", "n_final", "n_contigs"),
        "removed_contigs": ("removed", "n_removed"),
        "input_bp": ("input_bp", "bp_before", "total_bp_before"),
        "output_bp": ("output_bp", "bp_after", "total_bp", "retained_bp"),
        "removed_bp": ("removed_bp",),
    }.items():
        for key in keys:
            if isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool):
                result[dst] = value[key]
                break
    stats = value.get("stats")
    if isinstance(stats, dict):
        for nm in ("n_contigs", "total_bp"):
            if isinstance(stats.get(nm), (int, float)):
                result["output_contigs" if nm == "n_contigs" else "output_bp"] = stats[nm]
    for key in ("output", "masked_fasta", "unmasked_fasta", "final_latent", "coverage_tsv", "coverage_table",
                "coverage_features_path", "cluster_summary"):
        size = _path_stats(value.get(key))
        if size is not None:
            result["output_bytes"] = size
            break
    for key in ("output", "unmasked_fasta", "masked_fasta"):
        v = value.get(key)
        if _is_path(v) and str(v).lower().endswith(_FASTA_EXT):
            result["_output_fasta"] = str(v)
            break
    return result


def profile_function(name, function):
    """Wrap an existing function without changing its signature, result or exceptions."""
    if getattr(function, "_hyphae_profiled", False):
        return function

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        profiler = _instance
        try:
            label = _label(name, args, kwargs) if profiler is not None else None
        except Exception:
            label = None
        if profiler is None or label is None:
            return function(*args, **kwargs)
        try:
            in_bytes, fasta, npy, track, watch = _hints(name, args, kwargs)
            row = profiler.start(label, in_bytes, track=track, watch=watch)
            if fasta:
                row["_input_fasta"] = fasta
            elif npy:
                n = _npy_rows(npy)
                if n is not None:
                    row["input_contigs"] = n
        except Exception:
            return function(*args, **kwargs)
        stack = _stack()
        stack.append(row)
        try:
            result = function(*args, **kwargs)
            try:
                partial_failure = isinstance(result, dict) and (
                    result.get("status") == "failed" or bool(result.get("failed")))
                _unwind(profiler, stack, row)
                profiler.end(row, "failed" if partial_failure else "ok", _result_counts(result))
            except Exception:
                pass
            return result
        except BaseException as exc:
            try:
                _unwind(profiler, stack, row)
                profiler.end(row, "failed", {"error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
            raise
        finally:
            if any(r is row for r in stack):
                while stack and stack[-1] is not row:
                    stack.pop()
                stack.pop()

    wrapped._hyphae_profiled = True
    return wrapped


def _unwind(profiler, stack, row):
    """End rows leaked by an exception inside the wrapped call (e.g. manual_start without manual_end)."""
    while stack and stack[-1] is not row:
        profiler.end(stack.pop(), "interrupted")


def profile_functions(namespace, names):
    """Install optional reporting wrappers around existing stage functions."""
    for name in names:
        function = namespace.get(name)
        if callable(function):
            namespace[name] = profile_function(name, function)


# ------------------------------------------------------------------------------------------------ process backends
class _PsutilBackend:
    name = "psutil"

    def __init__(self):
        self.me = psutil.Process(os.getpid())

    def tree(self):
        try:
            return [self.me] + self.me.children(recursive=True)
        except (psutil.Error, OSError):
            return [self.me]

    @staticmethod
    def info(proc, need_cmd=False):
        try:
            cpu = proc.cpu_times()
            try:
                io = proc.io_counters()
                rd, wr = io.read_bytes, io.write_bytes
            except (psutil.Error, OSError, AttributeError):
                rd = wr = 0
            d = {"key": (proc.pid, proc.create_time()), "pid": proc.pid, "ppid": proc.ppid(), "cpu": cpu.user + cpu.system,
                 "rd": rd, "wr": wr, "rss": proc.memory_info().rss, "cmd": ""}
            if need_cmd:
                try:
                    d["cmd"] = " ".join(proc.cmdline())
                except (psutil.Error, OSError):
                    pass
            return d
        except (psutil.Error, OSError):
            return None


class _ProcBackend:
    name = "/proc"

    def __init__(self):
        self.me = os.getpid()
        if not os.path.isdir("/proc/self"):
            raise OSError("no /proc")

    def tree(self):
        children = {}
        try:
            names = os.listdir("/proc")
        except OSError:
            return [self.me]
        for n in names:
            if n.isdigit():
                try:
                    with open(f"/proc/{n}/stat", "rb") as f:
                        s = f.read()
                    children.setdefault(int(s[s.rfind(b")") + 2:].split()[1]), []).append(int(n))
                except (OSError, ValueError, IndexError):
                    continue
        out, stack = [], [self.me]
        while stack:
            p = stack.pop()
            out.append(p)
            stack.extend(children.get(p, ()))
        return out

    @staticmethod
    def info(pid, need_cmd=False):
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                s = f.read()
            fields = s[s.rfind(b")") + 2:].split()
            cpu = (int(fields[11]) + int(fields[12])) / _CLK
            d = {"key": (pid, int(fields[19])), "pid": pid, "ppid": int(fields[1]), "cpu": cpu, "rd": 0, "wr": 0, "rss": 0, "cmd": ""}
            with open(f"/proc/{pid}/statm") as f:
                d["rss"] = int(f.read().split()[1]) * _PAGE
            try:
                with open(f"/proc/{pid}/io") as f:
                    for line in f:
                        if line.startswith("read_bytes:"):
                            d["rd"] = int(line.split()[1])
                        elif line.startswith("write_bytes:"):
                            d["wr"] = int(line.split()[1])
            except OSError:
                pass
            if need_cmd:
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        d["cmd"] = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
                except OSError:
                    pass
            return d
        except (OSError, ValueError, IndexError):
            return None


def _make_backend():
    if psutil is not None:
        try:
            return _PsutilBackend()
        except Exception:
            pass
    try:
        return _ProcBackend()
    except Exception:
        return None


def _descendants(children, roots):
    out, stack = set(), list(roots)
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.add(p)
        stack.extend(children.get(p, ()))
    return out


def _entry_size(spec) -> int:
    """Size in bytes of a file, a directory tree or a glob pattern."""
    try:
        total = 0
        for p in (_glob.glob(spec) if any(c in spec for c in "*?[") else [spec]):
            if os.path.isfile(p):
                total += os.path.getsize(p)
            elif os.path.isdir(p):
                stack, n = [p], 0
                while stack and n < 300000:
                    d = stack.pop()
                    with os.scandir(d) as it:
                        for e in it:
                            n += 1
                            try:
                                if e.is_dir(follow_symlinks=False):
                                    stack.append(e.path)
                                else:
                                    total += e.stat(follow_symlinks=False).st_size
                            except OSError:
                                pass
        return total
    except OSError:
        return 0


# ------------------------------------------------------------------------------------------------ the profiler
class Profiler:
    def __init__(self, outdir, interval, threads=None, count_fasta=True, count_seconds=300):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.interval = max(0.2, float(interval))
        self.threads = int(threads) if threads else (os.cpu_count() or 1)
        self.count_fasta, self.count_seconds = bool(count_fasta), float(count_seconds)
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.rows, self.active = [], []
        self.stopped = threading.Event()
        self.started = time.monotonic()
        self.backend = _make_backend()
        self.process = self.backend            # kept for compatibility with older hooks
        self.me = os.getpid()
        self.peak_ram = 0
        self.peak_gpu_bytes = self.peak_gpu_util = None
        self.peak_disk_delta = 0
        self.initial_disk_used = self._disk_used()
        self._last_gpu = 0.0
        self._snap = {}                        # pid -> latest info
        self.host = self._host_info()
        self.pipeline_row = None
        self._tl_prev = {}                     # pid key -> (cpu, rd, wr) for the timeline deltas
        self._tl_t = None
        self._tl = None
        try:                                   # step-by-step timeline: one line per sample (RAM/CPU/IO over time)
            self._tl = open(self.outdir / "resource_timeline.tsv", "w", buffering=1)
            self._tl.write("unix_time\telapsed_s\tstage\tactive_stages\trss_total_gb\tcpu_cores_busy\t"
                           "read_mb_per_s\twrite_mb_per_s\tn_processes\n")
        except OSError:
            self._tl = None
        if self.backend is None:
            print("[resource_profiler] WARNING: neither psutil nor /proc is available -> RAM, CPU and disk I/O are NOT measured "
                  "(install psutil: conda install psutil)", file=sys.stderr)
        elif self.backend.name == "/proc":
            print("[resource_profiler] psutil not installed: using the /proc fallback (Linux). Install psutil for the full feature set.",
                  file=sys.stderr)
        self.sample()
        self.thread = threading.Thread(target=self._sample_loop, name="hyphaesbin-resource-profiler", daemon=True)
        self.thread.start()

    # ---- host / disk / gpu
    def _host_info(self):
        info = {"cpu_count": os.cpu_count(), "python": platform.python_version(), "platform": platform.platform(),
                "backend": self.backend.name if self.backend else None, "threads_configured": self.threads}
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    info["mem_total_gb"] = round(int(line.split()[1]) / 1048576, 1)
                    break
        except OSError:
            pass
        if shutil.which("nvidia-smi"):
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=10)
                info["gpu"] = [x.strip() for x in r.stdout.strip().splitlines()] if r.returncode == 0 and r.stdout.strip() else None
            except (OSError, subprocess.TimeoutExpired):
                info["gpu"] = None
        else:
            info["gpu"] = None
        return info

    def _disk_used(self):
        try:
            return shutil.disk_usage(self.outdir).used
        except OSError:
            return None

    def _gpu(self, pids):
        if not shutil.which("nvidia-smi"):
            return None, None
        try:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=2)
            used = 0
            for line in result.stdout.splitlines():
                cols = [v.strip() for v in line.split(",")]
                if len(cols) >= 2 and cols[0].isdigit() and int(cols[0]) in pids and cols[1].isdigit():
                    used += int(cols[1]) * 1024 * 1024
            util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=2)
            vals = [int(v.strip()) for v in util.stdout.splitlines() if v.strip().isdigit()]
            return used, max(vals) if vals else None      # utilisation is device-wide, not process-specific
        except (OSError, subprocess.TimeoutExpired):
            return None, None

    # ---- sampling
    def sample(self):
        try:
            self._sample()
        except Exception:
            pass

    def _sample(self):
        disk = self._disk_used()
        with self.lock:
            if disk is not None and self.initial_disk_used is not None:
                self.peak_disk_delta = max(self.peak_disk_delta, disk - self.initial_disk_used)
            actives = list(self.active)
        if self.backend is None:
            for row in actives:
                row["peak_disk_used_delta_bytes"] = max(row.get("peak_disk_used_delta_bytes") or 0, self.peak_disk_delta)
            return
        need_cmd = any(r.get("_track") for r in actives)
        snaps = {}
        for proc in self.backend.tree():
            info = self.backend.info(proc, need_cmd)
            if info:
                snaps[info["pid"]] = info
        children = {}
        for pid, i in snaps.items():
            children.setdefault(i["ppid"], []).append(pid)
        now = time.monotonic()
        with self.lock:
            self._snap = snaps
            total_rss = sum(i["rss"] for i in snaps.values())
            self.peak_ram = max(self.peak_ram, total_rss)
            self._timeline(now, snaps, total_rss)
            for row in self.active:
                if row.get("_track"):
                    roots = [p for p, i in snaps.items() if row["_track"] in i["cmd"] and p != self.me]
                    rs = set(roots)
                    top = [p for p in roots if i_parent_free(p, snaps, rs)]
                    members = _descendants(children, top)
                else:
                    members = set(snaps)
                _r = sum(snaps[p]["rss"] for p in members if p in snaps)
                row["peak_ram_bytes"] = max(row.get("peak_ram_bytes") or 0, _r)
                row["_ram_n"] = row.get("_ram_n", 0) + 1
                row["_ram_sum"] = row.get("_ram_sum", 0) + _r
                row["_ram_min"] = _r if row.get("_ram_min") is None else min(row["_ram_min"], _r)
                row.setdefault("_ram_first", _r)
                row["_ram_last"] = _r
                _rs = row.setdefault("_ram_s", [])
                _rs.append(_r)
                if len(_rs) > 4000:                 # bounded memory: keep every 2nd sample
                    del _rs[::2]
                row["peak_processes"] = max(row.get("peak_processes") or 0, len(members))
                last, base = row["_last"], row["_base"]
                cpu = rd = wr = 0.0
                for p in members:
                    i = snaps.get(p)
                    if not i:
                        continue
                    prev = last.get(i["key"], base.get(i["key"], (0.0, 0, 0)))
                    cur = (i["cpu"], i["rd"], i["wr"])
                    cpu += max(0.0, cur[0] - prev[0])
                    rd += max(0, cur[1] - prev[1])
                    wr += max(0, cur[2] - prev[2])
                    last[i["key"]] = cur
                row["_cpu"] += cpu
                row["_rd"] += rd
                row["_wr"] += wr
                row["peak_disk_used_delta_bytes"] = max(row.get("peak_disk_used_delta_bytes") or 0, self.peak_disk_delta)
                self._sample_watch(row, now)
        if now - self._last_gpu >= 5:
            self._last_gpu = now
            gpu_mem, gpu_util = self._gpu(set(snaps))
            with self.lock:
                for row in self.active:
                    if gpu_mem is not None:
                        row["peak_gpu_bytes"] = max(row.get("peak_gpu_bytes") or 0, gpu_mem)
                        self.peak_gpu_bytes = max(self.peak_gpu_bytes or 0, gpu_mem)
                    if gpu_util is not None:
                        row["peak_gpu_util_device_percent"] = max(row.get("peak_gpu_util_device_percent") or 0, gpu_util)
                        row.setdefault("_gpu_utils", []).append(gpu_util)
                        self.peak_gpu_util = max(self.peak_gpu_util or 0, gpu_util)

    def _timeline(self, now, snaps, total_rss):
        if self._tl is None:
            return
        try:
            cpu = rd = wr = 0.0
            cur = {}
            for i in snaps.values():
                c = (i["cpu"], i["rd"], i["wr"])
                cur[i["key"]] = c
                prev = self._tl_prev.get(i["key"])
                if prev is not None:
                    cpu += max(0.0, c[0] - prev[0])
                    rd += max(0, c[1] - prev[1])
                    wr += max(0, c[2] - prev[2])
            dt = (now - self._tl_t) if self._tl_t else None
            self._tl_prev, self._tl_t = cur, now
            names = [r["stage"] for r in self.active]
            inner = names[-1] if names else ""
            if dt:
                self._tl.write(f"{time.time():.1f}\t{now - self.started:.1f}\t{inner}\t{'|'.join(names[-4:])}\t"
                               f"{total_rss / 1073741824:.3f}\t{cpu / dt:.2f}\t{rd / dt / 1e6:.2f}\t{wr / dt / 1e6:.2f}\t"
                               f"{len(snaps)}\n")
        except Exception:
            pass

    def _sample_watch(self, row, now, force=False):
        if not row.get("_watch"):
            return
        if not force and now - row.get("_watch_t", 0) < max(5.0, 5 * self.interval):
            return
        row["_watch_t"] = now
        size = sum(_entry_size(w) for w in row["_watch"])
        if row.get("_watch0") is None:
            row["_watch0"] = size
        row["peak_watched_bytes"] = max(row.get("peak_watched_bytes") or 0, size)
        row["peak_watched_growth_bytes"] = max(0, row["peak_watched_bytes"] - row["_watch0"])

    def _sample_loop(self):
        while not self.stopped.wait(self.interval):
            self.sample()

    # ---- rows
    def start(self, label, input_bytes=None, track=None, watch=None):
        self.sample()
        with self.lock:
            parent = _stack()
            parent_name = parent[-1]["stage"] if parent else ""
            if not parent_name and str(label).startswith("mapping/"):
                mp = next((r for r in reversed(self.active) if r["stage"] == "step9_single_mapping"), None)
                parent_name = mp["stage"] if mp else ""
            base = {} if track else {i["key"]: (i["cpu"], i["rd"], i["wr"]) for i in self._snap.values()}
            row = {"stage": str(label), "parent": parent_name, "status": "running", "cache": "unknown", "started_unix": time.time(),
                   "input_bytes": input_bytes, "peak_ram_bytes": 0 if self.backend else None,
                   "_start": time.monotonic(), "_base": base, "_last": {}, "_cpu": 0.0, "_rd": 0, "_wr": 0,
                   "_track": track, "_watch": list(watch or []), "_watch_t": 0, "_watch0": None}
            self.active.append(row)
            self._sample_watch(row, time.monotonic(), force=True)
            return row

    def end(self, row, status, details=None):
        self.sample()
        with self.lock:
            index = next((i for i, r in enumerate(self.active) if r is row), None)
            if index is None:
                return
            self._sample_watch(row, time.monotonic(), force=True)
            self.active.pop(index)
            wall = max(0.0, time.monotonic() - row.pop("_start"))
            has = self.backend is not None
            cpu, rd, wr = row.pop("_cpu"), row.pop("_rd"), row.pop("_wr")
            for k in ("_base", "_last", "_track", "_watch", "_watch_t", "_watch0"):
                row.pop(k, None)
            _rs = row.pop("_ram_s", None)
            _n, _sum = row.pop("_ram_n", 0), row.pop("_ram_sum", 0)
            _mn, _f, _l = row.pop("_ram_min", None), row.pop("_ram_first", None), row.pop("_ram_last", None)
            if _n and has:
                srt = sorted(_rs or [])
                q = (lambda p: srt[min(len(srt) - 1, int(p * (len(srt) - 1) + 0.5))]) if srt else (lambda p: None)
                row.update({"ram_mean_bytes": int(_sum / _n), "ram_min_bytes": _mn, "ram_p50_bytes": q(0.5),
                            "ram_p95_bytes": q(0.95), "ram_at_start_bytes": _f, "ram_at_end_bytes": _l,
                            "ram_samples": _n})
            utils = row.pop("_gpu_utils", None)
            hits, misses = row.get("cache_hits", 0), row.get("cache_misses", 0)
            cache = "partial" if hits and misses else "hit" if hits else "miss" if misses else row.get("cache", "unknown")
            row.update({"status": status, "cache": cache, "ended_unix": time.time(), "wall_seconds": round(wall, 3),
                        "cpu_seconds": round(cpu, 3) if has else None,
                        "disk_read_bytes": int(rd) if has else None, "disk_write_bytes": int(wr) if has else None,
                        "threads_configured": self.threads})
            if has and wall:
                row["cpu_percent_one_core_scale"] = round(100 * cpu / wall, 1)
                row["avg_cores_busy"] = round(cpu / wall, 2)
                row["cpu_util_percent_of_threads"] = round(100 * cpu / wall / self.threads, 1)
            else:
                row["cpu_percent_one_core_scale"] = row["avg_cores_busy"] = row["cpu_util_percent_of_threads"] = None
            if utils:
                row["gpu_util_mean_device_percent"] = round(sum(utils) / len(utils), 1)
            if details:
                row.update(details)
            self.rows.append(row)
        self.flush()

    # ---- lazy contig/bp counting (kept out of the timed intervals)
    def _count_fasta(self, path, cache):
        try:
            p = Path(path)
            st = p.stat()
            key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
            if key in cache:
                return cache[key]
            n = bp = 0
            carry = b""
            hdr = re.compile(rb"\n>[^\n]*")
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    chunk = carry + chunk
                    cut = chunk.rfind(b"\n")
                    if cut < 0:
                        carry = chunk
                        continue
                    block, carry = chunk[:cut + 1], chunk[cut + 1:]
                    nb = b"\n" + block
                    n += nb.count(b"\n>")
                    rest = hdr.sub(b"", nb)
                    bp += len(rest) - rest.count(b"\n") - rest.count(b"\r")
            if carry:
                nb = b"\n" + carry
                n += nb.count(b"\n>")
                rest = hdr.sub(b"", nb)
                bp += len(rest) - rest.count(b"\n") - rest.count(b"\r")
            cache[key] = {"contigs": n, "bp": bp}
            return cache[key]
        except Exception:
            return None

    def _fill_counts(self):
        if not self.count_fasta:
            return
        cj = self.outdir / "fasta_counts_cache.json"
        try:
            cache = json.loads(cj.read_text()) if cj.exists() else {}
        except Exception:
            cache = {}
        t0, skipped = time.monotonic(), 0
        with self.lock:
            rows = list(self.rows)
        for row in rows:
            for src, dst_n, dst_bp in (("_input_fasta", "input_contigs", "input_bp"), ("_output_fasta", "output_contigs", "output_bp")):
                path = row.get(src)
                if not path or (row.get(dst_n) is not None and row.get(dst_bp) is not None):
                    continue
                if time.monotonic() - t0 > self.count_seconds:
                    skipped += 1
                    continue
                c = self._count_fasta(path, cache)
                if c:
                    if row.get(dst_n) is None:
                        row[dst_n] = c["contigs"]
                    if row.get(dst_bp) is None:
                        row[dst_bp] = c["bp"]
        if skipped:
            self.count_skipped = skipped
        try:
            cj.write_text(json.dumps(cache))
        except OSError:
            pass

    @staticmethod
    def _throughput(row):
        wall = row.get("wall_seconds")
        if not wall or row.get("status") == "cached":
            return
        if row.get("input_bp"):
            row["throughput_input_mb_per_hour"] = round(row["input_bp"] / 1e6 / (wall / 3600), 1)
        if row.get("input_contigs"):
            row["throughput_contigs_per_hour"] = round(row["input_contigs"] / (wall / 3600), 1)
        if row.get("input_bytes"):
            row["throughput_input_gb_per_hour"] = round(row["input_bytes"] / 1e9 / (wall / 3600), 3)

    # ---- reporting
    def flush(self):
        try:
            with self.write_lock:
                self._flush_locked()
        except Exception:
            pass

    def _flush_locked(self):
        with self.lock:
            rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in self.rows]
        write_reports(self.outdir, rows, self._summary_inputs())

    def _summary_inputs(self):
        return {"host": self.host, "peak_ram": self.peak_ram if self.backend else None, "peak_gpu_bytes": self.peak_gpu_bytes,
                "peak_gpu_util": self.peak_gpu_util, "peak_disk_delta": self.peak_disk_delta,
                "elapsed": round(time.monotonic() - self.started, 3), "count_skipped": getattr(self, "count_skipped", 0),
                "count_fasta": self.count_fasta}

    def finish(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        try:
            self.thread.join(timeout=3)
        except Exception:
            pass
        self.sample()
        pipeline = self.pipeline_row
        if pipeline is not None and any(r is pipeline for r in self.active):
            failed = any(r.get("status") == "failed" for r in self.rows)
            self.end(pipeline, "failed" if failed else "ok")
        with self.lock:
            pending = list(self.active)
        for row in pending:
            self.end(row, "interrupted")
        try:
            self._fill_counts()
        except Exception:
            pass
        with self.lock:
            for row in self.rows:
                self._throughput(row)
        self.flush()
        try:
            if self._tl is not None:
                self._tl.close()
        except Exception:
            pass


def i_parent_free(pid, snaps, matched):
    """True when none of pid's ancestors is itself a matched (tracked) process -> pid is a topmost match."""
    seen, cur = 0, snaps[pid]["ppid"]
    while cur in snaps and seen < 64:
        if cur in matched:
            return False
        cur, seen = snaps[cur]["ppid"], seen + 1
    return True


# ------------------------------------------------------------------------------------------------ report writing
_FAMILY = (
    ("coverage", ("run_coverage_features", "step1_load_and_validate", "step1b_reconcile_ids", "step2_prevalence_filter", "step3_normalize",
                  "step4_neighbors", "step5_distance_features", "step6_concatenate", "step7_realign")),
    ("tnf", ("run_tnf_wholecontig",)),
    ("te", ("run_te_branch", "run_te_composition", "run_mmseqs2_search", "build_feature_matrix_v22_5d")),
    ("encoder", ("run_encoder", "encoder/")),
    ("clustering", ("run_clustering", "run_hdbscan", "_fit_hdbscan", "assign_noise_contigs", "iterative_refinement", "subcluster_bins",
                    "merge_same_genome_bins", "recruit_short_contigs", "filter_bins", "run_skani_dedup")),
    ("multi_run", ("run_modality_runs",)),
    ("preprocessing", ("run_preprocessing", "mapping/", "classifier/")),
)


def _own_phase(stage):
    for fam, names in _FAMILY:
        for n in names:
            if stage == n or stage.startswith(n if n.endswith("/") else n + "/"):
                return fam
    if re.match(r"step\d+[a-z]?_", stage):
        return "preprocessing"
    return None


def _phase(row, by_name):
    cur, guard = row, 0
    while cur is not None and guard < 20:
        if cur["stage"].startswith(("run_eukcc",)):
            cur = by_name.get(cur.get("parent"))
            guard += 1
            continue
        p = _own_phase(cur["stage"])
        if p:
            return p
        cur, guard = by_name.get(cur.get("parent")), guard + 1
    return "other"


def parse_gnu_time(outdir):
    for name in ("pipeline_time.txt", "profiling/time_total.txt", "time_total.txt"):
        f = Path(outdir) / name
        if not f.exists():
            continue
        out = {"file": str(f)}
        try:
            for ln in open(f, errors="ignore"):
                ln = ln.strip()
                if ln.startswith("Elapsed (wall clock)"):
                    parts = [float(x) for x in ln.rsplit(": ", 1)[1].split(":")]
                    out["wall_seconds"] = sum(x * m for x, m in zip(reversed(parts), (1, 60, 3600)))
                elif ln.startswith("Maximum resident set size"):
                    out["max_rss_gb_single_process"] = round(int(ln.split(":")[1]) / 1048576, 2)
                elif ln.startswith("Percent of CPU"):
                    out["cpu_percent"] = ln.split(":")[1].strip()
                elif ln.startswith("File system inputs"):
                    out["fs_inputs_512B_blocks"] = int(ln.split(":")[1])
                elif ln.startswith("File system outputs"):
                    out["fs_outputs_512B_blocks"] = int(ln.split(":")[1])
                elif ln.startswith("Exit status"):
                    out["exit_status"] = int(ln.split(":")[1])
        except (OSError, ValueError):
            pass
        return out if len(out) > 1 else None
    return None


_TSV_FIELDS = ["stage", "parent", "phase", "status", "cache", "wall_seconds", "share_of_pipeline_percent", "cpu_seconds", "avg_cores_busy",
               "cpu_util_percent_of_threads", "peak_ram_bytes", "ram_mean_bytes", "ram_p50_bytes", "ram_p95_bytes",
               "ram_min_bytes", "ram_at_start_bytes", "ram_at_end_bytes", "ram_samples", "peak_processes", "peak_gpu_bytes", "peak_gpu_util_device_percent",
               "gpu_util_mean_device_percent", "disk_read_bytes", "disk_write_bytes", "peak_watched_growth_bytes", "peak_disk_used_delta_bytes",
               "input_bytes", "output_bytes", "input_contigs", "output_contigs", "removed_contigs", "input_bp", "output_bp", "removed_bp",
               "throughput_input_mb_per_hour", "throughput_contigs_per_hour", "throughput_input_gb_per_hour", "error"]


def _gb(v):
    return "n/a" if v is None else f"{v / 1073741824:.2f}"


def _hms(s):
    if s is None:
        return "n/a"
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def write_reports(outdir, rows, si):
    outdir = Path(outdir)
    by_name = {r["stage"]: r for r in rows}
    total = next((r.get("wall_seconds") for r in rows if r["stage"] == "pipeline_total"), si.get("elapsed"))
    for r in rows:
        r["phase"] = _phase(r, by_name)
        w = r.get("wall_seconds")
        r["share_of_pipeline_percent"] = round(100 * w / total, 2) if total and w is not None else None
    parents = {r.get("parent") for r in rows}
    measured = [r for r in rows if r["stage"] != "pipeline_total" and r.get("wall_seconds") is not None and r["status"] != "cached"]
    leaf = [r for r in measured if r["stage"] not in parents]
    top = [r for r in measured if r.get("parent") in ("", "pipeline_total")]
    b_leaf = max(leaf, key=lambda r: r["wall_seconds"], default=None)
    b_top = max(top, key=lambda r: r["wall_seconds"], default=None)
    peak = max((r for r in rows if r["stage"] != "pipeline_total"), key=lambda r: r.get("peak_ram_bytes") or 0, default=None)
    phase_wall = {}
    for r in leaf:
        phase_wall[r["phase"]] = phase_wall.get(r["phase"], 0.0) + r["wall_seconds"]
    gnu = parse_gnu_time(outdir)
    summary = {"total_wall_seconds": total, "bottleneck_top_level_stage": b_top["stage"] if b_top else None,
               "bottleneck_leaf_stage": b_leaf["stage"] if b_leaf else None,
               "bottleneck_leaf_share_percent": b_leaf.get("share_of_pipeline_percent") if b_leaf else None,
               "peak_ram_bytes": si.get("peak_ram"), "peak_ram_stage": peak["stage"] if peak and peak.get("peak_ram_bytes") else None,
               "peak_gpu_bytes": si.get("peak_gpu_bytes"), "peak_gpu_util_device_percent": si.get("peak_gpu_util"),
               "peak_disk_used_delta_bytes": si.get("peak_disk_delta"), "leaf_wall_by_phase_seconds": phase_wall,
               "host": si.get("host"), "gnu_time_whole_run": gnu}
    notes = {
        "wall": "Elapsed time of the stage incl. waiting and I/O. Nested rows overlap (a parent contains its children) and rows that run in parallel overlap in real time: never add them up.",
        "ram": "Sampled RSS summed over the stage's process tree (this process + descendants). RSS counts pages shared by forked workers once per process. A peak shorter than the sampling interval can be missed.",
        "cpu": "CPU seconds of the stage's processes from sampled counters; avg_cores_busy = cpu/wall; cpu_util% is relative to the configured thread count. CPU of a short-lived child that ends between two samples is partly missed.",
        "disk": "disk_read/write = bytes that reached the storage device (per-process counters, sampled). peak_watched_growth = peak growth of the stage's own output/temp files (e.g. BAM + samtools spill). peak_disk_used_delta = change of used bytes on the output filesystem since profiling began; it includes other users' writes and is context only.",
        "gpu": "GPU memory = sum over this process tree's compute PIDs (nvidia-smi); utilisation is device-wide; n/a without a GPU.",
        "counts": "Contigs/bp come from stage metadata, feature-matrix shapes or a FASTA count done once at exit (cached); n/a means unavailable, not zero.",
        "parallel": "Per-sample mapping rows are separated by matching each sample's own command line, so their RAM/CPU/disk are per sample.",
    }
    if si.get("count_skipped"):
        notes["counts_skipped"] = f"{si['count_skipped']} FASTA count(s) skipped (time budget)."
    data = {"schema_version": 2, "summary": summary, "measurement_notes": notes, "rows": rows}
    try:
        tmp = outdir / "resource_report.json.tmp"
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, outdir / "resource_report.json")
        with (outdir / "resource_report.tsv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=_TSV_FIELDS, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        (outdir / "resource_report.txt").write_text(_text_report(rows, summary, leaf, notes, parents), encoding="utf-8")
        (outdir / "resource_report.md").write_text("```\n" + (outdir / "resource_report.txt").read_text(encoding="utf-8") + "```\n", encoding="utf-8")
    except OSError:
        pass                                               # reporting must never fail the scientific pipeline


def _text_report(rows, s, leaf, notes, parents):
    host = s.get("host") or {}
    L = ["HYPHAESBIN RESOURCE REPORT", "=" * 110,
         f"host: {host.get('cpu_count')} CPUs | {host.get('mem_total_gb')} GB RAM | GPU: {host.get('gpu') or 'none detected'} | "
         f"python {host.get('python')} | process backend: {host.get('backend') or 'NONE (RAM/CPU/IO not measured)'} | threads configured: {host.get('threads_configured')}",
         f"pipeline wall time (profiled session): {_hms(s['total_wall_seconds'])}"]
    g = s.get("gnu_time_whole_run")
    if g:
        L.append(f"whole run (GNU time -v): wall {_hms(g.get('wall_seconds'))} | CPU {g.get('cpu_percent')} | max RSS of the largest single process "
                 f"{g.get('max_rss_gb_single_process')} GB | fs in/out (512B blocks) {g.get('fs_inputs_512B_blocks')}/{g.get('fs_outputs_512B_blocks')} | exit {g.get('exit_status')}")
    b = next((r for r in leaf if r["stage"] == s.get("bottleneck_leaf_stage")), None)
    if b:
        L.append(f"BOTTLENECK (leaf stage): {b['stage']} -> {_hms(b['wall_seconds'])} = {b.get('share_of_pipeline_percent')} % of pipeline wall | "
                 f"{b.get('avg_cores_busy')} cores busy on average ({b.get('cpu_util_percent_of_threads')} % of {b.get('threads_configured')} threads) | peak RAM {_gb(b.get('peak_ram_bytes'))} GB")
    L.append(f"top-level bottleneck: {s.get('bottleneck_top_level_stage')}")
    L.append(f"PEAK RAM (process tree RSS): {_gb(s.get('peak_ram_bytes'))} GB" + (f" in stage {s['peak_ram_stage']}" if s.get("peak_ram_stage") else "")
             + f" | peak GPU memory: {_gb(s.get('peak_gpu_bytes'))} GB | peak GPU util (device): {s.get('peak_gpu_util_device_percent')}")
    L += ["", "PER PHASE (sum of leaf-stage wall; parallel rows can make a phase exceed its elapsed time)"]
    tot = s["total_wall_seconds"] or 1
    for ph, w in sorted((s.get("leaf_wall_by_phase_seconds") or {}).items(), key=lambda kv: -kv[1]):
        L.append(f"  {ph:<16}{_hms(w):>10}  {100 * w / tot:6.1f} %")
    L += ["", "TOP 10 LEAF STAGES BY WALL TIME",
          f"  {'stage':<46}{'wall':>10}{'%pipe':>7}{'cores':>7}{'cpu%thr':>8}{'RAM GB':>8}{'read GB':>8}{'write GB':>9}"]
    for r in sorted(leaf, key=lambda r: -r["wall_seconds"])[:10]:
        L.append(f"  {r['stage'][:45]:<46}{_hms(r['wall_seconds']):>10}{(r.get('share_of_pipeline_percent') or 0):>7.1f}{(r.get('avg_cores_busy') or 0):>7.1f}"
                 f"{(r.get('cpu_util_percent_of_threads') or 0):>8.0f}{_gb(r.get('peak_ram_bytes')):>8}{_gb(r.get('disk_read_bytes')):>8}{_gb(r.get('disk_write_bytes')):>9}")
    L += ["", "ALL STAGES   (* = parent: contains its children; cache: hit / miss / partial / unknown)",
          f"  {'stage':<46}{'status':<11}{'cache':<8}{'wall':>10}{'%pipe':>7}{'cpu_s':>9}{'cores':>6}{'RAM GB':>8}{'GPU GB':>7}{'GPU%':>5}"
          f"{'in GB':>7}{'out GB':>7}{'contigs':>11}{'Mb/h':>9}{'rd GB':>7}{'wr GB':>7}{'tmp GB':>7}"]
    for r in rows:
        mark = "*" if r["stage"] in parents else " "
        L.append(f"  {(r['stage'] + mark)[:46]:<46}{r['status'][:10]:<11}{r.get('cache', '')[:7]:<8}{_hms(r.get('wall_seconds')):>10}"
                 f"{(r.get('share_of_pipeline_percent') if r.get('share_of_pipeline_percent') is not None else 'n/a'):>7}"
                 f"{(round(r['cpu_seconds']) if r.get('cpu_seconds') is not None else 'n/a'):>9}{(r.get('avg_cores_busy') if r.get('avg_cores_busy') is not None else 'n/a'):>6}"
                 f"{_gb(r.get('peak_ram_bytes')):>8}{_gb(r.get('peak_gpu_bytes')):>7}{(r.get('gpu_util_mean_device_percent') if r.get('gpu_util_mean_device_percent') is not None else 'n/a'):>5}"
                 f"{_gb(r.get('input_bytes')):>7}{_gb(r.get('output_bytes')):>7}"
                 f"{(int(r['input_contigs']) if r.get('input_contigs') is not None else 'n/a'):>11}{(r.get('throughput_input_mb_per_hour') if r.get('throughput_input_mb_per_hour') is not None else 'n/a'):>9}"
                 f"{_gb(r.get('disk_read_bytes')):>7}{_gb(r.get('disk_write_bytes')):>7}{_gb(r.get('peak_watched_growth_bytes')):>7}")
    L += ["", "RAM STEP BY STEP (GB; sampled RSS of the stage's process tree: start -> mean / p50 / p95 / peak -> end)",
          f"  {'stage':<46}{'start':>7}{'mean':>7}{'p50':>7}{'p95':>7}{'peak':>7}{'end':>7}{'cores':>7}{'rd GB':>7}{'wr GB':>7}"]
    for r in rows:
        if r.get("ram_samples"):
            L.append(f"  {r['stage'][:45]:<46}{_gb(r.get('ram_at_start_bytes')):>7}{_gb(r.get('ram_mean_bytes')):>7}"
                     f"{_gb(r.get('ram_p50_bytes')):>7}{_gb(r.get('ram_p95_bytes')):>7}{_gb(r.get('peak_ram_bytes')):>7}"
                     f"{_gb(r.get('ram_at_end_bytes')):>7}{(r.get('avg_cores_busy') if r.get('avg_cores_busy') is not None else 'n/a'):>7}"
                     f"{_gb(r.get('disk_read_bytes')):>7}{_gb(r.get('disk_write_bytes')):>7}")
    L.append("  full time series (every sample): resource_timeline.tsv")
    fails = [r for r in rows if r["status"] in ("failed", "interrupted")]
    if fails:
        L += ["", "FAILED / INTERRUPTED"] + [f"  {r['stage']}: {r['status']} {r.get('error') or ''}" for r in fails]
    L += ["", "MEASUREMENT NOTES"] + [f"  - {v}" for v in notes.values()]
    return "\n".join(L) + "\n"


def rebuild(outdir):
    """Rewrite resource_report.* from resource_report.json (e.g. after GNU time wrote pipeline_time.txt)."""
    f = Path(outdir) / "resource_report.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    s = d.get("summary", {})
    si = {"host": s.get("host"), "peak_ram": s.get("peak_ram_bytes"), "peak_gpu_bytes": s.get("peak_gpu_bytes"),
          "peak_gpu_util": s.get("peak_gpu_util_device_percent"), "peak_disk_delta": s.get("peak_disk_used_delta_bytes"),
          "elapsed": s.get("total_wall_seconds")}
    write_reports(Path(outdir), d["rows"], si)
    return Path(outdir) / "resource_report.txt"


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "report":
        p = rebuild(argv[1])
        print(p if p else f"no resource_report.json under {argv[1]}")
        return 0 if p else 1
    print("usage: python -m hyphaesbin.utils.resource_profiler report <outdir>")
    return 2


if __name__ == "__main__":
    sys.exit(_main())
