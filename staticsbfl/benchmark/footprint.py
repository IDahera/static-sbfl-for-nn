"""Process memory and wall-clock probes for the runtime comparison.

Standard library only, on purpose: the same file is imported by the PyTorch
runners (main environment, as ``staticsbfl.benchmark.footprint``) and by both
DeepFault ports (their own TensorFlow environments, which have no torch and
therefore load this file directly, as ``footprint``), so every tool is
measured by identical code. Do not import anything from ``staticsbfl`` here.

What is measured, and why this figure
-------------------------------------
On Apple silicon CPU and GPU share one physical memory. The figure that covers
both is the process **physical footprint** (``phys_footprint``), the number
Activity Monitor shows as "Memory". It includes Metal/MPS buffers the process
allocated; ``ru_maxrss`` (resident set size) does not. Measured on an M4 Pro: a
512 MB MPS tensor raised ``phys_footprint`` by ~600 MB and ``ru_maxrss`` by
9 MB. RSS is therefore recorded only for reference.

Two readings are taken:

``lifetime_max``  the kernel's own high-water mark of the footprint
                  (``ri_lifetime_max_phys_footprint``). Exact, never decreases,
                  so a *phase* peak is the increase of this mark over the phase.
``sampled peak``  a background thread polls the current footprint every few
                  milliseconds within one phase. A lower bound (a spike shorter
                  than the interval is missed), but it attributes a peak to the
                  phase even when an earlier phase was higher.

On Linux the same interface falls back to ``VmHWM``/``VmRSS`` from
``/proc/self/status`` (RSS, not footprint; no GPU memory) and says so in the
``source`` field. Every reading carries that field so a table can state what it
is based on.
"""

import ctypes
import json
import os
import resource
import sys
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

#: Default polling interval of the sampler thread, in seconds.
SAMPLE_INTERVAL = 0.005

#: Environment every measured process is started with. macOS libmalloc keeps
#: freed large blocks in a cache and marks their pages "reusable", which takes
#: them out of phys_footprint; when the process later reuses such a block, its
#: pages are NOT charged again, so the footprint undercounts live memory
#: (measured on an M4 Pro: 300 MB freed by numpy and re-allocated by torch were
#: charged as +0.6 MB). With the large cache off, every large allocation is
#: charged exactly. It must be set before the process starts.
EXACT_MALLOC_ENV = {"MallocLargeCache": "0"}


def exact_malloc() -> bool:
    """Whether this process runs under :data:`EXACT_MALLOC_ENV` (always true off macOS)."""
    return sys.platform != "darwin" or all(os.environ.get(k) == v for k, v in EXACT_MALLOC_ENV.items())

#: How memory is measured, printed at the top of every runtime log and by the
#: evaluation script, so no table is read without it.
MEASUREMENT_NOTE = (
    "Memory: every tool runs once per model in a FRESH process; nothing is shared "
    "between runs. The figure is the process physical footprint (macOS "
    "phys_footprint, the 'Memory' column of Activity Monitor), which on Apple "
    "silicon covers CPU heap AND GPU (Metal/MPS) buffers of the process in the "
    "shared memory; RSS (ru_maxrss) misses GPU buffers and is kept for reference "
    "only. Every measured process starts with MallocLargeCache=0: otherwise "
    "macOS does not re-charge freed-and-reused large blocks and the footprint "
    "undercounts live memory. Per phase two readings: 'peak' = increase of the kernel's exact "
    "lifetime high-water mark over the phase (0 when an earlier phase was "
    "higher), and 'sampled' = highest current footprint polled every "
    f"{SAMPLE_INTERVAL * 1000:.0f} ms within the phase, minus the footprint at its "
    "start (a lower bound). 'process max' = lifetime high-water mark of the whole "
    "process, including interpreter and framework import. For the PyTorch MPS "
    "runs, torch.mps.current_allocated_memory() is recorded as the GPU share; "
    "note that PyTorch's MPS allocator claims memory in heaps (about 1 GB on the "
    "first tensor on an M4 Pro), and the footprint shows the heap, not the tensor. "
    "On Linux the same fields hold RSS (VmHWM/VmRSS) and cannot see GPU memory."
)

_RUSAGE_INFO_V4 = 4


class _RUsageInfoV4(ctypes.Structure):
    """``struct rusage_info_v4`` from <sys/resource.h> (macOS 10.14+)."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
            "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
            "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
            "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime",
            "ri_diskio_bytesread", "ri_diskio_byteswritten",
            "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
            "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
            "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy",
            "ri_interval_max_phys_footprint", "ri_runnable_time",
        )
    ]


_libproc = None
if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError:  # pragma: no cover - libproc is part of every macOS
        _libproc = None

SOURCE = "macos_phys_footprint" if _libproc is not None else "linux_rss"


def _darwin_info() -> _RUsageInfoV4:
    info = _RUsageInfoV4()
    if _libproc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return info


def _linux_status() -> Dict[str, int]:
    values = {}
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(("VmHWM:", "VmRSS:")):
                key, amount, _unit = line.split()
                values[key[:-1]] = int(amount) * 1024
    return values


def current_bytes() -> int:
    """Current footprint (macOS) or RSS (Linux) of this process."""
    if _libproc is not None:
        return int(_darwin_info().ri_phys_footprint)
    return _linux_status()["VmRSS"]


def lifetime_max_bytes() -> int:
    """High-water mark of :func:`current_bytes` since the process started."""
    if _libproc is not None:
        return int(_darwin_info().ri_lifetime_max_phys_footprint)
    return _linux_status()["VmHWM"]


def max_rss_bytes() -> int:
    """``ru_maxrss`` in bytes (bytes on macOS, KiB on Linux). Reference only."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


class _Sampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.peak = current_bytes()
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="footprint-sampler", daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = current_bytes()
            if value > self.peak:
                self.peak = value
            self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "_Sampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak = max(self.peak, current_bytes())


class Recorder:
    """Named, sequential phases of one measured process.

    ``with recorder.phase("analysis"): ...`` records, per phase: wall-clock
    seconds, the footprint before and after, the sampled peak within the phase,
    and the lifetime high-water mark after it. A ``sync`` callable (e.g.
    ``torch.mps.synchronize``) is called at both phase boundaries so that
    asynchronous device work is attributed to the phase that queued it.
    """

    def __init__(self, *, sync=None, interval: float = SAMPLE_INTERVAL) -> None:
        self.sync = sync
        self.interval = interval
        self.phases: List[Dict[str, object]] = []
        self.process_start_bytes = current_bytes()
        self.extra: Dict[str, object] = {}

    @contextmanager
    def phase(self, name: str):
        """Time *name* and record its footprint: before, after, sampled peak, high-water mark."""
        if self.sync is not None:
            self.sync()
        before = current_bytes()
        hwm_before = lifetime_max_bytes()
        started = time.perf_counter()
        with _Sampler(self.interval) as sampler:
            try:
                yield
            finally:
                if self.sync is not None:
                    self.sync()
        seconds = time.perf_counter() - started
        after = current_bytes()
        hwm_after = lifetime_max_bytes()
        self.phases.append({
            "name": name,
            "seconds": seconds,
            "bytes_before": before,
            "bytes_after": after,
            "sampled_peak_bytes": sampler.peak,
            "sampled_peak_above_start_bytes": max(0, sampler.peak - before),
            "lifetime_max_after_bytes": hwm_after,
            "lifetime_max_increase_bytes": max(0, hwm_after - hwm_before),
            "samples": sampler.samples,
        })

    def as_dict(self) -> Dict[str, object]:
        """The record as plain data: memory source, process figures, phases, extras."""
        return {
            "memory_source": SOURCE,
            "sample_interval_seconds": self.interval,
            "process_start_bytes": self.process_start_bytes,
            "process_lifetime_max_bytes": lifetime_max_bytes(),
            "process_max_rss_bytes": max_rss_bytes(),
            "exact_malloc": exact_malloc(),
            "phases": self.phases,
            **self.extra,
        }

    def write(self, path: str) -> None:
        """Write the record as JSON (atomically: the orchestrator polls for it)."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
        os.replace(temporary, path)
