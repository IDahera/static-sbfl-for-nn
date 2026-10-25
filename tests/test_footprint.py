"""The memory probe against known allocations: does it measure what was allocated?

Every runtime figure rests on :mod:`staticsbfl.benchmark.footprint`, so these tests
allocate things of known size -- dense models of several sizes, plain arrays,
transient blocks, re-used blocks, MPS tensors -- inside :class:`Recorder`
phases and compare what the probe reports with the bytes actually allocated.

Each scenario runs in a **fresh child process started with**
:data:`EXACT_MALLOC_ENV`, exactly as ``runtime_benchmark.py`` starts every
measured tool: the footprint of one process must not depend on what an earlier
test in the same interpreter allocated and freed. One test documents why the
environment is needed: without it, macOS does not re-charge a freed and
re-used large block.

Tolerances: a model's footprint is its parameter bytes plus small Python and
allocator overhead, so a measured increase must lie within
``[0.9 x expected, 1.3 x expected + 8 MB]``. Every allocation is written to, so
its pages are resident, which is what a footprint counts.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from staticsbfl.benchmark import footprint
from staticsbfl.benchmark.footprint import EXACT_MALLOC_ENV, Recorder, current_bytes, lifetime_max_bytes
from staticsbfl.utils.paths import PROJECT_ROOT

MB = 2 ** 20
SLACK = 8 * MB

PRELUDE = textwrap.dedent("""
    import gc, json, sys, time
    import numpy as np, torch
    from staticsbfl.benchmark.footprint import Recorder, current_bytes, exact_malloc
    from staticsbfl.evaluation.runtime import per_model_summary
    from staticsbfl.models.architectures import parameter_bytes
    from staticsbfl.variation.builder import VariableMLP
    MB = 2 ** 20
    def dense_model_of(target_bytes):
        inputs = 1024
        width = max(1, target_bytes // 4 // (inputs + 1 + 10))
        return VariableMLP((width,), num_classes=10, input_shape=(inputs,))
    out = {"exact_malloc": exact_malloc()}
""")


def run_child(body: str, *, exact: bool = True) -> dict:
    """Run *body* after :data:`PRELUDE` in a fresh interpreter; it fills ``out``."""
    env = {k: v for k, v in os.environ.items() if k not in EXACT_MALLOC_ENV}
    if exact:
        env.update(EXACT_MALLOC_ENV)
    code = PRELUDE + textwrap.dedent(body) + "\nprint('RESULT' + json.dumps(out))\n"
    result = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT), env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("RESULT"))
    return json.loads(line[len("RESULT"):])


def within(measured: float, expected: float) -> bool:
    return 0.9 * expected <= measured <= 1.3 * expected + SLACK


# --------------------------------------------------------------------------
# The probe itself
# --------------------------------------------------------------------------


def test_the_probe_reads_this_process():
    assert current_bytes() > 0
    assert lifetime_max_bytes() >= current_bytes() * 0.95
    assert footprint.max_rss_bytes() > 0
    assert footprint.SOURCE in ("macos_phys_footprint", "linux_rss")


def test_the_child_processes_run_with_exact_malloc_accounting():
    assert run_child("")["exact_malloc"] is True
    if sys.platform == "darwin":
        assert run_child("", exact=False)["exact_malloc"] is False


def test_the_record_is_written_as_json_and_says_how_it_was_measured(tmp_path):
    recorder = Recorder(interval=0.001)
    with recorder.phase("work"):
        sum(range(1000))
    recorder.extra["tool"] = "probe-test"
    path = tmp_path / "sub" / "record.json"
    recorder.write(str(path))
    record = json.loads(path.read_text())
    assert record["tool"] == "probe-test" and record["memory_source"] == footprint.SOURCE
    assert "exact_malloc" in record and record["phases"][0]["name"] == "work"
    assert record["process_lifetime_max_bytes"] >= record["phases"][0]["bytes_after"] * 0.95
    assert not (tmp_path / "sub" / "record.json.tmp").exists()


# --------------------------------------------------------------------------
# Allocations of known size
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size_mb", [8, 32, 128, 512])
def test_a_model_raises_the_footprint_by_its_parameter_bytes(size_mb):
    out = run_child(f"""
        r = Recorder(interval=0.001)
        with r.phase("build"):
            model = dense_model_of({size_mb} * MB)
        p = r.phases[0]
        out.update(expected=parameter_bytes(model), kept=p["bytes_after"] - p["bytes_before"],
                   sampled=p["sampled_peak_above_start_bytes"], hwm=p["lifetime_max_increase_bytes"])
    """)
    assert out["expected"] == pytest.approx(size_mb * MB, rel=0.02)
    for reading in ("kept", "sampled"):
        assert within(out[reading], out["expected"]), (reading, out[reading] / MB, out["expected"] / MB)
    # The high-water mark only rises by what exceeds the process's earlier peak
    # (import transients), so it is at most the kept increase -- and equal to
    # it once the allocation is larger than those transients.
    assert out["hwm"] <= out["kept"] + SLACK
    if size_mb >= 128:
        assert within(out["hwm"], out["expected"])


def test_model_footprints_grow_in_proportion_to_their_size():
    out = run_child("""
        sizes = {}
        for size_mb in (16, 64, 256):
            r = Recorder(interval=0.001)
            with r.phase("build"):
                model = dense_model_of(size_mb * MB)
            sizes[size_mb] = r.phases[0]["bytes_after"] - r.phases[0]["bytes_before"]
            del model; gc.collect()
        out["sizes"] = sizes
    """)
    sizes = {int(k): v for k, v in out["sizes"].items()}
    assert 3.4 <= sizes[64] / sizes[16] <= 4.4
    assert 3.6 <= sizes[256] / sizes[64] <= 4.2


@pytest.mark.parametrize("size_mb", [16, 96])
def test_a_numpy_array_is_counted_at_its_nbytes_every_time(size_mb):
    out = run_child(f"""
        kept = []
        for _ in range(3):  # the same size three times: each must be charged again
            r = Recorder(interval=0.001)
            with r.phase("array"):
                a = np.ones({size_mb} * MB // 8)
            kept.append(r.phases[0]["bytes_after"] - r.phases[0]["bytes_before"])
            out["nbytes"] = a.nbytes
            del a; gc.collect()
        out["kept"] = kept
    """)
    for value in out["kept"]:
        assert within(value, out["nbytes"]), [v / MB for v in out["kept"]]


def test_a_transient_block_shows_in_the_sampled_peak_but_not_afterwards():
    out = run_child("""
        r = Recorder(interval=0.001)
        with r.phase("transient"):
            block = torch.empty(64 * MB // 4).fill_(1.0)
            for _ in range(20):
                block.add_(0.0)
            del block; gc.collect()
        p = r.phases[0]
        out.update(sampled=p["sampled_peak_above_start_bytes"], kept=p["bytes_after"] - p["bytes_before"],
                   hwm_after=p["lifetime_max_after_bytes"], peak=p["sampled_peak_bytes"])
    """)
    assert within(out["sampled"], 64 * MB)
    assert out["kept"] < 0.25 * 64 * MB
    assert out["hwm_after"] >= out["peak"] * 0.99


def test_a_freed_and_reused_block_is_charged_again():
    out = run_child("""
        a = np.ones(300 * MB // 8); del a; gc.collect()
        r = Recorder(interval=0.001)
        with r.phase("reuse"):
            t = torch.empty(300 * MB // 4).fill_(1.0)
        out["kept"] = r.phases[0]["bytes_after"] - r.phases[0]["bytes_before"]
    """)
    assert within(out["kept"], 300 * MB)


@pytest.mark.skipif(sys.platform != "darwin", reason="documents a macOS libmalloc behaviour")
def test_without_exact_malloc_macos_undercounts_a_reused_block():
    """Why EXACT_MALLOC_ENV exists: this is the error it removes."""
    out = run_child("""
        a = np.ones(300 * MB // 8); del a; gc.collect()
        before = current_bytes()
        t = torch.empty(300 * MB // 4).fill_(1.0)
        out["kept"] = current_bytes() - before
    """, exact=False)
    assert out["kept"] < 0.5 * 300 * MB


def test_the_per_model_summary_attributes_the_analysis_not_the_setup():
    out = run_child("""
        r = Recorder(interval=0.001)
        with r.phase("import"):
            setup = np.ones(32 * MB // 8)                       # stays alive: session state
        with r.phase("model_load"):
            model = dense_model_of(16 * MB)
        with r.phase("spectrum"):
            scratch = torch.empty(48 * MB // 4).fill_(2.0)      # transient analysis memory
            for _ in range(20):
                scratch.mul_(1.0)
            del scratch; gc.collect()
        out.update(per_model_summary(r.phases))
        out["expected"] = parameter_bytes(model) + 48 * MB
        out["phase_seconds"] = [p["seconds"] for p in r.phases]
    """)
    assert out["setup_seconds"] == pytest.approx(out["phase_seconds"][0])
    assert out["per_model_seconds"] == pytest.approx(sum(out["phase_seconds"][1:]))
    assert within(out["per_model_peak_above_start_bytes"], out["expected"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS")
@pytest.mark.skipif(sys.platform != "darwin", reason="phys_footprint is macOS only")
@pytest.mark.parametrize("size_mb", [64, 256])
def test_mps_memory_is_in_the_footprint_and_not_in_rss(size_mb):
    """The footprint shows what the Metal driver allocated for the process.

    PyTorch's MPS allocator claims memory in heaps (about 1 GB on the first real
    tensor on an M4 Pro), so the footprint rises by the heap, not by the tensor:
    it is compared with ``driver_allocated_memory``, the heap, while
    ``current_allocated_memory`` is the tensor. RSS sees neither.
    """
    out = run_child(f"""
        from staticsbfl.benchmark.footprint import max_rss_bytes
        torch.zeros(1, device="mps"); torch.mps.synchronize()   # Metal context outside the phase
        rss_before, gpu_before = max_rss_bytes(), torch.mps.current_allocated_memory()
        driver_before = torch.mps.driver_allocated_memory()
        r = Recorder(sync=torch.mps.synchronize, interval=0.001)
        with r.phase("mps"):
            t = torch.ones({size_mb} * MB // 4, device="mps")
        p = r.phases[0]
        out.update(kept=p["bytes_after"] - p["bytes_before"], gpu=torch.mps.current_allocated_memory() - gpu_before,
                   driver=torch.mps.driver_allocated_memory() - driver_before, rss=max_rss_bytes() - rss_before)
    """)
    assert out["gpu"] >= size_mb * MB                     # the tensor itself
    assert out["driver"] >= size_mb * MB                  # the heap that holds it
    assert within(out["kept"], out["driver"])             # footprint = what the driver allocated
    assert out["rss"] < 0.5 * size_mb * MB                # ru_maxrss does not see GPU buffers
