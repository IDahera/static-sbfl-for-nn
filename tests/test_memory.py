"""Does the memory accounting report the sizes we can compute by hand?

Three layers of claim, checked separately because they are worth different
amounts:

**Analytical.** ``tensor_bytes`` and ``counter_bytes`` are arithmetic on a shape
and a dtype. They are exact on every backend and are checked against sizes
written out by hand, small and large, so a dtype-width or factor-of-two mistake
cannot hide.

**Measured, device.** A :class:`MemoryProbe` around a known allocation must
report that allocation. On MPS this is checked exactly for sizes the allocator
does not have to round, and as a bounded over-report for one that it does --
the probe may never report *less* than was allocated.

**Measured, host.** On the CPU there is nothing exact to assert, and this file
says so rather than inventing a tolerance: ``ru_maxrss`` is a process-wide
high-water mark that never decreases, so it cannot attribute a peak to a block.
Only the contract is checked -- the fidelity label, which fields are populated,
and monotonicity.

CUDA is not tested against real hardware. Everything here was developed on an
Apple-silicon Mac, which has no CUDA device, so the driver-backed figures cannot
be produced or validated here. What *can* be tested without hardware is our own
branch -- that the CUDA path resets the peak counter, reads it back and labels
the result ``"exact"`` -- and that is done below against a stubbed
``torch.cuda``. It validates this module's logic, not PyTorch's accounting; a
container with a real CUDA device would be needed for the latter and is left as
future work.
"""

import sys

import pytest
import torch

from staticsbfl.benchmark.resources import (
    FIDELITIES, MemoryProbe, MemoryReading, _max_rss_bytes, counter_bytes, tensor_bytes,
)

KIB = 1024
MIB = 1024 * 1024

#: Sizes chosen so the expected byte count is obvious by inspection, and so that
#: both a small allocation (well under a page-table's worth) and a large one
#: (tens of megabytes, where an allocator's behaviour actually shows) are covered.
#: Powers of two: verified to be allocated exactly on MPS, without rounding.
ALIGNED_CASES = [
    pytest.param((16, 16), torch.float32, 16 * 16 * 4, id="tiny-f32-1KiB"),
    pytest.param((256, 256), torch.float32, 256 * 256 * 4, id="small-f32-256KiB"),
    pytest.param((256, 256), torch.int32, 256 * 256 * 4, id="small-i32-256KiB"),
    pytest.param((512, 512), torch.float16, 512 * 512 * 2, id="small-f16-512KiB"),
    pytest.param((1024, 1024), torch.float32, 1024 * 1024 * 4, id="large-f32-4MiB"),
    pytest.param((2048, 1024), torch.int64, 2048 * 1024 * 8, id="large-i64-16MiB"),
    pytest.param((4096, 4096), torch.float32, 4096 * 4096 * 4, id="large-f32-64MiB"),
]


# --------------------------------------------------------------------------
# Analytical: arithmetic on a shape and a dtype
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape,dtype,expected", ALIGNED_CASES)
def test_tensor_bytes_matches_the_hand_computed_size(shape, dtype, expected):
    assert tensor_bytes(torch.zeros(shape, dtype=dtype)) == expected


def test_tensor_bytes_is_spelled_out_for_the_documented_reference_sizes():
    """The four sizes quoted in the documentation, written out rather than derived."""
    assert tensor_bytes(torch.zeros(16, 16, dtype=torch.float32)) == 1 * KIB
    assert tensor_bytes(torch.zeros(256, 256, dtype=torch.float32)) == 256 * KIB
    assert tensor_bytes(torch.zeros(1024, 1024, dtype=torch.float32)) == 4 * MIB
    assert tensor_bytes(torch.zeros(4096, 4096, dtype=torch.float32)) == 64 * MIB


def test_tensor_bytes_counts_the_view_not_the_storage_behind_it():
    """A deliberate choice: a spectrum keeps its counters, not what they look into."""
    storage = torch.zeros(1024, 1024, dtype=torch.float32)
    view = storage[:16, :16]
    assert tensor_bytes(storage) == 4 * MIB
    assert tensor_bytes(view) == 1 * KIB


@pytest.mark.parametrize("shape,dtype,expected", ALIGNED_CASES)
def test_counter_bytes_is_exactly_two_counters_of_that_size(shape, dtype, expected):
    """a_f and a_s, nothing else -- n_f and n_s are derived and cost nothing."""
    assert counter_bytes([shape], dtype=dtype) == 2 * expected


def test_counter_bytes_for_the_reference_models():
    """Written out for the shapes the two fixed dense models produce."""
    # SmallMLP: 256 + 128 hidden units -> two int32 counters each.
    assert counter_bytes([(256,), (128,)]) == 2 * 384 * 4 == 3072

    # WideMLP: 2048 + 1024 + 512 hidden units.
    assert counter_bytes([(2048,), (1024,), (512,)]) == 2 * 3584 * 4 == 28_672


def test_counter_bytes_of_nothing_is_nothing():
    assert counter_bytes([]) == 0
    assert counter_bytes([()]) == 2 * 4  # a scalar unit is still one counter pair


def test_counter_bytes_scales_with_the_counter_dtype():
    """int16 counters halve the counter bytes -- and nothing else, as documented."""
    shape = [(1024, 1024)]
    assert counter_bytes(shape, dtype=torch.int16) == counter_bytes(shape) // 2
    assert counter_bytes(shape, dtype=torch.int64) == counter_bytes(shape) * 2


# --------------------------------------------------------------------------
# Measured on MPS
# --------------------------------------------------------------------------

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs an Apple MPS device"
)


@requires_mps
@pytest.mark.parametrize("shape,dtype,expected", ALIGNED_CASES)
def test_mps_probe_reports_a_known_allocation_exactly(shape, dtype, expected):
    """For a power-of-two size the MPS allocator does not round, so this is exact.

    The tensor outlives the probe deliberately. A block that is still live when
    the probe exits is caught by the final read regardless of where the sampler
    happened to be; one that is allocated and freed inside the block relies on
    the sampler landing on it, which is a separate and weaker claim -- see
    :func:`test_mps_sampling_can_miss_a_transient_shorter_than_its_interval`.
    """
    probe = MemoryProbe(torch.device("mps"))
    with probe:
        held = torch.zeros(shape, dtype=dtype, device="mps")
        torch.mps.synchronize()
    assert tensor_bytes(held) == expected
    assert probe.reading.fidelity == "sampled"
    assert probe.reading.peak_above_baseline_bytes == expected
    del held


@requires_mps
def test_mps_never_under_reports_a_size_the_allocator_has_to_round():
    """4,000,000 bytes is not a round block; MPS rounds it up by ~1.6%.

    The direction matters more than the amount. A probe that reported *less*
    than was allocated would make every memory claim in the cost study an
    underestimate; one that reports slightly more is the allocator's overhead
    showing through, which is information rather than error. The exact rounding
    granularity is an unspecified implementation detail and is deliberately not
    encoded here -- only the bound is.
    """
    expected = 1000 * 1000 * 4
    probe = MemoryProbe(torch.device("mps"))
    with probe:
        held = torch.zeros(1000, 1000, dtype=torch.float32, device="mps")
        torch.mps.synchronize()
    measured = probe.reading.peak_above_baseline_bytes
    assert measured >= expected
    assert measured <= expected * 1.05 + MIB
    del held


@requires_mps
def test_mps_reports_the_sum_of_several_live_allocations():
    """Three live 4 MiB blocks are 12 MiB, not 4 -- the peak is the live total."""
    probe = MemoryProbe(torch.device("mps"))
    with probe:
        held = [torch.zeros(1024, 1024, device="mps") for _ in range(3)]
        torch.mps.synchronize()
    assert probe.reading.peak_above_baseline_bytes == 3 * 4 * MIB
    del held


@requires_mps
def test_mps_sampling_can_miss_a_transient_shorter_than_its_interval():
    """The documented limitation, made concrete rather than left as a warning.

    A block allocated and freed well inside one sampling interval leaves nothing
    for the final read to find, so the sampled peak can come back at zero. That
    is why the figure is labelled a *lower bound* and why the driver allocation
    is reported next to it: the driver total is monotone and does register the
    demand even when the sampler misses it.

    The assertion is deliberately weak in one direction only -- the sampler may
    under-report, never over-report -- because how much it catches depends on
    thread scheduling and must not make the suite flaky.
    """
    probe = MemoryProbe(torch.device("mps"), sample_interval=1.0)
    with probe:
        transient = torch.zeros(1024, 1024, device="mps")
        del transient  # freed long before the one-second sampler wakes
    reading = probe.reading
    assert reading.peak_above_baseline_bytes <= 4 * MIB
    assert reading.fidelity == "sampled", "the label is what makes this readable"


@requires_mps
def test_mps_records_the_sampling_evidence_alongside_the_peak():
    """A sampled figure must carry what it was sampled from, or it is unreadable."""
    probe = MemoryProbe(torch.device("mps"))
    with probe:
        held = torch.zeros(1024, 1024, device="mps")
        torch.mps.synchronize()
    reading = probe.reading
    assert reading.fidelity in FIDELITIES
    assert reading.samples > 0
    assert reading.sample_interval > 0
    assert reading.driver_delta_bytes is not None
    assert reading.baseline_device_bytes is not None
    del held


@requires_mps
def test_an_empty_mps_block_reports_no_new_demand():
    probe = MemoryProbe(torch.device("mps"))
    with probe:
        pass
    assert probe.reading.peak_above_baseline_bytes == 0


# --------------------------------------------------------------------------
# Measured on the CPU -- the contract only
# --------------------------------------------------------------------------


def test_cpu_reports_the_process_high_water_and_no_device_figure(cpu):
    """There is no per-block CPU peak to report, and none is invented."""
    probe = MemoryProbe(cpu)
    with probe:
        held = torch.zeros(1024, 1024, dtype=torch.float32)
        held.fill_(1.0)  # touch the pages so they are resident
        del held
    reading = probe.reading
    assert reading.fidelity == "process_high_water"
    assert reading.peak_device_bytes is None
    assert reading.peak_above_baseline_bytes is None
    assert reading.driver_delta_bytes is None
    assert reading.process_rss_bytes > 0
    assert reading.process_rss_delta_bytes >= 0


def test_the_cpu_high_water_mark_never_decreases(cpu):
    """Which is exactly why it cannot attribute a peak to one block."""
    first = _max_rss_bytes()
    probe = MemoryProbe(cpu)
    with probe:
        held = torch.zeros(2048, 2048, dtype=torch.float32)
        held.fill_(1.0)
        del held
    assert probe.reading.process_rss_bytes >= first
    assert _max_rss_bytes() >= probe.reading.process_rss_bytes


def test_the_rss_unit_is_normalised_for_this_platform():
    """ru_maxrss is bytes on macOS and kilobytes on Linux; getting it wrong is 1024x.

    A plain sanity band rather than a precise figure: this process has torch
    imported, so its resident set is certainly more than 8 MiB and certainly
    less than 64 GiB. Either unit mistake falls outside that band.
    """
    assert 8 * MIB < _max_rss_bytes() < 64 * 1024 * MIB
    assert sys.platform in ("darwin", "linux", "win32")


def test_a_reading_serialises_with_its_fidelity_and_derived_peak():
    reading = MemoryReading(
        device="mps", fidelity="sampled",
        peak_device_bytes=10 * MIB, baseline_device_bytes=4 * MIB,
    )
    payload = reading.as_dict()
    assert payload["fidelity"] == "sampled"
    assert payload["peak_above_baseline_bytes"] == 6 * MIB
    assert set(payload) >= {"device", "fidelity", "samples", "sample_interval"}


def test_a_baseline_above_the_peak_clamps_to_zero_rather_than_going_negative():
    """Possible on MPS if another thread frees memory during the block."""
    reading = MemoryReading(
        device="mps", fidelity="sampled",
        peak_device_bytes=1 * MIB, baseline_device_bytes=4 * MIB,
    )
    assert reading.peak_above_baseline_bytes == 0


# --------------------------------------------------------------------------
# CUDA -- our branch, against a stub; the driver figures need real hardware
# --------------------------------------------------------------------------


class _FakeCuda:
    """The four calls :class:`MemoryProbe` makes on the CUDA path, and nothing else.

    Standing in for the driver is enough to test *our* logic -- that the peak
    counter is reset on entry, read back on exit, and labelled ``"exact"``. It is
    not enough to test PyTorch's accounting, which is the part that would need a
    real device. See this module's docstring.
    """

    def __init__(self, allocated=8 * MIB, peak=24 * MIB):
        self.allocated, self.peak = allocated, peak
        self.calls = []

    def empty_cache(self):
        self.calls.append("empty_cache")

    def memory_allocated(self, _device=None):
        self.calls.append("memory_allocated")
        return self.allocated

    def reset_peak_memory_stats(self, _device=None):
        self.calls.append("reset_peak_memory_stats")

    def max_memory_allocated(self, _device=None):
        self.calls.append("max_memory_allocated")
        return self.peak

    def synchronize(self, _device=None):
        self.calls.append("synchronize")


def test_the_cuda_branch_resets_reads_back_and_labels_the_result_exact(monkeypatch):
    fake = _FakeCuda(allocated=8 * MIB, peak=24 * MIB)
    monkeypatch.setattr(torch, "cuda", fake)

    probe = MemoryProbe(torch.device("cuda"))
    with probe:
        pass

    reading = probe.reading
    assert reading.fidelity == "exact"
    assert reading.peak_device_bytes == 24 * MIB
    assert reading.baseline_device_bytes == 8 * MIB
    assert reading.peak_above_baseline_bytes == 16 * MIB
    assert reading.driver_delta_bytes is None  # CUDA needs no driver proxy
    # The peak must be reset *before* the block, or it would report a stale high
    # water mark from whatever ran earlier in the process.
    assert fake.calls.index("reset_peak_memory_stats") < fake.calls.index(
        "max_memory_allocated"
    )
    assert "empty_cache" in fake.calls


def test_the_cuda_branch_can_skip_the_cache_flush_when_asked(monkeypatch):
    fake = _FakeCuda()
    monkeypatch.setattr(torch, "cuda", fake)
    with MemoryProbe(torch.device("cuda"), empty_cache=False):
        pass
    assert "empty_cache" not in fake.calls


@pytest.mark.skipif(torch.cuda.is_available(), reason="a real CUDA device is present")
def test_no_cuda_device_is_available_in_this_development_environment():
    """Documents why the CUDA figures above are stubbed rather than measured.

    Development happened on Apple silicon throughout. A container with a real
    CUDA device would be needed to validate PyTorch's own accounting on that
    backend, and building one is deliberately left as future work rather than
    approximated here.
    """
    assert not torch.cuda.is_available()
