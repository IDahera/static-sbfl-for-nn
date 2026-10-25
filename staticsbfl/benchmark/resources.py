"""Peak device memory, with an explicit honesty label on each reading.

The runtime comparison times HS@pt with :mod:`staticsbfl.benchmark.footprint`
(process footprint, every tool alike) and additionally records the device
allocator's view through :class:`MemoryProbe`. Its fidelity depends entirely on
the backend:

``cuda``   exact. ``reset_peak_memory_stats`` / ``max_memory_allocated`` are a
           high-water mark maintained by the allocator itself.
``mps``    sampled. PyTorch 2.x exposes ``current_allocated_memory`` and
           ``driver_allocated_memory`` but no peak counter, so a background
           thread polls the former every few milliseconds. A spike shorter than
           the sampling interval can be missed, so the sampled peak is a lower
           bound. ``driver_allocated_memory`` is reported alongside it: the MPS
           caching allocator rarely returns memory to the driver, so its
           increase over a block is a coarse upper bound on what that block
           newly demanded -- and it stays raised afterwards, which is why a
           sequence of blocks must be measured in a fixed order and reported
           with that order stated.
``cpu``    process high-water only. ``ru_maxrss`` never decreases, so it cannot
           attribute a peak to one block; it is reported as the process figure
           it is, not as a per-stage peak.

Nothing here tries to paper over those differences. A measurement whose
fidelity is ``"sampled"`` must be labelled as such wherever it is printed, and
the resulting artifacts carry the label so a table in the paper can too.
"""

import logging
import resource
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

#: How often the MPS sampler reads the allocator, in seconds. Small enough to
#: catch a batch-sized transient, large enough not to fight the main thread.
DEFAULT_SAMPLE_INTERVAL = 0.002

#: What a peak figure is actually worth, carried into every artifact.
FIDELITIES = ("exact", "sampled", "process_high_water", "unavailable")


def synchronize(device: torch.device) -> None:
    """Wait for *device* to finish its queued work. A no-op on the CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _max_rss_bytes() -> int:
    """Process high-water resident set size, normalised to bytes.

    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux; getting this wrong
    would silently report a 1024x error, so it is branched on explicitly.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


@dataclass
class MemoryReading:
    """What one measured block cost, and how much that number can be trusted."""

    device: str
    fidelity: str
    peak_device_bytes: Optional[int] = None
    baseline_device_bytes: Optional[int] = None
    driver_delta_bytes: Optional[int] = None
    process_rss_bytes: Optional[int] = None
    process_rss_delta_bytes: Optional[int] = None
    samples: int = 0
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL

    @property
    def peak_above_baseline_bytes(self) -> Optional[int]:
        """Device memory the block newly demanded, over what was already live."""
        if self.peak_device_bytes is None or self.baseline_device_bytes is None:
            return None
        return max(0, self.peak_device_bytes - self.baseline_device_bytes)

    def as_dict(self) -> Dict[str, Any]:
        data = dict(vars(self))
        data["peak_above_baseline_bytes"] = self.peak_above_baseline_bytes
        return data


class MemoryProbe:
    """Measure peak memory over a block, as well as the backend permits.

    Used as a context manager; :attr:`reading` is filled in on exit::

        probe = MemoryProbe(device)
        with probe:
            spectrum = compute(...)
        print(probe.reading.fidelity, probe.reading.peak_device_bytes)

    The MPS sampler runs in a daemon thread so an exception inside the block
    cannot leave it running. It reads through the same allocator the main thread
    uses; torch releases the GIL around kernel launches, so the sampler does get
    scheduled during a forward pass, but a transient shorter than
    *sample_interval* can still be missed. The resulting peak is therefore a
    lower bound and is labelled ``"sampled"``.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
        empty_cache: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.sample_interval = float(sample_interval)
        self.empty_cache = empty_cache
        self.reading: Optional[MemoryReading] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak = 0
        self._samples = 0
        self._baseline = 0
        self._driver_before = 0
        self._rss_before = 0

    # -- MPS sampling ------------------------------------------------------

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                current = torch.mps.current_allocated_memory()
            except RuntimeError:  # pragma: no cover - backend torn down
                return
            if current > self._peak:
                self._peak = current
            self._samples += 1
            self._stop.wait(self.sample_interval)

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "MemoryProbe":
        synchronize(self.device)
        self._rss_before = _max_rss_bytes()
        if self.device.type == "cuda":
            if self.empty_cache:
                torch.cuda.empty_cache()
            self._baseline = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.device.type == "mps":
            if self.empty_cache:
                torch.mps.empty_cache()
            self._baseline = torch.mps.current_allocated_memory()
            self._driver_before = torch.mps.driver_allocated_memory()
            self._peak = self._baseline
            self._samples = 0
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._sample_loop, name="mps-memory-sampler", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        synchronize(self.device)
        rss_after = _max_rss_bytes()
        if self.device.type == "cuda":
            self.reading = MemoryReading(
                device=str(self.device),
                fidelity="exact",
                peak_device_bytes=int(torch.cuda.max_memory_allocated(self.device)),
                baseline_device_bytes=int(self._baseline),
                process_rss_bytes=rss_after,
                process_rss_delta_bytes=rss_after - self._rss_before,
            )
        elif self.device.type == "mps":
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            # One final read: the block may have peaked between the last sample
            # and the synchronisation above.
            current = torch.mps.current_allocated_memory()
            peak = max(self._peak, current)
            self.reading = MemoryReading(
                device=str(self.device),
                fidelity="sampled",
                peak_device_bytes=int(peak),
                baseline_device_bytes=int(self._baseline),
                driver_delta_bytes=int(
                    torch.mps.driver_allocated_memory() - self._driver_before
                ),
                process_rss_bytes=rss_after,
                process_rss_delta_bytes=rss_after - self._rss_before,
                samples=self._samples,
                sample_interval=self.sample_interval,
            )
            self._thread = None
        else:
            self.reading = MemoryReading(
                device=str(self.device),
                fidelity="process_high_water",
                process_rss_bytes=rss_after,
                process_rss_delta_bytes=rss_after - self._rss_before,
            )


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Bytes the elements of *tensor* occupy: ``numel() * element_size()``.

    The analytical figure, exact on every backend and independent of any
    allocator. It deliberately counts the *elements* only -- not the Python
    object, not the allocator's rounding, and for a non-contiguous view not the
    storage it looks into. That is the right basis for "what does this spectrum
    keep?", and it is what the measured peaks are compared against: a measured
    figure below this is a bug in the measurement, and a measured figure above it
    is the allocator's overhead, which is worth seeing separately.
    """
    return tensor.numel() * tensor.element_size()


def counter_bytes(shapes, dtype: torch.dtype = torch.int32) -> int:
    """Bytes the persistent ``a_f``/``a_s`` accumulators need for *shapes*.

    This is the one memory figure that is exact on every backend, because it
    follows from the unit count rather than from an allocator: two counters per
    unit, at the counter dtype's width. It is the analytical companion to the
    measured peak, and the two answer different questions -- the counters are
    what a spectrum *keeps*, the peak is what computing it *transiently needed*.
    """
    width = torch.empty(0, dtype=dtype).element_size()
    total = 0
    for shape in shapes:
        units = 1
        for dim in shape:
            units *= int(dim)
        total += units
    return 2 * total * width
