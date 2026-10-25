"""Measurement for the runtime and memory comparison.

``footprint``  process memory (macOS phys_footprint) and phase timing, shared
               by every tool of the runtime comparison; standard library only.
``resources``  the device allocator's peak memory, labelled with how much the
               backend lets it be trusted (HS@pt records it next to the
               footprint).

The names below are imported lazily (PEP 562), so ``import
staticsbfl.benchmark.footprint`` does not pull in torch: the runtime comparison times
each tool's own framework import, and must not have imported torch before it.

Nothing in this package is on the path of a normal ``make score`` run.
It exists so the reference implementation never pays for the measurement.
"""

import importlib

_EXPORTS = {
    "resources": ("DEFAULT_SAMPLE_INTERVAL", "FIDELITIES", "MemoryProbe", "MemoryReading",
                  "counter_bytes", "synchronize", "tensor_bytes"),
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module 'staticsbfl.benchmark' has no attribute {name!r}")
    value = getattr(importlib.import_module(f"staticsbfl.benchmark.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
