# ICSoft@pt — the published ICSOFT 2025 spectrum code, on Python 3.14

Port of `fast-hs-adversarial-input-gen`
(commit `6e906ed`, the repository the ICSOFT 2025 paper links; its URL is
withheld for double-blind review) for the
runtime comparison. It runs in the main environment (Python 3.14.2,
PyTorch 2.10.0); upstream needs nothing beyond PyTorch for the spectrum.

Upstream computes a spectrum the way `3_models_get_sus.ipynb` does, **per
layer**: one full forward pass over the data with a hook on that layer, every
activation moved to the CPU and kept, then the four counts summed over the
stored `(samples, units)` tensor. That is the property worth measuring next to
HS@pt, which streams all layers in one pass and stores nothing.

## Files

| file | origin |
|---|---|
| `helper/suspiciousness.py` | upstream, minimal port (table below) |
| `helper/global_variables.py` | upstream, minimal port (table below) |
| `hit_spectrum.py` | upstream notebook cell "Hit Spectrum Helper" (`unsqueeze_tensors`, `hs_analysis`) as a module |
| `../icsoft_upstream/` | the unmodified upstream files: both helpers, the notebook (its stored cell outputs cleared; the code cells are verbatim), its config, the README |
| `../../staticsbfl/cli/runtime_icsoft.py` | **new**: the timed runner (one model per process) |

## Every change to upstream code, and whether it can affect the measurement

| # | where | change | why | effect on the measured time / memory |
|---|---|---|---|---|
| 1 | `helper/global_variables.py` | `ICSOFT_DEVICE=cpu\|mps\|cuda` overrides the device choice | upstream knows only "MPS, else CUDA", so a CPU run (the CPU-vs-CPU comparison with the TensorFlow ports) was impossible on a Mac | None: unset, the choice is upstream's; set, only the device differs. |
| 2 | `helper/suspiciousness.collect_activations` | new parameter `oracle="target"` (upstream) or `"correct"` | upstream's success is "the model predicts `target_class`" (ICSOFT's targeted setting); DeepFault and StaticSBFL use "the model predicts the true label". The comparison needs the same oracle for all tools | None: one comparison per sample either way. Only the meaning of `a_s` / `a_f` follows the oracle. |
| 3 | `hit_spectrum.hs_analysis` | the notebook globals `sus_config.activation_threshold` / `sus_config.verbose` became parameters (defaults 0.0 / False); the oracle is passed through | a module cannot read the notebook's globals | None for the threshold. `verbose=False` skips `log_diagnostics` (prints only). |

Not changed: `collect_activations` (hook, `.cpu()` per batch, `torch.cat`),
`unsqueeze_tensors` (`clone()` of the activations and targets),
`compute_hit_spectrum` (four `logical_and` sums), `get_tarantula`,
`get_ochiai`. ICSoft's `get_ochiai` uses `a_s` in its numerator (its targeted
reading of "suspicious"); it is kept, so ICSoft@pt's Ochiai is not the Ochiai
of the other tools and is left out of the fidelity check (counts and Tarantula
are compared). ICSOFT defines no D*.

## The runner (`staticsbfl/cli/runtime_icsoft.py`)

Run as the notebook runs it, with two settings changed for comparability:

| setting | notebook | runner | why / effect |
|---|---|---|---|
| loader | `DataLoader(subset, batch_size=64, shuffle=True)` | the same, over the shared pre-processed test set | identical traversal; the shuffle is seeded (counts do not depend on order) |
| samples | `sus_config.samples` (a subset) | the whole test set (10 000) | the same workload as every other tool |
| layers | one configured layer | every hidden layer, one `hs_analysis` each | a full-model spectrum, as the other tools compute; this is ICSoft's per-layer design, repeated |
| oracle | `target_class` | `oracle="correct"` | change #2 |
| timer | `perf_counter` from after the model load, incl. building the test loader | the shared probe, per phase (`data`, `model_load`, `spectrum`, `score`) | finer, and the same probe as every other tool |
