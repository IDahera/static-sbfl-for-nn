# baselines — the dynamic tools of the runtime comparison

Three published implementations, ported so that they run on the interpreter
of the main environment (Python 3.14.2) and can be applied to the phase-4
models. The goal of every port is **compatibility, not improvement**: upstream
functions are kept, in upstream order, with the same data structures and the
same loops. Every edited line carries a `# PORT:` comment, each port's README
lists every change with its possible effect on the measured time, and the
unmodified upstream sources sit next to the ports for `diff -u`.

| directory | name in the paper | upstream | execution model | environment |
|---|---|---|---|---|
| [`deepfault_tf1/`](deepfault_tf1/README.md) | **DF@graph** (`tf1` here) | [hfeniser/DeepFault](https://github.com/hfeniser/DeepFault) `7a068e7` | TensorFlow 1: static graph + `Session` (`tf.compat.v1`, eager off), Keras 2 API (`tf_keras`) | conda `deepfault-tf1` |
| [`deepfault_tf2/`](deepfault_tf2/README.md) | **DF@eager** (`tf2` here) | same | TensorFlow 2: eager, Keras 3 | conda `deepfault-tf2` |
| [`icsoft_pt/`](icsoft_pt/README.md) | **ICSoft@pt** (opt-in) | `fast-hs-adversarial-input-gen` `6e906ed` (the code the ICSOFT 2025 paper links; URL withheld for double-blind review) | PyTorch: one pass per layer, every activation stored, counts summed over the stored tensor | main env |
| [`deepfault_upstream/`](deepfault_upstream/) | — | DeepFault, verbatim | Keras 2.3.1 / TF 1.13 | (Python ≤ 3.7) |
| [`icsoft_upstream/`](icsoft_upstream/) | — | ICSoft, verbatim (helpers, notebook with its outputs cleared, config) | | |
| [`common/`](common/) | — | the runtime runner shared by both DeepFault ports | | standard library only |

**What "tf1" and "tf2" mean.** Both ports run the *same* TensorFlow 2.22
binary. They differ in execution model and Keras generation, which is what
separates TensorFlow 1 from TensorFlow 2 for code like DeepFault's:

| | DF@tf1 | DF@tf2 |
|---|---|---|
| execution | graph mode: `tf.compat.v1.disable_eager_execution()`, `K.function` runs a compiled graph in a `Session` | eager: every op runs immediately, no graph, no session |
| Keras | `tf_keras` 2.21 (the Keras 2 API DeepFault was written for) | Keras 3 (multi-backend; no `K.function`, `K.gradients`, `output_shape`) |
| gradients (input synthesis) | symbolic `K.gradients` | `tf.GradientTape` |
| uses `tf.compat.v1` | yes, only to switch eager off | no |

Eager execution is the most visible change of TensorFlow 2 but not the only
one (Keras 3, `tf.function`, removed sessions/placeholders/`contrib`, new
optimizers and saving formats); on DeepFault's measured path it is the
graph-vs-eager execution of the per-layer functors that differs. Names that
say this more precisely than "tf1/tf2": **DF-graph** and **DF-eager**.

The fourth dynamic tool, **HS@pt**, is not a baseline: it is this
repository's own streaming spectrum (`staticsbfl.suspiciousness.compute_hit_spectrum`,
one pass over the data for all layers, counts accumulated per batch, nothing
stored), timed by `staticsbfl/cli/runtime_hs.py`.

## Where what is

| what | where |
|---|---|
| ported code | `deepfault_tf1/*.py`, `deepfault_tf2/*.py`, `icsoft_pt/helper/*.py`, `icsoft_pt/hit_spectrum.py` |
| what was changed, and whether it can affect the measurement | `deepfault_tf1/README.md`, `deepfault_tf2/README.md`, `icsoft_pt/README.md` |
| the exact diff | `diff -u deepfault_upstream/utils.py deepfault_tf1/utils.py` (same for every file and port) |
| adapters to the phase-4 models (new files) | `deepfault_*/torch_bridge.py` (PyTorch weights → the port's own Keras format), `deepfault_*/runtime_run.py`, `common/deepfault_runtime.py`, `../staticsbfl/cli/runtime_icsoft.py` |
| pinned environments | `deepfault_tf1/requirements.txt` (+ `requirements-nodeps.txt`), `deepfault_tf2/requirements.txt`; set up by `make runtime-setup` |
| the measurement | `../staticsbfl/cli/runtime_benchmark.py` (orchestrator), `../staticsbfl/benchmark/footprint.py` (memory and time probe, shared by every tool) |

## Versions

All tools run on the same interpreter build, **Python 3.14.2**
(`h1858270_101_cp314`, Anaconda `main`), each in its own conda environment.
Every runtime run records the versions it actually used in
`cache/runtime/<dataset>/<mode>/environment.json`, and `make evaluate` prints
them.

| package | main env (StaticSBFL, HS@pt, ICSoft@pt) | `deepfault-tf1` | `deepfault-tf2` | upstream DeepFault (its README) |
|---|---|---|---|---|
| Python | 3.14.2 | 3.14.2 | 3.14.2 | not stated (TF 1.13 runs on ≤ 3.7) |
| TensorFlow | — | **2.22.0rc0**, eager off | **2.22.0rc0** | 1.13.2 |
| Keras | — | `tf_keras` **2.21.0** (Keras 2 API; installed `--no-deps`) | `keras-nightly` **3.16.0.dev2026092204** (Keras 3) | 2.3.1 |
| NumPy | 2.4.1 | 2.5.4 | 2.5.4 | not stated |
| h5py | — | 3.15.1 | 3.15.1 | not stated (< 3: uses `Dataset.value`) |
| scikit-learn | 1.8.0 | 1.9.1 (only `classification_report`, `train_test_split`) | 1.9.1 | not stated |
| PyTorch | 2.10.0 | — | — | — |

Why these and not others:

* **TensorFlow 1.x cannot be installed** on Python 3.14: its last release, 1.15,
  supports Python ≤ 3.7 and has no arm64 macOS build. DF@tf1 therefore runs on
  TensorFlow 2 with eager execution disabled, which restores the TF1 execution
  model — graph construction, `Session.run`, symbolic gradients — together with
  the legacy Keras 2 package `tf_keras`. This is the closest faithful TF1 on this
  interpreter; it is *not* the TF 1.13 binary, and the paper should say so.
* **TensorFlow 2.22.0rc0 is the only TensorFlow with a Python 3.14 wheel**
  (released 2026-09-23; TF 2.21.0 stops at Python 3.13). It is a release
  candidate; both environments pin it and the Keras nightly build it requires,
  so reruns use identical code.
* `tf_keras 2.21.0` declares `tensorflow<2.22`; it is installed with `--no-deps`
  and passes the end-to-end check below.
* No GPU for TensorFlow: there is no `tensorflow-metal` for Python 3.14. Both
  PyTorch tools are measured on CPU **and** MPS, so a CPU-vs-CPU comparison
  always exists.

## Checked

* **Upstream DeepFault flow, end to end, in both ports**: `train_mnist_nn.py`
  (1 epoch), then `run.py` with `tarantula`, `ochiai`, `dstar` and `random`,
  including input synthesis — all complete on Python 3.14.
* **Model conversion**: every phase-4 model converted by `torch_bridge.py`
  reproduces the PyTorch argmax on the 10 000 CIFAR-10 test images (agreement
  1.00000).
* **Spectra** (`small_mlp_s0`, 10 000 samples): DF@tf1 and DF@tf2 are
  bit-identical to each other; ICSoft@pt on the CPU is bit-identical to HS@pt
  on the CPU; against the score-step cache (counted on MPS) every tool differs
  in at most 2 of 384 neurons by one count — a pre-activation within float
  rounding of the threshold — with Spearman ρ ≥ 0.9999995. `fidelity.csv`
  repeats this check for every model of every run.

## How the runtime comparison uses them

```bash
make runtime-setup      # the two conda envs, pinned (the main env needs nothing new)
make runtime            # or, after light / heavy: make all
```

`staticsbfl/cli/runtime_benchmark.py` exports the phase-4 *designed* models and the
pre-processed test set as plain `.npz`, lets each DeepFault port convert the
weights into its own Keras format, and then starts one fresh process per
(tool, model, repeat). What each runner leaves out of the upstream flow, and
why, is in its docstring and in the port READMEs.

### How memory is measured

One file, `staticsbfl/benchmark/footprint.py`, measures every tool. It reads the
**process physical footprint** (`phys_footprint`, macOS `proc_pid_rusage`),
which on Apple silicon covers CPU heap *and* Metal/MPS buffers of the process.
`ru_maxrss` does not see GPU buffers (a 512 MB MPS tensor: footprint +600 MB,
`ru_maxrss` +9 MB) and is kept for reference only. Every measured process is
started with `MallocLargeCache=0`: otherwise macOS does not re-charge a freed
and re-used large block, and the footprint undercounts live memory (300 MB
re-used were charged as +0.6 MB). `tests/test_footprint.py` checks the probe
against models and arrays of known size, in fresh processes started the same
way. The full explanation is printed at the top of every runtime log and by
`make evaluate`.

## Licence

DeepFault is **GPL-3.0-or-later** (`deepfault_*/LICENSE`). What this repository
does to respect it:

* both ports, the upstream copy and `common/` stay under GPL-3.0-or-later,
  independently of the MIT licence of the rest of this repository; each
  directory has a `NOTICE` saying so (the NOTICE files name the repository the
  ports were first written in), with the original copyright (2019 Bogazici
  University), the upstream URL and commit;
* every **modified** file starts with an SPDX line, the original copyright and
  a dated modification notice (GPL-3 §5a); files the ports did not change stay
  byte-identical to upstream; files written for the ports carry an SPDX line;
* every change is marked `# PORT:` and listed in the port READMEs, and the
  unmodified source is shipped next to it (`deepfault_upstream/`);
* the `LICENSE` text travels with every copy. In a replication package, keep
  `baselines/` as its own directory with its `LICENSE` and `NOTICE` files, and
  cite DeepFault (Eniser, Gerasimou, Sen, FASE 2019).

ICSoft (`fast-hs-adversarial-input-gen`) carries no licence file upstream;
`icsoft_pt/` and `icsoft_upstream/` are included with the permission of its
copyright holder. DeepFault's
`Documents/` folder (third-party papers) and its pre-trained models are not
copied.
