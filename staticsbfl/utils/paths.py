"""Project paths.

Everything this project *produces* lands under ``cache/`` at the repository
root, and ``cache/`` is git-ignored. Nothing outside that directory is written
by a script, so a clean checkout plus one training run reproduces the whole
state, and `rm -rf cache` is always a safe reset.

    cache/data/                                 downloaded datasets (CIFAR-10, Fashion-MNIST)
    cache/variants/<dataset>/<phase>/...        the populations, light mode
    cache/variants-heavy/<dataset>/<phase>/...  the same layout, heavy mode
    cache/variants-debug/<dataset>/<phase>/...  the same layout, debug mode
    cache/runtime/...                           the runtime comparison and the cost report
    cache/evaluation/<dataset>/                 the final evaluation (log + report.html)

    <phase>/design.json          what the phase asks and how its populations
                                 were specified
    <phase>/models.csv           one row per generated model
    <phase>/activation_rates.csv per-layer activity diagnostics
    <phase>/neurons.npz          the neuron table
    <phase>/predictions/         predictor results, per metric
    <phase>/timings.json         measured time per step
    <phase>/<split>/<key>.pt     variant model weights
    <phase>/<split>/spectra/     their hit spectra and scores

Four files per phase rather than six: ``design.json`` is the specification (the
question, the regions each group was drawn from, the budget) and ``models.csv``
is the realisation (what was actually built and how good it turned out). The
earlier ``population.txt`` mixed the two and repeated every model row that
``manifest.csv`` already held, and ``generation_time.txt`` held three numbers
that belong with the specification they describe.
"""

from pathlib import Path

# .../staticsbfl/utils/paths.py -> the repository root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Every generated artefact lives here, and this directory is git-ignored.
CACHE_DIR = PROJECT_ROOT / "cache"

DATA_DIR = CACHE_DIR / "data"

# --- architecture-variation populations ---------------------------------------
# One directory per phase, split into train/ and test/ *by model*, because
# neurons of one model share everything that distinguishes them and must never
# straddle the split.
VARIANTS_DIR = CACHE_DIR / "variants"

#: The ``debug`` mode writes here instead, and nowhere else. A pipeline check
#: must not be able to overwrite a real population's design file or leave a
#: six-model neuron table where a hundred-model one belongs, so the two never
#: share a directory and ``rm -rf cache/variants-debug`` is always safe.
DEBUG_VARIANTS_DIR = CACHE_DIR / "variants-debug"

#: The ``heavy`` mode writes here, apart from ``light``. The two used to share
#: one tree, and training skips any cached checkpoint -- so a heavy run kept the
#: 3-epoch light models under their keys and trained only the rest for 10,
#: leaving two training budgets in a population whose design claims one. The
#: configurations are still prefixes of each other (``p2_train_000`` is the same
#: architecture and seed in both modes); only the trained artefacts are separate.
HEAVY_VARIANTS_DIR = CACHE_DIR / "variants-heavy"


#: Dataset a population is trained on when none is stated. Every path below is
#: namespaced by it, because a CIFAR-10 population and a Fashion-MNIST one share
#: every configuration key -- ``p1_train_000`` names a model in both -- and would
#: otherwise overwrite each other's checkpoints under the same filename.
DEFAULT_VARIANT_DATASET = "cifar10"


def variants_root(mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """Where a run in *mode* on *dataset* keeps its populations.

    Raises:
        ValueError: for an unknown mode -- a typo must not fall through to a
            real population's tree.
    """
    # Looked up at call time rather than held in a module constant, so a test
    # can redirect one root without rebuilding the mapping.
    roots = {"debug": DEBUG_VARIANTS_DIR, "light": VARIANTS_DIR, "heavy": HEAVY_VARIANTS_DIR}
    if mode not in roots:
        raise ValueError(f"Unknown mode {mode!r}. Choose from {sorted(roots)}.")
    return roots[mode] / dataset


def variant_dir(
    phase: str, split: str, mode: str = "light",
    dataset: str = DEFAULT_VARIANT_DATASET,
) -> Path:
    """``<root>/<phase>/<split>/`` -- where that population's models live."""
    return variants_root(mode, dataset) / phase / split


def variant_model_path(phase: str, split: str, key: str, mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """Where one variant model's weights are cached."""
    return variant_dir(phase, split, mode, dataset) / f"{key}.pt"


def variant_spectrum_path(phase: str, split: str, key: str, mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """Where one variant model's hit spectrum and scores are cached."""
    return variant_dir(phase, split, mode, dataset) / "spectra" / f"{key}.pt"


def variant_models_csv(phase: str, mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """One CSV per phase: every model's configuration and its trained quality.

    Named for what it holds. The previous name, ``manifest.csv``, said only that
    it was a list of something.
    """
    return variants_root(mode, dataset) / phase / "models.csv"


def variant_design(phase: str, mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """The phase specification: its question, populations, regions and budget.

    Machine-readable on purpose. The design of a population is not recoverable
    from the model files -- which region each group was drawn from, which axis it
    moved, what budget every model got -- and a later run must be able to check
    that it is extending the same design rather than a different one.
    """
    return variants_root(mode, dataset) / phase / "design.json"


def variant_neurons(phase: str, mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """The neuron table: one row per hidden unit, features plus targets."""
    return variants_root(mode, dataset) / phase / "neurons.npz"


def variant_activation_rates(phase: str, mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """Per-layer activity diagnostics -- the file that shows the dead units."""
    return variants_root(mode, dataset) / phase / "activation_rates.csv"


def variant_predictions_dir(phase: str, mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """Where one phase's predictor results live, one file per metric."""
    return variants_root(mode, dataset) / phase / "predictions"


def variant_predictions_csv(
    phase: str, metric: str, target: str, mode: str = "light",
    dataset: str = DEFAULT_VARIANT_DATASET,
) -> Path:
    """Results for one (phase, suspiciousness metric, target flavour).

    The metric is in the filename rather than a column because a run fits one
    metric at a time and a partial sweep must not look like a complete one.
    """
    return variant_predictions_dir(phase, mode, dataset) / f"{phase}__{metric}__{target}.csv"


def variant_correlation_log(phase: str, mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """The human-readable record of what every reported correlation means."""
    return variant_predictions_dir(phase, mode, dataset) / "correlations.log"


def variant_timings(phase: str, mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """Measured wall-clock time per step, the basis of every runtime estimate."""
    return variants_root(mode, dataset) / phase / "timings.json"


#: The runtime/memory comparison and the cost report. Per dataset the
#: test set is exported once; per dataset and mode the exported models, the
#: Keras conversions for both DeepFault ports and one record per measured
#: process. The cost report (``costs.csv``) sits at the top.
RUNTIME_DIR = CACHE_DIR / "runtime"


def runtime_root(mode: str = "light", dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """``cache/runtime/<dataset>/<mode>/`` -- one runtime run's exports and records."""
    variants_root(mode, dataset)  # validates the mode
    return RUNTIME_DIR / dataset / mode


def runtime_test_set(dataset: str = DEFAULT_VARIANT_DATASET) -> Path:
    """The pre-processed test set every runtime tool reads, identical for all."""
    return RUNTIME_DIR / dataset / "test_set.npz"


def ensure_dir(directory: Path) -> Path:
    """Create *directory* (with parents) if needed and return it."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
