"""The three generation phases of the study.

Each phase produces a *train* and a *test* population of architectures. The
phases form a generalization ladder, and it is worth being explicit that this is
what they are, because each one answers a different question:

    phase 1  one architecture, many initialisations.
             "How much does per-neuron suspiciousness move when nothing but the
             seed changes?" This is the **noise floor**: no static predictor can
             be more consistent than the quantity it predicts. Every later
             result is read against this number.

    phase 2  architectures sampled from one region.
             "Does a predictor trained on architectures from a region work on
             held-out architectures from the same region?" Interpolation.

    phase 3  train and test drawn from *disjoint* regions.
             "Does it still work on architectures unlike anything it saw?"
             Extrapolation, and the question a reviewer actually asks.

Splitting
---------
Every split is **by model, never by neuron**. Neurons of one model share its
architecture, its training run and its data, so putting some of a model's
neurons in train and the rest in test leaks almost everything that distinguishes
them and inflates the score. The split therefore happens here, over
configurations, before a single model is built.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import random

from staticsbfl.variation.config import (
    ArchitectureConfig,
    BoundsError,
    VariationBounds,
    sample_config_pool,
)

#: The phase names the CLI accepts, in the order they are meant to be run.
PHASE_NAMES: Tuple[str, ...] = ("phase1", "phase2", "phase3", "phase4")

#: Models per population, by run mode. The population a predictor is *fitted*
#: on is much larger than each population it is *evaluated* on: fitting is what
#: needs the data, while an evaluation group only has to estimate a mean rank
#: correlation, and every extra evaluation model costs a full training run.
#: ``debug`` exists to check that the pipeline runs and writes the files it
#: should, not to produce a result. Four fit models cannot support a predictor;
#: any number it reports is noise and the scripts say so.
POPULATION_SIZES: Dict[str, Dict[str, int]] = {
    "debug": {"fit": 20, "eval": 4},
    "light": {"fit": 200, "eval": 40},
    "heavy": {"fit": 400, "eval": 80},
}
# History: until 2026-10-22 light was 100/20, heavy 500/100 and debug 4/2. Every
# population is a prefix of the pool layout below, so light 200/40 extends the
# old light populations and heavy 400/80 is a subset of the old heavy ones:
# `make trim VMODE=heavy` removes what a smaller mode no longer uses.

#: Epochs per run mode. Deliberately short: these populations exist to be
#: compared with each other, and every model gets the identical budget, so the
#: absolute accuracy matters less than that nothing is trained longer.
#:
#: ``light`` stays at 3 on purpose. Fewer would make the study measure something
#: else: a shorter budget leaves more units dead, and dead units are exactly the
#: confound under investigation; and phase 1 estimates the reseeding noise floor,
#: which is larger for an undertrained model than for a fitted one. Training is
#: also not the bottleneck -- fitting the predictors is -- so shortening it buys
#: the wrong resource.
EPOCHS_BY_MODE = {"debug": 1, "light": 3, "heavy": 10}

#: Where evaluation seeds start, counted from ``seed_start``, for every phase.
#:
#: The rule this enforces: **nothing about an evaluation model may depend on how
#: big the fit population is.** Seeds used to be laid out as
#: ``seed_start + n_fit + index``, so raising the fit size moved every evaluation
#: seed -- the cached file for ``p1_test_000`` then belonged to a model that no
#: longer existed under that name. Offsets are fixed and far apart instead, so a
#: key names the same model in every mode. (Each mode still trains its own copy
#: under its own budget, in its own cache tree -- see ``staticsbfl.utils.paths``.)
EVAL_SEED_OFFSET = 10_000

#: Seed distance between two evaluation groups of the same phase. Larger than
#: any evaluation group will be, so groups cannot collide as sizes grow.
EVAL_GROUP_STRIDE = 1_000

#: The layout of phase 2's shared pool: the first MAX_FIT configurations are the
#: fit candidates, the next MAX_EVAL the evaluation candidates, and every mode
#: takes prefixes of both. FIXED, not derived from POPULATION_SIZES: the
#: evaluation slice starts at MAX_FIT, so deriving it from the sizes would turn
#: every phase-2 test model into a different architecture whenever the largest
#: mode shrank or grew. 500/100 is the layout every cached model was built with.
MAX_FIT = 500
MAX_EVAL = 100
if any(s["fit"] > MAX_FIT or s["eval"] > MAX_EVAL for s in POPULATION_SIZES.values()):
    raise BoundsError(f"A population exceeds the fixed pool layout {MAX_FIT}/{MAX_EVAL}.")

#: Backwards-compatible alias: the fit-population size per mode.
MODELS_PER_SPLIT = {mode: sizes["fit"] for mode, sizes in POPULATION_SIZES.items()}


def population_sizes(mode: str) -> Tuple[int, int]:
    """``(fit_size, eval_size)`` for *mode*.

    Raises:
        BoundsError: for an unknown mode.
    """
    if mode not in POPULATION_SIZES:
        raise BoundsError(f"Unknown mode {mode!r}. Choose from {list(POPULATION_SIZES)}.")
    sizes = POPULATION_SIZES[mode]
    return sizes["fit"], sizes["eval"]


# --- deliberately designed architectures, for phase 4 -------------------------
# Every other phase samples architectures at random, which is the first thing a
# reader objects to: networks used in practice are designed, not drawn from a
# uniform distribution. These are hand-picked dense architectures of the kind
# that actually appear in image-classification work, so that a predictor fitted
# on sampled architectures can be tested on designed ones.
DESIGNED_ARCHITECTURES: Tuple[Tuple[str, Tuple[int, ...], str], ...] = (
    # This project's own small dense baseline.
    ("small_mlp", (256, 128), "relu"),
    # This project's standard-sized dense counterpart.
    ("wide_mlp", (2048, 1024, 512), "relu"),
    # The classic MNIST-era shape, still a common default.
    ("classic_512_256", (512, 256), "relu"),
    # LeCun-style proportions.
    ("lecun_300_100", (300, 100), "relu"),
    # A deeper designed stack with a regular halving pattern.
    ("halving_1024", (1024, 512, 256, 128), "relu"),
)


@dataclass(frozen=True)
class PhaseSpec:
    """One phase: what it asks, and the populations that answer it.

    A phase has exactly one population the predictor is *fitted* on
    (``train_split``) and one or more it is *evaluated* on. Phases 1 and 2 use
    the plain ``train`` / ``test`` pair. Phase 3 uses a ``base`` population plus
    one validation group per varied axis, so that every group is compared
    against the same training region and a drop can be attributed to the single
    axis that group moved.
    """

    name: str
    question: str
    train_split: str
    populations: Dict[str, Tuple[ArchitectureConfig, ...]]
    regions: Dict[str, VariationBounds]

    @property
    def split_names(self) -> Tuple[str, ...]:
        """Every population name, the training one first."""
        others = [n for n in self.populations if n != self.train_split]
        return (self.train_split, *sorted(others))

    @property
    def train(self) -> Tuple[ArchitectureConfig, ...]:
        """The population the predictor is fitted on."""
        return self.populations[self.train_split]

    @property
    def test(self) -> Tuple[ArchitectureConfig, ...]:
        """Every population the predictor is evaluated on, concatenated."""
        return tuple(
            config
            for name in self.split_names[1:]
            for config in self.populations[name]
        )

    @property
    def evaluation_splits(self) -> Tuple[str, ...]:
        """Names of the populations the predictor is evaluated on."""
        return self.split_names[1:]

    @property
    def n_models(self) -> int:
        """Total models across every population of this phase."""
        return sum(len(group) for group in self.populations.values())

    def configs(self) -> List[ArchitectureConfig]:
        """Every configuration of this phase, training population first."""
        return [c for name in self.split_names for c in self.populations[name]]

    def describe(self) -> str:
        """A multi-line summary of the populations, for logs and the run record.

        A population holding a single architecture is described by that
        architecture rather than by its region. Printing the region there would
        be actively misleading: a fixed model of widths ``(256, 128, 64)`` has a
        region of ``width 64-256``, which reads as a range of sampled widths
        when in fact nothing varies but the seed.
        """
        lines = [f"{self.name}: {self.n_models} models across {len(self.populations)} population(s)"]
        for name in self.split_names:
            marker = "fit " if name == self.train_split else "eval"
            configs = self.populations[name]
            shapes = {(c.hidden_widths, c.activation) for c in configs}
            if len(shapes) == 1:
                widths, activation = next(iter(shapes))
                seeds = sorted(c.seed for c in configs)
                detail = (
                    f"FIXED architecture {'-'.join(str(w) for w in widths)} "
                    f"({activation}), seeds {seeds[0]}-{seeds[-1]}"
                )
            else:
                detail = f"sampled from {self.regions[name].describe()}"
            lines.append(
                f"    [{marker}] {name:16} {len(configs):3} models  {detail}"
            )
        return "\n".join(lines)


def phase1_configs(
    *,
    hidden_widths: Sequence[int] = (256, 128, 64),
    activation: str = "relu",
    n_fit: int = 50,
    n_eval: int = 10,
    seed_start: int = 42,
    family: str = "mlp",
) -> PhaseSpec:
    """Phase 1: one fixed architecture, consecutive initialisation seeds.

    Seeds run from *seed_start* upwards with no gaps, train split first, so the
    train and test populations are the same architecture under different
    initialisations. Nothing about the architecture varies -- that is the point.
    """
    widths = tuple(int(w) for w in hidden_widths)
    if not widths:
        raise BoundsError("phase 1 needs at least one hidden layer.")

    bounds = VariationBounds(
        min_depth=len(widths),
        max_depth=len(widths),
        min_width=min(widths),
        max_width=max(widths),
        activations=(activation,),
        family=family,
    )

    def make(split: str, offset: int, count: int) -> Tuple[ArchitectureConfig, ...]:
        """Consecutive seeds for one population, offset so no seed repeats."""
        return tuple(
            ArchitectureConfig(
                key=f"p1_{split}_{index:03d}",
                family=family,
                hidden_widths=widths,
                activation=activation,
                seed=seed_start + offset + index,
                phase="phase1",
                split=split,
            )
            for index in range(count)
        )

    return PhaseSpec(
        name="phase1",
        question="How far does per-neuron suspiciousness move under reseeding alone?",
        train_split="train",
        populations={
            "train": make("train", 0, n_fit),
            "test": make("test", EVAL_SEED_OFFSET, n_eval),
        },
        regions={"train": bounds, "test": bounds},
    )


def phase2_configs(
    *,
    bounds: Optional[VariationBounds] = None,
    n_fit: int = 50,
    n_eval: int = 10,
    rng_seed: int = 42,
    seed_start: int = 42,
) -> PhaseSpec:
    """Phase 2: architectures sampled from one region, split by model.

    A single pool is drawn and then partitioned, rather than drawing two pools,
    so train and test are guaranteed to come from the same distribution and to
    contain no duplicate architecture across the split.
    """
    bounds = bounds or VariationBounds()
    # Drawn and shuffled at the largest size any mode asks for, then sliced at
    # fixed positions. Drawing `n_fit + n_eval` and splitting at `n_fit` looked
    # equivalent and was not: the pool length fed the shuffle, so raising the fit
    # size reshuffled everything and `p2_train_000` silently became a different
    # architecture while keeping its name and its cached checkpoint.
    pool = sample_config_pool(
        bounds,
        MAX_FIT + MAX_EVAL,
        rng_seed=rng_seed,
        prefix="p2",
        phase="phase2",
        seed_start=seed_start,
    )
    rng = random.Random(rng_seed)
    rng.shuffle(pool)

    train = tuple(
        ArchitectureConfig(**{**vars(c), "split": "train", "key": f"p2_train_{i:03d}"})
        for i, c in enumerate(pool[:n_fit])
    )
    test = tuple(
        ArchitectureConfig(**{**vars(c), "split": "test", "key": f"p2_test_{i:03d}"})
        for i, c in enumerate(pool[MAX_FIT : MAX_FIT + n_eval])
    )

    return PhaseSpec(
        name="phase2",
        question="Does a predictor generalize to held-out architectures from the same region?",
        train_split="train",
        populations={"train": train, "test": test},
        regions={"train": bounds, "test": bounds},
    )


def phase3_configs(
    *,
    base_bounds: VariationBounds,
    extensions: Dict[str, VariationBounds],
    n_fit: int = 50,
    n_eval: int = 10,
    rng_seed: int = 42,
    seed_start: int = 42,
) -> PhaseSpec:
    """Phase 3: one base population, plus a validation group per varied axis.

    The predictor is fitted on the *base* population only. Each validation group
    holds every axis at the base region and moves exactly one to a disjoint
    range, so the groups are directly comparable with each other and a drop in
    performance can be attributed to the axis that group moved.

    Args:
        base_bounds: the region the predictor is trained on.
        extensions: ``{group name: bounds}``; each must be disjoint from
            *base_bounds*.
        n_fit: models in the base (fitting) population.
        n_eval: models in each validation group.

    Raises:
        BoundsError: if a validation region overlaps the base region on every
            axis, or if a group moves more than one axis away from the base.
    """
    populations: Dict[str, Tuple[ArchitectureConfig, ...]] = {}
    regions: Dict[str, VariationBounds] = {"base": base_bounds}

    populations["base"] = tuple(
        sample_config_pool(
            base_bounds, n_fit, rng_seed=rng_seed, prefix="p3_base",
            phase="phase3", split="base", seed_start=seed_start,
        )
    )

    offset = 1
    for group, bounds in extensions.items():
        if not base_bounds.overlaps(bounds):
            pass  # disjoint on at least one axis, which is what is wanted
        else:
            raise BoundsError(
                f"validation group {group!r} overlaps the base region on every "
                f"axis, so it would not test extrapolation.\n"
                f"  base:  {base_bounds.describe()}\n"
                f"  group: {bounds.describe()}"
            )
        moved = moved_axes(base_bounds, bounds)
        if len(moved) != 1:
            raise BoundsError(
                f"validation group {group!r} moves {len(moved)} axes "
                f"({', '.join(moved) or 'none'}); each group must move exactly "
                "one, or a drop cannot be attributed to an axis."
            )
        populations[group] = tuple(
            sample_config_pool(
                bounds, n_eval, rng_seed=rng_seed + offset, prefix=f"p3_{group}",
                phase="phase3", split=group,
                seed_start=seed_start + EVAL_SEED_OFFSET + offset * EVAL_GROUP_STRIDE,
            )
        )
        regions[group] = bounds
        offset += 1

    return PhaseSpec(
        name="phase3",
        question="Does a predictor extrapolate along one architecture axis at a time?",
        train_split="base",
        populations=populations,
        regions=regions,
    )


def moved_axes(base: VariationBounds, other: VariationBounds) -> List[str]:
    """Which axes differ between two regions: any of depth, width, activation."""
    moved = []
    if (base.min_depth, base.max_depth) != (other.min_depth, other.max_depth):
        moved.append("depth")
    if (base.min_width, base.max_width) != (other.min_width, other.max_width):
        moved.append("width")
    if set(base.activations) != set(other.activations):
        moved.append("activation")
    return moved


# --- the shipped phase-3 design ----------------------------------------------
# One base region; each validation group holds the other axes at the base and
# moves exactly one. That makes the groups comparable with each other, which
# three independently-based splits would not be.

PHASE3_BASE = VariationBounds(
    min_depth=2, max_depth=4,
    min_width=64, max_width=128,
    activations=("relu", "leaky_relu", "elu"),
)

PHASE3_EXTENSIONS = {
    # deeper than anything seen, same widths and activations
    "deeper": VariationBounds(
        min_depth=5, max_depth=10, min_width=64, max_width=128,
        activations=("relu", "leaky_relu", "elu"),
    ),
    # wider than anything seen, same depths and activations
    "wider": VariationBounds(
        min_depth=2, max_depth=4, min_width=129, max_width=512,
        activations=("relu", "leaky_relu", "elu"),
    ),
    # saturating activations, same depths and widths
    "saturating": VariationBounds(
        min_depth=2, max_depth=4, min_width=64, max_width=128,
        activations=("tanh", "sigmoid"),
    ),
}


def phase4_configs(
    *,
    bounds: Optional[VariationBounds] = None,
    n_fit: int = 50,
    n_eval: int = 10,
    rng_seed: int = 42,
    seed_start: int = 42,
) -> PhaseSpec:
    """Phase 4: fit on sampled architectures, evaluate on *designed* ones.

    Every other phase both trains and tests on architectures drawn from a
    distribution, which leaves the obvious objection unanswered: networks used
    in practice are deliberately designed, not sampled. This phase evaluates on
    :data:`DESIGNED_ARCHITECTURES` -- hand-picked dense shapes of the kind that
    actually appear in image classification -- while fitting on sampled ones.

    The designed models are trained here under the same budget, optimizer and
    dataset as the sampled ones, so the only thing that differs is how the
    architecture was chosen.

    The evaluation population repeats each designed architecture under as many
    seeds as ``n_eval`` allows, so that a single unlucky initialisation does not
    decide the result.
    """
    bounds = bounds or VariationBounds(
        min_depth=2, max_depth=4, min_width=64, max_width=2048,
        activations=("relu",),
    )
    fit = tuple(
        sample_config_pool(
            bounds, n_fit, rng_seed=rng_seed, prefix="p4_sampled",
            phase="phase4", split="sampled", seed_start=seed_start,
        )
    )

    # Seed-major, not architecture-major: enumerating one seed across every
    # architecture before moving to the next seed makes a smaller population a
    # prefix of a larger one, so a key names the same model in every mode. The
    # architecture-major order did not have that property and was the one group
    # in the whole study whose keys changed meaning when a population size
    # changed.
    seeds_each = max(1, n_eval // len(DESIGNED_ARCHITECTURES))
    designed: List[ArchitectureConfig] = []
    for repeat in range(seeds_each):
        for name, widths, activation in DESIGNED_ARCHITECTURES:
            designed.append(
                ArchitectureConfig(
                    key=f"p4_designed_{name}_s{repeat}",
                    family="mlp",
                    hidden_widths=widths,
                    activation=activation,
                    # Offset well past any fit population rather than past *this*
                    # one: a seed that moved with n_fit silently changed which
                    # model a cached key referred to.
                    seed=seed_start + EVAL_SEED_OFFSET + len(designed),
                    phase="phase4",
                    split="designed",
                )
            )
    designed = designed[:n_eval] if n_eval < len(designed) else designed

    designed_region = VariationBounds(
        min_depth=min(len(w) for _n, w, _a in DESIGNED_ARCHITECTURES),
        max_depth=max(len(w) for _n, w, _a in DESIGNED_ARCHITECTURES),
        min_width=min(min(w) for _n, w, _a in DESIGNED_ARCHITECTURES),
        max_width=max(max(w) for _n, w, _a in DESIGNED_ARCHITECTURES),
        activations=tuple({a for _n, _w, a in DESIGNED_ARCHITECTURES}),
    )

    return PhaseSpec(
        name="phase4",
        question="Does a predictor fitted on sampled architectures transfer to designed ones?",
        train_split="sampled",
        populations={"sampled": fit, "designed": tuple(designed)},
        regions={"sampled": bounds, "designed": designed_region},
    )


def build_phase(name: str, mode: str = "light") -> PhaseSpec:
    """Construct a phase by name, sized for the run mode.

    Args:
        name: one of :data:`PHASE_NAMES`.
        mode: "light" or "heavy"; sets the population sizes.

    Raises:
        BoundsError: for an unknown phase or mode.
    """
    if name not in PHASE_NAMES:
        raise BoundsError(f"Unknown phase {name!r}. Choose from {list(PHASE_NAMES)}.")
    n_fit, n_eval = population_sizes(mode)

    if name == "phase1":
        return phase1_configs(n_fit=n_fit, n_eval=n_eval)
    if name == "phase2":
        return phase2_configs(n_fit=n_fit, n_eval=n_eval)
    if name == "phase4":
        return phase4_configs(n_fit=n_fit, n_eval=n_eval)
    return phase3_configs(
        base_bounds=PHASE3_BASE, extensions=PHASE3_EXTENSIONS,
        n_fit=n_fit, n_eval=n_eval,
    )
