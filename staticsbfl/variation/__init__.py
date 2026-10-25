"""Architecture variation: the generated model populations of the study.

Three modules with one responsibility each:

    :mod:`~staticsbfl.variation.config`   what an architecture *is*, and how one is
                                    sampled from a bounded region. Pure; no torch.
    :mod:`~staticsbfl.variation.phases`   the three experiment populations and their
                                    train/test splits, which are always by model.
    :mod:`~staticsbfl.variation.builder`  turning a config into a torch module, and
                                    nothing else.
"""

from staticsbfl.variation.builder import (
    ACTIVATION_MODULES,
    BUILDERS,
    VariableMLP,
    build_model,
    make_activation,
)
from staticsbfl.variation.config import (
    ACTIVATION_IDS,
    ACTIVATIONS,
    FAMILIES,
    ArchitectureConfig,
    BoundsError,
    VariationBounds,
    sample_config,
    sample_config_pool,
    sample_widths,
    with_seed,
)
from staticsbfl.variation.phases import (
    DESIGNED_ARCHITECTURES,
    EPOCHS_BY_MODE,
    MODELS_PER_SPLIT,
    POPULATION_SIZES,
    PHASE3_BASE,
    PHASE3_EXTENSIONS,
    PHASE_NAMES,
    PhaseSpec,
    build_phase,
    moved_axes,
    phase1_configs,
    phase2_configs,
    phase3_configs,
    phase4_configs,
    population_sizes,
)

__all__ = [
    "ACTIVATION_MODULES", "BUILDERS", "VariableMLP", "build_model", "make_activation",
    "ACTIVATION_IDS", "ACTIVATIONS", "FAMILIES", "ArchitectureConfig", "BoundsError",
    "VariationBounds", "sample_config", "sample_config_pool", "sample_widths", "with_seed",
    "DESIGNED_ARCHITECTURES", "EPOCHS_BY_MODE", "MODELS_PER_SPLIT", "POPULATION_SIZES", "population_sizes", "PHASE3_BASE", "PHASE3_EXTENSIONS", "PHASE_NAMES", "moved_axes", "PhaseSpec", "build_phase",
    "phase1_configs", "phase2_configs", "phase3_configs", "phase4_configs",
]
