"""Cross-cutting helpers: devices, seeding, paths."""

from staticsbfl.utils.devices import DEVICE_CHOICES, device_option, resolve_device
from staticsbfl.utils.paths import (
    CACHE_DIR,
    DATA_DIR,
    VARIANTS_DIR,
    ensure_dir,
    variant_dir,
    variants_root,
    variant_activation_rates,
    variant_neurons,
    variant_correlation_log,
    variant_design,
    variant_models_csv,
    variant_predictions_csv,
    variant_predictions_dir,
    variant_model_path,
    variant_spectrum_path,
)
from staticsbfl.utils.seeding import DEFAULT_SEED, set_seed

__all__ = [
    "DEVICE_CHOICES",
    "device_option",
    "resolve_device",
    "CACHE_DIR",
    "DATA_DIR",
    "VARIANTS_DIR",
    "ensure_dir",
    "variant_dir",
    "variants_root",
    "variant_activation_rates",
    "variant_neurons",
    "variant_correlation_log",
    "variant_design",
    "variant_models_csv",
    "variant_predictions_csv",
    "variant_predictions_dir",
    "variant_model_path",
    "variant_spectrum_path",
    "DEFAULT_SEED",
    "set_seed",
]
