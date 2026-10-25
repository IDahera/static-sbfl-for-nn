"""The two hit-spectrum helpers of upstream ``3_models_get_sus.ipynb``, as a module.

Copied from the notebook cell "Hit Spectrum Helper" (upstream commit
``6e906ed``) so that a script can import them. Two changes, both marked
``PORT``: the notebook read the activation threshold and the verbosity from its
global ``sus_config``; here they are parameters with the notebook's values as
defaults (``activation-threshold: 0.0``, ``verbose: True`` -> ``False``, since
diagnostics print and only cost time), and the oracle of
:func:`helper.suspiciousness.collect_activations` is passed through.
"""

from typing import Tuple

import torch

from helper.suspiciousness import collect_activations, compute_hit_spectrum, log_diagnostics


def unsqueeze_tensors(activations: torch.Tensor,
                      targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    activations_copy = activations.clone()
    targets_copy = targets.clone()

    # Handle shapes based on layer type
    if len(activations.shape) == 4:  # Conv layer
        # For conv layer, we want to analyze each feature map position
        # Reshape activations to (batch_size, channels * height * width)
        b, c, h, w = activations_copy.shape
        activations_copy = activations_copy.view(b, c * h * w)
        targets_copy = targets_copy.unsqueeze(1).expand(-1, c * h * w)
    else:  # Dense layer
        targets_copy = targets_copy.unsqueeze(1).expand(-1, activations_copy.shape[1])

    return activations_copy, targets_copy


def hs_analysis(model, layer, test_loader, target_class=0,
                activation_threshold=0.0, verbose=False, oracle="target"):  # PORT: were sus_config globals
    """
    Analyze a specific layer of a model and compute hit spectrum.
    """
    # Collect activations
    activations, target_preds = collect_activations(model, layer, test_loader, target_class, oracle)  # PORT: oracle

    # Unsqueeze tensors
    activations, target_preds = unsqueeze_tensors(activations, target_preds)

    # Compute hit spectrum
    hs = compute_hit_spectrum(activations, target_preds, activation_threshold)  # PORT: was sus_config.activation_threshold

    if verbose:  # PORT: was sus_config.verbose
        log_diagnostics(activations,
                        target_preds,
                        activations > activation_threshold,
                        hs)

    return hs
