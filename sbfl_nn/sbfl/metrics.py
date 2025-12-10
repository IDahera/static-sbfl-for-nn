from dataclasses import dataclass
import torch
import pickle
import os
import torch.nn as nn

from sbfl_nn.utils.files import ensure_output_directory
from torch.utils.data import DataLoader
from typing import Tuple

# TODO Add documentation

@dataclass
class HitSpectrum:
    a_s: torch.Tensor  # activated & success
    a_f: torch.Tensor  # activated & failure
    n_s: torch.Tensor  # not activated & success
    n_f: torch.Tensor  # not activated & failure
    layer_shape: Tuple[int, ...]


@dataclass
class SuspiciousnessResult:
    ochiai: torch.Tensor
    tarantula: torch.Tensor
    computation_time: float
    model_name: str = ""
    layer_name: str = ""


def get_activations_with_hooks(model, input_data):
    """
    Extract activations from all layers using forward hooks
    """
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks for each layer
    hooks = []
    for name, module in model.named_modules():
        # Check for nn.Linear and nn.Conv2d layers to capture both dense and convolutional layers
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Forward pass
    output = model(input_data)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations, output


def compute_hit_spectrum(activations: torch.Tensor,
                         targets: torch.Tensor,
                         threshold: float = 0.0) -> HitSpectrum:
    # Compute hit spectrum
    active = activations > threshold

    return HitSpectrum(
        a_s=torch.sum(torch.logical_and(targets, active), dim=0),
        a_f=torch.sum(torch.logical_and(~targets, active), dim=0),
        n_s=torch.sum(torch.logical_and(targets, ~active), dim=0),
        n_f=torch.sum(torch.logical_and(~targets, ~active), dim=0),
        layer_shape=activations.shape[1:]
    )


def log_diagnostics(activations: torch.Tensor,
                    target: torch.Tensor,
                    active_neurons: torch.Tensor,
                    hs) -> None:
    print(f"\nDiagnostic Information:")
    print(f"Activation shape: {activations.shape}")
    print(f"Correct predictions shape: {target.shape}")
    print(f"Active neurons shape: {active_neurons.shape}")
    acc = target.any(dim=1).float().mean().item()
    print(f"Accuracy: {acc:.2%}")
    print(
        f"Average neuron activation rate: {active_neurons.float().mean().item():.2%}")

    print("\nHit Spectrum Summary:")
    print(f"a_s range: [{hs.a_s.min().item()}, {hs.a_s.max().item()}]")
    print(f"a_f range: [{hs.a_f.min().item()}, {hs.a_f.max().item()}]")
    print(f"n_s range: [{hs.n_s.min().item()}, {hs.n_s.max().item()}]")
    print(f"n_f range: [{hs.n_f.min().item()}, {hs.n_f.max().item()}]")


def get_ochiai(hs: HitSpectrum) -> torch.Tensor:
    a_s, a_f = hs.a_s, hs.a_f
    n_f = hs.n_f

    numerator = a_s.float()
    denominator = torch.sqrt((a_f.float() + n_f.float())
                             * (a_f.float() + a_s.float()))

    result = torch.div(numerator, denominator)
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


def get_tarantula(hs: HitSpectrum) -> torch.Tensor:
    a_s, a_f = hs.a_s, hs.a_f
    n_s, n_f = hs.n_s, hs.n_f

    numerator = torch.div(a_f.float(), (a_f.float() + n_f.float()))
    denominator_l = torch.div(a_f.float(), a_f.float() + n_f.float())
    denominator_r = torch.div(a_s.float(), a_s.float() + n_s.float())

    result = torch.div(numerator, denominator_l + denominator_r)

    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def unsqueeze_tensors(activations: torch.Tensor,
                      targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        targets_copy = targets_copy.unsqueeze(
            1).expand(-1, activations_copy.shape[1])

    return activations_copy, targets_copy


def save_sus_values(output_path: str, sus_result: SuspiciousnessResult) -> None:
    """Save suspiciousness results to a pickle file"""
    ensure_output_directory(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        pickle.dump(sus_result, f)
