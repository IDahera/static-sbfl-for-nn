from dataclasses import dataclass
import torch
import torch.nn as nn
from helper.global_variables import DEVICE
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


def collect_activations(model: nn.Module,
                        layer: nn.Module,
                        loader: DataLoader,
                        target_class: int) -> Tuple[torch.Tensor, torch.Tensor]:

    activations = []
    target_preds = []

    def act_hook(module, _, output):
        # Store the full batch of activations
        # Move to CPU to save GPU memory
        activations.append(output.detach().cpu())

    hook = layer.register_forward_hook(act_hook)

    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Binary vector!
            # Move to CPU
            batch_target = (predicted == target_class).bool().cpu()
            target_preds.append(batch_target)

    hook.remove()  # Clean up the hook

    return torch.cat(activations, dim=0), torch.cat(target_preds, dim=0)


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
    print(f"Average neuron activation rate: {
          active_neurons.float().mean().item():.2%}")

    print("\nHit Spectrum Summary:")
    print(f"a_s range: [{hs.a_s.min().item()}, {hs.a_s.max().item()}]")
    print(f"a_f range: [{hs.a_f.min().item()}, {hs.a_f.max().item()}]")
    print(f"n_s range: [{hs.n_s.min().item()}, {hs.n_s.max().item()}]")
    print(f"n_f range: [{hs.n_f.min().item()}, {hs.n_f.max().item()}]")


def get_ochiai(
    hs: HitSpectrum
) -> torch.Tensor:
    a_s, a_f = hs.a_s, hs.a_f,
    n_f = hs.n_f

    numerator = a_s.float()
    denominator = torch.sqrt((a_f.float() + n_f.float())
                             * (a_f.float() + a_s.float()))

    result = torch.div(numerator, denominator)
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


def get_tarantula(
    hs: HitSpectrum
) -> torch.Tensor:
    a_s, a_f = hs.a_s, hs.a_f
    n_s, n_f = hs.n_s, hs.n_f

    numerator = torch.div(a_f.float(), (a_f.float() + n_f.float()))
    denominator_l = torch.div(a_f.float(), a_f.float() + n_f.float())
    denominator_r = torch.div(a_s.float(), a_s.float() + n_s.float())

    result = torch.div(numerator, denominator_l + denominator_r)

    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result
