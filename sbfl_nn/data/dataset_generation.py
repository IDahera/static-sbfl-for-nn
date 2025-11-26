import torch
import pickle
import os

from itertools import product
from dataclasses import dataclass
from collections import namedtuple
from enum import Enum
from torch.utils.data import Dataset

from sbfl_nn.utils.files import ensure_output_directory
from sbfl_nn.utils.gpu import DEVICE


class LayerType(Enum):
    LINEAR = 1
    FLATTEN = 2
    CONV2D = 3
    CONV3D = 4
    MAXPOOL2D = 5
    AVGPOOL2D = 6
    MAXPOOL3D = 7
    AVGPOOL3D = 8


Features = namedtuple('Features', ['layer_index', 'layer_type', 'indices'])
Targets = namedtuple('Targets', ['target_ochiai', 'target_tarantula'])


class DatasetEntries:
    """Class to hold dataset entries."""

    def __init__(self,
                 layer_indices: torch.Tensor = torch.tensor([]).to(DEVICE),
                 layer_types: torch.Tensor = torch.tensor([]).to(DEVICE),
                 total_layers: torch.Tensor = torch.tensor([]).to(DEVICE),
                 neuron_indices: torch.Tensor = torch.tensor([]).to(DEVICE),
                 total_neurons: torch.Tensor = torch.tensor([]).to(DEVICE),

                 weight_l1_norms: torch.Tensor = torch.tensor([]).to(DEVICE),
                 weight_l2_norms: torch.Tensor = torch.tensor([]).to(DEVICE),
                 weight_mean: torch.Tensor = torch.tensor([]).to(DEVICE),
                 weight_std: torch.Tensor = torch.tensor([]).to(DEVICE),

                 bias_l1_norms: torch.Tensor = torch.tensor([]).to(DEVICE),
                 bias_l2_norms: torch.Tensor = torch.tensor([]).to(DEVICE),
                 bias_mean: torch.Tensor = torch.tensor([]).to(DEVICE),
                 bias_std: torch.Tensor = torch.tensor([]).to(DEVICE),

                 a_s: torch.Tensor = torch.tensor([]).to(DEVICE),
                 a_f: torch.Tensor = torch.tensor([]).to(DEVICE),
                 n_s: torch.Tensor = torch.tensor([]).to(DEVICE),
                 n_f: torch.Tensor = torch.tensor([]).to(DEVICE),

                 predecessor_layer_sizes: torch.Tensor = torch.tensor(
                     []).to(DEVICE),
                 successor_layer_sizes: torch.Tensor = torch.tensor(
                     []).to(DEVICE),
                 model_original_sizes: torch.Tensor = torch.tensor([]).to(DEVICE),
                 ochiai_values: torch.Tensor = torch.tensor([]).to(DEVICE),
                 tarantula_values: torch.Tensor = torch.tensor([]).to(DEVICE)):

        self.layer_indices = layer_indices
        self.layer_types = layer_types
        self.total_layers = total_layers
        self.neuron_indices = neuron_indices
        self.total_neurons = total_neurons

        self.weight_l1_norms = weight_l1_norms
        self.weight_l2_norms = weight_l2_norms
        self.weight_mean = weight_mean
        self.weight_std = weight_std

        self.bias_l1_norms = bias_l1_norms
        self.bias_l2_norms = bias_l2_norms
        self.bias_mean = bias_mean
        self.bias_std = bias_std

        self.predecessor_layer_sizes = predecessor_layer_sizes
        self.successor_layer_sizes = successor_layer_sizes

        self.model_original_sizes = model_original_sizes
        self.a_s = a_s
        self.a_f = a_f
        self.n_s = n_s
        self.n_f = n_f
        self.ochiai_values = ochiai_values
        self.tarantula_values = tarantula_values

    def expand_layer_indices(self, new_layer_indices):
        self.layer_indices = torch.cat(
            (self.layer_indices, new_layer_indices), dim=0)

    def expand_layer_types(self, new_layer_types):
        self.layer_types = torch.cat(
            (self.layer_types, new_layer_types), dim=0)

    def expand_total_layers(self, new_total_layers):
        self.total_layers = torch.cat(
            (self.total_layers, new_total_layers), dim=0)

    def expand_neuron_indices(self, new_neuron_indices):
        self.neuron_indices = torch.cat(
            (self.neuron_indices, new_neuron_indices), dim=0)

    def expand_weight_l1_norms(self, new_weight_l1_norms):
        self.weight_l1_norms = torch.cat(
            (self.weight_l1_norms, new_weight_l1_norms), dim=0)

    def expand_weight_l2_norms(self, new_weight_l2_norms):
        self.weight_l2_norms = torch.cat(
            (self.weight_l2_norms, new_weight_l2_norms), dim=0)

    def expand_weight_mean(self, new_weight_mean):
        self.weight_mean = torch.cat(
            (self.weight_mean, new_weight_mean), dim=0)

    def expand_weight_std(self, new_weight_std):
        self.weight_std = torch.cat(
            (self.weight_std, new_weight_std), dim=0)

    def expand_bias_l1_norms(self, new_bias_l1_norms):
        self.bias_l1_norms = torch.cat(
            (self.bias_l1_norms, new_bias_l1_norms), dim=0)

    def expand_bias_l2_norms(self, new_bias_l2_norms):
        self.bias_l2_norms = torch.cat(
            (self.bias_l2_norms, new_bias_l2_norms), dim=0)

    def expand_bias_mean(self, new_bias_mean):
        self.bias_mean = torch.cat(
            (self.bias_mean, new_bias_mean), dim=0)

    def expand_bias_std(self, new_bias_std):
        self.bias_std = torch.cat(
            (self.bias_std, new_bias_std), dim=0)

    def expand_total_neurons(self, new_total_neurons):
        self.total_neurons = torch.cat(
            (self.total_neurons, new_total_neurons), dim=0)

    def expand_model_original_sizes(self, new_model_original_sizes):
        self.model_original_sizes = torch.cat(
            (self.model_original_sizes, new_model_original_sizes), dim=0)

    def expand_a_s(self, new_a_s):
        self.a_s = torch.cat(
            (self.a_s, new_a_s), dim=0)

    def expand_a_f(self, new_a_f):
        self.a_f = torch.cat(
            (self.a_f, new_a_f), dim=0)

    def expand_n_s(self, new_n_s):
        self.n_s = torch.cat(
            (self.n_s, new_n_s), dim=0)

    def expand_n_f(self, new_n_f):
        self.n_f = torch.cat(
            (self.n_f, new_n_f), dim=0)

    def expand_ochiai_values(self, new_ochiai_values):
        self.ochiai_values = torch.cat(
            (self.ochiai_values, new_ochiai_values), dim=0)

    def expand_tarantula_values(self, new_tarantula_values):
        self.tarantula_values = torch.cat(
            (self.tarantula_values, new_tarantula_values), dim=0)

    def expand_predecessor_layer_sizes(self, new_predecessor_layer_sizes):
        self.predecessor_layer_sizes = torch.cat(
            (self.predecessor_layer_sizes, new_predecessor_layer_sizes), dim=0)

    def expand_successor_layer_sizes(self, new_successor_layer_sizes):
        self.successor_layer_sizes = torch.cat(
            (self.successor_layer_sizes, new_successor_layer_sizes), dim=0)

    def __safety_length_check(self):
        """Check if all stacks have the same length."""
        if len(self.layer_indices) != len(self.layer_types) or \
                len(self.layer_indices) != len(self.total_layers) or \
                len(self.layer_indices) != len(self.neuron_indices) or \
                len(self.layer_indices) != len(self.total_neurons) or \
                len(self.layer_indices) != len(self.predecessor_layer_sizes) or \
                len(self.layer_indices) != len(self.successor_layer_sizes) or \
                len(self.layer_indices) != len(self.model_original_sizes) or \
                len(self.layer_indices) != len(self.ochiai_values) or \
                len(self.layer_indices) != len(self.tarantula_values):

            raise ValueError(
                "All stacks must have the same length after expansion.")

    def process_layer_suspiciousness(self,
                                     model,
                                     sus_values,
                                     layer_name,
                                     layer_idx):
        """Process suspiciousness values for a single layer and append to data stacks.

        This function:
        1. Retrieves the layer from the model
        2. Gets the layer type (Linear, Conv, etc.)
        3. Extracts suspiciousness values for this layer
        4. Creates tensors with layer and neuron information
        5. Appends this data to the growing dataset stacks

        Parameters:
        -----------
        model : torch.nn.Module
            The neural network model containing the layer
        sus_values : dict
            Dictionary of suspiciousness values by layer
        layer_name : str
            Name of the current layer to process
        layer_idx : int or Tensor
            Index of the layer in the model's layer sequence
        """

        layer_type_id = DatasetEntriesHelper.get_layer_type_id(
            model,
            layer_name)

        if layer_type_id is None:
            # If layer type is not found, return without processing
            return

        # Get the layer weights and statistics
        layer = list(model.named_modules())[layer_idx][1]

        layer_weight_l1_norm = torch.norm(
            layer.weight, p=1).item() if hasattr(layer, 'weight') else None
        layer_weight_l2_norm = torch.norm(
            layer.weight, p=2).item() if hasattr(layer, 'weight') else None
        layer_weight_mean = layer.weight.mean().item() if hasattr(layer, 'weight') else None
        layer_weight_std = layer.weight.std().item() if hasattr(layer, 'weight') else None

        # Get the layer's biases and statistics
        layer_bias_l1_norm = torch.norm(layer.bias, p=1).item() if hasattr(
            layer, 'bias') and layer.bias is not None else None
        layer_bias_l2_norm = torch.norm(layer.bias, p=2).item() if hasattr(
            layer, 'bias') and layer.bias is not None else None
        layer_bias_mean = layer.bias.mean().item() if hasattr(
            layer, 'bias') and layer.bias is not None else None
        layer_bias_std = layer.bias.std().item() if hasattr(
            layer, 'bias') and layer.bias is not None else None

        # Get the layer distance to the output layer
        # layer_distance_to_output = len(list(model.named_modules())) - layer_idx - 1

        # get predecessor layer size
        predecessor_layer_size = DatasetEntriesHelper.get_predecessor_layer_size(
            model, layer_idx)

        # get successor layer size
        successor_layer_size = DatasetEntriesHelper.get_successor_layer_size(
            model, layer_idx)

        # Extract suspiciousness values and flatten them
        hs = sus_values[layer_name]["hit_spectrum"]
        a_s = hs.a_s.flatten()
        a_f = hs.a_f.flatten()
        n_s = hs.n_s.flatten()
        n_f = hs.n_f.flatten()

        ochiai_values = sus_values[layer_name]["ochiai"].flatten()
        tarantula_values = sus_values[layer_name]["tarantula"].flatten()

        # Get Layer Indices and Types
        num_elements = ochiai_values.numel()
        layer_indices, layer_types = DatasetEntriesHelper.create_layer_info_tensors(
            layer_idx,
            layer_type_id,
            num_elements)

        # Generate neuron indices (flattened sequential indices)
        indices_tensor = torch.arange(
            num_elements, dtype=torch.int32).unsqueeze(1).to(DEVICE)

        # Append data to stacks
        # Determine total layers and neurons in this layer
        total_layers = len(sus_values.keys())
        total_neurons_in_layer = num_elements

        # Create tensors to store these values, one per neuron
        total_layers_tensor = torch.full(
            (num_elements, 1), total_layers, dtype=torch.int32).to(DEVICE)
        total_neurons_tensor = torch.full(
            (num_elements, 1), total_neurons_in_layer, dtype=torch.int32).to(DEVICE)

        # Create tensors for weight statistics
        weight_l1_norm_tensor = torch.full(
            (num_elements, 1), layer_weight_l1_norm if layer_weight_l1_norm is not None else -1,
            dtype=torch.float32).to(DEVICE)
        weight_l2_norm_tensor = torch.full(
            (num_elements, 1), layer_weight_l2_norm if layer_weight_l2_norm is not None else -1,
            dtype=torch.float32).to(DEVICE)
        weight_mean_tensor = torch.full(
            (num_elements, 1), layer_weight_mean if layer_weight_mean is not None else -1,
            dtype=torch.float32).to(DEVICE)
        weight_std_tensor = torch.full(
            (num_elements, 1), layer_weight_std if layer_weight_std is not None else -1,
            dtype=torch.float32).to(DEVICE)

        # Create tensors for bias statistics
        bias_l1_norm_tensor = torch.full(
            (num_elements, 1), layer_bias_l1_norm if layer_bias_l1_norm is not None else -1,
            dtype=torch.float32).to(DEVICE)
        bias_l2_norm_tensor = torch.full(
            (num_elements, 1), layer_bias_l2_norm if layer_bias_l2_norm is not None else -1,
            dtype=torch.float32).to(DEVICE)
        bias_mean_tensor = torch.full(
            (num_elements, 1), layer_bias_mean if layer_bias_mean is not None else -1,
            dtype=torch.float32).to(DEVICE)
        bias_std_tensor = torch.full(
            (num_elements, 1), layer_bias_std if layer_bias_std is not None else -1,
            dtype=torch.float32).to(DEVICE)

        # Create tensors for predecessor and successor layer sizes
        predecessor_size_tensor = torch.full(
            (num_elements, 1), predecessor_layer_size if predecessor_layer_size is not None else -1,
            dtype=torch.int32).to(DEVICE)
        successor_size_tensor = torch.full(
            (num_elements, 1), successor_layer_size if successor_layer_size is not None else -1,
            dtype=torch.int32).to(DEVICE)

        # Calculate model original size directly from the model
        model_original_size = sum(p.numel() for p in model.parameters())
        model_original_sizes = torch.full(
            (num_elements, 1), model_original_size, dtype=torch.int64).to(DEVICE)

        # Expand all stacks with new data
        self.expand_layer_indices(layer_indices)
        self.expand_layer_types(layer_types)
        self.expand_total_layers(total_layers_tensor)
        self.expand_neuron_indices(indices_tensor)
        self.expand_total_neurons(total_neurons_tensor)

        self.expand_weight_l1_norms(weight_l1_norm_tensor)
        self.expand_weight_l2_norms(weight_l2_norm_tensor)
        self.expand_weight_mean(weight_mean_tensor)
        self.expand_weight_std(weight_std_tensor)

        self.expand_bias_l1_norms(bias_l1_norm_tensor)
        self.expand_bias_l2_norms(bias_l2_norm_tensor)
        self.expand_bias_mean(bias_mean_tensor)
        self.expand_bias_std(bias_std_tensor)

        self.expand_predecessor_layer_sizes(predecessor_size_tensor)
        self.expand_successor_layer_sizes(successor_size_tensor)

        self.expand_a_s(a_s)
        self.expand_a_f(a_f)
        self.expand_n_s(n_s)
        self.expand_n_f(n_f)
        self.expand_model_original_sizes(model_original_sizes)
        self.expand_ochiai_values(ochiai_values)
        self.expand_tarantula_values(tarantula_values)

        self.__safety_length_check()

        return


class DatasetEntriesHelper:

    def create_layer_info_tensors(layer_idx,
                                  layer_type_id,
                                  num_elements):
        """Create tensors with layer index and type information."""
        # Convert Tensor to scalar if needed
        if isinstance(layer_idx, torch.Tensor):
            # Safely handle empty tensors
            if layer_idx.numel() == 0:
                layer_idx = 0  # Default to 0 for empty tensors
            else:
                layer_idx = layer_idx.item()
        # Create a tensor filled with the layer index value, one per neuron
        layer_indices = torch.full(
            (num_elements, 1), layer_idx, dtype=torch.int32).to(DEVICE)
        # Create a tensor filled with the layer type value, one per neuron
        layer_types = torch.full(
            (num_elements, 1), layer_type_id.value, dtype=torch.int32).to(DEVICE)
        return layer_indices, layer_types

    def get_layer_from_model(model, layer_name):
        """Find a specific layer in the model by its name."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module

        return None

    def get_layer_type_id(model, layer_name):
        # Get the current layer from the model based on layer_name
        layer = DatasetEntriesHelper.get_layer_from_model(model, layer_name)

        if layer is None:
            # If layer is not found, return stacks unchanged
            return None

        # Get layer type identifier
        return get_layer_type(layer)

    def create_neuron_indices(shape):
        """Generate tensor of indices for each neuron based on layer shape."""
        # Create a list of all possible index combinations for the given shape
        # For example, a 2x3 shape would generate indices: [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        indices = list(product(*[range(s) for s in shape]))
        # Convert to tensor and move to the right device (CPU/GPU)
        return torch.tensor(indices, dtype=torch.int32).to(DEVICE)

    @staticmethod
    def get_predecessor_layer_size(model, current_layer_idx):
        """Get the number of neurons/features from the predecessor (input) layer."""
        layers = list(model.named_modules())

        # Handle edge case: if this is the first layer, predecessor is input size
        if current_layer_idx <= 1:  # Account for the model itself at index 0
            # For the first layer, look at its input features
            current_layer = layers[current_layer_idx][1]
            if hasattr(current_layer, 'in_features'):
                return current_layer.in_features
            elif hasattr(current_layer, 'in_channels'):
                return current_layer.in_channels
            else:
                return None

        # Find the previous layer that has neurons/features
        for i in range(current_layer_idx - 1, 0, -1):
            prev_layer = layers[i][1]

            # Check for Linear layer (most common)
            if hasattr(prev_layer, 'out_features'):
                return prev_layer.out_features
            # Check for Conv layers
            elif hasattr(prev_layer, 'out_channels'):
                return prev_layer.out_channels
            # Skip layers without output size (like activations, dropout)

        return None

    @staticmethod
    def get_successor_layer_size(model, current_layer_idx):
        """Get the number of neurons/features from the successor (output) layer."""
        layers = list(model.named_modules())

        # Handle edge case: if this is the last layer, successor size is output size
        if current_layer_idx >= len(layers) - 1:
            # For the last layer, look at its output features
            current_layer = layers[current_layer_idx][1]
            if hasattr(current_layer, 'out_features'):
                return current_layer.out_features
            elif hasattr(current_layer, 'out_channels'):
                return current_layer.out_channels
            else:
                return None

        # Find the next layer that has neurons/features
        for i in range(current_layer_idx + 1, len(layers)):
            next_layer = layers[i][1]

            # Check for Linear layer (most common)
            if hasattr(next_layer, 'in_features'):
                return next_layer.in_features
            # Check for Conv layers
            elif hasattr(next_layer, 'in_channels'):
                return next_layer.in_channels
            # Skip layers without input size (like activations, dropout)

        return None


class SusPredictionDataset(Dataset):
    def __init__(self,
                 stacks: DatasetEntries):
        """
        TODO

        Args:
            TODO
        """

        self.layer_indices = stacks.layer_indices
        self.layer_types = stacks.layer_types
        self.total_layers = stacks.total_layers
        self.neuron_indices = stacks.neuron_indices
        self.total_neurons = stacks.total_neurons

        self.weight_l1_norms = stacks.weight_l1_norms
        self.weight_l2_norms = stacks.weight_l2_norms
        self.weight_mean = stacks.weight_mean
        self.weight_std = stacks.weight_std

        self.bias_l1_norms = stacks.bias_l1_norms
        self.bias_l2_norms = stacks.bias_l2_norms
        self.bias_mean = stacks.bias_mean
        self.bias_std = stacks.bias_std

        self.a_s = stacks.a_s
        self.a_f = stacks.a_f
        self.n_s = stacks.n_s
        self.n_f = stacks.n_f

        self.predecessor_layer_sizes = stacks.predecessor_layer_sizes
        self.successor_layer_sizes = stacks.successor_layer_sizes
        self.model_original_sizes = stacks.model_original_sizes
        self.ochiai_values = stacks.ochiai_values
        self.tarantula_values = stacks.tarantula_values

    def __len__(self):
        return len(self.layer_indices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range.")

        return self.layer_indices[idx], \
            self.layer_types[idx], \
            self.total_layers[idx], \
            self.neuron_indices[idx], \
            self.total_neurons[idx], \
            self.weight_l1_norms[idx], \
            self.weight_l2_norms[idx], \
            self.weight_mean[idx], \
            self.weight_std[idx], \
            self.bias_l1_norms[idx], \
            self.bias_l2_norms[idx], \
            self.bias_mean[idx], \
            self.bias_std[idx], \
            self.predecessor_layer_sizes[idx], \
            self.successor_layer_sizes[idx], \
            self.model_original_sizes[idx], \
            self.a_s[idx], \
            self.a_f[idx], \
            self.n_s[idx], \
            self.n_f[idx], \
            self.ochiai_values[idx], \
            self.tarantula_values[idx]

    def get_numpy(self):
        """
        Convert the dataset entries to numpy arrays for easier processing.

        Returns:
            Tuple of numpy arrays: (features, targets)
        """
        features = torch.stack([
            self.layer_indices.flatten(),
            self.total_layers.flatten(),
            self.neuron_indices.flatten(),
            self.total_neurons.flatten(),
            self.layer_types.flatten(),

            self.weight_l1_norms.flatten(),
            self.weight_l2_norms.flatten(),
            self.weight_mean.flatten(),
            self.weight_std.flatten(),

            self.bias_l1_norms.flatten(),
            self.bias_l2_norms.flatten(),
            self.bias_mean.flatten(),
            self.bias_std.flatten(),

            self.predecessor_layer_sizes.flatten(),
            self.successor_layer_sizes.flatten(),

            self.layer_indices.flatten() / self.total_layers.flatten(),
            self.neuron_indices.flatten() / self.total_neurons.flatten(),
            self.model_original_sizes.flatten()
        ], dim=1).cpu().numpy()

        targets = torch.stack([
            self.a_s.flatten(),
            self.a_f.flatten(),
            self.n_s.flatten(),
            self.n_f.flatten(),
            self.ochiai_values.flatten(),
            self.tarantula_values.flatten()
        ], dim=1).cpu().numpy()

        return features, targets


def get_layer_type(layer: torch.nn.Module) -> LayerType:
    """
    Get the type of layer based on its class.

    Args:
        layer: A PyTorch layer (torch.nn.Module)

    Returns:
        LayerType: An enum representing the type of layer
    """
    if isinstance(layer, torch.nn.Linear):
        return LayerType.LINEAR
    elif isinstance(layer, torch.nn.Flatten):
        return LayerType.FLATTEN
    elif isinstance(layer, torch.nn.Conv2d):
        return LayerType.CONV2D
    elif isinstance(layer, torch.nn.Conv3d):
        return LayerType.CONV3D
    elif isinstance(layer, torch.nn.MaxPool2d):
        return LayerType.MAXPOOL2D
    elif isinstance(layer, torch.nn.AvgPool2d):
        return LayerType.AVGPOOL2D
    elif isinstance(layer, torch.nn.MaxPool3d):
        return LayerType.MAXPOOL3D
    elif isinstance(layer, torch.nn.AvgPool3d):
        return LayerType.AVGPOOL3D
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")


def expand_tensor_with_index(tensor, index):
    """
    Expand the last dimension of a tensor by adding an additional value (index).

    Args:
        tensor: A PyTorch tensor of arbitrary shape
        index: The index to be added to each element in the tensor

    Returns:
        A new tensor with the same shape as the input tensor, but with the last dimension expanded by 1.
    """
    # Get the original shape
    original_shape = tensor.shape

    # Create a new tensor with the last dimension expanded by 1
    expanded_tensor = torch.zeros(
        *original_shape[:-1], original_shape[-1] + 1, dtype=tensor.dtype, device=tensor.device)

    # Copy the original tensor values to the first part of the last dimension
    expanded_tensor[..., :original_shape[-1]] = tensor

    # Set the new value in the last position of the expanded dimension
    expanded_tensor[..., -1] = index

    return expanded_tensor


def create_layer_indices_and_values(tensor, layer_idx: int, layer_type: int):
    """
    Creates two tensors from the input tensor:
    1. A tensor containing layer index, layer type, and indices
    2. A tensor containing the original values

    Args:
        tensor: A PyTorch tensor of arbitrary shape
        layer_idx: Integer representing the layer index
        layer_type: Integer representing the layer type

    Returns:
        Tuple containing:
        - A 2D tensor with shape (n_elements, 2 + n_dims) where:
          - First column contains the layer_idx
          - Second column contains the layer_type
          - Remaining columns contain the original indices
        - A 1D tensor with the flattened values from the original tensor
        - The original tensor shape
    """
    # Get tensor shape and number of dimensions
    shape = tensor.shape

    # Create a flattened view of the values
    values = tensor.reshape(-1).to(DEVICE)

    # Generate all possible indices
    indices = list(product(*[range(s) for s in shape]))
    indices_tensor = torch.tensor(indices, dtype=torch.int32)

    # Create tensors for layer_idx and layer_type
    # (with the same number of rows as the flattened tensor)
    num_elements = tensor.numel()  # Number of elements in the tensor
    layer_idx_tensor = torch.full(
        (num_elements, 1), layer_idx, dtype=torch.int32)
    layer_type_tensor = torch.full(
        (num_elements, 1), layer_type, dtype=torch.int32)

    # Combine layer info and indices
    indices_info = torch.cat(
        [layer_idx_tensor, layer_type_tensor, indices_tensor], dim=1).to(DEVICE)

    return indices_info, values, shape


def save_dataset_values(output_path: str, dataset: dict[str, torch.Tensor]) -> None:
    """Save suspiciousness results to a pickle file"""
    ensure_output_directory(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
