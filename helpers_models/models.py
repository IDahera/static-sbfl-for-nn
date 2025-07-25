import torch
import torch.optim as optim
import os
import torch.nn.functional as F

from enum import Enum
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from helpers_general.logging import logging
from helpers_general.file_handling import ensure_output_directory
from helpers_general.gpu import DEVICE


class MLDataset(Enum):
    MNIST = 1
    FMNIST = 2
    CIFAR10 = 3


def load_dataset(dataset: MLDataset,
                 batch_size=64):
    """Load and return MNIST train and test dataloaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Set dataset to load
    available_dataset = {
        MLDataset.MNIST: datasets.MNIST,
        MLDataset.FMNIST: datasets.FashionMNIST,
        MLDataset.CIFAR10: datasets.CIFAR10
    }

    if dataset not in available_dataset:
        raise ValueError(f"Dataset {dataset} is not supported.")

    # Load Dataset
    train_dataset = available_dataset[dataset](
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = available_dataset[dataset](
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Initiate Dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


class DenseModel(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_sizes: list[int]):
        super(DenseModel, self).__init__()
        layers = []

        # Add input layer
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(input_size, layer_sizes[0]))

        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(layer_sizes[-1], output_size))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DenseModelHelper:
    def train(model: DenseModel,
              model_name: str,
              epochs: int,
              loader: DataLoader,
              optimizer: optim.Optimizer):

        for epoch in range(epochs):

            model.train()
            for data, label in tqdm(loader, desc=f"Training {model_name} ({epoch+1}/{epochs})"):
                data, label = data.to(DEVICE), label.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, label)
                loss.backward()
                optimizer.step()

        return model

    def evaluate(model: DenseModel,
                 test_loader: DataLoader) -> float:
        """Evaluate model accuracy on test data.

        This is a custom evaluation method, separate from the built-in eval() that PyTorch models have.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total

        return accuracy

    def save_model(model: DenseModel,
                   model_idx: int,
                   num_layers: int,
                   output_dir: str):
        """Save the model to disk."""
        output_file = f'{output_dir}/model_{model_idx}.pth'

        if os.path.exists(output_file):
            os.remove(output_file)
            logging.info(f"Deleted existing file: {output_file}")

        if ensure_output_directory(output_dir):
            torch.save(model, output_file)
            logging.info(
                f"Model {model_idx} saved: {num_layers} layers")
        else:
            logging.error("Cannot proceed due to directory issues")


