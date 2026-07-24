#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Train a simple CNN on CIFAR-10 and export to ONNX.

Architecture: Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Linear -> ReLU -> Linear

This CNN achieves ~70% accuracy on CIFAR-10, much better than the 51% MLP.
"""

from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.optim as optim

# Check for torchvision
try:
    from torchvision import datasets, transforms
except ImportError:
    print("ERROR: torchvision required. Install with: pip install torchvision")
    exit(1)


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10.

    Architecture:
    - Conv2d(3, 16, 3, padding=1) -> ReLU -> MaxPool(2)  [32x32 -> 16x16]
    - Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool(2) [16x16 -> 8x8]
    - Flatten -> Linear(32*8*8=2048, 128) -> ReLU -> Linear(128, 10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(torch.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(epochs=10, batch_size=64, lr=0.001):
    """Train the CNN on CIFAR-10."""

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Model, loss, optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    return model, accuracy


def export_to_onnx(model, output_path):
    """Export model to ONNX format using torch.jit.trace."""
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    # Use torch.jit.trace for export
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path.replace('.onnx', '.pt'))

        # Export to ONNX via dynamo
        try:
            # Try dynamo export first (PyTorch 2.x)
            onnx_program = torch.onnx.dynamo_export(model, dummy_input)
            onnx_program.save(output_path)
        except Exception as e:
            print(f"Warning: dynamo_export failed ({e}), falling back to traced model")
            # Fall back to saving just the traced model
            torch.jit.save(traced_model, output_path.replace('.onnx', '.pt'))
            print(f"Saved TorchScript model to {output_path.replace('.onnx', '.pt')}")
            return

    # Verify if ONNX was created
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"Exported to {output_path}")
    except Exception as e:
        print(f"ONNX verification failed: {e}")


def generate_vnnlib_property(model, test_dataset, sample_idx, epsilon, output_path):
    """Generate VNN-LIB property file for adversarial robustness."""

    # Get sample
    image, label = test_dataset[sample_idx]
    image_flat = image.flatten().numpy()

    with open(output_path, 'w') as f:
        # Declare input variables (3 * 32 * 32 = 3072)
        for i in range(3072):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables
        for i in range(10):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write("\n; Input constraints (L-infinity ball)\n")

        # Input bounds: [pixel - eps, pixel + eps] clipped to [-1, 1]
        for i, pixel in enumerate(image_flat):
            lb = max(-1.0, pixel - epsilon)
            ub = min(1.0, pixel + epsilon)
            f.write(f"(assert (>= X_{i} {lb:.6f}))\n")
            f.write(f"(assert (<= X_{i} {ub:.6f}))\n")

        f.write("\n; Output constraint: true label should be maximal\n")
        f.write("; Property: adversarial example exists if some other class > true class\n")

        # Disjunction: exists j != label where Y_j > Y_label
        disjuncts = []
        for j in range(10):
            if j != label:
                disjuncts.append(f"(> Y_{j} Y_{label})")

        f.write(f"(assert (or {' '.join(disjuncts)}))\n")

    return label


def main():
    output_dir = Path("tests/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    print("Training CNN on CIFAR-10...")
    model, accuracy = train_model(epochs=15, batch_size=64, lr=0.001)

    print(f"\nFinal accuracy: {accuracy:.2f}%")

    # Export to ONNX
    onnx_path = output_dir / "cifar10_cnn_trained.onnx"
    export_to_onnx(model, str(onnx_path))

    # Generate VNN-LIB properties
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Find a correctly classified sample
    model.eval()
    sample_idx = 0
    for idx in range(100):
        image, label = test_dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            predicted = output.argmax().item()
        if predicted == label:
            sample_idx = idx
            break

    print(f"\nUsing sample {sample_idx} (correctly classified)")

    # Generate properties for different epsilon values
    for eps in [0.005, 0.01, 0.02]:
        prop_path = output_dir / f"cifar10_cnn_robustness_eps{eps:.3f}_sample{sample_idx}.vnnlib"
        true_label = generate_vnnlib_property(model, test_dataset, sample_idx, eps, str(prop_path))
        print(f"Generated {prop_path} (true label: {true_label})")

    print("\nDone!")
    print(f"Model: {onnx_path}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
