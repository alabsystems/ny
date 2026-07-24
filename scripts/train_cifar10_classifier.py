#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Train a small CIFAR-10 classifier for meaningful verification benchmarks.

Creates a trained MLP (3072 -> 100 -> 100 -> 10) that achieves reasonable accuracy.
This enables meaningful verification: a trained classifier has a real
decision boundary that we can verify for robustness.

Note: CIFAR-10 is much harder than MNIST. A small MLP typically achieves ~55-60%
accuracy (vs 97% on MNIST). This is expected behavior - CIFAR-10 needs CNNs for
high accuracy. But even 55% accuracy provides a meaningful decision boundary.

Usage:
    python scripts/train_cifar10_classifier.py

Output:
    tests/models/cifar10_mlp_2x100_trained.onnx  - Trained ONNX model
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Model output path
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class Cifar10MLP(nn.Module):
    """Small MLP for CIFAR-10: 3072 -> 100 -> 100 -> 10 with ReLU."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3072)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation (logits)
        return x


def train_model(epochs=20, batch_size=128, lr=0.001):
    """Train the CIFAR-10 classifier."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training device: {device}")

    # Data transforms - normalize to [0, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
    ])

    # Load CIFAR-10
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(
        root=os.path.join(MODELS_DIR, "..", "..", "data"),
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=os.path.join(MODELS_DIR, "..", "..", "data"),
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model, loss, optimizer
    model = Cifar10MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_test_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_acc = 100.0 * correct / total

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)

        test_acc = 100.0 * test_correct / test_total
        print(f"  Epoch {epoch+1}/{epochs}: Train acc={train_acc:.2f}%, Test acc={test_acc:.2f}%")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()

    # Restore best model
    model.load_state_dict(best_model_state)
    return model, best_test_acc


def export_to_onnx(model, output_path):
    """Export trained model to ONNX format using onnx library directly.

    This avoids torch.onnx.export compatibility issues with newer PyTorch versions.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model.eval()
    model = model.to("cpu")

    # Extract weights from the trained model
    state = model.state_dict()
    w1 = state["fc1.weight"].numpy()
    b1 = state["fc1.bias"].numpy()
    w2 = state["fc2.weight"].numpy()
    b2 = state["fc2.bias"].numpy()
    w3 = state["fc3.weight"].numpy()
    b3 = state["fc3.bias"].numpy()

    # Create ONNX initializers
    w1_init = numpy_helper.from_array(w1, name="w1")
    b1_init = numpy_helper.from_array(b1, name="b1")
    w2_init = numpy_helper.from_array(w2, name="w2")
    b2_init = numpy_helper.from_array(b2, name="b2")
    w3_init = numpy_helper.from_array(w3, name="w3")
    b3_init = numpy_helper.from_array(b3, name="b3")

    # Layer 1: Linear + ReLU
    gemm1 = helper.make_node(
        "Gemm", inputs=["input", "w1", "b1"], outputs=["fc1"],
        alpha=1.0, beta=1.0, transA=0, transB=1
    )
    relu1 = helper.make_node("Relu", inputs=["fc1"], outputs=["relu1"])

    # Layer 2: Linear + ReLU
    gemm2 = helper.make_node(
        "Gemm", inputs=["relu1", "w2", "b2"], outputs=["fc2"],
        alpha=1.0, beta=1.0, transA=0, transB=1
    )
    relu2 = helper.make_node("Relu", inputs=["fc2"], outputs=["relu2"])

    # Layer 3: Linear (no activation - logits)
    gemm3 = helper.make_node(
        "Gemm", inputs=["relu2", "w3", "b3"], outputs=["output"],
        alpha=1.0, beta=1.0, transA=0, transB=1
    )

    # Create graph
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3072])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    graph = helper.make_graph(
        [gemm1, relu1, gemm2, relu2, gemm3],
        "cifar10_mlp_2x100_trained",
        [input_tensor],
        [output_tensor],
        [w1_init, b1_init, w2_init, b2_init, w3_init, b3_init]
    )

    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_model.ir_version = 9

    onnx.save(onnx_model, output_path)
    print(f"\nExported trained model to: {output_path}")


def create_vnnlib_for_real_image(model, test_dataset, epsilon=0.02, sample_idx=0):
    """Create VNN-LIB property file for a real CIFAR-10 test image.

    Uses an actual test image instead of a synthetic center point.
    This creates more meaningful verification properties.
    """
    # Get a test image and its label
    image, true_label = test_dataset[sample_idx]
    image = image.view(-1).numpy()  # Flatten to 3072

    # Verify the model's prediction matches the true label
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(image).unsqueeze(0))
        pred = output.argmax(dim=1).item()

    if pred != true_label:
        print(f"  Sample {sample_idx} misclassified (pred={pred}, true={true_label})")
        print("  Searching for correctly classified sample...")
        # Find a correctly classified sample
        for idx in range(len(test_dataset)):
            img, lbl = test_dataset[idx]
            img = img.view(-1).numpy()
            with torch.no_grad():
                out = model(torch.tensor(img).unsqueeze(0))
                p = out.argmax(dim=1).item()
            if p == lbl:
                sample_idx = idx
                image = img
                true_label = lbl
                pred = p
                print(f"  Using sample {idx} (label={lbl})")
                break

    # CIFAR-10 class names for reference
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    lines = []
    lines.append("; VNN-LIB robustness property for trained CIFAR-10 classifier")
    lines.append(f"; Sample index: {sample_idx}")
    lines.append(f"; True label: {true_label} ({class_names[true_label]})")
    lines.append(f"; Model prediction: {pred} ({class_names[pred]})")
    lines.append(f"; Epsilon: {epsilon}")
    lines.append("")

    # Declare variables
    for i in range(3072):
        lines.append(f"(declare-const X_{i} Real)")
    lines.append("")
    for i in range(10):
        lines.append(f"(declare-const Y_{i} Real)")
    lines.append("")

    # Input constraints: L-infinity ball around the real image
    lines.append("; Input bounds (L-infinity ball around real CIFAR-10 image)")
    for i in range(3072):
        lower = max(0.0, float(image[i]) - epsilon)
        upper = min(1.0, float(image[i]) + epsilon)
        lines.append(f"(assert (>= X_{i} {lower:.6f}))")
        lines.append(f"(assert (<= X_{i} {upper:.6f}))")
    lines.append("")

    # Output constraint: unsafe if any other class >= true class
    lines.append("; Unsafe region: misclassification")
    lines.append("(assert (or")
    for i in range(10):
        if i != true_label:
            lines.append(f"  (>= Y_{i} Y_{true_label})")
    lines.append("))")

    output_path = os.path.join(
        MODELS_DIR,
        f"cifar10_trained_robustness_eps{epsilon:.3f}_sample{sample_idx}.vnnlib"
    )
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Created property: {output_path}")
    print(f"  Class: {class_names[true_label]} (label={true_label})")

    return output_path


def main():
    print("=" * 60)
    print("Training CIFAR-10 Classifier for Verification Benchmarks")
    print("=" * 60)
    print()
    print("Note: CIFAR-10 is much harder than MNIST for MLPs.")
    print("Expected accuracy: ~55-60% (CNNs needed for >80%)")
    print()

    # Train the model
    model, test_acc = train_model(epochs=20)
    print(f"\nBest test accuracy: {test_acc:.2f}%")

    # Export to ONNX
    output_path = os.path.join(MODELS_DIR, "cifar10_mlp_2x100_trained.onnx")
    export_to_onnx(model, output_path)

    # Create VNN-LIB properties for real test images
    print("\nCreating VNN-LIB properties for real CIFAR-10 images...")
    transform = transforms.ToTensor()
    test_dataset = datasets.CIFAR10(
        root=os.path.join(MODELS_DIR, "..", "..", "data"),
        train=False,
        download=True,
        transform=transform
    )

    # Create properties with different epsilons
    for eps in [0.01, 0.02, 0.05]:
        create_vnnlib_for_real_image(model, test_dataset, epsilon=eps, sample_idx=0)

    print()
    print("=" * 60)
    print("Training complete!")
    print()
    print("Verification examples:")
    print("  # CROWN verification")
    print("  ny verify tests/models/cifar10_mlp_2x100_trained.onnx \\")
    print("    --property tests/models/cifar10_trained_robustness_eps0.010_sample*.vnnlib --method crown")
    print()
    print("  # Compare to untrained model")
    print("  ny verify tests/models/cifar10_mlp_2x100.onnx \\")
    print("    --property tests/models/cifar10_robustness_eps0.020_label0.vnnlib --method crown")
    print("=" * 60)


if __name__ == "__main__":
    main()
