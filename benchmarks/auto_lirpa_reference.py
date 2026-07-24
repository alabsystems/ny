#!/usr/bin/env python3
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Compute reference bounds using Auto-LiRPA for comparison with ny.

Usage: PYTHONPATH=research/repos/auto_LiRPA .venv/bin/python benchmarks/auto_lirpa_reference.py
"""
import json
import time

import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class ToyModel(torch.nn.Module):
    """Simple 2-layer network from Auto-LiRPA toy example.

    Architecture:
    - Linear: 2 -> 2 (w1)
    - ReLU
    - Linear: 2 -> 1 (w2)

    Weights:
    - w1 = [[1, -1], [2, -1]]
    - w2 = [[1, -1]]
    """
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2, bias=False)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(2, 1, bias=False)

        # Set weights explicitly
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([[1., -1.], [2., -1.]]))
            self.fc2.weight.copy_(torch.tensor([[1., -1.]]))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DeepModel(torch.nn.Module):
    """Deeper network for more interesting bounds comparison.

    Architecture:
    - Linear: 3 -> 4
    - ReLU
    - Linear: 4 -> 4
    - ReLU
    - Linear: 4 -> 2
    """
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 4, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4, 4, bias=True)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(4, 2, bias=True)

        # Set weights deterministically
        torch.manual_seed(42)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randn(4, 3) * 0.5)
            self.fc1.bias.copy_(torch.randn(4) * 0.1)
            self.fc2.weight.copy_(torch.randn(4, 4) * 0.5)
            self.fc2.bias.copy_(torch.randn(4) * 0.1)
            self.fc3.weight.copy_(torch.randn(2, 4) * 0.5)
            self.fc3.bias.copy_(torch.randn(2) * 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def compute_bounds(model, x, lower, upper, methods=None):
    """Compute bounds using various methods."""
    if methods is None:
        methods = ['IBP', 'CROWN', 'alpha-CROWN']
    results = {}

    # Wrap model
    lirpa_model = BoundedModule(model, torch.empty_like(x))

    # Compute forward pass
    pred = lirpa_model(x)
    results['prediction'] = pred.detach().numpy().tolist()

    # Set up perturbation with explicit bounds
    norm = float("inf")
    ptb = PerturbationLpNorm(norm=norm, x_L=lower, x_U=upper)
    bounded_x = BoundedTensor(x, ptb)

    for method in methods:
        start = time.perf_counter()
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=method)
        elapsed = time.perf_counter() - start

        results[method] = {
            'lower': lb.detach().numpy().flatten().tolist(),
            'upper': ub.detach().numpy().flatten().tolist(),
            'time_ms': elapsed * 1000
        }

    return results


def benchmark_toy_model():
    """Benchmark the toy model from Auto-LiRPA examples."""
    print("=== Toy Model (2-layer MLP) ===")

    model = ToyModel()

    # Input: center at [1, 1], bounds [-1, -2] to [2, 1]
    x = torch.tensor([[1., 1.]])
    lower = torch.tensor([[-1., -2.]])
    upper = torch.tensor([[2., 1.]])

    results = compute_bounds(model, x, lower, upper)

    # Print weights for reference
    print("\nWeights:")
    print(f"  fc1.weight = {model.fc1.weight.tolist()}")
    print(f"  fc2.weight = {model.fc2.weight.tolist()}")

    print("\nInput bounds:")
    print(f"  lower = {lower.tolist()}")
    print(f"  upper = {upper.tolist()}")

    print(f"\nPrediction at center: {results['prediction']}")

    for method in ['IBP', 'CROWN', 'alpha-CROWN']:
        r = results[method]
        print(f"\n{method}:")
        print(f"  lower = {r['lower']}")
        print(f"  upper = {r['upper']}")
        print(f"  time = {r['time_ms']:.3f} ms")

    return {
        'model': 'toy',
        'weights': {
            'fc1': model.fc1.weight.tolist(),
            'fc2': model.fc2.weight.tolist()
        },
        'input': {
            'center': x.tolist(),
            'lower': lower.tolist(),
            'upper': upper.tolist()
        },
        'results': results
    }


def benchmark_deep_model():
    """Benchmark a deeper model to show alpha-CROWN advantage."""
    print("\n\n=== Deep Model (3-layer MLP) ===")

    model = DeepModel()

    # Input: center at [0.5, 0.5, 0.5] with epsilon=0.1
    x = torch.tensor([[0.5, 0.5, 0.5]])
    epsilon = 0.1
    lower = x - epsilon
    upper = x + epsilon

    results = compute_bounds(model, x, lower, upper)

    # Print weights for reference
    print("\nWeights:")
    print(f"  fc1.weight = {model.fc1.weight.tolist()}")
    print(f"  fc1.bias = {model.fc1.bias.tolist()}")
    print(f"  fc2.weight = {model.fc2.weight.tolist()}")
    print(f"  fc2.bias = {model.fc2.bias.tolist()}")
    print(f"  fc3.weight = {model.fc3.weight.tolist()}")
    print(f"  fc3.bias = {model.fc3.bias.tolist()}")

    print("\nInput bounds:")
    print(f"  lower = {lower.tolist()}")
    print(f"  upper = {upper.tolist()}")

    print(f"\nPrediction at center: {results['prediction']}")

    for method in ['IBP', 'CROWN', 'alpha-CROWN']:
        r = results[method]
        width = [u - lo for lo, u in zip(r['lower'], r['upper'])]
        print(f"\n{method}:")
        print(f"  lower = {r['lower']}")
        print(f"  upper = {r['upper']}")
        print(f"  width = {width}")
        print(f"  time = {r['time_ms']:.3f} ms")

    return {
        'model': 'deep',
        'weights': {
            'fc1_weight': model.fc1.weight.tolist(),
            'fc1_bias': model.fc1.bias.tolist(),
            'fc2_weight': model.fc2.weight.tolist(),
            'fc2_bias': model.fc2.bias.tolist(),
            'fc3_weight': model.fc3.weight.tolist(),
            'fc3_bias': model.fc3.bias.tolist()
        },
        'input': {
            'center': x.tolist(),
            'lower': lower.tolist(),
            'upper': upper.tolist()
        },
        'results': results
    }


def benchmark_performance():
    """Run multiple iterations for performance comparison."""
    print("\n\n=== Performance Benchmarks ===")

    model = ToyModel()
    x = torch.tensor([[1., 1.]])
    lower = torch.tensor([[-1., -2.]])
    upper = torch.tensor([[2., 1.]])

    lirpa_model = BoundedModule(model, torch.empty_like(x))
    norm = float("inf")
    ptb = PerturbationLpNorm(norm=norm, x_L=lower, x_U=upper)
    bounded_x = BoundedTensor(x, ptb)

    # Warm-up
    for _ in range(5):
        lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
        lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')

    # Benchmark IBP
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
    ibp_time = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark CROWN
    start = time.perf_counter()
    for _ in range(n_iters):
        lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
    crown_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"\nToy Model ({n_iters} iterations):")
    print(f"  IBP:   {ibp_time:.3f} ms/iter")
    print(f"  CROWN: {crown_time:.3f} ms/iter")

    # Deep model
    deep = DeepModel()
    x_deep = torch.tensor([[0.5, 0.5, 0.5]])
    lower_deep = x_deep - 0.1
    upper_deep = x_deep + 0.1

    deep_model = BoundedModule(deep, torch.empty_like(x_deep))
    ptb_deep = PerturbationLpNorm(norm=norm, x_L=lower_deep, x_U=upper_deep)
    bounded_x_deep = BoundedTensor(x_deep, ptb_deep)

    # Warm-up
    for _ in range(5):
        deep_model.compute_bounds(x=(bounded_x_deep,), method='IBP')
        deep_model.compute_bounds(x=(bounded_x_deep,), method='CROWN')

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        deep_model.compute_bounds(x=(bounded_x_deep,), method='IBP')
    ibp_time_deep = (time.perf_counter() - start) / n_iters * 1000

    start = time.perf_counter()
    for _ in range(n_iters):
        deep_model.compute_bounds(x=(bounded_x_deep,), method='CROWN')
    crown_time_deep = (time.perf_counter() - start) / n_iters * 1000

    print(f"\nDeep Model ({n_iters} iterations):")
    print(f"  IBP:   {ibp_time_deep:.3f} ms/iter")
    print(f"  CROWN: {crown_time_deep:.3f} ms/iter")

    return {
        'toy': {'ibp_ms': ibp_time, 'crown_ms': crown_time},
        'deep': {'ibp_ms': ibp_time_deep, 'crown_ms': crown_time_deep}
    }


def main():
    torch.set_grad_enabled(False)

    all_results = {
        'tool': 'Auto-LiRPA',
        'torch_version': torch.__version__,
        'benchmarks': []
    }

    all_results['benchmarks'].append(benchmark_toy_model())
    all_results['benchmarks'].append(benchmark_deep_model())
    all_results['performance'] = benchmark_performance()

    # Save results to JSON for comparison
    output_path = 'benchmarks/auto_lirpa_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
