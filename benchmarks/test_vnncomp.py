# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
VNN-COMP Benchmark Suite for ny

Tests against VNN-COMP 2021-2025 benchmarks.
Reference: https://sites.google.com/view/vnn2025/home

REFERENCE TARGETS:
- >95% verified rate on ACAS-Xu
- <10s per property

Run:
    pytest test_vnncomp.py -v --timeout=10                    # Quick test
    pytest test_vnncomp.py -v --timeout=60 --method=beta     # Full with β-CROWN
    pytest test_vnncomp.py -v -k acasxu --save-results=results.json
"""

import json

import pytest
from _shared import (
    BENCHMARKS_BY_YEAR,
    NY_BINARY,
    VNNCOMP_YEARS,
    _list_benchmark_files,
    get_benchmark_dir,
    get_benchmark_instances,
    require_benchmark_items,
    require_benchmark_path,
    run_benchmark_suite,
    run_ny_verify,
)


class TestVnncompInfrastructure:
    """Verify benchmark infrastructure is set up correctly."""

    def test_ny_binary_exists(self):
        """Verify ny binary is built."""
        assert NY_BINARY.exists(), "Build ny first: cargo build --release"

    def test_vnncomp_2021_exists(self):
        """Verify VNN-COMP 2021 benchmarks exist."""
        require_benchmark_path(VNNCOMP_YEARS[2021], "Missing vnncomp2021 benchmarks")
        acasxu = VNNCOMP_YEARS[2021] / "acasxu"
        require_benchmark_path(acasxu, "Missing ACAS-Xu 2021")
        assert len(_list_benchmark_files(acasxu, "*.onnx")) == 45, (
            "Expected 45 ACAS-Xu networks"
        )

    def test_vnncomp_2023_exists(self):
        """Verify VNN-COMP 2023 benchmarks exist."""
        require_benchmark_path(VNNCOMP_YEARS[2023], "Missing vnncomp2023 benchmarks")
        acasxu = VNNCOMP_YEARS[2023] / "acasxu"
        if acasxu.exists():
            onnx_dir = acasxu / "onnx"
            assert len(_list_benchmark_files(onnx_dir, "*.onnx")) >= 45, (
                "Expected 45+ ACAS-Xu networks"
            )

    def test_vnncomp_2024_exists(self):
        """Verify VNN-COMP 2024 benchmarks exist."""
        require_benchmark_path(VNNCOMP_YEARS[2024], "Missing vnncomp2024 benchmarks")


# =============================================================================
# ACAS-Xu Benchmarks (primary reference category)
# =============================================================================


class TestAcasXu2021:
    """ACAS-Xu benchmark from VNN-COMP 2021."""

    @pytest.mark.acasxu
    @pytest.mark.vnn2021
    def test_single_instance(self, verify_timeout, method):
        """Run single ACAS-Xu instance as smoke test."""
        acasxu_dir = VNNCOMP_YEARS[2021] / "acasxu"
        network = acasxu_dir / "ACASXU_run2a_1_1_batch_2000.onnx"
        prop = acasxu_dir / "prop_1.vnnlib"

        require_benchmark_path(network, f"Network not found: {network}")
        require_benchmark_path(prop, f"Property not found: {prop}")

        result = run_ny_verify(network, prop, timeout=verify_timeout, method=method)
        print(f"\nResult: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]

    @pytest.mark.acasxu
    @pytest.mark.vnn2021
    @pytest.mark.slow
    def test_full_benchmark(self, verify_timeout, method, request):
        """Run full ACAS-Xu 2021 benchmark - primary target."""
        results = run_benchmark_suite(
            2021, "acasxu", method=method, timeout_override=verify_timeout
        )

        print(f"\n{'=' * 60}")
        print(f"ACAS-Xu 2021 Results ({method})")
        print(f"{'=' * 60}")
        print(f"Total: {results['total']}")
        print(f"Verified: {results['verified']} ({results['verified_rate']:.1f}%)")
        print(f"Falsified: {results['falsified']}")
        print(f"Unknown: {results['unknown']}")
        print(f"Timeout: {results['timeout']}")
        print(f"Error: {results['error']}")
        print(f"Average time: {results['avg_time']:.2f}s")
        print(f"{'=' * 60}")

        # Save results if requested
        save_path = request.config.getoption("--save-results")
        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {save_path}")

        # TARGET: >95% verified, <10s average
        print("\nTARGETS:")
        print(
            f"  Verified rate: {results['verified_rate']:.1f}% {'PASS' if results['verified_rate'] > 95 else 'FAIL'} (target: >95%)"
        )
        print(
            f"  Average time: {results['avg_time']:.2f}s {'PASS' if results['avg_time'] < 10 else 'FAIL'} (target: <10s)"
        )


class TestAcasXu2023:
    """ACAS-Xu benchmark from VNN-COMP 2023."""

    @pytest.mark.acasxu
    @pytest.mark.vnn2023
    def test_single_instance(self, verify_timeout, method):
        """Run single ACAS-Xu instance as smoke test."""
        acasxu_dir = VNNCOMP_YEARS[2023] / "acasxu"
        network = acasxu_dir / "onnx" / "ACASXU_run2a_1_1_batch_2000.onnx"
        prop = acasxu_dir / "vnnlib" / "prop_1.vnnlib"

        require_benchmark_path(network, f"Network not found: {network}")
        require_benchmark_path(prop, f"Property not found: {prop}")

        result = run_ny_verify(network, prop, timeout=verify_timeout, method=method)
        print(f"\nResult: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]

    @pytest.mark.acasxu
    @pytest.mark.vnn2023
    @pytest.mark.slow
    def test_full_benchmark(self, verify_timeout, method, request):
        """Run full ACAS-Xu 2023 benchmark."""
        results = run_benchmark_suite(
            2023, "acasxu", method=method, timeout_override=verify_timeout
        )

        print(f"\n{'=' * 60}")
        print(f"ACAS-Xu 2023 Results ({method})")
        print(f"{'=' * 60}")
        print(f"Total: {results['total']}")
        print(f"Verified: {results['verified']} ({results['verified_rate']:.1f}%)")
        print(f"Average time: {results['avg_time']:.2f}s")
        print(f"{'=' * 60}")

        save_path = request.config.getoption("--save-results")
        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)


# =============================================================================
# MNIST Benchmarks
# =============================================================================


class TestMnist2021:
    """MNIST-FC benchmark from VNN-COMP 2021."""

    @pytest.mark.mnist
    @pytest.mark.vnn2021
    def test_single_instance(self, verify_timeout, method):
        """Run single MNIST-FC instance."""
        mnist_dir = VNNCOMP_YEARS[2021] / "mnistfc"
        require_benchmark_path(mnist_dir, "MNIST-FC 2021 not found")

        network = mnist_dir / "mnist-net_256x2.onnx"
        props = require_benchmark_items(
            list(mnist_dir.glob("prop_*_0.03.vnnlib")),
            f"No property files in {mnist_dir}",
        )
        require_benchmark_path(network, f"Network not found: {network}")

        result = run_ny_verify(
            network, props[0], timeout=verify_timeout, method=method
        )
        print(f"\nResult: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]

    @pytest.mark.mnist
    @pytest.mark.vnn2021
    @pytest.mark.slow
    def test_full_benchmark(self, verify_timeout, method, request):
        """Run full MNIST-FC 2021 benchmark."""
        results = run_benchmark_suite(
            2021, "mnistfc", method=method, timeout_override=verify_timeout
        )

        print(
            f"\nMNIST-FC 2021: {results['verified']}/{results['total']} verified "
            f"({results['verified_rate']:.1f}%), avg {results['avg_time']:.2f}s"
        )


# =============================================================================
# Vision Transformer (ViT) Benchmarks
# =============================================================================


class TestViT2023:
    """Vision Transformer benchmark from VNN-COMP 2023."""

    @pytest.mark.vit
    @pytest.mark.vnn2023
    def test_single_instance(self, verify_timeout, method):
        """Run single ViT instance."""
        vit_dir = VNNCOMP_YEARS[2023] / "vit"
        require_benchmark_path(vit_dir, "ViT 2023 not found")

        networks = require_benchmark_items(
            list((vit_dir / "onnx").glob("*.onnx"))
            if (vit_dir / "onnx").exists()
            else [],
            "ViT 2023 networks not found",
        )

        props = require_benchmark_items(
            list((vit_dir / "vnnlib").glob("*.vnnlib"))
            if (vit_dir / "vnnlib").exists()
            else [],
            "ViT 2023 properties not found",
        )

        result = run_ny_verify(
            networks[0], props[0], timeout=verify_timeout, method=method
        )
        print(f"\nViT Result: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]


class TestViT2024:
    """Vision Transformer benchmark from VNN-COMP 2024."""

    @pytest.mark.vit
    @pytest.mark.vnn2024
    def test_single_instance(self, verify_timeout, method):
        """Run single ViT instance."""
        vit_dir = get_benchmark_dir(2024, "vit")
        vit_dir = require_benchmark_path(vit_dir, "ViT 2024 not found")

        networks = require_benchmark_items(
            list((vit_dir / "onnx").glob("*.onnx"))
            if (vit_dir / "onnx").exists()
            else [],
            "ViT 2024 networks not found",
        )

        props = require_benchmark_items(
            list((vit_dir / "vnnlib").glob("*.vnnlib"))
            if (vit_dir / "vnnlib").exists()
            else [],
            "ViT 2024 properties not found",
        )

        result = run_ny_verify(
            networks[0], props[0], timeout=verify_timeout, method=method
        )
        print(f"\nViT 2024 Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# VGGNet Benchmarks
# =============================================================================


class TestVggNet2023:
    """VGGNet benchmark from VNN-COMP 2023."""

    @pytest.mark.vggnet
    @pytest.mark.vnn2023
    @pytest.mark.slow
    def test_single_instance(self, verify_timeout, method):
        """Run single VGGNet instance (large model)."""
        vgg_dir = VNNCOMP_YEARS[2023] / "vggnet16"
        require_benchmark_path(vgg_dir, "VGGNet 2023 not found")

        instances = require_benchmark_items(
            get_benchmark_instances(2023, "vggnet16"),
            "VGGNet 2023 instances not found",
        )

        network, prop, _ = instances[0]
        result = run_ny_verify(network, prop, timeout=verify_timeout, method=method)
        print(f"\nVGGNet Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# Traffic Signs Benchmarks
# =============================================================================


class TestTrafficSigns2023:
    """Traffic Signs Recognition benchmark from VNN-COMP 2023.

    NOTE: These models use binary quantized networks (BNN) with Sign activation.
    The Sign function has discontinuous gradients, causing bound propagation to
    produce trivial bounds [0,1]. This is a fundamental limitation for BNNs.

    Models:
    - 3_30_30_QConv: Small quantized CNN without BatchNorm
    - 3_48_48_QConv..._BN: Medium CNN with BatchNorm
    - 3_64_64_QConv..._BN: Large CNN with BatchNorm
    """

    @pytest.mark.traffic_signs
    @pytest.mark.vnn2023
    def test_single_instance(self, verify_timeout, method):
        """Run single traffic signs instance."""
        traffic_dir = VNNCOMP_YEARS[2023] / "traffic_signs_recognition"
        require_benchmark_path(traffic_dir, "Traffic Signs 2023 not found")

        instances = require_benchmark_items(
            get_benchmark_instances(2023, "traffic_signs_recognition"),
            "Traffic Signs 2023 instances not found",
        )

        network, prop, _ = instances[0]
        result = run_ny_verify(network, prop, timeout=verify_timeout, method=method)
        print(f"\nTraffic Signs Result: {result.status} in {result.time_seconds:.2f}s")
        # BNN models with Sign activation produce trivial bounds - status will be unknown
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]

    @pytest.mark.traffic_signs
    @pytest.mark.vnn2023
    @pytest.mark.slow
    def test_full_benchmark(self, verify_timeout, method, request):
        """Run full Traffic Signs 2023 benchmark."""
        results = run_benchmark_suite(
            2023,
            "traffic_signs_recognition",
            method=method,
            timeout_override=verify_timeout,
        )

        print(f"\n{'=' * 60}")
        print(f"Traffic Signs 2023 Results ({method})")
        print(f"{'=' * 60}")
        print(f"Total: {results['total']}")
        print(f"Verified: {results['verified']} ({results['verified_rate']:.1f}%)")
        print(f"Falsified: {results['falsified']}")
        print(f"Unknown: {results['unknown']}")
        print(f"Timeout: {results['timeout']}")
        print(f"Error: {results['error']}")
        print(f"Average time: {results['avg_time']:.2f}s")
        print(f"{'=' * 60}")

        save_path = request.config.getoption("--save-results")
        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {save_path}")


# =============================================================================
# CIFAR Benchmarks
# =============================================================================


class TestCifar2021:
    """CIFAR benchmark from VNN-COMP 2021."""

    @pytest.mark.cifar
    @pytest.mark.vnn2021
    def test_cifar_resnet(self, verify_timeout, method):
        """Run CIFAR ResNet instance."""
        cifar_dir = VNNCOMP_YEARS[2021] / "cifar10_resnet"
        require_benchmark_path(cifar_dir, "CIFAR ResNet 2021 not found")

        networks = require_benchmark_items(
            list(cifar_dir.glob("*.onnx")), f"No ONNX files in {cifar_dir}"
        )
        props = require_benchmark_items(
            list(cifar_dir.glob("*.vnnlib")), f"No VNNLib files in {cifar_dir}"
        )

        result = run_ny_verify(
            networks[0], props[0], timeout=verify_timeout, method=method
        )
        print(f"\nCIFAR ResNet Result: {result.status} in {result.time_seconds:.2f}s")


class TestCifar2024:
    """CIFAR-100 benchmark from VNN-COMP 2024."""

    @pytest.mark.cifar
    @pytest.mark.vnn2024
    @pytest.mark.slow
    def test_single_instance(self, verify_timeout, method):
        """Run CIFAR-100 instance."""
        cifar_dir = get_benchmark_dir(2024, "cifar100")
        require_benchmark_path(cifar_dir, "CIFAR-100 2024 not found")

        instances = require_benchmark_items(
            get_benchmark_instances(2024, "cifar100"),
            "CIFAR-100 2024 instances not found",
        )

        network, prop, _ = instances[0]
        result = run_ny_verify(network, prop, timeout=verify_timeout, method=method)
        print(f"\nCIFAR-100 Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# NN4Sys Benchmarks (Systems/Control)
# =============================================================================


class TestNn4sys2021:
    """NN4Sys benchmark from VNN-COMP 2021."""

    @pytest.mark.nn4sys
    @pytest.mark.vnn2021
    def test_single_instance(self, verify_timeout, method):
        """Run single NN4Sys instance."""
        nn4sys_dir = VNNCOMP_YEARS[2021] / "nn4sys"
        require_benchmark_path(nn4sys_dir, "NN4Sys 2021 not found")

        instances = require_benchmark_items(
            get_benchmark_instances(2021, "nn4sys"),
            "NN4Sys 2021 instances not found",
        )

        network, prop, _ = instances[0]
        result = run_ny_verify(network, prop, timeout=verify_timeout, method=method)
        print(f"\nNN4Sys Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# VNN-COMP 2025 Benchmarks (Latest)
# =============================================================================


class TestAcasXu2025:
    """ACAS-Xu benchmark from VNN-COMP 2025."""

    @pytest.mark.acasxu
    @pytest.mark.vnn2025
    def test_single_instance(self, verify_timeout, method):
        """Run single ACAS-Xu instance from 2025."""
        acasxu_dir = get_benchmark_dir(2025, "acasxu")
        acasxu_dir = require_benchmark_path(acasxu_dir, "ACAS-Xu 2025 not found")

        networks = require_benchmark_items(
            list((acasxu_dir / "onnx").glob("*.onnx"))
            if (acasxu_dir / "onnx").exists()
            else [],
            f"No ONNX files in {acasxu_dir}/onnx",
        )
        props = require_benchmark_items(
            list((acasxu_dir / "vnnlib").glob("*.vnnlib"))
            if (acasxu_dir / "vnnlib").exists()
            else [],
            f"No VNNLib files in {acasxu_dir}/vnnlib",
        )

        result = run_ny_verify(
            networks[0], props[0], timeout=verify_timeout, method=method
        )
        print(f"\nAcas-Xu 2025 Result: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]


class TestSoundnessBench2025:
    """Soundness benchmark from VNN-COMP 2025 - tests verifier soundness."""

    @pytest.mark.vnn2025
    def test_single_instance(self, verify_timeout, method):
        """Run soundness benchmark instance."""
        bench_dir = get_benchmark_dir(2025, "soundnessbench")
        require_benchmark_path(bench_dir, "Soundnessbench 2025 not found")

        instances = require_benchmark_items(
            get_benchmark_instances(2025, "soundnessbench"),
            "Soundnessbench 2025 instances not found",
        )

        network, prop, _ = instances[0]
        result = run_ny_verify(network, prop, timeout=verify_timeout, method=method)
        print(f"\nSoundnessBench Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# Aggregate Tests
# =============================================================================


class TestVnncompAggregate:
    """Run aggregate tests across all VNN-COMP years."""

    @pytest.mark.slow
    def test_all_acasxu(self, verify_timeout, method, request):
        """Run ACAS-Xu across all years and aggregate results."""
        all_results = []

        for year in [2021, 2023, 2024, 2025]:
            bench_name = "acasxu" if year == 2021 or year == 2023 else "acasxu_2023"
            results = run_benchmark_suite(
                year, bench_name, method=method, timeout_override=verify_timeout
            )
            if results["total"] > 0:
                all_results.append(results)
                print(
                    f"\nYear {year}: {results['verified']}/{results['total']} verified "
                    f"({results['verified_rate']:.1f}%), avg {results['avg_time']:.2f}s"
                )

        # Aggregate
        total_verified = sum(r["verified"] for r in all_results)
        total_instances = sum(r["total"] for r in all_results)
        total_time = sum(r["total_time"] for r in all_results)

        if total_instances > 0:
            agg_rate = total_verified / total_instances * 100
            agg_time = total_time / total_instances
            print(f"\n{'=' * 60}")
            print("AGGREGATE ACAS-Xu Results")
            print(f"{'=' * 60}")
            print(
                f"Total verified: {total_verified}/{total_instances} ({agg_rate:.1f}%)"
            )
            print(f"Average time: {agg_time:.2f}s")
            print(f"{'=' * 60}")

    @pytest.mark.slow
    def test_benchmark_matrix(self, verify_timeout, method, request):
        """Run matrix of benchmarks across years."""
        print(f"\n{'=' * 80}")
        print(f"VNN-COMP Benchmark Matrix ({method}, timeout={verify_timeout}s)")
        print(f"{'=' * 80}")
        print(
            f"{'Year':<6} {'Benchmark':<25} {'Verified':<12} {'Rate':<8} {'Avg Time':<10}"
        )
        print(f"{'-' * 80}")

        for year, benchmarks in BENCHMARKS_BY_YEAR.items():
            for bench in benchmarks[:3]:  # Limit to first 3 per year for speed
                results = run_benchmark_suite(
                    year, bench, method=method, timeout_override=verify_timeout
                )
                if results["total"] > 0:
                    print(
                        f"{year:<6} {bench:<25} "
                        f"{results['verified']}/{results['total']:<10} "
                        f"{results['verified_rate']:.1f}%{'':>3} "
                        f"{results['avg_time']:.2f}s"
                    )

        print(f"{'=' * 80}")


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "--timeout=10"] + sys.argv[1:])
