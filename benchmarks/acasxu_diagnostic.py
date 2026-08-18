# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Explicit ACAS-Xu benchmark diagnostic for ny.

REFERENCE TARGETS:
- >95% verified rate
- <10s per property

Run with:
    cd benchmarks
    pytest acasxu_diagnostic.py -v --ny-benchmark-timeout=10
    pytest acasxu_diagnostic.py -v --ny-benchmark-timeout=60 \
        --ny-benchmark-method=beta  # with β-CROWN

Full benchmark:
    pytest acasxu_diagnostic.py -v --ny-benchmark-results=results.json
"""

import json
import math

import pytest
from _shared import (
    NY_BINARY,
    VNNCOMP_DIR,
    _list_benchmark_files,
    require_benchmark_path,
    run_ny_verify,
)

# ACAS-Xu directory
ACASXU_DIR = VNNCOMP_DIR / "acasxu"


def _require_ny_outcome(result, configured_timeout):
    """Reject invocation failures and validate the reported timing."""
    if result.status == "error":
        pytest.fail(f"ny verification failed: {result.error_message or 'unknown error'}")
    if result.status == "timeout":
        pytest.fail(
            f"ny exceeded the configured {configured_timeout}s diagnostic timeout"
        )
    assert result.status in {"verified", "falsified", "unknown"}, (
        f"Unexpected ny status: {result.status!r}"
    )
    assert math.isfinite(result.time_seconds) and result.time_seconds >= 0, (
        f"Invalid ny timing: {result.time_seconds!r}"
    )


class TestAcasXuBaseline:
    """Quick baseline tests to verify the benchmark infrastructure works."""

    def test_ny_binary_exists(self):
        """Verify ny binary is built."""
        assert NY_BINARY.exists(), "Build ny first: cargo build --release"

    def test_acasxu_benchmark_exists(self):
        """Verify ACAS-Xu benchmark files exist."""
        require_benchmark_path(ACASXU_DIR, "ACAS-Xu benchmarks not downloaded")
        networks = _list_benchmark_files(ACASXU_DIR, "*.onnx")
        properties = _list_benchmark_files(ACASXU_DIR, "*.vnnlib")
        assert len(networks) == 45, f"Expected 45 networks, got {len(networks)}"
        assert len(properties) == 10, f"Expected 10 properties, got {len(properties)}"

    def test_single_verification(self, verify_timeout, method):
        """Run a single verification to test infrastructure."""
        network = ACASXU_DIR / "ACASXU_run2a_1_1_batch_2000.onnx"
        prop = ACASXU_DIR / "prop_1.vnnlib"

        require_benchmark_path(network, f"Network not found: {network}")
        require_benchmark_path(prop, f"Property not found: {prop}")

        result = run_ny_verify(network, prop, timeout=verify_timeout, method=method)

        print(f"\nResult: {result.status}")
        print(f"Time: {result.time_seconds:.2f}s")
        if result.error_message:
            print(f"Error: {result.error_message}")

        # We don't assert verified here - just that it runs without crashing
        _require_ny_outcome(result, verify_timeout)


class TestAcasXuProperty1:
    """Test all 45 networks against property 1."""

    @pytest.mark.acasxu
    @pytest.mark.parametrize(
        "network_name",
        [
            f"ACASXU_run2a_{i}_{j}_batch_2000.onnx"
            for i in range(1, 6)
            for j in range(1, 10)
        ],
    )
    def test_prop1(self, network_name, verify_timeout, method):
        """Verify property 1 for each network."""
        network = ACASXU_DIR / network_name
        prop = ACASXU_DIR / "prop_1.vnnlib"

        require_benchmark_path(network, f"Network {network_name} not found")
        require_benchmark_path(prop, f"Property not found: {prop}")

        result = run_ny_verify(network, prop, timeout=verify_timeout, method=method)

        # Record result for aggregation
        pytest.current_result = result

        _require_ny_outcome(result, verify_timeout)

        # The diagnostic invocation must finish within its configured budget.
        assert result.time_seconds < verify_timeout, f"Timeout: {result.time_seconds}s"
        # Any sound terminal verdict is acceptable for this per-instance smoke test.


class TestAcasXuFullBenchmark:
    """Run the full ACAS-Xu benchmark suite."""

    @pytest.mark.acasxu
    @pytest.mark.slow
    def test_full_benchmark(self, verify_timeout, method, request):
        """
        Run all (network, property) pairs and compute aggregate metrics.

        This is the main benchmark to compare against α,β-CROWN.
        """
        results = []
        verified = 0
        falsified = 0
        unknown = 0
        timeout_count = 0
        error_count = 0
        total_time = 0.0

        # Load the instances CSV to know which (network, property) pairs to test
        instances_file = ACASXU_DIR / "acasxu_instances.csv"
        require_benchmark_path(
            instances_file, "Missing ACAS-Xu instance manifest for full benchmark"
        )
        import csv

        with open(instances_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            test_pairs = [(row[0], row[1]) for row in reader if len(row) >= 2]
        assert test_pairs, "ACAS-Xu instance manifest contained no benchmark pairs"

        print(f"\nRunning {len(test_pairs)} verification tasks...")
        print(f"Method: {method}, Timeout: {verify_timeout}s")
        print("-" * 60)

        for network_name, prop_name in test_pairs:
            network = ACASXU_DIR / network_name
            prop = ACASXU_DIR / prop_name

            require_benchmark_path(network, f"Network not found: {network}")
            require_benchmark_path(prop, f"Property not found: {prop}")

            result = run_ny_verify(
                network, prop, timeout=verify_timeout, method=method
            )
            results.append(result)

            total_time += result.time_seconds

            if result.status == "verified":
                verified += 1
                status_str = "✓"
            elif result.status == "falsified":
                falsified += 1
                status_str = "✗"
            elif result.status == "unknown":
                unknown += 1
                status_str = "?"
            elif result.status == "timeout":
                timeout_count += 1
                status_str = "T"
            else:
                error_count += 1
                status_str = "E"

            print(
                f"  {status_str} {network_name} × {prop_name}: {result.time_seconds:.2f}s"
            )

        # Aggregate metrics
        total = len(results)
        verified_rate = verified / total * 100 if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0

        print("-" * 60)
        print("\nRESULTS SUMMARY")
        print(f"  Total tasks:     {total}")
        print(f"  Verified:        {verified} ({verified_rate:.1f}%)")
        print(f"  Falsified:       {falsified}")
        print(f"  Unknown:         {unknown}")
        print(f"  Timeout:         {timeout_count}")
        print(f"  Error:           {error_count}")
        print(f"  Average time:    {avg_time:.2f}s")
        print(f"  Total time:      {total_time:.1f}s")

        print("\nTARGETS:")
        print(
            f"  Verified rate:   {verified_rate:.1f}% {'✓' if verified_rate > 95 else '✗'} (target: >95%)"
        )
        print(
            f"  Average time:    {avg_time:.2f}s {'✓' if avg_time < 10 else '✗'} (target: <10s)"
        )

        # Save results if requested
        save_path = request.config.getoption("--ny-benchmark-results")
        if save_path:
            output = {
                "method": method,
                "timeout": verify_timeout,
                "total": total,
                "verified": verified,
                "falsified": falsified,
                "unknown": unknown,
                "timeout_count": timeout_count,
                "error_count": error_count,
                "verified_rate": verified_rate,
                "average_time": avg_time,
                "total_time": total_time,
                "results": [
                    {
                        "network": r.network,
                        "property": r.property,
                        "status": r.status,
                        "time": r.time_seconds,
                    }
                    for r in results
                ],
            }
            with open(save_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to: {save_path}")

        # Assert targets
        assert total > 0, "Full ACAS-Xu benchmark contained no runnable instances"
        if error_count:
            pytest.fail(f"Full ACAS-Xu benchmark had {error_count} ny error(s)")
        if timeout_count:
            pytest.fail(
                f"Full ACAS-Xu benchmark had {timeout_count} configured-timeout outcome(s)"
            )
        assert verified_rate > 95, (
            f"Verified rate {verified_rate:.1f}% below target 95%"
        )
        assert avg_time < 10, f"Average time {avg_time:.2f}s above target 10s"


if __name__ == "__main__":
    # Quick test
    import sys

    pytest.main(
        [__file__, "-v", "--ny-benchmark-timeout=10"] + sys.argv[1:]
    )
