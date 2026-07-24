#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Detect macOS syspolicyd overload that blocks fresh binary execution (#4230).

When syspolicyd (Gatekeeper assessment daemon) is overwhelmed by concurrent
compilation, freshly linked binaries hang at _dyld_start waiting for security
assessment. This blocks all cargo test execution and binary launches.

Standalone usage:  python3 scripts/check_syspolicyd.py
Integrated usage:  called from system_health_check.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


_SYSPOLICYD_CPU_THRESHOLD = 50.0


def _get_syspolicyd_cpu() -> float | None:
    """Return syspolicyd CPU percentage, or None if not queryable."""
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pcpu,comm"],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    for line in result.stdout.splitlines():
        if "syspolicyd" in line:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                try:
                    return float(parts[0])
                except ValueError:
                    pass
            break
    return 0.0


def _run_binary_canary() -> tuple[bool, str]:
    """Compile and run a tiny C binary. Returns (ok, detail)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as src:
            src.write('#include <stdio.h>\nint main(void){puts("ok");return 0;}\n')
            src_path = src.name
    except OSError as e:
        return False, f"canary setup failed: {e}"

    bin_path = src_path.replace(".c", "")
    try:
        subprocess.run(["cc", "-o", bin_path, src_path],
                       capture_output=True, check=True, timeout=10)
        result = subprocess.run([bin_path], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "ok" in result.stdout:
            return True, "fresh binary started successfully"
        return False, f"exit={result.returncode}, stdout={result.stdout!r}"
    except subprocess.TimeoutExpired:
        return False, "fresh binary timed out at startup (_dyld_start hang likely)"
    except subprocess.CalledProcessError as e:
        return False, f"compilation failed: {e}"
    finally:
        for p in (src_path, bin_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def run_syspolicyd_check(hc: object) -> dict:
    """Run the syspolicyd health check. Populates hc if provided.

    Returns a dict with status, syspolicyd_cpu_pct, canary_ok, detail.
    """
    if sys.platform != "darwin":
        result = {"status": "skip", "reason": "not macOS"}
        if hc is not None:
            hc.skip("syspolicyd check is macOS-only")  # type: ignore[union-attr]
            hc.set_check_result("syspolicyd_health", result)  # type: ignore[union-attr]
        return result

    cpu = _get_syspolicyd_cpu()
    if cpu is None:
        result = {"status": "skip", "reason": "ps failed"}
        if hc is not None:
            hc.skip("Could not query syspolicyd CPU usage")  # type: ignore[union-attr]
            hc.set_check_result("syspolicyd_health", result)  # type: ignore[union-attr]
        return result

    canary_ok, canary_detail = _run_binary_canary()
    overloaded = cpu > _SYSPOLICYD_CPU_THRESHOLD

    result = {
        "syspolicyd_cpu_pct": cpu,
        "canary_ok": canary_ok,
        "detail": canary_detail,
    }

    if overloaded and not canary_ok:
        result["status"] = "fail"
        msg = (f"syspolicyd at {cpu:.0f}% CPU and fresh binaries cannot start. "
               "Rust test binaries will hang at _dyld_start (#4230).")
        if hc is not None:
            hc.error(msg)  # type: ignore[union-attr]
    elif overloaded:
        result["status"] = "warn"
        msg = (f"syspolicyd at {cpu:.0f}% CPU (canary passed but large "
               "binaries may stall). Monitor for _dyld_start hangs (#4230).")
        if hc is not None:
            hc.warn(msg)  # type: ignore[union-attr]
    elif not canary_ok:
        result["status"] = "fail"
        msg = (f"Fresh binary canary failed ({canary_detail}). "
               "Test execution may be blocked by macOS security policy (#4230).")
        if hc is not None:
            hc.error(msg)  # type: ignore[union-attr]
    else:
        result["status"] = "pass"
        msg = f"syspolicyd healthy ({cpu:.0f}% CPU, canary passed)"
        if hc is not None:
            hc.ok(msg)  # type: ignore[union-attr]

    if hc is not None:
        hc.set_check_result("syspolicyd_health", result)  # type: ignore[union-attr]

    return result


if __name__ == "__main__":
    info = run_syspolicyd_check(None)
    cpu = info.get("syspolicyd_cpu_pct", "N/A")
    canary = "PASS" if info.get("canary_ok") else "FAIL"
    detail = info.get("detail", "")
    status = info.get("status", "unknown")
    sys.stdout.write(f"syspolicyd CPU: {cpu}%\n")
    sys.stdout.write(f"Binary canary:  {canary} ({detail})\n")
    sys.stdout.write(f"Status:         {status}\n")
    sys.exit(0 if status in ("pass", "skip") else 1)
