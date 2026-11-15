"""Argv-only process execution with deadlines and explicitly labelled counters."""
from __future__ import annotations

import os
import platform
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .evidence import ExperimentError, immutable_json


def cache_census(path: Path) -> dict[str, Any]:
    started = time.monotonic()
    total = files = 0
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ExperimentError(f"cache must not contain symlinks: {entry}")
        if entry.is_file():
            total += entry.stat().st_size
            files += 1
    return {"logical_bytes": total, "files": files,
            "scan_seconds": time.monotonic() - started,
            "metric": "logical_file_bytes_not_resident_memory"}


def _rss(pid: int) -> int | None:
    try:
        if platform.system() == "Linux":
            # Includes children that remain in this process group, without
            # mistaking a host-wide memory number for the query's memory.
            pgid = os.getpgid(pid)
            result = subprocess.run(["ps", "-e", "-o", "pgid=,rss="],
                                    capture_output=True, timeout=1, check=False)
            if result.returncode:
                return None
            return sum(int(parts[1]) * 1024 for line in result.stdout.splitlines()
                       if len(parts := line.split()) == 2 and int(parts[0]) == pgid)
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                capture_output=True, timeout=1, check=False)
        return int(result.stdout.strip()) * 1024 if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    # All directly managed children have their own group. Escaped daemon
    # descendants are outside this contract; such arms are prohibited.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic() + 0.4
    while time.monotonic() < end:
        time.sleep(0.02)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin can report EPERM for a group containing only unreaped
        # zombies. Treat that as finished only after checking actual liveness.
        observed = subprocess.run(["ps", "-e", "-o", "pgid=,stat="],
                                  capture_output=True, timeout=1, check=False)
        if observed.returncode or any(
                len(parts := line.split()) == 2 and int(parts[0]) == proc.pid
                and not parts[1].startswith(b"Z") for line in observed.stdout.splitlines()):
            raise ExperimentError("deadline cleanup refused for a live process group")


def run_command(argv: list[str], *, cwd: Path, env: dict[str, str],
                stdout: Path, stderr: Path, timeout_seconds: float,
                sample_seconds: float = 0.1) -> dict[str, Any]:
    """Whole command deadline includes startup and monitoring.

    wait4 counters are for the child and its waited descendants. Linux sampled
    RSS sums the live process group; macOS samples the direct child only. Both
    are reported independently. Sampling is diagnostic, never proof evidence.
    """
    if not argv or any(not isinstance(a, str) or "\0" in a for a in argv):
        raise ExperimentError("command must be a nonempty NUL-free argv")
    started = time.monotonic()
    deadline = started + timeout_seconds
    peak = samples = 0
    timed_out = interrupted = False
    usage = None
    proc = None
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(_sig: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, interrupt)
        # O_EXCL covers raw process logs as well as summary JSON.
        with stdout.open("xb") as out, stderr.open("xb") as err:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err,
                                    start_new_session=True, shell=False)
            while True:
                waited, status, observed = os.wait4(proc.pid, os.WNOHANG)
                if waited:
                    usage = observed
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    _terminate(proc)
                    _, status, usage = os.wait4(proc.pid, 0)
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                rss = _rss(proc.pid)
                if rss is not None:
                    peak = max(peak, rss)
                    samples += 1
                time.sleep(min(sample_seconds, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        interrupted = True
        if proc is not None:
            _terminate(proc)
            try:
                _, status, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
            except ChildProcessError:
                pass
    finally:
        signal.signal(signal.SIGTERM, previous)
    wall = time.monotonic() - started
    # Darwin ru_maxrss uses bytes; Linux uses KiB. Block counters are OS units,
    # not fictitious byte counts, and include waited child process accounting.
    usage_json = None if usage is None else {
        "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
        "max_rss_bytes": int(usage.ru_maxrss) * (1 if platform.system() == "Darwin" else 1024),
        "input_blocks": usage.ru_inblock, "output_blocks": usage.ru_oublock,
        "scope": "wait4_child_and_waited_descendants",
    }
    return {"argv": argv, "wall_seconds": wall, "timeout_seconds": timeout_seconds,
            "timed_out": timed_out, "interrupted": interrupted,
            "returncode": None if proc is None else proc.returncode,
            "resource_usage": usage_json,
            "sampled_rss": {"status": "measured" if samples else "unavailable",
                            "peak_bytes": peak if samples else None, "samples": samples,
                            "scope": "live_process_group" if platform.system() == "Linux" else "direct_child"},
            "disk_io": {"status": "measured_os_blocks" if usage else "unavailable",
                        "input_blocks": None if usage is None else usage.ru_inblock,
                        "output_blocks": None if usage is None else usage.ru_oublock,
                        "byte_conversion": "unavailable"},
            "vram": {"status": "unavailable", "reason": "no_bound_device_sampler"},
            "platform": platform.platform()}


def worker(request: dict[str, Any]) -> int:
    """Run inside ny_box, separating its wait/quiescence cost from query cost."""
    run_dir = Path(request["run_dir"])
    cache = Path(request["cache_dir"])
    before = cache_census(cache)
    response = run_command(request["argv"], cwd=Path(request["cwd"]),
                           env=request["environment"], stdout=run_dir / "solver.stdout",
                           stderr=run_dir / "solver.stderr",
                           timeout_seconds=request["timeout_seconds"],
                           sample_seconds=request["sample_seconds"])
    response["cache"] = {"before": before, "after": cache_census(cache)}
    response["schema"] = "ny_experiment_process_v1"
    immutable_json(run_dir / "process.json", response)
    return 124 if response["timed_out"] else (130 if response["interrupted"] else 0)
