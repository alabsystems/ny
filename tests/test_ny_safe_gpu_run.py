# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "scripts" / "ny-safe-gpu-run"
ATTESTED_VMEM_KIB = 167_772_160
ATTESTED_VMEM_BYTES = ATTESTED_VMEM_KIB * 1024


def _guard_test_environment(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NY_GPU_LOCK_PATH"] = str(tmp_path / "gpu.lock")
    # The production default intentionally reserves 64 GiB. These integration
    # tests exercise process/cgroup behavior and must not depend on the size of
    # the host's temporary filesystem.
    env["NY_BUILD_MIN_FREE_KIB"] = "1"
    return env


def _fake_systemd_run(tmp_path: Path, env: dict[str, str]) -> Path:
    """Capture the transient-unit argv without requiring a user systemd."""
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    argument_log = tmp_path / "systemd-run.args"
    fake = fake_bin / "systemd-run"
    fake.write_text(
        "#!/bin/sh\n"
        ': > "$NY_SAFE_SYSTEMD_RUN_LOG"\n'
        'for argument do printf "%s\\n" "$argument" '
        '>> "$NY_SAFE_SYSTEMD_RUN_LOG"; done\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    fake_flock = fake_bin / "flock"
    fake_flock.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_flock.chmod(0o755)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["XDG_RUNTIME_DIR"] = str(tmp_path)
    env["NY_SAFE_SYSTEMD_RUN_LOG"] = str(argument_log)
    return argument_log


def _require_user_systemd() -> None:
    try:
        probe = subprocess.run(
            [
                "systemd-run",
                "--user",
                "--wait",
                "--collect",
                "--pipe",
                "--quiet",
                "--unit",
                f"ny-safe-pytest-probe-{os.getpid()}.service",
                "--",
                "/bin/true",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise AssertionError(
            "systemd-run is required for ny-safe-gpu-run integration tests"
        ) from error
    assert probe.returncode == 0, (
        "user systemd transient services are required for ny-safe-gpu-run "
        f"integration tests: {probe.stderr.strip()}"
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def external_guard_preserves_environment_status_and_attests_limits(
    tmp_path: Path,
) -> None:
    _require_user_systemd()
    env = _guard_test_environment(tmp_path)
    env["NY_SAFE_TEST_VALUE"] = "spaces * remain literal"
    result = subprocess.run(
        [
            str(GUARD),
            "/bin/bash",
            "-c",
            'cgroup="$(sed -n \'s/^0:://p\' /proc/self/cgroup)"; '
            'printf "vmem=%s\\nvalue=%s\\nmembership=%s\\n" '
            '"$(ulimit -v)" "$NY_SAFE_TEST_VALUE" "$cgroup"; '
            "grep '^Max address space' /proc/self/limits; "
            "for control in memory.high memory.max memory.swap.max pids.max cpu.max; do "
            'printf "%s=" "$control"; cat "/sys/fs/cgroup${cgroup}/${control}"; done; '
            "exit 23",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 23, result.stderr
    assert f"vmem={ATTESTED_VMEM_KIB}" in result.stdout
    rlimit_fields = next(
        line.split()
        for line in result.stdout.splitlines()
        if line.startswith("Max address space")
    )
    assert rlimit_fields == [
        "Max",
        "address",
        "space",
        str(ATTESTED_VMEM_BYTES),
        str(ATTESTED_VMEM_BYTES),
        "bytes",
    ]
    assert "value=spaces * remain literal" in result.stdout
    assert "/ny-build.slice/" in result.stdout
    assert "memory.high=68719476736" in result.stdout
    assert "memory.max=85899345920" in result.stdout
    assert "memory.swap.max=8589934592" in result.stdout
    assert "pids.max=4096" in result.stdout
    assert "cpu.max=1000000 100000" in result.stdout


@pytest.mark.parametrize(
    ("profile", "memory_high", "memory_max"),
    [
        (None, "68719476736", "85899345920"),
        ("gb10-80g", "68719476736", "85899345920"),
        ("wsl24-20g", "17179869184", "21474836480"),
    ],
)
def test_guard_applies_exact_reviewed_profile_to_leaf_service(
    tmp_path: Path,
    profile: str | None,
    memory_high: str,
    memory_max: str,
) -> None:
    env = _guard_test_environment(tmp_path)
    if profile is None:
        env.pop("NY_MEASURE_CONTAINMENT_PROFILE", None)
    else:
        env["NY_MEASURE_CONTAINMENT_PROFILE"] = profile
    argument_log = _fake_systemd_run(tmp_path, env)

    result = subprocess.run(
        [str(GUARD), "/bin/true"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = argument_log.read_text(encoding="utf-8").splitlines()
    high_properties = [
        argument
        for argument in arguments
        if argument.startswith("--property=MemoryHigh=")
    ]
    max_properties = [
        argument
        for argument in arguments
        if argument.startswith("--property=MemoryMax=")
    ]
    assert high_properties == [f"--property=MemoryHigh={memory_high}"]
    assert max_properties == [f"--property=MemoryMax={memory_max}"]
    assert arguments[-2:] == [str(ATTESTED_VMEM_KIB), "/bin/true"]


@pytest.mark.parametrize(
    "profile",
    ["", "20g", "GB10-80G", "wsl24-20g "],
)
def test_guard_rejects_unknown_containment_profile_before_launch(
    tmp_path: Path, profile: str
) -> None:
    env = _guard_test_environment(tmp_path)
    env["NY_MEASURE_CONTAINMENT_PROFILE"] = profile
    argument_log = _fake_systemd_run(tmp_path, env)

    result = subprocess.run(
        [str(GUARD), "/bin/true"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "NY_MEASURE_CONTAINMENT_PROFILE" in result.stderr
    assert not argument_log.exists(), "an invalid profile reached systemd-run"


def external_guard_preserves_literal_argument_boundaries(tmp_path: Path) -> None:
    _require_user_systemd()
    env = _guard_test_environment(tmp_path)
    arguments = [
        "",
        "spaces * remain literal",
        "--property=MemoryMax=infinity",
        "$HOME",
        "semi;colon",
        "line\nbreak",
    ]
    result = subprocess.run(
        [
            str(GUARD),
            "/usr/bin/python3",
            "-c",
            "import json, sys; print(json.dumps(sys.argv[1:]))",
            *arguments,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == arguments


def external_sigterm_kills_the_complete_guarded_process_tree(tmp_path: Path) -> None:
    _require_user_systemd()
    pid_file = tmp_path / "child.pid"
    descendant_pid_file = tmp_path / "descendant.pid"
    term_file = tmp_path / "term.seen"
    child_script = (
        f"echo $$ > {pid_file}; "
        f"(trap '' TERM; echo $BASHPID > {descendant_pid_file}; "
        "while :; do sleep 1; done) & "
        f"trap 'echo term > {term_file}; exit 0' TERM; "
        "wait"
    )
    env = _guard_test_environment(tmp_path)
    process = subprocess.Popen(
        [str(GUARD), "/bin/bash", "-c", child_script],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5.0
    while (
        not pid_file.exists() or not descendant_pid_file.exists()
    ) and time.monotonic() < deadline:
        time.sleep(0.02)
    if not pid_file.exists() or not descendant_pid_file.exists():
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f"guarded child did not start\nstdout={stdout}\nstderr={stderr}")
    child_pid = int(pid_file.read_text(encoding="utf-8").strip())
    descendant_pid = int(descendant_pid_file.read_text(encoding="utf-8").strip())

    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 143, (stdout, stderr)
    deadline = time.monotonic() + 2.0
    while _pid_alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_alive(child_pid), f"guard left child {child_pid} alive"
    assert not _pid_alive(descendant_pid), (
        f"guard left TERM-ignoring descendant {descendant_pid} alive"
    )
    assert term_file.exists(), "guarded child did not receive SIGTERM"


@pytest.mark.parametrize(
    "limit",
    [
        "0",
        "167772161",
        "unlimited",
        "-1",
        "999999999999999999999999",
        "not-an-integer",
    ],
)
def test_guard_rejects_unsafe_address_space_limits(limit: str) -> None:
    env = os.environ.copy()
    env["NY_GPU_VMEM_LIMIT_KIB"] = limit
    result = subprocess.run(
        [str(GUARD), "/bin/true"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "NY_GPU_VMEM_LIMIT_KIB" in result.stderr


@pytest.mark.parametrize(
    "reserve",
    ["0", "1073741825", "999999999999999999999999", "not-an-integer"],
)
def test_guard_rejects_unsafe_disk_reserves(reserve: str) -> None:
    env = os.environ.copy()
    env["NY_BUILD_MIN_FREE_KIB"] = reserve
    result = subprocess.run(
        [str(GUARD), "/bin/true"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "NY_BUILD_MIN_FREE_KIB" in result.stderr


def test_guard_refuses_low_disk_before_launch_and_probes_existing_parent(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    df_log = tmp_path / "df-path.txt"
    fake_df = fake_bin / "df"
    fake_df.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$2" > "$NY_SAFE_DF_LOG"\n'
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf 'fakefs 100 99 1 99%% /fake\\n'\n",
        encoding="utf-8",
    )
    fake_df.chmod(0o755)
    fake_flock = fake_bin / "flock"
    fake_flock.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_flock.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["XDG_RUNTIME_DIR"] = str(tmp_path)
    env["NY_SAFE_DF_LOG"] = str(df_log)
    env["NY_GPU_LOCK_PATH"] = str(tmp_path / "gpu.lock")
    env["NY_BUILD_MIN_FREE_KIB"] = "2"
    env["CARGO_TARGET_DIR"] = str(tmp_path / "missing" / "private-target")
    result = subprocess.run(
        [str(GUARD), "/bin/true"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 75
    assert "only 1 KiB free" in result.stderr
    assert "required reserve is 2 KiB" in result.stderr
    assert df_log.read_text(encoding="utf-8").strip() == str(tmp_path)
