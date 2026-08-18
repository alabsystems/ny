# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORECARD_SCRIPT = REPO_ROOT / "scripts" / "measure_ny_scorecard.sh"
RUN_ID = "20260718T120000Z-test"
AY_REV = "1560972ade2b04a702dfbd13a2de5444ea216009"
ATTESTED_VMEM_KIB = 167_772_160

# (memory.high, memory.max) per containment profile; must track
# CONTAINMENT_PROFILES in scripts/ny_measurement_provenance.py.
CONTAINMENT_PROFILE_BYTES = {
    "gb10-80g": (64 * 1024**3, 80 * 1024**3),
    "wsl24-20g": (16 * 1024**3, 20 * 1024**3),
}


def _assert_measurement_row(
    row: list[str],
    expected_prefix: list[str],
    run_id: str,
) -> None:
    assert len(row) == 7
    assert row[:5] == expected_prefix
    assert row[5].isdigit()
    assert int(row[5]) >= 0
    assert row[6] == run_id


def _run(*command: str, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _init_git_repo(path: Path) -> None:
    _run("git", "init", "-q", cwd=path)
    _run("git", "config", "user.name", "NY Test", cwd=path)
    _run("git", "config", "user.email", "ny-test@example.invalid", cwd=path)


def _lower_child_vmem_below_attestation() -> None:
    """Keep safety-test children contained while making attestation false."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    attested_bytes = ATTESTED_VMEM_KIB * 1024
    if soft == resource.RLIM_INFINITY or soft >= attested_bytes:
        lowered_soft = attested_bytes - 4096
    elif soft > 4096:
        lowered_soft = soft - 4096
    else:
        lowered_soft = soft
    try:
        resource.setrlimit(resource.RLIMIT_AS, (lowered_soft, hard))
    except (OSError, ValueError):
        # Darwin exposes a synthetic RLIMIT_AS that cannot be changed. Its
        # `ulimit -v` remains `unlimited`, already distinct from attestation.
        pass


def _install_fake_gpu_guard(tmp_path: Path) -> Path:
    """Install a portable contained-child attestation fixture."""
    fake_bin = tmp_path / "guard-bin"
    fake_bin.mkdir(exist_ok=True)
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        '#!/bin/bash\nset -eu\nbuiltin ulimit -v 167772160\nexec "$@"\n',
        encoding="utf-8",
    )
    guard.chmod(0o755)
    return fake_bin


def _install_fake_containment_snapshot(
    tmp_path: Path,
    scripts: Path,
    *,
    expected_cpus: int = 10,
    profile: str = "gb10-80g",
) -> None:
    """Patch only the private fixture copy to read a synthetic cgroup tree."""
    assert expected_cpus in {10, 20}
    memory_high, memory_max = CONTAINMENT_PROFILE_BYTES[profile]
    uid = os.getuid()
    cgroup_root = tmp_path / "fixture-cgroup"
    policy_membership = (
        f"/user.slice/user-{uid}.slice/user@{uid}.service/ny.slice/ny-build.slice"
    )
    membership = f"{policy_membership}/ny-safe-gpu-{uid}-1234-5678.service"
    policy = cgroup_root / policy_membership.lstrip("/")
    current = cgroup_root / membership.lstrip("/")
    current.mkdir(parents=True)
    (cgroup_root / "cgroup.controllers").write_text(
        "cpu memory pids\n", encoding="ascii"
    )
    policy_values = {
        "memory.high": str(memory_high),
        "memory.max": str(memory_max),
        "memory.swap.max": str(8 * 1024**3),
        "pids.max": "4096",
        "cpu.max": f"{expected_cpus * 100_000} 100000",
    }
    for name, value in policy_values.items():
        # The host slice is deliberately unconfigured. The real guard installs
        # every reviewed control on its exact transient service.
        parent_value = "max 100000" if name == "cpu.max" else "max"
        (policy / name).write_text(f"{parent_value}\n", encoding="ascii")
        (current / name).write_text(f"{value}\n", encoding="ascii")

    proc_cgroup = tmp_path / "fixture-proc-self-cgroup"
    proc_cgroup.write_text(f"0::{membership}\n", encoding="ascii")
    proc_mountinfo = tmp_path / "fixture-proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"39 29 0:33 / {cgroup_root} rw,nosuid,nodev - cgroup2 cgroup2 rw\n",
        encoding="ascii",
    )

    replacements = {
        scripts / "measure_ny_scorecard.sh": {
            'readonly scorecard_proc_cgroup="/proc/self/cgroup"': (
                f'readonly scorecard_proc_cgroup="{proc_cgroup}"'
            ),
            'readonly scorecard_proc_mountinfo="/proc/self/mountinfo"': (
                f'readonly scorecard_proc_mountinfo="{proc_mountinfo}"'
            ),
            'readonly scorecard_cgroup_root="/sys/fs/cgroup"': (
                f'readonly scorecard_cgroup_root="{cgroup_root}"'
            ),
        },
        scripts / "ny_measurement_provenance.py": {
            'PROC_SELF_CGROUP = Path("/proc/self/cgroup")': (
                f'PROC_SELF_CGROUP = Path("{proc_cgroup}")'
            ),
            'PROC_SELF_MOUNTINFO = Path("/proc/self/mountinfo")': (
                f'PROC_SELF_MOUNTINFO = Path("{proc_mountinfo}")'
            ),
            'CGROUP_V2_ROOT = Path("/sys/fs/cgroup")': (
                f'CGROUP_V2_ROOT = Path("{cgroup_root}")'
            ),
        },
    }
    for path, path_replacements in replacements.items():
        source = path.read_text(encoding="utf-8")
        for original, replacement in path_replacements.items():
            assert original in source
            source = source.replace(original, replacement, 1)
        path.write_text(source, encoding="utf-8")


def external_scorecard_reexecs_through_gpu_guard_before_measurement(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    guard_log = tmp_path / "guard.log"
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        "printf 'wrapped=%s\\n' \"${NY_MEASURE_SAFE_GPU_WRAPPED:-}\" "
        '> "$MEASUREMENT_GUARD_LOG"\n'
        "printf 'expected_cpus=%s\\n' "
        '"${NY_MEASURE_EXPECTED_CPUS:-}" >> "$MEASUREMENT_GUARD_LOG"\n'
        'printf \'arg=%s\\n\' "$@" >> "$MEASUREMENT_GUARD_LOG"\n',
        encoding="utf-8",
    )
    guard.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    environment["MEASUREMENT_GUARD_LOG"] = str(guard_log)
    # Python tooling commonly sets this. It must not be confused with the
    # shell startup variable named exactly ENV.
    environment["VIRTUAL_ENV"] = str(tmp_path / ".venv")
    environment.pop("NY_MEASURE_SAFE_GPU_WRAPPED", None)
    environment.pop("NY_MEASURE_EXPECTED_CPUS", None)

    result = subprocess.run(
        ["/bin/bash", str(SCORECARD_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        preexec_fn=_lower_child_vmem_below_attestation,
    )

    assert result.returncode == 0, result.stderr
    assert guard_log.read_text(encoding="utf-8").splitlines() == [
        "wrapped=1",
        "expected_cpus=10",
        "arg=/bin/bash",
        f"arg={SCORECARD_SCRIPT.resolve()}",
    ]


@pytest.mark.parametrize(
    "raw",
    ["", "0", "010", "11", "21", "+20", "20 ", "ten"],
)
def external_scorecard_rejects_invalid_expected_cpu_selector_before_guard(
    tmp_path: Path,
    raw: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    guard_log = tmp_path / "guard-called"
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        '#!/bin/bash\nprintf called > "$MEASUREMENT_GUARD_LOG"\n',
        encoding="utf-8",
    )
    guard.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BASH_FUNC_")
        and key not in {"BASH_ENV", "ENV", "BASHOPTS", "SHELLOPTS"}
    }
    environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    environment["MEASUREMENT_GUARD_LOG"] = str(guard_log)
    environment["NY_MEASURE_EXPECTED_CPUS"] = raw

    result = subprocess.run(
        ["/bin/bash", str(SCORECARD_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "NY_MEASURE_EXPECTED_CPUS must be exactly 10 or 20" in result.stderr
    assert result.stdout == ""
    assert not guard_log.exists()


@pytest.mark.parametrize(
    "raw",
    ["", "gb10", "wsl24-20G", "80g", " wsl24-20g", "gb10-80g ", "default"],
)
def external_scorecard_rejects_invalid_containment_profile_before_guard(
    tmp_path: Path,
    raw: str,
) -> None:
    """A near-miss profile name must fail closed, never fall back to a lane."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    guard_log = tmp_path / "guard-called"
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        '#!/bin/bash\nprintf called > "$MEASUREMENT_GUARD_LOG"\n',
        encoding="utf-8",
    )
    guard.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BASH_FUNC_")
        and key not in {"BASH_ENV", "ENV", "BASHOPTS", "SHELLOPTS"}
    }
    environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    environment["MEASUREMENT_GUARD_LOG"] = str(guard_log)
    environment["NY_MEASURE_CONTAINMENT_PROFILE"] = raw

    result = subprocess.run(
        ["/bin/bash", str(SCORECARD_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert (
        "NY_MEASURE_CONTAINMENT_PROFILE must be exactly gb10-80g or wsl24-20g"
        in result.stderr
    )
    assert result.stdout == ""
    assert not guard_log.exists()


def external_scorecard_refuses_measurement_without_gpu_guard(tmp_path: Path) -> None:
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    environment = os.environ.copy()
    environment["PATH"] = str(empty_bin)
    environment.pop("NY_MEASURE_SAFE_GPU_WRAPPED", None)

    result = subprocess.run(
        ["/bin/bash", str(SCORECARD_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        preexec_fn=_lower_child_vmem_below_attestation,
    )

    assert result.returncode == 2
    assert "refusing an unguarded run" in result.stderr
    assert result.stdout == ""


def external_scorecard_rejects_forged_gpu_guard_marker(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    guard_log = tmp_path / "guard-called"
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        '#!/bin/bash\nprintf called > "$MEASUREMENT_GUARD_LOG"\n',
        encoding="utf-8",
    )
    guard.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    environment["MEASUREMENT_GUARD_LOG"] = str(guard_log)
    environment["NY_MEASURE_SAFE_GPU_WRAPPED"] = "1"

    result = subprocess.run(
        ["/bin/bash", str(SCORECARD_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        preexec_fn=_lower_child_vmem_below_attestation,
    )

    assert result.returncode == 2
    assert "without complete containment attestation" in result.stderr
    assert not guard_log.exists(), "forged marker must not invoke or bypass the guard"


def external_scorecard_rejects_loader_injection_before_any_guard_child(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    guard_log = tmp_path / "guard-called"
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        '#!/bin/bash\nprintf called > "$MEASUREMENT_GUARD_LOG"\n',
        encoding="utf-8",
    )
    guard.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("BASH_FUNC_", "LD_"))
        and key not in {"BASH_ENV", "ENV", "BASHOPTS", "SHELLOPTS"}
    }
    environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    environment["MEASUREMENT_GUARD_LOG"] = str(guard_log)
    environment["LD_PRELOAD"] = ""

    result = subprocess.run(
        ["/bin/bash", str(SCORECARD_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "dynamic-loader injection" in result.stderr
    assert not guard_log.exists()


@pytest.mark.parametrize("injection", ["bash_env", "env", "exported_functions"])
def external_scorecard_rejects_shell_function_attestation_spoofing(
    tmp_path: Path,
    injection: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    guard_log = tmp_path / "guard-called"
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        '#!/bin/bash\nprintf called > "$MEASUREMENT_GUARD_LOG"\n',
        encoding="utf-8",
    )
    guard.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BASH_FUNC_")
        and key not in {"BASH_ENV", "ENV", "BASHOPTS", "SHELLOPTS"}
    }
    environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    environment["MEASUREMENT_GUARD_LOG"] = str(guard_log)
    if injection == "bash_env":
        bash_env = tmp_path / "bash-env"
        bash_env.write_text(
            "ulimit() { printf '167772160\\n'; }\n"
            "grep() { return 0; }\n"
            "cat() { printf '0::/ny-build.slice/forged.scope\\n'; }\n",
            encoding="utf-8",
        )
        environment["BASH_ENV"] = str(bash_env)
    elif injection == "env":
        environment["ENV"] = str(tmp_path / "shell-env")
    else:
        environment["BASH_FUNC_ulimit%%"] = "() {  printf '167772160\\n'\n}"
        environment["BASH_FUNC_grep%%"] = "() {  return 0\n}"
        environment["BASH_FUNC_cat%%"] = (
            "() {  printf '0::/ny-build.slice/forged.scope\\n'\n}"
        )
        environment["BASH_FUNC_builtin%%"] = "() {  return 0\n}"
        environment["BASH_FUNC_command%%"] = "() {  return 0\n}"
        environment["BASH_FUNC_ny-safe-gpu-run%%"] = "() {  return 0\n}"

    result = subprocess.run(
        ["/bin/bash", str(SCORECARD_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "rejects BASH_ENV" in result.stderr
    assert not guard_log.exists(), "shell injection must fail before invoking the guard"


@pytest.mark.parametrize(
    "bash_env_payload",
    [
        "unset BASH_ENV",
        (
            "readonly initial_shell_environment shell_environment_clean "
            "shell_environment_status; unset BASH_ENV"
        ),
        (
            "ulimit() { printf '167772160\\n'; }; "
            "grep() { return 0; }; "
            "cat() { printf '0::/ny-build.slice/forged.scope\\n'; }; "
            "builtin() { :; }; "
            "command() { :; }; "
            "ny-safe-gpu-run() { :; }; "
            "unset BASH_ENV"
        ),
    ],
)
def external_scorecard_rejects_self_erasing_bash_env_from_initial_snapshot(
    tmp_path: Path,
    bash_env_payload: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    guard_log = tmp_path / "guard-called"
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        '#!/bin/bash\nprintf called > "$MEASUREMENT_GUARD_LOG"\n',
        encoding="utf-8",
    )
    guard.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BASH_FUNC_")
        and key not in {"BASH_ENV", "ENV", "BASHOPTS", "SHELLOPTS"}
    }
    environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    environment["MEASUREMENT_GUARD_LOG"] = str(guard_log)

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'BASH_ENV=/dev/fd/3 /bin/bash "$1" 3<<<"$2"',
            "self-erasing-bash-env",
            str(SCORECARD_SCRIPT),
            bash_env_payload,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "rejects BASH_ENV" in result.stderr
    assert not guard_log.exists(), "initial environment rejection must precede guard"


def _fixture_repo(tmp_path: Path, *, expected_cpus: int = 10) -> Path:
    repo = tmp_path / "ny"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / ".gitignore").write_text(
        "/target/\n/reports/\n/benchmarks/\n__pycache__/\n", encoding="utf-8"
    )
    (repo / "rust-toolchain.toml").write_text(
        '[toolchain]\nchannel = "1.95.0"\ncomponents = ["rustfmt", "clippy"]\n',
        encoding="utf-8",
    )
    (repo / "Cargo.lock").write_text(
        "[[package]]\n"
        'name = "ay-test"\n'
        'version = "0.1.0"\n'
        f'source = "git+https://github.com/alabsystems/ay.git?rev={AY_REV}'
        f'#{AY_REV}"\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    for name in (
        "archive_vnncomp_sat_result.py",
        "measure_ny_scorecard.sh",
        "ny_measurement_provenance.py",
        "seal_ny_measurement_inputs.py",
    ):
        shutil.copyfile(REPO_ROOT / "scripts" / name, scripts / name)
    vnncomp_scripts = repo / "vnncomp_scripts"
    vnncomp_scripts.mkdir()
    receipt_helper = vnncomp_scripts / "submission_binary_receipt.sh"
    shutil.copyfile(
        REPO_ROOT / "vnncomp_scripts" / "submission_binary_receipt.sh",
        receipt_helper,
    )
    # The synthetic solver uses explicit test-only controls. Review them only
    # in this private committed fixture so the production allowlist remains
    # limited to real measurement/runtime inputs while env-i still reaches the
    # fake solver's fault-injection seams.
    fixture_provenance = scripts / "ny_measurement_provenance.py"
    fixture_source = fixture_provenance.read_text(encoding="utf-8")
    fixture_allowlist_anchor = '        "AY_LRA_WARM_SIMPLEX_STATE",\n'
    fixture_test_controls = "".join(
        f'        "{name}",\n'
        for name in (
            "MEASUREMENT_TEST_CUDA_SELFCHECK_FAIL",
            "MEASUREMENT_TEST_INVOCATION_LOG",
            "MEASUREMENT_TEST_MUTATE_SOLVER",
            "MEASUREMENT_TEST_RETARGET_CUDA_SOURCE",
            "MEASUREMENT_TEST_RETARGET_CUDA_TARGET",
            "MEASUREMENT_TEST_TAMPER_SEALED_CUDA",
        )
    )
    assert fixture_allowlist_anchor in fixture_source
    fixture_provenance.write_text(
        fixture_source.replace(
            fixture_allowlist_anchor,
            fixture_test_controls + fixture_allowlist_anchor,
            1,
        ),
        encoding="utf-8",
    )
    _install_fake_containment_snapshot(
        tmp_path,
        scripts,
        expected_cpus=expected_cpus,
    )

    fake_cuda_dir = repo / ".test-cuda-runtime"
    fake_cuda_dir.mkdir()
    for role, name in (
        ("driver", "libcuda.so.1"),
        ("cublas", "libcublas.so.13"),
        ("cublas_lt", "libcublasLt.so.13"),
    ):
        path = fake_cuda_dir / name
        path.write_bytes(f"synthetic {role} runtime\n".encode())
    # Exercise the numeric cudarc candidate spelling in the Bash source-path
    # gate as well as in the sealed alias namespace.
    shutil.copyfile(
        fake_cuda_dir / "libcuda.so.1",
        fake_cuda_dir / "libcuda64_132_0.so",
    )

    binary = repo / "target/release/ny"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        """#!/usr/bin/python3
import hashlib
import json
import os
import sys
from pathlib import Path

CANDIDATES = {
    "driver": [
        "libcuda.so",
        "libcuda64.so",
        "libcuda64_132_0.so",
        "libcuda.so.1",
        "libcuda.so.1",
    ],
    "cublas": [
        "libcublas.so",
        "libcublas64.so",
        "libcublas64_132_0.so",
        "libcublas.so.13",
    ],
    "cublas_lt": [
        "libcublasLt.so",
        "libcublasLt64.so",
        "libcublasLt64_132_0.so",
        "libcublasLt.so.13",
    ],
    "nvrtc": ["libnvrtc.so", "libnvrtc64_132_0.so", "libnvrtc.so.13"],
}


def fingerprint(path):
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def runtime_report():
    raw = os.environ.get("LD_LIBRARY_PATH", "")
    if not raw or os.pathsep in raw:
        raise SystemExit("fixture requires one exact CUDA loader directory")
    root = Path(raw)
    objects = []
    for role in ("driver", "cublas", "cublas_lt"):
        selected = next(
            (root / name for name in CANDIDATES[role] if (root / name).is_file()),
            None,
        )
        if selected is None:
            raise SystemExit(f"fixture CUDA role is unavailable: {role}")
        resolved = selected.resolve(strict=True)
        value = fingerprint(resolved)
        objects.append(
            {
                "role": role,
                "provider_symbol": f"fixture_{role}_symbol",
                "mapped_path": str(resolved),
                "resolved_path": str(resolved),
                "mapped_device_major": os.major(value["device"]),
                "mapped_device_minor": os.minor(value["device"]),
                "mapped_inode": value["inode"],
                "size_bytes": value["size_bytes"],
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                "fingerprint": value,
            }
        )
    print(
        json.dumps(
            {
                "schema": "ny_cuda_runtime_info_v3",
                "device_name": "synthetic CUDA device",
                "pageable_host_ptr": False,
                "pageable_memory_access": True,
                "pageable_access_uses_host_page_tables": False,
                "integrated_device": False,
                "ordinary_gemm_transport": "explicit-device-copy",
                "ordinary_gemm_transport_policy": "auto",
                "ordinary_gemm_transport_reason": "discrete-device",
                "explicit_device_copy": True,
                "discrete_mode": True,
                "deadline_f64_transport": "explicit-device-copy",
                "candidates": CANDIDATES,
                "objects": objects,
                "nvrtc_status": "not_loaded_feature_disabled",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


argument = sys.argv[1] if len(sys.argv) > 1 else ""
invocation_log = os.environ.get("MEASUREMENT_TEST_INVOCATION_LOG")
if invocation_log:
    with Path(invocation_log).open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "ambient_solver_marker": os.environ.get(
                        "AMBIENT_SOLVER_MARKER"
                    ),
                    "argument": argument,
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ),
                    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
                    "ny_no_cuda": os.environ.get("NY_NO_CUDA"),
                    "path": os.environ.get("PATH"),
                    "rustup_home": os.environ.get("RUSTUP_HOME"),
                },
                sort_keys=True,
            )
            + "\\n"
        )
if argument == "--version":
    print("ny 0.1.0-test")
elif argument == "--build-info":
    print("ny 0.1.0-test cuda=on mip=on")
elif argument == "--cuda-runtime-info":
    runtime_report()
elif argument == "--cuda-selfcheck":
    print("self-check wording is intentionally not an API", file=sys.stderr)
    if os.environ.get("MEASUREMENT_TEST_CUDA_SELFCHECK_FAIL") == "1":
        raise SystemExit(42)
    source_link = os.environ.get("MEASUREMENT_TEST_RETARGET_CUDA_SOURCE")
    source_target = os.environ.get("MEASUREMENT_TEST_RETARGET_CUDA_TARGET")
    if source_link and source_target:
        link = Path(source_link)
        link.unlink()
        link.symlink_to(source_target, target_is_directory=True)
    if os.environ.get("MEASUREMENT_TEST_TAMPER_SEALED_CUDA") == "1":
        runtime = Path(os.environ["LD_LIBRARY_PATH"])
        runtime.chmod(0o755)
        driver = runtime / "libcuda.so"
        driver.chmod(0o644)
        driver.write_bytes(b"tampered sealed driver\\n")
else:
    print(f"solver combined log for {sys.argv[5]}")
    print(f"solver ld_library_path:{os.environ.get('LD_LIBRARY_PATH', '')}")
    print(f"solver ny_no_cuda:{os.environ.get('NY_NO_CUDA', '')}")
    print("solver argv:" + "".join(f" <{value}>" for value in sys.argv[1:]))
    result_file = Path(sys.argv[6])
    if sys.argv[5].endswith("/violated.vnnlib"):
        verdict = "sat"
        result_file.write_text("sat\\n((X_0 0.25))\\n", encoding="utf-8")
    else:
        verdict = "unsat"
        result_file.write_text("unsat\\n", encoding="utf-8")
    Path(f"{result_file}.flight.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend_kind": "cuda",
                "backend_summary": "synthetic fixture",
                "host": {},
                "category": sys.argv[3],
                "budget_secs": int(sys.argv[7]),
                "ambient_env": {
                    name: value
                    for name, value in os.environ.items()
                    if name.startswith("NY_") or name == "OMP_NUM_THREADS"
                },
                "events": [
                    {
                        "method": "run_complete",
                        "status": "complete",
                        "reason": verdict,
                        "at_secs": 0.01,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if os.environ.get("MEASUREMENT_TEST_MUTATE_SOLVER") == "1":
        executable = Path(__file__)
        executable.chmod(0o755)
        with executable.open("a", encoding="utf-8") as output:
            output.write("\\n# measurement drift\\n")
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "fixture", cwd=repo)

    # Git records commit times with one-second resolution.  On a second
    # boundary, the synthetic binary can otherwise appear one second older
    # than the fixture commit even though it was created immediately before
    # that commit.  Model a completed post-commit build deterministically.
    fixture_epoch = int(
        subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    os.utime(binary, (fixture_epoch + 10, fixture_epoch + 10))

    # Automatic scorecard selection follows the organizer run_instance path:
    # all ordinary fixtures therefore need an authenticated binary receipt.
    # Use a tiny explicit rustc identity shim so the synthetic future toolchain
    # pin does not trigger a network install during tests.
    fake_rustc = tmp_path / "fixture-rustc"
    fake_rustc.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'rustc 1.95.0-test'\n", encoding="utf-8"
    )
    fake_rustc.chmod(0o755)
    receipt_environment = os.environ.copy()
    receipt_environment["RUSTC"] = str(fake_rustc)
    receipt = subprocess.run(
        [
            "bash",
            str(receipt_helper),
            "create-local",
            str(binary),
            str(repo),
            "mip,cuda",
        ],
        cwd=repo,
        env=receipt_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert receipt.returncode == 0, receipt.stderr

    benchmark = repo / "benchmarks/vnncomp2026"
    benchmark.mkdir(parents=True)
    _init_git_repo(benchmark)
    instance_dir = benchmark / "benchmarks/demo/2.0"
    instance_dir.mkdir(parents=True)
    (instance_dir / "instances.csv").write_text(
        "shared.onnx,violated.vnnlib,1\nshared.onnx,holds.vnnlib,1\n",
        encoding="utf-8",
    )
    (instance_dir / "shared.onnx").write_bytes(b"shared model")
    (instance_dir / "violated.vnnlib").write_text("; violated\n", encoding="utf-8")
    (instance_dir / "holds.vnnlib").write_text("; holds\n", encoding="utf-8")
    _run("git", "add", ".", cwd=benchmark)
    _run("git", "commit", "-qm", "fixture", cwd=benchmark)
    _run(
        "git",
        "remote",
        "add",
        "origin",
        "https://example.invalid/vnncomp-benchmarks.git",
        cwd=benchmark,
    )
    return repo


def _measurement_environment(
    repo: Path,
    tmp_path: Path,
    **overrides: str,
) -> dict[str, str]:
    fake_bin = _install_fake_gpu_guard(tmp_path)
    fake_ay = fake_bin / "ay"
    fake_ay.write_text(
        "#!/bin/sh\n"
        f"""printf '%s' "${{AMBIENT_SOLVER_MARKER:-}}" > "{tmp_path / 'ay-version-environment'}"\n"""
        "printf '%s\\n' 'ay 0.1.0-test' "
        f"'build.commit={AY_REV}'\n",
        encoding="utf-8",
    )
    fake_ay.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NY_", "BASH_FUNC_"))
        and not key.startswith("LD_")
        and not key.startswith(
            (
                "__EGL_",
                "__GL_",
                "__GLX_",
                "__NV_",
                "__VK_",
                "ACCELERATE_",
                "ACO_",
                "AMD_",
                "ANV_",
                "BLIS_",
                "CANDLE_",
                "CARGO_",
                "CUBLAS_",
                "CUBLASLT_",
                "CUDA_",
                "CUDNN_",
                "DISABLE_LAYER_",
                "DRI_",
                "EGL_",
                "ENABLE_LAYER_",
                "GALLIUM_",
                "GBM_",
                "GOMP_",
                "GOTO_",
                "INTEL_",
                "KMP_",
                "LC_",
                "LIBGL_",
                "LP_",
                "LVP_",
                "MALLOC_",
                "MATMUL_",
                "MESA_",
                "MKL_",
                "Malloc",
                "NOUVEAU_",
                "NVBLAS_",
                "NVIDIA_",
                "NVPRESENT_",
                "NVRTC_",
                "NVVM_",
                "OMP_",
                "OPENBLAS_",
                "ORT_",
                "PYTHON",
                "RADV_",
                "RAYON_",
                "RUST_",
                "RUSTUP_",
                "VECLIB_",
                "VK_",
                "VULKAN_",
                "WGPU_",
                "XDG_",
                "ZINK_",
            )
        )
        and not key.upper().startswith("MIMALLOC_")
        and key
        not in {
            "BASH_ENV",
            "DISABLE_LAYER_NV_OPTIMUS_1",
            "DRI_PRIME",
            "DYLD_FORCE_FLAT_NAMESPACE",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "ENV",
            "BASHOPTS",
            "DISPLAY",
            "GCONV_PATH",
            "GLIBC_TUNABLES",
            "LANGUAGE",
            "LOCPATH",
            "MALLOC_CONF",
            "MESA_VK_DEVICE_SELECT",
            "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE",
            "NODEVICE_SELECT",
            "SHELLOPTS",
            "TEMP",
            "TEMPDIR",
            "TMP",
            "WAYLAND_DISPLAY",
            "XAUTHORITY",
        }
    }
    environment.update(
        {
            "GITHUB_TOKEN": "must-not-be-recorded",
            "MEASUREMENT_TEST_INVOCATION_LOG": str(
                tmp_path / "solver-invocations.jsonl"
            ),
            # Keep AY provenance hermetic. A developer's installed AY must not
            # contaminate the synthetic fixture's revision-pinned executable.
            "NY_AY": str(fake_ay),
            "NY_BUILD_FEATURES": "mip,cuda",
            "NY_MEASURE_CAP": "2",
            "NY_MEASURE_CATS": "demo",
            "NY_MEASURE_RUN_ID": RUN_ID,
            "NY_MARGIN_ROW_CLASSWISE": "1",
            "NY_ROOT": str(repo),
            "NY_SCRATCH": str(tmp_path / "scratch"),
            "LD_LIBRARY_PATH": str(repo / ".test-cuda-runtime"),
            "PATH": f"{fake_bin}:{environment.get('PATH', '/usr/bin:/bin')}",
        }
    )
    environment.update(overrides)
    return environment


def external_sweep_rejects_unreviewed_solver_runtime_environment(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    invocation_log = tmp_path / "solver-invocations.jsonl"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            WGPU_BACKEND="vulkan",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "unreviewed solver-runtime environment controls" in result.stderr
    assert not invocation_log.exists()
    assert not (repo / "reports" / "measured-runs" / RUN_ID).exists()


def external_sweep_ignores_pythonpath_sitecustomize_before_rejecting_it(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    injected = tmp_path / "python-injected"
    injected.mkdir()
    marker = tmp_path / "sitecustomize-imported"
    (injected / "sitecustomize.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['SITECUSTOMIZE_MARKER']).write_text('imported')\n",
        encoding="utf-8",
    )
    environment = _measurement_environment(repo, tmp_path)
    environment["PYTHONPATH"] = str(injected)
    environment["SITECUSTOMIZE_MARKER"] = str(marker)

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "unreviewed solver-runtime environment controls" in result.stderr
    assert not marker.exists(), "scorecard Python must ignore PYTHONPATH before capture"
    assert not (tmp_path / "solver-invocations.jsonl").exists()
    assert not (repo / "reports" / "measured-runs" / RUN_ID).exists()


def external_scorecard_ignores_ignored_sibling_python_bytecode(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    marker = tmp_path / "ignored-pyc-executed"
    source = repo / "scripts/archive_vnncomp_sat_result.py"
    malicious_source = tmp_path / "archive_vnncomp_sat_result.py"
    malicious_source.write_text(
        source.read_text(encoding="utf-8").replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\n"
            "from pathlib import Path as _InjectedPath\n"
            f"_InjectedPath({str(marker)!r}).write_text('executed')\n",
            1,
        ),
        encoding="utf-8",
    )
    compile_result = subprocess.run(
        [
            "/usr/bin/python3",
            "-E",
            "-s",
            "-S",
            "-c",
            (
                "import importlib.util, pathlib, py_compile, sys; "
                "source, malicious = map(pathlib.Path, sys.argv[1:]); "
                "cache = pathlib.Path(importlib.util.cache_from_source(str(source))); "
                "cache.parent.mkdir(parents=True, exist_ok=True); "
                "py_compile.compile(str(malicious), cfile=str(cache), "
                "dfile=str(source), doraise=True, "
                "invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)"
            ),
            str(source),
            str(malicious_source),
        ],
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert not marker.exists(), "ignored sibling bytecode must never be imported"


def external_solver_and_identity_probes_receive_only_manifest_environment(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            AMBIENT_SOLVER_MARKER="must-not-reach-ny",
            CARGO_BUILD_JOBS="3",
            CUDA_VISIBLE_DEVICES="fixture-gpu",
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            RUSTFLAGS="-C debuginfo=0",
            RUST_TEST_THREADS="7",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    invocations = [
        json.loads(line)
        for line in (tmp_path / "solver-invocations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert invocations
    assert all(item["ambient_solver_marker"] is None for item in invocations)
    assert {
        item["cuda_visible_devices"] for item in invocations
    } == {"fixture-gpu"}
    assert {item["path"] for item in invocations} == {"/usr/bin:/bin"}
    assert {item["rustup_home"] for item in invocations} == {None}
    assert (tmp_path / "ay-version-environment").read_text(encoding="utf-8") == ""
    start_path = (
        repo / f"reports/measured-runs/{RUN_ID}/artifacts/runs/{RUN_ID}/start.json"
    )
    start = json.loads(start_path.read_text(encoding="utf-8"))
    solver_environment = start["measurement"]["solver_environment"]
    assert solver_environment["mode"] == "env-i-reviewed-record-v1"
    assert "AMBIENT_SOLVER_MARKER" not in solver_environment["values"]
    assert solver_environment["values"]["CUDA_VISIBLE_DEVICES"] == "fixture-gpu"
    assert solver_environment["values"]["PATH"] == "/usr/bin:/bin"
    for helper_only in (
        "CARGO_BUILD_JOBS",
        "NY_MEASURE_RUSTUP_BIN",
        "RUSTFLAGS",
        "RUSTUP_HOME",
        "RUST_TEST_THREADS",
    ):
        assert helper_only in start["environment"]["values"]
        assert helper_only not in solver_environment["values"]
        assert helper_only in start["measurement"]["solver_environment_unsets"]


def external_scorecard_ignores_path_prepended_env_and_timeout_shims(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    environment = _measurement_environment(
        repo,
        tmp_path,
        NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
    )
    fake_bin = Path(environment["PATH"].split(os.pathsep, maxsplit=1)[0])
    bash_marker = tmp_path / "fake-bash-called"
    env_marker = tmp_path / "fake-env-called"
    timeout_marker = tmp_path / "fake-timeout-called"
    (fake_bin / "bash").write_text(
        "#!/bin/sh\n"
        f"printf called > {bash_marker}\n"
        'exec /bin/bash "$@"\n',
        encoding="utf-8",
    )
    (fake_bin / "env").write_text(
        "#!/bin/sh\n"
        f"printf called > {env_marker}\n"
        "export AMBIENT_SOLVER_MARKER=injected-by-fake-env\n"
        'exec /usr/bin/env "$@"\n',
        encoding="utf-8",
    )
    (fake_bin / "timeout").write_text(
        "#!/bin/sh\n"
        f"printf called > {timeout_marker}\n"
        "export AMBIENT_SOLVER_MARKER=injected-by-fake-timeout\n"
        'exec /usr/bin/timeout "$@"\n',
        encoding="utf-8",
    )
    for executable in ("bash", "env", "timeout"):
        (fake_bin / executable).chmod(0o755)

    result = subprocess.run(
        ["/bin/bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert not bash_marker.exists()
    assert not env_marker.exists()
    assert not timeout_marker.exists()
    invocations = [
        json.loads(line)
        for line in (tmp_path / "solver-invocations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert invocations
    assert all(item["ambient_solver_marker"] is None for item in invocations)


def external_sweep_refuses_failed_sealed_cuda_selfcheck(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            MEASUREMENT_TEST_CUDA_SELFCHECK_FAIL="1",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "failed CUDA runtime/device qualification" in result.stderr
    assert "--cuda-selfcheck must exit successfully" in result.stderr
    assert not (repo / f"reports/measured-runs/{RUN_ID}/demo.csv").exists()


def external_explicit_noncuda_debug_measurement_skips_cuda_selfcheck(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            MEASUREMENT_TEST_CUDA_SELFCHECK_FAIL="1",
            NY_ALLOW_NONCUDA_MEASURE="1",
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    start_path = (
        repo / f"reports/measured-runs/{RUN_ID}/artifacts/runs/{RUN_ID}/start.json"
    )
    start = json.loads(start_path.read_text(encoding="utf-8"))
    assert start["dependencies"]["cuda_runtime"]["status"] == "not_required"
    assert start["environment"]["values"]["NY_NO_CUDA"] == "1"
    assert start["measurement"]["solver_environment_overrides"]["NY_NO_CUDA"] == "1"
    metadata_path = next(
        (repo / f"reports/measured-runs/{RUN_ID}/artifacts/demo").glob(
            f"*/{RUN_ID}.json"
        )
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    solver_log = (
        repo
        / f"reports/measured-runs/{RUN_ID}/artifacts"
        / metadata["solver_log"]["artifact"]
    ).read_text(encoding="utf-8")
    assert "solver ld_library_path:\n" in solver_log
    assert "solver ny_no_cuda:1\n" in solver_log
    invocations = [
        json.loads(line)
        for line in (tmp_path / "solver-invocations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    cpu_rows = [item for item in invocations if item["argument"] == "vnncomp"]
    assert cpu_rows
    assert all(item["ld_library_path"] is None for item in cpu_rows)
    assert all(item["ny_no_cuda"] == "1" for item in cpu_rows)


def external_source_cuda_symlink_retarget_after_start_cannot_change_measured_runtime(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    source_v1 = tmp_path / "cuda-source-v1"
    source_v2 = tmp_path / "cuda-source-v2"
    shutil.copytree(repo / ".test-cuda-runtime", source_v1)
    shutil.copytree(repo / ".test-cuda-runtime", source_v2)
    for path in source_v2.iterdir():
        path.write_bytes(b"replacement source must never execute\n")
    active = tmp_path / "cuda-active"
    active.symlink_to(source_v1, target_is_directory=True)
    isolated = repo / "reports/source-retarget"
    run_id = "20260718T121000Z-source-retarget"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            LD_LIBRARY_PATH=str(active),
            MEASUREMENT_TEST_RETARGET_CUDA_SOURCE=str(active),
            MEASUREMENT_TEST_RETARGET_CUDA_TARGET=str(source_v2),
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert active.resolve() == source_v2.resolve()
    start = json.loads(
        (isolated / f"artifacts/runs/{run_id}/start.json").read_text(encoding="utf-8")
    )
    runtime = start["dependencies"]["cuda_runtime"]
    assert all(
        Path(str(item["resolved_path"])).parent == source_v1
        for item in runtime["source_capture"]["objects"]
    )
    sealed_dir = Path(runtime["sealed_execution"]["path"])
    assert all(
        path.read_bytes().startswith(b"synthetic ")
        for path in sealed_dir.iterdir()
    )
    metadata_path = next((isolated / "artifacts/demo").glob(f"*/{run_id}.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    solver_log = (
        isolated / "artifacts" / metadata["solver_log"]["artifact"]
    ).read_text(encoding="utf-8")
    assert f"solver ld_library_path:{sealed_dir}" in solver_log


def external_sealed_cuda_runtime_tamper_is_rejected_before_first_row(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    isolated = repo / "reports/sealed-cuda-tamper"
    run_id = "20260718T121100Z-sealed-cuda-tamper"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            MEASUREMENT_TEST_TAMPER_SEALED_CUDA="1",
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "sealed CUDA runtime verification failed" in result.stderr
    csv_path = isolated / "demo.csv"
    assert not csv_path.exists() or csv_path.read_bytes() == b""
    completion = json.loads(
        (isolated / f"artifacts/runs/{run_id}/completion.json").read_text(
            encoding="utf-8"
        )
    )
    codes = {item["code"] for item in completion["integrity"]["violations"]}
    assert completion["completed_successfully"] is False
    assert "cuda_runtime_unavailable" in codes


def external_sweep_rejects_ld_library_path_with_non_cuda_override(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    unsafe = tmp_path / "unsafe-loader"
    shutil.copytree(repo / ".test-cuda-runtime", unsafe)
    # Use a non-DT_NEEDED override name so the fixture reaches the script's
    # pre-child gate; a fake libc would (correctly) be consumed before Bash can
    # execute any script-level check.
    (unsafe / "libfixture-injection.so").write_bytes(b"loader injection\n")
    isolated = repo / "reports/unsafe-loader"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            LD_LIBRARY_PATH=str(unsafe),
            NY_MEASURE_OUTPUT_DIR=str(isolated),
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "unsafe LD_LIBRARY_PATH entry" in result.stderr
    assert not isolated.exists()


def external_sweep_binds_rows_and_sat_artifact_to_completed_run(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(repo, tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    output_root = repo / f"reports/measured-runs/{RUN_ID}"
    artifact_root = output_root / "artifacts"
    csv_path = output_root / "demo.csv"
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert len(rows) == 2
    _assert_measurement_row(
        rows[0],
        ["demo", "shared.onnx", "violated.vnnlib", "0", "sat"],
        RUN_ID,
    )
    _assert_measurement_row(
        rows[1],
        ["demo", "shared.onnx", "holds.vnnlib", "0", "unsat"],
        RUN_ID,
    )

    run_dir = artifact_root / f"runs/{RUN_ID}"
    start_path = run_dir / "start.json"
    completion_path = run_dir / "completion.json"
    start_bytes = start_path.read_bytes()
    start = json.loads(start_bytes)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert start["measurement"]["output_dir"] == str(output_root.resolve())
    assert start["measurement"]["scratch_dir"] == str((tmp_path / "scratch").resolve())
    assert start["environment"]["values"]["NY_SCRATCH"] == str(tmp_path / "scratch")
    assert not (repo / "reports/measured/demo.csv").exists()
    assert start["solver_binary"]["declared_build_features"] == ["mip", "cuda"]
    sealed_solver = start["solver_binary"]["sealed_execution"]
    assert sealed_solver["path"] != start["solver_binary"]["path"]
    assert (
        Path(sealed_solver["path"]).read_bytes()
        == Path(start["solver_binary"]["path"]).read_bytes()
    )
    assert start["measurement"]["csv_columns"][-1] == "run_id"
    assert start["measurement"]["config_inputs"] is None
    assert "--configs-dir" not in start["measurement"]["solver_command_template"]
    expected_root_gemm = (
        "faer"
        if platform.system() == "Linux"
        and platform.machine().lower() in {"aarch64", "arm64"}
        else "ndarray"
    )
    assert start["environment"]["values"]["NY_ROOT_GEMM"] == expected_root_gemm
    assert start["environment"]["values"]["NY_MARGIN_ROW_CLASSWISE"] == "1"
    cuda_runtime = start["dependencies"]["cuda_runtime"]
    assert cuda_runtime["schema"] == "ny_measurement_cuda_runtime_v1"
    assert cuda_runtime["status"] == "captured"
    sealed_cuda = cuda_runtime["sealed_execution"]
    sealed_cuda_path = Path(sealed_cuda["path"])
    assert sealed_cuda["schema"] == "ny_measurement_sealed_cuda_runtime_v1"
    assert sealed_cuda_path.parent.name == "cuda-runtime"
    expected_cuda_names = {
        "libcuda.so",
        "libcuda.so.1",
        "libcuda64.so",
        "libcuda64_132_0.so",
        "libcublas.so",
        "libcublas.so.13",
        "libcublas64.so",
        "libcublas64_132_0.so",
        "libcublasLt.so",
        "libcublasLt.so.13",
        "libcublasLt64.so",
        "libcublasLt64_132_0.so",
    }
    assert {entry["name"] for entry in sealed_cuda["entries"]} == expected_cuda_names
    assert {path.name for path in sealed_cuda_path.iterdir()} == expected_cuda_names
    assert not any(path.is_symlink() for path in sealed_cuda_path.iterdir())
    entries_by_role: dict[str, list[dict[str, object]]] = {}
    for entry in sealed_cuda["entries"]:
        entries_by_role.setdefault(str(entry["role"]), []).append(entry)
        entry_path = Path(str(entry["path"]))
        assert entry_path.parent == sealed_cuda_path
        assert entry_path.stat().st_mode & 0o222 == 0
    for entries in entries_by_role.values():
        assert len({Path(str(entry["path"])).stat().st_ino for entry in entries}) == 1
    assert all(
        Path(str(item["resolved_path"])).parent == sealed_cuda_path
        for item in cuda_runtime["objects"]
    )
    assert (
        start["measurement"]["solver_environment_overrides"]["LD_LIBRARY_PATH"]
        == str(sealed_cuda_path)
    )
    invocations = [
        json.loads(line)
        for line in (tmp_path / "solver-invocations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    runtime_probes = [
        item for item in invocations if item["argument"] == "--cuda-runtime-info"
    ]
    assert len(runtime_probes) >= 6
    assert {
        item["ld_library_path"] for item in runtime_probes[:2]
    } == {str((repo / ".test-cuda-runtime").resolve())}
    assert {
        item["ld_library_path"] for item in runtime_probes[2:]
    } == {str(sealed_cuda_path)}
    for argument in ("--build-info", "--cuda-selfcheck", "vnncomp"):
        matching = [item for item in invocations if item["argument"] == argument]
        assert matching
        assert {
            item["ld_library_path"] for item in matching
        } == {str(sealed_cuda_path)}
    assert start["environment"]["values"]["NY_MEASURE_EXPECTED_CPUS"] == "10"
    assert start["environment"]["typed_values"]["NY_MEASURE_EXPECTED_CPUS"] == {
        "type": "enum",
        "value": "10",
        "allowed_values": ["10", "20"],
    }
    containment = start["host"]["containment"]
    assert containment["schema"] == "ny_measurement_containment_v1"
    assert containment["containment_profile"] == "gb10-80g"
    assert containment["membership"].endswith(
        f"/ny-safe-gpu-{os.getuid()}-1234-5678.service"
    )
    assert containment["policy_cgroup"] == containment["current_cgroup"]
    assert containment["effective"]["memory.high"]["value_bytes"] == 64 * 1024**3
    assert containment["effective"]["memory.max"]["value_bytes"] == 80 * 1024**3
    assert containment["effective"]["memory.swap.max"]["value_bytes"] == 8 * 1024**3
    assert containment["effective"]["pids.max"]["value"] == 4096
    assert containment["effective"]["cpu.max"]["equivalent_cpus"] == 10
    assert containment["policy"]["cpu.max"] == {
        "raw": "1000000 100000",
        "quota_us": 1_000_000,
        "period_us": 100_000,
        "equivalent_cpus": 10,
    }
    assert containment["rlimit_as"] == {
        "soft_bytes": 160 * 1024**3,
        "hard_bytes": 160 * 1024**3,
    }
    assert b"must-not-be-recorded" not in start_bytes
    assert completion["exit_status"] == 0
    assert completion["completed_successfully"] is True
    assert completion["integrity"]["status"] == "valid"
    assert completion["integrity"]["violations"] == []
    assert completion["integrity"]["checks"]["containment"]["status"] == "valid"
    assert completion["integrity"]["checks"]["cuda_runtime"]["status"] == "valid"
    run_evidence = completion["integrity"]["checks"]["run_evidence"]
    assert run_evidence["status"] == "valid"
    assert run_evidence["metadata_count"] == 2
    assert run_evidence["result_count"] == 2
    assert run_evidence["solver_log_count"] == 2
    assert run_evidence["preflight_count"] == 2
    assert run_evidence["csv_row_count"] == 2
    assert run_evidence["validated_record_count"] == 2
    assert len(run_evidence["records_sha256"]) == 64
    assert len(run_evidence["csv_evidence_sha256"]) == 64
    assert (
        completion["start_manifest_sha256"] == hashlib.sha256(start_bytes).hexdigest()
    )
    cache_path = run_dir / "input_hash_cache.json"
    assert completion["input_hash_cache"]["present"] is True
    assert completion["input_hash_cache"]["entry_count"] == 3
    assert (
        completion["input_hash_cache"]["sha256"]
        == hashlib.sha256(cache_path.read_bytes()).hexdigest()
    )

    metadata_paths = sorted((artifact_root / "demo").glob(f"*/{RUN_ID}.json"))
    assert len(metadata_paths) == 2
    metadata_by_verdict = {
        metadata["solver_verdict"]: (metadata_path, metadata)
        for metadata_path in metadata_paths
        for metadata in [json.loads(metadata_path.read_text(encoding="utf-8"))]
    }
    assert set(metadata_by_verdict) == {"sat", "unsat"}
    for metadata_path, metadata in metadata_by_verdict.values():
        assert metadata["start_manifest"] == f"runs/{RUN_ID}/start.json"
        assert (
            metadata["start_manifest_sha256"] == hashlib.sha256(start_bytes).hexdigest()
        )
        assert (
            metadata["raw_result_sha256"]
            == hashlib.sha256(
                metadata_path.with_suffix(".results").read_bytes()
            ).hexdigest()
        )
        solver_log_path = artifact_root / metadata["solver_log"]["artifact"]
        assert solver_log_path.read_bytes().startswith(b"solver combined log for ")
        assert (
            f"solver ld_library_path:{sealed_cuda_path}\n".encode()
            in solver_log_path.read_bytes()
        )
        for label in ("onnx", "vnnlib"):
            sealed_input = metadata["execution_inputs"][label]
            assert sealed_input["resolved_path"] != metadata[label]["resolved_path"]
            assert (
                Path(sealed_input["resolved_path"]).read_bytes()
                == Path(metadata[label]["resolved_path"]).read_bytes()
            )
            assert f" <{sealed_input['resolved_path']}>".encode() in (
                solver_log_path.read_bytes()
            )
        preflight = metadata["input_preflight"]
        assert (
            preflight["sha256"]
            == hashlib.sha256(
                (artifact_root / preflight["artifact"]).read_bytes()
            ).hexdigest()
        )
        assert b"--configs-dir" not in solver_log_path.read_bytes()
        assert metadata["config_inputs"] is None
        assert (
            metadata["solver_log"]["sha256"]
            == hashlib.sha256(solver_log_path.read_bytes()).hexdigest()
        )
        flight = metadata["flight_record"]
        assert flight["status"] == "captured"
        assert flight["record"]["category"] == metadata["category"]
        assert flight["record"]["budget_secs"] == metadata["timeout_seconds"]
        assert flight["record"]["ambient_env"]["OMP_NUM_THREADS"] == "1"
        assert flight["record"]["events"][-1] == {
            "at_secs": 0.01,
            "method": "run_complete",
            "reason": metadata["solver_verdict"],
            "status": "complete",
        }

    sat_path, sat_metadata = metadata_by_verdict["sat"]
    assert sat_path.with_suffix(".results").read_text(encoding="utf-8") == (
        "sat\n((X_0 0.25))\n"
    )
    assert sat_metadata["counterexample_validation"]["status"] == "not_checked"
    assert sat_metadata["onnx"]["sha256"] == hashlib.sha256(b"shared model").hexdigest()
    assert sat_metadata["onnx"]["hash_cache_hit"] is False
    unsat_path, unsat_metadata = metadata_by_verdict["unsat"]
    assert unsat_path.with_suffix(".results").read_bytes() == b"unsat\n"
    assert unsat_metadata["counterexample_validation"]["status"] == ("not_applicable")
    assert unsat_metadata["onnx"]["hash_cache_hit"] is True
    assert (
        unsat_metadata["vnnlib"]["sha256"] == hashlib.sha256(b"; holds\n").hexdigest()
    )


def external_explicit_twenty_cpu_lane_is_attested_and_provenanced(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path, expected_cpus=20)
    run_id = "20260718T121000Z-cpu20"
    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_EXPECTED_CPUS="20",
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    run_dir = repo / f"reports/measured-runs/{run_id}/artifacts/runs/{run_id}"
    start = json.loads((run_dir / "start.json").read_text(encoding="utf-8"))
    completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
    environment = start["environment"]
    assert environment["values"]["NY_MEASURE_EXPECTED_CPUS"] == "20"
    assert environment["typed_values"]["NY_MEASURE_EXPECTED_CPUS"] == {
        "type": "enum",
        "value": "20",
        "allowed_values": ["10", "20"],
    }
    containment = start["host"]["containment"]
    assert containment["policy"]["cpu.max"] == {
        "raw": "2000000 100000",
        "quota_us": 2_000_000,
        "period_us": 100_000,
        "equivalent_cpus": 20,
    }
    assert containment["effective"]["cpu.max"]["equivalent_cpus"] == 20
    assert completion["integrity"]["checks"]["containment"]["status"] == "valid"
    assert completion["completed_successfully"] is True


@pytest.mark.parametrize(
    ("policy_cpus", "selected_cpus"),
    [(10, "20"), (20, "10")],
)
def external_scorecard_rejects_selected_cpu_policy_mismatch(
    tmp_path: Path,
    policy_cpus: int,
    selected_cpus: str,
) -> None:
    repo = _fixture_repo(tmp_path, expected_cpus=policy_cpus)
    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_EXPECTED_CPUS=selected_cpus,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "ny-safe-gpu service cpu.max policy mismatch" in result.stderr
    assert not (repo / "reports").exists()


def external_default_scratch_is_unique_and_keyed_by_run_id(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    run_id = "20260718T121500Z-default-scratch"
    temp_root = tmp_path / "tmp"
    environment = _measurement_environment(
        repo,
        tmp_path,
        NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
        NY_MEASURE_RUN_ID=run_id,
        TMPDIR=str(temp_root),
    )
    environment.pop("NY_SCRATCH")

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    start_path = (
        repo / f"reports/measured-runs/{run_id}/artifacts/runs/{run_id}/start.json"
    )
    start = json.loads(start_path.read_text(encoding="utf-8"))
    expected_scratch = temp_root / "ny_measure_scratch" / run_id
    measurement = start["measurement"]
    assert measurement["scratch_dir"] == str(expected_scratch.resolve())
    assert measurement["result_file"] == str(
        (expected_scratch / "ny_vnncomp_result.txt").resolve()
    )
    assert measurement["solver_log_file"] == str(
        (expected_scratch / "ny_vnncomp_output.log").resolve()
    )
    assert "NY_SCRATCH" not in start["environment"]["values"]
    assert start["environment"]["values"]["TMPDIR"] == str(
        (expected_scratch / "tmp").resolve()
    )
    assert start["environment"]["values"]["HOME"] == str(
        (expected_scratch / "home").resolve()
    )


def external_scorecard_rejects_relative_scratch_before_solver(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_SCRATCH="relative-scratch",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "NY_SCRATCH must resolve from an absolute path" in result.stderr
    assert not (tmp_path / "solver-invocations.jsonl").exists()


@pytest.mark.parametrize("child", ["home", "tmp"])
@pytest.mark.parametrize("kind", ["directory", "symlink"])
def external_scorecard_rejects_preexisting_isolation_paths(
    tmp_path: Path,
    child: str,
    kind: str,
) -> None:
    repo = _fixture_repo(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    isolated = scratch / child
    if kind == "directory":
        isolated.mkdir()
        (isolated / "unrecorded-config").write_text("poison", encoding="utf-8")
    else:
        target = tmp_path / f"{child}-target"
        target.mkdir()
        isolated.symlink_to(target, target_is_directory=True)

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(repo, tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "isolated HOME/TMPDIR must not already exist" in result.stderr
    assert not (tmp_path / "solver-invocations.jsonl").exists()


def external_scorecard_rejects_symlinked_scratch_root(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    target = tmp_path / "scratch-target"
    target.mkdir()
    scratch = tmp_path / "scratch"
    scratch.symlink_to(target, target_is_directory=True)

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(repo, tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "scratch path must be a real directory" in result.stderr
    assert not (tmp_path / "solver-invocations.jsonl").exists()


def external_explicit_solver_binary_is_allowed_captured_and_sealed(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    external_binary = tmp_path / "private-target/release/ny"
    external_binary.parent.mkdir(parents=True)
    shutil.copy2(repo / "target/release/ny", external_binary)
    external_binary.write_bytes(external_binary.read_bytes() + b"\n# external build\n")
    external_binary.chmod(0o755)
    isolated = repo / "reports/external-solver"
    run_id = "20260718T123000Z-external-solver"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_BIN=str(external_binary),
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "explicit NY_MEASURE_BIN override" in result.stderr
    start = json.loads(
        (isolated / f"artifacts/runs/{run_id}/start.json").read_text(encoding="utf-8")
    )
    assert start["environment"]["values"]["NY_MEASURE_BIN"] == str(external_binary)
    assert start["solver_binary"]["path"] == str(external_binary.resolve())
    sealed_binary = Path(start["solver_binary"]["sealed_execution"]["path"])
    assert sealed_binary != external_binary
    assert sealed_binary.read_bytes() == external_binary.read_bytes()


def external_scorecard_rejects_receiptless_automatic_binary_before_results(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    (repo / "target/release/ny.receipt").unlink()
    isolated = repo / "reports/receipt-refusal"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_OUTPUT_DIR=str(isolated),
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "receipt must be a regular non-symlink file" in result.stderr
    assert "refusing stale or unproven automatic NY binary" in result.stderr
    assert not isolated.exists(), "receipt refusal must precede result-bank creation"
    assert not (tmp_path / "solver-invocations.jsonl").exists()


def external_integrity_failure_propagates_nonzero_from_completion_trap(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    isolated = repo / "reports/integrity-drift"
    run_id = "20260718T125000Z-integrity-drift"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            MEASUREMENT_TEST_MUTATE_SOLVER="1",
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1, f"{result.stdout}\n{result.stderr}"
    assert "completion integrity validation failed" in result.stderr
    completion = json.loads(
        (isolated / f"artifacts/runs/{run_id}/completion.json").read_text(
            encoding="utf-8"
        )
    )
    codes = {item["code"] for item in completion["integrity"]["violations"]}
    assert completion["exit_status"] == 0
    assert completion["completed_successfully"] is False
    assert completion["integrity"]["status"] == "invalid"
    assert "sealed_solver_binary_unavailable" in codes
    assert any(
        item["code"] == "sealed_solver_binary_unavailable"
        and "sealed file is writable" in item["detail"]
        for item in completion["integrity"]["violations"]
    )


def external_completion_write_failure_propagates_nonzero_from_trap(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    isolated = repo / "reports/completion-collision"
    run_id = "20260718T125500Z-completion-collision"
    run_dir = isolated / f"artifacts/runs/{run_id}"
    run_dir.mkdir(parents=True)
    completion_path = run_dir / "completion.json"
    completion_path.write_text("pre-existing evidence\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1, f"{result.stdout}\n{result.stderr}"
    assert "completion or integrity validation failed" in result.stderr
    assert completion_path.read_text(encoding="utf-8") == "pre-existing evidence\n"
    assert (run_dir / "start.json").is_file()


def external_isolated_output_reruns_legacy_row_and_honors_row_cap(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    legacy_dir = repo / "reports/measured"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "demo.csv").write_text(
        "demo,shared.onnx,violated.vnnlib,0,sat,0\n", encoding="utf-8"
    )
    isolated = repo / "reports/isolated-bank"
    run_id = "20260718T130000Z-isolated"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    with (isolated / "demo.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert len(rows) == 1
    _assert_measurement_row(
        rows[0],
        ["demo", "shared.onnx", "violated.vnnlib", "0", "sat"],
        run_id,
    )
    assert len((legacy_dir / "demo.csv").read_text(encoding="utf-8").splitlines()) == 1

    start = json.loads(
        (isolated / f"artifacts/runs/{run_id}/start.json").read_text(encoding="utf-8")
    )
    assert start["measurement"]["output_dir"] == str(isolated.resolve())
    assert start["measurement"]["max_rows_per_category"] == 1
    assert start["environment"]["values"]["NY_MEASURE_OUTPUT_DIR"] == str(isolated)
    artifacts = list((isolated / "artifacts/demo").glob(f"*/{run_id}.json"))
    assert len(artifacts) == 1


def external_exact_instance_selector_is_provenanced_and_runs_only_that_row(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    isolated = repo / "reports/selected-instance"
    run_id = "20260718T140000Z-selected"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_INSTANCE_INDEX="2",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
            NY_MEASURE_VNNLIB_VERSION="2.0",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    with (isolated / "demo.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert len(rows) == 1
    _assert_measurement_row(
        rows[0],
        ["demo", "shared.onnx", "holds.vnnlib", "0", "unsat"],
        run_id,
    )

    start = json.loads(
        (isolated / f"artifacts/runs/{run_id}/start.json").read_text(encoding="utf-8")
    )
    assert start["measurement"]["instance_index"] == 2
    assert start["measurement"]["vnnlib_version_selection"] == "2.0"
    assert start["environment"]["values"]["NY_MEASURE_INSTANCE_INDEX"] == "2"
    assert start["environment"]["values"]["NY_MEASURE_VNNLIB_VERSION"] == "2.0"
    metadata_paths = list((isolated / "artifacts/demo").glob(f"*/{run_id}.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    assert metadata["instance_index"] == 2


def external_multiple_vnnlib_versions_require_explicit_selection(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    benchmark = repo / "benchmarks/vnncomp2026/benchmarks/demo"
    shutil.copytree(benchmark / "2.0", benchmark / "1.0")
    isolated = repo / "reports/ambiguous-version"
    run_id = "20260718T150000Z-ambiguous"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert "multiple instances.csv files" in result.stderr
    assert "NY_MEASURE_VNNLIB_VERSION" in result.stderr
    assert not (isolated / "demo.csv").exists()

    completion = json.loads(
        (isolated / f"artifacts/runs/{run_id}/completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["exit_status"] == 1
    assert completion["completed_successfully"] is False


def external_top_level_instances_list_precedes_nested_payload_list(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    benchmark = repo / "benchmarks/vnncomp2026/benchmarks/demo"
    top_level = benchmark / "instances.csv"
    top_level.write_text(
        "shared.onnx,violated.vnnlib,1\nshared.onnx,holds.vnnlib,1\n",
        encoding="utf-8",
    )
    shutil.copyfile(benchmark / "2.0/shared.onnx", benchmark / "shared.onnx")
    shutil.copyfile(benchmark / "2.0/violated.vnnlib", benchmark / "violated.vnnlib")
    shutil.copyfile(benchmark / "2.0/holds.vnnlib", benchmark / "holds.vnnlib")
    payload = benchmark / "vnnlib"
    payload.mkdir()
    (payload / "instances.csv").write_text(
        "this,is,not-the-official-list\n", encoding="utf-8"
    )
    _run("git", "add", ".", cwd=repo / "benchmarks/vnncomp2026")
    _run(
        "git",
        "commit",
        "-qm",
        "fixture top-level list and nested payload",
        cwd=repo / "benchmarks/vnncomp2026",
    )
    isolated = repo / "reports/top-level-list"
    run_id = "20260718T153000Z-top-level"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    with (isolated / "demo.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert [row[2] for row in rows] == ["violated.vnnlib", "holds.vnnlib"]


def external_missing_input_fails_without_emitting_unarchived_csv_row(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    (repo / "benchmarks/vnncomp2026/benchmarks/demo/2.0/shared.onnx").unlink()
    isolated = repo / "reports/missing-input"
    run_id = "20260718T155000Z-missing"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "refusing to record an unarchived row with missing inputs" in result.stderr
    assert (isolated / "demo.csv").read_bytes() == b""
    completion = json.loads(
        (isolated / f"artifacts/runs/{run_id}/completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["exit_status"] == 1
    assert completion["completed_successfully"] is False


def external_configs_dir_is_passed_and_content_addressed(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    configs = tmp_path / "experiment-configs"
    preset_dir = configs / "vnncomp2025"
    preset_dir.mkdir(parents=True)
    preset = preset_dir / "demo.yaml"
    preset.write_text("verifier:\n  timeout: 2\n", encoding="utf-8")
    (configs / "README.txt").write_text("experiment A\n", encoding="utf-8")
    isolated = repo / "reports/external-configs"
    run_id = "20260718T160000Z-configs"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_CONFIGS_DIR=str(configs),
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    start_path = isolated / f"artifacts/runs/{run_id}/start.json"
    start = json.loads(start_path.read_text(encoding="utf-8"))
    config_inputs = start["measurement"]["config_inputs"]
    assert start["environment"]["values"]["NY_MEASURE_CONFIGS_DIR"] == str(configs)
    assert config_inputs["declared_path"] == str(configs)
    assert config_inputs["resolved_path"] == str(configs.resolve())
    assert config_inputs["entry_count"] == 3
    assert config_inputs["entries"] == [
        {
            "kind": "file",
            "path": "README.txt",
            "sha256": hashlib.sha256(b"experiment A\n").hexdigest(),
            "size_bytes": len(b"experiment A\n"),
        },
        {"kind": "directory", "path": "vnncomp2025"},
        {
            "kind": "file",
            "path": "vnncomp2025/demo.yaml",
            "sha256": hashlib.sha256(preset.read_bytes()).hexdigest(),
            "size_bytes": len(preset.read_bytes()),
        },
    ]
    manifest = {
        "schema": "ny_measurement_config_inputs_v1",
        "entries": config_inputs["entries"],
    }
    expected_manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    assert (
        config_inputs["manifest_sha256"]
        == hashlib.sha256(expected_manifest_bytes).hexdigest()
    )
    sealed_configs = start["measurement"]["sealed_config_inputs"]
    assert start["measurement"]["solver_command_template"][-2:] == [
        "--configs-dir",
        sealed_configs["resolved_path"],
    ]

    metadata_path = next((isolated / "artifacts/demo").glob(f"*/{run_id}.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["config_inputs"] == {
        key: config_inputs[key]
        for key in (
            "schema",
            "declared_path",
            "resolved_path",
            "entry_count",
            "manifest_sha256",
        )
    }
    solver_log = (
        isolated / "artifacts" / metadata["solver_log"]["artifact"]
    ).read_text(encoding="utf-8")
    assert f" <--configs-dir> <{sealed_configs['resolved_path']}>" in solver_log


def external_configs_dir_rejects_relative_or_missing_path(tmp_path: Path) -> None:
    for suffix, configs_dir, message in (
        ("relative", "configs", "must be an absolute path"),
        ("missing", str(tmp_path / "missing-configs"), "not an existing directory"),
    ):
        fixture_root = tmp_path / suffix
        fixture_root.mkdir()
        repo = _fixture_repo(fixture_root)
        isolated = repo / "reports/rejected-configs"
        run_id = f"20260718T170000Z-{suffix}"
        result = subprocess.run(
            ["bash", "scripts/measure_ny_scorecard.sh"],
            cwd=repo,
            env=_measurement_environment(
                repo,
                fixture_root,
                NY_MEASURE_CONFIGS_DIR=configs_dir,
                NY_MEASURE_OUTPUT_DIR=str(isolated),
                NY_MEASURE_RUN_ID=run_id,
            ),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 2
        assert message in result.stderr
        assert not isolated.exists()
