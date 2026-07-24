# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import resource
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "m7_cuda_validate.sh"
GUARD = REPO_ROOT / "scripts" / "ny-safe-gpu-run"
ATTESTED_VMEM_KIB = 83_886_080


def _lower_child_vmem_below_attestation() -> None:
    """Keep the child contained while making the exact M7 attestation false."""
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
        # Darwin exposes RLIMIT_AS but rejects attempts to change its synthetic
        # infinity value. Its `ulimit -v` remains `unlimited`, which is already
        # distinct from the guard's exact 80-GiB attestation and therefore
        # exercises the same re-exec/refusal paths without a child limit.
        pass


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
                f"ny-m7-pytest-probe-{os.getpid()}.service",
                "--",
                "/bin/true",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("systemd-run is unavailable on this host")
    if probe.returncode != 0:
        pytest.skip(
            f"user systemd transient services unavailable: {probe.stderr.strip()}"
        )


def test_m7_validation_reexecs_through_gpu_guard(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "guard.log"
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        'printf \'wrapped=%s\\n\' "${NY_M7_SAFE_GPU_WRAPPED:-}" > "$M7_GUARD_LOG"\n'
        'printf \'arg=%s\\n\' "$@" >> "$M7_GUARD_LOG"\n'
        'printf \'vmem=%s\\n\' "$(ulimit -v)" >> "$M7_GUARD_LOG"\n'
        "if grep -q '/ny-build.slice/' /proc/self/cgroup; then\n"
        "  printf 'cgroup=ny-build\\n' >> \"$M7_GUARD_LOG\"\n"
        "else\n"
        "  printf 'cgroup=other\\n' >> \"$M7_GUARD_LOG\"\n"
        "fi\n",
        encoding="utf-8",
    )
    guard.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["M7_GUARD_LOG"] = str(log)
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        preexec_fn=_lower_child_vmem_below_attestation,
    )

    assert result.returncode == 0, result.stderr
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["wrapped=1", "arg=bash", f"arg={SCRIPT.resolve()}"]
    assert lines[3].startswith("vmem=")
    assert lines[3] != f"vmem={ATTESTED_VMEM_KIB}"
    parent_cgroup = Path("/proc/self/cgroup")
    parent_is_guarded = parent_cgroup.is_file() and "/ny-build.slice/" in (
        parent_cgroup.read_text(encoding="utf-8")
    )
    assert lines[4] == ("cgroup=ny-build" if parent_is_guarded else "cgroup=other")


def test_m7_validation_refuses_an_unguarded_gpu_run(tmp_path: Path) -> None:
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{empty_bin}:/usr/bin:/bin"
    env.pop("NY_M7_SAFE_GPU_WRAPPED", None)

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        preexec_fn=_lower_child_vmem_below_attestation,
    )

    assert result.returncode == 2
    assert "refusing an unguarded GPU run" in result.stderr


def test_m7_validation_rejects_a_forged_wrapped_marker(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    guard_log = tmp_path / "guard-called"
    guard = fake_bin / "ny-safe-gpu-run"
    guard.write_text(
        '#!/bin/bash\nprintf called > "$M7_GUARD_LOG"\n',
        encoding="utf-8",
    )
    guard.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["M7_GUARD_LOG"] = str(guard_log)
    env["NY_M7_SAFE_GPU_WRAPPED"] = "1"

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        preexec_fn=_lower_child_vmem_below_attestation,
    )

    assert result.returncode == 2
    assert "without the required 80-GiB/slice attestation" in result.stderr
    assert not guard_log.exists(), "forged marker must not invoke or bypass the guard"


def test_m7_forces_the_full_competition_feature_tier() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "NY_ALLOW_DEGRADED_BUILD=0 NY_REQUIRE_MIP=1" in source


def test_m7_requires_the_actual_device_test_pipeline_status() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'device_test_status="${PIPESTATUS[0]}"' in source
    assert '[ "${device_test_status}" -eq 0 ] && grep -q "test result: ok"' in source
    assert "cargo status ${device_test_status}" in source


def test_m7_overrides_inherited_degraded_build_inside_real_guard(
    tmp_path: Path,
) -> None:
    _require_user_systemd()
    fake_repo = tmp_path / "repo"
    fake_script = fake_repo / "scripts/m7_cuda_validate.sh"
    fake_builder = fake_repo / "vnncomp_scripts/build_submission_binary.sh"
    fake_bin = tmp_path / "bin"
    fake_script.parent.mkdir(parents=True)
    fake_builder.parent.mkdir(parents=True)
    fake_bin.mkdir()
    fake_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    fake_builder.write_text(
        "#!/bin/bash\n"
        "printf '%s %s\\n' \"${NY_ALLOW_DEGRADED_BUILD:-unset}\" "
        '"${NY_REQUIRE_MIP:-unset}" > "$M7_TIER_LOG"\n'
        "exit 1\n",
        encoding="utf-8",
    )
    fake_script.chmod(0o755)
    fake_builder.chmod(0o755)
    (fake_bin / "ny-safe-gpu-run").symlink_to(GUARD)

    tier_log = tmp_path / "tier.log"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["M7_TIER_LOG"] = str(tier_log)
    env["NY_ALLOW_DEGRADED_BUILD"] = "1"
    env["NY_REQUIRE_MIP"] = "0"
    env["NY_GPU_LOCK_PATH"] = str(tmp_path / "gpu.lock")
    result = subprocess.run(
        [str(GUARD), "/bin/bash", str(fake_script)],
        cwd=fake_repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert tier_log.read_text(encoding="utf-8") == "0 1\n"
