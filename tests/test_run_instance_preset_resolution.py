# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Regression tests for the thin VNN-COMP shell wrapper and release build.

Preset resolution and branching now live in the native ``ny vnncomp`` command,
where Rust unit tests cover full-name/base-name and newest-directory precedence.
These tests pin the shell boundary: it must preserve the organizer protocol
arguments exactly and must not reintroduce legacy preset/branching policy.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORED_RUN_INSTANCE = REPO_ROOT / "run_instance.sh"
SHARED_RUN_INSTANCE = REPO_ROOT / "vnncomp_scripts" / "run_instance.sh"
BUILD_SUBMISSION_BINARY = REPO_ROOT / "vnncomp_scripts" / "build_submission_binary.sh"
SUBMISSION_BINARY_RECEIPT = (
    REPO_ROOT / "vnncomp_scripts" / "submission_binary_receipt.sh"
)


def _write_fake_ny(tmp_path: Path) -> Path:
    """Create a fake ny binary that records argv/environment and emits status."""
    ny = tmp_path / "fake_ny"
    ny_args_file = tmp_path / "ny-args.txt"
    ny_env_file = tmp_path / "ny-env.txt"
    ny.write_text(
        textwrap.dedent(f"""\
            #!/bin/bash
            printf '%s\n' "$@" > "{ny_args_file}"
            printf 'AY_MILP_SMT=%s\nAY_MILP_GUB_CLIQUE=%s\nAY_MILP_STAB_ORBIT=%s\nAY_MILP_COVER_MINIMAL=%s\nAY_MILP_NODE_PROP=%s\nAY_MILP_IMPLIED_COL_BOUNDS=%s\nAY_MILP_ADOPT_FT_MAX_ROWS=%s\nAY_MILP_NO_SHAPE_CPR=%s\nAY_DISABLE_PHASE_EPOCH_SKIP=%s\nAY_SAT_L0_UNSAT_TRACE=%s\nAY_DUMP_QUERY_DIR=%s\nNY_MARGIN_ROW_RESERVE_MAX_FRAC=%s\nNY_GPU_AUTHORITY_SELFARM=%s\nNY_UPFRONT_ATTACK=%s\nNY_SAFENLP_SHORT_GRACE=%s\nMIMALLOC_PURGE_DELAY=%s\nMIMALLOC_FUTURE_OPTION=%s\n' \
                "${{AY_MILP_SMT-<unset>}}" \
                "${{AY_MILP_GUB_CLIQUE-<unset>}}" \
                "${{AY_MILP_STAB_ORBIT-<unset>}}" \
                "${{AY_MILP_COVER_MINIMAL-<unset>}}" \
                "${{AY_MILP_NODE_PROP-<unset>}}" \
                "${{AY_MILP_IMPLIED_COL_BOUNDS-<unset>}}" \
                "${{AY_MILP_ADOPT_FT_MAX_ROWS-<unset>}}" \
                "${{AY_MILP_NO_SHAPE_CPR-<unset>}}" \
                "${{AY_DISABLE_PHASE_EPOCH_SKIP-<unset>}}" \
                "${{AY_SAT_L0_UNSAT_TRACE-<unset>}}" \
                "${{AY_DUMP_QUERY_DIR-<unset>}}" \
                "${{NY_MARGIN_ROW_RESERVE_MAX_FRAC-<unset>}}" \
                "${{NY_GPU_AUTHORITY_SELFARM-<unset>}}" \
                "${{NY_UPFRONT_ATTACK-<unset>}}" \
                "${{NY_SAFENLP_SHORT_GRACE-<unset>}}" \
                "${{MIMALLOC_PURGE_DELAY-<unset>}}" \
                "${{MIMALLOC_FUTURE_OPTION-<unset>}}" > "{ny_env_file}"
            echo '{{"status": "unknown"}}'
        """),
        encoding="utf-8",
    )
    ny.chmod(0o755)
    return ny


def _setup_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create minimal ONNX and VNNLIB fixture files."""
    onnx_file = tmp_path / "model.onnx"
    vnnlib_file = tmp_path / "prop.vnnlib"
    results_file = tmp_path / "results.txt"
    onnx_file.write_bytes(b"\x08\x01")
    vnnlib_file.write_text("", encoding="utf-8")
    return onnx_file, vnnlib_file, results_file


def _run_instance(
    tmp_path: Path,
    ny_path: Path,
    category: str,
    onnx: Path,
    vnnlib: Path,
    results: Path,
    *,
    runner: Path = SHARED_RUN_INSTANCE,
    extra_env: dict[str, str] | None = None,
    timeout: str = "10",
) -> subprocess.CompletedProcess[str]:
    """Run run_instance.sh with a fake ny binary."""
    env = os.environ.copy()
    env["NY_BIN"] = str(ny_path)
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        [
            "bash",
            str(runner),
            "v1",
            category,
            str(onnx),
            str(vnnlib),
            str(results),
            timeout,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _captured_ny_args(tmp_path: Path) -> list[str]:
    """Read the fake ny argv capture from disk."""
    args_file = tmp_path / "ny-args.txt"
    assert args_file.is_file(), f"missing fake ny args capture: {args_file}"
    return args_file.read_text(encoding="utf-8").splitlines()


def _captured_ny_env(tmp_path: Path) -> dict[str, str]:
    """Read the solver environment observed by the fake ny process."""
    env_file = tmp_path / "ny-env.txt"
    assert env_file.is_file(), f"missing fake ny environment capture: {env_file}"
    return dict(
        line.split("=", 1) for line in env_file.read_text(encoding="utf-8").splitlines()
    )


def _write_fake_cargo(tmp_path: Path) -> None:
    """Create a fake Cargo that emits one fresh, authenticated ny artifact."""
    cargo = tmp_path / "bin" / "cargo"
    cargo.parent.mkdir(parents=True, exist_ok=True)
    cargo.write_text(
        textwrap.dedent("""\
            #!/bin/bash
            set -euo pipefail

            printf '%s\\n' "$@" > "$PWD/cargo-args.txt"
            : "${CARGO_TARGET_DIR:?missing invocation-scoped target directory}"
            artifact="$CARGO_TARGET_DIR/release/ny"
            mkdir -p "$(dirname "$artifact")"
            cat > "$artifact" <<'EOF'
            #!/bin/bash
            echo fake-ny
            EOF
            chmod +x "$artifact"
            printf '{"reason":"compiler-artifact","manifest_path":"%s","target":{"kind":["bin"],"name":"ny"},"filenames":["%s"],"executable":"%s","fresh":false}\\n' \
                "$PWD/crates/ny-cli/Cargo.toml" "$artifact" "$artifact"
            printf '{"reason":"build-finished","success":true}\\n'
        """),
        encoding="utf-8",
    )
    cargo.chmod(0o755)


def _write_archive_source_identity(tmp_path: Path) -> None:
    lock = tmp_path / "Cargo.lock"
    lock.write_text("version = 4\n", encoding="utf-8")
    lock_sha256 = hashlib.sha256(lock.read_bytes()).hexdigest()
    (tmp_path / ".ny-vnncomp-source.txt").write_text(
        "schema=ny-vnncomp-source-v1\n"
        "ny_commit=0123456789abcdef0123456789abcdef01234567\n"
        f"cargo_lock_sha256={lock_sha256}\n",
        encoding="utf-8",
    )


def _copy_build_submission_binary(tmp_path: Path) -> Path:
    """Copy the submission build script into an isolated temp repo.

    shutil.copy preserves the source permission bits, so the copy is
    executable exactly when the checked-in script is — the same direct-exec
    contract install_tool.sh relies on. A chmod here would let this test pass
    against a committed non-executable script.
    """
    script = tmp_path / "vnncomp_scripts" / "build_submission_binary.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(BUILD_SUBMISSION_BINARY, script)
    shutil.copy(
        SUBMISSION_BINARY_RECEIPT,
        script.parent / "submission_binary_receipt.sh",
    )
    _write_archive_source_identity(tmp_path)
    return script


def _run_build_submission_binary(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run build_submission_binary.sh with a fake cargo shim."""
    script = _copy_build_submission_binary(tmp_path)
    _write_fake_cargo(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}:{env.get('PATH', '')}"
    return subprocess.run(
        [str(script)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _automatic_binary_fixture(tmp_path: Path) -> tuple[Path, Path]:
    scripts = tmp_path / "vnncomp_scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    runner = scripts / "run_instance.sh"
    helper = scripts / "submission_binary_receipt.sh"
    shutil.copy(SHARED_RUN_INSTANCE, runner)
    shutil.copy(SUBMISSION_BINARY_RECEIPT, helper)
    _write_archive_source_identity(tmp_path)

    recorded_fake = _write_fake_ny(tmp_path)
    automatic_binary = tmp_path / "target" / "release" / "ny"
    automatic_binary.parent.mkdir(parents=True)
    shutil.copy(recorded_fake, automatic_binary)
    return runner, automatic_binary


def _run_automatic_binary_fixture(
    tmp_path: Path,
    runner: Path,
    *,
    category: str = "cifar100_2024",
) -> subprocess.CompletedProcess[str]:
    onnx, vnnlib, results = _setup_fixtures(tmp_path)
    env = os.environ.copy()
    env.pop("NY_BIN", None)
    return subprocess.run(
        [
            "bash",
            str(runner),
            "v1",
            category,
            str(onnx),
            str(vnnlib),
            str(results),
            "20",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _initialize_git_fixture(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("/target/\n", encoding="utf-8")
    for args in [
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=NY Test",
            "-c",
            "user.email=ny@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
    ]:
        result = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "category",
    [
        # Categories with year suffix in preset filename — previously broken
        "safenlp_2024",
        "cifar100_2024",
        "dist_shift_2023",
        "acasxu_2023",
        "ml4acopf_2024",
        "tinyimagenet_2024",
        "yolo_2023",
        "traffic_signs_recognition_2023",
        # Categories without year suffix — should still work
        "nn4sys_2023",
        "relusplitter",
        "malbeware",
        "cersyve",
        "soundnessbench",
    ],
)
def test_native_vnncomp_delegation_preserves_category(
    tmp_path: Path, category: str
) -> None:
    """The shell must pass every category spelling to native policy unchanged."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)

    result = _run_instance(tmp_path, ny, category, onnx, vnnlib, results)

    assert result.returncode == 0, result.stderr
    assert _captured_ny_args(tmp_path) == [
        "vnncomp",
        "v1",
        category,
        str(onnx),
        str(vnnlib),
        str(results),
        "10",
    ]


@pytest.mark.parametrize(
    "category, timeout",
    [
        # The only two fractional per-instance timeouts in the VNN-COMP 2026 set.
        ("metaroom_2023", "210.0"),
        ("traffic_signs_recognition_2023", "480.0"),
    ],
)
def test_fractional_timeout_still_invokes_ny(
    tmp_path: Path, category: str, timeout: str
) -> None:
    """A fractional CSV timeout must not abort the wrapper before ny launches.

    The OS-level wall-clock backstop is computed with bash integer arithmetic
    (``$(( budget + 10 ))``), which aborts on a decimal point; under ``set -u``
    that left the timeout unset and the ``exec`` line failed BEFORE ny ran, so no
    verdict was ever written for metaroom_2023 (210.0) or
    traffic_signs_recognition_2023 (480.0) — two full categories scored 0. The
    wrapper now floors the budget for the backstop while forwarding the raw,
    unmodified fractional budget to ``ny vnncomp`` (which floors it internally).
    If ny is invoked at all, the wrapper survived the fractional input.
    """
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)

    result = _run_instance(
        tmp_path, ny, category, onnx, vnnlib, results, timeout=timeout
    )

    assert result.returncode == 0, result.stderr
    # ny actually ran (argv captured => the wrapper reached exec, did not crash),
    # and the RAW fractional budget was forwarded verbatim (ny floors internally).
    assert _captured_ny_args(tmp_path) == [
        "vnncomp",
        "v1",
        category,
        str(onnx),
        str(vnnlib),
        str(results),
        timeout,
    ]


@pytest.mark.parametrize("category", ["ml4acopf_2024", "safenlp_2024"])
def test_shell_does_not_inject_legacy_policy_flags(
    tmp_path: Path, category: str
) -> None:
    """Preset and branching policy belong solely to the native command."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)

    result = _run_instance(tmp_path, ny, category, onnx, vnnlib, results)
    ny_args = _captured_ny_args(tmp_path)

    assert result.returncode == 0, result.stderr
    assert ny_args[0] == "vnncomp"
    assert "--branching" not in ny_args, ny_args
    assert "--preset" not in ny_args, ny_args


def test_safenlp_short_grace_remains_default_dark(tmp_path: Path) -> None:
    """The measured-neutral short-grace experiment must not arm in scoring."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)

    result = _run_instance(
        tmp_path,
        ny,
        "safenlp_2024",
        onnx,
        vnnlib,
        results,
        timeout="20",
    )

    assert result.returncode == 0, result.stderr
    environment = _captured_ny_env(tmp_path)
    assert environment["NY_UPFRONT_ATTACK"] == "1"
    assert environment["NY_SAFENLP_SHORT_GRACE"] == "<unset>"


def test_safenlp_short_grace_preserves_explicit_operator_override(
    tmp_path: Path,
) -> None:
    """The shared helper preserves an explicit value for controlled A/B."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)

    result = _run_instance(
        tmp_path,
        ny,
        "safenlp_2024",
        onnx,
        vnnlib,
        results,
        extra_env={"NY_SAFENLP_SHORT_GRACE": "1"},
        timeout="20",
    )

    assert result.returncode == 0, result.stderr
    assert _captured_ny_env(tmp_path)["NY_SAFENLP_SHORT_GRACE"] == "1"


def test_scored_root_wrapper_sanitizes_operator_environment(tmp_path: Path) -> None:
    """The organizer-facing wrapper must start ny with deterministic runtime knobs."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)
    dump_dir = tmp_path / "ay-query-dumps"

    result = _run_instance(
        tmp_path,
        ny,
        "cgan2026",
        onnx,
        vnnlib,
        results,
        runner=SCORED_RUN_INSTANCE,
        extra_env={
            # Several AY experiments test presence rather than truthiness, so
            # adversarial false-looking values must disappear too.
            "AY_MILP_SMT": "0",
            "AY_MILP_GUB_CLIQUE": "1",
            "AY_MILP_STAB_ORBIT": "0",
            "AY_MILP_COVER_MINIMAL": "false",
            "AY_MILP_NODE_PROP": "1",
            "AY_MILP_IMPLIED_COL_BOUNDS": "0",
            "AY_MILP_ADOPT_FT_MAX_ROWS": "0",
            "AY_MILP_NO_SHAPE_CPR": "0",
            "AY_DISABLE_PHASE_EPOCH_SKIP": "0",
            "AY_SAT_L0_UNSAT_TRACE": "0",
            "AY_DUMP_QUERY_DIR": str(dump_dir),
            "NY_MARGIN_ROW_RESERVE_MAX_FRAC": "0.25",
            "NY_GPU_AUTHORITY_SELFARM": "1",
            "MIMALLOC_PURGE_DELAY": "0",
            "MIMALLOC_FUTURE_OPTION": "future-value",
        },
    )

    assert result.returncode == 0, result.stderr
    assert _captured_ny_env(tmp_path) == {
        "AY_MILP_SMT": "<unset>",
        "AY_MILP_GUB_CLIQUE": "<unset>",
        "AY_MILP_STAB_ORBIT": "<unset>",
        "AY_MILP_COVER_MINIMAL": "<unset>",
        "AY_MILP_NODE_PROP": "<unset>",
        "AY_MILP_IMPLIED_COL_BOUNDS": "<unset>",
        "AY_MILP_ADOPT_FT_MAX_ROWS": "<unset>",
        "AY_MILP_NO_SHAPE_CPR": "<unset>",
        "AY_DISABLE_PHASE_EPOCH_SKIP": "<unset>",
        "AY_SAT_L0_UNSAT_TRACE": "<unset>",
        "AY_DUMP_QUERY_DIR": "<unset>",
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC": "<unset>",
        "NY_GPU_AUTHORITY_SELFARM": "<unset>",
        "NY_UPFRONT_ATTACK": "<unset>",
        "NY_SAFENLP_SHORT_GRACE": "<unset>",
        "MIMALLOC_PURGE_DELAY": "<unset>",
        "MIMALLOC_FUTURE_OPTION": "<unset>",
    }


def test_shared_helper_preserves_operator_environment(tmp_path: Path) -> None:
    """Local developer A/B and query-capture runs retain explicit controls."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)
    dump_dir = tmp_path / "ay-query-dumps"

    result = _run_instance(
        tmp_path,
        ny,
        "cgan2026",
        onnx,
        vnnlib,
        results,
        runner=SHARED_RUN_INSTANCE,
        extra_env={
            "AY_MILP_SMT": "1",
            "AY_MILP_GUB_CLIQUE": "1",
            "AY_MILP_STAB_ORBIT": "0",
            "AY_MILP_COVER_MINIMAL": "false",
            "AY_MILP_NODE_PROP": "1",
            "AY_MILP_IMPLIED_COL_BOUNDS": "1",
            "AY_MILP_ADOPT_FT_MAX_ROWS": "16384",
            "AY_MILP_NO_SHAPE_CPR": "1",
            "AY_DISABLE_PHASE_EPOCH_SKIP": "1",
            "AY_SAT_L0_UNSAT_TRACE": "1",
            "AY_DUMP_QUERY_DIR": str(dump_dir),
            "NY_MARGIN_ROW_RESERVE_MAX_FRAC": "0.25",
            "NY_GPU_AUTHORITY_SELFARM": "1",
            "MIMALLOC_PURGE_DELAY": "17",
            "MIMALLOC_FUTURE_OPTION": "future-value",
        },
    )

    assert result.returncode == 0, result.stderr
    assert _captured_ny_env(tmp_path) == {
        "AY_MILP_SMT": "1",
        "AY_MILP_GUB_CLIQUE": "1",
        "AY_MILP_STAB_ORBIT": "0",
        "AY_MILP_COVER_MINIMAL": "false",
        "AY_MILP_NODE_PROP": "1",
        "AY_MILP_IMPLIED_COL_BOUNDS": "1",
        "AY_MILP_ADOPT_FT_MAX_ROWS": "16384",
        "AY_MILP_NO_SHAPE_CPR": "1",
        "AY_DISABLE_PHASE_EPOCH_SKIP": "1",
        "AY_SAT_L0_UNSAT_TRACE": "1",
        "AY_DUMP_QUERY_DIR": str(dump_dir),
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC": "0.25",
        "NY_GPU_AUTHORITY_SELFARM": "1",
        "NY_UPFRONT_ATTACK": "<unset>",
        "NY_SAFENLP_SHORT_GRACE": "<unset>",
        "MIMALLOC_PURGE_DELAY": "17",
        "MIMALLOC_FUTURE_OPTION": "future-value",
    }


def test_build_submission_binary_enables_mip_feature(tmp_path: Path) -> None:
    """build_submission_binary.sh should compile ny-cli with mip support."""
    result = _run_build_submission_binary(tmp_path)

    assert result.returncode == 0, (
        f"build_submission_binary.sh failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    cargo_args = (tmp_path / "cargo-args.txt").read_text(encoding="utf-8").splitlines()
    # The fake cargo accepts every invocation, so the script's FIRST fallback
    # tier ("mip,cuda") is the one that must land here.
    assert cargo_args[:5] == ["build", "--locked", "--release", "-p", "ny-cli"]
    assert cargo_args[-2:] == ["--features", "mip,cuda"]
    assert "--message-format=json-render-diagnostics" in cargo_args
    target_index = cargo_args.index("--target-dir")
    staging = Path(cargo_args[target_index + 1])
    assert staging.parent == tmp_path / "target"
    assert staging.name.startswith(".ny-submission-build.")
    assert (tmp_path / "target" / "release" / "ny").is_file(), (
        "expected build_submission_binary.sh to create target/release/ny"
    )
    receipt = tmp_path / "target" / "release" / "ny.receipt"
    assert receipt.is_file()
    assert "source_commit=0123456789abcdef0123456789abcdef01234567" in (
        receipt.read_text(encoding="utf-8")
    )


def test_build_submission_binary_fails_closed_when_full_tier_fails(
    tmp_path: Path,
) -> None:
    """Competition install must not silently accept a feature-degraded binary."""
    script = _copy_build_submission_binary(tmp_path)
    cargo = tmp_path / "bin" / "cargo"
    cargo.parent.mkdir(parents=True, exist_ok=True)
    cargo.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$PWD/cargo-args.txt"\nexit 9\n',
        encoding="utf-8",
    )
    cargo.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}:{env.get('PATH', '')}"

    result = subprocess.run(
        [str(script)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "required competition feature tier 'mip,cuda' failed" in result.stderr
    assert not (tmp_path / "target" / "release" / "ny").exists()


def test_automatic_binary_selection_rejects_missing_receipt(tmp_path: Path) -> None:
    runner, _ = _automatic_binary_fixture(tmp_path)

    result = _run_automatic_binary_fixture(tmp_path, runner)

    assert result.returncode == 1
    assert "receipt must be a regular non-symlink file" in result.stderr
    assert "refusing stale or unproven automatic NY binary" in result.stderr
    assert (tmp_path / "results.txt").read_text(encoding="utf-8") == "error\n"
    assert not (tmp_path / "ny-args.txt").exists()


def test_local_eval_wrappers_preserve_run_instance_refusal_stderr() -> None:
    for relative in (
        "scripts/local_eval/official_sample.sh",
        "scripts/local_eval/capability_triage.sh",
        "scripts/local_eval/build_test_v2.sh",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        invocation_lines = [
            line
            for line in source.splitlines()
            if "vnncomp_scripts/run_instance.sh" in line
        ]
        assert invocation_lines, relative
        assert all("2>&1" not in line for line in invocation_lines), relative


def test_automatic_binary_selection_accepts_matching_receipt(tmp_path: Path) -> None:
    runner, binary = _automatic_binary_fixture(tmp_path)
    helper = tmp_path / "vnncomp_scripts" / "submission_binary_receipt.sh"
    receipt = subprocess.run(
        ["bash", str(helper), "create-local", str(binary), str(tmp_path), "mip,cuda"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert receipt.returncode == 0, receipt.stderr

    result = _run_automatic_binary_fixture(tmp_path, runner)

    assert result.returncode == 0, result.stderr
    assert "NY binary receipt OK" in result.stderr
    assert _captured_ny_args(tmp_path)[0] == "vnncomp"


def test_automatic_binary_selection_rejects_stale_bytes(tmp_path: Path) -> None:
    runner, binary = _automatic_binary_fixture(tmp_path)
    helper = tmp_path / "vnncomp_scripts" / "submission_binary_receipt.sh"
    receipt = subprocess.run(
        ["bash", str(helper), "create-local", str(binary), str(tmp_path), "mip,cuda"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert receipt.returncode == 0, receipt.stderr
    with binary.open("ab") as changed:
        changed.write(b"\n# changed after receipt\n")

    result = _run_automatic_binary_fixture(tmp_path, runner)

    assert result.returncode == 1
    assert "stale/mismatched binary" in result.stderr
    assert not (tmp_path / "ny-args.txt").exists()


def test_git_source_receipt_rejects_new_untracked_build_input(tmp_path: Path) -> None:
    runner, binary = _automatic_binary_fixture(tmp_path)
    _initialize_git_fixture(tmp_path)
    helper = tmp_path / "vnncomp_scripts" / "submission_binary_receipt.sh"
    receipt = subprocess.run(
        ["bash", str(helper), "create-local", str(binary), str(tmp_path), "mip,cuda"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert receipt.returncode == 0, receipt.stderr
    source = tmp_path / "crates" / "new_build_input.rs"
    source.parent.mkdir()
    source.write_text("pub const VALUE: usize = 1;\n", encoding="utf-8")

    result = _run_automatic_binary_fixture(tmp_path, runner)

    assert result.returncode == 1
    assert "stale source identity" in result.stderr
    assert not (tmp_path / "ny-args.txt").exists()


def test_git_source_receipt_rejects_new_head(tmp_path: Path) -> None:
    runner, binary = _automatic_binary_fixture(tmp_path)
    _initialize_git_fixture(tmp_path)
    helper = tmp_path / "vnncomp_scripts" / "submission_binary_receipt.sh"
    receipt = subprocess.run(
        ["bash", str(helper), "create-local", str(binary), str(tmp_path), "mip,cuda"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert receipt.returncode == 0, receipt.stderr
    tracked = tmp_path / "source-version.txt"
    tracked.write_text("next source revision\n", encoding="utf-8")
    for args in [
        ["add", "source-version.txt"],
        [
            "-c",
            "user.name=NY Test",
            "-c",
            "user.email=ny@example.invalid",
            "commit",
            "-q",
            "-m",
            "advance source",
        ],
    ]:
        committed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert committed.returncode == 0, committed.stderr

    result = _run_automatic_binary_fixture(tmp_path, runner)

    assert result.returncode == 1
    assert "stale source identity" in result.stderr
    assert not (tmp_path / "ny-args.txt").exists()


def test_explicit_binary_override_remains_receipt_free(tmp_path: Path) -> None:
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)

    result = _run_instance(
        tmp_path,
        ny,
        "cifar100_2024",
        onnx,
        vnnlib,
        results,
    )

    assert result.returncode == 0, result.stderr
    assert "explicit NY_BIN override" in result.stderr
    assert not ny.with_suffix(".receipt").exists()
