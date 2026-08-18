# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ny_measurement_provenance.py"
ARCHIVE_SCRIPT = REPO_ROOT / "scripts" / "archive_vnncomp_sat_result.py"
AY_REV = "1560972ade2b04a702dfbd13a2de5444ea216009"
HOST_STATE_FIXTURE = {
    "schema": "ny_measurement_host_state_v1",
    "captured_at_utc": "2026-07-18T00:00:00.000000Z",
    "load_average": {},
    "processes": {},
    "gpu": {},
}
CONTAINMENT_FIXTURE = {
    "schema": "ny_measurement_containment_v1",
    "fixture": True,
}
CUDA_RUNTIME_FIXTURE = {
    "schema": "ny_measurement_cuda_runtime_v1",
    "status": "not_required",
    "reason": "cuda_build_feature_not_declared",
}
CGAN_GENERATOR_BRANCH_BOOLEAN_ENV = (
    "NY_BRANCH_STEM",
    "NY_BRANCH_STEM_PROBE",
    "NY_BRANCH_TRACE",
    "NY_MO_BAB_TRACE",
    "NY_UNSTABLE_COUNT",
)
CGAN_GENERATOR_BRANCH_ENV = (
    *CGAN_GENERATOR_BRANCH_BOOLEAN_ENV,
    "NY_BRANCH_STEM_K",
    "NY_BRANCH_STEM_NODES",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("ny_measurement_provenance", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


provenance = _load_module()
REAL_CAPTURE_MEASUREMENT_CONTAINMENT = provenance._capture_measurement_containment
REAL_CAPTURE_CUDA_RUNTIME_DEPENDENCY = (
    provenance._capture_cuda_runtime_dependency
)
REAL_CAPTURE_AND_SEAL_CUDA_RUNTIME_DEPENDENCY = (
    provenance._capture_and_seal_cuda_runtime_dependency
)


@pytest.fixture(autouse=True)
def _stub_measurement_containment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NY_MEASURE_EXPECTED_CPUS", raising=False)
    monkeypatch.setattr(
        provenance,
        "_capture_measurement_containment",
        lambda: CONTAINMENT_FIXTURE,
    )
    monkeypatch.setattr(
        provenance,
        "_capture_cuda_runtime_dependency",
        lambda _binary, _environment, _features: CUDA_RUNTIME_FIXTURE,
    )
    monkeypatch.setattr(
        provenance,
        "_capture_and_seal_cuda_runtime_dependency",
        lambda _binary, _environment, _features, _run_dir: CUDA_RUNTIME_FIXTURE,
    )
    monkeypatch.setattr(
        provenance,
        "_recapture_cuda_runtime_from_start",
        lambda start: start["dependencies"]["cuda_runtime"],
    )


def _load_archive_module():
    spec = importlib.util.spec_from_file_location(
        "archive_vnncomp_sat_result_for_provenance_tests", ARCHIVE_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


archive = _load_archive_module()


def _run(*command: str, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _init_git_repo(path: Path) -> None:
    _run("git", "init", "-q", cwd=path)
    _run("git", "config", "user.name", "NY Test", cwd=path)
    _run("git", "config", "user.email", "ny-test@example.invalid", cwd=path)


def _measurement_repos(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "ny"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / ".gitignore").write_text(
        "/target/\n/reports/\n/benchmarks/\n", encoding="utf-8"
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
    sweep = scripts / "measure_ny_scorecard.sh"
    sweep.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    binary = repo / "target" / "release" / "ny"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 'ny 0.1.0-test'\n", encoding="utf-8")
    binary.chmod(0o755)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "fixture", cwd=repo)
    # Git records whole-second commit epochs, which can tick over between the
    # fixture binary write and commit. Model a just-built solver deterministically.
    binary.touch()

    benchmark = tmp_path / "vnncomp-benchmarks"
    benchmark.mkdir()
    _init_git_repo(benchmark)
    benchmark_root = benchmark / "benchmarks"
    benchmark_root.mkdir()
    (benchmark_root / "README").write_text("fixture\n", encoding="utf-8")
    _run("git", "add", ".", cwd=benchmark)
    _run("git", "commit", "-qm", "fixture", cwd=benchmark)
    _run(
        "git",
        "remote",
        "add",
        "origin",
        "https://user:supersecret@github.com/example/bench.git?token=hidden",
        cwd=benchmark,
    )
    return repo, benchmark_root


def _clear_ny_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if (
            key.startswith(("NY_", "AY_", "MIMALLOC_", "LD_"))
            or key.upper().startswith("MIMALLOC_")
            or key.startswith(provenance.UNREVIEWED_SOLVER_RUNTIME_ENV_PREFIXES)
            or key in provenance.UNREVIEWED_SOLVER_RUNTIME_ENV_EXACT
            or key
            in {
                "DYLD_FORCE_FLAT_NAMESPACE",
                "DYLD_INSERT_LIBRARIES",
                "DYLD_LIBRARY_PATH",
            }
        ):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("GPU_AVAILABLE", raising=False)
    monkeypatch.setenv("NY_BUILD_FEATURES", "mip,cuda")


@pytest.mark.parametrize(
    "key",
    [
        "DYLD_FORCE_FLAT_NAMESPACE",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "LD_AUDIT",
        "LD_PRELOAD",
    ],
)
def test_measurement_environment_rejects_dynamic_loader_injection(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "")

    with pytest.raises(provenance.ProvenanceError, match="dynamic-loader injection"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    [
        "__GLX_VENDOR_LIBRARY_NAME",
        "__NV_PRIME_RENDER_OFFLOAD",
        "__VK_LAYER_NV_optimus",
        "ACCELERATE_NEW_LAPACK",
        "ACO_DEBUG",
        "AMD_CU_MASK",
        "ANV_DEBUG",
        "BLIS_NUM_THREADS",
        "CANDLE_GRAD_DO_NOT_DETACH",
        "CARGO_HOME",
        "CUBLASLT_LOG_LEVEL",
        "CUDA_DEVICE_ORDER",
        "CUDA_LAUNCH_BLOCKING",
        "CUDNN_LOGLEVEL_DBG",
        "DISABLE_LAYER_NV_OPTIMUS_1",
        "DRI_PRIME",
        "DRI_PRIME_DEBUG",
        "GALLIUM_DRIVER",
        "GBM_BACKEND",
        "GCONV_PATH",
        "GLIBC_TUNABLES",
        "GOMP_CPU_AFFINITY",
        "GOTO_NUM_THREADS",
        "INTEL_NO_HW",
        "KMP_AFFINITY",
        "LC_CTYPE",
        "LIBGL_ALWAYS_SOFTWARE",
        "LP_NUM_THREADS",
        "LVP_DEBUG",
        "MALLOC_CONF",
        "MATMUL_NUM_THREADS",
        "MESA_LOADER_DRIVER_OVERRIDE",
        "MESA_VK_DEVICE_SELECT",
        "MKL_DYNAMIC",
        "NODEVICE_SELECT",
        "NOUVEAU_USE_ZINK",
        "NVBLAS_GPU_LIST",
        "NVIDIA_TF32_OVERRIDE",
        "NVRTC_APPEND_FLAGS",
        "NVVM_IR_VER_CHK",
        "OMP_DYNAMIC",
        "OPENBLAS_CORETYPE",
        "ORT_LOAD_CONFIG_FROM_MODEL",
        "PYTHONPATH",
        "RADV_DEBUG",
        "RAYON_RS_NUM_CPUS",
        "RUST_MIN_STACK",
        "RUSTUP_DIST_SERVER",
        "TEMP",
        "TEMPDIR",
        "VECLIB_MAXIMUM_THREADS",
        "VK_ICD_FILENAMES",
        "VULKAN_SDK",
        "WGPU_ADAPTER_NAME",
        "WGPU_BACKEND",
        "XDG_CACHE_HOME",
        "ZINK_DEBUG",
        "mimalloc_eager_commit",
        "MiMaLlOc_Reserve_Huge_Os_Pages",
    ],
)
def test_measurement_environment_rejects_unreviewed_solver_runtime_control(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "1")

    with pytest.raises(
        provenance.ProvenanceError,
        match="unreviewed solver-runtime environment controls",
    ):
        provenance._capture_environment()


def test_measurement_environment_records_reviewed_solver_runtime_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    controls = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": "0",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "3",
        "ORT_LOG_SEVERITY_LEVEL": "4",
        "RAYON_NUM_THREADS": "5",
    }
    for key, value in controls.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    assert {key: environment["values"][key] for key in controls} == controls


@pytest.mark.parametrize(
    "value",
    [
        "",
        ":/opt/cuda/lib64",
        "/opt/cuda/lib64:",
        "/opt/cuda/lib64::/usr/lib",
        "relative/cuda",
        "/opt/cuda/lib64:relative/cuda",
    ],
)
def test_measurement_environment_rejects_unsafe_ld_library_path(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("LD_LIBRARY_PATH", value)

    with pytest.raises(provenance.ProvenanceError, match="LD_LIBRARY_PATH"):
        provenance._capture_environment()


def test_measurement_environment_accepts_dedicated_cuda_ld_library_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    first = tmp_path / "cuda-driver"
    second = tmp_path / "cuda-math"
    first.mkdir()
    second.mkdir()
    (first / "libcuda.so.1").write_bytes(b"driver")
    (second / "libcublas.so.13").write_bytes(b"cublas")
    value = f"{first}:{second}"
    monkeypatch.setenv("LD_LIBRARY_PATH", value)

    environment = provenance._capture_environment()

    assert environment["values"]["LD_LIBRARY_PATH"] == value


def test_measurement_environment_rejects_non_cuda_loader_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    loader_dir = tmp_path / "loader"
    loader_dir.mkdir()
    (loader_dir / "libcuda.so.1").write_bytes(b"driver")
    (loader_dir / "libc.so.6").write_bytes(b"injection")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(loader_dir))

    with pytest.raises(provenance.ProvenanceError, match="unsafe entry"):
        provenance._capture_environment()


def test_measurement_environment_rejects_system_loader_preload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    preload = tmp_path / "ld.so.preload"
    preload.write_text("/tmp/inject.so\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "SYSTEM_LD_SO_PRELOAD", preload)

    with pytest.raises(provenance.ProvenanceError, match="ld.so.preload"):
        provenance._capture_environment()


def _cuda_runtime_object(role: str, path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    file_stat = resolved.stat()
    fingerprint = provenance._file_fingerprint(resolved)
    return {
        "role": role,
        "provider_symbol": f"fixture_{role}_symbol",
        "mapped_path": str(resolved),
        "resolved_path": str(resolved),
        "mapped_device_major": os.major(file_stat.st_dev),
        "mapped_device_minor": os.minor(file_stat.st_dev),
        "mapped_inode": file_stat.st_ino,
        "size_bytes": file_stat.st_size,
        "sha256": provenance._sha256_file(resolved),
        "fingerprint": fingerprint,
    }


def _cuda_runtime_report(objects: list[dict[str, object]]) -> dict[str, object]:
    roles = {str(item["role"]) for item in objects}
    if "nvrtc" in roles and "nvrtc_builtins" in roles:
        nvrtc_status = "loaded_with_builtins"
    elif "nvrtc" in roles:
        nvrtc_status = "loaded"
    elif "nvrtc_builtins" in roles:
        nvrtc_status = "builtins_loaded_without_nvrtc"
    else:
        nvrtc_status = "not_loaded_feature_disabled"
    return {
        "schema": provenance.CUDA_RUNTIME_INFO_SCHEMA,
        "device_name": "fixture GPU",
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
        "candidates": {
            "driver": ["libcuda.so", "libcuda.so.1"],
            "cublas": ["libcublas.so", "libcublas.so.13"],
            "cublas_lt": ["libcublasLt.so", "libcublasLt.so.13"],
            "nvrtc": ["libnvrtc.so", "libnvrtc.so.13"],
        },
        "objects": objects,
        "nvrtc_status": nvrtc_status,
    }


def _stub_cuda_runtime_probe(
    monkeypatch: pytest.MonkeyPatch,
    reports: list[dict[str, object]],
) -> None:
    remaining = iter(reports)

    def fake_run(command, **_kwargs):
        assert command[-1] == "--cuda-runtime-info"
        report = next(remaining)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(json.dumps(report) + "\n").encode(),
            stderr=b"",
        )

    monkeypatch.setattr(provenance, "_run", fake_run)


@pytest.mark.parametrize(
    ("ordinary_transport", "discrete_mode"),
    [
        (None, False),
        ("explicit-device-copy", False),
        ("unified-memory", True),
        ("unknown-transport", False),
    ],
)
def test_cuda_runtime_probe_rejects_missing_or_inconsistent_ordinary_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ordinary_transport: str | None,
    discrete_mode: bool,
) -> None:
    report = _cuda_runtime_report([])
    if ordinary_transport is None:
        report.pop("ordinary_gemm_transport")
    else:
        report["ordinary_gemm_transport"] = ordinary_transport
    report["discrete_mode"] = discrete_mode
    _stub_cuda_runtime_probe(monkeypatch, [report])

    with pytest.raises(provenance.ProvenanceError, match="qualification is incomplete"):
        provenance._cuda_runtime_probe(tmp_path / "ny")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ordinary_gemm_transport_policy", "unknown-policy"),
        ("ordinary_gemm_transport_reason", "unknown-reason"),
        ("explicit_device_copy", False),
        ("integrated_device", True),
        ("integrated_device", 0),
        ("pageable_access_uses_host_page_tables", True),
        ("ordinary_gemm_transport_reason", "explicit-transport-override"),
        ("deadline_f64_transport", "direct-host-page-tables"),
    ],
)
def test_cuda_runtime_probe_rejects_inconsistent_auto_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    report = _cuda_runtime_report([])
    report[field] = value
    _stub_cuda_runtime_probe(monkeypatch, [report])

    with pytest.raises(provenance.ProvenanceError, match="qualification is incomplete"):
        provenance._cuda_runtime_probe(tmp_path / "ny")


def test_cuda_runtime_identity_hashes_required_and_optional_mapped_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        role: tmp_path / name
        for role, name in {
            "driver": "libcuda.so.1",
            "cublas": "libcublas.so.13",
            "cublas_lt": "libcublasLt.so.13",
            "nvrtc": "libnvrtc.so.13",
            "nvrtc_builtins": "libnvrtc-builtins.so.13.2",
        }.items()
    }
    for role, path in paths.items():
        path.write_bytes(f"{role} fixture\n".encode())
    report = _cuda_runtime_report(
        [_cuda_runtime_object(role, path) for role, path in paths.items()]
    )
    _stub_cuda_runtime_probe(monkeypatch, [report, report])

    identity = provenance._capture_cuda_runtime_identity(tmp_path / "ny")

    assert identity["schema"] == provenance.MEASUREMENT_CUDA_RUNTIME_SCHEMA
    assert identity["status"] == "captured"
    assert identity["probe"]["nvrtc_status"] == "loaded_with_builtins"
    captured = {item["role"]: item for item in identity["objects"]}
    assert set(captured) == set(paths)
    for role, path in paths.items():
        assert captured[role]["sha256"] == provenance._sha256_file(path)
        assert captured[role]["fingerprint"]["inode"] == path.stat().st_ino


def test_cuda_runtime_identity_records_nvrtc_as_not_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        role: tmp_path / name
        for role, name in {
            "driver": "libcuda.so.1",
            "cublas": "libcublas.so.13",
            "cublas_lt": "libcublasLt.so.13",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"fixture\n")
    report = _cuda_runtime_report(
        [_cuda_runtime_object(role, path) for role, path in paths.items()]
    )
    _stub_cuda_runtime_probe(monkeypatch, [report, report])

    identity = provenance._capture_cuda_runtime_identity(tmp_path / "ny")

    assert identity["probe"]["nvrtc_status"] == "not_loaded_feature_disabled"
    assert {item["role"] for item in identity["objects"]} == set(paths)


def test_cuda_runtime_probe_rejects_missing_transitive_cublas_lt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = tmp_path / "libcuda.so.1"
    cublas = tmp_path / "libcublas.so.13"
    driver.write_bytes(b"driver\n")
    cublas.write_bytes(b"cublas\n")
    report = _cuda_runtime_report(
        [
            _cuda_runtime_object("driver", driver),
            _cuda_runtime_object("cublas", cublas),
        ]
    )
    _stub_cuda_runtime_probe(monkeypatch, [report])

    with pytest.raises(provenance.ProvenanceError, match="cublas_lt"):
        provenance._capture_cuda_runtime_identity(tmp_path / "ny")


def test_cuda_runtime_identity_rejects_mapped_inode_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        role: tmp_path / name
        for role, name in {
            "driver": "libcuda.so.1",
            "cublas": "libcublas.so.13",
            "cublas_lt": "libcublasLt.so.13",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"fixture\n")
    objects = [
        _cuda_runtime_object(role, path) for role, path in paths.items()
    ]
    objects[0]["mapped_inode"] = int(objects[0]["mapped_inode"]) + 1
    objects[0]["fingerprint"]["inode"] = int(
        objects[0]["fingerprint"]["inode"]
    ) + 1
    report = _cuda_runtime_report(objects)
    _stub_cuda_runtime_probe(monkeypatch, [report])

    with pytest.raises(provenance.ProvenanceError, match="changed before"):
        provenance._capture_cuda_runtime_identity(tmp_path / "ny")


def test_cuda_runtime_identity_rejects_selection_change_during_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        role: tmp_path / name
        for role, name in {
            "driver": "libcuda.so.1",
            "cublas": "libcublas.so.13",
            "cublas_lt": "libcublasLt.so.13",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"fixture\n")
    first = _cuda_runtime_report(
        [_cuda_runtime_object(role, path) for role, path in paths.items()]
    )
    second = json.loads(json.dumps(first))
    second["ordinary_gemm_transport_policy"] = "override-explicit-device-copy"
    second["ordinary_gemm_transport_reason"] = "explicit-transport-override"
    _stub_cuda_runtime_probe(monkeypatch, [first, second])

    with pytest.raises(provenance.ProvenanceError, match="selection changed"):
        provenance._capture_cuda_runtime_identity(tmp_path / "ny")


def _fake_containment_tree(
    tmp_path: Path,
    *,
    expected_cpus: int = 10,
) -> tuple[Path, Path, Path, Path, str]:
    assert expected_cpus in {10, 20}
    uid = os.getuid()
    cgroup_root = tmp_path / "cgroup"
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
        "memory.high": str(provenance.EXPECTED_MEMORY_HIGH_BYTES),
        "memory.max": str(provenance.EXPECTED_MEMORY_MAX_BYTES),
        "memory.swap.max": str(provenance.EXPECTED_MEMORY_SWAP_MAX_BYTES),
        "pids.max": str(provenance.EXPECTED_PIDS_MAX),
        "cpu.max": f"{expected_cpus * 100_000} 100000",
    }
    for name, value in policy_values.items():
        parent_value = "max 100000" if name == "cpu.max" else "max"
        (policy / name).write_text(f"{parent_value}\n", encoding="ascii")
        (current / name).write_text(f"{value}\n", encoding="ascii")

    user_slice = cgroup_root / f"user.slice/user-{uid}.slice"
    for name, value in {
        "memory.high": "max",
        "memory.max": "max",
        "memory.swap.max": "max",
        "pids.max": "337805",
        "cpu.max": "max 100000",
    }.items():
        (user_slice / name).write_text(f"{value}\n", encoding="ascii")

    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text(f"0::{membership}\n", encoding="ascii")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"39 29 0:33 / {cgroup_root} rw,nosuid,nodev - cgroup2 cgroup2 rw\n",
        encoding="ascii",
    )
    return proc_cgroup, proc_mountinfo, cgroup_root, current, policy_membership


def _capture(
    repo: Path,
    benchmark_root: Path,
    *,
    run_id: str,
    configs_dir: Path | None = None,
) -> Path:
    return provenance.capture_start_manifest(
        repo_root=repo,
        binary=Path("target/release/ny"),
        benchmark_root=benchmark_root,
        artifact_root=Path("reports/measured/artifacts"),
        run_id=run_id,
        output_dir=Path("reports/measured"),
        scratch_dir=repo.parent / "scratch",
        result_file=repo.parent / "scratch" / "result.txt",
        solver_log_file=repo.parent / "scratch" / "solver.log",
        categories_raw="demo other_demo",
        timeout_cap_seconds=120,
        watchdog_grace_seconds=30,
        max_rows_per_category=0,
        instance_index=0,
        vnnlib_version="",
        sweep_script=Path("scripts/measure_ny_scorecard.sh"),
        configs_dir=configs_dir,
    )


def _seed_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
) -> tuple[Path, Path, Path, Path, Path]:
    repo, benchmark_root = _measurement_repos(tmp_path)
    onnx_file = benchmark_root / "model.onnx"
    vnnlib_file = benchmark_root / "property.vnnlib"
    onnx_file.write_bytes(b"model fixture\n")
    vnnlib_file.write_text("; property fixture\n", encoding="utf-8")
    _run("git", "add", ".", cwd=benchmark_root.parent)
    _run("git", "commit", "-qm", "add measurement inputs", cwd=benchmark_root.parent)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id=run_id)
    result_file = tmp_path / "result.txt"
    solver_log_file = tmp_path / "solver.log"
    result_file.write_bytes(b"unsat\n")
    solver_log_file.write_bytes(b"solver log\n")
    artifact_root = repo / "reports/measured/artifacts"
    preflight_manifest = archive.seal_inputs(
        artifact_root=artifact_root,
        run_id=run_id,
        category="demo",
        instance_index=1,
        onnx="model.onnx",
        vnnlib="property.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        start_manifest=start,
    )
    archived_result = archive.archive_result(
        result_file=result_file,
        solver_log_file=solver_log_file,
        artifact_root=artifact_root,
        run_id=run_id,
        category="demo",
        instance_index=1,
        onnx="model.onnx",
        vnnlib="property.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        solver_verdict="unsat",
        solver_exit_status=0,
        timeout_seconds=120,
        elapsed_seconds=9,
        source_csv="reports/measured/demo.csv",
        start_manifest=start,
        preflight_manifest=preflight_manifest,
    )
    csv_path = repo / "reports/measured/demo.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        f"demo,model.onnx,property.vnnlib,0,unsat,9,{run_id}\n",
        encoding="utf-8",
    )
    metadata_path = archived_result.parent / f"{run_id}.json"
    solver_log_path = archived_result.parent / f"{run_id}.solver.log"
    cache_path = start.with_name("input_hash_cache.json")
    return start, metadata_path, archived_result, solver_log_path, cache_path


def test_seal_file_refuses_content_matching_writable_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "sealed"
    contents = b"#!/bin/sh\nexit 0\n"
    source.write_bytes(contents)
    source.chmod(0o755)
    destination.write_bytes(contents)
    destination.chmod(0o755)

    with pytest.raises(provenance.ProvenanceError, match="sealed file is writable"):
        provenance._seal_file(source, destination, executable=True)

    assert destination.read_bytes() == contents
    assert destination.stat().st_mode & 0o777 == 0o755


def test_completion_rejects_post_start_writable_sealed_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="writable-sealed-solver")
    start_record = json.loads(start.read_text(encoding="utf-8"))
    sealed_path = Path(start_record["solver_binary"]["sealed_execution"]["path"])
    sealed_path.chmod(0o755)

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    violations = record["integrity"]["violations"]

    assert record["completed_successfully"] is False
    assert record["integrity"]["checks"]["sealed_solver_binary"]["status"] == "invalid"
    assert any(
        item["code"] == "sealed_solver_binary_unavailable"
        and "sealed file is writable" in item["detail"]
        for item in violations
    )


@pytest.mark.parametrize(
    ("selector", "expected_cpus"),
    [(None, 10), ("20", 20)],
)
def test_containment_capture_resolves_and_types_selected_cpu_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str | None,
    expected_cpus: int,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, current, _policy_membership = (
        _fake_containment_tree(tmp_path, expected_cpus=expected_cpus)
    )
    if selector is not None:
        monkeypatch.setenv("NY_MEASURE_EXPECTED_CPUS", selector)
    monkeypatch.setattr(
        provenance.resource,
        "getrlimit",
        lambda _resource: (
            provenance.EXPECTED_RLIMIT_AS_BYTES,
            provenance.EXPECTED_RLIMIT_AS_BYTES,
        ),
    )

    containment = REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
        proc_self_cgroup=proc_cgroup,
        proc_self_mountinfo=proc_mountinfo,
        cgroup_root=cgroup_root,
    )

    assert containment["schema"] == "ny_measurement_containment_v1"
    assert containment["membership"].endswith(
        f"/ny-safe-gpu-{os.getuid()}-1234-5678.service"
    )
    assert containment["current_cgroup"] == str(current.resolve())
    assert containment["policy_cgroup"] == str(current.resolve())
    assert containment["effective"]["memory.high"]["value_bytes"] == 64 * 1024**3
    assert containment["effective"]["memory.max"]["value_bytes"] == 80 * 1024**3
    assert containment["effective"]["memory.swap.max"]["value_bytes"] == 8 * 1024**3
    assert containment["effective"]["pids.max"]["value"] == 4096
    assert containment["effective"]["cpu.max"]["equivalent_cpus"] == expected_cpus
    assert containment["policy"]["cpu.max"] == {
        "raw": f"{expected_cpus * 100_000} 100000",
        "quota_us": expected_cpus * 100_000,
        "period_us": 100_000,
        "equivalent_cpus": expected_cpus,
    }
    assert containment["rlimit_as"] == {
        "soft_bytes": 160 * 1024**3,
        "hard_bytes": 160 * 1024**3,
    }


@pytest.mark.parametrize(
    ("control", "wrong"),
    [
        ("memory.high", str(63 * 1024**3)),
        ("memory.max", str(79 * 1024**3)),
        ("memory.swap.max", str(7 * 1024**3)),
        ("pids.max", "4095"),
        ("cpu.max", "900000 100000"),
    ],
)
def test_containment_capture_rejects_policy_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    wrong: str,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, current, _policy_membership = (
        _fake_containment_tree(tmp_path)
    )
    (current / control).write_text(f"{wrong}\n", encoding="ascii")
    monkeypatch.setattr(
        provenance.resource,
        "getrlimit",
        lambda _resource: (
            provenance.EXPECTED_RLIMIT_AS_BYTES,
            provenance.EXPECTED_RLIMIT_AS_BYTES,
        ),
    )

    with pytest.raises(provenance.ProvenanceError, match="reviewed policy"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )


def test_containment_capture_rejects_selected_cpu_policy_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, _current, _policy_membership = (
        _fake_containment_tree(tmp_path, expected_cpus=10)
    )
    monkeypatch.setenv("NY_MEASURE_EXPECTED_CPUS", "20")
    monkeypatch.setattr(
        provenance.resource,
        "getrlimit",
        lambda _resource: (
            provenance.EXPECTED_RLIMIT_AS_BYTES,
            provenance.EXPECTED_RLIMIT_AS_BYTES,
        ),
    )

    with pytest.raises(provenance.ProvenanceError, match="reviewed policy"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )


@pytest.mark.parametrize(
    ("control", "wrong", "message"),
    [
        ("memory.max", str(70 * 1024**3), "effective memory.max"),
        ("pids.max", "2048", "effective pids.max"),
        ("cpu.max", "500000 100000", "effective cpu.max"),
        ("cpu.max", "forged", "malformed cpu.max"),
    ],
)
def test_containment_capture_rejects_tighter_or_malformed_ancestor_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    wrong: str,
    message: str,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, _current, policy_membership = (
        _fake_containment_tree(tmp_path)
    )
    policy = cgroup_root / policy_membership.lstrip("/")
    (policy / control).write_text(f"{wrong}\n", encoding="ascii")
    monkeypatch.setattr(
        provenance.resource,
        "getrlimit",
        lambda _resource: (
            provenance.EXPECTED_RLIMIT_AS_BYTES,
            provenance.EXPECTED_RLIMIT_AS_BYTES,
        ),
    )

    with pytest.raises(provenance.ProvenanceError, match=message):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )


@pytest.mark.parametrize(
    ("soft_delta", "hard_delta"),
    [(-1, 0), (0, -1)],
)
def test_containment_capture_rejects_either_rlimit_as_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    soft_delta: int,
    hard_delta: int,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, _current, _policy_membership = (
        _fake_containment_tree(tmp_path)
    )
    expected = provenance.EXPECTED_RLIMIT_AS_BYTES
    monkeypatch.setattr(
        provenance.resource,
        "getrlimit",
        lambda _resource: (expected + soft_delta, expected + hard_delta),
    )

    with pytest.raises(provenance.ProvenanceError, match="RLIMIT_AS"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )


def test_containment_capture_rejects_wrong_membership_and_ambiguous_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, _current, _policy_membership = (
        _fake_containment_tree(tmp_path)
    )
    monkeypatch.setattr(
        provenance.resource,
        "getrlimit",
        lambda _resource: (
            provenance.EXPECTED_RLIMIT_AS_BYTES,
            provenance.EXPECTED_RLIMIT_AS_BYTES,
        ),
    )
    proc_cgroup.write_text("0::/ny-build.slice/forged.scope\n", encoding="ascii")
    with pytest.raises(provenance.ProvenanceError, match="exact ny-safe-gpu"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )

    _proc_cgroup, duplicate_mountinfo, _root, _current, _policy = (
        _fake_containment_tree(tmp_path / "duplicate")
    )
    duplicate_mountinfo.write_text(
        duplicate_mountinfo.read_text(encoding="ascii") * 2,
        encoding="ascii",
    )
    with pytest.raises(provenance.ProvenanceError, match="exactly one cgroup-v2"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=_proc_cgroup,
            proc_self_mountinfo=duplicate_mountinfo,
            cgroup_root=_root,
        )


def test_start_manifest_hashes_worktree_and_excludes_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    (repo / "notes.txt").write_text("untracked evidence\n", encoding="utf-8")
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MEASURE_CAP", "120")
    monkeypatch.setenv("NY_MARGIN_ROW_ADAPTIVE_RESERVE", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_STACK", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_CLASSWISE", "1")
    monkeypatch.setenv("NY_ROOT_GEMM", "faer")
    monkeypatch.setenv("NY_WARMUP_ITERS", "03")
    monkeypatch.setenv("NY_CROWN_CUT_SEGMENT", "0")
    monkeypatch.setenv("NY_INPUT_SPLIT_WARM_PARALLEL", "1")
    monkeypatch.setenv("GITHUB_TOKEN", "must-never-appear")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest_path = _capture(repo, benchmark_root, run_id="run-one")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)

    assert manifest["schema"] == "ny_measurement_start_v1"
    assert manifest["ny"]["clean"] is False
    assert manifest["ny"]["untracked_files"] == [
        {
            "kind": "file",
            "path": "notes.txt",
            "sha256": provenance._sha256(b"untracked evidence\n"),
            "size_bytes": len(b"untracked evidence\n"),
        }
    ]
    assert len(manifest["ny"]["worktree_evidence_sha256"]) == 64
    assert manifest["solver_binary"]["sha256"] == provenance._sha256_file(
        repo / "target/release/ny"
    )
    assert manifest["solver_binary"]["declared_build_features"] == ["mip", "cuda"]
    assert manifest["dependencies"]["ay"]["git_revision"] == AY_REV
    assert manifest["dependencies"]["cuda_runtime"] == CUDA_RUNTIME_FIXTURE
    assert manifest["rust_toolchain"]["channel"] == "1.95.0"
    assert manifest["rust_toolchain"]["probe_tool"]["sha256"]
    assert manifest["provenance_tools"]["git"]["sha256"]
    assert manifest["benchmark"]["remotes"] == [
        {
            "name": "origin",
            "fetch_url": "https://github.com/example/bench.git",
        }
    ]
    assert manifest["measurement"]["timeout_cap_seconds"] == 120
    assert manifest["measurement"]["categories"] == ["demo", "other_demo"]
    assert manifest["environment"]["values"]["NY_MEASURE_CAP"] == "120"
    assert manifest["environment"]["values"]["NY_MARGIN_ROW_ADAPTIVE_RESERVE"] == "1"
    assert manifest["environment"]["values"]["NY_MARGIN_ROW_DOMAIN_STACK"] == "1"
    assert manifest["environment"]["values"]["NY_MARGIN_ROW_CLASSWISE"] == "1"
    assert manifest["environment"]["values"]["NY_ROOT_GEMM"] == "faer"
    assert manifest["environment"]["values"]["NY_WARMUP_ITERS"] == "03"
    assert manifest["environment"]["values"]["NY_CROWN_CUT_SEGMENT"] == "0"
    assert manifest["environment"]["values"]["NY_INPUT_SPLIT_WARM_PARALLEL"] == "1"
    assert manifest["environment"]["typed_values"] == {
        "NY_CROWN_CUT_SEGMENT": {
            "type": "nonnegative_integer",
            "value": 0,
        },
        "NY_WARMUP_ITERS": {
            "type": "nonnegative_integer",
            "value": 3,
        },
    }
    assert manifest["host"]["containment"] == CONTAINMENT_FIXTURE
    assert manifest["host_state"] == HOST_STATE_FIXTURE
    assert b"must-never-appear" not in manifest_bytes
    assert b"supersecret" not in manifest_bytes
    assert b"token=hidden" not in manifest_bytes


def test_pinned_toolchain_version_probe_must_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "rust-toolchain.toml").write_text(
        '[toolchain]\nchannel = "1.95.0"\n',
        encoding="utf-8",
    )

    def fail_probe(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=b"",
            stderr=b"missing pinned toolchain",
        )

    monkeypatch.setattr(provenance, "_run", fail_probe)
    true_executable = shutil.which("true")
    assert true_executable is not None, "true is required by this fixture"
    with pytest.raises(
        provenance.ProvenanceError,
        match="pinned rustc version probe failed",
    ):
        provenance._parse_toolchain(
            repo,
            declared_tool_path=true_executable,
            declared_tool_kind="rustc",
        )


def test_untracked_content_changes_worktree_evidence_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    note = repo / "notes.txt"
    note.write_text("one\n", encoding="utf-8")
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    first = json.loads(_capture(repo, benchmark_root, run_id="run-one").read_text())
    note.write_text("two\n", encoding="utf-8")
    second = json.loads(_capture(repo, benchmark_root, run_id="run-two").read_text())

    assert (
        first["ny"]["worktree_evidence_sha256"]
        != second["ny"]["worktree_evidence_sha256"]
    )


def test_clean_worktree_keeps_zero_length_tracked_diff_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(_capture(repo, benchmark_root, run_id="clean").read_text())

    for worktree in (manifest["ny"], manifest["benchmark"]):
        assert worktree["clean"] is True
        assert worktree["tracked_diff_format"] == "ny_tracked_worktree_evidence_v2"
        assert worktree["tracked_diff_bytes"] == 0
        assert worktree["tracked_diff_sha256"] == provenance._sha256(b"")
        assert worktree["tracked_worktree_paths"] == []


def test_tracked_worktree_evidence_hashes_files_symlinks_and_large_deletions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    benchmark_repo = benchmark_root.parent
    archive = benchmark_root / "archive.bin"
    config = benchmark_root / "config.txt"
    link = benchmark_root / "current"
    archive.write_bytes(b"x" * (2 * 1024 * 1024))
    config.write_text("committed\n", encoding="utf-8")
    link.symlink_to("old-target")
    _run("git", "add", ".", cwd=benchmark_repo)
    _run("git", "commit", "-qm", "add tracked states", cwd=benchmark_repo)

    archive.unlink()
    config.write_text("staged\n", encoding="utf-8")
    _run("git", "add", "benchmarks/config.txt", cwd=benchmark_repo)
    config.write_text("worktree\n", encoding="utf-8")
    config.chmod(0o755)
    link.unlink()
    link.symlink_to("new-target")

    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="tracked-states").read_text()
    )
    benchmark = manifest["benchmark"]
    entries = {entry["path"]: entry for entry in benchmark["tracked_worktree_paths"]}

    assert benchmark["clean"] is False
    assert benchmark["tracked_diff_format"] == "ny_tracked_worktree_evidence_v2"
    assert benchmark["tracked_diff_bytes"] < 16 * 1024
    assert entries["benchmarks/archive.bin"] == {
        "kind": "missing",
        "path": "benchmarks/archive.bin",
    }
    assert entries["benchmarks/config.txt"] == {
        "kind": "file",
        "mode": 0o755,
        "path": "benchmarks/config.txt",
        "sha256": provenance._sha256(b"worktree\n"),
        "size_bytes": len(b"worktree\n"),
    }
    target = os.fsencode("new-target")
    assert entries["benchmarks/current"] == {
        "kind": "symlink",
        "mode": 0o777,
        "path": "benchmarks/current",
        "sha256": provenance._sha256(target),
        "size_bytes": len(target),
    }


def test_tracked_worktree_path_capture_rejects_parent_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    with pytest.raises(provenance.ProvenanceError, match="unsafe tracked worktree"):
        provenance._tracked_path_state(repo, "../outside")


def test_tracked_worktree_capture_forces_filemode_detection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    script = repo / "script.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    _run("git", "add", "script.sh", cwd=repo)
    _run("git", "commit", "-qm", "script", cwd=repo)
    _run("git", "config", "core.fileMode", "false", cwd=repo)
    script.chmod(0o755)

    evidence, entries = provenance._tracked_worktree_evidence(repo)

    assert evidence
    assert entries == [
        {
            "kind": "file",
            "mode": 0o755,
            "path": "script.sh",
            "sha256": provenance._sha256(b"#!/bin/sh\n"),
            "size_bytes": len(b"#!/bin/sh\n"),
        }
    ]


def test_staged_intermediate_change_alters_evidence_with_fixed_worktree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    path = repo / "state.txt"
    path.write_text("committed\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    _run("git", "commit", "-qm", "state", cwd=repo)

    path.write_text("staged-a\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    path.write_text("final-b\n", encoding="utf-8")
    first, first_entries = provenance._tracked_worktree_evidence(repo)

    path.write_text("staged-a2\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    path.write_text("final-b\n", encoding="utf-8")
    second, second_entries = provenance._tracked_worktree_evidence(repo)

    assert first_entries == second_entries
    assert first != second


def test_staged_add_delete_and_rename_bind_final_states(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    renamed = repo / "renamed.txt"
    deleted = repo / "deleted.txt"
    renamed.write_text("rename-content\n", encoding="utf-8")
    deleted.write_text("delete-content\n", encoding="utf-8")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "base", cwd=repo)
    renamed.rename(repo / "new-name.txt")
    deleted.unlink()
    (repo / "added.txt").write_text("added-content\n", encoding="utf-8")
    _run("git", "add", "-A", cwd=repo)

    evidence, entries = provenance._tracked_worktree_evidence(repo)
    by_path = {entry["path"]: entry for entry in entries}

    assert evidence
    assert set(by_path) == {
        "added.txt",
        "deleted.txt",
        "new-name.txt",
        "renamed.txt",
    }
    assert by_path["deleted.txt"]["kind"] == "missing"
    assert by_path["renamed.txt"]["kind"] == "missing"
    assert by_path["added.txt"]["sha256"] == provenance._sha256(b"added-content\n")
    assert by_path["new-name.txt"]["sha256"] == provenance._sha256(b"rename-content\n")


@pytest.mark.parametrize(
    "index_flag",
    ["--assume-unchanged", "--skip-worktree"],
)
def test_hidden_index_flags_fail_closed(
    tmp_path: Path,
    index_flag: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    path = repo / "hidden.txt"
    path.write_text("committed\n", encoding="utf-8")
    _run("git", "add", "hidden.txt", cwd=repo)
    _run("git", "commit", "-qm", "hidden", cwd=repo)
    _run("git", "update-index", index_flag, "hidden.txt", cwd=repo)
    path.write_text("hidden mutation\n", encoding="utf-8")

    with pytest.raises(provenance.ProvenanceError, match="unsupported Git"):
        provenance._tracked_worktree_evidence(repo)


def test_fsmonitor_valid_index_flag_parser_fails_closed() -> None:
    with pytest.raises(provenance.ProvenanceError, match="fsmonitor"):
        provenance._parse_index_flags(b"h tracked.txt\0", label="fsmonitor")


def test_git_environment_cannot_redirect_worktree_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    alternate = tmp_path / "alternate"
    for repo, content in ((target, "target\n"), (alternate, "alternate\n")):
        repo.mkdir()
        _init_git_repo(repo)
        (repo / "state.txt").write_text(content, encoding="utf-8")
        _run("git", "add", "state.txt", cwd=repo)
        _run("git", "commit", "-qm", "state", cwd=repo)
    (target / "state.txt").write_text("target-dirty\n", encoding="utf-8")
    monkeypatch.setenv("GIT_DIR", str(alternate / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(alternate))
    monkeypatch.setenv("GIT_INDEX_FILE", str(alternate / ".git" / "index"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(alternate))

    evidence, entries = provenance._tracked_worktree_evidence(target)

    assert evidence
    assert entries == [
        {
            "kind": "file",
            "mode": (target / "state.txt").stat().st_mode & 0o7777,
            "path": "state.txt",
            "sha256": provenance._sha256(b"target-dirty\n"),
            "size_bytes": len(b"target-dirty\n"),
        }
    ]


def test_replace_ref_cannot_hide_staged_and_worktree_change(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    path = repo / "state.txt"
    path.write_text("original\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    _run("git", "commit", "-qm", "original", cwd=repo)
    original = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path.write_text("replacement\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    _run("git", "commit", "-qm", "replacement", cwd=repo)
    replacement = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch_ref = subprocess.run(
        ["git", "symbolic-ref", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run("git", "update-ref", branch_ref, original, cwd=repo)
    _run("git", "replace", original, replacement, cwd=repo)

    evidence, entries = provenance._tracked_worktree_evidence(repo)

    assert evidence
    assert entries == [
        {
            "kind": "file",
            "mode": path.stat().st_mode & 0o7777,
            "path": "state.txt",
            "sha256": provenance._sha256(b"replacement\n"),
            "size_bytes": len(b"replacement\n"),
        }
    ]


@pytest.mark.parametrize(
    ("stage_record", "message"),
    [
        (
            b"100644 " + b"a" * 40 + b" 2\tconflict.txt\0",
            "unmerged",
        ),
        (
            b"160000 " + b"a" * 40 + b" 0\tsubmodule\0",
            "Gitlink",
        ),
    ],
)
def test_unmerged_and_gitlink_index_entries_fail_closed(
    stage_record: bytes,
    message: str,
) -> None:
    with pytest.raises(provenance.ProvenanceError, match=message):
        provenance._parse_index_stage(stage_record, object_format="sha1")


def test_tracked_special_file_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    fifo = repo / "tracked.fifo"
    os.mkfifo(fifo)

    with pytest.raises(provenance.ProvenanceError, match="special entry"):
        provenance._tracked_path_state(repo, "tracked.fifo")


def test_ancestor_symlink_escape_is_never_followed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    _init_git_repo(repo)
    tracked = repo / "tracked"
    tracked.mkdir()
    (tracked / "secret").write_text("inside\n", encoding="utf-8")
    (outside / "secret").write_text("must-not-be-read\n", encoding="utf-8")
    tracked.rename(repo / "tracked-original")
    (repo / "tracked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(provenance.ProvenanceError, match="safely open"):
        provenance._tracked_path_state(repo, "tracked/secret")


def test_final_file_to_symlink_swap_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.write_text("must-not-be-read\n", encoding="utf-8")
    _init_git_repo(repo)
    path = repo / "state"
    path.write_text("inside\n", encoding="utf-8")
    real_open = provenance.os.open
    swapped = False

    def swap_before_open(
        candidate: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if candidate == b"state" and dir_fd is not None and not swapped:
            swapped = True
            path.rename(repo / "state-original")
            path.symlink_to(outside)
        return real_open(candidate, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(provenance.os, "open", swap_before_open)
    with pytest.raises(provenance.ProvenanceError, match="safely open"):
        provenance._tracked_path_state(repo, "state")


def _assert_byte_paths_are_deterministic(
    tmp_path: Path,
    raw_names: list[bytes],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, name in enumerate(raw_names):
            try:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                    dir_fd=root_fd,
                )
            except OSError as error:
                if error.errno == errno.EILSEQ:
                    raise AssertionError(
                        "the selected filesystem cannot exercise the required "
                        "non-UTF-8 provenance contract"
                    ) from error
                raise
            try:
                os.write(descriptor, f"value-{index}\n".encode())
            finally:
                os.close(descriptor)
    finally:
        os.close(root_fd)

    first = provenance._tracked_path_states(repo, raw_names)
    second = provenance._tracked_path_states(repo, list(reversed(raw_names)))

    assert first == second
    assert [os.fsencode(entry["path"]) for entry in first] == sorted(raw_names)


def test_byte_paths_are_deterministic_and_not_normalized(tmp_path: Path) -> None:
    _assert_byte_paths_are_deterministic(
        tmp_path,
        [b"-leading", b"line\nbreak"],
    )


def external_non_utf8_byte_paths_are_deterministic_and_not_normalized(
    tmp_path: Path,
) -> None:
    _assert_byte_paths_are_deterministic(tmp_path, [b"nonutf8-\xff"])


def test_many_distinct_parents_keep_open_fds_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    paths: list[bytes] = []
    for index in range(512):
        relative = f"dir-{index:04d}"
        (repo / relative).mkdir()
        paths.append(os.fsencode(f"{relative}/missing"))

    real_open = provenance.os.open
    real_close = provenance.os.close
    live = 0
    max_live = 0

    def counted_open(*args: object, **kwargs: object) -> int:
        nonlocal live, max_live
        descriptor = real_open(*args, **kwargs)
        live += 1
        max_live = max(max_live, live)
        return descriptor

    def counted_close(descriptor: int) -> None:
        nonlocal live
        live -= 1
        real_close(descriptor)

    monkeypatch.setattr(provenance.os, "open", counted_open)
    monkeypatch.setattr(provenance.os, "close", counted_close)
    entries = provenance._tracked_path_states(repo, paths)

    assert len(entries) == len(paths)
    assert all(entry["kind"] == "missing" for entry in entries)
    assert max_live <= 2
    assert live == 0


def test_twenty_four_thousand_deletions_use_bounded_raw_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    paths = [f"archive/{index:05d}.gz".encode() for index in range(24_000)]
    oid = b"a" * 40
    zero = b"0" * 40
    raw = b"".join(
        b":100644 000000 " + oid + b" " + zero + b" D\0" + path + b"\0"
        for path in paths
    )
    index_stage = b"".join(b"100644 " + oid + b" 0\t" + path + b"\0" for path in paths)
    flags = b"".join(b"H " + path + b"\0" for path in paths)
    snapshot = {
        "head": oid,
        "object_format": "sha1",
        "index_stage": index_stage,
        "index_paths": paths,
        "flags_v": flags,
        "flags_f": flags,
    }
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        provenance,
        "_tracked_index_snapshot",
        lambda _repo: snapshot,
    )

    def fake_git(_repo: Path, *args: str, check: bool = True) -> bytes:
        del check
        commands.append(args)
        return raw if args[0] == "diff-files" else b""

    monkeypatch.setattr(provenance, "_git_evidence", fake_git)
    monkeypatch.setattr(
        provenance,
        "_tracked_path_states",
        lambda _repo, raw_paths: [
            {"path": os.fsdecode(path), "kind": "missing"}
            for path in sorted(set(raw_paths))
        ],
    )

    evidence, entries = provenance._tracked_worktree_evidence(repo)

    assert len(entries) == 24_000
    assert len(evidence) < 4 * 1024 * 1024
    assert all("--binary" not in command for command in commands)


def test_index_snapshot_change_during_capture_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    first = {
        "top_level": os.fsencode(repo),
        "head": b"a" * 40,
        "object_format": "sha1",
        "index_stage": b"",
        "index_paths": [],
        "flags_v": b"",
        "flags_f": b"",
    }
    second = {**first, "head": b"b" * 40}
    snapshots = iter((first, second))
    monkeypatch.setattr(
        provenance,
        "_tracked_index_snapshot",
        lambda _repo: next(snapshots),
    )
    monkeypatch.setattr(provenance, "_git_evidence", lambda *_args, **_kwargs: b"")

    with pytest.raises(provenance.ProvenanceError, match="HEAD or index changed"):
        provenance._tracked_worktree_evidence(repo)


def test_start_and_completion_records_are_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="immutable-run")

    with pytest.raises(FileExistsError, match="immutable evidence"):
        _capture(repo, benchmark_root, run_id="immutable-run")

    completion = provenance.create_completion(start_manifest=start, exit_status=143)
    record = json.loads(completion.read_text(encoding="utf-8"))
    assert record["exit_status"] == 143
    assert record["completed_successfully"] is False
    assert record["start_manifest_sha256"] == provenance._sha256(start.read_bytes())
    assert record["integrity"]["status"] == "valid"
    assert record["integrity"]["violations"] == []
    assert record["host_state"] == HOST_STATE_FIXTURE
    with pytest.raises(FileExistsError, match="immutable evidence"):
        provenance.create_completion(start_manifest=start, exit_status=0)


def test_completion_rejects_containment_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="containment-drift")
    monkeypatch.setattr(
        provenance,
        "_capture_measurement_containment",
        lambda: {
            "schema": "ny_measurement_containment_v1",
            "fixture": "drifted",
        },
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=143)
    record = json.loads(completion.read_text(encoding="utf-8"))
    violations = {item["code"] for item in record["integrity"]["violations"]}

    assert record["integrity"]["status"] == "invalid"
    assert record["integrity"]["checks"]["containment"]["status"] == "invalid"
    assert "containment_identity_mismatch" in violations


@pytest.mark.parametrize(
    "key",
    [
        "NY_CROWN_CUT_SEGMENT",
        "NY_IMB_OBJ",
        "NY_IMB_REGION_K",
        "NY_IMB_REPLAY_ONLY_LEAF",
        "NY_WARMUP_ITERS",
    ],
)
def test_typed_numeric_environment_rejects_non_numeric_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "not-a-number")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=key):
        _capture(repo, benchmark_root, run_id=f"invalid-{key.lower()}")


def test_positive_usize_environment_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in (
        "NY_BUILD_MIN_FREE_KIB",
        "NY_CUDA_WIDE_MAX_BYTES",
        "NY_GPU_LOCK_WAIT_SECS",
        "NY_GPU_VMEM_LIMIT_KIB",
        "NY_MO_GPU_CHUNK",
    ):
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NY_BUILD_MIN_FREE_KIB", "33554432", 33554432),
        ("NY_MO_GPU_CHUNK", "128", 128),
        ("NY_MO_GPU_CHUNK", "00128", 128),
        ("NY_CUDA_WIDE_MAX_BYTES", "2147483648", 2147483648),
        ("NY_CUDA_WIDE_MAX_BYTES", "0000000001", 1),
        ("NY_GPU_LOCK_WAIT_SECS", "03600", 3600),
        ("NY_GPU_VMEM_LIMIT_KIB", "167772160", 167772160),
    ],
)
def test_positive_usize_environment_preserves_raw_and_captures_typed_value(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "positive_integer",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "0", "+64", "-1", " 64", "64 ", "64.0"])
def test_mo_gpu_chunk_invalid_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_GPU_CHUNK", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MO_GPU_CHUNK"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    ["NY_MO_GPU_CHUNK", "NY_CUDA_WIDE_MAX_BYTES"],
)
def test_positive_usize_environment_accepts_native_maximum(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    usize_max = (sys.maxsize * 2) + 1
    monkeypatch.setenv(key, str(usize_max))

    environment = provenance._capture_environment()

    assert environment["typed_values"][key] == {
        "type": "positive_integer",
        "value": usize_max,
    }


@pytest.mark.parametrize(
    "key",
    ["NY_MO_GPU_CHUNK", "NY_CUDA_WIDE_MAX_BYTES"],
)
def test_positive_usize_environment_rejects_zero_and_overflow(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "0")
    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()

    usize_overflow = (sys.maxsize * 2) + 2
    monkeypatch.setenv(key, str(usize_overflow))
    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_mo_gpu_chunk_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_MO_GPU_CHUNK_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "64")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_guard_path_environment_is_captured_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    paths = {
        "NY_BUILD_DISK_PATH": "/private/build-target",
        "NY_GPU_LOCK_PATH": "/private/runtime/ny.lock",
    }
    for key, value in paths.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    for key, value in paths.items():
        assert environment["values"][key] == value
        assert key not in environment["typed_values"]


def test_noncuda_measure_override_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ALLOW_NONCUDA_MEASURE", "1")

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ALLOW_NONCUDA_MEASURE"] == "1"
    assert environment["typed_values"]["NY_ALLOW_NONCUDA_MEASURE"] == {
        "type": "boolean",
        "value": True,
    }


@pytest.mark.parametrize("raw", ["", "true", "yes", "2"])
def test_noncuda_measure_override_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ALLOW_NONCUDA_MEASURE", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ALLOW_NONCUDA_MEASURE"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_attack_point_fast_kernels_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ATTACK_POINT_FAST_KERNELS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ATTACK_POINT_FAST_KERNELS"] == raw
    assert environment["typed_values"]["NY_ATTACK_POINT_FAST_KERNELS"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_attack_point_fast_kernels_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ATTACK_POINT_FAST_KERNELS", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ATTACK_POINT_FAST_KERNELS"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_softmax_objective_envelope_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_SOFTMAX_OBJECTIVE_ENVELOPE", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_SOFTMAX_OBJECTIVE_ENVELOPE"] == raw
    assert environment["typed_values"]["NY_SOFTMAX_OBJECTIVE_ENVELOPE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_softmax_objective_envelope_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_SOFTMAX_OBJECTIVE_ENVELOPE", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_SOFTMAX_OBJECTIVE_ENVELOPE"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_root_alpha_phase_checkpoint_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_ALPHA_PHASE_CHECKPOINT", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ROOT_ALPHA_PHASE_CHECKPOINT"] == raw
    assert environment["typed_values"]["NY_ROOT_ALPHA_PHASE_CHECKPOINT"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_root_alpha_phase_checkpoint_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_ALPHA_PHASE_CHECKPOINT", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ROOT_ALPHA_PHASE_CHECKPOINT"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cgan_truncated_cache_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_SERVE_TRUNCATED_CACHE", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CROWN_SERVE_TRUNCATED_CACHE"] == raw
    assert environment["typed_values"]["NY_CROWN_SERVE_TRUNCATED_CACHE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "true", "yes", "2"])
def test_cgan_truncated_cache_gate_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_SERVE_TRUNCATED_CACHE", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_CROWN_SERVE_TRUNCATED_CACHE",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cut_crown_resident_shadow_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUT_CROWN_RESIDENT_SHADOW", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CUT_CROWN_RESIDENT_SHADOW"] == raw
    assert environment["typed_values"]["NY_CUT_CROWN_RESIDENT_SHADOW"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "true", "yes", "2"])
def test_cut_crown_resident_shadow_gate_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUT_CROWN_RESIDENT_SHADOW", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_CUT_CROWN_RESIDENT_SHADOW",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cut_crown_m2_projected_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUT_CROWN_M2_PROJECTED", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CUT_CROWN_M2_PROJECTED"] == raw
    assert environment["typed_values"]["NY_CUT_CROWN_M2_PROJECTED"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "true", "yes", "2"])
def test_cut_crown_m2_projected_gate_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUT_CROWN_M2_PROJECTED", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_CUT_CROWN_M2_PROJECTED",
    ):
        provenance._capture_environment()


def test_mip_dump_capture_path_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_MIP_DUMP" not in environment["values"]
    assert "NY_MIP_DUMP" not in environment["typed_values"]


def test_mip_dump_capture_path_is_exactly_provenanced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    dump_path = tmp_path / "shared-decision-corpus"
    monkeypatch.setenv("NY_MIP_DUMP", str(dump_path))

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_DUMP"] == str(dump_path)
    assert environment["typed_values"]["NY_MIP_DUMP"] == {
        "type": "absolute_path",
        "value": str(dump_path),
    }


@pytest.mark.parametrize("raw", ["", "relative/mip-dump"])
def test_mip_dump_capture_path_rejects_non_absolute_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_DUMP", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MIP_DUMP"):
        provenance._capture_environment()


def test_mip_dump_unknown_neighbor_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_MIP_DUMP_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "/tmp/unreviewed")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_kfsb_winner_oracle_environment_is_captured_and_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    raw_values = {
        "NY_BRANCH_KFSB_CHILDSIM": "1",
        "NY_KFSB_LAYER_QUOTA": "0",
        "NY_MO_ADAPTIVE_DEPTH_SELECT": "1",
        "NY_MO_ADAPTIVE_DEPTH_SHADOW": "1",
        "NY_MO_KFSB": "1",
        "NY_MO_KFSB_CACHED_LA": "1",
        "NY_MO_KFSB_CERT_REUSE": "1",
        "NY_MO_KFSB_CHUNK": "00064",
        "NY_MO_KFSB_F64_SHADOW": "1",
        "NY_MO_KFSB_K": "07",
        "NY_MO_KFSB_PROBE": "1",
        "NY_MO_KFSB_REDUCE": "max",
        "NY_MO_KFSB_WINNER_PROBE": "1",
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS": "03",
    }
    for key, value in raw_values.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    for key, value in raw_values.items():
        assert environment["values"][key] == value
    assert environment["typed_values"] == {
        "NY_MO_ADAPTIVE_DEPTH_SELECT": {"type": "boolean", "value": True},
        "NY_MO_ADAPTIVE_DEPTH_SHADOW": {"type": "boolean", "value": True},
        "NY_MO_KFSB_CERT_REUSE": {"type": "boolean", "value": True},
        "NY_MO_KFSB_F64_SHADOW": {"type": "boolean", "value": True},
        "NY_MO_KFSB_CHUNK": {"type": "positive_integer", "value": 64},
        "NY_MO_KFSB_K": {"type": "nonnegative_integer", "value": 7},
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS": {
            "type": "positive_integer",
            "value": 3,
        },
    }


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("NY_MO_KFSB_K", "-1"),
        ("NY_MO_KFSB_CHUNK", "0"),
        ("NY_MO_KFSB_WINNER_PROBE_DOMAINS", "not-a-number"),
    ],
)
def test_kfsb_winner_oracle_invalid_numeric_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_kfsb_f64_shadow_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_KFSB_F64_SHADOW", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MO_KFSB_F64_SHADOW"] == raw
    assert environment["typed_values"]["NY_MO_KFSB_F64_SHADOW"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_kfsb_f64_shadow_gate_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_KFSB_F64_SHADOW", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MO_KFSB_F64_SHADOW"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_kfsb_cert_reuse_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_KFSB_CERT_REUSE", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MO_KFSB_CERT_REUSE"] == raw
    assert environment["typed_values"]["NY_MO_KFSB_CERT_REUSE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_kfsb_cert_reuse_gate_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_KFSB_CERT_REUSE", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MO_KFSB_CERT_REUSE"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    [
        "NY_MO_ADAPTIVE_DEPTH_SHADOW",
        "NY_MO_ADAPTIVE_DEPTH_SELECT",
        "NY_MO_ADAPTIVE_DEPTH_COMMIT",
    ],
)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_adaptive_depth_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize(
    "key",
    [
        "NY_MO_ADAPTIVE_DEPTH_SHADOW",
        "NY_MO_ADAPTIVE_DEPTH_SELECT",
        "NY_MO_ADAPTIVE_DEPTH_COMMIT",
    ],
)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_adaptive_depth_gate_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_rel_bab_deadline_multiplier_is_captured_raw_and_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_REL_BAB_DEADLINE_MULT", "+01.400e0")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="rel-bab-mult-raw").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_REL_BAB_DEADLINE_MULT"] == "+01.400e0"
    assert environment["typed_values"]["NY_REL_BAB_DEADLINE_MULT"] == {
        "type": "bounded_float",
        "value": 1.4,
        "minimum": 1.0,
        "maximum": 10.0,
    }


@pytest.mark.parametrize(
    "raw", ["not-a-number", "NaN", "inf", "0.999", "10.001", " 1.4 "]
)
def test_rel_bab_deadline_multiplier_invalid_value_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_REL_BAB_DEADLINE_MULT", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="NY_REL_BAB_DEADLINE_MULT"):
        _capture(repo, benchmark_root, run_id="rel-bab-mult-invalid")

    assert not (
        repo / "reports/measured/artifacts/runs/rel-bab-mult-invalid/start.json"
    ).exists()


def test_rel_bab_deadline_multiplier_unknown_neighbor_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_REL_BAB_DEADLINE_MULT_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1.4")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        _capture(repo, benchmark_root, run_id="rel-bab-mult-unknown")

    assert not (
        repo / "reports/measured/artifacts/runs/rel-bab-mult-unknown/start.json"
    ).exists()


def test_margin_row_reserve_max_fraction_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_MARGIN_ROW_RESERVE_MAX_FRAC" not in environment["values"]
    assert "NY_MARGIN_ROW_RESERVE_MAX_FRAC" not in environment["typed_values"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.25", 0.25),
        ("0.5", 0.5),
        ("0.0001", 0.0001),
        ("0.9999999999999999", 0.9999999999999999),
    ],
)
def test_margin_row_reserve_max_fraction_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: float,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_RESERVE_MAX_FRAC", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MARGIN_ROW_RESERVE_MAX_FRAC"] == raw
    assert environment["typed_values"]["NY_MARGIN_ROW_RESERVE_MAX_FRAC"] == {
        "type": "open_unit_decimal_fraction",
        "value": expected,
        "minimum_exclusive": 0.0,
        "maximum_exclusive": 1.0,
    }


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0",
        "0.0",
        "1",
        "1.0",
        "-0.25",
        "+0.25",
        ".25",
        "00.25",
        "0.250",
        "2.5e-1",
        "NaN",
        "nan",
        "inf",
        "Infinity",
        " 0.25 ",
        # Both Python and Rust's IEEE-754 f64 parser round this to 1.0, which
        # the runtime declines even though the source decimal is below one.
        "0.99999999999999999",
    ],
)
def test_margin_row_reserve_max_fraction_rejects_noncanonical_or_runtime_invalid(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_RESERVE_MAX_FRAC", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_MARGIN_ROW_RESERVE_MAX_FRAC"
    ):
        provenance._capture_environment()


def test_margin_row_reserve_max_fraction_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_MARGIN_ROW_RESERVE_MAX_FRAC_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "0.25")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_endgame_grace_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_ENDGAME_GRACE_SECS" not in environment["values"]
    assert "NY_ENDGAME_GRACE_SECS" not in environment["typed_values"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0.0), ("12", 12.0), ("+01.25e1", 12.5), ("30.0", 30.0)],
)
def test_endgame_grace_is_captured_raw_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: float,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ENDGAME_GRACE_SECS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ENDGAME_GRACE_SECS"] == raw
    assert environment["typed_values"]["NY_ENDGAME_GRACE_SECS"] == {
        "type": "bounded_float",
        "value": expected,
        "minimum": 0.0,
        "maximum": 30.0,
    }


@pytest.mark.parametrize(
    "raw",
    ["", "not-a-number", "NaN", "inf", "-0.001", "30.001", " 12 ", "12s"],
)
def test_endgame_grace_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ENDGAME_GRACE_SECS", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ENDGAME_GRACE_SECS"):
        provenance._capture_environment()


def test_relational_mip_controls_are_captured_as_raw_launch_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    controls = {
        "NY_REL_DIFF_COUPLING": "1",
        "NY_REL_JOINT_RELU_CUTS": "1",
        "NY_REL_JOINT_RELU_CUTS_SUM": "0",
        "NY_REL_WHOLE_MIP_OBBT": "1",
        "NY_REL_WHOLE_MIP_OBBT_CHUNK": "08",
        "NY_REL_WHOLE_MIP_OBBT_COND": "1",
        "NY_REL_WHOLE_MIP_OBBT_COND_FRAC": "0.45",
        "NY_REL_WHOLE_MIP_OBBT_MAXN": "192",
        "NY_REL_WHOLE_MIP_OBBT_OUTER": "2",
        "NY_REL_WHOLE_MIP_OBBT_ROUNDS": "3",
        "NY_REL_WHOLE_MIP_OBBT_S": "12.5",
        "NY_REL_WHOLE_MIP_OBBT_WIDTH": "1000.0",
    }
    for key, value in controls.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    assert {key: environment["values"][key] for key in controls} == controls
    assert not controls.keys() & environment["typed_values"].keys()


@pytest.mark.parametrize(
    "key",
    ["NY_JOINT_MEAS_NCOLS", "NY_JOINT_MEAS_PERSOLVE_S", "NY_JOINT_MEAS_DEADLINE_S"],
)
def test_ignored_joint_cut_test_knobs_remain_outside_solver_measurement_authority(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "1")

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", False), ("0", False), ("1", True)],
)
def test_gpu_available_is_captured_raw_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("GPU_AVAILABLE", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="gpu-available-raw").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["GPU_AVAILABLE"] == raw
    assert environment["typed_values"]["GPU_AVAILABLE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["2", "00", "yes", "-1", " 1"])
def test_gpu_available_non_boolean_value_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("GPU_AVAILABLE", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="GPU_AVAILABLE"):
        _capture(repo, benchmark_root, run_id="gpu-available-invalid")

    assert not (
        repo / "reports/measured/artifacts/runs/gpu-available-invalid/start.json"
    ).exists()


def test_cuda_dgemm_triplet_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_CUDA_DGEMM_TRIPLET" not in environment["values"]
    assert "NY_CUDA_DGEMM_TRIPLET" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cuda_dgemm_triplet_is_captured_raw_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUDA_DGEMM_TRIPLET", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id=f"cuda-dgemm-triplet-{raw}").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_CUDA_DGEMM_TRIPLET"] == raw
    assert environment["typed_values"]["NY_CUDA_DGEMM_TRIPLET"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_cuda_dgemm_triplet_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUDA_DGEMM_TRIPLET", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_CUDA_DGEMM_TRIPLET"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "values"),
    [
        (
            "NY_CUDA_GEMM_TRANSPORT",
            [
                "auto",
                "direct-host-page-tables",
                "unified-memory",
                "explicit-device-copy",
            ],
        ),
        ("NY_GPU_DENORM_PRESERVE", ["auto", "0", "1"]),
        ("NY_WGPU_CROWN", ["auto", "0", "1"]),
    ],
)
def test_gpu_profile_enums_are_captured_exactly(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    values: list[str],
) -> None:
    for raw in values:
        _clear_ny_environment(monkeypatch)
        monkeypatch.setenv(key, raw)
        environment = provenance._capture_environment()
        assert environment["values"][key] == raw
        assert environment["typed_values"][key] == {
            "type": "enum",
            "value": raw,
            "allowed_values": sorted(values),
        }


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("NY_CUDA_GEMM_TRANSPORT", "explicit"),
        ("NY_GPU_DENORM_PRESERVE", "true"),
        ("NY_WGPU_CROWN", "true"),
    ],
)
def test_gpu_profile_enums_reject_unreviewed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)
    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cuda_crown_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUDA_CROWN", raw)
    environment = provenance._capture_environment()
    assert environment["typed_values"]["NY_CUDA_CROWN"] == {
        "type": "boolean",
        "value": expected,
    }


def test_safenlp_short_grace_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert environment["allowlist_schema"] == "ny_measurement_environment_v1"
    assert "NY_SAFENLP_SHORT_GRACE" not in environment["values"]
    assert "NY_SAFENLP_SHORT_GRACE" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_safenlp_short_grace_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_SAFENLP_SHORT_GRACE", raw)

    environment = provenance._capture_environment()

    assert environment["allowlist_schema"] == "ny_measurement_environment_v1"
    assert environment["values"]["NY_SAFENLP_SHORT_GRACE"] == raw
    assert environment["typed_values"]["NY_SAFENLP_SHORT_GRACE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_safenlp_short_grace_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_SAFENLP_SHORT_GRACE", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_SAFENLP_SHORT_GRACE"):
        provenance._capture_environment()


def test_convtranspose_sound_f64_gpu_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_CONVTRANSPOSE_SOUND_F64_GPU" not in environment["values"]
    assert "NY_CONVTRANSPOSE_SOUND_F64_GPU" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_convtranspose_sound_f64_gpu_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CONVTRANSPOSE_SOUND_F64_GPU", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CONVTRANSPOSE_SOUND_F64_GPU"] == raw
    assert environment["typed_values"]["NY_CONVTRANSPOSE_SOUND_F64_GPU"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_convtranspose_sound_f64_gpu_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CONVTRANSPOSE_SOUND_F64_GPU", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_CONVTRANSPOSE_SOUND_F64_GPU"
    ):
        provenance._capture_environment()


def test_root_post_c_survivor_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_ROOT_POST_C_SURVIVOR" not in environment["values"]
    assert "NY_ROOT_POST_C_SURVIVOR" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_root_post_c_survivor_explicit_override_is_captured_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_POST_C_SURVIVOR", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id=f"root-post-c-survivor-{raw}").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_ROOT_POST_C_SURVIVOR"] == raw
    assert environment["typed_values"]["NY_ROOT_POST_C_SURVIVOR"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_root_post_c_survivor_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_POST_C_SURVIVOR", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ROOT_POST_C_SURVIVOR"):
        provenance._capture_environment()


def test_root_post_c_survivor_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_ROOT_POST_C_SURVIVOR_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


@pytest.mark.parametrize("key", ["NY_ALPHA_FINAL_BOUND_ONLY", "NY_BN_FOLD_EXT"])
def test_alpha_final_and_bn_fold_experiments_are_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert key not in environment["values"]
    assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NY_ALPHA_FINAL_BOUND_ONLY", "0", False),
        ("NY_ALPHA_FINAL_BOUND_ONLY", "1", True),
        ("NY_BN_FOLD_EXT", "0", False),
        ("NY_BN_FOLD_EXT", "1", True),
    ],
)
def test_alpha_final_and_bn_fold_explicit_overrides_are_immutably_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    run_id = f"{key.lower().replace('_', '-')}-{raw}"

    manifest_path = _capture(repo, benchmark_root, run_id=run_id)
    manifest_bytes = manifest_path.read_bytes()
    environment = json.loads(manifest_bytes)["environment"]
    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }

    monkeypatch.setenv(key, "1" if raw == "0" else "0")
    with pytest.raises(FileExistsError, match="immutable evidence"):
        _capture(repo, benchmark_root, run_id=run_id)
    assert manifest_path.read_bytes() == manifest_bytes


@pytest.mark.parametrize("key", ["NY_ALPHA_FINAL_BOUND_ONLY", "NY_BN_FOLD_EXT"])
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_alpha_final_and_bn_fold_malformed_values_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize("key", ["NY_ALPHA_FINAL_BOUND_ONLY", "NY_BN_FOLD_EXT"])
def test_alpha_final_and_bn_fold_unknown_neighbors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = f"{key}_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_cgan_generator_branch_controls_are_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in CGAN_GENERATOR_BRANCH_ENV:
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


@pytest.mark.parametrize("key", CGAN_GENERATOR_BRANCH_BOOLEAN_ENV)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cgan_generator_branch_boolean_controls_are_captured_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


def test_cgan_generator_branch_scope_is_sealed_with_raw_node_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    raw_values = {
        "NY_BRANCH_STEM": "1",
        "NY_BRANCH_STEM_K": "0008",
        "NY_BRANCH_STEM_NODES": " Relu_6,Relu_9, Relu_12 ",
        "NY_BRANCH_STEM_PROBE": "0",
        "NY_BRANCH_TRACE": "1",
        "NY_MO_BAB_TRACE": "1",
        "NY_UNSTABLE_COUNT": "0",
    }
    for key, value in raw_values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="cgan-generator-branch").read_text()
    )
    environment = manifest["environment"]

    assert {
        key: environment["values"][key] for key in CGAN_GENERATOR_BRANCH_ENV
    } == raw_values
    assert environment["typed_values"] == {
        "NY_BRANCH_STEM": {"type": "boolean", "value": True},
        "NY_BRANCH_STEM_K": {"type": "positive_integer", "value": 8},
        "NY_BRANCH_STEM_PROBE": {"type": "boolean", "value": False},
        "NY_BRANCH_TRACE": {"type": "boolean", "value": True},
        "NY_MO_BAB_TRACE": {"type": "boolean", "value": True},
        "NY_UNSTABLE_COUNT": {"type": "boolean", "value": False},
    }


@pytest.mark.parametrize("key", CGAN_GENERATOR_BRANCH_BOOLEAN_ENV)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_cgan_generator_branch_boolean_controls_reject_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0",
        "+8",
        "-1",
        " 8",
        "8 ",
        "8.0",
        str((sys.maxsize * 2) + 2),
    ],
)
def test_cgan_generator_branch_depth_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_BRANCH_STEM_K", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_BRANCH_STEM_K"):
        provenance._capture_environment()


@pytest.mark.parametrize("key", CGAN_GENERATOR_BRANCH_ENV)
def test_cgan_generator_branch_unknown_neighbors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = f"{key}_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    [
        "NY_AY_BRANCH_HINTS",
        "NY_AY_MARGIN_REFRAME",
        "NY_AY_MILP_TALL_FLIP_CAP",
        "NY_AY_OBJECTIVE_FIRST_SAT",
        "NY_MIP_CERTIFIED_SHARED_TREE",
        "NY_MIP_SAFENLP_DIRECT_FIRST",
        "NY_MIP_SAFENLP_SHARED_PREFIX",
        "NY_MIP_SAFENLP_TARGET_FSB_PREFIX",
        "NY_MIP_SERIAL",
    ],
)
def test_ay_milp_canary_is_captured_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "1")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest_path = _capture(repo, benchmark_root, run_id=key.lower())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["environment"]["values"][key] == "1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", False), ("0", False), ("1", True)],
)
def test_ay_branch_hint_canary_is_sealed_as_a_strict_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_BRANCH_HINTS", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="ay-branch-hints-typed").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_AY_BRANCH_HINTS"] == raw
    assert environment["typed_values"]["NY_AY_BRANCH_HINTS"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["2", "yes", "true", "-1", " 1"])
def test_ay_branch_hint_canary_malformed_measurement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_BRANCH_HINTS", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="NY_AY_BRANCH_HINTS"):
        _capture(repo, benchmark_root, run_id="ay-branch-hints-malformed")

    assert not (
        repo / "reports/measured/artifacts/runs/ay-branch-hints-malformed/start.json"
    ).exists()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_ay_margin_reframe_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_MARGIN_REFRAME", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_AY_MARGIN_REFRAME"] == raw
    assert environment["typed_values"]["NY_AY_MARGIN_REFRAME"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_ay_margin_reframe_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_MARGIN_REFRAME", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_AY_MARGIN_REFRAME"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_ay_objective_first_sat_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_OBJECTIVE_FIRST_SAT", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_AY_OBJECTIVE_FIRST_SAT"] == raw
    assert environment["typed_values"]["NY_AY_OBJECTIVE_FIRST_SAT"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_ay_objective_first_sat_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_OBJECTIVE_FIRST_SAT", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_AY_OBJECTIVE_FIRST_SAT"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_sequential_mip_serial_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SERIAL", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_SERIAL"] == raw
    assert environment["typed_values"]["NY_MIP_SERIAL"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_sequential_mip_serial_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SERIAL", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MIP_SERIAL"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_mip_certified_shared_tree_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_CERTIFIED_SHARED_TREE", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_CERTIFIED_SHARED_TREE"] == raw
    assert environment["typed_values"]["NY_MIP_CERTIFIED_SHARED_TREE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_mip_certified_shared_tree_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_CERTIFIED_SHARED_TREE", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_MIP_CERTIFIED_SHARED_TREE"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_mip_safenlp_shared_prefix_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_SHARED_PREFIX", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_SAFENLP_SHARED_PREFIX"] == raw
    assert environment["typed_values"]["NY_MIP_SAFENLP_SHARED_PREFIX"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_mip_safenlp_shared_prefix_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_SHARED_PREFIX", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_MIP_SAFENLP_SHARED_PREFIX"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_mip_safenlp_target_fsb_prefix_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_TARGET_FSB_PREFIX", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_SAFENLP_TARGET_FSB_PREFIX"] == raw
    assert environment["typed_values"]["NY_MIP_SAFENLP_TARGET_FSB_PREFIX"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_mip_safenlp_target_fsb_prefix_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_TARGET_FSB_PREFIX", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_MIP_SAFENLP_TARGET_FSB_PREFIX"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_mip_safenlp_direct_first_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_DIRECT_FIRST", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_SAFENLP_DIRECT_FIRST"] == raw
    assert environment["typed_values"]["NY_MIP_SAFENLP_DIRECT_FIRST"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_mip_safenlp_direct_first_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_DIRECT_FIRST", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_MIP_SAFENLP_DIRECT_FIRST"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "value"),
    [("1", 1), ("05000", 5_000), ("60000", 60_000)],
)
def test_ay_node_warm_cap_is_sealed_as_a_bounded_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    value: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_NODE_WARM_CAP_MS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_AY_NODE_WARM_CAP_MS"] == raw
    assert environment["typed_values"]["NY_AY_NODE_WARM_CAP_MS"] == {
        "type": "bounded_positive_integer",
        "value": value,
        "minimum": 1,
        "maximum": 60_000,
    }


@pytest.mark.parametrize(
    "raw",
    ["", "0", "+5000", "-1", "60001", "5.0", " 5000", "5000 "],
)
def test_ay_node_warm_cap_malformed_measurement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_NODE_WARM_CAP_MS", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_AY_NODE_WARM_CAP_MS"):
        provenance._capture_environment()


def test_ay_node_warm_cap_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_NODE_WARM_CAP_MS_UNREVIEWED", "5000")

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_AY_NODE_WARM_CAP_MS_UNREVIEWED",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "value"),
    [("1", 1), ("0015", 15), ("3600", 3_600)],
)
def test_crown_ibp_collector_cap_is_sealed_as_a_bounded_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    value: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_IBP_COLLECTOR_CAP_SECS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CROWN_IBP_COLLECTOR_CAP_SECS"] == raw
    assert environment["typed_values"]["NY_CROWN_IBP_COLLECTOR_CAP_SECS"] == {
        "type": "bounded_positive_integer",
        "value": value,
        "minimum": 1,
        "maximum": 3_600,
    }


@pytest.mark.parametrize(
    "raw",
    ["", "0", "+15", "-1", "3601", "15.0", " 15", "15 "],
)
def test_crown_ibp_collector_cap_malformed_measurement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_IBP_COLLECTOR_CAP_SECS", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_CROWN_IBP_COLLECTOR_CAP_SECS"
    ):
        provenance._capture_environment()


def test_crown_chunk_aware_budget_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_CROWN_CHUNK_AWARE_BUDGET" not in environment["values"]
    assert "NY_CROWN_CHUNK_AWARE_BUDGET" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_crown_chunk_aware_budget_is_immutably_typed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_CHUNK_AWARE_BUDGET", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CROWN_CHUNK_AWARE_BUDGET"] == raw
    assert environment["typed_values"]["NY_CROWN_CHUNK_AWARE_BUDGET"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_crown_chunk_aware_budget_malformed_measurement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_CHUNK_AWARE_BUDGET", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_CROWN_CHUNK_AWARE_BUDGET"
    ):
        provenance._capture_environment()


def test_crown_chunk_aware_budget_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_CROWN_CHUNK_AWARE_BUDGET_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("NY_CONV_SKIP_DEAD_F32", "1"),
        ("NY_COMPACT_TAIL_K16", "1"),
        ("NY_ALPHA_REFRESH_FRACTION", "0.125"),
        ("NY_PGD_DIAG", "1"),
        ("NY_PGD_EXACT_BATCHED", "0"),
        ("NY_PGD_GAMA", "1"),
        ("NY_PGD_GAMA_LAMBDA", "50"),
        ("NY_PGD_GAMA_LIN_FRAC", "0.25"),
        ("NY_PGD_VJP_BATCH", "0"),
        ("NY_IMB", "1"),
        ("NY_IMB_AY_REGION_PROOF", "affine"),
        ("NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB", "1"),
        ("NY_IMB_BATCHED_REPLAY", "1"),
        ("NY_IMB_WIRE", "1"),
        ("NY_IMB_EARLY", "1"),
        ("NY_IMB_BUDGET_S", "300"),
        ("NY_IMB_LEAF_MODE", "crown_root"),
        ("NY_IMB_OBJ", "0"),
        ("NY_IMB_TAIL_ALPHA", "sample"),
        ("NY_IMB_TAIL_CERT_AY", "1"),
        ("NY_IMB_REGION_K", "2"),
        ("NY_IMB_REPLAY_ONLY", "1"),
        ("NY_IMB_REPLAY_ONLY_LEAF", "4"),
        ("NY_IMB_SELECTOR_K2_LIFT", "1"),
        ("NY_IMB_SELECTOR_K4_LIFT", "1"),
        ("NY_IMB_SELECTOR_RANGE_CRASH", "1"),
        ("NY_IMB_SELECTOR_SOLVE_PROFILE", "1"),
        ("NY_MIP_STABILITY_HINTS", "1"),
        ("NY_BAB_RESNET_REFOLD_GUARD", "0"),
        ("NY_PACKED_GRAPH_ALPHA_QUEUE", "1"),
        ("NY_CUDA_RESIDENT_PATCHES_ROOT", "1"),
        ("NY_ROOT_SKIP_ADAPTIVE_SPEC", "1"),
        ("NY_SEG_RESIDENT", "1"),
    ],
)
def test_dark_performance_canaries_are_captured_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest_path = _capture(repo, benchmark_root, run_id=f"dark-{key.lower()}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["environment"]["values"][key] == value


@pytest.mark.parametrize(
    ("key", "minimum", "maximum"),
    [
        ("NY_GAP_ATTRIBUTION_BUDGET_SECS", 1, 3_600),
        ("NY_GAP_ATTRIBUTION_ROWS", 1, 3),
    ],
)
def test_gap_attribution_integer_gates_are_bounded_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    minimum: int,
    maximum: int,
) -> None:
    for raw, expected in ((str(minimum), minimum), (str(maximum), maximum)):
        _clear_ny_environment(monkeypatch)
        monkeypatch.setenv(key, raw)

        environment = provenance._capture_environment()

        assert environment["values"][key] == raw
        assert environment["typed_values"][key] == {
            "type": "bounded_positive_integer",
            "value": expected,
            "minimum": minimum,
            "maximum": maximum,
        }


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("NY_GAP_ATTRIBUTION_BUDGET_SECS", "0"),
        ("NY_GAP_ATTRIBUTION_BUDGET_SECS", "3601"),
        ("NY_GAP_ATTRIBUTION_ROWS", "0"),
        ("NY_GAP_ATTRIBUTION_ROWS", "4"),
        ("NY_GAP_ATTRIBUTION_ROWS", "not-a-number"),
    ],
)
def test_gap_attribution_integer_gates_reject_unsealed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


EXACT_BOOLEAN_GATES = [
    "AY_LRA_WARM_SIMPLEX_STATE",
    "AY_MILP_NODE_PROP",
    "NY_ADAPTIVE_MICROBATCH_CONTROLLER",
    "NY_ALLOW_NONCUDA_MEASURE",
    "NY_ATTR_BRANCH",
    "NY_ATTR_BRANCH_DIAG",
    "NY_BAB_CLAUSE_LEARN",
    "NY_BAB_CLAUSE_REPLAY",
    "NY_BAB_RESNET_PARALLEL",
    "NY_BICCOS_BCP_SHADOW",
    "NY_BICCOS_Q_STAGE0",
    "NY_BICCOS_Q_STAGE1_REPLAY",
    "NY_COMPACT_TAIL_K16",
    "NY_CONE_REFRESH",
    "NY_CONSTRAINED_PATCHES_ALPHA_RELU_INPLACE",
    "NY_CONSTRAINED_PATCHES_BETA_SPARSE_CAPTURE",
    "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
    "NY_CROWN_IBP_SPARSE_RELU_ROWS",
    "NY_CUDA_RESIDENT_PATCHES_ROOT",
    "NY_DISABLE_CROWN_COLLECTION_CACHE",
    "NY_CUDA_WIDE",
    "NY_EFT_ERR",
    "NY_GAP_ATTRIBUTION",
    "NY_MARGIN_ROW_CONV_BWD_BLOCKED",
    "NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB",
    "NY_IMB_SELECTOR_K2_LIFT",
    "NY_IMB_SELECTOR_K4_LIFT",
    "NY_IMB_SELECTOR_RANGE_CRASH",
    "NY_IMB_SELECTOR_SOLVE_PROFILE",
    "NY_INTERM_REFINE_PROBE",
    "NY_MO_BETA_BASELINE_ONLY",
    "NY_MIP_SAFENLP_DIRECT_FIRST",
    "NY_MIP_SAFENLP_SHARED_PREFIX",
    "NY_MIP_SAFENLP_TARGET_FSB_PREFIX",
    "NY_MO_BETA_BASELINE_FIRST",
    "NY_MO_CUDA_BETA_SPSA",
    "NY_MO_CUDA_BOUNDED_SHARED_EXECUTOR",
    "NY_MO_CUDA_FACTORY_ENGINE_HANDOFF",
    "NY_MO_STALL_OBBT_CANARY",
    "NY_PACKED_GRAPH_ALPHA_QUEUE",
    "NY_PATCHES_DEADLINE_FLAT_BIAS",
    "NY_PATCHES_DEADLINE_PARALLEL_SCATTER",
    "NY_PATCHES_DEADLINE_RELU",
    "NY_PATCHES_EAGER_ERR",
    "NY_PATCHES_EAGER_ERR_7D",
    "NY_PHASE_TELEMETRY",
    "NY_RESNET_ERR_MERGE",
    "NY_ROOT_ALPHA_CUDA_MARGIN_LR_BRACKET",
    "NY_ROOT_ALPHA_CUDA_MARGIN_MW",
    "NY_ROOT_ALPHA_CUDA_MARGIN_STEP",
    "NY_ROOT_ALPHA_CUDA_MARGIN_TOPK",
    "NY_ROOT_ALPHA_CUDA_ROWS",
    "NY_ROOT_ALPHA_GPU",
    "NY_ROOT_ALPHA_MARGIN",
    "NY_ROOT_ALPHA_MARGIN_GRADIENT",
    "NY_ROOT_CRITICAL_GPU_ALPHA",
    "NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET",
    "NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET_CASCADE",
    "NY_ROOT_CRITICAL_GPU_ALPHA_LR_BRACKET",
    "NY_ROOT_CRITICAL_GPU_SPEC",
    "NY_ROOT_CROWN_INTERM",
    "NY_ROOT_INTERM_CUDA_FACTORY",
    "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
    "NY_ROOT_JOINT_INTERM_ALPHA_PROBE",
    "NY_ROOT_OUTPUT_CONDITIONED_HEAD",
    "NY_ROOT_SPARSE_INTERM_CROWN",
    "NY_ROOT_SKIP_ADAPTIVE_SPEC",
    "NY_SELECTIVE_ROOT_ALPHA",
    "NY_SEG_RESIDENT",
    "NY_SKIP_DISJ_PGD",
    "NY_WIDE_ACTIVE_COMPACTION_TELEMETRY",
]


@pytest.mark.parametrize("key", EXACT_BOOLEAN_GATES)
def test_exact_gate_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert key not in environment["values"]
    assert key not in environment["typed_values"]


@pytest.mark.parametrize("key", EXACT_BOOLEAN_GATES)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_exact_gate_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("key", EXACT_BOOLEAN_GATES)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_exact_gate_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_ay_milp_experiment_controls_are_default_dark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in ("AY_MILP_NODE_PROP",):
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    "raw",
    ["0", "1024", "200000000"],
)
def test_retired_ay_milp_prop_arm_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("AY_MILP_PROP_ARM", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match=r"unrecorded NY_\*/AY_\*/MIMALLOC_\*.*AY_MILP_PROP_ARM",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "unknown_key",
    [
        "AY_MILP_PROP_ARM_UNREVIEWED",
        "AY_MILP_NODE_PROP_UNREVIEWED",
    ],
)
def test_ay_milp_experiment_unknown_neighbors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    unknown_key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("AY_IDL_ENGINE", "0"),
        ("AY_MILP_FLIP_STALL", "120"),
    ],
)
def test_unreviewed_ay_pin_controls_remain_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    """A dependency bump must not silently grant inherited AY authority."""
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, value)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_root_crown_interm_secs_is_default_dark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_ROOT_CROWN_INTERM_SECS" not in environment["values"]
    assert "NY_ROOT_CROWN_INTERM_SECS" not in environment["typed_values"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0),
        ("00000060", 60),
        ("3600", 3_600),
    ],
)
def test_root_crown_interm_secs_is_captured_as_bounded_nonnegative_integer(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_CROWN_INTERM_SECS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ROOT_CROWN_INTERM_SECS"] == raw
    assert environment["typed_values"]["NY_ROOT_CROWN_INTERM_SECS"] == {
        "type": "bounded_nonnegative_integer",
        "value": expected,
        "minimum": 0,
        "maximum": 3_600,
    }


@pytest.mark.parametrize(
    "raw",
    ["", "-1", "+1", " 0", "0 ", "1.0", "3601", "not-a-number"],
)
def test_root_crown_interm_secs_rejects_out_of_range_or_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_CROWN_INTERM_SECS", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ROOT_CROWN_INTERM_SECS"):
        provenance._capture_environment()


ROOT_INTERM_CANONICAL_CAPS = (
    ("NY_ROOT_CROWN_INTERM_MAXDIM", 0, 20_000),
    ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM", 0, 8_192),
    ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS", 0, 512),
    ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS", 0, 4),
    ("NY_ROOT_SPARSE_INTERM_CROWN_SECS", 0, 8),
)


@pytest.mark.parametrize(("key", "minimum", "maximum"), ROOT_INTERM_CANONICAL_CAPS)
def test_root_interm_canonical_cap_is_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    minimum: int,
    maximum: int,
) -> None:
    del minimum, maximum
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert key not in environment["values"]
    assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    ("key", "raw", "expected", "minimum", "maximum"),
    [
        ("NY_ROOT_CROWN_INTERM_MAXDIM", "0", 0, 0, 20_000),
        ("NY_ROOT_CROWN_INTERM_MAXDIM", "512", 512, 0, 20_000),
        ("NY_ROOT_CROWN_INTERM_MAXDIM", "20000", 20_000, 0, 20_000),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM", "0", 0, 0, 8_192),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM", "8192", 8_192, 0, 8_192),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS", "0", 0, 0, 512),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS", "512", 512, 0, 512),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS", "0", 0, 0, 4),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS", "4", 4, 0, 4),
        ("NY_ROOT_SPARSE_INTERM_CROWN_SECS", "0", 0, 0, 8),
        ("NY_ROOT_SPARSE_INTERM_CROWN_SECS", "2", 2, 0, 8),
        ("NY_ROOT_SPARSE_INTERM_CROWN_SECS", "8", 8, 0, 8),
    ],
)
def test_root_interm_canonical_cap_is_captured_with_reviewed_bounds(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: int,
    minimum: int,
    maximum: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "bounded_canonical_nonnegative_integer",
        "value": expected,
        "minimum": minimum,
        "maximum": maximum,
    }


@pytest.mark.parametrize(("key", "minimum", "maximum"), ROOT_INTERM_CANONICAL_CAPS)
@pytest.mark.parametrize(
    "raw",
    ["", "00", "01", "-1", "+1", " 1", "1 ", "1.0", "not-a-number", "１"],
)
def test_root_interm_canonical_cap_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    minimum: int,
    maximum: int,
    raw: str,
) -> None:
    del minimum, maximum
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(("key", "minimum", "maximum"), ROOT_INTERM_CANONICAL_CAPS)
def test_root_interm_canonical_cap_rejects_values_above_reviewed_limit(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    minimum: int,
    maximum: int,
) -> None:
    del minimum
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, str(maximum + 1))

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    [
        "NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB",
        "NY_IMB_BATCHED_REPLAY",
        "NY_IMB_REPLAY_ONLY",
        "NY_IMB_SELECTOR_K2_LIFT",
        "NY_IMB_SELECTOR_K4_LIFT",
        "NY_IMB_SELECTOR_RANGE_CRASH",
        "NY_IMB_SELECTOR_SOLVE_PROFILE",
        "NY_IMB_TAIL_CERT_AY",
    ],
)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_imb_exact_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize(
    "key",
    [
        "NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB",
        "NY_IMB_BATCHED_REPLAY",
        "NY_IMB_REPLAY_ONLY",
        "NY_IMB_SELECTOR_K2_LIFT",
        "NY_IMB_SELECTOR_RANGE_CRASH",
        "NY_IMB_SELECTOR_SOLVE_PROFILE",
        "NY_IMB_TAIL_CERT_AY",
    ],
)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_imb_exact_gate_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize("raw", ["10", "20"])
def test_measurement_expected_cpus_is_captured_as_closed_enum(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MEASURE_EXPECTED_CPUS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MEASURE_EXPECTED_CPUS"] == raw
    assert environment["typed_values"]["NY_MEASURE_EXPECTED_CPUS"] == {
        "type": "enum",
        "value": raw,
        "allowed_values": ["10", "20"],
    }


@pytest.mark.parametrize(
    "raw",
    ["", "0", "010", "11", "21", "+20", "20 ", "ten"],
)
def test_measurement_expected_cpus_rejects_unreviewed_spellings(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MEASURE_EXPECTED_CPUS", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_MEASURE_EXPECTED_CPUS",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "mode", ["affine", "reachability", "residual", "selector", "shared"]
)
def test_imb_ay_region_proof_mode_is_captured_as_closed_enum(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_IMB_AY_REGION_PROOF", mode)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_IMB_AY_REGION_PROOF"] == mode
    assert environment["typed_values"]["NY_IMB_AY_REGION_PROOF"] == {
        "type": "enum",
        "value": mode,
        "allowed_values": ["affine", "reachability", "residual", "selector", "shared"],
    }


@pytest.mark.parametrize(
    "raw",
    ["", "Affine", "scalar", "default", " affine", "affine ", "unknown"],
)
def test_imb_ay_region_proof_mode_rejects_unreviewed_spellings(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_IMB_AY_REGION_PROOF", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_IMB_AY_REGION_PROOF"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NY_IMB_OBJ", "0", 0),
        ("NY_IMB_OBJ", "001", 1),
        ("NY_IMB_REPLAY_ONLY_LEAF", "4", 4),
        ("NY_IMB_REPLAY_ONLY_LEAF", "0004", 4),
    ],
)
def test_imb_replay_selectors_are_captured_as_decimal_usize(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "nonnegative_integer",
        "value": expected,
    }


def test_boxlift_environment_is_sealed_raw_and_unknown_neighbors_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    values = {
        "NY_BRANCH_RESCORE": "0",
        "NY_CLIP_INTERM_RESNET": "0",
        "NY_F64_LINEAGE_RECOVER": "1",
        "NY_INTERM_REFINE_SELECTIVE_TOPK": "020",
        "NY_PHASE_TELEMETRY": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS": "012",
        "NY_ROOT_JOINT_INTERM_ALPHA_LR": "0.10",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM": "02048",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL": "0008",
        "NY_ROOT_JOINT_INTERM_ALPHA_PROBE": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA_SECS": "20.0",
        "NY_ROOT_JOINT_MIN_REMAINING_SECS": "000",
        "NY_ROOT_SPARSE_INTERM_CROWN": "1",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM": "8192",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS": "512",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS": "4",
        "NY_ROOT_SPARSE_INTERM_CROWN_SECS": "2",
        "NY_ROOT_SPEC_PRUNE": "1",
    }
    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="boxlift-default").read_text()
    )
    for key in values:
        assert key not in default_manifest["environment"]["values"]

    for key, value in values.items():
        monkeypatch.setenv(key, value)
    raw_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="boxlift-raw").read_text()
    )
    environment = raw_manifest["environment"]
    sparse_cap_bounds = {
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM": (0, 8_192),
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS": (0, 512),
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS": (0, 4),
        "NY_ROOT_SPARSE_INTERM_CROWN_SECS": (0, 8),
    }
    for key, value in values.items():
        assert environment["values"][key] == value
        if key in {
            "NY_PHASE_TELEMETRY",
            "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
            "NY_ROOT_JOINT_INTERM_ALPHA_PROBE",
            "NY_ROOT_SPARSE_INTERM_CROWN",
        }:
            assert environment["typed_values"][key] == {
                "type": "boolean",
                "value": True,
            }
        elif key in sparse_cap_bounds:
            minimum, maximum = sparse_cap_bounds[key]
            assert environment["typed_values"][key] == {
                "type": "bounded_canonical_nonnegative_integer",
                "value": int(value),
                "minimum": minimum,
                "maximum": maximum,
            }
        else:
            assert key not in environment["typed_values"]

    unknown_key = "NY_ROOT_JOINT_INTERM_ALPHA_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")
    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        _capture(repo, benchmark_root, run_id="boxlift-unknown")
    assert not (
        repo / "reports/measured/artifacts/runs/boxlift-unknown/start.json"
    ).exists()


def test_exact_batched_route_switch_is_absent_by_default_and_not_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="exact-batched-default").read_text()
    )
    assert "NY_PGD_EXACT_BATCHED" not in default_manifest["environment"]["values"]

    # The runtime selects generic batching only for the exact string "0". Like
    # adjacent PGD route switches, provenance preserves other spellings as raw
    # evidence instead of normalizing them into a behavior-changing value.
    monkeypatch.setenv("NY_PGD_EXACT_BATCHED", "00")
    raw_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="exact-batched-raw").read_text()
    )
    assert raw_manifest["environment"]["values"]["NY_PGD_EXACT_BATCHED"] == "00"
    assert "NY_PGD_EXACT_BATCHED" not in raw_manifest["environment"]["typed_values"]


def test_spec_alpha_direct_is_captured_raw_and_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()
    assert "NY_SPEC_ALPHA_DIRECT" not in environment["values"]

    # Runtime activation is exact-string (`"1"` only). Preserve both active
    # and inactive spellings as raw launch evidence; do not normalize them.
    for raw in ("1", "0", "true"):
        monkeypatch.setenv("NY_SPEC_ALPHA_DIRECT", raw)
        environment = provenance._capture_environment()
        assert environment["values"]["NY_SPEC_ALPHA_DIRECT"] == raw
        assert "NY_SPEC_ALPHA_DIRECT" not in environment["typed_values"]


def test_rump_f64_engine_zero_is_captured_in_immutable_start_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="rump-f64-default").read_text()
    )
    assert "NY_RUMP_F64_ENGINE" not in default_manifest["environment"]["values"]

    # The runtime disables this engine only for exact "0". Preserve that launch
    # spelling as raw evidence instead of normalizing it into another behavior.
    monkeypatch.setenv("NY_RUMP_F64_ENGINE", "0")
    start_path = _capture(repo, benchmark_root, run_id="rump-f64-disabled")
    start_bytes = start_path.read_bytes()
    environment = json.loads(start_bytes)["environment"]
    assert environment["values"]["NY_RUMP_F64_ENGINE"] == "0"
    assert "NY_RUMP_F64_ENGINE" not in environment["typed_values"]

    monkeypatch.setenv("NY_RUMP_F64_ENGINE", "1")
    with pytest.raises(FileExistsError, match="immutable evidence"):
        _capture(repo, benchmark_root, run_id="rump-f64-disabled")
    assert start_path.read_bytes() == start_bytes


def test_postbab_ab_environment_is_absent_by_default_and_preserved_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="postbab-seeds-default").read_text()
    )
    default_values = default_manifest["environment"]["values"]
    assert "NY_POSTBAB_BAB_SEEDS" not in default_values
    assert "NY_POSTBAB_BAB_SEEDS_K" not in default_values
    assert "NY_POSTBAB_FRONTIER_FASTLANE" not in default_values
    assert "NY_POSTBAB_FRONTIER_FASTLANE_SECS" not in default_values
    assert "NY_POSTBAB_RESERVE_SECS" not in default_values
    assert "NY_ACASXU_PROF" not in default_values

    # The runtime enables export only for exact "1", trims/parses/clamps K,
    # trims/parses the reserve, and enables profiling on any present value.
    # Preserve those launch spellings instead of pre-normalizing their behavior.
    monkeypatch.setenv("NY_POSTBAB_BAB_SEEDS", "1")
    monkeypatch.setenv("NY_POSTBAB_BAB_SEEDS_K", " 0256 ")
    monkeypatch.setenv("NY_POSTBAB_FRONTIER_FASTLANE", "1")
    monkeypatch.setenv("NY_POSTBAB_FRONTIER_FASTLANE_SECS", " 0010 ")
    monkeypatch.setenv("NY_POSTBAB_RESERVE_SECS", " 0100 ")
    monkeypatch.setenv("NY_ACASXU_PROF", "0")
    raw_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="postbab-seeds-raw").read_text()
    )
    raw_environment = raw_manifest["environment"]
    assert raw_environment["values"]["NY_POSTBAB_BAB_SEEDS"] == "1"
    assert raw_environment["values"]["NY_POSTBAB_BAB_SEEDS_K"] == " 0256 "
    assert raw_environment["values"]["NY_POSTBAB_FRONTIER_FASTLANE"] == "1"
    assert raw_environment["values"]["NY_POSTBAB_FRONTIER_FASTLANE_SECS"] == " 0010 "
    assert raw_environment["values"]["NY_POSTBAB_RESERVE_SECS"] == " 0100 "
    assert raw_environment["values"]["NY_ACASXU_PROF"] == "0"
    assert "NY_POSTBAB_BAB_SEEDS" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_BAB_SEEDS_K" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_FRONTIER_FASTLANE" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_FRONTIER_FASTLANE_SECS" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_RESERVE_SECS" not in raw_environment["typed_values"]
    assert "NY_ACASXU_PROF" not in raw_environment["typed_values"]


def test_screen_schedule_environment_is_absent_by_default_and_preserved_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    keys = {
        "NY_SCREEN_WAVE_SIZE": " 0256 ",
        "NY_SCREEN_MVF_WAVE_SIZE": "0008",
        "NY_SCREEN_CELL_CHUNK": "0",
        "NY_SCREEN_MVF_CHUNK": "not-a-number",
        "NY_SCREEN_CROWN_MS": "5.5",
    }
    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="screen-schedule-default").read_text()
    )
    for key in keys:
        assert key not in default_manifest["environment"]["values"]

    for key, value in keys.items():
        monkeypatch.setenv(key, value)
    raw_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="screen-schedule-raw").read_text()
    )
    raw_environment = raw_manifest["environment"]
    for key, value in keys.items():
        assert raw_environment["values"][key] == value
        assert key not in raw_environment["typed_values"]


def test_start_binds_ay_executable_and_completion_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    ay_binary = tmp_path / "trusted-ay"
    ay_binary.write_text(
        f"#!/bin/sh\nprintf '%s\\n' 'ay fixture' 'build.commit={AY_REV}'\n",
        encoding="utf-8",
    )
    ay_binary.chmod(0o755)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY", str(ay_binary))
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    start = _capture(repo, benchmark_root, run_id="ay-drift")
    start_record = json.loads(start.read_text(encoding="utf-8"))
    executable = start_record["dependencies"]["ay"]["executable"]
    assert executable["declared_path"] == str(ay_binary)
    assert executable["resolved_path"] == str(ay_binary.resolve())
    assert executable["sha256"] == provenance._sha256_file(ay_binary)
    assert executable["build_commit"] == AY_REV
    assert f"build.commit={AY_REV}" in executable["version_stdout"]

    ay_binary.write_text(
        f"#!/bin/sh\nprintf '%s\\n' 'changed ay fixture' 'build.commit={AY_REV}'\n",
        encoding="utf-8",
    )
    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}
    assert record["integrity"]["status"] == "invalid"
    assert record["completed_successfully"] is False
    assert "ay_executable_identity_mismatch" in codes


@pytest.mark.parametrize(
    "version_output",
    [
        "ay fixture",
        "ay fixture\nbuild.commit=0123456789abcdef0123456789abcdef01234567",
        f"ay fixture\nbuild.commit={AY_REV}-dirty",
        f"ay fixture\nbuild.commit={AY_REV}\nbuild.commit={AY_REV}",
    ],
)
def test_start_rejects_ay_executable_not_built_from_exact_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_output: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    ay_binary = tmp_path / "unbound-ay"
    shell_lines = " ".join(repr(line) for line in version_output.splitlines())
    ay_binary.write_text(
        f"#!/bin/sh\nprintf '%s\\n' {shell_lines}\n",
        encoding="utf-8",
    )
    ay_binary.chmod(0o755)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY", str(ay_binary))
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    with pytest.raises(provenance.ProvenanceError, match=r"AY executable build\.commit"):
        _capture(repo, benchmark_root, run_id="ay-build-mismatch")


@pytest.mark.parametrize(
    ("changed_identity", "violation_code"),
    [
        ("solver", "solver_binary_sha256_mismatch"),
        ("ny_worktree", "ny_worktree_identity_mismatch"),
        ("benchmark", "benchmark_identity_mismatch"),
        ("config_inputs", "config_inputs_identity_mismatch"),
    ],
)
def test_completion_rejects_end_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_identity: str,
    violation_code: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "demo.yaml"
    config.write_text("verifier:\n  timeout: 1\n", encoding="utf-8")
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(
        repo,
        benchmark_root,
        run_id=f"drift-{changed_identity}",
        configs_dir=configs,
    )

    if changed_identity == "solver":
        (repo / "target/release/ny").write_text(
            "#!/bin/sh\necho 'changed ny'\n", encoding="utf-8"
        )
    elif changed_identity == "ny_worktree":
        (repo / "scripts/measure_ny_scorecard.sh").write_text(
            "#!/bin/bash\nexit 1\n", encoding="utf-8"
        )
    elif changed_identity == "benchmark":
        (benchmark_root / "README").write_text("changed fixture\n", encoding="utf-8")
    else:
        config.write_text("verifier:\n  timeout: 2\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}
    assert record["integrity"]["status"] == "invalid"
    assert record["completed_successfully"] is False
    assert violation_code in codes


def test_completion_rejects_cuda_runtime_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    monkeypatch.setattr(
        provenance,
        "_recapture_cuda_runtime_from_start",
        lambda _start: {
            **CUDA_RUNTIME_FIXTURE,
            "probe": {"fixture": False},
        },
    )
    start = _capture(repo, benchmark_root, run_id="cuda-runtime-drift")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "cuda_runtime_identity_mismatch" in codes
    assert record["integrity"]["checks"]["cuda_runtime"]["status"] == "invalid"


def test_cuda_runtime_dependency_is_explicitly_not_required_for_cpu_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "values": {
            "NY_ALLOW_NONCUDA_MEASURE": "1",
            "NY_NO_CUDA": "1",
        }
    }
    monkeypatch.setattr(
        provenance,
        "_capture_cuda_runtime_identity",
        lambda _binary: pytest.fail("CPU debug capture must not probe CUDA"),
    )

    identity = REAL_CAPTURE_CUDA_RUNTIME_DEPENDENCY(
        Path("/does/not/need/to/exist"),
        environment,
        ["mip", "cuda"],
    )

    assert identity == {
        "schema": provenance.MEASUREMENT_CUDA_RUNTIME_SCHEMA,
        "status": "not_required",
        "reason": "noncuda_measurement_explicitly_allowed",
    }


def test_cuda_runtime_cpu_debug_requires_exact_cpu_routing() -> None:
    with pytest.raises(provenance.ProvenanceError, match="NY_NO_CUDA=1"):
        REAL_CAPTURE_CUDA_RUNTIME_DEPENDENCY(
            Path("/does/not/need/to/exist"),
            {"values": {"NY_ALLOW_NONCUDA_MEASURE": "1"}},
            ["mip", "cuda"],
        )


def test_cuda_score_measurement_rejects_cpu_routing_override() -> None:
    with pytest.raises(provenance.ProvenanceError, match="NY_NO_CUDA is forbidden"):
        REAL_CAPTURE_CUDA_RUNTIME_DEPENDENCY(
            Path("/must/not/be/probed"),
            {"values": {"NY_NO_CUDA": "1"}},
            ["mip", "cuda"],
        )


def test_completion_rejects_malformed_input_hash_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="bad-cache")
    cache = start.with_name("input_hash_cache.json")
    cache.write_text(
        json.dumps(
            {
                "schema": "ny_measurement_input_hash_cache_v1",
                "run_id": "bad-cache",
                "start_manifest_sha256": provenance._sha256(start.read_bytes()),
                "entries": {"not-a-content-address": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}
    assert record["integrity"]["status"] == "invalid"
    assert record["completed_successfully"] is False
    assert "input_hash_cache_entries_invalid" in codes


def test_completion_records_complete_run_artifact_bijection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, _metadata, _result, _log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id="complete-evidence",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    run_evidence = record["integrity"]["checks"]["run_evidence"]
    cache_check = record["integrity"]["checks"]["input_hash_cache"]

    assert record["completed_successfully"] is True
    assert record["integrity"]["checks"]["cuda_runtime"]["status"] == "valid"
    assert run_evidence["status"] == "valid"
    assert run_evidence["metadata_count"] == 1
    assert run_evidence["result_count"] == 1
    assert run_evidence["solver_log_count"] == 1
    assert run_evidence["csv_row_count"] == 1
    assert run_evidence["validated_record_count"] == 1
    assert cache_check["entry_count"] == 2
    assert cache_check["referenced_entry_count"] == 2
    assert cache_check["rehashed_entry_count"] == 2


@pytest.mark.parametrize(
    ("artifact", "violation_code"),
    [
        ("metadata", "run_result_artifact_mismatch"),
        ("result", "run_result_artifact_mismatch"),
        ("solver_log", "run_solver_log_artifact_mismatch"),
    ],
)
def test_completion_rejects_tampered_run_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    violation_code: str,
) -> None:
    start, metadata_path, result_path, solver_log_path, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"tampered-{artifact}",
    )
    if artifact == "metadata":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["result_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    elif artifact == "result":
        result_path.write_bytes(b"sat\n((X_0 0.0))\n")
    else:
        solver_log_path.write_bytes(b"tampered solver log\n")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert violation_code in codes


def test_completion_rejects_sat_without_structured_witness_after_consistent_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "sat-without-witness"
    start, metadata_path, result_path, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
    )
    result_path.write_bytes(b"sat\n")
    result_digest = provenance._sha256_file(result_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "solver_verdict": "sat",
            "witness_present": True,
            "result_sha256": result_digest,
            "raw_result_sha256": result_digest,
            "counterexample_validation": {"status": "not_checked", "checker": None},
        }
    )
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    start_record = json.loads(start.read_text(encoding="utf-8"))
    csv_path = Path(start_record["measurement"]["output_dir"]) / "demo.csv"
    csv_path.write_text(
        f"demo,model.onnx,property.vnnlib,0,sat,9,{run_id}\n",
        encoding="utf-8",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_sat_witness_invalid" in codes


def test_completion_rejects_empty_unsat_after_consistent_digest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, metadata_path, result_path, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id="empty-unsat",
    )
    result_path.write_bytes(b"")
    result_digest = provenance._sha256_file(result_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["result_sha256"] = result_digest
    metadata["raw_result_sha256"] = result_digest
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_result_verdict_mismatch" in codes


def test_postflight_sat_witness_parser_accepts_vnnlib1_and_vnnlib2() -> None:
    assert provenance._structured_sat_assignment([b"((X_0 0.25))"])
    assert provenance._structured_sat_assignment(
        [b"X float32 [1, 2]", b"0.25", b"-0.5"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solver_exit_status", 1),
        ("solver_verdict", "maybe"),
        ("timeout_seconds", "120"),
    ],
)
def test_completion_rejects_invalid_verdict_status_or_timeout_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    start, metadata_path, _result, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"invalid-{field}",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_metadata_fields_invalid" in codes


@pytest.mark.parametrize(
    ("mutation", "violation_code"),
    [
        ("result", "run_evidence_file_changed_during_completion"),
        ("csv", "run_evidence_file_changed_during_completion"),
        ("namespace", "run_artifact_namespace_changed_during_completion"),
    ],
)
def test_completion_rechecks_artifact_and_csv_snapshot_after_cache_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    violation_code: str,
) -> None:
    run_id = f"snapshot-{mutation}"
    start, metadata_path, result_path, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
    )
    start_record = json.loads(start.read_text(encoding="utf-8"))
    csv_path = Path(start_record["measurement"]["output_dir"]) / "demo.csv"
    original_validate_cache = provenance._validate_input_hash_cache

    def mutate_after_cache(**kwargs):
        validated = original_validate_cache(**kwargs)
        if mutation == "result":
            result_path.write_bytes(b"unknown\n")
        elif mutation == "csv":
            csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n")
        else:
            orphan = metadata_path.parents[1] / "99999-race" / f"{run_id}.results"
            orphan.parent.mkdir()
            orphan.write_bytes(b"unsat\n")
        return validated

    monkeypatch.setattr(provenance, "_validate_input_hash_cache", mutate_after_cache)

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert violation_code in codes


@pytest.mark.parametrize("artifact", ["metadata", "result", "solver_log"])
def test_completion_rejects_deleted_or_orphaned_run_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    start, metadata_path, result_path, solver_log_path, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"deleted-{artifact}",
    )
    {
        "metadata": metadata_path,
        "result": result_path,
        "solver_log": solver_log_path,
    }[artifact].unlink()

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_artifact_trio_incomplete" in codes


def test_completion_rejects_extra_orphan_run_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "orphan-artifact"
    start, metadata_path, _result, _log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
    )
    orphan = metadata_path.parents[1] / "99999-orphan" / f"{run_id}.results"
    orphan.parent.mkdir()
    orphan.write_bytes(b"unsat\n")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_artifact_trio_incomplete" in codes


@pytest.mark.parametrize("mutation", ["delete", "digest", "extra"])
def test_completion_rejects_missing_corrupt_or_unreferenced_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    start, _metadata, _result, _log, cache_path = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"cache-{mutation}",
    )
    if mutation == "delete":
        cache_path.unlink()
    else:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if mutation == "digest":
            first = next(iter(cache["entries"].values()))
            first["sha256"] = "0" * 64
        else:
            extra = tmp_path / "unreferenced-input"
            extra.write_bytes(b"unreferenced\n")
            fingerprint = provenance._file_fingerprint(extra)
            key = provenance._input_cache_key(str(extra.resolve()), fingerprint)
            cache["entries"][key] = {
                "path": str(extra.resolve()),
                "fingerprint": fingerprint,
                "sha256": provenance._sha256_file(extra),
            }
        cache_path.write_text(json.dumps(cache) + "\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    expected = {
        "delete": "input_hash_cache_missing_for_run_artifacts",
        "digest": "input_hash_cache_entry_sha256_mismatch",
        "extra": "input_hash_cache_entry_unreferenced",
    }
    assert expected[mutation] in codes


def test_completion_rejects_csv_row_without_matching_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, _metadata, _result, _log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id="csv-tamper",
    )
    start_record = json.loads(start.read_text(encoding="utf-8"))
    csv_path = Path(start_record["measurement"]["output_dir"]) / "demo.csv"
    csv_path.write_text(
        "demo,model.onnx,property.vnnlib,0,sat,9,csv-tamper\n",
        encoding="utf-8",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_csv_artifact_bijection_mismatch" in codes


@pytest.mark.parametrize(
    "unknown_key",
    [
        "NY_UNREVIEWED_SECRET",
        "NY_RUMP_F64_ENGINE_UNREVIEWED",
        "NY_MULTINEURON",
        "NY_MULTINEURON_STEM",
    ],
)
def test_unknown_ny_environment_fails_closed_without_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unknown_key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(unknown_key, "do-not-record")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        _capture(repo, benchmark_root, run_id="rejected-run")

    assert not (
        repo / "reports/measured/artifacts/runs/rejected-run/start.json"
    ).exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("BASH_ENV", "/tmp/forged-scorecard-environment"),
        ("BASH_FUNC_ulimit%%", "() { echo 167772160; }"),
    ],
)
def test_shell_injection_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, value)

    with pytest.raises(provenance.ProvenanceError, match="unsafe shell launch"):
        provenance._capture_environment()


def test_inherited_ay_environment_fails_closed_without_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("AY_MILP_FLIP_CAP_SECS", "3")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="AY_MILP_FLIP_CAP_SECS"):
        _capture(repo, benchmark_root, run_id="rejected-ay-run")

    assert not (
        repo / "reports/measured/artifacts/runs/rejected-ay-run/start.json"
    ).exists()


@pytest.mark.parametrize(
    "key", ["MIMALLOC_PURGE_DELAY", "MIMALLOC_FUTURE_UNREVIEWED_OPTION"]
)
def test_mimalloc_environment_fails_closed_without_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "0")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=key):
        _capture(repo, benchmark_root, run_id="rejected-mimalloc-run")

    assert not (
        repo / "reports/measured/artifacts/runs/rejected-mimalloc-run/start.json"
    ).exists()


def test_missing_build_feature_declaration_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.delenv("NY_BUILD_FEATURES")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(
        provenance.ProvenanceError, match="NY_BUILD_FEATURES is required"
    ):
        _capture(repo, benchmark_root, run_id="missing-features")

    assert not (
        repo / "reports/measured/artifacts/runs/missing-features/start.json"
    ).exists()


def test_rejects_broad_or_git_metadata_output_roots(tmp_path: Path) -> None:
    repo = tmp_path / "ny"
    repo.mkdir()
    (repo / ".git").mkdir()

    with pytest.raises(provenance.ProvenanceError, match="unsafe broad"):
        provenance._validate_mutation_root(repo, repo, "measurement output directory")
    with pytest.raises(provenance.ProvenanceError, match="Git metadata"):
        provenance._validate_mutation_root(
            repo / ".git" / "measurements",
            repo,
            "measurement output directory",
        )


def test_measurement_output_inside_repo_must_be_ignored_and_untracked(
    tmp_path: Path,
) -> None:
    repo, _benchmark_root = _measurement_repos(tmp_path)
    tracked_dir = repo / "reports/measured"
    tracked_dir.mkdir(parents=True)
    (tracked_dir / "demo.csv").write_text("legacy row\n", encoding="utf-8")
    _run("git", "add", "-f", "reports/measured/demo.csv", cwd=repo)
    _run("git", "commit", "-qm", "track canonical scorecard", cwd=repo)

    with pytest.raises(provenance.ProvenanceError, match="contains tracked NY paths"):
        provenance._validate_mutation_root(
            tracked_dir,
            repo,
            "measurement output directory",
        )
    with pytest.raises(provenance.ProvenanceError, match="must be Git-ignored"):
        provenance._validate_mutation_root(
            repo / "unignored-output",
            repo,
            "measurement output directory",
        )

    provenance._validate_mutation_root(
        repo / "reports/isolated-run",
        repo,
        "measurement output directory",
    )


def test_process_snapshot_is_sorted_bounded_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_args = "worker --token top-secret " + "x" * 500
    stdout = (
        "10 1 S 20 3.0 2.0 python python job.py --api-key=hidden\n"
        f"20 1 R 5 99.0 1.0 worker {long_args}\n"
        "30 1 S 10 3.0 4.0 curl curl https://user:pass@example.test/x\n"
    )
    monkeypatch.setattr(
        provenance,
        "_snapshot_command",
        lambda _command: {
            "status": "ok",
            "returncode": 0,
            "stdout": stdout,
            "stdout_truncated": False,
            "stderr": "",
            "stderr_truncated": False,
        },
    )

    snapshot = provenance._process_snapshot()

    assert [entry["pid"] for entry in snapshot["entries"]] == [20, 30, 10]
    encoded = json.dumps(snapshot)
    assert "top-secret" not in encoded
    assert "hidden" not in encoded
    assert "user:pass" not in encoded
    assert snapshot["entries"][0]["args_redacted"].endswith("…")
    assert len(snapshot["entries"][0]["args_redacted"]) == provenance.PROCESS_ARGS_LIMIT


def test_host_state_schema_is_stable_when_commands_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        provenance.os,
        "getloadavg",
        lambda: (_ for _ in ()).throw(OSError("unsupported")),
    )
    monkeypatch.setattr(
        provenance,
        "_utc_now",
        lambda: "2026-07-18T01:02:03.000000Z",
    )

    state = provenance._capture_host_state()

    assert state["schema"] == "ny_measurement_host_state_v1"
    assert state["captured_at_utc"] == "2026-07-18T01:02:03.000000Z"
    assert state["load_average"] == {
        "available": False,
        "one_minute": None,
        "five_minutes": None,
        "fifteen_minutes": None,
    }
    assert state["processes"]["status"] == "unavailable"
    assert state["processes"]["entries"] == []
    assert state["gpu"]["utilization"]["status"] == "unavailable"
    assert state["gpu"]["compute_processes"]["status"] == "unavailable"


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_nn4sys_phase_event_gate_is_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_1D_PHASE_EVENTS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_NN4SYS_1D_PHASE_EVENTS"] == raw
    assert environment["typed_values"]["NY_NN4SYS_1D_PHASE_EVENTS"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_nn4sys_phase_event_gate_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_1D_PHASE_EVENTS", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_NN4SYS_1D_PHASE_EVENTS"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "value"),
    [("1", 1), ("00256", 256), ("4096", 4096)],
)
def test_nn4sys_phase_event_cap_is_bounded_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    value: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS"] == raw
    assert environment["typed_values"]["NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS"] == {
        "type": "bounded_positive_integer",
        "value": value,
        "minimum": 1,
        "maximum": 4096,
    }


@pytest.mark.parametrize("raw", ["", "0", "4097", "-1", "+1", " 1", "1 "])
def test_nn4sys_phase_event_cap_rejects_out_of_range_or_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_nn4sys_mvf_clip_gate_is_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_MVF_CLIP_DIAG", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_NN4SYS_MVF_CLIP_DIAG"] == raw
    assert environment["typed_values"]["NY_NN4SYS_MVF_CLIP_DIAG"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_nn4sys_mvf_clip_gate_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_MVF_CLIP_DIAG", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_NN4SYS_MVF_CLIP_DIAG"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("name", "maximum"),
    [
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", 4096),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", 16384),
    ],
)
@pytest.mark.parametrize("raw", ["1", "00256"])
def test_nn4sys_mvf_clip_caps_are_typed_bounded_positive_integers(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    maximum: int,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(name, raw)

    environment = provenance._capture_environment()

    assert environment["values"][name] == raw
    assert environment["typed_values"][name] == {
        "type": "bounded_positive_integer",
        "value": int(raw),
        "minimum": 1,
        "maximum": maximum,
    }


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", ""),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "0"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "-1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "+1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", " 1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "1 "),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "4097"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", ""),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "0"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "-1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "+1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", " 1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "1 "),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "16385"),
    ],
)
def test_nn4sys_mvf_clip_caps_reject_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(name, raw)

    with pytest.raises(provenance.ProvenanceError, match=name):
        provenance._capture_environment()


def test_smaller_profile_does_not_shrink_the_address_space_limit() -> None:
    """RLIMIT_AS is virtual address space, NOT the cgroup's physical cap.

    The CUDA driver and ONNX Runtime reserve tens of GiB of VA regardless of how
    much physical memory a host has. An earlier wsl24-20g profile attested
    RLIMIT_AS == memory.max, which starved those reservations: a cersyve instance
    that returns `sat` in 1s under an 80 GiB limit instead burned 106s and
    recorded `timeout` under a 20 GiB one, so the sweep produced all-timeout rows
    that looked like a real measurement.

    The address-space ceiling must remain common across physical profiles and
    strictly above even the largest profile's memory.max. Otherwise a virtual
    reservation becomes the allocation envelope before charged memory does.
    """
    profiles = provenance.CONTAINMENT_PROFILES
    assert set(profiles) == {"gb10-80g", "wsl24-20g"}
    assert provenance.EXPECTED_RLIMIT_AS_BYTES == 160 * 1024**3

    for name, profile in profiles.items():
        assert profile["rlimit_as_bytes"] == provenance.EXPECTED_RLIMIT_AS_BYTES, (
            f"{name}: RLIMIT_AS must stay at the guard ceiling; CUDA/ORT "
            f"reservations do not shrink with the host"
        )

    large = profiles["gb10-80g"]
    small = profiles["wsl24-20g"]
    assert large["memory_max_bytes"] < large["rlimit_as_bytes"]
    assert small["memory_max_bytes"] < small["rlimit_as_bytes"], (
        "wsl24-20g must contain physical memory more tightly than address space"
    )
    assert small["memory_high_bytes"] < small["memory_max_bytes"]


def test_stale_solver_binary_is_refused_before_measurement(tmp_path) -> None:
    """A binary older than the tree it will measure must stop the run.

    The manifest pinned the worktree HEAD and the binary sha256 independently,
    so sealing a binary built at commit A and then moving the worktree to
    commit B produced evidence that looked complete while measuring a
    mismatched pair. It happened: a sweep sealed configs from a commit adding
    the `bab.branching.input_split.sat_escape_branch` preset key while running
    a binary built before its reader existed. Every nn4sys instance
    fail-closed on the unrecognized key, banking 194 `unknown` rows that
    scored as a legitimate zero for the whole category.
    """
    import os

    repo = Path(__file__).resolve().parent.parent
    build_epoch = provenance._last_commit_epoch(repo, provenance._BUILD_INPUT_PATHS)
    config_epoch = provenance._last_commit_epoch(repo, provenance._BEHAVIOUR_INPUT_PATHS)
    assert build_epoch is not None and config_epoch is not None

    binary = tmp_path / "ny"
    binary.write_bytes(b"not a real solver")

    newest = max(build_epoch, config_epoch)
    os.utime(binary, (newest + 10, newest + 10))
    fresh = provenance._capture_build_coherence(repo, binary)
    assert fresh["binary_mtime_epoch"] == newest + 10
    assert fresh["build_inputs_last_commit_epoch"] == build_epoch

    oldest = min(build_epoch, config_epoch)
    os.utime(binary, (oldest - 10, oldest - 10))
    with pytest.raises(provenance.ProvenanceError) as excinfo:
        provenance._capture_build_coherence(repo, binary)
    assert "predates the worktree" in str(excinfo.value)
