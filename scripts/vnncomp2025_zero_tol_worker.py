#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Hermetic worker for the pinned VNN-COMP 2025 ZERO-TOL checker.

This file is retained byte-for-byte with the exact checker runtime.  It must
be started by the parent harness with the embedded interpreter's
``-I -S -B`` flags.  It intentionally has no NY/workspace imports.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import sys
from pathlib import Path
from typing import Any

PROTOCOL = "ny_vnncomp2025_zero_tol_worker_v1"
RUNTIME_ROOT = Path("<home>/ny-vnncomp2025-checker-exact-20260731T074000Z")
PYTHON_BASE = RUNTIME_ROOT / "python-base"
STDLIB_ROOT = PYTHON_BASE / "lib/python3.11"
SITE_PACKAGES = RUNTIME_ROOT / "lib/python3.11/site-packages"
SCORING_ROOT = RUNTIME_ROOT / "results/SCORING-ZERO-TOL"
WORKER_PATH = RUNTIME_ROOT / "harness/vnncomp2025_zero_tol_worker.py"
CPU_PROVIDER = "CPUExecutionProvider"
COUNTEREXAMPLE_ATOL = 1e-4
COUNTEREXAMPLE_RTOL = 1e-3
PINNED_REQUIREMENTS = {
    "cachier": "2.2.2",
    "coloredlogs": "15.0.1",
    "flatbuffers": "23.5.26",
    "humanfriendly": "10.0",
    "mpmath": "1.3.0",
    "numpy": "1.24.4",
    "onnx": "1.15.0",
    "onnxruntime": "1.16.3",
    "packaging": "23.2",
    "portalocker": "2.8.2",
    "protobuf": "4.25.1",
    "sympy": "1.12",
    "watchdog": "3.0.0",
}
METRIC_RE = re.compile(
    r"L-inf norm difference between onnx execution and CE file output:"
    r"\s*([^;\n]+?)\s*\(rel error:\s*([^)]+?)\s*\);"
)
REQUEST_KEYS = frozenset(
    {
        "protocol",
        "onnx_path",
        "vnnlib_path",
        "counterexample_path",
        "native_dependencies_path",
        "receipt_path",
        "abs_tolerance",
        "rel_tolerance",
    }
)
RESPONSE_KEYS = frozenset({"result", "message", "diff", "rel_error"})
FILE_RECEIPT_KEYS = frozenset({"sha256", "size_bytes"})
WORKER_RECEIPT_KEYS = frozenset(
    {
        "protocol",
        "request_sha256",
        "onnx",
        "vnnlib",
        "counterexample",
        "response_sha256",
        "native_dependencies_sha256",
    }
)
KNOWN_RESULTS = frozenset(
    {
        "correct",
        "correct_up_to_tolerance",
        "no_ce",
        "exec_doesnt_match",
        "spec_not_violated",
        "wrong_shape",
    }
)


class WorkerError(RuntimeError):
    """The hermetic worker boundary was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _file_receipt(path: Path) -> dict[str, object]:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise WorkerError(f"receipt input is not regular: {path}")
    digest = _sha256(path)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise WorkerError(f"receipt input changed while hashed: {path}")
    return {"sha256": digest, "size_bytes": after.st_size}


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkerError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise WorkerError(f"non-finite JSON constant: {value}")


def _json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            data,
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise WorkerError("request is not strict JSON") from error
    if not isinstance(value, dict):
        raise WorkerError("request is not a JSON object")
    return value


def _require_isolation() -> None:
    flags = sys.flags
    if (
        flags.isolated != 1
        or flags.no_site != 1
        or flags.ignore_environment != 1
        or flags.safe_path is not True
        or flags.dont_write_bytecode != 1
    ):
        raise WorkerError("worker was not launched with exact -I -S -B isolation")
    if (
        Path(sys.executable).resolve(strict=True) != PYTHON_BASE / "bin/python3.11"
        or Path(sys.prefix).resolve(strict=True) != PYTHON_BASE
        or Path(sys.base_prefix).resolve(strict=True) != PYTHON_BASE
        or Path(__file__).resolve(strict=True) != WORKER_PATH
    ):
        raise WorkerError("worker interpreter/source path differs")
    expected_path = [
        str(PYTHON_BASE / "lib/python311.zip"),
        str(STDLIB_ROOT),
        str(STDLIB_ROOT / "lib-dynload"),
    ]
    if sys.path != expected_path:
        raise WorkerError(f"worker initial import path differs: {sys.path!r}")
    python_zip = PYTHON_BASE / "lib/python311.zip"
    if python_zip.exists() or python_zip.is_symlink():
        raise WorkerError("unexpected higher-priority Python stdlib zip exists")
    if any(key.startswith(("PYTHON", "LD_")) for key in os.environ):
        raise WorkerError("worker environment contains Python/loader overrides")


def _add_site_packages() -> None:
    if SITE_PACKAGES.is_symlink() or not SITE_PACKAGES.is_dir():
        raise WorkerError("retained site-packages root is unavailable")
    sys.path.append(str(SITE_PACKAGES))


def _inside(path: Path, root: Path, label: str) -> Path:
    if path.is_symlink():
        raise WorkerError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise WorkerError(f"{label} escapes its isolated root") from error
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise WorkerError(f"{label} is not a regular file")
    return resolved


def _metrics(message: str) -> tuple[float, float]:
    if not isinstance(message, str):
        raise WorkerError("organizer message is not a string")
    matches = METRIC_RE.findall(message)
    if len(matches) != 1:
        raise WorkerError("organizer message has no unique diff/rel_error")
    values: list[float] = []
    for raw in matches[0]:
        try:
            value = float(raw.strip())
        except ValueError as error:
            raise WorkerError("organizer metric is not numeric") from error
        if not math.isfinite(value) or value < 0:
            raise WorkerError("organizer metric is not finite and nonnegative")
        values.append(value)
    return values[0], values[1]


def _native_dependencies() -> dict[str, str]:
    dependencies: dict[str, str] = {}
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WorkerError("could not inspect process mappings") from error
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1]:
            continue
        value = fields[5]
        if not value.startswith("/"):
            continue
        if value.endswith(" (deleted)"):
            raise WorkerError(f"executable mapping was deleted: {value}")
        path = Path(value)
        try:
            resolved = path.resolve(strict=True)
            mode = resolved.stat().st_mode
        except OSError as error:
            raise WorkerError(f"mapped dependency is unavailable: {path}") from error
        if not stat.S_ISREG(mode):
            raise WorkerError(f"mapped executable is not a regular file: {resolved}")
        dependencies[str(resolved)] = _sha256(resolved)
    if str(Path(sys.executable).resolve(strict=True)) not in dependencies:
        raise WorkerError("embedded Python is absent from executable mappings")
    return dict(sorted(dependencies.items()))


def _write_json_immutable(path: Path, value: object, label: str) -> None:
    if path.is_absolute() is False or path.parent.resolve(strict=True) != Path.cwd():
        raise WorkerError(f"{label} output path is outside worker cwd")
    if path.exists() or path.is_symlink():
        raise WorkerError(f"{label} output already exists")
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _probe() -> dict[str, Any]:
    _add_site_packages()
    import onnxruntime as ort  # noqa: PLC0415
    from onnxruntime.capi import (  # noqa: PLC0415
        onnxruntime_pybind11_state as pybind_state,
    )

    installed = {name: importlib.metadata.version(name) for name in PINNED_REQUIREMENTS}
    return {
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_version": platform.python_version(),
        "prefix": str(Path(sys.prefix).resolve(strict=True)),
        "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        "installed_versions": installed,
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "available_providers": ort.get_available_providers(),
        "ort_pybind_path": str(Path(pybind_state.__file__).resolve(strict=True)),
        "sys_path": list(sys.path),
    }


def _check(request: dict[str, Any]) -> dict[str, Any]:
    if set(request) != REQUEST_KEYS or request.get("protocol") != PROTOCOL:
        raise WorkerError("checker request does not match the exact protocol")
    if (
        request.get("abs_tolerance") != COUNTEREXAMPLE_ATOL
        or request.get("rel_tolerance") != COUNTEREXAMPLE_RTOL
    ):
        raise WorkerError("checker request tolerances differ")
    cwd = Path.cwd().resolve(strict=True)
    onnx_path = _inside(Path(request["onnx_path"]), cwd, "ONNX")
    vnnlib_path = _inside(Path(request["vnnlib_path"]), cwd, "VNN-LIB")
    counterexample_path = _inside(
        Path(request["counterexample_path"]), cwd, "counterexample"
    )
    native_path = Path(request["native_dependencies_path"])
    receipt_path = Path(request["receipt_path"])
    if native_path == receipt_path:
        raise WorkerError("worker evidence output paths collide")
    for path, label in (
        (native_path, "native-dependency"),
        (receipt_path, "receipt"),
    ):
        if path.exists() or path.is_symlink():
            raise WorkerError(f"{label} output already exists")

    onnx_receipt = _file_receipt(onnx_path)
    vnnlib_receipt = _file_receipt(vnnlib_path)
    counterexample_receipt = _file_receipt(counterexample_path)

    _add_site_packages()
    import onnxruntime as ort  # noqa: PLC0415

    sessions: list[list[str]] = []
    original_session = ort.InferenceSession

    def checked_session(*args: Any, **kwargs: Any) -> Any:
        session = original_session(*args, **kwargs)
        providers = session.get_providers()
        if providers != [CPU_PROVIDER]:
            raise WorkerError(
                f"organizer checker selected non-CPU providers: {providers}"
            )
        sessions.append(providers)
        return session

    ort.InferenceSession = checked_session
    sys.path.insert(0, str(SCORING_ROOT))
    from counterexamples import get_ce_diff  # noqa: PLC0415
    from settings import Settings  # noqa: PLC0415

    if (
        Settings.IGNORE_CE_Y is not False
        or Settings.COUNTEREXAMPLE_ATOL != COUNTEREXAMPLE_ATOL
        or Settings.COUNTEREXAMPLE_RTOL != COUNTEREXAMPLE_RTOL
    ):
        raise WorkerError("imported organizer settings differ")
    for module_name in ("counterexamples", "settings", "vnnlib"):
        module = sys.modules.get(module_name)
        module_path = Path(getattr(module, "__file__", "")).resolve(strict=True)
        if module_path.parent != SCORING_ROOT:
            raise WorkerError(
                f"organizer module escaped retained source: {module_name}"
            )

    result, message = get_ce_diff(
        str(onnx_path),
        str(vnnlib_path),
        str(counterexample_path),
        COUNTEREXAMPLE_ATOL,
        COUNTEREXAMPLE_RTOL,
    )
    if not sessions or any(value != [CPU_PROVIDER] for value in sessions):
        raise WorkerError("organizer checker did not execute on the CPU provider")
    if result not in KNOWN_RESULTS:
        raise WorkerError(f"unknown organizer result: {result!r}")
    diff, rel_error = _metrics(message)
    response = {
        "result": result,
        "message": message,
        "diff": diff,
        "rel_error": rel_error,
    }
    if set(response) != RESPONSE_KEYS:
        raise WorkerError("internal response shape differs")
    if (
        _file_receipt(onnx_path) != onnx_receipt
        or _file_receipt(vnnlib_path) != vnnlib_receipt
        or _file_receipt(counterexample_path) != counterexample_receipt
    ):
        raise WorkerError("checker input changed during organizer replay")
    native_dependencies = _native_dependencies()
    request_binding = {
        "protocol": PROTOCOL,
        "abs_tolerance": COUNTEREXAMPLE_ATOL,
        "rel_tolerance": COUNTEREXAMPLE_RTOL,
        "onnx": onnx_receipt,
        "vnnlib": vnnlib_receipt,
        "counterexample": counterexample_receipt,
    }
    receipt = {
        "protocol": PROTOCOL,
        "request_sha256": _canonical_sha256(request_binding),
        "onnx": onnx_receipt,
        "vnnlib": vnnlib_receipt,
        "counterexample": counterexample_receipt,
        "response_sha256": _canonical_sha256(response),
        "native_dependencies_sha256": _canonical_sha256(native_dependencies),
    }
    if set(receipt) != WORKER_RECEIPT_KEYS or any(
        set(receipt[label]) != FILE_RECEIPT_KEYS
        for label in ("onnx", "vnnlib", "counterexample")
    ):
        raise WorkerError("internal worker receipt shape differs")
    _write_json_immutable(native_path, native_dependencies, "native-dependency")
    _write_json_immutable(receipt_path, receipt, "receipt")
    return response


def main() -> int:
    try:
        _require_isolation()
        if sys.argv[1:] == ["--probe"]:
            response = _probe()
        elif sys.argv[1:] == ["--check"]:
            response = _check(_json(sys.stdin.buffer.read()))
        else:
            raise WorkerError("worker mode must be exactly --probe or --check")
        sys.stdout.write(json.dumps(response, sort_keys=True, allow_nan=False) + "\n")
        return 0
    except Exception as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
