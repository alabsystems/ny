#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Canonicalize only the Y values of a VNN-COMP SAT result.

The VNN-COMP 2025 zero-tolerance scorer compares the textual Y values with
outputs from its pinned ONNX Runtime.  Different ONNX Runtime releases or CPU
architectures can disagree by a few float32 ULPs even when the X assignment is
identical.  This tool preserves every X numeric token, evaluates that assignment
with a caller-pinned scorer runtime, and writes a new result plus a provenance
receipt.  It never edits the source result in place.

The canonicalized result is not a new counterexample search result.  Its X
assignment is exactly the solver assignment; only redundant Y annotations are
replaced.  VNN-COMP 2026 ignores those annotations and replays X itself.

TOL=0 REPLAY GATE (mandatory, fail-closed): Y-recomputation can CREATE
invalidity on low-margin rows — the recomputed outputs may land on the safe
side of the property boundary even though the solver outputs violated it.
Every canonicalization therefore replays the canonical witness through the
official VNN-COMP SCORING checker at zero tolerance and REFUSES to publish
unless (1) the written canonical Y values still violate the property at
input_tol=0/output_tol=0 and (2) the official checker judges the canonical
result file strictly `correct` at abs_tol=0/rel_tol=0.  The gate cannot be
skipped: `--vnnlib` and `--scoring-dir` are required arguments.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENTRY_RE = re.compile(r"\(\s*([XY])_(\d+)\s+([^\s()]+)\s*\)")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCHEMA = "ny_vnncomp_ce_y_canonicalization_v2"
REPLAY_GATE_SCHEMA = "ny_vnncomp_ce_y_tol0_replay_gate_v1"
# Sibling modules the official checker imports by name; hashed into the
# receipt so the gate verdict is pinned to checker bytes, not a directory name.
#
# The layout differs by edition: 2025 SCORING-* ships `vnnlib.py` and no
# `cex_checks.py`; 2026 SCORING ships `vnnlib_v1.py` and `cex_checks.py`.
# The load-bearing pair is required in every edition; the variants are sealed
# whenever present, and at least one vnnlib parser sibling must exist (the
# checker cannot parse properties without it).
OFFICIAL_CHECKER_FILES = (
    "counterexamples.py",
    "settings.py",
)
OFFICIAL_CHECKER_OPTIONAL_FILES = (
    "cex_checks.py",
    "vnnlib.py",
    "vnnlib_v1.py",
)
OFFICIAL_CHECKER_VNNLIB_SIBLINGS = ("vnnlib.py", "vnnlib_v1.py")
OFFICIAL_BENCHMARK_REPO_RE = re.compile(r"^vnncomp\d{4}_benchmarks$")


class CanonicalizationError(RuntimeError):
    """The result or canonicalizer environment failed a closed-world check."""


@dataclass(frozen=True)
class Assignment:
    x_tokens: tuple[str, ...]
    y_tokens: tuple[str, ...]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _absolute_lexical_path(path: Path, label: str) -> Path:
    """Return an absolute lexical path without resolving any component."""

    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise CanonicalizationError(f"{label} must not contain '..': {candidate}")
    if not candidate.name:
        raise CanonicalizationError(f"{label} must name a file: {candidate}")
    return candidate


def _open_parent_directory(
    path: Path,
    *,
    label: str,
    create: bool,
) -> int:
    """Open a path's parent one no-follow component at a time."""

    if os.name != "posix" or not all(
        hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW")
    ):
        raise CanonicalizationError(
            "symlink-safe path traversal requires POSIX O_DIRECTORY and O_NOFOLLOW"
        )
    path = _absolute_lexical_path(path, label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    current = Path(path.anchor)
    try:
        for component in path.parent.parts[1:]:
            current /= component
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent creator won. The no-follow open below decides
                    # whether it created the required directory or a hostile link.
                    pass
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as error:
                    if error.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise CanonicalizationError(
                            f"{label} contains a symlink or non-directory component: "
                            f"{current}"
                        ) from error
                    raise
            except OSError as error:
                if error.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise CanonicalizationError(
                        f"{label} contains a symlink or non-directory component: "
                        f"{current}"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stable_read(path: Path) -> tuple[bytes, dict[str, int | str]]:
    path = _absolute_lexical_path(path, "input path")
    parent = _open_parent_directory(path, label="input path", create=False)
    try:
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise CanonicalizationError(
                    f"input path contains a symlink component: {path}"
                ) from error
            raise
    finally:
        os.close(parent)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise CanonicalizationError(f"input is not a regular file: {path}")
        data = stream.read()
        after = os.fstat(stream.fileno())
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if fingerprint_before != fingerprint_after:
        raise CanonicalizationError(f"file changed while it was read: {path}")
    return data, {
        "path": str(path),
        "sha256": _sha256(data),
        "size_bytes": len(data),
    }


def _finite_token(token: str, label: str) -> None:
    try:
        value = float(token)
    except ValueError as error:
        raise CanonicalizationError(f"{label} is not a floating-point token") from error
    if not math.isfinite(value):
        raise CanonicalizationError(f"{label} must be finite")


def parse_sat_result(data: bytes) -> Assignment:
    """Parse the strict SMT-LIB-shaped result syntax emitted by NY."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CanonicalizationError("result is not UTF-8") from error
    lines = text.splitlines()
    if not lines or lines[0].strip().lower() != "sat":
        raise CanonicalizationError("result must start with a standalone SAT verdict")
    assignment_text = "\n".join(lines[1:]).strip()
    if len(assignment_text) < 2 or not (
        assignment_text.startswith("(") and assignment_text.endswith(")")
    ):
        raise CanonicalizationError("SAT result has no outer assignment list")

    body = assignment_text[1:-1]
    x_tokens: list[str] = []
    y_tokens: list[str] = []
    saw_y = False
    cursor = 0
    for match in ENTRY_RE.finditer(body):
        if body[cursor : match.start()].strip():
            raise CanonicalizationError("assignment contains unsupported syntax")
        namespace, index_text, token = match.groups()
        index = int(index_text)
        if namespace == "X":
            if saw_y:
                raise CanonicalizationError("X entries must precede all Y entries")
            if index != len(x_tokens):
                raise CanonicalizationError(
                    f"X indices must be contiguous from zero; got X_{index}"
                )
            _finite_token(token, f"X_{index}")
            x_tokens.append(token)
        else:
            saw_y = True
            if index != len(y_tokens):
                raise CanonicalizationError(
                    f"Y indices must be contiguous from zero; got Y_{index}"
                )
            _finite_token(token, f"Y_{index}")
            y_tokens.append(token)
        cursor = match.end()
    if body[cursor:].strip():
        raise CanonicalizationError("assignment contains trailing unsupported syntax")
    if not x_tokens:
        raise CanonicalizationError("assignment contains no X entries")
    if not y_tokens:
        raise CanonicalizationError("assignment contains no Y entries")
    return Assignment(tuple(x_tokens), tuple(y_tokens))


def render_sat_result(assignment: Assignment, outputs: Sequence[float]) -> bytes:
    """Render canonical Y decimals while retaining every original X token."""

    if len(outputs) != len(assignment.y_tokens):
        raise CanonicalizationError(
            "runtime output arity does not match the solver Y arity: "
            f"{len(outputs)} != {len(assignment.y_tokens)}"
        )
    lines = [f"(X_{index} {token})" for index, token in enumerate(assignment.x_tokens)]
    for index, value in enumerate(outputs):
        promoted = float(value)
        if not math.isfinite(promoted):
            raise CanonicalizationError(f"runtime returned non-finite Y_{index}")
        # repr(float) is the shortest decimal that round-trips to the exact
        # binary64 promotion.  The 2025 checker parses textual Y as binary64.
        lines.append(f"(Y_{index} {promoted!r})")
    return ("sat\n(" + "\n".join(lines) + ")\n").encode("utf-8")


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        import numpy as np  # noqa: PLC0415
        import onnx  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as error:
        raise CanonicalizationError(
            "numpy, onnx, and onnxruntime are required in the selected interpreter"
        ) from error
    return np, onnx, ort


def _model_input(model: Any) -> Any:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [
        value for value in model.graph.input if value.name not in initializer_names
    ]
    if len(inputs) != 1:
        raise CanonicalizationError(
            f"canonicalizer requires exactly one model input, found {len(inputs)}"
        )
    return inputs[0]


def _model_output(model: Any) -> Any:
    outputs = list(model.graph.output)
    if len(outputs) != 1:
        raise CanonicalizationError(
            f"canonicalizer requires exactly one model output, found {len(outputs)}"
        )
    return outputs[0]


def canonical_outputs(
    model_bytes: bytes,
    assignment: Assignment,
    *,
    required_ort_version: str,
    required_provider: str,
) -> tuple[list[float], dict[str, Any]]:
    """Evaluate X using the scorer-compatible ONNX Runtime configuration."""

    np, onnx, ort = _load_runtime()
    if ort.__version__ != required_ort_version:
        raise CanonicalizationError(
            "onnxruntime version mismatch: "
            f"required {required_ort_version}, found {ort.__version__}"
        )

    model = onnx.load_model_from_string(model_bytes)
    model_input = _model_input(model)
    model_output = _model_output(model)
    dimensions = model_input.type.tensor_type.shape.dim
    shape = tuple(dimension.dim_value if dimension.dim_value != 0 else 1 for dimension in dimensions)
    expected = math.prod(shape)
    if len(assignment.x_tokens) != expected:
        raise CanonicalizationError(
            f"X arity {len(assignment.x_tokens)} does not match model shape {shape}"
        )
    input_element_type = model_input.type.tensor_type.elem_type
    if input_element_type == onnx.TensorProto.FLOAT:
        input_dtype = np.float32
    elif input_element_type == onnx.TensorProto.DOUBLE:
        input_dtype = np.float64
    else:
        raise CanonicalizationError(
            "unsupported ONNX input dtype: the official checker accepts only "
            "FLOAT and DOUBLE"
        )
    x_values = np.asarray(
        [float(token) for token in assignment.x_tokens], dtype=input_dtype
    ).reshape(shape, order="C")

    available_providers = list(ort.get_available_providers())
    if required_provider not in available_providers:
        raise CanonicalizationError(
            f"required provider {required_provider!r} is unavailable: "
            f"{available_providers!r}"
        )
    options = ort.SessionOptions()
    # Match VNN-COMP 2025 SCORING-*/counterexamples.py.
    options.intra_op_num_threads = 12
    options.inter_op_num_threads = 12
    session = ort.InferenceSession(
        model.SerializeToString(),
        options,
        providers=[required_provider],
    )
    providers = list(session.get_providers())
    if providers != [required_provider]:
        raise CanonicalizationError(
            f"required provider {required_provider!r} was not selected exclusively; "
            f"session providers are {providers!r}"
        )
    session_inputs = session.get_inputs()
    if len(session_inputs) != 1:
        raise CanonicalizationError(
            f"runtime session has {len(session_inputs)} inputs instead of one"
        )
    if session_inputs[0].name != model_input.name:
        raise CanonicalizationError(
            "runtime input does not match the sole non-initializer graph input: "
            f"{session_inputs[0].name!r} != {model_input.name!r}"
        )
    session_outputs = session.get_outputs()
    if len(session_outputs) != 1:
        raise CanonicalizationError(
            f"runtime session has {len(session_outputs)} outputs instead of one"
        )
    if session_outputs[0].name != model_output.name:
        raise CanonicalizationError(
            "runtime output does not match the sole graph output: "
            f"{session_outputs[0].name!r} != {model_output.name!r}"
        )
    results = session.run(None, {session_inputs[0].name: x_values})
    if len(results) != 1:
        raise CanonicalizationError(
            f"runtime returned {len(results)} output tensors instead of one"
        )
    output_array = np.asarray(results[0])
    try:
        declared_output_dtype = np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(
                model_output.type.tensor_type.elem_type
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CanonicalizationError("unsupported ONNX output dtype") from error
    actual_output_dtype = np.dtype(output_array.dtype)
    if actual_output_dtype != declared_output_dtype:
        raise CanonicalizationError(
            "runtime output dtype does not match the graph declaration: "
            f"{actual_output_dtype} != {declared_output_dtype}"
        )
    if actual_output_dtype.kind not in "iuf":
        raise CanonicalizationError(
            f"runtime output dtype must be real numeric, found {actual_output_dtype}"
        )
    flat = output_array.flatten(order="C")
    outputs = [float(value) for value in flat]
    numpy_origin = Path(np.__file__).resolve()
    onnxruntime_origin = Path(ort.__file__).resolve()
    if not numpy_origin.is_file() or not onnxruntime_origin.is_file():
        raise CanonicalizationError(
            "runtime package origins must resolve to regular files"
        )
    return outputs, {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numpy_origin": str(numpy_origin),
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_origin": str(onnxruntime_origin),
        "available_providers": available_providers,
        "session_providers": providers,
        "selected_provider": required_provider,
        "selected_output_index": 0,
        "input_name": model_input.name,
        "input_dtype": str(np.dtype(input_dtype)),
        "input_shape": list(shape),
        "output_name": model_output.name,
        "output_dtype": str(actual_output_dtype),
        "output_shape": list(output_array.shape),
    }


def _load_official_checker(scoring_dir: Path) -> Path:
    """Resolve the checker directory without importing it in this process.

    Official checker modules use bare sibling imports. Importing them here
    would let an unrelated/poisoned parent ``sys.modules['settings']`` (or
    another sibling name) silently supply code from outside the sealed SCORING
    directory. The actual import and both validation calls happen in a fresh
    isolated interpreter in :func:`tol0_replay_gate`.
    """

    scoring_dir = scoring_dir.expanduser().resolve()
    counterexamples = scoring_dir / "counterexamples.py"
    if not scoring_dir.is_dir() or not counterexamples.is_file():
        raise CanonicalizationError(
            f"official SCORING directory is unavailable at {scoring_dir}; "
            "the tol=0 replay gate cannot run, refusing to canonicalize"
        )
    return scoring_dir


_OFFICIAL_CHECKER_WORKER = r"""
import contextlib
import importlib
import inspect
import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
scoring_dir = Path(payload["scoring_dir"]).resolve()
counterexamples_path = (scoring_dir / "counterexamples.py").resolve()
stage = "runtime binding"
try:
    # The interpreter is launched with -I, and these names are purged before
    # the sole trusted directory is prepended. This is intentionally stronger
    # than a temporary sys.path edit in the long-lived parent process.
    for name in ("counterexamples", "settings", "cex_checks", "vnnlib", "vnnlib_v1"):
        sys.modules.pop(name, None)
    isolated_stdlib_paths = [
        entry
        for entry in sys.path
        if entry
        and not any(
            component in {"site-packages", "dist-packages"}
            for component in Path(entry).parts
        )
    ]
    # Rebuild instead of extending: `-I` can retain a system site-packages root
    # ahead of a venv/user installation. The parent-authenticated package roots
    # must win in the exact declared order or the origin check below declines.
    sys.path[:] = [str(scoring_dir), *payload["dependency_paths"], *isolated_stdlib_paths]
    with contextlib.redirect_stdout(sys.stderr):
        numpy = importlib.import_module("numpy")
        onnxruntime = importlib.import_module("onnxruntime")
    runtime = payload["runtime_binding"]
    numpy_origin = Path(getattr(numpy, "__file__", "")).resolve()
    onnxruntime_origin = Path(getattr(onnxruntime, "__file__", "")).resolve()
    if str(getattr(numpy, "__version__", "")) != runtime["numpy_version"]:
        raise RuntimeError(
            "numpy version mismatch: "
            + repr(getattr(numpy, "__version__", None))
            + " != "
            + repr(runtime["numpy_version"])
        )
    if str(numpy_origin) != runtime["numpy_origin"]:
        raise RuntimeError(
            "numpy origin mismatch: "
            + repr(str(numpy_origin))
            + " != "
            + repr(runtime["numpy_origin"])
        )
    if str(getattr(onnxruntime, "__version__", "")) != runtime["onnxruntime_version"]:
        raise RuntimeError(
            "onnxruntime version mismatch: "
            + repr(getattr(onnxruntime, "__version__", None))
            + " != "
            + repr(runtime["onnxruntime_version"])
        )
    if str(onnxruntime_origin) != runtime["onnxruntime_origin"]:
        raise RuntimeError(
            "onnxruntime origin mismatch: "
            + repr(str(onnxruntime_origin))
            + " != "
            + repr(runtime["onnxruntime_origin"])
        )
    available_providers = list(onnxruntime.get_available_providers())
    if available_providers != runtime["available_providers"]:
        raise RuntimeError(
            "onnxruntime provider inventory mismatch: "
            + repr(available_providers)
            + " != "
            + repr(runtime["available_providers"])
        )
    required_provider = runtime["required_provider"]
    if required_provider not in available_providers:
        raise RuntimeError(
            "required provider unavailable in checker worker: "
            + repr(required_provider)
        )
    with contextlib.redirect_stdout(sys.stderr):
        provider_probe = onnxruntime.InferenceSession(
            payload["model_path"], providers=[required_provider]
        )
    selected_providers = list(provider_probe.get_providers())
    if selected_providers != [required_provider]:
        raise RuntimeError(
            "checker worker did not select the required provider exclusively: "
            + repr(selected_providers)
        )

    stage = "checker import"
    with contextlib.redirect_stdout(sys.stderr):
        checker = importlib.import_module("counterexamples")
    module_path = Path(getattr(checker, "__file__", "")).resolve()
    if module_path != counterexamples_path:
        raise RuntimeError(
            "counterexamples resolved outside configured SCORING directory: "
            + str(module_path)
        )
    for name in ("get_ce_diff", "is_specification_vio"):
        if not callable(getattr(checker, name, None)):
            raise RuntimeError(f"counterexamples.py does not define callable {name}")

    x_values = tuple(float(value) for value in payload["x_values"])
    y_values = tuple(float(value) for value in payload["y_values"])
    stage = "written-witness check"
    with contextlib.redirect_stdout(sys.stderr):
        sig_params = inspect.signature(checker.is_specification_vio).parameters
        if "input_tol" in sig_params and "output_tol" in sig_params:
            violated, witness_message = checker.is_specification_vio(
                payload["model_path"],
                payload["vnnlib_path"],
                x_values,
                y_values,
                input_tol=0.0,
                output_tol=0.0,
            )
        elif "tol" in sig_params:
            violated, witness_message = checker.is_specification_vio(
                payload["model_path"],
                payload["vnnlib_path"],
                x_values,
                y_values,
                0.0,
            )
        else:
            raise RuntimeError(
                "unrecognized is_specification_vio signature "
                + repr(sorted(sig_params))
            )
    result = {
        "written_violated": bool(violated),
        "written_message": str(witness_message),
    }
    if bool(violated):
        stage = "official replay check"
        with contextlib.redirect_stdout(sys.stderr):
            replay_result, replay_message = checker.get_ce_diff(
                payload["model_path"],
                payload["vnnlib_path"],
                payload["ce_path"],
                0.0,
                0.0,
            )
        result["replay_value"] = str(getattr(replay_result, "value", replay_result))
        result["replay_message"] = str(replay_message)
    print(json.dumps(result, sort_keys=True))
except BaseException as error:
    print(json.dumps({"error_stage": stage, "error": repr(error)}, sort_keys=True))
"""


_TRUSTED_PYTHON_INSTALL_PATHS_PROBE = r"""
import json
import site
import sysconfig

paths = []

def add(value):
    if isinstance(value, str):
        paths.append(value)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            add(entry)

for key in ("purelib", "platlib"):
    try:
        add(sysconfig.get_path(key))
    except (KeyError, TypeError):
        pass
try:
    add(site.getusersitepackages())
except (AttributeError, OSError):
    pass
try:
    add(site.getsitepackages())
except (AttributeError, OSError):
    pass

print(json.dumps(paths))
"""


def _isolated_python_environment() -> dict[str, str]:
    """Environment for an interpreter that cannot inherit Python path policy."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
    }
    if os.name == "posix":
        # `site.getusersitepackages()` consults HOME even under `python -I`.
        # Bind it to the OS account rather than an inherited, mutable HOME so
        # PYTHONUSERBASE/HOME cannot manufacture a checker dependency root.
        import pwd  # noqa: PLC0415

        environment["HOME"] = pwd.getpwuid(os.getuid()).pw_dir
    return environment


def _trusted_python_install_paths() -> list[str]:
    """Return interpreter-declared package roots, independent of ``sys.path``.

    The parent may have been launched with PYTHONPATH.  Filtering its live
    ``sys.path`` by a directory name such as ``site-packages`` would bless an
    attacker-selected path and partially undo the worker's ``-I`` isolation.
    Ask the same interpreter, in a scrubbed isolated process, for only its
    sysconfig/site install schemes instead.  The default user install root is
    retained because supported scorer environments commonly install numpy,
    ONNX Runtime, and cachier there rather than inside the venv.
    """

    completed = subprocess.run(
        [sys.executable, "-I", "-c", _TRUSTED_PYTHON_INSTALL_PATHS_PROBE],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_isolated_python_environment(),
        check=False,
    )
    if completed.returncode != 0:
        raise CanonicalizationError(
            "could not resolve isolated Python dependency roots: "
            f"interpreter exited {completed.returncode}"
        )
    try:
        reported = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise CanonicalizationError(
            "isolated Python dependency-root probe returned malformed JSON"
        ) from error
    if not isinstance(reported, list) or not all(
        isinstance(entry, str) for entry in reported
    ):
        raise CanonicalizationError(
            "isolated Python dependency-root probe returned an invalid path list"
        )

    trusted: list[str] = []
    seen: set[str] = set()
    for entry in reported:
        path = Path(entry)
        if not path.is_absolute() or not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved not in seen:
            trusted.append(resolved)
            seen.add(resolved)
    if not trusted:
        raise CanonicalizationError(
            "isolated Python dependency-root probe found no installed package roots"
        )
    return trusted


def _checker_runtime_binding(
    runtime: dict[str, Any],
    *,
    required_ort_version: str,
    required_provider: str,
) -> dict[str, Any]:
    """Extract the parent-authenticated runtime identity for the worker."""

    required_fields = {
        "numpy": str,
        "numpy_origin": str,
        "onnxruntime": str,
        "onnxruntime_origin": str,
        "available_providers": list,
        "session_providers": list,
        "selected_provider": str,
    }
    for field, expected_type in required_fields.items():
        if not isinstance(runtime.get(field), expected_type):
            raise CanonicalizationError(
                f"parent runtime evidence has invalid or missing {field!r}"
            )
    if runtime["onnxruntime"] != required_ort_version:
        raise CanonicalizationError(
            "parent runtime evidence does not match the required ONNX Runtime version"
        )
    if runtime["selected_provider"] != required_provider or runtime[
        "session_providers"
    ] != [required_provider]:
        raise CanonicalizationError(
            "parent runtime evidence does not bind the required provider exclusively"
        )
    for field in ("numpy_origin", "onnxruntime_origin"):
        origin = Path(runtime[field])
        if not origin.is_absolute() or not origin.is_file():
            raise CanonicalizationError(
                f"parent runtime evidence has invalid {field!r}"
            )
    providers = runtime["available_providers"]
    if not all(isinstance(provider, str) for provider in providers):
        raise CanonicalizationError(
            "parent runtime evidence has invalid provider inventory"
        )
    return {
        "numpy_version": runtime["numpy"],
        "numpy_origin": str(Path(runtime["numpy_origin"]).resolve()),
        "onnxruntime_version": runtime["onnxruntime"],
        "onnxruntime_origin": str(Path(runtime["onnxruntime_origin"]).resolve()),
        "available_providers": providers,
        "required_provider": required_provider,
    }


def _snapshot_official_checker(
    scoring_dir: Path,
) -> tuple[tempfile.TemporaryDirectory[str], Path, dict[str, Any]]:
    """Stable-read and stage exactly the checker source bytes we execute.

    The official 2025/2026 ``settings.py`` derives its benchmark checkout from
    ``__file__`` by walking from ``<layout>/<results-repo>/<SCORING>`` to a
    sibling ``<layout>/vnncompYYYY_benchmarks`` directory.  A flat snapshot
    changes that meaning and makes the genuine checker fail during import.
    Preserve the two-level results-repo/SCORING layout and expose only the
    explicitly named benchmark-repository siblings.  Checker imports still
    resolve solely from the private source snapshot; the links supply data
    layout, never Python modules or unsealed checker source.
    """

    scoring_dir = scoring_dir.expanduser().resolve()
    snapshot = tempfile.TemporaryDirectory(prefix="ny-ce-y-official-checker-")
    snapshot_root = Path(snapshot.name)
    source_layout_root = scoring_dir.parent.parent
    staged_dir = snapshot_root / scoring_dir.parent.name / scoring_dir.name
    staged_dir.mkdir(parents=True)
    benchmark_repos: dict[str, str] = {}
    try:
        layout_entries = sorted(
            source_layout_root.iterdir(), key=lambda path: path.name
        )
    except OSError as error:
        snapshot.cleanup()
        raise CanonicalizationError(
            f"could not inspect official checker layout root: {source_layout_root}"
        ) from error
    for entry in layout_entries:
        if OFFICIAL_BENCHMARK_REPO_RE.fullmatch(entry.name) is None:
            continue
        try:
            target = entry.resolve(strict=True)
        except OSError as error:
            snapshot.cleanup()
            raise CanonicalizationError(
                f"official benchmark repository is unavailable: {entry}"
            ) from error
        if not target.is_dir():
            snapshot.cleanup()
            raise CanonicalizationError(
                f"official benchmark repository is not a directory: {entry}"
            )
        staged_repo = snapshot_root / entry.name
        try:
            staged_repo.symlink_to(target, target_is_directory=True)
        except OSError as error:
            snapshot.cleanup()
            raise CanonicalizationError(
                f"could not preserve official checker benchmark layout for {entry}"
            ) from error
        benchmark_repos[entry.name] = str(target)
    files: dict[str, Any] = {}
    for name in OFFICIAL_CHECKER_FILES:
        path = scoring_dir / name
        if not path.is_file():
            snapshot.cleanup()
            raise CanonicalizationError(
                f"official checker file missing from SCORING directory: {path}"
            )
        data, _ = _stable_read(path)
        (staged_dir / name).write_bytes(data)
        files[name] = {
            "sha256": _sha256(data),
            "size_bytes": len(data),
        }
    for name in OFFICIAL_CHECKER_OPTIONAL_FILES:
        path = scoring_dir / name
        if path.is_file():
            data, _ = _stable_read(path)
            (staged_dir / name).write_bytes(data)
            files[name] = {
                "sha256": _sha256(data),
                "size_bytes": len(data),
            }
    if not any(name in files for name in OFFICIAL_CHECKER_VNNLIB_SIBLINGS):
        snapshot.cleanup()
        raise CanonicalizationError(
            "no vnnlib parser sibling (vnnlib.py / vnnlib_v1.py) in SCORING "
            f"directory {scoring_dir}; the official checker cannot run without one"
        )
    evidence = {
        "scoring_dir": str(scoring_dir),
        "files": files,
        "benchmark_repositories": benchmark_repos,
    }
    return snapshot, staged_dir, evidence


_GATE_WORKDIR: Path | None = None


def _gate_workdir() -> Path:
    """Per-process working directory for the official checker.

    The official checker memoizes via cachier with a relative `./cachier`
    directory that is resolved once and reused for the process lifetime, so
    the gate's cwd must outlive every gate call — a per-call temp cwd crashes
    the second call.  Model/vnnlib/counterexample files still live in unique
    per-call temp directories, so the checker's argument-hash memoization can
    never return a stale verdict for different bytes.
    """

    global _GATE_WORKDIR
    if _GATE_WORKDIR is None:
        _GATE_WORKDIR = Path(tempfile.mkdtemp(prefix="ny-ce-y-tol0-gate-cwd-"))
    return _GATE_WORKDIR


def tol0_replay_gate(
    checker: Path,
    *,
    model_bytes: bytes,
    vnnlib_bytes: bytes,
    assignment: Assignment,
    canonical_outputs_list: Sequence[float],
    output_bytes: bytes,
    runtime: dict[str, Any],
    required_ort_version: str,
    required_provider: str,
) -> dict[str, Any]:
    """Refuse the canonicalization unless the canonical witness still violates.

    Two independent zero-tolerance checks, both of which must pass:

    1. WRITTEN-WITNESS CHECK — the Y tokens that will be published must
       themselves violate the property at input_tol=0, output_tol=0
       (`is_specification_vio` on the canonical outputs). This is the check a
       broken canonicalization fails: if Y-recomputation crosses a low-margin
       property boundary, the published witness stops being a counterexample.
    2. OFFICIAL REPLAY CHECK — the canonical result file must be judged
       strictly CORRECT by the official checker at abs_tol=0, rel_tol=0
       (`get_ce_diff`, which ignores the file's Y and replays X through the
       network). This is the scorer's own semantics; CORRECT_UP_TO_TOLERANCE
       is NOT accepted.

    All checker work happens on private temp copies of the exact bytes that
    were hashed, inside a temp cwd (the official checker writes relative
    diagnostic files), with checker stdout diverted to stderr.
    """

    # Tuples, not lists: the official is_specification_vio is memoized by
    # cachier, which hashes its arguments (get_ce_diff passes tuples too).
    x_values = tuple(float(token) for token in assignment.x_tokens)
    y_values = tuple(float(value) for value in canonical_outputs_list)
    with tempfile.TemporaryDirectory(prefix="ny-ce-y-tol0-gate-") as raw:
        temp = Path(raw)
        model_path = temp / "model.onnx"
        vnnlib_path = temp / "property.vnnlib"
        ce_path = temp / "canonical.counterexample"
        model_path.write_bytes(model_bytes)
        vnnlib_path.write_bytes(vnnlib_bytes)
        # The official checker consumes the bare assignment list (the
        # `.counterexample` witness body); the leading `sat` verdict line is
        # the result-file framing and is not part of the witness.
        try:
            _verdict_line, assignment_body = output_bytes.decode("utf-8").split(
                "\n", 1
            )
        except ValueError as error:
            raise CanonicalizationError(
                "canonical result has no assignment body to replay"
            ) from error
        ce_path.write_bytes(assignment_body.encode("utf-8"))
        payload = {
            "scoring_dir": str(checker),
            "model_path": str(model_path),
            "vnnlib_path": str(vnnlib_path),
            "ce_path": str(ce_path),
            "x_values": x_values,
            "y_values": y_values,
            "dependency_paths": _trusted_python_install_paths(),
            "runtime_binding": _checker_runtime_binding(
                runtime,
                required_ort_version=required_ort_version,
                required_provider=required_provider,
            ),
        }
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _OFFICIAL_CHECKER_WORKER],
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            cwd=_gate_workdir(),
            env=_isolated_python_environment(),
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            raise CanonicalizationError(
                "tol=0 replay gate: isolated official checker process failed "
                f"with exit {completed.returncode}"
            )
        try:
            isolated = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise CanonicalizationError(
                "tol=0 replay gate: isolated official checker returned malformed JSON"
            ) from error
        if "error" in isolated:
            raise CanonicalizationError(
                "tol=0 replay gate: "
                f"{isolated.get('error_stage', 'official checker')} raised: "
                f"{isolated['error']}"
            )
        if not isolated.get("written_violated", False):
            raise CanonicalizationError(
                "tol=0 replay gate REFUSED: the canonical Y values no "
                "longer violate the property at zero tolerance; "
                "canonicalization would destroy the witness. "
                + str(isolated.get("written_message", "")).replace("\n", " | ")
            )
    replay_value = str(isolated.get("replay_value"))
    replay_message = str(isolated.get("replay_message", ""))
    if replay_value != "correct":
        raise CanonicalizationError(
            "tol=0 replay gate REFUSED: official checker verdict is "
            f"{replay_value!r}, not 'correct', at abs_tol=0 rel_tol=0. "
            + str(replay_message).replace("\n", " | ")
        )
    return {
        "schema": REPLAY_GATE_SCHEMA,
        "abs_tolerance": 0.0,
        "rel_tolerance": 0.0,
        "input_tolerance": 0.0,
        "output_tolerance": 0.0,
        "written_witness_violates_property": True,
        "official_replay_result": replay_value,
        "official_replay_message": str(replay_message),
    }


def _preflight_destination(path: Path, *, label: str) -> Path:
    path = _absolute_lexical_path(path, label)
    parent = _open_parent_directory(path, label=label, create=True)
    try:
        try:
            status = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return path
        if stat.S_ISLNK(status.st_mode):
            raise CanonicalizationError(
                f"{label} path contains a symlink component: {path}"
            )
        raise CanonicalizationError(f"refusing to overwrite existing {label}: {path}")
    finally:
        os.close(parent)


def _exclusive_write(path: Path, data: bytes, *, label: str) -> None:
    path = _absolute_lexical_path(path, f"{label} path")
    parent = _open_parent_directory(path, label=f"{label} path", create=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent)
    except FileExistsError as error:
        raise CanonicalizationError(
            f"refusing to overwrite existing {label}: {path}"
        ) from error
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise CanonicalizationError(
                f"{label} path contains a symlink component: {path}"
            ) from error
        raise
    finally:
        os.close(parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Preserve the exclusive partial artifact for diagnosis.  Never unlink a
        # path after releasing the exclusive descriptor because another process
        # could have replaced it.
        raise


def canonicalize(
    onnx_path: Path,
    result_path: Path,
    output_path: Path,
    *,
    vnnlib_path: Path,
    scoring_dir: Path,
    receipt_path: Path | None,
    required_ort_version: str,
    required_provider: str,
) -> dict[str, Any]:
    output_path = _absolute_lexical_path(output_path, "output path")
    if receipt_path is not None:
        receipt_path = _absolute_lexical_path(receipt_path, "receipt path")
        if output_path == receipt_path:
            raise CanonicalizationError(
                "output and receipt paths must be distinct"
            )
    model_bytes, model_evidence = _stable_read(onnx_path)
    result_bytes, result_evidence = _stable_read(result_path)
    vnnlib_bytes, vnnlib_evidence = _stable_read(vnnlib_path)
    checker_source = _load_official_checker(scoring_dir)
    checker_snapshot, checker, checker_evidence = _snapshot_official_checker(
        checker_source
    )
    assignment = parse_sat_result(result_bytes)
    try:
        outputs, runtime = canonical_outputs(
            model_bytes,
            assignment,
            required_ort_version=required_ort_version,
            required_provider=required_provider,
        )
        output_bytes = render_sat_result(assignment, outputs)
        replay_gate = tol0_replay_gate(
            checker,
            model_bytes=model_bytes,
            vnnlib_bytes=vnnlib_bytes,
            assignment=assignment,
            canonical_outputs_list=outputs,
            output_bytes=output_bytes,
            runtime=runtime,
            required_ort_version=required_ort_version,
            required_provider=required_provider,
        )
    finally:
        checker_snapshot.cleanup()

    old_y = [float(token) for token in assignment.y_tokens]
    max_y_change = max(abs(old - new) for old, new in zip(old_y, outputs))
    x_token_bytes = ("\n".join(assignment.x_tokens) + "\n").encode("utf-8")
    y_token_bytes = (
        "\n".join(repr(float(value)) for value in outputs) + "\n"
    ).encode("utf-8")
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "canonicalized_at_utc": _utc_now(),
        "source_result": result_evidence,
        "onnx": model_evidence,
        "vnnlib": vnnlib_evidence,
        "official_checker": checker_evidence,
        "replay_gate": replay_gate,
        "output_result": {
            "path": str(output_path),
            "sha256": _sha256(output_bytes),
            "size_bytes": len(output_bytes),
        },
        "assignment": {
            "x_count": len(assignment.x_tokens),
            "y_count": len(assignment.y_tokens),
            "x_numeric_tokens_sha256": _sha256(x_token_bytes),
            "canonical_y_tokens_sha256": _sha256(y_token_bytes),
            "max_absolute_y_change": max_y_change,
        },
        "runtime": runtime,
        "policy": {
            "x_tokens_preserved": True,
            "source_overwritten": False,
            "required_onnxruntime": required_ort_version,
            "required_provider": required_provider,
            "tol0_replay_gate": "required",
        },
    }
    # Keep the model read live until immediately before publication, then prove
    # the path still names the bytes evaluated above.
    current_model, current_model_evidence = _stable_read(onnx_path)
    if current_model != model_bytes or current_model_evidence != model_evidence:
        raise CanonicalizationError("ONNX model changed before publication")
    current_result, current_result_evidence = _stable_read(result_path)
    if current_result != result_bytes or current_result_evidence != result_evidence:
        raise CanonicalizationError("source result changed before publication")
    current_vnnlib, current_vnnlib_evidence = _stable_read(vnnlib_path)
    if current_vnnlib != vnnlib_bytes or current_vnnlib_evidence != vnnlib_evidence:
        raise CanonicalizationError("VNN-LIB property changed before publication")

    receipt_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    _preflight_destination(output_path, label="output")
    if receipt_path is not None:
        _preflight_destination(receipt_path, label="receipt")
    _exclusive_write(output_path, output_bytes, label="output")
    if receipt_path is not None:
        _exclusive_write(receipt_path, receipt_bytes, label="receipt")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vnnlib",
        type=Path,
        required=True,
        help="VNN-LIB property the witness must still violate at tol=0",
    )
    parser.add_argument(
        "--scoring-dir",
        type=Path,
        required=True,
        help=(
            "official vnncomp results SCORING directory providing "
            "counterexamples.py; the tol=0 replay gate refuses to run "
            "without it"
        ),
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--require-onnxruntime",
        required=True,
        help="Exact scorer runtime version, for example 1.16.3",
    )
    parser.add_argument(
        "--require-provider",
        default="CPUExecutionProvider",
        help="Provider that must be the committed session's only provider",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = canonicalize(
            args.onnx,
            args.result,
            args.output,
            vnnlib_path=args.vnnlib,
            scoring_dir=args.scoring_dir,
            receipt_path=args.receipt,
            required_ort_version=args.require_onnxruntime,
            required_provider=args.require_provider,
        )
    except (CanonicalizationError, OSError) as error:
        print(f"canonicalize_vnncomp_ce_y: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
