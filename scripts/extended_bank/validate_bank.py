#!/usr/bin/env python3
"""Validate and bank extended-track NY results.

For every source ``sat`` row, NY is rerun up to three times.  A reproduced SAT
is banked only when its complete raw witness satisfies the full VNN-LIB
property.  Reproduced SAT results and their validation records are retained as
read-only evidence.  Non-SAT source rows retain the historical banking
semantics: ``unsat`` stays ``unsat`` and every other verdict becomes
``unknown``.

Accepted headerless input schemas are:

* legacy extended bank: ``track,onnx,vnnlib,verdict,seconds``;
* measured result: ``track,onnx,vnnlib,prepared,verdict,seconds[,run_id]``.

The output schema remains ``track,onnx,vnnlib,verdict,seconds``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RETRIES = 3
SOLVED_VERDICTS = frozenset({"sat", "unsat"})
VERDICTS = frozenset({"sat", "unsat", "unknown", "timeout", "error"})
PREPARED_TOKENS = frozenset({"0", "prepared"})
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
NUMBER_TOKEN = re.compile(NUMBER)
CANONICAL_INDEX_TOKEN = re.compile(r"0|[1-9][0-9]*")
RAW_ASSIGNMENT = re.compile(r"\(\s*X_([^\s()]+)\s+([^\s()]+)\s*\)")
X_ASSIGNMENT_MARKER = re.compile(r"\(\s*X_")
MAX_WITNESS_INDEX = 10_000_000
MAX_WITNESS_ATOM_CHARS = 256
HEADER_SCHEMAS = {
    ("track", "onnx", "vnnlib", "verdict", "seconds"): "extended_bank_v1",
    (
        "track",
        "onnx",
        "vnnlib",
        "prepared",
        "verdict",
        "seconds",
    ): "measured_v1",
    (
        "track",
        "onnx",
        "vnnlib",
        "prepared",
        "verdict",
        "seconds",
        "run_id",
    ): "measured_v2",
}
HEADER_ALIASES = {
    "track": frozenset({"track", "category", "cat"}),
    "onnx": frozenset({"onnx", "model"}),
    "vnnlib": frozenset({"vnnlib", "property", "spec"}),
    "prepared": frozenset({"prepared"}),
    "verdict": frozenset({"verdict", "result", "status"}),
    "seconds": frozenset({"seconds", "secs", "time"}),
    "run_id": frozenset({"run_id"}),
}


VALIDATION_DEPENDENCIES = ("numpy", "onnx", "onnxruntime")


class BankValidationError(RuntimeError):
    """The source or validation evidence is unsafe or malformed."""


class EnvironmentDependencyError(RuntimeError):
    """The Python environment cannot import a required validation package."""


@dataclass(frozen=True)
class SourceResult:
    track: str
    onnx: str
    vnnlib: str
    verdict: str
    seconds: str
    schema: str
    line_number: int
    run_id: str | None = None


@dataclass(frozen=True)
class NyRun:
    verdict: str
    raw_result: bytes
    returncode: int
    timed_out: bool


@dataclass(frozen=True)
class Witness:
    values: dict[int, float]
    duplicate_indices: tuple[int, ...]
    duplicate_count: int
    parse_error: str | None


@dataclass(frozen=True)
class BoundInstance:
    row: SourceResult
    onnx_path: Path
    vnnlib_path: Path
    onnx_identity: dict[str, Any]
    vnnlib_identity: dict[str, Any]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_from_repo(value: Path | None, repo_root: Path, default: Path) -> Path:
    candidate = default if value is None else value.expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", help="extended-track category name")
    parser.add_argument("sweep_results_csv", type=Path, help="source results CSV")
    parser.add_argument(
        "rerun_budget",
        nargs="?",
        type=int,
        default=120,
        help="per-rerun NY budget in seconds (default: 120)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=f"NY repository root (default: {REPO_ROOT})",
    )
    parser.add_argument(
        "--ny-bin",
        type=Path,
        help="NY executable (default: <repo-root>/target/release/ny)",
    )
    parser.add_argument(
        "--ay-bin",
        type=Path,
        help="AY executable (default: sibling ay/target/release/ay)",
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        help=(
            "benchmark corpus root (default: "
            "<repo-root>/benchmarks/vnncomp2025/benchmarks)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="destination CSV (default: <repo-root>/reports/measured-ext/<track>.csv)",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        help=(
            "retained SAT-validation evidence directory "
            "(default: <output-dir>/evidence/<track>)"
        ),
    )
    return parser


def _normalized_header(row: Sequence[str]) -> tuple[str, ...]:
    return tuple(field.strip().lower() for field in row)


def _header_schema(row: Sequence[str]) -> str | None:
    normalized = _normalized_header(row)
    for canonical, schema in HEADER_SCHEMAS.items():
        if len(normalized) == len(canonical) and all(
            field in HEADER_ALIASES[expected]
            # Length equality above preserves strict pairing on Python 3.9,
            # which is the repository's documented minimum.
            for field, expected in zip(normalized, canonical)
        ):
            return schema
    return None


def _looks_like_header(row: Sequence[str]) -> bool:
    normalized = _normalized_header(row)
    if len(normalized) < 3:
        return False
    track_field = normalized[0] in HEADER_ALIASES["track"]
    onnx_field = normalized[1] in HEADER_ALIASES["onnx"]
    vnnlib_field = normalized[2] in HEADER_ALIASES["vnnlib"]
    return (track_field and (onnx_field or vnnlib_field)) or (
        onnx_field and vnnlib_field
    )


def _canonical_verdict(value: str, *, line_number: int) -> str:
    verdict = value.strip().lower()
    if verdict in PREPARED_TOKENS:
        raise BankValidationError(
            f"line {line_number}: prepared flag appears in the verdict column; "
            "the CSV schema is incomplete or unsupported"
        )
    if verdict not in VERDICTS:
        raise BankValidationError(
            f"line {line_number}: unsupported verdict {verdict!r}; expected one of "
            + ", ".join(sorted(VERDICTS))
        )
    return verdict


def _validated_seconds(value: str, *, line_number: int) -> str:
    seconds = value.strip()
    if not seconds:
        raise BankValidationError(f"line {line_number}: seconds field must be nonempty")
    if NUMBER_TOKEN.fullmatch(seconds) is None:
        raise BankValidationError(
            f"line {line_number}: seconds must use strict ASCII numeric syntax"
        )
    numeric = float(seconds)
    if not math.isfinite(numeric) or numeric < 0:
        raise BankValidationError(
            f"line {line_number}: seconds must be finite and nonnegative"
        )
    return seconds


def _validated_prepared(value: str, *, line_number: int) -> str:
    prepared = value.strip().lower()
    if prepared not in PREPARED_TOKENS:
        raise BankValidationError(
            f"line {line_number}: unsupported prepared flag {value!r}; "
            "expected '0' or 'prepared'"
        )
    return prepared


def parse_source_row(row: Sequence[str], line_number: int) -> SourceResult:
    """Parse one target-track row without guessing where its verdict lives."""
    if len(row) == 5:
        track, onnx, vnnlib, verdict, seconds = row
        schema = "extended_bank_v1"
        run_id = None
    elif len(row) in {6, 7}:
        track, onnx, vnnlib, _prepared, verdict, seconds = row[:6]
        _validated_prepared(_prepared, line_number=line_number)
        schema = "measured_v1" if len(row) == 6 else "measured_v2"
        run_id = (row[6].strip() or None) if len(row) == 7 else None
    else:
        raise BankValidationError(
            f"line {line_number}: expected 5, 6, or 7 columns, found {len(row)}"
        )
    track = track.strip()
    onnx = onnx.strip()
    vnnlib = vnnlib.strip()
    if not track or not onnx or not vnnlib:
        raise BankValidationError(
            f"line {line_number}: track, ONNX, and VNN-LIB fields must be nonempty"
        )
    if not onnx.lower().endswith(".onnx") or not vnnlib.lower().endswith(".vnnlib"):
        raise BankValidationError(
            f"line {line_number}: expected .onnx model and .vnnlib property paths"
        )
    return SourceResult(
        track=track,
        onnx=onnx,
        vnnlib=vnnlib,
        verdict=_canonical_verdict(verdict, line_number=line_number),
        seconds=_validated_seconds(seconds, line_number=line_number),
        schema=schema,
        line_number=line_number,
        run_id=run_id,
    )


def load_source_results(path: Path, track: str) -> list[SourceResult]:
    rows: list[SourceResult] = []
    schema_family: str | None = None
    declared_schema: str | None = None
    data_seen = False
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            for line_number, row in enumerate(csv.reader(source), 1):
                if not row or not any(field.strip() for field in row):
                    continue
                row_header_schema = _header_schema(row)
                if row_header_schema is not None:
                    if data_seen or declared_schema is not None:
                        raise BankValidationError(
                            f"line {line_number}: header must appear exactly once first"
                        )
                    declared_schema = row_header_schema
                    continue
                if _looks_like_header(row):
                    raise BankValidationError(
                        f"line {line_number}: unsupported or ambiguous CSV header"
                    )
                data_seen = True
                if row[0].strip() != track:
                    continue
                result = parse_source_row(row, line_number)
                if declared_schema is not None:
                    if result.schema != declared_schema:
                        raise BankValidationError(
                            f"line {line_number}: row does not match the declared "
                            f"{declared_schema} header schema"
                        )
                else:
                    family = (
                        "extended"
                        if result.schema == "extended_bank_v1"
                        else "measured"
                    )
                    if schema_family is None:
                        schema_family = family
                    elif schema_family != family:
                        raise BankValidationError(
                            f"line {line_number}: mixed headerless CSV schemas are forbidden"
                        )
                rows.append(result)
    except (OSError, UnicodeError, csv.Error) as error:
        raise BankValidationError(
            f"could not read source CSV {path}: {error}"
        ) from error
    return rows


def select_best(rows: Iterable[SourceResult]) -> dict[tuple[str, str], SourceResult]:
    """Preserve the legacy preference for a solved duplicate row."""
    best: dict[tuple[str, str], SourceResult] = {}
    for row in rows:
        key = (row.onnx, row.vnnlib)
        previous = best.get(key)
        if (
            previous is not None
            and previous.verdict in SOLVED_VERDICTS
            and row.verdict in SOLVED_VERDICTS
            and previous.verdict != row.verdict
        ):
            raise BankValidationError(
                f"conflicting solved verdicts for {row.onnx}, {row.vnnlib}: "
                f"{previous.verdict} on line {previous.line_number}, "
                f"{row.verdict} on line {row.line_number}"
            )
        if previous is None or (
            row.verdict in SOLVED_VERDICTS and previous.verdict not in SOLVED_VERDICTS
        ):
            best[key] = row
    return best


def parse_witness(raw_result: bytes) -> Witness:
    try:
        text = raw_result.decode("utf-8")
    except UnicodeDecodeError:
        return Witness({}, (), 0, "raw SAT result is not UTF-8")
    values: dict[int, float] = {}
    duplicate_prefix: list[int] = []
    duplicate_prefix_members: set[int] = set()
    duplicate_count = 0
    assignment_count = 0
    for match in RAW_ASSIGNMENT.finditer(text):
        assignment_count += 1
        if match.end(1) - match.start(1) > 8:
            return Witness({}, (), 0, "witness input index is too long")
        if match.end(2) - match.start(2) > MAX_WITNESS_ATOM_CHARS:
            return Witness({}, (), 0, "witness numeric value is too long")
        index_text = match.group(1)
        value_text = match.group(2)
        if CANONICAL_INDEX_TOKEN.fullmatch(index_text) is None or len(index_text) > 8:
            return Witness({}, (), 0, f"invalid witness input index X_{index_text}")
        index = int(index_text)
        if index > MAX_WITNESS_INDEX:
            return Witness(
                {}, (), 0, f"witness input index exceeds X_{MAX_WITNESS_INDEX}"
            )
        if NUMBER_TOKEN.fullmatch(value_text) is None:
            return Witness({}, (), 0, f"X_{index} has an invalid numeric value")
        value = float(value_text)
        if not math.isfinite(value):
            return Witness({}, (), 0, f"X_{index} is not finite")
        if index in values:
            duplicate_count += 1
            if len(duplicate_prefix) < 32 and index not in duplicate_prefix_members:
                duplicate_prefix.append(index)
                duplicate_prefix_members.add(index)
        values[index] = value
    if assignment_count != sum(1 for _ in X_ASSIGNMENT_MARKER.finditer(text)):
        return Witness({}, (), 0, "raw SAT result contains a malformed X_i assignment")
    if not values:
        return Witness({}, (), 0, "raw SAT result contains no X_i assignments")
    return Witness(
        values,
        tuple(sorted(duplicate_prefix)),
        duplicate_count,
        None,
    )


def _result_verdict(raw_result: bytes) -> str:
    try:
        first_line = raw_result.decode("utf-8").split("\n", 1)[0]
    except UnicodeDecodeError:
        return ""
    return first_line.strip().lower()


def run_ny(
    *,
    ny_bin: Path,
    ay_bin: Path,
    track: str,
    onnx_path: Path,
    vnnlib_path: Path,
    budget: int,
    environment: dict[str, str],
) -> NyRun:
    """Run NY with a private, automatically removed result directory."""
    with tempfile.TemporaryDirectory(prefix="ny-extended-bank-") as temporary:
        result_path = Path(temporary) / "result.txt"
        command = [
            str(ny_bin),
            "vnncomp",
            "v1",
            track,
            str(onnx_path),
            str(vnnlib_path),
            str(result_path),
            str(budget),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(environment, NY_AY=str(ay_bin), RUST_LOG="error"),
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=budget + 40)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
        if result_path.is_symlink():
            raise BankValidationError("NY result path unexpectedly became a symlink")
        try:
            raw_result = result_path.read_bytes()
        except FileNotFoundError:
            raw_result = b""
        return NyRun(
            verdict=_result_verdict(raw_result),
            raw_result=raw_result,
            returncode=returncode,
            timed_out=timed_out,
        )


def _stable_file_identity(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BankValidationError(f"evidence input does not exist: {path}") from error
    if not resolved.is_file():
        raise BankValidationError(f"evidence input is not a regular file: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_fingerprint != after_fingerprint:
        raise BankValidationError(f"evidence input changed while hashed: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": after.st_size,
        "sha256": digest.hexdigest(),
        "fingerprint": {
            "device": after.st_dev,
            "inode": after.st_ino,
            "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
        },
    }


def _optional_file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "present": False}
    identity = _stable_file_identity(path)
    identity["present"] = True
    return identity


def _require_unchanged(
    path: Path, expected: dict[str, Any], label: str, *, optional: bool = False
) -> None:
    observed = (
        _optional_file_identity(path) if optional else _stable_file_identity(path)
    )
    if observed != expected:
        raise BankValidationError(f"{label} changed during validation: {path}")


def _safe_stem(value: str) -> str:
    stem = Path(value).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-.")
    return (safe or "instance")[:80]


def _publish_read_only(
    *, root: Path, final_path: Path, data: bytes, accept_identical: bool
) -> Path:
    """Publish a complete file with link(2)'s no-overwrite semantics."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, final_path)
        except FileExistsError as error:
            if not accept_identical or not _existing_read_only_matches(
                final_path, data
            ):
                raise BankValidationError(
                    f"refusing to overwrite different evidence: {final_path}"
                ) from error
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return final_path


def _existing_read_only_matches(path: Path, expected: bytes) -> bool:
    """Compare an existing immutable artifact through a no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        path_stat = os.lstat(path)
        if stat.S_ISLNK(path_stat.st_mode):
            return False
        descriptor = os.open(path, flags | nofollow)
    except OSError:
        return False
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return False
        if (before.st_dev, before.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            return False
        if before.st_mode & 0o222:
            raise BankValidationError(f"existing evidence artifact is writable: {path}")
        if before.st_size != len(expected):
            return False
        expected_view = memoryview(expected)
        offset = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            end = offset + len(chunk)
            if expected_view[offset:end] != chunk:
                return False
            offset = end
        after = os.fstat(descriptor)
        return offset == len(expected) and (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
        )
    finally:
        os.close(descriptor)


def retain_validation_evidence(
    *,
    evidence_root: Path,
    raw_result: bytes,
    record: dict[str, Any],
    instance_name: str,
) -> Path:
    """Atomically publish complete raw bytes, then their validation sidecar."""
    evidence_root.mkdir(parents=True, exist_ok=True)
    raw_digest = _sha256(raw_result)
    raw_path = evidence_root / f"{raw_digest}.results"
    _publish_read_only(
        root=evidence_root,
        final_path=raw_path,
        data=raw_result,
        accept_identical=True,
    )
    complete_record = dict(record)
    complete_record["raw_result"] = {
        "artifact": raw_path.name,
        "sha256": raw_digest,
        "size_bytes": len(raw_result),
    }
    metadata = (
        json.dumps(complete_record, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    metadata_path = evidence_root / (
        f"{_safe_stem(instance_name)}-{uuid.uuid4().hex}.validation.json"
    )
    return _publish_read_only(
        root=evidence_root,
        final_path=metadata_path,
        data=metadata,
        accept_identical=False,
    )


def _atomic_write_rows(path: Path, rows: Iterable[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _benchmark_file(benchmark_dir: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or "\\" in value or "\0" in value:
        raise BankValidationError(
            f"benchmark path must be a relative POSIX path: {value!r}"
        )
    benchmark_dir = benchmark_dir.resolve(strict=True)
    candidate = benchmark_dir / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise BankValidationError(
            f"benchmark input does not exist: {candidate}"
        ) from error
    if not resolved.is_file():
        raise BankValidationError(f"benchmark input is not a regular file: {resolved}")
    try:
        resolved.relative_to(benchmark_dir)
    except ValueError as error:
        raise BankValidationError(
            f"benchmark input escapes its track directory: {value!r}"
        ) from error
    return resolved


def _reject_destination_collisions(
    *, output: Path, evidence_root: Path, inputs: Sequence[tuple[str, Path]]
) -> None:
    output = output.resolve()
    evidence_root = evidence_root.resolve()
    if output == evidence_root:
        raise BankValidationError("output CSV collides with the evidence root")
    try:
        output.relative_to(evidence_root)
    except ValueError:
        pass
    else:
        raise BankValidationError("output CSV is inside the evidence root")
    for label, path in inputs:
        resolved = path.resolve()
        if output == resolved or evidence_root == resolved:
            raise BankValidationError(
                f"output/evidence destination collides with {label}: {resolved}"
            )
        try:
            resolved.relative_to(evidence_root)
        except ValueError:
            pass
        else:
            raise BankValidationError(
                f"evidence root contains the bound {label}: {resolved}"
            )


def _require_validation_dependencies() -> None:
    """Import validation packages before any rerun.

    A missing interpreter package is an environment failure, never moat-breach
    evidence: the whole run aborts instead of mislabeling reproduced SATs as
    invalid counterexamples.
    """
    for package in VALIDATION_DEPENDENCIES:
        try:
            importlib.import_module(package)
        except ImportError as error:
            raise EnvironmentDependencyError(
                f"required validation package {package!r} is not importable: {error}"
            ) from error


def _load_validator() -> Any:
    # Deliberately delayed until a reproduced SAT needs validation.  In
    # particular, argparse --help does not import NumPy or ONNX Runtime.
    validator_path = SCRIPT_PATH.with_name("vnnlib_ce.py").resolve(strict=True)
    module_name = f"_ny_extended_bank_validator_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, validator_path)
    if spec is None or spec.loader is None:
        raise BankValidationError(f"could not load validator: {validator_path}")
    validator = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = validator
    try:
        spec.loader.exec_module(validator)
    finally:
        sys.modules.pop(module_name, None)
    loaded_path = Path(validator.__file__).resolve(strict=True)
    if loaded_path != validator_path:
        raise BankValidationError(
            f"validator loaded from unexpected path: {loaded_path}"
        )
    return validator


def _resolve_cli(args: argparse.Namespace) -> argparse.Namespace:
    repo_root = args.repo_root.expanduser().resolve()
    args.repo_root = repo_root
    args.ny_bin = _resolve_from_repo(
        args.ny_bin, repo_root, repo_root / "target/release/ny"
    )
    args.ay_bin = _resolve_from_repo(
        args.ay_bin, repo_root, repo_root.parent / "ay/target/release/ay"
    )
    args.bench_root = _resolve_from_repo(
        args.bench_root,
        repo_root,
        repo_root / "benchmarks/vnncomp2025/benchmarks",
    )
    args.output = _resolve_from_repo(
        args.output,
        repo_root,
        repo_root / "reports/measured-ext" / f"{args.track}.csv",
    )
    if args.evidence_root is None:
        args.evidence_root = args.output.parent / "evidence" / args.track
    else:
        args.evidence_root = _resolve_from_repo(
            args.evidence_root, repo_root, args.evidence_root
        )
    args.sweep_results_csv = args.sweep_results_csv.expanduser().resolve()
    return args


def bank(args: argparse.Namespace) -> int:
    if SAFE_COMPONENT.fullmatch(args.track) is None:
        raise BankValidationError(f"unsafe track name: {args.track!r}")
    if args.rerun_budget <= 0:
        raise BankValidationError("rerun budget must be greater than zero")
    if not args.ny_bin.is_file() or not os.access(args.ny_bin, os.X_OK):
        raise BankValidationError(f"NY executable is unavailable: {args.ny_bin}")

    source_identity = _stable_file_identity(args.sweep_results_csv)
    source_rows = load_source_results(args.sweep_results_csv, args.track)
    if not source_rows:
        raise BankValidationError(
            f"source CSV contains no rows for track {args.track!r}"
        )
    best = select_best(source_rows)
    try:
        benchmark_root = args.bench_root.resolve(strict=True)
        benchmark_dir = (benchmark_root / args.track).resolve(strict=True)
        benchmark_dir.relative_to(benchmark_root)
    except (OSError, ValueError) as error:
        raise BankValidationError(
            f"track benchmark directory is missing or escapes the corpus: {args.track}"
        ) from error
    bound: dict[tuple[str, str], BoundInstance] = {}
    for key, row in best.items():
        onnx_path = _benchmark_file(benchmark_dir, row.onnx)
        vnnlib_path = _benchmark_file(benchmark_dir, row.vnnlib)
        bound[key] = BoundInstance(
            row=row,
            onnx_path=onnx_path,
            vnnlib_path=vnnlib_path,
            onnx_identity=_stable_file_identity(onnx_path),
            vnnlib_identity=_stable_file_identity(vnnlib_path),
        )
    canonical: dict[tuple[Path, Path], BoundInstance] = {}
    for instance in bound.values():
        canonical_key = (instance.onnx_path, instance.vnnlib_path)
        previous = canonical.get(canonical_key)
        if (
            previous is not None
            and previous.row.verdict in SOLVED_VERDICTS
            and instance.row.verdict in SOLVED_VERDICTS
            and previous.row.verdict != instance.row.verdict
        ):
            raise BankValidationError(
                "conflicting solved verdicts resolve to the same benchmark inputs: "
                f"{previous.row.verdict} on line {previous.row.line_number}, "
                f"{instance.row.verdict} on line {instance.row.line_number}"
            )
        if previous is None or (
            instance.row.verdict in SOLVED_VERDICTS
            and previous.row.verdict not in SOLVED_VERDICTS
        ):
            canonical[canonical_key] = instance
    bound = {
        (instance.row.onnx, instance.row.vnnlib): instance
        for instance in canonical.values()
    }
    ny_identity = _stable_file_identity(args.ny_bin)
    ay_identity = _optional_file_identity(args.ay_bin)
    destination_inputs: list[tuple[str, Path]] = [
        ("source CSV", args.sweep_results_csv),
        ("NY executable", args.ny_bin),
        ("AY executable", args.ay_bin),
        ("counterexample validator", SCRIPT_PATH.with_name("vnnlib_ce.py")),
    ]
    for instance in bound.values():
        destination_inputs.extend(
            (
                ("ONNX model", instance.onnx_path),
                ("VNN-LIB property", instance.vnnlib_path),
            )
        )
    _reject_destination_collisions(
        output=args.output,
        evidence_root=args.evidence_root,
        inputs=destination_inputs,
    )
    environment = dict(os.environ)
    output_rows: list[list[str]] = []
    sat_count = unsat_count = other_count = invalid_count = lost_count = 0
    breaches: list[tuple[str, str, str]] = []

    for (onnx, vnnlib), instance in sorted(bound.items()):
        row = instance.row
        onnx_path = instance.onnx_path
        vnnlib_path = instance.vnnlib_path
        if row.verdict == "sat":
            reproduced = False
            for attempt in range(1, RETRIES + 1):
                run = run_ny(
                    ny_bin=args.ny_bin,
                    ay_bin=args.ay_bin,
                    track=args.track,
                    onnx_path=onnx_path,
                    vnnlib_path=vnnlib_path,
                    budget=args.rerun_budget,
                    environment=environment,
                )
                _require_unchanged(
                    args.sweep_results_csv, source_identity, "source CSV"
                )
                _require_unchanged(args.ny_bin, ny_identity, "NY executable")
                _require_unchanged(
                    args.ay_bin, ay_identity, "AY executable", optional=True
                )
                _require_unchanged(onnx_path, instance.onnx_identity, "ONNX model")
                _require_unchanged(
                    vnnlib_path, instance.vnnlib_identity, "VNN-LIB property"
                )
                if run.verdict != "sat":
                    continue
                reproduced = True
                witness = parse_witness(run.raw_result)
                in_box = False
                is_counterexample = False
                detail = witness.parse_error or ""
                output_assertions = 0
                input_assertions = 0
                expected_input_count = 0
                validator_identity: dict[str, Any] | None = None
                validator_versions: dict[str, str] = {}
                validator_path = SCRIPT_PATH.with_name("vnnlib_ce.py").resolve(
                    strict=True
                )
                try:
                    validator_identity = _stable_file_identity(validator_path)
                    validator = _load_validator()
                    if Path(validator.__file__).resolve(strict=True) != validator_path:
                        raise BankValidationError(
                            "counterexample validator did not load from its bound path"
                        )
                    _require_unchanged(
                        validator_path, validator_identity, "counterexample validator"
                    )
                    validator_versions = validator.runtime_versions()
                    requirements = validator.property_requirements(vnnlib_path)
                    expected_input_count = getattr(requirements, "input_count", None)
                    if expected_input_count is None:
                        expected_input_count = len(requirements.input_indices)
                    input_assertions = requirements.input_assertion_count
                    output_assertions = requirements.output_assertion_count
                    if witness.parse_error is None and witness.duplicate_count == 0:
                        in_box, is_counterexample, detail = validator.validate(
                            onnx_path, vnnlib_path, witness.values
                        )
                except BankValidationError:
                    raise
                except ImportError as error:
                    # A missing/broken validation package is an environment
                    # failure, never moat-breach evidence: abort the run.
                    raise EnvironmentDependencyError(
                        f"validation dependency import failed: {error}"
                    ) from error
                except Exception as error:  # fail closed, but retain the SAT bytes
                    in_box = False
                    is_counterexample = False
                    if not detail:
                        detail = f"validator error: {type(error).__name__}: {error}"
                if validator_identity is not None:
                    _require_unchanged(
                        validator_path, validator_identity, "counterexample validator"
                    )
                if witness.duplicate_count:
                    detail = "duplicate witness assignments for " + ", ".join(
                        f"X_{index}" for index in witness.duplicate_indices
                    )
                    if witness.duplicate_count > len(witness.duplicate_indices):
                        detail += ", ..."

                _require_unchanged(
                    args.sweep_results_csv, source_identity, "source CSV"
                )
                _require_unchanged(args.ny_bin, ny_identity, "NY executable")
                _require_unchanged(
                    args.ay_bin, ay_identity, "AY executable", optional=True
                )
                _require_unchanged(onnx_path, instance.onnx_identity, "ONNX model")
                _require_unchanged(
                    vnnlib_path, instance.vnnlib_identity, "VNN-LIB property"
                )

                banked_verdict = "sat" if is_counterexample else "unknown"
                evidence_record = {
                    "schema": "ny_extended_bank_validation_v1",
                    "validated_at_utc": _utc_now(),
                    "track": args.track,
                    "source": {
                        **source_identity,
                        "line_number": row.line_number,
                        "schema": row.schema,
                        "run_id": row.run_id,
                    },
                    "instance": {
                        "onnx": {
                            "declared_path": onnx,
                            **instance.onnx_identity,
                        },
                        "vnnlib": {
                            "declared_path": vnnlib,
                            **instance.vnnlib_identity,
                        },
                    },
                    "solver": {
                        "ny": ny_identity,
                        "ay": ay_identity,
                        "returncode": run.returncode,
                        "timed_out": run.timed_out,
                        "rerun_budget_seconds": args.rerun_budget,
                        "attempt": attempt,
                    },
                    "witness": {
                        "provided_input_indices": {
                            "encoding": "count_and_prefix",
                            "count": len(witness.values),
                            "prefix": list(itertools.islice(witness.values, 32)),
                        },
                        "expected_input_indices": {
                            "encoding": "contiguous_range",
                            "start": 0,
                            "stop_exclusive": expected_input_count,
                            "count": expected_input_count,
                        },
                        "duplicate_input_indices": {
                            "encoding": "count_and_prefix",
                            "count": witness.duplicate_count,
                            "prefix": list(witness.duplicate_indices),
                        },
                        "parse_error": witness.parse_error,
                    },
                    "validation": {
                        "validator": "vnnlib_ce_streaming_full_assert_v3",
                        "validator_file": validator_identity,
                        "runtime_versions": validator_versions,
                        "in_box": in_box,
                        "is_counterexample": is_counterexample,
                        "input_assertion_count": input_assertions,
                        "output_assertion_count": output_assertions,
                        "detail": detail,
                    },
                    "banked_verdict": banked_verdict,
                }
                evidence_path = retain_validation_evidence(
                    evidence_root=args.evidence_root,
                    raw_result=run.raw_result,
                    record=evidence_record,
                    instance_name=vnnlib,
                )
                if is_counterexample:
                    output_rows.append([args.track, onnx, vnnlib, "sat", row.seconds])
                    sat_count += 1
                else:
                    print(
                        f"  !!! MOAT: ny=sat but CE INVALID on {Path(vnnlib).name} "
                        f":: {detail} (evidence: {evidence_path})"
                    )
                    breaches.append((onnx, vnnlib, detail))
                    invalid_count += 1
                    output_rows.append(
                        [args.track, onnx, vnnlib, "unknown", row.seconds]
                    )
                break
            if not reproduced:
                print(
                    f"  .. ny=sat not reproduced in {RETRIES} tries on "
                    f"{Path(vnnlib).name} -> unknown (win not banked)"
                )
                output_rows.append([args.track, onnx, vnnlib, "unknown", row.seconds])
                lost_count += 1
        elif row.verdict == "unsat":
            output_rows.append([args.track, onnx, vnnlib, "unsat", row.seconds])
            unsat_count += 1
        else:
            output_rows.append([args.track, onnx, vnnlib, "unknown", row.seconds])
            other_count += 1

    _require_unchanged(args.sweep_results_csv, source_identity, "source CSV")
    _require_unchanged(args.ny_bin, ny_identity, "NY executable")
    _require_unchanged(args.ay_bin, ay_identity, "AY executable", optional=True)
    for instance in bound.values():
        _require_unchanged(instance.onnx_path, instance.onnx_identity, "ONNX model")
        _require_unchanged(
            instance.vnnlib_path, instance.vnnlib_identity, "VNN-LIB property"
        )
    _atomic_write_rows(args.output, output_rows)
    print(
        f"{args.track}: sat={sat_count} (all full-spec in-box validated) "
        f"unsat={unsat_count} unknown/other={other_count} "
        f"NOT-REPRODUCED={lost_count} INVALID-SAT={invalid_count} -> {args.output}"
    )
    print(
        "*** MOAT BREACH: invalid sat CE — INVESTIGATE ***"
        if breaches
        else (
            "MOAT: every banked ny=sat has a retained, complete, strictly-in-box "
            "counterexample validation record."
        )
    )
    return 3 if breaches else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _resolve_cli(parser.parse_args(argv))
    try:
        _require_validation_dependencies()
        return bank(args)
    except EnvironmentDependencyError as error:
        print(f"ENVIRONMENT ERROR: {error}", file=sys.stderr)
        return 4
    except BankValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
