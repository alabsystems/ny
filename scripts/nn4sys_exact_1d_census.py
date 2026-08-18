#!/usr/bin/env python3
"""Exact property census for the 32 open NN4SYS Main16 targets.

This is source-only research tooling.  It is not imported by the verifier and
has no score or verdict authority.

The two official MSCN-2048 model identities are pinned below.  Property
endpoints are compared with ``Decimal`` (never binary floating point), every
clause must contain a complete nonduplicated input box and exactly one scalar
``Y_0`` threshold, and all resource use is capped.  The companion Rust example
``nn4sys_one_axis_structure`` checks the actual post-loader ``GraphNetwork``
algebra for the free-axis sets emitted here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import stat
import sys
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, TextIO


NONDUAL_CARDINALITIES = (500, 750, 1000, 2120, 2680, 3240, 3800, 4360, 4920, 5480)
DUAL_CARDINALITIES = (
    240,
    360,
    480,
    600,
    720,
    840,
    960,
    2260,
    2890,
    3520,
    4150,
    4780,
    5410,
    6040,
    6670,
    7300,
    7930,
    8560,
    9190,
    9820,
    10450,
    11080,
)

NONDUAL_MODEL = "mscn_2048d.onnx"
DUAL_MODEL = "mscn_2048d_dual.onnx"
MODEL_SHA256 = {
    NONDUAL_MODEL: "efb7059381f569287f8c37ff30fb505205effce21fc22f5c8c4e3ef308365f30",
    DUAL_MODEL: "a86f4357cc2e07df6739d8242076ec71cbbe56aa8b0a0e4beed0323193663436",
}

DECLARE_RE = re.compile(r"^\(declare-const ([XY])_(\d+) Real\)$")
LOWER_RE = re.compile(r"\(>= X_(\d+) ([^()\s]+)\)")
UPPER_RE = re.compile(r"\(<= X_(\d+) ([^()\s]+)\)")
OUTPUT_RE = re.compile(r"\(([<>]=) Y_(\d+) ([^()\s]+)\)")

MAX_PROPERTY_BYTES = 128 << 20
MAX_MODEL_BYTES = 192 << 20
MAX_TOTAL_CLAUSES = 200_000
MAX_INPUTS = 308
DEFAULT_DEADLINE_SECONDS = 120.0
HARD_DEADLINE_SECONDS = 3600.0


class CensusError(RuntimeError):
    """The requested corpus cannot be classified exactly."""


@dataclass(frozen=True)
class Target:
    kind: str
    cardinality: int
    filename: str
    model: str
    inputs: int
    expected_clauses: int


@dataclass(frozen=True)
class PropertyCensus:
    target: Target
    observed_clauses: int
    free_dimension_histogram: Counter[int]
    one_axis_candidates: int
    piecewise_affine_pre_sigmoid: int
    degree_le_one_at_mul: int
    mul_nonlinear_obstructions: int
    dynamic_divisor_obstructions: int
    peelable_sigmoid: int
    constant_output: int
    free_axes: tuple[int, ...]
    divisor_min: Decimal | None
    divisor_max: Decimal | None


def official_targets() -> tuple[Target, ...]:
    nondual = tuple(
        Target(
            kind="nondual",
            cardinality=count,
            filename=f"cardinality_0_{count}_2048.vnnlib",
            model=NONDUAL_MODEL,
            inputs=154,
            expected_clauses=2 * count,
        )
        for count in NONDUAL_CARDINALITIES
    )
    dual = tuple(
        Target(
            kind="dual",
            cardinality=count,
            filename=f"cardinality_1_{count}_2048_dual.vnnlib",
            model=DUAL_MODEL,
            inputs=308,
            expected_clauses=count,
        )
        for count in DUAL_CARDINALITIES
    )
    return nondual + dual


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise CensusError("census wall-clock deadline exhausted")


def _decimal(
    value: str, *, path: Path, line_number: int, variable: str
) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise CensusError(
            f"{path}:{line_number}: {variable} has invalid decimal {value!r}"
        ) from error
    if not parsed.is_finite():
        raise CensusError(
            f"{path}:{line_number}: {variable} bound must be finite, got {value!r}"
        )
    return parsed


def _regular_file_size(path: Path, *, maximum: int, label: str) -> int:
    try:
        metadata = path.stat()
    except OSError as error:
        raise CensusError(f"{label} is unavailable: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise CensusError(f"{label} is not a regular file: {path}")
    if metadata.st_size > maximum:
        raise CensusError(
            f"{label} exceeds the {maximum}-byte resource cap: {path}"
        )
    return metadata.st_size


def _sha256(path: Path, *, deadline: float) -> str:
    _regular_file_size(path, maximum=MAX_MODEL_BYTES, label="ONNX model")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1 << 20):
                _check_deadline(deadline)
                digest.update(chunk)
    except OSError as error:
        raise CensusError(f"could not hash ONNX model {path}") from error
    return digest.hexdigest()


def verify_models(onnx_dir: Path, *, deadline: float) -> None:
    for filename, expected in MODEL_SHA256.items():
        path = onnx_dir / filename
        observed = _sha256(path, deadline=deadline)
        if observed != expected:
            raise CensusError(
                f"{path}: SHA-256 {observed} != official audited identity {expected}"
            )


def _feature_group(kind: str, axis: int) -> tuple[str, tuple[int, ...]] | None:
    """Return (role, divisor axes) for the audited MSCN slice/split geometry.

    ``role`` is ``feature``, ``weight``, or ``ignored``.  The official
    one-free-axis clauses all land on ``feature``.  An ignored coordinate makes
    the graph output constant; a weight coordinate makes the relevant `Div`
    denominator dynamic and is conservatively excluded from the PA-core count.
    """

    if kind == "nondual":
        local_axis = axis
    elif kind == "dual":
        # The dual graph is two disjoint 11x14 MSCN inputs followed by a
        # subtraction.  Normalize within the selected half.
        local_axis = axis % (11 * 14)
    else:
        raise AssertionError(kind)

    row, column = divmod(local_axis, 14)
    groups = (
        (range(0, 3), 6),
        (range(3, 9), 13),
        (range(9, 11), 6),
    )
    for rows, weight_column in groups:
        if row not in rows:
            continue
        branch_offset = axis - local_axis
        divisor_axes = tuple(branch_offset + current * 14 + weight_column for current in rows)
        if column < weight_column:
            return "feature", divisor_axes
        if column == weight_column:
            return "weight", divisor_axes
        return "ignored", divisor_axes
    return None


def census_property(path: Path, target: Target, *, deadline: float) -> PropertyCensus:
    _check_deadline(deadline)
    if path.name != target.filename:
        raise CensusError(
            f"target filename mismatch: expected {target.filename}, got {path.name}"
        )
    if target.inputs == 0 or target.inputs > MAX_INPUTS:
        raise CensusError(
            f"{path}: input count {target.inputs} is outside the 1..{MAX_INPUTS} cap"
        )
    _regular_file_size(path, maximum=MAX_PROPERTY_BYTES, label="VNNLIB property")

    declarations: set[tuple[str, int]] = set()
    assert_open = False
    assert_closed = False
    histogram: Counter[int] = Counter()
    free_axes: Counter[int] = Counter()
    one_axis_candidates = 0
    pa_core = 0
    degree_le_one_at_mul = 0
    mul_obstructions = 0
    dynamic_divisors = 0
    peelable_sigmoid = 0
    constant_output = 0
    divisors: list[Decimal] = []
    observed = 0

    try:
        source = path.open("r", encoding="utf-8")
    except OSError as error:
        raise CensusError(f"could not open property {path}") from error
    with source:
        for line_number, line in enumerate(source, start=1):
            if line_number % 64 == 0:
                _check_deadline(deadline)
            stripped = line.strip()
            if not stripped:
                continue
            declaration = DECLARE_RE.fullmatch(stripped)
            if declaration is not None:
                if assert_open or assert_closed:
                    raise CensusError(
                        f"{path}:{line_number}: declaration appears inside/after assertion"
                    )
                key = (declaration[1], int(declaration[2]))
                if key in declarations:
                    raise CensusError(
                        f"{path}:{line_number}: duplicate declaration {key[0]}_{key[1]}"
                    )
                declarations.add(key)
                continue
            if stripped == "(assert (or":
                if assert_open or assert_closed:
                    raise CensusError(
                        f"{path}:{line_number}: duplicate or misplaced assertion wrapper"
                    )
                assert_open = True
                continue
            if stripped == "))":
                if not assert_open or assert_closed:
                    raise CensusError(
                        f"{path}:{line_number}: duplicate or misplaced assertion close"
                    )
                assert_closed = True
                continue
            if (
                not assert_open
                or assert_closed
                or not stripped.startswith("(and ")
                or not stripped.endswith(")")
            ):
                raise CensusError(
                    f"{path}:{line_number}: unsupported canonical VNN-LIB syntax"
                )
            observed += 1
            if observed > target.expected_clauses:
                raise CensusError(
                    f"{path}: more than {target.expected_clauses} clauses"
                )

            lower_pairs = LOWER_RE.findall(line)
            upper_pairs = UPPER_RE.findall(line)
            if len(lower_pairs) != target.inputs or len(upper_pairs) != target.inputs:
                raise CensusError(
                    f"{path}:{line_number}: expected {target.inputs} lower/upper "
                    f"bounds, got {len(lower_pairs)}/{len(upper_pairs)}"
                )
            lower_raw = {int(axis): value for axis, value in lower_pairs}
            upper_raw = {int(axis): value for axis, value in upper_pairs}
            expected_axes = set(range(target.inputs))
            if (
                len(lower_raw) != len(lower_pairs)
                or len(upper_raw) != len(upper_pairs)
                or set(lower_raw) != expected_axes
                or set(upper_raw) != expected_axes
            ):
                raise CensusError(
                    f"{path}:{line_number}: input box is duplicate, missing, or noncontiguous"
                )

            lower: dict[int, Decimal] = {}
            upper: dict[int, Decimal] = {}
            ranged: list[int] = []
            for axis in range(target.inputs):
                low = _decimal(
                    lower_raw[axis],
                    path=path,
                    line_number=line_number,
                    variable=f"X_{axis} lower",
                )
                high = _decimal(
                    upper_raw[axis],
                    path=path,
                    line_number=line_number,
                    variable=f"X_{axis} upper",
                )
                if low > high:
                    raise CensusError(
                        f"{path}:{line_number}: X_{axis} lower {low} > upper {high}"
                    )
                lower[axis] = low
                upper[axis] = high
                if low != high:
                    ranged.append(axis)
            histogram[len(ranged)] += 1

            output = OUTPUT_RE.findall(line)
            if len(output) != 1 or output[0][1] != "0":
                raise CensusError(
                    f"{path}:{line_number}: expected exactly one scalar Y_0 threshold"
                )
            _decimal(
                output[0][2],
                path=path,
                line_number=line_number,
                variable="Y_0 threshold",
            )

            if len(ranged) != 1:
                continue
            one_axis_candidates += 1
            axis = ranged[0]
            free_axes[axis] += 1
            classified = _feature_group(target.kind, axis)
            if classified is None:
                raise CensusError(
                    f"{path}:{line_number}: axis {axis} is outside audited MSCN geometry"
                )
            role, divisor_axes = classified
            if role == "ignored":
                constant_output += 1
                continue
            # A single scalar can enter either the feature or weight side of
            # each MSCN Mul, never both.  Therefore no one-axis clause creates
            # a degree-2 product at Mul.
            degree_le_one_at_mul += 1
            if role == "weight":
                dynamic_divisors += 1
                continue

            if any(lower[index] != upper[index] for index in divisor_axes):
                raise CensusError(
                    f"{path}:{line_number}: feature axis {axis} unexpectedly shares "
                    "a dynamic divisor"
                )
            divisor = sum((lower[index] for index in divisor_axes), Decimal(0))
            if divisor <= 0:
                raise CensusError(
                    f"{path}:{line_number}: axis-constant divisor is not positive: {divisor}"
                )
            divisors.append(divisor)
            pa_core += 1
            peelable_sigmoid += 1

    _check_deadline(deadline)
    expected_declarations = {("X", axis) for axis in range(target.inputs)}
    expected_declarations.add(("Y", 0))
    if declarations != expected_declarations:
        raise CensusError(
            f"{path}: declaration surface is incomplete, duplicate, or noncanonical"
        )
    if not assert_open or not assert_closed:
        raise CensusError(f"{path}: incomplete assertion wrapper")
    if observed != target.expected_clauses:
        raise CensusError(
            f"{path}: expected {target.expected_clauses} clauses, observed {observed}"
        )
    if observed > MAX_TOTAL_CLAUSES:
        raise CensusError(f"{path}: clause count exceeds hard cap")
    return PropertyCensus(
        target=target,
        observed_clauses=observed,
        free_dimension_histogram=histogram,
        one_axis_candidates=one_axis_candidates,
        piecewise_affine_pre_sigmoid=pa_core,
        degree_le_one_at_mul=degree_le_one_at_mul,
        mul_nonlinear_obstructions=mul_obstructions,
        dynamic_divisor_obstructions=dynamic_divisors,
        peelable_sigmoid=peelable_sigmoid,
        constant_output=constant_output,
        free_axes=tuple(sorted(free_axes)),
        divisor_min=min(divisors) if divisors else None,
        divisor_max=max(divisors) if divisors else None,
    )


CSV_COLUMNS = (
    "kind",
    "model",
    "property",
    "observed_clauses",
    "free_dim_0",
    "free_dim_1",
    "free_dim_2",
    "free_dim_3",
    "free_dim_4",
    "free_dim_5",
    "free_dim_gt5",
    "one_axis_candidates",
    "piecewise_affine_pre_sigmoid",
    "degree_le_one_at_mul",
    "mul_nonlinear_obstructions",
    "dynamic_divisor_obstructions",
    "peelable_sigmoid",
    "constant_output",
    "free_axes",
    "constant_divisor_min",
    "constant_divisor_max",
)


def _row(census: PropertyCensus) -> dict[str, object]:
    histogram = census.free_dimension_histogram
    return {
        "kind": census.target.kind,
        "model": census.target.model,
        "property": census.target.filename,
        "observed_clauses": census.observed_clauses,
        "free_dim_0": histogram[0],
        "free_dim_1": histogram[1],
        "free_dim_2": histogram[2],
        "free_dim_3": histogram[3],
        "free_dim_4": histogram[4],
        "free_dim_5": histogram[5],
        "free_dim_gt5": sum(count for dims, count in histogram.items() if dims > 5),
        "one_axis_candidates": census.one_axis_candidates,
        "piecewise_affine_pre_sigmoid": census.piecewise_affine_pre_sigmoid,
        "degree_le_one_at_mul": census.degree_le_one_at_mul,
        "mul_nonlinear_obstructions": census.mul_nonlinear_obstructions,
        "dynamic_divisor_obstructions": census.dynamic_divisor_obstructions,
        "peelable_sigmoid": census.peelable_sigmoid,
        "constant_output": census.constant_output,
        "free_axes": ";".join(map(str, census.free_axes)),
        "constant_divisor_min": (
            str(census.divisor_min) if census.divisor_min is not None else ""
        ),
        "constant_divisor_max": (
            str(census.divisor_max) if census.divisor_max is not None else ""
        ),
    }


def write_csv(censuses: Iterable[PropertyCensus], destination: TextIO) -> None:
    censuses = list(censuses)
    writer = csv.DictWriter(destination, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for census in censuses:
        writer.writerow(_row(census))

    histogram: Counter[int] = Counter()
    for census in censuses:
        histogram.update(census.free_dimension_histogram)
    totals: dict[str, object] = {
        column: "" for column in CSV_COLUMNS
    }
    totals.update(
        {
            "kind": "TOTAL",
            "model": "2",
            "property": str(len(censuses)),
            "observed_clauses": sum(c.observed_clauses for c in censuses),
            "free_dim_0": histogram[0],
            "free_dim_1": histogram[1],
            "free_dim_2": histogram[2],
            "free_dim_3": histogram[3],
            "free_dim_4": histogram[4],
            "free_dim_5": histogram[5],
            "free_dim_gt5": sum(
                count for dims, count in histogram.items() if dims > 5
            ),
            "one_axis_candidates": sum(c.one_axis_candidates for c in censuses),
            "piecewise_affine_pre_sigmoid": sum(
                c.piecewise_affine_pre_sigmoid for c in censuses
            ),
            "degree_le_one_at_mul": sum(
                c.degree_le_one_at_mul for c in censuses
            ),
            "mul_nonlinear_obstructions": sum(
                c.mul_nonlinear_obstructions for c in censuses
            ),
            "dynamic_divisor_obstructions": sum(
                c.dynamic_divisor_obstructions for c in censuses
            ),
            "peelable_sigmoid": sum(c.peelable_sigmoid for c in censuses),
            "constant_output": sum(c.constant_output for c in censuses),
            "constant_divisor_min": min(
                c.divisor_min for c in censuses if c.divisor_min is not None
            ),
            "constant_divisor_max": max(
                c.divisor_max for c in censuses if c.divisor_max is not None
            ),
        }
    )
    writer.writerow(totals)


def census_corpus(root: Path, *, deadline: float) -> list[PropertyCensus]:
    onnx_dir = root / "onnx"
    vnnlib_dir = root / "vnnlib"
    verify_models(onnx_dir, deadline=deadline)
    censuses: list[PropertyCensus] = []
    total_clauses = 0
    for target in official_targets():
        _check_deadline(deadline)
        census = census_property(vnnlib_dir / target.filename, target, deadline=deadline)
        total_clauses += census.observed_clauses
        if total_clauses > MAX_TOTAL_CLAUSES:
            raise CensusError(
                f"corpus exceeds the {MAX_TOTAL_CLAUSES}-clause resource cap"
            )
        censuses.append(census)
    return censuses


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "nn4sys_root",
        type=Path,
        help="official NN4SYS directory containing onnx/ and vnnlib/",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=DEFAULT_DEADLINE_SECONDS,
        help=f"whole-process wall cap (default {DEFAULT_DEADLINE_SECONDS:g})",
    )
    parser.add_argument("--output", type=Path, help="write CSV here instead of stdout")
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.deadline_seconds)
        or args.deadline_seconds <= 0
        or args.deadline_seconds > HARD_DEADLINE_SECONDS
    ):
        parser.error(
            f"--deadline-seconds must be finite and in (0, {HARD_DEADLINE_SECONDS:g}]"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    deadline = time.monotonic() + args.deadline_seconds
    try:
        censuses = census_corpus(args.nn4sys_root, deadline=deadline)
        if args.output is None:
            write_csv(censuses, sys.stdout)
        else:
            with args.output.open("w", encoding="utf-8", newline="") as destination:
                write_csv(censuses, destination)
    except (CensusError, OSError) as error:
        print(f"nn4sys exact-1d census: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
