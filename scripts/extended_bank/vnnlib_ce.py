#!/usr/bin/env python3
"""Streaming full-assert VNN-LIB 1.x / 2.0 counterexample validation.

Properties are scanned in constant passes.  The tokenizer is incremental and
only one top-level s-expression is materialized at a time, with an explicit
per-expression cap.  This keeps multi-million-line properties bounded while
retaining fail-closed evaluation of every assertion.

Two variable dialects are read, and never both in one file:

* VNN-LIB 1.x -- ``(declare-const X_0 Real)`` and flat names ``X_0`` / ``Y_3``.
* VNN-LIB 2.0 -- a ``(declare-network ...)`` header that DECLARES the tensor
  shapes, and tensor-indexed names ``X[0,i]`` / ``X[0,r,c,ch]`` / ``Y[0,j]``.

The 2.0 flat index is derived from the DECLARED shape in row-major order and is
never assumed: ``X[0,r,c,ch]`` under ``[1,30,30,3]`` is ``r*90 + c*3 + ch``, and
the same derivation gives ``r*32*32 + ...`` under an NCHW ``[1,3,32,32]``.  Every
axis index is bounds-checked against the declared extent, and the declared input
shape is required to match the ONNX input shape before any witness is executed:
a flat mapping that disagreed with the tensor the network actually consumes
could ACCEPT A BAD WITNESS, which is far worse than refusing a good one.

Anything this validator cannot READ raises :class:`UnsupportedSyntaxError`,
which is deliberately distinct from every structural verdict.  A structural
refusal is a claim about a property that was fully read; an unsupported-syntax
error is the explicit admission that nothing was read at all.
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
import struct
import sys
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, TextIO

NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
NUMBER_TOKEN = re.compile(NUMBER)
INDEX = r"(?:0|[1-9][0-9]*)"
INDEX_TOKEN = re.compile(INDEX)
VARIABLE = re.compile(rf"([XY])_({INDEX})")
# VNN-LIB 2.0 tensor-indexed names.  A token never contains whitespace, so the
# index list is required to be whitespace-free -- exactly how the corpus writes
# it (``X[0,0,0,0]``).  A spaced form would tokenize apart and be refused as
# unreadable rather than silently mis-parsed.
TENSOR_VARIABLE = re.compile(rf"([XY])\[({INDEX}(?:,{INDEX})*)\]")
# ``[1, 30, 30, 3]`` tokenizes as ``[1,`` ``30,`` ``30,`` ``3]``; the pieces are
# rejoined with no separator before this matches.
SHAPE_LITERAL = re.compile(rf"\[({INDEX}(?:,{INDEX})*)\]")
# An atom that is plainly a variable reference this validator cannot resolve.
# Used to turn an unreadable NAME into an explicit error instead of silently
# treating the variable as absent -- the exact failure this module had.
VARIABLE_SHAPED = re.compile(r"[XY][A-Za-z0-9_]*")
TOKEN = re.compile(r"\(|\)|[^\s()]+")
COUNTEREXAMPLE_ASSIGNMENT = re.compile(r"\(\s*X_([^\s()]+)\s+([^\s()]+)\s*\)")
COUNTEREXAMPLE_ASSIGNMENT_MARKER = re.compile(r"\(\s*X_")
SOURCE_CHUNK_CHARS = 64 * 1024
MAX_ATOM_CHARS = 256
MAX_EXPRESSION_TOKENS = 100_000
MAX_EXPRESSION_DEPTH = 128
MAX_VARIABLE_INDEX = 10_000_000
RESULT_DETAIL_LIMIT = 32
# Detail prefix that marks "this file was not read" as distinct from every
# structural verdict.  Callers grep for it to keep an unreadable spec out of the
# admission ledger instead of scoring it as a refusal.
UNSUPPORTED_SYNTAX_PREFIX = "unsupported VNN-LIB syntax:"
MIXED_SYNTAX_MESSAGE = (
    "mixed VNN-LIB syntax in one file: VNN-LIB 1.x declare-const alongside a "
    "2.0 vnnlib-version/declare-network header"
)
SUPPORTED_LOGICS = frozenset({"QF_LRA", "QF_NRA"})
SUPPORTED_VNNLIB_VERSIONS = frozenset({"<2.0>", "2.0"})
MAX_TENSOR_RANK = 8
RUNTIME_DEPENDENCIES = ("numpy", "onnx", "onnxruntime")


class ValidationError(ValueError):
    """A VNN-LIB property cannot be validated safely."""


class UnsupportedSyntaxError(ValidationError):
    """This file's VNN-LIB dialect cannot be READ by this validator.

    Deliberately distinct from every structural verdict.  "This property is not
    searchable", "input constraints do not reference X_3", "property has no
    output-referencing assertions" are all claims ABOUT a property that was
    fully read.  This error is the opposite claim: the file was not read, so no
    statement about its contents -- searchable, admissible, or otherwise -- is
    being made, and callers must never fold it into an admission decision.

    It subclasses :class:`ValidationError` so that every existing fail-closed
    handler still refuses the witness; the distinction is in the type and the
    message, never in the safety.
    """


class MissingDependencyError(RuntimeError):
    """The Python environment cannot import a required runtime package."""


def require_runtime_dependencies() -> None:
    """Import runtime packages eagerly, before any result depends on them.

    A missing interpreter package is an environment failure, never evidence
    about a counterexample; callers abort instead of recording a verdict.
    """
    for package in RUNTIME_DEPENDENCIES:
        try:
            import_module(package)
        except ImportError as error:
            raise MissingDependencyError(
                f"required runtime package {package!r} is not importable: {error}"
            ) from error


@dataclass(frozen=True)
class TensorLayout:
    """Row-major flattening DERIVED from a declared VNN-LIB 2.0 tensor shape.

    Nothing here is assumed about the meaning of an axis: NHWC ``[1,30,30,3]``
    and NCHW ``[1,3,32,32]`` both flatten by the same row-major rule, which is
    the rule ``numpy.reshape`` uses when the witness is handed to ONNX Runtime.
    """

    label: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    size: int

    @classmethod
    def from_shape(cls, label: str, shape: Sequence[int]) -> TensorLayout:
        dims = list(shape)
        if not dims:
            raise UnsupportedSyntaxError(f"{label} is declared with an empty shape")
        if len(dims) > MAX_TENSOR_RANK:
            raise UnsupportedSyntaxError(
                f"{label} shape {dims} exceeds the supported rank {MAX_TENSOR_RANK}"
            )
        if any(extent < 1 for extent in dims):
            raise UnsupportedSyntaxError(
                f"{label} shape {dims} has a non-positive dimension"
            )
        if dims[0] != 1:
            raise UnsupportedSyntaxError(
                f"{label} shape {dims} has leading dimension {dims[0]}; only a "
                "batch of 1 is supported, and a larger batch is refused rather "
                "than flattened under a guessed layout"
            )
        size = math.prod(dims)
        if size > MAX_VARIABLE_INDEX:
            raise UnsupportedSyntaxError(
                f"{label} shape {dims} declares {size} elements, above the "
                f"supported streaming bound {MAX_VARIABLE_INDEX}"
            )
        strides = [1] * len(dims)
        for axis in range(len(dims) - 2, -1, -1):
            strides[axis] = strides[axis + 1] * dims[axis + 1]
        return cls(label, tuple(dims), tuple(strides), size)

    def flat_index(self, indices: Sequence[int]) -> int:
        """Row-major flat offset of ``label[indices]``, bounds-checked per axis."""
        if len(indices) != len(self.shape):
            raise UnsupportedSyntaxError(
                f"{self.label}{list(indices)} has rank {len(indices)}, but "
                f"{self.label} is declared with shape {list(self.shape)}"
            )
        flat = 0
        for axis, (index, extent) in enumerate(zip(indices, self.shape)):
            if index >= extent:
                raise ValidationError(
                    f"{self.label}{list(indices)} is out of range on axis {axis}: "
                    f"declared shape is {list(self.shape)}"
                )
            flat += index * self.strides[axis]
        return flat


@dataclass(frozen=True)
class SpecLayout:
    """The declared input/output tensors of a VNN-LIB 2.0 property."""

    inputs: TensorLayout
    outputs: TensorLayout


def _looks_like_variable(token: str) -> bool:
    """Is this atom plainly a variable reference, whatever dialect wrote it?"""
    if "[" in token or "]" in token:
        return True
    return VARIABLE_SHAPED.fullmatch(token) is not None


def resolve_variable(
    token: str, layout: SpecLayout | None = None
) -> tuple[str, int] | None:
    """Map a variable atom to ``(prefix, flat_index)`` under ``layout``.

    ``layout is None`` selects the VNN-LIB 1.x dialect and behaves exactly like
    the historical ``VARIABLE.fullmatch`` path.  Returns ``None`` only when the
    atom is not a variable reference at all (a numeral, ``true``/``false``).  An
    atom that IS a variable reference but cannot be expressed in the active
    dialect raises :class:`UnsupportedSyntaxError` and never returns ``None``,
    so an unreadable NAME can never be mistaken for an ABSENT variable.
    """
    if layout is None:
        match = VARIABLE.fullmatch(token)
        if match is not None:
            return match.group(1), int(match.group(2))
        if TENSOR_VARIABLE.fullmatch(token) is not None:
            raise UnsupportedSyntaxError(
                f"tensor-indexed name {token!r} appears in a property with no "
                "(declare-network ...) header, so no shape declares its layout"
            )
    else:
        match = TENSOR_VARIABLE.fullmatch(token)
        if match is not None:
            prefix = match.group(1)
            indices = [int(part) for part in match.group(2).split(",")]
            tensor = layout.inputs if prefix == "X" else layout.outputs
            return prefix, tensor.flat_index(indices)
        if VARIABLE.fullmatch(token) is not None:
            raise UnsupportedSyntaxError(
                f"legacy name {token!r} appears in a VNN-LIB 2.0 property; mixed "
                "variable syntax in one file is refused"
            )
    if _looks_like_variable(token):
        raise UnsupportedSyntaxError(
            f"unreadable variable name {token!r}; this validator reads X_i / Y_j "
            "(VNN-LIB 1.x) and X[...] / Y[...] (VNN-LIB 2.0)"
        )
    return None


def _shape_from_tokens(label: str, tokens: Sequence[Any]) -> tuple[int, ...]:
    if not tokens or not all(isinstance(token, str) for token in tokens):
        raise UnsupportedSyntaxError(f"{label} has a malformed shape declaration")
    joined = "".join(tokens)
    match = SHAPE_LITERAL.fullmatch(joined)
    if match is None:
        raise UnsupportedSyntaxError(
            f"{label} has an unreadable shape declaration {joined!r}"
        )
    return tuple(int(part) for part in match.group(1).split(","))


def _tensor_declaration(member: Sequence[Any], form: str, name: str) -> TensorLayout:
    """Read ``(declare-input X float32 [1, 30, 30, 3])`` into a layout."""
    if len(member) < 3 or not isinstance(member[1], str):
        raise UnsupportedSyntaxError(f"malformed {form} declaration")
    if member[1] != name:
        raise UnsupportedSyntaxError(
            f"{form} declares tensor {member[1]!r}; this validator reads only "
            f"single-tensor networks whose input is X and whose output is Y"
        )
    rest = list(member[2:])
    start = next(
        (
            position
            for position, token in enumerate(rest)
            if isinstance(token, str) and token.startswith("[")
        ),
        None,
    )
    if start is None:
        raise UnsupportedSyntaxError(f"{form} {name} declares no shape")
    if start > 1:
        # name, at most one dtype token, then the shape.  Anything else is a
        # form this validator has not been shown and must not guess at.
        raise UnsupportedSyntaxError(f"malformed {form} {name} declaration")
    # The dtype token is intentionally not constrained: it never changes the
    # row-major flattening, and the corpus writes both `float32` and `real`.
    return TensorLayout.from_shape(name, _shape_from_tokens(name, rest[start:]))


def _parse_declare_network(expression: Any) -> SpecLayout:
    if (
        not isinstance(expression, list)
        or len(expression) < 2
        or not isinstance(expression[1], str)
    ):
        raise UnsupportedSyntaxError("malformed declare-network declaration")
    inputs: TensorLayout | None = None
    outputs: TensorLayout | None = None
    for member in expression[2:]:
        if not isinstance(member, list) or not member or not isinstance(member[0], str):
            raise UnsupportedSyntaxError("malformed declare-network member")
        form = member[0]
        if form == "declare-input":
            if inputs is not None:
                raise UnsupportedSyntaxError(
                    "multi-input networks are not supported by this validator"
                )
            inputs = _tensor_declaration(member, form, "X")
        elif form == "declare-output":
            if outputs is not None:
                raise UnsupportedSyntaxError(
                    "multi-output networks are not supported by this validator"
                )
            outputs = _tensor_declaration(member, form, "Y")
        else:
            raise UnsupportedSyntaxError(
                f"unsupported declare-network member {form!r}"
            )
    if inputs is None or outputs is None:
        raise UnsupportedSyntaxError(
            "declare-network must declare exactly one input X and one output Y"
        )
    return SpecLayout(inputs, outputs)


@dataclass(frozen=True)
class PropertyRequirements:
    input_count: int
    input_assertion_count: int
    output_assertion_count: int

    @property
    def input_indices(self) -> range:
        """Compatibility view without allocating millions of Python integers."""
        return range(self.input_count)


@dataclass(frozen=True)
class PropertySummary:
    input_count: int
    output_count: int
    assertion_count: int
    domain_assertion_count: int
    input_assertion_count: int
    output_assertion_count: int
    # None for VNN-LIB 1.x; the declared tensor layouts for 2.0.
    layout: SpecLayout | None = None


@dataclass
class _IndexBitmap:
    label: str
    bits: bytearray = field(default_factory=bytearray)
    count: int = 0
    maximum: int = -1

    def add(self, index: int) -> None:
        if index < 0 or index > MAX_VARIABLE_INDEX:
            raise ValidationError(
                f"{self.label}_{index} exceeds the supported streaming index bound"
            )
        if index >= len(self.bits):
            target = max(index + 1, max(64, len(self.bits) * 2))
            target = min(target, MAX_VARIABLE_INDEX + 1)
            self.bits.extend(b"\0" * (target - len(self.bits)))
        if self.bits[index]:
            raise ValidationError(f"{self.label}_{index} appears more than once")
        self.bits[index] = 1
        self.count += 1
        self.maximum = max(self.maximum, index)

    def mark(self, index: int) -> None:
        if index < 0 or index > MAX_VARIABLE_INDEX:
            raise ValidationError(
                f"{self.label}_{index} exceeds the supported streaming index bound"
            )
        if index >= len(self.bits):
            target = max(index + 1, max(64, len(self.bits) * 2))
            target = min(target, MAX_VARIABLE_INDEX + 1)
            self.bits.extend(b"\0" * (target - len(self.bits)))
        if not self.bits[index]:
            self.bits[index] = 1
            self.count += 1
        self.maximum = max(self.maximum, index)

    def require_contiguous(self) -> int:
        if self.count == 0:
            raise ValidationError(f"property declares no {self.label}_i variables")
        expected = self.maximum + 1
        if self.count != expected:
            missing = next(index for index in range(expected) if not self.bits[index])
            raise ValidationError(
                f"property {self.label}_i declarations are not contiguous; "
                f"missing {self.label}_{missing}"
            )
        return expected

    def first_unmarked(self, stop: int) -> int | None:
        if len(self.bits) < stop:
            return len(self.bits)
        return next((index for index in range(stop) if not self.bits[index]), None)


@dataclass
class _ResultAccumulator:
    count: int = 0
    all_hold: bool = True
    sample: list[bool] = field(default_factory=list)

    def add(self, value: bool) -> None:
        self.count += 1
        self.all_hold = self.all_hold and value
        if len(self.sample) < RESULT_DETAIL_LIMIT:
            self.sample.append(value)

    def detail(self) -> str:
        if self.count <= RESULT_DETAIL_LIMIT:
            return str(self.sample)
        return (
            f"<count={self.count} all_hold={self.all_hold} "
            f"prefix={self.sample[:RESULT_DETAIL_LIMIT]}>"
        )


@dataclass(frozen=True)
class _VariableEnvironment:
    inputs: Mapping[int, float]
    outputs: Sequence[float] | None = None
    executed_inputs: bool = False
    # Appended last so every existing positional/keyword construction still
    # builds the historical VNN-LIB 1.x environment unchanged.
    layout: SpecLayout | None = None

    def resolve(self, token: str) -> tuple[str, int] | None:
        return resolve_variable(token, self.layout)

    def value_of(self, resolved: tuple[str, int]) -> float:
        prefix, index = resolved
        if prefix == "X":
            if index not in self.inputs:
                raise ValidationError(f"property references unavailable X_{index}")
            value = float(self.inputs[index])
            return _float32(value) if self.executed_inputs else value
        if self.outputs is None or index >= len(self.outputs):
            raise ValidationError(f"property references unavailable Y_{index}")
        return float(self.outputs[index])

    def lookup(self, token: str) -> float:
        resolved = self.resolve(token)
        if resolved is None:
            raise ValidationError(f"invalid variable name {token!r}")
        return self.value_of(resolved)


def runtime_versions() -> dict[str, str]:
    """Return validator/runtime versions without importing heavy packages."""
    versions = {
        "python": sys.version.split()[0],
        "validator": "streaming_full_assert_v3",
    }
    for distribution in ("numpy", "onnxruntime"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _source_chunks(source: str | Iterable[str] | TextIO) -> Iterator[str]:
    if isinstance(source, str):
        for offset in range(0, len(source), SOURCE_CHUNK_CHARS):
            yield source[offset : offset + SOURCE_CHUNK_CHARS]
        return
    reader = getattr(source, "read", None)
    if callable(reader):
        while chunk := reader(SOURCE_CHUNK_CHARS):
            if not isinstance(chunk, str):
                raise ValidationError("VNN-LIB source must contain text")
            yield chunk
        return
    for piece in source:
        if not isinstance(piece, str):
            raise ValidationError("VNN-LIB source must contain text")
        for offset in range(0, len(piece), SOURCE_CHUNK_CHARS):
            yield piece[offset : offset + SOURCE_CHUNK_CHARS]


def _segment_tokens(
    segment: str, pending: str, *, terminate: bool
) -> Generator[str, None, str]:
    bounded = pending + segment
    pending = ""
    for match in TOKEN.finditer(bounded):
        token = match.group(0)
        if token not in {"(", ")"} and len(token) > MAX_ATOM_CHARS:
            raise ValidationError(f"VNN-LIB atom exceeds {MAX_ATOM_CHARS} characters")
        if not terminate and token not in {"(", ")"} and match.end() == len(bounded):
            pending = token
        else:
            yield token
    return pending


def tokenize(source: str | Iterable[str] | TextIO) -> Iterator[str]:
    """Yield tokens from bounded chunks, including giant physical lines/comments."""
    pending = ""
    in_comment = False
    for chunk in _source_chunks(source):
        cursor = 0
        while cursor < len(chunk):
            if in_comment:
                newline = chunk.find("\n", cursor)
                if newline < 0:
                    cursor = len(chunk)
                    continue
                in_comment = False
                cursor = newline + 1
            semicolon = chunk.find(";", cursor)
            if semicolon < 0:
                pending = yield from _segment_tokens(
                    chunk[cursor:], pending, terminate=False
                )
                break
            pending = yield from _segment_tokens(
                chunk[cursor:semicolon], pending, terminate=True
            )
            in_comment = True
            cursor = semicolon + 1
    yield from _segment_tokens("", pending, terminate=True)


def _expressions(tokens: Iterable[str]) -> Iterator[Any]:
    stack: list[list[Any]] = []
    expression_tokens = 0
    for token in tokens:
        if not isinstance(token, str):
            raise ValidationError("VNN-LIB token stream must contain text")
        if token not in {"(", ")"} and len(token) > MAX_ATOM_CHARS:
            raise ValidationError(f"VNN-LIB atom exceeds {MAX_ATOM_CHARS} characters")
        expression_tokens += 1
        if expression_tokens > MAX_EXPRESSION_TOKENS:
            raise ValidationError(
                f"top-level expression exceeds {MAX_EXPRESSION_TOKENS} tokens"
            )
        if token == "(":
            if len(stack) >= MAX_EXPRESSION_DEPTH:
                raise ValidationError(
                    f"VNN-LIB expression nesting exceeds {MAX_EXPRESSION_DEPTH}"
                )
            stack.append([])
            continue
        if token == ")":
            if not stack:
                raise ValidationError("unexpected closing parenthesis")
            complete = stack.pop()
            if stack:
                stack[-1].append(complete)
            else:
                yield complete
                expression_tokens = 0
            continue
        if stack:
            stack[-1].append(token)
        else:
            yield token
            expression_tokens = 0
    if stack:
        raise ValidationError("unterminated VNN-LIB expression")


def parse(tokens: Iterable[str]) -> Any:
    """Parse the first expression from an incremental token iterable."""
    try:
        return next(_expressions(tokens))
    except StopIteration as error:
        raise ValidationError("unexpected end of VNN-LIB expression") from error


def parse_all(source: str | Iterable[str]) -> Iterator[Any]:
    """Yield parsed top-level expressions one at a time."""
    return _expressions(tokenize(source))


def _file_expressions(path: str | Path) -> Iterator[Any]:
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        yield from parse_all(source)


def _float32(value: float) -> float:
    try:
        return struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValidationError(
            f"witness value {value!r} cannot be represented as float32"
        ) from error


def _number(value: float | bool) -> float:
    if isinstance(value, bool):
        raise ValidationError("Boolean value used as a numeric term")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValidationError("non-finite arithmetic value")
    return numeric


def _boolean(value: float | bool) -> bool:
    if not isinstance(value, bool):
        raise ValidationError("assertion or Boolean operand is not Boolean")
    return value


def evaluate(
    node: Any, variables: _VariableEnvironment | Mapping[str, float]
) -> float | bool:
    if isinstance(node, str):
        if isinstance(variables, _VariableEnvironment):
            # Raises UnsupportedSyntaxError for a variable-shaped atom the
            # active dialect cannot express, so an unreadable name can never
            # fall through to "unknown atom" and then to a structural verdict.
            resolved = variables.resolve(node)
            if resolved is not None:
                return variables.value_of(resolved)
        elif VARIABLE.fullmatch(node):
            # Plain-mapping environments stay VNN-LIB 1.x only, byte-identically.
            if node not in variables:
                raise ValidationError(f"property references unavailable {node}")
            return _number(variables[node])
        if node == "true":
            return True
        if node == "false":
            return False
        if NUMBER_TOKEN.fullmatch(node) is None:
            raise ValidationError(f"unknown VNN-LIB atom {node!r}")
        return _number(float(node))
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValidationError("empty or malformed VNN-LIB expression")
    operator = node[0]
    if operator == "+":
        if len(node) < 3:
            raise ValidationError("'+' requires at least two operands")
        return _number(sum(_number(evaluate(arg, variables)) for arg in node[1:]))
    if operator == "-":
        if len(node) < 2:
            raise ValidationError("'-' requires at least one operand")
        if len(node) == 2:
            return _number(-_number(evaluate(node[1], variables)))
        return _number(
            _number(evaluate(node[1], variables))
            - sum(_number(evaluate(arg, variables)) for arg in node[2:])
        )
    if operator == "*":
        if len(node) < 3:
            raise ValidationError("'*' requires at least two operands")
        result = 1.0
        for argument in node[1:]:
            result = _number(result * _number(evaluate(argument, variables)))
        return result
    if operator in {">=", "<=", ">", "<"}:
        if len(node) != 3:
            raise ValidationError(f"{operator!r} requires exactly two operands")
        left = _number(evaluate(node[1], variables))
        right = _number(evaluate(node[2], variables))
        if operator == ">=":
            return left >= right
        if operator == "<=":
            return left <= right
        if operator == ">":
            return left > right
        return left < right
    if operator == "=":
        if len(node) < 3:
            raise ValidationError("'=' requires at least two operands")
        first = evaluate(node[1], variables)
        all_equal = True
        for argument in node[2:]:
            value = evaluate(argument, variables)
            if isinstance(value, bool) != isinstance(first, bool):
                raise ValidationError("'=' operands have incompatible types")
            all_equal = all_equal and value == first
        return all_equal
    if operator in {"and", "or"}:
        if len(node) < 2:
            raise ValidationError(f"{operator!r} requires at least one operand")
        result = operator == "and"
        for argument in node[1:]:
            value = _boolean(evaluate(argument, variables))
            result = result and value if operator == "and" else result or value
        return result
    if operator == "not":
        if len(node) != 2:
            raise ValidationError("'not' requires exactly one operand")
        return not _boolean(evaluate(node[1], variables))
    raise ValidationError(f"unknown VNN-LIB operator {operator!r}")


def _references(node: Any, prefix: str, layout: SpecLayout | None = None) -> set[int]:
    references: set[int] = set()
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            resolved = resolve_variable(item, layout)
            if resolved is not None and resolved[0] == prefix:
                index = resolved[1]
                if index > MAX_VARIABLE_INDEX:
                    raise ValidationError(
                        f"{prefix}_{index} exceeds the supported streaming index bound"
                    )
                references.add(index)
        elif isinstance(item, list):
            stack.extend(item)
    return references


def _top_level_kind(expression: Any) -> str:
    if (
        not isinstance(expression, list)
        or not expression
        or not isinstance(expression[0], str)
    ):
        raise ValidationError("bare, empty, or malformed top-level expression")
    kind = expression[0]
    if kind == "declare-const":
        if (
            len(expression) != 3
            or expression[2] != "Real"
            or not isinstance(expression[1], str)
            or VARIABLE.fullmatch(expression[1]) is None
        ):
            raise ValidationError("unsupported or malformed variable declaration")
        return kind
    if kind == "assert":
        if len(expression) != 2:
            raise ValidationError("every assert must contain exactly one expression")
        return kind
    if kind == "set-logic":
        if len(expression) != 2 or expression[1] not in SUPPORTED_LOGICS:
            raise ValidationError("unsupported or malformed set-logic declaration")
        return kind
    if kind == "vnnlib-version":
        if len(expression) != 2 or not isinstance(expression[1], str):
            raise ValidationError("malformed vnnlib-version declaration")
        if expression[1] not in SUPPORTED_VNNLIB_VERSIONS:
            raise UnsupportedSyntaxError(
                f"VNN-LIB version {expression[1]!r} is not supported by this validator"
            )
        return kind
    if kind == "declare-network":
        # Contents are validated by _parse_declare_network, which raises the
        # distinct unsupported-syntax error for every shape it cannot read.
        return kind
    raise ValidationError(f"unsupported top-level form {kind!r}")


def _assertion(expression: Any) -> Any | None:
    return expression[1] if _top_level_kind(expression) == "assert" else None


def _scan_property(vnnlib_path: str | Path) -> PropertySummary:
    declared_inputs = _IndexBitmap("X")
    declared_outputs = _IndexBitmap("Y")
    constrained_inputs = _IndexBitmap("X")
    assertion_count = domain_count = input_count = output_count = 0
    maximum_input_reference = maximum_output_reference = -1
    layout: SpecLayout | None = None
    legacy_declarations = 0
    tensor_forms = 0

    phase = "start"
    for expression in _file_expressions(vnnlib_path):
        kind = _top_level_kind(expression)
        if kind == "set-logic":
            if phase != "start":
                raise ValidationError("set-logic must precede every declaration")
            phase = "declarations"
            continue
        if kind == "vnnlib-version":
            if legacy_declarations:
                raise UnsupportedSyntaxError(MIXED_SYNTAX_MESSAGE)
            if phase != "start":
                raise ValidationError("vnnlib-version must precede every declaration")
            tensor_forms += 1
            phase = "declarations"
            continue
        if kind == "declare-network":
            if phase == "assertions":
                raise ValidationError("declare-network appears after an assertion")
            if layout is not None:
                raise UnsupportedSyntaxError(
                    "multi-network properties are not supported by this validator"
                )
            if legacy_declarations:
                raise UnsupportedSyntaxError(MIXED_SYNTAX_MESSAGE)
            layout = _parse_declare_network(expression)
            tensor_forms += 1
            phase = "declarations"
            continue
        if kind == "declare-const":
            if phase == "assertions":
                raise ValidationError("variable declaration appears after an assertion")
            if tensor_forms:
                raise UnsupportedSyntaxError(MIXED_SYNTAX_MESSAGE)
            legacy_declarations += 1
            phase = "declarations"
            match = VARIABLE.fullmatch(expression[1])
            if match is None:
                raise ValidationError("unsupported variable declaration name")
            prefix, index_text = match.groups()
            target = declared_inputs if prefix == "X" else declared_outputs
            target.add(int(index_text))
            continue
        phase = "assertions"
        assertion = expression[1]
        assertion_count += 1
        input_references = _references(assertion, "X", layout)
        output_references = _references(assertion, "Y", layout)
        if input_references:
            maximum_input_reference = max(
                maximum_input_reference, max(input_references)
            )
        if output_references:
            maximum_output_reference = max(
                maximum_output_reference, max(output_references)
            )
            output_count += 1
        else:
            domain_count += 1
            if input_references:
                input_count += 1
                for index in input_references:
                    constrained_inputs.mark(index)

    if layout is None:
        declared_input_count = declared_inputs.require_contiguous()
        declared_output_count = declared_outputs.require_contiguous()
    else:
        # 2.0 declares counts by SHAPE, not one declare-const per element, so
        # contiguity is structural rather than something to re-derive.
        declared_input_count = layout.inputs.size
        declared_output_count = layout.outputs.size
    if assertion_count == 0:
        raise ValidationError("property contains no assertions")
    if maximum_input_reference >= declared_input_count:
        raise ValidationError(
            f"property references undeclared X_{maximum_input_reference}"
        )
    if maximum_output_reference >= declared_output_count:
        raise ValidationError(
            f"property references undeclared Y_{maximum_output_reference}"
        )
    unconstrained = constrained_inputs.first_unmarked(declared_input_count)
    if unconstrained is not None:
        raise ValidationError(f"input constraints do not reference X_{unconstrained}")
    if output_count == 0:
        raise ValidationError("property has no output-referencing assertions")
    return PropertySummary(
        input_count=declared_input_count,
        output_count=declared_output_count,
        assertion_count=assertion_count,
        domain_assertion_count=domain_count,
        input_assertion_count=input_count,
        output_assertion_count=output_count,
        layout=layout,
    )


def property_requirements(vnnlib_path: str | Path) -> PropertyRequirements:
    summary = _scan_property(vnnlib_path)
    return PropertyRequirements(
        summary.input_count,
        summary.input_assertion_count,
        summary.output_assertion_count,
    )


def _witness_rejection(
    summary: PropertySummary, values: Mapping[int, float]
) -> str | None:
    extra_count = 0
    extra_prefix: list[int] = []
    for index in values:
        if index < 0 or index >= summary.input_count:
            extra_count += 1
            if len(extra_prefix) < 16:
                extra_prefix.append(index)
    valid_count = len(values) - extra_count
    missing: list[int] = []
    if valid_count != summary.input_count:
        for index in range(summary.input_count):
            if index not in values:
                missing.append(index)
                if len(missing) == 16:
                    break
    if missing or extra_count:
        parts = []
        if missing:
            suffix = "..." if valid_count + len(missing) < summary.input_count else ""
            parts.append(
                "missing " + ", ".join(f"X_{index}" for index in missing) + suffix
            )
        if extra_count:
            suffix = "..." if extra_count > len(extra_prefix) else ""
            parts.append(
                "unexpected "
                + ", ".join(f"X_{index}" for index in extra_prefix)
                + suffix
            )
        return "incomplete witness: " + "; ".join(parts)
    nonfinite_count = 0
    nonfinite_prefix: list[int] = []
    for index, value in values.items():
        if not math.isfinite(value):
            nonfinite_count += 1
            if len(nonfinite_prefix) < 16:
                nonfinite_prefix.append(index)
    if nonfinite_count:
        suffix = "..." if nonfinite_count > len(nonfinite_prefix) else ""
        return (
            "non-finite witness values: "
            + ", ".join(f"X_{index}" for index in nonfinite_prefix)
            + suffix
        )
    return None


def _evaluate_domains(
    vnnlib_path: str | Path,
    raw: _VariableEnvironment,
    executed: _VariableEnvironment,
) -> tuple[_ResultAccumulator, _ResultAccumulator]:
    raw_results = _ResultAccumulator()
    executed_results = _ResultAccumulator()
    layout = raw.layout
    for expression in _file_expressions(vnnlib_path):
        assertion = _assertion(expression)
        if assertion is None or _references(assertion, "Y", layout):
            continue
        raw_results.add(_boolean(evaluate(assertion, raw)))
        executed_results.add(_boolean(evaluate(assertion, executed)))
    return raw_results, executed_results


def _evaluate_full(
    vnnlib_path: str | Path,
    executed: _VariableEnvironment,
) -> tuple[_ResultAccumulator, _ResultAccumulator]:
    all_results = _ResultAccumulator()
    output_results = _ResultAccumulator()
    layout = executed.layout
    for expression in _file_expressions(vnnlib_path):
        assertion = _assertion(expression)
        if assertion is None:
            continue
        result = _boolean(evaluate(assertion, executed))
        all_results.add(result)
        if _references(assertion, "Y", layout):
            output_results.add(result)
    return all_results, output_results


def validate(
    onnx_path: str | Path,
    vnnlib_path: str | Path,
    values: dict[int, float],
) -> tuple[bool, bool, str]:
    try:
        summary = _scan_property(vnnlib_path)
    except UnsupportedSyntaxError as error:
        # DISTINCT from "invalid property structure": that verdict describes a
        # property this validator READ and rejected.  This one says the file was
        # never read, so no admission decision is being made about it.
        return False, False, f"{UNSUPPORTED_SYNTAX_PREFIX} {error}"
    except (OSError, UnicodeError, RecursionError, ValidationError) as error:
        return False, False, f"invalid property structure: {error}"
    rejection = _witness_rejection(summary, values)
    if rejection is not None:
        return False, False, rejection

    raw_environment = _VariableEnvironment(values, layout=summary.layout)
    executed_environment = _VariableEnvironment(
        values, executed_inputs=True, layout=summary.layout
    )
    try:
        raw_domain, executed_domain = _evaluate_domains(
            vnnlib_path, raw_environment, executed_environment
        )
    except UnsupportedSyntaxError as error:
        return False, False, f"{UNSUPPORTED_SYNTAX_PREFIX} {error}"
    except (OSError, UnicodeError, RecursionError, ValidationError) as error:
        return False, False, f"invalid input assertion: {error}"
    # In-box gate = RAW-value domain check, matching the official zero-tol
    # semantics: SCORING-ZERO-TOL/counterexamples.py evaluates
    # `is_specification_vio(..., tuple(x_list), ..., 0.0)` on the raw parsed
    # witness values; the float32 cast happens only inside the ORT execution
    # (`np.array(x_list, dtype=input_dtype)`). Requiring the f32-cast inputs to
    # ALSO satisfy the bounds is strictly stronger than official and is
    # unsatisfiable for specs that pin inputs with equality pairs on
    # non-f32-representable constants (cctsdb_yolo: `(>= X_i 0.24313725)` /
    # `(<= X_i 0.24313725)` — f32 rounding always crosses one side), so it
    # rejects counterexamples the organizer scores CORRECT (abc/PyRAT banked 28
    # such falsifications at zero-tol in 2025). The executed-f32 domain result
    # is still computed and reported below as a diagnostic, never as a gate.
    in_box = raw_domain.all_hold
    if not in_box:
        return (
            False,
            False,
            f"in_box=False complete_inputs={summary.input_count} "
            f"input_assertions={summary.input_assertion_count} "
            f"raw_domain_results={raw_domain.detail()} "
            f"executed_domain_results={executed_domain.detail()}",
        )
    if not executed_domain.all_hold:
        # Diagnostic only (official semantics does not gate on this): the
        # witness is in-box on raw values but its f32 image crosses a bound —
        # expected for non-f32-representable pinned constants.
        print(
            "  note: raw-domain in-box holds; f32-executed domain crosses a "
            f"bound (diagnostic only): {executed_domain.detail()}",
            flush=True,
        )

    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    input_array = np.fromiter(
        (_float32(values[index]) for index in range(summary.input_count)),
        dtype=np.float32,
        count=summary.input_count,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model_inputs = session.get_inputs()
    if len(model_inputs) != 1:
        raise ValidationError(
            f"expected one ONNX input tensor, found {len(model_inputs)}"
        )
    model_input = model_inputs[0]
    shape = [
        dimension if isinstance(dimension, int) else 1
        for dimension in model_input.shape
    ]
    if math.prod(shape) != summary.input_count:
        raise ValidationError(
            f"ONNX input shape {shape} does not match {summary.input_count} declared inputs"
        )
    if summary.layout is not None and tuple(shape) != summary.layout.inputs.shape:
        # Element counts agreeing is NOT enough: the 2.0 flat index is a
        # row-major offset into the DECLARED shape, and `input_array.reshape`
        # below folds it into the ONNX shape.  If the two shapes differ, those
        # two flattenings disagree and a witness could be accepted against a
        # permuted box.  Refuse rather than guess.
        raise ValidationError(
            f"ONNX input shape {shape} does not match the declared VNN-LIB input "
            f"shape {list(summary.layout.inputs.shape)}; the row-major flat index "
            "mapping would be ambiguous"
        )
    model_outputs = session.run(None, {model_input.name: input_array.reshape(shape)})
    if len(model_outputs) != 1:
        raise ValidationError(
            f"expected one ONNX output tensor, found {len(model_outputs)}"
        )
    output = model_outputs[0].flatten().astype(np.float64)
    if any(not math.isfinite(float(value)) for value in output):
        raise ValidationError("ONNX produced a non-finite output")
    if summary.output_count > len(output):
        raise ValidationError(
            f"property declares {summary.output_count} outputs, but ONNX produced {len(output)}"
        )
    if summary.layout is not None and len(output) != summary.output_count:
        raise ValidationError(
            f"declared VNN-LIB output shape {list(summary.layout.outputs.shape)} has "
            f"{summary.output_count} elements, but ONNX produced {len(output)}"
        )

    # Full-tree evaluation at (RAW witness X, ORT-executed Y) — the official
    # zero-tol semantics (SCORING-ZERO-TOL/counterexamples.py:
    # `is_specification_vio(..., tuple(x_list), tuple(used_output), 0.0)`).
    # Casting X through f32 here (the previous behavior) made mixed and
    # input assertions unsatisfiable on non-f32-representable pinned
    # constants (cctsdb_yolo), rejecting officially-CORRECT counterexamples.
    full_environment = _VariableEnvironment(
        values, output, executed_inputs=False, layout=summary.layout
    )
    try:
        all_results, output_results = _evaluate_full(vnnlib_path, full_environment)
    except UnsupportedSyntaxError as error:
        return True, False, f"{UNSUPPORTED_SYNTAX_PREFIX} {error}"
    except (OSError, UnicodeError, RecursionError, ValidationError) as error:
        return True, False, f"invalid output assertion: {error}"
    detail = (
        f"in_box=True complete_inputs={summary.input_count} "
        f"input_assertions={summary.input_assertion_count} "
        f"yasserts={summary.output_assertion_count} all_hold={all_results.all_hold} "
        f"raw_domain_results={raw_domain.detail()} "
        f"executed_domain_results={executed_domain.detail()} "
        f"output_results={output_results.detail()}"
    )
    return True, all_results.all_hold, detail


def _extract_cli_assignment(source: str) -> dict[int, float]:
    values: dict[int, float] = {}
    assignment_count = 0
    for match in COUNTEREXAMPLE_ASSIGNMENT.finditer(source):
        assignment_count += 1
        if match.end(1) - match.start(1) > 8:
            raise ValidationError("counterexample input index is too long")
        if match.end(2) - match.start(2) > MAX_ATOM_CHARS:
            raise ValidationError("counterexample numeric value is too long")
        index_text, value_text = match.groups()
        if INDEX_TOKEN.fullmatch(index_text) is None:
            raise ValidationError(f"invalid counterexample input index X_{index_text}")
        index = int(index_text)
        if index > MAX_VARIABLE_INDEX:
            raise ValidationError(
                f"counterexample input index exceeds X_{MAX_VARIABLE_INDEX}"
            )
        if NUMBER_TOKEN.fullmatch(value_text) is None:
            raise ValidationError(f"X_{index} has an invalid numeric value")
        if index in values:
            raise ValidationError(f"duplicate assignment for X_{index}")
        values[index] = float(value_text)
    markers = sum(1 for _ in COUNTEREXAMPLE_ASSIGNMENT_MARKER.finditer(source))
    if assignment_count != markers:
        raise ValidationError("counterexample contains a malformed X_i assignment")
    if not values:
        if TENSOR_VARIABLE.search(source) is not None:
            # The WITNESS format is flat `(X_i v)` in every tool this reads,
            # independently of the dialect the SPEC is written in.  Say so
            # explicitly rather than let a tensor-named witness read as an
            # empty one.
            raise UnsupportedSyntaxError(
                "counterexample uses tensor-indexed assignments; this reader "
                "accepts only flat (X_i value) witness assignments"
            )
        raise ValidationError("counterexample contains no X_i assignments")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx")
    parser.add_argument("vnnlib")
    parser.add_argument("counterexample")
    args = parser.parse_args(argv)
    try:
        require_runtime_dependencies()
    except MissingDependencyError as error:
        print(f"ENVIRONMENT ERROR: {error}", file=sys.stderr)
        return 3
    counterexample = Path(args.counterexample)
    if counterexample.suffix == ".gz":
        with gzip.open(counterexample, "rt", encoding="utf-8") as source:
            text = source.read()
    else:
        text = counterexample.read_text(encoding="utf-8")
    values = _extract_cli_assignment(text)
    in_box, is_counterexample, detail = validate(args.onnx, args.vnnlib, values)
    print(detail)
    if not is_counterexample and str(detail).startswith(UNSUPPORTED_SYNTAX_PREFIX):
        # Never "OUT-OF-BOX": that would report a containment decision this run
        # never made.  Exit 4 keeps it distinct from a real 1 (not a CE).
        print("UNREADABLE-SPEC-NO-VERDICT")
        return 4
    print(
        "GENUINE-IN-BOX-CE"
        if is_counterexample
        else ("OUT-OF-BOX" if not in_box else "IN-BOX-BUT-NOT-CE")
    )
    return 0 if is_counterexample else 1


if __name__ == "__main__":
    raise SystemExit(main())
