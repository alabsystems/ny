#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Shared library of the M0 arming harness (docs/engine_program/ENGINE_PROGRAM.md section 2, M0).

Tooling only: pure python3 + PyYAML. Nothing here runs cargo, an ny binary or the box. Every generator
built on it REFUSES (raises M0Error) rather than emit an artefact it cannot vouch for:

* VNN-LIB: a small S-expression reader that keeps token offsets, so a knife-edge rewrite replaces exactly
  one numeric token and leaves every other byte of the property untouched (routing fidelity).
* Witnesses: `sat` result files and `.counterexample[.gz]` bank artefacts, `(X_i v)` / `(Y_i v)` pairs.
* Config overlays: the textual yaml editor of scratchpad/w8b/make_cfgs_v2.py (comments preserved), verified
  yaml-aware (parse(edited) == parse(base) + exactly the intended sets/deletes, and no duplicate key).
* The attack-off table: every field of the preset schema's AttackPreset (plus the attack/PGD fields of
  PhaseBudgetPreset and InputSplitPreset) must be classified, or the negative-control generator refuses.
* Row sources: instances.csv, banks and the field `results.txt`, aligned BY INDEX with the key asserted at
  every index (never assumed).
* The runner spec (one TSV schema for negative controls, knife-edge rows, A/B arms and moat rows) and the
  judge that turns one run's raw log + flight record into OK / VOID / EMERGENCY.
"""

from __future__ import annotations

import copy
import csv
import glob
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

import yaml

SCHEMA = "m0-spec/v1"
DEFAULT_DELTAS = ("1e-3", "1e-6", "1e-9", "1e-12")
DECIDED = ("sat", "unsat")
VERDICT_TOKENS = ("sat", "unsat", "unknown", "timeout", "error")


class M0Error(Exception):
    """A refusal: the harness will not produce or accept an artefact it cannot vouch for."""


# ---------------------------------------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_text(path: str) -> str:
    """Read a text file, transparently gunzipping `*.gz`."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as fh:
            return fh.read()
    with open(path, encoding="utf-8", errors="strict") as fh:
        return fh.read()


def git_head(path: str) -> str | None:
    """HEAD of the git tree containing `path`, or None (a copied configs dir has no provenance)."""
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


# ---------------------------------------------------------------------------------------------------------
# VNN-LIB reader (offsets kept)
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Node:
    """An S-expression node. `items` is None for an atom; `start`/`end` are offsets into the source text."""

    items: list | None
    text: str | None
    start: int
    end: int

    @property
    def is_list(self) -> bool:
        return self.items is not None

    def head(self) -> str | None:
        if self.items and not self.items[0].is_list:
            return self.items[0].text
        return None


def parse_sexprs(text: str) -> list[Node]:
    """Parse every top-level form. `;` starts a comment to end of line; `"..."` is one atom."""
    stack: list[tuple[int, list]] = []
    top: list[Node] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c.isspace():
            i += 1
            continue
        if c == "(":
            stack.append((i, []))
            i += 1
            continue
        if c == ")":
            if not stack:
                raise M0Error(f"vnnlib: unbalanced ')' at offset {i}")
            start, items = stack.pop()
            node = Node(items, None, start, i + 1)
            (stack[-1][1] if stack else top).append(node)
            i += 1
            continue
        j = i
        if c == '"':
            j = text.find('"', i + 1)
            if j < 0:
                raise M0Error(f"vnnlib: unterminated string at offset {i}")
            j += 1
        else:
            while j < n and not text[j].isspace() and text[j] not in "();":
                j += 1
        node = Node(None, text[i:j], i, j)
        (stack[-1][1] if stack else top).append(node)
        i = j
    if stack:
        raise M0Error(f"vnnlib: unclosed '(' at offset {stack[-1][0]}")
    return top


VAR_RE = re.compile(r"^([XY])_(\d+)$")
TENSOR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[([0-9,]+)\]$")
CMP_OPS = ("<=", ">=", "<", ">")

# VNN-LIB 2.0 network declaration of the property being parsed: {name: ('X'|'Y', shape)}
_NET: dict[str, tuple[str, tuple[int, ...]]] = {}


def _flat(shape: tuple[int, ...], idx: list[int], what: str) -> int:
    if len(idx) != len(shape) or any(i < 0 or i >= d for i, d in zip(idx, shape)):
        raise M0Error(f"vnnlib: index {idx} outside declared shape {list(shape)} of {what}")
    flat = 0
    for i, d in zip(idx, shape):
        flat = flat * d + i
    return flat


def _var(text: str):
    m = VAR_RE.match(text)
    if m:
        return (m.group(1), int(m.group(2)))
    m = TENSOR_RE.match(text)
    if m and m.group(1) in _NET:
        kind, shape = _NET[m.group(1)]
        return (kind, _flat(shape, [int(v) for v in m.group(2).split(",")], m.group(1)))
    return None


def _operand(node: Node):
    """('X', i) | ('Y', i) | ('C', float, node), i flattened row-major for VNN-LIB 2.0 tensor syntax.
    Anything else (arithmetic, an undeclared tensor) is refused."""
    if node.is_list:
        raise M0Error(f"vnnlib: unsupported compound operand {_snippet(node)}")
    v = _var(node.text or "")
    if v:
        return v
    try:
        value = float(node.text)
    except (TypeError, ValueError):
        raise M0Error(f"vnnlib: unsupported operand {node.text!r}") from None
    if not math.isfinite(value):
        raise M0Error(f"vnnlib: non-finite constant {node.text!r}")
    return ("C", value, node)


_SRC: list[str] = [""]


def _snippet(node: Node) -> str:
    s = _SRC[0][node.start:node.end]
    return (s[:80] + "...") if len(s) > 80 else s


@dataclass
class Atom:
    """One comparison, normalised to `big >= small` (strict when the source said `<`/`>`).

    kind: 'threshold' (one output vs a constant), 'relational' (two outputs), 'input' (an input bound).
    For a threshold, `direction` is 'ge' (Y >= c) or 'le' (Y <= c) and `const_node` is the token a knife-edge
    rewrite replaces.
    """

    kind: str
    strict: bool
    start: int
    end: int
    out: int | None = None
    direction: str | None = None
    const: float | None = None
    const_node: Node | None = None
    big: int | None = None
    small: int | None = None
    xin: int | None = None

    def slack(self, y: dict[int, Fraction]) -> Fraction:
        """Exact amount by which the atom holds at outputs `y` (negative = violated)."""
        if self.kind == "threshold":
            yv = y[self.out]
            c = Fraction(self.const)
            return yv - c if self.direction == "ge" else c - yv
        if self.kind == "relational":
            return y[self.big] - y[self.small]
        raise M0Error("slack of an input atom")

    def holds(self, y: dict[int, Fraction]) -> bool:
        s = self.slack(y)
        return s > 0 if self.strict else s >= 0

    def outputs(self) -> list[int]:
        if self.kind == "threshold":
            return [self.out]
        if self.kind == "relational":
            return [self.big, self.small]
        return []


def _atom(node: Node) -> Atom:
    op = node.head()
    if op not in CMP_OPS or len(node.items) != 3:
        raise M0Error(f"vnnlib: unsupported atom {_snippet(node)}")
    left, right = _operand(node.items[1]), _operand(node.items[2])
    strict = op in ("<", ">")
    big, small = (left, right) if op in (">=", ">") else (right, left)
    if big[0] == "Y" and small[0] == "C":
        return Atom("threshold", strict, node.start, node.end, out=big[1], direction="ge", const=small[1],
                    const_node=small[2])
    if big[0] == "C" and small[0] == "Y":
        return Atom("threshold", strict, node.start, node.end, out=small[1], direction="le", const=big[1],
                    const_node=big[2])
    if big[0] == "Y" and small[0] == "Y":
        return Atom("relational", strict, node.start, node.end, big=big[1], small=small[1])
    if big[0] == "X" and small[0] == "C":
        return Atom("input", strict, node.start, node.end, xin=big[1], direction="ge", const=small[1])
    if big[0] == "C" and small[0] == "X":
        return Atom("input", strict, node.start, node.end, xin=small[1], direction="le", const=big[1])
    raise M0Error(f"vnnlib: unsupported comparison {_snippet(node)}")


MAX_DNF = 100_000


def _dnf(node: Node) -> list[list[Atom]]:
    if not node.is_list:
        raise M0Error(f"vnnlib: bare atom in a formula: {node.text!r}")
    head = node.head()
    if head == "and":
        out: list[list[Atom]] = [[]]
        for sub in node.items[1:]:
            out = [a + b for a in out for b in _dnf(sub)]
            if len(out) > MAX_DNF:
                raise M0Error("vnnlib: DNF blow-up")
        return out
    if head == "or":
        out = []
        for sub in node.items[1:]:
            out.extend(_dnf(sub))
        return out
    return [[_atom(node)]]


def _vars(node: Node) -> set[str]:
    if node.is_list:
        s: set[str] = set()
        for it in node.items:
            s |= _vars(it)
        return s
    v = _var(node.text or "")
    return {v[0]} if v else set()


@dataclass
class Property:
    text: str
    n_x: int
    n_y: int
    lower: dict[int, float]
    upper: dict[int, float]
    clauses: list[list[Atom]]
    in_name: str = "X"
    out_name: str = "Y"
    version: str = "1.0"


def _shape(items: list[Node], what: str) -> tuple[int, ...]:
    raw = " ".join(n.text or "" for n in items)
    m = re.match(r"^\[\s*([0-9,\s]+)\]$", raw.strip())
    if not m:
        raise M0Error(f"vnnlib: unsupported shape {raw!r} in {what}")
    return tuple(int(v) for v in m.group(1).replace(",", " ").split())


def _declare_network(form: Node) -> tuple[str, int, str, int]:
    """(declare-network N (declare-input X float32 [..]) (declare-output Y float32 [..])) -> names, sizes."""
    ins = [it for it in form.items[2:] if it.is_list and it.head() == "declare-input"]
    outs = [it for it in form.items[2:] if it.is_list and it.head() == "declare-output"]
    other = [it for it in form.items[2:] if not (it.is_list and it.head() in ("declare-input", "declare-output"))]
    if len(ins) != 1 or len(outs) != 1 or other:
        raise M0Error(f"vnnlib: unsupported network declaration {_snippet(form)}")
    res = []
    for decl, kind in ((ins[0], "X"), (outs[0], "Y")):
        if len(decl.items) < 4 or decl.items[1].is_list:
            raise M0Error(f"vnnlib: unsupported tensor declaration {_snippet(decl)}")
        name = decl.items[1].text
        shape = _shape(decl.items[3:], name)
        _NET[name] = (kind, shape)
        res.extend([name, math.prod(shape)])
    return res[0], res[1], res[2], res[3]


def parse_property(text: str) -> Property:
    """VNN-LIB 1.0 (`declare-const X_i`) and the single-network VNN-LIB 2.0 subset (`vnnlib-version`,
    `declare-network` with one input and one output tensor, `X[i,j,..]` flattened row-major): input-only
    asserts (atoms / `and`), output-only asserts (any and/or nesting of comparisons). Mixed input/output
    asserts, disjunct-scoped input atoms, arithmetic and multi-network declarations are refused: the
    rewriter would otherwise have to guess their semantics."""
    _SRC[0] = text
    _NET.clear()
    n_x = n_y = 0
    in_name, out_name, version = "X", "Y", "1.0"
    lower: dict[int, float] = {}
    upper: dict[int, float] = {}
    clauses: list[list[Atom]] = [[]]
    seen_output = False
    for form in parse_sexprs(text):
        if not form.is_list or not form.items:
            raise M0Error(f"vnnlib: stray top-level token {form.text!r}")
        head = form.head()
        if head == "vnnlib-version":
            version = " ".join(n.text or "" for n in form.items[1:]).strip("<> ")
            continue
        if head == "declare-network":
            if _NET:
                raise M0Error("vnnlib: more than one declared network (dual-network properties unsupported)")
            in_name, n_x, out_name, n_y = _declare_network(form)
            continue
        if head == "declare-const":
            m = VAR_RE.match(form.items[1].text or "") if len(form.items) >= 2 else None
            if not m:
                raise M0Error(f"vnnlib: unsupported declaration {_snippet(form)}")
            idx = int(m.group(2)) + 1
            if m.group(1) == "X":
                n_x = max(n_x, idx)
            else:
                n_y = max(n_y, idx)
            continue
        if head != "assert" or len(form.items) != 2:
            raise M0Error(f"vnnlib: unsupported top-level form {_snippet(form)}")
        body = form.items[1]
        kinds = _vars(body)
        if kinds == {"X"}:
            d = _dnf(body)
            if len(d) != 1:
                raise M0Error(f"vnnlib: disjunctive input constraint {_snippet(body)}")
            for a in d[0]:
                if a.kind != "input":
                    raise M0Error(f"vnnlib: unsupported input atom {_snippet(body)}")
                if a.direction == "ge":
                    lower[a.xin] = max(lower.get(a.xin, -math.inf), a.const)
                else:
                    upper[a.xin] = min(upper.get(a.xin, math.inf), a.const)
            continue
        if kinds == {"Y"}:
            seen_output = True
            d = _dnf(body)
            clauses = [a + b for a in clauses for b in d]
            if len(clauses) > MAX_DNF:
                raise M0Error("vnnlib: DNF blow-up")
            continue
        raise M0Error(f"vnnlib: assert mixing inputs and outputs (or neither) is unsupported: {_snippet(body)}")
    if not seen_output:
        raise M0Error("vnnlib: no output constraint")
    if n_x == 0 or n_y == 0:
        raise M0Error("vnnlib: no declared inputs or outputs")
    for i in range(n_x):
        if i not in lower or i not in upper:
            raise M0Error(f"vnnlib: input X_{i} is not bounded on both sides")
    return Property(text, n_x, n_y, lower, upper, clauses, in_name, out_name, version)


# ---------------------------------------------------------------------------------------------------------
# witnesses
# ---------------------------------------------------------------------------------------------------------

WITNESS_PAIR = re.compile(r"\(\s*([XY])_(\d+)\s+([^\s()]+)\s*\)")


@dataclass
class Witness:
    path: str
    sha256: str
    x: dict[int, float]
    y: dict[int, float]


BLOCK_HDR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z0-9_]+)\s+\[([0-9,\s]*)\]\s*$")


def _witness_blocks(path: str, lines: list[str], in_name: str, out_name: str) -> tuple[dict, dict]:
    """VNN-LIB 2.0 counterexample: `<name> <dtype> [dims]` then prod(dims) values, one per line."""
    blocks: dict[str, list[float]] = {}
    i = 0
    while i < len(lines):
        m = BLOCK_HDR.match(lines[i].strip())
        if not m:
            raise M0Error(f"witness {path}: expected a tensor header, got {lines[i][:60]!r}")
        name = m.group(1)
        n = math.prod(int(v) for v in m.group(3).replace(",", " ").split())
        vals = lines[i + 1:i + 1 + n]
        if len(vals) != n or name in blocks:
            raise M0Error(f"witness {path}: tensor {name} truncated or repeated")
        try:
            blocks[name] = [float(v) for v in vals]
        except ValueError:
            raise M0Error(f"witness {path}: non-numeric value in tensor {name}") from None
        i += 1 + n
    if set(blocks) != {in_name, out_name}:
        raise M0Error(f"witness {path}: tensors {sorted(blocks)} (want {in_name} and {out_name})")
    return dict(enumerate(blocks[in_name])), dict(enumerate(blocks[out_name]))


def load_witness(path: str, in_name: str = "X", out_name: str = "Y") -> Witness:
    """A `sat` result file (`sat` line then assignments) or a bank `.counterexample[.gz]` (assignments only),
    in the VNN-LIB 1.0 `(X_i v)` form or the 2.0 tensor-block form. Refuses a non-sat result file, a
    duplicate, missing or non-finite assignment, and a file with no X or no Y."""
    text = read_text(path)
    first = text.lstrip().split(None, 1)[0] if text.strip() else ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].strip() in ("sat", "violated"):
        lines = lines[1:]
        first = lines[0].lstrip().split(None, 1)[0] if lines else ""
    if first in ("unsat", "unknown", "timeout", "error", "holds", ""):
        raise M0Error(f"witness {path}: result '{first}' carries no counterexample")
    if not first.startswith("("):
        x, y = _witness_blocks(path, lines, in_name, out_name)
        for d in (x, y):
            if any(not math.isfinite(v) for v in d.values()):
                raise M0Error(f"witness {path}: non-finite value")
        return Witness(path, sha256_file(path), x, y)
    x: dict[int, float] = {}
    y: dict[int, float] = {}
    for kind, idx, val in WITNESS_PAIR.findall(text):
        v = float(val)
        if not math.isfinite(v):
            raise M0Error(f"witness {path}: non-finite {kind}_{idx}")
        tgt = x if kind == "X" else y
        if int(idx) in tgt:
            raise M0Error(f"witness {path}: duplicate {kind}_{idx}")
        tgt[int(idx)] = v
    if not x or not y:
        raise M0Error(f"witness {path}: no X or no Y assignment")
    return Witness(path, sha256_file(path), x, y)


def load_y_values(path: str) -> dict[int, float]:
    """An outputs-only file in the witness format (e.g. a higher-precision forward at x*)."""
    y: dict[int, float] = {}
    for kind, idx, val in WITNESS_PAIR.findall(read_text(path)):
        if kind == "Y":
            v = float(val)
            if not math.isfinite(v) or int(idx) in y:
                raise M0Error(f"y-values {path}: bad or duplicate Y_{idx}")
            y[int(idx)] = v
    if not y:
        raise M0Error(f"y-values {path}: no Y assignment")
    return y


def witness_in_box(prop: Property, x: dict[int, float]) -> Fraction:
    """Exact check that x* lies in the input box; returns the smallest distance to a face."""
    best: Fraction | None = None
    for i in range(prop.n_x):
        if i not in x:
            raise M0Error(f"witness lacks X_{i}")
        v, lo, hi = Fraction(x[i]), Fraction(prop.lower[i]), Fraction(prop.upper[i])
        if v < lo or v > hi:
            raise M0Error(f"witness X_{i}={x[i]!r} lies outside [{prop.lower[i]!r}, {prop.upper[i]!r}]")
        d = min(v - lo, hi - v)
        best = d if best is None else min(best, d)
    extra = sorted(k for k in x if k >= prop.n_x)
    if extra:
        raise M0Error(f"witness assigns undeclared inputs {extra[:5]}")
    return best if best is not None else Fraction(0)


# ---------------------------------------------------------------------------------------------------------
# knife-edge arithmetic
# ---------------------------------------------------------------------------------------------------------


def f64_floor(q: Fraction) -> float:
    """Largest f64 <= q (exact)."""
    x = float(q)
    if Fraction(x) > q:
        x = math.nextafter(x, -math.inf)
    return x


def f64_ceil(q: Fraction) -> float:
    """Smallest f64 >= q (exact)."""
    x = float(q)
    if Fraction(x) < q:
        x = math.nextafter(x, math.inf)
    return x


def ulp_f32(y: float) -> float:
    """Spacing of binary32 values at |y| (the resolution of an f32 network output)."""
    a = abs(y)
    if a < 2.0 ** -126:
        return 2.0 ** -149
    _, e = math.frexp(a)
    return 2.0 ** (e - 24)


def plain_decimal(x: float) -> str:
    """A decimal spelling with no exponent that parses back to exactly `x` (Rust's f64 parser is correctly
    rounded, so an exact decimal of the f64 value round-trips)."""
    if x == 0.0:
        return "0.0"
    s = repr(x)
    if "e" in s or "E" in s:
        s = format(Decimal(s), "f")
        if "." not in s:
            s += ".0"
    if float(s) != x:
        raise M0Error(f"decimal spelling {s} does not round-trip {x!r}")
    return s


def fmt_sig(q: Fraction, digits: int = 6) -> str:
    return f"{float(q):.{digits}e}"


def knife_edge_rewrite(prop: Property, y: dict[int, float], delta: str, clause: str | int = "auto",
                       allow_partial: bool = False, allow_non_witness: bool = False) -> tuple[str, dict]:
    """Rewrite every threshold atom of the selected clause(s) so the point with outputs `y` satisfies it
    with an exact slack of `delta` (the smallest f64 threshold that keeps slack >= delta).

    clause='auto' selects every clause that already holds at y (a witnessed clause); an int selects that one
    clause (its relational atoms must hold at y). A selected clause with no threshold atom cannot be moved to
    the edge (ny refuses offsets on relational atoms: normalize/convert.rs "Relational constraint must have
    zero constant"); that is refused unless allow_partial. Clauses that do not hold at y are left untouched.
    """
    d = Fraction(Decimal(delta))
    if d <= 0:
        raise M0Error(f"delta must be > 0, got {delta}")
    yq = {k: Fraction(v) for k, v in y.items()}
    needed = sorted({o for cl in prop.clauses for a in cl for o in a.outputs()})
    missing = [o for o in needed if o not in yq]
    if missing:
        raise M0Error(f"y-values lack outputs {missing[:8]} referenced by the property")
    holding = [k for k, cl in enumerate(prop.clauses) if all(a.holds(yq) for a in cl)]
    if clause == "auto":
        if not holding:
            if allow_non_witness:
                raise M0Error("allow_non_witness needs an explicit clause index (no clause holds at y)")
            raise M0Error("x* is not a witness: no clause of the property holds at its outputs")
        selected = holding
    else:
        k = int(clause)
        if not 0 <= k < len(prop.clauses):
            raise M0Error(f"clause {k} out of range 0..{len(prop.clauses) - 1}")
        if k not in holding and not allow_non_witness:
            raise M0Error(f"clause {k} does not hold at x* (pass allow_non_witness for a synthetic sat)")
        bad = [a for a in prop.clauses[k] if a.kind == "relational" and not a.holds(yq)]
        if bad:
            raise M0Error(f"clause {k}: a relational atom is violated at x*; no threshold shift can fix it")
        selected = [k]
    rewrites: dict[int, dict] = {}
    partial = []
    for k in selected:
        cl = prop.clauses[k]
        th = [a for a in cl if a.kind == "threshold"]
        if not th:
            partial.append(k)
            continue
        for a in th:
            yv = yq[a.out]
            new = f64_floor(yv - d) if a.direction == "ge" else f64_ceil(yv + d)
            slack = (yv - Fraction(new)) if a.direction == "ge" else (Fraction(new) - yv)
            if slack < d or (a.strict and slack <= 0):
                raise M0Error("internal: rewritten slack below delta")
            prev = rewrites.get(a.const_node.start)
            if prev is not None and prev["new"] != new:
                raise M0Error("a shared threshold token would need two different values")
            rewrites[a.const_node.start] = {
                "clause": k, "output": a.out, "direction": a.direction, "strict": a.strict,
                "old": a.const, "old_text": a.const_node.text, "y": y[a.out], "new": new,
                "new_text": plain_decimal(new), "slack": fmt_sig(slack),
                "slack_minus_delta": fmt_sig(slack - d), "ulp_f32_y": ulp_f32(y[a.out]),
                "natural_slack": fmt_sig(a.slack(yq)), "_node": a.const_node,
            }
    if partial and not allow_partial:
        raise M0Error(
            f"clause(s) {partial[:6]} hold at x* but carry no threshold atom (relational only): the knife-edge "
            "is not expressible in VNN-LIB for ny (a relational atom cannot take an offset); pass allow_partial "
            "to emit the rung with those clauses at their natural slack")
    text = prop.text
    for start in sorted(rewrites, reverse=True):
        r = rewrites[start]
        node = r["_node"]
        text = text[:node.start] + r["new_text"] + text[node.end:]
    report_rewrites = [{k: v for k, v in r.items() if k != "_node"} for _, r in sorted(rewrites.items())]
    touched = sorted({r["clause"] for r in report_rewrites})
    report = {
        "delta": delta,
        "clauses_total": len(prop.clauses),
        "clauses_holding_at_witness": holding[:50],
        "clauses_rewritten": touched,
        "clauses_relational_only_at_natural_slack": partial,
        "knife_edge_exact": not partial and bool(report_rewrites),
        "sub_f32_resolution": any(Fraction(Decimal(delta)) < Fraction(r["ulp_f32_y"]) for r in report_rewrites),
        "rewrites": report_rewrites,
    }
    return text, report


# ---------------------------------------------------------------------------------------------------------
# yaml overlays (textual editor from scratchpad/w8b/make_cfgs_v2.py, plus delete, plus verification)
# ---------------------------------------------------------------------------------------------------------

KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_-]+):(\s.*)?$")


class StrictLoader(yaml.SafeLoader):
    """SafeLoader that REFUSES a duplicate mapping key (serde refuses one at preset load; PyYAML would
    silently keep the last)."""


def _strict_mapping(loader, node, deep=False):
    seen = set()
    for knode, _ in node.value:
        k = loader.construct_object(knode, deep=deep)
        if k in seen:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {k!r}", knode.start_mark)
        seen.add(k)
    return loader.construct_mapping(node, deep=deep)


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _strict_mapping)


def strict_load(text: str):
    return yaml.load(text, Loader=StrictLoader)


def fmt_yaml(v) -> str:
    """A scalar spelling both PyYAML (YAML 1.1) and serde_yaml (1.2) read as the same typed value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = plain_decimal(v)
        return s if "." in s else s + ".0"
    if isinstance(v, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*$", v) and v not in ("true", "false", "null"):
        return v
    raise M0Error(f"unsupported yaml scalar {v!r}")


def _scan(lines: list[str]) -> dict[str, tuple[int, int, bool]]:
    out, stack = {}, []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("- "):
            continue
        m = KEY_RE.match(ln.rstrip("\n"))
        if not m:
            continue
        ind = len(m.group(1))
        while stack and stack[-1][0] >= ind:
            stack.pop()
        stack.append((ind, m.group(2)))
        rest = (m.group(3) or "").split("#", 1)[0].strip()
        out[".".join(k for _, k in stack)] = (i, ind, bool(rest))
    return out


def _child_indent(lines: list[str], hdr_idx: int, hdr_ind: int) -> int:
    for ln in lines[hdr_idx + 1:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        ind = len(ln) - len(ln.lstrip(" "))
        return ind if ind > hdr_ind else hdr_ind + 2
    return hdr_ind + 2


def yaml_set_key(lines: list[str], dotted: str, value, tag: str) -> None:
    keys = dotted.split(".")
    idx = _scan(lines)
    if dotted in idx:
        i, ind, scalar = idx[dotted]
        if not scalar:
            raise M0Error(f"refusing: {dotted} exists as a mapping, not a scalar")
        lines[i] = " " * ind + f"{keys[-1]}: {fmt_yaml(value)}  # {tag} (was: {lines[i].strip()})\n"
        return
    for k in range(len(keys) - 1, 0, -1):
        pre = ".".join(keys[:k])
        if pre in idx:
            i, ind, scalar = idx[pre]
            if scalar:
                raise M0Error(f"refusing: {pre} is a scalar, cannot nest {dotted}")
            ci = _child_indent(lines, i, ind)
            new = []
            for j, key in enumerate(keys[k:]):
                last = j == len(keys) - k - 1
                new.append(" " * (ci + 2 * j) + (f"{key}: {fmt_yaml(value)}  # {tag}\n" if last
                                                 else f"{key}:  # {tag}\n"))
            lines[i + 1:i + 1] = new
            return
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    for j, key in enumerate(keys):
        last = j == len(keys) - 1
        lines.append(" " * (2 * j) + (f"{key}: {fmt_yaml(value)}  # {tag}\n" if last
                                      else f"{key}:  # {tag} (new section)\n"))


def yaml_delete_key(lines: list[str], dotted: str, tag: str) -> bool:
    """Replace a SCALAR key's line with a comment. Returns False when the key is absent. A mapping, or a
    scalar whose value continues on the next line, is refused (the textual editor cannot see its extent)."""
    idx = _scan(lines)
    if dotted not in idx:
        return False
    i, ind, scalar = idx[dotted]
    if not scalar:
        raise M0Error(f"refusing to delete {dotted}: it is a mapping")
    for ln in lines[i + 1:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if len(ln) - len(ln.lstrip(" ")) > ind:
            raise M0Error(f"refusing to delete {dotted}: its value continues on the next line")
        break
    lines[i] = " " * ind + f"# {tag}: deleted `{lines[i].strip()}`\n"
    return True


def deep_get(d, dotted: str, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_has(d, dotted: str) -> bool:
    sentinel = object()
    return deep_get(d, dotted, sentinel) is not sentinel


def deep_set(d, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        if d.get(k) is None:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def deep_del(d, dotted: str) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    del d[keys[-1]]


def apply_overlay(base_text: str, sets: dict, deletes: list[str], tag: str) -> str:
    """Textual edit + yaml-aware verification: the edited file must parse (strictly) to exactly the base
    document with `sets` applied and `deletes` removed."""
    base_doc = strict_load(base_text) or {}
    lines = base_text.splitlines(keepends=True)
    want = copy.deepcopy(base_doc)
    for k in deletes:
        if deep_has(want, k):
            yaml_delete_key(lines, k, tag)
            deep_del(want, k)
    for k, v in sets.items():
        yaml_set_key(lines, k, v, tag)
        deep_set(want, k, v)
    out = "".join(lines)
    try:
        got = strict_load(out) or {}
    except yaml.YAMLError as e:
        raise M0Error(f"overlay produced invalid yaml: {e}") from None
    if got != want:
        raise M0Error("overlay verification failed: parse(edited) != parse(base) + edits")
    return out


def copy_configs(src: str, dst: str) -> None:
    """Copy a configs tree. Refuses an existing destination (this harness never deletes anything)."""
    if os.path.exists(dst):
        raise M0Error(f"refusing: {dst} already exists (this harness never deletes; pick a new --out)")
    if not os.path.isdir(src):
        raise M0Error(f"configs dir missing: {src}")
    shutil.copytree(src, dst, symlinks=False)


def resolve_preset(configs_dir: str, category: str) -> str | None:
    """Mirror of commands::vnncomp::resolve_preset_path: `vnncomp*` dirs newest (name-descending) first, the
    full lower-cased category name, then the name without a trailing `_20NN`. Returns a path RELATIVE to
    configs_dir. The runner re-checks every row with the binary's own resolver (`ny vnncomp plan`)."""
    lower = category.lower()
    base = re.sub(r"_20[0-9][0-9]$", "", lower)
    cands = [lower] if base == lower else [lower, base]
    base_name = os.path.basename(os.path.normpath(configs_dir))
    if base_name.startswith("vnncomp") and os.path.isdir(configs_dir):
        years = [configs_dir]
    else:
        years = sorted((p for p in glob.glob(os.path.join(configs_dir, "vnncomp*")) if os.path.isdir(p)),
                       key=os.path.basename, reverse=True)
    for yd in years:
        for c in cands:
            p = os.path.join(yd, c + ".yaml")
            if os.path.isfile(p):
                return os.path.relpath(p, configs_dir)
    return None


# ---------------------------------------------------------------------------------------------------------
# the attack-off table ("every attack key off", ENGINE_PROGRAM M0 item 2)
# ---------------------------------------------------------------------------------------------------------

# action: master = always written; set = rewritten only when present with another value (absent is already
# the off value); delete = removed when present (absent is off); keep = a parameter of a lane the masters
# switch off, or a key that is not a sat-finding switch. Every schema field must appear here (fail closed).
ATTACK_TABLE: dict[str, tuple[str, object, str]] = {
    # --- masters
    "attack.pgd_order": ("master", "skip", "engine PGD off: enable_pgd_attack=false (preset/apply.rs:1011)"),
    "attack.upfront_attack": ("master", False, "typed kill: UpfrontAttackMaster::DisabledByPreset"),
    "bab.phase_budget.vnncomp_post_bab_attack": ("master", False, "post-BaB wrapper falsifier off"),
    "bab.phase_budget.post_bab_pgd_fraction": ("master", 0.0, "with the wrapper off: the tail goes to proof "
                                               "(PostBabWrapperRoute::ProofTailByPreset)"),
    "bab.phase_budget.upfront_pgd_fraction": ("master", 0.0, "no upfront PGD slice"),
    "bab.phase_budget.disjunctive_pgd_fraction": ("master", 0.0, "no disjunctive global PGD slice"),
    "bab.phase_budget.attack_extension_fraction": ("master", 0.0, "no adaptive attack extension"),
    # --- invalid or forcing when the masters are off
    "attack.ny_pgd_order_compat": ("delete", None, "refused with a disabled pgd_order (apply.rs:69-73)"),
    "attack.upfront_attack_probe_first": ("delete", None, "absent == disarmed; an explicit false is refused "
                                          "by only_the_traffic_signs_preset_arms_the_probe_first_lane"),
    "attack.vnncomp_upfront_relational_exact_gradient": ("delete", None, "widens upfront admission"),
    "attack.f64_target_margin_ulps": ("delete", None, "arms the post-BaB f64 repair (0 is invalid)"),
    "attack.f64_lift_step": ("delete", None, "valid only with f64_target_margin_ulps"),
    "attack.f64_basin_hops": ("delete", None, "valid only with f64_lift_step: lp_maximin"),
    "bab.phase_budget.disjunctive_pgd_min_secs": ("delete", None, "an absolute FLOOR would force PGD seconds"),
    "bab.phase_budget.post_bab_reserve_secs": ("delete", None, "a reserve for the disabled wrapper"),
    # --- independent sat-finding lanes, off
    "attack.falsify_portfolio": ("set", False, "falsification portfolio lane"),
    "attack.bnn_sign_space": ("set", False, "BNN LP sign-space lane"),
    "attack.bnn_ste_pgd": ("set", False, "BNN STE-PGD lane"),
    "attack.point_folded_twin": ("set", False, "post-BaB point-folded twin attack"),
    "attack.relational_point_vjp": ("set", False, "original-tower point-VJP attack/template advice"),
    "attack.open_disjunct_restrict": ("set", False, "per-open-disjunct attack stage"),
    "attack.dense_low_dim_sweep": ("set", False, "dense grid sweep pre-PGD phase"),
    "attack.pgd_disj_odi": ("set", False, "disjunctive ODI lanes"),
    "attack.pgd_disj_focus_lanes": ("set", 0, "disjunctive focus lanes"),
    "attack.post_bab_restarts": ("set", 0, "post-BaB diversified restarts"),
    "attack.corner_seed_restarts": ("set", 0, "corner-seeded restarts"),
    "attack.flat_surrogate_transfer": ("set", False, "surrogate->real transfer stage"),
    "bab.phase_budget.sequential_early_quit_attack": ("set", False, "sequential early-quit PGD fallback"),
    "bab.branching.input_split.lowdim_leaf_attack_every": ("set", 0, "lowdim leaf attack"),
    "attack.concurrent": ("set", False, "#attack-concurrent races the UPFRONT attack against the proof; the masters "
                          "already switch that attack off, and an armed race is pinned dark so a control never "
                          "runs one"),
    # --- parameters of lanes the masters switch off, or not sat-finding switches
    **{f"attack.{k}": ("keep", None, "parameter of a lane the masters switch off") for k in (
        "pgd_restarts", "pgd_steps", "pgd_exact_batched", "pgd_fast_grad", "pgd_vjp_batch", "pgd_vjp_k",
        "pgd_legacy_lanes", "pgd_odi_steps", "pgd_odi_eta", "pgd_vjp_steps", "pgd_lane_window", "pgd_exploit",
        "pgd_hinge_tau", "pgd_alpha", "pgd_alpha_scale", "pgd_lr_decay", "attack_tolerance",
        "pgd_restart_when_stuck", "upfront_attack_cap_secs", "upfront_attack_auto_cap_secs",
        "upfront_attack_fraction", "bnn_attack_fraction", "bnn_attack_cap_secs", "active_set_repair_secs",
        "attack_mode", "osi_steps", "surrogate_sign_gradient", "dense_sweep_max_dims", "dense_sweep_points",
        "upfront_step_geometry", "open_disjunct_min_secs", "open_disjunct_max_secs", "open_disjunct_restarts",
        "open_disjunct_root_wait_secs", "open_disjunct_max_frac", "bnn_sign_space_minimal_move",
        "bnn_sign_space_trust_region", "falsify_portfolio_secs", "flat_surrogate_max_blocks",
        "pre_softmax_objective", "lane_budget_frame", "lane_budget_allocator", "lane_value_scheduler",
        "graph_free_candidate_sidefile")},
    "bab.phase_budget.disjunctive_pgd_max_secs": ("keep", None, "a ceiling on a zero slice"),
    "bab.phase_budget.disjunctive_pgd_from_phase_start": ("keep", None, "anchoring of a zero slice"),
    "bab.phase_budget.disjunctive_pgd_stall_window_fraction": ("keep", None, "cutoff of a zero slice"),
    "bab.phase_budget.demand_sized_disjunctive_pgd": ("keep", None, "sizing rule of a zero slice"),
    "bab.phase_budget.demand_sized_post_bab_reserve": ("keep", None, "sizing rule of the (proof-owned) tail"),
}

# the schema structs whose fields the table must classify, and which of their fields are in scope
SCHEMA_SCOPE = (
    ("AttackPreset", "attack.", None),
    ("PhaseBudgetPreset", "bab.phase_budget.", re.compile(r"attack|pgd|post_bab")),
    ("InputSplitPreset", "bab.branching.input_split.", re.compile(r"attack")),
)

# the flight-receipt levers that must read "off" on every negative-control run (names and JSON types
# verified against scratchpad/w10b/v2/FDB_t1513_fdb_satctl.res.flight.json)
NEGCTL_FLIGHT = (
    "lever:NY_UPFRONT_ATTACK=false@config",
    "lever:NY_POSTBAB_ATTACK=false@config",
    "lever:NY_UPFRONT_ATTACK_PROBE_FIRST=false@*",
    "lever:NY_FALSIFY_PORTFOLIO=false@*",
    "lever:NY_BNN_SIGN_SPACE=false@*",
    "lever:NY_BNN_STE_PGD=false@*",
    "lever:NY_ATTACK_POINT_FOLDED_TWIN=false@*",
    "lever:NY_POSTBAB_RESTARTS=0@*",
    "lever:NY_PGD_DISJ_ODI=false@*",
    "lever:NY_PGD_DISJ_FOCUS_LANES=0@*",
    "lever:NY_LOWDIM_LEAF_ATTACK_EVERY=0@*",
    "event:upfront_attack_policy:master=disabled_by_preset route=disabled_by_preset ",
    "event:postbab_wrapper_policy:disabled_by_preset=1 route=(disabled_by_preset|proof_tail_by_preset)$",
)


def schema_fields(schema_path: str, struct: str) -> list[str]:
    """Field names of `pub(crate) struct <struct> { ... }` in the preset schema source."""
    if not os.path.isfile(schema_path):
        raise M0Error(f"preset schema source missing: {schema_path}")
    src = open(schema_path, encoding="utf-8").read()
    m = re.search(r"^pub\(crate\) struct " + re.escape(struct) + r" \{\n(.*?)^\}", src, re.S | re.M)
    if not m:
        raise M0Error(f"struct {struct} not found in {schema_path}")
    fields = re.findall(r"^    pub\(crate\) ([a-z_0-9]+):", m.group(1), re.M)
    if not fields:
        raise M0Error(f"struct {struct} in {schema_path} has no fields (positive control failed)")
    return fields


def check_attack_table(schema_path: str) -> list[str]:
    """Every in-scope schema field must be classified and every classified key must exist in the schema.
    Returns the in-scope dotted keys; raises M0Error listing every violation at once."""
    in_scope: list[str] = []
    problems: list[str] = []
    for struct, prefix, filt in SCHEMA_SCOPE:
        for f in schema_fields(schema_path, struct):
            if filt is None or filt.search(f):
                in_scope.append(prefix + f)
    for k in in_scope:
        if k not in ATTACK_TABLE:
            problems.append(f"UNCLASSIFIED attack key {k} (add it to ATTACK_TABLE with a reason)")
    for k in ATTACK_TABLE:
        if k not in in_scope:
            problems.append(f"STALE table key {k} (not in the schema at {schema_path})")
    if problems:
        raise M0Error("attack table does not match the preset schema:\n  " + "\n  ".join(problems))
    return in_scope


def attack_off_edits(doc: dict) -> tuple[dict, list[str], list[str]]:
    """(sets, deletes, log) that switch every attack key off in one category document."""
    sets: dict = {}
    deletes: list[str] = []
    log: list[str] = []
    for key, (action, off, why) in ATTACK_TABLE.items():
        present = deep_has(doc, key)
        cur = deep_get(doc, key)
        if action == "master":
            sets[key] = off
            log.append(f"set {key}: {fmt_yaml(off)} (was {cur!r}) -- {why}")
        elif action == "set" and present and cur != off:
            sets[key] = off
            log.append(f"set {key}: {fmt_yaml(off)} (was {cur!r}) -- {why}")
        elif action == "delete" and present:
            deletes.append(key)
            log.append(f"delete {key} (was {cur!r}) -- {why}")
    return sets, deletes, log


# ---------------------------------------------------------------------------------------------------------
# tiers (ENGINE_PROGRAM M0 items 2 and 6; engagement lines verified against the base source and W-series logs)
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    name: str
    capability: str
    categories: tuple[str, ...]
    arm: dict
    need: tuple[str, ...]
    alarm: tuple[str, ...]
    flight: tuple[str, ...]
    markers: tuple[str, ...]
    rows: tuple[tuple[str, str, str, str], ...]  # (row id, category, onnx substring, vnnlib substring)
    class_alarm: str | None = None


MR_UNSAT = r"margin-row BaB \(concurrent\): UNSAT"

TIERS: dict[str, Tier] = {
    "frozen-dual": Tier(
        "frozen-dual", "frozen-dual-bab", ("cifar100_2024", "tinyimagenet_2024"),
        {"margin_row.frozen_dual_bab": True},
        (r"\[frozen-dual\] armed=1 objectives=[0-9]+ .*\bnodes=[1-9][0-9]*",),
        (r"\[frozen-dual\] armed=1 .*\bverdict=all_closed\b", MR_UNSAT),
        ("lever:NY_MARGIN_ROW_FROZEN_DUAL_BAB=true@config",),
        ("[frozen-dual] armed=1 objectives=", "[frozen-dual] class=", "margin-row BaB (concurrent): UNSAT in "),
        (("t1831", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "_prop_idx_1831_"),
         ("t5529", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "_prop_idx_5529_"),
         ("t1513", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "_prop_idx_1513_"),
         ("c1515", "cifar100_2024", "CIFAR100_resnet_large.onnx", "_prop_idx_1515_"),
         ("c8452", "cifar100_2024", "CIFAR100_resnet_large.onnx", "_prop_idx_8452_")),
        class_alarm=r"\[frozen-dual\] class={j} .*\bverdict=closed\b",
    ),
    "cgan-child-zono": Tier(
        "cgan-child-zono", "cgan-child-zono", ("cgan2026",),
        {"bab.alpha_crown.cgan_child_zono_refresh.enabled": True,
         "bab.alpha_crown.cgan_child_zono_refresh.max_generators": 16384},
        (r"\[cgan-child-zono\] armed=1 max_generators=[0-9]+ .*source=config key=bab\.alpha_crown\."
         r"cgan_child_zono_refresh",
         r"\[cgan-child-zono\] armed=1 children=[1-9][0-9]* installed=[1-9]"),
        (),
        (),
        ("[cgan-child-zono] armed=1 children=", "[cgan-child-zono] armed=1 max_generators="),
        (("r4", "cgan2026", "cGAN_imgSz32_nCh_3.onnx", "cGAN_imgSz32_nCh_3_prop_0_"),
         ("r9", "cgan2026", "cGAN_imgSz64_nCh_1.onnx", "cGAN_imgSz64_nCh_1_prop_1_")),
    ),
    "fwd-root-screen": Tier(
        "fwd-root-screen", "fwd-root-screen", ("ml4acopf_2024",),
        {"bab.forward_root_screen": True, "bab.forward_root_screen_max_mb": 8192},
        (r"source=config key=bab\.forward_root_screen\b", r"\[fwd-root\] ops=\S+ admitted=1",
         r"\[fwd-root\] union screen_refuted=[0-9]+"),
        (r"\[fwd-root\] union screen_refuted=[0-9]+ precheck_refuted=\S+ union_refuted=([0-9]+) "
         r"new_by_screen=[0-9]+ total=\1\b",),
        (),
        ("[fwd-root] union screen_refuted=", " source=config key=bab.forward_root_screen"),
        (("14lres_p1", "ml4acopf_2024", "/14_ieee_ml4acopf-linear-residual.onnx", "/14_ieee_prop1.vnnlib"),
         ("14lnon_p1", "ml4acopf_2024", "/14_ieee_ml4acopf-linear-nonresidual.onnx", "/14_ieee_prop1.vnnlib"),
         ("118lnon_p5", "ml4acopf_2024", "/118_ieee_ml4acopf-linear-nonresidual.onnx", "/118_ieee_prop5.vnnlib"),
         ("300base_p2", "ml4acopf_2024", "/300_ieee_ml4acopf.onnx", "/300_ieee_prop2.vnnlib"),
         ("300lres_p2", "ml4acopf_2024", "/300_ieee_ml4acopf-linear-residual.onnx", "/300_ieee_prop2.vnnlib")),
    ),
    "bi-window-joint-refresh": Tier(
        "bi-window-joint-refresh", "bi-window-joint-refresh", ("challenging_certified_training_2026",),
        {"margin_row.bi_window": True, "margin_row.bi_topk": 8192, "margin_row.joint_refresh": 4,
         "margin_row.joint_refresh_secs": 60.0},
        (r"\[bi-window\] done: pairs=[1-9]", r"\[joint-refresh\] done rounds=[1-9]"),
        (MR_UNSAT,),
        ("lever:NY_MARGIN_ROW_BI_WINDOW=true@config", "lever:NY_MARGIN_ROW_JOINT_REFRESH=4@config"),
        ("[bi-window] done: pairs=", "[joint-refresh] done rounds="),
        (("8090", "challenging_certified_training_2026", "/cifar10_eps8_cnn7.onnx", "_idx8090_"),
         ("8397", "challenging_certified_training_2026", "/cifar10_eps8_wide_cnn7.onnx", "_idx8397_")),
    ),
}

BASE_MARKERS = ("Running VNN-COMP instance", "Loading preset:", "Result: ", "[stop] verdict=",
                "[proof-profile] mode=")

# In-run SELF-CONTRADICTIONS, alarmed on EVERY row whatever its truth: a lane certified UNSAT while ONNX Runtime had
# reconfirmed a counterexample for the same instance. #sat-precedence then publishes `unknown` (or the sat), so the
# published verdict hides it -- but one of the two answers is wrong (vnncomp.rs publish_lane_unsat_and_exit and
# refuse_unsat_against_confirmed_counterexample_in). The flight record carries the same fact as a `sat_precedence`
# event whose reason says REFUSED (a DEFERRED one is not a contradiction: the candidate may still be downgraded).
SELF_CONTRADICTION_ALARMS = (
    r"EARLY EXIT REFUSED — this instance holds an ONNX-Runtime-",
    r"REFUSING to publish `unsat` — this instance holds an ONNX-Runtime-reconfirmed counterexample",
)
SELF_CONTRADICTION_FLIGHT = ("sat_precedence", r"\bREFUSED\b")
SELF_CONTRADICTION_MARKERS = ("EARLY EXIT REFUSED", "REFUSING to publish `unsat`")
# pin-strings fragments behind SAT_ROW_ALARMS: an alarm whose line the pin cannot print is a dark alarm
SAT_ROW_MARKERS = ("margin-row BaB (concurrent): UNSAT in ", "[frozen-dual] armed=1 objectives=",
                   "[fwd-root] union screen_refuted=")

# Certified-unsat lines that must never appear on a truly-SAT row (truth=sat, or banked sat), applied to EVERY row
# kind -- MOAT rows included -- because a lane's unsat can lose the publication claim (or be refused) and so never
# reach the published verdict: the verdict check alone would miss it.
SAT_ROW_ALARMS = (
    MR_UNSAT,
    r"\[frozen-dual\] armed=1 .*\bverdict=all_closed\b",
    r"\[fwd-root\] union screen_refuted=[0-9]+ precheck_refuted=\S+ union_refuted=([0-9]+) new_by_screen=[0-9]+ "
    r"total=\1\b",
)


def witness_classes(prop: Property, y: dict[int, float]) -> list[int]:
    """Classification rows: the competing outputs whose clause holds at the witness (the frozen-dual class
    ids that must never be closed). Only clauses made of one relational atom `Y_j >= Y_t` count."""
    yq = {k: Fraction(v) for k, v in y.items()}
    out = []
    for cl in prop.clauses:
        if len(cl) == 1 and cl[0].kind == "relational" and cl[0].holds(yq):
            out.append(cl[0].big)
    return sorted(set(out))


# ---------------------------------------------------------------------------------------------------------
# benchmarks, banks, field truth (aligned BY INDEX, key asserted)
# ---------------------------------------------------------------------------------------------------------


def _rel(p: str) -> str:
    p = p.strip().strip('"')
    return p[2:] if p.startswith("./") else p


def stem(p: str) -> str:
    b = os.path.basename(p.strip())
    for e in (".gz", ".onnx", ".vnnlib"):
        if b.endswith(e):
            b = b[: -len(e)]
    return b


def parse_model_field(field_text: str) -> list[tuple[str, str]] | None:
    """A paired/relational model field `[('f', 'a.onnx'), ('g', 'b.onnx')]` (isomorphic/monotonic acasxu) as
    [(name, path)], or None for a single model path. Parsed as a literal, never evaluated."""
    import ast
    s = field_text.strip()
    if not s.startswith("["):
        return None
    try:
        val = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        raise M0Error(f"unparseable model field {s[:80]!r}") from None
    if not isinstance(val, list) or not val or not all(
            isinstance(p, tuple) and len(p) == 2 and all(isinstance(x, str) for x in p) for p in val):
        raise M0Error(f"model field is not a list of (name, path) pairs: {s[:80]!r}")
    return [(n, _rel(p)) for n, p in val]


def model_identity(field_text: str) -> tuple:
    pairs = parse_model_field(field_text)
    if pairs is None:
        p = _rel(field_text)
        return (os.path.dirname(p), stem(p))
    return tuple((n, os.path.dirname(p), stem(p)) for n, p in pairs)


def model_key(field_text: str) -> str:
    """The scorer's model half of a row key: the model stem, or the member names joined by '_'."""
    pairs = parse_model_field(field_text)
    return stem(_rel(field_text)) if pairs is None else "_".join(n for n, _ in pairs)


def model_argument(base: str, field_text: str, check_files: bool = True) -> str:
    """The `ny vnncomp v1` model argument: an absolute path, or the paired literal with every member made
    absolute (what vnncomp_sweep forwards, commands/vnncomp_reseed.rs absolute_argument)."""
    pairs = parse_model_field(field_text)
    if pairs is None:
        p = os.path.join(base, _rel(field_text))
        return _existing(p) if check_files else p
    parts = []
    for n, p in pairs:
        ap = os.path.realpath(os.path.join(base, p))
        if check_files and not os.path.isfile(ap):
            raise M0Error(f"benchmark file missing: {ap}")
        parts.append("('" + n.replace("\\", "\\\\").replace("'", "\\'") + "', '"
                     + ap.replace("\\", "\\\\").replace("'", "\\'") + "')")
    return "[" + ", ".join(parts) + "]"


def model_files(arg: str) -> list[str]:
    pairs = parse_model_field(arg)
    return [arg] if pairs is None else [p for _, p in pairs]


@dataclass
class InstRow:
    index: int
    onnx_rel: str
    vnnlib_rel: str
    budget: int
    onnx: str
    vnnlib: str

    @property
    def key(self) -> str:
        return f"{model_key(self.onnx_rel)}-{stem(self.vnnlib_rel)}"


def _existing(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isfile(path + ".gz"):
        return path + ".gz"
    raise M0Error(f"benchmark file missing: {path}")


def load_instances(csv_path: str, check_files: bool = True) -> list[InstRow]:
    if not os.path.isfile(csv_path):
        raise M0Error(f"instances.csv missing: {csv_path}")
    base = os.path.dirname(os.path.abspath(csv_path))
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.reader(fh):
            if len(r) < 3 or not r[0].strip():
                continue
            o, v = (r[0].strip() if parse_model_field(r[0]) else _rel(r[0])), _rel(r[1])
            budget = int(float(r[2]))
            if budget <= 0:
                raise M0Error(f"{csv_path}: non-positive budget on row {len(rows)}")
            op = model_argument(base, o, check_files)
            vp = os.path.join(base, v)
            if check_files:
                vp = _existing(vp)
            rows.append(InstRow(len(rows), o, v, budget, op, vp))
    if not rows:
        raise M0Error(f"{csv_path}: no rows")
    return rows


def scored_instances_csv(ny_root: str, year: int, category: str) -> str:
    """2026: the bank manifest's instances_csv (the version the bank was measured on, sha256-checked);
    2025: the flat benchmarks/vnncomp2025/benchmarks/<cat>/instances.csv."""
    if year == 2026:
        man = os.path.join(ny_root, "reports/measured-2026/bank_manifest.json")
        if not os.path.isfile(man):
            raise M0Error(f"bank manifest missing: {man}")
        d = json.load(open(man))
        b = d.get("banks", {}).get(category)
        if not b:
            raise M0Error(f"{category} is not in the 2026 bank manifest")
        p = os.path.join(ny_root, "benchmarks/vnncomp2026", b["instances_csv"])
        if not os.path.isfile(p):
            raise M0Error(f"instances.csv missing: {p}")
        if sha256_file(p) != b["instances_csv_sha256"]:
            # bytes differ (e.g. CRLF line ends): accept only when the ROWS are identical to the manifest's
            with open(p, newline="") as fh:
                cur = [[c.strip() for c in r[:3]] for r in csv.reader(fh) if r and r[0].strip()]
            want = [[str(c).strip() for c in r[:3]] for r in b.get("instances", [])]
            if cur != want:
                raise M0Error(f"{p}: rows differ from the bank manifest (the bank was measured on other rows)")
        return p
    if year == 2025:
        p = os.path.join(ny_root, "benchmarks/vnncomp2025/benchmarks", category, "instances.csv")
        if not os.path.isfile(p):
            raise M0Error(f"instances.csv missing: {p}")
        return p
    raise M0Error(f"unsupported year {year}")


def bank_csv(ny_root: str, year: int, category: str) -> str:
    return os.path.join(ny_root, "reports/measured-2026" if year == 2026 else "reports/measured", f"{category}.csv")


def load_bank(path: str) -> list[tuple[str, str, str]]:
    if not os.path.isfile(path):
        raise M0Error(f"bank missing: {path}")
    out = []
    with open(path, newline="") as fh:
        for r in csv.reader(fh):
            if len(r) < 5 or r[0].strip().lower() in ("cat", "category", "benchmark"):
                continue  # a header line (reports/measured/lsnc_relu.csv carries one)
            verdict = (r[4] if len(r) >= 6 else r[3]).strip().lower()
            o = r[1].strip() if parse_model_field(r[1]) else _rel(r[1])
            out.append((o, _rel(r[2]), verdict))
    return out


def _row_identity(o: str, v: str) -> tuple:
    v = _rel(v)
    return (model_identity(o), os.path.dirname(v), stem(v))


ALIGN_MODES: dict[str, str] = {}


def align_bank(inst: list[InstRow], bank: list[tuple[str, str, str]], label: str) -> list[str]:
    """Bank verdict per instances.csv index. BY INDEX when the (onnx, vnnlib) pair matches at every index;
    otherwise BY KEY, but only when every key is unique on both sides and the key sets are equal (a
    permutation, the scratchpad/refine2/compare_bank.py rule). Anything else is refused. The mode used is
    recorded in ALIGN_MODES[label]."""
    if len(bank) != len(inst):
        raise M0Error(f"{label}: bank has {len(bank)} rows, instances.csv {len(inst)} (cannot align)")
    for r, (_, _, verdict) in zip(inst, bank):
        if verdict not in VERDICT_TOKENS:
            raise M0Error(f"{label}: bank row {r.index} has verdict {verdict!r}")
    ids_i = [_row_identity(r.onnx_rel, r.vnnlib_rel) for r in inst]
    ids_b = [_row_identity(o, v) for o, v, _ in bank]
    if ids_i == ids_b:
        ALIGN_MODES[label] = "by-index"
        return [b[2] for b in bank]
    first = next(k for k, (a, b) in enumerate(zip(ids_i, ids_b)) if a != b)
    if len(set(ids_i)) == len(ids_i) and len(set(ids_b)) == len(ids_b) and set(ids_i) == set(ids_b):
        by_key = {k: b[2] for k, b in zip(ids_b, bank)}
        ALIGN_MODES[label] = "by-key(permutation)"
        return [by_key[k] for k in ids_i]
    o, v, _ = bank[first]
    r = inst[first]
    raise M0Error(f"{label}: bank row {first} is {o},{v} but instances.csv says {r.onnx_rel},{r.vnnlib_rel}, and "
                  "the rows are not a unique-key permutation of each other (refusing to guess an alignment)")


@dataclass
class RowTruth:
    index: int
    key: str
    true_result: str | None
    fals_ok: list[str] = field(default_factory=list)
    ver_ok: list[str] = field(default_factory=list)

    def token(self) -> str:
        """sat when a counterexample scored (a checker-validated CE exists); unsat when a verification scored
        and no counterexample did; '-' otherwise (no field evidence)."""
        if self.fals_ok:
            return "sat"
        if self.ver_ok:
            return "unsat"
        return "-"


ROW_RE = re.compile(r"^Row: \['([^']+)'")
TRUE_RE = re.compile(r"^True Result: (\S+)")
SCORE_RE = re.compile(r"^(\d+): (\S+) score: (-?\d+), is_ver: (\w+), is_fals: (\w+)")


def parse_results_txt(path: str, year: int) -> dict[str, dict[int, RowTruth]]:
    if not os.path.isfile(path):
        raise M0Error(f"field results missing: {path}")
    cats: dict[str, dict[int, RowTruth]] = {}
    cur = None
    row: RowTruth | None = None
    pending_key = None
    for line in open(path, errors="replace"):
        s = line.strip()
        m = re.match(rf"^Category {year}_(\S+):$", s)
        if m:
            cur = m.group(1)
            cats.setdefault(cur, {})
            row, pending_key = None, None
            continue
        if cur is None:
            continue
        m = ROW_RE.match(s)
        if m:
            pending_key, row = m.group(1), None
            continue
        m = TRUE_RE.match(s)
        if m and pending_key is not None:
            row = RowTruth(-1, pending_key, m.group(1))
            continue
        m = SCORE_RE.match(s)
        if m and row is not None:
            idx = int(m.group(1))
            if row.index == -1:
                row.index = idx
                if idx in cats[cur] and cats[cur][idx].key != row.key:
                    raise M0Error(f"{path}: {year}_{cur} index {idx} carries two keys")
                cats[cur][idx] = row
            elif row.index != idx:
                raise M0Error(f"{path}: {year}_{cur} row {row.key} has score lines for two indices")
            score, is_ver, is_fals = int(m.group(3)), m.group(4) == "True", m.group(5) == "True"
            if score > 0 and is_fals:
                row.fals_ok.append(m.group(2))
            if score > 0 and is_ver:
                row.ver_ok.append(m.group(2))
    return cats


def align_truth(inst: list[InstRow], truth: dict[int, RowTruth] | None, label: str) -> list[RowTruth | None]:
    """Field truth per instances.csv index; the scorer's row key must equal the instances.csv key."""
    if truth is None:
        return [None] * len(inst)
    out = []
    for r in inst:
        t = truth.get(r.index)
        if t is not None and t.key != r.key:
            raise M0Error(f"{label}: field row {r.index} is {t.key} but instances.csv says {r.key}")
        out.append(t)
    return out


def find_row(inst: list[InstRow], onnx_sub: str, vnnlib_sub: str, label: str) -> InstRow:
    hits = [r for r in inst if onnx_sub in "/" + r.onnx_rel and vnnlib_sub in "/" + r.vnnlib_rel]
    if len(hits) != 1:
        raise M0Error(f"{label}: selector ({onnx_sub}, {vnnlib_sub}) matches {len(hits)} rows (need exactly 1)")
    return hits[0]


# ---------------------------------------------------------------------------------------------------------
# the runner spec
# ---------------------------------------------------------------------------------------------------------

SPEC_COLUMNS = ("tag", "kind", "tier", "year", "category", "onnx", "vnnlib", "budget", "configs_dir", "preset",
                "truth", "bank", "index", "reps_c", "reps_p", "need", "alarm", "flight", "group", "note")
KINDS = ("NEGCTL", "KNIFE", "AB", "MOAT")
NEED_SEP, ALARM_SEP, FLIGHT_SEP = " && ", " || ", " && "


def spec_row(**kw) -> dict:
    row = {c: "-" for c in SPEC_COLUMNS}
    for k, v in kw.items():
        if k not in SPEC_COLUMNS:
            raise M0Error(f"unknown spec column {k}")
        row[k] = str(v) if v is not None and v != "" else "-"
    return row


def write_spec(path: str, rows: list[dict], append: bool = False) -> None:
    errs = validate_spec(rows, check_files=False)
    if errs:
        raise M0Error("spec rows invalid:\n  " + "\n  ".join(errs))
    exists = os.path.isfile(path)
    if exists and not append:
        raise M0Error(f"refusing to overwrite {path}")
    if exists:
        prev = read_spec(path)
        clash = {r["tag"] for r in prev} & {r["tag"] for r in rows}
        if clash:
            raise M0Error(f"duplicate tags vs {path}: {sorted(clash)[:5]}")
    with open(path, "a" if exists else "w") as fh:
        if not exists:
            fh.write("#" + SCHEMA + "\t" + "\t".join(SPEC_COLUMNS) + "\n")
        for r in rows:
            for c in SPEC_COLUMNS:
                if "\t" in r[c] or "\n" in r[c]:
                    raise M0Error(f"{r['tag']}: column {c} contains a tab/newline")
            fh.write("\t".join(r[c] for c in SPEC_COLUMNS) + "\n")


def read_spec(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise M0Error(f"spec missing: {path}")
    rows = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n")
        if header != "#" + SCHEMA + "\t" + "\t".join(SPEC_COLUMNS):
            raise M0Error(f"{path}: not an {SCHEMA} spec (header mismatch)")
        for n, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != len(SPEC_COLUMNS):
                raise M0Error(f"{path}:{n}: {len(parts)} columns, want {len(SPEC_COLUMNS)}")
            rows.append(dict(zip(SPEC_COLUMNS, parts)))
    return rows


def split_terms(value: str, sep: str) -> list[str]:
    return [] if value == "-" else value.split(sep)


def validate_spec(rows: list[dict], check_files: bool = True) -> list[str]:
    errs: list[str] = []
    seen: set[str] = set()
    for r in rows:
        t = r.get("tag", "?")
        if not re.match(r"^[A-Za-z0-9_.+-]+$", t):
            errs.append(f"{t}: tag must be [A-Za-z0-9_.+-]+")
        if t in seen:
            errs.append(f"{t}: duplicate tag")
        seen.add(t)
        if r["kind"] not in KINDS:
            errs.append(f"{t}: kind {r['kind']!r}")
        if r["tier"] != "-" and r["tier"] not in TIERS:
            errs.append(f"{t}: unknown tier {r['tier']!r}")
        if not r["budget"].isdigit() or int(r["budget"]) <= 0:
            errs.append(f"{t}: budget {r['budget']!r}")
        for c in ("reps_c", "reps_p"):
            if not r[c].isdigit():
                errs.append(f"{t}: {c} {r[c]!r}")
        if r["reps_c"].isdigit() and r["reps_p"].isdigit() and int(r["reps_c"]) + int(r["reps_p"]) == 0:
            errs.append(f"{t}: zero runs")
        if r["truth"] not in ("sat", "unsat", "-"):
            errs.append(f"{t}: truth {r['truth']!r}")
        if r["bank"] not in VERDICT_TOKENS + ("-",):
            errs.append(f"{t}: bank {r['bank']!r}")
        if r["kind"] in ("NEGCTL", "KNIFE") and r["truth"] != "sat":
            errs.append(f"{t}: a {r['kind']} row must be a truly-SAT row (truth=sat)")
        for col, sep in (("need", NEED_SEP), ("alarm", ALARM_SEP)):
            for term in split_terms(r[col], sep):
                try:
                    re.compile(term[1:] if col == "need" and term.startswith("!") else term)
                except re.error as e:
                    errs.append(f"{t}: {col} regex {term!r}: {e}")
        for term in split_terms(r["flight"], FLIGHT_SEP):
            if not parse_flight_check(term):
                errs.append(f"{t}: flight check {term!r}")
        try:
            files = model_files(r["onnx"]) + [r["vnnlib"]]
        except M0Error as e:
            errs.append(f"{t}: {e}")
            files = [r["vnnlib"]]
        for p in files + [r["configs_dir"]]:
            if not os.path.isabs(p):
                errs.append(f"{t}: path must be absolute: {p}")
        if check_files:
            for p in files:
                if not os.path.isfile(p):
                    errs.append(f"{t}: file missing: {p}")
            if not os.path.isdir(r["configs_dir"]):
                errs.append(f"{t}: configs_dir missing: {r['configs_dir']}")
            elif r["preset"] != "-" and not os.path.isfile(os.path.join(r["configs_dir"], r["preset"])):
                errs.append(f"{t}: preset missing: {r['configs_dir']}/{r['preset']}")
    return errs


def parse_flight_check(term: str):
    m = re.match(r"^lever:([A-Z0-9_]+)=([^@]+)@([a-z_*]+)$", term)
    if m:
        try:
            value = json.loads(m.group(2))
        except ValueError:
            return None
        return ("lever", m.group(1), value, m.group(3))
    m = re.match(r"^(event|noevent):([a-z0-9_]+):(.+)$", term)
    if m:
        try:
            re.compile(m.group(3))
        except re.error:
            return None
        return (m.group(1), m.group(2), m.group(3))
    return None


# ---------------------------------------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------------------------------------

RESULT_RE = re.compile(r"^Result: (\S+)", re.M)
STOP_RE = re.compile(r"\[stop\] verdict=\S+ wall_s=([0-9.]+)")
# `timeout` exit status when it killed the command: the runner wraps every run in the competition's OS backstop
# (vnncomp_scripts/run_instance.sh: `timeout $((budget + 10))`), so 124 means the run overran its scored budget and
# what competition scores is whatever the results file held at the kill.
BACKSTOP_RC = 124


def res_token(res_text: str | None) -> str | None:
    """First token of a results file (`sat` + witness, or a bare verdict); None when no file was supplied, ''
    when the file is missing or empty."""
    if res_text is None:
        return None
    for ln in res_text.splitlines():
        if ln.strip():
            return ln.split()[0].strip().lower()
    return ""


@dataclass
class Judgement:
    tag: str
    profile: str
    rep: int
    verdict: str
    status: str  # OK | VOID | EMERGENCY
    outcome: str  # held | lost | converted | flip | n/a
    wall: str
    reasons: list[str]

    def line(self) -> str:
        reasons = [re.sub(r"[\t\n\r;]", " ", r) for r in self.reasons]
        return "\t".join([self.tag, self.profile, str(self.rep), self.verdict, self.status, self.outcome,
                          self.wall, ";".join(reasons) or "-"])


JUDGED_COLUMNS = ("tag", "profile", "rep", "verdict", "status", "outcome", "wall", "reasons")


def same_json(a, b) -> bool:
    """JSON value equality that does not let Python's True == 1 / False == 0 pass (a bool lever must read a
    bool, a count must read a number)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def _flight_checks(row: dict, flight: dict | None) -> list[str]:
    bad: list[str] = []
    terms = split_terms(row["flight"], FLIGHT_SEP)
    if not terms:
        return bad
    if flight is None:
        return ["flight-record-missing"]
    levers = {lv.get("name"): lv for lv in ((flight.get("levers") or {}).get("receipt") or {}).get("levers", [])}
    events = [e for e in flight.get("events") or [] if isinstance(e, dict)]
    for term in terms:
        chk = parse_flight_check(term)
        if chk is None:
            bad.append(f"flight-check-malformed:{term}")
            continue
        if chk[0] == "lever":
            _, name, value, source = chk
            lv = levers.get(name)
            if lv is None:
                bad.append(f"flight-lever-absent:{name}")
            elif not same_json(lv.get("value"), value):
                bad.append(f"flight-lever:{name}={json.dumps(lv.get('value'))}(want {json.dumps(value)})")
            elif source != "*" and lv.get("source") != source:
                bad.append(f"flight-lever-source:{name}@{lv.get('source')}(want {source})")
        else:
            kind, method, ere = chk
            hits = [e for e in events if e.get("method") == method and re.search(ere, str(e.get("reason") or ""))]
            if kind == "event" and not hits:
                bad.append(f"flight-event-absent:{method}~{ere}")
            if kind == "noevent" and hits:
                bad.append(f"flight-event-present:{method}~{ere}")
    return bad


def judge_run(row: dict, profile: str, rep: int, raw: str, flight: dict | None, res_text: str | None = None,
              box_rc: int | None = None) -> Judgement:
    """One run -> OK / VOID / EMERGENCY. Soundness checks (truth contradiction, bank flip, alarm) are applied
    FIRST and regardless of engagement: an unsat on a truly-SAT row is an emergency even when the tier line is
    missing. VOID means the run proves nothing (wrong instance/preset/profile, tier dark, attacks not
    confirmed off, no Result line) -- never a pass.

    `res_text` is the run's RESULTS FILE after the process exited (what competition scores; '' when missing).
    When supplied it is the verdict of record, and EVERY decided verdict the run stated -- each `Result:` line
    and the results file -- goes through the soundness checks: a decided token in either place is a claim the
    run made. A log/results-file disagreement is VOID. `box_rc` == BACKSTOP_RC (the competition's `timeout
    budget+10` fired) makes the results file the scored verdict even though the log never printed one."""
    body = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("[ny-box"))
    verdicts = RESULT_RE.findall(body)
    log_verdict = verdicts[-1] if verdicts else None
    res_verdict = res_token(res_text)
    backstop = box_rc == BACKSTOP_RC
    if res_text is not None:
        verdict = res_verdict or "none"
    else:
        verdict = log_verdict or "none"
    claimed = sorted({v for v in verdicts if v in DECIDED} | ({res_verdict} & set(DECIDED)))
    m = STOP_RE.findall(body)
    wall = m[-1] if m else ("backstop" if backstop else "-")
    emerg: list[str] = []
    void: list[str] = []
    notes: list[str] = []
    truth, bank = row["truth"], row["bank"]
    if len(claimed) > 1:
        emerg.append(f"SELF-CONTRADICTION(one run stated {' and '.join(claimed)}: log={','.join(verdicts) or '-'} "
                     f"res={res_verdict if res_verdict is not None else '-'})")
    for v in claimed:
        if truth in DECIDED and v != truth:
            emerg.append(f"TRUTH-CONTRADICTION({v} on a {truth} row)")
    outcome = "n/a"
    if bank in DECIDED:
        flipped = [v for v in claimed if v != bank]
        if flipped:
            emerg.append(f"BANK-FLIP(banked {bank}, now {'/'.join(flipped)})")
            outcome = "flip"
        elif verdict == bank:
            outcome = "held"
        else:
            outcome = "lost"
    elif verdict in DECIDED:
        outcome = "converted"
    alarms = list(split_terms(row["alarm"], ALARM_SEP)) + list(SELF_CONTRADICTION_ALARMS)
    if truth == "sat" or bank == "sat":
        alarms += [a for a in SAT_ROW_ALARMS if a not in alarms]
    for ere in alarms:
        hit = re.search(ere, body, re.M)
        if hit:
            emerg.append(f"ALARM[{hit.group(0)[:120]}]")
    method, ere = SELF_CONTRADICTION_FLIGHT
    for e in ((flight or {}).get("events") or []):
        if isinstance(e, dict) and e.get("method") == method and re.search(ere, str(e.get("reason") or "")):
            emerg.append(f"ALARM[flight {method}: {str(e.get('reason'))[:100]}]")
    if len(set(verdicts)) > 1:
        void.append(f"multiple-Result-tokens:{','.join(verdicts)}")
    if res_text is not None:
        if not res_verdict:
            void.append("results-file-missing-or-empty")
        elif log_verdict is None:
            if backstop:
                notes.append(f"note:killed-at-backstop(scored={res_verdict})")
            else:
                void.append(f"no-Result-line(res={res_verdict}; not a backstop kill: box_rc={box_rc})")
        elif log_verdict != res_verdict:
            void.append(f"res-log-mismatch(res={res_verdict},log={log_verdict})")
    if verdict not in VERDICT_TOKENS:
        if verdict != "none":
            void.append(f"unknown-verdict:{verdict}")
        elif log_verdict is None:
            void.append("no-Result-line")
        # else: the log printed one but the results file is empty -- already recorded above
    run_line = re.search(r"^Running VNN-COMP instance:.*$", body, re.M)
    if not run_line or f"vnnlib='{row['vnnlib']}'" not in run_line.group(0):
        void.append("wrong-or-missing-instance-line")
    pre = re.search(r"^Loading preset: (\S+)", body, re.M)
    if row["preset"] == "-":
        if pre:
            void.append(f"preset:{pre.group(1)}(want none: the category has no preset)")
    else:
        want_pre = os.path.normpath(os.path.join(row["configs_dir"], row["preset"]))
        if not pre or os.path.normpath(pre.group(1)) != want_pre:
            void.append(f"preset:{pre.group(1) if pre else 'none'}(want {want_pre})")
    if f"[proof-profile] mode={profile} source=cli" not in body:
        void.append(f"profile-unproven(want [proof-profile] mode={profile} source=cli)")
    for term in split_terms(row["need"], NEED_SEP):
        if term.startswith("!"):
            if re.search(term[1:], body, re.M):
                void.append(f"dark-violation:{term[1:][:80]}")
        elif not re.search(term, body, re.M):
            void.append(f"need-absent:{term[:80]}")
    void.extend(_flight_checks(row, flight))
    # the run's OWN zero-NY_* proof (R8): the flight record snapshots every NY_* plus the thread-count variables
    # the process saw (flight.rs ambient_env_from); anything but the runner's OMP_NUM_THREADS=1 is not what
    # run_instance.sh runs. Absent snapshot (an older pin): nothing to read, step 0 is the check.
    amb = (flight or {}).get("ambient_env")
    if isinstance(amb, dict):
        extra = sorted(k for k, v in amb.items() if (k, v) != ("OMP_NUM_THREADS", "1"))
        if extra:
            void.append(f"ambient-env:{','.join(extra)[:120]}")
    status = "EMERGENCY" if emerg else ("VOID" if void else "OK")
    return Judgement(row["tag"], profile, rep, verdict, status, outcome, wall, emerg + void + notes)


def read_judged(path: str) -> list[Judgement]:
    """EVERY judged line, in file order (a re-judged run appears more than once: see collapse_runs)."""
    out = []
    if not os.path.isfile(path):
        return out
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) != len(JUDGED_COLUMNS) or p[0] == "tag":
            continue
        out.append(Judgement(p[0], p[1], int(p[2]), p[3], p[4], p[5], p[6], [] if p[7] == "-" else p[7].split(";")))
    return out


def collapse_runs(lines: list[Judgement]) -> dict[tuple[str, str, int], Judgement]:
    """One judgement per (tag, profile, rep): the LAST line, except that an EMERGENCY is STICKY -- a later
    re-judgement of the same run (a re-run after its marker was removed) can never make an emergency vanish
    from the record."""
    out: dict[tuple[str, str, int], Judgement] = {}
    for j in lines:
        k = (j.tag, j.profile, j.rep)
        if k in out and out[k].status == "EMERGENCY" and j.status != "EMERGENCY":
            continue
        out[k] = j
    return out


def comparable(j: Judgement) -> bool:
    """A decided verdict that is a claim about THIS row's instance. VOID does not disqualify it (a run with the tier
    dark, the flight unproven or another preset still stated sat/unsat about the same instance, and a sat<->unsat
    pair means one of them is wrong); only a run that cannot be tied to the instance is excluded."""
    return j.verdict in DECIDED and not any(r.startswith("wrong-or-missing-instance-line") for r in j.reasons)


def judge_profiles(row: dict, runs: list[Judgement]) -> list[tuple[str, str]]:
    """Cross-run checks for one row: (status, reason) pairs, status EMERGENCY | NOTE.

    * any sat<->unsat difference between two runs of the row that name its instance (either profile, any rep, VOID
      or not, every judged line including superseded ones) = EMERGENCY VERDICT-FLIP;
    * ENGINE_PROGRAM section 1.7 mode identity: a verdict Proof decides must be decided the same way in >= 2
      of 3 valid Competition reps (ceil(2n/3) of n) = else EMERGENCY PROFILE-DECIDED-SPLIT;
    * Competition deciding what Proof does not is a NOTE (Proof carries extra write-only/audit work, so it may
      run out of time; the rule is one-directional in the program)."""
    mine = [j for j in runs if j.tag == row["tag"]]
    valid = [j for j in collapse_runs(mine).values() if j.status != "VOID"]
    out: list[tuple[str, str]] = []
    claims = [j for j in mine if comparable(j)]
    if len({j.verdict for j in claims}) > 1:
        out.append(("EMERGENCY", "VERDICT-FLIP(" + ",".join(
            f"{j.profile}/r{j.rep}={j.verdict}{'' if j.status != 'VOID' else '[VOID]'}" for j in claims) + ")"))
    comp = [j for j in valid if j.profile == "competition"]
    proof = [j for j in valid if j.profile == "proof"]
    for v in sorted({j.verdict for j in proof if j.verdict in DECIDED}):
        if not comp:
            out.append(("NOTE", f"PROFILE-UNPAIRED(proof decided {v}; no valid competition run)"))
            continue
        need = math.ceil(2 * len(comp) / 3)
        got = sum(1 for j in comp if j.verdict == v)
        if got < need:
            out.append(("EMERGENCY", f"PROFILE-DECIDED-SPLIT(proof {v}; competition {got}/{len(comp)} < {need})"))
    for v in sorted({j.verdict for j in comp if j.verdict in DECIDED}):
        if proof and not any(j.verdict == v for j in proof):
            out.append(("NOTE", f"PROFILE-PROOF-SLOWER(competition {v}; proof undecided)"))
    return out


def instance_key(row: dict) -> tuple[str, str]:
    return (row["onnx"], row["vnnlib"])


def judge_instances(rows: dict[str, dict], runs: list[Judgement], only: set[tuple[str, str]] | None = None
                    ) -> list[tuple[str, str]]:
    """Cross-TAG check: the same (model, property) instance appears under several tags (a NEGCTL row, the A/B
    arms, the moat row). A sat on one and an unsat on another is a soundness EMERGENCY whatever the configs, the
    profile or the VOID status (see `comparable`). Contradictions inside one tag are judge_profiles' VERDICT-FLIP."""
    by_inst: dict[tuple[str, str], list[Judgement]] = {}
    for j in runs:
        r = rows.get(j.tag)
        if r is None or not comparable(j):
            continue
        k = instance_key(r)
        if only is not None and k not in only:
            continue
        by_inst.setdefault(k, []).append(j)
    out: list[tuple[str, str]] = []
    for k, js in by_inst.items():
        if len({j.verdict for j in js}) > 1 and len({j.tag for j in js}) > 1:
            out.append(("EMERGENCY", f"INSTANCE-FLIP({os.path.basename(k[1])}: " + ",".join(
                f"{j.tag}/{j.profile}/r{j.rep}={j.verdict}" for j in js) + ")"))
    return out
