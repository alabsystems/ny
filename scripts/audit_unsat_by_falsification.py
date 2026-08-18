#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Ground-truth-free falsification audit of banked ``unsat`` rows.

WHAT THIS IS
============
Every soundness check the campaign owns today is RELATIVE TO OTHER SOLVERS: the
moat flags an ny ``unsat`` only where the official field produced an ACCEPTED
counterexample on that same row.  That detector is structurally blind wherever
no such counterexample can exist, in two distinct ways:

  row_blind       the field never decided the row at all, so there is nothing
                  to contradict an ny verdict with;
  category_blind  the field accepted ZERO counterexamples anywhere in the
                  category, so the per-row moat cannot fire on ANY row of it --
                  ``vit_2023``, ``nn4sys`` and ``yolo_2023`` are entirely in
                  this state.

``--list-blind-rows`` prints exactly which banked rows are in that shadow, and
refuses to answer at all when the official tree cannot be read, because
"0 blind rows" is the most dangerous possible wrong answer.

This tool asks the NETWORK instead of the field.  A banked ``unsat`` claims the
property holds over the whole input box.  That claim is refutable by a single
point: any ``x`` in the box whose true ``f(x)`` satisfies every assertion in the
VNN-LIB property PROVES the banked ``unsat`` wrong.  So: spend a large OFFLINE
budget hunting for such a point, and validate any hit through the same
zero-tolerance ORT oracle (``scripts/extended_bank/vnnlib_ce.py``) the bank
already trusts for banking ``sat`` rows.

>>> THE PROOF IS ONE-SIDED. <<<
Finding a counterexample PROVES the banked ``unsat`` is wrong.
NOT finding one PROVES NOTHING AT ALL.
A clean run of this tool is *not* a verification, *not* a soundness
certificate, and *not* evidence that the bank is correct.  It is an
unsuccessful search.  Any reader who converts "0 refutations" into "the bank is
sound" has misread the instrument.  Every report header, every per-row status
and the exit message repeat this on purpose.

HOW THE SEARCH WORKS
====================
The VNN-LIB spec is parsed with the trusted parser (``vnnlib_ce.parse_all``) in
a single streaming pass.  Pure-input assertions become a sampling box (with
disjunctive input domains expanded into multiple boxes); output-referencing
assertions are compiled into a vectorised margin/satisfaction evaluator over
float64 numpy batches that mirrors ``vnnlib_ce.evaluate`` semantics exactly
(raw float64 X for assertions, float32 X into ORT, float64 Y out -- the official
zero-tolerance convention).  A top-level ``or`` in the output condition is split
into per-disjunct TARGETS so the search can steer at one disjunct at a time
instead of only at the max.

Because the search is offline it can afford strategies the scored path never
could, and deliberately mixes families so that different failure modes are
covered:

  special / axis      declared box corners, centre, per-coordinate extremes
  corners_full        exhaustive 2^d corner enumeration for low free dimension
  corners_random      random vertex sampling of the box (L-inf robustness)
  grid                dense Cartesian sweep for <=3 free inputs
  coord_sweep         one-coordinate-at-a-time dense sweeps from the incumbent
  halton              low-discrepancy space filling
  uniform             plain uniform random
  spsa                SPSA-estimated-gradient projected ascent, many restarts
  nes                 antithetic NES-estimated-gradient ascent, many restarts
  square              random block sign-flip local search (black-box, gradient
                      free -- catches piecewise-constant / integer-gated nets
                      where every gradient estimate is identically zero)

CALIBRATION
===========
A search tool that never finds anything is indistinguishable from a broken one.
``--self-test`` runs the identical search against banked ``sat`` rows -- rows
where a counterexample is asserted to exist -- and reports the hit rate.  That
number is the only honest way to read a miss.

ROW STATUSES
============
  REFUTED                 a point in the box satisfies every assertion and
                          ``vnnlib_ce.validate`` graded it GENUINE-IN-BOX-CE.
                          The banked ``unsat`` is WRONG.
  REFUTED-REDUCED-ORACLE  same, except the FULL validator refuses this property
                          class outright (nn4sys constrains its inputs only
                          inside mixed X/Y disjunctions, which
                          ``_scan_property`` rejects).  Confirmed instead by
                          ``vnnlib_ce``'s own evaluator on a batch-1 ORT run.
                          Reported separately; never merged into REFUTED.
  CANDIDATE-UNCONFIRMED   the search evaluator said counterexample and no
                          oracle agreed.  NOT a refutation; needs a human.
  NO-CE-FOUND             the search failed.  PROVES NOTHING.
  SKIPPED                 never searched; the reason is attached and counted.

EXIT CODES
==========
  0  no refutation (NOT a soundness result -- see above)
  1  at least one banked unsat refuted (either oracle)
  2  at least one unconfirmed candidate needing human follow-up
  3  environment failure (missing runtime package, unreadable inputs)
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
for _extra in (str(SCRIPT_DIR), str(SCRIPT_DIR / "extended_bank")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import vnnlib_ce  # noqa: E402  (scripts/extended_bank/vnnlib_ce.py -- the trusted oracle)

ONE_SIDED_BANNER = (
    "ONE-SIDED INSTRUMENT: a hit PROVES the banked unsat wrong; a miss PROVES "
    "NOTHING. This is a failed search, never a verification, never a soundness "
    "certificate, and never evidence that the bank is correct."
)

# Row statuses.  Only REFUTED is a proof of anything.
ST_REFUTED = "REFUTED"  # full-oracle validated in-box CE -> the banked unsat is WRONG
ST_REFUTED_REDUCED = "REFUTED-REDUCED-ORACLE"  # see reduced_oracle() -- still a proof,
# but the full validator structurally cannot express this property class
ST_CANDIDATE = "CANDIDATE-UNCONFIRMED"  # our evaluator says CE, oracle would not confirm
ST_NO_CE = "NO-CE-FOUND"  # searched, found nothing -- PROVES NOTHING
ST_SKIP = "SKIPPED"  # not searched at all; reason attached

MAX_TARGETS = 4096
MAX_DOMAIN_BOXES = 256
MAX_COMPLEX_DOMAIN_ASSERTS = 4096
MAX_OUTPUT_ASSERTS = 4096
DEFAULT_BATCH = 256
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)


# --------------------------------------------------------------------------
# Bank enumeration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BankRow:
    category: str
    onnx: str
    vnnlib: str
    verdict: str
    seconds: str
    source: str  # "<csv path>:<line>"

    @property
    def pair(self) -> tuple[str, str]:
        return (self.onnx.lstrip("./"), self.vnnlib.lstrip("./"))

    @property
    def label(self) -> str:
        return f"{self.category}/{self.onnx}::{self.vnnlib}"


class BankFormatError(ValueError):
    """A verdict bank cannot be audited without dropping malformed rows."""


def read_bank(directories: list[Path]) -> list[BankRow]:
    """Every row of every bank CSV, schema-aware.

    Two schemas are in use and they put the verdict in DIFFERENT columns:
      5 cols  cat,onnx,vnnlib,verdict,secs                     (reports/measured-ext)
      6/7     cat,onnx,vnnlib,prepared,verdict,secs[,run_id]   (reports/measured)
    Guessing "the column after the vnnlib path" silently reads ``prepared`` as
    the verdict on the 7-column schema and reports ZERO unsat rows for
    acasxu_2023 -- i.e. it would make this audit vacuous.  Widths are therefore
    dispatched explicitly and anything else is refused loudly.
    """
    rows: list[BankRow] = []
    problems: list[str] = []
    for directory in directories:
        for path in sorted(directory.glob("*.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                for line_number, parts in enumerate(csv.reader(handle), 1):
                    if not parts or not parts[0].strip():
                        continue
                    if parts[0].strip().lower() in {"cat", "category", "onnx"}:
                        continue
                    if len(parts) == 5:
                        verdict_index = 3
                    elif len(parts) in {6, 7}:
                        verdict_index = 4
                    else:
                        problems.append(
                            f"{path}:{line_number}: unsupported {len(parts)}-column row"
                        )
                        continue
                    onnx, vnnlib = parts[1].strip(), parts[2].strip()
                    if any(
                        marker in onnx or marker in vnnlib
                        for marker in ("test_nano", "test_tiny")
                    ):
                        continue  # unscored harness-overhead instances
                    rows.append(
                        BankRow(
                            category=parts[0].strip(),
                            onnx=onnx,
                            vnnlib=vnnlib,
                            verdict=parts[verdict_index].strip().lower(),
                            seconds=parts[5 if len(parts) >= 6 else 4].strip(),
                            source=f"{path}:{line_number}",
                        )
                    )
    if problems:
        details = "\n".join(f"  {problem}" for problem in problems)
        raise BankFormatError(f"bank rows could not be parsed:\n{details}")
    return rows


def dedupe(rows: list[BankRow]) -> list[BankRow]:
    """One entry per (category, onnx, vnnlib, verdict); keep the first source."""
    seen: set[tuple[str, str, str, str]] = set()
    out: list[BankRow] = []
    for row in rows:
        identity = (row.category, *row.pair, row.verdict)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Field ground truth (used ONLY to prioritise, never to decide)
# --------------------------------------------------------------------------


def load_field_ground_truth(
    official_dir: Path,
) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    """(category, onnx_tail2, vnnlib_tail2) -> field verdict summary.

    ``true_result`` is the official ZERO-TOL value; ``accepted_falsifiers`` is
    the set of tools whose counterexample the organizers ACCEPTED.  ``gt``
    collapses to ``violated`` / ``holds`` / ``none``.

    TWO independent blindness flags come out of this, and the second is much
    the larger:

    ``row_blind``       the field never decided this row at all
                        (``true_result == '-'``), so there is nothing to
                        contradict an ny verdict with.
    ``category_blind``  the field produced ZERO accepted counterexamples
                        ANYWHERE in the category.  The moat fires on an ny
                        ``unsat`` only when some tool's counterexample was
                        accepted on that row, so in a category where no
                        counterexample was ever accepted the moat cannot fire
                        on ANY row -- every banked ``unsat`` there is
                        unfalsifiable by the existing detector, whatever its
                        own ``true_result`` says.
    """
    notes: list[str] = []
    try:
        import ny_retroactive_scorecard as scorecard  # noqa: PLC0415
    except Exception as error:  # pragma: no cover - optional annotation only
        notes.append(f"field ground truth unavailable: {error}")
        return {}, notes

    results_txt = official_dir / "SCORING-ZERO-TOL" / "results.txt"
    reference_csv = official_dir / "alpha_beta_crown" / "results.csv"
    for required in (results_txt, reference_csv):
        try:
            present = required.is_file()
        except OSError as error:  # e.g. a broken/looping symlink in the shared tree
            present = False
            notes.append(f"field ground truth path unusable: {required}: {error}")
        if not present:
            notes.append(f"field ground truth unavailable: missing {required}")
            return {}, notes
    notes.append(
        "field ground truth: "
        + ", ".join(
            f"{path.name} sha256={_sha256(path)[:16]} bytes={path.stat().st_size}"
            for path in (results_txt, reference_csv)
        )
    )
    try:
        parsed = scorecard.parse_results_txt(results_txt)
        order = scorecard.load_reference_instance_order(reference_csv)
    except Exception as error:  # pragma: no cover
        notes.append(f"field ground truth unavailable: {error}")
        return {}, notes

    out: dict[tuple[str, str, str], dict] = {}
    for category, payload in parsed.items():
        instances = order.get(category, [])
        for index, row in payload.get("rows", {}).items():
            if index >= len(instances):
                notes.append(
                    f"{category}: results.txt index {index} exceeds reference order "
                    f"({len(instances)} rows); that row is left un-annotated"
                )
                continue
            onnx_tail, vnnlib_tail, _occurrence = instances[index]
            true_result = row.get("true", "-")
            falsifiers = sorted(row.get("fals", set()))
            if true_result == "sat" or falsifiers:
                classification = "violated"
            elif true_result == "unsat":
                classification = "holds"
            else:
                classification = "none"
            key = (category, onnx_tail, vnnlib_tail)
            # Repeated (onnx, vnnlib) pairs exist (sat_relu); keep the first and
            # note the collision rather than silently overwriting.
            out.setdefault(
                key,
                {
                    "true_result": true_result,
                    "accepted_falsifiers": falsifiers,
                    "gt": classification,
                    "row_index": index,
                },
            )

    accepted_per_category: Counter[str] = Counter()
    for (category, _onnx, _vnnlib), info in out.items():
        if info["accepted_falsifiers"]:
            accepted_per_category[category] += 1
    for (category, _onnx, _vnnlib), info in out.items():
        info["category_accepted_ce_count"] = accepted_per_category[category]
        info["category_blind"] = accepted_per_category[category] == 0
        info["row_blind"] = info["gt"] == "none"
        info["field_blind"] = info["category_blind"] or info["row_blind"]
    return out, notes


UNANNOTATED = {
    "true_result": "?",
    "accepted_falsifiers": [],
    "gt": "unannotated",
    "category_accepted_ce_count": 0,
    "category_blind": True,
    "row_blind": True,
    "field_blind": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tail2(path: str) -> str:
    parts = [c for c in path.split("/") if c not in ("", ".")]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else path)


# --------------------------------------------------------------------------
# Spec model: streaming parse -> sampling boxes + vectorised evaluators
# --------------------------------------------------------------------------


class SpecError(RuntimeError):
    """The property could not be turned into a searchable model."""


def _literal(token: Any) -> float | None:
    if isinstance(token, str) and vnnlib_ce.NUMBER_TOKEN.fullmatch(token):
        return float(token)
    return None


def _input_variable(token: Any) -> int | None:
    if not isinstance(token, str):
        return None
    match = vnnlib_ce.VARIABLE.fullmatch(token)
    if match is not None and match.group(1) == "X":
        return int(match.group(2))
    return None


_FLIP = {">=": "<=", "<=": ">=", ">": "<", "<": ">"}


def simple_input_bounds(node: Any) -> list[tuple[int, str, float]] | None:
    """Flatten ``node`` into ``(index, op, constant)`` input bounds, or None.

    Accepts ``(and ...)`` nests of ``(<=|>=|<|>|= X_i c)`` in either operand
    order.  Anything else returns None and the caller keeps the assertion for
    exact evaluation instead of guessing at it.
    """
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        return None
    operator = node[0]
    if operator == "and":
        collected: list[tuple[int, str, float]] = []
        for child in node[1:]:
            child_bounds = simple_input_bounds(child)
            if child_bounds is None:
                return None
            collected.extend(child_bounds)
        return collected
    if operator in {">=", "<=", ">", "<", "="} and len(node) == 3:
        index, constant, effective = _input_variable(node[1]), _literal(node[2]), operator
        if index is None or constant is None:
            index, constant = _input_variable(node[2]), _literal(node[1])
            if index is None or constant is None:
                return None
            effective = _FLIP.get(operator, operator)
        if effective == "=":
            return [(index, ">=", constant), (index, "<=", constant)]
        return [(index, effective, constant)]
    return None


def _input_bounds_within(node: Any) -> list[tuple[int, str, float]]:
    """Input bounds found inside a MIXED conjunction that also mentions Y.

    ``simple_input_bounds`` refuses a node the moment it meets a Y term, which
    is right for deciding "is this a pure input assertion".  But nn4sys states
    its whole input interval only inside ``(and (>= X_0 a) (<= X_0 b)
    (<= Y_0 c))``: skipping those conjuncts leaves an unbounded box and the row
    is dropped as unsearchable.  This pulls out the input conjuncts alone, for
    SAMPLING only -- correctness still comes from evaluating the full assertion.
    """
    if not isinstance(node, list) or not node or node[0] != "and":
        single = simple_input_bounds(node)
        return single or []
    collected: list[tuple[int, str, float]] = []
    for child in node[1:]:
        child_bounds = simple_input_bounds(child)
        if child_bounds is not None:
            collected.extend(child_bounds)
    return collected


def _tighten(box: tuple[Any, Any], bounds: list[tuple[int, str, float]]) -> None:
    import numpy as np  # noqa: PLC0415

    low, high = box
    for index, operator, constant in bounds:
        if index >= low.size:
            raise SpecError(f"bound references undeclared X_{index}")
        if operator in {">=", ">"}:
            low[index] = max(low[index], constant)
        else:
            high[index] = min(high[index], constant)
    del np


@dataclass
class SpecModel:
    input_count: int
    output_count: int
    boxes: list[tuple[Any, Any]]
    domain_asserts: list[Any]  # pure-input assertions kept for exact evaluation
    output_asserts: list[Any]  # assertions that reference at least one Y
    disjuncts: list[Any]  # top-level `or` arms of the single splittable output assert
    disjunct_parent: int  # index into output_asserts that `disjuncts` replaces
    notes: list[str]


def load_spec(path: Path, max_bytes: int) -> SpecModel:
    import numpy as np  # noqa: PLC0415

    size = path.stat().st_size
    if size > max_bytes:
        raise SpecError(
            f"spec is {size / 1e6:.1f} MB, above --max-spec-mb; not searched"
        )

    declared_inputs = 0
    declared_outputs = 0
    domain_asserts: list[Any] = []
    output_asserts: list[Any] = []
    or_domain: list[list[tuple[int, str, float]]] | None = None
    simple: list[tuple[int, str, float]] = []
    notes: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as source:
        for expression in vnnlib_ce.parse_all(source):
            if not isinstance(expression, list) or not expression:
                continue
            head = expression[0]
            if head == "declare-const" and len(expression) == 3:
                match = vnnlib_ce.VARIABLE.fullmatch(str(expression[1]))
                if match is None:
                    continue
                index = int(match.group(2))
                if match.group(1) == "X":
                    declared_inputs = max(declared_inputs, index + 1)
                else:
                    declared_outputs = max(declared_outputs, index + 1)
                continue
            if head != "assert" or len(expression) != 2:
                continue
            body = expression[1]
            if vnnlib_ce._references(body, "Y"):
                if len(output_asserts) >= MAX_OUTPUT_ASSERTS:
                    raise SpecError(
                        f"more than {MAX_OUTPUT_ASSERTS} output assertions; not searched"
                    )
                output_asserts.append(body)
                continue
            bounds = simple_input_bounds(body)
            if bounds is not None:
                simple.extend(bounds)
                continue
            if (
                isinstance(body, list)
                and body
                and body[0] == "or"
                and or_domain is None
                and all(simple_input_bounds(arm) is not None for arm in body[1:])
            ):
                or_domain = [simple_input_bounds(arm) or [] for arm in body[1:]]
                domain_asserts.append(body)  # still checked exactly
                continue
            if len(domain_asserts) >= MAX_COMPLEX_DOMAIN_ASSERTS:
                raise SpecError("too many non-box input assertions; not searched")
            domain_asserts.append(body)
            notes.append("input domain contains a non-box assertion")

    if declared_inputs == 0:
        raise SpecError("property declares no X_i")
    if declared_outputs == 0:
        raise SpecError("property declares no Y_j")
    if not output_asserts:
        raise SpecError("property has no output-referencing assertion")

    base_low = np.full(declared_inputs, -np.inf, dtype=np.float64)
    base_high = np.full(declared_inputs, np.inf, dtype=np.float64)
    _tighten((base_low, base_high), simple)

    boxes: list[tuple[Any, Any]] = []
    if or_domain is None:
        boxes.append((base_low, base_high))
    else:
        for arm in or_domain[:MAX_DOMAIN_BOXES]:
            low, high = base_low.copy(), base_high.copy()
            _tighten((low, high), arm)
            if np.all(low <= high):
                boxes.append((low, high))
        if len(or_domain) > MAX_DOMAIN_BOXES:
            notes.append(
                f"disjunctive input domain truncated to {MAX_DOMAIN_BOXES} of "
                f"{len(or_domain)} arms -- the remainder was NOT searched"
            )
        if not boxes:
            raise SpecError("every disjunctive input box is empty")

    # Splittable output disjunction: the arm structure lets the search steer at
    # one target class at a time instead of only at the max over all of them.
    disjuncts: list[Any] = []
    disjunct_parent = -1
    for position, assertion in enumerate(output_asserts):
        if isinstance(assertion, list) and assertion and assertion[0] == "or":
            arms = list(assertion[1:])
            if len(arms) > MAX_TARGETS:
                # Evenly spread rather than truncating to a prefix: nn4sys
                # lindex specs carry thousands of arms and the first N of them
                # cover only one end of the input range.
                stride = len(arms) / MAX_TARGETS
                arms = [arms[int(i * stride)] for i in range(MAX_TARGETS)]
                notes.append(
                    f"output disjunction sampled to {MAX_TARGETS} of "
                    f"{len(assertion) - 1} arms -- the rest were never targeted "
                    "individually (the whole-condition target still covers them)"
                )
            disjuncts = arms
            disjunct_parent = position
            break

    return SpecModel(
        input_count=declared_inputs,
        output_count=declared_outputs,
        boxes=boxes,
        domain_asserts=domain_asserts,
        output_asserts=output_asserts,
        disjuncts=disjuncts,
        disjunct_parent=disjunct_parent,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Vectorised margin / satisfaction evaluator
#
# Mirrors vnnlib_ce.evaluate exactly (float64 throughout, RAW X values, ORT Y),
# but over a whole batch at once.  ``holds`` is the exact boolean the oracle
# computes; ``margin`` is a smooth surrogate used only to steer the search
# (>= 0 means "satisfied", modulo strict comparisons).
# --------------------------------------------------------------------------

Evaluator = Callable[[Any, Any], Any]


def compile_numeric(node: Any) -> Evaluator:
    import numpy as np  # noqa: PLC0415

    if isinstance(node, str):
        match = vnnlib_ce.VARIABLE.fullmatch(node)
        if match is not None:
            index = int(match.group(2))
            if match.group(1) == "X":
                return lambda x, y, i=index: x[:, i]
            return lambda x, y, i=index: y[:, i]
        value = _literal(node)
        if value is None:
            raise SpecError(f"unknown VNN-LIB atom {node!r}")
        return lambda x, y, v=float(value): np.full(x.shape[0], v, dtype=np.float64)
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise SpecError("malformed numeric expression")
    operator = node[0]
    parts = [compile_numeric(argument) for argument in node[1:]]
    if operator == "+":
        return lambda x, y, p=parts: sum(f(x, y) for f in p)
    if operator == "-":
        if len(parts) == 1:
            return lambda x, y, p=parts: -p[0](x, y)

        def subtract(x, y, p=parts):
            total = p[0](x, y)
            for f in p[1:]:
                total = total - f(x, y)
            return total

        return subtract
    if operator == "*":

        def multiply(x, y, p=parts):
            total = p[0](x, y)
            for f in p[1:]:
                total = total * f(x, y)
            return total

        return multiply
    raise SpecError(f"unsupported numeric operator {operator!r}")


def compile_boolean(node: Any) -> Callable[[Any, Any], tuple[Any, Any]]:
    """Return ``fn(x, y) -> (margin, holds)`` for a boolean VNN-LIB node."""
    import numpy as np  # noqa: PLC0415

    if isinstance(node, str):
        if node in {"true", "false"}:
            truth = node == "true"
            return lambda x, y, t=truth: (
                np.full(x.shape[0], 1.0 if t else -1.0),
                np.full(x.shape[0], t, dtype=bool),
            )
        raise SpecError(f"non-Boolean atom {node!r} used as an assertion")
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise SpecError("malformed boolean expression")
    operator = node[0]

    if operator in {">=", "<=", ">", "<"} and len(node) == 3:
        left, right = compile_numeric(node[1]), compile_numeric(node[2])
        if operator in {">=", ">"}:
            strict = operator == ">"

            def greater(x, y, a=left, b=right, s=strict):
                margin = a(x, y) - b(x, y)
                return margin, (margin > 0) if s else (margin >= 0)

            return greater
        strict = operator == "<"

        def lesser(x, y, a=left, b=right, s=strict):
            margin = b(x, y) - a(x, y)
            return margin, (margin > 0) if s else (margin >= 0)

        return lesser

    if operator == "=" and len(node) >= 3:
        parts = [compile_numeric(argument) for argument in node[1:]]

        def equals(x, y, p=parts):
            first = p[0](x, y)
            margin = np.zeros(x.shape[0], dtype=np.float64)
            holds = np.ones(x.shape[0], dtype=bool)
            for f in p[1:]:
                difference = np.abs(f(x, y) - first)
                margin = np.minimum(margin, -difference)
                holds &= difference == 0
            return margin, holds

        return equals

    if operator in {"and", "or"} and len(node) >= 2:
        parts = [compile_boolean(argument) for argument in node[1:]]
        conjunctive = operator == "and"

        def combine(x, y, p=parts, c=conjunctive):
            margin, holds = p[0](x, y)
            for f in p[1:]:
                other_margin, other_holds = f(x, y)
                if c:
                    margin = np.minimum(margin, other_margin)
                    holds = holds & other_holds
                else:
                    margin = np.maximum(margin, other_margin)
                    holds = holds | other_holds
            return margin, holds

        return combine

    if operator == "not" and len(node) == 2:
        inner = compile_boolean(node[1])

        def negate(x, y, f=inner):
            margin, holds = f(x, y)
            return -margin, ~holds

        return negate

    raise SpecError(f"unsupported boolean operator {operator!r}")


def compile_conjunction(nodes: list[Any]) -> Callable[[Any, Any], tuple[Any, Any]]:
    import numpy as np  # noqa: PLC0415

    if not nodes:
        return lambda x, y: (
            np.zeros(x.shape[0], dtype=np.float64),
            np.ones(x.shape[0], dtype=bool),
        )
    parts = [compile_boolean(node) for node in nodes]

    def conjunction(x, y, p=parts):
        margin, holds = p[0](x, y)
        for f in p[1:]:
            other_margin, other_holds = f(x, y)
            margin = np.minimum(margin, other_margin)
            holds = holds & other_holds
        return margin, holds

    return conjunction


# --------------------------------------------------------------------------
# ONNX execution
# --------------------------------------------------------------------------


class ModelError(RuntimeError):
    """The ONNX model cannot be executed the way the oracle executes it."""


class OnnxRunner:
    """Batched ORT execution that degrades to batch-1 rather than lying.

    Batching is a THROUGHPUT device only.  Every candidate this class helps find
    is re-executed by ``vnnlib_ce.validate`` at batch 1 on the untouched model
    before it is called a refutation, so a batching discrepancy can only cost us
    a find -- it can never manufacture one.  The probe below still checks
    batch-vs-single agreement and records the observed deviation so the effort
    numbers stay interpretable.
    """

    def __init__(self, path: Path, input_count: int, threads: int = 0) -> None:
        import numpy as np  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415

        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        options.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise ModelError(f"ORT cannot load the model: {error}") from error
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ModelError(f"expected one ONNX input tensor, found {len(inputs)}")
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise ModelError(f"expected one ONNX output tensor, found {len(outputs)}")
        self.name = inputs[0].name
        raw_shape = list(inputs[0].shape)
        self.shape = [d if isinstance(d, int) else 1 for d in raw_shape]
        if math.prod(self.shape) != input_count:
            raise ModelError(
                f"ONNX input shape {raw_shape} does not match {input_count} "
                "declared inputs"
            )
        self.symbolic_batch = bool(raw_shape) and not isinstance(raw_shape[0], int)
        self.batch_limit = 1
        self.session_runs = 0  # real ORT invocations, not candidate batches
        self.batch_probe: dict[str, Any] = {"attempted": False}
        # A leading axis can only carry a batch when the model actually has one:
        # a symbolic first dim, or a literal 1.  A rank-1 input of shape [12296]
        # has NO batch axis -- treating dim 0 as one reshapes 12296 values into
        # a length-1 tensor and the run dies.
        self.batch_suffix: list[int] | None = (
            self.shape[1:]
            if raw_shape and (self.symbolic_batch or self.shape[0] == 1)
            else None
        )
        if self.batch_suffix is not None:
            self._probe_batch(np)

    def _probe_batch(self, np) -> None:
        self.batch_probe["attempted"] = True
        probe = np.zeros((2, math.prod(self.shape)), dtype=np.float32)
        probe[1] = 1e-3
        try:
            batched = self._run_raw(probe)
            single = np.concatenate(
                [self._run_raw(probe[0:1]), self._run_raw(probe[1:2])], axis=0
            )
        except Exception as error:
            self.batch_probe["ok"] = False
            self.batch_probe["reason"] = str(error)[:200]
            self.batch_suffix = None
            return
        if batched.shape != single.shape:
            self.batch_probe["ok"] = False
            self.batch_probe["reason"] = "batched output shape differs"
            self.batch_suffix = None
            return
        deviation = float(np.max(np.abs(batched - single))) if batched.size else 0.0
        self.batch_probe["max_abs_deviation"] = deviation
        if not math.isfinite(deviation) or deviation > 1e-5:
            self.batch_probe["ok"] = False
            self.batch_probe["reason"] = f"batch/single deviation {deviation:g}"
            self.batch_suffix = None
            return
        self.batch_probe["ok"] = True
        self.batch_limit = DEFAULT_BATCH

    def _run_raw(self, flat):
        batch = flat.shape[0]
        if self.batch_suffix is None:
            if batch != 1:
                raise ModelError("model has no batch axis; only batch 1 is runnable")
            tensor = flat.reshape(self.shape)  # exactly the oracle's own reshape
        else:
            tensor = flat.reshape([batch, *self.batch_suffix])
        self.session_runs += 1
        result = self.session.run(None, {self.name: tensor})[0]
        return result.reshape(batch, -1).astype("float64")

    def run(self, flat):
        """flat: (B, n_in) float32 -> (B, n_out) float64."""
        import numpy as np  # noqa: PLC0415

        if flat.shape[0] <= self.batch_limit:
            return self._run_raw(flat)
        chunks = [
            self._run_raw(flat[start : start + self.batch_limit])
            for start in range(0, flat.shape[0], self.batch_limit)
        ]
        return np.concatenate(chunks, axis=0)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@dataclass
class Effort:
    points: int = 0
    forward_calls: int = 0
    wall_seconds: float = 0.0
    per_strategy: dict[str, int] = field(default_factory=dict)
    best_margin: float = float("-inf")
    targets: int = 0
    free_inputs: int = 0
    pinned_inputs: int = 0
    batch_limit: int = 1

    def as_dict(self) -> dict:
        return {
            "points_tried": self.points,
            "ort_forward_calls": self.forward_calls,
            "wall_seconds": round(self.wall_seconds, 3),
            "strategies_run": dict(self.per_strategy),
            "best_full_property_margin": (
                None if self.best_margin == float("-inf") else self.best_margin
            ),
            "targets": self.targets,
            "free_inputs": self.free_inputs,
            "pinned_inputs": self.pinned_inputs,
            "ort_batch_limit": self.batch_limit,
        }


class Hit(Exception):
    def __init__(self, point, output, target: str, strategy: str) -> None:
        super().__init__("candidate counterexample")
        self.point = point
        self.output = output
        self.target = target
        self.strategy = strategy


class Searcher:
    """Hunt one box/target pair for a point satisfying EVERY assertion."""

    def __init__(
        self,
        model: SpecModel,
        runner: OnnxRunner,
        low,
        high,
        steer: Callable[[Any, Any], tuple[Any, Any]],
        gate: Callable[[Any, Any], tuple[Any, Any]],
        effort: Effort,
        rng,
        target_name: str,
    ) -> None:
        import numpy as np  # noqa: PLC0415

        self.np = np
        self.model = model
        self.runner = runner
        self.steer = steer
        self.gate = gate
        self.effort = effort
        self.rng = rng
        self.target_name = target_name

        self.low = low
        self.high = high
        free = np.nonzero(high > low)[0]
        self.free = free
        self.free_low = low[free].copy()
        self.free_high = high[free].copy()

        # Every point is generated so that ORT sees exactly the float64 values
        # the assertions were checked against: pinned coordinates keep their
        # exact (possibly non-float32-representable) constant -- which is what
        # makes cctsdb-style equality-pinned specs falsifiable at all -- and
        # free coordinates are snapped onto the float32 grid inside the box.
        self.use_f32 = np.zeros(free.size, dtype=bool)
        self.f32_low = np.zeros(free.size, dtype=np.float32)
        self.f32_high = np.zeros(free.size, dtype=np.float32)
        for position in range(free.size):
            low_f32 = _f32_above(self.free_low[position], np)
            high_f32 = _f32_below(self.free_high[position], np)
            if (
                np.isfinite(low_f32)
                and np.isfinite(high_f32)
                and float(low_f32) <= float(high_f32)
            ):
                self.use_f32[position] = True
                self.f32_low[position] = low_f32
                self.f32_high[position] = high_f32

        self.base = np.where(np.isfinite(low), low, 0.0).astype(np.float64)
        finite_both = np.isfinite(low) & np.isfinite(high)
        self.base[finite_both] = 0.5 * (low[finite_both] + high[finite_both])
        self.centre_free = self.snap(self.base[free][None, :])[0]
        self.base[free] = self.centre_free
        pinned = np.nonzero(high == low)[0]
        self.base[pinned] = low[pinned]
        self.best_free = self.centre_free.copy()
        self.best_margin = float("-inf")

    # -- point construction -------------------------------------------------

    def snap(self, values):
        np = self.np
        out = np.array(values, dtype=np.float64, copy=True)
        mask = self.use_f32
        if mask.any():
            narrowed = out[:, mask].astype(np.float32)
            narrowed = np.clip(narrowed, self.f32_low[mask], self.f32_high[mask])
            out[:, mask] = narrowed.astype(np.float64)
        rest = ~mask
        if rest.any():
            out[:, rest] = np.clip(out[:, rest], self.free_low[rest], self.free_high[rest])
        return out

    def materialise(self, free_values):
        np = self.np
        points = np.repeat(self.base[None, :], free_values.shape[0], axis=0)
        points[:, self.free] = self.snap(free_values)
        return points

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, free_values, strategy: str):
        """Run a batch; raise Hit on the first point satisfying everything."""
        np = self.np
        if free_values.shape[0] == 0:
            return
        points = self.materialise(free_values)
        outputs = self.runner.run(points.astype(np.float32))
        self.effort.points += points.shape[0]
        # A candidate batch is NOT a forward pass: when ORT refuses to batch the
        # model, one batch of 256 costs 256 separate session.run calls.  Counting
        # batches there would understate the work by the batch factor and make
        # slow rows look identical to fast ones.
        self.effort.forward_calls = self.runner.session_runs
        self.effort.per_strategy[strategy] = (
            self.effort.per_strategy.get(strategy, 0) + points.shape[0]
        )
        _steer_margin, _ = self.steer(points, outputs)
        gate_margin, gate_holds = self.gate(points, outputs)
        best = int(np.argmax(_steer_margin))
        if float(_steer_margin[best]) > self.best_margin:
            self.best_margin = float(_steer_margin[best])
            self.best_free = points[best][self.free].copy()
        self.effort.best_margin = max(self.effort.best_margin, float(gate_margin.max()))
        if gate_holds.any():
            index = int(np.argmax(gate_holds))
            raise Hit(points[index], outputs[index], self.target_name, strategy)
        return _steer_margin

    # -- strategies ---------------------------------------------------------

    def run(self, deadline: float, plan: list[tuple[str, float]], batch: int) -> None:
        for name, share in plan:
            if time.monotonic() >= deadline:
                return
            remaining = deadline - time.monotonic()
            stop = time.monotonic() + max(0.05, remaining * share)
            handler = getattr(self, f"_strategy_{name}", None)
            if handler is None:
                continue
            handler(min(stop, deadline), batch)

    def _strategy_special(self, deadline: float, batch: int) -> None:
        np = self.np
        n = self.free.size
        if n == 0:
            self.evaluate(np.zeros((1, 0)), "special")
            return
        patterns = [
            self.free_low,
            self.free_high,
            self.centre_free,
            np.where(np.arange(n) % 2 == 0, self.free_low, self.free_high),
            np.where(np.arange(n) % 2 == 0, self.free_high, self.free_low),
            np.where(np.arange(n) % 3 == 0, self.free_high, self.free_low),
            0.5 * (self.free_low + self.centre_free),
            0.5 * (self.free_high + self.centre_free),
        ]
        self.evaluate(np.array(patterns, dtype=np.float64), "special")

    def _strategy_axis(self, deadline: float, batch: int) -> None:
        """Each free coordinate driven to each bound, others at the centre."""
        np = self.np
        n = self.free.size
        if n == 0:
            return
        order = self.rng.permutation(n)
        cursor = 0
        while cursor < n and time.monotonic() < deadline:
            block = order[cursor : cursor + max(1, batch // 2)]
            cursor += block.size
            candidates = np.repeat(self.centre_free[None, :], 2 * block.size, axis=0)
            for position, coordinate in enumerate(block):
                candidates[2 * position, coordinate] = self.free_low[coordinate]
                candidates[2 * position + 1, coordinate] = self.free_high[coordinate]
            self.evaluate(candidates, "axis")

    def _strategy_corners_full(self, deadline: float, batch: int) -> None:
        np = self.np
        n = self.free.size
        if n == 0 or n > 20:
            return
        total = 1 << n
        emitted = 0
        while emitted < total and time.monotonic() < deadline:
            size = min(batch, total - emitted)
            indices = np.arange(emitted, emitted + size, dtype=np.int64)
            bits = ((indices[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
            self.evaluate(
                np.where(bits, self.free_high[None, :], self.free_low[None, :]),
                "corners_full",
            )
            emitted += size

    def _strategy_corners_random(self, deadline: float, batch: int) -> None:
        np = self.np
        n = self.free.size
        if n == 0:
            return
        while time.monotonic() < deadline:
            bits = self.rng.random((batch, n)) < 0.5
            self.evaluate(
                np.where(bits, self.free_high[None, :], self.free_low[None, :]),
                "corners_random",
            )

    def _strategy_grid(self, deadline: float, batch: int) -> None:
        """Dense Cartesian sweep -- the only thing that finds a violating cell in
        a piecewise-constant or integer-gated network, where every gradient
        estimate is identically zero."""
        np = self.np
        n = self.free.size
        if n == 0 or n > 3:
            return
        per_axis = {1: 20000, 2: 320, 3: 46}[n]
        axes = [
            np.linspace(self.free_low[i], self.free_high[i], per_axis) for i in range(n)
        ]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, n)
        for start in range(0, mesh.shape[0], batch):
            if time.monotonic() >= deadline:
                return
            self.evaluate(mesh[start : start + batch], "grid")

    def _strategy_coord_sweep(self, deadline: float, batch: int) -> None:
        np = self.np
        n = self.free.size
        if n == 0:
            return
        steps = max(8, min(batch, 128))
        for coordinate in self.rng.permutation(n):
            if time.monotonic() >= deadline:
                return
            candidates = np.repeat(self.best_free[None, :], steps, axis=0)
            candidates[:, coordinate] = np.linspace(
                self.free_low[coordinate], self.free_high[coordinate], steps
            )
            self.evaluate(candidates, "coord_sweep")

    def _strategy_halton(self, deadline: float, batch: int) -> None:
        np = self.np
        n = self.free.size
        if n == 0:
            return
        dimensions = min(n, len(PRIMES))
        index = int(self.rng.integers(1, 10_000))
        while time.monotonic() < deadline:
            block = np.empty((batch, n), dtype=np.float64)
            indices = np.arange(index, index + batch)
            index += batch
            for position in range(n):
                base = PRIMES[position % dimensions]
                block[:, position] = _van_der_corput(indices, base, np)
            self.evaluate(
                self.free_low[None, :]
                + block * (self.free_high - self.free_low)[None, :],
                "halton",
            )

    def _strategy_uniform(self, deadline: float, batch: int) -> None:
        np = self.np
        n = self.free.size
        if n == 0:
            return
        span = self.free_high - self.free_low
        while time.monotonic() < deadline:
            self.evaluate(
                self.free_low[None, :] + self.rng.random((batch, n)) * span[None, :],
                "uniform",
            )

    def _strategy_spsa(self, deadline: float, batch: int) -> None:
        """SPSA-estimated-gradient projected ascent with random restarts.

        Stands in for PGD: no autodiff is available through ORT, and a two-sided
        random-direction estimate needs only forward passes.  Restarts are the
        point -- a single ascent run is exactly the local search the scored path
        already does and would add nothing.
        """
        np = self.np
        n = self.free.size
        if n == 0:
            return
        span = self.free_high - self.free_low
        pairs = max(2, min(batch // 2, 32))
        while time.monotonic() < deadline:
            current = (
                self.free_low + self.rng.random(n) * span
                if self.rng.random() < 0.7
                else self.best_free.copy()
            )
            step = 0.25
            for iteration in range(64):
                if time.monotonic() >= deadline:
                    return
                perturbation = np.where(self.rng.random((pairs, n)) < 0.5, -1.0, 1.0)
                probe = 0.02 * span[None, :]
                plus = np.clip(
                    current[None, :] + perturbation * probe,
                    self.free_low,
                    self.free_high,
                )
                minus = np.clip(
                    current[None, :] - perturbation * probe,
                    self.free_low,
                    self.free_high,
                )
                margins = self.evaluate(
                    np.concatenate([plus, minus], axis=0), "spsa"
                )
                if margins is None:
                    return
                difference = (margins[:pairs] - margins[pairs:])[:, None]
                gradient = np.mean(difference * perturbation, axis=0)
                norm = np.abs(gradient).max()
                if norm == 0 or not math.isfinite(norm):
                    break  # flat objective: SPSA is blind here, other lanes cover it
                current = np.clip(
                    current + step * span * np.sign(gradient),
                    self.free_low,
                    self.free_high,
                )
                step = max(0.01, step * 0.94)

    def _strategy_nes(self, deadline: float, batch: int) -> None:
        np = self.np
        n = self.free.size
        if n == 0:
            return
        span = self.free_high - self.free_low
        pairs = max(2, min(batch // 2, 32))
        while time.monotonic() < deadline:
            current = self.best_free.copy()
            sigma = 0.1
            for iteration in range(48):
                if time.monotonic() >= deadline:
                    return
                noise = self.rng.standard_normal((pairs, n))
                plus = np.clip(
                    current[None, :] + sigma * span * noise, self.free_low, self.free_high
                )
                minus = np.clip(
                    current[None, :] - sigma * span * noise, self.free_low, self.free_high
                )
                margins = self.evaluate(np.concatenate([plus, minus], axis=0), "nes")
                if margins is None:
                    return
                weights = margins[:pairs] - margins[pairs:]
                scale = np.abs(weights).max()
                if scale == 0 or not math.isfinite(scale):
                    break
                gradient = (weights / scale)[:, None] * noise
                current = np.clip(
                    current + 0.15 * span * np.mean(gradient, axis=0),
                    self.free_low,
                    self.free_high,
                )
                sigma = max(0.01, sigma * 0.96)

    def _strategy_square(self, deadline: float, batch: int) -> None:
        """Random block sign-flip hill climbing -- purely comparison-driven, so
        it keeps working when the objective is flat and every gradient estimate
        (SPSA, NES) collapses to zero."""
        np = self.np
        n = self.free.size
        if n == 0:
            return
        while time.monotonic() < deadline:
            current = self.best_free.copy()
            incumbent = -np.inf
            fraction = 0.5
            for iteration in range(96):
                if time.monotonic() >= deadline:
                    return
                size = max(1, int(n * fraction))
                candidates = np.repeat(current[None, :], batch, axis=0)
                for row in range(batch):
                    picks = self.rng.choice(n, size=size, replace=False)
                    to_high = self.rng.random(size) < 0.5
                    candidates[row, picks] = np.where(
                        to_high, self.free_high[picks], self.free_low[picks]
                    )
                margins = self.evaluate(candidates, "square")
                if margins is None:
                    return
                best = int(np.argmax(margins))
                if margins[best] > incumbent:
                    incumbent = float(margins[best])
                    current = candidates[best].copy()
                fraction = max(1.0 / max(n, 1), fraction * 0.85)


def _f32_above(value: float, np):
    """Smallest float32 that is >= ``value`` (inf when none exists)."""
    if not math.isfinite(value):
        return np.float32(np.inf) if value > 0 else np.float32(-np.inf)
    candidate = np.float32(value)
    if float(candidate) < value:
        candidate = np.nextafter(candidate, np.float32(np.inf), dtype=np.float32)
    return candidate


def _f32_below(value: float, np):
    if not math.isfinite(value):
        return np.float32(np.inf) if value > 0 else np.float32(-np.inf)
    candidate = np.float32(value)
    if float(candidate) > value:
        candidate = np.nextafter(candidate, np.float32(-np.inf), dtype=np.float32)
    return candidate


def _van_der_corput(indices, base: int, np):
    result = np.zeros(indices.shape, dtype=np.float64)
    denominator = 1.0
    remaining = indices.astype(np.int64).copy()
    while np.any(remaining > 0):
        denominator *= base
        result += (remaining % base) / denominator
        remaining //= base
    return result


DEFAULT_PLAN = [
    ("special", 0.02),
    ("corners_random", 0.04),
    ("corners_full", 0.10),
    ("grid", 0.12),
    ("spsa", 0.28),
    ("square", 0.16),
    ("nes", 0.12),
    ("axis", 0.06),
    ("coord_sweep", 0.06),
    ("halton", 0.05),
    ("uniform", 0.05),
]


# --------------------------------------------------------------------------
# Per-row audit
# --------------------------------------------------------------------------


DEFAULT_EXPRESSION_CAP = vnnlib_ce.MAX_EXPRESSION_TOKENS


def audit_row(
    row: BankRow,
    benchmark_root: Path,
    budget_seconds: float,
    evidence_dir: Path | None,
    max_spec_mb: float,
    seed: int,
    threads: int,
    batch: int,
    max_expression_tokens: int = 0,
) -> dict:
    started = time.monotonic()
    # vnnlib_ce caps a single top-level expression at 100k tokens as a DoS guard
    # against untrusted input.  Whole nn4sys families are ONE assertion far above
    # that, so the guard -- not the semantics -- is what refuses them, and it
    # refuses them in the trusted validator too.  Raising it here is an explicit,
    # recorded opt-in that changes nothing but the bound: the same tokenizer,
    # parser and evaluator run, on benchmark files that are not adversarial.
    if max_expression_tokens > DEFAULT_EXPRESSION_CAP:
        vnnlib_ce.MAX_EXPRESSION_TOKENS = max_expression_tokens
    else:
        vnnlib_ce.MAX_EXPRESSION_TOKENS = DEFAULT_EXPRESSION_CAP
    effort = Effort()
    record: dict[str, Any] = {
        "category": row.category,
        "onnx": row.onnx,
        "vnnlib": row.vnnlib,
        "banked_verdict": row.verdict,
        "banked_seconds": row.seconds,
        "bank_source": row.source,
        "status": ST_SKIP,
        "detail": "",
        "one_sided_note": ONE_SIDED_BANNER,
    }

    directory = benchmark_root / row.category
    onnx_path = directory / row.onnx.lstrip("./")
    vnnlib_path = directory / row.vnnlib.lstrip("./")
    record["onnx_path"] = str(onnx_path)
    record["vnnlib_path"] = str(vnnlib_path)
    if not onnx_path.is_file() or not vnnlib_path.is_file():
        record["detail"] = "instance files not found under --benchmark-root"
        record["effort"] = effort.as_dict()
        return record

    import numpy as np  # noqa: PLC0415

    try:
        model = load_spec(vnnlib_path, int(max_spec_mb * 1e6))
    except (SpecError, vnnlib_ce.ValidationError, OSError, RecursionError) as error:
        record["detail"] = f"spec not searchable: {error}"
        record["effort"] = effort.as_dict()
        return record
    if model.notes:
        record["spec_notes"] = sorted(set(model.notes))
    if vnnlib_ce.MAX_EXPRESSION_TOKENS != DEFAULT_EXPRESSION_CAP:
        record["parser_expression_cap"] = {
            "default": DEFAULT_EXPRESSION_CAP,
            "raised_to": vnnlib_ce.MAX_EXPRESSION_TOKENS,
            "note": (
                "vnnlib_ce's untrusted-input token guard was raised for this "
                "offline audit; parsing and evaluation are otherwise identical. "
                "The SHIPPED validator still refuses specs above the default."
            ),
        }

    can_validate, precheck_reason = oracle_precheck(vnnlib_path)
    record["full_oracle_can_validate_this_spec"] = can_validate
    if not can_validate:
        record["full_oracle_refusal"] = precheck_reason

    try:
        runner = OnnxRunner(onnx_path, model.input_count, threads=threads)
    except ModelError as error:
        record["detail"] = f"model not executable: {error}"
        record["effort"] = effort.as_dict()
        return record
    record["batch_probe"] = runner.batch_probe
    effort.batch_limit = runner.batch_limit
    # Keep one candidate batch under ~64 MB of float64 regardless of how many
    # inputs the spec declares; a 750k-input spec at batch 256 would otherwise
    # try to allocate 1.5 GB per strategy step.
    batch = max(1, min(batch, 8_000_000 // max(1, model.input_count)))

    gate = compile_conjunction(model.domain_asserts + model.output_asserts)

    # Targets: the whole output condition, plus one per top-level `or` arm so
    # the search can chase a single disjunct instead of only their maximum.
    targets: list[tuple[str, Any, Any, Any]] = []
    unusable_boxes = 0

    def usable(low, high) -> bool:
        return bool(
            np.all(np.isfinite(low)) and np.all(np.isfinite(high)) and np.all(low <= high)
        )

    for box_index, (low, high) in enumerate(model.boxes):
        suffix = f"box{box_index}" if len(model.boxes) > 1 else ""
        if usable(low, high):
            targets.append(
                (f"all{suffix}", low, high, compile_conjunction(model.output_asserts))
            )
        else:
            unusable_boxes += 1
        others = [
            assertion
            for position, assertion in enumerate(model.output_asserts)
            if position != model.disjunct_parent
        ]
        for arm_index, arm in enumerate(model.disjuncts):
            # An arm may carry the ONLY input bounds in the whole file -- nn4sys
            # states its input interval exclusively inside mixed (X, Y)
            # disjuncts.  Tightening happens BEFORE the usability test so those
            # specs are searched instead of skipped as "unbounded".
            arm_low, arm_high = low.copy(), high.copy()
            arm_bounds = simple_input_bounds(arm) or _input_bounds_within(arm)
            if arm_bounds:
                try:
                    _tighten((arm_low, arm_high), arm_bounds)
                except SpecError:
                    continue
            if not usable(arm_low, arm_high):
                continue
            targets.append(
                (
                    f"{suffix}arm{arm_index}" if suffix else f"arm{arm_index}",
                    arm_low,
                    arm_high,
                    compile_conjunction([arm, *others]),
                )
            )
    if not targets:
        record["detail"] = (
            "no finite non-empty input box could be derived (unbounded or "
            "contradictory input domain); not searched"
        )
        record["effort"] = effort.as_dict()
        return record
    if len(targets) > MAX_TARGETS:
        stride = len(targets) / MAX_TARGETS
        record.setdefault("spec_notes", []).append(
            f"targets sampled to {MAX_TARGETS} of {len(targets)}"
        )
        targets = [targets[int(i * stride)] for i in range(MAX_TARGETS)]
    if unusable_boxes:
        record.setdefault("spec_notes", []).append(
            f"{unusable_boxes} declared input box(es) were unbounded or empty "
            "and could not be sampled directly"
        )
    effort.targets = len(targets)

    rng = np.random.default_rng(
        abs(hash((row.category, row.onnx, row.vnnlib, seed))) % (2**63)
    )
    deadline = started + budget_seconds
    # The whole-condition target gets the first 40% of the budget; the remaining
    # 60% is shared round-robin by the per-disjunct targets.
    hit: Hit | None = None
    try:
        primary = targets[0]
        searcher = Searcher(
            model, runner, primary[1], primary[2], primary[3], gate, effort, rng,
            primary[0],
        )
        effort.free_inputs = int(searcher.free.size)
        effort.pinned_inputs = int(model.input_count - searcher.free.size)
        searcher.run(min(deadline, started + budget_seconds * 0.4), DEFAULT_PLAN, batch)

        rest = targets[1:]
        if rest and time.monotonic() < deadline:
            slice_seconds = (deadline - time.monotonic()) / len(rest)
            for name, low, high, steer in rest:
                if time.monotonic() >= deadline:
                    break
                sub = Searcher(
                    model, runner, low, high, steer, gate, effort, rng, name
                )
                sub.run(
                    min(deadline, time.monotonic() + slice_seconds), DEFAULT_PLAN, batch
                )
        # Anything left over goes back into the strongest lanes.
        while time.monotonic() < deadline:
            searcher.run(
                deadline,
                [("spsa", 0.4), ("square", 0.3), ("corners_random", 0.15),
                 ("uniform", 0.15)],
                batch,
            )
    except Hit as found:
        hit = found
    except (SpecError, ModelError) as error:
        effort.wall_seconds = time.monotonic() - started
        record["status"] = ST_SKIP
        record["detail"] = f"search aborted: {error}"
        record["effort"] = effort.as_dict()
        return record

    effort.wall_seconds = time.monotonic() - started
    record["effort"] = effort.as_dict()

    if hit is None:
        record["status"] = ST_NO_CE
        record["detail"] = (
            "no counterexample found within the budget. THIS PROVES NOTHING: "
            "the row is not verified, not confirmed, and not cleared."
        )
        return record

    # -- a hit is an INCIDENT: re-prove it through the trusted oracle ---------
    values = {index: float(hit.point[index]) for index in range(model.input_count)}
    try:
        in_box, is_counterexample, detail = vnnlib_ce.validate(
            onnx_path, vnnlib_path, values
        )
    except Exception as error:  # oracle refused; report, do not claim
        in_box, is_counterexample, detail = False, False, f"oracle error: {error}"

    record["oracle"] = {
        "validator": "scripts/extended_bank/vnnlib_ce.py",
        "mode": "full",
        "versions": vnnlib_ce.runtime_versions(),
        "in_box": bool(in_box),
        "is_counterexample": bool(is_counterexample),
        "verdict": (
            "GENUINE-IN-BOX-CE"
            if is_counterexample
            else ("OUT-OF-BOX" if not in_box else "IN-BOX-BUT-NOT-CE")
        ),
        "detail": detail,
    }

    reduced_holds = False
    if not is_counterexample and not can_validate:
        try:
            reduced_holds, reduced_detail, reduced_output = reduced_oracle(
                onnx_path, vnnlib_path, values
            )
        except Exception as error:
            reduced_holds, reduced_detail, reduced_output = (
                False,
                f"reduced oracle error: {error}",
                [],
            )
        record["reduced_oracle"] = {
            "why": (
                "the full validator cannot express this property class: "
                + precheck_reason
            ),
            "all_assertions_hold": bool(reduced_holds),
            "detail": reduced_detail,
            "ort_output": reduced_output[: model.output_count],
            "caveat": (
                "vnnlib_ce's parser, evaluator and batch-1 ORT execution all "
                "ran; only _scan_property's structural completeness check was "
                "skipped, because it is what refuses the file."
            ),
        }
    record["found_by"] = {"strategy": hit.strategy, "target": hit.target}
    record["ort_output"] = [float(v) for v in hit.output[: model.output_count]]
    record["violated_constraints"] = explain(model, hit.point, hit.output, np)

    counterexample_path = None
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{row.category}__{Path(row.onnx).stem}__{Path(row.vnnlib).stem}"
        counterexample_path = evidence_dir / f"{stem}.counterexample"
        counterexample_path.write_text(
            "(\n"
            + "\n".join(f"(X_{index} {values[index]!r})" for index in sorted(values))
            + "\n)\n",
            encoding="utf-8",
        )
        record["counterexample_file"] = str(counterexample_path)
        record["reproduce"] = (
            f"python3 scripts/extended_bank/vnnlib_ce.py {onnx_path} "
            f"{vnnlib_path} {counterexample_path}"
        )

    agrees_with_bank = row.verdict == "sat"
    if is_counterexample:
        record["status"] = ST_REFUTED
        record["detail"] = (
            "A point inside the declared input box satisfies every assertion of "
            "the property, validated at ZERO tolerance by the same ORT oracle "
            "the bank uses to accept sat witnesses. "
            + (
                "The banked verdict is 'sat', so this AGREES with the bank and "
                "only calibrates the search."
                if agrees_with_bank
                else f"REFUTED: the banked '{row.verdict}' is WRONG."
            )
        )
    elif reduced_holds:
        record["status"] = ST_REFUTED_REDUCED
        record["detail"] = (
            "Every assertion of the property holds at this point under "
            "vnnlib_ce's own evaluator and a batch-1 ORT run, but the FULL "
            "validator refuses this property class outright, so this is NOT a "
            "GENUINE-IN-BOX-CE record. "
            + (
                "The banked verdict is 'sat'; this only calibrates the search."
                if agrees_with_bank
                else f"The banked '{row.verdict}' is contradicted, pending a "
                "human check of the reduced-oracle caveat."
            )
        )
    else:
        record["status"] = ST_CANDIDATE
        record["detail"] = (
            "the batched search evaluator judged this point a counterexample but "
            "the trusted oracle did not confirm it. NOT a refutation. Needs a "
            "human: either the point is genuinely not a CE (batching/precision "
            "difference) or the oracle cannot express this property."
        )
    return record


def explain(model: SpecModel, point, output, np) -> list[dict]:
    """Which assertions the witness satisfies, and by how much."""
    rows: list[dict] = []
    x = point[None, :]
    y = output[None, :]
    for position, assertion in enumerate(model.output_asserts):
        try:
            margin, holds = compile_boolean(assertion)(x, y)
        except SpecError as error:
            rows.append({"assertion": position, "error": str(error)})
            continue
        entry = {
            "assertion_index": position,
            "kind": "output",
            "margin": float(margin[0]),
            "satisfied": bool(holds[0]),
        }
        if (
            isinstance(assertion, list)
            and assertion
            and assertion[0] == "or"
            and len(assertion) - 1 <= 512
        ):
            satisfied_arms = []
            for arm_index, arm in enumerate(assertion[1:]):
                try:
                    _m, arm_holds = compile_boolean(arm)(x, y)
                except SpecError:
                    continue
                if bool(arm_holds[0]):
                    satisfied_arms.append(
                        {"arm": arm_index, "expression": _render(arm)}
                    )
            entry["satisfied_disjuncts"] = satisfied_arms[:16]
        rows.append(entry)
    return rows


def oracle_precheck(vnnlib_path: Path) -> tuple[bool, str]:
    """Can ``vnnlib_ce.validate`` express this property at all?

    ``_scan_property`` requires every declared ``X_i`` to be constrained by an
    INPUT-ONLY assertion.  ``nn4sys`` never satisfies that: its specs state the
    input interval only inside a mixed ``(or (and (>= X_0 a) (<= X_0 b)
    (<= Y_0 c)) ...)`` disjunction, so the full validator refuses the file
    before ever running the model.  Every ``nn4sys`` row is therefore outside
    the reach of the bank's own zero-tolerance CE checker -- worth knowing
    whether or not this audit finds anything there.
    """
    try:
        vnnlib_ce._scan_property(vnnlib_path)
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, ""


def reduced_oracle(
    onnx_path: Path, vnnlib_path: Path, values: dict[int, float]
) -> tuple[bool, str, list[float]]:
    """Batch-1 re-check for properties the FULL validator refuses structurally.

    Everything that decides the verdict is still the trusted code path:
    ``vnnlib_ce`` parses the property, ``vnnlib_ce.evaluate`` supplies the
    semantics, and a fresh single-point ORT session on the untouched model
    supplies Y -- exactly as ``vnnlib_ce.validate`` does.  The ONLY thing
    skipped is ``_scan_property``'s structural completeness scan, which is what
    refused the file.  Because the input bounds live inside the same assertions
    being evaluated, the in-box condition is still checked here; it is simply
    checked as part of the property rather than as a separate gate.

    A hit confirmed only this way is reported as REFUTED-REDUCED-ORACLE and is
    NEVER merged into the REFUTED count.
    """
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    count = len(values)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    model_input = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in model_input.shape]
    if math.prod(shape) != count:
        return False, f"ONNX input shape {shape} does not match {count} inputs", []
    array = np.fromiter(
        (vnnlib_ce._float32(values[i]) for i in range(count)),
        dtype=np.float32,
        count=count,
    )
    outputs = session.run(None, {model_input.name: array.reshape(shape)})
    if len(outputs) != 1:
        return False, f"expected one ONNX output tensor, found {len(outputs)}", []
    output = outputs[0].flatten().astype(np.float64)
    environment = vnnlib_ce._VariableEnvironment(values, output, executed_inputs=False)
    all_results, output_results = vnnlib_ce._evaluate_full(vnnlib_path, environment)
    detail = (
        f"reduced: all_hold={all_results.all_hold} "
        f"assertions={all_results.count} "
        f"all_results={all_results.detail()} "
        f"output_results={output_results.detail()}"
    )
    return bool(all_results.all_hold), detail, [float(v) for v in output]


def _render(node: Any, limit: int = 240) -> str:
    if isinstance(node, str):
        return node
    text = "(" + " ".join(_render(child, limit) for child in node) + ")"
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _worker(payload: dict) -> dict:
    """Never let one bad row abort a whole audit -- a crash is a SKIP, reported."""
    row = BankRow(**payload.pop("row"))
    try:
        return audit_row(row=row, **payload)
    except BaseException as error:  # noqa: BLE001 - reported, never swallowed
        import traceback  # noqa: PLC0415

        return {
            "category": row.category,
            "onnx": row.onnx,
            "vnnlib": row.vnnlib,
            "banked_verdict": row.verdict,
            "banked_seconds": row.seconds,
            "bank_source": row.source,
            "status": ST_SKIP,
            "detail": f"auditor crashed: {type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-2000:],
            "effort": Effort().as_dict(),
            "one_sided_note": ONE_SIDED_BANNER,
        }


def build_report(records: list[dict], args, notes: list[str], coverage: dict) -> dict:
    statuses = Counter(record["status"] for record in records)
    return {
        "tool": "scripts/audit_unsat_by_falsification.py",
        "READ_THIS_FIRST": ONE_SIDED_BANNER,
        "what_a_miss_means": (
            "NO-CE-FOUND means the search failed. It does NOT mean the row is "
            "verified, sound, correct, or cleared. This audit can only ever "
            "produce refutations; it can never produce confirmations."
        ),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "git_head": _git_head(),
        "runtime_versions": vnnlib_ce.runtime_versions(),
        "arguments": {
            "budget_seconds": args.budget_seconds,
            "categories": args.categories,
            "only_blind": args.only_blind,
            "self_test": args.self_test,
            "seed": args.seed,
            "jobs": args.jobs,
            "threads": args.threads,
            "batch": args.batch,
            "max_spec_mb": args.max_spec_mb,
            "max_expression_tokens": args.max_expression_tokens or DEFAULT_EXPRESSION_CAP,
            "bank_dirs": [str(p) for p in args.bank],
            "benchmark_root": str(args.benchmark_root),
        },
        "coverage": coverage,
        "status_counts": dict(statuses),
        "refutations": [r for r in records if r["status"] == ST_REFUTED],
        "reduced_oracle_refutations": [
            r for r in records if r["status"] == ST_REFUTED_REDUCED
        ],
        "unconfirmed_candidates": [r for r in records if r["status"] == ST_CANDIDATE],
        "rows": records,
        "notes": notes,
    }


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return "unknown"


def merge_reports(paths: list[Path]) -> dict:
    """Fold several audit runs into one coverage picture.

    Rows audited more than once (a cheap sweep plus a deep pass) collapse to the
    run that spent the most points, so the merged table never claims less effort
    than was actually spent -- and never claims more coverage than was.
    """
    best: dict[tuple[str, str, str], dict] = {}
    banked: dict[str, int] = {}
    sources: list[dict] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        sources.append(
            {
                "path": str(path),
                "generated_utc": report.get("generated_utc"),
                "git_head": report.get("git_head"),
                "budget_seconds": report.get("arguments", {}).get("budget_seconds"),
                "rows": len(report.get("rows", [])),
            }
        )
        banked.update(report.get("coverage", {}).get("per_category_banked", {}))
        for row in report.get("rows", []):
            identity = (row["category"], row["onnx"], row["vnnlib"])
            previous = best.get(identity)
            if previous is None or row["effort"]["points_tried"] >= previous["effort"][
                "points_tried"
            ]:
                best[identity] = row

    per_category: dict[str, Counter] = defaultdict(Counter)
    points: Counter[str] = Counter()
    wall: Counter[str] = Counter()
    for (category, _onnx, _vnnlib), row in best.items():
        per_category[category][row["status"]] += 1
        points[category] += row["effort"]["points_tried"]
        wall[category] += row["effort"]["wall_seconds"]

    skips: Counter[tuple[str, str]] = Counter()
    oracle_refused: Counter[str] = Counter()
    for row in best.values():
        if row["status"] == ST_SKIP:
            skips[(row["category"], row["detail"][:100])] += 1
        if row.get("full_oracle_can_validate_this_spec") is False:
            oracle_refused[row["category"]] += 1

    return {
        "READ_THIS_FIRST": ONE_SIDED_BANNER,
        "sources": sources,
        "per_category": {
            category: {
                "banked_unsat": banked.get(category, 0),
                "audited": sum(counts.values()),
                **{status: count for status, count in counts.items()},
                "points_tried": points[category],
                "wall_seconds": round(wall[category], 1),
            }
            for category, counts in sorted(per_category.items())
        },
        "not_audited_per_category": {
            category: banked[category] - sum(per_category[category].values())
            for category in sorted(banked)
            if banked[category] > sum(per_category[category].values())
        },
        "skip_reasons": [
            {"category": category, "count": count, "reason": reason}
            for (category, reason), count in sorted(skips.items())
        ],
        "full_oracle_refuses_spec_per_category": dict(sorted(oracle_refused.items())),
        "refutations": [
            row for row in best.values() if row["status"].startswith("REFUTED")
        ],
        "unconfirmed_candidates": [
            row for row in best.values() if row["status"] == ST_CANDIDATE
        ],
    }


def print_merged(merged: dict) -> None:
    print(merged["READ_THIS_FIRST"])
    print("\nmerged from:")
    for source in merged["sources"]:
        print(
            f"  {source['path']}  rows={source['rows']} "
            f"budget={source['budget_seconds']}s head={source['git_head'][:12]}"
        )
    header = (
        f"\n{'category':30s} {'banked':>6s} {'audit':>6s} {'REFUT':>6s} {'REDUC':>6s} "
        f"{'CAND':>5s} {'no-ce':>6s} {'skip':>5s} {'Mpoints':>9s} {'wall_s':>8s}"
    )
    print(header)
    print("-" * (len(header) - 1))
    totals: Counter[str] = Counter()
    for category, row in merged["per_category"].items():
        print(
            f"{category:30s} {row['banked_unsat']:6d} {row['audited']:6d} "
            f"{row.get(ST_REFUTED, 0):6d} {row.get(ST_REFUTED_REDUCED, 0):6d} "
            f"{row.get(ST_CANDIDATE, 0):5d} {row.get(ST_NO_CE, 0):6d} "
            f"{row.get(ST_SKIP, 0):5d} {row['points_tried'] / 1e6:9.1f} "
            f"{row['wall_seconds']:8.0f}"
        )
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals[key] += value
    print("-" * (len(header) - 1))
    print(
        f"{'TOTAL':30s} {totals['banked_unsat']:6d} {totals['audited']:6d} "
        f"{totals[ST_REFUTED]:6d} {totals[ST_REFUTED_REDUCED]:6d} "
        f"{totals[ST_CANDIDATE]:5d} {totals[ST_NO_CE]:6d} {totals[ST_SKIP]:5d} "
        f"{totals['points_tried'] / 1e6:9.1f} {totals['wall_seconds']:8.0f}"
    )
    if merged["not_audited_per_category"]:
        print("\nNOT audited -- these banked unsat rows were never examined:")
        for category, count in merged["not_audited_per_category"].items():
            print(f"  {category:30s} {count}")
    if merged["skip_reasons"]:
        print("\nskip reasons:")
        for entry in merged["skip_reasons"]:
            print(f"  {entry['category']:28s} {entry['count']:4d}  {entry['reason']}")
    if merged["full_oracle_refuses_spec_per_category"]:
        print(
            "\nrows whose spec the FULL zero-tolerance validator refuses "
            "structurally (a hit there could only ever be REFUTED-REDUCED-ORACLE):"
        )
        for category, count in merged["full_oracle_refuses_spec_per_category"].items():
            print(f"  {category:30s} {count}")
    print(f"\nREFUTATIONS: {len(merged['refutations'])}")
    for row in merged["refutations"]:
        print(
            f"  {row['status']}  {row['category']} {row['onnx']} {row['vnnlib']} "
            f"({row['bank_source']})"
        )
    if not merged["refutations"]:
        print(
            "\n*** NOT A SOUNDNESS RESULT. *** No row above was verified. The "
            "search did not find a counterexample on the rows it looked at; the "
            "rows listed as NOT audited were never looked at."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bank",
        type=Path,
        action="append",
        default=None,
        help="bank directory of verdict CSVs (repeatable); "
        "defaults to reports/measured and reports/measured-ext",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=REPO / "benchmarks" / "vnncomp2025" / "benchmarks",
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=REPO / "external_tools" / "vnncomp2025_results",
        help="official results tree, used ONLY to annotate/prioritise rows",
    )
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument(
        "--only-blind",
        action="store_true",
        help="restrict to banked unsat rows the field never decided -- exactly "
        "where the counterexample-based moat cannot see",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="audit banked SAT rows instead: a counterexample is asserted to "
        "exist there, so the hit rate calibrates how much a miss is worth",
    )
    parser.add_argument("--budget-seconds", type=float, default=60.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--sample-per-category",
        type=int,
        default=0,
        help="audit at most N evenly-spread rows per category (a SAMPLE -- the "
        "coverage table reports exactly which rows were left unexamined)",
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="ORT intra-op threads per worker (-1: 1 when --jobs > 1, else ORT's "
        "default). Raise it for models ORT refuses to batch -- vggnet16 runs "
        "batch-1 at ~0.25 s a forward, so threads are the only throughput left",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-spec-mb", type=float, default=64.0)
    parser.add_argument(
        "--max-expression-tokens",
        type=int,
        default=0,
        help="raise vnnlib_ce's per-expression token guard for this run (0 keeps "
        f"its shipped {vnnlib_ce.MAX_EXPRESSION_TOKENS}); needed to look at "
        "nn4sys at all, and recorded on every row it affects",
    )
    parser.add_argument("--out", type=Path, default=None, help="write JSON report here")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="directory for counterexample files produced on a hit",
    )
    parser.add_argument(
        "--retry-skipped",
        type=Path,
        default=None,
        help="audit only the rows an earlier report left SKIPPED, so a raised "
        "--max-spec-mb / --max-expression-tokens can close that gap without "
        "re-running everything",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="*",
        default=None,
        help="merge existing report JSONs into one coverage table and exit",
    )
    parser.add_argument(
        "--list-blind-rows",
        action="store_true",
        help="print the banked-unsat rows with NO field ground truth and exit",
    )
    args = parser.parse_args(argv)
    if args.bank is None:
        args.bank = [REPO / "reports" / "measured", REPO / "reports" / "measured-ext"]

    if args.merge:
        merged = merge_reports(list(args.merge))
        print_merged(merged)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        return 1 if merged["refutations"] else 0

    try:
        vnnlib_ce.require_runtime_dependencies()
    except vnnlib_ce.MissingDependencyError as error:
        print(f"ENVIRONMENT ERROR: {error}", file=sys.stderr)
        return 3

    notes: list[str] = []
    try:
        bank_rows = dedupe(read_bank([p for p in args.bank if p.is_dir()]))
    except BankFormatError as error:
        print(f"ENVIRONMENT ERROR: {error}", file=sys.stderr)
        return 3
    missing = [p for p in args.bank if not p.is_dir()]
    for path in missing:
        notes.append(f"bank directory missing, NOT audited: {path}")
    if not args.benchmark_root.is_dir():
        print(
            f"ENVIRONMENT ERROR: benchmark root {args.benchmark_root} is missing. "
            "A worktree without benchmarks/ makes this audit VACUOUS.",
            file=sys.stderr,
        )
        return 3

    annotations, gt_notes = load_field_ground_truth(args.official)
    notes.extend(gt_notes)
    if not annotations:
        message = (
            "NO field ground truth was loaded, so no row can be classified as "
            "moat-blind and prioritisation is disabled. "
            + " ".join(gt_notes)
        )
        notes.append(message)
        if args.only_blind or args.list_blind_rows:
            # Refusing here is the whole point: a blind-row inventory computed
            # without ground truth would silently report ZERO blind rows, which
            # is the most dangerous possible wrong answer for this audit.
            print(f"ENVIRONMENT ERROR: {message}", file=sys.stderr)
            return 3
        print(f"WARNING: {message}", file=sys.stderr)

    wanted_verdict = "sat" if args.self_test else "unsat"
    population = [row for row in bank_rows if row.verdict == wanted_verdict]

    def annotate(row: BankRow) -> dict:
        return annotations.get(
            (row.category, tail2(row.onnx), tail2(row.vnnlib)), UNANNOTATED
        )

    blind = [row for row in population if annotate(row)["field_blind"]]
    row_blind = [row for row in population if annotate(row)["row_blind"]]
    if args.list_blind_rows:
        print(ONE_SIDED_BANNER)
        print(
            f"\nBanked '{wanted_verdict}' rows the counterexample-based moat "
            f"CANNOT see: {len(blind)} of {len(population)}.\n\n"
            "  row_blind       the field never decided the row (True Result '-')\n"
            "  category_blind  the field accepted ZERO counterexamples anywhere\n"
            "                  in the category, so the per-row (v)/is_fals moat\n"
            "                  can never fire on ANY row of it\n"
        )
        print(
            f"  {'category':34s} {'blind':>6s} {'row_blind':>10s} "
            f"{'cat_blind':>10s} {'banked':>7s}"
        )
        banked_per_category = Counter(row.category for row in population)
        blind_per_category = Counter(row.category for row in blind)
        rowblind_per_category = Counter(row.category for row in row_blind)
        for category in sorted(banked_per_category):
            example = next(r for r in population if r.category == category)
            info = annotate(example)
            print(
                f"  {category:34s} {blind_per_category[category]:6d} "
                f"{rowblind_per_category[category]:10d} "
                f"{'YES' if info['category_blind'] else '-':>10s} "
                f"{banked_per_category[category]:7d}"
            )
        print()
        for row in sorted(blind, key=lambda r: (r.category, r.onnx, r.vnnlib)):
            info = annotate(row)
            print(
                f"{row.category},{row.onnx},{row.vnnlib},"
                f"true_result={info['true_result']},"
                f"row_blind={info['row_blind']},"
                f"category_blind={info['category_blind']},{row.source}"
            )
        return 0

    selected = blind if args.only_blind else population
    if args.categories:
        selected = [row for row in selected if row.category in args.categories]
    # Field-blind rows first: that is exactly where nothing else can catch us.
    selected.sort(
        key=lambda r: (
            0 if annotate(r)["field_blind"] else 1,
            r.category,
            r.onnx,
            r.vnnlib,
        )
    )
    if args.retry_skipped:
        earlier = json.loads(args.retry_skipped.read_text(encoding="utf-8"))
        wanted = {
            (row["category"], row["onnx"], row["vnnlib"])
            for row in earlier.get("rows", [])
            if row["status"] == ST_SKIP
        }
        notes.append(
            f"restricted to the {len(wanted)} rows left SKIPPED by "
            f"{args.retry_skipped}"
        )
        selected = [
            row for row in selected if (row.category, row.onnx, row.vnnlib) in wanted
        ]
    if args.sample_per_category:
        grouped: dict[str, list[BankRow]] = defaultdict(list)
        for row in selected:
            grouped[row.category].append(row)
        sampled: list[BankRow] = []
        for category in sorted(grouped):
            rows = grouped[category]
            take = min(args.sample_per_category, len(rows))
            stride = max(1, len(rows) // take)
            sampled.extend(rows[::stride][:take])
        selected = sampled
    if args.max_rows:
        selected = selected[: args.max_rows]

    print(ONE_SIDED_BANNER)
    print(
        f"\nauditing {len(selected)} banked '{wanted_verdict}' rows "
        f"(bank holds {len(population)}; {len(blind)} of them have no field "
        f"ground truth) at {args.budget_seconds:g}s each, jobs={args.jobs}\n",
        flush=True,
    )

    payloads = [
        {
            "row": row.__dict__,
            "benchmark_root": args.benchmark_root,
            "budget_seconds": args.budget_seconds,
            "evidence_dir": args.evidence_dir,
            "max_spec_mb": args.max_spec_mb,
            "max_expression_tokens": args.max_expression_tokens or DEFAULT_EXPRESSION_CAP,
            "seed": args.seed,
            "threads": (1 if args.jobs > 1 else 0) if args.threads < 0 else args.threads,
            "batch": args.batch,
            "max_expression_tokens": args.max_expression_tokens,
        }
        for row in selected
    ]

    records: list[dict] = []
    started = time.monotonic()
    if args.jobs > 1:
        with futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for record in pool.map(_worker, payloads, chunksize=1):
                records.append(record)
                _emit(record, len(records), len(payloads))
    else:
        for payload in payloads:
            record = _worker(payload)
            records.append(record)
            _emit(record, len(records), len(payloads))

    for record in records:
        info = annotations.get(
            (record["category"], tail2(record["onnx"]), tail2(record["vnnlib"])),
            UNANNOTATED,
        )
        record["field"] = info
        record["field_blind"] = info["field_blind"]

    coverage = {
        "bank_rows_total": len(bank_rows),
        f"banked_{wanted_verdict}_rows_total": len(population),
        "rows_audited": len(records),
        f"banked_{wanted_verdict}_rows_NOT_audited": len(population) - len(records),
        "moat_blind_rows_total": len(blind),
        "moat_blind_rows_audited": sum(1 for r in records if r.get("field_blind")),
        "row_blind_rows_total": len(row_blind),
        "per_category_audited": dict(Counter(r["category"] for r in records)),
        "per_category_banked": dict(Counter(r.category for r in population)),
        "wall_seconds": round(time.monotonic() - started, 1),
    }

    report = build_report(records, args, notes, coverage)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=False), "utf-8")

    print_summary(report, wanted_verdict)

    if args.self_test:
        # Hits here agree with the bank; they calibrate, they do not accuse.
        return 0
    if report["refutations"] or report["reduced_oracle_refutations"]:
        return 1
    if report["unconfirmed_candidates"]:
        return 2
    return 0


def _emit(record: dict, index: int, total: int) -> None:
    effort = record.get("effort", {})
    marker = {
        ST_REFUTED: "!! REFUTED !!",
        ST_REFUTED_REDUCED: "!! REFUTED (reduced oracle) !!",
        ST_CANDIDATE: "?? CANDIDATE ??",
        ST_NO_CE: "no-ce (PROVES NOTHING)",
        ST_SKIP: "SKIPPED",
    }[record["status"]]
    print(
        f"[{index}/{total}] {record['category']}/{Path(record['vnnlib']).name} "
        f"{marker} points={effort.get('points_tried', 0)} "
        f"wall={effort.get('wall_seconds', 0)}s "
        f"best_margin={effort.get('best_full_property_margin')}"
        + (f" -- {record['detail']}" if record["status"] == ST_SKIP else ""),
        flush=True,
    )


def print_summary(report: dict, wanted_verdict: str) -> None:
    coverage = report["coverage"]
    print("\n" + "=" * 78)
    print(report["READ_THIS_FIRST"])
    print("=" * 78)
    print(f"\nCOVERAGE (what was actually looked at)")
    for key, value in coverage.items():
        if isinstance(value, dict):
            continue
        print(f"  {key:44s} {value}")
    print("\n  audited per category:")
    for category, count in sorted(coverage["per_category_audited"].items()):
        banked = coverage["per_category_banked"].get(category, 0)
        print(f"    {category:36s} {count:5d} of {banked} banked")
    not_audited = {
        category: count
        for category, count in coverage["per_category_banked"].items()
        if count > coverage["per_category_audited"].get(category, 0)
    }
    if not_audited:
        print("\n  NOT (fully) audited -- these rows were not looked at:")
        for category, count in sorted(not_audited.items()):
            done = coverage["per_category_audited"].get(category, 0)
            print(f"    {category:36s} {count - done:5d} of {count} banked left")

    print("\nOUTCOMES")
    for status, count in sorted(report["status_counts"].items()):
        print(f"  {status:24s} {count}")
    skips = Counter(
        r["detail"].split(":")[0]
        for r in report["rows"]
        if r["status"] == ST_SKIP
    )
    if skips:
        print("\n  skip reasons:")
        for reason, count in skips.most_common():
            print(f"    {reason:60s} {count}")

    hits = report["refutations"] + report["reduced_oracle_refutations"]
    if wanted_verdict == "sat":
        # Calibration mode: a hit AGREES with the bank. It measures the search,
        # it does not accuse anything.
        print("\n" + "=" * 78)
        audited = coverage["rows_audited"]
        rate = (len(hits) / audited * 100) if audited else 0.0
        print(
            f"SELF-TEST CALIBRATION: the search independently rediscovered a "
            f"validated counterexample on {len(hits)} of {audited} banked SAT "
            f"rows ({rate:.0f}%)."
        )
        print(
            "Read every NO-CE-FOUND on the unsat side against this number: a "
            "miss is only as meaningful as the search is strong, and this is "
            "the only measurement of that strength."
        )
        print("=" * 78)
        return
    if hits:
        print("\n" + "!" * 78)
        print(f"{len(hits)} BANKED ROW(S) REFUTED -- treat as an incident")
        print("!" * 78)
        for record in hits:
            print(f"\n  row            {record['category']} {record['onnx']} {record['vnnlib']}")
            print(f"  banked         {record['banked_verdict']} @ {record['banked_seconds']}s ({record['bank_source']})")
            print(f"  status         {record['status']}")
            print(f"  oracle         {record['oracle']['verdict']}")
            print(f"  oracle detail  {record['oracle']['detail'][:300]}")
            if record.get("reduced_oracle"):
                print(f"  reduced oracle {record['reduced_oracle']['detail'][:300]}")
                print(f"  reduced caveat {record['reduced_oracle']['why']}")
            print(f"  ORT output     {record['ort_output'][:16]}")
            for constraint in record["violated_constraints"][:4]:
                print(f"  satisfied      {constraint}")
            if record.get("counterexample_file"):
                print(f"  witness        {record['counterexample_file']}")
                print(f"  reproduce      {record['reproduce']}")
    else:
        print(
            f"\nNo banked {wanted_verdict} row was refuted in this run.\n"
            "*** THAT IS NOT A SOUNDNESS RESULT. *** Nothing here was verified. "
            "The search simply did not find a counterexample within the budget "
            "on the rows it looked at. Rows listed as NOT audited above were "
            "never examined at all."
        )
    if report["unconfirmed_candidates"]:
        print(
            f"\n{len(report['unconfirmed_candidates'])} UNCONFIRMED candidate(s) "
            "need a human: the search evaluator called them counterexamples and "
            "the trusted oracle did not agree."
        )


if __name__ == "__main__":
    raise SystemExit(main())
