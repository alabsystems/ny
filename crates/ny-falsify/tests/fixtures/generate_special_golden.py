#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Regenerate `special_golden.rs`. Run from the repo root:

    python3 crates/ny-falsify/tests/fixtures/generate_special_golden.py

Captures the EXACT point matrix the shipped Python `_strategy_special` emits.
It imports the real 2410-line auditor and instantiates its real `Searcher`; only
the model/runner/objective seams are stubbed, and `evaluate` is intercepted so
the strategy's own point construction (including `materialise`/`snap`) is what
gets recorded. No strategy code is reimplemented here, which is the entire
reason the resulting fixture is worth anything: it is the Python portfolio's own
output, not a second opinion about what it ought to be.

Values are printed with `repr`, whose shortest-round-trip form parses back to
the identical f64 in Rust, so the comparison downstream is bit-for-bit.
"""
import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).with_name("special_golden.rs")

spec = importlib.util.spec_from_file_location(
    "auditor", ROOT / "scripts" / "audit_unsat_by_falsification.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["auditor"] = mod
spec.loader.exec_module(mod)

# A box chosen to exercise every branch of the snapping contract at once: a
# bound that is not float32-representable, a PINNED coordinate, an asymmetric
# interval, and an interval so narrow its midpoint is denormal-adjacent.
LOW = np.array([-0.3035311561, 0.1, -0.5, 2.0, 0.0, -1.0], dtype=np.float64)
HIGH = np.array([0.6798577687, 0.2, 0.5, 2.0, 1.0, 1e-7], dtype=np.float64)

captured = []


class StubRunner:
    session_runs = 0

    def run(self, points):
        raise AssertionError("unreachable: evaluate is intercepted")


searcher = mod.Searcher.__new__(mod.Searcher)
mod.Searcher.__init__(
    searcher,
    model=None,
    runner=StubRunner(),
    low=LOW,
    high=HIGH,
    steer=lambda p, o: (np.zeros(len(p)), np.zeros(len(p), dtype=bool)),
    gate=lambda p, o: (np.zeros(len(p)), np.zeros(len(p), dtype=bool)),
    effort=mod.Effort(),
    rng=np.random.default_rng(20260808),
    target_name="all",
)


def intercept(free_values, strategy):
    values = np.asarray(free_values, dtype=np.float64)
    captured.append((values.copy(), searcher.materialise(values).copy()))
    return np.zeros(len(values))


searcher.evaluate = intercept
searcher._strategy_special(deadline=float("inf"), batch=256)
assert len(captured) == 1, captured
raw, materialised = captured[0]


def literal(value: float) -> str:
    text = repr(float(value))
    assert text not in ("inf", "-inf", "nan"), text
    return text if ("e" in text or "." in text) else text + ".0"


def rows(matrix, name, doc):
    out = [f"/// {line}\n" for line in doc]
    out.append(f"pub const {name}: [[f64; {matrix.shape[1]}]; {matrix.shape[0]}] = [\n")
    for row in matrix:
        out.append("    [%s],\n" % ", ".join(literal(v) for v in row))
    out.append("];\n")
    return "".join(out)


text = f"""// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GOLDEN FIXTURE -- GENERATED, DO NOT HAND-EDIT.
//!
//! Regenerate with `generate_special_golden.py` in this directory. Produced by
//! importing `scripts/audit_unsat_by_falsification.py`, constructing its real
//! `Searcher` over the box below, and intercepting the argument its real
//! `_strategy_special` hands to `evaluate`.
//!
//! The box exercises every branch of the snapping contract at once: a bound
//! that is not float32-representable (`-0.3035311561`, `0.1`), a PINNED
//! coordinate (index 3, `2.0 == 2.0`), an asymmetric interval, and an interval
//! so narrow that its midpoint is denormal-adjacent (`[-1.0, 1e-7]`).
//! numpy version at generation time: {np.__version__}.

/// Declared lower bounds.
pub const LOW: [f64; {len(LOW)}] = [{", ".join(literal(v) for v in LOW)}];
/// Declared upper bounds.
pub const HIGH: [f64; {len(HIGH)}] = [{", ".join(literal(v) for v in HIGH)}];
/// Free coordinate indices the Python `Searcher` derived (index 3 is pinned).
pub const FREE_INDICES: [usize; {len(searcher.free)}] = [{", ".join(str(int(i)) for i in searcher.free)}];

{rows(raw, "RAW_FREE_PATTERNS", ["The eight patterns in FREE coordinates, before snapping, in emission order."])}
{rows(materialised, "MATERIALISED_POINTS", ["The eight points as ORT would see them: free coordinates snapped onto the", "float32 grid inside the box, pinned coordinates left exact."])}"""

OUT.write_text(text)
print(f"wrote {OUT} ({raw.shape[0]} patterns, {materialised.shape[1]} declared inputs)")
