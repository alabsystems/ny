#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""
Emit a JSON differential fixture for the Rust 2-ReLU multi-neuron hull tests.

For a fixed set of SHARED affine pre-activation maps x = W u + c over an input
box [u_lo, u_hi], this reuses the *validated* numpy/scipy oracle in
`validate_hull.py` (all checks pass) to dump, per case:

  - W, c, u_lo, u_hi                         (the shared inputs)
  - the octahedral P bound vector `b`        (producer output, f64)
  - the lifted arrangement vertices V        (Theorem 2(iii), sorted)
  - the excluded box corner + its residual   (tightness witness, §1.3.1 / (T))
  - `both_unstable`                          (guard)

The Rust differential test (soundness gate (c)) loads (W,c,box), runs its OWN
independent pipeline, and asserts its V-set and excluded-corner residual match
this oracle's within tolerance. Same inputs, independently-computed outputs.

Run:  python3 gen_fixture.py > multineuron_fixture.json
"""

import json
import sys

import numpy as np

import validate_hull as vh


# Shared, explicit cases (NOT tied to validate_hull's RNG stream, so the fixture
# is stable and reproducible). Cover: the symmetric diamond, a rotation, a
# sheared/asymmetric map, and a couple of general 2x3 maps.
CASES = [
    ("diamond", [[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0], [-1.0, -1.0], [1.0, 1.0]),
    ("asym_shear", [[1.0, 0.4], [0.3, -1.0]], [0.1, -0.2], [-1.0, -1.0], [1.0, 1.0]),
    ("wide_x1", [[1.5, 0.5], [0.5, 1.0]], [0.0, 0.2], [-1.0, -1.0], [1.0, 1.0]),
    ("map3a", [[1.0, -0.7, 0.5], [0.6, 1.0, -0.4]], [0.05, -0.1],
     [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
    ("map3b", [[0.8, 0.9, -0.3], [-0.5, 0.7, 0.9]], [-0.15, 0.2],
     [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
]


def emit_case(name, W, c, u_lo, u_hi):
    W = np.array(W, dtype=float)
    c = np.array(c, dtype=float)
    u_lo = np.array(u_lo, dtype=float)
    u_hi = np.array(u_hi, dtype=float)

    A, b, (l1, u1, l2, u2) = vh.octahedral_P(W, c, u_lo, u_hi)
    both_unstable = bool((l1 < 0 < u1) and (l2 < 0 < u2))

    out = {
        "name": name,
        "W": W.tolist(),
        "c": c.tolist(),
        "u_lo": u_lo.tolist(),
        "u_hi": u_hi.tolist(),
        "octa_b": b.tolist(),  # order: u1,-l1,u2,-l2,su,-sl,du,-dl
        "l1": l1, "u1": u1, "l2": l2, "u2": u2,
        "both_unstable": both_unstable,
    }
    if not both_unstable:
        out["verts"] = []
        out["excluded_corner_residual"] = None
        out["corner_reachable"] = None
        return out

    verts = vh.arrangement_lifted_vertices(A, b)
    # sort rows for a stable set comparison
    vsorted = verts[np.lexsort(verts.T[::-1])]
    mnA, mnb = vh.hull_facets(verts)
    mnb = vh.outward_round_rhs(mnA, mnb, verts)

    corner = np.array([[u1, u2, vh.relu(u1), vh.relu(u2)]])
    mn_slack = float((corner @ mnA.T - mnb).max())
    corner_reachable = bool(np.all(A @ np.array([u1, u2]) <= b + 1e-9))

    out["verts"] = vsorted.tolist()
    out["excluded_corner_residual"] = mn_slack
    out["corner_reachable"] = corner_reachable
    return out


def main():
    fixture = {
        "seed_note": "explicit shared cases; oracle = validate_hull.py",
        "cases": [emit_case(*case) for case in CASES],
    }
    json.dump(fixture, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
