// Copyright 2026 Andrew Yates
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
//! numpy version at generation time: 2.0.2.

/// Declared lower bounds.
pub const LOW: [f64; 6] = [-0.3035311561, 0.1, -0.5, 2.0, 0.0, -1.0];
/// Declared upper bounds.
pub const HIGH: [f64; 6] = [0.6798577687, 0.2, 0.5, 2.0, 1.0, 1e-07];
/// Free coordinate indices the Python `Searcher` derived (index 3 is pinned).
pub const FREE_INDICES: [usize; 5] = [0, 1, 2, 4, 5];

/// The eight patterns in FREE coordinates, before snapping, in emission order.
pub const RAW_FREE_PATTERNS: [[f64; 5]; 8] = [
    [-0.3035311561, 0.1, -0.5, 0.0, -1.0],
    [0.6798577687, 0.2, 0.5, 1.0, 1e-07],
    [
        0.18816331028938293,
        0.15000000596046448,
        0.0,
        0.5,
        -0.4999999403953552,
    ],
    [-0.3035311561, 0.2, -0.5, 1.0, -1.0],
    [0.6798577687, 0.1, 0.5, 0.0, 1e-07],
    [0.6798577687, 0.1, -0.5, 1.0, -1.0],
    [
        -0.05768392290530852,
        0.12500000298023223,
        -0.25,
        0.25,
        -0.7499999701976776,
    ],
    [
        0.4340105394946915,
        0.17500000298023224,
        0.25,
        0.75,
        -0.2499999201976776,
    ],
];

/// The eight points as ORT would see them: free coordinates snapped onto the
/// float32 grid inside the box, pinned coordinates left exact.
pub const MATERIALISED_POINTS: [[f64; 6]; 8] = [
    [
        -0.30353114008903503,
        0.10000000149011612,
        -0.5,
        2.0,
        0.0,
        -1.0,
    ],
    [
        0.6798577308654785,
        0.19999998807907104,
        0.5,
        2.0,
        1.0,
        9.999999406318238e-08,
    ],
    [
        0.18816331028938293,
        0.15000000596046448,
        0.0,
        2.0,
        0.5,
        -0.4999999403953552,
    ],
    [
        -0.30353114008903503,
        0.19999998807907104,
        -0.5,
        2.0,
        1.0,
        -1.0,
    ],
    [
        0.6798577308654785,
        0.10000000149011612,
        0.5,
        2.0,
        0.0,
        9.999999406318238e-08,
    ],
    [
        0.6798577308654785,
        0.10000000149011612,
        -0.5,
        2.0,
        1.0,
        -1.0,
    ],
    [-0.05768392235040665, 0.125, -0.25, 2.0, 0.25, -0.75],
    [
        0.4340105354785919,
        0.17499999701976776,
        0.25,
        2.0,
        0.75,
        -0.24999992549419403,
    ],
];
