// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// B5 — per-unstable |coefficient| mass collector for the certified margin-row
// GPU backward. Concatenated AFTER ds_primitives.wgsl at pipeline build.
//
// ============================ HEURISTIC ONLY ============================
// THIS KERNEL CARRIES NO CERTIFIED CHARGE, AND MUST NOT.
//
// It mirrors the CPU arm at engine.rs:600-611:
//
//     sums[pos] = sum over rows r of |L[unst[pos], r]|
//
// on the PRE-gate-transform frontier. Every consumer of the resulting
// `PassOut::coll` is a RANKER: the trunk shortlist (bab.rs:3781-3806) and the
// tier-0 candidate pool (bab.rs:~3703). Nothing verdict-bearing reads it —
// the same discipline the #alpha-opt capture states at engine.rs:634. So this
// plane is deliberately absent from `e_widen`/`eb_widen`, from the flush and
// split floors, and from `rho_star_ds`: there is no error lane to charge,
// because no bound is derived from these numbers.
//
// The accumulation is nevertheless ds (not plain f32) for a REPRODUCIBILITY
// reason, not a soundness one: the shortlist is a sort key, so f32 fold noise
// would reorder candidates and change the SHAPE of the search tree between
// otherwise identical runs (the banked "marginal rows need >= 3 runs/arm"
// problem). At ds precision the published value agrees with the CPU f64 sum
// to ~2^-44 relative, so rankings flip only on true ties.
// ========================================================================

struct CollectParams {
    // Number of unstable positions in `idx` (= RootGates::layers[li].unst.len()).
    positions: u32,
    // Pass rows R: the coefficient plane is (n, R) row-major.
    rows: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: CollectParams;
// Absolute neuron index per position — `LayerGates::unst`, host-validated
// against the plane width (transaction.rs::push_collect refuses otherwise).
@group(0) @binding(1) var<storage, read> idx: array<u32>;
// The PRE-transform coefficient plane, ds32, (n, rows) row-major.
@group(0) @binding(2) var<storage, read> coeff_in: array<vec2<f32>>;
// Per-position ds sum of |coefficient| over the pass rows.
@group(0) @binding(3) var<storage, read_write> sums: array<vec2<f32>>;

// Workgroup size 256 (the chunk-256 moat recipe); one thread per position,
// folding the row axis serially in ds — the same element ORDER as the CPU
// arm's `.iter().map(|v| v.abs()).sum()`, so the host twin
// (ds.rs::collect_abs_twin) mirrors it op-for-op and the M1-style bit-parity
// harness stays meaningful.
@compute @workgroup_size(256)
fn collect_abs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= params.positions {
        return;
    }
    let j = idx[p];
    var acc = vec2<f32>(0.0, 0.0);
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        let v = coeff_in[j * params.rows + r];
        // |hi + lo| as a ds pair: negating BOTH components is exact (a sign
        // flip is exact in every band) and preserves the ds invariant, so
        // this is the magnitude of the pair, not an f32 approximation of it.
        let a = select(vec2<f32>(-v.x, -v.y), v, v.x >= 0.0);
        acc = ds_add(acc, a);
    }
    sums[p] = acc;
}
