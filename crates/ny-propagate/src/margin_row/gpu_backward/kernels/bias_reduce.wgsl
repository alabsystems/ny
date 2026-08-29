// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Per-row bias reduction for the certified margin-row GPU backward (design
// section 5, the "bias tree-reduce" kernel). Concatenated AFTER
// ds_primitives.wgsl at pipeline build.
//
// Folds one (n, rows) ds plane into the per-row running bias accumulator:
//
//   b[r]  += sum_j scale[j] * plane[j, r]           (ds value)
//   eb[r] += sum_j |scale[j]| * err[j, r]                       (E carry)
//          + berr * sum_j (labs + err[j, r])                    (abs bias ball)
//          + g_reduce * sum_j |scale[j] * plane[j, r]|          (own ds fold)
//          + 2 * DS_UNIT * |b_running|                          (running carry)
//   with labs = |plane.hi| + |plane.lo|, |scale[j]| = |hi| + |lo|.
//
// SCALE IS A ds32 PAIR (split on host from the engine's f64 bias), so the u32
// bias downcast never enters this lane; the split residual is charged through
// `berr` by the host (transaction.rs ds_bias_abs).
//
// One dispatch per bias site, three call shapes (host builder,
// transaction.rs):
//   * head bias:  plane = seed, scale = ds-split b1, berr = split ball;
//   * relu intercepts: plane = the gate kernel's per-element ds partials,
//     use_scale = 0 (the intercept is already folded in; its error plane
//     already carries (E + U_DS|partial|) * |c|), berr = 0;
//   * conv bias:  plane = out-side coefficients, scale = ds-split
//     bias_expanded, berr = fold bias_err (+) split ball.
// This mirrors the CPU arms (engine.rs relu-bias / conv-bias loops): the
// (er + gam_b*mag)*|c| term splits into |c|*er (the E carry line) and the
// fold's own gamma on the |c||L| mass — which is EXACTLY g_reduce * sum|q|,
// charged ONCE. (The pre-tightening kernel ALSO took a caller g_src =
// gamma_ds(n+2) on the same mass: a measured DOUBLE CHARGE of the fold
// envelope, removed in the error-charge session — loosening-only removal is
// wrong-way-proof because g_reduce still dominates the fold per the
// gamma_ds_dot derivation.) The (mag+er)*berr term is the berr line, and the
// CPU's per-step 2u*|b_running| carry becomes the per-dispatch
// 2*DS_UNIT*|b_running| line (one ds_add into the accumulator per dispatch,
// charged on the post-add running magnitude).
//
// TWO ENTRIES (perf session, B2):
//   * `bias_reduce` — the v1 serial reference: ONE THREAD PER ROW,
//     j-ascending fold. Kept as the executable spec and parity anchor; no
//     longer dispatched by the production builder.
//   * `bias_reduce_wg` — the PRODUCTION tree reduce: one 256-lane WORKGROUP
//     per row. Lane l serially folds the contiguous block
//     j in [l*chunk, min((l+1)*chunk, n)), chunk = ceil(n/256), in the SAME
//     per-element order and with the SAME ds ops as the serial entry; the
//     256 lane partials are then combined by a FIXED binary tree
//     (offset 128, 64, .., 1) with workgroup barriers between levels.
//     DETERMINISM: the blocking and the tree are functions of (n, lane)
//     only, so the fold order is fixed and the host twin
//     (ds.rs::bias_reduce_wg_twin) mirrors it OP-FOR-OP — the M1-grade
//     bit-parity gate stays meaningful.
//
// CHARGE (B2, argument at transaction.rs::push_reduce): g_reduce =
// gamma_ds_dot(n+2) is valid for ANY summation order — every multiply
// rounds relative to its term and every add relative to a partial sum of a
// SUBSET of terms, both <= the absolute mass sum|q|; a tree over n terms
// commits n multiplies + at most n-1 non-trivial adds (a ds_add of an
// exact-zero padding-lane partial commits NO rounding: every two_sum with a
// zero operand is exact and the renorm of an in-invariant pair is exact),
// i.e. exactly the serial rounding count, so the serial charge strictly
// dominates. The E-side f32 sums (esum/msum) change fold order too; their
// host egress widening keeps the serial op count (6n+16 >= the tree's
// 2n fma + n + 2(n-1) adds for all n).
//
// The f32 roundings of the E-side sums are host-widened at egress
// (1 + gamma_f32(k)); nothing here rounds a term the host does not widen.

struct ReduceParams {
    n: u32,
    rows: u32,
    use_scale: u32,
    _pad0: u32,
    berr: f32,
    g_reduce: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> rp: ReduceParams;
// The (n, rows) ds plane to fold, index = j * rows + r.
@group(0) @binding(1) var<storage, read> red_plane: array<vec2<f32>>;
// Its elementwise error plane (f32, nonnegative).
@group(0) @binding(2) var<storage, read> red_err: array<f32>;
// Per-j scale as ds32 PAIRS (length n; bind a 1-pair dummy when
// use_scale = 0 — the identity pair is applied in-kernel instead).
@group(0) @binding(3) var<storage, read> red_scale: array<vec2<f32>>;
// Running per-row bias accumulator (ds, length rows).
@group(0) @binding(4) var<storage, read_write> bias_acc: array<vec2<f32>>;
// Running per-row bias-error accumulator (f32, length rows).
@group(0) @binding(5) var<storage, read_write> ebias_acc: array<f32>;

// Workgroup size 256 (the chunk-256 recipe); one thread per pass row.
@compute @workgroup_size(256)
fn bias_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= rp.rows {
        return;
    }
    var acc = vec2<f32>(0.0, 0.0);
    var esum = 0.0;
    var msum = 0.0;
    for (var j = 0u; j < rp.n; j = j + 1u) {
        let p = red_plane[j * rp.rows + r];
        var w = vec2<f32>(1.0, 0.0);
        if rp.use_scale == 1u {
            w = red_scale[j];
        }
        let q = ds_mul(p, w);
        acc = ds_add(acc, q);
        let labs = ds_abs_bound(p);
        let e = red_err[j * rp.rows + r];
        esum = fma(ds_abs_bound(w), e, esum);
        esum = fma(rp.berr, labs + e, esum);
        msum = msum + ds_abs_bound(q);
    }
    let nb = ds_add(bias_acc[r], acc);
    bias_acc[r] = nb;
    // Running-accumulator carry (mirrors the CPU's 2u*|b_running| per step;
    // one ds_add per dispatch here, charged at 2x DS_UNIT for headroom).
    let carry = 2.0 * DS_UNIT * ds_abs_bound(nb);
    ebias_acc[r] = ebias_acc[r] + esum + rp.g_reduce * msum + carry;
}

// ---------------------------------------------------------------------------
// B2 tree reduce (production entry): one 256-lane workgroup per row.
// ---------------------------------------------------------------------------

var<workgroup> red_acc: array<vec2<f32>, 256>;
var<workgroup> red_esum: array<f32, 256>;
var<workgroup> red_msum: array<f32, 256>;

@compute @workgroup_size(256)
fn bias_reduce_wg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = wid.x; // one workgroup per row; uniform within the workgroup
    if r >= rp.rows {
        return;
    }
    let lane = lid.x;
    // Contiguous per-lane block, SAME per-element body as the serial entry.
    let chunk = (rp.n + 255u) / 256u;
    let j0 = lane * chunk;
    let j1 = min(j0 + chunk, rp.n);
    var acc = vec2<f32>(0.0, 0.0);
    var esum = 0.0;
    var msum = 0.0;
    for (var j = j0; j < j1; j = j + 1u) {
        let p = red_plane[j * rp.rows + r];
        var w = vec2<f32>(1.0, 0.0);
        if rp.use_scale == 1u {
            w = red_scale[j];
        }
        let q = ds_mul(p, w);
        acc = ds_add(acc, q);
        let labs = ds_abs_bound(p);
        let e = red_err[j * rp.rows + r];
        esum = fma(ds_abs_bound(w), e, esum);
        esum = fma(rp.berr, labs + e, esum);
        msum = msum + ds_abs_bound(q);
    }
    red_acc[lane] = acc;
    red_esum[lane] = esum;
    red_msum[lane] = msum;
    workgroupBarrier();
    // Fixed binary tree; empty lanes hold exact zeros (exact ds_add, no
    // rounding — see the header charge note).
    var off = 128u;
    loop {
        if off == 0u {
            break;
        }
        if lane < off {
            red_acc[lane] = ds_add(red_acc[lane], red_acc[lane + off]);
            red_esum[lane] = red_esum[lane] + red_esum[lane + off];
            red_msum[lane] = red_msum[lane] + red_msum[lane + off];
        }
        workgroupBarrier();
        off = off >> 1u;
    }
    if lane == 0u {
        let nb = ds_add(bias_acc[r], red_acc[0]);
        bias_acc[r] = nb;
        let carry = 2.0 * DS_UNIT * ds_abs_bound(nb);
        ebias_acc[r] = ebias_acc[r] + red_esum[0] + rp.g_reduce * red_msum[0] + carry;
    }
}
