// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Head-gemm backward (the margin seed fold) for the certified margin-row GPU
// backward (design section 5, kernel "head Gemm backward"). Concatenated
// AFTER ds_primitives.wgsl at pipeline build.
//
// CPU reference: engine.rs, the seed-through-gemm1 arm —
//   L0[k, r] = sum_j W1[j, k] * S[j, r]        (value)
//   E0[k, r] = g_head * sum_j |W1[j, k]| * |S[j, r]|   (certified error)
// with the seed S = (n_y, rows) at the head pre-activation y. The seed's own
// certified error and its f32-downcast residual NEVER reach the device: they
// are concretized into eb by the host's seed_penalty (gpu_seam.rs), so the
// device treats S as exact and the E gather has no incoming err plane.
//
// Certified error (design section 4.2 "Head Gemm backward" row, tightened in
// the error-charge session): the WEIGHTS ride as ds32 PAIRS split from the
// engine's own f64 head weights (normal-band residual <= 2^-48 relative;
// the f32-subnormal band rides the host split-floor channel —
// transaction.rs ds_split_excess, verify-fix 2026-08-20), so the u32
// downcast ball never enters this lane. The value fold runs in ds
// (DWTimesDW + accurate DWPlusDW, theorem-backed), so the host charges
// g_head = up(gamma_ds_dot(n_y + 2) (+) rel) — the per-tap ds envelope
// composed with the ds-split representation ball (the head's f64 weights are
// the lane's EXACT algebra, so there is no fold term; transaction.rs
// ds_weight_rel(0)) — on the (|w.hi| + |w.lo|)|S| mass gathered here. The
// f32 roundings of the mass gather itself are host-widened by the E-lane
// egress factor (1 + gamma_f32(k)); the fold order per output element is
// FIXED (j ascending), so the host ds twin is bit-comparable.
//
// The bias half of the head (b += S @ b1) is a separate bias_reduce dispatch
// on the SEED plane with scale = b1 — not this kernel.

struct HeadParams {
    // Seed rows (head width) n_y, output width n_h, pass rows.
    n_y: u32,
    n_h: u32,
    rows: u32,
    // gamma_ds_dot(n_y + 2) (+) ds-split ball, f64->f32 rounded UP on host.
    g_head: f32,
}

@group(0) @binding(0) var<uniform> hp: HeadParams;
// Head weight as ds32 PAIRS, row-major (n_y, n_h): w[j * n_h + k] — the
// lane's gemm1 layout, split on host from the engine's f64 weights.
@group(0) @binding(1) var<storage, read> head_w: array<vec2<f32>>;
// Seed plane (n_y, rows) ds32, index = j * rows + r (exact: lo = 0 from the
// host's checked downcast; kept ds so the fold shape matches every other lane).
@group(0) @binding(2) var<storage, read> seed_in: array<vec2<f32>>;
// Output coefficient plane (n_h, rows) ds32.
@group(0) @binding(3) var<storage, read_write> head_out: array<vec2<f32>>;
// Output certified-error plane (n_h, rows) f32 (host-widened at egress).
@group(0) @binding(4) var<storage, read_write> head_err: array<f32>;

// Workgroup size 256: the chunk-256 recipe. One thread per (k, r).
@compute @workgroup_size(256)
fn head_gemm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = hp.n_h * hp.rows;
    if idx >= total {
        return;
    }
    let k = idx / hp.rows;
    let r = idx % hp.rows;

    var acc = vec2<f32>(0.0, 0.0); // ds value accumulator
    var mass = 0.0;                // sum (|w.hi| + |w.lo|) * |S| (E gather)
    for (var j = 0u; j < hp.n_y; j = j + 1u) {
        let w = head_w[j * hp.n_h + k];
        let s = seed_in[j * hp.rows + r];
        let term = ds_mul(s, w);
        acc = ds_add(acc, term);
        mass = fma(ds_abs_bound(w), ds_abs_bound(s), mass);
    }
    head_out[idx] = acc;
    head_err[idx] = hp.g_head * mass;
}
