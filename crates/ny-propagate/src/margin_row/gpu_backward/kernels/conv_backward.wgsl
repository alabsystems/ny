// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Conv backward (transposed-conv gather) for the certified margin-row GPU
// backward (design section 5, kernel 3). Concatenated AFTER
// ds_primitives.wgsl at pipeline build.
//
// One thread computes ONE (input position, pass row) output element:
//   L_in[ci, iy, ix, row] = sum over (co, kh, kw) with
//       oy = (iy + pad_h - kh) / stride_h, ox = (ix + pad_w - kw) / stride_w
//       (integer, in range)
//   of W[co, ci, kh, kw] * L_out[co, oy, ox, row]
// — the adjoint of the forward conv, matching the CPU lane's
// conv_apply_backward gather.
//
// Certified error (design section 4.2, "conv backward" row): the VALUE lane
// accumulates in ds32 through the theorem-backed ds ops, so its entire charge
// is the HOST-side a-priori gamma_ds(k) envelope with the ds unit U_DS =
// 2^-44 — ~2.6e-10 relative at k = 4608, i.e. f64-grade, which is the entire
// departure from the verdict-breaking f32 Higham charge (design section 2.1:
// gamma multiplies |L| mass either way, but with U_DS instead of 2^-24 the
// product is ~1e8x below margin scale instead of above it). The E-lane
// gathers |W| * E in plain f32, host-widened by (1 + gamma_f32(k)) in f64 at
// egress. The model-error ball (weight_rel_err) stays a host-side
// CertifiedWeightError charge exactly as in gpu_seam.rs. The `residual`
// binding is RESERVED for the implementation session's optional a-posteriori
// refinement (per-tap EFT residual capture); the skeleton zeroes it.
//
// UNDERFLOW: products in the 2^-101 band void the TwoProduct exactness
// theorem; rather than branch per tap, the HOST charges the conditional
// 2^-126-per-tap floor for the whole dispatch (design 4.1) — absolute,
// margin-invisible, and sound by domination. The word channel itself is an
// ADMISSION requirement, not an assumption: see the SUBNORMAL CONTRACT in
// ds_primitives.wgsl and ../authority.rs (review item 3).
//
// STATUS: skeleton delivery — correctness-shaped, PERF-NAIVE (no tiling, no
// shared-memory staging; the implementation session applies the chunk-256
// workgroup recipe and weight staging). NOT device-validated; the host entry
// refuses before any dispatch can reach it (M1/M2 gates, design section 7).

struct ConvParams {
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    out_h: u32,
    out_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    rows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> cp: ConvParams;
// Weights, [co][ci][kh][kw] row-major (the wmat layout the lane compiles).
@group(0) @binding(1) var<storage, read> weight: array<f32>;
// Outgoing-side coefficient plane (n_out, rows), ds32, index = j*rows + row.
@group(0) @binding(2) var<storage, read> coeff_out_side: array<vec2<f32>>;
// Incoming-side coefficient plane (n_in, rows), ds32.
@group(0) @binding(3) var<storage, read_write> coeff_in_side: array<vec2<f32>>;
// E-lane planes, plain f32 (host-widened at egress).
@group(0) @binding(4) var<storage, read> err_out_side: array<f32>;
@group(0) @binding(5) var<storage, read_write> err_in_side: array<f32>;
// RESERVED refinement channel: per-element a-posteriori EFT residual
// magnitudes (implementation-session option). The skeleton zeroes it; the
// binding exists now so the bind-group layout does not change under the M1
// bit-compare when the refinement lands.
@group(0) @binding(6) var<storage, read_write> residual: array<f32>;

@compute @workgroup_size(256)
fn conv_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_in = cp.in_channels * cp.in_h * cp.in_w;
    let total = n_in * cp.rows;
    if idx >= total {
        return;
    }
    let row = idx % cp.rows;
    let pos = idx / cp.rows;
    let ci = pos / (cp.in_h * cp.in_w);
    let iy = (pos / cp.in_w) % cp.in_h;
    let ix = pos % cp.in_w;

    var acc = vec2<f32>(0.0, 0.0); // ds value accumulator
    var errg = 0.0; // |W| * E gather

    for (var co = 0u; co < cp.out_channels; co = co + 1u) {
        for (var kh = 0u; kh < cp.kernel_h; kh = kh + 1u) {
            let ny = iy + cp.pad_h;
            if ny < kh {
                continue;
            }
            let dy = ny - kh;
            if dy % cp.stride_h != 0u {
                continue;
            }
            let oy = dy / cp.stride_h;
            if oy >= cp.out_h {
                continue;
            }
            for (var kw = 0u; kw < cp.kernel_w; kw = kw + 1u) {
                let nx = ix + cp.pad_w;
                if nx < kw {
                    continue;
                }
                let dx = nx - kw;
                if dx % cp.stride_w != 0u {
                    continue;
                }
                let ox = dx / cp.stride_w;
                if ox >= cp.out_w {
                    continue;
                }
                let w = weight[((co * cp.in_channels + ci) * cp.kernel_h + kh) * cp.kernel_w + kw];
                let oidx = ((co * cp.out_h + oy) * cp.out_w + ox) * cp.rows + row;

                // Value: term = L_out * w (DWTimesFP3), acc += term (accurate
                // DWPlusDW) — both theorem-backed, <= 2u^2 / 3u^2 relative per
                // op, so the HOST's a-priori gamma_ds(k) envelope (~2.6e-10
                // relative at k = 4608) is the entire value-lane charge:
                // already ~1e8x below margin scale, no per-tap residual needed
                // for soundness.
                let term = ds_mul_f32(coeff_out_side[oidx], w);
                acc = ds_add(acc, term);

                // E-lane: |W| gather of the upstream elementwise error.
                errg = fma(abs(w), err_out_side[oidx], errg);
            }
        }
    }

    coeff_in_side[idx] = acc;
    err_in_side[idx] = errg;
    // Reserved refinement channel (see the binding comment): the skeleton's
    // value-lane charge is the host-side gamma_ds envelope, so nothing is
    // owed here yet. Zeroed, not left stale.
    residual[idx] = 0.0;
}
