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
// Certified error (design section 4.2 "conv backward" row, tightened in the
// error-charge session): the WEIGHTS ride as ds32 PAIRS split from the
// engine's own f64 parameters (hi = fl32(w), lo = fl32(w - hi); residual
// <= 2^-48 relative in the NORMAL band — transaction.rs DS_SPLIT_REL; the
// f32-subnormal band rides the host split-floor channel of the verify-fix,
// transaction.rs ds_split_excess), so the u32 downcast ball
// (2^-24-relative, the term MEASURED to dominate the published radius ~5e6x
// over the CPU pass) never exists on this lane. The VALUE lane accumulates in
// ds32 through the theorem-backed ds ops (ds_mul = one-fma DWTimesDW2,
// xl*yl-dropping, published <= 6u^2 — NOT the xl*yl-carrying Alg-12's 5u^2;
// accurate ds_add <= 3u^2/(1-4u)), so its entire charge is the a-priori
// per-tap envelope gamma_ds_dot(k) at 10u^2/tap — f64-grade, the entire
// departure from the verdict-breaking f32 Higham charge (design section 2.1).
// That charge RIDES THE MASS, so it is gathered here (the CPU's combined_err
// shape, engine.rs conv arm): per source element the E term is
//   e_src = err + g_src * (|L| + err),
//   g_src = up(gamma_ds_dot(k_bwd + 2) (+) rel)          [host uniform]
// where rel composes the engine's own FOLD ball (cv.weight_rel_err — the
// model error the CPU arm charges too) with the ds-split residual
// (transaction.rs ds_weight_rel — the CertifiedWeightError R/(1-R) shape with
// the split unit in place of u32). The kernel gathers
// E_in = sum (|w.hi| + |w.lo|) * e_src in plain f32, host-widened by
// (1 + gamma_f32(k)) in f64 at egress. The `residual` binding is RESERVED for
// an optional a-posteriori refinement (per-tap EFT residual capture); it is
// zeroed, not left stale.
//
// UNDERFLOW: products in the 2^-101 band void the TwoProduct exactness
// theorem; rather than branch per tap, the HOST charges the conditional
// 2^-126-per-tap floor for the whole dispatch (design 4.1) — absolute,
// margin-invisible, and sound by domination. The word channel itself is an
// ADMISSION requirement, not an assumption: see the SUBNORMAL CONTRACT in
// ds_primitives.wgsl and ../authority.rs (review item 3).
//
// STATUS: dispatched by the L-lane transaction (transaction.rs); one thread
// per (input position, row), chunk-256 workgroups; dispatch cap
// width * R <= 65535 * 256 (transaction.rs MAX_WORKGROUPS_X, fix A5).
//
// B3 WEIGHT STAGING (perf session; MEMORY ACCESS ONLY): threads of one
// workgroup are 256 consecutive idx values = consecutive input positions x
// rows, which OVERWHELMINGLY share one input channel ci — and every
// position in a channel gathers from the SAME (co, kh, kw) weight slice
// W[.., ci, .., ..]. The kernel therefore stages that shared slice in
// workgroup memory in co-CHUNKS (up to CONV_STAGE_TAPS ds pairs at a time,
// cooperative load, one barrier per chunk) and each thread reads its taps
// from the stage when its ci matches the workgroup's shared ci, falling
// back to the identical global-memory value otherwise (a workgroup
// straddling a channel boundary, or khw > CONV_STAGE_TAPS, keeps
// correctness with partial/no reuse). PER-ELEMENT ARITHMETIC ORDER IS
// UNCHANGED: the co-chunk loop walks co ascending exactly as the flat co
// loop did, kh/kw inside, and the staged value is the same 8 bytes the
// global read would return — so device-vs-host BIT IDENTITY against the
// UNCHANGED host twin (tests.rs twin::conv_backward) is preserved and
// re-pinned. Barriers sit only in workgroup-uniform control flow (chunk
// loop bounds come from the uniform block; per-thread `continue`s are
// inside the barrier-free body).

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
    // gamma_ds_dot(k_bwd + 2) composed with the fold+split weight ball,
    // f64->f32 rounded UP on host (the combined_err `g` of the CPU conv arm,
    // ds units).
    g_src: f32,
}

@group(0) @binding(0) var<uniform> cp: ConvParams;
// Weights as ds32 PAIRS, [co][ci][kh][kw] row-major (the wmat layout the lane
// compiles), split on host from the engine's f64 parameters.
@group(0) @binding(1) var<storage, read> weight: array<vec2<f32>>;
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

// B3 staging tile: up to this many ds pairs of the shared-ci weight slice
// held in workgroup memory per co-chunk (8 KiB — within every adapter's
// 16 KiB minimum workgroup-storage limit alongside nothing else).
const CONV_STAGE_TAPS: u32 = 1024u;

var<workgroup> conv_stage: array<vec2<f32>, 1024>;

@compute @workgroup_size(256)
fn conv_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let idx = gid.x;
    let n_in = cp.in_channels * cp.in_h * cp.in_w;
    let total = n_in * cp.rows;
    if total == 0u {
        return; // uniform: whole grid exits before any barrier
    }
    // No early per-thread return (barriers below need uniform control flow):
    // inactive tail threads decode a clamped, valid index and never write.
    let is_live = idx < total;
    let safe_idx = min(idx, total - 1u);
    let row = safe_idx % cp.rows;
    let pos = safe_idx / cp.rows;
    let ci = pos / (cp.in_h * cp.in_w);
    let iy = (pos / cp.in_w) % cp.in_h;
    let ix = pos % cp.in_w;

    // The workgroup's shared input channel = the FIRST thread's (uniform).
    let base_pos = min(wid.x * 256u, total - 1u) / cp.rows;
    let ci_shared = base_pos / (cp.in_h * cp.in_w);
    let khw = cp.kernel_h * cp.kernel_w;
    // co-chunk width; 0 disables staging (kernel slice too large to tile).
    var cos_per_chunk = 0u;
    if khw > 0u && khw <= CONV_STAGE_TAPS {
        cos_per_chunk = min(CONV_STAGE_TAPS / khw, cp.out_channels);
    }

    var acc = vec2<f32>(0.0, 0.0); // ds value accumulator
    var errg = 0.0; // |W| * E gather

    // Uniform co-chunk loop (bounds from the uniform block only).
    var c0 = 0u;
    loop {
        if c0 >= cp.out_channels {
            break;
        }
        var c1 = cp.out_channels;
        if cos_per_chunk > 0u {
            c1 = min(c0 + cos_per_chunk, cp.out_channels);
        }
        // Cooperative stage of W[c0..c1, ci_shared, :, :] (no barrier inside;
        // the load loop bounds are uniform, lanes stride the taps).
        if cos_per_chunk > 0u {
            let taps = (c1 - c0) * khw;
            var t = lid.x;
            loop {
                if t >= taps {
                    break;
                }
                let sco = c0 + t / khw;
                let sk = t % khw;
                let skh = sk / cp.kernel_w;
                let skw = sk % cp.kernel_w;
                conv_stage[t] =
                    weight[((sco * cp.in_channels + ci_shared) * cp.kernel_h + skh)
                        * cp.kernel_w + skw];
                t = t + 256u;
            }
        }
        workgroupBarrier();

        if is_live {
            // SAME tap order as the pre-B3 flat loop: co ascending across
            // chunks, kh/kw inside — bit identity with the unchanged twin.
            for (var co = c0; co < c1; co = co + 1u) {
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
                        // Staged read when this thread shares the workgroup's
                        // channel; identical bytes from global otherwise.
                        var w = vec2<f32>(0.0, 0.0);
                        if cos_per_chunk > 0u && ci == ci_shared {
                            w = conv_stage[(co - c0) * khw + kh * cp.kernel_w + kw];
                        } else {
                            w = weight[((co * cp.in_channels + ci) * cp.kernel_h + kh)
                                * cp.kernel_w + kw];
                        }
                        let oidx = ((co * cp.out_h + oy) * cp.out_w + ox) * cp.rows + row;

                        // Value: term = L_out * w (DWTimesDW2 <= 6u^2), acc +=
                        // term (accurate DWPlusDW <= 3u^2/(1-4u)) — both
                        // theorem-backed, so the HOST's a-priori gamma_ds_dot(k)
                        // per-tap envelope is the entire value-lane charge:
                        // f64-grade, no per-tap residual needed for soundness.
                        let src = coeff_out_side[oidx];
                        let term = ds_mul(src, w);
                        acc = ds_add(acc, term);

                        // E-lane: |W| gather (|w.hi| + |w.lo| >= |w|) of the
                        // combined source error e_src = err + g_src*(|L| + err)
                        // — the value fold's ds charge and the fold+split
                        // weight ball riding the |W||L| mass they belong to
                        // (header derivation).
                        let e = err_out_side[oidx];
                        let e_src = fma(cp.g_src, ds_abs_bound(src) + e, e);
                        errg = fma(ds_abs_bound(w), e_src, errg);
                    }
                }
            }
        }
        // Protect the stage from the next chunk's overwrite.
        workgroupBarrier();
        c0 = c1;
    }

    if is_live {
        coeff_in_side[idx] = acc;
        err_in_side[idx] = errg;
        // Reserved refinement channel (see the binding comment): the
        // skeleton's value-lane charge is the host-side gamma_ds envelope,
        // so nothing is owed here yet. Zeroed, not left stale.
        residual[idx] = 0.0;
    }
}
