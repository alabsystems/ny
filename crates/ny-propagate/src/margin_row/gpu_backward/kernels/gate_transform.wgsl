// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// ReLU gate backward transform for the certified margin-row GPU backward
// (design section 5, kernel 2). Concatenated AFTER ds_primitives.wgsl at
// pipeline build; uses ds_mul_f32 from there.
//
// Semantics (re-derived 2026-08-19 against the CPU gate arm,
// engine.rs:653-735, and gpu_seam's Activation mapping): with vp = max(v, 0)
// and vn = v - vp, the LOWER lane maps v -> vp*alpha + vn*s and accumulates
// the intercept against the NEGATIVE part (`part = v.min(0.0)`,
// engine.rs:667); the UPPER lane maps v -> vp*s + vn*alpha and accumulates
// the intercept against the POSITIVE part (`part = v.max(0.0)`). For a
// scalar v exactly one part is nonzero, so per element this is: select the
// slope AND the intercept on the sign of v. The intercept SIDE is therefore
// sign-of-v dependent and differs between lanes — a single intercept slot
// cannot express the upper lane (the original skeleton hard-coded the v < 0
// side, which silently DROPPED the upper lane's nonnegative c*vp bias mass:
// an upper bound too small, i.e. the unsound direction — caught by the
// adversarial-review re-derivation). The gate therefore carries an intercept
// PAIR and the host bakes the lane direction into the full vec4:
//   LOWER: (slope for v>=0, slope for v<0, icept for v>=0, icept for v<0)
//        = (alpha, s, 0, c)
//   UPPER: (s, alpha, c, 0)
// making the kernel genuinely direction-agnostic — one more thing that
// cannot drift between lanes.
//
// Certified error (design section 4.2, "gate transform" row, as corrected by
// adversarial-review item 4): the value multiply runs in ds (DWTimesFP3), a
// ds COMPOSITION whose O(u_ds) residue the host envelopes with one op of
// U_DS; the intercept partial c*v is the SAME ds composition (ds_mul_f32 —
// the fma fold of c*v.lo and the renorm each commit one rounding, so it is
// NOT an EFT identity) and is charged the SAME one-op U_DS envelope on the
// intercept-partial plane. The E-lane contracts by
// m = max(|slope_pos|, |slope_neg|) <= 1 exactly as the CPU lane's `ms`
// does. A separate ds tree-reduce pass (an implementation-session kernel)
// folds partials into the per-row bias with its own residual egress.
// Nothing in this kernel rounds without the residue being either carried in
// a ds pair or enveloped host-side.
//
// STATUS: skeleton delivery — structurally complete, NOT device-validated;
// the module's host entry refuses before any dispatch can reach it (M1/M2
// gates, design section 7).

struct GateParams {
    // Neurons in this layer (n) and pass rows (r): planes are (n, r)
    // row-major, index = j * rows + row.
    neurons: u32,
    rows: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: GateParams;
// Gate vec4, one per neuron: x = slope for v >= 0, y = slope for v < 0,
// z = intercept for v >= 0, w = intercept for v < 0 (exactly one of z/w is
// zero per lane — see the header derivation; vec4 also keeps the 16-byte
// stride explicit: array<vec3<f32>> has 16-byte stride anyway and a packed
// host upload would silently misalign — the exact class of silent-no-op the
// >8-binding trap already cost a debug cycle upstream). Host-downcast with
// the outward_gate_f32 intercept repair (gpu_seam.rs) so the f32 gate still
// encloses the real ReLU on [l, u].
@group(0) @binding(1) var<storage, read> gate: array<vec4<f32>>;
// Incoming coefficient plane, ds32.
@group(0) @binding(2) var<storage, read> coeff_in: array<vec2<f32>>;
// Outgoing coefficient plane, ds32.
@group(0) @binding(3) var<storage, read_write> coeff_out: array<vec2<f32>>;
// Certified elementwise error plane (nonnegative, plain f32 — host widens
// the whole plane by (1 + gamma_f32(k)) in f64 at egress, design 4.2).
@group(0) @binding(4) var<storage, read> err_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> err_out: array<f32>;
// Per-element ds intercept partials `c * v`, consumed by the bias
// tree-reduce pass.
@group(0) @binding(6) var<storage, read_write> bias_partial: array<vec2<f32>>;

// Workgroup size 256: the measured-moat chunk-256 recipe (margin-row
// twin-wall lane memory); one thread per (neuron, row) element.
@compute @workgroup_size(256)
fn gate_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.neurons * params.rows;
    if idx >= total {
        return;
    }
    let j = idx / params.rows;
    let g = gate[j];

    let v = coeff_in[idx];
    // Slope selection on the sign of the incoming coefficient's VALUE. The
    // ds value's sign is the sign of hi except when hi == 0 (then lo, with
    // |lo| <= ulp(hi)/2 = 0, is also 0) — so hi's sign bit decides exactly.
    let nonneg = v.x >= 0.0;
    let slope = select(g.y, g.x, nonneg);
    let icept = select(g.w, g.z, nonneg);

    // Value: ds * f32, EFT-exact up to the ds O(u^2) residue which the host
    // envelopes with U_DS (design 4.2 last row).
    coeff_out[idx] = ds_mul_f32(v, slope);

    // E-lane: contraction by m <= 1 (exact rule from the CPU lane) plus the
    // multiply's own representable magnitude growth. Plain f32; the host
    // widens the plane on egress, so no directed rounding is needed here.
    let m = max(abs(g.x), abs(g.y));
    err_out[idx] = err_in[idx] * m;

    // Intercept partial: c * v via the SHARED ds multiply (bit-comparable
    // with the host twin's ds_mul_f32), written to the partial plane for the
    // reduce pass. This is a ds COMPOSITION, not an EFT identity — the host
    // charges one op of U_DS on this plane (design 4.2, review item 4).
    // icept is 0 on the pass-through side, so the partial vanishes there.
    bias_partial[idx] = ds_mul_f32(v, icept);
}
