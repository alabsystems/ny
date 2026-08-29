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
// ds COMPOSITION whose O(u_ds) residue is charged IN-KERNEL as one op of
// DS_UNIT on the output magnitude (err_out); the intercept partial c*v is
// the SAME ds composition (ds_mul_f32 — the fma fold of c*v.lo and the renorm
// each commit one rounding, so it is NOT an EFT identity) and is charged the
// SAME one-op DS_UNIT envelope, PLUS the upstream carry (E * |c|, the CPU's
// (er + gam_b*mag)*cj shape with the gam_b half charged by the reduce pass's
// g_reduce on the partial mass), on the partial_err plane. The E-lane
// contracts by m = max(|slope_pos|, |slope_neg|) exactly as the CPU lane's
// `ms` does. The bias_reduce pass folds (bias_partial, partial_err) into the
// per-row bias accumulators. The f32 roundings of the E-side expressions are
// host-widened at egress; nothing here rounds a term the host does not widen.

struct GateParams {
    // Neurons in this layer (n) and pass rows (r): planes are (n, r)
    // row-major, index = j * rows + row.
    neurons: u32,
    rows: u32,
    // B4: number of per-(row, neuron) gate OVERRIDE records in `exc`.
    // Always 0 for the `gate_transform` entry (which does not bind `exc`);
    // the host emits `gate_transform_exc` only when it is nonzero, so a pass
    // without exceptions dispatches byte-identically to the pre-B4 build.
    exc_count: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: GateParams;
// Gate ds-PAIR, TWO vec4s per neuron (hi quad at 2j, lo quad at 2j + 1):
// x = slope for v >= 0, y = slope for v < 0, z = intercept for v >= 0,
// w = intercept for v < 0 (exactly one of z/w is zero per lane — see the
// header derivation). The gates ride as ds SPLITS of the CPU lane's f64
// gates (hi = fl32(g), lo = fl32(g - hi); normal-band residual <= 2^-48
// relative — the f32-subnormal band rides the host split-floor channel,
// charged below), so the device certifies the SAME relaxation the CPU pass
// uses — the precondition of the M2 coefficient-overlap assertion. vec4
// keeps the 16-byte stride explicit (a packed host upload would silently
// misalign — the exact class of silent-no-op the >8-binding trap already
// cost a debug cycle upstream).
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
// Per-element error of the intercept partials: E*|c| upstream carry plus the
// partial's own one-op DS_UNIT charge. Consumed by the same reduce pass.
@group(0) @binding(7) var<storage, read_write> partial_err: array<f32>;

// B4 EXCEPTION TABLE (`gate_transform_exc` only): three vec4<f32> per record.
//   slot 0: bitcast<u32> (neuron, row_start, row_end, 0)
//   slot 1: the hi quad of the OVERRIDE gate, same direction-baked layout
//   slot 2: the lo quad
// One buffer rather than three keeps the exception entry at 8 storage
// bindings — the wgpu default per-stage limit. Exceeding it fails the bind
// group ASYNC: the dispatch silently no-ops and the readback returns zeros,
// which is exactly how an earlier taint probe "measured" words that never
// reached a shader.
//
// PER-NEURON RECORDS, PER-ELEMENT MATCH (D6 clarification, impl-v9): the host
// refuses a duplicated (neuron, row) ELEMENT, not a duplicated row — two
// records on the SAME row at DIFFERENT neurons are legal and each is applied
// by its own neuron's thread (the match below tests `j == neuron` too). Within
// one neuron the emitted row blocks are disjoint by construction (the rows are
// sorted and unique), so at most one record can match any element and the scan
// order is immaterial.
//
// THE HEADER IS BIT-CAST SUBNORMAL f32 — IT DEPENDS ON THE DenormPreserve
// ADMISSION (FIX D7, impl-v9). This binding is `array<vec4<f32>>`, so every
// header word with 0 < w < 2^23 — every realistic neuron/row index — is an
// f32 SUBNORMAL bit pattern. An adapter that flushed subnormals ON LOAD would
// read all three header lanes as 0, every record would target
// (neuron 0, rows [0, 0)), NO override would be applied on the device while
// the CPU arm applies it, and the two relaxations would differ silently on
// exactly the rows banked as child bounds. This is structurally prevented
// upstream — `DeviceParityProof::qualify` refuses unless the device's RESOLVED
// DenormPreserve word channel is ON (authority.rs:81-86) and only that proof
// mints the VerdictAuthority a transaction requires — but the dependency is
// invisible from here, so it is stated at both ends and pinned by
// `tests::exception_header_words_are_subnormals_that_depend_on_denorm_preserve`.
@group(0) @binding(8) var<storage, read> exc: array<vec4<f32>>;

// The whole per-element body, factored so the base entry and the B4
// exception entry execute the IDENTICAL instruction sequence on the quad
// they resolved — the exception path can therefore only differ by WHICH quad
// it passes, never by how the quad is applied or charged.
fn gate_apply(idx: u32, ghi: vec4<f32>, glo: vec4<f32>) {
    let v = coeff_in[idx];
    // Slope selection on the sign of the incoming coefficient's VALUE. The
    // ds value's sign is the sign of hi except when hi == 0 (then lo, with
    // |lo| <= ulp(hi)/2 = 0, is also 0) — so hi's sign bit decides exactly.
    let nonneg = v.x >= 0.0;
    let slope = vec2<f32>(select(ghi.y, ghi.x, nonneg), select(glo.y, glo.x, nonneg));
    let icept = vec2<f32>(select(ghi.w, ghi.z, nonneg), select(glo.w, glo.z, nonneg));

    // Value: ds * ds (one-fma DWTimesDW2 — the xl*yl-dropping shape, published
    // bound <= 6u^2; the 5u^2 sometimes quoted is the xl*yl-CARRYING Alg-12's,
    // a different sequence). Charged below at 2*DS_UNIT on the output
    // magnitude: one DS_UNIT (16u^2) dominates the multiply's own residue,
    // the second the gate-split representation ball — together they keep the
    // pass an enclosure of the CPU lane's OWN f64 relaxation (M2
    // precondition). SPLIT-BALL COVERAGE (verify-fix, 2026-08-20): the gates
    // are split HOST-side, and DS_UNIT * |o| covers the split residual only
    // in the NORMAL band (>= 2^-45 * pair-mass * |v| away from the o-flush
    // corner); the f32-subnormal band's ABSOLUTE residue is charged exactly
    // by the host split-floor channel (transaction.rs ds_split_excess with
    // DS_GATE_COVER), and the o-flush corner (|slope*v| < 2^-126) is
    // dominated by the per-element egress flush floor. The kernel itself
    // never splits, so no device-side band needs bounding here.
    let o = ds_mul(v, slope);
    coeff_out[idx] = o;

    // E-lane: contraction by m (an upper bound of the applied |slope|, the
    // CPU lane's `ms` rule) plus the value charge above. Plain f32; the host
    // widens the plane on egress, so no directed rounding is needed here.
    let m = max(ds_abs_bound(vec2<f32>(ghi.x, glo.x)), ds_abs_bound(vec2<f32>(ghi.y, glo.y)));
    err_out[idx] = fma(err_in[idx], m, 2.0 * DS_UNIT * ds_abs_bound(o));

    // Intercept partial: c * v via the SHARED ds multiply (bit-comparable
    // with the host twin's ds_mul), written to the partial plane for the
    // reduce pass. This is a ds COMPOSITION, not an EFT identity (review
    // item 4): its error plane carries the upstream E carry plus the
    // 2*DS_UNIT own/representation charge on the partial's magnitude; the
    // reduce pass adds its own gamma_ds fold charge on the partial mass.
    //
    // The E carry multiplies mi = max over BOTH intercept sides, NOT the
    // selected side: the STORED v's sign can differ from the exact one's
    // (|v_exact| <= E across the kink), and the true partial map
    // v -> v * icept(sign v) has global Lipschitz constant mi. Charging the
    // selected side would charge ZERO on the pass-through side while the
    // exact partial can be as large as |c| * E — an UNDERCHARGE, the
    // false-VERIFY direction (the CPU arm charges |c_j| unconditionally).
    let p = ds_mul(v, icept);
    bias_partial[idx] = p;
    let mi = max(ds_abs_bound(vec2<f32>(ghi.z, glo.z)), ds_abs_bound(vec2<f32>(ghi.w, glo.w)));
    partial_err[idx] = fma(err_in[idx], mi, 2.0 * DS_UNIT * ds_abs_bound(p));
}

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
    gate_apply(idx, gate[2u * j], gate[2u * j + 1u]);
}

// B4 (batched-children scoring): the same transform with per-(row, neuron)
// gate OVERRIDES applied. The override REPLACES the whole direction-baked
// quad — slopes AND intercepts — so `m` and `mi` inside `gate_apply` are
// derived from the OVERRIDDEN quad automatically.
//
// DERIVATION OBLIGATION (ii), the `E*mi` kink term, restated for exceptions:
// `mi = max(|icept+|, |icept-|)` and `m = max(|slope+|, |slope-|)` must be
// the maxima over BOTH sides of the quad the element actually applies. The
// STORED v's sign can differ from the exact one's across the kink
// (|v_exact| <= E there), so the true map `v -> v*icept(sign v)` has global
// Lipschitz constant `mi` — charging the SELECTED side would charge zero on
// the pass-through side while the exact partial can reach |icept|*E. On an
// exception row those maxima must follow the OVERRIDE: an override with
// |c2| > |c_j| that kept the base `mi` would under-charge, the false-VERIFY
// direction. Substituting the whole quad before `gate_apply` is what makes
// that structural rather than a rule someone has to remember.
//
// DERIVATION OBLIGATION (i), the intercept sign select: the record's quad is
// ABSOLUTE, not a delta against the base intercept — this lane's intercept
// partial is produced HERE (`p = ds_mul(v, icept)`), unlike the CPU arm
// whose bias loop already folded the base `c_j` and therefore applies
// `vn*(c2 - c_j)` (engine.rs:766-770). The host bakes the same sign rule as
// the base quad (LOWER `(a2, s2, 0, c2)`, UPPER `(s2, a2, c2, 0)`), so for
// v < 0 the lower lane yields `vn*s2` and `vn*c2` and for v >= 0 `vp*a2` and
// 0 — the CPU element exactly.
@compute @workgroup_size(256)
fn gate_transform_exc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.neurons * params.rows;
    if idx >= total {
        return;
    }
    let j = idx / params.rows;
    let row = idx % params.rows;
    var ghi = gate[2u * j];
    var glo = gate[2u * j + 1u];
    for (var k: u32 = 0u; k < params.exc_count; k = k + 1u) {
        // `hdr`, not `meta`: `meta` is a WGSL RESERVED KEYWORD, and a parse
        // failure here disables the DenormPreserve passthrough for the WHOLE
        // module (the loader falls back to plain WGSL), which makes the lane's
        // admission refuse — fail-closed, but silently dark. Caught on the
        // adapter, not by any host test.
        let hdr = exc[3u * k];
        let neuron = bitcast<u32>(hdr.x);
        let r0 = bitcast<u32>(hdr.y);
        let r1 = bitcast<u32>(hdr.z);
        if j == neuron && row >= r0 && row < r1 {
            ghi = exc[3u * k + 1u];
            glo = exc[3u * k + 2u];
        }
    }
    gate_apply(idx, ghi, glo);
}
