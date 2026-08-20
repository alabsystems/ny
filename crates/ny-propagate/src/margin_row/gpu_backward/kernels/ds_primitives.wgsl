// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Double-single (ds32) EFT primitives for the certified margin-row GPU
// backward (design: docs/GPU_CERTIFIED_LANE_BACKWARD_DESIGN_2026-08-19.md,
// section 4-5). Host twin: ../ds.rs — op-for-op the same sequences; the M1
// moat gate is BIT-IDENTITY between the two on this adapter.
//
// HARDWARE CONTRACT (measured on the GB10 Vulkan adapter,
// ny-gpu ops/double_single_probe.rs, banked 2026-07-23):
//   * fma TwoProduct  `e = fma(a, b, -p)`            — BIT-EXACT (0 ULP).
//   * fma-barrier TwoSum (every Knuth subtraction routed through
//     `fma(-1.0, x, y)`)                             — BIT-EXACT (0 ULP).
//   * plain Knuth TwoSum and Dekker split            — COMPILER-DESTROYED
//     (fast-math folds the algebraically-zero compensation / FMA-contracts
//     the split). NEVER write the plain forms in any kernel that includes
//     this file.
// The driver does NOT canonicalize `fma(-1.0, x, y)` back to a subtraction,
// which is the entire barrier mechanism. Any future toolchain that starts
// doing so is caught by the ONCE-PER-PROCESS on-device self-check
// (../authority.rs), which re-runs the probe lanes at admit time — not only
// by a one-time M1 qualification session.
//
// RISK CONCENTRATION (adversarial review, 2026-08-19): the barrier TwoSum
// stays exact even under a NON-FUSED fma — `fma(-1.0, x, y)` is the single
// rounding of `y - x` under both the fused and the emulated `a*b + c`
// definition — but eft_two_prod's residual is exact ONLY when fma is truly
// fused: a de-fused fma makes `e` identically zero and silently collapses
// the value path to plain f32 (~2^-24) while the host still charges
// u_ds = 2^-44, a ~1e6x UNDERCHARGE (false-VERIFY class). The WGSL spec does
// NOT guarantee fusion ("inherits from e1*e2+e3"), so ALL fma risk
// concentrates on eft_two_prod — which is exactly what the admit-time
// self-check re-validates on the running driver before any authority.
//
// SUBNORMAL CONTRACT (ADMISSION REQUIREMENT, adversarial-review item 3):
// the DenormPreserve word channel is NOT assumed. `DenormPreservePolicy::Auto`
// resolves to the adapter's `passthrough_supported`
// (ny-gpu shader_loading.rs:100, `resolve_denorm_preserve`) — i.e. silently
// OFF on an adapter without support, where FTZ flushes subnormal TwoSum
// residuals to zero and voids the exact-residual identity beyond the charged
// fma-result band (an UNDERCHARGE, false-VERIFY direction). The host
// therefore REFUSES admission unless the device's RESOLVED
// `denorm_preserve_enabled` is ON — enforced in the only constructor of the
// parity proof (../authority.rs, `DeviceParityProof::qualify`). Residual
// caveat that remains even with the channel ON: an fma whose RESULT is
// subnormal may still flush; the host charges a conditional absolute floor
// (2^-126 per fused rounding in the flush band, design section 4.1) — the
// kernels do not need to branch on it.

// A ds32 value is vec2<f32>(hi, lo): the unevaluated sum hi + lo with
// |lo| <= ulp(hi)/2 after renormalization.

// TwoProduct: p = fl(a*b), e exact with a*b = p + e (away from the
// underflow band |p| < 2^-101, where the HOST charges the 2^-126 floor).
fn eft_two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

// Knuth TwoSum, fma-barrier form: s = fl(a+b), t exact with a + b = s + t.
// Each subtraction of the classical sequence is one fma(-1.0, x, y) — the
// same single rounding, opaque to reassociation.
fn eft_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = fma(-1.0, a, s); // s - a
    let sb = fma(-1.0, bb, s); // s - bb
    let da = fma(-1.0, sb, a); // a - (s - bb)
    let db = fma(-1.0, bb, b); // b - bb
    let t = da + db;
    return vec2<f32>(s, t);
}

// Fast two-sum (Dekker), barrier form. Analysis precondition |a| >= |b|;
// used only inside the composite shapes whose published error bounds cover
// this exact sequence (Joldes-Muller-Popescu, ACM TOMS 2017).
fn eft_fast_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = fma(-1.0, a, s); // s - a
    let t = fma(-1.0, bb, b); // b - bb
    return vec2<f32>(s, t);
}

// Renormalize a (possibly overlapping) pair into the ds invariant.
fn ds_renorm(hi: f32, lo: f32) -> vec2<f32> {
    return eft_fast_two_sum(hi, lo);
}

// ds + f32 (DWPlusFP, Algorithm 4). Relative error <= 2u^2, u = 2^-24.
fn ds_add_f32(x: vec2<f32>, b: f32) -> vec2<f32> {
    let st = eft_two_sum(x.x, b);
    return ds_renorm(st.x, st.y + x.y);
}

// ds + ds (ACCURATE DWPlusDW, Algorithm 6). Relative error <= 3u^2/(1-4u),
// unconditional — the sloppy Algorithm 5 is FORBIDDEN here: its error is
// unbounded under hi-part cancellation, which is the CROWN backward regime.
// Host twin: ds.rs ds_add, op-for-op.
fn ds_add(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let s = eft_two_sum(x.x, y.x);
    let t = eft_two_sum(x.y, y.y);
    let c = s.y + t.x;
    let v = eft_fast_two_sum(s.x, c);
    let w = t.y + v.y;
    return ds_renorm(v.x, w);
}

// ds * f32 (DWTimesFP with fma, Algorithm 9). Relative error <= 2u^2.
fn ds_mul_f32(x: vec2<f32>, w: f32) -> vec2<f32> {
    let pe = eft_two_prod(x.x, w);
    return ds_renorm(pe.x, fma(x.y, w, pe.y));
}
