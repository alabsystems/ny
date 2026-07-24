// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared host-side sound-arithmetic constants for the GPU verdict paths.
//!
//! `gamma_k_f32`, `combine_slack_f32`, and `up_f32` were originally private in
//! `ops/crown_backward_sound_resident.rs`. The sound GPU IBP forward
//! (`docs/SOUND_GPU_IBP_PLAN.md` §2.1) needs the same host-side error sizing, so
//! they are hoisted here to ONE `pub(crate)` home — CROWN backward, the sound
//! concretize, and the sound IBP forward all share this single copy instead of
//! carrying divergent duplicates. All three are HOST-side f64 helpers rounded
//! OUTWARD to f32 uniforms; no f64 ever enters a WGSL body.

/// f32 unit roundoff `u = 2^-24`.
const U: f64 = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24

/// γ_k = k·u/(1−k·u) (the f32 dot-product backward-error factor for a length-k
/// reduction), clamped to `2·k·u` past the half-way point.
pub(crate) fn gamma_k_f32(k: usize) -> f32 {
    let ku = (k as f64) * U;
    (if ku < 0.5 { ku / (1.0 - ku) } else { 2.0 * ku }) as f32
}

/// SOUNDNESS slack for the AW-error combine on the f32 error GEMMs.
///
/// The combine reads `s_prod = fl(|A|@|W|)` and `prop = fl(err@|W|)`, both f32
/// dot products over the length-`k` contraction. By Higham each is ≥
/// `(1−γ_k)·(exact sum)`, so the exact sum ≤ `f32_result/(1−γ_k)`. To turn the
/// on-device `(γ_k·s_prod + prop)` (which uses the UNDER-reported f32 products)
/// into an OUTWARD bound on `γ_k·S_exact + prop_exact`, scale by
/// `slack ≥ 1/(1−γ_k)`. We add four ULPs of headroom `(1+u)^4` for the combine's
/// own f32 ops (γ·s multiply, the add, the slack multiply, the +additive) and
/// round the f32 cast UP. `γ_k` here is `gamma_k_f32(k)` (the SAME factor the
/// reductions incur). For the small `k` where the old fixed 1.000001 was already
/// adequate this evaluates to ~1.0000xx as well, but it now SCALES with k so wide
/// contractions are covered.
pub(crate) fn combine_slack_f32(k: usize) -> f32 {
    let g = f64::from(gamma_k_f32(k));
    // 1/(1−γ_k); γ_k<1 always (gamma_k_f32 clamps at 2·k·u and the combine path is
    // only taken for finite layers), but guard the degenerate case defensively.
    let inv = if g < 1.0 { 1.0 / (1.0 - g) } else { 2.0 };
    let headroom = (1.0 + U).powi(4); // 4 combine f32 ops, each ≤ (1+u) growth
    up_f32(inv * headroom)
}

/// SOUNDNESS slack for the EFT residual channel (#eft-err).
///
/// The EFT twin's per-element residual sum `R = Σ|ep| + Σ|es|` is accumulated
/// in f32 over `2k` non-negative terms (plain adds), so by Higham the f32
/// result ≥ `(1−γ_{2k})·R_exact` — recovering the outward bound needs
/// `1/(1−γ_{2k})`. Headroom `(1+u)^6` covers the min-combine's own f32 ops:
/// the `|V−value|` subtraction/abs, the `R+d` add, the `·r_slack` multiply,
/// the `prop·slack` product, the cross add, and the `+flush`. Rounded UP.
pub(crate) fn eft_r_slack_f32(k: usize) -> f32 {
    let g = f64::from(gamma_k_f32(2 * k + 2));
    let inv = if g < 1.0 { 1.0 / (1.0 - g) } else { 2.0 };
    let headroom = (1.0 + U).powi(6);
    up_f32(inv * headroom)
}

/// Round an `f64` UP to `f32` (outward, toward +∞ in magnitude for positive `x`;
/// for negative `x` steps toward the value that is `>= x`).
pub(crate) fn up_f32(x: f64) -> f32 {
    let n = x as f32;
    if n.is_finite() && f64::from(n) < x {
        f32::from_bits(if n > 0.0 {
            n.to_bits() + 1
        } else {
            n.to_bits() - 1
        })
    } else {
        n
    }
}
