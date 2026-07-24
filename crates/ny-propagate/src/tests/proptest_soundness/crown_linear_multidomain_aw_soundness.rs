// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STRICT, ZERO-TOLERANCE soundness tests for the MULTI-DOMAIN GEMM-batched
//! linear CROWN-backward coefficient certificate (#vnncomp-aw-soundness).
//!
//! The bug: `propagate_linear_batched_with_engine`
//! (`linear/crown_batched_multi_domain.rs`) — the path the β-CROWN / BaB verdict
//! kernel calls for EVERY domain at EVERY Linear node
//! (`beta_crown/.../batched/backward_core.rs:139`) — stored the round-to-nearest
//! **f32** `A·W` GEMM coefficient with NO certified coefficient error, while
//! Linear declares `propagates_coeff_err = true`. The dispatcher therefore TRUSTS
//! this path to carry the error, but it dropped it → the concretized β-CROWN/BaB
//! bound is TIGHTER than the proven-sound scalar path = a FALSE PROOF on the BaB
//! verdict path (the catastrophic case in a proof-carrying verifier).
//!
//! These tests assert the CERTIFICATE CONTRACT directly: for every output
//! coefficient `(i, j)`,
//!
//!     certified_err[i, j]  >=  |stored_f32_coeff[i, j] − f64_recompute[i, j]|
//!
//! where `f64_recompute` is the SAME `A·W` contraction accumulated in f64 (exact
//! f32→f64 widening; f32×f32 is exact in f64, so only the f64 sum rounds — a tiny
//! `γ_n^f64·S` residual far below the f32 gap). On the BUGGY code the result
//! carries NO error matrices (treated as 0), so any nonzero gap FAILS; after the
//! fix the certified `γ_n^f32·S + propagated-incoming` covers the gap for every
//! coefficient. A second test concretizes end-to-end against an exact corner.

use crate::{LinearBounds, LinearLayer};
use ndarray::{Array1, Array2};
use ny_core::NaiveCpuGemmEngine;
use ny_tensor::BoundedTensor;

/// Exact-real (f64) linear CROWN-backward coefficient: the SAME `A·W` contraction
/// the layer runs, accumulated in f64. f32→f64 widening is exact and f32×f32 is
/// exact in f64, so only the f64 sum rounds (a tiny residual). Treated as the
/// ground-truth real coefficient for the contract check.
fn aw_f64_ref(a: &Array2<f32>, w: &Array2<f32>) -> Array2<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = w.ncols();
    debug_assert_eq!(w.nrows(), k);
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for kk in 0..k {
            let av = a[[i, kk]] as f64;
            if av == 0.0 {
                continue;
            }
            for j in 0..n {
                out[[i, j]] += av * (w[[kk, j]] as f64);
            }
        }
    }
    out
}

/// A WIDE linear layer + WIDE single-row spec whose f32 `A·W` contraction
/// genuinely rounds (the products and their 64-wide sum exceed f32's 24-bit
/// exact-integer range), so `|stored_f32 − f64_recompute|` is multiple ULP — the
/// regime that exposed the conv/linear `A·W` undercount class.
fn wide_layer_and_spec() -> (LinearLayer, LinearBounds) {
    let h = 64usize; // contraction width (weight_rows) and read-out width
    let out_dim = 64usize;
    // Weight W: h x out_dim, fine-dyadic wide-range values (k/2^13, |k| up to a
    // few thousand) — each exact in one f32, but the 64-wide sum rounds.
    let w = Array2::from_shape_fn((h, out_dim), |(k, j)| {
        let raw = (((k * 131 + j * 977 + 17) % 11999) as i32) - 6000;
        raw as f32 / 8192.0
    });
    let layer = LinearLayer::new(w, None).unwrap();

    // Spec A: out_rows x h. Mixed large +/- coefficients to drive cancellation in
    // the composed contraction.
    let out_rows = 8usize;
    let a = Array2::from_shape_fn((out_rows, h), |(i, k)| {
        let raw = (((i * 911 + k * 613 + 3) % 15999) as i32) - 8000;
        raw as f32 / 8192.0
    });
    let bounds = LinearBounds::from_coefficients(a.clone(), a).unwrap();
    (layer, bounds)
}

/// CONTRACT (multi-domain batched): the certified per-coefficient error returned
/// by `propagate_linear_batched_with_engine` must COVER the actual gap between the
/// stored f32 coefficient and the exact f64 recompute, with ZERO tolerance.
///
/// BUGGY code: returns no error matrices (treated as 0) → FAILS (reports the
/// worst-case undercount). PATCHED: `γ_n^f32·S` covers the gap → PASSES.
#[test]
fn multidomain_aw_cert_covers_f32_error_zero_tolerance() {
    let (layer, spec) = wide_layer_and_spec();
    let engine = NaiveCpuGemmEngine;

    // Two identical domains stacked through the single batched GEMM (exercises the
    // multi-domain stacking, not just a single-domain shortcut).
    let results = layer
        .propagate_linear_batched_with_engine(&[&spec, &spec], &engine)
        .expect("multi-domain batched backward");
    assert_eq!(results.len(), 2);

    let w = &layer.weight;
    let lower_ref = aw_f64_ref(spec.lower_a(), w);
    let upper_ref = aw_f64_ref(spec.upper_a(), w);

    let mut worst_ratio = 0.0f64;
    let mut worst_gap = 0.0f64;
    let mut worst_cert = 0.0f64;
    let mut violations = 0usize;

    for res in &results {
        // The certified error MUST be present (Linear propagates_coeff_err = true).
        let le = res
            .lower_a_err()
            .expect("multi-domain batched must attach lower certified error");
        let ue = res
            .upper_a_err()
            .expect("multi-domain batched must attach upper certified error");
        let nrows = res.lower_a().nrows();
        let ncols = res.lower_a().ncols();
        for i in 0..nrows {
            for j in 0..ncols {
                for (stored, refv, cert) in [
                    (res.lower_a()[[i, j]], lower_ref[[i, j]], le[[i, j]]),
                    (res.upper_a()[[i, j]], upper_ref[[i, j]], ue[[i, j]]),
                ] {
                    let gap = (stored as f64 - refv).abs();
                    let cert = cert as f64;
                    if gap > 0.0 {
                        let ratio = if cert > 0.0 {
                            gap / cert
                        } else {
                            f64::INFINITY
                        };
                        if ratio > worst_ratio {
                            worst_ratio = ratio;
                            worst_gap = gap;
                            worst_cert = cert;
                        }
                    }
                    if cert < gap {
                        violations += 1;
                    }
                }
            }
        }
    }

    assert_eq!(
        violations, 0,
        "UNSOUND multi-domain A·W certificate: {violations} coefficient(s) where certified error \
         < |stored_f32 − f64_recompute|. Worst undercount: gap={worst_gap:e} > cert={worst_cert:e} \
         (ratio {worst_ratio:e}). The batched β-CROWN/BaB verdict path stored the f32 A·W \
         coefficient with no (or too-small) certified error → false-proof risk."
    );
    // Sanity: the wide net actually exercises a real f32 gap (else the test is vacuous).
    assert!(
        worst_gap > 0.0,
        "test net did not produce any f32 A·W rounding gap (test would be vacuous)"
    );
}

/// END-TO-END concretization soundness: a 1×H read-out spec composed through a
/// wide input Linear whose composed x0 coefficient is EXACTLY 0 must NOT
/// over-claim a positive lower bound over a strictly-positive box. Routed through
/// the multi-domain batched path with the certified error consumed by
/// `concretize_sound`.
#[test]
fn multidomain_aw_corner_flip_zero_tolerance() {
    // Reuse the corner-flip vectors from the N-D batched test: Σ_k OW·W0 = 0 in
    // rationals, but the round-to-nearest f32 A·W rounds the x0 coefficient up.
    const OW: [f32; 64] = super::crown_linear_aw_soundness::CORNER_FLIP_OW;
    const W0: [f32; 64] = super::crown_linear_aw_soundness::CORNER_FLIP_W0;
    let h = OW.len();

    // Input layer W1: H x 2, column 0 = W0, column 1 = 0. No bias.
    let w1 = Array2::from_shape_fn((h, 2), |(k, j)| if j == 0 { W0[k] } else { 0.0 });
    let input_linear = LinearLayer::new(w1, None).unwrap();

    // Spec/read-out: a single row = OW (1 x H), no bias.
    let spec_a = Array2::from_shape_fn((1, h), |(_, k)| OW[k]);
    let spec = LinearBounds::from_coefficients(spec_a.clone(), spec_a).unwrap();

    // Compose through the input Linear via the MULTI-DOMAIN batched engine path.
    let engine = NaiveCpuGemmEngine;
    let composed = input_linear
        .propagate_linear_batched_with_engine(&[&spec], &engine)
        .expect("multi-domain batched backward")
        .into_iter()
        .next()
        .unwrap();

    // Concretize over the box x0 ∈ [1, 2], x1 ∈ [1, 2] (strictly positive).
    let box_in = BoundedTensor::new(
        Array1::from_vec(vec![1.0f32, 1.0]).into_dyn(),
        Array1::from_vec(vec![2.0f32, 2.0]).into_dyn(),
    )
    .unwrap();
    let out = composed.concretize_sound(&box_in);
    let l_f32 = out.flatten().lower()[[0]];

    // The exact coefficient for x0 is 0 and there is no bias, so the true minimum
    // over the box is exactly 0. A sound `L_f32` must satisfy `L_f32 <= 0`.
    assert!(
        l_f32 <= 0.0,
        "UNSOUND multi-domain A·W corner flip: concretized lower bound {l_f32} > 0, but the exact \
         x0 coefficient is 0 (Σ OW·W0 = 0) so the true minimum over x0∈[1,2] is 0. The f32 A·W \
         rounded the coefficient positive and the missing certified error let concretize \
         over-claim — a false VERIFIED on the β-CROWN/BaB path."
    );
}
