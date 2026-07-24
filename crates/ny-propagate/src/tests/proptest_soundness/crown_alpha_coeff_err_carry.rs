// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Enclosure soundness for the sign-stability certified coefficient-error carry
//! in `ReLULayer::propagate_linear_with_alpha` (task #35, #cgan-coeff-err-fold).
//!
//! The α-linear ReLU backward receives incoming linear bounds whose coefficients
//! may carry a certified per-coefficient error (attached upstream by the sound
//! Conv/ConvTranspose/activation backwards). Historically this function DROPPED
//! that error — the false-proof direction: a downstream backward then reads the
//! composed f32 coefficient as exact and under-counts the true distance to the
//! real coefficient.
//!
//! Ground-truth family. The incoming lower row `Σ la_i·r_i` (`r_i = relu(z_i)`)
//! with certified error `err_la_i` certifies a lower bound on EVERY realization
//! `Σ a*_i·r_i` where the true downstream coefficient `a*_i ∈ [la_i-err_i,
//! la_i+err_i]`; symmetrically the upper row bounds `Σ c*_i·r_i` from above with
//! `c*_i ∈ [ua_i-err_i, ua_i+err_i]`.
//!
//! Property. After the α backward composes the ReLU relaxation and the result is
//! concretized over the pre-activation box (which applies the CARRIED error
//! penalty), the scalar lower/upper bounds must ENCLOSE the entire ground-truth
//! family for all `z` in the box. The error-dropping version can violate this;
//! the sign-stability carry cannot.

use crate::layers::ReLULayer;
use crate::LinearBounds;
use ndarray::{Array1, Array2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

const N: usize = 3;

/// Worst-case (minimizing) realization of `Σ coeff_i·relu(z_i)` over the box,
/// choosing per neuron the box corner that minimizes each independent term.
fn phi_min(coeff: &[f64], l: &[f32], u: &[f32]) -> f64 {
    let mut s = 0.0;
    for i in 0..coeff.len() {
        let r_lo = (l[i].max(0.0)) as f64; // relu(l_i) — monotone lower corner
        let r_hi = (u[i].max(0.0)) as f64; // relu(u_i) — monotone upper corner
        s += if coeff[i] >= 0.0 {
            coeff[i] * r_lo
        } else {
            coeff[i] * r_hi
        };
    }
    s
}

/// Worst-case (maximizing) realization of `Σ coeff_i·relu(z_i)` over the box.
fn phi_max(coeff: &[f64], l: &[f32], u: &[f32]) -> f64 {
    let mut s = 0.0;
    for i in 0..coeff.len() {
        let r_lo = (l[i].max(0.0)) as f64;
        let r_hi = (u[i].max(0.0)) as f64;
        s += if coeff[i] >= 0.0 {
            coeff[i] * r_hi
        } else {
            coeff[i] * r_lo
        };
    }
    s
}

fn build_incoming(la: &[f32], ua: &[f32], ela: &[f32], eua: &[f32]) -> LinearBounds {
    let mut b = LinearBounds::new(
        Array2::from_shape_vec((1, N), la.to_vec()).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, N), ua.to_vec()).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    b.set_coeff_err(
        Array2::from_shape_vec((1, N), ela.to_vec()).unwrap(),
        Array2::from_shape_vec((1, N), eua.to_vec()).unwrap(),
    );
    b
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1500) })]

    /// The concretized composed bounds (WITH the carried error) must enclose the
    /// whole ground-truth coefficient family over the pre-activation box.
    ///
    /// The error ranges deliberately allow `err_i >= |coeff_i|`, so a substantial
    /// fraction of entries are SIGN-AMBIGUOUS (`|a| <= err_a`) and exercise the
    /// hull-cover branch, while the rest exercise the tight sign-stable branch.
    #[ntest::timeout(30000)]
    #[test]
    fn enclosure_alpha_err_carry(
        // Pre-activation box: mix of crossing / active / inactive neurons.
        lbase in prop::collection::vec(-6.0f32..1.5, N),
        width in prop::collection::vec(0.02f32..6.0, N),
        la in prop::collection::vec(-4.0f32..4.0, N),
        ua in prop::collection::vec(-4.0f32..4.0, N),
        ela in prop::collection::vec(0.0f32..5.0, N),
        eua in prop::collection::vec(0.0f32..5.0, N),
        alpha_l in prop::collection::vec(0.0f32..1.0, N),
        alpha_u in prop::collection::vec(0.0f32..1.0, N),
    ) {
        let l: Vec<f32> = lbase.clone();
        let u: Vec<f32> = (0..N).map(|i| lbase[i] + width[i]).collect();

        let incoming = build_incoming(&la, &ua, &ela, &eua);
        let pre = BoundedTensor::new(
            Array1::from(l.clone()).into_dyn(),
            Array1::from(u.clone()).into_dyn(),
        ).unwrap();

        let layer = ReLULayer::new();
        let (result, _gl, _gu) = layer
            .propagate_linear_with_alpha(
                &incoming,
                &pre,
                &Array1::from(alpha_l.clone()),
                Some(&Array1::from(alpha_u.clone())),
            )
            .unwrap();

        // The sign-stability composed error is discharged LOCALLY into the bias
        // over the pre-activation box (task #35 throughput fix), so the output is
        // err-free but its bias is widened OUTWARD to cover the coefficient
        // uncertainty. Enclosure below verifies that widening is sufficient.
        let out = result.concretize_sound(&pre);
        let lb = out.lower()[0] as f64;
        let ub = out.upper()[0] as f64;

        // Worst-case realizations of the ground-truth family over the box.
        let a_min: Vec<f64> = (0..N).map(|i| (la[i] - ela[i]) as f64).collect();
        let c_max: Vec<f64> = (0..N).map(|i| (ua[i] + eua[i]) as f64).collect();
        let true_lower_min = phi_min(&a_min, &l, &u);
        let true_upper_max = phi_max(&c_max, &l, &u);

        let scale = 1.0 + true_lower_min.abs() + true_upper_max.abs()
            + lb.abs() + ub.abs();
        let tol = 1e-3 + 1e-4 * scale;

        prop_assert!(
            lb <= true_lower_min + tol,
            "LOWER enclosure violated: lb={lb} > worst-case truth={true_lower_min} \
             (tol={tol})\n l={l:?} u={u:?} la={la:?} ela={ela:?} alpha_l={alpha_l:?}"
        );
        prop_assert!(
            ub + tol >= true_upper_max,
            "UPPER enclosure violated: ub={ub} < worst-case truth={true_upper_max} \
             (tol={tol})\n l={l:?} u={u:?} ua={ua:?} eua={eua:?} alpha_u={alpha_u:?}"
        );

        // Also probe a grid of concrete z's (enclosure must hold pointwise, not
        // only at the analytic worst corner).
        let grid = |lo: f32, hi: f32| -> Vec<f32> {
            (0..5).map(|k| lo + (hi - lo) * (k as f32) / 4.0).collect()
        };
        let g: Vec<Vec<f32>> = (0..N).map(|i| grid(l[i], u[i])).collect();
        for &z0 in &g[0] {
            for &z1 in &g[1] {
                for &z2 in &g[2] {
                    let z = [z0, z1, z2];
                    let r: Vec<f64> = z.iter().map(|&v| (v.max(0.0)) as f64).collect();
                    let lo_val: f64 = (0..N).map(|i| a_min[i] * r[i]).sum();
                    let hi_val: f64 = (0..N).map(|i| c_max[i] * r[i]).sum();
                    prop_assert!(lb <= lo_val + tol, "pointwise lower z={z:?}");
                    prop_assert!(ub + tol >= hi_val, "pointwise upper z={z:?}");
                }
            }
        }
    }

    /// No incoming error → BYTE-IDENTICAL output to a `None`-err reference, with
    /// no error attached. This guards the "byte-identical where err is zero"
    /// requirement structurally.
    #[ntest::timeout(20000)]
    #[test]
    fn byte_identical_when_no_incoming_err(
        lbase in prop::collection::vec(-6.0f32..1.5, N),
        width in prop::collection::vec(0.02f32..6.0, N),
        la in prop::collection::vec(-4.0f32..4.0, N),
        ua in prop::collection::vec(-4.0f32..4.0, N),
        alpha_l in prop::collection::vec(0.0f32..1.0, N),
        alpha_u in prop::collection::vec(0.0f32..1.0, N),
    ) {
        let l = lbase.clone();
        let u: Vec<f32> = (0..N).map(|i| lbase[i] + width[i]).collect();
        let pre = BoundedTensor::new(
            Array1::from(l).into_dyn(),
            Array1::from(u).into_dyn(),
        ).unwrap();
        let layer = ReLULayer::new();
        let al = Array1::from(alpha_l);
        let au = Array1::from(alpha_u);

        // Reference: no err attached at all.
        let no_err = LinearBounds::new(
            Array2::from_shape_vec((1, N), la.clone()).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, N), ua.clone()).unwrap(),
            Array1::zeros(1),
        ).unwrap();
        let (r_ref, _, _) = layer
            .propagate_linear_with_alpha(&no_err, &pre, &al, Some(&au))
            .unwrap();

        // Same coefficients but an ALL-ZERO certified error attached: the carry
        // branch runs but must produce numerically identical coeffs/bias.
        let zero_err = build_incoming(&la, &ua, &[0.0; N], &[0.0; N]);
        let (r_zero, _, _) = layer
            .propagate_linear_with_alpha(&zero_err, &pre, &al, Some(&au))
            .unwrap();

        prop_assert!(r_ref.lower_a_err().is_none(), "no-err ref must stay err-free");
        for j in 0..1 {
            for i in 0..N {
                prop_assert_eq!(r_ref.lower_a[[j, i]], r_zero.lower_a[[j, i]]);
                prop_assert_eq!(r_ref.upper_a[[j, i]], r_zero.upper_a[[j, i]]);
            }
            prop_assert_eq!(r_ref.lower_b[j], r_zero.lower_b[j]);
            prop_assert_eq!(r_ref.upper_b[j], r_zero.upper_b[j]);
        }
    }
}

/// Focused witness: with a genuinely sign-ambiguous incoming coefficient on a
/// crossing neuron, the carried error strictly loosens the concretized bound
/// below the error-free (dropped) value — and the dropped value would be
/// UNSOUND for a realization inside the certified error interval.
#[test]
fn carry_prevents_false_proof_witness() {
    let layer = ReLULayer::new();
    // One crossing neuron: pre-activation [-2, 2]. Stored coeff la = 0.1 but the
    // certified error is 0.5, so the true coefficient could be as low as -0.4.
    let la = [0.1f32];
    let ua = [0.1f32];
    let ela = [0.5f32];
    let eua = [0.5f32];
    let l = [-2.0f32];
    let u = [2.0f32];

    let mut incoming = LinearBounds::new(
        Array2::from_shape_vec((1, 1), la.to_vec()).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), ua.to_vec()).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    incoming.set_coeff_err(
        Array2::from_shape_vec((1, 1), ela.to_vec()).unwrap(),
        Array2::from_shape_vec((1, 1), eua.to_vec()).unwrap(),
    );
    let pre = BoundedTensor::new(
        Array1::from(l.to_vec()).into_dyn(),
        Array1::from(u.to_vec()).into_dyn(),
    )
    .unwrap();
    let alpha = Array1::from(vec![0.5f32]);

    let (with_err, _, _) = layer
        .propagate_linear_with_alpha(&incoming, &pre, &alpha, None)
        .unwrap();
    // The composed error is discharged into the bias over the pre-activation box,
    // so the output is err-free but its lower bias is pushed DOWN to cover the
    // coefficient uncertainty.
    let lb_carry = with_err.concretize_sound(&pre).lower()[0] as f64;

    // Reference WITHOUT the certified error (the historical drop).
    let no_err = LinearBounds::new(
        Array2::from_shape_vec((1, 1), la.to_vec()).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), ua.to_vec()).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    let (dropped, _, _) = layer
        .propagate_linear_with_alpha(&no_err, &pre, &alpha, None)
        .unwrap();
    let lb_drop = dropped.concretize_sound(&pre).lower()[0] as f64;

    // The carry must be a strictly looser (smaller) lower bound.
    assert!(
        lb_carry < lb_drop,
        "carry lb {lb_carry} should be below dropped lb {lb_drop}"
    );

    // The dropped bound is UNSOUND: for the true coefficient a* = la - err = -0.4
    // and z = u = 2 (relu = 2), the objective is -0.4 * 2 = -0.8, which lies
    // BELOW the dropped lower bound (false proof). The carry encloses it.
    let true_val = (la[0] - ela[0]) as f64 * u[0].max(0.0) as f64; // -0.8
    assert!(
        lb_drop > true_val + 1e-6,
        "dropped bound {lb_drop} should exceed the true value {true_val} (that is the false proof it enables)"
    );
    assert!(
        lb_carry <= true_val + 1e-6,
        "carry bound {lb_carry} must enclose the true value {true_val}"
    );
}
