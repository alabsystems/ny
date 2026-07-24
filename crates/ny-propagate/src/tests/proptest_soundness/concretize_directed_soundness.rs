// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Directed-rounding concretization soundness (#concretize-soundness-hardening).
//!
//! Closes the latent verdict-path gaps a float-op audit flagged:
//!
//! 1. `LinearBounds::concretize_sound` — the directed-outward f64→f32 cast that the
//!    sole former production caller (`forward_linear.rs::concretize_to_node_shape`)
//!    was routed to. Property: for a random `LinearBounds` and input box, the
//!    concretized f32 bound ENCLOSES the true f64 linear-form range (computed exactly
//!    via the positive/negative coefficient split) with ZERO violations.
//!
//! 2. `interval_mul_for_bounds` — the 4-corner product now rounds each endpoint
//!    OUTWARD. Property: for any random interval × interval, the returned interval
//!    encloses every real product `a·b` of the corners with ZERO violations.
//!
//! These are the now-routed/now-directed paths; the plain `concretize` (round-to-
//! nearest) is intentionally retained for non-verdict use and is NOT asserted sound.

use crate::bounds::interval_mul_for_bounds;
use crate::LinearBounds;
use ndarray::{Array1, Array2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// True f64 lower/upper of the linear form `a·x + b` over the box `x ∈ [xl, xu]`,
/// extremized by the sign of each coefficient (positive coeff → x at the matching
/// box endpoint). All arithmetic in f64 so it is the reference the f32 concretize
/// must enclose.
fn true_linear_range_f64(
    a: &Array2<f32>,
    b: &Array1<f32>,
    xl: &[f32],
    xu: &[f32],
    minimize: bool,
) -> Vec<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut out = vec![0.0_f64; m];
    for i in 0..m {
        let mut acc = b[i] as f64;
        for j in 0..n {
            let c = a[[i, j]] as f64;
            let lo = xl[j] as f64;
            let hi = xu[j] as f64;
            // To MINIMIZE a·x: positive coeff picks lo, negative picks hi.
            // To MAXIMIZE: positive coeff picks hi, negative picks lo.
            let pick = if minimize {
                if c >= 0.0 {
                    lo
                } else {
                    hi
                }
            } else if c >= 0.0 {
                hi
            } else {
                lo
            };
            acc += c * pick;
        }
        out[i] = acc;
    }
    out
}

/// Strategy: a coefficient in a bounded, finite range.
fn coeff() -> impl Strategy<Value = f32> {
    -8.0f32..8.0f32
}

/// Strategy: an input box (lo, hi) with lo <= hi, magnitudes bounded.
fn box1d() -> impl Strategy<Value = (f32, f32)> {
    (-8.0f32..8.0f32, -8.0f32..8.0f32).prop_map(|(a, b)| (a.min(b), a.max(b)))
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(800) })]

    /// `LinearBounds::concretize_sound` ENCLOSES the true f64 linear-form range.
    ///
    /// The lower bound uses `(lower_a, lower_b)` and must be <= the true minimum of
    /// that lower linear form over the box; the upper bound uses `(upper_a, upper_b)`
    /// and must be >= the true maximum of that upper linear form. ZERO violations.
    #[ntest::timeout(20000)]
    #[test]
    fn concretize_sound_encloses_true_f64_range_3x3(
        la in proptest::collection::vec(coeff(), 9),
        ua in proptest::collection::vec(coeff(), 9),
        lb in proptest::collection::vec(coeff(), 3),
        ub in proptest::collection::vec(coeff(), 3),
        boxes in proptest::collection::vec(box1d(), 3),
    ) {
        let lower_a = Array2::from_shape_vec((3, 3), la).unwrap();
        let upper_a = Array2::from_shape_vec((3, 3), ua).unwrap();
        let lower_b = Array1::from_vec(lb);
        let upper_b = Array1::from_vec(ub);

        let bounds = LinearBounds::new(
            lower_a.clone(),
            lower_b.clone(),
            upper_a.clone(),
            upper_b.clone(),
        ).unwrap();

        let xl: Vec<f32> = boxes.iter().map(|&(l, _)| l).collect();
        let xu: Vec<f32> = boxes.iter().map(|&(_, u)| u).collect();
        let input = BoundedTensor::new(
            Array1::from_vec(xl.clone()).into_dyn(),
            Array1::from_vec(xu.clone()).into_dyn(),
        ).unwrap();

        let conc = bounds.concretize_sound(&input);

        // True f64 min of the LOWER linear form and max of the UPPER linear form.
        let true_lower_min = true_linear_range_f64(&lower_a, &lower_b, &xl, &xu, true);
        let true_upper_max = true_linear_range_f64(&upper_a, &upper_b, &xl, &xu, false);

        for i in 0..3 {
            let cl = conc.lower()[[i]] as f64;
            let cu = conc.upper()[[i]] as f64;
            prop_assert!(
                cl <= true_lower_min[i],
                "concretize_sound lower[{i}]={cl} NOT <= true f64 min {} \
                 (UNSOUND: bound is optimistically high)",
                true_lower_min[i]
            );
            prop_assert!(
                cu >= true_upper_max[i],
                "concretize_sound upper[{i}]={cu} NOT >= true f64 max {} \
                 (UNSOUND: bound is optimistically low)",
                true_upper_max[i]
            );
        }
    }

    /// `concretize_sound` is at least as wide as the plain `concretize`: the directed
    /// path never produces a TIGHTER bound than the round-to-nearest path (it only
    /// widens outward). This is the routing-safety invariant for path 1.
    #[ntest::timeout(20000)]
    #[test]
    fn concretize_sound_widens_plain_concretize(
        la in proptest::collection::vec(coeff(), 4),
        ua in proptest::collection::vec(coeff(), 4),
        lb in proptest::collection::vec(coeff(), 2),
        ub in proptest::collection::vec(coeff(), 2),
        boxes in proptest::collection::vec(box1d(), 2),
    ) {
        let lower_a = Array2::from_shape_vec((2, 2), la).unwrap();
        let upper_a = Array2::from_shape_vec((2, 2), ua).unwrap();
        let lower_b = Array1::from_vec(lb);
        let upper_b = Array1::from_vec(ub);

        let bounds = LinearBounds::new(lower_a, lower_b, upper_a, upper_b).unwrap();

        let xl: Vec<f32> = boxes.iter().map(|&(l, _)| l).collect();
        let xu: Vec<f32> = boxes.iter().map(|&(_, u)| u).collect();
        let input = BoundedTensor::new(
            Array1::from_vec(xl).into_dyn(),
            Array1::from_vec(xu).into_dyn(),
        ).unwrap();

        let plain = bounds.concretize(&input);
        let sound = bounds.concretize_sound(&input);

        for i in 0..2 {
            prop_assert!(
                sound.lower()[[i]] <= plain.lower()[[i]],
                "sound lower[{i}]={} must be <= plain lower[{i}]={}",
                sound.lower()[[i]], plain.lower()[[i]]
            );
            prop_assert!(
                sound.upper()[[i]] >= plain.upper()[[i]],
                "sound upper[{i}]={} must be >= plain upper[{i}]={}",
                sound.upper()[[i]], plain.upper()[[i]]
            );
        }
    }

    /// `interval_mul_for_bounds` ENCLOSES every real product of the two intervals.
    /// The true product range of `[a_l,a_u]·[b_l,b_u]` is the min/max of the four
    /// corner products computed in f64 (exact for these magnitudes). The directed
    /// f32 result must enclose it with ZERO violations.
    #[ntest::timeout(20000)]
    #[test]
    fn interval_mul_for_bounds_encloses_true_product(
        (a_l, a_u) in box1d(),
        (b_l, b_u) in box1d(),
    ) {
        let (lo, hi) = interval_mul_for_bounds(a_l, a_u, b_l, b_u);

        // True corner products in f64 (f32×f32 promotes exactly to f64).
        let corners = [
            (a_l as f64) * (b_l as f64),
            (a_l as f64) * (b_u as f64),
            (a_u as f64) * (b_l as f64),
            (a_u as f64) * (b_u as f64),
        ];
        let true_min = corners.iter().cloned().fold(f64::INFINITY, f64::min);
        let true_max = corners.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        prop_assert!(
            (lo as f64) <= true_min,
            "interval_mul lower {lo} NOT <= true min {true_min} (UNSOUND inward bias)"
        );
        prop_assert!(
            (hi as f64) >= true_max,
            "interval_mul upper {hi} NOT >= true max {true_max} (UNSOUND inward bias)"
        );
        prop_assert!(lo <= hi, "interval_mul produced inverted interval [{lo}, {hi}]");
    }
}
