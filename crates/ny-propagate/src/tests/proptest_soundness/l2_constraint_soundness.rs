// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness proptests for the RMSNorm → Linear L2 (Cauchy–Schwarz) lever.
//!
//! Two obligations, checked over random input boxes by sampling concrete points
//! (interior grid + all corners):
//!
//! 1. **L2 radius is a true bound.** The annotation `‖y − center‖₂ ≤ radius`
//!    attached to the RMSNorm IBP output must hold for the real RMSNorm output
//!    `y = g·x/rms` at every sampled `x` in the box (center = origin here).
//!
//! 2. **Linear tightening stays sound and only tightens.** Feeding the annotated
//!    RMSNorm output into `Linear::propagate_ibp`, every sampled true output
//!    `W·y + b` must lie inside the (Cauchy–Schwarz-intersected) Linear bounds,
//!    and those bounds must be ⊆ the plain box bounds (never wider).
//!
//! These guard the central soundness claim: Cauchy–Schwarz is exact and the
//! intersection only removes infeasible mass.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{layernorm, rms_norm, sample_points, valid_interval};
use crate::layers::common::BoundPropagation;
use crate::layers::linear::LinearLayer;
use crate::layers::normalization::{LayerNormLayer, RmsNormLayer};

/// Build an n-dim input box from per-coordinate (lower, upper) pairs.
fn box_from_pairs(pairs: &[(f32, f32)]) -> BoundedTensor {
    let lower: Vec<f32> = pairs.iter().map(|&(l, _)| l).collect();
    let upper: Vec<f32> = pairs.iter().map(|&(_, u)| u).collect();
    let n = pairs.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper).unwrap(),
    )
    .unwrap()
}

/// Enumerate a grid of sample points: all 2^n corners plus the per-axis
/// midpoint sweep, capped for tractability.
fn grid_samples(pairs: &[(f32, f32)], per_axis: usize) -> Vec<Array1<f32>> {
    let n = pairs.len();
    let mut out = Vec::new();
    // All corners (n is small in these tests).
    for mask in 0..(1u32 << n) {
        let x: Vec<f32> = (0..n)
            .map(|i| {
                if mask & (1 << i) != 0 {
                    pairs[i].1
                } else {
                    pairs[i].0
                }
            })
            .collect();
        out.push(Array1::from_vec(x));
    }
    // Per-axis interior sweep (vary one axis, others at midpoint).
    let mid: Vec<f32> = pairs.iter().map(|&(l, u)| f32::midpoint(l, u)).collect();
    for axis in 0..n {
        for s in sample_points(pairs[axis].0, pairs[axis].1, per_axis) {
            let mut x = mid.clone();
            x[axis] = s;
            out.push(Array1::from_vec(x));
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(250) })]

    /// Obligation 1: the attached L2 radius bounds the true ‖rms_norm(x)‖₂.
    #[ntest::timeout(20000)]
    #[test]
    fn l2_radius_bounds_true_rmsnorm_norm_4d(
        p0 in valid_interval(4.0),
        p1 in valid_interval(4.0),
        p2 in valid_interval(4.0),
        p3 in valid_interval(4.0),
        g0 in -2.0f32..2.0,
        g1 in -2.0f32..2.0,
        g2 in -2.0f32..2.0,
        g3 in -2.0f32..2.0,
        eps in 1e-6f32..1e-2,
    ) {
        let pairs = [p0, p1, p2, p3];
        let input = box_from_pairs(&pairs);
        let ny = Array1::from_vec(vec![g0, g1, g2, g3]);
        let layer = RmsNormLayer::new(ny.clone(), eps).unwrap();

        let out = layer.propagate_ibp(&input).unwrap();
        // The conservative RMSNorm IBP path must attach the L2 sphere.
        let c = out.l2_constraint().expect("RMSNorm IBP should attach an L2 constraint");
        prop_assert_eq!(c.axis(), 0); // last (= only) axis of a 1D input
        // 1D input ⇒ radius is rank-0 (single slice).
        let radius = c.radius().iter().next().copied().unwrap();
        prop_assert!(radius.is_finite() && radius >= 0.0);

        // center is the origin here; verify against the true output norm.
        for x in grid_samples(&pairs, 5) {
            let y = rms_norm(&x, &ny, eps);
            // ‖y − center‖₂ with center = 0.
            let mut sumsq = 0.0f64;
            for (yi, ci) in y.iter().zip(c.center().iter()) {
                let d = (*yi as f64) - (*ci as f64);
                sumsq += d * d;
            }
            let true_norm = sumsq.sqrt() as f32;
            prop_assert!(
                true_norm <= radius + 1e-4,
                "L2 radius unsound: true ‖y‖₂={} > radius={} (x={:?})",
                true_norm, radius, x
            );
        }
    }

    /// Obligation 2: the Cauchy–Schwarz-tightened Linear output (a) contains
    /// every true output `W·rms_norm(x)+b`, and (b) is never wider than the box.
    #[ntest::timeout(20000)]
    #[test]
    fn linear_cs_tightening_sound_and_tighter_4to3(
        p0 in valid_interval(3.0),
        p1 in valid_interval(3.0),
        p2 in valid_interval(3.0),
        p3 in valid_interval(3.0),
        // Linear weights [out=3, in=4] and bias.
        w in proptest::array::uniform12(-1.0f32..1.0),
        b in proptest::array::uniform3(-0.5f32..0.5),
        eps in 1e-6f32..1e-2,
    ) {
        let pairs = [p0, p1, p2, p3];
        let input = box_from_pairs(&pairs);
        let ny = Array1::ones(4);
        let norm = RmsNormLayer::new(ny.clone(), eps).unwrap();
        let normed = norm.propagate_ibp(&input).unwrap();
        prop_assert!(
            normed.l2_constraint().is_some(),
            "RMSNorm IBP must attach the L2 constraint exercised by this property"
        );

        let weight = Array2::from_shape_vec((3, 4), w.to_vec()).unwrap();
        let bias = Array1::from_vec(b.to_vec());
        let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

        // Tightened (L2-aware) output, and the plain box output (constraint stripped).
        let tightened = linear.propagate_ibp(&normed).unwrap();
        let mut normed_box = normed;
        normed_box.clear_l2_constraint();
        let box_out = linear.propagate_ibp(&normed_box).unwrap();

        // (b) Tightened ⊆ box: never wider on either side.
        for o in 0..3 {
            prop_assert!(
                tightened.lower()[o] >= box_out.lower()[o] - 1e-4,
                "tightening widened the lower bound at o={}: {} < box {}",
                o, tightened.lower()[o], box_out.lower()[o]
            );
            prop_assert!(
                tightened.upper()[o] <= box_out.upper()[o] + 1e-4,
                "tightening widened the upper bound at o={}: {} > box {}",
                o, tightened.upper()[o], box_out.upper()[o]
            );
            // And still a valid interval.
            prop_assert!(tightened.lower()[o] <= tightened.upper()[o] + 1e-6);
        }

        // (a) Containment: every true W·rms_norm(x)+b lies inside the tightened box.
        for x in grid_samples(&pairs, 5) {
            let y = rms_norm(&x, &ny, eps);
            for o in 0..3 {
                let mut acc = bias[o] as f64;
                for j in 0..4 {
                    acc += (weight[[o, j]] as f64) * (y[j] as f64);
                }
                let out_o = acc as f32;
                prop_assert!(
                    out_o >= tightened.lower()[o] - 1e-3,
                    "CS tightening excluded a feasible point (below) at o={}: out={} < lower={}",
                    o, out_o, tightened.lower()[o]
                );
                prop_assert!(
                    out_o <= tightened.upper()[o] + 1e-3,
                    "CS tightening excluded a feasible point (above) at o={}: out={} > upper={}",
                    o, out_o, tightened.upper()[o]
                );
            }
        }
    }

    /// Obligation 2', CHEAP PATH: the O(out + in) box-midpoint Cauchy–Schwarz
    /// bound (nominal `mid_o = (box_lo+box_hi)/2 = W·z_mid`, recentring margin
    /// `d = ‖z_mid − center‖₂`, ULP margin `μ`) must still enclose every sampled
    /// true `W·rms_norm(x)+b`. RMSNorm's origin-centred output box is generally
    /// ASYMMETRIC (z_mid ≠ 0), so this exercises the `d > 0` recentring term that
    /// makes the cheap nominal sound even though it is NOT `W·center`. Asymmetry is
    /// forced by biasing the input intervals away from 0.
    #[ntest::timeout(20000)]
    #[test]
    fn cheap_midpoint_cs_encloses_true_rmsnorm_4to3(
        // Strictly-positive / strictly-negative intervals ⇒ asymmetric z box.
        p0 in (0.5f32..3.0).prop_flat_map(|a| (a..3.5).prop_map(move |b| (a, b))),
        p1 in (-3.0f32..-0.5).prop_flat_map(|a| (a..-0.2).prop_map(move |b| (a, b))),
        p2 in (0.3f32..2.0).prop_flat_map(|a| (a..2.5).prop_map(move |b| (a, b))),
        p3 in (-2.5f32..-0.3).prop_flat_map(|a| (a..-0.1).prop_map(move |b| (a, b))),
        w in proptest::array::uniform12(-1.0f32..1.0),
        b in proptest::array::uniform3(-0.5f32..0.5),
        // Make the first, strictly-positive coordinate nonzero constructively;
        // this guarantees a genuinely off-centre RMSNorm output box.
        g0 in prop_oneof![-2.0f32..=-0.1, 0.1f32..=2.0],
        g1 in -2.0f32..2.0,
        g2 in -2.0f32..2.0,
        g3 in -2.0f32..2.0,
        eps in 1e-6f32..1e-2,
    ) {
        let pairs = [p0, p1, p2, p3];
        let input = box_from_pairs(&pairs);
        let ny = Array1::from_vec(vec![g0, g1, g2, g3]);
        let norm = RmsNormLayer::new(ny.clone(), eps).unwrap();
        let normed = norm.propagate_ibp(&input).unwrap();

        // Confirm the box really is asymmetric about the origin (center): the
        // recentring margin d must be non-trivial for this test to be meaningful.
        let c = normed.l2_constraint().ok_or_else(|| TestCaseError::fail(
            "RMSNorm IBP must attach the L2 constraint exercised by the cheap-path property",
        ))?;
        let (zl, zu) = normed.lower_upper();
        let mut d_sq = 0.0f64;
        for ((zlv, zuv), cv) in zl.iter().zip(zu.iter()).zip(c.center().iter()) {
            let z_mid = f64::midpoint(*zlv as f64, *zuv as f64);
            let diff = z_mid - (*cv as f64);
            d_sq += diff * diff;
        }
        prop_assert!(
            d_sq.sqrt() > 1e-3,
            "constructively asymmetric generator produced a centred RMSNorm box"
        );

        let weight = Array2::from_shape_vec((3, 4), w.to_vec()).unwrap();
        let bias = Array1::from_vec(b.to_vec());
        let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

        let tightened = linear.propagate_ibp(&normed).unwrap();
        let mut normed_box = normed.clone();
        normed_box.clear_l2_constraint();
        let box_out = linear.propagate_ibp(&normed_box).unwrap();

        for o in 0..3 {
            // (b) Cheap CS still ⊆ box (only tightens).
            prop_assert!(tightened.lower()[o] >= box_out.lower()[o] - 1e-4);
            prop_assert!(tightened.upper()[o] <= box_out.upper()[o] + 1e-4);
            prop_assert!(tightened.lower()[o] <= tightened.upper()[o] + 1e-6);
        }
        // (a) Containment: every true W·rms_norm(x)+b inside the cheap-path box.
        for x in grid_samples(&pairs, 5) {
            let y = rms_norm(&x, &ny, eps);
            for o in 0..3 {
                let mut acc = bias[o] as f64;
                for j in 0..4 {
                    acc += (weight[[o, j]] as f64) * (y[j] as f64);
                }
                let out_o = acc as f32;
                prop_assert!(
                    out_o >= tightened.lower()[o] - 1e-3,
                    "cheap CS excluded a feasible point (below) o={}: {} < {}",
                    o, out_o, tightened.lower()[o]
                );
                prop_assert!(
                    out_o <= tightened.upper()[o] + 1e-3,
                    "cheap CS excluded a feasible point (above) o={}: {} > {}",
                    o, out_o, tightened.upper()[o]
                );
            }
        }
    }

    /// Obligation 1, LayerNorm (Standard): the attached L2 radius bounds the true
    /// ‖layernorm(x) − beta‖₂. The ball is centred at `beta` here, not the origin.
    #[ntest::timeout(20000)]
    #[test]
    fn l2_radius_bounds_true_layernorm_norm_4d(
        p0 in valid_interval(4.0),
        p1 in valid_interval(4.0),
        p2 in valid_interval(4.0),
        p3 in valid_interval(4.0),
        g in proptest::array::uniform4(-2.0f32..2.0),
        beta in proptest::array::uniform4(-1.0f32..1.0),
        eps in 1e-6f32..1e-2,
    ) {
        let pairs = [p0, p1, p2, p3];
        let input = box_from_pairs(&pairs);
        let ny = Array1::from_vec(g.to_vec());
        let beta_arr = Array1::from_vec(beta.to_vec());
        let layer = LayerNormLayer::new(ny.clone(), beta_arr.clone(), eps).unwrap();

        let out = layer.propagate_ibp(&input).unwrap();
        let c = out.l2_constraint().expect("Standard LayerNorm IBP should attach an L2 constraint");
        prop_assert_eq!(c.axis(), 0);
        // center must equal beta on this 1D slice.
        for (ci, bi) in c.center().iter().zip(beta_arr.iter()) {
            prop_assert!((ci - bi).abs() < 1e-6);
        }
        let radius = c.radius().iter().next().copied().unwrap();
        prop_assert!(radius.is_finite() && radius >= 0.0);

        for x in grid_samples(&pairs, 5) {
            let y = layernorm(&x, &ny, &beta_arr, eps);
            let mut sumsq = 0.0f64;
            for (yi, ci) in y.iter().zip(c.center().iter()) {
                let d = (*yi as f64) - (*ci as f64);
                sumsq += d * d;
            }
            let true_norm = sumsq.sqrt() as f32;
            prop_assert!(
                true_norm <= radius + 1e-4,
                "LayerNorm L2 radius unsound: true ‖y−beta‖₂={} > radius={} (x={:?})",
                true_norm, radius, x
            );
        }
    }
}
