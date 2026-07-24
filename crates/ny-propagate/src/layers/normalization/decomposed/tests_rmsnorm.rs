// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for decomposed RmsNorm CROWN backward propagation.
//!
//! Verifies `decomposed_rms_norm_crown_backward` from `rmsnorm.rs` produces
//! sound linear bounds that always contain the true RmsNorm output.
//!
//! Part of #4209.

use super::rmsnorm::decomposed_rms_norm_crown_backward;
use super::tests_support::{constant_batched_bounds, interpolate};
use ndarray::{arr1, Array1, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// Compute true RmsNorm output for a single sample.
///
/// RmsNorm(x)[i] = ny[i] * x[i] / sqrt(mean(x²) + eps)
fn true_rmsnorm(x: &[f32], ny: &[f32], eps: f32) -> Vec<f64> {
    let n = x.len() as f64;
    let mean_sq = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / n;
    let rms = (mean_sq + f64::from(eps)).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| f64::from(ny[i]) * f64::from(v) / rms)
        .collect()
}

/// Create identity upstream bounds: A = eye(n), b = 0.
fn identity_upstream(n: usize) -> crate::BatchedLinearBounds {
    let eye = Array2::eye(n);
    let zeros = Array1::zeros(n);
    constant_batched_bounds(eye.clone(), zeros.clone(), eye, zeros, n)
}

#[ntest::timeout(10000)]
#[test]
fn test_rmsnorm_identity_upstream_returns_ok() -> Result<()> {
    let n = 3;
    let ny = arr1(&[1.0, 1.0, 1.0]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5]).into_dyn(),
    )?;
    let upstream = identity_upstream(n);

    let result = decomposed_rms_norm_crown_backward(&upstream, &ny, eps, &x_ibp)?;

    assert_eq!(result.validation.total_rows, n);
    assert_eq!(
        result.bounds.lower_a().shape()[result.bounds.lower_a().ndim() - 1],
        n
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_rmsnorm_soundness_at_interval_center() -> Result<()> {
    let n = 3;
    let ny = arr1(&[1.0, 1.0, 1.0]);
    let eps = 1e-5;
    let x_lower = [0.5_f32, 1.0, 1.5];
    let x_upper = [1.5_f32, 2.0, 2.5];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream(n);

    let result = decomposed_rms_norm_crown_backward(&upstream, &ny, eps, &x_ibp)?;
    let bounds = &result.bounds;

    let x_center: Vec<f32> = x_lower
        .iter()
        .zip(x_upper.iter())
        .map(|(&l, &u)| f32::midpoint(l, u))
        .collect();
    let true_output = true_rmsnorm(&x_center, ny.as_slice().unwrap(), eps);

    let point = BoundedTensor::new(
        Array1::from_vec(x_center.clone()).into_dyn(),
        Array1::from_vec(x_center).into_dyn(),
    )?;
    let result_ibp = bounds.concretize_sound(&point)?;

    for j in 0..n {
        let lower = f64::from(result_ibp.lower().as_slice().unwrap()[j]);
        let upper = f64::from(result_ibp.upper().as_slice().unwrap()[j]);
        assert!(
            lower <= true_output[j] + 1e-4,
            "dim {j}: lower {lower} > true {} + tol",
            true_output[j]
        );
        assert!(
            upper >= true_output[j] - 1e-4,
            "dim {j}: upper {upper} < true {} - tol",
            true_output[j]
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_rmsnorm_soundness_at_corners() -> Result<()> {
    let n = 2;
    let ny = arr1(&[1.2, 0.8]);
    let eps = 1e-5;
    let x_lower = [0.5_f32, 1.0];
    let x_upper = [1.5_f32, 2.0];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream(n);

    let result = decomposed_rms_norm_crown_backward(&upstream, &ny, eps, &x_ibp)?;
    let bounds = &result.bounds;
    let ny_slice = ny.as_slice().unwrap();

    for &x0 in &[x_lower[0], x_upper[0]] {
        for &x1 in &[x_lower[1], x_upper[1]] {
            let x = vec![x0, x1];
            let true_output = true_rmsnorm(&x, ny_slice, eps);
            let point = BoundedTensor::new(
                Array1::from_vec(x.clone()).into_dyn(),
                Array1::from_vec(x.clone()).into_dyn(),
            )?;
            let result_ibp = bounds.concretize_sound(&point)?;

            for j in 0..n {
                let lower = f64::from(result_ibp.lower().as_slice().unwrap()[j]);
                let upper = f64::from(result_ibp.upper().as_slice().unwrap()[j]);
                assert!(
                    lower <= true_output[j] + 1e-3,
                    "corner ({x0},{x1}) dim {j}: lower {lower} > true {}",
                    true_output[j]
                );
                assert!(
                    upper >= true_output[j] - 1e-3,
                    "corner ({x0},{x1}) dim {j}: upper {upper} < true {}",
                    true_output[j]
                );
            }
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_rmsnorm_ny_scaling() -> Result<()> {
    let n = 3;
    let ny = arr1(&[2.0, 0.5, 1.5]);
    let eps = 1e-5;
    let x_lower = [0.3_f32, 0.8, 1.2];
    let x_upper = [0.7_f32, 1.2, 1.6];
    let x_ibp = BoundedTensor::new(arr1(&x_lower).into_dyn(), arr1(&x_upper).into_dyn())?;
    let upstream = identity_upstream(n);

    let result = decomposed_rms_norm_crown_backward(&upstream, &ny, eps, &x_ibp)?;
    let bounds = &result.bounds;
    let ny_slice = ny.as_slice().unwrap();

    // Check center point
    let x_center: Vec<f32> = x_lower
        .iter()
        .zip(x_upper.iter())
        .map(|(&l, &u)| f32::midpoint(l, u))
        .collect();
    let true_output = true_rmsnorm(&x_center, ny_slice, eps);
    let point = BoundedTensor::new(
        Array1::from_vec(x_center.clone()).into_dyn(),
        Array1::from_vec(x_center).into_dyn(),
    )?;
    let result_ibp = bounds.concretize_sound(&point)?;

    for j in 0..n {
        let lower = f64::from(result_ibp.lower().as_slice().unwrap()[j]);
        let upper = f64::from(result_ibp.upper().as_slice().unwrap()[j]);
        assert!(
            lower <= true_output[j] + 1e-3,
            "dim {j}: lower {lower} > true {}",
            true_output[j]
        );
        assert!(
            upper >= true_output[j] - 1e-3,
            "dim {j}: upper {upper} < true {}",
            true_output[j]
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_rmsnorm_dimension_mismatch_error() {
    let n = 3;
    let ny = arr1(&[1.0, 1.0]); // wrong size
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.5, 1.0, 1.5]).into_dyn(),
        arr1(&[1.5, 2.0, 2.5]).into_dyn(),
    )
    .unwrap();
    let upstream = identity_upstream(n);

    let result = decomposed_rms_norm_crown_backward(&upstream, &ny, eps, &x_ibp);
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_rmsnorm_fallback_count_valid() -> Result<()> {
    let n = 4;
    let ny = arr1(&[1.0, 0.5, 2.0, 0.75]);
    let eps = 1e-5;
    let x_ibp = BoundedTensor::new(
        arr1(&[0.2, 0.8, 1.2, 1.8]).into_dyn(),
        arr1(&[0.6, 1.2, 1.6, 2.2]).into_dyn(),
    )?;
    let upstream = identity_upstream(n);

    let result = decomposed_rms_norm_crown_backward(&upstream, &ny, eps, &x_ibp)?;
    assert!(result.validation.fallback_rows <= result.validation.total_rows);
    Ok(())
}

// =============================================================================
// GenBaB inv_rms norm-branching: override soundness + tightening (#norm-genbab)
// =============================================================================

use super::rmsnorm::{decomposed_rms_norm_crown_backward_with_override, InvRmsOverride};

/// Compute the IBP-derived inv_rms range for a single normalization group,
/// mirroring the directed-rounded interval arithmetic inside
/// `decomposed_rms_norm_crown_backward`. Used to pick valid split points.
fn ibp_inv_rms_range(x_l: &[f32], x_u: &[f32], eps: f32) -> (f32, f32) {
    use crate::layers::normalization::math_common::square_interval_bounds;
    use ny_tensor::{next_down_f32, next_up_f32};
    let n = x_l.len();
    let nf = n as f64;
    let mut var_l = 0.0f64;
    let mut var_u = 0.0f64;
    for i in 0..n {
        let (sq_l, sq_u) = square_interval_bounds(x_l[i], x_u[i]);
        var_l += sq_l as f64;
        var_u += sq_u as f64;
    }
    let var_l = next_down_f32((var_l / nf) as f32);
    let var_u = next_up_f32((var_u / nf) as f32);
    let var_eps_l = next_down_f32((var_l as f64 + eps as f64) as f32);
    let var_eps_u = next_up_f32((var_u as f64 + eps as f64) as f32);
    let rms_l = next_down_f32((var_eps_l as f64).sqrt() as f32);
    let rms_u = next_up_f32((var_eps_u as f64).sqrt() as f32);
    (next_down_f32(1.0 / rms_u), next_up_f32(1.0 / rms_l))
}

/// HYPOTHESIS TEST: on the swiglu-residual regime (wide input `[-1,1]`,
/// eps=1e-5), the un-narrowed decomposed RmsNorm collapses every row to the
/// fused IBP (A = 0), but a narrowed inv_rms override survives
/// `validate_norm_against_fused_ibp` (A != 0) and concretizes TIGHTER.
///
/// This is the root-cause demonstration: splitting the inv_rms range is what
/// lets the decomposed relaxation beat the fused IBP. Part of #norm-genbab.
#[ntest::timeout(20000)]
#[test]
fn test_rmsnorm_inv_rms_override_survives_fused_ibp() -> Result<()> {
    // Mirror build_swiglu_residual_kernel's norm regime: x in [-1,1], n large,
    // ny = 1, eps = 1e-5.
    let n = 64;
    let ny = Array1::from_elem(n, 1.0f32);
    let eps = 1e-5_f32;
    let x_l = vec![-1.0f32; n];
    let x_u = vec![1.0f32; n];
    let x_ibp = BoundedTensor::new(
        Array1::from_vec(x_l.clone()).into_dyn(),
        Array1::from_vec(x_u.clone()).into_dyn(),
    )?;
    let upstream = identity_upstream(n);

    // (a) Un-narrowed: every row collapses to fused IBP.
    let base = decomposed_rms_norm_crown_backward(&upstream, &ny, eps, &x_ibp)?;
    assert_eq!(
        base.validation.fallback_rows, base.validation.total_rows,
        "baseline (wide inv_rms) should collapse ALL rows to fused IBP"
    );

    // Baseline concretized envelope width (identity upstream => per-element
    // RmsNorm output bounds), over the wide input box.
    let identity = crate::BatchedLinearBounds::identity(x_ibp.shape())
        .map_err(|e| ny_core::NyError::InternalError(format!("identity: {e}")))?;
    let base_id = decomposed_rms_norm_crown_backward(&identity, &ny, eps, &x_ibp)?;
    let base_conc = base_id.bounds.concretize_sound(&x_ibp)?;
    let base_width: f32 = base_conc
        .lower()
        .iter()
        .zip(base_conc.upper().iter())
        .map(|(&l, &u)| u - l)
        .fold(0.0, f32::max);

    // (b) A narrowed inv_rms window TIGHTENS the bound. The improvement comes
    // either from the decomposed relaxation surviving the (now per-group
    // tightened) fused-IBP fallback OR from the tightened fallback itself
    // (`ny·x·[inv_lo,inv_hi]`); both are sound and both shrink the envelope.
    let (inv_l, inv_u) = ibp_inv_rms_range(&x_l, &x_u, eps);
    let center = 1.0 / ((1.0f32 / 3.0) + eps).sqrt();
    let half = 0.5f32; // a moderately narrow window around the bulk inv_rms.
    let lo = (center - half).max(inv_l);
    let hi = (center + half).min(inv_u);
    let narrowed_id = decomposed_rms_norm_crown_backward_with_override(
        &identity,
        &ny,
        eps,
        &x_ibp,
        Some(InvRmsOverride::single_group(0, lo, hi)),
    )?;
    let narrowed_conc = narrowed_id.bounds.concretize_sound(&x_ibp)?;
    let narrowed_width: f32 = narrowed_conc
        .lower()
        .iter()
        .zip(narrowed_conc.upper().iter())
        .map(|(&l, &u)| u - l)
        .fold(0.0, f32::max);

    assert!(
        narrowed_width < base_width,
        "narrowed inv_rms override (window [{lo:.3},{hi:.3}]) must TIGHTEN the \
         concretized envelope: base_width={base_width}, narrowed_width={narrowed_width}"
    );
    Ok(())
}

/// SOUNDNESS: for a sub-interval inv_rms override, the resulting CROWN bound
/// must still contain the true RmsNorm output for every concrete x whose
/// inv_rms(x) lies inside the override range. (The override is only USED on its
/// own subregion; here we sample exactly that subregion.)
#[ntest::timeout(20000)]
#[test]
fn test_rmsnorm_inv_rms_override_sound_on_subregion() -> Result<()> {
    let n = 8;
    let ny = Array1::from_elem(n, 1.0f32);
    let eps = 1e-5_f32;
    let x_l = vec![-1.0f32; n];
    let x_u = vec![1.0f32; n];
    let x_ibp = BoundedTensor::new(
        Array1::from_vec(x_l.clone()).into_dyn(),
        Array1::from_vec(x_u.clone()).into_dyn(),
    )?;
    let upstream = identity_upstream(n);

    let (inv_l, inv_u) = ibp_inv_rms_range(&x_l, &x_u, eps);
    let mid = f32::midpoint(inv_l, inv_u);

    // Lower-inv_rms child: inv_rms in [inv_l, mid] <=> larger ||x||.
    let ov = InvRmsOverride::single_group(0, inv_l, mid);
    let res =
        decomposed_rms_norm_crown_backward_with_override(&upstream, &ny, eps, &x_ibp, Some(ov))?;

    // Sample x in the box; keep ONLY samples whose inv_rms(x) falls in the
    // override sub-range [inv_l, mid] (the child's subregion). The bound must
    // contain the true output for all such x.
    let mut checked = 0;
    for seed in 0..2000u32 {
        // Cheap deterministic pseudo-random sample in [-1,1]^n.
        let mut x = vec![0.0f32; n];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for xi in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *xi = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        let mean_sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
        let inv_rms = (1.0 / (mean_sq + eps as f64).sqrt()) as f32;
        if inv_rms < inv_l || inv_rms > mid {
            continue;
        }
        checked += 1;
        let true_out = true_rmsnorm(&x, ny.as_slice().unwrap(), eps);
        let point = BoundedTensor::new(
            Array1::from_vec(x.clone()).into_dyn(),
            Array1::from_vec(x.clone()).into_dyn(),
        )?;
        let conc = res.bounds.concretize_sound(&point)?;
        for j in 0..n {
            let lo = f64::from(conc.lower().as_slice().unwrap()[j]);
            let hi = f64::from(conc.upper().as_slice().unwrap()[j]);
            assert!(
                lo <= true_out[j] + 1e-2,
                "subregion-sound lower violated dim {j}: lo={lo} > true={} ",
                true_out[j]
            );
            assert!(
                hi >= true_out[j] - 1e-2,
                "subregion-sound upper violated dim {j}: hi={hi} < true={}",
                true_out[j]
            );
        }
    }
    assert!(
        checked > 5,
        "expected several in-subregion samples, got {checked}"
    );
    Ok(())
}

/// SOUNDNESS (union cover): the two sibling children — inv_rms in [inv_l, mid]
/// and [mid, inv_u] — must between them contain the true output for EVERY x in
/// the box. We verify that for every sampled x, at least one child's bound
/// contains the true output (the one whose range covers inv_rms(x)).
#[ntest::timeout(20000)]
#[test]
fn test_rmsnorm_inv_rms_split_union_covers() -> Result<()> {
    let n = 8;
    let ny = Array1::from_elem(n, 1.0f32);
    let eps = 1e-5_f32;
    let x_l = vec![-1.0f32; n];
    let x_u = vec![1.0f32; n];
    let x_ibp = BoundedTensor::new(
        Array1::from_vec(x_l.clone()).into_dyn(),
        Array1::from_vec(x_u.clone()).into_dyn(),
    )?;
    let upstream = identity_upstream(n);

    let (inv_l, inv_u) = ibp_inv_rms_range(&x_l, &x_u, eps);
    let mid = f32::midpoint(inv_l, inv_u);

    let lo_child = decomposed_rms_norm_crown_backward_with_override(
        &upstream,
        &ny,
        eps,
        &x_ibp,
        Some(InvRmsOverride::single_group(0, inv_l, mid)),
    )?;
    let hi_child = decomposed_rms_norm_crown_backward_with_override(
        &upstream,
        &ny,
        eps,
        &x_ibp,
        Some(InvRmsOverride::single_group(0, mid, inv_u)),
    )?;

    for seed in 0..3000u32 {
        let mut x = vec![0.0f32; n];
        let mut s = seed.wrapping_mul(40503).wrapping_add(7);
        for xi in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *xi = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        let mean_sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
        let inv_rms = (1.0 / (mean_sq + eps as f64).sqrt()) as f32;
        // Select the child whose range covers inv_rms(x) (the one BaB would use
        // for this x). Boundary x (inv_rms == mid) belongs to both; pick lo.
        let child = if inv_rms <= mid { &lo_child } else { &hi_child };

        let true_out = true_rmsnorm(&x, ny.as_slice().unwrap(), eps);
        let point = BoundedTensor::new(
            Array1::from_vec(x.clone()).into_dyn(),
            Array1::from_vec(x.clone()).into_dyn(),
        )?;
        let conc = child.bounds.concretize_sound(&point)?;
        for j in 0..n {
            let lo = f64::from(conc.lower().as_slice().unwrap()[j]);
            let hi = f64::from(conc.upper().as_slice().unwrap()[j]);
            assert!(
                lo <= true_out[j] + 1e-2 && hi >= true_out[j] - 1e-2,
                "union-cover violated for x with inv_rms={inv_rms} at dim {j}: \
                 [{lo}, {hi}] excludes true={}",
                true_out[j]
            );
        }
    }
    Ok(())
}

/// MEASUREMENT (#norm-genbab): how the concretized `sum_i RmsNorm(x)_i` lower
/// bound improves as the inv_rms window narrows toward the worst corner x=-1
/// (inv_rms ~ 1). Prints the bound vs window so we can see whether splitting
/// inv_rms moves the objective and how deep BaB must go.
#[ntest::timeout(20000)]
#[ignore = "diagnostic measurement, run explicitly"]
#[test]
fn measure_rmsnorm_sum_lower_vs_window() -> Result<()> {
    let n = 8;
    let ny = Array1::from_elem(n, 1.0f32);
    let eps = 1e-5_f32;
    let x_l = vec![-1.0f32; n];
    let x_u = vec![1.0f32; n];
    let x_ibp = BoundedTensor::new(
        Array1::from_vec(x_l.clone()).into_dyn(),
        Array1::from_vec(x_u.clone()).into_dyn(),
    )?;
    // upstream A = row of ones (objective = sum of RmsNorm outputs).
    let a = Array2::from_elem((1, n), 1.0f32);
    let b = Array1::zeros(1);
    let upstream = constant_batched_bounds(a.clone(), b.clone(), a, b, n);

    let (inv_l, inv_u) = ibp_inv_rms_range(&x_l, &x_u, eps);
    eprintln!(
        "inv_rms IBP range [{inv_l}, {inv_u}]; analytic sum-min = {}",
        -(n as f32)
    );
    // Window anchored at inv_l (worst corner) with shrinking width.
    for k in 0..14 {
        let hi = (inv_l + (inv_u - inv_l) / 2.0f32.powi(k)).min(inv_u);
        let res = decomposed_rms_norm_crown_backward_with_override(
            &upstream,
            &ny,
            eps,
            &x_ibp,
            Some(InvRmsOverride::single_group(0, inv_l, hi)),
        )?;
        let conc = res.bounds.concretize_sound(&x_ibp)?;
        let lo = conc.lower().as_slice().unwrap()[0];
        eprintln!(
            "k={k} window=[{inv_l:.3},{hi:.3}] survived={}/{} sum_lower={lo:.4}",
            res.validation.total_rows - res.validation.fallback_rows,
            res.validation.total_rows,
        );
    }
    // High-tail windows [lo_h, inv_u] (small ‖x‖) stay at the standard fused
    // envelope: RmsNorm output is SCALE-INVARIANT in ‖x‖, so narrowing inv_rms
    // alone does NOT tighten the high tail (|x_i·inv_rms| ≤ √n regardless of
    // ‖x‖). Only the low-inv_rms (box-saturating) region tightens — that is
    // where the worst case lives, but isolating it from the wide IBP range is
    // the search-efficiency cost.
    for k in 0..4 {
        let lo_h = (inv_u / 2.0f32.powi(k + 1)).max(inv_l);
        let res = decomposed_rms_norm_crown_backward_with_override(
            &upstream,
            &ny,
            eps,
            &x_ibp,
            Some(InvRmsOverride::single_group(0, lo_h, inv_u)),
        )?;
        let conc = res.bounds.concretize_sound(&x_ibp)?;
        let lo = conc.lower().as_slice().unwrap()[0];
        let hi = conc.upper().as_slice().unwrap()[0];
        eprintln!("HIGH k={k} window=[{lo_h:.2},{inv_u:.2}] sum_bound=[{lo:.3},{hi:.3}]");
    }
    Ok(())
}

/// Override is intersected, never widening: an absurdly wide override
/// (covering all of inv_rms and beyond) must reproduce the un-narrowed result
/// exactly (same fallback count), proving intersection semantics.
#[ntest::timeout(10000)]
#[test]
fn test_rmsnorm_inv_rms_override_never_widens() -> Result<()> {
    let n = 16;
    let ny = Array1::from_elem(n, 1.0f32);
    let eps = 1e-5_f32;
    let x_l = vec![-1.0f32; n];
    let x_u = vec![1.0f32; n];
    let x_ibp = BoundedTensor::new(
        Array1::from_vec(x_l).into_dyn(),
        Array1::from_vec(x_u).into_dyn(),
    )?;
    let upstream = identity_upstream(n);

    let base = decomposed_rms_norm_crown_backward(&upstream, &ny, eps, &x_ibp)?;
    let wide = decomposed_rms_norm_crown_backward_with_override(
        &upstream,
        &ny,
        eps,
        &x_ibp,
        Some(InvRmsOverride::single_group(0, 0.0, 1e9)),
    )?;
    assert_eq!(
        base.validation.fallback_rows, wide.validation.fallback_rows,
        "an all-covering override must not change the result (intersection only narrows)"
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(64) })]

    #[test]
    fn proptest_rmsnorm_decomposed_contains_true_output(
        x0_l in 0.3f32..2.0,
        x0_w in 0.05f32..0.6,
        x1_l in 0.3f32..2.0,
        x1_w in 0.05f32..0.6,
        x2_l in 0.3f32..2.0,
        x2_w in 0.05f32..0.6,
        // Randomized ny to stress-test scaling interaction with CROWN relaxation
        g0 in 0.3f32..2.5,
        g1 in 0.3f32..2.5,
        g2 in 0.3f32..2.5,
        t0 in 0.0f32..1.0,
        t1 in 0.0f32..1.0,
        t2 in 0.0f32..1.0,
    ) {
        let n = 3;
        let ny = arr1(&[g0, g1, g2]);
        let eps = 1e-5_f32;
        let x_lower = [x0_l, x1_l, x2_l];
        let x_upper = [x0_l + x0_w, x1_l + x1_w, x2_l + x2_w];
        let x_ibp = BoundedTensor::new(
            arr1(&x_lower).into_dyn(),
            arr1(&x_upper).into_dyn(),
        ).unwrap();
        let upstream = identity_upstream(n);

        let result = decomposed_rms_norm_crown_backward(&upstream, &ny, eps, &x_ibp);
        // Use prop_assume! so proptest generates replacement cases for numerically
        // ill-conditioned inputs, rather than silently counting errors as passes.
        // Without this, a regression that always returns Err passes the test vacuously.
        prop_assume!(result.is_ok(), "decomposed rmsnorm returned error: {:?}", result.err());
        let result = result.unwrap();
        let bounds = &result.bounds;

        let x_sample = vec![
            interpolate(x_lower[0], x_upper[0], t0),
            interpolate(x_lower[1], x_upper[1], t1),
            interpolate(x_lower[2], x_upper[2], t2),
        ];
        let true_output = true_rmsnorm(&x_sample, ny.as_slice().unwrap(), eps);

        let point = BoundedTensor::new(
            Array1::from_vec(x_sample.clone()).into_dyn(),
            Array1::from_vec(x_sample).into_dyn(),
        ).unwrap();
        let result_ibp = bounds.concretize_sound(&point).unwrap();

        for j in 0..n {
            let lower = f64::from(result_ibp.lower().as_slice().unwrap()[j]);
            let upper = f64::from(result_ibp.upper().as_slice().unwrap()[j]);
            prop_assert!(
                lower <= true_output[j] + 1e-2,
                "dim {}: lower {} > true {} + tol", j, lower, true_output[j]
            );
            prop_assert!(
                upper >= true_output[j] - 1e-2,
                "dim {}: upper {} < true {} - tol", j, upper, true_output[j]
            );
        }
    }
}
