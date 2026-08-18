// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Independent certificate checker for complete clipping dual witnesses.
//!
//! The coordinate-ascent solver is useful for *finding* multipliers, but its
//! ordinary `f32` accumulation is not itself a proof that the reported bound is
//! outward rounded.  This module treats every multiplier as untrusted and
//! re-evaluates the corresponding Lagrangian dual bound from the original
//! objective, constraints, and input box.
//!
//! Every source value admitted to a witness is a finite `f32` and therefore an
//! exact real dyadic number; malformed rows fall back to the zero witness.
//! Every arithmetic operation below is performed in `f64` and immediately
//! widened by one adjacent representable value.  Thus `q_lo <= c + beta*A <=
//! q_hi` is maintained coefficient by coefficient, and each box corner product
//! is rounded toward `-inf`.  The final `f32` conversion is checked in `f64` and
//! stepped outward when necessary.  The checker never consumes the optimizer's
//! claimed objective value.

use ndarray::{Array2, Array3, ArrayD};
use ny_core::{NyError, Result};
use ny_tensor::next_down_f32;
#[cfg(test)]
use std::time::Instant;

use super::objective::broadcast_objective_with_deadline_check;
use super::{check_clip_deadline_with, CLIP_DEADLINE_POLL_STRIDE};

/// Certify all rows of a complete-clipping dual witness.
///
/// `a_work`/`b_work` must be the exact (possibly reordered) constraint rows used
/// to produce `beta_store`.  A malformed multiplier row is not an error: that
/// row falls back to the independently checked box-only witness `beta = 0`.
#[cfg(test)]
pub(super) fn certify_dual_witness(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    objective: &ArrayD<f32>,
    a_work: &Array3<f32>,
    b_work: &Array2<f32>,
    beta_store: &Array3<f32>,
    sign: f32,
    deadline: Option<Instant>,
) -> Result<ArrayD<f32>> {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    certify_dual_witness_with_deadline_check(
        x_l,
        x_u,
        objective,
        a_work,
        b_work,
        beta_store,
        sign,
        &mut past_deadline,
    )
}

pub(super) fn certify_dual_witness_with_deadline_check<F>(
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    objective: &ArrayD<f32>,
    a_work: &Array3<f32>,
    b_work: &Array2<f32>,
    beta_store: &Array3<f32>,
    sign: f32,
    past_deadline: &mut F,
) -> Result<ArrayD<f32>>
where
    F: FnMut() -> bool,
{
    let x_l = x_l
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| NyError::InvalidSpec(format!("clip certificate x_l shape: {e}")))?;
    let x_u = x_u
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| NyError::InvalidSpec(format!("clip certificate x_u shape: {e}")))?;
    let [batch, x_dim]: [usize; 2] = x_l
        .shape()
        .try_into()
        .map_err(|_| NyError::InvalidSpec("clip certificate x_l must be 2D".into()))?;
    if x_u.shape() != [batch, x_dim] {
        return Err(NyError::InvalidSpec(format!(
            "clip certificate x_u shape {:?} != [{batch}, {x_dim}]",
            x_u.shape()
        )));
    }

    let obj_shape = objective.shape();
    let h_dim = match obj_shape {
        [h, d] if *d == x_dim => *h,
        [b, h, d] if *b == batch && *d == x_dim => *h,
        _ => {
            return Err(NyError::InvalidSpec(format!(
                "clip certificate objective shape {obj_shape:?} is incompatible with [{batch}, {x_dim}]"
            )))
        }
    };
    let n_constraints = a_work.shape().get(1).copied().unwrap_or(0);
    if a_work.shape() != [batch, n_constraints, x_dim]
        || b_work.shape() != [batch, n_constraints]
        || beta_store.shape() != [batch, h_dim, n_constraints]
    {
        return Err(NyError::InvalidSpec(format!(
            "clip certificate witness shape mismatch: A={:?} b={:?} beta={:?}, expected A=[{batch},{n_constraints},{x_dim}] b=[{batch},{n_constraints}] beta=[{batch},{h_dim},{n_constraints}]",
            a_work.shape(),
            b_work.shape(),
            beta_store.shape(),
        )));
    }
    // This check precedes `broadcast_objective`, the certified output, and all
    // per-row witness vectors. Shape products are fallible and the hard cap is
    // shared with the proposal generator.
    super::validate_clip_work_budget(batch, h_dim, n_constraints, x_dim)?;
    check_clip_deadline_with(past_deadline, "certificate allocations")?;
    let objective =
        broadcast_objective_with_deadline_check(objective, batch, h_dim, x_dim, past_deadline)?;

    if !sign.is_finite() {
        return Err(NyError::InvalidSpec(
            "clip certificate sign must be finite".into(),
        ));
    }
    let valid_cells = batch.checked_mul(n_constraints).ok_or_else(|| {
        NyError::InvalidSpec("clip certificate constraint validity shape overflow".into())
    })?;
    let mut valid_constraint = Vec::new();
    valid_constraint
        .try_reserve_exact(valid_cells)
        .map_err(|e| {
            NyError::InvalidSpec(format!("clip certificate validity allocation failed: {e}"))
        })?;
    valid_constraint.resize(valid_cells, true);
    for b in 0..batch {
        for j in 0..x_dim {
            if j.is_multiple_of(1024) {
                check_clip_deadline_with(past_deadline, "certificate box validation")?;
            }
            let (lo, hi) = (x_l[[b, j]], x_u[[b, j]]);
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(NyError::InvalidSpec(format!(
                    "clip certificate invalid input box at batch={b} dim={j}"
                )));
            }
        }
        for k in 0..n_constraints {
            if k.is_multiple_of(64) {
                check_clip_deadline_with(past_deadline, "certificate constraint validation")?;
            }
            let mut row_finite = true;
            for j in 0..x_dim {
                if j.is_multiple_of(1024) {
                    check_clip_deadline_with(past_deadline, "certificate constraint row")?;
                }
                if !a_work[[b, k, j]].is_finite() {
                    row_finite = false;
                    break;
                }
            }
            valid_constraint[b * n_constraints + k] = b_work[[b, k]].is_finite() && row_finite;
        }
    }

    let maximize = sign > 0.0;
    check_clip_deadline_with(past_deadline, "certificate output allocation")?;
    let mut certified = Array2::<f32>::zeros((batch, h_dim));
    for b in 0..batch {
        for h in 0..h_dim {
            if h.is_multiple_of(8) {
                check_clip_deadline_with(past_deadline, "certificate objective row")?;
            }
            let mut c = Vec::new();
            c.try_reserve_exact(x_dim).map_err(|e| {
                NyError::InvalidSpec(format!(
                    "clip certificate objective-row allocation failed: {e}"
                ))
            })?;
            for j in 0..x_dim {
                if j.is_multiple_of(1024) {
                    check_clip_deadline_with(past_deadline, "certificate objective copy")?;
                }
                let value = objective[[b, h, j]];
                if !value.is_finite() {
                    return Err(NyError::InvalidSpec(format!(
                        "clip certificate non-finite objective at batch={b} row={h} dim={j}"
                    )));
                }
                c.push(if maximize { -value } else { value });
            }

            // The zero witness is always valid and makes a malformed proposal a
            // conservative no-op instead of an authority failure.
            let baseline = certify_min_row_with_deadline_check(
                &c,
                &[],
                b,
                &x_l,
                &x_u,
                a_work,
                b_work,
                past_deadline,
            )?;
            check_clip_deadline_with(past_deadline, "certificate witness allocation")?;
            let mut betas = Vec::new();
            betas.try_reserve_exact(n_constraints).map_err(|e| {
                NyError::InvalidSpec(format!("clip certificate witness allocation failed: {e}"))
            })?;
            let mut proposed_valid = true;
            for k in 0..n_constraints {
                if k.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "certificate witness copy")?;
                }
                let value = beta_store[[b, h, k]];
                proposed_valid &=
                    valid_constraint[b * n_constraints + k] && value.is_finite() && value >= 0.0;
                betas.push(value);
            }
            let proposed = if proposed_valid {
                certify_min_row_with_deadline_check(
                    &c,
                    &betas,
                    b,
                    &x_l,
                    &x_u,
                    a_work,
                    b_work,
                    past_deadline,
                )?
            } else {
                baseline
            };
            let lower = baseline.max(proposed);
            let stored_lower = f64_to_f32_down(lower);
            // `stored_lower <= min(-objective)` implies
            // `-stored_lower >= max(objective)`.  f32 negation is exact.
            certified[[b, h]] = if maximize {
                -stored_lower
            } else {
                stored_lower
            };
        }
    }
    check_clip_deadline_with(past_deadline, "certificate completion")?;
    Ok(certified.into_dyn())
}

/// Outward evaluation of one minimization witness.
///
/// An empty `betas` slice means the all-zero witness; otherwise its length must
/// equal the number of constraints.  The returned finite `f64` is no greater
/// than the exact-real Lagrangian minimum over the input box.
#[cfg(test)]
fn certify_min_row(
    objective: &[f32],
    betas: &[f32],
    batch: usize,
    x_l: &ndarray::ArrayView2<'_, f32>,
    x_u: &ndarray::ArrayView2<'_, f32>,
    a_work: &Array3<f32>,
    b_work: &Array2<f32>,
    deadline: Option<Instant>,
) -> Result<f64> {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    certify_min_row_with_deadline_check(
        objective,
        betas,
        batch,
        x_l,
        x_u,
        a_work,
        b_work,
        &mut past_deadline,
    )
}

fn certify_min_row_with_deadline_check<F>(
    objective: &[f32],
    betas: &[f32],
    batch: usize,
    x_l: &ndarray::ArrayView2<'_, f32>,
    x_u: &ndarray::ArrayView2<'_, f32>,
    a_work: &Array3<f32>,
    b_work: &Array2<f32>,
    past_deadline: &mut F,
) -> Result<f64>
where
    F: FnMut() -> bool,
{
    let x_dim = objective.len();
    let n_constraints = a_work.shape()[1];
    if !betas.is_empty() && betas.len() != n_constraints {
        return Err(NyError::InvalidSpec(format!(
            "clip certificate beta row length {} != {n_constraints}",
            betas.len()
        )));
    }

    // For feasible A_k*x+b_k <= 0 and beta_k >= 0,
    // objective(x) >= objective(x) + sum beta_k(A_k*x+b_k).
    let mut lower = 0.0f64;
    let mut coefficient_cells = 0usize;
    for j in 0..x_dim {
        if j.is_multiple_of(1024) {
            check_clip_deadline_with(past_deadline, "certificate coefficient fold")?;
        }
        let c = f64::from(objective[j]);
        let mut q_lo = c;
        let mut q_hi = c;
        if !betas.is_empty() {
            for k in 0..n_constraints {
                if coefficient_cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "certificate constraint coefficient")?;
                }
                let (prod_lo, prod_hi) =
                    mul_interval(f64::from(betas[k]), f64::from(a_work[[batch, k, j]]));
                q_lo = add_down(q_lo, prod_lo);
                q_hi = add_up(q_hi, prod_hi);
                coefficient_cells = coefficient_cells.saturating_add(1);
            }
        }

        let lo = f64::from(x_l[[batch, j]]);
        let hi = f64::from(x_u[[batch, j]]);
        // A bilinear function on [q_lo,q_hi] x [lo,hi] reaches its minimum
        // at a corner.  Each candidate is individually rounded downward.
        let term = [
            mul_down(q_lo, lo),
            mul_down(q_lo, hi),
            mul_down(q_hi, lo),
            mul_down(q_hi, hi),
        ]
        .into_iter()
        .fold(f64::INFINITY, f64::min);
        lower = add_down(lower, term);
    }
    if !betas.is_empty() {
        for k in 0..n_constraints {
            if k.is_multiple_of(1024) {
                check_clip_deadline_with(past_deadline, "certificate bias fold")?;
            }
            lower = add_down(
                lower,
                mul_down(f64::from(betas[k]), f64::from(b_work[[batch, k]])),
            );
        }
    }
    if lower.is_nan() {
        return Err(NyError::InvalidSpec("clip certificate produced NaN".into()));
    }
    check_clip_deadline_with(past_deadline, "certificate row completion")?;
    Ok(lower)
}

fn mul_interval(a: f64, b: f64) -> (f64, f64) {
    (mul_down(a, b), mul_up(a, b))
}

fn add_down(a: f64, b: f64) -> f64 {
    next_down_f64(a + b)
}

fn add_up(a: f64, b: f64) -> f64 {
    next_up_f64(a + b)
}

fn mul_down(a: f64, b: f64) -> f64 {
    next_down_f64(a * b)
}

fn mul_up(a: f64, b: f64) -> f64 {
    next_up_f64(a * b)
}

fn next_down_f64(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() || bits == f64::NEG_INFINITY.to_bits() {
        return value;
    }
    if magnitude == 0 {
        return -f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn next_up_f64(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() || bits == f64::INFINITY.to_bits() {
        return value;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn f64_to_f32_down(value: f64) -> f32 {
    if value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value >= f64::from(f32::MAX) {
        return f32::MAX;
    }
    if value < -f64::from(f32::MAX) {
        return f32::NEG_INFINITY;
    }
    let candidate = value as f32;
    if f64::from(candidate) <= value {
        candidate
    } else {
        next_down_f32(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};
    use num_rational::BigRational;
    use num_traits::ToPrimitive;

    fn exact(value: f32) -> BigRational {
        BigRational::from_float(value).expect("finite f32")
    }

    fn exact_dual_min(
        c: &[f32],
        beta: &[f32],
        a: &Array3<f32>,
        b: &Array2<f32>,
        lo: &[f32],
        hi: &[f32],
    ) -> BigRational {
        let mut total = BigRational::from_integer(0.into());
        for j in 0..c.len() {
            let mut q = exact(c[j]);
            for k in 0..beta.len() {
                q += exact(beta[k]) * exact(a[[0, k, j]]);
            }
            total += if q >= BigRational::from_integer(0.into()) {
                q * exact(lo[j])
            } else {
                q * exact(hi[j])
            };
        }
        for k in 0..beta.len() {
            total += exact(beta[k]) * exact(b[[0, k]]);
        }
        total
    }

    #[test]
    fn outward_evaluator_is_below_exact_dyadic_dual() {
        // Cancellation-heavy coefficients exercise both q interval directions
        // and the final 2-D box corner choice.
        let a = arr3(&[[[16_777_216.0f32, -0.75], [-16_777_216.0, 0.5], [1.25, -2.0]]]);
        let b = arr2(&[[-0.5f32, 0.25, -0.125]]);
        let lo = arr2(&[[-1.0f32, -0.25]]);
        let hi = arr2(&[[0.75f32, 2.0]]);
        let c = [0.3f32, -1.1];
        let beta = [0.7f32, 0.7, 1.3];
        let got = certify_min_row(&c, &beta, 0, &lo.view(), &hi.view(), &a, &b, None)
            .expect("finite certificate");
        let oracle = exact_dual_min(
            &c,
            &beta,
            &a,
            &b,
            lo.row(0).as_slice().unwrap(),
            hi.row(0).as_slice().unwrap(),
        );
        assert!(
            BigRational::from_float(got).expect("finite result") <= oracle,
            "outward result {got} exceeded exact dual {}",
            oracle.to_f64().unwrap()
        );
    }

    #[test]
    fn outward_evaluator_matches_exact_direction_over_generated_witnesses() {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut sample = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 32) as i32 % 257) as f32) / 16.0
        };
        for case in 0..64 {
            let dim = 7;
            let n_constraints = 4;
            let mut lo_v = Vec::with_capacity(dim);
            let mut hi_v = Vec::with_capacity(dim);
            let mut c = Vec::with_capacity(dim);
            for _ in 0..dim {
                let x = sample();
                let y = sample();
                lo_v.push(x.min(y));
                hi_v.push(x.max(y));
                c.push(sample());
            }
            let mut a_v = Vec::with_capacity(n_constraints * dim);
            let mut b_v = Vec::with_capacity(n_constraints);
            let mut beta = Vec::with_capacity(n_constraints);
            for _ in 0..n_constraints {
                for _ in 0..dim {
                    a_v.push(sample());
                }
                b_v.push(sample());
                beta.push(sample().abs());
            }
            let lo = Array2::from_shape_vec((1, dim), lo_v.clone()).unwrap();
            let hi = Array2::from_shape_vec((1, dim), hi_v.clone()).unwrap();
            let a = Array3::from_shape_vec((1, n_constraints, dim), a_v).unwrap();
            let b = Array2::from_shape_vec((1, n_constraints), b_v).unwrap();
            let got = certify_min_row(&c, &beta, 0, &lo.view(), &hi.view(), &a, &b, None)
                .expect("finite generated certificate");
            let oracle = exact_dual_min(&c, &beta, &a, &b, &lo_v, &hi_v);
            assert!(
                BigRational::from_float(got).expect("finite result") <= oracle,
                "case {case}: outward result {got} exceeded exact dual {}",
                oracle.to_f64().unwrap()
            );
        }
    }

    #[test]
    fn certified_lower_and_upper_use_the_dual_not_reported_float_bound() {
        // Domain x in [0,1], necessary constraint x >= 0.5 encoded as
        // -x+0.5 <= 0. beta=1 proves min(x)>=0.5 and max(-x)<=-0.5.
        let lo = arr2(&[[0.0f32]]).into_dyn();
        let hi = arr2(&[[1.0f32]]).into_dyn();
        let a = arr3(&[[[-1.0f32]]]);
        let b = arr2(&[[0.5f32]]);
        let beta = arr3(&[[[1.0f32]]]);

        let lower = certify_dual_witness(
            &lo,
            &hi,
            &arr2(&[[1.0f32]]).into_dyn(),
            &a,
            &b,
            &beta,
            -1.0,
            None,
        )
        .expect("lower certificate");
        let upper_neg_x = certify_dual_witness(
            &lo,
            &hi,
            &arr2(&[[-1.0f32]]).into_dyn(),
            &a,
            &b,
            &beta,
            1.0,
            None,
        )
        .expect("upper certificate");
        assert!(lower[[0, 0]] <= 0.5 && lower[[0, 0]] > 0.499_99);
        assert!(upper_neg_x[[0, 0]] >= -0.5 && upper_neg_x[[0, 0]] < -0.499_99);
    }

    #[test]
    fn malformed_beta_row_falls_back_to_checked_box_bound() {
        let lo = arr2(&[[-2.0f32]]).into_dyn();
        let hi = arr2(&[[3.0f32]]).into_dyn();
        let a = arr3(&[[[1.0f32]]]);
        let b = arr2(&[[-1.0f32]]);
        for bad in [-1.0f32, f32::NAN, f32::INFINITY] {
            let beta = arr3(&[[[bad]]]);
            let got = certify_dual_witness(
                &lo,
                &hi,
                &arr2(&[[1.0f32]]).into_dyn(),
                &a,
                &b,
                &beta,
                -1.0,
                None,
            )
            .expect("malformed beta should use baseline");
            assert!(got[[0, 0]] <= -2.0);
        }
    }

    #[test]
    fn zero_beta_certificate_matches_box_witness_in_both_directions() {
        let lo = arr2(&[[-2.0f32, -1.0]]).into_dyn();
        let hi = arr2(&[[3.0f32, 4.0]]).into_dyn();
        let objective = arr2(&[[2.0f32, -3.0]]).into_dyn();
        let a = arr3(&[[[1.0f32, -1.0], [-0.5, 0.25]]]);
        let b = arr2(&[[-100.0f32, -100.0]]);
        let beta = Array3::<f32>::zeros((1, 1, 2));

        let lower = certify_dual_witness(&lo, &hi, &objective, &a, &b, &beta, -1.0, None).unwrap();
        let upper = certify_dual_witness(&lo, &hi, &objective, &a, &b, &beta, 1.0, None).unwrap();
        // Exact box extrema are -16 and 9. The checked zero witness may only
        // widen outward, never inherit a proposal-side tightening.
        assert!(lower[[0, 0]] <= -16.0 && lower[[0, 0]] > -16.001);
        assert!(upper[[0, 0]] >= 9.0 && upper[[0, 0]] < 9.001);
    }

    #[test]
    fn checker_rejects_malformed_rearranged_witness_shapes() {
        let lo = arr2(&[[0.0f32]]).into_dyn();
        let hi = arr2(&[[1.0f32]]).into_dyn();
        let objective = arr2(&[[1.0f32]]).into_dyn();
        // A/b represent two already-rearranged rows, but beta has only one
        // column. Pairing it with either row would certify the wrong witness.
        let a = arr3(&[[[-1.0f32], [1.0]]]);
        let b = arr2(&[[0.5f32, -0.75]]);
        let beta = Array3::<f32>::zeros((1, 1, 1));
        let err = certify_dual_witness(&lo, &hi, &objective, &a, &b, &beta, -1.0, None)
            .expect_err("rearranged witness shape mismatch must fail closed");
        assert!(err.to_string().contains("witness shape mismatch"));
    }

    #[test]
    fn f64_to_f32_down_is_directional_at_half_ulp() {
        let one = 1.0f32;
        let next = f32::from_bits(one.to_bits() + 1);
        let midpoint = f64::midpoint(f64::from(one), f64::from(next));
        let just_below = f64::from_bits(midpoint.to_bits() - 1);
        assert_eq!(f64_to_f32_down(just_below), one);
        assert!(f64::from(f64_to_f32_down(-just_below)) <= -just_below);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn x1_many_constraints_poll_inside_certificate_coefficient_fold() {
        let n = 32_768usize;
        let objective = [1.0f32];
        let beta = vec![0.25f32; n];
        let lo = arr2(&[[-1.0f32]]);
        let hi = arr2(&[[1.0f32]]);
        let a = Array3::from_elem((1, n, 1), 0.5f32);
        let b = Array2::from_elem((1, n), -1.0f32);
        let mut polls = 0usize;
        let mut expire = || {
            polls += 1;
            polls >= 4
        };
        let err = certify_min_row_with_deadline_check(
            &objective,
            &beta,
            0,
            &lo.view(),
            &hi.view(),
            &a,
            &b,
            &mut expire,
        )
        .expect_err("X=1 must poll within the large constraint coefficient fold");
        assert!(matches!(err, NyError::DeadlineExceeded(_)));
        assert!(err.to_string().contains("constraint coefficient"));
        assert_eq!(polls, 4);
    }

    #[test]
    fn certificate_row_tail_expiry_cannot_return_a_result() {
        let objective = [1.0f32];
        let lo = arr2(&[[-1.0f32]]);
        let hi = arr2(&[[1.0f32]]);
        let a = Array3::<f32>::zeros((1, 0, 1));
        let b = Array2::<f32>::zeros((1, 0));
        let mut polls = 0usize;
        let mut expire_at_tail = || {
            polls += 1;
            polls >= 2
        };
        let err = certify_min_row_with_deadline_check(
            &objective,
            &[],
            0,
            &lo.view(),
            &hi.view(),
            &a,
            &b,
            &mut expire_at_tail,
        )
        .expect_err("expiry at the final row poll must refuse the result");
        assert!(err.to_string().contains("row completion"));
        assert_eq!(polls, 2);
    }
}
