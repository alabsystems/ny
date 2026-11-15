// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound softmax CROWN backward propagation via LSE affine bounds.
//!
//! Computes provably valid linear relaxations of softmax using the
//! log-sum-exp (LSE) decomposition: softmax_i(x) = exp(x_i) / sum_exp(x).
//! The numerator exp(x_i) is bounded by its chord (upper) and tangent (lower),
//! and the denominator sum_exp is bounded similarly, yielding affine bounds
//! on each softmax output.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::sync::OnceLock;

use crate::LinearBounds;

use super::super::layer::SoftmaxLayer;
use super::super::utils;
use super::super::weighted_support::{Interval as I, WeightedScoreBox};

const SOFTMAX_OBJECTIVE_ENVELOPE_ENV: &str = "NY_SOFTMAX_OBJECTIVE_ENVELOPE";

#[inline]
pub(super) fn softmax_objective_envelope_gate(raw: Option<&str>) -> bool {
    raw == Some("1")
}

#[inline]
fn softmax_objective_envelope_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = SOFTMAX_OBJECTIVE_ENVELOPE_TEST_OVERRIDE.with(|cell| cell.get()) {
        return enabled;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        softmax_objective_envelope_gate(
            std::env::var(SOFTMAX_OBJECTIVE_ENVELOPE_ENV)
                .ok()
                .as_deref(),
        )
    })
}

#[cfg(test)]
thread_local! {
    static SOFTMAX_OBJECTIVE_ENVELOPE_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Run a unit-test closure with a thread-local gate override. This exercises the
/// real 2-D, N-D, batched, and flat-grouped dispatch without mutating the
/// process environment seen by parallel tests.
#[cfg(test)]
pub(super) fn with_softmax_objective_envelope_for_test<T>(
    enabled: bool,
    f: impl FnOnce() -> T,
) -> T {
    struct Reset<'a> {
        cell: &'a std::cell::Cell<Option<bool>>,
        previous: Option<bool>,
    }

    impl Drop for Reset<'_> {
        fn drop(&mut self) {
            self.cell.set(self.previous);
        }
    }

    SOFTMAX_OBJECTIVE_ENVELOPE_TEST_OVERRIDE.with(|cell| {
        let reset = Reset {
            cell,
            previous: cell.replace(Some(enabled)),
        };
        let result = f();
        drop(reset);
        result
    })
}

impl SoftmaxLayer {
    pub(super) fn propagate_linear_with_bounds_1d_sound(
        &self,
        bounds: &LinearBounds,
        pre_lower: &Array1<f32>,
        pre_upper: &Array1<f32>,
    ) -> Result<LinearBounds> {
        super::grouped_sound::checkpoint()?;
        let num_neurons = pre_lower.len();
        if pre_upper.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![pre_upper.len()],
            });
        }
        if bounds.num_inputs() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }
        if num_neurons == 0 {
            return Ok(bounds.clone());
        }
        if bounds
            .num_outputs()
            .checked_mul(num_neurons)
            .and_then(|n| n.checked_mul(8))
            .is_none_or(|bytes| bytes > 128 * 1024 * 1024)
        {
            return Err(NyError::UnsupportedOp(
                "checked softmax output memory admission".into(),
            ));
        }

        let has_non_finite =
            pre_lower.iter().any(|&v| !v.is_finite()) || pre_upper.iter().any(|&v| !v.is_finite());
        if has_non_finite {
            let mut result = checked_constant_from_probability_box(
                bounds,
                &Array1::zeros(num_neurons),
                &Array1::ones(num_neurons),
            )?;
            charge_incoming_error(&mut result, bounds);
            return Ok(result);
        }

        let weighted = self.weighted_objective_bounds(bounds, pre_lower, pre_upper)?;
        // The scalar route stays linear in width; an intermediate identity
        // direction may still inherently require a large M-by-T result, but
        // no T-by-T LSE Jacobian is built on the vocabulary-sized route.
        if num_neurons
            .checked_mul(num_neurons)
            .is_none_or(|n| n > 4_000_000)
        {
            super::grouped_sound::checkpoint()?;
            return Ok(weighted);
        }
        let Some((lower_a, lower_b, upper_a, upper_b)) =
            self.softmax_lse_affine_bounds(pre_lower, pre_upper)
        else {
            super::grouped_sound::checkpoint()?;
            return Ok(weighted);
        };

        let mut lse_candidate =
            self.apply_affine_bounds(bounds, &lower_a, &lower_b, &upper_a, &upper_b)?;
        charge_incoming_error(&mut lse_candidate, bounds);
        let candidate = select_complete_rows(lse_candidate, weighted, pre_lower, pre_upper)?;
        super::grouped_sound::checkpoint()?;
        if !softmax_objective_envelope_enabled() {
            return Ok(candidate);
        }

        let result =
            self.select_softmax_objective_envelope(bounds, pre_lower, pre_upper, candidate)?;
        super::grouped_sound::checkpoint()?;
        Ok(result)
    }

    /// Checked fixed-direction support is active in every sound softmax route.
    /// It is a constant carrier, never a disguised affine slope or sampled bound.
    pub(super) fn weighted_objective_bounds(
        &self,
        bounds: &LinearBounds,
        lower: &Array1<f32>,
        upper: &Array1<f32>,
    ) -> Result<LinearBounds> {
        let l: Vec<f64> = lower.iter().map(|&x| x as f64).collect();
        let u: Vec<f64> = upper.iter().map(|&x| x as f64).collect();
        let score = WeightedScoreBox::new(&l, &u);
        super::grouped_sound::checkpoint()?;
        let score = score
            .ok_or_else(|| NyError::UnsupportedOp("invalid weighted softmax score box".into()))?;
        let mut lb = Array1::zeros(bounds.num_outputs());
        let mut ub = lb.clone();
        for row in 0..bounds.num_outputs() {
            super::grouped_sound::checkpoint()?;
            let lc: Vec<f64> = bounds
                .lower_a()
                .row(row)
                .iter()
                .map(|&x| x as f64)
                .collect();
            let uc: Vec<f64> = bounds
                .upper_a()
                .row(row)
                .iter()
                .map(|&x| x as f64)
                .collect();
            lb[row] = next_down_f32(
                I::point(bounds.lower_b()[row] as f64)
                    .add(I::point(score.lower(&lc)))
                    .lo as f32,
            );
            ub[row] = next_up_f32(
                I::point(bounds.upper_b()[row] as f64)
                    .add(I::point(score.upper(&uc)))
                    .hi as f32,
            );
        }
        let mut result = LinearBounds::new_or_conservative(
            Array2::zeros((bounds.num_outputs(), lower.len())),
            lb,
            Array2::zeros((bounds.num_outputs(), lower.len())),
            ub,
        )?;
        charge_incoming_error(&mut result, bounds);
        super::grouped_sound::checkpoint()?;
        Ok(result)
    }

    /// Select the stronger of two independently sound backward bounds for
    /// each incoming objective row and each bound direction.
    ///
    /// The LSE relaxation retains affine dependence on the logits. The second
    /// candidate first computes the exact coordinate-wise Softmax interval for
    /// this 1-D group, then composes the incoming objective with that box as a
    /// constant bound. Comparing their sound concretizations over the current
    /// logit box makes the choice local to precisely the domain where either
    /// candidate is used. A selected side is copied as one complete A row plus
    /// its bias; coefficients from different candidates are never blended.
    fn select_softmax_objective_envelope(
        &self,
        bounds: &LinearBounds,
        pre_lower: &Array1<f32>,
        pre_upper: &Array1<f32>,
        lse_candidate: LinearBounds,
    ) -> Result<LinearBounds> {
        let logit_box =
            BoundedTensor::new(pre_lower.clone().into_dyn(), pre_upper.clone().into_dyn())?;
        // This extracted group is always exactly 1-D. Resolve it explicitly as
        // axis 0; using self.propagate_ibp() here would incorrectly reinterpret
        // a positive axis retained from the original 2-D/N-D Softmax layer.
        let l: Vec<f64> = pre_lower.iter().map(|&x| x as f64).collect();
        let u: Vec<f64> = pre_upper.iter().map(|&x| x as f64).collect();
        let scores = WeightedScoreBox::new(&l, &u)
            .ok_or_else(|| NyError::UnsupportedOp("invalid softmax coordinate score box".into()))?;
        let mut lo = Array1::zeros(l.len());
        let mut hi = lo.clone();
        let mut direction = vec![0.0; l.len()];
        for i in 0..l.len() {
            direction[i] = 1.0;
            lo[i] = next_down_f32(scores.lower(&direction) as f32).max(0.0);
            hi[i] = next_up_f32(scores.upper(&direction) as f32).min(1.0);
            direction[i] = 0.0;
        }
        let mut constant_candidate = checked_constant_from_probability_box(bounds, &lo, &hi)?;
        charge_incoming_error(&mut constant_candidate, bounds);
        let _ = logit_box;
        select_complete_rows(lse_candidate, constant_candidate, pre_lower, pre_upper)
    }

    // Justification: The 4-tuple return (lower_A, lower_b, upper_A, upper_b) represents
    // the standard affine bound coefficient structure for softmax-via-LSE relaxation.
    // Extracting a struct would add overhead for a private method with one call site.
    #[allow(clippy::type_complexity)]
    /// The LSE affine bounds of one softmax group (row `i` of each matrix is
    /// output `i`). `pub(crate)` so the vit root α search copy
    /// (#vit-root-alpha) can carry the lane's own frozen relaxation.
    pub(crate) fn softmax_lse_affine_bounds(
        &self,
        pre_lower: &Array1<f32>,
        pre_upper: &Array1<f32>,
    ) -> Option<(Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>)> {
        super::super::checked_lse::affine_bounds(pre_lower, pre_upper)
    }

    pub(super) fn apply_affine_bounds(
        &self,
        bounds: &LinearBounds,
        lower_a: &Array2<f32>,
        lower_b: &Array1<f32>,
        upper_a: &Array2<f32>,
        upper_b: &Array1<f32>,
    ) -> Result<LinearBounds> {
        utils::apply_affine_bounds_f64(bounds, lower_a, lower_b, upper_a, upper_b)
    }
}

fn charge_incoming_error(result: &mut LinearBounds, incoming: &LinearBounds) {
    // p lies on the simplex, so sum err_i*p_i <= max err_i. This charge
    // represents the original objective's coefficient uncertainty; it cannot
    // be recovered later from the newly composed coefficient-error arrays.
    for row in 0..incoming.num_outputs() {
        let le = incoming
            .lower_a_err()
            .map_or(0.0, |e| e.row(row).iter().copied().fold(0.0_f32, f32::max));
        let ue = incoming
            .upper_a_err()
            .map_or(0.0, |e| e.row(row).iter().copied().fold(0.0_f32, f32::max));
        if le > 0.0 {
            result.lower_b[row] = next_down_f32(
                I::point(result.lower_b[row] as f64)
                    .sub(I::point(le as f64))
                    .lo as f32,
            );
        }
        if ue > 0.0 {
            result.upper_b[row] = next_up_f32(
                I::point(result.upper_b[row] as f64)
                    .add(I::point(ue as f64))
                    .hi as f32,
            );
        }
    }
}

fn checked_constant_from_probability_box(
    bounds: &LinearBounds,
    lower: &Array1<f32>,
    upper: &Array1<f32>,
) -> Result<LinearBounds> {
    let mut lb = Array1::zeros(bounds.num_outputs());
    let mut ub = lb.clone();
    for row in 0..bounds.num_outputs() {
        let mut lo = I::point(bounds.lower_b()[row] as f64);
        let mut hi = I::point(bounds.upper_b()[row] as f64);
        for j in 0..lower.len() {
            let p = I::new(lower[j] as f64, upper[j] as f64);
            lo = lo.add(I::point(bounds.lower_a()[[row, j]] as f64).mul(p));
            hi = hi.add(I::point(bounds.upper_a()[[row, j]] as f64).mul(p));
        }
        lb[row] = next_down_f32(lo.lo as f32);
        ub[row] = next_up_f32(hi.hi as f32);
    }
    LinearBounds::new_or_conservative(
        Array2::zeros((bounds.num_outputs(), lower.len())),
        lb,
        Array2::zeros((bounds.num_outputs(), lower.len())),
        ub,
    )
}

fn select_complete_rows(
    first: LinearBounds,
    second: LinearBounds,
    lower: &Array1<f32>,
    upper: &Array1<f32>,
) -> Result<LinearBounds> {
    let domain = BoundedTensor::new(lower.clone().into_dyn(), upper.clone().into_dyn())?;
    let fc = first.concretize_sound(&domain);
    let sc = second.concretize_sound(&domain);
    let mut la = first.lower_a().clone();
    let mut ua = first.upper_a().clone();
    let mut lb = first.lower_b().clone();
    let mut ub = first.upper_b().clone();
    let mut le = first
        .lower_a_err()
        .cloned()
        .unwrap_or_else(|| Array2::zeros(la.raw_dim()));
    let mut ue = first
        .upper_a_err()
        .cloned()
        .unwrap_or_else(|| Array2::zeros(ua.raw_dim()));
    for row in 0..first.num_outputs() {
        if sc.lower()[[row]] > fc.lower()[[row]] {
            la.row_mut(row).assign(&second.lower_a().row(row));
            lb[row] = second.lower_b()[row];
            if let Some(err) = second.lower_a_err() {
                le.row_mut(row).assign(&err.row(row));
            } else {
                le.row_mut(row).fill(0.0);
            }
        }
        if sc.upper()[[row]] < fc.upper()[[row]] {
            ua.row_mut(row).assign(&second.upper_a().row(row));
            ub[row] = second.upper_b()[row];
            if let Some(err) = second.upper_a_err() {
                ue.row_mut(row).assign(&err.row(row));
            } else {
                ue.row_mut(row).fill(0.0);
            }
        }
    }
    LinearBounds::new_or_conservative_with_err(la, lb, ua, ub, le, ue)
}
