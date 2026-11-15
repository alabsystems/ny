// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared sound assembly for arbitrary softmax axes. Every group keeps its
//! incoming error carrier, folds its constructed coefficient error on the
//! exact logit domain, and adds bias with outward arithmetic. The original
//! objective bias is seeded once; splitting a rounded bias is unnecessary.

use super::super::layer::SoftmaxLayer;
use super::super::weighted_support::Interval as I;
use crate::LinearBounds;
use ndarray::{Array1, Array2};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

/// Fold one softmax group's coefficient error over its logit box, leaving a
/// result that carries no error plane, for assemblies that copy coefficients
/// and biases only.
///
/// An ordered logit box with an infinite endpoint is a legal group domain: the
/// 1-D route answers it with a checked constant carrier, where a strict
/// `BoundedTensor::new` fold domain here refused every such group.
///
/// A box with a non-finite endpoint that also holds a NaN or an inverted pair
/// keeps that refusal and its kind, whether or not the result carries an error
/// plane. The 1-D route answers such a box too (its carrier holds for every
/// real logit), but it marks a failed or infeasible upstream bound, so the N-D
/// assemblies stay fail-closed on it. The kind matters: the backward dispatch
/// degrades `NumericalInstability` to IBP, while the `InvalidSpec` that
/// `new_allow_infinite` reports for an inverted box aborts the pass. An
/// all-finite box is validated exactly as the strict domain validated it.
///
/// The fold keeps a row's error plane when its penalty is not finite, so such
/// a row is degraded to the conservative row instead of losing its error in
/// the copy.
pub(super) fn fold_group_coeff_err(
    result: &mut LinearBounds,
    lower: Array1<f32>,
    upper: Array1<f32>,
) -> Result<()> {
    let non_finite = lower.iter().chain(upper.iter()).any(|v| !v.is_finite());
    let unordered = lower
        .iter()
        .zip(upper.iter())
        .any(|(&l, &u)| l.is_nan() || u.is_nan() || l > u);
    if non_finite && unordered {
        return Err(NyError::NumericalInstability(
            "softmax group logit box has a NaN or inverted non-finite endpoint".into(),
        ));
    }
    let domain = BoundedTensor::new_allow_infinite(lower.into_dyn(), upper.into_dyn())?;
    if !result.has_coeff_err() {
        return Ok(());
    }
    result.fold_coeff_err_over_box_eager_with_deadline(
        &domain,
        ny_core::instance_budget::deadline(),
    )?;
    result.discharge_coeff_err_to_conservative();
    Ok(())
}

pub(super) fn checkpoint() -> Result<()> {
    if ny_core::instance_budget::deadline().is_some_and(|d| std::time::Instant::now() >= d) {
        Err(NyError::DeadlineExceeded(
            "checked softmax propagation".into(),
        ))
    } else {
        Ok(())
    }
}

impl SoftmaxLayer {
    pub(super) fn propagate_grouped_sound(
        &self,
        bounds: &LinearBounds,
        pre: &BoundedTensor,
        axis: usize,
    ) -> Result<LinearBounds> {
        checkpoint()?;
        if bounds.num_inputs() != pre.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![pre.len()],
                got: vec![bounds.num_inputs()],
            });
        }
        let width = pre.shape()[axis];
        let outputs = bounds.num_outputs();
        if width == 0 || pre.is_empty() {
            return Ok(bounds.clone());
        }
        let inner = checked_shape_product(&pre.shape()[axis + 1..])
            .ok_or_else(|| NyError::InvalidSpec("softmax group stride overflow".into()))?;
        let outer = pre.len() / width / inner;
        let lo: Vec<f32> = pre.lower().iter().copied().collect();
        let hi: Vec<f32> = pre.upper().iter().copied().collect();
        let mut la = Array2::zeros((outputs, pre.len()));
        let mut ua = la.clone();
        let mut lb: Vec<I> = bounds
            .lower_b()
            .iter()
            .map(|&x| I::point(x as f64))
            .collect();
        let mut ub: Vec<I> = bounds
            .upper_b()
            .iter()
            .map(|&x| I::point(x as f64))
            .collect();
        for outside in 0..outer {
            for inside in 0..inner {
                checkpoint()?;
                let indices: Vec<usize> = (0..width)
                    .map(|j| (outside * width + j) * inner + inside)
                    .collect();
                let gl = Array1::from_iter(indices.iter().map(|&i| lo[i]));
                let gu = Array1::from_iter(indices.iter().map(|&i| hi[i]));
                let gla = Array2::from_shape_fn((outputs, width), |(o, j)| {
                    bounds.lower_a()[[o, indices[j]]]
                });
                let gua = Array2::from_shape_fn((outputs, width), |(o, j)| {
                    bounds.upper_a()[[o, indices[j]]]
                });
                let group = if bounds.lower_a_err().is_some() || bounds.upper_a_err().is_some() {
                    let gle = Array2::from_shape_fn((outputs, width), |(o, j)| {
                        bounds.lower_a_err().map_or(0.0, |e| e[[o, indices[j]]])
                    });
                    let gue = Array2::from_shape_fn((outputs, width), |(o, j)| {
                        bounds.upper_a_err().map_or(0.0, |e| e[[o, indices[j]]])
                    });
                    LinearBounds::new_or_conservative_with_err(
                        gla,
                        Array1::zeros(outputs),
                        gua,
                        Array1::zeros(outputs),
                        gle,
                        gue,
                    )?
                } else {
                    LinearBounds::new_or_conservative(
                        gla,
                        Array1::zeros(outputs),
                        gua,
                        Array1::zeros(outputs),
                    )?
                };
                let mut result = self.propagate_linear_with_bounds_1d_sound(&group, &gl, &gu)?;
                fold_group_coeff_err(&mut result, gl, gu)?;
                for o in 0..outputs {
                    for (j, &index) in indices.iter().enumerate() {
                        la[[o, index]] = result.lower_a()[[o, j]];
                        ua[[o, index]] = result.upper_a()[[o, j]];
                    }
                    lb[o] = lb[o].add(I::point(result.lower_b()[o] as f64));
                    ub[o] = ub[o].add(I::point(result.upper_b()[o] as f64));
                }
            }
        }
        checkpoint()?;
        LinearBounds::new_or_conservative(
            la,
            Array1::from_iter(lb.iter().map(|b| next_down_f32(b.lo as f32))),
            ua,
            Array1::from_iter(ub.iter().map(|b| next_up_f32(b.hi as f32))),
        )
    }
}
