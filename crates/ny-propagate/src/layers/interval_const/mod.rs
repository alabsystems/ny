// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Layers whose frozen constant is known only up to a rigorous ENCLOSURE.
//!
//! # Why this is a separate `Layer` variant and not a field on `LinearLayer`
//!
//! This is the load-bearing design decision of the interval-carrier program,
//! and it is a soundness decision rather than a style one.
//!
//! A radius stored as an extra `Option<Array2<f32>>` field beside
//! `LinearLayer::weight` would leave every existing reader — the ~40 external
//! `weight()` call sites, the GPU and CUDA sound-CROWN residents, the graph-MIP
//! encoder that emits a HARD EQUALITY row straight into an exact-rational
//! Farkas route, the zonotope affine map, the neuron-elimination fold — quietly
//! computing with the MIDPOINT and dropping the width. That is not a
//! conservative failure. It is precisely the unsound point fold this carrier
//! exists to avoid, and on the MIP route it would produce a rigorously checked
//! infeasibility certificate for a network ny is not actually verifying: the
//! 807bf511 failure (nineteen false `unsat` verdicts) one layer down.
//!
//! A distinct variant inverts that default. Every exhaustive `match` over
//! `Layer` becomes a COMPILE ERROR until someone decides what to do, and every
//! non-exhaustive site already fails closed — the binary/ternary dispatches
//! return `UnsupportedOp`, the graph-MIP encoder's catch-all is `bail!`, the
//! batched-CROWN dispatcher's residual arm returns "not implemented". A
//! consumer nobody remembered costs a PROOF, never a false verdict.
//!
//! # What is carried, and the enclosure it stands for
//!
//! Both variants below are affine in the activation with an INTERVAL constant.
//! In each case the layer's semantics is a SET of functions, and every bound it
//! produces is valid for every member of that set — hence for the true one:
//!
//! * [`IntervalConstKind::Linear`]: `y = x W^T` for some
//!   `W in [Wc - R, Wc + R]` (`Wc`, `R` in `LinearLayer`'s `(out, in)` layout,
//!   `R >= 0`).
//! * [`IntervalConstKind::Offset`]: `y = x + c` for some
//!   `c in [center - radius, center + radius]`, broadcast ONNX-style. A
//!   subtraction `x - k` is carried here with `center = -k`; negation is exact.
//!
//! # The bound rules, and why each only widens
//!
//! **IBP, Linear.** Run the existing exact-midpoint IBP, then widen by
//! `R * m` with `m_j = max(|x_l[j]|, |x_u[j]|)`, accumulated in f64 and rounded
//! outward. For every `W` in the box and every `x` in the box,
//! `|(W - Wc) x|_i <= sum_j R[i,j] * m[j]` termwise, so the widened interval
//! contains `W x`. Widening AFTER the midpoint call is what makes the reuse
//! safe even though that call may intersect with an L2 Cauchy-Schwarz row
//! bound: the intersection still upper-bounds `Wc x`, and the widening is added
//! on top of whatever survived it.
//!
//! **IBP, Offset.** `[x_l + center - radius, x_u + center + radius]` with
//! directed binary32 addition. Immediate.
//!
//! **CROWN backward, Linear.** Run the existing exact-midpoint backward, which
//! stores `A Wc` and its own certified `gamma_n * S` coefficient error. Then ADD
//! `(|A| + err_in) R` into the ALREADY EXISTING certified per-coefficient error
//! channel `LinearBounds::lower_a_err` / `upper_a_err`, whose documented
//! invariant is exactly what is needed:
//! `|stored - true_coeff| <= err`, with `true_coeff` the exact real coefficient.
//! For the true weight `W = Wc + dW`, `|dW| <= R`:
//! `|(A Wc)[i,j] - (A W)[i,j]| = |(A dW)[i,j]| <= (|A| R)[i,j]`, and the
//! incoming `err_in` covers the gap between the stored `A` and the true `A`.
//! The channel is then discharged at concretize by the pre-existing OUTWARD
//! `sum_j err[i,j] * max(|in_l[j]|, |in_u[j]|)` penalty. No new soundness
//! theory, no new discharge site, no new invariant — and the penalty TIGHTENS
//! under branch-and-bound as the domain box shrinks.
//!
//! **CROWN backward, Offset.** The coefficients are untouched (`A' = A`), so
//! there is no coefficient error to add at all. Only the bias moves:
//! `b_lower -= sum_j |A_lower[i,j]| * radius[j]` (rounded down) and
//! `b_upper += sum_j |A_upper[i,j]| * radius[j]` (rounded up), plus the exact
//! midpoint contribution. Since `A` does not change, no sign can cancel and no
//! corner is selected.
//!
//! # What is deliberately NOT implemented
//!
//! Batched CROWN, Patches CROWN, MIP encoding, the GPU/CUDA sound-CROWN
//! residents, zonotope, SDP and neuron elimination. Each of those either fails
//! to compile against the new variant or hits an existing fail-closed arm, and
//! that is the intended state: they degrade to the CPU scalar path, which is
//! where the enclosure is discharged correctly.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{add_down_f32, add_up_f32, next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;

use crate::layers::common::BoundPropagation;
use crate::layers::linear::LinearLayer;
use crate::LinearBounds;

/// Which interval-constant shape this layer carries.
#[derive(Debug, Clone)]
enum IntervalConstKind {
    /// `y = x W^T`, `W in [Wc - R, Wc + R]`.
    Linear {
        /// The midpoint layer. Every exact-path cache, faer matrix and kernel
        /// of the ordinary linear layer is reused verbatim for `Wc`. Boxed so
        /// this variant does not dwarf the offset one.
        center: Box<LinearLayer>,
        /// Non-negative half-width, same `(out, in)` layout as `Wc`.
        radius: Array2<f32>,
    },
    /// `y = x + c`, `c in [center - radius, center + radius]`, broadcast.
    Offset {
        /// Boxed only to keep the two variants comparable in size.
        center: Box<ArrayD<f32>>,
        radius: Box<ArrayD<f32>>,
        /// Flat element count of the activation this layer is built for.
        ///
        /// The CROWN backward below keeps `A' = A`, which is correct ONLY when
        /// the constant broadcasts INTO the activation rather than expanding
        /// it. `propagate_linear` sees the OUTPUT width (`A.ncols()`) and
        /// nothing else, so the equality it needs is pinned here, at
        /// construction, from the declared input shape — and checked on every
        /// backward call. A layer built without that shape is refused outright.
        input_len: usize,
    },
}

/// An affine layer whose constant operand is an interval rather than a point.
#[derive(Debug, Clone)]
pub struct IntervalConstLayer {
    kind: IntervalConstKind,
}

impl IntervalConstLayer {
    /// Build `y = x W^T` with `W` enclosed by `center +- radius`.
    ///
    /// `radius` must match `center`'s shape, be finite and non-negative, and be
    /// nonzero somewhere — a zero radius is the exact case and must be built as
    /// an ordinary [`LinearLayer`] so it stays indistinguishable from today.
    pub fn try_new_linear(center: LinearLayer, radius: Array2<f32>) -> Result<Self> {
        if radius.shape() != [center.out_features(), center.in_features()] {
            return Err(NyError::ShapeMismatch {
                expected: vec![center.out_features(), center.in_features()],
                got: radius.shape().to_vec(),
            });
        }
        if !radius.iter().all(|r| r.is_finite() && *r >= 0.0) {
            return Err(NyError::InvalidSpec(
                "interval Linear radius must be finite and non-negative".to_string(),
            ));
        }
        if radius.iter().all(|r| *r == 0.0) {
            return Err(NyError::InvalidSpec(
                "interval Linear radius is identically zero — build a plain Linear layer"
                    .to_string(),
            ));
        }
        Ok(Self {
            kind: IntervalConstKind::Linear {
                center: Box::new(center),
                radius,
            },
        })
    }

    /// Build `y = x + c` with `c` enclosed by `center +- radius`.
    ///
    /// A subtraction is carried by negating `center` at the call site.
    pub fn try_new_offset(
        center: ArrayD<f32>,
        radius: ArrayD<f32>,
        input_len: usize,
    ) -> Result<Self> {
        if center.shape() != radius.shape() {
            return Err(NyError::ShapeMismatch {
                expected: center.shape().to_vec(),
                got: radius.shape().to_vec(),
            });
        }
        if !center.iter().all(|c| c.is_finite()) {
            return Err(NyError::InvalidSpec(
                "interval Offset center must be finite".to_string(),
            ));
        }
        if !radius.iter().all(|r| r.is_finite() && *r >= 0.0) {
            return Err(NyError::InvalidSpec(
                "interval Offset radius must be finite and non-negative".to_string(),
            ));
        }
        if radius.iter().all(|r| *r == 0.0) {
            return Err(NyError::InvalidSpec(
                "interval Offset radius is identically zero — build a plain constant layer"
                    .to_string(),
            ));
        }
        // The CROWN backward below keeps `A' = A`, which is only correct when
        // the constant broadcasts INTO the activation rather than expanding it.
        // Reject a radius carrying a length-1 axis unless it is a scalar: with
        // no length-1 axis, the constant's shape is exactly a trailing suffix of
        // the output shape, which is what the flat index mapping assumes.
        if radius.ndim() > 0 && radius.len() > 1 && radius.shape().contains(&1) {
            return Err(NyError::UnsupportedOp(
                "interval Offset with an interior broadcast axis is not carried".to_string(),
            ));
        }
        if input_len == 0 || !input_len.is_multiple_of(radius.len().max(1)) {
            return Err(NyError::UnsupportedOp(format!(
                "interval Offset constant of {} elements does not tile an activation of {} \
                 elements; the backward step would not be an identity on the coefficients",
                radius.len(),
                input_len
            )));
        }
        Ok(Self {
            kind: IntervalConstKind::Offset {
                center: Box::new(center),
                radius: Box::new(radius),
                input_len,
            },
        })
    }

    /// Human-readable discriminator for diagnostics.
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self.kind {
            IntervalConstKind::Linear { .. } => "IntervalLinear",
            IntervalConstKind::Offset { .. } => "IntervalOffset",
        }
    }

    /// Largest half-width carried, for diagnostics.
    #[must_use]
    pub fn max_radius(&self) -> f32 {
        match &self.kind {
            IntervalConstKind::Linear { radius, .. } => {
                radius.iter().copied().fold(0.0_f32, f32::max)
            }
            IntervalConstKind::Offset { radius, .. } => {
                radius.iter().copied().fold(0.0_f32, f32::max)
            }
        }
    }

    fn ibp_linear(
        center: &LinearLayer,
        radius: &Array2<f32>,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let midpoint = center.propagate_ibp(input)?;
        let in_features = center.in_features();
        let out_features = center.out_features();
        let shape = input.shape();
        if shape.last().copied() != Some(in_features) {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_features],
                got: shape.to_vec(),
            });
        }
        let rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

        let lower = input.lower().as_standard_layout().into_owned();
        let upper = input.upper().as_standard_layout().into_owned();
        let lower = lower
            .as_slice()
            .ok_or_else(|| NyError::InvalidSpec("interval Linear IBP: input not dense".into()))?;
        let upper = upper
            .as_slice()
            .ok_or_else(|| NyError::InvalidSpec("interval Linear IBP: input not dense".into()))?;

        // delta[r, o] = sum_j R[o, j] * max(|x_l|, |x_u|)[r, j], f64 with the
        // final cast pushed UP: the widening must never be smaller than asked.
        let mut delta = vec![0.0_f32; rows * out_features];
        for r in 0..rows {
            let base = r * in_features;
            for o in 0..out_features {
                let mut acc = 0.0_f64;
                for j in 0..in_features {
                    let m = f64::from(lower[base + j].abs().max(upper[base + j].abs()));
                    acc += f64::from(radius[[o, j]]) * m;
                }
                delta[r * out_features + o] = next_up_f32(acc as f32);
            }
        }

        let mut widened_lower = midpoint.lower().clone();
        let mut widened_upper = midpoint.upper().clone();
        if widened_lower.len() != delta.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![delta.len()],
                got: vec![widened_lower.len()],
            });
        }
        for (out, d) in widened_lower.iter_mut().zip(delta.iter()) {
            *out = add_down_f32(*out, -*d);
        }
        for (out, d) in widened_upper.iter_mut().zip(delta.iter()) {
            *out = add_up_f32(*out, *d);
        }
        BoundedTensor::new(widened_lower, widened_upper)
    }

    fn ibp_offset(
        center: &ArrayD<f32>,
        radius: &ArrayD<f32>,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let shape = IxDyn(input.shape());
        let center = center.broadcast(shape.clone()).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "interval Offset constant {:?} does not broadcast INTO input {:?}; expanding \
                 the activation is not carried",
                center.shape(),
                input.shape()
            ))
        })?;
        let radius = radius
            .broadcast(shape)
            .ok_or_else(|| NyError::InvalidSpec("interval Offset radius broadcast".into()))?;
        let lower = ndarray::Zip::from(input.lower())
            .and(&center)
            .and(&radius)
            .map_collect(|&x, &c, &r| add_down_f32(add_down_f32(x, c), -r));
        let upper = ndarray::Zip::from(input.upper())
            .and(&center)
            .and(&radius)
            .map_collect(|&x, &c, &r| add_up_f32(add_up_f32(x, c), r));
        BoundedTensor::new(lower, upper)
    }

    fn crown_linear<'a>(
        center: &LinearLayer,
        radius: &Array2<f32>,
        bounds: &'a LinearBounds,
    ) -> Result<Cow<'a, LinearBounds>> {
        let mut composed = center.propagate_linear(bounds)?.into_owned();
        let a_lower = bounds.lower_a();
        let a_upper = bounds.upper_a();
        if a_lower.ncols() != center.out_features() {
            return Err(NyError::ShapeMismatch {
                expected: vec![center.out_features()],
                got: vec![a_lower.ncols()],
            });
        }
        // (|A| + err_in) * R, rounded UP. `err_in` covers the gap between the
        // stored incoming coefficient and the true one, so the product bounds
        // |A_true * dW| for every dW inside the radius.
        let discharge = |a: &Array2<f32>, err: Option<&Array2<f32>>| -> Array2<f32> {
            let rows = a.nrows();
            let cols = radius.ncols();
            let mut out = Array2::<f32>::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    let mut acc = 0.0_f64;
                    for k in 0..a.ncols() {
                        let mag = f64::from(a[[i, k]].abs())
                            + err.map_or(0.0, |e| f64::from(e[[i, k]].abs()));
                        acc += mag * f64::from(radius[[k, j]]);
                    }
                    out[[i, j]] = next_up_f32(acc as f32);
                }
            }
            out
        };
        let extra_lower = discharge(a_lower, bounds.lower_a_err());
        let extra_upper = discharge(a_upper, bounds.upper_a_err());

        let base_lower = composed
            .lower_a_err()
            .cloned()
            .unwrap_or_else(|| Array2::zeros(extra_lower.raw_dim()));
        let base_upper = composed
            .upper_a_err()
            .cloned()
            .unwrap_or_else(|| Array2::zeros(extra_upper.raw_dim()));
        if base_lower.shape() != extra_lower.shape() || base_upper.shape() != extra_upper.shape() {
            return Err(NyError::ShapeMismatch {
                expected: extra_lower.shape().to_vec(),
                got: base_lower.shape().to_vec(),
            });
        }
        let total_lower = ndarray::Zip::from(&base_lower)
            .and(&extra_lower)
            .map_collect(|&b, &e| add_up_f32(b, e));
        let total_upper = ndarray::Zip::from(&base_upper)
            .and(&extra_upper)
            .map_collect(|&b, &e| add_up_f32(b, e));
        composed.set_coeff_err(total_lower, total_upper);
        Ok(Cow::Owned(composed))
    }

    fn crown_offset<'a>(
        center: &ArrayD<f32>,
        radius: &ArrayD<f32>,
        input_len: usize,
        bounds: &'a LinearBounds,
    ) -> Result<Cow<'a, LinearBounds>> {
        let layer_dim = bounds.lower_a().ncols();
        if layer_dim != input_len {
            // The constant EXPANDED the activation, so the backward step is a
            // column reduction rather than an identity. Refuse.
            return Err(NyError::UnsupportedOp(format!(
                "interval Offset CROWN: output width {layer_dim} differs from the activation \
                 width {input_len}; an expanding offset is not carried"
            )));
        }
        let n = radius.len();
        if n == 0 || !layer_dim.is_multiple_of(n) {
            return Err(NyError::UnsupportedOp(format!(
                "interval Offset CROWN: constant of {n} elements does not tile a layer \
                 dimension of {layer_dim}"
            )));
        }
        let center_flat: Vec<f32> = center.iter().copied().collect();
        let radius_flat: Vec<f32> = radius.iter().copied().collect();

        let mut lower_b = bounds.lower_b().clone();
        let mut upper_b = bounds.upper_b().clone();
        for i in 0..bounds.num_outputs() {
            // Exact midpoint contribution in f64, plus the radius penalty; the
            // f32 cast is the only rounding and it is directed outward.
            let mut mid_lower = 0.0_f64;
            let mut mid_upper = 0.0_f64;
            let mut pen_lower = 0.0_f64;
            let mut pen_upper = 0.0_f64;
            for j in 0..layer_dim {
                let al = f64::from(bounds.lower_a()[[i, j]]);
                let au = f64::from(bounds.upper_a()[[i, j]]);
                let c = f64::from(center_flat[j % n]);
                let r = f64::from(radius_flat[j % n]);
                mid_lower += al * c;
                mid_upper += au * c;
                pen_lower += al.abs() * r;
                pen_upper += au.abs() * r;
            }
            lower_b[i] = next_down_f32((f64::from(lower_b[i]) + mid_lower - pen_lower) as f32);
            upper_b[i] = next_up_f32((f64::from(upper_b[i]) + mid_upper + pen_upper) as f32);
        }
        // Coefficients are untouched, so the incoming certified error rides
        // through unchanged and no new coefficient error is created.
        Ok(Cow::Owned(
            LinearBounds::from_prevalidated_parts_with_optional_err(
                bounds.lower_a().clone(),
                lower_b,
                bounds.upper_a().clone(),
                upper_b,
                bounds.lower_a_err().cloned(),
                bounds.upper_a_err().cloned(),
            )?,
        ))
    }
}

impl BoundPropagation for IntervalConstLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        match &self.kind {
            IntervalConstKind::Linear { center, radius } => Self::ibp_linear(center, radius, input),
            IntervalConstKind::Offset {
                center,
                radius,
                input_len,
            } => {
                if input.len() != *input_len {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![*input_len],
                        got: vec![input.len()],
                    });
                }
                Self::ibp_offset(center, radius, input)
            }
        }
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        match &self.kind {
            IntervalConstKind::Linear { center, radius } => {
                Self::crown_linear(center, radius, bounds)
            }
            IntervalConstKind::Offset {
                center,
                radius,
                input_len,
            } => Self::crown_offset(center, radius, *input_len, bounds),
        }
    }
}

#[cfg(test)]
mod tests;
