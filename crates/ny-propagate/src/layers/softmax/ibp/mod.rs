// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Axis, Zip};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;

use crate::bounds::nan_propagating_max;
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

use super::layer::SoftmaxLayer;
use super::utils::sanitize_softmax_unit_bounds;

impl SoftmaxLayer {
    /// IBP propagation that accounts for a prepended restart axis.
    ///
    /// When restart batching adds a leading axis, positive stored axes (which
    /// used unbatched convention `axis - 1` at load time) must shift right by
    /// one to resolve against the correct sample-space dimension.
    ///
    /// Part of #4096.
    pub fn propagate_ibp_preserve_leading_axis(
        &self,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let ndim = input.shape().len();
        let axis = crate::layers::common::resolve_axis_i32_with_restored_leading_axis(
            self.axis, ndim, "Softmax",
        )?;
        Self::propagate_ibp_with_axis(input, axis)
    }

    /// Shared IBP implementation parameterized by resolved axis.
    fn propagate_ibp_with_axis(input: &BoundedTensor, axis: usize) -> Result<BoundedTensor> {
        let mut output_lower = input.lower().clone();
        let mut output_upper = input.upper().clone();

        Zip::from(output_lower.lanes_mut(Axis(axis)))
            .and(output_upper.lanes_mut(Axis(axis)))
            .and(input.lower().lanes(Axis(axis)))
            .and(input.upper().lanes(Axis(axis)))
            .for_each(
                |mut out_lower_lane, mut out_upper_lane, in_lower_lane, in_upper_lane| {
                    let mut ok = true;

                    let mut max_upper = f32::NEG_INFINITY;
                    for &u in in_upper_lane.iter() {
                        if u.is_nan() || u == f32::INFINITY {
                            ok = false;
                            break;
                        }
                        max_upper = nan_propagating_max(max_upper, u);
                    }
                    for &l in in_lower_lane.iter() {
                        if l.is_nan() || l == f32::INFINITY {
                            ok = false;
                            break;
                        }
                    }

                    if !ok || !max_upper.is_finite() {
                        out_lower_lane.fill(0.0);
                        out_upper_lane.fill(1.0);
                        return;
                    }

                    // Sanity guard: every score must be finite after the max-upper shift.
                    // (The per-ratio shift below uses its own dominant term, but this shared
                    // shift confirms no exp overflows; non-finite -> conservative [0, 1].)
                    // #2423/#3245: f64 throughout for precision + directed-rounding soundness.
                    let n = in_lower_lane.len();
                    let max_upper_f64 = max_upper as f64;
                    for i in 0..n {
                        let el = ((in_lower_lane[i] as f64) - max_upper_f64).exp();
                        let eu = ((in_upper_lane[i] as f64) - max_upper_f64).exp();
                        if !el.is_finite() || !eu.is_finite() {
                            ok = false;
                            break;
                        }
                    }

                    if !ok {
                        out_lower_lane.fill(0.0);
                        out_upper_lane.fill(1.0);
                        return;
                    }

                    // Second pass: per-coordinate monotone optimum with a PER-RATIO
                    // shift, in f64, cast to f32 with directed rounding. (#3245, #4231)
                    //
                    // The exact per-coordinate optimum is:
                    //   p_hi[i] = exp(u_i) / (exp(u_i) + sum_{j!=i} exp(l_j))
                    //   p_lo[i] = exp(l_i) / (exp(l_i) + sum_{j!=i} exp(u_j))
                    //
                    // SOUNDNESS (#4231): the previous code shifted EVERY exp by a single
                    // shared `max_upper` and added SOFTMAX_EPSILON to the denominator. In
                    // the underflow / large-score-gap regime (a key whose own scores sit
                    // ~745+ below the shared shift) that drives the numerator exp(u_i - M)
                    // to ~0 while SOFTMAX_EPSILON swamps the surviving sub-1e-12 terms, so
                    // p_hi of a REACHABLE key (one that can rise to win the row) collapses
                    // to ~0 and the IBP interval EXCLUDES the reachable true softmax — a
                    // FALSE certificate.
                    //
                    // Fix: shift each ratio by its OWN dominant term so the dominant exp is
                    // exactly exp(0)=1 and never underflows, and the denominator is always
                    // >= 1 (no epsilon needed, nothing to swamp). The shift cancels exactly
                    // in the ratio, so the NORMAL regime stays the exact corner optimum.
                    for i in 0..n {
                        let ui = in_upper_lane[i] as f64;
                        let li = in_lower_lane[i] as f64;

                        // ---- p_hi[i] = exp(u_i) / (exp(u_i) + sum_{j!=i} exp(l_j)) ----
                        // Shift reference = max(u_i, max_{j!=i} l_j): the dominant term of
                        // the numerator+denominator. Every shifted exponent is <= 0, so the
                        // dominant term is exp(0)=1 (no underflow) and denom_hi >= 1.
                        let mut ref_hi = ui;
                        for (j, &lj) in in_lower_lane.iter().enumerate() {
                            if j != i {
                                ref_hi = ref_hi.max(lj as f64);
                            }
                        }
                        let num_hi = (ui - ref_hi).exp();
                        let mut denom_hi = num_hi;
                        for (j, &lj) in in_lower_lane.iter().enumerate() {
                            if j != i {
                                denom_hi += (lj as f64 - ref_hi).exp();
                            }
                        }

                        // ---- p_lo[i] = exp(l_i) / (exp(l_i) + sum_{j!=i} exp(u_j)) ----
                        // Shift reference = max(l_i, max_{j!=i} u_j).
                        let mut ref_lo = li;
                        for (j, &uj) in in_upper_lane.iter().enumerate() {
                            if j != i {
                                ref_lo = ref_lo.max(uj as f64);
                            }
                        }
                        let num_lo = (li - ref_lo).exp();
                        let mut denom_lo = num_lo;
                        for (j, &uj) in in_upper_lane.iter().enumerate() {
                            if j != i {
                                denom_lo += (uj as f64 - ref_lo).exp();
                            }
                        }

                        // Outward rounding: p_lo rounds DOWN, p_hi rounds UP. The f64
                        // computation error is far below an f32 ULP for O(1) ratios, so the
                        // directed-rounding step (+ the sanitize margin) is a sound outward
                        // widening. Guard non-finite/non-positive denominators -> NaN ->
                        // sanitize widens to the conservative endpoint.
                        let raw_lower = if denom_lo.is_finite() && denom_lo > 0.0 {
                            next_down_f32((num_lo / denom_lo) as f32)
                        } else {
                            f32::NAN
                        };
                        let raw_upper = if denom_hi.is_finite() && denom_hi > 0.0 {
                            next_up_f32((num_hi / denom_hi) as f32)
                        } else {
                            f32::NAN
                        };

                        let (lower, upper) = sanitize_softmax_unit_bounds(raw_lower, raw_upper);
                        out_lower_lane[i] = lower;
                        out_upper_lane[i] = upper;
                    }
                },
            );

        BoundedTensor::new(output_lower, output_upper)
    }
}

impl BoundPropagation for SoftmaxLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        let axis = crate::layers::common::resolve_axis_i32(self.axis, ndim, "Softmax")?;
        Self::propagate_ibp_with_axis(input, axis)
    }

    /// Softmax CROWN propagation is not yet implemented.
    /// For now, use IBP propagation via propagate_ibp() for Softmax layers.
    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedConfiguration(
            "Softmax CROWN linear relaxation not implemented. \
             Use IBP propagation for networks with Softmax."
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        SoftmaxLayer::propagate_linear_with_bounds(
            self,
            bounds,
            pre_activation,
            self.soundness_mode(),
        )
    }
}

#[cfg(test)]
mod tests;
