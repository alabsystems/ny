// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD};
use ny_core::{NyError, Result, VerificationSoundnessMode};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use std::ops::Range;
use tracing::debug;

use super::super::common::BoundPropagation;
use super::bounds::constant_bounds_from_output;
use super::utils::{sanitize_softmax_unit_bounds, SOFTMAX_EPSILON};
use crate::bounds::nan_propagating_max;
use crate::LinearBounds;

mod batched;

/// Causal softmax layer for decoder attention.
///
/// In causal attention, position i can only attend to positions j where j <= i.
/// An optional sliding window further restricts attention to the most recent
/// `window_size + 1` keys for each query row.
/// This is implemented by applying a mask before softmax:
/// - Unmasked positions: softmax computed normally over the active range
/// - Masked positions outside that range: output is exactly 0
#[derive(Debug, Clone)]
pub struct CausalSoftmaxLayer {
    /// Dimension along which to apply softmax (default: -1)
    pub axis: i32,
    /// Use sound (no sampling) relaxation for CROWN.
    ///
    /// When true, CROWN linearization is disabled and we fall back to
    /// constant bounds derived from IBP outputs.
    pub sound: bool,
    /// Optional causal sliding window size. `None` means full causal attention.
    /// `Some(0)` means self-only attention.
    pub window_size: Option<usize>,
}

impl CausalSoftmaxLayer {
    /// Create a new Causal Softmax layer.
    pub fn new(axis: i32) -> Self {
        Self {
            axis,
            sound: true,
            window_size: None,
        }
    }

    /// Enable or disable sound (no sampling) CROWN mode.
    pub fn with_sound_mode(mut self, enabled: bool) -> Self {
        self.sound = enabled;
        self
    }

    /// Enable heuristic sampling-based CROWN relaxation (not provably sound).
    pub fn with_heuristic_sampling(mut self, enabled: bool) -> Self {
        self.sound = !enabled;
        self
    }

    /// Restrict each query row to a causal sliding window.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = Some(window_size);
        self
    }

    /// Returns the current verification soundness mode (Sound or Heuristic).
    pub fn soundness_mode(&self) -> VerificationSoundnessMode {
        if self.sound {
            VerificationSoundnessMode::Sound
        } else {
            VerificationSoundnessMode::Heuristic
        }
    }

    fn active_range(&self, row_idx: usize, seq_k: usize) -> Range<usize> {
        let active_end = row_idx.saturating_add(1).min(seq_k);
        let active_start = match self.window_size {
            Some(window_size) => active_end.saturating_sub(window_size.saturating_add(1)),
            None => 0,
        };
        active_start..active_end
    }

    fn ibp_row_bounds<F>(
        &self,
        row_idx: usize,
        seq_k: usize,
        mut get_bounds: F,
    ) -> Vec<(usize, f32, f32)>
    where
        F: FnMut(usize) -> (f32, f32),
    {
        let active = self.active_range(row_idx, seq_k);
        if active.start == active.end {
            return Vec::new();
        }

        let active_len = active.end - active.start;
        let mut lower_vals = Vec::with_capacity(active_len);
        let mut upper_vals = Vec::with_capacity(active_len);
        let mut max_upper = f32::NEG_INFINITY;

        for j in active.clone() {
            let (l, u) = get_bounds(j);
            if !l.is_finite() || !u.is_finite() {
                return active.map(|idx| (idx, 0.0, 1.0)).collect();
            }
            lower_vals.push(l as f64);
            upper_vals.push(u as f64);
            max_upper = nan_propagating_max(max_upper, u);
        }

        if !max_upper.is_finite() {
            return active.map(|idx| (idx, 0.0, 1.0)).collect();
        }

        // Sanity guard: every score must be finite after the max-upper shift
        // (confirms no exp overflow); otherwise widen the active block to [0, 1].
        let max_upper_f64 = max_upper as f64;
        for offset in 0..active_len {
            let el = (lower_vals[offset] - max_upper_f64).exp();
            let eu = (upper_vals[offset] - max_upper_f64).exp();
            if !el.is_finite() || !eu.is_finite() {
                return active.map(|idx| (idx, 0.0, 1.0)).collect();
            }
        }

        // Per-coordinate monotone optimum with a PER-RATIO shift (#4231).
        //
        //   p_hi[i] = exp(u_i) / (exp(u_i) + sum_{j!=i} exp(l_j))
        //   p_lo[i] = exp(l_i) / (exp(l_i) + sum_{j!=i} exp(u_j))
        //
        // SOUNDNESS: a single shared max-upper shift plus SOFTMAX_EPSILON in the
        // denominator under-approximates p_hi of a REACHABLE key in the underflow /
        // large-score-gap regime (its numerator exp(u_i - M) collapses to ~0 and the
        // epsilon swamps the surviving sub-1e-12 terms), yielding a FALSE certificate.
        // Shifting each ratio by its OWN dominant term keeps the dominant exp at
        // exp(0)=1 (no underflow), makes the denominator >= 1 (no epsilon needed), and
        // leaves the NORMAL regime exact (the shift cancels in the ratio).
        active
            .enumerate()
            .map(|(offset, j)| {
                let li = lower_vals[offset];
                let ui = upper_vals[offset];

                // p_hi[i]: shift by max(u_i, max_{k!=i} l_k).
                let mut ref_hi = ui;
                for (k, &lk) in lower_vals.iter().enumerate() {
                    if k != offset {
                        ref_hi = ref_hi.max(lk);
                    }
                }
                let num_hi = (ui - ref_hi).exp();
                let mut denom_hi = num_hi;
                for (k, &lk) in lower_vals.iter().enumerate() {
                    if k != offset {
                        denom_hi += (lk - ref_hi).exp();
                    }
                }

                // p_lo[i]: shift by max(l_i, max_{k!=i} u_k).
                let mut ref_lo = li;
                for (k, &uk) in upper_vals.iter().enumerate() {
                    if k != offset {
                        ref_lo = ref_lo.max(uk);
                    }
                }
                let num_lo = (li - ref_lo).exp();
                let mut denom_lo = num_lo;
                for (k, &uk) in upper_vals.iter().enumerate() {
                    if k != offset {
                        denom_lo += (uk - ref_lo).exp();
                    }
                }

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
                let (lb, ub) = sanitize_softmax_unit_bounds(raw_lower, raw_upper);
                (j, lb, ub)
            })
            .collect()
    }
}

impl BoundPropagation for CausalSoftmaxLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();

        // Causal softmax requires at least 2D for attention pattern
        if ndim < 2 {
            return Err(NyError::InvalidSpec(format!(
                "Causal softmax requires at least 2D input, got {}D",
                ndim
            )));
        }

        // Validate axis even though CausalSoftmax always operates on last 2 dims
        crate::layers::common::resolve_axis_i32(self.axis, ndim, "CausalSoftmax")?;

        // Causal mask requires the two last dimensions to form the attention matrix
        // Shape: [..., seq_q, seq_k] where causal means j <= i for position i
        let seq_q = shape[ndim - 2];
        let seq_k = shape[ndim - 1];

        if seq_q > seq_k {
            return Err(NyError::InvalidSpec(format!(
                "Causal softmax requires seq_q ({}) <= seq_k ({})",
                seq_q, seq_k
            )));
        }

        let mut output_lower = ArrayD::<f32>::zeros(input.lower().raw_dim());
        let mut output_upper = ArrayD::<f32>::zeros(input.upper().raw_dim());

        // Process based on dimensionality
        match ndim {
            2 => {
                // 2D: [seq_q, seq_k]
                for i in 0..seq_q {
                    for (j, lb, ub) in self.ibp_row_bounds(i, seq_k, |j| {
                        (input.lower()[[i, j]], input.upper()[[i, j]])
                    }) {
                        output_lower[[i, j]] = lb;
                        output_upper[[i, j]] = ub;
                    }
                }
            }
            3 => {
                // 3D: [batch, seq_q, seq_k]
                let batch = shape[0];
                for b in 0..batch {
                    for i in 0..seq_q {
                        for (j, lb, ub) in self.ibp_row_bounds(i, seq_k, |j| {
                            (input.lower()[[b, i, j]], input.upper()[[b, i, j]])
                        }) {
                            output_lower[[b, i, j]] = lb;
                            output_upper[[b, i, j]] = ub;
                        }
                    }
                }
            }
            4 => {
                // 4D: [batch, heads, seq_q, seq_k]
                let batch = shape[0];
                let heads = shape[1];
                for b in 0..batch {
                    for h in 0..heads {
                        for i in 0..seq_q {
                            for (j, lb, ub) in self.ibp_row_bounds(i, seq_k, |j| {
                                (input.lower()[[b, h, i, j]], input.upper()[[b, h, i, j]])
                            }) {
                                output_lower[[b, h, i, j]] = lb;
                                output_upper[[b, h, i, j]] = ub;
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(NyError::InvalidSpec(format!(
                    "Causal softmax not implemented for {}D tensors",
                    ndim
                )));
            }
        }

        // CausalSoftmax per-element guards above clamp all outputs to [0,1] via
        // sanitize_softmax_unit_bounds, so NaN repair is not needed here.
        // See #3060 for analysis of why this path is safe without repair.
        BoundedTensor::new(output_lower, output_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "CausalSoftmax is nonlinear — use propagate_linear_with_bounds with pre-activation bounds".to_string(),
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
        CausalSoftmaxLayer::propagate_linear_with_bounds(
            self,
            bounds,
            pre_activation,
            self.soundness_mode(),
        )
    }
}

impl CausalSoftmaxLayer {
    /// Evaluate causal softmax for a single row i.
    /// Input x has length seq_k, output has length seq_k.
    /// Positions outside the active causal/sliding window range are 0.
    fn eval_row(&self, x: &Array1<f32>, row_idx: usize) -> Array1<f32> {
        let seq_k = x.len();
        let mut out = Array1::zeros(seq_k);
        let active = self.active_range(row_idx, seq_k);

        if active.start == active.end {
            return out;
        }

        // Compute softmax over active positions
        let max_val = x
            .slice(ndarray::s![active.clone()])
            .fold(f32::NEG_INFINITY, |a, &b| nan_propagating_max(a, b));
        let mut sum_exp = 0.0_f32;
        for j in active.clone() {
            let e = (x[j] - max_val).exp();
            out[j] = e;
            sum_exp += e;
        }

        let inv_sum = 1.0 / (sum_exp + SOFTMAX_EPSILON);
        for j in active {
            out[j] *= inv_sum;
        }

        out
    }

    /// Compute Jacobian of causal softmax for a single row i.
    /// Returns a seq_k x seq_k matrix where J[output_j, input_k].
    /// Positions outside the active causal/sliding window range are 0.
    fn jacobian_row(&self, x: &Array1<f32>, row_idx: usize) -> Array2<f32> {
        let seq_k = x.len();
        let mut jac = Array2::zeros((seq_k, seq_k));
        let active = self.active_range(row_idx, seq_k);

        if active.start == active.end {
            return jac;
        }

        // Get softmax values for active positions
        let s = self.eval_row(x, row_idx);

        // Softmax Jacobian: J[j,k] = s[j] * (δ[j,k] - s[k]) on the active block.
        for j in active.clone() {
            for k in active.clone() {
                let delta = if j == k { 1.0 } else { 0.0 };
                jac[[j, k]] = s[j] * (delta - s[k]);
            }
        }

        jac
    }

    /// CROWN backward propagation through CausalSoftmax with pre-activation bounds.
    ///
    /// For causal softmax with shape [seq_q, seq_k]:
    /// - Row i applies softmax over its active causal/sliding-window range
    /// - Positions outside that range have output = 0 (masked)
    ///
    /// Uses Jacobian-based linear approximation at the interval center,
    /// with sampling to estimate approximation error (heuristic).
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        soundness: VerificationSoundnessMode,
    ) -> Result<LinearBounds> {
        let effective_soundness = if self.sound {
            if soundness == VerificationSoundnessMode::Heuristic {
                debug!("CausalSoftmax heuristic requested, but layer is in sound mode; using IBP constant bounds");
            }
            VerificationSoundnessMode::Sound
        } else {
            soundness
        };

        if effective_soundness == VerificationSoundnessMode::Sound {
            debug!("CausalSoftmax sound mode: using IBP-derived constant bounds");
            let output_bounds = self.propagate_ibp(pre_activation)?;
            return constant_bounds_from_output(bounds, &output_bounds);
        }
        debug!("CausalSoftmax heuristic mode: sampling-based bounds (not sound)");

        let shape = pre_activation.shape();
        let ndim = shape.len();

        // Causal softmax requires at least 2D
        if ndim < 2 {
            return Err(NyError::InvalidSpec(format!(
                "Causal softmax CROWN requires at least 2D input, got {}D",
                ndim
            )));
        }

        let seq_q = shape[ndim - 2];
        let seq_k = shape[ndim - 1];
        let total_size = seq_q * seq_k;

        if bounds.num_inputs() != total_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![bounds.num_inputs()],
            });
        }

        // Reject non-finite pre-activation bounds (#2591).
        // Returning bounds.clone() (identity passthrough) was unsound because
        // CausalSoftmax is not the identity function. Return NumericalInstability
        // so the network propagation falls back to IBP/constant bounds.
        if pre_activation.lower().iter().any(|&v| !v.is_finite())
            || pre_activation.upper().iter().any(|&v| !v.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "CausalSoftmax heuristic CROWN: non-finite pre-activation bounds".to_string(),
            ));
        }

        let num_outputs = bounds.num_outputs();

        let pre_lower_rows = pre_activation
            .lower()
            .view()
            .into_shape_with_order((seq_q, seq_k))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![total_size],
                got: pre_activation.lower().shape().to_vec(),
            })?;
        let pre_upper_rows = pre_activation
            .upper()
            .view()
            .into_shape_with_order((seq_q, seq_k))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![total_size],
                got: pre_activation.upper().shape().to_vec(),
            })?;

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, total_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, total_size));
        let mut new_lower_b = bounds.lower_b().to_owned();
        let mut new_upper_b = bounds.upper_b().to_owned();

        for row_idx in 0..seq_q {
            let row_start = row_idx * seq_k;
            let row_end = row_start + seq_k;
            let row_bounds = LinearBounds::new(
                bounds
                    .lower_a()
                    .slice(ndarray::s![.., row_start..row_end])
                    .to_owned(),
                Array1::zeros(num_outputs),
                bounds
                    .upper_a()
                    .slice(ndarray::s![.., row_start..row_end])
                    .to_owned(),
                Array1::zeros(num_outputs),
            )?;
            let row_result = self.propagate_linear_row_with_bounds_heuristic(
                &row_bounds,
                &pre_lower_rows.row(row_idx).to_owned(),
                &pre_upper_rows.row(row_idx).to_owned(),
                row_idx,
            )?;

            new_lower_a
                .slice_mut(ndarray::s![.., row_start..row_end])
                .assign(row_result.lower_a());
            new_upper_a
                .slice_mut(ndarray::s![.., row_start..row_end])
                .assign(row_result.upper_a());

            for out_idx in 0..num_outputs {
                // Each row helper already returns a conservative bias interval.
                // Re-apply directed rounding while summing the independent row
                // contributions so the scalar path stays conservative without
                // re-materializing the dense Jacobian (#1954).
                new_lower_b[out_idx] =
                    next_down_f32(new_lower_b[out_idx] + row_result.lower_b()[out_idx]);
                new_upper_b[out_idx] =
                    next_up_f32(new_upper_b[out_idx] + row_result.upper_b()[out_idx]);
            }
        }

        LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_1954;
