// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SPSA (Simultaneous Perturbation Stochastic Approximation) gradient estimation.
//!
//! SPSA uses random perturbations to estimate gradients with only 2 function evaluations,
//! regardless of input dimension. This is much more efficient than finite differences
//! for high-dimensional inputs.

use ndarray::{Array1, ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::RngExt;

use crate::layers::Layer;
use crate::Network;

use super::eval::output_value;
use super::PgdAttacker;

const SPSA_DELTA_SCALE_MULTIPLIER: f32 = 4.0;
const SPSA_MAX_DELTA_RANGE_RATIO: f32 = 0.5;
const SPSA_ZERO_DIFF_EPS: f32 = 1e-8;

impl PgdAttacker<'_> {
    fn spsa_max_delta(&self, input_bounds: &BoundedTensor) -> f32 {
        let max_width = input_bounds
            .lower()
            .iter()
            .zip(input_bounds.upper().iter())
            .map(|(lower, upper)| upper - lower)
            .fold(0.0_f32, f32::max);
        (max_width * SPSA_MAX_DELTA_RANGE_RATIO).max(self.config.spsa_delta)
    }

    /// Estimate gradient using SPSA (baseline, test-only).
    ///
    /// Does NOT project probes into the input box. Used as a baseline comparison
    /// in tests to demonstrate the value of bounded and smooth-Sign variants.
    #[cfg(test)]
    pub(crate) fn estimate_gradient_spsa(
        &self,
        network: &Network,
        input: &ArrayD<f32>,
        output_idx: usize,
        rng: &mut StdRng,
    ) -> Result<(ArrayD<f32>, usize)> {
        let delta = self.config.spsa_delta;
        let n = input.len();
        let mut evals = 0;

        // Generate random perturbation vector (Bernoulli +/-1)
        let perturbation: Array1<f32> = (0..n)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let perturbation = perturbation
            .into_shape_with_order(IxDyn(input.shape()))
            .map_err(|err| {
                NyError::InvalidSpec(format!(
                    "SPSA perturbation reshape failed for input shape {:?}: {err}",
                    input.shape()
                ))
            })?;

        // Evaluate at x + delta * perturbation
        let input_plus = input + &perturbation * delta;
        let output_plus = self.evaluate(network, &input_plus)?;
        evals += 1;

        // Evaluate at x - delta * perturbation
        let input_minus = input - &perturbation * delta;
        let output_minus = self.evaluate(network, &input_minus)?;
        evals += 1;

        // SPSA gradient estimate: (f(x+) - f(x-)) / (2 * delta * perturbation)
        let y_plus = output_value(&output_plus, output_idx)?;
        let y_minus = output_value(&output_minus, output_idx)?;
        let diff = y_plus - y_minus;

        // Gradient estimate for each dimension
        let gradient = &perturbation * (diff / (2.0 * delta));

        Ok((gradient, evals))
    }

    /// Estimate gradient using SPSA while keeping probes inside the input box.
    ///
    /// Two strategies depending on network type (#3769):
    ///
    /// **Networks with Sign layers (BNNs):** Use smooth Sign relaxation
    /// (`tanh(β*x)` instead of `sign(x)`) for SPSA probes. This gives a
    /// meaningful gradient everywhere, eliminating the need for multi-scale
    /// delta growth. The gradient points toward Sign thresholds, guiding PGD
    /// toward counterexamples that cross decision boundaries.
    ///
    /// **Other networks:** Use multi-scale SPSA delta growth. Small radii are
    /// cheap and work well on smooth networks, but piecewise-constant
    /// activations can return all-zero finite differences. Grow the radius
    /// geometrically inside the legal input box before giving up.
    pub(in crate::pgd_attack) fn estimate_gradient_spsa_with_bounds(
        &self,
        network: &Network,
        input: &ArrayD<f32>,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        rng: &mut StdRng,
    ) -> Result<(ArrayD<f32>, usize)> {
        let has_sign = network.layers.iter().any(|l| matches!(l, Layer::Sign(_)));
        if has_sign {
            return self.estimate_gradient_spsa_smooth_sign(
                network,
                input,
                input_bounds,
                output_idx,
                rng,
            );
        }

        let base_delta = self.config.spsa_delta;
        let n = input.len();
        let mut evals = 0;

        let perturbation: Array1<f32> = (0..n)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let perturbation = perturbation
            .into_shape_with_order(IxDyn(input.shape()))
            .map_err(|err| {
                NyError::InvalidSpec(format!(
                    "SPSA perturbation reshape failed for input shape {:?}: {err}",
                    input.shape()
                ))
            })?;

        let max_delta = self.spsa_max_delta(input_bounds);

        let mut delta = base_delta;
        let mut last_gradient;

        loop {
            let input_plus = self.project(&(input + &perturbation * delta), input_bounds);
            let input_minus = self.project(&(input - &perturbation * delta), input_bounds);
            let output_plus = self.evaluate(network, &input_plus)?;
            evals += 1;
            let output_minus = self.evaluate(network, &input_minus)?;
            evals += 1;

            let y_plus = output_value(&output_plus, output_idx)?;
            let y_minus = output_value(&output_minus, output_idx)?;
            let diff = y_plus - y_minus;
            last_gradient = &perturbation * (diff / (2.0 * delta));

            if !diff.is_finite() || diff.abs() > SPSA_ZERO_DIFF_EPS || delta >= max_delta {
                return Ok((last_gradient, evals));
            }

            let next_delta = (delta * SPSA_DELTA_SCALE_MULTIPLIER).min(max_delta);
            if (next_delta - delta).abs() <= f32::EPSILON {
                break;
            }
            delta = next_delta;
        }

        Ok((last_gradient, evals))
    }

    /// SPSA gradient estimation using smooth Sign relaxation (#3769).
    ///
    /// For networks with Sign layers, finite-difference through the original
    /// discrete Sign gives zero gradient unless the perturbation happens to cross
    /// a threshold. Evaluating through `tanh(β*x)` instead gives a continuous
    /// gradient that points toward Sign thresholds at all input values.
    ///
    /// Uses only 2 network evaluations (no multi-scale growth needed since the
    /// smooth approximation always produces nonzero finite differences near
    /// thresholds).
    pub(super) fn estimate_gradient_spsa_smooth_sign(
        &self,
        network: &Network,
        input: &ArrayD<f32>,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        rng: &mut StdRng,
    ) -> Result<(ArrayD<f32>, usize)> {
        let delta = self.config.spsa_delta;
        let n = input.len();

        let perturbation: Array1<f32> = (0..n)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let perturbation = perturbation
            .into_shape_with_order(IxDyn(input.shape()))
            .map_err(|err| {
                NyError::InvalidSpec(format!(
                    "SPSA perturbation reshape failed for input shape {:?}: {err}",
                    input.shape()
                ))
            })?;

        let input_plus = self.project(&(input + &perturbation * delta), input_bounds);
        let input_minus = self.project(&(input - &perturbation * delta), input_bounds);

        // Evaluate through the configured Sign surrogate for gradient signal
        // (tanh smooth relaxation by default, plain STE when
        // `surrogate_sign_gradient` is set — #surrogate-sign).
        let surrogate = self.attack_sign_surrogate();
        let output_plus = self.evaluate_sign_surrogate(network, &input_plus, surrogate)?;
        let output_minus = self.evaluate_sign_surrogate(network, &input_minus, surrogate)?;

        let y_plus = output_value(&output_plus, output_idx)?;
        let y_minus = output_value(&output_minus, output_idx)?;
        let diff = y_plus - y_minus;

        let gradient = &perturbation * (diff / (2.0 * delta));

        Ok((gradient, 2))
    }

    /// Batched SPSA gradient estimation with multi-scale delta growth.
    pub(super) fn estimate_gradient_spsa_batch_with_bounds(
        &self,
        network: &Network,
        inputs: &ArrayD<f32>,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        rngs: &mut [StdRng],
    ) -> Result<(ArrayD<f32>, usize)> {
        // Dispatch to smooth Sign path for BNN networks (#3769)
        let has_sign = network.layers.iter().any(|l| matches!(l, Layer::Sign(_)));
        if has_sign {
            return self.estimate_gradient_spsa_batch_smooth_sign(
                network,
                inputs,
                input_bounds,
                output_idx,
                rngs,
            );
        }

        let Some((&batch_size, input_shape)) = inputs.shape().split_first() else {
            return Err(NyError::InvalidSpec(
                "batched SPSA requires input shape [N, ...]".to_string(),
            ));
        };
        if batch_size != rngs.len() {
            return Err(NyError::InvalidSpec(format!(
                "batched SPSA RNG mismatch: {} inputs but {} RNGs",
                batch_size,
                rngs.len(),
            )));
        }

        let features: usize = input_shape.iter().product();
        let base_delta = self.config.spsa_delta;
        let max_delta = self.spsa_max_delta(input_bounds);
        let mut evals = 0;
        let mut gradient_batch = ArrayD::zeros(IxDyn(inputs.shape()));
        let mut delta = base_delta;

        let mut perturbation_batch = ArrayD::zeros(IxDyn(inputs.shape()));
        for (batch_idx, rng) in rngs.iter_mut().enumerate() {
            let perturbation: Array1<f32> = (0..features)
                .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
                .collect();
            let perturbation = perturbation
                .into_shape_with_order(IxDyn(input_shape))
                .map_err(|err| {
                    NyError::InvalidSpec(format!(
                        "batched SPSA perturbation reshape failed for input shape {:?}: {err}",
                        input_shape,
                    ))
                })?;
            perturbation_batch
                .index_axis_mut(Axis(0), batch_idx)
                .assign(&perturbation);
        }

        let mut resolved = vec![false; batch_size];
        loop {
            let input_plus =
                self.project_batch(&(inputs + &(&perturbation_batch * delta)), input_bounds)?;
            let input_minus =
                self.project_batch(&(inputs - &(&perturbation_batch * delta)), input_bounds)?;

            let stacked = ndarray::concatenate(Axis(0), &[input_plus.view(), input_minus.view()])
                .map_err(|e| {
                NyError::InternalError(format!("PGD batched: SPSA concat failed: {e}"))
            })?;
            let outputs = self.evaluate_batch(network, &stacked)?;
            evals += 2 * batch_size;

            let (output_plus, output_minus) = outputs.view().split_at(Axis(0), batch_size);
            let mut unresolved = 0usize;

            for (batch_idx, is_resolved) in resolved.iter_mut().enumerate() {
                if *is_resolved {
                    continue;
                }

                let y_plus = output_plus
                    .index_axis(Axis(0), batch_idx)
                    .iter()
                    .nth(output_idx)
                    .copied()
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "output_idx {} out of range for output with {} elements",
                            output_idx,
                            output_plus.index_axis(Axis(0), batch_idx).len()
                        ))
                    })?;
                let y_minus = output_minus
                    .index_axis(Axis(0), batch_idx)
                    .iter()
                    .nth(output_idx)
                    .copied()
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "output_idx {} out of range for output with {} elements",
                            output_idx,
                            output_minus.index_axis(Axis(0), batch_idx).len()
                        ))
                    })?;

                let diff = y_plus - y_minus;
                let gradient = perturbation_batch.index_axis(Axis(0), batch_idx).to_owned()
                    * (diff / (2.0 * delta));
                gradient_batch
                    .index_axis_mut(Axis(0), batch_idx)
                    .assign(&gradient);

                if !diff.is_finite() || diff.abs() > SPSA_ZERO_DIFF_EPS || delta >= max_delta {
                    *is_resolved = true;
                } else {
                    unresolved += 1;
                }
            }

            if unresolved == 0 {
                break;
            }

            let next_delta = (delta * SPSA_DELTA_SCALE_MULTIPLIER).min(max_delta);
            if (next_delta - delta).abs() <= f32::EPSILON {
                break;
            }
            delta = next_delta;
        }

        Ok((gradient_batch, evals))
    }

    /// Batched SPSA with smooth Sign relaxation (#3769).
    ///
    /// Single-pass: evaluate all restarts through `tanh(β*x)` in one batch,
    /// compute finite-difference gradient. No multi-scale delta growth needed.
    fn estimate_gradient_spsa_batch_smooth_sign(
        &self,
        network: &Network,
        inputs: &ArrayD<f32>,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        rngs: &mut [StdRng],
    ) -> Result<(ArrayD<f32>, usize)> {
        let Some((&batch_size, input_shape)) = inputs.shape().split_first() else {
            return Err(NyError::InvalidSpec(
                "batched SPSA requires input shape [N, ...]".to_string(),
            ));
        };
        if batch_size != rngs.len() {
            return Err(NyError::InvalidSpec(format!(
                "batched SPSA RNG mismatch: {} inputs but {} RNGs",
                batch_size,
                rngs.len(),
            )));
        }

        let features: usize = input_shape.iter().product();
        let delta = self.config.spsa_delta;
        let mut gradient_batch = ArrayD::zeros(IxDyn(inputs.shape()));

        let mut perturbation_batch = ArrayD::zeros(IxDyn(inputs.shape()));
        for (batch_idx, rng) in rngs.iter_mut().enumerate() {
            let perturbation: Array1<f32> = (0..features)
                .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
                .collect();
            let perturbation = perturbation
                .into_shape_with_order(IxDyn(input_shape))
                .map_err(|err| {
                    NyError::InvalidSpec(format!(
                        "batched SPSA perturbation reshape failed for input shape {:?}: {err}",
                        input_shape,
                    ))
                })?;
            perturbation_batch
                .index_axis_mut(Axis(0), batch_idx)
                .assign(&perturbation);
        }

        let input_plus =
            self.project_batch(&(inputs + &(&perturbation_batch * delta)), input_bounds)?;
        let input_minus =
            self.project_batch(&(inputs - &(&perturbation_batch * delta)), input_bounds)?;

        let stacked = ndarray::concatenate(Axis(0), &[input_plus.view(), input_minus.view()])
            .map_err(|e| {
                NyError::InternalError(format!("PGD batched smooth Sign: SPSA concat failed: {e}"))
            })?;

        // Evaluate through the configured Sign surrogate (#surrogate-sign)
        let outputs =
            self.evaluate_batch_sign_surrogate(network, &stacked, self.attack_sign_surrogate())?;

        let (output_plus, output_minus) = outputs.view().split_at(Axis(0), batch_size);

        for batch_idx in 0..batch_size {
            let y_plus = output_plus
                .index_axis(Axis(0), batch_idx)
                .iter()
                .nth(output_idx)
                .copied()
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "output_idx {} out of range for output with {} elements",
                        output_idx,
                        output_plus.index_axis(Axis(0), batch_idx).len()
                    ))
                })?;
            let y_minus = output_minus
                .index_axis(Axis(0), batch_idx)
                .iter()
                .nth(output_idx)
                .copied()
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "output_idx {} out of range for output with {} elements",
                        output_idx,
                        output_minus.index_axis(Axis(0), batch_idx).len()
                    ))
                })?;

            let diff = y_plus - y_minus;
            let gradient = perturbation_batch.index_axis(Axis(0), batch_idx).to_owned()
                * (diff / (2.0 * delta));
            gradient_batch
                .index_axis_mut(Axis(0), batch_idx)
                .assign(&gradient);
        }

        Ok((gradient_batch, 2 * batch_size))
    }
}
