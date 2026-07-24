// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Network evaluation: concrete forward, smooth Sign relaxation, and batch evaluation.

use ndarray::{ArrayD, IxDyn};
use ny_core::{GpuIbpResult, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::contiguous_flat_slice;
use crate::layers::common::BoundPropagation;
use crate::layers::Layer;
use crate::network::ibp::try_lower_dense_chain;
use crate::Network;

use super::{CachedGpuIbpPlanEntry, CachedGpuIbpPlanKey, PgdAttacker};

/// Sharpness parameter for smooth Sign approximation during attack.
///
/// tanh(SMOOTH_SIGN_BETA * x) closely approximates sign(x) but has
/// nonzero gradient everywhere, giving SPSA a meaningful gradient signal
/// through BNN Sign layers (#3769).
///
/// At beta=10: tanh(10 * 0.1) = tanh(1) ≈ 0.76 (gradient ≈ 4.2)
///             tanh(10 * 0.5) ≈ 1.0           (gradient ≈ 0)
pub(in crate::pgd_attack) const SMOOTH_SIGN_BETA: f32 = 10.0;

/// Attack-only forward replacement for `Layer::Sign` during gradient
/// estimation (#surrogate-sign). Violation checks always use the TRUE Sign
/// forward; candidates are re-validated downstream, so soundness is
/// unaffected regardless of variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::pgd_attack) enum SignSurrogate {
    /// Legacy smooth relaxation `tanh(β·x)` (#3769). Finite differences
    /// vanish once `|β·x| ≳ 10` — BNN pre-activations saturate it.
    Tanh(f32),
    /// Plain straight-through estimator: sign(x) → x, so `d/dx sign(x) = 1`
    /// everywhere. Keeps a finite-difference signal at any activation scale.
    Ste,
}

impl SignSurrogate {
    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            SignSurrogate::Tanh(beta) => (beta * x).tanh(),
            SignSurrogate::Ste => x,
        }
    }
}

/// Extract a single output value by index, returning an error if out of bounds.
///
/// PGD attack relies on correct output indexing from the verification spec.
/// Silent fallback to 0.0 on OOB would mask spec mismatches, making the attack
/// produce wrong gradients (SPSA) or wrong violation checks.
pub(in crate::pgd_attack) fn output_value(output: &ArrayD<f32>, idx: usize) -> Result<f32> {
    output.iter().nth(idx).copied().ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "output_idx {} out of range for output with {} elements",
            idx,
            output.len()
        ))
    })
}

fn bounded_tensor_from_gpu_ibp_result(result: GpuIbpResult) -> Result<BoundedTensor> {
    let lower = ArrayD::from_shape_vec(IxDyn(&result.output_shape), result.lower_bounds).map_err(
        |err| NyError::InternalError(format!("PGD cached GPU IBP: output shape mismatch: {err}")),
    )?;
    let upper = ArrayD::from_shape_vec(IxDyn(&result.output_shape), result.upper_bounds).map_err(
        |err| NyError::InternalError(format!("PGD cached GPU IBP: output shape mismatch: {err}")),
    )?;
    BoundedTensor::new(lower, upper)
}

impl PgdAttacker<'_> {
    fn evaluate_with_cached_model_plan(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
    ) -> Result<Option<BoundedTensor>> {
        let Some(plan_factory) = self
            .engine
            .and_then(|engine| engine.as_gpu_ibp_forward_ext())
        else {
            return Ok(None);
        };

        let key = CachedGpuIbpPlanKey::new(network, input_bounds.shape());
        let input_lower = contiguous_flat_slice(input_bounds.lower());
        let input_upper = contiguous_flat_slice(input_bounds.upper());
        let mut cached = self.model_plan.lock().map_err(|_| {
            NyError::InternalError("PGD cached model-plan mutex poisoned".to_string())
        })?;

        let needs_refresh = cached
            .as_ref()
            .map(|entry| {
                entry.key.layers_ptr != key.layers_ptr
                    || entry.key.layer_count != key.layer_count
                    || entry.key.input_shape != key.input_shape
            })
            .unwrap_or(true);

        if needs_refresh {
            let plan = match try_lower_dense_chain(network, input_bounds.shape()) {
                Some(layers) => {
                    match plan_factory.prepare_model_plan(&layers, input_bounds.shape()) {
                        Ok(plan) => plan,
                        Err(err) => {
                            debug!(
                            ?err,
                            "PGD cached GPU IBP plan preparation failed; falling back to existing IBP path"
                        );
                            None
                        }
                    }
                }
                None => None,
            };
            *cached = Some(CachedGpuIbpPlanEntry { key, plan });
        }

        let Some(entry) = cached.as_mut() else {
            return Ok(None);
        };
        let Some(plan) = entry.plan.as_ref() else {
            return Ok(None);
        };

        match plan.ibp_forward_cached(
            input_lower.as_ref(),
            input_upper.as_ref(),
            input_bounds.shape(),
        ) {
            Ok(result) => bounded_tensor_from_gpu_ibp_result(result).map(Some),
            Err(err) => {
                debug!(
                    ?err,
                    "PGD cached GPU IBP execution failed; clearing plan and falling back"
                );
                entry.plan = None;
                Ok(None)
            }
        }
    }

    /// Evaluate network at a concrete point.
    ///
    /// Returns the network output as a flat f32 array.
    pub(crate) fn evaluate(&self, network: &Network, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let input_bounds = BoundedTensor::concrete(input.clone())?;
        if let Some(output_bounds) = self.evaluate_with_cached_model_plan(network, &input_bounds)? {
            // The cached plan only fires for pure dense chains
            // (Linear/ReLU/Flatten/Reshape), which preserve a point exactly, so the
            // box stays degenerate — its center is the faithful forward value.
            return Ok(output_bounds.center());
        }
        // TRUE concrete (point) forward: collapse to the interval center after each
        // layer. The whole-box `propagate_ibp_with_engine` returns a NON-degenerate
        // box even for a point input — per-layer soundness widening (esp. BatchNorm)
        // amplified by a deep conv stack — so `.lower()` is NOT the forward value and
        // fabricates false counterexamples (cgan_2023 unknown-downgrade). #cgan-eval.
        let output_bounds = network.propagate_concrete_point(&input_bounds, self.eval_engine())?;
        Ok(output_bounds.center())
    }

    /// Evaluate network at a batch of concrete points.
    ///
    /// `inputs` has shape `[N, ...input_shape]` where N is the batch size.
    /// Returns output of shape `[N, ...output_shape]`.
    ///
    /// For concrete inputs (lower == upper), the IBP GEMM engine path uses a
    /// single GEMM per linear layer regardless of batch size, making this
    /// dramatically more efficient than N individual `evaluate()` calls.
    ///
    /// Reference: alpha-beta-CROWN attack_pgd.py:267 batches all restarts
    /// into one forward pass via `model(inputs.view(-1, *X_shape[2:]))`.
    pub(super) fn evaluate_batch(
        &self,
        network: &Network,
        inputs: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        let input_bounds = BoundedTensor::concrete(inputs.clone())?;
        if let Some(output_bounds) = self.evaluate_with_cached_model_plan(network, &input_bounds)? {
            // Dense-chain plan preserves a point exactly (see `evaluate`).
            return Ok(output_bounds.center());
        }
        // Batched PGD prepends a restart axis; preserve it across sequential
        // Flatten/Reshape layers so the stored unbatched network contract still
        // reaches downstream Linear layers (#4345). TRUE point forward (center-
        // collapse per layer) so per-layer widening cannot amplify (#cgan-eval).
        let output_bounds = network
            .propagate_concrete_point_preserve_leading_axis(&input_bounds, self.eval_engine())?;
        Ok(output_bounds.center())
    }

    /// The Sign surrogate the attack's gradient estimation should evaluate
    /// through: plain STE when `surrogate_sign_gradient` is set, the legacy
    /// `tanh(β·x)` smooth relaxation otherwise (#surrogate-sign).
    pub(super) fn attack_sign_surrogate(&self) -> SignSurrogate {
        if self.config.surrogate_sign_gradient {
            SignSurrogate::Ste
        } else {
            SignSurrogate::Tanh(SMOOTH_SIGN_BETA)
        }
    }

    /// Evaluate concrete input through a network where Sign layers are replaced
    /// with `tanh(β*x)` — a smooth approximation that preserves Sign's overall
    /// shape but has nonzero gradient everywhere (#3769).
    ///
    /// Used during SPSA gradient estimation only. The violation check still uses
    /// the original Sign layer so soundness is unaffected.
    ///
    /// For BNNs (Binary Neural Networks), Sign has zero derivative almost
    /// everywhere, causing SPSA to see zero finite differences. The smooth
    /// approximation gives SPSA a meaningful gradient direction toward Sign
    /// thresholds.
    #[cfg(test)]
    pub(super) fn evaluate_smooth_sign(
        &self,
        network: &Network,
        input: &ArrayD<f32>,
        beta: f32,
    ) -> Result<ArrayD<f32>> {
        self.evaluate_sign_surrogate(network, input, SignSurrogate::Tanh(beta))
    }

    /// Evaluate concrete input through a network where Sign layers are replaced
    /// by the configured attack surrogate (#3769, #surrogate-sign).
    pub(super) fn evaluate_sign_surrogate(
        &self,
        network: &Network,
        input: &ArrayD<f32>,
        surrogate: SignSurrogate,
    ) -> Result<ArrayD<f32>> {
        let mut current = BoundedTensor::concrete(input.clone())?;
        for (i, layer) in network.layers.iter().enumerate() {
            let next = match layer {
                Layer::Sign(_) => {
                    let smoothed = current.lower().mapv(|x| surrogate.apply(x));
                    BoundedTensor::concrete(smoothed)?
                }
                Layer::Linear(linear) => {
                    linear.propagate_ibp_with_engine(&current, self.eval_engine())?
                }
                _ => layer.propagate_ibp(&current)?,
            };
            // NaN check consistent with IBP forward path
            if next.lower().iter().any(|v| v.is_nan()) {
                return Err(NyError::NumericalInstability(format!(
                    "smooth Sign evaluation NaN at layer {} ({})",
                    i,
                    layer.layer_type()
                )));
            }
            // Collapse to the interval center so per-layer soundness widening cannot
            // amplify across the network (#cgan-eval); a point input must stay a point.
            current = BoundedTensor::concrete(next.center())?;
        }
        Ok(current.center())
    }

    /// Evaluate a batch of concrete inputs with the configured Sign surrogate
    /// (#3769, #surrogate-sign).
    pub(super) fn evaluate_batch_sign_surrogate(
        &self,
        network: &Network,
        inputs: &ArrayD<f32>,
        surrogate: SignSurrogate,
    ) -> Result<ArrayD<f32>> {
        let mut current = BoundedTensor::concrete(inputs.clone())?;
        for (i, layer) in network.layers.iter().enumerate() {
            let next = match layer {
                Layer::Sign(_) => {
                    let smoothed = current.lower().mapv(|x| surrogate.apply(x));
                    BoundedTensor::concrete(smoothed)?
                }
                Layer::Linear(linear) => {
                    linear.propagate_ibp_with_engine(&current, self.eval_engine())?
                }
                Layer::Flatten(layer) => layer.propagate_ibp_preserve_leading_axis(&current)?,
                Layer::Reshape(layer) => layer.propagate_ibp_preserve_leading_axis(&current)?,
                _ => layer.propagate_ibp(&current)?,
            };
            if next.lower().iter().any(|v| v.is_nan()) {
                return Err(NyError::NumericalInstability(format!(
                    "smooth Sign batch evaluation NaN at layer {} ({})",
                    i,
                    layer.layer_type()
                )));
            }
            // Collapse to the interval center (#cgan-eval): keep a point a point.
            current = BoundedTensor::concrete(next.center())?;
        }
        Ok(current.center())
    }
}
