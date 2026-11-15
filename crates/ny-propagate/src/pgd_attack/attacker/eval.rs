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
use crate::network::ibp::{
    propagate_concrete_point_until, try_lower_dense_chain, try_lower_dense_chain_until,
    LeadingAxisMode,
};
use crate::Network;

use super::{CachedGpuIbpPlanEntry, CachedGpuIbpPlanKey, PgdAttacker};

/// Sharpness parameter for smooth Sign approximation during attack.
///
/// tanh(SMOOTH_SIGN_BETA * x) closely approximates sign(x) but has
/// nonzero gradient everywhere, giving SPSA a meaningful gradient signal
/// through BNN Sign layers.
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
    /// Legacy smooth relaxation `tanh(β·x)`. Finite differences
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
    fn attack_point_tensor(&self, values: ArrayD<f32>) -> Result<BoundedTensor> {
        if self.effective_deadline().is_some() {
            BoundedTensor::concrete_with_poll(values, || {
                self.checkpoint("during point construction")
            })
        } else {
            BoundedTensor::concrete(values)
        }
    }

    fn attack_point_center(&self, bounds: &BoundedTensor) -> Result<ArrayD<f32>> {
        if self.effective_deadline().is_some() {
            bounds.center_with_poll(|| self.checkpoint("during point center"))
        } else {
            Ok(bounds.center())
        }
    }

    fn apply_sign_surrogate_until(
        &self,
        current: &BoundedTensor,
        surrogate: SignSurrogate,
    ) -> Result<BoundedTensor> {
        self.checkpoint("before Sign surrogate allocation")?;
        let smoothed = if self.effective_deadline().is_some() {
            let mut values = ArrayD::zeros(IxDyn(current.shape()));
            for (index, (value, &source)) in
                values.iter_mut().zip(current.lower().iter()).enumerate()
            {
                if index.is_multiple_of(4_096) {
                    self.checkpoint("during Sign surrogate transformation")?;
                }
                *value = surrogate.apply(source);
            }
            values
        } else {
            current.lower().mapv(|x| surrogate.apply(x))
        };
        self.attack_point_tensor(smoothed)
    }

    fn evaluate_with_cached_model_plan(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
    ) -> Result<Option<BoundedTensor>> {
        self.checkpoint("before cached point evaluation")?;
        let deadline = self.effective_deadline();
        let Some(plan_factory) = self
            .engine
            .and_then(|engine| engine.as_gpu_ibp_forward_ext())
        else {
            return Ok(None);
        };

        let key = CachedGpuIbpPlanKey::new(network, input_bounds.shape());
        let input_lower = contiguous_flat_slice(input_bounds.lower());
        let input_upper = contiguous_flat_slice(input_bounds.upper());
        let mut cached = if deadline.is_some() {
            loop {
                self.checkpoint("while acquiring cached plan")?;
                match self.model_plan.try_lock() {
                    Ok(guard) => break guard,
                    Err(std::sync::TryLockError::WouldBlock) => std::thread::yield_now(),
                    Err(std::sync::TryLockError::Poisoned(_)) => {
                        return Err(NyError::InternalError(
                            "PGD cached model-plan mutex poisoned".into(),
                        ));
                    }
                }
            }
        } else {
            self.model_plan.lock().map_err(|_| {
                NyError::InternalError("PGD cached model-plan mutex poisoned".to_string())
            })?
        };

        let needs_refresh = cached
            .as_ref()
            .map(|entry| {
                entry.key.layers_ptr != key.layers_ptr
                    || entry.key.layer_count != key.layer_count
                    || entry.key.input_shape != key.input_shape
                    || entry.deadline_aware != deadline.is_some()
            })
            .unwrap_or(true);

        if needs_refresh {
            let layers = match deadline {
                Some(limit) => try_lower_dense_chain_until(network, input_bounds.shape(), limit)?,
                None => try_lower_dense_chain(network, input_bounds.shape()),
            };
            let plan = match layers {
                Some(layers) => {
                    self.checkpoint("before cached plan preparation")?;
                    let prepared = match deadline {
                        Some(limit) => plan_factory.prepare_model_plan_with_deadline(
                            &layers,
                            input_bounds.shape(),
                            limit,
                        ),
                        None => plan_factory.prepare_model_plan(&layers, input_bounds.shape()),
                    };
                    match prepared {
                        Ok(plan) => plan,
                        Err(err) if Self::has_deadline_source(&err) => return Err(err),
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
            self.checkpoint("after cached plan preparation")?;
            *cached = Some(CachedGpuIbpPlanEntry {
                key,
                deadline_aware: deadline.is_some(),
                plan,
            });
        }

        let Some(entry) = cached.as_mut() else {
            return Ok(None);
        };
        let Some(plan) = entry.plan.as_ref() else {
            return Ok(None);
        };
        if deadline.is_some() && !plan.supports_deadline_execution() {
            return Ok(None);
        }

        self.checkpoint("before cached plan execution")?;
        let evaluated = match deadline {
            Some(limit) => plan.ibp_forward_cached_with_deadline(
                input_lower.as_ref(),
                input_upper.as_ref(),
                input_bounds.shape(),
                limit,
            ),
            None => plan.ibp_forward_cached(
                input_lower.as_ref(),
                input_upper.as_ref(),
                input_bounds.shape(),
            ),
        };
        self.checkpoint("after cached plan execution")?;
        match evaluated {
            Ok(result) => {
                let output = bounded_tensor_from_gpu_ibp_result(result)?;
                self.checkpoint("after cached output construction")?;
                Ok(Some(output))
            }
            Err(err) if Self::has_deadline_source(&err) => Err(err),
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
        self.checkpoint("before point input allocation")?;
        let input_bounds = self.attack_point_tensor(input.clone())?;
        if let Some(output_bounds) = self.evaluate_with_cached_model_plan(network, &input_bounds)? {
            // The cached plan only fires for pure dense chains
            // (Linear/ReLU/Flatten/Reshape), which preserve a point exactly, so the
            // box stays degenerate — its center is the faithful forward value.
            return self.attack_point_center(&output_bounds);
        }
        // TRUE concrete (point) forward: collapse to the interval center after each
        // layer. The whole-box `propagate_ibp_with_engine` returns a NON-degenerate
        // box even for a point input — per-layer soundness widening (esp. BatchNorm)
        // amplified by a deep conv stack — so `.lower()` is NOT the forward value and
        // fabricates false counterexamples (cgan_2023 unknown-downgrade). #cgan-eval.
        let output_bounds = propagate_concrete_point_until(
            network,
            &input_bounds,
            self.eval_engine(),
            LeadingAxisMode::Plain,
            self.effective_deadline(),
        )?;
        self.checkpoint("after point evaluation")?;
        self.attack_point_center(&output_bounds)
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
        self.checkpoint("before batch input allocation")?;
        let input_bounds = self.attack_point_tensor(inputs.clone())?;
        if let Some(output_bounds) = self.evaluate_with_cached_model_plan(network, &input_bounds)? {
            // Dense-chain plan preserves a point exactly (see `evaluate`).
            return self.attack_point_center(&output_bounds);
        }
        // Batched PGD prepends a restart axis; preserve it across sequential
        // Flatten/Reshape layers so the stored unbatched network contract still
        // reaches downstream Linear layers. TRUE point forward (center-
        // collapse per layer) so per-layer widening cannot amplify (#cgan-eval).
        let output_bounds = propagate_concrete_point_until(
            network,
            &input_bounds,
            self.eval_engine(),
            LeadingAxisMode::PreserveLeadingAxis,
            self.effective_deadline(),
        )?;
        self.checkpoint("after batch evaluation")?;
        self.attack_point_center(&output_bounds)
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
    /// shape but has nonzero gradient everywhere.
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
    /// by the configured attack surrogate (#surrogate-sign).
    pub(super) fn evaluate_sign_surrogate(
        &self,
        network: &Network,
        input: &ArrayD<f32>,
        surrogate: SignSurrogate,
    ) -> Result<ArrayD<f32>> {
        self.checkpoint("before Sign surrogate input allocation")?;
        let mut current = self.attack_point_tensor(input.clone())?;
        for (i, layer) in network.layers.iter().enumerate() {
            self.checkpoint("before Sign surrogate layer")?;
            let next = match layer {
                Layer::Sign(_) => self.apply_sign_surrogate_until(&current, surrogate)?,
                Layer::Linear(linear) => {
                    if self.effective_deadline().is_some() {
                        linear.propagate_concrete_point_with_engine_and_deadline(
                            &current,
                            self.eval_engine(),
                            self.effective_deadline(),
                        )?
                    } else {
                        linear.propagate_ibp_with_engine(&current, self.eval_engine())?
                    }
                }
                _ => layer.propagate_ibp(&current)?,
            };
            // NaN check consistent with IBP forward path
            for (index, value) in next.lower().iter().enumerate() {
                if index.is_multiple_of(4_096) {
                    self.checkpoint("during Sign surrogate result validation")?;
                }
                if value.is_nan() {
                    return Err(NyError::NumericalInstability(format!(
                        "smooth Sign evaluation NaN at layer {} ({})",
                        i,
                        layer.layer_type()
                    )));
                }
            }
            // Collapse to the interval center so per-layer soundness widening cannot
            // amplify across the network (#cgan-eval); a point input must stay a point.
            current = self.attack_point_tensor(self.attack_point_center(&next)?)?;
        }
        self.checkpoint("after Sign surrogate evaluation")?;
        self.attack_point_center(&current)
    }

    /// Evaluate a batch of concrete inputs with the configured Sign surrogate
    /// (#surrogate-sign).
    pub(super) fn evaluate_batch_sign_surrogate(
        &self,
        network: &Network,
        inputs: &ArrayD<f32>,
        surrogate: SignSurrogate,
    ) -> Result<ArrayD<f32>> {
        self.checkpoint("before batched Sign surrogate input allocation")?;
        let mut current = self.attack_point_tensor(inputs.clone())?;
        for (i, layer) in network.layers.iter().enumerate() {
            self.checkpoint("before batched Sign surrogate layer")?;
            let next = match layer {
                Layer::Sign(_) => self.apply_sign_surrogate_until(&current, surrogate)?,
                Layer::Linear(linear) => {
                    if self.effective_deadline().is_some() {
                        linear.propagate_concrete_point_with_engine_and_deadline(
                            &current,
                            self.eval_engine(),
                            self.effective_deadline(),
                        )?
                    } else {
                        linear.propagate_ibp_with_engine(&current, self.eval_engine())?
                    }
                }
                Layer::Flatten(layer) => layer.propagate_ibp_preserve_leading_axis(&current)?,
                Layer::Reshape(layer) => layer.propagate_ibp_preserve_leading_axis(&current)?,
                _ => layer.propagate_ibp(&current)?,
            };
            for (index, value) in next.lower().iter().enumerate() {
                if index.is_multiple_of(4_096) {
                    self.checkpoint("during batched Sign surrogate result validation")?;
                }
                if value.is_nan() {
                    return Err(NyError::NumericalInstability(format!(
                        "smooth Sign batch evaluation NaN at layer {} ({})",
                        i,
                        layer.layer_type()
                    )));
                }
            }
            // Collapse to the interval center (#cgan-eval): keep a point a point.
            current = self.attack_point_tensor(self.attack_point_center(&next)?)?;
        }
        self.checkpoint("after batched Sign surrogate evaluation")?;
        self.attack_point_center(&current)
    }
}
