// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};

use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::config::AdaptiveOptConfig;
use crate::beta_crown::state::BetaState;

use super::types::{CutKind, CutMetadata, CutTerm};

/// A cutting plane constraint derived from a verified subdomain.
///
/// Cutting planes capture the relationship: "this combination of neuron states
/// leads to verification." For unverified regions, at least one neuron must
/// be in a different state than what was verified.
///
/// Mathematical form: sum_i(coeff_i * x_i) >= bias
/// where x_i is the pre-activation value of neuron i.
#[derive(Debug, Clone)]
pub struct CuttingPlane {
    /// Terms in the linear constraint.
    pub(crate) terms: Vec<CutTerm>,
    /// Right-hand side of the constraint.
    pub(crate) bias: f32,
    /// Lagrangian multiplier for this cut (dual variable, must be >= 0).
    pub(crate) lambda: f32,
    /// Gradient of lambda for optimization.
    pub(crate) lambda_grad: f32,
    /// First moment estimate for Adam optimizer.
    pub(crate) lambda_m: f32,
    /// Second moment estimate for Adam optimizer.
    pub(crate) lambda_v: f32,
    /// Source domain depth (for debugging/analysis).
    pub(crate) source_depth: usize,
    /// Cut metadata for eviction/freshness tracking.
    pub(crate) metadata: CutMetadata,
}

/// Result of BICCOS-style constraint strengthening.
#[derive(Debug, Clone)]
pub struct StrengthenedCut {
    /// The strengthened cut (may be identical to the original cut).
    pub cut: CuttingPlane,
    /// Split history corresponding to the strengthened cut.
    pub history: SplitHistory,
    /// Number of constraints dropped during strengthening.
    pub dropped_constraints: usize,
}

impl CuttingPlane {
    /// Create a new cutting plane with validated lambda.
    ///
    /// Lambda must be >= 0 and finite (Lagrangian multiplier invariant).
    /// Bias must be finite.
    /// Optimizer state (lambda_grad, lambda_m, lambda_v) initialized to zero.
    pub fn new(
        terms: Vec<CutTerm>,
        bias: f32,
        lambda: f32,
        source_depth: usize,
        metadata: CutMetadata,
    ) -> Result<Self> {
        if !bias.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "CuttingPlane bias must be finite, got {bias}"
            )));
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(NyError::NumericalInstability(format!(
                "CuttingPlane lambda must be finite and >= 0, got {lambda}"
            )));
        }
        // Validate term coefficients: NaN/Inf coefficients would propagate
        // through evaluate() and corrupt bound computations (#3076).
        for (i, term) in terms.iter().enumerate() {
            if !term.coefficient.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "CuttingPlane term[{i}] coefficient must be finite, got {}",
                    term.coefficient
                )));
            }
        }
        Ok(Self {
            terms,
            bias,
            lambda,
            lambda_grad: 0.0,
            lambda_m: 0.0,
            lambda_v: 0.0,
            source_depth,
            metadata,
        })
    }

    /// Set lambda value. Validates lambda >= 0 and finite.
    ///
    /// All writes to `lambda` from outside the optimizer loop should go through
    /// this method to enforce the invariant that lambda is a valid Lagrangian
    /// multiplier.
    pub fn set_lambda(&mut self, value: f32) -> Result<()> {
        if !value.is_finite() || value < 0.0 {
            return Err(NyError::NumericalInstability(format!(
                "CuttingPlane lambda must be finite and >= 0, got {value}"
            )));
        }
        self.lambda = value;
        Ok(())
    }

    /// Reset lambda and all optimizer state to zero.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
        self.lambda_grad = 0.0;
        self.lambda_m = 0.0;
        self.lambda_v = 0.0;
    }

    /// Get lambda value (Lagrangian multiplier).
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Get lambda gradient.
    pub fn lambda_grad(&self) -> f32 {
        self.lambda_grad
    }

    /// Set lambda gradient value.
    ///
    /// Gradient: d(lb)/d(lambda) = bias - constraint_min.
    /// The cut adds: lambda * (bias - constraint_min) to the lower bound.
    /// NaN/Inf gradients are reset to 0.0 to prevent corruption of Adam
    /// moment estimates (#3076).
    pub fn set_lambda_grad(&mut self, value: f32) {
        self.lambda_grad = if value.is_finite() {
            value
        } else {
            tracing::warn!(
                "NaN/Inf in CuttingPlane::set_lambda_grad, clamping to 0.0: value={value}"
            );
            0.0
        };
    }

    /// Get bias (right-hand side of the constraint).
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Perform SGD gradient ascent step on lambda.
    ///
    /// Applies: lambda += lr * lambda_grad, then projects to [0, MAX_LAMBDA].
    /// NaN guard: if lambda becomes non-finite after the update, reset to 0.0.
    /// Returns the absolute lambda gradient (for convergence tracking).
    ///
    /// Reference: alpha-beta-CROWN optimize loop SGD path.
    pub fn gradient_step_sgd(&mut self, lr: f32) -> f32 {
        let grad_abs = self.lambda_grad.abs();
        // Gradient ascent step (maximize lower bound)
        self.lambda += lr * self.lambda_grad;
        // Project to feasible region: 0 <= lambda <= MAX_LAMBDA
        // NaN guard: clamp() preserves NaN (IEEE 754). Reset to 0.0. (#2598)
        const MAX_LAMBDA: f32 = 10.0;
        if self.lambda.is_finite() {
            self.lambda = self.lambda.clamp(0.0, MAX_LAMBDA);
        } else {
            tracing::warn!(
                "NaN/Inf in CuttingPlane::gradient_step_sgd, resetting lambda to 0.0: lambda={}, lambda_grad={}",
                self.lambda, self.lambda_grad
            );
            self.lambda = 0.0;
        }
        grad_abs
    }

    /// Create a cutting plane from a verified domain's split history.
    ///
    /// When a domain is verified with split history {(l1,n1,active), (l2,n2,inactive), ...},
    /// it means that constraining neurons to these states leads to lb > threshold.
    /// The cut encodes: "NOT all of these constraints can be true in an unverified region"
    /// which translates to: sum_i(sign_i * indicator_i) >= 1
    pub fn from_verified_domain(history: &SplitHistory) -> Result<Option<Self>> {
        if history.constraints.is_empty() {
            return Ok(None);
        }

        let terms: Vec<CutTerm> = history
            .constraints
            .iter()
            .map(|c| CutTerm {
                layer_idx: c.layer_idx,
                neuron_idx: c.neuron_idx,
                // Sign based on constraint: +1 if active (x >= 0), -1 if inactive (x <= 0)
                coefficient: if c.is_active { 1.0 } else { -1.0 },
            })
            .collect();

        // Bias computation follows BICCOS: bias = (count of active neurons) - 1
        // The constraint form is: sum(coeff_i * z_i) <= bias
        // where z_i is the ReLU indicator (0 if inactive, 1 if active)
        // This constraint encodes: "can't have all neurons in their verified states"
        let active_count = terms.iter().filter(|t| t.coefficient > 0.0).count();
        let bias = (active_count as f32) - 1.0;

        let source_depth = terms.len();
        Ok(Some(Self::new(
            terms,
            bias,
            0.0,
            source_depth,
            CutMetadata::new(0, CutKind::Verified),
        )?))
    }

    /// Create a strengthened cut using BICCOS-style constraint selection.
    ///
    /// This selects a subset of constraints using influence scores (stored in history)
    /// and β values. Constraints are kept if:
    /// - β > 0 (active dual contribution), OR
    /// - score ranks above the drop_ratio cutoff (influence-based retention).
    pub fn from_verified_domain_strengthened(
        history: &SplitHistory,
        beta_state: &BetaState,
        drop_ratio: f32,
    ) -> Result<Option<StrengthenedCut>> {
        if history.constraints.is_empty() {
            return Ok(None);
        }

        let drop_ratio = if drop_ratio.is_finite() {
            drop_ratio
        } else {
            0.0
        };

        let mut scored: Vec<(usize, f32, bool)> = history
            .constraints
            .iter()
            .enumerate()
            .filter_map(|(idx, constraint)| {
                if constraint.score.is_finite() {
                    let beta = beta_state
                        .beta(constraint.layer_idx, constraint.neuron_idx)
                        .unwrap_or(0.0);
                    Some((idx, constraint.score, beta > 0.0))
                } else {
                    None
                }
            })
            .collect();

        let keep_all_unranked = scored.is_empty();
        let mut drop_flags = vec![false; history.constraints.len()];
        if !keep_all_unranked && drop_ratio > 0.0 {
            // SAFETY: drop_ratio is guaranteed finite by the is_finite() guard above,
            // so clamped is in [0.0, 1.0] and the product with scored.len() is finite
            // and non-negative. The `as usize` truncation is well-defined.
            let clamped = drop_ratio.clamp(0.0, 1.0);
            let drop_count = (clamped * (scored.len() as f32)).floor() as usize;
            if drop_count > 0 {
                scored.sort_by(|a, b| crate::cmp_utils::nan_propagating_cmp(&a.1, &b.1));
                let mut dropped = 0usize;
                for (idx, _score, beta_positive) in scored {
                    if beta_positive {
                        continue;
                    }
                    drop_flags[idx] = true;
                    dropped += 1;
                    if dropped >= drop_count {
                        break;
                    }
                }
            }
        }

        let mut kept_constraints: Vec<_> = Vec::with_capacity(history.constraints.len());
        for (idx, constraint) in history.constraints.iter().enumerate() {
            let beta = beta_state
                .beta(constraint.layer_idx, constraint.neuron_idx)
                .unwrap_or(0.0);
            let keep = beta > 0.0
                || keep_all_unranked
                || (constraint.score.is_finite() && !drop_flags[idx]);
            if keep {
                kept_constraints.push(*constraint);
            }
        }

        if kept_constraints.is_empty() {
            return Ok(None);
        }

        let dropped_constraints = history
            .constraints
            .len()
            .saturating_sub(kept_constraints.len());

        let terms: Vec<CutTerm> = kept_constraints
            .iter()
            .map(|c| CutTerm {
                layer_idx: c.layer_idx,
                neuron_idx: c.neuron_idx,
                coefficient: if c.is_active { 1.0 } else { -1.0 },
            })
            .collect();

        let active_count = kept_constraints.iter().filter(|c| c.is_active).count();
        let bias = (active_count as f32) - 1.0;

        let source_depth = terms.len();
        let cut = Self::new(
            terms,
            bias,
            0.0,
            source_depth,
            CutMetadata::new(0, CutKind::Verified),
        )?;

        let mut strengthened_history = SplitHistory::new();
        for constraint in kept_constraints {
            strengthened_history.add_constraint(constraint);
        }

        Ok(Some(StrengthenedCut {
            cut,
            history: strengthened_history,
            dropped_constraints,
        }))
    }

    /// Check if this cut is redundant with a domain's current constraints.
    ///
    /// A cut is redundant if the domain already implies all the cut's constraints
    /// are satisfied (all neurons are in the states specified by the cut).
    pub fn is_redundant_for(&self, history: &SplitHistory) -> bool {
        // Count how many of the cut's terms are already satisfied by the domain
        let satisfied = self
            .terms
            .iter()
            .filter(|term| {
                history
                    .is_constrained(term.layer_idx, term.neuron_idx)
                    .map(|is_active| {
                        // Term is satisfied if constraint matches cut's expectation
                        (term.coefficient > 0.0 && is_active)
                            || (term.coefficient < 0.0 && !is_active)
                    })
                    .unwrap_or(false)
            })
            .count();

        // Cut is redundant if all terms are already satisfied
        // (means this domain is already in the verified region)
        satisfied == self.terms.len()
    }

    /// Evaluate the cut's contribution to the bound.
    ///
    /// Returns the Lagrangian term: lambda * (sum_i(coeff_i * x_i) - bias)
    /// For lower bound maximization, this is added when the constraint is satisfied.
    pub fn evaluate(&self, pre_activations: &[(f32, f32)]) -> f32 {
        let constraint_value: f32 = self
            .terms
            .iter()
            .map(|term| {
                // Use lower bound if coefficient positive, upper if negative
                // This gives the worst-case (minimum) value of the constraint.
                // Bounds-checked: stale cuts may reference neurons beyond the slice (#2860).
                let (lo, hi) = pre_activations
                    .get(term.neuron_idx)
                    .copied()
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
                if term.coefficient > 0.0 {
                    term.coefficient * lo
                } else {
                    term.coefficient * hi
                }
            })
            .sum();

        let result = self.lambda * (constraint_value - self.bias);
        // NaN guard: if lambda or any pre-activation is NaN, return 0.0
        // (neutral contribution) rather than propagating NaN. (#2598)
        if result.is_finite() {
            result
        } else {
            tracing::warn!(
                "NaN/Inf in CuttingPlane::evaluate, returning 0.0: result={result}, lambda={}, bias={}",
                self.lambda, self.bias
            );
            0.0
        }
    }

    /// Reset gradients for a new optimization iteration.
    pub fn zero_grad(&mut self) {
        self.lambda_grad = 0.0;
    }

    /// Perform Adam gradient step on lambda.
    pub fn gradient_step_adam(&mut self, config: &AdaptiveOptConfig, t: usize) {
        let eps = config.epsilon;
        let beta1 = config.beta1;
        let beta2 = config.beta2;
        let lr = config.lr_lambda.unwrap_or(config.beta_lr);
        // Guard against t=0: bias correction divides by (1 - beta^t), which is
        // zero when t=0 (#2575).
        let t = t.max(1);

        // Update biased first moment estimate
        self.lambda_m = beta1 * self.lambda_m + (1.0 - beta1) * self.lambda_grad;

        // Update biased second moment estimate
        self.lambda_v = beta2 * self.lambda_v + (1.0 - beta2) * self.lambda_grad * self.lambda_grad;

        // Compute bias-corrected estimates
        // Guard: beta=1.0 makes denominator=0, causing division by zero (#2575, #2586).
        // Matches alpha.rs:116-117 pattern.
        // Saturate t to i32::MAX to prevent silent wrapping in powi (#2840).
        let t_i32 = t.min(i32::MAX as usize) as i32;
        let m_hat = if config.bias_correction {
            self.lambda_m / (1.0 - beta1.powi(t_i32)).max(f32::EPSILON)
        } else {
            self.lambda_m
        };

        let v_hat = if config.bias_correction {
            self.lambda_v / (1.0 - beta2.powi(t_i32)).max(f32::EPSILON)
        } else {
            self.lambda_v
        };

        // Gradient ascent step (maximize lower bound)
        let update = lr * m_hat / (v_hat.sqrt() + eps);
        self.lambda += update;

        // Project to feasible region: 0 <= lambda <= MAX_LAMBDA
        // Upper bound prevents lambda explosion and maintains soundness
        const MAX_LAMBDA: f32 = 10.0;
        // NaN guard: clamp() propagates NaN (IEEE 754). Check lambda, m, v —
        // lambda can become NaN independently if lr is NaN while grad is zero
        // (#2598, #3076).
        if !self.lambda.is_finite() || !self.lambda_m.is_finite() || !self.lambda_v.is_finite() {
            tracing::warn!(
                "NaN/Inf in CuttingPlane::gradient_step_adam, resetting lambda/m/v/grad to 0.0: lambda={}, lambda_m={}, lambda_v={}",
                self.lambda, self.lambda_m, self.lambda_v
            );
            self.lambda = 0.0;
            self.lambda_m = 0.0;
            self.lambda_v = 0.0;
            self.lambda_grad = 0.0;
        } else {
            self.lambda = self.lambda.clamp(0.0, MAX_LAMBDA);
        }
    }
}
