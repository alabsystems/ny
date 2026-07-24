// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};

use crate::beta_crown::branching::GraphSplitHistory;

use super::types::{CutKind, CutMetadata, GraphCutTerm};

/// A cutting plane constraint for GraphNetwork verification.
///
/// Graph cutting planes encode verified domain configurations:
/// "If these neurons are all in these states, the property is verified."
/// The cut constraint prevents the verifier from redundantly exploring
/// regions already proven by earlier verified subdomains.
#[derive(Debug, Clone)]
pub struct GraphCuttingPlane {
    /// Terms of the cut (neuron references and coefficients).
    pub(crate) terms: Vec<GraphCutTerm>,
    /// Right-hand side bias of the constraint.
    pub(crate) bias: f32,
    /// Lagrangian multiplier for this cut (optimized during bound computation).
    pub(crate) lambda: f32,
    /// Gradient of the objective w.r.t. lambda.
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

impl GraphCuttingPlane {
    /// Create a new graph cutting plane with validated lambda.
    ///
    /// Lambda must be >= 0 and finite (Lagrangian multiplier invariant).
    /// Bias must be finite.
    /// Optimizer state (lambda_grad, lambda_m, lambda_v) initialized to zero.
    pub fn new(
        terms: Vec<GraphCutTerm>,
        bias: f32,
        lambda: f32,
        source_depth: usize,
        metadata: CutMetadata,
    ) -> Result<Self> {
        if !bias.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "GraphCuttingPlane bias must be finite, got {bias}"
            )));
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(NyError::NumericalInstability(format!(
                "GraphCuttingPlane lambda must be finite and >= 0, got {lambda}"
            )));
        }
        // Validate term coefficients: NaN/Inf coefficients would propagate
        // through evaluate() and corrupt bound computations (#3076).
        for (i, term) in terms.iter().enumerate() {
            if !term.coefficient.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "GraphCuttingPlane term[{i}] coefficient must be finite, got {}",
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
    pub fn set_lambda(&mut self, value: f32) -> Result<()> {
        if !value.is_finite() || value < 0.0 {
            return Err(NyError::NumericalInstability(format!(
                "GraphCuttingPlane lambda must be finite and >= 0, got {value}"
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
    /// NaN/Inf gradients are reset to 0.0 to prevent corruption of Adam
    /// moment estimates (#3076).
    pub fn set_lambda_grad(&mut self, value: f32) {
        self.lambda_grad = if value.is_finite() { value } else { 0.0 };
    }

    /// Get bias (right-hand side of the constraint).
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Create a cutting plane from a verified GraphNetwork domain's split history.
    ///
    /// When a domain is verified with split history {(node1,n1,active), (node2,n2,inactive), ...},
    /// it means that constraining neurons to these states leads to lb > threshold.
    /// The cut encodes: "NOT all of these constraints can be true in an unverified region"
    pub fn from_verified_domain(history: &GraphSplitHistory) -> Result<Option<Self>> {
        if history.constraints.is_empty() || !history.genbab_constraints.is_empty() {
            return Ok(None);
        }

        let terms: Vec<GraphCutTerm> = history
            .constraints
            .iter()
            .map(|c| GraphCutTerm {
                node_name: c.node_name.clone(),
                neuron_idx: c.neuron_idx,
                // Sign based on constraint: +1 if active (x >= 0), -1 if inactive (x <= 0)
                coefficient: if c.is_active { 1.0 } else { -1.0 },
            })
            .collect();

        // Bias computation follows BICCOS: bias = (count of active neurons) - 1
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

    /// Check if this cut is redundant with a domain's current constraints.
    pub fn is_redundant_for(&self, history: &GraphSplitHistory) -> bool {
        let satisfied = self
            .terms
            .iter()
            .filter(|term| {
                history
                    .is_constrained(&term.node_name, term.neuron_idx)
                    .map(|is_active| {
                        (term.coefficient > 0.0 && is_active)
                            || (term.coefficient < 0.0 && !is_active)
                    })
                    .unwrap_or(false)
            })
            .count();

        satisfied == self.terms.len()
    }

    /// Reset gradients for a new optimization iteration.
    pub fn zero_grad(&mut self) {
        self.lambda_grad = 0.0;
    }

    /// Update lambda using Adam optimizer.
    pub fn update_lambda_adam(&mut self, lr: f32, beta1: f32, beta2: f32, epsilon: f32, t: usize) {
        // Guard against t=0: bias correction divides by (1 - beta^t), which is
        // zero when t=0 (#2575).
        let t = t.max(1);

        // Update biased first moment estimate
        self.lambda_m = beta1 * self.lambda_m + (1.0 - beta1) * self.lambda_grad;

        // Update biased second raw moment estimate
        self.lambda_v = beta2 * self.lambda_v + (1.0 - beta2) * self.lambda_grad * self.lambda_grad;

        // Compute bias-corrected estimates
        // Guard: beta=1.0 makes denominator=0, causing division by zero (#2575, #2586).
        // Matches alpha.rs:116-117 pattern.
        let t_f32 = t as f32;
        let m_hat = self.lambda_m / (1.0 - beta1.powf(t_f32)).max(f32::EPSILON);
        let v_hat = self.lambda_v / (1.0 - beta2.powf(t_f32)).max(f32::EPSILON);

        // Update lambda (maximize, so add gradient)
        self.lambda += lr * m_hat / (v_hat.sqrt() + epsilon);

        // Project to feasible region: 0 <= lambda <= MAX_LAMBDA
        // Upper bound prevents lambda explosion and maintains soundness
        const MAX_LAMBDA: f32 = 10.0;
        // NaN guard: clamp() propagates NaN (IEEE 754). Check lambda, m, v —
        // lambda can become NaN independently if lr is NaN while grad is zero
        // (#2598, #3076).
        if !self.lambda.is_finite() || !self.lambda_m.is_finite() || !self.lambda_v.is_finite() {
            self.lambda = 0.0;
            self.lambda_m = 0.0;
            self.lambda_v = 0.0;
            self.lambda_grad = 0.0;
        } else {
            self.lambda = self.lambda.clamp(0.0, MAX_LAMBDA);
        }
    }
}
