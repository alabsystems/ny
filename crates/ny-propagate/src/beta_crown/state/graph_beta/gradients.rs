// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::core::GraphBetaState;
use crate::bounds::GraphAlphaCrownIntermediate;
use ny_core::nan_propagating_max;

impl GraphBetaState {
    /// Compute analytical β gradients from stored A matrices.
    ///
    /// For each constrained neuron at (node_name, neuron_idx), the gradient is:
    ///   ∂lb/∂β = -sign * sensitivity
    ///
    /// where sensitivity measures how changes at that neuron affect the output.
    /// The A matrix at each ReLU has shape (num_outputs, num_neurons), where
    /// A[j, i] is the coefficient from output j to neuron i.
    ///
    /// For lower bound optimization, positive A coefficients indicate the neuron
    /// contributes to tightening the lower bound, so the sensitivity is:
    ///   sensitivity = sum_j(A[j, neuron_idx]) for all outputs
    ///
    /// Returns the maximum gradient magnitude for convergence checking.
    pub fn compute_analytical_gradients(
        &mut self,
        intermediate: &GraphAlphaCrownIntermediate,
    ) -> f32 {
        let mut max_grad = 0.0f32;

        for entry in &mut self.entries {
            let a_column = match intermediate.beta_a_column(&entry.node_name, entry.neuron_idx) {
                Some(column) => column,
                None => {
                    entry.grad = 0.0;
                    continue;
                }
            };

            let mut sensitivity = 0.0f32;
            for &coefficient in a_column.iter() {
                sensitivity += coefficient;
            }

            let grad = -entry.sign * sensitivity;
            entry.grad = grad;
            max_grad = nan_propagating_max(max_grad, grad.abs());
        }

        max_grad
    }

    /// Compute analytical β gradients for multi-objective verification.
    ///
    /// Disjunctive: optimize min margin (all objectives must verify → focus on worst).
    /// Conjunctive: optimize max margin (any objective suffices → focus on best). #3334
    ///
    /// The gradient is computed for the "critical" objective (the one with min/max margin).
    /// This is a subgradient of the max-min (or max-max) function.
    ///
    /// Arguments:
    /// - `intermediate`: A matrices from backward pass (without objective applied)
    /// - `obj_bounds`: Lower bounds for each objective (pre-computed)
    /// - `objectives`: Coefficient vectors for each objective
    /// - `thresholds`: Threshold values for each objective
    /// - `verified_mask`: Mask indicating which objectives are already verified
    /// - `conjunctive`: If true, select max-margin objective (AND property semantics)
    ///
    /// Returns the maximum gradient magnitude for convergence checking.
    pub fn compute_analytical_gradients_multi_objective(
        &mut self,
        intermediate: &GraphAlphaCrownIntermediate,
        obj_bounds: &[(f32, f32)],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        verified_mask: &[bool],
        conjunctive: bool,
    ) -> f32 {
        debug_assert_eq!(
            obj_bounds.len(),
            thresholds.len(),
            "compute_analytical_gradients_multi_objective: obj_bounds/thresholds mismatch ({} vs {})",
            obj_bounds.len(),
            thresholds.len()
        );
        debug_assert_eq!(
            obj_bounds.len(),
            objectives.len(),
            "compute_analytical_gradients_multi_objective: obj_bounds/objectives mismatch ({} vs {})",
            obj_bounds.len(),
            objectives.len()
        );
        debug_assert_eq!(
            obj_bounds.len(),
            verified_mask.len(),
            "compute_analytical_gradients_multi_objective: obj_bounds/verified_mask mismatch ({} vs {})",
            obj_bounds.len(),
            verified_mask.len()
        );

        let critical_idx = if conjunctive {
            self.select_critical_conjunctive(
                intermediate,
                objectives,
                obj_bounds,
                thresholds,
                verified_mask,
            )
        } else {
            self.select_critical_disjunctive(obj_bounds, thresholds, verified_mask)
        };

        let critical_idx = match critical_idx {
            Some(idx) => idx,
            None => {
                for entry in &mut self.entries {
                    entry.grad = 0.0;
                }
                return 0.0;
            }
        };

        let critical_objective = &objectives[critical_idx];
        self.compute_gradients_for_objective(intermediate, critical_objective)
    }

    /// Compute analytical β gradients when intermediate rows already correspond
    /// one-to-one with objective rows from a dense spec matrix (#4306).
    pub fn compute_analytical_gradients_multi_objective_spec_rows(
        &mut self,
        intermediate: &GraphAlphaCrownIntermediate,
        obj_bounds: &[(f32, f32)],
        thresholds: &[f32],
        verified_mask: &[bool],
        conjunctive: bool,
    ) -> f32 {
        debug_assert_eq!(
            obj_bounds.len(),
            thresholds.len(),
            "compute_analytical_gradients_multi_objective_spec_rows: obj_bounds/thresholds mismatch ({} vs {})",
            obj_bounds.len(),
            thresholds.len()
        );
        debug_assert_eq!(
            obj_bounds.len(),
            verified_mask.len(),
            "compute_analytical_gradients_multi_objective_spec_rows: obj_bounds/verified_mask mismatch ({} vs {})",
            obj_bounds.len(),
            verified_mask.len()
        );

        let critical_idx = if conjunctive {
            self.select_critical_conjunctive_spec_rows(
                intermediate,
                obj_bounds,
                thresholds,
                verified_mask,
            )
        } else {
            self.select_critical_disjunctive(obj_bounds, thresholds, verified_mask)
        };

        let critical_idx = match critical_idx {
            Some(idx) => idx,
            None => {
                for entry in &mut self.entries {
                    entry.grad = 0.0;
                }
                return 0.0;
            }
        };

        self.compute_gradients_for_spec_row(intermediate, critical_idx)
    }

    /// Select critical objective for disjunctive mode: minimum margin (bottleneck).
    fn select_critical_disjunctive(
        &self,
        obj_bounds: &[(f32, f32)],
        thresholds: &[f32],
        verified_mask: &[bool],
    ) -> Option<usize> {
        let mut critical_idx = None;
        let mut critical_margin = f32::INFINITY;
        for (i, ((lb, _ub), &threshold)) in obj_bounds.iter().zip(thresholds).enumerate() {
            if i < verified_mask.len() && verified_mask[i] {
                continue;
            }
            let margin = lb - threshold;
            let margin = if margin.is_nan() {
                f32::NEG_INFINITY
            } else {
                margin
            };
            if margin < critical_margin {
                critical_margin = margin;
                critical_idx = Some(i);
            }
        }
        critical_idx
    }

    /// Select critical objective for conjunctive mode: maximum β sensitivity.
    ///
    /// For conjunctive (AND) properties, ANY single objective being verified
    /// suffices. We select the objective whose bounds respond most to β, i.e.,
    /// the one with the largest total |sensitivity| summed across β entries.
    /// This avoids premature convergence caused by flip-flopping between a
    /// β-responsive and a β-insensitive objective based on post-hoc margins.
    ///
    /// Fallback: if no objective has non-zero sensitivity (all A matrices are
    /// zero for all objectives), fall back to max post-hoc margin selection.
    fn select_critical_conjunctive(
        &self,
        intermediate: &GraphAlphaCrownIntermediate,
        objectives: &[Vec<f32>],
        obj_bounds: &[(f32, f32)],
        thresholds: &[f32],
        verified_mask: &[bool],
    ) -> Option<usize> {
        let mut best_idx = None;
        let mut best_sensitivity = 0.0f32;

        for (i, objective) in objectives.iter().enumerate() {
            if i < verified_mask.len() && verified_mask[i] {
                continue;
            }
            let mut total_sensitivity = 0.0f32;
            for entry in &self.entries {
                let a_column = match intermediate.beta_a_column(&entry.node_name, entry.neuron_idx)
                {
                    Some(column) => column,
                    None => continue,
                };
                let mut sensitivity = 0.0f32;
                for (j, &a_jk) in a_column.iter().enumerate() {
                    let c_j = if j < objective.len() {
                        objective[j]
                    } else {
                        0.0
                    };
                    sensitivity += c_j * a_jk;
                }
                total_sensitivity += sensitivity.abs();
            }
            if total_sensitivity > best_sensitivity {
                best_sensitivity = total_sensitivity;
                best_idx = Some(i);
            }
        }

        if best_idx.is_none() {
            return self.select_max_margin_fallback(obj_bounds, thresholds, verified_mask);
        }

        best_idx
    }

    /// Select critical objective for conjunctive mode when each A row is already
    /// the spec-guided objective row from a dense spec matrix.
    fn select_critical_conjunctive_spec_rows(
        &self,
        intermediate: &GraphAlphaCrownIntermediate,
        obj_bounds: &[(f32, f32)],
        thresholds: &[f32],
        verified_mask: &[bool],
    ) -> Option<usize> {
        let mut best_idx = None;
        let mut best_sensitivity = 0.0f32;

        for objective_idx in 0..obj_bounds.len() {
            if objective_idx < verified_mask.len() && verified_mask[objective_idx] {
                continue;
            }

            let mut total_sensitivity = 0.0f32;
            for entry in &self.entries {
                let a_column = match intermediate.beta_a_column(&entry.node_name, entry.neuron_idx)
                {
                    Some(column) => column,
                    None => continue,
                };
                let Some(&sensitivity) = a_column.get(objective_idx) else {
                    continue;
                };
                total_sensitivity += sensitivity.abs();
            }
            if total_sensitivity > best_sensitivity {
                best_sensitivity = total_sensitivity;
                best_idx = Some(objective_idx);
            }
        }

        if best_idx.is_none() {
            return self.select_max_margin_fallback(obj_bounds, thresholds, verified_mask);
        }

        best_idx
    }

    fn select_max_margin_fallback(
        &self,
        obj_bounds: &[(f32, f32)],
        thresholds: &[f32],
        verified_mask: &[bool],
    ) -> Option<usize> {
        let mut fallback_idx = None;
        let mut fallback_margin = f32::NEG_INFINITY;
        for (i, ((lb, _ub), &threshold)) in obj_bounds.iter().zip(thresholds).enumerate() {
            if i < verified_mask.len() && verified_mask[i] {
                continue;
            }
            let margin = lb - threshold;
            let margin = if margin.is_nan() {
                f32::NEG_INFINITY
            } else {
                margin
            };
            if margin > fallback_margin {
                fallback_margin = margin;
                fallback_idx = Some(i);
            }
        }
        fallback_idx
    }

    /// Compute gradients for all β entries w.r.t. a specific objective.
    ///
    /// Returns max |gradient| across all entries (for convergence checking).
    fn compute_gradients_for_objective(
        &mut self,
        intermediate: &GraphAlphaCrownIntermediate,
        objective: &[f32],
    ) -> f32 {
        let mut max_grad = 0.0f32;

        for entry in &mut self.entries {
            let a_column = match intermediate.beta_a_column(&entry.node_name, entry.neuron_idx) {
                Some(column) => column,
                None => {
                    entry.grad = 0.0;
                    continue;
                }
            };

            let mut sensitivity = 0.0f32;
            for (j, &a_jk) in a_column.iter().enumerate() {
                let c_j = if j < objective.len() {
                    objective[j]
                } else {
                    0.0
                };
                sensitivity += c_j * a_jk;
            }

            let grad = -entry.sign * sensitivity;
            entry.grad = grad;
            max_grad = nan_propagating_max(max_grad, grad.abs());
        }

        max_grad
    }

    /// Compute gradients when each intermediate row is already the target objective.
    fn compute_gradients_for_spec_row(
        &mut self,
        intermediate: &GraphAlphaCrownIntermediate,
        objective_idx: usize,
    ) -> f32 {
        let mut max_grad = 0.0f32;

        for entry in &mut self.entries {
            let a_column = match intermediate.beta_a_column(&entry.node_name, entry.neuron_idx) {
                Some(column) => column,
                None => {
                    entry.grad = 0.0;
                    continue;
                }
            };
            let Some(&sensitivity) = a_column.get(objective_idx) else {
                entry.grad = 0.0;
                continue;
            };

            let grad = -entry.sign * sensitivity;
            entry.grad = grad;
            max_grad = nan_propagating_max(max_grad, grad.abs());
        }

        max_grad
    }
}
