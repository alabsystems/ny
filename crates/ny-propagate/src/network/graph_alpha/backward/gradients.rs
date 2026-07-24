// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::GraphAlphaCrownIntermediate;
use crate::network::core::GraphNetwork;

use ndarray::Array1;
use ny_tensor::BoundedTensor;

impl GraphNetwork {
    /// Compute chain-rule gradients for GraphNetwork DAG α-CROWN.
    ///
    /// For each unstable neuron i in ReLU node k:
    /// ∂(output_lower_sum)/∂α_k[i] = Σ_j A_to_relu[j,i] × input_contribution[i]
    ///
    /// Where:
    /// - A_to_relu[j,i] is the coefficient from output j to neuron i (before ReLU k)
    /// - input_contribution captures how the neuron value affects downstream computation
    ///
    /// This chains gradients through all DOWNSTREAM layers in the DAG (the A
    /// matrix at the ReLU is exact), but it is still the LOCAL approximation
    /// UPSTREAM: it substitutes `pre_lower[i]` for the true factor — the
    /// RELAXED-linear forward evaluation of neuron i's pre-activation at the
    /// final row's concretization argmin x*. The finite-difference oracle
    /// (`backward/true_grad_oracle_tests.rs`) shows the true gradient is
    /// `max(ν,0)·ĥ_i(x*)`; this local rule matches it only when x* happens to
    /// minimize neuron i's own pre-activation, and can have the WRONG SIGN
    /// otherwise (it degraded the post-split wide-α ascent in both lr signs —
    /// #cifar100 task 11). It remains a useful warmup heuristic at the root,
    /// where it empirically converges; gradients are non-soundness-critical.
    pub(in crate::network::graph_alpha) fn compute_graph_chain_rule_gradients(
        &self,
        _input: &BoundedTensor,
        relu_nodes: &[String],
        intermediate: &GraphAlphaCrownIntermediate,
    ) -> Vec<Array1<f32>> {
        let mut gradients: Vec<Array1<f32>> = Vec::with_capacity(relu_nodes.len());

        for relu_name in relu_nodes {
            // Get A matrix at this ReLU (before ReLU applied)
            let a_at_relu = match intermediate.a_at_relu(relu_name) {
                Some(a) => a,
                None => {
                    // No intermediate stored for this ReLU — use pre-ReLU bounds
                    // to determine correct gradient length (#1937). A length-1
                    // fallback would panic in alpha update when the ReLU has >1 neuron.
                    let n = intermediate
                        .pre_relu_bounds(relu_name)
                        .map(|(lower, _)| lower.len())
                        .unwrap_or_else(|| {
                            tracing::warn!(
                                "AnalyticChain: missing both A matrix and pre-ReLU bounds for '{}' (#1937)",
                                relu_name
                            );
                            0
                        });
                    gradients.push(Array1::zeros(n));
                    continue;
                }
            };

            // Get pre-ReLU bounds
            let (pre_lower, pre_upper) = match intermediate.pre_relu_bounds(relu_name) {
                Some(b) => b,
                None => {
                    gradients.push(Array1::zeros(a_at_relu.ncols()));
                    continue;
                }
            };

            let n_neurons = pre_lower.len();
            let num_outputs = a_at_relu.nrows();
            let mut grad = Array1::<f32>::zeros(n_neurons);

            // For each neuron in this ReLU layer
            for i in 0..n_neurons {
                let l = pre_lower[i];
                let u = pre_upper[i];

                // Guard: non-finite pre-ReLU bounds cannot produce meaningful gradients.
                // IEEE-754: NaN comparisons return false, so `l >= 0.0 || u <= 0.0`
                // would fail for NaN bounds, treating them as "unstable" and flowing
                // NaN into gradient arithmetic. Explicitly skip non-finite.
                // Mirrors sequential path guard in helpers.rs (#2809).
                if !l.is_finite() || !u.is_finite() {
                    continue;
                }

                // Only unstable neurons (l < 0 < u) have non-zero gradient
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Compute gradient contribution from all output dimensions
                // For lower relaxation y >= α*x with x ∈ [l, u] where l < 0 < u:
                // - Contribution to lower bound = A[j,i] * α * min(x) = A[j,i] * α * l
                // - Gradient ∂bound/∂α = A[j,i] * l
                // Note: l < 0 for unstable neurons, so gradient is typically negative
                // when A[j,i] > 0, meaning increasing α decreases the lower bound.
                let mut grad_i = 0.0f32;

                for j in 0..num_outputs {
                    let a_ji = a_at_relu[[j, i]];

                    // Guard: non-finite A coefficient cannot produce meaningful
                    // gradient contributions. Mirrors sequential path guard (#2809).
                    if !a_ji.is_finite() {
                        continue;
                    }

                    // When A >= 0, lower relaxation uses y >= α*x
                    // The binding point is x = l (lower bound), not u
                    // because we minimize α*x over [l,u] with α >= 0 and l < 0
                    if a_ji > 0.0 {
                        // Lower relaxation active: y >= α*x
                        // Contribution to lower bound: A[j,i] * α * l
                        // Gradient w.r.t. α: A[j,i] * l
                        grad_i += a_ji * l;
                    }
                    // When A < 0, upper relaxation y <= (u/(u-l))*(x-l) is used
                    // This doesn't depend on α, so gradient is 0
                }

                grad[i] = grad_i;
            }

            gradients.push(grad);
        }

        gradients
    }
}
