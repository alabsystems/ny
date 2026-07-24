// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proactive cut generation for BICCOS-lite.
//!
//! Generates cutting planes before branch-and-bound starts, based on initial
//! pre-activation bounds. This gives the optimizer lambda variables to work
//! with from iteration 0, rather than waiting for domains to verify.

use std::sync::atomic::Ordering;

use ny_core::Result;
use tracing::debug;

use crate::beta_crown::bab_cuts::{CutKind, CutMetadata, GraphCutTerm, GraphCuttingPlane};

use super::GraphCutPool;

impl GraphCutPool {
    /// Generate proactive cuts for unstable ReLUs before BaB starts.
    ///
    /// This implements BICCOS-lite: instead of waiting for domains to verify
    /// (which may never happen on hard instances), we generate cuts proactively
    /// based on the initial bounds.
    ///
    /// The cuts encode pairwise neuron implications:
    /// - For pairs of unstable neurons in consecutive layers
    /// - Encodes: "at least one of these neurons must be stable"
    ///
    /// This gives the optimizer lambda variables to work with from iteration 0.
    ///
    /// # Arguments
    /// * `graph` - The GraphNetwork being verified
    /// * `node_bounds` - Initial bounds for each node (from alpha-CROWN or CROWN-IBP)
    /// * `max_cuts` - Maximum number of proactive cuts to generate
    ///
    /// # Returns
    /// Number of cuts generated, or error if cut construction fails
    pub fn generate_proactive_cuts(
        &mut self,
        graph: &crate::GraphNetwork,
        node_bounds: &std::collections::HashMap<String, std::sync::Arc<crate::BoundedTensor>>,
        max_cuts: usize,
    ) -> Result<usize> {
        use crate::layers::Layer;

        // Find all ReLU nodes and their unstable neurons in deterministic order.
        let mut relu_unstable: Vec<(String, Vec<usize>, std::sync::Arc<crate::BoundedTensor>)> =
            Vec::new();
        let mut seen_nodes = std::collections::HashSet::new();

        for name in &graph.node_order {
            let Some(node) = graph.nodes.get(name) else {
                continue;
            };
            if !seen_nodes.insert(name) {
                continue;
            }
            if !matches!(node.layer, Layer::ReLU(_)) {
                continue;
            }

            // Get bounds for this ReLU's input (pre-activation).
            let Some(input_name) = node.inputs.first() else {
                debug!("Skipping ReLU node '{}' with no inputs", name);
                continue;
            };
            let Some(bounds) = node_bounds.get(input_name) else {
                debug!(
                    "Skipping ReLU node '{}' due to missing input bounds for '{}'",
                    name, input_name
                );
                continue;
            };

            let flat = bounds.flatten();
            let mut unstable = Vec::new();

            for i in 0..flat.len() {
                let lb = flat.lower()[[i]];
                let ub = flat.upper()[[i]];
                if !lb.is_finite() || !ub.is_finite() {
                    continue;
                }
                // Neuron is unstable if it crosses zero.
                if lb < 0.0 && ub > 0.0 {
                    unstable.push(i);
                }
            }

            if !unstable.is_empty() {
                relu_unstable.push((name.clone(), unstable, std::sync::Arc::clone(bounds)));
            }
        }

        if relu_unstable.is_empty() {
            return Ok(0);
        }

        let mut cuts_generated = 0;

        // Strategy 1: Generate single-neuron "indicator" cuts for highly unstable neurons
        // These encode the constraint that z_i in {0, 1} (binary indicator)
        // We prioritize neurons with balanced pre-activation bounds (close to zero)
        for (node_name, unstable_neurons, bounds) in &relu_unstable {
            if cuts_generated >= max_cuts {
                break;
            }

            // Sort neurons by "instability score" (how close to zero the bounds are).
            let flat = bounds.flatten();
            let mut scored_neurons: Vec<(usize, f32)> = unstable_neurons
                .iter()
                .filter_map(|&idx| {
                    if idx < flat.len() {
                        let lb = flat.lower()[[idx]];
                        let ub = flat.upper()[[idx]];
                        if !lb.is_finite() || !ub.is_finite() {
                            return None;
                        }
                        // Score = how "balanced" the neuron is (closer to 0 = higher score).
                        // Use |lb| / (|lb| + ub) as balance metric.
                        let denom = lb.abs() + ub;
                        if denom <= 0.0 || !denom.is_finite() {
                            return None;
                        }
                        let balance = lb.abs() / denom;
                        let score = 1.0 - (balance - 0.5).abs() * 2.0; // 1.0 = perfectly balanced
                        Some((idx, score))
                    } else {
                        None
                    }
                })
                .collect();

            // Sort by score descending (most balanced first, NaN last — #2995), then by index for determinism.
            scored_neurons.sort_by(|a, b| {
                crate::cmp_utils::nan_last_descending_cmp(&a.1, &b.1).then_with(|| a.0.cmp(&b.0))
            });

            // Take top neurons for single-neuron cuts.
            for (neuron_idx, _score) in scored_neurons.iter().take(5) {
                if cuts_generated >= max_cuts {
                    break;
                }

                // Create a single-neuron "active" cut.
                // This encodes: z_i should be pushed toward 1 (active).
                let active_cut = GraphCuttingPlane::new(
                    vec![GraphCutTerm {
                        node_name: node_name.clone(),
                        neuron_idx: *neuron_idx,
                        coefficient: 1.0, // Active constraint
                    }],
                    0.5,  // Midpoint bias
                    0.01, // Small initial lambda (will be optimized)
                    0,    // Proactive cut (depth 0)
                    CutMetadata::new(0, CutKind::Proactive),
                )?;

                self.total_generated += 1;
                active_cut.metadata.reset(
                    self.iter_counter.load(Ordering::Relaxed),
                    CutKind::Proactive,
                );
                if self.insert_cut(active_cut, CutKind::Proactive) {
                    cuts_generated += 1;
                }
            }
        }

        // Strategy 2: Generate pairwise cuts between consecutive layers
        // These encode implications like: "if neuron i is active, neuron j is likely active"
        if cuts_generated < max_cuts && relu_unstable.len() >= 2 {
            for window in relu_unstable.windows(2) {
                if cuts_generated >= max_cuts {
                    break;
                }

                let (node1, unstable1, _) = &window[0];
                let (node2, unstable2, _) = &window[1];
                let Some(node2_def) = graph.nodes.get(node2) else {
                    continue;
                };
                if !node2_def.inputs.iter().any(|input| input == node1) {
                    continue;
                }

                // Create pairwise cuts for a subset of neurons
                let pairs_to_create = ((max_cuts - cuts_generated) / 4).clamp(1, 10);
                let mut pairs_created = 0;

                for &n1 in unstable1.iter().take(5) {
                    if pairs_created >= pairs_to_create {
                        break;
                    }
                    for &n2 in unstable2.iter().take(2) {
                        if pairs_created >= pairs_to_create || cuts_generated >= max_cuts {
                            break;
                        }

                        // Create cut: z_n1 + z_n2 >= 1 (at least one active).
                        // Encoded as: sum(coeffs * z) >= bias
                        // With coeffs=[+1, +1], bias=1 means: z_n1 + z_n2 >= 1
                        let pairwise_cut = GraphCuttingPlane::new(
                            vec![
                                GraphCutTerm {
                                    node_name: node1.clone(),
                                    neuron_idx: n1,
                                    coefficient: 1.0,
                                },
                                GraphCutTerm {
                                    node_name: node2.clone(),
                                    neuron_idx: n2,
                                    coefficient: 1.0,
                                },
                            ],
                            1.0,  // At least one active.
                            0.01, // Small initial lambda
                            0,    // Proactive cut
                            CutMetadata::new(0, CutKind::Proactive),
                        )?;

                        self.total_generated += 1;
                        pairwise_cut.metadata.reset(
                            self.iter_counter.load(Ordering::Relaxed),
                            CutKind::Proactive,
                        );
                        if self.insert_cut(pairwise_cut, CutKind::Proactive) {
                            cuts_generated += 1;
                            pairs_created += 1;
                        }
                    }
                }
            }
        }

        debug!(
            "Generated {} proactive cuts from {} ReLU nodes with unstable neurons",
            cuts_generated,
            relu_unstable.len()
        );

        Ok(cuts_generated)
    }
}
