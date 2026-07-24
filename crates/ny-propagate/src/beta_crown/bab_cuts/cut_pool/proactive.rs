// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proactive cut generation for BICCOS-lite.
//!
//! Generates cutting planes before BaB starts, based on initial bounds.
//! Single-neuron "indicator" cuts for highly unstable neurons and
//! pairwise cuts between consecutive ReLU layers.

use std::sync::atomic::Ordering;

use ny_core::Result;

use super::CutPool;
use crate::beta_crown::bab_cuts::{CutKind, CutMetadata, CutTerm, CuttingPlane};

impl CutPool {
    /// Generate proactive cuts for unstable ReLUs before BaB starts.
    ///
    /// This implements BICCOS-lite for sequential networks: instead of waiting
    /// for domains to verify (which may never happen on hard instances), we
    /// generate cuts proactively based on the initial bounds.
    ///
    /// The cuts encode pairwise neuron implications:
    /// - For pairs of unstable neurons in consecutive ReLU layers
    /// - Encodes: "at least one of these neurons must be stable"
    ///
    /// This gives the optimizer lambda variables to work with from iteration 0.
    ///
    /// # Arguments
    /// * `network` - The Network being verified
    /// * `layer_bounds` - IBP/CROWN-IBP bounds for each layer (output of each layer)
    /// * `max_cuts` - Maximum number of proactive cuts to generate
    ///
    /// # Returns
    /// Number of cuts generated, or error if cut construction fails
    pub fn generate_proactive_cuts(
        &mut self,
        network: &crate::Network,
        layer_bounds: &[crate::BoundedTensor],
        max_cuts: usize,
    ) -> Result<usize> {
        use crate::layers::Layer;

        // Find all ReLU layers and their unstable neurons
        // Note: layer_bounds[i] is the OUTPUT of layer i, so for ReLU at index i,
        // we need the PREVIOUS layer's output (i-1) for pre-activation bounds
        let mut relu_unstable: Vec<(usize, Vec<usize>)> = Vec::new();

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !matches!(layer, Layer::ReLU(_)) {
                continue;
            }

            // Get pre-activation bounds (output of previous layer)
            // For first layer, we can't have ReLU (would be no-op on input)
            if layer_idx == 0 {
                continue;
            }

            let pre_bounds = &layer_bounds[layer_idx - 1];
            let flat = pre_bounds.flatten();
            let mut unstable = Vec::new();

            for i in 0..flat.len() {
                let lb = flat.lower()[[i]];
                let ub = flat.upper()[[i]];
                if !lb.is_finite() || !ub.is_finite() {
                    continue;
                }
                // Neuron is unstable if it crosses zero
                if lb < 0.0 && ub > 0.0 {
                    unstable.push(i);
                }
            }

            if !unstable.is_empty() {
                relu_unstable.push((layer_idx, unstable));
            }
        }

        if relu_unstable.is_empty() {
            return Ok(0);
        }

        let mut cuts_generated = 0;

        // Strategy 1: Generate single-neuron "indicator" cuts for highly unstable neurons
        // These encode the constraint that z_i ∈ {0, 1} (binary indicator)
        // We prioritize neurons with balanced pre-activation bounds (close to zero)
        for &(layer_idx, ref unstable_neurons) in &relu_unstable {
            if cuts_generated >= max_cuts {
                break;
            }

            let pre_bounds = &layer_bounds[layer_idx - 1];
            let flat = pre_bounds.flatten();

            // Score neurons by "instability" (how close to zero the bounds are)
            let mut scored_neurons: Vec<(usize, f32)> = unstable_neurons
                .iter()
                .filter_map(|&idx| {
                    if idx < flat.len() {
                        let lb = flat.lower()[[idx]];
                        let ub = flat.upper()[[idx]];
                        if !lb.is_finite() || !ub.is_finite() {
                            return None;
                        }
                        // Score = how "balanced" the neuron is (closer to 0 = higher score)
                        // Use |lb| / (|lb| + ub) as balance metric
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

            // Take top neurons for single-neuron cuts
            for (neuron_idx, _score) in scored_neurons.iter().take(5) {
                if cuts_generated >= max_cuts {
                    break;
                }

                // Create a single-neuron "active" cut
                // This encodes: z_i should be pushed toward 1 (active)
                let active_cut = CuttingPlane::new(
                    vec![CutTerm {
                        layer_idx,
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

        // Strategy 2: Generate pairwise cuts between consecutive ReLU layers
        // These encode implications like: "if neuron i is active, neuron j is likely active"
        if cuts_generated < max_cuts && relu_unstable.len() >= 2 {
            for window in relu_unstable.windows(2) {
                if cuts_generated >= max_cuts {
                    break;
                }

                let (layer1, unstable1) = &window[0];
                let (layer2, unstable2) = &window[1];

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

                        // Create cut: z_n1 + z_n2 >= 1 (at least one active)
                        // Encoded as: sum(coeffs * z) >= bias
                        // With coeffs=[+1, +1], bias=1 means: z_n1 + z_n2 >= 1
                        let pairwise_cut = CuttingPlane::new(
                            vec![
                                CutTerm {
                                    layer_idx: *layer1,
                                    neuron_idx: n1,
                                    coefficient: 1.0,
                                },
                                CutTerm {
                                    layer_idx: *layer2,
                                    neuron_idx: n2,
                                    coefficient: 1.0,
                                },
                            ],
                            1.0,  // bias
                            0.01, // lambda
                            0,    // Proactive cut (depth 0)
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

        Ok(cuts_generated)
    }
}
