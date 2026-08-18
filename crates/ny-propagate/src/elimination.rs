// Copyright 2026 Andrew Yates.
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// Copyright 2026 Andrew Yates.
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dead neuron elimination: construct optimized network from analysis.
//!
//! Given an [`AnalysisResult`] from [`crate::analysis::analyze_neurons`], removes
//! provably dead and constant neurons from a sequential network, producing a
//! smaller equivalent network and an [`EliminationCertificate`] documenting
//! every transformation.
//!
//! # Algorithm
//!
//! For each (Linear, ReLU) layer pair in a sequential network:
//! - **Dead neurons** (upper <= 0): remove the weight row from the preceding
//!   Linear layer and the corresponding column from the following Linear layer.
//! - **Constant neurons** (|upper - lower| < eps): absorb the constant output
//!   into the following Linear layer's bias, then remove the row/column.
//! - **Always-active neurons** (lower >= 0): keep the neuron; if *all* neurons
//!   in a ReLU layer are always-active, the ReLU is identity and can be removed,
//!   enabling fusion of adjacent Linear layers via [`merge_linear`].
//! - **Unstable neurons**: kept unchanged.
//!
//! Reference: alpha-beta-CROWN prunes dead neurons during BaB preprocessing.
//! See `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/operators/relu.py`.

use crate::analysis::{AnalysisResult, NeuronStatus};
use crate::equivalence::{verify_equivalence, EquivalenceResult};
use crate::layers::linear::{merge_linear, LinearLayer};
use crate::layers::{Layer, ReLULayer};
use crate::network::{GraphNetwork, Network};
use crate::types::PropagationConfig;
use ndarray::{Array1, Axis};
use ny_core::{Bound, NyError, Result};

/// Deferred column operations to apply to the next Linear layer.
///
/// When dead/constant neurons are removed from a ReLU layer, the corresponding
/// columns must be removed from the *following* Linear layer's weight matrix,
/// and constant values must be absorbed into its bias.
struct PendingColumnOps {
    /// Column indices to keep in the next Linear layer (sorted).
    keep_cols: Vec<usize>,
    /// Bias adjustments from absorbed constants: (original_col_index, constant_value).
    bias_absorb: Vec<(usize, f32)>,
}

/// What happened to a neuron during elimination.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum EliminationAction {
    /// Neuron was dead (upper <= 0) and its row/column was removed.
    RemovedDead,
    /// Neuron was always-active (lower >= 0). If the entire ReLU was
    /// always-active, the ReLU layer was removed.
    RemovedAlwaysActive,
    /// Neuron had approximately constant output; its value was folded into
    /// the next layer's bias.
    AbsorbedConstant {
        /// The constant value that was absorbed.
        value: f32,
    },
    /// Neuron was kept (unstable or always-active in a mixed ReLU).
    Kept,
}

/// Record of what happened to one neuron during elimination.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EliminationEntry {
    /// Original layer index of the ReLU in the source network.
    pub layer_index: usize,
    /// Neuron index within that ReLU layer.
    pub neuron_index: usize,
    /// Classification from analysis.
    pub status: NeuronStatus,
    /// Pre-activation lower bound that justified the classification.
    pub lower_bound: f32,
    /// Pre-activation upper bound that justified the classification.
    pub upper_bound: f32,
    /// Action taken during elimination.
    pub action: EliminationAction,
}

/// Certificate documenting every transformation applied during elimination.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EliminationCertificate {
    /// Per-neuron records.
    pub entries: Vec<EliminationEntry>,
    /// Total neurons in the original network (at ReLU layers).
    pub neurons_before: usize,
    /// Total neurons remaining after elimination.
    pub neurons_after: usize,
    /// Number of layers in the original network.
    pub layers_before: usize,
    /// Number of layers in the optimized network.
    pub layers_after: usize,
}

impl EliminationCertificate {
    /// Fraction of neurons eliminated (0.0 = none, 1.0 = all).
    #[must_use]
    pub fn elimination_fraction(&self) -> f32 {
        if self.neurons_before == 0 {
            return 0.0;
        }
        1.0 - (self.neurons_after as f32 / self.neurons_before as f32)
    }
}

/// Construct an optimized network with dead and constant neurons removed.
///
/// The optimized network is semantically equivalent to the original for
/// all inputs within the region used for the analysis. The returned
/// certificate documents every transformation for auditing and equivalence
/// proofs.
///
/// # Evidence Status
///
/// The dead/always-active neuron classification is the intended C012 proof
/// target in clean. Until that theorem is promoted to `KernelProved`, this
/// optimization relies on the local interval checks here plus regression tests,
/// not on a public clean proof badge.
///
/// # Arguments
///
/// * `network` -- The sequential network to optimize.
/// * `analysis` -- Result of [`crate::analysis::analyze_neurons`] on the same network.
///
/// # Errors
///
/// Returns [`NyError::InvalidSpec`] if:
/// - The analysis refers to a layer index beyond the network's layers.
/// - A Linear layer referenced during row/column removal has incompatible shape.
pub fn eliminate_dead_neurons(
    network: &Network,
    analysis: &AnalysisResult,
) -> Result<(Network, EliminationCertificate)> {
    let layers = network.layers();

    // Group analysis neurons by their ReLU layer_index.
    let mut relu_neurons: std::collections::BTreeMap<usize, Vec<_>> =
        std::collections::BTreeMap::new();
    for na in &analysis.neurons {
        relu_neurons.entry(na.layer_index).or_default().push(na);
    }

    let mut state = EliminationState {
        pending: None,
        optimized_layers: Vec::with_capacity(layers.len()),
        entries: Vec::with_capacity(analysis.neurons.len()),
        neurons_after: 0,
    };
    let mut i = 0;

    while i < layers.len() {
        // Check for (Linear, ReLU) pair.
        let is_linear_relu = i + 1 < layers.len()
            && matches!(&layers[i], Layer::Linear(_))
            && matches!(&layers[i + 1], Layer::ReLU(_));

        if is_linear_relu {
            process_linear_relu_pair(layers, i, &relu_neurons, &mut state)?;
            i += 2;
        } else {
            process_non_relu_layer(layers, i, &mut state)?;
            i += 1;
        }
    }

    // Try to merge consecutive Linear layers (from always-active ReLU removal).
    let merged_layers = merge_consecutive_linears(state.optimized_layers);

    let mut result = Network::new();
    for layer in &merged_layers {
        result.add_layer(layer.clone());
    }

    let certificate = EliminationCertificate {
        entries: state.entries,
        neurons_before: analysis.total_neurons,
        neurons_after: state.neurons_after,
        layers_before: layers.len(),
        layers_after: merged_layers.len(),
    };

    Ok((result, certificate))
}

/// Mutable accumulator state threaded through elimination passes (#2622).
struct EliminationState {
    pending: Option<PendingColumnOps>,
    optimized_layers: Vec<Layer>,
    entries: Vec<EliminationEntry>,
    neurons_after: usize,
}

/// Process a (Linear, ReLU) pair during elimination.
fn process_linear_relu_pair(
    layers: &[Layer],
    i: usize,
    relu_neurons: &std::collections::BTreeMap<usize, Vec<&crate::analysis::NeuronAnalysis>>,
    state: &mut EliminationState,
) -> Result<()> {
    let relu_layer_idx = i + 1;
    let linear = match &layers[i] {
        Layer::Linear(l) => l,
        _ => unreachable!(),
    };

    // Apply pending column ops from a previous ReLU elimination.
    let linear = if let Some(ref ops) = state.pending {
        apply_column_ops(linear, ops)?
    } else {
        linear.clone()
    };
    state.pending = None;

    let neuron_analyses = relu_neurons.get(&relu_layer_idx);
    if let Some(analyses) = neuron_analyses {
        classify_and_eliminate(&linear, analyses, relu_layer_idx, state)?;
    } else {
        // No analysis for this ReLU -- pass through unchanged.
        state.optimized_layers.push(Layer::Linear(linear));
        state.optimized_layers.push(layers[i + 1].clone());
    }
    Ok(())
}

/// Classify neurons in one ReLU layer and build elimination actions.
fn classify_and_eliminate(
    linear: &LinearLayer,
    analyses: &[&crate::analysis::NeuronAnalysis],
    _relu_layer_idx: usize,
    state: &mut EliminationState,
) -> Result<()> {
    let mut keep_indices: Vec<usize> = Vec::new();
    let mut has_unstable = false;
    let mut bias_absorptions: Vec<(usize, f32)> = Vec::new();

    for na in analyses.iter() {
        let action = match na.status {
            NeuronStatus::Dead => EliminationAction::RemovedDead,
            NeuronStatus::AlwaysActive => {
                keep_indices.push(na.neuron_index);
                EliminationAction::Kept
            }
            NeuronStatus::Constant(val) => {
                let relu_val = val.max(0.0);
                bias_absorptions.push((na.neuron_index, relu_val));
                EliminationAction::AbsorbedConstant { value: relu_val }
            }
            NeuronStatus::Unstable => {
                keep_indices.push(na.neuron_index);
                has_unstable = true;
                EliminationAction::Kept
            }
        };

        state.entries.push(EliminationEntry {
            layer_index: na.layer_index,
            neuron_index: na.neuron_index,
            status: na.status,
            lower_bound: na.lower_bound,
            upper_bound: na.upper_bound,
            action,
        });
    }

    if keep_indices.is_empty() {
        state.pending = Some(PendingColumnOps {
            keep_cols: Vec::new(),
            bias_absorb: bias_absorptions,
        });
        return Ok(());
    }

    let reduced_linear = select_rows(linear, &keep_indices)?;

    if !has_unstable {
        state.optimized_layers.push(Layer::Linear(reduced_linear));
        for entry in state.entries.iter_mut().rev().take(analyses.len()) {
            if entry.action == EliminationAction::Kept && entry.status == NeuronStatus::AlwaysActive
            {
                entry.action = EliminationAction::RemovedAlwaysActive;
            }
        }
    } else {
        state.optimized_layers.push(Layer::Linear(reduced_linear));
        state.optimized_layers.push(Layer::ReLU(ReLULayer));
    }

    state.neurons_after += keep_indices.len();
    state.pending = Some(PendingColumnOps {
        keep_cols: keep_indices,
        bias_absorb: bias_absorptions,
    });
    Ok(())
}

/// Process a layer that is not part of a (Linear, ReLU) pair.
fn process_non_relu_layer(layers: &[Layer], i: usize, state: &mut EliminationState) -> Result<()> {
    match &layers[i] {
        Layer::Linear(l) => {
            let adjusted = if let Some(ref ops) = state.pending {
                apply_column_ops(l, ops)?
            } else {
                l.clone()
            };
            state.pending = None;
            state.optimized_layers.push(Layer::Linear(adjusted));
        }
        other => {
            state.pending = None;
            state.optimized_layers.push(other.clone());
        }
    }
    Ok(())
}

/// Select specific rows from a LinearLayer (keeping only `indices` rows).
fn select_rows(layer: &LinearLayer, indices: &[usize]) -> Result<LinearLayer> {
    if indices.is_empty() {
        return Err(NyError::InvalidSpec(
            "Cannot create LinearLayer with zero output features".to_string(),
        ));
    }

    let new_weight = layer.weight().select(Axis(0), indices);
    let new_bias = layer.bias().map(|b| b.select(Axis(0), indices));
    LinearLayer::new(new_weight, new_bias)
}

/// Apply deferred column operations to a LinearLayer.
///
/// 1. Absorb constant values into the bias.
/// 2. Select only the `keep_cols` columns (or return error if empty).
fn apply_column_ops(layer: &LinearLayer, ops: &PendingColumnOps) -> Result<LinearLayer> {
    let out_features = layer.out_features();

    // Step 1: absorb constants into bias.
    let mut bias = layer
        .bias()
        .cloned()
        .unwrap_or_else(|| Array1::zeros(out_features));
    for &(col, val) in &ops.bias_absorb {
        if col < layer.in_features() {
            for k in 0..out_features {
                bias[k] += layer.weight()[[k, col]] * val;
            }
        }
    }

    // Step 2: select columns.
    if ops.keep_cols.is_empty() {
        return Err(NyError::InvalidSpec(
            "All input neurons eliminated; cannot create zero-width Linear layer".to_string(),
        ));
    }

    let new_weight = layer.weight().select(Axis(1), &ops.keep_cols);
    LinearLayer::new(new_weight, Some(bias))
}

/// Merge consecutive Linear layers in the layer list.
fn merge_consecutive_linears(layers: Vec<Layer>) -> Vec<Layer> {
    if layers.is_empty() {
        return layers;
    }

    let mut merged: Vec<Layer> = Vec::with_capacity(layers.len());
    for layer in layers {
        let should_merge = matches!(
            (&merged.last(), &layer),
            (Some(Layer::Linear(_)), Layer::Linear(_))
        );
        if should_merge {
            let prev = merged.pop().expect("checked non-empty");
            if let (Layer::Linear(l1), Layer::Linear(l2)) = (prev, &layer) {
                merged.push(Layer::Linear(merge_linear(&l1, l2)));
            }
        } else {
            merged.push(layer);
        }
    }
    merged
}

/// Bundled result of elimination with verified equivalence.
///
/// Returned by [`eliminate_and_verify`] after performing dead neuron elimination
/// and verifying the optimized network produces equivalent outputs via
/// [`verify_equivalence`].
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub struct EliminationVerification {
    /// The optimized network with dead/constant neurons removed.
    pub optimized: Network,
    /// Certificate documenting every transformation applied.
    pub certificate: EliminationCertificate,
    /// Result of formal equivalence verification between the original and
    /// optimized networks.
    pub equivalence: EquivalenceResult,
}

/// Eliminate dead neurons and verify equivalence of the optimized network.
///
/// This is the recommended high-level entry point for dead neuron elimination.
/// It performs three steps:
///
/// 1. Eliminate dead/constant/always-active neurons via [`eliminate_dead_neurons`].
/// 2. Convert both original and optimized networks to [`GraphNetwork`] form.
/// 3. Run [`verify_equivalence`] to formally verify that the optimized network
///    produces outputs within `epsilon` of the original for all inputs in the
///    specified input region.
///
/// # Arguments
///
/// * `network` -- The sequential network to optimize.
/// * `analysis` -- Result of [`crate::analysis::analyze_neurons`] on the same network.
/// * `input_bounds` -- Per-element input bounds defining the verification region.
/// * `epsilon` -- Maximum allowed output difference for equivalence verification.
/// * `config` -- Propagation configuration for the equivalence verifier.
///
/// # Errors
///
/// Returns [`NyError`] if elimination fails, graph conversion fails, or
/// equivalence verification encounters an internal error.
pub fn eliminate_and_verify(
    network: &Network,
    analysis: &AnalysisResult,
    input_bounds: &[Bound],
    epsilon: f32,
    config: PropagationConfig,
) -> Result<EliminationVerification> {
    let (optimized, certificate) = eliminate_dead_neurons(network, analysis)?;

    let graph_orig = GraphNetwork::from_sequential(network)?;
    let graph_opt = GraphNetwork::from_sequential(&optimized)?;

    let equivalence = verify_equivalence(&graph_orig, &graph_opt, input_bounds, epsilon, config)?;

    Ok(EliminationVerification {
        optimized,
        certificate,
        equivalence,
    })
}

#[cfg(test)]
#[path = "elimination_tests.rs"]
mod tests;
