// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared constraint lookup building and pre-constraint application.
//!
//! This module deduplicates the constraint setup code that was previously
//! copy-pasted across three functions in `constraints/mod.rs`:
//! - `propagate_crown_with_graph_constraints`
//! - `compute_constrained_forward_bounds`
//! - `propagate_crown_with_graph_constraints_storing_intermediates`

use std::collections::HashMap;

use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::branching::{GenBabConstraint, GraphNeuronConstraint};
use crate::{GraphNetwork, Layer};

/// Constraint lookup maps built from ReLU and GenBaB neuron constraints.
///
/// - `by_relu`: Maps ReLU node name -> (neuron_idx -> is_active). Used during the
///   backward pass to select the correct CROWN relaxation slope for each constrained neuron.
/// - `pre`: Maps pre-activation node name -> list of (neuron_idx, is_active, relu_name).
///   Used during the forward pass to tighten bounds at pre-activation nodes before
///   their consumer ReLU nodes are evaluated.
/// - `pre_genbab`: Maps pre-activation node name -> list of (neuron_idx, split_point,
///   is_upper_branch, genbab_node_name). Used during the forward pass to tighten bounds
///   at pre-activation nodes of general nonlinearities (GeLU, Sigmoid, Tanh, etc.).
///   (#2399)
pub(in crate::beta_crown::engine::graph::propagation) struct ConstraintLookups {
    pub by_relu: HashMap<String, HashMap<usize, bool>>,
    pub pre: HashMap<String, Vec<(usize, bool, String)>>,
    pub pre_genbab: HashMap<String, Vec<(usize, f32, bool, String)>>,
}

/// Build constraint lookup maps from ReLU and GenBaB neuron constraints.
///
/// Validates that each ReLU constraint references an existing ReLU node and each GenBaB
/// constraint references an existing node in the graph. Returns an error if a constraint
/// references a missing node or a ReLU constraint references a non-ReLU node.
///
/// # Arguments
/// * `constraints` - The ReLU neuron constraints from a domain's split history
/// * `genbab_constraints` - The GenBaB constraints for general nonlinearities (#2399)
/// * `graph` - The graph network for node lookups
pub(super) fn build_constraint_lookups(
    constraints: &[GraphNeuronConstraint],
    genbab_constraints: &[GenBabConstraint],
    graph: &GraphNetwork,
) -> Result<ConstraintLookups> {
    let mut by_relu: HashMap<String, HashMap<usize, bool>> = HashMap::new();
    let mut pre: HashMap<String, Vec<(usize, bool, String)>> = HashMap::new();
    let mut pre_genbab: HashMap<String, Vec<(usize, f32, bool, String)>> = HashMap::new();

    for c in constraints {
        by_relu
            .entry(c.node_name.clone())
            .or_default()
            .insert(c.neuron_idx, c.is_active);

        let relu_node = graph.nodes.get(&c.node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Graph constraint references missing node '{}'",
                c.node_name
            ))
        })?;
        if !matches!(relu_node.layer, Layer::ReLU(_) | Layer::Sign(_)) {
            return Err(NyError::InvalidSpec(format!(
                "Graph constraint references non-ReLU/Sign node '{}'",
                c.node_name
            )));
        }
        // #2098: Reject nodes with empty inputs rather than fabricating "_input".
        let pre_name = relu_node.inputs.first().cloned().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ReLU node '{}' has no inputs — cannot determine pre-activation",
                c.node_name
            ))
        })?;
        pre.entry(pre_name)
            .or_default()
            .push((c.neuron_idx, c.is_active, c.node_name.clone()));
    }

    // #2399: Build GenBaB pre-activation lookup.
    // GenBaB constraints specify a split_point for general nonlinearities (GeLU, Sigmoid, etc.).
    // We map each constraint to the pre-activation node of the constrained nonlinear node,
    // so the forward pass can tighten pre-activation bounds at the split_point.
    for c in genbab_constraints {
        let genbab_node = graph.nodes.get(c.node_name()).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "GenBaB constraint references missing node '{}'",
                c.node_name()
            ))
        })?;
        // Resolve the pre-activation node for the input this split actually
        // subdivided. For unary nonlinearities `input_index` is None/0 (the only
        // input). For binary McCormick ops (MulBinary z = x·y, BilinearCrown
        // z = Q@Kᵀ) a split on input 1 MUST clamp input 1's node — not
        // `inputs.first()`. The prior hard-coded `.first()` misrouted every
        // second-input split to the first input: a hard index error when the two
        // inputs have different lengths ("child propagation failed" → Unknown at
        // depth 0), or — worse — a silent clamp of the wrong neuron of the wrong
        // input that excludes reachable values (unsound) while never tightening
        // the intended input's McCormick envelope. (#mul-genbab)
        let input_idx = c.input_index().unwrap_or(0);
        let pre_name = genbab_node.inputs.get(input_idx).cloned().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "GenBaB node '{}' has no input at index {} — cannot determine pre-activation",
                c.node_name(),
                input_idx
            ))
        })?;
        pre_genbab.entry(pre_name).or_default().push((
            c.neuron_idx(),
            c.split_point(),
            c.is_upper_branch(),
            c.node_name().to_string(),
        ));
    }

    Ok(ConstraintLookups {
        by_relu,
        pre,
        pre_genbab,
    })
}

/// Apply pre-activation constraints to tighten bounds.
///
/// For each constraint:
/// - Active (is_active=true): clamp lower bound to max(lower, 0.0) — neuron >= 0
/// - Inactive (is_active=false): clamp upper bound to min(upper, 0.0) — neuron <= 0
///
/// Returns an error if any constraint produces an infeasible domain (lower > upper).
pub(super) fn apply_pre_constraints(
    bounds: &BoundedTensor,
    constraints: &[(usize, bool, String)],
) -> Result<BoundedTensor> {
    if constraints.is_empty() {
        return Ok(bounds.clone());
    }

    let flat = bounds.flatten();
    let shape = bounds.shape().to_vec();
    let mut lower = flat.lower().clone();
    let mut upper = flat.upper().clone();

    for (neuron_idx, is_active, relu_name) in constraints {
        if *neuron_idx >= flat.len() {
            return Err(NyError::InvalidSpec(format!(
                "Constraint out of range: relu='{}' idx={} len={}",
                relu_name,
                neuron_idx,
                flat.len()
            )));
        }
        if *is_active {
            // NaN-safe: propagate NaN instead of silently clamping to 0.0 (#2643)
            lower[[*neuron_idx]] = nan_propagating_max(lower[[*neuron_idx]], 0.0);
        } else {
            upper[[*neuron_idx]] = nan_propagating_min(upper[[*neuron_idx]], 0.0);
        }
        if lower[[*neuron_idx]] > upper[[*neuron_idx]] {
            // #2926: Constraint application produced inverted bounds — domain is empty.
            return Err(NyError::InfeasibleDomain(format!(
                "pre-constraint at relu='{}' idx={}: [{}, {}]",
                relu_name,
                neuron_idx,
                lower[[*neuron_idx]],
                upper[[*neuron_idx]]
            )));
        }
    }

    let lower_arr = lower
        .into_shape_clone(ndarray::IxDyn(&shape))
        .map_err(|e| NyError::InvalidSpec(format!("shape error: {}", e)))?;
    let upper_arr = upper
        .into_shape_clone(ndarray::IxDyn(&shape))
        .map_err(|e| NyError::InvalidSpec(format!("shape error: {}", e)))?;
    BoundedTensor::new(lower_arr, upper_arr)
}

/// Apply GenBaB pre-activation constraints to tighten bounds at arbitrary split points.
///
/// Unlike ReLU constraints which split at 0.0, GenBaB constraints split at an arbitrary
/// `split_point` for general nonlinearities (GeLU, Sigmoid, Tanh, etc.).
///
/// For each constraint:
/// - Upper branch (is_upper_branch=true): clamp lower bound to max(lower, split_point)
///   — neuron >= split_point
/// - Lower branch (is_upper_branch=false): clamp upper bound to min(upper, split_point)
///   — neuron <= split_point
///
/// Returns an error if any constraint produces an infeasible domain (lower > upper).
/// Multiple constraints on the same neuron (dual-sided pinching) are applied sequentially;
/// infeasibility from crossing split points is detected after each constraint.
///
/// # Arguments
/// * `bounds` - The current pre-activation bounds to tighten
/// * `constraints` - Tuples of `(neuron_idx, split_point, is_upper_branch, genbab_node_name)`
///
/// Reference: α,β-CROWN GenBaB branching applies the same split_point clamping
/// in the forward pass. See `auto_LiRPA/branching_domains.py`.
/// (#2399)
pub(super) fn apply_genbab_pre_constraints(
    bounds: &BoundedTensor,
    constraints: &[(usize, f32, bool, String)],
) -> Result<BoundedTensor> {
    if constraints.is_empty() {
        return Ok(bounds.clone());
    }

    let flat = bounds.flatten();
    let shape = bounds.shape().to_vec();
    let mut lower = flat.lower().clone();
    let mut upper = flat.upper().clone();

    for (neuron_idx, split_point, is_upper_branch, genbab_node_name) in constraints {
        if *neuron_idx >= flat.len() {
            return Err(NyError::InvalidSpec(format!(
                "GenBaB constraint out of range: node='{}' idx={} len={}",
                genbab_node_name,
                neuron_idx,
                flat.len()
            )));
        }
        if *is_upper_branch {
            // Upper branch: x >= split_point → tighten lower bound
            // NaN-safe: propagate NaN instead of silently clamping (#2643)
            lower[[*neuron_idx]] = nan_propagating_max(lower[[*neuron_idx]], *split_point);
        } else {
            // Lower branch: x <= split_point → tighten upper bound
            upper[[*neuron_idx]] = nan_propagating_min(upper[[*neuron_idx]], *split_point);
        }
        if lower[[*neuron_idx]] > upper[[*neuron_idx]] {
            // #2926: Constraint application produced inverted bounds — domain is empty.
            return Err(NyError::InfeasibleDomain(format!(
                "genbab pre-constraint at node='{}' idx={} split={}: [{}, {}]",
                genbab_node_name,
                neuron_idx,
                split_point,
                lower[[*neuron_idx]],
                upper[[*neuron_idx]]
            )));
        }
    }

    let lower_arr = lower
        .into_shape_clone(ndarray::IxDyn(&shape))
        .map_err(|e| NyError::InvalidSpec(format!("shape error: {}", e)))?;
    let upper_arr = upper
        .into_shape_clone(ndarray::IxDyn(&shape))
        .map_err(|e| NyError::InvalidSpec(format!("shape error: {}", e)))?;
    BoundedTensor::new(lower_arr, upper_arr)
}
