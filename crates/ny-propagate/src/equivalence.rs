// Copyright 2026 Andrew Yates.
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// Copyright 2026 Andrew Yates.
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Equivalence verification for neural networks via difference network construction.
//!
//! Given two networks f and g, verify that ||f(x) - g(x)|| < eps for all x in
//! an input region by constructing a difference network h(x) = f(x) - g(x) and
//! verifying that h's outputs lie within [-eps, eps].
//!
//! # Algorithm
//!
//! 1. Build a merged DAG where both f and g share `NETWORK_INPUT`.
//! 2. Prefix all node names with `a_` (network f) and `b_` (network g) to avoid collisions.
//! 3. Add a final `SubLayer` node computing `a_output - b_output`.
//! 4. Run bound propagation (IBP, CROWN, or alpha-CROWN) on the merged network.
//! 5. Check if all output bounds fall within [-eps, eps].
//!
//! Reference: "Equivalence Checking of Neural Networks" (Teuber et al., 2021)
//! uses a similar difference-network approach for Linf equivalence verification.

use ny_core::{Bound, NyError, Result, VerificationResult, VerificationSpec};

use crate::layers::binary_ops::SubLayer;
use crate::layers::Layer;
use crate::network::{GraphNetwork, GraphNode, NETWORK_INPUT};
use crate::types::PropagationConfig;
use crate::verifier::Verifier;

/// Result of an equivalence verification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EquivalenceResult {
    /// The two networks are verified equivalent within epsilon.
    /// `bound` is the tightest worst-case output difference proven by the verifier.
    Equivalent {
        /// Worst-case proven output difference (max of |lower|, |upper| across outputs).
        bound: f64,
    },
    /// The verifier could not prove equivalence. The bound exceeds epsilon.
    /// This does NOT mean the networks are definitely different -- it means
    /// the verifier's bounds were too loose to prove equivalence.
    NotEquivalent {
        /// Worst-case bound on output difference found by the verifier.
        worst_case_bound: f64,
    },
    /// Verification timed out or returned an inconclusive result.
    Unknown {
        /// Best bound achieved before timeout/inconclusive result.
        best_bound: f64,
    },
}

/// Build a difference network h(x) = f(x) - g(x) from two `GraphNetwork`s.
///
/// Both networks share the same `NETWORK_INPUT`. Node names are prefixed with
/// `a_` and `b_` respectively to prevent collisions. A final `SubLayer` node
/// computes the element-wise difference of the two output nodes.
///
/// # Errors
///
/// Returns `NyError::InvalidSpec` if:
/// - Either network has no output node set
/// - Output dimensions are statically detectable as mismatched (e.g., both
///   output layers are Linear with different `out_features`)
/// - Node name collision occurs after prefixing
///
/// Note: when output dimensions cannot be statically inferred (e.g., non-Linear
/// output layers), the mismatch will be caught at propagation time by
/// `SubLayer::propagate_ibp` returning a `ShapeMismatch` error. For a
/// guaranteed up-front check, use [`verify_equivalence`] which runs a cheap IBP
/// forward pass to validate output dimensions before constructing the difference
/// network.
pub fn build_difference_network(
    network_a: &GraphNetwork,
    network_b: &GraphNetwork,
) -> Result<GraphNetwork> {
    if network_a.output_name().is_empty() {
        return Err(NyError::InvalidSpec(
            "network_a has no output node set".to_string(),
        ));
    }
    if network_b.output_name().is_empty() {
        return Err(NyError::InvalidSpec(
            "network_b has no output node set".to_string(),
        ));
    }

    // Best-effort static check: if both output layers are Linear, verify their
    // output dimensions match. This catches the common case early with a clear
    // error message instead of a confusing ShapeMismatch from SubLayer later.
    if let (Some(node_a), Some(node_b)) = (
        network_a.node(network_a.output_name()),
        network_b.node(network_b.output_name()),
    ) {
        if let (Layer::Linear(lin_a), Layer::Linear(lin_b)) = (node_a.layer(), node_b.layer()) {
            let dim_a = lin_a.out_features();
            let dim_b = lin_b.out_features();
            if dim_a != dim_b {
                return Err(NyError::InvalidSpec(format!(
                    "Cannot build difference network: output dimensions differ \
                     (network_a output={dim_a}, network_b output={dim_b})"
                )));
            }
        }
    }

    // Choose collision-free prefixes for the two networks.
    let names_a: Vec<&str> = network_a.node_names().iter().map(|s| s.as_str()).collect();
    let names_b: Vec<&str> = network_b.node_names().iter().map(|s| s.as_str()).collect();
    let (prefix_a, prefix_b) = find_collision_free_prefixes(&names_a, &names_b);

    let mut diff_graph = GraphNetwork::new();

    // Add all nodes from network_a with prefix_a
    for name in network_a.node_names() {
        let node = network_a
            .node(name)
            .ok_or_else(|| NyError::InternalError(format!("node '{name}' missing in a")))?;

        let prefixed_name = format!("{prefix_a}{name}");
        let prefixed_inputs: Vec<String> = node
            .inputs()
            .iter()
            .map(|input| {
                if input == NETWORK_INPUT {
                    NETWORK_INPUT.to_string()
                } else {
                    format!("{prefix_a}{input}")
                }
            })
            .collect();

        let prefixed_node = GraphNode::new(prefixed_name, node.layer().clone(), prefixed_inputs);
        diff_graph.try_add_node(prefixed_node)?;
    }

    // Add all nodes from network_b with prefix_b
    for name in network_b.node_names() {
        let node = network_b
            .node(name)
            .ok_or_else(|| NyError::InternalError(format!("node '{name}' missing in b")))?;

        let prefixed_name = format!("{prefix_b}{name}");
        let prefixed_inputs: Vec<String> = node
            .inputs()
            .iter()
            .map(|input| {
                if input == NETWORK_INPUT {
                    NETWORK_INPUT.to_string()
                } else {
                    format!("{prefix_b}{input}")
                }
            })
            .collect();

        let prefixed_node = GraphNode::new(prefixed_name, node.layer().clone(), prefixed_inputs);
        diff_graph.try_add_node(prefixed_node)?;
    }

    // Add the final Sub node: diff_output = a_output - b_output
    let a_output = format!("{prefix_a}{}", network_a.output_name());
    let b_output = format!("{prefix_b}{}", network_b.output_name());
    let diff_node = GraphNode::binary("diff_output", Layer::Sub(SubLayer), a_output, b_output);
    diff_graph.try_add_node(diff_node)?;
    diff_graph.set_output("diff_output");

    // Carry the loaders' DECLARED SHAPES onto the prefixed copies (and the
    // shared `_input` sentinel when both sources agree). Exact downstream
    // encoders (graph-MIP) use declared shapes for index/broadcast math and
    // fail closed without them — the stitched net previously dropped them
    // all, so e.g. the ACAS input-normalization SubConstant declined every
    // encode of a difference network (#relational-bab edge escalation).
    for (name, shape) in &network_a.declared_shapes {
        if name == NETWORK_INPUT {
            if network_b
                .declared_shape(NETWORK_INPUT)
                .is_none_or(|b| b == shape.as_slice())
            {
                diff_graph.set_declared_shape(NETWORK_INPUT, shape.clone());
            }
        } else {
            diff_graph.set_declared_shape(format!("{prefix_a}{name}"), shape.clone());
        }
    }
    for (name, shape) in &network_b.declared_shapes {
        if name != NETWORK_INPUT {
            diff_graph.set_declared_shape(format!("{prefix_b}{name}"), shape.clone());
        }
    }
    if let Some(shape) = network_a.declared_shape(network_a.output_name()) {
        diff_graph.set_declared_shape("diff_output", shape.to_vec());
    }

    Ok(diff_graph)
}

/// Candidate prefix pairs for difference network construction.
const PREFIX_CANDIDATES: &[(&str, &str)] = &[
    ("a_", "b_"),
    ("net_a_", "net_b_"),
    ("left_", "right_"),
    ("first_", "second_"),
];

/// Reserved node names that prefixed names must not collide with.
const RESERVED_NAMES: &[&str] = &["diff_output", NETWORK_INPUT];

/// Find collision-free prefix pair for two networks.
///
/// A collision occurs when a prefixed node name from one network matches a
/// prefixed name from the other, or when a prefixed name matches a reserved
/// sentinel name (NETWORK_INPUT, diff_output).
fn find_collision_free_prefixes<'a>(names_a: &[&str], names_b: &[&str]) -> (&'a str, &'a str) {
    for &(pa, pb) in PREFIX_CANDIDATES {
        let mut collision = false;
        // Check if any prefixed name from A collides with reserved names
        for name in names_a {
            let prefixed = format!("{pa}{name}");
            if RESERVED_NAMES.contains(&prefixed.as_str()) {
                collision = true;
                break;
            }
        }
        if collision {
            continue;
        }
        // Check if any prefixed name from B collides with reserved names
        for name in names_b {
            let prefixed = format!("{pb}{name}");
            if RESERVED_NAMES.contains(&prefixed.as_str()) {
                collision = true;
                break;
            }
        }
        if collision {
            continue;
        }
        // Check cross-collision: prefixed A name == prefixed B name
        // This can only happen if pa==pb and names overlap, or if
        // pa+name_a == pb+name_b for some pair. The latter is extremely
        // unlikely with distinct prefixes, but check anyway.
        for name_a in names_a {
            let pa_name = format!("{pa}{name_a}");
            for name_b in names_b {
                let pb_name = format!("{pb}{name_b}");
                if pa_name == pb_name {
                    collision = true;
                    break;
                }
            }
            if collision {
                break;
            }
        }
        if !collision {
            return (pa, pb);
        }
    }
    // Fallback: use hash-based prefixes (extremely unlikely to reach here)
    // Using a simple counter suffix on "net_" as a last resort.
    ("__diff_a_", "__diff_b_")
}

/// Verify that two networks produce equivalent outputs within epsilon.
///
/// Constructs a difference network h(x) = f(x) - g(x) and verifies that
/// all outputs of h lie within [-eps, eps] for all inputs in the specified region.
///
/// # Arguments
///
/// * `network_a` - First network (f)
/// * `network_b` - Second network (g)
/// * `input_bounds` - Per-element input bounds defining the verification region
/// * `epsilon` - Maximum allowed output difference
/// * `config` - Propagation configuration (IBP, CROWN, alpha-CROWN, etc.)
///
/// # Errors
///
/// Returns `NyError` on invalid inputs or propagation failure.
pub fn verify_equivalence(
    network_a: &GraphNetwork,
    network_b: &GraphNetwork,
    input_bounds: &[Bound],
    epsilon: f32,
    config: PropagationConfig,
) -> Result<EquivalenceResult> {
    if epsilon <= 0.0 {
        return Err(NyError::InvalidSpec("epsilon must be positive".to_string()));
    }

    // Validate output dimensions match via cheap IBP forward on each network.
    let input_bt = Verifier::bounds_to_tensor(input_bounds, None)?;
    let ibp_a = network_a.propagate_ibp(&input_bt)?;
    let ibp_b = network_b.propagate_ibp(&input_bt)?;
    let dim_a = ibp_a.lower().len();
    let dim_b = ibp_b.lower().len();
    if dim_a != dim_b {
        return Err(NyError::InvalidSpec(format!(
            "Cannot verify equivalence: output dimensions differ (a={dim_a}, b={dim_b})"
        )));
    }

    let diff_network = build_difference_network(network_a, network_b)?;

    // Count the number of output dimensions.
    // We need output_bounds for the spec, but we don't know the output dimension
    // at construction time. Use IBP on the diff network to determine the output size,
    // then build the actual spec.
    //
    // Alternative: infer from the last Linear layer's weight shape, but that's fragile.
    // Instead, just propagate IBP cheaply to get the output shape.
    let input_bt = Verifier::bounds_to_tensor(input_bounds, None)?;
    let ibp_output = diff_network.propagate_ibp(&input_bt)?;
    let num_outputs = ibp_output.lower().len();

    // Build output spec: all outputs must be in [-eps, eps]
    let output_bounds: Vec<Bound> = (0..num_outputs)
        .map(|_| Bound::new(-epsilon, epsilon))
        .collect();

    let spec = VerificationSpec::new(input_bounds.to_vec(), output_bounds)?;

    let verifier = Verifier::new(config);
    let result = verifier.verify_graph(&diff_network, &spec)?;

    Ok(interpret_result(&result))
}

/// Compute worst-case output difference from a list of output bounds.
fn worst_case_from_bounds(bounds: &[Bound]) -> f64 {
    bounds
        .iter()
        .map(|b| b.lower().abs().max(b.upper().abs()) as f64)
        .fold(0.0_f64, f64::max)
}

/// Interpret a `VerificationResult` into an `EquivalenceResult`.
fn interpret_result(result: &VerificationResult) -> EquivalenceResult {
    match result {
        VerificationResult::Verified {
            output_bounds,
            provenance: _,
            proof: _,
            actual_method: _,
        } => {
            let bound = worst_case_from_bounds(output_bounds);
            EquivalenceResult::Equivalent { bound }
        }
        VerificationResult::Unknown { bounds, .. } => {
            let worst_case_bound = worst_case_from_bounds(bounds);
            EquivalenceResult::NotEquivalent { worst_case_bound }
        }
        VerificationResult::Violated { .. } => {
            // Violated means a counterexample was found (should not happen for
            // bound propagation methods, but handle gracefully)
            EquivalenceResult::NotEquivalent {
                worst_case_bound: f64::INFINITY,
            }
        }
        VerificationResult::Timeout { partial_bounds, .. } => {
            let best_bound = partial_bounds
                .as_ref()
                .map(|b| worst_case_from_bounds(b))
                .unwrap_or(f64::INFINITY);
            EquivalenceResult::Unknown { best_bound }
        }
    }
}

#[cfg(test)]
#[path = "equivalence_tests.rs"]
mod tests;
