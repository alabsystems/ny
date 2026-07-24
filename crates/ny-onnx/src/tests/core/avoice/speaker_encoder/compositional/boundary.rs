// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::collections::{HashMap, HashSet};

/// Discovered ECAPA composition boundary at the MFA concat node.
///
/// The three `block_outputs` are the SE-Res2Net block output node names
/// (x2, x3, x4) in topological order. The `mfa_concat` is the unique
/// three-input Concat node on the main path that consumes all three.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EcapaCompositionBoundary {
    /// Block output node names [x2, x3, x4] in topological order.
    pub(super) block_outputs: [String; 3],
    /// The MFA concat node name.
    pub(super) mfa_concat: String,
    /// The raw ONNX concat axis (may be negative, e.g. -2 for the ECAPA MFA
    /// concat). Resolve against the actual rank of the tensors being
    /// concatenated (via `ny_core::resolve_axis`) before use — the extracted
    /// block-output bounds are batch-squeezed relative to the original graph, so
    /// a fixed positive axis (or a raw `as usize` cast of a negative axis, which
    /// wraps to `usize::MAX - 1`) would be wrong.
    pub(super) concat_axis: i64,
}

/// Discover the ECAPA MFA composition boundary in the given graph.
///
/// Algorithm (from the design doc):
/// 1. Build a topological index for deterministic ordering.
/// 2. Compute the ancestor closure of the output node by reverse DFS.
/// 3. Find Concat nodes in the ancestor set with exactly 3 dynamic inputs.
/// 4. Keep only candidates whose three inputs form a strict ancestor chain.
/// 5. Require exactly one surviving candidate.
pub(super) fn discover_ecapa_composition_boundary(
    graph: &GraphNetwork,
) -> Result<EcapaCompositionBoundary, String> {
    let topo_order = graph
        .topological_sort()
        .map_err(|e| format!("topological sort failed: {e}"))?;
    let topo_index: HashMap<&str, usize> = topo_order
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect();

    let output_name = graph.output_name();
    let ancestors = ancestor_closure(graph, output_name);

    let mut candidates: Vec<(&str, [String; 3], i64)> = Vec::new();
    for name in &topo_order {
        if !ancestors.contains(name.as_str()) {
            continue;
        }
        let node = graph.node(name).unwrap();
        if !matches!(node.layer(), Layer::Concat(_)) {
            continue;
        }
        let inputs: Vec<&String> = node
            .inputs()
            .iter()
            .filter(|inp| *inp != ny_propagate::NETWORK_INPUT)
            .collect();
        if inputs.len() != 3 {
            continue;
        }

        let mut sorted_inputs: Vec<&String> = inputs;
        sorted_inputs
            .sort_by_key(|inp| topo_index.get(inp.as_str()).copied().unwrap_or(usize::MAX));

        let concat_layer = match node.layer() {
            Layer::Concat(concat) => concat,
            _ => unreachable!(),
        };
        // Keep the raw (possibly negative) ONNX axis; it is resolved against the
        // concrete tensor rank at the point of use. Casting a negative axis to
        // `usize` here is the historical MFA-concat panic (`-2 as usize`).
        let raw_axis = concat_layer.axis;

        candidates.push((
            name.as_str(),
            [
                sorted_inputs[0].clone(),
                sorted_inputs[1].clone(),
                sorted_inputs[2].clone(),
            ],
            raw_axis,
        ));
    }

    let mut surviving: Vec<(&str, [String; 3], i64)> = Vec::new();
    for (concat_name, inputs, raw_axis) in &candidates {
        let reaches_01 = reaches(graph, &inputs[0], &inputs[1]);
        let reaches_02 = reaches(graph, &inputs[0], &inputs[2]);
        let reaches_12 = reaches(graph, &inputs[1], &inputs[2]);
        if reaches_01 && reaches_02 && reaches_12 {
            surviving.push((concat_name, inputs.clone(), *raw_axis));
        }
    }

    // The ASP context concat ([x, mu_expand, sg_expand]) also passes the
    // strict-chain filter when the mean/std Expand nodes survive conversion
    // as dynamic nodes: mu derives from x and sg derives from expanded mu.
    // The MFA concat is distinguished structurally as the most-upstream
    // candidate: its output flows (through the mfa conv/relu) into every
    // downstream candidate, while no candidate reaches back into it. Keep
    // only candidates that reach all other candidates.
    if surviving.len() > 1 {
        let names: Vec<&str> = surviving.iter().map(|(name, _, _)| *name).collect();
        surviving = surviving
            .iter()
            .filter(|(name, _, _)| {
                names
                    .iter()
                    .all(|other| other == name || reaches(graph, name, other))
            })
            .cloned()
            .collect();
    }

    if surviving.len() != 1 {
        return Err(format!(
            "expected exactly 1 MFA concat candidate with strict ancestor chain, \
             found {}; candidates={:?}",
            surviving.len(),
            surviving
                .iter()
                .map(|(name, inputs, _)| format!("{name}: {:?}", inputs))
                .collect::<Vec<_>>()
        ));
    }

    let (concat_name, inputs, raw_axis) = surviving.into_iter().next().unwrap();
    Ok(EcapaCompositionBoundary {
        block_outputs: inputs,
        mfa_concat: concat_name.to_string(),
        concat_axis: raw_axis,
    })
}

/// Compute the set of all ancestor node names reachable from `target` by
/// following input edges backward. Includes `target` itself.
fn ancestor_closure<'a>(graph: &'a GraphNetwork, target: &str) -> HashSet<&'a str> {
    let mut visited = HashSet::new();
    let mut stack = vec![target.to_string()];
    while let Some(name) = stack.pop() {
        if name == ny_propagate::NETWORK_INPUT {
            continue;
        }
        if let Some(node) = graph.node(&name) {
            let node_name: &str = node.name();
            if !visited.insert(node_name) {
                continue;
            }
            for inp in node.inputs() {
                if inp != ny_propagate::NETWORK_INPUT && !visited.contains(inp.as_str()) {
                    stack.push(inp.clone());
                }
            }
        }
    }
    visited
}

/// Check whether `from` reaches `to` by forward traversal through the graph.
/// This is equivalent to checking if `from` is an ancestor of `to`.
fn reaches(graph: &GraphNetwork, from: &str, to: &str) -> bool {
    ancestor_closure(graph, to).contains(from)
}
