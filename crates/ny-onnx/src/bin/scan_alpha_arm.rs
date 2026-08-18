// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! Capability-gate scanner for the alpha reference-bounds collector.
//!
//! Statically classifies which arm of
//! `collect_alpha_reference_bounds_with_engine_and_source` a model would take,
//! using the verifier's own graph build
//! (`CompoundNodePolicy::DecomposeNormalization`). MEASUREMENT TOOLING ONLY: it
//! computes no bounds and nothing on a verdict path calls it.
//!
//! Why it exists: #linearizenn-dense-dag-ref was "a tightening that already
//! existed was gated off for a whole graph class", and its blast radius was
//! first estimated from ONE model per category — which missed a third category
//! the new arm reaches. Capability gates are per-GRAPH and categories are
//! heterogeneous, so the only honest blast radius is a per-MODEL sweep. Run
//! this over a whole benchmark tree before moving any `is_dag` /
//! `has_conv_layers` / node-count / binary-op guard:
//!
//! ```text
//! find benchmarks/vnncomp2025 -name '*.onnx' | sort \
//!   | xargs -n1 target/release/scan_alpha_arm > arms.tsv
//! ```
//!
//! Columns: model, node count, activation count, `is_dag`, `has_conv`,
//! `has_conv2d`, carries a binary-relaxation op, carries any binary op, arm.
//! The arm column mirrors the branch order in `graph_alpha/bounds/alpha.rs`
//! for `fix_interm_bounds = true` (the default).

use ny_onnx::{load_onnx, CompoundNodePolicy, GraphNetworkOptions};
use ny_propagate::{BoundPropagation, Layer};
use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!("model\tnodes\tacts\tis_dag\thas_conv\thas_conv2d\tbinary_relax\tbinary_op\tarm");
    for path in &args {
        match classify(path) {
            Ok(line) => println!("{line}"),
            Err(e) => println!("{path}\tERR\t-\t-\t-\t-\t-\t-\t{e}"),
        }
    }
}

fn classify(path: &str) -> Result<String, String> {
    let model = load_onnx(path).map_err(|e| format!("load: {e}"))?;
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        })
        .map_err(|e| format!("graph: {e}"))?;
    let exec_order = graph
        .exec_order()
        .map_err(|e| format!("exec: {e}"))?
        .to_vec();

    // Replica of GraphNetwork::is_sequential_graph
    let mut consumer_count: HashMap<String, usize> = HashMap::new();
    consumer_count.insert("_input".to_string(), 0);
    let mut has_binary_op = false;
    for name in &exec_order {
        if let Some(node) = graph.node(name) {
            if node.layer().is_binary() {
                has_binary_op = true;
            }
            for input_name in node.inputs() {
                *consumer_count.entry(input_name.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut branching = false;
    for (name, count) in &consumer_count {
        if name == "_input" {
            if *count > 1 {
                branching = true;
            }
        } else if *count > 1 {
            branching = true;
        }
    }
    let is_sequential = exec_order.is_empty() || (!has_binary_op && !branching);
    let is_dag = !is_sequential;

    let act_count = exec_order
        .iter()
        .filter(|n| {
            graph
                .node(n)
                .is_some_and(|node| node.layer().requires_pre_activation_bounds())
        })
        .count();

    let binary_relax = graph.node_names().iter().any(|n| {
        graph.node(n).is_some_and(|node| {
            matches!(
                node.layer(),
                Layer::MatMul(_)
                    | Layer::BilinearCrown(_)
                    | Layer::MulBinary(_)
                    | Layer::GroupNorm(_)
            )
        })
    });
    let has_conv = graph.node_names().iter().any(|n| {
        graph.node(n).is_some_and(|node| {
            matches!(
                node.layer(),
                Layer::Conv2d(_)
                    | Layer::Conv1d(_)
                    | Layer::ConvTranspose2d(_)
                    | Layer::ConvTranspose1d(_)
            )
        })
    });
    let has_conv2d = graph.node_names().iter().any(|n| {
        graph
            .node(n)
            .is_some_and(|node| matches!(node.layer(), Layer::Conv2d(_)))
    });
    let has_conv2d_exec = exec_order.iter().any(|n| {
        graph
            .node(n)
            .is_some_and(|node| matches!(node.layer(), Layer::Conv2d(_)))
    });

    let n_nodes = graph.num_nodes();
    let should_use_crown_ibp = !binary_relax;
    let per_node_ok = should_use_crown_ibp && n_nodes <= 50;

    // fix_interm_bounds = true (default)
    let deep_seq = !is_dag && act_count >= 3 && should_use_crown_ibp;
    let dense_dag = is_dag && act_count >= 3 && !has_conv && per_node_ok;
    let conv_dag = is_dag && has_conv2d_exec;

    let arm = if deep_seq {
        "CROWN_IBP(deep_seq)"
    } else if dense_dag {
        "CROWN_IBP(dense_dag)"
    } else if conv_dag {
        "FORWARD_LINEAR(conv_dag)"
    } else {
        "IBP"
    };

    Ok(format!(
        "{path}\t{n_nodes}\t{act_count}\t{is_dag}\t{has_conv}\t{has_conv2d}\t{binary_relax}\t{has_binary_op}\t{arm}"
    ))
}
