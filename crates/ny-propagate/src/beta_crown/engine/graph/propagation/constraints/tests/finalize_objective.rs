// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #finalize-objective: the constrained backward's terminal bound must be a
//! bound on what the pass was SEEDED with.
//!
//! `initialize_constrained_backward` seeds the walk with the spec matrix `C`
//! (rows `num_specs`), the objective row `c` (one row), or the identity, so a
//! walk that reaches the network input publishes a bound on `C·out`, `c·out`
//! or `out`. `finalize_constrained_backward` has a second terminal branch for
//! a walk that never accumulated anything at `NETWORK_INPUT`: it published the
//! raw forward (IBP) box of `out` as the result. That box is a bound on `out`,
//! not on `c·out`: for a single-output net and `c = [-1]` its lower endpoint
//! is `min out` where the true minimum of `-out` is `-max out`, and for a
//! multi-row spec it has the wrong shape entirely. `BetaCrownVerifier::
//! propagate_crown_with_graph_beta` and `evaluate_graph_child_bounds` read the
//! scalar objective bound off it with `lower_scalar`/`upper_scalar`.
//!
//! The branch condition is `input_accumulated == false` after the reverse walk
//! over `exec_order`. With the full topological `exec_order` every production
//! caller passes, every well-formed graph accumulates at the input (the
//! topological sort rejects a dangling input reference, and a zero-input node
//! fails `require_unary_input`), so the branch is latent there; these tests
//! reach it through the shared core directly with an `exec_order` that stops
//! before the input, the only state in which the branch runs.
//!
//! The same walk has the partial form of the defect: when SOME path reaches
//! the input but another is still pending at a node the walk never processed,
//! finalization concretized the input block alone and dropped the pending
//! contribution. That must refuse, not publish.

use ndarray::{arr1, arr2, Array2};

use crate::beta_crown::{GraphCrownContext, GraphSplitHistory};
use crate::layers::AddLayer;
use crate::{
    BetaCrownConfig, BetaCrownVerifier, BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer,
    ReLULayer,
};

use super::super::backward::{BackwardMode, BackwardParams};
use super::super::lookups::build_constraint_lookups;
use super::super::patches::ConstrainedPatchesPolicy;

/// Input samples covering `x ∈ [-1, 1]`, all exact in f64.
fn samples() -> impl Iterator<Item = f64> {
    (-20..=20).map(|i| f64::from(i) / 20.0)
}

/// `out = relu(x)`, `x ∈ [-1, 1]`: one output, forward box `[0, 1]`.
fn single_relu() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
    graph.set_output("relu1");
    let input =
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).expect("box");
    (graph, input)
}

fn standard_mode() -> BackwardMode {
    BackwardMode::Standard
}

fn storing_mode(graph: &GraphNetwork) -> BackwardMode {
    BackwardMode::StoringIntermediates {
        lookups: Box::new(build_constraint_lookups(&[], &[], graph).expect("lookups")),
    }
}

/// Run the shared constrained backward on `graph` with the given seed and an
/// explicit `exec_order`.
fn run_backward(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    exec_order: &[String],
    objective: Option<&[f32]>,
    spec_matrix: Option<&Array2<f32>>,
    mode: BackwardMode,
) -> ny_core::Result<BoundedTensor> {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_la_warm_start: false,
        ..Default::default()
    });
    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);
    let (mut bounds_cache, constrained_input) = verifier
        .compute_constrained_forward_bounds(graph, input, &history, None, None)
        .expect("forward constrained bounds");
    let params = BackwardParams {
        graph,
        constrained_input: &constrained_input,
        exec_order,
        context: &context,
        beta_state: None,
        objective,
        spec_matrix,
        seed_cache: None,
        capture_linear_bounds: false,
        deadline: None,
        patches_policy: ConstrainedPatchesPolicy::selective_matrix_reentry(),
    };
    verifier
        .backward_crown_constrained(&params, &mut bounds_cache, mode)
        .map(|result| result.output_bounds)
}

/// Every row `r` of the result must enclose `rows[r] · out(x)` at every
/// sample, and the result must have exactly one entry per seeded row.
fn violations(
    site: &str,
    bounds: &BoundedTensor,
    rows: &[f32],
    truth: impl Fn(f64) -> f64,
) -> Vec<String> {
    let flat = bounds.flatten();
    let (lower, upper) = (flat.lower(), flat.upper());
    println!(
        "[finalize-objective] site={site} rows={rows:?} bounds=[{:?}, {:?}]",
        lower.iter().copied().collect::<Vec<f32>>(),
        upper.iter().copied().collect::<Vec<f32>>()
    );
    if flat.len() != rows.len() {
        return vec![format!(
            "{site}: result has {} entries for {} seeded rows (shape {:?})",
            flat.len(),
            rows.len(),
            bounds.shape()
        )];
    }
    let mut out = Vec::new();
    for (row_idx, &c) in rows.iter().enumerate() {
        let (lo, hi) = (f64::from(lower[[row_idx]]), f64::from(upper[[row_idx]]));
        for x in samples() {
            let value = f64::from(c) * truth(x);
            if lo > value || hi < value {
                out.push(format!(
                    "{site} row={row_idx} c={c}: x={x} -> {value} outside [{lo:e}, {hi:e}]"
                ));
            }
        }
    }
    out
}

/// RED on the pre-fix tree: the never-reached-input branch published the raw
/// output box `[0, 1]` for `c = [-1]` (true range `[-1, 0]`), `c = [2]`
/// (true range `[0, 2]`), and a one-entry box for the two-row spec
/// `[[-1], [2]]`. The identity seed keeps the raw box, which IS its bound.
#[test]
fn constrained_backward_unreached_input_maps_the_output_box_through_the_objective() {
    let (graph, input) = single_relu();
    let relu = |x: f64| x.max(0.0);
    let no_walk: Vec<String> = Vec::new();
    let mut found = Vec::new();

    for c in [-1.0f32, 2.0] {
        for (label, mode) in [
            ("standard", standard_mode()),
            ("storing", storing_mode(&graph)),
        ] {
            let bounds = run_backward(&graph, &input, &no_walk, Some(&[c]), None, mode)
                .expect("the never-reached-input branch publishes a bound");
            found.extend(violations(
                &format!("objective/{label}"),
                &bounds,
                &[c],
                relu,
            ));
        }
    }

    let spec = arr2(&[[-1.0f32], [2.0]]);
    let bounds = run_backward(&graph, &input, &no_walk, None, Some(&spec), standard_mode())
        .expect("spec seed publishes a bound");
    found.extend(violations("spec_matrix", &bounds, &[-1.0, 2.0], relu));

    let bounds = run_backward(&graph, &input, &no_walk, None, None, standard_mode())
        .expect("identity seed publishes a bound");
    found.extend(violations("identity", &bounds, &[1.0], relu));
    assert_eq!(
        bounds.shape(),
        [1],
        "the identity seed's bound is the raw output box, in the output's shape"
    );

    // Control: the same graph with the full exec_order reaches the input,
    // so the objective bounds come from the ordinary concretization.
    let full = graph.exec_order().expect("dag").to_vec();
    for c in [-1.0f32, 2.0] {
        let bounds = run_backward(&graph, &input, &full, Some(&[c]), None, standard_mode())
            .expect("full walk");
        found.extend(violations("full_walk_control", &bounds, &[c], relu));
    }

    assert!(
        found.is_empty(),
        "the constrained backward published a bound that is not a bound on the seeded \
         objective: {found:#?}"
    );
    println!(
        "[finalize-objective] TEST_COMPLETE \
         name=constrained_backward_unreached_input_maps_the_output_box_through_the_objective"
    );
}

/// RED on the pre-fix tree: `out = a + b` with `a = b = x`, `x ∈ [-1, 1]`
/// (true range `[-2, 2]`). A walk over `[out, b]` reaches the input through
/// `b` while `a`'s block is still pending at a node it never processes;
/// finalization concretized the input block alone and published `[-1, 1]`.
/// The fixed tree refuses (an `Err` is an unresolved domain, which is sound);
/// a published bound must enclose the truth.
#[test]
fn constrained_backward_never_drops_a_pending_contribution() {
    let one = || LinearLayer::new(arr2(&[[1.0f32]]), None).expect("linear");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("a", Layer::Linear(one())));
    graph.add_node(GraphNode::from_input("b", Layer::Linear(one())));
    graph.add_node(GraphNode::binary("out", Layer::Add(AddLayer), "a", "b"));
    graph.set_output("out");
    let input =
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).expect("box");
    let partial = vec!["b".to_string(), "out".to_string()];
    let sum = |x: f64| 2.0 * x;

    let mut found = Vec::new();
    for c in [1.0f32, -1.0] {
        match run_backward(&graph, &input, &partial, Some(&[c]), None, standard_mode()) {
            Ok(bounds) => found.extend(violations("partial_walk", &bounds, &[c], sum)),
            Err(error) => println!("[finalize-objective] partial_walk c={c} refused: {error}"),
        }
    }
    // Control: the full walk carries both paths and encloses the truth.
    let full = graph.exec_order().expect("dag").to_vec();
    for c in [1.0f32, -1.0] {
        let bounds = run_backward(&graph, &input, &full, Some(&[c]), None, standard_mode())
            .expect("full walk");
        found.extend(violations("full_walk_control", &bounds, &[c], sum));
    }
    assert!(
        found.is_empty(),
        "the constrained backward dropped a pending contribution: {found:#?}"
    );
    println!(
        "[finalize-objective] TEST_COMPLETE name=constrained_backward_never_drops_a_pending_contribution"
    );
}
