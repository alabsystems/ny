// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the Graph-MIP LEAF oracle (increment 6):
//   * the premise pin step — clamped boxes + fixed indicator columns match the
//     manually-stabilized expectation;
//   * the budget / eligibility gates;
//   * the end-to-end admission policy on a tiny net (certified-UNSAT ⇒
//     VerifiedAllRows; SAT ⇒ graph-forward-confirmed Violated).

use super::*;

#[cfg(test)]
use super::super::graph_mip::encode_graph;

#[cfg(test)]
use ndarray::{Array1, Array2, ArrayD, IxDyn};
#[cfg(test)]
use ny_propagate::beta_crown::graph_mip_leaf::LeafSplit;
#[cfg(test)]
use ny_propagate::layers::{LinearLayer, ReLULayer};
#[cfg(test)]
use ny_propagate::{GraphNode, NETWORK_INPUT};

#[test]
fn graph_mip_leaf_gate_value_contract() {
    assert!(
        graph_mip_leaf_enabled_from_value(None),
        "unset is default-on"
    );
    assert!(
        graph_mip_leaf_enabled_from_value(Some("1")),
        "explicit 1 remains enabled"
    );
    assert!(
        graph_mip_leaf_enabled_from_value(Some("false")),
        "only exact 0 disables"
    );
    assert!(
        !graph_mip_leaf_enabled_from_value(Some("0")),
        "0 is the kill switch"
    );
}

/// input(2) -> linear (identity 2x2) -> relu, output = relu.
#[cfg(test)]
fn tiny_graph() -> GraphNetwork {
    let w = Array2::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0]).unwrap();
    let b = Array1::from_vec(vec![0.0f32, 0.0]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "lin",
        Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["lin".to_string()],
    ));
    graph.set_output("relu");
    graph
}

#[cfg(test)]
fn boxed(lo: Vec<f32>, hi: Vec<f32>) -> Arc<BoundedTensor> {
    let n = lo.len();
    Arc::new(
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lo).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), hi).unwrap(),
        )
        .unwrap(),
    )
}

#[cfg(test)]
fn tiny_node_bounds() -> HashMap<String, Arc<BoundedTensor>> {
    let mut m = HashMap::new();
    // Both pre-activation neurons unstable in [-1, 1].
    m.insert("lin".to_string(), boxed(vec![-1.0, -1.0], vec![1.0, 1.0]));
    m
}

#[cfg(test)]
fn split(neuron: usize, active: bool) -> LeafSplit {
    LeafSplit {
        relu_node: "relu".to_string(),
        neuron_idx: neuron,
        is_active: active,
    }
}

/// The clamp step reproduces emit_hard_six `clamp()`: active raises the
/// pre-activation lower to 0, inactive lowers the upper to 0; other neurons
/// untouched; an infeasible clamp fails closed.
#[test]
fn clamp_step_matches_manual_stabilization() {
    let graph = tiny_graph();
    let nb = tiny_node_bounds();

    let flat = clamped_flat_bounds(&graph, &nb, &[split(0, true)]).expect("clamp");
    let lin = &flat["lin"];
    assert_eq!(lin[0].lower(), 0.0, "active premise raises lower to 0");
    assert_eq!(lin[0].upper(), 1.0, "active premise keeps upper");
    assert_eq!(lin[1].lower(), -1.0, "other neuron untouched");

    let flat = clamped_flat_bounds(&graph, &nb, &[split(1, false)]).expect("clamp");
    let lin = &flat["lin"];
    assert_eq!(lin[1].upper(), 0.0, "inactive premise lowers upper to 0");
    assert_eq!(lin[1].lower(), -1.0, "inactive premise keeps lower");

    // Infeasible premise ∧ box (active on a strictly-negative box) fails closed.
    let mut nb_neg = HashMap::new();
    nb_neg.insert("lin".to_string(), boxed(vec![-2.0, -1.0], vec![-1.0, 1.0]));
    assert!(
        clamped_flat_bounds(&graph, &nb_neg, &[split(0, true)]).is_none(),
        "infeasible clamp must fail closed"
    );

    // Unknown ReLU node fails closed.
    let bad = LeafSplit {
        relu_node: "nope".to_string(),
        neuron_idx: 0,
        is_active: true,
    };
    assert!(clamped_flat_bounds(&graph, &nb, &[bad]).is_none());
}

/// The free-binary count is taken on the CLAMPED (un-inflated) bounds, so
/// premise-pinned neurons do not count against the budget.
#[test]
fn free_binary_count_excludes_pinned_neurons() {
    let graph = tiny_graph();
    let nb = tiny_node_bounds();

    let flat = clamped_flat_bounds(&graph, &nb, &[]).expect("clamp");
    assert_eq!(
        free_binary_count(&graph, &flat),
        Some(2),
        "both unstable at root"
    );

    let flat = clamped_flat_bounds(&graph, &nb, &[split(0, true)]).expect("clamp");
    assert_eq!(
        free_binary_count(&graph, &flat),
        Some(1),
        "an active premise pins one neuron (l = 0 is not < 0)"
    );

    let flat = clamped_flat_bounds(&graph, &nb, &[split(0, true), split(1, false)]).expect("clamp");
    assert_eq!(
        free_binary_count(&graph, &flat),
        Some(0),
        "both premises pin"
    );
}

/// The fix step pins exactly the premise binaries' indicator columns
/// (`[1,1]` active / `[0,0]` inactive) and leaves the other binaries free —
/// the manually-stabilized expectation for the encoded problem.
#[test]
fn fix_step_pins_indicator_columns() {
    let graph = tiny_graph();
    let nb = tiny_node_bounds();
    // Clamp neuron 0 active; both neurons remain (marginally) unstable after
    // the encoder's DELTA inflation, so BOTH binaries exist pre-fix.
    let flat = clamped_flat_bounds(&graph, &nb, &[split(0, true)]).expect("clamp");
    let input_bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
    let mut enc = encode_graph(&graph, &input_bounds, &flat).expect("encode");
    assert_eq!(
        enc.binary_vars.len(),
        2,
        "DELTA keeps the clamped neuron marginal"
    );
    assert_eq!(
        enc.binary_keys,
        vec![("relu".to_string(), 0), ("relu".to_string(), 1)],
        "binary identity keys"
    );

    let pinned = fix_split_binaries(&graph, &mut enc, &[split(0, true)]);
    assert_eq!(pinned, 1, "one premise pinned");
    let z0 = enc.binary_vars[0];
    let z1 = enc.binary_vars[1];
    let c0 = &enc.problem.cols()[z0.0];
    let c1 = &enc.problem.cols()[z1.0];
    assert_eq!((c0.lb, c0.ub), (1.0, 1.0), "active premise fixes z = 1");
    assert_eq!((c1.lb, c1.ub), (0.0, 1.0), "other binary stays free");

    // Inactive premise on the other neuron fixes z = 0.
    let pinned = fix_split_binaries(&graph, &mut enc, &[split(1, false)]);
    assert_eq!(pinned, 1);
    let c1 = &enc.problem.cols()[z1.0];
    assert_eq!((c1.lb, c1.ub), (0.0, 0.0), "inactive premise fixes z = 0");

    // A premise with no matching binary (unknown neuron) is skipped, not an error.
    let pinned = fix_split_binaries(&graph, &mut enc, &[split(7, true)]);
    assert_eq!(pinned, 0, "no binary for that neuron: skipped");
}

/// Budget gates: the cumulative cap declines once spent exceeds
/// `first_remaining × frac`, and a too-small remaining slice declines.
#[test]
fn budget_gates_decline() {
    let solver = GraphMipLeafSolver::new(MipBackend::Ay);
    // No deadline: infinite remaining, default slice admitted.
    let slice = solver.admit_slice(None).expect("no-deadline slice");
    assert!(slice > 0.0);

    // Exhaust the cumulative cap: first_remaining = 100 s, spend 60 s (> 50%).
    {
        let mut b = solver.budget.lock().unwrap();
        b.first_remaining = Some(100.0);
        b.spent = 60.0;
    }
    assert!(
        solver.admit_slice(None).is_none(),
        "cumulative cap must decline further leaves"
    );

    // A nearly-expired deadline (remaining/4 < 1 s) declines.
    let solver = GraphMipLeafSolver::new(MipBackend::Ay);
    let deadline = Instant::now() + std::time::Duration::from_secs(2);
    assert!(
        solver.admit_slice(Some(deadline)).is_none(),
        "sub-second slice must decline"
    );
}

/// Depth / eligibility gates on `solve_leaf` itself.
#[test]
fn solve_leaf_gates_shallow_and_empty() {
    let graph = tiny_graph();
    let nb = tiny_node_bounds();
    let input = boxed(vec![-1.0, -1.0], vec![1.0, 1.0]);
    let solver = GraphMipLeafSolver::new(MipBackend::Ay);

    // Depth below the default min (4) declines.
    let req = GraphMipLeafRequest {
        graph: &graph,
        input_bounds: &input,
        node_bounds: &nb,
        splits: vec![],
        rows: vec![(vec![1.0, 0.0], -0.5)],
        depth: 1,
        deadline: None,
    };
    assert!(matches!(
        solver.solve_leaf(&req),
        GraphMipLeafVerdict::Undecided
    ));

    // No undecided rows declines.
    let req = GraphMipLeafRequest {
        graph: &graph,
        input_bounds: &input,
        node_bounds: &nb,
        splits: vec![],
        rows: vec![],
        depth: 8,
        deadline: None,
    };
    assert!(matches!(
        solver.solve_leaf(&req),
        GraphMipLeafVerdict::Undecided
    ));
}

/// End-to-end admission on the tiny net: `relu(x) <= -0.5` is infeasible
/// (ReLU outputs are nonnegative) ⇒ every row certified-UNSAT ⇒
/// `VerifiedAllRows`. This is the readiness gate for the ay ladder: if ay
/// stops certifying these root-level infeasibilities, this test fails.
#[test]
fn solve_leaf_admits_certified_unsat() {
    let graph = tiny_graph();
    let nb = tiny_node_bounds();
    let input = boxed(vec![-1.0, -1.0], vec![1.0, 1.0]);
    let solver = GraphMipLeafSolver::new(MipBackend::Ay);
    let req = GraphMipLeafRequest {
        graph: &graph,
        input_bounds: &input,
        node_bounds: &nb,
        // Row: y_0 <= -0.5 — impossible for a ReLU output.
        rows: vec![(vec![1.0, 0.0], -0.5)],
        splits: vec![split(1, false)],
        depth: 8,
        deadline: None,
    };
    match solver.solve_leaf(&req) {
        GraphMipLeafVerdict::VerifiedAllRows => {}
        other => panic!("expected certified VerifiedAllRows, got {other:?}"),
    }
}

/// End-to-end SAT arm: `relu(x) <= 0.5` is satisfiable (e.g. x = 0) and the
/// witness must be confirmed by the graph forward ⇒ `Violated` with an in-box
/// witness whose forward margin honors the row.
#[test]
fn solve_leaf_confirms_sat_witness_via_graph_forward() {
    let graph = tiny_graph();
    let nb = tiny_node_bounds();
    let input = boxed(vec![-1.0, -1.0], vec![1.0, 1.0]);
    let solver = GraphMipLeafSolver::new(MipBackend::Ay);
    let req = GraphMipLeafRequest {
        graph: &graph,
        input_bounds: &input,
        node_bounds: &nb,
        rows: vec![(vec![1.0, 0.0], 0.5)],
        splits: vec![],
        depth: 8,
        deadline: None,
    };
    match solver.solve_leaf(&req) {
        GraphMipLeafVerdict::Violated { witness, output } => {
            assert_eq!(witness.len(), 2, "witness is the flattened input");
            assert!(
                witness.iter().all(|w| (-1.0..=1.0).contains(w)),
                "witness clamped into the domain box"
            );
            // Confirmed margin: relu(x)_0 <= 0.5 at the revalidated output.
            assert!(
                output[0] <= 0.5 + 1e-6,
                "graph-forward margin confirms the row"
            );
        }
        other => panic!("expected confirmed Violated, got {other:?}"),
    }
}

/// Defect-3 containment: a PANIC inside the leaf solve is caught at the
/// oracle boundary and degrades to `Undecided` — it can never escape into the
/// BaB loop (the measured prop1498 run-killing mode).
#[test]
fn oracle_boundary_contains_panics() {
    let verdict = contain_leaf_panics(|| panic!("simulated solver worker panic"));
    assert!(matches!(verdict, GraphMipLeafVerdict::Undecided));
    // String payloads too.
    let verdict = contain_leaf_panics(|| panic!("{}", String::from("heap payload")));
    assert!(matches!(verdict, GraphMipLeafVerdict::Undecided));
    // And a non-panicking closure passes its verdict through.
    let verdict = contain_leaf_panics(|| GraphMipLeafVerdict::VerifiedAllRows);
    assert!(matches!(verdict, GraphMipLeafVerdict::VerifiedAllRows));
}

/// Defect-3 containment: an internal Err (here: a malformed row whose
/// coefficient length cannot be emitted) yields `Undecided`, never an Err out
/// of the infallible oracle.
#[test]
fn solver_error_degrades_to_undecided() {
    let graph = tiny_graph();
    let nb = tiny_node_bounds();
    let input = boxed(vec![-1.0, -1.0], vec![1.0, 1.0]);
    let solver = GraphMipLeafSolver::new(MipBackend::Ay);
    let req = GraphMipLeafRequest {
        graph: &graph,
        input_bounds: &input,
        node_bounds: &nb,
        // 3 coefficients vs 2 output columns: row emission fails internally.
        rows: vec![(vec![1.0, 0.0, 5.0], -0.5)],
        splits: vec![],
        depth: 8,
        deadline: None,
    };
    assert!(matches!(
        solver.solve_leaf(&req),
        GraphMipLeafVerdict::Undecided
    ));
}

/// Defect-1 leaf-scale gates: a leaf whose free-binary count exceeds the
/// LEAF cap (its own, far below the whole-net cap) declines without encoding;
/// same for the nnz estimate.
#[test]
fn leaf_scale_gates_decline_oversize_leaves() {
    let graph = tiny_graph();
    let nb = tiny_node_bounds();

    // Exact estimator parity on the tiny net: 2x2 linear
    // (out×(in+1)=6) + two unstable ReLUs × seven row coefficients = 20.
    let flat = clamped_flat_bounds(&graph, &nb, &[]).expect("clamp");
    let nnz = estimate_encode_nnz(&graph, &flat).expect("estimate");
    assert_eq!(nnz, 20, "estimator must count every big-M coefficient");

    // Free-binary gate: solve_leaf_gated declines when free > leaf cap.
    // (Direct gate check via the counting helpers — env-free.)
    assert_eq!(free_binary_count(&graph, &flat), Some(2));
    assert!(
        2 <= leaf_max_binaries(),
        "tiny net fits the default leaf cap"
    );
    // The default caps themselves: leaf cap is w5-scale, far below whole-net.
    assert_eq!(leaf_max_binaries(), 96, "default leaf binary cap");
    assert_eq!(leaf_max_nnz(), 5_000_000, "default leaf nnz cap");
}

/// Defect-3: after a confirmed SAT the oracle latches OFF — subsequent
/// consults decline immediately (no repeated solver spend), and the verdict
/// that latched it was `Violated` (advisory; the loop requeues).
#[test]
fn confirmed_sat_latches_oracle_off() {
    let graph = tiny_graph();
    let nb = tiny_node_bounds();
    let input = boxed(vec![-1.0, -1.0], vec![1.0, 1.0]);
    let solver = GraphMipLeafSolver::new(MipBackend::Ay);
    let req = GraphMipLeafRequest {
        graph: &graph,
        input_bounds: &input,
        node_bounds: &nb,
        rows: vec![(vec![1.0, 0.0], 0.5)], // satisfiable: relu(x) <= 0.5
        splits: vec![],
        depth: 8,
        deadline: None,
    };
    assert!(matches!(
        solver.solve_leaf(&req),
        GraphMipLeafVerdict::Violated { .. }
    ));
    // Second consult: latched off, declines without solving.
    assert!(matches!(
        solver.solve_leaf(&req),
        GraphMipLeafVerdict::Undecided
    ));
}

// ═════════════════════════════════════════════════════════════════════════════
// #relational-bab: edge-domain escalation with the REAL exact solver.
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod relational_edge_milp {
    use ndarray::{arr1, arr2};
    use ny_propagate::{
        BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic,
        GraphNetwork, GraphNode, Layer,
    };
    use ny_tensor::BoundedTensor;
    use std::sync::Arc;
    use std::time::Duration;

    /// Two near-identical 2-D towers joined by Sub — the miniature of the
    /// relational difference net whose plain-CROWN slack floors above the
    /// true deviation (the parity-probe fixture).
    /// Counting delegate around the REAL solver: proves the oracle is
    /// load-bearing (consult + verdict distribution visible to asserts).
    struct CountingOracle {
        inner: super::super::GraphMipLeafSolver,
        consults: std::sync::atomic::AtomicUsize,
        verified: std::sync::atomic::AtomicUsize,
    }

    impl ny_propagate::beta_crown::graph_mip_leaf::GraphMipLeafOracle for CountingOracle {
        fn solve_leaf(
            &self,
            req: &ny_propagate::beta_crown::graph_mip_leaf::GraphMipLeafRequest<'_>,
        ) -> ny_propagate::beta_crown::graph_mip_leaf::GraphMipLeafVerdict {
            use std::sync::atomic::Ordering;
            self.consults.fetch_add(1, Ordering::SeqCst);
            let verdict = self.inner.solve_leaf(req);
            if matches!(
                verdict,
                ny_propagate::beta_crown::graph_mip_leaf::GraphMipLeafVerdict::VerifiedAllRows
            ) {
                self.verified.fetch_add(1, Ordering::SeqCst);
            }
            verdict
        }
    }

    pub(super) fn sub_towers_graph() -> GraphNetwork {
        use ny_propagate::layers::{LinearLayer, ReLULayer, SubLayer};
        let wa1 = arr2(&[[1.0_f32, -0.7], [0.6, 1.1], [-0.9, 0.8]]);
        let wb1 = wa1.mapv(|v| v * 1.01);
        let wa2 = arr2(&[[0.8_f32, -1.2, 0.5]]);
        let wb2 = wa2.mapv(|v| v * 0.99);
        let ba1 = arr1(&[0.05_f32, -0.1, 0.2]);
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "a_l1",
            Layer::Linear(LinearLayer::new(wa1, Some(ba1.clone())).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "a_r1",
            Layer::ReLU(ReLULayer),
            vec!["a_l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "a_l2",
            Layer::Linear(LinearLayer::new(wa2, None).unwrap()),
            vec!["a_r1".to_string()],
        ));
        graph.add_node(GraphNode::from_input(
            "b_l1",
            Layer::Linear(LinearLayer::new(wb1, Some(ba1)).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "b_r1",
            Layer::ReLU(ReLULayer),
            vec!["b_l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "b_l2",
            Layer::Linear(LinearLayer::new(wb2, None).unwrap()),
            vec!["b_r1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "diff",
            Layer::Sub(SubLayer),
            vec!["a_l2".to_string(), "b_l2".to_string()],
        ));
        graph.set_output("diff");
        graph
    }

    /// End-to-end with the REAL certified solver: pick the band threshold at
    /// runtime BETWEEN the (dense-grid) sampled true minimum and the CROWN
    /// root bound, so plain CROWN fails the row while the exact MILP decides
    /// it. Asserts the escalated lane verifies with FAR fewer domains than
    /// the budget — the edge domain was DECIDED, not split forever.
    #[test]
    fn real_oracle_decides_edge_domain_end_to_end() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("debug")
            .try_init();
        let graph = sub_towers_graph();
        let input = BoundedTensor::new(
            arr1(&[-0.5_f32, -0.5]).into_dyn(),
            arr1(&[0.5_f32, 0.5]).into_dyn(),
        )
        .unwrap();

        // Dense-grid UNDER-estimate of the true minimum of h(x) (exact point
        // forwards; 101x101 fixed grid — deterministic).
        let mut sampled_min = f32::INFINITY;
        for i in 0..=200 {
            for j in 0..=200 {
                let x0 = -0.5 + i as f32 / 200.0;
                let x1 = -0.5 + j as f32 / 200.0;
                let pt = BoundedTensor::new(arr1(&[x0, x1]).into_dyn(), arr1(&[x0, x1]).into_dyn())
                    .unwrap();
                let out = graph.propagate_ibp(&pt).unwrap();
                sampled_min = sampled_min.min(out.lower()[[0]]);
            }
        }
        // CROWN root lower bound for the +e row.
        let spec = arr2(&[[1.0_f32]]);
        let nb = graph
            .collect_crown_ibp_bounds_dag_with_engine(&input, None)
            .unwrap();
        let (crown, _) = graph
            .propagate_crown_with_specs_and_node_bounds_and_linear(&input, &spec, None, &nb)
            .unwrap();
        let crown_lower = crown.flatten().lower()[[0]];
        assert!(
            sampled_min > crown_lower + 1e-3,
            "fixture must have relaxation slack (sampled_min {sampled_min} vs crown {crown_lower})"
        );
        // Threshold 95% of the way from the CROWN bound to the sampled min:
        // still below the true min (the 101x101 grid under-estimates it by
        // far less than the remaining 5% of the slack range, so the exact
        // MILP proves it), but deep inside the relaxation floor — plain CROWN
        // needs boxes far beyond the 60-domain budget below.
        let threshold = crown_lower + 0.95 * (sampled_min - crown_lower);

        let run = |edge: bool| {
            let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::InputSplit,
                use_alpha_crown: false,
                enable_relaxed_clip: false,
                enable_pgd_attack: false,
                reorder_bab: true,
                batch_size: 8,
                max_domains: 48,
                max_depth: 100,
                timeout: Duration::from_mins(1),
                beta_iterations: 0,
                input_split_edge_milp: edge,
                input_split_edge_milp_gap: 1.0,
                input_split_edge_milp_depth: 1,
                ..BetaCrownConfig::default()
            });
            let oracle = Arc::new(CountingOracle {
                inner: super::super::GraphMipLeafSolver::new(ny_mip::MipBackend::Ay),
                consults: std::sync::atomic::AtomicUsize::new(0),
                verified: std::sync::atomic::AtomicUsize::new(0),
            });
            let verifier = if edge {
                verifier.with_graph_mip_leaf_oracle(oracle.clone())
            } else {
                verifier
            };
            let result = verifier
                .verify_graph_input_split_multi_clause_disjunctive(
                    &graph,
                    &input,
                    &[vec![1.0_f32]],
                    &[threshold],
                    &[1usize],
                    None,
                    None,
                )
                .expect("lane completes");
            let consults = oracle.consults.load(std::sync::atomic::Ordering::SeqCst);
            let oracle_verified = oracle.verified.load(std::sync::atomic::Ordering::SeqCst);
            eprintln!(
                "[edge-milp e2e] edge={edge}: {:?} domains={} consults={consults} oracle_verified={oracle_verified}",
                result.result, result.domains_explored
            );
            (result, consults, oracle_verified)
        };

        // BASELINE discriminator: without the oracle the 48-domain budget is
        // far too small for plain CROWN to reach floor-clearing box widths —
        // if this ever verifies, the fixture stopped exercising the
        // escalation and must be re-tightened.
        let (baseline, baseline_consults, _) = run(false);
        assert_eq!(baseline_consults, 0, "no oracle attached => no consults");
        assert!(
            !matches!(baseline.result, BabVerificationStatus::Verified),
            "baseline (no oracle) unexpectedly verified in {} domains — fixture too easy",
            baseline.domains_explored
        );

        // NY_GRAPH_MIP_LEAF_MIN_DEPTH default 4 gates the solver itself; the
        // lane gate is 1 — domains escalate from depth 4 on.
        let (escalated, consults, oracle_verified) = run(true);
        assert!(consults > 0, "the real oracle must have been consulted");
        assert!(
            matches!(escalated.result, BabVerificationStatus::Verified),
            "real exact solver must decide the edge domains (got {:?} after {} domains, {consults} consults, {oracle_verified} oracle-verified)",
            escalated.result,
            escalated.domains_explored
        );
        assert!(
            oracle_verified > 0,
            "at least one edge domain must be DECIDED by the exact solver"
        );
        eprintln!(
            "[edge-milp e2e] threshold {threshold:.5} (crown {crown_lower:.5}, sampled_min {sampled_min:.5})"
        );
    }
}

/// DIAGNOSTIC (#relational-bab live 1µs decline): replicate the edge
/// escalation's request on the REAL instance_0 iso difference net and a deep
/// center sub-box; print the verdict + timing + the free-binary count under
/// BOTH map conventions.
pub(crate) mod live_decline_probe {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use ny_propagate::beta_crown::graph_mip_leaf::{
        GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict,
    };
    use ny_tensor::BoundedTensor;

    pub(crate) fn probe_real_diffnet_edge_request(base: &std::path::Path) {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("debug")
            .try_init();
        let f = base.join("onnx/original/ACASXU_run2a_2_4_batch_2000.onnx");
        let g = base.join("onnx/perturbed/ACASXU_run2a_2_4_batch_2000_perturbed_0.onnx");
        let graph_f = crate::commands::vnncomp::load_graph_network(&f).expect("load f");
        let graph_g = crate::commands::vnncomp::load_graph_network(&g).expect("load g");
        let diff = ny_propagate::build_difference_network(&graph_f, &graph_g).expect("diff");
        let spec = ny_onnx::vnnlib::load_vnnlib(&base.join("vnnlib/instance_0.vnnlib")).unwrap();
        let dual = spec.dual_network.expect("dual");
        let bounds = crate::commands::vnncomp::bounds_from_f64(&dual.f_input_bounds).unwrap();
        // Deep center sub-box (the edge regime).
        let deep: Vec<f32> = Vec::new();
        let _ = deep;
        let lo: Vec<f32> = bounds
            .iter()
            .map(|b| f32::midpoint(b.lower(), b.upper()) - (b.upper() - b.lower()) / 128.0)
            .collect();
        let hi: Vec<f32> = bounds
            .iter()
            .map(|b| f32::midpoint(b.lower(), b.upper()) + (b.upper() - b.lower()) / 128.0)
            .collect();
        let _ = (lo, hi);
        for frac in [64.0f32, 512.0, 4096.0, 65536.0] {
            let lo: Vec<f32> = bounds
                .iter()
                .map(|b| {
                    f32::midpoint(b.lower(), b.upper()) - (b.upper() - b.lower()) / (2.0 * frac)
                })
                .collect();
            let hi: Vec<f32> = bounds
                .iter()
                .map(|b| {
                    f32::midpoint(b.lower(), b.upper()) + (b.upper() - b.lower()) / (2.0 * frac)
                })
                .collect();
            let input = BoundedTensor::new(
                ndarray::Array1::from(lo).into_dyn(),
                ndarray::Array1::from(hi).into_dyn(),
            )
            .unwrap();
            let collected = diff.collect_node_bounds(&input).expect("collect");
            let node_bounds: HashMap<String, Arc<BoundedTensor>> = collected
                .into_iter()
                .map(|(k, v)| (k, Arc::new(v)))
                .collect();
            let mut unstable = 0usize;
            for name in diff.exec_order().unwrap() {
                let node = diff.node(name).unwrap();
                if !matches!(node.layer(), ny_propagate::Layer::ReLU(_)) {
                    continue;
                }
                if let Some(bt) = node.inputs().first().and_then(|p| node_bounds.get(p)) {
                    let f = bt.flatten();
                    unstable += (0..f.len())
                        .filter(|&i| f.lower()[[i]] < 0.0 && f.upper()[[i]] > 0.0)
                        .count();
                }
            }
            let solver = super::GraphMipLeafSolver::new(ny_mip::MipBackend::Ay);
            let req = GraphMipLeafRequest {
                graph: &diff,
                input_bounds: &input,
                node_bounds: &node_bounds,
                splits: Vec::new(),
                rows: vec![(vec![-1.0, 0.0, 0.0, 0.0, 0.0], -0.05_f32)],
                depth: 30,
                deadline: Some(Instant::now() + std::time::Duration::from_secs(30)),
            };
            let start = Instant::now();
            let verdict = solver.solve_leaf(&req);
            eprintln!(
                "[probe] 1/{frac} box: unstable={unstable} verdict={:?} wall={:.3}s",
                match verdict {
                    GraphMipLeafVerdict::VerifiedAllRows => "VerifiedAllRows",
                    GraphMipLeafVerdict::Violated { .. } => "Violated",
                    GraphMipLeafVerdict::Undecided => "Undecided",
                },
                start.elapsed().as_secs_f64()
            );
        }
    }
}

/// #relational-bab: ZERO-BINARY edge domains (every neuron stable on the box)
/// are pure LPs and MUST decide through the certified enumeration lane
/// (`2^0 = 1` Farkas-certified leaf). This is the oracle-level pin for the
/// live "free_binaries=0" class.
#[cfg(test)]
mod zero_binary_edge {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use ndarray::arr1;
    use ny_propagate::beta_crown::graph_mip_leaf::{
        GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict,
    };
    use ny_tensor::BoundedTensor;

    #[test]
    fn zero_binary_edge_domain_decides_certified() {
        let graph = super::relational_edge_milp::sub_towers_graph();
        // Box near (0.4, 0.4): every pre-activation strictly positive in both
        // towers => ALL relus stable => zero free binaries (a pure LP).
        let input = BoundedTensor::new(
            arr1(&[0.38_f32, 0.38]).into_dyn(),
            arr1(&[0.42_f32, 0.42]).into_dyn(),
        )
        .unwrap();
        let collected = graph.collect_node_bounds(&input).expect("collect");
        let node_bounds: HashMap<String, Arc<BoundedTensor>> = collected
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();
        // Sanity: genuinely zero unstable neurons on this box.
        let mut unstable = 0usize;
        for name in graph.exec_order().unwrap() {
            let node = graph.node(name).unwrap();
            if !matches!(node.layer(), ny_propagate::Layer::ReLU(_)) {
                continue;
            }
            let bt = node
                .inputs()
                .first()
                .and_then(|p| node_bounds.get(p))
                .expect("pre bounds");
            let f = bt.flatten();
            unstable += (0..f.len())
                .filter(|&i| f.lower()[[i]] < 0.0 && f.upper()[[i]] > 0.0)
                .count();
        }
        assert_eq!(
            unstable, 0,
            "fixture must be a zero-binary (all-stable) box"
        );

        // Row `+h > true_min - 1` is TRUE with a huge margin: the decision LP
        // (h <= threshold) is infeasible and must come back CERTIFIED.
        let out = graph
            .propagate_ibp(
                &BoundedTensor::new(
                    arr1(&[0.4_f32, 0.4]).into_dyn(),
                    arr1(&[0.4_f32, 0.4]).into_dyn(),
                )
                .unwrap(),
            )
            .unwrap();
        let center = out.lower()[[0]];
        let threshold = center - 1.0;

        let solver = super::GraphMipLeafSolver::new(ny_mip::MipBackend::Ay);
        let req = GraphMipLeafRequest {
            graph: &graph,
            input_bounds: &input,
            node_bounds: &node_bounds,
            splits: Vec::new(),
            rows: vec![(vec![1.0_f32], threshold)],
            depth: 30,
            deadline: Some(Instant::now() + std::time::Duration::from_secs(30)),
        };
        let verdict = solver.solve_leaf(&req);
        assert!(
            matches!(verdict, GraphMipLeafVerdict::VerifiedAllRows),
            "zero-binary edge domain must decide via the certified LP lane"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// #rel-whole-mip: WHOLE-NET certified-UNSAT MILP on a DIFFERENCE network.
// ═════════════════════════════════════════════════════════════════════════════
pub(crate) mod whole_net_diff_mip {
    #[cfg(test)]
    use ndarray::{arr1, arr2};
    use ny_core::Bound;
    #[cfg(test)]
    use ny_propagate::layers::{LinearLayer, ReLULayer};
    use ny_propagate::{build_difference_network, GraphNetwork};
    #[cfg(test)]
    use ny_propagate::{GraphNode, Layer};

    /// One isomorphic-style tower: input(2) → Linear(2→3) → ReLU → Linear(3→1),
    /// with an optional per-layer weight scale to model the g perturbation.
    #[cfg(test)]
    fn tower(scale: f32) -> GraphNetwork {
        let w1 = arr2(&[[1.0_f32, -0.6], [0.5, 0.8], [-0.9, 0.4]]).mapv(|v| v * scale);
        let b1 = arr1(&[0.1_f32, -0.2, 0.05]);
        let w2 = arr2(&[[0.7_f32, -0.5, 0.9]]).mapv(|v| v * scale);
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "l2",
            Layer::Linear(LinearLayer::new(w2, None).unwrap()),
            vec!["r1".to_string()],
        ));
        g.set_output("l2");
        g
    }

    fn band_rows(n_out: usize, eps: f32) -> Vec<(Vec<f32>, f32)> {
        // |h_i| ≤ eps ⇒ decision rows: refute h_i < -eps (row +e_i, thr -eps)
        // AND refute h_i > eps (row -e_i, thr -eps).
        let mut rows = Vec::new();
        for i in 0..n_out {
            let mut p = vec![0.0f32; n_out];
            p[i] = 1.0;
            rows.push((p.clone(), -eps));
            p[i] = -1.0;
            rows.push((p, -eps));
        }
        rows
    }

    /// The whole-net certified MILP verifies a difference network whose band
    /// genuinely holds — the token-authorized finisher that closes iso
    /// holdouts the instant ay can solve full-rung.
    #[test]
    fn whole_net_certifies_true_band() {
        let f = tower(1.0);
        let g = tower(1.02); // 2% perturbation ⇒ small, genuine difference
        let diff = build_difference_network(&f, &g).expect("diff net");
        let input_bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
        // ε̂ comfortably above the true |h| (a ~2% perturbation of ~O(1)
        // activations) but the band is a genuine over-approx obligation.
        let rows = band_rows(1, 1.0);
        let ok = super::super::whole_net_certified_band_unsat(
            &diff,
            &input_bounds,
            &rows,
            30.0,
            Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
        );
        assert!(
            ok,
            "whole-net certified MILP must verify a genuinely-holding band"
        );
    }

    /// A band that is genuinely VIOLATED must NOT be certified (0-wrong):
    /// with ε̂ far below the true deviation the decision MIP is FEASIBLE, so no
    /// certified-UNSAT — the finisher returns false and the caller keeps its
    /// inconclusive verdict.
    #[test]
    fn whole_net_rejects_violated_band() {
        let f = tower(1.0);
        let g = tower(1.5); // large perturbation ⇒ |h| is large
        let diff = build_difference_network(&f, &g).expect("diff net");
        let input_bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
        let rows = band_rows(1, 1e-4); // ε̂ far below true |h| ⇒ violated
        let ok = super::super::whole_net_certified_band_unsat(
            &diff,
            &input_bounds,
            &rows,
            30.0,
            Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
        );
        assert!(!ok, "a violated band must never be certified (0-wrong)");
    }

    /// PROPERTY-CONDITIONED OBBT (`NY_REL_WHOLE_MIP_OBBT_COND=1`) must still
    /// verify a genuinely-holding band: the violation-conditioned boxes only
    /// SHRINK the sound α-CROWN big-M, so the certified-UNSAT verdict is
    /// preserved (and reached from a tighter model).
    #[test]
    fn whole_net_conditioned_certifies_true_band() {
        let _env_lock = ny_test_utils::env::lock_env();
        let _g = ny_test_utils::env::ScopedEnvVar::set("NY_REL_WHOLE_MIP_OBBT_COND", "1");
        let f = tower(1.0);
        let g = tower(1.02);
        let diff = build_difference_network(&f, &g).expect("diff net");
        let input_bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
        let rows = band_rows(1, 1.0);
        let ok = super::super::whole_net_certified_band_unsat(
            &diff,
            &input_bounds,
            &rows,
            30.0,
            Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
        );
        assert!(
            ok,
            "conditioned whole-net MILP must still verify a genuinely-holding band"
        );
    }

    /// SOUNDNESS MOAT for property-conditioned OBBT: a genuinely VIOLATED band
    /// must NEVER be certified. The violation-conditioned bounds are a valid
    /// OUTER bound over the violation region (the OBBT LP is the triangle
    /// relaxation ⊇ the violation region), so they can never cut the violating
    /// input off — the decision MIP stays FEASIBLE and the finisher returns
    /// false. A false UNSAT here would certify a false property.
    #[test]
    fn whole_net_conditioned_rejects_violated_band() {
        let _env_lock = ny_test_utils::env::lock_env();
        let _g = ny_test_utils::env::ScopedEnvVar::set("NY_REL_WHOLE_MIP_OBBT_COND", "1");
        let f = tower(1.0);
        let g = tower(1.5); // large perturbation ⇒ |h| large ⇒ band violated
        let diff = build_difference_network(&f, &g).expect("diff net");
        let input_bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
        let rows = band_rows(1, 1e-4);
        let ok = super::super::whole_net_certified_band_unsat(
            &diff,
            &input_bounds,
            &rows,
            30.0,
            Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
        );
        assert!(
            !ok,
            "conditioned OBBT must NEVER certify a violated band (over-tight bounds \
             would be a false HOLDS)"
        );
    }

    /// Direct callers without an outer deadline must fail open instead of
    /// panicking while converting or adding adversarial floating-point slices.
    #[test]
    fn whole_net_rejects_invalid_or_overflowing_slices_without_panic() {
        let diff = build_difference_network(&tower(1.0), &tower(1.02)).expect("diff net");
        let input_bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
        let rows = band_rows(1, 1.0);
        for slice in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1e300] {
            let start = std::time::Instant::now();
            assert!(
                !super::super::whole_net_certified_band_unsat(
                    &diff,
                    &input_bounds,
                    &rows,
                    slice,
                    None,
                ),
                "invalid/unrepresentable slice {slice} must fail open"
            );
            assert!(
                start.elapsed() < std::time::Duration::from_secs(1),
                "invalid/unrepresentable slice {slice} must fail before preprocessing"
            );
        }
    }

    /// #rel-whole-mip CRUX MEASUREMENT (a probe, not a pass/fail assertion).
    ///
    /// Build ONE real iso instance's whole-net difference-network decision MILP
    /// and hand each band row to ay (bb2b6088, in-process). Compares node-bound
    /// tightness across IBP / CROWN-IBP(default) / CROWN-full(width-threshold=0)
    /// and solves the tightest, printing size + big-M + verdicts. Bounded so it
    /// can't hang.
    ///
    /// MEASURED (instance_6, ε=0.05, 2026-07-17) — the tight-bounds hypothesis
    /// is REFUTED: bound-tightening does NOT reduce the whole-net root binaries.
    ///   IBP                : binaries=562  out.max|h|=87348
    ///   CROWN-IBP(default) : binaries=546  out.max|h|=14479   (current finisher)
    ///   CROWN-full(wthr=0) : binaries=546  out.max|h|=14479   (IDENTICAL to default)
    /// maxBigM = 3.3e6 across ALL three (unchanged by CROWN). α-CROWN is code-
    /// confirmed to leave intermediate boxes at the reference map (best_bounds is
    /// init'd to reference_bounds; the α loop optimizes the OUTPUT node only —
    /// alpha.rs), so α ≡ CROWN-IBP = 546. CONCLUSION: the ~546 unstable neurons
    /// are GENUINE — the two ACAS towers' individual neurons truly span both
    /// signs over the undivided 5-D box; this is not a looseness artifact and no
    /// whole-box bound method reaches ay's w5 rung (53-83). The tractable
    /// ~80-binary MILP exists ONLY per-SUBDOMAIN (after input splitting) — the
    /// per-domain leaf oracle path (NY_REL_EDGE_MILP), not the whole-net root.
    ///
    /// Run through `ny vnncomp-research graph-mip whole-net-bb2b6088
    /// --bench-dir <isomorphic-acas-2.0>`.
    ///
    /// Overrides (env): NY_ISO_F_ONNX, NY_ISO_G_ONNX, NY_ISO_VNNLIB,
    ///   NY_ISO_ROW_SECS (default 60), NY_ISO_TOTAL_SECS (480).
    /// MEASUREMENT — coupled-OBBT big-M shrink on the real iso diff nets.
    ///
    /// For each instance, replays the production finisher's tightening pipeline
    /// (CROWN-IBP node boxes → α-CROWN ∩ → coupled OBBT) and reports the
    /// pre-activation box-width distribution at each stage plus the OBBT cost.
    /// This isolates the OBBT lever's effect on the difference-net big-M.
    ///
    /// Run through `ny vnncomp-research graph-mip obbt-box-width
    /// --bench-dir <isomorphic-acas-2.0>`.
    ///
    /// Overrides (env): NY_ISO_INSTANCES ("0,1,6"), NY_ISO_OBBT_BUDGET_S
    ///   (20). Reads the same NY_REL_WHOLE_MIP_OBBT_* knobs
    ///   the production path reads.
    pub(crate) fn measure_obbt_box_width_iso(base: &std::path::Path) -> anyhow::Result<()> {
        use ndarray::Array1;
        use ny_tensor::BoundedTensor;
        use std::collections::HashMap;
        use std::path::Path;
        use std::time::{Duration, Instant};

        let dir = base.display().to_string();
        let instance_list =
            std::env::var("NY_ISO_INSTANCES").unwrap_or_else(|_| "0,1,2,3,4,5,6,7,8,9".into());
        let instances: Vec<usize> = instance_list
            .split(',')
            .map(|s| {
                s.trim()
                    .parse()
                    .map_err(|e| anyhow::anyhow!("invalid NY_ISO_INSTANCES entry {s:?}: {e}"))
            })
            .collect::<anyhow::Result<_>>()?;
        anyhow::ensure!(
            !instances.is_empty(),
            "NY_ISO_INSTANCES must select at least one row"
        );
        let obbt_budget_s: f64 = std::env::var("NY_ISO_OBBT_BUDGET_S")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20.0);

        let csv = format!("{dir}/instances.csv");
        let csv_text = std::fs::read_to_string(&csv).expect("read validated instances.csv");
        let lines: Vec<&str> = csv_text.lines().collect();

        println!("\n════════ coupled-OBBT big-M shrink — iso diff nets ════════");
        println!("bench dir: {dir}");
        for &idx in &instances {
            let line = lines
                .get(idx)
                .ok_or_else(|| anyhow::anyhow!("instances.csv has no row {idx}"))?;
            // Parse the f/g onnx relative paths + the vnnlib path out of the row.
            let f_rel = line
                .split("'onnx/")
                .nth(1)
                .and_then(|s| s.split('\'').next())
                .map(|s| format!("onnx/{s}"));
            let g_rel = line
                .split("'onnx/")
                .nth(2)
                .and_then(|s| s.split('\'').next())
                .map(|s| format!("onnx/{s}"));
            let v_rel = line
                .split(",./")
                .nth(1)
                .and_then(|s| s.split(',').next())
                .map(|s| s.to_string());
            let f_rel =
                f_rel.ok_or_else(|| anyhow::anyhow!("instance {idx}: missing first ONNX path"))?;
            let g_rel =
                g_rel.ok_or_else(|| anyhow::anyhow!("instance {idx}: missing second ONNX path"))?;
            let v_rel =
                v_rel.ok_or_else(|| anyhow::anyhow!("instance {idx}: missing VNN-LIB path"))?;
            let fp = format!("{dir}/{f_rel}");
            let gp = format!("{dir}/{g_rel}");
            let vp = format!("{dir}/{v_rel}");
            for p in [&fp, &gp, &vp] {
                anyhow::ensure!(
                    Path::new(p).is_file(),
                    "instance {idx}: missing benchmark file {p}"
                );
            }

            let spec = ny_onnx::vnnlib::load_vnnlib(&vp).expect("load vnnlib");
            let dual = spec.dual_network.as_ref().expect("dual-network spec");
            let epsilon = match dual.property {
                ny_onnx::vnnlib::DualNetworkProperty::EpsilonEquivalence { epsilon } => epsilon,
                ref other => anyhow::bail!(
                    "instance {idx}: expected epsilon-equivalence property, found {other:?}"
                ),
            };
            let input_bounds = crate::commands::vnncomp::bounds_from_f64(&dual.f_input_bounds)
                .expect("input bounds");
            let graph_f =
                crate::commands::vnncomp::load_graph_network(Path::new(&fp)).expect("load f");
            let graph_g =
                crate::commands::vnncomp::load_graph_network(Path::new(&gp)).expect("load g");
            let diff = build_difference_network(&graph_f, &graph_g).expect("diff net");

            let lo: Vec<f32> = input_bounds.iter().map(|b| b.lower()).collect();
            let hi: Vec<f32> = input_bounds.iter().map(|b| b.upper()).collect();
            let input =
                BoundedTensor::new(Array1::from(lo).into_dyn(), Array1::from(hi).into_dyn())
                    .expect("input tensor");

            // Production node boxes (CROWN-IBP).
            let node_bt = diff
                .collect_crown_ibp_bounds_dag_with_deadline_and_engine(&input, None, None)
                .expect("crown-ibp node bounds");
            let mut flat: HashMap<String, Vec<Bound>> = HashMap::new();
            for (name, bt) in &node_bt {
                flat.insert(
                    name.clone(),
                    crate::commands::beta_crown::mip_preprocess::bounded_tensor_to_bounds(bt)
                        .expect("flatten"),
                );
            }

            // Declared-shape clone (production `shaped`).
            let mut shaped = diff.clone();
            if shaped.declared_shape(ny_propagate::NETWORK_INPUT).is_none() {
                shaped.set_declared_shape(ny_propagate::NETWORK_INPUT, input.shape().to_vec());
            }
            for (name, bt) in &node_bt {
                if shaped.declared_shape(name).is_none() {
                    shaped.set_declared_shape(name.clone(), bt.shape().to_vec());
                }
            }

            let band = ny_tensor::next_down_f32(epsilon as f32).max(0.0);
            let count_below_band = |f: &HashMap<String, Vec<Bound>>| -> usize {
                f.values()
                    .flat_map(|v| v.iter())
                    .filter(|b| {
                        let w = f64::from(b.upper() - b.lower());
                        w.is_finite() && w <= 2.0 * f64::from(band)
                    })
                    .count()
            };

            let (m0, md0, o0, tot0) = super::super::box_width_stats(&flat);
            let b0 = count_below_band(&flat);
            let far = Instant::now() + Duration::from_hours(1);
            let skip_alpha = std::env::var("NY_ISO_SKIP_ALPHA").is_ok();
            if !skip_alpha {
                super::super::intersect_alpha_crown_tightening(&diff, &input, &mut flat, far);
            }
            let (m1, md1, o1, _) = super::super::box_width_stats(&flat);
            // Snapshot the α-CROWN boxes so the conditioned pass starts from the
            // SAME baseline as the unconditioned one (apples-to-apples shrink).
            let flat_alpha = flat.clone();

            let obbt_deadline = Instant::now() + Duration::from_secs_f64(obbt_budget_s);
            let t0 = Instant::now();
            let diag = super::super::obbt_tighten_boxes(
                &shaped,
                &input_bounds,
                &mut flat,
                obbt_deadline,
                None,
            );
            let obbt_wall = t0.elapsed().as_secs_f64();
            let (m2, md2, o2, _) = super::super::box_width_stats(&flat);

            println!("\n── instance {idx}  ({v_rel})  ε={epsilon}  ({tot0} finite widths) ──");
            println!(
                "  BEFORE           max={m0:>12.1}  median={md0:>10.3}  >1000={o0:>4}  ≤band={b0:>4}",
            );
            println!("  α-CROWN ∩        max={m1:>12.1}  median={md1:>10.3}  >1000={o1:>4}");
            println!(
                "  + coupled OBBT   max={m2:>12.1}  median={md2:>10.3}  >1000={o2:>4}  ≤band={:>4}   ({obbt_wall:.1}s)",
                count_below_band(&flat)
            );
            println!(
                "     OBBT coverage: selectable={} targets={} tightened_cols={} boxes_shrunk={}",
                diag.selectable, diag.targets, diag.tightened_cols, diag.boxes_shrunk
            );

            // PROPERTY-CONDITIONED OBBT: add the FIRST band-violation row
            // (h_0 < -ε ⇒ coeffs=+e_0, threshold=-ε) into the OBBT LP so every
            // intermediate is bounded WITHIN that row's violation region. Same
            // α-CROWN baseline, same budget — isolates the conditioning lever.
            let n_out = crate::commands::beta_crown::graph_mip::encode_graph_with_deadline(
                &shaped,
                &input_bounds,
                &flat_alpha,
                None,
            )
            .map(|e| e.output_vars.len())
            .unwrap_or(0);
            if n_out > 0 {
                let eps = ny_tensor::next_up_f32(epsilon as f32);
                let mut coeffs = vec![0.0f32; n_out];
                coeffs[0] = 1.0;
                let mut flat_cond = flat_alpha.clone();
                let cond_deadline = Instant::now() + Duration::from_secs_f64(obbt_budget_s);
                let t1 = Instant::now();
                let cdiag = super::super::obbt_tighten_boxes(
                    &shaped,
                    &input_bounds,
                    &mut flat_cond,
                    cond_deadline,
                    Some((&coeffs, -(eps as f64))),
                );
                let cond_wall = t1.elapsed().as_secs_f64();
                let (m3, md3, o3, _) = super::super::box_width_stats(&flat_cond);
                println!(
                    "  + COND OBBT(row0) max={m3:>12.1}  median={md3:>10.3}  >1000={o3:>4}  ≤band={:>4}   ({cond_wall:.1}s)",
                    count_below_band(&flat_cond)
                );
                println!(
                    "     cond coverage: selectable={} targets={} tightened_cols={} boxes_shrunk={}",
                    cdiag.selectable, cdiag.targets, cdiag.tightened_cols, cdiag.boxes_shrunk
                );
            }
        }
        println!("═══════════════════════════════════════════════════════════\n");
        Ok(())
    }

    pub(crate) fn measure_iso_diff_whole_net_mip_bb2b6088(base: &std::path::Path) {
        use ndarray::Array1;
        use ny_mip::{MipBackend, MipConfig, MipResult, MipSolver};
        use ny_propagate::NETWORK_INPUT;
        use ny_tensor::BoundedTensor;
        use std::collections::HashMap;
        use std::path::Path;
        use std::time::{Duration, Instant};

        let dir = base.display().to_string();
        let f_rel = std::env::var("NY_ISO_F_ONNX")
            .unwrap_or_else(|_| "onnx/original/ACASXU_run2a_3_8_batch_2000.onnx".into());
        let g_rel = std::env::var("NY_ISO_G_ONNX").unwrap_or_else(|_| {
            "onnx/perturbed/ACASXU_run2a_3_8_batch_2000_perturbed_6.onnx".into()
        });
        let v_rel =
            std::env::var("NY_ISO_VNNLIB").unwrap_or_else(|_| "vnnlib/instance_6.vnnlib".into());
        let row_secs: f64 = std::env::var("NY_ISO_ROW_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60.0);
        let total_secs: f64 = std::env::var("NY_ISO_TOTAL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(480.0);

        let fp = format!("{dir}/{f_rel}");
        let gp = format!("{dir}/{g_rel}");
        let vp = format!("{dir}/{v_rel}");
        for p in [&fp, &gp, &vp] {
            assert!(
                Path::new(p).is_file(),
                "validated benchmark file disappeared: {p}"
            );
        }

        // ε + shared input box from the parsed dual spec (production path).
        let spec = ny_onnx::vnnlib::load_vnnlib(&vp).expect("load vnnlib");
        let dual = spec.dual_network.as_ref().expect("dual-network spec");
        let epsilon = match dual.property {
            ny_onnx::vnnlib::DualNetworkProperty::EpsilonEquivalence { epsilon } => epsilon,
            ref other => panic!("not an epsilon-equivalence iso instance: {other:?}"),
        };
        let input_bounds =
            crate::commands::vnncomp::bounds_from_f64(&dual.f_input_bounds).expect("input bounds");

        // Difference network h = f − g (two ACAS towers).
        let graph_f =
            crate::commands::vnncomp::load_graph_network(Path::new(&fp)).expect("load f onnx");
        let graph_g =
            crate::commands::vnncomp::load_graph_network(Path::new(&gp)).expect("load g onnx");
        let diff = build_difference_network(&graph_f, &graph_g).expect("diff net");

        // Replicate the production encode (whole_net_certified_band_unsat_inner)
        // so we can read the true MILP size + CROWN root gap.
        let lo: Vec<f32> = input_bounds.iter().map(|b| b.lower()).collect();
        let hi: Vec<f32> = input_bounds.iter().map(|b| b.upper()).collect();
        let input = BoundedTensor::new(Array1::from(lo).into_dyn(), Array1::from(hi).into_dyn())
            .expect("input tensor");
        // ── node-bound TIGHTNESS comparison ──────────────────────────────
        // The 546 binaries + ±14000 big-Ms in the root-IBP run: a looseness
        // artifact (fixable by tighter per-node boxes) or genuine (the
        // undivided box makes the towers' neurons truly unstable)? Encode under
        // three bound methods and compare binaries + big-M. The finisher fix
        // should adopt whichever is tightest.
        let out_name = diff.output_name().to_string();
        let encode_with = |node_bounds_bt: &HashMap<String, BoundedTensor>| {
            let mut flat: HashMap<String, Vec<Bound>> = HashMap::new();
            for (name, bt) in node_bounds_bt {
                flat.insert(
                    name.clone(),
                    crate::commands::beta_crown::mip_preprocess::bounded_tensor_to_bounds(bt)
                        .expect("flatten node bounds"),
                );
            }
            let mut shaped = diff.clone();
            if shaped.declared_shape(NETWORK_INPUT).is_none() {
                shaped.set_declared_shape(NETWORK_INPUT, input.shape().to_vec());
            }
            for (name, bt) in node_bounds_bt {
                if shaped.declared_shape(name).is_none() {
                    shaped.set_declared_shape(name.clone(), bt.shape().to_vec());
                }
            }
            let enc = crate::commands::beta_crown::graph_mip::encode_graph_with_deadline(
                &shaped,
                &input_bounds,
                &flat,
                None,
            )
            .expect("encode diff net");
            // Worst big-M proxy: max |bound| across all node boxes.
            let max_m = flat
                .values()
                .flat_map(|v| v.iter())
                .map(|b| b.lower().abs().max(b.upper().abs()))
                .fold(0.0f32, f32::max);
            let out_h = flat
                .get(&out_name)
                .map(|ob| {
                    ob.iter()
                        .map(|b| b.lower().abs().max(b.upper().abs()))
                        .fold(0.0f32, f32::max)
                })
                .unwrap_or(f32::NAN);
            (enc, max_m, out_h)
        };

        println!("\n════════ iso diff-net WHOLE-NET MILP — ay bb2b6088 ════════");
        println!("instance : {v_rel}   ε = {epsilon}");
        println!("── node-bound tightness (binaries + big-M) ──");

        let ibp = diff.collect_node_bounds(&input).expect("ibp node bounds");
        let crown_default = diff
            .collect_crown_ibp_bounds_dag_with_engine(&input, None)
            .expect("crown-ibp default");
        let crown_full = diff
            .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_width_threshold(
                &input,
                ibp.clone(),
                None,
                0.0, // tighten EVERY node (no width skip)
            )
            .expect("crown full")
            .bounds;

        let (enc_ibp, m_ibp, h_ibp) = encode_with(&ibp);
        let (enc_def, m_def, h_def) = encode_with(&crown_default);
        let (enc_full, m_full, h_full) = encode_with(&crown_full);
        let line = |lbl: &str,
                    e: &crate::commands::beta_crown::graph_mip::GraphMipEncoding,
                    mm: f32,
                    oh: f32| {
            let nnz: usize = e.problem.rows().iter().map(|r| r.coeffs.len()).sum();
            println!(
                "  {lbl:<20} binaries={:<4} cols={:<5} rows={:<5} nnz={:<6} maxBigM={mm:>12.2} out.max|h|={oh:.3}",
                e.binary_vars.len(),
                e.problem.num_cols(),
                e.problem.num_rows(),
                nnz
            );
        };
        line("IBP", &enc_ibp, m_ibp, h_ibp);
        line("CROWN-IBP(default)", &enc_def, m_def, h_def);
        line("CROWN-full(wthr=0)", &enc_full, m_full, h_full);

        // Solve with the TIGHTEST encoding (fewest binaries; tie → CROWN-full).
        let b_ibp = enc_ibp.binary_vars.len();
        let b_def = enc_def.binary_vars.len();
        let b_full = enc_full.binary_vars.len();
        let (chosen_label, encoded) = if b_full <= b_def && b_full <= b_ibp {
            ("CROWN-full(wthr=0)", enc_full)
        } else if b_def <= b_ibp {
            ("CROWN-IBP(default)", enc_def)
        } else {
            ("IBP", enc_ibp)
        };
        let odim = encoded.output_vars.len();
        println!(
            "→ solving with {chosen_label} ({} binaries, maxBigM {:.1})",
            encoded.binary_vars.len(),
            [
                ("CROWN-full(wthr=0)", m_full),
                ("CROWN-IBP(default)", m_def),
                ("IBP", m_ibp)
            ]
            .iter()
            .find(|(l, _)| *l == chosen_label)
            .map(|(_, m)| *m)
            .unwrap_or(f32::NAN)
        );

        // Conservative inward ε (matches production's inward rounding direction).
        let eps32 = ny_tensor::next_down_f32(epsilon as f32).max(0.0);
        let rows = band_rows(odim, eps32);
        println!(
            "solving {} band rows  (per-row cap {row_secs}s, overall cap {total_secs}s)",
            rows.len()
        );

        let overall = Instant::now() + Duration::from_secs_f64(total_secs);
        let mut certified = 0usize;
        let mut sat = 0usize;
        let mut timed_out = 0usize;
        let mut attempted = 0usize;
        for (ri, (coeffs, thr)) in rows.iter().enumerate() {
            if Instant::now() >= overall {
                println!("row {ri}: SKIPPED (overall cap reached)");
                continue;
            }
            let remaining = overall
                .saturating_duration_since(Instant::now())
                .as_secs_f64();
            let budget = row_secs.min(remaining).max(0.5);
            let mut enc = encoded.clone();
            enc.add_violation_row(coeffs, f64::from(*thr))
                .expect("violation row");
            let solver = MipSolver::new(
                enc.into_parts(),
                MipConfig {
                    backend: MipBackend::Ay,
                    timeout_secs: budget,
                    parallel_split: 1,
                    ..Default::default()
                },
            );
            let t0 = Instant::now();
            let res = solver.check_feasibility();
            let wall = t0.elapsed().as_secs_f64();
            attempted += 1;
            let tag = match &res {
                Ok(MipResult::Unsat { certified: true }) => {
                    certified += 1;
                    "UNSAT(certified)".to_string()
                }
                Ok(MipResult::Unsat { certified: false }) => "UNSAT(uncertified)".to_string(),
                Ok(MipResult::Sat { .. }) => {
                    sat += 1;
                    "SAT(violation-in-box)".to_string()
                }
                Ok(other) => {
                    timed_out += 1;
                    format!("{other:?}")
                }
                Err(e) => format!("ERROR({e})"),
            };
            println!("row {ri:2} (thr={thr:+.4}): {tag} in {wall:.1}s");
        }

        println!("──────── RESULT ────────");
        println!(
            "certified-UNSAT: {certified}/{}   SAT: {sat}   inconclusive: {timed_out}   attempted: {attempted}",
            rows.len()
        );
        let verdict = if sat > 0 {
            "INSTANCE IS SAT (property false on the box) — pick another for the UNSAT probe"
        } else if certified == rows.len() {
            "WHOLE BAND CERTIFIED — ay bb2b6088 CLOSES this instance at the given budget"
        } else {
            "STILL GATED — not all rows certified in budget (ladder needs more)"
        };
        println!("verdict: {verdict}");
        println!("═══════════════════════════════════════════════════════════\n");
    }

    // ── shared helpers for the k-vs-depth + ay-ceiling measurement ──────────
    fn load_iso_diff(base: &std::path::Path) -> (GraphNetwork, Vec<Bound>, f64, String) {
        let dir = base.display().to_string();
        let f_rel = std::env::var("NY_ISO_F_ONNX")
            .unwrap_or_else(|_| "onnx/original/ACASXU_run2a_3_8_batch_2000.onnx".into());
        let g_rel = std::env::var("NY_ISO_G_ONNX").unwrap_or_else(|_| {
            "onnx/perturbed/ACASXU_run2a_3_8_batch_2000_perturbed_6.onnx".into()
        });
        let v_rel =
            std::env::var("NY_ISO_VNNLIB").unwrap_or_else(|_| "vnnlib/instance_6.vnnlib".into());
        let fp = format!("{dir}/{f_rel}");
        let gp = format!("{dir}/{g_rel}");
        let vp = format!("{dir}/{v_rel}");
        for p in [&fp, &gp, &vp] {
            assert!(
                std::path::Path::new(p).is_file(),
                "validated benchmark file disappeared: {p}"
            );
        }
        let spec = ny_onnx::vnnlib::load_vnnlib(&vp).expect("load vnnlib");
        let dual = spec.dual_network.as_ref().expect("dual-network spec");
        let epsilon = match dual.property {
            ny_onnx::vnnlib::DualNetworkProperty::EpsilonEquivalence { epsilon } => epsilon,
            ref other => panic!("not an epsilon-equivalence iso instance: {other:?}"),
        };
        let input_bounds =
            crate::commands::vnncomp::bounds_from_f64(&dual.f_input_bounds).expect("input bounds");
        let gf = crate::commands::vnncomp::load_graph_network(std::path::Path::new(&fp))
            .expect("load f onnx");
        let gg = crate::commands::vnncomp::load_graph_network(std::path::Path::new(&gp))
            .expect("load g onnx");
        let diff = build_difference_network(&gf, &gg).expect("diff net");
        (diff, input_bounds, epsilon, v_rel)
    }

    /// Encode the diff net over `input_bounds` with CROWN-IBP node boxes (the
    /// finisher path) and return the encoding (k = binary_vars.len()).
    fn encode_over_box(
        diff: &GraphNetwork,
        input_bounds: &[Bound],
    ) -> crate::commands::beta_crown::graph_mip::GraphMipEncoding {
        use ny_propagate::NETWORK_INPUT;
        use ny_tensor::BoundedTensor;
        use std::collections::HashMap;
        let lo: Vec<f32> = input_bounds.iter().map(|b| b.lower()).collect();
        let hi: Vec<f32> = input_bounds.iter().map(|b| b.upper()).collect();
        let input = BoundedTensor::new(
            ndarray::Array1::from(lo).into_dyn(),
            ndarray::Array1::from(hi).into_dyn(),
        )
        .expect("input tensor");
        let nb = diff
            .collect_crown_ibp_bounds_dag_with_engine(&input, None)
            .expect("crown-ibp");
        let mut flat: HashMap<String, Vec<Bound>> = HashMap::new();
        for (n, bt) in &nb {
            flat.insert(
                n.clone(),
                crate::commands::beta_crown::mip_preprocess::bounded_tensor_to_bounds(bt)
                    .expect("flatten"),
            );
        }
        let mut shaped = diff.clone();
        if shaped.declared_shape(NETWORK_INPUT).is_none() {
            shaped.set_declared_shape(NETWORK_INPUT, input.shape().to_vec());
        }
        for (n, bt) in &nb {
            if shaped.declared_shape(n).is_none() {
                shaped.set_declared_shape(n.clone(), bt.shape().to_vec());
            }
        }
        crate::commands::beta_crown::graph_mip::encode_graph_with_deadline(
            &shaped,
            input_bounds,
            &flat,
            None,
        )
        .expect("encode diff net")
    }

    fn widest_dim(b: &[Bound]) -> usize {
        (0..b.len())
            .max_by(|&i, &j| {
                (b[i].upper() - b[i].lower())
                    .partial_cmp(&(b[j].upper() - b[j].lower()))
                    .unwrap()
            })
            .unwrap()
    }

    fn bisect(b: &[Bound], d: usize) -> (Vec<Bound>, Vec<Bound>) {
        let mid = f32::midpoint(b[d].lower(), b[d].upper());
        let mut lo = b.to_vec();
        let mut hi = b.to_vec();
        lo[d] = Bound::new(b[d].lower(), mid);
        hi[d] = Bound::new(mid, b[d].upper());
        (lo, hi)
    }

    /// DECISIVE MEASUREMENT (A + B): does input-splitting reach an ay-tractable
    /// per-domain k at a shallow-enough depth to close the 10 on a long budget,
    /// or is it explosion-dead?
    ///
    /// A — k(depth): greedily descend the input-split tree (bisect the WIDEST
    ///     dim, keep the HARDER child = more unstable neurons) and record the
    ///     worst-leaf binary count k at each depth, with the leaf count 2^depth.
    /// B — ay ceiling: at sub-boxes giving k ≈ 80/150/250/350, solve real
    ///     per-domain band rows on ay bb2b6088 with a 12s cap; the largest k
    ///     that certifies is the tractability ceiling.
    /// Verdict = the leaf count 2^depth at that ceiling k.
    ///
    /// MEASURED (instance_6, 2026-07-17) — VERDICT: EXPLOSION-DEAD.
    ///   A) worst-leaf k(depth) is GRADUAL, not a cliff:
    ///        d0=546  d4=463  d8=328  d12=242  d16=116  d18=95  d20=63
    ///      → reaching ay's w5 rung (k≤83) needs depth ≥19 = ≥262 144 leaves.
    ///   B) ay bb2b6088 certified NO per-domain leaf in bounded time: k=250
    ///      (depth 12) ran >400s, and even k=63 (w5-rung scale, depth 20) never
    ///      returned in >450s — the 12s cap is ignored and the small-LP /
    ///      many-exact-binary leaf solve does not terminate with a certified
    ///      verdict (severe deadline-honor gap and/or genuine intractability;
    ///      needs ay-side triage — file-to-ay).
    ///   VERDICT ARITHMETIC: even granting a hypothetical fast leaf solve, the
    ///   shallowest depth with k in ay's plausible range is ≥19 (≥262K leaves);
    ///   at any realistic per-leaf wall that is astronomically infeasible. And
    ///   empirically ay does not close even one such leaf. So 90/100 is the
    ///   budget-bounded MEASURED ceiling; 100/100 needs BOTH a fundamentally
    ///   better splitter (low-k at shallow depth) AND an exact MILP that
    ///   certifies these ~60-250-binary leaves in seconds. Neither exists today.
    ///
    /// Run through `ny vnncomp-research graph-mip k-vs-depth-ay
    /// --bench-dir <isomorphic-acas-2.0>`.
    pub(crate) fn measure_iso_k_vs_depth_and_ay_ceiling(base: &std::path::Path) {
        use ny_mip::{MipBackend, MipConfig, MipResult, MipSolver};
        use std::time::Instant;

        let (diff, input_bounds, epsilon, v_rel) = load_iso_diff(base);
        let eps32 = ny_tensor::next_down_f32(epsilon as f32).max(0.0);

        // ── MEASUREMENT A: worst-leaf k vs depth ──────────────────────────
        println!("\n════════ MEASUREMENT A: k vs depth — instance {v_rel} ════════");
        let max_depth = 20usize;
        let mut cur = input_bounds;
        let enc0 = encode_over_box(&diff, &cur);
        let odim = enc0.output_vars.len();
        let mut cur_k = enc0.binary_vars.len();
        let mut curve: Vec<(usize, usize, Vec<Bound>)> = Vec::new();
        for depth in 0..=max_depth {
            curve.push((depth, cur_k, cur.clone()));
            if depth < max_depth {
                let d = widest_dim(&cur);
                let (lc, hc) = bisect(&cur, d);
                let kl = encode_over_box(&diff, &lc).binary_vars.len();
                let kh = encode_over_box(&diff, &hc).binary_vars.len();
                if kh >= kl {
                    cur = hc;
                    cur_k = kh;
                } else {
                    cur = lc;
                    cur_k = kl;
                }
            }
        }
        println!("depth :  k (unstable neurons) | leaves 2^depth");
        for &(d, k, _) in &curve {
            println!("  {d:2}  :  {k:4}                   | {}", 1u64 << d);
        }

        // ── MEASUREMENT B: ay bb2b6088 per-domain ceiling (12s/row cap) ────
        println!("\n════════ MEASUREMENT B: ay bb2b6088 per-domain ceiling (12s/row) ════════");
        // w5-rung first (k≈53-83 = the scale ay's bb2b6088 "w5 salvage"
        // targets): if the SMALLEST leaves don't certify, the per-domain path
        // is dead regardless of depth. Then climb to find the exact ceiling.
        // Target ks env-tunable (comma-separated) so a re-measure can walk the
        // band in order and gate later (dangerous) ks on earlier certification.
        let targets: Vec<usize> = std::env::var("NY_ISO_B_TARGETS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse().ok())
                    .collect::<Vec<usize>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![63, 80, 120, 180, 250]);
        let per_target_rows: usize = std::env::var("NY_ISO_B_ROWS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let cap: f64 = std::env::var("NY_ISO_B_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12.0);
        let rows = band_rows(odim, eps32);
        for &tk in &targets {
            let (bd, bk, bx) = curve
                .iter()
                .min_by_key(|(_, k, _)| (*k as i64 - tk as i64).abs())
                .unwrap();
            let enc = encode_over_box(&diff, bx);
            let mut cert = 0usize;
            let mut att = 0usize;
            let mut worst = 0.0f64;
            // Representative rows per k (bound the wall; the per-row solve is
            // the difficulty unit — a full leaf is ~10× this).
            for (ri, (coeffs, thr)) in rows.iter().enumerate().take(per_target_rows) {
                let mut e = enc.clone();
                e.add_violation_row(coeffs, f64::from(*thr)).expect("row");
                let solver = MipSolver::new(
                    e.into_parts(),
                    MipConfig {
                        backend: MipBackend::Ay,
                        timeout_secs: cap,
                        parallel_split: 1,
                        ..Default::default()
                    },
                );
                let t0 = Instant::now();
                let r = solver.check_feasibility();
                let w = t0.elapsed().as_secs_f64();
                att += 1;
                worst = worst.max(w);
                let tag = match &r {
                    Ok(MipResult::Unsat { certified: true }) => {
                        cert += 1;
                        "UNSAT(certified)"
                    }
                    Ok(MipResult::Unsat { certified: false }) => "UNSAT(uncertified)",
                    Ok(MipResult::Sat { .. }) => "SAT",
                    Ok(_) => "inconclusive",
                    Err(_) => "ERROR",
                };
                println!("  k≈{tk} (actual k={bk} @depth {bd}) row{ri}: {tag} in {w:.1}s");
            }
            println!(
                "  → k={bk} @depth {bd}: {cert}/{att} rows certified in ≤12s (worst {worst:.1}s), leaves=2^{bd}={}",
                1u64 << bd
            );
        }
        println!("═══════════════════════════════════════════════════════════════\n");
    }

    /// #rel-corpus EMIT: serialize the ACAS isomorphic diff-leaf decision MILP
    /// to STANDALONE exact-rational QF_LRA SMT-LIB at three split points so the
    /// ay solver team can iterate on this class WITHOUT an ny build in the loop:
    ///   (a) whole-net root  (depth 0,   target_k≈546);
    ///   (b) a shallow leaf  (target_k≈150, depth picked from the k-vs-depth
    ///       curve — the harder input-split child at each bisection);
    ///   (c) a deep leaf     (depth 20,  target_k≈63).
    ///
    /// The encode path is the production finisher path (`encode_over_box` =
    /// CROWN-IBP node boxes → `encode_graph_with_deadline`), IDENTICAL to
    /// MEASUREMENT A, so k(depth) matches the measured curve
    /// (d0=546 … d16=116 … d20=63). Each `_dec` file is the encoding PLUS one
    /// representative band-decision row (`h_0 <= -ε`, the first `band_rows`
    /// entry) → `check-sat`; the MIP is infeasible iff that band row is
    /// VERIFIED on the box. Each `_min` file drops the row and instead
    /// `(minimize c<y_0>)` — the optimization spelling of the same question
    /// (min h_0 over the box vs -ε). Both go through the ay-native lowering
    /// `ny_mip::to_smtlib_{decision,minimize}` (= the exact bytes the ay backend
    /// streams), so binaries are `{0,1}` disjunctions and every coefficient is
    /// an exact dyadic rational.
    ///
    /// Files land in the explicit output directory supplied by the research
    /// CLI; the ACTUAL measured k is used in each filename.
    pub(crate) fn emit_iso_diff_smtlib_corpus(base: &std::path::Path, out_dir: &std::path::Path) {
        let (diff, input_bounds, epsilon, v_rel) = load_iso_diff(base);
        let eps32 = ny_tensor::next_down_f32(epsilon as f32).max(0.0);

        std::fs::create_dir_all(out_dir).expect("create corpus dir");

        // ── build the worst-leaf k-vs-depth curve (== MEASUREMENT A) ──────────
        let max_depth = 20usize;
        let mut cur = input_bounds;
        let enc0 = encode_over_box(&diff, &cur);
        let odim = enc0.output_vars.len();
        let mut cur_k = enc0.binary_vars.len();
        let mut curve: Vec<(usize, usize, Vec<Bound>)> = Vec::new();
        for depth in 0..=max_depth {
            curve.push((depth, cur_k, cur.clone()));
            if depth < max_depth {
                let d = widest_dim(&cur);
                let (lc, hc) = bisect(&cur, d);
                let kl = encode_over_box(&diff, &lc).binary_vars.len();
                let kh = encode_over_box(&diff, &hc).binary_vars.len();
                if kh >= kl {
                    cur = hc;
                    cur_k = kh;
                } else {
                    cur = lc;
                    cur_k = kl;
                }
            }
        }
        println!("\n════════ EMIT iso diff-net SMT-LIB corpus — instance {v_rel} ════════");
        println!("ε = {epsilon}  (inward-rounded eps32 = {eps32})  output dim = {odim}");
        println!("k-vs-depth curve (worst input-split child):");
        for &(d, k, _) in &curve {
            println!("  depth {d:2} : k = {k:4}   (leaves 2^{d} = {})", 1u64 << d);
        }

        // Representative band-decision row: refute h_0 < -ε ⇒ row +e_0, thr -ε.
        let rows = band_rows(odim, eps32);
        let (coeffs0, thr0) = rows[0].clone();

        // ── three split points: (label, target_k, forced_depth?) ──────────────
        // root is pinned to depth 0; the two leaves are the curve entries whose
        // k is closest to the target.
        let picks: [(&str, usize, Option<usize>); 3] = [
            ("root", 546, Some(0)),
            ("leaf", 150, None),
            ("leaf", 63, Some(20)),
        ];

        println!("\n── emitted files ──");
        for (label, target_k, forced_depth) in picks {
            let (depth, k, bx) = match forced_depth {
                Some(fd) => {
                    let (d, k, b) = &curve[fd];
                    (*d, *k, b.clone())
                }
                None => {
                    let (d, k, b) = curve
                        .iter()
                        .min_by_key(|(_, k, _)| (*k as i64 - target_k as i64).abs())
                        .unwrap();
                    (*d, *k, b.clone())
                }
            };

            // Encode this box (CROWN-IBP finisher path). The `_min` objective is
            // the first output column y_0; the `_dec` problem adds the band row.
            let enc_min = encode_over_box(&diff, &bx);
            let obj_col = enc_min.output_vars[0];
            let mut enc_dec = enc_min.clone();
            enc_dec
                .add_violation_row(&coeffs0, f64::from(thr0))
                .expect("add band-decision row");

            let dec_problem = enc_dec.into_parts().problem;
            let min_problem = enc_min.into_parts().problem;

            let nnz = |p: &ny_mip::MilpProblem| -> usize {
                p.rows().iter().map(|r| r.coeffs.len()).sum()
            };
            let cols = dec_problem.num_cols();
            let rows_dec = dec_problem.num_rows();
            let nnz_dec = nnz(&dec_problem);
            let binaries = dec_problem.cols().iter().filter(|c| c.integer).count();

            let dec_txt = ny_mip::to_smtlib_decision(&dec_problem).expect("lower decision SMT-LIB");
            let min_txt =
                ny_mip::to_smtlib_minimize(&min_problem, obj_col).expect("lower minimize SMT-LIB");

            let dec_name = format!("acasxu_iso_inst6_{label}{k}_dec.smt2");
            let min_name = format!("acasxu_iso_inst6_{label}{k}_min.smt2");
            let dec_path = out_dir.join(&dec_name);
            let min_path = out_dir.join(&min_name);
            std::fs::write(&dec_path, &dec_txt).expect("write dec smt2");
            std::fs::write(&min_path, &min_txt).expect("write min smt2");

            println!(
                "  {dec_name}: target_k={target_k} actual_k={k} depth={depth} \
                 cols={cols} rows={rows_dec} nnz={nnz_dec} binaries={binaries} \
                 bytes={} form=dec",
                dec_txt.len()
            );
            println!(
                "  {min_name}: target_k={target_k} actual_k={k} depth={depth} \
                 cols={} rows={} nnz={} binaries={} bytes={} form=min (obj=min c{})",
                min_problem.num_cols(),
                min_problem.num_rows(),
                nnz(&min_problem),
                min_problem.cols().iter().filter(|c| c.integer).count(),
                min_txt.len(),
                obj_col.0
            );
        }
        println!("corpus dir: {}", out_dir.display());
        println!("═══════════════════════════════════════════════════════════════\n");
    }

    /// #rel-multineuron LEVER PROBE — multi-neuron (k-ReLU / octahedral)
    /// intra-layer coupling on the diff-net band objective, vs input-split depth.
    /// The one SOUND method that provably beats the triangle-LP optimum (α gives
    /// 0 gain there). Intra-layer pair coupling — NOT the f-g paired-ReLU
    /// coupling already measured dead by the superseded offline prototype.
    /// Fires on the MLP diff net via
    /// NY_MULTINEURON_MLP=1 (a 1-line target_relu_nodes scope extension: the
    /// main-path pair machinery is layer-agnostic — combined_rows_octahedra does
    /// set_output(pre_node), so it works on Add/MatMul-fed ReLUs; sound max fold).
    ///
    /// MEASURED (instance_6, 2026-07-17) — VERDICT: multineuron is DEAD for the 10.
    ///   • The lever FIRES on the MLP diff net (capability unlocked): 12 ReLU
    ///     targets (AddConstant-fed), facets MASSIVELY non-degenerate (top score
    ///     up to 2.9e6 — the ACAS diff net has rich excluded-corner structure,
    ///     unlike cGAN's degenerate 0.0). The coordinator was RIGHT about that.
    ///   • BUT the premise (root band-objective bound ≈ 0.0501, need >1e-4) is
    ///     REFUTED. The β=0 objective α-CROWN bound at the ROOT is −61560, not
    ///     0.0501 (0.0501 was a DEEP-BaB-converged figure). obj margin_min vs
    ///     depth (widest-dim hardest child): d0=−61560 d4=−13037 d8=−2452
    ///     d12=−1094 d16=−584 — still 4 orders from −ε at depth 16.
    ///   • Multineuron GAIN = 0 at every depth: despite score 2.9e6 facets, β>0
    ///     MONOTONICALLY WORSENS the margin (−61560→−72016), so the sound max
    ///     fold keeps β=0. Rich coupling structure does NOT translate to an
    ///     objective-LB gain through the k-ReLU Lagrangian injection here.
    ///   ⇒ shallow-split + multineuron does NOT finish it (bound astronomically
    ///     loose at shallow depth AND the lever adds nothing). Not the not-ay-
    ///     gated shot. 90/100 stands. The MLP-scope unlock (sound, default-OFF)
    ///     is kept for other relational surfaces where facets may actually bind.
    ///
    pub(crate) fn measure_iso_multineuron_root_tightening(base: &std::path::Path) {
        let (diff, input_bounds, epsilon, v_rel) = load_iso_diff(base);
        let eps = epsilon as f32;

        // multineuron-tightened band-objective margin_min over ONE sub-box.
        // (Nested fn: its own imports — enclosing-fn `use` is not inherited.)
        fn mn_obj_margin_min(diff: &GraphNetwork, box_bounds: &[Bound]) -> Option<f32> {
            use ny_propagate::bounds::AlphaCrownConfig;
            use ny_tensor::BoundedTensor;
            let lo: Vec<f32> = box_bounds.iter().map(|b| b.lower()).collect();
            let hi: Vec<f32> = box_bounds.iter().map(|b| b.upper()).collect();
            let input = BoundedTensor::new(
                ndarray::Array1::from(lo).into_dyn(),
                ndarray::Array1::from(hi).into_dyn(),
            )
            .ok()?;
            let cfg = AlphaCrownConfig::default();
            let (nb, alpha) = diff.collect_alpha_crown_bounds_dag(&input, &cfg).ok()?;
            let out_name = diff.output_name().to_string();
            let ob = nb.get(&out_name)?;
            let ob_lo: Vec<f32> = ob.lower().iter().copied().collect();
            let ob_hi: Vec<f32> = ob.upper().iter().copied().collect();
            let odim = ob_lo.len();
            // Band objectives ±e_i + valid (loose box) baseline; the β=0 injected
            // backward computes the true objective-specific α-CROWN bound.
            let mut objectives: Vec<Vec<f32>> = Vec::new();
            let mut baseline: Vec<(f32, f32)> = Vec::new();
            for i in 0..odim {
                let mut cneg = vec![0.0f32; odim];
                cneg[i] = -1.0;
                objectives.push(cneg);
                baseline.push((-ob_hi[i], -ob_lo[i]));
                let mut cpos = vec![0.0f32; odim];
                cpos[i] = 1.0;
                objectives.push(cpos);
                baseline.push((ob_lo[i], ob_hi[i]));
            }
            let t = ny_propagate::multineuron::root_inject::tighten_root_objective_bounds(
                diff,
                &input,
                &objectives,
                None,
                &nb,
                Some(&alpha),
                &baseline,
                None,
            );
            Some(t.iter().map(|&(l, _)| l).fold(f32::INFINITY, f32::min))
        }

        println!("\n════════ MULTINEURON obj-bound vs DEPTH — instance {v_rel} ════════");
        println!(
            "ε = {epsilon}   armed = {}",
            ny_propagate::multineuron::root_inject::enabled()
                && ny_propagate::multineuron::root_inject::mlp_enabled()
        );
        println!(
            "(multineuron-tightened band-objective margin_min; certify if ≥ −ε = {:.5})",
            -eps
        );
        println!("watch [multineuron] stderr: facet 'top score'>0 = non-degenerate; β>0 vs β=0 injected = facet gain");

        // Descend the split tree (bisect widest, keep the HARDER child = more
        // unstable) and probe multineuron at a few depths. If the objective
        // margin approaches −ε at a SHALLOW depth → shallow-split + multineuron
        // finishes it; if it stays astronomically loose → the lever is dead.
        let probe_depths = [0usize, 4, 8, 12, 16];
        let mut cur = input_bounds;
        for depth in 0..=16 {
            if probe_depths.contains(&depth) {
                let k = encode_over_box(&diff, &cur).binary_vars.len();
                match mn_obj_margin_min(&diff, &cur) {
                    Some(v) => println!(
                        "depth {depth:2}: k={k:3}  multineuron obj margin_min = {v:.5}  certifies={}",
                        v >= -eps
                    ),
                    None => println!("depth {depth:2}: k={k:3}  (multineuron probe failed)"),
                }
            }
            if depth < 16 {
                let d = widest_dim(&cur);
                let (lc, hc) = bisect(&cur, d);
                let kl = encode_over_box(&diff, &lc).binary_vars.len();
                let kh = encode_over_box(&diff, &hc).binary_vars.len();
                cur = if kh >= kl { hc } else { lc };
            }
        }
        println!("═══════════════════════════════════════════════════════════\n");
    }
}
