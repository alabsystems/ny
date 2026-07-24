// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Loop-level oracles for GRAPH-engine BaB conflict-clause learning (v2 port,
//! `conflict_clauses_graph`). Store-level discriminating oracles — purity
//! guard, record+prune round trip, gate-off inertness — live next to the
//! store in `conflict_clauses_graph::tests`; here we drive the REAL graph BaB
//! loops end to end and assert the A/B contract of the gate:
//!
//!   gate OFF (default) => the disabled store no-ops both entry points, the
//!   loop is byte-identical to baseline;
//!   gate ON => region-inclusion pruning may close domains early but must
//!   NEVER change a verdict (it only closes domains already proven safe by a
//!   same-run certificate).

use super::prelude::*;
use crate::beta_crown::conflict_clauses::set_test_gate_override;

/// Single-objective graph loop (`verify_graph_relu_split_with_bounds`):
/// gate ON vs OFF must produce the identical final verdict across thresholds
/// covering Verified, in-between, and PotentialViolation.
#[ntest::timeout(30000)]
#[test]
fn test_graph_relu_split_with_bounds_gate_on_vs_off_identical_verdicts() {
    use crate::beta_crown::domain::GraphPrecomputedBounds;

    // simple_graph_network: 2 -> Linear -> ReLU -> Linear -> 1 with two
    // unstable ReLUs over [-1, 1]^2 — the loop actually branches.
    let graph = super::gpu_bab::simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let config = || BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 200,
        max_depth: 10,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };

    for threshold in [-10.0f32, -0.5, 100.0] {
        set_test_gate_override(Some(false));
        let verifier_off = BetaCrownVerifier::new(config());
        let (node_bounds, output_bounds) = verifier_off
            .compute_initial_graph_bounds(&graph, &input, None)
            .unwrap();
        let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);
        let off = verifier_off
            .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], threshold, &precomputed)
            .expect("gate-off verify must succeed");

        set_test_gate_override(Some(true));
        let verifier_on = BetaCrownVerifier::new(config());
        let on = verifier_on
            .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], threshold, &precomputed)
            .expect("gate-on verify must succeed");
        set_test_gate_override(None);

        assert_eq!(
            std::mem::discriminant(&off.result),
            std::mem::discriminant(&on.result),
            "graph clause learning changed the verdict at threshold {threshold}: \
             OFF={:?} ON={:?}",
            off.result,
            on.result
        );
        // Spot-check the two decided variants exactly: a sound pruner may only
        // close domains an OFF-run certificate already implies safe.
        if matches!(
            off.result,
            BabVerificationStatus::Verified | BabVerificationStatus::PotentialViolation
        ) {
            assert_eq!(off.result, on.result);
        }
        // Pruning can only SKIP bound work, never add it.
        assert!(
            on.domains_explored <= off.domains_explored,
            "gate ON must not explore more domains than OFF at threshold {threshold} \
             (ON={} OFF={})",
            on.domains_explored,
            off.domains_explored
        );
    }
}

/// Multi-objective graph loop: gate ON vs OFF must produce the identical
/// final verdict (disjunctive lane, both objectives must verify).
#[ntest::timeout(30000)]
#[test]
fn test_graph_multi_objective_gate_on_vs_off_identical_verdicts() {
    let graph = super::gpu_bab::simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let config = || BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 200,
        max_depth: 10,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let objectives = vec![vec![1.0f32], vec![-1.0f32]];

    for thresholds in [[-10.0f32, -10.0], [-0.5f32, -0.5]] {
        set_test_gate_override(Some(false));
        let off = BetaCrownVerifier::new(config())
            .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
            .expect("gate-off multi-objective verify must succeed");

        set_test_gate_override(Some(true));
        let on = BetaCrownVerifier::new(config())
            .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
            .expect("gate-on multi-objective verify must succeed");
        set_test_gate_override(None);

        assert_eq!(
            std::mem::discriminant(&off.result),
            std::mem::discriminant(&on.result),
            "multi-objective clause learning changed the verdict at thresholds {thresholds:?}: \
             OFF={:?} ON={:?}",
            off.result,
            on.result
        );
        if matches!(
            off.result,
            BabVerificationStatus::Verified | BabVerificationStatus::PotentialViolation
        ) {
            assert_eq!(off.result, on.result);
        }
    }
}
