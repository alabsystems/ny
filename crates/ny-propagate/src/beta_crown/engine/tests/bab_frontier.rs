// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Engine-level #bab-frontier oracle (docs/BAB_FRONTIER_SEEDING_DESIGN.md):
//! a tiny input-split net driven into the DOMAIN-LIMIT exhaust exit must
//! export its surviving-unverified queue as attack seeds — subboxes of the
//! root box at depth > 0, centers inside the root box — and export NOTHING
//! when the `NY_POSTBAB_BAB_SEEDS` gate is off (the default).

use std::time::Duration;

use ndarray::arr1;
use ny_tensor::BoundedTensor;

use super::simple_network;
use crate::beta_crown::{
    reset_bab_frontier_export, take_bab_frontier_seeds, BetaCrownConfig, BetaCrownVerifier,
    BranchingHeuristic,
};
use crate::GraphNetwork;

/// `simple_network()` computes `|x0 - x1|` (min 0 on the diagonal), so
/// threshold 0.0 can never verify on any diagonal-straddling subbox: the BaB
/// loop keeps splitting until `max_domains` forces the domain-limit exit with
/// a non-empty surviving queue of input-split children.
fn run_exhausting_input_split_verify() {
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 4,
        batch_size: 1,
        timeout: Duration::from_secs(30),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    // The verdict itself is not under test (Unknown{domain limit} or a PGD
    // fallback outcome); the frontier recording fires at the exhaust exit
    // regardless, strictly before the fallback.
    let _ = verifier.verify(&network, &input, 0.0).unwrap();
}

#[ntest::timeout(60000)]
#[test]
fn domain_limit_gate_accepts_only_exact_modes_1_and_2() {
    crate::tests::with_env_edits(|env| {
        // The producer and consumer both expose exactly modes 1/2. Mode 3 is
        // deliberately retired; whitespace is not normalized for the mode.
        let cases = [
            (None, false),
            (Some("0"), false),
            (Some("1"), true),
            (Some("2"), true),
            (Some("3"), false),
            (Some(" 1"), false),
            (Some("1 "), false),
            (Some("bad"), false),
        ];
        env.remove("NY_POSTBAB_BAB_SEEDS_K");
        let mut mode_1_seeds = None;
        for (raw, expected_live) in cases {
            match raw {
                Some(value) => env.set("NY_POSTBAB_BAB_SEEDS", value),
                None => env.remove("NY_POSTBAB_BAB_SEEDS"),
            }

            reset_bab_frontier_export();
            run_exhausting_input_split_verify();
            let seeds = take_bab_frontier_seeds();
            assert_eq!(
                !seeds.is_empty(),
                expected_live,
                "raw mode {raw:?} must {} the producer",
                if expected_live { "arm" } else { "not arm" }
            );
            if raw == Some("2") {
                assert!(
                    seeds.iter().all(|s| s.corners.is_empty()),
                    "mode 2 with no JointMarginCloser attached carries no corner payload"
                );
            }
            if raw == Some("1") {
                mode_1_seeds = Some(seeds);
            }
        }

        // Inspect one live run in detail: the surviving queue at the domain-limit
        // exit is exported in violation-priority order as valid input subboxes.
        let seeds = mode_1_seeds.expect("exact mode 1 must export a frontier");
        reset_bab_frontier_export();
        for (i, seed) in seeds.iter().enumerate() {
            assert_eq!(seed.center.len(), 2, "seed {i}: original X-space arity");
            assert_eq!(seed.box_lo.len(), 2);
            assert_eq!(seed.box_hi.len(), 2);
            assert!(
                seed.depth > 0,
                "seed {i}: only input-split children survive the explored root"
            );
            for d in 0..2 {
                // Every exported box is a subbox of the root box...
                assert!(
                    seed.box_lo[d] >= -1.0 && seed.box_hi[d] <= 1.0,
                    "seed {i} dim {d}: [{}, {}] escapes the root box",
                    seed.box_lo[d],
                    seed.box_hi[d]
                );
                // ...and every center lies inside its own box (hence the root box).
                assert!(
                    seed.box_lo[d] <= seed.center[d] && seed.center[d] <= seed.box_hi[d],
                    "seed {i} dim {d}: center {} outside [{}, {}]",
                    seed.center[d],
                    seed.box_lo[d],
                    seed.box_hi[d]
                );
            }
            assert!(seed.margin.is_finite(), "seed {i}: finite margin");
        }
        // Export order is most-violation-first (priority DESC); this run is
        // the lower-bound direction (`verify_upper_bound=false`), where
        // priority == -lower_bound, so margins come out most-NEGATIVE first.
        for pair in seeds.windows(2) {
            assert!(
                pair[0].margin <= pair[1].margin,
                "frontier must be sorted most-violation-first ({} > {})",
                pair[0].margin,
                pair[1].margin
            );
        }
    });
}

/// GRAPH-lane variant of the exhaust driver: the same `|x0 - x1|` net routed
/// through the GRAPH/GPU BaB DomainList engine (`verify_graph_gpu_domain_list`)
/// in input-split mode, driven into the in-loop `check_termination`
/// domain-limit exit with a non-empty surviving DomainList of bisected
/// children.
fn run_exhausting_graph_gpu_input_split_verify() {
    let network = simple_network();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 4,
        batch_size: 1,
        timeout: Duration::from_secs(30),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    // The verdict itself is not under test (Unknown{domain limit}); the
    // frontier recording fires at the graph-lane exhaust exit regardless.
    let _ = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &[1.0], 0.0, None, None)
        .unwrap();
}

/// #bab-frontier graph lane oracle: a graph-lane (DomainList GPU BaB engine)
/// exhaust must export its surviving input-split frontier — in-box subboxes of
/// the root at depth > 0 — when the gate is on, and NOTHING when the gate is
/// off (the byte-identical default).
#[ntest::timeout(60000)]
#[test]
fn graph_lane_exhaust_exports_in_box_frontier_when_gated_on_and_nothing_when_off() {
    crate::tests::with_serialized_env_vars(&[("NY_POSTBAB_BAB_SEEDS", "0")], || {
        // Gate OFF (explicit '0' = the unset default): no recording.
        reset_bab_frontier_export();
        run_exhausting_graph_gpu_input_split_verify();
        assert!(
            take_bab_frontier_seeds().is_empty(),
            "gate off must record nothing (byte-identical pipeline)"
        );
    });

    crate::tests::with_serialized_env_vars(&[("NY_POSTBAB_BAB_SEEDS", "1")], || {
        // Gate ON: the surviving DomainList at the domain-limit exit exports.
        reset_bab_frontier_export();
        run_exhausting_graph_gpu_input_split_verify();
        let seeds = take_bab_frontier_seeds();
        assert!(
            !seeds.is_empty(),
            "graph-lane exhaustion with surviving unverified input-split domains must export a frontier"
        );
        for (i, seed) in seeds.iter().enumerate() {
            assert_eq!(seed.center.len(), 2, "seed {i}: original X-space arity");
            assert_eq!(seed.box_lo.len(), 2);
            assert_eq!(seed.box_hi.len(), 2);
            assert!(
                seed.depth > 0,
                "seed {i}: only input-split children own a subbox (root is skipped)"
            );
            let mut is_strict_subbox = false;
            for d in 0..2 {
                // Every exported box is a subbox of the root box...
                assert!(
                    seed.box_lo[d] >= -1.0 && seed.box_hi[d] <= 1.0,
                    "seed {i} dim {d}: [{}, {}] escapes the root box",
                    seed.box_lo[d],
                    seed.box_hi[d]
                );
                // ...and every center lies inside its own box (hence the root box).
                assert!(
                    seed.box_lo[d] <= seed.center[d] && seed.center[d] <= seed.box_hi[d],
                    "seed {i} dim {d}: center {} outside [{}, {}]",
                    seed.center[d],
                    seed.box_lo[d],
                    seed.box_hi[d]
                );
                if seed.box_lo[d] > -1.0 || seed.box_hi[d] < 1.0 {
                    is_strict_subbox = true;
                }
            }
            assert!(
                is_strict_subbox,
                "seed {i}: whole-root-box domains must be skipped, not exported"
            );
            assert!(seed.margin.is_finite(), "seed {i}: finite margin");
        }
        // Export order is most-violation-first (priority DESC); this run is
        // the lower-bound direction (`verify_upper_bound=false`), where
        // priority == -lower_bound, so margins come out most-NEGATIVE first.
        for pair in seeds.windows(2) {
            assert!(
                pair[0].margin <= pair[1].margin,
                "graph frontier must be sorted most-violation-first ({} > {})",
                pair[0].margin,
                pair[1].margin
            );
        }
    });
}
