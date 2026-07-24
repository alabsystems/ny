// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! AdaIN-specific graph CROWN-IBP gate regressions.

use super::{GraphNetwork, GraphNode};
use crate::layers::{AdaIN1dLayer, InstanceNorm1dLayer, Layer, ReLULayer};

/// AdaIN1d now routes `IbpValidated` CROWN through its effective
/// InstanceNorm1d equivalent, so AdaIN-only CNN graphs are also allowed to use
/// CROWN-IBP intermediates. Part of #3912.
#[ntest::timeout(10000)]
#[test]
fn test_should_use_crown_ibp_intermediates_adain_3912() {
    let mut adain_only = GraphNetwork::new();
    adain_only.add_node(GraphNode::from_input(
        "adain",
        Layer::AdaIN1d(
            AdaIN1dLayer::new_identity_style(
                InstanceNorm1dLayer::new_default(2, 1e-5).expect("invariant: valid test eps"),
            )
            .expect("invariant: valid identity-style AdaIN"),
        ),
    ));
    adain_only.set_output("adain");
    assert!(
        adain_only.should_use_crown_ibp_intermediates(),
        "AdaIN1d-only graph SHOULD use CROWN-IBP intermediates once it routes through effective InstanceNorm (#3912)"
    );

    let mut relu_adain = GraphNetwork::new();
    relu_adain.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    relu_adain.add_node(GraphNode::new(
        "adain",
        Layer::AdaIN1d(
            AdaIN1dLayer::new_identity_style(
                InstanceNorm1dLayer::new_default(2, 1e-5).expect("invariant: valid test eps"),
            )
            .expect("invariant: valid identity-style AdaIN"),
        ),
        vec!["relu".into()],
    ));
    relu_adain.set_output("adain");
    assert!(
        relu_adain.should_use_crown_ibp_intermediates(),
        "ReLU+AdaIN1d graph SHOULD use CROWN-IBP intermediates after the AdaIN routing fix (#3912)"
    );
}

/// #binary-relax-crown-ibp (DARK, `NY_CROWN_IBP_BINARY=1`).
///
/// The binary-relaxation blocklist is a COST guard, not a soundness guard: the
/// CROWN-IBP collector intersects every per-node CROWN bound with that node's
/// IBP bound, so admitting MatMul / MulBinary / BilinearCrown can only tighten
/// the map it would otherwise refuse to compute. The dark lane therefore
/// admits exactly those three ops and nothing else — `GroupNorm` stays blocked
/// in both directions, and the shipped (gate-off) predicate is unchanged.
///
/// Env is read once per process through a `OnceLock`, so this test drives the
/// pure predicate directly rather than mutating the environment.
#[ntest::timeout(10000)]
#[test]
fn crown_ibp_binary_relax_admits_only_binary_relaxation_ops() {
    use crate::layers::binary_ops::{BilinearCrownLayer, MatMulLayer, MulBinaryLayer};
    use crate::layers::GroupNormLayer;

    let binary_layers: Vec<(&str, Layer)> = vec![
        ("mm", Layer::MatMul(MatMulLayer::new(false, None))),
        (
            "bc",
            Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        ),
        ("mul", Layer::MulBinary(MulBinaryLayer)),
    ];
    for (label, layer) in binary_layers {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("r1", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::from_input("r2", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            label,
            layer,
            vec!["r1".to_string(), "r2".to_string()],
        ));
        graph.set_output(label);
        assert!(
            !graph.crown_ibp_intermediates_allowed(false),
            "{label}: shipped blocklist must keep refusing CROWN-IBP intermediates"
        );
        assert!(
            graph.crown_ibp_intermediates_allowed(true),
            "{label}: the dark binary-relax lane must admit CROWN-IBP intermediates"
        );
    }

    // GroupNorm is NOT part of the dark lane: blocked either way.
    let mut gn = GraphNetwork::new();
    gn.add_node(GraphNode::from_input(
        "gn",
        Layer::GroupNorm(
            GroupNormLayer::new_default(2, 1, 1e-5).expect("invariant: valid test eps"),
        ),
    ));
    gn.set_output("gn");
    assert!(!gn.crown_ibp_intermediates_allowed(false));
    assert!(
        !gn.crown_ibp_intermediates_allowed(true),
        "GroupNorm must stay blocked even under the dark binary-relax lane"
    );

    // Gate-off default: the env-reading entry point is the `false` predicate
    // whenever `NY_CROWN_IBP_BINARY` is unset (the test process default).
    let mut relu = GraphNetwork::new();
    relu.add_node(GraphNode::from_input("r", Layer::ReLU(ReLULayer)));
    relu.set_output("r");
    assert_eq!(
        relu.should_use_crown_ibp_intermediates(),
        relu.crown_ibp_intermediates_allowed(
            crate::network::core::graph::crown_ibp_binary_relax_enabled()
        ),
        "the env entry point must agree with the pure predicate it delegates to"
    );
}
