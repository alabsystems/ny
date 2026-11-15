// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::layers::Conv2dLayer;
use crate::network::core::GraphNode;
use ndarray::{ArrayD, IxDyn};

fn graph(conv: Conv2dLayer) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.set_output("conv");
    graph
}

fn input(channels: usize, height: usize, width: usize) -> BoundedTensor {
    let shape = IxDyn(&[1, channels, height, width]);
    BoundedTensor::new(
        ArrayD::from_elem(shape.clone(), -1.),
        ArrayD::from_elem(shape, 1.),
    )
    .unwrap()
}

fn strided() -> Conv2dLayer {
    Conv2dLayer::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 3, 3]), 0.125),
        None,
        (2, 2),
        (1, 1),
    )
    .unwrap()
}

#[test]
fn conv_walk_propagated_stride_grid_is_output_and_matches_validated_fallback() {
    let conv = strided();
    assert_eq!(conv.input_shape, None);
    let source = graph(conv.clone());
    let bounds = source.collect_node_bounds(&input(1, 8, 8)).unwrap();
    assert_eq!(bounds["conv"].lower().shape(), &[1, 1, 4, 4]);
    assert!(bounds["conv"].lower().iter().any(|v| *v < 0.));
    assert!(bounds["conv"].upper().iter().any(|v| *v > 0.));
    assert_eq!(source.collector_walk_macs("conv", 8), None);
    let expected = 2 * 8 * 4 * 4 * 9;
    assert_eq!(expected, 2304);
    assert_eq!(
        source.collector_walk_macs_with_shapes("conv", 8, Some(&bounds)),
        Some(expected)
    );
    // The bug applied the same output transform to the propagated 4x4 grid.
    assert_eq!(conv.output_size(4, 4).unwrap(), (2, 2));
    assert_eq!(2 * 8 * 2 * 2 * 9, 576);
    let mut declared = conv.clone();
    declared.set_input_shape(8, 8);
    assert_eq!(
        graph(declared).collector_walk_macs("conv", 8),
        Some(expected)
    );
    // Actual mapped output geometry wins over stale optional input metadata.
    let mut stale = conv;
    stale.set_input_shape(16, 16);
    assert_eq!(
        graph(stale).collector_walk_macs_with_shapes("conv", 8, Some(&bounds)),
        Some(expected)
    );
    assert_eq!(
        source.collector_walk_macs_with_shapes("conv", 0, Some(&bounds)),
        None
    );
}

#[test]
fn conv_walk_propagated_dilation_and_groups_keep_actual_contraction() {
    // Each case uses real forward propagation; no hand-injected output shape.
    for (channels, out_channels, groups, dilation, stride, padding, side, out_side) in [
        (1, 1, 1, (2, 2), (1, 1), (0, 0), 10, 6),
        (4, 4, 2, (1, 1), (2, 2), (1, 1), 8, 4),
        (4, 4, 4, (2, 2), (2, 2), (2, 2), 8, 4),
    ] {
        let mut conv = Conv2dLayer::new_dilated(
            ArrayD::from_elem(IxDyn(&[out_channels, channels / groups, 3, 3]), 0.125),
            None,
            stride,
            padding,
            dilation,
            groups,
        )
        .unwrap();
        let source = graph(conv.clone());
        let bounds = source
            .collect_node_bounds(&input(channels, side, side))
            .unwrap();
        assert_eq!(
            bounds["conv"].lower().shape(),
            &[1, out_channels, out_side, out_side]
        );
        let expected = 2
            * 3
            * out_side as u128
            * out_side as u128
            * out_channels as u128
            * (channels / groups) as u128
            * 9;
        assert_eq!(
            source.collector_walk_macs_with_shapes("conv", 3, Some(&bounds)),
            Some(expected)
        );
        conv.set_input_shape(side, side);
        assert_eq!(graph(conv).collector_walk_macs("conv", 3), Some(expected));
    }
}

#[test]
fn conv_walk_mapped_grid_preserves_geometry_validation_and_unknown_refusal() {
    let conv = strided();
    let source = graph(conv.clone());
    let bounds = source.collect_node_bounds(&input(1, 8, 8)).unwrap();
    for bad in 0..3 {
        let mut invalid = conv.clone();
        match bad {
            0 => invalid.stride = (0, 2),
            1 => invalid.dilation = (0, 1),
            _ => invalid.groups = 0,
        }
        assert_eq!(
            graph(invalid).collector_walk_macs_with_shapes("conv", 8, Some(&bounds)),
            None,
            "a mapped grid must not bypass publicly mutable geometry validation"
        );
    }
    let empty = HashMap::new();
    assert_eq!(
        source.collector_walk_macs_with_shapes("conv", 8, Some(&empty)),
        None
    );
    let rank_two = HashMap::from([(
        "conv".to_string(),
        BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[4, 4]))).unwrap(),
    )]);
    assert_eq!(
        source.collector_walk_macs_with_shapes("conv", 8, Some(&rank_two)),
        None
    );
    let zero_spatial = HashMap::from([(
        "conv".to_string(),
        BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 1, 0, 4]))).unwrap(),
    )]);
    assert_eq!(
        source.collector_walk_macs_with_shapes("conv", 8, Some(&zero_spatial)),
        None
    );
    let mut declared = conv.clone();
    declared.set_input_shape(8, 8);
    assert_eq!(
        graph(declared).collector_walk_macs_with_shapes("conv", 8, Some(&rank_two)),
        Some(2304),
        "missing usable mapped geometry retains the validated input-shape fallback"
    );
    let mut overflow = conv;
    overflow.set_input_shape(usize::MAX, usize::MAX);
    assert_eq!(graph(overflow).collector_walk_macs("conv", 8), None);
}

#[test]
fn conv_walk_corrected_grid_selects_fewer_rows_under_unchanged_fixed_budget() {
    let source = graph(strided());
    let bounds = source.collect_node_bounds(&input(1, 8, 8)).unwrap();
    let model = WalkCostModel::new(1000.);
    let grant = 1.;
    let choose = |old_double_stride: bool| {
        [8, 4, 2, 1].into_iter().find(|&rows| {
            let macs = if old_double_stride {
                2 * rows as u128 * 2 * 2 * 9
            } else {
                source
                    .collector_walk_macs_with_shapes("conv", rows, Some(&bounds))
                    .unwrap()
            };
            matches!(
                admit_walk_with_estimate(model.estimate_secs(macs).unwrap(), grant, grant, false),
                WalkAdmissionDecision::Admit
            )
        })
    };
    assert_eq!(choose(true), Some(8));
    assert_eq!(choose(false), Some(4));
    // This is the real dry admission policy, not a timing claim. It changes
    // neither its rate prior/padding nor the authority supplied to a walk.
    assert_eq!(grant, 1.);
    assert_eq!(model.correction(), WALK_RATE_PRIOR_CORRECTION);
}
