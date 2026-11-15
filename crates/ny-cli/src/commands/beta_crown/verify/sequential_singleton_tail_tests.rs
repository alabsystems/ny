// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{arr1, arr2, ArrayD};
use ny_propagate::layers::ReLULayer;

fn concrete(network: &Network, point: ArrayD<f32>) -> Vec<f32> {
    network
        .propagate_concrete_point(&BoundedTensor::concrete(point).unwrap(), None)
        .unwrap()
        .center()
        .iter()
        .copied()
        .collect()
}

fn hinge_network() -> Network {
    let mut network = Network::new();
    network.add_layer(PropLayer::Linear(
        LinearLayer::new(arr2(&[[1.0], [-1.0]]), None).unwrap(),
    ));
    network.add_layer(PropLayer::ReLU(ReLULayer));
    network.add_layer(PropLayer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0], [0.0, 0.0]]), Some(arr1(&[0.25, 0.0]))).unwrap(),
    ));
    network
}

#[test]
fn singleton_reduction_preserves_relu_alpha_eligibility_and_exact_objective() {
    let original = hinge_network();
    let (signed, compact) = reduced_objective_networks(&original, arr2(&[[1.0, -1.0]])).unwrap();
    assert_eq!(compact.num_layers(), original.num_layers() + 1);
    assert_eq!(signed.num_layers(), compact.num_layers());
    assert!(BetaCrownConfig::default().use_alpha_crown);
    assert_eq!(
        compact
            .layers()
            .iter()
            .filter(|l| matches!(l, PropLayer::ReLU(_)))
            .count(),
        1
    );
    // These are the exact unsupported layer families tested before sequential
    // alpha-CROWN enters its optimizer. The reduction must not introduce one.
    assert!(!compact.layers().iter().any(|l| matches!(
        l,
        PropLayer::Conv2d(_) | PropLayer::ConvTranspose2d(_) | PropLayer::MaxPool2d(_)
    )));
    let mut legacy = signed;
    legacy.add_layer(PropLayer::Reshape(ReshapeLayer::new(vec![1, 1, 1])));
    legacy.add_layer(PropLayer::MaxPool2d(MaxPool2dLayer::new(
        (1, 1),
        (1, 1),
        (0, 0),
    )));
    legacy.add_layer(PropLayer::Reshape(ReshapeLayer::new(vec![1])));
    for x in [-1.0_f32, -0.25, 0.0, 0.5, 1.0] {
        let point = arr1(&[x]).into_dyn();
        let expected = x.abs() + 0.25;
        assert_eq!(concrete(&compact, point.clone()), vec![expected]);
        assert_eq!(concrete(&legacy, point), vec![expected]);
    }
}

#[test]
fn multi_rhs_reduction_retains_exact_maximum_layout_and_joint_semantics() {
    let mut original = Network::new();
    original.add_layer(PropLayer::Linear(
        LinearLayer::new(
            arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            Some(arr1(&[0.0, 0.0, 0.5])),
        )
        .unwrap(),
    ));
    let (signed, maximum) =
        reduced_objective_networks(&original, arr2(&[[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])).unwrap();
    assert_eq!(maximum.num_layers(), signed.num_layers() + 3);
    let tail = &maximum.layers()[signed.num_layers()..];
    let [PropLayer::Reshape(pre), PropLayer::MaxPool2d(pool), PropLayer::Reshape(post)] = tail
    else {
        panic!("multi-RHS reduction must retain its exact maximum tail");
    };
    assert_eq!(pre.target_shape, [1, 1, 2]);
    assert_eq!(post.target_shape, [1]);
    assert_eq!(pool.kernel_size, (1, 2));
    assert_eq!(pool.stride, (1, 2));
    assert_eq!(pool.padding, (0, 0));
    for (x, y) in [(-1.0_f32, 1.0_f32), (1.0, -1.0), (0.25, 0.75)] {
        let point = arr1(&[x, y]).into_dyn();
        let expected = vec![x - y, x - 0.5];
        assert_eq!(concrete(&signed, point.clone()), expected);
        assert_eq!(
            concrete(&maximum, point),
            vec![expected[0].max(expected[1])]
        );
    }
}

#[test]
fn singleton_reduction_keeps_both_relational_directions_and_rejects_empty_rows() {
    let original = hinge_network();
    let (_, positive) = reduced_objective_networks(&original, arr2(&[[1.0, -1.0]])).unwrap();
    let (_, negative) = reduced_objective_networks(&original, arr2(&[[-1.0, 1.0]])).unwrap();
    for x in [-1.0_f32, 0.0, 1.0] {
        let point = arr1(&[x]).into_dyn();
        assert_eq!(concrete(&positive, point.clone()), vec![x.abs() + 0.25]);
        assert_eq!(concrete(&negative, point), vec![-(x.abs() + 0.25)]);
    }
    assert!(reduced_objective_networks(&original, Array2::zeros((0, 2))).is_err());
}
