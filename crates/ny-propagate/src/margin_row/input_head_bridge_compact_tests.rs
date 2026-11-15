// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{LinearLayer, ReshapeLayer};
use crate::{Layer, Network};
use ndarray::{arr1, arr2};

fn compact_network(spec: &InputHeadSpec, unfused: bool) -> Network {
    let legacy = scalar_network(spec, unfused);
    let mut compact = Network::new();
    for layer in &legacy.layers()[..legacy.num_layers() - 3] {
        compact.add_layer(layer.clone());
    }
    compact
}

fn replace_diff(network: &mut Network, weights: Array2<f32>, bias: Option<ndarray::Array1<f32>>) {
    let last = network.num_layers() - 1;
    network.layers_mut()[last] = Layer::Linear(LinearLayer::new(weights, bias).unwrap());
}

#[test]
fn compact_and_legacy_singleton_tails_authenticate_the_same_source_objective() {
    let source = unsafe_source();
    let carry = continuation(&source, &[vec![1.0, -0.5]]);
    for unfused in [false, true] {
        let compact = compact_network(&source.spec, unfused);
        let legacy = scalar_network(&source.spec, unfused);
        assert!(carry.matches_scalar_network(&compact));
        assert!(carry.matches_scalar_network(&legacy));
        assert!(!carry.matches_original_network(&compact));
        assert!(
            !carry.dependent_use(),
            "authentication alone is not a Source prune"
        );
        for x in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            let point = ny_tensor::BoundedTensor::concrete(arr1(&[x]).into_dyn()).unwrap();
            let compact_value = compact
                .propagate_concrete_point(&point, None)
                .unwrap()
                .center();
            let legacy_value = legacy
                .propagate_concrete_point(&point, None)
                .unwrap()
                .center();
            assert_eq!(
                compact_value.iter().copied().collect::<Vec<_>>(),
                vec![x + 1.0 / 16.0]
            );
            assert_eq!(
                compact_value.iter().copied().collect::<Vec<_>>(),
                legacy_value.iter().copied().collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn compact_singleton_tail_refuses_wrong_coefficients_bias_and_output_width() {
    let source = unsafe_source();
    let carry = continuation(&source, &[vec![1.0, -0.5]]);
    for unfused in [false, true] {
        let compact = compact_network(&source.spec, unfused);
        for coefficients in [[-1.0_f32, 1.0], [1.0_f32.next_up(), -1.0], [1.0, 0.0]] {
            let mut wrong = compact.clone();
            replace_diff(
                &mut wrong,
                Array2::from_shape_vec((1, 2), coefficients.to_vec()).unwrap(),
                None,
            );
            assert!(!carry.matches_scalar_network(&wrong));
        }
        let mut offset = compact.clone();
        replace_diff(&mut offset, arr2(&[[1.0, -1.0]]), Some(arr1(&[1.0 / 16.0])));
        assert!(!carry.matches_scalar_network(&offset));
        let mut two_rows = compact.clone();
        replace_diff(&mut two_rows, arr2(&[[1.0, -1.0], [1.0, -1.0]]), None);
        assert!(!carry.matches_scalar_network(&two_rows));
        let mut zero_bias = compact;
        replace_diff(&mut zero_bias, arr2(&[[1.0, -1.0]]), Some(arr1(&[0.0])));
        assert!(carry.matches_scalar_network(&zero_bias));
    }
}

#[test]
fn compact_singleton_tail_refuses_extra_operators_and_original_coefficient_mutation() {
    let source = unsafe_source();
    let carry = continuation(&source, &[vec![1.0, -0.5]]);
    for unfused in [false, true] {
        let compact = compact_network(&source.spec, unfused);
        let mut extra = compact.clone();
        extra.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1])));
        assert!(!carry.matches_scalar_network(&extra));
        let mut wrong_original = compact.clone();
        let Layer::Linear(first) = &compact.layers()[0] else {
            panic!("first affine layer");
        };
        let mut weights = first.weight().clone();
        weights[[0, 0]] = weights[[0, 0]].next_up();
        wrong_original.layers_mut()[0] =
            Layer::Linear(LinearLayer::new(weights, first.bias().cloned()).unwrap());
        assert!(!carry.matches_scalar_network(&wrong_original));
    }
}

#[test]
fn compact_singleton_tail_preserves_target_direction_and_unsafe_source_floor() {
    let source = unsafe_source();
    let opposite = Source::build(&source.spec, &[-1.0], &[1.0], 1, 0, deadline()).unwrap();
    let forward = continuation(&source, &[vec![1.0, -0.5]]);
    let reverse = continuation(&opposite, &[vec![-1.0, 0.5]]);
    let mut compact = compact_network(&source.spec, false);
    assert!(forward.matches_scalar_network(&compact));
    assert!(!reverse.matches_scalar_network(&compact));
    replace_diff(&mut compact, arr2(&[[-1.0, 1.0]]), None);
    assert!(reverse.matches_scalar_network(&compact));
    assert!(!forward.matches_scalar_network(&compact));
    // The genuinely violating original endpoint remains below zero. Accepting
    // a different layout cannot change or complete the Source's own proof.
    assert!(
        forward
            .try_support_box(&[-1.0], &[-1.0], deadline())
            .unwrap()
            <= -15.0 / 16.0
    );
    assert!(!forward.has_complete_source_cover(deadline()));
    assert!(forward.try_finish(deadline()).is_none());
}
