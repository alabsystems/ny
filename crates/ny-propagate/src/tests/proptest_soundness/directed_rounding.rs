// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{Layer, LinearLayer, Network, ReLULayer};
use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::valid_interval;

// =============================================================================
// SOUND PROPAGATION (DIRECTED ROUNDING) TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(100) })]

    /// propagate_ibp_sound should produce bounds at least as wide as propagate_ibp.
#[ntest::timeout(10000)]
    #[test]
    fn sound_propagation_widens_bounds(
        w11 in -3.0f32..3.0,
        w12 in -3.0f32..3.0,
        w21 in -3.0f32..3.0,
        w22 in -3.0f32..3.0,
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let linear = LinearLayer::new(weight, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let normal = network.propagate_ibp(&input).unwrap();
        let sound = network.propagate_ibp_sound(&input).unwrap();

        for i in 0..2 {
            prop_assert!(
                sound.lower()[[i]] <= normal.lower()[[i]],
                "Sound lower bound should be <= normal at {}: {} <= {}",
                i, sound.lower()[[i]], normal.lower()[[i]]
            );
            prop_assert!(
                sound.upper()[[i]] >= normal.upper()[[i]],
                "Sound upper bound should be >= normal at {}: {} >= {}",
                i, sound.upper()[[i]], normal.upper()[[i]]
            );
        }
    }

    /// collect_ibp_bounds_sound should widen every intermediate layer bound.
    #[ntest::timeout(10000)]
    #[test]
    fn sound_collect_bounds_widens_each_layer(
        w11 in -3.0f32..3.0,
        w12 in -3.0f32..3.0,
        w21 in -3.0f32..3.0,
        w22 in -3.0f32..3.0,
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let linear = LinearLayer::new(weight, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        ).unwrap();

        let normal = network.collect_ibp_bounds(&input).unwrap();
        let sound = network.collect_ibp_bounds_sound(&input).unwrap();

        prop_assert_eq!(sound.len(), normal.len());
        for (layer_idx, (sound_layer, normal_layer)) in sound.iter().zip(normal.iter()).enumerate() {
            for i in 0..normal_layer.len() {
                prop_assert!(
                    sound_layer.lower()[[i]] <= normal_layer.lower()[[i]],
                    "Layer {layer_idx} lower should be <= normal at {i}: {} <= {}",
                    sound_layer.lower()[[i]],
                    normal_layer.lower()[[i]],
                );
                prop_assert!(
                    sound_layer.upper()[[i]] >= normal_layer.upper()[[i]],
                    "Layer {layer_idx} upper should be >= normal at {i}: {} >= {}",
                    sound_layer.upper()[[i]],
                    normal_layer.upper()[[i]],
                );
            }
        }
    }

    /// Final layer from collect_ibp_bounds_sound should equal propagate_ibp_sound output.
    #[ntest::timeout(10000)]
    #[test]
    fn sound_collect_bounds_matches_propagate_sound(
        w11 in -3.0f32..3.0,
        w12 in -3.0f32..3.0,
        w21 in -3.0f32..3.0,
        w22 in -3.0f32..3.0,
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let linear = LinearLayer::new(weight, None).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn(),
        ).unwrap();

        let propagated = network.propagate_ibp_sound(&input).unwrap();
        let collected = network.collect_ibp_bounds_sound(&input).unwrap();
        let final_collected = collected.last().unwrap();

        for i in 0..propagated.len() {
            prop_assert_eq!(final_collected.lower()[[i]], propagated.lower()[[i]]);
            prop_assert_eq!(final_collected.upper()[[i]], propagated.upper()[[i]]);
        }
    }
}
