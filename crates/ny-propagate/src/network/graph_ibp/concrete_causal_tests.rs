// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::layers::{CausalSoftmaxLayer, Layer, LinearLayer};
use crate::{GraphNetwork, GraphNode};
use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;
use std::time::Instant;

fn graph(window: Option<usize>) -> GraphNetwork {
    let mut network = GraphNetwork::new();
    let mut softmax = CausalSoftmaxLayer::new(-1);
    softmax.window_size = window;
    network.add_node(GraphNode::from_input(
        "probability",
        Layer::CausalSoftmax(softmax),
    ));
    network.set_output("probability");
    network
}

#[test]
fn graph_point_long_context_remains_nonuniform_beyond_checked_interval_work_cap() {
    let shape = [2, 1024, 1024];
    let values = ArrayD::from_shape_fn(IxDyn(&shape), |i| ((i[2] % 7) as f32 - 3.0) * 0.125);
    let input = BoundedTensor::concrete(values).unwrap();
    let network = graph(None);
    let point = network
        .propagate_concrete_point(&input, None, None)
        .unwrap();
    assert_eq!(point.lower(), point.upper());
    let shift = 0.375_f64;
    let weights: Vec<_> = (0..1024)
        .map(|k| ((((k % 7) as f64) - 3.0) * 0.125 - shift).exp())
        .collect();
    let mass: f64 = weights.iter().sum();
    for h in 0..2 {
        for k in 0..1024 {
            assert_eq!(point.lower()[[h, 1023, k]], (weights[k] / mass) as f32);
        }
        assert!(point.lower()[[h, 1023, 6]] > point.lower()[[h, 1023, 0]]);
        assert_eq!(point.lower()[[h, 0, 0]], 1.0);
        assert_eq!(point.lower()[[h, 0, 1]], 0.0);
    }
    // At this geometry the checked transcendental work exceeds the original
    // ten-billion cap. Its conservative [0,1] fallback stays on interval paths;
    // its midpoint is not the concrete model's probability.
    let plain = network.propagate_ibp(&input).unwrap();
    assert_eq!(plain.lower()[[0, 1023, 0]], 0.0);
    assert_eq!(plain.upper()[[0, 1023, 0]], 1.0);
    for interval in [plain, network.propagate_ibp_sound(&input).unwrap()] {
        // The sound graph adapter may widen the layer's [0,1] endpoints by
        // one outward ULP. Require containment rather than bitwise equality.
        assert!(interval.lower()[[0, 1023, 0]] <= 0.0);
        assert!(interval.upper()[[0, 1023, 0]] >= 1.0);
        assert_ne!(point.lower()[[0, 1023, 0]], interval.center()[[0, 1023, 0]]);
    }
}

#[test]
fn graph_point_causal_masks_bias_and_absolute_deadline_are_preserved() {
    let mut network = graph(Some(1));
    network.add_node(GraphNode::new(
        "biased",
        Layer::Linear(
            LinearLayer::from_flat(vec![1.0, -2.0, 3.0], 1, 3, Some(vec![0.75])).unwrap(),
        ),
        vec!["probability".into()],
    ));
    network.set_output("biased");
    let input = BoundedTensor::concrete(
        ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            vec![0.0, 7.0, -9.0, 0.0, 1.0, 9.0, -8.0, 0.0, 1.0],
        )
        .unwrap(),
    )
    .unwrap();
    let point = network
        .propagate_concrete_point(&input, None, None)
        .unwrap();
    assert_eq!(point.shape(), &[3, 1]);
    assert_eq!(point.lower()[[0, 0]], 1.75);
    let p = (1.0_f64 / (1.0 + (-1.0_f64).exp())) as f32;
    let expected = (1.0 - p) * (-2.0) + p * 3.0 + 0.75;
    assert!((point.lower()[[2, 0]] - expected).abs() < 1e-6);
    assert_eq!(point.lower(), point.upper());
    assert!(matches!(
        network.propagate_concrete_point(&input, None, Some(Instant::now())),
        Err(NyError::DeadlineExceeded(_))
    ));
    let interval = BoundedTensor::new(
        input.lower().mapv(|x| x - 0.1),
        input.upper().mapv(|x| x + 0.1),
    )
    .unwrap();
    let enclosure = graph(Some(1)).propagate_ibp_sound(&interval).unwrap();
    assert!(enclosure.lower()[[2, 2]] < p && p < enclosure.upper()[[2, 2]]);
    assert!(enclosure.lower()[[2, 0]] <= 0.0);
    assert!(enclosure.upper()[[2, 0]] >= 0.0);
    assert!(enclosure.width()[[2, 0]] < f32::EPSILON);
}
