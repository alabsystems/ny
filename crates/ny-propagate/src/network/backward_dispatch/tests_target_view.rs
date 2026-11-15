// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{dispatch_backward_layer, BackwardDispatchResult, DispatchContext, NodeBoundsView};
use crate::bounds::LinearBounds;
use crate::layers::{BatchNormLayer, Layer, LinearLayer, ReLULayer, SubLayer};
use crate::network::core::NETWORK_INPUT;
use crate::MulBinaryRelaxationMode;
use ndarray::{Array1, Array2};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn bounds(radius: f32) -> BoundedTensor {
    BoundedTensor::new(
        Array1::from_vec(vec![-radius, 0.25]).into_dyn(),
        Array1::from_vec(vec![radius, 0.75]).into_dyn(),
    )
    .unwrap()
}

fn historical_projection(
    tightened: &HashMap<String, BoundedTensor>,
    ibp: &HashMap<String, BoundedTensor>,
    node: &str,
    inputs: &[String],
) -> HashMap<String, BoundedTensor> {
    let mut combined = HashMap::new();
    for input in inputs {
        if input == NETWORK_INPUT {
            continue;
        }
        if let Some(b) = tightened.get(input).or_else(|| ibp.get(input)) {
            combined.insert(input.clone(), b.clone());
        }
    }
    if let Some(b) = tightened.get(node).or_else(|| ibp.get(node)) {
        combined.insert(node.to_string(), b.clone());
    }
    combined
}

#[test]
fn target_view_borrows_tightened_first_and_preserves_exact_projection() {
    let tightened = HashMap::from([
        ("a".into(), bounds(0.5)),
        ("out".into(), bounds(1.0)),
        ("unrelated".into(), bounds(100.0)),
        (NETWORK_INPUT.into(), bounds(200.0)),
    ]);
    let ibp = HashMap::from([
        ("a".into(), bounds(2.0)),
        ("b".into(), bounds(3.0)),
        ("out".into(), bounds(4.0)),
        ("other".into(), bounds(500.0)),
    ]);
    let inputs = vec!["a".into(), "b".into(), "a".into(), NETWORK_INPUT.into()];
    let view = NodeBoundsView::target(&tightened, &ibp, "out", &inputs);
    assert!(std::ptr::eq(
        view.get("a").unwrap(),
        &raw const tightened["a"]
    ));
    assert!(std::ptr::eq(view.get("b").unwrap(), &raw const ibp["b"]));
    assert!(std::ptr::eq(
        view.get("out").unwrap(),
        &raw const tightened["out"]
    ));
    let old = historical_projection(&tightened, &ibp, "out", &inputs);
    for name in [
        "a",
        "b",
        "out",
        "unrelated",
        "other",
        NETWORK_INPUT,
        "missing",
    ] {
        assert_eq!(view.get(name).is_some(), old.contains_key(name), "{name}");
        if let (Some(actual), Some(expected)) = (view.get(name), old.get(name)) {
            assert_eq!(actual.lower(), expected.lower());
            assert_eq!(actual.upper(), expected.upper());
        }
    }
}

#[test]
fn target_view_missing_entries_aliases_and_domains_keep_original_lookup_contract() {
    let empty = HashMap::new();
    let first = HashMap::from([("x".into(), bounds(1.0))]);
    let second = HashMap::from([("x".into(), bounds(7.0))]);
    for inputs in [vec![], vec!["x".into()], vec!["missing".into()]] {
        for node in ["x", "missing", NETWORK_INPUT] {
            let view = NodeBoundsView::target(&empty, &first, node, &inputs);
            let old = historical_projection(&empty, &first, node, &inputs);
            for name in ["x", "missing", NETWORK_INPUT] {
                assert_eq!(view.get(name).is_some(), old.contains_key(name));
            }
        }
    }
    let inputs = vec!["x".into()];
    let a = NodeBoundsView::target(&empty, &first, "out", &inputs);
    let b = NodeBoundsView::target(&empty, &second, "out", &inputs);
    assert!(std::ptr::eq(a.get("x").unwrap(), &raw const first["x"]));
    assert!(std::ptr::eq(b.get("x").unwrap(), &raw const second["x"]));
    assert_ne!(a.get("x").unwrap().upper(), b.get("x").unwrap().upper());
}

fn same_linear(a: &LinearBounds, b: &LinearBounds) {
    let bits = |xs: Vec<f32>| xs.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for (x, y) in [
        (
            a.lower_a().iter().copied().collect(),
            b.lower_a().iter().copied().collect(),
        ),
        (
            a.upper_a().iter().copied().collect(),
            b.upper_a().iter().copied().collect(),
        ),
        (
            a.lower_b().iter().copied().collect(),
            b.lower_b().iter().copied().collect(),
        ),
        (
            a.upper_b().iter().copied().collect(),
            b.upper_b().iter().copied().collect(),
        ),
    ] {
        assert_eq!(bits(x), bits(y));
    }
    for (x, y) in [
        (a.lower_a_err(), b.lower_a_err()),
        (a.upper_a_err(), b.upper_a_err()),
    ] {
        assert_eq!(x.is_some(), y.is_some());
        if let (Some(x), Some(y)) = (x, y) {
            assert_eq!(
                bits(x.iter().copied().collect()),
                bits(y.iter().copied().collect())
            );
        }
    }
}

fn same_dispatch(a: BackwardDispatchResult, b: BackwardDispatchResult) {
    match (a, b) {
        (BackwardDispatchResult::Single(a), BackwardDispatchResult::Single(b)) => {
            same_linear(&a, &b)
        }
        (
            BackwardDispatchResult::Binary {
                bounds_a: a1,
                bounds_b: a2,
                bias_lower: al,
                bias_upper: au,
            },
            BackwardDispatchResult::Binary {
                bounds_a: b1,
                bounds_b: b2,
                bias_lower: bl,
                bias_upper: bu,
            },
        ) => {
            same_linear(&a1, &b1);
            same_linear(&a2, &b2);
            assert_eq!(al.mapv(f32::to_bits), bl.mapv(f32::to_bits));
            assert_eq!(au.mapv(f32::to_bits), bu.mapv(f32::to_bits));
        }
        (a, b) => panic!("unexpected dispatch pair: {a:?} {b:?}"),
    }
}

#[test]
fn target_view_dispatch_matches_owned_projection_with_certified_error_channels() {
    let input = bounds(2.0);
    let tightened = HashMap::from([("a".into(), bounds(0.5)), ("out".into(), bounds(0.75))]);
    let ibp = HashMap::from([
        ("a".into(), bounds(4.0)),
        ("b".into(), bounds(1.0)),
        ("out".into(), bounds(6.0)),
    ]);
    let layers = [
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((2, 2), vec![0.7, -0.3, 0.2, 1.1]).unwrap(),
                Some(Array1::from_vec(vec![0.1, -0.2])),
            )
            .unwrap(),
        ),
        Layer::ReLU(ReLULayer),
        Layer::BatchNorm(
            BatchNormLayer::from_scale_bias(
                Array1::from_vec(vec![1.2, -0.7]).into_dyn(),
                Array1::from_vec(vec![0.05, -0.15]).into_dyn(),
            )
            .unwrap(),
        ),
        Layer::Sub(SubLayer),
    ];
    let mut supported_pairs = 0;
    let mut caller_owned_relu_pairs = 0;
    for layer in &layers {
        let inputs = if matches!(layer, Layer::Sub(_)) {
            vec!["a".into(), "b".into()]
        } else {
            vec!["a".into()]
        };
        let old = historical_projection(&tightened, &ibp, "out", &inputs);
        for deadline in [None, Some(Instant::now() + Duration::from_secs(30))] {
            for errors in [false, true] {
                let mut seed = LinearBounds::identity(2);
                if errors {
                    seed.lower_a_err = Some(Array2::from_elem((2, 2), 0.00001));
                    seed.upper_a_err = Some(Array2::from_elem((2, 2), 0.00002));
                }
                let ctx = |node_bounds| DispatchContext {
                    node_name: "out",
                    layer,
                    inputs: &inputs,
                    pre_activation: &tightened["a"],
                    network_input: &input,
                    node_bounds,
                    engine: None,
                    deadline,
                    bilinear_alphas: None,
                    mul_binary_relaxation: MulBinaryRelaxationMode::default(),
                    mul_binary_alphas: None,
                    norm_inv_rms_override: None,
                };
                let a = dispatch_backward_layer(&ctx((&old).into()), &seed).unwrap();
                let b = dispatch_backward_layer(
                    &ctx(NodeBoundsView::target(&tightened, &ibp, "out", &inputs)),
                    &seed,
                )
                .unwrap();
                if matches!(layer, Layer::ReLU(_)) {
                    // The real target-backward caller handles ReLU before
                    // entering this shared dispatcher. Keep every ReLU case
                    // as an exact refusal control rather than requiring a
                    // bound from an intentionally unsupported entry point.
                    let expected = "ReLU must be handled by caller (requires alpha/beta state)";
                    match (a, b) {
                        (
                            BackwardDispatchResult::Unsupported(a),
                            BackwardDispatchResult::Unsupported(b),
                        ) => {
                            assert_eq!(a, expected);
                            assert_eq!(b, expected);
                        }
                        (a, b) => panic!("ReLU caller contract changed: {a:?} {b:?}"),
                    }
                    caller_owned_relu_pairs += 1;
                } else {
                    same_dispatch(a, b);
                    supported_pairs += 1;
                }
            }
        }
    }
    // A blanket Unsupported comparison must never make this fixture green.
    assert_eq!(supported_pairs, 12);
    assert_eq!(caller_owned_relu_pairs, 4);
}
