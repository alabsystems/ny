// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::streaming::*;

fn constant_bounds(dim: usize, lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[dim]), lower),
        ArrayD::from_elem(ndarray::IxDyn(&[dim]), upper),
    )
    .unwrap()
}

fn assert_uniform_bounds(bounds: &BoundedTensor, lower: f32, upper: f32) {
    for (&actual_lower, &actual_upper) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert_eq!(
            actual_lower, lower,
            "expected uniform lower bound {lower}, got {actual_lower}"
        );
        assert_eq!(
            actual_upper, upper,
            "expected uniform upper bound {upper}, got {actual_upper}"
        );
    }
}

fn diagnostic_point_layer() -> Layer {
    use crate::certify::work::WorkBudget;
    use crate::layers::{PointLinearBuildPolicy, PointLinearLayer};
    use std::time::Duration;

    let budget = WorkBudget::new(None, 1 << 20, 1 << 20, 1 << 20, 16, Duration::ZERO)
        .expect("bounded original-weight fixture");
    let weights = Array2::eye(1);
    Layer::PointLinear(
        PointLinearLayer::from_original_view(
            weights.view(),
            false,
            None,
            PointLinearBuildPolicy {
                budget: &budget,
                deadline: None,
                max_resident_bytes: 1 << 20,
                max_scratch_bytes: 1 << 20,
                max_work: 1 << 20,
            },
        )
        .expect("diagnostic original-weight fixture"),
    )
}

fn diagnostic_refusal<T>(result: ny_core::Result<T>) {
    match result {
        Err(ny_core::NyError::UnsupportedOp(message)) => {
            assert!(message.contains("PointLinear") && message.contains("point-only"));
        }
        Err(error) => panic!("expected typed diagnostic proof refusal, got {error}"),
        Ok(_) => panic!("diagnostic storage must not acquire checkpoint proof authority"),
    }
}

#[test]
fn proof_point_only_cached_streaming_checkpoint_refuses_before_exact_return() {
    let compact = diagnostic_point_layer();
    let input = create_input(1);
    let cached = constant_bounds(1, 123.0, 123.0);
    for compressed in [false, true] {
        let mut checkpointed = if compressed {
            CheckpointedBounds::new_compressed(input.clone(), 1, 0.0)
        } else {
            CheckpointedBounds::new(input.clone(), 1)
        };
        checkpointed.add_checkpoint(0, cached.clone());
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(Array2::eye(1), None).unwrap(),
        ));
        // The exact-index shortcut really is populated, and ordinary storage
        // retains its original behavior in both precision modes.
        let ordinary = checkpointed.bounds_at(0, &network).unwrap();
        assert_uniform_bounds(&ordinary, 123.0, 123.0);

        // Current-layer authority is checked after public network mutation,
        // before exact cached output or even index validation can return.
        network.layers[0] = compact.clone();
        diagnostic_refusal(checkpointed.bounds_at(0, &network));
        diagnostic_refusal(checkpointed.bounds_at(usize::MAX, &network));

        // An eligible prefix cannot hide a diagnostic suffix behind an exact
        // prefix checkpoint. The guard scans the current complete network.
        network.layers[0] = Layer::Linear(LinearLayer::new(Array2::eye(1), None).unwrap());
        network.add_layer(compact.clone());
        diagnostic_refusal(checkpointed.bounds_at(0, &network));
    }
}

#[test]
fn proof_point_only_streaming_collector_refuses_before_checkpoint_storage() {
    let mut diagnostic = Network::new();
    diagnostic.add_layer(diagnostic_point_layer());
    let input = create_input(1);
    for compressed in [false, true] {
        let verifier = StreamingVerifier::new(StreamingConfig {
            checkpoint_interval: 1,
            use_f16_checkpoints: compressed,
            ..Default::default()
        });
        diagnostic_refusal(verifier.collect_checkpointed_bounds(&diagnostic, &input));
        diagnostic_refusal(verifier.propagate_crown_streaming(&diagnostic, &input));
        diagnostic_refusal(verifier.propagate_crown_batched_streaming(&diagnostic, &input));

        let mut ordinary = Network::new();
        ordinary.add_layer(Layer::Linear(
            LinearLayer::new(Array2::eye(1), None).unwrap(),
        ));
        let checkpoints = verifier
            .collect_checkpointed_bounds(&ordinary, &input)
            .unwrap();
        assert_eq!(checkpoints.num_checkpoints(), 1);
        let bounds = checkpoints.bounds_at(0, &ordinary).unwrap();
        assert!(bounds.lower()[0] <= -1.0 && bounds.upper()[0] >= 1.0);
        let crown = verifier
            .propagate_crown_streaming(&ordinary, &input)
            .unwrap();
        assert!(crown.lower()[0] <= -1.0 && crown.upper()[0] >= 1.0);
    }
}

fn assert_endpoint_bits_match(actual: &BoundedTensor, expected: &BoundedTensor, layer: usize) {
    assert_eq!(actual.shape(), expected.shape());
    for (&got, &want) in actual.lower().iter().zip(expected.lower().iter()) {
        assert_eq!(got.to_bits(), want.to_bits(), "lower at layer {layer}");
    }
    for (&got, &want) in actual.upper().iter().zip(expected.upper().iter()) {
        assert_eq!(got.to_bits(), want.to_bits(), "upper at layer {layer}");
    }
}

#[test]
fn streaming_checkpoint_recomputation_matches_collected_forward_policy() {
    use crate::layers::ReLULayer;

    let mut network = Network::new();
    for (weights, bias) in [
        (vec![0.75, -0.5, -0.25, 1.25], vec![0.125, -0.375]),
        (vec![-1.0, 0.25, 0.5, 0.875], vec![0.25, -0.0625]),
        (vec![0.5, 0.5, -0.125, 0.25], vec![-0.03125, 0.25]),
    ] {
        network.add_layer(Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((2, 2), weights).unwrap(),
                Some(Array1::from_vec(bias)),
            )
            .unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
    }
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            Array2::from_shape_vec((1, 2), vec![-1.0, 0.5]).unwrap(),
            Some(Array1::from_vec(vec![0.125])),
        )
        .unwrap(),
    ));
    let input = BoundedTensor::new(
        Array1::from_vec(vec![-0.75, -0.5]).into_dyn(),
        Array1::from_vec(vec![0.5, 1.0]).into_dyn(),
    )
    .unwrap();
    let collect = |interval| {
        StreamingVerifier::new(StreamingConfig {
            checkpoint_interval: interval,
            ..Default::default()
        })
        .collect_checkpointed_bounds(&network, &input)
        .unwrap()
    };
    let dense = collect(1);
    let sparse = collect(3);
    // These lookups exercise both recomputation from input and from an exact
    // intermediate checkpoint, with genuine signed, nonzero coefficients.
    assert_eq!(sparse.find_nearest_checkpoint_before(1).unwrap().0, -1);
    assert_eq!(sparse.find_nearest_checkpoint_before(4).unwrap().0, 2);
    for layer in 0..network.layers.len() {
        assert_endpoint_bits_match(
            &sparse.bounds_at(layer, &network).unwrap(),
            &dense.bounds_at(layer, &network).unwrap(),
            layer,
        );
    }
}

fn exact_rational(value: f32) -> num_rational::BigRational {
    num_rational::BigRational::from_float(f64::from(value)).unwrap()
}

/// Independent exact arithmetic on the original affine coefficients and ReLU
/// values. No IBP, CROWN, directed rounding, or floating point reduction is
/// shared with the forward enclosure being checked.
fn original_rational_outputs(
    network: &Network,
    point: &[f32],
) -> Vec<Vec<num_rational::BigRational>> {
    use num_traits::Zero;

    let mut current: Vec<_> = point.iter().copied().map(exact_rational).collect();
    let mut outputs = Vec::new();
    for layer in &network.layers {
        current = match layer {
            Layer::Linear(linear) => (0..linear.out_features())
                .map(|row| {
                    let mut value = linear
                        .bias()
                        .map_or_else(num_rational::BigRational::zero, |bias| {
                            exact_rational(bias[row])
                        });
                    for (column, x) in current.iter().enumerate() {
                        value += exact_rational(linear.weight()[[row, column]]) * x;
                    }
                    value
                })
                .collect(),
            Layer::ReLU(_) => current
                .into_iter()
                .map(|value| value.max(num_rational::BigRational::zero()))
                .collect(),
            _ => panic!("oracle fixture supports only original affine and ReLU layers"),
        };
        outputs.push(current.clone());
    }
    outputs
}

fn assert_contains_rational(bounds: &BoundedTensor, exact: &[num_rational::BigRational]) {
    assert_eq!(bounds.len(), exact.len());
    for ((&lower, &upper), value) in bounds.lower().iter().zip(bounds.upper().iter()).zip(exact) {
        assert!(
            exact_rational(lower) <= *value,
            "lower excludes original exact value"
        );
        assert!(
            exact_rational(upper) >= *value,
            "upper excludes original exact value"
        );
    }
}

#[test]
fn streaming_checkpoint_near_zero_relu_cancellation_has_rational_enclosures() {
    use crate::layers::ReLULayer;

    let large = 2f32.powi(30);
    let small = 2f32.powi(-30);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            Array2::from_shape_vec((1, 3), vec![large, 1.0, -large]).unwrap(),
            None,
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            Array2::from_shape_vec((1, 1), vec![1024.0]).unwrap(),
            Some(Array1::from_vec(vec![-2f32.powi(-21)])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            Array2::from_shape_vec((1, 1), vec![-0.5]).unwrap(),
            Some(Array1::from_vec(vec![2f32.powi(-22)])),
        )
        .unwrap(),
    ));
    let input = BoundedTensor::new(
        Array1::from_vec(vec![1.0, -small, 1.0]).into_dyn(),
        Array1::from_vec(vec![1.0, small, 1.0]).into_dyn(),
    )
    .unwrap();
    // The naive f64 reduction loses the exact positive term before ReLU.
    assert_eq!(
        (f64::from(large) + f64::from(small)) - f64::from(large),
        0.0
    );
    for compressed in [false, true] {
        let checkpoints = StreamingVerifier::new(StreamingConfig {
            checkpoint_interval: 2,
            use_f16_checkpoints: compressed,
            f16_widening_epsilon: 0.0,
        })
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();
        for middle in [-small, 0.0, small] {
            let original = original_rational_outputs(&network, &[1.0, middle, 1.0]);
            for (layer, exact) in original.iter().enumerate() {
                assert_contains_rational(&checkpoints.bounds_at(layer, &network).unwrap(), exact);
            }
        }
    }
}

#[test]
fn streaming_checkpoint_average_pool_uses_checked_accumulation_for_collection_and_replay() {
    use crate::layers::{AveragePoolLayer, BoundPropagation, ReLULayer};
    use num_bigint::BigInt;
    use num_traits::Zero;

    let large = 2f32.powi(30);
    let small = 2f32.powi(-30);
    let pool = AveragePoolLayer::new((1, 3), (1, 3), (0, 0), false);
    let mut network = Network::new();
    network.add_layer(Layer::AveragePool(pool.clone()));
    network.add_layer(Layer::ReLU(ReLULayer));
    // Opposite cancellation signs in separate batches, plus an ordinary
    // nonzero batch. No floating point averaging is used by the oracle.
    let values = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[3, 1, 1, 3]),
        vec![large, small, -large, large, -small, -large, 1.0, 2.0, 3.0],
    )
    .unwrap();
    let input = BoundedTensor::concrete(values).unwrap();
    let third = num_rational::BigRational::from_integer(BigInt::from(3));
    let exact_pool = vec![
        exact_rational(small) / &third,
        -exact_rational(small) / &third,
        exact_rational(2.0),
    ];
    let raw = pool.propagate_ibp(&input).unwrap();
    // Liveness: the raw sum plus even the old driver's one-ULP margin cannot
    // cover this value. The shared route must select the checked pool method.
    let mut old_margin = raw;
    old_margin.round_for_soundness_n_ulps_inplace(1);
    assert!(exact_rational(old_margin.upper()[[0, 0, 0, 0]]) < exact_pool[0]);
    assert!(exact_rational(old_margin.lower()[[1, 0, 0, 0]]) > exact_pool[1]);
    let exact_relu: Vec<_> = exact_pool
        .iter()
        .cloned()
        .map(|value| value.max(num_rational::BigRational::zero()))
        .collect();
    let collect = |interval| {
        StreamingVerifier::new(StreamingConfig {
            checkpoint_interval: interval,
            ..Default::default()
        })
        .collect_checkpointed_bounds(&network, &input)
        .unwrap()
    };
    let dense = collect(1);
    let sparse = collect(3);
    assert_eq!(sparse.find_nearest_checkpoint_before(0).unwrap().0, -1);
    for (layer, exact) in [exact_pool, exact_relu].iter().enumerate() {
        let stored = dense.bounds_at(layer, &network).unwrap();
        let recomputed = sparse.bounds_at(layer, &network).unwrap();
        assert_contains_rational(&stored, exact);
        assert_contains_rational(&recomputed, exact);
        assert_endpoint_bits_match(&recomputed, &stored, layer);
    }
}

/// Like `assert_uniform_bounds` but tolerates the near-zero OUTWARD floor that sound
/// IBP propagation publishes, and ONLY in the outward direction.
///
/// Use for bounds that come from network propagation rather than hand-constructed
/// fixtures.
///
/// # Why the tolerance is `f32::MIN_POSITIVE`-scaled and no longer ~1e-45
///
/// (#linear-ibp-higham) `linear::ibp::propagate_ibp_1d` now charges the Higham
/// residual `up(γ^f64_{n+2}·S_safe + 2u₆₄·|y|)` before narrowing back to f32. These
/// fixtures use `create_test_network`, whose weights and biases are all zero, so
/// `S = 0` and `y = 0` and the charge itself is one f64 subnormal. The visible ~1e-38
/// comes from the guarded narrowing instead: `linear_f64_to_f32_down`/`_up` refuse to
/// PUBLISH a subnormal f32 endpoint and flush every `|v| < f32::MIN_POSITIVE` outward
/// to `∓f32::MIN_POSITIVE`. That is a fixed point — a zero-weight layer maps
/// `[∓MIN_POSITIVE]` back to itself — so depth does not grow it.
///
/// `collect_checkpointed_bounds` then adds its own `in_features + 2` ULP margin after
/// every layer, and these fixtures are 8 wide. The same forward step is used
/// for collection and recomputation, so both call shapes carry that margin:
///
/// - a STORED checkpoint carries that margin and sits at
///   `∓f32::from_bits(f32::MIN_POSITIVE.to_bits() + 10)` = ∓1.1754958e-38;
/// - a RECOMPUTED box carries the identical margin and endpoints.
///
/// Both shapes are asserted through this helper, so the cap is the larger of the two.
/// Before the certificate landed the narrowing was a bare `next_down_f32(x as f32)` /
/// `next_up_f32(x as f32)`, which on a zero result gives one subnormal — so the stored
/// checkpoint sat at 11 subnormals (∓1.5414e-44, that one plus the 10-ULP margin) and
/// the recomputed box at one (∓1.4013e-45). Both are ~6 orders under
/// `f32::MIN_POSITIVE`, which is why the old cap held. Every endpoint moved OUTWARD;
/// the box still encloses the fixture's true zero.
///
/// # What this still catches
///
/// The comparison is now DIRECTIONAL, which the old symmetric `|diff| < tol` was not.
/// A lower bound may sit at or below `lower` but never above it, and an upper at or
/// above `upper` but never below. So a propagation that rounds the wrong way — the
/// only direction that can exclude a true value — fails here even by one ULP, where
/// the old form accepted it as long as the slip stayed small. The outward slack is
/// capped at the derived floor, so a charge that grows past it fails too.
fn assert_uniform_bounds_near(bounds: &BoundedTensor, lower: f32, upper: f32) {
    // `f32::MIN_POSITIVE` (the subnormal-flush floor) shifted by the 10-ULP
    // (`in_features + 2`, `in_features == 8`) checkpoint margin. Derived above.
    let outward_cap = f32::from_bits(f32::MIN_POSITIVE.to_bits() + 10);
    for (&actual_lower, &actual_upper) in bounds.lower().iter().zip(bounds.upper().iter()) {
        // Split from the range check on purpose: an INWARD move is a soundness alarm,
        // an overshoot is only a charge that grew, and the messages must not blur them.
        assert!(
            actual_lower <= lower,
            "lower bound moved INWARD (unsound direction): got {actual_lower}, must be <= {lower}"
        );
        assert!(
            actual_lower >= lower - outward_cap,
            "lower bound overshot the certified near-zero floor: got {actual_lower}, \
             must be >= {} (diff: {})",
            lower - outward_cap,
            actual_lower - lower
        );
        assert!(
            actual_upper >= upper,
            "upper bound moved INWARD (unsound direction): got {actual_upper}, must be >= {upper}"
        );
        assert!(
            actual_upper <= upper + outward_cap,
            "upper bound overshot the certified near-zero floor: got {actual_upper}, \
             must be <= {} (diff: {})",
            upper + outward_cap,
            actual_upper - upper
        );
        assert!(
            actual_lower <= actual_upper,
            "inverted bounds: lower {actual_lower} > upper {actual_upper}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_basic() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 20);

    // Add checkpoints at layers 4, 9, 14, 19
    for i in [4, 9, 14, 19] {
        let bounds = create_input(10);
        checkpointed.add_checkpoint(i, bounds);
    }

    assert_eq!(checkpointed.num_checkpoints(), 4);
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_checkpoints() {
    let network = create_test_network(20, 10, 10);
    let input = create_input(10);

    let config = StreamingConfig {
        checkpoint_interval: 5,
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);

    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // With interval=5 and 20 layers, checkpoints at: 4, 9, 14, 19
    assert_eq!(checkpointed.num_checkpoints(), 4);
}

#[ntest::timeout(10000)]
#[test]
fn test_find_nearest_checkpoint() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input.clone(), 20);

    // Add checkpoints at layers 4, 9, 14
    checkpointed.add_checkpoint(4, constant_bounds(10, -4.0, 4.0));
    checkpointed.add_checkpoint(9, constant_bounds(10, -9.0, 9.0));
    checkpointed.add_checkpoint(14, constant_bounds(10, -14.0, 14.0));

    // Test find_nearest_checkpoint_before
    // Layer 6 should find checkpoint at 4
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(6).unwrap();
    assert_eq!(idx, 4);
    assert_uniform_bounds(&bounds, -4.0, 4.0);

    // Layer 12 should find checkpoint at 9
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(12).unwrap();
    assert_eq!(idx, 9);
    assert_uniform_bounds(&bounds, -9.0, 9.0);

    // Layer 2 should find no checkpoint (returns -1)
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(2).unwrap();
    assert_eq!(idx, -1);
    assert_eq!(bounds.lower(), input.lower());
    assert_eq!(bounds.upper(), input.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_is_compressed_f32() {
    let input = create_input(10);
    let checkpointed = CheckpointedBounds::new(input, 10);
    assert!(!checkpointed.is_compressed());
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_compression_stats_f32() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 10);
    checkpointed.add_checkpoint(0, create_input(10));

    // f32 storage should return None for compression stats
    assert!(checkpointed.compression_stats().is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_empty_checkpoints() {
    let input = create_input(10);
    let checkpointed = CheckpointedBounds::new(input, 10);

    assert_eq!(checkpointed.num_checkpoints(), 0);
    assert!(checkpointed.last_checkpoint().unwrap().is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpointed_bounds_last_checkpoint_f32() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 10);

    let bounds1 = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[10]), -1.0_f32),
        ArrayD::from_elem(ndarray::IxDyn(&[10]), 1.0_f32),
    )
    .unwrap();
    let bounds2 = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[10]), -2.0_f32),
        ArrayD::from_elem(ndarray::IxDyn(&[10]), 2.0_f32),
    )
    .unwrap();

    checkpointed.add_checkpoint(3, bounds1);
    checkpointed.add_checkpoint(7, bounds2);

    let last = checkpointed.last_checkpoint().unwrap();
    assert!(last.is_some());
    let last_bounds = last.unwrap();
    // Should be bounds2 since it's at the highest index
    assert!((last_bounds.lower()[0] - (-2.0)).abs() < 1e-6);
    assert!((last_bounds.upper()[0] - 2.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_get_bounds_at_invalid_layer() {
    let input = create_input(10);
    let checkpointed = CheckpointedBounds::new(input, 5);
    let network = create_test_network(5, 10, 10);

    // Layer 10 is out of range (max is 4)
    let result = checkpointed.bounds_at(10, &network);
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_get_bounds_at_exact_checkpoint() {
    let input = create_input(8);
    let network = create_test_network(10, 8, 8);

    let config = StreamingConfig {
        checkpoint_interval: 3,
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);
    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // Layer 2 should be a checkpoint (index 2, which is (2+1)%3 == 0)
    // Actually with interval=3, checkpoints at 2, 5, 8, 9
    let bounds = checkpointed.bounds_at(2, &network).unwrap();

    // With zero-weight network and [-1, 1] input, all layer outputs are mathematically zero.
    // Sound propagation still publishes a small OUTWARD floor around that zero (the
    // subnormal-flush floor plus the checkpoint ULP margin; see
    // `assert_uniform_bounds_near`), so use the directional near-zero check.
    assert_eq!(bounds.lower().shape(), &[8]);
    assert_uniform_bounds_near(&bounds, 0.0, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_get_bounds_at_recomputes_after_checkpoint_boundary() {
    let input = create_input(8);
    let network = create_test_network(10, 8, 8);

    let config = StreamingConfig {
        checkpoint_interval: 3,
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);
    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // Layer 3 sits immediately after checkpoint layer 2, so this must use the
    // recomputation path instead of the exact-checkpoint fast path.
    let (start_idx, start_bounds) = checkpointed.find_nearest_checkpoint_before(3).unwrap();
    assert_eq!(start_idx, 2);
    assert_uniform_bounds_near(&start_bounds, 0.0, 0.0);

    let recomputed = checkpointed.bounds_at(3, &network).unwrap();
    assert_uniform_bounds_near(&recomputed, 0.0, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_find_nearest_checkpoint_at_exact() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 20);

    checkpointed.add_checkpoint(5, constant_bounds(10, -5.0, 5.0));
    checkpointed.add_checkpoint(10, constant_bounds(10, -10.0, 10.0));

    // At exact checkpoint should return that checkpoint
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(5).unwrap();
    assert_eq!(idx, 5);
    assert_uniform_bounds(&bounds, -5.0, 5.0);

    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(10).unwrap();
    assert_eq!(idx, 10);
    assert_uniform_bounds(&bounds, -10.0, 10.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_find_nearest_checkpoint_at_layer_zero() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input.clone(), 20);

    checkpointed.add_checkpoint(5, constant_bounds(10, -5.0, 5.0));

    // Layer 0 should return -1 (no checkpoint before layer 0)
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(0).unwrap();
    assert_eq!(idx, -1);
    assert_eq!(bounds.lower(), input.lower());
    assert_eq!(bounds.upper(), input.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_memory_bytes_empty() {
    let input = create_input(10);
    let checkpointed = CheckpointedBounds::new(input, 10);

    // Just input memory (10 elements * 4 bytes * 2 (lower+upper))
    let mem = checkpointed.memory_bytes();
    assert_eq!(mem, 10 * 4 * 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_memory_bytes_with_checkpoints() {
    let input = create_input(100);
    let mut checkpointed = CheckpointedBounds::new(input, 10);

    for i in 0..5 {
        checkpointed.add_checkpoint(i, create_input(100));
    }

    // input + 5 checkpoints, each 100 elements * 4 bytes * 2
    let expected = (1 + 5) * 100 * 4 * 2;
    assert_eq!(checkpointed.memory_bytes(), expected);
}

#[ntest::timeout(10000)]
#[test]
fn test_add_checkpoint_maintains_sort_order() {
    let input = create_input(10);
    let mut checkpointed = CheckpointedBounds::new(input, 20);

    // Add checkpoints out of order
    checkpointed.add_checkpoint(10, constant_bounds(10, -10.0, 10.0));
    checkpointed.add_checkpoint(5, constant_bounds(10, -5.0, 5.0));
    checkpointed.add_checkpoint(15, constant_bounds(10, -15.0, 15.0));
    checkpointed.add_checkpoint(0, constant_bounds(10, 0.0, 0.0));

    assert_eq!(checkpointed.num_checkpoints(), 4);

    // Verify ordering by finding nearest checkpoint
    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(3).unwrap();
    assert_eq!(idx, 0);
    assert_uniform_bounds(&bounds, 0.0, 0.0);

    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(7).unwrap();
    assert_eq!(idx, 5);
    assert_uniform_bounds(&bounds, -5.0, 5.0);

    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(12).unwrap();
    assert_eq!(idx, 10);
    assert_uniform_bounds(&bounds, -10.0, 10.0);

    let (idx, bounds) = checkpointed.find_nearest_checkpoint_before(20).unwrap();
    assert_eq!(idx, 15);
    assert_uniform_bounds(&bounds, -15.0, 15.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpoint_interval_zero_clamped() {
    let network = create_test_network(5, 8, 8);
    let input = create_input(8);

    let config = StreamingConfig {
        checkpoint_interval: 0, // Should be clamped to 1
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);

    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // With interval clamped to 1, should have 5 checkpoints (one per layer)
    assert_eq!(checkpointed.num_checkpoints(), 5);
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_checkpoints_empty_network() {
    let network = Network::new();
    let input = create_input(10);

    let config = StreamingConfig::default();
    let verifier = StreamingVerifier::new(config);

    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    assert_eq!(checkpointed.num_checkpoints(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpoint_interval_larger_than_network() {
    let network = create_test_network(3, 8, 8);
    let input = create_input(8);

    let config = StreamingConfig {
        checkpoint_interval: 100, // Much larger than network size
        ..Default::default()
    };
    let verifier = StreamingVerifier::new(config);

    let checkpointed = verifier
        .collect_checkpointed_bounds(&network, &input)
        .unwrap();

    // Should have exactly 1 checkpoint (at the last layer)
    assert_eq!(checkpointed.num_checkpoints(), 1);
}
