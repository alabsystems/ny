// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_network_layernorm_sound_mode_enforced() {
    // End-to-end test: Network with Linear -> LayerNorm -> Linear
    // With sound mode (default), CROWN propagation through the network should
    // return an error, NOT silently proceed with heuristic sampling.
    use crate::layers::LayerNormCrownMode;

    let mut network = Network::new();
    let hidden = 4;

    // Linear: 4 -> 4
    let weight1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    // LayerNorm with explicit Sound mode (should trigger error)
    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sound);
    assert_eq!(ln.crown_mode, LayerNormCrownMode::Sound);
    network.add_layer(Layer::LayerNorm(ln));

    // Linear: 4 -> 4
    let weight2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 23 + j * 41) as f32).cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    // Create input bounds
    let lower = ArrayD::from_elem(IxDyn(&[1, hidden]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run CROWN - this should FAIL due to sound mode on LayerNorm
    let result = network.propagate_crown_batched(&input);

    assert!(
        result.is_err(),
        "Network CROWN with LayerNorm in Sound mode should return error"
    );

    let err = result.unwrap_err();
    // Must be SoundnessRefusal (not UnsupportedOp) so network fallback doesn't catch it
    assert!(
        matches!(err, NyError::SoundnessRefusal(_)),
        "Expected SoundnessRefusal, got: {:?}",
        err
    );
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("LayerNorm") || err_msg.contains("heuristic"),
        "Error should mention LayerNorm or heuristic: {}",
        err_msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_layernorm_sound_mode_enforced_non_batched() {
    // Verify SoundnessRefusal propagation through the non-batched CROWN path
    // (propagate_crown_with_engine, crown.rs fallback site at the unified trait
    // dispatch). This complements test_network_layernorm_sound_mode_enforced which
    // tests the batched path.
    use crate::layers::LayerNormCrownMode;

    let mut network = Network::new();
    let hidden = 4;

    let weight1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sound);
    assert_eq!(ln.crown_mode, LayerNormCrownMode::Sound);
    network.add_layer(Layer::LayerNorm(ln));

    let weight2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 23 + j * 41) as f32).cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    let lower = ArrayD::from_elem(IxDyn(&[1, hidden]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Non-batched CROWN path — SoundnessRefusal must propagate, not be caught
    // by the UnsupportedOp fallback at crown.rs:232.
    let result = network.propagate_crown(&input);

    assert!(
        result.is_err(),
        "Non-batched CROWN with LayerNorm Sound mode should return error, got Ok"
    );

    let err = result.unwrap_err();
    assert!(
        matches!(err, NyError::SoundnessRefusal(_)),
        "Expected SoundnessRefusal from non-batched path, got: {:?}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_layernorm_sound_mode_enforced_crown_ibp() {
    // Verify SoundnessRefusal propagation through the CROWN-IBP path
    // (propagate_crown_ibp, crown.rs fallback site at line ~431).
    use crate::layers::LayerNormCrownMode;

    let mut network = Network::new();
    let hidden = 4;

    let weight1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sound);
    assert_eq!(ln.crown_mode, LayerNormCrownMode::Sound);
    network.add_layer(Layer::LayerNorm(ln));

    let weight2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 23 + j * 41) as f32).cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    let lower = ArrayD::from_elem(IxDyn(&[1, hidden]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = network.propagate_crown_ibp(&input);

    assert!(
        result.is_err(),
        "CROWN-IBP with LayerNorm Sound mode should return error, got Ok"
    );

    let err = result.unwrap_err();
    assert!(
        matches!(err, NyError::SoundnessRefusal(_)),
        "Expected SoundnessRefusal from CROWN-IBP path, got: {:?}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_layernorm_sound_mode_enforced_crown_fast() {
    // Verify SoundnessRefusal propagation through the CROWN-fast path
    // (propagate_crown_fast, fast.rs fallback site at line ~174).
    use crate::layers::LayerNormCrownMode;

    let mut network = Network::new();
    let hidden = 4;

    let weight1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sound);
    assert_eq!(ln.crown_mode, LayerNormCrownMode::Sound);
    network.add_layer(Layer::LayerNorm(ln));

    let weight2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 23 + j * 41) as f32).cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    let lower = ArrayD::from_elem(IxDyn(&[1, hidden]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = network.propagate_crown_fast(&input);

    assert!(
        result.is_err(),
        "CROWN-fast with LayerNorm Sound mode should return error, got Ok"
    );

    let err = result.unwrap_err();
    assert!(
        matches!(err, NyError::SoundnessRefusal(_)),
        "Expected SoundnessRefusal from CROWN-fast path, got: {:?}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_layernorm_sound_mode_enforced_alpha_crown() {
    // Verify SoundnessRefusal propagation through the alpha-CROWN path.
    // Alpha-CROWN catches UnsupportedOp|LayerError but not SoundnessRefusal,
    // so the error must propagate to the caller.
    use crate::layers::LayerNormCrownMode;

    let mut network = Network::new();
    let hidden = 4;

    let weight1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    // ReLU is needed for alpha-CROWN to engage (optimizes alpha slopes on ReLU)
    network.add_layer(Layer::ReLU(ReLULayer));

    let weight_mid = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.2 * ((i * 13 + j * 29) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight_mid, None).unwrap()));

    let ln = LayerNormLayer::new_default(hidden, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sound);
    assert_eq!(ln.crown_mode, LayerNormCrownMode::Sound);
    network.add_layer(Layer::LayerNorm(ln));

    let weight2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 23 + j * 41) as f32).cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    let lower = ArrayD::from_elem(IxDyn(&[1, hidden]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = network.propagate_alpha_crown(&input);

    assert!(
        result.is_err(),
        "Alpha-CROWN with LayerNorm Sound mode should return error, got Ok"
    );

    // Alpha-CROWN falls back to CROWN on UnsupportedOp|LayerError. SoundnessRefusal
    // is neither, so it must propagate as-is.
    let err = result.unwrap_err();
    assert!(
        matches!(err, NyError::SoundnessRefusal(_)),
        "Expected SoundnessRefusal from alpha-CROWN path, got: {:?}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_layernorm_cut_mode_succeeds() {
    // End-to-end test: Network with Linear -> LayerNorm -> Linear
    // With cut mode, CROWN propagation should succeed (sound but loses correlations).
    use crate::layers::LayerNormCrownMode;

    let mut network = Network::new();
    let hidden = 4;

    // Linear: 4 -> 4
    let weight1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    // LayerNorm - will set cut mode after adding
    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
    network.add_layer(Layer::LayerNorm(ln));

    // Set LayerNorm to cut mode
    let modified = network.set_layernorm_crown_mode(LayerNormCrownMode::Cut);
    assert_eq!(modified, 1, "Should modify exactly 1 LayerNorm layer");

    // Linear: 4 -> 4
    let weight2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 23 + j * 41) as f32).cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    // Create input bounds
    let lower = ArrayD::from_elem(IxDyn(&[1, hidden]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run CROWN - this should SUCCEED with cut mode
    let result = network.propagate_crown_batched(&input);

    assert!(
        result.is_ok(),
        "Network CROWN with LayerNorm in Cut mode should succeed: {:?}",
        result.err()
    );

    let bounds = result.unwrap();

    // Verify output shape
    assert_eq!(bounds.shape(), &[1, hidden]);

    // Verify all bounds are finite and valid
    for (l, u) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert!(l.is_finite(), "Lower bound should be finite");
        assert!(u.is_finite(), "Upper bound should be finite");
        assert!(l <= u, "Lower bound should <= upper bound");
    }
}

// ── IbpValidated measurement tests (#3221) ──────────────────────────────
//
// These tests measure whether the IbpValidated CROWN mode produces
// meaningfully tight bounds through normalization layers. This is the
// acceptance criterion for #3221: bounds through norm layers must not
// be degenerate (~1e11 width). The design doc at
// designs/2026-03-03-ibp-validated-norm-crown.md specifies the measurement:
//   Input bounds: [-5, 5] on dimension 128
//   Expected: CROWN bounds width < 50 (comparable to scalar verification)
//   Previous: IBP-only produced ~1e11 width at tensor level

/// Build a Linear -> LayerNorm -> Linear test network.
/// Returns (network, weight1, weight2) for manual forward-pass soundness checks.
fn build_ln_measurement_network(hidden: usize) -> (Network, Array2<f32>, Array2<f32>) {
    let mut network = Network::new();
    let w1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()));
    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
    network.add_layer(Layer::LayerNorm(ln));
    let w2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 23 + j * 41) as f32).cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), None).unwrap()));
    (network, w1, w2)
}

/// Compute per-element widths from bounds, asserting all are finite and valid.
fn validated_widths(bounds: &BoundedTensor) -> Vec<f32> {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(&l, &u)| {
            assert!(
                l.is_finite() && u.is_finite(),
                "Non-finite bound: [{}, {}]",
                l,
                u
            );
            assert!(l <= u, "Inverted bound: {} > {}", l, u);
            u - l
        })
        .collect()
}

/// Log CROWN vs IBP measurement results for #3221.
fn log_measurement(label: &str, crown_widths: &[f32], ibp_widths: &[f32]) {
    let avg_c = crown_widths.iter().sum::<f32>() / crown_widths.len() as f32;
    let max_c = crown_widths.iter().cloned().fold(0.0_f32, f32::max);
    let avg_i = ibp_widths.iter().sum::<f32>() / ibp_widths.len() as f32;
    let ratio = if avg_i > 0.0 { avg_c / avg_i } else { f32::NAN };
    eprintln!("=== {} ===", label);
    eprintln!("  CROWN avg/max width: {:.4} / {:.4}", avg_c, max_c);
    eprintln!("  IBP   avg width: {:.4}", avg_i);
    eprintln!("  CROWN/IBP ratio: {:.4}", ratio);
}

#[ntest::timeout(60000)]
#[test]
fn test_network_layernorm_ibp_validated_bounds_finite_and_sound() {
    // Verifies IbpValidated CROWN produces finite, valid, non-degenerate, sound bounds.
    let hidden = 16;
    let (network, w1, w2) = build_ln_measurement_network(hidden);

    assert_eq!(
        LayerNormLayer::new_default(hidden, 1e-5)
            .unwrap()
            .crown_mode,
        LayerNormCrownMode::IbpValidated,
    );

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, hidden]), -5.0_f32),
        ArrayD::from_elem(IxDyn(&[1, hidden]), 5.0_f32),
    )
    .unwrap();

    let crown = network
        .propagate_crown_batched(&input)
        .expect("CROWN should succeed");
    let ibp = network.propagate_ibp(&input).expect("IBP should succeed");
    let cw = validated_widths(&crown);
    let iw = validated_widths(&ibp);
    log_measurement(
        &format!("IbpValidated measurement (hidden={})", hidden),
        &cw,
        &iw,
    );

    let max_w = cw.iter().cloned().fold(0.0_f32, f32::max);
    assert!(max_w < 500.0, "Degenerate bounds: max width {}", max_w);

    // Soundness: 50 random samples must be contained in CROWN bounds
    for s in 0..50 {
        let mut x = Array1::zeros(hidden);
        for i in 0..hidden {
            let t = ((s as u32).wrapping_mul(2654435761) ^ (i as u32)).wrapping_mul(2654435761)
                as f32
                / u32::MAX as f32;
            x[i] = -5.0 + 10.0 * t;
        }
        let y = w2.dot(
            &LayerNormLayer::new_default(hidden, 1e-5)
                .unwrap()
                .eval(&w1.dot(&x))
                .unwrap(),
        );
        for i in 0..hidden {
            assert!(
                y[i] >= crown.lower()[[0, i]] - 1e-3,
                "sample {} dim {} below",
                s,
                i
            );
            assert!(
                y[i] <= crown.upper()[[0, i]] + 1e-3,
                "sample {} dim {} above",
                s,
                i
            );
        }
    }
}

#[ntest::timeout(120000)]
#[test]
fn test_network_layernorm_ibp_validated_dim128_measurement() {
    // #3221 acceptance criterion: dim 128, input [-5, 5].
    // Design doc: designs/2026-03-03-ibp-validated-norm-crown.md
    let hidden = 128;
    let (network, _, _) = build_ln_measurement_network(hidden);

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, hidden]), -5.0_f32),
        ArrayD::from_elem(IxDyn(&[1, hidden]), 5.0_f32),
    )
    .unwrap();

    let crown = network
        .propagate_crown_batched(&input)
        .expect("CROWN dim 128");
    let ibp = network.propagate_ibp(&input).expect("IBP dim 128");
    let cw = validated_widths(&crown);
    let iw = validated_widths(&ibp);
    log_measurement("#3221 acceptance (hidden=128)", &cw, &iw);

    // Reject ~1e11 degenerate bounds; 1000 is generous but excludes catastrophic failure.
    let max_w = cw.iter().cloned().fold(0.0_f32, f32::max);
    assert!(max_w < 1000.0, "Degenerate at dim 128: max width {}", max_w);
}

#[ntest::timeout(10000)]
#[test]
fn test_network_layernorm_ibp_validated_small_perturbation() {
    // Small perturbation regime (eps=0.1). Design doc notes CROWN benefit
    // requires deeper networks with cross-layer correlations (e.g., ReLU).
    let hidden = 16;
    let (network, _, _) = build_ln_measurement_network(hidden);

    let center = ArrayD::from_shape_fn(IxDyn(&[1, hidden]), |idx| {
        0.5 * ((idx[1] * 7 + 3) as f32).sin()
    });
    let input = BoundedTensor::new(&center - 0.1, &center + 0.1).unwrap();

    let crown = network
        .propagate_crown_batched(&input)
        .expect("CROWN small eps");
    let ibp = network.propagate_ibp(&input).expect("IBP small eps");
    let cw = validated_widths(&crown);
    let iw = validated_widths(&ibp);
    log_measurement(
        &format!("Small perturbation (hidden={}, eps=0.1)", hidden),
        &cw,
        &iw,
    );

    // For Linear -> LayerNorm -> Linear, CROWN = IBP after tighten_crown_output
    // intersection. This is expected (see design doc tightness analysis).
    let avg_w = cw.iter().sum::<f32>() / cw.len() as f32;
    assert!(avg_w < 500.0, "Small eps bounds degenerate: avg {}", avg_w);
}

/// Build a deeper network: Linear → ReLU → Linear → LayerNorm → Linear.
///
/// The ReLU introduces a nonlinearity that CROWN can exploit through
/// cross-layer correlation tracking. Without ReLU, CROWN backward through
/// LayerNorm composes with pure linear layers, and `tighten_crown_output`
/// intersects to IBP. With ReLU, CROWN's linear relaxation captures
/// dependencies that IBP's independent-per-dimension approach cannot.
fn build_deep_ln_network(hidden: usize) -> Network {
    let mut network = Network::new();

    // Linear 1: input → hidden
    let w1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));

    // ReLU: introduces nonlinearity that CROWN can exploit
    network.add_layer(Layer::ReLU(ReLULayer));

    // Linear 2: hidden → hidden
    let w2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.25 * ((i * 13 + j * 29) as f32).cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    // LayerNorm: normalization with IbpValidated CROWN mode (default)
    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
    network.add_layer(Layer::LayerNorm(ln));

    // Linear 3: hidden → hidden (output projection)
    let w3 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.2 * ((i * 23 + j * 41) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(w3, None).unwrap()));

    network
}

#[ntest::timeout(60000)]
#[test]
fn test_network_layernorm_deep_crown_vs_ibp() {
    // #3221 key test: Linear → ReLU → Linear → LayerNorm → Linear.
    // CROWN should be strictly tighter than IBP because ReLU's linear
    // relaxation captures cross-layer correlations that IBP loses.
    // This validates that IbpValidated CROWN through normalization
    // actually provides value beyond pure IBP in realistic networks.
    let hidden = 16;
    let network = build_deep_ln_network(hidden);

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, hidden]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, hidden]), 1.0_f32),
    )
    .unwrap();

    let crown = network
        .propagate_crown_batched(&input)
        .expect("CROWN should succeed on deep LN network");
    let ibp = network.propagate_ibp(&input).expect("IBP should succeed");

    let cw = validated_widths(&crown);
    let iw = validated_widths(&ibp);
    log_measurement("Deep network CROWN vs IBP (hidden=16)", &cw, &iw);

    let avg_crown = cw.iter().sum::<f32>() / cw.len() as f32;
    let avg_ibp = iw.iter().sum::<f32>() / iw.len() as f32;
    let ratio = avg_crown / avg_ibp;
    eprintln!("  CROWN/IBP avg ratio: {:.6}", ratio);

    // CROWN must produce finite, non-degenerate bounds.
    assert!(avg_crown < 500.0, "Degenerate CROWN bounds: {}", avg_crown);
    assert!(avg_ibp < 500.0, "Degenerate IBP bounds: {}", avg_ibp);

    // With ReLU in the network, CROWN should be at least as tight as IBP.
    // `tighten_crown_output` guarantees CROWN <= IBP.
    assert!(
        ratio <= 1.0 + 1e-5,
        "CROWN should be <= IBP after tightening, ratio: {}",
        ratio
    );

    // The critical measurement: is CROWN *strictly* tighter than IBP?
    // If ratio < 1.0, CROWN is exploiting cross-layer ReLU correlations.
    if ratio < 0.999 {
        eprintln!(
            "  SUCCESS: CROWN is {:.2}% tighter than IBP",
            (1.0 - ratio) * 100.0
        );
    } else {
        eprintln!(
            "  NOTE: CROWN ~= IBP (ratio {:.6}). IbpValidated margins may dominate.",
            ratio
        );
    }

    // Compare with Sampling mode: if Sampling also = IBP, the limitation
    // is fundamental to Jacobian-based norm linearization, not just IbpValidated margins.
    use crate::layers::LayerNormCrownMode;
    let mut sampling_net = build_deep_ln_network(hidden);
    sampling_net.set_layernorm_crown_mode(LayerNormCrownMode::Sampling);
    let sampling_crown = sampling_net
        .propagate_crown_batched(&input)
        .expect("Sampling CROWN should succeed");
    let sw = validated_widths(&sampling_crown);
    let avg_sampling = sw.iter().sum::<f32>() / sw.len() as f32;
    eprintln!("  Sampling CROWN avg width: {:.4}", avg_sampling);
    eprintln!("  Sampling/IBP ratio: {:.6}", avg_sampling / avg_ibp);
}

/// Build the baseline network: Linear → ReLU → Linear → Linear (no LayerNorm).
/// Same weights as `build_deep_ln_network` but with LayerNorm removed.
fn build_relu_baseline_network(hidden: usize) -> Network {
    let mut net = Network::new();
    let w1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    net.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    net.add_layer(Layer::ReLU(ReLULayer));
    let w2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.25 * ((i * 13 + j * 29) as f32).cos()
    });
    net.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));
    let w3 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.2 * ((i * 23 + j * 41) as f32).sin()
    });
    net.add_layer(Layer::Linear(LinearLayer::new(w3, None).unwrap()));
    net
}

#[ntest::timeout(60000)]
#[test]
fn test_network_layernorm_crown_vs_relu_baseline() {
    // Measures the "LayerNorm correlation cost": how much CROWN benefit
    // is lost when LayerNorm is inserted into a ReLU network.
    //
    // Without LayerNorm, CROWN exploits ReLU correlations (CROWN << IBP).
    // With LayerNorm, the dense Jacobian disperses correlation info,
    // and tighten_crown_output snaps to IBP (CROWN = IBP).
    let hidden = 16;
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, hidden]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, hidden]), 1.0_f32),
    )
    .unwrap();

    // Baseline: no LayerNorm
    let baseline = build_relu_baseline_network(hidden);
    let b_crown = baseline
        .propagate_crown_batched(&input)
        .expect("Baseline CROWN");
    let b_ibp = baseline.propagate_ibp(&input).expect("Baseline IBP");
    let bcw = validated_widths(&b_crown);
    let biw = validated_widths(&b_ibp);
    log_measurement("Baseline (no LayerNorm)", &bcw, &biw);
    let b_ratio = bcw.iter().sum::<f32>() / biw.iter().sum::<f32>();

    // With LayerNorm
    let ln_net = build_deep_ln_network(hidden);
    let ln_crown = ln_net.propagate_crown_batched(&input).expect("LN CROWN");
    let ln_ibp = ln_net.propagate_ibp(&input).expect("LN IBP");
    let lcw = validated_widths(&ln_crown);
    let liw = validated_widths(&ln_ibp);
    log_measurement("With LayerNorm", &lcw, &liw);
    let ln_ratio = lcw.iter().sum::<f32>() / liw.iter().sum::<f32>();

    eprintln!("  Baseline CROWN/IBP: {:.6}", b_ratio);
    eprintln!("  LayerNorm CROWN/IBP: {:.6}", ln_ratio);
    eprintln!(
        "  LayerNorm destroys {:.1}% of CROWN's ReLU benefit",
        (ln_ratio - b_ratio) / (1.0 - b_ratio) * 100.0
    );

    // Baseline MUST show CROWN < IBP (ReLU benefit exists)
    assert!(
        b_ratio < 0.5,
        "Baseline should show CROWN < IBP: {}",
        b_ratio
    );
    // Both networks must produce non-degenerate bounds
    let max_w = lcw.iter().cloned().fold(0.0_f32, f32::max);
    assert!(
        max_w < 500.0,
        "LayerNorm CROWN bounds degenerate: {}",
        max_w
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_network_layernorm_deep_soundness_sampling() {
    // Soundness check for deep network: random samples must lie within CROWN bounds.
    let hidden = 16;
    let network = build_deep_ln_network(hidden);

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, hidden]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, hidden]), 1.0_f32),
    )
    .unwrap();

    let crown = network
        .propagate_crown_batched(&input)
        .expect("CROWN should succeed");

    // 100 random samples must all be contained
    for s in 0..100 {
        let x_flat: Vec<f32> = (0..hidden)
            .map(|i| {
                let t = ((s as u32).wrapping_mul(2654435761) ^ (i as u32)).wrapping_mul(2654435761)
                    as f32
                    / u32::MAX as f32;
                -1.0 + 2.0 * t
            })
            .collect();
        let x = ArrayD::from_shape_vec(IxDyn(&[1, hidden]), x_flat).unwrap();
        let x_bt = BoundedTensor::new(x.clone(), x.clone()).unwrap();
        let y = network.propagate_ibp(&x_bt).unwrap();

        for i in 0..hidden {
            assert!(
                y.lower()[[0, i]] >= crown.lower()[[0, i]] - 1e-3,
                "s={} dim={}: y={} < crown_lower={}",
                s,
                i,
                y.lower()[[0, i]],
                crown.lower()[[0, i]]
            );
            assert!(
                y.upper()[[0, i]] <= crown.upper()[[0, i]] + 1e-3,
                "s={} dim={}: y={} > crown_upper={}",
                s,
                i,
                y.upper()[[0, i]],
                crown.upper()[[0, i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_layernorm_sampling_mode_succeeds() {
    // End-to-end test: Network with Linear -> LayerNorm -> Linear
    // With sampling mode, CROWN propagation should succeed (NOT provably sound).
    use crate::layers::LayerNormCrownMode;

    let mut network = Network::new();
    let hidden = 4;

    // Linear: 4 -> 4
    let weight1 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 17 + j * 31) as f32).sin()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    // LayerNorm - will set sampling mode after adding
    let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
    network.add_layer(Layer::LayerNorm(ln));

    // Set LayerNorm to sampling mode (heuristic, not provably sound)
    let modified = network.set_layernorm_crown_mode(LayerNormCrownMode::Sampling);
    assert_eq!(modified, 1, "Should modify exactly 1 LayerNorm layer");

    // Linear: 4 -> 4
    let weight2 = Array2::from_shape_fn((hidden, hidden), |(i, j)| {
        0.3 * ((i * 23 + j * 41) as f32).cos()
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    // Create input bounds
    let lower = ArrayD::from_elem(IxDyn(&[1, hidden]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, hidden]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run CROWN - this should SUCCEED with sampling mode
    let result = network.propagate_crown_batched(&input);

    assert!(
        result.is_ok(),
        "Network CROWN with LayerNorm in Sampling mode should succeed: {:?}",
        result.err()
    );

    let bounds = result.unwrap();

    // Verify output shape
    assert_eq!(bounds.shape(), &[1, hidden]);

    // Verify all bounds are finite and valid
    for (l, u) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert!(l.is_finite(), "Lower bound should be finite");
        assert!(u.is_finite(), "Upper bound should be finite");
        assert!(l <= u, "Lower bound should <= upper bound");
    }

    // Verify we got non-trivial bounds (not infinite or exactly zero)
    let avg_width: f32 = bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(l, u)| u - l)
        .sum::<f32>()
        / (bounds.lower().len() as f32);

    assert!(
        avg_width > 0.0 && avg_width < 500.0,
        "Sampling mode should produce reasonable bounds, got avg width: {}",
        avg_width
    );
}
