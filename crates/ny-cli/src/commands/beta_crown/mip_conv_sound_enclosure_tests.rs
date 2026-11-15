// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #mip-conv-sound-enclosure — the sequential MIP escalation's Big-M constants
// must come from the CERTIFIED IBP collector.
//
// Cancellation in a round-to-nearest f32 reduction can collapse the exact
// pre-activation 2^24 * 1 + 1 * 1 - 2^24 = 1 to zero. Keep that historical
// inward [0, 0] box as an explicit negative control. Production collectors
// may improve; they must enclose the exact value rather than preserve the bug.
// The identity output reaches the unsafe region z >= 0.5, so a sound encoding
// must admit its witness. The certified Conv2d widening encloses the value.
//
// (c) end-to-end through `verify_with_mip` is not constructible in this
// harness: it returns `Result<()>`, publishes the verdict on stdout, and needs
// a `BetaCrownModel` plus vnnlib/model paths. The encode→solve→map level below
// IS the mechanism the audit identified (the certificate is a statement about
// the encoded rows), so it is asserted there instead.
//
// The seam is TYPE-ENFORCED (`mip_sound_intermediate_bounds.rs`): both MIP
// collectors return `SoundIntermediateBounds`, whose only constructors are the
// certified collector and CROWN over that certified vector, and
// `convert_intermediate_bounds` accepts nothing else. That is why the
// negative control in (c) has to hand-build the encoder's Big-M rows from the
// deliberately inward box, bypassing the typed converter only in the control.

use super::super::mip_preprocess::{
    bounded_tensor_to_bounds, convert_intermediate_bounds, extract_linear_relu_params,
};
use super::super::mip_single_hidden::{
    collect_exact_single_hidden_intermediate_bounds, is_single_hidden_linear_relu_linear,
};
use super::super::mip_sound_intermediate_bounds::SoundIntermediateBounds;
use super::intermediate_bounds::collect_mip_intermediate_bounds_with_deadline;
use super::*;
use ndarray::{arr2, Array1, ArrayD, IxDyn};
use ny_propagate::layers::{Conv2dLayer, FlattenLayer, ReLULayer};
use std::time::{Duration, Instant};

const TWO_POW_24: f32 = 16_777_216.0;
/// Exact real-arithmetic pre-activation of the conv neuron at the fixture point.
const TRUE_PRE_ACTIVATION: f32 = 1.0;

/// Conv2d(1→1, 1×2 kernel [2^24, 1], bias −2^24) → ReLU → Flatten → Gemm([[1]])
/// over the degenerate input box x = [1, 1] of shape [C=1, H=1, W=2].
/// This is malbeware's `Conv → ReLU → Flatten → Gemm` topology, which matches
/// the single-hidden MIP fast path after `unfold_conv2d_to_linear`.
fn build_cancellation_conv_fixture() -> (Network, BoundedTensor) {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2]), vec![TWO_POW_24, 1.0]).unwrap();
    let bias = Array1::from(vec![-TWO_POW_24]);
    let conv = Conv2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::flatten_all()));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), None).unwrap(),
    ));

    let point = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0f32, 1.0]).unwrap();
    let input = BoundedTensor::new(point.clone(), point).unwrap();
    (network, input)
}

fn scalar_box(bounds: &BoundedTensor) -> (f32, f32) {
    assert_eq!(
        bounds.lower().len(),
        1,
        "fixture conv output is a single neuron"
    );
    // Shape-agnostic on purpose: the certified conv arm keeps [C, H, W] while a
    // CROWN concretization may hand the single element back flattened.
    (
        *bounds.lower().iter().next().unwrap(),
        *bounds.upper().iter().next().unwrap(),
    )
}

fn encloses((lo, hi): (f32, f32), value: f32) -> bool {
    lo <= value && value <= hi
}

fn assert_bit_equal(name: &str, got: &[BoundedTensor], want: &[BoundedTensor]) {
    assert_eq!(got.len(), want.len(), "{name}: layer count");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.lower(), w.lower(), "{name}: layer {i} lower");
        assert_eq!(g.upper(), w.upper(), "{name}: layer {i} upper");
    }
}

/// Both production collectors preserve the exact cancellation witness. The
/// explicit inward control still demonstrates why an unchecked Big-M is unsafe.
#[test]
fn mip_conv_sound_enclosure_collectors_preserve_the_exact_cancellation_witness() {
    let (network, input) = build_cancellation_conv_fixture();
    let plain = network.collect_ibp_bounds(&input).unwrap();
    let sound = network.collect_ibp_bounds_sound(&input).unwrap();
    let inward_control = (0.0, 0.0);
    assert!(!encloses(inward_control, TRUE_PRE_ACTIVATION));
    for (name, bounds) in [
        ("ordinary collector", &plain),
        ("certified collector", &sound),
    ] {
        let actual = scalar_box(&bounds[0]);
        assert!(
            encloses(actual, TRUE_PRE_ACTIVATION),
            "{name}: {actual:?} must enclose the exact value {TRUE_PRE_ACTIVATION}"
        );
    }
}

/// (b) Both MIP intermediate-bound collectors now return the SOUND box: the
/// single-hidden fast path (malbeware's route) bit-for-bit, the general path
/// bit-for-bit on deadline expiry (its IBP fallback), and enclosing the truth
/// with a live deadline (sound IBP ∩ sound CROWN).
#[test]
fn mip_conv_sound_enclosure_mip_collectors_return_the_sound_box() {
    let (network, input) = build_cancellation_conv_fixture();
    let sound = network.collect_ibp_bounds_sound(&input).unwrap();
    let plain = network.collect_ibp_bounds(&input).unwrap();

    // The fixture takes the SAME route malbeware takes: after unfolding the
    // Conv2d and stripping Flatten it is exactly Linear -> ReLU -> Linear.
    let mip_network = canonicalize_mip_feedforward_network(&network, input.shape()).unwrap();
    assert!(
        is_single_hidden_linear_relu_linear(&mip_network),
        "fixture must match the single-hidden fast path after canonicalization"
    );

    // Both collectors return the typed seam; `as_slice()` is its only view.
    let fast: SoundIntermediateBounds =
        collect_exact_single_hidden_intermediate_bounds(&network, &input).unwrap();
    assert_bit_equal(
        "single-hidden fast path vs collect_ibp_bounds_sound",
        fast.as_slice(),
        &sound,
    );
    assert!(encloses(scalar_box(&plain[0]), TRUE_PRE_ACTIVATION));

    let expired = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    let general_expired: SoundIntermediateBounds =
        collect_mip_intermediate_bounds_with_deadline(&network, &input, expired).unwrap();
    assert_bit_equal(
        "general path (expired deadline) vs collect_ibp_bounds_sound",
        general_expired.as_slice(),
        &sound,
    );

    let general_live =
        collect_mip_intermediate_bounds_with_deadline(&network, &input, None).unwrap();
    let live_box = scalar_box(&general_live.as_slice()[0]);
    assert!(
        encloses(live_box, TRUE_PRE_ACTIVATION),
        "general path (live deadline) box {live_box:?} must enclose {TRUE_PRE_ACTIVATION}"
    );
    assert!(
        live_box.0 < 0.0 && live_box.1 > 0.0,
        "general path (live deadline) box {live_box:?} must leave the ReLU UNSTABLE"
    );

    // Big-M conversion selects the conv output as the only pre-activation
    // vector and carries the sound endpoints verbatim.
    let converted = convert_intermediate_bounds(&fast, &network).unwrap();
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].len(), 1);
    let (lo, hi) = scalar_box(&sound[0]);
    assert_eq!(converted[0][0].lower(), lo);
    assert_eq!(converted[0][0].upper(), hi);
}

/// Encode the fixture with the given Big-M rows (one `Vec<Bound>` per hidden
/// ReLU), add the unsafe region `z >= 0.5`, and run the in-process AY
/// feasibility check.
///
/// Takes the encoder's rows rather than a bound vector because the converter
/// now accepts only `SoundIntermediateBounds`: the sound arms go through
/// `convert_intermediate_bounds`, the negative control builds deliberately inward rows
/// by hand (see the file header).
fn encode_and_check(
    network: &Network,
    input: &BoundedTensor,
    intermediate_bounds_vec: Vec<Vec<Bound>>,
) -> MipResult {
    let mip_network = canonicalize_mip_feedforward_network(network, input.shape()).unwrap();
    let (weights, _structural_biases, layer_dims) =
        extract_linear_relu_params(&mip_network).unwrap();
    let biases = mip_network.exact_biases().to_vec();
    assert_eq!(biases.len(), weights.len());
    let input_bounds = bounded_tensor_to_bounds(input).unwrap();
    let mut encoder = encode_feedforward(
        &weights,
        &biases,
        &layer_dims,
        &input_bounds,
        &intermediate_bounds_vec,
    )
    .unwrap();
    // Unsafe region: the true output z = ReLU(1) = 1 satisfies z >= 0.5, so the
    // property is VIOLATED; a sound encoding must not be infeasible.
    encoder.constrain_output_geq_const(0, 0.5).unwrap();
    let parts = encoder.into_parts();
    let config = MipConfig {
        backend: MipBackend::Ay,
        parallel_split: 1,
        timeout_secs: 30.0,
        ..MipConfig::default()
    };
    MipSolver::new(parts, config)
        .check_feasibility()
        .expect("ay feasibility check should not error")
}

/// A deliberately inward Big-M still yields a certificate for the wrong
/// encoded model. The typed sound collectors must admit the actual witness.
/// Keep the certified-UNSAT negative control independent of production's
/// ordinary collector, which now encloses this cancellation fixture too.
#[test]
fn mip_conv_sound_enclosure_encoded_model_is_not_a_false_unsat() {
    let (network, input) = build_cancellation_conv_fixture();

    // Deliberately supply an inward inactive row. The zero singleton selects
    // the encoder's active (lower >= 0) arm and preserves the exact witness;
    // [-1, 0] instead exercises its inactive arm while still excluding 1.
    // Bypass the typed converter solely to show that an encoded-model
    // certificate cannot replace the original network's enclosure premise.
    let inward_rows = vec![vec![Bound::new(-1.0, 0.0)]];
    let inward_result = encode_and_check(&network, &input, inward_rows);
    match inward_result {
        MipResult::Unsat { certified: true } => {
            let mapped = map_mip_nonsat_result(MipResult::Unsat { certified: true }, 1);
            assert!(
                matches!(mapped, VerificationResult::Verified { .. }),
                "a certified Unsat from the inward box maps to Verified — the minted proof; \
                 got {mapped:?}"
            );
        }
        MipResult::Unsat { certified: false } => panic!(
            "plain-box encoding of a violated property came back Unsat WITHOUT a checked \
             certificate: admission would refuse it, so this arm no longer demonstrates the \
             minted proof — the exposure this test pins is the CERTIFIED lane \
             (check_feasibility runs with require_certificates(true))"
        ),
        other => panic!(
            "plain-box encoding of a violated property must be infeasible (the defect the fix \
             closes); got {other:?} — the fixture no longer exhibits the exposure"
        ),
    }

    // The fix: both MIP collectors hand the encoder the certified box, and
    // only that typed box can pass through `convert_intermediate_bounds`.
    for (name, bounds) in [
        (
            "single-hidden fast path",
            collect_exact_single_hidden_intermediate_bounds(&network, &input).unwrap(),
        ),
        (
            "general path",
            collect_mip_intermediate_bounds_with_deadline(&network, &input, None).unwrap(),
        ),
    ] {
        let rows = convert_intermediate_bounds(&bounds, &network).unwrap();
        let result = encode_and_check(&network, &input, rows);
        assert!(
            !matches!(result, MipResult::Unsat { .. }),
            "{name}: the sound-box encoding must NOT be infeasible on a violated property; got \
             {result:?}"
        );
        if let MipResult::Sat { output_values, .. } = &result {
            assert_eq!(output_values.len(), 1, "{name}: one output");
            assert!(
                (output_values[0] - f64::from(TRUE_PRE_ACTIVATION)).abs() < 1e-6,
                "{name}: the exact solver should reach the true output z = 1, got {:?}",
                output_values
            );
        }
    }
}
