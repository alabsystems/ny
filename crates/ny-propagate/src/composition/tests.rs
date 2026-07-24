// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for multi-network composition (Packet A, #3517).
//! All tests use synthetic BoundedTensor values — no ONNX, no fixture dir.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_core::{MethodUsed, NyError, SoundnessProvenance};
use ny_tensor::BoundedTensor;

use super::certificate::{BoundCertificate, BoundProvenance};
use super::mixer::{compose_linear_mix, MixerSpec};
use super::properties::{check_ducking_snr, check_priority_routing, check_spatial_ild};

fn scalar_bounds(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![lower]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![upper]).unwrap(),
    )
    .unwrap()
}

fn vec_bounds(lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper).unwrap(),
    )
    .unwrap()
}

fn certificate(
    model_id: &str,
    output_bounds: BoundedTensor,
    actual_method: MethodUsed,
) -> BoundCertificate {
    BoundCertificate::try_new(
        model_id,
        output_bounds,
        actual_method,
        SoundnessProvenance::sound(),
    )
    .expect("supported method")
}

#[test]
fn test_bound_certificate_construction_3517() {
    let bounds = vec_bounds(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]);
    let cert = certificate("lead_voice", bounds, MethodUsed::Crown);

    assert_eq!(cert.model_id(), "lead_voice");
    assert_eq!(cert.provenance(), BoundProvenance::Crown);
    assert_eq!(cert.actual_method(), &MethodUsed::Crown);
    assert_eq!(cert.output_bounds().shape(), &[4]);
    assert_eq!(
        cert.output_bounds().lower().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        cert.output_bounds().upper().as_slice().unwrap(),
        &[5.0, 6.0, 7.0, 8.0]
    );

    // Provenance round-trip via clone.
    let p2 = cert.provenance();
    assert_eq!(p2, BoundProvenance::Crown);
}

#[test]
fn test_bound_provenance_maps_supported_method_tags_exhaustively_3920() {
    assert_eq!(
        BoundProvenance::try_from(&MethodUsed::Ibp).unwrap(),
        BoundProvenance::Ibp
    );
    assert_eq!(
        BoundProvenance::try_from(&MethodUsed::IbpF64).unwrap(),
        BoundProvenance::Ibp
    );
    assert_eq!(
        BoundProvenance::try_from(&MethodUsed::Crown).unwrap(),
        BoundProvenance::Crown
    );
    assert_eq!(
        BoundProvenance::try_from(&MethodUsed::CrownF64).unwrap(),
        BoundProvenance::Crown
    );
    assert_eq!(
        BoundProvenance::try_from(&MethodUsed::SdpCrown).unwrap(),
        BoundProvenance::Crown
    );
    assert_eq!(
        BoundProvenance::try_from(&MethodUsed::AlphaCrown).unwrap(),
        BoundProvenance::AlphaCrown
    );

    let mip_error = BoundProvenance::try_from(&MethodUsed::Mip)
        .expect_err("MIP should not masquerade as a bound certificate provenance");
    assert!(
        mip_error.to_string().contains("Mip"),
        "unexpected MIP provenance error: {mip_error}"
    );

    let other_error = BoundProvenance::try_from(&MethodUsed::Other("Custom".to_string()))
        .expect_err("custom methods should fail closed until a chain policy exists");
    assert!(
        other_error.to_string().contains("Custom"),
        "unexpected custom-method provenance error: {other_error}"
    );
}

#[test]
fn test_bound_provenance_rejects_beta_crown_for_packet_a_3920() {
    let error = BoundProvenance::try_from(&MethodUsed::BetaCrown)
        .expect_err("Packet A must fail closed on beta-crown provenance");
    assert!(
        matches!(error, NyError::UnsupportedOp(ref message) if message.contains("BetaCrown")),
        "unexpected beta-crown provenance error: {error}"
    );
}

#[test]
fn test_compose_linear_mix_two_voices_3517() {
    // Voice A: output in [1.0, 3.0], gain [0.8, 1.0], pan (0.7, 0.3)
    // Voice B: output in [0.5, 2.0], gain [0.8, 1.0], pan (0.3, 0.7)
    let cert_a = certificate("voice_a", scalar_bounds(1.0, 3.0), MethodUsed::Crown);
    let cert_b = certificate("voice_b", scalar_bounds(0.5, 2.0), MethodUsed::Ibp);

    let spec = MixerSpec {
        voice_gains: HashMap::from([
            ("voice_a".to_string(), scalar_bounds(0.8, 1.0)),
            ("voice_b".to_string(), scalar_bounds(0.8, 1.0)),
        ]),
        spatial_pan: HashMap::from([
            ("voice_a".to_string(), (0.7, 0.3)),
            ("voice_b".to_string(), (0.3, 0.7)),
        ]),
    };

    let (left, right) = compose_linear_mix(&[cert_a, cert_b], &spec).unwrap();

    // Voice A left: 4-corner of gain [0.8, 1.0] * voice [1.0, 3.0]:
    //   products = {0.8, 2.4, 1.0, 3.0} → min=0.8, max=3.0
    //   left_pan=0.7: [0.7*0.8, 0.7*3.0] = [0.56, 2.1]
    // Voice B left: gain [0.8, 1.0] * voice [0.5, 2.0]:
    //   products = {0.4, 1.6, 0.5, 2.0} → min=0.4, max=2.0
    //   left_pan=0.3: [0.3*0.4, 0.3*2.0] = [0.12, 0.6]
    // Total left: [0.56+0.12, 2.1+0.6] = [0.68, 2.7]
    let left_l = left.lower().as_slice().unwrap()[0];
    let left_u = left.upper().as_slice().unwrap()[0];
    assert!(
        (left_l - 0.68).abs() < 1e-5,
        "left lower: got {left_l}, expected 0.68"
    );
    assert!(
        (left_u - 2.7).abs() < 1e-5,
        "left upper: got {left_u}, expected 2.7"
    );

    // Voice A right: [0.3*0.8, 0.3*3.0] = [0.24, 0.9]
    // Voice B right: [0.7*0.4, 0.7*2.0] = [0.28, 1.4]
    // Total right: [0.24+0.28, 0.9+1.4] = [0.52, 2.3]
    let right_l = right.lower().as_slice().unwrap()[0];
    let right_u = right.upper().as_slice().unwrap()[0];
    assert!(
        (right_l - 0.52).abs() < 1e-5,
        "right lower: got {right_l}, expected 0.52"
    );
    assert!(
        (right_u - 2.3).abs() < 1e-5,
        "right upper: got {right_u}, expected 2.3"
    );
}

#[test]
fn test_compose_linear_mix_negative_pan_preserves_bounds_3517() {
    let cert = certificate("voice_a", scalar_bounds(1.0, 3.0), MethodUsed::Crown);

    let spec = MixerSpec {
        voice_gains: HashMap::from([("voice_a".to_string(), scalar_bounds(0.8, 1.0))]),
        spatial_pan: HashMap::from([("voice_a".to_string(), (-0.7, 0.25))]),
    };

    let (left, right) = compose_linear_mix(&[cert], &spec).unwrap();

    let left_l = left.lower().as_slice().unwrap()[0];
    let left_u = left.upper().as_slice().unwrap()[0];
    assert!(
        (left_l - (-2.1)).abs() < 1e-5,
        "negative pan should flip the product interval lower bound, got {left_l}"
    );
    assert!(
        (left_u - (-0.56)).abs() < 1e-5,
        "negative pan should flip the product interval upper bound, got {left_u}"
    );
    assert!(
        left_l <= left_u,
        "negative pan must not invert the mixed bounds"
    );

    let right_l = right.lower().as_slice().unwrap()[0];
    let right_u = right.upper().as_slice().unwrap()[0];
    assert!(
        (right_l - 0.2).abs() < 1e-5,
        "positive pan should keep the product interval order, got {right_l}"
    );
    assert!(
        (right_u - 0.75).abs() < 1e-5,
        "positive pan should keep the product interval order, got {right_u}"
    );
}

#[test]
fn test_compose_linear_mix_directed_rounding_contains_exact_sum_3517() {
    let cert_a = certificate("voice_a", scalar_bounds(1.0, 1.0), MethodUsed::Crown);
    let cert_b = certificate("voice_b", scalar_bounds(1.0, 1.0), MethodUsed::Crown);

    let spec = MixerSpec {
        voice_gains: HashMap::from([
            ("voice_a".to_string(), scalar_bounds(1.0, 1.0)),
            ("voice_b".to_string(), scalar_bounds(1.0, 1.0)),
        ]),
        spatial_pan: HashMap::from([
            ("voice_a".to_string(), (0.1, 0.0)),
            ("voice_b".to_string(), (0.2, 0.0)),
        ]),
    };

    let (left, _) = compose_linear_mix(&[cert_a, cert_b], &spec).unwrap();
    let left_l = left.lower().as_slice().unwrap()[0] as f64;
    let left_u = left.upper().as_slice().unwrap()[0] as f64;
    let exact_sum = (0.1_f32 as f64) + (0.2_f32 as f64);

    assert!(
        left_l <= exact_sum && exact_sum <= left_u,
        "directed rounding must contain the exact sum: {left_l} <= {exact_sum} <= {left_u}"
    );
    assert!(
        left_l < left_u,
        "non-representable point sum should widen after directed rounding"
    );
}

#[test]
fn test_compose_linear_mix_preserves_large_finite_bounds_3517() {
    let exact = 2.0e20_f32;
    let cert = certificate("voice_a", scalar_bounds(exact, exact), MethodUsed::Crown);

    let spec = MixerSpec {
        voice_gains: HashMap::from([("voice_a".to_string(), scalar_bounds(1.0, 1.0))]),
        spatial_pan: HashMap::from([("voice_a".to_string(), (1.0, 0.0))]),
    };

    let (left, _) = compose_linear_mix(&[cert], &spec).unwrap();
    let left_l = left.lower().as_slice().unwrap()[0];
    let left_u = left.upper().as_slice().unwrap()[0];

    assert!(
        left_l.is_finite(),
        "large finite lower bound should remain finite"
    );
    assert!(
        left_u.is_finite(),
        "large finite upper bound should remain finite"
    );
    assert!(
        left_l <= exact && exact <= left_u,
        "large finite exact value must stay inside the composed interval"
    );
    assert!(
        left_u > 1.0e10_f32,
        "large finite outputs must not be clamped to the old sanitize threshold"
    );
}

#[test]
fn test_compose_linear_mix_preserves_output_shape_3517() {
    let cert = certificate(
        "voice_a",
        BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.5, 2.5]).unwrap(),
        )
        .unwrap(),
        MethodUsed::Crown,
    );

    let spec = MixerSpec {
        voice_gains: HashMap::from([("voice_a".to_string(), scalar_bounds(1.0, 1.0))]),
        spatial_pan: HashMap::from([("voice_a".to_string(), (1.0, 0.0))]),
    };

    let (left, right) = compose_linear_mix(&[cert], &spec).unwrap();
    assert_eq!(left.shape(), &[2, 1]);
    assert_eq!(right.shape(), &[2, 1]);
    let left_lower = left.lower().iter().copied().collect::<Vec<_>>();
    let left_upper = left.upper().iter().copied().collect::<Vec<_>>();
    assert!(
        left_lower[0] <= 1.0 && 1.5 <= left_upper[0],
        "left channel should preserve element 0 bounds within directed rounding"
    );
    assert!(
        left_lower[1] <= 2.0 && 2.5 <= left_upper[1],
        "left channel should preserve element 1 bounds within directed rounding"
    );
}

#[test]
fn test_compose_linear_mix_rejects_mismatched_gain_shape_3517() {
    let cert = certificate(
        "voice_a",
        vec_bounds(vec![1.0, 2.0, 3.0], vec![1.5, 2.5, 3.5]),
        MethodUsed::Crown,
    );

    let spec = MixerSpec {
        voice_gains: HashMap::from([(
            "voice_a".to_string(),
            vec_bounds(vec![0.8, 0.9], vec![1.0, 1.1]),
        )]),
        spatial_pan: HashMap::from([("voice_a".to_string(), (0.5, 0.5))]),
    };

    let err = compose_linear_mix(&[cert], &spec).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("must be scalar or match output shape"),
        "expected gain-shape validation error, got: {message}"
    );
}

#[test]
fn test_compose_linear_mix_rejects_mismatched_voice_shape_3517() {
    let cert_a = certificate(
        "voice_a",
        vec_bounds(vec![1.0, 2.0], vec![1.5, 2.5]),
        MethodUsed::Crown,
    );
    let cert_b = certificate("voice_b", scalar_bounds(0.5, 0.75), MethodUsed::Ibp);

    let spec = MixerSpec {
        voice_gains: HashMap::from([
            ("voice_a".to_string(), scalar_bounds(1.0, 1.0)),
            ("voice_b".to_string(), scalar_bounds(1.0, 1.0)),
        ]),
        spatial_pan: HashMap::from([
            ("voice_a".to_string(), (0.5, 0.5)),
            ("voice_b".to_string(), (0.5, 0.5)),
        ]),
    };

    let err = compose_linear_mix(&[cert_a, cert_b], &spec).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("output shape"),
        "expected voice-shape validation error, got: {message}"
    );
}

#[test]
fn test_priority_routing_lead_above_backing_3517() {
    let lead = scalar_bounds(0.9, 1.0);
    let backing_1 = scalar_bounds(0.1, 0.3);
    let backing_2 = scalar_bounds(0.05, 0.2);

    let result = check_priority_routing(&lead, &[&backing_1, &backing_2]);
    assert!(
        result.verified,
        "lead lower (0.9) > max backing upper (0.3), should verify"
    );
    assert!(
        (result.bound_value - 0.6).abs() < 1e-5,
        "margin should be 0.9 - 0.3 = 0.6"
    );
}

#[test]
fn test_priority_routing_overlap_not_verified_3517() {
    let lead = scalar_bounds(0.2, 0.5);
    let backing = scalar_bounds(0.4, 0.8);

    let result = check_priority_routing(&lead, &[&backing]);
    assert!(
        !result.verified,
        "lead lower (0.2) < backing upper (0.8), should not verify"
    );
}

#[test]
fn test_ducking_snr_above_threshold_3517() {
    // lead [0.8, 1.0], background [0.01, 0.05], threshold 12.0 dB
    // abs_min(lead) = 0.8, abs_max(bg) = 0.05
    // SNR_lower = 20 * log10(0.8 / 0.05) = 20 * log10(16) ≈ 24.08 dB
    let lead = scalar_bounds(0.8, 1.0);
    let background = scalar_bounds(0.01, 0.05);

    let result = check_ducking_snr(&lead, &background, 12.0);
    assert!(result.verified, "SNR ~24 dB should exceed 12 dB threshold");
    let expected_snr = 20.0 * (0.8_f64 / 0.05).log10();
    assert!(
        (result.bound_value - expected_snr).abs() < 0.01,
        "bound_value {} should be ≈ {:.2} dB",
        result.bound_value,
        expected_snr,
    );
}

#[test]
fn test_ducking_snr_lead_contains_zero_3517() {
    // lead [-0.1, 0.5], background [0.01, 0.05], threshold 12.0 dB
    // abs_min(lead) = 0 (range contains zero), so SNR = -∞
    let lead = scalar_bounds(-0.1, 0.5);
    let background = scalar_bounds(0.01, 0.05);

    let result = check_ducking_snr(&lead, &background, 12.0);
    assert!(
        !result.verified,
        "lead contains zero → abs_min=0 → SNR=-∞, should not verify"
    );
    assert!(
        result.bound_value == f64::NEG_INFINITY,
        "SNR should be -∞ when lead range contains zero"
    );
}

#[test]
fn test_spatial_ild_separated_voices_3517() {
    // Voice A panned hard left (0.9, 0.1), voice B panned hard right (0.1, 0.9).
    // Both with power bounds [0.5, 1.0].
    //
    // At left ear:
    //   level A = 0.9 * [0.5, 1.0] → min=0.45, max=0.9
    //   level B = 0.1 * [0.5, 1.0] → min=0.05, max=0.1
    //   ILD (A over B) = 20 * log10(0.45 / 0.1) = 20 * log10(4.5) ≈ 13.06 dB
    //
    // Threshold 6.0 dB → verified=true
    let voice_a = scalar_bounds(0.5, 1.0);
    let voice_b = scalar_bounds(0.5, 1.0);

    let result = check_spatial_ild(
        &voice_a,
        &voice_b,
        (0.9, 0.1), // A panned left
        (0.1, 0.9), // B panned right
        6.0,
    );
    assert!(
        result.verified,
        "well-separated panning should exceed 6 dB threshold, got {:.2} dB",
        result.bound_value
    );
    let expected_ild = 20.0 * (0.45_f64 / 0.1).log10();
    assert!(
        (result.bound_value - expected_ild).abs() < 0.01,
        "ILD should be ≈ {:.2} dB, got {:.2} dB",
        expected_ild,
        result.bound_value
    );
}

#[test]
fn test_spatial_ild_colocated_not_separated_3517() {
    // Both voices at center pan (0.5, 0.5), same power [0.5, 1.0].
    //
    // At left ear:
    //   level A = 0.5 * [0.5, 1.0] → min=0.25, max=0.5
    //   level B = 0.5 * [0.5, 1.0] → min=0.25, max=0.5
    //   ILD (A over B) = 20 * log10(0.25 / 0.5) ≈ -6.02 dB
    //   ILD (B over A) = 20 * log10(0.25 / 0.5) ≈ -6.02 dB
    //
    // Neither voice dominates → cannot verify spatial separation.
    let voice_a = scalar_bounds(0.5, 1.0);
    let voice_b = scalar_bounds(0.5, 1.0);

    let result = check_spatial_ild(
        &voice_a,
        &voice_b,
        (0.5, 0.5), // A at center
        (0.5, 0.5), // B at center
        3.0,
    );
    assert!(
        !result.verified,
        "co-located voices should not verify spatial separation, got {:.2} dB",
        result.bound_value
    );
}
