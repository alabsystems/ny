// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common;
use super::*;
use crate::tests::fixtures::optional_test_model;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::layers::{ReduceSumLayer, SigmoidLayer};
use ny_propagate::BoundPropagation;

pub(super) const KOKORO_DURATION_BUCKETS: usize = 50;
pub(super) const KOKORO_DURATION_MIN_FRAMES: f32 = 1.0;
const KOKORO_DURATION_MAX_FRAMES: f32 = 50.0;
pub(super) const KOKORO_DEFAULT_SPEED: f32 = 1.0;

// ---------------------------------------------------------------------------
// Contract accessor boundary (#3917)
//
// Centralizes exporter-owned duration head/bin/frame semantics behind a cached,
// fallback-safe accessor. `KOKORO_REAL_DURATION_SEQUENCE_LEN = 4` in
// `real_export.rs` stays local — it is a proof-budget constant, not sidecar
// metadata.
// ---------------------------------------------------------------------------

pub(super) struct KokoroDurationFixtureContract {
    pub duration_head: &'static str,
    pub duration_bin_count: usize,
    pub min_duration_frames: f32,
    pub max_duration_frames: f32,
}

fn default_duration_fixture_contract() -> KokoroDurationFixtureContract {
    KokoroDurationFixtureContract {
        duration_head: "independent_sigmoid_sum",
        duration_bin_count: KOKORO_DURATION_BUCKETS,
        min_duration_frames: KOKORO_DURATION_MIN_FRAMES,
        max_duration_frames: KOKORO_DURATION_MAX_FRAMES,
    }
}

/// Return the cached duration-predictor fixture contract.
///
/// Consults `load_avoice_contract()` once, validates the sidecar if present,
/// and falls back to the current local constants when no sidecar exists.
pub(super) fn kokoro_duration_fixture_contract() -> &'static KokoroDurationFixtureContract {
    use std::sync::OnceLock;
    static CONTRACT: OnceLock<KokoroDurationFixtureContract> = OnceLock::new();
    CONTRACT.get_or_init(|| {
        if let Some(model_path) = optional_test_model("kokoro_duration_predictor.onnx") {
            match load_avoice_contract(&model_path) {
                Ok(Some(contract)) => {
                    let constraints = &contract.constraints;
                    KokoroDurationFixtureContract {
                        duration_head: match constraints.duration_head.as_deref() {
                            Some("independent_sigmoid_sum") | None => "independent_sigmoid_sum",
                            Some(other) => panic!(
                                "duration predictor sidecar duration_head must be \
                                 'independent_sigmoid_sum', got '{other}' at {model_path:?}"
                            ),
                        },
                        duration_bin_count: constraints
                            .duration_bin_count
                            .unwrap_or(KOKORO_DURATION_BUCKETS),
                        min_duration_frames: constraints
                            .min_duration_frames
                            .unwrap_or(KOKORO_DURATION_MIN_FRAMES),
                        max_duration_frames: constraints
                            .max_duration_frames
                            .unwrap_or(KOKORO_DURATION_MAX_FRAMES),
                    }
                }
                Ok(None) => default_duration_fixture_contract(),
                Err(e) => {
                    panic!(
                        "failed to load duration predictor contract sidecar at {model_path:?}: {e}"
                    )
                }
            }
        } else {
            default_duration_fixture_contract()
        }
    })
}

/// Convert Kokoro duration logits `[B, T, 50]` into expected durations `[B, T]`.
///
/// The exported `kokoro_duration_predictor.onnx` surface emits the final
/// duration logits directly. Kokoro treats those 50 logits as *independent*
/// Bernoulli bins and computes durations with `sigmoid(logits).sum(-1)`, not a
/// categorical softmax expectation over bucket indices.
///
/// Sources:
/// - `./avoice/scripts/export_kokoro_onnx.py` (`duration_logits [1, T, 50]`)
/// - `./avoice/crates/avoice-tts/src/kokoro/model_ops.rs`
pub(super) fn kokoro_expected_duration_bounds_from_logits(logits: &BoundedTensor) -> BoundedTensor {
    let probs = SigmoidLayer::new()
        .propagate_ibp(logits)
        .expect("Kokoro expected-duration sigmoid head should propagate via IBP");
    ReduceSumLayer::new(vec![-1], false)
        .propagate_ibp(&probs)
        .expect("Kokoro expected-duration sum head should propagate via IBP")
}

/// Apply the avoice production `duration_to_counts` post-processing to
/// duration logits.
///
/// Production path:
///   `sigmoid(logits).sum(-1) / speed`, with invalid `speed` normalized to
///   `1.0`, then clamped to `[1, 50]`.
///
/// The clamp is monotone increasing, so interval endpoints can be clamped
/// independently without losing soundness.
///
/// Sources:
/// - `./avoice/crates/avoice-tts/src/kokoro/model_ops.rs`
/// - `./avoice/scripts/debug_prosody_forward.py`
pub(super) fn kokoro_duration_count_bounds_from_logits(
    logits: &BoundedTensor,
    speed: f32,
) -> BoundedTensor {
    let speed = if speed.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        KOKORO_DEFAULT_SPEED
    } else {
        speed
    };

    let contract = kokoro_duration_fixture_contract();
    let expected_duration = kokoro_expected_duration_bounds_from_logits(logits);
    let scaled_duration = expected_duration.scale(1.0 / speed);
    let lower = scaled_duration
        .lower()
        .mapv(|value| value.clamp(contract.min_duration_frames, contract.max_duration_frames));
    let upper = scaled_duration
        .upper()
        .mapv(|value| value.clamp(contract.min_duration_frames, contract.max_duration_frames));
    BoundedTensor::new(lower, upper)
        .expect("Kokoro duration count clamp should preserve finite, ordered bounds")
}

// ---------------------------------------------------------------------------
// Synthetic proof head tests (no ONNX model required)
// ---------------------------------------------------------------------------

#[test]
fn test_kokoro_expected_duration_zero_logits_uses_sigmoid_sum_3497() {
    let shape = IxDyn(&[1, 2, KOKORO_DURATION_BUCKETS]);
    let logits = ArrayD::zeros(shape);
    let bounds = BoundedTensor::new(logits.clone(), logits).expect("point logits bounds");

    let expected_duration = kokoro_expected_duration_bounds_from_logits(&bounds);

    assert_eq!(
        expected_duration.lower().shape(),
        &[1, 2],
        "expected-duration head should reduce the 50-bin axis"
    );
    assert_eq!(
        expected_duration.upper().shape(),
        &[1, 2],
        "expected-duration upper shape should match lower shape"
    );

    for (idx, (&lower, &upper)) in expected_duration
        .lower()
        .iter()
        .zip(expected_duration.upper().iter())
        .enumerate()
    {
        assert!(
            (lower - 25.0).abs() < 1e-4,
            "zero logits should yield 50 * sigmoid(0) = 25.0 at row {idx}, got {lower}"
        );
        assert!(
            (upper - 25.0).abs() < 1e-4,
            "zero logits should yield a point interval at row {idx}, got {upper}"
        );
    }
}

#[test]
fn test_kokoro_expected_duration_positive_lower_bound_for_finite_logits_3497() {
    let lower = ArrayD::from_elem(IxDyn(&[1, 3, KOKORO_DURATION_BUCKETS]), -10.0);
    let upper = ArrayD::from_elem(IxDyn(&[1, 3, KOKORO_DURATION_BUCKETS]), -5.0);
    let logits_bounds =
        BoundedTensor::new(lower, upper).expect("finite Kokoro duration logits bounds");

    let expected_duration = kokoro_expected_duration_bounds_from_logits(&logits_bounds);

    common::assert_finite_and_ordered(
        &expected_duration,
        "Kokoro expected-duration Bernoulli-sum bounds",
    );
    for (idx, &lower) in expected_duration.lower().iter().enumerate() {
        assert!(
            lower > 0.0,
            "finite duration logits should keep a strictly positive expected-duration lower bound at row {idx}, got {lower}"
        );
    }
    for (idx, &upper) in expected_duration.upper().iter().enumerate() {
        assert!(
            upper < KOKORO_DURATION_BUCKETS as f32,
            "expected duration should stay below the 50-bin ceiling at row {idx}, got {upper}"
        );
    }
}

#[test]
fn test_kokoro_duration_counts_speed_guard_and_scaling_3497() {
    let shape = IxDyn(&[1, 2, KOKORO_DURATION_BUCKETS]);
    let logits = ArrayD::zeros(shape);
    let bounds = BoundedTensor::new(logits.clone(), logits).expect("point logits bounds");

    let default_counts = kokoro_duration_count_bounds_from_logits(&bounds, KOKORO_DEFAULT_SPEED);
    let faster_counts = kokoro_duration_count_bounds_from_logits(&bounds, 2.0);
    let invalid_speed_counts = kokoro_duration_count_bounds_from_logits(&bounds, 0.0);

    for (idx, ((&default_lower, &faster_lower), &invalid_lower)) in default_counts
        .lower()
        .iter()
        .zip(faster_counts.lower().iter())
        .zip(invalid_speed_counts.lower().iter())
        .enumerate()
    {
        assert!(
            (default_lower - 25.0).abs() < 1e-4,
            "zero logits at speed=1 should yield 25.0 continuous frames at row {idx}, got {default_lower}"
        );
        assert!(
            (faster_lower - 12.5).abs() < 1e-4,
            "zero logits at speed=2 should halve the continuous duration count at row {idx}, got {faster_lower}"
        );
        assert!(
            (invalid_lower - default_lower).abs() < 1e-6,
            "non-positive speed should normalize to 1.0 at row {idx}: invalid={invalid_lower}, default={default_lower}"
        );
    }
}

#[test]
fn test_kokoro_duration_counts_clamp_to_production_frame_range_3497() {
    let mut logits = ArrayD::from_elem(IxDyn(&[1, 2, KOKORO_DURATION_BUCKETS]), -20.0);
    logits.index_axis_mut(ndarray::Axis(1), 1).fill(20.0);
    let bounds = BoundedTensor::new(logits.clone(), logits).expect("point logits bounds");

    let duration_counts = kokoro_duration_count_bounds_from_logits(&bounds, 0.5);

    let lower = duration_counts.lower();
    let upper = duration_counts.upper();
    assert!(
        (lower[[0, 0]] - KOKORO_DURATION_MIN_FRAMES).abs() < 1e-4
            && (upper[[0, 0]] - KOKORO_DURATION_MIN_FRAMES).abs() < 1e-4,
        "very negative logits should clamp to the 1-frame minimum, got [{}, {}]",
        lower[[0, 0]],
        upper[[0, 0]]
    );
    assert!(
        (lower[[0, 1]] - KOKORO_DURATION_BUCKETS as f32).abs() < 1e-4
            && (upper[[0, 1]] - KOKORO_DURATION_BUCKETS as f32).abs() < 1e-4,
        "scaled high-probability durations should clamp to the 50-frame maximum, got [{}, {}]",
        lower[[0, 1]],
        upper[[0, 1]]
    );
}

/// Assert positive expected-duration property on a `BoundedTensor`.
pub(super) fn assert_positive_expected_durations(durations: &BoundedTensor) {
    common::assert_finite_and_ordered(durations, "expected-duration bounds");
    for (idx, &lower) in durations.lower().iter().enumerate() {
        assert!(
            lower > 0.0,
            "expected duration lower bound must be strictly positive at row {idx}, got {lower}"
        );
    }
    for (idx, &upper) in durations.upper().iter().enumerate() {
        assert!(
            upper <= KOKORO_DURATION_BUCKETS as f32 + 1e-4,
            "expected duration upper bound must stay at or below {KOKORO_DURATION_BUCKETS} \
             (allowing float32 sigmoid saturation) at row {idx}, got {upper}"
        );
    }
}

/// Assert the production-aligned continuous duration-count bounds.
pub(super) fn assert_production_duration_counts(durations: &BoundedTensor) {
    common::assert_finite_and_ordered(durations, "production duration count bounds");
    for (idx, &lower) in durations.lower().iter().enumerate() {
        assert!(
            lower >= KOKORO_DURATION_MIN_FRAMES,
            "duration count lower bound must stay at or above {} frame(s) at row {idx}, got {lower}",
            KOKORO_DURATION_MIN_FRAMES
        );
    }
    for (idx, &upper) in durations.upper().iter().enumerate() {
        assert!(
            upper <= KOKORO_DURATION_BUCKETS as f32,
            "duration count upper bound must stay at or below {KOKORO_DURATION_BUCKETS} frames at row {idx}, got {upper}"
        );
    }
}

pub(super) fn avg_bound_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(lo, hi)| hi - lo)
        .sum::<f32>()
        / bounds.lower().len() as f32
}

// ---------------------------------------------------------------------------
// Fallback regression test (#3917)
// ---------------------------------------------------------------------------

#[test]
fn test_kokoro_duration_fixture_contract_fallback_matches_constants_3917() {
    let contract = kokoro_duration_fixture_contract();
    assert_eq!(
        contract.duration_head, "independent_sigmoid_sum",
        "fallback duration_head should be independent_sigmoid_sum"
    );
    assert_eq!(
        contract.duration_bin_count, KOKORO_DURATION_BUCKETS,
        "fallback duration_bin_count should match KOKORO_DURATION_BUCKETS"
    );
    assert!(
        (contract.min_duration_frames - KOKORO_DURATION_MIN_FRAMES).abs() < f32::EPSILON,
        "fallback min_duration_frames should match KOKORO_DURATION_MIN_FRAMES"
    );
    assert!(
        (contract.max_duration_frames - KOKORO_DURATION_MAX_FRAMES).abs() < f32::EPSILON,
        "fallback max_duration_frames should match KOKORO_DURATION_MAX_FRAMES"
    );
}
