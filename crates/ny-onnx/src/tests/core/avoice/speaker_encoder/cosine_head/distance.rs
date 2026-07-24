// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::shared::{SPEAKER_DISTANCE_ACCEPTANCE_UPPER, VACUOUS_COSINE_DISTANCE_UPPER};

pub(in super::super) fn speaker_cosine_distance_upper(
    dot_lower: f32,
    norm_sq_upper: f32,
) -> (f32, bool) {
    if dot_lower > 0.0 && norm_sq_upper > 0.0 {
        (
            1.0 - (dot_lower as f64 / (norm_sq_upper as f64).sqrt()) as f32,
            true,
        )
    } else {
        (VACUOUS_COSINE_DISTANCE_UPPER, false)
    }
}

pub(super) fn assert_speaker_cosine_distance_bound(
    dot_lower: f32,
    dot_upper: f32,
    norm_sq_lower: f32,
    norm_sq_upper: f32,
) {
    eprintln!(
        "CROWN results: dot=[{dot_lower}, {dot_upper}], \
         norm_sq=[{norm_sq_lower}, {norm_sq_upper}]"
    );

    let (distance_upper, has_nonvacuous_distance_bound) =
        speaker_cosine_distance_upper(dot_lower, norm_sq_upper);
    eprintln!(
        "CROWN results: dot=[{dot_lower}, {dot_upper}], \
         norm_sq=[{norm_sq_lower}, {norm_sq_upper}], \
         distance_upper={distance_upper}"
    );
    assert!(
        distance_upper.is_finite(),
        "cosine distance upper should be finite, got {distance_upper}"
    );
    if has_nonvacuous_distance_bound {
        assert!(
            distance_upper < SPEAKER_DISTANCE_ACCEPTANCE_UPPER,
            "speaker cosine distance upper should meet #3499 acceptance (< {}), got {}; \
             dot=[{}, {}], norm_sq=[{}, {}]",
            SPEAKER_DISTANCE_ACCEPTANCE_UPPER,
            distance_upper,
            dot_lower,
            dot_upper,
            norm_sq_lower,
            norm_sq_upper
        );
    } else {
        assert_eq!(
            distance_upper, VACUOUS_COSINE_DISTANCE_UPPER,
            "speaker cosine distance should use the explicit vacuous sentinel while \
             the remaining speaker proof surface stays vacuous"
        );
    }
}

#[test]
fn test_speaker_cosine_distance_upper_returns_vacuous_sentinel_for_nonpositive_dot_3499() {
    let (distance_upper, is_nonvacuous) = speaker_cosine_distance_upper(-0.25, 4.0);

    assert!(!is_nonvacuous, "nonpositive dot lower should stay vacuous");
    assert_eq!(distance_upper, VACUOUS_COSINE_DISTANCE_UPPER);
}

#[test]
fn test_speaker_cosine_distance_upper_returns_acceptance_candidate_for_positive_components_3499() {
    let (distance_upper, is_nonvacuous) = speaker_cosine_distance_upper(1.9, 4.0);

    assert!(
        is_nonvacuous,
        "positive dot/norm components should be non-vacuous"
    );
    assert!(
        (distance_upper - 0.05).abs() < 1e-6,
        "expected analytic cosine distance upper of 0.05, got {distance_upper}"
    );
}
