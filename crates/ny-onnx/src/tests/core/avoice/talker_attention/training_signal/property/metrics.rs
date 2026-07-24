// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Direct spec-guided CROWN oracle and metric parity contract helpers.
//!
//! Validates that Packet C's recorded property metrics match the direct
//! spec-guided propagation result.
//!
//! Part of #4089 property lane decomposition.

use super::fixture::TalkerPropertySweepFixture;
use super::*;

pub(super) fn assert_property_provenance_and_metrics(record: &WeakRegionRecord) {
    assert_eq!(record.method_requested, "spec_guided_crown");
    assert!(
        record.method_actual == "spec_guided_crown" || record.method_actual == "forward_fallback",
        "unexpected method_actual: {}",
        record.method_actual
    );
    match record.provenance {
        BoundsProvenance::Crown => assert_eq!(record.method_actual, "spec_guided_crown"),
        BoundsProvenance::ForwardFallback(_) => {
            assert_eq!(record.method_actual, "forward_fallback")
        }
    }

    let slack = record
        .certified_slack_min
        .expect("certified_slack_min must be Some for property objective");
    assert!(slack.is_finite(), "certified_slack_min not finite: {slack}");

    let obj_max = record
        .objective_width_max
        .expect("objective_width_max must be Some for property objective");
    assert!(
        obj_max.is_finite(),
        "objective_width_max not finite: {obj_max}"
    );

    let obj_mean = record
        .objective_width_mean
        .expect("objective_width_mean must be Some for property objective");
    assert!(
        obj_mean.is_finite(),
        "objective_width_mean not finite: {obj_mean}"
    );

    assert!(
        record.output_width_max.is_finite(),
        "output_width_max must be finite, got {}",
        record.output_width_max
    );
}

pub(super) fn direct_property_metrics(
    fixture: &TalkerPropertySweepFixture,
) -> (BoundsProvenance, f32, f32, f32) {
    let direct_spec = fixture
        .graph
        .propagate_crown_with_specs_and_provenance_and_engine_with_node_bounds_and_deadline(
            &fixture.input,
            &fixture.spec_matrix,
            None,
            &fixture.node_bounds,
            None,
        )
        .expect("direct spec-guided CROWN should succeed on short-seq talker");
    let direct_slack = direct_spec
        .bounds
        .lower()
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let mut direct_obj_max = f32::NEG_INFINITY;
    let mut direct_width_total = 0.0_f32;
    let mut direct_width_count = 0usize;
    for (&upper, &lower) in direct_spec
        .bounds
        .upper()
        .iter()
        .zip(direct_spec.bounds.lower().iter())
    {
        let width = upper - lower;
        direct_obj_max = direct_obj_max.max(width);
        direct_width_total += width;
        direct_width_count += 1;
    }
    assert!(
        direct_width_count > 0,
        "direct property oracle must produce at least one spec bound"
    );
    (
        direct_spec.provenance,
        direct_slack,
        direct_obj_max,
        direct_width_total / direct_width_count as f32,
    )
}

pub(super) fn assert_property_record_matches_direct_metrics(
    record: &WeakRegionRecord,
    direct_provenance: BoundsProvenance,
    direct_slack: f32,
    direct_obj_max: f32,
    direct_obj_mean: f32,
) {
    assert_eq!(
        record.provenance, direct_provenance,
        "report provenance must match the direct spec-guided propagation result"
    );
    let expected_method_actual = match direct_provenance {
        BoundsProvenance::Crown => "spec_guided_crown",
        BoundsProvenance::ForwardFallback(_) => "forward_fallback",
    };
    assert_eq!(
        record.method_actual, expected_method_actual,
        "method_actual must reflect the direct propagation provenance"
    );

    let recorded_slack = record
        .certified_slack_min
        .expect("certified_slack_min must be populated for property objective");
    let recorded_obj_max = record
        .objective_width_max
        .expect("objective_width_max must be populated for property objective");
    let recorded_obj_mean = record
        .objective_width_mean
        .expect("objective_width_mean must be populated for property objective");

    assert!(
        (recorded_slack - direct_slack).abs() <= 1e-5,
        "certified_slack_min must equal the direct min spec lower bound: record={recorded_slack}, direct={direct_slack}"
    );
    assert!(
        (recorded_obj_max - direct_obj_max).abs() <= 1e-5,
        "objective_width_max must equal the direct max spec width: record={recorded_obj_max}, direct={direct_obj_max}"
    );
    assert!(
        (recorded_obj_mean - direct_obj_mean).abs() <= 1e-5,
        "objective_width_mean must equal the direct mean spec width: record={recorded_obj_mean}, direct={direct_obj_mean}"
    );
}
