// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common;
use super::proof_head::{
    assert_positive_expected_durations, assert_production_duration_counts, avg_bound_width,
    kokoro_duration_count_bounds_from_logits, kokoro_expected_duration_bounds_from_logits,
    KOKORO_DEFAULT_SPEED, KOKORO_DURATION_BUCKETS,
};
use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::types::BoundsProvenance;
use ny_propagate::GraphNetwork;

/// Load the surrogate ONNX model and convert to a graph.
///
/// The surrogate is a feed-forward model (MatMul+Add) with the same
/// input/output contract as the real Kokoro duration predictor:
///   `encoded_features [1, T, D]` → `duration_logits [1, T, 50]`
///
/// The real model uses BiLSTM (supported after #3497 LSTM unrolling work).
/// The feed-forward surrogate is retained for fast regression.
pub(super) fn load_surrogate_graph() -> (OnnxModel, GraphNetwork) {
    let path = require_test_model("kokoro_duration_predictor_surrogate.onnx");
    let model = load_onnx(&path).expect("duration predictor surrogate should load");
    let graph = model
        .to_graph_network()
        .expect("duration predictor surrogate should convert to GraphNetwork");
    (model, graph)
}

/// Run IBP through the surrogate graph and apply the proof head.
fn surrogate_ibp_expected_durations(
    model: &OnnxModel,
    graph: &GraphNetwork,
    epsilon: f32,
) -> BoundedTensor {
    let input_spec = common::input_spec_by_name(model, "encoded_features");
    let shape = common::unbatched_shape_from_input_spec(input_spec, 4, "encoded_features");
    let center = ArrayD::zeros(IxDyn(&shape));
    let input = BoundedTensor::from_epsilon(center, epsilon).expect("valid epsilon ball");
    let logits = graph.propagate_ibp(&input).expect("IBP should succeed");
    kokoro_expected_duration_bounds_from_logits(&logits)
}

#[cfg_attr(not(debug_assertions), ntest::timeout(60000))]
#[test]
fn test_duration_predictor_surrogate_graph_ibp_positive_expected_duration_3497() {
    let (model, graph) = load_surrogate_graph();

    // Verify input/output shape contract
    assert_eq!(model.network.inputs.len(), 1, "single activation input");
    let input_spec = common::input_spec_by_name(&model, "encoded_features");
    assert_eq!(
        input_spec.shape.len(),
        3,
        "encoded_features should be 3D [B, T, D]"
    );

    // Run IBP → proof head → check logit shape
    let logits = graph
        .propagate_ibp(
            &BoundedTensor::from_epsilon(
                ArrayD::zeros(IxDyn(&common::unbatched_shape_from_input_spec(
                    input_spec,
                    4,
                    "encoded_features",
                ))),
                1e-3,
            )
            .expect("epsilon ball"),
        )
        .expect("IBP");
    common::assert_finite_and_ordered(&logits, "duration predictor logit bounds");
    assert_eq!(
        logits.lower().shape().last().copied(),
        Some(KOKORO_DURATION_BUCKETS)
    );

    // Verify the non-vacuous Bernoulli-sum property and the production count surface.
    let expected_durations = kokoro_expected_duration_bounds_from_logits(&logits);
    assert_positive_expected_durations(&expected_durations);
    let duration_counts = kokoro_duration_count_bounds_from_logits(&logits, KOKORO_DEFAULT_SPEED);
    assert_production_duration_counts(&duration_counts);

    eprintln!("--- Duration predictor surrogate IBP results ---");
    eprintln!(
        "  avg expected-duration width: {:.6}",
        avg_bound_width(&expected_durations)
    );
    eprintln!(
        "  avg production-count width: {:.6}",
        avg_bound_width(&duration_counts)
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(60000))]
#[test]
fn test_duration_predictor_surrogate_tighter_bounds_at_smaller_epsilon_3497() {
    let (model, graph) = load_surrogate_graph();

    // Measure bound width at two epsilon levels
    let widths: Vec<f32> = [1e-2, 1e-4]
        .iter()
        .map(|&eps| {
            let durations = surrogate_ibp_expected_durations(&model, &graph, eps);
            avg_bound_width(&durations)
        })
        .collect();

    // Tighter input region (smaller epsilon) should produce tighter bounds
    assert!(
        widths[1] < widths[0],
        "smaller epsilon should yield tighter duration bounds: eps=1e-2 width={:.6}, eps=1e-4 width={:.6}",
        widths[0],
        widths[1]
    );
}

// ---------------------------------------------------------------------------
// CROWN backward verification (ONNX → graph → CROWN → proof head → property)
//
// The surrogate is a feed-forward network (MatMul+Add → logits), which is
// purely linear. CROWN backward through a linear network should produce bounds
// at least as tight as IBP. The external sigmoid+sum proof head then benefits
// from tighter logit bounds.
//
// This validates that the CROWN pipeline works end-to-end through the duration
// predictor graph and the external Bernoulli-sum proof head.
//
// Sources:
// - crates/ny-onnx/src/tests/core/avoice/talker/crown_boundary.rs (pattern)
// - alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/abcrown.py (CROWN backward reference)
// ---------------------------------------------------------------------------

/// Run CROWN backward through the surrogate graph.
///
/// Returns the raw logit bounds and provenance. The caller applies the
/// external sigmoid+sum proof head to get expected-duration bounds.
fn surrogate_crown_logits(
    model: &OnnxModel,
    graph: &GraphNetwork,
    epsilon: f32,
) -> (BoundedTensor, BoundsProvenance) {
    let input_spec = common::input_spec_by_name(model, "encoded_features");
    let shape = common::unbatched_shape_from_input_spec(input_spec, 4, "encoded_features");
    let center = ArrayD::zeros(IxDyn(&shape));
    let input = BoundedTensor::from_epsilon(center, epsilon).expect("valid epsilon ball");
    let crown_result = graph
        .propagate_crown_with_provenance(&input)
        .expect("CROWN backward should succeed on linear surrogate");
    (crown_result.bounds, crown_result.provenance)
}

/// CROWN backward on the surrogate produces valid, non-vacuous expected
/// durations via the sigmoid+sum proof head.
///
/// Since the surrogate is purely linear (MatMul+Add), CROWN backward should
/// stay on the backward path (not fall back to IBP) and produce bounds that
/// are at least as tight as IBP.
#[cfg_attr(not(debug_assertions), ntest::timeout(60000))]
#[test]
fn test_duration_predictor_surrogate_crown_positive_expected_duration_3497() {
    let (model, graph) = load_surrogate_graph();
    let epsilon = 1e-3;

    let (crown_logits, provenance) = surrogate_crown_logits(&model, &graph, epsilon);

    assert_eq!(
        provenance,
        BoundsProvenance::Crown,
        "linear surrogate should use CROWN backward, not fall back to IBP"
    );

    let crown_durations = kokoro_expected_duration_bounds_from_logits(&crown_logits);
    assert_positive_expected_durations(&crown_durations);

    let crown_counts =
        kokoro_duration_count_bounds_from_logits(&crown_logits, KOKORO_DEFAULT_SPEED);
    assert_production_duration_counts(&crown_counts);

    eprintln!("--- Duration predictor surrogate CROWN results ---");
    eprintln!("  provenance: {provenance:?}");
    eprintln!(
        "  avg expected-duration width: {:.6}",
        avg_bound_width(&crown_durations)
    );
    eprintln!(
        "  avg production-count width: {:.6}",
        avg_bound_width(&crown_counts)
    );
}

/// CROWN bounds on the surrogate are no looser than IBP bounds.
///
/// For a purely linear network, CROWN should produce identical or tighter
/// bounds. This test compares the two methods side by side on the same
/// input and asserts that CROWN widths do not exceed IBP widths.
#[cfg_attr(not(debug_assertions), ntest::timeout(60000))]
#[test]
fn test_duration_predictor_surrogate_crown_no_looser_than_ibp_3497() {
    let (model, graph) = load_surrogate_graph();
    let epsilon = 1e-3;

    let ibp_durations = surrogate_ibp_expected_durations(&model, &graph, epsilon);
    let (crown_logits, provenance) = surrogate_crown_logits(&model, &graph, epsilon);
    let crown_durations = kokoro_expected_duration_bounds_from_logits(&crown_logits);

    assert_eq!(
        provenance,
        BoundsProvenance::Crown,
        "linear surrogate should stay on the CROWN path"
    );

    let ibp_width = avg_bound_width(&ibp_durations);
    let crown_width = avg_bound_width(&crown_durations);

    // CROWN should produce no-looser bounds per dimension.
    for (idx, ((&ibp_lo, &ibp_hi), (&crown_lo, &crown_hi))) in ibp_durations
        .lower()
        .iter()
        .zip(ibp_durations.upper().iter())
        .zip(
            crown_durations
                .lower()
                .iter()
                .zip(crown_durations.upper().iter()),
        )
        .enumerate()
    {
        assert!(
            crown_lo >= ibp_lo - 1e-5,
            "CROWN lower bound should be >= IBP lower at dim {idx}: crown={crown_lo}, ibp={ibp_lo}"
        );
        assert!(
            crown_hi <= ibp_hi + 1e-5,
            "CROWN upper bound should be <= IBP upper at dim {idx}: crown={crown_hi}, ibp={ibp_hi}"
        );
    }

    eprintln!("--- IBP vs CROWN comparison ---");
    eprintln!("  IBP avg width:   {ibp_width:.6}");
    eprintln!("  CROWN avg width: {crown_width:.6}");
    eprintln!(
        "  tightening: {:.2}%",
        if ibp_width > 0.0 {
            (1.0 - crown_width / ibp_width) * 100.0
        } else {
            0.0
        }
    );
}
