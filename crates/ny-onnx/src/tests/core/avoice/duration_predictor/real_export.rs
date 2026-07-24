// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common;
use super::bilstm::assert_crown_no_looser_than_ibp_durations;
use super::proof_head::{
    assert_positive_expected_durations, assert_production_duration_counts, avg_bound_width,
    kokoro_duration_count_bounds_from_logits, kokoro_duration_fixture_contract,
    kokoro_expected_duration_bounds_from_logits, KOKORO_DEFAULT_SPEED,
};
use super::*;
use crate::onnx_proto::ModelProto;
use crate::test_fixtures::specialize_kokoro_duration_predictor_for_lstm_unroll;
use prost::Message;

const KOKORO_REAL_DURATION_PREDICTOR_FILE: &str = "kokoro_duration_predictor.onnx";
const KOKORO_REAL_DURATION_SEQUENCE_LEN: usize = 4;
const KOKORO_REAL_DURATION_EPSILON: f32 = 1e-3;

/// Epsilon values for the non-vacuous sweep, ordered largest-to-smallest.
/// The real Kokoro BiLSTM (640→320 hidden, trained weights) amplifies the
/// input perturbation significantly more than the toy surrogate (8→4).
const KOKORO_REAL_EPSILON_SWEEP: &[f32] = &[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8];

/// Load the real Kokoro duration predictor export and convert to a graph.
///
/// The artifact is discovered through the shared avoice fixture contract:
/// repo-local `tests/models/`, `NY_TEST_MODELS_DIR`, or the standard
/// `~/avoice/models/kokoro-v1_0/onnx/` handoff path.
///
/// The current LSTM lowering path requires a concrete sequence length, so the
/// test harness specializes the symbolic export dimension `T` to the short
/// sequence length used by this proof lane before invoking `load_onnx_bytes`.
fn load_real_duration_predictor_graph() -> (OnnxModel, GraphNetwork) {
    let path =
        require_test_model_with_hint(KOKORO_REAL_DURATION_PREDICTOR_FILE, AVOICE_TEST_MODEL_HINT);
    let bytes = std::fs::read(&path).expect("real Kokoro duration predictor bytes should read");
    let mut proto =
        ModelProto::decode(bytes.as_slice()).expect("real Kokoro duration predictor proto");
    specialize_kokoro_duration_predictor_for_lstm_unroll(
        &mut proto,
        KOKORO_REAL_DURATION_SEQUENCE_LEN as i64,
    );
    let model = crate::loader::load_onnx_bytes(
        "kokoro_duration_predictor_real_seq_specialized",
        &proto.encode_to_vec(),
    )
    .expect("real Kokoro duration predictor should load after specializing sequence length");
    let graph = model
        .to_graph_network()
        .expect("real Kokoro duration predictor should convert to GraphNetwork");
    (model, graph)
}

fn real_duration_predictor_input(model: &OnnxModel, epsilon: f32) -> BoundedTensor {
    let input_spec = common::input_spec_by_name(model, "encoded_features");
    let shape = common::unbatched_shape_from_input_spec(
        input_spec,
        KOKORO_REAL_DURATION_SEQUENCE_LEN,
        "encoded_features",
    );
    let center = ArrayD::zeros(IxDyn(&shape));
    BoundedTensor::from_epsilon(center, epsilon).expect("valid epsilon ball")
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_real_duration_predictor_graph_ibp_positive_expected_duration_3497() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_duration_predictor.onnx");
    let (model, graph) = load_real_duration_predictor_graph();

    assert_eq!(model.network.inputs.len(), 1, "single activation input");
    let input_spec = common::input_spec_by_name(&model, "encoded_features");
    assert_eq!(
        input_spec.shape.len(),
        3,
        "encoded_features should be 3D [B, T, 640]"
    );
    assert_eq!(
        input_spec.shape.last().copied(),
        Some(640),
        "encoded_features final dim should match the exported Kokoro hidden size"
    );

    let input = real_duration_predictor_input(&model, KOKORO_REAL_DURATION_EPSILON);
    let logits = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed on the real Kokoro duration predictor");

    common::assert_finite_and_ordered(&logits, "real Kokoro duration predictor logit bounds");
    assert_eq!(
        logits.lower().shape().last().copied(),
        Some(kokoro_duration_fixture_contract().duration_bin_count)
    );

    let expected_durations = kokoro_expected_duration_bounds_from_logits(&logits);
    assert_positive_expected_durations(&expected_durations);
    let duration_counts = kokoro_duration_count_bounds_from_logits(&logits, KOKORO_DEFAULT_SPEED);
    assert_production_duration_counts(&duration_counts);

    eprintln!("--- Real Kokoro duration predictor IBP results ---");
    eprintln!(
        "  avg expected-duration width: {:.6}",
        avg_bound_width(&expected_durations)
    );
    eprintln!(
        "  avg production-count width: {:.6}",
        avg_bound_width(&duration_counts)
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_real_duration_predictor_crown_no_looser_than_ibp_3497() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_duration_predictor.onnx");
    let (model, graph) = load_real_duration_predictor_graph();
    let input = real_duration_predictor_input(&model, KOKORO_REAL_DURATION_EPSILON);

    let ibp_logits = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed on the real Kokoro duration predictor");
    let ibp_durations = kokoro_expected_duration_bounds_from_logits(&ibp_logits);

    let crown_result = graph
        .propagate_crown_with_provenance(&input)
        .expect("CROWN backward should succeed on the real Kokoro duration predictor");
    common::assert_finite_and_ordered(
        &crown_result.bounds,
        "real Kokoro duration predictor CROWN logit bounds",
    );

    let crown_durations = kokoro_expected_duration_bounds_from_logits(&crown_result.bounds);
    assert_positive_expected_durations(&crown_durations);
    let crown_counts =
        kokoro_duration_count_bounds_from_logits(&crown_result.bounds, KOKORO_DEFAULT_SPEED);
    assert_production_duration_counts(&crown_counts);

    assert_crown_no_looser_than_ibp_durations(
        &crown_durations,
        &ibp_durations,
        &crown_result.provenance,
        "Real Kokoro",
    );
}

/// Epsilon sensitivity sweep: find the critical epsilon where the real Kokoro
/// duration predictor bounds become non-vacuous.
///
/// At eps=1e-3 the trained BiLSTM (640-dim input, 320 hidden) amplifies the
/// perturbation enough to saturate all 50 sigmoid bins, yielding max-width
/// bounds (width=50.0). This test sweeps down to discover the onset of
/// tightening.
///
/// This is the diagnostic gate for the "Verified on Kokoro duration predictor"
/// acceptance criterion in #3497.
///
/// Sources:
/// - designs/2026-03-11-avoice-phase1-onnx-execution.md (section 4)
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_real_duration_predictor_epsilon_sensitivity_3497() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_duration_predictor.onnx");
    let (model, graph) = load_real_duration_predictor_graph();
    let max_width = kokoro_duration_fixture_contract().duration_bin_count as f32;
    let mut found_nontrivial = false;

    eprintln!("--- Real Kokoro duration predictor epsilon sensitivity ---");
    for &eps in KOKORO_REAL_EPSILON_SWEEP {
        let input = real_duration_predictor_input(&model, eps);
        let logits = graph
            .propagate_ibp(&input)
            .expect("IBP should succeed at every epsilon");
        let durations = kokoro_expected_duration_bounds_from_logits(&logits);
        let width = avg_bound_width(&durations);
        let tightening_pct = (1.0 - width / max_width) * 100.0;

        eprintln!(
            "  eps={eps:.0e}: avg duration width={width:.6}, tightening vs max={tightening_pct:.2}%"
        );
        if width < max_width - 0.01 {
            found_nontrivial = true;
        }
    }

    assert!(
        found_nontrivial,
        "At least one epsilon in the sweep should produce non-vacuous IBP bounds \
         (avg width < {max_width}). If all are vacuous, the model's weight magnitudes \
         may require even smaller epsilon or a non-zero center."
    );
}

/// CROWN backward at a non-vacuous epsilon: verify CROWN tightens over IBP
/// on the real Kokoro duration predictor.
///
/// Finds the largest epsilon where IBP bounds are not fully saturated (the
/// strongest verifiable claim), then runs CROWN backward and checks:
/// 1. CROWN bounds are no looser than IBP per dimension
/// 2. CROWN achieves measurable tightening (width ratio < 1.0)
///
/// The BiLSTM gate activations (sigmoid/tanh) benefit from CROWN linear
/// relaxation, as demonstrated by 59.82% tightening on the synthetic BiLSTM
/// surrogate. The real model should show comparable behavior when the input
/// perturbation does not saturate the gates.
///
/// Sources:
/// - designs/2026-03-11-avoice-phase1-onnx-execution.md (section 4)
/// - reports/research/issue-3497-monotonicity-current.md (BiLSTM tightening)
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_real_duration_predictor_crown_tightens_at_nontrivial_epsilon_3497() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_duration_predictor.onnx");
    let (model, graph) = load_real_duration_predictor_graph();
    let max_width = kokoro_duration_fixture_contract().duration_bin_count as f32;

    // Find the largest epsilon that produces non-vacuous IBP bounds.
    let mut best_eps = None;
    for &eps in KOKORO_REAL_EPSILON_SWEEP {
        let input = real_duration_predictor_input(&model, eps);
        let logits = graph.propagate_ibp(&input).expect("IBP should succeed");
        let durations = kokoro_expected_duration_bounds_from_logits(&logits);
        let width = avg_bound_width(&durations);
        if width < max_width - 0.01 {
            best_eps = Some(eps);
            break;
        }
    }

    let eps = best_eps.expect(
        "no epsilon in the sweep produced non-vacuous IBP bounds — \
         the model may need even smaller epsilon or a non-zero center",
    );
    eprintln!("--- Real Kokoro CROWN at non-vacuous eps={eps:.0e} ---");

    let input = real_duration_predictor_input(&model, eps);

    let ibp_logits = graph.propagate_ibp(&input).expect("IBP should succeed");
    let ibp_durations = kokoro_expected_duration_bounds_from_logits(&ibp_logits);

    let crown_result = graph
        .propagate_crown_with_provenance(&input)
        .expect("CROWN backward should succeed on the real Kokoro duration predictor");
    common::assert_finite_and_ordered(
        &crown_result.bounds,
        "real Kokoro CROWN logit bounds at non-vacuous epsilon",
    );

    let crown_durations = kokoro_expected_duration_bounds_from_logits(&crown_result.bounds);
    assert_positive_expected_durations(&crown_durations);
    let crown_counts =
        kokoro_duration_count_bounds_from_logits(&crown_result.bounds, KOKORO_DEFAULT_SPEED);
    assert_production_duration_counts(&crown_counts);

    assert_crown_no_looser_than_ibp_durations(
        &crown_durations,
        &ibp_durations,
        &crown_result.provenance,
        &format!("Real Kokoro (eps={eps:.0e})"),
    );

    let ibp_width = avg_bound_width(&ibp_durations);
    let crown_width = avg_bound_width(&crown_durations);
    let tightening = if ibp_width > 0.0 {
        (1.0 - crown_width / ibp_width) * 100.0
    } else {
        0.0
    };

    eprintln!("  IBP width: {ibp_width:.6}, CROWN width: {crown_width:.6}");
    eprintln!("  CROWN tightening over IBP: {tightening:.2}%");
    eprintln!("  provenance: {:?}", crown_result.provenance);

    assert!(
        tightening > 0.0,
        "CROWN should tighten vs IBP at eps={eps:.0e}: IBP width={ibp_width:.6}, \
         CROWN width={crown_width:.6}. The BiLSTM gate activations (sigmoid/tanh) \
         should benefit from CROWN linear relaxation."
    );
}
