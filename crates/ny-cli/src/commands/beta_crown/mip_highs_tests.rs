// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Tests for mip_highs.rs — Part of #3817.

use super::super::mip_preprocess::{
    bounded_tensor_to_bounds, convert_intermediate_bounds, extract_linear_relu_params,
};
use super::super::mip_single_hidden::{
    collect_exact_single_hidden_intermediate_bounds, is_single_hidden_linear_relu_linear,
};
use super::intermediate_bounds::mip_crown_ibp_budget_secs;
use super::warm_start::build_warm_start_vector;
use super::*;
use ndarray::{arr1, arr2};
use ny_mip::encode_feedforward;
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{Layer, Network, PhaseBudgetConfig};
use std::time::{Duration, Instant};

fn build_anti_correlated_fixture() -> (Network, BoundedTensor) {
    // Network: 2 -> 2 (ReLU) -> 1
    // W1 has anti-correlated rows so IBP over-approximates the ReLU input.
    // CROWN backward can exploit the linear dependency to get tighter bounds.
    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let w2 = arr2(&[[1.0, 1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    (network, input)
}

#[test]
fn safenlp_raw_size_guard_is_bounded_before_canonicalization() {
    let compact_fixture = SafeNlpRawAdmissionSize {
        input_dim: 2,
        hidden_dim: 128,
        last_input_dim: 128,
        output_dim: 2,
        source_elements: Some(2 * 128 + 128 + 128 * 2 + 2),
    };
    assert!(safenlp_raw_size_within_guard(compact_fixture, 2, 2, 2));
    let official_row45 = SafeNlpRawAdmissionSize {
        input_dim: 30,
        source_elements: Some(30 * 128 + 128 + 128 * 2 + 2),
        ..compact_fixture
    };
    assert!(safenlp_raw_size_within_guard(official_row45, 30, 30, 2));

    let boundary = SafeNlpRawAdmissionSize {
        input_dim: SAFENLP_DIRECT_MIP_FIRST_MAX_INPUT_DIM,
        hidden_dim: SAFENLP_DIRECT_MIP_FIRST_MAX_HIDDEN_DIM,
        last_input_dim: SAFENLP_DIRECT_MIP_FIRST_MAX_HIDDEN_DIM,
        output_dim: SAFENLP_DIRECT_MIP_FIRST_MAX_OUTPUT_DIM,
        source_elements: Some(SAFENLP_DIRECT_MIP_FIRST_MAX_SOURCE_ELEMENTS),
    };
    assert!(safenlp_raw_size_within_guard(
        boundary,
        boundary.input_dim,
        boundary.input_dim,
        boundary.output_dim,
    ));

    for oversized in [
        SafeNlpRawAdmissionSize {
            input_dim: SAFENLP_DIRECT_MIP_FIRST_MAX_INPUT_DIM + 1,
            ..boundary
        },
        SafeNlpRawAdmissionSize {
            hidden_dim: SAFENLP_DIRECT_MIP_FIRST_MAX_HIDDEN_DIM + 1,
            last_input_dim: SAFENLP_DIRECT_MIP_FIRST_MAX_HIDDEN_DIM + 1,
            ..boundary
        },
        SafeNlpRawAdmissionSize {
            output_dim: SAFENLP_DIRECT_MIP_FIRST_MAX_OUTPUT_DIM + 1,
            ..boundary
        },
        SafeNlpRawAdmissionSize {
            source_elements: Some(SAFENLP_DIRECT_MIP_FIRST_MAX_SOURCE_ELEMENTS + 1),
            ..boundary
        },
        SafeNlpRawAdmissionSize {
            source_elements: None,
            ..boundary
        },
    ] {
        assert!(!safenlp_raw_size_within_guard(
            oversized,
            oversized.input_dim,
            oversized.input_dim,
            oversized.output_dim,
        ));
    }

    assert!(!safenlp_raw_size_within_guard(
        SafeNlpRawAdmissionSize {
            last_input_dim: boundary.hidden_dim - 1,
            ..boundary
        },
        boundary.input_dim,
        boundary.input_dim,
        boundary.output_dim,
    ));
    assert!(!safenlp_raw_size_within_guard(
        boundary,
        boundary.input_dim - 1,
        boundary.input_dim,
        boundary.output_dim,
    ));
    assert!(!safenlp_raw_size_within_guard(
        boundary,
        boundary.input_dim,
        boundary.input_dim - 1,
        boundary.output_dim,
    ));
    assert!(!safenlp_raw_size_within_guard(
        boundary,
        boundary.input_dim,
        boundary.input_dim,
        boundary.output_dim - 1,
    ));
}

/// Regression test for #3817: MIP intermediate bounds use CROWN-IBP, not IBP.
///
/// Builds a Linear -> ReLU -> Linear network where CROWN-IBP is strictly
/// tighter than IBP on the hidden layer. Verifies that
/// `collect_mip_intermediate_bounds` returns CROWN-IBP bounds (matching
/// `collect_crown_ibp_bounds`) and that these differ from plain IBP.
#[test]
fn test_collect_mip_intermediate_bounds_uses_crown_ibp() {
    let (network, input) = build_anti_correlated_fixture();

    // The helper under test
    let mip_bounds = collect_mip_intermediate_bounds_with_deadline(&network, &input, None).unwrap();

    // Ground truth: direct CROWN-IBP call
    let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    // Plain IBP for comparison
    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();

    // Assert: helper matches CROWN-IBP exactly
    assert_eq!(mip_bounds.len(), crown_ibp_bounds.len());
    for (i, (mip, crown)) in mip_bounds.iter().zip(crown_ibp_bounds.iter()).enumerate() {
        assert_eq!(
            mip.lower(),
            crown.lower(),
            "layer {} lower mismatch between helper and CROWN-IBP",
            i
        );
        assert_eq!(
            mip.upper(),
            crown.upper(),
            "layer {} upper mismatch between helper and CROWN-IBP",
            i
        );
    }

    // Assert: at least one layer where CROWN-IBP is strictly tighter than IBP.
    // Without this, the test could pass on a trivial network where IBP == CROWN-IBP.
    let mut found_tighter = false;
    for (crown, ibp) in crown_ibp_bounds.iter().zip(ibp_bounds.iter()) {
        let crown_lower = crown.lower().iter().copied().collect::<Vec<_>>();
        let ibp_lower = ibp.lower().iter().copied().collect::<Vec<_>>();
        let crown_upper = crown.upper().iter().copied().collect::<Vec<_>>();
        let ibp_upper = ibp.upper().iter().copied().collect::<Vec<_>>();

        for j in 0..crown_lower.len() {
            if crown_lower[j] > ibp_lower[j] + 1e-6 || crown_upper[j] < ibp_upper[j] - 1e-6 {
                found_tighter = true;
                break;
            }
        }
        if found_tighter {
            break;
        }
    }
    assert!(
        found_tighter,
        "CROWN-IBP should be strictly tighter than IBP on at least one hidden neuron \
         for this anti-correlated weight fixture"
    );
}

#[test]
fn test_collect_mip_intermediate_bounds_elapsed_deadline_falls_back_to_ibp() {
    let (network, input) = build_anti_correlated_fixture();

    let past_deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    let mip_bounds =
        collect_mip_intermediate_bounds_with_deadline(&network, &input, past_deadline).unwrap();
    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();

    assert_eq!(mip_bounds.len(), ibp_bounds.len());
    for (i, (mip, ibp)) in mip_bounds.iter().zip(ibp_bounds.iter()).enumerate() {
        assert_eq!(
            mip.lower(),
            ibp.lower(),
            "layer {} lower mismatch between expired-deadline helper and IBP",
            i
        );
        assert_eq!(
            mip.upper(),
            ibp.upper(),
            "layer {} upper mismatch between expired-deadline helper and IBP",
            i
        );
    }
}

#[test]
fn test_single_hidden_exact_affine_fast_path_preserves_big_m_bounds_3864() {
    let (network, input) = build_anti_correlated_fixture();

    assert!(is_single_hidden_linear_relu_linear(&network));

    let fast_path_bounds =
        collect_exact_single_hidden_intermediate_bounds(&network, &input).unwrap();
    let current_bounds =
        collect_mip_intermediate_bounds_with_deadline(&network, &input, None).unwrap();

    let fast_path_intermediate = convert_intermediate_bounds(&fast_path_bounds, &network).unwrap();
    let current_intermediate = convert_intermediate_bounds(&current_bounds, &network).unwrap();

    assert_eq!(fast_path_intermediate.len(), 1);
    assert_eq!(current_intermediate.len(), 1);
    for (layer_idx, (fast, current)) in fast_path_intermediate
        .iter()
        .zip(current_intermediate.iter())
        .enumerate()
    {
        assert_eq!(
            fast.len(),
            current.len(),
            "layer {} neuron count mismatch between fast path and current path",
            layer_idx
        );
        for (neuron_idx, (fast_bound, current_bound)) in fast.iter().zip(current.iter()).enumerate()
        {
            assert_eq!(
                fast_bound.lower(),
                current_bound.lower(),
                "layer {} neuron {} lower mismatch between fast path and current path",
                layer_idx,
                neuron_idx
            );
            assert_eq!(
                fast_bound.upper(),
                current_bound.upper(),
                "layer {} neuron {} upper mismatch between fast path and current path",
                layer_idx,
                neuron_idx
            );
        }
    }
}

#[test]
fn test_build_warm_start_vector_matches_encoder_column_order_3865() {
    let (network, input) = build_anti_correlated_fixture();
    let candidate = arr1(&[1.0f32, -1.0f32]).into_dyn();
    let exact_bounds = collect_exact_single_hidden_intermediate_bounds(&network, &input).unwrap();
    let intermediate_bounds = convert_intermediate_bounds(&exact_bounds, &network).unwrap();
    let (weights, biases, layer_dims) = extract_linear_relu_params(&network).unwrap();
    let input_bounds = bounded_tensor_to_bounds(&input).unwrap();
    let encoder = encode_feedforward(
        &weights,
        &biases,
        &layer_dims,
        &input_bounds,
        &intermediate_bounds,
    )
    .unwrap();
    let num_cols = encoder.num_cols();

    let warm_start = build_warm_start_vector(
        &candidate,
        &weights,
        &biases,
        &layer_dims,
        &intermediate_bounds,
        num_cols,
    )
    .expect("warm-start vector should build for matching input dimension");

    assert_eq!(num_cols, 9);
    assert_eq!(
        warm_start,
        vec![1.0, -1.0, 2.0, -2.0, 2.0, 1.0, 0.0, 0.0, 2.0]
    );
}

#[test]
fn test_mip_crown_ibp_budget_scales_with_timeout() {
    // 5% fraction: 20s → 1.0s, 120s → capped at 2.0s, 4s → floored at 0.25s
    let policy = PhaseBudgetConfig::default();
    assert_eq!(mip_crown_ibp_budget_secs(20.0, &policy), 1.0);
    assert_eq!(mip_crown_ibp_budget_secs(120.0, &policy), 2.0);
    assert_eq!(mip_crown_ibp_budget_secs(4.0, &policy), 0.25);
}

// ---------------------------------------------------------------------------
// Soundness gate: clamp + independent forward-pass revalidation of MIP `Sat`
// witnesses before emitting `Violated`. A wrong VNN-COMP verdict is -150, so a
// witness that is out-of-box or that the f64->f32 cast moved off the violation
// must be demoted to `Unknown` rather than emitted as `sat`.
// ---------------------------------------------------------------------------

/// Identity-ish network: 2 -> 1 Linear with weight [[1, 0]] and no bias, so the
/// forward output is simply input[0]. Lets the test reason exactly about the
/// independent forward pass over the box [0,1] x [0,1].
fn build_passthrough_fixture() -> (Network, BoundedTensor) {
    let w = arr2(&[[1.0, 0.0]]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w, None).unwrap()));
    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    (network, input)
}

/// A genuine in-box violation must stay `Violated`: the clamp is a no-op and the
/// independent forward pass still violates the spec with a comfortable margin.
#[test]
fn test_revalidate_keeps_genuine_in_box_violation() {
    let (network, input) = build_passthrough_fixture();
    // Unsafe region: output[0] > 0.5. Witness [0.95, 0.5] is in-box; forward
    // output = 0.95 > 0.5 (margin 0.45 >> 1e-5).
    let constraints = vec![OutputConstraint::GreaterThanConst(0, 0.5)];
    let raw_input = vec![0.95_f64, 0.5_f64];

    let result = revalidate_mip_witness(&network, &input, &raw_input, &constraints, 1);

    match result {
        VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => {
            // Clamp was a no-op: counterexample equals the (in-box) witness.
            assert!(
                (counterexample[0] - 0.95).abs() < 1e-6,
                "input[0] preserved"
            );
            assert!((counterexample[1] - 0.5).abs() < 1e-6, "input[1] preserved");
            // Output is the RE-EVALUATED forward output, not a solver relaxation.
            assert!((output[0] - 0.95).abs() < 1e-6, "re-evaluated output[0]");
        }
        other => panic!("expected Violated for genuine in-box violation, got {other:?}"),
    }
}

/// An out-of-box witness whose violation only holds outside the box must demote
/// to `Unknown`: clamping brings it back in-box, where the spec no longer holds.
#[test]
fn test_revalidate_demotes_out_of_box_witness() {
    let (network, input) = build_passthrough_fixture();
    // Unsafe region: output[0] > 1.5. The only way to satisfy this is input[0] >
    // 1.5, which is OUTSIDE the box [0,1]. The raw witness [1.6, 0.5] satisfies
    // it but is out-of-box; clamping yields [1.0, 0.5] -> output 1.0 < 1.5.
    let constraints = vec![OutputConstraint::GreaterThanConst(0, 1.5)];
    let raw_input = vec![1.6_f64, 0.5_f64];

    let result = revalidate_mip_witness(&network, &input, &raw_input, &constraints, 1);

    match result {
        VerificationResult::Unknown { .. } => {} // expected: demoted, never -150
        other => panic!("expected Unknown for out-of-box witness, got {other:?}"),
    }
}

/// A MIP-slack witness that is in-box but whose TRUE forward output does not
/// violate the spec (the relaxation reported a violating output the real network
/// never produces) must demote to `Unknown`.
#[test]
fn test_revalidate_demotes_in_box_non_violating_witness() {
    let (network, input) = build_passthrough_fixture();
    // Unsafe region: output[0] > 0.9. Witness [0.5, 0.5] is in-box; the true
    // forward output is 0.5, which does NOT exceed 0.9. (A relaxed MIP could
    // have claimed a violating output here; the independent pass rejects it.)
    let constraints = vec![OutputConstraint::GreaterThanConst(0, 0.9)];
    let raw_input = vec![0.5_f64, 0.5_f64];

    let result = revalidate_mip_witness(&network, &input, &raw_input, &constraints, 1);

    match result {
        VerificationResult::Unknown { .. } => {} // expected: in-box but not unsafe
        other => panic!("expected Unknown for in-box non-violating witness, got {other:?}"),
    }
}

/// A borderline witness whose forward output sits exactly on the threshold is
/// demoted: the epsilon margin guard (1e-5) absorbs f64->f32 cast drift that
/// could flip the verdict between our pass and the organizer's onnxruntime.
#[test]
fn test_revalidate_confirms_sub_eps_margin_witness_exact_semantics() {
    let (network, input) = build_passthrough_fixture();
    // Unsafe region: output[0] > 0.5. Witness gives forward output 0.5 + 1e-7:
    // strictly greater, so under exact SMT-LIB semantics this IS a violation.
    // The old blanket eps guard (margin >= 1e-5) demoted it — and thereby
    // demoted every real sat_relu witness, whose margins are exactly 0.0 or a
    // few ULPs by construction. The revalidation now confirms on exact
    // semantics; cross-implementation robustness is arbitrated by the
    // trusted-ORT vnncomp gate before any sat is scored.
    let constraints = vec![OutputConstraint::GreaterThanConst(0, 0.5)];
    let raw_input = vec![0.5_f64 + 1e-7, 0.5_f64];

    let result = revalidate_mip_witness(&network, &input, &raw_input, &constraints, 1);

    match result {
        VerificationResult::Violated { .. } => {} // expected: genuine strict violation
        other => {
            panic!("expected Violated for a genuinely-violating sub-eps witness, got {other:?}")
        }
    }

    // At EXACT equality a strict constraint is NOT violated: still demoted.
    let raw_equal = vec![0.5_f64, 0.5_f64];
    let result = revalidate_mip_witness(&network, &input, &raw_equal, &constraints, 1);
    match result {
        VerificationResult::Unknown { .. } => {} // expected: strict `>` fails at equality
        other => panic!("expected Unknown for exact-equality strict witness, got {other:?}"),
    }

    // ...but a NON-STRICT constraint at exact equality IS violated (the
    // sat_relu shape: satisfying assignments land dyadic-exactly ON the
    // threshold).
    let nonstrict = vec![OutputConstraint::GreaterEqConst(0, 0.5)];
    let result = revalidate_mip_witness(&network, &input, &raw_equal, &nonstrict, 1);
    match result {
        VerificationResult::Violated { .. } => {} // expected: >= satisfied at equality
        other => panic!("expected Violated for exact-equality non-strict witness, got {other:?}"),
    }
}

/// A nudge JUST outside the box (input[0] slightly above the upper bound) is
/// clamped back exactly to the bound; if the violation only existed at the
/// nudged value, the result demotes. Here the threshold is set so the in-box
/// clamped value is NOT unsafe, exercising the "nudged outside the box" case
/// called out in the task.
#[test]
fn test_revalidate_demotes_witness_nudged_just_outside_box() {
    let (network, input) = build_passthrough_fixture();
    // Box upper for input[0] is 1.0. Unsafe region: output[0] > 1.0 + 1e-3.
    // The raw witness nudges input[0] to 1.0005 (just past the box) to fake a
    // violation; clamping to 1.0 -> output 1.0, which is NOT > 1.001.
    let constraints = vec![OutputConstraint::GreaterThanConst(0, 1.0 + 1e-3)];
    let raw_input = vec![1.0005_f64, 0.5_f64];

    let result = revalidate_mip_witness(&network, &input, &raw_input, &constraints, 1);

    match result {
        VerificationResult::Unknown { .. } => {} // expected: clamp removed the fake violation
        other => panic!("expected Unknown for witness nudged outside the box, got {other:?}"),
    }
}

/// ADVERSARIAL (independent verifier): an in-box witness whose TRUE forward
/// output does not violate the spec must be demoted to `Unknown`, never emitted
/// as `sat`. Uses the 2->2(ReLU)->1 anti-correlated fixture so the witness is a
/// realistic MIP slack point (not the trivial passthrough net). The witness is
/// asserted to be strictly inside the box first, so this exercises the
/// "relaxation reported a violation the real net never produces" hole, which is
/// the -150 case the organizer's onnxruntime would catch.
#[test]
fn adversarial_in_box_non_violating_witness_demoted_to_unknown() {
    let (network, input) = build_anti_correlated_fixture();
    // Box is [-1,1]^2. Pick a witness strictly inside the box.
    let raw_input = vec![0.25_f64, -0.25_f64];

    // Confirm the witness is in-box so clamping is a NO-OP (isolate the
    // "true output doesn't violate" failure mode from the out-of-box mode).
    let lo: Vec<f32> = input.lower().iter().copied().collect();
    let hi: Vec<f32> = input.upper().iter().copied().collect();
    for (k, &v) in raw_input.iter().enumerate() {
        assert!(
            (v as f32) >= lo[k] && (v as f32) <= hi[k],
            "test setup: witness coord {k} must be in-box",
        );
    }

    // Compute the network's TRUE output at this witness by hand:
    //   W1 = [[1,-1],[-1,1]], pre = [x0-x1, -x0+x1] = [0.5, -0.5]
    //   relu = [0.5, 0.0]; W2 = [[1,1]] -> out = 0.5.
    let true_out =
        independent_mip_forward(&network, &arr1(&[0.25f32, -0.25f32]).into_dyn()).unwrap();
    let y0 = true_out.iter().next().copied().unwrap();
    assert!(
        (y0 - 0.5).abs() < 1e-6,
        "sanity: true forward output is 0.5"
    );

    // Unsafe region claimed by a (fictional) relaxed MIP sat: output[0] > 2.0.
    // The true output 0.5 does NOT satisfy this, so any 'sat' here would be an
    // INVALID witness the organizer rejects -> must demote.
    let constraints = vec![OutputConstraint::GreaterThanConst(0, 2.0)];

    let result = revalidate_mip_witness(&network, &input, &raw_input, &constraints, 1);

    match result {
        VerificationResult::Unknown { .. } => {} // correct: -150 hole closed
        VerificationResult::Violated { output, .. } => panic!(
            "SOUNDNESS HOLE: invalid in-box witness emitted as Violated with output {output:?} \
             (true output is 0.5, spec demands >2.0); this would score -150 in VNN-COMP"
        ),
        other => panic!("expected Unknown for invalid in-box witness, got {other:?}"),
    }
}

/// Empty constraint list cannot confirm any violation -> demote to `Unknown`
/// (defensive; the encoder always asserts at least one constraint on a real sat).
#[test]
fn test_revalidate_demotes_empty_constraints() {
    let (network, input) = build_passthrough_fixture();
    let constraints: Vec<OutputConstraint> = vec![];
    let raw_input = vec![0.5_f64, 0.5_f64];

    let result = revalidate_mip_witness(&network, &input, &raw_input, &constraints, 1);

    assert!(
        matches!(result, VerificationResult::Unknown { .. }),
        "empty constraints must not confirm a violation"
    );
}

#[test]
fn uncertified_mip_unsat_is_unknown() {
    let result = map_mip_nonsat_result(MipResult::Unsat { certified: false }, 2);
    assert!(
        matches!(result, VerificationResult::Unknown { .. }),
        "solver infeasibility without checked evidence must not verify"
    );

    let certified = map_mip_nonsat_result(MipResult::Unsat { certified: true }, 2);
    assert!(matches!(certified, VerificationResult::Verified { .. }));
}

#[test]
fn every_disjunctive_clause_requires_certified_unsat() {
    assert_eq!(
        disjunctive_proof_status(false, false, false),
        DisjunctiveProofStatus::Verified
    );
    assert!(matches!(
        disjunctive_proof_status(false, true, false),
        DisjunctiveProofStatus::Unknown(_)
    ));
    assert!(matches!(
        disjunctive_proof_status(true, false, false),
        DisjunctiveProofStatus::Unknown(_)
    ));
    assert_eq!(
        disjunctive_proof_status(false, false, true),
        DisjunctiveProofStatus::Timeout
    );
}

fn margin_reframe_fixture(row_lb: f64, row_ub: f64) -> (ny_mip::MipParts, ny_mip::ir::Row) {
    let mut problem = ny_mip::ir::MilpProblem::new();
    let output = problem.add_col(0.0, -2.0, 2.0);
    let row = problem.add_row(row_lb, row_ub, [(output, 1.0)]);
    (
        ny_mip::MipParts {
            problem,
            input_vars: vec![],
            output_vars: vec![output],
            binary_vars: vec![],
            binary_widths: vec![],
            num_cols: 1,
        },
        row,
    )
}

#[test]
fn direct_mip_margin_reframe_gate_is_exact_and_default_dark() {
    assert!(!ay_margin_reframe_enabled_from_value(None));
    assert!(ay_margin_reframe_enabled_from_value(Some("1")));
    assert_eq!(
        required_safenlp_shared_prefix_ingress_from_margin_value(Some("1")),
        MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix,
        "the exact third gate must select the typed marked-prefix entry"
    );
    assert_eq!(
        required_safenlp_shared_prefix_ingress_from_margin_value(None),
        MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix,
        "unset must retain the existing required plain-prefix entry"
    );
    for malformed in ["", "0", "true", "01", " 1", "1 "] {
        assert!(
            !ay_margin_reframe_enabled_from_value(Some(malformed)),
            "{malformed:?} must not arm the margin reframe"
        );
        assert_eq!(
            required_safenlp_shared_prefix_ingress_from_margin_value(Some(malformed)),
            MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix,
            "{malformed:?} must retain required plain-prefix ingress"
        );
    }

    let (mut parts, row) = margin_reframe_fixture(f64::NEG_INFINITY, 0.25);
    let rows_before = parts.problem.rows().to_vec();
    assert!(
        !maybe_mark_unique_ay_margin(false, MipBackend::Ay, &mut parts, &[row])
            .expect("the default-off path must be inert")
    );
    assert_eq!(parts.problem.margin_row(), None);
    assert_eq!(
        parts.problem.rows(),
        rows_before,
        "default-off must leave the historical feasibility rows byte-identical"
    );
}

#[test]
fn marked_required_ingress_capture_is_terminal_and_plain_paths_stay_historical() {
    use crate::commands::beta_crown::output::{
        begin_capture, end_capture, take_captured_terminal_ingress, CapturedTerminalIngress,
    };

    begin_capture();
    capture_terminal_safenlp_ingress(required_safenlp_shared_prefix_ingress_from_margin_value(
        Some("1"),
    ));
    assert_eq!(
        take_captured_terminal_ingress(),
        CapturedTerminalIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix,
        "the exact third gate must preserve terminal ownership from typed ingress to vnncomp"
    );
    end_capture();

    for ingress in [
        MipFeasibilityIngress::Historical,
        MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix,
    ] {
        begin_capture();
        capture_terminal_safenlp_ingress(ingress);
        assert_eq!(
            take_captured_terminal_ingress(),
            CapturedTerminalIngress::None,
            "historical and required-plain ingress must retain the old post-BaB policy"
        );
        end_capture();
    }

    begin_capture();
    capture_terminal_safenlp_ingress(
        MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix,
    );
    end_capture();
    begin_capture();
    assert_eq!(
        take_captured_terminal_ingress(),
        CapturedTerminalIngress::None,
        "a new capture must reset terminal ownership from the prior instance"
    );
    end_capture();

    // A typed mark outside the in-process vnncomp capture must be inert.
    capture_terminal_safenlp_ingress(
        MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix,
    );
    assert_eq!(
        take_captured_terminal_ingress(),
        CapturedTerminalIngress::None
    );
}

#[test]
fn direct_mip_margin_reframe_marks_only_one_in_process_ay_row() {
    for (lb, ub) in [(f64::NEG_INFINITY, 0.25), (-0.25, f64::INFINITY)] {
        let (mut parts, row) = margin_reframe_fixture(lb, ub);
        let rows_before = parts.problem.rows().to_vec();
        assert!(
            maybe_mark_unique_ay_margin(true, MipBackend::Ay, &mut parts, &[row])
                .expect("a unique one-sided AY row must be markable")
        );
        assert_eq!(parts.problem.margin_row(), Some(row));
        assert_eq!(
            parts.problem.rows(),
            rows_before,
            "marking is metadata-only; it must not alter the original model"
        );
    }

    let (mut subprocess, row) = margin_reframe_fixture(f64::NEG_INFINITY, 0.25);
    assert!(
        !maybe_mark_unique_ay_margin(true, MipBackend::AyProc, &mut subprocess, &[row],)
            .expect("the subprocess backend must decline without mutation")
    );
    assert_eq!(subprocess.problem.margin_row(), None);

    let (mut multi, first) = margin_reframe_fixture(f64::NEG_INFINITY, 0.25);
    let second = multi
        .problem
        .add_row(-0.5, f64::INFINITY, [(multi.output_vars[0], 1.0)]);
    assert!(
        !maybe_mark_unique_ay_margin(true, MipBackend::Ay, &mut multi, &[first, second],)
            .expect("multi-row conjunctions must stay on plain feasibility")
    );
    assert_eq!(multi.problem.margin_row(), None);
}

#[test]
fn direct_mip_margin_reframe_rejects_a_malformed_named_row() {
    let (mut equality, row) = margin_reframe_fixture(0.25, 0.25);
    let error = maybe_mark_unique_ay_margin(true, MipBackend::Ay, &mut equality, &[row])
        .expect_err("an equality is not a one-sided decision margin");
    assert!(
        error.to_string().contains("single one-sided inequality"),
        "unexpected fail-closed error: {error}"
    );
    assert_eq!(
        equality.problem.margin_row(),
        None,
        "a rejected marker must not leave partial authority behind"
    );
}

fn certified_shared_tree_spec(constraint: OutputConstraint) -> VnnLibSpec {
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 2;
    spec.output_constraints = vec![constraint];
    spec
}

fn certified_shared_tree_encoder() -> ny_mip::MipEncoder {
    let weights = vec![
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        vec![
            1.0, 1.0, 1.0, 1.0, 1.0, //
            -1.0, -1.0, -1.0, -1.0, -1.0,
        ],
    ];
    let biases = vec![vec![0.0; 5], vec![1.0, -1.0]];
    let layer_dims = vec![5, 5, 2];
    let input_bounds = vec![Bound::new(-4.0, 4.0); 5];
    // Widths are [2, 3, 3, 5, 0.5]. The width-3 tie must retain encoder
    // insertion order, so the expected four-way order is [3, 1, 2, 0].
    let intermediate_bounds = vec![vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-2.0, 1.0),
        Bound::new(-1.5, 1.5),
        Bound::new(-4.0, 1.0),
        Bound::new(-0.25, 0.25),
    ]];
    encode_feedforward(
        &weights,
        &biases,
        &layer_dims,
        &input_bounds,
        &intermediate_bounds,
    )
    .expect("five-ReLU shared-tree fixture must encode")
}

#[test]
fn certified_shared_tree_gate_and_preclone_admission_are_exact() {
    assert!(!mip_certified_shared_tree_enabled_from_value(None));
    assert!(mip_certified_shared_tree_enabled_from_value(Some("1")));
    for malformed in ["", "0", "true", "01", "2", "-1", " 1", "1 "] {
        assert!(
            !mip_certified_shared_tree_enabled_from_value(Some(malformed)),
            "{malformed:?} must not arm the certified tree"
        );
    }

    let spec = certified_shared_tree_spec(OutputConstraint::LessEq(0, 1));
    let live = Instant::now() + Duration::from_secs(1);
    assert!(certified_shared_tree_preclone_eligible(
        true,
        false,
        MipBackend::Ay,
        true,
        &spec,
        live,
    ));
    for admitted in [
        certified_shared_tree_preclone_eligible(false, false, MipBackend::Ay, true, &spec, live),
        certified_shared_tree_preclone_eligible(true, true, MipBackend::Ay, true, &spec, live),
        certified_shared_tree_preclone_eligible(true, false, MipBackend::AyProc, true, &spec, live),
        certified_shared_tree_preclone_eligible(true, false, MipBackend::Ay, false, &spec, live),
        certified_shared_tree_preclone_eligible(
            true,
            false,
            MipBackend::Ay,
            true,
            &spec,
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("past instant"),
        ),
    ] {
        assert!(!admitted);
    }

    let constant = certified_shared_tree_spec(OutputConstraint::LessEqConst(0, 0.0));
    assert!(!certified_shared_tree_preclone_eligible(
        true,
        false,
        MipBackend::Ay,
        true,
        &constant,
        live,
    ));
    let mut disjunctive = spec.clone();
    disjunctive.is_disjunction = true;
    assert!(!certified_shared_tree_preclone_eligible(
        true,
        false,
        MipBackend::Ay,
        true,
        &disjunctive,
        live,
    ));

    // The documented conflict policy is "tree refuses, margin fallback
    // survives": prove the fallback can still stamp the unique unsafe row.
    let mut fallback = certified_shared_tree_encoder();
    let rows = add_vnnlib_constraints(&mut fallback, &spec).expect("fallback row");
    let mut fallback = fallback.into_parts();
    assert!(
        maybe_mark_unique_ay_margin(true, MipBackend::Ay, &mut fallback, &rows,)
            .expect("margin fallback must remain available")
    );
    assert_eq!(fallback.problem.margin_row(), Some(rows[0]));
}

#[test]
fn certified_shared_tree_orients_pairwise_rows_and_selects_stable_widest_four() {
    let parts = certified_shared_tree_encoder().into_parts();
    let expected_splits = [
        parts.binary_vars[3],
        parts.binary_vars[1],
        parts.binary_vars[2],
        parts.binary_vars[0],
    ];
    let y0 = parts.output_vars[0];
    let y1 = parts.output_vars[1];

    for (constraint, objective) in [
        (OutputConstraint::LessEq(0, 1), [(y0, 1.0), (y1, -1.0)]),
        (OutputConstraint::LessThan(0, 1), [(y0, 1.0), (y1, -1.0)]),
        (OutputConstraint::GreaterEq(0, 1), [(y1, 1.0), (y0, -1.0)]),
        (OutputConstraint::GreaterThan(0, 1), [(y1, 1.0), (y0, -1.0)]),
        (OutputConstraint::LessEq(1, 0), [(y1, 1.0), (y0, -1.0)]),
    ] {
        let plan =
            certified_shared_tree_plan(true, &certified_shared_tree_spec(constraint), &parts)
                .expect("eligible pairwise row");
        assert_eq!(plan.objective, objective);
        assert_eq!(plan.splits, expected_splits);
    }
}

#[test]
fn certified_shared_tree_rejects_ambiguous_or_malformed_base_models() {
    let pairwise = certified_shared_tree_spec(OutputConstraint::LessEq(0, 1));

    let mut too_few = certified_shared_tree_encoder().into_parts();
    too_few.binary_vars.truncate(3);
    too_few.binary_widths.truncate(3);
    assert!(certified_shared_tree_plan(true, &pairwise, &too_few).is_none());

    let mut one_fixed = certified_shared_tree_encoder().into_parts();
    let widest = one_fixed.binary_vars[3];
    one_fixed.problem.fix_col(widest, 0.0);
    let fixed_plan =
        certified_shared_tree_plan(true, &pairwise, &one_fixed).expect("four unfixed remain");
    assert_eq!(
        fixed_plan.splits,
        [
            one_fixed.binary_vars[1],
            one_fixed.binary_vars[2],
            one_fixed.binary_vars[0],
            one_fixed.binary_vars[4],
        ],
        "a fixed widest binary is not an unfixed split candidate"
    );

    let mut nonfinite = certified_shared_tree_encoder().into_parts();
    nonfinite.binary_widths[0] = f64::NAN;
    assert!(certified_shared_tree_plan(true, &pairwise, &nonfinite).is_none());

    let mut marked = certified_shared_tree_encoder().into_parts();
    let marked_output = marked.output_vars[0];
    let row = marked
        .problem
        .add_row(f64::NEG_INFINITY, 0.0, [(marked_output, 1.0)]);
    marked.problem.mark_margin_row(row).expect("fixture marker");
    assert!(certified_shared_tree_plan(true, &pairwise, &marked).is_none());

    for constraint in [
        OutputConstraint::LessEqConst(0, 0.0),
        OutputConstraint::LessEq(0, 0),
        OutputConstraint::LessEq(0, 2),
    ] {
        assert!(certified_shared_tree_plan(
            true,
            &certified_shared_tree_spec(constraint),
            &certified_shared_tree_encoder().into_parts(),
        )
        .is_none());
    }
    let mut multi = pairwise;
    multi
        .output_constraints
        .push(OutputConstraint::GreaterEq(0, 1));
    assert!(certified_shared_tree_plan(
        true,
        &multi,
        &certified_shared_tree_encoder().into_parts(),
    )
    .is_none());
}

#[test]
fn certified_shared_tree_off_path_preserves_historical_problem_bytes() {
    let spec = certified_shared_tree_spec(OutputConstraint::LessEq(0, 1));
    let mut historical = certified_shared_tree_encoder();
    let mut observed = certified_shared_tree_encoder();

    assert!(!certified_shared_tree_preclone_eligible(
        false,
        false,
        MipBackend::Ay,
        true,
        &spec,
        Instant::now() + Duration::from_secs(1),
    ));
    add_vnnlib_constraints(&mut historical, &spec).expect("historical unsafe row");
    add_vnnlib_constraints(&mut observed, &spec).expect("default-dark unsafe row");
    let historical = historical.into_parts();
    let observed = observed.into_parts();

    assert_eq!(historical.problem.cols(), observed.problem.cols());
    assert_eq!(historical.problem.rows(), observed.problem.rows());
    assert_eq!(
        historical.problem.margin_row(),
        observed.problem.margin_row()
    );
    assert_eq!(historical.input_vars, observed.input_vars);
    assert_eq!(historical.output_vars, observed.output_vars);
    assert_eq!(historical.binary_vars, observed.binary_vars);
    assert_eq!(historical.binary_widths, observed.binary_widths);
    assert_eq!(historical.num_cols, observed.num_cols);
}

fn certified_shared_tree_proof_parts(comparison_output: f64) -> ny_mip::MipParts {
    // The root relaxation takes every binary at 1/2 and z=1/2. Every one of
    // the sixteen complete assignments forces z>=1. Choosing comparison
    // output 3/4 therefore requires exactly the fixed four-way tree; choosing
    // 1/4 is already strictly excluded at the root; choosing 1 leaves a
    // feasible equality in every complete assignment.
    let mut problem = ny_mip::ir::MilpProblem::new();
    let splits: [ny_mip::ir::Col; 4] =
        std::array::from_fn(|_| problem.add_integer_col(0.0, 0.0, 1.0));
    let z = problem.add_col(0.0, 0.0, 2.0);
    let comparison = problem.add_col(0.0, comparison_output, comparison_output);
    for split in splits {
        problem.add_row(0.0, f64::INFINITY, [(z, 1.0), (split, -1.0)]);
        problem.add_row(1.0, f64::INFINITY, [(z, 1.0), (split, 1.0)]);
    }
    ny_mip::MipParts {
        num_cols: problem.num_cols(),
        problem,
        input_vars: vec![],
        output_vars: vec![z, comparison],
        binary_vars: splits.to_vec(),
        binary_widths: vec![4.0, 3.0, 2.0, 1.0],
    }
}

static CERTIFIED_SHARED_TREE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ntest::timeout(30_000)]
fn certified_shared_tree_accepts_only_replayed_root_or_full_sixteen_leaf_proofs() {
    let _guard = CERTIFIED_SHARED_TREE_TEST_LOCK.lock().unwrap();
    let spec = certified_shared_tree_spec(OutputConstraint::LessEq(0, 1));

    let root = certified_shared_tree_proof_parts(0.25);
    let root_plan = certified_shared_tree_plan(true, &spec, &root).expect("root plan");
    let root_proof =
        try_certified_shared_tree(&root, &root_plan, Instant::now() + Duration::from_secs(10))
            .expect("strict root lower row must be replayed");
    assert!(matches!(
        root_proof.proof_route,
        CertifiedLinearLowerProofRoute::RelaxationEntailment
            | CertifiedLinearLowerProofRoute::RootFarkas
    ));
    assert_eq!(root_proof.ay_tree_leaves, 0);
    assert_eq!(root_proof.ny_cert_farkas_replays, 1);

    let full = certified_shared_tree_proof_parts(0.75);
    let full_plan = certified_shared_tree_plan(true, &spec, &full).expect("full-tree plan");
    let full_proof =
        try_certified_shared_tree(&full, &full_plan, Instant::now() + Duration::from_secs(10))
            .expect("all sixteen assignments must replay");
    assert_eq!(
        full_proof.proof_route,
        CertifiedLinearLowerProofRoute::TreeFarkas
    );
    assert_eq!(full_proof.ay_tree_leaves, 16);
    assert_eq!(full_proof.ny_cert_farkas_replays, 16);
}

#[test]
#[ntest::timeout(30_000)]
fn certified_shared_tree_equality_deadline_and_worker_decline_are_never_proofs() {
    let _guard = CERTIFIED_SHARED_TREE_TEST_LOCK.lock().unwrap();
    let spec = certified_shared_tree_spec(OutputConstraint::LessEq(0, 1));
    let equality = certified_shared_tree_proof_parts(1.0);
    let plan = certified_shared_tree_plan(true, &spec, &equality).expect("equality plan");

    assert!(try_certified_shared_tree(
        &equality,
        &plan,
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("past instant"),
    )
    .is_none());

    let admission = ny_mip::CertifiedLinearLowerWorkerAdmission::try_acquire()
        .expect("reserve the exact worker to force a caller decline");
    assert!(
        try_certified_shared_tree(&equality, &plan, Instant::now() + Duration::from_secs(1),)
            .is_none()
    );
    drop(admission);

    assert!(
        try_certified_shared_tree(&equality, &plan, Instant::now() + Duration::from_secs(10),)
            .is_none()
    );
}

#[test]
fn objective_first_sat_gate_and_budget_are_fail_closed() {
    assert!(!ay_objective_first_sat_enabled_from_value(None));
    assert!(ay_objective_first_sat_enabled_from_value(Some("1")));
    for malformed in ["", "0", "true", "01", " 1", "1 "] {
        assert!(
            !ay_objective_first_sat_enabled_from_value(Some(malformed)),
            "{malformed:?} must not arm the dark lane"
        );
    }

    assert_eq!(
        objective_first_sat_budget(true, MipBackend::Ay, 1, 40.0, 40.0),
        Some(ObjectiveFirstSatBudget {
            probe_secs: 8.0,
            envelope_secs: 40.0,
        })
    );
    assert_eq!(
        objective_first_sat_budget(true, MipBackend::Ay, 1, 100.0, 100.0),
        Some(ObjectiveFirstSatBudget {
            probe_secs: 10.0,
            envelope_secs: 100.0,
        }),
        "the probe must obey its hard 10-second cap"
    );
    assert_eq!(
        objective_first_sat_budget(false, MipBackend::Ay, 1, 40.0, 40.0),
        None
    );
    assert_eq!(
        objective_first_sat_budget(true, MipBackend::AyProc, 1, 40.0, 40.0),
        None
    );
    assert_eq!(
        objective_first_sat_budget(true, MipBackend::Ay, 2, 40.0, 40.0),
        None,
        "a multi-row conjunction needs a max-min construction and must be refused"
    );
    assert_eq!(
        objective_first_sat_budget(true, MipBackend::Ay, 1, f64::NAN, 40.0),
        None
    );
    assert_eq!(
        objective_first_sat_budget(true, MipBackend::Ay, 1, 40.0, f64::NAN),
        None
    );
}

#[test]
fn required_shared_prefix_never_launches_unconfirmed_sat_retry() {
    assert!(unconfirmed_sat_retry_allowed(
        MipFeasibilityIngress::Historical,
        true,
        false
    ));
    assert!(!unconfirmed_sat_retry_allowed(
        MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix,
        true,
        false
    ));
    assert!(!unconfirmed_sat_retry_allowed(
        MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix,
        true,
        false
    ));
    assert!(!unconfirmed_sat_retry_allowed(
        MipFeasibilityIngress::Historical,
        false,
        false
    ));
    assert!(!unconfirmed_sat_retry_allowed(
        MipFeasibilityIngress::Historical,
        true,
        true
    ));
}

#[test]
fn sequential_mip_serial_gate_is_exact_and_fail_closed() {
    let historical = MipConfig::default().parallel_split;
    assert_eq!(historical, 0);
    assert_eq!(sequential_mip_parallel_split_from_value(None), historical);
    assert_eq!(
        sequential_mip_parallel_split_from_value(Some("0")),
        historical
    );
    assert_eq!(sequential_mip_parallel_split_from_value(Some("1")), 1);
    for malformed in ["", "true", "yes", "01", "2", "-1", " 1", "1 "] {
        assert_eq!(
            sequential_mip_parallel_split_from_value(Some(malformed)),
            historical,
            "malformed gate {malformed:?} must preserve historical auto split"
        );
    }
}

#[test]
fn objective_first_sat_window_floor_is_one_shared_absolute_envelope() {
    // Discriminating regression for the audit failure: the 4s probe is sized
    // from the caller's 20s slice, but a serial window solve historically owns
    // a 100s floor. The fallback gets the SAME 100s absolute deadline, not a
    // fresh 100s after the probe.
    let budget = objective_first_sat_budget(true, MipBackend::Ay, 1, 20.0, 100.0)
        .expect("window-floor schedule");
    assert_eq!(budget.probe_secs, 4.0);
    assert_eq!(budget.envelope_secs, 100.0);
    assert_eq!(budget.fallback_secs_after_elapsed(4.0), Some(96.0));
    assert_eq!(
        budget.probe_secs + budget.fallback_secs_after_elapsed(4.0).unwrap(),
        100.0,
        "armed wall allocation must equal, never exceed, objective-off"
    );

    // An early decline reclaims unused probe time, while setup/replay charged
    // after a full probe reduces the remaining slice.
    assert_eq!(budget.fallback_secs_after_elapsed(1.25), Some(98.75));
    assert_eq!(budget.fallback_secs_after_elapsed(4.75), Some(95.25));
    assert_eq!(
        1.25 + budget.fallback_secs_after_elapsed(1.25).unwrap(),
        100.0,
        "early decline may reclaim time but cannot extend the envelope"
    );
    let setup_secs = 0.5;
    let replay_secs = 0.25;
    let total_pre_fallback_secs = setup_secs + budget.probe_secs + replay_secs;
    assert_eq!(
        setup_secs
            + budget.probe_secs
            + replay_secs
            + budget
                .fallback_secs_after_elapsed(total_pre_fallback_secs)
                .unwrap(),
        100.0,
        "probe plus setup/replay plus fallback must remain inside 100s"
    );
    assert_eq!(budget.fallback_secs_after_elapsed(100.0), None);
    assert_eq!(budget.fallback_secs_after_elapsed(f64::NAN), None);

    let ledger = ObjectiveFirstSatLedger {
        deadline: Instant::now() + Duration::from_secs(100),
    };
    let fallback = objective_first_sat_fallback_config(
        MipConfig {
            timeout_secs: 20.0,
            ..MipConfig::default()
        },
        budget,
        ledger,
    );
    assert_eq!(fallback.timeout_secs, 100.0);
    assert_eq!(fallback.ay_hard_deadline, Some(ledger.deadline));

    // Sub-gate models retain the original 20s envelope: 4 + 16, never 24.
    let sub_gate =
        objective_first_sat_budget(true, MipBackend::Ay, 1, 20.0, 20.0).expect("sub-gate schedule");
    assert_eq!(sub_gate.fallback_secs_after_elapsed(4.0), Some(16.0));
    assert_eq!(
        sub_gate.probe_secs + sub_gate.fallback_secs_after_elapsed(4.0).unwrap(),
        20.0
    );
    assert_eq!(
        total_pre_fallback_secs
            + sub_gate
                .fallback_secs_after_elapsed(total_pre_fallback_secs)
                .unwrap(),
        20.0,
        "sub-gate setup/probe/replay/fallback must remain inside 20s"
    );
}

#[test]
fn objective_first_sat_budget_refuses_exhausted_deadline_boundaries() {
    let exact_min = objective_first_sat_budget(true, MipBackend::Ay, 1, 0.1, 0.1)
        .expect("0.1s is the documented minimum");
    assert!((exact_min.probe_secs - 0.02).abs() < f64::EPSILON);
    assert!(exact_min.fallback_secs_after_elapsed(0.02).is_some());
    assert_eq!(
        objective_first_sat_budget(true, MipBackend::Ay, 1, 0.099, 0.099),
        None
    );

    let budget = ObjectiveFirstSatBudget {
        probe_secs: 1.0,
        envelope_secs: 10.0,
    };
    assert!(budget.fallback_secs_after_elapsed(9.998).is_some());
    assert_eq!(budget.fallback_secs_after_elapsed(9.9995), None);
    assert_eq!(budget.fallback_secs_after_elapsed(10.0), None);
    assert_eq!(budget.fallback_secs_after_elapsed(-0.1), None);
}

#[test]
fn objective_first_sat_ledger_never_extends_outer_deadline() {
    let budget = ObjectiveFirstSatBudget {
        probe_secs: 10.0,
        envelope_secs: 60.0,
    };
    let outer = Instant::now() + Duration::from_secs(2);
    let ledger =
        ObjectiveFirstSatLedger::start(budget, Some(outer)).expect("live capped objective ledger");
    assert!(
        ledger.deadline <= outer,
        "reanchoring a relative envelope must not extend the caller deadline"
    );
    assert!(
        ledger.probe_secs(budget).is_some_and(|secs| secs <= 2.0),
        "probe must be capped to both its nominal slice and the outer deadline"
    );
    let expired = ObjectiveFirstSatLedger {
        deadline: Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("test instant supports a one-millisecond subtraction"),
    };
    assert!(expired.expired());
    assert_eq!(
        expired.probe_secs(budget),
        None,
        "an expired ledger cannot grant a new relative probe clock"
    );
    assert_eq!(
        ObjectiveFirstSatLedger::start(budget, Some(Instant::now())),
        None,
        "an exhausted outer deadline must not arm a probe"
    );
}

#[test]
fn objective_first_sat_probe_still_requires_concrete_replay() {
    let (network, input) = build_passthrough_fixture();
    let constraints = vec![OutputConstraint::GreaterThanConst(0, 0.5)];

    let confirmed = revalidate_objective_first_sat_probe(
        OneSidedSatProbe::Witness(ny_mip::OneSidedSatWitness {
            objective: 0.95,
            output_values: vec![1234.0],
            input_values: vec![0.95, 0.5],
        }),
        &network,
        &input,
        &constraints,
        1,
    );
    assert!(
        matches!(confirmed, Some(VerificationResult::Violated { .. })),
        "the original concrete forward, not the solver output, must confirm the point"
    );

    let rejected = revalidate_objective_first_sat_probe(
        OneSidedSatProbe::Witness(ny_mip::OneSidedSatWitness {
            objective: 0.25,
            output_values: vec![9999.0],
            input_values: vec![0.25, 0.5],
        }),
        &network,
        &input,
        &constraints,
        1,
    );
    assert!(
        rejected.is_none(),
        "a solver candidate that misses the true property must fall back"
    );

    let infeasible = revalidate_objective_first_sat_probe(
        OneSidedSatProbe::Declined(ny_mip::OneSidedSatDecline::InfeasibleIgnored),
        &network,
        &input,
        &constraints,
        1,
    );
    assert!(
        infeasible.is_none(),
        "even exact solver infeasibility has no verdict authority in this lane"
    );
}
