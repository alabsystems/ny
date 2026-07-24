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
