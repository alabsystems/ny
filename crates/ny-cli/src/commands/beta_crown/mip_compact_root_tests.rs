// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use ny_onnx::vnnlib::OutputConstraint;

fn run(bias: f64, allowed: bool) -> (MipParts, StrengtheningOutcome) {
    let w = vec![
        vec![
            1., 1., 0., 0., 1., -1., 0., 0., 0., 0., 1., 1., 0., 0., 1., -1., 1., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0.,
        ],
        vec![-1., -1., -1., -1., 1., 1., -0.5, 0., 0., 0., 0., 0., 0., 0.],
    ];
    let b = vec![vec![0., 0., 0., 0., 2., 2., 1.], vec![bias, 0.]];
    let dims = vec![4, 7, 2];
    let input = vec![Bound::new(-1., 1.); 4];
    let bounds = vec![vec![Bound::new(-2., 2.); 4]
        .into_iter()
        .chain([Bound::new(1., 3.), Bound::new(1., 3.), Bound::new(1., 1.)])
        .collect()];
    let mut spec = VnnLibSpec::new();
    spec.num_inputs = 4;
    spec.num_outputs = 2;
    spec.input_bounds = vec![(-1., 1.); 4];
    spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];
    let mut e = encode_feedforward(&w, &b, &dims, &input, &bounds).unwrap();
    let rows = add_vnnlib_constraints(&mut e, &spec).unwrap();
    let mut parts = e.into_parts();
    let before = parts.problem.clone();
    let outcome = strengthen_with_short_proof(
        &mut parts,
        true,
        &w,
        &b,
        &dims,
        &bounds,
        &input,
        &spec,
        &rows,
        Instant::now() + Duration::from_secs(30),
        allowed,
    )
    .unwrap();
    assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
    assert_eq!(parts.problem.cols(), before.cols());
    (parts, outcome)
}

#[test]
fn compact_root_cli_authenticates_original_before_positive_floor_and_needs_only_shared_root_lp() {
    let (_, outcome) = run(-1.25, true);
    assert_eq!(
        outcome.cuts.proposal_lp_calls, 1,
        "one actual attaining root proposal"
    );
    assert!(
        outcome.cuts.checked_hull_lower < -1.749,
        "old LP scalar is not overwritten"
    );
    assert_eq!(
        (outcome.cuts.compact_cuts, outcome.cuts.compact_pairs),
        (1, 2)
    );
    let proof = outcome.aggregate.unwrap();
    assert!(proof.verified());
    assert!(proof.lower() > 0.249 && proof.lower() <= 0.25);
    assert!(
        outcome.continuation.is_none(),
        "positive direct replay spends no root/child LP in a tree"
    );
}

#[test]
fn compact_root_cli_disabled_publication_keeps_checked_geometry_without_minting_native_proof() {
    let (_, outcome) = run(-1.25, false);
    assert!(outcome.aggregate.is_none() && outcome.continuation.is_none());
    assert_eq!(outcome.cuts.proposal_lp_calls, 1);
    assert_eq!(outcome.cuts.compact_cuts, 1);
    assert!(outcome.cuts.compact_root_lower.is_some_and(|b| b > 0.249));
    // Required shared/canary/bridge guards all flow through this same allowed
    // value; their complete eight-case policy test remains in the old module.
    for (shared, canary, bridge) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        assert!(!short_proof_allowed(shared, canary, bridge));
    }
}

#[test]
fn compact_root_cli_unsafe_original_cannot_be_closed_by_the_positive_root_proof() {
    let (_, outcome) = run(-1.75, true);
    assert!(outcome.aggregate.is_none());
    assert!(outcome.cuts.compact_root_lower.is_some_and(|b| b <= -0.25));
    assert!(outcome
        .continuation
        .as_ref()
        .is_none_or(|proof| !proof.verified()));
    assert_eq!(outcome.cuts.proposal_lp_calls, 1);
}

#[test]
fn compact_root_cli_rendered_native_proof_is_not_labeled_as_ay_unsat() {
    struct Capture;
    impl Drop for Capture {
        fn drop(&mut self) {
            crate::commands::beta_crown::output::end_capture();
        }
    }
    let (_, outcome) = run(-1.25, true);
    let proof = outcome.aggregate.unwrap();
    assert!(proof.verified());
    for native in [true, false] {
        crate::commands::beta_crown::output::begin_capture();
        let _capture = Capture;
        let result = VerificationResult::Verified {
            provenance: Default::default(),
            output_bounds: vec![Bound::new(0.25, 1.)],
            proof: None,
            actual_method: Some(if native {
                ny_core::MethodUsed::BetaCrown
            } else {
                ny_core::MethodUsed::MipHiGHS
            }),
        };
        let refused = print_result_with_native_hull(
            &result,
            None,
            None,
            0.,
            0.,
            Duration::ZERO,
            MipBackend::Ay,
            native,
            None,
            true,
        )
        .unwrap();
        assert!(!refused);
        let rendered = crate::commands::beta_crown::output::take_captured_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(value["status"], "verified");
        assert_eq!(
            value["method"],
            if native { "native-hull" } else { "mip-ay" }
        );
    }
}
