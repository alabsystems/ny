// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual CLI-selected root coupling, checked global improvement and original graph outcomes.
use super::default_fixture::{end, lp, point, q, rows_hold, solve, Fixture, D, EPS, H};
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

fn appended(f: &Fixture, p: &HullMarginProblem) -> (MipParts, HullMipCutReport) {
    // No installed mode bit, Plan, pair, multipliers or captured Point.
    let attempt = propose_root_mip_cuts_with_root_coupling_attempt(p, end(), Duration::ZERO);
    assert!(
        attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked,
        "capture={:?}",
        attempt.primal
    );
    let cuts = attempt.cuts.expect("actual root-policy owner");
    assert_eq!(cuts.proposal_lp_calls, 1);
    let initial = &cuts.signed_report.as_ref().unwrap().root_coupling;
    assert!(
        initial.requested && initial.admitted && initial.selected,
        "root={initial:?}"
    );
    assert_eq!(
        (
            initial.producer_attempts,
            initial.support_attempts,
            initial.completed_supports,
            initial.checker_slots
        ),
        (1, 1, 1, 1)
    );
    assert_eq!(initial.work_units, LIMIT);
    assert!((PREFLIGHT..=LIMIT).contains(&initial.used_work_units));
    assert!(initial.admitted_bytes > 0 && initial.actual_check_work > 0);
    assert!(initial.actual_check_work <= initial.quoted_check_work);
    assert_eq!(initial.parent_count, 6);
    assert!(initial.parent_indices[..6].windows(2).all(|w| w[0] < w[1]));
    assert_eq!((initial.selected_negatives, initial.tail_negatives), (6, 0));
    assert_eq!((initial.old_masks, initial.new_masks), (24, 76));
    assert_eq!(
        (initial.positive_heads, initial.recipient_positive_heads),
        (1, 0)
    );
    let before = initial.before_global_lower.unwrap();
    let proposed = initial.proposed_lower.unwrap();
    let after = initial.after_global_lower.unwrap();
    assert!(
        before.is_finite() && proposed.is_finite() && after > before,
        "strict checked global improvement: {initial:?}"
    );
    assert_eq!(after.to_bits(), proposed.to_bits());
    assert_eq!(initial.stop_reason, "root-improved");
    let saved = initial.clone();
    let mut parts = f.parts();
    let original = parts.problem.clone();
    let binaries = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let old_rows = original.num_rows();
    let report = cuts.append_to(&mut parts, &f.encoding(), end()).unwrap();
    assert_eq!(parts.problem.cols(), original.cols());
    assert_eq!(&parts.problem.rows()[..old_rows], original.rows());
    assert_eq!(parts.binary_vars, binaries);
    assert_eq!(parts.binary_widths, widths);
    assert_eq!(parts.problem.num_rows() - old_rows, report.cuts);
    assert_eq!(report.proposal_lp_calls, 1);
    let signed = report.signed.as_ref().unwrap();
    assert_eq!(signed.root_coupling, saved);
    assert_eq!(signed.source_replays, 1);
    assert!(signed.work_units <= MAX_WORK && signed.admitted_bytes <= MAX_BYTES);
    // The admitted root stage suppresses all later append-only optional work;
    // old original/global/component/phase rows still retain their real checks.
    assert_eq!(signed.component_rows.joint.producer_attempts, 0);
    assert_eq!(signed.component_rows.domination.producer_attempts, 0);
    assert!(!signed.component_rows.joint.prepaid_requested);
    for x in [-1.0_f64, -0.5, 0., 0.5, 1.] {
        for padding in [-1.0_f64, 0., 1.] {
            let mut input = vec![q(padding); D];
            input[0] = q(x);
            for zero_mask in [0, (1usize << H) - 1] {
                rows_hold(
                    &parts.problem,
                    &point(f, &parts, &input, zero_mask),
                    Some(f.encoding().unsafe_row.0),
                );
            }
        }
    }
    if after > 0. {
        // A real second whole-Source replay, after full original-model binding,
        // must support the positive global fact, without a third publication.
        let proof = cuts
            .signed_root_proof(end())
            .expect("positive original Source replay");
        assert!(proof.verified() && proof.lower() > 0.);
    }
    (parts, report)
}

#[test]
fn root_coupling_default_public_entry_strictly_improves_checked_global_bound_and_actual_safe_model()
{
    let f = Fixture::new(0.25);
    let p = f.source();
    let (parts, report) = appended(&f, &p);
    let root = &report.signed.as_ref().unwrap().root_coupling;
    assert!(
        root.before_global_lower.unwrap() <= 0. && root.after_global_lower.unwrap() > 0.,
        "old global must remain open and the new checked Plan must close: {root:?}"
    );
    let original = f.parts();
    let property = f.encoding().unsafe_row.0;
    let before = lp(&original.problem, &original, property);
    assert!(
        (before - (0.25 - 3. - 6. * EPS)).abs() < 1e-6,
        "original Big-M={before}"
    );
    let after = lp(&parts.problem, &parts, property);
    let true_minimum = 0.25 - 6. * EPS;
    assert!(
        after > 0. && after <= true_minimum + 1e-6,
        "actual augmented LP={after}"
    );
    assert!(after + 1e-6 >= root.after_global_lower.unwrap());
    let mut minimum = vec![q(1.); D];
    minimum[0] = Q::zero();
    let graph = point(&f, &parts, &minimum, 0);
    rows_hold(&parts.problem, &graph, Some(property));
    assert_eq!(graph[parts.output_vars[0].0], q(true_minimum));
    let result = solve(parts);
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}

fn actual_sat(bias: f64, expected: f64) {
    let f = Fixture::new(bias);
    let p = f.source();
    let (parts, report) = appended(&f, &p);
    let root = &report.signed.as_ref().unwrap().root_coupling;
    assert!(
        root.before_global_lower.unwrap() < root.after_global_lower.unwrap()
            && root.after_global_lower.unwrap() <= expected,
        "checked improvement must not exclude the original witness: {root:?}"
    );
    let mut witness = vec![q(1.); D];
    witness[0] = Q::zero();
    // Seven coincident zero heads,including the relaxed original positive phase:
    // both closed cells are retained in all128 original phase combinations.
    let zero_heads = [0, 1, 2, 9, 10, 11, 12];
    for cells in 0..128usize {
        let zero_mask = zero_heads
            .iter()
            .enumerate()
            .fold(0usize, |mask, (i, &head)| {
                mask | (((cells >> i) & 1) << head)
            });
        let graph = point(&f, &parts, &witness, zero_mask);
        rows_hold(&parts.problem, &graph, None);
        assert_eq!(graph[parts.output_vars[0].0], q(expected));
        assert!(graph[parts.output_vars[0].0] <= Q::zero());
    }
    match solve(parts) {
        crate::MipResult::Sat { input_values, .. } => {
            let input: Vec<_> = input_values.iter().map(|&x| q(x)).collect();
            let original = f.original();
            let graph = point(&f, &original, &input, 0);
            rows_hold(&original.problem, &graph, None);
            assert!(graph[original.output_vars[0].0] <= Q::zero());
        }
        other => panic!("{other:?}"),
    }
}
#[test]
fn root_coupling_default_public_entry_preserves_original_unsafe_sat_witness() {
    actual_sat(6. * EPS - 0.25, -0.25);
}
#[test]
fn root_coupling_default_public_entry_preserves_original_closed_zero_tie() {
    actual_sat(6. * EPS, 0.);
}
