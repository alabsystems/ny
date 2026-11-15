// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(20)
}
fn boxed_pivot() -> SourceModel {
    SourceModel::parse(
        "PAIRED_LP_GEQ_V1 3 2 0 1\n0 1 0.25 0.5 0 0\n1 2 1 1 0 -1 0\n2 1 2 1 0\n0 2\n",
    )
    .unwrap()
}

#[test]
fn virtual_pivot_boxes_recover_bounds_lost_by_forcing_pivot_residual_zero() {
    let source = boxed_pivot();
    let reduced = build(&source, deadline(), 128 * 1024 * 1024).unwrap();
    assert_eq!(reduced.free, 1);
    assert_eq!(reduced.rows.len(), 4);
    for (sign, lower, role) in [
        (1.0, 0.25, ReducedRole::PivotLower(1)),
        (-1.0, -0.5, ReducedRole::PivotUpper(1)),
    ] {
        let (_, objective_scale) = reduced.objective(&source, 0, sign).unwrap();
        let index = reduced
            .rows
            .iter()
            .position(|row| match (row.role, role) {
                (ReducedRole::PivotLower(a), ReducedRole::PivotLower(b))
                | (ReducedRole::PivotUpper(a), ReducedRole::PivotUpper(b)) => a == b,
                _ => false,
            })
            .unwrap();
        let mut raw = vec![0.0; reduced.rows.len()];
        raw[index] = reduced.rows[index].scale / objective_scale;
        let lifted = reduced
            .lift(&source, &raw, 0, sign, 1.0, objective_scale, deadline())
            .unwrap();
        assert_eq!(lifted.equalities[0], -sign);
        assert_eq!(
            exact_replay(&source, &lifted, 0, sign, deadline()).unwrap(),
            rational(lower)
        );
        let forced_zero = reduced
            .lift(
                &source,
                &vec![0.0; reduced.rows.len()],
                0,
                sign,
                1.0,
                objective_scale,
                deadline(),
            )
            .unwrap();
        let old_score = exact_replay(&source, &forced_zero, 0, sign, deadline()).unwrap();
        assert_eq!(old_score, rational(if sign > 0.0 { 0.0 } else { -1.0 }));
    }
}

#[test]
fn objective_and_row_scaling_lift_duplicate_source_contributions_with_correct_sign() {
    let source = SourceModel::parse("PAIRED_LP_GEQ_V1 3 2 1 1\n0 1 0 1 0 0\n1 3 1 1 0 -0.5 0 -0.5 0\n2 1 2 1 0\n2 0 2 0 2 1\n0 2\n").unwrap();
    let reduced = build(&source, deadline(), 128 * 1024 * 1024).unwrap();
    let (_, objective_scale) = reduced.objective(&source, 0, 1.0).unwrap();
    assert_eq!(objective_scale, 0.5);
    assert_eq!(reduced.rows[0].scale, 2.0);
    let mut raw = vec![0.0; reduced.rows.len()];
    raw[0] = 1.0;
    let positive = reduced
        .lift(&source, &raw, 0, 1.0, 1.0, objective_scale, deadline())
        .unwrap();
    let negative = reduced
        .lift(&source, &raw, 0, 1.0, -1.0, objective_scale, deadline())
        .unwrap();
    assert_eq!(positive.inequalities, vec![0.25]);
    assert_eq!(negative.inequalities, vec![0.0]);
    assert_eq!(
        exact_replay(&source, &positive, 0, 1.0, deadline()).unwrap(),
        rational(0.25)
    );
    assert_eq!(
        exact_replay(&source, &negative, 0, 1.0, deadline()).unwrap(),
        rational(0.0)
    );
}

#[test]
fn simultaneous_virtual_sides_price_the_net_box_residual_safely() {
    let source = boxed_pivot();
    let reduced = build(&source, deadline(), 128 * 1024 * 1024).unwrap();
    let (_, scale) = reduced.objective(&source, 0, 1.0).unwrap();
    let mut raw = vec![0.0; reduced.rows.len()];
    for (i, row) in reduced.rows.iter().enumerate() {
        let weight = match row.role {
            ReducedRole::PivotLower(1) => 1.25,
            ReducedRole::PivotUpper(1) => 0.25,
            _ => 0.0,
        };
        raw[i] = weight * row.scale / scale;
    }
    let dual = reduced
        .lift(&source, &raw, 0, 1.0, 1.0, scale, deadline())
        .unwrap();
    assert_eq!(dual.equalities[0], -1.0);
    assert_eq!(
        exact_replay(&source, &dual, 0, 1.0, deadline()).unwrap(),
        rational(0.25)
    );
}

#[test]
fn reduced_ay_solves_the_box_fixture_and_retains_counterexample_range() {
    let source = boxed_pivot();
    let result = solve_reduced_float(&source, deadline(), 128 * 1024 * 1024).unwrap();
    assert_eq!(
        result.finished_objectives, 2,
        "{:?} {:?}",
        result.diagnostics, result.termination_reason
    );
    assert_eq!(result.reduced_free_variables, Some(1));
    assert!(result.alternate_dual_text.is_some());
    for d in &result.diagnostics {
        assert_eq!(d.status, "untrusted_reduced_float_duals");
        assert!(!d.exact_primal_checked);
        assert!(d.ay_exact_optimum.is_none());
        let expected = if d.sign > 0 { 0.25 } else { -0.5 };
        assert!((d.numerical_proposal_lower.unwrap() - expected).abs() < 1e-10);
        assert!((d.reduced_proposal_lower.unwrap() - expected).abs() < 1e-10);
        assert!(d.reduction_lift_gap.unwrap().abs() < 1e-10);
    }
    // x=y=.4,z=0 is an exact feasible LP point outside band .1. Neither
    // approximate conditioning nor multiplier lifting can make it disappear.
    let point = vec![rational(0.4), rational(0.4), rational(0.0)];
    check_original_primal(&source, &point, deadline()).unwrap();
}

#[test]
fn dependent_pivots_in_source_row_order_replay_canonical_archived_duals() {
    // Authored rows deliberately appear as pivot b, then a, then z.
    // a=x, b=2*a+1, with a's box essential to min/max b.
    let source = SourceModel::parse("PAIRED_LP_GEQ_V1 4 3 0 1\n0 1 .25 .5 1.5 2 0 0\n2 2 2 1 1 -2 1\n1 2 1 1 0 -1 0\n3 1 3 1 0\n2 3\n").unwrap();
    let reduced = build(&source, deadline(), 128 * 1024 * 1024).unwrap();
    assert_eq!(reduced.pivots, vec![(1, 1), (2, 0), (3, 2)]);
    assert_eq!(
        &reduced.expressions[2 * reduced.stride..3 * reduced.stride],
        &[1.0, 2.0]
    );
    let mut candidates = Vec::new();
    for (sign, expected) in [(1.0, 1.5), (-1.0, -2.0)] {
        let (_, scale) = reduced.objective(&source, 0, sign).unwrap();
        let mut raw = vec![0.0; reduced.rows.len()];
        for (i, row) in reduced.rows.iter().enumerate() {
            let selected = (sign > 0.0 && matches!(row.role, ReducedRole::PivotLower(1)))
                || (sign < 0.0 && matches!(row.role, ReducedRole::PivotUpper(1)));
            if selected {
                raw[i] = 2.0 * row.scale / scale;
            }
        }
        let dual = reduced
            .lift(&source, &raw, 0, sign, 1.0, scale, deadline())
            .unwrap();
        assert_eq!(dual.equalities, vec![sign, 0.0, -sign]);
        assert_eq!(
            exact_replay(&source, &dual, 0, sign, deadline()).unwrap(),
            rational(expected)
        );
        assert_eq!(
            reduced
                .numerical_lower(&source, &raw, 0, sign, 1.0, scale, deadline())
                .unwrap(),
            expected
        );
        candidates.push(dual);
    }
    let text = source.dual_text(&candidates);
    let replay = diagnose_reduced_dual_text(&source, &text, deadline(), 128 * 1024 * 1024).unwrap();
    assert_eq!(replay.len(), 2);
    assert_eq!(replay[0].numerical_source_lower, 1.5);
    assert_eq!(replay[1].canonical_reduced_lower, -2.0);
    assert!(replay.iter().all(|r| r.substitution_gap == 0.0));
    assert!(diagnose_reduced_dual_text(
        &source,
        &format!("{text} 0"),
        deadline(),
        128 * 1024 * 1024
    )
    .is_err());
    assert!(diagnose_reduced_dual_text(
        &source,
        &text,
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        128 * 1024 * 1024
    )
    .is_err());
}

#[test]
fn malformed_archived_diagnostics_reject_trailing_text_signs_dimensions_and_expiry() {
    let source =
        SourceModel::parse("PAIRED_LP_GEQ_V1 2 1 1 1\n0 1 0 0\n1 1 1 1 0\n1 0 1 0\n0 1\n").unwrap();
    let text = source.dual_text(&[source.zero_dual(), source.zero_dual()]);
    assert!(diagnose_reduced_dual_text(&source, &text, deadline(), 128 * 1024 * 1024).is_ok());
    for bad in [
        format!("{text} 0"),
        "PAIRED_LP_DUAL_GEQ_V1 1 1 1 0 -1 0 0".into(),
        "PAIRED_LP_DUAL_GEQ_V1 1 2 1 0 0 0 0".into(),
        "PAIRED_LP_DUAL_GEQ_V1 1 1 1 0 NaN 0 0".into(),
    ] {
        assert!(diagnose_reduced_dual_text(&source, &bad, deadline(), 128 * 1024 * 1024).is_err());
    }
    assert!(diagnose_reduced_dual_text(
        &source,
        &text,
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        128 * 1024 * 1024
    )
    .is_err());
}

#[test]
fn extreme_approximation_invalid_duals_and_expiry_return_no_new_authority() {
    let source = boxed_pivot();
    assert!(build(
        &source,
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        128 * 1024 * 1024
    )
    .is_err());
    assert!(build(&source, deadline(), 16 * 1024 * 1024).is_err());
    assert!(solve_reduced_float(
        &source,
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        128 * 1024 * 1024
    )
    .is_err());
    let reduced = build(&source, deadline(), 128 * 1024 * 1024).unwrap();
    assert!(reduced
        .lift(&source, &[f64::NAN; 4], 0, 1.0, 1.0, 1.0, deadline())
        .is_err());
    assert!(reduced
        .lift(&source, &[1.0], 0, 1.0, 1.0, 1.0, deadline())
        .is_err());
    let extreme = SourceModel::parse("PAIRED_LP_GEQ_V1 3 2 0 1\n0 1 0 1e100 0 1e100\n1 2 1 1 0 -1e100 0\n2 2 2 1 1 -1e100 0\n0 2\n").unwrap();
    assert!(build(&extreme, deadline(), 128 * 1024 * 1024).is_err());
}

fn mixed_density_source() -> SourceModel {
    let mut text = String::from("PAIRED_LP_GEQ_V1 12 1 5 1\n");
    for _ in 0..11 {
        text.push_str("0 1 ");
    }
    text.push_str("0 0\n11 1 11 1 0\n");
    // Row 0: nine nonzero terms, no output column: omitted by lean.
    text.push('9');
    for j in 0..9 {
        write!(&mut text, " {j} 1").unwrap();
    }
    text.push_str(" 2\n2 0 1 1 1 0.25\n");
    // Row 2 is dense, but MUST remain because original output 9 appears.
    text.push_str("10");
    for j in 0..10 {
        write!(&mut text, " {j} 1").unwrap();
    }
    text.push_str(" 2\n10");
    for j in 0..9 {
        write!(&mut text, " {j} 1").unwrap();
    }
    // An exact zero output term cannot force dense numerical advice to stay.
    text.push_str(" 9 0 0\n11");
    for j in 0..8 {
        write!(&mut text, " {j} 1").unwrap();
    }
    text.push_str(" 8 0 8 0 10 0 0\n9 11\n");
    SourceModel::parse(&text).unwrap()
}

#[test]
fn lean_selection_is_deterministic_preserves_dimensions_and_zero_lifts_original_indices() {
    let source = mixed_density_source();
    let reduced = build_with_policy(
        &source,
        deadline(),
        128 * 1024 * 1024,
        ReducedProposalPolicy::LeanSparseOutput,
    )
    .unwrap();
    assert_eq!(reduced.free, 11);
    assert_eq!(
        reduced.expressions.len(),
        source.bounds.len() * reduced.stride
    );
    assert_eq!(
        reduced.rows.iter().map(|r| r.role).collect::<Vec<_>>(),
        vec![
            ReducedRole::Source(1),
            ReducedRole::Source(2),
            ReducedRole::Source(4)
        ]
    );
    let s = &reduced.statistics;
    assert_eq!(s.selected_source_inequalities, 3);
    assert_eq!(s.omitted_source_inequalities, 2);
    assert_eq!(s.virtual_pivot_bound_rows, 0);
    assert_eq!(s.omitted_virtual_pivot_bound_rows, 2);
    assert_eq!(s.reduced_rows, 3);
    assert_eq!(s.reduced_nonzeros, 20);
    let (_, scale) = reduced.objective(&source, 0, 1.0).unwrap();
    let raw: Vec<_> = reduced
        .rows
        .iter()
        .zip([0.5, 1.5, 2.5])
        .map(|(row, weight)| weight * row.scale / scale)
        .collect();
    let lifted = reduced
        .lift(&source, &raw, 0, 1.0, 1.0, scale, deadline())
        .unwrap();
    assert_eq!(lifted.inequalities, vec![0.0, 0.5, 1.5, 0.0, 2.5]);
    assert_eq!(lifted.equalities.len(), source.equalities.len());
    // Row thinning really enlarges the advice feasible set. This point
    // satisfies every selected row but violates omitted original row 0.
    let mut thin_point = vec![rational(0.0); 12];
    thin_point[0] = rational(0.5);
    thin_point[2] = rational(0.5);
    thin_point[9] = rational(1.0);
    for i in [1, 2, 4] {
        let row = &source.inequalities[i];
        let lhs: BigRational = row
            .terms
            .iter()
            .map(|&(j, a)| rational(a) * &thin_point[j])
            .sum();
        assert!(lhs >= rational(row.rhs));
    }
    assert!(check_original_primal(&source, &thin_point, deadline()).is_err());
    // This is an exact original-model point, not merely a lean-model point.
    let mut point = vec![rational(1.0); 12];
    point[9] = rational(0.5);
    point[11] = rational(0.0);
    check_original_primal(&source, &point, deadline()).unwrap();
    for sign in [1.0, -1.0] {
        let (_, scale) = reduced.objective(&source, 0, sign).unwrap();
        let raw: Vec<_> = reduced
            .rows
            .iter()
            .map(|row| 3.5 * row.scale / scale)
            .collect();
        let lifted = reduced
            .lift(&source, &raw, 0, sign, 1.0, scale, deadline())
            .unwrap();
        assert_eq!(lifted.inequalities[0], 0.0);
        assert_eq!(lifted.inequalities[3], 0.0);
        let bound = exact_replay(&source, &lifted, 0, sign, deadline()).unwrap();
        assert!(bound <= rational(sign * 0.5));
        if sign < 0.0 {
            // The retained source counterexample outside band .1 prevents an
            // unsafe band proof despite dropped rows and pivot-box advice.
            assert!(bound < rational(-0.1));
        }
    }
}

#[test]
fn full_policy_default_retains_every_row_and_every_pivot_bound() {
    assert_eq!(
        ReducedProposalPolicy::default(),
        ReducedProposalPolicy::Full
    );
    let source = mixed_density_source();
    let default = build(&source, deadline(), 128 * 1024 * 1024).unwrap();
    let full = build_with_policy(
        &source,
        deadline(),
        128 * 1024 * 1024,
        ReducedProposalPolicy::Full,
    )
    .unwrap();
    assert_eq!(default.expressions, full.expressions);
    assert_eq!(default.rows.len(), full.rows.len());
    for (a, b) in default.rows.iter().zip(&full.rows) {
        assert_eq!(a.role, b.role);
        assert_eq!(a.terms, b.terms);
        assert_eq!(a.rhs, b.rhs);
        assert_eq!(a.scale, b.scale);
    }
    assert_eq!(default.statistics.selected_source_inequalities, 5);
    assert_eq!(default.statistics.omitted_source_inequalities, 0);
    assert_eq!(default.statistics.virtual_pivot_bound_rows, 2);
    assert_eq!(default.statistics.omitted_virtual_pivot_bound_rows, 0);
    assert_eq!(default.statistics.reduced_rows, 7);
    assert_eq!(default.statistics.reduced_nonzeros, 38);
}

#[test]
fn omitted_pivot_advice_can_weaken_proposals_but_never_remove_original_counterexamples() {
    let source = boxed_pivot();
    let reduced = build_with_policy(
        &source,
        deadline(),
        128 * 1024 * 1024,
        ReducedProposalPolicy::LeanSparseOutput,
    )
    .unwrap();
    assert!(reduced.rows.is_empty());
    assert_eq!(reduced.statistics.omitted_virtual_pivot_bound_rows, 4);
    let point = vec![rational(0.4), rational(0.4), rational(0.0)];
    check_original_primal(&source, &point, deadline()).unwrap();
    // Equality elimination alone admits x=y=0, but original y's pivot
    // lower box still rejects that advice-only primal when source-replayed.
    assert!(check_original_primal(&source, &vec![rational(0.0); 3], deadline()).is_err());
    for (sign, expected) in [(1.0, 0.0), (-1.0, -1.0)] {
        let (_, scale) = reduced.objective(&source, 0, sign).unwrap();
        let dual = reduced
            .lift(&source, &[], 0, sign, 1.0, scale, deadline())
            .unwrap();
        assert_eq!(
            exact_replay(&source, &dual, 0, sign, deadline()).unwrap(),
            rational(expected)
        );
        assert!(rational(expected) <= rational(sign * 0.4));
    }
}

#[test]
fn reduced_objective_mapping_rejects_other_sources_bad_signs_and_invalid_scales() {
    let source = boxed_pivot();
    let reduced = build_with_policy(
        &source,
        deadline(),
        128 * 1024 * 1024,
        ReducedProposalPolicy::LeanSparseOutput,
    )
    .unwrap();
    let mut other = source.clone();
    other.outputs[0] = (1, 2);
    assert!(reduced.objective(&other, 0, 1.0).is_err());
    assert!(reduced
        .lift(&other, &[], 0, 1.0, 1.0, 1.0, deadline())
        .is_err());
    assert!(reduced
        .objective(&source, source.outputs.len(), 1.0)
        .is_err());
    for sign in [0.0, f64::NAN, f64::INFINITY] {
        assert!(reduced.objective(&source, 0, sign).is_err());
        assert!(reduced
            .lift(&source, &[], 0, sign, 1.0, 1.0, deadline())
            .is_err());
    }
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(reduced
            .lift(&source, &[], 0, 1.0, 1.0, scale, deadline())
            .is_err());
    }
}

#[test]
fn lean_declines_deadlines_and_memory_caps_without_proof_authority() {
    let source = mixed_density_source();
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    for policy in [
        ReducedProposalPolicy::Full,
        ReducedProposalPolicy::LeanSparseOutput,
    ] {
        assert!(build_with_policy(&source, expired, 128 * 1024 * 1024, policy).is_err());
        assert!(build_with_policy(&source, deadline(), 16 * 1024 * 1024, policy).is_err());
        assert!(
            solve_reduced_float_with_policy(&source, expired, 128 * 1024 * 1024, policy).is_err()
        );
        assert!(
            solve_reduced_float_with_policy(&source, deadline(), 2 * 1024 * 1024, policy).is_err()
        );
    }
    let reduced = build_with_policy(
        &source,
        deadline(),
        128 * 1024 * 1024,
        ReducedProposalPolicy::LeanSparseOutput,
    )
    .unwrap();
    assert!(reduced
        .lift(&source, &[0.0; 3], 0, 1.0, 1.0, 1.0, expired)
        .is_err());
}

#[test]
fn lean_ay_worker_reports_selection_and_full_length_candidates() {
    let source = boxed_pivot();
    let result = solve_reduced_float_with_policy(
        &source,
        deadline(),
        128 * 1024 * 1024,
        ReducedProposalPolicy::LeanSparseOutput,
    )
    .unwrap();
    let statistics = result.statistics.unwrap();
    assert_eq!(statistics.policy, ReducedProposalPolicy::LeanSparseOutput);
    assert_eq!(statistics.selected_source_inequalities, 0);
    assert_eq!(statistics.omitted_virtual_pivot_bound_rows, 4);
    assert_eq!(statistics.reduced_free_variables, 1);
    assert_eq!(statistics.reduced_nonzeros, 0);
    assert_eq!(result.result.variables, 3);
    assert_eq!(result.result.equalities, 2);
    assert_eq!(result.result.inequalities, 0);
    assert_eq!(
        result.result.finished_objectives, 2,
        "{:?} {:?}",
        result.result.diagnostics, result.result.termination_reason
    );
    assert!(result
        .result
        .dual_text
        .starts_with("PAIRED_LP_DUAL_GEQ_V1 2 0 1\n"));
    assert!(result
        .result
        .diagnostics
        .iter()
        .all(|d| !d.exact_primal_checked && d.ay_exact_optimum.is_none()));
}
