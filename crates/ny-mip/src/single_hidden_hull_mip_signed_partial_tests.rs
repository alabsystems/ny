// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Fixed-partition scalar-aggregate strict controls; authored and unexecuted.
use super::test_fixture::*;
use crate::{MipBackend, MipConfig, MipResult, MipSolver};
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

#[test]
fn signed_partial_all_half_whole_local_families_keep_the_padded_rational_gap() {
    let epsilon = q(EPS);
    let bias = q(0.25);
    let ideal = Q::new(9.into(), 64.into());
    assert_eq!(&bias - q(14.) * &epsilon, ideal);
    for (positive, n1, n3, expected) in [
        (q(3. / 16.), q(3. / 8.), q(1. / 4.), q(-1. / 8.)),
        (
            q(1. / 4.),
            q(1. / 2.),
            Q::new(1.into(), 3.into()),
            q(-1. / 4.),
        ),
    ] {
        // Fixed negative partition A=1..8,B=9..16 and scalar-aggregate
        // row form only: arbitrary head-equality cuts can remove this point.
        // All padding inputs=-1, outputs=selectors=0 in every lift atom.
        for units in 0..=2 {
            let mass = q(2. * f64::from(units));
            let fa = &mass * &positive - &n1;
            let fb = (q(4.) - &mass) * &positive - q(3.) * &n3;
            for (coefficient, value) in [(&mass - q(1.), fa), (q(1.) - &mass, fb)] {
                let r = value / coefficient;
                assert!(r >= Q::zero() && r <= q(0.5));
                // Equal atoms at +/-2r have mean input0, selector1/2, ReLU r.
                let active = q(2.) * &r;
                let inactive = -&active;
                assert!(active <= q(1.) && inactive >= q(-1.));
                assert!((&active + &inactive).is_zero());
                assert_eq!(active / q(2.), r);
            }
        }
        assert_eq!(&bias + q(4.) * positive - n1 - q(3.) * n3, expected);
        assert!(expected < Q::zero());
    }
    // Principal phases plus one padding phase: ALL eight closed cells.
    // Thirteen omitted heads use secants, one common G_y=-eps/2 for all cells.
    for mask in 0..8 {
        let alpha = Q::new(((mask & 1) + 3 * ((mask >> 1) & 1)).into(), 4.into());
        assert!(
            (q(4.) * alpha - q((mask & 1) as f64) - q(3. * ((mask >> 1) & 1) as f64)).is_zero()
        );
        let mut lower = -q(6.5) * &epsilon;
        let residual = &epsilon / q(2.) - &epsilon * q(((mask >> 2) & 1) as f64);
        lower -= residual.abs();
        assert_eq!(lower, -q(7.) * &epsilon);
        assert_eq!(&bias + lower - q(7.) * &epsilon, ideal);
    }
}

#[test]
fn signed_partial_original_encoder_all_mass_prefixes_strict_lp_and_actual_ay_certificate() {
    let f = Fixture::new(0.25);
    let original = f.parts();
    let unsafe_row = f.encoding(&original).unsafe_row.0;
    let mut combined = original.problem.clone();
    let mut partial_rows = Vec::new();
    for units in 0..=2 {
        let (parts, report, before) = append(&f, units);
        let joint = &report.signed.as_ref().unwrap().component_rows.joint;
        let domination = report
            .signed
            .as_ref()
            .unwrap()
            .component_rows
            .domination
            .appended_rows;
        let joint_start = parts.problem.num_rows() - joint.appended_rows;
        let old_end = joint_start - domination;
        for row in &parts.problem.rows()[before..old_end] {
            combined.add_row(
                row.lb,
                row.ub,
                row.coeffs.iter().map(|&(i, a)| (crate::ir::Col(i), a)),
            );
        }
        if units == 1 {
            partial_rows = parts.problem.rows()[joint_start..].to_vec();
        }
        assert!(report.signed.as_ref().unwrap().lower.unwrap() < 0.);
    }
    for dyadic in [false, true] {
        let v = fractional(&f, &original, dyadic);
        check_rows(&combined, &v);
        assert!(v[original.output_vars[0].0] < Q::zero());
    }
    let before = lp(&combined, &original, unsafe_row);
    assert!(
        before <= -0.249999,
        "old independent-component root={before}"
    );
    assert_eq!(partial_rows.len(), 1);
    for row in partial_rows {
        combined.add_row(
            row.lb,
            row.ub,
            row.coeffs.iter().map(|&(i, a)| (crate::ir::Col(i), a)),
        );
    }
    let after = lp(&combined, &original, unsafe_row);
    assert!(
        (after - 9. / 64.).abs() < 1e-6,
        "partial union root={after}"
    );
    let (parts, _, _) = append(&f, 1);
    let result = MipSolver::new(
        parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.,
            ay_hard_deadline: Some(end()),
            ..Default::default()
        },
    )
    .check_feasibility()
    .unwrap();
    assert!(
        matches!(result, MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}

#[test]
fn signed_partial_automatic_large_pair_checks_eight_masks_and_original_graph_inclusion() {
    let f = Fixture::new(0.25);
    let (parts, report, _) = append(&f, 1);
    let joint = &report.signed.as_ref().unwrap().component_rows.joint;
    assert!(joint.partial_eligible_pairs >= 1);
    assert!(joint.partial_work_units > 0);
    assert_eq!(joint.partial_tail_rechecks, 26);
    assert_eq!(joint.completed_pairs, 1);
    let unsafe_row = f.original().problem.num_rows() - 1;
    for x in [-1.0_f64, -0.5, 0., 0.5, 1.] {
        for pattern in 0..4 {
            let mut inputs = vec![Q::zero(); D];
            inputs[0] = q(x);
            for i in 1..D {
                inputs[i] = q(match pattern {
                    0 => -1.,
                    1 => 0.,
                    2 => 1.,
                    _ => {
                        if i % 2 == 0 {
                            1.
                        } else {
                            -1.
                        }
                    }
                });
            }
            check_rows_except(
                &parts.problem,
                &graph(&f, &parts, &inputs, 0),
                Some(unsafe_row),
            );
        }
    }
    // Closed zero preactivations admit both selectors; the graph remains fixed.
    let inputs = vec![Q::zero(); D];
    for mask in 0..32 {
        let zero_mask = (mask & 1)
            | (((mask >> 1) & 1) << 1)
            | (((mask >> 2) & 1) << 9)
            | (((mask >> 3) & 1) << 2)
            | (((mask >> 4) & 1) << 10);
        check_rows_except(
            &parts.problem,
            &graph(&f, &parts, &inputs, zero_mask),
            Some(unsafe_row),
        );
    }
}

#[test]
fn signed_partial_original_exact_padding_ties_and_unsafe_witnesses_remain_actual_sat() {
    for bias in [14. * EPS, 14. * EPS - 0.25] {
        let f = Fixture::graph_control(bias);
        let (parts, _, _) = append(&f, 1);
        let mut inputs = vec![q(1.); D];
        inputs[0] = Q::zero();
        for mask in 0..8 {
            let zero_mask = (mask & 1) | (((mask >> 1) & 1) << 1) | (((mask >> 2) & 1) << 9);
            let witness = graph(&f, &parts, &inputs, zero_mask);
            assert_eq!(witness[parts.output_vars[0].0], q(bias) - q(14.) * q(EPS));
            check_rows(&parts.problem, &witness);
        }
        let result = MipSolver::new(
            parts,
            MipConfig {
                backend: MipBackend::Ay,
                parallel_split: 1,
                lp_tighten: false,
                timeout_secs: 10.,
                ay_hard_deadline: Some(end()),
                ..Default::default()
            },
        )
        .check_feasibility()
        .unwrap();
        assert!(matches!(result, MipResult::Sat { .. }), "{result:?}");
    }
}
