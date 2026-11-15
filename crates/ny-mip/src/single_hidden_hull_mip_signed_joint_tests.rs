// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored joint-root and authority controls; no executed-result claim.
use super::checked::union;
use super::test_fixture::*;
use super::*;
use crate::{MipBackend, MipConfig, MipResult, MipSolver};
use num_rational::BigRational as Q;
use num_traits::Zero;

#[test]
fn signed_joint_all_half_whole_scalar_component_families_keep_a_strict_rational_root_gap() {
    // At x=0 and delta=.5, conv(x,delta,R(x)) permits every r in[0,.5].
    // These are the EXACT local scalar-aggregate graphs, not all head equalities.
    for (positive, n1, n2) in [
        (q(0.25), q(0.5), Q::new(1.into(), 3.into())),
        (q(3. / 16.), q(3. / 8.), q(0.25)),
    ] {
        for units in 0..=2 {
            let mass = q(2. * f64::from(units));
            let ca = &mass - q(1.);
            let cb = q(1.) - &mass;
            let fa = &mass * &positive - &n1;
            let fb = (q(4.) - &mass) * &positive - q(3.) * &n2;
            for (c, value) in [(ca, fa), (cb, fb)] {
                let r = value / c;
                assert!(r >= Q::zero() && r <= q(0.5));
            }
        }
        assert!(q(4.) * positive - n1 - q(3.) * n2 < q(-0.25));
    }
    // Every union mask has its own positive plane, with one common G=0.
    for mask in 0..4 {
        let alpha = Q::new(((mask & 1) + 3 * ((mask >> 1) & 1)).into(), 4.into());
        let residual = q(4.) * alpha - q((mask & 1) as f64) - q(3. * ((mask >> 1) & 1) as f64);
        assert!(residual.is_zero());
    }
}
#[test]
fn signed_joint_original_encoder_all_mass_rows_strict_root_and_actual_ay_certificate() {
    let f = Fixture::new(0.25);
    let original = f.parts();
    let unsafe_row = f.encoding(&original).unsafe_row.0;
    let mut combined = original.problem.clone();
    let mut union_rows = Vec::new();
    for units in 0..=2 {
        let (parts, report, before) = append(&f, units);
        let joint = report
            .signed
            .as_ref()
            .unwrap()
            .component_rows
            .joint
            .appended_rows;
        let domination = report
            .signed
            .as_ref()
            .unwrap()
            .component_rows
            .domination
            .appended_rows;
        let end_joint = parts.problem.num_rows() - joint;
        let end_old = end_joint - domination;
        for r in &parts.problem.rows()[before..end_old] {
            combined.add_row(
                r.lb,
                r.ub,
                r.coeffs.iter().map(|&(i, a)| (crate::ir::Col(i), a)),
            );
        }
        if units == 1 {
            union_rows = parts.problem.rows()[end_joint..].to_vec();
        }
        assert!(report.signed.as_ref().unwrap().lower.unwrap() < 0.);
    }
    for dyadic in [false, true] {
        let v = fractional(&f, &original, dyadic);
        check_rows(&combined, &v);
        assert!(v[original.output_vars[0].0] < Q::zero());
    }
    let before = lp(&combined, &original, unsafe_row);
    assert!(before <= -0.249999, "old root={before}");
    for r in union_rows {
        combined.add_row(
            r.lb,
            r.ub,
            r.coeffs.iter().map(|&(i, a)| (crate::ir::Col(i), a)),
        );
    }
    let after = lp(&combined, &original, unsafe_row);
    assert!((after - 0.25).abs() < 1e-6, "joint root={after}");
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
fn signed_joint_source_kernel_checks_every_union_mask_after_generic_alpha_steering() {
    let f = Fixture::new(0.25);
    let p = f.source();
    let c = cuts(&p, 1);
    let row = c.signed.as_ref().unwrap();
    let choices = resources::choices(row, &c, end(), &mut || true).unwrap();
    assert_eq!(choices.len(), 1);
    let mut merged = union(row, &p, &choices[0]).unwrap();
    let mut count = 0;
    steering::steer(&p, &mut merged, end(), &mut || true, &mut count).unwrap();
    assert_eq!(count, 4);
    for mask in 0..4 {
        let expected = ((mask & 1) + 3 * ((mask >> 1) & 1)) as f64 / 4.;
        assert!((merged.alpha[mask] - expected).abs() < 1e-12);
        let lower = certificate::mask_lower(&c, &merged, mask, end(), &mut || true).unwrap();
        assert!(lower <= 1e-12 && lower > -1e-10, "mask {mask}: {lower}");
    }
    assert_eq!(merged.positive, vec![0]);
    assert_eq!(merged.positive_units, vec![2]);
    assert_eq!(merged.negative, vec![1, 2]);
    assert_eq!(merged.g, vec![0.]);
}
#[test]
fn signed_joint_original_unsafe_ties_and_all_closed_zero_selector_choices_survive() {
    for bias in [-0.25, 0.] {
        let f = Fixture::new(bias);
        let (parts, _, _) = append(&f, 1);
        for mask in 0..8 {
            check_rows(
                &parts.problem,
                &point(
                    &f,
                    &parts,
                    Q::zero(),
                    [Q::zero(), Q::zero(), Q::zero()],
                    mask,
                ),
            );
        }
        for x in [-1.0_f64, -0.5, 0., 0.5, 1.] {
            let r = q(x.max(0.));
            check_rows_except(
                &parts.problem,
                &point(&f, &parts, q(x), [r.clone(), r.clone(), r], 0),
                Some(f.original().problem.num_rows() - 1),
            );
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
