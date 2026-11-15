// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original graph/IR/certificate controls, authored externally and unexecuted.
use super::test_fixture::*;
use super::*;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

#[test]
fn signed_domination_all_half_whole_scalar_and_negative_projections_keep_a_rational_gap() {
    // Every allowed one/two-component scalar projection can use its own lift.
    // These are not full output hulls, arbitrary repartitions or three-component unions.
    for n in [8, 16] {
        for mass in [0, 12, 24] {
            let value = q(mass as f64) * q(3. / 16.) - q(n as f64) * q(0.25);
            let r = &value / q((mass - n) as f64);
            assert!(r >= Q::zero() && r <= q(0.5));
            let positive_atom = q(2.) * &r;
            assert!(positive_atom >= Q::zero() && positive_atom <= q(1.));
            assert_eq!(q((mass - n) as f64) * r, value);
        }
    }
    // One common negative-graph mixture x=+/-1/2 has all N=1/4 and delta=1/2.
    // Hence no valid negative-only individual/group/SDP row excludes that projection.
    assert_eq!((q(0.5) + Q::zero()) / q(2.), q(0.25));
    assert_eq!(q(24.) * q(3. / 16.) - q(24.) * q(0.25) + q(0.25), q(-1.25));
    assert_eq!(q(0.25) - q(24.) * q(EPS), q(5. / 32.));
}

#[test]
fn signed_domination_sparse_two_positive_row_strictly_exceeds_single_partner_center_planes() {
    let f = Fixture::sparse(0.25);
    let p = f.source();
    // For any lambda>=0, every affine majorant has center intercept at least
    // (|1-lambda|+1)/2 >=1/2. These rational values exercise both sides of its kink.
    for lambda in [0., 0.5, 1., 2., 4.] {
        let unavoidable = ((q(1.) - q(lambda)).abs() + q(1.)) / q(2.);
        assert!(unavoidable >= q(0.5));
        for positive in [0, 1] {
            let proposal = kernel::Proposal {
                head: 2,
                positive: vec![positive],
                lambda: vec![lambda],
                v: vec![
                    0.5 * (1. - lambda * usize::from(positive == 0) as f64),
                    0.5 * (1. - lambda * usize::from(positive == 1) as f64),
                ],
            };
            let result = kernel::upper(&p, &proposal, end(), &mut || true);
            if lambda == 0. {
                // The full mathematical family includes zero; the nontrivial-row API declines it.
                assert!(result.is_none());
            } else {
                assert!(q(result.unwrap()) >= unavoidable);
            }
        }
    }
    let proposal = kernel::Proposal {
        head: 2,
        positive: vec![0, 1],
        lambda: vec![1., 1.],
        v: vec![0.; 2],
    };
    let upper = kernel::upper(&p, &proposal, end(), &mut || true).unwrap();
    assert!((0. ..1e-12).contains(&upper));
    let original = f.parts();
    let mut center = point(
        &f,
        &original,
        &[Q::zero(), Q::zero()],
        Some(&[Q::zero(), Q::zero(), q(0.5)]),
        0,
    );
    for col in f.original().binary_vars {
        center[col.0] = q(0.5);
    }
    rows_hold(&original.problem, &center, None);
    assert_eq!(center[original.output_vars[0].0], q(-0.25));
    assert!(q(0.5) > q(upper));
}

#[test]
fn signed_domination_nonidentical_original_encoder_all_allocations_close_with_actual_ay() {
    let f = Fixture::large(0.25);
    let original = f.parts();
    let unsafe_row = f.encoding().unsafe_row.0;
    let mut old = original.problem.clone();
    let mut domination_rows = Vec::new();
    let allocations = [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ];
    for allocation in allocations {
        let (parts, report, before) = append(&f, &allocation);
        let s = report.signed.as_ref().unwrap();
        let dom = &s.component_rows.domination;
        assert!(dom.admitted && dom.attempted);
        assert_eq!(dom.funded_targets, 24);
        assert_eq!(dom.completed_supports, 24);
        assert_eq!(dom.appended_rows, 24);
        assert_eq!(dom.checker_slots, 1);
        assert_eq!(
            dom.sparse_fits, 0,
            "one positive partner requires no sparse fit"
        );
        assert!(dom.work_units <= 664_696 && dom.prepared_head_terms >= 48);
        assert!(
            s.lower.unwrap() < 0.,
            "new graph cuts must not replace old scalar authority"
        );
        let joint = s.component_rows.joint.appended_rows;
        let dom_end = parts.problem.num_rows() - joint;
        let dom_start = dom_end - dom.appended_rows;
        assert!(dom_start >= before);
        for row in parts.problem.rows()[before..dom_start]
            .iter()
            .chain(parts.problem.rows()[dom_end..].iter())
        {
            old.add_row(
                row.lb,
                row.ub,
                row.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
            );
        }
        if domination_rows.is_empty() {
            domination_rows = parts.problem.rows()[dom_start..dom_end].to_vec();
        }
    }
    let fake = fractional(&f, &original);
    rows_hold(&old, &fake, None);
    assert_eq!(fake[original.output_vars[0].0], q(-1.25));
    let before = lp(&old, &original, unsafe_row);
    assert!(before <= -1.249999, "old root={before}");
    for row in domination_rows {
        old.add_row(
            row.lb,
            row.ub,
            row.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
        );
    }
    let after = lp(&old, &original, unsafe_row);
    assert!((after - 5. / 32.).abs() < 1e-6, "domination root={after}");
    let (parts, _, _) = append(&f, &[1, 1, 0]);
    let result = solve(parts);
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}

#[test]
fn signed_domination_sparse_original_encoder_gap_and_actual_ay_keep_old_scalar_owner() {
    let f = Fixture::sparse(0.25);
    let (parts, report, before) = append(&f, &[2]);
    let s = report.signed.as_ref().unwrap();
    let dom = &s.component_rows.domination;
    assert_eq!(dom.appended_rows, 1);
    assert!(dom.sparse_fits > 0);
    assert_eq!(dom.prepared_head_terms, 3);
    assert!(s.lower.unwrap() < 0.);
    let dom_end = parts.problem.num_rows() - s.component_rows.joint.appended_rows;
    let dom_start = dom_end - dom.appended_rows;
    assert!(dom_start >= before);
    let mut old = f.parts().problem;
    for row in parts.problem.rows()[before..dom_start]
        .iter()
        .chain(parts.problem.rows()[dom_end..].iter())
    {
        old.add_row(
            row.lb,
            row.ub,
            row.coeffs.iter().map(|&(j, a)| (crate::ir::Col(j), a)),
        );
    }
    let original = f.parts();
    let unsafe_row = f.encoding().unsafe_row.0;
    assert!(lp(&old, &original, unsafe_row) <= -0.249999);
    assert!((lp(&parts.problem, &parts, unsafe_row) - 0.25).abs() < 1e-6);
    let result = solve(parts);
    assert!(
        matches!(result, crate::MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}

#[test]
fn signed_domination_original_unsafe_ties_and_closed_zero_graphs_remain_actual_sat() {
    for bias in [24. * EPS, 24. * EPS - 0.25] {
        let f = Fixture::large(bias);
        let (parts, report, _) = append(&f, &[1, 1, 0]);
        assert_eq!(
            report
                .signed
                .as_ref()
                .unwrap()
                .component_rows
                .domination
                .appended_rows,
            24
        );
        let all_one = vec![q(1.); 25];
        let original = point(&f, &parts, &all_one, None, 0);
        rows_hold(&parts.problem, &original, None);
        assert_eq!(original[parts.output_vars[0].0], q(bias - 24. * EPS));
        for zero_mask in [0, usize::MAX, 0x1555555, 0xaaaaaa] {
            rows_hold(
                &parts.problem,
                &point(&f, &parts, &vec![Q::zero(); 25], None, zero_mask),
                Some(f.encoding().unsafe_row.0),
            );
        }
        let result = solve(parts);
        match result {
            crate::MipResult::Sat { input_values, .. } => {
                let inputs: Vec<_> = input_values.iter().map(|&x| q(x)).collect();
                let original = f.original();
                let replay = point(&f, &original, &inputs, None, 0);
                rows_hold(&original.problem, &replay, None);
                assert!(replay[original.output_vars[0].0] <= Q::zero());
            }
            other => panic!("{other:?}"),
        }
    }
    for bias in [0., -0.25] {
        let f = Fixture::sparse(bias);
        let (parts, _, _) = append(&f, &[2]);
        for mask in 0..8 {
            rows_hold(
                &parts.problem,
                &point(&f, &parts, &[Q::zero(), Q::zero()], None, mask),
                None,
            );
        }
        assert!(matches!(solve(parts), crate::MipResult::Sat { .. }));
    }
}
