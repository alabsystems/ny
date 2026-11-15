// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact geometry, original-encoder LP checks, and publication refusal controls.
use super::test_fixture::*;
use super::*;
use crate::MipResult;
use num_rational::BigRational as Q;
use num_traits::Zero;
#[test]
fn signed_phase_rows_strict_original_bigm_triangle_lp_and_actual_ay_certificate() {
    let f = Fixture::new(1.5);
    let (parts, report, unsafe_row) = append(&f);
    let phase = phase_prefix(&f, &parts, &report);
    let old = lp_floor(&phase, unsafe_row, true);
    let new = lp_floor(&phase, unsafe_row, false);
    let full = lp_floor(&parts, unsafe_row, false);
    assert!(full >= new - 1e-6);
    assert_eq!(
        report
            .signed
            .as_ref()
            .unwrap()
            .component_rows
            .domination
            .appended_rows,
        2
    );
    assert!((old + 0.5).abs() < 1e-6, "old={old}");
    assert!((new - 1. / 6.).abs() < 1e-6, "new={new}");
    let result = solve(parts);
    assert!(
        matches!(result, MipResult::Unsat { certified: true }),
        "{result:?}"
    );
}
#[test]
fn signed_phase_rows_exact_rational_dyadic_multipliers_close_the_actual_exported_rows() {
    let f = Fixture::new(1.5);
    let (full, report, _) = append(&f);
    let parts = phase_prefix(&f, &full, &report);
    let original = f.original();
    let encoding = f.encoding(&original);
    let heads = binding::head_columns(&encoding, &original).unwrap();
    let mut combined = vec![Q::zero(); parts.num_cols];
    let mut rhs = Q::zero();
    for row in parts.problem.rows().iter().rev().take(2) {
        let w = Q::new(43.into(), 128.into());
        rhs += &w * q(row.lb);
        for &(j, a) in &row.coeffs {
            combined[j] += &w * q(a);
        }
    }
    for j in [2, 5] {
        let selector = original.binary_vars[j];
        let upper = original
            .problem
            .rows()
            .iter()
            .find(|r| {
                r.ub == 1.
                    && r.lb == f64::NEG_INFINITY
                    && r.coeffs.contains(&(heads[j].0, 1.))
                    && r.coeffs.contains(&(selector.0, 1.))
                    && r.coeffs.contains(&(1 + j, -1.))
            })
            .unwrap();
        let w = Q::new(43.into(), 64.into());
        rhs -= &w * q(upper.ub);
        for &(i, a) in &upper.coeffs {
            combined[i] -= &w * q(a);
        }
        // Original affine equality cancels its preactivation in the caller frame.
        let affine = &original.problem.rows()[j];
        assert_eq!(affine.lb, affine.ub);
        rhs += &w * q(affine.lb);
        for &(i, a) in &affine.coeffs {
            combined[i] += &w * q(a);
        }
    }
    let mut objective = vec![Q::zero(); parts.num_cols];
    for (j, h) in heads.iter().enumerate() {
        objective[h.0] = q(f.w[1][j]);
    }
    let mut lower = q(f.b[1][0]) + rhs;
    for (i, (want, got)) in objective.iter().zip(&combined).enumerate() {
        let residual = want - got;
        if residual.is_zero() {
            continue;
        }
        let col = &parts.problem.cols()[i];
        assert!(
            col.lb.is_finite() && col.ub.is_finite(),
            "unpriced column{i}"
        );
        lower += (&residual * q(col.lb)).min(&residual * q(col.ub));
    }
    assert!(lower > q(0.15) && lower <= Q::new(5.into(), 32.into()));
}
fn graph_point(f: &Fixture, parts: &MipParts, x: f64, zero_phases: usize) -> Vec<Q> {
    let original = f.original();
    let encoding = f.encoding(&original);
    let heads = binding::head_columns(&encoding, &original).unwrap();
    let mut v = vec![Q::zero(); parts.num_cols];
    v[parts.input_vars[0].0] = q(x);
    for j in 0..6 {
        let z = f.w[0][j] * x + f.b[0][j];
        v[1 + j] = q(z);
        v[heads[j].0] = q(z.max(0.));
        let bit = if j == 2 { 0 } else { 1 };
        v[original.binary_vars[j].0] = q(if z == 0. && (j == 2 || j == 5) {
            ((zero_phases >> bit) & 1) as f64
        } else if z > 0. {
            1.
        } else {
            0.
        });
    }
    v[parts.output_vars[0].0] =
        q(f.b[1][0]) + (0..6).map(|j| q(f.w[1][j]) * &v[heads[j].0]).sum::<Q>();
    v
}
fn check_rows(parts: &MipParts, v: &[Q]) {
    for row in parts.problem.rows() {
        let lhs: Q = row.coeffs.iter().map(|(j, a)| q(*a) * &v[*j]).sum();
        assert!(!row.lb.is_finite() || lhs >= q(row.lb));
        assert!(!row.ub.is_finite() || lhs <= q(row.ub));
    }
}
#[test]
fn signed_phase_rows_original_unsafe_tie_and_both_closed_zero_cells_are_retained() {
    for bias in [-0.25, 0.] {
        let f = Fixture::new(bias);
        let (parts, _, _) = append(&f);
        for mask in 0..4 {
            check_rows(&parts, &graph_point(&f, &parts, 0., mask));
        }
        for x in [-1., -0.5, 0., 0.5, 1.] {
            check_rows(&parts, &graph_point(&f, &parts, x, 0));
        }
        let result = solve(parts);
        match result {
            MipResult::Sat { input_values, .. } => {
                assert!((-1. ..=1.).contains(&input_values[0]));
                assert!(bias <= 0.);
            }
            other => panic!("{other:?}"),
        }
    }
}
#[test]
fn signed_phase_rows_corruption_missing_coverage_and_foreign_binding_keep_old_rows() {
    for missing in [false, true] {
        let f = Fixture::new(1.5);
        let p = f.source();
        let c = cuts(&p);
        let row = c.signed.as_ref().unwrap();
        let parts = f.parts();
        let encoding = f.encoding(&parts);
        let original = f.original();
        let heads = binding::head_columns(&encoding, &original).unwrap();
        let replay = row.replay_inner(&c, end(), &mut || true, false).unwrap();
        let minima = replay.component_minima.unwrap();
        let mut conditional = replay.conditional.unwrap();
        if missing {
            conditional.values[0].clear();
        } else {
            conditional.values[0][0][0] += 1.;
        }
        let component_rows::Prepared::Rows(old) =
            component_rows::prepare(row, &c, &parts, &heads, &minima, end(), &mut || true).unwrap()
        else {
            panic!()
        };
        assert_eq!(old.len(), 2);
        let before = parts.problem.clone();
        let lifts = binding::with_phase_binding_test(&c, &parts, &encoding, |binding| {
            prepare(
                row,
                &c,
                &parts,
                binding,
                &conditional,
                &minima,
                &old,
                end(),
                &mut || true,
            )
        })
        .unwrap()
        .unwrap();
        assert_eq!(lifts, [] as [RowSpec; 0]);
        assert_eq!(parts.problem, before);
        assert_eq!(row.stats().component_rows.phase.stop_reason, "coverage");
        let foreign = f.parts();
        assert!(
            binding::with_phase_binding_test(&c, &parts, &encoding, |binding| prepare(
                row,
                &c,
                &foreign,
                binding,
                &conditional,
                &minima,
                &old,
                end(),
                &mut || true
            ))
            .unwrap()
            .is_none()
        );
    }
}
#[test]
fn signed_phase_rows_binding_excludes_relaxed_positive_and_stable_negative_selectors() {
    let f = Fixture::new(1.5);
    let p = f.source();
    let c = cuts(&p);
    let parts = f.parts();
    let enc = f.encoding(&parts);
    binding::with_phase_binding_test(&c, &parts, &enc, |b| {
        for j in [0, 1, 3, 4] {
            assert!(b.column(j).is_none());
        }
        assert_eq!(b.column(2), Some(parts.binary_vars[0]));
        assert_eq!(b.column(5), Some(parts.binary_vars[1]));
    })
    .unwrap();
    let mut foreign = f.parts();
    let head = foreign.input_vars[0];
    foreign.problem.fix_col(head, 0.);
    assert!(binding::with_phase_binding_test(&c, &foreign, &enc, |_| ()).is_none());
    let mut f = Fixture::new(1.5);
    f.w[0][2] = 0.;
    f.b[0][2] = 1.;
    f.bounds[0][2] = Bound::new(1., 1.);
    let p = f.source();
    let c = cuts(&p);
    let parts = f.original();
    let enc = f.encoding(&parts);
    binding::with_phase_binding_test(&c, &parts, &enc, |b| {
        assert!(b.column(2).is_none());
        assert!(b.column(5).is_some());
    })
    .unwrap();
}
#[test]
fn signed_phase_rows_work_refusal_pays_setup_preserves_old_admission_and_two_passes() {
    let f = Fixture::new(1.5);
    let p = f.source();
    let mut c = base(&p);
    let plan = plan();
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let row = make_row(&c, &plan, &b);
    let old = component_rows::quote(&c, &plan).unwrap();
    let old_setup = (p.hidden_dim + p.input_dim + plan.components.len() + 1) * 32;
    let phase_setup = setup(&c, &plan).unwrap();
    {
        let mut state = b.state.lock().unwrap();
        state.work = MAX_WORK - state.reserved - old.work - old_setup - phase_setup;
    }
    let before = b.state.lock().unwrap().work;
    component_rows::admit(&c, &plan, &b).unwrap();
    {
        let state = b.state.lock().unwrap();
        let report = &state.component_rows;
        assert!(report.admitted && !report.phase.admitted);
        assert_eq!(report.phase.stop_reason, "work");
        assert_eq!(report.phase.work_units, phase_setup);
    }
    assert_eq!(
        b.state.lock().unwrap().work,
        before + old_setup + phase_setup
    );
    c.signed = Some(row);
    let mut parts = f.parts();
    let enc = f.encoding(&parts);
    let report = c.append_to(&mut parts, &enc, end()).unwrap();
    assert_eq!(report.signed.unwrap().component_rows.appended_rows, 2);
    assert!(c
        .signed
        .as_ref()
        .unwrap()
        .replay_with_cache(&c, end(), &mut || true)
        .is_some());
    assert_eq!(b.state.lock().unwrap().source_replays, 2);
    assert_eq!(b.state.lock().unwrap().work, MAX_WORK);
}
#[test]
fn signed_phase_rows_payload_refusal_and_deadline_are_optional_without_quota_renewal() {
    let f = Fixture::new(1.5);
    let p = f.source();
    let c = base(&p);
    let plan = plan();
    let mut b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let row = make_row(&c, &plan, &b);
    b.admitted_bytes = MAX_BYTES - component_rows::quote(&c, &plan).unwrap().bytes;
    component_rows::admit(&c, &plan, &b).unwrap();
    assert!(b.state.lock().unwrap().component_rows.admitted);
    assert!(!b.state.lock().unwrap().component_rows.phase.admitted);
    assert_eq!(
        b.state.lock().unwrap().component_rows.phase.stop_reason,
        "payload"
    );
    let before = b.state.lock().unwrap().work;
    assert!(row
        .replay_inner(&c, Instant::now(), &mut || true, false)
        .is_none());
    assert_eq!(b.state.lock().unwrap().work, before);
    assert_eq!(b.state.lock().unwrap().source_replays, 0);
}
#[test]
fn signed_phase_rows_partial_final_mask_refuses_whole_publication_and_completed_rows_survive_second(
) {
    let f = Fixture::new(1.5);
    let p = f.source();
    let c = cuts(&p);
    let row = c.signed.as_ref().unwrap();
    let before = row.budget.state.lock().unwrap().checked_masks;
    assert!(row
        .replay_inner(
            &c,
            end(),
            &mut || row.budget.state.lock().unwrap().checked_masks < before + 3,
            false
        )
        .is_none());
    assert_eq!(row.stats().source_replays, 0);
    assert_eq!(row.stats().component_rows.phase.appended_rows, 0);
    let p = f.source();
    let c = cuts(&p);
    let mut parts = f.parts();
    let enc = f.encoding(&parts);
    c.append_to(&mut parts, &enc, end()).unwrap();
    let row = c.signed.as_ref().unwrap();
    let first = row.stats();
    assert_eq!(first.component_rows.phase.appended_rows, 2);
    let remaining = row.budget.remaining().unwrap();
    row.budget.charge(remaining).unwrap();
    assert!(row.replay_with_cache(&c, end(), &mut || true).is_some());
    let second = row.stats();
    assert_eq!(second.source_replays, 2);
    assert_eq!(second.component_rows.phase.appended_rows, 2);
    assert_eq!(second.component_rows.phase.stop_reason, "appended");
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    assert_eq!(row.stats().component_rows.phase.appended_rows, 2);
}

#[test]
fn signed_phase_rows_directed_favored_sides_subnormal_gain_and_overflow_are_conservative() {
    for pair in [[0., -1.], [-1., 0.], [0., f64::from_bits(2)]] {
        let minimum = pair[0].min(pair[1]);
        let (_, a, lower) = plane(minimum, pair).unwrap();
        for side in 0..2 {
            // Export is 2(F-Gx) + a*delta >= lower. Check both exact cells.
            assert!(q(lower) - q(a) * Q::from_integer(side.into()) <= q(2.) * q(pair[side]));
        }
    }
    let (_, a, _) = plane(0., [0., f64::from_bits(2)]).unwrap();
    assert!(a < 0. && a != 0.);
    assert_eq!(q(a), -q(2.) * q(f64::from_bits(1)));
    assert!(plane(-0.25 * f64::MAX, [-0.25 * f64::MAX, f64::MAX]).is_none());
    assert!(plane(0., [f64::NAN, 0.]).is_none());
}

#[test]
fn signed_phase_rows_conditional_geometry_cannot_be_reissued_under_a_fresh_ledger_or_witness() {
    let f = Fixture::new(1.5);
    let p = f.source();
    let first = cuts(&p);
    let second = cuts(&p);
    let row1 = first.signed.as_ref().unwrap();
    let row2 = second.signed.as_ref().unwrap();
    let old = row1
        .replay_inner(&first, end(), &mut || true, false)
        .unwrap()
        .conditional
        .unwrap();
    let checked = row2
        .replay_inner(&second, end(), &mut || true, false)
        .unwrap();
    let minima = checked.component_minima.unwrap();
    assert!(!old.valid(row2, &minima));
    let mut changed = checked.conditional.unwrap();
    changed.witness = Some(Arc::new((*row2.witness).clone()));
    assert!(!changed.valid(row2, &minima));
    assert_eq!(row1.stats().source_replays, 1);
    assert_eq!(row2.stats().source_replays, 1);
}

#[test]
fn signed_phase_rows_borrowed_binding_mint_and_interrupted_prepare_cannot_renew_work() {
    use std::cell::Cell;
    let f = Fixture::new(1.5);
    let p = f.source();
    let c = cuts(&p);
    let row = c.signed.as_ref().unwrap();
    let parts = f.parts();
    let enc = f.encoding(&parts);
    let authenticated = Cell::new(false);
    let after_auth_polls = Cell::new(0usize);
    let before = row.stats();
    binding::with_phase_binding_test_polled(
        &c,
        &parts,
        &enc,
        &mut || {
            if authenticated.get() {
                after_auth_polls.set(after_auth_polls.get() + 1);
                false
            } else {
                true
            }
        },
        &mut || authenticated.set(true),
        |binding| {
            assert!(binding.matches(&c, &parts));
            assert_eq!(binding.column(2), Some(parts.binary_vars[0]));
            assert_eq!(binding.column(5), Some(parts.binary_vars[1]));
            assert!(binding.column(usize::MAX).is_none());
        },
    )
    .unwrap();
    assert!(authenticated.get());
    assert_eq!(after_auth_polls.get(), 0);
    assert_eq!(row.stats().work_units, before.work_units);
    assert_eq!(
        row.stats().component_rows.reserved_work_units,
        before.component_rows.reserved_work_units
    );
    let mut parts = f.parts();
    let enc = f.encoding(&parts);
    let original = parts.problem.clone();
    let held = row.budget.state.lock().unwrap().component_reserved;
    assert!(held > 0);
    // The old component rows are fully prepared before the first phase lookup.
    assert!(c
        .append_with_poll(&mut parts, &enc, end(), &mut || row
            .stats()
            .component_rows
            .prepared_rows
            < 2)
        .is_none());
    assert_eq!(parts.problem, original);
    let first = row.stats();
    assert_eq!(first.source_replays, 1);
    assert_eq!(first.component_rows.collection_attempts, 1);
    assert_eq!(first.component_rows.phase.collection_attempts, 1);
    assert_eq!(first.component_rows.reserved_work_units, 0);
    assert_eq!(first.component_rows.phase.prepared_rows, 0);
    assert_eq!(first.component_rows.phase.appended_rows, 0);
    let phase_work = first.component_rows.phase.work_units;
    assert!(row.replay_with_cache(&c, end(), &mut || true).is_some());
    let second = row.stats();
    assert_eq!(second.source_replays, 2);
    assert_eq!(second.component_rows.phase.work_units, phase_work);
    assert_eq!(second.component_rows.phase.collection_attempts, 1);
    let total = second.work_units;
    assert!(c.append_to(&mut parts, &enc, end()).is_none());
    let last = row.stats();
    assert_eq!(last.work_units, total);
    assert_eq!(last.component_rows.phase.work_units, phase_work);
    assert_eq!(last.component_rows.phase.collection_attempts, 1);
    assert_eq!(parts.problem, original);
    assert!(last.work_units <= MAX_WORK);
}
