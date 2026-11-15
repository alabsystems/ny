// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact original-affine reuse: complete payment, sealed ownership, atomic export.
use super::super::test_fixture::{self as fixture, end, point, q, Fixture};
use super::*;
use num_rational::BigRational as Q;

fn copies() -> Fixture {
    // Forty original negative ReLUs require dense admission. The negative
    // margin keeps the old root open independently of the proposal's strength.
    let (d, h) = (2, 42);
    let mut w = vec![1.; d * h];
    w[1] = 0.;
    w[2] = 0.;
    let mut c = vec![0.; 2 * h];
    c[..2].fill(40.);
    c[2..h].fill(-1.);
    let mut bounds = vec![Bound::new(-2., 2.); h];
    bounds[..2].fill(Bound::new(-1., 1.));
    Fixture {
        w: vec![w, c],
        b: vec![vec![0.; h], vec![-0.25, 0.]],
        dims: vec![d, h, 2],
        input: vec![Bound::new(-1., 1.); d],
        bounds: vec![bounds],
    }
}
fn actual(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    let answer = propose_root_mip_cuts_with_signed_refinement_attempt(p, end(), Duration::ZERO);
    assert!(answer.primal.same_call && answer.primal.decoded && answer.primal.banked);
    let cuts = answer.cuts.unwrap();
    assert_eq!(cuts.proposal_lp_calls, 1);
    cuts
}
fn paid(cuts: &HullRootMipCuts<'_>) -> (resources::Ticket, Vec<usize>) {
    let row = cuts.signed.as_ref().unwrap();
    row.replay_inner(cuts, row.authority(), &mut || true, false)
        .unwrap();
    let account = cuts.primal.as_ref().unwrap().lock().unwrap();
    let ticket = resources::admit_point(row, cuts, row.budget.search_end, Some(&account))
        .unwrap_or_else(|| panic!("dense alias admission refused: {:?}", row.stats()));
    let negative = account
        .point(cuts.problem, ticket.end, &row.budget.state)
        .unwrap()
        .negative
        .to_vec();
    assert_eq!((ticket.targets, ticket.alias_limit()), (1, 40));
    assert_eq!(negative, (2..42).collect::<Vec<_>>());
    resources::begin_advice(row, &ticket).unwrap();
    (ticket, negative)
}
fn proposal() -> kernel::Proposal {
    kernel::Proposal {
        head: 2,
        positive: vec![0, 1],
        lambda: vec![1., 1.],
        v: vec![0.; 2],
    }
}

#[test]
fn exact_affine_alias_quote_prepays_complete_domain_and_row_batch_under_original_cap() {
    assert_eq!(quote(2, 2, 40), Some((7_104, 12_416)));
    assert_eq!(quote(2, 2, 0), Some((0, 0)));
    assert!(quote(2, 2, 513).is_none());
    assert!(quote(32, 32, 40).is_none());
    assert!(quote(usize::MAX, 2, 40).is_none());
    let f = copies();
    let p = f.source();
    let cuts = actual(&p);
    let row = cuts.signed.as_ref().unwrap();
    let (ticket, _) = paid(&cuts);
    let quoted = ticket.dense_quote().unwrap();
    let canonical = dense::quote_with_aliases(
        &p,
        row.plan.components.len(),
        row.stats().component_rows.domination.capture_work_units,
        2,
        0,
        2,
    )
    .unwrap();
    assert_eq!(quoted.total_work - canonical.total_work, 7_104);
    assert_eq!(quoted.bytes - canonical.bytes, 12_416);
    let state = row.budget.state.lock().unwrap();
    let report = &state.component_rows.domination;
    assert_eq!(
        (
            report.funded_alias_heads,
            report.alias_work_units,
            report.alias_bytes
        ),
        (40, 7_104, 12_416)
    );
    assert_eq!(report.total_optional_work_units, quoted.total_work);
    assert!(quoted.total_work <= resources::MAX_DOMINATION_WORK);
    assert!(state.work + state.reserved + state.component_reserved <= MAX_WORK);
    assert_eq!(
        (
            report.support_attempts,
            report.checked_alias_rows,
            report.prepared_rows
        ),
        (0, 0, 0)
    );
}

#[test]
fn exact_affine_alias_collection_requires_original_bits_and_sorted_negative_members() {
    let p = copies().source();
    let negative: Vec<_> = (2..42).collect();
    let mut polls = 0;
    assert_eq!(
        collect(&p, 2, &negative, 40, end(), &mut || {
            polls += 1;
            true
        }),
        Some(negative.clone())
    );
    assert_eq!(polls, 40 * (p.input_dim + 1) + 1);
    for mode in 0..3 {
        let mut changed = p.clone();
        match mode {
            0 => changed.b1[41] = f64::from_bits(1),
            1 => changed.w1[41 * 2 + 1] = f64::from_bits(1f64.to_bits() + 1),
            _ => changed.b1[41] = -0.,
        }
        assert_eq!(
            collect(&changed, 2, &negative, 40, end(), &mut || true),
            Some((2..41).collect())
        );
    }
    for mode in 0..6 {
        let mut members = negative.clone();
        match mode {
            0 => members.swap(0, 1),
            1 => members[39] = members[38],
            2 => members[39] = p.hidden_dim,
            3 => members[0] = 0,
            4 => {
                members.remove(0);
            }
            _ => {
                members.pop();
            }
        }
        assert!(collect(&p, 2, &members, 40, end(), &mut || true).is_none());
    }
    assert!(collect(&p, 2, &negative[1..], 39, end(), &mut || true).is_none());
    for value in [f64::NAN, f64::NEG_INFINITY, 0., 1.] {
        let mut changed = p.clone();
        changed.c[41] = value;
        assert!(collect(&changed, 2, &negative, 40, end(), &mut || true).is_none());
    }
    let mut malformed = p.clone();
    malformed.w1.pop();
    assert!(collect(&malformed, 2, &negative, 40, end(), &mut || true).is_none());
    assert_eq!(
        collect(&p, 2, &negative, 40, end(), &mut || true),
        Some(negative)
    );
}

#[test]
fn exact_affine_alias_cancellation_and_late_clock_keep_paid_debt_and_single_support() {
    for expired in [false, true] {
        let f = copies();
        let p = f.source();
        let cuts = actual(&p);
        let row = cuts.signed.as_ref().unwrap();
        let (ticket, negative) = paid(&cuts);
        let before = {
            let s = row.budget.state.lock().unwrap();
            (s.work, s.reserved, s.checks)
        };
        // Six support polls, then all 120 member/coordinate polls, then the
        // collector's final poll. Cancellation there must discard all aliases.
        let final_poll = 1 + 2 + 2 + 1 + 40 * 3 + 1;
        let mut polls = 0;
        let deadline = if expired {
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("overflow when subtracting duration from instant")
        } else {
            ticket.end
        };
        assert!(kernel::check_with_aliases(
            row,
            &cuts,
            &ticket,
            proposal(),
            &negative,
            deadline,
            &mut || {
                polls += 1;
                polls < final_poll
            }
        )
        .is_none());
        assert_eq!(polls, if expired { 0 } else { final_poll });
        assert!(kernel::check_with_aliases(
            row,
            &cuts,
            &ticket,
            proposal(),
            &negative,
            end(),
            &mut || true
        )
        .is_none());
        assert!(resources::begin_advice(row, &ticket).is_none());
        let state = row.budget.state.lock().unwrap();
        assert_eq!((state.work, state.reserved, state.checks), before);
        let report = &state.component_rows.domination;
        assert_eq!(
            (
                report.support_attempts,
                report.completed_supports,
                report.checked_alias_rows
            ),
            (1, 0, 0)
        );
        assert_eq!(report.prepared_rows, 0);
    }
}

#[test]
fn exact_affine_alias_token_binds_members_quota_source_ledger_and_deadline() {
    let f = copies();
    let p = f.source();
    let cuts = actual(&p);
    let row = cuts.signed.as_ref().unwrap();
    let (ticket, negative) = paid(&cuts);
    let mut token = kernel::check_with_aliases(
        row,
        &cuts,
        &ticket,
        proposal(),
        &negative,
        ticket.end,
        &mut || true,
    )
    .unwrap();
    assert!(token.owned(row));
    let aliases = token.aliases.clone();
    for changed in [vec![], vec![3], vec![2, 2], vec![3, 2], vec![2, usize::MAX]] {
        token.aliases = changed;
        assert!(!token.owned(row));
    }
    token.aliases = aliases;
    for limit in [0, 39, 41, usize::MAX] {
        token.alias_limit = limit;
        assert!(!token.owned(row));
    }
    token.alias_limit = 40;
    token.source ^= 1;
    assert!(!token.owned(row));
    token.source ^= 1;
    let ledger = Arc::clone(&token.ledger);
    token.ledger = Arc::new(std::sync::Mutex::new(Counts::default()));
    assert!(!token.owned(row));
    token.ledger = ledger;
    let witness = Arc::clone(&token.witness);
    token.witness = Arc::new((*witness).clone());
    assert!(!token.owned(row));
    token.witness = witness;
    let signature = Arc::clone(&token.signature);
    token.signature = Arc::new((*signature).clone());
    assert!(!token.owned(row));
    token.signature = signature;
    let accepted_end = token.end;
    token.end += Duration::from_secs(1);
    assert!(!token.owned(row));
    token.end = accepted_end;
    token.bits[0] ^= 1;
    assert!(!token.owned(row));
    token.bits[0] ^= 1;
    assert!(token.owned(row));
    assert_eq!(row.stats().component_rows.domination.completed_supports, 1);
}

#[test]
fn exact_affine_alias_export_is_atomic_and_once_only_for_all_original_columns() {
    let mut complete_polls = None;
    for mode in 0..3 {
        let cancel = mode == 1;
        let expired = mode == 2;
        let f = copies();
        let p = f.source();
        let parts = f.parts();
        let cuts = actual(&p);
        let row = cuts.signed.as_ref().unwrap();
        let (ticket, negative) = paid(&cuts);
        let token = kernel::check_with_aliases(
            row,
            &cuts,
            &ticket,
            proposal(),
            &negative,
            ticket.end,
            &mut || true,
        )
        .unwrap();
        assert_eq!(token.aliases, negative);
        let before_rows = parts.problem.num_rows();
        let paid_work = row.budget.state.lock().unwrap().work;
        let mut polls = 0;
        let deadline = if expired {
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("overflow when subtracting duration from instant")
        } else {
            end()
        };
        let result = binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
            let rows = kernel::export(
                row,
                &cuts,
                &parts,
                binding,
                std::slice::from_ref(&token),
                deadline,
                &mut || {
                    polls += 1;
                    !cancel || Some(polls) != complete_polls
                },
            );
            if let Some(rows) = &rows {
                assert_eq!(rows.len(), 40);
                for (r, &head) in rows.iter().zip(&negative) {
                    assert_eq!(
                        r.coeffs,
                        vec![
                            (binding.head_column(head).unwrap().0, 1.),
                            (binding.head_column(0).unwrap().0, -1.),
                            (binding.head_column(1).unwrap().0, -1.)
                        ]
                    );
                    assert_eq!(r.ub.to_bits(), token.upper.to_bits());
                    assert_eq!(r.lb, f64::NEG_INFINITY);
                }
            }
            assert!(kernel::export(
                row,
                &cuts,
                &parts,
                binding,
                std::slice::from_ref(&token),
                end(),
                &mut || true
            )
            .is_none());
            rows
        })
        .unwrap();
        assert_eq!(parts.problem.num_rows(), before_rows);
        let report = row.stats().component_rows.domination;
        assert_eq!((report.export_attempts, report.completed_supports), (1, 1));
        assert_eq!(row.budget.state.lock().unwrap().work, paid_work);
        if cancel || expired {
            assert_eq!(Some(polls), if expired { Some(0) } else { complete_polls });
            assert!(result.is_none());
            assert_eq!((report.prepared_rows, report.prepared_alias_rows), (0, 0));
        } else {
            complete_polls = Some(polls);
            let rows = result.unwrap();
            assert_eq!(
                (
                    report.prepared_rows,
                    report.prepared_alias_rows,
                    report.prepared_terms
                ),
                (40, 40, 120)
            );
            for x in [-1., -0.5, 0., 0.5, 1.] {
                for y in [-1., -0.5, 0., 0.5, 1.] {
                    let values = point(&f, &parts, &[q(x), q(y)], None, 0);
                    for r in &rows {
                        let lhs: Q = r.coeffs.iter().map(|(j, a)| q(*a) * &values[*j]).sum();
                        assert!(lhs <= q(r.ub));
                    }
                }
            }
        }
    }
}

#[test]
fn exact_affine_alias_no_match_still_pays_and_sparse_ticket_cannot_buy_reuse() {
    let mut f = copies();
    // Preserve the real graph and its original refinement workload. Strict
    // bit equality conservatively excludes these numerically equal biases;
    // changed weights need not leave budget for this independent stage.
    f.b[0][3..].fill(-0.);
    let p = f.source();
    assert_eq!(p.b1[2].to_bits(), 0);
    assert!(p.b1[3..].iter().all(|b| b.to_bits() == (-0f64).to_bits()));
    let cuts = actual(&p);
    let row = cuts.signed.as_ref().unwrap();
    let (ticket, negative) = paid(&cuts);
    let before = row.budget.state.lock().unwrap().work;
    let token = kernel::check_with_aliases(
        row,
        &cuts,
        &ticket,
        proposal(),
        &negative,
        ticket.end,
        &mut || true,
    )
    .unwrap();
    assert_eq!(token.aliases, vec![2]);
    assert_eq!(token.alias_limit, 40);
    assert_eq!(row.budget.state.lock().unwrap().work, before);
    let report = row.stats().component_rows.domination;
    assert_eq!(
        (
            report.funded_alias_heads,
            report.checked_alias_rows,
            report.alias_work_units
        ),
        (40, 1, 7_104)
    );

    let f = Fixture::sparse(-0.25);
    let p = f.source();
    let cuts = fixture::cuts(&p, &[2]);
    let row = cuts.signed.as_ref().unwrap();
    row.replay_inner(&cuts, end(), &mut || true, false).unwrap();
    let ticket = resources::admit(row, &cuts, end()).unwrap();
    assert_eq!(ticket.alias_limit(), 0);
    assert!(
        kernel::check_with_aliases(row, &cuts, &ticket, proposal(), &[2], end(), &mut || true)
            .is_none()
    );
    assert!(kernel::check(row, &cuts, &ticket, proposal(), end(), &mut || true).is_none());
    let report = row.stats().component_rows.domination;
    assert_eq!(
        (
            report.support_attempts,
            report.completed_supports,
            report.checked_alias_rows
        ),
        (1, 0, 0)
    );
}

#[test]
fn dense_quote_retained_total_matches_checked_debt_at_the_original_cap() {
    let p = copies().source();
    let cuts = actual(&p);
    let row = cuts.signed.as_ref().unwrap();
    let before = row.stats();
    let base = dense::quote_with_aliases(&p, 1, 0, 2, 0, 2).unwrap();
    let capture = resources::MAX_DOMINATION_WORK
        .checked_sub(base.incremental_work)
        .unwrap();
    let exact = dense::quote_with_aliases(&p, 1, capture, 2, 0, 2).unwrap();
    assert_eq!(
        exact.checked_total(capture),
        Some(resources::MAX_DOMINATION_WORK)
    );
    let mut wrong_total = exact.clone();
    wrong_total.total_work -= 1;
    assert_eq!(wrong_total.checked_total(capture), None);
    let mut overflow = exact.clone();
    overflow.incremental_work = usize::MAX;
    assert_eq!(overflow.checked_total(capture), None);
    assert_eq!(exact.checked_total(usize::MAX), None);
    assert!(dense::quote_with_aliases(&p, 1, capture.checked_add(1).unwrap(), 2, 0, 2).is_none());
    assert_eq!(row.stats().work_units, before.work_units);
    assert_eq!(row.stats().admitted_bytes, before.admitted_bytes);
    let (ticket, _) = paid(&cuts);
    assert!(ticket.dense_quote().is_some());
}
