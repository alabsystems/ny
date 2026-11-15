// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Integration controls in native_publication, with real paid admission.
//! Supplied checked geometry is distinct from automatic discovery and native LP results.
use super::*;
use crate::single_hidden_hull::{certify_hull_lower_bound, HullMarginProblem, HullMultipliers};
use ny_core::Bound;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[path = "single_hidden_hull_mip_signed_native_selected_tests.rs"]
mod selected_tests;

static NATIVE_TESTS: Mutex<()> = Mutex::new(());
pub(super) fn native_test_guard() -> MutexGuard<'static, ()> {
    // This mutex protects scheduling only; it stores no fixture state. A
    // failed assertion must remain failed without preventing later independent
    // controls from exercising their real paths and reporting their outcomes.
    NATIVE_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    fn zero_unary() -> Self {
        let e = 1. / 1024.;
        Self {
            // Component: 1 - ReLU(x) - ReLU(-x). The stable-live
            // leftover preserves a real original-LP activity source.
            weights: vec![
                vec![
                    0., 0., 0., 0.25, 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0.,
                ],
                vec![1., 1., -1., -1., -e, 0., 0., 0., 0., 0.],
            ],
            biases: vec![vec![1., 1., 0., 0., -1.], vec![-2., 0.]],
            bounds: vec![vec![
                Bound::new(1., 1.),
                Bound::new(0.75, 1.25),
                Bound::new(-1., 1.),
                Bound::new(-1., 1.),
                Bound::new(-1., -1.),
            ]],
        }
    }
    fn new() -> Self {
        let e = 1. / 1024.;
        Self {
            weights: vec![
                vec![
                    1.,
                    0.,
                    0.,
                    0.,
                    1.,
                    0.,
                    1.,
                    0.,
                    e,
                    0.,
                    1.,
                    2. * e,
                    0.,
                    0.,
                    0.,
                ],
                vec![1., 1., -1., -1., -e, 0., 0., 0., 0., 0.],
            ],
            biases: vec![vec![0., 0., 0., 0., -1.], vec![11. / 16., 0.]],
            bounds: vec![vec![
                Bound::new(-1., 1.),
                Bound::new(-1., 1.),
                Bound::new(-(1. + e) as f32, (1. + e) as f32),
                Bound::new(-(1. + 2. * e) as f32, (1. + 2. * e) as f32),
                Bound::new(-1., -1.),
            ]],
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            3,
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; 3],
            vec![1.; 3],
        )
        .unwrap()
    }
    fn original(&self) -> MipParts {
        let mut encoder = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &[3, 5, 2],
            &[Bound::new(-1., 1.); 3],
            &self.bounds,
        )
        .unwrap();
        encoder.constrain_output_leq_row(0, 1).unwrap();
        encoder.into_parts()
    }
    fn encoding(&self) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.weights,
            biases: &self.biases,
            dimensions: &[3, 5, 2],
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(self.original().problem.num_rows() - 1),
        }
    }
}
fn supplied_plan() -> Plan {
    Plan {
        components: vec![
            Component {
                negative: vec![2, 3],
                positive: vec![0, 1],
                positive_units: vec![2, 2],
                g: vec![0.; 3],
                alpha: vec![0., 0., 0., 0., 0., 1., 0.75, 1.],
            },
            Component {
                negative: vec![4],
                positive: vec![],
                positive_units: vec![],
                g: vec![0.; 3],
                alpha: vec![],
            },
        ],
        leftover: vec![],
    }
}
fn paid(p: &HullMarginProblem) -> HullRootMipCuts<'_> {
    paid_with_plan(p, supplied_plan())
}
fn paid_with_plan(p: &HullMarginProblem, plan: Plan) -> HullRootMipCuts<'_> {
    paid_with_policy(p, plan, false)
}
fn paid_with_policy(p: &HullMarginProblem, plan: Plan, conditional: bool) -> HullRootMipCuts<'_> {
    let until = end();
    // Actual original root LP supplies this opaque count receipt and activity.
    // No record_root_lp_counts call, synthetic raw primal or receipt constructor.
    let mut attempt = primal_capture::Attempt::new_with_native_counts(p, until);
    let mut cuts = propose_root_mip_cuts_inner(p, until, Some(&mut attempt))
        .expect("one real original root LP");
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert!(cuts.lp_primal_report().admitted && cuts.lp_primal_report().same_call);
    // Only the constructor can mint NativeConstruction before Budget escapes.
    // Capture already owns another ledger Arc; a strong-count test is invalid.
    let mut b = if conditional {
        Budget::new_with_portfolio_conditions(&cuts, until, Duration::ZERO)
    } else {
        Budget::new_with_native_rows(&cuts, until, Duration::ZERO)
    }
    .unwrap();
    {
        let counts = b.state.lock().unwrap();
        assert!(counts.native_rows.reserve_attempted);
        assert!(counts.native_rows.reservation.is_some());
        assert!(counts.native_rows.admission.is_none());
    }
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    root_coupling::admit(&cuts, &plan, cost, &mut b).unwrap();
    let seal = Arc::new(signature(&cuts).collect());
    component_rows::admit(&cuts, &plan, &b).unwrap();
    assert!(b.state.lock().unwrap().component_rows.phase.cube.admitted);
    let before = b.state.lock().unwrap().work;
    fund(&cuts, &b).expect("real full native work debit");
    let funding = funding(&b);
    {
        let counts = b.state.lock().unwrap();
        assert!(counts.work > before && counts.native_rows.admission.is_some());
        assert!(counts.native_rows.fund_attempted);
        assert_eq!(
            counts.work - before,
            funding.shared_work_units + funding.funded_calls * funding.per_call_work_units
        );
    }
    cuts.signed = Some(row(&cuts, &plan, checked, cost, &b, &seal));
    cuts
}

#[test]
fn losing_cube_conditional_funds_one_native_call_and_preserves_compatibility() {
    let _guard = native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    for conditional in [false, true] {
        let cuts = paid_with_policy(&p, supplied_plan(), conditional);
        let b = &cuts.signed.as_ref().unwrap().budget;
        let before = cuts.native_funding_report();
        assert_eq!(before.funded_calls, if conditional { 1 } else { 2 });
        assert!(before.setup_work_units > 0 && before.admitted_bytes > 0);
        assert_eq!(
            before.prepaid_work_units,
            before.setup_work_units
                + before.shared_work_units
                + before.funded_calls * before.per_call_work_units
        );
        assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
        let debt = b.state.lock().unwrap().work;
        assert!(fund(&cuts, b).is_none());
        assert_eq!(b.state.lock().unwrap().work, debt);
        assert_eq!(cuts.native_funding_report(), before);
    }
}

#[test]
fn losing_cube_consumed_conditional_credit_declines_next_hook_without_mutation() {
    let _guard = native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let cuts = paid_with_policy(&p, supplied_plan(), true);
    let _parts = append(&cuts, &f);
    let owner = cuts.signed.as_ref().unwrap();
    let pair = published(&cuts);
    let token = bind(owner, &cuts, pair.end(), &mut || true).unwrap();
    let bounds = preactivation_bounds(&p);
    let roles = classify(&p, &bounds);
    let mut first = build_hull_lp(&p, &roles, &bounds).unwrap();
    let before = first.model.num_rows();
    let mut calls = 0;
    let funded = cuts.native_funding_report();
    let result = token.with_node(
        &p,
        &[],
        &mut first,
        &[],
        pair.end(),
        &mut || true,
        |_, _| None,
        |actual, _| {
            calls += 1;
            assert_eq!(actual.model.num_rows(), before + 2);
        },
    );
    assert!(matches!(result, bank::Visit::Committed(Some(()))));
    assert_eq!(calls, 1);
    let mut next = build_hull_lp(&p, &roles, &bounds).unwrap();
    let result = token.with_node(
        &p,
        &[],
        &mut next,
        &[],
        pair.end(),
        &mut || true,
        |_, _| None,
        |_, _| {
            calls += 1;
        },
    );
    assert!(matches!(result, bank::Visit::Declined));
    assert_eq!(calls, 1);
    assert_eq!(next.model.num_rows(), before);
    assert_eq!(cuts.native_funding_report(), funded);
    assert_eq!(
        cuts.native_rows_report(),
        SignedNativeRowsReport {
            attempted: 1,
            committed: 1,
            priced: 0
        }
    );
}
fn published(cuts: &HullRootMipCuts<'_>) -> Published {
    cuts.signed
        .as_ref()
        .unwrap()
        .budget
        .state
        .lock()
        .unwrap()
        .native_rows
        .published
        .clone()
        .expect("actual complete unary+Cube publication")
}
fn append(cuts: &HullRootMipCuts<'_>, f: &Fixture) -> Box<MipParts> {
    // Context authenticates this stable address; do not move stack Parts afterward.
    let mut parts = Box::new(f.original());
    let prefix = parts.problem.clone();
    let binaries = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let report = cuts
        .append_to(parts.as_mut(), &f.encoding(), end())
        .unwrap();
    let stats = report.signed.as_ref().unwrap();
    assert_eq!(report.proposal_lp_calls, 1);
    assert_eq!(stats.source_replays, 1);
    assert!(stats.component_rows.phase.cube.appended_rows > 0);
    assert_eq!(parts.problem.cols(), prefix.cols());
    assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
    assert_eq!(parts.binary_vars, binaries);
    assert_eq!(parts.binary_widths, widths);
    assert_eq!(parts.problem.num_rows() - prefix.num_rows(), report.cuts);
    assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
    let pair = published(cuts);
    assert_eq!(pair.payload.context.component, 0);
    assert_eq!(
        pair.payload.context.parts,
        std::ptr::from_ref(parts.as_ref()) as usize
    );
    assert_eq!(pair.payload.context.original_rows, prefix.num_rows());
    assert_eq!(
        pair.payload.context.original_columns,
        parts.problem.num_cols()
    );
    assert_eq!(
        pair.payload.context.original_rows + report.cuts,
        parts.problem.num_rows()
    );
    for plane in pair.planes().unwrap() {
        assert!(plane.terms().windows(2).all(|w| w[0].atom < w[1].atom));
        let mut expected: Vec<_> = plane
            .terms()
            .iter()
            .map(|t| (t.original_column, t.coefficient_bits))
            .collect();
        expected.sort_unstable();
        let matches = parts.problem.rows()[prefix.num_rows()..]
            .iter()
            .filter(|row| {
                let mut actual: Vec<_> =
                    row.coeffs.iter().map(|&(c, a)| (c, a.to_bits())).collect();
                actual.sort_unstable();
                row.lb.to_bits() == plane.lower_bits
                    && row.ub == f64::INFINITY
                    && actual == expected
            })
            .count();
        assert_eq!(
            matches, 1,
            "token must name an actual appended row's exact stored bits"
        );
    }
    parts
}

#[test]
fn native_zero_unary_keeps_checked_component_and_cube_without_extra_capture() {
    let _serial = native_test_guard();
    let f = Fixture::zero_unary();
    let p = f.source();
    let plan = Plan {
        components: vec![
            Component {
                negative: vec![2, 3],
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0.; 3],
                alpha: vec![1.; 4],
            },
            Component {
                negative: vec![4],
                positive: vec![],
                positive_units: vec![],
                g: vec![0.; 3],
                alpha: vec![],
            },
        ],
        leftover: vec![1],
    };
    // The supplied Plan is checked against the original Source. LP counts,
    // activity, admission, replay and both published rows are all real.
    let cuts = paid_with_plan(&p, plan);
    let _parts = append(&cuts, &f);
    let owner = cuts.signed.as_ref().unwrap();
    let stats = owner.stats();
    assert_eq!(stats.source_replays, 1);
    assert_eq!(stats.component_rows.phase.cube.appended_rows, 1);
    assert_eq!(
        stats.component_rows.phase.appended_rows, 1,
        "no unary lift exists"
    );
    let pair = published(&cuts);
    let planes = pair.planes().unwrap();
    assert!(planes[0]
        .terms()
        .iter()
        .all(|t| !matches!(t.atom, Atom::Selector(_))));
    assert_eq!(
        planes[1]
            .terms()
            .iter()
            .filter(|t| matches!(t.atom, Atom::Selector(_)))
            .count(),
        2
    );
    assert!(pair.payload.unary_attempted && pair.payload.cube_attempted);
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
    let before = owner.stats();
    let bound = bind(owner, &cuts, pair.end(), &mut || true).unwrap();
    assert!(Arc::ptr_eq(&bound.published.payload, &pair.payload));
    assert!(bind(owner, &cuts, pair.end(), &mut || true).is_none());
    assert_eq!(owner.stats().work_units, before.work_units);
    assert_eq!(owner.stats().admitted_bytes, before.admitted_bytes);
    assert_eq!(cuts.proposal_lp_calls, 1);
}

#[test]
fn native_real_paid_cube_pair_keeps_stored_bits_and_binds_once() {
    let _serial = native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let cuts = paid(&p);
    let _parts = append(&cuts, &f);
    let owner = cuts.signed.as_ref().unwrap();
    let pair = published(&cuts);
    let debt = owner.stats();
    let token = bind(owner, &cuts, pair.end(), &mut || true).unwrap();
    assert!(Arc::ptr_eq(&token.published.payload, &pair.payload));
    assert!(std::ptr::eq(token.source, std::ptr::from_ref(&p)));
    let mut polls = 0;
    assert!(bind(owner, &cuts, pair.end(), &mut || {
        polls += 1;
        true
    })
    .is_none());
    assert_eq!(polls, 0);
    assert_eq!(owner.stats().work_units, debt.work_units);
    assert_eq!(owner.stats().admitted_bytes, debt.admitted_bytes);
    assert_eq!(cuts.proposal_lp_calls, 1);
}

#[test]
fn native_real_pair_refuses_foreign_source_and_owner_ledger() {
    let _serial = native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let cuts = paid(&p);
    let _parts = append(&cuts, &f);
    let owner = cuts.signed.as_ref().unwrap();
    let pair = published(&cuts);
    let other_source = p.clone();
    let m = HullMultipliers::zeros(&other_source);
    let lower = certify_hull_lower_bound(&other_source, &m).unwrap();
    let foreign = prepare(&other_source, m, lower, end(), &mut || true).unwrap();
    let mut polls = 0;
    assert!(bind(owner, &foreign, pair.end(), &mut || {
        polls += 1;
        true
    })
    .is_none());
    assert_eq!(polls, 0);
    assert!(bind(owner, &cuts, pair.end(), &mut || true).is_none());
    // A second genuine paid owner shares source bytes but has a distinct ledger.
    let other = paid(&p);
    let other_owner = other.signed.as_ref().unwrap();
    assert!(!Arc::ptr_eq(&owner.budget.state, &other_owner.budget.state));
    assert!(!pair.payload.context.owner(other_owner));
    assert!(pair.payload.context.owner(owner));
}

#[test]
fn native_foreign_parts_cannot_arm_a_real_staged_pair() {
    let _serial = native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let cuts = paid(&p);
    let owner = cuts.signed.as_ref().unwrap();
    let debt = owner.stats();
    let mut parts = Box::new(f.original());
    let foreign = Box::new(f.original());
    let mut tried = false;
    let report = cuts
        .append_with_poll(parts.as_mut(), &f.encoding(), end(), &mut || {
            let ready = {
                let counts = owner.budget.state.lock().unwrap();
                counts
                    .native_rows
                    .pending
                    .as_ref()
                    .is_some_and(|p| p.unary.is_some() && p.cube.is_some())
            };
            if ready && !tried {
                tried = true;
                arm(owner, foreign.as_ref(), 1);
            }
            true
        })
        .expect("native identity refusal retains valid old row families");
    assert!(tried);
    assert!(
        report
            .signed
            .as_ref()
            .unwrap()
            .component_rows
            .phase
            .cube
            .appended_rows
            > 0
    );
    let counts = owner.budget.state.lock().unwrap();
    assert!(counts.native_rows.pending.is_none() && counts.native_rows.published.is_none());
    drop(counts);
    assert!(bind(owner, &cuts, owner.budget.authority, &mut || true).is_none());
    assert!(owner.stats().work_units >= debt.work_units);
    assert_eq!(owner.stats().admitted_bytes, debt.admitted_bytes);
}

#[test]
fn native_real_staging_cancel_resets_pending_without_refund_or_retry() {
    let _serial = native_test_guard();
    for point in 0..3 {
        let f = Fixture::new();
        let p = f.source();
        let cuts = paid(&p);
        let owner = cuts.signed.as_ref().unwrap();
        let debt = owner.stats();
        let mut parts = Box::new(f.original());
        let prefix = parts.problem.clone();
        let mut stopped = false;
        let result = cuts.append_with_poll(parts.as_mut(), &f.encoding(), end(), &mut || {
            if stopped {
                return false;
            }
            let stop = {
                let counts = owner.budget.state.lock().unwrap();
                counts
                    .native_rows
                    .pending
                    .as_ref()
                    .is_some_and(|p| match point {
                        0 => true,
                        1 => p.unary.is_some(),
                        _ => p.unary.is_some() && p.cube.is_some(),
                    })
            };
            stopped |= stop;
            !stop
        });
        assert!(stopped && result.is_none());
        assert_eq!(parts.problem, prefix);
        {
            let counts = owner.budget.state.lock().unwrap();
            assert!(
                counts.native_rows.begun
                    && counts.native_rows.pending.is_none()
                    && counts.native_rows.published.is_none()
            );
        }
        assert!(owner.stats().work_units >= debt.work_units);
        assert_eq!(owner.stats().admitted_bytes, debt.admitted_bytes);
        let _old_only = cuts.append_to(parts.as_mut(), &f.encoding(), end());
        let counts = owner.budget.state.lock().unwrap();
        assert!(counts.native_rows.pending.is_none() && counts.native_rows.published.is_none());
        drop(counts);
        assert!(owner.stats().work_units >= debt.work_units);
    }
}

#[test]
fn native_real_pair_expired_bind_consumes_its_once_latch() {
    let _serial = native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let cuts = paid(&p);
    let _parts = append(&cuts, &f);
    let owner = cuts.signed.as_ref().unwrap();
    let pair = published(&cuts);
    let debt = owner.stats();
    assert!(bind(owner, &cuts, Instant::now(), &mut || true).is_none());
    assert!(owner.budget.state.lock().unwrap().native_rows.bound);
    let mut polls = 0;
    assert!(bind(owner, &cuts, pair.end(), &mut || {
        polls += 1;
        true
    })
    .is_none());
    assert_eq!(polls, 0);
    assert_eq!(owner.stats().work_units, debt.work_units);
}

#[test]
fn native_committed_missing_dual_hook_is_once_and_keeps_old_price_independent() {
    let _serial = native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let cuts = paid(&p);
    let _parts = append(&cuts, &f);
    let owner = cuts.signed.as_ref().unwrap();
    let pair = published(&cuts);
    let token = bind(owner, &cuts, pair.end(), &mut || true).unwrap();
    let bounds = preactivation_bounds(&p);
    let roles = classify(&p, &bounds);
    let make = || build_hull_lp(&p, &roles, &bounds).unwrap();
    let mut lp = make();
    assert_eq!((lp.model.num_cols(), lp.model.num_rows()), (11, 6));
    assert_eq!(
        (0..lp.model.num_rows())
            .map(|i| lp.model.row(lp.model.row_at(i).unwrap()).0.len())
            .sum::<usize>(),
        16
    );
    let wrong_source = p.clone();
    let mut wrong_polls = 0;
    let mut wrong_calls = 0;
    let declined = token.with_node(
        &wrong_source,
        &[],
        &mut lp,
        &[],
        pair.end(),
        &mut || {
            wrong_polls += 1;
            true
        },
        |_, _| None,
        |_, _| {
            wrong_calls += 1;
        },
    );
    assert!(matches!(declined, bank::Visit::Declined));
    assert_eq!((wrong_polls, wrong_calls), (0, 0));
    assert_eq!((lp.model.num_cols(), lp.model.num_rows()), (11, 6));
    let mut calls = 0;
    let mut certificates = 0;
    let result = token.with_node(
        &p,
        &[],
        &mut lp,
        &[],
        pair.end(),
        &mut || true,
        |_, _| {
            certificates += 1;
            None
        },
        |actual, _certify| {
            calls += 1;
            assert_eq!(actual.model.num_rows(), 8);
            None::<f64>
        },
    );
    assert!(matches!(result, bank::Visit::Committed(Some(None))));
    assert_eq!((calls, certificates), (1, 0));
    // A fresh existing-call hook receives both sign candidates. Old proof uses
    // the full augmented row vector; absence from either decoder cannot suppress it.
    let mut lp = make();
    let mut old_calls = 0;
    let mut runs = 0;
    let result = token.with_node(
        &p,
        &[],
        &mut lp,
        &[],
        pair.end(),
        &mut || true,
        |actual, y| {
            old_calls += 1;
            let m = multipliers_from_duals(&p, &roles, actual, y)?;
            bab::certify_hull_node_lower_bound(&p, &[], &m, &[])
        },
        |actual, certify| {
            runs += 1;
            let y = vec![0.; actual.model.num_rows()];
            let m = multipliers_from_duals(&p, &roles, actual, &y).unwrap();
            let old = bab::certify_hull_node_lower_bound(&p, &[], &m, &[]).unwrap();
            assert!(certify(&y).unwrap() >= old);
            let neg: Vec<_> = y.iter().map(|x| -x).collect();
            assert!(certify(&neg).unwrap() >= old);
        },
    );
    assert!(matches!(result, bank::Visit::Committed(Some(()))));
    assert_eq!((runs, old_calls), (1, 2));
    assert_eq!(cuts.proposal_lp_calls, 1); // hook stand-in executes no extra LP
}

#[test]
fn native_retained_paid_caps_refuse_each_altered_limit_without_spending() {
    let _serial = native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let cuts = paid(&p);
    let _parts = append(&cuts, &f);
    let owner = cuts.signed.as_ref().unwrap();
    let pair = published(&cuts);
    let before = owner.stats();
    let native_before = cuts.native_rows_report();
    assert!(pair.payload.context.owner(owner));
    let bounds = preactivation_bounds(&p);
    let roles = classify(&p, &bounds);
    for field in 0..8 {
        let mut context = pair.payload.context.clone();
        let limit = match field {
            0 => &mut context.caps.components,
            1 => &mut context.caps.positive,
            2 => &mut context.caps.negative,
            3 => &mut context.caps.semantic_row,
            4 => &mut context.caps.semantic_pair,
            5 => &mut context.caps.expanded_pair,
            6 => &mut context.caps.raw_expansion_pair,
            7 => &mut context.caps.batch_rows,
            _ => unreachable!(),
        };
        *limit = limit.checked_add(1).unwrap();
        assert!(!context.owner(owner));
        // No planes are needed: the paid-cap gate must precede staging,
        // optional credit consumption, polls, or the existing LP callback.
        let forged = BoundPublished {
            source: &p,
            published: Published {
                payload: Arc::new(Payload {
                    context,
                    unary: None,
                    cube: None,
                    unary_attempted: false,
                    cube_attempted: false,
                    extra: None,
                    extra_attempted: false,
                    armed_rows: None,
                }),
            },
        };
        let mut lp = build_hull_lp(&p, &roles, &bounds).unwrap();
        let rows = lp.model.num_rows();
        let mut polls = 0;
        let mut calls = 0;
        let result = forged.with_node(
            &p,
            &[],
            &mut lp,
            &[],
            pair.end(),
            &mut || {
                polls += 1;
                true
            },
            |_, _| None,
            |_, _| {
                calls += 1;
            },
        );
        assert!(matches!(result, bank::Visit::Declined));
        assert_eq!((polls, calls), (0, 0));
        assert_eq!(lp.model.num_rows(), rows);
        assert_eq!(cuts.native_rows_report(), native_before);
        assert_eq!(owner.stats().work_units, before.work_units);
        assert_eq!(owner.stats().admitted_bytes, before.admitted_bytes);
    }
    assert!(pair.payload.context.owner(owner));
    assert!(bind(owner, &cuts, pair.end(), &mut || true).is_some());
    assert_eq!(cuts.native_rows_report(), native_before);
}
