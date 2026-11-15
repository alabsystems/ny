// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Irreversible child credit, with one real automatic native-funded route.
use super::super::tests::{cached, checked_row, one, plan, problem};
use super::*;
use crate::single_hidden_hull::mip_cuts::signed::tests::{cuts, end};

fn conditional_source() -> HullMarginProblem {
    // The existing automatically discovered conditional separation fixture.
    // Use the ordinary constructor, without supplied geometry or LP receipts.
    let (d, h) = (30, 18);
    let mut first = vec![0.; d * h];
    let mut biases = vec![0.; h];
    for j in 0..h {
        first[j * d + d - 1] = 1.;
    }
    first[2 * d - 1] = -1.;
    biases[1] = -1. / 64.;
    for i in 0..5 {
        let j = 3 + i;
        let m = 2. + i as f64 / 64.;
        first[j * d + d - 1] = m;
        biases[j] = -m - 0.25;
    }
    let mut output = vec![0.; 2 * h];
    output[0] = 28.;
    output[1] = 1.;
    output[2] = -97. / 8.;
    output[3..h].fill(-1.);
    HullMarginProblem::new(
        first,
        biases,
        d,
        &output,
        &[5. - 1. / 512., 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.; d],
        vec![1.; d],
    )
    .unwrap()
}

#[test]
fn child_prepay_default_native_route_replays_and_binds_after_spare_work_is_exhausted() {
    let p = conditional_source();
    let attempt =
        propose_root_mip_cuts_with_conditional_mass_portfolio_attempt(&p, end(), Duration::ZERO);
    assert!(attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked);
    let c = attempt.cuts.unwrap();
    let row = c.signed.as_ref().unwrap();
    let b = &row.budget;
    let q = quote(&c, &row.plan).unwrap();
    assert_eq!(c.proposal_lp_calls, 1);
    assert_eq!(c.native_funding_report().funded_calls, 1);
    assert!(row.lower <= 0.);
    let base_bytes = b.admitted_bytes;
    let (paid_bytes, paid_work) = {
        let counts = b.state.lock().unwrap();
        assert_eq!(counts.replay_attempts, 0);
        assert_eq!(counts.child.collection_attempts, 0);
        assert_eq!(counts.prepaid_collection_work, q.collection_work);
        assert_eq!(counts.prepaid_bind_work, q.bind_work);
        // This default route also prepays the optional corner extension and
        // its ordinary entry credits. Retain the whole paid aggregate while
        // keeping the original collection/bind holds exact above.
        assert!(counts.child.corner_axes > 0);
        assert!(counts.child.cache_bytes > q.bytes);
        assert!(counts.child.work_units > q.collection_work + q.bind_work);
        assert!(counts.reserved > 0 && counts.component_reserved > 0);
        (counts.child.cache_bytes, counts.child.work_units)
    };
    // Spend only ordinary remaining credit; both Source passes and component
    // publication retain their independent existing holds.
    b.charge(b.remaining().unwrap()).unwrap();
    assert_eq!(b.remaining(), Some(0));
    let first = row.replay(&c, end(), &mut || true).unwrap();
    assert!(first.cache.is_none());
    assert_eq!(row.stats().source_replays, 1);
    let second = row.replay_with_cache(&c, end(), &mut || true).unwrap();
    let cache = second
        .cache
        .expect("paid second Source pass must retain its cache");
    assert_eq!(row.stats().source_replays, 2);
    assert_eq!(cache.stats().cached_masks, q.masks);
    assert_eq!(cache.stats().cache_bytes, paid_bytes);
    assert_eq!(b.remaining(), Some(0));
    let before = {
        let counts = b.state.lock().unwrap();
        assert_eq!(counts.prepaid_collection_work, 0);
        assert_eq!(counts.prepaid_bind_work, q.bind_work);
        assert_eq!(counts.child.work_units, paid_work);
        (
            counts.work,
            counts.reserved,
            counts.component_reserved,
            counts.child.work_units,
        )
    };
    let bound = cache.bind(&c, end(), &mut || true).unwrap();
    let expected_root = cache
        .stats()
        .corner_root_lower
        .map_or(row.lower, |corner| row.lower.max(corner));
    assert_eq!(
        bound.root_lower().unwrap().to_bits(),
        expected_root.to_bits()
    );
    {
        let counts = b.state.lock().unwrap();
        assert_eq!(counts.prepaid_bind_work, 0);
        assert_eq!(counts.child.bind_attempts, 1);
        assert_eq!(
            (
                counts.work,
                counts.reserved,
                counts.component_reserved,
                counts.child.work_units,
            ),
            before
        );
    }
    assert_eq!(b.admitted_bytes, base_bytes);
    assert_eq!(b.remaining(), Some(0));
    assert!(cache.bind(&c, end(), &mut || true).is_none());
    assert!(row.replay_with_cache(&c, end(), &mut || true).is_none());
    assert_eq!(row.stats().source_replays, 2);
    assert_eq!(c.proposal_lp_calls, 1);
}

fn native_source() -> HullMarginProblem {
    let e = 1. / 1024.;
    HullMarginProblem::new(
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
        vec![0., 0., 0., 0., -1.],
        3,
        &[1., 1., -1., -1., -e, 0., 0., 0., 0., 0.],
        &[11. / 16., 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.; 3],
        vec![1.; 3],
    )
    .unwrap()
}

fn funded(p: &HullMarginProblem) -> (Box<HullRootMipCuts<'_>>, Budget, Plan) {
    let until = end();
    let mut attempt = primal_capture::Attempt::new_with_native_counts(p, until);
    // A stable address also permits a real root-coupling ownership check after
    // this helper returns. No native bank, reservation or count is fabricated.
    let c = Box::new(propose_root_mip_cuts_inner(p, until, Some(&mut attempt)).unwrap());
    let mut b = Budget::new_with_portfolio_conditions(&c, until, Duration::ZERO).unwrap();
    b.phase_all_enabled = true;
    b.phase_cube_enabled = true;
    let plan = Plan {
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
    };
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    root_coupling::admit(&c, &plan, cost, &mut b).unwrap();
    component_rows::admit(&c, &plan, &b).unwrap();
    native_publication::fund(&c, &b).unwrap();
    assert_eq!(c.proposal_lp_calls, 1);
    assert_eq!(native_publication::funding(&b).funded_calls, 1);
    (c, b, plan)
}

#[test]
fn child_prepay_exact_and_one_over_limits_preserve_native_and_coupling_authority() {
    let p = native_source();
    for (extra_work, extra_bytes) in [(0, 0), (1, 0), (0, 1)] {
        let (c, b, plan) = funded(&p);
        let q = quote(&c, &plan).unwrap();
        let prepaid = q.collection_work + q.bind_work;
        let base_bytes = b.admitted_bytes;
        let ticket = b.root_coupling.as_ref().unwrap();
        assert!(ticket.owned(&c, &b));
        for refusal in 0..4 {
            let mut foreign = b.clone();
            match refusal {
                0 => foreign.source ^= 1,
                1 => foreign.authority = c.deadline + Duration::from_secs(1),
                2 => {
                    foreign.search_end = Instant::now()
                        .checked_sub(Duration::from_secs(1))
                        .expect("test clock can represent a past deadline")
                }
                _ => foreign.portfolio_phase_enabled = false,
            }
            assert!(prepay(&c, &plan, &foreign).is_none());
            assert_eq!(b.state.lock().unwrap().child.stop_reason, "");
        }
        b.charge(b.remaining().unwrap() - prepaid + extra_work)
            .unwrap();
        let before = {
            let mut counts = b.state.lock().unwrap();
            // Boundary ledger only: keep the real funded capability and work,
            // and account for additional simultaneous shared payload exactly.
            counts.shared_bytes = MAX_BYTES + extra_bytes
                - b.admitted_bytes
                - counts.direction.cache_bytes
                - counts.component_rows.admitted_bytes
                - q.bytes;
            (counts.work, counts.reserved, counts.component_reserved)
        };
        let accepted = extra_work == 0 && extra_bytes == 0;
        assert_eq!(prepay(&c, &plan, &b).is_some(), accepted);
        {
            let counts = b.state.lock().unwrap();
            assert_eq!(counts.reserved, before.1);
            assert_eq!(counts.component_reserved, before.2);
            if accepted {
                assert_eq!(counts.work, before.0 + prepaid);
                assert_eq!(
                    counts.work + counts.reserved + counts.component_reserved,
                    MAX_WORK
                );
                assert_eq!(counts.child.work_units, prepaid);
                assert_eq!(counts.child.cache_bytes, q.bytes);
                assert_eq!(counts.prepaid_collection_work, q.collection_work);
                assert_eq!(counts.prepaid_bind_work, q.bind_work);
            } else {
                assert_eq!(counts.work, before.0);
                assert_eq!(counts.child.work_units, 0);
                assert_eq!(counts.child.cache_bytes, 0);
                assert_eq!(counts.prepaid_collection_work, 0);
                assert_eq!(counts.prepaid_bind_work, 0);
                assert_eq!(counts.child.stop_reason, "prepay-refused");
            }
        }
        if accepted {
            let direction = b.state.lock().unwrap().direction.cache_bytes;
            assert_eq!(b.payload_bytes().unwrap() + direction, MAX_BYTES);
        }
        assert!(prepay(&c, &plan, &b.clone()).is_none());
        assert_eq!(b.admitted_bytes, base_bytes);
        assert!(ticket.owned(&c, &b));
        assert_eq!(native_publication::funding(&b).funded_calls, 1);
    }
}

// Isolated ledger controls deliberately create no native receipt or publication
// authority. The production route above independently proves actual prepayment.
fn exhausted(c: &HullRootMipCuts<'_>, plan: &Plan) -> Budget {
    let b = Budget::new(c, end(), Duration::ZERO).unwrap();
    let q = quote(c, plan).unwrap();
    {
        let mut counts = b.state.lock().unwrap();
        counts.reserved = 31;
        counts.component_reserved = 47;
        counts.work = MAX_WORK - 31 - 47;
        counts.child.work_units = q.collection_work + q.bind_work;
        counts.child.cache_bytes = q.bytes;
        counts.prepaid_collection_work = q.collection_work;
        counts.prepaid_bind_work = q.bind_work;
    }
    b
}

#[test]
fn child_prepay_collection_exact_credit_and_bytes_are_once_only() {
    let p = one(0.25);
    let c = cuts(&p);
    let plan = plan(&p);
    let q = quote(&c, &plan).unwrap();
    for (short_work, short_bytes) in [(0, 0), (1, 0), (0, 1)] {
        let b = exhausted(&c, &plan);
        {
            let mut counts = b.state.lock().unwrap();
            counts.prepaid_collection_work -= short_work;
            counts.child.cache_bytes -= short_bytes;
        }
        let permit = admit(&c, &plan, &b, end());
        assert_eq!(permit.is_some(), short_work == 0 && short_bytes == 0);
        if let Some(permit) = permit {
            assert_eq!(permit.bytes, q.bytes);
            assert_eq!(permit.masks, q.masks);
            assert!(Arc::ptr_eq(&permit.state, &b.state));
        }
        let alias = b.clone();
        assert!(admit(&c, &plan, &alias, end()).is_none());
        drop(alias);
        let counts = b.state.lock().unwrap();
        assert_eq!(counts.child.collection_attempts, 1);
        assert_eq!(counts.prepaid_collection_work, 0);
        assert_eq!(counts.prepaid_bind_work, q.bind_work);
        assert_eq!(counts.child.cache_bytes, q.bytes - short_bytes);
        assert_eq!(
            (counts.work, counts.reserved, counts.component_reserved),
            (MAX_WORK - 78, 31, 47)
        );
    }
}

#[test]
fn child_prepay_collection_burns_before_foreign_deadline_and_failed_quote() {
    let p = one(0.25);
    let foreign = one(0.25);
    let c = cuts(&p);
    let other = cuts(&foreign);
    let plan = plan(&p);
    for failure in 0..3 {
        let b = exhausted(&c, &plan);
        let mut malformed = plan.clone();
        malformed.components[0].negative.clear();
        let result = match failure {
            0 => admit(&other, &plan, &b, end()),
            1 => admit(
                &c,
                &plan,
                &b,
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("test clock can represent a past deadline"),
            ),
            _ => admit(&c, &malformed, &b, end()),
        };
        assert!(result.is_none());
        assert!(admit(&c, &plan, &b, end()).is_none());
        let counts = b.state.lock().unwrap();
        assert_eq!(counts.child.collection_attempts, 1);
        assert_eq!(counts.prepaid_collection_work, 0);
        assert_eq!(
            counts.prepaid_bind_work,
            quote(&c, &plan).unwrap().bind_work
        );
        assert_eq!(
            (counts.work, counts.reserved, counts.component_reserved),
            (MAX_WORK - 78, 31, 47)
        );
    }
}

#[test]
fn child_prepay_smaller_collection_keeps_full_held_payload() {
    let p = problem(
        vec![1., -1.],
        vec![0.; 2],
        vec![-1.; 2],
        0.25,
        vec![-1.],
        vec![1.],
    );
    let c = cuts(&p);
    let big = plan(&p);
    let b = exhausted(&c, &big);
    let held = quote(&c, &big).unwrap().bytes;
    let mut small = big.clone();
    small.components[0].negative.pop();
    assert_eq!(
        big.components[0].negative.len(),
        small.components[0].negative.len() + 1
    );
    assert!(quote(&c, &small).unwrap().bytes < held);
    let permit = admit(&c, &small, &b, end()).unwrap();
    assert_eq!(permit.bytes, held);
    assert_eq!(b.state.lock().unwrap().child.cache_bytes, held);
}

#[test]
fn child_prepay_first_bind_credit_never_funds_a_child_call_or_retry() {
    let p = one(0.25);
    let c = cuts(&p);
    let plan = plan(&p);
    let cost = quote(&c, &plan).unwrap().bind_work;
    for failure in 0..3 {
        let b = exhausted(&c, &plan);
        // A child query cannot take credit reserved for authentication.
        assert!(b.child_charge(1, true, end()).is_none());
        assert_eq!(b.state.lock().unwrap().prepaid_bind_work, cost);
        let until = if failure == 2 {
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock can represent a past deadline")
        } else {
            end()
        };
        assert_eq!(
            b.child_charge(cost + usize::from(failure == 1), false, until)
                .is_some(),
            failure == 0
        );
        let alias = b.clone();
        assert!(alias.child_charge(cost, false, end()).is_none());
        drop(alias);
        let counts = b.state.lock().unwrap();
        assert_eq!(counts.prepaid_bind_work, 0);
        assert_eq!(counts.child.bind_attempts, 2);
        assert_eq!(counts.child.attempted_calls, 1);
        assert_eq!(
            (counts.work, counts.reserved, counts.component_reserved),
            (MAX_WORK - 78, 31, 47)
        );
    }
}

#[test]
fn child_prepay_real_cache_authentication_failure_burns_bind_without_refund() {
    let p = one(0.25);
    let foreign = one(0.25);
    let c = cuts(&p);
    let other = cuts(&foreign);
    let mut oversized = one(0.25);
    oversized.input_dim = usize::MAX;
    let mut overflow = cuts(&foreign);
    overflow.problem = &oversized;
    let plan = plan(&p);
    for failure in 0..4 {
        let row = checked_row(&c, &plan);
        let cache = cached(&c, &row);
        let b = &row.budget;
        let cost = quote(&c, &plan).unwrap().bind_work;
        assert_eq!(
            cost,
            64 * (p.input_dim * p.hidden_dim + cache.seal.len() + p.hidden_dim + 1)
        );
        // This is an isolated first-bind ledger control, with a real checked
        // cache and an explicit debit rather than a fabricated native bank.
        b.charge(cost).unwrap();
        {
            let mut counts = b.state.lock().unwrap();
            counts.prepaid_bind_work = cost;
            counts.child.work_units += cost;
        }
        b.charge(b.remaining().unwrap()).unwrap();
        let before = b.state.lock().unwrap().work;
        let result = match failure {
            0 => cache.bind(&other, end(), &mut || true),
            1 => cache.bind(&c, end(), &mut || false),
            2 => cache.bind(
                &c,
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("test clock can represent a past deadline"),
                &mut || true,
            ),
            _ => cache.bind(&overflow, end(), &mut || true),
        };
        assert!(result.is_none());
        assert!(cache.bind(&c, end(), &mut || true).is_none());
        let counts = b.state.lock().unwrap();
        assert_eq!(counts.prepaid_bind_work, 0);
        assert_eq!(counts.child.bind_attempts, 2);
        assert_eq!(counts.work, before);
        assert_eq!(counts.source_replays, 2);
    }
}

#[test]
fn child_prepay_unfunded_entry_keeps_legacy_collection() {
    let p = one(0.25);
    let c = cuts(&p);
    let plan = plan(&p);
    let mut b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    b.portfolio_phase_enabled = true;
    assert_eq!(native_publication::funding(&b).funded_calls, 0);
    assert!(prepay(&c, &plan, &b).is_none());
    assert_eq!(b.state.lock().unwrap().child.stop_reason, "");
    assert!(admit(&c, &plan, &b, end()).is_some());
    assert_eq!(
        b.state.lock().unwrap().child.work_units,
        quote(&c, &plan).unwrap().collection_work
    );
}

#[test]
fn child_prepay_quote_matches_the_1308_mask_partition_and_full_bind_seal() {
    let (d, h) = (30, 128);
    let mut objective = vec![1.; h];
    objective[..54].fill(-1.);
    let p = problem(
        vec![0.; d * h],
        vec![0.; h],
        objective,
        0.,
        vec![-1.; d],
        vec![1.; d],
    );
    let c = cuts(&p);
    let mut next = 0;
    let components = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 8, 8, 8, 8, 8]
        .into_iter()
        .map(|size| {
            let negative = (next..next + size).collect();
            next += size;
            Component {
                negative,
                positive: vec![],
                positive_units: vec![],
                g: vec![0.; d],
                alpha: vec![],
            }
        })
        .collect();
    assert_eq!(next, 54);
    let q = quote(
        &c,
        &Plan {
            components,
            leftover: (54..h).collect(),
        },
    )
    .unwrap();
    assert_eq!(q.masks, 1_308);
    assert_eq!(q.bytes, 44_512);
    assert_eq!(q.collection_work, 24_592);
    assert_eq!(q.bind_work, 371_072);
}
