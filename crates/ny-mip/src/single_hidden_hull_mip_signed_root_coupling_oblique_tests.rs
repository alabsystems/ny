// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Distinct-affine supplied-owner controls; no automatic LP-discovery claim.
use super::super::super::super::default_fixture::end;
use super::*;

const EPS: f64 = 1. / 256.;

fn source(bias: f64) -> HullMarginProblem {
    let mut weights = vec![0.; 18 * 3];
    weights[..6].copy_from_slice(&[1., 1., 0., 1., 1., EPS]);
    weights[2 * 3] = 1.;
    weights[3 * 3 + 1] = 1.;
    let mut biases = vec![0.; 18];
    for j in 4..18 {
        biases[j] = -(j as f64 + 1.) / 16.;
    }
    let mut output = vec![0.; 36];
    output[0] = 2_f64.next_up().next_up();
    output[1] = 1_f64.next_up().next_up();
    output[2] = (-1_f64).next_up().next_up();
    output[3] = (-1_f64).next_up().next_up();
    for c in &mut output[4..18] {
        *c = (-1_f64 / 64.).next_up().next_up();
    }
    let p = HullMarginProblem::new(
        weights,
        biases,
        3,
        &output,
        &[bias.next_up().next_up(), 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.; 3],
        vec![1.; 3],
    )
    .unwrap();
    assert_eq!(p.d, bias);
    assert_eq!(&p.c[..4], &[2., 1., -1., -1.]);
    p
}

fn plan() -> Plan {
    Plan {
        components: vec![
            Component {
                negative: (2..10).collect(),
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0.; 3],
                alpha: vec![0.; 256],
            },
            Component {
                negative: (10..18).collect(),
                positive: vec![1],
                positive_units: vec![2],
                g: vec![0.; 3],
                alpha: vec![1.; 256],
            },
        ],
        leftover: vec![],
    }
}

fn prepared(
    p: &HullMarginProblem,
) -> (
    HullRootMipCuts<'_>,
    Plan,
    Budget,
    certificate::Checked,
    usize,
) {
    let mut multipliers = HullMultipliers::zeros(p);
    multipliers.alpha[1] = 1.;
    let lower = certify_hull_lower_bound(p, &multipliers).unwrap();
    let cuts = prepare(p, multipliers, lower, end(), &mut || true).unwrap();
    let plan = plan();
    plan.valid(&cuts).unwrap();
    let b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    (cuts, plan, b, checked, cost)
}

fn paid(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &mut Budget,
    cost: usize,
    gap: usize,
) -> (Arc<Ticket>, Choice) {
    let (ticket, _) = admit(cuts, plan, cost, b).unwrap();
    ticket.begin(cuts, b).unwrap();
    let (_, _, base) = quote::choice(
        plan,
        3,
        0,
        1,
        false,
        [true, true],
        cost,
        resources::words(plan).unwrap(),
    )
    .unwrap();
    let face = quote::transported_extra(plan, 3, 0, 1).unwrap();
    let oblique = quote::oblique_extra(plan, 3, 0, 1).unwrap();
    let leave = base + face + oblique - gap;
    let consume = ticket.remaining(cuts, b).unwrap() - leave;
    ticket.pay(cuts, b, consume).unwrap();
    let choice = choose(cuts, plan, b, &ticket, cost, &mut || true).unwrap();
    ticket.pay(cuts, b, choice.total).unwrap();
    ticket.state.lock().unwrap().build = true;
    (ticket, choice)
}

#[test]
fn oblique_distinct_affine_coordinate_stall_and_checked_safe_unsafe_zero() {
    // Run the unchanged canonical and face producers with exactly their fee.
    let p = source(2.5);
    let (cuts, old, mut b, _, cost) = prepared(&p);
    let extra = quote::oblique_extra(&old, 3, 0, 1).unwrap();
    let (ticket, choice) = paid(&cuts, &old, &mut b, cost, extra);
    assert!(ticket.state.lock().unwrap().oblique_choice.is_none());
    let mut cache = build_cache(&cuts, &old, choice, &b, &ticket, &mut || true).unwrap();
    endpoints(&cuts, &old, choice, &mut cache, &b, &ticket, &mut || true).unwrap();
    let (t, canonical) =
        choose_line(&cuts, &old, choice, &cache, &b, &ticket, &mut || true).unwrap();
    assert_eq!((t, canonical), (0.5, -3. - EPS));
    let face = propose(&cuts, &old, choice, &mut cache, &b, &ticket, &mut || true).unwrap();
    assert_eq!(face.score, -3. - EPS);
    assert!(face.axis.is_none());
    for faces in [
        [-1., -1., -2. - EPS, -EPS],
        [-1., -1., -2. - EPS, -EPS],
        [-1., -1., -2. - EPS, -2. + EPS],
    ] {
        assert_eq!(best_exchange(-1., 1., faces), Some((0., -3. - EPS)));
    }

    for bias in [2.5, 0.75, 1.] {
        let p = source(bias);
        let (cuts, mut plan, mut b, mut checked, mut cost) = prepared(&p);
        let debt = b.state.lock().unwrap().work;
        assert!((checked.root_lower - (bias - 4. - EPS)).abs() < 1e-9);
        assert!(super::super::super::super::run(
            &cuts,
            &mut plan,
            &mut checked,
            &mut cost,
            f64::NEG_INFINITY,
            &mut b,
            &mut || true,
        ));
        let state = b.state.lock().unwrap();
        let r = &state.root_coupling;
        assert!(r.selected && r.completed_supports == 1, "{r:?}");
        assert_eq!((r.old_masks, r.new_masks), (512, 512));
        assert_eq!(r.line_mask_attempts, 1024);
        assert_eq!(r.checker_slots, 1);
        assert_eq!(state.work, debt + LIMIT);
        assert!(r.used_work_units <= LIMIT);
        drop(state);
        assert!(
            b.root_coupling
                .as_ref()
                .unwrap()
                .state
                .lock()
                .unwrap()
                .oblique_started
        );
        let target = bias - 2. - EPS;
        assert!(checked.root_lower > target - 1e-9 && checked.root_lower <= target);
        assert_eq!(checked.root_lower > 0., bias == 2.5);
        assert_eq!(plan.components[0].g, [-0.5, -0.5, 0.]);
        assert_eq!(plan.components[1].g, [0.5, 0.5, 0.]);
        assert_eq!(plan.components[0].negative, (2..10).collect::<Vec<_>>());
        assert_eq!(plan.components[1].negative, (10..18).collect::<Vec<_>>());
        for x in [-1_f64, 0., 1.] {
            for y in [-1_f64, 0., 1.] {
                for z in [-1_f64, 0., 1.] {
                    let original = bias + 2. * (x + y).max(0.) + (x + y + EPS * z).max(0.)
                        - x.max(0.)
                        - y.max(0.);
                    assert!(checked.root_lower <= original);
                }
            }
        }
        // (1,-1,-1) attains the exact graph minimum d-1.
        assert_eq!(bias - 1. > 0., bias == 2.5);
        assert!(b.payload_bytes().unwrap() <= MAX_BYTES);
    }
}

#[test]
fn oblique_partial_bundle_and_active_only_step_are_bounded() {
    let gradients = [-2., 0., 0., 0., -2., 0.];
    assert_eq!(
        min_norm(&gradients, 2, 3, &mut || true).unwrap(),
        [-1., -1., 0.]
    );
    assert!(min_norm(&[-1., 1.], 2, 1, &mut || true).is_none());
    assert!(min_norm(&[f64::MAX], 1, 1, &mut || true).is_none());
    assert!(min_norm(&[], 0, 1, &mut || true).is_none());
    assert!(min_norm(&[1.; 5], 5, 1, &mut || true).is_none());
    assert!(min_norm(&gradients, 2, 3, &mut || false).is_none());
    let mut cache = Cache::new(3, 2, 1).unwrap();
    cache
        .delta
        .copy_from_slice(&[0., -1., 0., 0., 0., 0., -1., 0., 0., 1., 1., EPS]);
    let mut active = Active::new();
    active.record(0, 0, -1.);
    active.record(0, 1, -1.);
    active.record(1, 2, -2. - EPS);
    assert_eq!(step(&cache, &active, &[-1., -1., 0.], 3), Some(0.5));
    assert!(step(&cache, &active, &[0.; 3], 3).is_none());
    // Only the first two exact minima survive; a lower row resets the bundle.
    active.record(0, 7, -1.);
    assert_eq!(active.rows[0], [0, 1]);
    active.record(0, 8, -2.);
    assert_eq!((active.rows[0][0], active.count[0]), (8, 1));
}

#[test]
fn oblique_optional_adjacent_work_and_single_paid_capability() {
    for gap in [0usize, 1] {
        let p = source(2.5);
        let (cuts, plan, mut b, _, cost) = prepared(&p);
        let (_, payload, _, _) = quote::admission_with_transport(&plan, &b).unwrap();
        let (_, _, base) = quote::choice(
            &plan,
            3,
            0,
            1,
            false,
            [true, true],
            cost,
            resources::words(&plan).unwrap(),
        )
        .unwrap();
        let face = quote::transported_extra(&plan, 3, 0, 1).unwrap();
        let extra = quote::oblique_extra(&plan, 3, 0, 1).unwrap();
        let (ticket, choice) = paid(&cuts, &plan, &mut b, cost, gap);
        assert_eq!((choice.a, choice.b, choice.sa, choice.sb), (0, 1, 0, None));
        assert!(choice.transported);
        assert_eq!(choice.total, base + face + if gap == 0 { extra } else { 0 });
        assert_eq!(ticket.payload, payload); // no new admission or byte debt
        {
            let mut state = ticket.state.lock().unwrap();
            assert_eq!(state.oblique_choice.is_some(), gap == 0);
            state.transported_started = true;
        }
        let used = ticket.state.lock().unwrap().used;
        assert_eq!(
            claim(&cuts, choice, &b, &ticket, &mut || true).is_some(),
            gap == 0
        );
        assert!(claim(&cuts, choice, &b, &ticket, &mut || true).is_none());
        assert_eq!(ticket.state.lock().unwrap().used, used);
    }
}

#[test]
fn oblique_cancel_keeps_original_owner_and_paid_debt() {
    let p = source(2.5);
    let (cuts, plan, mut b, checked, cost) = prepared(&p);
    let bits = witness(&plan);
    let lower = checked.root_lower.to_bits();
    let (ticket, choice) = paid(&cuts, &plan, &mut b, cost, 0);
    let debt = b.state.lock().unwrap().work;
    let used = ticket.state.lock().unwrap().used;
    let mut cache = build_cache(&cuts, &plan, choice, &b, &ticket, &mut || true).unwrap();
    let mut cancelled = false;
    let proposal = propose(&cuts, &plan, choice, &mut cache, &b, &ticket, &mut || {
        let keep = !ticket.state.lock().unwrap().oblique_started;
        cancelled |= !keep;
        keep
    });
    assert!(cancelled && proposal.is_none());
    assert_eq!(witness(&plan), bits);
    assert_eq!(checked.root_lower.to_bits(), lower);
    assert_eq!(b.state.lock().unwrap().work, debt);
    assert_eq!(ticket.state.lock().unwrap().used, used);
    assert!(!ticket.state.lock().unwrap().checker);
    assert_eq!(b.state.lock().unwrap().root_coupling.completed_supports, 0);
}

#[test]
fn oblique_strict_global_floor_keeps_original_checked_owner() {
    let p = source(2.5);
    let (cuts, mut plan, mut b, mut checked, mut cost) = prepared(&p);
    let bits = witness(&plan);
    let lower = checked.root_lower.to_bits();
    let old_cost = cost;
    assert!(super::super::super::super::run(
        &cuts,
        &mut plan,
        &mut checked,
        &mut cost,
        0.5 - EPS,
        &mut b,
        &mut || true,
    ));
    let state = b.state.lock().unwrap();
    assert_eq!(state.root_coupling.completed_supports, 1);
    assert!(!state.root_coupling.selected);
    assert!(state.root_coupling.proposed_lower.unwrap() <= 0.5 - EPS);
    assert_eq!(witness(&plan), bits);
    assert_eq!(checked.root_lower.to_bits(), lower);
    assert_eq!(cost, old_cost);
}

#[test]
fn oblique_foreign_unpaid_and_zero_bundle_fail_closed() {
    let p = source(2.5);
    let (cuts, plan, mut b, _, cost) = prepared(&p);
    let (foreign, _, _, _, _) = prepared(&p);
    let (ticket, choice) = paid(&cuts, &plan, &mut b, cost, 0);
    {
        let mut state = ticket.state.lock().unwrap();
        state.transported_started = true;
        state.paid_choice = false;
    }
    assert!(claim(&cuts, choice, &b, &ticket, &mut || true).is_none());
    ticket.state.lock().unwrap().paid_choice = true;
    assert!(claim(&foreign, choice, &b, &ticket, &mut || true).is_none());
    assert!(claim(&cuts, choice, &b, &ticket, &mut || false).is_none());
    assert!(!ticket.state.lock().unwrap().oblique_started);
    let cache = Cache::new(3, 256, 256).unwrap();
    let mut active = Active::new();
    active.record(0, 0, 0.);
    active.record(1, 256, 0.);
    let mut face = Proposal {
        endpoint: vec![0.; 512],
        targets: vec![0.; 6],
        score: 0.,
        axis: None,
        stored_changes: [0.; 2],
    };
    let endpoint = face.endpoint.as_ptr();
    let targets = face.targets.as_ptr();
    assert!(improve(
        &cuts,
        &plan,
        choice,
        &cache,
        &active,
        &mut face,
        &b,
        &ticket,
        &mut || true
    )
    .is_none());
    assert_eq!(face.score, 0.);
    assert_eq!(face.endpoint.as_ptr(), endpoint);
    assert_eq!(face.targets.as_ptr(), targets);
    assert!(ticket.state.lock().unwrap().oblique_started);
    assert!(!ticket.state.lock().unwrap().checker);
}

#[test]
fn oblique_literal_tariff_and_existing_scratch_envelope() {
    let mut plan = plan();
    // Metadata-only tariff shape; the extra fourteen entries are never run.
    for _ in 2..16 {
        plan.components.push(Component {
            negative: vec![],
            positive: vec![],
            positive_units: vec![],
            g: vec![],
            alpha: vec![],
        });
    }
    plan.leftover = vec![0; 11];
    let extra = quote::oblique_extra(&plan, 30, 0, 1).unwrap();
    assert_eq!(extra, 307_664);
    let total = 7_611_112 + extra;
    assert_eq!(total, 7_918_776);
    assert!(total < LIMIT);
    assert!(quote::oblique_extra(&plan, usize::MAX, 0, 1).is_none());
    assert!(quote::oblique_extra(&plan, 30, 0, 99).is_none());
    // Both explicit scratch phases are beneath the old six-width quote.
    for d in [1usize, 3, 30, 1024] {
        assert!(5 * d <= 6 * (d + 1));
        assert!(d + 2 * d + 2 * (d + 1) <= 6 * (d + 1));
    }
    // Actual control-layout bound, including the enlarged once-capability,
    // Arc counters, twelve Vec headers, active rows and retained face axes.
    assert!(
        size_of::<Ticket>()
            + 2 * size_of::<usize>()
            + 12 * size_of::<Vec<f64>>()
            + size_of::<Active>()
            + 4 * size_of::<Axis>()
            <= 1536
    );
}
