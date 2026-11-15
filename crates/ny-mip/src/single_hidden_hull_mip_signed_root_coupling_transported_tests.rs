// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Supplied-owner engineering controls, not a claim of new LP discovery.
use super::super::super::default_fixture::end;
use super::*;

fn source(bias: f64) -> HullMarginProblem {
    let mut weights = vec![1.; 17];
    weights[9..].fill(-1.);
    let mut output = vec![0.; 34];
    output[0] = 0.5_f64.next_up().next_up();
    for c in &mut output[1..17] {
        *c = (-0.125_f64).next_up().next_up();
    }
    let p = HullMarginProblem::new(
        weights,
        vec![0.; 17],
        1,
        &output,
        &[bias.next_up().next_up(), 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        vec![-1.],
        vec![1.],
    )
    .unwrap();
    assert_eq!(p.d, bias);
    assert_eq!(p.c[0], 0.5);
    assert!(p.c[1..].iter().all(|c| *c == -0.125));
    p
}

fn plan() -> Plan {
    Plan {
        components: vec![
            Component {
                negative: (1..9).collect(),
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0.],
                alpha: vec![0.5; 256],
            },
            Component {
                negative: (9..17).collect(),
                positive: vec![],
                positive_units: vec![],
                g: vec![0.],
                alpha: vec![],
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
    multipliers.alpha[0] = 0.5;
    let lower = certify_hull_lower_bound(p, &multipliers).unwrap();
    let cuts = prepare(p, multipliers, lower, end(), &mut || true).unwrap();
    let mut plan = plan();
    for c in &mut plan.components {
        c.g = vec![0.; p.input_dim];
    }
    plan.valid(&cuts).unwrap();
    let b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
    (cuts, plan, b, checked, cost)
}

fn half_choice() -> Choice {
    Choice {
        a: 0,
        b: 1,
        sa: 0,
        sb: None,
        fast: [true, true],
        transported: true,
        check: 0,
        search: 0,
        total: 0,
        score: 0.,
    }
}

#[test]
fn transported_half_seed_embeds_every_original_mask_and_checks_source_bits() {
    let p = source(1.25);
    let (cuts, plan, _, _, _) = prepared(&p);
    let c = half_choice();
    assert!(eligible(&cuts, &plan, c));
    let targets = directions(&cuts, &plan, c).unwrap();
    assert_eq!(targets, [-0.125, 0.125]);
    assert_eq!(
        targets[0] + targets[1],
        plan.components[0].g[0] + plan.components[1].g[0]
    );
    for mask in 0..256usize {
        let r = mask.count_ones() as f64 / 8.;
        assert_eq!(0.5 * 0.5 - r, 0.25 * 0.5 - r - targets[0]);
        assert_eq!(r, 0.25 * 0.5 + r - targets[1]);
    }
    let mut foreign = plan;
    foreign.components[0].alpha.fill(0.25);
    assert!(!eligible(&cuts, &foreign, c));
    let mut nonuniform = foreign;
    nonuniform.components[0].alpha[1] = 0.5;
    let mut c = c;
    c.fast[0] = false;
    assert!(!eligible(&cuts, &nonuniform, c));
}

#[test]
fn transported_uniform_half_crosses_safe_and_preserves_unsafe_and_closed_zero() {
    for bias in [1.25, 0.75, 1.] {
        let p = source(bias);
        let (cuts, mut plan, mut b, mut checked, mut cost) = prepared(&p);
        let before = checked.root_lower;
        assert!((before - (bias - 1.75)).abs() < 1e-9);
        let debt = b.state.lock().unwrap().work;
        let checks = b.state.lock().unwrap().checks;
        assert!(super::super::super::run(
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
        assert_eq!(r.producer_attempts, 1);
        assert_eq!(r.checker_slots, 1);
        assert_eq!((r.old_masks, r.new_masks), (512, 512));
        assert_eq!(r.line_mask_attempts, 1024);
        assert_eq!(state.work, debt + LIMIT);
        assert_eq!(state.checks, checks + 1);
        assert!(r.used_work_units <= LIMIT);
        assert!(b.root_coupling.as_ref().unwrap().transported_payload);
        drop(state);
        assert!(checked.root_lower > bias - 1. - 1e-9 && checked.root_lower <= bias - 1.);
        assert!(checked.root_lower > before + 0.749999);
        assert_eq!(checked.root_lower > 0., bias > 1.);
        for x in [-1_f64, 0., 1.] {
            let original = bias + 0.5 * x.max(0.) - x.max(0.) - (-x).max(0.);
            assert!(checked.root_lower <= original);
        }
        assert_eq!(plan.components[0].negative, (1..9).collect::<Vec<_>>());
        assert_eq!(plan.components[1].negative, (9..17).collect::<Vec<_>>());
        assert_eq!(plan.components[0].positive_units, [1]);
        assert_eq!(plan.components[1].positive_units, [1]);
        assert!(b.payload_bytes().unwrap() <= MAX_BYTES);
    }
}

#[test]
fn transported_strict_global_floor_keeps_original_checked_owner() {
    let p = source(1.25);
    let (cuts, mut plan, mut b, mut checked, mut cost) = prepared(&p);
    let bits = witness(&plan);
    let original = checked.root_lower.to_bits();
    let original_cost = cost;
    assert!(super::super::super::run(
        &cuts,
        &mut plan,
        &mut checked,
        &mut cost,
        0.25,
        &mut b,
        &mut || true,
    ));
    let state = b.state.lock().unwrap();
    let r = &state.root_coupling;
    assert_eq!(r.completed_supports, 1);
    assert!(!r.selected);
    assert!(r.proposed_lower.unwrap() <= 0.25);
    assert_eq!(witness(&plan), bits);
    assert_eq!(checked.root_lower.to_bits(), original);
    assert_eq!(cost, original_cost);
}

#[test]
fn transported_cancel_after_canonical_search_keeps_original_debt_and_owner() {
    let p = source(1.25);
    let (cuts, mut plan, mut b, mut checked, mut cost) = prepared(&p);
    let bits = witness(&plan);
    let lower = checked.root_lower.to_bits();
    let old_cost = cost;
    let debt = b.state.lock().unwrap().work;
    let state = b.state.clone();
    let mut cancelled = false;
    assert!(super::super::super::run(
        &cuts,
        &mut plan,
        &mut checked,
        &mut cost,
        f64::NEG_INFINITY,
        &mut b,
        &mut || {
            if state.lock().unwrap().root_coupling.line_mask_attempts > 512 {
                cancelled = true;
                false
            } else {
                true
            }
        },
    ));
    assert!(cancelled);
    assert_eq!(witness(&plan), bits);
    assert_eq!(checked.root_lower.to_bits(), lower);
    assert_eq!(cost, old_cost);
    let state = b.state.lock().unwrap();
    assert_eq!(state.work, debt + LIMIT);
    assert_eq!(state.root_coupling.completed_supports, 0);
    assert_eq!(state.root_coupling.checker_slots, 1);
}

#[test]
fn transported_faces_find_exact_interval_and_refuse_nonfinite_arithmetic() {
    // The uniform fixture's post-alpha residual faces.
    let faces = [-0.125, -0.625, -0.875, 0.];
    let (s, value) = best_exchange(-1., 1., faces).unwrap();
    assert_eq!(s, -0.25);
    assert_eq!(value, -1.);
    assert!(best_exchange(1., 1., faces).is_none());
    assert!(best_exchange(-1., 1., [f64::MAX, -f64::MAX, 0., 0.]).is_none());
    assert!(best_exchange(-f64::MAX, f64::MAX, faces).is_none());
}

#[test]
fn transported_fixed_axes_and_optional_work_are_bounded() {
    let mut p = source(1.25);
    // Axis metadata is independent of head/mask data.
    p.input_dim = 6;
    p.lower = vec![-1.; 6];
    p.upper = vec![1.; 6];
    let (selected, count) = axes(&p, &[1., 4., 2., 4., 0., 3., 0., 0., 0., 0., 0., 0.]).unwrap();
    assert_eq!(count, 4);
    assert_eq!(selected.map(|a| a.index), [1, 3, 5, 2]);
    let p = source(1.25);
    let (_, plan, b, _, _) = prepared(&p);
    let (meta, baseline, _, words) = quote::admission(&plan, &b).unwrap();
    let (m2, extended, w2, enabled) = quote::admission_with_transport(&plan, &b).unwrap();
    assert_eq!((meta, words), (m2, w2));
    assert!(enabled && extended >= baseline);
    let extra = quote::transported_extra(&plan, 1, 0, 1).unwrap();
    assert_eq!(
        extra,
        64 * 512 * 2 + 20 * 512 + 128 * 2 + 16 * 10 * 2 + 4096
    );
    assert!(extra < LIMIT);

    // A real wider Source makes search storage, not nine final Plans, dominate.
    let mut wide = source(1.25);
    wide.input_dim = 30;
    wide.w1 = vec![0.; 17 * 30];
    for j in 0..17 {
        wide.w1[j * 30] = if j < 9 { 1. } else { -1. };
    }
    wide.lower = vec![-1.; 30];
    wide.upper = vec![1.; 30];
    let (_, wide_plan, mut wide_budget, _, _) = prepared(&wide);
    let (_, baseline, _, _) = quote::admission(&wide_plan, &wide_budget).unwrap();
    let (_, extended, _, enabled) =
        quote::admission_with_transport(&wide_plan, &wide_budget).unwrap();
    assert!(enabled && extended > baseline);
    let state = wide_budget.state.lock().unwrap();
    let other = state.shared_bytes
        + state.direction.cache_bytes
        + state.child.cache_bytes
        + state.component_rows.admitted_bytes;
    drop(state);
    wide_budget.admitted_bytes = MAX_BYTES - other - baseline;
    let (_, actual, _, enabled) =
        quote::admission_with_transport(&wide_plan, &wide_budget).unwrap();
    assert_eq!((actual, enabled), (baseline, false));
    wide_budget.admitted_bytes = MAX_BYTES - other - extended;
    let (_, actual, _, enabled) =
        quote::admission_with_transport(&wide_plan, &wide_budget).unwrap();
    assert_eq!((actual, enabled), (extended, true));

    // Purchase adjacent remaining-work states through the real private ledger.
    for missing in [0usize, 1] {
        let (cuts, plan, mut b, _, cost) = prepared(&p);
        let (ticket, _) = admit(&cuts, &plan, cost, &mut b).unwrap();
        ticket.begin(&cuts, &b).unwrap();
        let (_, _, base) = quote::choice(
            &plan,
            1,
            0,
            1,
            false,
            [true, true],
            cost,
            resources::words(&plan).unwrap(),
        )
        .unwrap();
        let extra = quote::transported_extra(&plan, 1, 0, 1).unwrap();
        let leave = base + extra - missing;
        let consume = ticket.remaining(&cuts, &b).unwrap() - leave;
        ticket.pay(&cuts, &b, consume).unwrap();
        let choice = choose(&cuts, &plan, &b, &ticket, cost, &mut || true).unwrap();
        assert_eq!((choice.a, choice.b, choice.sa, choice.sb), (0, 1, 0, None));
        assert_eq!(choice.transported, missing == 0);
        assert_eq!(choice.total, if missing == 0 { base + extra } else { base });
        if missing == 0 {
            ticket.pay(&cuts, &b, choice.total).unwrap();
            ticket.state.lock().unwrap().build = true;
            let mut cache = build_cache(&cuts, &plan, choice, &b, &ticket, &mut || true).unwrap();
            assert!(propose(&cuts, &plan, choice, &mut cache, &b, &ticket, &mut || true).is_some());
            assert!(ticket.state.lock().unwrap().transported_started);
            ticket.alive(&cuts, &b, &mut || true).unwrap();
            assert!(propose(&cuts, &plan, choice, &mut cache, &b, &ticket, &mut || true).is_none());
        }
    }
}

#[test]
fn transported_saturated_nonzero_exchange_crosses_checked_safe_unsafe_and_tie() {
    for bias in [0.75_f64, 0.25, 0.5] {
        let mut weights = vec![1.; 48];
        weights[16..32].fill(-1.);
        weights[40..48].fill(-1.);
        let mut biases = vec![-2.; 48];
        biases[0] = 0.;
        biases[16] = 0.;
        biases[32..].fill(0.);
        let mut output = vec![0.; 96];
        for c in &mut output[..32] {
            *c = (1. / 256_f64).next_up().next_up();
        }
        output[0] = 0.5_f64.next_up().next_up();
        output[16] = 0.5_f64.next_up().next_up();
        for c in &mut output[32..48] {
            *c = (-0.125_f64).next_up().next_up();
        }
        let p = HullMarginProblem::new(
            weights,
            biases,
            1,
            &output,
            &[bias.next_up().next_up(), 0.],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.],
            vec![1.],
        )
        .unwrap();
        let mut multipliers = HullMultipliers::zeros(&p);
        multipliers.alpha[0] = 0.5;
        multipliers.alpha[16] = 0.5;
        let lower = certify_hull_lower_bound(&p, &multipliers).unwrap();
        let cuts = prepare(&p, multipliers, lower, end(), &mut || true).unwrap();
        let mut plan = Plan {
            components: (0..2)
                .map(|part| {
                    let mut alpha = vec![0.; 256 * 16];
                    for mask in 0..256 {
                        alpha[mask * 16] = 0.5;
                    }
                    Component {
                        negative: (32 + 8 * part..40 + 8 * part).collect(),
                        positive: (16 * part..16 + 16 * part).collect(),
                        positive_units: vec![2; 16],
                        alpha,
                        g: vec![0.],
                    }
                })
                .collect(),
            leftover: vec![],
        };
        plan.valid(&cuts).unwrap();
        let mut b = Budget::new(&cuts, end(), Duration::ZERO).unwrap();
        let mut cost = plan.cost(&p).unwrap();
        b.reserve(cost).unwrap();
        b.proof(cost, false).unwrap();
        let mut checked = certificate::check(&cuts, &plan, &b, b.search_end, &mut || true).unwrap();
        let before = checked.root_lower;
        assert!((before - (bias - 1.5)).abs() < 1e-9);
        let mut choice = half_choice();
        choice.sb = Some(0);
        assert!(eligible(&cuts, &plan, choice));
        assert_eq!(directions(&cuts, &plan, choice).unwrap(), [-0.5, 0.5]);
        assert!(super::super::super::run(
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
        assert_eq!(r.line_mask_attempts, 1024);
        assert_eq!((r.old_masks, r.new_masks), (512, 512));
        assert!(r.used_work_units <= LIMIT);
        drop(state);
        assert!(checked.root_lower > bias - 0.5 - 1e-9 && checked.root_lower <= bias - 0.5);
        assert!(checked.root_lower > before + 0.999999);
        assert_eq!(checked.root_lower > 0., bias > 0.5);
        assert_eq!(plan.components[0].positive[0], 16);
        assert_eq!(plan.components[1].positive[0], 0);
        for c in &plan.components {
            assert_eq!(c.positive.len(), 16);
            assert_eq!(c.positive_units, [2; 16]);
        }
        for x in [-1_f64, 0., 1.] {
            assert!(checked.root_lower <= bias - 0.5 * x.abs());
        }
    }
}
