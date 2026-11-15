// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Charged real-worker, tail-authority, identity and refusal controls; unrun.
use super::super::{checked, resources, start, witness_matches};
use super::test_fixture::*;
use super::*;

fn seal(p: &HullMarginProblem, plan: Plan, m: HullMultipliers) -> HullRootMipCuts<'_> {
    let lower = certify_hull_lower_bound(p, &m).unwrap();
    let mut c = super::super::super::super::prepare(p, m, lower, end(), &mut || true).unwrap();
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(p).unwrap();
    b.reserve(cost).unwrap();
    b.proof(cost, false).unwrap();
    let checked =
        super::super::super::certificate::check(&c, &plan, &b, b.search_end, &mut || true).unwrap();
    let signature = Arc::new(signature(&c).collect());
    let row = row(&c, &plan, checked, cost, &b, &signature);
    component_rows::admit(&c, &plan, &b).unwrap();
    c.signed = Some(row);
    c
}
fn setup(c: &HullRootMipCuts<'_>) {
    let row = c.signed.as_ref().unwrap();
    row.replay_inner(c, end(), &mut || true, false).unwrap();
    assert!(start(row));
    let (work, bytes) = resources::setup(row, c).unwrap();
    resources::pay(row, work, bytes, end()).unwrap();
    assert!(witness_matches(row));
}
fn paid_witness(c: &HullRootMipCuts<'_>) -> Witness {
    setup(c);
    let row = c.signed.as_ref().unwrap();
    let choices = resources::choices(row, c, end(), &mut || true).unwrap();
    let choice = &choices[resources::first_partial(row, &choices).unwrap()];
    resources::pay(row, choice.work, choice.bytes, end()).unwrap();
    {
        let mut counts = row.budget.state.lock().unwrap();
        assert!(counts.checks < MAX_CHECKS);
        counts.checks += 1;
    }
    let mut w = geometry(row, c, choice).unwrap();
    prepare_tail(row, c, &mut w, end(), &mut || true).unwrap();
    w
}
#[test]
fn signed_partial_paid_d30_worker_preserves_both_publications_and_original_caps() {
    let f = Fixture::new(0.25);
    let base = f.source();
    let mut p = base.clone();
    p.input_dim = 30;
    p.lower = vec![-1.; 30];
    p.upper = vec![1.; 30];
    p.w1 = vec![0.; 30 * H];
    for j in 0..H {
        for i in 0..D {
            p.w1[j * 30 + i] = base.w_row(j)[i];
        }
    }
    let mut plan = plan(&p, 1);
    for c in &mut plan.components {
        c.g = vec![0.; 30];
    }
    let mut m = HullMultipliers::zeros(&p);
    for j in 2..H {
        if j != 9 {
            for i in 0..30 {
                m.g[j * 30 + i] = 0.5 * p.w_row(j)[i];
            }
        }
    }
    let c = seal(&p, plan, m);
    let row = c.signed.as_ref().unwrap();
    let free = row.budget.remaining().unwrap();
    assert!(free > 1_154_224);
    row.budget.charge(free - 1_154_224).unwrap();
    setup(&c);
    let token = checked::produce(row, &c, end(), &mut || true).unwrap();
    assert!(token.owned(row));
    assert_eq!(token.planes.len(), 1);
    assert!(token.planes[0].lower > -14. * EPS - 1e-10);
    let s = row.stats();
    let r = &s.component_rows.joint;
    assert_eq!(r.partial_completed_pairs, 1);
    assert_eq!(r.partial_checked_masks, 8);
    assert_eq!(r.partial_tail_rechecks, 26);
    assert!(r.partial_work_units > 0);
    let counts = row.budget.state.lock().unwrap();
    assert!(counts.reserved > 0);
    drop(counts);
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_some());
    assert_eq!(row.stats().source_replays, 2);
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_none());
    assert!(row.stats().work_units <= MAX_WORK);
}
#[test]
fn signed_partial_no_credit_or_payload_keeps_original_rows_and_publication() {
    for payload in [false, true] {
        let f = Fixture::new(0.25);
        let p = f.source();
        let mut c = cuts(&p, 1);
        if payload {
            let row = c.signed.as_mut().unwrap();
            let held = row
                .budget
                .state
                .lock()
                .unwrap()
                .component_rows
                .admitted_bytes;
            row.budget.admitted_bytes = MAX_BYTES - held;
        } else {
            let row = c.signed.as_ref().unwrap();
            row.budget.charge(row.budget.remaining().unwrap()).unwrap();
        }
        let mut parts = f.parts();
        let original = parts.problem.clone();
        let enc = f.encoding(&parts);
        let report = c.append_to(&mut parts, &enc, end()).unwrap();
        let s = report.signed.unwrap();
        assert_eq!(s.component_rows.appended_rows, 2);
        assert_eq!(s.component_rows.joint.partial_paid_pairs, 0);
        assert_eq!(s.component_rows.joint.partial_appended_rows, 0);
        assert_eq!(
            &parts.problem.rows()[..original.num_rows()],
            original.rows()
        );
        let row = c.signed.as_ref().unwrap();
        assert!(row.replay_inner(&c, end(), &mut || true, false).is_some());
        assert_eq!(row.stats().source_replays, 2);
        assert!(row.stats().work_units <= MAX_WORK);
    }
}
#[test]
fn signed_partial_final_selected_leaf_refusal_is_once_only_and_retains_old_rows() {
    let f = Fixture::new(0.25);
    let p = f.source();
    let c = cuts(&p, 1);
    let row = c.signed.as_ref().unwrap();
    let parts = f.parts();
    let enc = f.encoding(&parts);
    let mut refused = false;
    let rows = binding::prepare_rows(&c, &parts, &enc, end(), &mut || {
        if !refused && row.stats().component_rows.joint.partial_checked_masks == 7 {
            refused = true;
            false
        } else {
            true
        }
    })
    .unwrap();
    assert!(refused);
    assert_ne!((rows).len(), 0);
    let before = row.stats();
    let r = &before.component_rows.joint;
    assert_eq!(before.source_replays, 1);
    assert_eq!(before.component_rows.prepared_rows, 2);
    assert!(before.component_rows.phase.prepared_rows > 0);
    assert_eq!(r.partial_checked_masks, 7);
    assert_eq!(r.partial_prepared_rows, 0);
    assert_eq!(r.partial_paid_pairs, 1);
    binding::with_phase_binding_test(&c, &parts, &enc, |binding| {
        assert_eq!(
            (prepare(row, &c, &parts, binding, end(), &mut || true)).len(),
            0
        );
    })
    .unwrap();
    assert_eq!(
        row.stats().component_rows.joint,
        before.component_rows.joint
    );
    assert_eq!(row.stats().work_units, before.work_units);
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_some());
    assert_eq!(row.stats().source_replays, 2);
}
#[test]
fn signed_partial_full_checker_rejects_dropped_lower_tail_and_changed_support() {
    for mode in 0..6 {
        let f = Fixture::graph_control(14. * EPS - 0.25);
        let p = f.source();
        let c = cuts(&p, 1);
        let row = c.signed.as_ref().unwrap();
        let mut w = paid_witness(&c);
        // At y=1 these omitted terms are NEGATIVE. Treating them as zero or
        // as lower ReLU planes could make an unsafe graph appear positive.
        match mode {
            0 => w.tail.clear(),
            1 => {
                w.tail_constant = I::zero();
                w.tail_coeff.fill(I::zero());
            }
            2 => w.tail[0].upper = 0.,
            3 => w.tail[0].g[0] = 0.25,
            4 => w.body.negative[1] = w.body.negative[0],
            _ => {
                w.negative.pop();
            }
        }
        assert!(certificate::check(row, &c, &w, end(), &mut || true).is_none());
        assert_eq!(row.stats().component_rows.joint.partial_checked_masks, 0);
        assert!(row.replay_inner(&c, end(), &mut || true, false).is_some());
    }
}
#[test]
fn signed_partial_checked_integrity_binds_mapping_tail_alpha_direction_and_owners() {
    for mode in 0..10 {
        let f = Fixture::new(0.25);
        let p = f.source();
        let c = cuts(&p, 1);
        let other = cuts(&p, 1);
        let row = c.signed.as_ref().unwrap();
        setup(&c);
        let mut token = checked::produce(row, &c, end(), &mut || true).unwrap();
        assert!(token.owned(row));
        assert!(!token.owned(other.signed.as_ref().unwrap()));
        if mode == 0 {
            token.witness = Arc::new((*row.witness).clone());
        } else if mode == 1 {
            token.signature = Arc::new((*row.signature).clone());
        } else if mode == 2 {
            token.ledger = Arc::new(std::sync::Mutex::new(Counts::default()));
        } else {
            let geometry = &mut token.planes[0].geometry;
            let details = geometry.partial.as_mut().expect("partial");
            match mode {
                3 => geometry.body.negative.swap(0, 1),
                4 => details.tail[0].g[0] += 0.25,
                5 => details.tail[0].upper += 0.25,
                6 => details.tail_constant.lo += 0.25,
                7 => details.tail_coeff[0].hi += 0.25,
                8 => geometry.body.alpha[0] += 0.25,
                _ => geometry.body.g[0] += 0.25,
            }
        }
        assert!(!token.owned(row));
        let parts = f.parts();
        let enc = f.encoding(&parts);
        binding::with_phase_binding_test(&c, &parts, &enc, |binding| {
            assert!(
                checked::export(row, &c, &parts, binding, &token, end(), &mut || true).is_none()
            );
        })
        .unwrap();
    }
}
#[test]
fn signed_partial_expiry_foreign_model_and_attempts_cannot_renew_authority() {
    let f = Fixture::new(0.25);
    let p = f.source();
    let c = cuts(&p, 1);
    let row = c.signed.as_ref().unwrap();
    row.replay_inner(&c, end(), &mut || true, false).unwrap();
    let parts = f.parts();
    let foreign = f.parts();
    let enc = f.encoding(&parts);
    let before = row.stats();
    binding::with_phase_binding_test(&c, &parts, &enc, |binding| {
        assert_eq!(
            (prepare(row, &c, &foreign, binding, end(), &mut || true)).len(),
            0
        );
    })
    .unwrap();
    assert_eq!(row.stats().work_units, before.work_units);
    assert!(!row.stats().component_rows.joint.attempted);
    binding::with_phase_binding_test(&c, &parts, &enc, |binding| {
        assert_eq!(
            (prepare(row, &c, &parts, binding, Instant::now(), &mut || true)).len(),
            0
        );
        assert_eq!(
            (prepare(row, &c, &parts, binding, end(), &mut || true)).len(),
            0
        );
    })
    .unwrap();
    assert_eq!(row.stats().work_units, before.work_units);
    assert_eq!(row.stats().component_rows.joint.producer_attempts, 1);
    assert_eq!(row.stats().component_rows.joint.checker_slots, 0);
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_some());
    assert_eq!(row.stats().source_replays, 2);
}
#[test]
fn signed_partial_source_tail_support_subnormal_cancellation_and_overflow_are_safe() {
    let tiny = f64::from_bits(1);
    for (w, b, g, l, u) in [
        (tiny, 0., 0., 0., 1.),
        (1., -0.25, 0.1, -2., 3.),
        (1., 0., 1., 0., 0.),
    ] {
        let upper = majorant_offset_upper(&[w], b, &[g], &[l], &[u]).unwrap();
        for x in [l, 0.5 * l + 0.5 * u, u] {
            let z = q(w) * q(x) + q(b);
            let relu = z.max(q(0.));
            assert!(relu <= q(g) * q(x) + q(upper));
            assert!(-relu >= -q(g) * q(x) - q(upper));
        }
    }
    assert!(majorant_offset_upper(&[f64::MAX], f64::MAX, &[0.], &[1.], &[1.]).is_none());
    assert!(majorant_offset_upper(&[1.], 0., &[f64::INFINITY], &[-1.], &[1.]).is_none());
    assert!(majorant_offset_upper(&[f64::MAX], 0., &[f64::MAX], &[-1.], &[1.]).is_none());
}
#[test]
fn signed_partial_large_priority_capacity_and_legacy_small_completion_share_one_slot() {
    let f = Fixture::new(0.25);
    let mut p = f.source();
    p.hidden_dim = 20;
    for _ in 0..3 {
        let mut w = vec![0.; D];
        w[1] = 1.;
        p.w1.extend(w);
        p.b1.push(0.);
    }
    p.c.extend([2. * EPS, -EPS, -EPS]);
    let mut plan = plan(&p, 1);
    plan.components.push(Component {
        negative: vec![18],
        positive: vec![17],
        positive_units: vec![2],
        g: vec![0.; D],
        alpha: vec![0., 0.5],
    });
    plan.components.push(Component {
        negative: vec![19],
        positive: vec![],
        positive_units: vec![],
        g: vec![0.; D],
        alpha: vec![],
    });
    let m = HullMultipliers::zeros(&p);
    let c = seal(&p, plan, m);
    let row = c.signed.as_ref().unwrap();
    setup(&c);
    let token = checked::produce(row, &c, end(), &mut || true).unwrap();
    assert_eq!(token.planes.len(), 2);
    assert!(token.planes[0].geometry.partial.is_some());
    assert!(token.planes[1].geometry.partial.is_none());
    let r = row.stats().component_rows.joint;
    assert_eq!(r.partial_paid_pairs, 1);
    assert_eq!(r.paid_pairs, 2);
    assert_eq!(r.checker_slots, 1);
    assert_eq!(r.partial_checked_masks, 8);
    assert!(r.checked_masks > 8);
    let mut a = row.plan.components[0].clone();
    let mut b = row.plan.components[1].clone();
    a.positive = vec![0; 17];
    b.positive = vec![0; 16];
    assert!(quote(&p, &a, &b).is_none());
    a.positive = vec![0];
    b.positive = vec![];
    b.negative.push(18);
    assert!(quote(&p, &a, &b).is_none());
    assert!(row.replay_inner(&c, end(), &mut || true, false).is_some());
    assert_eq!(row.stats().source_replays, 2);
}
