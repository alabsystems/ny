// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::tests::{cuts, cycle, end};
use super::*;
fn row_fixture() -> (HullMarginProblem, Instant) {
    (cycle(9, 285. / 32.), end())
}
#[test]
fn global_sdp_malformed_shapes_dimensions_and_nonfinite_source_decline_without_indexing() {
    let p = cycle(9, 285. / 32.);
    let mut c = cuts(&p);
    c.multipliers.alpha.pop();
    assert!(propose(&c, end(), Duration::ZERO).is_none());
    let mut c = cuts(&p);
    c.multipliers.g[0] = f64::NAN;
    assert!(propose(&c, end(), Duration::ZERO).is_none());
    let valid = cycle(9, 285. / 32.);
    let mut broken = valid.clone();
    broken.input_dim = usize::MAX;
    let mut c = cuts(&valid);
    c.problem = &broken;
    assert!(certificate::system(&c, &[], end(), &mut || true).is_none());
    assert!(propose(&c, end(), Duration::ZERO).is_none());
}
#[test]
fn global_sdp_bad_factor_constant_symmetry_and_last_poll_refuse() {
    let p = cycle(9, 285. / 32.);
    let c = cuts(&p);
    let mut s = certificate::system(&c, &[0.; 9], end(), &mut || true).unwrap();
    let mut lambda = vec![1.; s.n];
    let mut l = vec![0.; s.n * s.n];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        lambda[0] = bad;
        assert!(certificate::certify(&s, &lambda, &l, end(), &mut || true).is_none());
        lambda[0] = 1.;
        l[0] = bad;
        assert!(certificate::certify(&s, &lambda, &l, end(), &mut || true).is_none());
        l[0] = 0.;
    }
    l[1] = 0.25;
    assert!(certificate::certify(&s, &lambda, &l, end(), &mut || true).is_none());
    l[1] = 0.;
    let old = s.q[1];
    s.q[1].hi = old.hi + 1.;
    assert!(certificate::certify(&s, &lambda, &l, end(), &mut || true).is_none());
    s.q[1] = old;
    s.constant.lo = 1.;
    s.constant.hi = 0.;
    assert!(certificate::certify(&s, &lambda, &l, end(), &mut || true).is_none());
    s.constant = certificate::I::zero();
    let mut polls = 0;
    assert!(certificate::certify(&s, &lambda, &l, end(), &mut || {
        polls += 1;
        polls <= s.n
    })
    .is_none());
}
#[test]
fn global_sdp_signature_rejects_foreign_and_same_address_changed_original_source() {
    let (mut p, t) = row_fixture();
    let row = {
        let c = cuts(&p);
        propose(&c, t, Duration::ZERO).unwrap()
    };
    let other = p.clone();
    let other_cuts = cuts(&other);
    assert!(row.replay(&other_cuts, end(), &mut || true).is_none());
    p.b1[0] = 0.25;
    let changed = cuts(&p);
    assert!(row.replay(&changed, end(), &mut || true).is_none());
    assert_eq!(row.stats().source_replays, 0);
}
#[test]
fn global_sdp_expired_original_clock_cannot_be_renewed_by_a_later_caller() {
    let (p, t) = row_fixture();
    let c = cuts(&p);
    let mut row = propose(&c, t, Duration::ZERO).unwrap();
    row.budget.authority = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(row.replay(&c, end(), &mut || true).is_none());
    assert!(propose(
        &c,
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("overflow when subtracting duration from instant"),
        Duration::ZERO
    )
    .is_none());
    assert_eq!(c.proposal_lp_calls, 0);
}
#[test]
fn global_sdp_two_reserved_full_replays_survive_exhausted_optional_work() {
    let (p, t) = row_fixture();
    let c = cuts(&p);
    let row = propose(&c, t, Duration::ZERO).unwrap();
    let left = {
        let s = row.budget.counts.lock().unwrap();
        MAX_WORK - s.work - s.reserved
    };
    row.budget.charge(left, false).unwrap();
    assert!(row.budget.charge(1, false).is_none());
    for _ in 0..2 {
        assert!(row.replay(&c, end(), &mut || true).unwrap().lower > 0.005);
    }
    assert!(row.replay(&c, end(), &mut || true).is_none());
    let s = row.stats();
    assert_eq!(s.source_replays, 2);
    assert!(s.work_units <= MAX_WORK);
    assert_eq!(row.budget.counts.lock().unwrap().reserved, 0);
}
#[test]
fn global_sdp_failed_later_certificate_does_not_mutate_the_completed_baseline_or_tuple() {
    let (p, t) = row_fixture();
    let c = cuts(&p);
    let mut row = propose(&c, t, Duration::ZERO).unwrap();
    let scalar = c.checked_lower.to_bits();
    let m = c.multipliers.clone();
    let old_lower = row.lower;
    row.l[0] = f64::NAN;
    assert!(row.replay(&c, end(), &mut || true).is_none());
    assert_eq!(row.lower.to_bits(), old_lower.to_bits());
    assert_eq!(c.checked_lower.to_bits(), scalar);
    assert_eq!(c.multipliers, m);
    assert_eq!(row.stats().source_replays, 0);
}
#[test]
fn global_sdp_dimension_and_nominal_payload_caps_refuse_before_work() {
    let p = cycle(49, 48.75);
    let c = cuts(&p);
    assert!(propose(&c, end(), Duration::ZERO).is_none());
    assert!(Budget::new(&c, end(), Duration::ZERO, usize::MAX).is_none());
    let p = cycle(9, 285. / 32.);
    let c = cuts(&p);
    let b = Budget::new(&c, end(), Duration::ZERO, 19).unwrap();
    assert!(b.admitted_bytes <= MAX_BYTES);
    assert!(b.charge(usize::MAX, false).is_none());
    assert_eq!(b.counts.lock().unwrap().work, 0);
}
#[test]
fn global_sdp_search_cannot_spend_reserved_publication_or_extend_search_clock() {
    let p = cycle(9, 285. / 32.);
    let c = cuts(&p);
    let mut b = Budget::new(&c, end(), Duration::ZERO, 19).unwrap();
    let left = {
        let s = b.counts.lock().unwrap();
        MAX_WORK - s.reserved
    };
    b.charge(left, false).unwrap();
    assert!(b.factor(19).is_none());
    assert_eq!(b.counts.lock().unwrap().factors, 0);
    b.search_end = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(b.iteration().is_none());
    assert!(b.authority > Instant::now());
}
