// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Supplied nominal line controls; Source certification is tested separately.
use super::*;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn source(
    rows: Vec<Vec<f64>>,
    bias: Vec<f64>,
    coefficients: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
) -> HullMarginProblem {
    let d = lower.len();
    assert!(rows.iter().all(|row| row.len() == d));
    let mut output: Vec<_> = coefficients.iter().map(|v| v.next_up().next_up()).collect();
    output.extend(vec![0.; coefficients.len()]);
    let p = HullMarginProblem::new(
        rows.into_iter().flatten().collect(),
        bias,
        d,
        &output,
        &[0., 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        lower,
        upper,
    )
    .unwrap();
    assert_eq!(p.c, coefficients);
    p
}
fn part(p: &HullMarginProblem, positive: Vec<usize>, negative: Vec<usize>) -> Component {
    let count = positive.len();
    Component {
        alpha: vec![0.; (1usize << negative.len()) * count],
        negative,
        positive,
        positive_units: vec![2; count],
        g: vec![0.; p.input_dim],
    }
}
fn stall(d: usize) -> (HullMarginProblem, Component) {
    let mut rows = vec![vec![0.; d]; 3];
    rows[0][0] = 1.;
    rows[1][0] = -1.;
    rows[2][1] = 1.;
    let mut lower = vec![0.; d];
    let mut upper = vec![0.; d];
    lower[0] = -1.;
    upper[0] = 1.;
    lower[1] = -0.5;
    upper[1] = 0.5;
    let p = source(rows, vec![0.5, 0.5, 0.], vec![1., 1., -1.], lower, upper);
    let c = part(&p, vec![0, 1], vec![2]);
    (p, c)
}
// Literal old two-sweep mechanism on the supplied D2 fixture, including its
// unchanged-cache reference. This is not a default-discovery or LP claim.
fn old_scalar_sweeps(p: &HullMarginProblem, c: &Component, mask: usize) -> Vec<f64> {
    let mut alpha = vec![0.; c.positive.len()];
    let (mut q, mut constant) = search::mask_cache(p, c, &c.g, mask, &alpha).unwrap();
    for _ in 0..2 {
        for (slot, &head) in c.positive.iter().enumerate() {
            let old = alpha[slot];
            let weight = c.nominal_coefficient(p, slot).unwrap();
            let bias = weight * p.b1[head];
            let base = constant - old * bias;
            let direction: Vec<_> = p.w_row(head).iter().map(|v| weight * v).collect();
            let residual: Vec<_> = q.iter().zip(&direction).map(|(q, d)| q - old * d).collect();
            let chosen = search::legacy_choose(
                p, &residual, &direction, base, bias, old, &q, constant, &mut 0,
            )
            .unwrap();
            alpha[slot] = chosen;
            constant = base + chosen * bias;
            for i in 0..p.input_dim {
                q[i] = residual[i] + chosen * direction[i];
            }
        }
    }
    alpha
}

#[test]
fn root_line_funded_schedule_replaces_credit_and_preserves_legacy_fallbacks() {
    let s = Schedule::new(30, 6, 16, 3).unwrap();
    assert!(s.enabled);
    assert_eq!(
        (s.credit, s.line_work, s.visits, s.sweeps, s.actual_bound),
        (1_778_688, 1_005_856, 2, 1, 1_598_752)
    );
    let s = Schedule::new(30, 6, 16, 5).unwrap();
    assert_eq!(
        (s.credit, s.visits, s.actual_bound),
        (2_964_480, 5, 2_488_096)
    );
    let s = Schedule::new(2, 1, 2, 2).unwrap();
    assert_eq!(
        (s.credit, s.line_work, s.visits, s.actual_bound),
        (1152, 1040, 0, 1040)
    );
    let adjacent = Schedule::new(2, 1, 2, 1).unwrap();
    assert!(!adjacent.enabled);
    assert_eq!(
        (
            adjacent.credit,
            adjacent.visits,
            adjacent.sweeps,
            adjacent.actual_bound
        ),
        (576, 1, 2, 576)
    );
    for (d, k, p, active) in [(30, 6, 1, 1), (1, 1, 2, 2), (30, 2, 16, 0)] {
        let s = Schedule::new(d, k, p, active).unwrap();
        assert!(!s.enabled);
        assert_eq!((s.visits, s.sweeps, s.actual_bound), (active, 2, s.credit));
    }
    for d in [1, 2, 30, 128] {
        for k in [1, 6] {
            for p in [1, 2, 16] {
                for active in [0, p] {
                    let s = Schedule::new(d, k, p, active).unwrap();
                    assert!(s.actual_bound <= s.credit);
                    if s.enabled {
                        assert!(s.line_work <= s.actual_bound && s.visits <= active);
                        assert_eq!(s.sweeps, 1);
                    } else {
                        assert_eq!((s.visits, s.sweeps), (active, 2));
                    }
                }
            }
        }
    }
    for shape in [
        (0, 1, 2, 2),
        (129, 1, 2, 2),
        (2, 0, 2, 2),
        (2, 7, 2, 2),
        (2, 1, 0, 0),
        (2, 1, 17, 2),
        (2, 1, 2, 3),
        (usize::MAX, 1, 2, 2),
    ] {
        assert!(Schedule::new(shape.0, shape.1, shape.2, shape.3).is_none());
    }
}

#[test]
fn root_line_all_ones_escapes_two_mask_coordinate_stall_with_zero_direction() {
    let (p, c) = stall(2);
    // K0(a,b)=(a+b)/2-|a-b|; K1=K0-1/2. Every nonzero
    // single-coordinate step from (0,0) loses t/2; the joint endpoint gains 1.
    for t in [0.25_f64, 0.5, 1.] {
        assert_eq!(0.5 * t - t.abs(), -0.5 * t);
    }
    let e = end();
    let mut run = Run::new(Schedule::new(2, 1, 2, 2).unwrap(), &p, e);
    for mask in 0..2 {
        assert_eq!(old_scalar_sweeps(&p, &c, mask), vec![0., 0.]);
        let mut alpha = vec![0., 0.];
        let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, mask, &alpha).unwrap();
        assert_eq!(score(&p, &q, constant), Some(-0.5 * mask as f64));
        let outcome = run
            .apply(
                &p,
                &c,
                &c.g,
                mask,
                &mut alpha,
                &mut q,
                &mut constant,
                e,
                &mut || true,
            )
            .unwrap();
        assert!(outcome.completed && outcome.improved);
        assert_eq!(outcome.proposals, 3); // The genuinely cheaper legacy helper.
        assert_eq!(alpha, vec![1., 1.]);
        assert_eq!(q, search::mask_cache(&p, &c, &c.g, mask, &alpha).unwrap().0);
        assert_eq!(score(&p, &q, constant), Some(1. - 0.5 * mask as f64));
    }
}

#[test]
fn root_line_dispatches_the_cheaper_real_sorted_helper_on_padded_dimensions() {
    let (p, c) = stall(30);
    assert!(search::alpha::work(30).unwrap() < search::legacy_work(30).unwrap());
    let e = end();
    let mut run = Run::new(Schedule::new(30, 1, 2, 2).unwrap(), &p, e);
    for mask in 0..2 {
        let mut alpha = vec![0., 0.];
        let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, mask, &alpha).unwrap();
        let out = run
            .apply(
                &p,
                &c,
                &c.g,
                mask,
                &mut alpha,
                &mut q,
                &mut constant,
                e,
                &mut || true,
            )
            .unwrap();
        assert!(out.completed && out.improved);
        assert_eq!(out.proposals, 1); // Sorted helper deduplicates the endpoints.
        assert_eq!(alpha, vec![1., 1.]);
        assert_eq!(score(&p, &q, constant), Some(1. - 0.5 * mask as f64));
    }
}

#[test]
fn root_line_actual_rebuild_tie_preserves_alpha_and_signed_zero_bits() {
    let eps = 2_f64.powi(-54);
    let p = source(
        vec![vec![2., 0.], vec![0., 0.], vec![1., 0.], vec![eps, 0.]],
        vec![1., 0., 0., 0.],
        vec![1., 1., -1., -1.],
        vec![-1., 0.],
        vec![1., 0.],
    );
    let c = part(&p, vec![0, 1], vec![2, 3]);
    let e = end();
    let mut run = Run::new(Schedule::new(2, 2, 2, 2).unwrap(), &p, e);
    for mask in 0..3 {
        let mut alpha = vec![1., 1.];
        let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, mask, &alpha).unwrap();
        let out = run
            .apply(
                &p,
                &c,
                &c.g,
                mask,
                &mut alpha,
                &mut q,
                &mut constant,
                e,
                &mut || true,
            )
            .unwrap();
        assert!(out.completed && !out.improved);
    }
    let mut alpha = vec![0.5, -0.];
    let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, 3, &alpha).unwrap();
    assert_eq!((q[0], constant), (-eps, 0.5));
    let theta =
        search::legacy_choose(&p, &q, &[1., 0.], constant, 0.5, 0., &q, constant, &mut 0).unwrap();
    assert_eq!(theta, eps);
    assert_eq!(0.5 + theta * 0.5, 0.5); // The stored first alpha rounds back.
    assert!(score(&p, &[0., 0.], 0.5).unwrap() > score(&p, &q, constant).unwrap());
    let before = (bits(&alpha), bits(&q), constant.to_bits());
    let out = run
        .apply(
            &p,
            &c,
            &c.g,
            3,
            &mut alpha,
            &mut q,
            &mut constant,
            e,
            &mut || true,
        )
        .unwrap();
    assert!(out.completed && !out.improved);
    assert_eq!((bits(&alpha), bits(&q), constant.to_bits()), before);
}

#[test]
fn root_line_numeric_refusals_are_atomic_and_consume_only_their_mask() {
    let (p, c) = stall(2);
    let e = end();
    let mut run = Run::new(Schedule::new(2, 1, 2, 2).unwrap(), &p, e);
    for (mask, bad) in [f64::NAN, -0.25].into_iter().enumerate() {
        let mut alpha = vec![bad, 0.];
        let mut q = vec![0., 0.];
        let mut constant: f64 = 0.;
        let before = (bits(&alpha), bits(&q), constant.to_bits());
        let out = run
            .apply(
                &p,
                &c,
                &c.g,
                mask,
                &mut alpha,
                &mut q,
                &mut constant,
                e,
                &mut || true,
            )
            .unwrap();
        assert!(!out.completed && !out.improved);
        assert_eq!((bits(&alpha), bits(&q), constant.to_bits()), before);
        let mut polls = 0;
        assert!(run
            .apply(
                &p,
                &c,
                &c.g,
                mask,
                &mut alpha,
                &mut q,
                &mut constant,
                e,
                &mut || {
                    polls += 1;
                    true
                }
            )
            .is_none());
        assert_eq!(polls, 0);
    }
    let p = source(
        vec![vec![0., 0.]; 3],
        vec![f64::MAX, f64::MAX, 0.],
        vec![1., 1., -1.],
        vec![-1., -1.],
        vec![1., 1.],
    );
    let c = part(&p, vec![0, 1], vec![2]);
    let mut run = Run::new(Schedule::new(2, 1, 2, 2).unwrap(), &p, e);
    let mut alpha = vec![0.5, 0.5];
    let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, 0, &alpha).unwrap();
    assert_eq!(constant, f64::MAX);
    let before = (bits(&alpha), bits(&q), constant.to_bits());
    let out = run
        .apply(
            &p,
            &c,
            &c.g,
            0,
            &mut alpha,
            &mut q,
            &mut constant,
            e,
            &mut || true,
        )
        .unwrap();
    assert!(!out.completed && !out.improved);
    assert_eq!((bits(&alpha), bits(&q), constant.to_bits()), before);
}

#[test]
fn root_line_transient_and_final_cancellation_are_sticky_and_atomic() {
    let (p, c) = stall(2);
    let e = end();
    let schedule = Schedule::new(2, 1, 2, 2).unwrap();
    let mut reference = Run::new(schedule, &p, e);
    let mut alpha = vec![0., 0.];
    let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, 0, &alpha).unwrap();
    let mut complete_polls = 0;
    assert!(
        reference
            .apply(
                &p,
                &c,
                &c.g,
                0,
                &mut alpha,
                &mut q,
                &mut constant,
                e,
                &mut || {
                    complete_polls += 1;
                    true
                }
            )
            .unwrap()
            .improved
    );
    assert!(complete_polls >= 4);
    for fail_at in [1, complete_polls] {
        let mut run = Run::new(schedule, &p, e);
        let mut alpha = vec![0., -0.];
        let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, 0, &alpha).unwrap();
        let before = (bits(&alpha), bits(&q), constant.to_bits());
        let mut calls = 0;
        assert!(run
            .apply(
                &p,
                &c,
                &c.g,
                0,
                &mut alpha,
                &mut q,
                &mut constant,
                e,
                &mut || {
                    calls += 1;
                    calls != fail_at
                }
            )
            .is_none());
        assert_eq!(calls, fail_at);
        assert_eq!((bits(&alpha), bits(&q), constant.to_bits()), before);
        let mut retry_polls = 0;
        assert!(run
            .apply(
                &p,
                &c,
                &c.g,
                0,
                &mut alpha,
                &mut q,
                &mut constant,
                e,
                &mut || {
                    retry_polls += 1;
                    true
                }
            )
            .is_none());
        assert_eq!(retry_polls, 0);
    }
}

#[test]
fn root_line_original_and_shortened_deadlines_cannot_be_extended() {
    let (p, c) = stall(2);
    let future = end();
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    let schedule = Schedule::new(2, 1, 2, 2).unwrap();
    for (original_end, call_end) in [(expired, future), (future, expired)] {
        let mut run = Run::new(schedule, &p, original_end);
        let mut alpha = vec![0., -0.];
        let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, 0, &alpha).unwrap();
        let before = (bits(&alpha), bits(&q), constant.to_bits());
        let mut polls = 0;
        assert!(run
            .apply(
                &p,
                &c,
                &c.g,
                0,
                &mut alpha,
                &mut q,
                &mut constant,
                call_end,
                &mut || {
                    polls += 1;
                    true
                }
            )
            .is_none());
        assert!(run
            .apply(
                &p,
                &c,
                &c.g,
                0,
                &mut alpha,
                &mut q,
                &mut constant,
                future,
                &mut || {
                    polls += 1;
                    true
                }
            )
            .is_none());
        assert_eq!(polls, 0);
        assert_eq!((bits(&alpha), bits(&q), constant.to_bits()), before);
    }
}

#[test]
fn root_line_mask_cursor_and_source_identity_reject_unpaid_reentry() {
    let (p, c) = stall(2);
    let foreign = p.clone();
    let e = end();
    let mut run = Run::new(Schedule::new(2, 1, 2, 2).unwrap(), &p, e);
    let mut alpha = vec![0., 0.];
    let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, 0, &alpha).unwrap();
    let mut polls = 0;
    assert!(run
        .apply(
            &foreign,
            &c,
            &c.g,
            0,
            &mut alpha,
            &mut q,
            &mut constant,
            e,
            &mut || {
                polls += 1;
                true
            }
        )
        .is_none());
    assert!(run
        .apply(
            &p,
            &c,
            &c.g,
            1,
            &mut alpha,
            &mut q,
            &mut constant,
            e,
            &mut || {
                polls += 1;
                true
            }
        )
        .is_none());
    let mut malformed = c.clone();
    malformed.positive[0] = p.hidden_dim;
    assert!(run
        .apply(
            &p,
            &malformed,
            &c.g,
            0,
            &mut alpha,
            &mut q,
            &mut constant,
            e,
            &mut || {
                polls += 1;
                true
            }
        )
        .is_none());
    assert_eq!(polls, 0);
    assert!(
        run.apply(
            &p,
            &c,
            &c.g,
            0,
            &mut alpha,
            &mut q,
            &mut constant,
            e,
            &mut || true
        )
        .unwrap()
        .improved
    );
    assert!(run
        .apply(
            &p,
            &c,
            &c.g,
            0,
            &mut alpha,
            &mut q,
            &mut constant,
            e,
            &mut || {
                polls += 1;
                true
            }
        )
        .is_none());
    let mut alpha = vec![0., 0.];
    let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, 1, &alpha).unwrap();
    assert!(
        run.apply(
            &p,
            &c,
            &c.g,
            1,
            &mut alpha,
            &mut q,
            &mut constant,
            e,
            &mut || true
        )
        .unwrap()
        .improved
    );
    assert!(run
        .apply(
            &p,
            &c,
            &c.g,
            2,
            &mut alpha,
            &mut q,
            &mut constant,
            e,
            &mut || {
                polls += 1;
                true
            }
        )
        .is_none());
    assert_eq!(polls, 0);
}

#[test]
fn root_line_refuses_inconsistent_paid_work_without_polling_or_cache_changes() {
    let (p, c) = stall(2);
    let valid = Schedule::new(2, 1, 2, 2).unwrap();
    assert!(valid.enabled);
    for field in 0..3 {
        let mut forged = valid;
        match field {
            0 => forged.line_work = forged.actual_bound.checked_add(1).unwrap(),
            1 => forged.actual_bound = forged.credit.checked_add(1).unwrap(),
            2 => forged.credit = forged.line_work.checked_sub(1).unwrap(),
            _ => unreachable!(),
        }
        let mut run = Run::new(forged, &p, end());
        let mut alpha = vec![0.; c.positive.len()];
        let (mut q, mut constant) = search::mask_cache(&p, &c, &c.g, 0, &alpha).unwrap();
        let before = (bits(&alpha), bits(&q), constant.to_bits());
        let mut polls = 0;
        assert!(run
            .apply(
                &p,
                &c,
                &c.g,
                0,
                &mut alpha,
                &mut q,
                &mut constant,
                end(),
                &mut || {
                    polls += 1;
                    true
                },
            )
            .is_none());
        assert_eq!(polls, 0);
        assert_eq!((bits(&alpha), bits(&q), constant.to_bits()), before);
        assert_eq!(run.next_mask, 0);
        assert_eq!(run.schedule.credit, forged.credit);
    }
    assert!(!Run::new(valid, &p, end()).cancelled);
}
