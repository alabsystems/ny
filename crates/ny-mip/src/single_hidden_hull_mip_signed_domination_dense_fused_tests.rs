// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Literal pre-fusion arithmetic controls.
use super::{seed, Seed};
use crate::single_hidden_hull::primal_advice::Point;
use crate::single_hidden_hull::HullMarginProblem;
use std::time::{Duration, Instant};

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn center(l: f64, u: f64) -> Option<f64> {
    finite(0.5 * l + 0.5 * u)
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}
fn problem(rows: &[&[f64]], bias: &[f64], lower: &[f64], upper: &[f64]) -> HullMarginProblem {
    let h = rows.len();
    let mut output = vec![1.; h];
    output[h - 1] = -1.;
    output.resize(2 * h, 0.);
    HullMarginProblem::new(
        rows.iter().flat_map(|r| r.iter().copied()).collect(),
        bias.to_vec(),
        lower.len(),
        &output,
        &[0., 0.],
        2,
        &[(0, 1.), (1, -1.)],
        0.,
        lower.to_vec(),
        upper.to_vec(),
    )
    .unwrap()
}
fn point(input: &[f64], values: &[f64], positive: &[usize]) -> Point {
    Point {
        input: input.to_vec().into_boxed_slice(),
        values: values.to_vec().into_boxed_slice(),
        positive: positive.to_vec().into_boxed_slice(),
        negative: vec![values.len() - 1].into_boxed_slice(),
    }
}

/// Literal pool/fit from the reviewed pre-fusion dense source a2c14a42.
/// This intentionally recomputes center, width, pb and denominator at every
/// original location. It neither calls the fused cache nor changes grouping.
fn literal_seed(p: &HullMarginProblem, point: &Point, head: usize, q: usize) -> Option<Seed> {
    let d = p.input_dim;
    let mut centered_target = p.b1[head];
    for i in 0..d {
        centered_target =
            finite(centered_target + p.w_row(head)[i] * center(p.lower[i], p.upper[i])?)?;
    }
    let mut pool: Vec<(f64, usize)> = Vec::with_capacity(q);
    for &j in &point.positive {
        if *p.c.get(j)? <= 0. || pool.iter().any(|(_, old)| *old == j) {
            continue;
        }
        let mut pb = p.b1[j];
        let (mut numerator, mut denominator) = (0., 0.);
        for i in 0..d {
            let width = finite(0.5 * p.upper[i] - 0.5 * p.lower[i])?;
            pb = finite(pb + p.w_row(j)[i] * center(p.lower[i], p.upper[i])?)?;
            let a = finite(p.w_row(head)[i] * width)?;
            let b = finite(p.w_row(j)[i] * width)?;
            numerator = finite(numerator + finite(a * b)?)?;
            denominator = finite(denominator + finite(b * b)?)?;
        }
        numerator = finite(numerator + finite(centered_target * pb)?)?;
        denominator = finite(denominator + finite(pb * pb)?)?;
        if numerator <= 0. || denominator <= 0. {
            continue;
        }
        let estimate = finite(numerator / denominator)?.clamp(0., 256.);
        let penalty = finite(1. + finite(estimate * finite(point.values[j])?.max(0.))?)?;
        let score = finite(finite(estimate * numerator)? / penalty)?;
        if score <= 0. {
            continue;
        }
        let slot = pool
            .iter()
            .position(|&(old, k)| score > old || (score == old && j < k))
            .unwrap_or(pool.len());
        if slot < q {
            pool.insert(slot, (score, j));
            if pool.len() > q {
                pool.pop();
            }
        }
    }
    let mut residual = p.w_row(head).to_vec();
    let mut bias = p.b1[head];
    let (mut positive, mut lambda) = (Vec::with_capacity(q), Vec::with_capacity(q));
    for (_, j) in pool {
        let mut rb = bias;
        let mut pb = p.b1[j];
        let (mut numerator, mut denominator) = (0., 0.);
        for i in 0..d {
            let midpoint = center(p.lower[i], p.upper[i])?;
            let width = finite(0.5 * p.upper[i] - 0.5 * p.lower[i])?;
            rb = finite(rb + residual[i] * midpoint)?;
            pb = finite(pb + p.w_row(j)[i] * midpoint)?;
            let a = finite(residual[i] * width)?;
            let b = finite(p.w_row(j)[i] * width)?;
            numerator = finite(numerator + finite(a * b)?)?;
            denominator = finite(denominator + finite(b * b)?)?;
        }
        numerator = finite(numerator + finite(rb * pb)?)?;
        denominator = finite(denominator + finite(pb * pb)?)?;
        let gain = if denominator > 0. {
            finite(numerator / denominator)?.clamp(0., 256.)
        } else {
            0.
        };
        if gain == 0. {
            continue;
        }
        bias = finite(bias - finite(gain * p.b1[j])?)?;
        for (i, r) in residual.iter_mut().enumerate() {
            *r = finite(*r - finite(gain * p.w_row(j)[i])?)?;
        }
        positive.push(j);
        lambda.push(gain);
    }
    (!positive.is_empty()).then_some(Seed {
        positive,
        lambda,
        residual,
        bias,
    })
}
fn parity(p: &HullMarginProblem, point: &Point, head: usize, q: usize) -> Seed {
    let old = literal_seed(p, point, head, q).expect("control has a nonzero old seed");
    let actual = seed(p, point, head, q, end(), &mut || true).expect("fused seed");
    assert_eq!(actual.positive.len(), actual.lambda.len());
    assert!(actual.positive.len() <= q);
    let retained: Vec<_> = actual
        .positive
        .iter()
        .copied()
        .zip(&actual.lambda)
        .filter(|(_, a)| **a != 0.)
        .map(|(j, a)| (j, a.to_bits()))
        .collect();
    let expected: Vec<_> = old
        .positive
        .iter()
        .copied()
        .zip(&old.lambda)
        .map(|(j, a)| (j, a.to_bits()))
        .collect();
    assert_eq!(retained, expected, "exact old nonzero IDs/coefficient bits");
    assert_eq!(bits(&actual.residual), bits(&old.residual), "residual bits");
    assert_eq!(actual.bias.to_bits(), old.bias.to_bits(), "bias bits");
    actual
}

#[test]
fn fused_seed_literal_old_fit_bits_survive_signed_translated_and_fixed_coordinates() {
    let p = problem(
        &[
            &[1., -2., 0.5],
            &[-0.25, 1., 2.],
            &[2., 1., -1.],
            &[2.5, -1., 0.25],
        ],
        &[0.25, -0.75, 2., -1.125],
        &[-3., -0.5, 2.],
        &[1., 4., 2.],
    );
    let hint = point(&[-0.5, 2., 2.], &[0.125, 0.75, 0., 1.], &[0, 1, 2]);
    for q in [1, 2, 3] {
        parity(&p, &hint, 3, q);
    }
}

#[test]
fn fused_seed_equal_and_near_scores_keep_original_heads_not_feature_indices() {
    let tiny = 2f64.powi(-27);
    let near = 2f64.powi(-25);
    let p = problem(
        &[&[1., tiny], &[1., -tiny], &[1., near], &[1., 0.]],
        &[0.; 4],
        &[-1.; 2],
        &[1.; 2],
    );
    // A private malformed ordering is still untrusted advice. The literal
    // ranking tie uses original head ID even when cache indices are reversed.
    let hint = point(&[0.; 2], &[0., 0., 0., 1.], &[2, 1, 0]);
    let actual = parity(&p, &hint, 3, 2);
    assert_eq!(actual.positive, vec![0, 1]);
    assert_eq!(actual.lambda[0], 1.);
    assert!(actual.lambda[1] > 0. && actual.lambda[1] < 1e-12);
}

#[test]
fn fused_seed_retains_zero_gain_partner_without_changing_signed_zero_residual_or_bias() {
    let p = problem(
        &[&[0., 1.], &[-0., 1.], &[-0., 1.]],
        &[0., -0., -0.],
        &[-1.; 2],
        &[1.; 2],
    );
    let hint = point(&[0.; 2], &[0., 0., 1.], &[0, 1]);
    let actual = parity(&p, &hint, 2, 2);
    assert_eq!(actual.positive, vec![0, 1]);
    assert_eq!(actual.lambda, vec![1., 0.]);
    assert_eq!(actual.residual[0].to_bits(), (-0f64).to_bits());
    assert_eq!(actual.bias.to_bits(), (-0f64).to_bits());
}

#[test]
fn fused_seed_cached_bias_and_total_denominator_preserve_finite_cancellation_bits() {
    let large = 2f64.powi(40);
    let p = problem(
        &[&[large, 1., -large], &[0., 1., 0.], &[large, 2., -large]],
        &[-2. * large, 0., -2. * large],
        &[2., -2., 1.],
        &[4., 2., 1.],
    );
    let hint = point(&[3., 0., 1.], &[0., 0., 1.], &[0, 1]);
    let actual = parity(&p, &hint, 2, 2);
    assert_eq!(actual.positive, vec![0, 1]);
    assert_eq!(actual.lambda, vec![1., 1.]);
    assert!(actual.residual.iter().all(|x| *x == 0.));
    assert_eq!(actual.bias, 0.);

    // The old subtractive violation can survive rounding while the additive
    // seed price rounds above the target. A cheaper nonviolating replacement
    // must preserve that useful old scalar/chain pair.
    let target = 1f64;
    let product = 2f64.powi(-54);
    let old_best = target - 2f64.powi(-53);
    let (mut lhs, mut head_price) = (target, 0.);
    for _ in 0..5 {
        lhs -= product;
        head_price += product;
    }
    let full_seed_price = head_price + old_best;
    assert_eq!(lhs, target);
    assert!(lhs > old_best);
    assert_eq!(full_seed_price, target + 2f64.powi(-52));
    let mut candidate = super::super::super::priced::Candidate {
        lambda: vec![1.; 5],
        slope: vec![0.],
        value: target,
        attempted: true,
        completed: true,
        improved: true,
    };
    assert_eq!(target, candidate.value);
    assert!(candidate.value < full_seed_price);
    assert!(!super::super::accept_priced(
        &candidate,
        Some(full_seed_price),
        target
    ));
    candidate.value = old_best;
    assert!(super::super::accept_priced(
        &candidate,
        Some(full_seed_price),
        target
    ));
    candidate.lambda.fill(-0.);
    assert!(!super::super::accept_priced(
        &candidate,
        Some(full_seed_price),
        target
    ));
}

#[test]
fn fused_seed_nonfinite_hint_and_finite_affine_overflow_refuse_without_partial_seed() {
    let p = problem(
        &[&[1., 0.], &[0., 1.], &[1., 1.]],
        &[0.; 3],
        &[-1.; 2],
        &[1.; 2],
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let hint = point(&[0.; 2], &[bad, 0., 1.], &[0, 1]);
        assert!(literal_seed(&p, &hint, 2, 2).is_none());
        assert!(seed(&p, &hint, 2, 2, end(), &mut || true).is_none());
    }
    let mut overflow = p.clone();
    overflow.lower = vec![1., -1.];
    overflow.upper = vec![2., 1.];
    overflow.w1[4] = f64::MAX;
    let hint = point(&[1.5, 0.], &[0., 0., 1.], &[0, 1]);
    assert!(literal_seed(&overflow, &hint, 2, 2).is_none());
    assert!(seed(&overflow, &hint, 2, 2, end(), &mut || true).is_none());
    assert_eq!(p.w1[4], 1.);
    assert_eq!(p.lower, [-1.; 2]);
    assert_eq!(p.upper, [1.; 2]);
}

#[test]
fn fused_seed_cancellation_and_expired_original_end_do_not_finish_partial_fit() {
    let p = problem(
        &[&[1., 0.], &[1., 0.], &[0., 1.], &[1., 1.]],
        &[0.; 4],
        &[-1.; 2],
        &[1.; 2],
    );
    let hint = point(&[0.; 2], &[0., 0., 0., 1.], &[0, 1, 2]);
    assert!(seed(&p, &hint, 3, 3, end(), &mut || true).is_some());
    for stop_at in [1, 4] {
        let mut seen = 0;
        assert!(seed(&p, &hint, 3, 3, end(), &mut || {
            seen += 1;
            seen < stop_at
        })
        .is_none());
        assert!(seen > 0 && seen <= stop_at);
    }
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("overflow when subtracting duration from instant");
    assert!(seed(&p, &hint, 3, 3, expired, &mut || true).is_none());
}

#[derive(Clone, Copy)]
struct OldEntry {
    input: usize,
    width: f64,
    increment: f64,
    normalized: f64,
    sign: f64,
}
/// Literal original insertion helper, independent of the retained merge sorter.
fn literal_chain(
    c: &[f64],
    bias: f64,
    lower: &[f64],
    upper: &[f64],
    advice: &[f64],
) -> Option<Vec<f64>> {
    let n = c.len();
    if n > 128 || lower.len() != n || upper.len() != n || advice.len() != n || !bias.is_finite() {
        return None;
    }
    for i in 0..n {
        if !c[i].is_finite()
            || !lower[i].is_finite()
            || !upper[i].is_finite()
            || !advice[i].is_finite()
            || lower[i] > upper[i]
        {
            return None;
        }
    }
    let mut value = bias;
    let mut order: Vec<OldEntry> = Vec::with_capacity(n);
    for i in 0..n {
        if c[i] == 0. {
            continue;
        }
        let (base, sign) = if c[i] > 0. {
            (lower[i], 1.)
        } else {
            (upper[i], -1.)
        };
        value = finite(value + finite(c[i] * base)?)?;
        if lower[i] == upper[i] {
            continue;
        }
        let width = finite(upper[i] - lower[i])?;
        let increment = finite(c[i].abs() * width)?;
        if width <= 0. || increment <= 0. {
            return None;
        }
        let point = advice[i].clamp(lower[i], upper[i]);
        let offset = finite(if sign > 0. {
            point - lower[i]
        } else {
            upper[i] - point
        })?;
        let normalized = finite(offset / width)?.clamp(0., 1.);
        let entry = OldEntry {
            input: i,
            width,
            increment,
            normalized,
            sign,
        };
        let mut at = order.len();
        while at > 0
            && (order[at - 1].normalized < normalized
                || (order[at - 1].normalized == normalized && order[at - 1].input > i))
        {
            at -= 1;
        }
        order.insert(at, entry);
    }
    let mut slope = vec![0.; n];
    let mut previous = value.max(0.);
    for entry in order {
        value = finite(value + entry.increment)?;
        let next = value.max(0.);
        let delta = finite(next - previous)?;
        if delta < 0. {
            return None;
        }
        slope[entry.input] = finite(entry.sign * finite(delta / entry.width)?)?;
        previous = next;
    }
    Some(slope)
}
fn chain_parity(c: &[f64], bias: f64, lower: &[f64], upper: &[f64], advice: &[f64]) {
    let old = literal_chain(c, bias, lower, upper, advice);
    let actual = super::super::super::chain::nominal_chain_direction(c, bias, lower, upper, advice);
    match (old, actual) {
        (Some(old), Some(actual)) => {
            assert_eq!(bits(&old), bits(&actual), "literal old slope bits")
        }
        (None, None) => {}
        _ => panic!("retained chain changed acceptance"),
    }
}

#[test]
fn fused_retained_chain_merge_preserves_literal_insertion_arithmetic_and_tie_bits() {
    let tiny = [2f64.powi(-54), 2f64.powi(-55)];
    chain_parity(&[-1., -1.], 0., &[0.; 2], &[1.; 2], &tiny);
    // Ordinary comparisons treat normalized -0/+0 as equal. total_cmp would
    // reverse these heads and change their slopes despite equal advice means.
    chain_parity(&[1., 1.], -0.5, &[0.; 2], &[1.; 2], &[-0., 0.]);
    chain_parity(
        &[-0., 1., -1., 2.],
        -0.,
        &[-1., -2., 0.5, 1.],
        &[1., 2., 0.5, 3.],
        &[-0., 0., 0.5, 2.],
    );
    for n in [1, 2, 3, 31, 32, 33, 127, 128] {
        let c: Vec<_> = (0..n)
            .map(|i| match i % 4 {
                0 => -1.,
                1 => 0.,
                2 => 0.5,
                _ => 1.,
            })
            .collect();
        let hint: Vec<_> = (0..n).map(|i| (i % 5) as f64 / 2. - 1.).collect();
        let lower = vec![-1.; n];
        let upper = vec![1.; n];
        assert!(literal_chain(&c, -0.25, &lower, &upper, &hint).is_some());
        chain_parity(&c, -0.25, &lower, &upper, &hint);
    }
    chain_parity(&[f64::from_bits(1)], 0., &[0.], &[f64::from_bits(1)], &[0.]);
    chain_parity(&[1., f64::MAX], f64::MAX, &[-1.; 2], &[1.; 2], &[0.; 2]);
}
