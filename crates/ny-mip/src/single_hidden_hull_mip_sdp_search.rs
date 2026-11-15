// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded scalar logdet/Newton PROPOSALS. No float result grants a bound.
use super::*;
const MAX_NEWTON: usize = 48;
const MAX_BACKTRACK: usize = 6;
pub(super) fn factor(a: &[f64], n: usize, b: &Budget) -> Option<Vec<f64>> {
    b.factor(n)?;
    let mut l = vec![0.; n * n];
    for i in 0..n {
        if !live(b.search_end) {
            return None;
        }
        for j in 0..=i {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                if v <= 0. {
                    return None;
                }
                v.sqrt()
            } else {
                v / l[j * n + j]
            };
            if !l[i * n + j].is_finite() {
                return None;
            }
        }
    }
    Some(l)
}
fn solve(l: &[f64], rhs: &[f64], n: usize, b: &Budget) -> Option<Vec<f64>> {
    b.charge(4 * n * n, false)?;
    let mut x = rhs.to_vec();
    for i in 0..n {
        if !live(b.search_end) {
            return None;
        }
        for j in 0..i {
            x[i] -= l[i * n + j] * x[j];
        }
        x[i] /= l[i * n + i];
        if !x[i].is_finite() {
            return None;
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= l[j * n + i] * x[j];
        }
        x[i] /= l[i * n + i];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}
fn inverse(l: &[f64], n: usize, b: &Budget) -> Option<Vec<f64>> {
    let mut out = vec![0.; n * n];
    let mut rhs = vec![0.; n];
    for j in 0..n {
        rhs[j] = 1.;
        let x = solve(l, &rhs, n, b)?;
        rhs[j] = 0.;
        for i in 0..n {
            out[i * n + j] = x[i];
        }
    }
    Some(out)
}
fn matrix(q: &[f64], lambda: &[f64], n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|v| {
            if v / n == v % n {
                lambda[v / n] - q[v]
            } else {
                -q[v]
            }
        })
        .collect()
}
fn objective(lambda: &[f64], l: &[f64], n: usize, mu: f64) -> Option<f64> {
    finite(lambda.iter().sum::<f64>() - 2. * mu * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>())
}
fn full_candidate(
    active: &[usize],
    small_lambda: &[f64],
    small_l: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let k = active.len();
    let mut lambda = vec![0.; n];
    let mut l = vec![0.; n * n];
    for (i, &ii) in active.iter().enumerate() {
        lambda[ii] = small_lambda[i];
        for j in 0..=i {
            l[ii * n + active[j]] = small_l[i * k + j];
        }
    }
    (lambda, l)
}
/// Point-zero coordinates are omitted ONLY by the float proposer. The full
/// Source interval matrix, including their rounding defects, is always checked.
// Preserve the proposal's original addition-then-scale rounding and finite
// domain. An overflowed symmetric sum must still fail at the charged factor
// barrier, rather than acquire a wider domain through overflow-safe midpoint.
fn symmetric_inverse_mean(a: f64, b: f64) -> Option<f64> {
    let sum = a + b;
    if !sum.is_finite() {
        return None;
    }
    Some(sum * 0.5)
}

pub(super) fn run(
    s: &System,
    b: &Budget,
    consider: &mut dyn FnMut(Vec<f64>, Vec<f64>),
) -> Option<usize> {
    let active: Vec<_> = (0..s.n)
        .filter(|&i| (0..s.n).any(|j| s.q[i * s.n + j].lo.midpoint(s.q[i * s.n + j].hi) != 0.))
        .collect();
    let n = active.len();
    b.counts.lock().ok()?.active = n;
    if n == 0 {
        consider(vec![0.; s.n], vec![0.; s.n * s.n]);
        return Some(0);
    }
    b.charge(8 * s.n * s.n, false)?;
    let q: Vec<_> = active
        .iter()
        .flat_map(|&i| {
            active
                .iter()
                .map(move |&j| s.q[i * s.n + j].lo.midpoint(s.q[i * s.n + j].hi))
        })
        .collect();
    let scale = (0..n)
        .map(|i| q[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum::<f64>())
        .fold(0., f64::max)
        .max(1e-8);
    let mut lambda: Vec<_> = (0..n)
        .map(|i| q[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum::<f64>() + scale * 0.25)
        .collect();
    let mut mu = scale * 0.25;
    let mut last: Option<Vec<f64>> = None;
    for step in 0..MAX_NEWTON {
        if b.iteration().is_none() {
            break;
        }
        let Some(l) = factor(&matrix(&q, &lambda, n), n, b) else {
            break;
        };
        if step % 8 == 7 {
            let (ll, ff) = full_candidate(&active, &lambda, &l, s.n);
            consider(ll, ff);
        }
        let Some(inv) = inverse(&l, n, b) else {
            last = Some(l);
            break;
        };
        b.charge(8 * n * n, false)?;
        let gradient: Vec<_> = (0..n).map(|i| 1. - mu * inv[i * n + i]).collect();
        if gradient.iter().any(|v| !v.is_finite()) {
            break;
        }
        let h: Vec<_> = (0..n * n)
            .map(|ij| {
                let i = ij / n;
                let j = ij % n;
                let Some(v) = symmetric_inverse_mean(inv[i * n + j], inv[j * n + i]) else {
                    // Keep factor's existing charge, refusal and completed-step
                    // retention even when the symmetric sum is nonfinite.
                    return f64::NAN;
                };
                mu * v * v + if i == j { 1e-12 / scale } else { 0. }
            })
            .collect();
        let Some(hl) = factor(&h, n, b) else {
            last = Some(l);
            break;
        };
        let rhs: Vec<_> = gradient.iter().map(|v| -v).collect();
        let Some(direction) = solve(&hl, &rhs, n, b) else {
            last = Some(l);
            break;
        };
        let Some(old) = objective(&lambda, &l, n, mu) else {
            break;
        };
        let slope: f64 = gradient.iter().zip(&direction).map(|(a, bb)| a * bb).sum();
        let mut accepted = false;
        let mut t = 1.;
        for _ in 0..MAX_BACKTRACK {
            b.charge(8 * n * n, false)?;
            let trial: Vec<_> = lambda
                .iter()
                .zip(&direction)
                .map(|(v, dx)| v + t * dx)
                .collect();
            if trial.iter().all(|v| v.is_finite() && *v > 0.) {
                if let Some(tl) = factor(&matrix(&q, &trial, n), n, b) {
                    if objective(&trial, &tl, n, mu)
                        .is_some_and(|v| v <= old + 1e-4 * t * slope.min(0.))
                    {
                        lambda = trial;
                        last = Some(tl);
                        accepted = true;
                        break;
                    }
                }
            }
            t *= 0.5;
        }
        if !accepted {
            last = Some(l);
        }
        let norm = gradient.iter().map(|v| v.abs()).fold(0., f64::max);
        if norm < 0.2 || step % 3 == 2 || !accepted {
            mu = (mu * 0.2).max(scale * 1e-12);
        }
    }
    // A failed later step keeps a previously completed proposal. Its factor
    // must correspond to the SAME lambda, never the previous iteration's bits.
    if let Some(l) = last {
        let (ll, ff) = full_candidate(&active, &lambda, &l, s.n);
        consider(ll, ff);
    }
    Some(n)
}

#[cfg(test)]
mod symmetric_mean_tests {
    use super::super::tests::{cuts, cycle, end};
    use super::*;

    #[test]
    fn symmetric_sum_keeps_legacy_bits_refusal_and_actual_factor_debit() {
        let tiny = f64::from_bits(1);
        let cases = [
            (0., 0.),
            (-0., -0.),
            (-0., 0.),
            (tiny, 0.),
            (tiny, tiny),
            (-tiny, tiny),
            (f64::MIN_POSITIVE, tiny),
            (1., 1.0000000000000002),
            (f64::MAX, -f64::MAX),
            (f64::MAX, 0.),
            (f64::MAX, f64::MAX),
            (-f64::MAX, -f64::MAX),
            (f64::INFINITY, 1.),
            (f64::NEG_INFINITY, f64::INFINITY),
            (f64::NAN, 1.),
        ];
        let p = cycle(2, 0.);
        let cuts = cuts(&p);
        for (a, b) in cases {
            let legacy_sum = a + b;
            let legacy_mean = legacy_sum * 0.5;
            let checked = symmetric_inverse_mean(a, b);
            if legacy_sum.is_finite() {
                assert_eq!(checked.unwrap().to_bits(), legacy_mean.to_bits());
            } else {
                assert!(checked.is_none());
            }
            let legacy_h = 0.25 * legacy_mean * legacy_mean + 1e-12;
            let checked_h = checked.map_or(f64::NAN, |v| 0.25 * v * v + 1e-12);
            let old = Budget::new(&cuts, end(), Duration::ZERO, 2).unwrap();
            let new = Budget::new(&cuts, end(), Duration::ZERO, 2).unwrap();
            let legacy_factor = factor(&[legacy_h], 1, &old);
            let checked_factor = factor(&[checked_h], 1, &new);
            assert_eq!(
                checked_factor.as_ref().map(|v| v[0].to_bits()),
                legacy_factor.as_ref().map(|v| v[0].to_bits()),
            );
            let old_counts = old.counts.lock().unwrap();
            let new_counts = new.counts.lock().unwrap();
            assert_eq!(
                (new_counts.work, new_counts.factors),
                (old_counts.work, old_counts.factors)
            );
            assert_eq!((new_counts.work, new_counts.factors), (2, 1));
            assert_eq!(new_counts.reserved, old_counts.reserved);
        }
        assert!(symmetric_inverse_mean(f64::MAX, f64::MAX).is_none());
        assert_eq!(f64::MAX.midpoint(f64::MAX), f64::MAX);
    }
}
