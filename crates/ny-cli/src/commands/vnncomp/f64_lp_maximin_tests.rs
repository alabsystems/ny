// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #sbench-f64-maximin unit tests: the maximin LP against an independent
//! reference, the acceptance rule of the SLP step observed from outside, and
//! the hop cap / deadline. The plan-integrated tests (planted thin region,
//! slab soundness, dark-by-default) live in `f64_target_attack_tests.rs`.

use std::cell::RefCell;
use std::time::{Duration, Instant};

use super::{lp_maximin_lift_core, rounding_direction, solve_maximin_lp, HopRng, MaximinLp};

/// Deterministic test generator (xorshift64).
struct TestRng(u64);

impl TestRng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn symmetric(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }
}

fn row_value(margins: &[f64], rows: &[Vec<f64>], delta: &[f64]) -> f64 {
    rows.iter()
        .zip(margins)
        .map(|(row, &m)| m + row.iter().zip(delta).map(|(j, x)| j * x).sum::<f64>())
        .fold(f64::INFINITY, f64::min)
}

/// Solve the `n x n` system `a x = b` by Gaussian elimination with partial
/// pivoting; `None` when (numerically) singular.
fn solve_small(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let acc: f64 = b[r] - (r + 1..n).map(|c| a[r][c] * x[c]).sum::<f64>();
        x[r] = acc / a[r][r];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Independent reference for small problems: the maximum of the concave
/// piecewise-linear `min_i (m_i + J_i . D)` over the box is attained at a
/// point where `n` linearly independent equations hold among the box faces
/// `D_d = lower_d | upper_d` and the tie planes `(J_i - J_j) . D = m_j - m_i`
/// (a vertex of the hypograph). Enumerate every such point.
fn reference_lp_by_vertex_enumeration(
    margins: &[f64],
    rows: &[Vec<f64>],
    lower: &[f64],
    upper: &[f64],
) -> f64 {
    let n = lower.len();
    let k = rows.len();
    let mut planes: Vec<(Vec<f64>, f64)> = Vec::new();
    for d in 0..n {
        let mut e = vec![0.0; n];
        e[d] = 1.0;
        planes.push((e.clone(), lower[d]));
        planes.push((e, upper[d]));
    }
    for i in 0..k {
        for j in i + 1..k {
            let a: Vec<f64> = (0..n).map(|d| rows[i][d] - rows[j][d]).collect();
            planes.push((a, margins[j] - margins[i]));
        }
    }
    let mut best = f64::NEG_INFINITY;
    let mut pick = vec![0usize; n];
    // Iterate every n-subset of the planes.
    fn rec(
        start: usize,
        depth: usize,
        pick: &mut Vec<usize>,
        planes: &[(Vec<f64>, f64)],
        visit: &mut dyn FnMut(&[usize]),
    ) {
        if depth == pick.len() {
            visit(pick.as_slice());
            return;
        }
        for q in start..planes.len() {
            pick[depth] = q;
            rec(q + 1, depth + 1, pick, planes, visit);
        }
    }
    let mut visit = |chosen: &[usize]| {
        let a: Vec<Vec<f64>> = chosen.iter().map(|&q| planes[q].0.clone()).collect();
        let b: Vec<f64> = chosen.iter().map(|&q| planes[q].1).collect();
        if let Some(x) = solve_small(a, b) {
            let inside = x
                .iter()
                .zip(lower.iter().zip(upper))
                .all(|(v, (lo, hi))| *v >= lo - 1e-12 && *v <= hi + 1e-12);
            if inside {
                let clamped: Vec<f64> = x
                    .iter()
                    .zip(lower.iter().zip(upper))
                    .map(|(v, (lo, hi))| v.clamp(*lo, *hi))
                    .collect();
                best = best.max(row_value(margins, rows, &clamped));
            }
        }
    };
    rec(0, 0, &mut pick, &planes, &mut visit);
    best
}

/// The dual objective `sum_i y_i m_i + sum_d max(g_d lower_d, g_d upper_d)`,
/// `g = J^T y`: by weak duality an upper bound on the LP optimum for every
/// `y >= 0` summing to one, and equal to it at the optimal multipliers.
fn dual_objective(
    lp: &MaximinLp,
    margins: &[f64],
    rows: &[Vec<f64>],
    lower: &[f64],
    upper: &[f64],
) -> f64 {
    let n = lower.len();
    let mut g = vec![0.0f64; n];
    for (row, &y) in rows.iter().zip(&lp.duals) {
        for d in 0..n {
            g[d] += y * row[d];
        }
    }
    let mut value: f64 = lp.duals.iter().zip(margins).map(|(y, m)| y * m).sum();
    for d in 0..n {
        value += (g[d] * lower[d]).max(g[d] * upper[d]);
    }
    value
}

/// The in-crate simplex (the only solver; a float LP suffices because the
/// step is non-authoritative) matches an INDEPENDENT reference: exhaustive
/// vertex enumeration on small problems, and its own optimality certificate
/// (the dual objective of its multipliers equals its primal value) on medium
/// ones. Every returned step lies inside the box.
#[test]
fn lp_maximin_fallback_matches_the_lp_on_small_problems() {
    let mut rng = TestRng(0x00F3_11F7_5EED_0001);
    let mut checked = 0usize;
    for trial in 0..400 {
        let n = 1 + trial % 3;
        let k = 1 + (trial / 3) % 3;
        let coeff_scale = 10f64.powf(-3.0 + 4.0 * rng.unit());
        let margin_scale = 10f64.powf(-7.0 + 6.0 * rng.unit());
        let rows: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..n).map(|_| coeff_scale * rng.symmetric()).collect())
            .collect();
        let margins: Vec<f64> = (0..k).map(|_| margin_scale * rng.symmetric()).collect();
        let radius = 10f64.powf(-8.0 + 8.0 * rng.unit());
        let lower: Vec<f64> = (0..n).map(|_| -radius * rng.unit()).collect();
        let upper: Vec<f64> = (0..n).map(|_| radius * rng.unit()).collect();
        let lp = solve_maximin_lp(&margins, &rows, &lower, &upper, None)
            .unwrap_or_else(|| panic!("trial {trial}: a well-posed LP must solve"));
        assert!(lp.optimal, "trial {trial}: tiny LPs finish");
        for d in 0..n {
            assert!(
                lp.delta[d] >= lower[d] && lp.delta[d] <= upper[d],
                "trial {trial}: step outside the box"
            );
        }
        assert_eq!(lp.predicted, row_value(&margins, &rows, &lp.delta));
        let reference = reference_lp_by_vertex_enumeration(&margins, &rows, &lower, &upper);
        let spread = margins.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - margins.iter().copied().fold(f64::INFINITY, f64::min);
        let reach: f64 = rows
            .iter()
            .map(|row| {
                row.iter()
                    .zip(lower.iter().zip(&upper))
                    .map(|(j, (lo, hi))| j.abs() * (hi - lo))
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let scale = spread.max(reach).max(1e-300);
        assert!(
            (lp.predicted - reference).abs() <= 1e-9 * scale,
            "trial {trial} (n={n}, k={k}): simplex {} vs vertex enumeration {reference}",
            lp.predicted
        );
        checked += 1;
    }
    assert_eq!(checked, 400);

    for trial in 0..60 {
        let n = 20 + trial % 21;
        let k = 4 + trial % 9;
        let rows: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..n).map(|_| rng.symmetric()).collect())
            .collect();
        let margins: Vec<f64> = (0..k).map(|_| 1e-5 * rng.symmetric()).collect();
        let radius = 10f64.powf(-6.0 + 5.0 * rng.unit());
        let lower: Vec<f64> = (0..n).map(|_| -radius * rng.unit()).collect();
        let upper: Vec<f64> = (0..n).map(|_| radius * rng.unit()).collect();
        let lp = solve_maximin_lp(&margins, &rows, &lower, &upper, None).expect("solves");
        assert!(lp.optimal, "medium trial {trial}");
        let total: f64 = lp.duals.iter().sum();
        assert!(
            (total - 1.0).abs() <= 1e-6,
            "medium trial {trial}: multipliers sum to {total}"
        );
        let dual = dual_objective(&lp, &margins, &rows, &lower, &upper);
        let scale = rows
            .iter()
            .map(|row| {
                row.iter()
                    .zip(lower.iter().zip(&upper))
                    .map(|(j, (lo, hi))| j.abs() * (hi - lo))
                    .sum::<f64>()
            })
            .fold(1e-5f64, f64::max);
        assert!(
            (dual - lp.predicted).abs() <= 1e-8 * scale,
            "medium trial {trial}: dual {dual} vs primal {} (not optimal)",
            lp.predicted
        );
    }
}

/// Structural edge cases: malformed input declines; a zero-width coordinate
/// and a zero-gradient coordinate never move; a single row is bang-bang.
#[test]
fn lp_maximin_handles_degenerate_columns_and_refuses_malformed_input() {
    let rows = vec![vec![2.0, -3.0, 0.0, 5.0]];
    let lower = vec![-0.1, -0.2, -0.3, 0.0];
    let upper = vec![0.1, 0.2, 0.3, 0.0];
    let lp = solve_maximin_lp(&[-1e-3], &rows, &lower, &upper, None).expect("solves");
    assert_eq!(lp.delta, vec![0.1, -0.2, 0.0, 0.0]);
    assert!((lp.predicted - (-1e-3 + 0.2 + 0.6)).abs() < 1e-15);
    assert!(
        solve_maximin_lp(&[], &[], &[], &[], None).is_none(),
        "no rows"
    );
    assert!(
        solve_maximin_lp(&[0.0], &[vec![1.0]], &[0.1], &[-0.1], None).is_none(),
        "an inverted interval"
    );
    assert!(
        solve_maximin_lp(&[f64::NAN], &[vec![1.0]], &[-0.1], &[0.1], None).is_none(),
        "a non-finite margin"
    );
    assert!(
        solve_maximin_lp(&[0.0], &[vec![1.0, 2.0]], &[-0.1], &[0.1], None).is_none(),
        "an arity mismatch"
    );
    assert!(
        solve_maximin_lp(&[0.0, 0.0], &[vec![0.0], vec![0.0]], &[-0.1], &[0.1], None).is_none(),
        "no step can matter"
    );
    // The rounding direction weighs the binding rows by their multipliers.
    let rows = vec![vec![1.0, 0.0], vec![-1.0, 1.0]];
    let lp =
        solve_maximin_lp(&[0.0, 0.0], &rows, &[-1.0, -1.0], &[1.0, 1.0], None).expect("solves");
    let direction = rounding_direction(&lp, &[0.0, 0.0], &rows);
    assert_eq!(direction.len(), 2);
    assert!(direction.iter().all(|v| v.is_finite()));
}

/// The hop generator is deterministic and uniform on `[0, 1)`.
#[test]
fn hop_rng_is_deterministic_and_in_range() {
    let mut a = HopRng::new(42);
    let mut b = HopRng::new(42);
    let mut c = HopRng::new(43);
    let mut differs = false;
    for _ in 0..10_000 {
        let (x, y, z) = (a.unit(), b.unit(), c.unit());
        assert_eq!(x.to_bits(), y.to_bits());
        assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&z));
        differs |= x != z;
    }
    assert!(differs);
    assert_eq!(
        HopRng::new(0).unit().to_bits(),
        HopRng::new(0).unit().to_bits()
    );
}

/// A smooth non-concave landscape: `m_k(x) = c_k - sum_d a_kd (x_d - p_kd)^2
/// + 0.05 sin(9 x_(k mod n))` on `[0, 1]^6`.
struct Landscape {
    c: Vec<f64>,
    a: Vec<Vec<f64>>,
    p: Vec<Vec<f64>>,
}

impl Landscape {
    fn new(seed: u64, k: usize, n: usize, shift: f64) -> Self {
        let mut rng = TestRng(seed);
        Self {
            c: (0..k).map(|_| shift + 0.01 * rng.symmetric()).collect(),
            a: (0..k)
                .map(|_| (0..n).map(|_| 0.5 + rng.unit()).collect())
                .collect(),
            p: (0..k)
                .map(|_| (0..n).map(|_| rng.unit()).collect())
                .collect(),
        }
    }

    fn margins(&self, x: &[f32]) -> Vec<f64> {
        let n = x.len();
        (0..self.c.len())
            .map(|k| {
                let mut v = self.c[k];
                for d in 0..n {
                    let t = f64::from(x[d]) - self.p[k][d];
                    v -= self.a[k][d] * t * t;
                }
                v + 0.05 * (9.0 * f64::from(x[k % n])).sin()
            })
            .collect()
    }

    fn gradient(&self, x: &[f32]) -> Vec<Vec<f64>> {
        let n = x.len();
        (0..self.c.len())
            .map(|k| {
                let mut g: Vec<f64> = (0..n)
                    .map(|d| -2.0 * self.a[k][d] * (f64::from(x[d]) - self.p[k][d]))
                    .collect();
                g[k % n] += 0.45 * (9.0 * f64::from(x[k % n])).cos();
                g
            })
            .collect()
    }
}

/// THE ACCEPTANCE RULE, observed from outside: the Jacobian is only ever
/// requested at the run's start and at ACCEPTED iterates, so the exact
/// minimum margins at successive Jacobian requests must strictly increase —
/// even when the Jacobian the LP is fed is badly wrong (true gradient plus
/// large deterministic noise), so that many proposed steps are worse.
#[test]
fn lp_maximin_step_never_accepts_a_worse_exact_margin() {
    let n = 6;
    let landscape = Landscape::new(0x5EED_0000_ACCE_0001, 5, n, -0.3);
    let log: RefCell<Vec<(Vec<f32>, f64)>> = RefCell::new(Vec::new());
    let jac_points: RefCell<Vec<usize>> = RefCell::new(Vec::new());
    let noise = RefCell::new(TestRng(0xBAD0_6AD1_0000_0001));
    let mut eval = |x: &[f32]| -> Option<(Vec<f64>, usize)> {
        let margins = landscape.margins(x);
        let min = margins.iter().copied().fold(f64::INFINITY, f64::min);
        let mut log = log.borrow_mut();
        log.push((x.to_vec(), min));
        Some((margins, log.len() - 1))
    };
    let mut jacobian = |index: &usize| -> Option<Vec<Vec<f64>>> {
        jac_points.borrow_mut().push(*index);
        let x = log.borrow()[*index].0.clone();
        let mut rng = noise.borrow_mut();
        Some(
            landscape
                .gradient(&x)
                .into_iter()
                .map(|row| row.into_iter().map(|g| g + 3.0 * rng.symmetric()).collect())
                .collect(),
        )
    };
    let mut consulted = 0usize;
    let mut trusted = |_x: &[f32]| -> Option<(bool, f64)> {
        consulted += 1;
        Some((false, -1.0))
    };
    let lo = [0.0f32; 6];
    let hi = [1.0f32; 6];
    let seed = [0.31f32, 0.62, 0.17, 0.88, 0.45, 0.53];
    let (outcome, stats) = lp_maximin_lift_core::<usize>(
        &mut eval,
        &mut jacobian,
        &mut trusted,
        &lo,
        &hi,
        &seed,
        1e3,
        0,
        7,
        Instant::now() + Duration::from_secs(30),
        400,
    )
    .expect("in scope");
    let points = jac_points.borrow();
    let log = log.borrow();
    assert!(stats.accepted >= 1, "some step must be accepted: {stats:?}");
    assert!(
        stats.slp_iters > stats.accepted,
        "the noisy Jacobian must produce rejected steps: {stats:?}"
    );
    // One Jacobian at the start and one after each acceptance that another
    // iteration followed.
    assert!(
        points.len() == stats.accepted + 1 || points.len() == stats.accepted,
        "{} Jacobians for {} accepted steps",
        points.len(),
        stats.accepted
    );
    for pair in points.windows(2) {
        let (before, after) = (log[pair[0]].1, log[pair[1]].1);
        assert!(
            after > before,
            "accepted a step from {before:e} to {after:e} (not a strict improvement)"
        );
    }
    // The outcome's best point is the best exact point ever evaluated.
    let best_seen = log
        .iter()
        .map(|(_, m)| *m)
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(outcome.best_margin_f64, best_seen);
    assert!(outcome.best_margin_f64 >= outcome.initial_margin_f64);
    assert!(
        outcome.violation.is_none(),
        "the trusted forward never confirmed"
    );
    if outcome.best_margin_f64 < 0.0 {
        assert_eq!(
            consulted, 0,
            "never consulted below an exact margin of zero"
        );
    }
}

/// The hop cap and the deadline both bind: exactly `cap` hops when the
/// target is unreachable and time is plentiful; a prompt stop at the
/// deadline with an effectively unbounded cap; nothing at all past it.
#[test]
fn basin_hops_respect_the_deadline_and_the_cap() {
    let n = 6;
    // Every margin is below zero everywhere: the target is unreachable and
    // the trusted forward is never consulted.
    let landscape = Landscape::new(0x5EED_0000_0000_B0B5, 4, n, -5.0);
    let mut eval =
        |x: &[f32]| -> Option<(Vec<f64>, Vec<f32>)> { Some((landscape.margins(x), x.to_vec())) };
    let mut jacobian = |x: &Vec<f32>| -> Option<Vec<Vec<f64>>> { Some(landscape.gradient(x)) };
    let mut consulted = 0usize;
    let mut trusted = |_x: &[f32]| -> Option<(bool, f64)> {
        consulted += 1;
        None
    };
    let lo = [0.0f32; 6];
    let hi = [1.0f32; 6];
    let seed = [0.5f32; 6];

    let mut run = |hops: usize, deadline: Instant| {
        lp_maximin_lift_core::<Vec<f32>>(
            &mut eval,
            &mut jacobian,
            &mut trusted,
            &lo,
            &hi,
            &seed,
            1e-6,
            hops,
            11,
            deadline,
            200,
        )
        .expect("in scope")
    };

    let (_, capped) = run(7, Instant::now() + Duration::from_secs(60));
    assert_eq!(capped.hops, 7, "{capped:?}");
    assert_eq!(capped.stop, "hops exhausted");
    assert!(capped.slp_iters > 0);

    let (_, none) = run(0, Instant::now() + Duration::from_secs(60));
    assert_eq!(none.hops, 0);
    assert_eq!(none.stop, "slp converged");

    let slice = Duration::from_millis(300);
    let started = Instant::now();
    let (outcome, timed) = run(crate::preset::F64_BASIN_HOPS_MAX, started + slice);
    let elapsed = started.elapsed();
    assert_eq!(timed.stop, "deadline", "{timed:?}");
    assert!(timed.hops < crate::preset::F64_BASIN_HOPS_MAX);
    assert!(
        elapsed < slice + Duration::from_secs(3),
        "the lift overran its deadline: {elapsed:?}"
    );
    assert!(outcome.violation.is_none());

    let (_, expired) = run(1_000, Instant::now());
    assert_eq!(expired.stop, "deadline");
    assert_eq!(expired.slp_iters, 0);
    assert_eq!(expired.hops, 0);

    assert_eq!(consulted, 0, "the exact margin never cleared zero");
}
