// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness and proposer tests for #hull-root (`single_hidden_hull.rs`).
//!
//! The load-bearing property is that [`certify_hull_lower_bound`] is a valid
//! lower bound on the TRUE (real-arithmetic) margin minimum for ARBITRARY
//! multipliers. It is checked against dense float sampling (>= 1e5 points in
//! total, with a 1e-9 relative tolerance for the sampler's own rounding) and,
//! at every box vertex plus the best sample of each net, against an
//! EXACT-RATIONAL evaluation of the margin with no tolerance at all.

use super::*;
use num_rational::BigRational;
use num_traits::Zero;
use std::time::{Duration, Instant};

/// Deterministic xorshift64 (tests only).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        v
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn below(&mut self, n: usize) -> usize {
        #[allow(clippy::cast_possible_truncation)]
        let v = (self.next_u64() % n as u64) as usize;
        v
    }
}

/// A random single-hidden-layer net with two outputs and margin `Y0 - Y1`.
struct Net {
    n: usize,
    h: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

const MARGIN: [(usize, f64); 2] = [(0, 1.0), (1, -1.0)];

impl Net {
    fn random(rng: &mut Rng, n: usize, h: usize) -> Self {
        let w1 = (0..h * n).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let b1 = (0..h).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let w2 = (0..2 * h).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let b2 = (0..2).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let mut lower = Vec::with_capacity(n);
        let mut upper = Vec::with_capacity(n);
        for _ in 0..n {
            let l = rng.uniform(-1.0, 0.5);
            let width = if rng.below(10) == 0 {
                0.0
            } else {
                rng.uniform(0.0, 1.5)
            };
            lower.push(l);
            upper.push(l + width);
        }
        Self {
            n,
            h,
            w1,
            b1,
            w2,
            b2,
            lower,
            upper,
        }
    }

    fn problem(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.w1.clone(),
            self.b1.clone(),
            self.n,
            &self.w2,
            &self.b2,
            2,
            &MARGIN,
            0.0,
            self.lower.clone(),
            self.upper.clone(),
        )
        .expect("random net is admitted")
    }

    /// Plain-float margin (sampler only; compared with a tolerance).
    fn margin_f64(&self, x: &[f64]) -> f64 {
        let mut m = self.b2[0] - self.b2[1];
        for j in 0..self.h {
            let mut z = self.b1[j];
            for i in 0..self.n {
                z += self.w1[j * self.n + i] * x[i];
            }
            m += (self.w2[j] - self.w2[self.h + j]) * z.max(0.0);
        }
        m
    }

    /// Exact-rational margin: no rounding anywhere.
    fn margin_exact(&self, x: &[f64]) -> BigRational {
        let q = |v: f64| BigRational::from_float(v).expect("finite");
        let mut m = q(self.b2[0]) - q(self.b2[1]);
        for j in 0..self.h {
            let mut z = q(self.b1[j]);
            for i in 0..self.n {
                z += q(self.w1[j * self.n + i]) * q(x[i]);
            }
            if z > BigRational::zero() {
                m += (q(self.w2[j]) - q(self.w2[self.h + j])) * z;
            }
        }
        m
    }

    fn vertices(&self) -> Vec<Vec<f64>> {
        (0..1usize << self.n)
            .map(|mask| {
                (0..self.n)
                    .map(|i| {
                        if (mask >> i) & 1 == 1 {
                            self.upper[i]
                        } else {
                            self.lower[i]
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Uniform points, random vertices, and points projected onto a neuron's
    /// kink (`z_j = 0`) — where a piecewise-linear minimum tends to sit.
    fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let mut x: Vec<f64> = (0..self.n)
            .map(|i| match rng.below(4) {
                0 => self.lower[i],
                1 => self.upper[i],
                _ => rng.uniform(self.lower[i], self.upper[i]),
            })
            .collect();
        if rng.below(3) == 0 {
            let j = rng.below(self.h);
            let i = rng.below(self.n);
            let wji = self.w1[j * self.n + i];
            if wji != 0.0 {
                let mut rest = self.b1[j];
                for k in 0..self.n {
                    if k != i {
                        rest += self.w1[j * self.n + k] * x[k];
                    }
                }
                let xi = -rest / wji;
                if xi >= self.lower[i] && xi <= self.upper[i] {
                    x[i] = xi;
                }
            }
        }
        x
    }
}

/// Multiplier families: random, out-of-range, degenerate, proposer-shaped
/// and overflow-scale. None of them is "good"; all must certify validly.
fn multiplier_families(rng: &mut Rng, net: &Net) -> Vec<HullMultipliers> {
    let (h, n) = (net.h, net.n);
    let random = |rng: &mut Rng, a: (f64, f64), g: (f64, f64)| HullMultipliers {
        alpha: (0..h).map(|_| rng.uniform(a.0, a.1)).collect(),
        g: (0..h * n).map(|_| rng.uniform(g.0, g.1)).collect(),
    };
    let proposer_shaped = HullMultipliers {
        alpha: (0..h).map(|_| rng.unit()).collect(),
        g: net.w1.iter().map(|&w| rng.unit() * w).collect(),
    };
    let huge = HullMultipliers {
        alpha: vec![1.0; h],
        g: (0..h * n)
            .map(|_| if rng.below(2) == 0 { 1e150 } else { -1e150 })
            .collect(),
    };
    vec![
        random(&mut *rng, (-0.5, 1.5), (-3.0, 3.0)),
        random(&mut *rng, (0.0, 1.0), (-0.1, 0.1)),
        random(&mut *rng, (0.0, 1.0), (-1e-300, 1e-300)),
        HullMultipliers {
            alpha: vec![f64::NAN; h],
            g: net.w1.clone(),
        },
        HullMultipliers {
            alpha: vec![1.0; h],
            g: vec![0.0; h * n],
        },
        proposer_shaped,
        huge,
    ]
}

#[test]
fn hull_closed_form_valid_for_arbitrary_multipliers() {
    let mut rng = Rng::new(0x5eed_4c04);
    let mut samples_total = 0usize;
    let mut exact_checks = 0usize;
    let mut certified_values = 0usize;
    for trial in 0..60usize {
        let n = 1 + trial % 3;
        let h = 1 + trial % 5;
        let net = Net::random(&mut rng, n, h);
        let problem = net.problem();
        let mut certs: Vec<f64> = multiplier_families(&mut rng, &net)
            .iter()
            .filter_map(|m| certify_hull_lower_bound(&problem, m))
            .collect();
        let report = hull_root_bound(&problem, Instant::now() + Duration::from_secs(20));
        certs.extend(report.certified);
        assert!(
            report.certified.is_some(),
            "trial {trial}: the triangle multipliers must always certify"
        );
        certified_values += certs.len();
        let best = certs.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut argmin = net.lower.clone();
        let mut min_seen = f64::INFINITY;
        for _ in 0..1_700 {
            let x = net.sample(&mut rng);
            let v = net.margin_f64(&x);
            assert!(
                best <= v + 1e-9 * (1.0 + v.abs()),
                "trial {trial}: certified {best} exceeds the sampled margin {v} at {x:?}"
            );
            if v < min_seen {
                min_seen = v;
                argmin = x;
            }
            samples_total += 1;
        }
        let mut exact_points = net.vertices();
        exact_points.push(argmin);
        for x in &exact_points {
            let exact = net.margin_exact(x);
            for &c in &certs {
                let c_exact = BigRational::from_float(c).expect("certified values are finite");
                assert!(
                    c_exact <= exact,
                    "trial {trial}: certified {c} exceeds the EXACT margin at {x:?}"
                );
                exact_checks += 1;
            }
        }
    }
    assert!(samples_total >= 100_000, "only {samples_total} samples");
    assert!(exact_checks > 0 && certified_values > 0);
}

#[test]
fn hull_majorant_offset_is_an_exact_rational_upper_bound() {
    let mut rng = Rng::new(0x0ff5_e7c4);
    let q = |v: f64| BigRational::from_float(v).expect("finite");
    for _ in 0..500 {
        let n = 1 + rng.below(6);
        let w: Vec<f64> = (0..n).map(|_| rng.uniform(-3.0, 3.0)).collect();
        let g: Vec<f64> = (0..n).map(|_| rng.uniform(-3.0, 3.0)).collect();
        let b = rng.uniform(-2.0, 2.0);
        let lower: Vec<f64> = (0..n).map(|_| rng.uniform(-2.0, 0.0)).collect();
        let upper: Vec<f64> = lower.iter().map(|&l| l + rng.uniform(0.0, 2.0)).collect();
        let h = majorant_offset_upper(&w, b, &g, &lower, &upper).expect("finite inputs");
        // Exact: max(max_x -g.x, max_x (w-g).x + b), per-coordinate corners.
        let mut branch_zero = BigRational::zero();
        let mut branch_active = q(b);
        for i in 0..n {
            let (l, u) = (q(lower[i]), q(upper[i]));
            let ng = -q(g[i]);
            let delta = q(w[i]) - q(g[i]);
            branch_zero += std::cmp::max(&ng * &l, &ng * &u);
            branch_active += std::cmp::max(&delta * &l, &delta * &u);
        }
        let exact = std::cmp::max(branch_zero, branch_active);
        assert!(
            q(h) >= exact,
            "rounded-up offset {h} is below the exact offset"
        );
        let slack = q(h) - &exact;
        assert!(
            slack <= q(1e-12),
            "offset slack {slack} is far beyond outward rounding"
        );
    }
}

/// `m(x) = constant - relu(x1 + x2 - 1) + 0.9 relu(x1) + 0.1 relu(x2)` on
/// `[0,1]^2`. The concave envelope of `relu(x1 + x2 - 1)` there is
/// `min(x1, x2)`, so the ideal-hull minimum is exactly `constant`, while the
/// triangle relaxation only reaches `constant - 0.4`.
fn and_gadget(constant: f64) -> HullMarginProblem {
    HullMarginProblem::new(
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        vec![-1.0, 0.0, 0.0],
        2,
        &[-1.0, 0.9, 0.1, 0.0, 0.0, 0.0],
        &[constant, 0.0],
        2,
        &MARGIN,
        0.0,
        vec![0.0, 0.0],
        vec![1.0, 1.0],
    )
    .expect("gadget is admitted")
}

#[test]
fn hull_lp_recovers_the_ideal_envelope_on_the_and_gadget() {
    let problem = and_gadget(0.05);
    // The known-optimal multipliers certify on their own (no LP involved).
    let mut optimal = HullMultipliers::zeros(&problem);
    optimal.alpha = vec![0.0, 1.0, 1.0];
    optimal.g[0] = 0.9;
    optimal.g[1] = 0.1;
    let direct = certify_hull_lower_bound(&problem, &optimal).expect("finite");
    assert!(
        direct > 0.0499 && direct <= 0.05,
        "direct certificate {direct}"
    );

    let report = hull_root_bound(&problem, Instant::now() + Duration::from_secs(20));
    assert_eq!(report.lp_status, HullLpStatus::Solved, "{report:?}");
    assert_eq!(report.route, Some(HullBoundRoute::Lp), "{report:?}");
    assert_eq!((report.neg_unstable, report.pos_unstable), (1, 0));
    let certified = report.certified.expect("certified");
    assert!(
        certified > 0.0499 && certified <= 0.05,
        "LP-proposed certificate {certified} is not the ideal hull value"
    );
    assert!(report.verified());
    let lp = report.lp_value.expect("telemetry LP value");
    assert!(
        (lp - certified).abs() <= 1e-9,
        "dual-mapping gap: lp {lp} vs certified {certified}"
    );

    // The triangle relaxation alone does NOT verify this row.
    let bounds = preactivation_bounds(&problem);
    let roles = classify(&problem, &bounds);
    let triangle = triangle_multipliers(&problem, &roles, &bounds);
    let tri = certify_hull_lower_bound(&problem, &triangle).expect("finite");
    assert!((tri + 0.35).abs() < 1e-9, "triangle value {tri}");
}

#[test]
fn hull_does_not_verify_a_false_bound_on_the_and_gadget() {
    // True minimum is -0.05 (attained at (0,0) and (1,1)): a SAT row.
    let problem = and_gadget(-0.05);
    let report = hull_root_bound(&problem, Instant::now() + Duration::from_secs(20));
    assert!(!report.verified(), "{report:?}");
    let certified = report.certified.expect("certified");
    assert!(
        certified <= -0.05,
        "certified {certified} exceeds the true minimum -0.05"
    );
}

#[test]
fn hull_subgradient_fallback_certifies_only_valid_bounds() {
    let problem = and_gadget(0.05);
    let bounds = preactivation_bounds(&problem);
    let roles = classify(&problem, &bounds);
    let start = triangle_multipliers(&problem, &roles, &bounds);
    let baseline = certify_hull_lower_bound(&problem, &start);
    let (improved, iterations) = subgradient_ascent(
        &problem,
        &roles,
        &bounds,
        &start,
        baseline,
        Instant::now() + Duration::from_secs(20),
        HULL_SUBGRADIENT_MAX_ITERS,
    );
    assert!(iterations > 0);
    let improved = improved.expect("the ascent must beat the triangle start");
    assert!(
        improved > baseline.expect("baseline certifies"),
        "no improvement over the triangle"
    );
    assert!(
        improved <= 0.05,
        "subgradient certificate {improved} exceeds the optimum"
    );
}

#[test]
fn hull_lp_value_matches_certificate_on_random_mid_size_nets() {
    let mut rng = Rng::new(0xa11c_e5ed);
    let mut lp_checked = 0usize;
    for trial in 0..4 {
        let net = Net::random(&mut rng, 8, 24);
        let problem = net.problem();
        let report = hull_root_bound(&problem, Instant::now() + Duration::from_secs(60));
        let certified = report.certified.expect("certified");
        if report.unstable > 0 {
            lp_checked += 1;
            assert_eq!(
                report.lp_status,
                HullLpStatus::Solved,
                "trial {trial}: {report:?}"
            );
            let lp = report.lp_value.expect("telemetry LP value");
            assert!(
                (lp - certified).abs() <= 1e-6 * (1.0 + lp.abs()),
                "trial {trial}: dual-mapping gap lp {lp} vs certified {certified}"
            );
        }
        for x in net.vertices() {
            let exact = net.margin_exact(&x);
            assert!(
                BigRational::from_float(certified).expect("finite") <= exact,
                "trial {trial}: certified {certified} exceeds an exact vertex margin"
            );
        }
        for _ in 0..2_000 {
            let x = net.sample(&mut rng);
            let v = net.margin_f64(&x);
            assert!(certified <= v + 1e-9 * (1.0 + v.abs()), "trial {trial}");
        }
    }
    // Fail closed: if no trial had an unstable neuron, the LP / dual-mapping
    // half of this test checked nothing.
    assert!(
        lp_checked > 0,
        "no trial had an unstable neuron — the LP/dual-mapping check is VOID"
    );
}

#[test]
fn hull_certifier_declines_malformed_multipliers() {
    let problem = and_gadget(0.05);
    let good = HullMultipliers::zeros(&problem);
    assert!(certify_hull_lower_bound(&problem, &good).is_some());
    let short_alpha = HullMultipliers {
        alpha: vec![0.0; 2],
        ..good.clone()
    };
    assert_eq!(certify_hull_lower_bound(&problem, &short_alpha), None);
    let short_g = HullMultipliers {
        g: vec![0.0; 5],
        ..good.clone()
    };
    assert_eq!(certify_hull_lower_bound(&problem, &short_g), None);
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad = good.clone();
        bad.g[0] = poison; // neuron 0 has c < 0: its g row is used
        assert_eq!(
            certify_hull_lower_bound(&problem, &bad),
            None,
            "a non-finite majorant slope must decline"
        );
    }
}

#[test]
fn hull_problem_rejects_invalid_inputs() {
    fn admit(
        lower: Vec<f64>,
        upper: Vec<f64>,
        w1: Vec<f64>,
        objective: &[(usize, f64)],
    ) -> Result<(), HullDecline> {
        HullMarginProblem::new(
            w1,
            vec![0.0],
            2,
            &[1.0, -1.0],
            &[0.0, 0.0],
            2,
            objective,
            0.0,
            lower,
            upper,
        )
        .map(|_| ())
    }
    assert_eq!(
        admit(vec![0.0, 0.0], vec![1.0, 1.0], vec![1.0, 1.0], &MARGIN),
        Ok(())
    );
    assert_eq!(
        admit(vec![0.0, 2.0], vec![1.0, 1.0], vec![1.0, 1.0], &MARGIN),
        Err(HullDecline::InvalidBox)
    );
    assert_eq!(
        admit(vec![0.0, 0.0], vec![1.0, 1.0], vec![f64::NAN, 1.0], &MARGIN),
        Err(HullDecline::NonFiniteParameter)
    );
    assert_eq!(
        admit(vec![0.0, 0.0], vec![1.0, 1.0], vec![1.0], &MARGIN),
        Err(HullDecline::ShapeMismatch)
    );
    assert_eq!(
        admit(vec![0.0, 0.0], vec![1.0, 1.0], vec![1.0, 1.0], &[(5, 1.0)]),
        Err(HullDecline::ObjectiveIndexOutOfRange)
    );
    assert_eq!(
        admit(vec![0.0, 0.0], vec![1.0, 1.0], vec![1.0, 1.0], &[]),
        Err(HullDecline::EmptyDimension)
    );
}

#[test]
fn hull_margin_coefficients_round_down() {
    let mut rng = Rng::new(0x00c0_ffee);
    let q = |v: f64| BigRational::from_float(v).expect("finite");
    for _ in 0..200 {
        let h = 1 + rng.below(8);
        let w2: Vec<f64> = (0..2 * h).map(|_| rng.uniform(-5.0, 5.0)).collect();
        let b2 = [rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0)];
        let objective = [(0, rng.uniform(-2.0, 2.0)), (1, rng.uniform(-2.0, 2.0))];
        let constant = rng.uniform(-1.0, 1.0);
        let problem = HullMarginProblem::new(
            vec![1.0; h],
            vec![0.0; h],
            1,
            &w2,
            &b2,
            2,
            &objective,
            constant,
            vec![0.0],
            vec![1.0],
        )
        .expect("admitted");
        for j in 0..h {
            let exact = q(objective[0].1) * q(w2[j]) + q(objective[1].1) * q(w2[h + j]);
            assert!(q(problem.margin_coefficients()[j]) <= exact);
        }
        let exact_d = q(constant) + q(objective[0].1) * q(b2[0]) + q(objective[1].1) * q(b2[1]);
        assert!(q(problem.margin_constant()) <= exact_d);
    }
}
