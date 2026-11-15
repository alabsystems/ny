// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness and search tests for #hull-bab (`single_hidden_hull_bab.rs`).
//!
//! The load-bearing property is that [`certify_hull_node_lower_bound`] is a
//! valid lower bound on the TRUE (real-arithmetic) margin minimum over the
//! node REGION `B ∩ {sigma_k z_k >= 0}` for ARBITRARY multipliers and
//! ARBITRARY `beta`. It is checked against dense float sampling of the region
//! (1e-9 relative tolerance for the sampler's own rounding) and, at every box
//! vertex and the best sample that lie in the region EXACTLY, against an
//! exact-rational margin with no tolerance.
//!
//! The search itself is checked against an EXACT oracle on 1-D and 2-D nets
//! (the minimum of a piecewise-linear function over a box is attained at a
//! vertex of the kink arrangement, which is enumerable there): it never
//! verifies a net whose minimum is negative, and it proves most nets whose
//! minimum is positive.

use super::*;
use crate::single_hidden_hull::{certify_hull_lower_bound, hull_root_bound};
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

const MARGIN: [(usize, f64); 2] = [(0, 1.0), (1, -1.0)];

/// A random single-hidden-layer net with two outputs and margin `Y0 - Y1`.
#[derive(Clone)]
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

fn q(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite")
}

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
            lower.push(l);
            upper.push(l + rng.uniform(0.1, 1.5));
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

    fn z_f64(&self, j: usize, x: &[f64]) -> f64 {
        let mut z = self.b1[j];
        for i in 0..self.n {
            z += self.w1[j * self.n + i] * x[i];
        }
        z
    }

    fn z_exact(&self, j: usize, x: &[f64]) -> BigRational {
        let mut z = q(self.b1[j]);
        for i in 0..self.n {
            z += q(self.w1[j * self.n + i]) * q(x[i]);
        }
        z
    }

    fn margin_f64(&self, x: &[f64]) -> f64 {
        let mut m = self.b2[0] - self.b2[1];
        for j in 0..self.h {
            m += (self.w2[j] - self.w2[self.h + j]) * self.z_f64(j, x).max(0.0);
        }
        m
    }

    fn margin_exact(&self, x: &[f64]) -> BigRational {
        let mut m = q(self.b2[0]) - q(self.b2[1]);
        for j in 0..self.h {
            let z = self.z_exact(j, x);
            if z > BigRational::zero() {
                m += (q(self.w2[j]) - q(self.w2[self.h + j])) * z;
            }
        }
        m
    }

    /// Float region membership with a guard band: `None` when a split
    /// pre-activation is within `1e-12` of its kink (ambiguous in float).
    fn in_region_f64(&self, splits: &[HullSplit], x: &[f64]) -> Option<bool> {
        for s in splits {
            let z = self.z_f64(s.neuron as usize, x);
            if z.abs() <= 1e-12 {
                return None;
            }
            if (z > 0.0) != s.active {
                return Some(false);
            }
        }
        Some(true)
    }

    fn in_region_exact(&self, splits: &[HullSplit], x: &[f64]) -> bool {
        splits.iter().all(|s| {
            let z = self.z_exact(s.neuron as usize, x);
            if s.active {
                z >= BigRational::zero()
            } else {
                z <= BigRational::zero()
            }
        })
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

    /// Uniform points, vertices, and points projected onto a random kink.
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

    /// EXACT minimum (up to float evaluation) for `n <= 2`: every vertex of
    /// the kink arrangement inside the box. Returns `(min, argmin)`.
    fn oracle_min(&self) -> (f64, Vec<f64>) {
        assert!(self.n <= 2, "the vertex oracle is for n <= 2");
        let mut points = self.vertices();
        let inside = |x: &[f64]| {
            (0..self.n).all(|i| x[i] >= self.lower[i] - 1e-12 && x[i] <= self.upper[i] + 1e-12)
        };
        let clamp = |x: Vec<f64>| -> Vec<f64> {
            (0..self.n)
                .map(|i| x[i].clamp(self.lower[i], self.upper[i]))
                .collect()
        };
        if self.n == 1 {
            for j in 0..self.h {
                let w = self.w1[j];
                if w != 0.0 {
                    let x = vec![-self.b1[j] / w];
                    if inside(&x) {
                        points.push(clamp(x));
                    }
                }
            }
        } else {
            for j in 0..self.h {
                let (a, b, c) = (self.w1[2 * j], self.w1[2 * j + 1], self.b1[j]);
                for x0 in [self.lower[0], self.upper[0]] {
                    if b != 0.0 {
                        let x = vec![x0, -(c + a * x0) / b];
                        if inside(&x) {
                            points.push(clamp(x));
                        }
                    }
                }
                for x1 in [self.lower[1], self.upper[1]] {
                    if a != 0.0 {
                        let x = vec![-(c + b * x1) / a, x1];
                        if inside(&x) {
                            points.push(clamp(x));
                        }
                    }
                }
                for k in (j + 1)..self.h {
                    let (d, e, f) = (self.w1[2 * k], self.w1[2 * k + 1], self.b1[k]);
                    let det = a * e - b * d;
                    if det.abs() > 1e-12 {
                        let x = vec![(-c * e + f * b) / det, (-a * f + d * c) / det];
                        if inside(&x) {
                            points.push(clamp(x));
                        }
                    }
                }
            }
        }
        let mut best = (f64::INFINITY, points[0].clone());
        for x in points {
            let v = self.margin_f64(&x);
            if v < best.0 {
                best = (v, x);
            }
        }
        best
    }
}

/// Random distinct splits (at most `max`), random phases.
fn random_splits(rng: &mut Rng, h: usize, max: usize) -> Vec<HullSplit> {
    let count = rng.below(max.min(h) + 1);
    let mut chosen: Vec<usize> = Vec::new();
    while chosen.len() < count {
        let j = rng.below(h);
        if !chosen.contains(&j) {
            chosen.push(j);
        }
    }
    chosen
        .into_iter()
        .map(|j| HullSplit {
            neuron: u32::try_from(j).expect("small"),
            active: rng.below(2) == 0,
        })
        .collect()
}

/// Multiplier families: none is "good"; all must certify validly.
fn multiplier_families(rng: &mut Rng, net: &Net) -> Vec<HullMultipliers> {
    let (h, n) = (net.h, net.n);
    let random = |rng: &mut Rng, a: (f64, f64), g: (f64, f64)| HullMultipliers {
        alpha: (0..h).map(|_| rng.uniform(a.0, a.1)).collect(),
        g: (0..h * n).map(|_| rng.uniform(g.0, g.1)).collect(),
    };
    vec![
        random(&mut *rng, (-0.5, 1.5), (-3.0, 3.0)),
        random(&mut *rng, (0.0, 1.0), (-0.1, 0.1)),
        HullMultipliers {
            alpha: vec![f64::NAN; h],
            g: net.w1.clone(),
        },
        HullMultipliers {
            alpha: (0..h).map(|_| rng.unit()).collect(),
            g: net.w1.iter().map(|&w| rng.unit() * w).collect(),
        },
        HullMultipliers {
            alpha: vec![1.0; h],
            g: vec![0.0; h * n],
        },
    ]
}

/// `beta` families: zero, positive, mixed-sign (negatives must be clamped,
/// never used), NaN (treated as 0), subnormal, and large-but-finite.
fn beta_families(rng: &mut Rng, k: usize) -> Vec<Vec<f64>> {
    vec![
        vec![0.0; k],
        (0..k).map(|_| rng.uniform(0.0, 3.0)).collect(),
        (0..k).map(|_| rng.uniform(-3.0, 3.0)).collect(),
        vec![f64::NAN; k],
        vec![1e-310; k],
        (0..k).map(|_| rng.uniform(0.0, 1e6)).collect(),
    ]
}

#[test]
fn hull_node_certifier_valid_for_arbitrary_multipliers_and_betas() {
    let mut rng = Rng::new(0xbab0_5eed);
    let mut region_samples = 0usize;
    let mut exact_checks = 0usize;
    let mut certified_values = 0usize;
    let mut lp_node_values = 0usize;
    for trial in 0..60usize {
        let n = 1 + trial % 3;
        let h = 2 + trial % 5;
        let net = Net::random(&mut rng, n, h);
        let problem = net.problem();
        let splits = random_splits(&mut rng, h, 3);
        let mut certs: Vec<f64> = Vec::new();
        for m in multiplier_families(&mut rng, &net) {
            for beta in beta_families(&mut rng, splits.len()) {
                certs.extend(certify_hull_node_lower_bound(&problem, &splits, &m, &beta));
            }
        }
        // The node proposer's own certificate must be contained too.
        let bounds = preactivation_bounds(&problem);
        let roles = classify(&problem, &bounds);
        let outcome = bound_node_counted(
            &problem,
            &roles,
            &bounds,
            &splits,
            Instant::now() + Duration::from_secs(20),
            None,
            None,
        );
        if let Some(v) = outcome.certified {
            certs.push(v);
            lp_node_values += usize::from(outcome.route == Some(HullBoundRoute::Lp));
        }
        certified_values += certs.len();
        let best = certs.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut argmin: Option<Vec<f64>> = None;
        let mut min_seen = f64::INFINITY;
        for _ in 0..3_000 {
            let x = net.sample(&mut rng);
            if net.in_region_f64(&splits, &x) != Some(true) {
                continue;
            }
            let v = net.margin_f64(&x);
            assert!(
                best <= v + 1e-9 * (1.0 + v.abs()),
                "trial {trial}: node certificate {best} exceeds the sampled margin {v} at \
                 {x:?} inside the region of {splits:?}"
            );
            region_samples += 1;
            if v < min_seen {
                min_seen = v;
                argmin = Some(x);
            }
        }
        let mut exact_points = net.vertices();
        exact_points.extend(argmin);
        for x in exact_points
            .iter()
            .filter(|x| net.in_region_exact(&splits, x))
        {
            let exact = net.margin_exact(x);
            for &c in &certs {
                assert!(
                    q(c) <= exact,
                    "trial {trial}: node certificate {c} exceeds the EXACT margin at {x:?}"
                );
                exact_checks += 1;
            }
        }
    }
    // Fail closed: a sampler that never lands in a region checks nothing.
    assert!(
        region_samples >= 20_000,
        "only {region_samples} region samples"
    );
    assert!(exact_checks > 0, "no exact-rational check ran");
    assert!(certified_values > 0 && lp_node_values > 0);
}

#[test]
fn hull_node_certifier_with_no_splits_is_bit_identical_to_the_root_certifier() {
    let mut rng = Rng::new(0x0001_d3a1);
    let mut compared = 0usize;
    for trial in 0..40usize {
        let net = Net::random(&mut rng, 1 + trial % 4, 1 + trial % 6);
        let problem = net.problem();
        for m in multiplier_families(&mut rng, &net) {
            let root = certify_hull_lower_bound(&problem, &m);
            let node = certify_hull_node_lower_bound(&problem, &[], &m, &[]);
            assert_eq!(
                root.map(f64::to_bits),
                node.map(f64::to_bits),
                "trial {trial}"
            );
            compared += usize::from(root.is_some());
        }
    }
    assert!(compared > 0);
}

#[test]
fn hull_node_certifier_declines_malformed_inputs() {
    let net = abs_net(0.25);
    let problem = net.problem();
    let m = HullMultipliers::zeros(&problem);
    let split = |neuron: u32, active: bool| HullSplit { neuron, active };
    assert!(certify_hull_node_lower_bound(&problem, &[split(0, true)], &m, &[0.0]).is_some());
    // beta length must match the split list.
    assert_eq!(
        certify_hull_node_lower_bound(&problem, &[split(0, true)], &m, &[]),
        None
    );
    // A duplicate neuron (even with agreeing phases) or an out-of-range one.
    assert_eq!(
        certify_hull_node_lower_bound(&problem, &[split(0, true), split(0, true)], &m, &[0.0, 0.0]),
        None
    );
    assert_eq!(
        certify_hull_node_lower_bound(&problem, &[split(9, true)], &m, &[0.0]),
        None
    );
    // An infinite beta declines; NaN and negative beta are clamped to 0.
    for poison in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            certify_hull_node_lower_bound(&problem, &[split(0, true)], &m, &[poison]),
            None
        );
    }
    let zero = certify_hull_node_lower_bound(&problem, &[split(0, true)], &m, &[0.0]);
    for clamped in [f64::NAN, -5.0, -1e300] {
        assert_eq!(
            certify_hull_node_lower_bound(&problem, &[split(0, true)], &m, &[clamped]),
            zero,
            "beta {clamped} must be treated as 0"
        );
    }
    // Shape mismatches.
    let short = HullMultipliers {
        alpha: vec![0.0; 1],
        ..m.clone()
    };
    assert_eq!(
        certify_hull_node_lower_bound(&problem, &[], &short, &[]),
        None
    );
    let mut bad_g = m;
    bad_g.g[0] = f64::NAN; // neuron 0 has c < 0: its g row is read when free
    assert_eq!(
        certify_hull_node_lower_bound(&problem, &[], &bad_g, &[]),
        None
    );
}

/// `m(x) = C - relu(x) + relu(2x) - relu(-x) + relu(-2x) = C + |x|` on
/// `[-1, 1]`. Neurons 0 and 2 are concave (`c = -1`), 1 and 3 convex. The
/// ideal hull of each concave neuron over the box is its chord, so the ROOT
/// hull value is `C - 1`; fixing both concave neurons makes every leaf exact
/// with minimum `C` (at `x = 0`). `C = 0.25`: the root fails, the tree
/// proves. `C < 0`: SAT at `x = 0`.
fn abs_net(constant: f64) -> Net {
    Net {
        n: 1,
        h: 4,
        w1: vec![1.0, 2.0, -1.0, -2.0],
        b1: vec![0.0; 4],
        w2: vec![-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        b2: vec![constant, 0.0],
        lower: vec![-1.0],
        upper: vec![1.0],
    }
}

fn far() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

fn assert_report_invariants(report: &HullBabReport) {
    if report.verified() {
        assert_eq!(report.stop, HullBabStop::Proved);
        assert_eq!(report.open, 0);
        assert_eq!(report.worst, None);
    }
    if report.stop == HullBabStop::Proved {
        assert_eq!(report.open, 0, "Proved with open nodes: {report:?}");
    } else {
        assert!(!report.verified(), "{report:?}");
    }
    assert!(report.closed <= report.nodes);
}

#[test]
fn hull_bab_proves_the_abs_gadget_where_the_root_fails() {
    let problem = abs_net(0.25).problem();
    let root = hull_root_bound(&problem, far());
    assert!(
        !root.verified(),
        "the root must NOT prove this row: {root:?}"
    );
    let root_cert = root.certified.expect("root certifies");
    assert!(
        (-0.76..=-0.74).contains(&root_cert),
        "root hull value {root_cert} is not C - 1"
    );
    for workers in [1, 3] {
        let report = hull_bab(
            &problem,
            far(),
            HullBabOptions {
                workers,
                ..HullBabOptions::default()
            },
        );
        assert_report_invariants(&report);
        assert!(report.verified(), "workers={workers}: {report:?}");
        assert_eq!((report.neg_unstable, report.pos_unstable), (2, 2));
        assert!(report.nodes >= 3, "a proof needs at least one split");
        assert!(
            report.nodes <= 31,
            "tree far larger than 2 concave neurons need"
        );
        assert!(report.root_certified.is_some_and(|v| v <= 0.0));
    }
}

#[test]
fn hull_bab_never_verifies_the_sat_abs_gadget() {
    for constant in [-0.25, -1e-3] {
        let problem = abs_net(constant).problem();
        let report = hull_bab(
            &problem,
            far(),
            HullBabOptions {
                workers: 2,
                ..HullBabOptions::default()
            },
        );
        assert_report_invariants(&report);
        assert!(!report.verified(), "C={constant}: {report:?}");
        assert_eq!(report.stop, HullBabStop::Exhausted, "{report:?}");
        let worst = report.worst.expect("open frontier");
        assert!(
            worst <= constant,
            "C={constant}: worst open bound {worst} exceeds the true minimum"
        );
    }
}

#[test]
fn hull_node_bound_certifies_an_empty_split_region() {
    // m(x) = -1 - relu(x + 0.5) - relu(-x + 0.5) on [-1, 1]: negative
    // EVERYWHERE (a SAT row). Neuron 0 inactive needs x <= -0.5, neuron 1
    // inactive needs x >= 0.5: the node region is EMPTY. With beta = 0 the
    // node minorant is the constant -1 (not a proof); only the elastic LP's
    // halfspace duals can price the emptiness into a positive bound.
    let net = Net {
        n: 1,
        h: 2,
        w1: vec![1.0, -1.0],
        b1: vec![0.5, 0.5],
        w2: vec![-1.0, -1.0, 0.0, 0.0],
        b2: vec![-1.0, 0.0],
        lower: vec![-1.0],
        upper: vec![1.0],
    };
    let problem = net.problem();
    let bounds = preactivation_bounds(&problem);
    let roles = classify(&problem, &bounds);
    let splits = [
        HullSplit {
            neuron: 0,
            active: false,
        },
        HullSplit {
            neuron: 1,
            active: false,
        },
    ];
    let no_beta = certify_hull_node_lower_bound(
        &problem,
        &splits,
        &HullMultipliers::zeros(&problem),
        &[0.0, 0.0],
    )
    .expect("finite");
    assert!(
        no_beta <= -0.99,
        "beta = 0 must not prove the node: {no_beta}"
    );
    let outcome = bound_node_counted(&problem, &roles, &bounds, &splits, far(), None, None);
    assert_eq!(outcome.lp_status, HullLpStatus::Solved, "{outcome:?}");
    assert_eq!(outcome.route, Some(HullBoundRoute::Lp), "{outcome:?}");
    let certified = outcome.certified.expect("certified");
    assert!(
        certified > 0.0,
        "an empty region must certify positive, got {certified}"
    );
    // And no region point exists to contradict it (exact check on a grid).
    for step in 0..=200_i32 {
        let x = [-1.0 + f64::from(step) * 0.01];
        assert!(!net.in_region_exact(&splits, &x));
    }
    // The same net is SAT on the box: the tree must never verify it.
    let report = hull_bab(&problem, far(), HullBabOptions::default());
    assert_report_invariants(&report);
    assert!(!report.verified(), "{report:?}");
}

#[test]
fn hull_bab_never_verifies_a_negative_minimum_on_random_nets() {
    // EXACT vertex oracle on 1-D/2-D nets; the constant is shifted so the
    // true minimum is exactly +delta or -delta (up to float evaluation).
    let mut rng = Rng::new(0x5a7_c0de);
    let (mut sat_cases, mut unsat_cases, mut unsat_verified) = (0usize, 0usize, 0usize);
    let mut frontier_checks = 0usize;
    for trial in 0..32usize {
        let n = 1 + trial % 2;
        let h = 2 + trial % 5;
        let mut net = Net::random(&mut rng, n, h);
        let (raw_min, _) = net.oracle_min();
        let delta = if trial % 2 == 0 { 0.05 } else { -0.05 };
        net.b2[0] += delta - raw_min;
        let (true_min, argmin) = net.oracle_min();
        let problem = net.problem();
        let report = hull_bab(
            &problem,
            Instant::now() + Duration::from_secs(20),
            HullBabOptions {
                workers: 1 + trial % 3,
                ..HullBabOptions::default()
            },
        );
        assert_report_invariants(&report);
        // The FRONTIER invariant, checked EXACTLY (no tolerance): the open and
        // closed leaves cover the box, every open effective bound is `<= 0`,
        // and every bound is `<=` its region's minimum, so neither the root
        // certificate nor the lowest open bound can exceed the true margin at
        // ANY box point — in particular at the oracle's (clamped) argmin. This
        // holds for every stop, verified or not, so it is exercised on every
        // SAT net (whose frontier is never empty).
        let at_argmin = net.margin_exact(&argmin);
        if let Some(root) = report.root_certified {
            assert!(
                q(root) <= at_argmin,
                "trial {trial}: root certificate {root} exceeds the exact margin at {argmin:?}"
            );
        }
        if let Some(worst) = report.worst.filter(|w| w.is_finite()) {
            assert!(
                q(worst) <= at_argmin,
                "trial {trial}: lowest open bound {worst} exceeds the exact margin at \
                 {argmin:?}: {report:?}"
            );
            frontier_checks += 1;
        }
        if true_min < -1e-9 {
            sat_cases += 1;
            // A genuine counterexample, confirmed in exact arithmetic.
            assert!(
                net.margin_exact(&argmin) < BigRational::zero(),
                "trial {trial}: oracle argmin is not an exact counterexample"
            );
            assert!(
                !report.verified(),
                "trial {trial}: VERIFIED a net with a counterexample at {argmin:?} \
                 (true min {true_min}): {report:?}"
            );
        } else if true_min > 1e-9 {
            unsat_cases += 1;
            unsat_verified += usize::from(report.verified());
        }
        if report.verified() {
            assert!(
                true_min > 0.0,
                "trial {trial}: verified but the exact oracle minimum is {true_min}"
            );
        }
    }
    eprintln!(
        "HULL_BAB_ORACLE sat_cases={sat_cases} unsat_cases={unsat_cases} \
         unsat_verified={unsat_verified} frontier_checks={frontier_checks}"
    );
    assert!(
        sat_cases > 0 && unsat_cases > 0,
        "both branches must be exercised"
    );
    // Fail closed: the frontier invariant above must actually have run.
    assert!(frontier_checks > 0, "the frontier invariant never ran");
    // Leaves are exact LPs, so the tree is complete on nets this small; a
    // tree that proves fewer than half of them is not the designed search.
    assert!(
        unsat_verified * 2 >= unsat_cases && unsat_verified > 0,
        "only {unsat_verified}/{unsat_cases} positive-minimum nets proved"
    );
}

#[test]
fn hull_bab_stops_undecided_at_the_deadline_and_the_node_cap() {
    let problem = abs_net(0.25).problem();
    // A deadline already in the past: the root LP is skipped, the triangle
    // root cannot prove it, and the search stops before branching.
    let past = hull_bab(&problem, Instant::now(), HullBabOptions::default());
    assert_report_invariants(&past);
    assert!(!past.verified(), "{past:?}");
    assert_eq!(past.stop, HullBabStop::Deadline);
    assert_eq!(past.nodes, 1);
    assert_eq!(past.root_lp_status, HullLpStatus::Skipped);

    let capped = hull_bab(
        &problem,
        far(),
        HullBabOptions {
            max_nodes: 1,
            workers: 1,
        },
    );
    assert_report_invariants(&capped);
    assert!(!capped.verified(), "{capped:?}");
    assert_eq!(capped.stop, HullBabStop::NodeCap);
    assert_eq!(capped.open, 1);
}

#[test]
fn hull_bab_verifies_at_the_root_without_branching_when_the_root_proves() {
    // The and-gadget of the root tests: the ideal hull is exact at the root.
    let problem = HullMarginProblem::new(
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        vec![-1.0, 0.0, 0.0],
        2,
        &[-1.0, 0.9, 0.1, 0.0, 0.0, 0.0],
        &[0.05, 0.0],
        2,
        &MARGIN,
        0.0,
        vec![0.0, 0.0],
        vec![1.0, 1.0],
    )
    .expect("gadget is admitted");
    let report = hull_bab(&problem, far(), HullBabOptions::default());
    assert_report_invariants(&report);
    assert!(report.verified(), "{report:?}");
    assert_eq!((report.nodes, report.closed, report.max_depth), (1, 1, 0));
}
