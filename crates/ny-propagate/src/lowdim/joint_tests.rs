// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-lane stage 2 — unit tests of the joint quadratic bound's math.
//!
//! The load-bearing assertion is CONTAINMENT against values the function
//! attains (box vertices plus random interior points, evaluated here in plain
//! f64 by hand), for EVERY decomposition the bound might receive — including
//! the deliberately useless identity split (`sweeps = 0`). The NEGATIVE
//! CONTROL shows the eigen-residual charge is what makes that work: with the
//! identity split and the charge switched off, the same containment check
//! fails.

use super::super::SparseTerm;
use super::*;

/// Deterministic LCG in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// `phi(dx) = g.dx + dx'H dx` and `dx'H dx`, in plain f64.
fn phi(h: &[f64], g: &[f64], x: &[f64]) -> (f64, f64) {
    let d = x.len();
    let mut quad = 0.0;
    for j in 0..d {
        for k in 0..d {
            quad += h[j * d + k] * x[j] * x[k];
        }
    }
    let lin: f64 = g.iter().zip(x).map(|(a, b)| a * b).sum();
    (lin + quad, quad)
}

/// Every vertex of `|dx| <= xr` plus `count` random interior points.
fn points(rng: &mut Lcg, xr: &[f64], count: usize) -> Vec<Vec<f64>> {
    let d = xr.len();
    let mut out: Vec<Vec<f64>> = (0..(1usize << d.min(10)))
        .map(|v| {
            (0..d)
                .map(|j| if (v >> j) & 1 == 1 { xr[j] } else { -xr[j] })
                .collect()
        })
        .collect();
    for _ in 0..count {
        out.push(xr.iter().map(|r| r * rng.next()).collect());
    }
    out
}

/// The first containment failure of `dev` on `h, g` over the sample, if any.
fn escapes(h: &[f64], g: &[f64], dev: &Dev, sample: &[Vec<f64>]) -> Option<String> {
    for x in sample {
        let (f, q) = phi(h, g, x);
        let tol = 1e-12 * (1.0 + f.abs() + q.abs());
        let checks = [
            (f, dev.up + dev.ech, "phi <= up"),
            (-f, dev.dn + dev.ech, "-phi <= dn"),
            (q, dev.ur + dev.ech, "quad <= ur"),
            (-q, dev.lr + dev.ech, "-quad <= lr"),
        ];
        for (value, bound, what) in checks {
            if value > bound + tol {
                return Some(format!("{what}: {value} > {bound} at {x:?}"));
            }
        }
    }
    None
}

/// NEGATIVE CONTROL. `dx'Hdx = 2 x0 x1` on the unit box attains 2 at (1, 1)
/// and -2 at (1, -1). The identity split (`sweeps = 0`: lambda = diag(H) = 0,
/// U = I) explains NONE of it, so without the residual charge the bound
/// claims ~0 — and the containment check must catch that. With the charge
/// the same identity split is sound, and a converged Jacobi split is sound
/// AND tight.
#[test]
fn an_uncharged_eigen_residual_is_caught() {
    let h = [0.0, 1.0, 1.0, 0.0];
    let g = [0.0, 0.0];
    let xr = [1.0, 1.0];
    let mut rng = Lcg(1);
    let sample = points(&mut rng, &xr, 200);

    let uncharged = deviation(&h, &g, &xr, 0, false).expect("finite");
    assert_eq!(uncharged.ech, 0.0, "the control really disables the charge");
    assert!(
        uncharged.up < 2.0,
        "the control must MISS the attained max 2 (else it proves nothing): {uncharged:?}"
    );
    assert!(
        escapes(&h, &g, &uncharged, &sample).is_some(),
        "the containment check failed to catch an uncharged residual: {uncharged:?}"
    );

    let charged = deviation(&h, &g, &xr, 0, true).expect("finite");
    assert!(
        charged.ech >= 2.0,
        "the identity split's residual is all of H"
    );
    assert_eq!(escapes(&h, &g, &charged, &sample), None);

    let good = deviation(&h, &g, &xr, JACOBI_SWEEPS, true).expect("finite");
    assert_eq!(escapes(&h, &g, &good, &sample), None);
    assert!(
        good.up + good.ech <= 2.0 + 1e-9 && good.dn + good.ech <= 2.0 + 1e-9,
        "a converged split is exact on the vertices: {good:?}"
    );
}

/// Random indefinite quadratics, every dimension through the vertex limit and
/// past it (the O(d^2) branch alone), every decomposition quality.
#[test]
fn random_quadratics_are_contained_for_any_decomposition() {
    let mut rng = Lcg(77);
    for d in [1usize, 2, 3, 5, 6, 8, 10] {
        for trial in 0..6 {
            let mut h = vec![0.0; d * d];
            for j in 0..d {
                for k in j..d {
                    let v = 2.0 * rng.next();
                    h[j * d + k] = v;
                    h[k * d + j] = v;
                }
            }
            let g: Vec<f64> = (0..d).map(|_| 3.0 * rng.next()).collect();
            let xr: Vec<f64> = (0..d)
                .map(|j| {
                    if trial == 5 && j == 0 {
                        0.0
                    } else {
                        0.05 + rng.next().abs()
                    }
                })
                .collect();
            let sample = points(&mut rng, &xr, 800);
            for sweeps in [0usize, 1, JACOBI_SWEEPS] {
                let dev = deviation(&h, &g, &xr, sweeps, true).expect("finite");
                if let Some(why) = escapes(&h, &g, &dev, &sample) {
                    panic!("d={d} trial={trial} sweeps={sweeps}: {why} ({dev:?})");
                }
            }
        }
    }
}

/// `vertex_bounds` is exact on a convex quadratic: its `up` equals the
/// vertex maximum (to rounding), which the O(d^2) bound generally is not.
#[test]
fn the_vertex_bound_is_tight_on_a_convex_quadratic() {
    // H = [[2, 1], [1, 2]] (eigenvalues 1 and 3), g = (1, -1), xr = (1, 1).
    let h = [2.0, 1.0, 1.0, 2.0];
    let g = [1.0, -1.0];
    let xr = [1.0, 1.0];
    let exact = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
        .iter()
        .map(|&(a, b)| phi(&h, &g, &[a, b]).0)
        .fold(f64::NEG_INFINITY, f64::max);
    let dev = deviation(&h, &g, &xr, JACOBI_SWEEPS, true).expect("finite");
    assert!(dev.up + dev.ech >= exact);
    assert!(
        dev.up + dev.ech <= exact + 1e-9,
        "vertex max {exact} vs bound {}",
        dev.up + dev.ech
    );
}

/// An operand function inside `m.dx + p +- rad`, evaluated in plain f64.
struct Opnd {
    m: Vec<f64>,
    p: f64,
    rad: f64,
    phase: f64,
}

impl Opnd {
    fn random(rng: &mut Lcg, d: usize, rad: f64) -> Self {
        Self {
            m: (0..d).map(|_| 2.0 * rng.next()).collect(),
            p: rng.next(),
            rad,
            phase: rng.next(),
        }
    }
    /// Stays within `rad` of the midpoint affine (|sin| <= 1).
    fn eval(&self, dx: &[f64]) -> f64 {
        let lin: f64 = self.m.iter().zip(dx).map(|(a, b)| a * b).sum();
        let wiggle: f64 = dx.iter().sum::<f64>() * 7.0 + self.phase;
        lin + self.p + self.rad * wiggle.sin()
    }
    /// The joint's view of it: magnitudes rounded well up.
    fn affine(&self, xr: &[f64]) -> Affine {
        let d = xr.len();
        let mut m = [0.0f64; MD];
        m[..d].copy_from_slice(&self.m);
        let mag: f64 = self.m.iter().zip(xr).map(|(a, r)| a.abs() * r).sum::<f64>() + self.p.abs();
        let mag = mag * (1.0 + 1e-9) + 1e-300;
        Affine {
            m,
            p: self.p,
            rad: self.rad,
            mag,
            vmag: mag + self.rad * (1.0 + 1e-9),
        }
    }
}

/// `sum_t w_t a_t b_t + k` inside the joint interval AND between its planes,
/// with operand residuals, coefficient errors and a constant interval.
#[test]
fn site_bound_contains_the_sum_and_its_planes_hold_on_the_box() {
    let mut rng = Lcg(2026);
    for d in [2usize, 3, 6] {
        for trial in 0..8 {
            let xc: Vec<f64> = (0..d).map(|_| rng.next()).collect();
            let xr: Vec<f64> = (0..d).map(|_| 0.01 + 0.5 * rng.next().abs()).collect();
            let n_terms = 1 + trial % 6;
            let rad = [0.0, 1e-3, 0.05][trial % 3];
            let ops: Vec<(f64, f64, Opnd, Opnd)> = (0..n_terms)
                .map(|_| {
                    let w = rng.next();
                    let w_err = if trial % 2 == 1 { 1e-3 } else { 0.0 };
                    (
                        w,
                        w_err,
                        Opnd::random(&mut rng, d, rad),
                        Opnd::random(&mut rng, d, rad),
                    )
                })
                .collect();
            let terms: Vec<Term> = ops
                .iter()
                .map(|(w, w_err, a, b)| {
                    let (aa, bb) = (a.affine(&xr), b.affine(&xr));
                    Term {
                        w: *w,
                        w_err: *w_err,
                        z_mag: aa.vmag * bb.vmag * (1.0 + 1e-9),
                        a: aa,
                        b: bb,
                    }
                })
                .collect();
            let (k_lo, k_hi) = (-0.25, 0.125);
            let sb = site_bound(&terms, k_lo, k_hi, &xc, &xr, JACOBI_SWEEPS, true)
                .expect("finite site bound");
            for _ in 0..600 {
                let dx: Vec<f64> = xr.iter().map(|r| r * rng.next()).collect();
                let x: Vec<f64> = xc.iter().zip(&dx).map(|(c, e)| c + e).collect();
                let k = k_lo + (k_hi - k_lo) * 0.5 * (rng.next() + 1.0);
                let value: f64 = ops
                    .iter()
                    .map(|(w, w_err, a, b)| (w + w_err * rng.next()) * a.eval(&dx) * b.eval(&dx))
                    .sum::<f64>()
                    + k;
                let plane = |c: f64| sb.g[..d].iter().zip(&x).map(|(g, v)| g * v).sum::<f64>() + c;
                let tol = 1e-12 * (1.0 + value.abs());
                assert!(
                    sb.lo <= value + tol && value <= sb.hi + tol,
                    "d={d} trial={trial}: {value} outside [{}, {}]",
                    sb.lo,
                    sb.hi
                );
                assert!(
                    plane(sb.c_lo) <= value + tol && value <= plane(sb.c_hi) + tol,
                    "d={d} trial={trial}: {value} outside the planes [{}, {}]",
                    plane(sb.c_lo),
                    plane(sb.c_hi)
                );
            }
        }
    }
}

/// NEGATIVE CONTROL at the site level: `a = dx0`, `b = dx1` on the unit box
/// (`sum = dx0 dx1`, attaining 1 at (1, 1)). With the identity split and no
/// residual charge the published upper bound misses 1; charged, it covers it.
#[test]
fn a_site_bound_without_the_residual_charge_is_caught() {
    let unit = |j: usize| {
        let mut m = [0.0f64; MD];
        m[j] = 1.0;
        Affine {
            m,
            p: 0.0,
            rad: 0.0,
            mag: 1.0,
            vmag: 1.0,
        }
    };
    let terms = [Term {
        w: 1.0,
        w_err: 0.0,
        z_mag: 1.0,
        a: unit(0),
        b: unit(1),
    }];
    let (xc, xr) = ([0.0, 0.0], [1.0, 1.0]);
    let attained = 1.0; // dx0 * dx1 at (1, 1)
    let bad = site_bound(&terms, 0.0, 0.0, &xc, &xr, 0, false).expect("finite");
    assert!(
        bad.hi < attained,
        "the control must miss the attained value (else it proves nothing): {}",
        bad.hi
    );
    let charged = site_bound(&terms, 0.0, 0.0, &xc, &xr, 0, true).expect("finite");
    assert!(charged.hi >= attained && charged.lo <= -attained);
    let good = site_bound(&terms, 0.0, 0.0, &xc, &xr, JACOBI_SWEEPS, true).expect("finite");
    assert!(
        good.hi >= attained && good.hi <= attained + 1e-9,
        "{}",
        good.hi
    );
}

/// `reduce` keeps the pair with the smaller RIGOROUS radius, and the value
/// stays within that radius of the chosen midpoint affine.
#[test]
fn reduce_takes_the_tighter_pair_and_its_radius_holds() {
    // f(x) = x0^2 on [0, 1] x [-1, 1]; loose forms [-1, 2], tight chords.
    let (lo, hi) = ([0.0, -1.0], [1.0, 1.0]);
    let (mut xc, mut xr) = ([0.0f64; 2], [0.0f64; 2]);
    assert!(centre_radius(&lo, &hi, &mut xc, &mut xr));
    let loose_lo = [0.0, 0.0, -1.0];
    let loose_hi = [0.0, 0.0, 2.0];
    // x0^2 >= x0 - 0.25 (tangent at 1/2), x0^2 <= x0 (chord on [0, 1]).
    let tight_lo = [1.0, 0.0, -0.25];
    let tight_hi = [1.0, 0.0, 0.0];
    let only_loose = reduce(&[&loose_lo[..]], &[&loose_hi[..]], &xc, &xr, 1.0).expect("finite");
    let both = reduce(
        &[&loose_lo[..], &tight_lo[..]],
        &[&loose_hi[..], &tight_hi[..]],
        &xc,
        &xr,
        1.0,
    )
    .expect("finite");
    assert!(
        both.rad < only_loose.rad,
        "{} vs {}",
        both.rad,
        only_loose.rad
    );
    assert!(
        both.rad <= 0.125 + 1e-12,
        "the tight pair's half-gap is 1/8: {}",
        both.rad
    );
    let mut rng = Lcg(3);
    for _ in 0..500 {
        let x = [f64::midpoint(rng.next(), 1.0), rng.next()];
        let dx = [x[0] - xc[0], x[1] - xc[1]];
        let mid = both.m[0] * dx[0] + both.m[1] * dx[1] + both.p;
        let f = x[0] * x[0];
        assert!(
            (f - mid).abs() <= both.rad + 1e-15,
            "|{f} - {mid}| > {}",
            both.rad
        );
        assert!(mid.abs() <= both.mag + 1e-15);
    }
}

/// Only a width-1 reduction whose terms read DISTINCT elements of a product
/// is a site.
#[test]
fn only_a_scalar_reduction_of_distinct_product_elements_is_a_site() {
    let term = |src: u32, idx: u32| SparseTerm {
        src,
        idx,
        coef: 1.0,
        err: 0.0,
    };
    let sparse = |rows: Vec<Vec<SparseTerm>>| {
        let n = rows.len();
        Op::Sparse {
            terms: rows,
            k_lo: vec![0.0; n],
            k_hi: vec![0.0; n],
            fanin: 1,
        }
    };
    let base = vec![Op::Input, Op::Identity { src: 0 }, Op::Mul { a: 0, b: 1 }];
    let sizes = vec![3, 3, 3, 1];
    let mut ops = base.clone();
    ops.push(sparse(vec![vec![term(2, 0), term(2, 1), term(2, 2)]]));
    let sites = detect_sites(&ops, &sizes);
    assert_eq!(sites.len(), 1);
    assert_eq!(
        (sites[0].node, sites[0].mul, sites[0].a, sites[0].b),
        (3, 2, 0, 1)
    );

    let mut dup = base.clone();
    dup.push(sparse(vec![vec![term(2, 0), term(2, 0)]]));
    assert!(detect_sites(&dup, &sizes).is_empty(), "a repeated element");

    let mut not_mul = base.clone();
    not_mul.push(sparse(vec![vec![term(1, 0), term(1, 1)]]));
    assert!(detect_sites(&not_mul, &sizes).is_empty(), "not a product");

    let mut wide = base;
    wide.push(sparse(vec![vec![term(2, 0)], vec![term(2, 1)]]));
    assert!(
        detect_sites(&wide, &[3, 3, 3, 2]).is_empty(),
        "a width-2 reduction"
    );
}
