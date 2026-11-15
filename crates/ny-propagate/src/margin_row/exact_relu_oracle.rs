// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! review(eng14): a TEST-ONLY exact oracle for one ReLU head layer over a
//! small input box,
//!
//! ```text
//! m(x) = c + sum_j q_j * ReLU(a_j . x + b_j),   lo <= x <= hi,  x in Q^d,
//! ```
//!
//! with every quantity an exact rational (binary64 values are read exactly).
//! On every cell of the arrangement of the hyperplanes `a_j . x + b_j = 0`
//! inside the box, `m` is affine, so its minimum over the box is attained at a
//! cell vertex; every cell vertex is the unique solution of `d` linearly
//! independent equations drawn from those hyperplanes and the box facets.
//! [`vertices`] enumerates every such solution inside the box, so the
//! smallest exact value of `m` over them is the TRUE minimum: the number a
//! certified lower bound may never exceed.

use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// Inputs the enumeration admits (`C(h + 2d, d)` exact solves).
const MAX_DIM: usize = 4;

/// The exact rational value of a finite binary64.
pub(crate) fn rat(x: f64) -> BigRational {
    BigRational::from_float(x).expect("a finite binary64")
}

/// `max(v, 0)`, exactly.
pub(crate) fn relu(v: BigRational) -> BigRational {
    if v.is_negative() {
        BigRational::zero()
    } else {
        v
    }
}

/// One ReLU preactivation `a . x + b`.
#[derive(Clone, Debug)]
pub(crate) struct Line {
    pub(crate) a: Vec<BigRational>,
    pub(crate) b: BigRational,
}

impl Line {
    /// The preactivation from binary64 coefficients, read exactly.
    pub(crate) fn of(a: &[f64], b: f64) -> Self {
        Self {
            a: a.iter().map(|&v| rat(v)).collect(),
            b: rat(b),
        }
    }

    /// `a . x + b` at `x`, exactly.
    pub(crate) fn at(&self, x: &[BigRational]) -> BigRational {
        assert_eq!(self.a.len(), x.len());
        self.a
            .iter()
            .zip(x)
            .fold(self.b.clone(), |sum, (a, x)| sum + a * x)
    }
}

/// The unique solution of `m x = r`, by exact elimination (`None` when `m`
/// is singular).
fn solve(mut m: Vec<Vec<BigRational>>, mut r: Vec<BigRational>) -> Option<Vec<BigRational>> {
    let n = r.len();
    for col in 0..n {
        let pivot = (col..n).find(|&i| !m[i][col].is_zero())?;
        m.swap(col, pivot);
        r.swap(col, pivot);
        let row = m[col].clone();
        let rhs = r[col].clone();
        for i in 0..n {
            if i == col || m[i][col].is_zero() {
                continue;
            }
            let f = &m[i][col] / &row[col];
            for (k, v) in row.iter().enumerate().skip(col) {
                m[i][k] -= &f * v;
            }
            r[i] -= &f * &rhs;
        }
    }
    Some((0..n).map(|i| &r[i] / &m[i][i]).collect())
}

/// Every vertex of the arrangement of `lines` inside the box `[lo, hi]`
/// (duplicates are harmless; the box corners are always among them).
pub(crate) fn vertices(
    lo: &[BigRational],
    hi: &[BigRational],
    lines: &[Line],
) -> Vec<Vec<BigRational>> {
    let d = lo.len();
    assert!((1..=MAX_DIM).contains(&d) && hi.len() == d, "a small box");
    assert!((0..d).all(|k| lo[k] <= hi[k]), "an ordered box");
    assert!(lines.iter().all(|l| l.a.len() == d), "lines over the box");
    // Every equation `a . x = rhs`: the head hyperplanes, then the facets.
    let mut planes: Vec<(Vec<BigRational>, BigRational)> =
        lines.iter().map(|l| (l.a.clone(), -l.b.clone())).collect();
    for k in 0..d {
        for bound in [&lo[k], &hi[k]] {
            let mut a = vec![BigRational::zero(); d];
            a[k] = rat(1.0);
            planes.push((a, bound.clone()));
        }
    }
    let inside = |x: &[BigRational]| (0..d).all(|k| lo[k] <= x[k] && x[k] <= hi[k]);
    let mut out = Vec::new();
    let mut pick = Vec::with_capacity(d);
    fn walk(
        planes: &[(Vec<BigRational>, BigRational)],
        d: usize,
        from: usize,
        pick: &mut Vec<usize>,
        visit: &mut dyn FnMut(&[usize]),
    ) {
        if pick.len() == d {
            visit(pick);
            return;
        }
        for i in from..planes.len() {
            pick.push(i);
            walk(planes, d, i + 1, pick, visit);
            pick.pop();
        }
    }
    walk(&planes, d, 0, &mut pick, &mut |chosen| {
        let m = chosen.iter().map(|&i| planes[i].0.clone()).collect();
        let r = chosen.iter().map(|&i| planes[i].1.clone()).collect();
        if let Some(x) = solve(m, r) {
            if inside(&x) {
                out.push(x);
            }
        }
    });
    out
}

/// The exact value of `c + sum_j q_j * ReLU(lines_j(x))`.
pub(crate) fn value(
    c: &BigRational,
    q: &[BigRational],
    lines: &[Line],
    x: &[BigRational],
) -> BigRational {
    assert_eq!(q.len(), lines.len());
    q.iter()
        .zip(lines)
        .fold(c.clone(), |sum, (q, l)| sum + q * relu(l.at(x)))
}

/// The exact minimum over the box of `c + sum_j q_j * ReLU(lines_j(x))`.
pub(crate) fn min_over_box(
    lo: &[BigRational],
    hi: &[BigRational],
    c: &BigRational,
    q: &[BigRational],
    lines: &[Line],
) -> BigRational {
    vertices(lo, hi, lines)
        .iter()
        .map(|x| value(c, q, lines, x))
        .min()
        .expect("a box has corners")
}

/// The oracle's own control: an interior line/line crossing that no corner
/// or edge reaches is found; the minimum is attained (it is a value of `m`);
/// and no point of a dense dyadic grid lies below it.
#[test]
fn exact_relu_oracle_is_attained_and_below_a_dense_grid() {
    let lo = [rat(-1.0), rat(-1.0)];
    let hi = [rat(1.0), rat(1.0)];
    // |x0 - 1/4| + |x1 + 3/8|: minimum 0 ONLY at the interior crossing
    // (1/4, -3/8).
    let lines = vec![
        Line::of(&[1.0, 0.0], -0.25),
        Line::of(&[-1.0, 0.0], 0.25),
        Line::of(&[0.0, 1.0], 0.375),
        Line::of(&[0.0, -1.0], -0.375),
    ];
    let q = vec![rat(1.0); 4];
    assert_eq!(min_over_box(&lo, &hi, &rat(0.0), &q, &lines), rat(0.0));
    // A three-dimensional point minimum, |x0| + |x1 - x0| + |x2 + x1|,
    // attained only at the origin where three head planes meet.
    let lo3 = [rat(-1.0), rat(-1.0), rat(-1.0)];
    let hi3 = [rat(1.0), rat(1.0), rat(1.0)];
    let lines3: Vec<Line> = [[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        .iter()
        .flat_map(|a| {
            let neg: Vec<f64> = a.iter().map(|v| -v).collect();
            [Line::of(a, 0.0), Line::of(&neg, 0.0)]
        })
        .collect();
    let q3 = vec![rat(1.0); lines3.len()];
    assert_eq!(
        min_over_box(&lo3, &hi3, &rat(0.125), &q3, &lines3),
        rat(0.125)
    );
    // A grid control on a mixed-sign layer.
    let lines = vec![
        Line::of(&[0.5, -0.75], 0.125),
        Line::of(&[-1.0, 0.25], -0.375),
        Line::of(&[0.25, 1.0], 0.0625),
    ];
    let q = vec![rat(-1.0), rat(0.75), rat(-0.5)];
    let c = rat(0.25);
    let min = min_over_box(&lo, &hi, &c, &q, &lines);
    assert!(
        vertices(&lo, &hi, &lines)
            .iter()
            .any(|x| value(&c, &q, &lines, x) == min),
        "the minimum is a value of m"
    );
    for i in 0..=64 {
        for k in 0..=64 {
            let x = [
                rat(-1.0 + f64::from(i) / 32.0),
                rat(-1.0 + f64::from(k) / 32.0),
            ];
            assert!(
                min <= value(&c, &q, &lines, &x),
                "a grid point lies below the minimum"
            );
        }
    }
}
