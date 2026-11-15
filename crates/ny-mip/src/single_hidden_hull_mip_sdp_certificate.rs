// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Source-authored symmetric interval Q and arbitrary-Gram DD repair.
use super::*;
#[derive(Clone, Copy, Debug)]
pub(super) struct I {
    pub lo: f64,
    pub hi: f64,
}
impl I {
    pub(super) fn point(x: f64) -> Option<Self> {
        x.is_finite().then_some(Self { lo: x, hi: x })
    }
    pub(super) fn zero() -> Self {
        Self { lo: 0., hi: 0. }
    }
    pub(super) fn add(self, b: Self) -> Option<Self> {
        if b.lo == 0. && b.hi == 0. {
            return Some(self);
        }
        if self.lo == 0. && self.hi == 0. {
            return Some(b);
        }
        Some(Self {
            lo: finite(add_down(self.lo, b.lo))?,
            hi: finite(add_up(self.hi, b.hi))?,
        })
    }
    pub(super) fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    pub(super) fn sub(self, b: Self) -> Option<Self> {
        self.add(b.neg())
    }
    pub(super) fn scale(self, k: f64) -> Option<Self> {
        if !k.is_finite() {
            return None;
        }
        if k == 0. {
            return Some(Self::zero());
        }
        if k == 1. {
            return Some(self);
        }
        if k == -1. {
            return Some(self.neg());
        }
        let a = mul_down(self.lo, k).min(mul_down(self.hi, k));
        let b = mul_up(self.lo, k).max(mul_up(self.hi, k));
        Some(Self {
            lo: finite(a)?,
            hi: finite(b)?,
        })
    }
}
#[derive(Debug)]
pub(super) struct System {
    pub n: usize,
    pub q: Vec<I>,
    pub constant: I,
}
fn symmetric(q: &mut [I], n: usize, i: usize, j: usize, v: I) {
    q[i * n + j] = v;
    q[j * n + i] = v;
}
pub(super) fn system(
    cuts: &HullRootMipCuts<'_>,
    g: &[f64],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<System> {
    require_hull_environment().ok()?;
    let p = cuts.problem;
    let d = p.input_dim;
    if !capacity(p)
        || p.w1.len() != d.checked_mul(p.hidden_dim)?
        || p.b1.len() != p.hidden_dim
        || p.c.len() != p.hidden_dim
        || p.lower.len() != d
        || p.upper.len() != d
        || g.len() != d
        || g.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let heads: Vec<_> =
        p.c.iter()
            .enumerate()
            .filter_map(|(j, c)| (*c < 0.).then_some(j))
            .collect();
    let n = 1 + d + heads.len();
    if heads.is_empty() || n > MAX_N {
        return None;
    }
    let mut center = vec![0.; d];
    let mut radius = vec![0.; d];
    for i in 0..d {
        let (l, u) = (p.lower[i], p.upper[i]);
        if !l.is_finite() || !u.is_finite() || l > u {
            return None;
        }
        center[i] = l.midpoint(u);
        if center[i] < l || center[i] > u {
            return None;
        }
        radius[i] = finite(sub_up(u, center[i]).max(sub_up(center[i], l)).max(0.))?;
    }
    let mut q = vec![I::zero(); n * n];
    let mut constant = I::zero();
    let mut linear = vec![I::zero(); d];
    for (role, &j) in heads.iter().enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        let a = finite(-p.c[j])?;
        let mut bias = I::point(p.b1[j])?;
        for i in 0..d {
            bias = bias.add(I::point(p.w_row(j)[i])?.scale(center[i])?)?;
        }
        let half_bias = bias.scale(a)?.scale(0.5)?;
        constant = constant.add(half_bias)?;
        symmetric(&mut q, n, 0, 1 + d + role, half_bias.scale(0.5)?);
        for i in 0..d {
            let half = I::point(p.w_row(j)[i])?
                .scale(radius[i])?
                .scale(a)?
                .scale(0.5)?;
            linear[i] = linear[i].add(half)?;
            symmetric(&mut q, n, 1 + i, 1 + d + role, half.scale(0.5)?);
        }
    }
    for i in 0..d {
        if !live(end) || !poll() {
            return None;
        }
        constant = constant.sub(I::point(g[i])?.scale(center[i])?)?;
        let v = linear[i]
            .sub(I::point(g[i])?.scale(radius[i])?)?
            .scale(0.5)?;
        symmetric(&mut q, n, 0, 1 + i, v);
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some(System { n, q, constant })
}
/// Returns (upper, sum of directed diagonal repairs). L is never trusted.
pub(super) fn certify(
    s: &System,
    lambda: &[f64],
    l: &[f64],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<(f64, f64)> {
    require_hull_environment().ok()?;
    let n = s.n;
    if n == 0
        || n > MAX_N
        || s.q.len() != n * n
        || lambda.len() != n
        || l.len() != n * n
        || !s.constant.lo.is_finite()
        || !s.constant.hi.is_finite()
        || s.constant.lo > s.constant.hi
        || lambda.iter().chain(l).any(|v| !v.is_finite())
    {
        return None;
    }
    let mut off = vec![0.; n];
    let mut diag = vec![0.; n];
    for i in 0..n {
        if !live(end) || !poll() {
            return None;
        }
        if l[i * n + i + 1..(i + 1) * n].iter().any(|v| *v != 0.) {
            return None;
        }
        for j in 0..=i {
            let a = s.q[i * n + j];
            let b = s.q[j * n + i];
            if !a.lo.is_finite()
                || !a.hi.is_finite()
                || a.lo > a.hi
                || a.lo.to_bits() != b.lo.to_bits()
                || a.hi.to_bits() != b.hi.to_bits()
                || (i == j && (a.lo != 0. || a.hi != 0.))
            {
                return None;
            }
            let mut gram = I::zero();
            for k in 0..=j {
                gram = gram.add(I::point(l[i * n + k])?.scale(l[j * n + k])?)?;
            }
            let e = (if i == j {
                I::point(lambda[i])?
            } else {
                I::zero()
            })
            .sub(a)?
            .sub(gram)?;
            if i == j {
                diag[i] = e.lo;
            } else {
                let v = e.lo.abs().max(e.hi.abs());
                off[i] = finite(add_up(off[i], v))?;
                off[j] = finite(add_up(off[j], v))?;
            }
        }
    }
    let mut upper = s.constant.hi;
    let mut correction = 0.;
    for i in 0..n {
        let rho = finite(sub_up(off[i], diag[i]))?.max(0.);
        let fixed = finite(add_up(lambda[i], rho))?;
        if fixed < 0. {
            return None;
        }
        upper = finite(add_up(upper, fixed))?;
        correction = finite(add_up(correction, rho))?;
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some((upper, correction))
}
