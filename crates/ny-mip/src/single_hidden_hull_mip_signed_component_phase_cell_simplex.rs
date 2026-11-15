// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded revised-simplex advice. No floating solver status grants authority.
use super::*;

pub(in crate::single_hidden_hull::mip_cuts::signed) struct Simplex {
    pub values: [f64; 256],
    pub activity: [f64; 8],
    pub mapped: [usize; 8],
    pub k: usize,
    pub basis: [usize; 9],
    pub mass: [f64; 9],
    inverse: [[f64; 9]; 9],
}
impl Simplex {
    pub fn new(
        values: &[f64],
        mapped: &[bool],
        activity: &[f64],
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        let n = mapped.len();
        if !(1..=8).contains(&n) || values.len() != 1 << n || activity.len() != n {
            return None;
        }
        let mut out = Self {
            values: [f64::INFINITY; 256],
            activity: [0.; 8],
            mapped: [0; 8],
            k: 0,
            basis: [0; 9],
            mass: [0.; 9],
            inverse: [[0.; 9]; 9],
        };
        for j in 0..n {
            if !poll() || !activity[j].is_finite() || !(0. ..=1.).contains(&activity[j]) {
                return None;
            }
            if mapped[j] {
                out.mapped[out.k] = j;
                out.activity[out.k] = activity[j];
                out.k += 1;
            }
        }
        // Eliminating an unmapped selector means minimizing over its phases,
        // never fixing it to zero or keeping its marginal as a constraint.
        for (mask, &value) in values.iter().enumerate() {
            if !poll() {
                return None;
            }
            finite(2. * value)?;
            let mut projected = 0;
            for i in 0..out.k {
                projected |= ((mask >> out.mapped[i]) & 1) << i;
            }
            out.values[projected] = out.values[projected].min(value);
        }
        let mut order = [0usize; 8];
        for (j, v) in order[..out.k].iter_mut().enumerate() {
            *v = j;
        }
        order[..out.k]
            .sort_unstable_by(|&a, &b| out.activity[b].total_cmp(&out.activity[a]).then(a.cmp(&b)));
        out.inverse[0][0] = 1.;
        if out.k != 0 {
            out.inverse[0][order[0] + 1] = -1.;
            for j in 1..=out.k {
                out.basis[j] = out.basis[j - 1] | (1 << order[j - 1]);
                out.inverse[j][order[j - 1] + 1] = 1.;
                if j < out.k {
                    out.inverse[j][order[j] + 1] = -1.;
                }
            }
        }
        Some(out)
    }

    fn evaluate(&mut self, poll: &mut dyn FnMut() -> bool) -> Option<[f64; 9]> {
        let mut dual = [0.; 9];
        for i in 0..=self.k {
            if !poll() {
                return None;
            }
            let cost = finite(2. * self.values[self.basis[i]])?;
            let mut mass = self.inverse[i][0];
            for j in 0..=self.k {
                dual[j] = finite(dual[j] + finite(cost * self.inverse[i][j])?)?;
                if j != 0 {
                    mass = finite(mass + finite(self.inverse[i][j] * self.activity[j - 1])?)?;
                }
            }
            // Slight negative residuals are tolerated as advice only. The
            // normalized transport witness below supplies the gap authority.
            if mass < -1e-9 {
                return None;
            }
            self.mass[i] = mass;
        }
        Some(dual)
    }

    fn entering(
        &self,
        dual: &[f64; 9],
        bland: bool,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<(Option<usize>, f64)> {
        let mut basic = [false; 256];
        for &mask in &self.basis[..=self.k] {
            basic[mask] = true;
        }
        let mut enter: Option<usize> = None;
        let (mut price, mut least, mut lower) = (0., 0., f64::INFINITY);
        for i in 0usize..1 << self.k {
            if !poll() {
                return None;
            }
            // Gray order changes one bit, eliminating the 256-entry price
            // scratch without another mask walk or per-mask dot product.
            let mask = i ^ (i >> 1);
            if i != 0 {
                let bit = i.trailing_zeros() as usize;
                let delta = dual[bit + 1];
                price = finite(
                    price
                        + if mask & (1 << bit) != 0 {
                            delta
                        } else {
                            -delta
                        },
                )?;
            }
            let floor = finite(2. * self.values[mask] - price)?;
            lower = lower.min(floor);
            let reduced = finite(floor - dual[0])?;
            // Largest violation with a mask-index tie, or Bland after repeated
            // degenerate steps. Only the hard cap bounds floating execution.
            if !basic[mask]
                && reduced < -1e-12
                && enter.is_none_or(|old| {
                    if bland {
                        mask < old
                    } else {
                        reduced < least || (reduced == least && mask < old)
                    }
                })
            {
                enter = Some(mask);
                least = reduced;
            }
        }
        for j in 0..self.k {
            lower = finite(lower + finite(dual[j + 1] * self.activity[j])?)?;
        }
        // Advice only: complete directed repair of the retained coefficients
        // remains mandatory, irrespective of this nominal score or status.
        Some((enter, lower))
    }

    fn pivot(&mut self, entering: usize, poll: &mut dyn FnMut() -> bool) -> Option<f64> {
        let mut direction = [0.; 9];
        let mut leaving: Option<(usize, f64)> = None;
        for i in 0..=self.k {
            if !poll() {
                return None;
            }
            let mut u = self.inverse[i][0];
            for j in 0..self.k {
                if entering & (1 << j) != 0 {
                    u = finite(u + self.inverse[i][j + 1])?;
                }
            }
            direction[i] = u;
            if u > 1e-12 {
                let ratio = finite(self.mass[i].max(0.) / u)?;
                if leaving.is_none_or(|(old, old_ratio)| {
                    ratio < old_ratio || (ratio == old_ratio && self.basis[i] < self.basis[old])
                }) {
                    leaving = Some((i, ratio));
                }
            }
        }
        let (leaving, step) = leaving?;
        let pivot = direction[leaving];
        for j in 0..=self.k {
            self.inverse[leaving][j] = finite(self.inverse[leaving][j] / pivot)?;
        }
        for i in 0..=self.k {
            if !poll() {
                return None;
            }
            if i == leaving {
                continue;
            }
            for j in 0..=self.k {
                self.inverse[i][j] =
                    finite(self.inverse[i][j] - finite(direction[i] * self.inverse[leaving][j])?)?;
            }
        }
        self.basis[leaving] = entering;
        Some(step)
    }

    pub fn run(
        &mut self,
        limit: usize,
        r: &mut SignedTableSupportReport,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<[f64; 8]> {
        if limit > PIVOTS || r.pivots != 0 {
            return None;
        }
        let (mut best, mut best_score) = ([0.; 8], f64::NEG_INFINITY);
        let (mut degenerate, mut bland) = (0usize, false);
        loop {
            let dual = self.evaluate(poll)?;
            let (entering, score) = self.entering(&dual, bland, poll)?;
            if score > best_score {
                best_score = score;
                for j in 0..self.k {
                    best[self.mapped[j]] = -dual[j + 1];
                }
            }
            if entering.is_none() || r.pivots >= limit {
                r.stop_reason = if entering.is_none() {
                    "nominal-optimum"
                } else {
                    "pivot-limit"
                };
                return Some(best);
            }
            // Count an attempted update even if numeric failure/cancellation
            // interrupts it; no pivot or work allowance is refunded.
            r.pivots += 1;
            let step = self.pivot(entering?, poll)?;
            degenerate = if step <= 1e-12 { degenerate + 1 } else { 0 };
            bland |= degenerate >= 2;
        }
    }
}
