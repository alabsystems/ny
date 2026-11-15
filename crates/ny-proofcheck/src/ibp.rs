// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Outward-rounded interval forward propagation over the IR (IBP), the
//! per-tensor magnitudes the back-substitution charges against, and the
//! rigorous point evaluation used by refutation and self-containment.
//!
//! Every operation rounds its lower end down and its upper end up after
//! EACH correctly rounded f64 operation, so every returned interval contains
//! the exact real-arithmetic value of the tensor for every input in the box.

use crate::arith::{down, up, Iv};
use crate::lower::{Bn, Conv, Gemm, Net, Op};

/// Per-tensor intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds {
    /// Lower ends, per tensor.
    pub lo: Vec<Vec<f64>>,
    /// Upper ends, per tensor.
    pub hi: Vec<Vec<f64>>,
}

impl Bounds {
    /// All tensors empty (to be filled in topological order).
    #[must_use]
    pub fn empty(net: &Net) -> Self {
        Self {
            lo: vec![Vec::new(); net.tensors.len()],
            hi: vec![Vec::new(); net.tensors.len()],
        }
    }

    /// Whether every filled interval is finite and ordered.
    #[must_use]
    pub fn all_finite(&self) -> bool {
        self.lo.iter().zip(&self.hi).all(|(l, h)| {
            l.iter()
                .zip(h)
                .all(|(a, b)| a.is_finite() && b.is_finite() && a <= b)
        })
    }
}

fn conv_fwd(c: &Conv, xl: &[f64], xu: &[f64], yl: &mut [f64], yu: &mut [f64]) {
    let (ih, iw) = c.ihw;
    let (oh, ow) = c.ohw;
    let (sh, sw) = c.stride;
    let (pt, pl) = (c.pad.0 as isize, c.pad.1 as isize);
    for co in 0..c.co {
        for y in 0..oh {
            for x in 0..ow {
                let mut lo = c.b[co];
                let mut hi = c.b[co];
                for ci in 0..c.ci {
                    for ky in 0..c.kh {
                        let iy = (y * sh) as isize + ky as isize - pt;
                        if iy < 0 || iy >= ih as isize {
                            continue;
                        }
                        for kx in 0..c.kw {
                            let ix = (x * sw) as isize + kx as isize - pl;
                            if ix < 0 || ix >= iw as isize {
                                continue;
                            }
                            let w = c.w[((co * c.ci + ci) * c.kh + ky) * c.kw + kx];
                            let j = (ci * ih + iy as usize) * iw + ix as usize;
                            if w >= 0.0 {
                                lo = down(lo + down(w * xl[j]));
                                hi = up(hi + up(w * xu[j]));
                            } else {
                                lo = down(lo + down(w * xu[j]));
                                hi = up(hi + up(w * xl[j]));
                            }
                        }
                    }
                }
                let o = (co * oh + y) * ow + x;
                yl[o] = lo;
                yu[o] = hi;
            }
        }
    }
}

fn gemm_fwd(g: &Gemm, xl: &[f64], xu: &[f64], yl: &mut [f64], yu: &mut [f64]) {
    for j in 0..g.n_out {
        let row = &g.w[j * g.n_in..(j + 1) * g.n_in];
        let mut lo = g.b[j];
        let mut hi = g.b[j];
        for (i, &w) in row.iter().enumerate() {
            if w >= 0.0 {
                lo = down(lo + down(w * xl[i]));
                hi = up(hi + up(w * xu[i]));
            } else {
                lo = down(lo + down(w * xu[i]));
                hi = up(hi + up(w * xl[i]));
            }
        }
        yl[j] = lo;
        yu[j] = hi;
    }
}

fn bn_fwd(b: &Bn, xl: &[f64], xu: &[f64], yl: &mut [f64], yu: &mut [f64]) {
    for i in 0..xl.len() {
        let c = i / b.hw;
        let s = Iv::new(b.s_lo[c], b.s_hi[c]);
        let v = s * Iv::new(xl[i], xu[i]) + Iv::new(b.sh_lo[c], b.sh_hi[c]);
        yl[i] = v.lo;
        yu[i] = v.hi;
    }
}

/// `max(v, 0)` for one interval end, NaN-PRESERVING. `f64::max` returns the
/// non-NaN operand, so `NaN.max(0.0)` is `0.0`: an end that overflowed into
/// NaN upstream (`0 · inf`, `inf − inf`) would be laundered into the finite
/// `0`, and a point evaluation would then carry an enclosure that EXCLUDES the
/// real value (a false `refuted` on a true `unsat`; review(b2)). NaN stays
/// NaN, so every consumer sees "no enclosure".
#[inline]
fn relu_end(v: f64) -> f64 {
    if v.is_nan() {
        v
    } else {
        v.max(0.0)
    }
}

/// Interval forward of node `k` from the bounds already in `b`.
pub fn forward_node(net: &Net, k: usize, b: &mut Bounds) {
    let node = &net.nodes[k];
    let n = net.tensors[node.out].size;
    let mut yl = vec![0.0; n];
    let mut yu = vec![0.0; n];
    match &node.op {
        Op::Conv(c) => conv_fwd(c, &b.lo[c.x], &b.hi[c.x], &mut yl, &mut yu),
        Op::Gemm(g) => gemm_fwd(g, &b.lo[g.x], &b.hi[g.x], &mut yl, &mut yu),
        Op::Bn(bn) | Op::Scale(bn) => bn_fwd(bn, &b.lo[bn.x], &b.hi[bn.x], &mut yl, &mut yu),
        Op::Select(sel) => {
            // Exact copies: no arithmetic, nothing to round.
            for (k, &(s, i)) in sel.src.iter().enumerate() {
                let t = sel.xs[s];
                yl[k] = b.lo[t][i];
                yu[k] = b.hi[t][i];
            }
        }
        Op::Add { a, b: rb } => {
            for i in 0..n {
                let v = Iv::new(b.lo[*a][i], b.hi[*a][i]) + Iv::new(b.lo[*rb][i], b.hi[*rb][i]);
                yl[i] = v.lo;
                yu[i] = v.hi;
            }
        }
        Op::AddConst { x, c } => {
            for i in 0..n {
                yl[i] = down(b.lo[*x][i] + c[i]);
                yu[i] = up(b.hi[*x][i] + c[i]);
            }
        }
        Op::Relu { x } => {
            for i in 0..n {
                yl[i] = relu_end(b.lo[*x][i]);
                yu[i] = relu_end(b.hi[*x][i]);
            }
        }
        Op::Identity { x } => {
            yl.copy_from_slice(&b.lo[*x]);
            yu.copy_from_slice(&b.hi[*x]);
        }
    }
    b.lo[node.out] = yl;
    b.hi[node.out] = yu;
}

/// Interval bounds of every tensor over the box `[lo, hi]` (pure IBP).
#[must_use]
pub fn forward(net: &Net, lo: &[f64], hi: &[f64]) -> Bounds {
    let mut b = Bounds::empty(net);
    b.lo[net.input] = lo.to_vec();
    b.hi[net.input] = hi.to_vec();
    for k in 0..net.nodes.len() {
        forward_node(net, k, &mut b);
    }
    b
}

/// Rigorous evaluation at a point: every tensor's enclosure of its exact
/// real-arithmetic value at `x`.
#[must_use]
pub fn eval_point(net: &Net, x: &[f64]) -> Bounds {
    forward(net, x, x)
}

/// Magnitudes of every tensor (`M_t,i = max(|lo|, |hi|)`), their upward
/// totals, and per affine node the upward vector `Q = |W| · M_in` the
/// back-substitution charges its fan-in rounding against.
#[derive(Debug, Clone, PartialEq)]
pub struct Magnitudes {
    /// Per-tensor magnitudes.
    pub m: Vec<Vec<f64>>,
    /// Per-tensor upward totals `Σ_i M_t,i`.
    pub msum: Vec<f64>,
    /// Per node: `Q` for Conv / Gemm (indexed by output element), else empty.
    pub q: Vec<Vec<f64>>,
    /// Per node: an upper bound on the number of terms one INPUT element
    /// receives from this node's backward product (its backward fan-in).
    pub fanin: Vec<usize>,
}

fn conv_q(c: &Conv, m: &[f64]) -> Vec<f64> {
    let (ih, iw) = c.ihw;
    let (oh, ow) = c.ohw;
    let (sh, sw) = c.stride;
    let (pt, pl) = (c.pad.0 as isize, c.pad.1 as isize);
    let mut q = vec![0.0; c.co * oh * ow];
    for co in 0..c.co {
        for y in 0..oh {
            for x in 0..ow {
                let mut s = 0.0f64;
                for ci in 0..c.ci {
                    for ky in 0..c.kh {
                        let iy = (y * sh) as isize + ky as isize - pt;
                        if iy < 0 || iy >= ih as isize {
                            continue;
                        }
                        for kx in 0..c.kw {
                            let ix = (x * sw) as isize + kx as isize - pl;
                            if ix < 0 || ix >= iw as isize {
                                continue;
                            }
                            let w = c.w[((co * c.ci + ci) * c.kh + ky) * c.kw + kx];
                            let j = (ci * ih + iy as usize) * iw + ix as usize;
                            s = up(s + up(w.abs() * m[j]));
                        }
                    }
                }
                q[(co * oh + y) * ow + x] = s;
            }
        }
    }
    q
}

fn gemm_q(g: &Gemm, m: &[f64]) -> Vec<f64> {
    (0..g.n_out)
        .map(|j| {
            let row = &g.w[j * g.n_in..(j + 1) * g.n_in];
            let mut s = 0.0f64;
            for (w, mi) in row.iter().zip(m) {
                s = up(s + up(w.abs() * *mi));
            }
            s
        })
        .collect()
}

/// Magnitude vector of one tensor and its upward total.
#[must_use]
pub fn tensor_mag(lo: &[f64], hi: &[f64]) -> (Vec<f64>, f64) {
    let m: Vec<f64> = lo
        .iter()
        .zip(hi)
        .map(|(a, b)| a.abs().max(b.abs()))
        .collect();
    let s = crate::arith::sum_up(&m);
    (m, s)
}

/// `Q` and the backward fan-in of node `k`, given its input's magnitudes.
#[must_use]
pub fn node_q(net: &Net, k: usize, m_in: &[f64]) -> (Vec<f64>, usize) {
    match &net.nodes[k].op {
        Op::Conv(c) => (conv_q(c, m_in), c.co * c.kh * c.kw),
        Op::Gemm(g) => (gemm_q(g, m_in), g.n_out),
        _ => (Vec::new(), 1),
    }
}

impl Magnitudes {
    /// An empty record sized for `net` (filled with [`Self::set_tensor`] and
    /// [`Self::set_node`] in topological order).
    #[must_use]
    pub fn empty(net: &Net) -> Self {
        Self {
            m: vec![Vec::new(); net.tensors.len()],
            msum: vec![0.0; net.tensors.len()],
            q: vec![Vec::new(); net.nodes.len()],
            fanin: vec![1; net.nodes.len()],
        }
    }

    /// Record tensor `t`'s magnitudes from its bounds.
    pub fn set_tensor(&mut self, t: usize, lo: &[f64], hi: &[f64]) {
        let (m, s) = tensor_mag(lo, hi);
        self.m[t] = m;
        self.msum[t] = s;
    }

    /// Record node `k`'s `Q` from its (already recorded) input magnitudes.
    pub fn set_node(&mut self, net: &Net, k: usize) {
        let x = net.nodes[k].op.inputs()[0];
        let (q, f) = node_q(net, k, &self.m[x]);
        self.q[k] = q;
        self.fanin[k] = f;
    }
}

/// Magnitudes of `bounds` over the net.
#[must_use]
pub fn magnitudes(net: &Net, bounds: &Bounds) -> Magnitudes {
    let mut mags = Magnitudes::empty(net);
    for t in 0..net.tensors.len() {
        mags.set_tensor(t, &bounds.lo[t], &bounds.hi[t]);
    }
    for k in 0..net.nodes.len() {
        mags.set_node(net, k);
    }
    mags
}
