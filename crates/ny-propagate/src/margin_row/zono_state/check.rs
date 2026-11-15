// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pointwise CONTAINMENT checker of a frozen dual state (diagnostic; the unit
//! tests and the real-ONNX soundness pin both use it). Nothing verdict-bearing
//! calls this module.
//!
//! For a concrete input `x` it runs the twin net forward in DOUBLE-DOUBLE
//! arithmetic (~2^-104 relative per operation, full double-double operands
//! throughout — no per-layer truncation to binary64), forms the generator
//! vector `e(x)` the state's claim is about (input symbols and one relaxation
//! symbol per unstable neuron, from the TRUE pre-activations), and reports how
//! far each claim is from failing:
//!
//! * `box_excess`: `max(l - z, z - u)` over every ReLU pre-activation (the
//!   frame's premise; `<= 0` inside);
//! * `symbol_excess`: `max |e_k| - 1` over every symbol;
//! * `row_excess`: `max_k |z_k - c_k - a_k . e| - r_k`;
//! * `obj_excess`: `|m - c_obj - d . e| - r_obj`, `m = y_t - y_class`.
//!
//! A claim holds when its excess is `<= 0`. The evaluation itself carries a
//! double-double error far below [`DD_TOLERANCE`], which is what a caller
//! should compare against.

use crate::zono_core::ZonoRefusal;

use super::super::net::{TwinNet, TwinOp};
use super::build::FrozenDualState;
use super::frame::FrozenDualFrame;

/// Excess a caller should tolerate as double-double evaluation noise (the
/// state's remainders are > 1e-17 on any realistic row; a genuine defect of
/// that size would still show).
pub const DD_TOLERANCE: f64 = 1e-20;

/// A double-double value `hi + lo`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dd {
    /// Leading part.
    pub hi: f64,
    /// Trailing part.
    pub lo: f64,
}

/// `self + o` (the double-double sum; was the inherent `add`).
impl std::ops::Add for Dd {
    type Output = Self;

    fn add(self, o: Self) -> Self {
        let (s, e) = Self::two_sum(self.hi, o.hi);
        let (t, f) = Self::two_sum(self.lo, o.lo);
        let r = Self::quick(s, e + t);
        Self::quick(r.hi, r.lo + f)
    }
}

impl std::ops::AddAssign for Dd {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

/// `self - o` (was the inherent `sub`).
impl std::ops::Sub for Dd {
    type Output = Self;

    fn sub(self, o: Self) -> Self {
        self + Self {
            hi: -o.hi,
            lo: -o.lo,
        }
    }
}

/// `self * o` (was the inherent `mul`).
impl std::ops::Mul for Dd {
    type Output = Self;

    fn mul(self, o: Self) -> Self {
        let p = self.hi * o.hi;
        let e = self.hi.mul_add(o.hi, -p);
        Self::quick(p, e + (self.hi * o.lo + self.lo * o.hi))
    }
}

impl Dd {
    /// Exact embedding of a binary64 value.
    #[must_use]
    pub fn from_f64(v: f64) -> Self {
        Self { hi: v, lo: 0.0 }
    }

    /// Nearest binary64.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        (s, e)
    }

    fn quick(a: f64, b: f64) -> Self {
        let s = a + b;
        let e = b - (s - a);
        Self { hi: s, lo: e }
    }

    /// `self * b` for a binary64 `b` (exact leading product by FMA).
    #[must_use]
    pub fn mul_f64(self, b: f64) -> Self {
        let p = self.hi * b;
        let e = self.hi.mul_add(b, -p);
        Self::quick(p, e + self.lo * b)
    }

    /// `self / b` for a binary64 `b != 0` (two correction steps).
    #[must_use]
    pub fn div_f64(self, b: f64) -> Self {
        let q1 = self.hi / b;
        let r = self - Self::from_f64(q1).mul_f64(b);
        let q2 = r.hi / b;
        let r2 = r - Self::from_f64(q2).mul_f64(b);
        let q3 = r2.hi / b;
        Self::quick(q1, q2) + Self::from_f64(q3)
    }

    /// `|self|`.
    #[must_use]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 || (self.hi == 0.0 && self.lo < 0.0) {
            Self {
                hi: -self.hi,
                lo: -self.lo,
            }
        } else {
            self
        }
    }

    /// `max(self, 0)`.
    #[must_use]
    pub fn relu(self) -> Self {
        if self.hi > 0.0 || (self.hi == 0.0 && self.lo > 0.0) {
            self
        } else {
            Self::from_f64(0.0)
        }
    }
}

/// Double-double forward of the STORED twin net at `x`: every tensor, `t[0]`
/// the input and `t[k + 1]` the output of op `k`.
///
/// The stored network is one member of the certified BatchNorm-fold error
/// ball every charge covers, so the state must contain it.
pub fn dd_forward(net: &TwinNet, x: &[f64]) -> Result<Vec<Vec<Dd>>, ZonoRefusal> {
    if x.len() != net.n_in {
        return Err(ZonoRefusal::Shape("dd_forward input".to_string()));
    }
    let mut t: Vec<Vec<Dd>> = Vec::with_capacity(net.ops.len() + 1);
    t.push(x.iter().map(|&v| Dd::from_f64(v)).collect());
    for (k, op) in net.ops.iter().enumerate() {
        let out: Vec<Dd> = match op {
            TwinOp::Conv(cv) => {
                let src = &t[cv.input];
                let co = cv.kernel.0;
                let p = cv.oshape.1 * cv.oshape.2;
                let kk = cv.k_fwd;
                let mut out = vec![Dd::from_f64(0.0); co * p];
                for oc in 0..co {
                    for sp in 0..p {
                        let mut acc = Dd::from_f64(cv.bias[oc]);
                        for tap in 0..kk {
                            let gi = cv.gather[tap * p + sp];
                            if gi == usize::MAX {
                                continue;
                            }
                            let w = cv.wmat[oc * kk + tap];
                            if w != 0.0 {
                                acc += src[gi].mul_f64(w);
                            }
                        }
                        out[oc * p + sp] = acc;
                    }
                }
                out
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                ..
            } => t[*input]
                .iter()
                .enumerate()
                .map(|(j, v)| v.mul_f64(scale[j]) + Dd::from_f64(shift[j]))
                .collect(),
            TwinOp::Relu { input, .. } => t[*input].iter().map(|v| v.relu()).collect(),
            TwinOp::Add { lhs, rhs } => {
                t[*lhs].iter().zip(&t[*rhs]).map(|(a, b)| *a + *b).collect()
            }
            TwinOp::Flatten { input } => t[*input].clone(),
            TwinOp::Gemm {
                input,
                weight,
                bias,
                shape,
                ..
            } => {
                let (no, ni) = *shape;
                let src = &t[*input];
                (0..no)
                    .map(|o| {
                        let mut acc = Dd::from_f64(bias[o]);
                        for (i, &w) in weight[o * ni..(o + 1) * ni].iter().enumerate() {
                            if w != 0.0 {
                                acc += src[i].mul_f64(w);
                            }
                        }
                        acc
                    })
                    .collect()
            }
        };
        if out.len() != net.tsize[k + 1] {
            return Err(ZonoRefusal::Shape(format!("dd_forward op {k}")));
        }
        t.push(out);
    }
    Ok(t)
}

/// The containment report of one point (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct PointCheck {
    /// `max(l - z, z - u)` over every relu pre-activation.
    pub box_excess: f64,
    /// `max |e| - 1` over every symbol.
    pub symbol_excess: f64,
    /// `max_k |z_k - c_k - a_k . e| - r_k`.
    pub row_excess: f64,
    /// `|m - c_obj - d . e| - r_obj`.
    pub obj_excess: f64,
    /// The margin `y_t - y_class` at `x` (double-double, rounded).
    pub margin: f64,
    /// Per state row: the pre-activation `z_k(x)` (double-double, rounded).
    pub z: Vec<f64>,
}

impl PointCheck {
    /// True when every claim holds within [`DD_TOLERANCE`].
    #[must_use]
    pub fn holds(&self) -> bool {
        self.box_excess <= DD_TOLERANCE
            && self.symbol_excess <= DD_TOLERANCE
            && self.row_excess <= DD_TOLERANCE
            && self.obj_excess <= DD_TOLERANCE
    }
}

/// Check `state` (built over `frame` for `net`) at the input point `x`.
pub fn check_point(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    state: &FrozenDualState,
    x: &[f64],
) -> Result<PointCheck, ZonoRefusal> {
    // Reconstructing the objective requires a witness for every relaxation
    // symbol. A streaming search state retains only selected rows.
    if state.n_gens != frame.n_gens || state.num_unstable() != frame.n_unstable {
        return Err(ZonoRefusal::Shape(
            "frozen-dual point check requires the complete state".to_string(),
        ));
    }
    let vals = dd_forward(net, x)?;
    let n = state.n_gens;
    let mut e = vec![Dd::from_f64(0.0); n];
    let mut symbol_excess = f64::NEG_INFINITY;
    for (gi, &dim) in frame.in_cols.iter().enumerate() {
        let v = (Dd::from_f64(x[dim]) - Dd::from_f64(frame.mid[dim])).div_f64(frame.rad[dim]);
        symbol_excess = symbol_excess.max((v.abs() - Dd::from_f64(1.0)).to_f64());
        e[gi] = v;
    }
    let mut box_excess = f64::NEG_INFINITY;
    for layer in &frame.layers {
        let z = &vals[layer.input];
        for (j, zj) in z.iter().enumerate() {
            box_excess = box_excess
                .max((Dd::from_f64(layer.l[j]) - *zj).to_f64())
                .max((*zj - Dd::from_f64(layer.u[j])).to_f64());
        }
    }
    let s_total = state.num_unstable();
    let mut zs = vec![0.0f64; s_total];
    let mut zdd = vec![Dd::from_f64(0.0); s_total];
    for k in 0..s_total {
        let layer = &frame.layers[state.layer[k]];
        let z = vals[layer.input][state.neuron[k]];
        zdd[k] = z;
        zs[k] = z.to_f64();
        // e_k = (relu(z) - lambda z - mu) / mu
        let ek =
            (z.relu() - z.mul_f64(state.lam[k]) - Dd::from_f64(state.mu[k])).div_f64(state.mu[k]);
        symbol_excess = symbol_excess.max((ek.abs() - Dd::from_f64(1.0)).to_f64());
        e[state.enc[k]] = ek;
    }
    let affine = |c: f64, coeffs: &[f64]| -> Dd {
        let mut acc = Dd::from_f64(c);
        for (&a, ev) in coeffs.iter().zip(&e) {
            if a != 0.0 {
                acc += ev.mul_f64(a);
            }
        }
        acc
    };
    let mut row_excess = f64::NEG_INFINITY;
    for k in 0..s_total {
        let pred = affine(state.c_in[k], state.row(k));
        let gap = (zdd[k] - pred).abs() - Dd::from_f64(state.r[k]);
        row_excess = row_excess.max(gap.to_f64());
    }
    let logits = &vals[frame.out_tensor];
    let m = logits[state.t] - logits[state.class];
    let pred = affine(state.c_obj, &state.d);
    let obj_excess = ((m - pred).abs() - Dd::from_f64(state.r_obj)).to_f64();
    Ok(PointCheck {
        box_excess,
        symbol_excess,
        row_excess,
        obj_excess,
        margin: m.to_f64(),
        z: zs,
    })
}
