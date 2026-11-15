// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! V2 — the vit root α search (ENGINE_PROGRAM §2 V, `#vit-root-alpha`),
//! UNTRUSTED, on the CPU.
//!
//! # What it searches
//!
//! The graph lane's root bound on a margin row `c·y` is ONE backward pass over
//! the network with every intermediate box frozen at the lane's certified
//! value. Three families of relaxation parameters decide it:
//!
//! * ReLU lower slopes `α ∈ [0, 1]` of the MLP blocks (lower line `y ≥ α x`
//!   on unstable neurons, upper chord fixed);
//! * the interpolated McCormick planes of every var×var attention product
//!   (`Q·Kᵀ`, `attn·V`): per product element `(i, j, l)`, `α[0]` interpolates
//!   the LOWER plane between `L2` (r = 0) and `L1` (r = 1) and `α[2]` the
//!   UPPER plane between `U2` and `U1` — the lane's `[4, m, n, k]` layout,
//!   whose `[1]`/`[3]` (upper-direction planes) no margin row reads and which
//!   this search therefore never moves;
//! * the interpolated McCormick planes of the fixed-index softmax's
//!   `exp × 1/Σexp` MulBinary (`[2, n]`: `[0]` lower, `[1]` upper).
//!
//! Every other nonlinearity is relaxed with FROZEN lines: the monolithic
//! softmax with the lane's own LSE affine bounds (carried in the net), Exp and
//! Reciprocal with the tangent at the box midpoint (lower, both convex) and the
//! chord (upper), Clip with the chord slope shifted to touch the function's
//! extremes (its breakpoints). Those only have to be CLOSE to the lane's
//! choices for the search to point the right way; the lane's certified pass
//! decides.
//!
//! # The objective and its gradient (boxes frozen: exact, not a surrogate)
//!
//! With the boxes frozen, a row's bound is `min_x L_α(x)` for the linear
//! function the backward pass assembles, attained at the input corner `x*`
//! the backward coefficients select. Around a generic `α` no backward
//! coefficient changes sign, so the bound EQUALS `c · f_α(x*)`, where `f_α` is
//! the network with every relaxed op replaced by the line/plane the backward
//! chose. Hence (one forward walk of `f_α` at `x*` per row):
//!
//! * ReLU:      `∂/∂α_j = Λ_j · x̂_j`  (unstable, `Λ_j > 0`);
//! * McCormick: `∂/∂r   = w · (P1 − P2)(x̂, ŷ)` with `w` the backward weight on
//!   the product element (`Λ · scale`) and `P1 − P2` the difference of the
//!   two supporting planes at the linearised operand values.
//!
//! The loss is abc's: the sum of the bounds of the rows not yet above their
//! threshold. Projected Adam (lr decay 0.98 per iteration), and the iterate
//! with the best (hinge, worst slack) score is returned — the lane's own
//! ranking. Early exits: every row above threshold (`verified`), no score
//! improvement for [`PATIENCE`] iterations (`patience`), the iteration cap
//! (`iters`) or the deadline (`deadline`).
//!
//! # Cost shape and R6
//!
//! Rows are independent: each worker evaluates whole rows (backward, the
//! linearised forward and the row's sparse gradient) and the gradients are
//! folded in ROW ORDER after every chunk, so the result does not depend on the
//! worker count or the chunking. A chunk holds at most `threads` rows' working
//! sets, admitted against the request's memory cap.
//!
//! # Trust
//!
//! UNTRUSTED end to end: f32, round to nearest, no rounding charged. The only
//! product is a parameter HINT; the lane's ingest gate refuses non-finite
//! values, clamps the rest into `[0, 1]`, runs ONE certified evaluation and
//! keeps it only when strictly better.

use std::time::Instant;

use ny_core::search::{
    GraphAlphaHint, GraphRootOutcome, GraphRootRequest, GraphRootStats, GraphSearchNet,
    GraphSearchOp, Refusal, TensorBox, UnaryKind, Untrusted, GATHER_PAD,
};

use crate::program::NONE;
use crate::real::Real;

/// Iterations without a score improvement before the search stops.
pub(crate) const PATIENCE: usize = 10;

/// Per-iteration learning-rate decay (abc `ExponentialLR` 0.98).
pub(crate) const LR_DECAY: f64 = 0.98;

const ADAM_B1: f64 = 0.9;
const ADAM_B2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;

/// The step used when the request carries no usable one.
pub(crate) const DEFAULT_LR: f64 = 0.1;

/// ReLU neuron status.
pub(crate) const INACTIVE: u8 = 0;
pub(crate) const ACTIVE: u8 = 1;
pub(crate) const UNSTABLE: u8 = 2;

// ---------------------------------------------------------------------------
//  The lowered program
// ---------------------------------------------------------------------------

/// One lowered op; tensor `k + 1` is the output of op `k`.
pub(crate) enum GOp<R> {
    /// `out[o] = bias[o] + Σ w · inputs[slot][index]`, terms `(out, slot, index, w)`.
    Sparse {
        inputs: Vec<usize>,
        terms: Vec<(u32, u32, u32, R)>,
        bias: Vec<R>,
    },
    /// Last-axis dense layer, row-major `[n_out][n_in]`.
    Dense {
        input: usize,
        rows: usize,
        n_in: usize,
        n_out: usize,
        w: Vec<R>,
        b: Vec<R>,
    },
    /// Convolution, `[p][k]` gather, `[cout][k]` weights.
    Conv {
        input: usize,
        cout: usize,
        k: usize,
        p: usize,
        gather_pk: Vec<u32>,
        w: Vec<R>,
        b: Vec<R>,
    },
    /// ReLU at site `site`.
    Relu { input: usize, site: usize },
    /// Var×var product at site `site`.
    Bilinear { a: usize, b: usize, site: usize },
    /// Elementwise product at site `site`.
    Mul { a: usize, b: usize, site: usize },
    /// Elementwise frozen lines `[slope, icpt]`.
    Lines {
        input: usize,
        lower: Vec<[R; 2]>,
        upper: Vec<[R; 2]>,
    },
    /// Grouped softmax with frozen affine bounds.
    Softmax {
        input: usize,
        groups: usize,
        len: usize,
        la: Vec<R>,
        lb: Vec<R>,
        ua: Vec<R>,
        ub: Vec<R>,
    },
}

/// A ReLU site: its input, per-neuron status, upper chord and starting slopes.
pub(crate) struct ReluSite<R> {
    pub(crate) op: usize,
    pub(crate) input: usize,
    pub(crate) status: Vec<u8>,
    pub(crate) slope: Vec<R>,
    pub(crate) icpt: Vec<R>,
    pub(crate) init: Vec<R>,
}

impl<R: Real> ReluSite<R> {
    /// The site of ReLU op `op` over the reference box `[l, u]` of its input
    /// `input`: per-neuron status, the upper chord of every unstable neuron
    /// and the starting slopes (`entries`' slopes for `op` when their width
    /// matches, else the adaptive heuristic). Refuses when an unstable neuron
    /// has a non-finite side.
    pub(crate) fn from_box(
        op: usize,
        input: usize,
        l: &[f64],
        u: &[f64],
        entries: &[(usize, Vec<f32>)],
    ) -> Result<Self, Refusal> {
        let n = l.len();
        let mut status = vec![INACTIVE; n];
        let mut slope = vec![R::ZERO; n];
        let mut icpt = vec![R::ZERO; n];
        for j in 0..n {
            let (lj, uj) = (l[j], u[j]);
            if lj >= 0.0 {
                status[j] = ACTIVE;
            } else if uj > 0.0 {
                if !(lj.is_finite() && uj.is_finite()) {
                    return Err(Refusal::Nothing(
                        "an unstable ReLU has a non-finite reference box",
                    ));
                }
                status[j] = UNSTABLE;
                let s = uj / (uj - lj);
                slope[j] = R::from_f64(s);
                icpt[j] = R::from_f64(-s * lj);
            }
        }
        let init = init_params::<R>(entries, op, n, |j| if u[j] >= -l[j] { 1.0 } else { 0.0 });
        Ok(Self {
            op,
            input,
            status,
            slope,
            icpt,
            init,
        })
    }
}

/// What one backward / linearised-forward pass reads besides the program: the
/// ReLU sites (the program's own for the root, a domain's re-derived ones for
/// M7), the input box of the concretisation, and the Lagrangian INJECTIONS
/// `(tensor, neuron, coefficient)` — each added to the backward coefficient of
/// its tensor right before that tensor's producer propagates it (M7's
/// β-CROWN split multipliers: coefficient `−β_k s_k` on the split neuron's
/// pre-activation). The root view carries no injection, so it runs the V2
/// search bit for bit.
pub(crate) struct View<'a, R> {
    pub(crate) relu: &'a [ReluSite<R>],
    pub(crate) lo: &'a [R],
    pub(crate) hi: &'a [R],
    pub(crate) inject: &'a [(usize, u32, R)],
}

/// A var×var product site: shape, operand boxes and starting parameters.
pub(crate) struct BilSite<R> {
    pub(crate) op: usize,
    pub(crate) a: usize,
    pub(crate) b: usize,
    pub(crate) batch: usize,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) tb: bool,
    scale: R,
    xl: Vec<R>,
    xu: Vec<R>,
    yl: Vec<R>,
    yu: Vec<R>,
    init: Vec<R>,
}

impl<R> BilSite<R> {
    /// Parameters per plane block (`m n k`).
    pub(crate) fn block(&self) -> usize {
        self.m * self.n * self.k
    }

    /// Flat A index of `(batch, i, l)`.
    #[inline]
    fn a_index(&self, bb: usize, i: usize, l: usize) -> usize {
        (bb * self.m + i) * self.k + l
    }

    /// Flat B index of `(batch, l, j)` (stored `[n][k]` when transposed).
    #[inline]
    fn b_index(&self, bb: usize, l: usize, j: usize) -> usize {
        if self.tb {
            (bb * self.n + j) * self.k + l
        } else {
            (bb * self.k + l) * self.n + j
        }
    }
}

/// An elementwise product site: operand boxes and starting parameters.
pub(crate) struct MulSite<R> {
    pub(crate) op: usize,
    pub(crate) a: usize,
    pub(crate) b: usize,
    xl: Vec<R>,
    xu: Vec<R>,
    yl: Vec<R>,
    yu: Vec<R>,
    init: Vec<R>,
}

/// The lowered search program plus the request's objective.
pub(crate) struct GProgram<R> {
    pub(crate) tsize: Vec<usize>,
    pub(crate) ops: Vec<GOp<R>>,
    pub(crate) relu: Vec<ReluSite<R>>,
    pub(crate) bil: Vec<BilSite<R>>,
    pub(crate) mul: Vec<MulSite<R>>,
    lo: Vec<R>,
    hi: Vec<R>,
    pub(crate) rows: Vec<Vec<R>>,
    pub(crate) thr: Vec<f64>,
}

/// Every searched parameter, by site.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Alpha<R> {
    pub(crate) relu: Vec<Vec<R>>,
    pub(crate) bil: Vec<Vec<R>>,
    pub(crate) mul: Vec<Vec<R>>,
}

impl<R: Real> Alpha<R> {
    pub(crate) fn zeros_like(&self) -> Self {
        let z =
            |v: &[Vec<R>]| -> Vec<Vec<R>> { v.iter().map(|s| vec![R::ZERO; s.len()]).collect() };
        Self {
            relu: z(self.relu.as_slice()),
            bil: z(self.bil.as_slice()),
            mul: z(self.mul.as_slice()),
        }
    }

    pub(crate) fn len(&self) -> usize {
        let n = |v: &[Vec<R>]| v.iter().map(Vec::len).sum::<usize>();
        n(self.relu.as_slice()) + n(self.bil.as_slice()) + n(self.mul.as_slice())
    }

    /// `self += other`, entry by entry; a non-finite entry of `other` adds 0.
    pub(crate) fn add_finite(&mut self, other: &Self) {
        let add = |dst: &mut [Vec<R>], src: &[Vec<R>]| {
            for (d, s) in dst.iter_mut().zip(src) {
                for (x, &y) in d.iter_mut().zip(s) {
                    if y.is_finite() {
                        *x += y;
                    }
                }
            }
        };
        add(self.relu.as_mut_slice(), other.relu.as_slice());
        add(self.bil.as_mut_slice(), other.bil.as_slice());
        add(self.mul.as_mut_slice(), other.mul.as_slice());
    }
}

/// Frozen boxes of one tensor, borrowed from the request.
type BoxRef<'a> = (&'a [f64], &'a [f64]);

fn box_of<'a>(boxes: &[Option<BoxRef<'a>>], t: usize) -> Result<BoxRef<'a>, Refusal> {
    boxes
        .get(t)
        .copied()
        .flatten()
        .ok_or_else(|| Refusal::Invalid(format!("no reference box for tensor {t}")))
}

fn finite_box_of<'a>(boxes: &[Option<BoxRef<'a>>], t: usize) -> Result<BoxRef<'a>, Refusal> {
    let (l, u) = box_of(boxes, t)?;
    if l.iter().chain(u).all(|v| v.is_finite()) {
        Ok((l, u))
    } else {
        Err(Refusal::Nothing(
            "a relaxed product operand has a non-finite reference box",
        ))
    }
}

/// A site's starting parameters from the request: the entry for `op` when its
/// length is `len`, else `fallback`; non-finite values take `fallback`, the
/// rest are clamped into `[0, 1]`.
fn init_params<R: Real>(
    entries: &[(usize, Vec<f32>)],
    op: usize,
    len: usize,
    fallback: impl Fn(usize) -> f64,
) -> Vec<R> {
    let given = entries
        .iter()
        .find(|(o, v)| *o == op && v.len() == len)
        .map(|(_, v)| v);
    (0..len)
        .map(|i| {
            let v = given.map_or(f64::NAN, |v| f64::from(v[i]));
            let v = if v.is_finite() { v } else { fallback(i) };
            R::from_f64(v.clamp(0.0, 1.0))
        })
        .collect()
}

/// The frozen search lines `(lower, upper)`, each `[slope, icpt]`, of `kind`
/// over `[l, u]`, or `None` when they are not finite (the search refuses).
pub(crate) fn unary_lines(kind: UnaryKind, l: f64, u: f64) -> Option<([f64; 2], [f64; 2])> {
    if !(l.is_finite() && u.is_finite() && l <= u) {
        return None;
    }
    let lines = match kind {
        UnaryKind::Exp => {
            let mid = f64::midpoint(l, u);
            let em = mid.exp();
            let lower = [em, em * (1.0 - mid)];
            let upper = if u > l {
                let (el, eu) = (l.exp(), u.exp());
                let s = (eu - el) / (u - l);
                [s, el - s * l]
            } else {
                lower
            };
            (lower, upper)
        }
        UnaryKind::Reciprocal => {
            if l <= 0.0 {
                return None;
            }
            let mid = f64::midpoint(l, u);
            let lower = [-1.0 / (mid * mid), 2.0 / mid];
            let upper = if u > l {
                [-1.0 / (l * u), 1.0 / l + 1.0 / u]
            } else {
                lower
            };
            (lower, upper)
        }
        UnaryKind::Clip { lo, hi } => {
            let f = |x: f64| x.max(lo).min(hi);
            if u <= l {
                let c = f(l);
                ([0.0, c], [0.0, c])
            } else {
                // f - s x is piecewise linear with breakpoints lo and hi, so
                // its extremes over [l, u] sit at an end or a breakpoint.
                let s = (f(u) - f(l)) / (u - l);
                let mut cmin = f64::INFINITY;
                let mut cmax = f64::NEG_INFINITY;
                for p in [l, u, lo, hi] {
                    if p >= l && p <= u {
                        let c = f(p) - s * p;
                        cmin = cmin.min(c);
                        cmax = cmax.max(c);
                    }
                }
                ([s, cmin], [s, cmax])
            }
        }
    };
    let finite = lines.0.iter().chain(lines.1.iter()).all(|v| v.is_finite());
    finite.then_some(lines)
}

/// Interpolated McCormick planes `[lax, lay, lc, uax, uay, uc]` of `x·y` over
/// `[xl, xu] × [yl, yu]`: the lower plane at `r_lo` (`L2` at 0, `L1` at 1),
/// the upper plane at `r_up` (`U2` at 0, `U1` at 1). The lane's
/// `interpolated_mccormick` coefficients, without its certification (search).
#[inline]
pub(crate) fn mccormick(xl: f64, xu: f64, yl: f64, yu: f64, r_lo: f64, r_up: f64) -> [f64; 6] {
    [
        (yl - yu) * r_lo + yu,
        (xl - xu) * r_lo + xu,
        (yu * xu - yl * xl) * r_lo - yu * xu,
        (yu - yl) * r_up + yl,
        (xl - xu) * r_up + xu,
        (yl * xu - yu * xl) * r_up - yl * xu,
    ]
}

/// `∂/∂r` of the lower (`L1 − L2`) or upper (`U1 − U2`) plane at `(x, y)`.
#[inline]
fn plane_derivative(xl: f64, xu: f64, yl: f64, yu: f64, x: f64, y: f64, lower: bool) -> f64 {
    if lower {
        (yl - yu) * x + (xl - xu) * y + (xu * yu - xl * yl)
    } else {
        (yu - yl) * x + (xl - xu) * y + (xu * yl - xl * yu)
    }
}

/// `true` when `lo`/`hi` is a finite ordered box of width `n`.
pub(crate) fn finite_ordered_box(lo: &[f64], hi: &[f64], n: usize) -> bool {
    lo.len() == n
        && hi.len() == n
        && lo
            .iter()
            .zip(hi)
            .all(|(l, u)| l.is_finite() && u.is_finite() && l <= u)
}

/// `true` when `b` is a well-formed reference box of a tensor of `tsize`
/// (sides may be infinite, never NaN, never inverted).
pub(crate) fn reference_box_ok(b: &TensorBox, tsize: &[usize]) -> bool {
    b.tensor < tsize.len()
        && b.l.len() == tsize[b.tensor]
        && b.u.len() == tsize[b.tensor]
        && b.l
            .iter()
            .zip(&b.u)
            .all(|(l, u)| !l.is_nan() && !u.is_nan() && l <= u)
}

impl<R: Real> GProgram<R> {
    /// Lower a validated net and its request into the search scalar.
    pub(crate) fn lower(net: &GraphSearchNet, req: &GraphRootRequest) -> Result<Self, Refusal> {
        Self::lower_parts(
            net,
            &req.lo,
            &req.hi,
            &req.boxes,
            &req.rows,
            &req.thresholds,
            &req.init,
        )
    }

    /// [`Self::lower`] from the request's parts (M7 lowers its batch's shared
    /// box and objective the same way, with no starting parameters).
    pub(crate) fn lower_parts(
        net: &GraphSearchNet,
        lo: &[f64],
        hi: &[f64],
        req_boxes: &[TensorBox],
        rows: &[Vec<f64>],
        thresholds: &[f64],
        start: &GraphAlphaHint,
    ) -> Result<Self, Refusal> {
        let tsize = net.tensor_sizes().map_err(Refusal::Invalid)?;
        if tsize.iter().any(|&n| n >= NONE as usize) {
            return Err(Refusal::Invalid(
                "a tensor is too wide for u32 indices".into(),
            ));
        }
        let n_in = net.n_in;
        if lo.len() != n_in || hi.len() != n_in {
            return Err(Refusal::Invalid(format!(
                "input box has {}/{} sides, net expects {n_in}",
                lo.len(),
                hi.len()
            )));
        }
        if !finite_ordered_box(lo, hi, n_in) {
            return Err(Refusal::Invalid(
                "input box is not a finite ordered box".into(),
            ));
        }
        let n_out = tsize[net.output()];
        if rows.is_empty() {
            return Err(Refusal::Nothing("no objective row"));
        }
        if thresholds.len() != rows.len()
            || rows
                .iter()
                .any(|row| row.len() != n_out || !row.iter().all(|v| v.is_finite()))
            || !thresholds.iter().all(|v| v.is_finite())
        {
            return Err(Refusal::Invalid(
                "objective rows are malformed or non-finite".into(),
            ));
        }
        let mut boxes: Vec<Option<BoxRef<'_>>> = vec![None; tsize.len()];
        for b in req_boxes {
            if !reference_box_ok(b, &tsize) {
                return Err(Refusal::Invalid(format!(
                    "reference box for tensor {} is malformed",
                    b.tensor
                )));
            }
            boxes[b.tensor] = Some((b.l.as_slice(), b.u.as_slice()));
        }
        let cvt = |v: &[f64]| -> Vec<R> { v.iter().map(|&x| R::from_f64(x)).collect() };

        let mut ops = Vec::with_capacity(net.ops.len());
        let mut relu = Vec::new();
        let mut bil = Vec::new();
        let mut mul = Vec::new();
        for (k, op) in net.ops.iter().enumerate() {
            let lowered = match op {
                GraphSearchOp::Sparse {
                    inputs,
                    terms,
                    bias,
                    ..
                } => GOp::Sparse {
                    inputs: inputs.clone(),
                    terms: terms
                        .iter()
                        .map(|t| (t.out, t.slot, t.index, R::from_f64(t.weight)))
                        .collect(),
                    bias: cvt(bias),
                },
                GraphSearchOp::Dense {
                    input,
                    rows,
                    n_in,
                    n_out,
                    weight,
                    bias,
                } => GOp::Dense {
                    input: *input,
                    rows: *rows,
                    n_in: *n_in,
                    n_out: *n_out,
                    w: cvt(weight),
                    b: cvt(bias),
                },
                GraphSearchOp::Conv(c) => {
                    let mut gather_pk = vec![NONE; c.k * c.p];
                    for t in 0..c.k {
                        for sp in 0..c.p {
                            let g = c.gather[t * c.p + sp];
                            gather_pk[sp * c.k + t] = if g == GATHER_PAD { NONE } else { g };
                        }
                    }
                    GOp::Conv {
                        input: c.input,
                        cout: c.cout,
                        k: c.k,
                        p: c.p,
                        gather_pk,
                        w: cvt(&c.weight),
                        b: cvt(&c.bias),
                    }
                }
                GraphSearchOp::Relu { input } => {
                    let (l, u) = box_of(&boxes, *input)?;
                    relu.push(ReluSite::from_box(k, *input, l, u, &start.relu)?);
                    GOp::Relu {
                        input: *input,
                        site: relu.len() - 1,
                    }
                }
                GraphSearchOp::Bilinear {
                    a,
                    b,
                    batch,
                    m,
                    n,
                    k: kk,
                    transpose_b,
                    scale,
                } => {
                    let (xl, xu) = finite_box_of(&boxes, *a)?;
                    let (yl, yu) = finite_box_of(&boxes, *b)?;
                    let block = m * n * kk;
                    let init = init_params::<R>(&start.bilinear, k, 4 * block, |_| 1.0);
                    bil.push(BilSite {
                        op: k,
                        a: *a,
                        b: *b,
                        batch: *batch,
                        m: *m,
                        n: *n,
                        k: *kk,
                        tb: *transpose_b,
                        scale: R::from_f64(*scale),
                        xl: cvt(xl),
                        xu: cvt(xu),
                        yl: cvt(yl),
                        yu: cvt(yu),
                        init,
                    });
                    GOp::Bilinear {
                        a: *a,
                        b: *b,
                        site: bil.len() - 1,
                    }
                }
                GraphSearchOp::MulBinary { a, b } => {
                    let (xl, xu) = finite_box_of(&boxes, *a)?;
                    let (yl, yu) = finite_box_of(&boxes, *b)?;
                    let n = xl.len();
                    let init = init_params::<R>(&start.mul_binary, k, 2 * n, |_| 0.5);
                    mul.push(MulSite {
                        op: k,
                        a: *a,
                        b: *b,
                        xl: cvt(xl),
                        xu: cvt(xu),
                        yl: cvt(yl),
                        yu: cvt(yu),
                        init,
                    });
                    GOp::Mul {
                        a: *a,
                        b: *b,
                        site: mul.len() - 1,
                    }
                }
                GraphSearchOp::Unary { input, kind } => {
                    let (l, u) = box_of(&boxes, *input)?;
                    let mut lower = Vec::with_capacity(l.len());
                    let mut upper = Vec::with_capacity(l.len());
                    for (&lj, &uj) in l.iter().zip(u) {
                        let (lo_line, up_line) = unary_lines(*kind, lj, uj)
                            .ok_or(Refusal::Nothing("a frozen unary relaxation is not finite"))?;
                        lower.push([R::from_f64(lo_line[0]), R::from_f64(lo_line[1])]);
                        upper.push([R::from_f64(up_line[0]), R::from_f64(up_line[1])]);
                    }
                    GOp::Lines {
                        input: *input,
                        lower,
                        upper,
                    }
                }
                GraphSearchOp::Softmax {
                    input,
                    groups,
                    len,
                    lower_a,
                    lower_b,
                    upper_a,
                    upper_b,
                } => GOp::Softmax {
                    input: *input,
                    groups: *groups,
                    len: *len,
                    la: cvt(lower_a),
                    lb: cvt(lower_b),
                    ua: cvt(upper_a),
                    ub: cvt(upper_b),
                },
            };
            ops.push(lowered);
        }
        Ok(Self {
            tsize,
            ops,
            relu,
            bil,
            mul,
            lo: cvt(lo),
            hi: cvt(hi),
            rows: rows.iter().map(|row| cvt(row)).collect(),
            thr: thresholds.to_vec(),
        })
    }

    /// The request's starting parameters.
    pub(crate) fn init_alpha(&self) -> Alpha<R> {
        Alpha {
            relu: self.relu.iter().map(|s| s.init.clone()).collect(),
            bil: self.bil.iter().map(|s| s.init.clone()).collect(),
            mul: self.mul.iter().map(|s| s.init.clone()).collect(),
        }
    }

    /// Resident bytes of the lowered program (admission).
    pub(crate) fn bytes(&self) -> u64 {
        let r = R::BYTES as u64;
        let ops: u64 = self
            .ops
            .iter()
            .map(|op| match op {
                GOp::Sparse { terms, bias, .. } => {
                    (terms.len() as u64) * (12 + r) + (bias.len() as u64) * r
                }
                GOp::Dense { w, b, .. } => ((w.len() + b.len()) as u64) * r,
                GOp::Conv {
                    gather_pk, w, b, ..
                } => (gather_pk.len() as u64) * 4 + ((w.len() + b.len()) as u64) * r,
                GOp::Lines { lower, .. } => (lower.len() as u64) * 4 * r,
                GOp::Softmax { la, lb, .. } => ((la.len() + lb.len()) as u64) * 2 * r,
                GOp::Relu { .. } | GOp::Bilinear { .. } | GOp::Mul { .. } => 0,
            })
            .fold(0u64, u64::saturating_add);
        let sites: u64 = self
            .relu
            .iter()
            .map(|s| (s.status.len() as u64) * (1 + 3 * r))
            .chain(
                self.bil
                    .iter()
                    .map(|s| ((2 * s.xl.len() + 2 * s.yl.len() + s.init.len()) as u64) * r),
            )
            .chain(
                self.mul
                    .iter()
                    .map(|s| ((4 * s.xl.len() + s.init.len()) as u64) * r),
            )
            .fold(0u64, u64::saturating_add);
        ops.saturating_add(sites)
    }

    /// Bytes of one iteration's plane tables.
    pub(crate) fn planes_bytes(&self) -> u64 {
        let elems: usize = self
            .bil
            .iter()
            .map(|s| s.batch * s.block())
            .chain(self.mul.iter().map(|s| s.xl.len()))
            .sum();
        (elems as u64).saturating_mul(6 * R::BYTES as u64)
    }

    /// Bytes of one in-flight row: the backward coefficients and linearised
    /// values of every tensor, the input corner and the row's gradient.
    pub(crate) fn row_bytes(&self, alpha_len: usize) -> u64 {
        let tensors: usize = self.tsize.iter().sum();
        ((2 * tensors + self.tsize[0] + alpha_len) as u64).saturating_mul(R::BYTES as u64)
    }

    /// The root view: the program's own sites and input box, no injection.
    pub(crate) fn root_view(&self) -> View<'_, R> {
        View {
            relu: &self.relu,
            lo: &self.lo,
            hi: &self.hi,
            inject: &[],
        }
    }

    /// The best parameters as the public hint (stable ReLU neurons carry their
    /// exact limit: 1 active, 0 inactive).
    fn hint(&self, alpha: &Alpha<R>) -> GraphAlphaHint {
        self.hint_with(&self.relu, alpha)
    }

    /// [`Self::hint`] under the ReLU sites `relu` (a domain's re-derived
    /// statuses decide which neurons carry their exact limit).
    pub(crate) fn hint_with(&self, relu: &[ReluSite<R>], alpha: &Alpha<R>) -> GraphAlphaHint {
        let f = |v: &[R]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
        GraphAlphaHint {
            relu: relu_hint(relu, &alpha.relu),
            bilinear: self
                .bil
                .iter()
                .zip(&alpha.bil)
                .map(|(s, a)| (s.op, f(a)))
                .collect(),
            mul_binary: self
                .mul
                .iter()
                .zip(&alpha.mul)
                .map(|(s, a)| (s.op, f(a)))
                .collect(),
        }
    }
}

/// The public slopes of `sites` at `alpha` (stable neurons carry their exact
/// limit: 1 active, 0 inactive).
pub(crate) fn relu_hint<R: Real>(
    sites: &[ReluSite<R>],
    alpha: &[Vec<R>],
) -> Vec<(usize, Vec<f32>)> {
    sites
        .iter()
        .zip(alpha)
        .map(|(s, a)| {
            let v: Vec<f32> = s
                .status
                .iter()
                .zip(a)
                .map(|(&st, x)| match st {
                    ACTIVE => 1.0,
                    UNSTABLE => x.to_f32(),
                    _ => 0.0,
                })
                .collect();
            (s.op, v)
        })
        .collect()
}

// ---------------------------------------------------------------------------
//  One iteration's planes
// ---------------------------------------------------------------------------

/// The McCormick planes of every product element at the current parameters.
/// Bilinear: `[batch][m][n][k]` elements (the parameters are shared over the
/// batch, the boxes are not); MulBinary: `[n]`.
pub(crate) struct Planes<R> {
    bil: Vec<Vec<[R; 6]>>,
    mul: Vec<Vec<[R; 6]>>,
}

fn to_r6<R: Real>(p: [f64; 6]) -> [R; 6] {
    [
        R::from_f64(p[0]),
        R::from_f64(p[1]),
        R::from_f64(p[2]),
        R::from_f64(p[3]),
        R::from_f64(p[4]),
        R::from_f64(p[5]),
    ]
}

pub(crate) fn planes<R: Real>(prog: &GProgram<R>, alpha: &Alpha<R>) -> Planes<R> {
    let bil = prog
        .bil
        .iter()
        .zip(&alpha.bil)
        .map(|(s, a)| {
            let block = s.block();
            let mut out = Vec::with_capacity(s.batch * block);
            for bb in 0..s.batch {
                for i in 0..s.m {
                    for j in 0..s.n {
                        for l in 0..s.k {
                            let e = (i * s.n + j) * s.k + l;
                            let (ai, bi) = (s.a_index(bb, i, l), s.b_index(bb, l, j));
                            out.push(to_r6(mccormick(
                                s.xl[ai].to_f64(),
                                s.xu[ai].to_f64(),
                                s.yl[bi].to_f64(),
                                s.yu[bi].to_f64(),
                                a[e].to_f64(),
                                a[2 * block + e].to_f64(),
                            )));
                        }
                    }
                }
            }
            out
        })
        .collect();
    let mul = prog
        .mul
        .iter()
        .zip(&alpha.mul)
        .map(|(s, a)| {
            let n = s.xl.len();
            (0..n)
                .map(|j| {
                    to_r6(mccormick(
                        s.xl[j].to_f64(),
                        s.xu[j].to_f64(),
                        s.yl[j].to_f64(),
                        s.yu[j].to_f64(),
                        a[j].to_f64(),
                        a[n + j].to_f64(),
                    ))
                })
                .collect()
        })
        .collect();
    Planes { bil, mul }
}

// ---------------------------------------------------------------------------
//  One row: backward, linearised forward, gradient
// ---------------------------------------------------------------------------

/// Propagate one op's output coefficients `lt` into its inputs; returns the
/// op's constant contribution.
fn back_op<R: Real>(
    prog: &GProgram<R>,
    view: &View<'_, R>,
    alpha: &Alpha<R>,
    planes: &Planes<R>,
    k: usize,
    lt: &[R],
    lam: &mut [Vec<R>],
) -> f64 {
    let mut cst = 0.0f64;
    match &prog.ops[k] {
        GOp::Sparse {
            inputs,
            terms,
            bias,
        } => {
            for (&v, &b) in lt.iter().zip(bias) {
                if v != R::ZERO {
                    cst += (v * b).to_f64();
                }
            }
            for &(o, slot, idx, w) in terms {
                let v = lt[o as usize];
                if v != R::ZERO {
                    lam[inputs[slot as usize]][idx as usize] += v * w;
                }
            }
        }
        GOp::Dense {
            input,
            rows,
            n_in,
            n_out,
            w,
            b,
        } => {
            let (n_in, n_out) = (*n_in, *n_out);
            for r in 0..*rows {
                for o in 0..n_out {
                    let v = lt[r * n_out + o];
                    if v == R::ZERO {
                        continue;
                    }
                    cst += (v * b[o]).to_f64();
                    let wrow = &w[o * n_in..(o + 1) * n_in];
                    let dst = &mut lam[*input][r * n_in..(r + 1) * n_in];
                    for (d, &wi) in dst.iter_mut().zip(wrow) {
                        *d += v * wi;
                    }
                }
            }
        }
        GOp::Conv {
            input,
            cout,
            k: kt,
            p,
            gather_pk,
            w,
            b,
        } => {
            let (kt, p) = (*kt, *p);
            for o in 0..*cout {
                for sp in 0..p {
                    let v = lt[o * p + sp];
                    if v == R::ZERO {
                        continue;
                    }
                    cst += (v * b[o]).to_f64();
                    let taps = &gather_pk[sp * kt..(sp + 1) * kt];
                    let wrow = &w[o * kt..(o + 1) * kt];
                    let dst = &mut lam[*input];
                    for (&g, &wi) in taps.iter().zip(wrow) {
                        if g != NONE {
                            dst[g as usize] += v * wi;
                        }
                    }
                }
            }
        }
        GOp::Relu { input, site } => {
            let s = &view.relu[*site];
            let a = &alpha.relu[*site];
            let dst = &mut lam[*input];
            for (j, &v) in lt.iter().enumerate() {
                if v == R::ZERO {
                    continue;
                }
                match s.status[j] {
                    ACTIVE => dst[j] += v,
                    UNSTABLE => {
                        if v > R::ZERO {
                            dst[j] += v * a[j];
                        } else {
                            dst[j] += v * s.slope[j];
                            cst += (v * s.icpt[j]).to_f64();
                        }
                    }
                    _ => {}
                }
            }
        }
        GOp::Bilinear { a, b, site } => {
            let s = &prog.bil[*site];
            let pl = &planes.bil[*site];
            for bb in 0..s.batch {
                for i in 0..s.m {
                    for j in 0..s.n {
                        let c = (bb * s.m + i) * s.n + j;
                        let w = lt[c] * s.scale;
                        if w == R::ZERO {
                            continue;
                        }
                        let lower = w >= R::ZERO;
                        for l in 0..s.k {
                            let e = &pl[c * s.k + l];
                            let (ax, ay, cc) = if lower {
                                (e[0], e[1], e[2])
                            } else {
                                (e[3], e[4], e[5])
                            };
                            lam[*a][s.a_index(bb, i, l)] += w * ax;
                            lam[*b][s.b_index(bb, l, j)] += w * ay;
                            cst += (w * cc).to_f64();
                        }
                    }
                }
            }
        }
        GOp::Mul { a, b, site } => {
            let pl = &planes.mul[*site];
            for (j, &v) in lt.iter().enumerate() {
                if v == R::ZERO {
                    continue;
                }
                let e = &pl[j];
                let (ax, ay, cc) = if v >= R::ZERO {
                    (e[0], e[1], e[2])
                } else {
                    (e[3], e[4], e[5])
                };
                lam[*a][j] += v * ax;
                lam[*b][j] += v * ay;
                cst += (v * cc).to_f64();
            }
        }
        GOp::Lines {
            input,
            lower,
            upper,
        } => {
            let dst = &mut lam[*input];
            for (j, &v) in lt.iter().enumerate() {
                if v == R::ZERO {
                    continue;
                }
                let line = if v >= R::ZERO { lower[j] } else { upper[j] };
                dst[j] += v * line[0];
                cst += (v * line[1]).to_f64();
            }
        }
        GOp::Softmax {
            input,
            groups,
            len,
            la,
            lb,
            ua,
            ub,
        } => {
            let len = *len;
            let dst = &mut lam[*input];
            for g in 0..*groups {
                for i in 0..len {
                    let o = g * len + i;
                    let v = lt[o];
                    if v == R::ZERO {
                        continue;
                    }
                    let (row, bc) = if v >= R::ZERO {
                        (&la[o * len..(o + 1) * len], lb[o])
                    } else {
                        (&ua[o * len..(o + 1) * len], ub[o])
                    };
                    cst += (v * bc).to_f64();
                    for (d, &coef) in dst[g * len..(g + 1) * len].iter_mut().zip(row) {
                        *d += v * coef;
                    }
                }
            }
        }
    }
    cst
}

/// Add `view`'s injections on tensor `t` into its coefficients `lt`.
fn inject_into<R: Real>(view: &View<'_, R>, t: usize, lt: &mut [R]) {
    for &(tensor, neuron, c) in view.inject {
        if tensor == t {
            lt[neuron as usize] += c;
        }
    }
}

/// The backward pass of one objective row. Fills `lam` (the coefficient of
/// every tensor, INCLUDING the view's injections) and returns the row's bound
/// (concretised at the view's input box).
pub(crate) fn backward<R: Real>(
    prog: &GProgram<R>,
    view: &View<'_, R>,
    alpha: &Alpha<R>,
    planes: &Planes<R>,
    row: &[R],
    lam: &mut [Vec<R>],
) -> f64 {
    for (t, buf) in lam.iter_mut().enumerate() {
        buf.clear();
        buf.resize(prog.tsize[t], R::ZERO);
    }
    let out = prog.ops.len();
    lam[out].copy_from_slice(row);
    let mut cst = 0.0f64;
    for k in (0..prog.ops.len()).rev() {
        let t = k + 1;
        let mut lt = std::mem::take(&mut lam[t]);
        inject_into(view, t, &mut lt);
        if lt.iter().any(|&v| v != R::ZERO) {
            cst += back_op(prog, view, alpha, planes, k, &lt, lam);
        }
        lam[t] = lt;
    }
    inject_into(view, 0, &mut lam[0]);
    let mut bound = cst;
    for (i, &v) in lam[0].iter().enumerate() {
        let x = if v >= R::ZERO { view.lo[i] } else { view.hi[i] };
        bound += (v * x).to_f64();
    }
    bound
}

/// The linearised network `f_α` (every relaxed op replaced by the line or
/// plane the backward chose, read off the sign of `lam` at its output) at the
/// point `x`; fills `val` for every tensor.
pub(crate) fn forward_lin<R: Real>(
    prog: &GProgram<R>,
    view: &View<'_, R>,
    alpha: &Alpha<R>,
    planes: &Planes<R>,
    lam: &[Vec<R>],
    x: &[R],
    val: &mut [Vec<R>],
) {
    val[0].clear();
    val[0].extend_from_slice(x);
    for k in 0..prog.ops.len() {
        let t = k + 1;
        let (before, after) = val.split_at_mut(t);
        let out = &mut after[0];
        out.clear();
        out.resize(prog.tsize[t], R::ZERO);
        let lt = &lam[t];
        match &prog.ops[k] {
            GOp::Sparse {
                inputs,
                terms,
                bias,
            } => {
                out.copy_from_slice(bias);
                for &(o, slot, idx, w) in terms {
                    out[o as usize] += w * before[inputs[slot as usize]][idx as usize];
                }
            }
            GOp::Dense {
                input,
                rows,
                n_in,
                n_out,
                w,
                b,
            } => {
                let (n_in, n_out) = (*n_in, *n_out);
                let src = &before[*input];
                for r in 0..*rows {
                    for o in 0..n_out {
                        let mut acc = b[o];
                        let wrow = &w[o * n_in..(o + 1) * n_in];
                        for (&wi, &xi) in wrow.iter().zip(&src[r * n_in..(r + 1) * n_in]) {
                            acc += wi * xi;
                        }
                        out[r * n_out + o] = acc;
                    }
                }
            }
            GOp::Conv {
                input,
                cout,
                k: kt,
                p,
                gather_pk,
                w,
                b,
            } => {
                let (kt, p) = (*kt, *p);
                let src = &before[*input];
                for o in 0..*cout {
                    for sp in 0..p {
                        let mut acc = b[o];
                        for t in 0..kt {
                            let g = gather_pk[sp * kt + t];
                            if g != NONE {
                                acc += w[o * kt + t] * src[g as usize];
                            }
                        }
                        out[o * p + sp] = acc;
                    }
                }
            }
            GOp::Relu { input, site } => {
                let s = &view.relu[*site];
                let a = &alpha.relu[*site];
                let src = &before[*input];
                for j in 0..src.len() {
                    out[j] = match s.status[j] {
                        ACTIVE => src[j],
                        UNSTABLE => {
                            if lt[j] > R::ZERO {
                                a[j] * src[j]
                            } else {
                                s.slope[j] * src[j] + s.icpt[j]
                            }
                        }
                        _ => R::ZERO,
                    };
                }
            }
            GOp::Bilinear { a, b, site } => {
                let s = &prog.bil[*site];
                let pl = &planes.bil[*site];
                let (va, vb) = (&before[*a], &before[*b]);
                for bb in 0..s.batch {
                    for i in 0..s.m {
                        for j in 0..s.n {
                            let c = (bb * s.m + i) * s.n + j;
                            let lower = lt[c] * s.scale >= R::ZERO;
                            let mut acc = R::ZERO;
                            for l in 0..s.k {
                                let e = &pl[c * s.k + l];
                                let (ax, ay, cc) = if lower {
                                    (e[0], e[1], e[2])
                                } else {
                                    (e[3], e[4], e[5])
                                };
                                acc += ax * va[s.a_index(bb, i, l)]
                                    + ay * vb[s.b_index(bb, l, j)]
                                    + cc;
                            }
                            out[c] = s.scale * acc;
                        }
                    }
                }
            }
            GOp::Mul { a, b, site } => {
                let pl = &planes.mul[*site];
                let (va, vb) = (&before[*a], &before[*b]);
                for j in 0..out.len() {
                    let e = &pl[j];
                    let (ax, ay, cc) = if lt[j] >= R::ZERO {
                        (e[0], e[1], e[2])
                    } else {
                        (e[3], e[4], e[5])
                    };
                    out[j] = ax * va[j] + ay * vb[j] + cc;
                }
            }
            GOp::Lines {
                input,
                lower,
                upper,
            } => {
                let src = &before[*input];
                for j in 0..src.len() {
                    let line = if lt[j] >= R::ZERO { lower[j] } else { upper[j] };
                    out[j] = line[0] * src[j] + line[1];
                }
            }
            GOp::Softmax {
                input,
                groups,
                len,
                la,
                lb,
                ua,
                ub,
            } => {
                let len = *len;
                let src = &before[*input];
                for g in 0..*groups {
                    let seg = &src[g * len..(g + 1) * len];
                    for i in 0..len {
                        let o = g * len + i;
                        let (row, bc) = if lt[o] >= R::ZERO {
                            (&la[o * len..(o + 1) * len], lb[o])
                        } else {
                            (&ua[o * len..(o + 1) * len], ub[o])
                        };
                        let mut acc = bc;
                        for (&coef, &xv) in row.iter().zip(seg) {
                            acc += coef * xv;
                        }
                        out[o] = acc;
                    }
                }
            }
        }
    }
}

/// The row's gradient with respect to every searched parameter (envelope
/// rule, exact for frozen boxes; see the module docs). `g` must be zero.
pub(crate) fn gradient<R: Real>(
    prog: &GProgram<R>,
    view: &View<'_, R>,
    lam: &[Vec<R>],
    val: &[Vec<R>],
    g: &mut Alpha<R>,
) {
    for (s, gs) in view.relu.iter().zip(g.relu.iter_mut()) {
        let lt = &lam[s.op + 1];
        let x = &val[s.input];
        for (j, gj) in gs.iter_mut().enumerate() {
            if s.status[j] == UNSTABLE && lt[j] > R::ZERO {
                *gj += lt[j] * x[j];
            }
        }
    }
    for (s, gs) in prog.bil.iter().zip(g.bil.iter_mut()) {
        let lt = &lam[s.op + 1];
        let (va, vb) = (&val[s.a], &val[s.b]);
        let block = s.block();
        for bb in 0..s.batch {
            for i in 0..s.m {
                for j in 0..s.n {
                    let c = (bb * s.m + i) * s.n + j;
                    let w = (lt[c] * s.scale).to_f64();
                    if w == 0.0 {
                        continue;
                    }
                    let lower = w > 0.0;
                    for l in 0..s.k {
                        let (ai, bi) = (s.a_index(bb, i, l), s.b_index(bb, l, j));
                        let d = plane_derivative(
                            s.xl[ai].to_f64(),
                            s.xu[ai].to_f64(),
                            s.yl[bi].to_f64(),
                            s.yu[bi].to_f64(),
                            va[ai].to_f64(),
                            vb[bi].to_f64(),
                            lower,
                        );
                        let e = (i * s.n + j) * s.k + l;
                        let slot = if lower { e } else { 2 * block + e };
                        gs[slot] += R::from_f64(w * d);
                    }
                }
            }
        }
    }
    for (s, gs) in prog.mul.iter().zip(g.mul.iter_mut()) {
        let lt = &lam[s.op + 1];
        let (va, vb) = (&val[s.a], &val[s.b]);
        let n = s.xl.len();
        for j in 0..n {
            let v = lt[j].to_f64();
            if v == 0.0 {
                continue;
            }
            let lower = v > 0.0;
            let d = plane_derivative(
                s.xl[j].to_f64(),
                s.xu[j].to_f64(),
                s.yl[j].to_f64(),
                s.yu[j].to_f64(),
                va[j].to_f64(),
                vb[j].to_f64(),
                lower,
            );
            let slot = if lower { j } else { n + j };
            gs[slot] += R::from_f64(v * d);
        }
    }
}

/// One row's outputs.
pub(crate) struct RowOut<R> {
    pub(crate) bound: f64,
    /// `c · f_α(x*) + Σ_i c_i ẑ_i(x*)`: the linearised objective (with the
    /// view's injections) at the backward's input corner (equals `bound` up
    /// to rounding; a test oracle).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) lin_value: f64,
    pub(crate) grad: Option<Alpha<R>>,
    /// `∂ bound / ∂ c_i` per injection of the view (envelope rule: the
    /// injected tensor's linearised value `ẑ_i(x*)`); empty for the root
    /// view or a non-finite bound.
    pub(crate) inject_grad: Vec<f64>,
}

/// Evaluate row `r` at `alpha` (bound, linearised value and gradient) under
/// `view`.
pub(crate) fn eval_row<R: Real>(
    prog: &GProgram<R>,
    view: &View<'_, R>,
    alpha: &Alpha<R>,
    planes: &Planes<R>,
    r: usize,
) -> RowOut<R> {
    let n = prog.tsize.len();
    let mut lam: Vec<Vec<R>> = vec![Vec::new(); n];
    let bound = backward(prog, view, alpha, planes, &prog.rows[r], &mut lam);
    if !bound.is_finite() {
        return RowOut {
            bound,
            lin_value: f64::NAN,
            grad: None,
            inject_grad: Vec::new(),
        };
    }
    let x: Vec<R> = lam[0]
        .iter()
        .enumerate()
        .map(|(i, &v)| if v >= R::ZERO { view.lo[i] } else { view.hi[i] })
        .collect();
    let mut val: Vec<Vec<R>> = vec![Vec::new(); n];
    forward_lin(prog, view, alpha, planes, &lam, &x, &mut val);
    let inject_grad: Vec<f64> = view
        .inject
        .iter()
        .map(|&(t, j, _)| val[t][j as usize].to_f64())
        .collect();
    let lin_value: f64 = prog.rows[r]
        .iter()
        .zip(&val[prog.ops.len()])
        .map(|(&c, &y)| (c * y).to_f64())
        .sum::<f64>()
        + view
            .inject
            .iter()
            .zip(&inject_grad)
            .map(|(&(_, _, c), &z)| c.to_f64() * z)
            .sum::<f64>();
    let mut g = alpha.zeros_like();
    gradient(prog, view, &lam, &val, &mut g);
    RowOut {
        bound,
        lin_value,
        grad: Some(g),
        inject_grad,
    }
}

// ---------------------------------------------------------------------------
//  The search
// ---------------------------------------------------------------------------

/// The lane's ranking of a set of row bounds: the hinge `Σ min(0, slack)`,
/// then the worst slack. A NaN slack is the worst possible slack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Score {
    pub(crate) hinge: f64,
    pub(crate) min_slack: f64,
    pub(crate) verified: usize,
}

pub(crate) fn score(bounds: &[f64], thr: &[f64]) -> Score {
    let mut hinge = 0.0f64;
    let mut min_slack = f64::INFINITY;
    let mut verified = 0usize;
    for (&b, &t) in bounds.iter().zip(thr) {
        let slack = b - t;
        let slack = if slack.is_nan() {
            f64::NEG_INFINITY
        } else {
            slack
        };
        if slack > 0.0 {
            verified += 1;
        }
        hinge += slack.min(0.0);
        min_slack = min_slack.min(slack);
    }
    Score {
        hinge,
        min_slack,
        verified,
    }
}

pub(crate) fn improves(candidate: Score, incumbent: Score) -> bool {
    candidate.hinge > incumbent.hinge
        || (candidate.hinge == incumbent.hinge && candidate.min_slack > incumbent.min_slack)
}

/// One full pass: every row's bound, and the sum (in ROW ORDER) of the
/// gradients of the rows not above their threshold.
pub(crate) struct Pass<R> {
    pub(crate) bounds: Vec<f64>,
    pub(crate) grad: Alpha<R>,
}

/// Evaluate every row, `threads` rows at a time, folding gradients in row
/// order after each chunk (independent of the thread count and chunking).
pub(crate) fn eval_pass<R: Real>(
    prog: &GProgram<R>,
    view: &View<'_, R>,
    alpha: &Alpha<R>,
    planes: &Planes<R>,
    threads: usize,
) -> Pass<R> {
    let n = prog.rows.len();
    let threads = threads.clamp(1, n.max(1));
    let mut bounds = vec![f64::NAN; n];
    let mut grad = alpha.zeros_like();
    let mut start = 0usize;
    while start < n {
        let end = (start + threads).min(n);
        let mut outs: Vec<Option<RowOut<R>>> = (start..end).map(|_| None).collect();
        if end - start == 1 {
            outs[0] = Some(eval_row(prog, view, alpha, planes, start));
        } else {
            std::thread::scope(|scope| {
                for (q, slot) in outs.iter_mut().enumerate() {
                    let r = start + q;
                    scope.spawn(move || {
                        *slot = Some(eval_row(prog, view, alpha, planes, r));
                    });
                }
            });
        }
        for (q, out) in outs.into_iter().enumerate() {
            let r = start + q;
            let Some(out) = out else {
                continue;
            };
            bounds[r] = out.bound;
            let slack = out.bound - prog.thr[r];
            if slack.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                if let Some(g) = out.grad.as_ref() {
                    grad.add_finite(g);
                }
            }
        }
        start = end;
    }
    Pass { bounds, grad }
}

/// Projected Adam ASCENT on the free parameters: unstable ReLU slopes, the
/// lower-direction planes `[0]`/`[2]` of every product and both MulBinary
/// planes. Everything else keeps its value.
pub(crate) struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
}

impl Adam {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    /// One step; `relu` decides which slopes are free (its UNSTABLE neurons).
    pub(crate) fn step<R: Real>(
        &mut self,
        prog: &GProgram<R>,
        relu: &[ReluSite<R>],
        alpha: &mut Alpha<R>,
        grad: &Alpha<R>,
        lr: f64,
        t: usize,
    ) {
        let t = i32::try_from(t.max(1)).unwrap_or(i32::MAX);
        let bc1 = 1.0 - ADAM_B1.powi(t);
        let bc2 = 1.0 - ADAM_B2.powi(t);
        let mut at = 0usize;
        let (m, v) = (&mut self.m, &mut self.v);
        let mut update = |a: &mut R, g: R, free: bool, at: usize| {
            if !free {
                return;
            }
            let g = g.to_f64();
            let g = if g.is_finite() { g } else { 0.0 };
            m[at] = ADAM_B1 * m[at] + (1.0 - ADAM_B1) * g;
            v[at] = ADAM_B2 * v[at] + (1.0 - ADAM_B2) * g * g;
            let step = lr * (m[at] / bc1) / ((v[at] / bc2).sqrt() + ADAM_EPS);
            let next = (a.to_f64() + step).clamp(0.0, 1.0);
            if next.is_finite() {
                *a = R::from_f64(next);
            }
        };
        for ((s, a), g) in relu.iter().zip(&mut alpha.relu).zip(&grad.relu) {
            for j in 0..a.len() {
                update(&mut a[j], g[j], s.status[j] == UNSTABLE, at);
                at += 1;
            }
        }
        for ((s, a), g) in prog.bil.iter().zip(&mut alpha.bil).zip(&grad.bil) {
            let block = s.block();
            for e in 0..a.len() {
                let plane = e / block.max(1);
                update(&mut a[e], g[e], plane == 0 || plane == 2, at);
                at += 1;
            }
        }
        for (a, g) in alpha.mul.iter_mut().zip(&grad.mul) {
            for j in 0..a.len() {
                update(&mut a[j], g[j], true, at);
                at += 1;
            }
        }
    }
}

/// Search controls.
pub(crate) struct GConfig {
    pub(crate) iters: usize,
    pub(crate) lr: f64,
    pub(crate) deadline: Option<Instant>,
    pub(crate) threads: usize,
    pub(crate) mem_cap: u64,
}

/// What [`search`] found.
pub(crate) struct GResult<R> {
    /// The returned parameters in the search scalar (tests compare them).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) best: Alpha<R>,
    pub(crate) hint: GraphAlphaHint,
    pub(crate) iters: usize,
    pub(crate) evals: usize,
    pub(crate) init: Score,
    pub(crate) best_score: Score,
    pub(crate) threads: usize,
    pub(crate) stop: &'static str,
}

/// Run the search from the request's starting parameters.
pub(crate) fn search<R: Real>(prog: &GProgram<R>, cfg: &GConfig) -> Result<GResult<R>, Refusal> {
    if cfg.deadline.is_some_and(|d| Instant::now() >= d) {
        return Err(Refusal::Deadline);
    }
    let mut alpha = prog.init_alpha();
    let alpha_len = alpha.len();
    let n_rows = prog.rows.len();
    // Resident: the program, one iteration's planes, the current and best
    // parameters, the gradient accumulator (R) and the Adam moments (f64).
    let fixed = prog
        .bytes()
        .saturating_add(prog.planes_bytes())
        .saturating_add((alpha_len as u64).saturating_mul(3 * R::BYTES as u64 + 16));
    let per_row = prog.row_bytes(alpha_len);
    let mut threads = cfg.threads.clamp(1, n_rows.max(1));
    while threads > 1 && fixed.saturating_add(per_row.saturating_mul(threads as u64)) > cfg.mem_cap
    {
        threads -= 1;
    }
    let need = fixed.saturating_add(per_row.saturating_mul(threads as u64));
    if need > cfg.mem_cap {
        return Err(Refusal::Memory {
            need,
            cap: cfg.mem_cap,
        });
    }

    let view = prog.root_view();
    let mut pass = eval_pass(prog, &view, &alpha, &planes(prog, &alpha), threads);
    let init = score(&pass.bounds, &prog.thr);
    let mut best = alpha.clone();
    let mut best_score = init;
    let mut adam = Adam::new(alpha_len);
    let mut lr = cfg.lr;
    let mut iters = 0usize;
    let mut evals = 1usize;
    let mut no_improve = 0usize;
    let stop = loop {
        let current = score(&pass.bounds, &prog.thr);
        if current.verified == n_rows {
            break "verified";
        }
        if iters >= cfg.iters {
            break "iters";
        }
        if no_improve >= PATIENCE {
            break "patience";
        }
        if cfg.deadline.is_some_and(|d| Instant::now() >= d) {
            break "deadline";
        }
        adam.step(prog, &prog.relu, &mut alpha, &pass.grad, lr, iters + 1);
        lr *= LR_DECAY;
        iters += 1;
        pass = eval_pass(prog, &view, &alpha, &planes(prog, &alpha), threads);
        evals += 1;
        let s = score(&pass.bounds, &prog.thr);
        if improves(s, best_score) {
            best = alpha.clone();
            best_score = s;
            no_improve = 0;
        } else {
            no_improve += 1;
        }
    };
    let hint = prog.hint(&best);
    Ok(GResult {
        best,
        hint,
        iters,
        evals,
        init,
        best_score,
        threads,
        stop,
    })
}

/// Lower, search in `R`, and wrap the result as untrusted outputs.
pub(crate) fn run_graph_root<R: Real>(
    net: &GraphSearchNet,
    request: &GraphRootRequest,
    threads: usize,
) -> Result<GraphRootOutcome, Refusal> {
    let t0 = Instant::now();
    let prog = GProgram::<R>::lower(net, request)?;
    let cfg = GConfig {
        iters: request.iters,
        lr: if request.lr.is_finite() && request.lr > 0.0 {
            request.lr
        } else {
            DEFAULT_LR
        },
        deadline: request.deadline,
        threads,
        mem_cap: request.mem_cap_bytes,
    };
    let result = search(&prog, &cfg)?;
    Ok(GraphRootOutcome {
        alpha: Untrusted::new(result.hint),
        stats: GraphRootStats {
            iters: result.iters,
            evals: result.evals,
            search_secs: t0.elapsed().as_secs_f64(),
            init_slack: Untrusted::new(result.init.min_slack),
            best_slack: Untrusted::new(result.best_score.min_slack),
            rows: prog.rows.len(),
            verified_init: result.init.verified,
            verified_best: result.best_score.verified,
            relu_sites: prog.relu.len(),
            bilinear_sites: prog.bil.len(),
            mul_sites: prog.mul.len(),
            threads: result.threads,
            stop: result.stop,
        },
    })
}
