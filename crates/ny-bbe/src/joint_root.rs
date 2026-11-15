// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M5a — abc's joint-alpha ROOT search, UNTRUSTED, on the CPU
//! (ENGINE_PROGRAM §2 M5a; `study_abc.md` §9 items 1-3 and 9).
//!
//! # The recipe
//!
//! 1. **CROWN init** (abc `init_alpha`). Every ReLU pre-activation is a START
//!    NODE. In op order, each start node's rows — only the neurons unstable in
//!    the current box (the lane's certified box when one is supplied: abc's
//!    `aux_reference_bounds`), lower AND upper — are bounded by one CROWN
//!    backward pass with the adaptive slopes `alpha = [u >= -l]` on every
//!    earlier ReLU, and the box is intersected shrink-only with the result.
//!    The neurons still unstable afterwards are the site's slope UNIVERSE (the
//!    sparse-feature alpha of abc: stable neurons carry no alpha).
//! 2. **Spec prune** (abc `prune_after_crown`). The objective rows are
//!    `y_target - y_j` on the last tensor. Classes the CROWN init already
//!    bounds above zero leave the optimisation (99 -> 1 on every abc probe row).
//! 3. **Per-start-node alpha with an in-loop refresh.** Every intermediate
//!    start node owns its own slopes for every earlier ReLU, per ROW (abc's
//!    sparse-spec alpha; shared per start node when the per-row set does not
//!    fit the memory cap), and the objective owns one slope vector. Each joint
//!    iteration walks the start nodes in op order: it re-bounds the rows still
//!    unstable with the node's current slopes against the boxes ALREADY
//!    REFRESHED this iteration, intersects the box (running best, as abc keeps
//!    the best intermediate bounds across iterations), and takes one projected
//!    Adam step on the node's slopes (lr decay 0.98 per iteration). The
//!    objective pass then bounds the surviving classes and steps its slopes on
//!    the classes not yet above zero (abc's stop-criterion loss mask).
//! 4. **Early exits.** All surviving classes above zero (`verified`), no
//!    objective improvement for [`PATIENCE`] iterations (`patience`), the
//!    iteration cap (`iters`), or the deadline (`deadline`).
//!
//! # The gradient (per start node, boxes frozen within the pass)
//!
//! A row's CROWN lower bound is `min_x L(x)` over the input box for the
//! linear function `L` the backward pass assembles. With every other line
//! choice held, `d L / d alpha_j = Lambda_j * z_j(x*)` where `Lambda_j >= 0` is
//! the backward coefficient at the ReLU output, `x*` the concretisation
//! vertex (`mid - sign(Lambda_in) * rad`) and `z_j(x*)` the pre-activation of
//! the LINEARISED network at `x*` — one forward walk per row through the same
//! lines the backward pass chose. This is abc's gradient for a start node
//! with the intermediate boxes treated as constants; the program's kill line
//! says to add the TRUE joint adjoint (through the boxes) only if S3 fails.
//!
//! # Cost shape
//!
//! Rows run in blocks of [`LANES`] (SIMD across rows); a block's rows are
//! grouped by spatial position, so they share one receptive field, and every
//! op touches only the positions the block's coefficients reach — the CPU
//! analogue of abc's patches mode. Blocks are independent, so the result
//! does not depend on the thread count (R6).
//!
//! A block's slope gradient is SPARSE (one entry per owned unstable neuron its
//! cone reaches), and a gradient pass streams its blocks in chunks sized from
//! the transient share of the admitted memory, absorbing each chunk into the
//! Adam step in block order before the next runs. So a pass holds one chunk of
//! gradients, never `rows * stride` values, and the result does not depend on
//! the chunk size either. A per-row start node updates its moments row by row
//! (each slot has exactly one row), so it needs no `slots * stride`
//! accumulator beside its admitted slopes.
//!
//! # Trust
//!
//! UNTRUSTED end to end: f32, round to nearest, no rounding charged, no
//! certified box. Its only product is a slope HINT; the lane's certify-once
//! ingest gate clamps it and one certified evaluation decides whether it is
//! kept.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use ny_core::search::{Refusal, RootSearchRequest};

use crate::program::{Op, Program, NONE};
use crate::real::Real;

/// Rows per SIMD block.
pub(crate) const LANES: usize = 8;

/// Joint iterations without an objective improvement before the search stops
/// (abc `early_stop_patience`).
pub(crate) const PATIENCE: usize = 10;

/// Per-iteration learning-rate decay (abc `ExponentialLR` 0.98).
const LR_DECAY: f64 = 0.98;

const ADAM_B1: f64 = 0.9;
const ADAM_B2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;

/// Search controls.
pub(crate) struct Config {
    /// Joint iterations after the CROWN init.
    pub(crate) iters: usize,
    /// Adam step.
    pub(crate) lr: f64,
    /// Wall deadline.
    pub(crate) deadline: Option<Instant>,
    /// Worker threads (>= 1).
    pub(crate) threads: usize,
    /// Admitted bytes (workspaces + slopes + the streamed gradient chunks).
    pub(crate) mem_cap: u64,
    /// Share intermediate slopes per start node even when per-row fits.
    pub(crate) force_shared: bool,
    /// TEST ONLY: blocks per gradient chunk (`None` = sized from the
    /// transient share of `mem_cap`, never below the worker count).
    pub(crate) chunk_blocks: Option<usize>,
}

/// One backward row: `sign * e_a` (`b == NONE`) or `e_a - e_b` (objective).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Row {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) neg: bool,
    /// Slope slot of the owning start node.
    pub(crate) slot: u32,
}

/// A ReLU site's search box, upper chords and slope universe.
pub(crate) struct SiteBox<R> {
    pub(crate) l: Vec<R>,
    pub(crate) u: Vec<R>,
    slope: Vec<R>,
    icpt: Vec<R>,
    /// Neurons carrying a slope (unstable after the CROWN init), ascending.
    pub(crate) uni: Vec<u32>,
    /// Position in `uni` per neuron, [`NONE`] outside it.
    pub(crate) pos: Vec<u32>,
}

impl<R: Real> SiteBox<R> {
    fn new(n: usize) -> Self {
        Self {
            l: vec![R::NEG_INFINITY; n],
            u: vec![R::INFINITY; n],
            slope: vec![R::ZERO; n],
            icpt: vec![R::INFINITY; n],
            uni: Vec::new(),
            pos: vec![NONE; n],
        }
    }

    #[inline]
    pub(crate) fn unstable(&self, j: usize) -> bool {
        self.l[j] < R::ZERO && self.u[j] > R::ZERO
    }

    /// CROWN's adaptive lower slope `[u >= -l]`.
    #[inline]
    fn adaptive(&self, j: usize) -> R {
        if self.u[j] >= -self.l[j] {
            R::ONE
        } else {
            R::ZERO
        }
    }

    /// Re-derive the upper chords from the current box.
    fn refresh_chords(&mut self) {
        for j in 0..self.l.len() {
            let (l, u) = (self.l[j], self.u[j]);
            let (s, c) = if !(l < R::ZERO && u > R::ZERO) {
                (R::ZERO, R::ZERO)
            } else if l.is_finite() && u.is_finite() {
                let s = u / (u - l);
                (s, -(s * l))
            } else if u.is_finite() {
                (R::ZERO, u)
            } else {
                (R::ZERO, R::INFINITY)
            };
            self.slope[j] = s;
            self.icpt[j] = c;
        }
    }
}

/// One worker's reusable buffers: a `LANES`-wide coefficient/value buffer per
/// tensor, with generation marks and the touched positions of the block.
pub(crate) struct Workspace<R> {
    buf: Vec<Vec<R>>,
    mark: Vec<Vec<u32>>,
    touched: Vec<Vec<u32>>,
    stamp: u32,
    /// `Lambda` per owned slope, SLOPE-major (`at * LANES + q`). All zero
    /// between blocks: a block zeroes exactly the entries it wrote
    /// (`lam_at`), so no block pays `O(stride)` to clear it.
    lam: Vec<R>,
    /// The slopes this block wrote into `lam`.
    lam_at: Vec<u32>,
    /// The block's SPARSE slope gradient: one `(slope, per-lane value)` per
    /// owned unstable neuron the block reaches, in forward-walk order.
    grad: Vec<(u32, [R; LANES])>,
}

impl<R: Real> Workspace<R> {
    pub(crate) fn new(tsize: &[usize]) -> Self {
        Self {
            buf: tsize.iter().map(|&n| vec![R::ZERO; n * LANES]).collect(),
            mark: tsize.iter().map(|&n| vec![0u32; n]).collect(),
            touched: tsize.iter().map(|_| Vec::new()).collect(),
            stamp: 0,
            lam: Vec::new(),
            lam_at: Vec::new(),
            grad: Vec::new(),
        }
    }

    fn next_gen(&mut self) {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            for marks in &mut self.mark {
                marks.fill(0);
            }
            self.stamp = 1;
        }
    }

    /// Zero every touched lane and forget the touched sets.
    fn clear(&mut self) {
        for t in 0..self.touched.len() {
            let mut list = std::mem::take(&mut self.touched[t]);
            let buf = &mut self.buf[t];
            for &i in &list {
                let i = i as usize;
                for q in 0..LANES {
                    buf[i * LANES + q] = R::ZERO;
                }
            }
            list.clear();
            self.touched[t] = list;
        }
    }
}

#[inline(always)]
fn lane<R: Copy>(buf: &[R], i: usize) -> [R; LANES] {
    let cell: &[R; LANES] = buf[i * LANES..(i + 1) * LANES]
        .try_into()
        .expect("lane width");
    *cell
}

#[inline(always)]
fn lane_mut<R>(buf: &mut [R], i: usize) -> &mut [R; LANES] {
    (&mut buf[i * LANES..(i + 1) * LANES])
        .try_into()
        .expect("lane width")
}

#[inline(always)]
fn touch(mark: &mut [u32], list: &mut Vec<u32>, stamp: u32, i: usize) {
    if mark[i] != stamp {
        mark[i] = stamp;
        list.push(i as u32);
    }
}

/// Everything one pass reads (shared, read-only, across the workers).
struct PassCtx<'a, R> {
    prog: &'a Program<R>,
    boxes: &'a [SiteBox<R>],
    mid: &'a [R],
    rad: &'a [R],
    /// Seeded tensor.
    start: usize,
    /// Ops `k < limit` are walked.
    limit: usize,
    /// Slope offset of every owned site (`off.len()` = sites owned).
    off: &'a [usize],
    /// Slopes per slot.
    stride: usize,
    /// `slots * stride` slopes.
    alpha: &'a [R],
    want_grad: bool,
}

/// One block's bounds and (optionally) its SPARSE per-row slope gradients:
/// `(slope, value per lane)` for every owned unstable neuron in the block's
/// receptive cone; every other slope's gradient is zero. Sparse because a
/// dense `LANES * stride` vector per block made every gradient pass cost (and
/// hold) `rows * stride` values whatever the cones' sizes.
pub(crate) struct BlockOut<R> {
    pub(crate) lb: [R; LANES],
    pub(crate) grad: Vec<(u32, [R; LANES])>,
}

/// Bound one block of rows: sparse CROWN backward, concretisation and, for
/// gradients, the linearised forward walk from the concretisation vertices.
#[allow(clippy::too_many_lines)]
fn run_block<R: Real>(ctx: &PassCtx<'_, R>, ws: &mut Workspace<R>, rows: &[Row]) -> BlockOut<R> {
    ws.next_gen();
    let stamp = ws.stamp;
    let stride = ctx.stride;
    let mut bias = [R::ZERO; LANES];
    if ctx.want_grad {
        // Grow only; the all-zero invariant makes the new tail and the old
        // entries alike zero.
        if ws.lam.len() < LANES * stride {
            ws.lam.resize(LANES * stride, R::ZERO);
        }
        ws.lam_at.clear();
        ws.grad.clear();
    }
    let mut slot_base = [0usize; LANES];
    for (q, row) in rows.iter().enumerate().take(LANES) {
        slot_base[q] = row.slot as usize * stride;
        let a = row.a as usize;
        touch(
            &mut ws.mark[ctx.start],
            &mut ws.touched[ctx.start],
            stamp,
            a,
        );
        ws.buf[ctx.start][a * LANES + q] += if row.neg { -R::ONE } else { R::ONE };
        if row.b != NONE {
            let b = row.b as usize;
            touch(
                &mut ws.mark[ctx.start],
                &mut ws.touched[ctx.start],
                stamp,
                b,
            );
            ws.buf[ctx.start][b * LANES + q] -= R::ONE;
        }
    }

    // ---- backward: op outputs -> op inputs, in reverse op order -----------
    for k in (0..ctx.limit).rev() {
        let out = k + 1;
        if ws.touched[out].is_empty() {
            continue;
        }
        let list = std::mem::take(&mut ws.touched[out]);
        let obuf = std::mem::take(&mut ws.buf[out]);
        match &ctx.prog.ops[k] {
            Op::Conv {
                input,
                k: taps,
                p,
                gather_pk,
                w,
                b,
            } => {
                let (a, taps, p) = (*input, *taps, *p);
                for &idx in &list {
                    let idx = idx as usize;
                    let (o, sp) = (idx / p, idx % p);
                    let v = lane(&obuf, idx);
                    let bo = b[o];
                    for q in 0..LANES {
                        bias[q] += bo * v[q];
                    }
                    let g_row = &gather_pk[sp * taps..(sp + 1) * taps];
                    let w_row = &w[o * taps..(o + 1) * taps];
                    for (&g, &wt) in g_row.iter().zip(w_row) {
                        if g == NONE || wt == R::ZERO {
                            continue;
                        }
                        let g = g as usize;
                        touch(&mut ws.mark[a], &mut ws.touched[a], stamp, g);
                        let d = lane_mut(&mut ws.buf[a], g);
                        for q in 0..LANES {
                            d[q] += wt * v[q];
                        }
                    }
                }
            }
            Op::Affine {
                input,
                scale,
                shift,
            } => {
                let a = *input;
                for &j in &list {
                    let j = j as usize;
                    let v = lane(&obuf, j);
                    let (sc, sh) = (scale[j], shift[j]);
                    for q in 0..LANES {
                        bias[q] += sh * v[q];
                    }
                    touch(&mut ws.mark[a], &mut ws.touched[a], stamp, j);
                    let d = lane_mut(&mut ws.buf[a], j);
                    for q in 0..LANES {
                        d[q] += sc * v[q];
                    }
                }
            }
            Op::Relu { input, site } => {
                let (a, s) = (*input, *site);
                let sb = &ctx.boxes[s];
                let owned = s < ctx.off.len();
                for &j in &list {
                    let j = j as usize;
                    let v = lane(&obuf, j);
                    touch(&mut ws.mark[a], &mut ws.touched[a], stamp, j);
                    let (l, u) = (sb.l[j], sb.u[j]);
                    if l >= R::ZERO {
                        let d = lane_mut(&mut ws.buf[a], j);
                        for q in 0..LANES {
                            d[q] += v[q];
                        }
                    } else if u <= R::ZERO {
                        // Inactive: the output is exactly zero.
                    } else {
                        let (sl, ic) = (sb.slope[j], sb.icpt[j]);
                        let p = sb.pos[j];
                        if owned && p != NONE {
                            let at = ctx.off[s] + p as usize;
                            let d = lane_mut(&mut ws.buf[a], j);
                            for q in 0..LANES {
                                let vq = v[q];
                                if vq >= R::ZERO {
                                    d[q] += ctx.alpha[slot_base[q] + at] * vq;
                                } else {
                                    d[q] += sl * vq;
                                    bias[q] += ic * vq;
                                }
                            }
                            if ctx.want_grad {
                                // Each (site, neuron) is visited once per
                                // block, so `at` is recorded once.
                                *lane_mut(&mut ws.lam, at) = v;
                                ws.lam_at.push(at as u32);
                            }
                        } else {
                            let a0 = sb.adaptive(j);
                            let d = lane_mut(&mut ws.buf[a], j);
                            for q in 0..LANES {
                                let vq = v[q];
                                if vq >= R::ZERO {
                                    d[q] += a0 * vq;
                                } else {
                                    d[q] += sl * vq;
                                    bias[q] += ic * vq;
                                }
                            }
                        }
                    }
                }
            }
            Op::Add { lhs, rhs } => {
                let (x, y) = (*lhs, *rhs);
                for &j in &list {
                    let j = j as usize;
                    let v = lane(&obuf, j);
                    touch(&mut ws.mark[x], &mut ws.touched[x], stamp, j);
                    let d = lane_mut(&mut ws.buf[x], j);
                    for q in 0..LANES {
                        d[q] += v[q];
                    }
                    touch(&mut ws.mark[y], &mut ws.touched[y], stamp, j);
                    let d = lane_mut(&mut ws.buf[y], j);
                    for q in 0..LANES {
                        d[q] += v[q];
                    }
                }
            }
            Op::Copy { input } => {
                let a = *input;
                for &j in &list {
                    let j = j as usize;
                    let v = lane(&obuf, j);
                    touch(&mut ws.mark[a], &mut ws.touched[a], stamp, j);
                    let d = lane_mut(&mut ws.buf[a], j);
                    for q in 0..LANES {
                        d[q] += v[q];
                    }
                }
            }
            Op::Gemm { input, n_in, w, b } => {
                let (a, n_in) = (*input, *n_in);
                for &o in &list {
                    let o = o as usize;
                    let v = lane(&obuf, o);
                    let bo = b[o];
                    for q in 0..LANES {
                        bias[q] += bo * v[q];
                    }
                    let w_row = &w[o * n_in..(o + 1) * n_in];
                    for (i, &wi) in w_row.iter().enumerate() {
                        if wi == R::ZERO {
                            continue;
                        }
                        touch(&mut ws.mark[a], &mut ws.touched[a], stamp, i);
                        let d = lane_mut(&mut ws.buf[a], i);
                        for q in 0..LANES {
                            d[q] += wi * v[q];
                        }
                    }
                }
            }
        }
        ws.buf[out] = obuf;
        ws.touched[out] = list;
    }

    // ---- concretise over the input box; keep x* for the forward walk ------
    let mut lb = bias;
    {
        let list = std::mem::take(&mut ws.touched[0]);
        let buf0 = &mut ws.buf[0];
        for &i in &list {
            let i = i as usize;
            let (m, r) = (ctx.mid[i], ctx.rad[i]);
            let cell = lane_mut(buf0, i);
            for q in 0..LANES {
                let v = cell[q];
                lb[q] += v * m - v.abs() * r;
                if ctx.want_grad {
                    cell[q] = if v > R::ZERO {
                        m - r
                    } else if v < R::ZERO {
                        m + r
                    } else {
                        m
                    };
                }
            }
        }
        ws.touched[0] = list;
    }

    // ---- forward: the linearised network at x*, harvesting Lambda * z -----
    if ctx.want_grad {
        for k in 0..ctx.limit {
            let out = k + 1;
            if ws.touched[out].is_empty() {
                continue;
            }
            let list = std::mem::take(&mut ws.touched[out]);
            let mut obuf = std::mem::take(&mut ws.buf[out]);
            match &ctx.prog.ops[k] {
                Op::Conv {
                    input,
                    k: taps,
                    p,
                    gather_pk,
                    w,
                    b,
                } => {
                    let (a, taps, p) = (*input, *taps, *p);
                    let src = &ws.buf[a];
                    for &idx in &list {
                        let idx = idx as usize;
                        let (o, sp) = (idx / p, idx % p);
                        let mut acc = [b[o]; LANES];
                        let g_row = &gather_pk[sp * taps..(sp + 1) * taps];
                        let w_row = &w[o * taps..(o + 1) * taps];
                        for (&g, &wt) in g_row.iter().zip(w_row) {
                            if g == NONE || wt == R::ZERO {
                                continue;
                            }
                            let z = lane(src, g as usize);
                            for q in 0..LANES {
                                acc[q] += wt * z[q];
                            }
                        }
                        *lane_mut(&mut obuf, idx) = acc;
                    }
                }
                Op::Affine {
                    input,
                    scale,
                    shift,
                } => {
                    let src = &ws.buf[*input];
                    for &j in &list {
                        let j = j as usize;
                        let z = lane(src, j);
                        let (sc, sh) = (scale[j], shift[j]);
                        let cell = lane_mut(&mut obuf, j);
                        for q in 0..LANES {
                            cell[q] = sc * z[q] + sh;
                        }
                    }
                }
                Op::Relu { input, site } => {
                    let (a, s) = (*input, *site);
                    let sb = &ctx.boxes[s];
                    let owned = s < ctx.off.len();
                    for &j in &list {
                        let j = j as usize;
                        let z = lane(&ws.buf[a], j);
                        let (l, u) = (sb.l[j], sb.u[j]);
                        let mut h = [R::ZERO; LANES];
                        if l >= R::ZERO {
                            h = z;
                        } else if u <= R::ZERO {
                            // Inactive: zero.
                        } else {
                            let (sl, ic) = (sb.slope[j], sb.icpt[j]);
                            let p = sb.pos[j];
                            if owned && p != NONE {
                                let at = ctx.off[s] + p as usize;
                                let lam = lane(&ws.lam, at);
                                let mut g = [R::ZERO; LANES];
                                for q in 0..LANES {
                                    if lam[q] >= R::ZERO {
                                        h[q] = ctx.alpha[slot_base[q] + at] * z[q];
                                        g[q] = lam[q] * z[q];
                                    } else {
                                        h[q] = sl * z[q] + ic;
                                    }
                                }
                                ws.grad.push((at as u32, g));
                            } else {
                                // Unreachable on a gradient pass (every site
                                // a start node reaches is owned and its
                                // unstable neurons are in the universe); kept
                                // consistent with the backward fallback.
                                let a0 = sb.adaptive(j);
                                for q in 0..LANES {
                                    h[q] = a0 * z[q];
                                }
                            }
                        }
                        *lane_mut(&mut obuf, j) = h;
                    }
                }
                Op::Add { lhs, rhs } => {
                    for &j in &list {
                        let j = j as usize;
                        let x = lane(&ws.buf[*lhs], j);
                        let y = lane(&ws.buf[*rhs], j);
                        let cell = lane_mut(&mut obuf, j);
                        for q in 0..LANES {
                            cell[q] = x[q] + y[q];
                        }
                    }
                }
                Op::Copy { input } => {
                    for &j in &list {
                        let j = j as usize;
                        let z = lane(&ws.buf[*input], j);
                        *lane_mut(&mut obuf, j) = z;
                    }
                }
                Op::Gemm { input, n_in, w, b } => {
                    let (a, n_in) = (*input, *n_in);
                    let src = &ws.buf[a];
                    for &o in &list {
                        let o = o as usize;
                        let mut acc = [b[o]; LANES];
                        let w_row = &w[o * n_in..(o + 1) * n_in];
                        for (i, &wi) in w_row.iter().enumerate() {
                            if wi == R::ZERO {
                                continue;
                            }
                            let z = lane(src, i);
                            for q in 0..LANES {
                                acc[q] += wi * z[q];
                            }
                        }
                        *lane_mut(&mut obuf, o) = acc;
                    }
                }
            }
            ws.buf[out] = obuf;
            ws.touched[out] = list;
        }
    }

    let grad = if ctx.want_grad {
        // Restore the all-zero invariant of `lam` for the next block.
        let written = std::mem::take(&mut ws.lam_at);
        for &at in &written {
            *lane_mut(&mut ws.lam, at as usize) = [R::ZERO; LANES];
        }
        ws.lam_at = written;
        std::mem::take(&mut ws.grad)
    } else {
        Vec::new()
    };
    ws.clear();
    BlockOut { lb, grad }
}

/// One chunk's blocks, in block order (`None` = cut by the deadline or lost
/// to a worker panic).
pub(crate) struct PassOut<R> {
    pub(crate) blocks: Vec<Option<BlockOut<R>>>,
    pub(crate) cut: bool,
    pub(crate) panicked: bool,
}

/// Run blocks `range` of `rows` on the workers (`blocks[i]` is block
/// `range.start + i`). Blocks are claimed from an atomic counter; each
/// block's result depends only on its rows, so the assembled chunk is
/// independent of the thread count and the claim order.
fn run_pass<R: Real>(
    ctx: &PassCtx<'_, R>,
    rows: &[Row],
    range: std::ops::Range<usize>,
    workspaces: &mut [Workspace<R>],
    deadline: Option<Instant>,
) -> PassOut<R> {
    let first = range.start;
    let n_blocks = range.len();
    let mut blocks: Vec<Option<BlockOut<R>>> = (0..n_blocks).map(|_| None).collect();
    if n_blocks == 0 {
        return PassOut {
            blocks,
            cut: false,
            panicked: false,
        };
    }
    let next = AtomicUsize::new(0);
    let cut = AtomicBool::new(false);
    let workers = workspaces.len().min(n_blocks).max(1);
    let results: Vec<Option<Vec<(usize, BlockOut<R>)>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = workspaces
            .iter_mut()
            .take(workers)
            .map(|ws| {
                let (next, cut) = (&next, &cut);
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        if cut.load(Ordering::Relaxed) {
                            break;
                        }
                        let local = next.fetch_add(1, Ordering::Relaxed);
                        if local >= n_blocks {
                            break;
                        }
                        if deadline.is_some_and(|d| Instant::now() >= d) {
                            cut.store(true, Ordering::Relaxed);
                            break;
                        }
                        let bi = first + local;
                        let end = ((bi + 1) * LANES).min(rows.len());
                        done.push((local, run_block(ctx, ws, &rows[bi * LANES..end])));
                    }
                    done
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().ok()).collect()
    });
    let mut panicked = false;
    for result in results {
        match result {
            Some(done) => {
                for (local, out) in done {
                    blocks[local] = Some(out);
                }
            }
            None => panicked = true,
        }
    }
    let cut = cut.into_inner() || blocks.iter().any(Option::is_none);
    PassOut {
        blocks,
        cut,
        panicked,
    }
}

/// Every row's bound after a pass (`None`: its block was cut or lost).
pub(crate) struct RowsOut<R> {
    pub(crate) lb: Vec<Option<R>>,
    pub(crate) cut: bool,
    pub(crate) panicked: bool,
    /// Gradient bytes retained at once (the largest chunk's sparse lists).
    pub(crate) held_grad_bytes: u64,
}

/// Bytes of one sparse gradient entry.
const fn grad_entry_bytes<R>() -> u64 {
    size_of::<(u32, [R; LANES])>() as u64
}

/// A pass over all of `rows` WITHOUT gradients: one chunk, since a block
/// then retains only its `LANES` bounds.
fn bound_pass<R: Real>(
    ctx: &PassCtx<'_, R>,
    rows: &[Row],
    workspaces: &mut [Workspace<R>],
    deadline: Option<Instant>,
) -> RowsOut<R> {
    debug_assert!(!ctx.want_grad, "a gradient pass must stream (grad_pass)");
    let n_blocks = rows.len().div_ceil(LANES);
    let out = run_pass(ctx, rows, 0..n_blocks, workspaces, deadline);
    let mut lb = vec![None; rows.len()];
    for (bi, blk) in out.blocks.iter().enumerate() {
        let Some(blk) = blk else {
            continue;
        };
        for q in 0..LANES {
            let r = bi * LANES + q;
            if r >= rows.len() {
                break;
            }
            lb[r] = Some(blk.lb[q]);
        }
    }
    RowsOut {
        lb,
        cut: out.cut,
        panicked: out.panicked,
        held_grad_bytes: 0,
    }
}

/// A GRADIENT pass over `rows`, run in chunks of `chunk` blocks. As each
/// chunk completes, its sparse gradients are absorbed into `step` in BLOCK
/// ORDER (row order) and dropped, so a pass holds at most one chunk of
/// gradients whatever its row count, and the result does not depend on the
/// chunk size or the thread count (R6). A cut or a lost block ends the pass
/// after its chunk's completed blocks are absorbed.
fn grad_pass<R: Real>(
    ctx: &PassCtx<'_, R>,
    rows: &[Row],
    workspaces: &mut [Workspace<R>],
    deadline: Option<Instant>,
    chunk: usize,
    step: &mut Step<'_, R>,
    keep: impl Fn(R) -> bool,
) -> RowsOut<R> {
    debug_assert!(ctx.want_grad);
    let n_blocks = rows.len().div_ceil(LANES);
    let chunk = chunk.max(1);
    let mut lb = vec![None; rows.len()];
    let (mut cut, mut panicked, mut held_grad_bytes) = (false, false, 0u64);
    let mut first = 0usize;
    while first < n_blocks {
        let end = first.saturating_add(chunk).min(n_blocks);
        let out = run_pass(ctx, rows, first..end, workspaces, deadline);
        let held: u64 = out
            .blocks
            .iter()
            .flatten()
            .map(|blk| blk.grad.len() as u64)
            .fold(0u64, u64::saturating_add);
        held_grad_bytes = held_grad_bytes.max(held.saturating_mul(grad_entry_bytes::<R>()));
        for (local, blk) in out.blocks.iter().enumerate() {
            let Some(blk) = blk else {
                continue;
            };
            let bi = first + local;
            for q in 0..LANES {
                let r = bi * LANES + q;
                let Some(row) = rows.get(r) else {
                    break;
                };
                lb[r] = Some(blk.lb[q]);
                if keep(blk.lb[q]) {
                    step.absorb(row, &blk.grad, q);
                }
            }
        }
        cut |= out.cut;
        panicked |= out.panicked;
        if out.cut || out.panicked {
            break;
        }
        first = end;
    }
    RowsOut {
        lb,
        cut,
        panicked,
        held_grad_bytes,
    }
}

/// Per-row lower bounds of a pass, as f64 (`NaN` and missing blocks read as
/// `-inf`: no information).
fn lbs_of<R: Real>(lb: &[Option<R>]) -> Vec<f64> {
    lb.iter()
        .map(|v| {
            v.map_or(f64::NEG_INFINITY, |v| {
                let v = v.to_f64();
                if v.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    v
                }
            })
        })
        .collect()
}

/// Intersect a site's box with the bounds of its rows (running best).
fn apply_box_rows<R: Real>(sb: &mut SiteBox<R>, rows: &[Row], lbs: &[Option<R>]) {
    for (row, lb) in rows.iter().zip(lbs) {
        let Some(lb) = *lb else {
            continue;
        };
        if !lb.is_finite() {
            continue;
        }
        let j = row.a as usize;
        if row.neg {
            sb.u[j] = sb.u[j].min(-lb);
        } else {
            sb.l[j] = sb.l[j].max(lb);
        }
        if sb.l[j] > sb.u[j] {
            // A crossing is f32 rounding on a (near-)point interval.
            let m = R::from_f64(0.5) * (sb.l[j] + sb.u[j]);
            sb.l[j] = m;
            sb.u[j] = m;
        }
    }
}

/// Slope offsets of the first `n_before` sites and their total.
fn offsets<R>(boxes: &[SiteBox<R>], n_before: usize) -> (Vec<usize>, usize) {
    let mut off = Vec::with_capacity(n_before);
    let mut total = 0usize;
    for sb in &boxes[..n_before] {
        off.push(total);
        total += sb.uni.len();
    }
    (off, total)
}

/// CROWN's adaptive slopes for the first `n_before` sites' universes.
fn adaptive_alpha<R: Real>(boxes: &[SiteBox<R>], n_before: usize) -> Vec<R> {
    let mut alpha = Vec::new();
    for sb in &boxes[..n_before] {
        for &j in &sb.uni {
            alpha.push(sb.adaptive(j as usize));
        }
    }
    alpha
}

/// Lower and upper rows of a site's unstable neurons, grouped so a block's
/// rows share a spatial position (one receptive field).
fn unit_rows<R: Real>(sb: &SiteBox<R>, spatial: usize, init: bool, per_row: bool) -> Vec<Row> {
    let p = spatial.max(1);
    let mut picks: Vec<(u32, u32)> = if init {
        (0..sb.l.len())
            .filter(|&j| sb.unstable(j))
            .map(|j| (j as u32, 0))
            .collect()
    } else {
        sb.uni
            .iter()
            .enumerate()
            .filter(|&(_, &j)| sb.unstable(j as usize))
            .map(|(pos, &j)| (j, pos as u32))
            .collect()
    };
    picks.sort_by_key(|&(j, _)| {
        let j = j as usize;
        (j % p, j / p)
    });
    let mut rows = Vec::with_capacity(2 * picks.len());
    for (j, pos) in picks {
        for neg in [false, true] {
            let slot = if init {
                0
            } else if per_row {
                2 * pos + u32::from(neg)
            } else {
                u32::from(neg)
            };
            rows.push(Row {
                a: j,
                b: NONE,
                neg,
                slot,
            });
        }
    }
    rows
}

/// A start node's slopes with their Adam state.
struct Node<R> {
    /// The site bounded (`sites.len()` for the objective).
    site: usize,
    off: Vec<usize>,
    stride: usize,
    per_row: bool,
    alpha: Vec<R>,
    m: Vec<R>,
    v: Vec<R>,
    steps: i32,
}

impl<R: Real> Node<R> {
    fn new(site: usize, off: Vec<usize>, stride: usize, slots: usize, init: &[R]) -> Self {
        let mut alpha = Vec::with_capacity(slots * stride);
        for _ in 0..slots {
            alpha.extend_from_slice(init);
        }
        Self {
            site,
            off,
            stride,
            per_row: false,
            m: vec![R::ZERO; alpha.len()],
            v: vec![R::ZERO; alpha.len()],
            alpha,
            steps: 0,
        }
    }
}

/// One projected Adam ASCENT step of a [`Node`], fed row by row (in row
/// order) while its gradient pass streams, then applied by
/// [`Step::finish`] once the pass is over (the pass reads the slopes, so
/// they move only afterwards).
///
/// A slot's gradient is the sum of its admitted rows' gradients. A PER-ROW
/// node's slot receives exactly one row per pass, so its Adam moments are
/// updated as that row arrives, through a one-slot scratch, and no
/// `slots * stride` accumulator is ever allocated (it would equal the whole
/// per-row slope store, outside the admission). A shared node's and the
/// objective's few slots sum their rows in `acc`.
struct Step<'a, R> {
    m: &'a mut [R],
    v: &'a mut [R],
    stride: usize,
    slots: usize,
    per_row: bool,
    /// Shared nodes: `slots * stride` summed gradients.
    acc: Vec<R>,
    /// Per-row nodes: one slot's dense gradient (all zero between rows).
    scratch: Vec<R>,
    hit: Vec<bool>,
    b1: R,
    b2: R,
    ob1: R,
    ob2: R,
}

impl<'a, R: Real> Step<'a, R> {
    fn new(m: &'a mut [R], v: &'a mut [R], stride: usize, per_row: bool) -> Self {
        let slots = m.len().checked_div(stride).unwrap_or(0);
        Self {
            m,
            v,
            stride,
            slots,
            per_row,
            acc: if per_row {
                Vec::new()
            } else {
                vec![R::ZERO; slots * stride]
            },
            scratch: if per_row {
                vec![R::ZERO; stride]
            } else {
                Vec::new()
            },
            hit: vec![false; slots],
            b1: R::from_f64(ADAM_B1),
            b2: R::from_f64(ADAM_B2),
            ob1: R::from_f64(1.0 - ADAM_B1),
            ob2: R::from_f64(1.0 - ADAM_B2),
        }
    }

    /// Absorb lane `q` of a block's sparse gradient for `row` (non-finite
    /// components contribute nothing).
    fn absorb(&mut self, row: &Row, grad: &[(u32, [R; LANES])], q: usize) {
        let slot = row.slot as usize;
        if slot >= self.slots {
            return;
        }
        let stride = self.stride;
        let base = slot * stride;
        if self.per_row {
            if self.hit[slot] {
                // `unit_rows` gives every per-row slot at most one row.
                debug_assert!(false, "per-row slot {slot} received a second row");
                return;
            }
            for &(at, g) in grad {
                if g[q].is_finite() {
                    self.scratch[at as usize] += g[q];
                }
            }
            for i in 0..stride {
                let g = self.scratch[i];
                self.m[base + i] = self.b1 * self.m[base + i] + self.ob1 * g;
                self.v[base + i] = self.b2 * self.v[base + i] + self.ob2 * g * g;
            }
            for &(at, _) in grad {
                self.scratch[at as usize] = R::ZERO;
            }
        } else {
            for &(at, g) in grad {
                if g[q].is_finite() {
                    self.acc[base + at as usize] += g[q];
                }
            }
        }
        self.hit[slot] = true;
    }

    /// Take the step on every slot that received a row: Adam (bias-corrected
    /// at the node's step count), projected onto `[0, 1]`.
    fn finish(self, alpha: &mut [R], steps: &mut i32, lr: f64) {
        *steps = steps.saturating_add(1);
        let stride = self.stride;
        if stride == 0 {
            return;
        }
        let t = *steps;
        let c1 = 1.0 - ADAM_B1.powi(t);
        let c2 = R::from_f64(1.0 - ADAM_B2.powi(t));
        let step = R::from_f64(lr / c1);
        let eps = R::from_f64(ADAM_EPS);
        let Self {
            m,
            v,
            acc,
            hit,
            per_row,
            b1,
            b2,
            ob1,
            ob2,
            ..
        } = self;
        for slot in (0..hit.len()).filter(|&slot| hit[slot]) {
            for i in slot * stride..(slot + 1) * stride {
                if !per_row {
                    let g = acc[i];
                    m[i] = b1 * m[i] + ob1 * g;
                    v[i] = b2 * v[i] + ob2 * g * g;
                }
                let update = step * m[i] / ((v[i] / c2).sqrt() + eps);
                alpha[i] = (alpha[i] + update).max(R::ZERO).min(R::ONE);
            }
        }
    }
}

/// The search's result: the hint plus telemetry, and (for tests) the state
/// the reported root bound was computed from.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct SearchResult<R> {
    pub(crate) hint_trunk: Vec<(usize, Vec<f32>)>,
    pub(crate) hint_head: Vec<f32>,
    pub(crate) crown_lb: f64,
    pub(crate) root_lb: f64,
    pub(crate) iters: usize,
    pub(crate) passes: usize,
    pub(crate) specs: usize,
    pub(crate) pruned: usize,
    pub(crate) rows: usize,
    pub(crate) share_rows: bool,
    pub(crate) threads: usize,
    pub(crate) tightened: usize,
    pub(crate) stabilized: usize,
    pub(crate) stop: &'static str,
    /// Per adversarial class: the bound at the best iteration (the CROWN-init
    /// bound for a pruned class).
    pub(crate) spec_lb: Vec<f64>,
    /// Per adversarial class: optimised (true) or pruned by the init.
    pub(crate) surviving: Vec<bool>,
    /// The objective's slopes at the best iteration (universe layout).
    pub(crate) root_alpha: Vec<R>,
    /// The boxes the best iteration's objective pass used.
    pub(crate) root_boxes: Vec<(Vec<R>, Vec<R>)>,
    /// Every site's slope universe.
    pub(crate) universes: Vec<Vec<u32>>,
    /// The most gradient bytes any pass held at once (one streamed chunk).
    pub(crate) peak_grad_bytes: u64,
    /// The share of the admitted bytes reserved for the streamed gradient
    /// chunks and a step's accumulator.
    pub(crate) transient_cap: u64,
}

fn expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

fn snapshot_boxes<R: Real>(boxes: &[SiteBox<R>]) -> Vec<(Vec<R>, Vec<R>)> {
    boxes.iter().map(|b| (b.l.clone(), b.u.clone())).collect()
}

/// Run the M5a search. See the module docs.
#[allow(clippy::too_many_lines)]
pub(crate) fn search<R: Real>(
    prog: &Program<R>,
    req: &RootSearchRequest,
    cfg: &Config,
) -> Result<SearchResult<R>, Refusal> {
    // ---- the request -------------------------------------------------------
    let n_out = prog.tsize[prog.out_tensor()];
    if req.lo.len() != prog.n_in || req.hi.len() != prog.n_in {
        return Err(Refusal::Invalid(format!(
            "input box has {}/{} sides, the net takes {}",
            req.lo.len(),
            req.hi.len(),
            prog.n_in
        )));
    }
    if req
        .lo
        .iter()
        .zip(&req.hi)
        .any(|(l, h)| !(l.is_finite() && h.is_finite() && l <= h))
    {
        return Err(Refusal::Invalid("input box is empty or not finite".into()));
    }
    if req.adv.is_empty() {
        return Err(Refusal::Nothing("no adversarial class"));
    }
    if req.target >= n_out || req.adv.iter().any(|&j| j >= n_out || j == req.target) {
        return Err(Refusal::Invalid(format!(
            "target {} / adversarial classes out of range for {n_out} outputs",
            req.target
        )));
    }
    if expired(cfg.deadline) {
        return Err(Refusal::Deadline);
    }
    let mid: Vec<R> = req
        .lo
        .iter()
        .zip(&req.hi)
        .map(|(l, h)| R::from_f64(0.5 * (l + h)))
        .collect();
    let rad: Vec<R> = req
        .lo
        .iter()
        .zip(&req.hi)
        .map(|(l, h)| R::from_f64(0.5 * (h - l)))
        .collect();

    // ---- boxes: +-inf, then the certified reference where supplied --------
    let n_sites = prog.sites.len();
    let mut boxes: Vec<SiteBox<R>> = prog.sites.iter().map(|s| SiteBox::new(s.n)).collect();
    let mut reference: Vec<Option<(Vec<R>, Vec<R>)>> = (0..n_sites).map(|_| None).collect();
    for rb in &req.ref_boxes {
        let Some(s) = prog.sites.iter().position(|site| site.op == rb.op) else {
            return Err(Refusal::Invalid(format!(
                "reference box names op {}, which is not a ReLU",
                rb.op
            )));
        };
        let n = prog.sites[s].n;
        if rb.l.len() != n || rb.u.len() != n {
            return Err(Refusal::Invalid(format!(
                "reference box of op {} has the wrong width",
                rb.op
            )));
        }
        if rb
            .l
            .iter()
            .zip(&rb.u)
            .any(|(l, u)| l.is_nan() || u.is_nan() || l > u)
        {
            return Err(Refusal::Invalid(format!(
                "reference box of op {} is crossed or NaN",
                rb.op
            )));
        }
        let l: Vec<R> = rb.l.iter().map(|&v| R::from_f64(v)).collect();
        let u: Vec<R> = rb.u.iter().map(|&v| R::from_f64(v)).collect();
        boxes[s].l.clone_from(&l);
        boxes[s].u.clone_from(&u);
        reference[s] = Some((l, u));
    }
    for sb in &mut boxes {
        sb.refresh_chords();
    }

    // ---- workers and memory -----------------------------------------------
    let max_stride: usize = prog.sites.iter().map(|s| s.n).sum();
    let per_ws = prog.workspace_bytes(LANES, max_stride).max(1);
    // The per-site state held for the whole search comes off the top: box,
    // chords, reference copy and best-iteration snapshot (8 values) plus the
    // slope-position table, per ReLU neuron.
    let state_bytes = (max_stride as u64).saturating_mul(8 * R::BYTES as u64 + 4);
    let cap = cfg.mem_cap.saturating_sub(state_bytes);
    let ws_cap = cap / 2;
    let threads = usize::try_from((ws_cap / per_ws).min(cfg.threads.max(1) as u64)).unwrap_or(1);
    if threads == 0 {
        return Err(Refusal::Memory {
            need: state_bytes.saturating_add(per_ws.saturating_mul(2)),
            cap: cfg.mem_cap,
        });
    }
    let rest = cap.saturating_sub(per_ws.saturating_mul(threads as u64));
    // A quarter of what the workspaces leave is reserved for the TRANSIENT
    // gradient state of one pass: the streamed chunk of sparse block
    // gradients plus a step's accumulator (`<= 2 * stride` values). The rest
    // holds every start node's slopes and Adam moments.
    let transient_cap = rest / 4;
    let alpha_budget = rest - transient_cap;
    // Blocks per gradient chunk. A block's sparse gradient has at most one
    // entry per owned slope (its cone can reach them all), so this many
    // worst-case blocks fit the transient share. Never below the worker
    // count: one worst-case block per worker is already inside the
    // workspace estimate (`Program::workspace_bytes`).
    let chunk_for = |stride: usize| -> usize {
        if let Some(forced) = cfg.chunk_blocks {
            return forced.max(1);
        }
        let per_block = (stride as u64)
            .saturating_mul(grad_entry_bytes::<R>())
            .max(1);
        let acc = (2 * stride as u64).saturating_mul(R::BYTES as u64);
        let fit = transient_cap.saturating_sub(acc) / per_block;
        usize::try_from(fit).unwrap_or(usize::MAX).max(threads)
    };
    let mut peak_grad_bytes = 0u64;
    let slope_bytes = |slots: usize, stride: usize| -> u64 {
        (slots as u64)
            .saturating_mul(stride as u64)
            .saturating_mul(3 * R::BYTES as u64)
    };
    let mut workspaces: Vec<Workspace<R>> =
        (0..threads).map(|_| Workspace::new(&prog.tsize)).collect();
    let mut alpha_used = 0u64;
    let mut passes = 0usize;
    let mut rows_total = 0usize;
    let mut share_rows = false;
    let mut nodes: Vec<Node<R>> = Vec::new();

    // ---- 1. CROWN init, site by site ---------------------------------------
    for s in 0..n_sites {
        let site = &prog.sites[s];
        let (off, stride) = offsets(&boxes, s);
        let init_alpha = adaptive_alpha(&boxes, s);
        let rows = unit_rows(&boxes[s], prog.spatial[site.input], true, false);
        let out = {
            let ctx = PassCtx {
                prog,
                boxes: &boxes,
                mid: &mid,
                rad: &rad,
                start: site.input,
                limit: site.op,
                off: &off,
                stride,
                alpha: &init_alpha,
                want_grad: false,
            };
            bound_pass(&ctx, &rows, &mut workspaces, cfg.deadline)
        };
        passes += 1;
        if out.panicked {
            return Err(Refusal::Invalid("a search worker panicked".into()));
        }
        if out.cut {
            return Err(Refusal::Deadline);
        }
        apply_box_rows(&mut boxes[s], &rows, &out.lb);
        boxes[s].refresh_chords();
        let uni: Vec<u32> = (0..site.n)
            .filter(|&j| boxes[s].unstable(j))
            .map(|j| j as u32)
            .collect();
        for (p, &j) in uni.iter().enumerate() {
            boxes[s].pos[j as usize] = p as u32;
        }
        boxes[s].uni = uni;
        rows_total += rows.len();
        // An intermediate start node that owns slopes (the first ReLU's
        // pre-activation is affine in the input: its init bound is exact).
        if stride > 0 && !boxes[s].uni.is_empty() {
            let per_row_slots = 2 * boxes[s].uni.len();
            let per_row = !cfg.force_shared
                && alpha_used.saturating_add(slope_bytes(per_row_slots, stride)) <= alpha_budget;
            let slots = if per_row { per_row_slots } else { 2 };
            if alpha_used.saturating_add(slope_bytes(slots, stride)) <= alpha_budget {
                share_rows |= !per_row;
                alpha_used = alpha_used.saturating_add(slope_bytes(slots, stride));
                let mut node = Node::new(s, off, stride, slots, &init_alpha);
                node.per_row = per_row;
                nodes.push(node);
            } else {
                // Not even shared slopes fit: the site keeps its init box.
                share_rows = true;
            }
        }
    }

    // ---- 2. the objective at the CROWN init, and the spec prune ------------
    let (out_off, out_stride) = offsets(&boxes, n_sites);
    if alpha_used.saturating_add(slope_bytes(1, out_stride)) > alpha_budget {
        return Err(Refusal::Memory {
            need: alpha_used.saturating_add(slope_bytes(1, out_stride)),
            cap: alpha_budget,
        });
    }
    let out_init = adaptive_alpha(&boxes, n_sites);
    let spec_rows: Vec<Row> = req
        .adv
        .iter()
        .map(|&j| Row {
            a: req.target as u32,
            b: j as u32,
            neg: false,
            slot: 0,
        })
        .collect();
    let init = {
        let ctx = PassCtx {
            prog,
            boxes: &boxes,
            mid: &mid,
            rad: &rad,
            start: prog.out_tensor(),
            limit: prog.ops.len(),
            off: &out_off,
            stride: out_stride,
            alpha: &out_init,
            want_grad: false,
        };
        bound_pass(&ctx, &spec_rows, &mut workspaces, cfg.deadline)
    };
    passes += 1;
    if init.panicked {
        return Err(Refusal::Invalid("a search worker panicked".into()));
    }
    if init.cut {
        return Err(Refusal::Deadline);
    }
    let init_lb = lbs_of(&init.lb);
    let crown_lb = init_lb.iter().copied().fold(f64::INFINITY, f64::min);
    let surviving: Vec<bool> = init_lb.iter().map(|&v| v <= 0.0).collect();
    let pruned = surviving.iter().filter(|&&keep| !keep).count();
    let surv_idx: Vec<usize> = (0..spec_rows.len()).filter(|&i| surviving[i]).collect();
    let surv_rows: Vec<Row> = surv_idx.iter().map(|&i| spec_rows[i]).collect();

    let mut out_node = Node::new(n_sites, out_off, out_stride, 1, &out_init);
    let mut best_obj = if surv_idx.is_empty() {
        f64::INFINITY
    } else {
        surv_idx
            .iter()
            .map(|&i| init_lb[i])
            .fold(f64::INFINITY, f64::min)
    };
    let mut spec_lb = init_lb;
    let mut root_alpha = out_node.alpha.clone();
    let mut root_boxes = snapshot_boxes(&boxes);

    // ---- 3-4. joint iterations with early exits ----------------------------
    let mut stop = if surv_idx.is_empty() {
        "crown_verified"
    } else {
        "iters"
    };
    let mut iters_done = 0usize;
    let mut patience = 0usize;
    if !surv_idx.is_empty() {
        'joint: for it in 1..=cfg.iters {
            if expired(cfg.deadline) {
                stop = "deadline";
                break;
            }
            let lr = cfg.lr * LR_DECAY.powi(i32::try_from(it - 1).unwrap_or(i32::MAX));
            for node in &mut nodes {
                let s = node.site;
                let site = &prog.sites[s];
                let rows = unit_rows(&boxes[s], prog.spatial[site.input], false, node.per_row);
                if rows.is_empty() {
                    continue;
                }
                let chunk = chunk_for(node.stride);
                let mut step = Step::new(&mut node.m, &mut node.v, node.stride, node.per_row);
                let out = {
                    let ctx = PassCtx {
                        prog,
                        boxes: &boxes,
                        mid: &mid,
                        rad: &rad,
                        start: site.input,
                        limit: site.op,
                        off: &node.off,
                        stride: node.stride,
                        alpha: &node.alpha,
                        want_grad: true,
                    };
                    grad_pass(
                        &ctx,
                        &rows,
                        &mut workspaces,
                        cfg.deadline,
                        chunk,
                        &mut step,
                        |_| true,
                    )
                };
                passes += 1;
                peak_grad_bytes = peak_grad_bytes.max(out.held_grad_bytes);
                if out.panicked {
                    return Err(Refusal::Invalid("a search worker panicked".into()));
                }
                apply_box_rows(&mut boxes[s], &rows, &out.lb);
                boxes[s].refresh_chords();
                step.finish(&mut node.alpha, &mut node.steps, lr);
                if out.cut {
                    stop = "deadline";
                    break 'joint;
                }
            }
            let chunk = chunk_for(out_node.stride);
            let mut step = Step::new(&mut out_node.m, &mut out_node.v, out_node.stride, false);
            let out = {
                let ctx = PassCtx {
                    prog,
                    boxes: &boxes,
                    mid: &mid,
                    rad: &rad,
                    start: prog.out_tensor(),
                    limit: prog.ops.len(),
                    off: &out_node.off,
                    stride: out_node.stride,
                    alpha: &out_node.alpha,
                    want_grad: true,
                };
                // abc's loss mask: a class already above zero contributes no
                // gradient (a NaN bound is not "above zero" and still steers).
                grad_pass(
                    &ctx,
                    &surv_rows,
                    &mut workspaces,
                    cfg.deadline,
                    chunk,
                    &mut step,
                    |lb| lb.partial_cmp(&R::ZERO) != Some(std::cmp::Ordering::Greater),
                )
            };
            passes += 1;
            peak_grad_bytes = peak_grad_bytes.max(out.held_grad_bytes);
            if out.panicked {
                return Err(Refusal::Invalid("a search worker panicked".into()));
            }
            if out.cut {
                stop = "deadline";
                break;
            }
            iters_done = it;
            let lbs = lbs_of(&out.lb);
            let obj = lbs.iter().copied().fold(f64::INFINITY, f64::min);
            if obj > best_obj {
                best_obj = obj;
                for (k, &i) in surv_idx.iter().enumerate() {
                    spec_lb[i] = lbs[k];
                }
                root_alpha.clone_from(&out_node.alpha);
                root_boxes = snapshot_boxes(&boxes);
                patience = 0;
            } else {
                patience += 1;
            }
            if lbs.iter().all(|&v| v > 0.0) {
                stop = "verified";
                break;
            }
            // The step on the rows the loss mask admitted above.
            step.finish(&mut out_node.alpha, &mut out_node.steps, lr);
            if patience >= PATIENCE {
                stop = "patience";
                break;
            }
        }
    }

    // ---- the hint: the objective's best slopes, exact limits when stable ---
    let mut hint_trunk = Vec::new();
    let mut hint_head = Vec::new();
    for (s, site) in prog.sites.iter().enumerate() {
        let sb = &boxes[s];
        let slopes: Vec<f32> = (0..site.n)
            .map(|j| {
                if sb.l[j] >= R::ZERO {
                    1.0
                } else if sb.u[j] <= R::ZERO {
                    0.0
                } else if sb.pos[j] != NONE {
                    root_alpha[out_node.off[s] + sb.pos[j] as usize].to_f32()
                } else {
                    sb.adaptive(j).to_f32()
                }
            })
            .collect();
        if site.head {
            hint_head = slopes;
        } else {
            hint_trunk.push((site.op, slopes));
        }
    }

    // ---- telemetry ---------------------------------------------------------
    let mut tightened = 0usize;
    let mut stabilized = 0usize;
    for (s, rbox) in reference.iter().enumerate() {
        let Some((rl, ru)) = rbox else {
            continue;
        };
        let sb = &boxes[s];
        for j in 0..rl.len() {
            tightened += usize::from(sb.l[j] > rl[j]) + usize::from(sb.u[j] < ru[j]);
            stabilized += usize::from(rl[j] < R::ZERO && ru[j] > R::ZERO && !sb.unstable(j));
        }
    }
    let root_lb = spec_lb.iter().copied().fold(f64::INFINITY, f64::min);
    Ok(SearchResult {
        hint_trunk,
        hint_head,
        crown_lb,
        root_lb,
        iters: iters_done,
        passes,
        specs: spec_rows.len(),
        pruned,
        rows: rows_total,
        share_rows,
        threads,
        tightened,
        stabilized,
        stop,
        spec_lb,
        surviving,
        root_alpha,
        root_boxes,
        universes: boxes.iter().map(|b| b.uni.clone()).collect(),
        peak_grad_bytes,
        transient_cap,
    })
}

/// TEST SURFACE: bound `rows` (each `(a, b, neg)`) of one start node — site
/// `start` (`None` = the objective) — with the given boxes, universes and
/// ONE slot of slopes, returning each row's bound and slope gradient.
#[cfg(test)]
pub(crate) fn eval_rows_for_test<R: Real>(
    prog: &Program<R>,
    lo: &[f64],
    hi: &[f64],
    boxes: &[(Vec<R>, Vec<R>)],
    universes: &[Vec<u32>],
    start: Option<usize>,
    rows: &[(u32, u32, bool)],
    alpha: &[R],
) -> Vec<(f64, Vec<R>)> {
    let mut site_boxes: Vec<SiteBox<R>> = Vec::with_capacity(boxes.len());
    for (s, (l, u)) in boxes.iter().enumerate() {
        let mut sb = SiteBox::new(l.len());
        sb.l.clone_from(l);
        sb.u.clone_from(u);
        sb.refresh_chords();
        for (p, &j) in universes[s].iter().enumerate() {
            sb.pos[j as usize] = p as u32;
        }
        sb.uni.clone_from(&universes[s]);
        site_boxes.push(sb);
    }
    let n_before = start.unwrap_or(prog.sites.len());
    let (off, stride) = offsets(&site_boxes, n_before);
    assert_eq!(alpha.len(), stride, "one slot of slopes");
    let mid: Vec<R> = lo
        .iter()
        .zip(hi)
        .map(|(l, h)| R::from_f64(0.5 * (l + h)))
        .collect();
    let rad: Vec<R> = lo
        .iter()
        .zip(hi)
        .map(|(l, h)| R::from_f64(0.5 * (h - l)))
        .collect();
    let (seed, limit) = match start {
        Some(s) => (prog.sites[s].input, prog.sites[s].op),
        None => (prog.out_tensor(), prog.ops.len()),
    };
    let rows: Vec<Row> = rows
        .iter()
        .map(|&(a, b, neg)| Row { a, b, neg, slot: 0 })
        .collect();
    let ctx = PassCtx {
        prog,
        boxes: &site_boxes,
        mid: &mid,
        rad: &rad,
        start: seed,
        limit,
        off: &off,
        stride,
        alpha,
        want_grad: true,
    };
    let mut workspaces = vec![Workspace::new(&prog.tsize)];
    let out = run_pass(
        &ctx,
        &rows,
        0..rows.len().div_ceil(LANES),
        &mut workspaces,
        None,
    );
    assert!(!out.cut && !out.panicked);
    (0..rows.len())
        .map(|r| {
            let blk = out.blocks[r / LANES].as_ref().expect("block");
            let q = r % LANES;
            // Densify the sparse gradient (absent slopes have gradient 0).
            let mut grad = vec![R::ZERO; stride];
            for &(at, g) in &blk.grad {
                grad[at as usize] = g[q];
            }
            (blk.lb[q].to_f64(), grad)
        })
        .collect()
}
