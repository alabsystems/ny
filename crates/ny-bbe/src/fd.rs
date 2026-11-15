// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M4 — the frozen-dual LEVEL SEARCH, UNTRUSTED, on the CPU
//! (ENGINE_PROGRAM §2 M4; `margin_row.frozen_dual_bab_search: cpu_f32`).
//!
//! # What it computes
//!
//! For every node of one BFS level chunk (all nodes share the depth `D`), the
//! frozen-dual tier's per-node search step — the formulas of `ny-propagate`'s
//! `zono_bab::eval_search_node`, in the search scalar `R`:
//!
//! 1. the node value `g(L) = c0 + sum c0s - sum_j |rc_j| - sum (L0 b0 + L1 b1)`
//!    with `rc = d_base + sum_i (rho_i + sign_i (L0_i - L1_i)) a_i` at the WARM
//!    start, its absolute-term sum `S` and its supergradient;
//! 2. the slack-CLAMPED point, the projected supergradient and its direction
//!    `da` in `rc` space, the log-bucket line search (the step the children
//!    inherit) and the clamped FAR probe;
//! 3. FAR-FIRST: the far point is scored next; the clamped and FLOOR points
//!    are scored only for a node neither the warm nor the far point already
//!    scores clearly positive (`v > tau`, `tau = 4 gamma_f32(D) S`), because
//!    such a node will most likely certify and then has no children;
//! 4. the children's warm start (one line-search step while the node is not
//!    already positive at the warm, clamped or floor point).
//!
//! It returns, per node, the value and `S` at each point, the clamped and far
//! multipliers and the children's warm start ([`LevelOut`]).
//!
//! # Trust
//!
//! UNTRUSTED end to end: f32 (f64 in the in-crate reference instantiation),
//! round to nearest, any summation order (column tiles are folded as partial
//! sums), no rounding charged, a polynomial `ln` and `exp` in the line search
//! (platform-independent, so the f32 search is the same bits on every host and
//! ny-cuda's kernels can reproduce it bit for bit — `fast_exp_f32`). The lane
//! re-scores a proposed point with its own certified f64 evaluator at the SAME
//! multipliers (every non-negative point is a valid dual point) and prunes only
//! on that; a node it cannot certify is split. A bug here costs time.
//!
//! # Cost shape
//!
//! Nodes run in blocks of [`FD_LANES`] (SIMD across nodes: every row
//! coefficient is broadcast against a lane vector of multipliers), columns in
//! tiles of [`FD_TILE`] so a block's `rc` tile and the rows' tile segments
//! stay in L1/L2 while all `D` rows stream through. Blocks are independent and
//! scattered back in node order, and every lane's sums are formed row by row
//! (see `axpy2`), so a node's output depends on neither the thread count nor
//! which nodes share its block — hence not on how the lane chunks a level
//! (R6; `fd_search_is_thread_count_and_chunk_independent` pins both, with rows
//! that are all-zero in some blocks and live in others).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ny_core::search::{
    FdSearchProblem, LevelChunk, LevelOut, LevelStats, Refusal, FD_CLAMPED, FD_FAR, FD_FLOOR,
    FD_LOG_BUCKETS, FD_POINTS, FD_SEARCH_TOL, FD_WARM,
};

use crate::real::Real;

/// Nodes per SIMD block.
pub(crate) const FD_LANES: usize = 16;

/// Column tile width: the shared contract [`ny_core::search::FD_SUM_TILE`]
/// (it fixes the f32 summation order the device kernels reproduce).
pub(crate) const FD_TILE: usize = ny_core::search::FD_SUM_TILE;

/// One value per lane (node) of a block.
type Lanes<R> = [R; FD_LANES];

/// The scalar operations the level search needs beyond [`Real`].
pub(crate) trait FdReal: Real {
    /// Natural log for the line search's bucket placement ONLY (f32: a
    /// branch-light polynomial, ~1e-7 relative; f64: libm, the reference).
    fn ln_search(self) -> Self;
    /// `e^x` (once per node; f32: the platform-independent
    /// [`fast_exp_f32`]; f64: libm, the reference).
    fn exp(self) -> Self;
    /// Round toward minus infinity.
    fn floor(self) -> Self;
    /// Widen a stored f32 (exact for both instantiations).
    fn from_f32(v: f32) -> Self;
}

/// `ln x` for `x` positive and normal (the line search passes
/// `max(e, 1e-12)`): exponent plus the odd series of `atanh` on the mantissa
/// reduced to `[sqrt(1/2), sqrt(2))`, truncated after `s^7` (error below
/// `2 s^9 / 9 <= 3e-8`). `+inf` maps to `+inf`, as libm's `ln` does.
#[inline(always)]
pub(crate) fn fast_ln_f32(x: f32) -> f32 {
    if x == f32::INFINITY {
        return x;
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if m > std::f32::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let series = 2.0 * s * (1.0 + s2 * (1.0 / 3.0 + s2 * (0.2 + s2 * (1.0 / 7.0))));
    e as f32 * std::f32::consts::LN_2 + series
}

/// `e^x` for the line search's step (once per node).
///
/// A Cody-Waite reduction `x = n ln2 + r` (`|r| <= ln2 / 2`, `ln2` split in a
/// 9-bit head and a tail so `n * head` is exact), a degree-6 Taylor polynomial
/// in `r` (relative error below `4e-7`) and an exact scaling by `2^n` in two
/// power-of-two factors. Built from `+ - * floor` in a fixed order, so it is
/// the SAME BITS on every IEEE binary32 platform without FMA contraction —
/// unlike the host libm's `expf` (glibc, macOS and the device's `expf` all
/// differ in the last bit). ny-cuda's kernels mirror this function operation
/// for operation, which is what lets their known-answer tests demand the CPU
/// search's bits. `NaN -> NaN`; above `88.72283` -> `+inf`; below
/// `-87.33654` -> `0` (a flushed step that small is irrelevant to the search).
#[inline(always)]
pub(crate) fn fast_exp_f32(x: f32) -> f32 {
    const LOG2E: f32 = f32::from_bits(0x3FB8_AA3B);
    const LN2_HI: f32 = f32::from_bits(0x3F31_8000);
    const LN2_LO: f32 = f32::from_bits(0xB95E_8083);
    const HI: f32 = f32::from_bits(0x42B1_7217);
    const LO: f32 = f32::from_bits(0xC2AE_AC4F);
    const C3: f32 = f32::from_bits(0x3E2A_AAAB);
    const C4: f32 = f32::from_bits(0x3D2A_AAAB);
    const C5: f32 = f32::from_bits(0x3C08_8889);
    const C6: f32 = f32::from_bits(0x3AB6_0B61);
    if x.is_nan() {
        return x;
    }
    if x > HI {
        return f32::INFINITY;
    }
    if x < LO {
        return 0.0;
    }
    let nf = (x * LOG2E + 0.5).floor();
    let r = (x - nf * LN2_HI) - nf * LN2_LO;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (C3 + r * (C4 + r * (C5 + r * C6)))));
    // `nf` is an integer in `-126..=128`, so both halves are in `-63..=64`.
    let n = nf as i32;
    let h = n / 2;
    p * pow2_f32(h) * pow2_f32(n - h)
}

/// `2^k` for `k` in `-126..=127` (exact, from the exponent bits).
#[inline(always)]
fn pow2_f32(k: i32) -> f32 {
    f32::from_bits(((k + 127) as u32) << 23)
}

impl FdReal for f32 {
    #[inline(always)]
    fn ln_search(self) -> Self {
        fast_ln_f32(self)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        fast_exp_f32(self)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f32::floor(self)
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl FdReal for f64 {
    #[inline(always)]
    fn ln_search(self) -> Self {
        f64::ln(self)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f64::floor(self)
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        f64::from(v)
    }
}

/// `gamma_k` of binary32 (`k u / (1 - k u)`, `u = 2^-24`).
pub(crate) fn gamma_f32(k: usize) -> f64 {
    let ku = k as f64 * f64::from(f32::EPSILON) / 2.0;
    if ku < 1.0 {
        ku / (1.0 - ku)
    } else {
        f64::INFINITY
    }
}

/// The relative search tolerance of a depth-`d` node: a value above
/// `tau_rel(d) * S` is "clearly positive" for the far-first skip. The lane
/// uses the same `4 gamma_f32(D)` to decide which nodes to certify.
pub(crate) fn tau_rel(depth: usize) -> f64 {
    4.0 * gamma_f32(depth)
}

/// A proposed multiplier as stored search state: the f64 search's `narrow`
/// (NaN and non-positive -> 0, beyond `f32::MAX` -> `f32::MAX`), so a step
/// that overflows f32 still proposes a finite, non-negative point.
#[inline(always)]
fn store<R: FdReal>(v: R) -> f32 {
    let x = v.to_f32();
    if x.is_nan() || x <= 0.0 {
        0.0
    } else if x >= f32::MAX {
        f32::MAX
    } else {
        x
    }
}

/// Sanitise a stored multiplier into the search's domain: finite and `>= 0`
/// (a NaN, an infinity or a negative becomes `0`, exactly the lane's `widen`).
#[inline(always)]
fn sanitize<R: FdReal>(v: f32) -> R {
    if v.is_finite() && v > 0.0 {
        R::from_f32(v)
    } else {
        R::ZERO
    }
}

// ---------------------------------------------------------------------------
//  The uploaded search copy
// ---------------------------------------------------------------------------

/// One frozen-dual problem in the search scalar.
#[derive(Debug, Clone)]
pub(crate) struct FdProgram<R> {
    pub(crate) n: usize,
    pub(crate) depth: usize,
    a: Vec<R>,
    enc: Vec<usize>,
    rho: Vec<[R; 2]>,
    c0s: Vec<[R; 2]>,
    b0: Vec<[R; 2]>,
    b1: Vec<[R; 2]>,
    row_abs: Vec<R>,
    d: Vec<R>,
    c0: R,
    abs_d: R,
}

impl<R: FdReal> FdProgram<R> {
    /// Lower a validated search copy (fails closed on an invalid one).
    pub(crate) fn lower(p: &FdSearchProblem) -> Result<Self, Refusal> {
        p.validate().map_err(Refusal::Invalid)?;
        let pair = |v: &[[f32; 2]]| -> Vec<[R; 2]> {
            v.iter()
                .map(|x| [R::from_f32(x[0]), R::from_f32(x[1])])
                .collect()
        };
        Ok(Self {
            n: p.n,
            depth: p.depth,
            a: p.a.iter().map(|&v| R::from_f32(v)).collect(),
            enc: p.enc.iter().map(|&e| e as usize).collect(),
            rho: pair(&p.rho),
            c0s: pair(&p.c0s),
            b0: pair(&p.b0),
            b1: pair(&p.b1),
            row_abs: p.row_abs.iter().map(|&v| R::from_f32(v)).collect(),
            d: p.d.iter().map(|&v| R::from_f32(v)).collect(),
            c0: R::from_f32(p.c0),
            abs_d: R::from_f32(p.abs_d),
        })
    }

    /// Resident bytes.
    pub(crate) fn bytes(&self) -> u64 {
        let r = R::BYTES as u64;
        let words = (self.a.len() + self.d.len() + self.row_abs.len() + 8 * self.depth) as u64;
        words
            .saturating_mul(r)
            .saturating_add((self.enc.len() * size_of::<usize>()) as u64)
    }

    /// The initial `rc` of every node of `depth` splits: `d` with the split
    /// rows' own columns zeroed.
    fn d_base(&self, depth: usize) -> Vec<R> {
        let mut base = self.d.clone();
        for &e in &self.enc[..depth] {
            base[e] = R::ZERO;
        }
        base
    }

    fn row(&self, i: usize, j0: usize, j1: usize) -> &[R] {
        &self.a[i * self.n + j0..i * self.n + j1]
    }
}

/// Per-thread working-set bytes of the level search at `(n, depth)`.
pub(crate) fn work_bytes<R: FdReal>(n: usize, depth: usize) -> u64 {
    let lanes = FD_LANES as u64;
    let r = R::BYTES as u64;
    let lane_rows = (2 * n as u64)
        .saturating_add(FD_TILE as u64)
        .saturating_add(2 * depth as u64);
    let node_rows = 11 * depth as u64;
    lane_rows
        .saturating_mul(lanes)
        .saturating_add(node_rows.saturating_mul(lanes))
        .saturating_add(lanes * FD_LOG_BUCKETS as u64)
        .saturating_mul(r)
        .saturating_add(lanes.saturating_mul(depth as u64))
        .saturating_add(64 * 1024)
}

// ---------------------------------------------------------------------------
//  Per-thread buffers
// ---------------------------------------------------------------------------

/// Which multipliers a gathered lane is scored at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Point {
    Clamped,
    Floor,
    Far,
}

/// A block's per-node rows (`FD_LANES x depth`, node-major) and scalars.
struct Nodes<R> {
    sides: Vec<u8>,
    l0: Vec<R>,
    l1: Vec<R>,
    s0: Vec<R>,
    s1: Vec<R>,
    c0: Vec<R>,
    c1: Vec<R>,
    f0: Vec<R>,
    f1: Vec<R>,
    sp0: Vec<R>,
    sp1: Vec<R>,
    zero: Vec<R>,
    active: [bool; FD_LANES],
    clamped_differs: [bool; FD_LANES],
    any_mult: [bool; FD_LANES],
    far_differs: [bool; FD_LANES],
    slope0: Lanes<R>,
    eta: Lanes<R>,
    /// `(value, S)` per point, NaN when not evaluated.
    val: [[R; FD_POINTS]; FD_LANES],
    abs: [[R; FD_POINTS]; FD_LANES],
}

/// Lane-major scratch.
struct Scratch<R> {
    mult: Vec<Lanes<R>>,
    p: Vec<Lanes<R>>,
    rc: Vec<Lanes<R>>,
    aux: Vec<Lanes<R>>,
    tile: Vec<Lanes<R>>,
    buckets: Vec<R>,
}

struct Work<R> {
    depth: usize,
    nodes: Nodes<R>,
    s: Scratch<R>,
}

impl<R: FdReal> Work<R> {
    fn new(n: usize, depth: usize) -> Self {
        let ld = || vec![R::ZERO; FD_LANES * depth];
        let nan = R::from_f64(f64::NAN);
        Self {
            depth,
            nodes: Nodes {
                sides: vec![0; FD_LANES * depth],
                l0: ld(),
                l1: ld(),
                s0: ld(),
                s1: ld(),
                c0: ld(),
                c1: ld(),
                f0: ld(),
                f1: ld(),
                sp0: ld(),
                sp1: ld(),
                zero: vec![R::ZERO; depth],
                active: [false; FD_LANES],
                clamped_differs: [false; FD_LANES],
                any_mult: [false; FD_LANES],
                far_differs: [false; FD_LANES],
                slope0: [R::ZERO; FD_LANES],
                eta: [R::ZERO; FD_LANES],
                val: [[nan; FD_POINTS]; FD_LANES],
                abs: [[nan; FD_POINTS]; FD_LANES],
            },
            s: Scratch {
                mult: vec![[R::ZERO; FD_LANES]; depth],
                p: vec![[R::ZERO; FD_LANES]; depth],
                rc: vec![[R::ZERO; FD_LANES]; n],
                aux: vec![[R::ZERO; FD_LANES]; n],
                tile: vec![[R::ZERO; FD_LANES]; FD_TILE.min(n).max(1)],
                buckets: vec![R::ZERO; FD_LANES * FD_LOG_BUCKETS],
            },
        }
    }
}

/// Bytes a fresh workspace of `(n, depth)` reserves (the memory-admission
/// test compares it against [`work_bytes`]).
#[cfg(test)]
pub(crate) fn workspace_capacity_bytes<R: FdReal>(n: usize, depth: usize) -> u64 {
    let w = Work::<R>::new(n, depth);
    let r = R::BYTES as u64;
    let lane = (FD_LANES as u64) * r;
    let nd = &w.nodes;
    let node_rows = [
        &nd.l0, &nd.l1, &nd.s0, &nd.s1, &nd.c0, &nd.c1, &nd.f0, &nd.f1, &nd.sp0, &nd.sp1, &nd.zero,
    ]
    .iter()
    .map(|v| v.capacity() as u64 * r)
    .sum::<u64>();
    let lane_rows = [&w.s.mult, &w.s.p, &w.s.rc, &w.s.aux, &w.s.tile]
        .iter()
        .map(|v| v.capacity() as u64 * lane)
        .sum::<u64>();
    node_rows
        + lane_rows
        + w.s.buckets.capacity() as u64 * r
        + nd.sides.capacity() as u64
        + size_of::<Nodes<R>>() as u64
}

// ---------------------------------------------------------------------------
//  The kernels
// ---------------------------------------------------------------------------

/// The `rc`-independent sums of one scored point.
#[derive(Clone, Copy)]
struct Sums<R> {
    c0path: R,
    c0abs: R,
    lag: R,
    lagabs: R,
    tsum: R,
}

/// Row multipliers and prologue sums of lane `q` at `(m0, m1)` (`None` on a
/// non-finite multiplier: the point is refused). Writes `mult[i][q]`.
#[inline]
fn prologue<R: FdReal>(
    prog: &FdProgram<R>,
    sides: &[u8],
    m0: &[R],
    m1: &[R],
    mult: &mut [Lanes<R>],
    q: usize,
) -> Option<Sums<R>> {
    let mut s = Sums {
        c0path: prog.c0,
        c0abs: prog.c0.abs(),
        lag: R::ZERO,
        lagabs: R::ZERO,
        tsum: R::ZERO,
    };
    for (i, row) in mult.iter_mut().enumerate() {
        let a = m0[i].max(R::ZERO);
        let b = m1[i].max(R::ZERO);
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let si = usize::from(sides[i]);
        let sign = if si == 0 { R::ONE } else { -R::ONE };
        let rho = prog.rho[i][si];
        row[q] = rho + sign * (a - b);
        let cs = prog.c0s[i][si];
        s.c0path += cs;
        s.c0abs += cs.abs();
        let b0 = prog.b0[i][si];
        let b1 = prog.b1[i][si];
        s.lag += a * b0 + b * b1;
        s.lagabs += a * b0.abs() + b * b1.abs();
        s.tsum += (rho.abs() + a + b) * prog.row_abs[i];
    }
    Some(s)
}

/// `t[j][q] += m[q] a_j` (one row).
#[inline(always)]
fn axpy<R: Real>(t: &mut [Lanes<R>], r: &[R], m: &Lanes<R>) {
    for (x, &av) in t.iter_mut().zip(r) {
        for (xq, &mq) in x.iter_mut().zip(m) {
            *xq += mq * av;
        }
    }
}

/// `t[j][q] += m0[q] a0_j`, then `t[j][q] += m1[q] a1_j`: two rows per pass
/// over the tile, but each lane's sum is still formed ROW BY ROW, exactly as
/// two [`axpy`] calls would. That is what makes a lane's result independent of
/// the other nodes in its block: [`accumulate`] skips a row only when it is
/// zero in EVERY lane and pairs the rows it keeps, so which rows are visited
/// and how they pair depends on the block's composition — but a row that is
/// zero in this lane adds an exact zero here (`x + 0 a = x`, up to the sign of
/// a zero, which no output reads), so skipping or visiting it and pairing it
/// either way leave the lane's bits unchanged. Folding the pair first
/// (`x + (m0 a0 + m1 a1)`) would round differently per pairing and make a
/// node's search depend on how the lane chunks a level.
#[inline(always)]
fn axpy2<R: Real>(t: &mut [Lanes<R>], r0: &[R], m0: &Lanes<R>, r1: &[R], m1: &Lanes<R>) {
    for ((x, &a0), &a1) in t.iter_mut().zip(r0).zip(r1) {
        for ((xq, &p), &q) in x.iter_mut().zip(m0).zip(m1) {
            *xq = (*xq + p * a0) + q * a1;
        }
    }
}

/// Add every row of `mult` (lane-major) over columns `j0..j1` into `t`.
#[inline]
fn accumulate<R: FdReal>(
    prog: &FdProgram<R>,
    mult: &[Lanes<R>],
    j0: usize,
    j1: usize,
    t: &mut [Lanes<R>],
) {
    let zero = |m: &Lanes<R>| m.iter().all(|&v| v == R::ZERO);
    let depth = mult.len();
    let mut i = 0usize;
    while i < depth {
        if zero(&mult[i]) {
            i += 1;
            continue;
        }
        if i + 1 < depth && !zero(&mult[i + 1]) {
            axpy2(
                t,
                prog.row(i, j0, j1),
                &mult[i],
                prog.row(i + 1, j0, j1),
                &mult[i + 1],
            );
            i += 2;
        } else {
            axpy(t, prog.row(i, j0, j1), &mult[i]);
            i += 1;
        }
    }
}

/// `sum |t|` per lane (one tile, a partial sum).
#[inline(always)]
fn tile_abs<R: FdReal>(t: &[Lanes<R>]) -> Lanes<R> {
    let mut part = [R::ZERO; FD_LANES];
    for x in t {
        for (pq, &xq) in part.iter_mut().zip(x) {
            *pq += xq.abs();
        }
    }
    part
}

/// Score lane `q`'s prologue `sums` with `sum |rc| = sumabs`: `(value, S)`.
#[inline]
fn finish<R: FdReal>(prog: &FdProgram<R>, s: &Sums<R>, sumabs: R) -> (R, R) {
    let value = s.c0path - sumabs - s.lag;
    let big_s = s.c0abs + sumabs + prog.abs_d + s.tsum + s.lagabs;
    (value, big_s)
}

/// The WARM point of every loaded node: value, `S`, the `rc` rows (kept in
/// `scratch.rc` for the line search) and the supergradient (`s0`, `s1`).
fn eval_warm<R: FdReal>(prog: &FdProgram<R>, d_base: &[R], nb: usize, w: &mut Work<R>) -> u64 {
    let d = w.depth;
    let n = prog.n;
    let nodes = &mut w.nodes;
    let s = &mut w.s;
    let mut sums: [Option<Sums<R>>; FD_LANES] = [None; FD_LANES];
    for (q, slot) in sums.iter_mut().enumerate() {
        if q < nb {
            let r = q * d..(q + 1) * d;
            *slot = prologue(
                prog,
                &nodes.sides[r.clone()],
                &nodes.l0[r.clone()],
                &nodes.l1[r],
                &mut s.mult,
                q,
            );
        }
        if slot.is_none() {
            for m in &mut s.mult {
                m[q] = R::ZERO;
            }
        }
    }
    let mut sumabs = [R::ZERO; FD_LANES];
    s.p.fill([R::ZERO; FD_LANES]);
    let mut j0 = 0usize;
    while j0 < n {
        let j1 = (j0 + FD_TILE).min(n);
        let t = &mut s.rc[j0..j1];
        for (x, &dv) in t.iter_mut().zip(&d_base[j0..j1]) {
            *x = [dv; FD_LANES];
        }
        accumulate(prog, &s.mult, j0, j1, t);
        let part = tile_abs(t);
        for (a, &b) in sumabs.iter_mut().zip(&part) {
            *a += b;
        }
        // Supergradient fold: p_i += sum_j (rc_j < 0 ? a_ij : -a_ij).
        for (i, pi) in s.p.iter_mut().enumerate() {
            let row = prog.row(i, j0, j1);
            let mut part = [R::ZERO; FD_LANES];
            for (x, &av) in t.iter().zip(row) {
                let nav = -av;
                for (pq, &xq) in part.iter_mut().zip(x) {
                    *pq += if xq < R::ZERO { av } else { nav };
                }
            }
            for (a, &b) in pi.iter_mut().zip(&part) {
                *a += b;
            }
        }
        j0 = j1;
    }
    let nan = R::from_f64(f64::NAN);
    let mut refused = 0u64;
    for q in 0..FD_LANES {
        nodes.val[q] = [nan; FD_POINTS];
        nodes.abs[q] = [nan; FD_POINTS];
        nodes.active[q] = false;
        let Some(sm) = sums[q].filter(|_| q < nb) else {
            refused += u64::from(q < nb);
            continue;
        };
        let (value, big_s) = finish(prog, &sm, sumabs[q]);
        if !value.is_finite() || !big_s.is_finite() {
            refused += 1;
            continue;
        }
        nodes.active[q] = true;
        nodes.val[q][FD_WARM] = value;
        nodes.abs[q][FD_WARM] = big_s;
        for i in 0..d {
            let si = usize::from(nodes.sides[q * d + i]);
            let sign = if si == 0 { R::ONE } else { -R::ONE };
            let pv = s.p[i][q];
            nodes.s0[q * d + i] = sign * pv - prog.b0[i][si];
            nodes.s1[q * d + i] = -sign * pv - prog.b1[i][si];
        }
    }
    refused
}

/// Score gathered `(node, point)` lanes (at most [`FD_LANES`]): value and `S`
/// into `nodes.val` / `nodes.abs`. Returns the number scored.
fn eval_points<R: FdReal>(
    prog: &FdProgram<R>,
    d_base: &[R],
    lanes: &[(usize, Point)],
    w: &mut Work<R>,
) -> u64 {
    debug_assert!(lanes.len() <= FD_LANES);
    let d = w.depth;
    let n = prog.n;
    let nodes = &mut w.nodes;
    let s = &mut w.s;
    let mut sums: [Option<Sums<R>>; FD_LANES] = [None; FD_LANES];
    for (q, slot) in sums.iter_mut().enumerate() {
        if let Some(&(b, point)) = lanes.get(q) {
            let r = b * d..(b + 1) * d;
            let (m0, m1): (&[R], &[R]) = match point {
                Point::Clamped => (&nodes.c0[r.clone()], &nodes.c1[r.clone()]),
                Point::Floor => (&nodes.zero, &nodes.zero),
                Point::Far => (&nodes.f0[r.clone()], &nodes.f1[r.clone()]),
            };
            *slot = prologue(prog, &nodes.sides[r], m0, m1, &mut s.mult, q);
        }
        if slot.is_none() {
            for m in &mut s.mult {
                m[q] = R::ZERO;
            }
        }
    }
    let mut sumabs = [R::ZERO; FD_LANES];
    let mut j0 = 0usize;
    while j0 < n {
        let j1 = (j0 + FD_TILE).min(n);
        let t = &mut s.tile[..j1 - j0];
        for (x, &dv) in t.iter_mut().zip(&d_base[j0..j1]) {
            *x = [dv; FD_LANES];
        }
        accumulate(prog, &s.mult, j0, j1, t);
        let part = tile_abs(t);
        for (a, &b) in sumabs.iter_mut().zip(&part) {
            *a += b;
        }
        j0 = j1;
    }
    for (q, &(b, point)) in lanes.iter().enumerate() {
        let at = match point {
            Point::Clamped => FD_CLAMPED,
            Point::Floor => FD_FLOOR,
            Point::Far => FD_FAR,
        };
        if let Some(sm) = sums[q] {
            let (value, big_s) = finish(prog, &sm, sumabs[q]);
            if value.is_finite() && big_s.is_finite() {
                nodes.val[b][at] = value;
                nodes.abs[b][at] = big_s;
            }
        }
    }
    lanes.len() as u64
}

/// The projected supergradient, the slack-clamped point and the line-search
/// direction's multipliers of every active node (`eval_search_node` (2) and
/// its slope loop); writes the direction into `scratch.mult`.
fn step_prepare<R: FdReal>(w: &mut Work<R>) {
    let d = w.depth;
    let tol = R::from_f64(FD_SEARCH_TOL);
    let nodes = &mut w.nodes;
    w.s.mult.fill([R::ZERO; FD_LANES]);
    for q in 0..FD_LANES {
        if !nodes.active[q] {
            continue;
        }
        let mut clamped_differs = false;
        let mut any_mult = false;
        let mut slope0 = R::ZERO;
        for i in 0..d {
            let k = q * d + i;
            let (l0, l1, s0, s1) = (nodes.l0[k], nodes.l1[k], nodes.s0[k], nodes.s1[k]);
            let a = if s0 < R::ZERO { R::ZERO } else { l0 };
            let c = if s1 < R::ZERO { R::ZERO } else { l1 };
            clamped_differs |= a != l0 || c != l1;
            any_mult |= l0 != R::ZERO || l1 != R::ZERO;
            nodes.c0[k] = a;
            nodes.c1[k] = c;
            let p0 = if l0 <= tol && s0 < R::ZERO {
                R::ZERO
            } else {
                s0
            };
            let p1 = if l1 <= tol && s1 < R::ZERO {
                R::ZERO
            } else {
                s1
            };
            nodes.sp0[k] = p0;
            nodes.sp1[k] = p1;
            slope0 += p0 * s0 + p1 * s1;
            let sign = if nodes.sides[k] == 0 { R::ONE } else { -R::ONE };
            w.s.mult[i][q] = sign * (p0 - p1);
        }
        nodes.clamped_differs[q] = clamped_differs;
        nodes.any_mult[q] = any_mult;
        nodes.slope0[q] = slope0;
    }
}

/// The direction `da` (from 0, the rows at `scratch.mult`), the log-bucket
/// line search of every active node (`zono_bab::log_bucket_step`) and the
/// clamped far point. Consumes `scratch.rc` (overwritten with breakpoints).
fn line_search<R: FdReal>(prog: &FdProgram<R>, w: &mut Work<R>) {
    let n = prog.n;
    let d = w.depth;
    let nodes = &mut w.nodes;
    let s = &mut w.s;
    let two = R::from_f64(2.0);
    let mut emin = [R::INFINITY; FD_LANES];
    let mut emax = [R::NEG_INFINITY; FD_LANES];
    // Pass 1, per tile: da, then each column's breakpoint `e = -rc / da` and
    // weight `2 |da|` where the step crosses a kink (weight 0 elsewhere).
    let mut j0 = 0usize;
    while j0 < n {
        let j1 = (j0 + FD_TILE).min(n);
        let t = &mut s.aux[j0..j1];
        for x in t.iter_mut() {
            *x = [R::ZERO; FD_LANES];
        }
        accumulate(prog, &s.mult, j0, j1, t);
        for (rcj, auxj) in s.rc[j0..j1].iter_mut().zip(t.iter_mut()) {
            for q in 0..FD_LANES {
                let r = rcj[q];
                let v = auxj[q];
                let crosses = (v > R::ZERO && r < R::ZERO) || (v < R::ZERO && r > R::ZERO);
                if crosses {
                    let e = -r / v;
                    emin[q] = emin[q].min(e);
                    emax[q] = emax[q].max(e);
                    rcj[q] = e;
                    auxj[q] = two * v.abs();
                } else {
                    auxj[q] = R::ZERO;
                }
            }
        }
        j0 = j1;
    }
    let floor_e = R::from_f64(1e-12);
    let kb = R::from_f64(FD_LOG_BUCKETS as f64);
    let top = FD_LOG_BUCKETS - 1;
    let mut le = [R::ZERO; FD_LANES];
    let mut lw = [R::ONE; FD_LANES];
    for q in 0..FD_LANES {
        emin[q] = emin[q].max(floor_e);
        emax[q] = emax[q].max(floor_e);
        le[q] = emin[q].ln_search();
        lw[q] = (emax[q].ln_search() - le[q]).max(R::from_f64(1e-30)) / kb;
    }
    // Pass 2: the weighted histogram of the breakpoints' log positions.
    s.buckets.fill(R::ZERO);
    for (rcj, auxj) in s.rc.iter().zip(s.aux.iter()) {
        for q in 0..FD_LANES {
            let wgt = auxj[q];
            if wgt > R::ZERO {
                let raw = (rcj[q].max(floor_e).ln_search() - le[q]) / lw[q];
                let f = raw.floor();
                // NaN and non-positive -> 0, like the f64 search.
                let idx = if f > R::ZERO {
                    if f >= R::from_f64(top as f64) {
                        top
                    } else {
                        f.to_f64() as usize
                    }
                } else {
                    0
                };
                s.buckets[q * FD_LOG_BUCKETS + idx] += wgt;
            }
        }
    }
    let tol = R::from_f64(FD_SEARCH_TOL);
    let far_scale = R::from_f64(1e6);
    for q in 0..FD_LANES {
        if !nodes.active[q] {
            continue;
        }
        let slope0 = nodes.slope0[q];
        let mut cum = R::ZERO;
        let mut first = None;
        for (k, &b) in s.buckets[q * FD_LOG_BUCKETS..(q + 1) * FD_LOG_BUCKETS]
            .iter()
            .enumerate()
        {
            cum += b;
            if cum >= slope0 {
                first = Some(k);
                break;
            }
        }
        let eta = first.map_or(emax[q], |k| (le[q] + R::from_f64(k as f64) * lw[q]).exp());
        let eta = if eta.is_finite() { eta } else { R::ZERO };
        nodes.eta[q] = eta;
        // The clamped far probe.
        let mut ebnd = R::INFINITY;
        for i in 0..d {
            let k = q * d + i;
            if nodes.sp0[k] < -tol {
                ebnd = ebnd.min(-nodes.l0[k] / nodes.sp0[k]);
            }
            if nodes.sp1[k] < -tol {
                ebnd = ebnd.min(-nodes.l1[k] / nodes.sp1[k]);
            }
        }
        let far = if ebnd.is_finite() {
            ebnd
        } else {
            emax[q] * far_scale
        };
        let mut eta_c = if first.is_some() { eta } else { far };
        if !(slope0 > tol && eta_c.is_finite()) {
            eta_c = R::ZERO;
        }
        let eta_c = eta_c.max(R::ZERO);
        let mut far_differs = false;
        for i in 0..d {
            let k = q * d + i;
            let a = (nodes.l0[k] + eta_c * nodes.sp0[k]).max(R::ZERO);
            let c = (nodes.l1[k] + eta_c * nodes.sp1[k]).max(R::ZERO);
            far_differs |= a != nodes.l0[k] || c != nodes.l1[k];
            nodes.f0[k] = a;
            nodes.f1[k] = c;
        }
        nodes.far_differs[q] = far_differs;
    }
}

/// The per-block outputs (node-major slices of the chunk's [`LevelOut`]).
struct BlockSlot<'a> {
    first: usize,
    value: &'a mut [f32],
    scale: &'a mut [f32],
    clamped: &'a mut [f32],
    far: &'a mut [f32],
    next: &'a mut [f32],
}

/// Search the nodes loaded in `w` (`nb` of them) and write their outputs.
/// Returns `(evaluations, refused nodes)`.
fn search_block<R: FdReal>(
    prog: &FdProgram<R>,
    d_base: &[R],
    nb: usize,
    w: &mut Work<R>,
    slot: &mut BlockSlot<'_>,
) -> (u64, u64) {
    let d = w.depth;
    let refused = eval_warm(prog, d_base, nb, w);
    let mut evals = (nb as u64).saturating_sub(refused);
    step_prepare(w);
    line_search(prog, w);
    // F: the far points that moved.
    let mut lanes: Vec<(usize, Point)> = Vec::with_capacity(FD_LANES);
    for b in 0..nb {
        if w.nodes.active[b] {
            if w.nodes.far_differs[b] {
                lanes.push((b, Point::Far));
            } else {
                w.nodes.val[b][FD_FAR] = w.nodes.val[b][FD_WARM];
                w.nodes.abs[b][FD_FAR] = w.nodes.abs[b][FD_WARM];
            }
        }
    }
    evals += eval_points(prog, d_base, &lanes, w);
    // C: clamped and floor, far-first (skipped on a clearly positive node).
    let rel = R::from_f64(tau_rel(d));
    let strong = |v: R, s: R| v.is_finite() && s.is_finite() && v > rel * s;
    lanes.clear();
    for b in 0..nb {
        if !w.nodes.active[b] {
            continue;
        }
        let (v, a) = (w.nodes.val[b], w.nodes.abs[b]);
        if !w.nodes.clamped_differs[b] {
            w.nodes.val[b][FD_CLAMPED] = v[FD_WARM];
            w.nodes.abs[b][FD_CLAMPED] = a[FD_WARM];
        }
        if !w.nodes.any_mult[b] {
            w.nodes.val[b][FD_FLOOR] = v[FD_WARM];
            w.nodes.abs[b][FD_FLOOR] = a[FD_WARM];
        }
        if strong(v[FD_WARM], a[FD_WARM]) || strong(v[FD_FAR], a[FD_FAR]) {
            continue;
        }
        if w.nodes.clamped_differs[b] {
            lanes.push((b, Point::Clamped));
        }
        if w.nodes.any_mult[b] {
            lanes.push((b, Point::Floor));
        }
    }
    for part in lanes.chunks(FD_LANES) {
        evals += eval_points(prog, d_base, part, w);
    }
    // Out: values, multipliers and the children's warm start.
    let tol = R::from_f64(FD_SEARCH_TOL);
    let nodes = &w.nodes;
    for b in 0..nb {
        let r = b * d..(b + 1) * d;
        let (vo, mo) = (b * FD_POINTS, b * 2 * d);
        if !nodes.active[b] {
            // Refused: no value; every proposal is the parent's point.
            for (i, k) in r.enumerate() {
                let (a, c) = (store(nodes.l0[k]), store(nodes.l1[k]));
                for buf in [&mut *slot.clamped, &mut *slot.far, &mut *slot.next] {
                    buf[mo + i] = a;
                    buf[mo + d + i] = c;
                }
            }
            continue;
        }
        for p in 0..FD_POINTS {
            slot.value[vo + p] = nodes.val[b][p].to_f32();
            slot.scale[vo + p] = nodes.abs[b][p].to_f32();
        }
        let v = nodes.val[b];
        let mut pre_best = v[FD_WARM];
        for p in [FD_CLAMPED, FD_FLOOR] {
            if v[p].is_finite() {
                pre_best = pre_best.max(v[p]);
            }
        }
        let pending = pre_best <= tol;
        let eta_s = if pending && nodes.slope0[b] > tol {
            nodes.eta[b].max(R::ZERO)
        } else {
            R::ZERO
        };
        for (i, k) in r.enumerate() {
            slot.clamped[mo + i] = store(nodes.c0[k]);
            slot.clamped[mo + d + i] = store(nodes.c1[k]);
            slot.far[mo + i] = store(nodes.f0[k]);
            slot.far[mo + d + i] = store(nodes.f1[k]);
            slot.next[mo + i] = store(nodes.l0[k] + eta_s * nodes.sp0[k]);
            slot.next[mo + d + i] = store(nodes.l1[k] + eta_s * nodes.sp1[k]);
        }
    }
    (evals, refused)
}

// ---------------------------------------------------------------------------
//  The level call
// ---------------------------------------------------------------------------

fn past(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

/// Load block `slot.first ..` of `chunk` into `w`; returns the node count.
fn load<R: FdReal>(chunk: &LevelChunk<'_>, first: usize, w: &mut Work<R>) -> usize {
    let d = chunk.depth;
    let nb = FD_LANES.min(chunk.count - first);
    let src = first * d..(first + nb) * d;
    let nodes = &mut w.nodes;
    nodes.sides[..nb * d].copy_from_slice(&chunk.sides[src.clone()]);
    for (dst, &v) in nodes.l0[..nb * d].iter_mut().zip(&chunk.l0[src.clone()]) {
        *dst = sanitize(v);
    }
    for (dst, &v) in nodes.l1[..nb * d].iter_mut().zip(&chunk.l1[src]) {
        *dst = sanitize(v);
    }
    nb
}

/// Search one level chunk into `out` on `threads` workers (see the module
/// docs). Refuses a malformed chunk, a depth beyond the upload and a passed
/// deadline; a worker panic is a refusal, never a partial result.
pub(crate) fn fd_level<R: FdReal>(
    prog: &FdProgram<R>,
    chunk: &LevelChunk<'_>,
    out: &mut LevelOut,
    threads: usize,
) -> Result<LevelStats, Refusal> {
    let t0 = Instant::now();
    chunk.validate().map_err(Refusal::Invalid)?;
    if chunk.depth > prog.depth {
        return Err(Refusal::Invalid(format!(
            "fd level chunk: depth {} beyond the uploaded {} rows",
            chunk.depth, prog.depth
        )));
    }
    if past(chunk.deadline) {
        return Err(Refusal::Deadline);
    }
    let (count, d) = (chunk.count, chunk.depth);
    out.prepare(count, d);
    if count == 0 {
        return Ok(LevelStats::default());
    }
    let d_base = prog.d_base(d);
    let w = out.writer();
    let mut slots: Vec<BlockSlot<'_>> = Vec::with_capacity(count.div_ceil(FD_LANES));
    for (k, ((((value, scale), clamped), far), next)) in w
        .value
        .chunks_mut(FD_LANES * FD_POINTS)
        .zip(w.scale.chunks_mut(FD_LANES * FD_POINTS))
        .zip(w.clamped.chunks_mut(FD_LANES * 2 * d))
        .zip(w.far.chunks_mut(FD_LANES * 2 * d))
        .zip(w.next.chunks_mut(FD_LANES * 2 * d))
        .enumerate()
    {
        slots.push(BlockSlot {
            first: k * FD_LANES,
            value,
            scale,
            clamped,
            far,
            next,
        });
    }
    let workers = threads.max(1).min(slots.len());
    let cut = AtomicBool::new(false);
    let queue = Mutex::new(slots.into_iter());
    let run = || -> (u64, u64) {
        let mut work = Work::<R>::new(prog.n, d);
        let (mut evals, mut refused) = (0u64, 0u64);
        loop {
            if cut.load(Ordering::Relaxed) {
                break;
            }
            if past(chunk.deadline) {
                cut.store(true, Ordering::Relaxed);
                break;
            }
            let next = queue.lock().ok().and_then(|mut it| it.next());
            let Some(mut slot) = next else {
                break;
            };
            let nb = load(chunk, slot.first, &mut work);
            let (e, r) = search_block(prog, &d_base, nb, &mut work, &mut slot);
            evals += e;
            refused += r;
        }
        (evals, refused)
    };
    let totals: Result<Vec<(u64, u64)>, Refusal> = if workers == 1 {
        Ok(vec![run()])
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers).map(|_| scope.spawn(&run)).collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| Refusal::Invalid("fd level search worker panicked".into()))
                })
                .collect()
        })
    };
    let totals = totals?;
    if cut.load(Ordering::Relaxed) {
        return Err(Refusal::Deadline);
    }
    Ok(LevelStats {
        nodes: count as u64,
        evals: totals.iter().map(|t| t.0).sum(),
        refused: totals.iter().map(|t| t.1).sum(),
        secs: t0.elapsed().as_secs_f64(),
        threads: workers,
    })
}

/// A backend's upload: the lowered program and its worker count.
#[derive(Debug)]
pub(crate) struct Upload<R> {
    pub(crate) prog: FdProgram<R>,
    pub(crate) threads: usize,
}
