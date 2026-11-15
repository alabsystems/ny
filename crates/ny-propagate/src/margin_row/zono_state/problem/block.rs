// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#fd-block-eval` (ENGINE_PROGRAM M1): the frozen-dual node bound of
//! [`FrozenDualProblem::eval_node`], evaluated for a BLOCK of same-depth
//! nodes at once.
//!
//! # Contract: bit-identical to `eval_node`; no fused multiply-add
//!
//! For every node, [`FrozenDualProblem::eval_block`] returns the SAME bits
//! (`value`, `certified`, `charge` by `to_bits`) as `eval_node` on that node,
//! the same `rc` row and the same subgradient, and refuses exactly the nodes
//! `eval_node` refuses. It adds NO certificate arithmetic: the certificate is
//! still the one `eval_node` computes, so a bug here can only break the
//! equality the tests assert, never loosen a bound silently. Any change in
//! this file must keep `fd_block_eval_bit_identical_*` green.
//!
//! Why the bits cannot move:
//!
//! * the per-row PROLOGUE and the scalar EPILOGUE are the very functions
//!   `eval_node` runs (`prologue_row`, `finish_node` in `problem.rs`);
//! * the `rc` accumulation adds, for every element `j`, the rows in `i`
//!   order, skipping a zero multiplier exactly as `eval_node` does; tiling
//!   the columns and interleaving the nodes of a block never reorders any
//!   single element's sequence of operations;
//! * the two folds (`sum_j |rc_j|` and each row's `sum_j (rc_j < 0 ? a : -a)`)
//!   are sequential left folds in increasing `j`, seeded with the seed of
//!   `Iterator::sum` ([`sum_seed`]); carrying the accumulator across column
//!   tiles in increasing `j` reproduces them exactly. Interleaving the
//!   independent `(node, row)` chains is the whole speed-up: it removes the
//!   add-latency bound of one long dependent chain per row;
//! * Rust never contracts `a * b + c` into a fused multiply-add, and
//!   vectorising across independent lanes or columns changes no element.
//!
//! # Layout
//!
//! Lane-major: one `[f64; BLOCK_NODES]` per column (the `rc` tile), per row
//! (multipliers, subgradient accumulators), so every per-lane loop has a
//! constant trip count. Lanes past the block's node count, and lanes of nodes
//! refused in their prologue, carry zero multipliers and are never read back.

use super::{FrozenDualProblem, NodeBound, NodeSums, Side, Subgradient};

/// Nodes evaluated together (the 128-node work item is 8 blocks).
pub(crate) const BLOCK_NODES: usize = 16;

/// Column tile width: a `D x T` row tile (~57 KB at `D = 28`) plus the
/// `T x B` `rc` tile (32 KB) stay cache-resident while `B` nodes reuse them.
pub(crate) const BLOCK_COLS: usize = 256;

/// One value per lane (node) of a block.
pub(crate) type Lanes = [f64; BLOCK_NODES];

/// Block geometry (the tests sweep it; the lane uses [`BlockGeom::LANE`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockGeom {
    /// Nodes evaluated together, clamped to `1..=BLOCK_NODES`.
    pub nodes: usize,
    /// Column tile width, clamped to `>= 1`.
    pub cols: usize,
}

impl BlockGeom {
    /// The geometry the search uses.
    pub(crate) const LANE: Self = Self {
        nodes: BLOCK_NODES,
        cols: BLOCK_COLS,
    };
}

/// Optional outputs of [`FrozenDualProblem::eval_block`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockWant {
    /// Materialise every node's `rc` row ([`BlockScratch::rc_row`]).
    pub rc: bool,
    /// Write every node's subgradient (`s0` / `s1` rows).
    pub subgrad: bool,
}

/// How one split row acts on a block's ACTIVE lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowKind {
    /// Every active lane's multiplier is (signed) zero: the row changes no
    /// active lane (the reference skips it on every node).
    Skip,
    /// No active lane needs the skip: `x + m a` on every lane.
    Plain,
    /// Some active lane has a zero multiplier: select per lane, so that lane
    /// keeps its accumulator's bits (a `-0.0` included) exactly as the
    /// reference's skip does.
    Select,
}

/// Reusable per-thread buffers of the block evaluator.
#[derive(Debug, Clone, Default)]
pub(crate) struct BlockScratch {
    /// `d` with the call depth's own columns zeroed (the initial `rc`).
    d_base: Vec<f64>,
    /// Row multipliers, `depth` rows of lanes.
    mult: Vec<Lanes>,
    /// Per-row action over the active lanes.
    kind: Vec<RowKind>,
    /// Subgradient accumulators, `depth` rows of lanes.
    p: Vec<Lanes>,
    /// The `rc` column tile.
    tile: Vec<Lanes>,
    /// Materialised `rc` rows (`count x n`), written ONLY by calls with
    /// `want.rc`; other calls leave it untouched.
    rc: Vec<f64>,
}

impl BlockScratch {
    /// Buffers sized for blocks of `count` nodes of `depth` splits over `n`
    /// generators, so the lane never grows them.
    pub(crate) fn with_capacity(n: usize, depth: usize, count: usize) -> Self {
        Self {
            d_base: Vec::with_capacity(n),
            mult: Vec::with_capacity(depth),
            kind: Vec::with_capacity(depth),
            p: Vec::with_capacity(depth),
            tile: Vec::with_capacity(BLOCK_COLS.min(n).max(1)),
            rc: Vec::with_capacity(count.saturating_mul(n)),
        }
    }

    /// Node `b`'s `rc` row from the last call with `want.rc` (complete only
    /// when that node's `value` was finite; SEARCH SIGNAL ONLY, like
    /// `NodeScratch::rc`).
    pub(crate) fn rc_row(&self, b: usize, n: usize) -> &[f64] {
        &self.rc[b * n..(b + 1) * n]
    }

    /// Bytes currently reserved by the buffers (the memory-admission test
    /// compares this against the search's per-thread estimate).
    #[cfg(test)]
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let f = size_of::<f64>() as u64;
        let l = size_of::<Lanes>() as u64;
        let k = size_of::<RowKind>() as u64;
        let floats = (self.d_base.capacity() as u64).saturating_add(self.rc.capacity() as u64);
        let lanes = (self.mult.capacity() as u64)
            .saturating_add(self.p.capacity() as u64)
            .saturating_add(self.tile.capacity() as u64);
        floats
            .saturating_mul(f)
            .saturating_add(lanes.saturating_mul(l))
            .saturating_add((self.kind.capacity() as u64).saturating_mul(k))
    }
}

/// The seed `Iterator::sum` starts an `f64` fold from on this toolchain
/// (`-0.0` since the float `Sum` change, `+0.0` before it): every fold here
/// starts from it, so an empty or all-`-0.0` fold matches `.sum()` bit for
/// bit whichever the toolchain uses (`fd_block_sum_seed_matches_iterator_sum`
/// pins it).
#[inline]
pub(crate) fn sum_seed() -> f64 {
    std::iter::empty::<f64>().sum()
}

/// Row `i`'s columns `j0..j1`.
#[inline]
fn row_of(a: &[f64], n: usize, i: usize, j0: usize, j1: usize) -> &[f64] {
    &a[i * n + j0..i * n + j1]
}

/// Classify every row over the `active` lanes. With `exact_zero` a mixed row
/// selects per lane (the reference's skip, exact even on a `-0.0`
/// accumulator). Without it a mixed row is `Plain`: allowed ONLY for an
/// accumulator initialised to `+0.0`, which can never become `-0.0` under
/// round-to-nearest addition (`-0.0` needs `-0.0 + -0.0`; an exact
/// cancellation gives `+0.0`), so adding a zero product (`+-0.0`, the rows
/// being finite) is the identity on it.
fn classify_rows(
    mult: &[Lanes],
    active: &[bool; BLOCK_NODES],
    exact_zero: bool,
    kind: &mut [RowKind],
) {
    for (k, m) in kind.iter_mut().zip(mult) {
        let mut zero = false;
        let mut nonzero = false;
        for (&on, &v) in active.iter().zip(m) {
            if on {
                if v != 0.0 {
                    nonzero = true;
                } else {
                    zero = true;
                }
            }
        }
        *k = if !nonzero {
            RowKind::Skip
        } else if zero && exact_zero {
            RowKind::Select
        } else {
            RowKind::Plain
        };
    }
}

/// `tile[jj][lane] += mult[i][lane] * a[i][j0 + jj]` (i.e. `x = x + m a`) for
/// `i = 0, 1, ...` IN ORDER (per element: the reference's row order), per
/// each row's [`RowKind`].
fn axpy_tile(
    a: &[f64],
    n: usize,
    j0: usize,
    j1: usize,
    mult: &[Lanes],
    kind: &[RowKind],
    tile: &mut [Lanes],
) {
    let depth = mult.len();
    let mut i = 0usize;
    while i < depth {
        match kind[i] {
            RowKind::Skip => i += 1,
            RowKind::Select => {
                axpy_select(row_of(a, n, i, j0, j1), &mult[i], tile);
                i += 1;
            }
            RowKind::Plain => {
                if i + 1 < depth && kind[i + 1] == RowKind::Plain {
                    // Two rows per pass over the tile: per element still row
                    // `i` first, then row `i + 1`.
                    axpy_plain2(
                        row_of(a, n, i, j0, j1),
                        &mult[i],
                        row_of(a, n, i + 1, j0, j1),
                        &mult[i + 1],
                        tile,
                    );
                    i += 2;
                } else {
                    axpy_plain(row_of(a, n, i, j0, j1), &mult[i], tile);
                    i += 1;
                }
            }
        }
    }
}

#[inline]
fn axpy_plain(r: &[f64], m: &Lanes, tile: &mut [Lanes]) {
    for (t, &av) in tile.iter_mut().zip(r) {
        for (x, &mv) in t.iter_mut().zip(m) {
            *x += mv * av;
        }
    }
}

#[inline]
fn axpy_plain2(r0: &[f64], m0: &Lanes, r1: &[f64], m1: &Lanes, tile: &mut [Lanes]) {
    for ((t, &a0), &a1) in tile.iter_mut().zip(r0).zip(r1) {
        let mut x = *t;
        for (xv, &mv) in x.iter_mut().zip(m0) {
            *xv += mv * a0;
        }
        for (xv, &mv) in x.iter_mut().zip(m1) {
            *xv += mv * a1;
        }
        *t = x;
    }
}

#[inline]
fn axpy_select(r: &[f64], m: &Lanes, tile: &mut [Lanes]) {
    for (t, &av) in tile.iter_mut().zip(r) {
        for (xv, &mv) in t.iter_mut().zip(m) {
            let x = *xv;
            *xv = if mv != 0.0 { x + mv * av } else { x };
        }
    }
}

/// `abs[lane] += |rc|` over the tile's columns in increasing `j`
/// (`rc.iter().map(|v| v.abs()).sum()`, carried across tiles).
#[inline]
fn fold_abs(tile: &[Lanes], abs: &mut Lanes) {
    let mut acc = *abs;
    for t in tile {
        for (s, &v) in acc.iter_mut().zip(t) {
            *s += v.abs();
        }
    }
    *abs = acc;
}

/// `p[i][lane] += (rc < 0 ? a_ij : -a_ij)` over the tile's columns
/// in increasing `j`, every row (the reference's per-row subgradient fold,
/// carried across tiles).
#[inline]
fn fold_subgrad(a: &[f64], n: usize, j0: usize, j1: usize, tile: &[Lanes], p: &mut [Lanes]) {
    for (i, pi) in p.iter_mut().enumerate() {
        let r = row_of(a, n, i, j0, j1);
        let mut acc = *pi;
        for (t, &av) in tile.iter().zip(r) {
            let nav = -av;
            for (s, &rv) in acc.iter_mut().zip(t) {
                *s += if rv < 0.0 { av } else { nav };
            }
        }
        *pi = acc;
    }
}

impl FrozenDualProblem {
    /// Evaluate `out.len()` nodes of ONE `depth` (row-major
    /// `out.len() x depth` inputs `sides`, `l0`, `l1`).
    ///
    /// CONTRACT: `out[b]` is bit-identical (`value`, `certified`, `charge` by
    /// `to_bits`) to `self.eval_node(sides_b, l0_b, l1_b, ..)`; with
    /// `want.rc`, `rc_row(b)` ([`BlockScratch::rc_row`]) is bit-identical to
    /// `NodeScratch::rc()` after that call (when its value is finite); with
    /// `want.subgrad` and `s0` / `s1` of length `out.len() x depth`, their
    /// rows are bit-identical to `eval_node`'s `Subgradient` for every node the
    /// reference did not refuse in its prologue (it leaves them stale there,
    /// and so does this). A node the reference refuses is refused here, and
    /// ONLY that node; a depth beyond [`Self::max_depth`] or an input length
    /// mismatch refuses every node (the `eval_batch` rule).
    pub(crate) fn eval_block(
        &self,
        depth: usize,
        sides: &[Side],
        l0: &[f64],
        l1: &[f64],
        geom: BlockGeom,
        want: BlockWant,
        ws: &mut BlockScratch,
        s0: &mut [f64],
        s1: &mut [f64],
        out: &mut [NodeBound],
    ) {
        self.eval_block_inner(depth, sides, l0, l1, geom, want, ws, s0, s1, out, true);
    }

    /// [`Self::eval_block`] with the certificate optionally DISABLED. The
    /// `charge_on = false` arm exists ONLY for the mutation pin (the same
    /// contract as `eval_node_inner`); the search passes `true`.
    pub(crate) fn eval_block_inner(
        &self,
        depth: usize,
        sides: &[Side],
        l0: &[f64],
        l1: &[f64],
        geom: BlockGeom,
        want: BlockWant,
        ws: &mut BlockScratch,
        s0: &mut [f64],
        s1: &mut [f64],
        out: &mut [NodeBound],
        charge_on: bool,
    ) {
        let count = out.len();
        let n = self.n;
        if want.rc {
            let len = count.saturating_mul(n);
            if ws.rc.len() != len {
                ws.rc.resize(len, 0.0);
            }
        }
        let rows = count.checked_mul(depth);
        if depth > self.order.len()
            || rows != Some(sides.len())
            || rows != Some(l0.len())
            || rows != Some(l1.len())
        {
            out.fill(NodeBound::refused());
            return;
        }
        let sub_ok = want.subgrad && rows == Some(s0.len()) && rows == Some(s1.len());
        let group = geom.nodes.clamp(1, BLOCK_NODES);
        let cols = geom.cols.max(1);
        let seed = sum_seed();
        // The initial `rc` of every node of this depth: `d` with the split
        // rows' own columns zeroed (before any row is added, as in
        // `eval_node`).
        ws.d_base.clear();
        ws.d_base.extend_from_slice(&self.d);
        for &e in &self.enc[..depth] {
            ws.d_base[e] = 0.0;
        }
        ws.mult.clear();
        ws.mult.resize(depth, [0.0; BLOCK_NODES]);
        ws.kind.clear();
        ws.kind.resize(depth, RowKind::Skip);
        ws.p.clear();
        ws.p.resize(depth, [seed; BLOCK_NODES]);
        let tile_len = cols.min(n);
        if ws.tile.len() < tile_len {
            ws.tile.resize(tile_len, [0.0; BLOCK_NODES]);
        }
        let mut g0 = 0usize;
        while g0 < count {
            let nb = group.min(count - g0);
            // (1) Per-lane prologue: the SAME `prologue_row` `eval_node` runs.
            // A refused lane gets zero multipliers and no epilogue.
            let mut sums: [Option<NodeSums>; BLOCK_NODES] = [None; BLOCK_NODES];
            for (lane, slot) in sums.iter_mut().enumerate() {
                if lane >= nb {
                    for m in &mut ws.mult {
                        m[lane] = 0.0;
                    }
                    continue;
                }
                let base = (g0 + lane) * depth;
                let mut acc = self.node_sums_seed();
                let mut ok = true;
                for i in 0..depth {
                    if let Some(m) =
                        self.prologue_row(i, sides[base + i], l0[base + i], l1[base + i], &mut acc)
                    {
                        ws.mult[i][lane] = m;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    *slot = Some(acc);
                } else {
                    for m in &mut ws.mult {
                        m[lane] = 0.0;
                    }
                }
            }
            let mut active = [false; BLOCK_NODES];
            for (on, slot) in active.iter_mut().zip(&sums) {
                *on = slot.is_some();
            }
            classify_rows(&ws.mult, &active, true, &mut ws.kind);
            ws.p.fill([seed; BLOCK_NODES]);
            let mut abs: Lanes = [seed; BLOCK_NODES];
            // (2)-(4) Per column tile, in increasing `j`: init, accumulate the
            // rows in `i` order, then carry both folds across the tile.
            let mut j0 = 0usize;
            while j0 < n {
                let j1 = (j0 + cols).min(n);
                let tile = &mut ws.tile[..j1 - j0];
                for (t, &dv) in tile.iter_mut().zip(&ws.d_base[j0..j1]) {
                    *t = [dv; BLOCK_NODES];
                }
                axpy_tile(&self.a, n, j0, j1, &ws.mult, &ws.kind, tile);
                if want.rc {
                    for lane in 0..nb {
                        let at = (g0 + lane) * n;
                        for (dst, t) in ws.rc[at + j0..at + j1].iter_mut().zip(tile.iter()) {
                            *dst = t[lane];
                        }
                    }
                }
                fold_abs(tile, &mut abs);
                if sub_ok {
                    fold_subgrad(&self.a, n, j0, j1, tile, &mut ws.p);
                }
                j0 = j1;
            }
            // (5) Per-lane epilogue: the SAME `finish_node` `eval_node` runs.
            let p = &ws.p;
            for (lane, slot) in sums.iter().enumerate().take(nb) {
                let b = g0 + lane;
                let Some(acc) = *slot else {
                    out[b] = NodeBound::refused();
                    continue;
                };
                let r = b * depth..(b + 1) * depth;
                let sg = if sub_ok {
                    Some(Subgradient {
                        s0: &mut s0[r.clone()],
                        s1: &mut s1[r.clone()],
                    })
                } else {
                    None
                };
                out[b] = self.finish_node(&sides[r], acc, abs[lane], sg, |i| p[i][lane], charge_on);
            }
            g0 += nb;
        }
    }

    /// `out[b n + j] = sum_i mult[b depth + i] a_ij` for `count` nodes, each
    /// started from `+0.0` with the rows added in `i` order and a zero
    /// multiplier adding nothing: bit-identical, node by node, to
    ///
    /// ```text
    /// v = [0.0; n]; for i in 0..depth { if m_i != 0.0 { v_j += m_i a_ij } }
    /// ```
    ///
    /// (stage B's line-search direction `da`). A zero multiplier on a mixed
    /// row is applied as a plain add, which is exact here because the
    /// accumulator starts at `+0.0` (see `classify_rows`). Returns `false`,
    /// writing nothing, on a depth beyond [`Self::max_depth`] or a length
    /// mismatch.
    pub(crate) fn block_axpy_rows(
        &self,
        depth: usize,
        count: usize,
        mult: &[f64],
        geom: BlockGeom,
        ws: &mut BlockScratch,
        out: &mut [f64],
    ) -> bool {
        let n = self.n;
        if depth > self.order.len()
            || count.checked_mul(depth) != Some(mult.len())
            || count.checked_mul(n) != Some(out.len())
        {
            return false;
        }
        let group = geom.nodes.clamp(1, BLOCK_NODES);
        let cols = geom.cols.max(1);
        ws.mult.clear();
        ws.mult.resize(depth, [0.0; BLOCK_NODES]);
        ws.kind.clear();
        ws.kind.resize(depth, RowKind::Skip);
        let tile_len = cols.min(n);
        if ws.tile.len() < tile_len {
            ws.tile.resize(tile_len, [0.0; BLOCK_NODES]);
        }
        let mut g0 = 0usize;
        while g0 < count {
            let nb = group.min(count - g0);
            let mut active = [false; BLOCK_NODES];
            for (lane, on) in active.iter_mut().enumerate() {
                *on = lane < nb;
                for (i, m) in ws.mult.iter_mut().enumerate() {
                    m[lane] = if lane < nb {
                        mult[(g0 + lane) * depth + i]
                    } else {
                        0.0
                    };
                }
            }
            classify_rows(&ws.mult, &active, false, &mut ws.kind);
            let mut j0 = 0usize;
            while j0 < n {
                let j1 = (j0 + cols).min(n);
                let tile = &mut ws.tile[..j1 - j0];
                tile.fill([0.0; BLOCK_NODES]);
                axpy_tile(&self.a, n, j0, j1, &ws.mult, &ws.kind, tile);
                for lane in 0..nb {
                    let at = (g0 + lane) * n;
                    for (dst, t) in out[at + j0..at + j1].iter_mut().zip(tile.iter()) {
                        *dst = t[lane];
                    }
                }
                j0 = j1;
            }
            g0 += nb;
        }
        true
    }
}
