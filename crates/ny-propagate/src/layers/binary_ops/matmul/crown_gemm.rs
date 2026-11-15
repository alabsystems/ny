// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-batched-bilinear (ENGINE_PROGRAM §2 V1): the batched, GEMM-shaped
//! McCormick backward for var×var `MatMul` / `BilinearCrown` (typed
//! `bab.bilinear_gemm_kernel`, dark by default), BIT-IDENTICAL to the serial
//! nest it replaces.
//!
//! # What it replaces
//!
//! The certified McCormick backward of a var×var product (`C = s·A·B` or
//! `s·A·Bᵀ`: vit's `Q·Kᵀ` and `attn·V`) is, in BOTH of its arms, a
//! single-threaded five-deep scalar nest `rows × batch × m × n × k`:
//!
//! * the midpoint arm, `MatMulLayer::propagate_linear_binary_with_kernels`
//!   (`crown_dense.rs`), used by every CROWN-IBP collector walk;
//! * the interpolated α arm,
//!   `BilinearCrownLayer::propagate_linear_binary_with_alpha_kernels`
//!   (`bilinear/linear.rs`), used by the alpha-post pass, the α-margin ascent
//!   and BaB child walks.
//!
//! Each iteration recomputes the element's McCormick plane (for the α arm, two
//! `interpolated_mccormick` calls) although the plane depends only on the
//! element and the SIGN of the weight, never on the row. On vit_2023's ibp model
//! that is 13,872 iterations per row per attention crossing, and it is why the
//! CROWN-IBP collector completes 1 of 19 targets (scratchpad/vbs/diag6/vit.md
//! §2). #vit-collector-throughput's row-parallel twin (`bab.bilinear_rowpar_kernel`)
//! runs rows on rayon but keeps the per-element scalar body and recomputes the
//! whole plane grid once per chunk of rows.
//!
//! # The shape of this kernel
//!
//! It is the outer-product (rank-1 update) form of the two GEMMs inside the
//! nest, with the right-hand panel SELECTED per (row, j) by the weight's sign:
//!
//! ```text
//! A side, slab row (b,i):  Acoef[r,(b,i,:)] += Σ_j w[r,(b,i,j)] · AX_sel(w)[(b,i,j), :]
//! B side, slab col (b,j):  Bcoef[r,(b,j,:)] += Σ_i w[r,(b,i,j)] · AY_sel(w)[(b,i,j), :]
//! certified S (f64):       the same updates with |w|·|AX|, |w|·|AY|
//! constant (f64):          c[r] = bias[r] + Σ_(b,i,j,l) w[r,(b,i,j)] · C_sel(w)[(b,i,j), l]
//! ```
//!
//! 1. **Batched planes.** Every element's four candidate planes
//!    `{Lower, Upper} × {w >= 0, w < 0}` are computed ONCE per call (in parallel
//!    over slabs) into a structure-of-arrays table ([`McCormickPlaneTable`]),
//!    shared read-only by every row.
//! 2. **GEMM-shaped updates.** Rows run in chunks on rayon and, inside a chunk,
//!    in tiles of [`GEMM_ROW_TILE`]. For each slab `(b, i, j)` each row adds
//!    `w · panel` to its `k` A cells (contiguous) and its `k` B cells
//!    (contiguous for `Bᵀ`, stride `n` otherwise) in tight loops the compiler
//!    can vectorize, one for the f32 cells and one for the f64 `S` cells.
//! 3. **Interleaved constant chains.** The f64 constant fold is a dependency
//!    chain per row; the tile advances [`GEMM_ROW_TILE`] independent chains in
//!    lockstep instead of one.
//!
//! # Why it is BIT-IDENTICAL to the serial nest (the soundness argument)
//!
//! No new relaxation and no new certificate: the published object is produced
//! by the same IEEE operations, on the same operands, in the same order per
//! accumulator, and published by the same tail.
//!
//! * **Rows are independent.** The nest indexes every accumulator by the row
//!   (`[out_idx, ..]` coefficient and `S` cells, the per-row constant folds) and
//!   nothing reads another row's accumulator, so chunking, tiling and the
//!   thread count cannot change a value.
//! * **Each accumulator sees the nest's sequence.** The nest visits `(b, i, j,
//!   l)` lexicographically. An A cell `(r, b, i, l)` is therefore updated once
//!   per `j` in increasing `j`; a B cell `(r, b, j, l)` once per `i` in
//!   increasing `i`; the row's two constant folds once per `(b, i, j, l)` in
//!   lexicographic order. This kernel visits the slabs `(b, i, j)`
//!   lexicographically for every row and `l` increasing inside a slab, so every
//!   accumulator's own sequence is unchanged; only the interleaving ACROSS
//!   accumulators changes, which no accumulator can observe.
//! * **Each update is the nest's expression.** `cell += w * ax` in f32 (Rust
//!   never contracts `a * b + c` into an FMA), `s += |w as f64| * |ax as f64|`
//!   (an exact product), and the fold's `term = w as f64 * c as f64` added to
//!   the value and to both magnitude sums with `adds += 1`. The weight is the
//!   nest's `w = coefficient * scale` in f32.
//! * **The same guards.** A zero weight is skipped by the nest's predicate
//!   `w != 0.0` (so `-0.0` too), and the plane is selected by the nest's
//!   predicate `w >= 0.0` (so a NaN weight takes the negative-weight plane and
//!   poisons exactly the cells it poisons in the nest). The α arm's non-finite
//!   element (unreachable behind `validate_mccormick_inputs`, replicated
//!   anyway) skips the coefficient update and forces each nonzero-weight fold
//!   trivial exactly where the nest does.
//! * **The same planes.** The midpoint arm's table stores
//!   `select_mccormick_plane(.., ±1.0, dir)` from the nest's own midpoint
//!   anchors (the function reads the weight ONLY through `w >= 0.0`,
//!   #vit-collector-throughput); the α arm's table stores the two
//!   `interpolated_mccormick` results of the element's clamped parameters, a
//!   pure function of row-independent inputs.
//! * **The same tail.** Publication is `McCormickRowparBuffers::publish`, the
//!   nest's tail verbatim (exact halving, directed casts, the
//!   `γ^{f32}_{depth+2}·S` coefficient charge), and every constant fold is
//!   finished by the CALLING ARM's own `widen_mccormick_constant`.
//!
//! So the certified error accounting is the nest's, unchanged.
//! `bilinear_gemm_bit_identical_to_serial` pins all of it on vit shapes (both
//! arms, both entries), `bilinear_gemm_thread_count_independent` pins the
//! schedule independence, and
//! `bilinear_gemm_contains_exact_product_on_random_attention_blocks` checks the
//! published relation against the exact product in rational arithmetic.
//!
//! # Why the coefficient cells are not accumulated in f64
//!
//! ENGINE_PROGRAM §2 V1 asks for a "batched certified bilinear backward in f64"
//! with "the same certified error accounting". Everything the certificate
//! consists of — `S`, the constant fold and every charge — is already f64 here,
//! as in the nest. Accumulating the f32 COEFFICIENT cells in f64 too would
//! (a) break bit-identity, turning a re-scheduling of existing certified
//! arithmetic into NEW certified arithmetic (the program's rule: every
//! published bound is certified by existing or bit-identical code), and
//! (b) buy nothing under the same accounting, because the published
//! `γ^{f32}_{depth+2}·S` charge does not shrink when the accumulation is more
//! accurate. It is deliberately not done; a tighter-charge f64 form is its own
//! milestone with its own certificate tests.
//!
//! # Not here
//!
//! No deadline poll inside the kernel (neither the nest nor the row-parallel
//! twin has one; the walk polls between steps, and this kernel is the fastest
//! of the three). No GPU: R2, the device never certifies.

use std::borrow::Cow;
use std::time::{Duration, Instant};

use ny_tensor::BoundedTensor;

use super::crown_dense::{
    hoisted_midpoint_planes, mccormick_rowpar_chunk_rows, widen_mccormick_constant,
    McCormickRowChunk, McCormickRowparBuffers, McCormickRowparShape,
};
use super::{BoundDir, LinearBounds, MatMulLayer, Result};

/// Rows advanced together inside a chunk. Their constant folds are independent
/// dependency chains, interleaved for instruction-level parallelism; their
/// coefficient updates share the slab's plane panels while they are hot. Any
/// value is correct (rows are independent); 8 keeps the per-tile state in
/// registers and the stack.
pub(in crate::layers::binary_ops) const GEMM_ROW_TILE: usize = 8;

/// Largest plane table the kernel builds (48 bytes per element). Above it the
/// kernel declines and the caller's next choice runs. vit_2023's ibp attention
/// product needs 13,872 elements (0.67 MB).
const PLANE_TABLE_MAX_BYTES: usize = 256 << 20;

/// Plane selection in a [`McCormickPlaneTable`] panel: the (direction,
/// weight-sign) pair the serial nests choose with `w >= 0.0`. The order is
/// `hoisted_midpoint_planes`' order.
pub(in crate::layers::binary_ops) const SEL_LOWER_NONNEG: usize = 0;
/// Lower direction, `w < 0` (or NaN).
pub(in crate::layers::binary_ops) const SEL_LOWER_NEG: usize = 1;
/// Upper direction, `w >= 0`.
pub(in crate::layers::binary_ops) const SEL_UPPER_NONNEG: usize = 2;
/// Upper direction, `w < 0` (or NaN).
pub(in crate::layers::binary_ops) const SEL_UPPER_NEG: usize = 3;

const COMP_AX: usize = 0;
const COMP_AY: usize = 1;
const COMP_C: usize = 2;

/// The four planes `(ax, ay, c)` of one element, in selection order
/// (`SEL_LOWER_NONNEG`, `SEL_LOWER_NEG`, `SEL_UPPER_NONNEG`, `SEL_UPPER_NEG`).
pub(in crate::layers::binary_ops) type ElementPlanes = [(f32, f32, f32); 4];

/// The arm's own constant certificate: `(accumulated, magnitude_sum,
/// term_magnitude_sum, adds, dir) -> published f64 constant`. Both arms'
/// `widen_mccormick_constant` have exactly this signature.
pub(in crate::layers::binary_ops) type WidenFn = fn(f64, f64, f64, usize, BoundDir) -> f64;

/// How the kernel schedules rows: `chunk_rows` rows per rayon task, and
/// whether the chunks (and the plane-table build) may fan out at all. Neither
/// can change a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::layers::binary_ops) struct GemmSchedule {
    pub(in crate::layers::binary_ops) chunk_rows: usize,
    pub(in crate::layers::binary_ops) parallel: bool,
}

impl GemmSchedule {
    /// The production schedule: the row-parallel twin's chunk size rounded up
    /// to whole tiles, fanned out under the same nesting policy
    /// (`faer_parallelism::nested_rayon_fanout_permitted`).
    pub(in crate::layers::binary_ops) fn production(num_outputs: usize) -> Self {
        let rows = mccormick_rowpar_chunk_rows(num_outputs);
        Self {
            chunk_rows: rows.div_ceil(GEMM_ROW_TILE).saturating_mul(GEMM_ROW_TILE),
            parallel: crate::faer_parallelism::nested_rayon_fanout_permitted(),
        }
    }
}

/// Every element's four candidate McCormick planes, computed once per call.
///
/// Layout `[slab][sel][component][l]` with `slab = (batch * m + i) * n + j`,
/// `sel` in selection order and `component` in `(ax, ay, c)`, so one
/// (slab, sel, component) panel is `k` contiguous `f32`s.
pub(in crate::layers::binary_ops) struct McCormickPlaneTable {
    k: usize,
    planes: Vec<f32>,
    /// `[slab][l]`: 1 where the element builder returned `None` (the α arm's
    /// non-finite operand box). Always 0 for the midpoint arm, whose non-finite
    /// planes are ordinary (trivial) table entries exactly as in its nest.
    nonfinite: Vec<u8>,
    /// Per slab: any `nonfinite` entry set.
    slab_nonfinite: Vec<bool>,
    /// Wall time of the build (telemetry only).
    pub(in crate::layers::binary_ops) build_elapsed: Duration,
}

impl McCormickPlaneTable {
    /// Build the table from `element(batch, i, j, l)`, which returns the
    /// element's planes in selection order, or `None` for an element the arm
    /// treats as non-finite. Returns `None` (the caller declines) for an empty
    /// shape or a table above [`PLANE_TABLE_MAX_BYTES`].
    ///
    /// `element` must be a pure function of its arguments: slabs are built in
    /// parallel when `parallel` is set.
    pub(in crate::layers::binary_ops) fn build<F>(
        shape: &McCormickRowparShape,
        parallel: bool,
        element: F,
    ) -> Option<Self>
    where
        F: Fn(usize, usize, usize, usize) -> Option<ElementPlanes> + Sync,
    {
        let started = Instant::now();
        let (m, n, k) = (shape.m, shape.n, shape.k);
        if k == 0 || m == 0 || n == 0 || shape.batch_size == 0 {
            return None;
        }
        let slabs = shape.batch_size.checked_mul(m)?.checked_mul(n)?;
        let elements = slabs.checked_mul(k)?;
        let floats = elements.checked_mul(12)?;
        if floats.checked_mul(size_of::<f32>())? > PLANE_TABLE_MAX_BYTES {
            return None;
        }
        let mut planes = vec![0.0_f32; floats];
        let mut nonfinite = vec![0_u8; elements];
        let slab_len = 12 * k;
        let fill = |slab: usize, out: &mut [f32], flags: &mut [u8]| {
            let j = slab % n;
            let i = (slab / n) % m;
            let batch = slab / (m * n);
            for (l, flag) in flags.iter_mut().enumerate() {
                match element(batch, i, j, l) {
                    Some(element_planes) => {
                        for (sel, &(ax, ay, c)) in element_planes.iter().enumerate() {
                            out[(sel * 3 + COMP_AX) * k + l] = ax;
                            out[(sel * 3 + COMP_AY) * k + l] = ay;
                            out[(sel * 3 + COMP_C) * k + l] = c;
                        }
                    }
                    None => *flag = 1,
                }
            }
        };
        if parallel && slabs > 1 {
            use rayon::prelude::*;
            planes
                .par_chunks_mut(slab_len)
                .zip(nonfinite.par_chunks_mut(k))
                .enumerate()
                .for_each(|(slab, (out, flags))| fill(slab, out, flags));
        } else {
            for (slab, (out, flags)) in planes
                .chunks_mut(slab_len)
                .zip(nonfinite.chunks_mut(k))
                .enumerate()
            {
                fill(slab, out, flags);
            }
        }
        let slab_nonfinite = nonfinite
            .chunks(k)
            .map(|flags| flags.iter().any(|&flag| flag != 0))
            .collect();
        Some(Self {
            k,
            planes,
            nonfinite,
            slab_nonfinite,
            build_elapsed: started.elapsed(),
        })
    }

    /// The `k` values of one (slab, selection, component) panel.
    #[inline]
    fn panel(&self, slab: usize, sel: usize, component: usize) -> &[f32] {
        let start = ((slab * 4 + sel) * 3 + component) * self.k;
        &self.planes[start..start + self.k]
    }

    /// The slab's non-finite mask, or `None` when every element is finite.
    #[inline]
    fn nonfinite_mask(&self, slab: usize) -> Option<&[u8]> {
        self.slab_nonfinite[slab].then(|| &self.nonfinite[slab * self.k..(slab + 1) * self.k])
    }
}

/// The operands of one McCormick backward as flat row-major slices, gathered
/// and length-checked exactly as the row-parallel twin does.
pub(in crate::layers::binary_ops) struct McCormickOperands<'a> {
    pub(in crate::layers::binary_ops) a_lo: Cow<'a, [f32]>,
    pub(in crate::layers::binary_ops) a_hi: Cow<'a, [f32]>,
    pub(in crate::layers::binary_ops) b_lo: Cow<'a, [f32]>,
    pub(in crate::layers::binary_ops) b_hi: Cow<'a, [f32]>,
    w_lo: Cow<'a, [f32]>,
    w_hi: Cow<'a, [f32]>,
    bias_lo: Cow<'a, [f32]>,
    bias_hi: Cow<'a, [f32]>,
    total_c: usize,
}

impl<'a> McCormickOperands<'a> {
    /// Gather the operands, or `None` (the caller declines) when there is
    /// nothing to do or any buffer length disagrees with the parsed dims.
    pub(in crate::layers::binary_ops) fn gather(
        bounds: &'a LinearBounds,
        input_a_bounds: &'a BoundedTensor,
        input_b_bounds: &'a BoundedTensor,
        shape: &McCormickRowparShape,
    ) -> Option<Self> {
        let num_outputs = bounds.num_outputs();
        let total_c = bounds.num_inputs();
        let operands = Self {
            a_lo: crate::contiguous_flat_slice(input_a_bounds.lower()),
            a_hi: crate::contiguous_flat_slice(input_a_bounds.upper()),
            b_lo: crate::contiguous_flat_slice(input_b_bounds.lower()),
            b_hi: crate::contiguous_flat_slice(input_b_bounds.upper()),
            w_lo: crate::contiguous_flat_slice(bounds.lower_a()),
            w_hi: crate::contiguous_flat_slice(bounds.upper_a()),
            bias_lo: crate::contiguous_flat_slice(bounds.lower_b()),
            bias_hi: crate::contiguous_flat_slice(bounds.upper_b()),
            total_c,
        };
        let agree = num_outputs > 0
            && shape.total_a_size > 0
            && shape.total_b_size > 0
            && operands.a_lo.len() == shape.total_a_size
            && operands.a_hi.len() == shape.total_a_size
            && operands.b_lo.len() == shape.total_b_size
            && operands.b_hi.len() == shape.total_b_size
            && Some(operands.w_lo.len()) == num_outputs.checked_mul(total_c)
            && operands.w_hi.len() == operands.w_lo.len()
            && operands.bias_lo.len() == num_outputs
            && operands.bias_hi.len() == num_outputs
            && Some(total_c) == shape.batch_size.checked_mul(shape.c_size_per_batch);
        agree.then_some(operands)
    }

    /// The kernel's view of these operands, a table built from them, and the
    /// layer scale.
    pub(in crate::layers::binary_ops) fn inputs<'b>(
        &'b self,
        shape: &'b McCormickRowparShape,
        table: &'b McCormickPlaneTable,
        scale: f32,
    ) -> McCormickGemmInputs<'b> {
        McCormickGemmInputs {
            shape,
            table,
            w_lo: &self.w_lo,
            w_hi: &self.w_hi,
            bias_lo: &self.bias_lo,
            bias_hi: &self.bias_hi,
            total_c: self.total_c,
            scale,
        }
    }
}

/// Everything the row kernel reads (all shared, read-only).
pub(in crate::layers::binary_ops) struct McCormickGemmInputs<'a> {
    shape: &'a McCormickRowparShape,
    table: &'a McCormickPlaneTable,
    w_lo: &'a [f32],
    w_hi: &'a [f32],
    bias_lo: &'a [f32],
    bias_hi: &'a [f32],
    total_c: usize,
    scale: f32,
}

/// The nest's per-row constant-fold state (`DenseConstFold` in the midpoint
/// arm, `McCormickConstFold` in the α arm — the same four fields, the same
/// updates), finished by the arm's own [`WidenFn`].
#[derive(Clone, Copy)]
struct GemmConstFold {
    accumulated: f64,
    magnitude_sum: f64,
    term_magnitude_sum: f64,
    adds: usize,
}

impl GemmConstFold {
    /// Seeded with the incoming f32 bias (exact in f64), as in both nests.
    #[inline]
    fn new(bias: f32) -> Self {
        let accumulated = f64::from(bias);
        Self {
            accumulated,
            magnitude_sum: accumulated.abs(),
            term_magnitude_sum: 0.0,
            adds: 0,
        }
    }

    /// `term = w as f64 * c as f64` (exact) added to the value and to both
    /// magnitude sums; one more add.
    #[inline]
    fn add_term(&mut self, w: f32, plane_const: f32) {
        let term = f64::from(w) * f64::from(plane_const);
        self.accumulated += term;
        let magnitude = term.abs();
        self.magnitude_sum += magnitude;
        self.term_magnitude_sum += magnitude;
        self.adds += 1;
    }

    /// The α arm's `McCormickConstFold::force_trivial`, field for field.
    #[inline]
    fn force_trivial(&mut self, dir: BoundDir) {
        self.accumulated = match dir {
            BoundDir::Lower => f64::NEG_INFINITY,
            BoundDir::Upper => f64::INFINITY,
        };
        self.magnitude_sum = f64::INFINITY;
        self.term_magnitude_sum = f64::INFINITY;
        self.adds = self.adds.saturating_add(1);
    }

    #[inline]
    fn finish(&self, widen: WidenFn, dir: BoundDir) -> f64 {
        widen(
            self.accumulated,
            self.magnitude_sum,
            self.term_magnitude_sum,
            self.adds,
            dir,
        )
    }
}

/// One weight's rank-1 update of `k` coefficient cells and their certified
/// `S` cells: `cells[base + l*stride] += w * coeffs[l]` (f32) and
/// `sums[base + l*stride] += |w| * |coeffs[l]|` (f64, `wabs = |w as f64|`),
/// skipping every `l` the mask marks non-finite.
#[inline]
#[allow(clippy::too_many_arguments)]
fn panel_axpy(
    cells: &mut [f32],
    sums: &mut [f64],
    base: usize,
    stride: usize,
    w: f32,
    wabs: f64,
    coeffs: &[f32],
    skip: Option<&[u8]>,
) {
    let len = coeffs.len();
    if skip.is_none() && stride == 1 {
        for (cell, &a) in cells[base..base + len].iter_mut().zip(coeffs) {
            *cell += w * a;
        }
        for (sum, &a) in sums[base..base + len].iter_mut().zip(coeffs) {
            *sum += wabs * f64::from(a).abs();
        }
        return;
    }
    for (l, &a) in coeffs.iter().enumerate() {
        if skip.is_some_and(|mask| mask[l] != 0) {
            continue;
        }
        let idx = base + l * stride;
        cells[idx] += w * a;
        sums[idx] += wabs * f64::from(a).abs();
    }
}

/// One tile of `tile_rows` consecutive rows of `chunk`, starting at the
/// chunk-local row `tile_start`: every slab, then the published constants.
fn gemm_row_tile(
    inputs: &McCormickGemmInputs<'_>,
    chunk: &mut McCormickRowChunk<'_>,
    tile_start: usize,
    tile_rows: usize,
    widen: WidenFn,
) {
    let shape = inputs.shape;
    let table = inputs.table;
    let k = shape.k;
    let total_a = shape.total_a_size;
    let total_b = shape.total_b_size;
    let total_c = inputs.total_c;
    let b_stride = if shape.transpose_b { 1 } else { shape.n };

    let mut lower = [GemmConstFold::new(0.0); GEMM_ROW_TILE];
    let mut upper = [GemmConstFold::new(0.0); GEMM_ROW_TILE];
    for t in 0..tile_rows {
        let row = chunk.row_start + tile_start + t;
        lower[t] = GemmConstFold::new(inputs.bias_lo[row]);
        upper[t] = GemmConstFold::new(inputs.bias_hi[row]);
    }
    let mut w_lower = [0.0_f32; GEMM_ROW_TILE];
    let mut w_upper = [0.0_f32; GEMM_ROW_TILE];
    let mut sel_lower = [SEL_LOWER_NONNEG; GEMM_ROW_TILE];
    let mut sel_upper = [SEL_UPPER_NONNEG; GEMM_ROW_TILE];

    for batch in 0..shape.batch_size {
        for i in 0..shape.m {
            let a_base = shape.a_flat(batch, i, 0);
            for j in 0..shape.n {
                let slab = (batch * shape.m + i) * shape.n + j;
                let c_flat = shape.c_flat(batch, i, j);
                // `b_flat(batch, j, l) = b_base + l * b_stride` in both layouts.
                let b_base = shape.b_flat(batch, j, 0);

                // The nest's weights and plane choices for this slab.
                let mut any_weight = false;
                for t in 0..tile_rows {
                    let row = chunk.row_start + tile_start + t;
                    let wl = inputs.w_lo[row * total_c + c_flat] * inputs.scale;
                    let wu = inputs.w_hi[row * total_c + c_flat] * inputs.scale;
                    w_lower[t] = wl;
                    w_upper[t] = wu;
                    sel_lower[t] = if wl >= 0.0 {
                        SEL_LOWER_NONNEG
                    } else {
                        SEL_LOWER_NEG
                    };
                    sel_upper[t] = if wu >= 0.0 {
                        SEL_UPPER_NONNEG
                    } else {
                        SEL_UPPER_NEG
                    };
                    any_weight |= wl != 0.0 || wu != 0.0;
                }
                // Every action below is guarded by `w != 0.0`, in the nest as
                // here, so a slab no row of the tile weights is a no-op.
                if !any_weight {
                    continue;
                }
                let skip = table.nonfinite_mask(slab);

                // Coefficient and certified-S cells (one update per cell per
                // slab, as in the nest).
                for t in 0..tile_rows {
                    let r = tile_start + t;
                    let a_row = r * total_a + a_base;
                    let b_row = r * total_b + b_base;
                    let wl = w_lower[t];
                    if wl != 0.0 {
                        let sel = sel_lower[t];
                        let wabs = f64::from(wl).abs();
                        panel_axpy(
                            &mut *chunk.lower_a_a,
                            &mut *chunk.lower_s_a,
                            a_row,
                            1,
                            wl,
                            wabs,
                            table.panel(slab, sel, COMP_AX),
                            skip,
                        );
                        panel_axpy(
                            &mut *chunk.lower_a_b,
                            &mut *chunk.lower_s_b,
                            b_row,
                            b_stride,
                            wl,
                            wabs,
                            table.panel(slab, sel, COMP_AY),
                            skip,
                        );
                    }
                    let wu = w_upper[t];
                    if wu != 0.0 {
                        let sel = sel_upper[t];
                        let wabs = f64::from(wu).abs();
                        panel_axpy(
                            &mut *chunk.upper_a_a,
                            &mut *chunk.upper_s_a,
                            a_row,
                            1,
                            wu,
                            wabs,
                            table.panel(slab, sel, COMP_AX),
                            skip,
                        );
                        panel_axpy(
                            &mut *chunk.upper_a_b,
                            &mut *chunk.upper_s_b,
                            b_row,
                            b_stride,
                            wu,
                            wabs,
                            table.panel(slab, sel, COMP_AY),
                            skip,
                        );
                    }
                }

                // Constant folds: each row's chain advances in the nest's
                // (slab, l) order; the tile's rows interleave.
                let empty: &[f32] = &[];
                let mut c_lower = [empty; GEMM_ROW_TILE];
                let mut c_upper = [empty; GEMM_ROW_TILE];
                for t in 0..tile_rows {
                    c_lower[t] = table.panel(slab, sel_lower[t], COMP_C);
                    c_upper[t] = table.panel(slab, sel_upper[t], COMP_C);
                }
                for l in 0..k {
                    let nonfinite = skip.is_some_and(|mask| mask[l] != 0);
                    for t in 0..tile_rows {
                        let wl = w_lower[t];
                        if wl != 0.0 {
                            if nonfinite {
                                lower[t].force_trivial(BoundDir::Lower);
                            } else {
                                lower[t].add_term(wl, c_lower[t][l]);
                            }
                        }
                        let wu = w_upper[t];
                        if wu != 0.0 {
                            if nonfinite {
                                upper[t].force_trivial(BoundDir::Upper);
                            } else {
                                upper[t].add_term(wu, c_upper[t][l]);
                            }
                        }
                    }
                }
            }
        }
    }

    for t in 0..tile_rows {
        let r = tile_start + t;
        chunk.lower_b_total[r] = lower[t].finish(widen, BoundDir::Lower);
        chunk.upper_b_total[r] = upper[t].finish(widen, BoundDir::Upper);
    }
}

/// Run the GEMM-shaped backward over every row and publish the two operand
/// shares through the nest's own tail. Returns the published pair and whether
/// the chunks ran in parallel, or `None` when the accumulators cannot be
/// allocated for this shape (the caller declines).
pub(in crate::layers::binary_ops) fn mccormick_gemm_backward(
    inputs: &McCormickGemmInputs<'_>,
    num_outputs: usize,
    schedule: GemmSchedule,
    widen: WidenFn,
) -> Result<Option<((LinearBounds, LinearBounds), bool)>> {
    let shape = inputs.shape;
    let Some(mut buffers) =
        McCormickRowparBuffers::new(num_outputs, shape.total_a_size, shape.total_b_size)
    else {
        return Ok(None);
    };
    let parallel = buffers.run_scheduled(
        schedule.chunk_rows,
        schedule.parallel,
        |chunk: &mut McCormickRowChunk<'_>| {
            let mut tile_start = 0_usize;
            while tile_start < chunk.rows {
                let tile_rows = GEMM_ROW_TILE.min(chunk.rows - tile_start);
                gemm_row_tile(inputs, chunk, tile_start, tile_rows, widen);
                tile_start += tile_rows;
            }
        },
    );
    let published = buffers.publish(shape.n, shape.m)?;
    Ok(Some((published, parallel)))
}

impl MatMulLayer {
    /// #vit-batched-bilinear: the GEMM-shaped twin of the serial midpoint nest
    /// in [`Self::propagate_linear_binary_with_kernels`] — BIT-IDENTICAL to it
    /// (module docs).
    ///
    /// `schedule == None` is the production schedule; tests pass explicit ones.
    /// Returns `Ok(None)` (the caller's next choice then runs) for a shape it
    /// does not handle: no rows, an empty operand, a buffer whose length
    /// disagrees with the parsed dims, or a plane table above the size cap.
    pub(in crate::layers::binary_ops) fn propagate_linear_binary_gemm(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        shape: &McCormickRowparShape,
        scale: f32,
        schedule: Option<GemmSchedule>,
    ) -> Result<Option<(LinearBounds, LinearBounds)>> {
        let started = Instant::now();
        let Some(operands) =
            McCormickOperands::gather(bounds, input_a_bounds, input_b_bounds, shape)
        else {
            return Ok(None);
        };
        let num_outputs = bounds.num_outputs();
        let schedule = schedule.unwrap_or_else(|| GemmSchedule::production(num_outputs));
        let (a_lo, a_hi) = (&*operands.a_lo, &*operands.a_hi);
        let (b_lo, b_hi) = (&*operands.b_lo, &*operands.b_hi);
        let Some(table) = McCormickPlaneTable::build(shape, schedule.parallel, |batch, i, j, l| {
            let a_flat = shape.a_flat(batch, i, l);
            let b_flat = shape.b_flat(batch, j, l);
            Some(hoisted_midpoint_planes(
                a_lo[a_flat],
                a_hi[a_flat],
                b_lo[b_flat],
                b_hi[b_flat],
            ))
        }) else {
            return Ok(None);
        };
        let inputs = operands.inputs(shape, &table, scale);
        let Some((published, parallel)) =
            mccormick_gemm_backward(&inputs, num_outputs, schedule, widen_mccormick_constant)?
        else {
            return Ok(None);
        };
        crate::layers::collector_kernels::record_bilinear_gemm(
            false,
            num_outputs,
            parallel,
            started.elapsed(),
            table.build_elapsed,
        );
        Ok(Some(published))
    }
}
