// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense binary64 linear kernels applied to BATCHES of flat rows.
//!
//! A zonotope's centre, its `K` generator rows and its magnitude vectors are
//! all "rows" pushed through the same linear map, so every kernel here maps a
//! row-major `rows x n_in` block to a row-major `rows x n_out` block. This is
//! the production arithmetic Z1 measured: a dense im2col GEMM (faer, always
//! `Par::Seq`, so a child's result is bit-identical whether children run
//! serially or on parallel workers).
//!
//! Every kernel computes a plain round-to-nearest binary64 linear combination
//! in SOME summation order. Callers charge the result with an order-
//! independent Higham envelope over [`LinearKernel::fan_in`] (forward) or
//! [`LinearKernel::fan_out`] (adjoint) terms per output; FMA-contracted and
//! blocked summation are both inside that envelope.
//!
//! #m8-cgan-leaf-batch: every FORWARD kernel also runs under [`RowPar::Rayon`],
//! which dispatches the reference's OWN row blocks (the `step_by(block)` /
//! `step_by(chunk)` loops below) as independent rayon tasks: each task makes the
//! identical `(b, n_in)` GEMM call (same slice, same shape, same im2col /
//! transpose) on a DISJOINT output slice, with its scratch allocated per task.
//! No arithmetic changes, so a row-parallel call is bitwise the sequential one
//! (`zono_row_parallel_kernels_are_bit_identical_to_sequential`); the adjoint
//! kernels are untouched.
//!
//! Geometry conventions (`ConvGeom`): an IMAGE of `channels x height x width`
//! values and a PATCH GRID of `out_h x out_w` positions; patch row `p = (oy,
//! ox)`, patch column `(ci, a, b)` reads image pixel `(ci, oy*sh + a - ph,
//! ox*sw + b - pw)` (zero outside the image). For `Conv2d` the image is the
//! layer input; for `ConvTranspose2d` the image is the layer OUTPUT and the
//! grid is the input, which is the ONNX `ConvTranspose` semantics with
//! symmetric `pads` cropping the full output.

use std::time::Instant;

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;

use super::ZonoRefusal;

/// Target element count for one scratch buffer (`8 * 2^22` bytes = 32 MiB).
pub const CHUNK_ELEMS: usize = 1 << 22;

/// How a forward kernel call (and the row-wise transformer loops that follow
/// it) schedules its rows (#m8-cgan-leaf-batch).
///
/// [`RowPar::Rayon`] runs the reference's own row blocks as rayon tasks over
/// disjoint output slices; every task performs the exact call the sequential
/// loop performs for that block, so the result is bitwise identical. It is
/// scheduling, not arithmetic: never a soundness input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RowPar {
    /// One block after another on the calling thread (the reference).
    #[default]
    Seq,
    /// The reference's blocks as independent rayon tasks.
    Rayon,
}

impl RowPar {
    /// Stable lowercase token for observables.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::Seq => "seq",
            Self::Rayon => "rayon",
        }
    }
}

/// Patch geometry of a 2-D convolution (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvGeom {
    /// Image channels.
    pub channels: usize,
    /// Image height.
    pub height: usize,
    /// Image width.
    pub width: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
    /// Symmetric vertical padding.
    pub pad_h: usize,
    /// Symmetric horizontal padding.
    pub pad_w: usize,
    /// Patch-grid height.
    pub out_h: usize,
    /// Patch-grid width.
    pub out_w: usize,
}

impl ConvGeom {
    /// Build a geometry, refusing zero strides/extents and kernels larger than
    /// the padded image. The grid size is the usual floor formula.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<Self, ZonoRefusal> {
        if channels == 0
            || height == 0
            || width == 0
            || kernel_h == 0
            || kernel_w == 0
            || stride_h == 0
            || stride_w == 0
        {
            return Err(ZonoRefusal::Shape(format!(
                "degenerate convolution geometry c={channels} h={height} w={width} \
                 k={kernel_h}x{kernel_w} s={stride_h}x{stride_w}"
            )));
        }
        let padded_h = height
            .checked_add(2 * pad_h)
            .ok_or_else(|| ZonoRefusal::Shape("padded height overflows".to_string()))?;
        let padded_w = width
            .checked_add(2 * pad_w)
            .ok_or_else(|| ZonoRefusal::Shape("padded width overflows".to_string()))?;
        if padded_h < kernel_h || padded_w < kernel_w {
            return Err(ZonoRefusal::Shape(format!(
                "kernel {kernel_h}x{kernel_w} exceeds padded image {padded_h}x{padded_w}"
            )));
        }
        Ok(Self {
            channels,
            height,
            width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h: (padded_h - kernel_h) / stride_h + 1,
            out_w: (padded_w - kernel_w) / stride_w + 1,
        })
    }

    /// Image element count.
    #[must_use]
    pub fn image_len(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// Patch-grid position count.
    #[must_use]
    pub fn grid_len(&self) -> usize {
        self.out_h * self.out_w
    }

    /// Patch length (`channels * kernel_h * kernel_w`).
    #[must_use]
    pub fn patch_len(&self) -> usize {
        self.channels * self.kernel_h * self.kernel_w
    }

    /// Image pixel read by patch position `(oy, ox)` at kernel offset `(a, b)`,
    /// or `None` when it falls in the padding.
    #[inline]
    fn pixel(&self, oy: usize, ox: usize, a: usize, b: usize) -> Option<(usize, usize)> {
        let iy = (oy * self.stride_h + a).checked_sub(self.pad_h)?;
        let ix = (ox * self.stride_w + b).checked_sub(self.pad_w)?;
        (iy < self.height && ix < self.width).then_some((iy, ix))
    }
}

/// Gather `image` into `patches` (`grid_len x patch_len`, row-major).
pub fn im2col(image: &[f64], geom: &ConvGeom, patches: &mut [f64]) {
    let pl = geom.patch_len();
    let hw = geom.height * geom.width;
    debug_assert_eq!(image.len(), geom.image_len());
    debug_assert_eq!(patches.len(), geom.grid_len() * pl);
    for oy in 0..geom.out_h {
        for ox in 0..geom.out_w {
            let row = &mut patches[(oy * geom.out_w + ox) * pl..][..pl];
            let mut col = 0usize;
            for ci in 0..geom.channels {
                let plane = &image[ci * hw..(ci + 1) * hw];
                for a in 0..geom.kernel_h {
                    for b in 0..geom.kernel_w {
                        row[col] = geom
                            .pixel(oy, ox, a, b)
                            .map_or(0.0, |(iy, ix)| plane[iy * geom.width + ix]);
                        col += 1;
                    }
                }
            }
        }
    }
}

/// Scatter-ADD `patches` (`grid_len x patch_len`) into `image` (the adjoint of
/// [`im2col`]).
pub fn col2im_add(patches: &[f64], geom: &ConvGeom, image: &mut [f64]) {
    let pl = geom.patch_len();
    let hw = geom.height * geom.width;
    debug_assert_eq!(image.len(), geom.image_len());
    debug_assert_eq!(patches.len(), geom.grid_len() * pl);
    for oy in 0..geom.out_h {
        for ox in 0..geom.out_w {
            let row = &patches[(oy * geom.out_w + ox) * pl..][..pl];
            let mut col = 0usize;
            for ci in 0..geom.channels {
                let plane = &mut image[ci * hw..(ci + 1) * hw];
                for a in 0..geom.kernel_h {
                    for b in 0..geom.kernel_w {
                        if let Some((iy, ix)) = geom.pixel(oy, ox, a, b) {
                            plane[iy * geom.width + ix] += row[col];
                        }
                        col += 1;
                    }
                }
            }
        }
    }
}

/// `dst (m x n, row-major) = lhs (m x k) * rhs (k x n)`, sequential faer GEMM.
pub(super) fn gemm_into(
    dst: &mut [f64],
    m: usize,
    n: usize,
    lhs: MatRef<'_, f64>,
    rhs: MatRef<'_, f64>,
) {
    debug_assert_eq!(dst.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    if lhs.ncols() == 0 {
        dst.fill(0.0);
        return;
    }
    let view = MatMut::from_row_major_slice_mut(dst, m, n);
    matmul(view, Accum::Replace, lhs, rhs, 1.0, Par::Seq);
}

/// Refuse with [`ZonoRefusal::Deadline`] once `deadline` has passed.
#[inline]
pub(crate) fn check_deadline(deadline: Option<Instant>) -> Result<(), ZonoRefusal> {
    match deadline {
        Some(limit) if Instant::now() >= limit => Err(ZonoRefusal::Deadline),
        _ => Ok(()),
    }
}

/// A linear map applied row-wise. Each variant stores its matrix in binary64
/// (exact images of the model's `f32` parameters) plus its entry-wise absolute
/// value for the magnitude charges.
#[derive(Debug, Clone)]
pub enum LinearKernel {
    /// `y = W x`, `W` row-major `n_out x n_in`.
    Dense {
        /// Input length.
        n_in: usize,
        /// Output length.
        n_out: usize,
        /// Weights, row-major `n_out x n_in`.
        w: Vec<f64>,
        /// `|W|`.
        w_abs: Vec<f64>,
    },
    /// 2-D convolution; `k` is row-major `out_channels x patch_len`, i.e. the
    /// ONNX `(Cout, Cin, kh, kw)` kernel flattened.
    Conv2d {
        /// Input image / patch geometry.
        geom: ConvGeom,
        /// Output channels.
        out_channels: usize,
        /// Kernel matrix.
        k: Vec<f64>,
        /// `|k|`.
        k_abs: Vec<f64>,
    },
    /// 2-D transposed convolution; `geom` describes the OUTPUT image and the
    /// input grid; `k` is row-major `in_channels x patch_len`, i.e. the ONNX
    /// `(Cin, Cout, kh, kw)` kernel flattened.
    ConvTranspose2d {
        /// Output image / input grid geometry.
        geom: ConvGeom,
        /// Input channels.
        in_channels: usize,
        /// Kernel matrix.
        k: Vec<f64>,
        /// `|k|`.
        k_abs: Vec<f64>,
    },
}

impl LinearKernel {
    /// Input length.
    #[must_use]
    pub fn n_in(&self) -> usize {
        match self {
            Self::Dense { n_in, .. } => *n_in,
            Self::Conv2d { geom, .. } => geom.image_len(),
            Self::ConvTranspose2d {
                geom, in_channels, ..
            } => in_channels * geom.grid_len(),
        }
    }

    /// Output length.
    #[must_use]
    pub fn n_out(&self) -> usize {
        match self {
            Self::Dense { n_out, .. } => *n_out,
            Self::Conv2d {
                geom, out_channels, ..
            } => out_channels * geom.grid_len(),
            Self::ConvTranspose2d { geom, .. } => geom.image_len(),
        }
    }

    /// Upper bound on the number of product terms summed into one FORWARD
    /// output element.
    #[must_use]
    pub fn fan_in(&self) -> usize {
        match self {
            Self::Dense { n_in, .. } => *n_in,
            Self::Conv2d { geom, .. } => geom.patch_len(),
            Self::ConvTranspose2d {
                geom, in_channels, ..
            } => in_channels * geom.kernel_h * geom.kernel_w,
        }
    }

    /// Upper bound on the number of product terms summed into one ADJOINT
    /// output element.
    #[must_use]
    pub fn fan_out(&self) -> usize {
        match self {
            Self::Dense { n_out, .. } => *n_out,
            Self::Conv2d {
                geom, out_channels, ..
            } => out_channels * geom.kernel_h * geom.kernel_w,
            Self::ConvTranspose2d { geom, .. } => geom.patch_len(),
        }
    }

    /// Multiply-accumulates of one forward row (admission / telemetry).
    #[must_use]
    pub fn macs_per_row(&self) -> usize {
        self.n_out().saturating_mul(self.fan_in())
    }

    /// `out[r] = A x[r]` (or `|A| x[r]` when `abs`) for `rows` rows.
    pub fn forward_rows(
        &self,
        x: &[f64],
        rows: usize,
        abs: bool,
        out: &mut [f64],
        deadline: Option<Instant>,
    ) -> Result<(), ZonoRefusal> {
        self.forward_rows_par(x, rows, abs, out, deadline, RowPar::Seq)
            .map(|_| ())
    }

    /// [`Self::forward_rows`] under a row schedule. Returns the number of row
    /// blocks the call ran (its row TASKS under [`RowPar::Rayon`]; telemetry
    /// only). Bitwise identical to the sequential call for every `par`.
    pub fn forward_rows_par(
        &self,
        x: &[f64],
        rows: usize,
        abs: bool,
        out: &mut [f64],
        deadline: Option<Instant>,
        par: RowPar,
    ) -> Result<usize, ZonoRefusal> {
        self.forward_rows_chunked(x, rows, abs, out, deadline, par, CHUNK_ELEMS)
    }

    /// [`Self::forward_rows_par`] with an explicit scratch target, so the tests
    /// can put many blocks on a small fixture. Production passes
    /// [`CHUNK_ELEMS`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_rows_chunked(
        &self,
        x: &[f64],
        rows: usize,
        abs: bool,
        out: &mut [f64],
        deadline: Option<Instant>,
        par: RowPar,
        chunk_elems: usize,
    ) -> Result<usize, ZonoRefusal> {
        let (n_in, n_out) = (self.n_in(), self.n_out());
        if x.len() != rows * n_in || out.len() != rows * n_out {
            return Err(ZonoRefusal::Internal(format!(
                "kernel forward: {} inputs / {} outputs for {rows} rows of {n_in} -> {n_out}",
                x.len(),
                out.len()
            )));
        }
        if rows == 0 {
            return Ok(0);
        }
        match self {
            Self::Dense { w, w_abs, .. } => {
                let w = if abs { w_abs } else { w };
                let block = dense_block_rows(n_in, n_out, chunk_elems);
                dense_forward_rows(x, rows, n_in, n_out, w, block, out, deadline, par)
            }
            Self::Conv2d {
                geom,
                out_channels,
                k,
                k_abs,
            } => {
                let k = if abs { k_abs } else { k };
                conv2d_forward_rows(
                    x,
                    rows,
                    geom,
                    *out_channels,
                    k,
                    out,
                    deadline,
                    par,
                    chunk_elems,
                )
            }
            Self::ConvTranspose2d {
                geom,
                in_channels,
                k,
                k_abs,
            } => {
                let k = if abs { k_abs } else { k };
                conv_transpose2d_forward_rows(
                    x,
                    rows,
                    geom,
                    *in_channels,
                    k,
                    out,
                    deadline,
                    par,
                    chunk_elems,
                )
            }
        }
    }

    /// Scratch elements ONE forward row task of `rows` rows allocates under
    /// [`RowPar::Rayon`] (the conv kernels' `patches` + transposed product; a
    /// dense block writes straight into the output and allocates nothing).
    /// This is the formula the kernels allocate with, so the program's
    /// admission estimate ([`super::ZonoProgram::row_task_scratch_bytes`])
    /// and the kernels cannot drift apart.
    #[must_use]
    pub fn task_scratch_elems(&self, rows: usize) -> usize {
        self.task_scratch_elems_chunked(rows, CHUNK_ELEMS)
    }

    pub(crate) fn task_scratch_elems_chunked(&self, rows: usize, chunk_elems: usize) -> usize {
        match self {
            Self::Dense { .. } => 0,
            Self::Conv2d {
                geom, out_channels, ..
            } => {
                let (hw, pl) = (geom.grid_len(), geom.patch_len());
                let chunk = rows_per_chunk(hw * pl.max(*out_channels), chunk_elems).min(rows);
                chunk * hw * (pl + out_channels)
            }
            Self::ConvTranspose2d {
                geom, in_channels, ..
            } => {
                let (hw, pl) = (geom.grid_len(), geom.patch_len());
                let chunk = rows_per_chunk(hw * pl.max(*in_channels), chunk_elems).min(rows);
                chunk * hw * (in_channels + pl)
            }
        }
    }

    /// `out[r] = A^T a[r]` (or `|A|^T a[r]` when `abs`) for `rows` rows.
    pub fn adjoint_rows(
        &self,
        a: &[f64],
        rows: usize,
        abs: bool,
        out: &mut [f64],
        deadline: Option<Instant>,
    ) -> Result<(), ZonoRefusal> {
        let (n_in, n_out) = (self.n_in(), self.n_out());
        if a.len() != rows * n_out || out.len() != rows * n_in {
            return Err(ZonoRefusal::Internal(format!(
                "kernel adjoint: {} inputs / {} outputs for {rows} rows of {n_out} -> {n_in}",
                a.len(),
                out.len()
            )));
        }
        if rows == 0 {
            return Ok(());
        }
        match self {
            Self::Dense { w, w_abs, .. } => {
                let w = if abs { w_abs } else { w };
                let rhs = MatRef::from_row_major_slice(w, n_out, n_in);
                let block = (CHUNK_ELEMS / n_in.max(n_out).max(1)).max(1);
                for r0 in (0..rows).step_by(block) {
                    check_deadline(deadline)?;
                    let b = block.min(rows - r0);
                    let lhs =
                        MatRef::from_row_major_slice(&a[r0 * n_out..(r0 + b) * n_out], b, n_out);
                    gemm_into(&mut out[r0 * n_in..(r0 + b) * n_in], b, n_in, lhs, rhs);
                }
                Ok(())
            }
            Self::Conv2d {
                geom,
                out_channels,
                k,
                k_abs,
            } => {
                let k = if abs { k_abs } else { k };
                conv2d_adjoint_rows(a, rows, geom, *out_channels, k, out, deadline)
            }
            Self::ConvTranspose2d {
                geom,
                in_channels,
                k,
                k_abs,
            } => {
                let k = if abs { k_abs } else { k };
                conv_transpose2d_adjoint_rows(a, rows, geom, *in_channels, k, out, deadline)
            }
        }
    }
}

/// Rows per chunk so that the larger scratch buffer stays near `chunk_elems`
/// ([`CHUNK_ELEMS`] in production).
fn rows_per_chunk(per_row_elems: usize, chunk_elems: usize) -> usize {
    (chunk_elems / per_row_elems.max(1)).max(1)
}

/// Rows per dense GEMM block (the reference's `block`).
fn dense_block_rows(n_in: usize, n_out: usize, chunk_elems: usize) -> usize {
    (chunk_elems / n_in.max(n_out).max(1)).max(1)
}

/// The dense forward: `out` block `r0..r0 + b` is `x` block times `W^T`, one
/// `(b, n_in) x (n_in, n_out)` GEMM per block. Under [`RowPar::Rayon`] the
/// blocks are rayon tasks over disjoint `out` slices, each making the SAME
/// call on the SAME slices as the sequential loop.
#[allow(clippy::too_many_arguments)]
fn dense_forward_rows(
    x: &[f64],
    rows: usize,
    n_in: usize,
    n_out: usize,
    w: &[f64],
    block: usize,
    out: &mut [f64],
    deadline: Option<Instant>,
    par: RowPar,
) -> Result<usize, ZonoRefusal> {
    let tasks = rows.div_ceil(block);
    if par == RowPar::Rayon && tasks > 1 && n_out > 0 {
        out.par_chunks_mut(block * n_out)
            .enumerate()
            .try_for_each(|(t, dst)| {
                check_deadline(deadline)?;
                let r0 = t * block;
                let b = block.min(rows - r0);
                // W row-major (n_out x n_in) is W^T column-major (n_in x n_out).
                let rhs = MatRef::from_column_major_slice(w, n_in, n_out);
                let lhs = MatRef::from_row_major_slice(&x[r0 * n_in..(r0 + b) * n_in], b, n_in);
                gemm_into(dst, b, n_out, lhs, rhs);
                Ok(())
            })?;
        return Ok(tasks);
    }
    // W row-major (n_out x n_in) is W^T column-major (n_in x n_out).
    let rhs = MatRef::from_column_major_slice(w, n_in, n_out);
    for r0 in (0..rows).step_by(block) {
        check_deadline(deadline)?;
        let b = block.min(rows - r0);
        let lhs = MatRef::from_row_major_slice(&x[r0 * n_in..(r0 + b) * n_in], b, n_in);
        gemm_into(&mut out[r0 * n_out..(r0 + b) * n_out], b, n_out, lhs, rhs);
    }
    Ok(tasks)
}

/// One reference block of the Conv2d forward: `b` rows of `x_block` into the
/// `b` rows of `dst`, through `patches` (at least `b * hw * pl`) and `z` (at
/// least `b * hw * out_channels`). Both schedules call exactly this.
#[allow(clippy::too_many_arguments)]
fn conv2d_forward_block(
    x_block: &[f64],
    b: usize,
    geom: &ConvGeom,
    out_channels: usize,
    k: &[f64],
    patches: &mut [f64],
    z: &mut [f64],
    dst_block: &mut [f64],
) {
    let (hw, pl, n_in) = (geom.grid_len(), geom.patch_len(), geom.image_len());
    let n_out = out_channels * hw;
    // k row-major (Cout x pl) is k^T column-major (pl x Cout).
    let rhs = MatRef::from_column_major_slice(k, pl, out_channels);
    for bi in 0..b {
        im2col(
            &x_block[bi * n_in..(bi + 1) * n_in],
            geom,
            &mut patches[bi * hw * pl..(bi + 1) * hw * pl],
        );
    }
    let lhs = MatRef::from_row_major_slice(&patches[..b * hw * pl], b * hw, pl);
    gemm_into(
        &mut z[..b * hw * out_channels],
        b * hw,
        out_channels,
        lhs,
        rhs,
    );
    for bi in 0..b {
        let dst = &mut dst_block[bi * n_out..(bi + 1) * n_out];
        let src = &z[bi * hw * out_channels..(bi + 1) * hw * out_channels];
        for p in 0..hw {
            for c in 0..out_channels {
                dst[c * hw + p] = src[p * out_channels + c];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv2d_forward_rows(
    x: &[f64],
    rows: usize,
    geom: &ConvGeom,
    out_channels: usize,
    k: &[f64],
    out: &mut [f64],
    deadline: Option<Instant>,
    par: RowPar,
    chunk_elems: usize,
) -> Result<usize, ZonoRefusal> {
    let (hw, pl, n_in) = (geom.grid_len(), geom.patch_len(), geom.image_len());
    let n_out = out_channels * hw;
    let chunk = rows_per_chunk(hw * pl.max(out_channels), chunk_elems).min(rows);
    let tasks = rows.div_ceil(chunk);
    if par == RowPar::Rayon && tasks > 1 && n_out > 0 {
        out.par_chunks_mut(chunk * n_out)
            .enumerate()
            .try_for_each(|(t, dst)| {
                check_deadline(deadline)?;
                let r0 = t * chunk;
                let b = chunk.min(rows - r0);
                // Per-task scratch of exactly the block's size (the
                // reference's buffers sliced to `b`).
                let mut patches = vec![0.0f64; b * hw * pl];
                let mut z = vec![0.0f64; b * hw * out_channels];
                conv2d_forward_block(
                    &x[r0 * n_in..(r0 + b) * n_in],
                    b,
                    geom,
                    out_channels,
                    k,
                    &mut patches,
                    &mut z,
                    dst,
                );
                Ok(())
            })?;
        return Ok(tasks);
    }
    let mut patches = vec![0.0f64; chunk * hw * pl];
    let mut z = vec![0.0f64; chunk * hw * out_channels];
    for r0 in (0..rows).step_by(chunk) {
        check_deadline(deadline)?;
        let b = chunk.min(rows - r0);
        conv2d_forward_block(
            &x[r0 * n_in..(r0 + b) * n_in],
            b,
            geom,
            out_channels,
            k,
            &mut patches,
            &mut z,
            &mut out[r0 * n_out..(r0 + b) * n_out],
        );
    }
    Ok(tasks)
}

fn conv2d_adjoint_rows(
    a: &[f64],
    rows: usize,
    geom: &ConvGeom,
    out_channels: usize,
    k: &[f64],
    out: &mut [f64],
    deadline: Option<Instant>,
) -> Result<(), ZonoRefusal> {
    let (hw, pl, n_in) = (geom.grid_len(), geom.patch_len(), geom.image_len());
    let n_out = out_channels * hw;
    let chunk = rows_per_chunk(hw * pl.max(out_channels), CHUNK_ELEMS).min(rows);
    let rhs = MatRef::from_row_major_slice(k, out_channels, pl);
    let mut at = vec![0.0f64; chunk * hw * out_channels];
    let mut dp = vec![0.0f64; chunk * hw * pl];
    for r0 in (0..rows).step_by(chunk) {
        check_deadline(deadline)?;
        let b = chunk.min(rows - r0);
        for bi in 0..b {
            let src = &a[(r0 + bi) * n_out..(r0 + bi + 1) * n_out];
            let dst = &mut at[bi * hw * out_channels..(bi + 1) * hw * out_channels];
            for c in 0..out_channels {
                for p in 0..hw {
                    dst[p * out_channels + c] = src[c * hw + p];
                }
            }
        }
        let lhs = MatRef::from_row_major_slice(&at[..b * hw * out_channels], b * hw, out_channels);
        gemm_into(&mut dp[..b * hw * pl], b * hw, pl, lhs, rhs);
        for bi in 0..b {
            let dst = &mut out[(r0 + bi) * n_in..(r0 + bi + 1) * n_in];
            dst.fill(0.0);
            col2im_add(&dp[bi * hw * pl..(bi + 1) * hw * pl], geom, dst);
        }
    }
    Ok(())
}

/// One reference block of the ConvTranspose2d forward: `b` rows of
/// `x_block` into the `b` rows of `dst_block`, through `xt` (at least
/// `b * hw * in_channels`) and `t` (at least `b * hw * pl`). Both schedules
/// call exactly this.
#[allow(clippy::too_many_arguments)]
fn conv_transpose2d_forward_block(
    x_block: &[f64],
    b: usize,
    geom: &ConvGeom,
    in_channels: usize,
    k: &[f64],
    xt: &mut [f64],
    t: &mut [f64],
    dst_block: &mut [f64],
) {
    let (hw, pl, n_out) = (geom.grid_len(), geom.patch_len(), geom.image_len());
    let n_in = in_channels * hw;
    let rhs = MatRef::from_row_major_slice(k, in_channels, pl);
    for bi in 0..b {
        let src = &x_block[bi * n_in..(bi + 1) * n_in];
        let dst = &mut xt[bi * hw * in_channels..(bi + 1) * hw * in_channels];
        for c in 0..in_channels {
            for p in 0..hw {
                dst[p * in_channels + c] = src[c * hw + p];
            }
        }
    }
    let lhs = MatRef::from_row_major_slice(&xt[..b * hw * in_channels], b * hw, in_channels);
    gemm_into(&mut t[..b * hw * pl], b * hw, pl, lhs, rhs);
    for bi in 0..b {
        let dst = &mut dst_block[bi * n_out..(bi + 1) * n_out];
        dst.fill(0.0);
        col2im_add(&t[bi * hw * pl..(bi + 1) * hw * pl], geom, dst);
    }
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose2d_forward_rows(
    x: &[f64],
    rows: usize,
    geom: &ConvGeom,
    in_channels: usize,
    k: &[f64],
    out: &mut [f64],
    deadline: Option<Instant>,
    par: RowPar,
    chunk_elems: usize,
) -> Result<usize, ZonoRefusal> {
    let (hw, pl, n_out) = (geom.grid_len(), geom.patch_len(), geom.image_len());
    let n_in = in_channels * hw;
    let chunk = rows_per_chunk(hw * pl.max(in_channels), chunk_elems).min(rows);
    let tasks = rows.div_ceil(chunk);
    if par == RowPar::Rayon && tasks > 1 && n_out > 0 {
        out.par_chunks_mut(chunk * n_out)
            .enumerate()
            .try_for_each(|(task, dst)| {
                check_deadline(deadline)?;
                let r0 = task * chunk;
                let b = chunk.min(rows - r0);
                let mut xt = vec![0.0f64; b * hw * in_channels];
                let mut t = vec![0.0f64; b * hw * pl];
                conv_transpose2d_forward_block(
                    &x[r0 * n_in..(r0 + b) * n_in],
                    b,
                    geom,
                    in_channels,
                    k,
                    &mut xt,
                    &mut t,
                    dst,
                );
                Ok(())
            })?;
        return Ok(tasks);
    }
    let mut xt = vec![0.0f64; chunk * hw * in_channels];
    let mut t = vec![0.0f64; chunk * hw * pl];
    for r0 in (0..rows).step_by(chunk) {
        check_deadline(deadline)?;
        let b = chunk.min(rows - r0);
        conv_transpose2d_forward_block(
            &x[r0 * n_in..(r0 + b) * n_in],
            b,
            geom,
            in_channels,
            k,
            &mut xt,
            &mut t,
            &mut out[r0 * n_out..(r0 + b) * n_out],
        );
    }
    Ok(tasks)
}

fn conv_transpose2d_adjoint_rows(
    a: &[f64],
    rows: usize,
    geom: &ConvGeom,
    in_channels: usize,
    k: &[f64],
    out: &mut [f64],
    deadline: Option<Instant>,
) -> Result<(), ZonoRefusal> {
    let (hw, pl, n_out) = (geom.grid_len(), geom.patch_len(), geom.image_len());
    let n_in = in_channels * hw;
    let chunk = rows_per_chunk(hw * pl.max(in_channels), CHUNK_ELEMS).min(rows);
    // k row-major (Cin x pl) is k^T column-major (pl x Cin).
    let rhs = MatRef::from_column_major_slice(k, pl, in_channels);
    let mut patches = vec![0.0f64; chunk * hw * pl];
    let mut dxt = vec![0.0f64; chunk * hw * in_channels];
    for r0 in (0..rows).step_by(chunk) {
        check_deadline(deadline)?;
        let b = chunk.min(rows - r0);
        for bi in 0..b {
            im2col(
                &a[(r0 + bi) * n_out..(r0 + bi + 1) * n_out],
                geom,
                &mut patches[bi * hw * pl..(bi + 1) * hw * pl],
            );
        }
        let lhs = MatRef::from_row_major_slice(&patches[..b * hw * pl], b * hw, pl);
        gemm_into(
            &mut dxt[..b * hw * in_channels],
            b * hw,
            in_channels,
            lhs,
            rhs,
        );
        for bi in 0..b {
            let dst = &mut out[(r0 + bi) * n_in..(r0 + bi + 1) * n_in];
            let src = &dxt[bi * hw * in_channels..(bi + 1) * hw * in_channels];
            for c in 0..in_channels {
                for p in 0..hw {
                    dst[c * hw + p] = src[p * in_channels + c];
                }
            }
        }
    }
    Ok(())
}
