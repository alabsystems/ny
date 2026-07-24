// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{NyError, Result};

use super::{PatchesData, UnstableIdx};

/// Validate patches tensor shape and return a reference to it.
pub(super) fn validate_patches_shape(
    patches_data: &PatchesData,
    row_count: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
) -> Result<(&ArrayD<f32>, bool)> {
    let patches = patches_data.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("PatchesData: not identity but patches tensor is None".into())
    })?;
    let shape = patches.shape();
    match shape.len() {
        6 => {
            if shape[0] != out_c || shape[1] != out_h || shape[2] != out_w || shape[3] != in_c {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_c, out_h, out_w, in_c],
                    got: vec![shape[0], shape[1], shape[2], shape[3]],
                });
            }
            Ok((patches, false))
        }
        7 => {
            if shape[0] != row_count
                || shape[1] != out_c
                || shape[2] != out_h
                || shape[3] != out_w
                || shape[4] != in_c
            {
                return Err(NyError::ShapeMismatch {
                    expected: vec![row_count, out_c, out_h, out_w, in_c],
                    got: vec![shape[0], shape[1], shape[2], shape[3], shape[4]],
                });
            }
            Ok((patches, true))
        }
        _ => Err(NyError::ShapeMismatch {
            expected: vec![6, 7],
            got: vec![shape.len()],
        }),
    }
}

/// Precompute input-flat-index mapping using `inplace_unfold` from ny-tensor.
pub(super) fn compute_unfold_index_map(
    patches_data: &PatchesData,
    kh: usize,
    kw: usize,
) -> Result<ArrayD<f32>> {
    let (in_c, in_h, in_w) = patches_data.input_shape;
    let index_image = ArrayD::from_shape_fn(IxDyn(&[in_c, in_h, in_w]), |idx| {
        (idx[0] * in_h * in_w + idx[1] * in_w + idx[2] + 1) as f32
    });
    ny_tensor::inplace_unfold(
        &index_image,
        (kh, kw),
        patches_data.stride,
        patches_data.padding,
    )
}

/// Precompute, per output spatial position `(oh, ow)`, the list of
/// `(block_offset, in_flat)` pairs for the kernel taps that land inside the
/// input (index_map entry `> 0`).
///
/// `block_offset` is the offset of the tap within the contiguous
/// `(in_c, kh, kw)` block (i.e. `(ic*kh + ki)*kw + kj`), shared by every output
/// channel / row since the receptive-field geometry depends only on `(oh, ow)`.
/// Hoisting this out of the `oc`/`row` loops removes the redundant
/// `index_map[[oh, ow, ic, ki, kj]] > 0` test (and its strided ndarray indexing)
/// that the original code repeated for every `oc`/`row`.
///
/// The pairs are emitted in `(oh, ow, ic, ki, kj)` order, identical to the
/// original loop nesting, so the downstream `+=` accumulation order — and thus
/// the float result — is bit-for-bit unchanged.
pub(super) struct UnfoldPlan {
    /// For position p = oh*out_w + ow: `(block_offset, in_flat)` taps in scan order.
    taps: Vec<(usize, usize)>,
    /// `offsets[p]..offsets[p + 1]` indexes the taps for position p.
    offsets: Vec<usize>,
    out_w: usize,
}

impl UnfoldPlan {
    pub(super) fn build(index_map: &[f32], out_h: usize, out_w: usize, block: usize) -> Self {
        let positions = out_h * out_w;
        let mut taps = Vec::new();
        let mut offsets = Vec::with_capacity(positions + 1);
        offsets.push(0);
        for p in 0..positions {
            let base = p * block;
            let map_block = &index_map[base..base + block];
            for (block_offset, &idx_1based) in map_block.iter().enumerate() {
                if idx_1based > 0.0 {
                    taps.push((block_offset, (idx_1based as usize) - 1));
                }
            }
            offsets.push(taps.len());
        }
        UnfoldPlan {
            taps,
            offsets,
            out_w,
        }
    }

    #[inline]
    pub(super) fn taps_for(&self, oh: usize, ow: usize) -> &[(usize, usize)] {
        let p = oh * self.out_w + ow;
        &self.taps[self.offsets[p]..self.offsets[p + 1]]
    }
}

/// Build an [`UnfoldPlan`] from an (index-map) tensor, materializing a flat view
/// first (contiguous fast path, strided copy fallback). Shared by the dense
/// scatter and the sparse patches-native concretize
/// (`PatchesLinearBounds::concretize_sound_sparse`), so both derive the exact
/// same receptive-field tap geometry.
pub(super) fn build_unfold_plan(
    index_map: &ArrayD<f32>,
    out_h: usize,
    out_w: usize,
    block: usize,
) -> UnfoldPlan {
    let mut scratch = Vec::new();
    let flat = as_flat(index_map, &mut scratch);
    UnfoldPlan::build(flat, out_h, out_w, block)
}

/// Index the index_map / patches as flat row-major slices when contiguous,
/// matching the layout `inplace_unfold` and the patches builders produce.
/// Falls back to a strided copy (rare; only if some upstream produced a
/// non-standard layout) so results stay correct.
pub(super) fn as_flat<'a>(arr: &'a ArrayD<f32>, scratch: &'a mut Vec<f32>) -> &'a [f32] {
    match arr.as_slice() {
        Some(s) => s,
        None => {
            scratch.clear();
            scratch.extend(arr.iter().copied());
            scratch.as_slice()
        }
    }
}

/// Scatter patches coefficients into dense matrix using a precomputed unfold index map.
///
/// Row range (#patches-row-range): materializes only output positions with
/// `out_flat = oc*out_h*out_w + oh*out_w + ow` in `[row_start, row_end)`,
/// written to dense row `out_flat - row_start`. Rows are fully independent
/// (each output position owns its dense row and its per-row `+=` order is
/// unchanged), so any range split is bit-identical to the corresponding slice
/// of the full `(0, out_dim)` scatter.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_with_unfold_map(
    dense: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    row_start: usize,
    row_end: usize,
) {
    let block = in_c * kh * kw;
    let mut idx_scratch = Vec::new();
    let mut pat_scratch = Vec::new();
    let index_flat = as_flat(index_map, &mut idx_scratch);
    let plan = UnfoldPlan::build(index_flat, out_h, out_w, block);
    let patches_flat = as_flat(patches, &mut pat_scratch);

    let positions = out_h * out_w;
    // Ascending `out_flat` decomposed to (oc, oh, ow) is exactly the original
    // oc -> oh -> ow nesting restricted to the range (`pat_base` telescopes to
    // `out_flat * block`).
    for out_flat in row_start..row_end.min(out_c * positions) {
        let pos = out_flat % positions;
        let oh = pos / out_w;
        let ow = pos % out_w;
        let pat_base = out_flat * block;
        let row = dense
            .row_mut(out_flat - row_start)
            .into_slice()
            .expect("dense rows are contiguous");
        for &(block_offset, in_flat) in plan.taps_for(oh, ow) {
            row[in_flat] += patches_flat[pat_base + block_offset];
        }
    }
}

/// Accumulate, per dense cell `(out_flat, in_flat)`, the number of patch taps that
/// land there (`count`) and the sum of their absolute scattered magnitudes
/// (`absacc`), using the SAME unfold geometry as [`scatter_with_unfold_map`]. Used
/// by `to_dense` to build the overlap-aware certified coefficient error:
/// `err[i,j] = next_up(count[i,j]·err_row[i] + γ_count^f32·absacc[i,j])`.
///
/// 6D-only (f32 accumulators are sufficient because the 6D per-cell tap count is
/// 0/1 — each `in_flat` occurs at most once per output position); the 7D
/// explicit-rows layout uses [`scatter_rows_err_accumulators`] instead.
///
/// Row range (#patches-row-range): same `[row_start, row_end)` output-position
/// window as [`scatter_with_unfold_map`], writing accumulator row
/// `out_flat - row_start` — per-row independent, bit-identical to the slice of
/// the full accumulation.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_err_accumulators(
    count: &mut Array2<f32>,
    absacc: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    row_start: usize,
    row_end: usize,
) {
    let block = in_c * kh * kw;
    let mut idx_scratch = Vec::new();
    let mut pat_scratch = Vec::new();
    let index_flat = as_flat(index_map, &mut idx_scratch);
    let plan = UnfoldPlan::build(index_flat, out_h, out_w, block);
    let patches_flat = as_flat(patches, &mut pat_scratch);

    let positions = out_h * out_w;
    for out_flat in row_start..row_end.min(out_c * positions) {
        let pos = out_flat % positions;
        let oh = pos / out_w;
        let ow = pos % out_w;
        let pat_base = out_flat * block;
        let crow = count
            .row_mut(out_flat - row_start)
            .into_slice()
            .expect("count rows are contiguous");
        let arow = absacc
            .row_mut(out_flat - row_start)
            .into_slice()
            .expect("absacc rows are contiguous");
        for &(block_offset, in_flat) in plan.taps_for(oh, ow) {
            crow[in_flat] += 1.0;
            arow[in_flat] += patches_flat[pat_base + block_offset].abs();
        }
    }
}

/// Scatter row-aware dense patches into a dense matrix using an unfold index map.
///
/// Row range (#patches-row-range): materializes spec rows `[row_start,
/// row_end)` (axis 0 of the 7D patches tensor), writing dense row
/// `row - row_start`. Per-row independent, bit-identical to the slice of the
/// full `(0, row_count)` scatter.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_rows_with_unfold_map(
    dense: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    row_start: usize,
    row_end: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) {
    let block = in_c * kh * kw;
    let positions = out_h * out_w;
    let per_row = out_c * positions * block;
    let mut idx_scratch = Vec::new();
    let mut pat_scratch = Vec::new();
    let index_flat = as_flat(index_map, &mut idx_scratch);
    let plan = UnfoldPlan::build(index_flat, out_h, out_w, block);
    let patches_flat = as_flat(patches, &mut pat_scratch);

    for row in row_start..row_end {
        let row_base = row * per_row;
        let out_row = dense
            .row_mut(row - row_start)
            .into_slice()
            .expect("dense rows are contiguous");
        for oc in 0..out_c {
            let oc_base = row_base + oc * positions * block;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let pat_base = oc_base + (oh * out_w + ow) * block;
                    for &(block_offset, in_flat) in plan.taps_for(oh, ow) {
                        out_row[in_flat] += patches_flat[pat_base + block_offset];
                    }
                }
            }
        }
    }
}

/// Accumulate, per dense cell `(spec_row, in_flat)`, the number of plan taps
/// that land there (`count`) and the **f64** sum of their absolute scattered
/// magnitudes (`absacc`), mirroring [`scatter_rows_with_unfold_map`]'s tap
/// geometry EXACTLY (same `UnfoldPlan`, same
/// `row -> oc -> oh -> ow -> (ic, ki, kj)` loop order). Used by `to_dense` to
/// build the overlap-aware certified coefficient error for the 7D
/// explicit-rows layout (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §3):
/// `err[r,j] = next_up(count[r,j]·err_row[r] + γ_count^f32·absacc[r,j])`.
///
/// The accumulators are f64, deviating from the f32 6D
/// [`scatter_err_accumulators`] (spec R1): the γ-chaining argument
/// `γ(N)·S_hat >= γ(N−1)·S_true` needs `S_hat` within `2^-28` relative of
/// `S_true`, which f64 same-sign summation guarantees for every `N < 2^24`;
/// the f32 mirror only closes for `N <~ 4096`, while 7D per-cell tap counts on
/// the scored workload reach `~1e4..5e5`. f64 `count += 1.0` is exact to
/// `2^53`, also eliminating f32 count saturation at `2^24`.
///
/// Row range (#patches-row-range): same `[row_start, row_end)` spec-row window
/// as [`scatter_rows_with_unfold_map`], writing accumulator row
/// `row - row_start` — per-row independent, bit-identical to the slice of the
/// full accumulation.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_rows_err_accumulators(
    count: &mut Array2<f64>,
    absacc: &mut Array2<f64>,
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    row_start: usize,
    row_end: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) {
    let block = in_c * kh * kw;
    let positions = out_h * out_w;
    let per_row = out_c * positions * block;
    let mut idx_scratch = Vec::new();
    let mut pat_scratch = Vec::new();
    let index_flat = as_flat(index_map, &mut idx_scratch);
    let plan = UnfoldPlan::build(index_flat, out_h, out_w, block);
    let patches_flat = as_flat(patches, &mut pat_scratch);

    for row in row_start..row_end {
        let row_base = row * per_row;
        let crow = count
            .row_mut(row - row_start)
            .into_slice()
            .expect("count rows are contiguous");
        let arow = absacc
            .row_mut(row - row_start)
            .into_slice()
            .expect("absacc rows are contiguous");
        for oc in 0..out_c {
            let oc_base = row_base + oc * positions * block;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let pat_base = oc_base + (oh * out_w + ow) * block;
                    for &(block_offset, in_flat) in plan.taps_for(oh, ow) {
                        crow[in_flat] += 1.0;
                        arow[in_flat] += f64::from(patches_flat[pat_base + block_offset].abs());
                    }
                }
            }
        }
    }
}

/// Scatter sparse patches into a dense matrix using an unfold index map.
///
/// Row range (#patches-row-range): sparse rows whose flat output position
/// falls outside `[row_start, row_end)` are skipped; in-range rows write dense
/// row `out_flat - row_start`. Sparse-index visiting order is unchanged, so
/// any range split is bit-identical to the corresponding slice of the full
/// scatter.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_sparse_with_unfold_map(
    dense: &mut Array2<f32>,
    sparse_patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    idx: &UnstableIdx,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    row_start: usize,
    row_end: usize,
) {
    let block = in_c * kh * kw;
    let mut idx_scratch = Vec::new();
    let mut pat_scratch = Vec::new();
    let index_flat = as_flat(index_map, &mut idx_scratch);
    let plan = UnfoldPlan::build(index_flat, out_h, out_w, block);
    let patches_flat = as_flat(sparse_patches, &mut pat_scratch);

    for (i, ((&c, &h), &w)) in idx
        .channels
        .iter()
        .zip(idx.heights.iter())
        .zip(idx.widths.iter())
        .enumerate()
    {
        let out_flat = c * out_h * out_w + h * out_w + w;
        if out_flat < row_start || out_flat >= row_end {
            continue;
        }
        let pat_base = i * block;
        let row = dense
            .row_mut(out_flat - row_start)
            .into_slice()
            .expect("dense rows are contiguous");
        for &(block_offset, in_flat) in plan.taps_for(h, w) {
            row[in_flat] += patches_flat[pat_base + block_offset];
        }
    }
}

/// Scatter row-aware sparse patches into a dense matrix using an unfold index map.
///
/// Row range (#patches-row-range): materializes spec rows `[row_start,
/// row_end)` (axis 0 of the 5D sparse patches tensor), writing dense row
/// `row - row_start` — per-row independent, bit-identical to the slice of the
/// full `(0, row_count)` scatter.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_sparse_rows_with_unfold_map(
    dense: &mut Array2<f32>,
    sparse_patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    row_start: usize,
    row_end: usize,
    idx: &UnstableIdx,
    in_c: usize,
    kh: usize,
    kw: usize,
) {
    let block = in_c * kh * kw;
    let unstable_size = idx.len();
    let index_shape = index_map.shape();
    let (out_h, out_w) = (index_shape[0], index_shape[1]);
    let mut idx_scratch = Vec::new();
    let mut pat_scratch = Vec::new();
    let index_flat = as_flat(index_map, &mut idx_scratch);
    let plan = UnfoldPlan::build(index_flat, out_h, out_w, block);
    let patches_flat = as_flat(sparse_patches, &mut pat_scratch);

    for row in row_start..row_end {
        let out_row = dense
            .row_mut(row - row_start)
            .into_slice()
            .expect("dense rows are contiguous");
        for (i, (&h, &w)) in idx.heights.iter().zip(idx.widths.iter()).enumerate() {
            let pat_base = (row * unstable_size + i) * block;
            for &(block_offset, in_flat) in plan.taps_for(h, w) {
                out_row[in_flat] += patches_flat[pat_base + block_offset];
            }
        }
    }
}
