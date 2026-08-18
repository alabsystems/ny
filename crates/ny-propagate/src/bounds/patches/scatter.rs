// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, ArrayD};
use ny_core::{checked_shape_product, NyError, Result};
use std::mem::size_of;

#[cfg(test)]
use ndarray::IxDyn;

#[cfg(test)]
use std::ops::Deref;

use super::{PatchesData, PatchesMaterializationDeadline, PatchesMemoryAdmission, UnstableIdx};

#[inline]
fn unfold_plan_storage_bytes(
    tap_capacity: usize,
    offset_capacity: usize,
    legacy_f32_capacity: usize,
) -> usize {
    tap_capacity
        .saturating_mul(size_of::<(usize, usize)>())
        .saturating_add(offset_capacity.saturating_mul(size_of::<usize>()))
        .saturating_add(legacy_f32_capacity.saturating_mul(size_of::<f32>()))
}

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

/// Exact receptive-field map used by every patches scatter consumer.
///
/// Production storage contains only zero-based `usize` input indices.  Padding
/// taps are omitted from [`UnfoldPlan::taps`], after being represented as
/// `None` by the checked geometry mapper.  This removes the historical
/// one-based f32 index carrier, where adjacent columns alias above `2^24`.
pub(super) struct UnfoldIndexMap {
    plan: UnfoldPlan,
    shape: [usize; 5],
    len: usize,

    // Existing equivalence tests intentionally retain their pre-optimization
    // f32 reference implementation.  Keep that view test-only: no production
    // consumer can accidentally recover the aliased carrier through `Deref`.
    #[cfg(test)]
    legacy_f32: ArrayD<f32>,
    #[cfg(test)]
    legacy_capacity: usize,
}

impl UnfoldIndexMap {
    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Heap bytes retained by this exact unfold map while a scatter is live.
    ///
    /// The map is part of the Patches-to-Dense peak, so callers must account
    /// for it together with the dense pair rather than admitting each
    /// allocation independently.
    pub(super) fn memory_bytes(&self) -> usize {
        #[cfg(test)]
        let legacy_capacity = self.legacy_capacity;
        #[cfg(not(test))]
        let legacy_capacity = 0usize;
        unfold_plan_storage_bytes(
            self.plan.taps.capacity(),
            self.plan.offsets.capacity(),
            legacy_capacity,
        )
    }
}

#[cfg(test)]
impl Deref for UnfoldIndexMap {
    type Target = ArrayD<f32>;

    fn deref(&self) -> &Self::Target {
        &self.legacy_f32
    }
}

/// Precompute an exact integer input-flat-index map for any validated patch
/// geometry. Affine and anchored mappings share the same compact scatter plan.
pub(super) fn compute_unfold_index_map(
    patches_data: &PatchesData,
    kh: usize,
    kw: usize,
) -> Result<UnfoldIndexMap> {
    let mut deadline = PatchesMaterializationDeadline::new(None);
    compute_unfold_index_map_with_deadline(patches_data, kh, kw, 0, &mut deadline)
}

/// Deadline-aware form of [`compute_unfold_index_map`]. The exact geometry and
/// emission order are unchanged; only cooperative checks are interleaved.
pub(super) fn compute_unfold_index_map_with_deadline(
    patches_data: &PatchesData,
    kh: usize,
    kw: usize,
    resident_base_bytes: usize,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<UnfoldIndexMap> {
    deadline.checkpoint("before unfold-map validation")?;
    let (in_c, in_h, in_w) = patches_data.input_shape;
    let _in_dim = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "patch input size overflows: {in_c} * {in_h} * {in_w}"
        ))
    })?;
    let geometry = patches_data.validated_geometry_for_with_poll((kh, kw), deadline)?;
    deadline.checkpoint("after unfold-map geometry validation")?;
    let (out_h, out_w) = (patches_data.output_shape.1, patches_data.output_shape.2);

    let block = checked_shape_product(&[in_c, kh, kw]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "patch unfold block size overflows: {in_c} * {kh} * {kw}"
        ))
    })?;
    let positions = checked_shape_product(&[out_h, out_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "patch unfold position count overflows: {out_h} * {out_w}"
        ))
    })?;
    let len = positions.checked_mul(block).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "patch unfold map size overflows: {positions} * {block}"
        ))
    })?;
    let offset_count = positions
        .checked_add(1)
        .ok_or_else(|| NyError::InvalidSpec("patch unfold offset count overflows usize".into()))?;

    // Budget the worst-case fully in-range plan before asking the allocator for
    // capacity. In cfg(test), include the legacy f32 mirror as well: it is
    // touched after plan construction and used to be able to cgroup-OOM even
    // when both Vec reservations individually succeeded.
    #[cfg(test)]
    let legacy_capacity = len;
    #[cfg(not(test))]
    let legacy_capacity = 0usize;
    let required_bytes = resident_base_bytes.saturating_add(unfold_plan_storage_bytes(
        len,
        offset_count,
        legacy_capacity,
    ));
    let mut admission = PatchesMemoryAdmission::check(required_bytes, "patch unfold plan")?;

    let mut taps = Vec::new();
    deadline.checkpoint("before unfold-map tap allocation")?;
    taps.try_reserve_exact(len)
        .map_err(|_| admission.allocation_error("patch unfold tap allocation"))?;
    deadline.checkpoint("after unfold-map tap allocation")?;
    admission.reconcile_vec_capacity::<(usize, usize)>(
        len,
        taps.capacity(),
        "patch unfold tap allocation",
    )?;
    let mut offsets = Vec::new();
    deadline.checkpoint("before unfold-map offset allocation")?;
    offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| admission.allocation_error("patch unfold offset allocation"))?;
    deadline.checkpoint("after unfold-map offset allocation")?;
    admission.reconcile_vec_capacity::<usize>(
        offset_count,
        offsets.capacity(),
        "patch unfold offset allocation",
    )?;
    offsets.push(0);

    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut block_offset = 0usize;
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let input_flat = geometry.input_flat_index(
                            (oh, ow),
                            ic,
                            (ki, kj),
                            patches_data.input_shape,
                        )?;
                        if let Some(input_flat) = input_flat {
                            taps.push((block_offset, input_flat));
                        }
                        block_offset += 1;
                        deadline.work(1, "during unfold-map fill")?;
                    }
                }
            }
            debug_assert_eq!(block_offset, block);
            offsets.push(taps.len());
            deadline.work(1, "during unfold-map position fill")?;
        }
    }
    deadline.checkpoint("after unfold-map fill")?;

    let plan = UnfoldPlan {
        taps,
        offsets,
        out_h,
        out_w,
        block,
    };

    #[cfg(test)]
    let (legacy_f32, legacy_capacity) = {
        let mut data = Vec::new();
        deadline.checkpoint("before unfold-map legacy-test allocation")?;
        data.try_reserve_exact(len)
            .map_err(|_| admission.allocation_error("patch unfold legacy-test allocation"))?;
        deadline.checkpoint("after unfold-map legacy-test allocation")?;
        admission.reconcile_vec_capacity::<f32>(
            len,
            data.capacity(),
            "patch unfold legacy-test allocation",
        )?;
        let capacity = data.capacity();
        let mut filled = 0usize;
        while filled < len {
            let end = filled
                .saturating_add(PatchesMaterializationDeadline::CHECK_STRIDE)
                .min(len);
            data.resize(end, 0.0f32);
            deadline.work(end - filled, "during unfold-map legacy-test zero fill")?;
            filled = end;
        }
        for p in 0..positions {
            deadline.work(1, "during unfold-map legacy-test position fill")?;
            for &(block_offset, input_flat) in &plan.taps[plan.offsets[p]..plan.offsets[p + 1]] {
                let one_based = input_flat.checked_add(1).ok_or_else(|| {
                    NyError::InvalidSpec("patch unfold legacy-test index overflows usize".into())
                })?;
                data[p * block + block_offset] = one_based as f32;
                deadline.work(1, "during unfold-map legacy-test mirror fill")?;
            }
        }
        let array = ArrayD::from_shape_vec(IxDyn(&[out_h, out_w, in_c, kh, kw]), data).map_err(
            |error| {
                NyError::InternalError(format!(
                    "patch unfold legacy-test shape construction failed: {error}"
                ))
            },
        )?;
        (array, capacity)
    };

    deadline.checkpoint("after unfold-map construction")?;

    Ok(UnfoldIndexMap {
        plan,
        shape: [out_h, out_w, in_c, kh, kw],
        len,
        #[cfg(test)]
        legacy_f32,
        #[cfg(test)]
        legacy_capacity,
    })
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
    out_h: usize,
    out_w: usize,
    block: usize,
}

impl UnfoldPlan {
    #[inline]
    pub(super) fn block(&self) -> usize {
        self.block
    }

    #[inline]
    pub(super) fn output_size(&self) -> (usize, usize) {
        (self.out_h, self.out_w)
    }

    #[inline]
    pub(super) fn positions(&self) -> usize {
        // Construction stores exactly one terminal offset after every output
        // position, so this avoids repeating spatial multiplication in each
        // consumer.
        self.offsets.len() - 1
    }

    #[inline]
    pub(super) fn taps_for(&self, oh: usize, ow: usize) -> &[(usize, usize)] {
        debug_assert!(oh < self.out_h);
        debug_assert!(ow < self.out_w);
        let p = oh * self.out_w + ow;
        &self.taps[self.offsets[p]..self.offsets[p + 1]]
    }
}

/// Borrow the exact [`UnfoldPlan`] from an index map.
///
/// Shared by the dense scatter and the sparse patches-native concretize
/// (`PatchesLinearBounds::concretize_sound_sparse`), so both derive the exact
/// same receptive-field tap geometry.
pub(super) fn build_unfold_plan(index_map: &UnfoldIndexMap) -> &UnfoldPlan {
    &index_map.plan
}

/// Index patches as flat row-major slices when contiguous, matching the layout
/// the patches builders produce.
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

#[inline]
fn flat_scratch_bytes(arr: &ArrayD<f32>) -> usize {
    if arr.as_slice().is_some() {
        0
    } else {
        arr.len().saturating_mul(size_of::<f32>())
    }
}

/// Fallible counterpart of [`as_flat`] for proof paths whose allocations are
/// resource-authoritative. Contiguous arrays remain a zero-allocation fast
/// path; only a genuinely strided carrier reserves scratch.
#[allow(dead_code)] // Retained no-deadline sibling API; proof paths use the polling form.
pub(super) fn try_as_flat<'a>(
    arr: &'a ArrayD<f32>,
    scratch: &'a mut Vec<f32>,
    site: &'static str,
) -> Result<&'a [f32]> {
    let mut deadline = PatchesMaterializationDeadline::new(None);
    let required_bytes = flat_scratch_bytes(arr);
    let mut admission = PatchesMemoryAdmission::check(required_bytes, site)?;
    try_as_flat_with_deadline(arr, scratch, site, &mut admission, &mut deadline)
}

/// Deadline-aware form of [`try_as_flat`]. A non-contiguous carrier is copied
/// in bounded chunks; contiguous storage stays a zero-allocation exact borrow.
pub(super) fn try_as_flat_with_deadline<'a>(
    arr: &'a ArrayD<f32>,
    scratch: &'a mut Vec<f32>,
    site: &'static str,
    admission: &mut PatchesMemoryAdmission,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<&'a [f32]> {
    match arr.as_slice() {
        Some(slice) => {
            deadline.checkpoint("after contiguous patches borrow")?;
            Ok(slice)
        }
        None => {
            scratch.clear();
            deadline.checkpoint("before strided patches scratch allocation")?;
            scratch
                .try_reserve_exact(arr.len())
                .map_err(|_| admission.allocation_error(site))?;
            deadline.checkpoint("after strided patches scratch allocation")?;
            admission.reconcile_vec_capacity::<f32>(arr.len(), scratch.capacity(), site)?;
            for value in arr.iter().copied() {
                scratch.push(value);
                deadline.work(1, "during strided patches scratch copy")?;
            }
            deadline.checkpoint("after strided patches scratch copy")?;
            Ok(scratch.as_slice())
        }
    }
}

/// Scatter patches coefficients into dense matrix using a precomputed unfold index map.
///
/// Row range (#patches-row-range): materializes only output positions with
/// `out_flat = oc*out_h*out_w + oh*out_w + ow` in `[row_start, row_end)`,
/// written to dense row `out_flat - row_start`. Rows are fully independent
/// (each output position owns its dense row and every valid input cell receives
/// at most one tap), so any range split is bit-identical to the corresponding slice
/// of the full `(0, out_dim)` scatter.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn scatter_with_unfold_map(
    dense: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    row_start: usize,
    row_end: usize,
) -> Result<()> {
    let mut deadline = PatchesMaterializationDeadline::new(None);
    let mut admission = PatchesMemoryAdmission::check(
        flat_scratch_bytes(patches),
        "patches dense scatter scratch",
    )?;
    scatter_with_unfold_map_with_deadline(
        dense,
        patches,
        index_map,
        out_c,
        out_h,
        out_w,
        in_c,
        kh,
        kw,
        row_start,
        row_end,
        &mut admission,
        &mut deadline,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_with_unfold_map_with_deadline(
    dense: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    row_start: usize,
    row_end: usize,
    admission: &mut PatchesMemoryAdmission,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<()> {
    let mut pat_scratch = Vec::new();
    let plan = build_unfold_plan(index_map);
    debug_assert_eq!(plan.output_size(), (out_h, out_w));
    debug_assert_eq!(checked_shape_product(&[in_c, kh, kw]), Some(plan.block()));
    let (_out_h, out_w) = plan.output_size();
    let block = plan.block();
    let patches_flat = try_as_flat_with_deadline(
        patches,
        &mut pat_scratch,
        "patches dense scatter scratch",
        admission,
        deadline,
    )?;

    let positions = plan.positions();
    // Ascending `out_flat` decomposed to (oc, oh, ow) is exactly the original
    // oc -> oh -> ow nesting restricted to the range (`pat_base` telescopes to
    // `out_flat * block`).
    for out_flat in row_start..row_end.min(out_c * positions) {
        deadline.work(1, "during patches dense scatter row walk")?;
        let pos = out_flat % positions;
        let oh = pos / out_w;
        let ow = pos % out_w;
        let pat_base = out_flat * block;
        let row = dense
            .row_mut(out_flat - row_start)
            .into_slice()
            .expect("dense rows are contiguous");
        for &(block_offset, in_flat) in plan.taps_for(oh, ow) {
            // A 6D output position maps each kernel tap to a distinct input
            // cell. Assignment is therefore the exact intended scatter and,
            // unlike `0.0 += source`, cannot flush a subnormal source merely by
            // passing it through binary32 arithmetic.
            row[in_flat] = patches_flat[pat_base + block_offset];
            deadline.work(1, "during patches dense scatter")?;
        }
    }
    deadline.checkpoint("after patches dense scatter")?;
    Ok(())
}

/// Accumulate, per dense cell `(out_flat, in_flat)`, the number of patch taps that
/// land there (`count`) and the sum of their absolute scattered magnitudes
/// (`absacc`), using the SAME unfold geometry as
/// [`scatter_with_unfold_map_with_deadline`]. Used by `to_dense` to build the
/// overlap-aware certified coefficient error:
/// `err[i,j] = next_up(count[i,j]·err_row[i] + γ_count^f32·absacc[i,j])`.
///
/// 6D-only (f32 accumulators are sufficient because the 6D per-cell tap count is
/// 0/1 — each `in_flat` occurs at most once per output position); the 7D
/// explicit-rows layout uses [`scatter_rows_err_accumulators`] instead.
///
/// Row range (#patches-row-range): same `[row_start, row_end)` output-position
/// window as [`scatter_with_unfold_map_with_deadline`], writing accumulator row
/// `out_flat - row_start` — per-row independent, bit-identical to the slice of
/// the full accumulation.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // Retained no-deadline sibling API; materialization uses the polling form.
pub(super) fn scatter_err_accumulators(
    count: &mut Array2<f32>,
    absacc: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    row_start: usize,
    row_end: usize,
) -> Result<()> {
    let mut deadline = PatchesMaterializationDeadline::new(None);
    let mut admission = PatchesMemoryAdmission::check(
        flat_scratch_bytes(patches),
        "patches coefficient-error scatter scratch",
    )?;
    scatter_err_accumulators_with_deadline(
        count,
        absacc,
        patches,
        index_map,
        out_c,
        out_h,
        out_w,
        in_c,
        kh,
        kw,
        row_start,
        row_end,
        &mut admission,
        &mut deadline,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_err_accumulators_with_deadline(
    count: &mut Array2<f32>,
    absacc: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    row_start: usize,
    row_end: usize,
    admission: &mut PatchesMemoryAdmission,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<()> {
    let mut pat_scratch = Vec::new();
    let plan = build_unfold_plan(index_map);
    debug_assert_eq!(plan.output_size(), (out_h, out_w));
    debug_assert_eq!(checked_shape_product(&[in_c, kh, kw]), Some(plan.block()));
    let (_out_h, out_w) = plan.output_size();
    let block = plan.block();
    let patches_flat = try_as_flat_with_deadline(
        patches,
        &mut pat_scratch,
        "patches coefficient-error scatter scratch",
        admission,
        deadline,
    )?;

    let positions = plan.positions();
    for out_flat in row_start..row_end.min(out_c * positions) {
        deadline.work(1, "during patches coefficient-error row walk")?;
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
            // The 6D mapping is injective within one output row, so assignment
            // retains the exact count/magnitude bit pattern without a
            // DAZ-sensitive f32 add or `.abs()`.
            crow[in_flat] = 1.0;
            let value = patches_flat[pat_base + block_offset];
            arow[in_flat] = f32::from_bits(value.to_bits() & 0x7fff_ffff);
            deadline.work(1, "during patches coefficient-error scatter")?;
        }
    }
    deadline.checkpoint("after patches coefficient-error scatter")?;
    Ok(())
}

/// Scatter row-aware dense patches into a dense matrix using an unfold index map.
///
/// Row range (#patches-row-range): materializes spec rows `[row_start,
/// row_end)` (axis 0 of the 7D patches tensor), writing dense row
/// `row - row_start`. Per-row independent, bit-identical to the slice of the
/// full `(0, row_count)` scatter.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn scatter_rows_with_unfold_map(
    dense: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    row_start: usize,
    row_end: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) -> Result<()> {
    let mut deadline = PatchesMaterializationDeadline::new(None);
    let mut admission = PatchesMemoryAdmission::check(
        flat_scratch_bytes(patches),
        "patches explicit-row dense scatter scratch",
    )?;
    scatter_rows_with_unfold_map_with_deadline(
        dense,
        patches,
        index_map,
        row_start,
        row_end,
        out_c,
        out_h,
        out_w,
        in_c,
        kh,
        kw,
        &mut admission,
        &mut deadline,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_rows_with_unfold_map_with_deadline(
    dense: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    row_start: usize,
    row_end: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    admission: &mut PatchesMemoryAdmission,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<()> {
    let mut pat_scratch = Vec::new();
    let plan = build_unfold_plan(index_map);
    debug_assert_eq!(plan.output_size(), (out_h, out_w));
    debug_assert_eq!(checked_shape_product(&[in_c, kh, kw]), Some(plan.block()));
    let (out_h, out_w) = plan.output_size();
    let block = plan.block();
    let positions = plan.positions();
    let per_row = out_c * positions * block;
    let patches_flat = try_as_flat_with_deadline(
        patches,
        &mut pat_scratch,
        "patches explicit-row dense scatter scratch",
        admission,
        deadline,
    )?;

    for row in row_start..row_end {
        deadline.work(1, "during patches explicit-row dense row walk")?;
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
                        deadline.work(1, "during patches explicit-row dense scatter")?;
                    }
                }
            }
        }
    }
    deadline.checkpoint("after patches explicit-row dense scatter")?;
    Ok(())
}

/// Scatter only selected input-flat columns from row-aware 7D patches.
///
/// `selected_columns` must be strictly ascending and duplicate-free. Output
/// column `compact_col` corresponds to input-flat column
/// `selected_columns[compact_col]`.
///
/// The canonical [`UnfoldPlan`] is filtered once, before the expensive
/// `row -> oc -> oh -> ow` walk. Each retained tap keeps its original
/// `(ic, ki, kj)` position within that walk, so every selected dense cell sees
/// the exact same `+=` sequence as [`scatter_rows_with_unfold_map_with_deadline`]. Filtering
/// avoids visiting the overwhelmingly many taps that cannot contribute to a
/// constrained beta neuron.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_rows_selected_columns_with_unfold_map(
    selected: &mut Array2<f32>,
    selected_columns: &[usize],
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    row_count: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) {
    debug_assert_eq!(selected.nrows(), row_count);
    debug_assert_eq!(selected.ncols(), selected_columns.len());
    debug_assert!(selected_columns.windows(2).all(|pair| pair[0] < pair[1]));

    let mut pat_scratch = Vec::new();
    let full_plan = build_unfold_plan(index_map);
    debug_assert_eq!(full_plan.output_size(), (out_h, out_w));
    debug_assert_eq!(
        checked_shape_product(&[in_c, kh, kw]),
        Some(full_plan.block())
    );
    let (out_h, out_w) = full_plan.output_size();
    let block = full_plan.block();
    let positions = full_plan.positions();
    let per_row = out_c * positions * block;
    let patches_flat = as_flat(patches, &mut pat_scratch);

    // Preserve the canonical tap order while replacing the global input-flat
    // destination with its compact selected-column destination.
    let mut selected_taps = Vec::new();
    let mut selected_offsets = Vec::with_capacity(positions + 1);
    selected_offsets.push(0);
    for oh in 0..out_h {
        for ow in 0..out_w {
            for &(block_offset, in_flat) in full_plan.taps_for(oh, ow) {
                if let Ok(compact_col) = selected_columns.binary_search(&in_flat) {
                    selected_taps.push((block_offset, compact_col));
                }
            }
            selected_offsets.push(selected_taps.len());
        }
    }

    for row in 0..row_count {
        let row_base = row * per_row;
        let out_row = selected
            .row_mut(row)
            .into_slice()
            .expect("selected rows are contiguous");
        for oc in 0..out_c {
            let oc_base = row_base + oc * positions * block;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let position = oh * out_w + ow;
                    let pat_base = oc_base + position * block;
                    for &(block_offset, compact_col) in
                        &selected_taps[selected_offsets[position]..selected_offsets[position + 1]]
                    {
                        out_row[compact_col] += patches_flat[pat_base + block_offset];
                    }
                }
            }
        }
    }
}

/// Accumulate, per dense cell `(spec_row, in_flat)`, the number of plan taps
/// that land there (`count`) and the **f64** sum of their absolute scattered
/// magnitudes (`absacc`), mirroring [`scatter_rows_with_unfold_map_with_deadline`]'s tap
/// geometry EXACTLY (same `UnfoldPlan`, same
/// `row -> oc -> oh -> ow -> (ic, ki, kj)` loop order). Used by `to_dense` to
/// build the overlap-aware certified coefficient error for the 7D
/// explicit-rows layout (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §3):
/// `err[r,j] = next_up(count[r,j]·err_row[r] + γ_count^f32·absacc[r,j])`.
///
/// Tap magnitudes are decoded from their binary32 bits rather than via f32
/// `.abs()`/conversion, so a subnormal source remains visible when the host has
/// DAZ enabled. The accumulators are f64, deviating from the f32 6D
/// [`scatter_err_accumulators`] (spec R1): the γ-chaining argument
/// `γ(N)·S_hat >= γ(N−1)·S_true` needs `S_hat` within `2^-28` relative of
/// `S_true`, which f64 same-sign summation guarantees for every `N < 2^24`;
/// the f32 mirror only closes for `N <~ 4096`, while 7D per-cell tap counts on
/// the scored workload reach `~1e4..5e5`. f64 `count += 1.0` is exact to
/// `2^53`, also eliminating f32 count saturation at `2^24`.
///
/// Row range (#patches-row-range): same `[row_start, row_end)` spec-row window
/// as [`scatter_rows_with_unfold_map_with_deadline`], writing accumulator row
/// `row - row_start` — per-row independent, bit-identical to the slice of the
/// full accumulation.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // Retained no-deadline sibling API; materialization uses the polling form.
pub(super) fn scatter_rows_err_accumulators(
    count: &mut Array2<f64>,
    absacc: &mut Array2<f64>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    row_start: usize,
    row_end: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) -> Result<()> {
    let mut deadline = PatchesMaterializationDeadline::new(None);
    let mut admission = PatchesMemoryAdmission::check(
        flat_scratch_bytes(patches),
        "patches explicit-row error scatter scratch",
    )?;
    scatter_rows_err_accumulators_with_deadline(
        count,
        absacc,
        patches,
        index_map,
        row_start,
        row_end,
        out_c,
        out_h,
        out_w,
        in_c,
        kh,
        kw,
        &mut admission,
        &mut deadline,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_rows_err_accumulators_with_deadline(
    count: &mut Array2<f64>,
    absacc: &mut Array2<f64>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    row_start: usize,
    row_end: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    admission: &mut PatchesMemoryAdmission,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<()> {
    let mut pat_scratch = Vec::new();
    let plan = build_unfold_plan(index_map);
    debug_assert_eq!(plan.output_size(), (out_h, out_w));
    debug_assert_eq!(checked_shape_product(&[in_c, kh, kw]), Some(plan.block()));
    let (out_h, out_w) = plan.output_size();
    let block = plan.block();
    let positions = plan.positions();
    let per_row = out_c * positions * block;
    let patches_flat = try_as_flat_with_deadline(
        patches,
        &mut pat_scratch,
        "patches explicit-row error scatter scratch",
        admission,
        deadline,
    )?;

    for row in row_start..row_end {
        deadline.work(1, "during patches explicit-row error row walk")?;
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
                        let value = patches_flat[pat_base + block_offset];
                        arow[in_flat] += ny_core::f32_to_f64_exact(f32::from_bits(
                            value.to_bits() & 0x7fff_ffff,
                        ));
                        deadline.work(1, "during patches explicit-row error scatter")?;
                    }
                }
            }
        }
    }
    deadline.checkpoint("after patches explicit-row error scatter")?;
    Ok(())
}

/// Scatter sparse patches into a dense matrix using an unfold index map.
///
/// Row range (#patches-row-range): sparse rows whose flat output position
/// falls outside `[row_start, row_end)` are skipped; in-range rows write dense
/// row `out_flat - row_start`. Sparse-index visiting order is unchanged, so
/// any range split is bit-identical to the corresponding slice of the full
/// scatter.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn scatter_sparse_with_unfold_map(
    dense: &mut Array2<f32>,
    sparse_patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    idx: &UnstableIdx,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    row_start: usize,
    row_end: usize,
) -> Result<()> {
    let mut deadline = PatchesMaterializationDeadline::new(None);
    let mut admission = PatchesMemoryAdmission::check(
        flat_scratch_bytes(sparse_patches),
        "sparse patches dense scatter scratch",
    )?;
    scatter_sparse_with_unfold_map_with_deadline(
        dense,
        sparse_patches,
        index_map,
        idx,
        out_h,
        out_w,
        in_c,
        kh,
        kw,
        row_start,
        row_end,
        &mut admission,
        &mut deadline,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_sparse_with_unfold_map_with_deadline(
    dense: &mut Array2<f32>,
    sparse_patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    idx: &UnstableIdx,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    row_start: usize,
    row_end: usize,
    admission: &mut PatchesMemoryAdmission,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<()> {
    let mut pat_scratch = Vec::new();
    let plan = build_unfold_plan(index_map);
    debug_assert_eq!(plan.output_size(), (out_h, out_w));
    debug_assert_eq!(checked_shape_product(&[in_c, kh, kw]), Some(plan.block()));
    let (out_h, out_w) = plan.output_size();
    let block = plan.block();
    let patches_flat = try_as_flat_with_deadline(
        sparse_patches,
        &mut pat_scratch,
        "sparse patches dense scatter scratch",
        admission,
        deadline,
    )?;

    for (i, ((&c, &h), &w)) in idx
        .channels
        .iter()
        .zip(idx.heights.iter())
        .zip(idx.widths.iter())
        .enumerate()
    {
        deadline.work(1, "during sparse patches index walk")?;
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
            // One sparse output row has the same injective tap mapping as 6D
            // dense patches; avoid an unnecessary DAZ/FTZ-sensitive 0 += x.
            row[in_flat] = patches_flat[pat_base + block_offset];
            deadline.work(1, "during sparse patches dense scatter")?;
        }
    }
    deadline.checkpoint("after sparse patches dense scatter")?;
    Ok(())
}

/// Scatter row-aware sparse patches into a dense matrix using an unfold index map.
///
/// Row range (#patches-row-range): materializes spec rows `[row_start,
/// row_end)` (axis 0 of the 5D sparse patches tensor), writing dense row
/// `row - row_start` — per-row independent, bit-identical to the slice of the
/// full `(0, row_count)` scatter.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn scatter_sparse_rows_with_unfold_map(
    dense: &mut Array2<f32>,
    sparse_patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    row_start: usize,
    row_end: usize,
    idx: &UnstableIdx,
    in_c: usize,
    kh: usize,
    kw: usize,
) -> Result<()> {
    let mut deadline = PatchesMaterializationDeadline::new(None);
    let mut admission = PatchesMemoryAdmission::check(
        flat_scratch_bytes(sparse_patches),
        "sparse explicit-row dense scatter scratch",
    )?;
    scatter_sparse_rows_with_unfold_map_with_deadline(
        dense,
        sparse_patches,
        index_map,
        row_start,
        row_end,
        idx,
        in_c,
        kh,
        kw,
        &mut admission,
        &mut deadline,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_sparse_rows_with_unfold_map_with_deadline(
    dense: &mut Array2<f32>,
    sparse_patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    row_start: usize,
    row_end: usize,
    idx: &UnstableIdx,
    in_c: usize,
    kh: usize,
    kw: usize,
    admission: &mut PatchesMemoryAdmission,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<()> {
    let unstable_size = idx.len();
    let index_shape = index_map.shape();
    let (out_h, out_w) = (index_shape[0], index_shape[1]);
    let mut pat_scratch = Vec::new();
    let plan = build_unfold_plan(index_map);
    debug_assert_eq!(plan.output_size(), (out_h, out_w));
    debug_assert_eq!(checked_shape_product(&[in_c, kh, kw]), Some(plan.block()));
    let block = plan.block();
    let patches_flat = try_as_flat_with_deadline(
        sparse_patches,
        &mut pat_scratch,
        "sparse explicit-row dense scatter scratch",
        admission,
        deadline,
    )?;

    for row in row_start..row_end {
        deadline.work(1, "during sparse explicit-row row walk")?;
        let out_row = dense
            .row_mut(row - row_start)
            .into_slice()
            .expect("dense rows are contiguous");
        for (i, (&h, &w)) in idx.heights.iter().zip(idx.widths.iter()).enumerate() {
            deadline.work(1, "during sparse explicit-row index walk")?;
            let pat_base = (row * unstable_size + i) * block;
            for &(block_offset, in_flat) in plan.taps_for(h, w) {
                out_row[in_flat] += patches_flat[pat_base + block_offset];
                deadline.work(1, "during sparse explicit-row dense scatter")?;
            }
        }
    }
    deadline.checkpoint("after sparse explicit-row dense scatter")?;
    Ok(())
}

#[cfg(test)]
mod exact_map_tests {
    use super::{build_unfold_plan, compute_unfold_index_map, unfold_plan_storage_bytes};
    use crate::bounds::patches::{PatchGeometry, PatchesData, PatchesMemoryAdmission};
    use ny_core::NyError;
    use std::mem::size_of;

    fn geometry_data(
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    ) -> PatchesData {
        PatchesData {
            patches: None,
            geometry: PatchGeometry::affine(stride, padding),
            identity: true,
            output_shape,
            input_shape,
            unstable_idx: None,
            coeff_err: None,
        }
    }

    #[test]
    fn exact_unfold_plan_preserves_affine_stride_and_padding() {
        // in=2x3, k=2x2, stride=1, left/top pad=1 -> out=2x3.
        let data = geometry_data((1, 2, 3), (1, 2, 3), (1, 1), (1, 0, 1, 0));
        let map = compute_unfold_index_map(&data, 2, 2).unwrap();
        assert_eq!(map.shape(), &[2, 3, 1, 2, 2]);
        assert_eq!(map.len(), 24);

        let plan = build_unfold_plan(&map);
        assert_eq!(plan.taps_for(0, 0), &[(3, 0)]);
        assert_eq!(plan.taps_for(0, 1), &[(2, 0), (3, 1)]);
        assert_eq!(plan.taps_for(1, 2), &[(0, 1), (1, 2), (2, 4), (3, 5)]);
    }

    #[test]
    fn exact_unfold_plan_rejects_metadata_extent_mismatch() {
        let data = geometry_data((1, 3, 3), (1, 2, 3), (1, 1), (1, 0, 1, 0));
        assert!(compute_unfold_index_map(&data, 2, 2).is_err());
    }

    #[test]
    fn exact_unfold_plan_supports_separable_anchored_origins() {
        let data = PatchesData {
            patches: None,
            geometry: PatchGeometry::anchored(vec![-1, 2], vec![0, 2, 4]).unwrap(),
            identity: true,
            output_shape: (1, 2, 3),
            input_shape: (1, 4, 5),
            unstable_idx: None,
            coeff_err: None,
        };
        let map = compute_unfold_index_map(&data, 2, 2).unwrap();
        assert_eq!(map.shape(), &[2, 3, 1, 2, 2]);
        let plan = build_unfold_plan(&map);
        assert_eq!(plan.taps_for(0, 0), &[(2, 0), (3, 1)]);
        assert_eq!(plan.taps_for(0, 2), &[(2, 4)]);
        assert_eq!(plan.taps_for(1, 1), &[(0, 12), (1, 13), (2, 17), (3, 18)]);
    }

    #[test]
    fn unfold_plan_budget_refuses_before_reserving_or_touching_memory() {
        crate::tests::with_crown_dense_budget_mb("1", || {
            let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
            assert_eq!(budget, 1024 * 1024, "fixture requires 1 MiB of headroom");
            let bytes_per_position =
                size_of::<(usize, usize)>() + size_of::<usize>() + size_of::<f32>();
            let width = budget
                .checked_div(bytes_per_position)
                .and_then(|value| value.checked_add(1))
                .expect("configured CROWN budget should leave a representable refusal fixture");
            let data = geometry_data((1, 1, width), (1, 1, width), (1, 1), (0, 0, 0, 0));
            assert!(matches!(
                compute_unfold_index_map(&data, 1, 1),
                Err(NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes,
                    site: "patch unfold plan",
                }) if required_bytes > budget_bytes && budget_bytes == budget
            ));
        });
    }

    #[test]
    fn unfold_plan_receipt_accepts_exact_total_and_refuses_budget_minus_one() {
        let resident = 19usize;
        let required = resident.saturating_add(unfold_plan_storage_bytes(3, 4, 5));
        assert!(PatchesMemoryAdmission::check_with_budget(
            required,
            required,
            "unfold receipt exact test",
        )
        .is_ok());
        assert!(matches!(
            PatchesMemoryAdmission::check_with_budget(
                required,
                required - 1,
                "unfold receipt budget-minus-one test",
            ),
            Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "unfold receipt budget-minus-one test",
            }) if required_bytes == required && budget_bytes == required - 1
        ));
    }
}
