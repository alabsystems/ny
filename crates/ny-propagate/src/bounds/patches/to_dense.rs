// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use std::mem::size_of;
use std::time::Instant;

use crate::bounds::LinearBounds;
use crate::execution_telemetry::{
    record_patches_materialization_attempt, record_patches_materialization_refusal,
    record_patches_materialization_success, PatchesCoefficientErrorDisposition,
    PatchesMaterializationGeometry, PatchesMaterializationMemoryReceipt,
    PatchesMaterializationRefusal,
};

use super::scatter::{
    build_unfold_plan, compute_unfold_index_map_with_deadline,
    scatter_err_accumulators_with_deadline, scatter_rows_err_accumulators_with_deadline,
    scatter_rows_with_unfold_map_with_deadline, scatter_sparse_rows_with_unfold_map_with_deadline,
    scatter_sparse_with_unfold_map_with_deadline, scatter_with_unfold_map_with_deadline,
    validate_patches_shape, UnfoldIndexMap,
};
use super::{
    PatchGeometry, PatchesLinearBounds, PatchesMaterializationDeadline,
    PatchesMaterializationPurpose, PatchesMemoryAdmission, UnstableIdx,
};

#[inline]
fn allocation_bytes<T>(elements: usize) -> usize {
    elements.saturating_mul(size_of::<T>())
}

#[inline]
fn matrix_elements(rows: usize, columns: usize) -> usize {
    rows.saturating_mul(columns)
}

#[inline]
fn matrix_bytes<T>(rows: usize, columns: usize) -> usize {
    allocation_bytes::<T>(matrix_elements(rows, columns))
}

#[inline]
fn scratch_bytes(array: &ArrayD<f32>) -> usize {
    if array.as_slice().is_some() {
        0
    } else {
        allocation_bytes::<f32>(array.len())
    }
}

fn has_scattered_subnormal_6d_with_poll(
    array: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    row_start: usize,
    row_end: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<bool> {
    let plan = build_unfold_plan(index_map);
    let (_out_h, out_w) = plan.output_size();
    let positions = plan.positions();
    for out_flat in row_start..row_end.min(out_c.saturating_mul(positions)) {
        let oc = out_flat / positions;
        let position = out_flat % positions;
        let oh = position / out_w;
        let ow = position % out_w;
        for &(block_offset, _) in plan.taps_for(oh, ow) {
            let ic = block_offset / (kh * kw);
            let tap = block_offset % (kh * kw);
            let ki = tap / kw;
            let kj = tap % kw;
            let value = array[[oc, oh, ow, ic, ki, kj]];
            let magnitude = value.to_bits() & 0x7fff_ffff;
            if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
                deadline.checkpoint("after patches subnormal scan")?;
                return Ok(true);
            }
            deadline.work(1, "during patches subnormal scan")?;
        }
    }
    deadline.checkpoint("after patches subnormal scan")?;
    Ok(false)
}

fn try_filled_array2<T: Clone>(
    rows: usize,
    columns: usize,
    fill: T,
    admission: &mut PatchesMemoryAdmission,
    site: &'static str,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<Array2<T>> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| admission.allocation_error(site))?;
    let mut data = Vec::new();
    deadline.checkpoint(site)?;
    data.try_reserve_exact(elements)
        .map_err(|_| admission.allocation_error(site))?;
    deadline.checkpoint(site)?;
    admission.reconcile_vec_capacity::<T>(elements, data.capacity(), site)?;
    let mut filled = 0usize;
    while filled < elements {
        let end = filled
            .saturating_add(PatchesMaterializationDeadline::CHECK_STRIDE)
            .min(elements);
        data.resize(end, fill.clone());
        deadline.work(end - filled, site)?;
        filled = end;
    }
    deadline.checkpoint(site)?;
    let array = Array2::from_shape_vec((rows, columns), data).map_err(|error| {
        NyError::InternalError(format!(
            "{site}: checked dense shape construction failed: {error}"
        ))
    })?;
    deadline.checkpoint(site)?;
    Ok(array)
}

fn try_zeroed_array1(
    len: usize,
    admission: &mut PatchesMemoryAdmission,
    site: &'static str,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<Array1<f32>> {
    let mut data = Vec::new();
    deadline.checkpoint(site)?;
    data.try_reserve_exact(len)
        .map_err(|_| admission.allocation_error(site))?;
    deadline.checkpoint(site)?;
    admission.reconcile_vec_capacity::<f32>(len, data.capacity(), site)?;
    let mut filled = 0usize;
    while filled < len {
        let end = filled
            .saturating_add(PatchesMaterializationDeadline::CHECK_STRIDE)
            .min(len);
        data.resize(end, 0.0);
        deadline.work(end - filled, site)?;
        filled = end;
    }
    deadline.checkpoint(site)?;
    let array = Array1::from_vec(data);
    deadline.checkpoint(site)?;
    Ok(array)
}

/// Fallibly allocate and cooperatively fill one f32 vector while reporting the
/// enclosing operation's already-computed live peak on any refusal.
pub(super) fn try_filled_f32_vec(
    len: usize,
    fill: f32,
    admission: &mut PatchesMemoryAdmission,
    site: &'static str,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<Vec<f32>> {
    let mut data = Vec::new();
    deadline.checkpoint(site)?;
    data.try_reserve_exact(len)
        .map_err(|_| admission.allocation_error(site))?;
    deadline.checkpoint(site)?;
    admission.reconcile_vec_capacity::<f32>(len, data.capacity(), site)?;
    let mut filled = 0usize;
    while filled < len {
        let end = filled
            .saturating_add(PatchesMaterializationDeadline::CHECK_STRIDE)
            .min(len);
        data.resize(end, fill);
        deadline.work(end - filled, site)?;
        filled = end;
    }
    deadline.checkpoint(site)?;
    Ok(data)
}

fn try_copy_bias_range(
    bias: &Array1<f32>,
    row_start: usize,
    row_end: usize,
    admission: &mut PatchesMemoryAdmission,
    site: &'static str,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<Array1<f32>> {
    let len = row_end - row_start;
    let mut data = Vec::new();
    deadline.checkpoint(site)?;
    data.try_reserve_exact(len)
        .map_err(|_| admission.allocation_error(site))?;
    deadline.checkpoint(site)?;
    admission.reconcile_vec_capacity::<f32>(len, data.capacity(), site)?;
    for row in row_start..row_end {
        data.push(bias[row]);
        deadline.work(1, site)?;
    }
    deadline.checkpoint(site)?;
    let array = Array1::from_vec(data);
    deadline.checkpoint(site)?;
    Ok(array)
}

/// Apply `new_or_conservative[_with_err]`'s numeric firewall to already-owned
/// buffers in place. This preserves its whole-relation conservative degrade
/// without allocating a second dense pair or `mapv` copies of error matrices.
#[allow(clippy::too_many_arguments)]
fn finish_dense_in_place(
    mut lower_a: Array2<f32>,
    mut lower_b: Array1<f32>,
    mut upper_a: Array2<f32>,
    mut upper_b: Array1<f32>,
    mut lower_err: Option<Array2<f32>>,
    mut upper_err: Option<Array2<f32>>,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<LinearBounds> {
    let coefficient_shape = lower_a.raw_dim();
    let err_shapes_match = lower_err
        .as_ref()
        .is_none_or(|error| error.raw_dim() == coefficient_shape)
        && upper_err
            .as_ref()
            .is_none_or(|error| error.raw_dim() == coefficient_shape);
    let mut numeric_bad = false;
    for value in &lower_a {
        deadline.work(1, "during lower coefficient firewall scan")?;
        if !value.is_finite() {
            numeric_bad = true;
            break;
        }
    }
    if !numeric_bad {
        for value in &upper_a {
            deadline.work(1, "during upper coefficient firewall scan")?;
            if !value.is_finite() {
                numeric_bad = true;
                break;
            }
        }
    }
    if !numeric_bad {
        for value in &lower_b {
            deadline.work(1, "during lower bias firewall scan")?;
            if value.is_nan() {
                numeric_bad = true;
                break;
            }
        }
    }
    if !numeric_bad {
        for value in &upper_b {
            deadline.work(1, "during upper bias firewall scan")?;
            if value.is_nan() {
                numeric_bad = true;
                break;
            }
        }
    }

    if numeric_bad || !err_shapes_match {
        tracing::warn!(
            "LinearBounds NaN/error-shape firewall: Patches-to-Dense produced an invalid relation; falling back to conservative bounds"
        );
        for value in &mut lower_a {
            *value = 0.0;
            deadline.work(1, "during lower coefficient firewall repair")?;
        }
        for value in &mut upper_a {
            *value = 0.0;
            deadline.work(1, "during upper coefficient firewall repair")?;
        }
        for value in &mut lower_b {
            *value = f32::NEG_INFINITY;
            deadline.work(1, "during lower bias firewall repair")?;
        }
        for value in &mut upper_b {
            *value = f32::INFINITY;
            deadline.work(1, "during upper bias firewall repair")?;
        }
        if !err_shapes_match {
            lower_err = None;
            upper_err = None;
        }
    }

    for error in lower_err.iter_mut().chain(upper_err.iter_mut()) {
        for value in error.iter_mut() {
            if !value.is_finite() || *value < 0.0 {
                *value = f32::INFINITY;
            }
            deadline.work(1, "during coefficient-error firewall scan")?;
        }
    }

    deadline.checkpoint("before dense wrapping")?;
    let bounds = LinearBounds::from_prevalidated_parts_with_optional_err(
        lower_a, lower_b, upper_a, upper_b, lower_err, upper_err,
    )?;
    deadline.checkpoint("after dense wrapping")?;
    Ok(bounds)
}

struct FullDenseMaterialization {
    bounds: LinearBounds,
    receipt: PatchesMaterializationMemoryReceipt,
    coefficient_error: PatchesCoefficientErrorDisposition,
}

impl FullDenseMaterialization {
    fn new(bounds: LinearBounds, admission: &PatchesMemoryAdmission) -> Self {
        let coefficient_error = if bounds.has_coeff_err() {
            PatchesCoefficientErrorDisposition::Materialized
        } else {
            PatchesCoefficientErrorDisposition::Absent
        };
        Self {
            bounds,
            receipt: admission.receipt(),
            coefficient_error,
        }
    }
}

#[derive(Clone, Copy)]
enum SideErrorAllocation {
    /// A global error pair is required, but this exact 6D side only needs its
    /// final zero error matrix.
    Zero,
    /// 6D carried error: two f32 accumulators and one f32 result.
    Dense6,
    /// 7D explicit rows: two f64 accumulators and one f32 result.
    Rows7,
}

/// Add one side's error-materialization phase to the peak and return the bytes
/// retained after that side finishes (its final f32 error matrix).
fn account_error_side(
    base_bytes: usize,
    persistent_err_bytes: usize,
    matrix_bytes: usize,
    scratch_bytes: usize,
    kind: SideErrorAllocation,
    peak_bytes: &mut usize,
) -> usize {
    let current = base_bytes.saturating_add(persistent_err_bytes);
    let phase_peak = match kind {
        SideErrorAllocation::Zero => current.saturating_add(matrix_bytes),
        SideErrorAllocation::Dense6 => current
            .saturating_add(matrix_bytes.saturating_mul(2))
            .saturating_add(scratch_bytes.max(matrix_bytes)),
        SideErrorAllocation::Rows7 => current
            .saturating_add(matrix_bytes.saturating_mul(4))
            .saturating_add(scratch_bytes.max(matrix_bytes)),
    };
    *peak_bytes = (*peak_bytes).max(phase_peak);
    persistent_err_bytes.saturating_add(matrix_bytes)
}

/// Pure total-live-peak accounting for a dense (6D/7D) materialization.
/// `resident_bytes` includes the borrowed source Patches carrier and any
/// operation-owned buffers retained by an enclosing request. `None` on both
/// error sides is the exact-6D fast path; when a global error pair is required,
/// both sides are `Some` and each completed error matrix remains live while the
/// next side is built.
#[allow(clippy::too_many_arguments)]
fn dense_materialization_peak_bytes(
    resident_bytes: usize,
    map_bytes: usize,
    matrix_bytes: usize,
    bias_pair_bytes: usize,
    lower_scratch_bytes: usize,
    upper_scratch_bytes: usize,
    lower_error: Option<SideErrorAllocation>,
    upper_error: Option<SideErrorAllocation>,
) -> usize {
    let dense_bias_base = map_bytes
        .saturating_add(matrix_bytes.saturating_mul(2))
        .saturating_add(bias_pair_bytes);
    let mut peak_bytes = map_bytes
        .saturating_add(matrix_bytes)
        .saturating_add(lower_scratch_bytes);
    peak_bytes = peak_bytes.max(
        map_bytes
            .saturating_add(matrix_bytes.saturating_mul(2))
            .saturating_add(upper_scratch_bytes),
    );
    peak_bytes = peak_bytes.max(dense_bias_base);

    if let (Some(lower_kind), Some(upper_kind)) = (lower_error, upper_error) {
        let retained = account_error_side(
            dense_bias_base,
            0,
            matrix_bytes,
            lower_scratch_bytes,
            lower_kind,
            &mut peak_bytes,
        );
        account_error_side(
            dense_bias_base,
            retained,
            matrix_bytes,
            upper_scratch_bytes,
            upper_kind,
            &mut peak_bytes,
        );
    }
    resident_bytes.saturating_add(peak_bytes)
}

/// Logical allocations retained while one materialized `LinearBounds` block
/// is inside `LinearBounds::concretize_sound`.
///
/// `BoundedTensor::flatten` owns lower/upper f32 copies, the scalar kernel owns
/// lower/upper f64 endpoints, and publication owns lower/upper f32 result
/// arrays. The four shape words conservatively cover the largest simultaneous
/// pair of ndarray shape `Vec`s created by validation/publication. These phases
/// do not all overlap, but charging them additively is an intentionally
/// conservative preflight: when added as a resident base to the dense-block
/// admission, the receipt covers the source Patches, full output pair, block A
/// and error matrices, block biases, both flattened inputs, both f64 endpoints,
/// and both returned f32 endpoints in one fail-closed total.
///
/// Like the pre-existing carrier charge, nested ndarray buffers expose logical
/// length rather than backing capacity. The process-envelope headroom described
/// on [`PatchesMemoryAdmission`] covers allocator slack outside the Vecs this
/// module reserves and reconciles directly.
fn chunked_concretization_nested_bytes(rows: usize, in_dim: usize) -> usize {
    let flattened_input_pair = allocation_bytes::<f32>(in_dim).saturating_mul(2);
    let f64_endpoint_pair = allocation_bytes::<f64>(rows).saturating_mul(2);
    let f32_result_pair = allocation_bytes::<f32>(rows).saturating_mul(2);
    let shape_words = allocation_bytes::<usize>(4);
    flattened_input_pair
        .saturating_add(f64_endpoint_pair)
        .saturating_add(f32_result_pair)
        .saturating_add(shape_words)
}

/// Conservative row-sizing slope for chunked materialize-then-concretize.
/// The dense 7D peak is eight f32-equivalent cells per input column; add two
/// block bias values and the nested f64/f32 result pairs. The flattened input
/// pair and shape words are block-fixed and are deducted separately.
fn chunked_concretization_per_row_bytes(in_dim: usize) -> usize {
    let dense_block = in_dim.saturating_mul(8).saturating_mul(size_of::<f32>());
    let block_bias_pair = allocation_bytes::<f32>(2);
    let nested_result_pairs = allocation_bytes::<f64>(2).saturating_add(allocation_bytes::<f32>(2));
    dense_block
        .saturating_add(block_bias_pair)
        .saturating_add(nested_result_pairs)
        .max(1)
}

fn chunked_concretization_fixed_bytes(in_dim: usize) -> usize {
    allocation_bytes::<f32>(in_dim)
        .saturating_mul(2)
        .saturating_add(allocation_bytes::<usize>(4))
}

/// Decode a carried non-negative binary32 error without presenting a
/// subnormal operand to DAZ-sensitive conversion/comparison instructions.
#[inline]
pub(super) fn nonnegative_f32_error_or_infinity(value: f32) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let exponent = magnitude >> 23;
    if exponent == 0xff || (bits >> 31 != 0 && magnitude != 0) {
        f64::INFINITY
    } else if magnitude == 0 {
        0.0
    } else {
        ny_core::f32_to_f64_exact(value)
    }
}

/// Absolute magnitude of a binary32 bit pattern, decoded exactly in binary64.
/// Masking the sign before the bit-wise decoder avoids both f32 `.abs()` and
/// ordinary `f32 -> f64`, either of which may observe DAZ state.
#[inline]
pub(super) fn f32_abs_exact(value: f32) -> f64 {
    ny_core::f32_to_f64_exact(f32::from_bits(value.to_bits() & 0x7fff_ffff))
}

/// Normalize a published coefficient center so a later FTZ/DAZ consumer sees
/// the same zero/normal value. The exact magnitude removed from a subnormal is
/// returned for addition to its certificate; signed zero is preserved.
#[inline]
fn normalize_subnormal_center(center: &mut f32) -> f64 {
    let bits = center.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
        let charge = f32_abs_exact(*center);
        *center = f32::from_bits(bits & 0x8000_0000);
        charge
    } else {
        0.0
    }
}

/// Absolute certificate for DAZ of source taps plus FTZ of binary32 partial
/// and final sums in one explicit-row scatter cell. The conventional
/// `4N*FLT_MIN` bound covers all three effects with one additional unit of
/// headroom. A zero absolute accumulator proves every contributing tap was
/// exact zero, so the historical exact-zero certificate remains zero.
#[inline]
fn rows_f32_underflow_charge(count: f64, absacc: f64) -> f64 {
    if absacc == 0.0 {
        return 0.0;
    }
    if !count.is_finite() || count > (1_u64 << f64::MANTISSA_DIGITS) as f64 {
        return f64::INFINITY;
    }
    4.0 * count * ny_core::f32_to_f64_exact(f32::MIN_POSITIVE)
}

/// Preserve the historical one-f32-ULP outward publication for normal terms,
/// while never returning a positive binary32 subnormal that FTZ could erase.
#[inline]
pub(super) fn publish_error_up_normal(value: f64) -> f32 {
    if value.is_nan() || value < 0.0 || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }
    let min_normal = ny_core::f32_to_f64_exact(f32::MIN_POSITIVE);
    if value < min_normal {
        return f32::MIN_POSITIVE;
    }
    // This cast is reached only in the binary32-normal range. Retain the
    // historical strict one-ULP outward step for byte stability there.
    ny_tensor::next_up_f32(value as f32)
}

/// Build the overlap-aware dense certified-error matrix for one **6D** side
/// from an optional carried per-row `coeff_err` and any subnormal center
/// normalization (#patches-coeff-err-soundness). For each dense cell `(i,j)`
/// that receives `count` patch taps of row `i` (whose bit-exact absolute
/// scattered magnitudes sum to `absacc`):
///   `err[i,j] = publish_normal_up(
///       count·err_row[i] + γ_count^f32·absacc + center_flush)`.
/// First term over-bounds the sum of the `count` carried coefficient deviations
/// (each ≤ `err_row[i]`); the historical gamma term remains conservative after
/// the now-exact injective scatter assignment. A subnormal center is replaced
/// by signed zero and its exact bit-decoded magnitude is charged. Uses the SAME
/// unfold geometry as the coefficient scatter, so it is overlap-exact. Returns
/// a `(row_end - row_start) × in_dim` matrix (0 where no tap lands): rows
/// `[row_start, row_end)` of the full grid (#patches-row-range), with `err_row`
/// still indexed by the GLOBAL row.
///
/// 6D variant (f32 accumulators; per-cell tap count is 0/1, so f32 suffices and
/// stays byte-identical to the certified 6D design). The 7D explicit-rows layout
/// goes through [`patches_err_matrix_rows`] instead.
#[allow(clippy::too_many_arguments)]
fn patches_err_matrix(
    center: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    err_row: Option<&Array1<f32>>,
    row_start: usize,
    row_end: usize,
    in_dim: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    admission: &mut PatchesMemoryAdmission,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<Array2<f32>> {
    let n_rows = row_end - row_start;
    let mut count = try_filled_array2(
        n_rows,
        in_dim,
        0.0_f32,
        admission,
        "patches 6D error-count allocation",
        deadline,
    )?;
    let mut absacc = try_filled_array2(
        n_rows,
        in_dim,
        0.0_f32,
        admission,
        "patches 6D error-magnitude allocation",
        deadline,
    )?;
    scatter_err_accumulators_with_deadline(
        &mut count,
        &mut absacc,
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
        admission,
        deadline,
    )?;
    let mut err = try_filled_array2(
        n_rows,
        in_dim,
        0.0_f32,
        admission,
        "patches 6D certified-error allocation",
        deadline,
    )?;
    for (local, i) in (row_start..row_end).enumerate() {
        // Length is checked before this infallible kernel is entered.  Invalid
        // certificates must poison outward: `f64::max` maps NaN to its other
        // operand, which previously reinterpreted NaN as an exact zero error.
        let er = match err_row {
            None => 0.0,
            Some(error) => nonnegative_f32_error_or_infinity(error[i]),
        };
        for j in 0..in_dim {
            let c = count[[local, j]];
            if c > 0.0 {
                let gamma = crate::layers::linear::crown_single_gamma_n_f32(c as usize);
                let oe = f32_abs_exact(absacc[[local, j]]);
                let center_flush = normalize_subnormal_center(&mut center[[local, j]]);
                let term = ny_core::f32_to_f64_exact(c) * er + gamma * oe + center_flush;
                err[[local, j]] = publish_error_up_normal(term);
            }
            deadline.work(1, "during patches 6D certified-error finalization")?;
        }
    }
    deadline.checkpoint("after patches 6D certified-error finalization")?;
    Ok(err)
}

/// Build the overlap-aware dense certified-error matrix for one **7D
/// explicit-rows** side (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §3). The err
/// index is the SPEC row (axis 0, length `row_count`, == the bias length). For
/// each dense cell `(r,j)` that receives `count` plan taps of spec row `r`
/// (bit-exact absolute scattered magnitudes f64-summed into `absacc`):
///   `err[r,j] = publish_normal_up(count·err_row[r] +
///       γ_count^f32·absacc + 4·count·FLT_MIN + center_flush)`.
///
/// Emitted even for `err_row == None` (`e_r = 0`): unlike the 6D layout the 7D
/// scatter genuinely accumulates multiple taps per dense cell, so the
/// `γ(N)·S_hat` accumulation-rounding term exists on every side (spec R2).
/// `4N·FLT_MIN` certifies source DAZ and partial/result FTZ in the raw f32
/// scatter. A subnormal stored center is replaced by signed zero and its exact
/// bit-decoded magnitude is added to the certificate, so published centers and
/// positive errors are always zero, binary32-normal, or infinity. Accumulators
/// are f64 (spec R1; see [`scatter_rows_err_accumulators`]).
///
/// Non-finite/negative carried err poisons the row to `+INF` (outward degrade,
/// spec I5), matching the hardened 6D behavior (R3). The zero-magnitude
/// accumulator is short-circuited before multiplying by the
/// possibly-infinite `γ` so `INF·0` can never produce NaN.
///
/// Row range (#patches-row-range): emits rows `[row_start, row_end)` of the
/// spec-row axis (a `(row_end - row_start) × in_dim` matrix); `err_row` is
/// still indexed by the GLOBAL spec row.
///
/// The `err_row` length check (== `row_count`, spec I6) lives at the call site
/// in `materialize_dense_patches_to_dense`; allocation and strided-scratch
/// failures remain structured `CpuMemoryExceeded` results.
#[allow(clippy::too_many_arguments)]
fn try_patches_err_matrix_rows(
    center: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    err_row: Option<&Array1<f32>>,
    row_start: usize,
    row_end: usize,
    in_dim: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    admission: &mut PatchesMemoryAdmission,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<Array2<f32>> {
    let n_rows = row_end - row_start;
    let mut count = try_filled_array2(
        n_rows,
        in_dim,
        0.0_f64,
        admission,
        "patches 7D error-count allocation",
        deadline,
    )?;
    let mut absacc = try_filled_array2(
        n_rows,
        in_dim,
        0.0_f64,
        admission,
        "patches 7D error-magnitude allocation",
        deadline,
    )?;
    scatter_rows_err_accumulators_with_deadline(
        &mut count,
        &mut absacc,
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
        admission,
        deadline,
    )?;
    let mut err = try_filled_array2(
        n_rows,
        in_dim,
        0.0_f32,
        admission,
        "patches 7D certified-error allocation",
        deadline,
    )?;
    for (local, r) in (row_start..row_end).enumerate() {
        // Sanitize per row (spec I5): non-finite or negative carried err maps
        // to +INF (poisons outward); None means the side is exact (0).
        let er = match err_row {
            None => 0.0f64,
            Some(e) => {
                let v = e[r]; // length == row_count checked at the call site (I6)
                nonnegative_f32_error_or_infinity(v)
            }
        };
        for j in 0..in_dim {
            let c = count[[local, j]];
            if c > 0.0 {
                let gamma = crate::layers::linear::crown_single_gamma_n_f32(c as usize);
                let oe = absacc[[local, j]];
                // Short-circuit oe == 0 before multiplying by the possibly
                // infinite γ (c >= 2^24 ⇒ γ = +INF, and INF·0 = NaN); the
                // correct value there is the pure carry term (spec I5/C2).
                let acc = if oe == 0.0 { 0.0 } else { gamma * oe };
                let underflow = rows_f32_underflow_charge(c, oe);
                let center_flush = normalize_subnormal_center(&mut center[[local, j]]);
                // The historical carry + gamma term is retained byte-for-byte
                // for normal-scale data. New absolute flush charges only affect
                // the subnormal/cancellation regime they certify; non-finite
                // arithmetic publishes +INF rather than a NaN certificate.
                let term = c * er + acc + underflow + center_flush;
                err[[local, j]] = publish_error_up_normal(term);
            }
            deadline.work(1, "during patches 7D certified-error finalization")?;
        }
    }
    deadline.checkpoint("after patches 7D certified-error finalization")?;
    Ok(err)
}

/// Test oracle seam retaining the historical direct helper surface. Production
/// always passes the enclosing full-materialization admission instead.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn patches_err_matrix_rows(
    patches: &ArrayD<f32>,
    index_map: &UnfoldIndexMap,
    err_row: Option<&Array1<f32>>,
    row_start: usize,
    row_end: usize,
    in_dim: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) -> Array2<f32> {
    let n_rows = row_end - row_start;
    let matrix = matrix_bytes::<f32>(n_rows, in_dim);
    let required = matrix
        .saturating_mul(5)
        .saturating_add(matrix.max(scratch_bytes(patches)));
    let mut admission = PatchesMemoryAdmission::check(required, "patches 7D error test oracle")
        .expect("small direct error oracle must fit the configured dense budget");
    let mut deadline = PatchesMaterializationDeadline::new(None);
    let mut center = try_filled_array2(
        n_rows,
        in_dim,
        0.0_f32,
        &mut admission,
        "patches 7D error oracle center allocation",
        &mut deadline,
    )
    .expect("small direct error oracle center allocation");
    scatter_rows_with_unfold_map_with_deadline(
        &mut center,
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
    .expect("small direct error oracle center scatter");
    try_patches_err_matrix_rows(
        &mut center,
        patches,
        index_map,
        err_row,
        row_start,
        row_end,
        in_dim,
        out_c,
        out_h,
        out_w,
        in_c,
        kh,
        kw,
        &mut admission,
        &mut deadline,
    )
    .expect("small direct error oracle allocation")
}

#[cfg(test)]
use super::record_patches_to_dense_call_site;

impl PatchesLinearBounds {
    /// Convert to dense LinearBounds by materializing the full A-matrix.
    ///
    /// This is the fallback for layers that don't natively support Patches.
    /// For a Patches A with shape (out_c, out_h, out_w, in_c, kH, kW),
    /// affine or anchored typed geometry, and input_shape (in_c, in_h, in_w):
    ///
    /// 1. For each output position (oc, oh, ow):
    ///    - Compute the input receptive field position
    ///    - Place the patch coefficients into the correct input positions
    /// 2. Flatten to Array2<f32> of shape (out_dim, in_dim)
    ///
    /// Memory: allocates the full dense matrix. Only use at Patches termination
    /// points (Linear layer) or as fallback for unsupported layers. For
    /// over-budget relations, [`to_dense_rows`](Self::to_dense_rows) /
    /// [`concretize_sound_chunked`](Self::concretize_sound_chunked) bound the
    /// peak allocation (#patches-row-range).
    ///
    /// Builds one checked exact-integer unfold plan, shared between lower_a and
    /// upper_a. This avoids redundant position computation, eliminates the old
    /// f32 index carrier's >2^24 aliasing, and keeps bounds checks out of the
    /// scatter loop.
    ///
    /// Reference: alpha-beta-CROWN `auto_LiRPA/patches.py` (Patches.to_matrix)
    #[track_caller]
    pub(crate) fn to_dense(&self) -> Result<LinearBounds> {
        self.to_dense_for_purpose(PatchesMaterializationPurpose::Other)
    }

    /// No-deadline full materialization with an explicit semantic purpose.
    #[track_caller]
    pub(crate) fn to_dense_for_purpose(
        &self,
        purpose: PatchesMaterializationPurpose,
    ) -> Result<LinearBounds> {
        self.to_dense_with_deadline_for_purpose(None, purpose)
    }

    /// Deadline-aware full materialization. The deadline covers validation,
    /// allocation, fill/scatter, certified-error construction, numeric
    /// firewall scans, and the final post-wrap publication checkpoint.
    #[track_caller]
    pub(crate) fn to_dense_with_deadline(&self, deadline: Option<Instant>) -> Result<LinearBounds> {
        self.to_dense_with_deadline_for_purpose(deadline, PatchesMaterializationPurpose::Other)
    }

    /// Deadline-aware full materialization with an explicit semantic purpose.
    #[track_caller]
    pub(crate) fn to_dense_with_deadline_for_purpose(
        &self,
        deadline: Option<Instant>,
        purpose: PatchesMaterializationPurpose,
    ) -> Result<LinearBounds> {
        self.to_dense_with_deadline_and_resident_for_purpose(deadline, 0, purpose)
    }

    /// Deadline-aware full materialization while another request-owned payload
    /// remains live. `resident_base_bytes` excludes `self`; the materializer
    /// charges the borrowed Patches carrier exactly once in addition to this
    /// base and every operation-local allocation.
    #[allow(dead_code)]
    #[track_caller]
    pub(crate) fn to_dense_with_deadline_and_resident(
        &self,
        deadline: Option<Instant>,
        resident_base_bytes: usize,
    ) -> Result<LinearBounds> {
        self.to_dense_with_deadline_and_resident_for_purpose(
            deadline,
            resident_base_bytes,
            PatchesMaterializationPurpose::Other,
        )
    }

    /// Resident-memory-aware full materialization with an explicit semantic purpose.
    #[track_caller]
    pub(crate) fn to_dense_with_deadline_and_resident_for_purpose(
        &self,
        deadline: Option<Instant>,
        resident_base_bytes: usize,
        purpose: PatchesMaterializationPurpose,
    ) -> Result<LinearBounds> {
        #[cfg(test)]
        {
            let location = std::panic::Location::caller();
            record_patches_to_dense_call_site(format!("{}:{}", location.file(), location.line()));
        }
        let geometry = match (&self.lower_a.geometry, &self.upper_a.geometry) {
            (PatchGeometry::Affine(_), PatchGeometry::Affine(_)) => {
                PatchesMaterializationGeometry::Affine
            }
            (PatchGeometry::Anchored(_), PatchGeometry::Anchored(_)) => {
                PatchesMaterializationGeometry::Anchored
            }
            _ => PatchesMaterializationGeometry::Conflicting,
        };
        let input_coefficient_error =
            self.lower_a.coeff_err.is_some() || self.upper_a.coeff_err.is_some();
        record_patches_materialization_attempt(
            purpose,
            deadline.is_some(),
            geometry,
            input_coefficient_error,
        );
        // #patches-drop (dark, NY_PATCHES_CARRIER_TRACE=1, print-only): this is
        // the ONE funnel every full Patches->Dense carrier conversion reaches —
        // `CrownBounds::into_dense*`, `CrownBounds::ensure_dense*` and the
        // `to_dense*` wrappers all bottom out here, and every one of them is
        // `#[track_caller]`, so the propagated caller location names the site
        // that densified rather than this materializer. `Location::caller()` is
        // taken in the outcome arms below and still resolves to that caller.
        //
        // The deadline STATE is classified HERE, before the work, and the LINE
        // is emitted from the outcome arms. Both halves are deliberate:
        //   * classify first, because the materializer's own polls can expire a
        //     `live` deadline mid-flight and a state read afterwards would
        //     report `expired` for work that started with authority to spare;
        //   * emit from the arms, because a refusal DOES NOT DROP THE CARRIER.
        //     `CrownBounds::ensure_dense_with_deadline_for_purpose` leaves the
        //     Patches carrier untouched on `Err`, and
        //     `prepare_plain_dense_boundary_for_purpose` maps
        //     `DeadlineExceeded`/`CpuMemoryExceeded` to a CROWN->IBP fallback
        //     with the carrier still Patches. An entry-emitted line would name
        //     a non-densifying site in exactly the expiry regime this probe
        //     exists for.
        //
        // Gate first: unset, this is one latched-string compare producing
        // `None` — no clock read, no `Location::caller()`, no formatting, no
        // allocation. Armed, it is a `&'static str` in an `Option`, so the
        // binding is `Copy`, borrows nothing, and adds no drop to this scope.
        let carrier_trace_deadline = crate::patches_carrier_trace::enabled()
            .then(|| crate::patches_carrier_trace::deadline_state(deadline));
        let mut deadline = PatchesMaterializationDeadline::new(deadline);
        let materialized = if resident_base_bytes == 0 {
            self.to_dense_with_poll_receipt(&mut deadline)
        } else {
            self.to_dense_with_poll_and_resident(&mut deadline, resident_base_bytes)
        };
        match materialized {
            Ok(materialized) => {
                record_patches_materialization_success(
                    purpose,
                    materialized.coefficient_error,
                    materialized.receipt,
                );
                // #patches-drop: the ONLY arm that actually replaced the
                // carrier. `outcome=ok` is what a reader greps for.
                if let Some(deadline_state) = carrier_trace_deadline {
                    crate::patches_carrier_trace::record_densify(
                        std::panic::Location::caller(),
                        self,
                        deadline_state,
                        purpose,
                        None,
                    );
                }
                Ok(materialized.bounds)
            }
            Err(error) => {
                let refusal = match &error {
                    NyError::CpuMemoryExceeded { .. } => PatchesMaterializationRefusal::Memory,
                    NyError::DeadlineExceeded(_) => PatchesMaterializationRefusal::Deadline,
                    _ => PatchesMaterializationRefusal::Semantic,
                };
                record_patches_materialization_refusal(purpose, refusal);
                // #patches-drop: an ATTEMPT, not a drop — every consumer of this
                // `Err` is transactional and the carrier stays Patches.
                if let Some(deadline_state) = carrier_trace_deadline {
                    crate::patches_carrier_trace::record_densify(
                        std::panic::Location::caller(),
                        self,
                        deadline_state,
                        purpose,
                        Some(refusal),
                    );
                }
                Err(error)
            }
        }
    }

    #[cfg(test)]
    fn to_dense_with_poll(
        &self,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<LinearBounds> {
        self.to_dense_with_poll_receipt(deadline)
            .map(|materialized| materialized.bounds)
    }

    fn to_dense_with_poll_receipt(
        &self,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<FullDenseMaterialization> {
        self.to_dense_with_poll_and_resident(deadline, 0)
    }

    fn to_dense_with_poll_and_resident(
        &self,
        deadline: &mut PatchesMaterializationDeadline,
        resident_base_bytes: usize,
    ) -> Result<FullDenseMaterialization> {
        deadline.checkpoint("before full materialization")?;
        let total = self.dense_rows_total()?;
        deadline.checkpoint("after full row-count validation")?;
        self.to_dense_rows_impl(0, total, resident_base_bytes, deadline)
    }

    /// Materialize exactly rows `[row_start, row_end)` of the full dense form
    /// (#patches-row-range): the A rows, the bias slice, and the certified
    /// coeff-err rows are the bit-identical `[row_start, row_end)` slice of
    /// what [`to_dense`](Self::to_dense) builds — every scatter/err/bias write
    /// is row-local, and per-row accumulation order is preserved by the range
    /// kernels. `to_dense()` is exactly `to_dense_rows(0, total)`: ONE code
    /// path, so the full-range behavior stays byte-identical.
    // Stage-4 compatibility seam: staged callers need the historical
    // no-deadline row API while deadline-authoritative consumers migrate.
    #[allow(dead_code)]
    pub(crate) fn to_dense_rows(&self, row_start: usize, row_end: usize) -> Result<LinearBounds> {
        self.to_dense_rows_with_deadline(row_start, row_end, None)
    }

    /// Deadline-aware row-range materialization. A failure returns no partial
    /// `LinearBounds`; the borrowed Patches carrier is never mutated.
    // Stage-4 deadline seam retained for the next bounded caller migration.
    #[allow(dead_code)]
    pub(crate) fn to_dense_rows_with_deadline(
        &self,
        row_start: usize,
        row_end: usize,
        deadline: Option<Instant>,
    ) -> Result<LinearBounds> {
        let mut deadline = PatchesMaterializationDeadline::new(deadline);
        self.to_dense_rows_with_poll(row_start, row_end, &mut deadline)
    }

    // Request-local implementation delegate for the staged row deadline API.
    #[allow(dead_code)]
    fn to_dense_rows_with_poll(
        &self,
        row_start: usize,
        row_end: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<LinearBounds> {
        self.to_dense_rows_with_poll_and_resident(row_start, row_end, 0, deadline)
    }

    /// Internal row seam for callers that already own request-local buffers.
    /// `resident_base_bytes` excludes `self`; every variant adds the borrowed
    /// source Patches footprint exactly once before admitting allocations.
    fn to_dense_rows_with_poll_and_resident(
        &self,
        row_start: usize,
        row_end: usize,
        resident_base_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<LinearBounds> {
        deadline.checkpoint("before row-range materialization")?;
        let total = self.dense_rows_total()?;
        if row_start > row_end || row_end > total {
            return Err(NyError::InvalidSpec(format!(
                "patches to_dense_rows: invalid row range [{row_start}, {row_end}) \
                 for {total} dense rows"
            )));
        }
        deadline.checkpoint("after row-range validation")?;
        self.to_dense_rows_impl(row_start, row_end, resident_base_bytes, deadline)
            .map(|materialized| materialized.bounds)
    }

    /// Number of rows of the full dense materialization (`to_dense()`'s
    /// `num_outputs`), per variant: sparse layouts expand to the full output
    /// grid unless they carry explicit spec rows (5D, one dense row per spec
    /// row); identity is one row per output position; dense 6D/7D patches carry
    /// `row_count` logical rows. This is the row-range domain for
    /// [`to_dense_rows`](Self::to_dense_rows).
    fn dense_rows_total(&self) -> Result<usize> {
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "patches to_dense: output dims overflow: {out_c} * {out_h} * {out_w}"
            ))
        })?;
        if self.lower_a.unstable_idx.is_some() || self.upper_a.unstable_idx.is_some() {
            let explicit_rows = self.lower_a.patches.as_ref().is_some_and(|p| p.ndim() == 5);
            return Ok(if explicit_rows {
                self.row_count
            } else {
                out_dim
            });
        }
        if self.lower_a.identity && self.upper_a.identity {
            return Ok(out_dim);
        }
        Ok(self.row_count)
    }

    /// Shared range-materialization core (see [`to_dense_rows`]). The range is
    /// pre-validated against [`dense_rows_total`](Self::dense_rows_total).
    fn to_dense_rows_impl(
        &self,
        row_start: usize,
        row_end: usize,
        resident_base_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<FullDenseMaterialization> {
        self.validate_row_count()?;
        deadline.checkpoint("after patches row-count validation")?;
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let (in_c, in_h, in_w) = self.lower_a.input_shape;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "patches to_dense: output dims overflow: {out_c} * {out_h} * {out_w}"
            ))
        })?;
        let in_dim = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "patches to_dense: input dims overflow: {in_c} * {in_h} * {in_w}"
            ))
        })?;

        if self.lower_a.unstable_idx.is_some() || self.upper_a.unstable_idx.is_some() {
            return self.sparse_to_dense(row_start, row_end, resident_base_bytes, deadline);
        }
        if self.lower_a.identity && self.upper_a.identity {
            return self.identity_to_dense(
                out_dim,
                in_dim,
                row_start,
                row_end,
                resident_base_bytes,
                deadline,
            );
        }

        self.materialize_dense_patches_to_dense(
            out_c,
            out_h,
            out_w,
            in_c,
            in_dim,
            row_start,
            row_end,
            resident_base_bytes,
            deadline,
        )
    }

    fn identity_to_dense(
        &self,
        out_dim: usize,
        in_dim: usize,
        row_start: usize,
        row_end: usize,
        resident_base_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<FullDenseMaterialization> {
        self.lower_a
            .validate_common_geometry_with_poll(&self.upper_a, deadline)?;
        self.lower_a.validate_identity_geometry()?;
        self.upper_a.validate_identity_geometry()?;
        // Identity patches must be exact: every identity constructor sets
        // coeff_err None (patches.rs identity/sparse_identity,
        // types.rs materialize_identity), and this path emits no err matrix,
        // so a carried Some here would be silently dropped (unsound).
        if self.lower_a.coeff_err.is_some() || self.upper_a.coeff_err.is_some() {
            return Err(NyError::InternalError(
                "identity patches to_dense: coeff_err carried on an exact identity path".into(),
            ));
        }
        if self.row_count != out_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim],
                got: vec![self.row_count],
            });
        }
        if out_dim != in_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim],
                got: vec![in_dim],
            });
        }
        let n_rows = row_end - row_start;
        let local_bytes = matrix_bytes::<f32>(n_rows, in_dim)
            .saturating_mul(2)
            .saturating_add(allocation_bytes::<f32>(n_rows).saturating_mul(2));
        let required_bytes = self
            .memory_bytes()
            .saturating_add(resident_base_bytes)
            .saturating_add(local_bytes);
        let mut admission =
            PatchesMemoryAdmission::check(required_bytes, "patches identity to_dense")?;
        deadline.checkpoint("after patches identity memory admission")?;
        // Rows [row_start, row_end) of the identity: a single 1.0 at column
        // `row` (full range reproduces `Array2::eye(out_dim)` exactly).
        let mut lower_a = try_filled_array2(
            n_rows,
            in_dim,
            0.0_f32,
            &mut admission,
            "patches identity lower allocation",
            deadline,
        )?;
        let mut upper_a = try_filled_array2(
            n_rows,
            in_dim,
            0.0_f32,
            &mut admission,
            "patches identity upper allocation",
            deadline,
        )?;
        for row in row_start..row_end {
            lower_a[[row - row_start, row]] = 1.0;
            upper_a[[row - row_start, row]] = 1.0;
            deadline.work(1, "during patches identity diagonal fill")?;
        }
        let lower_b = try_copy_bias_range(
            &self.lower_b,
            row_start,
            row_end,
            &mut admission,
            "patches identity lower-bias allocation",
            deadline,
        )?;
        let upper_b = try_copy_bias_range(
            &self.upper_b,
            row_start,
            row_end,
            &mut admission,
            "patches identity upper-bias allocation",
            deadline,
        )?;
        let bounds =
            finish_dense_in_place(lower_a, lower_b, upper_a, upper_b, None, None, deadline)?;
        Ok(FullDenseMaterialization::new(bounds, &admission))
    }

    #[allow(clippy::too_many_arguments)]
    fn materialize_dense_patches_to_dense(
        &self,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        in_c: usize,
        in_dim: usize,
        row_start: usize,
        row_end: usize,
        resident_base_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<FullDenseMaterialization> {
        // Reject malformed proof metadata before building an unfold map or
        // allocating dense matrices. Every certificate is row-wide on both
        // supported layouts.
        for err in [
            self.lower_a.coeff_err.as_ref(),
            self.upper_a.coeff_err.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if err.len() != self.row_count {
                return Err(NyError::ShapeMismatch {
                    expected: vec![self.row_count],
                    got: vec![err.len()],
                });
            }
        }
        let (lower_patches, lower_explicit_rows) =
            validate_patches_shape(&self.lower_a, self.row_count, out_c, out_h, out_w, in_c)?;
        let lower_shape = lower_patches.shape();
        let (kh, kw) = if lower_explicit_rows {
            (lower_shape[5], lower_shape[6])
        } else {
            (lower_shape[4], lower_shape[5])
        };
        let (upper_patches, upper_explicit_rows) =
            validate_patches_shape(&self.upper_a, self.row_count, out_c, out_h, out_w, in_c)?;

        // A 6D (broadcast) side has exactly one logical row per output
        // position; a `row_count` mismatch used to scatter past the dense
        // allocation (index panic) or leave phantom bias-only rows. Reject it
        // with a clean error so the caller falls back to the sound dense path
        // (#patches-row-range hardening; the range math below indexes the
        // output grid by the same row space).
        if !lower_explicit_rows || !upper_explicit_rows {
            let out_dim = out_c * out_h * out_w;
            if self.row_count != out_dim {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_dim],
                    got: vec![self.row_count],
                });
            }
        }

        // Geometry cross-check (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §3.3, B2):
        // the upper scatter (and err materialization) reuses the LOWER side's
        // unfold index_map, which is only sound when both sides agree on the
        // receptive-field geometry. Divergence used to silently mis-scatter the
        // upper COEFFICIENTS (or slice-panic); reject it with a hard error so
        // the caller falls back to the sound dense path. Run ONCE per call
        // (whole-relation property, independent of the row range).
        let upper_shape = upper_patches.shape();
        let (ukh, ukw) = if upper_explicit_rows {
            (upper_shape[5], upper_shape[6])
        } else {
            (upper_shape[4], upper_shape[5])
        };
        if (ukh, ukw) != (kh, kw) {
            return Err(NyError::ShapeMismatch {
                expected: vec![kh, kw],
                got: vec![ukh, ukw],
            });
        }
        self.lower_a
            .validate_common_geometry_with_poll(&self.upper_a, deadline)?;
        let resident_bytes = self.memory_bytes().saturating_add(resident_base_bytes);
        let index_map = compute_unfold_index_map_with_deadline(
            &self.lower_a,
            kh,
            kw,
            resident_bytes,
            deadline,
        )?;

        let n_rows = row_end - row_start;
        let dense_matrix_bytes = matrix_bytes::<f32>(n_rows, in_dim);
        let bias_pair_bytes = allocation_bytes::<f32>(n_rows).saturating_mul(2);
        let map_bytes = index_map.memory_bytes();
        let lower_scratch_bytes = scratch_bytes(lower_patches);
        let upper_scratch_bytes = scratch_bytes(upper_patches);
        // A generic Patches carrier is not guaranteed to come from a producer
        // which already normalized subnormals. Detect that narrow case before
        // admission so 6D can publish a signed-zero center plus an exact error
        // charge without taxing the historical normal-data exact fast path.
        let lower_has_subnormal = !lower_explicit_rows
            && has_scattered_subnormal_6d_with_poll(
                lower_patches,
                &index_map,
                row_start,
                row_end,
                out_c,
                kh,
                kw,
                deadline,
            )?;
        let upper_has_subnormal = !upper_explicit_rows
            && has_scattered_subnormal_6d_with_poll(
                upper_patches,
                &index_map,
                row_start,
                row_end,
                out_c,
                kh,
                kw,
                deadline,
            )?;
        let need_err = self.lower_a.coeff_err.is_some()
            || self.upper_a.coeff_err.is_some()
            || lower_explicit_rows
            || upper_explicit_rows
            || lower_has_subnormal
            || upper_has_subnormal;

        let (lower_error, upper_error) = if need_err {
            let lower = if lower_explicit_rows {
                SideErrorAllocation::Rows7
            } else if self.lower_a.coeff_err.is_some() || lower_has_subnormal {
                SideErrorAllocation::Dense6
            } else {
                SideErrorAllocation::Zero
            };
            let upper = if upper_explicit_rows {
                SideErrorAllocation::Rows7
            } else if self.upper_a.coeff_err.is_some() || upper_has_subnormal {
                SideErrorAllocation::Dense6
            } else {
                SideErrorAllocation::Zero
            };
            (Some(lower), Some(upper))
        } else {
            (None, None)
        };
        // Account the exact allocation timeline. The map remains live through
        // both dense scatters and error construction. Error accumulators are
        // transient, while the completed lower error remains live during the
        // upper side. A strided patch scratch and the final error matrix are
        // not live simultaneously, hence their `max` in `account_error_side`.
        let peak_bytes = dense_materialization_peak_bytes(
            resident_bytes,
            map_bytes,
            dense_matrix_bytes,
            bias_pair_bytes,
            lower_scratch_bytes,
            upper_scratch_bytes,
            lower_error,
            upper_error,
        );
        let mut admission =
            PatchesMemoryAdmission::check(peak_bytes, "patches full dense materialization")?;
        deadline.checkpoint("after patches full dense memory admission")?;

        let mut lower_dense = try_filled_array2(
            n_rows,
            in_dim,
            0.0_f32,
            &mut admission,
            "patches lower dense allocation",
            deadline,
        )?;
        if lower_explicit_rows {
            scatter_rows_with_unfold_map_with_deadline(
                &mut lower_dense,
                lower_patches,
                &index_map,
                row_start,
                row_end,
                out_c,
                out_h,
                out_w,
                in_c,
                kh,
                kw,
                &mut admission,
                deadline,
            )?;
        } else {
            scatter_with_unfold_map_with_deadline(
                &mut lower_dense,
                lower_patches,
                &index_map,
                out_c,
                out_h,
                out_w,
                in_c,
                kh,
                kw,
                row_start,
                row_end,
                &mut admission,
                deadline,
            )?;
        }

        let mut upper_dense = try_filled_array2(
            n_rows,
            in_dim,
            0.0_f32,
            &mut admission,
            "patches upper dense allocation",
            deadline,
        )?;
        if upper_explicit_rows {
            scatter_rows_with_unfold_map_with_deadline(
                &mut upper_dense,
                upper_patches,
                &index_map,
                row_start,
                row_end,
                out_c,
                out_h,
                out_w,
                in_c,
                kh,
                kw,
                &mut admission,
                deadline,
            )?;
        } else {
            scatter_with_unfold_map_with_deadline(
                &mut upper_dense,
                upper_patches,
                &index_map,
                out_c,
                out_h,
                out_w,
                in_c,
                kh,
                kw,
                row_start,
                row_end,
                &mut admission,
                deadline,
            )?;
        }

        let lower_b = try_copy_bias_range(
            &self.lower_b,
            row_start,
            row_end,
            &mut admission,
            "patches lower-bias allocation",
            deadline,
        )?;
        let upper_b = try_copy_bias_range(
            &self.upper_b,
            row_start,
            row_end,
            &mut admission,
            "patches upper-bias allocation",
            deadline,
        )?;

        // Attach the certified coefficient error (#patches-coeff-err-soundness,
        // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §3.1). A side needs a
        // materialized err matrix when it carries a coeff_err (any layout) OR
        // when it is explicit-rows (7D) — INCLUDING the `(None,None)` pair:
        // the 7D scatter genuinely accumulates multiple taps per dense cell,
        // so the `γ(N)·S_hat` accumulation-rounding term exists even at
        // `e_r = 0` (spec R2). The plain `new_or_conservative` fast path
        // remains ONLY for `(None,None)` with both sides 6D (per-cell tap
        // count is provably 0/1 there) — byte-identical to the exact path.
        if !need_err {
            let bounds = finish_dense_in_place(
                lower_dense,
                lower_b,
                upper_dense,
                upper_b,
                None,
                None,
                deadline,
            )?;
            return Ok(FullDenseMaterialization::new(bounds, &admission));
        }
        // Per-side dispatch on that side's OWN layout flag / err (mixed 6D/7D
        // pairs pass validation and each get their own treatment).
        let mut side_err = |center: &mut Array2<f32>,
                            err: Option<&Array1<f32>>,
                            patches: &ArrayD<f32>,
                            explicit_rows: bool,
                            has_subnormal: bool|
         -> Result<Array2<f32>> {
            // Length and the combined live peak were preflighted before the
            // dense pair, so the kernels may index directly and every reserve
            // failure remains a typed resource refusal.
            if explicit_rows {
                try_patches_err_matrix_rows(
                    center,
                    patches,
                    &index_map,
                    err,
                    row_start,
                    row_end,
                    in_dim,
                    out_c,
                    out_h,
                    out_w,
                    in_c,
                    kh,
                    kw,
                    &mut admission,
                    deadline,
                )
            } else {
                if err.is_some() || has_subnormal {
                    patches_err_matrix(
                        center,
                        patches,
                        &index_map,
                        err,
                        row_start,
                        row_end,
                        in_dim,
                        out_c,
                        out_h,
                        out_w,
                        in_c,
                        kh,
                        kw,
                        &mut admission,
                        deadline,
                    )
                } else {
                    // A 6D side with no carried err is exact (≤ 1 tap/cell).
                    try_filled_array2(
                        n_rows,
                        in_dim,
                        0.0_f32,
                        &mut admission,
                        "patches exact-side error allocation",
                        deadline,
                    )
                }
            }
        };
        let lower_err = side_err(
            &mut lower_dense,
            self.lower_a.coeff_err.as_ref(),
            lower_patches,
            lower_explicit_rows,
            lower_has_subnormal,
        )?;
        let upper_err = side_err(
            &mut upper_dense,
            self.upper_a.coeff_err.as_ref(),
            upper_patches,
            upper_explicit_rows,
            upper_has_subnormal,
        )?;
        let bounds = finish_dense_in_place(
            lower_dense,
            lower_b,
            upper_dense,
            upper_b,
            Some(lower_err),
            Some(upper_err),
            deadline,
        )?;
        Ok(FullDenseMaterialization::new(bounds, &admission))
    }

    /// Convert sparse patches to dense LinearBounds.
    ///
    /// Sparse patches have 4D shape `(unstable_size, in_c, kH, kW)` with
    /// `unstable_idx` mapping each sparse row to an `(c, h, w)` output position.
    /// This scatters the sparse rows into a full `(out_dim, in_dim)` dense matrix
    /// with zeros for stable neuron rows.
    ///
    /// Bias vectors are similarly expanded from sparse (len=unstable_size) to
    /// full (len=out_dim), with zero for stable positions.
    ///
    /// Row range (#patches-row-range): only rows `[row_start, row_end)` of the
    /// full dense form are materialized (sparse rows landing outside the range
    /// are skipped; in-range rows shift down by `row_start`).
    fn sparse_to_dense(
        &self,
        row_start: usize,
        row_end: usize,
        resident_base_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<FullDenseMaterialization> {
        // Validate the complete lower/upper sparse contract before deriving a
        // map from the lower side or allocating either dense matrix.
        let idx = self.validate_sparse_pair_with_poll(deadline)?;

        // Scope guard (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md I2/B6): the sparse
        // layout stays `coeff_err = None` at every site and this path emits no
        // err matrix, so a carried Some reaching it would be silently dropped
        // — the false-VERIFIED direction. Convert that guard violation into a
        // hard error.
        if self.lower_a.coeff_err.is_some() || self.upper_a.coeff_err.is_some() {
            return Err(NyError::InternalError(
                "sparse patches to_dense: coeff_err carried on sparse path (unsupported; \
                 sparse stays exact by scope guard)"
                    .into(),
            ));
        }
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let (in_c, in_h, in_w) = self.lower_a.input_shape;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "patches sparse_to_dense: output dims overflow: {out_c} * {out_h} * {out_w}"
            ))
        })?;
        let in_dim = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "patches sparse_to_dense: input dims overflow: {in_c} * {in_h} * {in_w}"
            ))
        })?;

        if self.lower_a.identity && self.upper_a.identity {
            return self.sparse_identity_to_dense(
                idx,
                out_dim,
                in_dim,
                out_h,
                out_w,
                row_start,
                row_end,
                resident_base_bytes,
                deadline,
            );
        }

        let (lower_dense, upper_dense, explicit_rows, admission) = self
            .materialize_sparse_dense_pair(
                idx,
                out_h,
                out_w,
                in_c,
                in_dim,
                row_start,
                row_end,
                resident_base_bytes,
                deadline,
            )?;
        let mut admission = admission;
        let (lower_b, upper_b) = if explicit_rows {
            (
                try_copy_bias_range(
                    &self.lower_b,
                    row_start,
                    row_end,
                    &mut admission,
                    "sparse patches lower-bias allocation",
                    deadline,
                )?,
                try_copy_bias_range(
                    &self.upper_b,
                    row_start,
                    row_end,
                    &mut admission,
                    "sparse patches upper-bias allocation",
                    deadline,
                )?,
            )
        } else {
            Self::expand_sparse_bias(
                &self.lower_b,
                &self.upper_b,
                idx,
                row_start,
                row_end,
                out_h,
                out_w,
                &mut admission,
                deadline,
            )?
        };
        let bounds = finish_dense_in_place(
            lower_dense,
            lower_b,
            upper_dense,
            upper_b,
            None,
            None,
            deadline,
        )?;
        Ok(FullDenseMaterialization::new(bounds, &admission))
    }

    #[allow(clippy::too_many_arguments)]
    fn materialize_sparse_dense_pair(
        &self,
        idx: &UnstableIdx,
        out_h: usize,
        out_w: usize,
        in_c: usize,
        in_dim: usize,
        row_start: usize,
        row_end: usize,
        resident_base_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<(Array2<f32>, Array2<f32>, bool, PatchesMemoryAdmission)> {
        let lower_patches = self.lower_a.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("sparse_to_dense: lower patches tensor is None".into())
        })?;
        let shape = lower_patches.shape();
        let explicit_rows = match shape.len() {
            4 => false,
            5 => {
                if shape[0] != self.row_count {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![self.row_count],
                        got: vec![shape[0]],
                    });
                }
                true
            }
            _ => {
                return Err(NyError::ShapeMismatch {
                    expected: vec![4, 5],
                    got: vec![shape.len()],
                });
            }
        };
        let (kh, kw) = if explicit_rows {
            (shape[3], shape[4])
        } else {
            (shape[2], shape[3])
        };
        let resident_bytes = self.memory_bytes().saturating_add(resident_base_bytes);
        let index_map = compute_unfold_index_map_with_deadline(
            &self.lower_a,
            kh,
            kw,
            resident_bytes,
            deadline,
        )?;
        let n_rows = row_end - row_start;
        let upper_patches = self.upper_a.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("sparse_to_dense: upper patches tensor is None".into())
        })?;
        let dense_matrix_bytes = matrix_bytes::<f32>(n_rows, in_dim);
        let bias_pair_bytes = allocation_bytes::<f32>(n_rows).saturating_mul(2);
        let map_bytes = index_map.memory_bytes();
        let mut peak_bytes = resident_bytes
            .saturating_add(map_bytes)
            .saturating_add(dense_matrix_bytes)
            .saturating_add(scratch_bytes(lower_patches));
        peak_bytes = peak_bytes.max(
            resident_bytes
                .saturating_add(map_bytes)
                .saturating_add(dense_matrix_bytes.saturating_mul(2))
                .saturating_add(scratch_bytes(upper_patches)),
        );
        // The map drops when this helper returns; the bias pair is allocated
        // afterward while only the completed dense pair remains live.
        peak_bytes = peak_bytes.max(
            resident_bytes
                .saturating_add(dense_matrix_bytes.saturating_mul(2))
                .saturating_add(bias_pair_bytes),
        );
        let mut admission = PatchesMemoryAdmission::check(peak_bytes, "sparse patches to_dense")?;
        deadline.checkpoint("after sparse patches memory admission")?;
        let mut lower_dense = try_filled_array2(
            n_rows,
            in_dim,
            0.0_f32,
            &mut admission,
            "sparse patches lower dense allocation",
            deadline,
        )?;
        let mut upper_dense = try_filled_array2(
            n_rows,
            in_dim,
            0.0_f32,
            &mut admission,
            "sparse patches upper dense allocation",
            deadline,
        )?;

        if explicit_rows {
            scatter_sparse_rows_with_unfold_map_with_deadline(
                &mut lower_dense,
                lower_patches,
                &index_map,
                row_start,
                row_end,
                idx,
                in_c,
                kh,
                kw,
                &mut admission,
                deadline,
            )?;
            scatter_sparse_rows_with_unfold_map_with_deadline(
                &mut upper_dense,
                upper_patches,
                &index_map,
                row_start,
                row_end,
                idx,
                in_c,
                kh,
                kw,
                &mut admission,
                deadline,
            )?;
        } else {
            scatter_sparse_with_unfold_map_with_deadline(
                &mut lower_dense,
                lower_patches,
                &index_map,
                idx,
                out_h,
                out_w,
                in_c,
                kh,
                kw,
                row_start,
                row_end,
                &mut admission,
                deadline,
            )?;
            scatter_sparse_with_unfold_map_with_deadline(
                &mut upper_dense,
                upper_patches,
                &index_map,
                idx,
                out_h,
                out_w,
                in_c,
                kh,
                kw,
                row_start,
                row_end,
                &mut admission,
                deadline,
            )?;
        }

        Ok((lower_dense, upper_dense, explicit_rows, admission))
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_identity_to_dense(
        &self,
        idx: &UnstableIdx,
        out_dim: usize,
        in_dim: usize,
        out_h: usize,
        out_w: usize,
        row_start: usize,
        row_end: usize,
        resident_base_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<FullDenseMaterialization> {
        if out_dim != in_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim],
                got: vec![in_dim],
            });
        }
        let n_rows = row_end - row_start;
        let local_bytes = matrix_bytes::<f32>(n_rows, in_dim)
            .saturating_mul(2)
            .saturating_add(allocation_bytes::<f32>(n_rows).saturating_mul(2));
        let required_bytes = self
            .memory_bytes()
            .saturating_add(resident_base_bytes)
            .saturating_add(local_bytes);
        let mut admission =
            PatchesMemoryAdmission::check(required_bytes, "sparse identity patches to_dense")?;
        deadline.checkpoint("after sparse identity patches memory admission")?;
        let mut lower_a = try_filled_array2(
            n_rows,
            in_dim,
            0.0_f32,
            &mut admission,
            "sparse identity lower dense allocation",
            deadline,
        )?;
        let mut upper_a = try_filled_array2(
            n_rows,
            in_dim,
            0.0_f32,
            &mut admission,
            "sparse identity upper dense allocation",
            deadline,
        )?;
        let mut lower_b = try_zeroed_array1(
            n_rows,
            &mut admission,
            "sparse identity lower-bias allocation",
            deadline,
        )?;
        let mut upper_b = try_zeroed_array1(
            n_rows,
            &mut admission,
            "sparse identity upper-bias allocation",
            deadline,
        )?;
        for i in 0..idx.len() {
            deadline.work(1, "during sparse identity patches index walk")?;
            let flat = idx.flat_index(i, out_h, out_w);
            if flat < row_start || flat >= row_end {
                continue;
            }
            lower_a[[flat - row_start, flat]] = 1.0;
            upper_a[[flat - row_start, flat]] = 1.0;
            lower_b[flat - row_start] = self.lower_b[i];
            upper_b[flat - row_start] = self.upper_b[i];
            deadline.work(1, "during sparse identity patches fill")?;
        }
        let bounds =
            finish_dense_in_place(lower_a, lower_b, upper_a, upper_b, None, None, deadline)?;
        Ok(FullDenseMaterialization::new(bounds, &admission))
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_sparse_bias(
        sparse_lower: &Array1<f32>,
        sparse_upper: &Array1<f32>,
        idx: &UnstableIdx,
        row_start: usize,
        row_end: usize,
        out_h: usize,
        out_w: usize,
        admission: &mut PatchesMemoryAdmission,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        let n_rows = row_end - row_start;
        let mut lower_b = try_zeroed_array1(
            n_rows,
            admission,
            "sparse patches expanded lower-bias allocation",
            deadline,
        )?;
        let mut upper_b = try_zeroed_array1(
            n_rows,
            admission,
            "sparse patches expanded upper-bias allocation",
            deadline,
        )?;
        for i in 0..idx.len() {
            deadline.work(1, "during sparse patches expanded bias index walk")?;
            let flat = idx.flat_index(i, out_h, out_w);
            if flat < row_start || flat >= row_end {
                continue;
            }
            lower_b[flat - row_start] = sparse_lower[i];
            upper_b[flat - row_start] = sparse_upper[i];
            deadline.work(1, "during sparse patches expanded bias fill")?;
        }
        deadline.checkpoint("after sparse patches expanded bias fill")?;
        Ok((lower_b, upper_b))
    }

    /// Concretize over `input` blockwise (#patches-row-range): loop
    /// `to_dense_rows(r0, r1)` blocks sized to `max_block_bytes`, run the SAME
    /// `LinearBounds::concretize_sound` (directed f64→f32 rounding + certified
    /// coeff-err folding — NOT reimplemented here) on each block, and assemble
    /// the full flat `[rows]` concrete bounds.
    ///
    /// BIT-IDENTICAL to `self.to_dense()?.concretize_sound(input)`: the dense
    /// scatter, the err materialization, and the concretize dot products are
    /// all row-local, so each row takes exactly the value it takes in the
    /// single-shot path (pinned by
    /// `test_concretize_sound_chunked_matches_unchunked`). The one caveat is
    /// the whole-matrix NaN firewall (`LinearBounds::new_or_conservative`): a
    /// NaN coefficient degrades only its own BLOCK's rows to conservative
    /// instead of all rows — each row is still either its exact single-shot
    /// value or the sound `[-inf, +inf]` degrade, never tighter.
    ///
    /// Memory: peak is O(block) instead of O(rows × in_dim) — the fix for the
    /// VGG16 final-densify abort (3.2M × 150K rows ≈ 1.9 TB per matrix). Every
    /// block receipt includes the borrowed source Patches carrier and the
    /// already-live full output pair, then adds the lower/upper A pair,
    /// lower/upper err pair, block biases, transient f64 err accumulators
    /// (count + absacc), `BoundedTensor::flatten`'s lower/upper input copies,
    /// the lower/upper f64 concretization endpoints, and the returned
    /// lower/upper f32 block result. Thus blocks share one total-live budget
    /// rather than each independently consuming the full allowance. The
    /// immutable caller-owned input tensor belongs to the enclosing graph-state
    /// headroom; both input copies allocated by concretization are charged.
    ///
    /// `deadline` covers output allocation/fill, every row materialization,
    /// block boundary, certified dense reduction, endpoint publication and
    /// validation, output copy, and final wrapping. The completed block uses the
    /// fallible `LinearBounds` concretization face with this same request-local
    /// deadline state; expiry or allocation refusal therefore discards all local
    /// output and returns atomically.
    pub(crate) fn concretize_sound_chunked(
        &self,
        input: &BoundedTensor,
        max_block_bytes: usize,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let mut deadline = PatchesMaterializationDeadline::new(deadline);
        deadline.checkpoint("before patches chunked concretization")?;
        let total = self.dense_rows_total()?;
        let (in_c, in_h, in_w) = self.lower_a.input_shape;
        let in_dim = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "patches chunked concretize: input dims overflow: {in_c} * {in_h} * {in_w}"
            ))
        })?;
        // Row sizing is only a first estimate; every actual block is admitted
        // by the exact conservative total-live receipt below. Include the 7D
        // dense peak, block biases, nested f64/f32 result pairs, and deduct the
        // nested flattened-input/shape fixed cost before choosing a row count.
        // Exact 6D relations therefore get smaller blocks than strictly needed.
        let per_row_bytes = chunked_concretization_per_row_bytes(in_dim);
        let nested_fixed_bytes = chunked_concretization_fixed_bytes(in_dim);
        let output_pair_bytes = allocation_bytes::<f32>(total).saturating_mul(2);
        let source_bytes = self.memory_bytes();
        let mut output_admission = PatchesMemoryAdmission::check(
            source_bytes.saturating_add(output_pair_bytes),
            "patches chunked concretize output pair",
        )?;
        let mut out_lower = try_filled_f32_vec(
            total,
            0.0,
            &mut output_admission,
            "patches chunked concretize lower output allocation",
            &mut deadline,
        )?;
        let mut out_upper = try_filled_f32_vec(
            total,
            0.0,
            &mut output_admission,
            "patches chunked concretize upper output allocation",
            &mut deadline,
        )?;
        let output_resident_bytes = out_lower
            .capacity()
            .saturating_add(out_upper.capacity())
            .saturating_mul(size_of::<f32>());
        let total_resident_bytes = source_bytes.saturating_add(output_resident_bytes);
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let available_block_bytes = max_block_bytes
            .min(budget_bytes.saturating_sub(total_resident_bytes))
            .saturating_sub(nested_fixed_bytes);
        let rows_per_block = (available_block_bytes / per_row_bytes).max(1);
        let mut r0 = 0usize;
        while r0 < total {
            deadline.checkpoint("between patches chunked concretize blocks")?;
            let r1 = r0.saturating_add(rows_per_block).min(total);
            let nested_concretization_bytes = chunked_concretization_nested_bytes(r1 - r0, in_dim);
            let block = self.to_dense_rows_with_poll_and_resident(
                r0,
                r1,
                output_resident_bytes.saturating_add(nested_concretization_bytes),
                &mut deadline,
            )?;
            let concrete = block.concretize_sound_with_deadline_state(input, &mut deadline)?;
            deadline.checkpoint("after patches chunked block concretization")?;
            let lo = concrete.lower();
            let lo = lo.as_slice().ok_or_else(|| {
                NyError::InvalidSpec("patches chunked concretize: lower not contiguous".to_string())
            })?;
            let up = concrete.upper();
            let up = up.as_slice().ok_or_else(|| {
                NyError::InvalidSpec("patches chunked concretize: upper not contiguous".to_string())
            })?;
            let mut copied = 0usize;
            while copied < r1 - r0 {
                let end = copied
                    .saturating_add(PatchesMaterializationDeadline::CHECK_STRIDE / 2)
                    .min(r1 - r0);
                out_lower[r0 + copied..r0 + end].copy_from_slice(&lo[copied..end]);
                out_upper[r0 + copied..r0 + end].copy_from_slice(&up[copied..end]);
                deadline.work(
                    (end - copied).saturating_mul(2),
                    "during patches chunked output copy",
                )?;
                copied = end;
            }
            r0 = r1;
        }

        deadline.checkpoint("before patches chunked output wrapping")?;
        let lower = ArrayD::from_shape_vec(IxDyn(&[total]), out_lower)
            .map_err(|e| NyError::InvalidSpec(format!("patches chunked concretize lower: {e}")))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&[total]), out_upper)
            .map_err(|e| NyError::InvalidSpec(format!("patches chunked concretize upper: {e}")))?;
        // ±inf rows are legal here (concretize_sound emits the sound
        // [-inf, +inf] degrade for repaired rows), matching the single-shot
        // result exactly.
        let bounded = BoundedTensor::new_allow_infinite_with_poll(lower, upper, || {
            deadline.checkpoint("during patches chunked output validation")
        })?;
        deadline.checkpoint("after patches chunked output wrapping")?;
        Ok(bounded)
    }
}

#[cfg(test)]
mod anchored_tests {
    use super::*;
    use crate::bounds::patches::{PatchGeometry, PatchesData, UnstableIdx};
    use std::time::Duration;

    fn anchored_bounds(patches: ArrayD<f32>, row_count: usize) -> PatchesLinearBounds {
        let data = PatchesData {
            coeff_err: None,
            patches: Some(patches),
            geometry: PatchGeometry::anchored(vec![0], vec![0, 3]).unwrap(),
            identity: false,
            output_shape: (1, 1, 2),
            input_shape: (1, 1, 4),
            unstable_idx: None,
        };
        PatchesLinearBounds {
            row_count,
            lower_a: data.clone(),
            lower_b: Array1::zeros(row_count),
            upper_a: data,
            upper_b: Array1::zeros(row_count),
        }
    }

    fn explicit_row_bounds(row_count: usize) -> PatchesLinearBounds {
        let values: Vec<f32> = (0..row_count)
            .map(|row| ((row % 17) as f32 - 8.0) * 0.125)
            .collect();
        let data = PatchesData {
            coeff_err: Some(Array1::from_elem(row_count, f32::from_bits(0x3380_0000))),
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[row_count, 1, 1, 1, 1, 1, 1]), values)
                    .expect("explicit-row fixture shape"),
            ),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 1),
            unstable_idx: None,
        };
        PatchesLinearBounds {
            row_count,
            lower_a: data.clone(),
            lower_b: Array1::from_iter((0..row_count).map(|row| row as f32 * 0.25)),
            upper_a: data,
            upper_b: Array1::from_iter((0..row_count).map(|row| row as f32 * 0.5)),
        }
    }

    fn assert_dense_exact(actual: &LinearBounds, expected: &LinearBounds) {
        assert_eq!(actual.lower_a(), expected.lower_a());
        assert_eq!(actual.lower_b(), expected.lower_b());
        assert_eq!(actual.upper_a(), expected.upper_a());
        assert_eq!(actual.upper_b(), expected.upper_b());
        assert_eq!(actual.lower_a_err(), expected.lower_a_err());
        assert_eq!(actual.upper_a_err(), expected.upper_a_err());
    }

    fn assert_patches_exact(actual: &PatchesLinearBounds, expected: &PatchesLinearBounds) {
        assert_eq!(actual.row_count, expected.row_count);
        assert_eq!(actual.lower_a.coeff_err, expected.lower_a.coeff_err);
        assert_eq!(actual.lower_a.patches, expected.lower_a.patches);
        assert_eq!(actual.lower_a.geometry, expected.lower_a.geometry);
        assert_eq!(actual.lower_a.identity, expected.lower_a.identity);
        assert_eq!(actual.lower_a.output_shape, expected.lower_a.output_shape);
        assert_eq!(actual.lower_a.input_shape, expected.lower_a.input_shape);
        assert_eq!(actual.lower_a.unstable_idx, expected.lower_a.unstable_idx);
        assert_eq!(actual.lower_b, expected.lower_b);
        assert_eq!(actual.upper_a.coeff_err, expected.upper_a.coeff_err);
        assert_eq!(actual.upper_a.patches, expected.upper_a.patches);
        assert_eq!(actual.upper_a.geometry, expected.upper_a.geometry);
        assert_eq!(actual.upper_a.identity, expected.upper_a.identity);
        assert_eq!(actual.upper_a.output_shape, expected.upper_a.output_shape);
        assert_eq!(actual.upper_a.input_shape, expected.upper_a.input_shape);
        assert_eq!(actual.upper_a.unstable_idx, expected.upper_a.unstable_idx);
        assert_eq!(actual.upper_b, expected.upper_b);
    }

    #[test]
    fn deadline_aware_full_and_row_materialization_are_bit_exact() {
        crate::tests::with_env_edits(|_env| {
            let bounds = explicit_row_bounds(7);
            let baseline = bounds.to_dense().expect("unbounded full materialization");
            let bounded = bounds
                .to_dense_with_deadline(Some(Instant::now() + Duration::from_secs(30)))
                .expect("live deadline full materialization");
            assert_dense_exact(&bounded, &baseline);

            let baseline_rows = bounds
                .to_dense_rows(2, 6)
                .expect("unbounded row materialization");
            let bounded_rows = bounds
                .to_dense_rows_with_deadline(2, 6, Some(Instant::now() + Duration::from_secs(30)))
                .expect("live deadline row materialization");
            assert_dense_exact(&bounded_rows, &baseline_rows);

            let anchored = anchored_bounds(
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 1, 1, 1]), vec![2.0, -3.0])
                    .expect("anchored fixture shape"),
                2,
            );
            let anchored_baseline = anchored.to_dense().expect("unbounded anchored materialize");
            let anchored_bounded = anchored
                .to_dense_with_deadline(Some(Instant::now() + Duration::from_secs(30)))
                .expect("live deadline anchored materialize");
            assert_dense_exact(&anchored_bounded, &anchored_baseline);
        });
    }

    #[test]
    fn full_materializer_publishes_exact_typed_receipts_for_distinct_branches() {
        let _telemetry_lock = crate::execution_telemetry::TEST_LOCK
            .lock()
            .expect("telemetry test lock");
        let _run = crate::execution_telemetry::begin_run();

        let _ = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2))
            .to_dense_for_purpose(PatchesMaterializationPurpose::NetworkInputTerminal)
            .expect("identity materialization");
        let _ = explicit_row_bounds(2)
            .to_dense_with_deadline_for_purpose(
                Some(Instant::now() + Duration::from_secs(30)),
                PatchesMaterializationPurpose::LatentInputCrossover,
            )
            .expect("explicit-row materialization");
        let _ = PatchesLinearBounds::sparse_identity(
            (1, 2, 2),
            (1, 2, 2),
            UnstableIdx {
                channels: vec![0, 0],
                heights: vec![0, 1],
                widths: vec![0, 1],
            },
        )
        .to_dense_for_purpose(PatchesMaterializationPurpose::Other)
        .expect("sparse-identity materialization");
        anchored_bounds(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 1, 1, 1]), vec![2.0, -3.0])
                .expect("anchored fixture shape"),
            2,
        )
        .to_dense_with_deadline_for_purpose(
            Some(Instant::now()),
            PatchesMaterializationPurpose::Other,
        )
        .expect_err("expired anchored materialization");

        let observed = crate::execution_telemetry::snapshot().patches_materialization;
        assert!(!observed.attribution_conflict);
        assert_eq!(observed.attempts, 4);
        assert_eq!(observed.succeeded, 3);
        assert_eq!(observed.refused, 1);
        assert_eq!(observed.network_input_terminal.succeeded, 1);
        assert_eq!(observed.latent_input_crossover.succeeded, 1);
        assert_eq!(observed.other.succeeded, 1);
        assert_eq!(observed.other.refused, 1);
        assert_eq!(observed.finite_deadline_attempts, 2);
        assert_eq!(observed.no_deadline_attempts, 2);
        assert_eq!(observed.affine_geometry_attempts, 3);
        assert_eq!(observed.anchored_geometry_attempts, 1);
        assert_eq!(observed.input_coefficient_error_attempts, 1);
        assert_eq!(observed.coefficient_error_absent, 2);
        assert_eq!(observed.coefficient_error_materialized, 1);
        assert_eq!(observed.deadline_refusals, 1);
        assert_eq!(observed.memory_receipt_outcomes, 3);
        assert_eq!(
            observed
                .nominal_required_bytes
                .checked_add(observed.capacity_overage_bytes),
            Some(observed.admitted_bytes)
        );
        assert!(observed.admitted_bytes <= observed.budget_bytes);
    }

    #[test]
    fn expired_and_forced_mid_pipeline_deadlines_publish_no_dense_result() {
        crate::tests::with_env_edits(|_env| {
            let bounds = explicit_row_bounds(PatchesMaterializationDeadline::CHECK_STRIDE + 1);
            let expected = bounds.clone();

            let expired = bounds
                .to_dense_with_deadline(Some(Instant::now()))
                .expect_err("expired deadline must refuse before materialization");
            assert!(matches!(expired, NyError::DeadlineExceeded(_)));
            assert_patches_exact(&bounds, &expected);

            let mut mid_scatter = PatchesMaterializationDeadline::forced_at(
                "during patches explicit-row dense scatter",
            );
            let error = bounds
                .to_dense_with_poll(&mut mid_scatter)
                .expect_err("forced mid-scatter deadline must discard local buffers");
            assert!(matches!(error, NyError::DeadlineExceeded(_)));
            assert_patches_exact(&bounds, &expected);

            let mut post_wrap = PatchesMaterializationDeadline::forced_at("after dense wrapping");
            let error = bounds
                .to_dense_with_poll(&mut post_wrap)
                .expect_err("post-wrap deadline must withhold the completed local result");
            assert!(matches!(error, NyError::DeadlineExceeded(_)));
            assert_patches_exact(&bounds, &expected);
        });
    }

    #[test]
    fn dense_peak_accounting_pins_6d_7d_and_overflow_boundaries() {
        // R=13 source/caller resident, U=11 map, M=100 one dense/error matrix, B=20 bias pair,
        // lower/upper strided scratch 7/9. With no error matrices the
        // completed dense+bias local base (11 + 2*100 + 20) is 231; the total
        // peak including R is 244.
        assert_eq!(
            dense_materialization_peak_bytes(13, 11, 100, 20, 7, 9, None, None),
            244
        );

        // Carried 6D: lower phase = 231 + 2M + max(7,M) = 531.
        // Its completed error remains live, so upper = 231 + M + 2M +
        // max(9,M) = 631 locally / 644 with R. The extra retained M is the
        // central two-side proof.
        assert_eq!(
            dense_materialization_peak_bytes(
                13,
                11,
                100,
                20,
                7,
                9,
                Some(SideErrorAllocation::Dense6),
                Some(SideErrorAllocation::Dense6),
            ),
            644
        );

        // Explicit 7D replaces the two f32 accumulators with two f64
        // accumulators (4M): upper = 231 + retained M + 4M + M = 831 locally /
        // 844 with R.
        assert_eq!(
            dense_materialization_peak_bytes(
                13,
                11,
                100,
                20,
                7,
                9,
                Some(SideErrorAllocation::Rows7),
                Some(SideErrorAllocation::Rows7),
            ),
            844
        );

        assert_eq!(
            dense_materialization_peak_bytes(usize::MAX, 11, 100, 20, 7, 9, None, None),
            usize::MAX,
            "overflow must saturate to a fail-closed required byte count"
        );
        assert_eq!(matrix_bytes::<f32>(usize::MAX, 2), usize::MAX);
    }

    #[test]
    fn dense_admission_accepts_exact_budget_and_refuses_one_byte_over() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "1");
            let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
            assert!(budget > 0);
            assert!(PatchesMemoryAdmission::check(budget - 1, "boundary test").is_ok());
            assert!(PatchesMemoryAdmission::check(budget, "boundary test").is_ok());
            assert!(matches!(
                PatchesMemoryAdmission::check(budget + 1, "boundary test"),
                Err(NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes,
                    site: "boundary test",
                }) if required_bytes == budget + 1 && budget_bytes == budget
            ));
        });
    }

    #[test]
    fn chunked_receipt_charges_resident_outputs_and_all_nested_concretize_buffers() {
        let rows = 3usize;
        let in_dim = 5usize;
        let source_bytes = 13usize;
        let full_output_pair_bytes = 17usize;
        let nested = chunked_concretization_nested_bytes(rows, in_dim);
        assert_eq!(
            nested,
            allocation_bytes::<f32>(in_dim)
                .saturating_mul(2)
                .saturating_add(allocation_bytes::<f64>(rows).saturating_mul(2))
                .saturating_add(allocation_bytes::<f32>(rows).saturating_mul(2))
                .saturating_add(allocation_bytes::<usize>(4))
        );

        // The block receipt sees its source and already-live full output pair,
        // plus both flattened inputs, f64 endpoints, returned f32 endpoints,
        // and the complete dense-block peak (including its bias pair).
        let resident = source_bytes
            .saturating_add(full_output_pair_bytes)
            .saturating_add(nested);
        let required = dense_materialization_peak_bytes(
            resident,
            11,
            100,
            20,
            7,
            9,
            Some(SideErrorAllocation::Rows7),
            Some(SideErrorAllocation::Rows7),
        );
        assert!(PatchesMemoryAdmission::check_with_budget(
            required,
            required,
            "chunked nested exact receipt"
        )
        .is_ok());
        assert!(matches!(
            PatchesMemoryAdmission::check_with_budget(
                required,
                required - 1,
                "chunked nested budget-minus-one receipt",
            ),
            Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "chunked nested budget-minus-one receipt",
            }) if required_bytes == required && budget_bytes == required - 1
        ));
    }

    #[test]
    fn chunked_blocks_charge_source_and_already_live_output_pair() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "1");
            let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
            assert!(budget >= 22, "fixture needs a nontrivial dense budget");
            let dim = budget / 22;
            let bounds = PatchesLinearBounds::identity((1, 1, dim), (1, 1, dim));
            let source_bytes = bounds.memory_bytes();
            let output_pair_bytes = allocation_bytes::<f32>(dim).saturating_mul(2);
            let one_row_local_bytes = matrix_bytes::<f32>(1, dim)
                .saturating_mul(2)
                .saturating_add(2 * size_of::<f32>());
            assert!(source_bytes.saturating_add(output_pair_bytes) <= budget);
            assert!(one_row_local_bytes <= budget);
            assert!(
                source_bytes
                    .saturating_add(output_pair_bytes)
                    .saturating_add(one_row_local_bytes)
                    > budget,
                "the cumulative live peak, not any isolated allocation, must refuse"
            );

            let input = BoundedTensor::new(
                ArrayD::zeros(IxDyn(&[1, 1, dim])),
                ArrayD::ones(IxDyn(&[1, 1, dim])),
            )
            .unwrap();
            assert!(matches!(
                bounds.concretize_sound_chunked(&input, budget, None),
                Err(NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes,
                    site: "patches identity to_dense",
                }) if required_bytes > budget_bytes && budget_bytes == budget
            ));
        });
    }

    #[test]
    fn anchored_to_dense_supports_6d_and_7d_exact_maps() {
        let patches_6d =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 1, 1, 1]), vec![2.0, 5.0]).unwrap();
        let dense_6d = anchored_bounds(patches_6d, 2).to_dense().unwrap();
        assert_eq!(
            dense_6d.lower_a(),
            &Array2::from_shape_vec((2, 4), vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]).unwrap()
        );
        assert_eq!(dense_6d.upper_a(), dense_6d.lower_a());

        let patches_7d =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2, 1, 1, 1]), vec![2.0, 5.0]).unwrap();
        let dense_7d = anchored_bounds(patches_7d, 1).to_dense().unwrap();
        assert_eq!(
            dense_7d.lower_a(),
            &Array2::from_shape_vec((1, 4), vec![2.0, 0.0, 0.0, 5.0]).unwrap()
        );
        assert_eq!(dense_7d.upper_a(), dense_7d.lower_a());
    }

    #[test]
    fn explicit_rows_scatter_certifies_modeled_ftz_daz_for_both_signs_and_cancellation() {
        fn flush_f32(value: f32) -> f32 {
            let bits = value.to_bits();
            if bits & 0x7f80_0000 == 0 {
                f32::from_bits(bits & 0x8000_0000)
            } else {
                value
            }
        }

        fn modeled_ftz_daz_sum(taps: [f32; 2]) -> f32 {
            let mut acc = 0.0f32;
            for tap in taps {
                acc = flush_f32(flush_f32(acc) + flush_f32(tap));
            }
            acc
        }

        let min_subnormal = f32::from_bits(1);
        let max_subnormal = f32::from_bits(f32::MIN_POSITIVE.to_bits() - 1);
        let cases = [
            ("positive DAZ source", [min_subnormal, 0.0]),
            ("negative DAZ source", [-min_subnormal, -0.0]),
            (
                "positive cancellation result",
                [f32::MIN_POSITIVE, -max_subnormal],
            ),
            (
                "negative cancellation result",
                [-f32::MIN_POSITIVE, max_subnormal],
            ),
        ];

        for (case, taps) in cases {
            let data = PatchesData {
                coeff_err: None,
                patches: Some(
                    ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2, 1, 1, 1]), taps.to_vec()).unwrap(),
                ),
                // Both output positions intentionally address the same input
                // cell, forcing the explicit-row raw f32 += path.
                geometry: PatchGeometry::anchored(vec![0], vec![0, 0]).unwrap(),
                identity: false,
                output_shape: (1, 1, 2),
                input_shape: (1, 1, 1),
                unstable_idx: None,
            };
            let bounds = PatchesLinearBounds {
                row_count: 1,
                lower_a: data.clone(),
                lower_b: Array1::zeros(1),
                upper_a: data,
                upper_b: Array1::zeros(1),
            };
            let dense = bounds.to_dense().unwrap();
            let center = dense.lower_a()[[0, 0]];
            let center_magnitude = center.to_bits() & 0x7fff_ffff;
            assert!(
                center_magnitude == 0 || center_magnitude >= f32::MIN_POSITIVE.to_bits(),
                "{case}: published center must be zero or normal, got {center:e}"
            );
            assert_eq!(dense.upper_a()[[0, 0]].to_bits(), center.to_bits());

            let err = dense.lower_a_err().expect("7D intrinsic certificate")[[0, 0]];
            let err_magnitude = err.to_bits() & 0x7fff_ffff;
            assert!(
                err == f32::INFINITY || err_magnitude >= f32::MIN_POSITIVE.to_bits(),
                "{case}: positive error must be normal or infinity, got {err:e}"
            );
            assert_eq!(
                dense.upper_a_err().expect("7D intrinsic certificate")[[0, 0]].to_bits(),
                err.to_bits()
            );

            let exact_sum = taps.into_iter().map(ny_core::f32_to_f64_exact).sum::<f64>();
            let modeled = ny_core::f32_to_f64_exact(modeled_ftz_daz_sum(taps));
            let published = ny_core::f32_to_f64_exact(center);
            let certified = ny_core::f32_to_f64_exact(err);
            assert!(
                certified >= (exact_sum - published).abs(),
                "{case}: error excludes exact stored-tap sum"
            );
            assert!(
                certified >= (modeled - published).abs(),
                "{case}: error excludes modeled FTZ/DAZ accumulation"
            );
        }

        let tiny = ny_core::f32_to_f64_exact(min_subnormal);
        assert_eq!(
            publish_error_up_normal(tiny).to_bits(),
            f32::MIN_POSITIVE.to_bits(),
            "subnormal certificate terms must publish without an f64->f32 subnormal cast"
        );
        assert!(nonnegative_f32_error_or_infinity(-min_subnormal).is_infinite());
    }

    #[test]
    fn six_dimensional_subnormal_centers_are_zeroed_and_certified_for_both_signs() {
        let tiny = f32::from_bits(1);
        for value in [tiny, -tiny] {
            let data = PatchesData {
                coeff_err: None,
                patches: Some(
                    ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 1]), vec![value]).unwrap(),
                ),
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: false,
                output_shape: (1, 1, 1),
                input_shape: (1, 1, 1),
                unstable_idx: None,
            };
            let bounds = PatchesLinearBounds {
                row_count: 1,
                lower_a: data.clone(),
                lower_b: Array1::zeros(1),
                upper_a: data,
                upper_b: Array1::zeros(1),
            };
            let dense = bounds.to_dense().unwrap();
            let center = dense.lower_a()[[0, 0]];
            assert_eq!(center.to_bits(), value.to_bits() & 0x8000_0000);
            let error = dense.lower_a_err().expect("subnormal center certificate")[[0, 0]];
            assert!(error.is_finite() && error >= f32::MIN_POSITIVE);
            assert!(ny_core::f32_to_f64_exact(error) >= ny_core::f32_to_f64_exact(value).abs());
            assert_eq!(dense.upper_a()[[0, 0]].to_bits(), center.to_bits());
            assert_eq!(
                dense.upper_a_err().expect("subnormal center certificate")[[0, 0]].to_bits(),
                error.to_bits()
            );
        }
    }

    #[test]
    fn anchored_full_dense_budget_refuses_while_row_chunk_stays_available() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "1");
            let out_w = 257usize;
            let in_w = 513usize;
            let columns = (0..out_w)
                .map(|column| i128::try_from(column * 2).unwrap())
                .collect();
            let data = PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_elem(IxDyn(&[1, 1, out_w, 1, 1, 1]), 1.0)),
                geometry: PatchGeometry::anchored(vec![0], columns).unwrap(),
                identity: false,
                output_shape: (1, 1, out_w),
                input_shape: (1, 1, in_w),
                unstable_idx: None,
            };
            let bounds = PatchesLinearBounds {
                row_count: out_w,
                lower_a: data.clone(),
                lower_b: Array1::zeros(out_w),
                upper_a: data,
                upper_b: Array1::zeros(out_w),
            };
            let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
            assert!(
                bounds.memory_bytes() < budget,
                "fixture's carried Patches relation must itself fit below the budget"
            );

            let error = bounds
                .to_dense()
                .expect_err("the full lower/upper dense pair must exceed one MiB");
            match error {
                NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes,
                    site,
                } => {
                    assert!(required_bytes > budget_bytes);
                    assert_eq!(site, "patches full dense materialization");
                }
                other => panic!("expected typed dense-budget refusal, got {other:?}"),
            }

            let first_row = bounds
                .to_dense_rows(0, 1)
                .expect("row-range materialization remains the bounded fallback");
            assert_eq!(first_row.lower_a().shape(), &[1, in_w]);
            assert_eq!(first_row.lower_a()[[0, 0]].to_bits(), 1.0f32.to_bits());
            assert_eq!(first_row.upper_a(), first_row.lower_a());
        });
    }

    #[test]
    fn anchored_to_dense_rejects_overflowing_metadata_before_allocation() {
        let data = PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[1, 1, 1, 1, 1, 1]), 1.0)),
            geometry: PatchGeometry::anchored(vec![0], vec![0]).unwrap(),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (usize::MAX, 2, 1),
            unstable_idx: None,
        };
        let bounds = PatchesLinearBounds {
            row_count: 1,
            lower_a: data.clone(),
            lower_b: Array1::zeros(1),
            upper_a: data,
            upper_b: Array1::zeros(1),
        };
        assert!(matches!(bounds.to_dense(), Err(NyError::InvalidSpec(_))));
    }

    #[test]
    fn noncontiguous_anchored_scratch_preserves_exact_bits() {
        let standard =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2, 1, 1]), vec![1.0_f32, 2.0, 3.0, 4.0])
                .unwrap();
        let strided = standard.permuted_axes(IxDyn(&[0, 1, 3, 2, 4, 5]));
        assert!(strided.as_slice().is_none(), "fixture must force scratch");
        let data = PatchesData {
            coeff_err: None,
            patches: Some(strided),
            geometry: PatchGeometry::anchored(vec![0], vec![0, 1]).unwrap(),
            identity: false,
            output_shape: (1, 1, 2),
            input_shape: (2, 1, 2),
            unstable_idx: None,
        };
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: data.clone(),
            lower_b: Array1::from_vec(vec![0.25, -0.5]),
            upper_a: data,
            upper_b: Array1::from_vec(vec![0.75, 1.5]),
        };
        let dense = bounds.to_dense().unwrap();
        let expected =
            Array2::from_shape_vec((2, 4), vec![1.0_f32, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 4.0])
                .unwrap();
        assert!(dense
            .lower_a()
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        assert!(dense
            .upper_a()
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        assert_eq!(dense.lower_b(), &Array1::from_vec(vec![0.25, -0.5]));
        assert_eq!(dense.upper_b(), &Array1::from_vec(vec![0.75, 1.5]));
    }

    #[test]
    fn anchored_sparse_4d_and_5d_are_typed_refusals() {
        for shape in [&[1, 1, 1, 1][..], &[1, 1, 1, 1, 1][..]] {
            let data = PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::zeros(IxDyn(shape))),
                geometry: PatchGeometry::anchored(vec![0], vec![0, 3]).unwrap(),
                identity: false,
                output_shape: (1, 1, 2),
                input_shape: (1, 1, 4),
                unstable_idx: Some(UnstableIdx {
                    channels: vec![0],
                    heights: vec![0],
                    widths: vec![0],
                }),
            };
            let bounds = PatchesLinearBounds {
                row_count: 1,
                lower_a: data.clone(),
                lower_b: Array1::zeros(1),
                upper_a: data,
                upper_b: Array1::zeros(1),
            };
            assert!(matches!(
                bounds.to_dense(),
                Err(NyError::UnsupportedConfiguration(_))
            ));

            let lo = ArrayD::zeros(IxDyn(&[1, 1, 4]));
            let hi = ArrayD::ones(IxDyn(&[1, 1, 4]));
            let input = BoundedTensor::new(lo, hi).unwrap();
            assert!(matches!(
                bounds.concretize_sound_sparse(&input, None),
                Err(NyError::UnsupportedConfiguration(_))
            ));
        }
    }

    #[test]
    fn virtual_identity_emissions_refuse_anchored_geometry() {
        let anchored = PatchGeometry::anchored(vec![0], vec![0, 1]).unwrap();
        let mut dense_identity = PatchesLinearBounds::identity((1, 1, 2), (1, 1, 2));
        dense_identity.lower_a.geometry = anchored.clone();
        dense_identity.upper_a.geometry = anchored.clone();
        assert!(matches!(
            dense_identity.to_dense(),
            Err(NyError::UnsupportedConfiguration(_))
        ));
        let lo = ArrayD::zeros(IxDyn(&[1, 1, 2]));
        let hi = ArrayD::ones(IxDyn(&[1, 1, 2]));
        let input = BoundedTensor::new(lo, hi).unwrap();
        assert!(matches!(
            dense_identity.concretize_sound_sparse(&input, None),
            Err(NyError::UnsupportedConfiguration(_))
        ));

        let idx = UnstableIdx {
            channels: vec![0],
            heights: vec![0],
            widths: vec![1],
        };
        let mut sparse_identity = PatchesLinearBounds::sparse_identity((1, 1, 2), (1, 1, 2), idx);
        sparse_identity.lower_a.geometry = anchored.clone();
        sparse_identity.upper_a.geometry = anchored;
        assert!(matches!(
            sparse_identity.to_dense(),
            Err(NyError::UnsupportedConfiguration(_))
        ));
        assert!(matches!(
            sparse_identity.concretize_sound_sparse(&input, None),
            Err(NyError::UnsupportedConfiguration(_))
        ));
    }
}
