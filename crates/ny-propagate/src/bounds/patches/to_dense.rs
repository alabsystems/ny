// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{s, Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;

use super::scatter::{
    compute_unfold_index_map, scatter_err_accumulators, scatter_rows_err_accumulators,
    scatter_rows_with_unfold_map, scatter_sparse_rows_with_unfold_map,
    scatter_sparse_with_unfold_map, scatter_with_unfold_map, validate_patches_shape,
};
use super::{PatchesLinearBounds, UnstableIdx};

/// Build the overlap-aware dense certified-error matrix for one **6D** side from
/// the carried per-row `coeff_err` (#patches-coeff-err-soundness). For each dense
/// cell `(i,j)` that receives `count` patch taps of row `i` (whose absolute
/// scattered magnitudes sum to `absacc`):
///   `err[i,j] = next_up( count·err_row[i] + γ_count^f32·absacc )`.
/// First term over-bounds the sum of the `count` carried coefficient deviations
/// (each ≤ `err_row[i]`); second the f32 `+=` accumulation rounding. `next_up` ⇒
/// outward. Uses the SAME unfold geometry as the coefficient scatter, so it is
/// overlap-exact. Returns a `(row_end - row_start) × in_dim` matrix (0 where no
/// tap lands): rows `[row_start, row_end)` of the full grid (#patches-row-range),
/// with `err_row` still indexed by the GLOBAL row.
///
/// 6D variant (f32 accumulators; per-cell tap count is 0/1, so f32 suffices and
/// stays byte-identical to the certified 6D design). The 7D explicit-rows layout
/// goes through [`patches_err_matrix_rows`] instead.
#[allow(clippy::too_many_arguments)]
fn patches_err_matrix(
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    err_row: &Array1<f32>,
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
    let mut count = Array2::<f32>::zeros((n_rows, in_dim));
    let mut absacc = Array2::<f32>::zeros((n_rows, in_dim));
    scatter_err_accumulators(
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
    );
    let mut err = Array2::<f32>::zeros((n_rows, in_dim));
    for (local, i) in (row_start..row_end).enumerate() {
        let er = f64::from(err_row.get(i).copied().unwrap_or(0.0)).max(0.0);
        for j in 0..in_dim {
            let c = count[[local, j]];
            if c > 0.0 {
                let gamma = crate::layers::linear::crown_single_gamma_n_f32(c as usize);
                let term = c as f64 * er + gamma * f64::from(absacc[[local, j]]);
                err[[local, j]] = ny_tensor::next_up_f32(term as f32);
            }
        }
    }
    err
}

/// Build the overlap-aware dense certified-error matrix for one **7D
/// explicit-rows** side (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §3). The err
/// index is the SPEC row (axis 0, length `row_count`, == the bias length). For
/// each dense cell `(r,j)` that receives `count` plan taps of spec row `r`
/// (absolute scattered magnitudes f64-summed into `absacc`):
///   `err[r,j] = next_up( count·err_row[r] + γ_count^f32·absacc )`  (0 if count 0).
///
/// Emitted even for `err_row == None` (`e_r = 0`): unlike the 6D layout the 7D
/// scatter genuinely accumulates multiple taps per dense cell, so the
/// `γ(N)·S_hat` accumulation-rounding term exists on every side (spec R2).
/// Accumulators are f64 (spec R1; see [`scatter_rows_err_accumulators`]).
///
/// Non-finite/negative carried err poisons the row to `+INF` (outward degrade,
/// spec I5) — NEVER the 6D `NaN -> 0` `.max(0.0)` false-proof hazard (R3). The
/// zero-magnitude accumulator is short-circuited before multiplying by the
/// possibly-infinite `γ` so `INF·0` can never produce NaN.
///
/// Row range (#patches-row-range): emits rows `[row_start, row_end)` of the
/// spec-row axis (a `(row_end - row_start) × in_dim` matrix); `err_row` is
/// still indexed by the GLOBAL spec row.
///
/// Infallible: the `err_row` length check (== `row_count`, spec I6) lives at
/// the call site in `materialize_dense_patches_to_dense`.
#[allow(clippy::too_many_arguments)]
pub(super) fn patches_err_matrix_rows(
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
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
    let mut count = Array2::<f64>::zeros((n_rows, in_dim));
    let mut absacc = Array2::<f64>::zeros((n_rows, in_dim));
    scatter_rows_err_accumulators(
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
    );
    let mut err = Array2::<f32>::zeros((n_rows, in_dim));
    for (local, r) in (row_start..row_end).enumerate() {
        // Sanitize per row (spec I5): non-finite or negative carried err maps
        // to +INF (poisons outward); None means the side is exact (0).
        let er = match err_row {
            None => 0.0f64,
            Some(e) => {
                let v = e[r]; // length == row_count checked at the call site (I6)
                if v.is_finite() && v >= 0.0 {
                    f64::from(v)
                } else {
                    f64::INFINITY
                }
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
                // c·er ∈ {finite ≥ 0, +INF} (c > 0, er sanitized ≥ 0), acc
                // likewise, so the sum is never NaN; +INF stays +INF through
                // the cast and next_up (degrade poison).
                let term = c * er + acc;
                err[[local, j]] = ny_tensor::next_up_f32(term as f32);
            }
        }
    }
    err
}

#[cfg(test)]
use super::record_patches_to_dense_call_site;

impl PatchesLinearBounds {
    /// Convert to dense LinearBounds by materializing the full A-matrix.
    ///
    /// This is the fallback for layers that don't natively support Patches.
    /// For a Patches A with shape (out_c, out_h, out_w, in_c, kH, kW),
    /// stride/padding, and input_shape (in_c, in_h, in_w):
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
    /// Uses `inplace_unfold` (im2col) from ny-tensor to precompute the
    /// input index mapping once, shared between lower_a and upper_a. This
    /// avoids redundant position computation and simplifies the scatter loop
    /// by eliminating inline bounds checking.
    ///
    /// Reference: alpha-beta-CROWN `auto_LiRPA/patches.py` (Patches.to_matrix)
    #[track_caller]
    pub(crate) fn to_dense(&self) -> Result<LinearBounds> {
        #[cfg(test)]
        {
            let location = std::panic::Location::caller();
            record_patches_to_dense_call_site(format!("{}:{}", location.file(), location.line()));
        }
        let total = self.dense_rows_total()?;
        self.to_dense_rows_impl(0, total)
    }

    /// Materialize exactly rows `[row_start, row_end)` of the full dense form
    /// (#patches-row-range): the A rows, the bias slice, and the certified
    /// coeff-err rows are the bit-identical `[row_start, row_end)` slice of
    /// what [`to_dense`](Self::to_dense) builds — every scatter/err/bias write
    /// is row-local, and per-row accumulation order is preserved by the range
    /// kernels. `to_dense()` is exactly `to_dense_rows(0, total)`: ONE code
    /// path, so the full-range behavior stays byte-identical.
    pub(crate) fn to_dense_rows(&self, row_start: usize, row_end: usize) -> Result<LinearBounds> {
        let total = self.dense_rows_total()?;
        if row_start > row_end || row_end > total {
            return Err(NyError::InvalidSpec(format!(
                "patches to_dense_rows: invalid row range [{row_start}, {row_end}) \
                 for {total} dense rows"
            )));
        }
        self.to_dense_rows_impl(row_start, row_end)
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
    fn to_dense_rows_impl(&self, row_start: usize, row_end: usize) -> Result<LinearBounds> {
        self.validate_row_count()?;
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
            return self.sparse_to_dense(row_start, row_end);
        }
        if self.lower_a.identity && self.upper_a.identity {
            return self.identity_to_dense(out_dim, in_dim, row_start, row_end);
        }

        self.materialize_dense_patches_to_dense(
            out_c, out_h, out_w, in_c, in_dim, row_start, row_end,
        )
    }

    fn identity_to_dense(
        &self,
        out_dim: usize,
        in_dim: usize,
        row_start: usize,
        row_end: usize,
    ) -> Result<LinearBounds> {
        // Identity patches must be exact: every identity constructor sets
        // coeff_err None (patches.rs identity/sparse_identity,
        // types.rs materialize_identity), and this path emits no err matrix,
        // so a carried Some here would be silently dropped (unsound).
        debug_assert!(
            self.lower_a.coeff_err.is_none() && self.upper_a.coeff_err.is_none(),
            "identity patches must be exact (coeff_err None on both sides)"
        );
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
        // Rows [row_start, row_end) of the identity: a single 1.0 at column
        // `row` (full range reproduces `Array2::eye(out_dim)` exactly).
        let mut eye = Array2::<f32>::zeros((row_end - row_start, in_dim));
        for row in row_start..row_end {
            eye[[row - row_start, row]] = 1.0;
        }
        LinearBounds::new_or_conservative(
            eye.clone(),
            self.lower_b.slice(s![row_start..row_end]).to_owned(),
            eye,
            self.upper_b.slice(s![row_start..row_end]).to_owned(),
        )
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
    ) -> Result<LinearBounds> {
        let (lower_patches, lower_explicit_rows) =
            validate_patches_shape(&self.lower_a, self.row_count, out_c, out_h, out_w, in_c)?;
        let lower_shape = lower_patches.shape();
        let (kh, kw) = if lower_explicit_rows {
            (lower_shape[5], lower_shape[6])
        } else {
            (lower_shape[4], lower_shape[5])
        };
        let index_map = compute_unfold_index_map(&self.lower_a, kh, kw)?;

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
        if (ukh, ukw) != (kh, kw)
            || self.upper_a.stride != self.lower_a.stride
            || self.upper_a.padding != self.lower_a.padding
            || self.upper_a.input_shape != self.lower_a.input_shape
        {
            let l = &self.lower_a;
            let u = &self.upper_a;
            return Err(NyError::ShapeMismatch {
                expected: vec![
                    kh,
                    kw,
                    l.stride.0,
                    l.stride.1,
                    l.padding.0,
                    l.padding.1,
                    l.padding.2,
                    l.padding.3,
                    l.input_shape.0,
                    l.input_shape.1,
                    l.input_shape.2,
                ],
                got: vec![
                    ukh,
                    ukw,
                    u.stride.0,
                    u.stride.1,
                    u.padding.0,
                    u.padding.1,
                    u.padding.2,
                    u.padding.3,
                    u.input_shape.0,
                    u.input_shape.1,
                    u.input_shape.2,
                ],
            });
        }

        let n_rows = row_end - row_start;
        let mut lower_dense = Array2::<f32>::zeros((n_rows, in_dim));
        if lower_explicit_rows {
            scatter_rows_with_unfold_map(
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
            );
        } else {
            scatter_with_unfold_map(
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
            );
        }

        let mut upper_dense = Array2::<f32>::zeros((n_rows, in_dim));
        if upper_explicit_rows {
            scatter_rows_with_unfold_map(
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
            );
        } else {
            scatter_with_unfold_map(
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
            );
        }

        let lower_b = self.lower_b.slice(s![row_start..row_end]).to_owned();
        let upper_b = self.upper_b.slice(s![row_start..row_end]).to_owned();

        // Attach the certified coefficient error (#patches-coeff-err-soundness,
        // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §3.1). A side needs a
        // materialized err matrix when it carries a coeff_err (any layout) OR
        // when it is explicit-rows (7D) — INCLUDING the `(None,None)` pair:
        // the 7D scatter genuinely accumulates multiple taps per dense cell,
        // so the `γ(N)·S_hat` accumulation-rounding term exists even at
        // `e_r = 0` (spec R2). The plain `new_or_conservative` fast path
        // remains ONLY for `(None,None)` with both sides 6D (per-cell tap
        // count is provably 0/1 there) — byte-identical to the exact path.
        let need_err = self.lower_a.coeff_err.is_some()
            || self.upper_a.coeff_err.is_some()
            || lower_explicit_rows
            || upper_explicit_rows;
        if !need_err {
            return LinearBounds::new_or_conservative(lower_dense, lower_b, upper_dense, upper_b);
        }
        // Per-side dispatch on that side's OWN layout flag / err (mixed 6D/7D
        // pairs pass validation and each get their own treatment).
        let side_err = |err: Option<&Array1<f32>>,
                        patches: &ArrayD<f32>,
                        explicit_rows: bool|
         -> Result<Array2<f32>> {
            // Hard length check for the 7D arm ONLY (spec I6): `patches_err_matrix_rows`
            // indexes `e[r]` directly, so a `Some` err of the wrong length must route
            // the caller to its sound dense fallback rather than panic. The 6D arm keeps
            // its silent `.get(i).unwrap_or(0.0)` read (in `patches_err_matrix`) so it
            // stays byte-identical to the certified 6D path — matching every producer
            // site, which guards this check behind `explicit_rows` too. (Valid inputs
            // always have `err.len() == row_count`, indexed identically to the bias.)
            if explicit_rows {
                if let Some(er) = err {
                    if er.len() != self.row_count {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![self.row_count],
                            got: vec![er.len()],
                        });
                    }
                }
            }
            Ok(if explicit_rows {
                patches_err_matrix_rows(
                    patches, &index_map, err, row_start, row_end, in_dim, out_c, out_h, out_w,
                    in_c, kh, kw,
                )
            } else {
                match err {
                    Some(er) => patches_err_matrix(
                        patches, &index_map, er, row_start, row_end, in_dim, out_c, out_h, out_w,
                        in_c, kh, kw,
                    ),
                    // A 6D side with no carried err is exact (≤ 1 tap/cell).
                    None => Array2::<f32>::zeros((n_rows, in_dim)),
                }
            })
        };
        let lower_err = side_err(
            self.lower_a.coeff_err.as_ref(),
            lower_patches,
            lower_explicit_rows,
        )?;
        let upper_err = side_err(
            self.upper_a.coeff_err.as_ref(),
            upper_patches,
            upper_explicit_rows,
        )?;
        LinearBounds::new_or_conservative_with_err(
            lower_dense,
            lower_b,
            upper_dense,
            upper_b,
            lower_err,
            upper_err,
        )
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
    fn sparse_to_dense(&self, row_start: usize, row_end: usize) -> Result<LinearBounds> {
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

        let idx = self
            .lower_a
            .unstable_idx
            .as_ref()
            .ok_or_else(|| NyError::InternalError("sparse_to_dense: no unstable_idx".into()))?;

        // One-time crash guard (#hotpath robustness): the sparse layout is turned
        // into unchecked flat row indices (`flat_index`) used to index dense rows,
        // the dense diagonal, and expanded bias vectors. Reject a layout whose
        // parallel index vectors disagree, or whose `(c,h,w)` lands outside the
        // output grid, with a clean error so the caller falls back to dense CROWN
        // instead of panicking. No bound math changes. Run ONCE per call.
        idx.validate(out_c, out_h, out_w, None)?;

        if self.lower_a.identity && self.upper_a.identity {
            // Identity sparse `to_dense` indexes lower_b[i]/upper_b[i] for every
            // sparse row, so the sparse bias length must match the index count.
            idx.validate(out_c, out_h, out_w, Some(self.lower_b.len()))?;
            idx.validate(out_c, out_h, out_w, Some(self.upper_b.len()))?;
            return self
                .sparse_identity_to_dense(idx, out_dim, in_dim, out_h, out_w, row_start, row_end);
        }

        let (lower_dense, upper_dense, explicit_rows) = self
            .materialize_sparse_dense_pair(idx, out_h, out_w, in_c, in_dim, row_start, row_end)?;
        let (lower_b, upper_b) = if explicit_rows {
            (
                self.lower_b.slice(s![row_start..row_end]).to_owned(),
                self.upper_b.slice(s![row_start..row_end]).to_owned(),
            )
        } else {
            // expand_sparse_bias reads sparse_lower[i]/sparse_upper[i] for every
            // sparse index, so the sparse bias vectors must match the index count.
            idx.validate(out_c, out_h, out_w, Some(self.lower_b.len()))?;
            idx.validate(out_c, out_h, out_w, Some(self.upper_b.len()))?;
            Self::expand_sparse_bias(
                &self.lower_b,
                &self.upper_b,
                idx,
                row_start,
                row_end,
                out_h,
                out_w,
            )
        };
        LinearBounds::new_or_conservative(lower_dense, lower_b, upper_dense, upper_b)
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
    ) -> Result<(Array2<f32>, Array2<f32>, bool)> {
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
        let index_map = compute_unfold_index_map(&self.lower_a, kh, kw)?;
        let n_rows = row_end - row_start;
        let mut lower_dense = Array2::<f32>::zeros((n_rows, in_dim));
        let mut upper_dense = Array2::<f32>::zeros((n_rows, in_dim));
        let upper_patches = self.upper_a.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("sparse_to_dense: upper patches tensor is None".into())
        })?;

        if explicit_rows {
            scatter_sparse_rows_with_unfold_map(
                &mut lower_dense,
                lower_patches,
                &index_map,
                row_start,
                row_end,
                idx,
                in_c,
                kh,
                kw,
            );
            scatter_sparse_rows_with_unfold_map(
                &mut upper_dense,
                upper_patches,
                &index_map,
                row_start,
                row_end,
                idx,
                in_c,
                kh,
                kw,
            );
        } else {
            scatter_sparse_with_unfold_map(
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
            );
            scatter_sparse_with_unfold_map(
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
            );
        }

        Ok((lower_dense, upper_dense, explicit_rows))
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
    ) -> Result<LinearBounds> {
        if out_dim != in_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim],
                got: vec![in_dim],
            });
        }
        let n_rows = row_end - row_start;
        let mut lower_a = Array2::<f32>::zeros((n_rows, in_dim));
        let mut upper_a = Array2::<f32>::zeros((n_rows, in_dim));
        let mut lower_b = Array1::<f32>::zeros(n_rows);
        let mut upper_b = Array1::<f32>::zeros(n_rows);
        for i in 0..idx.len() {
            let flat = idx.flat_index(i, out_h, out_w);
            if flat < row_start || flat >= row_end {
                continue;
            }
            lower_a[[flat - row_start, flat]] = 1.0;
            upper_a[[flat - row_start, flat]] = 1.0;
            lower_b[flat - row_start] = self.lower_b[i];
            upper_b[flat - row_start] = self.upper_b[i];
        }
        LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
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
    ) -> (Array1<f32>, Array1<f32>) {
        let n_rows = row_end - row_start;
        let mut lower_b = Array1::<f32>::zeros(n_rows);
        let mut upper_b = Array1::<f32>::zeros(n_rows);
        for i in 0..idx.len() {
            let flat = idx.flat_index(i, out_h, out_w);
            if flat < row_start || flat >= row_end {
                continue;
            }
            lower_b[flat - row_start] = sparse_lower[i];
            upper_b[flat - row_start] = sparse_upper[i];
        }
        (lower_b, upper_b)
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
    /// VGG16 final-densify abort (3.2M × 150K rows ≈ 1.9 TB per matrix). The
    /// per-row budget accounts for the lower/upper A pair, the lower/upper err
    /// pair, and the transient f64 err accumulators (count + absacc); at least
    /// one row per block is always materialized.
    ///
    /// `deadline` is checked between blocks: on expiry the caller receives
    /// `DeadlineExceeded` (never a partial result) and falls back soundly,
    /// matching the per-node budget handling of the CROWN backward walk.
    pub(crate) fn concretize_sound_chunked(
        &self,
        input: &BoundedTensor,
        max_block_bytes: usize,
        deadline: Option<std::time::Instant>,
    ) -> Result<BoundedTensor> {
        let total = self.dense_rows_total()?;
        let (in_c, in_h, in_w) = self.lower_a.input_shape;
        let in_dim = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "patches chunked concretize: input dims overflow: {in_c} * {in_h} * {in_w}"
            ))
        })?;
        // Peak per-row block footprint in f32 cells: lower/upper A (2) +
        // lower/upper certified err (2) + transient f64 err accumulators
        // (count + absacc = 4 f32-equivalents) = 8 cells of 4 bytes per input
        // column. Conservative for exact 6D relations (no err matrices), which
        // simply get smaller blocks than strictly necessary.
        let per_row_bytes = in_dim
            .saturating_mul(8)
            .saturating_mul(size_of::<f32>())
            .max(1);
        let rows_per_block = (max_block_bytes / per_row_bytes).max(1);

        let mut out_lower = vec![0.0_f32; total];
        let mut out_upper = vec![0.0_f32; total];
        let mut r0 = 0usize;
        while r0 < total {
            if let Some(d) = deadline {
                if std::time::Instant::now() >= d {
                    return Err(NyError::DeadlineExceeded(format!(
                        "patches chunked concretize: deadline exceeded at row {r0} of {total}"
                    )));
                }
            }
            let r1 = (r0 + rows_per_block).min(total);
            let block = self.to_dense_rows(r0, r1)?;
            let concrete = block.concretize_sound(input);
            let lo = concrete.lower();
            let lo = lo.as_slice().ok_or_else(|| {
                NyError::InvalidSpec("patches chunked concretize: lower not contiguous".to_string())
            })?;
            let up = concrete.upper();
            let up = up.as_slice().ok_or_else(|| {
                NyError::InvalidSpec("patches chunked concretize: upper not contiguous".to_string())
            })?;
            out_lower[r0..r1].copy_from_slice(lo);
            out_upper[r0..r1].copy_from_slice(up);
            r0 = r1;
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&[total]), out_lower)
            .map_err(|e| NyError::InvalidSpec(format!("patches chunked concretize lower: {e}")))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&[total]), out_upper)
            .map_err(|e| NyError::InvalidSpec(format!("patches chunked concretize upper: {e}")))?;
        // ±inf rows are legal here (concretize_sound emits the sound
        // [-inf, +inf] degrade for repaired rows), matching the single-shot
        // result exactly.
        BoundedTensor::new_allow_infinite(lower, upper)
    }
}
