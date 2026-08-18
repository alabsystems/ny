// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-native sparse concretization (#patches-sparse-concretize).
//!
//! The blockwise dense fallback
//! ([`PatchesLinearBounds::concretize_sound_chunked`](super::PatchesLinearBounds::concretize_sound_chunked))
//! bounds *memory* by materializing the `[rows × in_dim]` dense A-pair in row
//! blocks, but every block still runs `LinearBounds::concretize_sound`, whose
//! inner dot product traverses **all** `in_dim` input columns per row. On VGG16
//! conv targets that is `3.2M rows × 150528 cols ≈ 4.8e11` element visits — a
//! single-threaded timeout — even though each output neuron only depends on the
//! ≤ `in_c·kH·kW` input pixels in its receptive field.
//!
//! This module concretizes each output row **directly from the patches taps**,
//! visiting only its receptive-field columns (for a conv1 target, 27 taps
//! instead of 150528 columns — a ~5500× per-row cut, and far more for the
//! whole-image dense loop it replaces).
//!
//! ## Why it is EXACT (bit-identical to the dense path)
//!
//! For output row `i`, the dense concretize computes, over `j = 0..in_dim`, an
//! f64 running sum of `safe_mul(la_pos,in_l)+safe_mul(la_neg,in_u)` (plus the
//! certified err penalty). A column `j` outside neuron `i`'s receptive field has
//! coefficient **exactly 0** — the patches scatter never writes it — so its dense
//! contribution `safe_mul(0,·)+safe_mul(0,·)` and its err term `0·mag` are
//! exactly `0.0`, an f64 no-op add, and it can never trip the `CROWN_COEFF_MAX`
//! overflow guard. Hence visiting only the receptive-field columns **in
//! increasing global column index order** reproduces the dense running sum at
//! every step. The per-tap coefficient is the SAME f32 value the dense scatter
//! writes (each 6D/4D output position touches each input pixel at most once, so
//! there is no overlap accumulation to reorder), and the err value is the SAME
//! `next_up(count·err_row + γ_count·|coeff|)` that
//! [`patches_err_matrix`](super::to_dense) materializes (with `count = 1`). The
//! shared certified kernel
//! [`concretize_row_directed`](crate::bounds::concretize_row_directed) then
//! applies the identical directed cast + repair. The tap order emitted by the
//! `UnfoldPlan` is `(ic, ki, kj)`-lexicographic, which is strictly increasing in
//! the input flat index `ic·in_h·in_w + ih·in_w + iw` for either a fixed affine
//! or anchored window origin, so no sort is needed. Result: **bit-for-bit** equal to
//! `self.to_dense()?.concretize_sound(input)`, pinned by
//! `sparse_concretize_matches_dense_bit_identical`.
//!
//! The one documented deviation matches
//! [`concretize_sound_chunked`](super::PatchesLinearBounds::concretize_sound_chunked)'s
//! own caveat: the whole-matrix NaN/Inf firewall
//! (`LinearBounds::new_or_conservative`) degrades ALL rows to `[-inf, +inf]` on a
//! single non-finite coefficient, whereas the per-row kernel degrades only the
//! offending row (still the sound `[-inf, +inf]`, and every other row keeps its
//! exact value). Non-finite coefficients require f32 overflow inside the scatter
//! sum and never arise for well-formed relaxations; when they do the per-row
//! result is strictly sound and no wider than necessary.
//!
//! ## Scope
//!
//! Fast-pathed layouts: affine or anchored 6D dense patches (with or without a
//! carried per-row `coeff_err`), affine 4D sparse patches (exact by scope
//! guard), dense identity, and affine sparse identity. Anchored sparse layouts
//! are a typed refusal until their coefficient-error route exists. The 7D / 5D
//! explicit-rows layouts (whose per-cell tap counts
//! overlap and whose err accumulators are f64) return
//! [`NyError::UnsupportedOp`](ny_core::NyError::UnsupportedOp) so the caller falls
//! back to the certified dense-chunked path — sound, just not sped up.

use std::{mem::size_of, time::Instant};

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::concretize_row_directed;
use crate::layers::linear::crown_single_gamma_n_f32;

use super::scatter::{
    build_unfold_plan, compute_unfold_index_map_with_deadline, try_as_flat_with_deadline,
    validate_patches_shape,
};
use super::to_dense::{
    f32_abs_exact, nonnegative_f32_error_or_infinity, publish_error_up_normal, try_filled_f32_vec,
};
use super::{PatchesLinearBounds, PatchesMaterializationDeadline, PatchesMemoryAdmission};

fn try_reserve_row_workspace(
    vectors: &mut [&mut Vec<f32>],
    elements_each: usize,
    admission: &mut PatchesMemoryAdmission,
    site: &'static str,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<()> {
    for vector in vectors {
        deadline.checkpoint(site)?;
        vector
            .try_reserve_exact(elements_each)
            .map_err(|_| admission.allocation_error(site))?;
        deadline.checkpoint(site)?;
        admission.reconcile_vec_capacity::<f32>(elements_each, vector.capacity(), site)?;
    }
    Ok(())
}

#[inline]
fn strided_scratch_bytes(array: &ArrayD<f32>) -> usize {
    if array.as_slice().is_some() {
        0
    } else {
        array.len().saturating_mul(size_of::<f32>())
    }
}

fn check_sparse_memory(
    required_bytes: usize,
    site: &'static str,
) -> Result<PatchesMemoryAdmission> {
    PatchesMemoryAdmission::check(required_bytes, site)
}

/// Assemble the flat `[total]` concrete bounds (±inf rows are legal — the
/// per-row kernel emits the sound `[-inf, +inf]` degrade for repaired rows).
fn finalize(
    out_lower: Vec<f32>,
    out_upper: Vec<f32>,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<BoundedTensor> {
    let total = out_lower.len();
    deadline.checkpoint("before patches sparse concretize output wrapping")?;
    let lower = ArrayD::from_shape_vec(IxDyn(&[total]), out_lower)
        .map_err(|e| NyError::InvalidSpec(format!("patches sparse concretize lower: {e}")))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&[total]), out_upper)
        .map_err(|e| NyError::InvalidSpec(format!("patches sparse concretize upper: {e}")))?;
    let bounded = BoundedTensor::new_allow_infinite(lower, upper)?;
    deadline.checkpoint("after patches sparse concretize output wrapping")?;
    Ok(bounded)
}

impl PatchesLinearBounds {
    /// Concretize over `input` visiting only each row's receptive-field taps.
    ///
    /// BIT-IDENTICAL to `self.to_dense()?.concretize_sound(input)` for the
    /// fast-pathed layouts (see the module docs for the exactness argument), at
    /// `O(Σ_rows receptive_field)` instead of `O(rows × in_dim)`.
    ///
    /// Returns [`NyError::UnsupportedOp`] for layouts this fast path does not
    /// cover (7D / 5D explicit-rows, non-conforming geometry), signaling the
    /// caller to fall back to the certified
    /// [`concretize_sound_chunked`](Self::concretize_sound_chunked). Other
    /// errors (`DeadlineExceeded`, malformed layouts) propagate unchanged.
    pub(crate) fn concretize_sound_sparse(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let mut deadline = PatchesMaterializationDeadline::new(deadline);
        deadline.checkpoint("before patches sparse concretization")?;
        let (in_c, in_h, in_w) = self.lower_a.input_shape;
        let in_dim = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "patches sparse concretize: input dims overflow: {in_c} * {in_h} * {in_w}"
            ))
        })?;
        let input_scratch_bytes = strided_scratch_bytes(input.lower())
            .saturating_add(strided_scratch_bytes(input.upper()));
        let source_bytes = self.memory_bytes();
        let mut input_admission = PatchesMemoryAdmission::check(
            source_bytes.saturating_add(input_scratch_bytes),
            "patches sparse concretize input scratch pair",
        )?;
        let mut in_l_scratch = Vec::new();
        let mut in_u_scratch = Vec::new();
        let in_l = try_as_flat_with_deadline(
            input.lower(),
            &mut in_l_scratch,
            "patches sparse concretize lower-input scratch",
            &mut input_admission,
            &mut deadline,
        )?;
        let in_u = try_as_flat_with_deadline(
            input.upper(),
            &mut in_u_scratch,
            "patches sparse concretize upper-input scratch",
            &mut input_admission,
            &mut deadline,
        )?;
        if in_l.len() != in_dim || in_u.len() != in_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![in_l.len().max(in_u.len())],
            });
        }

        // Every paired path below reuses the lower-side map for upper
        // coefficients, so authenticate their typed geometry first.
        self.lower_a
            .validate_common_geometry_with_poll(&self.upper_a, &mut deadline)?;
        deadline.checkpoint("after patches sparse concretize validation")?;

        let input_scratch_bytes =
            input_scratch_bytes.saturating_add(input_admission.capacity_overage_bytes());

        if self.lower_a.unstable_idx.is_some() || self.upper_a.unstable_idx.is_some() {
            self.sparse_concretize_sparse_layout(
                in_l,
                in_u,
                in_dim,
                input_scratch_bytes,
                &mut deadline,
            )
        } else if self.lower_a.identity && self.upper_a.identity {
            self.sparse_concretize_identity(in_l, in_u, in_dim, input_scratch_bytes, &mut deadline)
        } else {
            self.sparse_concretize_dense_6d(in_l, in_u, input_scratch_bytes, &mut deadline)
        }
    }

    /// 6D dense patches (broadcast rows over the output grid), with or without a
    /// carried per-row `coeff_err`.
    fn sparse_concretize_dense_6d(
        &self,
        in_l: &[f32],
        in_u: &[f32],
        input_scratch_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<BoundedTensor> {
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let (in_c, _, _) = self.lower_a.input_shape;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec("patches sparse concretize: output dims overflow".into())
        })?;

        let (lower_patches, lower_explicit) =
            validate_patches_shape(&self.lower_a, self.row_count, out_c, out_h, out_w, in_c)?;
        let (upper_patches, upper_explicit) =
            validate_patches_shape(&self.upper_a, self.row_count, out_c, out_h, out_w, in_c)?;
        // 7D explicit-rows: overlapping per-cell taps + f64 err accumulators.
        // Not fast-pathed — fall back to the certified dense-chunked path.
        if lower_explicit || upper_explicit {
            return Err(NyError::UnsupportedOp(
                "patches sparse concretize: 7D explicit-rows not fast-pathed".into(),
            ));
        }
        // 6D broadcast: exactly one logical row per output position.
        if self.row_count != out_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim],
                got: vec![self.row_count],
            });
        }
        self.validate_row_count()?;

        let lshape = lower_patches.shape();
        let (kh, kw) = (lshape[4], lshape[5]);
        let ushape = upper_patches.shape();
        if (ushape[4], ushape[5]) != (kh, kw) {
            return Err(NyError::UnsupportedOp(
                "patches sparse concretize: lower/upper kernel mismatch".into(),
            ));
        }
        let lower_err = self.lower_a.coeff_err.as_ref();
        let upper_err = self.upper_a.coeff_err.as_ref();
        for err in [lower_err, upper_err].into_iter().flatten() {
            if err.len() != self.row_count {
                return Err(NyError::ShapeMismatch {
                    expected: vec![self.row_count],
                    got: vec![err.len()],
                });
            }
        }
        let resident_before_map = self.memory_bytes().saturating_add(input_scratch_bytes);
        let index_map = compute_unfold_index_map_with_deadline(
            &self.lower_a,
            kh,
            kw,
            resident_before_map,
            deadline,
        )?;
        let plan = build_unfold_plan(&index_map);
        let block = plan.block();

        let need_err = lower_err.is_some() || upper_err.is_some();
        let output_pair_bytes = out_dim.saturating_mul(2).saturating_mul(size_of::<f32>());
        let workspace_vectors = if need_err { 6usize } else { 4usize };
        let workspace_bytes = block
            .saturating_mul(workspace_vectors)
            .saturating_mul(size_of::<f32>());
        let resident_bytes = self
            .memory_bytes()
            .saturating_add(input_scratch_bytes)
            .saturating_add(index_map.memory_bytes())
            .saturating_add(strided_scratch_bytes(lower_patches))
            .saturating_add(strided_scratch_bytes(upper_patches));
        let required_bytes = resident_bytes
            .saturating_add(output_pair_bytes)
            .saturating_add(workspace_bytes);
        let mut admission = check_sparse_memory(
            required_bytes,
            "patches sparse concretize dense-6D materialization",
        )?;

        let mut lscratch = Vec::new();
        let mut uscratch = Vec::new();
        let lower_flat = try_as_flat_with_deadline(
            lower_patches,
            &mut lscratch,
            "patches sparse concretize lower scratch",
            &mut admission,
            deadline,
        )?;
        let upper_flat = try_as_flat_with_deadline(
            upper_patches,
            &mut uscratch,
            "patches sparse concretize upper scratch",
            &mut admission,
            deadline,
        )?;

        // 6D per-cell tap count is provably 1 (each output position touches each
        // input pixel at most once), so the accumulation-rounding term is γ(1)·|coeff|.
        let gamma1 = crown_single_gamma_n_f32(1);

        let positions = plan.positions();
        let mut la_c: Vec<f32> = Vec::new();
        let mut ua_c: Vec<f32> = Vec::new();
        let mut inl_c: Vec<f32> = Vec::new();
        let mut inu_c: Vec<f32> = Vec::new();
        let mut le_c: Vec<f32> = Vec::new();
        let mut ue_c: Vec<f32> = Vec::new();

        let mut out_lower = try_filled_f32_vec(
            out_dim,
            0.0,
            &mut admission,
            "patches sparse concretize dense-6D lower output allocation",
            deadline,
        )?;
        let mut out_upper = try_filled_f32_vec(
            out_dim,
            0.0,
            &mut admission,
            "patches sparse concretize dense-6D upper output allocation",
            deadline,
        )?;
        if need_err {
            let mut workspaces = [
                &mut la_c, &mut ua_c, &mut inl_c, &mut inu_c, &mut le_c, &mut ue_c,
            ];
            try_reserve_row_workspace(
                &mut workspaces,
                block,
                &mut admission,
                "patches sparse concretize dense-6D row workspace",
                deadline,
            )?;
        } else {
            let mut workspaces = [&mut la_c, &mut ua_c, &mut inl_c, &mut inu_c];
            try_reserve_row_workspace(
                &mut workspaces,
                block,
                &mut admission,
                "patches sparse concretize dense-6D row workspace",
                deadline,
            )?;
        }

        for out_flat in 0..out_dim {
            deadline.work(1, "during patches sparse dense-6D row loop")?;
            let pos = out_flat % positions;
            let oh = pos / out_w;
            let ow = pos % out_w;
            let pat_base = out_flat * block;

            la_c.clear();
            ua_c.clear();
            inl_c.clear();
            inu_c.clear();
            le_c.clear();
            ue_c.clear();

            for &(block_offset, in_flat) in plan.taps_for(oh, ow) {
                let lv = lower_flat[pat_base + block_offset];
                let uv = upper_flat[pat_base + block_offset];
                la_c.push(lv);
                ua_c.push(uv);
                inl_c.push(in_l[in_flat]);
                inu_c.push(in_u[in_flat]);
                if need_err {
                    // Mirror patches_err_matrix (count = 1, absacc = |coeff|)
                    // then new_or_conservative_with_err's sanitize. A side with
                    // no carried err materializes a zero err matrix (0.0 here).
                    let le_col = match lower_err {
                        Some(er) => {
                            let erv = nonnegative_f32_error_or_infinity(er[out_flat]);
                            publish_error_up_normal(erv + gamma1 * f32_abs_exact(lv))
                        }
                        None => 0.0,
                    };
                    let ue_col = match upper_err {
                        Some(er) => {
                            let erv = nonnegative_f32_error_or_infinity(er[out_flat]);
                            publish_error_up_normal(erv + gamma1 * f32_abs_exact(uv))
                        }
                        None => 0.0,
                    };
                    le_c.push(le_col);
                    ue_c.push(ue_col);
                }
                deadline.work(1, "during patches sparse dense-6D tap loop")?;
            }

            let lb = self.lower_b[out_flat];
            let ub = self.upper_b[out_flat];
            let (le_opt, ue_opt) = if need_err {
                (Some(le_c.as_slice()), Some(ue_c.as_slice()))
            } else {
                (None, None)
            };
            let (l, u) =
                concretize_row_directed(lb, ub, &inl_c, &inu_c, &la_c, &ua_c, le_opt, ue_opt);
            out_lower[out_flat] = l;
            out_upper[out_flat] = u;
        }

        finalize(out_lower, out_upper, deadline)
    }

    /// Dense identity (`A = I`, exact): each output row `i` is a single column
    /// `i` with coefficient `1.0`.
    fn sparse_concretize_identity(
        &self,
        in_l: &[f32],
        in_u: &[f32],
        in_dim: usize,
        input_scratch_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<BoundedTensor> {
        self.lower_a.validate_identity_geometry()?;
        self.upper_a.validate_identity_geometry()?;
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec("patches sparse concretize: output dims overflow".into())
        })?;
        if out_dim != in_dim || self.row_count != out_dim {
            return Err(NyError::UnsupportedOp(
                "patches sparse concretize: identity dim mismatch".into(),
            ));
        }
        self.validate_row_count()?;
        if self.lower_a.coeff_err.is_some() || self.upper_a.coeff_err.is_some() {
            return Err(NyError::InternalError(
                "patches sparse concretize: coeff_err carried on an exact identity path".into(),
            ));
        }

        let required_bytes = out_dim
            .saturating_mul(2)
            .saturating_mul(size_of::<f32>())
            .saturating_add(input_scratch_bytes)
            .saturating_add(self.memory_bytes());
        let mut admission = check_sparse_memory(
            required_bytes,
            "patches sparse concretize identity materialization",
        )?;
        let mut out_lower = try_filled_f32_vec(
            out_dim,
            0.0,
            &mut admission,
            "patches sparse concretize identity lower output allocation",
            deadline,
        )?;
        let mut out_upper = try_filled_f32_vec(
            out_dim,
            0.0,
            &mut admission,
            "patches sparse concretize identity upper output allocation",
            deadline,
        )?;
        for out_flat in 0..out_dim {
            deadline.work(1, "during patches sparse identity row loop")?;
            let lb = self.lower_b[out_flat];
            let ub = self.upper_b[out_flat];
            let (l, u) = concretize_row_directed(
                lb,
                ub,
                &[in_l[out_flat]],
                &[in_u[out_flat]],
                &[1.0],
                &[1.0],
                None,
                None,
            );
            out_lower[out_flat] = l;
            out_upper[out_flat] = u;
        }
        finalize(out_lower, out_upper, deadline)
    }

    /// 4D sparse patches and sparse identity (both exact — the sparse layout
    /// carries no `coeff_err` by scope guard). The full `out_dim` grid is
    /// emitted; positions with no unstable neuron are the all-zero row
    /// (`concretize_row_directed(0, 0, …)`, identical to `expand_sparse_bias`'s
    /// zero-filled non-sparse rows).
    fn sparse_concretize_sparse_layout(
        &self,
        in_l: &[f32],
        in_u: &[f32],
        in_dim: usize,
        input_scratch_bytes: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<BoundedTensor> {
        // Authenticate both indices, tensor prefixes, and bias contracts before
        // allocating the full output vectors or reusing the lower unfold map.
        let idx = self.validate_sparse_pair_with_poll(deadline)?;

        // Scope guard mirror (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md I2/B6): the
        // sparse layout stays exact; a carried Some is a hard error here too.
        if self.lower_a.coeff_err.is_some() || self.upper_a.coeff_err.is_some() {
            return Err(NyError::InternalError(
                "sparse patches concretize: coeff_err carried on sparse path (unsupported; \
                 sparse stays exact by scope guard)"
                    .into(),
            ));
        }
        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let (in_c, _, _) = self.lower_a.input_shape;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec("patches sparse concretize: output dims overflow".into())
        })?;

        // Non-sparse rows: all-zero coefficients, zero bias (expand_sparse_bias).
        let (empty_l, empty_u) = concretize_row_directed(0.0, 0.0, &[], &[], &[], &[], None, None);

        if self.lower_a.identity && self.upper_a.identity {
            if out_dim != in_dim {
                return Err(NyError::UnsupportedOp(
                    "patches sparse concretize: sparse identity dim mismatch".into(),
                ));
            }
            let required_bytes = out_dim
                .saturating_mul(2)
                .saturating_mul(size_of::<f32>())
                .saturating_add(input_scratch_bytes)
                .saturating_add(self.memory_bytes());
            let mut admission = check_sparse_memory(
                required_bytes,
                "patches sparse identity concretize materialization",
            )?;
            let mut out_lower = try_filled_f32_vec(
                out_dim,
                empty_l,
                &mut admission,
                "patches sparse identity lower output allocation",
                deadline,
            )?;
            let mut out_upper = try_filled_f32_vec(
                out_dim,
                empty_u,
                &mut admission,
                "patches sparse identity upper output allocation",
                deadline,
            )?;
            for i in 0..idx.len() {
                deadline.work(1, "during sparse identity concretize row loop")?;
                let flat = idx.flat_index(i, out_h, out_w);
                let lb = self.lower_b[i];
                let ub = self.upper_b[i];
                let (l, u) = concretize_row_directed(
                    lb,
                    ub,
                    &[in_l[flat]],
                    &[in_u[flat]],
                    &[1.0],
                    &[1.0],
                    None,
                    None,
                );
                out_lower[flat] = l;
                out_upper[flat] = u;
            }
            return finalize(out_lower, out_upper, deadline);
        }

        let lower_patches = self.lower_a.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("sparse concretize: lower patches tensor is None".into())
        })?;
        let upper_patches = self.upper_a.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("sparse concretize: upper patches tensor is None".into())
        })?;
        let lshape = lower_patches.shape();
        let ushape = upper_patches.shape();
        // Only 4D `(unstable_size, in_c, kH, kW)` is fast-pathed; 5D explicit
        // spec rows fall back to the certified dense path.
        if lshape.len() != 4 || ushape.len() != 4 {
            return Err(NyError::UnsupportedOp(
                "patches sparse concretize: only 4D sparse fast-pathed".into(),
            ));
        }
        if lshape[1] != in_c {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_c],
                got: vec![lshape[1]],
            });
        }
        let (kh, kw) = (lshape[2], lshape[3]);
        if ushape[1] != in_c || ushape[2] != kh || ushape[3] != kw {
            return Err(NyError::UnsupportedOp(
                "patches sparse concretize: lower/upper sparse shape mismatch".into(),
            ));
        }

        let resident_before_map = self.memory_bytes().saturating_add(input_scratch_bytes);
        let index_map = compute_unfold_index_map_with_deadline(
            &self.lower_a,
            kh,
            kw,
            resident_before_map,
            deadline,
        )?;
        let plan = build_unfold_plan(&index_map);
        let block = plan.block();
        let output_pair_bytes = out_dim.saturating_mul(2).saturating_mul(size_of::<f32>());
        let workspace_bytes = block.saturating_mul(4).saturating_mul(size_of::<f32>());
        let resident_bytes = self
            .memory_bytes()
            .saturating_add(input_scratch_bytes)
            .saturating_add(index_map.memory_bytes())
            .saturating_add(strided_scratch_bytes(lower_patches))
            .saturating_add(strided_scratch_bytes(upper_patches));
        let required_bytes = resident_bytes
            .saturating_add(output_pair_bytes)
            .saturating_add(workspace_bytes);
        let mut admission =
            check_sparse_memory(required_bytes, "sparse patches concretize materialization")?;
        let mut lscratch = Vec::new();
        let mut uscratch = Vec::new();
        let lower_flat = try_as_flat_with_deadline(
            lower_patches,
            &mut lscratch,
            "sparse patches concretize lower scratch",
            &mut admission,
            deadline,
        )?;
        let upper_flat = try_as_flat_with_deadline(
            upper_patches,
            &mut uscratch,
            "sparse patches concretize upper scratch",
            &mut admission,
            deadline,
        )?;

        let mut la_c: Vec<f32> = Vec::new();
        let mut ua_c: Vec<f32> = Vec::new();
        let mut inl_c: Vec<f32> = Vec::new();
        let mut inu_c: Vec<f32> = Vec::new();

        let mut out_lower = try_filled_f32_vec(
            out_dim,
            empty_l,
            &mut admission,
            "sparse patches concretize lower output allocation",
            deadline,
        )?;
        let mut out_upper = try_filled_f32_vec(
            out_dim,
            empty_u,
            &mut admission,
            "sparse patches concretize upper output allocation",
            deadline,
        )?;
        {
            let mut workspaces = [&mut la_c, &mut ua_c, &mut inl_c, &mut inu_c];
            try_reserve_row_workspace(
                &mut workspaces,
                block,
                &mut admission,
                "sparse patches concretize row workspace",
                deadline,
            )?;
        }

        for i in 0..idx.len() {
            deadline.work(1, "during sparse patches concretize row loop")?;
            let out_flat = idx.flat_index(i, out_h, out_w);
            let h = idx.heights[i];
            let w = idx.widths[i];
            let pat_base = i * block;

            la_c.clear();
            ua_c.clear();
            inl_c.clear();
            inu_c.clear();
            for &(block_offset, in_flat) in plan.taps_for(h, w) {
                la_c.push(lower_flat[pat_base + block_offset]);
                ua_c.push(upper_flat[pat_base + block_offset]);
                inl_c.push(in_l[in_flat]);
                inu_c.push(in_u[in_flat]);
                deadline.work(1, "during sparse patches concretize tap loop")?;
            }
            let lb = self.lower_b[i];
            let ub = self.upper_b[i];
            let (l, u) = concretize_row_directed(lb, ub, &inl_c, &inu_c, &la_c, &ua_c, None, None);
            out_lower[out_flat] = l;
            out_upper[out_flat] = u;
        }

        finalize(out_lower, out_upper, deadline)
    }
}
