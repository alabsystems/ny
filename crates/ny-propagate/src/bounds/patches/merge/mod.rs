// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches merge primitives for residual (Add/Sub) fan-in in graph CROWN.
//!
//! When two branches of a residual connection both carry `PatchesLinearBounds`,
//! these helpers allow merging them in patches form instead of forcing dense
//! materialization. This is the key optimization for CNN DAGs (#4382).
//!
//! Reference: alpha-beta-CROWN `auto_LiRPA/patches.py:147-171` (`Patches.__add__`)

use ndarray::{Array1, ArrayD, Axis, IxDyn, Zip};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
use rayon::prelude::*;
use std::mem::size_of;

use super::types::{PatchGeometry, PatchesData};
use super::PatchesLinearBounds;

#[cfg(test)]
mod merge_tests;

#[derive(Clone, Copy)]
struct ResidualCloneAdmission {
    required_bytes: usize,
    budget_bytes: usize,
}

impl ResidualCloneAdmission {
    fn error(self, site: &'static str) -> NyError {
        NyError::CpuMemoryExceeded {
            required_bytes: self.required_bytes,
            budget_bytes: self.budget_bytes,
            site,
        }
    }

    fn reconcile(
        self,
        allocated_elements: usize,
        remaining_elements: usize,
        site: &'static str,
    ) -> Result<()> {
        let required_bytes = allocated_elements
            .checked_add(remaining_elements)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .unwrap_or(usize::MAX);
        if required_bytes > self.budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes: self.budget_bytes,
                site,
            });
        }
        Ok(())
    }
}

fn try_copy_residual_array(
    source: &ArrayD<f32>,
    negate: bool,
    allocated_elements: &mut usize,
    remaining_elements: &mut usize,
    admission: ResidualCloneAdmission,
    site: &'static str,
) -> Result<ArrayD<f32>> {
    let len = source.len();
    *remaining_elements = remaining_elements
        .checked_sub(len)
        .ok_or_else(|| admission.error(site))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| admission.error(site))?;
    *allocated_elements = allocated_elements.saturating_add(values.capacity());
    admission.reconcile(*allocated_elements, *remaining_elements, site)?;
    if negate {
        values.extend(source.iter().map(|value| -*value));
    } else {
        values.extend(source.iter().copied());
    }
    ArrayD::from_shape_vec(IxDyn(source.shape()), values).map_err(|error| {
        NyError::InternalError(format!(
            "{site}: checked residual patch shape construction failed: {error}"
        ))
    })
}

fn try_zero_residual_bias(
    len: usize,
    allocated_elements: &mut usize,
    remaining_elements: &mut usize,
    admission: ResidualCloneAdmission,
    site: &'static str,
) -> Result<Array1<f32>> {
    *remaining_elements = remaining_elements
        .checked_sub(len)
        .ok_or_else(|| admission.error(site))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| admission.error(site))?;
    *allocated_elements = allocated_elements.saturating_add(values.capacity());
    admission.reconcile(*allocated_elements, *remaining_elements, site)?;
    values.resize(len, 0.0);
    Ok(Array1::from_vec(values))
}

fn try_copy_residual_error(
    source: &Array1<f32>,
    allocated_elements: &mut usize,
    remaining_elements: &mut usize,
    admission: ResidualCloneAdmission,
    site: &'static str,
) -> Result<Array1<f32>> {
    let len = source.len();
    *remaining_elements = remaining_elements
        .checked_sub(len)
        .ok_or_else(|| admission.error(site))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| admission.error(site))?;
    *allocated_elements = allocated_elements.saturating_add(values.capacity());
    admission.reconcile(*allocated_elements, *remaining_elements, site)?;
    values.extend(source.iter().copied());
    Ok(Array1::from_vec(values))
}

impl PatchesLinearBounds {
    /// Reference helper retained for merge unit tests. Production residual
    /// fan-out uses [`Self::try_clone_residual_branch`] so large coefficients
    /// are receipted and allocated fallibly.
    #[cfg(test)]
    pub(crate) fn clone_with_zero_bias(&self) -> Self {
        PatchesLinearBounds {
            row_count: self.row_count,
            lower_a: self.lower_a.clone(),
            lower_b: Array1::zeros(self.lower_b.len()),
            upper_a: self.upper_a.clone(),
            upper_b: Array1::zeros(self.upper_b.len()),
        }
    }

    /// Fallibly duplicate one residual branch while zeroing its bias.
    ///
    /// The original relation is moved to the left branch by the caller, so
    /// this is the only full-size coefficient copy needed for `Add`/`Sub`.
    /// Every owned f32 buffer is jointly receipted and allocated through
    /// `try_reserve_exact`; Anchored geometry is Arc-shared without copying its
    /// origin vectors. Sparse carriers and a virtual identity that would need a
    /// materialized `-I` are deliberately left to the Dense fallback.
    pub(crate) fn try_clone_residual_branch(&self, negate: bool) -> Result<Self> {
        const SITE: &str = "Patches residual branch clone";

        self.lower_a.validate_common_geometry(&self.upper_a)?;
        if self.lower_a.unstable_idx.is_some() || self.upper_a.unstable_idx.is_some() {
            return Err(NyError::UnsupportedConfiguration(
                "Patches residual branch clone does not support sparse carriers".into(),
            ));
        }
        if self.lower_b.len() != self.row_count || self.upper_b.len() != self.row_count {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.row_count, self.row_count],
                got: vec![self.lower_b.len(), self.upper_b.len()],
            });
        }
        if negate && (self.lower_a.identity || self.upper_a.identity) {
            return Err(NyError::UnsupportedConfiguration(
                "Patches residual subtraction requires Dense fallback for virtual identity".into(),
            ));
        }

        let required_elements = [
            self.lower_a.patches.as_ref().map_or(0, ArrayD::len),
            self.lower_a.coeff_err.as_ref().map_or(0, Array1::len),
            self.upper_a.patches.as_ref().map_or(0, ArrayD::len),
            self.upper_a.coeff_err.as_ref().map_or(0, Array1::len),
            self.lower_b.len(),
            self.upper_b.len(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .unwrap_or(usize::MAX);
        let required_bytes = required_elements.saturating_mul(size_of::<f32>());
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let admission = ResidualCloneAdmission {
            required_bytes,
            budget_bytes,
        };
        if required_bytes > budget_bytes {
            return Err(admission.error(SITE));
        }

        let mut allocated_elements = 0usize;
        let mut remaining_elements = required_elements;
        let (lower_a, upper_a) = {
            let mut copy_side = |source: &PatchesData| -> Result<PatchesData> {
                let patches = source
                    .patches
                    .as_ref()
                    .map(|array| {
                        try_copy_residual_array(
                            array,
                            negate,
                            &mut allocated_elements,
                            &mut remaining_elements,
                            admission,
                            SITE,
                        )
                    })
                    .transpose()?;
                let coeff_err = source
                    .coeff_err
                    .as_ref()
                    .map(|array| {
                        try_copy_residual_error(
                            array,
                            &mut allocated_elements,
                            &mut remaining_elements,
                            admission,
                            SITE,
                        )
                    })
                    .transpose()?;
                Ok(PatchesData {
                    coeff_err,
                    patches,
                    geometry: source.geometry.clone(),
                    identity: source.identity,
                    output_shape: source.output_shape,
                    input_shape: source.input_shape,
                    unstable_idx: None,
                })
            };

            (copy_side(&self.lower_a)?, copy_side(&self.upper_a)?)
        };
        let lower_b = try_zero_residual_bias(
            self.lower_b.len(),
            &mut allocated_elements,
            &mut remaining_elements,
            admission,
            SITE,
        )?;
        let upper_b = try_zero_residual_bias(
            self.upper_b.len(),
            &mut allocated_elements,
            &mut remaining_elements,
            admission,
            SITE,
        )?;
        debug_assert_eq!(remaining_elements, 0);

        Ok(PatchesLinearBounds {
            row_count: self.row_count,
            lower_a,
            lower_b,
            upper_a,
            upper_b,
        })
    }

    /// Reference negation helper retained for merge unit tests. Production
    /// residual subtraction uses [`Self::try_clone_residual_branch`].
    ///
    /// For `Sub` backward: the right-hand input sees `-A`, each relation keeping
    /// its OWN coefficients. `lower_a' = -lower_a`, `upper_a' = -upper_a`,
    /// biases zeroed.
    ///
    /// There is deliberately NO lower/upper swap. CROWN composes by
    /// substitution: `obj >= lower_a·(u - v) + lower_b` gives `-lower_a` as
    /// `v`'s lower coefficient. Swapping is the rule for negating a bounded
    /// QUANTITY, not a relation's coefficients, and it is not conservative —
    /// see the dense counterpart `binary_ops/sub.rs::propagate_linear_binary`,
    /// where the swapped form produced a demonstrably FALSE bound.
    ///
    /// If either side is identity, materializes it first since the virtual
    /// identity flag only encodes `+I` (no compact `-I` variant).
    #[cfg(test)]
    pub(crate) fn negated_zero_bias(&self) -> Result<Self> {
        let lower_src = if self.lower_a.identity {
            self.lower_a.try_materialize_identity()?
        } else {
            self.lower_a.clone()
        };
        let upper_src = if self.upper_a.identity {
            self.upper_a.try_materialize_identity()?
        } else {
            self.upper_a.clone()
        };

        Ok(PatchesLinearBounds {
            row_count: self.row_count,
            lower_a: lower_src.negated(),
            lower_b: Array1::zeros(self.lower_b.len()),
            upper_a: upper_src.negated(),
            upper_b: Array1::zeros(self.upper_b.len()),
        })
    }

    /// Try to merge another compatible patches contribution in-place.
    ///
    /// Returns `Ok(true)` if merge succeeded (patches stayed in patches form).
    /// Returns `Ok(false)` if carriers are incompatible (caller should fall back
    /// to dense promotion).
    ///
    /// Reference: alpha-beta-CROWN `patches.py:147-171` (`Patches.__add__`)
    pub(crate) fn try_merge_inplace(&mut self, other: &Self) -> Result<bool> {
        // Authenticate both dual-side carriers before inspecting the affine
        // representation or reconciling padding.  Besides guaranteeing that
        // lower/upper use one exact map, this rejects malformed affine
        // metadata before padding arithmetic, allocation, or receiver mutation.
        self.lower_a.validate_common_geometry(&self.upper_a)?;
        other.lower_a.validate_common_geometry(&other.upper_a)?;

        // Padding reconciliation below is an affine-only operation. Refuse an
        // anchored carrier before taking ownership or allocating padded copies;
        // the graph accumulator will promote both contributions to Dense.
        if self
            .lower_a
            .geometry
            .require_affine("Patches lower merge")
            .is_err()
            || self
                .upper_a
                .geometry
                .require_affine("Patches upper merge")
                .is_err()
            || other
                .lower_a
                .geometry
                .require_affine("Patches lower merge peer")
                .is_err()
            || other
                .upper_a
                .geometry
                .require_affine("Patches upper merge peer")
                .is_err()
        {
            return Ok(false);
        }
        if !self.check_merge_compatibility(other) {
            return Ok(false);
        }

        let self_lower_geometry = self
            .lower_a
            .geometry
            .require_affine("Patches lower merge")?;
        let other_lower_geometry = other
            .lower_a
            .geometry
            .require_affine("Patches lower merge peer")?;
        let self_upper_geometry = self
            .upper_a
            .geometry
            .require_affine("Patches upper merge")?;
        let other_upper_geometry = other
            .upper_a
            .geometry
            .require_affine("Patches upper merge peer")?;
        let lower_padding = match reconcile_padding(
            self_lower_geometry.padding(),
            other_lower_geometry.padding(),
        ) {
            Some(p) => p,
            None => return Ok(false),
        };
        let upper_padding = match reconcile_padding(
            self_upper_geometry.padding(),
            other_upper_geometry.padding(),
        ) {
            Some(p) => p,
            None => return Ok(false),
        };

        let (other_lower, other_upper) = materialize_pair_cloned(other)?;
        let (mut self_lower, mut self_upper) = self.take_or_materialize_pair()?;

        if !try_merge_patches_pair(
            &mut self_lower,
            &other_lower,
            lower_padding,
            &mut self_upper,
            &other_upper,
            upper_padding,
        ) {
            self.lower_a = self_lower;
            self.upper_a = self_upper;
            return Ok(false);
        }

        self.lower_a = self_lower;
        self.upper_a = self_upper;
        self.lower_b = outward_round_add_lower_1d(&self.lower_b, &other.lower_b);
        self.upper_b = outward_round_add_upper_1d(&self.upper_b, &other.upper_b);
        Ok(true)
    }

    fn check_merge_compatibility(&self, other: &Self) -> bool {
        self.row_count == other.row_count
            // Sparse Add/Sub performs rounded coefficient arithmetic but has no
            // 4D/5D error receipt. Decline before mutation until that channel
            // exists; the caller's dense promotion is the sound path.
            && self.lower_a.unstable_idx.is_none()
            && self.upper_a.unstable_idx.is_none()
            && other.lower_a.unstable_idx.is_none()
            && other.upper_a.unstable_idx.is_none()
            && metadata_compatible(&self.lower_a, &other.lower_a)
            && metadata_compatible(&self.upper_a, &other.upper_a)
            && same_representation_family(&self.lower_a, &other.lower_a)
            && same_representation_family(&self.upper_a, &other.upper_a)
            // Error/layout alignment runs before `take_or_materialize_pair`, so
            // a malformed carrier cannot mutate `self` or turn a missing error
            // entry into an exact zero during merge.
            && coefficient_error_layout_aligned(&self.lower_a, self.row_count)
            && coefficient_error_layout_aligned(&self.upper_a, self.row_count)
            && coefficient_error_layout_aligned(&other.lower_a, self.row_count)
            && coefficient_error_layout_aligned(&other.upper_a, self.row_count)
    }

    fn take_or_materialize_pair(&mut self) -> Result<(PatchesData, PatchesData)> {
        // Complete every fallible allocation before replacing either receiver
        // field, preserving try_merge_inplace's atomic refusal contract.
        let materialized_lower = if self.lower_a.identity {
            Some(self.lower_a.try_materialize_identity()?)
        } else {
            None
        };
        let materialized_upper = if self.upper_a.identity {
            Some(self.upper_a.try_materialize_identity()?)
        } else {
            None
        };
        let placeholder = PatchesData {
            coeff_err: None,
            patches: None,
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: true,
            output_shape: (0, 0, 0),
            input_shape: (0, 0, 0),
            unstable_idx: None,
        };
        let lower = match materialized_lower {
            Some(materialized) => materialized,
            None => std::mem::replace(&mut self.lower_a, placeholder.clone()),
        };
        let upper = match materialized_upper {
            Some(materialized) => materialized,
            None => std::mem::replace(&mut self.upper_a, placeholder),
        };
        Ok((lower, upper))
    }
}

fn materialize_pair_cloned(p: &PatchesLinearBounds) -> Result<(PatchesData, PatchesData)> {
    let lower = if p.lower_a.identity {
        p.lower_a.try_materialize_identity()?
    } else {
        p.lower_a.clone()
    };
    let upper = if p.upper_a.identity {
        p.upper_a.try_materialize_identity()?
    } else {
        p.upper_a.clone()
    };
    Ok((lower, upper))
}

/// Try to merge patches tensors in-place. Returns false if shapes don't match.
fn try_merge_patches_pair(
    self_lower: &mut PatchesData,
    other_lower: &PatchesData,
    lower_padding: (usize, usize, usize, usize),
    self_upper: &mut PatchesData,
    other_upper: &PatchesData,
    upper_padding: (usize, usize, usize, usize),
) -> bool {
    let (Some(slp), Some(olp)) = (self_lower.patches.as_ref(), other_lower.patches.as_ref()) else {
        return false;
    };
    let (Some(sup), Some(oup)) = (self_upper.patches.as_ref(), other_upper.patches.as_ref()) else {
        return false;
    };

    let Ok(self_lower_geometry) = self_lower.geometry.require_affine("Patches lower merge") else {
        return false;
    };
    let Ok(other_lower_geometry) = other_lower
        .geometry
        .require_affine("Patches lower merge peer")
    else {
        return false;
    };
    let Ok(self_upper_geometry) = self_upper.geometry.require_affine("Patches upper merge") else {
        return false;
    };
    let Ok(other_upper_geometry) = other_upper
        .geometry
        .require_affine("Patches upper merge peer")
    else {
        return false;
    };

    let slp = pad_patches_if_needed(slp, self_lower_geometry.padding(), lower_padding);
    let olp = pad_patches_if_needed(olp, other_lower_geometry.padding(), lower_padding);
    let sup = pad_patches_if_needed(sup, self_upper_geometry.padding(), upper_padding);
    let oup = pad_patches_if_needed(oup, other_upper_geometry.padding(), upper_padding);

    if slp.shape() != olp.shape() || sup.shape() != oup.shape() {
        return false;
    }

    let merged_lower = outward_round_add_lower(&slp, &olp);
    let merged_upper = outward_round_add_upper(&sup, &oup);

    // Certified per-row coefficient error for the residual (Add) merge
    // (#patches-coeff-err-soundness, HOLE 3; 7D lift:
    // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §4). Carried per side, computed from
    // the *stored* merged coefficients; even exact (None+None) inputs pick up
    // the directed-rounding term, so this always yields Some for the 6D dense
    // and 7D explicit-rows layouts. Sparse carriers are refused by the
    // compatibility preflight before this helper because they cannot yet
    // publish this intrinsic rounding receipt.
    let lower_err = merge_coeff_err(
        &merged_lower,
        self_lower.coeff_err.as_ref(),
        other_lower.coeff_err.as_ref(),
        self_lower.unstable_idx.is_some(),
    );
    let upper_err = merge_coeff_err(
        &merged_upper,
        self_upper.coeff_err.as_ref(),
        other_upper.coeff_err.as_ref(),
        self_upper.unstable_idx.is_some(),
    );
    self_lower.coeff_err = lower_err;
    self_upper.coeff_err = upper_err;

    self_lower.patches = Some(merged_lower);
    self_lower.geometry = PatchGeometry::affine(self_lower_geometry.stride(), lower_padding);
    self_lower.identity = false;

    self_upper.patches = Some(merged_upper);
    self_upper.geometry = PatchGeometry::affine(self_upper_geometry.stride(), upper_padding);
    self_upper.identity = false;

    true
}

/// Certified per-row coefficient error for a residual (Add) patch merge
/// (#patches-coeff-err-soundness, HOLE 3). The two branches' stored coefficients
/// are summed with directed *outward* rounding (`next_down`/`next_up`, <=1.5 ulp
/// per element). With `U = 2^-24` (the f32 unit round-off) and
/// `RowMaxAbs(merged, i) = max_j |merged_stored[i, j]|`:
///
///   new_err[i] = next_up( a_err[i] + b_err[i] + 3*U*RowMaxAbs(merged, i) )
///
/// SOUND: for every stored coeff `j` in logical row `i`, with `s = a_s + b_s`
/// (the exact real sum of the two stored coeffs),
///   |merged_stored - merged_true|
///     <= |round_dir(s) - s| + |a_s - a_true| + |b_s - b_true|
///     <= 1.5*ulp(merged_stored) + a_err[i] + b_err[i]
///     <= 3*U*|merged_stored|   + a_err[i] + b_err[i]        (ulp(w) <= 2U*|w|)
///     <= 3*U*RowMaxAbs(merged, i) + a_err[i] + b_err[i].
/// The merge rounding is never zero, so exact (`None`+`None`) inputs still emit
/// `Some(3*U*RowMaxAbs)`.
///
/// Logical-row semantics per layout:
///
/// - **6D dense** `[out_c, out_h, out_w, in_c, kh, kw]`: row
///   `i = oc*out_h*out_w + oh*out_w + ow` matches the row bias indexing; err
///   length `out_c*out_h*out_w`. A mismatched length is rejected by the merge
///   gate and independently poisons this helper's whole side to `+INF`; invalid
///   values likewise sanitize outward instead of becoming exact zero.
/// - **7D explicit-rows** `[rows, out_c, out_h, out_w, in_c, kh, kw]`
///   (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §4): the err index is the SPEC row
///   (axis 0); `RowMaxAbs` reduces over ALL 6 trailing axes so the single row
///   scalar covers every stored coefficient of the spec row (I1). Err length
///   `rows` (== `row_count` == bias length, enforced by the merge alignment
///   gate in `check_merge_compatibility`). Carried non-finite/negative err
///   entries sanitize to `+INF` (outward degrade poison, I5) — NEVER NaN -> 0;
///   a length-mismatched `Some` (unreachable post-gate) poisons the whole side
///   to `+INF` instead of the 6D-style silent read (I6). Emits `Some` even for
///   `None`+`None` (the directed outward add rounding is intrinsic;
///   `new_err[r] >= 2^-149` always). No bias discharge: biases merge
///   independently with their own directed rounding, so zero discharge is exact.
///
/// Sparse (`unstable_idx` Some) or any other ndim returns `None`; callers must
/// refuse those layouts before arithmetic because their overlap-aware error
/// scatter is not implemented.
fn merge_coeff_err(
    merged: &ArrayD<f32>,
    a_err: Option<&Array1<f32>>,
    b_err: Option<&Array1<f32>>,
    sparse: bool,
) -> Option<Array1<f32>> {
    // f32 unit round-off 2^-24; 3*U over-bounds the <=1.5-ulp directed-add error.
    const U: f64 = 1.0 / (1u64 << 24) as f64;
    if sparse {
        return None;
    }
    match merged.ndim() {
        6 => {
            let sh = merged.shape();
            let (out_c, out_h, out_w, in_c, kh, kw) = (sh[0], sh[1], sh[2], sh[3], sh[4], sh[5]);
            let rows = out_c * out_h * out_w;
            let block = in_c * kh * kw;
            let mut ne = Array1::<f32>::zeros(rows);
            let a_bad = a_err.is_some_and(|e| e.len() != rows);
            let b_bad = b_err.is_some_and(|e| e.len() != rows);
            let row_err = |err: Option<&Array1<f32>>, bad: bool, row: usize| -> f64 {
                let Some(error) = err else {
                    return 0.0;
                };
                if bad {
                    return f64::INFINITY;
                }
                let value = error[row];
                if value.is_finite() && value >= 0.0 {
                    f64::from(value)
                } else {
                    f64::INFINITY
                }
            };

            // Row `i` owns the CONTIGUOUS trailing block `[ic, ki, kj]`, so on a
            // standard-layout tensor the per-row max of `|coeff|` is a flat scan
            // over `block`-sized chunks. Profiling a cifar100 CROWN-IBP
            // collection put this function at the top of the ny profile: the
            // previous six-deep loop paid ndarray's checked 6-D `[[..]]` index
            // (bounds check + stride arithmetic) per element, and the residual
            // `Add` route made merges frequent enough for that to matter.
            //
            // Bit-identical: `max` over `|x|` is associative and commutative and
            // exact in f64, so visiting the same elements in a different order
            // yields the same `rowmax`, and the surrounding arithmetic is
            // untouched. Non-standard layouts fall back to the indexed walk.
            let row_terms = |i: usize, rowmax: f64| -> f32 {
                let ae = row_err(a_err, a_bad, i);
                let be = row_err(b_err, b_bad, i);
                next_up_f32((ae + be + 3.0 * U * rowmax) as f32)
            };

            match (merged.as_slice(), ne.as_slice_mut()) {
                (Some(flat), Some(out)) if block > 0 && flat.len() == rows * block => {
                    // Rows are disjoint and each is a max over its own contiguous
                    // block, so this parallelises with no summation-order change —
                    // `max` over `|x|` is associative, commutative and exact in
                    // f64, and `row_terms` is a pure function of `(i, rowmax)`.
                    // Bit-identical to the serial fill.
                    //
                    // Sample-profiling a cifar100 CROWN-IBP collection put this
                    // function at the top of the ny profile while 64% of the
                    // machine sat idle: the compose alternates brief parallel GEMM
                    // bursts with long single-threaded full-tensor bookkeeping,
                    // and this scan is one of those serial stretches.
                    out.par_iter_mut()
                        .zip(flat.par_chunks_exact(block))
                        .enumerate()
                        .for_each(|(i, (dst, chunk))| {
                            let mut rowmax = 0.0f64;
                            for &v in chunk {
                                let a = f64::from(v).abs();
                                if a > rowmax {
                                    rowmax = a;
                                }
                            }
                            *dst = row_terms(i, rowmax);
                        });
                }
                _ => {
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let i = oc * out_h * out_w + oh * out_w + ow;
                                let mut rowmax = 0.0f64;
                                for ic in 0..in_c {
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let a =
                                                f64::from(merged[[oc, oh, ow, ic, ki, kj]]).abs();
                                            if a > rowmax {
                                                rowmax = a;
                                            }
                                        }
                                    }
                                }
                                ne[i] = row_terms(i, rowmax);
                            }
                        }
                    }
                }
            }
            Some(ne)
        }
        7 => {
            let rows = merged.shape()[0];
            // Belt-and-braces poison (unreachable post-gate, spec §4.1/B5): a
            // length-mismatched Some means an inconsistent carrier; poison the
            // whole side to +INF (outward degrade) — never the 6D-style silent
            // `.get(r).unwrap_or(0.0)` under-count (I6).
            let a_bad = a_err.is_some_and(|e| e.len() != rows);
            let b_bad = b_err.is_some_and(|e| e.len() != rows);
            // Consumption sanitize (I5): non-finite or negative carried err
            // maps to +INF (poisons outward); NaN NEVER maps to 0.
            let row_err = |err: Option<&Array1<f32>>, bad: bool, r: usize| -> f64 {
                let Some(e) = err else { return 0.0 };
                if bad {
                    return f64::INFINITY;
                }
                let v = e[r];
                if v.is_finite() && v >= 0.0 {
                    f64::from(v)
                } else {
                    f64::INFINITY
                }
            };
            let mut ne = Array1::<f32>::zeros(rows);
            for (r, row) in merged.axis_iter(Axis(0)).enumerate() {
                // f64 exact max of |merged| over the 6 trailing axes. `merged`
                // is NaN-free (outward_round_add_* maps non-finite sums to
                // ±INF), so rowmax ∈ [0, +INF] and — with ae/be sanitized to
                // [0, +INF] and the finite constant 3U — the term below is
                // never NaN (no 0·INF product exists in this formula).
                let mut rowmax = 0.0f64;
                for &v in row.iter() {
                    let a = f64::from(v).abs();
                    if a > rowmax {
                        rowmax = a;
                    }
                }
                let ae = row_err(a_err, a_bad, r);
                let be = row_err(b_err, b_bad, r);
                // f64 evaluation, ONE outward next_up at the f32 cast (I4).
                ne[r] = next_up_f32((ae + be + 3.0 * U * rowmax) as f32);
            }
            Some(ne)
        }
        _ => None,
    }
}

impl PatchesData {
    /// Negate all patch coefficients. Used by `negated_zero_bias`.
    #[cfg(test)]
    pub(super) fn negated(&self) -> PatchesData {
        let neg_patches = self.patches.as_ref().map(|p| {
            let mut out = p.clone();
            out.mapv_inplace(|v| -v);
            out
        });
        // Certified error (#patches-coeff-err-soundness, HOLE 3 / Sub-negate;
        // 7D lift: docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §4.1): negation is an
        // exact sign flip, so |-stored - (-true)| = |stored - true| <= err[i]
        // and the per-row error is carried unchanged; zero discharge (IEEE
        // negation is exact for all values, and the row axis is unpermuted).
        // The lower<->upper *swap* for `Sub` backward is done by the caller
        // (`negated_zero_bias` negates each side in place). Negation does not
        // change row identity in any layout, so carrying the receipt is exact.
        // Unsupported sparse consumers must see and refuse that receipt; dropping
        // it here would turn an unknown coefficient into an exact one.
        let coeff_err = self.coeff_err.clone();
        PatchesData {
            coeff_err,
            patches: neg_patches,
            geometry: self.geometry.clone(),
            identity: false,
            output_shape: self.output_shape,
            input_shape: self.input_shape,
            unstable_idx: self.unstable_idx.clone(),
        }
    }
}

/// Validate the coefficient-error row contract for every mergeable layout.
///
/// Identity has no stored coefficients and therefore cannot carry an error.
/// A 6D error is indexed by spatial output row; a 7D error by explicit spec
/// row. Sparse 4D/5D transport remains unsupported, so it may not carry an
/// error at all. Returning `false` keeps the merge atomic and routes the caller
/// to its established dense/refusal path.
fn coefficient_error_layout_aligned(p: &PatchesData, row_count: usize) -> bool {
    if p.identity {
        return p.coeff_err.is_none();
    }
    match &p.patches {
        Some(t) if t.ndim() == 6 => checked_shape_product(&t.shape()[..3])
            .is_some_and(|rows| p.coeff_err.as_ref().is_none_or(|e| e.len() == rows)),
        Some(t) if t.ndim() == 7 => {
            t.shape()[0] == row_count && p.coeff_err.as_ref().is_none_or(|e| e.len() == row_count)
        }
        Some(t) if t.ndim() == 4 || t.ndim() == 5 => p.coeff_err.is_none(),
        _ => false,
    }
}

fn metadata_compatible(a: &PatchesData, b: &PatchesData) -> bool {
    // First compare the whole descriptor. If padding differs, retain the
    // historical affine padding-reconciliation path, but never reinterpret an
    // anchored descriptor as a stride/padding pair.
    let geometry_compatible = if a.geometry == b.geometry {
        true
    } else {
        match (
            a.geometry.require_affine("Patches merge metadata"),
            b.geometry.require_affine("Patches merge peer metadata"),
        ) {
            (Ok(a), Ok(b)) => a.stride() == b.stride(),
            _ => false,
        }
    };
    a.output_shape == b.output_shape
        && a.input_shape == b.input_shape
        && geometry_compatible
        && unstable_idx_eq(&a.unstable_idx, &b.unstable_idx)
}

fn unstable_idx_eq(
    a: &Option<super::types::UnstableIdx>,
    b: &Option<super::types::UnstableIdx>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.channels == b.channels && a.heights == b.heights && a.widths == b.widths
        }
        _ => false,
    }
}

fn same_representation_family(a: &PatchesData, b: &PatchesData) -> bool {
    if a.identity || b.identity {
        return true;
    }
    match (&a.patches, &b.patches) {
        (Some(pa), Some(pb)) => {
            let a_sparse = a.unstable_idx.is_some();
            let b_sparse = b.unstable_idx.is_some();
            a_sparse == b_sparse && pa.ndim() == pb.ndim()
        }
        (None, None) => true,
        _ => false,
    }
}

/// Reconcile padding between two patches contributions.
/// Reference: alpha-beta-CROWN `patches.py:157-167`
pub(super) fn reconcile_padding(
    a: (usize, usize, usize, usize),
    b: (usize, usize, usize, usize),
) -> Option<(usize, usize, usize, usize)> {
    if a == b {
        return Some(a);
    }
    let a_dominates = a.0 >= b.0 && a.1 >= b.1 && a.2 >= b.2 && a.3 >= b.3;
    let b_dominates = b.0 >= a.0 && b.1 >= a.1 && b.2 >= a.2 && b.3 >= a.3;
    if a_dominates {
        Some(a)
    } else if b_dominates {
        Some(b)
    } else {
        None
    }
}

fn pad_patches_if_needed(
    patches: &ArrayD<f32>,
    current: (usize, usize, usize, usize),
    target: (usize, usize, usize, usize),
) -> ArrayD<f32> {
    if current == target {
        return patches.clone();
    }
    let ndim = patches.ndim();
    let dl = target.0 - current.0;
    let dr = target.1 - current.1;
    let dt = target.2 - current.2;
    let db = target.3 - current.3;

    let old_shape = patches.shape();
    let kh_idx = ndim - 2;
    let kw_idx = ndim - 1;

    let mut new_shape: Vec<usize> = old_shape.to_vec();
    new_shape[kh_idx] = old_shape[kh_idx] + dt + db;
    new_shape[kw_idx] = old_shape[kw_idx] + dl + dr;

    let mut padded = ArrayD::zeros(IxDyn(&new_shape));
    let dst_slice: Vec<ndarray::SliceInfoElem> = (0..ndim)
        .map(|i| {
            let (start, end) = if i == kh_idx {
                (dt, dt + old_shape[kh_idx])
            } else if i == kw_idx {
                (dl, dl + old_shape[kw_idx])
            } else {
                (0, old_shape[i])
            };
            ndarray::SliceInfoElem::Slice {
                start: start as isize,
                end: Some(end as isize),
                step: 1,
            }
        })
        .collect();
    padded.slice_mut(dst_slice.as_slice()).assign(patches);
    padded
}

fn outward_round_add_lower(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    let mut result = ArrayD::zeros(a.raw_dim());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &av, &bv| {
            let sum = av + bv;
            *r = if sum.is_finite() {
                next_down_f32(sum)
            } else {
                f32::NEG_INFINITY
            };
        });
    result
}

fn outward_round_add_upper(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    let mut result = ArrayD::zeros(a.raw_dim());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &av, &bv| {
            let sum = av + bv;
            *r = if sum.is_finite() {
                next_up_f32(sum)
            } else {
                f32::INFINITY
            };
        });
    result
}

fn outward_round_add_lower_1d(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    let mut result = Array1::zeros(a.len());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &av, &bv| {
            let sum = av + bv;
            *r = if sum.is_finite() {
                next_down_f32(sum)
            } else {
                f32::NEG_INFINITY
            };
        });
    result
}

fn outward_round_add_upper_1d(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    let mut result = Array1::zeros(a.len());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &av, &bv| {
            let sum = av + bv;
            *r = if sum.is_finite() {
                next_up_f32(sum)
            } else {
                f32::INFINITY
            };
        });
    result
}
