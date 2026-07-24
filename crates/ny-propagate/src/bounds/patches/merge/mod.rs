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
use ny_core::Result;
use ny_tensor::{next_down_f32, next_up_f32};

use super::types::PatchesData;
use super::PatchesLinearBounds;

#[cfg(test)]
mod merge_tests;

impl PatchesLinearBounds {
    /// Clone this bounds with bias vectors zeroed out.
    ///
    /// Used by the residual passthrough helper: the original bias is carried
    /// separately through `accumulate_bias_to_network_input_crown`, so the
    /// per-input patches carriers must have zero bias to avoid double-counting.
    pub(crate) fn clone_with_zero_bias(&self) -> Self {
        PatchesLinearBounds {
            row_count: self.row_count,
            lower_a: self.lower_a.clone(),
            lower_b: Array1::zeros(self.lower_b.len()),
            upper_a: self.upper_a.clone(),
            upper_b: Array1::zeros(self.upper_b.len()),
        }
    }

    /// Negate and swap lower/upper A-coefficients, with zero bias.
    ///
    /// For `Sub` backward: the right-hand input sees `-A` with lower/upper
    /// swapped. `lower_a' = -upper_a`, `upper_a' = -lower_a`, biases zeroed.
    ///
    /// If either side is identity, materializes it first since the virtual
    /// identity flag only encodes `+I` (no compact `-I` variant).
    pub(crate) fn negated_swapped_zero_bias(&self) -> Self {
        let lower_src = if self.upper_a.identity {
            self.upper_a.materialize_identity()
        } else {
            self.upper_a.clone()
        };
        let upper_src = if self.lower_a.identity {
            self.lower_a.materialize_identity()
        } else {
            self.lower_a.clone()
        };

        PatchesLinearBounds {
            row_count: self.row_count,
            lower_a: lower_src.negated(),
            lower_b: Array1::zeros(self.lower_b.len()),
            upper_a: upper_src.negated(),
            upper_b: Array1::zeros(self.upper_b.len()),
        }
    }

    /// Try to merge another compatible patches contribution in-place.
    ///
    /// Returns `Ok(true)` if merge succeeded (patches stayed in patches form).
    /// Returns `Ok(false)` if carriers are incompatible (caller should fall back
    /// to dense promotion).
    ///
    /// Reference: alpha-beta-CROWN `patches.py:147-171` (`Patches.__add__`)
    pub(crate) fn try_merge_inplace(&mut self, other: &Self) -> Result<bool> {
        if !self.check_merge_compatibility(other) {
            return Ok(false);
        }

        let lower_padding = match reconcile_padding(self.lower_a.padding, other.lower_a.padding) {
            Some(p) => p,
            None => return Ok(false),
        };
        let upper_padding = match reconcile_padding(self.upper_a.padding, other.upper_a.padding) {
            Some(p) => p,
            None => return Ok(false),
        };

        let (mut self_lower, mut self_upper) = self.take_or_materialize_pair();
        let (other_lower, other_upper) = materialize_pair_cloned(other);

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
            && metadata_compatible(&self.lower_a, &other.lower_a)
            && metadata_compatible(&self.upper_a, &other.upper_a)
            && same_representation_family(&self.lower_a, &other.lower_a)
            && same_representation_family(&self.upper_a, &other.upper_a)
            // 7D explicit-rows alignment gate (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md
            // §4.1): every 7D carrier must have its spec-row axis (axis 0) equal
            // to `row_count` and any carried err of that length, or the merge is
            // rejected (=> dense promotion, the certified fallback). Runs before
            // `take_or_materialize_pair`, so rejection mutates nothing.
            && explicit_rows_aligned(&self.lower_a, self.row_count)
            && explicit_rows_aligned(&self.upper_a, self.row_count)
            && explicit_rows_aligned(&other.lower_a, self.row_count)
            && explicit_rows_aligned(&other.upper_a, self.row_count)
    }

    fn take_or_materialize_pair(&mut self) -> (PatchesData, PatchesData) {
        let placeholder = PatchesData {
            coeff_err: None,
            patches: None,
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            identity: true,
            output_shape: (0, 0, 0),
            input_shape: (0, 0, 0),
            unstable_idx: None,
        };
        let lower = if self.lower_a.identity {
            self.lower_a.materialize_identity()
        } else {
            std::mem::replace(&mut self.lower_a, placeholder.clone())
        };
        let upper = if self.upper_a.identity {
            self.upper_a.materialize_identity()
        } else {
            std::mem::replace(&mut self.upper_a, placeholder)
        };
        (lower, upper)
    }
}

fn materialize_pair_cloned(p: &PatchesLinearBounds) -> (PatchesData, PatchesData) {
    let lower = if p.lower_a.identity {
        p.lower_a.materialize_identity()
    } else {
        p.lower_a.clone()
    };
    let upper = if p.upper_a.identity {
        p.upper_a.materialize_identity()
    } else {
        p.upper_a.clone()
    };
    (lower, upper)
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

    let slp = pad_patches_if_needed(slp, self_lower.padding, lower_padding);
    let olp = pad_patches_if_needed(olp, other_lower.padding, lower_padding);
    let sup = pad_patches_if_needed(sup, self_upper.padding, upper_padding);
    let oup = pad_patches_if_needed(oup, other_upper.padding, upper_padding);

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
    // and 7D explicit-rows layouts. Sparse keeps None.
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
    self_lower.padding = lower_padding;
    self_lower.identity = false;

    self_upper.patches = Some(merged_upper);
    self_upper.padding = upper_padding;
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
///   length `out_c*out_h*out_w`. This arm is kept byte-identical to the
///   certified 6D design (including its silent `.get(i).unwrap_or(0.0)` reads;
///   hardening them is a 6D follow-up).
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
/// Sparse (`unstable_idx` Some) or any other ndim keeps `None`: the
/// overlap-aware to_dense err scatter for those isn't ready, so emitting err
/// there would be unsound.
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
            let mut ne = Array1::<f32>::zeros(out_c * out_h * out_w);
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let i = oc * out_h * out_w + oh * out_w + ow;
                        let mut rowmax = 0.0f64;
                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let a = f64::from(merged[[oc, oh, ow, ic, ki, kj]]).abs();
                                    if a > rowmax {
                                        rowmax = a;
                                    }
                                }
                            }
                        }
                        let ae = a_err.map_or(0.0, |e| f64::from(e.get(i).copied().unwrap_or(0.0)));
                        let be = b_err.map_or(0.0, |e| f64::from(e.get(i).copied().unwrap_or(0.0)));
                        ne[i] = next_up_f32((ae + be + 3.0 * U * rowmax) as f32);
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
    /// Negate all patch coefficients. Used by `negated_swapped_zero_bias`.
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
        // (`negated_swapped_zero_bias` negates `upper_a` into `lower_a` and
        // vice-versa), so swapping falls out of each source carrying its own
        // err. 6D dense and 7D explicit-rows; sparse (unstable_idx Some) /
        // other ndims keep None (their to_dense err scatter isn't ready --
        // setting err there would be unsound).
        let coeff_err = match &neg_patches {
            Some(p) if (p.ndim() == 6 || p.ndim() == 7) && self.unstable_idx.is_none() => {
                self.coeff_err.clone()
            }
            _ => None,
        };
        PatchesData {
            coeff_err,
            patches: neg_patches,
            stride: self.stride,
            padding: self.padding,
            identity: false,
            output_shape: self.output_shape,
            input_shape: self.input_shape,
            unstable_idx: self.unstable_idx.clone(),
        }
    }
}

/// Alignment gate for the 7D explicit-rows layout
/// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §4.1): a side carrying a 7D tensor
/// must have its spec-row axis (axis 0) equal to the carrier `row_count`, and
/// any carried `coeff_err` must have that same length — the err index is the
/// spec row (I1), so a mismatch means the carrier is inconsistent and the
/// merge must be rejected (`Ok(false)` => dense promotion, which is
/// err-sound). Identity sides (no tensor) pass; 6D and sparse layouts are
/// deliberately untouched (6D byte-identity, I2/G2).
fn explicit_rows_aligned(p: &PatchesData, row_count: usize) -> bool {
    match &p.patches {
        Some(t) if t.ndim() == 7 => {
            t.shape()[0] == row_count && p.coeff_err.as_ref().is_none_or(|e| e.len() == row_count)
        }
        _ => true,
    }
}

fn metadata_compatible(a: &PatchesData, b: &PatchesData) -> bool {
    a.output_shape == b.output_shape
        && a.input_shape == b.input_shape
        && a.stride == b.stride
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
