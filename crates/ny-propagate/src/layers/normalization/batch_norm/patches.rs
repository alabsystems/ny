// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-mode CROWN backward propagation for BatchNorm.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{f32_to_f64_exact, f64_to_f32_down, f64_to_f32_up, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
#[cfg(test)]
use std::cell::Cell;
use std::mem::size_of;
use std::time::Instant;

use super::math::{nonnegative_add_up, nonnegative_mul_up};
use super::types::{BatchNormChannelAxisHint, BatchNormLayer};
use crate::bounds::{certified_affine_sum_f32, OutwardDirection};
use crate::layers::linear::bias::{add_f64_down, add_f64_up, nonnegative_f32_error_or_infinity};

const ANCHORED_BN_POLL_COORDS: usize = 4_096;
const ANCHORED_BN_ZERO_FILL_CHUNK: usize = ANCHORED_BN_POLL_COORDS;

struct AnchoredBatchNormAdmission {
    source_resident_bytes: usize,
    required_bytes: usize,
    budget_bytes: usize,
    remaining_planned_bytes: usize,
    allocated_capacity_bytes: usize,
}

impl AnchoredBatchNormAdmission {
    fn new(source_resident_bytes: usize, planned_allocation_bytes: usize) -> Result<Self> {
        Self::with_budget(
            source_resident_bytes,
            planned_allocation_bytes,
            anchored_bn_budget_bytes(),
        )
    }

    fn with_budget(
        source_resident_bytes: usize,
        planned_allocation_bytes: usize,
        budget_bytes: usize,
    ) -> Result<Self> {
        let required_bytes = source_resident_bytes
            .checked_add(planned_allocation_bytes)
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "Anchored BatchNorm total-live byte count overflows usize".into(),
                )
            })?;
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "Anchored BatchNorm Patches work buffers",
            });
        }
        Ok(Self {
            source_resident_bytes,
            required_bytes,
            budget_bytes,
            remaining_planned_bytes: planned_allocation_bytes,
            allocated_capacity_bytes: 0,
        })
    }

    fn reserve<T>(&mut self, len: usize, site: &'static str) -> Result<Vec<T>> {
        let planned = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            NyError::InvalidSpec(format!("{site}: allocation byte count overflows usize"))
        })?;
        self.remaining_planned_bytes = self
            .remaining_planned_bytes
            .checked_sub(planned)
            .ok_or_else(|| {
                NyError::InternalError(format!("{site}: allocation was absent from preflight"))
            })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: self.required_bytes,
                budget_bytes: self.budget_bytes,
                site,
            })?;
        let capacity_bytes =
            values
                .capacity()
                .checked_mul(size_of::<T>())
                .ok_or(NyError::CpuMemoryExceeded {
                    required_bytes: usize::MAX,
                    budget_bytes: self.budget_bytes,
                    site,
                })?;
        self.allocated_capacity_bytes =
            self.allocated_capacity_bytes.saturating_add(capacity_bytes);
        let actual_peak = self
            .source_resident_bytes
            .saturating_add(self.allocated_capacity_bytes)
            .saturating_add(self.remaining_planned_bytes);
        if actual_peak > self.budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: actual_peak,
                budget_bytes: self.budget_bytes,
                site,
            });
        }
        Ok(values)
    }

    fn zeroed<T: Clone>(
        &mut self,
        len: usize,
        zero: T,
        site: &'static str,
        poll: &mut impl FnMut() -> Result<()>,
    ) -> Result<Vec<T>> {
        let mut values = self.reserve::<T>(len, site)?;
        while values.len() < len {
            poll()?;
            let end = len.min(values.len().saturating_add(ANCHORED_BN_ZERO_FILL_CHUNK));
            values.resize(end, zero.clone());
        }
        poll()?;
        Ok(values)
    }
}

#[cfg(test)]
thread_local! {
    static ANCHORED_BN_TEST_BUDGET: Cell<Option<usize>> = const { Cell::new(None) };
}

fn anchored_bn_budget_bytes() -> usize {
    #[cfg(test)]
    if let Some(value) = ANCHORED_BN_TEST_BUDGET.with(Cell::get) {
        return value;
    }
    crate::network::crown_memory::cpu_crown_dense_budget_bytes()
}

#[cfg(test)]
fn with_anchored_bn_budget_for_test<T>(budget: usize, run: impl FnOnce() -> T) -> T {
    ANCHORED_BN_TEST_BUDGET.with(|slot| {
        let previous = slot.replace(Some(budget));
        struct Restore<'a> {
            slot: &'a Cell<Option<usize>>,
            previous: Option<usize>,
        }
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.slot.set(self.previous);
            }
        }
        let _restore = Restore { slot, previous };
        run()
    })
}

fn anchored_bn_bytes<T>(len: usize, label: &str) -> Result<usize> {
    len.checked_mul(size_of::<T>())
        .ok_or_else(|| NyError::InvalidSpec(format!("{label}: byte count overflows usize")))
}

fn anchored_bn_sum_bytes(parts: &[usize]) -> Result<usize> {
    parts.iter().try_fold(0usize, |total, &part| {
        total.checked_add(part).ok_or_else(|| {
            NyError::InvalidSpec("Anchored BatchNorm work-buffer bytes overflow usize".into())
        })
    })
}

fn anchored_bn_planned_bytes(
    map_len: usize,
    patch_elements: usize,
    logical_rows: usize,
) -> Result<usize> {
    anchored_bn_sum_bytes(&[
        anchored_bn_bytes::<bool>(map_len, "Anchored BatchNorm tap map")?,
        anchored_bn_bytes::<f32>(
            patch_elements.checked_mul(2).ok_or_else(|| {
                NyError::InvalidSpec(
                    "Anchored BatchNorm paired coefficient count overflows usize".into(),
                )
            })?,
            "Anchored BatchNorm output coefficients",
        )?,
        anchored_bn_bytes::<f64>(
            logical_rows.checked_mul(4).ok_or_else(|| {
                NyError::InvalidSpec("Anchored BatchNorm bias/widen count overflows usize".into())
            })?,
            "Anchored BatchNorm f64 bias and widen",
        )?,
        anchored_bn_bytes::<f32>(
            logical_rows.checked_mul(4).ok_or_else(|| {
                NyError::InvalidSpec(
                    "Anchored BatchNorm error/published-bias count overflows usize".into(),
                )
            })?,
            "Anchored BatchNorm error and published bias",
        )?,
    ])
}

#[inline]
fn anchored_bn_flush_charge(value: f32) -> f64 {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < 0x0080_0000 {
        f32_to_f64_exact(value).abs()
    } else {
        0.0
    }
}

#[inline]
fn anchored_bn_normalize_center(value: f32) -> f32 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude != 0 && magnitude < 0x0080_0000 {
        f32::from_bits(bits & 0x8000_0000)
    } else {
        value
    }
}

#[inline]
fn checked_anchored_bn_tap_index(
    oh: usize,
    ow: usize,
    ic: usize,
    ki: usize,
    kj: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) -> Result<usize> {
    oh.checked_mul(out_w)
        .and_then(|value| value.checked_add(ow))
        .and_then(|value| value.checked_mul(in_c))
        .and_then(|value| value.checked_add(ic))
        .and_then(|value| value.checked_mul(kh))
        .and_then(|value| value.checked_add(ki))
        .and_then(|value| value.checked_mul(kw))
        .and_then(|value| value.checked_add(kj))
        .ok_or_else(|| NyError::InvalidSpec("Anchored BatchNorm tap index overflows usize".into()))
}

#[inline]
fn checked_anchored_bn_output_index(
    oc: usize,
    oh: usize,
    ow: usize,
    spatial_positions: usize,
    out_w: usize,
) -> Result<usize> {
    oc.checked_mul(spatial_positions)
        .and_then(|value| oh.checked_mul(out_w).and_then(|row| value.checked_add(row)))
        .and_then(|value| value.checked_add(ow))
        .ok_or_else(|| {
            NyError::InvalidSpec("Anchored BatchNorm output-row index overflows usize".into())
        })
}

#[inline]
fn anchored_bn_publish_error(value: f64) -> f32 {
    if value.is_nan() || value < 0.0 || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }
    let published = next_up_f32(value as f32);
    let magnitude = published.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < 0x0080_0000 {
        f32::MIN_POSITIVE
    } else {
        published
    }
}

/// Patches-mode CROWN backward through BatchNorm.
///
/// BatchNorm is a per-channel linear operation: y[c,h,w] = scale[c] * x[c,h,w] + bias[c].
/// In the Patches representation [oc, oh, ow, ic, ki, kj], the `ic` dimension directly
/// corresponds to the BatchNorm channel, so backward is simple:
///   - Scale each coefficient by scale[ic]
///   - Add bias contribution to the output bias vectors
///
/// No upper/lower swap needed: CROWN backward composes by substitution (exact linear
/// layer), not IBP. Negative scale just flips coefficient sign.
///
/// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md Phase 2
/// Part of #2613
impl crate::layers::common::PatchesPropagation for BatchNormLayer {
    fn propagate_patches(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        let mut deadline = crate::bounds::patches::PatchesMaterializationDeadline::new(None);
        self.propagate_patches_impl(bounds, &mut deadline, false)
    }
}

impl BatchNormLayer {
    pub(crate) fn propagate_patches_with_deadline(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        deadline: Instant,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        let mut deadline =
            crate::bounds::patches::PatchesMaterializationDeadline::new(Some(deadline));
        self.validate_affine_parameters_with_poll(&mut deadline)?;
        bounds
            .lower_a
            .validate_common_geometry_with_poll(&bounds.upper_a, &mut deadline)?;
        if !matches!(
            &bounds.lower_a.geometry,
            crate::bounds::patches::PatchGeometry::Anchored(_)
        ) {
            return Err(NyError::UnsupportedConfiguration(
                "finite-deadline BatchNorm Patches is implemented for Anchored geometry only"
                    .into(),
            ));
        }
        self.propagate_patches_impl(bounds, &mut deadline, true)
    }

    fn validate_affine_parameters_with_poll(
        &self,
        deadline: &mut crate::bounds::patches::PatchesMaterializationDeadline,
    ) -> Result<()> {
        let expected_shape = [self.num_channels];
        if self.num_channels == 0
            || self.scale.shape() != expected_shape
            || self.bias.shape() != expected_shape
            || self.scale_err.shape() != expected_shape
            || self.bias_err.shape() != expected_shape
        {
            return Err(NyError::InvalidSpec(format!(
                "BatchNorm: affine vectors must all have shape [{0}] with {0} > 0",
                self.num_channels
            )));
        }
        for (&scale, &bias) in self.scale.iter().zip(self.bias.iter()) {
            deadline.work(1, "while validating BatchNorm affine values")?;
            if !scale.is_finite() || !bias.is_finite() {
                return Err(NyError::InvalidSpec(
                    "BatchNorm: affine scale and bias must be finite".to_string(),
                ));
            }
        }
        for (&scale_error, &bias_error) in self.scale_err.iter().zip(self.bias_err.iter()) {
            deadline.work(1, "while validating BatchNorm affine error radii")?;
            if !scale_error.is_finite()
                || scale_error < 0.0
                || !bias_error.is_finite()
                || bias_error < 0.0
            {
                return Err(NyError::InvalidSpec(
                    "BatchNorm: affine error radii must be finite and non-negative".to_string(),
                ));
            }
        }
        match self.channel_axis_hint {
            Some(BatchNormChannelAxisHint::Fixed(axis)) if axis > 1 => {
                return Err(NyError::InvalidSpec(format!(
                    "BatchNorm: unsupported channel-axis hint {axis}; expected 0 or 1"
                )));
            }
            Some(BatchNormChannelAxisHint::OnnxNchw { authored_rank }) if authored_rank < 2 => {
                return Err(NyError::InvalidSpec(format!(
                    "BatchNorm: ONNX NCHW provenance requires authored rank at least 2, got {authored_rank}"
                )));
            }
            _ => {}
        }
        deadline.checkpoint("after validating BatchNorm affine parameters")
    }

    fn propagate_patches_impl(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        deadline: &mut crate::bounds::patches::PatchesMaterializationDeadline,
        geometry_prevalidated: bool,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        if !geometry_prevalidated {
            self.validate_affine_parameters()?;
        }
        use crate::bounds::patches::{
            CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds,
        };

        deadline.checkpoint("before BatchNorm Patches geometry authentication")?;
        if !geometry_prevalidated {
            bounds
                .lower_a
                .validate_common_geometry_with_poll(&bounds.upper_a, deadline)?;
        }
        if matches!(&bounds.lower_a.geometry, PatchGeometry::Anchored(_)) {
            return self.propagate_anchored_patches(bounds, deadline);
        }
        let affine_geometry = bounds
            .lower_a
            .geometry
            .require_affine("BatchNorm Patches backward")?;
        let row_count = bounds.row_count;

        let process_patches = |patches_data: &PatchesData,
                               bias_vec: &Array1<f32>,
                               direction: OutwardDirection|
         -> Result<(PatchesData, Array1<f64>, Array1<f64>)> {
            let (out_c, out_h, out_w) = patches_data.output_shape;
            let (in_c, _in_h, _in_w) = patches_data.input_shape;

            // Channel count must match BatchNorm num_channels
            if in_c != self.num_channels {
                return Err(NyError::ShapeMismatch {
                    expected: vec![self.num_channels],
                    got: vec![in_c],
                });
            }

            let mut new_bias = bias_vec.mapv(f32_to_f64_exact);

            // Determine the patches layout for the non-identity case so the bias-length
            // guard matches the destination index used below.
            //   - rank-6 [oc, oh, ow, ic, ki, kj]   : bias is per patches output neuron
            //     (`n = oc*out_h*out_w + oh*out_w + ow`), so bias len must equal the
            //     output-neuron count.
            //   - rank-7 [row, oc, oh, ow, ic, ki, kj] (EXPLICIT-ROWS, e.g. the 1x1 conv
            //     re-entry on a rank-3 spatial spec): bias is per spec-row, so bias len
            //     must equal `row_count`.
            // Identity patches always use the per-neuron layout below, so treat them as
            // rank-6 for the guard.
            // For some specs (e.g. cgan's disjunctive multi-clause input split, where
            // the spec/bias vector is per spec-row, not per layer neuron) the counts
            // differ: a shorter bias panics with an ndarray index-out-of-bounds, and a
            // longer bias would leave trailing rows without their BatchNorm bias
            // contribution (silently unsound). In either mismatch, return an error so
            // `try_patches_or_dense_fallback` drops to the dense BatchNorm backward,
            // which handles arbitrary spec layouts exactly. SOUND: dense is exact;
            // patches and dense agree when the layout matches (the common case, so no
            // perf regression there).
            let explicit_rows = if patches_data.identity {
                false
            } else {
                let shape = patches_data
                    .patches
                    .as_ref()
                    .ok_or_else(|| {
                        NyError::InternalError(
                            "PatchesData: not identity but patches tensor is None".into(),
                        )
                    })?
                    .shape()
                    .to_vec();
                match shape.len() {
                    6 => false,
                    7 => {
                        if shape[0] != row_count {
                            return Err(NyError::ShapeMismatch {
                                expected: vec![row_count],
                                got: vec![shape[0]],
                            });
                        }
                        true
                    }
                    _ => {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![6, 7],
                            got: vec![shape.len()],
                        });
                    }
                }
            };

            if explicit_rows {
                // Explicit-rows: bias is per spec-row.
                if row_count != new_bias.len() {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![row_count],
                        got: vec![new_bias.len()],
                    });
                }
            } else {
                let out_neuron_count = out_c
                    .checked_mul(out_h)
                    .and_then(|x| x.checked_mul(out_w))
                    .ok_or_else(|| {
                        NyError::InternalError(
                            "BatchNorm patches: output-neuron count overflow".into(),
                        )
                    })?;
                if out_neuron_count != new_bias.len() {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![out_neuron_count],
                        got: vec![new_bias.len()],
                    });
                }
            }

            if patches_data.identity {
                // Identity patches: each output (oc, oh, ow) maps to itself with coeff 1.0.
                // In identity mode, the coefficient for output neuron (oc, oh, ow)
                // references input neuron at the same position with channel oc.
                // But oc corresponds to the output channel of the previous conv,
                // which IS the BatchNorm's input channel. So:
                //   new_coeff = 1.0 * scale[oc]  (the identity coeff times BN scale)
                //   delta_bias = 1.0 * bias[oc]   (one coefficient per output neuron)
                //
                // After scaling, this is no longer identity — materialize.
                // Actually, we can represent this as a materialized 6D tensor
                // where patches[oc,oh,ow,oc,0,0] = scale[oc] (only diagonal ic=oc).
                let mut patches_arr =
                    ArrayD::<f32>::zeros(IxDyn(&[out_c, out_h, out_w, in_c, 1, 1]));
                for oc in 0..out_c.min(in_c) {
                    let s = self.scale[[oc]];
                    let b = f32_to_f64_exact(self.bias[[oc]]);
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            patches_arr[[oc, oh, ow, oc, 0, 0]] = s;
                            let n = oc * out_h * out_w + oh * out_w + ow;
                            // Each identity position contributes 1.0 * bias[oc]
                            new_bias[n] += b;
                        }
                    }
                }
                // Certified error channel (#patches-coeff-err-soundness). The
                // materialized diagonal coefficient is scale[oc] (a direct f32
                // assignment), whose gap to the real BN scale is scale_err[oc].
                // HOLE1: the folded diagonal bias is bias[oc], gap bias_err[oc].
                // Identity input carries no coefficient error (old_err defaults to
                // 0). This is the per-neuron 6D dense layout; a sparse
                // (unstable_idx Some) identity is not wired for the dense to_dense
                // err scatter, so keep the channel None + zero discharge there
                // (prior behavior, no regression).
                let (coeff_err, widen): (Option<Array1<f32>>, Array1<f64>) =
                    if patches_data.unstable_idx.is_some() {
                        (None, Array1::<f64>::zeros(new_bias.len()))
                    } else {
                        let rnd = crate::layers::linear::crown_single_gamma_n_f32(1);
                        let old = patches_data.coeff_err.as_ref();
                        let mut ne = Array1::<f32>::zeros(new_bias.len());
                        let mut wd = Array1::<f64>::zeros(new_bias.len());
                        for oc in 0..out_c.min(in_c) {
                            let s = f64::from(self.scale[[oc]]).abs();
                            let se = f64::from(self.scale_err[[oc]]);
                            let bb = f64::from(self.bias[[oc]]).abs();
                            let be = f64::from(self.bias_err[[oc]]);
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    let n = oc * out_h * out_w + oh * out_w + ow;
                                    let oe = old.map_or(0.0, |e| {
                                        f64::from(e.get(n).copied().unwrap_or(0.0))
                                    });
                                    ne[n] = next_up_f32((rnd * s + se + (s + se) * oe) as f32);
                                    wd[n] = oe * bb + be;
                                }
                            }
                        }
                        (Some(ne), wd)
                    };
                let new_data = PatchesData {
                    coeff_err,
                    patches: Some(patches_arr),
                    geometry: patches_data.geometry.clone(),
                    identity: false,
                    output_shape: patches_data.output_shape,
                    input_shape: patches_data.input_shape,
                    unstable_idx: None,
                };
                return Ok((new_data, new_bias, widen));
            }

            // Non-identity: scale existing patches coefficients by scale[ic]
            // and accumulate bias contributions.
            let patches = patches_data.patches.as_ref().ok_or_else(|| {
                NyError::InternalError(
                    "PatchesData: not identity but patches tensor is None".into(),
                )
            })?;
            let shape = patches.shape();
            // kh/kw are the trailing kernel axes. For the EXPLICIT-ROWS (rank-7)
            // layout [row, oc, oh, ow, ic, ki, kj] they are axes 5/6; for the rank-6
            // layout [oc, oh, ow, ic, ki, kj] they are axes 4/5.
            let (kh, kw) = if explicit_rows {
                (shape[5], shape[6])
            } else {
                (shape[4], shape[5])
            };

            let mut new_patches = patches.clone();

            // Extract stride/padding for bounds checking.
            // Padding-zone positions map to virtual zero-input — their coefficients
            // are correctly dropped by to_dense(), but we must also exclude them
            // from the bias sum. Reference: PatchesData::scatter_patches_to_dense
            // (bounds/patches.rs:273-278).
            let (sh, sw) = affine_geometry.stride();
            let (pad_left, _pad_right, pad_top, _pad_bottom) = affine_geometry.padding();
            let (_in_c, in_h, in_w) = patches_data.input_shape;

            // Scale coefficients by per-channel scale and accumulate bias.
            // Coefficient scaling applies to ALL positions (padding-zone coefficients
            // stay dead through to_dense()). Bias accumulation only includes valid
            // (non-padding) positions. The coefficient transform (`new = coeff * scale[ic]`)
            // and the padding-zone-excluded bias contribution (`Σ_valid(coeff) * bias[ic]`)
            // are identical in both arms; only the index arity and the bias destination
            // differ (per spec-row for rank-7 vs per output neuron for rank-6), matching
            // the audited ReLU/elementwise and conv compute_patches_bias handlers.
            if explicit_rows {
                for row in 0..row_count {
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let mut bias_accum = 0.0_f64;
                                for ic in 0..in_c {
                                    let s = self.scale[[ic]];
                                    let b = self.bias[[ic]] as f64;
                                    let mut channel_sum = 0.0_f64;
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let coeff = patches[[row, oc, oh, ow, ic, ki, kj]];
                                            new_patches[[row, oc, oh, ow, ic, ki, kj]] = coeff * s;
                                            // Only include valid (non-padding) positions in bias sum
                                            let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                            let iw_raw =
                                                (ow * sw + kj) as isize - pad_left as isize;
                                            if ih_raw >= 0
                                                && (ih_raw as usize) < in_h
                                                && iw_raw >= 0
                                                && (iw_raw as usize) < in_w
                                            {
                                                channel_sum += coeff as f64;
                                            }
                                        }
                                    }
                                    bias_accum += channel_sum * b;
                                }
                                new_bias[row] += bias_accum;
                            }
                        }
                    }
                }
            } else {
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let n = oc * out_h * out_w + oh * out_w + ow;
                            let mut bias_terms = Vec::with_capacity(in_c * kh * kw);
                            for ic in 0..in_c {
                                let s = self.scale[[ic]];
                                let b = self.bias[[ic]];
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let coeff = patches[[oc, oh, ow, ic, ki, kj]];
                                        new_patches[[oc, oh, ow, ic, ki, kj]] = coeff * s;
                                        // Only include valid (non-padding) positions in bias sum
                                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                        if ih_raw >= 0
                                            && (ih_raw as usize) < in_h
                                            && iw_raw >= 0
                                            && (iw_raw as usize) < in_w
                                        {
                                            bias_terms.push((coeff, b));
                                        }
                                    }
                                }
                            }
                            new_bias[n] =
                                certified_affine_sum_f32(bias_vec[n], bias_terms, direction);
                        }
                    }
                }
            }

            // Certified error channel (#patches-coeff-err-soundness). BN substitutes
            // z = scale·x + bn_bias; the coefficient on x scales by scale[ic]. With
            // rnd = γ_1 = u/(1-u) >= 2^-24 (sound single-f32-multiply rounding
            // factor), gain = max_c(|scale[c]|+scale_err[c]) and c(j) = ic:
            //   new_err[i] = next_up( max_j(rnd·|new_coeff[i,j]| + |coeff[i,j]|·scale_err[c(j)])
            //                         + gain·old_err[i] ).
            // HOLE1: the folded bias Σ_valid coeff·bn_bias picks up
            //   widen[i] = Σ_valid (|coeff|·bias_err
            //                              + old_err·(|bn_bias|+bias_err))
            // over the SAME valid (non-padding) taps as the bias loop; discharged
            // outward by the caller (lower_b down, upper_b up).
            //
            // Layout dispatch (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §8): ONLY the
            // sparse layout (unstable_idx Some) stays None + zero discharge — the
            // sparse to_dense err scatter is out of scope and hard-guarded
            // downstream, and checking sparseness FIRST keeps that true even for a
            // hypothetical sparse rank-7 tensor.
            //
            // 7D explicit-rows arm: the err index is the SPEC row (axis 0, length
            // row_count == bias length, invariant I1): MAX-lift of the per-tap
            // coefficient bracket terms over the whole row (one scalar must cover
            // every coefficient of the row) and SUM-lift of the bias-widen terms
            // (every position folds into the ONE spec-row bias slot). Both dense
            // layouts carry the `oe·(bb+be)` cross term required by exact algebra
            // (gap = -(a·β + α·bias + α·β), |α|<=oe, |β|<=be; spec R5).
            // Additionally, per lead adjudication A1 the 7D f64 bias fold's own accumulation
            //       rounding is discharged outright: gbar·ABS with
            //       gbar = γ^f64(8·row_volume+16),
            //       ABS = |b[row]| + Σ_valid |a|·|bn_bias[ic]|, and the carried widen
            //       sum is inflated by (1+gbar) to cover its own f64
            //       nearest-summation under-estimate (gbar has >= 4x headroom over
            //       the actual γ of both folds; saturates to +INF -> outward poison).
            // Hard guards (I5/I6): a Some old err whose len != row_count is
            // Err(ShapeMismatch) => the caller's sound dense-BN fallback; a
            // non-finite or negative old_err[row] poisons the row (+INF err, +INF
            // widen, so the caller's discharge yields a -INF/+INF vacuous bias),
            // NEVER NaN; every 0·INF product is short-circuited.
            //
            // 6D dense arm: the affine bias fold uses the shared certified
            // cancellation-safe reduction, while its nonnegative coefficient-
            // error/widen reductions round every binary64 operation upward.
            let (coeff_err, widen): (Option<Array1<f32>>, Array1<f64>) = if patches_data
                .unstable_idx
                .is_some()
            {
                (None, Array1::<f64>::zeros(new_bias.len()))
            } else if explicit_rows {
                let rnd = crate::layers::linear::crown_single_gamma_n_f32(1);
                let mut gain = 0.0f64;
                for c in 0..in_c {
                    let g = f64::from(self.scale[[c]]).abs() + f64::from(self.scale_err[[c]]);
                    if g > gain {
                        gain = g;
                    }
                }
                let old = patches_data.coeff_err.as_ref();
                if let Some(e) = old {
                    if e.len() != row_count {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![row_count],
                            got: vec![e.len()],
                        });
                    }
                }
                // A1 fold-discharge factor. `row_volume` = per-row tap count;
                // the value bias fold performs <= 4·row_volume + 4 f64 nearest
                // roundings per row, each |θ| <= gbar := γ^f64(8·row_volume+16)
                // (>= 4x headroom, which also absorbs the f64 under-estimates
                // of ABS and of the carried widen sum, plus the final
                // combination roundings). Accepted regime (lead E3/F3): row
                // addend count n < 2^28 — cifar-scale rows are ~4e6, 60x under
                // it; beyond, gbar merely grows (saturating to +INF => outward
                // poison), still sound.
                let row_volume = out_c
                    .checked_mul(out_h)
                    .and_then(|x| x.checked_mul(out_w))
                    .and_then(|x| x.checked_mul(in_c))
                    .and_then(|x| x.checked_mul(kh))
                    .and_then(|x| x.checked_mul(kw))
                    .unwrap_or(usize::MAX);
                debug_assert!(
                    row_volume < (1usize << 28),
                    "BN 7D err pass: row addend count {row_volume} exceeds the \
                         documented n < 2^28 regime (still sound: gbar only grows)"
                );
                let gbar = crate::layers::linear::crown_single_gamma_n_f64(
                    row_volume.saturating_mul(8).saturating_add(16),
                );
                let mut ne = Array1::<f32>::zeros(new_bias.len());
                let mut wd = Array1::<f64>::zeros(new_bias.len());
                for row in 0..row_count {
                    // Direct index — length validated above (never the 6D-style
                    // silent `.get(i).unwrap_or(0.0)`, spec I6/R6).
                    let oe = old.map_or(0.0, |e| f64::from(e[row]));
                    if !oe.is_finite() || oe < 0.0 {
                        // Poison the row outward (I5): +INF err (degrades at
                        // consumption), +INF widen (the caller's discharge gives
                        // -INF lower / +INF upper — vacuous, NaN-free since the
                        // folded bias is finite-or-INF, never matched against a
                        // 0 factor). Skip the accumulation entirely so INF never
                        // meets a 0 multiplicand (e.g. bb + be == 0).
                        ne[row] = f32::INFINITY;
                        wd[row] = f64::INFINITY;
                        continue;
                    }
                    let mut cast = 0.0f64;
                    let mut wsum = 0.0f64;
                    // ABS is initialized with the incoming |bias|: the value
                    // fold's f64 `+=` chain starts from it, so Higham's
                    // Σ|addends| must include it.
                    let mut abs_sum = f64::from(bias_vec[row]).abs();
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                for ic in 0..in_c {
                                    let se = f64::from(self.scale_err[[ic]]);
                                    let be = f64::from(self.bias_err[[ic]]);
                                    let bb = f64::from(self.bias[[ic]]).abs();
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let coeff =
                                                f64::from(patches[[row, oc, oh, ow, ic, ki, kj]])
                                                    .abs();
                                            let ncoeff = f64::from(
                                                new_patches[[row, oc, oh, ow, ic, ki, kj]],
                                            )
                                            .abs();
                                            // Coefficient bracket term: MAX over
                                            // ALL taps of the row, padding-zone
                                            // taps INCLUDED (they are scaled too
                                            // and only die at to_dense()), same
                                            // as the 6D arm.
                                            let t = rnd * ncoeff + coeff * se;
                                            if t > cast {
                                                cast = t;
                                            }
                                            // HOLE1: only valid (non-padding)
                                            // taps fold into the bias, mirroring
                                            // the bias loop above.
                                            let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                            let iw_raw =
                                                (ow * sw + kj) as isize - pad_left as isize;
                                            if ih_raw >= 0
                                                && (ih_raw as usize) < in_h
                                                && iw_raw >= 0
                                                && (iw_raw as usize) < in_w
                                            {
                                                // Per-tap exact algebra incl. the
                                                // oe·be cross term (spec R5):
                                                // |a·b - a_true·b_real|
                                                //   <= |a|·be + oe·(bb + be).
                                                // oe == 0 short-circuits so a
                                                // degenerate +INF channel bias
                                                // cannot make 0·INF = NaN (I5).
                                                let cross =
                                                    if oe == 0.0 { 0.0 } else { oe * (bb + be) };
                                                wsum += coeff * be + cross;
                                                // A zero stored coefficient
                                                // contributes exactly 0 to the
                                                // value fold; skip it so 0·INF
                                                // (degenerate bb) cannot poison
                                                // ABS with NaN (I5).
                                                if coeff != 0.0 {
                                                    abs_sum += coeff * bb;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // oe == 0 short-circuit: gain can be +INF (degenerate scale
                    // channel) and 0·INF = NaN (I5).
                    let ne_val = if oe == 0.0 { cast } else { cast + gain * oe };
                    let ne_f32 = next_up_f32(ne_val as f32);
                    // Defensive: the err channel is "finite or +INF", never NaN.
                    // (A NaN stored coefficient — the pre-existing value-path
                    // 0·INF — is skipped by the max above since NaN comparisons
                    // are false; keep the emission NaN-free regardless.)
                    ne[row] = if ne_f32.is_nan() {
                        f32::INFINITY
                    } else {
                        ne_f32
                    };
                    // Widen: carried terms inflated by (1+gbar) (covers wsum's
                    // own f64 summation under-estimate) + the A1 fold discharge
                    // gbar·ABS. Zero operands are short-circuited before the
                    // possibly-saturated (+INF) gbar so the correct limit of a
                    // zero sum stays 0 — pure carry, adjudication C2 analog.
                    let carried = if wsum == 0.0 {
                        0.0
                    } else {
                        wsum * (1.0 + gbar)
                    };
                    let fold = if abs_sum == 0.0 { 0.0 } else { gbar * abs_sum };
                    let w = carried + fold;
                    // Residual non-finite/negative (NaN compares false) maps to
                    // +INF: outward poison, never NaN (I5).
                    wd[row] = if w >= 0.0 { w } else { f64::INFINITY };
                }
                (Some(ne), wd)
            } else {
                let rnd = crate::layers::linear::crown_single_gamma_n_f32(1);
                let mut gain = 0.0f64;
                for c in 0..in_c {
                    let g = nonnegative_add_up(
                        f32_to_f64_exact(self.scale[[c]]).abs(),
                        f32_to_f64_exact(self.scale_err[[c]]),
                    );
                    if g > gain {
                        gain = g;
                    }
                }
                let old = patches_data.coeff_err.as_ref();
                if let Some(error) = old {
                    if error.len() != new_bias.len() {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![new_bias.len()],
                            got: vec![error.len()],
                        });
                    }
                }
                let mut ne = Array1::<f32>::zeros(new_bias.len());
                let mut wd = Array1::<f64>::zeros(new_bias.len());
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let n = oc * out_h * out_w + oh * out_w + ow;
                            let oe = old.map_or(0.0, |e| {
                                f32_to_f64_exact(e.get(n).copied().unwrap_or(0.0))
                            });
                            let mut cast = 0.0f64;
                            let mut wsum = 0.0f64;
                            for ic in 0..in_c {
                                let se = f32_to_f64_exact(self.scale_err[[ic]]);
                                let be = f32_to_f64_exact(self.bias_err[[ic]]);
                                let bb = f32_to_f64_exact(self.bias[[ic]]).abs();
                                let true_bias_magnitude = nonnegative_add_up(bb, be);
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let coeff =
                                            f32_to_f64_exact(patches[[oc, oh, ow, ic, ki, kj]])
                                                .abs();
                                        let ncoeff =
                                            f32_to_f64_exact(new_patches[[oc, oh, ow, ic, ki, kj]])
                                                .abs();
                                        let t = nonnegative_add_up(
                                            nonnegative_mul_up(rnd, ncoeff),
                                            nonnegative_mul_up(coeff, se),
                                        );
                                        if t > cast {
                                            cast = t;
                                        }
                                        // HOLE1: only valid (non-padding) taps
                                        // fold into the bias, mirroring the bias
                                        // loop above.
                                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                        if ih_raw >= 0
                                            && (ih_raw as usize) < in_h
                                            && iw_raw >= 0
                                            && (iw_raw as usize) < in_w
                                        {
                                            let term = nonnegative_add_up(
                                                nonnegative_mul_up(coeff, be),
                                                nonnegative_mul_up(oe, true_bias_magnitude),
                                            );
                                            wsum = nonnegative_add_up(wsum, term);
                                        }
                                    }
                                }
                            }
                            let ne_value = nonnegative_add_up(cast, nonnegative_mul_up(gain, oe));
                            ne[n] = next_up_f32(ne_value as f32);
                            wd[n] = wsum;
                        }
                    }
                }
                (Some(ne), wd)
            };
            let new_data = PatchesData {
                coeff_err,
                patches: Some(new_patches),
                geometry: patches_data.geometry.clone(),
                identity: false,
                output_shape: patches_data.output_shape,
                input_shape: patches_data.input_shape,
                unstable_idx: None,
            };
            // HOLE1 bias discharge (`widen`) + directed rounding (#1745) are applied
            // by the caller (lower_b -= widen then round down, upper_b += widen up).
            Ok((new_data, new_bias, widen))
        };

        let (new_lower_a, new_lower_b, widen_lower) =
            process_patches(&bounds.lower_a, &bounds.lower_b, OutwardDirection::Lower)?;
        let (new_upper_a, new_upper_b, widen_upper) =
            process_patches(&bounds.upper_a, &bounds.upper_b, OutwardDirection::Upper)?;

        // Discharge the HOLE1 bias widening OUTWARD, then directed rounding (#1745):
        // lower bounds subtract widen and round down, upper bounds add widen and
        // round up. Each branch uses its own path's widen (lower_a's vs upper_a's
        // coeffs/old_err). widen >= 0, so this only ever loosens the bound; the
        // single f64->f32 cast per cell preserves the #1745 soundness.
        let new_lower_b = ndarray::Zip::from(&new_lower_b)
            .and(&widen_lower)
            .map_collect(|&b, &w| next_down_f32((b - w) as f32));
        let new_upper_b = ndarray::Zip::from(&new_upper_b)
            .and(&widen_upper)
            .map_collect(|&b, &w| next_up_f32((b + w) as f32));

        Ok(CrownBounds::Patches(Box::new(PatchesLinearBounds {
            row_count: bounds.row_count,
            lower_a: new_lower_a,
            lower_b: new_lower_b,
            upper_a: new_upper_a,
            upper_b: new_upper_b,
        })))
    }

    fn propagate_anchored_patches(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        deadline: &mut crate::bounds::patches::PatchesMaterializationDeadline,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};
        use crate::bounds::safe_math::safe_mul_for_bounds_f64;
        use ny_core::dd::next_up_f64;

        let mut poll = || deadline.checkpoint("during Anchored BatchNorm Patches backward");
        poll()?;

        if bounds.lower_a.identity
            || bounds.upper_a.identity
            || bounds.lower_a.unstable_idx.is_some()
            || bounds.upper_a.unstable_idx.is_some()
        {
            return Err(NyError::UnsupportedConfiguration(
                "Anchored BatchNorm Patches requires materialized dense 6D/7D carriers".into(),
            ));
        }

        let lower_patches = bounds.lower_a.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("Anchored lower PatchesData has no coefficient tensor".into())
        })?;
        let upper_patches = bounds.upper_a.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("Anchored upper PatchesData has no coefficient tensor".into())
        })?;
        if lower_patches.shape() != upper_patches.shape() {
            return Err(NyError::ShapeMismatch {
                expected: lower_patches.shape().to_vec(),
                got: upper_patches.shape().to_vec(),
            });
        }
        let lower_slice = lower_patches.as_slice().ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "Anchored BatchNorm Patches requires contiguous coefficients".into(),
            )
        })?;
        let upper_slice = upper_patches.as_slice().ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "Anchored BatchNorm Patches requires contiguous coefficients".into(),
            )
        })?;

        let (out_c, out_h, out_w) = bounds.lower_a.output_shape;
        let (in_c, in_h, in_w) = bounds.lower_a.input_shape;
        if in_c != self.num_channels {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.num_channels],
                got: vec![in_c],
            });
        }
        let shape = lower_patches.shape();
        let explicit_rows = match shape.len() {
            6 => false,
            7 => {
                if shape[0] != bounds.row_count {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![bounds.row_count],
                        got: vec![shape[0]],
                    });
                }
                true
            }
            rank => {
                return Err(NyError::ShapeMismatch {
                    expected: vec![6, 7],
                    got: vec![rank],
                });
            }
        };
        let (kh, kw) = if explicit_rows {
            (shape[5], shape[6])
        } else {
            (shape[4], shape[5])
        };
        // Authenticate metadata products before expected-shape construction
        // or allocation planning. A malformed huge carrier must refuse while
        // the borrowed source is still untouched.
        let spatial_positions = out_h.checked_mul(out_w).ok_or_else(|| {
            NyError::InvalidSpec("Anchored BatchNorm spatial-position count overflows usize".into())
        })?;
        let output_positions = out_c.checked_mul(spatial_positions).ok_or_else(|| {
            NyError::InvalidSpec("Anchored BatchNorm output-position count overflows usize".into())
        })?;
        let expected_shape = if explicit_rows {
            vec![bounds.row_count, out_c, out_h, out_w, in_c, kh, kw]
        } else {
            vec![out_c, out_h, out_w, in_c, kh, kw]
        };
        if shape != expected_shape.as_slice() {
            return Err(NyError::ShapeMismatch {
                expected: expected_shape,
                got: shape.to_vec(),
            });
        }
        let logical_rows = if explicit_rows {
            bounds.row_count
        } else {
            if bounds.row_count != output_positions {
                return Err(NyError::ShapeMismatch {
                    expected: vec![output_positions],
                    got: vec![bounds.row_count],
                });
            }
            output_positions
        };
        if bounds.lower_b.len() != logical_rows || bounds.upper_b.len() != logical_rows {
            return Err(NyError::ShapeMismatch {
                expected: vec![logical_rows, logical_rows],
                got: vec![bounds.lower_b.len(), bounds.upper_b.len()],
            });
        }
        for error in [
            bounds.lower_a.coeff_err.as_ref(),
            bounds.upper_a.coeff_err.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if error.len() != logical_rows {
                return Err(NyError::ShapeMismatch {
                    expected: vec![logical_rows],
                    got: vec![error.len()],
                });
            }
        }

        let patch_volume = in_c
            .checked_mul(kh)
            .and_then(|value| value.checked_mul(kw))
            .ok_or_else(|| {
                NyError::InvalidSpec("Anchored BatchNorm tap count overflows usize".into())
            })?;
        let patch_elements = if explicit_rows {
            bounds
                .row_count
                .checked_mul(output_positions)
                .and_then(|value| value.checked_mul(patch_volume))
        } else {
            output_positions.checked_mul(patch_volume)
        }
        .ok_or_else(|| {
            NyError::InvalidSpec("Anchored BatchNorm coefficient count overflows usize".into())
        })?;
        if lower_slice.len() != patch_elements || upper_slice.len() != patch_elements {
            return Err(NyError::ShapeMismatch {
                expected: vec![patch_elements, patch_elements],
                got: vec![lower_slice.len(), upper_slice.len()],
            });
        }
        let map_len = out_h
            .checked_mul(out_w)
            .and_then(|value| value.checked_mul(patch_volume))
            .ok_or_else(|| {
                NyError::InvalidSpec("Anchored BatchNorm tap-map count overflows usize".into())
            })?;

        // Total-live receipt: the borrowed source remains resident throughout
        // the transaction, including A/b/coeff_err and Arc-backed geometry.
        // `memory_bytes` conservatively charges shared geometry once per side.
        let planned_allocation_bytes =
            anchored_bn_planned_bytes(map_len, patch_elements, logical_rows)?;
        let source_resident_bytes = bounds.memory_bytes();
        let mut admission =
            AnchoredBatchNormAdmission::new(source_resident_bytes, planned_allocation_bytes)?;

        let mut valid_map = admission.zeroed(
            map_len,
            false,
            "Anchored BatchNorm tap-map allocation",
            &mut poll,
        )?;
        let mut map_cursor = 0usize;
        for oh in 0..out_h {
            for ow in 0..out_w {
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            if map_cursor.is_multiple_of(ANCHORED_BN_POLL_COORDS) {
                                poll()?;
                            }
                            valid_map[map_cursor] = bounds
                                .lower_a
                                .geometry
                                .input_flat_index((oh, ow), ic, (ki, kj), (in_c, in_h, in_w))?
                                .is_some();
                            map_cursor = map_cursor.checked_add(1).ok_or_else(|| {
                                NyError::InvalidSpec(
                                    "Anchored BatchNorm tap-map cursor overflows usize".into(),
                                )
                            })?;
                        }
                    }
                }
            }
        }
        if map_cursor != map_len {
            return Err(NyError::InternalError(format!(
                "Anchored BatchNorm tap-map length {map_cursor} differs from planned {map_len}"
            )));
        }
        poll()?;

        let mut gain = 0.0f64;
        for ic in 0..in_c {
            if ic.is_multiple_of(ANCHORED_BN_POLL_COORDS) {
                poll()?;
            }
            let candidate = nonnegative_add_up(
                f32_to_f64_exact(self.scale[[ic]]).abs(),
                f32_to_f64_exact(self.scale_err[[ic]]),
            );
            if candidate > gain {
                gain = candidate;
            }
        }

        let ((new_lower_a, lower_bias64, lower_widen), (new_upper_a, upper_bias64, upper_widen)) = {
            let mut process_side = |patches_data: &PatchesData,
                                    source: &[f32],
                                    bias: &Array1<f32>,
                                    direction: OutwardDirection|
             -> Result<(PatchesData, Array1<f64>, Array1<f64>)> {
                let mut new_values = admission
                    .reserve::<f32>(patch_elements, "Anchored BatchNorm coefficient allocation")?;
                let mut new_bias_values = admission
                    .reserve::<f64>(logical_rows, "Anchored BatchNorm f64 bias allocation")?;
                let mut new_error_values = admission.reserve::<f32>(
                    logical_rows,
                    "Anchored BatchNorm coefficient-error allocation",
                )?;
                let mut widen_values = admission
                    .reserve::<f64>(logical_rows, "Anchored BatchNorm bias-widen allocation")?;
                let old_error = patches_data.coeff_err.as_ref();
                let mut source_cursor = 0usize;
                let mut work = 0usize;

                let mut consume_tap = |oh: usize,
                                       ow: usize,
                                       ic: usize,
                                       ki: usize,
                                       kj: usize,
                                       oe: f64,
                                       bias_accum: &mut f64,
                                       coefficient_error: &mut f64,
                                       widen: &mut f64,
                                       poll_tap: &mut dyn FnMut() -> Result<()>|
                 -> Result<()> {
                    if work.is_multiple_of(ANCHORED_BN_POLL_COORDS) {
                        poll_tap()?;
                    }
                    work = work.saturating_add(1);
                    let coeff = *source.get(source_cursor).ok_or_else(|| {
                    NyError::InternalError(format!(
                        "Anchored BatchNorm source cursor {source_cursor} exceeds coefficient length {}",
                        source.len()
                    ))
                })?;
                    source_cursor = source_cursor.checked_add(1).ok_or_else(|| {
                        NyError::InvalidSpec(
                            "Anchored BatchNorm source cursor overflows usize".into(),
                        )
                    })?;
                    let scale = self.scale[[ic]];
                    let stored = anchored_bn_normalize_center(coeff * scale);
                    new_values.push(stored);

                    let coeff64 = f32_to_f64_exact(coeff);
                    let scale64 = f32_to_f64_exact(scale);
                    let stored64 = f32_to_f64_exact(stored);
                    let exact_product = coeff64 * scale64;
                    let raw_gap = (exact_product - stored64).abs();
                    let arithmetic_gap = if raw_gap == 0.0 {
                        0.0
                    } else if raw_gap.is_finite() {
                        next_up_f64(raw_gap)
                    } else {
                        f64::INFINITY
                    };
                    let intrinsic_gap =
                        nonnegative_add_up(arithmetic_gap, anchored_bn_flush_charge(stored));
                    let parameter_gap =
                        nonnegative_mul_up(coeff64.abs(), f32_to_f64_exact(self.scale_err[[ic]]));
                    let tap_error = nonnegative_add_up(intrinsic_gap, parameter_gap);
                    if tap_error > *coefficient_error {
                        *coefficient_error = tap_error;
                    }

                    let map_index =
                        checked_anchored_bn_tap_index(oh, ow, ic, ki, kj, out_w, in_c, kh, kw)?;
                    if *valid_map.get(map_index).ok_or_else(|| {
                    NyError::InternalError(format!(
                        "Anchored BatchNorm tap index {map_index} exceeds authenticated map length {}",
                        valid_map.len()
                    ))
                })? {
                    let bn_bias = self.bias[[ic]];
                    let term = safe_mul_for_bounds_f64(coeff64, f32_to_f64_exact(bn_bias));
                    *bias_accum = match direction {
                        OutwardDirection::Lower => add_f64_down(*bias_accum, term),
                        OutwardDirection::Upper => add_f64_up(*bias_accum, term),
                    };

                    let bias_magnitude = nonnegative_add_up(
                        f32_to_f64_exact(bn_bias).abs(),
                        f32_to_f64_exact(self.bias_err[[ic]]),
                    );
                    let tap_widen = nonnegative_add_up(
                        nonnegative_mul_up(coeff64.abs(), f32_to_f64_exact(self.bias_err[[ic]])),
                        nonnegative_mul_up(oe, bias_magnitude),
                    );
                    *widen = nonnegative_add_up(*widen, tap_widen);
                }
                    Ok(())
                };

                if explicit_rows {
                    for row in 0..logical_rows {
                        poll()?;
                        let oe = old_error
                            .map_or(0.0, |error| nonnegative_f32_error_or_infinity(error[row]));
                        let mut bias_accum = f32_to_f64_exact(bias[row]);
                        let mut coefficient_error = 0.0f64;
                        let mut widen = 0.0f64;
                        for _oc in 0..out_c {
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    for ic in 0..in_c {
                                        for ki in 0..kh {
                                            for kj in 0..kw {
                                                consume_tap(
                                                    oh,
                                                    ow,
                                                    ic,
                                                    ki,
                                                    kj,
                                                    oe,
                                                    &mut bias_accum,
                                                    &mut coefficient_error,
                                                    &mut widen,
                                                    &mut poll,
                                                )?;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        coefficient_error =
                            nonnegative_add_up(coefficient_error, nonnegative_mul_up(gain, oe));
                        new_bias_values.push(bias_accum);
                        new_error_values.push(anchored_bn_publish_error(coefficient_error));
                        widen_values.push(widen);
                    }
                } else {
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let row = checked_anchored_bn_output_index(
                                    oc,
                                    oh,
                                    ow,
                                    spatial_positions,
                                    out_w,
                                )?;
                                poll()?;
                                let oe = old_error.map_or(0.0, |error| {
                                    nonnegative_f32_error_or_infinity(error[row])
                                });
                                let mut bias_accum = f32_to_f64_exact(bias[row]);
                                let mut coefficient_error = 0.0f64;
                                let mut widen = 0.0f64;
                                for ic in 0..in_c {
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            consume_tap(
                                                oh,
                                                ow,
                                                ic,
                                                ki,
                                                kj,
                                                oe,
                                                &mut bias_accum,
                                                &mut coefficient_error,
                                                &mut widen,
                                                &mut poll,
                                            )?;
                                        }
                                    }
                                }
                                coefficient_error = nonnegative_add_up(
                                    coefficient_error,
                                    nonnegative_mul_up(gain, oe),
                                );
                                new_bias_values.push(bias_accum);
                                new_error_values.push(anchored_bn_publish_error(coefficient_error));
                                widen_values.push(widen);
                            }
                        }
                    }
                }
                if source_cursor != patch_elements
                    || new_values.len() != patch_elements
                    || new_bias_values.len() != logical_rows
                    || new_error_values.len() != logical_rows
                    || widen_values.len() != logical_rows
                {
                    return Err(NyError::InternalError(
                        "Anchored BatchNorm work-buffer length invariant failed".into(),
                    ));
                }
                poll()?;

                let new_patches = ArrayD::from_shape_vec(lower_patches.raw_dim(), new_values)
                    .map_err(|error| {
                        NyError::InternalError(format!(
                            "Anchored BatchNorm coefficient shape construction failed: {error}"
                        ))
                    })?;
                Ok((
                    PatchesData {
                        coeff_err: Some(Array1::from_vec(new_error_values)),
                        patches: Some(new_patches),
                        geometry: patches_data.geometry.clone(),
                        identity: false,
                        output_shape: patches_data.output_shape,
                        input_shape: patches_data.input_shape,
                        unstable_idx: None,
                    },
                    Array1::from_vec(new_bias_values),
                    Array1::from_vec(widen_values),
                ))
            };

            let lower = process_side(
                &bounds.lower_a,
                lower_slice,
                &bounds.lower_b,
                OutwardDirection::Lower,
            )?;
            let upper = process_side(
                &bounds.upper_a,
                upper_slice,
                &bounds.upper_b,
                OutwardDirection::Upper,
            )?;
            (lower, upper)
        };

        let mut new_lower_bias = admission.reserve::<f32>(
            logical_rows,
            "Anchored BatchNorm published lower-bias allocation",
        )?;
        let mut new_upper_bias = admission.reserve::<f32>(
            logical_rows,
            "Anchored BatchNorm published upper-bias allocation",
        )?;
        for row in 0..logical_rows {
            if row.is_multiple_of(ANCHORED_BN_POLL_COORDS) {
                poll()?;
            }
            new_lower_bias.push(f64_to_f32_down(add_f64_down(
                lower_bias64[row],
                -lower_widen[row],
            )));
            new_upper_bias.push(f64_to_f32_up(add_f64_up(
                upper_bias64[row],
                upper_widen[row],
            )));
        }
        poll()?;

        Ok(CrownBounds::Patches(Box::new(PatchesLinearBounds {
            row_count: bounds.row_count,
            lower_a: new_lower_a,
            lower_b: Array1::from_vec(new_lower_bias),
            upper_a: new_upper_a,
            upper_b: Array1::from_vec(new_upper_bias),
        })))
    }
}

#[cfg(test)]
mod anchored_budget_tests {
    use super::{
        anchored_bn_planned_bytes, with_anchored_bn_budget_for_test, AnchoredBatchNormAdmission,
        BatchNormLayer,
    };
    use crate::bounds::patches::{PatchGeometry, PatchesData, PatchesLinearBounds};
    use ndarray::{Array1, ArrayD, IxDyn};
    use ny_core::{NyError, Result};

    fn fixture(width: usize) -> (BatchNormLayer, PatchesLinearBounds) {
        let layer = BatchNormLayer {
            scale: ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.75]).unwrap(),
            bias: ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.125]).unwrap(),
            scale_err: ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
            bias_err: ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
            num_channels: 1,
            channel_axis_hint: None,
        };
        let geometry = PatchGeometry::anchored(vec![0], vec![0]).unwrap();
        let make_side = || PatchesData {
            coeff_err: Some(Array1::from_vec(vec![0.25])),
            patches: Some(ArrayD::from_elem(IxDyn(&[1, 1, 1, 1, 1, width]), 0.5f32)),
            geometry: geometry.clone(),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, width),
            unstable_idx: None,
        };
        (
            layer,
            PatchesLinearBounds {
                row_count: 1,
                lower_a: make_side(),
                lower_b: Array1::from_vec(vec![0.0]),
                upper_a: make_side(),
                upper_b: Array1::from_vec(vec![0.0]),
            },
        )
    }

    #[test]
    fn anchored_bn_total_live_exact_budget_and_minus_one_are_atomic() -> Result<()> {
        let width = 32_768usize;
        let (layer, bounds) = fixture(width);
        let planned = anchored_bn_planned_bytes(width, width, 1)?;
        let source = bounds.memory_bytes();
        let total = source.checked_add(planned).unwrap();
        assert!(source > 1 && planned < total);

        assert!(AnchoredBatchNormAdmission::with_budget(source, planned, total).is_ok());
        assert!(matches!(
            AnchoredBatchNormAdmission::with_budget(source, planned, total - 1),
            Err(NyError::CpuMemoryExceeded { .. })
        ));

        let completed = with_anchored_bn_budget_for_test(total, || {
            layer.propagate_patches_with_deadline(
                &bounds,
                std::time::Instant::now() + std::time::Duration::from_secs(30),
            )
        });
        assert!(completed.is_ok(), "exact total-live budget must admit");

        let lower_before = bounds.lower_a.patches.as_ref().unwrap().clone();
        let upper_before = bounds.upper_a.patches.as_ref().unwrap().clone();
        let lower_bias_before = bounds.lower_b.clone();
        let upper_bias_before = bounds.upper_b.clone();
        let refused = with_anchored_bn_budget_for_test(total - 1, || {
            layer.propagate_patches_with_deadline(
                &bounds,
                std::time::Instant::now() + std::time::Duration::from_secs(30),
            )
        });
        assert!(matches!(refused, Err(NyError::CpuMemoryExceeded { .. })));
        assert_eq!(bounds.lower_a.patches.as_ref().unwrap(), &lower_before);
        assert_eq!(bounds.upper_a.patches.as_ref().unwrap(), &upper_before);
        assert_eq!(bounds.lower_b, lower_bias_before);
        assert_eq!(bounds.upper_b, upper_bias_before);
        Ok(())
    }
}
