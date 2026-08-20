// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-mode CROWN backward for element-wise activation functions.
//!
//! Operates on 6D patches tensors [oc, oh, ow, ic, ki, kj] from Conv2d backward,
//! or 4D sparse patches tensors [sparse_idx, ic, ki, kj] when unstable_idx is set.

use ndarray::ArrayD;
use ny_core::{
    checked_shape_product, f32_to_f64_exact, f64_to_f32_down, f64_to_f32_up, NyError, Result,
};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use rayon::prelude::*;
use std::borrow::Cow;
#[cfg(test)]
use std::cell::Cell;
use std::mem::size_of;
use std::sync::Mutex;
use std::time::Instant;

use super::compose;
use super::crown_patches_sparse::backward_patches_sparse;
use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds};
use crate::layers::linear::bias::{add_f64_down, add_f64_up};

const EXPLICIT_ROW_DEADLINE_POLL_COORDS: usize = 4_096;
const ANCHORED_ZERO_FILL_CHUNK: usize = EXPLICIT_ROW_DEADLINE_POLL_COORDS;

/// Total-live receipt for an Anchored carrier transformation. The incoming
/// carrier is borrowed and remains resident for the whole transaction, so the
/// receipt covers it plus every additional buffer retained at the peak.
/// `try_reserve_exact` is followed by a capacity
/// reconciliation because allocators may legally round a request upward.
struct AnchoredActivationAdmission {
    source_resident_bytes: usize,
    required_bytes: usize,
    budget_bytes: usize,
    remaining_planned_bytes: usize,
    allocated_capacity_bytes: usize,
}

impl AnchoredActivationAdmission {
    fn new(source_resident_bytes: usize, planned_allocation_bytes: usize) -> Result<Self> {
        Self::with_budget(
            source_resident_bytes,
            planned_allocation_bytes,
            anchored_activation_budget_bytes(),
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
                    "Anchored activation total-live byte count overflows usize".into(),
                )
            })?;
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "Anchored activation Patches work buffers",
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
        poll: &impl Fn() -> Result<()>,
    ) -> Result<Vec<T>> {
        let mut values = self.reserve::<T>(len, site)?;
        while values.len() < len {
            poll()?;
            let end = len.min(values.len().saturating_add(ANCHORED_ZERO_FILL_CHUNK));
            values.resize(end, zero.clone());
        }
        poll()?;
        Ok(values)
    }
}

#[cfg(test)]
thread_local! {
    static ANCHORED_ACTIVATION_TEST_BUDGET: Cell<Option<usize>> = const { Cell::new(None) };
}

fn anchored_activation_budget_bytes() -> usize {
    #[cfg(test)]
    if let Some(value) = ANCHORED_ACTIVATION_TEST_BUDGET.with(Cell::get) {
        return value;
    }
    crate::network::crown_memory::cpu_crown_dense_budget_bytes()
}

#[cfg(test)]
fn with_anchored_activation_budget_for_test<T>(budget: usize, run: impl FnOnce() -> T) -> T {
    ANCHORED_ACTIVATION_TEST_BUDGET.with(|slot| {
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

fn checked_bytes<T>(len: usize, label: &str) -> Result<usize> {
    len.checked_mul(size_of::<T>())
        .ok_or_else(|| NyError::InvalidSpec(format!("{label}: byte count overflows usize")))
}

fn checked_sum_bytes(parts: &[usize], label: &str) -> Result<usize> {
    parts.iter().try_fold(0usize, |total, &part| {
        total
            .checked_add(part)
            .ok_or_else(|| NyError::InvalidSpec(format!("{label}: byte count overflows usize")))
    })
}

fn anchored_activation_planned_bytes(
    map_len: usize,
    num_input_neurons: usize,
    lower_input_scratch_len: usize,
    upper_input_scratch_len: usize,
    patch_elements: usize,
    logical_rows: usize,
) -> Result<usize> {
    let paired_patch_elements = patch_elements.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec("Anchored activation paired coefficient count overflows usize".into())
    })?;
    let paired_rows = logical_rows.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec("Anchored activation paired row count overflows usize".into())
    })?;
    let error_and_bias_rows = logical_rows.checked_mul(4).ok_or_else(|| {
        NyError::InvalidSpec(
            "Anchored activation error/published-bias count overflows usize".into(),
        )
    })?;
    checked_sum_bytes(
        &[
            checked_bytes::<usize>(map_len, "Anchored activation tap map")?,
            checked_bytes::<crate::layers::activations::LinearRelaxation>(
                num_input_neurons,
                "Anchored activation relaxations",
            )?,
            checked_bytes::<f32>(
                lower_input_scratch_len,
                "Anchored activation lower input scratch",
            )?,
            checked_bytes::<f32>(
                upper_input_scratch_len,
                "Anchored activation upper input scratch",
            )?,
            checked_bytes::<f32>(paired_patch_elements, "Anchored activation A")?,
            checked_bytes::<f64>(paired_rows, "Anchored activation b64")?,
            // Vec<bool> capacity is measured in elements. Charging one byte
            // per logical flag is exact for its payload and conservative with
            // any allocator capacity rounding reconciled by the receipt.
            checked_bytes::<bool>(paired_rows, "Anchored activation non-finite flags")?,
            checked_bytes::<f32>(
                error_and_bias_rows,
                "Anchored activation error and published bias",
            )?,
        ],
        "Anchored activation work buffers",
    )
}

#[inline]
fn f32_nonzero_by_bits(value: f32) -> bool {
    value.to_bits() & 0x7fff_ffff != 0
}

#[inline]
fn f32_negative_by_bits(value: f32) -> bool {
    value.to_bits() >> 31 != 0
}

#[inline]
fn f32_nonzero_subnormal(value: f32) -> bool {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    magnitude != 0 && magnitude < 0x0080_0000
}

#[inline]
fn signed_zero(value: f32) -> f32 {
    f32::from_bits(value.to_bits() & 0x8000_0000)
}

/// Outward non-negative binary64 addition. This is used only by the Anchored
/// proof channel; the affine compatibility path retains its historical bits.
#[inline]
fn activation_nonnegative_add_up(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() || left < 0.0 || right < 0.0 {
        return f64::INFINITY;
    }
    if left == 0.0 {
        return right;
    }
    if right == 0.0 {
        return left;
    }
    let sum = left + right;
    if sum == 0.0 {
        0.0
    } else if sum.is_finite() {
        ny_core::dd::next_up_f64(sum)
    } else {
        f64::INFINITY
    }
}

/// Outward non-negative binary64 multiplication with the exact zero limit
/// preserved (`0 * +INF` is a zero contribution, not NaN).
#[inline]
fn activation_nonnegative_mul_up(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() || left < 0.0 || right < 0.0 {
        return f64::INFINITY;
    }
    if left == 0.0 || right == 0.0 {
        return 0.0;
    }
    let product = left * right;
    if product.is_finite() {
        ny_core::dd::next_up_f64(product)
    } else {
        f64::INFINITY
    }
}

/// Authenticate an Anchored relaxation at the representation seam.
///
/// The callback receives outward, non-subnormal endpoints, so ReLU's sign
/// classification and crossing-chord arithmetic cannot DAZ a genuine
/// +/-subnormal endpoint to the wrong arm. A nonzero subnormal relaxation
/// field may itself be the one-minsub sentinel left after an FTZ conversion;
/// its unknown real magnitude is therefore bounded by `f32::MIN_POSITIVE`,
/// not merely by the stored minsub. Such slopes are normalized to signed zero
/// and discharged over the authenticated input interval; such intercepts are
/// likewise normalized and discharged directionally. Published fields are
/// zero, normal, or non-finite -- never subnormal.
fn harden_anchored_relaxation(
    lower: f32,
    upper: f32,
    relaxation: crate::layers::activations::LinearRelaxation,
) -> crate::layers::activations::LinearRelaxation {
    let max_input_magnitude = f32_to_f64_exact(lower)
        .abs()
        .max(f32_to_f64_exact(upper).abs());
    let min_normal = f32_to_f64_exact(f32::MIN_POSITIVE);

    let harden_lower = |slope: f32, intercept: f32| {
        let slope_is_tiny = f32_nonzero_subnormal(slope);
        let intercept_is_tiny = f32_nonzero_subnormal(intercept);
        let slope_risk = if slope_is_tiny {
            activation_nonnegative_mul_up(min_normal, max_input_magnitude)
        } else {
            0.0
        };
        let intercept_risk = if intercept_is_tiny { min_normal } else { 0.0 };
        let risk = activation_nonnegative_add_up(slope_risk, intercept_risk);
        let center = if intercept_is_tiny {
            0.0
        } else {
            f32_to_f64_exact(intercept)
        };
        (
            if slope_is_tiny {
                signed_zero(slope)
            } else {
                slope
            },
            f64_to_f32_down(add_f64_down(center, -risk)),
        )
    };
    let harden_upper = |slope: f32, intercept: f32| {
        let slope_is_tiny = f32_nonzero_subnormal(slope);
        let intercept_is_tiny = f32_nonzero_subnormal(intercept);
        let slope_risk = if slope_is_tiny {
            activation_nonnegative_mul_up(min_normal, max_input_magnitude)
        } else {
            0.0
        };
        let intercept_risk = if intercept_is_tiny { min_normal } else { 0.0 };
        let risk = activation_nonnegative_add_up(slope_risk, intercept_risk);
        let center = if intercept_is_tiny {
            0.0
        } else {
            f32_to_f64_exact(intercept)
        };
        (
            if slope_is_tiny {
                signed_zero(slope)
            } else {
                slope
            },
            f64_to_f32_up(add_f64_up(center, risk)),
        )
    };

    let (lower_slope, lower_intercept) =
        harden_lower(relaxation.lower_slope, relaxation.lower_intercept);
    let (upper_slope, upper_intercept) =
        harden_upper(relaxation.upper_slope, relaxation.upper_intercept);
    crate::layers::activations::LinearRelaxation::new(
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    )
}

#[inline]
fn anchored_compose(
    coeff: f32,
    slope: f32,
    intercept: f32,
    direction: crate::bounds::OutwardDirection,
) -> compose::ComposeResult {
    // Bit tests, rather than floating comparisons, keep +/-minsub live even
    // on a DAZ host. Magnitude-zero (including -0) is exactly inert.
    if !f32_nonzero_by_bits(coeff) {
        return compose::ComposeResult::ZERO;
    }
    let exact_product = f32_to_f64_exact(coeff) * f32_to_f64_exact(slope);
    let intercept_contrib = f32_to_f64_exact(coeff) * f32_to_f64_exact(intercept);
    let f32_max = f32_to_f64_exact(f32::MAX);
    let nonfinite = !exact_product.is_finite() || exact_product.abs() > f32_max;
    let new_coeff = if nonfinite {
        0.0
    } else {
        let magnitude = exact_product.abs();
        let min_normal = f32_to_f64_exact(f32::MIN_POSITIVE);
        if magnitude == 0.0 || magnitude < min_normal {
            // A hardware f64->f32 conversion may FTZ this range. Publish a
            // no-subnormal directed endpoint directly; coeff_err below
            // certifies its exact-real gap.
            match direction {
                crate::bounds::OutwardDirection::Lower => f64_to_f32_down(exact_product),
                crate::bounds::OutwardDirection::Upper => f64_to_f32_up(exact_product),
            }
        } else {
            // Preserve the historical normal-range compose bits: round the
            // exact product to nearest f32, then take the legacy one-ULP
            // directed step. Inputs cannot be DAZ here because the exact
            // product has already been decoded from bits in binary64.
            let nearest = exact_product as f32;
            let directed = match direction {
                crate::bounds::OutwardDirection::Lower => next_down_f32(nearest),
                crate::bounds::OutwardDirection::Upper => next_up_f32(nearest),
            };
            if f32_nonzero_subnormal(directed) {
                match direction {
                    crate::bounds::OutwardDirection::Lower => f64_to_f32_down(exact_product),
                    crate::bounds::OutwardDirection::Upper => f64_to_f32_up(exact_product),
                }
            } else {
                directed
            }
        }
    };
    compose::ComposeResult {
        new_coeff,
        intercept_contrib,
        nonfinite,
    }
}

#[inline]
fn anchored_compose_lower(
    coeff: f32,
    relax: &crate::layers::activations::LinearRelaxation,
) -> compose::ComposeResult {
    if !f32_nonzero_by_bits(coeff) {
        return compose::ComposeResult::ZERO;
    }
    let (slope, intercept) = if f32_negative_by_bits(coeff) {
        (relax.upper_slope, relax.upper_intercept)
    } else {
        (relax.lower_slope, relax.lower_intercept)
    };
    anchored_compose(
        coeff,
        slope,
        intercept,
        crate::bounds::OutwardDirection::Lower,
    )
}

#[inline]
fn anchored_compose_upper(
    coeff: f32,
    relax: &crate::layers::activations::LinearRelaxation,
) -> compose::ComposeResult {
    if !f32_nonzero_by_bits(coeff) {
        return compose::ComposeResult::ZERO;
    }
    let (slope, intercept) = if f32_negative_by_bits(coeff) {
        (relax.lower_slope, relax.lower_intercept)
    } else {
        (relax.upper_slope, relax.upper_intercept)
    };
    anchored_compose(
        coeff,
        slope,
        intercept,
        crate::bounds::OutwardDirection::Upper,
    )
}

#[inline]
fn checked_activation_tap_index(
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
        .ok_or_else(|| NyError::InvalidSpec("Anchored activation tap index overflows usize".into()))
}

#[inline]
fn checked_activation_output_index(
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
            NyError::InvalidSpec("Anchored activation output-row index overflows usize".into())
        })
}

#[inline]
fn checked_activation_explicit_tap_index(
    oc: usize,
    oh: usize,
    ow: usize,
    ic: usize,
    ki: usize,
    kj: usize,
    taps_per_output_channel: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) -> Result<usize> {
    let tap = checked_activation_tap_index(oh, ow, ic, ki, kj, out_w, in_c, kh, kw)?;
    oc.checked_mul(taps_per_output_channel)
        .and_then(|value| value.checked_add(tap))
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "Anchored activation explicit-row tap index overflows usize".into(),
            )
        })
}

#[inline]
fn stored_f32_flush_charge(value: f32) -> f64 {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < 0x0080_0000 {
        f32_to_f64_exact(value).abs()
    } else {
        0.0
    }
}

/// Exact-real coefficient product gap plus the charge needed if a later
/// FTZ/DAZ consumer observes a stored subnormal center as signed zero.
#[inline]
fn activation_product_gap(coeff: f32, slope: f32, stored: f32) -> f64 {
    let exact_product = f32_to_f64_exact(coeff) * f32_to_f64_exact(slope);
    let stored_exact = f32_to_f64_exact(stored);
    let raw_arithmetic_gap = (exact_product - stored_exact).abs();
    let arithmetic_gap =
        if raw_arithmetic_gap != 0.0 && exact_product.abs() < f32_to_f64_exact(f32::MIN_POSITIVE) {
            // A tiny exact product compared with a no-subnormal normal endpoint
            // can span more than binary64's 53 significand bits. Step that
            // subtraction outward; the normal-range legacy case below is exact.
            ny_core::dd::next_up_f64(raw_arithmetic_gap)
        } else {
            raw_arithmetic_gap
        };
    let flush_charge = stored_f32_flush_charge(stored);
    if !arithmetic_gap.is_finite() || !flush_charge.is_finite() {
        return f64::INFINITY;
    }
    // A normal-range product of two exact binary32 values and its adjacent
    // binary32 center fit exactly in binary64. Preserve that exact historical
    // gap when no FTZ charge is present; an unconditional next-up here moved
    // normal 6D receipts by one f32 ULP after final publication. Only the
    // underflow-span case above and a genuinely new two-term sum need outward
    // binary64 steps.
    activation_nonnegative_add_up(arithmetic_gap, flush_charge)
}

#[inline]
fn publish_activation_error_up(value: f64) -> f32 {
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

fn activation_input_slice<'a, const POLL: bool, P>(
    values: &'a ArrayD<f32>,
    admission: Option<&mut AnchoredActivationAdmission>,
    site: &'static str,
    poll: &P,
) -> Result<Cow<'a, [f32]>>
where
    P: Fn() -> Result<()> + Sync,
{
    if let Some(slice) = values.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }

    if let Some(admission) = admission {
        let mut copy = admission.reserve::<f32>(values.len(), site)?;
        for (index, &value) in values.iter().enumerate() {
            if index.is_multiple_of(EXPLICIT_ROW_DEADLINE_POLL_COORDS) && POLL {
                poll()?;
            }
            copy.push(value);
        }
        if POLL {
            poll()?;
        }
        Ok(Cow::Owned(copy))
    } else {
        Ok(Cow::Owned(values.iter().copied().collect()))
    }
}

#[inline]
fn explicit_row_deadline_parallel_admitted(region_seq_inner: bool) -> bool {
    !region_seq_inner
}

/// Validate the contiguous 6D/7D layout admitted by the finite-deadline ReLU
/// route and return the checked 7D per-spec-row coordinate count when present.
///
/// The historical no-deadline path deliberately retains its existing layout
/// handling. This hard gate exists because the cooperative route slices both
/// sides and every private output by a common row geometry; accepting a
/// mismatched side could silently truncate a Rayon zip or panic during a bias
/// update.
fn validate_deadline_materialized_geometry(bounds: &PatchesLinearBounds) -> Result<Option<usize>> {
    if bounds.lower_a.identity
        || bounds.upper_a.identity
        || bounds.lower_a.unstable_idx.is_some()
        || bounds.upper_a.unstable_idx.is_some()
    {
        return Err(NyError::UnsupportedConfiguration(
            "finite-deadline ReLU Patches requires materialized dense patches".into(),
        ));
    }

    let lower = bounds.lower_a.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("Non-identity lower PatchesData has no patches tensor".into())
    })?;
    let upper = bounds.upper_a.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("Non-identity upper PatchesData has no patches tensor".into())
    })?;
    if !matches!(lower.ndim(), 6 | 7) {
        return Err(NyError::ShapeMismatch {
            expected: vec![6, 7],
            got: vec![lower.ndim()],
        });
    }
    if upper.shape() != lower.shape() {
        return Err(NyError::ShapeMismatch {
            expected: lower.shape().to_vec(),
            got: upper.shape().to_vec(),
        });
    }

    let (out_c, out_h, out_w) = bounds.lower_a.output_shape;
    let (in_c, in_h, in_w) = bounds.lower_a.input_shape;
    let shape = lower.shape();
    let expected_shape = if shape.len() == 7 {
        vec![
            bounds.row_count,
            out_c,
            out_h,
            out_w,
            in_c,
            shape[5],
            shape[6],
        ]
    } else {
        vec![out_c, out_h, out_w, in_c, shape[4], shape[5]]
    };
    if shape != expected_shape.as_slice() {
        return Err(NyError::ShapeMismatch {
            expected: expected_shape,
            got: shape.to_vec(),
        });
    }
    if bounds.upper_a.output_shape != bounds.lower_a.output_shape
        || bounds.upper_a.input_shape != bounds.lower_a.input_shape
    {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_c, out_h, out_w, in_c, in_h, in_w],
            got: vec![
                bounds.upper_a.output_shape.0,
                bounds.upper_a.output_shape.1,
                bounds.upper_a.output_shape.2,
                bounds.upper_a.input_shape.0,
                bounds.upper_a.input_shape.1,
                bounds.upper_a.input_shape.2,
            ],
        });
    }
    // Paired typed geometry was authenticated by the common validator before
    // this layout gate. Do not compare Anchored origin vectors again here: the
    // production finite wrapper has already scanned them cooperatively.
    let logical_rows = if shape.len() == 7 {
        bounds.row_count
    } else {
        checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec("finite-deadline ReLU Patches spatial row count overflow".into())
        })?
    };
    if bounds.lower_b.len() != logical_rows || bounds.upper_b.len() != logical_rows {
        return Err(NyError::ShapeMismatch {
            expected: vec![logical_rows, logical_rows],
            got: vec![bounds.lower_b.len(), bounds.upper_b.len()],
        });
    }
    if lower.as_slice().is_none() || upper.as_slice().is_none() {
        return Err(NyError::UnsupportedConfiguration(
            "finite-deadline ReLU Patches requires contiguous materialized patches".into(),
        ));
    }

    if shape.len() == 7 {
        checked_shape_product(&shape[1..]).map(Some).ok_or_else(|| {
            NyError::InvalidSpec(
                "finite-deadline ReLU Patches explicit-row coordinate count overflow".into(),
            )
        })
    } else {
        checked_shape_product(shape).map(|_| None).ok_or_else(|| {
            NyError::InvalidSpec(
                "finite-deadline ReLU Patches spatial coordinate count overflow".into(),
            )
        })
    }
}

#[inline(always)]
fn poll_explicit_row_coordinate<const POLL: bool, P>(
    coordinates_since_poll: &mut usize,
    poll: &P,
) -> Result<()>
where
    P: Fn() -> Result<()> + Sync,
{
    if POLL {
        *coordinates_since_poll += 1;
        if *coordinates_since_poll >= EXPLICIT_ROW_DEADLINE_POLL_COORDS {
            poll()?;
            *coordinates_since_poll = 0;
        }
    }
    Ok(())
}

/// CROWN backward for element-wise activations in Patches mode.
///
/// This is the Patches-mode equivalent of [`super::crown_elementwise_backward`].
/// Instead of operating on Dense A-matrices (Array2), it scales the 6D patches
/// tensor by per-INPUT-neuron relaxation slopes and updates the bias vectors
/// with intercept contributions.
///
/// Each patches coefficient at [oc, oh, ow, ic, ki, kj] connects a specification
/// output (oc, oh, ow) to an INPUT neuron at position:
///   ih = oh * stride_h + ki - pad_top
///   iw = ow * stride_w + kj - pad_left
/// The relaxation slope for that coefficient is determined by the pre-activation
/// bounds of the mapped input neuron, not the output position.
///
/// Handles identity patches by materializing them first, since element-wise
/// scaling produces non-identity results.
///
/// # Arguments
/// * `bounds` - Incoming Patches linear bounds from layers above
/// * `pre_activation` - Pre-activation bounds for the activation's input neurons.
///   Must have shape matching `bounds.lower_a.input_shape` (the space the patches
///   reference into).
/// * `relaxation_fn` - `(l, u) -> LinearRelaxation`
///
/// Reference: alpha-beta-CROWN auto_LiRPA/operators/relu.py (Patches backward)
/// Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md Phase 2, Step 8
/// Part of #2613
pub(crate) fn crown_elementwise_backward_patches<F>(
    bounds: &PatchesLinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<CrownBounds>
where
    F: Fn(f32, f32) -> crate::layers::activations::LinearRelaxation,
{
    let probe_start = Instant::now();
    let result = crown_elementwise_backward_patches_impl::<false, _, _>(
        bounds,
        pre_activation,
        relaxation_fn,
        true,
        false,
        &|| Ok(()),
    );
    PATCHES_BWD_NANOS.fetch_add(
        u64::try_from(probe_start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
    PATCHES_BWD_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    result
}

/// Cooperative finite-deadline face used for Anchored ReLU carriers and the
/// default-dark spec explicit-row dispatch.
///
/// One unchanged absolute deadline is shared by all workers. The helper
/// refuses sparse, identity, non-contiguous, and non-6D/7D layouts before
/// allocating output scratch. An error drops every private coefficient, bias,
/// non-finite, and certificate buffer before any [`CrownBounds`] is returned.
pub(crate) fn crown_elementwise_backward_patches_with_deadline<F>(
    bounds: &PatchesLinearBounds,
    pre_activation: &BoundedTensor,
    deadline: Instant,
    relaxation_fn: F,
) -> Result<CrownBounds>
where
    F: Fn(f32, f32) -> crate::layers::activations::LinearRelaxation,
{
    let mut deadline_state =
        crate::bounds::patches::PatchesMaterializationDeadline::new(Some(deadline));
    bounds
        .lower_a
        .validate_common_geometry_with_poll(&bounds.upper_a, &mut deadline_state)?;
    let deadline_state = Mutex::new(deadline_state);
    let poll = || {
        deadline_state
            .lock()
            .map_err(|_| {
                NyError::InternalError(
                    "ReLU Patches deadline state was poisoned by a worker".into(),
                )
            })?
            .checkpoint("during Anchored ReLU Patches backward")
    };
    let probe_start = Instant::now();
    let parallel = explicit_row_deadline_parallel_admitted(crate::imb::region_seq_inner());
    if !parallel {
        PATCHES_BWD_SERIAL_ROWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let result = crown_elementwise_backward_patches_impl::<true, _, _>(
        bounds,
        pre_activation,
        relaxation_fn,
        parallel,
        true,
        &poll,
    );
    PATCHES_BWD_NANOS.fetch_add(
        u64::try_from(probe_start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
    PATCHES_BWD_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    result
}

#[cfg(test)]
pub(crate) fn crown_elementwise_backward_patches_with_poll_for_test<F, P>(
    bounds: &PatchesLinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
    parallel_rows: bool,
    poll: &P,
) -> Result<CrownBounds>
where
    F: Fn(f32, f32) -> crate::layers::activations::LinearRelaxation,
    P: Fn() -> Result<()> + Sync,
{
    crown_elementwise_backward_patches_impl::<true, _, _>(
        bounds,
        pre_activation,
        relaxation_fn,
        parallel_rows,
        false,
        poll,
    )
}

/// Temporary share probes (read via the collection dump lever).
pub(crate) static PATCHES_BWD_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static PATCHES_BWD_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static PATCHES_BWD_SERIAL_ROWS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn crown_elementwise_backward_patches_impl<const POLL: bool, F, P>(
    bounds: &PatchesLinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
    parallel_rows: bool,
    geometry_prevalidated: bool,
    poll: &P,
) -> Result<CrownBounds>
where
    F: Fn(f32, f32) -> crate::layers::activations::LinearRelaxation,
    P: Fn() -> Result<()> + Sync,
{
    if POLL {
        poll()?;
    }
    if !geometry_prevalidated {
        bounds.lower_a.validate_common_geometry(&bounds.upper_a)?;
    }
    let anchored_geometry = matches!(&bounds.lower_a.geometry, PatchGeometry::Anchored(_));
    if anchored_geometry
        && (bounds.lower_a.identity
            || bounds.upper_a.identity
            || bounds.lower_a.unstable_idx.is_some()
            || bounds.upper_a.unstable_idx.is_some())
    {
        return Err(NyError::UnsupportedConfiguration(
            "Anchored activation Patches requires materialized dense 6D/7D carriers".into(),
        ));
    }
    let affine_geometry = if anchored_geometry {
        None
    } else {
        Some(
            bounds
                .lower_a
                .geometry
                .require_affine("ReLU Patches backward")?,
        )
    };
    let checked_deadline_row_volume = if POLL {
        validate_deadline_materialized_geometry(bounds)?
    } else {
        None
    };

    let (out_c, out_h, out_w) = bounds.lower_a.output_shape;

    // Pre-activation bounds must match the patches' input_shape (the neuron space)
    let (in_c_shape, in_h_shape, in_w_shape) = bounds.lower_a.input_shape;
    let num_input_neurons = checked_shape_product(&[in_c_shape, in_h_shape, in_w_shape])
        .ok_or_else(|| NyError::InvalidSpec("ReLU Patches input-neuron count overflow".into()))?;
    if pre_activation.len() != num_input_neurons {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_input_neurons],
            got: vec![pre_activation.len()],
        });
    }

    // Materialize identity patches if needed — element-wise scaling makes them
    // non-identity. The admitted deadline route already required materialized
    // 6D/7D inputs, so borrow its giant tensors instead of cloning them. The
    // no-deadline compatibility path deliberately retains the historical
    // owned clones.
    let lower_a_data: Cow<'_, PatchesData> = if bounds.lower_a.identity {
        Cow::Owned(bounds.lower_a.try_materialize_identity()?)
    } else if POLL || anchored_geometry {
        Cow::Borrowed(&bounds.lower_a)
    } else {
        Cow::Owned(bounds.lower_a.clone())
    };
    let upper_a_data: Cow<'_, PatchesData> = if bounds.upper_a.identity {
        Cow::Owned(bounds.upper_a.try_materialize_identity()?)
    } else if POLL || anchored_geometry {
        Cow::Borrowed(&bounds.upper_a)
    } else {
        Cow::Owned(bounds.upper_a.clone())
    };

    let lower_patches = lower_a_data.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("Non-identity PatchesData has no patches tensor".into())
    })?;
    let upper_patches = upper_a_data.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("Non-identity PatchesData has no patches tensor".into())
    })?;
    if upper_patches.shape() != lower_patches.shape() {
        return Err(NyError::ShapeMismatch {
            expected: lower_patches.shape().to_vec(),
            got: upper_patches.shape().to_vec(),
        });
    }

    // Sparse patches: 4D (unstable_size, in_c, kH, kW). Delegate to sparse path.
    // Part of #2613 Phase 4 step 19
    if lower_a_data.unstable_idx.is_some() {
        // Sparse carriers are refused by the finite-deadline validator and are
        // affine-only. Preserve their historical relaxation construction here;
        // the Anchored admission below must not reserve work that this route
        // never consumes.
        let relaxations: Vec<_> = pre_activation
            .lower()
            .iter()
            .zip(pre_activation.upper().iter())
            .map(|(&lower, &upper)| relaxation_fn(lower, upper))
            .collect();
        return backward_patches_sparse(
            &lower_a_data,
            &upper_a_data,
            lower_patches,
            upper_patches,
            bounds,
            &relaxations,
            (in_c_shape, in_h_shape, in_w_shape),
        );
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
        _ => {
            return Err(NyError::ShapeMismatch {
                expected: vec![6, 7],
                got: vec![shape.len()],
            });
        }
    };
    let (in_c, kh, kw) = if explicit_rows {
        (shape[4], shape[5], shape[6])
    } else {
        (shape[3], shape[4], shape[5])
    };
    let expected_shape = if explicit_rows {
        vec![bounds.row_count, out_c, out_h, out_w, in_c_shape, kh, kw]
    } else {
        vec![out_c, out_h, out_w, in_c_shape, kh, kw]
    };
    if shape != expected_shape.as_slice() {
        return Err(NyError::ShapeMismatch {
            expected: expected_shape,
            got: shape.to_vec(),
        });
    }
    let patch_volume = checked_shape_product(&[in_c, kh, kw])
        .ok_or_else(|| NyError::InvalidSpec("ReLU Patches tap count overflow".into()))?;
    let spatial_positions = out_h.checked_mul(out_w).ok_or_else(|| {
        NyError::InvalidSpec("ReLU Patches spatial-position count overflows usize".into())
    })?;
    let output_positions = out_c.checked_mul(spatial_positions).ok_or_else(|| {
        NyError::InvalidSpec("ReLU Patches output-position count overflows usize".into())
    })?;
    // Contiguous chunk size of one explicit-rows SPEC row (7D layout only;
    // hoisted so both the compose pass and the err pass share it, spec §6.3).
    let row_volume = output_positions.checked_mul(patch_volume).ok_or_else(|| {
        NyError::InvalidSpec("ReLU Patches explicit-row volume overflows usize".into())
    })?;
    if let Some(validated) = checked_deadline_row_volume {
        if validated != row_volume {
            return Err(NyError::ShapeMismatch {
                expected: vec![row_volume],
                got: vec![validated],
            });
        }
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

    // A carried coeff_err is indexed by the logical row for both 6D spatial
    // and 7D explicit-row layouts. A short vector must never be interpreted as
    // an exact zero tail: that is the false-proof direction.
    for err in [
        lower_a_data.coeff_err.as_ref(),
        upper_a_data.coeff_err.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if err.len() != logical_rows {
            return Err(NyError::ShapeMismatch {
                expected: vec![logical_rows],
                got: vec![err.len()],
            });
        }
    }

    // An Anchored route borrows the incoming pair, so its total-live receipt
    // includes the complete source carrier (A/b/coeff_err and Arc-backed
    // geometry, conservatively charged once per side by `memory_bytes`) plus
    // every new buffer retained by this transaction. Affine callers retain
    // their historical admission policy.
    let taps_per_output_channel = checked_shape_product(&[out_h, out_w, in_c, kh, kw])
        .ok_or_else(|| NyError::InvalidSpec("activation tap-plane size overflows usize".into()))?;
    let map_len = if anchored_geometry {
        taps_per_output_channel
    } else {
        0
    };
    let pre_lower_needs_copy = pre_activation.lower().as_slice().is_none();
    let pre_upper_needs_copy = pre_activation.upper().as_slice().is_none();
    let patch_elements = lower_patches.len();
    let planned_allocation_bytes = if anchored_geometry {
        anchored_activation_planned_bytes(
            map_len,
            num_input_neurons,
            if pre_lower_needs_copy {
                num_input_neurons
            } else {
                0
            },
            if pre_upper_needs_copy {
                num_input_neurons
            } else {
                0
            },
            patch_elements,
            logical_rows,
        )?
    } else {
        0
    };
    let source_resident_bytes = if anchored_geometry {
        bounds.memory_bytes()
    } else {
        0
    };
    let mut anchored_admission = anchored_geometry
        .then(|| AnchoredActivationAdmission::new(source_resident_bytes, planned_allocation_bytes))
        .transpose()?;

    let pre_lower_values = activation_input_slice::<POLL, _>(
        pre_activation.lower(),
        anchored_admission.as_mut(),
        "Anchored activation lower input scratch",
        poll,
    )?;
    let pre_upper_values = activation_input_slice::<POLL, _>(
        pre_activation.upper(),
        anchored_admission.as_mut(),
        "Anchored activation upper input scratch",
        poll,
    )?;
    if pre_lower_values.len() != num_input_neurons || pre_upper_values.len() != num_input_neurons {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_input_neurons, num_input_neurons],
            got: vec![pre_lower_values.len(), pre_upper_values.len()],
        });
    }

    let mut relaxations = if let Some(admission) = anchored_admission.as_mut() {
        admission.reserve::<crate::layers::activations::LinearRelaxation>(
            num_input_neurons,
            "Anchored activation relaxations",
        )?
    } else {
        Vec::with_capacity(num_input_neurons)
    };
    for index in 0..num_input_neurons {
        if index.is_multiple_of(EXPLICIT_ROW_DEADLINE_POLL_COORDS) && POLL {
            poll()?;
        }
        if anchored_geometry {
            // Convert from exact f32 bits to outward no-subnormal endpoints
            // before the callback. This closes ReLU's DAZ-sensitive sign and
            // crossing-chord seam for +/-minsub pre-activations.
            let lower = f64_to_f32_down(f32_to_f64_exact(pre_lower_values[index]));
            let upper = f64_to_f32_up(f32_to_f64_exact(pre_upper_values[index]));
            relaxations.push(harden_anchored_relaxation(
                lower,
                upper,
                relaxation_fn(lower, upper),
            ));
        } else {
            relaxations.push(relaxation_fn(
                pre_lower_values[index],
                pre_upper_values[index],
            ));
        }
    }
    if POLL {
        poll()?;
    }

    // The Anchored map is authenticated and built exactly once. Every value
    // and coefficient-error pass consumes this same table, preventing the two
    // proof channels from disagreeing about a padded tap. `usize::MAX` denotes
    // the typed geometry's zero-padding result.
    let anchored_input_map = if anchored_geometry {
        let admission = anchored_admission.as_mut().ok_or_else(|| {
            NyError::InternalError("Anchored activation admission is missing".into())
        })?;
        let mut map =
            admission.reserve::<usize>(map_len, "Anchored activation tap-map allocation")?;
        let mut work = 0usize;
        for oh in 0..out_h {
            for ow in 0..out_w {
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            if work.is_multiple_of(EXPLICIT_ROW_DEADLINE_POLL_COORDS) && POLL {
                                poll()?;
                            }
                            work = work.saturating_add(1);
                            map.push(
                                lower_a_data
                                    .geometry
                                    .input_flat_index(
                                        (oh, ow),
                                        ic,
                                        (ki, kj),
                                        (in_c_shape, in_h_shape, in_w_shape),
                                    )?
                                    .unwrap_or(usize::MAX),
                            );
                        }
                    }
                }
            }
        }
        if map.len() != map_len {
            return Err(NyError::InternalError(format!(
                "Anchored activation tap-map length {} differs from planned {map_len}",
                map.len()
            )));
        }
        if POLL {
            poll()?;
        }
        Some(map)
    } else {
        None
    };

    // Geometry equality was validated before either side was materialized.
    let affine_parameters = affine_geometry.map(|geometry| {
        let (sh, sw) = geometry.stride();
        let (pad_left, _pad_right, pad_top, _pad_bottom) = geometry.padding();
        (sh, sw, pad_left, pad_top)
    });
    let mapped_input_flat =
        |oh: usize, ow: usize, ic: usize, ki: usize, kj: usize| -> Result<Option<usize>> {
            if let Some(map) = anchored_input_map.as_ref() {
                let index = checked_activation_tap_index(oh, ow, ic, ki, kj, out_w, in_c, kh, kw)?;
                let value = *map.get(index).ok_or_else(|| {
                    NyError::InternalError(format!(
                        "Anchored activation tap index {index} exceeds authenticated map length {}",
                        map.len()
                    ))
                })?;
                return Ok((value != usize::MAX).then_some(value));
            }
            let (sh, sw, pad_left, pad_top) = affine_parameters
                .expect("non-Anchored Patches geometry was authenticated as affine");
            let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
            let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
            if ih_raw < 0
                || (ih_raw as usize) >= in_h_shape
                || iw_raw < 0
                || (iw_raw as usize) >= in_w_shape
            {
                Ok(None)
            } else {
                let channel_stride = in_h_shape.checked_mul(in_w_shape).ok_or_else(|| {
                    NyError::InvalidSpec("ReLU Patches input channel stride overflows usize".into())
                })?;
                let spatial = (ih_raw as usize)
                    .checked_mul(in_w_shape)
                    .and_then(|value| value.checked_add(iw_raw as usize))
                    .ok_or_else(|| {
                        NyError::InvalidSpec(
                            "ReLU Patches input spatial index overflows usize".into(),
                        )
                    })?;
                let value = ic
                    .checked_mul(channel_stride)
                    .and_then(|value| value.checked_add(spatial))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("ReLU Patches input-flat index overflows usize".into())
                    })?;
                Ok(Some(value))
            }
        };

    // Create output patches and bias. Bias uses f64 to prevent catastrophic
    // cancellation (#1745), matching the Dense path in crown_elementwise_backward_indexed.
    if POLL {
        poll()?;
    }
    let mut new_lower_patches = if let Some(admission) = anchored_admission.as_mut() {
        let values = admission.zeroed(
            patch_elements,
            0.0f32,
            "Anchored activation lower coefficient allocation",
            poll,
        )?;
        ArrayD::from_shape_vec(lower_patches.raw_dim(), values).map_err(|error| {
            NyError::InternalError(format!(
                "Anchored activation lower coefficient shape construction failed: {error}"
            ))
        })?
    } else {
        ArrayD::<f32>::zeros(lower_patches.raw_dim())
    };
    if POLL {
        poll()?;
    }
    let mut new_upper_patches = if let Some(admission) = anchored_admission.as_mut() {
        let values = admission.zeroed(
            patch_elements,
            0.0f32,
            "Anchored activation upper coefficient allocation",
            poll,
        )?;
        ArrayD::from_shape_vec(upper_patches.raw_dim(), values).map_err(|error| {
            NyError::InternalError(format!(
                "Anchored activation upper coefficient shape construction failed: {error}"
            ))
        })?
    } else {
        ArrayD::<f32>::zeros(upper_patches.raw_dim())
    };
    if POLL {
        poll()?;
    }
    let mut new_lower_b_f64 = if let Some(admission) = anchored_admission.as_mut() {
        let mut values = admission.reserve::<f64>(
            logical_rows,
            "Anchored activation lower f64 bias allocation",
        )?;
        for (index, &value) in bounds.lower_b.iter().enumerate() {
            if index.is_multiple_of(EXPLICIT_ROW_DEADLINE_POLL_COORDS) && POLL {
                poll()?;
            }
            values.push(f32_to_f64_exact(value));
        }
        ndarray::Array1::from_vec(values)
    } else {
        bounds.lower_b.mapv(|x| x as f64)
    };
    let mut new_upper_b_f64 = if let Some(admission) = anchored_admission.as_mut() {
        let mut values = admission.reserve::<f64>(
            logical_rows,
            "Anchored activation upper f64 bias allocation",
        )?;
        for (index, &value) in bounds.upper_b.iter().enumerate() {
            if index.is_multiple_of(EXPLICIT_ROW_DEADLINE_POLL_COORDS) && POLL {
                poll()?;
            }
            values.push(f32_to_f64_exact(value));
        }
        ndarray::Array1::from_vec(values)
    } else {
        bounds.upper_b.mapv(|x| x as f64)
    };

    // Track non-finite rows for ±Inf fallback (#3009)
    let mut lower_nonfinite = if let Some(admission) = anchored_admission.as_mut() {
        admission.zeroed(
            logical_rows,
            false,
            "Anchored activation lower non-finite allocation",
            poll,
        )?
    } else {
        vec![false; logical_rows]
    };
    let mut upper_nonfinite = if let Some(admission) = anchored_admission.as_mut() {
        admission.zeroed(
            logical_rows,
            false,
            "Anchored activation upper non-finite allocation",
            poll,
        )?
    } else {
        vec![false; logical_rows]
    };

    // Anchored coefficients and intercept folds cross a proof seam: bit-based
    // sign selection keeps subnormal coefficients alive on DAZ hosts, exact
    // binary64 products avoid an f32 FTZ center, and every bias addition is
    // directed. The affine route deliberately retains its historical compose
    // implementation and bit pattern.
    let compose_lower_for_geometry =
        |coeff: f32, relax: &crate::layers::activations::LinearRelaxation| {
            if anchored_geometry {
                anchored_compose_lower(coeff, relax)
            } else {
                compose::compose_lower(coeff, relax)
            }
        };
    let compose_upper_for_geometry =
        |coeff: f32, relax: &crate::layers::activations::LinearRelaxation| {
            if anchored_geometry {
                anchored_compose_upper(coeff, relax)
            } else {
                compose::compose_upper(coeff, relax)
            }
        };
    let add_lower_bias = |accumulator: &mut f64, contribution: f64| {
        if anchored_geometry {
            *accumulator = add_f64_down(*accumulator, contribution);
        } else {
            *accumulator += contribution;
        }
    };
    let add_upper_bias = |accumulator: &mut f64, contribution: f64| {
        if anchored_geometry {
            *accumulator = add_f64_up(*accumulator, contribution);
        } else {
            *accumulator += contribution;
        }
    };
    let coefficient_nonzero_for_geometry = |value: f32| {
        if anchored_geometry {
            f32_nonzero_by_bits(value)
        } else {
            value != 0.0
        }
    };
    let coefficient_positive_for_geometry = |value: f32| {
        if anchored_geometry {
            f32_nonzero_by_bits(value) && !f32_negative_by_bits(value)
        } else {
            value > 0.0
        }
    };
    let proof_add_nonnegative = |left: f64, right: f64| {
        if anchored_geometry {
            activation_nonnegative_add_up(left, right)
        } else {
            left + right
        }
    };
    let proof_mul_nonnegative = |left: f64, right: f64| {
        if anchored_geometry {
            activation_nonnegative_mul_up(left, right)
        } else {
            left * right
        }
    };
    let discharge_lower = |bias: &mut f64, discharge: f64| {
        if discharge.is_finite() {
            if anchored_geometry {
                *bias = add_f64_down(*bias, -discharge);
            } else {
                *bias -= discharge;
            }
        } else {
            *bias = f64::NEG_INFINITY;
        }
    };
    let discharge_upper = |bias: &mut f64, discharge: f64| {
        if discharge.is_finite() {
            if anchored_geometry {
                *bias = add_f64_up(*bias, discharge);
            } else {
                *bias += discharge;
            }
        } else {
            *bias = f64::INFINITY;
        }
    };

    // Each output row owns a disjoint contiguous chunk of the standard-layout
    // patches tensors plus its own b/nonfinite slot — no cross-row state — so
    // rows compose in parallel with the per-row tap order unchanged
    // (value-identical to the serial loop). Non-standard layout (as_slice
    // returns None) falls back to the serial path.
    if explicit_rows {
        let compose_row_7d = |lp_r: &[f32],
                              up_r: &[f32],
                              nlp_r: &mut [f32],
                              nup_r: &mut [f32],
                              nlb_r: &mut f64,
                              nub_r: &mut f64,
                              lnf_r: &mut bool,
                              unf_r: &mut bool|
         -> Result<()> {
            let mut coordinates_since_poll = 0usize;
            if POLL {
                poll()?;
            }
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    poll_explicit_row_coordinate::<POLL, _>(
                                        &mut coordinates_since_poll,
                                        poll,
                                    )?;
                                    let Some(input_flat) = mapped_input_flat(oh, ow, ic, ki, kj)?
                                    else {
                                        continue;
                                    };
                                    let relax = &relaxations[input_flat];
                                    // Flat tap index within the row's contiguous chunk
                                    let t = checked_activation_explicit_tap_index(
                                        oc,
                                        oh,
                                        ow,
                                        ic,
                                        ki,
                                        kj,
                                        taps_per_output_channel,
                                        out_w,
                                        in_c,
                                        kh,
                                        kw,
                                    )?;

                                    let lr = compose_lower_for_geometry(lp_r[t], relax);
                                    nlp_r[t] = lr.new_coeff;
                                    add_lower_bias(nlb_r, lr.intercept_contrib);
                                    *lnf_r |= lr.nonfinite;

                                    let ur = compose_upper_for_geometry(up_r[t], relax);
                                    nup_r[t] = ur.new_coeff;
                                    add_upper_bias(nub_r, ur.intercept_contrib);
                                    *unf_r |= ur.nonfinite;
                                }
                            }
                        }
                    }
                }
            }
            if POLL {
                poll()?;
            }
            Ok(())
        };

        let ran_flat = row_volume > 0
            && match (
                lower_patches.as_slice(),
                upper_patches.as_slice(),
                new_lower_patches.as_slice_mut(),
                new_upper_patches.as_slice_mut(),
                new_lower_b_f64.as_slice_mut(),
                new_upper_b_f64.as_slice_mut(),
            ) {
                (Some(lp), Some(up), Some(nlp), Some(nup), Some(nlb), Some(nub)) => {
                    if POLL && !parallel_rows {
                        nlp.chunks_mut(row_volume)
                            .zip(nup.chunks_mut(row_volume))
                            .zip(lp.chunks(row_volume))
                            .zip(up.chunks(row_volume))
                            .zip(&mut nlb[..bounds.row_count])
                            .zip(&mut nub[..bounds.row_count])
                            .zip(&mut lower_nonfinite)
                            .zip(&mut upper_nonfinite)
                            .try_for_each(|item| {
                                let (
                                    ((((((nlp_r, nup_r), lp_r), up_r), nlb_r), nub_r), lnf_r),
                                    unf_r,
                                ) = item;
                                compose_row_7d(lp_r, up_r, nlp_r, nup_r, nlb_r, nub_r, lnf_r, unf_r)
                            })?;
                    } else if POLL {
                        nlp.par_chunks_mut(row_volume)
                            .zip(nup.par_chunks_mut(row_volume))
                            .zip(lp.par_chunks(row_volume))
                            .zip(up.par_chunks(row_volume))
                            .zip(&mut nlb[..bounds.row_count])
                            .zip(&mut nub[..bounds.row_count])
                            .zip(&mut lower_nonfinite)
                            .zip(&mut upper_nonfinite)
                            .try_for_each(|item| {
                                let (
                                    ((((((nlp_r, nup_r), lp_r), up_r), nlb_r), nub_r), lnf_r),
                                    unf_r,
                                ) = item;
                                compose_row_7d(lp_r, up_r, nlp_r, nup_r, nlb_r, nub_r, lnf_r, unf_r)
                            })?;
                    } else {
                        nlp.par_chunks_mut(row_volume)
                            .zip(nup.par_chunks_mut(row_volume))
                            .zip(lp.par_chunks(row_volume))
                            .zip(up.par_chunks(row_volume))
                            .zip(&mut nlb[..bounds.row_count])
                            .zip(&mut nub[..bounds.row_count])
                            .zip(&mut lower_nonfinite)
                            .zip(&mut upper_nonfinite)
                            .for_each(|item| {
                                let (
                                    ((((((nlp_r, nup_r), lp_r), up_r), nlb_r), nub_r), lnf_r),
                                    unf_r,
                                ) = item;
                                compose_row_7d(
                                    lp_r, up_r, nlp_r, nup_r, nlb_r, nub_r, lnf_r, unf_r,
                                )
                                .expect("deadline polling is disabled");
                            });
                    }
                    true
                }
                _ => false,
            };

        if !ran_flat {
            for row in 0..bounds.row_count {
                let mut coordinates_since_poll = 0usize;
                if POLL {
                    poll()?;
                }
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        poll_explicit_row_coordinate::<POLL, _>(
                                            &mut coordinates_since_poll,
                                            poll,
                                        )?;
                                        let Some(input_flat) =
                                            mapped_input_flat(oh, ow, ic, ki, kj)?
                                        else {
                                            continue;
                                        };
                                        let relax = &relaxations[input_flat];

                                        let la = lower_patches[[row, oc, oh, ow, ic, ki, kj]];
                                        let lr = compose_lower_for_geometry(la, relax);
                                        new_lower_patches[[row, oc, oh, ow, ic, ki, kj]] =
                                            lr.new_coeff;
                                        add_lower_bias(
                                            &mut new_lower_b_f64[row],
                                            lr.intercept_contrib,
                                        );
                                        lower_nonfinite[row] |= lr.nonfinite;

                                        let ua = upper_patches[[row, oc, oh, ow, ic, ki, kj]];
                                        let ur = compose_upper_for_geometry(ua, relax);
                                        new_upper_patches[[row, oc, oh, ow, ic, ki, kj]] =
                                            ur.new_coeff;
                                        add_upper_bias(
                                            &mut new_upper_b_f64[row],
                                            ur.intercept_contrib,
                                        );
                                        upper_nonfinite[row] |= ur.nonfinite;
                                    }
                                }
                            }
                        }
                    }
                }
                if POLL {
                    poll()?;
                }
            }
        }
    } else {
        let compose_row_6d = |j: usize,
                              lp_j: &[f32],
                              up_j: &[f32],
                              nlp_j: &mut [f32],
                              nup_j: &mut [f32],
                              nlb_j: &mut f64,
                              nub_j: &mut f64,
                              lnf_j: &mut bool,
                              unf_j: &mut bool|
         -> Result<()> {
            let oh = (j % spatial_positions) / out_w;
            let ow = j % out_w;
            let mut coordinates_since_poll = 0usize;
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        poll_explicit_row_coordinate::<POLL, _>(&mut coordinates_since_poll, poll)?;
                        let Some(input_flat) = mapped_input_flat(oh, ow, ic, ki, kj)? else {
                            continue;
                        };
                        let relax = &relaxations[input_flat];
                        // Flat tap index within the row's contiguous chunk
                        let t = checked_activation_tap_index(0, 0, ic, ki, kj, 1, in_c, kh, kw)?;

                        let lr = compose_lower_for_geometry(lp_j[t], relax);
                        nlp_j[t] = lr.new_coeff;
                        add_lower_bias(nlb_j, lr.intercept_contrib);
                        *lnf_j |= lr.nonfinite;

                        let ur = compose_upper_for_geometry(up_j[t], relax);
                        nup_j[t] = ur.new_coeff;
                        add_upper_bias(nub_j, ur.intercept_contrib);
                        *unf_j |= ur.nonfinite;
                    }
                }
            }
            if POLL {
                poll()?;
            }
            Ok(())
        };

        let ran_parallel = patch_volume > 0
            && match (
                lower_patches.as_slice(),
                upper_patches.as_slice(),
                new_lower_patches.as_slice_mut(),
                new_upper_patches.as_slice_mut(),
                new_lower_b_f64.as_slice_mut(),
                new_upper_b_f64.as_slice_mut(),
            ) {
                (Some(lp), Some(up), Some(nlp), Some(nup), Some(nlb), Some(nub)) => {
                    let rows = nlp
                        .par_chunks_mut(patch_volume)
                        .zip(nup.par_chunks_mut(patch_volume))
                        .zip(lp.par_chunks(patch_volume))
                        .zip(up.par_chunks(patch_volume))
                        .zip(&mut nlb[..logical_rows])
                        .zip(&mut nub[..logical_rows])
                        .zip(&mut lower_nonfinite)
                        .zip(&mut upper_nonfinite)
                        .enumerate();
                    if POLL {
                        rows.try_for_each(
                            |(
                                j,
                                (((((((nlp_j, nup_j), lp_j), up_j), nlb_j), nub_j), lnf_j), unf_j),
                            )| {
                                compose_row_6d(
                                    j, lp_j, up_j, nlp_j, nup_j, nlb_j, nub_j, lnf_j, unf_j,
                                )
                            },
                        )?;
                    } else {
                        rows.for_each(
                            |(
                                j,
                                (((((((nlp_j, nup_j), lp_j), up_j), nlb_j), nub_j), lnf_j), unf_j),
                            )| {
                                compose_row_6d(
                                    j, lp_j, up_j, nlp_j, nup_j, nlb_j, nub_j, lnf_j, unf_j,
                                )
                                .expect("deadline polling is disabled");
                            },
                        );
                    }
                    true
                }
                _ => false,
            };

        if !ran_parallel {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let j =
                            checked_activation_output_index(oc, oh, ow, spatial_positions, out_w)?;
                        let mut coordinates_since_poll = 0usize;

                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    poll_explicit_row_coordinate::<POLL, _>(
                                        &mut coordinates_since_poll,
                                        poll,
                                    )?;
                                    let Some(input_flat) = mapped_input_flat(oh, ow, ic, ki, kj)?
                                    else {
                                        continue;
                                    };
                                    let relax = &relaxations[input_flat];

                                    let la = lower_patches[[oc, oh, ow, ic, ki, kj]];
                                    let lr = compose_lower_for_geometry(la, relax);
                                    new_lower_patches[[oc, oh, ow, ic, ki, kj]] = lr.new_coeff;
                                    add_lower_bias(&mut new_lower_b_f64[j], lr.intercept_contrib);
                                    lower_nonfinite[j] |= lr.nonfinite;

                                    let ua = upper_patches[[oc, oh, ow, ic, ki, kj]];
                                    let ur = compose_upper_for_geometry(ua, relax);
                                    new_upper_patches[[oc, oh, ow, ic, ki, kj]] = ur.new_coeff;
                                    add_upper_bias(&mut new_upper_b_f64[j], ur.intercept_contrib);
                                    upper_nonfinite[j] |= ur.nonfinite;
                                }
                            }
                        }
                        if POLL {
                            poll()?;
                        }
                    }
                }
            }
        }
    }

    if POLL {
        poll()?;
    }

    // #3009: Non-finite row fallback for Patches activation CROWN backward.
    let mut lower_affected = 0usize;
    let mut upper_affected = 0usize;
    for (index, (&lower, &upper)) in lower_nonfinite
        .iter()
        .zip(upper_nonfinite.iter())
        .enumerate()
    {
        if index.is_multiple_of(EXPLICIT_ROW_DEADLINE_POLL_COORDS) && POLL {
            poll()?;
        }
        lower_affected += usize::from(lower);
        upper_affected += usize::from(upper);
    }
    compose::log_nonfinite_fallback(
        "Patches activation",
        lower_affected,
        upper_affected,
        logical_rows,
    );

    // Certified coefficient error + intercept-error discharge
    // (#patches-coeff-err-soundness; 7D lift per
    // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §6). Mirrors the Dense
    // `crown_activation_error_step`: the activation backward scales each
    // incoming coefficient `a` by a per-neuron relaxation slope, so the stored
    // f32 coefficient `next_down/up_f32(fl32(a·slope))` differs from the true
    // real coefficient by (1) the incoming per-row error `a_err` possibly
    // flipping `a`'s sign and selecting the OTHER envelope slope — bounded by
    // `a_err·(|lower_slope|+|upper_slope|)` — and (2) the EXACT f32-multiply +
    // directed-rounding gap `|a·slope_used − stored|` (computed here, reduced
    // to the per-row max; both are over-bounds of the true per-coefficient
    // max, since max(x+y) ≤ max x + max y). The relaxation intercept folded
    // into the bias likewise picks up
    // `a_err·(|lower_intercept|+|upper_intercept|)`, discharged OUTWARD into
    // the f64 bias BEFORE the directed cast below.
    //
    // 6D arm: err index = flat output position (out_c·out_h·out_w rows);
    // byte-identical to the certified 6D design. 7D explicit-rows arm: the
    // err index is the SPEC row (axis 0, len row_count == bias len, spec I1);
    // the identical per-tap formulas are reduced over the WHOLE spec row —
    // MAX for the err terms (one scalar must cover every coefficient of the
    // row), SUM for the bias discharges (every output position's fold lands
    // in the row's single bias slot) — plus two 7D-only f64-summation
    // discharges beyond the literal 6D mirror (spec §6.1/R4, adjudication
    // A1): `(1+gbar)` on the intercept sum IS covers the nearest-f64
    // summation under-estimate of IS at 7D tap counts, and `gbar·ABS`
    // certifies the f64 rounding of the compose pass's own intercept fold
    // (up to row_volume adds per row — ~2^-29 relative at cifar scale, which
    // can escape the 0.5-ulp32 cast slack under bias cancellation), where
    // gbar = γ_(8·row_volume+16). Both terms only widen and vanish as
    // row_volume shrinks.
    let (lower_coeff_err, upper_coeff_err) = if explicit_rows {
        let old_lower_err = lower_a_data.coeff_err.as_ref();
        let old_upper_err = upper_a_data.coeff_err.as_ref();
        let mut new_lower_err = if let Some(admission) = anchored_admission.as_mut() {
            ndarray::Array1::from_vec(admission.zeroed(
                logical_rows,
                0.0f32,
                "Anchored activation lower coefficient-error allocation",
                poll,
            )?)
        } else {
            ndarray::Array1::<f32>::zeros(logical_rows)
        };
        let mut new_upper_err = if let Some(admission) = anchored_admission.as_mut() {
            ndarray::Array1::from_vec(admission.zeroed(
                logical_rows,
                0.0f32,
                "Anchored activation upper coefficient-error allocation",
                poll,
            )?)
        } else {
            ndarray::Array1::<f32>::zeros(logical_rows)
        };

        // gbar = γ_(8·row_volume+16) (Higham, f64 unit roundoff): ≥ 4×
        // headroom over the γ_(2·rv+4) needed by the IS/ABS accumulation
        // deficits, `+16` covering the small-row_volume corner (spec §6.2
        // (2d)). Saturating: absurd row volumes drive gbar → +INF, which
        // poisons the bias outward rather than under-counting.
        let gamma_bar = crate::layers::linear::crown_single_gamma_n_f64(
            row_volume.saturating_mul(8).saturating_add(16),
        );

        // I5 sanitize at consumption: non-finite or NEGATIVE carried err
        // poisons to +INF (outward degrade), NEVER NaN -> 0 (false-proof
        // hazard). Direct index is total: length was hard-checked above.
        let sanitize = |v: f32| -> f64 {
            if v.is_finite() && v >= 0.0 {
                f32_to_f64_exact(v)
            } else {
                f64::INFINITY
            }
        };

        // Per-row err pass (READ-ONLY over all coefficient tensors, runs
        // after compose so the stored new coefficients give EXACT gaps; spec
        // I3). Per spec row r, per side σ, in f64 over the fixed serial tap
        // order oc->oh->ow->ic->ki->kj (same padding predicate and
        // input_flat mapping as compose_row_7d):
        //   MSS  = max_t (|ls|+|us|)              (err term, MAX-lift)
        //   IS   = Σ_t (|li|+|ui|)                (every valid tap, incl a==0)
        //   GAP_σ = max_{t, a_σ≠0} |f64(a_σ)·f64(s_σ(a_σ)) − f64(stored_σ)|
        //   ABS_σ = |f64(b_σ[r])| + Σ_{t, a_σ≠0} |f64(a_σ)·f64(i_σ(a_σ))|
        //   D_σ  = gbar·ABS_σ + (oe_σ≠0 ? oe_σ·(IS·(1+gbar)) : 0)
        //     -> lower b −= D_l / upper b += D_u (non-finite D poisons ∓INF)
        //   err_σ[r] = 0.0 on σ-nonfinite rows (vacuous certificate),
        //     else next_up_f32((oe_σ·MSS [if oe_σ≠0] + GAP_σ) as f32),
        //     +INF if non-finite (never NaN).
        // Writes only err[r] and b[r]; rows are disjoint, so the parallel
        // driver is bitwise identical to the serial fallback.
        let err_row_7d = |row: usize,
                          lp_r: &[f32],
                          up_r: &[f32],
                          nlp_r: &[f32],
                          nup_r: &[f32],
                          nlb_r: &mut f64,
                          nub_r: &mut f64,
                          nle_r: &mut f32,
                          nue_r: &mut f32|
         -> Result<()> {
            let mut coordinates_since_poll = 0usize;
            if POLL {
                poll()?;
            }
            let oe_l = old_lower_err.map_or(0.0, |e| sanitize(e[row]));
            let oe_u = old_upper_err.map_or(0.0, |e| sanitize(e[row]));

            let mut max_slope_sum = 0.0f64;
            let mut int_sum = 0.0f64;
            let mut max_lower_gap = 0.0f64;
            let mut max_upper_gap = 0.0f64;
            // ABS_σ initialized with |b_σ[r]| — the compose fold accumulates
            // starting from the incoming bias, so Higham's γ_n·ABS bound
            // must include it (spec §6.2 (2b)).
            let mut abs_lower_sum = f32_to_f64_exact(bounds.lower_b[row]).abs();
            let mut abs_upper_sum = f32_to_f64_exact(bounds.upper_b[row]).abs();
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    poll_explicit_row_coordinate::<POLL, _>(
                                        &mut coordinates_since_poll,
                                        poll,
                                    )?;
                                    let Some(input_flat) = mapped_input_flat(oh, ow, ic, ki, kj)?
                                    else {
                                        continue;
                                    };
                                    let relax = &relaxations[input_flat];
                                    let t = checked_activation_explicit_tap_index(
                                        oc,
                                        oh,
                                        ow,
                                        ic,
                                        ki,
                                        kj,
                                        taps_per_output_channel,
                                        out_w,
                                        in_c,
                                        kh,
                                        kw,
                                    )?;

                                    let ss = proof_add_nonnegative(
                                        f32_to_f64_exact(relax.lower_slope).abs(),
                                        f32_to_f64_exact(relax.upper_slope).abs(),
                                    );
                                    if ss > max_slope_sum {
                                        max_slope_sum = ss;
                                    }
                                    int_sum = proof_add_nonnegative(
                                        int_sum,
                                        proof_add_nonnegative(
                                            f32_to_f64_exact(relax.lower_intercept).abs(),
                                            f32_to_f64_exact(relax.upper_intercept).abs(),
                                        ),
                                    );

                                    // EXACT directed-rounding gap + |a·i| fold
                                    // magnitude per side (mirror compose_*):
                                    // compose_lower uses lower slope/intercept
                                    // for a>0 else upper; compose_upper the
                                    // reverse. a==0 taps skip both — compose
                                    // stores 0 exactly and folds no intercept.
                                    let la = lp_r[t];
                                    if coefficient_nonzero_for_geometry(la) {
                                        let (slope_used, intercept_used) =
                                            if coefficient_positive_for_geometry(la) {
                                                (relax.lower_slope, relax.lower_intercept)
                                            } else {
                                                (relax.upper_slope, relax.upper_intercept)
                                            };
                                        let gap = activation_product_gap(la, slope_used, nlp_r[t]);
                                        if gap > max_lower_gap {
                                            max_lower_gap = gap;
                                        }
                                        abs_lower_sum = proof_add_nonnegative(
                                            abs_lower_sum,
                                            proof_mul_nonnegative(
                                                f32_to_f64_exact(la).abs(),
                                                f32_to_f64_exact(intercept_used).abs(),
                                            ),
                                        );
                                    }
                                    let ua = up_r[t];
                                    if coefficient_nonzero_for_geometry(ua) {
                                        let (slope_used, intercept_used) =
                                            if coefficient_positive_for_geometry(ua) {
                                                (relax.upper_slope, relax.upper_intercept)
                                            } else {
                                                (relax.lower_slope, relax.lower_intercept)
                                            };
                                        let gap = activation_product_gap(ua, slope_used, nup_r[t]);
                                        if gap > max_upper_gap {
                                            max_upper_gap = gap;
                                        }
                                        abs_upper_sum = proof_add_nonnegative(
                                            abs_upper_sum,
                                            proof_mul_nonnegative(
                                                f32_to_f64_exact(ua).abs(),
                                                f32_to_f64_exact(intercept_used).abs(),
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Bias discharge D_σ into the f64 accumulator BEFORE the directed
            // cast (spec I4). `oe == 0` short-circuits `0·∞ = NaN` for
            // degenerate ±∞ relaxation intercepts (spec I5); any non-finite D
            // (∞ from a degenerate intercept in the receptive field, or NaN
            // from ∞·0) poisons the bias OUTWARD — skipping would emit a
            // finite bound the true range can escape (false-VERIFIED class).
            let inflated_intercept_sum =
                proof_mul_nonnegative(int_sum, proof_add_nonnegative(1.0, gamma_bar));
            let disc_l = proof_add_nonnegative(
                proof_mul_nonnegative(gamma_bar, abs_lower_sum),
                if oe_l != 0.0 {
                    proof_mul_nonnegative(oe_l, inflated_intercept_sum)
                } else {
                    0.0
                },
            );
            discharge_lower(nlb_r, disc_l);
            let disc_u = proof_add_nonnegative(
                proof_mul_nonnegative(gamma_bar, abs_upper_sum),
                if oe_u != 0.0 {
                    proof_mul_nonnegative(oe_u, inflated_intercept_sum)
                } else {
                    0.0
                },
            );
            discharge_upper(nub_r, disc_u);

            // Err emission: f64 compute, one outward next_up_f32 at the f32
            // cast (spec I4). Non-finite (∞ overflow or NaN from ∞·0) emits
            // +INF — the degrade poison — NEVER NaN (spec I5). Nonfinite
            // rows are zeroed + bias-poisoned by the #3009 fallback below, so
            // err 0.0 is exact there (vacuous certificate).
            let lterm = if oe_l != 0.0 {
                proof_mul_nonnegative(oe_l, max_slope_sum)
            } else {
                0.0
            };
            let uterm = if oe_u != 0.0 {
                proof_mul_nonnegative(oe_u, max_slope_sum)
            } else {
                0.0
            };
            let lv = proof_add_nonnegative(lterm, max_lower_gap);
            let uv = proof_add_nonnegative(uterm, max_upper_gap);
            *nle_r = if lower_nonfinite[row] {
                0.0
            } else if !lv.is_finite() {
                f32::INFINITY
            } else {
                publish_activation_error_up(lv)
            };
            *nue_r = if upper_nonfinite[row] {
                0.0
            } else if !uv.is_finite() {
                f32::INFINITY
            } else {
                publish_activation_error_up(uv)
            };
            if POLL {
                poll()?;
            }
            Ok(())
        };

        let ran_flat = row_volume > 0
            && match (
                lower_patches.as_slice(),
                upper_patches.as_slice(),
                new_lower_patches.as_slice(),
                new_upper_patches.as_slice(),
                new_lower_b_f64.as_slice_mut(),
                new_upper_b_f64.as_slice_mut(),
                new_lower_err.as_slice_mut(),
                new_upper_err.as_slice_mut(),
            ) {
                (
                    Some(lp),
                    Some(up),
                    Some(nlp),
                    Some(nup),
                    Some(nlb),
                    Some(nub),
                    Some(nle),
                    Some(nue),
                ) => {
                    if POLL && !parallel_rows {
                        nle.iter_mut()
                            .zip(nue.iter_mut())
                            .zip(&mut nlb[..bounds.row_count])
                            .zip(&mut nub[..bounds.row_count])
                            .zip(lp.chunks(row_volume))
                            .zip(up.chunks(row_volume))
                            .zip(nlp.chunks(row_volume))
                            .zip(nup.chunks(row_volume))
                            .enumerate()
                            .try_for_each(|item| {
                                let (
                                    row,
                                    (
                                        ((((((nle_r, nue_r), nlb_r), nub_r), lp_r), up_r), nlp_r),
                                        nup_r,
                                    ),
                                ) = item;
                                err_row_7d(
                                    row, lp_r, up_r, nlp_r, nup_r, nlb_r, nub_r, nle_r, nue_r,
                                )
                            })?;
                    } else if POLL {
                        nle.par_iter_mut()
                            .zip(nue.par_iter_mut())
                            .zip(&mut nlb[..bounds.row_count])
                            .zip(&mut nub[..bounds.row_count])
                            .zip(lp.par_chunks(row_volume))
                            .zip(up.par_chunks(row_volume))
                            .zip(nlp.par_chunks(row_volume))
                            .zip(nup.par_chunks(row_volume))
                            .enumerate()
                            .try_for_each(|item| {
                                let (
                                    row,
                                    (
                                        ((((((nle_r, nue_r), nlb_r), nub_r), lp_r), up_r), nlp_r),
                                        nup_r,
                                    ),
                                ) = item;
                                err_row_7d(
                                    row, lp_r, up_r, nlp_r, nup_r, nlb_r, nub_r, nle_r, nue_r,
                                )
                            })?;
                    } else {
                        nle.par_iter_mut()
                            .zip(nue.par_iter_mut())
                            .zip(&mut nlb[..bounds.row_count])
                            .zip(&mut nub[..bounds.row_count])
                            .zip(lp.par_chunks(row_volume))
                            .zip(up.par_chunks(row_volume))
                            .zip(nlp.par_chunks(row_volume))
                            .zip(nup.par_chunks(row_volume))
                            .enumerate()
                            .for_each(|item| {
                                let (
                                    row,
                                    (
                                        ((((((nle_r, nue_r), nlb_r), nub_r), lp_r), up_r), nlp_r),
                                        nup_r,
                                    ),
                                ) = item;
                                err_row_7d(
                                    row, lp_r, up_r, nlp_r, nup_r, nlb_r, nub_r, nle_r, nue_r,
                                )
                                .expect("deadline polling is disabled");
                            });
                    }
                    true
                }
                _ => false,
            };

        if !ran_flat {
            for row in 0..bounds.row_count {
                let mut coordinates_since_poll = 0usize;
                if POLL {
                    poll()?;
                }
                let oe_l = old_lower_err.map_or(0.0, |e| sanitize(e[row]));
                let oe_u = old_upper_err.map_or(0.0, |e| sanitize(e[row]));

                let mut max_slope_sum = 0.0f64;
                let mut int_sum = 0.0f64;
                let mut max_lower_gap = 0.0f64;
                let mut max_upper_gap = 0.0f64;
                let mut abs_lower_sum = f32_to_f64_exact(bounds.lower_b[row]).abs();
                let mut abs_upper_sum = f32_to_f64_exact(bounds.upper_b[row]).abs();
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        poll_explicit_row_coordinate::<POLL, _>(
                                            &mut coordinates_since_poll,
                                            poll,
                                        )?;
                                        let Some(input_flat) =
                                            mapped_input_flat(oh, ow, ic, ki, kj)?
                                        else {
                                            continue;
                                        };
                                        let relax = &relaxations[input_flat];

                                        let ss = proof_add_nonnegative(
                                            f32_to_f64_exact(relax.lower_slope).abs(),
                                            f32_to_f64_exact(relax.upper_slope).abs(),
                                        );
                                        if ss > max_slope_sum {
                                            max_slope_sum = ss;
                                        }
                                        int_sum = proof_add_nonnegative(
                                            int_sum,
                                            proof_add_nonnegative(
                                                f32_to_f64_exact(relax.lower_intercept).abs(),
                                                f32_to_f64_exact(relax.upper_intercept).abs(),
                                            ),
                                        );

                                        // EXACT gap + |a·i| magnitude per side
                                        // (see the closure above).
                                        let la = lower_patches[[row, oc, oh, ow, ic, ki, kj]];
                                        if coefficient_nonzero_for_geometry(la) {
                                            let (slope_used, intercept_used) =
                                                if coefficient_positive_for_geometry(la) {
                                                    (relax.lower_slope, relax.lower_intercept)
                                                } else {
                                                    (relax.upper_slope, relax.upper_intercept)
                                                };
                                            let gap = activation_product_gap(
                                                la,
                                                slope_used,
                                                new_lower_patches[[row, oc, oh, ow, ic, ki, kj]],
                                            );
                                            if gap > max_lower_gap {
                                                max_lower_gap = gap;
                                            }
                                            abs_lower_sum = proof_add_nonnegative(
                                                abs_lower_sum,
                                                proof_mul_nonnegative(
                                                    f32_to_f64_exact(la).abs(),
                                                    f32_to_f64_exact(intercept_used).abs(),
                                                ),
                                            );
                                        }
                                        let ua = upper_patches[[row, oc, oh, ow, ic, ki, kj]];
                                        if coefficient_nonzero_for_geometry(ua) {
                                            let (slope_used, intercept_used) =
                                                if coefficient_positive_for_geometry(ua) {
                                                    (relax.upper_slope, relax.upper_intercept)
                                                } else {
                                                    (relax.lower_slope, relax.lower_intercept)
                                                };
                                            let gap = activation_product_gap(
                                                ua,
                                                slope_used,
                                                new_upper_patches[[row, oc, oh, ow, ic, ki, kj]],
                                            );
                                            if gap > max_upper_gap {
                                                max_upper_gap = gap;
                                            }
                                            abs_upper_sum = proof_add_nonnegative(
                                                abs_upper_sum,
                                                proof_mul_nonnegative(
                                                    f32_to_f64_exact(ua).abs(),
                                                    f32_to_f64_exact(intercept_used).abs(),
                                                ),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Discharge + err write — same rule as the closure above
                // (parallel and serial drivers are bitwise identical).
                let inflated_intercept_sum =
                    proof_mul_nonnegative(int_sum, proof_add_nonnegative(1.0, gamma_bar));
                let disc_l = proof_add_nonnegative(
                    proof_mul_nonnegative(gamma_bar, abs_lower_sum),
                    if oe_l != 0.0 {
                        proof_mul_nonnegative(oe_l, inflated_intercept_sum)
                    } else {
                        0.0
                    },
                );
                discharge_lower(&mut new_lower_b_f64[row], disc_l);
                let disc_u = proof_add_nonnegative(
                    proof_mul_nonnegative(gamma_bar, abs_upper_sum),
                    if oe_u != 0.0 {
                        proof_mul_nonnegative(oe_u, inflated_intercept_sum)
                    } else {
                        0.0
                    },
                );
                discharge_upper(&mut new_upper_b_f64[row], disc_u);

                let lterm = if oe_l != 0.0 {
                    proof_mul_nonnegative(oe_l, max_slope_sum)
                } else {
                    0.0
                };
                let uterm = if oe_u != 0.0 {
                    proof_mul_nonnegative(oe_u, max_slope_sum)
                } else {
                    0.0
                };
                let lv = proof_add_nonnegative(lterm, max_lower_gap);
                let uv = proof_add_nonnegative(uterm, max_upper_gap);
                new_lower_err[row] = if lower_nonfinite[row] {
                    0.0
                } else if !lv.is_finite() {
                    f32::INFINITY
                } else {
                    publish_activation_error_up(lv)
                };
                new_upper_err[row] = if upper_nonfinite[row] {
                    0.0
                } else if !uv.is_finite() {
                    f32::INFINITY
                } else {
                    publish_activation_error_up(uv)
                };
                if POLL {
                    poll()?;
                }
            }
        }
        // Always Some/Some: the GAP terms (and the gbar·ABS discharge) are
        // intrinsic to the compose pass, even with exact (None) inputs.
        (Some(new_lower_err), Some(new_upper_err))
    } else {
        let old_lower_err = lower_a_data.coeff_err.as_ref();
        let old_upper_err = upper_a_data.coeff_err.as_ref();
        let mut new_lower_err = if let Some(admission) = anchored_admission.as_mut() {
            ndarray::Array1::from_vec(admission.zeroed(
                logical_rows,
                0.0f32,
                "Anchored activation lower coefficient-error allocation",
                poll,
            )?)
        } else {
            ndarray::Array1::<f32>::zeros(logical_rows)
        };
        let mut new_upper_err = if let Some(admission) = anchored_admission.as_mut() {
            ndarray::Array1::from_vec(admission.zeroed(
                logical_rows,
                0.0f32,
                "Anchored activation upper coefficient-error allocation",
                poll,
            )?)
        } else {
            ndarray::Array1::<f32>::zeros(logical_rows)
        };
        let sanitize = |value: f32| -> f64 {
            if value.is_finite() && value >= 0.0 {
                f32_to_f64_exact(value)
            } else {
                f64::INFINITY
            }
        };

        // Per-row parallel error step: row j reads its own old err / patches
        // chunk / already-written new patches chunk and writes only err[j] and
        // b[j] — no cross-row state, tap order within the row unchanged
        // (int_sum accumulates in the serial ic/ki/kj order). Value-identical.
        let err_row_6d = |j: usize,
                          lp_j: &[f32],
                          up_j: &[f32],
                          nlp_j: &[f32],
                          nup_j: &[f32],
                          nlb_j: &mut f64,
                          nub_j: &mut f64,
                          nle_j: &mut f32,
                          nue_j: &mut f32|
         -> Result<()> {
            let oh = (j % spatial_positions) / out_w;
            let ow = j % out_w;
            let mut coordinates_since_poll = 0usize;
            if POLL {
                poll()?;
            }
            let oe_l = old_lower_err.map_or(0.0, |e| sanitize(e[j]));
            let oe_u = old_upper_err.map_or(0.0, |e| sanitize(e[j]));

            let mut max_slope_sum = 0.0f64;
            let mut int_sum = 0.0f64;
            let mut max_lower_gap = 0.0f64;
            let mut max_upper_gap = 0.0f64;
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        poll_explicit_row_coordinate::<POLL, _>(&mut coordinates_since_poll, poll)?;
                        let Some(input_flat) = mapped_input_flat(oh, ow, ic, ki, kj)? else {
                            continue;
                        };
                        let relax = &relaxations[input_flat];
                        let t = checked_activation_tap_index(0, 0, ic, ki, kj, 1, in_c, kh, kw)?;

                        let ss = proof_add_nonnegative(
                            f32_to_f64_exact(relax.lower_slope).abs(),
                            f32_to_f64_exact(relax.upper_slope).abs(),
                        );
                        if ss > max_slope_sum {
                            max_slope_sum = ss;
                        }
                        int_sum = proof_add_nonnegative(
                            int_sum,
                            proof_add_nonnegative(
                                f32_to_f64_exact(relax.lower_intercept).abs(),
                                f32_to_f64_exact(relax.upper_intercept).abs(),
                            ),
                        );

                        // EXACT directed-rounding gap per side (mirror compose_*):
                        // compose_lower uses lower_slope for a>0 else upper_slope;
                        // compose_upper uses upper_slope for a>0 else lower_slope.
                        let la = lp_j[t];
                        if coefficient_nonzero_for_geometry(la) {
                            let slope_used = if coefficient_positive_for_geometry(la) {
                                relax.lower_slope
                            } else {
                                relax.upper_slope
                            };
                            let gap = activation_product_gap(la, slope_used, nlp_j[t]);
                            if gap > max_lower_gap {
                                max_lower_gap = gap;
                            }
                        }
                        let ua = up_j[t];
                        if coefficient_nonzero_for_geometry(ua) {
                            let slope_used = if coefficient_positive_for_geometry(ua) {
                                relax.upper_slope
                            } else {
                                relax.lower_slope
                            };
                            let gap = activation_product_gap(ua, slope_used, nup_j[t]);
                            if gap > max_upper_gap {
                                max_upper_gap = gap;
                            }
                        }
                    }
                }
            }

            // Discharge the incoming-error intercept perturbation OUTWARD into the
            // f64 bias. `oe == 0` short-circuits `0·∞ = NaN` (nothing to
            // discharge). An INFINITE discharge (oe > 0 with a degenerate ±∞
            // relaxation intercept in the receptive field) means the certificate
            // admits a coefficient sign that folds an infinite intercept — the
            // only sound bound is vacuous, so poison the bias outward rather
            // than skipping (skipping would emit a finite bound the true range
            // can escape: false-VERIFIED class).
            if oe_l != 0.0 {
                let disc_l = proof_mul_nonnegative(oe_l, int_sum);
                discharge_lower(nlb_j, disc_l);
            }
            if oe_u != 0.0 {
                let disc_u = proof_mul_nonnegative(oe_u, int_sum);
                discharge_upper(nub_j, disc_u);
            }

            // `if oe == 0` short-circuits `0·∞ = NaN` for degenerate ∞ slopes.
            let lterm = if oe_l != 0.0 {
                proof_mul_nonnegative(oe_l, max_slope_sum)
            } else {
                0.0
            };
            let uterm = if oe_u != 0.0 {
                proof_mul_nonnegative(oe_u, max_slope_sum)
            } else {
                0.0
            };
            let lower_total = proof_add_nonnegative(lterm, max_lower_gap);
            let upper_total = proof_add_nonnegative(uterm, max_upper_gap);
            *nle_j = if lower_nonfinite[j] {
                0.0
            } else if !lower_total.is_finite() {
                f32::INFINITY
            } else {
                publish_activation_error_up(lower_total)
            };
            *nue_j = if upper_nonfinite[j] {
                0.0
            } else if !upper_total.is_finite() {
                f32::INFINITY
            } else {
                publish_activation_error_up(upper_total)
            };
            if POLL {
                poll()?;
            }
            Ok(())
        };

        let ran_parallel = patch_volume > 0
            && match (
                lower_patches.as_slice(),
                upper_patches.as_slice(),
                new_lower_patches.as_slice(),
                new_upper_patches.as_slice(),
                new_lower_b_f64.as_slice_mut(),
                new_upper_b_f64.as_slice_mut(),
                new_lower_err.as_slice_mut(),
                new_upper_err.as_slice_mut(),
            ) {
                (
                    Some(lp),
                    Some(up),
                    Some(nlp),
                    Some(nup),
                    Some(nlb),
                    Some(nub),
                    Some(nle),
                    Some(nue),
                ) => {
                    if POLL {
                        nle.par_iter_mut()
                            .zip(nue.par_iter_mut())
                            .zip(&mut nlb[..logical_rows])
                            .zip(&mut nub[..logical_rows])
                            .zip(lp.par_chunks(patch_volume))
                            .zip(up.par_chunks(patch_volume))
                            .zip(nlp.par_chunks(patch_volume))
                            .zip(nup.par_chunks(patch_volume))
                            .enumerate()
                            .try_for_each(
                                |(
                                    j,
                                    (
                                        ((((((nle_j, nue_j), nlb_j), nub_j), lp_j), up_j), nlp_j),
                                        nup_j,
                                    ),
                                )| {
                                    err_row_6d(
                                        j, lp_j, up_j, nlp_j, nup_j, nlb_j, nub_j, nle_j, nue_j,
                                    )
                                },
                            )?;
                    } else {
                        nle.par_iter_mut()
                            .zip(nue.par_iter_mut())
                            .zip(&mut nlb[..logical_rows])
                            .zip(&mut nub[..logical_rows])
                            .zip(lp.par_chunks(patch_volume))
                            .zip(up.par_chunks(patch_volume))
                            .zip(nlp.par_chunks(patch_volume))
                            .zip(nup.par_chunks(patch_volume))
                            .enumerate()
                            .for_each(
                                |(
                                    j,
                                    (
                                        ((((((nle_j, nue_j), nlb_j), nub_j), lp_j), up_j), nlp_j),
                                        nup_j,
                                    ),
                                )| {
                                    err_row_6d(
                                        j, lp_j, up_j, nlp_j, nup_j, nlb_j, nub_j, nle_j, nue_j,
                                    )
                                    .expect("deadline polling is disabled");
                                },
                            );
                    }
                    true
                }
                _ => false,
            };

        if !ran_parallel {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let j =
                            checked_activation_output_index(oc, oh, ow, spatial_positions, out_w)?;
                        let mut coordinates_since_poll = 0usize;
                        if POLL {
                            poll()?;
                        }
                        let oe_l = old_lower_err.map_or(0.0, |e| sanitize(e[j]));
                        let oe_u = old_upper_err.map_or(0.0, |e| sanitize(e[j]));

                        let mut max_slope_sum = 0.0f64;
                        let mut int_sum = 0.0f64;
                        let mut max_lower_gap = 0.0f64;
                        let mut max_upper_gap = 0.0f64;
                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    poll_explicit_row_coordinate::<POLL, _>(
                                        &mut coordinates_since_poll,
                                        poll,
                                    )?;
                                    let Some(input_flat) = mapped_input_flat(oh, ow, ic, ki, kj)?
                                    else {
                                        continue;
                                    };
                                    let relax = &relaxations[input_flat];

                                    let ss = proof_add_nonnegative(
                                        f32_to_f64_exact(relax.lower_slope).abs(),
                                        f32_to_f64_exact(relax.upper_slope).abs(),
                                    );
                                    if ss > max_slope_sum {
                                        max_slope_sum = ss;
                                    }
                                    int_sum = proof_add_nonnegative(
                                        int_sum,
                                        proof_add_nonnegative(
                                            f32_to_f64_exact(relax.lower_intercept).abs(),
                                            f32_to_f64_exact(relax.upper_intercept).abs(),
                                        ),
                                    );

                                    // EXACT directed-rounding gap per side (mirror compose_*):
                                    // compose_lower uses lower_slope for a>0 else upper_slope;
                                    // compose_upper uses upper_slope for a>0 else lower_slope.
                                    let la = lower_patches[[oc, oh, ow, ic, ki, kj]];
                                    if coefficient_nonzero_for_geometry(la) {
                                        let slope_used = if coefficient_positive_for_geometry(la) {
                                            relax.lower_slope
                                        } else {
                                            relax.upper_slope
                                        };
                                        let gap = activation_product_gap(
                                            la,
                                            slope_used,
                                            new_lower_patches[[oc, oh, ow, ic, ki, kj]],
                                        );
                                        if gap > max_lower_gap {
                                            max_lower_gap = gap;
                                        }
                                    }
                                    let ua = upper_patches[[oc, oh, ow, ic, ki, kj]];
                                    if coefficient_nonzero_for_geometry(ua) {
                                        let slope_used = if coefficient_positive_for_geometry(ua) {
                                            relax.upper_slope
                                        } else {
                                            relax.lower_slope
                                        };
                                        let gap = activation_product_gap(
                                            ua,
                                            slope_used,
                                            new_upper_patches[[oc, oh, ow, ic, ki, kj]],
                                        );
                                        if gap > max_upper_gap {
                                            max_upper_gap = gap;
                                        }
                                    }
                                }
                            }
                        }

                        // Discharge the incoming-error intercept perturbation OUTWARD into
                        // the f64 bias — see the parallel path above for the soundness
                        // argument (infinite discharge poisons the bias outward).
                        if oe_l != 0.0 {
                            let disc_l = proof_mul_nonnegative(oe_l, int_sum);
                            discharge_lower(&mut new_lower_b_f64[j], disc_l);
                        }
                        if oe_u != 0.0 {
                            let disc_u = proof_mul_nonnegative(oe_u, int_sum);
                            discharge_upper(&mut new_upper_b_f64[j], disc_u);
                        }

                        // `if oe == 0` short-circuits `0·∞ = NaN` for degenerate ∞ slopes.
                        let lterm = if oe_l != 0.0 {
                            proof_mul_nonnegative(oe_l, max_slope_sum)
                        } else {
                            0.0
                        };
                        let uterm = if oe_u != 0.0 {
                            proof_mul_nonnegative(oe_u, max_slope_sum)
                        } else {
                            0.0
                        };
                        let lower_total = proof_add_nonnegative(lterm, max_lower_gap);
                        let upper_total = proof_add_nonnegative(uterm, max_upper_gap);
                        new_lower_err[j] = if lower_nonfinite[j] {
                            0.0
                        } else if !lower_total.is_finite() {
                            f32::INFINITY
                        } else {
                            publish_activation_error_up(lower_total)
                        };
                        new_upper_err[j] = if upper_nonfinite[j] {
                            0.0
                        } else if !upper_total.is_finite() {
                            f32::INFINITY
                        } else {
                            publish_activation_error_up(upper_total)
                        };
                        if POLL {
                            poll()?;
                        }
                    }
                }
            }
        }
        (Some(new_lower_err), Some(new_upper_err))
    };

    if POLL {
        poll()?;
    }
    let mut new_lower_b = if let Some(admission) = anchored_admission.as_mut() {
        let mut values = admission.reserve::<f32>(
            logical_rows,
            "Anchored activation lower published-bias allocation",
        )?;
        for (index, &value) in new_lower_b_f64.iter().enumerate() {
            if index.is_multiple_of(EXPLICIT_ROW_DEADLINE_POLL_COORDS) && POLL {
                poll()?;
            }
            values.push(f64_to_f32_down(value));
        }
        ndarray::Array1::from_vec(values)
    } else {
        new_lower_b_f64.mapv(|x| next_down_f32(x as f32))
    };
    let mut new_upper_b = if let Some(admission) = anchored_admission.as_mut() {
        let mut values = admission.reserve::<f32>(
            logical_rows,
            "Anchored activation upper published-bias allocation",
        )?;
        for (index, &value) in new_upper_b_f64.iter().enumerate() {
            if index.is_multiple_of(EXPLICIT_ROW_DEADLINE_POLL_COORDS) && POLL {
                poll()?;
            }
            values.push(f64_to_f32_up(value));
        }
        ndarray::Array1::from_vec(values)
    } else {
        new_upper_b_f64.mapv(|x| next_up_f32(x as f32))
    };

    if explicit_rows {
        for row in 0..bounds.row_count {
            if lower_nonfinite[row] {
                let mut coordinates_since_poll = 0usize;
                if POLL {
                    poll()?;
                }
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        poll_explicit_row_coordinate::<POLL, _>(
                                            &mut coordinates_since_poll,
                                            poll,
                                        )?;
                                        new_lower_patches[[row, oc, oh, ow, ic, ki, kj]] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
                new_lower_b[row] = f32::NEG_INFINITY;
            }
            if upper_nonfinite[row] {
                let mut coordinates_since_poll = 0usize;
                if POLL {
                    poll()?;
                }
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        poll_explicit_row_coordinate::<POLL, _>(
                                            &mut coordinates_since_poll,
                                            poll,
                                        )?;
                                        new_upper_patches[[row, oc, oh, ow, ic, ki, kj]] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
                new_upper_b[row] = f32::INFINITY;
            }
        }
    } else {
        for j in 0..logical_rows {
            if j.is_multiple_of(EXPLICIT_ROW_DEADLINE_POLL_COORDS) && POLL {
                poll()?;
            }
            let oc = j / spatial_positions;
            let oh = (j % spatial_positions) / out_w;
            let ow = j % out_w;
            if lower_nonfinite[j] {
                let mut coordinates_since_poll = 0usize;
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            poll_explicit_row_coordinate::<POLL, _>(
                                &mut coordinates_since_poll,
                                poll,
                            )?;
                            new_lower_patches[[oc, oh, ow, ic, ki, kj]] = 0.0;
                        }
                    }
                }
                new_lower_b[j] = f32::NEG_INFINITY;
            }
            if upper_nonfinite[j] {
                let mut coordinates_since_poll = 0usize;
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            poll_explicit_row_coordinate::<POLL, _>(
                                &mut coordinates_since_poll,
                                poll,
                            )?;
                            new_upper_patches[[oc, oh, ow, ic, ki, kj]] = 0.0;
                        }
                    }
                }
                new_upper_b[j] = f32::INFINITY;
            }
        }
    }

    if POLL {
        poll()?;
    }
    let mut folded = PatchesLinearBounds {
        row_count: bounds.row_count,
        lower_a: PatchesData {
            coeff_err: lower_coeff_err,
            patches: Some(new_lower_patches),
            geometry: lower_a_data.geometry.clone(),
            identity: false,
            output_shape: lower_a_data.output_shape,
            input_shape: lower_a_data.input_shape,
            unstable_idx: None,
        },
        lower_b: new_lower_b,
        upper_a: PatchesData {
            coeff_err: upper_coeff_err,
            patches: Some(new_upper_patches),
            geometry: upper_a_data.geometry.clone(),
            identity: false,
            output_shape: upper_a_data.output_shape,
            input_shape: upper_a_data.input_shape,
            unstable_idx: None,
        },
        upper_b: new_upper_b,
    };

    // #patches-eager-err: see the twin call in crown_patches_alpha.rs and the
    // enclosure argument in bounds/patches/eager_err.rs. Discharging against the
    // pre-activation cut here is what keeps the Patches path from paying an
    // IBP-scale, depth-amplified penalty for an error the dense path retires
    // immediately.
    // The eager fold helper is infallible and has no cooperative-deadline
    // surface. Retaining the carried error is conservative, so the finite
    // Anchored route defers this optional tightening rather than running an
    // uninterruptible O(A) materialization after its last poll.
    if crate::bounds::patches::eager_err_enabled() && !anchored_geometry {
        folded.fold_coeff_err_over_box_eager(pre_activation);
    }

    Ok(CrownBounds::Patches(Box::new(folded)))
}

#[cfg(test)]
mod deadline_admission_tests {
    use super::{
        anchored_activation_planned_bytes, checked_activation_tap_index,
        explicit_row_deadline_parallel_admitted, harden_anchored_relaxation,
        with_anchored_activation_budget_for_test, AnchoredActivationAdmission,
    };
    use crate::bounds::patches::{PatchGeometry, PatchesData, PatchesLinearBounds};
    use crate::layers::activations::LinearRelaxation;
    use ndarray::{Array1, ArrayD, IxDyn};
    use ny_core::NyError;
    use ny_tensor::BoundedTensor;

    #[test]
    fn explicit_row_deadline_refuses_nested_parallelism() {
        assert!(explicit_row_deadline_parallel_admitted(false));
        assert!(
            !explicit_row_deadline_parallel_admitted(true),
            "an inner parallel-region worker must use the serial row driver"
        );
    }

    #[test]
    fn anchored_relu_min_subnormal_crossing_chord_is_bit_exact_and_outward() {
        let tiny = f32::from_bits(1);
        let lower = ny_core::f64_to_f32_down(ny_core::f32_to_f64_exact(-tiny));
        let upper = ny_core::f64_to_f32_up(ny_core::f32_to_f64_exact(tiny));
        let relaxation = harden_anchored_relaxation(
            lower,
            upper,
            crate::layers::activations::relu::relu_linear_relaxation(lower, upper),
        );
        for field in [
            relaxation.lower_slope,
            relaxation.lower_intercept,
            relaxation.upper_slope,
            relaxation.upper_intercept,
        ] {
            let magnitude = field.to_bits() & 0x7fff_ffff;
            assert!(
                magnitude == 0 || magnitude >= 0x0080_0000,
                "Anchored relaxation field is subnormal: {field:e}"
            );
        }
        for x in [-tiny, 0.0, tiny] {
            let x64 = ny_core::f32_to_f64_exact(x);
            let relu = x64.max(0.0);
            let lower_line = ny_core::f32_to_f64_exact(relaxation.lower_slope) * x64
                + ny_core::f32_to_f64_exact(relaxation.lower_intercept);
            let upper_line = ny_core::f32_to_f64_exact(relaxation.upper_slope) * x64
                + ny_core::f32_to_f64_exact(relaxation.upper_intercept);
            assert!(lower_line <= relu, "lower chord escaped at {x:e}");
            assert!(upper_line >= relu, "upper chord escaped at {x:e}");
        }
    }

    #[test]
    fn anchored_activation_tap_index_overflow_is_typed() {
        assert!(matches!(
            checked_activation_tap_index(usize::MAX, 1, 0, 0, 0, 2, 1, 1, 1),
            Err(NyError::InvalidSpec(_))
        ));
    }

    #[test]
    fn anchored_activation_total_live_exact_budget_and_minus_one_are_atomic() {
        let width = 32_768usize;
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
        let bounds = PatchesLinearBounds {
            row_count: 1,
            lower_a: make_side(),
            lower_b: Array1::from_vec(vec![0.125]),
            upper_a: make_side(),
            upper_b: Array1::from_vec(vec![-0.25]),
        };
        let pre_activation = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, width]), -1.0),
            ArrayD::from_elem(IxDyn(&[1, 1, width]), 2.0),
        )
        .unwrap();
        let planned = anchored_activation_planned_bytes(width, width, 0, 0, width, 1).unwrap();
        let source = bounds.memory_bytes();
        let total = source.checked_add(planned).unwrap();
        assert!(source > 1 && planned < total);

        assert!(AnchoredActivationAdmission::with_budget(source, planned, total).is_ok());
        assert!(matches!(
            AnchoredActivationAdmission::with_budget(source, planned, total - 1),
            Err(NyError::CpuMemoryExceeded { .. })
        ));

        let completed = with_anchored_activation_budget_for_test(total, || {
            super::crown_elementwise_backward_patches_with_deadline(
                &bounds,
                &pre_activation,
                std::time::Instant::now() + std::time::Duration::from_secs(30),
                |_lower, _upper| LinearRelaxation::identity(),
            )
        });
        assert!(completed.is_ok(), "exact total-live budget must admit");

        let lower_before = bounds.lower_a.patches.as_ref().unwrap().clone();
        let upper_before = bounds.upper_a.patches.as_ref().unwrap().clone();
        let lower_bias_before = bounds.lower_b.clone();
        let upper_bias_before = bounds.upper_b.clone();
        let refused = with_anchored_activation_budget_for_test(total - 1, || {
            super::crown_elementwise_backward_patches_with_deadline(
                &bounds,
                &pre_activation,
                std::time::Instant::now() + std::time::Duration::from_secs(30),
                |_lower, _upper| LinearRelaxation::identity(),
            )
        });
        assert!(matches!(refused, Err(NyError::CpuMemoryExceeded { .. })));
        assert_eq!(bounds.lower_a.patches.as_ref().unwrap(), &lower_before);
        assert_eq!(bounds.upper_a.patches.as_ref().unwrap(), &upper_before);
        assert_eq!(bounds.lower_b, lower_bias_before);
        assert_eq!(bounds.upper_b, upper_bias_before);
    }
}
