// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-native CROWN backward for `ConvTranspose2dLayer` (LEVER 2).
//!
//! The dense ConvTranspose2d CROWN backward
//! (`bound_transpose.rs::propagate_linear_with_engine`) materializes a full
//! `[target_dim x conv_in]` dense coefficient pair. On the 28,800-dim cGAN
//! `BatchNormalization_11` target this measured 52.5s, starving alpha-opt and
//! BaB. This module keeps the coefficient in patches form (O(spec_rows x window
//! x kernel), ~ms) for the historical no-deadline stride-1 route and a
//! certified Anchored route for every stride under finite authority (as well as
//! every stride>1 request). The Anchored route supports exact full identities
//! and materialized 6D/7D composition.
//!
//! ## The reduction (stage 2a scope)
//!
//! A `ConvTranspose2d` forward `y = conv_transpose(x, W)` (ONNX kernel layout
//! `W[in_c, out_c, kh, kw]`, stride 1, dilation 1, output_padding 0, padding
//! `(ph, pw)`) has a CROWN backward that is *exactly* the CROWN backward of a
//! plain `Conv2d` with
//!   - kernel `Kc[oc, ic, ki, kj] = W[ic, oc, kh-1-ki, kw-1-kj]`
//!     (in/out channels swapped **and** the kernel spatially flipped), and
//!   - padding `Cp = (kh-1-ph, kw-1-pw)`, and
//!   - the SAME bias (per output channel, broadcast over the output grid).
//!
//! This is the standard "the adjoint of a transposed convolution is a
//! convolution" identity specialized to stride 1; it is verified numerically
//! against the dense `conv2d_forward_batched_gemm` backward (the exact operator
//! the dense path evaluates) in the proptest harness and unit tests. Reducing to
//! `Conv2dLayer::propagate_patches_engine` means this no-deadline stride-1
//! ConvTranspose backward introduces **zero new bound math**: the identity
//! build, non-identity composition, certified `coeff_err`, the outward-rounded
//! bias, and the `should_fallback_to_dense` memory guard are all the
//! already-proven Conv2d patches path (`bound_patches.rs`), which is pinned
//! bit-equivalent-to-dense by `crown_patches.rs`.
//!
//! ## Corners routed to the sound dense fallback
//!
//! The no-deadline stride-1 reduction still routes these configurations to
//! dense:
//!   - `dilation != (1, 1)`;
//!   - `output_padding != (0, 0)` (unreachable for stride 1, guarded anyway);
//!   - `padding.0 > kh-1` or `padding.1 > kw-1` — the equivalent Conv2d padding
//!     `kh-1-ph` would be negative (not representable without cropping);
//!   - `input_shape` unset;
//!   - a NaN kernel (→ `NumericalInstability`);
//!   - and, inside the delegated Conv2d path, non-identity incoming patches that
//!     carry nonzero composed padding (the Conv2d composition soundness guard).
//!
//! ## STAGE 2b/4 (stride>1): exact inverse-grid admission
//!
//! A forward stride-s ConvTranspose upsamples, so its CROWN backward
//! *downsamples* the coefficient grid (output pixels -> input pixels, /s). The
//! phase-partition (split the output grid into the s^2 residue classes
//! `(oh mod s, ow mod s)`; each class is a stride-1 ConvTranspose backward on the
//! decimated sub-grid with the per-phase kernel slice `W[:, :, (ph+a)%s :: s,
//! (pw+b)%s :: s]`, reducing to the stage-2a Conv2d patches path) COMPUTES the
//! backward correctly — validated bit-exact to the dense path for stride-2 and
//! stride-3 in `crown_patches_convtranspose.rs`
//! (`proptest_convtranspose2d_phase_partition_{identity,nonidentity}`).
//!
//! The historical affine geometry could not encode the required floor map.
//! `PatchesData` now has an exact separable `Anchored` geometry, and generic
//! dense scatter/materialization can carry it. For a full virtual identity the
//! backward has no coefficient contraction. DAZ-stable stored coefficients are
//! copied directly; binary32-subnormal weights are normalized to a zero center
//! and certified by a per-row `coeff_err`, so later sign selection cannot lose
//! them under DAZ. Thus `coeff_err=None` is retained only when every emitted
//! coefficient is DAZ-stable. This module constructs that relation directly
//! for arbitrary positive dilation, padding, and valid output-padding. Both
//! Anchored planners honor one unchanged optional node deadline from admission
//! through publication. Finite stride-1 requests use these planners too; this
//! avoids the historical equivalent-Conv2d construction's unbudgeted kernel
//! allocation while preserving that route byte-for-byte for no-deadline
//! callers. Every later spatial consumer either understands the anchored
//! origins or refuses them before arithmetic, after which the dispatcher can
//! take its own typed fallback.
//!
//! Materialized 6D and explicit-row 7D carriers compose directly on the same
//! inverse grid. Duplicate destinations are consolidated in a directed-f64
//! interval, and the published per-row `coeff_err` includes both contraction/
//! cast error and the incoming certificate transported through `|kernel|`.
//! Sparse carriers and mixed identity/materialized pairs remain typed
//! refusals. The historical no-deadline route also retains its stored-size
//! crossover. Under finite authority the cooperative Anchored composition may
//! continue past that optimization crossover when its total-live resident
//! receipt fits the configured budget, deferring materialization to the
//! caller's explicit semantic crossover.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{
    checked_shape_product,
    dd::{next_down_f64, next_up_f64},
    GemmEngine, NyError, Result,
};
use std::{mem::size_of, time::Instant};

#[cfg(test)]
use std::cell::Cell;

use super::{Conv2dLayer, ConvTranspose2dLayer};
use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds};
use crate::layers::common::PatchesPropagation;
use crate::layers::convolution::crown_helpers::{
    add_f32_bias_down_no_subnormal, add_f32_bias_up_no_subnormal, guard_nan_weights_with_poll,
};

const PLANNER_POLL_ELEMENTS: usize = 4_096;
const ZERO_FILL_CHUNK_ELEMENTS: usize = 65_536;
const F64_FRACTION_BITS: u32 = 52;
const F64_EXPONENT_BIAS: i32 = 1023;

#[cfg(test)]
thread_local! {
    static FORCED_DEADLINE_POLLS: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Test-only deterministic expiry installed on the current worker thread.
#[cfg(test)]
pub(crate) struct ConvTransposePatchesDeadlineFailpoint {
    previous: Option<usize>,
}

#[cfg(test)]
impl ConvTransposePatchesDeadlineFailpoint {
    pub(crate) fn after_successful_polls(polls: usize) -> Self {
        let previous = FORCED_DEADLINE_POLLS.with(|remaining| remaining.replace(Some(polls)));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for ConvTransposePatchesDeadlineFailpoint {
    fn drop(&mut self) {
        FORCED_DEADLINE_POLLS.with(|remaining| remaining.set(self.previous));
    }
}

#[inline]
fn check_convtranspose_patches_deadline(deadline: Option<Instant>, phase: &str) -> Result<()> {
    #[cfg(test)]
    let forced_expiry = deadline.is_some()
        && FORCED_DEADLINE_POLLS.with(|remaining| match remaining.get() {
            Some(0) => true,
            Some(polls) => {
                remaining.set(Some(polls - 1));
                false
            }
            None => false,
        });
    #[cfg(not(test))]
    let forced_expiry = false;

    if forced_expiry || deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(format!(
            "ConvTranspose2d Patches backward: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

#[inline]
fn poll_convtranspose_planner(
    work_since_poll: &mut usize,
    deadline: Option<Instant>,
    phase: &str,
) -> Result<()> {
    *work_since_poll = work_since_poll.saturating_add(1);
    if *work_since_poll >= PLANNER_POLL_ELEMENTS {
        *work_since_poll = 0;
        check_convtranspose_patches_deadline(deadline, phase)?;
    }
    Ok(())
}

fn checked_i128(value: usize, axis: &str) -> Result<i128> {
    i128::try_from(value).map_err(|_| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d anchored planner: {axis} value exceeds i128"
        ))
    })
}

/// Decode a binary32 bit pattern without presenting a subnormal source to a
/// hardware conversion instruction. This makes the proof independent of DAZ.
#[inline]
fn ct_f32_to_f64_exact_bits(value: f32) -> f64 {
    let bits = value.to_bits();
    let sign = u64::from(bits >> 31) << 63;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    match (exponent, fraction) {
        (0, 0) => f64::from_bits(sign),
        (0, _) => {
            let leading = fraction.ilog2();
            let unbiased_exponent = leading as i32 - 149;
            let exponent64 = (unbiased_exponent + F64_EXPONENT_BIAS) as u64;
            let leading_bit = 1_u32 << leading;
            let fraction64 = u64::from(fraction - leading_bit) << (F64_FRACTION_BITS - leading);
            f64::from_bits(sign | (exponent64 << F64_FRACTION_BITS) | fraction64)
        }
        (0xff, 0) => f64::from_bits(sign | (0x7ff_u64 << F64_FRACTION_BITS)),
        (0xff, _) => f64::NAN,
        _ => {
            let unbiased_exponent = exponent as i32 - 127;
            let exponent64 = (unbiased_exponent + F64_EXPONENT_BIAS) as u64;
            let fraction64 = u64::from(fraction) << (F64_FRACTION_BITS - 23);
            f64::from_bits(sign | (exponent64 << F64_FRACTION_BITS) | fraction64)
        }
    }
}

#[inline]
fn ct_nonnegative_error_or_infinity(value: f32) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let exponent = magnitude >> 23;
    if exponent == 0xff || (bits >> 31 != 0 && magnitude != 0) {
        f64::INFINITY
    } else {
        ct_f32_to_f64_exact_bits(value)
    }
}

#[inline]
fn ct_f32_is_subnormal(value: f32) -> bool {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits()
}

#[inline]
fn ct_add_f64_down(accumulator: f64, term: f64) -> f64 {
    if accumulator.is_nan() || term.is_nan() {
        return f64::NEG_INFINITY;
    }
    if term == 0.0 {
        return accumulator;
    }
    let sum = accumulator + term;
    if sum.is_nan() {
        f64::NEG_INFINITY
    } else {
        next_down_f64(sum)
    }
}

#[inline]
fn ct_add_f64_up(accumulator: f64, term: f64) -> f64 {
    if accumulator.is_nan() || term.is_nan() {
        return f64::INFINITY;
    }
    if term == 0.0 {
        return accumulator;
    }
    let sum = accumulator + term;
    if sum.is_nan() {
        f64::INFINITY
    } else {
        next_up_f64(sum)
    }
}

#[inline]
fn ct_mul_f64_up(lhs: f64, rhs: f64) -> f64 {
    if lhs == 0.0 || rhs == 0.0 {
        return 0.0;
    }
    let product = lhs * rhs;
    if product.is_nan() {
        f64::INFINITY
    } else {
        next_up_f64(product)
    }
}

#[inline]
fn ct_publish_error_up_normal(value: f64) -> f32 {
    if value.is_nan() || value < 0.0 || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }
    let min_normal = ct_f32_to_f64_exact_bits(f32::MIN_POSITIVE);
    if value <= min_normal {
        return ny_tensor::next_up_f32(f32::MIN_POSITIVE);
    }
    let nearest = value as f32;
    if !nearest.is_finite() {
        return f32::INFINITY;
    }
    let upward = if ct_f32_to_f64_exact_bits(nearest) >= value {
        nearest
    } else {
        ny_tensor::next_up_f32(nearest)
    };
    ny_tensor::next_up_f32(upward)
}

#[inline]
fn ct_publish_lower_no_subnormal(value: f64) -> f32 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == f64::INFINITY {
        return f32::INFINITY;
    }
    let min_normal = ct_f32_to_f64_exact_bits(f32::MIN_POSITIVE);
    if value != 0.0 && value.abs() < min_normal {
        return if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        };
    }
    let nearest = value as f32;
    if nearest == f32::INFINITY {
        return f32::MAX;
    }
    if nearest == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let directed = if ct_f32_to_f64_exact_bits(nearest) <= value {
        nearest
    } else {
        ny_tensor::next_down_f32(nearest)
    };
    let magnitude = directed.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
        if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        }
    } else {
        directed
    }
}

#[inline]
fn ct_publish_upper_no_subnormal(value: f64) -> f32 {
    if value.is_nan() || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let min_normal = ct_f32_to_f64_exact_bits(f32::MIN_POSITIVE);
    if value != 0.0 && value.abs() < min_normal {
        return if value.is_sign_negative() {
            0.0
        } else {
            f32::MIN_POSITIVE
        };
    }
    let nearest = value as f32;
    if nearest == f32::INFINITY {
        return f32::INFINITY;
    }
    if nearest == f32::NEG_INFINITY {
        return f32::MIN;
    }
    let directed = if ct_f32_to_f64_exact_bits(nearest) >= value {
        nearest
    } else {
        ny_tensor::next_up_f32(nearest)
    };
    let magnitude = directed.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
        if value.is_sign_negative() {
            0.0
        } else {
            f32::MIN_POSITIVE
        }
    } else {
        directed
    }
}

/// Pick a finite, non-subnormal stored coefficient and bound the distance from
/// it to a directed binary64 interval. Overflow/non-finite intervals never
/// publish an infinity coefficient: they degrade through a finite zero center
/// and `+INF` error instead.
#[inline]
fn ct_coefficient_center_and_intrinsic(lower: f64, upper: f64) -> (f32, f64) {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return (0.0, f64::INFINITY);
    }
    let nearest = lower as f32;
    let magnitude = nearest.to_bits() & 0x7fff_ffff;
    let center =
        if !nearest.is_finite() || (magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits()) {
            0.0
        } else {
            nearest
        };
    if !nearest.is_finite() {
        return (center, f64::INFINITY);
    }
    let center64 = ct_f32_to_f64_exact_bits(center);
    // Preserve an exact endpoint match. Applying `next_up_f64(0)` here would
    // invent a positive intrinsic error for an exact all-zero contraction;
    // its later normal-only publication would then inflate that fiction to a
    // binary32 normal. Nonzero distances still take one outward binary64 ULP.
    let lower_distance = (center64 - lower).abs();
    let upper_distance = (upper - center64).abs();
    let lower_gap = if lower_distance == 0.0 {
        0.0
    } else {
        next_up_f64(lower_distance)
    };
    let upper_gap = if upper_distance == 0.0 {
        0.0
    } else {
        next_up_f64(upper_distance)
    };
    (center, lower_gap.max(upper_gap))
}

/// Smallest integer not below `numerator / denominator` for a positive divisor.
fn ceil_div_euclid(numerator: i128, denominator: i128) -> Result<i128> {
    if denominator <= 0 {
        return Err(NyError::InvalidSpec(
            "ConvTranspose2d anchored planner requires a positive divisor".into(),
        ));
    }
    let quotient = numerator.div_euclid(denominator);
    if numerator.rem_euclid(denominator) == 0 {
        Ok(quotient)
    } else {
        quotient.checked_add(1).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored planner ceil division overflows i128".into(),
            )
        })
    }
}

/// Fixed patch extent for one inverse ConvTranspose axis.
///
/// The candidate input coordinates lie in an interval of width
/// `floor(dilation * (kernel - 1) / stride)`, including both endpoints.
fn anchored_axis_extent(
    kernel: usize,
    dilation: usize,
    stride: usize,
    axis: &str,
) -> Result<usize> {
    let receptive_extent = kernel
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored {axis} receptive extent overflows usize"
            ))
        })?;
    receptive_extent
        .checked_div(stride)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored {axis} patch extent overflows usize"
            ))
        })
}

/// Exact signed input coordinate contributed by one output/kernel pair.
///
/// Forward geometry is `q = x*stride + k*dilation - padding`; a backward
/// coefficient exists exactly when the inverse numerator is divisible by the
/// stride. The Euclidean operations keep negative padded coordinates exact.
fn anchored_axis_candidate(
    output: usize,
    padding: usize,
    kernel_index: usize,
    dilation: usize,
    stride: usize,
    axis: &str,
) -> Result<Option<i128>> {
    let output = checked_i128(output, axis)?;
    let padding = checked_i128(padding, axis)?;
    let kernel_index = checked_i128(kernel_index, axis)?;
    let dilation = checked_i128(dilation, axis)?;
    let stride = checked_i128(stride, axis)?;
    let padded_output = output.checked_add(padding).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d anchored {axis} padded coordinate overflows i128"
        ))
    })?;
    let dilated_tap = kernel_index.checked_mul(dilation).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d anchored {axis} dilated tap overflows i128"
        ))
    })?;
    let numerator = padded_output.checked_sub(dilated_tap).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d anchored {axis} inverse coordinate overflows i128"
        ))
    })?;
    if numerator.rem_euclid(stride) == 0 {
        Ok(Some(numerator.div_euclid(stride)))
    } else {
        Ok(None)
    }
}

/// Canonical lower endpoint of the inverse receptive interval for one output.
fn anchored_axis_origin(
    output: usize,
    padding: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    axis: &str,
) -> Result<i128> {
    let output = checked_i128(output, axis)?;
    let padding = checked_i128(padding, axis)?;
    let kernel_minus_one = checked_i128(
        kernel.checked_sub(1).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored {axis} kernel must be nonzero"
            ))
        })?,
        axis,
    )?;
    let dilation = checked_i128(dilation, axis)?;
    let stride = checked_i128(stride, axis)?;
    let receptive_extent = kernel_minus_one.checked_mul(dilation).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d anchored {axis} receptive extent overflows i128"
        ))
    })?;
    let numerator = output
        .checked_add(padding)
        .and_then(|value| value.checked_sub(receptive_extent))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored {axis} origin numerator overflows i128"
            ))
        })?;
    ceil_div_euclid(numerator, stride)
}

fn build_anchored_axis(
    output_extent: usize,
    padding: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    axis: &str,
    required_bytes: usize,
    budget_bytes: usize,
    deadline: Option<Instant>,
) -> Result<Vec<i128>> {
    let mut origins = Vec::new();
    origins
        .try_reserve_exact(output_extent)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "ConvTranspose2d anchored origin allocation",
        })?;
    let mut work_since_poll = 0usize;
    for output in 0..output_extent {
        poll_convtranspose_planner(
            &mut work_since_poll,
            deadline,
            "during anchored-origin planning",
        )?;
        origins.push(anchored_axis_origin(
            output, padding, kernel, dilation, stride, axis,
        )?);
    }
    check_convtranspose_patches_deadline(deadline, "after anchored-origin planning")?;
    Ok(origins)
}

fn allocate_zeroed_patch_values(
    elements: usize,
    required_bytes: usize,
    budget_bytes: usize,
    side: &'static str,
    deadline: Option<Instant>,
) -> Result<Vec<f32>> {
    check_convtranspose_patches_deadline(deadline, "before coefficient allocation")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: side,
        })?;
    check_convtranspose_patches_deadline(deadline, "after coefficient allocation")?;
    while values.len() < elements {
        let remaining = elements - values.len();
        let chunk = remaining.min(ZERO_FILL_CHUNK_ELEMENTS);
        values.resize(values.len() + chunk, 0.0);
        check_convtranspose_patches_deadline(deadline, "during coefficient zero-fill")?;
    }
    Ok(values)
}

/// Reconcile allocator-reported capacities plus the minimum bytes of allocations
/// still to come. `try_reserve_exact` may legally round upward, so the initial
/// requested-length preflight is necessary but not sufficient on every allocator.
fn enforce_anchored_capacity_budget(
    capacities: &[(usize, usize)],
    minimum_remaining_bytes: usize,
    budget_bytes: usize,
    site: &'static str,
) -> Result<usize> {
    let required_bytes = capacities
        .iter()
        .try_fold(0usize, |total, &(count, width)| {
            count
                .checked_mul(width)
                .and_then(|bytes| total.checked_add(bytes))
        });
    let required_bytes = required_bytes
        .and_then(|bytes| bytes.checked_add(minimum_remaining_bytes))
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored allocated capacity bytes overflow usize".into(),
            )
        })?;
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site,
        });
    }
    Ok(required_bytes)
}

fn checked_patch_offset(indices: [usize; 6], shape: [usize; 6], len: usize) -> Result<usize> {
    let mut offset = 0usize;
    for (index, extent) in indices.into_iter().zip(shape) {
        if index >= extent {
            return Err(NyError::InternalError(format!(
                "ConvTranspose2d anchored planner index {index} exceeds extent {extent}"
            )));
        }
        offset = offset
            .checked_mul(extent)
            .and_then(|value| value.checked_add(index))
            .ok_or_else(|| {
                NyError::InvalidSpec("ConvTranspose2d anchored patch offset overflows usize".into())
            })?;
    }
    if offset >= len {
        return Err(NyError::InternalError(format!(
            "ConvTranspose2d anchored planner offset {offset} exceeds allocation {len}"
        )));
    }
    Ok(offset)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AnchoredCompositionSidePlan {
    explicit_rows: bool,
    tensor_rows: usize,
    spec_c: usize,
    spec_h: usize,
    spec_w: usize,
    previous_h: usize,
    previous_w: usize,
    composed_h: usize,
    composed_w: usize,
    patch_volume: usize,
    coefficient_elements: usize,
}

impl AnchoredCompositionSidePlan {
    fn shape(self, input_channels: usize) -> Vec<usize> {
        if self.explicit_rows {
            vec![
                self.tensor_rows,
                self.spec_c,
                self.spec_h,
                self.spec_w,
                input_channels,
                self.composed_h,
                self.composed_w,
            ]
        } else {
            vec![
                self.spec_c,
                self.spec_h,
                self.spec_w,
                input_channels,
                self.composed_h,
                self.composed_w,
            ]
        }
    }

    fn positions_per_row(self) -> Result<usize> {
        checked_shape_product(&[self.spec_c, self.spec_h, self.spec_w]).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored composition position count overflows usize".into(),
            )
        })
    }
}

fn composed_anchored_axis_extent(
    previous_extent: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    axis: &str,
) -> Result<usize> {
    let previous_span = previous_extent.checked_sub(1).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d anchored composition {axis} incoming extent must be nonzero"
        ))
    })?;
    let kernel_span = kernel
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored composition {axis} kernel span overflows usize"
            ))
        })?;
    previous_span
        .checked_add(kernel_span)
        .and_then(|span| span.checked_div(stride))
        .and_then(|extent| extent.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored composition {axis} result extent overflows usize"
            ))
        })
}

fn composed_anchored_axis_origin(
    incoming_origin: i128,
    padding: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    axis: &str,
) -> Result<i128> {
    let padding = checked_i128(padding, axis)?;
    let kernel_minus_one = checked_i128(
        kernel.checked_sub(1).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored composition {axis} kernel must be nonzero"
            ))
        })?,
        axis,
    )?;
    let dilation = checked_i128(dilation, axis)?;
    let stride = checked_i128(stride, axis)?;
    let kernel_span = kernel_minus_one.checked_mul(dilation).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d anchored composition {axis} kernel span overflows i128"
        ))
    })?;
    let numerator = incoming_origin
        .checked_add(padding)
        .and_then(|value| value.checked_sub(kernel_span))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored composition {axis} origin overflows i128"
            ))
        })?;
    ceil_div_euclid(numerator, stride)
}

fn composed_anchored_axis_candidate(
    incoming_coordinate: i128,
    padding: usize,
    kernel_index: usize,
    dilation: usize,
    stride: usize,
    axis: &str,
) -> Result<Option<i128>> {
    let padding = checked_i128(padding, axis)?;
    let kernel_index = checked_i128(kernel_index, axis)?;
    let dilation = checked_i128(dilation, axis)?;
    let stride = checked_i128(stride, axis)?;
    let dilated_tap = kernel_index.checked_mul(dilation).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d anchored composition {axis} dilated tap overflows i128"
        ))
    })?;
    let numerator = incoming_coordinate
        .checked_add(padding)
        .and_then(|value| value.checked_sub(dilated_tap))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored composition {axis} inverse coordinate overflows i128"
            ))
        })?;
    if numerator.rem_euclid(stride) == 0 {
        Ok(Some(numerator.div_euclid(stride)))
    } else {
        Ok(None)
    }
}

fn allocate_zeroed_f64_values(
    elements: usize,
    required_bytes: usize,
    budget_bytes: usize,
    side: &'static str,
    deadline: Option<Instant>,
) -> Result<Vec<f64>> {
    check_convtranspose_patches_deadline(deadline, "before directed scratch allocation")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: side,
        })?;
    check_convtranspose_patches_deadline(deadline, "after directed scratch allocation")?;
    while values.len() < elements {
        let remaining = elements - values.len();
        values.resize(values.len() + remaining.min(ZERO_FILL_CHUNK_ELEMENTS), 0.0);
        check_convtranspose_patches_deadline(deadline, "during directed scratch zero-fill")?;
    }
    check_convtranspose_patches_deadline(deadline, "after directed scratch zero-fill")?;
    Ok(values)
}

fn zero_directed_scratch(values: &mut [f64], deadline: Option<Instant>) -> Result<()> {
    for chunk in values.chunks_mut(ZERO_FILL_CHUNK_ELEMENTS) {
        chunk.fill(0.0);
        check_convtranspose_patches_deadline(deadline, "during directed scratch reset")?;
    }
    Ok(())
}

fn checked_composition_bytes(elements: usize, width: usize, name: &str) -> Result<usize> {
    elements.checked_mul(width).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d anchored composition {name} bytes overflow usize"
        ))
    })
}

fn checked_composition_byte_sum(parts: &[usize], name: &str) -> Result<usize> {
    parts.iter().try_fold(0usize, |total, &part| {
        total.checked_add(part).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored composition {name} bytes overflow usize"
            ))
        })
    })
}

impl PatchesPropagation for ConvTranspose2dLayer {
    /// CROWN backward with Patches coefficients. Delegates to the engine-aware
    /// variant with no engine.
    ///
    /// No-deadline stride 1 supports identity and non-identity composition
    /// through the equivalent Conv2d. Finite stride 1 and every stride>1
    /// request support a full virtual identity and materialized nonsparse
    /// 6D/7D carriers through exact anchored geometry; closed corners return a
    /// typed refusal so the caller takes dense CROWN.
    fn propagate_patches(&self, bounds: &PatchesLinearBounds) -> Result<CrownBounds> {
        self.propagate_patches_engine(bounds, None)
    }
}

impl ConvTranspose2dLayer {
    /// Build the flipped-and-channel-swapped Conv2d kernel `Kc` whose CROWN
    /// backward equals this ConvTranspose2d's CROWN backward (stride 1):
    /// `Kc[oc, ic, ki, kj] = W[ic, oc, kh-1-ki, kw-1-kj]`.
    ///
    /// `W` is the ONNX ConvTranspose layout `(in_c, out_c, kh, kw)`; `Kc` is the
    /// Conv2d layout `(out_c, in_c, kh, kw)`. Pure data movement (no bound math).
    fn crown_backward_equivalent_conv2d_kernel_with_deadline(
        &self,
        deadline: Option<Instant>,
    ) -> Result<ArrayD<f32>> {
        check_convtranspose_patches_deadline(deadline, "before equivalent-kernel construction")?;
        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let (kh, kw) = self.kernel_size();
        let mut kc = ArrayD::<f32>::zeros(IxDyn(&[out_c, in_c, kh, kw]));
        for oc in 0..out_c {
            check_convtranspose_patches_deadline(
                deadline,
                "during equivalent-kernel construction",
            )?;
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        kc[[oc, ic, ki, kj]] = self.kernel[[ic, oc, kh - 1 - ki, kw - 1 - kj]];
                    }
                }
            }
        }
        check_convtranspose_patches_deadline(deadline, "after equivalent-kernel construction")?;
        Ok(kc)
    }

    fn validate_anchored_composition_geometry_with_deadline(
        data: &PatchesData,
        kernel: (usize, usize),
        deadline: Option<Instant>,
    ) -> Result<()> {
        let (spec_c, spec_h, spec_w) = data.output_shape;
        let (_, input_h, input_w) = data.input_shape;
        if spec_c == 0 || spec_h == 0 || spec_w == 0 || kernel.0 == 0 || kernel.1 == 0 {
            return Err(NyError::InvalidSpec(
                "ConvTranspose2d anchored composition geometry requires nonzero dimensions".into(),
            ));
        }
        match &data.geometry {
            PatchGeometry::Affine(_) => {
                let affine = data
                    .geometry
                    .require_affine("ConvTranspose2d anchored composition geometry")?;
                let actual = affine.output_size((input_h, input_w), kernel)?;
                if actual != (spec_h, spec_w) {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![spec_h, spec_w],
                        got: vec![actual.0, actual.1],
                    });
                }
            }
            PatchGeometry::Anchored(_) => {
                let last_h = checked_i128(kernel.0 - 1, "incoming anchored kernel row")?;
                let last_w = checked_i128(kernel.1 - 1, "incoming anchored kernel column")?;
                let mut work_since_poll = 0usize;
                for row in 0..spec_h {
                    let origin = data.geometry.origin((row, 0))?.0;
                    origin.checked_add(last_h).ok_or_else(|| {
                        NyError::InvalidSpec(
                            "ConvTranspose2d incoming anchored row plus kernel overflows i128"
                                .into(),
                        )
                    })?;
                    poll_convtranspose_planner(
                        &mut work_since_poll,
                        deadline,
                        "during incoming anchored-row validation",
                    )?;
                }
                for column in 0..spec_w {
                    let origin = data.geometry.origin((0, column))?.1;
                    origin.checked_add(last_w).ok_or_else(|| {
                        NyError::InvalidSpec(
                            "ConvTranspose2d incoming anchored column plus kernel overflows i128"
                                .into(),
                        )
                    })?;
                    poll_convtranspose_planner(
                        &mut work_since_poll,
                        deadline,
                        "during incoming anchored-column validation",
                    )?;
                }
                // The scans above prove that both axes are at least as long as
                // the declared output. Probe the first out-of-range coordinate
                // too, so excess typed metadata cannot be silently ignored.
                // For Anchored geometry `origin` is a checked vector lookup;
                // an exact axis must refuse each boundary index.
                if data.geometry.origin((spec_h, 0)).is_ok()
                    || data.geometry.origin((0, spec_w)).is_ok()
                {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![spec_h, spec_w],
                        got: vec![spec_h.saturating_add(1), spec_w.saturating_add(1)],
                    });
                }
                check_convtranspose_patches_deadline(
                    deadline,
                    "after incoming anchored-axis extent validation",
                )?;
            }
        }
        check_convtranspose_patches_deadline(deadline, "after incoming geometry validation")
    }

    fn validate_anchored_composition_common_geometry_with_deadline(
        lower: &PatchesData,
        upper: &PatchesData,
        kernel: (usize, usize),
        deadline: Option<Instant>,
    ) -> Result<()> {
        if lower.identity != upper.identity
            || lower.input_shape != upper.input_shape
            || lower.output_shape != upper.output_shape
        {
            return Err(NyError::InvalidSpec(
                "ConvTranspose2d anchored composition lower/upper geometry metadata differs".into(),
            ));
        }
        Self::validate_anchored_composition_geometry_with_deadline(lower, kernel, deadline)?;
        Self::validate_anchored_composition_geometry_with_deadline(upper, kernel, deadline)?;
        match (&lower.geometry, &upper.geometry) {
            (PatchGeometry::Affine(_), PatchGeometry::Affine(_)) => {
                let lower_affine = lower
                    .geometry
                    .require_affine("ConvTranspose2d lower common geometry")?;
                let upper_affine = upper
                    .geometry
                    .require_affine("ConvTranspose2d upper common geometry")?;
                if lower_affine != upper_affine {
                    return Err(NyError::InvalidSpec(
                        "ConvTranspose2d anchored composition affine descriptors differ".into(),
                    ));
                }
                return check_convtranspose_patches_deadline(
                    deadline,
                    "after affine common-geometry comparison",
                );
            }
            (PatchGeometry::Anchored(_), PatchGeometry::Anchored(_)) => {}
            _ => {
                return Err(NyError::InvalidSpec(
                    "ConvTranspose2d anchored composition lower/upper geometry variants differ"
                        .into(),
                ));
            }
        }
        let (_, spec_h, spec_w) = lower.output_shape;
        let mut work_since_poll = 0usize;
        for row in 0..spec_h {
            if lower.geometry.origin((row, 0))?.0 != upper.geometry.origin((row, 0))?.0 {
                return Err(NyError::InvalidSpec(format!(
                    "ConvTranspose2d anchored composition lower/upper row origin differs at {row}"
                )));
            }
            poll_convtranspose_planner(
                &mut work_since_poll,
                deadline,
                "during lower/upper row-origin comparison",
            )?;
        }
        for column in 0..spec_w {
            if lower.geometry.origin((0, column))?.1 != upper.geometry.origin((0, column))?.1 {
                return Err(NyError::InvalidSpec(format!(
                    "ConvTranspose2d anchored composition lower/upper column origin differs at {column}"
                )));
            }
            poll_convtranspose_planner(
                &mut work_since_poll,
                deadline,
                "during lower/upper column-origin comparison",
            )?;
        }
        check_convtranspose_patches_deadline(deadline, "after common-geometry comparison")
    }

    fn anchored_composition_side_plan(
        &self,
        data: &PatchesData,
        row_count: usize,
        conv_output_shape: (usize, usize, usize),
        conv_input_shape: (usize, usize, usize),
        deadline: Option<Instant>,
    ) -> Result<AnchoredCompositionSidePlan> {
        if data.identity {
            return Err(NyError::UnsupportedConfiguration(
                "ConvTranspose2d anchored composition requires two materialized sides; mixed identity/materialized bounds use dense CROWN"
                    .into(),
            ));
        }
        if data.unstable_idx.is_some() {
            return Err(NyError::UnsupportedConfiguration(
                "ConvTranspose2d anchored composition does not support sparse 4D/5D carriers; use dense CROWN"
                    .into(),
            ));
        }
        let incoming = data.patches.as_ref().ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored composition materialized side has no coefficient tensor"
                    .into(),
            )
        })?;
        let shape = incoming.shape();
        let explicit_rows = match shape.len() {
            6 => false,
            7 => true,
            dimension => {
                return Err(NyError::ShapeMismatch {
                    expected: vec![6, 7],
                    got: vec![dimension],
                });
            }
        };
        if shape.contains(&0) {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose2d anchored composition dimensions must be nonzero, got {shape:?}"
            )));
        }

        let (spec_c, spec_h, spec_w) = data.output_shape;
        let (tensor_rows, prefix_start, channel_axis, height_axis, width_axis) = if explicit_rows {
            (shape[0], 1usize, 4usize, 5usize, 6usize)
        } else {
            (1usize, 0usize, 3usize, 4usize, 5usize)
        };
        if shape[prefix_start..prefix_start + 3] != [spec_c, spec_h, spec_w]
            || shape[channel_axis] != conv_output_shape.0
        {
            let expected = if explicit_rows {
                vec![
                    row_count,
                    spec_c,
                    spec_h,
                    spec_w,
                    conv_output_shape.0,
                    shape[height_axis],
                    shape[width_axis],
                ]
            } else {
                vec![
                    spec_c,
                    spec_h,
                    spec_w,
                    conv_output_shape.0,
                    shape[height_axis],
                    shape[width_axis],
                ]
            };
            return Err(NyError::ShapeMismatch {
                expected,
                got: shape.to_vec(),
            });
        }
        if data.input_shape != conv_output_shape {
            return Err(NyError::ShapeMismatch {
                expected: vec![
                    conv_output_shape.0,
                    conv_output_shape.1,
                    conv_output_shape.2,
                ],
                got: vec![data.input_shape.0, data.input_shape.1, data.input_shape.2],
            });
        }
        let positions = checked_shape_product(&[spec_c, spec_h, spec_w]).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored composition logical position count overflows usize"
                    .into(),
            )
        })?;
        let expected_rows = if explicit_rows {
            tensor_rows
        } else {
            positions
        };
        if row_count != expected_rows {
            return Err(NyError::ShapeMismatch {
                expected: vec![expected_rows],
                got: vec![row_count],
            });
        }
        if let Some(error) = data.coeff_err.as_ref() {
            if error.len() != row_count {
                return Err(NyError::ShapeMismatch {
                    expected: vec![row_count],
                    got: vec![error.len()],
                });
            }
        }

        let previous_h = shape[height_axis];
        let previous_w = shape[width_axis];
        Self::validate_anchored_composition_geometry_with_deadline(
            data,
            (previous_h, previous_w),
            deadline,
        )?;
        let (kernel_h, kernel_w) = self.kernel_size();
        let composed_h = composed_anchored_axis_extent(
            previous_h,
            kernel_h,
            self.dilation.0,
            self.stride.0,
            "row",
        )?;
        let composed_w = composed_anchored_axis_extent(
            previous_w,
            kernel_w,
            self.dilation.1,
            self.stride.1,
            "column",
        )?;
        let patch_volume = checked_shape_product(&[conv_input_shape.0, composed_h, composed_w])
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "ConvTranspose2d anchored composition patch volume overflows usize".into(),
                )
            })?;
        let coefficient_elements = if explicit_rows {
            checked_shape_product(&[
                tensor_rows,
                spec_c,
                spec_h,
                spec_w,
                conv_input_shape.0,
                composed_h,
                composed_w,
            ])
        } else {
            checked_shape_product(&[
                spec_c,
                spec_h,
                spec_w,
                conv_input_shape.0,
                composed_h,
                composed_w,
            ])
        }
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored composition result shape overflows usize".into(),
            )
        })?;

        // Preserve the historical no-deadline optimization decision exactly:
        // compare the actual stored tensor against the equivalent Dense matrix.
        // In 7D the spec-position slab is repeated for every explicit row, so
        // the ordinary patch-area-only crossover would miss that factor.
        //
        // A finite request cannot safely take an unbounded same-relation Dense
        // retry at this seam. Its Anchored implementation is already
        // cooperative and admits the truthful total-live resident peak below,
        // so let that authority decide whether the transaction may proceed and
        // defer Dense conversion to the caller's explicit semantic crossover.
        if deadline.is_none() {
            let dense_elements = checked_shape_product(&[
                row_count,
                conv_input_shape.0,
                conv_input_shape.1,
                conv_input_shape.2,
            ])
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "ConvTranspose2d anchored composition dense crossover overflows usize".into(),
                )
            })?;
            if coefficient_elements >= dense_elements {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "ConvTranspose2d anchored composition stores {coefficient_elements} coefficients, reaching dense size {dense_elements}; use dense CROWN"
                )));
            }
        }

        Ok(AnchoredCompositionSidePlan {
            explicit_rows,
            tensor_rows,
            spec_c,
            spec_h,
            spec_w,
            previous_h,
            previous_w,
            composed_h,
            composed_w,
            patch_volume,
            coefficient_elements,
        })
    }

    fn build_composed_anchored_axis(
        &self,
        incoming_geometry: &PatchGeometry,
        output_extent: usize,
        row_axis: bool,
        required_bytes: usize,
        budget_bytes: usize,
        deadline: Option<Instant>,
    ) -> Result<Vec<i128>> {
        let axis = if row_axis { "row" } else { "column" };
        let (padding, kernel, dilation, stride) = if row_axis {
            (
                self.padding.0,
                self.kernel_size().0,
                self.dilation.0,
                self.stride.0,
            )
        } else {
            (
                self.padding.1,
                self.kernel_size().1,
                self.dilation.1,
                self.stride.1,
            )
        };
        check_convtranspose_patches_deadline(deadline, "before composed anchored-axis allocation")?;
        let mut origins = Vec::new();
        origins
            .try_reserve_exact(output_extent)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "ConvTranspose2d composed anchored origin allocation",
            })?;
        check_convtranspose_patches_deadline(deadline, "after composed anchored-axis allocation")?;
        let mut work_since_poll = 0usize;
        for output in 0..output_extent {
            poll_convtranspose_planner(
                &mut work_since_poll,
                deadline,
                "during composed anchored-axis construction",
            )?;
            let incoming_origin = if row_axis {
                incoming_geometry.origin((output, 0))?.0
            } else {
                incoming_geometry.origin((0, output))?.1
            };
            origins.push(composed_anchored_axis_origin(
                incoming_origin,
                padding,
                kernel,
                dilation,
                stride,
                axis,
            )?);
        }
        check_convtranspose_patches_deadline(
            deadline,
            "after composed anchored-axis construction",
        )?;
        Ok(origins)
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_anchored_composition_side(
        &self,
        data: &PatchesData,
        plan: AnchoredCompositionSidePlan,
        (in_h, in_w): (usize, usize),
        row_origins: &[i128],
        column_origins: &[i128],
        kernel_gain: f64,
        coefficient_values: &mut [f32],
        coefficient_errors: &mut [f32],
        scratch_lower: &mut [f64],
        scratch_upper: &mut [f64],
        deadline: Option<Instant>,
    ) -> Result<()> {
        let incoming = data.patches.as_ref().ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored composition side lost its coefficient tensor".into(),
            )
        })?;
        let positions_per_row = plan.positions_per_row()?;
        let expected_error_count = if plan.explicit_rows {
            plan.tensor_rows
        } else {
            positions_per_row
        };
        if coefficient_values.len() != plan.coefficient_elements
            || coefficient_errors.len() != expected_error_count
            || scratch_lower.len() < plan.patch_volume
            || scratch_upper.len() < plan.patch_volume
            || row_origins.len() != plan.spec_h
            || column_origins.len() != plan.spec_w
        {
            return Err(NyError::InternalError(
                "ConvTranspose2d anchored composition allocation/plan mismatch".into(),
            ));
        }
        let (_, out_h, out_w) = data.input_shape;
        let in_c = self.in_channels();
        let out_h_i128 = checked_i128(out_h, "output row extent")?;
        let out_w_i128 = checked_i128(out_w, "output column extent")?;
        let in_h_i128 = checked_i128(in_h, "input row extent")?;
        let in_w_i128 = checked_i128(in_w, "input column extent")?;
        let (kernel_h, kernel_w) = self.kernel_size();
        let mut work_since_poll = 0usize;

        for tensor_row in 0..plan.tensor_rows {
            for spec_channel in 0..plan.spec_c {
                for spec_row in 0..plan.spec_h {
                    let old_origin_h = data.geometry.origin((spec_row, 0))?.0;
                    let new_origin_h = row_origins[spec_row];
                    for spec_column in 0..plan.spec_w {
                        let old_origin_w = data.geometry.origin((0, spec_column))?.1;
                        let new_origin_w = column_origins[spec_column];
                        let lower = &mut scratch_lower[..plan.patch_volume];
                        let upper = &mut scratch_upper[..plan.patch_volume];
                        zero_directed_scratch(lower, deadline)?;
                        zero_directed_scratch(upper, deadline)?;

                        for output_channel in 0..data.input_shape.0 {
                            for incoming_tap_h in 0..plan.previous_h {
                                poll_convtranspose_planner(
                                    &mut work_since_poll,
                                    deadline,
                                    "during anchored composition row traversal",
                                )?;
                                let qh = old_origin_h
                                    .checked_add(checked_i128(incoming_tap_h, "incoming row tap")?)
                                    .ok_or_else(|| {
                                        NyError::InvalidSpec(
                                            "ConvTranspose2d anchored composition incoming row overflows i128"
                                                .into(),
                                        )
                                    })?;
                                if qh < 0 || qh >= out_h_i128 {
                                    continue;
                                }
                                for kernel_h_index in 0..kernel_h {
                                    poll_convtranspose_planner(
                                        &mut work_since_poll,
                                        deadline,
                                        "during anchored composition row-kernel traversal",
                                    )?;
                                    let Some(input_h) = composed_anchored_axis_candidate(
                                        qh,
                                        self.padding.0,
                                        kernel_h_index,
                                        self.dilation.0,
                                        self.stride.0,
                                        "row",
                                    )?
                                    else {
                                        continue;
                                    };
                                    if input_h < 0 || input_h >= in_h_i128 {
                                        continue;
                                    }
                                    let destination_h = input_h
                                        .checked_sub(new_origin_h)
                                        .and_then(|value| usize::try_from(value).ok())
                                        .filter(|&value| value < plan.composed_h)
                                        .ok_or_else(|| {
                                            NyError::InternalError(format!(
                                                "ConvTranspose2d composed row tap outside plan: input={input_h}, origin={new_origin_h}, extent={}",
                                                plan.composed_h
                                            ))
                                        })?;

                                    for incoming_tap_w in 0..plan.previous_w {
                                        poll_convtranspose_planner(
                                            &mut work_since_poll,
                                            deadline,
                                            "during anchored composition column traversal",
                                        )?;
                                        let qw = old_origin_w
                                            .checked_add(checked_i128(
                                                incoming_tap_w,
                                                "incoming column tap",
                                            )?)
                                            .ok_or_else(|| {
                                                NyError::InvalidSpec(
                                                    "ConvTranspose2d anchored composition incoming column overflows i128"
                                                        .into(),
                                                )
                                            })?;
                                        if qw < 0 || qw >= out_w_i128 {
                                            continue;
                                        }
                                        let incoming_value = if plan.explicit_rows {
                                            incoming[[
                                                tensor_row,
                                                spec_channel,
                                                spec_row,
                                                spec_column,
                                                output_channel,
                                                incoming_tap_h,
                                                incoming_tap_w,
                                            ]]
                                        } else {
                                            incoming[[
                                                spec_channel,
                                                spec_row,
                                                spec_column,
                                                output_channel,
                                                incoming_tap_h,
                                                incoming_tap_w,
                                            ]]
                                        };
                                        let incoming_value =
                                            ct_f32_to_f64_exact_bits(incoming_value);

                                        for kernel_w_index in 0..kernel_w {
                                            poll_convtranspose_planner(
                                                &mut work_since_poll,
                                                deadline,
                                                "during anchored composition column-kernel traversal",
                                            )?;
                                            let Some(input_w) = composed_anchored_axis_candidate(
                                                qw,
                                                self.padding.1,
                                                kernel_w_index,
                                                self.dilation.1,
                                                self.stride.1,
                                                "column",
                                            )?
                                            else {
                                                continue;
                                            };
                                            if input_w < 0 || input_w >= in_w_i128 {
                                                continue;
                                            }
                                            let destination_w = input_w
                                                .checked_sub(new_origin_w)
                                                .and_then(|value| usize::try_from(value).ok())
                                                .filter(|&value| value < plan.composed_w)
                                                .ok_or_else(|| {
                                                    NyError::InternalError(format!(
                                                        "ConvTranspose2d composed column tap outside plan: input={input_w}, origin={new_origin_w}, extent={}",
                                                        plan.composed_w
                                                    ))
                                                })?;

                                            for input_channel in 0..in_c {
                                                poll_convtranspose_planner(
                                                    &mut work_since_poll,
                                                    deadline,
                                                    "during anchored composition contraction",
                                                )?;
                                                // A product of two binary32 values is
                                                // exact in binary64 (at most 48
                                                // significand bits); only the directed
                                                // additions below need an interval.
                                                let weight = ct_f32_to_f64_exact_bits(
                                                    self.kernel[[
                                                        input_channel,
                                                        output_channel,
                                                        kernel_h_index,
                                                        kernel_w_index,
                                                    ]],
                                                );
                                                let term = incoming_value * weight;
                                                let destination = (input_channel * plan.composed_h
                                                    + destination_h)
                                                    * plan.composed_w
                                                    + destination_w;
                                                lower[destination] =
                                                    ct_add_f64_down(lower[destination], term);
                                                upper[destination] =
                                                    ct_add_f64_up(upper[destination], term);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let position =
                            (spec_channel * plan.spec_h + spec_row) * plan.spec_w + spec_column;
                        let flat_position = tensor_row
                            .checked_mul(positions_per_row)
                            .and_then(|value| value.checked_add(position))
                            .ok_or_else(|| {
                                NyError::InvalidSpec(
                                    "ConvTranspose2d anchored composition position offset overflows usize"
                                        .into(),
                                )
                            })?;
                        let coefficient_offset = flat_position
                            .checked_mul(plan.patch_volume)
                            .ok_or_else(|| {
                                NyError::InvalidSpec(
                                    "ConvTranspose2d anchored composition coefficient offset overflows usize"
                                        .into(),
                                )
                            })?;
                        let error_index = if plan.explicit_rows {
                            tensor_row
                        } else {
                            position
                        };
                        let old_error = data.coeff_err.as_ref().map_or(0.0, |error| {
                            ct_nonnegative_error_or_infinity(error[error_index])
                        });
                        // Exact-zero first: an all-zero kernel maps even an
                        // INF-poisoned incoming row to exact zero, never 0*INF.
                        let carry = if old_error == 0.0 || kernel_gain == 0.0 {
                            0.0
                        } else {
                            ct_mul_f64_up(old_error, kernel_gain)
                        };
                        for destination in 0..plan.patch_volume {
                            poll_convtranspose_planner(
                                &mut work_since_poll,
                                deadline,
                                "during anchored composition coefficient publication",
                            )?;
                            let (center, intrinsic) = ct_coefficient_center_and_intrinsic(
                                lower[destination],
                                upper[destination],
                            );
                            coefficient_values[coefficient_offset + destination] = center;
                            let total_error = ct_add_f64_up(intrinsic, carry);
                            let published = ct_publish_error_up_normal(total_error);
                            if published > coefficient_errors[error_index] {
                                coefficient_errors[error_index] = published;
                            }
                        }
                    }
                }
            }
        }
        check_convtranspose_patches_deadline(deadline, "after anchored composition contraction")
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_anchored_composition_bias_side(
        &self,
        data: &PatchesData,
        plan: AnchoredCompositionSidePlan,
        old_bias: &Array1<f32>,
        layer_bias: Option<&Array1<f32>>,
        lower_endpoint: bool,
        output: &mut Vec<f32>,
        deadline: Option<Instant>,
    ) -> Result<()> {
        if output.capacity() < old_bias.len() {
            return Err(NyError::InternalError(format!(
                "ConvTranspose2d anchored composition bias capacity too small: expected at least {}, got {}",
                old_bias.len(),
                output.capacity()
            )));
        }
        let Some(layer_bias) = layer_bias else {
            let mut work_since_poll = 0usize;
            for &value in old_bias.iter() {
                output.push(value);
                poll_convtranspose_planner(
                    &mut work_since_poll,
                    deadline,
                    "during anchored composition bias copy",
                )?;
            }
            return check_convtranspose_patches_deadline(
                deadline,
                "after anchored composition bias copy",
            );
        };
        let incoming = data.patches.as_ref().ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored bias composition lost its coefficient tensor".into(),
            )
        })?;
        let positions_per_row = plan.positions_per_row()?;
        let (_, out_h, out_w) = data.input_shape;
        let out_h_i128 = checked_i128(out_h, "bias output row extent")?;
        let out_w_i128 = checked_i128(out_w, "bias output column extent")?;
        let mut work_since_poll = 0usize;

        for logical_row in 0..old_bias.len() {
            let mut sum = ct_f32_to_f64_exact_bits(old_bias[logical_row]);
            let mut bias_factor = 0.0f64;
            let (first_position, last_position) = if plan.explicit_rows {
                (0usize, positions_per_row)
            } else {
                (logical_row, logical_row + 1)
            };
            let tensor_row = if plan.explicit_rows { logical_row } else { 0 };
            for position in first_position..last_position {
                let spec_channel = position / (plan.spec_h * plan.spec_w);
                let spatial = position % (plan.spec_h * plan.spec_w);
                let spec_row = spatial / plan.spec_w;
                let spec_column = spatial % plan.spec_w;
                let old_origin = data.geometry.origin((spec_row, spec_column))?;

                for incoming_tap_h in 0..plan.previous_h {
                    poll_convtranspose_planner(
                        &mut work_since_poll,
                        deadline,
                        "during anchored bias row traversal",
                    )?;
                    let qh = old_origin
                        .0
                        .checked_add(checked_i128(incoming_tap_h, "bias incoming row tap")?)
                        .ok_or_else(|| {
                            NyError::InvalidSpec(
                                "ConvTranspose2d anchored bias row coordinate overflows i128"
                                    .into(),
                            )
                        })?;
                    if qh < 0 || qh >= out_h_i128 {
                        continue;
                    }
                    for incoming_tap_w in 0..plan.previous_w {
                        poll_convtranspose_planner(
                            &mut work_since_poll,
                            deadline,
                            "during anchored bias column traversal",
                        )?;
                        let qw = old_origin
                            .1
                            .checked_add(checked_i128(incoming_tap_w, "bias incoming column tap")?)
                            .ok_or_else(|| {
                                NyError::InvalidSpec(
                                    "ConvTranspose2d anchored bias column coordinate overflows i128"
                                        .into(),
                                )
                            })?;
                        if qw < 0 || qw >= out_w_i128 {
                            continue;
                        }
                        for output_channel in 0..data.input_shape.0 {
                            poll_convtranspose_planner(
                                &mut work_since_poll,
                                deadline,
                                "during anchored composition bias fold",
                            )?;
                            let coefficient = if plan.explicit_rows {
                                incoming[[
                                    tensor_row,
                                    spec_channel,
                                    spec_row,
                                    spec_column,
                                    output_channel,
                                    incoming_tap_h,
                                    incoming_tap_w,
                                ]]
                            } else {
                                incoming[[
                                    spec_channel,
                                    spec_row,
                                    spec_column,
                                    output_channel,
                                    incoming_tap_h,
                                    incoming_tap_w,
                                ]]
                            };
                            let bias = ct_f32_to_f64_exact_bits(layer_bias[output_channel]);
                            let term = ct_f32_to_f64_exact_bits(coefficient) * bias;
                            sum = if lower_endpoint {
                                ct_add_f64_down(sum, term)
                            } else {
                                ct_add_f64_up(sum, term)
                            };
                            // Count every valid repeated factor. In 7D this runs
                            // over the entire position slab of the explicit row;
                            // in 6D it covers that logical position's full patch.
                            bias_factor = ct_add_f64_up(bias_factor, bias.abs());
                        }
                    }
                }
            }

            let old_error = data.coeff_err.as_ref().map_or(0.0, |error| {
                ct_nonnegative_error_or_infinity(error[logical_row])
            });
            let penalty = if old_error == 0.0 || bias_factor == 0.0 {
                0.0
            } else {
                ct_mul_f64_up(old_error, bias_factor)
            };
            let widened = if lower_endpoint {
                ct_add_f64_down(sum, -penalty)
            } else {
                ct_add_f64_up(sum, penalty)
            };
            output.push(if lower_endpoint {
                ct_publish_lower_no_subnormal(widened)
            } else {
                ct_publish_upper_no_subnormal(widened)
            });
        }
        check_convtranspose_patches_deadline(deadline, "after anchored composition bias fold")
    }

    fn validate_anchored_composition_finite_sources(
        &self,
        bounds: &PatchesLinearBounds,
        deadline: Option<Instant>,
    ) -> Result<()> {
        let mut work_since_poll = 0usize;
        for value in self.kernel.iter().chain(
            self.bias
                .as_ref()
                .into_iter()
                .flat_map(|values| values.iter()),
        ) {
            poll_convtranspose_planner(
                &mut work_since_poll,
                deadline,
                "during anchored composition layer-source scan",
            )?;
            if !value.is_finite() {
                return Err(NyError::NumericalInstability(
                    "ConvTranspose2d anchored composition requires finite kernel/bias sources"
                        .into(),
                ));
            }
        }
        for (name, data, bias) in [
            ("lower", &bounds.lower_a, &bounds.lower_b),
            ("upper", &bounds.upper_a, &bounds.upper_b),
        ] {
            let coefficients = data.patches.as_ref().ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "ConvTranspose2d anchored composition {name} side has no coefficients"
                ))
            })?;
            for value in coefficients.iter().chain(bias.iter()) {
                poll_convtranspose_planner(
                    &mut work_since_poll,
                    deadline,
                    "during anchored composition bound-source scan",
                )?;
                if !value.is_finite() {
                    return Err(NyError::NumericalInstability(format!(
                        "ConvTranspose2d anchored composition {name} coefficient/bias contains a non-finite source"
                    )));
                }
            }
        }
        check_convtranspose_patches_deadline(deadline, "after anchored composition source scan")
    }

    fn validate_anchored_layer_finite_sources(&self, deadline: Option<Instant>) -> Result<()> {
        let mut work_since_poll = 0usize;
        for value in self.kernel.iter().chain(
            self.bias
                .as_ref()
                .into_iter()
                .flat_map(|values| values.iter()),
        ) {
            poll_convtranspose_planner(
                &mut work_since_poll,
                deadline,
                "during anchored layer-source scan",
            )?;
            if !value.is_finite() {
                return Err(NyError::NumericalInstability(
                    "ConvTranspose2d anchored Patches requires finite kernel/bias sources".into(),
                ));
            }
        }
        check_convtranspose_patches_deadline(deadline, "after anchored layer-source scan")
    }

    fn anchored_composition_kernel_gain(&self, deadline: Option<Instant>) -> Result<f64> {
        let mut worst = 0.0f64;
        let mut work_since_poll = 0usize;
        for input_channel in 0..self.in_channels() {
            let mut channel_sum = 0.0f64;
            for output_channel in 0..self.out_channels() {
                for kernel_h in 0..self.kernel_size().0 {
                    for kernel_w in 0..self.kernel_size().1 {
                        poll_convtranspose_planner(
                            &mut work_since_poll,
                            deadline,
                            "during anchored composition kernel-gain scan",
                        )?;
                        channel_sum = ct_add_f64_up(
                            channel_sum,
                            ct_f32_to_f64_exact_bits(
                                self.kernel[[input_channel, output_channel, kernel_h, kernel_w]],
                            )
                            .abs(),
                        );
                    }
                }
            }
            worst = worst.max(channel_sum);
        }
        check_convtranspose_patches_deadline(
            deadline,
            "after anchored composition kernel-gain scan",
        )?;
        Ok(worst)
    }

    /// Compose a materialized 6D or explicit-row 7D carrier through a
    /// positive-stride ConvTranspose directly on the exact inverse grid.
    #[cfg(test)]
    pub(crate) fn propagate_anchored_composition_with_deadline(
        &self,
        bounds: &PatchesLinearBounds,
        deadline: Option<Instant>,
    ) -> Result<CrownBounds> {
        let input_shape = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose2d anchored composition requires input_shape".into(),
            )
        })?;
        self.propagate_anchored_composition_with_deadline_for_input_shape(
            bounds,
            deadline,
            None,
            input_shape,
        )
    }

    #[cfg(test)]
    pub(crate) fn propagate_anchored_composition_with_deadline_and_budget_for_test(
        &self,
        bounds: &PatchesLinearBounds,
        deadline: Option<Instant>,
        budget_bytes: usize,
    ) -> Result<CrownBounds> {
        let input_shape = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose2d anchored composition requires input_shape".into(),
            )
        })?;
        self.propagate_anchored_composition_with_deadline_for_input_shape(
            bounds,
            deadline,
            Some(budget_bytes),
            input_shape,
        )
    }

    fn propagate_anchored_composition_with_deadline_for_input_shape(
        &self,
        bounds: &PatchesLinearBounds,
        deadline: Option<Instant>,
        budget_override: Option<usize>,
        (in_h, in_w): (usize, usize),
    ) -> Result<CrownBounds> {
        check_convtranspose_patches_deadline(deadline, "before anchored composition")?;
        let (in_c, out_c) = self.validate_geometry()?;
        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let conv_output_shape = (out_c, out_h, out_w);
        let conv_input_shape = (in_c, in_h, in_w);

        if bounds.lower_b.len() != bounds.row_count || bounds.upper_b.len() != bounds.row_count {
            return Err(NyError::ShapeMismatch {
                expected: vec![bounds.row_count, bounds.row_count],
                got: vec![bounds.lower_b.len(), bounds.upper_b.len()],
            });
        }
        let lower_plan = self.anchored_composition_side_plan(
            &bounds.lower_a,
            bounds.row_count,
            conv_output_shape,
            conv_input_shape,
            deadline,
        )?;
        let upper_plan = self.anchored_composition_side_plan(
            &bounds.upper_a,
            bounds.row_count,
            conv_output_shape,
            conv_input_shape,
            deadline,
        )?;
        // A shared output geometry cannot authenticate unequal raw tap extents.
        // Reject every plan mismatch before the first allocation.
        if lower_plan != upper_plan {
            return Err(NyError::ShapeMismatch {
                expected: lower_plan.shape(in_c),
                got: upper_plan.shape(in_c),
            });
        }
        Self::validate_anchored_composition_common_geometry_with_deadline(
            &bounds.lower_a,
            &bounds.upper_a,
            (lower_plan.previous_h, lower_plan.previous_w),
            deadline,
        )?;
        self.validate_anchored_composition_finite_sources(bounds, deadline)?;

        let axis_elements = lower_plan
            .spec_h
            .checked_add(lower_plan.spec_w)
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "ConvTranspose2d anchored composition axis count overflows usize".into(),
                )
            })?;
        let axis_bytes = checked_composition_bytes(axis_elements, size_of::<i128>(), "axis")?;
        let lower_coefficient_bytes = checked_composition_bytes(
            lower_plan.coefficient_elements,
            size_of::<f32>(),
            "lower coefficient",
        )?;
        let upper_coefficient_bytes = checked_composition_bytes(
            upper_plan.coefficient_elements,
            size_of::<f32>(),
            "upper coefficient",
        )?;
        let one_error_bytes =
            checked_composition_bytes(bounds.row_count, size_of::<f32>(), "coefficient error")?;
        let one_bias_bytes = checked_composition_bytes(bounds.row_count, size_of::<f32>(), "bias")?;
        let scratch_elements = lower_plan.patch_volume.max(upper_plan.patch_volume);
        let one_scratch_bytes =
            checked_composition_bytes(scratch_elements, size_of::<f64>(), "scratch")?;
        let retained_bytes = checked_composition_byte_sum(
            &[
                bounds.lower_a.memory_bytes(),
                bounds.upper_a.memory_bytes(),
                checked_composition_bytes(
                    bounds.lower_b.len(),
                    size_of::<f32>(),
                    "retained lower bias",
                )?,
                checked_composition_bytes(
                    bounds.upper_b.len(),
                    size_of::<f32>(),
                    "retained upper bias",
                )?,
            ],
            "retained input",
        )?;
        let persistent_bytes = checked_composition_byte_sum(
            &[
                axis_bytes,
                lower_coefficient_bytes,
                upper_coefficient_bytes,
                one_error_bytes,
                one_error_bytes,
                one_bias_bytes,
                one_bias_bytes,
            ],
            "persistent result",
        )?;
        let required_bytes = checked_composition_byte_sum(
            &[
                retained_bytes,
                persistent_bytes,
                one_scratch_bytes,
                one_scratch_bytes,
            ],
            "resident peak",
        )?;
        let budget_bytes = budget_override
            .unwrap_or_else(crate::network::crown_memory::cpu_crown_dense_budget_bytes);
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "ConvTranspose2d anchored composition resident peak",
            });
        }

        let row_origins = self.build_composed_anchored_axis(
            &bounds.lower_a.geometry,
            lower_plan.spec_h,
            true,
            required_bytes,
            budget_bytes,
            deadline,
        )?;
        enforce_anchored_capacity_budget(
            &[(row_origins.capacity(), size_of::<i128>())],
            checked_composition_byte_sum(
                &[
                    retained_bytes,
                    checked_composition_bytes(
                        lower_plan.spec_w,
                        size_of::<i128>(),
                        "remaining column axis",
                    )?,
                    lower_coefficient_bytes,
                    upper_coefficient_bytes,
                    one_error_bytes,
                    one_error_bytes,
                    one_bias_bytes,
                    one_bias_bytes,
                    one_scratch_bytes,
                    one_scratch_bytes,
                ],
                "remaining after row axis",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored composition row-axis capacity",
        )?;
        let column_origins = self.build_composed_anchored_axis(
            &bounds.lower_a.geometry,
            lower_plan.spec_w,
            false,
            required_bytes,
            budget_bytes,
            deadline,
        )?;
        let axis_capacities = [
            (row_origins.capacity(), size_of::<i128>()),
            (column_origins.capacity(), size_of::<i128>()),
        ];
        enforce_anchored_capacity_budget(
            &axis_capacities,
            checked_composition_byte_sum(
                &[
                    retained_bytes,
                    lower_coefficient_bytes,
                    upper_coefficient_bytes,
                    one_error_bytes,
                    one_error_bytes,
                    one_bias_bytes,
                    one_bias_bytes,
                    one_scratch_bytes,
                    one_scratch_bytes,
                ],
                "remaining after axes",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored composition axis capacities",
        )?;
        let mut lower_values = allocate_zeroed_patch_values(
            lower_plan.coefficient_elements,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored composition lower coefficients",
            deadline,
        )?;
        let lower_coefficient_capacities = [
            axis_capacities[0],
            axis_capacities[1],
            (lower_values.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &lower_coefficient_capacities,
            checked_composition_byte_sum(
                &[
                    retained_bytes,
                    upper_coefficient_bytes,
                    one_error_bytes,
                    one_error_bytes,
                    one_bias_bytes,
                    one_bias_bytes,
                    one_scratch_bytes,
                    one_scratch_bytes,
                ],
                "remaining after lower coefficients",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored composition lower coefficient capacity",
        )?;
        let mut upper_values = allocate_zeroed_patch_values(
            upper_plan.coefficient_elements,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored composition upper coefficients",
            deadline,
        )?;
        let coefficient_capacities = [
            lower_coefficient_capacities[0],
            lower_coefficient_capacities[1],
            lower_coefficient_capacities[2],
            (upper_values.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &coefficient_capacities,
            checked_composition_byte_sum(
                &[
                    retained_bytes,
                    one_error_bytes,
                    one_error_bytes,
                    one_bias_bytes,
                    one_bias_bytes,
                    one_scratch_bytes,
                    one_scratch_bytes,
                ],
                "remaining after coefficients",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored composition coefficient capacities",
        )?;
        let mut lower_errors = allocate_zeroed_patch_values(
            bounds.row_count,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored composition lower error",
            deadline,
        )?;
        let lower_error_capacities = [
            coefficient_capacities[0],
            coefficient_capacities[1],
            coefficient_capacities[2],
            coefficient_capacities[3],
            (lower_errors.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &lower_error_capacities,
            checked_composition_byte_sum(
                &[
                    retained_bytes,
                    one_error_bytes,
                    one_bias_bytes,
                    one_bias_bytes,
                    one_scratch_bytes,
                    one_scratch_bytes,
                ],
                "remaining after lower error",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored composition lower-error capacity",
        )?;
        let mut upper_errors = allocate_zeroed_patch_values(
            bounds.row_count,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored composition upper error",
            deadline,
        )?;
        let error_capacities = [
            lower_error_capacities[0],
            lower_error_capacities[1],
            lower_error_capacities[2],
            lower_error_capacities[3],
            lower_error_capacities[4],
            (upper_errors.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &error_capacities,
            checked_composition_byte_sum(
                &[
                    retained_bytes,
                    one_bias_bytes,
                    one_bias_bytes,
                    one_scratch_bytes,
                    one_scratch_bytes,
                ],
                "remaining after errors",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored composition error capacities",
        )?;
        let mut lower_bias_values = allocate_zeroed_patch_values(
            bounds.row_count,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored composition lower bias",
            deadline,
        )?;
        lower_bias_values.clear();
        let lower_bias_capacities = [
            error_capacities[0],
            error_capacities[1],
            error_capacities[2],
            error_capacities[3],
            error_capacities[4],
            error_capacities[5],
            (lower_bias_values.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &lower_bias_capacities,
            checked_composition_byte_sum(
                &[
                    retained_bytes,
                    one_bias_bytes,
                    one_scratch_bytes,
                    one_scratch_bytes,
                ],
                "remaining after lower bias",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored composition lower-bias capacity",
        )?;
        let mut upper_bias_values = allocate_zeroed_patch_values(
            bounds.row_count,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored composition upper bias",
            deadline,
        )?;
        upper_bias_values.clear();
        let persistent_capacities = [
            lower_bias_capacities[0],
            lower_bias_capacities[1],
            lower_bias_capacities[2],
            lower_bias_capacities[3],
            lower_bias_capacities[4],
            lower_bias_capacities[5],
            lower_bias_capacities[6],
            (upper_bias_values.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &persistent_capacities,
            checked_composition_byte_sum(
                &[retained_bytes, one_scratch_bytes, one_scratch_bytes],
                "remaining after persistent allocations",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored composition persistent capacities",
        )?;
        let mut scratch_lower = allocate_zeroed_f64_values(
            scratch_elements,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored composition lower directed scratch",
            deadline,
        )?;
        let lower_scratch_capacities = [
            persistent_capacities[0],
            persistent_capacities[1],
            persistent_capacities[2],
            persistent_capacities[3],
            persistent_capacities[4],
            persistent_capacities[5],
            persistent_capacities[6],
            persistent_capacities[7],
            (scratch_lower.capacity(), size_of::<f64>()),
        ];
        enforce_anchored_capacity_budget(
            &lower_scratch_capacities,
            checked_composition_byte_sum(
                &[retained_bytes, one_scratch_bytes],
                "remaining after lower scratch",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored composition lower-scratch capacity",
        )?;
        let mut scratch_upper = allocate_zeroed_f64_values(
            scratch_elements,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored composition upper directed scratch",
            deadline,
        )?;

        // `try_reserve_exact` may round every allocation upward. Reconcile all
        // simultaneously resident capacities, plus the borrowed input carrier,
        // before the first coefficient is written.
        enforce_anchored_capacity_budget(
            &[
                lower_scratch_capacities[0],
                lower_scratch_capacities[1],
                lower_scratch_capacities[2],
                lower_scratch_capacities[3],
                lower_scratch_capacities[4],
                lower_scratch_capacities[5],
                lower_scratch_capacities[6],
                lower_scratch_capacities[7],
                lower_scratch_capacities[8],
                (scratch_upper.capacity(), size_of::<f64>()),
            ],
            retained_bytes,
            budget_bytes,
            "ConvTranspose2d anchored composition allocated resident peak",
        )?;

        let kernel_gain = self.anchored_composition_kernel_gain(deadline)?;
        self.fill_anchored_composition_side(
            &bounds.lower_a,
            lower_plan,
            (in_h, in_w),
            &row_origins,
            &column_origins,
            kernel_gain,
            &mut lower_values,
            &mut lower_errors,
            &mut scratch_lower,
            &mut scratch_upper,
            deadline,
        )?;
        self.fill_anchored_composition_side(
            &bounds.upper_a,
            upper_plan,
            (in_h, in_w),
            &row_origins,
            &column_origins,
            kernel_gain,
            &mut upper_values,
            &mut upper_errors,
            &mut scratch_lower,
            &mut scratch_upper,
            deadline,
        )?;
        self.fill_anchored_composition_bias_side(
            &bounds.lower_a,
            lower_plan,
            &bounds.lower_b,
            self.bias.as_ref(),
            true,
            &mut lower_bias_values,
            deadline,
        )?;
        self.fill_anchored_composition_bias_side(
            &bounds.upper_a,
            upper_plan,
            &bounds.upper_b,
            self.bias.as_ref(),
            false,
            &mut upper_bias_values,
            deadline,
        )?;
        drop(scratch_lower);
        drop(scratch_upper);

        let geometry = PatchGeometry::anchored(row_origins, column_origins)?;
        let lower_patches = ArrayD::from_shape_vec(IxDyn(&lower_plan.shape(in_c)), lower_values)
            .map_err(|error| {
                NyError::InternalError(format!(
                    "ConvTranspose2d anchored composition lower reshape failed: {error}"
                ))
            })?;
        let upper_patches = ArrayD::from_shape_vec(IxDyn(&upper_plan.shape(in_c)), upper_values)
            .map_err(|error| {
                NyError::InternalError(format!(
                    "ConvTranspose2d anchored composition upper reshape failed: {error}"
                ))
            })?;
        let output_shape = bounds.lower_a.output_shape;
        let result = PatchesLinearBounds {
            row_count: bounds.row_count,
            lower_a: PatchesData {
                patches: Some(lower_patches),
                geometry: geometry.clone(),
                identity: false,
                output_shape,
                input_shape: conv_input_shape,
                unstable_idx: None,
                coeff_err: Some(Array1::from_vec(lower_errors)),
            },
            lower_b: Array1::from_vec(lower_bias_values),
            upper_a: PatchesData {
                patches: Some(upper_patches),
                geometry,
                identity: false,
                output_shape,
                input_shape: conv_input_shape,
                unstable_idx: None,
                coeff_err: Some(Array1::from_vec(upper_errors)),
            },
            upper_b: Array1::from_vec(upper_bias_values),
        };
        let result = CrownBounds::Patches(Box::new(result));
        check_convtranspose_patches_deadline(deadline, "after anchored composition wrapping")?;
        Ok(result)
    }

    /// Construct the exact Anchored backward relation for a full virtual
    /// identity seed at any positive stride.
    ///
    /// No multiply or reduction is performed. DAZ-stable output coefficients
    /// are copied from one original kernel entry; binary32-subnormal entries
    /// use a zero center plus a normal per-row certificate on both sides. The
    /// irregular inverse map is stored in `PatchGeometry::Anchored`.
    /// Materialized 6D/7D carriers use the separate certified composition
    /// helper. Both routes share the caller's optional absolute deadline.
    fn propagate_anchored_identity_with_deadline_and_budget_for_input_shape(
        &self,
        bounds: &PatchesLinearBounds,
        deadline: Option<Instant>,
        budget_override: Option<usize>,
        (in_h, in_w): (usize, usize),
    ) -> Result<CrownBounds> {
        check_convtranspose_patches_deadline(deadline, "before anchored identity validation")?;

        // Authenticate the exact lower/upper pair inside the materializer, not
        // only at its current dispatcher. This rejects unequal input/output
        // metadata, unequal typed geometry, hidden tensors, and any coeff_err
        // that this direct identity route would otherwise silently drop.
        if bounds.lower_a.input_shape != bounds.upper_a.input_shape
            || bounds.lower_a.output_shape != bounds.upper_a.output_shape
        {
            return Err(NyError::InvalidSpec(
                "ConvTranspose2d anchored identity lower/upper shape metadata differs".into(),
            ));
        }
        bounds.lower_a.validate_common_geometry(&bounds.upper_a)?;
        bounds.lower_a.validate_identity_geometry()?;
        bounds.upper_a.validate_identity_geometry()?;
        if bounds.lower_a.unstable_idx.is_some() || bounds.upper_a.unstable_idx.is_some() {
            return Err(NyError::UnsupportedConfiguration(
                "ConvTranspose2d anchored identity Patches does not yet support sparse identity; use dense CROWN"
                    .into(),
            ));
        }

        let (in_c, out_c) = self.validate_geometry()?;
        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        if out_h == 0 || out_w == 0 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d anchored identity Patches requires non-empty output axes, got ({out_h},{out_w}); use dense CROWN"
            )));
        }
        let output_shape = (out_c, out_h, out_w);
        if bounds.lower_a.output_shape != output_shape {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_c, out_h, out_w],
                got: vec![
                    bounds.lower_a.output_shape.0,
                    bounds.lower_a.output_shape.1,
                    bounds.lower_a.output_shape.2,
                ],
            });
        }
        if bounds.upper_a.output_shape != output_shape {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_c, out_h, out_w],
                got: vec![
                    bounds.upper_a.output_shape.0,
                    bounds.upper_a.output_shape.1,
                    bounds.upper_a.output_shape.2,
                ],
            });
        }

        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored identity output size overflows usize".into(),
            )
        })?;
        if bounds.row_count != out_dim
            || bounds.lower_b.len() != out_dim
            || bounds.upper_b.len() != out_dim
        {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim, out_dim, out_dim],
                got: vec![bounds.row_count, bounds.lower_b.len(), bounds.upper_b.len()],
            });
        }
        let mut validation_work = 0usize;
        for value in bounds.lower_b.iter().chain(bounds.upper_b.iter()) {
            poll_convtranspose_planner(
                &mut validation_work,
                deadline,
                "while validating incoming identity bias",
            )?;
            if value.is_nan() {
                return Err(NyError::NumericalInstability(
                    "ConvTranspose2d anchored identity Patches: incoming bias contains NaN".into(),
                ));
            }
        }
        check_convtranspose_patches_deadline(deadline, "after incoming identity bias validation")?;

        let mut identity_requires_coeff_err = false;
        let mut kernel_scan_work = 0usize;
        for &coefficient in self.kernel.iter() {
            poll_convtranspose_planner(
                &mut kernel_scan_work,
                deadline,
                "while scanning identity coefficients for DAZ instability",
            )?;
            identity_requires_coeff_err |= ct_f32_is_subnormal(coefficient);
        }
        check_convtranspose_patches_deadline(deadline, "after identity coefficient DAZ scan")?;

        let (kh, kw) = self.kernel_size();
        let patch_h = anchored_axis_extent(kh, self.dilation.0, self.stride.0, "row")?;
        let patch_w = anchored_axis_extent(kw, self.dilation.1, self.stride.1, "column")?;
        // Compare the RAW stored rectangle. Unlike a hypothetical trimmed
        // carrier, this planner allocates every out-of-image tap too; clamping
        // either axis here could admit an asymmetric patch larger than Dense.
        let patch_area = patch_h.checked_mul(patch_w).ok_or_else(|| {
            NyError::InvalidSpec("ConvTranspose2d anchored patch area overflows usize".into())
        })?;
        let input_area = in_h.checked_mul(in_w).ok_or_else(|| {
            NyError::InvalidSpec("ConvTranspose2d anchored input area overflows usize".into())
        })?;
        // Refuse before allocating either side at the established patches/dense
        // crossover. Materializing Dense while both fresh patch arrays remain
        // resident has a larger peak than either representation and would make
        // the preflight incomplete. The dispatcher can instead densify the
        // still-virtual identity and run the exact dense ConvTranspose path.
        if patch_area >= input_area {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d anchored identity patch area {patch_area} reaches input area {input_area}; use dense CROWN"
            )));
        }
        let patch_shape = [out_c, out_h, out_w, in_c, patch_h, patch_w];
        let patch_elements = checked_shape_product(&patch_shape).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored identity patch shape overflows usize".into(),
            )
        })?;
        let patch_bytes = patch_elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "ConvTranspose2d anchored identity patch bytes overflow usize".into(),
                )
            })?;
        let bias_bytes = out_dim.checked_mul(size_of::<f32>()).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored identity bias bytes overflow usize".into(),
            )
        })?;
        let axis_elements = out_h.checked_add(out_w).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored identity axis count overflows usize".into(),
            )
        })?;
        let axis_bytes = axis_elements
            .checked_mul(size_of::<i128>())
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "ConvTranspose2d anchored identity axis bytes overflow usize".into(),
                )
            })?;
        let both_patch_bytes = patch_bytes.checked_mul(2).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored paired patch bytes overflow usize".into(),
            )
        })?;
        let both_bias_bytes = bias_bytes.checked_mul(2).ok_or_else(|| {
            NyError::InvalidSpec("ConvTranspose2d anchored paired bias bytes overflow usize".into())
        })?;
        let one_error_bytes = if identity_requires_coeff_err {
            checked_composition_bytes(out_dim, size_of::<f32>(), "identity coefficient error")?
        } else {
            0
        };
        let both_error_bytes = one_error_bytes.checked_mul(2).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored paired coefficient-error bytes overflow usize".into(),
            )
        })?;
        // The source remains borrowed for the entire operation. Charge both
        // carriers (including typed-geometry backing storage) and both source
        // bias vectors alongside the fresh result and axes.
        let retained_bytes = checked_composition_byte_sum(
            &[
                bounds.lower_a.memory_bytes(),
                bounds.upper_a.memory_bytes(),
                checked_composition_bytes(
                    bounds.lower_b.len(),
                    size_of::<f32>(),
                    "retained lower identity bias",
                )?,
                checked_composition_bytes(
                    bounds.upper_b.len(),
                    size_of::<f32>(),
                    "retained upper identity bias",
                )?,
            ],
            "anchored identity retained input",
        )?;
        let required_bytes = checked_composition_byte_sum(
            &[
                retained_bytes,
                both_patch_bytes,
                both_bias_bytes,
                both_error_bytes,
                axis_bytes,
            ],
            "anchored identity total-live resident peak",
        )?;
        let budget_bytes = budget_override
            .unwrap_or_else(crate::network::crown_memory::cpu_crown_dense_budget_bytes);
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "ConvTranspose2d anchored identity Patches result",
            });
        }
        check_convtranspose_patches_deadline(deadline, "after anchored memory preflight")?;

        let row_origins = build_anchored_axis(
            out_h,
            self.padding.0,
            kh,
            self.dilation.0,
            self.stride.0,
            "row",
            required_bytes,
            budget_bytes,
            deadline,
        )?;
        let column_axis_minimum = out_w.checked_mul(size_of::<i128>()).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored column allocation bytes overflow usize".into(),
            )
        })?;
        let remaining_after_row = checked_composition_byte_sum(
            &[
                retained_bytes,
                column_axis_minimum,
                both_patch_bytes,
                both_bias_bytes,
                both_error_bytes,
            ],
            "anchored identity remaining after row axis",
        )?;
        enforce_anchored_capacity_budget(
            &[(row_origins.capacity(), size_of::<i128>())],
            remaining_after_row,
            budget_bytes,
            "ConvTranspose2d anchored row capacity",
        )?;
        let column_origins = build_anchored_axis(
            out_w,
            self.padding.1,
            kw,
            self.dilation.1,
            self.stride.1,
            "column",
            required_bytes,
            budget_bytes,
            deadline,
        )?;
        let axis_capacities = [
            (row_origins.capacity(), size_of::<i128>()),
            (column_origins.capacity(), size_of::<i128>()),
        ];
        enforce_anchored_capacity_budget(
            &axis_capacities,
            checked_composition_byte_sum(
                &[
                    retained_bytes,
                    both_patch_bytes,
                    both_bias_bytes,
                    both_error_bytes,
                ],
                "anchored identity remaining after axes",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored axis capacities",
        )?;

        let mut lower_values = allocate_zeroed_patch_values(
            patch_elements,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored lower coefficient allocation",
            deadline,
        )?;
        let lower_capacity_parts = [
            axis_capacities[0],
            axis_capacities[1],
            (lower_values.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &lower_capacity_parts,
            checked_composition_byte_sum(
                &[
                    retained_bytes,
                    patch_bytes,
                    both_bias_bytes,
                    both_error_bytes,
                ],
                "anchored identity remaining after lower coefficients",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored lower coefficient capacity",
        )?;
        let mut upper_values = allocate_zeroed_patch_values(
            patch_elements,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored upper coefficient allocation",
            deadline,
        )?;
        let coefficient_capacity_parts = [
            lower_capacity_parts[0],
            lower_capacity_parts[1],
            lower_capacity_parts[2],
            (upper_values.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &coefficient_capacity_parts,
            checked_composition_byte_sum(
                &[retained_bytes, both_bias_bytes, both_error_bytes],
                "anchored identity remaining after coefficients",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored coefficient capacities",
        )?;
        let error_elements = if identity_requires_coeff_err {
            out_dim
        } else {
            0
        };
        let mut lower_errors = allocate_zeroed_patch_values(
            error_elements,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored lower coefficient-error allocation",
            deadline,
        )?;
        let lower_error_capacity_parts = [
            coefficient_capacity_parts[0],
            coefficient_capacity_parts[1],
            coefficient_capacity_parts[2],
            coefficient_capacity_parts[3],
            (lower_errors.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &lower_error_capacity_parts,
            checked_composition_byte_sum(
                &[retained_bytes, both_bias_bytes, one_error_bytes],
                "anchored identity remaining after lower coefficient error",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored lower coefficient-error capacity",
        )?;
        let mut upper_errors = allocate_zeroed_patch_values(
            error_elements,
            required_bytes,
            budget_bytes,
            "ConvTranspose2d anchored upper coefficient-error allocation",
            deadline,
        )?;
        let certified_capacity_parts = [
            lower_error_capacity_parts[0],
            lower_error_capacity_parts[1],
            lower_error_capacity_parts[2],
            lower_error_capacity_parts[3],
            lower_error_capacity_parts[4],
            (upper_errors.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &certified_capacity_parts,
            checked_composition_byte_sum(
                &[retained_bytes, both_bias_bytes],
                "anchored identity remaining after coefficient errors",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored coefficient-error capacities",
        )?;
        check_convtranspose_patches_deadline(deadline, "after anchored proof allocation")?;

        // Fill only residue-compatible kernel taps. `candidate-origin` is the
        // exact tap in the compact input-coordinate interval. Assignment (not
        // addition) is correct because positive dilation makes the kernel tap
        // for a fixed (output,input) coordinate unique.
        let mut work_since_poll = 0usize;
        for oc in 0..out_c {
            for oh in 0..out_h {
                let origin_h = row_origins[oh];
                for ow in 0..out_w {
                    let origin_w = column_origins[ow];
                    let logical_row = (oc * out_h + oh) * out_w + ow;
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            poll_convtranspose_planner(
                                &mut work_since_poll,
                                deadline,
                                "during anchored coefficient fill",
                            )?;
                            let Some(input_h) = anchored_axis_candidate(
                                oh,
                                self.padding.0,
                                ki,
                                self.dilation.0,
                                self.stride.0,
                                "row",
                            )?
                            else {
                                continue;
                            };
                            let tap_h = input_h
                                .checked_sub(origin_h)
                                .and_then(|value| usize::try_from(value).ok())
                                .filter(|&value| value < patch_h)
                                .ok_or_else(|| {
                                    NyError::InternalError(format!(
                                        "ConvTranspose2d anchored row tap outside planned extent: output={oh}, kernel={ki}, input={input_h}, origin={origin_h}, extent={patch_h}"
                                    ))
                                })?;
                            for kj in 0..kw {
                                poll_convtranspose_planner(
                                    &mut work_since_poll,
                                    deadline,
                                    "during anchored coefficient fill",
                                )?;
                                let Some(input_w) = anchored_axis_candidate(
                                    ow,
                                    self.padding.1,
                                    kj,
                                    self.dilation.1,
                                    self.stride.1,
                                    "column",
                                )?
                                else {
                                    continue;
                                };
                                let tap_w = input_w
                                    .checked_sub(origin_w)
                                    .and_then(|value| usize::try_from(value).ok())
                                    .filter(|&value| value < patch_w)
                                    .ok_or_else(|| {
                                        NyError::InternalError(format!(
                                            "ConvTranspose2d anchored column tap outside planned extent: output={ow}, kernel={kj}, input={input_w}, origin={origin_w}, extent={patch_w}"
                                        ))
                                    })?;
                                let offset = checked_patch_offset(
                                    [oc, oh, ow, ic, tap_h, tap_w],
                                    patch_shape,
                                    patch_elements,
                                )?;
                                let source = self.kernel[[ic, oc, ki, kj]];
                                let coefficient = if ct_f32_is_subnormal(source) {
                                    let published = ct_publish_error_up_normal(
                                        ct_f32_to_f64_exact_bits(source).abs(),
                                    );
                                    if published > lower_errors[logical_row] {
                                        lower_errors[logical_row] = published;
                                    }
                                    if published > upper_errors[logical_row] {
                                        upper_errors[logical_row] = published;
                                    }
                                    0.0
                                } else {
                                    source
                                };
                                lower_values[offset] = coefficient;
                                upper_values[offset] = coefficient;
                            }
                        }
                    }
                }
            }
        }
        check_convtranspose_patches_deadline(deadline, "after anchored coefficient fill")?;

        let mut lower_bias_values = Vec::new();
        lower_bias_values
            .try_reserve_exact(out_dim)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "ConvTranspose2d anchored lower bias allocation",
            })?;
        let lower_bias_capacity_parts = [
            certified_capacity_parts[0],
            certified_capacity_parts[1],
            certified_capacity_parts[2],
            certified_capacity_parts[3],
            certified_capacity_parts[4],
            certified_capacity_parts[5],
            (lower_bias_values.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &lower_bias_capacity_parts,
            checked_composition_byte_sum(
                &[retained_bytes, bias_bytes],
                "anchored identity remaining after lower bias",
            )?,
            budget_bytes,
            "ConvTranspose2d anchored lower bias capacity",
        )?;
        let mut upper_bias_values = Vec::new();
        upper_bias_values
            .try_reserve_exact(out_dim)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site: "ConvTranspose2d anchored upper bias allocation",
            })?;
        let all_capacity_parts = [
            lower_bias_capacity_parts[0],
            lower_bias_capacity_parts[1],
            lower_bias_capacity_parts[2],
            lower_bias_capacity_parts[3],
            lower_bias_capacity_parts[4],
            lower_bias_capacity_parts[5],
            lower_bias_capacity_parts[6],
            (upper_bias_values.capacity(), size_of::<f32>()),
        ];
        enforce_anchored_capacity_budget(
            &all_capacity_parts,
            retained_bytes,
            budget_bytes,
            "ConvTranspose2d anchored resident capacities",
        )?;
        let spatial = out_h.checked_mul(out_w).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d anchored identity spatial size overflows usize".into(),
            )
        })?;
        for row in 0..out_dim {
            poll_convtranspose_planner(
                &mut work_since_poll,
                deadline,
                "during anchored identity bias propagation",
            )?;
            if let Some(bias) = self.bias.as_ref() {
                let oc = row / spatial;
                lower_bias_values.push(add_f32_bias_down_no_subnormal(
                    bounds.lower_b[row],
                    bias[oc],
                ));
                upper_bias_values.push(add_f32_bias_up_no_subnormal(bounds.upper_b[row], bias[oc]));
            } else {
                lower_bias_values.push(bounds.lower_b[row]);
                upper_bias_values.push(bounds.upper_b[row]);
            }
        }

        // `try_reserve_exact` is permitted to round capacity upward. Reconcile
        // the allocator-reported capacities before publication so the resident
        // result never exceeds the configured budget even on such an allocator.
        enforce_anchored_capacity_budget(
            &all_capacity_parts,
            retained_bytes,
            budget_bytes,
            "ConvTranspose2d anchored identity allocated capacities",
        )?;
        check_convtranspose_patches_deadline(deadline, "before anchored identity publication")?;

        let geometry = PatchGeometry::anchored(row_origins, column_origins)?;
        let lower_patches =
            ArrayD::from_shape_vec(IxDyn(&patch_shape), lower_values).map_err(|error| {
                NyError::InternalError(format!(
                    "ConvTranspose2d anchored lower patch shape construction failed: {error}"
                ))
            })?;
        let upper_patches =
            ArrayD::from_shape_vec(IxDyn(&patch_shape), upper_values).map_err(|error| {
                NyError::InternalError(format!(
                    "ConvTranspose2d anchored upper patch shape construction failed: {error}"
                ))
            })?;

        let result = PatchesLinearBounds {
            row_count: out_dim,
            lower_a: PatchesData {
                patches: Some(lower_patches),
                geometry: geometry.clone(),
                identity: false,
                output_shape,
                input_shape: (in_c, in_h, in_w),
                unstable_idx: None,
                coeff_err: identity_requires_coeff_err.then(|| Array1::from_vec(lower_errors)),
            },
            lower_b: Array1::from_vec(lower_bias_values),
            upper_a: PatchesData {
                patches: Some(upper_patches),
                geometry,
                identity: false,
                output_shape,
                input_shape: (in_c, in_h, in_w),
                unstable_idx: None,
                coeff_err: identity_requires_coeff_err.then(|| Array1::from_vec(upper_errors)),
            },
            upper_b: Array1::from_vec(upper_bias_values),
        };

        let result = Box::new(result);
        check_convtranspose_patches_deadline(deadline, "after anchored result wrapping")?;
        Ok(CrownBounds::Patches(result))
    }

    #[cfg(test)]
    pub(crate) fn propagate_anchored_identity_with_budget_for_test(
        &self,
        bounds: &PatchesLinearBounds,
        budget_bytes: usize,
    ) -> Result<CrownBounds> {
        let input_shape = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose2d Patches CROWN test helper requires input_shape".into(),
            )
        })?;
        self.propagate_anchored_identity_with_deadline_and_budget_for_input_shape(
            bounds,
            None,
            Some(budget_bytes),
            input_shape,
        )
    }

    /// Engine-aware patches ConvTranspose2d CROWN backward.
    ///
    /// The no-deadline stride-1 case reduces to the equivalent `Conv2d`
    /// (flip+swap kernel, adjusted padding, same bias). Finite stride 1 and
    /// every stride>1 request use either the certified direct identity planner
    /// or the certified directed-f64 composition planner. Returned bounds are
    /// over this layer's input space, exactly like dense
    /// `propagate_linear_with_engine`.
    pub(crate) fn propagate_patches_engine(
        &self,
        bounds: &PatchesLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownBounds> {
        self.propagate_patches_engine_and_deadline(bounds, engine, None)
    }

    pub(crate) fn propagate_patches_engine_and_deadline(
        &self,
        bounds: &PatchesLinearBounds,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<CrownBounds> {
        let input_shape = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose2d Patches CROWN requires input_shape to be set. Use with_input_shape() or set_input_shape()."
                    .to_string(),
            )
        })?;
        self.propagate_patches_engine_and_deadline_for_input_shape(
            bounds,
            engine,
            deadline,
            input_shape,
        )
    }

    /// Borrowing shape-override variant for deadline-bearing dispatchers. The
    /// layer's kernel and bias remain borrowed, so installing current spatial
    /// metadata cannot trigger an unpollable deep clone.
    pub(crate) fn propagate_patches_engine_and_deadline_for_input_shape(
        &self,
        bounds: &PatchesLinearBounds,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        input_shape @ (in_h, in_w): (usize, usize),
    ) -> Result<CrownBounds> {
        check_convtranspose_patches_deadline(deadline, "before dispatch")?;
        self.validate_geometry()?;

        // The equivalent-Conv2d construction below allocates and copies a full
        // transformed kernel without a complete fallible memory receipt. Keep
        // it as the historical no-deadline stride-1 implementation, but route
        // every finite request through the already cooperative, budgeted
        // Anchored planners. Their checked inverse-grid formulas are valid for
        // every positive stride and reduce exactly when stride == 1.
        if self.stride != (1, 1) || deadline.is_some() {
            let lower_identity = bounds.lower_a.identity;
            let upper_identity = bounds.upper_a.identity;
            if lower_identity != upper_identity {
                return Err(NyError::UnsupportedConfiguration(
                    "ConvTranspose2d anchored Patches requires both sides to be identity or both materialized; mixed sides use dense CROWN"
                        .into(),
                ));
            }
            self.validate_anchored_layer_finite_sources(deadline)?;
            return if lower_identity {
                bounds.lower_a.validate_common_geometry(&bounds.upper_a)?;
                self.propagate_anchored_identity_with_deadline_and_budget_for_input_shape(
                    bounds,
                    deadline,
                    None,
                    input_shape,
                )
            } else {
                self.propagate_anchored_composition_with_deadline_for_input_shape(
                    bounds,
                    deadline,
                    None,
                    input_shape,
                )
            };
        }

        // A second ConvTranspose can receive Anchored coefficients from the
        // identity route. Refuse that typed variant in O(1), before common
        // validation would scan every origin on both sides.
        bounds
            .lower_a
            .geometry
            .require_affine("stride-1 ConvTranspose2d Patches propagation")?;
        bounds
            .upper_a
            .geometry
            .require_affine("stride-1 ConvTranspose2d Patches propagation")?;
        if self.dilation != (1, 1) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d Patches CROWN does not support dilation {:?}; use dense CROWN",
                self.dilation
            )));
        }
        // For stride 1 the layer constructor already forces output_padding == 0
        // (output_padding < stride); guard defensively regardless.
        if self.output_padding != (0, 0) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d Patches CROWN does not support output_padding {:?}; use dense CROWN",
                self.output_padding
            )));
        }

        let (kh, kw) = self.kernel_size();
        let (ph, pw) = self.padding;
        // The equivalent Conv2d padding is Cp = (kh-1-ph, kw-1-pw). When the
        // ConvTranspose padding exceeds kernel-1 in a dimension, Cp would be
        // negative — not representable as a Conv2d padding without cropping the
        // kernel/patch. Route that corner to the sound dense path.
        if ph > kh - 1 || pw > kw - 1 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d Patches CROWN cannot represent padding ({ph},{pw}) > kernel-1 \
                 ({},{}) as an equivalent Conv2d padding (stage 2a); use dense CROWN",
                kh - 1,
                kw - 1
            )));
        }
        let cph = kh - 1 - ph;
        let cpw = kw - 1 - pw;
        bounds.lower_a.validate_common_geometry(&bounds.upper_a)?;
        guard_nan_weights_with_poll(
            &self.kernel,
            self.bias.as_ref(),
            "ConvTranspose2d Patches",
            &mut || check_convtranspose_patches_deadline(deadline, "during NaN guard"),
        )?;

        // Reduce to the equivalent Conv2d and reuse its proven patches path
        // (identity build, non-identity composition, certified coeff_err, and the
        // outward-rounded bias). The Conv2d INPUT space equals this
        // ConvTranspose's INPUT space ((in_c, in_h, in_w)), and the Conv2d OUTPUT
        // space equals this ConvTranspose's OUTPUT space ((out_c, out_h, out_w)),
        // so the incoming patches (over the ConvTranspose output space) and the
        // result (over the ConvTranspose input space) map through unchanged.
        let kc = self.crown_backward_equivalent_conv2d_kernel_with_deadline(deadline)?;
        let equiv_conv =
            Conv2dLayer::with_input_shape(kc, self.bias.clone(), (1, 1), (cph, cpw), in_h, in_w)?;
        equiv_conv.propagate_patches_engine_and_deadline(bounds, engine, deadline)
    }
}
