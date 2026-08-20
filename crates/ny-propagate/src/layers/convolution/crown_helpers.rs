// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2};
use ny_core::{
    dd::{next_down_f64, next_up_f64},
    is_crown_coeff_safe, NyError, Result,
};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use tracing::{debug, warn};

/// Temporary share probes for the certified conv arms (read via the
/// collection dump lever).
pub(crate) static CONV_BIAS_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static CONV_ERR_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(crate) struct ShareTimer(&'static std::sync::atomic::AtomicU64, std::time::Instant);
impl ShareTimer {
    pub(crate) fn new(sink: &'static std::sync::atomic::AtomicU64) -> Self {
        Self(sink, std::time::Instant::now())
    }
}
impl Drop for ShareTimer {
    fn drop(&mut self) {
        self.0.fetch_add(
            u64::try_from(self.1.elapsed().as_nanos()).unwrap_or(u64::MAX),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

use crate::LinearBounds;

const F64_FRACTION_BITS: u32 = 52;
const F64_EXPONENT_BIAS: i32 = 1023;
const CROWN_HOST_POLL_ELEMENTS: usize = 4_096;

#[inline]
fn poll_crown_host_work<F>(work_since_poll: &mut usize, poll: &mut F) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    *work_since_poll += 1;
    if *work_since_poll >= CROWN_HOST_POLL_ELEMENTS {
        *work_since_poll = 0;
        poll()?;
    }
    Ok(())
}

/// Elements this reduction may consume before it must offer the caller a poll.
///
/// The budget is the SAME `CROWN_HOST_POLL_ELEMENTS` the per-element accounting
/// enforced; only the granularity moves. Sizing the first block by what is left
/// of the current budget keeps the guarantee exact rather than approximate: at
/// most `CROWN_HOST_POLL_ELEMENTS` accounted units pass between two polls, just
/// as before.
#[inline]
fn crown_host_poll_block_len(work_since_poll: usize, remaining: usize) -> usize {
    CROWN_HOST_POLL_ELEMENTS
        .saturating_sub(work_since_poll)
        .max(1)
        .min(remaining)
}

/// Account a whole BLOCK of host work at once and poll if the budget is spent.
///
/// WHY THIS EXISTS. `poll_crown_host_work` was called from inside the innermost
/// `for k` of every row reduction below. Its body is trivial, but its SHAPE is
/// not: it takes `&mut usize` and `&mut F` and returns `Result`, so the inner
/// loop contained an opaque call with a `?` early-return. That is unvectorizable
/// by construction — the compiler cannot prove the call has no side effects on
/// the accumulator, and cannot hoist a branch that may return. The reductions
/// therefore ran one f32 at a time.
///
/// Profiled on `cifar_bias_field_46`: `conv_coeff_err_matrix_with_poll` was
/// 1,503 of 7,826 main-thread samples while `gemm_f64`'s NEON microkernel took
/// 792, against 10,650 at the last commit that proved the row. "Make it
/// interruptible" had been implemented as "make it scalar".
///
/// The values are untouched. Iteration order is unchanged — which matters,
/// because `add_f64_up` is an outward-rounded accumulation and therefore
/// order-dependent — and the poll only moves out of the loop body.
#[inline]
fn poll_crown_host_block<F>(work_since_poll: &mut usize, units: usize, poll: &mut F) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    *work_since_poll += units;
    if *work_since_poll >= CROWN_HOST_POLL_ELEMENTS {
        *work_since_poll = 0;
        poll()?;
    }
    Ok(())
}

/// Decode a binary32 bit pattern without presenting a subnormal binary32
/// operand to a hardware conversion instruction. A binary32 subnormal is a
/// normal binary64 value, so all subsequent certificate arithmetic is outside
/// the binary64 FTZ/DAZ range.
#[inline]
fn f32_to_f64_exact(value: f32) -> f64 {
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

/// Decode a certified binary32 error, failing closed on invalid metadata.
///
/// The sign and exponent checks are bitwise: a floating-point comparison can
/// treat a negative subnormal as `-0.0` when DAZ is enabled and would therefore
/// accept an illegal negative error.
#[inline]
fn nonnegative_f32_error_or_infinity(value: f32) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let exponent = magnitude >> 23;
    if exponent == 0xff || (bits >> 31 != 0 && magnitude != 0) {
        f64::INFINITY
    } else {
        f32_to_f64_exact(value)
    }
}

#[inline]
fn binary32_min_normal_as_f64() -> f64 {
    f64::from_bits(((F64_EXPONENT_BIAS - 126) as u64) << F64_FRACTION_BITS)
}

/// One binary64 step down, except that a binary32-subnormal-range result is
/// replaced by the appropriate binary32 normal floor. This intentionally gives
/// up subnormal tightness so publication remains sound when FTZ is enabled.
#[inline]
fn next_down_f64_for_f32(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude > f64::INFINITY.to_bits() {
        return f64::NEG_INFINITY;
    }
    if bits == f64::NEG_INFINITY.to_bits() {
        return f64::NEG_INFINITY;
    }
    if bits == f64::INFINITY.to_bits() {
        return f64::MAX;
    }

    let min_normal = binary32_min_normal_as_f64();
    if magnitude == 0 {
        return -min_normal;
    }
    if magnitude < min_normal.to_bits() {
        return if bits & 0x8000_0000_0000_0000 != 0 {
            -min_normal
        } else {
            0.0
        };
    }

    let stepped = if bits & 0x8000_0000_0000_0000 == 0 {
        bits - 1
    } else {
        bits + 1
    };
    let result = f64::from_bits(stepped);
    let result_magnitude = stepped & 0x7fff_ffff_ffff_ffff;
    if result_magnitude != 0 && result_magnitude < min_normal.to_bits() {
        if stepped & 0x8000_0000_0000_0000 != 0 {
            -min_normal
        } else {
            0.0
        }
    } else {
        result
    }
}

#[inline]
fn next_up_f64_for_f32(value: f64) -> f64 {
    -next_down_f64_for_f32(-value)
}

/// Directed binary64-to-binary32 lower conversion that never publishes a
/// subnormal binary32 endpoint.
#[inline]
fn f64_to_f32_down_no_subnormal(value: f64) -> f32 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }

    let min_normal = binary32_min_normal_as_f64();
    if value.abs() < min_normal {
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
    if f32_to_f64_exact(nearest) <= value {
        nearest
    } else {
        next_down_f32(nearest)
    }
}

/// Directed binary64-to-binary32 upper conversion that never publishes a
/// subnormal binary32 endpoint.
#[inline]
fn f64_to_f32_up_no_subnormal(value: f64) -> f32 {
    if value.is_nan() || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }

    let min_normal = binary32_min_normal_as_f64();
    if value.abs() < min_normal {
        return if value.is_sign_negative() {
            0.0
        } else {
            f32::MIN_POSITIVE
        };
    }

    let nearest = value as f32;
    if nearest == f32::NEG_INFINITY {
        return f32::MIN;
    }
    if nearest == f32::INFINITY {
        return f32::INFINITY;
    }
    if f32_to_f64_exact(nearest) >= value {
        nearest
    } else {
        next_up_f32(nearest)
    }
}

/// Add two binary32 affine constants and publish a sound lower endpoint.
///
/// Both operands are decoded from their bits so DAZ cannot erase a subnormal
/// source.  The binary64 addition and the final binary32 conversion are each
/// directed downward, and the publication firewall never emits a binary32
/// subnormal that FTZ could subsequently move inward.
#[inline]
pub(crate) fn add_f32_bias_down_no_subnormal(lhs: f32, rhs: f32) -> f32 {
    let sum = add_f64_down(f32_to_f64_exact(lhs), f32_to_f64_exact(rhs));
    f64_to_f32_down_no_subnormal(sum)
}

/// Add two binary32 affine constants and publish a sound upper endpoint.
///
/// This is the upward-directed twin of
/// [`add_f32_bias_down_no_subnormal`].
#[inline]
pub(crate) fn add_f32_bias_up_no_subnormal(lhs: f32, rhs: f32) -> f32 {
    let sum = add_f64_up(f32_to_f64_exact(lhs), f32_to_f64_exact(rhs));
    f64_to_f32_up_no_subnormal(sum)
}

/// Publish a positive binary64 certificate term strictly above its computed
/// value. The unconditional step is important at an exact binary32 boundary:
/// the preceding round-to-nearest binary64 expression could have rounded down
/// onto that boundary even though the corresponding real expression is just
/// above it.
#[inline]
fn publish_error_up(value: f64) -> f32 {
    next_up_f32(f64_to_f32_up_no_subnormal(value))
}

/// SOUND outward widening for a conv-family IBP FORWARD (#vnncomp-aw-soundness).
///
/// The plain conv/transpose `propagate_ibp` accumulates each output over `macs` products
/// in round-to-nearest f32 (no f64, no directed rounding), so it can deviate from the true
/// value by the Higham bound `γ_macs · Σ_k |W_ok|·|x_k|`, which under cancellation vastly
/// exceeds the generic 1-ULP `round_for_soundness` widening — yielding a box that EXCLUDES
/// the true value (a false-proof on the intermediate-bound / verdict path).
///
/// `y` is the plain f32 forward; `s` is the per-output abssum
/// `S_o = Σ_k |W_ok|·max(|x_l_k|,|x_u_k|)` (same shape as `y`), obtained by running the SAME
/// interval forward with `|kernel|` (so `W+ = |W|`, `W- = 0`) on the degenerate `max(|l|,|u|)`
/// box — this handles 1D/2D/grouped/transpose uniformly. `macs` is an UPPER bound on the
/// per-output f32 accumulation depth (`(in_c/groups)·∏kernel_spatial`).
///
/// Folds the certified error
///
/// `err_o = up(γ_{macs+2}·S_safe + 2u·|y_o| + U_abs)`
///
/// OUTWARD. `S_safe = (S_o + U_abs)/(1−γ_macs) ≥ S_true` corrects both relative
/// roundoff and product/addition underflow in the independently computed `S`.
/// Here `U_abs = n·η/(1−n·u)`, `η=2^-126`, and `n=6·macs+2`: two W+/W-
/// forwards, each conservatively charged one product, one dot accumulation, and
/// one scatter accumulation per possible term, plus their combine and bias.
/// Thus every product, addition, and S-pass flush-to-zero event contributes at
/// most `η`, including the case where normal operands have an exact subnormal
/// product. The denominator covers amplification by later relative roundings.
///
/// Endpoint arithmetic is performed in binary64 from a bit-exact binary32
/// decode and published with directed conversion. A result in the binary32
/// subnormal range is widened to the adjacent normal floor, so the certificate
/// does not depend on subnormal results surviving FTZ. (Callers must separately
/// handle DAZ-sensitive *source operands*, which can be amplified arbitrarily.)
pub(crate) fn higham_widen_ibp(
    y: &BoundedTensor,
    s: &ArrayD<f32>,
    macs: usize,
) -> Result<BoundedTensor> {
    const U: f64 = 1.0 / (1u64 << 24) as f64; // f32 unit roundoff 2^-24
    let k = (macs.saturating_add(2)) as f64;
    let gamma = if k * U < 1.0 {
        (k * U) / (1.0 - k * U)
    } else {
        f64::INFINITY
    };
    let gamma_macs = {
        let m = macs as f64;
        if m * U < 1.0 {
            (m * U) / (1.0 - m * U)
        } else {
            f64::INFINITY
        }
    };
    let s_inflate = if gamma_macs < 1.0 {
        1.0 / (1.0 - gamma_macs)
    } else {
        f64::INFINITY
    };
    // A lower-level forward is allowed to use two split convolutions. Charge
    // every possible product, inner-dot add, and scatter add in both forwards,
    // even though one split product is normally exactly zero. This deliberately
    // over-counts but makes the absolute FTZ term independent of backend
    // summation order.
    let underflow_ops = macs.saturating_mul(6).saturating_add(2);
    let underflow_n = underflow_ops as f64;
    let underflow_nu = underflow_n * U;
    let underflow_abs = if underflow_nu < 1.0 {
        (underflow_n * binary32_min_normal_as_f64()) / (1.0 - underflow_nu)
    } else {
        f64::INFINITY
    };

    let mut lower = y.lower().to_owned();
    let mut upper = y.upper().to_owned();
    ndarray::Zip::from(&mut lower)
        .and(&mut upper)
        .and(s)
        .for_each(|lo_o, up_o, &s_o| {
            let lo64 = f32_to_f64_exact(*lo_o);
            let up64 = f32_to_f64_exact(*up_o);
            let mag = lo64.abs().max(up64.abs());
            let s_observed = f32_to_f64_exact(s_o);
            let s_safe = if s_observed.is_finite() && s_observed >= 0.0 {
                (s_observed + underflow_abs) * s_inflate
            } else {
                f64::INFINITY
            };
            let err_estimate = gamma * s_safe + 2.0 * U * mag + underflow_abs;
            // One outward binary32 step dominates the handful of binary64
            // roundings in the positive error expression.
            let err = publish_error_up(err_estimate);
            if err.is_finite() {
                let err64 = f32_to_f64_exact(err);
                let widened_lower = next_down_f64_for_f32(lo64 - err64);
                let widened_upper = next_up_f64_for_f32(up64 + err64);
                *lo_o = f64_to_f32_down_no_subnormal(widened_lower);
                *up_o = f64_to_f32_up_no_subnormal(widened_upper);
            } else {
                *lo_o = f32::NEG_INFINITY;
                *up_o = f32::INFINITY;
            }
        });
    BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
}

/// Guard: reject NaN weights at CROWN backward entry. (#2747)
pub(crate) fn guard_nan_weights(
    kernel: &ArrayD<f32>,
    bias: Option<&Array1<f32>>,
    layer_name: &str,
) -> Result<()> {
    guard_nan_weights_with_poll(kernel, bias, layer_name, &mut || Ok(()))
}

/// Pollable form of [`guard_nan_weights`] for finite execution authorities.
///
/// The ordinary helper delegates here with a no-op poll, preserving its error
/// taxonomy and messages. Finite callers can bound both the kernel and bias
/// scans without duplicating the corrupted-model guard.
pub(crate) fn guard_nan_weights_with_poll<F>(
    kernel: &ArrayD<f32>,
    bias: Option<&Array1<f32>>,
    layer_name: &str,
    poll: &mut F,
) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    poll()?;
    let mut work_since_poll = 0usize;
    for value in kernel {
        if value.is_nan() {
            warn!("{layer_name} CROWN backward: kernel contains NaN");
            return Err(NyError::NumericalInstability(format!(
                "{layer_name} CROWN backward: kernel contains NaN — corrupted model weights"
            )));
        }
        poll_crown_host_work(&mut work_since_poll, poll)?;
    }
    if let Some(bias) = bias {
        for value in bias {
            if value.is_nan() {
                warn!("{layer_name} CROWN backward: bias contains NaN");
                return Err(NyError::NumericalInstability(format!(
                    "{layer_name} CROWN backward: bias contains NaN — corrupted model weights"
                )));
            }
            poll_crown_host_work(&mut work_since_poll, poll)?;
        }
    }
    poll()
}

/// Detect rows with unsafe coefficients and fall back to +/-inf bias. (#2812, #2681)
pub(crate) fn detect_and_fix_nonfinite_rows(
    lower_a: &mut Array2<f32>,
    upper_a: &mut Array2<f32>,
    lower_b: &mut Array1<f32>,
    upper_b: &mut Array1<f32>,
    conv_in_size: usize,
    layer_name: &str,
) -> (usize, usize) {
    detect_and_fix_nonfinite_rows_with_poll(
        lower_a,
        upper_a,
        lower_b,
        upper_b,
        conv_in_size,
        layer_name,
        &mut || Ok(()),
    )
    .expect("infallible Conv CROWN host poll")
}

/// Pollable form of [`detect_and_fix_nonfinite_rows`] for callers carrying a
/// bounded-executor or explicit wall-clock authority.
pub(crate) fn detect_and_fix_nonfinite_rows_with_poll<F>(
    lower_a: &mut Array2<f32>,
    upper_a: &mut Array2<f32>,
    lower_b: &mut Array1<f32>,
    upper_b: &mut Array1<f32>,
    conv_in_size: usize,
    layer_name: &str,
    poll: &mut F,
) -> Result<(usize, usize)>
where
    F: FnMut() -> Result<()>,
{
    debug_assert_eq!(lower_a.ncols(), conv_in_size);
    debug_assert_eq!(upper_a.ncols(), conv_in_size);
    debug_assert_eq!(lower_a.nrows(), upper_a.nrows());
    debug_assert_eq!(lower_a.nrows(), lower_b.len());
    debug_assert_eq!(upper_a.nrows(), upper_b.len());

    let num_outputs = lower_a.nrows();
    let mut lower_affected = 0usize;
    let mut upper_affected = 0usize;
    let mut work_since_poll = 0usize;

    poll()?;
    for row_idx in 0..num_outputs {
        let mut lower_has_nonfinite = false;
        for col_idx in 0..conv_in_size {
            if !is_crown_coeff_safe(lower_a[[row_idx, col_idx]]) {
                lower_has_nonfinite = true;
                break;
            }
            poll_crown_host_work(&mut work_since_poll, poll)?;
        }
        if lower_has_nonfinite {
            for col_idx in 0..conv_in_size {
                lower_a[[row_idx, col_idx]] = 0.0;
                poll_crown_host_work(&mut work_since_poll, poll)?;
            }
            lower_b[row_idx] = f32::NEG_INFINITY;
            lower_affected += 1;
        }

        let mut upper_has_nonfinite = false;
        for col_idx in 0..conv_in_size {
            if !is_crown_coeff_safe(upper_a[[row_idx, col_idx]]) {
                upper_has_nonfinite = true;
                break;
            }
            poll_crown_host_work(&mut work_since_poll, poll)?;
        }
        if upper_has_nonfinite {
            for col_idx in 0..conv_in_size {
                upper_a[[row_idx, col_idx]] = 0.0;
                poll_crown_host_work(&mut work_since_poll, poll)?;
            }
            upper_b[row_idx] = f32::INFINITY;
            upper_affected += 1;
        }
    }

    if lower_affected > 0 || upper_affected > 0 {
        debug!(
            "{layer_name} CROWN backward: non-finite A-matrix overflow in {lower_affected}/{num_outputs} lower rows, \
             {upper_affected}/{num_outputs} upper rows — falling back to ±inf bias for affected rows"
        );
    }

    poll()?;
    Ok((lower_affected, upper_affected))
}

#[inline]
fn add_f64_down(acc: f64, term: f64) -> f64 {
    if term == 0.0 {
        return acc;
    }
    let sum = acc + term;
    if sum.is_nan() {
        f64::NEG_INFINITY
    } else {
        next_down_f64(sum)
    }
}

#[inline]
fn add_f64_up(acc: f64, term: f64) -> f64 {
    if term == 0.0 {
        return acc;
    }
    let sum = acc + term;
    if sum.is_nan() {
        f64::INFINITY
    } else {
        next_up_f64(sum)
    }
}

/// Compute convolution bias folds for flattened rows.
///
/// In addition to the stored `A * bias`, this folds the incoming certified
/// coefficient error into the constant:
///
/// `lower -= Σ A_err_lower[j] * |bias[channel(j)]|`
///
/// `upper += Σ A_err_upper[j] * |bias[channel(j)]|`.
///
/// Every f64 addition is directed outward. Products of two finite binary32
/// values are exact in binary64. Invalid error entries fail closed to an
/// infinite penalty when their corresponding bias is nonzero.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_conv_bias_rows_f64(
    lower_a: ArrayView2<'_, f32>,
    lower_a_err: Option<ArrayView2<'_, f32>>,
    lower_b: ArrayView1<'_, f32>,
    upper_a: ArrayView2<'_, f32>,
    upper_a_err: Option<ArrayView2<'_, f32>>,
    upper_b: ArrayView1<'_, f32>,
    bias: &Array1<f32>,
    out_c: usize,
    spatial_size: usize,
) -> Result<(Array1<f32>, Array1<f32>)> {
    compute_conv_bias_rows_f64_with_poll(
        lower_a,
        lower_a_err,
        lower_b,
        upper_a,
        upper_a_err,
        upper_b,
        bias,
        out_c,
        spatial_size,
        &mut || Ok(()),
    )
}

/// Pollable form of [`compute_conv_bias_rows_f64`] for finite execution
/// authorities. Arithmetic and publication order are otherwise identical.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_conv_bias_rows_f64_with_poll<F>(
    lower_a: ArrayView2<'_, f32>,
    lower_a_err: Option<ArrayView2<'_, f32>>,
    lower_b: ArrayView1<'_, f32>,
    upper_a: ArrayView2<'_, f32>,
    upper_a_err: Option<ArrayView2<'_, f32>>,
    upper_b: ArrayView1<'_, f32>,
    bias: &Array1<f32>,
    out_c: usize,
    spatial_size: usize,
    poll: &mut F,
) -> Result<(Array1<f32>, Array1<f32>)>
where
    F: FnMut() -> Result<()>,
{
    let mid_dim = out_c.checked_mul(spatial_size).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "convolution bias geometry overflows: {out_c} * {spatial_size}"
        ))
    })?;
    let rows = lower_a.nrows();

    if bias.len() != out_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_c],
            got: vec![bias.len()],
        });
    }
    for (name, shape) in [("lower_a", lower_a.shape()), ("upper_a", upper_a.shape())] {
        if shape != [rows, mid_dim] {
            return Err(NyError::InvalidSpec(format!(
                "convolution bias {name} shape mismatch: expected [{rows}, {mid_dim}], got {shape:?}"
            )));
        }
    }
    if lower_b.len() != rows || upper_b.len() != rows {
        return Err(NyError::InvalidSpec(format!(
            "convolution bias row mismatch: A has {rows} rows, lower_b has {}, upper_b has {}",
            lower_b.len(),
            upper_b.len()
        )));
    }
    for (name, err) in [
        ("lower_a_err", lower_a_err.as_ref()),
        ("upper_a_err", upper_a_err.as_ref()),
    ] {
        if let Some(err) = err {
            if err.shape() != [rows, mid_dim] {
                return Err(NyError::InvalidSpec(format!(
                    "convolution bias {name} shape mismatch: expected [{rows}, {mid_dim}], got {:?}",
                    err.shape()
                )));
            }
        }
    }

    poll()?;
    let mut new_lower_b = Array1::<f32>::zeros(rows);
    let mut new_upper_b = Array1::<f32>::zeros(rows);
    let mut work_since_poll = 0usize;
    for row in 0..rows {
        let mut lower = f32_to_f64_exact(lower_b[row]);
        let mut upper = f32_to_f64_exact(upper_b[row]);
        let mut lower_penalty = 0.0_f64;
        let mut upper_penalty = 0.0_f64;

        for c in 0..out_c {
            // Keep the zero-bias fast path bounded too: it skips all spatial
            // loops below but may still traverse a large rows×channels grid.
            poll_crown_host_work(&mut work_since_poll, poll)?;
            let bias_value = f32_to_f64_exact(bias[c]);
            let bias_abs = bias_value.abs();
            let start = c * spatial_size;
            let end = start + spatial_size;

            if bias_value != 0.0 {
                for col in start..end {
                    let lower_term = f32_to_f64_exact(lower_a[[row, col]]) * bias_value;
                    let upper_term = f32_to_f64_exact(upper_a[[row, col]]) * bias_value;
                    lower = add_f64_down(lower, lower_term);
                    upper = add_f64_up(upper, upper_term);
                    poll_crown_host_work(&mut work_since_poll, poll)?;
                }
            }

            if bias_abs != 0.0 {
                if let Some(err) = lower_a_err.as_ref() {
                    for col in start..end {
                        let value = nonnegative_f32_error_or_infinity(err[[row, col]]);
                        if !value.is_finite() {
                            lower_penalty = f64::INFINITY;
                            break;
                        }
                        lower_penalty = add_f64_up(lower_penalty, value * bias_abs);
                        poll_crown_host_work(&mut work_since_poll, poll)?;
                    }
                }
                if let Some(err) = upper_a_err.as_ref() {
                    for col in start..end {
                        let value = nonnegative_f32_error_or_infinity(err[[row, col]]);
                        if !value.is_finite() {
                            upper_penalty = f64::INFINITY;
                            break;
                        }
                        upper_penalty = add_f64_up(upper_penalty, value * bias_abs);
                        poll_crown_host_work(&mut work_since_poll, poll)?;
                    }
                }
            }
        }

        lower = add_f64_down(lower, -lower_penalty);
        upper = add_f64_up(upper, upper_penalty);
        new_lower_b[row] = f64_to_f32_down_no_subnormal(lower);
        new_upper_b[row] = f64_to_f32_up_no_subnormal(upper);
    }

    poll()?;
    Ok((new_lower_b, new_upper_b))
}

/// Compute broadcast convolution bias contribution and certified-error widening
/// in f64 with directed rounding. (#2812, #1863)
pub(crate) fn compute_conv_bias_f64(
    bounds: &LinearBounds,
    bias: Option<&Array1<f32>>,
    out_c: usize,
    spatial_size: usize,
) -> Result<(Array1<f32>, Array1<f32>)> {
    compute_conv_bias_f64_with_poll(bounds, bias, out_c, spatial_size, &mut || Ok(()))
}

/// Pollable form of [`compute_conv_bias_f64`].
pub(crate) fn compute_conv_bias_f64_with_poll<F>(
    bounds: &LinearBounds,
    bias: Option<&Array1<f32>>,
    out_c: usize,
    spatial_size: usize,
    poll: &mut F,
) -> Result<(Array1<f32>, Array1<f32>)>
where
    F: FnMut() -> Result<()>,
{
    let _share_timer = ShareTimer::new(&CONV_BIAS_NANOS);
    match bias {
        Some(bias) => compute_conv_bias_rows_f64_with_poll(
            bounds.lower_a().view(),
            bounds.lower_a_err().map(|err| err.view()),
            bounds.lower_b().view(),
            bounds.upper_a().view(),
            bounds.upper_a_err().map(|err| err.view()),
            bounds.upper_b().view(),
            bias,
            out_c,
            spatial_size,
            poll,
        ),
        None => {
            poll()?;
            let mut lower = Array1::<f32>::zeros(bounds.lower_b().len());
            let mut upper = Array1::<f32>::zeros(bounds.upper_b().len());
            let mut work_since_poll = 0usize;
            for (dst, &src) in lower.iter_mut().zip(bounds.lower_b().iter()) {
                *dst = src;
                poll_crown_host_work(&mut work_since_poll, poll)?;
            }
            for (dst, &src) in upper.iter_mut().zip(bounds.upper_b().iter()) {
                *dst = src;
                poll_crown_host_work(&mut work_since_poll, poll)?;
            }
            poll()?;
            Ok((lower, upper))
        }
    }
}

/// Whether the conv CROWN-backward f64-recomputes the coefficient.
///
/// Currently `true` for ALL contraction widths — the conv backward ALWAYS
/// f64-accumulates the coefficient and certifies `cast_err + γ_n^f64·S` (tight,
/// matching Linear's `aw_f64_with_abssum`). A small-n fast path that keeps the f32
/// GEMM coefficient + the (sound but ~2^29× larger) `γ_n^f32·S` factor was tried
/// to skip the recompute on tiny convs, but its looser certified error pushed
/// CROWN's concretized bounds past the IBP bound on tightness/CROWN⊆IBP tests, so
/// it is disabled. The f64 recompute's `γ_n^f64·S` is sub-ULP, keeping bounds tight
/// AND sound (#vnncomp-aw-soundness). The hook is retained so a future tightness-
/// preserving fast path can be slotted in centrally.
#[inline]
pub(crate) fn conv_should_f64_recompute(_n_contraction: usize) -> bool {
    true
}

/// #wall-deadwork gate — DEFAULT ON since 2026-07-20 (`NY_CONV_SKIP_DEAD_F32=0`
/// is the kill-switch).
///
/// Under `conv_should_f64_recompute` (unconditionally true today) the f32
/// coefficient GEMM pair's A-values are discarded on BOTH downstream paths:
/// recompute success overwrites them with the directed-rounded f64 result, and
/// recompute failure degrades the row to ±inf bias. The pair contributes only
/// buffer allocation and the per-node deadline check, so the skip replaces it
/// with direct allocation plus an explicit deadline check, and runs the two f64
/// recomputes concurrently (each is internally deterministic; the certified
/// error channel is summation-order independent, so the join is bit-safe).
///
/// Flip evidence (ledger 2026-07-19/20): bitwise-identity oracle on the
/// recompute path; expired-deadline and mem-cap oracles; 145/145 conv suite;
/// 226 production wall runs with zero anomalies + 2 banked-unsat guards
/// FASTER (62→54s, 59→53s); ~25% measured on the root-CROWN-intersect phase.
/// Deadline timing is intentionally not byte-identical: an already-expired
/// deadline aborts even on small work the unchunked pair would have finished.
/// Also, a future deadline can expire inside the uninterruptible f64 join after
/// the legacy chunked f32 pair would have polled and aborted. The f64 recompute
/// was already uninterruptible on the legacy path; the skip merely starts it
/// sooner. Either case can affect fallback timing, never bound soundness.
#[inline]
pub(crate) fn conv_skip_dead_f32_enabled() -> bool {
    std::env::var("NY_CONV_SKIP_DEAD_F32").ok().as_deref() != Some("0")
}

/// Build the certified per-coefficient error matrix for a conv backward
/// (#vnncomp-aw-soundness — conv f32-accumulation bug). Shared by the scalar and
/// batched paths.
///
/// Error = `cast_err + γ·S + prop`, with the row-constant over-bound
/// `S[i,p] ≤ row_max(a,i)·‖kernel‖_1` (sub-ULP once multiplied by γ, so the
/// over-bound is harmless) and, for the incoming-error propagation term, either
///
///   - `prop_exact = Some(P)` (#cgan-conv-err-compose): `P` is the incoming error
///     matrix composed through the SAME backward column transform as the
///     coefficients, but with `|kernel|`: `P[i,p] = Σ_j err_in[i,j]·|K_{j→p}|`
///     (computed by the caller as one extra f32 conv/GEMM on non-negative data).
///     This is the EXACT first-order bound `|Σ_j (a±e)_j·K_{j→p} − Σ_j a_j·K_{j→p}|
///     ≤ Σ_j e_j·|K_{j→p}|`; the f32 evaluation of `P` itself is inflated by
///     `(1+γ_{n}^f32)` — sound for any summation order because every summand is
///     non-negative (Higham §4.2: relative error of a non-negative sum ≤ γ_n).
///     Non-finite entries (INF-poisoned incoming rows) stay `+INF` (outward).
///   - `prop_exact = None`: the legacy row-constant over-bound
///     `prop[i] = row_max(err_in,i)·‖kernel‖_1`, applied to every column. Sound
///     but catastrophically loose on real conv stacks: `‖kernel‖_1` sums over the
///     WHOLE kernel (all output channels × all taps) while a single input column
///     only ever receives `fan-out ≪ ‖kernel‖_1` of it, so the certified error
///     grows by ~`‖kernel‖_1 / mean-column-L1` (100–1000×) per conv layer and,
///     after the discharge at the next non-carrier layer, dominated the CROWN
///     width on cGAN-class conv/BN stacks (BN_5 2.05×, Conv_19 404× vs exact).
///
/// Two SOUND γ modes, selected by the caller:
///   - `coeff_f64 = Some(f64-recompute)` → `γ = γ_n^f64` and `cast_err =
///     |f64 − stored_f32|` (the stored coefficient is the directed f32 of the f64
///     recompute). Tight; used on wide contractions.
///   - `coeff_f64 = None` → `γ = γ_n^f32` and `cast_err = 0` (the stored
///     coefficient is the f32 GEMM result, whose error `γ_n^f32·S` is itself the
///     bound). Cheap; used on small contractions where this is already tight.
///
/// `in_a` is the incoming coefficient block and `in_err` the incoming certified
/// error, both flattened to `(rows, mid_dim)`.
pub(crate) fn conv_coeff_err_matrix(
    in_a: &Array2<f32>,
    in_err: Option<&Array2<f32>>,
    stored: &Array2<f32>,
    coeff_f64: Option<&Array2<f64>>,
    kernel_l1: f64,
    n_contraction: usize,
    prop_exact: Option<&Array2<f32>>,
    // Per-COLUMN kernel L1 norm (#patches-perchannel-l1). Column `p` of the
    // output block belongs to one input channel, and only that channel's kernel
    // slice can reach it, so the scalar whole-kernel `kernel_l1` over-charges
    // both the intrinsic and the carry by ~in_c_per_group. `None` keeps the
    // scalar norm; any entry is used only when it is smaller (so this can only
    // tighten).
    kernel_l1_per_col: Option<&[f64]>,
) -> Array2<f32> {
    conv_coeff_err_matrix_downgraded(
        in_a,
        in_err,
        stored,
        coeff_f64,
        kernel_l1,
        n_contraction,
        prop_exact,
        kernel_l1_per_col,
        None,
    )
}

/// [`conv_coeff_err_matrix`] with the S2 downgrade-only seam exposed
/// (`docs/CONV_CROWN_WALL_DESIGN_2026-07-27.md` §S2).
///
/// `measured_accum_err[i,p]`, when supplied, is an a-posteriori (EFT-measured)
/// bound on the ACCUMULATION error of the coefficient this call is certifying —
/// the exact quantity the a-priori `γ_n·S` term over-bounds. It is combined
/// through `ny_core::eft::combine_downgrade_only_f64` and therefore:
///
/// * can only ever LOWER that one term (the certified bound can only improve);
/// * is discarded — leaving the a-priori term byte-identical — when it is
///   absent, shape-mismatched, negative, NaN, or infinite;
/// * never touches `cast`, `prop`, `ftz` or `daz`, which bound different error
///   sources (storage rounding, carried error, and subnormal flush) that an
///   accumulation-residual measurement says nothing about.
///
/// CALLER OBLIGATION, and it is the whole soundness argument: the measurement
/// must be the residual of the SAME executed fold that produced `stored`.
/// A residual measured on a differently-ordered reduction bounds a different
/// number. `None` — every call site today — is always safe.
pub(crate) fn conv_coeff_err_matrix_downgraded(
    in_a: &Array2<f32>,
    in_err: Option<&Array2<f32>>,
    stored: &Array2<f32>,
    coeff_f64: Option<&Array2<f64>>,
    kernel_l1: f64,
    n_contraction: usize,
    prop_exact: Option<&Array2<f32>>,
    kernel_l1_per_col: Option<&[f64]>,
    measured_accum_err: Option<&Array2<f32>>,
) -> Array2<f32> {
    conv_coeff_err_matrix_downgraded_with_poll(
        in_a,
        in_err,
        stored,
        coeff_f64,
        kernel_l1,
        n_contraction,
        prop_exact,
        kernel_l1_per_col,
        measured_accum_err,
        &mut || Ok(()),
    )
    .expect("infallible Conv coefficient-error host poll")
}

/// Pollable form of [`conv_coeff_err_matrix`] for finite execution
/// authorities. The returned certificate is bit-identical when the authority
/// remains live.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_coeff_err_matrix_with_poll<F>(
    in_a: &Array2<f32>,
    in_err: Option<&Array2<f32>>,
    stored: &Array2<f32>,
    coeff_f64: Option<&Array2<f64>>,
    kernel_l1: f64,
    n_contraction: usize,
    prop_exact: Option<&Array2<f32>>,
    kernel_l1_per_col: Option<&[f64]>,
    poll: &mut F,
) -> Result<Array2<f32>>
where
    F: FnMut() -> Result<()>,
{
    let _share_timer = ShareTimer::new(&CONV_ERR_NANOS);
    conv_coeff_err_matrix_downgraded_with_poll(
        in_a,
        in_err,
        stored,
        coeff_f64,
        kernel_l1,
        n_contraction,
        prop_exact,
        kernel_l1_per_col,
        None,
        poll,
    )
}

/// Shared implementation of the downgrade-only and finite-authority seams.
/// Keeping the measurement and poll as independent inputs ensures the EFT
/// tightening cannot bypass deadline checks and an absent measurement remains
/// bit-identical to the a-priori certificate.
#[allow(clippy::too_many_arguments)]
fn conv_coeff_err_matrix_downgraded_with_poll<F>(
    in_a: &Array2<f32>,
    in_err: Option<&Array2<f32>>,
    stored: &Array2<f32>,
    coeff_f64: Option<&Array2<f64>>,
    kernel_l1: f64,
    n_contraction: usize,
    prop_exact: Option<&Array2<f32>>,
    kernel_l1_per_col: Option<&[f64]>,
    measured_accum_err: Option<&Array2<f32>>,
    poll: &mut F,
) -> Result<Array2<f32>>
where
    F: FnMut() -> Result<()>,
{
    let nrows = stored.nrows();
    let ncols = stored.ncols();
    let recompute_ok = coeff_f64.is_some_and(|c| c.dim() == (nrows, ncols));
    // f64 sum error when the coefficient is f64-accumulated; otherwise the f32
    // GEMM coefficient's error is the (larger) f32 growth factor. Both sound.
    let gamma = if recompute_ok {
        crate::layers::linear::crown_single_gamma_n_f64(n_contraction)
    } else {
        crate::layers::linear::crown_single_gamma_n_f32(n_contraction)
    };
    // ROW SLICES, not `a[[i, k]]`. The two-dimensional index recomputes
    // `i * stride + k` and bounds-checks BOTH axes on every element; across a
    // 16k-wide row that address arithmetic IS the loop, not the certificate
    // arithmetic. `a.row(i)` yields the same elements in the same order, so the
    // fold — and `add_f64_up`'s order-dependent result — is untouched.
    let row_max =
        |a: &Array2<f32>, i: usize, work_since_poll: &mut usize, poll: &mut F| -> Result<f64> {
            let mut m = 0.0f64;
            let ncols = a.ncols();
            let row = a.row(i);
            let mut k0 = 0usize;
            while k0 < ncols {
                let k1 = k0 + crown_host_poll_block_len(*work_since_poll, ncols - k0);
                for k in k0..k1 {
                    let entry = row[k];
                    let v = f32_to_f64_exact(entry).abs();
                    if !v.is_finite() {
                        return Ok(f64::INFINITY);
                    }
                    if v > m {
                        m = v;
                    }
                }
                poll_crown_host_block(work_since_poll, k1 - k0, poll)?;
                k0 = k1;
            }
            Ok(m)
        };
    // #f32-cert-underflow-floor: row L1 of |A|, the operand-side half of the
    // DAZ (denormals-are-zero) flush floor. `crown_single.rs`'s
    // `daz_operand_flush_floor` charges `(rowL1(A)[i] + colL1(W)[j]) * FLT_MIN`;
    // `kl1` below is already the column L1 of the kernel, so this supplies the
    // missing half. Summed in f64 over f32 magnitudes, so it cannot itself
    // underflow away.
    let row_l1 =
        |a: &Array2<f32>, i: usize, work_since_poll: &mut usize, poll: &mut F| -> Result<f64> {
            let mut s = 0.0f64;
            let ncols = a.ncols();
            let row = a.row(i);
            let mut k0 = 0usize;
            while k0 < ncols {
                let k1 = k0 + crown_host_poll_block_len(*work_since_poll, ncols - k0);
                for k in k0..k1 {
                    let entry = row[k];
                    let value = f32_to_f64_exact(entry).abs();
                    if !value.is_finite() {
                        return Ok(f64::INFINITY);
                    }
                    // Sequential `add_f64_up`, in the original k order. Blocking
                    // must not reassociate this: the outward round makes the sum
                    // order-dependent, so a per-block partial would be a
                    // DIFFERENT certificate, not a faster one.
                    s = add_f64_up(s, value);
                }
                poll_crown_host_block(work_since_poll, k1 - k0, poll)?;
                k0 = k1;
            }
            Ok(s)
        };
    let error_row_max =
        |a: &Array2<f32>, i: usize, work_since_poll: &mut usize, poll: &mut F| -> Result<f64> {
            let mut m = 0.0f64;
            let ncols = a.ncols();
            let row = a.row(i);
            let mut k0 = 0usize;
            while k0 < ncols {
                let k1 = k0 + crown_host_poll_block_len(*work_since_poll, ncols - k0);
                for k in k0..k1 {
                    let entry = row[k];
                    let value = nonnegative_f32_error_or_infinity(entry);
                    if !value.is_finite() {
                        return Ok(f64::INFINITY);
                    }
                    if value > m {
                        m = value;
                    }
                }
                poll_crown_host_block(work_since_poll, k1 - k0, poll)?;
                k0 = k1;
            }
            Ok(m)
        };
    // Exact prop path: inflate the f32-evaluated non-negative composition by
    // (1 + γ_{n+2}^f32) to cover its own accumulation rounding (n products, ≤ n−1
    // adds, +2 headroom for the per-product rounding), and only when the shape
    // matches the output block (defensive: mismatch falls back to the row bound).
    let prop_exact = prop_exact.filter(|p| p.dim() == (nrows, ncols));
    // S2 downgrade-only arm. Shape-mismatch is a refusal, exactly as for
    // `prop_exact`; `None` means "no measurement", which the combinator below
    // renders as the a-priori term unchanged.
    let measured_accum_err = measured_accum_err.filter(|m| m.dim() == (nrows, ncols));
    let prop_inflate =
        1.0 + crate::layers::linear::crown_single_gamma_n_f32(n_contraction.saturating_add(2));
    poll()?;
    let mut err = Array2::<f32>::zeros((nrows, ncols));
    let mut work_since_poll = 0usize;
    poll()?;
    let col_l1 = |p: usize| -> f64 {
        match kernel_l1_per_col {
            Some(v) if v.len() == ncols && v[p].is_finite() && v[p] >= 0.0 => v[p].min(kernel_l1),
            _ => kernel_l1,
        }
    };
    for i in 0..nrows {
        let row_in_max = row_max(in_a, i, &mut work_since_poll, poll)?;
        // Only needed when the f32 certificate is live; skip the extra pass
        // otherwise so the always-on f64 path costs nothing new.
        let row_in_l1 = if recompute_ok {
            0.0
        } else {
            row_l1(in_a, i, &mut work_since_poll, poll)?
        };
        let row_err_max = match in_err {
            Some(error) => error_row_max(error, i, &mut work_since_poll, poll)?,
            None => 0.0,
        };
        for p in 0..ncols {
            let kl1 = col_l1(p);
            // #f32-cert-underflow-floor: `gamma * row_max * kernel_L1` is Higham
            // Thm 3.1, which ASSUMES NO UNDERFLOW. In the subnormal range each
            // rounding contributes an ABSOLUTE eta that no RELATIVE factor can
            // cover, so the relative term alone is NOT a bound there.
            //
            // Adversarial review (2026-07-29) built an exact-rational
            // counterexample on the `coeff_f64 = None` branch: out_c=4, 3x3
            // kernel with all-subnormal weights, n_contraction=36 gives a
            // certified `gamma*S = 1.69e-46` against a measured true error of
            // `4.25e-45` -- a 25x violation, i.e. the certified interval did NOT
            // contain the true coefficient.
            //
            // This repo's own f32-sound primitive already charges the term:
            // `crown_single.rs`'s `aw_f32_sound_bound` uses
            // `gamma*s + ftz + daz`, with `ftz = 4*k*2^-126`. Charge the same
            // underflow floor here. It is a pure WIDENING (added, never
            // subtracted) and it is ~1.7e-36 at k=36, dwarfing the 4.25e-45 the
            // counterexample needed.
            //
            // NOTE this branch is DEAD today -- `conv_should_f64_recompute`
            // returns `true` unconditionally, so `recompute_ok` holds and
            // `gamma` is the f64 factor, for which the f64 `cast_err` term
            // already captures storage rounding exactly. This charge exists so
            // that making that function consult `n_contraction` (the change that
            // would let a 576-row Conv_25 walk finish inside its time budget)
            // does not silently enable an under-charged certificate.
            let (ftz, daz) = if recompute_ok {
                (0.0, 0.0)
            } else {
                (
                    4.0 * (n_contraction as f64) * 2f64.powi(-126),
                    // DAZ operand-flush floor, mirroring
                    // `crown_single.rs::daz_operand_flush_floor`:
                    // `(rowL1(A) + colL1(W)) * FLT_MIN`. Covers a backend that
                    // flushes subnormal OPERANDS to zero, which the relative
                    // Higham factor also cannot see.
                    (row_in_l1 + kl1) * f64::from(f32::MIN_POSITIVE),
                )
            };
            // A-priori Higham accumulation charge, and the S2 seam that may
            // only ever lower it. `measured_accum_err = None` (every call site
            // today) makes `combine_downgrade_only_f64` return `higham_accum`
            // itself, so `s` is bit-identical to the pre-S2 expression.
            let higham_accum = gamma * row_in_max * kl1;
            let measured_accum = measured_accum_err.map_or(f64::INFINITY, |m| {
                nonnegative_f32_error_or_infinity(m[[i, p]])
            });
            let s =
                ny_core::eft::combine_downgrade_only_f64(higham_accum, measured_accum) + ftz + daz;
            let prop_row = if prop_exact.is_some() {
                0.0
            } else {
                row_err_max * kl1
            };
            let cast = coeff_f64.filter(|_| recompute_ok).map_or(0.0, |c| {
                (c[[i, p]] - f32_to_f64_exact(stored[[i, p]])).abs()
            });
            let prop = match prop_exact {
                Some(pe) => {
                    let value = nonnegative_f32_error_or_infinity(pe[[i, p]]);
                    if value.is_finite() {
                        value * prop_inflate
                    } else {
                        f64::INFINITY
                    }
                }
                None => prop_row,
            };
            err[[i, p]] = publish_error_up(cast + s + prop);
        }
        // One offer per output ROW rather than per output element. The per-row
        // budget is the row width, so the accounting is the same total; what
        // goes away is an opaque `?`-returning call in the innermost loop of the
        // publication pass.
        poll_crown_host_block(&mut work_since_poll, ncols, poll)?;
    }
    poll()?;
    Ok(err)
}

/// Batched-conv alias kept for the batched call sites (identical semantics).
pub(crate) use conv_coeff_err_matrix as batched_conv_coeff_err;
pub(crate) use conv_coeff_err_matrix_with_poll as batched_conv_coeff_err_with_poll;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn higham_error_publication_steps_past_binary32_boundary() {
        let boundary = f32_to_f64_exact(1.0);
        assert_eq!(publish_error_up(boundary), next_up_f32(1.0));

        // The subnormal range is first raised to the normal floor, then stepped
        // once more. No published certificate term depends on FTZ state.
        let tiny = f64::from_bits(1);
        assert!(publish_error_up(tiny) > f32::MIN_POSITIVE);
    }

    /// The block budget is the guarantee, so pin it directly: never more than
    /// `CROWN_HOST_POLL_ELEMENTS` accounted units between two polls, and never a
    /// zero-length block (which would spin).
    #[test]
    fn poll_block_length_never_exceeds_the_element_budget() {
        for spent in [
            0,
            1,
            7,
            CROWN_HOST_POLL_ELEMENTS - 1,
            CROWN_HOST_POLL_ELEMENTS,
        ] {
            for remaining in [1, 5, CROWN_HOST_POLL_ELEMENTS * 3 + 7] {
                let len = crown_host_poll_block_len(spent, remaining);
                assert!(len >= 1, "spent={spent} remaining={remaining}");
                assert!(len <= remaining, "spent={spent} remaining={remaining}");
                assert!(
                    spent + len <= CROWN_HOST_POLL_ELEMENTS || len == 1,
                    "a block may not overrun the budget: spent={spent} len={len}"
                );
            }
        }
    }

    /// MOVING THE POLL MUST NOT MOVE THE CERTIFICATE.
    ///
    /// The row reductions now consume a block between polls instead of one
    /// element. That is only legitimate while the ARITHMETIC is untouched, and
    /// the fixture is built to catch the specific way it could be broken: the
    /// row is far wider than one poll block and its magnitudes are chosen so
    /// that `add_f64_up`'s outward rounding is ORDER-DEPENDENT (a huge leading
    /// term that absorbs each following one). Summing per-block partials and
    /// combining them — the obvious "optimization" — lands on a different
    /// number, so this test fails if anyone reassociates the L1 accumulation.
    ///
    /// `coeff_f64 = None` is what selects the f32 arm, which is the arm that
    /// evaluates the row L1 at all.
    #[test]
    fn block_polling_leaves_the_order_sensitive_certificate_bit_identical() {
        let width = CROWN_HOST_POLL_ELEMENTS * 3 + 7;
        let mut input = Array2::<f32>::from_elem((1, width), 1.0);
        input[[0, 0]] = 1.0e16;
        let stored = Array2::<f32>::from_elem((1, width), 0.5);

        // An independent sequential reference for the order-sensitive term.
        let mut reference_l1 = 0.0f64;
        for k in 0..width {
            reference_l1 = add_f64_up(reference_l1, f32_to_f64_exact(input[[0, k]]).abs());
        }
        // Order matters here, and this is the assertion that says so: the
        // block-partial alternative differs.
        let mut blockwise = 0.0f64;
        for chunk_start in (0..width).step_by(CROWN_HOST_POLL_ELEMENTS) {
            let mut partial = 0.0f64;
            for k in chunk_start..(chunk_start + CROWN_HOST_POLL_ELEMENTS).min(width) {
                partial = add_f64_up(partial, f32_to_f64_exact(input[[0, k]]).abs());
            }
            blockwise = add_f64_up(blockwise, partial);
        }
        assert_ne!(
            reference_l1.to_bits(),
            blockwise.to_bits(),
            "fixture is not order-sensitive, so it cannot police reassociation"
        );

        let mut polls = 0usize;
        let polled = conv_coeff_err_matrix_with_poll(
            &input,
            None,
            &stored,
            None,
            1.0,
            width,
            None,
            None,
            &mut || {
                polls += 1;
                Ok(())
            },
        )
        .expect("live authority must publish");
        let unpolled = conv_coeff_err_matrix(&input, None, &stored, None, 1.0, width, None, None);

        assert_eq!(polled.dim(), unpolled.dim());
        for (a, b) in polled.iter().zip(unpolled.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "the pollable and no-op-poll certificates must agree bit for bit"
            );
        }
        // And the polls did not simply disappear: several blocks of work were
        // accounted, so the caller kept its chance to interrupt.
        assert!(
            polls >= 3,
            "block polling must still offer the poll: {polls}"
        );
    }

    #[test]
    fn pollable_conv_certificate_helpers_preserve_terminal_authority() {
        let width = CROWN_HOST_POLL_ELEMENTS * 2;
        let input = Array2::<f32>::ones((1, width));
        let stored = Array2::<f32>::zeros((1, width));
        let recomputed = Array2::<f64>::zeros((1, width));
        let mut polls = 0usize;
        let error = conv_coeff_err_matrix_with_poll(
            &input,
            None,
            &stored,
            Some(&recomputed),
            1.0,
            1,
            None,
            None,
            &mut || {
                polls += 1;
                if polls >= 3 {
                    Err(NyError::DeadlineExceeded(
                        "injected post-f64 certificate expiry".into(),
                    ))
                } else {
                    Ok(())
                }
            },
        )
        .expect_err("coefficient-error construction must preserve terminal expiry");
        assert!(error.is_deadline_exceeded(), "wrong error: {error}");

        // A zero-valued bias skips all spatial arithmetic. The outer row/channel
        // traversal must still poll instead of becoming an unbounded fast path.
        let channels = CROWN_HOST_POLL_ELEMENTS * 2;
        let coefficients = Array2::<f32>::zeros((1, channels));
        let bias = Array1::<f32>::zeros(channels);
        let constant = Array1::<f32>::zeros(1);
        let mut polls = 0usize;
        let error = compute_conv_bias_rows_f64_with_poll(
            coefficients.view(),
            None,
            constant.view(),
            coefficients.view(),
            None,
            constant.view(),
            &bias,
            channels,
            1,
            &mut || {
                polls += 1;
                if polls >= 2 {
                    Err(NyError::DeadlineExceeded(
                        "injected zero-bias traversal expiry".into(),
                    ))
                } else {
                    Ok(())
                }
            },
        )
        .expect_err("zero-bias traversal must preserve terminal expiry");
        assert!(error.is_deadline_exceeded(), "wrong error: {error}");
    }

    #[test]
    fn conv_bias_folds_incoming_coefficient_error_outward() {
        let mut bounds = LinearBounds::new(
            Array2::zeros((1, 1)),
            array![0.0],
            Array2::zeros((1, 1)),
            array![0.0],
        )
        .expect("valid bounds");
        bounds.set_coeff_err(Array2::ones((1, 1)), Array2::ones((1, 1)));

        let (lower, upper) =
            compute_conv_bias_f64(&bounds, Some(&array![1.0]), 1, 1).expect("bias fold");

        assert!(
            lower[0] <= -1.0,
            "lower bias must include -A_err*|bias|, got {}",
            lower[0]
        );
        assert!(
            upper[0] >= 1.0,
            "upper bias must include +A_err*|bias|, got {}",
            upper[0]
        );
    }

    #[test]
    fn conv_bias_error_fold_respects_channels_and_zero_bias() {
        let lower_a = array![[1.0, -2.0, 3.0, 4.0]];
        let upper_a = lower_a.clone();
        let lower_err = array![[0.5, 1.0, f32::INFINITY, -1.0]];
        let upper_err = lower_err.clone();
        let zero_b = array![0.0];
        let bias = array![2.0, 0.0];

        let (lower, upper) = compute_conv_bias_rows_f64(
            lower_a.view(),
            Some(lower_err.view()),
            zero_b.view(),
            upper_a.view(),
            Some(upper_err.view()),
            zero_b.view(),
            &bias,
            2,
            2,
        )
        .expect("bias fold");

        // Nominal: (1 - 2) * 2 = -2. Error: (0.5 + 1) * 2 = 3.
        // Invalid errors in the zero-bias channel contribute exactly zero.
        assert!(lower[0] <= -5.0, "lower={}", lower[0]);
        assert!(upper[0] >= 1.0, "upper={}", upper[0]);
        assert!(lower[0].is_finite() && upper[0].is_finite());
    }

    #[test]
    fn conv_bias_rejects_negative_subnormal_error_by_bits() {
        let invalid = f32::from_bits(0x8000_0001);
        let zero = array![0.0];
        let (lower, upper) = compute_conv_bias_rows_f64(
            array![[0.0]].view(),
            Some(array![[invalid]].view()),
            zero.view(),
            array![[0.0]].view(),
            Some(array![[invalid]].view()),
            zero.view(),
            &array![1.0],
            1,
            1,
        )
        .expect("invalid metadata must poison rather than fail to decode");

        assert_eq!(lower[0], f32::NEG_INFINITY);
        assert_eq!(upper[0], f32::INFINITY);
    }

    #[test]
    fn conv_coefficient_error_validation_is_daz_independent() {
        let invalid = f32::from_bits(0x8000_0001);
        let input = array![[0.0]];
        let stored = array![[0.0]];
        let reference = Array2::<f64>::zeros((1, 1));

        let carried = conv_coeff_err_matrix(
            &input,
            Some(&array![[invalid]]),
            &stored,
            Some(&reference),
            1.0,
            1,
            None,
            None,
        );
        assert_eq!(carried[[0, 0]], f32::INFINITY);

        let exact_path = conv_coeff_err_matrix(
            &input,
            None,
            &stored,
            Some(&reference),
            1.0,
            1,
            Some(&array![[invalid]]),
            None,
        );
        assert_eq!(exact_path[[0, 0]], f32::INFINITY);
    }

    // -----------------------------------------------------------------------
    // S2 downgrade-only seam (docs/CONV_CROWN_WALL_DESIGN_2026-07-27.md §S2)
    // -----------------------------------------------------------------------

    /// A fixture whose a-priori accumulation charge is the dominant term, so a
    /// change to it is visible in the published certificate. `coeff_f64 = None`
    /// selects the `γ_n^f32` branch, where that charge is ~2^29× larger than on
    /// the f64-recompute branch (and where a measured channel would pay).
    fn seam_fixture() -> (Array2<f32>, Array2<f32>) {
        (array![[1.0f32, -1.0, 1.0, -1.0]], array![[0.0f32]])
    }

    /// The whole safety argument, mechanically: no measurement can WIDEN the
    /// published certificate, and every refusal shape leaves it BIT-identical
    /// to the a-priori channel.
    #[test]
    fn measured_accum_arm_never_widens_and_refuses_bit_identically() {
        let (in_a, stored) = seam_fixture();
        let baseline = conv_coeff_err_matrix(&in_a, None, &stored, None, 4.0, 4, None, None);
        assert!(
            baseline[[0, 0]] > 0.0 && baseline[[0, 0]].is_finite(),
            "fixture must publish a live a-priori charge, got {}",
            baseline[[0, 0]]
        );

        // Every invalid / absent measurement leaves the incumbent untouched.
        for refusal in [
            None,
            Some(array![[f32::NAN]]),
            Some(array![[f32::INFINITY]]),
            Some(array![[-1.0f32]]),
            // Shape mismatch: a measurement for a different block must not be
            // silently indexed into this one.
            Some(array![[0.0f32, 0.0]]),
        ] {
            let out = conv_coeff_err_matrix_downgraded(
                &in_a,
                None,
                &stored,
                None,
                4.0,
                4,
                None,
                None,
                refusal.as_ref(),
            );
            assert_eq!(
                out[[0, 0]].to_bits(),
                baseline[[0, 0]].to_bits(),
                "a refused measurement must leave the a-priori charge \
                 bit-identical (refusal = {refusal:?})"
            );
        }

        // A LOOSER measurement must lose to the a-priori charge — the case the
        // brief calls out explicitly.
        let looser = array![[1.0f32]];
        let out = conv_coeff_err_matrix_downgraded(
            &in_a,
            None,
            &stored,
            None,
            4.0,
            4,
            None,
            None,
            Some(&looser),
        );
        assert_eq!(
            out[[0, 0]].to_bits(),
            baseline[[0, 0]].to_bits(),
            "Higham must win when the measured arm is looser"
        );
    }

    /// ...and a TIGHTER measurement is actually admitted, or the seam is a
    /// decoration that could never deliver S2.
    #[test]
    fn measured_accum_arm_tightens_only_the_accumulation_term() {
        let (in_a, stored) = seam_fixture();
        let baseline = conv_coeff_err_matrix(&in_a, None, &stored, None, 4.0, 4, None, None);
        let tighter = Array2::<f32>::zeros((1, 1));
        let out = conv_coeff_err_matrix_downgraded(
            &in_a,
            None,
            &stored,
            None,
            4.0,
            4,
            None,
            None,
            Some(&tighter),
        );
        assert!(
            out[[0, 0]] < baseline[[0, 0]],
            "a zero-residual measurement must tighten: {} vs {}",
            out[[0, 0]],
            baseline[[0, 0]]
        );
        // The ftz/daz underflow floors survive: zeroing the accumulation term
        // must NOT zero the published certificate, because those floors bound a
        // different (absolute, subnormal) error source.
        assert!(
            out[[0, 0]] > 0.0,
            "the subnormal-flush floors were dropped along with the a-priori \
             accumulation term"
        );
    }
}
