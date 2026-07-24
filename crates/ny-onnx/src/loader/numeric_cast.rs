// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use tracing::warn;

/// Maximum integer magnitude exactly representable as f32.
///
/// f32 has a 24-bit mantissa, so integers with |value| > 2^24 lose precision.
/// Values beyond this threshold silently round to the nearest representable
/// float, which is incorrect for shape indices and sentinel values like
/// `i64::MAX`.
const F32_INT_EXACT_LIMIT: i64 = 1 << 24; // 16_777_216

/// Convert i64 to f32 with precision-loss warning.
///
/// ONNX commonly stores shape indices and sentinel values (e.g., `i64::MAX` for
/// Slice end) as INT64. Plain `as f32` silently loses precision for
/// |value| > 16_777_216 (2^24, the f32 mantissa limit). This function logs
/// a warning on precision loss so model loading issues are diagnosable.
///
/// # SAFETY(as f32): Guarded — warns on values outside f32 exact-integer range.
pub(crate) fn i64_to_f32_warned(value: i64, context: &str) -> f32 {
    if value.unsigned_abs() > F32_INT_EXACT_LIMIT as u64 {
        warn!(
            "i64→f32 precision loss: {value} exceeds f32 exact-integer range ±{F32_INT_EXACT_LIMIT} \
             (context: {context})"
        );
    }
    value as f32
}

/// Convert i64 to f32 without permitting precision loss.
///
/// Integer-valued shape and index tensors must stay exact. Callers that can
/// route the original i64 payload through `WeightStore::integers` should do so;
/// this helper exists for the remaining float-only boundaries and fails closed
/// when the cast would round.
pub(crate) fn i64_to_f32_checked(value: i64, context: &str) -> Result<f32> {
    if value.unsigned_abs() > F32_INT_EXACT_LIMIT as u64 {
        return Err(NyError::ModelLoad(format!(
            "i64→f32 precision loss: {value} exceeds f32 exact-integer range ±{F32_INT_EXACT_LIMIT} \
             (context: {context})"
        )));
    }
    Ok(value as f32)
}

/// Convert i32 to f32 with precision-loss warning.
///
/// i32 values with |value| > 2^24 (16_777_216) lose precision when cast to f32.
/// This covers ~74% of i32's range (2^24..2^31), so the warning fires only for
/// large constants like tensor shape values or padding amounts.
///
/// # SAFETY(as f32): Guarded — warns on values outside f32 exact-integer range.
pub(crate) fn i32_to_f32_warned(value: i32, context: &str) -> f32 {
    if (value as i64).unsigned_abs() > F32_INT_EXACT_LIMIT as u64 {
        warn!(
            "i32→f32 precision loss: {value} exceeds f32 exact-integer range ±{F32_INT_EXACT_LIMIT} \
             (context: {context})"
        );
    }
    value as f32
}

/// Convert f64 to f32 with an explicit out-of-range check.
///
/// Returns `(converted, loses_precision)`. Callers should aggregate any
/// `loses_precision` observations into a single warning instead of logging once
/// per element.
pub(crate) fn f64_to_f32_checked(value: f64, context: &str) -> Result<(f32, bool)> {
    if value.is_finite() && value.abs() > f32::MAX as f64 {
        return Err(NyError::ModelLoad(format!(
            "f64→f32 out of range: {value} exceeds f32 finite range ±{} (context: {context})",
            f32::MAX
        )));
    }

    let converted = value as f32;
    let loses_precision = value.is_finite() && (converted as f64) != value;
    Ok((converted, loses_precision))
}
