// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Lowering for frozen constants the const-folder could only enclose.
//!
//! An interval-valued constant is INVISIBLE to `WeightStore::get`, so every
//! converter that has not been taught about it simply does not find the
//! constant and takes its existing path — which, for a binary op with one
//! missing operand, ends in `UnsupportedOp` and a sound unbounded skip. The two
//! shapes below are the explicit, audited opt-ins.
//!
//! Nothing here ever produces a POINT layer from an interval. The midpoint is
//! only ever handed to `IntervalConstLayer`, which carries the half-width with
//! it and discharges it in every bound it computes.

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::sub_up_f32;

/// Convert a rigorous `[lower, upper]` enclosure into the `center +- radius`
/// form the propagation layer consumes.
///
/// The conversion must not shrink the set: `center - radius <= lower` and
/// `center + radius >= upper` must hold in exact arithmetic. The midpoint is
/// computed in f64 and each half-width is rounded UP (`sub_up_f32`), so the
/// stored radius is at least the true distance to each endpoint.
///
/// Returns `Err(UnsupportedOp)` when the enclosure is degenerate (zero width —
/// that is the exact case and must be lowered as a point layer) or when the
/// midpoint is not finite.
pub(crate) fn center_and_radius(
    lower: &ArrayD<f32>,
    upper: &ArrayD<f32>,
    context: &str,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    if lower.shape() != upper.shape() {
        return Err(NyError::ShapeMismatch {
            expected: lower.shape().to_vec(),
            got: upper.shape().to_vec(),
        });
    }
    let mut center = ArrayD::<f32>::zeros(lower.raw_dim());
    let mut radius = ArrayD::<f32>::zeros(lower.raw_dim());
    let mut any_width = false;
    for ((c, r), (&l, &u)) in center
        .iter_mut()
        .zip(radius.iter_mut())
        .zip(lower.iter().zip(upper.iter()))
    {
        if !l.is_finite() || !u.is_finite() || l > u {
            return Err(NyError::UnsupportedOp(format!(
                "{context}: interval constant endpoint is not a finite ordered pair"
            )));
        }
        let mid = ((f64::from(l) + f64::from(u)) * 0.5) as f32;
        if !mid.is_finite() {
            return Err(NyError::UnsupportedOp(format!(
                "{context}: interval constant midpoint overflows binary32"
            )));
        }
        // Round each half-width UP, then take the larger: the stored radius is
        // >= max(center - l, u - center) in exact arithmetic, so the carried set
        // still contains [l, u].
        let below = sub_up_f32(mid, l);
        let above = sub_up_f32(u, mid);
        let half = below.max(above).max(0.0);
        if !half.is_finite() {
            return Err(NyError::UnsupportedOp(format!(
                "{context}: interval constant half-width overflows binary32"
            )));
        }
        if half > 0.0 {
            any_width = true;
        }
        *c = mid;
        *r = half;
    }
    if !any_width {
        return Err(NyError::UnsupportedOp(format!(
            "{context}: interval constant has zero width — lower it as a point constant"
        )));
    }
    Ok((center, radius))
}

#[cfg(test)]
mod tests {
    use super::center_and_radius;
    use ndarray::{ArrayD, IxDyn};

    fn arr(values: &[f32]) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape")
    }

    #[test]
    fn center_plus_radius_still_contains_the_endpoints() {
        let lower = arr(&[-1.0e-7, 3.0, -5.5]);
        let upper = arr(&[2.0e-7, 3.000_001, -5.499_999]);
        let (center, radius) = center_and_radius(&lower, &upper, "test").expect("carrier");
        for i in 0..3 {
            assert!(
                f64::from(center[i]) - f64::from(radius[i]) <= f64::from(lower[i]),
                "center-radius must not rise above the lower endpoint at {i}"
            );
            assert!(
                f64::from(center[i]) + f64::from(radius[i]) >= f64::from(upper[i]),
                "center+radius must not fall below the upper endpoint at {i}"
            );
        }
    }

    #[test]
    fn zero_width_is_refused_rather_than_carried() {
        let point = arr(&[1.0, 2.0]);
        assert!(center_and_radius(&point, &point, "test").is_err());
    }
}
