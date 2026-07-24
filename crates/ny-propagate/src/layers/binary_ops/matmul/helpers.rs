// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, ArrayView2, Axis, Ix2};
use ny_tensor::BoundedTensor;

use super::{NyError, Result};

pub(super) fn is_perturbed(bounds: &BoundedTensor) -> bool {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .any(|(&l, &u)| l != u)
}

pub(super) fn bounds_all_finite(bounds: &BoundedTensor) -> bool {
    bounds.lower().iter().all(|v| v.is_finite()) && bounds.upper().iter().all(|v| v.is_finite())
}

pub(super) fn view_batch_2d<'a>(
    bounds: &'a BoundedTensor,
    batch_indices: &[usize],
    context: &str,
) -> Result<(ArrayView2<'a, f32>, ArrayView2<'a, f32>)> {
    let mut lower_view = bounds.lower().view();
    let mut upper_view = bounds.upper().view();
    for &idx in batch_indices {
        lower_view = lower_view.index_axis_move(Axis(0), idx);
        upper_view = upper_view.index_axis_move(Axis(0), idx);
    }
    let lower_2d = lower_view
        .into_dimensionality::<Ix2>()
        .map_err(|_| NyError::InvalidSpec(format!("{}: lower not 2D", context)))?;
    let upper_2d = upper_view
        .into_dimensionality::<Ix2>()
        .map_err(|_| NyError::InvalidSpec(format!("{}: upper not 2D", context)))?;
    Ok((lower_2d, upper_2d))
}

/// Apply optional scaling to interval bound arrays.
///
/// NaN/Inf repair is NOT done here — it happens at the BoundedTensor type boundary
/// via `new_repaired(Conservative)` when the caller constructs the final tensor (#3423).
pub(super) fn apply_scale(lower: &mut Array2<f32>, upper: &mut Array2<f32>, scale: Option<f32>) {
    if let Some(scale) = scale {
        if scale >= 0.0 {
            lower.mapv_inplace(|v| v * scale);
            upper.mapv_inplace(|v| v * scale);
        } else {
            let lower_scaled = upper.mapv(|v| v * scale);
            let upper_scaled = lower.mapv(|v| v * scale);
            *lower = lower_scaled;
            *upper = upper_scaled;
        }
    }
}
