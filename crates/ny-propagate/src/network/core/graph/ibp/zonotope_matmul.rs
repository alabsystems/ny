// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::MatMulLayer;
use ny_core::Result;
use ny_tensor::ZonotopeTensor;
use tracing::debug;

pub(super) fn propagate_disjoint_matmul(
    matmul: &MatMulLayer,
    lhs: &ZonotopeTensor,
    rhs: &ZonotopeTensor,
) -> Result<ZonotopeTensor> {
    let candidate = lhs.matmul_disjoint(rhs)?;
    let candidate = apply_scale_if_present(candidate, matmul.scale)?;
    let candidate_bounds = candidate.to_bounded_tensor()?;

    let lhs_bounds = lhs.to_bounded_tensor()?;
    let rhs_bounds = rhs.to_bounded_tensor()?;
    let fallback_bounds = matmul.propagate_ibp_binary(&lhs_bounds, &rhs_bounds)?;
    if candidate_bounds.max_width() <= fallback_bounds.max_width() {
        return Ok(candidate);
    }

    debug!(
        "MatMul zonotope: disjoint-symbol path wider than IBP fallback ({} > {}), keeping interval result",
        candidate_bounds.max_width(),
        fallback_bounds.max_width()
    );
    Ok(ZonotopeTensor::from_bounded_tensor(&fallback_bounds))
}

fn apply_scale_if_present(zonotope: ZonotopeTensor, scale: Option<f32>) -> Result<ZonotopeTensor> {
    let Some(scale) = scale else {
        return Ok(zonotope);
    };
    let scale_tensor = ndarray::ArrayD::from_elem(zonotope.shape().to_vec(), scale);
    zonotope.mul_constant(&scale_tensor)
}
