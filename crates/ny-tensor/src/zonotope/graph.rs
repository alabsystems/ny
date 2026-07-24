// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use std::borrow::Cow;

use super::ZonotopeTensor;

/// Additional operations for GraphNetwork integration.
impl ZonotopeTensor {
    fn broadcast_constant<'a>(&self, constant: &'a ArrayD<f32>) -> Result<Cow<'a, ArrayD<f32>>> {
        if constant.shape() == self.element_shape.as_slice() {
            return Ok(Cow::Borrowed(constant));
        }

        let reshaped = if constant.ndim() == 1
            && self.element_shape.len() == 3
            && constant.shape()[0] == self.element_shape[0]
        {
            Cow::Owned(
                constant
                    .clone()
                    .into_shape_with_order(IxDyn(&[constant.shape()[0], 1, 1]))
                    .map_err(|_| NyError::ShapeMismatch {
                        expected: vec![constant.shape()[0], 1, 1],
                        got: constant.shape().to_vec(),
                    })?,
            )
        } else {
            Cow::Borrowed(constant)
        };

        let broadcasted = reshaped
            .broadcast(IxDyn(&self.element_shape))
            .ok_or_else(|| {
                NyError::shape_mismatch(self.element_shape.clone(), constant.shape().to_vec())
            })?
            .to_owned();
        Ok(Cow::Owned(broadcasted))
    }

    /// Element-wise addition by a constant tensor.
    ///
    /// Each element of the zonotope is shifted by the corresponding constant.
    /// z_i + c_i = (center_i + c_i) + Σⱼ (coeffⱼ,ᵢ · eⱼ)
    pub fn add_constant(&self, constant: &ArrayD<f32>) -> Result<Self> {
        let broadcasted = self.broadcast_constant(constant)?;
        let mut result = self.clone();
        // Add constant only to the center (index 0)
        let mut center = result.coeffs.index_axis_mut(Axis(0), 0);
        center += &*broadcasted;
        Ok(result)
    }

    /// Element-wise multiplication by a constant tensor.
    ///
    /// z_i * c_i = (center_i * c_i) + Σⱼ (coeffⱼ,ᵢ * c_i · eⱼ)
    pub fn mul_constant(&self, constant: &ArrayD<f32>) -> Result<Self> {
        let broadcasted = self.broadcast_constant(constant)?;
        let mut result = self.clone();
        // Multiply all coefficients (center and error terms) by the constant
        for i in 0..=self.n_error_terms {
            let mut row = result.coeffs.index_axis_mut(Axis(0), i);
            row *= &*broadcasted;
        }
        Ok(result)
    }
}
