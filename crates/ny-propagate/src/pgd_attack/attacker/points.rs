// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Point projection and sampling helpers for PGD attack.

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::RngExt;

use super::PgdAttacker;

impl PgdAttacker<'_> {
    /// Project point back to input bounds. NaN replaced with lower bound (#2721)
    /// since `f32::clamp` returns NaN for NaN input, which would panic in
    /// `BoundedTensor::concrete`.
    pub(crate) fn project(&self, x: &ArrayD<f32>, bounds: &BoundedTensor) -> ArrayD<f32> {
        let lower = bounds.lower();
        let upper = bounds.upper();

        // Element-wise clipping with NaN replacement
        let mut result = x.clone();
        for (val, (l, u)) in result.iter_mut().zip(lower.iter().zip(upper.iter())) {
            if val.is_nan() {
                *val = *l;
            } else {
                *val = val.clamp(*l, *u);
            }
        }
        result
    }

    /// Project a batch of points back to input bounds.
    pub(super) fn project_batch(
        &self,
        x_batch: &ArrayD<f32>,
        bounds: &BoundedTensor,
    ) -> Result<ArrayD<f32>> {
        let Some((&batch_size, sample_shape)) = x_batch.shape().split_first() else {
            return Err(NyError::InvalidSpec(
                "batched SPSA requires at least one batch dimension".to_string(),
            ));
        };
        if sample_shape != bounds.shape() {
            return Err(NyError::InvalidSpec(format!(
                "batched SPSA shape mismatch: batch sample shape {:?} != bounds shape {:?}",
                sample_shape,
                bounds.shape(),
            )));
        }

        let mut projected = x_batch.clone();
        for batch_idx in 0..batch_size {
            let sample = x_batch.index_axis(Axis(0), batch_idx).to_owned();
            let clipped = self.project(&sample, bounds);
            projected
                .index_axis_mut(Axis(0), batch_idx)
                .assign(&clipped);
        }
        Ok(projected)
    }

    /// Sample a random point within input bounds.
    pub(crate) fn sample_uniform(&self, bounds: &BoundedTensor, rng: &mut StdRng) -> ArrayD<f32> {
        let lower = bounds.lower();
        let upper = bounds.upper();

        let mut result = ArrayD::zeros(IxDyn(bounds.shape()));
        for (val, (l, u)) in result.iter_mut().zip(lower.iter().zip(upper.iter())) {
            *val = rng.random_range(*l..=*u);
        }
        result
    }
}
