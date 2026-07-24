// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batch layout and tensor-shaping helpers for batched graph PGD.

use anyhow::{anyhow, Result};
use ndarray::{ArrayBase, ArrayD, Axis, Data, IxDyn, Slice};
use ny_tensor::BoundedTensor;

#[derive(Clone, Copy)]
pub(super) struct SimpleRng(u64);

impl SimpleRng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 & 0xFFFF_FFFF) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub(super) fn next_bool(&mut self) -> bool {
        self.next_u32() & 1 == 0
    }
}

#[derive(Clone, Copy)]
pub(super) enum RestartBatchLayout {
    PrependAxis,
    FoldLeadingAxis { chunk: usize },
}

impl RestartBatchLayout {
    pub(super) fn batch_shape(
        self,
        input_shape: &[usize],
        num_restarts: usize,
    ) -> Result<Vec<usize>> {
        match self {
            Self::PrependAxis => {
                let mut shape = Vec::with_capacity(input_shape.len() + 1);
                shape.push(num_restarts);
                shape.extend_from_slice(input_shape);
                Ok(shape)
            }
            Self::FoldLeadingAxis { chunk } => {
                let mut shape = input_shape.to_vec();
                shape[0] = chunk.checked_mul(num_restarts).ok_or_else(|| {
                    anyhow!("graph PGD batch size overflow: {chunk} * {num_restarts}")
                })?;
                Ok(shape)
            }
        }
    }

    pub(super) fn leading_extent(self, num_restarts: usize) -> Result<usize> {
        match self {
            Self::PrependAxis => Ok(num_restarts),
            Self::FoldLeadingAxis { chunk } => chunk
                .checked_mul(num_restarts)
                .ok_or_else(|| anyhow!("graph PGD split index overflow: {chunk} * {num_restarts}")),
        }
    }

    pub(super) fn axis_range(self, batch_idx: usize) -> std::ops::Range<usize> {
        match self {
            Self::PrependAxis => batch_idx..batch_idx + 1,
            Self::FoldLeadingAxis { chunk } => {
                let start = batch_idx * chunk;
                start..start + chunk
            }
        }
    }
}

pub(super) fn batched_item<S>(
    batch: &ArrayBase<S, IxDyn>,
    batch_idx: usize,
    layout: RestartBatchLayout,
) -> ArrayD<f32>
where
    S: Data<Elem = f32>,
{
    match layout {
        RestartBatchLayout::PrependAxis => {
            batch.index_axis(Axis(0), batch_idx).to_owned().into_dyn()
        }
        RestartBatchLayout::FoldLeadingAxis { .. } => batch
            .slice_axis(Axis(0), Slice::from(layout.axis_range(batch_idx)))
            .to_owned()
            .into_dyn(),
    }
}

pub(super) fn assign_batched_item(
    batch: &mut ArrayD<f32>,
    batch_idx: usize,
    layout: RestartBatchLayout,
    item: &ArrayD<f32>,
) {
    match layout {
        RestartBatchLayout::PrependAxis => {
            batch.index_axis_mut(Axis(0), batch_idx).assign(item);
        }
        RestartBatchLayout::FoldLeadingAxis { .. } => {
            batch
                .slice_axis_mut(Axis(0), Slice::from(layout.axis_range(batch_idx)))
                .assign(item);
        }
    }
}

fn fill_uniform_restart(
    batch: &mut ArrayD<f32>,
    batch_idx: usize,
    layout: RestartBatchLayout,
    lower: &ArrayD<f32>,
    upper: &ArrayD<f32>,
    rng: &mut SimpleRng,
) {
    match layout {
        RestartBatchLayout::PrependAxis => {
            for (p, (lo, hi)) in batch
                .index_axis_mut(Axis(0), batch_idx)
                .iter_mut()
                .zip(lower.iter().zip(upper.iter()))
            {
                *p = lo + rng.next_f32() * (hi - lo);
            }
        }
        RestartBatchLayout::FoldLeadingAxis { .. } => {
            for (p, (lo, hi)) in batch
                .slice_axis_mut(Axis(0), Slice::from(layout.axis_range(batch_idx)))
                .iter_mut()
                .zip(lower.iter().zip(upper.iter()))
            {
                *p = lo + rng.next_f32() * (hi - lo);
            }
        }
    }
}

pub(super) fn fill_restart_perturbation(
    perturbation_batch: &mut ArrayD<f32>,
    batch_idx: usize,
    layout: RestartBatchLayout,
    rng: &mut SimpleRng,
) {
    match layout {
        RestartBatchLayout::PrependAxis => {
            for value in perturbation_batch
                .index_axis_mut(Axis(0), batch_idx)
                .iter_mut()
            {
                *value = if rng.next_bool() { 1.0 } else { -1.0 };
            }
        }
        RestartBatchLayout::FoldLeadingAxis { .. } => {
            for value in perturbation_batch
                .slice_axis_mut(Axis(0), Slice::from(layout.axis_range(batch_idx)))
                .iter_mut()
            {
                *value = if rng.next_bool() { 1.0 } else { -1.0 };
            }
        }
    }
}

pub(super) fn sample_uniform_batch(
    input: &BoundedTensor,
    rngs: &mut [SimpleRng],
    layout: RestartBatchLayout,
) -> Result<ArrayD<f32>> {
    let lower = input.lower();
    let upper = input.upper();
    let mut batch = ArrayD::zeros(IxDyn(&layout.batch_shape(lower.shape(), rngs.len())?));
    for (batch_idx, rng) in rngs.iter_mut().enumerate() {
        fill_uniform_restart(&mut batch, batch_idx, layout, lower, upper, rng);
    }
    Ok(batch)
}

pub(super) fn sample_uniform_point(input: &BoundedTensor, rng: &mut SimpleRng) -> ArrayD<f32> {
    let lower = input.lower();
    let upper = input.upper();
    let mut point = ArrayD::zeros(IxDyn(lower.shape()));
    for (p, (lo, hi)) in point.iter_mut().zip(lower.iter().zip(upper.iter())) {
        *p = lo + rng.next_f32() * (hi - lo);
    }
    point
}
