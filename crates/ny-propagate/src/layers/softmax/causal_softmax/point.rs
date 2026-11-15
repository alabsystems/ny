// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Noncertifying concrete-point causal softmax. Platform binary64 libm and
//! nearest binary32 publication are diagnostic arithmetic, not a real-region
//! enclosure or a deployed-device execution contract. Graph and sequential
//! dispatchers may use this only on their non-sound concrete-point branches.

use super::CausalSoftmaxLayer;
use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, f32_to_f64_exact, NyError, Result};
use ny_tensor::BoundedTensor;
use std::{borrow::Cow, time::Instant};

const MAX_OUTPUT_BYTES: usize = 256 * 1024 * 1024;
const MAX_SCRATCH_BYTES: usize = 512 * 1024 * 1024;

struct PointPlan {
    slices: usize,
    queries: usize,
    keys: usize,
    elements: usize,
    width: usize,
    work: u64,
}

impl PointPlan {
    fn new(shape: &[usize], window: Option<usize>) -> Result<Self> {
        let overflow = || NyError::UnsupportedOp("causal point admission overflow".into());
        if !(2..=4).contains(&shape.len()) {
            return Err(NyError::InvalidSpec(
                "causal point requires rank2..4".into(),
            ));
        }
        let (queries, keys) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        if queries > keys {
            return Err(NyError::InvalidSpec(
                "causal point requires seq_q<=seq_k".into(),
            ));
        }
        let elements = checked_shape_product(shape).ok_or_else(overflow)?;
        let slices = checked_shape_product(&shape[..shape.len() - 2]).ok_or_else(overflow)?;
        let output_bytes = elements.checked_mul(8).ok_or_else(overflow)?;
        let width = window.map_or(queries, |w| queries.min(w.saturating_add(1)));
        // Dense point endpoints, a possible noncontiguous-input copy, row
        // exponent scratch and metadata coexist. The enclosing point ledger
        // prices/owns these buffers; no separate ledger is created here.
        let scratch = elements
            .checked_mul(12)
            .and_then(|n| width.checked_mul(8).and_then(|row| n.checked_add(row)))
            .and_then(|n| n.checked_add(16 * 1024))
            .ok_or_else(overflow)?;
        if output_bytes > MAX_OUTPUT_BYTES || scratch > MAX_SCRATCH_BYTES {
            return Err(NyError::UnsupportedOp(
                "causal point memory admission".into(),
            ));
        }
        let pairs = width
            .checked_mul(width.checked_add(1).ok_or_else(overflow)?)
            .map(|n| n / 2)
            .and_then(|n| {
                (queries - width)
                    .checked_mul(width)
                    .and_then(|tail| n.checked_add(tail))
            })
            .and_then(|n| n.checked_mul(slices))
            .ok_or_else(overflow)?;
        let work = (pairs as u64)
            .checked_mul(64)
            .and_then(|n| {
                (elements as u64)
                    .checked_mul(32)
                    .and_then(|scan| n.checked_add(scan))
            })
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(overflow)?;
        Ok(Self {
            slices,
            queries,
            keys,
            elements,
            width,
            work,
        })
    }
}

fn checkpoint(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        Err(NyError::DeadlineExceeded("concrete causal softmax".into()))
    } else {
        Ok(())
    }
}

impl CausalSoftmaxLayer {
    /// Conservative O(visible-pairs) work for this concrete profile, including
    /// all dense input scans and output validation/copies. Used by the original
    /// managed decoder point lease; it grants no new work or deadline.
    pub fn point_work_bound(shape: &[usize], window: Option<usize>) -> Result<u64> {
        Ok(PointPlan::new(shape, window)?.work)
    }

    pub(crate) fn propagate_concrete_point_with_deadline(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let deadline = match (deadline, ny_core::instance_budget::deadline()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.point_with_poll(input, &mut || checkpoint(deadline))
    }

    fn point_with_poll(
        &self,
        input: &BoundedTensor,
        poll: &mut impl FnMut() -> Result<()>,
    ) -> Result<BoundedTensor> {
        poll()?;
        let plan = PointPlan::new(input.shape(), self.window_size)?;
        crate::layers::common::resolve_axis_i32(self.axis, input.shape().len(), "CausalSoftmax")?;
        // Do not silently turn an interval into a diagnostic point. The graph
        // point dispatchers already collapse their inputs before this call.
        for (i, (&lower, &upper)) in input.lower().iter().zip(input.upper()).enumerate() {
            if i % 1024 == 0 {
                poll()?;
            }
            let same = lower.to_bits() == upper.to_bits()
                || (lower.to_bits() & 0x7fff_ffff == 0 && upper.to_bits() & 0x7fff_ffff == 0);
            if !lower.is_finite() || !upper.is_finite() || !same {
                return Err(NyError::UnsupportedOp(
                    "causal concrete route requires a finite degenerate input".into(),
                ));
            }
        }
        poll()?;
        let values = if let Some(data) = input.lower().as_slice() {
            Cow::Borrowed(data)
        } else {
            let mut data = Vec::with_capacity(plan.elements);
            for (i, &value) in input.lower().iter().enumerate() {
                if i % 1024 == 0 {
                    poll()?;
                }
                data.push(value);
            }
            Cow::Owned(data)
        };
        let mut output = Vec::with_capacity(plan.elements);
        while output.len() < plan.elements {
            poll()?;
            output.resize((output.len() + 4096).min(plan.elements), 0.0_f32);
        }
        let mut weights = Vec::with_capacity(plan.width);
        for slice in 0..plan.slices {
            for query in 0..plan.queries {
                poll()?;
                let active = self.active_range(query, plan.keys);
                if active.is_empty() {
                    continue;
                }
                let base = (slice * plan.queries + query) * plan.keys;
                let mut shift = f64::NEG_INFINITY;
                for key in active.clone() {
                    if key % 64 == 0 {
                        poll()?;
                    }
                    shift = shift.max(f32_to_f64_exact(values[base + key]));
                }
                weights.clear();
                let mut mass = 0.0_f64;
                for key in active.clone() {
                    if key % 64 == 0 {
                        poll()?;
                    }
                    // NONCERTIFYING libm evaluation. One dominant exp(0)=1
                    // keeps the denominator positive without an epsilon.
                    let weight = (f32_to_f64_exact(values[base + key]) - shift).exp();
                    if !weight.is_finite() {
                        return Err(NyError::NumericalInstability(
                            "causal point exponential".into(),
                        ));
                    }
                    weights.push(weight);
                    mass += weight;
                }
                if !mass.is_finite() || mass <= 0.0 {
                    return Err(NyError::NumericalInstability(
                        "causal point denominator".into(),
                    ));
                }
                for (key, weight) in active.zip(&weights) {
                    if key % 64 == 0 {
                        poll()?;
                    }
                    output[base + key] = (weight / mass) as f32;
                }
            }
        }
        poll()?;
        let array = ArrayD::from_shape_vec(IxDyn(input.shape()), output)
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
        let result = BoundedTensor::concrete_with_poll(array, poll)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
