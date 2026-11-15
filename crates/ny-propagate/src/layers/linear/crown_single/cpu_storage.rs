// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Allocation, product and final-publication policy for the ONE CPU Linear
//! propagation body. Ordinary callers use the original owning operations.
//! Alternate owners must keep every returned allocation's claim live and
//! discard all private outputs on error, including the final checkpoint.
//! This interface does not itself establish finite source/allocator coverage.

use super::*;
use std::ops::{Index, IndexMut};

pub(crate) type CpuLayout = super::super::layout::LinearBackwardLayout;

pub(crate) trait CpuStorage {
    type Coefficients: IndexMut<[usize; 2], Output = f32>;
    type Accumulator: IndexMut<usize, Output = f64>;
    type Bias: IndexMut<usize, Output = f32>;
    type RowFlags: IndexMut<usize, Output = bool>;
    type Product: Index<[usize; 2], Output = f64>;
    type Incoming: Index<[usize; 2], Output = f32>;
    type Stack;
    type Output;

    /// Runs before the first allocation. The policy must check any unsupported
    /// dispatch/source class here without constructing a new query authority.
    fn admit<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        bounds: &LinearBounds,
        layout: &CpuLayout,
        deadline: Option<Instant>,
        allow_engine: bool,
        stack_positions: bool,
        work: &mut P,
    ) -> Result<()>;
    fn coefficient_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Coefficients>;
    fn accumulator_zeros(&mut self, rows: usize) -> Result<Self::Accumulator>;
    fn row_flags_zeros(&mut self, rows: usize) -> Result<Self::RowFlags>;
    fn accumulator_slice_mut(accumulator: &mut Self::Accumulator) -> Result<&mut [f64]>;
    fn position_stack<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        bounds: &LinearBounds,
        layout: &CpuLayout,
        deadline: Option<Instant>,
        allow_engine: bool,
        work: &mut P,
    ) -> Result<Option<Self::Stack>>;
    fn aw_for_position<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        bounds: &LinearBounds,
        layout: &CpuLayout,
        position: usize,
        deadline: Option<Instant>,
        allow_engine: bool,
        legacy_f32: bool,
        stack: Option<&Self::Stack>,
        work: &mut P,
    ) -> Result<(Self::Product, Self::Product)>;
    fn incoming_error<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        error: &Array2<f32>,
        offset: usize,
        contraction: usize,
        deadline: Option<Instant>,
        work: &mut P,
    ) -> Result<Self::Incoming>;
    fn finish_biases<P: CooperativeWork>(
        &mut self,
        lower: &Self::Accumulator,
        upper: &Self::Accumulator,
        old_lower: &Array1<f32>,
        old_upper: &Array1<f32>,
        has_bias: bool,
        work: &mut P,
    ) -> Result<(Self::Bias, Self::Bias)>;
    fn publish<P: CooperativeWork>(
        &mut self,
        lower_a: Self::Coefficients,
        lower_b: Self::Bias,
        upper_a: Self::Coefficients,
        upper_b: Self::Bias,
        lower_err: Self::Coefficients,
        upper_err: Self::Coefficients,
        work: &mut P,
    ) -> Result<Self::Output>;
}

/// Borrow the immutable construction-time weight in its original faer layout.
/// This accessor creates no lazy absolute/widened cache or payload copy.
pub(crate) fn original_weight(layer: &LinearLayer) -> faer::MatRef<'_, f32> {
    layer.weight_faer().as_ref()
}

pub(super) struct OwnedCpuStorage;

impl CpuStorage for OwnedCpuStorage {
    type Coefficients = Array2<f32>;
    type Accumulator = Array1<f64>;
    type Bias = Array1<f32>;
    type RowFlags = Vec<bool>;
    type Product = Array2<f64>;
    type Incoming = Array2<f32>;
    type Stack = PositionStackedAw;
    type Output = LinearBounds;

    #[inline]
    fn admit<P: CooperativeWork>(
        &mut self,
        _: &LinearLayer,
        _: &LinearBounds,
        _: &CpuLayout,
        _: Option<Instant>,
        _: bool,
        _: bool,
        _: &mut P,
    ) -> Result<()> {
        Ok(())
    }
    #[inline]
    fn coefficient_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Coefficients> {
        Ok(Array2::<f32>::zeros((rows, cols)))
    }
    #[inline]
    fn accumulator_zeros(&mut self, rows: usize) -> Result<Self::Accumulator> {
        Ok(Array1::<f64>::zeros(rows))
    }
    #[inline]
    fn row_flags_zeros(&mut self, rows: usize) -> Result<Self::RowFlags> {
        Ok(vec![false; rows])
    }
    #[inline]
    fn accumulator_slice_mut(accumulator: &mut Self::Accumulator) -> Result<&mut [f64]> {
        contiguous_flat_slice_mut(accumulator)
    }
    #[inline]
    fn position_stack<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        bounds: &LinearBounds,
        layout: &CpuLayout,
        deadline: Option<Instant>,
        allow_engine: bool,
        _: &mut P,
    ) -> Result<Option<Self::Stack>> {
        position_stacked_aw(layer, bounds, layout, deadline, allow_engine)
    }
    #[inline]
    fn aw_for_position<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        bounds: &LinearBounds,
        layout: &CpuLayout,
        pos: usize,
        deadline: Option<Instant>,
        allow_global_f64_engine: bool,
        legacy_f32: bool,
        position_stack: Option<&Self::Stack>,
        _: &mut P,
    ) -> Result<(Self::Product, Self::Product)> {
        let num_outputs = layout.num_outputs;
        let in_features = layout.in_features;
        let in_start = pos * layout.out_features;
        let (stacked_a64, stacked_s) = if legacy_f32 {
            let lower_block = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                bounds.lower_a()[[i, in_start + j]]
            });
            let upper_block = Mat::<f32>::from_fn(num_outputs, layout.out_features, |i, j| {
                bounds.upper_a()[[i, in_start + j]]
            });
            let ml = mat_mul(&lower_block, layer.weight_faer());
            let mu = mat_mul(&upper_block, layer.weight_faer());
            let a64 = Array2::from_shape_fn((2 * num_outputs, in_features), |(i, j)| {
                if i < num_outputs {
                    f32_to_f64_exact(ml[(i, j)])
                } else {
                    f32_to_f64_exact(mu[(i - num_outputs, j)])
                }
            });
            let s = Array2::<f64>::zeros((2 * num_outputs, in_features));
            (a64, s)
        } else if let Some(stack) = position_stack.as_ref() {
            // #vit-collector-throughput P3: this position's rows of the ONE
            // stacked product — the same pair the per-position call below
            // returns (see `position_stacked_aw`).
            stack.position_block(pos)
        } else {
            // Stack lower OVER upper into one (2·num_outputs × out_features) block:
            // both sides multiply the SAME weight, so the f64 A·W + abs-sum runs as
            // ONE engine GEMM pair over the stacked block instead of two — halving
            // the cuBLAS launch/sync/transfer overhead on the sound-f64 seam (and
            // pushing more work above the offload gate). Rows are independent in
            // `aw_f64_with_abssum` (per-`i` f64 accumulation), so the split halves
            // are bit-identical to two separate calls on the CPU path.
            let stacked = Mat::<f32>::from_fn(2 * num_outputs, layout.out_features, |i, j| {
                if i < num_outputs {
                    bounds.lower_a()[[i, in_start + j]]
                } else {
                    bounds.upper_a()[[i - num_outputs, in_start + j]]
                }
            });
            if allow_global_f64_engine {
                aw_f64_with_abssum_and_deadline_for_layer(layer, &stacked, deadline)?
            } else {
                aw_f64_with_abssum_pollable_cpu_for_layer(layer, &stacked, deadline)?
            }
        };
        Ok((stacked_a64, stacked_s))
    }
    #[inline]
    fn incoming_error<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        error: &Array2<f32>,
        in_start: usize,
        out_features: usize,
        deadline: Option<Instant>,
        _: &mut P,
    ) -> Result<Self::Incoming> {
        match deadline {
            Some(limit) => incoming_error_product_deadline(
                error,
                in_start,
                out_features,
                layer.weight_abs_faer(),
                limit,
            ),
            None => {
                incoming_error_product(error, in_start, out_features, layer.weight_abs_faer(), None)
            }
        }
    }
    #[inline]
    fn finish_biases<P: CooperativeWork>(
        &mut self,
        lower: &Self::Accumulator,
        upper: &Self::Accumulator,
        old_lower: &Array1<f32>,
        old_upper: &Array1<f32>,
        has_bias: bool,
        _: &mut P,
    ) -> Result<(Self::Bias, Self::Bias)> {
        Ok(if has_bias {
            finalize_bias_directed(lower, upper, old_lower, old_upper)
        } else {
            (old_lower.clone(), old_upper.clone())
        })
    }
    #[inline]
    fn publish<P: CooperativeWork>(
        &mut self,
        lower_a: Self::Coefficients,
        lower_b: Self::Bias,
        upper_a: Self::Coefficients,
        upper_b: Self::Bias,
        lower_err: Self::Coefficients,
        upper_err: Self::Coefficients,
        _: &mut P,
    ) -> Result<Self::Output> {
        LinearBounds::new_or_conservative_with_err(
            lower_a, lower_b, upper_a, upper_b, lower_err, upper_err,
        )
    }
}

#[cfg(test)]
#[path = "cpu_storage_tests.rs"]
mod tests;
