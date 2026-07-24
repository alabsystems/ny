// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP (Interval Bound Propagation) methods for sequential networks.
//!
//! This module contains IBP propagation and bound collection methods for `Network`.
//! Decomposed into submodules by concern: pure-IBP forward, CROWN-IBP hybrid
//! collection, partial CROWN backward dispatch, and sparse merge utilities.

mod crown_ibp;
mod crown_ibp_forward;
mod crown_partial;
mod crown_partial_gpu;
mod forward;
pub(crate) mod helpers;
mod sparse_merge;
#[cfg(test)]
mod tests;

pub(crate) use forward::try_lower_dense_chain;

use crate::types::CrownIbpBoundsResult;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;

use super::core::Network;

/// Extension trait for IBP propagation on sequential networks.
pub(crate) trait NetworkIbpExt {
    /// Propagate bounds through the entire network using IBP.
    fn propagate_ibp_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor>;

    /// Propagate bounds through the entire network using IBP with an optional GEMM engine.
    fn propagate_ibp_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// Propagate bounds through the network while preserving a prepended leading axis.
    fn propagate_ibp_with_engine_preserve_leading_axis_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// True concrete (point) forward, collapsing to the interval center after each
    /// layer so per-layer soundness widening cannot amplify across a deep network.
    fn propagate_concrete_point_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// Point forward (as above) preserving a prepended leading restart axis.
    fn propagate_concrete_point_preserve_leading_axis_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// Propagate bounds with strict soundness guarantees (directed rounding).
    fn propagate_ibp_sound_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor>;

    /// Strict-soundness IBP forward with an optional GPU engine
    /// (`docs/SOUND_GPU_IBP_PLAN.md` §6.3). Routes a sequential dense chain onto the
    /// certified sound GPU path when the soundness gate is engaged and the engine
    /// advertises `provides_sound_gpu_ibp`; otherwise (and on any failure) uses the
    /// proven-sound CPU loop.
    fn propagate_ibp_sound_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// Run IBP forward pass and collect bounds at each layer.
    fn collect_ibp_bounds_impl(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>>;

    /// Run IBP forward pass and collect bounds at each layer with directed rounding.
    fn collect_ibp_bounds_sound_impl(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>>;

    /// Run CROWN-IBP to collect tighter intermediate bounds.
    fn collect_crown_ibp_bounds_impl(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>>;

    /// Run CROWN-IBP with a custom GEMM engine.
    fn collect_crown_ibp_bounds_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Vec<BoundedTensor>>;

    /// Run CROWN-IBP and collect fallback diagnostics for intermediate layers.
    fn collect_crown_ibp_bounds_with_status_impl(
        &self,
        input: &BoundedTensor,
    ) -> Result<CrownIbpBoundsResult>;

    /// Run CROWN-IBP with custom GEMM engine and collect fallback diagnostics.
    fn collect_crown_ibp_bounds_with_engine_and_status_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownIbpBoundsResult>;

    /// Run CROWN-IBP with deadline enforcement (#3328).
    fn collect_crown_ibp_bounds_with_engine_deadline_and_status_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<CrownIbpBoundsResult>;

    /// Run CROWN-IBP with pre-computed IBP bounds (#3397).
    ///
    /// When `precomputed_ibp` is `Some`, the internal IBP forward pass is skipped
    /// and the provided per-layer bounds are used directly. This avoids redundant
    /// IBP computation when the caller has already run IBP (e.g., for a soundness
    /// check or an earlier verification stage).
    ///
    /// The provided bounds must have exactly one entry per layer.
    fn collect_crown_ibp_bounds_with_precomputed_ibp_impl(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Vec<BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<CrownIbpBoundsResult>;

    /// Like [`Self::collect_crown_ibp_bounds_with_precomputed_ibp_impl`], with
    /// an explicit per-node time-budget policy (#4413, #cgan-bn11-budget).
    ///
    /// Used by the graph collector's sequential fast path so a preset-raised
    /// floor/cap survives the delegation. The default-budget wrappers keep the
    /// built-in constants byte-identically.
    fn collect_crown_ibp_bounds_with_precomputed_ibp_and_budget_impl(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Vec<BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        per_node_time_budget: &crate::types::CrownIbpPerNodeTimeBudget,
    ) -> Result<CrownIbpBoundsResult>;

    /// Core CROWN-IBP collection that accepts optional pre-computed IBP (#3397).
    fn collect_crown_ibp_bounds_core(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Option<Vec<BoundedTensor>>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<CrownIbpBoundsResult>;

    /// Run IBP forward pass with deadline enforcement (#3328).
    fn collect_ibp_bounds_with_deadline_impl(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<Vec<BoundedTensor>>;
}

impl NetworkIbpExt for Network {
    #[inline]
    fn propagate_ibp_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        forward::propagate_ibp(self, input)
    }

    #[inline]
    fn propagate_ibp_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        forward::propagate_ibp_with_engine(self, input, engine)
    }

    #[inline]
    fn propagate_ibp_with_engine_preserve_leading_axis_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        forward::propagate_ibp_with_engine_preserve_leading_axis(self, input, engine)
    }

    #[inline]
    fn propagate_concrete_point_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        forward::propagate_concrete_point_plain(self, input, engine)
    }

    #[inline]
    fn propagate_concrete_point_preserve_leading_axis_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        forward::propagate_concrete_point_preserve_leading_axis(self, input, engine)
    }

    #[inline]
    fn propagate_ibp_sound_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        forward::propagate_ibp_sound(self, input)
    }

    #[inline]
    fn propagate_ibp_sound_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        forward::propagate_ibp_sound_with_engine(self, input, engine)
    }

    #[inline]
    fn collect_ibp_bounds_impl(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>> {
        forward::collect_ibp_bounds(self, input)
    }

    #[inline]
    fn collect_ibp_bounds_sound_impl(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>> {
        forward::collect_ibp_bounds_sound(self, input)
    }

    #[inline]
    fn collect_crown_ibp_bounds_impl(&self, input: &BoundedTensor) -> Result<Vec<BoundedTensor>> {
        Ok(self
            .collect_crown_ibp_bounds_with_status_impl(input)?
            .bounds)
    }

    #[inline]
    fn collect_crown_ibp_bounds_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Vec<BoundedTensor>> {
        Ok(self
            .collect_crown_ibp_bounds_with_engine_and_status_impl(input, engine)?
            .bounds)
    }

    #[inline]
    fn collect_crown_ibp_bounds_with_status_impl(
        &self,
        input: &BoundedTensor,
    ) -> Result<CrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_with_engine_and_status_impl(input, None)
    }

    #[inline]
    fn collect_crown_ibp_bounds_with_engine_and_status_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<CrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_with_engine_deadline_and_status_impl(input, engine, None)
    }

    #[inline]
    fn collect_crown_ibp_bounds_with_engine_deadline_and_status_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<CrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_core(input, None, engine, deadline)
    }

    #[inline]
    fn collect_crown_ibp_bounds_with_precomputed_ibp_impl(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Vec<BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<CrownIbpBoundsResult> {
        self.collect_crown_ibp_bounds_core(input, Some(precomputed_ibp), engine, deadline)
    }

    #[inline]
    fn collect_crown_ibp_bounds_with_precomputed_ibp_and_budget_impl(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Vec<BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        per_node_time_budget: &crate::types::CrownIbpPerNodeTimeBudget,
    ) -> Result<CrownIbpBoundsResult> {
        crown_ibp::collect_core(
            self,
            input,
            Some(precomputed_ibp),
            engine,
            deadline,
            per_node_time_budget,
        )
    }

    #[inline]
    fn collect_crown_ibp_bounds_core(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Option<Vec<BoundedTensor>>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<CrownIbpBoundsResult> {
        crown_ibp::collect_core(
            self,
            input,
            precomputed_ibp,
            engine,
            deadline,
            &crate::types::CrownIbpPerNodeTimeBudget::default(),
        )
    }

    #[inline]
    fn collect_ibp_bounds_with_deadline_impl(
        &self,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<Vec<BoundedTensor>> {
        forward::collect_ibp_bounds_with_deadline(self, input, deadline)
    }
}
