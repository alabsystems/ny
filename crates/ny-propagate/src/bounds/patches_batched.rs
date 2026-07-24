// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched patches mode bounds for the CROWN graph engine.
//!
//! Phase 4 of #2613. Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md

use ndarray::{ArrayD, IxDyn};
use ny_core::Result;

use super::patches::{CrownBounds, PatchesLinearBounds};
use super::{BatchedLinearBounds, BatchedLinearBounds64, LinearBounds};
use crate::network::crown_memory::BatchedDenseMaterializationEstimate;

/// Batched wrapper enum for Patches mode in the batched CROWN graph engine.
///
/// Mirrors [`CrownBounds`] for the batched backward path. The batched graph
/// engine (`crown_batched.rs`) operates on `HashMap<String, BatchedCrownBounds>`
/// instead of `HashMap<String, BatchedLinearBounds>` to support Patches mode.
///
/// **MVP constraint:** The Patches variant stores unbatched `PatchesLinearBounds`.
/// This is correct because Conv2d backward is specification-independent — the same
/// kernel applies to all specs. At nonlinear layers (where per-spec slopes differ),
/// `ensure_batched_dense()` converts to `BatchedLinearBounds` before dispatch.
// BatchedLinearBounds is 496 bytes (hot path in backward loop for transformers);
// Patches is heap-allocated via Box. The size difference is acceptable — boxing
// Dense would add deref overhead on every backward step for non-CNN networks.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum BatchedCrownBounds {
    /// Standard dense batched A-matrix bounds (transformers, post-activation).
    Dense(BatchedLinearBounds),
    /// Dense merge-point accumulator kept in f64 until the node is consumed.
    Dense64(BatchedLinearBounds64),
    /// Sparse conv patches bounds, shared across specifications.
    /// Uses unbatched PatchesLinearBounds because conv backward is spec-independent.
    Patches(Box<PatchesLinearBounds>),
}

impl BatchedCrownBounds {
    /// Convert to `BatchedLinearBounds`, materializing Dense if Patches. Consumes self.
    ///
    /// For the Patches variant, materializes the full dense A-matrix via
    /// `PatchesLinearBounds::to_dense()` then wraps as `BatchedLinearBounds`
    /// with no batch dimensions (single-spec equivalent).
    pub(crate) fn into_batched_dense(self) -> Result<BatchedLinearBounds> {
        match self {
            BatchedCrownBounds::Dense(blb) => Ok(blb),
            BatchedCrownBounds::Dense64(blb) => Ok(blb.into_f32()),
            BatchedCrownBounds::Patches(plb) => Self::patches_to_batched_linear(&plb),
        }
    }

    /// Convert Patches to Dense in-place, returning `&mut BatchedLinearBounds`.
    ///
    /// If already Dense, returns the inner `BatchedLinearBounds` directly.
    /// If Patches, materializes to Dense first.
    pub(crate) fn ensure_batched_dense(&mut self) -> Result<&mut BatchedLinearBounds> {
        if matches!(
            self,
            BatchedCrownBounds::Patches(_) | BatchedCrownBounds::Dense64(_)
        ) {
            // Temporary placeholder — immediately replaced below.
            let dense =
                match std::mem::replace(self, BatchedCrownBounds::Dense(Self::placeholder_dense()))
                {
                    BatchedCrownBounds::Patches(plb) => Self::patches_to_batched_linear(&plb)?,
                    BatchedCrownBounds::Dense64(blb) => blb.into_f32(),
                    _ => unreachable!(),
                };
            *self = BatchedCrownBounds::Dense(dense);
        }
        match self {
            BatchedCrownBounds::Dense(blb) => Ok(blb),
            BatchedCrownBounds::Dense64(_) | BatchedCrownBounds::Patches(_) => unreachable!(),
        }
    }

    /// Budget-checked conversion to Dense (#3550). Returns `CpuMemoryExceeded` if
    /// the Patches-to-Dense materialization would exceed the CPU dense budget.
    ///
    /// Dense variant passes through without a budget check (already materialized).
    pub(crate) fn into_batched_dense_checked(
        self,
        site: &'static str,
    ) -> Result<BatchedLinearBounds> {
        if let BatchedCrownBounds::Patches(ref plb) = self {
            let (out_dim, in_dim) = plb.dense_pair_shape()?;
            BatchedDenseMaterializationEstimate::new(site, 1, out_dim, in_dim).check_budget()?;
        }
        self.into_batched_dense()
    }

    /// Budget-checked in-place conversion to Dense (#3550).
    ///
    /// If already Dense, returns the inner reference directly.
    /// If Patches, checks budget before materializing.
    pub(crate) fn ensure_batched_dense_checked(
        &mut self,
        site: &'static str,
    ) -> Result<&mut BatchedLinearBounds> {
        if let BatchedCrownBounds::Patches(ref plb) = self {
            let (out_dim, in_dim) = plb.dense_pair_shape()?;
            BatchedDenseMaterializationEstimate::new(site, 1, out_dim, in_dim).check_budget()?;
        }
        self.ensure_batched_dense()
    }

    /// Helper: convert `PatchesLinearBounds` to `BatchedLinearBounds`.
    ///
    /// Materializes the dense A-matrices and wraps with flattened 1D shapes.
    /// The resulting `BatchedLinearBounds` has no batch dimensions — the
    /// A-matrices are `[out_dim, in_dim]` (2D) and biases are `[out_dim]` (1D).
    fn patches_to_batched_linear(plb: &PatchesLinearBounds) -> Result<BatchedLinearBounds> {
        let dense_lb = plb.to_dense()?;
        Self::linear_to_batched(&dense_lb)
    }

    /// Helper: wrap a `LinearBounds` as `BatchedLinearBounds` with no batch dimensions.
    fn linear_to_batched(lb: &LinearBounds) -> Result<BatchedLinearBounds> {
        let out_dim = lb.num_outputs();
        let in_dim = lb.num_inputs();
        // KEEP unchecked: LinearBounds already validated these arrays; into_dyn()
        // only changes the view rank for batched storage.
        Ok(BatchedLinearBounds::from_parts_unchecked(
            lb.lower_a().clone().into_dyn(),
            lb.lower_b().clone().into_dyn(),
            lb.upper_a().clone().into_dyn(),
            lb.upper_b().clone().into_dyn(),
            vec![in_dim],
            vec![out_dim],
        ))
    }

    /// Convert an unbatched `CrownBounds` to `BatchedCrownBounds`.
    ///
    /// Used when the Patches backward path produces a result that needs to
    /// be stored in the batched bounds map. Preserves the variant:
    /// - `CrownBounds::Dense` → `BatchedCrownBounds::Dense`
    /// - `CrownBounds::Patches` → `BatchedCrownBounds::Patches`
    pub(crate) fn from_crown_bounds(cb: CrownBounds) -> Result<BatchedCrownBounds> {
        match cb {
            CrownBounds::Dense(lb) => Ok(BatchedCrownBounds::Dense(Self::linear_to_batched(&lb)?)),
            CrownBounds::Patches(pb) => Ok(BatchedCrownBounds::Patches(pb)),
        }
    }

    /// Total heap memory used by the current bounds representation, in bytes.
    ///
    /// Dispatches to the appropriate variant's memory tracking.
    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            BatchedCrownBounds::Dense(blb) => blb.memory_bytes(),
            BatchedCrownBounds::Dense64(blb) => blb.memory_bytes(),
            BatchedCrownBounds::Patches(plb) => plb.memory_bytes(),
        }
    }

    /// Whether this is currently in Patches mode.
    pub(crate) fn is_patches(&self) -> bool {
        matches!(self, BatchedCrownBounds::Patches(_))
    }

    /// Merge another dense contribution into this bounds entry, promoting the
    /// dense payload to an f64 accumulator on the first merge.
    pub(crate) fn merge_dense_checked(
        &mut self,
        new_bounds: BatchedLinearBounds,
        site: &'static str,
    ) -> Result<()> {
        if matches!(self, BatchedCrownBounds::Patches(_)) {
            let _ = self.ensure_batched_dense_checked(site)?;
        }

        let merged =
            match std::mem::replace(self, BatchedCrownBounds::Dense(Self::placeholder_dense())) {
                BatchedCrownBounds::Dense(existing) => {
                    let mut accumulator = BatchedLinearBounds64::from_f32(&existing);
                    accumulator.accumulate(&new_bounds);
                    BatchedCrownBounds::Dense64(accumulator)
                }
                BatchedCrownBounds::Dense64(mut existing) => {
                    existing.accumulate(&new_bounds);
                    BatchedCrownBounds::Dense64(existing)
                }
                BatchedCrownBounds::Patches(_) => unreachable!(),
            };
        *self = merged;
        Ok(())
    }

    fn placeholder_dense() -> BatchedLinearBounds {
        // KEEP unchecked: placeholder dense value is all zeros with matching
        // zero-sized shapes, used only during temporary enum replacement.
        BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&[0, 0])),
            ArrayD::zeros(IxDyn(&[0])),
            ArrayD::zeros(IxDyn(&[0, 0])),
            ArrayD::zeros(IxDyn(&[0])),
            vec![0],
            vec![0],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::{next_down_f32, next_up_f32};

    fn scalar_batched_linear_bounds(value: f32) -> BatchedLinearBounds {
        BatchedLinearBounds::from_parts_unchecked(
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), value),
            ArrayD::from_elem(IxDyn(&[1, 1]), value),
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), value),
            ArrayD::from_elem(IxDyn(&[1, 1]), value),
            vec![1, 1],
            vec![1, 1],
        )
    }

    #[test]
    fn test_batched_crown_bounds_into_dense_passthrough() -> Result<()> {
        let blb = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&[4, 4])),
            ArrayD::zeros(IxDyn(&[4])),
            ArrayD::zeros(IxDyn(&[4, 4])),
            ArrayD::zeros(IxDyn(&[4])),
            vec![4],
            vec![4],
        );
        let bcb = BatchedCrownBounds::Dense(blb);
        let result = bcb.into_batched_dense()?;
        assert_eq!(result.lower_a().shape(), &[4, 4]);
        Ok(())
    }

    #[test]
    fn test_batched_crown_bounds_patches_to_dense() -> Result<()> {
        let shape = (1, 2, 2); // 1 channel, 2x2
        let dim = 4;
        let plb = PatchesLinearBounds::identity(shape, shape);
        let bcb = BatchedCrownBounds::Patches(Box::new(plb));
        let result = bcb.into_batched_dense()?;
        // Should produce [4, 4] identity in BatchedLinearBounds
        assert_eq!(result.lower_a().shape(), &[dim, dim]);
        assert_eq!(result.upper_a().shape(), &[dim, dim]);
        assert_eq!(result.lower_b().shape(), &[dim]);
        // Verify identity: diagonal entries = 1.0
        for i in 0..dim {
            assert_eq!(result.lower_a()[[i, i]], 1.0);
            assert_eq!(result.upper_a()[[i, i]], 1.0);
        }
        Ok(())
    }

    #[test]
    fn test_batched_crown_bounds_ensure_dense() -> Result<()> {
        let shape = (1, 2, 2);
        let plb = PatchesLinearBounds::identity(shape, shape);
        let mut bcb = BatchedCrownBounds::Patches(Box::new(plb));
        let blb = bcb.ensure_batched_dense()?;
        assert_eq!(blb.lower_a().shape(), &[4, 4]);
        // Should be Dense variant after ensure
        assert!(matches!(bcb, BatchedCrownBounds::Dense(_)));
        Ok(())
    }

    #[test]
    fn test_batched_crown_bounds_merge_promotes_dense64_until_materialization_3904() -> Result<()> {
        let contributions = [1_099_511_627_776.0_f32, 1.0_f32, -1_099_511_627_776.0_f32];
        let mut bcb = BatchedCrownBounds::Dense(scalar_batched_linear_bounds(contributions[0]));

        bcb.merge_dense_checked(
            scalar_batched_linear_bounds(contributions[1]),
            "test_batched_crown_bounds_merge_promotes_dense64_until_materialization_3904:first",
        )?;
        assert!(
            matches!(bcb, BatchedCrownBounds::Dense64(_)),
            "first merge should promote the dense payload to the f64 accumulator"
        );

        bcb.merge_dense_checked(
            scalar_batched_linear_bounds(contributions[2]),
            "test_batched_crown_bounds_merge_promotes_dense64_until_materialization_3904:second",
        )?;
        assert!(
            matches!(bcb, BatchedCrownBounds::Dense64(_)),
            "subsequent merges should keep the f64 accumulator alive until materialization"
        );

        let merged = bcb.into_batched_dense()?;
        assert_eq!(merged.lower_a()[[0, 0, 0]], next_down_f32(1.0));
        assert_eq!(merged.lower_b()[[0, 0]], next_down_f32(1.0));
        assert_eq!(merged.upper_a()[[0, 0, 0]], next_up_f32(1.0));
        assert_eq!(merged.upper_b()[[0, 0]], next_up_f32(1.0));
        Ok(())
    }
}
