// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #margin-subset-seed: scoped publication of the spec-referenced OUTPUT
//! indices for the initial-bounds computation.
//!
//! A single-margin VNN-COMP property (e.g. vggnet16_2022 spec1's
//! `(>= Y_200 Y_177)`) references only k of the network's `output_dim`
//! outputs, yet every full-width OUTPUT-node CROWN backward seeds the whole
//! `[output_dim x output_dim]` identity — on VGG16 that materializes
//! `[1000 x 401408]` conv coefficient buffers (~1.6 GiB each) for 998 rows the
//! verdict never reads. The relu-split initial-bounds computation publishes
//! the referenced indices through a thread-local guard for its own scope. The
//! CROWN-IBP collector seeds ONLY those k identity rows at the OUTPUT node and
//! SCATTERS the k tight CROWN rows over the node's sound IBP/forward bounds for
//! the remaining rows. The separate root alpha-CROWN backward deliberately
//! stays full-width because its optimizer consumes every output row.
//!
//! SOUNDNESS: each identity seed row is an independent linear objective — the
//! backward walk, the per-row CROWN error term, and the per-row concretize are
//! all row-local (the same row-independence the #patches-obj-chunk objective
//! chunking relies on), so the k computed rows are bit-identical to their
//! full-width counterparts. Unreferenced rows keep the node's sound IBP /
//! forward bounds (a valid, merely looser, enclosure). Consumers intersect
//! with IBP exactly as before, so every row of the result remains a valid
//! enclosure.
//!
//! THREADING: the guard is thread-local. The initial-bounds computation and
//! its OUTPUT-node collector pass run synchronously on the publishing thread
//! (rayon parallelism only exists BELOW the seed-shaping decision point), so a
//! worker thread never observes a stale or foreign publication.

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    static PUBLISHED: RefCell<Option<Arc<[usize]>>> = const { RefCell::new(None) };
}

/// Minimum OUTPUT width for the margin-subset seed to engage. Below this the
/// full identity backward is cheap and bit-identical behavior is preferred;
/// at/above it (vggnet16: 1000) the unreferenced rows dominate the cost.
pub(crate) const MARGIN_SUBSET_MIN_OUTPUT_DIM: usize = 512;

/// RAII guard publishing the spec-referenced OUTPUT indices for its scope.
///
/// Nesting-safe: restores the previous publication (usually `None`) on drop,
/// so an inner scope cannot leak its indices into an outer computation.
pub(crate) struct MarginOutputSeedGuard {
    prev: Option<Arc<[usize]>>,
}

impl MarginOutputSeedGuard {
    /// Publish the given OUTPUT indices (sorted, deduplicated) for the scope
    /// of the returned guard. An empty set publishes nothing (subset seeding
    /// stays disengaged, full-width behavior everywhere).
    pub(crate) fn publish(mut indices: Vec<usize>) -> Self {
        indices.sort_unstable();
        indices.dedup();
        let next: Option<Arc<[usize]>> = if indices.is_empty() {
            None
        } else {
            Some(indices.into())
        };
        let prev = PUBLISHED.with(|slot| slot.replace(next));
        Self { prev }
    }

    /// Publish the output indices a linear objective vector actually reads:
    /// the positions with nonzero coefficients. For vggnet16 spec1's
    /// `(>= Y_200 Y_177)` margin objective that is `{177, 200}`.
    pub(crate) fn publish_from_objective(objective: &[f32]) -> Self {
        Self::publish(
            objective
                .iter()
                .enumerate()
                .filter(|(_, &coeff)| coeff != 0.0)
                .map(|(idx, _)| idx)
                .collect(),
        )
    }
}

impl Drop for MarginOutputSeedGuard {
    fn drop(&mut self) {
        PUBLISHED.with(|slot| {
            *slot.borrow_mut() = self.prev.take();
        });
    }
}

/// The published margin OUTPUT indices, if subset seeding should engage for a
/// node of width `output_dim`. Returns `None` (full-width behavior) when:
/// - nothing is published on this thread (every non-initial-bounds caller),
/// - `output_dim` is below [`MARGIN_SUBSET_MIN_OUTPUT_DIM`],
/// - the publication does not describe a strict subset of `0..output_dim`
///   (an index out of range means the node is NOT the spec's output vector —
///   engaging there would scatter rows to wrong coordinates; fail closed).
pub(crate) fn margin_subset_indices(output_dim: usize) -> Option<Arc<[usize]>> {
    if output_dim < MARGIN_SUBSET_MIN_OUTPUT_DIM {
        return None;
    }
    PUBLISHED.with(|slot| {
        let published = slot.borrow();
        let indices = published.as_ref()?;
        if indices.is_empty() || indices.len() >= output_dim {
            return None;
        }
        // Sorted on publish: the last index is the maximum.
        if *indices.last()? >= output_dim {
            return None;
        }
        Some(Arc::clone(indices))
    })
}

/// #margin-subset-seed: scatter k tight CROWN rows over the node's sound
/// IBP/forward bounds.
///
/// Referenced flat positions take the CROWN row values; every other position
/// keeps `base`'s (sound, merely looser) enclosure. Callers that intersect the
/// result with IBP do so exactly as for full-width CROWN maps, so every row of
/// the final bound remains a valid enclosure regardless of which source it
/// came from.
///
/// Moved here from the CROWN-IBP collector (crown_tighten.rs) so the root
/// CROWN backward and the DAG alpha per-iteration backward share the exact
/// same scatter (#margin-subset-alpha).
pub(crate) fn scatter_margin_rows_over_bounds(
    base: &BoundedTensor,
    indices: &[usize],
    lower_rows: &[f32],
    upper_rows: &[f32],
) -> Result<BoundedTensor> {
    if indices.len() != lower_rows.len() || indices.len() != upper_rows.len() {
        return Err(NyError::InvalidSpec(format!(
            "margin-subset scatter: {} indices but {}/{} rows",
            indices.len(),
            lower_rows.len(),
            upper_rows.len()
        )));
    }
    let flat = base.flatten();
    let mut lower = flat.lower().to_owned();
    let mut upper = flat.upper().to_owned();
    for ((&idx, &lo), &up) in indices.iter().zip(lower_rows).zip(upper_rows) {
        if idx >= lower.len() {
            return Err(NyError::InvalidSpec(format!(
                "margin-subset scatter: index {idx} out of range for {} elements",
                lower.len()
            )));
        }
        lower[[idx]] = lo;
        upper[[idx]] = up;
    }
    // Allow infinite endpoints (a degraded CROWN row is still a valid, merely
    // vacuous, enclosure); NaN is rejected downstream by the IBP intersection.
    let scattered = BoundedTensor::new_allow_infinite(lower, upper)?;
    scattered.reshape(base.shape())
}

/// Scatter a k-row concretized subset bound (`subset`, flat length k) over the
/// full-width `base` enclosure. Convenience wrapper over
/// [`scatter_margin_rows_over_bounds`] for callers holding a `BoundedTensor`
/// of the k computed rows.
pub(crate) fn scatter_subset_bounds_over_base(
    base: &BoundedTensor,
    indices: &[usize],
    subset: &BoundedTensor,
) -> Result<BoundedTensor> {
    if subset.len() != indices.len() {
        return Err(NyError::InvalidSpec(format!(
            "margin-subset scatter: subset bound has {} elements for {} indices",
            subset.len(),
            indices.len()
        )));
    }
    let flat = subset.flatten();
    let lower: Vec<f32> = flat.lower().iter().copied().collect();
    let upper: Vec<f32> = flat.upper().iter().copied().collect();
    scatter_margin_rows_over_bounds(base, indices, &lower, &upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpublished_returns_none() {
        assert!(margin_subset_indices(1000).is_none());
    }

    #[test]
    fn guard_publishes_for_scope_and_restores_on_drop() {
        {
            let _guard = MarginOutputSeedGuard::publish(vec![200, 177, 200]);
            let idx = margin_subset_indices(1000).expect("published indices engage");
            assert_eq!(&idx[..], &[177, 200], "sorted + deduped");
        }
        assert!(
            margin_subset_indices(1000).is_none(),
            "drop restores the empty publication"
        );
    }

    #[test]
    fn nested_guards_restore_outer_publication() {
        let _outer = MarginOutputSeedGuard::publish(vec![3]);
        {
            let _inner = MarginOutputSeedGuard::publish(vec![7]);
            assert_eq!(&margin_subset_indices(600).unwrap()[..], &[7]);
        }
        assert_eq!(&margin_subset_indices(600).unwrap()[..], &[3]);
    }

    #[test]
    fn objective_extraction_takes_nonzero_positions() {
        let mut objective = vec![0.0f32; 1000];
        objective[200] = 1.0;
        objective[177] = -1.0;
        let _guard = MarginOutputSeedGuard::publish_from_objective(&objective);
        assert_eq!(&margin_subset_indices(1000).unwrap()[..], &[177, 200]);
    }

    #[test]
    fn scatter_subset_bounds_over_base_replaces_only_referenced_rows() {
        use ndarray::arr1;
        let base = BoundedTensor::new(
            arr1(&[-10.0_f32, -10.0, -10.0, -10.0]).into_dyn(),
            arr1(&[10.0_f32, 10.0, 10.0, 10.0]).into_dyn(),
        )
        .unwrap();
        let subset = BoundedTensor::new(
            arr1(&[-1.0_f32, 2.0]).into_dyn(),
            arr1(&[1.5_f32, 3.0]).into_dyn(),
        )
        .unwrap();
        let scattered = scatter_subset_bounds_over_base(&base, &[1, 3], &subset).unwrap();
        assert_eq!(
            scattered.lower().as_slice().unwrap(),
            &[-10.0, -1.0, -10.0, 2.0]
        );
        assert_eq!(
            scattered.upper().as_slice().unwrap(),
            &[10.0, 1.5, 10.0, 3.0]
        );
        // Row-count mismatch fails closed.
        assert!(scatter_subset_bounds_over_base(&base, &[1], &subset).is_err());
    }

    #[test]
    fn empty_objective_publishes_nothing() {
        let _guard = MarginOutputSeedGuard::publish_from_objective(&[0.0, 0.0]);
        assert!(margin_subset_indices(1000).is_none());
    }

    #[test]
    fn fails_closed_below_min_dim_out_of_range_and_non_strict_subset() {
        let _guard = MarginOutputSeedGuard::publish(vec![177, 200]);
        // Below the engagement width.
        assert!(margin_subset_indices(511).is_none());
        // Index 200 out of range for a 600-wide node? No — in range; engages.
        assert!(margin_subset_indices(600).is_some());
        // Out of range for a node narrower than the max index: fail closed.
        assert!(margin_subset_indices(0).is_none());
        let _guard2 = MarginOutputSeedGuard::publish(vec![599, 700]);
        assert!(
            margin_subset_indices(600).is_none(),
            "max index 700 >= dim 600 must fail closed"
        );
        // Not a STRICT subset: k == dim fails closed.
        let all: Vec<usize> = (0..512).collect();
        let _guard3 = MarginOutputSeedGuard::publish(all);
        assert!(margin_subset_indices(512).is_none());
    }
}
