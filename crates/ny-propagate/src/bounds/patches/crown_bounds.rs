// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::Result;
use std::time::Instant;

use crate::bounds::LinearBounds;

use super::{PatchesLinearBounds, PatchesMaterializationPurpose};

/// Wrapper enum: the backward engine operates on this instead of bare LinearBounds.
///
/// This is the key type that enables Patches mode without changing LinearBounds.
/// The backward engine loop dispatches on this enum. Layers that don't support
/// Patches natively receive Dense (via automatic conversion).
// LinearBounds is 224 bytes (hot path in backward loop); Patches is heap-allocated
// via Box. The size difference is acceptable — boxing Dense would add deref overhead
// on every backward step for non-CNN networks (the common case).
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum CrownBounds {
    /// Standard dense A-matrix bounds. Wraps the existing LinearBounds unchanged.
    Dense(LinearBounds),
    /// Sparse conv patches bounds for CNN optimization.
    Patches(Box<PatchesLinearBounds>),
}

impl CrownBounds {
    /// Get as Dense, converting if necessary. Consumes self.
    #[allow(dead_code)]
    #[track_caller]
    pub(crate) fn into_dense(self) -> Result<LinearBounds> {
        self.into_dense_for_purpose(PatchesMaterializationPurpose::Other)
    }

    /// Consuming conversion with an explicit semantic purpose.
    #[allow(dead_code)]
    #[track_caller]
    pub(crate) fn into_dense_for_purpose(
        self,
        purpose: PatchesMaterializationPurpose,
    ) -> Result<LinearBounds> {
        self.into_dense_with_deadline_for_purpose(None, purpose)
    }

    /// Deadline-aware consuming conversion. Already-dense carriers require no
    /// materialization and are returned unchanged.
    #[track_caller]
    pub(crate) fn into_dense_with_deadline(
        self,
        deadline: Option<Instant>,
    ) -> Result<LinearBounds> {
        self.into_dense_with_deadline_for_purpose(deadline, PatchesMaterializationPurpose::Other)
    }

    /// Deadline-aware consuming conversion with an explicit semantic purpose.
    #[track_caller]
    pub(crate) fn into_dense_with_deadline_for_purpose(
        self,
        deadline: Option<Instant>,
        purpose: PatchesMaterializationPurpose,
    ) -> Result<LinearBounds> {
        match self {
            CrownBounds::Dense(lb) => Ok(lb),
            CrownBounds::Patches(pb) => pb.to_dense_with_deadline_for_purpose(deadline, purpose),
        }
    }

    /// Convert Patches to Dense in-place, then return mutable ref to LinearBounds.
    ///
    /// If already Dense, returns the inner LinearBounds directly.
    /// If Patches, materializes to Dense first.
    #[allow(dead_code)]
    #[track_caller]
    pub(crate) fn ensure_dense(&mut self) -> Result<&mut LinearBounds> {
        self.ensure_dense_for_purpose(PatchesMaterializationPurpose::Other)
    }

    /// In-place conversion with an explicit semantic purpose.
    #[allow(dead_code)]
    #[track_caller]
    pub(crate) fn ensure_dense_for_purpose(
        &mut self,
        purpose: PatchesMaterializationPurpose,
    ) -> Result<&mut LinearBounds> {
        self.ensure_dense_with_deadline_for_purpose(None, purpose)
    }

    /// Deadline-aware transactional in-place conversion. The completed dense
    /// relation is installed only after its final deadline checkpoint; every
    /// failure leaves the original Patches carrier byte-for-byte untouched.
    #[track_caller]
    pub(crate) fn ensure_dense_with_deadline(
        &mut self,
        deadline: Option<Instant>,
    ) -> Result<&mut LinearBounds> {
        self.ensure_dense_with_deadline_for_purpose(deadline, PatchesMaterializationPurpose::Other)
    }

    /// Deadline-aware transactional in-place conversion with an explicit
    /// semantic purpose.
    #[track_caller]
    pub(crate) fn ensure_dense_with_deadline_for_purpose(
        &mut self,
        deadline: Option<Instant>,
        purpose: PatchesMaterializationPurpose,
    ) -> Result<&mut LinearBounds> {
        if let CrownBounds::Patches(pb) = self {
            // Convert while borrowing the exact carrier, then publish.  A
            // malformed geometry or resource refusal must leave `self`
            // untouched; installing a 0x0 Dense sentinel before this fallible
            // call let callers accidentally retry an unrelated relation.
            let dense = pb.to_dense_with_deadline_for_purpose(deadline, purpose)?;
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return Err(ny_core::NyError::DeadlineExceeded(
                    "patches materialization: deadline exceeded before CrownBounds publication"
                        .into(),
                ));
            }
            *self = CrownBounds::Dense(dense);
        }
        match self {
            CrownBounds::Dense(lb) => Ok(lb),
            CrownBounds::Patches(_) => unreachable!(),
        }
    }

    /// Total logical heap payload used by the current bounds representation,
    /// in bytes. Backing capacity is not exposed by shared ndarray storage.
    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            CrownBounds::Dense(lb) => lb.memory_bytes(),
            CrownBounds::Patches(pb) => pb.memory_bytes(),
        }
    }

    /// Whether this is currently in Patches mode.
    pub(crate) fn is_patches(&self) -> bool {
        matches!(self, CrownBounds::Patches(_))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, ArrayD, IxDyn};

    use super::*;
    use crate::bounds::patches::{PatchGeometry, PatchesData};

    fn anchored_fixture(valid_axes: bool) -> PatchesLinearBounds {
        let geometry =
            PatchGeometry::anchored(if valid_axes { vec![0, 1] } else { vec![0] }, vec![0, 1])
                .expect("fixture axes are non-empty");
        let data = PatchesData {
            coeff_err: None,
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 1, 1, 1]), vec![0.25, 0.5, 0.75, 1.0])
                    .expect("fixture shape and data length agree"),
            ),
            geometry,
            identity: false,
            output_shape: (1, 2, 2),
            input_shape: (1, 2, 2),
            unstable_idx: None,
        };
        PatchesLinearBounds {
            row_count: 4,
            lower_a: data.clone(),
            lower_b: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            upper_a: data,
            upper_b: Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]),
        }
    }

    fn assert_exact_patches(actual: &PatchesLinearBounds, expected: &PatchesLinearBounds) {
        fn assert_data(actual: &PatchesData, expected: &PatchesData) {
            assert_eq!(actual.coeff_err, expected.coeff_err);
            assert_eq!(actual.patches, expected.patches);
            assert_eq!(actual.geometry, expected.geometry);
            assert_eq!(actual.identity, expected.identity);
            assert_eq!(actual.output_shape, expected.output_shape);
            assert_eq!(actual.input_shape, expected.input_shape);
            assert_eq!(actual.unstable_idx, expected.unstable_idx);
        }

        assert_eq!(actual.row_count, expected.row_count);
        assert_data(&actual.lower_a, &expected.lower_a);
        assert_eq!(actual.lower_b, expected.lower_b);
        assert_data(&actual.upper_a, &expected.upper_a);
        assert_eq!(actual.upper_b, expected.upper_b);
    }

    fn assert_carrier_is_exact_patches(actual: &CrownBounds, expected: &PatchesLinearBounds) {
        match actual {
            CrownBounds::Patches(actual) => assert_exact_patches(actual, expected),
            CrownBounds::Dense(_) => {
                panic!("failed ensure_dense replaced the original Patches carrier")
            }
        }
    }

    #[test]
    fn ensure_dense_malformed_geometry_failure_is_transactional() {
        let expected = anchored_fixture(false);
        let mut bounds = CrownBounds::Patches(Box::new(expected.clone()));

        let error = bounds
            .ensure_dense()
            .expect_err("mismatched anchored axes must be rejected");
        assert!(
            matches!(error, ny_core::NyError::ShapeMismatch { .. }),
            "expected typed shape refusal, got {error:?}"
        );
        assert_carrier_is_exact_patches(&bounds, &expected);
    }

    #[test]
    fn ensure_dense_budget_failure_is_transactional() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "0");
            let expected = anchored_fixture(true);
            let mut bounds = CrownBounds::Patches(Box::new(expected.clone()));

            let error = bounds
                .ensure_dense()
                .expect_err("zero budget must refuse anchored unfold-plan allocation");
            assert!(
                matches!(error, ny_core::NyError::CpuMemoryExceeded { .. }),
                "expected typed memory refusal, got {error:?}"
            );
            assert_carrier_is_exact_patches(&bounds, &expected);
        });
    }

    #[test]
    fn ensure_dense_deadline_failure_is_transactional() {
        let expected = anchored_fixture(true);
        let mut bounds = CrownBounds::Patches(Box::new(expected.clone()));

        let error = bounds
            .ensure_dense_with_deadline(Some(Instant::now()))
            .expect_err("expired materialization must not replace the carrier");
        assert!(
            matches!(error, ny_core::NyError::DeadlineExceeded(_)),
            "expected typed deadline refusal, got {error:?}"
        );
        assert_carrier_is_exact_patches(&bounds, &expected);
    }
}
