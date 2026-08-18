// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Move-owned objective/threshold allocation provenance for retained BaB.
//!
//! The ordinary verifier accepts borrowed slices, which erase the outer
//! `Vec` capacities and therefore cannot authorize an exact retained-host
//! baseline. This module provides an opt-in owner that preserves those
//! allocations by move. It remains deliberately non-authorizing: an accounted
//! receipt is only one input to a future finalized-root custody receipt.
//! Neither construction nor observation proves sign normalization, aggregation
//! sense, graph/output association, or finalized-root custody. In particular,
//! the expected output width is caller-supplied observation geometry.

use std::fmt;
use std::mem::size_of;

use ny_core::{Result, GPU_BAB_BOUND_MAX_ARENA_VALUES, GPU_BAB_BOUND_MAX_OBJECTIVES};
use sha2::{Digest, Sha256};

const RESIDENT_OBJECTIVE_SCHEMA_VERSION_V1: u32 = 1;
const RESIDENT_OBJECTIVE_ACCOUNTING_MODEL_V1: u32 = 1;
const OBJECTIVE_POLL_STRIDE: usize = 1024;

/// Original sign-normalized objective rows and positional thresholds.
///
/// Construction only moves the two supplied `Vec` owners. It performs no
/// validation, hashing, allocation, or copying, so an unarmed/default-dark
/// verifier pays no request-scaled work. Retained-v1 observation is explicit
/// and lazy through [`Self::observe_for_resident_v1`].
///
/// This owner intentionally exposes no mutation, `Clone`, serde, or public
/// consuming decomposition. Moving the owner preserves the outer objective
/// allocation, every inner row allocation, and the threshold allocation.
///
/// ```compile_fail
/// use ny_propagate::OwnedSignNormalizedObjectiveSet;
/// let owner = OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![0.0]);
/// let _duplicate = owner.clone();
/// ```
///
/// ```compile_fail
/// use ny_propagate::OwnedSignNormalizedObjectiveSet;
/// let mut owner = OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![0.0]);
/// owner.rows()[0][0] = 2.0;
/// ```
///
/// ```compile_fail
/// use ny_propagate::OwnedSignNormalizedObjectiveSet;
/// let mut owner = OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![0.0]);
/// owner.thresholds()[0] = 1.0;
/// ```
///
/// ```compile_fail
/// use ny_propagate::OwnedSignNormalizedObjectiveSet;
/// let owner = OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![0.0]);
/// let _parts = owner.into_parts();
/// ```
///
/// ```compile_fail
/// use ny_propagate::OwnedSignNormalizedObjectiveSet;
/// fn requires_serialize<T: serde::Serialize>(_: &T) {}
/// let owner = OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![0.0]);
/// requires_serialize(&owner);
/// ```
///
/// ```compile_fail
/// use ny_propagate::OwnedSignNormalizedObjectiveSet;
/// fn requires_deserialize<T: serde::de::DeserializeOwned>() {}
/// requires_deserialize::<OwnedSignNormalizedObjectiveSet>();
/// ```
#[must_use]
pub struct OwnedSignNormalizedObjectiveSet {
    rows: Vec<Vec<f32>>,
    thresholds: Vec<f32>,
}

impl OwnedSignNormalizedObjectiveSet {
    /// Move the original objective and threshold owners without inspecting
    /// their contents or capacities.
    #[inline]
    pub fn new(rows: Vec<Vec<f32>>, thresholds: Vec<f32>) -> Self {
        Self { rows, thresholds }
    }

    /// Borrow the original objective rows in source order.
    #[inline]
    pub fn rows(&self) -> &[Vec<f32>] {
        &self.rows
    }

    /// Borrow the original positional thresholds in source order.
    #[inline]
    pub fn thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Expose allocation-only custody for unit KATs without creating a
    /// retained observation or receipt. This hook is absent from production
    /// builds and therefore cannot become admission authority.
    #[cfg(test)]
    pub(crate) fn allocation_custody_for_test(
        &self,
    ) -> (*const Vec<f32>, usize, *const f32, usize) {
        (
            self.rows.as_ptr(),
            self.rows.capacity(),
            self.thresholds.as_ptr(),
            self.thresholds.capacity(),
        )
    }

    /// Validate and account this exact owner for the retained-v1 host model.
    ///
    /// The caller supplies a cooperative poll that owns its absolute deadline
    /// semantics. `Unsupported` means only that a core retained-v1 count/arena
    /// cap excludes this source and permits an untouched legacy path; it does
    /// not certify remaining geometry, threshold cardinality, or row contents.
    /// `Invalid` is a nonfallback malformed source, observation-geometry, or
    /// checked-arithmetic result for a source not already excluded by those
    /// caps. `Accounted` remains non-authorizing and borrow-bound.
    pub fn observe_for_resident_v1(
        &self,
        expected_output_width: usize,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
    ) -> Result<ResidentObjectiveObservationV1<'_>> {
        check("resident objective admission")?;

        let objective_count = self.rows.len();
        if objective_count == 0 {
            return Ok(ResidentObjectiveObservationV1::Invalid {
                source: self,
                reason: ResidentObjectiveInvalidV1::Empty,
            });
        }
        if objective_count > GPU_BAB_BOUND_MAX_OBJECTIVES {
            return Ok(ResidentObjectiveObservationV1::Unsupported {
                source: self,
                reason: ResidentObjectiveUnsupportedV1::ObjectiveCountExceedsV1 {
                    count: objective_count,
                    maximum: GPU_BAB_BOUND_MAX_OBJECTIVES,
                },
            });
        }
        if expected_output_width == 0 {
            return Ok(ResidentObjectiveObservationV1::Invalid {
                source: self,
                reason: ResidentObjectiveInvalidV1::ExpectedOutputWidthZero,
            });
        }
        let Some(coefficient_values) = objective_count.checked_mul(expected_output_width) else {
            return Ok(ResidentObjectiveObservationV1::Invalid {
                source: self,
                reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
            });
        };
        if coefficient_values > GPU_BAB_BOUND_MAX_ARENA_VALUES {
            return Ok(ResidentObjectiveObservationV1::Unsupported {
                source: self,
                reason: ResidentObjectiveUnsupportedV1::CoefficientArenaExceedsV1 {
                    values: coefficient_values,
                    maximum: GPU_BAB_BOUND_MAX_ARENA_VALUES,
                },
            });
        }
        if self.thresholds.len() != objective_count {
            return Ok(ResidentObjectiveObservationV1::Invalid {
                source: self,
                reason: ResidentObjectiveInvalidV1::ObjectiveThresholdCountMismatch {
                    objectives: objective_count,
                    thresholds: self.thresholds.len(),
                },
            });
        }

        let mut logical = Sha256::new();
        logical.update(b"ny.resident-bab.objectives.logical.v1\0");
        logical.update(RESIDENT_OBJECTIVE_SCHEMA_VERSION_V1.to_le_bytes());
        if hash_usize(&mut logical, objective_count).is_none()
            || hash_usize(&mut logical, expected_output_width).is_none()
            || hash_usize(&mut logical, self.thresholds.len()).is_none()
        {
            return Ok(ResidentObjectiveObservationV1::Invalid {
                source: self,
                reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
            });
        }

        let mut capacity = Sha256::new();
        capacity.update(b"ny.resident-bab.objectives.capacity.v1\0");
        capacity.update(RESIDENT_OBJECTIVE_ACCOUNTING_MODEL_V1.to_le_bytes());
        for value in [
            size_of::<OwnedSignNormalizedObjectiveSet>(),
            size_of::<Vec<f32>>(),
            size_of::<f32>(),
            self.rows.len(),
            self.rows.capacity(),
            expected_output_width,
            self.thresholds.len(),
            self.thresholds.capacity(),
        ] {
            if hash_usize(&mut capacity, value).is_none() {
                return Ok(ResidentObjectiveObservationV1::Invalid {
                    source: self,
                    reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
                });
            }
        }

        let Some(outer_row_bytes) = self.rows.capacity().checked_mul(size_of::<Vec<f32>>()) else {
            return Ok(ResidentObjectiveObservationV1::Invalid {
                source: self,
                reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
            });
        };
        let Some(threshold_bytes) = self.thresholds.capacity().checked_mul(size_of::<f32>()) else {
            return Ok(ResidentObjectiveObservationV1::Invalid {
                source: self,
                reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
            });
        };
        let Some(mut charged_bytes) = size_of::<OwnedSignNormalizedObjectiveSet>()
            .checked_add(outer_row_bytes)
            .and_then(|bytes| bytes.checked_add(threshold_bytes))
        else {
            return Ok(ResidentObjectiveObservationV1::Invalid {
                source: self,
                reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
            });
        };

        for (objective_index, row) in self.rows.iter().enumerate() {
            check("resident objective row")?;
            if row.len() != expected_output_width {
                return Ok(ResidentObjectiveObservationV1::Invalid {
                    source: self,
                    reason: ResidentObjectiveInvalidV1::RowWidthMismatch {
                        objective_index,
                        expected: expected_output_width,
                        actual: row.len(),
                    },
                });
            }
            let Some(row_bytes) = row.capacity().checked_mul(size_of::<f32>()) else {
                return Ok(ResidentObjectiveObservationV1::Invalid {
                    source: self,
                    reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
                });
            };
            let Some(next_charged_bytes) = charged_bytes.checked_add(row_bytes) else {
                return Ok(ResidentObjectiveObservationV1::Invalid {
                    source: self,
                    reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
                });
            };
            charged_bytes = next_charged_bytes;

            if hash_usize(&mut logical, objective_index).is_none()
                || hash_usize(&mut logical, row.len()).is_none()
                || hash_usize(&mut capacity, objective_index).is_none()
                || hash_usize(&mut capacity, row.len()).is_none()
                || hash_usize(&mut capacity, row.capacity()).is_none()
            {
                return Ok(ResidentObjectiveObservationV1::Invalid {
                    source: self,
                    reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
                });
            }
            for (coefficient_index, &coefficient) in row.iter().enumerate() {
                if coefficient_index.is_multiple_of(OBJECTIVE_POLL_STRIDE) {
                    check("resident objective coefficient")?;
                }
                if !coefficient.is_finite() {
                    return Ok(ResidentObjectiveObservationV1::Invalid {
                        source: self,
                        reason: ResidentObjectiveInvalidV1::NonFiniteCoefficient {
                            objective_index,
                            coefficient_index,
                        },
                    });
                }
                logical.update(coefficient.to_bits().to_le_bytes());
            }
            check("resident objective row completion")?;
        }

        for (objective_index, &threshold) in self.thresholds.iter().enumerate() {
            if objective_index.is_multiple_of(OBJECTIVE_POLL_STRIDE) {
                check("resident objective threshold")?;
            }
            if !threshold.is_finite() {
                return Ok(ResidentObjectiveObservationV1::Invalid {
                    source: self,
                    reason: ResidentObjectiveInvalidV1::NonFiniteThreshold { objective_index },
                });
            }
            logical.update(threshold.to_bits().to_le_bytes());
        }
        check("resident objective threshold completion")?;
        if hash_usize(&mut capacity, charged_bytes).is_none() {
            return Ok(ResidentObjectiveObservationV1::Invalid {
                source: self,
                reason: ResidentObjectiveInvalidV1::ArithmeticOverflow,
            });
        }
        let logical_identity_sha256 = logical.finalize().into();
        let capacity_identity_sha256 = capacity.finalize().into();
        check("resident objective observation completion")?;

        Ok(ResidentObjectiveObservationV1::Accounted(
            ResidentObjectiveReceiptV1 {
                source: self,
                logical_identity_sha256,
                capacity_identity_sha256,
                charged_bytes,
            },
        ))
    }
}

/// Ordinary retained-v1 capacity miss. No byte receipt was issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResidentObjectiveUnsupportedV1 {
    ObjectiveCountExceedsV1 { count: usize, maximum: usize },
    CoefficientArenaExceedsV1 { values: usize, maximum: usize },
}

/// Malformed source/observation geometry or checked-arithmetic failure.
///
/// This is nonfallback, but is not by itself evidence that the owner was the
/// source of the invalidity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResidentObjectiveInvalidV1 {
    Empty,
    ObjectiveThresholdCountMismatch {
        objectives: usize,
        thresholds: usize,
    },
    ExpectedOutputWidthZero,
    RowWidthMismatch {
        objective_index: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteCoefficient {
        objective_index: usize,
        coefficient_index: usize,
    },
    NonFiniteThreshold {
        objective_index: usize,
    },
    ArithmeticOverflow,
}

/// Borrow-bound, non-authorizing retained-v1 objective observation.
#[must_use = "only Unsupported permits untouched legacy fallback"]
#[non_exhaustive]
pub enum ResidentObjectiveObservationV1<'a> {
    Accounted(ResidentObjectiveReceiptV1<'a>),
    Unsupported {
        source: &'a OwnedSignNormalizedObjectiveSet,
        reason: ResidentObjectiveUnsupportedV1,
    },
    Invalid {
        source: &'a OwnedSignNormalizedObjectiveSet,
        reason: ResidentObjectiveInvalidV1,
    },
}

impl ResidentObjectiveObservationV1<'_> {
    #[inline]
    pub fn source(&self) -> &OwnedSignNormalizedObjectiveSet {
        match self {
            Self::Accounted(receipt) => receipt.source,
            Self::Unsupported { source, .. } | Self::Invalid { source, .. } => source,
        }
    }

    /// Report whether this non-authorizing observation was a clean v1 capacity
    /// exclusion. A future finalized-root authority must still decide whether
    /// the exact legacy owner may continue untouched.
    #[inline]
    pub fn permits_legacy_fallback(&self) -> bool {
        matches!(self, Self::Unsupported { .. })
    }
}

impl fmt::Debug for ResidentObjectiveObservationV1<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accounted(receipt) => formatter.debug_tuple("Accounted").field(receipt).finish(),
            Self::Unsupported { reason, .. } => formatter
                .debug_struct("Unsupported")
                .field("reason", reason)
                .finish_non_exhaustive(),
            Self::Invalid { reason, .. } => formatter
                .debug_struct("Invalid")
                .field("reason", reason)
                .finish_non_exhaustive(),
        }
    }
}

/// Exact Rust-owner capacity charge and identities for one borrowed source.
///
/// This receipt covers the fixed owner, outer row allocation, every inner row
/// allocation, and the threshold allocation. It deliberately excludes
/// allocator bookkeeping, size-class slack, process RSS, graph/root state, and
/// every other phase allocation. It is neither `Clone` nor `Copy`, retains a
/// real source borrow, and cannot authorize execution on its own.
///
/// ```compile_fail
/// use ny_propagate::{
///     OwnedSignNormalizedObjectiveSet, ResidentObjectiveObservationV1,
/// };
/// let owner = OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![0.0]);
/// let mut check = |_| Ok(());
/// let receipt = match owner.observe_for_resident_v1(1, &mut check).unwrap() {
///     ResidentObjectiveObservationV1::Accounted(receipt) => receipt,
///     _ => unreachable!(),
/// };
/// let _duplicate = receipt.clone();
/// ```
///
/// ```compile_fail
/// use ny_propagate::{
///     OwnedSignNormalizedObjectiveSet, ResidentObjectiveObservationV1,
///     ResidentObjectiveReceiptV1,
/// };
/// fn detached() -> ResidentObjectiveReceiptV1<'static> {
///     let owner = OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![0.0]);
///     let mut check = |_| Ok(());
///     match owner.observe_for_resident_v1(1, &mut check).unwrap() {
///         ResidentObjectiveObservationV1::Accounted(receipt) => receipt,
///         _ => unreachable!(),
///     }
/// }
/// ```
#[must_use]
pub struct ResidentObjectiveReceiptV1<'a> {
    source: &'a OwnedSignNormalizedObjectiveSet,
    logical_identity_sha256: [u8; 32],
    capacity_identity_sha256: [u8; 32],
    charged_bytes: usize,
}

impl ResidentObjectiveReceiptV1<'_> {
    #[inline]
    pub fn logical_identity_sha256(&self) -> &[u8; 32] {
        &self.logical_identity_sha256
    }

    #[inline]
    pub fn capacity_identity_sha256(&self) -> &[u8; 32] {
        &self.capacity_identity_sha256
    }

    #[inline]
    pub fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }

    #[inline]
    pub fn matches_source(&self, source: &OwnedSignNormalizedObjectiveSet) -> bool {
        std::ptr::eq(self.source, source)
    }
}

impl fmt::Debug for ResidentObjectiveReceiptV1<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResidentObjectiveReceiptV1")
            .field("logical_identity_sha256", &self.logical_identity_sha256)
            .field("capacity_identity_sha256", &self.capacity_identity_sha256)
            .field("charged_bytes", &self.charged_bytes)
            .finish_non_exhaustive()
    }
}

fn hash_usize(hash: &mut Sha256, value: usize) -> Option<()> {
    let value = u64::try_from(value).ok()?;
    hash.update(value.to_le_bytes());
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::NyError;

    fn vec_with_capacity(values: &[f32], capacity: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(capacity);
        result.extend_from_slice(values);
        result
    }

    fn rows_with_capacity(rows: Vec<Vec<f32>>, capacity: usize) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(capacity);
        result.extend(rows);
        result
    }

    fn move_owner(owner: OwnedSignNormalizedObjectiveSet) -> OwnedSignNormalizedObjectiveSet {
        owner
    }

    fn observe(
        owner: &OwnedSignNormalizedObjectiveSet,
        width: usize,
    ) -> ResidentObjectiveObservationV1<'_> {
        owner
            .observe_for_resident_v1(width, &mut |_| Ok(()))
            .expect("nonfailing poll")
    }

    fn accounted(
        owner: &OwnedSignNormalizedObjectiveSet,
        width: usize,
    ) -> ResidentObjectiveReceiptV1<'_> {
        match observe(owner, width) {
            ResidentObjectiveObservationV1::Accounted(receipt) => receipt,
            other => panic!("expected Accounted, got {other:?}"),
        }
    }

    fn fixture_with_capacities(
        outer_capacity: usize,
        first_capacity: usize,
        second_capacity: usize,
        threshold_capacity: usize,
    ) -> OwnedSignNormalizedObjectiveSet {
        let rows = rows_with_capacity(
            vec![
                vec_with_capacity(&[1.0, -0.0], first_capacity),
                vec_with_capacity(&[-2.0, 3.0], second_capacity),
            ],
            outer_capacity,
        );
        let thresholds = vec_with_capacity(&[0.5, -0.25], threshold_capacity);
        OwnedSignNormalizedObjectiveSet::new(rows, thresholds)
    }

    #[test]
    fn construction_and_moves_preserve_every_owner_pointer_and_capacity() {
        let rows = rows_with_capacity(
            vec![
                vec_with_capacity(&[1.0, 2.0], 11),
                vec_with_capacity(&[3.0, 4.0], 13),
            ],
            7,
        );
        let outer_pointer = rows.as_ptr();
        let outer_capacity = rows.capacity();
        let row_pointers = [rows[0].as_ptr(), rows[1].as_ptr()];
        let row_capacities = [rows[0].capacity(), rows[1].capacity()];
        let thresholds = vec_with_capacity(&[0.0, 1.0], 17);
        let threshold_pointer = thresholds.as_ptr();
        let threshold_capacity = thresholds.capacity();

        let owner = OwnedSignNormalizedObjectiveSet::new(rows, thresholds);
        let owner = move_owner(owner);
        assert_eq!(owner.rows.as_ptr(), outer_pointer);
        assert_eq!(owner.rows.capacity(), outer_capacity);
        assert_eq!(owner.rows[0].as_ptr(), row_pointers[0]);
        assert_eq!(owner.rows[1].as_ptr(), row_pointers[1]);
        assert_eq!(owner.rows[0].capacity(), row_capacities[0]);
        assert_eq!(owner.rows[1].capacity(), row_capacities[1]);
        assert_eq!(owner.thresholds.as_ptr(), threshold_pointer);
        assert_eq!(owner.thresholds.capacity(), threshold_capacity);
    }

    #[test]
    fn constructor_is_move_only_and_defers_all_validation() {
        let owner =
            OwnedSignNormalizedObjectiveSet::new(vec![vec![f32::NAN], vec![]], vec![f32::INFINITY]);
        assert_eq!(owner.len(), 2);
        assert_eq!(owner.rows()[0][0].to_bits(), f32::NAN.to_bits());
        assert_eq!(owner.thresholds(), &[f32::INFINITY]);
    }

    #[test]
    fn accounted_charge_includes_fixed_outer_rows_and_threshold_capacities() {
        let owner = fixture_with_capacities(7, 11, 13, 17);
        let receipt = accounted(&owner, 2);
        let expected = size_of::<OwnedSignNormalizedObjectiveSet>()
            + owner.rows.capacity() * size_of::<Vec<f32>>()
            + owner
                .rows
                .iter()
                .map(|row| row.capacity() * 4)
                .sum::<usize>()
            + owner.thresholds.capacity() * 4;
        assert_eq!(receipt.charged_bytes(), expected);
        assert!(receipt.matches_source(&owner));
        assert!(!receipt
            .logical_identity_sha256()
            .iter()
            .all(|&byte| byte == 0));
        assert!(!receipt
            .capacity_identity_sha256()
            .iter()
            .all(|&byte| byte == 0));
    }

    #[test]
    fn v1_identity_transcripts_have_known_answers() {
        let owner = fixture_with_capacities(2, 2, 2, 2);
        assert_eq!(owner.rows.capacity(), 2);
        assert_eq!(owner.rows[0].capacity(), 2);
        assert_eq!(owner.rows[1].capacity(), 2);
        assert_eq!(owner.thresholds.capacity(), 2);
        let receipt = accounted(&owner, 2);
        assert_eq!(
            receipt.logical_identity_sha256(),
            &[
                0x6f, 0xd3, 0x45, 0xfa, 0x48, 0x79, 0x01, 0x43, 0x56, 0x58, 0xfa, 0x1e, 0x3c, 0x35,
                0xc1, 0xb1, 0xe0, 0x59, 0xc1, 0xfc, 0x36, 0xec, 0x2d, 0x93, 0xcd, 0xe9, 0x72, 0x53,
                0x02, 0x8f, 0xc5, 0xd6,
            ]
        );

        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(size_of::<OwnedSignNormalizedObjectiveSet>(), 48);
            assert_eq!(size_of::<Vec<f32>>(), 24);
            assert_eq!(receipt.charged_bytes(), 120);
            assert_eq!(
                receipt.capacity_identity_sha256(),
                &[
                    0x78, 0x45, 0x76, 0x22, 0xd9, 0x62, 0x1c, 0x74, 0x67, 0x03, 0x82, 0x56, 0xd2,
                    0x2a, 0x0f, 0xca, 0x36, 0xf5, 0x20, 0xa1, 0x0e, 0x85, 0x85, 0x36, 0x8b, 0x19,
                    0xf9, 0x7a, 0xf4, 0x47, 0x20, 0x73,
                ]
            );
        }
    }

    #[test]
    fn logical_and_capacity_identities_separate_bits_order_and_spare_capacity() {
        let compact = fixture_with_capacities(2, 2, 2, 2);
        let spare = fixture_with_capacities(9, 19, 23, 29);
        let compact_receipt = accounted(&compact, 2);
        let spare_receipt = accounted(&spare, 2);
        assert_eq!(
            compact_receipt.logical_identity_sha256(),
            spare_receipt.logical_identity_sha256()
        );
        assert_ne!(
            compact_receipt.capacity_identity_sha256(),
            spare_receipt.capacity_identity_sha256()
        );

        let permuted = OwnedSignNormalizedObjectiveSet::new(
            vec![vec![-2.0, 3.0], vec![1.0, -0.0]],
            vec![-0.25, 0.5],
        );
        assert_ne!(
            compact_receipt.logical_identity_sha256(),
            accounted(&permuted, 2).logical_identity_sha256()
        );
        let signed_zero = OwnedSignNormalizedObjectiveSet::new(
            vec![vec![1.0, 0.0], vec![-2.0, 3.0]],
            vec![0.5, -0.25],
        );
        assert_ne!(
            compact_receipt.logical_identity_sha256(),
            accounted(&signed_zero, 2).logical_identity_sha256()
        );
        let threshold_signed_zero = OwnedSignNormalizedObjectiveSet::new(
            vec![vec![1.0, -0.0], vec![-2.0, 3.0]],
            vec![0.5, 0.0],
        );
        let threshold_negative_zero = OwnedSignNormalizedObjectiveSet::new(
            vec![vec![1.0, -0.0], vec![-2.0, 3.0]],
            vec![0.5, -0.0],
        );
        assert_ne!(
            accounted(&threshold_signed_zero, 2).logical_identity_sha256(),
            accounted(&threshold_negative_zero, 2).logical_identity_sha256()
        );

        let bit_changed = OwnedSignNormalizedObjectiveSet::new(
            rows_with_capacity(
                vec![
                    vec_with_capacity(&[1.0, -0.0], 2),
                    vec_with_capacity(&[-2.0, 4.0], 2),
                ],
                2,
            ),
            vec_with_capacity(&[0.5, -0.25], 2),
        );
        let bit_changed_receipt = accounted(&bit_changed, 2);
        assert_ne!(
            compact_receipt.logical_identity_sha256(),
            bit_changed_receipt.logical_identity_sha256()
        );
        assert_eq!(
            compact_receipt.capacity_identity_sha256(),
            bit_changed_receipt.capacity_identity_sha256()
        );

        let redistributed_first = fixture_with_capacities(2, 8, 8, 8);
        let redistributed_second = fixture_with_capacities(3, 6, 6, 6);
        let first_receipt = accounted(&redistributed_first, 2);
        let second_receipt = accounted(&redistributed_second, 2);
        assert_eq!(
            first_receipt.charged_bytes(),
            second_receipt.charged_bytes()
        );
        assert_ne!(
            first_receipt.capacity_identity_sha256(),
            second_receipt.capacity_identity_sha256()
        );
    }

    #[test]
    fn structural_and_nonfinite_sources_are_invalid_and_nonfallback() {
        let cases = [
            (
                OwnedSignNormalizedObjectiveSet::new(vec![], vec![]),
                1,
                ResidentObjectiveInvalidV1::Empty,
            ),
            (
                OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![]),
                1,
                ResidentObjectiveInvalidV1::ObjectiveThresholdCountMismatch {
                    objectives: 1,
                    thresholds: 0,
                },
            ),
            (
                OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![0.0]),
                0,
                ResidentObjectiveInvalidV1::ExpectedOutputWidthZero,
            ),
            (
                OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![0.0]),
                2,
                ResidentObjectiveInvalidV1::RowWidthMismatch {
                    objective_index: 0,
                    expected: 2,
                    actual: 1,
                },
            ),
            (
                OwnedSignNormalizedObjectiveSet::new(vec![vec![f32::INFINITY]], vec![0.0]),
                1,
                ResidentObjectiveInvalidV1::NonFiniteCoefficient {
                    objective_index: 0,
                    coefficient_index: 0,
                },
            ),
            (
                OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0]], vec![f32::NAN]),
                1,
                ResidentObjectiveInvalidV1::NonFiniteThreshold { objective_index: 0 },
            ),
            (
                OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0], vec![2.0]], vec![0.0, 0.0]),
                usize::MAX,
                ResidentObjectiveInvalidV1::ArithmeticOverflow,
            ),
        ];
        for (owner, width, expected) in cases {
            let observation = observe(&owner, width);
            assert!(!observation.permits_legacy_fallback());
            assert!(std::ptr::eq(observation.source(), &raw const owner));
            assert!(matches!(
                observation,
                ResidentObjectiveObservationV1::Invalid { reason, .. } if reason == expected
            ));
        }
    }

    #[test]
    fn core_count_and_arena_caps_are_the_only_clean_unsupported_results() {
        let mut rows = Vec::with_capacity(GPU_BAB_BOUND_MAX_OBJECTIVES + 1);
        rows.resize_with(GPU_BAB_BOUND_MAX_OBJECTIVES + 1, Vec::new);
        let too_many = OwnedSignNormalizedObjectiveSet::new(rows, Vec::new());
        let observation = observe(&too_many, 1);
        assert!(observation.permits_legacy_fallback());
        assert!(matches!(
            observation,
            ResidentObjectiveObservationV1::Unsupported {
                reason: ResidentObjectiveUnsupportedV1::ObjectiveCountExceedsV1 { .. },
                ..
            }
        ));

        let too_wide = OwnedSignNormalizedObjectiveSet::new(vec![vec![1.0], vec![2.0]], vec![0.0]);
        let observation = observe(&too_wide, GPU_BAB_BOUND_MAX_ARENA_VALUES / 2 + 1);
        assert!(observation.permits_legacy_fallback());
        assert!(matches!(
            observation,
            ResidentObjectiveObservationV1::Unsupported {
                reason: ResidentObjectiveUnsupportedV1::CoefficientArenaExceedsV1 { .. },
                ..
            }
        ));
    }

    #[test]
    fn cap_exclusions_precede_malformed_contents_without_scaled_polling() {
        let mut rows = Vec::with_capacity(GPU_BAB_BOUND_MAX_OBJECTIVES + 1);
        rows.resize_with(GPU_BAB_BOUND_MAX_OBJECTIVES + 1, Vec::new);
        let too_many = OwnedSignNormalizedObjectiveSet::new(rows, Vec::new());
        let mut count_polls = 0usize;
        let observation = too_many
            .observe_for_resident_v1(1, &mut |_| {
                count_polls += 1;
                Ok(())
            })
            .expect("count-cap admission poll");
        assert_eq!(count_polls, 1);
        assert!(matches!(
            observation,
            ResidentObjectiveObservationV1::Unsupported {
                reason: ResidentObjectiveUnsupportedV1::ObjectiveCountExceedsV1 { .. },
                ..
            }
        ));
        let mut zero_width_polls = 0usize;
        let observation = too_many
            .observe_for_resident_v1(0, &mut |_| {
                zero_width_polls += 1;
                Ok(())
            })
            .expect("count cap precedes width geometry");
        assert_eq!(zero_width_polls, 1);
        assert!(matches!(
            observation,
            ResidentObjectiveObservationV1::Unsupported {
                reason: ResidentObjectiveUnsupportedV1::ObjectiveCountExceedsV1 { .. },
                ..
            }
        ));

        let too_wide =
            OwnedSignNormalizedObjectiveSet::new(vec![vec![f32::NAN], vec![]], Vec::new());
        let mut arena_polls = 0usize;
        let observation = too_wide
            .observe_for_resident_v1(GPU_BAB_BOUND_MAX_ARENA_VALUES / 2 + 1, &mut |_| {
                arena_polls += 1;
                Ok(())
            })
            .expect("arena-cap admission poll");
        assert_eq!(arena_polls, 1);
        assert!(matches!(
            observation,
            ResidentObjectiveObservationV1::Unsupported {
                reason: ResidentObjectiveUnsupportedV1::CoefficientArenaExceedsV1 { .. },
                ..
            }
        ));
    }

    #[test]
    fn polling_failure_propagates_without_becoming_invalid_or_unsupported() {
        let owner = OwnedSignNormalizedObjectiveSet::new(
            vec![vec![1.0; OBJECTIVE_POLL_STRIDE * 2]],
            vec![0.0],
        );
        let mut calls = 0usize;
        let error = owner
            .observe_for_resident_v1(OBJECTIVE_POLL_STRIDE * 2, &mut |_| {
                calls += 1;
                if calls == 4 {
                    Err(NyError::DeadlineExceeded("injected objective poll".into()))
                } else {
                    Ok(())
                }
            })
            .expect_err("injected deadline must propagate");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert_eq!(calls, 4);
    }

    #[test]
    fn threshold_stride_and_final_publication_remain_interruptible() {
        let objective_count = OBJECTIVE_POLL_STRIDE + 1;
        let owner = OwnedSignNormalizedObjectiveSet::new(
            vec![vec![1.0]; objective_count],
            vec![0.0; objective_count],
        );

        let mut threshold_polls = 0usize;
        let error = owner
            .observe_for_resident_v1(1, &mut |label| {
                if label == "resident objective threshold" {
                    threshold_polls += 1;
                    if threshold_polls == 2 {
                        return Err(NyError::DeadlineExceeded(
                            "injected threshold stride".into(),
                        ));
                    }
                }
                Ok(())
            })
            .expect_err("second threshold stride must propagate");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert_eq!(threshold_polls, 2);

        let mut publication_poll_seen = false;
        let error = owner
            .observe_for_resident_v1(1, &mut |label| {
                if label == "resident objective observation completion" {
                    publication_poll_seen = true;
                    return Err(NyError::DeadlineExceeded(
                        "injected final publication".into(),
                    ));
                }
                Ok(())
            })
            .expect_err("final publication poll must propagate");
        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert!(publication_poll_seen);
    }

    #[test]
    fn observation_receipt_cannot_be_detached_from_an_equal_owner() {
        let first = fixture_with_capacities(2, 2, 2, 2);
        let second = fixture_with_capacities(2, 2, 2, 2);
        let receipt = accounted(&first, 2);
        assert!(receipt.matches_source(&first));
        assert!(!receipt.matches_source(&second));
    }
}
