// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Allocation provenance for the retained owners inside [`LinearLayer`].
//!
//! This is deliberately a non-authorizing component observation. It records
//! the exact requested payload capacities retained by the five ndarray owners,
//! the three pinned-faer matrices, and the three lazy transpose caches. Arc
//! control identities are exposed so a future graph owner can deduplicate
//! allocations shared by cloned layers. Nothing here sums a graph, finalizes a
//! root, composes a static payload, or authorizes retained execution.
//!
//! The model is qualified to ndarray 0.17.2, faer 0.24.0, and the workspace's
//! Rust 1.95.0 toolchain. It excludes allocator bookkeeping, allocator
//! size-class slack, and process RSS.

use std::alloc::Layout;
use std::fmt;
use std::mem::{needs_drop, size_of};
use std::ops::Deref;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, OnceLock};

use faer::Mat;
use ndarray::{Array, Array1, Axis, Dimension, Slice};
use ny_core::Result;
use ny_tensor::next_up_f32;

use super::LinearLayer;
use crate::bounds::{nan_propagating_max_zero, nan_propagating_min_zero};

const LINEAR_ALLOCATION_ACCOUNTING_MODEL_V1: u32 = 1;
const FAER_F32_ALIGNMENT_BYTES_V1: usize = 64;
const OBSERVATION_POLL_STRIDE: usize = 1024;

/// Which retained ndarray owner a fact or refusal describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearNdArrayRoleV1 {
    Weight,
    Bias,
    PositiveWeight,
    NegativeWeight,
    RowL2Norms,
}

/// Which private faer matrix a fact or invariant failure describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearFaerRoleV1 {
    PositiveTranspose,
    NegativeTranspose,
    Weight,
}

/// Which lazy row-major transpose cache a fact or refusal describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearTransposeCacheRoleV1 {
    Weight,
    PositiveWeight,
    NegativeWeight,
}

/// Clean V1 capability miss. No component fragment was issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum LinearLayerAllocationUnsupportedV1 {
    NdArrayNonCanonicalCLayout { role: LinearNdArrayRoleV1 },
    EmptyWeightDimension,
    ColdTransposeCache { role: LinearTransposeCacheRoleV1 },
}

/// Hard owner/model invariant failure. This must never become a clean miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum LinearLayerAllocationInvalidV1 {
    NdArrayLayoutArithmeticOverflow {
        role: LinearNdArrayRoleV1,
    },
    NdArrayReconstructionRejected {
        role: LinearNdArrayRoleV1,
    },
    NdArrayRawOffsetInvariant {
        role: LinearNdArrayRoleV1,
    },
    NdArrayLogicalPointerInvariant {
        role: LinearNdArrayRoleV1,
    },
    NdArrayPayloadBytesOverflow {
        role: LinearNdArrayRoleV1,
    },
    BiasLengthMismatch,
    DerivedNdArrayShapeMismatch {
        role: LinearNdArrayRoleV1,
    },
    DerivedNdArrayValueMismatch {
        role: LinearNdArrayRoleV1,
        index: usize,
    },
    FaerShapeMismatch {
        role: LinearFaerRoleV1,
    },
    FaerNegativeColumnStride {
        role: LinearFaerRoleV1,
    },
    FaerColumnStrideMismatch {
        role: LinearFaerRoleV1,
    },
    FaerAllocationBytesOverflow {
        role: LinearFaerRoleV1,
    },
    FaerValueMismatch {
        role: LinearFaerRoleV1,
        row: usize,
        column: usize,
    },
    TransposeCacheControlAlias {
        first: LinearTransposeCacheRoleV1,
        second: LinearTransposeCacheRoleV1,
    },
    TransposeCacheLengthMismatch {
        role: LinearTransposeCacheRoleV1,
        expected: usize,
        actual: usize,
    },
    TransposeCacheValueMismatch {
        role: LinearTransposeCacheRoleV1,
        index: usize,
    },
    TransposeCachePayloadBytesOverflow {
        role: LinearTransposeCacheRoleV1,
    },
    ArcControlAllocationLayoutOverflow,
}

/// Exact retained backing-Vec facts for one ndarray owner.
///
/// Addresses are process-local custody witnesses, not stable serialized
/// identity. The future graph receipt must hash shape/offset/capacity facts and
/// use addresses only for live source matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LinearNdArrayAllocationFactV1 {
    pub(crate) role: LinearNdArrayRoleV1,
    pub(crate) rank: usize,
    pub(crate) shape: [usize; 2],
    pub(crate) raw_allocation_identity: usize,
    pub(crate) logical_data_identity: usize,
    pub(crate) raw_len: usize,
    pub(crate) logical_offset: Option<usize>,
    pub(crate) element_capacity: usize,
    pub(crate) exact_payload_capacity_bytes: usize,
}

/// Exact faer 0.24.0 allocation-request facts for one private matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LinearFaerAllocationFactV1 {
    pub(crate) role: LinearFaerRoleV1,
    pub(crate) allocation_identity: usize,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) column_stride: usize,
    pub(crate) exact_allocation_request_bytes: usize,
}

/// Exact Vec capacity and toolchain-qualified Arc-control request for one
/// immutable, filled transpose cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LinearTransposeCacheAllocationFactV1 {
    pub(crate) role: LinearTransposeCacheRoleV1,
    pub(crate) arc_control_identity: usize,
    pub(crate) vec_data_identity: usize,
    pub(crate) element_len: usize,
    pub(crate) element_capacity: usize,
    pub(crate) exact_payload_capacity_bytes: usize,
    pub(crate) arc_control_allocation_request_bytes: usize,
}

/// Borrow-bound, non-authorizing facts for exactly one [`LinearLayer`].
///
/// This fragment deliberately has no graph total: cloned Linear layers can
/// share cache Arcs, so a future private graph owner must deduplicate cache
/// facts by `arc_control_identity` before charging them.
#[must_use]
pub(crate) struct LinearLayerAllocationFragmentV1<'a> {
    source: &'a LinearLayer,
    accounting_model: u32,
    ndarray_facts: [Option<LinearNdArrayAllocationFactV1>; 5],
    faer_facts: [LinearFaerAllocationFactV1; 3],
    transpose_cache_facts: [LinearTransposeCacheAllocationFactV1; 3],
}

impl LinearLayerAllocationFragmentV1<'_> {
    #[inline]
    pub(crate) fn matches_source(&self, source: &LinearLayer) -> bool {
        std::ptr::eq(self.source, source)
    }

    #[inline]
    pub(crate) fn accounting_model(&self) -> u32 {
        self.accounting_model
    }

    #[inline]
    pub(crate) fn ndarray_facts(&self) -> &[Option<LinearNdArrayAllocationFactV1>; 5] {
        &self.ndarray_facts
    }

    #[inline]
    pub(crate) fn faer_facts(&self) -> &[LinearFaerAllocationFactV1; 3] {
        &self.faer_facts
    }

    #[inline]
    pub(crate) fn transpose_cache_facts(&self) -> &[LinearTransposeCacheAllocationFactV1; 3] {
        &self.transpose_cache_facts
    }
}

impl fmt::Debug for LinearLayerAllocationFragmentV1<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LinearLayerAllocationFragmentV1")
            .field("accounting_model", &self.accounting_model)
            .field("ndarray_facts", &self.ndarray_facts)
            .field("faer_facts", &self.faer_facts)
            .field("transpose_cache_facts", &self.transpose_cache_facts)
            .finish_non_exhaustive()
    }
}

/// Borrow-bound result of observing one exact Linear owner.
#[must_use = "only Unsupported is a clean V1 component miss"]
#[non_exhaustive]
// Keep the borrow-bound facts inline. Boxing only the Accounted variant would
// create a detached heap allocation while observing retained allocations.
#[allow(clippy::large_enum_variant)]
pub(crate) enum LinearLayerAllocationObservationV1<'a> {
    Accounted(LinearLayerAllocationFragmentV1<'a>),
    Unsupported {
        source: &'a LinearLayer,
        reason: LinearLayerAllocationUnsupportedV1,
    },
    Invalid {
        source: &'a LinearLayer,
        reason: LinearLayerAllocationInvalidV1,
    },
}

impl LinearLayerAllocationObservationV1<'_> {
    #[inline]
    pub(crate) fn source(&self) -> &LinearLayer {
        match self {
            Self::Accounted(fragment) => fragment.source,
            Self::Unsupported { source, .. } | Self::Invalid { source, .. } => source,
        }
    }

    #[inline]
    pub(crate) fn permits_legacy_fallback(&self) -> bool {
        matches!(self, Self::Unsupported { .. })
    }
}

impl fmt::Debug for LinearLayerAllocationObservationV1<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accounted(fragment) => {
                formatter.debug_tuple("Accounted").field(fragment).finish()
            }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackedArrayUnsupportedV1 {
    NonCanonicalCLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackedArrayInvalidV1 {
    LayoutArithmeticOverflow,
    ReconstructionRejected,
    RawOffsetInvariant,
    LogicalPointerInvariant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TrackedArrayAccountedV1 {
    raw_allocation_identity: usize,
    logical_data_identity: usize,
    raw_len: usize,
    logical_offset: Option<usize>,
    element_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackedArrayStateV1 {
    Accounted(TrackedArrayAccountedV1),
    Unsupported(TrackedArrayUnsupportedV1),
    Invalid(TrackedArrayInvalidV1),
}

/// Production owner used by Linear's fixed-rank ndarray fields.
///
/// It exposes shared array behavior through `Deref`, never `DerefMut`. Clone
/// clones the actual ndarray and then captures the clone's allocation instead
/// of copying source provenance.
pub(crate) struct TrackedLinearArray<D: Dimension> {
    value: Array<f32, D>,
    allocation: TrackedArrayStateV1,
}

impl<D: Dimension> fmt::Debug for TrackedLinearArray<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(formatter)
    }
}

impl<D: Dimension> Deref for TrackedLinearArray<D> {
    type Target = Array<f32, D>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<D: Dimension> Clone for TrackedLinearArray<D> {
    fn clone(&self) -> Self {
        Self::new(self.value.clone())
    }
}

impl<D: Dimension> TrackedLinearArray<D> {
    pub(crate) fn new(value: Array<f32, D>) -> Self {
        let canonical_layout = match exact_default_c_strides(&value) {
            Ok(canonical) => canonical,
            Err(invalid) => {
                return Self {
                    value,
                    allocation: TrackedArrayStateV1::Invalid(invalid),
                };
            }
        };
        if !canonical_layout {
            return Self {
                value,
                allocation: TrackedArrayStateV1::Unsupported(
                    TrackedArrayUnsupportedV1::NonCanonicalCLayout,
                ),
            };
        }

        let shape = value.raw_dim();
        let Some(logical_values) = value.as_slice() else {
            return Self {
                value,
                allocation: TrackedArrayStateV1::Invalid(
                    TrackedArrayInvalidV1::ReconstructionRejected,
                ),
            };
        };
        if ndarray::ArrayView1::from(logical_values)
            .into_shape_with_order(shape.clone())
            .is_err()
        {
            return Self {
                value,
                allocation: TrackedArrayStateV1::Invalid(
                    TrackedArrayInvalidV1::ReconstructionRejected,
                ),
            };
        }

        let logical_len = value.len();
        let logical_data_identity = value.as_ptr() as usize;
        let (raw, reported_offset) = value.into_raw_vec_and_offset();
        let raw_allocation_identity = raw.as_ptr() as usize;
        let raw_len = raw.len();
        let element_capacity = raw.capacity();

        let mut first_invalid = None;
        let (slice_start, slice_end, logical_offset) = if logical_len == 0 {
            if reported_offset.is_some() {
                first_invalid = Some(TrackedArrayInvalidV1::RawOffsetInvariant);
            }
            (0, 0, None)
        } else {
            let derived_offset = logical_data_identity
                .checked_sub(raw_allocation_identity)
                .filter(|bytes| bytes % size_of::<f32>() == 0)
                .map(|bytes| bytes / size_of::<f32>());
            let valid_span = |offset: usize| {
                offset
                    .checked_add(logical_len)
                    .filter(|&end| end <= raw_len)
                    .map(|end| (offset, end))
            };
            let derived_span = derived_offset.and_then(valid_span);
            let reported_span = reported_offset.and_then(valid_span);
            let usable_span = derived_span.or(reported_span).unwrap_or_else(|| {
                panic!("Linear ndarray lost its reconstructible nonempty raw span")
            });
            if derived_span != Some(usable_span) || reported_span != Some(usable_span) {
                first_invalid = Some(TrackedArrayInvalidV1::RawOffsetInvariant);
            }
            (usable_span.0, usable_span.1, Some(usable_span.0))
        };

        let linear = Array1::from_vec(raw);
        let sliced = linear.slice_axis_move(Axis(0), Slice::from(slice_start..slice_end));
        let rebuilt = sliced
            .into_shape_with_order(shape.clone())
            .unwrap_or_else(|error| {
                panic!("prevalidated Linear ndarray reconstruction failed: {error}")
            });

        if rebuilt.raw_dim() != shape || !exact_default_c_strides(&rebuilt).unwrap_or(false) {
            first_invalid.get_or_insert(TrackedArrayInvalidV1::ReconstructionRejected);
        }
        if logical_len != 0 && rebuilt.as_ptr() as usize != logical_data_identity {
            first_invalid.get_or_insert(TrackedArrayInvalidV1::LogicalPointerInvariant);
        }

        let allocation = match first_invalid {
            Some(invalid) => TrackedArrayStateV1::Invalid(invalid),
            None => TrackedArrayStateV1::Accounted(TrackedArrayAccountedV1 {
                raw_allocation_identity,
                logical_data_identity,
                raw_len,
                logical_offset,
                element_capacity,
            }),
        };
        Self {
            value: rebuilt,
            allocation,
        }
    }

    #[inline]
    pub(crate) fn as_array(&self) -> &Array<f32, D> {
        &self.value
    }

    fn allocation_fact(
        &self,
        role: LinearNdArrayRoleV1,
    ) -> std::result::Result<LinearNdArrayAllocationFactV1, TrackedArrayObservationFailureV1> {
        let accounted = match self.allocation {
            TrackedArrayStateV1::Accounted(accounted) => accounted,
            TrackedArrayStateV1::Unsupported(reason) => {
                return Err(TrackedArrayObservationFailureV1::Unsupported(reason));
            }
            TrackedArrayStateV1::Invalid(reason) => {
                return Err(TrackedArrayObservationFailureV1::Invalid(reason));
            }
        };

        let current_logical_identity = self.value.as_ptr() as usize;
        if !self.value.is_empty() && current_logical_identity != accounted.logical_data_identity {
            return Err(TrackedArrayObservationFailureV1::Invalid(
                TrackedArrayInvalidV1::LogicalPointerInvariant,
            ));
        }
        if let Some(offset) = accounted.logical_offset {
            let Some(offset_bytes) = offset.checked_mul(size_of::<f32>()) else {
                return Err(TrackedArrayObservationFailureV1::Invalid(
                    TrackedArrayInvalidV1::RawOffsetInvariant,
                ));
            };
            if accounted.raw_allocation_identity.checked_add(offset_bytes)
                != Some(accounted.logical_data_identity)
            {
                return Err(TrackedArrayObservationFailureV1::Invalid(
                    TrackedArrayInvalidV1::RawOffsetInvariant,
                ));
            }
        }
        let exact_payload_capacity_bytes = accounted
            .element_capacity
            .checked_mul(size_of::<f32>())
            .ok_or(TrackedArrayObservationFailureV1::PayloadBytesOverflow)?;
        let mut shape = [0usize; 2];
        shape[..self.value.ndim()].copy_from_slice(self.value.shape());
        Ok(LinearNdArrayAllocationFactV1 {
            role,
            rank: self.value.ndim(),
            shape,
            raw_allocation_identity: accounted.raw_allocation_identity,
            logical_data_identity: accounted.logical_data_identity,
            raw_len: accounted.raw_len,
            logical_offset: accounted.logical_offset,
            element_capacity: accounted.element_capacity,
            exact_payload_capacity_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackedArrayObservationFailureV1 {
    Unsupported(TrackedArrayUnsupportedV1),
    Invalid(TrackedArrayInvalidV1),
    PayloadBytesOverflow,
}

fn exact_default_c_strides<D: Dimension>(
    value: &Array<f32, D>,
) -> std::result::Result<bool, TrackedArrayInvalidV1> {
    let shape = value.shape();
    let strides = value.strides();
    if shape.contains(&0) {
        return Ok(strides.iter().all(|&stride| stride == 0));
    }

    let mut expected = 1usize;
    for (&dimension, &stride) in shape.iter().zip(strides).rev() {
        let expected_stride = isize::try_from(expected)
            .map_err(|_| TrackedArrayInvalidV1::LayoutArithmeticOverflow)?;
        if stride != expected_stride {
            return Ok(false);
        }
        expected = expected
            .checked_mul(dimension)
            .ok_or(TrackedArrayInvalidV1::LayoutArithmeticOverflow)?;
    }
    Ok(true)
}

impl LinearLayer {
    /// Observe all retained Linear allocations without materializing a cache.
    ///
    /// The caller owns deadline semantics through `check`. A cold cache or a
    /// valid-but-unmodelled ndarray layout is a clean `Unsupported` component
    /// miss. Malformed private derived state, checked arithmetic failure, or a
    /// provenance invariant failure is `Invalid`. Parameter finiteness is not
    /// an allocation invariant: constructor-valid nonfinite values remain
    /// accountable when every immutable derived copy agrees bit-for-bit. A
    /// later semantic-eligibility gate owns any finiteness policy. Only an
    /// all-warm, fully validated layer yields a borrow-bound component fragment.
    pub(crate) fn observe_host_allocations_v1(
        &self,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
    ) -> Result<LinearLayerAllocationObservationV1<'_>> {
        check("linear allocation admission")?;

        let mut first_unsupported = None;
        let mut ndarray_facts = [None; 5];
        for (slot, role, capture) in [
            (
                0,
                LinearNdArrayRoleV1::Weight,
                self.weight.allocation_fact(LinearNdArrayRoleV1::Weight),
            ),
            (
                2,
                LinearNdArrayRoleV1::PositiveWeight,
                self.w_pos
                    .allocation_fact(LinearNdArrayRoleV1::PositiveWeight),
            ),
            (
                3,
                LinearNdArrayRoleV1::NegativeWeight,
                self.w_neg
                    .allocation_fact(LinearNdArrayRoleV1::NegativeWeight),
            ),
            (
                4,
                LinearNdArrayRoleV1::RowL2Norms,
                self.row_l2_norms
                    .allocation_fact(LinearNdArrayRoleV1::RowL2Norms),
            ),
        ] {
            match capture {
                Ok(fact) => ndarray_facts[slot] = Some(fact),
                Err(failure) => match map_array_failure(role, failure) {
                    ArrayFailureMappedV1::Unsupported(reason) => {
                        first_unsupported.get_or_insert(reason);
                    }
                    ArrayFailureMappedV1::Invalid(reason) => {
                        return Ok(LinearLayerAllocationObservationV1::Invalid {
                            source: self,
                            reason,
                        });
                    }
                },
            }
        }
        if let Some(bias) = &self.bias {
            match bias.allocation_fact(LinearNdArrayRoleV1::Bias) {
                Ok(fact) => ndarray_facts[1] = Some(fact),
                Err(failure) => match map_array_failure(LinearNdArrayRoleV1::Bias, failure) {
                    ArrayFailureMappedV1::Unsupported(reason) => {
                        first_unsupported.get_or_insert(reason);
                    }
                    ArrayFailureMappedV1::Invalid(reason) => {
                        return Ok(LinearLayerAllocationObservationV1::Invalid {
                            source: self,
                            reason,
                        });
                    }
                },
            }
        }

        let (out_features, in_features) = self.weight.dim();
        if out_features == 0 || in_features == 0 {
            // Zero-axis Linear layers are constructor-valid and remain legacy
            // executable, but V1 deliberately does not publish their dangling
            // empty-owner identities. Continue validating all private state so
            // this clean model miss cannot hide a harder invariant failure.
            first_unsupported
                .get_or_insert(LinearLayerAllocationUnsupportedV1::EmptyWeightDimension);
        }
        if self
            .bias
            .as_ref()
            .is_some_and(|bias| bias.len() != out_features)
        {
            return Ok(LinearLayerAllocationObservationV1::Invalid {
                source: self,
                reason: LinearLayerAllocationInvalidV1::BiasLengthMismatch,
            });
        }
        for (role, actual) in [
            (LinearNdArrayRoleV1::PositiveWeight, self.w_pos.dim()),
            (LinearNdArrayRoleV1::NegativeWeight, self.w_neg.dim()),
        ] {
            if actual != (out_features, in_features) {
                return Ok(LinearLayerAllocationObservationV1::Invalid {
                    source: self,
                    reason: LinearLayerAllocationInvalidV1::DerivedNdArrayShapeMismatch { role },
                });
            }
        }
        if self.row_l2_norms.len() != out_features {
            return Ok(LinearLayerAllocationObservationV1::Invalid {
                source: self,
                reason: LinearLayerAllocationInvalidV1::DerivedNdArrayShapeMismatch {
                    role: LinearNdArrayRoleV1::RowL2Norms,
                },
            });
        }

        let faer_facts = match self.observe_faer_facts_v1() {
            Ok(facts) => facts,
            Err(reason) => {
                return Ok(LinearLayerAllocationObservationV1::Invalid {
                    source: self,
                    reason,
                });
            }
        };
        if let Err(reason) = self.validate_parameter_and_derived_values_v1(check)? {
            return Ok(LinearLayerAllocationObservationV1::Invalid {
                source: self,
                reason,
            });
        }

        let transpose_cache_facts =
            match self.observe_transpose_cache_facts_v1(check, &mut first_unsupported)? {
                Ok(Some(facts)) => facts,
                Ok(None) => {
                    let reason = first_unsupported
                        .expect("a missing cache fact records one clean unsupported reason");
                    return Ok(LinearLayerAllocationObservationV1::Unsupported {
                        source: self,
                        reason,
                    });
                }
                Err(reason) => {
                    return Ok(LinearLayerAllocationObservationV1::Invalid {
                        source: self,
                        reason,
                    });
                }
            };

        debug_assert!(first_unsupported.is_none());
        check("linear allocation fragment publication")?;
        Ok(LinearLayerAllocationObservationV1::Accounted(
            LinearLayerAllocationFragmentV1 {
                source: self,
                accounting_model: LINEAR_ALLOCATION_ACCOUNTING_MODEL_V1,
                ndarray_facts,
                faer_facts,
                transpose_cache_facts,
            },
        ))
    }

    fn observe_faer_facts_v1(
        &self,
    ) -> std::result::Result<[LinearFaerAllocationFactV1; 3], LinearLayerAllocationInvalidV1> {
        let (out_features, in_features) = self.weight.dim();
        Ok([
            faer_allocation_fact_v1(
                &self.w_pos_t_faer,
                LinearFaerRoleV1::PositiveTranspose,
                in_features,
                out_features,
            )?,
            faer_allocation_fact_v1(
                &self.w_neg_t_faer,
                LinearFaerRoleV1::NegativeTranspose,
                in_features,
                out_features,
            )?,
            faer_allocation_fact_v1(
                &self.weight_faer,
                LinearFaerRoleV1::Weight,
                out_features,
                in_features,
            )?,
        ])
    }

    fn validate_parameter_and_derived_values_v1(
        &self,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
    ) -> Result<std::result::Result<(), LinearLayerAllocationInvalidV1>> {
        let (out_features, in_features) = self.weight.dim();
        let mut flat_index = 0usize;
        for row in 0..out_features {
            if row.is_multiple_of(OBSERVATION_POLL_STRIDE) {
                check("linear allocation row norm values")?;
            }
            let mut row_sum_squares = 0.0_f64;
            for column in 0..in_features {
                if flat_index.is_multiple_of(OBSERVATION_POLL_STRIDE) {
                    check("linear allocation parameter values")?;
                }
                let weight = self.weight[[row, column]];
                let weight_f64 = f64::from(weight);
                row_sum_squares += weight_f64 * weight_f64;
                let positive = nan_propagating_max_zero(weight);
                let negative = nan_propagating_min_zero(weight);
                if self.w_pos[[row, column]].to_bits() != positive.to_bits() {
                    return Ok(Err(
                        LinearLayerAllocationInvalidV1::DerivedNdArrayValueMismatch {
                            role: LinearNdArrayRoleV1::PositiveWeight,
                            index: flat_index,
                        },
                    ));
                }
                if self.w_neg[[row, column]].to_bits() != negative.to_bits() {
                    return Ok(Err(
                        LinearLayerAllocationInvalidV1::DerivedNdArrayValueMismatch {
                            role: LinearNdArrayRoleV1::NegativeWeight,
                            index: flat_index,
                        },
                    ));
                }
                for (role, actual, expected) in [
                    (
                        LinearFaerRoleV1::Weight,
                        self.weight_faer[(row, column)],
                        weight,
                    ),
                    (
                        LinearFaerRoleV1::PositiveTranspose,
                        self.w_pos_t_faer[(column, row)],
                        positive,
                    ),
                    (
                        LinearFaerRoleV1::NegativeTranspose,
                        self.w_neg_t_faer[(column, row)],
                        negative,
                    ),
                ] {
                    if actual.to_bits() != expected.to_bits() {
                        return Ok(Err(LinearLayerAllocationInvalidV1::FaerValueMismatch {
                            role,
                            row,
                            column,
                        }));
                    }
                }
                flat_index += 1;
            }
            let expected_row_norm = next_up_f32(row_sum_squares.sqrt() as f32);
            if self.row_l2_norms[row].to_bits() != expected_row_norm.to_bits() {
                return Ok(Err(
                    LinearLayerAllocationInvalidV1::DerivedNdArrayValueMismatch {
                        role: LinearNdArrayRoleV1::RowL2Norms,
                        index: row,
                    },
                ));
            }
        }
        if let Some(bias) = &self.bias {
            for (index, _) in bias.iter().enumerate() {
                if index.is_multiple_of(OBSERVATION_POLL_STRIDE) {
                    check("linear allocation bias values")?;
                }
            }
        }
        Ok(Ok(()))
    }

    fn observe_transpose_cache_facts_v1(
        &self,
        check: &mut dyn FnMut(&'static str) -> Result<()>,
        first_unsupported: &mut Option<LinearLayerAllocationUnsupportedV1>,
    ) -> Result<
        std::result::Result<
            Option<[LinearTransposeCacheAllocationFactV1; 3]>,
            LinearLayerAllocationInvalidV1,
        >,
    > {
        let caches = [
            (
                LinearTransposeCacheRoleV1::Weight,
                &self.weight_t_rm,
                self.weight.as_array(),
            ),
            (
                LinearTransposeCacheRoleV1::PositiveWeight,
                &self.w_pos_t_rm,
                self.w_pos.as_array(),
            ),
            (
                LinearTransposeCacheRoleV1::NegativeWeight,
                &self.w_neg_t_rm,
                self.w_neg.as_array(),
            ),
        ];
        for first in 0..caches.len() {
            for second in (first + 1)..caches.len() {
                if Arc::ptr_eq(caches[first].1, caches[second].1) {
                    return Ok(Err(
                        LinearLayerAllocationInvalidV1::TransposeCacheControlAlias {
                            first: caches[first].0,
                            second: caches[second].0,
                        },
                    ));
                }
            }
        }

        let Some(arc_control_allocation_request_bytes) = arc_control_allocation_request_bytes_v1()
        else {
            return Ok(Err(
                LinearLayerAllocationInvalidV1::ArcControlAllocationLayoutOverflow,
            ));
        };
        let mut facts = [None; 3];
        for (slot, (role, cell, source)) in caches.into_iter().enumerate() {
            let Some(values) = cell.get() else {
                first_unsupported
                    .get_or_insert(LinearLayerAllocationUnsupportedV1::ColdTransposeCache { role });
                continue;
            };
            let Some(expected_len) = source.nrows().checked_mul(source.ncols()) else {
                return Ok(Err(
                    LinearLayerAllocationInvalidV1::TransposeCacheLengthMismatch {
                        role,
                        expected: usize::MAX,
                        actual: values.len(),
                    },
                ));
            };
            if values.len() != expected_len {
                return Ok(Err(
                    LinearLayerAllocationInvalidV1::TransposeCacheLengthMismatch {
                        role,
                        expected: expected_len,
                        actual: values.len(),
                    },
                ));
            }
            let mut index = 0usize;
            for column in 0..source.ncols() {
                for row in 0..source.nrows() {
                    if index.is_multiple_of(OBSERVATION_POLL_STRIDE) {
                        check("linear allocation transpose cache values")?;
                    }
                    if values[index].to_bits() != source[[row, column]].to_bits() {
                        return Ok(Err(
                            LinearLayerAllocationInvalidV1::TransposeCacheValueMismatch {
                                role,
                                index,
                            },
                        ));
                    }
                    index += 1;
                }
            }
            let Some(exact_payload_capacity_bytes) =
                values.capacity().checked_mul(size_of::<f32>())
            else {
                return Ok(Err(
                    LinearLayerAllocationInvalidV1::TransposeCachePayloadBytesOverflow { role },
                ));
            };
            facts[slot] = Some(LinearTransposeCacheAllocationFactV1 {
                role,
                arc_control_identity: Arc::as_ptr(cell) as usize,
                vec_data_identity: values.as_ptr() as usize,
                element_len: values.len(),
                element_capacity: values.capacity(),
                exact_payload_capacity_bytes,
                arc_control_allocation_request_bytes,
            });
        }

        if first_unsupported.is_some() {
            return Ok(Ok(None));
        }
        Ok(Ok(Some(facts.map(|fact| {
            fact.expect("all warm Linear caches produced allocation facts")
        }))))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArrayFailureMappedV1 {
    Unsupported(LinearLayerAllocationUnsupportedV1),
    Invalid(LinearLayerAllocationInvalidV1),
}

fn map_array_failure(
    role: LinearNdArrayRoleV1,
    failure: TrackedArrayObservationFailureV1,
) -> ArrayFailureMappedV1 {
    match failure {
        TrackedArrayObservationFailureV1::Unsupported(
            TrackedArrayUnsupportedV1::NonCanonicalCLayout,
        ) => ArrayFailureMappedV1::Unsupported(
            LinearLayerAllocationUnsupportedV1::NdArrayNonCanonicalCLayout { role },
        ),
        TrackedArrayObservationFailureV1::Invalid(
            TrackedArrayInvalidV1::LayoutArithmeticOverflow,
        ) => ArrayFailureMappedV1::Invalid(
            LinearLayerAllocationInvalidV1::NdArrayLayoutArithmeticOverflow { role },
        ),
        TrackedArrayObservationFailureV1::Invalid(
            TrackedArrayInvalidV1::ReconstructionRejected,
        ) => ArrayFailureMappedV1::Invalid(
            LinearLayerAllocationInvalidV1::NdArrayReconstructionRejected { role },
        ),
        TrackedArrayObservationFailureV1::Invalid(TrackedArrayInvalidV1::RawOffsetInvariant) => {
            ArrayFailureMappedV1::Invalid(
                LinearLayerAllocationInvalidV1::NdArrayRawOffsetInvariant { role },
            )
        }
        TrackedArrayObservationFailureV1::Invalid(
            TrackedArrayInvalidV1::LogicalPointerInvariant,
        ) => ArrayFailureMappedV1::Invalid(
            LinearLayerAllocationInvalidV1::NdArrayLogicalPointerInvariant { role },
        ),
        TrackedArrayObservationFailureV1::PayloadBytesOverflow => ArrayFailureMappedV1::Invalid(
            LinearLayerAllocationInvalidV1::NdArrayPayloadBytesOverflow { role },
        ),
    }
}

fn faer_allocation_fact_v1(
    matrix: &Mat<f32>,
    role: LinearFaerRoleV1,
    expected_nrows: usize,
    expected_ncols: usize,
) -> std::result::Result<LinearFaerAllocationFactV1, LinearLayerAllocationInvalidV1> {
    if matrix.nrows() != expected_nrows || matrix.ncols() != expected_ncols {
        return Err(LinearLayerAllocationInvalidV1::FaerShapeMismatch { role });
    }
    let column_stride = usize::try_from(matrix.col_stride())
        .map_err(|_| LinearLayerAllocationInvalidV1::FaerNegativeColumnStride { role })?;
    let expected_column_stride = expected_faer_f32_column_stride_v1(expected_nrows)
        .ok_or(LinearLayerAllocationInvalidV1::FaerAllocationBytesOverflow { role })?;
    if column_stride != expected_column_stride {
        return Err(LinearLayerAllocationInvalidV1::FaerColumnStrideMismatch { role });
    }
    let exact_allocation_request_bytes = column_stride
        .checked_mul(expected_ncols)
        .and_then(|elements| elements.checked_mul(size_of::<f32>()))
        .ok_or(LinearLayerAllocationInvalidV1::FaerAllocationBytesOverflow { role })?;
    Ok(LinearFaerAllocationFactV1 {
        role,
        allocation_identity: matrix.as_ptr() as usize,
        nrows: expected_nrows,
        ncols: expected_ncols,
        column_stride,
        exact_allocation_request_bytes,
    })
}

fn expected_faer_f32_column_stride_v1(nrows: usize) -> Option<usize> {
    debug_assert!(size_of::<f32>().is_power_of_two());
    debug_assert!(!needs_drop::<f32>());
    nrows.checked_next_multiple_of(FAER_F32_ALIGNMENT_BYTES_V1 / size_of::<f32>())
}

fn arc_control_allocation_request_bytes_v1() -> Option<usize> {
    // Rust 1.95.0 `ArcInner<T>` is #[repr(C)] { strong, weak, data }.
    // Reproduce that allocation request without naming std's private type.
    let (layout, _) = Layout::new::<AtomicUsize>()
        .extend(Layout::new::<AtomicUsize>())
        .ok()?;
    let (layout, _) = layout.extend(Layout::new::<OnceLock<Vec<f32>>>()).ok()?;
    Some(layout.pad_to_align().size())
}

#[cfg(test)]
mod tests;
