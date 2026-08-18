// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Allocation-capacity provenance for [`BoundedTensor`].
//!
//! This module deliberately accounts a narrow, deterministic ownership model:
//! exact backing-`Vec<f32>` payload capacity plus a conservative ownership
//! charge for dimension/stride metadata. It does not claim allocator
//! bookkeeping, allocator size-class slack, process RSS, or authorization to
//! open a retained execution phase.
//!
//! V1 `Accounted` results are implementation-qualified to exactly ndarray
//! 0.17.2. Any ndarray upgrade requires a representation/reconstruction
//! re-audit and the hostile known-answer tests in this module.

use std::fmt;
use std::mem::size_of;
use std::ops::Deref;

use ndarray::{Array1, ArrayD, Axis, IxDyn, Slice};
use serde::{Serialize, Serializer};

use super::BoundedTensor;

/// Maximum dynamic-array rank supported by the V1 capacity receipt.
pub const BOUNDED_TENSOR_HOST_ALLOCATION_MAX_RANK_V1: usize = 4;

/// Which endpoint caused a V1 provenance refusal or invariant failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundedTensorHostAllocationEndpointV1 {
    Lower,
    Upper,
}

/// A clean capability miss. No byte count is authoritative in this state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BoundedTensorHostAllocationUnsupportedV1 {
    /// The L2 annotation owns additional arrays that V1 deliberately does not
    /// account.
    L2ConstraintPresent,
    /// The endpoint is not exact default row-major layout. Merely being
    /// contiguous is insufficient because rebuilding must preserve strides.
    EndpointNonCanonicalCLayout {
        endpoint: BoundedTensorHostAllocationEndpointV1,
    },
    /// Rebuilding this dynamic rank would cross V1's bounded metadata surface.
    EndpointRankExceedsV1 {
        endpoint: BoundedTensorHostAllocationEndpointV1,
        rank: usize,
    },
}

/// A hard provenance invariant failure. Callers must not reinterpret this as a
/// clean unsupported/fallback result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BoundedTensorHostAllocationInvalidV1 {
    EndpointLayoutArithmeticOverflow {
        endpoint: BoundedTensorHostAllocationEndpointV1,
    },
    EndpointReconstructionRejected {
        endpoint: BoundedTensorHostAllocationEndpointV1,
    },
    EndpointRawOffsetInvariant {
        endpoint: BoundedTensorHostAllocationEndpointV1,
    },
    EndpointLogicalPointerInvariant {
        endpoint: BoundedTensorHostAllocationEndpointV1,
    },
    EndpointPayloadBytesOverflow {
        endpoint: BoundedTensorHostAllocationEndpointV1,
    },
    EndpointDimensionStrideBytesOverflow {
        endpoint: BoundedTensorHostAllocationEndpointV1,
    },
    TotalChargedBytesOverflow,
}

/// Borrow-bound, non-authorizing V1 allocation provenance.
///
/// `Unsupported` is an ordinary capability miss. `Invalid` is a hard
/// provenance failure and must never be laundered into a lower-bound byte
/// estimate or a clean retained-path decline.
#[must_use]
#[derive(Debug)]
#[non_exhaustive]
pub enum BoundedTensorHostAllocationProvenanceV1<'a> {
    Accounted(BoundedTensorHostAllocationReceiptV1<'a>),
    Unsupported(BoundedTensorHostAllocationUnsupportedV1),
    Invalid(BoundedTensorHostAllocationInvalidV1),
}

/// Exact endpoint payload capacities plus conservative V1 metadata charging.
///
/// The receipt retains a real borrow of its source tensor, so the tensor cannot
/// be mutably replaced while the receipt is live. It is deliberately neither
/// `Clone` nor `Copy` and does not authorize retained execution. Its accounting
/// claim is qualified to exactly ndarray 0.17.2; an ndarray upgrade requires
/// re-audit and rerunning V1's hostile known-answer tests.
#[must_use]
pub struct BoundedTensorHostAllocationReceiptV1<'a> {
    source: &'a BoundedTensor,
    exact_payload_capacity_bytes: usize,
    conservative_dimension_stride_bytes: usize,
    accountable_charged_bytes: usize,
}

impl fmt::Debug for BoundedTensorHostAllocationReceiptV1<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundedTensorHostAllocationReceiptV1")
            .field("lower_element_capacity", &self.lower_element_capacity())
            .field("upper_element_capacity", &self.upper_element_capacity())
            .field(
                "exact_payload_capacity_bytes",
                &self.exact_payload_capacity_bytes,
            )
            .field(
                "conservative_dimension_stride_bytes",
                &self.conservative_dimension_stride_bytes,
            )
            .field("accountable_charged_bytes", &self.accountable_charged_bytes)
            .finish_non_exhaustive()
    }
}

impl BoundedTensorHostAllocationReceiptV1<'_> {
    #[inline]
    pub fn lower_element_capacity(&self) -> usize {
        self.source.lower.accounted_allocation().element_capacity
    }

    #[inline]
    pub fn upper_element_capacity(&self) -> usize {
        self.source.upper.accounted_allocation().element_capacity
    }

    /// Exact `Vec<f32>` capacity bytes for both endpoints.
    #[inline]
    pub fn exact_payload_capacity_bytes(&self) -> usize {
        self.exact_payload_capacity_bytes
    }

    /// Conservative charge for both endpoints' dimension and stride metadata.
    /// V1 charges two rank-sized `usize` sequences per `ArrayD`, even when the
    /// pinned ndarray representation stores them inline. A ledger that also
    /// charges the inline outer owner may therefore intentionally double-charge
    /// these bytes.
    #[inline]
    pub fn conservative_dimension_stride_bytes(&self) -> usize {
        self.conservative_dimension_stride_bytes
    }

    /// Total V1 charge: exact payload capacity bytes plus the conservative
    /// dimension/stride ownership charge. This is not exact dynamic heap usage
    /// or allocator RSS.
    #[inline]
    pub fn accountable_charged_bytes(&self) -> usize {
        self.accountable_charged_bytes
    }

    /// Whether `source` is the exact object whose borrow this receipt retains.
    #[inline]
    pub fn matches_source(&self, source: &BoundedTensor) -> bool {
        std::ptr::eq(self.source, source)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EndpointAllocationUnsupportedV1 {
    NonCanonicalCLayout,
    RankExceedsV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EndpointAllocationInvalidV1 {
    LayoutArithmeticOverflow,
    ReconstructionRejected,
    RawOffsetInvariant,
    LogicalPointerInvariant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EndpointAllocationAccountedV1 {
    element_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EndpointAllocationStateV1 {
    Accounted(EndpointAllocationAccountedV1),
    Unsupported(EndpointAllocationUnsupportedV1),
    Invalid(EndpointAllocationInvalidV1),
}

pub(super) struct TrackedArrayD {
    value: ArrayD<f32>,
    allocation: EndpointAllocationStateV1,
}

impl fmt::Debug for TrackedArrayD {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(formatter)
    }
}

impl Serialize for TrackedArrayD {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

impl Deref for TrackedArrayD {
    type Target = ArrayD<f32>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl TrackedArrayD {
    pub(super) fn new(value: ArrayD<f32>) -> Self {
        let rank = value.ndim();
        if rank > BOUNDED_TENSOR_HOST_ALLOCATION_MAX_RANK_V1 {
            return Self {
                value,
                allocation: EndpointAllocationStateV1::Unsupported(
                    EndpointAllocationUnsupportedV1::RankExceedsV1,
                ),
            };
        }

        let canonical_layout = match exact_default_c_strides(&value) {
            Ok(canonical) => canonical,
            Err(invalid) => {
                return Self {
                    value,
                    allocation: EndpointAllocationStateV1::Invalid(invalid),
                };
            }
        };
        if !canonical_layout {
            return Self {
                value,
                allocation: EndpointAllocationStateV1::Unsupported(
                    EndpointAllocationUnsupportedV1::NonCanonicalCLayout,
                ),
            };
        }

        let mut saved_shape = [0usize; BOUNDED_TENSOR_HOST_ALLOCATION_MAX_RANK_V1];
        saved_shape[..rank].copy_from_slice(value.shape());
        let shape = &saved_shape[..rank];

        // Validate the exact reshape before consuming ownership. The actual
        // post-consumption reshape below has the same one-dimensional logical
        // source and target, so a disagreement is an ndarray invariant bug.
        let Some(logical_values) = value.as_slice() else {
            return Self {
                value,
                allocation: EndpointAllocationStateV1::Invalid(
                    EndpointAllocationInvalidV1::ReconstructionRejected,
                ),
            };
        };
        if ndarray::ArrayView1::from(logical_values)
            .into_shape_with_order(IxDyn(shape))
            .is_err()
        {
            return Self {
                value,
                allocation: EndpointAllocationStateV1::Invalid(
                    EndpointAllocationInvalidV1::ReconstructionRejected,
                ),
            };
        }

        let logical_len = value.len();
        let logical_ptr = value.as_ptr();
        let (raw, reported_offset) = value.into_raw_vec_and_offset();
        let raw_ptr = raw.as_ptr();
        let raw_len = raw.len();
        let element_capacity = raw.capacity();

        let mut first_invalid = None;
        let (slice_start, slice_end) = if logical_len == 0 {
            if reported_offset.is_some() {
                first_invalid = Some(EndpointAllocationInvalidV1::RawOffsetInvariant);
            }
            // ndarray defines an empty owned array's raw offset as None. The
            // logical pointer is therefore non-authoritative; canonicalize it
            // to the allocation base while retaining the full raw Vec.
            (0, 0)
        } else {
            let logical_addr = logical_ptr as usize;
            let raw_addr = raw_ptr as usize;
            let derived_offset = logical_addr
                .checked_sub(raw_addr)
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
                // Safe ndarray ownership promises that a nonempty logical C
                // span lies inside the returned Vec. If neither independently
                // derived offset can recover that span, the consumed owner
                // cannot be reconstructed without fabricating values. This is
                // the one narrow fail-stop boundary, not a fallback outcome.
                panic!("BoundedTensor endpoint lost its reconstructible nonempty raw span")
            });
            if reported_span != Some(usable_span) || derived_span != Some(usable_span) {
                first_invalid = Some(EndpointAllocationInvalidV1::RawOffsetInvariant);
            }
            usable_span
        };

        let linear = Array1::from_vec(raw);
        let sliced = linear.slice_axis_move(Axis(0), Slice::from(slice_start..slice_end));
        let rebuilt = sliced
            .into_shape_with_order(IxDyn(shape))
            .unwrap_or_else(|error| {
                // An identical borrowed one-dimensional reshape was accepted
                // above. ndarray's consuming API does not return the owner on
                // error, so disagreement here is an impossible fail-stop
                // library invariant, never a clean or typed fallback result.
                panic!("prevalidated BoundedTensor endpoint reconstruction failed: {error}")
            });

        if rebuilt.shape() != shape || !exact_default_c_strides(&rebuilt).unwrap_or(false) {
            first_invalid.get_or_insert(EndpointAllocationInvalidV1::ReconstructionRejected);
        }
        if logical_len != 0 && rebuilt.as_ptr() != logical_ptr {
            first_invalid.get_or_insert(EndpointAllocationInvalidV1::LogicalPointerInvariant);
        }

        let allocation = match first_invalid {
            Some(invalid) => EndpointAllocationStateV1::Invalid(invalid),
            None => EndpointAllocationStateV1::Accounted(EndpointAllocationAccountedV1 {
                element_capacity,
            }),
        };
        Self {
            value: rebuilt,
            allocation,
        }
    }

    #[inline]
    pub(super) fn as_array(&self) -> &ArrayD<f32> {
        &self.value
    }

    #[inline]
    pub(super) fn into_array(self) -> ArrayD<f32> {
        self.value
    }

    #[inline]
    pub(super) fn allocation(&self) -> EndpointAllocationStateV1 {
        self.allocation
    }

    #[inline]
    fn accounted_allocation(&self) -> EndpointAllocationAccountedV1 {
        match self.allocation {
            EndpointAllocationStateV1::Accounted(receipt) => receipt,
            EndpointAllocationStateV1::Unsupported(_) | EndpointAllocationStateV1::Invalid(_) => {
                unreachable!("accounted BoundedTensor receipt references an ineligible endpoint")
            }
        }
    }

    #[inline]
    pub(super) fn fill(&mut self, value: f32) {
        self.value.fill(value);
    }

    #[inline]
    pub(super) fn mapv_inplace(&mut self, map: impl FnMut(f32) -> f32) {
        self.value.mapv_inplace(map);
    }

    #[inline]
    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.value.iter_mut()
    }

    #[inline]
    pub(super) fn fill_axis_index(&mut self, axis: Axis, index: usize, value: f32) {
        self.value.index_axis_mut(axis, index).fill(value);
    }
}

fn exact_default_c_strides(value: &ArrayD<f32>) -> Result<bool, EndpointAllocationInvalidV1> {
    let shape = value.shape();
    let strides = value.strides();
    if shape.contains(&0) {
        return Ok(strides.iter().all(|&stride| stride == 0));
    }

    let mut expected = 1usize;
    for (&dimension, &stride) in shape.iter().zip(strides).rev() {
        let expected_stride = isize::try_from(expected)
            .map_err(|_| EndpointAllocationInvalidV1::LayoutArithmeticOverflow)?;
        if stride != expected_stride {
            return Ok(false);
        }
        expected = expected
            .checked_mul(dimension)
            .ok_or(EndpointAllocationInvalidV1::LayoutArithmeticOverflow)?;
    }
    Ok(true)
}

impl BoundedTensor {
    /// Return non-authorizing V1 allocation provenance for this exact tensor.
    ///
    /// No partial/lower-bound byte count is returned. Unsupported layout,
    /// rank, or L2 ownership produces `Unsupported`; arithmetic or ownership
    /// invariant failures produce `Invalid` and must fail closed.
    ///
    /// The accounted receipt retains this tensor's borrow, preventing a setter
    /// or other mutable operation while the receipt remains live.
    ///
    /// ```compile_fail
    /// use ny_tensor::BoundedTensor;
    ///
    /// let mut bounds = BoundedTensor::new_conservative(&[1]);
    /// let provenance = bounds.host_allocation_provenance_v1();
    /// bounds.clear_l2_constraint();
    /// drop(provenance);
    /// ```
    pub fn host_allocation_provenance_v1(&self) -> BoundedTensorHostAllocationProvenanceV1<'_> {
        let lower = self.lower.allocation();
        let upper = self.upper.allocation();

        if let EndpointAllocationStateV1::Invalid(reason) = lower {
            return BoundedTensorHostAllocationProvenanceV1::Invalid(map_invalid(
                BoundedTensorHostAllocationEndpointV1::Lower,
                reason,
            ));
        }
        if let EndpointAllocationStateV1::Invalid(reason) = upper {
            return BoundedTensorHostAllocationProvenanceV1::Invalid(map_invalid(
                BoundedTensorHostAllocationEndpointV1::Upper,
                reason,
            ));
        }
        if self.l2.is_some() {
            return BoundedTensorHostAllocationProvenanceV1::Unsupported(
                BoundedTensorHostAllocationUnsupportedV1::L2ConstraintPresent,
            );
        }
        if let EndpointAllocationStateV1::Unsupported(reason) = lower {
            return BoundedTensorHostAllocationProvenanceV1::Unsupported(map_unsupported(
                BoundedTensorHostAllocationEndpointV1::Lower,
                reason,
                self.lower.ndim(),
            ));
        }
        if let EndpointAllocationStateV1::Unsupported(reason) = upper {
            return BoundedTensorHostAllocationProvenanceV1::Unsupported(map_unsupported(
                BoundedTensorHostAllocationEndpointV1::Upper,
                reason,
                self.upper.ndim(),
            ));
        }

        let EndpointAllocationStateV1::Accounted(lower) = lower else {
            unreachable!()
        };
        let EndpointAllocationStateV1::Accounted(upper) = upper else {
            unreachable!()
        };
        let (lower_payload_bytes, lower_dimension_stride_bytes) = match endpoint_charges(
            BoundedTensorHostAllocationEndpointV1::Lower,
            lower.element_capacity,
            self.lower.ndim(),
        ) {
            Ok(charges) => charges,
            Err(invalid) => return BoundedTensorHostAllocationProvenanceV1::Invalid(invalid),
        };
        let (upper_payload_bytes, upper_dimension_stride_bytes) = match endpoint_charges(
            BoundedTensorHostAllocationEndpointV1::Upper,
            upper.element_capacity,
            self.upper.ndim(),
        ) {
            Ok(charges) => charges,
            Err(invalid) => return BoundedTensorHostAllocationProvenanceV1::Invalid(invalid),
        };
        let Some(exact_payload_capacity_bytes) =
            lower_payload_bytes.checked_add(upper_payload_bytes)
        else {
            return BoundedTensorHostAllocationProvenanceV1::Invalid(
                BoundedTensorHostAllocationInvalidV1::TotalChargedBytesOverflow,
            );
        };
        let Some(conservative_dimension_stride_bytes) =
            lower_dimension_stride_bytes.checked_add(upper_dimension_stride_bytes)
        else {
            return BoundedTensorHostAllocationProvenanceV1::Invalid(
                BoundedTensorHostAllocationInvalidV1::TotalChargedBytesOverflow,
            );
        };
        let Some(accountable_charged_bytes) =
            exact_payload_capacity_bytes.checked_add(conservative_dimension_stride_bytes)
        else {
            return BoundedTensorHostAllocationProvenanceV1::Invalid(
                BoundedTensorHostAllocationInvalidV1::TotalChargedBytesOverflow,
            );
        };

        BoundedTensorHostAllocationProvenanceV1::Accounted(BoundedTensorHostAllocationReceiptV1 {
            source: self,
            exact_payload_capacity_bytes,
            conservative_dimension_stride_bytes,
            accountable_charged_bytes,
        })
    }
}

fn endpoint_charges(
    endpoint: BoundedTensorHostAllocationEndpointV1,
    element_capacity: usize,
    rank: usize,
) -> Result<(usize, usize), BoundedTensorHostAllocationInvalidV1> {
    let payload_capacity_bytes = element_capacity
        .checked_mul(size_of::<f32>())
        .ok_or(BoundedTensorHostAllocationInvalidV1::EndpointPayloadBytesOverflow { endpoint })?;
    let dimension_stride_bytes = rank
        .checked_mul(size_of::<usize>())
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or(
            BoundedTensorHostAllocationInvalidV1::EndpointDimensionStrideBytesOverflow { endpoint },
        )?;
    Ok((payload_capacity_bytes, dimension_stride_bytes))
}

fn map_unsupported(
    endpoint: BoundedTensorHostAllocationEndpointV1,
    reason: EndpointAllocationUnsupportedV1,
    rank: usize,
) -> BoundedTensorHostAllocationUnsupportedV1 {
    match reason {
        EndpointAllocationUnsupportedV1::NonCanonicalCLayout => {
            BoundedTensorHostAllocationUnsupportedV1::EndpointNonCanonicalCLayout { endpoint }
        }
        EndpointAllocationUnsupportedV1::RankExceedsV1 => {
            BoundedTensorHostAllocationUnsupportedV1::EndpointRankExceedsV1 { endpoint, rank }
        }
    }
}

fn map_invalid(
    endpoint: BoundedTensorHostAllocationEndpointV1,
    reason: EndpointAllocationInvalidV1,
) -> BoundedTensorHostAllocationInvalidV1 {
    match reason {
        EndpointAllocationInvalidV1::LayoutArithmeticOverflow => {
            BoundedTensorHostAllocationInvalidV1::EndpointLayoutArithmeticOverflow { endpoint }
        }
        EndpointAllocationInvalidV1::ReconstructionRejected => {
            BoundedTensorHostAllocationInvalidV1::EndpointReconstructionRejected { endpoint }
        }
        EndpointAllocationInvalidV1::RawOffsetInvariant => {
            BoundedTensorHostAllocationInvalidV1::EndpointRawOffsetInvariant { endpoint }
        }
        EndpointAllocationInvalidV1::LogicalPointerInvariant => {
            BoundedTensorHostAllocationInvalidV1::EndpointLogicalPointerInvariant { endpoint }
        }
    }
}

#[cfg(test)]
pub(super) fn invalid_state_for_test(reason: EndpointAllocationInvalidV1) -> TrackedArrayD {
    TrackedArrayD {
        value: ArrayD::zeros(IxDyn(&[1])),
        allocation: EndpointAllocationStateV1::Invalid(reason),
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use std::ops::Range;

    use ndarray::{s, Array1, ArrayD, IxDyn, ShapeBuilder};

    use super::*;
    use crate::bounded_tensor::{L2Constraint, RepairStrategy};

    struct RawExpectation {
        values: Vec<f32>,
        base: *const f32,
        capacity: usize,
        offset: Option<usize>,
    }

    fn spare_vec(values: &[f32], requested_capacity: usize) -> Vec<f32> {
        let mut values_with_spare = Vec::with_capacity(requested_capacity);
        values_with_spare.extend_from_slice(values);
        values_with_spare
    }

    fn sliced_array(
        values: &[f32],
        requested_capacity: usize,
        logical: Range<usize>,
    ) -> (ArrayD<f32>, RawExpectation) {
        let raw = spare_vec(values, requested_capacity);
        let expectation = RawExpectation {
            values: raw.clone(),
            base: raw.as_ptr(),
            capacity: raw.capacity(),
            offset: Some(logical.start),
        };
        let array = Array1::from_vec(raw)
            .slice_axis_move(Axis(0), Slice::from(logical))
            .into_dyn();
        (array, expectation)
    }

    fn empty_backed_array(
        values: &[f32],
        requested_capacity: usize,
    ) -> (ArrayD<f32>, RawExpectation) {
        let raw = spare_vec(values, requested_capacity);
        let expectation = RawExpectation {
            values: raw.clone(),
            base: raw.as_ptr(),
            capacity: raw.capacity(),
            offset: None,
        };
        let array = ArrayD::from_shape_vec(IxDyn(&[0]).strides(IxDyn(&[0])), raw)
            .expect("an empty default-stride shape may retain a full backing Vec");
        (array, expectation)
    }

    fn filled_with_capacity(shape: &[usize], value: f32, requested_capacity: usize) -> ArrayD<f32> {
        let elements = shape.iter().product();
        let mut values = Vec::with_capacity(requested_capacity);
        values.resize(elements, value);
        ArrayD::from_shape_vec(IxDyn(shape), values).expect("valid fixture shape")
    }

    fn accounted(bounds: &BoundedTensor) -> BoundedTensorHostAllocationReceiptV1<'_> {
        match bounds.host_allocation_provenance_v1() {
            BoundedTensorHostAllocationProvenanceV1::Accounted(receipt) => receipt,
            other => panic!("expected accounted provenance, got {other:?}"),
        }
    }

    fn unsupported(bounds: &BoundedTensor) -> BoundedTensorHostAllocationUnsupportedV1 {
        match bounds.host_allocation_provenance_v1() {
            BoundedTensorHostAllocationProvenanceV1::Unsupported(reason) => reason,
            other => panic!("expected unsupported provenance, got {other:?}"),
        }
    }

    fn assert_raw(array: ArrayD<f32>, expected: RawExpectation) {
        let (raw, offset) = array.into_raw_vec_and_offset();
        assert_eq!(raw, expected.values);
        assert_eq!(raw.as_ptr(), expected.base);
        assert_eq!(raw.capacity(), expected.capacity);
        assert_eq!(offset, expected.offset);
    }

    struct TensorSnapshot {
        lower: ArrayD<f32>,
        upper: ArrayD<f32>,
        lower_ptr: *const f32,
        upper_ptr: *const f32,
        lower_capacity: usize,
        upper_capacity: usize,
    }

    fn snapshot(bounds: &BoundedTensor) -> TensorSnapshot {
        let (lower_capacity, upper_capacity) = {
            let receipt = accounted(bounds);
            (
                receipt.lower_element_capacity(),
                receipt.upper_element_capacity(),
            )
        };
        TensorSnapshot {
            lower: bounds.lower().clone(),
            upper: bounds.upper().clone(),
            lower_ptr: bounds.lower().as_ptr(),
            upper_ptr: bounds.upper().as_ptr(),
            lower_capacity,
            upper_capacity,
        }
    }

    fn assert_unchanged(bounds: &BoundedTensor, snapshot: &TensorSnapshot) {
        assert_eq!(bounds.lower(), &snapshot.lower);
        assert_eq!(bounds.upper(), &snapshot.upper);
        assert_eq!(bounds.lower().as_ptr(), snapshot.lower_ptr);
        assert_eq!(bounds.upper().as_ptr(), snapshot.upper_ptr);
        let receipt = accounted(bounds);
        assert_eq!(receipt.lower_element_capacity(), snapshot.lower_capacity);
        assert_eq!(receipt.upper_element_capacity(), snapshot.upper_capacity);
    }

    #[test]
    fn nonzero_offset_reseal_preserves_full_vec_pointer_len_capacity_and_values() {
        let (lower, lower_raw) =
            sliced_array(&[-90.0, -80.0, -3.0, -2.0, -1.0, 0.0, 70.0], 17, 2..6);
        let (upper, upper_raw) =
            sliced_array(&[90.0, 80.0, 70.0, 1.0, 2.0, 3.0, 4.0, 60.0], 23, 3..7);
        let lower_logical_ptr = lower.as_ptr();
        let upper_logical_ptr = upper.as_ptr();

        let bounds = BoundedTensor::new(lower, upper).expect("ordered finite endpoints");
        assert_eq!(
            bounds.lower(),
            &Array1::from_vec(vec![-3.0, -2.0, -1.0, 0.0]).into_dyn()
        );
        assert_eq!(
            bounds.upper(),
            &Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]).into_dyn()
        );
        assert_eq!(bounds.lower().as_ptr(), lower_logical_ptr);
        assert_eq!(bounds.upper().as_ptr(), upper_logical_ptr);

        {
            let receipt = accounted(&bounds);
            assert!(receipt.matches_source(&bounds));
            assert_eq!(receipt.lower_element_capacity(), lower_raw.capacity);
            assert_eq!(receipt.upper_element_capacity(), upper_raw.capacity);
            assert_eq!(
                receipt.exact_payload_capacity_bytes(),
                (lower_raw.capacity + upper_raw.capacity) * size_of::<f32>()
            );
            assert_eq!(
                receipt.conservative_dimension_stride_bytes(),
                4 * size_of::<usize>()
            );
            assert_eq!(
                receipt.accountable_charged_bytes(),
                receipt.exact_payload_capacity_bytes()
                    + receipt.conservative_dimension_stride_bytes()
            );
        }

        let (lower, upper) = bounds.into_parts();
        assert_raw(lower, lower_raw);
        assert_raw(upper, upper_raw);
    }

    #[test]
    fn empty_none_offset_reseal_preserves_full_backing_vec_and_capacity() {
        let (lower, lower_raw) = empty_backed_array(&[-3.0, -2.0, -1.0], 19);
        let (upper, upper_raw) = empty_backed_array(&[1.0, 2.0], 29);
        assert_eq!(lower.strides(), &[0]);
        assert_eq!(upper.strides(), &[0]);

        let bounds = BoundedTensor::new(lower, upper).expect("empty bounds are valid");
        {
            let receipt = accounted(&bounds);
            assert_eq!(receipt.lower_element_capacity(), lower_raw.capacity);
            assert_eq!(receipt.upper_element_capacity(), upper_raw.capacity);
        }

        // Empty `as_ptr()` is deliberately non-authoritative. Raw allocation
        // identity, initialized contents, length, capacity, and None offset are
        // nevertheless all preserved.
        let (lower, upper) = bounds.into_parts();
        assert_raw(lower, lower_raw);
        assert_raw(upper, upper_raw);
    }

    #[test]
    fn canonical_scalar_and_empty_shapes_through_rank_four_are_panic_free_and_accounted() {
        for shape in [
            &[][..],
            &[0][..],
            &[2][..],
            &[0, 3][..],
            &[2, 0, 3][..],
            &[1, 2, 0, 3][..],
            &[1, 1, 2, 3][..],
        ] {
            let bounds = BoundedTensor::new(
                ArrayD::from_elem(IxDyn(shape), -1.0),
                ArrayD::from_elem(IxDyn(shape), 1.0),
            )
            .unwrap();
            let receipt = accounted(&bounds);
            assert_eq!(bounds.shape(), shape);
            assert_eq!(
                receipt.conservative_dimension_stride_bytes(),
                2 * 2 * shape.len() * size_of::<usize>()
            );
        }
    }

    #[test]
    fn bounded_tensor_inline_provenance_footprint_is_pinned() {
        let legacy_owner_bytes =
            2 * size_of::<ArrayD<f32>>() + size_of::<Option<Box<L2Constraint>>>();
        assert_eq!(
            size_of::<EndpointAllocationAccountedV1>(),
            size_of::<usize>()
        );
        assert_eq!(
            size_of::<TrackedArrayD>(),
            size_of::<ArrayD<f32>>() + size_of::<EndpointAllocationStateV1>()
        );
        assert_eq!(
            size_of::<BoundedTensor>(),
            legacy_owner_bytes + 2 * size_of::<EndpointAllocationStateV1>()
        );
        #[cfg(target_pointer_width = "64")]
        {
            // The pre-provenance owner was 232 bytes. One compact, closed
            // 16-byte status/capacity state per endpoint makes V1 264 bytes.
            assert_eq!(legacy_owner_bytes, 232);
            assert_eq!(size_of::<EndpointAllocationStateV1>(), 16);
            assert_eq!(size_of::<BoundedTensor>(), 264);
        }
    }

    #[test]
    fn rank_four_nonzero_offset_reseal_preserves_shape_strides_and_backing_vec() {
        let (lower, lower_raw) = sliced_array(&[-90.0, -4.0, -3.0, -2.0, -1.0, 90.0], 19, 1..5);
        let lower = lower.into_shape_with_order(IxDyn(&[1, 1, 2, 2])).unwrap();
        let (upper, upper_raw) = sliced_array(&[90.0, 1.0, 2.0, 3.0, 4.0, 90.0], 23, 1..5);
        let upper = upper.into_shape_with_order(IxDyn(&[1, 1, 2, 2])).unwrap();
        let lower_ptr = lower.as_ptr();
        let upper_ptr = upper.as_ptr();

        let bounds = BoundedTensor::new(lower, upper).unwrap();
        assert_eq!(bounds.shape(), &[1, 1, 2, 2]);
        assert_eq!(bounds.lower().strides(), &[4, 4, 2, 1]);
        assert_eq!(bounds.upper().strides(), &[4, 4, 2, 1]);
        assert_eq!(bounds.lower().as_ptr(), lower_ptr);
        assert_eq!(bounds.upper().as_ptr(), upper_ptr);
        let _ = accounted(&bounds);

        let (lower, upper) = bounds.into_parts();
        assert_raw(lower, lower_raw);
        assert_raw(upper, upper_raw);
    }

    #[test]
    fn payload_capacity_not_logical_length_and_rank_charge_is_conservative() {
        let lower_raw = spare_vec(&[-1.0; 6], 17);
        let lower_capacity = lower_raw.capacity();
        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 6]), lower_raw).unwrap();
        let upper_raw = spare_vec(&[1.0; 6], 23);
        let upper_capacity = upper_raw.capacity();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 6]), upper_raw).unwrap();
        let bounds = BoundedTensor::new(lower, upper).unwrap();
        let receipt = accounted(&bounds);

        // Read the actual source capacities rather than assuming the allocator
        // returns exactly the requested amount.
        assert_eq!(receipt.lower_element_capacity(), lower_capacity);
        assert_eq!(receipt.upper_element_capacity(), upper_capacity);
        assert_ne!(receipt.exact_payload_capacity_bytes(), bounds.len() * 2 * 4);
        assert_eq!(
            receipt.conservative_dimension_stride_bytes(),
            2 * 2 * 4 * size_of::<usize>()
        );
    }

    #[test]
    fn rank_five_and_eight_are_cleanly_unsupported_and_left_untouched() {
        for shape in [&[1, 1, 1, 1, 2][..], &[1, 1, 1, 1, 1, 1, 1, 2][..]] {
            let lower = filled_with_capacity(shape, -1.0, 13);
            let upper = filled_with_capacity(shape, 1.0, 17);
            let lower_ptr = lower.as_ptr();
            let lower_strides = lower.strides().to_vec();
            let rank = shape.len();
            let bounds = BoundedTensor::new(lower, upper).unwrap();

            assert_eq!(
                unsupported(&bounds),
                BoundedTensorHostAllocationUnsupportedV1::EndpointRankExceedsV1 {
                    endpoint: BoundedTensorHostAllocationEndpointV1::Lower,
                    rank,
                }
            );
            assert_eq!(bounds.lower().as_ptr(), lower_ptr);
            assert_eq!(bounds.lower().strides(), lower_strides);
        }
    }

    #[test]
    fn noncanonical_layouts_are_unsupported_without_lower_bound_accounting() {
        let singleton_custom =
            ArrayD::from_shape_vec(IxDyn(&[1, 2]).strides(IxDyn(&[99, 1])), vec![-1.0, -1.0])
                .unwrap();
        assert!(singleton_custom.is_standard_layout());
        let singleton_strides = singleton_custom.strides().to_vec();
        let singleton_ptr = singleton_custom.as_ptr();
        let singleton =
            BoundedTensor::new(singleton_custom, ArrayD::from_elem(IxDyn(&[1, 2]), 1.0)).unwrap();
        assert_eq!(
            unsupported(&singleton),
            BoundedTensorHostAllocationUnsupportedV1::EndpointNonCanonicalCLayout {
                endpoint: BoundedTensorHostAllocationEndpointV1::Lower,
            }
        );
        assert_eq!(singleton.lower().strides(), singleton_strides);
        assert_eq!(singleton.lower().as_ptr(), singleton_ptr);

        let fortran = ArrayD::from_shape_vec(IxDyn(&[2, 2]).f(), vec![-1.0; 4]).unwrap();
        let fortran_strides = fortran.strides().to_vec();
        let fortran_bounds =
            BoundedTensor::new(fortran, ArrayD::from_elem(IxDyn(&[2, 2]), 1.0)).unwrap();
        assert!(matches!(
            unsupported(&fortran_bounds),
            BoundedTensorHostAllocationUnsupportedV1::EndpointNonCanonicalCLayout { .. }
        ));
        assert_eq!(fortran_bounds.lower().strides(), fortran_strides);

        let reversed = Array1::from_vec(vec![-1.0, -2.0, -3.0, -4.0])
            .slice_move(s![..;-1])
            .into_dyn();
        let reversed_strides = reversed.strides().to_vec();
        let reversed_bounds =
            BoundedTensor::new(reversed, ArrayD::from_elem(IxDyn(&[4]), 1.0)).unwrap();
        assert!(matches!(
            unsupported(&reversed_bounds),
            BoundedTensorHostAllocationUnsupportedV1::EndpointNonCanonicalCLayout { .. }
        ));
        assert_eq!(reversed_bounds.lower().strides(), reversed_strides);

        let custom =
            ArrayD::from_shape_vec(IxDyn(&[2, 2]).strides(IxDyn(&[3, 1])), vec![-1.0; 5]).unwrap();
        let custom_strides = custom.strides().to_vec();
        let custom_bounds =
            BoundedTensor::new(custom, ArrayD::from_elem(IxDyn(&[2, 2]), 1.0)).unwrap();
        assert!(matches!(
            unsupported(&custom_bounds),
            BoundedTensorHostAllocationUnsupportedV1::EndpointNonCanonicalCLayout { .. }
        ));
        assert_eq!(custom_bounds.lower().strides(), custom_strides);
    }

    #[test]
    fn setters_refresh_capacity_atomically_and_can_transition_to_unsupported() {
        let mut bounds = BoundedTensor::new(
            filled_with_capacity(&[2], 0.0, 11),
            filled_with_capacity(&[2], 10.0, 13),
        )
        .unwrap();
        let (before_lower_capacity, upper_capacity) = {
            let before = accounted(&bounds);
            (
                before.lower_element_capacity(),
                before.upper_element_capacity(),
            )
        };

        let (replacement, replacement_raw) = sliced_array(&[-50.0, -2.0, 2.0, 50.0], 31, 1..3);
        bounds.set_lower(replacement).unwrap();
        let refreshed = accounted(&bounds);
        assert_eq!(refreshed.lower_element_capacity(), replacement_raw.capacity);
        assert_eq!(refreshed.upper_element_capacity(), upper_capacity);
        assert_ne!(refreshed.lower_element_capacity(), before_lower_capacity);
        let _ = refreshed;

        let lower_ptr = bounds.lower().as_ptr();
        let snapshot = bounds.lower().clone();
        let snapshot_capacity = accounted(&bounds).lower_element_capacity();
        assert!(bounds
            .set_lower(ArrayD::from_elem(IxDyn(&[2]), 20.0))
            .is_err());
        assert_eq!(bounds.lower(), &snapshot);
        assert_eq!(bounds.lower().as_ptr(), lower_ptr);
        assert_eq!(
            accounted(&bounds).lower_element_capacity(),
            snapshot_capacity
        );

        let noncanonical = Array1::from_vec(vec![-3.0, -2.0])
            .slice_move(s![..;-1])
            .into_dyn();
        bounds.set_lower(noncanonical).unwrap();
        assert!(matches!(
            unsupported(&bounds),
            BoundedTensorHostAllocationUnsupportedV1::EndpointNonCanonicalCLayout {
                endpoint: BoundedTensorHostAllocationEndpointV1::Lower,
            }
        ));
    }

    #[test]
    fn every_setter_validation_failure_preserves_values_owners_and_receipts() {
        let mut bounds = BoundedTensor::new(
            filled_with_capacity(&[2], 0.0, 11),
            filled_with_capacity(&[2], 10.0, 13),
        )
        .unwrap();

        for rejected in [
            ArrayD::zeros(IxDyn(&[1])),
            Array1::from_vec(vec![f32::NAN, 0.0]).into_dyn(),
            Array1::from_vec(vec![f32::INFINITY, 0.0]).into_dyn(),
            Array1::from_vec(vec![11.0, 0.0]).into_dyn(),
        ] {
            let before = snapshot(&bounds);
            assert!(bounds.set_lower(rejected).is_err());
            assert_unchanged(&bounds, &before);
        }

        for rejected in [
            ArrayD::zeros(IxDyn(&[1])),
            Array1::from_vec(vec![f32::NAN, 10.0]).into_dyn(),
            Array1::from_vec(vec![f32::INFINITY, 10.0]).into_dyn(),
            Array1::from_vec(vec![-1.0, 10.0]).into_dyn(),
        ] {
            let before = snapshot(&bounds);
            assert!(bounds.set_upper(rejected).is_err());
            assert_unchanged(&bounds, &before);
        }
    }

    #[test]
    fn clone_and_clone_from_reseal_the_actual_new_allocations() {
        let (lower, _) = sliced_array(&[-9.0, -2.0, -1.0, 9.0], 29, 1..3);
        let (upper, _) = sliced_array(&[9.0, 2.0, 3.0, 9.0], 31, 1..3);
        let source = BoundedTensor::new(lower, upper).unwrap();
        let source_capacities = {
            let receipt = accounted(&source);
            (
                receipt.lower_element_capacity(),
                receipt.upper_element_capacity(),
            )
        };

        let cloned = source.clone();
        assert_ne!(cloned.lower().as_ptr(), source.lower().as_ptr());
        let clone_capacities = {
            let receipt = accounted(&cloned);
            (
                receipt.lower_element_capacity(),
                receipt.upper_element_capacity(),
            )
        };
        let (cloned_lower, cloned_upper) = cloned.into_parts();
        assert_eq!(
            cloned_lower.into_raw_vec_and_offset().0.capacity(),
            clone_capacities.0
        );
        assert_eq!(
            cloned_upper.into_raw_vec_and_offset().0.capacity(),
            clone_capacities.1
        );
        assert_eq!(
            (
                accounted(&source).lower_element_capacity(),
                accounted(&source).upper_element_capacity(),
            ),
            source_capacities
        );

        let mut target = BoundedTensor::new_conservative(&[2]);
        target.clone_from(&source);
        let target_capacities = {
            let receipt = accounted(&target);
            (
                receipt.lower_element_capacity(),
                receipt.upper_element_capacity(),
            )
        };
        let (target_lower, target_upper) = target.into_parts();
        assert_eq!(
            target_lower.into_raw_vec_and_offset().0.capacity(),
            target_capacities.0
        );
        assert_eq!(
            target_upper.into_raw_vec_and_offset().0.capacity(),
            target_capacities.1
        );
    }

    #[test]
    fn deserialize_preserves_wire_shape_and_captures_decoded_capacity() {
        let (lower, _) = sliced_array(&[-9.0, -2.0, -1.0, 9.0], 37, 1..3);
        let (upper, _) = sliced_array(&[9.0, 2.0, 3.0, 9.0], 41, 1..3);
        let source = BoundedTensor::new(lower, upper).unwrap();
        let encoded = serde_json::to_string(&source).unwrap();
        let object = serde_json::from_str::<serde_json::Value>(&encoded).unwrap();
        let keys = object.as_object().unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains_key("lower"));
        assert!(keys.contains_key("upper"));

        let decoded: BoundedTensor = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.lower(), source.lower());
        assert_eq!(decoded.upper(), source.upper());
        let decoded_capacities = {
            let receipt = accounted(&decoded);
            (
                receipt.lower_element_capacity(),
                receipt.upper_element_capacity(),
            )
        };
        let (lower, upper) = decoded.into_parts();
        let (lower_raw, lower_offset) = lower.into_raw_vec_and_offset();
        let (upper_raw, upper_offset) = upper.into_raw_vec_and_offset();
        assert_eq!(lower_raw.capacity(), decoded_capacities.0);
        assert_eq!(upper_raw.capacity(), decoded_capacities.1);
        assert_eq!(lower_offset, Some(0));
        assert_eq!(upper_offset, Some(0));
    }

    #[test]
    fn l2_is_unsupported_without_discarding_endpoint_receipts() {
        let bounds = BoundedTensor::new(
            filled_with_capacity(&[2], -1.0, 11),
            filled_with_capacity(&[2], 1.0, 13),
        )
        .unwrap();
        let capacities = {
            let receipt = accounted(&bounds);
            (
                receipt.lower_element_capacity(),
                receipt.upper_element_capacity(),
            )
        };
        let constraint = L2Constraint::new(
            ArrayD::zeros(IxDyn(&[2])),
            ArrayD::from_elem(IxDyn(&[]), 1.0),
            0,
            &[2],
        )
        .unwrap();
        let mut bounds = bounds.with_l2_constraint(constraint);
        assert_eq!(
            unsupported(&bounds),
            BoundedTensorHostAllocationUnsupportedV1::L2ConstraintPresent
        );
        bounds.clear_l2_constraint();
        let receipt = accounted(&bounds);
        assert_eq!(
            (
                receipt.lower_element_capacity(),
                receipt.upper_element_capacity(),
            ),
            capacities
        );
    }

    #[test]
    fn representative_constructor_numeric_and_shape_paths_are_accounted() {
        let base = Array1::from_vec(vec![-1.0, 0.0]).into_dyn();
        let upper = Array1::from_vec(vec![1.0, 2.0]).into_dyn();
        let mut values = vec![
            BoundedTensor::new(base.clone(), upper.clone()).unwrap(),
            BoundedTensor::new_allow_infinite(base.clone(), upper.clone()).unwrap(),
            BoundedTensor::concrete(base.clone()).unwrap(),
            BoundedTensor::from_epsilon(base.clone(), 0.5).unwrap(),
            BoundedTensor::new_conservative(&[2]),
            BoundedTensor::new_repaired(base, upper, RepairStrategy::Conservative).unwrap(),
        ];
        values.push(values[0].round_for_soundness());
        values.push(values[0].reshape(&[1, 2]).unwrap());
        values.push(values[0].flatten());
        for value in &values {
            let receipt = accounted(value);
            assert!(receipt.accountable_charged_bytes() >= receipt.exact_payload_capacity_bytes());
        }
    }

    #[test]
    fn every_bounded_tensor_constructor_records_the_published_allocations() {
        let lower = Array1::from_vec(vec![-1.0, 0.0]).into_dyn();
        let upper = Array1::from_vec(vec![1.0, 2.0]).into_dyn();
        let concrete = Array1::from_vec(vec![0.0, 1.0]).into_dyn();
        let base = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let constructors = vec![
            BoundedTensor::new(lower.clone(), upper.clone()).unwrap(),
            BoundedTensor::new_allow_infinite(lower.clone(), upper.clone()).unwrap(),
            BoundedTensor::new_allow_infinite_with_poll(lower.clone(), upper.clone(), || Ok(()))
                .unwrap(),
            BoundedTensor::new_conservative(&[2]),
            BoundedTensor::new_conservative_with_poll(&[2], || Ok(())).unwrap(),
            BoundedTensor::concrete(concrete.clone()).unwrap(),
            BoundedTensor::concrete_with_poll(concrete, || Ok(())).unwrap(),
            BoundedTensor::from_epsilon(lower.clone(), 0.5).unwrap(),
            BoundedTensor::from_parts_unchecked(lower.clone(), upper.clone()),
            BoundedTensor::new_unchecked(lower.clone(), upper.clone()).unwrap(),
            BoundedTensor::new_sanitized(lower.clone(), upper.clone(), 10.0).unwrap(),
            base.sanitize(10.0),
            BoundedTensor::new_repaired(lower.clone(), upper.clone(), RepairStrategy::Strict)
                .unwrap(),
            BoundedTensor::new_repaired(lower.clone(), upper.clone(), RepairStrategy::Conservative)
                .unwrap(),
            BoundedTensor::new_repaired(lower.clone(), upper.clone(), RepairStrategy::Widen)
                .unwrap(),
            BoundedTensor::new_repaired_with_poll(
                lower.clone(),
                upper.clone(),
                RepairStrategy::Strict,
                || Ok(()),
            )
            .unwrap(),
            BoundedTensor::new_repaired_with_poll(
                lower,
                upper,
                RepairStrategy::Conservative,
                || Ok(()),
            )
            .unwrap(),
        ];

        for bounds in constructors {
            let capacities = {
                let receipt = accounted(&bounds);
                (
                    receipt.lower_element_capacity(),
                    receipt.upper_element_capacity(),
                )
            };
            let (lower, upper) = bounds.into_parts();
            assert_eq!(lower.into_raw_vec_and_offset().0.capacity(), capacities.0);
            assert_eq!(upper.into_raw_vec_and_offset().0.capacity(), capacities.1);
        }
    }

    #[test]
    fn every_reachable_injected_invariant_maps_to_hard_invalid() {
        let cases = [
            (
                EndpointAllocationInvalidV1::LayoutArithmeticOverflow,
                BoundedTensorHostAllocationInvalidV1::EndpointLayoutArithmeticOverflow {
                    endpoint: BoundedTensorHostAllocationEndpointV1::Lower,
                },
            ),
            (
                EndpointAllocationInvalidV1::ReconstructionRejected,
                BoundedTensorHostAllocationInvalidV1::EndpointReconstructionRejected {
                    endpoint: BoundedTensorHostAllocationEndpointV1::Lower,
                },
            ),
            (
                EndpointAllocationInvalidV1::RawOffsetInvariant,
                BoundedTensorHostAllocationInvalidV1::EndpointRawOffsetInvariant {
                    endpoint: BoundedTensorHostAllocationEndpointV1::Lower,
                },
            ),
            (
                EndpointAllocationInvalidV1::LogicalPointerInvariant,
                BoundedTensorHostAllocationInvalidV1::EndpointLogicalPointerInvariant {
                    endpoint: BoundedTensorHostAllocationEndpointV1::Lower,
                },
            ),
        ];
        for (internal, expected) in cases {
            let mut bounds = BoundedTensor::new_conservative(&[1]);
            bounds.lower = invalid_state_for_test(internal);
            assert!(matches!(
                bounds.host_allocation_provenance_v1(),
                BoundedTensorHostAllocationProvenanceV1::Invalid(actual) if actual == expected
            ));
        }
    }

    #[test]
    fn derived_endpoint_charge_overflow_is_hard_invalid() {
        assert_eq!(
            endpoint_charges(BoundedTensorHostAllocationEndpointV1::Lower, usize::MAX, 0),
            Err(
                BoundedTensorHostAllocationInvalidV1::EndpointPayloadBytesOverflow {
                    endpoint: BoundedTensorHostAllocationEndpointV1::Lower,
                }
            )
        );
        assert_eq!(
            endpoint_charges(BoundedTensorHostAllocationEndpointV1::Upper, 0, usize::MAX),
            Err(
                BoundedTensorHostAllocationInvalidV1::EndpointDimensionStrideBytesOverflow {
                    endpoint: BoundedTensorHostAllocationEndpointV1::Upper,
                }
            )
        );
    }

    #[test]
    fn total_charge_overflow_is_hard_invalid_not_saturated() {
        let mut bounds = BoundedTensor::new_conservative(&[1]);
        let EndpointAllocationStateV1::Accounted(mut lower) = bounds.lower.allocation else {
            panic!("fixture must be accounted")
        };
        let EndpointAllocationStateV1::Accounted(mut upper) = bounds.upper.allocation else {
            panic!("fixture must be accounted")
        };
        lower.element_capacity = usize::MAX / size_of::<f32>();
        upper.element_capacity = usize::MAX / size_of::<f32>();
        bounds.lower.allocation = EndpointAllocationStateV1::Accounted(lower);
        bounds.upper.allocation = EndpointAllocationStateV1::Accounted(upper);
        assert!(matches!(
            bounds.host_allocation_provenance_v1(),
            BoundedTensorHostAllocationProvenanceV1::Invalid(
                BoundedTensorHostAllocationInvalidV1::TotalChargedBytesOverflow
            )
        ));
    }
}
