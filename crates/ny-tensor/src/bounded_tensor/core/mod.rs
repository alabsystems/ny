// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core bounded tensor structure and construction helpers.
//!
//! Split into submodules by concern:
//! - `constructors`: Creation, setters, sanitization
//! - `numeric`: Width, rounding, intersection, repair
//! - `shape_ops`: Reshape, flatten, slice, expand, concat, stack

mod allocation_provenance;
pub(crate) mod constructors;
mod inversion_repair;
mod l2;
mod numeric;
mod shape_ops;

pub use super::l2_constraint::L2Constraint;
pub use allocation_provenance::{
    BoundedTensorHostAllocationEndpointV1, BoundedTensorHostAllocationInvalidV1,
    BoundedTensorHostAllocationProvenanceV1, BoundedTensorHostAllocationReceiptV1,
    BoundedTensorHostAllocationUnsupportedV1, BOUNDED_TENSOR_HOST_ALLOCATION_MAX_RANK_V1,
};
pub use constructors::RepairStrategy;
pub use inversion_repair::{repair_inverted_bounds, repair_inverted_bounds_nd, InversionRepair};

use allocation_provenance::TrackedArrayD;
use ndarray::ArrayD;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

/// A tensor where each element has certified lower and upper bounds.
///
/// May optionally carry an [`L2Constraint`] (a per-normalization-slice
/// Euclidean ball) that lets the immediately-downstream `Linear` tighten its box
/// interval via the exact Cauchy–Schwarz row bound. The annotation is purely a
/// soundness-preserving tightening hint: it is `None` by default, is dropped by
/// every value-transforming op (only re-attached deliberately by normalization
/// IBP), and is intersected — never used to widen — at the `Linear`. It is
/// **not** serialized, so it never enters a persisted certificate.
#[derive(Debug, Serialize)]
pub struct BoundedTensor {
    /// Lower bounds for each element.
    lower: TrackedArrayD,
    /// Upper bounds for each element.
    upper: TrackedArrayD,
    /// Optional Euclidean-ball annotation (see [`L2Constraint`]). Not part of the
    /// certificate; skipped during (de)serialization and re-derived on load.
    #[serde(skip)]
    l2: Option<Box<L2Constraint>>,
}

#[derive(Deserialize)]
struct SerializedBoundedTensor {
    lower: ArrayD<f32>,
    upper: ArrayD<f32>,
}

impl<'de> Deserialize<'de> for BoundedTensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serialized = SerializedBoundedTensor::deserialize(deserializer)?;
        let lower = serialized.lower;
        let upper = serialized.upper;

        if lower.shape() != upper.shape() {
            return Err(D::Error::custom(format!(
                "BoundedTensor lower shape {:?} does not match upper shape {:?}",
                lower.shape(),
                upper.shape()
            )));
        }
        if lower.iter().any(|value| value.is_nan()) || upper.iter().any(|value| value.is_nan()) {
            return Err(D::Error::custom(
                "BoundedTensor serialized bounds contain NaN",
            ));
        }

        // `mark_infeasible_*` deliberately uses (+inf, -inf) as the one
        // canonical inverted representation. Reject every other inversion so
        // malformed serialized bounds cannot masquerade as an empty domain.
        let valid = ndarray::Zip::from(&lower)
            .and(&upper)
            .all(|&l, &u| l <= u || (l == f32::INFINITY && u == f32::NEG_INFINITY));
        if !valid {
            return Err(D::Error::custom(
                "BoundedTensor serialized data contains a non-canonical inversion",
            ));
        }

        Ok(Self::from_parts_with_l2(lower, upper, None))
    }
}

impl Clone for BoundedTensor {
    fn clone(&self) -> Self {
        Self::from_parts_with_l2(
            self.lower.as_array().clone(),
            self.upper.as_array().clone(),
            self.l2.clone(),
        )
    }
}

impl BoundedTensor {
    /// Read-only view of the lower bounds.
    #[inline]
    pub fn lower(&self) -> &ArrayD<f32> {
        self.lower.as_array()
    }

    /// Read-only view of the upper bounds.
    #[inline]
    pub fn upper(&self) -> &ArrayD<f32> {
        self.upper.as_array()
    }

    /// Read-only views of both lower and upper bounds.
    #[inline]
    pub fn lower_upper(&self) -> (&ArrayD<f32>, &ArrayD<f32>) {
        (self.lower.as_array(), self.upper.as_array())
    }

    /// Consume the tensor and return owned lower/upper arrays.
    #[inline]
    pub fn into_parts(self) -> (ArrayD<f32>, ArrayD<f32>) {
        (self.lower.into_array(), self.upper.into_array())
    }

    /// Internal constructor for trusted callers within this crate.
    ///
    /// No runtime checks in release builds. Debug builds assert shape match.
    #[inline]
    pub(crate) fn from_parts_unchecked(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Self {
        debug_assert_eq!(
            lower.shape(),
            upper.shape(),
            "from_parts_unchecked: lower shape {:?} != upper shape {:?}",
            lower.shape(),
            upper.shape()
        );
        debug_assert!(
            lower.iter().all(|v| !v.is_nan()),
            "from_parts_unchecked: lower contains NaN"
        );
        debug_assert!(
            upper.iter().all(|v| !v.is_nan()),
            "from_parts_unchecked: upper contains NaN"
        );
        debug_assert!(
            ndarray::Zip::from(&lower)
                .and(&upper)
                .all(|&l, &u| !l.is_finite() || !u.is_finite() || l <= u),
            "from_parts_unchecked: found finite lower > upper (inverted bounds)"
        );
        Self::from_parts_with_l2(lower, upper, None)
    }

    #[inline]
    fn from_parts_with_l2(
        lower: ArrayD<f32>,
        upper: ArrayD<f32>,
        l2: Option<Box<L2Constraint>>,
    ) -> Self {
        Self {
            lower: TrackedArrayD::new(lower),
            upper: TrackedArrayD::new(upper),
            l2,
        }
    }

    /// Check if array contains NaN or Inf values.
    /// Used by constructors and overflow checks to enforce finite bounds.
    #[inline]
    pub(crate) fn has_nan_or_inf(arr: &ArrayD<f32>) -> bool {
        arr.iter().any(|&v| v.is_nan() || v.is_infinite())
    }

    /// Shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.lower.as_array().shape()
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.lower.ndim()
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Check if tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }
}

#[cfg(test)]
mod serde_tests {
    use super::*;
    use ndarray::arr1;

    #[derive(Serialize)]
    struct RawBoundedTensor {
        lower: ArrayD<f32>,
        upper: ArrayD<f32>,
    }

    #[test]
    fn serde_round_trip_preserves_valid_bounds() {
        let bounds =
            BoundedTensor::new(arr1(&[-1.0, 0.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn())
                .expect("valid bounds");

        let encoded = serde_json::to_string(&bounds).expect("serialize");
        let decoded: BoundedTensor = serde_json::from_str(&encoded).expect("deserialize");

        assert_eq!(decoded.lower(), bounds.lower());
        assert_eq!(decoded.upper(), bounds.upper());
        assert!(!decoded.has_l2_constraint());
    }

    #[test]
    fn serde_rejects_shape_mismatch() {
        let raw = RawBoundedTensor {
            lower: arr1(&[-1.0, 0.0]).into_dyn(),
            upper: arr1(&[1.0]).into_dyn(),
        };
        let encoded = serde_json::to_string(&raw).expect("serialize malformed fixture");

        assert!(serde_json::from_str::<BoundedTensor>(&encoded).is_err());
    }

    #[test]
    fn serde_rejects_non_canonical_inversion() {
        let raw = RawBoundedTensor {
            lower: arr1(&[2.0]).into_dyn(),
            upper: arr1(&[1.0]).into_dyn(),
        };
        let encoded = serde_json::to_string(&raw).expect("serialize malformed fixture");

        assert!(serde_json::from_str::<BoundedTensor>(&encoded).is_err());
    }
}
