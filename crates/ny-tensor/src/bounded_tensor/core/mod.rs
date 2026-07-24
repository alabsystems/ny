// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core bounded tensor structure and construction helpers.
//!
//! Split into submodules by concern:
//! - `constructors`: Creation, setters, sanitization
//! - `numeric`: Width, rounding, intersection, repair
//! - `shape_ops`: Reshape, flatten, slice, expand, concat, stack

pub(crate) mod constructors;
mod inversion_repair;
mod l2;
mod numeric;
mod shape_ops;

pub use super::l2_constraint::L2Constraint;
pub use constructors::RepairStrategy;
pub use inversion_repair::{repair_inverted_bounds, repair_inverted_bounds_nd, InversionRepair};

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// A tensor where each element has certified lower and upper bounds.
///
/// May optionally carry an [`L2Constraint`] (a per-normalization-slice
/// Euclidean ball) that lets the immediately-downstream `Linear` tighten its box
/// interval via the exact Cauchy–Schwarz row bound. The annotation is purely a
/// soundness-preserving tightening hint: it is `None` by default, is dropped by
/// every value-transforming op (only re-attached deliberately by normalization
/// IBP), and is intersected — never used to widen — at the `Linear`. It is
/// **not** serialized, so it never enters a persisted certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedTensor {
    /// Lower bounds for each element.
    lower: ArrayD<f32>,
    /// Upper bounds for each element.
    upper: ArrayD<f32>,
    /// Optional Euclidean-ball annotation (see [`L2Constraint`]). Not part of the
    /// certificate; skipped during (de)serialization and re-derived on load.
    #[serde(skip)]
    l2: Option<Box<L2Constraint>>,
}

impl BoundedTensor {
    /// Read-only view of the lower bounds.
    #[inline]
    pub fn lower(&self) -> &ArrayD<f32> {
        &self.lower
    }

    /// Read-only view of the upper bounds.
    #[inline]
    pub fn upper(&self) -> &ArrayD<f32> {
        &self.upper
    }

    /// Read-only views of both lower and upper bounds.
    #[inline]
    pub fn lower_upper(&self) -> (&ArrayD<f32>, &ArrayD<f32>) {
        (&self.lower, &self.upper)
    }

    /// Consume the tensor and return owned lower/upper arrays.
    #[inline]
    pub fn into_parts(self) -> (ArrayD<f32>, ArrayD<f32>) {
        (self.lower, self.upper)
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
        Self {
            lower,
            upper,
            l2: None,
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
        self.lower.shape()
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
