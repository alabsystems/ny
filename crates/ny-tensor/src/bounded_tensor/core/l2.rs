// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Optional [`L2Constraint`] accessors / builders on [`BoundedTensor`].
//!
//! The annotation is a soundness-preserving tightening hint (a per-slice
//! Euclidean ball). It is attached deliberately by normalization IBP and read by
//! the immediately-downstream `Linear`; every value-transforming op drops it via
//! its fresh-tensor constructor (`l2: None`). See [`L2Constraint`].

use super::{BoundedTensor, L2Constraint};

impl BoundedTensor {
    /// Attach an L2 (Euclidean-ball) constraint, returning the annotated tensor.
    ///
    /// Used by normalization IBP outputs. The constraint MUST be a proven,
    /// outward-rounded enclosure of the true `||z - center||_2` per slice (this
    /// is the caller's obligation; see [`L2Constraint::new`]). Re-validates the
    /// shape against this tensor and DROPS the annotation (returns `self`
    /// unchanged) if it does not match — losing the tightening is sound.
    #[inline]
    #[must_use]
    pub fn with_l2_constraint(mut self, constraint: L2Constraint) -> Self {
        if constraint.center().shape() == self.shape() {
            self.l2 = Some(Box::new(constraint));
        }
        self
    }

    /// The attached L2 constraint, if any.
    #[inline]
    pub fn l2_constraint(&self) -> Option<&L2Constraint> {
        self.l2.as_deref()
    }

    /// Whether an L2 constraint is attached.
    #[inline]
    pub fn has_l2_constraint(&self) -> bool {
        self.l2.is_some()
    }

    /// Drop any attached L2 constraint in place.
    ///
    /// Callers that mutate `lower`/`upper` directly (rather than rebuilding) and
    /// cannot maintain the sphere invariant should call this. Always sound:
    /// dropping the annotation only forgoes tightening.
    #[inline]
    pub fn clear_l2_constraint(&mut self) {
        self.l2 = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn bt(shape: &[usize]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(shape), -1.0),
            ArrayD::from_elem(IxDyn(shape), 1.0),
        )
        .unwrap()
    }

    fn constraint(shape: &[usize], axis: usize) -> L2Constraint {
        let radius_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter_map(|(d, &s)| if d == axis { None } else { Some(s) })
            .collect();
        L2Constraint::new(
            ArrayD::zeros(IxDyn(shape)),
            ArrayD::from_elem(IxDyn(&radius_shape), 2.0),
            axis,
            shape,
        )
        .unwrap()
    }

    #[test]
    fn default_has_no_constraint() {
        assert!(!bt(&[3]).has_l2_constraint());
        assert!(bt(&[3]).l2_constraint().is_none());
    }

    #[test]
    fn attach_and_read_back() {
        let t = bt(&[4, 8]).with_l2_constraint(constraint(&[4, 8], 1));
        assert!(t.has_l2_constraint());
        assert_eq!(t.l2_constraint().unwrap().axis(), 1);
    }

    #[test]
    fn attach_drops_on_shape_mismatch() {
        // Constraint built for [2, 8] cannot attach to a [4, 8] tensor.
        let t = bt(&[4, 8]).with_l2_constraint(constraint(&[2, 8], 1));
        assert!(!t.has_l2_constraint());
    }

    #[test]
    fn clear_removes_constraint() {
        let mut t = bt(&[4, 8]).with_l2_constraint(constraint(&[4, 8], 1));
        assert!(t.has_l2_constraint());
        t.clear_l2_constraint();
        assert!(!t.has_l2_constraint());
    }

    #[test]
    // The "redundant" clone is the behavior under test: Clone must carry the
    // L2 constraint over to the copy.
    #[allow(clippy::redundant_clone)]
    fn clone_preserves_constraint() {
        // Sound: a clone describes the same values, so the same sphere bounds it.
        let t = bt(&[4, 8]).with_l2_constraint(constraint(&[4, 8], 1));
        assert!(t.clone().has_l2_constraint());
    }

    #[test]
    fn reshape_drops_constraint() {
        // Value-transforming / rebuilding ops drop the annotation (sound default).
        let t = bt(&[4, 8]).with_l2_constraint(constraint(&[4, 8], 1));
        assert!(!t.reshape(&[8, 4]).unwrap().has_l2_constraint());
        assert!(!t.flatten().has_l2_constraint());
    }
}
