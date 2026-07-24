// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Optional per-tensor L2 (Euclidean-ball) constraint annotation.
//!
//! Normalization layers (RMSNorm / LayerNorm / GroupNorm / InstanceNorm) emit
//! an output `z` whose *whole vector* (per normalization slice) lies on/inside a
//! sphere: `sum_i z_i^2 = n * var / (var + eps) <= n`, hence `||z||_2 <= sqrt(n)`
//! (LayerNorm is zero-mean so the tighter `sqrt(n-1)` also holds).
//!
//! Box-CROWN / IBP bound each coordinate `|z_i| <= sqrt(n)` *independently*, so a
//! downstream `Linear` row `w·z` is bounded by the box as `||w||_1 * sqrt(n)` —
//! exponentially looser than the EXACT Cauchy–Schwarz bound `||w||_2 * sqrt(n)`.
//! The normalization output is a SPHERE that the FFN otherwise sees as a BOX.
//!
//! This type carries the sphere (a `center` vector plus a per-slice `radius`)
//! alongside the interval bounds so that the immediately-downstream `Linear`
//! (IBP and CROWN concretization) can intersect the box interval with the exact
//! Cauchy–Schwarz interval. **Intersection only ever tightens** — it can never
//! widen a bound — so attaching or dropping this annotation is always sound.
//!
//! ## Soundness contract
//!
//! `radius[slice]` MUST be a proven upper bound on the true Euclidean distance
//! `||z_slice - center_slice||_2` over every point of the input box. The producer
//! is responsible for directed-OUTWARD rounding of `center` and `radius`. Any op
//! that cannot maintain this invariant must DROP the annotation (the default):
//! losing the tightening is sound; keeping a stale sphere would not be.

use ndarray::ArrayD;

/// A per-normalization-slice Euclidean-ball annotation on a [`super::BoundedTensor`].
///
/// Semantics: along `axis`, for every fixed assignment of the remaining
/// ("batch") indices, the sub-vector `v` of the annotated tensor satisfies
/// `||v - center_slice||_2 <= radius[batch_index]`, where `center_slice` is the
/// matching sub-vector of [`center`](Self::center).
///
/// Not serialized: this is an inference-time tightening hint derived from the
/// layer semantics, never part of a persisted certificate (see `#[serde(skip)]`
/// on the field in `BoundedTensor`). It must be re-derived on load.
#[derive(Debug, Clone, PartialEq)]
pub struct L2Constraint {
    /// Center vector. Same shape as the annotated tensor.
    center: ArrayD<f32>,
    /// Per-slice L2 radius. Shape == the tensor shape with `axis` removed
    /// (rank `ndim - 1`); `radius[batch_index]` bounds the slice at that batch
    /// position. A proven OUTWARD-rounded upper bound on the true distance.
    radius: ArrayD<f32>,
    /// The axis the Euclidean ball is taken over (the normalization axis).
    axis: usize,
}

impl L2Constraint {
    /// Construct a constraint. Returns `None` (drop the annotation, sound) if the
    /// shapes are inconsistent, the axis is out of range, or any value is
    /// non-finite or negative — i.e. anything that would make the sphere
    /// unusable. A dropped annotation only loses tightening.
    ///
    /// `center` must match `tensor_shape`. `radius` must have rank
    /// `tensor_shape.len() - 1` and equal `tensor_shape` with `axis` removed.
    pub fn new(
        center: ArrayD<f32>,
        radius: ArrayD<f32>,
        axis: usize,
        tensor_shape: &[usize],
    ) -> Option<Self> {
        let ndim = tensor_shape.len();
        if ndim == 0 || axis >= ndim {
            return None;
        }
        if center.shape() != tensor_shape {
            return None;
        }
        // radius rank/shape == tensor_shape with `axis` removed.
        let expected_radius_shape: Vec<usize> = tensor_shape
            .iter()
            .enumerate()
            .filter_map(|(d, &s)| if d == axis { None } else { Some(s) })
            .collect();
        if radius.shape() != expected_radius_shape.as_slice() {
            return None;
        }
        if center.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Radius must be finite and non-negative to be a usable sphere.
        if radius.iter().any(|&r| !r.is_finite() || r < 0.0) {
            return None;
        }
        Some(Self {
            center,
            radius,
            axis,
        })
    }

    /// The center vector (same shape as the annotated tensor).
    #[inline]
    pub fn center(&self) -> &ArrayD<f32> {
        &self.center
    }

    /// Per-slice radius (rank = tensor rank - 1; `axis` removed).
    #[inline]
    pub fn radius(&self) -> &ArrayD<f32> {
        &self.radius
    }

    /// The normalization axis the ball is taken over.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn zeros(shape: &[usize]) -> ArrayD<f32> {
        ArrayD::zeros(IxDyn(shape))
    }

    #[test]
    fn accepts_valid_last_axis_constraint() {
        let c = L2Constraint::new(
            zeros(&[4, 8]),
            ArrayD::from_elem(IxDyn(&[4]), 2.0),
            1,
            &[4, 8],
        );
        let c = c.expect("valid constraint");
        assert_eq!(c.axis(), 1);
        assert_eq!(c.center().shape(), &[4, 8]);
        assert_eq!(c.radius().shape(), &[4]);
    }

    #[test]
    fn rejects_center_shape_mismatch() {
        assert!(L2Constraint::new(
            zeros(&[4, 7]),
            ArrayD::from_elem(IxDyn(&[4]), 2.0),
            1,
            &[4, 8]
        )
        .is_none());
    }

    #[test]
    fn rejects_wrong_radius_rank() {
        // radius must be tensor_shape with `axis` removed → [4], not [4, 8].
        assert!(L2Constraint::new(
            zeros(&[4, 8]),
            ArrayD::from_elem(IxDyn(&[4, 8]), 2.0),
            1,
            &[4, 8]
        )
        .is_none());
    }

    #[test]
    fn rejects_axis_out_of_range() {
        assert!(L2Constraint::new(
            zeros(&[4, 8]),
            ArrayD::from_elem(IxDyn(&[4]), 2.0),
            2,
            &[4, 8]
        )
        .is_none());
    }

    #[test]
    fn rejects_negative_or_nonfinite_radius() {
        assert!(L2Constraint::new(
            zeros(&[4, 8]),
            ArrayD::from_elem(IxDyn(&[4]), -1.0),
            1,
            &[4, 8]
        )
        .is_none());
        assert!(L2Constraint::new(
            zeros(&[4, 8]),
            ArrayD::from_elem(IxDyn(&[4]), f32::NAN),
            1,
            &[4, 8]
        )
        .is_none());
        assert!(L2Constraint::new(
            zeros(&[4, 8]),
            ArrayD::from_elem(IxDyn(&[4]), f32::INFINITY),
            1,
            &[4, 8]
        )
        .is_none());
    }

    #[test]
    fn rejects_nonfinite_center() {
        let mut center = zeros(&[4]);
        center[[0]] = f32::NAN;
        assert!(L2Constraint::new(center, ArrayD::from_elem(IxDyn(&[]), 2.0), 0, &[4]).is_none());
    }

    #[test]
    fn one_dim_radius_is_rank_zero() {
        // 1D tensor, axis 0 → radius rank-0 (scalar).
        let c = L2Constraint::new(zeros(&[5]), ArrayD::from_elem(IxDyn(&[]), 3.0), 0, &[5])
            .expect("valid 1D constraint");
        assert_eq!(c.radius().ndim(), 0);
    }
}
