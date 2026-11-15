// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-margin-alpha: the CERTIFY-ONCE margin publication.
//!
//! The certified C-row margin ascent (`network/graph_alpha/propagate_dag/
//! margin_alpha.rs`) optimizes per-spec ReLU α and bilinear McCormick
//! interpolation parameters against the property's margin rows. Neither the
//! per-spec δ nor the bilinear parameters fit the shared `GraphAlphaState`
//! the downstream spec-guided CROWN pass consumes, so without this object the
//! gain would be discarded at the root.
//!
//! Instead the ascent restores the whole committed optimizer state (ReLU α,
//! per-spec δ, bilinear parameters, checkpointed together), runs ONE fresh
//! certified backward with exactly that state, and records the resulting
//! certified bounds on every margin row here. The multi-objective root then
//! intersects them into its own objective bounds.
//!
//! # Why the intersection is sound
//!
//! Both operands are certified enclosures of the SAME scalar `c_r^T f(x)` over
//! the SAME input box: the root's from its spec-guided CROWN pass, this one
//! from a CROWN pass whose every relaxation is valid for any α ∈ [0,1] and any
//! r ∈ [0,1], concretized with the crate's outward-rounded `concretize_sound`.
//! Keeping the larger lower and the smaller upper bound of each row is
//! therefore an enclosure of every reachable value. An inverted intersection
//! would mean one operand is unsound; the merge then refuses the WHOLE
//! publication (all-or-nothing) and prints a soundness alarm rather than
//! guessing which operand to trust.
//!
//! # Why the identity check exists
//!
//! The publication rides on the root `GraphAlphaState`, which is cloned into
//! warm starts and caches. It is valid only for the input box and objective
//! rows it was certified against, so the root refuses it unless the box
//! (shape and every endpoint) and every objective coefficient, threshold and
//! direction match BIT-FOR-BIT. A publication can never be applied to a child
//! domain's box.

use super::AlphaSpecEarlyExit;
use ny_tensor::BoundedTensor;

/// The EXACT identity of an input box: its shape and the bit pattern of every
/// lower and upper endpoint, in logical (row-major) order.
///
/// Identity, not a hash: the root refuses a publication whose box is not
/// bit-for-bit its own, so a stale or misrouted publication can never be
/// merged into a domain it was not certified for. (#vit-margin-alpha review
/// 2026-09-24: this was a 64-bit FNV-1a fingerprint while the contract below
/// promised BIT-FOR-BIT; a hash is not that, and on a soundness-bearing merge
/// the few KiB an exact copy costs — 24 KiB on vit's 3072-element box — buy
/// the stated contract instead of a probabilistic one.)
#[derive(Debug, Clone, PartialEq, Eq)]
struct InputBoxIdentity {
    shape: Vec<usize>,
    lower_bits: Vec<u32>,
    upper_bits: Vec<u32>,
}

impl InputBoxIdentity {
    fn of(input: &BoundedTensor) -> Self {
        Self {
            shape: input.shape().to_vec(),
            lower_bits: input.lower().iter().map(|value| value.to_bits()).collect(),
            upper_bits: input.upper().iter().map(|value| value.to_bits()).collect(),
        }
    }

    fn matches(&self, input: &BoundedTensor) -> bool {
        self.shape.as_slice() == input.shape()
            && self.lower_bits.len() == input.lower().len()
            && self.upper_bits.len() == input.upper().len()
            && input
                .lower()
                .iter()
                .zip(&self.lower_bits)
                .all(|(value, &bits)| value.to_bits() == bits)
            && input
                .upper()
                .iter()
                .zip(&self.upper_bits)
                .all(|(value, &bits)| value.to_bits() == bits)
    }
}

/// Why the root declined to merge a publication. Every variant leaves the
/// root's objective bounds untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginPublicationRefusal {
    /// Different number of rows than the root's objective set.
    RowCount,
    /// An objective coefficient differs (bit pattern or length).
    ObjectiveMismatch,
    /// A threshold differs (bit pattern).
    ThresholdMismatch,
    /// The verification direction (`verify_upper_bound`) differs.
    DirectionMismatch,
    /// The input box the publication was certified over is not this box.
    InputMismatch,
    /// The published lower/upper intersection with the root's own bounds is
    /// EMPTY on some row: one of two certified enclosures is unsound.
    Disjoint,
}

impl MarginPublicationRefusal {
    /// Stable token for the unconditional observable.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RowCount => "row_count",
            Self::ObjectiveMismatch => "objective_mismatch",
            Self::ThresholdMismatch => "threshold_mismatch",
            Self::DirectionMismatch => "direction_mismatch",
            Self::InputMismatch => "input_mismatch",
            Self::Disjoint => "disjoint_SOUNDNESS_ALARM",
        }
    }
}

/// What one successful merge changed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarginMergeStats {
    /// Rows whose lower bound rose.
    pub raised_lower: usize,
    /// Rows whose upper bound fell.
    pub lowered_upper: usize,
    /// Rows verified before the merge (by each row's own rule).
    pub verified_before: usize,
    /// Rows verified after the merge.
    pub verified_after: usize,
    /// Worst signed slack before the merge (`> 0` means verified).
    pub min_slack_before: f32,
    /// Worst signed slack after the merge.
    pub min_slack_after: f32,
}

/// Certified bounds on every margin row, from ONE fresh certified pass with
/// the committed margin-ascent state. See the module docs.
#[derive(Debug, Clone)]
pub struct MarginAlphaPublication {
    rows: Vec<AlphaSpecEarlyExit>,
    /// Certified bounds on `c_r^T y` (the ORIGINAL, unflipped objective).
    lower: Vec<f32>,
    upper: Vec<f32>,
    input_identity: InputBoxIdentity,
    committed_iteration: usize,
    spec_slots: usize,
    matmul_alpha_nodes: usize,
}

impl MarginAlphaPublication {
    /// Build a publication, or `None` when the payload is malformed: length
    /// mismatch, a NaN endpoint, an inverted interval, or an infinity on the
    /// wrong side (`+inf` lower / `-inf` upper claims a value no real number
    /// has). A malformed payload is never published (fail closed to "no
    /// publication").
    ///
    /// A VACUOUS side (`-inf` lower, `+inf` upper) is admitted: the merge is a
    /// max/min, so a vacuous side leaves the root's own bound untouched, and
    /// one row whose certified pass saturated must not throw away the rows
    /// that did not.
    #[must_use]
    pub(crate) fn new(
        rows: Vec<AlphaSpecEarlyExit>,
        lower: Vec<f32>,
        upper: Vec<f32>,
        input: &BoundedTensor,
        committed_iteration: usize,
        spec_slots: usize,
        matmul_alpha_nodes: usize,
    ) -> Option<Self> {
        if rows.is_empty() || lower.len() != rows.len() || upper.len() != rows.len() {
            return None;
        }
        let well_formed = lower.iter().zip(&upper).all(|(&lo, &hi)| {
            !lo.is_nan()
                && !hi.is_nan()
                && lo <= hi
                && lo != f32::INFINITY
                && hi != f32::NEG_INFINITY
        });
        if !well_formed {
            return None;
        }
        Some(Self {
            rows,
            lower,
            upper,
            input_identity: InputBoxIdentity::of(input),
            committed_iteration,
            spec_slots,
            matmul_alpha_nodes,
        })
    }

    /// The objective rows the bounds were certified for.
    #[must_use]
    pub fn rows(&self) -> &[AlphaSpecEarlyExit] {
        &self.rows
    }

    /// Certified lower bounds on `c_r^T y`, one per row.
    #[must_use]
    pub fn lower(&self) -> &[f32] {
        &self.lower
    }

    /// Certified upper bounds on `c_r^T y`, one per row.
    #[must_use]
    pub fn upper(&self) -> &[f32] {
        &self.upper
    }

    /// Ascent iteration whose state was committed and re-certified.
    #[must_use]
    pub fn committed_iteration(&self) -> usize {
        self.committed_iteration
    }

    /// Per-spec δ slots active in the committed state.
    #[must_use]
    pub fn spec_slots(&self) -> usize {
        self.spec_slots
    }

    /// Bilinear/MatMul nodes whose interpolation parameters were optimized.
    #[must_use]
    pub fn matmul_alpha_nodes(&self) -> usize {
        self.matmul_alpha_nodes
    }

    /// Whether this publication was certified for exactly this box and these
    /// objective rows. BIT-FOR-BIT: a publication is identity-bound, never
    /// "close enough".
    ///
    /// # Errors
    ///
    /// The first mismatch found, as a [`MarginPublicationRefusal`].
    pub fn describes(
        &self,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        verify_upper_bound: bool,
    ) -> Result<(), MarginPublicationRefusal> {
        if objectives.len() != self.rows.len() || thresholds.len() != self.rows.len() {
            return Err(MarginPublicationRefusal::RowCount);
        }
        for ((row, objective), &threshold) in self.rows.iter().zip(objectives).zip(thresholds) {
            if row.objective.len() != objective.len()
                || row
                    .objective
                    .iter()
                    .zip(objective)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(MarginPublicationRefusal::ObjectiveMismatch);
            }
            if row.threshold.to_bits() != threshold.to_bits() {
                return Err(MarginPublicationRefusal::ThresholdMismatch);
            }
            if row.verify_upper_bound != verify_upper_bound {
                return Err(MarginPublicationRefusal::DirectionMismatch);
            }
        }
        if !self.input_identity.matches(input) {
            return Err(MarginPublicationRefusal::InputMismatch);
        }
        Ok(())
    }

    /// Intersect the published rows into `existing` (one `(lower, upper)` per
    /// objective, the root's own certified bounds), ALL-OR-NOTHING.
    ///
    /// A non-finite existing endpoint is simply replaced from the other side
    /// by the max/min rule. If ANY row's intersection is empty, `existing` is
    /// left untouched and [`MarginPublicationRefusal::Disjoint`] is returned:
    /// two certified enclosures of one value cannot be disjoint, so one of
    /// them is unsound and neither may be trusted to decide which.
    ///
    /// # Errors
    ///
    /// [`MarginPublicationRefusal::RowCount`] on a length mismatch,
    /// [`MarginPublicationRefusal::Disjoint`] as above.
    pub fn merge_into(
        &self,
        existing: &mut [(f32, f32)],
    ) -> Result<MarginMergeStats, MarginPublicationRefusal> {
        if existing.len() != self.rows.len() {
            return Err(MarginPublicationRefusal::RowCount);
        }
        let slack = |row: &AlphaSpecEarlyExit, lo: f32, hi: f32| -> f32 {
            if row.verify_upper_bound {
                row.threshold - hi
            } else {
                lo - row.threshold
            }
        };
        let mut merged: Vec<(f32, f32)> = Vec::with_capacity(existing.len());
        let mut stats = MarginMergeStats {
            raised_lower: 0,
            lowered_upper: 0,
            verified_before: 0,
            verified_after: 0,
            min_slack_before: f32::INFINITY,
            min_slack_after: f32::INFINITY,
        };
        for (index, &(old_lo, old_hi)) in existing.iter().enumerate() {
            let row = &self.rows[index];
            let (pub_lo, pub_hi) = (self.lower[index], self.upper[index]);
            // `f32::max`/`min` ignore a NaN operand, and the publication is
            // finite by construction, so a NaN existing endpoint is replaced.
            let new_lo = pub_lo.max(old_lo);
            let new_hi = pub_hi.min(old_hi);
            if new_lo > new_hi {
                return Err(MarginPublicationRefusal::Disjoint);
            }
            if new_lo > old_lo || (old_lo.is_nan() && new_lo.is_finite()) {
                stats.raised_lower += 1;
            }
            if new_hi < old_hi || (old_hi.is_nan() && new_hi.is_finite()) {
                stats.lowered_upper += 1;
            }
            if row.is_verified(old_lo, old_hi) {
                stats.verified_before += 1;
            }
            if row.is_verified(new_lo, new_hi) {
                stats.verified_after += 1;
            }
            let before = slack(row, old_lo, old_hi);
            if before.is_nan() || before < stats.min_slack_before {
                stats.min_slack_before = before;
            }
            let after = slack(row, new_lo, new_hi);
            if after < stats.min_slack_after {
                stats.min_slack_after = after;
            }
            merged.push((new_lo, new_hi));
        }
        existing.copy_from_slice(&merged);
        Ok(stats)
    }

    /// The root's single entry point: identity check, all-or-nothing merge,
    /// and the UNCONDITIONAL observable
    ///
    /// ```text
    /// [alpha-margin-publish] rows=9 merged=1 raised=1 lowered_upper=0 verified_before=8 verified_after=9 min_slack_before=-5.448e-1 min_slack_after=1.2e-2 committed=17 spec_slots=9 matmul_alpha_nodes=6
    /// [alpha-margin-publish] rows=9 merged=0 refused=input_mismatch
    /// ```
    ///
    /// Printed whenever a publication reaches the root, so a publication that
    /// was built but never merged is distinguishable from one that was never
    /// built (the latter prints no line at all).
    pub(crate) fn merge_into_root(
        &self,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        verify_upper_bound: bool,
        existing: &mut [(f32, f32)],
    ) -> Result<MarginMergeStats, MarginPublicationRefusal> {
        let outcome = self
            .describes(input, objectives, thresholds, verify_upper_bound)
            .and_then(|()| self.merge_into(existing));
        match &outcome {
            Ok(stats) => eprintln!(
                "[alpha-margin-publish] rows={} merged=1 raised={} lowered_upper={} \
                 verified_before={} verified_after={} min_slack_before={:.6e} \
                 min_slack_after={:.6e} committed={} spec_slots={} matmul_alpha_nodes={}",
                self.rows.len(),
                stats.raised_lower,
                stats.lowered_upper,
                stats.verified_before,
                stats.verified_after,
                stats.min_slack_before,
                stats.min_slack_after,
                self.committed_iteration,
                self.spec_slots,
                self.matmul_alpha_nodes,
            ),
            Err(refusal) => eprintln!(
                "[alpha-margin-publish] rows={} merged=0 refused={}",
                self.rows.len(),
                refusal.as_str()
            ),
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, ArrayD, IxDyn};

    fn row(objective: &[f32], threshold: f32) -> AlphaSpecEarlyExit {
        AlphaSpecEarlyExit {
            objective: objective.to_vec(),
            threshold,
            verify_upper_bound: false,
        }
    }

    fn input() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 0.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.5]).unwrap(),
        )
        .unwrap()
    }

    fn publication(lower: &[f32], upper: &[f32]) -> MarginAlphaPublication {
        MarginAlphaPublication::new(
            vec![row(&[1.0, -1.0, 0.0], 0.0), row(&[1.0, 0.0, -1.0], 0.0)],
            lower.to_vec(),
            upper.to_vec(),
            &input(),
            3,
            2,
            1,
        )
        .expect("well-formed publication")
    }

    /// The merge keeps the tighter side of each row and nothing else.
    #[test]
    fn margin_publication_merge_keeps_the_tighter_side_of_each_row() {
        let publication = publication(&[0.25, -3.0], &[2.0, 5.0]);
        let mut existing = vec![(-0.5_f32, 1.5_f32), (-1.0, 4.0)];
        let stats = publication.merge_into(&mut existing).expect("merges");
        assert_eq!(existing, vec![(0.25, 1.5), (-1.0, 4.0)]);
        assert_eq!(stats.raised_lower, 1);
        assert_eq!(stats.lowered_upper, 0);
        assert_eq!(stats.verified_before, 0);
        assert_eq!(stats.verified_after, 1);
        assert_eq!(stats.min_slack_before, -1.0);
        assert_eq!(stats.min_slack_after, -1.0);
    }

    /// Two certified enclosures of one value cannot be disjoint; if they are,
    /// NOTHING is merged — not even the rows that intersect cleanly.
    #[test]
    fn margin_publication_disjoint_row_refuses_the_whole_publication() {
        let publication = publication(&[0.25, 6.0], &[2.0, 7.0]);
        let original = vec![(-0.5_f32, 1.5_f32), (-1.0, 4.0)];
        let mut existing = original.clone();
        assert_eq!(
            publication.merge_into(&mut existing),
            Err(MarginPublicationRefusal::Disjoint)
        );
        assert_eq!(existing, original, "all-or-nothing: no row may move");
    }

    /// Identity is bit-for-bit: a different box, a different coefficient, a
    /// different threshold, or a flipped direction each refuses.
    #[test]
    fn margin_publication_refuses_any_other_box_or_objective() {
        let publication = publication(&[0.25, -3.0], &[2.0, 5.0]);
        let objectives = vec![vec![1.0, -1.0, 0.0], vec![1.0, 0.0, -1.0]];
        let thresholds = [0.0_f32, 0.0];
        assert_eq!(
            publication.describes(&input(), &objectives, &thresholds, false),
            Ok(())
        );
        let other_box = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.5]).into_dyn(),
            arr1(&[1.0_f32, 1.25]).into_dyn(),
        )
        .unwrap();
        assert_eq!(
            publication.describes(&other_box, &objectives, &thresholds, false),
            Err(MarginPublicationRefusal::InputMismatch)
        );
        // Identity is EXACT, not numeric and not a hash: the same values in
        // another shape, and a sign-of-zero change, each refuse.
        let reshaped = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-1.0, 0.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.5]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            publication.describes(&reshaped, &objectives, &thresholds, false),
            Err(MarginPublicationRefusal::InputMismatch)
        );
        let zero_box = |zero: f32| {
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![zero, 0.5]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.5]).unwrap(),
            )
            .unwrap()
        };
        let at_zero = MarginAlphaPublication::new(
            vec![row(&[1.0, -1.0, 0.0], 0.0), row(&[1.0, 0.0, -1.0], 0.0)],
            vec![0.25, -3.0],
            vec![2.0, 5.0],
            &zero_box(0.0),
            3,
            2,
            1,
        )
        .expect("well-formed publication");
        assert_eq!(
            at_zero.describes(&zero_box(0.0), &objectives, &thresholds, false),
            Ok(())
        );
        assert_eq!(
            at_zero.describes(&zero_box(-0.0), &objectives, &thresholds, false),
            Err(MarginPublicationRefusal::InputMismatch),
            "-0.0 and 0.0 compare equal numerically but are different bits"
        );
        let mut other_objectives = objectives.clone();
        other_objectives[1][2] = -0.5;
        assert_eq!(
            publication.describes(&input(), &other_objectives, &thresholds, false),
            Err(MarginPublicationRefusal::ObjectiveMismatch)
        );
        assert_eq!(
            publication.describes(&input(), &objectives, &[0.0, 1e-7], false),
            Err(MarginPublicationRefusal::ThresholdMismatch)
        );
        assert_eq!(
            publication.describes(&input(), &objectives, &thresholds, true),
            Err(MarginPublicationRefusal::DirectionMismatch)
        );
        assert_eq!(
            publication.describes(&input(), &objectives[..1], &thresholds[..1], false),
            Err(MarginPublicationRefusal::RowCount)
        );
    }

    /// A malformed payload is never constructed.
    #[test]
    fn margin_publication_refuses_malformed_payloads() {
        let rows = || vec![row(&[1.0, -1.0], 0.0)];
        assert!(
            MarginAlphaPublication::new(rows(), vec![1.0], vec![0.5], &input(), 0, 0, 0).is_none()
        );
        assert!(
            MarginAlphaPublication::new(rows(), vec![f32::NAN], vec![0.5], &input(), 0, 0, 0)
                .is_none()
        );
        assert!(MarginAlphaPublication::new(rows(), vec![], vec![], &input(), 0, 0, 0).is_none());
        assert!(
            MarginAlphaPublication::new(rows(), vec![-1.0], vec![0.5], &input(), 0, 0, 0).is_some()
        );
        // Wrong-side infinities claim a value no real number has.
        for (lo, hi) in [
            (f32::INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::NEG_INFINITY),
            (0.0, f32::NAN),
        ] {
            assert!(
                MarginAlphaPublication::new(rows(), vec![lo], vec![hi], &input(), 0, 0, 0)
                    .is_none(),
                "[{lo}, {hi}] must be refused"
            );
        }
        // A vacuous side is admitted and merges as a no-op on that side.
        let vacuous = MarginAlphaPublication::new(
            rows(),
            vec![f32::NEG_INFINITY],
            vec![f32::INFINITY],
            &input(),
            0,
            0,
            0,
        )
        .expect("a vacuous row is well-formed");
        let mut existing = vec![(-0.25_f32, 0.75_f32)];
        let stats = vacuous.merge_into(&mut existing).expect("merges");
        assert_eq!(existing, vec![(-0.25, 0.75)]);
        assert_eq!((stats.raised_lower, stats.lowered_upper), (0, 0));
    }
}
