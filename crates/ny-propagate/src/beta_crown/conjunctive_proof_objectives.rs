// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Authenticated proof-only objectives for conjunctive input splitting.
//!
//! This module deliberately exposes no general synthetic-objective constructor.
//! The only admitted transform is the exact Cersyve two-row conic implication
//!
//! ```text
//! [1, 0] <= 0  AND  [0, -1] <= 0
//!              IMPLIES
//!            [1, -1] <= 0.
//! ```
//!
//! More generally, every finite `lambda_0, lambda_1 >= 0` gives a valid conic
//! implication. Consequently, proving a strict lower bound above the matching
//! weighted threshold for any such combination refutes the original
//! conjunction. The immutable provenance below keeps that authority attached
//! until the input-split proof boundary.

/// Origin of one row in an authenticated conjunctive proof plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConjunctiveProofObjectiveProvenance {
    /// A row copied verbatim from the original output constraints.
    OriginalConstraint { source_index: usize },
    /// Authority to form finite non-negative conic combinations of two source
    /// rows at the proof boundary. The stored third row is the unit/unit
    /// canonical witness; runtime affine closure may choose other non-negative
    /// multipliers without changing the underlying implication.
    NonnegativeConicClosure {
        lhs_source_index: usize,
        rhs_source_index: usize,
    },
}

/// Immutable rows, thresholds, and provenance admitted to the proof-only
/// conjunctive graph input-split verifier.
///
/// Fields are private and the sole constructor recognizes one bit-exact row
/// shape with semantic-zero thresholds, so callers cannot mint provenance for
/// arbitrary synthetic objectives. The resulting typed plan is sound wherever
/// those rows really are the two output constraints; the CLI's additional AST
/// authentication is a deliberately narrower rollout policy.
#[derive(Clone, Debug)]
pub struct ConjunctiveProofObjectives {
    objectives: Vec<Vec<f32>>,
    thresholds: Vec<f32>,
    provenance: Vec<ConjunctiveProofObjectiveProvenance>,
}

impl ConjunctiveProofObjectives {
    /// Recognize the exact two-row Cersyve shape and append its unit conic sum.
    ///
    /// Objective-row comparisons are bit-exact. Thresholds must both be
    /// semantic zero; their IEEE sign is preserved as source provenance but has
    /// no proof meaning.
    pub fn try_exact_two_row_zero_threshold_unit_conic(
        objectives: &[Vec<f32>],
        thresholds: &[f32],
    ) -> Option<Self> {
        const FIRST: [u32; 2] = [1.0f32.to_bits(), 0.0f32.to_bits()];
        const SECOND: [u32; 2] = [0.0f32.to_bits(), (-1.0f32).to_bits()];

        if objectives.len() != 2
            || thresholds.len() != 2
            || objectives[0].len() != 2
            || objectives[1].len() != 2
            || objectives[0][0].to_bits() != FIRST[0]
            || objectives[0][1].to_bits() != FIRST[1]
            || objectives[1][0].to_bits() != SECOND[0]
            || objectives[1][1].to_bits() != SECOND[1]
            // `!= 0.0` rejects NaN and every nonzero while admitting both IEEE
            // signed-zero spellings. The unit conic threshold below is exactly
            // zero for all four sign combinations.
            || thresholds[0] != 0.0f32
            || thresholds[1] != 0.0f32
        {
            return None;
        }

        Some(Self {
            objectives: vec![
                objectives[0].clone(),
                objectives[1].clone(),
                vec![1.0, -1.0],
            ],
            thresholds: vec![thresholds[0], thresholds[1], 0.0],
            provenance: vec![
                ConjunctiveProofObjectiveProvenance::OriginalConstraint { source_index: 0 },
                ConjunctiveProofObjectiveProvenance::OriginalConstraint { source_index: 1 },
                ConjunctiveProofObjectiveProvenance::NonnegativeConicClosure {
                    lhs_source_index: 0,
                    rhs_source_index: 1,
                },
            ],
        })
    }

    /// Number of proof rows (two original rows plus one derived row).
    pub fn len(&self) -> usize {
        self.objectives.len()
    }

    /// Whether the plan contains no rows. Authenticated plans are never empty.
    pub fn is_empty(&self) -> bool {
        self.objectives.is_empty()
    }

    /// Typed origin of each proof row, in row order.
    pub fn provenance(&self) -> &[ConjunctiveProofObjectiveProvenance] {
        &self.provenance
    }

    pub(crate) fn objectives(&self) -> &[Vec<f32>] {
        &self.objectives
    }

    pub(crate) fn thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    /// Recheck the sealed constructor invariant immediately before proof use.
    pub(crate) fn has_valid_provenance(&self) -> bool {
        if self.objectives.len() != 3 || self.thresholds.len() != 3 || self.provenance.len() != 3 {
            return false;
        }
        let Some(expected) = Self::try_exact_two_row_zero_threshold_unit_conic(
            &self.objectives[..2],
            &self.thresholds[..2],
        ) else {
            return false;
        };
        let objectives_match =
            self.objectives
                .iter()
                .zip(&expected.objectives)
                .all(|(actual_row, expected_row)| {
                    actual_row.len() == expected_row.len()
                        && actual_row
                            .iter()
                            .zip(expected_row)
                            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
                });
        let thresholds_match = self
            .thresholds
            .iter()
            .zip(&expected.thresholds)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits());

        objectives_match && thresholds_match && self.provenance == expected.provenance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_originals() -> (Vec<Vec<f32>>, Vec<f32>) {
        (vec![vec![1.0, 0.0], vec![0.0, -1.0]], vec![0.0, -0.0])
    }

    #[test]
    fn exact_shape_appends_authenticated_unit_conic_sum_without_mutating_sources() {
        let (objectives, thresholds) = exact_originals();
        let original_objectives = objectives.clone();
        let original_thresholds = thresholds.clone();

        let plan = ConjunctiveProofObjectives::try_exact_two_row_zero_threshold_unit_conic(
            &objectives,
            &thresholds,
        )
        .expect("exact Cersyve shape should be admitted");

        assert_eq!(objectives, original_objectives);
        assert_eq!(thresholds, original_thresholds);
        assert_eq!(
            plan.objectives,
            vec![vec![1.0, 0.0], vec![0.0, -1.0], vec![1.0, -1.0]]
        );
        assert_eq!(plan.thresholds[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(plan.thresholds[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(plan.thresholds[2].to_bits(), 0.0f32.to_bits());
        assert_eq!(
            plan.provenance(),
            &[
                ConjunctiveProofObjectiveProvenance::OriginalConstraint { source_index: 0 },
                ConjunctiveProofObjectiveProvenance::OriginalConstraint { source_index: 1 },
                ConjunctiveProofObjectiveProvenance::NonnegativeConicClosure {
                    lhs_source_index: 0,
                    rhs_source_index: 1,
                },
            ]
        );
        assert!(plan.has_valid_provenance());
    }

    #[test]
    fn exact_detector_accepts_every_signed_zero_placement() {
        let (objectives, _) = exact_originals();
        for thresholds in [[0.0, 0.0], [0.0, -0.0], [-0.0, 0.0], [-0.0, -0.0]] {
            let plan = ConjunctiveProofObjectives::try_exact_two_row_zero_threshold_unit_conic(
                &objectives,
                &thresholds,
            )
            .expect("zero sign carries no conic proof meaning");
            assert_eq!(plan.thresholds[0].to_bits(), thresholds[0].to_bits());
            assert_eq!(plan.thresholds[1].to_bits(), thresholds[1].to_bits());
            assert!(plan.has_valid_provenance());
        }
    }

    #[test]
    fn exact_detector_rejects_permuted_scaled_and_extra_rows() {
        let (objectives, thresholds) = exact_originals();
        let permuted = vec![objectives[1].clone(), objectives[0].clone()];
        assert!(
            ConjunctiveProofObjectives::try_exact_two_row_zero_threshold_unit_conic(
                &permuted,
                &thresholds
            )
            .is_none()
        );

        let scaled = vec![vec![2.0, 0.0], vec![0.0, -2.0]];
        assert!(
            ConjunctiveProofObjectives::try_exact_two_row_zero_threshold_unit_conic(
                &scaled,
                &thresholds
            )
            .is_none()
        );

        let mut extra = objectives;
        extra.push(vec![1.0, -1.0]);
        assert!(
            ConjunctiveProofObjectives::try_exact_two_row_zero_threshold_unit_conic(
                &extra,
                &[0.0, -0.0, 0.0]
            )
            .is_none()
        );
    }

    #[test]
    fn exact_detector_rejects_nonfinite_or_nearby_values() {
        let (objectives, thresholds) = exact_originals();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.000_000_1] {
            let mut changed = objectives.clone();
            changed[0][0] = value;
            assert!(
                ConjunctiveProofObjectives::try_exact_two_row_zero_threshold_unit_conic(
                    &changed,
                    &thresholds
                )
                .is_none()
            );
        }

        let mut nonfinite_thresholds = thresholds;
        nonfinite_thresholds[0] = f32::NAN;
        assert!(
            ConjunctiveProofObjectives::try_exact_two_row_zero_threshold_unit_conic(
                &objectives,
                &nonfinite_thresholds
            )
            .is_none()
        );
    }

    #[test]
    fn proof_boundary_revalidation_rejects_mutated_row_threshold_or_provenance() {
        let (objectives, thresholds) = exact_originals();
        let plan = ConjunctiveProofObjectives::try_exact_two_row_zero_threshold_unit_conic(
            &objectives,
            &thresholds,
        )
        .unwrap();

        let mut row_mutation = plan.clone();
        row_mutation.objectives[2][0] = 2.0;
        assert!(!row_mutation.has_valid_provenance());

        let mut threshold_mutation = plan.clone();
        threshold_mutation.thresholds[2] = -0.0;
        assert!(!threshold_mutation.has_valid_provenance());

        let mut provenance_mutation = plan;
        provenance_mutation.provenance[2] =
            ConjunctiveProofObjectiveProvenance::OriginalConstraint { source_index: 2 };
        assert!(!provenance_mutation.has_valid_provenance());
    }
}
