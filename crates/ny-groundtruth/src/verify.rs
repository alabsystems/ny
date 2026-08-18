// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verify a network against a ground-truth graph (plan §2, property form).
//!
//! [`verify_against_ground_truth`] reduces `f(x) ⋈ g(x)` on a box to a
//! property of the difference network `h(x) = f(x) − g(x)` built by
//! [`build_difference_network`], exactly as the M0 spike did
//! (`crates/ny-propagate/tests/ground_truth_m0.rs`):
//!
//! - [`Relation::Dominates`]: `f(x) ≥ g(x)` ⇔ every output of `h` lies in
//!   `[0, +∞)`;
//! - [`Relation::AbsBound`]: `|f(x) − g(x)| ≤ ε` ⇔ every output of `h` lies
//!   in `[−ε, ε]` (ε is rounded *down* to f32 so the checked property is
//!   never weaker than requested).
//!
//! If bound propagation cannot prove the property, a grid witness search
//! evaluates `h` at concrete points via zero-width certified IBP
//! ([`GraphNetwork::propagate_ibp_sound`], whose per-node outward widening
//! turns the f32 point evaluation into a true enclosure of the real-arithmetic
//! value) and reports [`GroundTruthOutcome::Falsified`] only when the
//! enclosure certainly violates the property (M0 finding (b): the
//! bound-propagation verifier alone reports Unknown, not a counterexample,
//! so the falsified direction needs a concrete evaluation pass).

use ndarray::Array1;
use ny_core::{Bound, VerificationResult, VerificationSoundnessMode, VerificationSpec};
use ny_propagate::{
    build_difference_network, GraphNetwork, PropagationConfig, PropagationMethod, Verifier,
};
use ny_tensor::{next_down_f32, BoundedTensor};

use crate::error::{GroundTruthError, Result};

/// Relation between the network `f` and the ground truth `g` on the region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Relation {
    /// `f(x) ≥ g(x)` for all `x` in the region (dominance).
    Dominates,
    /// `|f(x) − g(x)| ≤ ε` for all `x` in the region.
    AbsBound(f64),
}

/// Outcome of a ground-truth verification query.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum GroundTruthOutcome {
    /// The relation is proved on the whole region; `difference_bounds` are
    /// the certified bounds on `f − g`.
    Verified {
        /// Certified output bounds of the difference network.
        difference_bounds: Vec<Bound>,
    },
    /// A concrete point in the region certainly violates the relation:
    /// the sound enclosure of `f(x*) − g(x*)` lies strictly outside the
    /// admissible output set.
    Falsified {
        /// The witness point `x*` (inside the input region).
        witness: Vec<f32>,
        /// Sound enclosure of the violating output of `f(x*) − g(x*)`.
        difference: Bound,
    },
    /// Neither proved nor concretely falsified (bounds too loose and the
    /// witness grid found no certain violation).
    Unknown {
        /// Best achieved bounds on `f − g` over the region.
        difference_bounds: Vec<Bound>,
    },
}

/// Options for [`verify_against_ground_truth_with`].
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    /// Bound-propagation configuration for the difference network. Defaults
    /// to CROWN — plain IBP decorrelates the shared input and cannot prove
    /// dominance even for `f = g + margin` (M0 IBP-looseness note).
    pub config: PropagationConfig,
    /// Witness-search grid resolution per input dimension (min 2 unless the
    /// dimension is degenerate). The total point count is capped; the
    /// per-dimension resolution shrinks for high-dimensional regions.
    pub witness_grid: usize,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            config: PropagationConfig {
                method: PropagationMethod::Crown,
                ..Default::default()
            },
            witness_grid: 5,
        }
    }
}

/// Cap on the total number of witness-grid evaluations.
const MAX_WITNESS_POINTS: usize = 20_000;

/// Verify `f` against the ground truth `g` on the box `input_bounds`, with
/// default options (CROWN + 5-per-dimension witness grid).
pub fn verify_against_ground_truth(
    f: &GraphNetwork,
    g: &GraphNetwork,
    relation: Relation,
    input_bounds: &[Bound],
) -> Result<GroundTruthOutcome> {
    verify_against_ground_truth_with(f, g, relation, input_bounds, &VerifyOptions::default())
}

/// Verify `f` against the ground truth `g` on the box `input_bounds`.
///
/// Internally: `build_difference_network(f, g)`, then the M0 verify path
/// (`Verifier::verify_graph` with a `[0, +∞)` or `[−ε, ε]` output spec),
/// then — if unproven — the grid witness search.
pub fn verify_against_ground_truth_with(
    f: &GraphNetwork,
    g: &GraphNetwork,
    relation: Relation,
    input_bounds: &[Bound],
    options: &VerifyOptions,
) -> Result<GroundTruthOutcome> {
    let epsilon = validated_epsilon(relation)?;
    let h = build_difference_network(f, g)?;

    // Cheap zero-arity probe: determines the output dimension and catches
    // f/g shape mismatches up front (same approach as verify_equivalence).
    // Certified variant: the probe's enclosure doubles as the fallback
    // `difference_bounds`, which must never be tighter than sound.
    let input_tensor = Verifier::bounds_to_tensor(input_bounds, None)?;
    let probe = h.propagate_ibp_sound(&input_tensor)?;
    let num_outputs = probe.lower().len();

    let output_bound = match epsilon {
        None => Bound::new_allow_infinite(0.0, f32::INFINITY),
        Some(eps) => Bound::new(-eps, eps),
    };
    let spec = VerificationSpec::new(input_bounds.to_vec(), vec![output_bound; num_outputs])?;
    let verifier = Verifier::new(options.config.clone());
    let result = verifier.verify_graph(&h, &spec)?;

    if let Some(output_bounds) = sound_verified_bounds(&result) {
        return Ok(GroundTruthOutcome::Verified {
            difference_bounds: output_bounds.to_vec(),
        });
    }
    let best_bounds = match result {
        // Heuristic provenance cannot prove the universal relation. Keep the
        // candidate bounds, then continue with the sound witness search.
        VerificationResult::Verified { output_bounds, .. } => output_bounds,
        VerificationResult::Violated { counterexample, .. } => {
            // Trust but verify: only report Falsified if a sound concrete
            // evaluation confirms the violation.
            if point_in_box(&counterexample, input_bounds) {
                if let Some(outcome) = certain_violation_at(&h, &counterexample, epsilon)? {
                    return Ok(outcome);
                }
            }
            enclosure_bounds(&probe)
        }
        VerificationResult::Unknown { bounds, .. } => bounds,
        VerificationResult::Timeout { partial_bounds, .. } => {
            partial_bounds.unwrap_or_else(|| enclosure_bounds(&probe))
        }
    };

    // Falsified direction: grid search for a point whose sound enclosure
    // certainly violates the relation.
    if let Some(outcome) = grid_witness(&h, input_bounds, epsilon, options.witness_grid)? {
        return Ok(outcome);
    }
    Ok(GroundTruthOutcome::Unknown {
        difference_bounds: best_bounds,
    })
}

fn sound_verified_bounds(result: &VerificationResult) -> Option<&[Bound]> {
    match result {
        VerificationResult::Verified {
            provenance,
            output_bounds,
            ..
        } if provenance.mode() == VerificationSoundnessMode::Sound => Some(output_bounds),
        _ => None,
    }
}

/// Validate the relation's ε and round it down to f32 (sound direction:
/// the verified property is at least as strong as the f64 request).
fn validated_epsilon(relation: Relation) -> Result<Option<f32>> {
    match relation {
        Relation::Dominates => Ok(None),
        Relation::AbsBound(eps) => {
            if !eps.is_finite() {
                return Err(GroundTruthError::NonFiniteParameter {
                    name: "epsilon".to_string(),
                    value: eps,
                });
            }
            let mut eps32 = eps as f32;
            if f64::from(eps32) > eps {
                eps32 = next_down_f32(eps32);
            }
            if eps32 <= 0.0 {
                return Err(GroundTruthError::DegenerateParameter {
                    name: "epsilon".to_string(),
                    reason: format!(
                        "must be strictly positive (after sound f32 rounding), got {eps}"
                    ),
                });
            }
            Ok(Some(eps32))
        }
    }
}

/// Does the sound enclosure `[lo, hi]` certainly violate the relation?
fn certainly_violates(epsilon: Option<f32>, lo: f32, hi: f32) -> bool {
    match epsilon {
        None => hi < 0.0,                   // f − g certainly negative
        Some(eps) => lo > eps || hi < -eps, // |f − g| certainly beyond ε
    }
}

/// Evaluate `h` at a concrete point via zero-width certified IBP (the per-node
/// outward widening makes the result a true enclosure of the real-arithmetic
/// value, so f32 evaluation rounding can never manufacture a violation) and
/// return a Falsified outcome if any output certainly violates the relation.
fn certain_violation_at(
    h: &GraphNetwork,
    point: &[f32],
    epsilon: Option<f32>,
) -> Result<Option<GroundTruthOutcome>> {
    let arr = Array1::from(point.to_vec()).into_dyn();
    let tensor = BoundedTensor::new(arr.clone(), arr)?;
    let enclosure = h.propagate_ibp_sound(&tensor)?;
    for (&lo, &hi) in enclosure.lower().iter().zip(enclosure.upper().iter()) {
        if certainly_violates(epsilon, lo, hi) {
            return Ok(Some(GroundTruthOutcome::Falsified {
                witness: point.to_vec(),
                difference: Bound::new_allow_infinite(lo, hi),
            }));
        }
    }
    Ok(None)
}

fn point_in_box(point: &[f32], input_bounds: &[Bound]) -> bool {
    point.len() == input_bounds.len()
        && point
            .iter()
            .zip(input_bounds.iter())
            .all(|(&x, b)| x >= b.lower() && x <= b.upper())
}

/// Convert an IBP output enclosure into per-output `Bound`s.
fn enclosure_bounds(enclosure: &BoundedTensor) -> Vec<Bound> {
    enclosure
        .lower()
        .iter()
        .zip(enclosure.upper().iter())
        .map(|(&lo, &hi)| Bound::new_allow_infinite(lo, hi))
        .collect()
}

/// Number of grid points to evaluate, capped even when the Cartesian product
/// overflows `usize` or the minimum two-point resolution is already too large.
fn grid_point_budget(counts: &[usize]) -> usize {
    counts
        .iter()
        .try_fold(1_usize, |acc, &count| acc.checked_mul(count))
        .unwrap_or(MAX_WITNESS_POINTS)
        .min(MAX_WITNESS_POINTS)
}

fn capped_grid_resolution(requested: usize, varying_dimensions: usize) -> usize {
    let requested = requested.clamp(2, MAX_WITNESS_POINTS);
    let fits = |resolution: usize| {
        (0..varying_dimensions)
            .try_fold(1_usize, |product, _| {
                product
                    .checked_mul(resolution)
                    .filter(|&next| next <= MAX_WITNESS_POINTS)
            })
            .is_some()
    };
    if varying_dimensions <= 1 || !fits(2) {
        return if varying_dimensions <= 1 {
            requested
        } else {
            2
        };
    }

    let (mut low, mut high) = (2, requested);
    while low < high {
        let midpoint = low + (high - low).div_ceil(2);
        if fits(midpoint) {
            low = midpoint;
        } else {
            high = midpoint - 1;
        }
    }
    low
}

/// Grid witness search over the input box (M0 finding (b)): evaluate `h` at
/// evenly spaced points (endpoints included) and return the first point whose
/// sound enclosure certainly violates the relation.
fn grid_witness(
    h: &GraphNetwork,
    input_bounds: &[Bound],
    epsilon: Option<f32>,
    grid: usize,
) -> Result<Option<GroundTruthOutcome>> {
    let dim = input_bounds.len();
    if dim == 0
        || input_bounds
            .iter()
            .any(|b| !b.lower().is_finite() || !b.upper().is_finite())
    {
        return Ok(None); // no finite box to sample
    }

    // Per-dimension point counts: degenerate dimensions get one point;
    // shrink the resolution until the total fits the cap.
    let varying_dimensions = input_bounds
        .iter()
        .filter(|bound| bound.lower() != bound.upper())
        .count();
    let resolution = capped_grid_resolution(grid, varying_dimensions);
    let counts: Vec<usize> = input_bounds
        .iter()
        .map(|bound| {
            if bound.lower() == bound.upper() {
                1
            } else {
                resolution
            }
        })
        .collect();

    let mut index = vec![0_usize; dim];
    for _ in 0..grid_point_budget(&counts) {
        let point: Vec<f32> = index
            .iter()
            .zip(input_bounds.iter())
            .zip(counts.iter())
            .map(|((&i, b), &n)| {
                if n == 1 {
                    b.lower()
                } else {
                    let t = i as f64 / (n - 1) as f64;
                    let width = f64::from(b.upper()) - f64::from(b.lower());
                    let x = f64::from(b.lower()) + t * width;
                    // Clamp so FP rounding cannot push the sample outside the box.
                    (x as f32).clamp(b.lower(), b.upper())
                }
            })
            .collect();
        if let Some(outcome) = certain_violation_at(h, &point, epsilon)? {
            return Ok(Some(outcome));
        }

        // Odometer increment.
        let mut carry = true;
        for (i, count) in index.iter_mut().zip(counts.iter()) {
            *i += 1;
            if *i < *count {
                carry = false;
                break;
            }
            *i = 0;
        }
        if carry {
            return Ok(None);
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::SoundnessProvenance;

    fn verified_with(provenance: SoundnessProvenance) -> VerificationResult {
        VerificationResult::Verified {
            provenance,
            output_bounds: vec![Bound::new(-1.0, 1.0)],
            proof: None,
            actual_method: None,
        }
    }

    #[test]
    fn only_sound_verified_results_are_treated_as_proofs() {
        assert!(sound_verified_bounds(&verified_with(SoundnessProvenance::sound())).is_some());
        assert!(
            sound_verified_bounds(&verified_with(SoundnessProvenance::heuristic())).is_none(),
            "heuristic bounds must not become an unqualified Verified outcome"
        );
    }

    #[test]
    fn epsilon_is_rounded_toward_zero() {
        // 1e-9 has no exact f32; the f32 nearest value is above it, so the
        // sound direction rounds down.
        let eps = validated_epsilon(Relation::AbsBound(1e-9))
            .unwrap()
            .unwrap();
        assert!(f64::from(eps) <= 1e-9);
        assert!(eps > 0.0);

        assert!(matches!(
            validated_epsilon(Relation::AbsBound(0.0)),
            Err(GroundTruthError::DegenerateParameter { .. })
        ));
        assert!(matches!(
            validated_epsilon(Relation::AbsBound(f64::NAN)),
            Err(GroundTruthError::NonFiniteParameter { .. })
        ));
        assert!(validated_epsilon(Relation::Dominates).unwrap().is_none());
    }

    #[test]
    fn witness_grid_budget_is_hard_capped() {
        assert_eq!(grid_point_budget(&[2, 3, 4]), 24);
        assert_eq!(
            grid_point_budget(&vec![2; usize::BITS as usize]),
            MAX_WITNESS_POINTS,
            "overflowing Cartesian products must still honor the cap"
        );
        assert_eq!(
            grid_point_budget(&[MAX_WITNESS_POINTS, 2]),
            MAX_WITNESS_POINTS
        );
        let two_dimensional = capped_grid_resolution(usize::MAX, 2);
        assert!(
            two_dimensional * two_dimensional <= MAX_WITNESS_POINTS
                && (two_dimensional + 1) * (two_dimensional + 1) > MAX_WITNESS_POINTS,
            "resolution must be the largest square grid within the hard cap"
        );
        assert_eq!(capped_grid_resolution(usize::MAX, 20), 2);
    }

    #[test]
    fn violation_predicate_matches_relations() {
        // Dominates: only a certainly-negative enclosure counts.
        assert!(certainly_violates(None, -2.0, -1.0));
        assert!(!certainly_violates(None, -1.0, 0.0));
        assert!(!certainly_violates(None, 0.5, 1.0));
        // AbsBound(1): certainly above 1 or certainly below -1.
        assert!(certainly_violates(Some(1.0), 1.5, 2.0));
        assert!(certainly_violates(Some(1.0), -3.0, -1.5));
        assert!(!certainly_violates(Some(1.0), -0.5, 0.5));
        assert!(!certainly_violates(Some(1.0), 0.5, 1.5));
    }
}
