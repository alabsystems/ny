// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Whole-field "no-escape" continuous-domain tolerance verification.
//!
//! Sampled inspection compares MEASURED POINTS to a nominal surface and reports
//! the max deviation — but a spike *between* samples is invisible. This module
//! reframes the landed ground-truth machinery ([`crate::verify`]) as a
//! **whole-field tolerance guarantee**: given a surrogate `f` of the measured
//! field (a `GraphNetwork` — a small NN, or a piecewise/interpolant graph) and
//! the nominal surface `g` as a ground-truth graph (plane / sphere / cylinder /
//! … residual from [`crate::builders`]), prove
//!
//! ```text
//! ∀ x in region:  |f(x) − g(x)| ≤ tol
//! ```
//!
//! over the **whole** input box — not a finite point grid. This is exactly the
//! [`Relation::AbsBound`] property of the difference network `h = f − g`, so
//! [`verify_whole_field_tolerance`] is a thin, semantically-documented wrapper
//! over [`verify_against_ground_truth`] + its grid witness search:
//!
//! - CROWN proves `h(x) ∈ [−tol, tol]` on the whole box ⇒ [`WholeFieldOutcome::Conforms`]
//!   with a certificate (the certified deviation enclosure — a sound bound on
//!   the max field deviation, no escape between samples);
//! - a concrete point whose sound zero-width enclosure certainly exceeds `tol`
//!   ⇒ [`WholeFieldOutcome::Violates`] with a witness and a locator region;
//! - bounds too loose and no certain violation found ⇒
//!   [`WholeFieldOutcome::Unknown`] (a CROWN-loose case that the caller may
//!   escalate to the exact SMT / Route-B path, [`crate::escalate`]).
//!
//! # Honest scope (the modeling boundary)
//!
//! What is *certified* is a property of the surrogate `f`: over the whole
//! region, `|f − g| ≤ tol` (Conforms), or a point where it fails (Violates).
//! The surrogate's own fidelity to the *physical* surface between the measured
//! points is a **modeling assumption**, documented and owned by the caller —
//! the same honest boundary as the ground-truth M-series (the guarantee is
//! "no escape *in the model* `f`", and `f`'s faithfulness to the scan is the
//! upstream fit-quality question). What this replaces is the *sampled* verdict:
//! where sampled inspection can silently pass a field that spikes between
//! probes, a `Conforms` verdict here has proved there is no such spike in `f`.
//!
//! # Example
//!
//! ```rust
//! use ny_core::Bound;
//! use ny_groundtruth::{signed_plane_distance, verify_whole_field_tolerance, WholeFieldOutcome};
//! # fn main() -> Result<(), ny_groundtruth::GroundTruthError> {
//! // Nominal: the plane z = 0 (a flat mating face), as a ground-truth graph.
//! let nominal = signed_plane_distance([0.0, 0.0, 1.0], 0.0)?;
//! // Measured-field surrogate: here the nominal itself (deviation ≡ 0) — a
//! // real surrogate would be a fitted NN / interpolant loaded as a GraphNetwork.
//! let field = signed_plane_distance([0.0, 0.0, 1.0], 0.0)?;
//! let region = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0), Bound::new(-0.5, 0.5)];
//! match verify_whole_field_tolerance(&field, &nominal, &region, 0.1)? {
//!     WholeFieldOutcome::Conforms { cert } => {
//!         println!("whole-field conformance: |f − g| ≤ {} over the whole region", cert.tol);
//!     }
//!     other => panic!("expected Conforms, got {other:?}"),
//! }
//! # Ok(())
//! # }
//! ```

use ny_core::Bound;
use ny_propagate::GraphNetwork;

use crate::error::Result;
use crate::verify::{
    verify_against_ground_truth_with, GroundTruthOutcome, Relation, VerifyOptions,
};

/// A whole-field tolerance certificate: a sound enclosure of the deviation
/// field `f − g` over the ENTIRE region, proving `|f − g| ≤ tol` everywhere.
///
/// The enclosure is the CROWN-certified output bound of the difference network
/// (the same sound bound the M1 verify path returns); by construction each
/// output bound lies inside `[−tol, tol]`. It is a *whole-field* fact — there
/// is no point in the box, sampled or not, where the deviation escapes it.
#[derive(Debug, Clone)]
pub struct WholeFieldCertificate {
    /// Certified per-output enclosure of the deviation field `f − g` over the
    /// whole box. Each `Bound` is `⊆ [−tol', tol']` for the sound-rounded
    /// tolerance `tol' ≤ tol`.
    pub deviation_bounds: Vec<Bound>,
    /// The requested tolerance the deviation was proved to respect (the checked
    /// property is at least this strong: the verifier rounds `tol` *down* to
    /// f32, so `Conforms` implies `|f − g| ≤ tol`).
    pub tol: f64,
}

impl WholeFieldCertificate {
    /// The certified worst-case absolute deviation over the whole region:
    /// `max(|lo|, |hi|)` across every output. By construction `≤ tol` — this is
    /// the proved "max field deviation", the number sampled inspection can only
    /// estimate.
    #[must_use]
    pub fn max_abs_deviation(&self) -> f32 {
        self.deviation_bounds
            .iter()
            .map(|b| b.lower().abs().max(b.upper().abs()))
            .fold(0.0_f32, f32::max)
    }
}

/// Outcome of a whole-field tolerance query.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum WholeFieldOutcome {
    /// Proved on the whole region: `|f − g| ≤ tol` everywhere, with a
    /// certificate ([`WholeFieldCertificate`]) carrying the certified
    /// deviation enclosure.
    Conforms {
        /// The whole-field conformance certificate.
        cert: WholeFieldCertificate,
    },
    /// A concrete point in the region certainly violates the tolerance: the
    /// sound enclosure of `f(x*) − g(x*)` lies entirely outside `[−tol, tol]`.
    Violates {
        /// The witness point `x*` (inside the region) whose deviation is
        /// *certified* out of tolerance.
        witness: Vec<f32>,
        /// A small neighborhood box around `x*` — a locator for where the
        /// field escapes tolerance (for refinement / display, e.g. "near
        /// (u, v)"). The *certified* fact is the point violation at `witness`;
        /// this box is a search neighborhood, not a proved all-points-violate
        /// region.
        witness_region: Vec<Bound>,
        /// Sound enclosure of the violating deviation `f(x*) − g(x*)`.
        difference: Bound,
    },
    /// Neither proved nor concretely falsified: the CROWN enclosure of the
    /// deviation field is too loose to fit `[−tol, tol]` and the witness grid
    /// found no certain violation. The honest non-answer — escalate to the
    /// exact SMT path ([`crate::escalate`]) or refine the region / grid.
    Unknown {
        /// Best achieved bounds on `f − g` over the region.
        deviation_bounds: Vec<Bound>,
    },
}

/// Half-width of the witness locator box as a fraction of each input
/// dimension's width (a display neighborhood around the certified witness
/// point; see [`WholeFieldOutcome::Violates::witness_region`]).
const WITNESS_LOCATOR_FRACTION: f32 = 0.05;

/// Verify the surrogate measured field `field` against the nominal ground-truth
/// graph `nominal` over the whole `region` box, to tolerance `tol` (default
/// options: CROWN + a 5-per-dimension witness grid).
///
/// Proves `∀ x ∈ region: |field(x) − nominal(x)| ≤ tol` — a whole-field
/// (continuous-domain) tolerance guarantee, not a point-sampled one. See the
/// [module docs](crate::wholefield) for the certified-vs-modeling boundary.
///
/// # Errors
/// As [`verify_against_ground_truth_with`]: a non-finite / non-positive `tol`,
/// an `f`/`g` shape mismatch, or a propagation error.
pub fn verify_whole_field_tolerance(
    field: &GraphNetwork,
    nominal: &GraphNetwork,
    region: &[Bound],
    tol: f64,
) -> Result<WholeFieldOutcome> {
    verify_whole_field_tolerance_with(field, nominal, region, tol, &VerifyOptions::default())
}

/// [`verify_whole_field_tolerance`] with explicit [`VerifyOptions`] (CROWN
/// configuration + witness-grid resolution).
///
/// # Errors
/// As [`verify_whole_field_tolerance`].
pub fn verify_whole_field_tolerance_with(
    field: &GraphNetwork,
    nominal: &GraphNetwork,
    region: &[Bound],
    tol: f64,
    options: &VerifyOptions,
) -> Result<WholeFieldOutcome> {
    let outcome =
        verify_against_ground_truth_with(field, nominal, Relation::AbsBound(tol), region, options)?;
    Ok(match outcome {
        GroundTruthOutcome::Verified { difference_bounds } => WholeFieldOutcome::Conforms {
            cert: WholeFieldCertificate {
                deviation_bounds: difference_bounds,
                tol,
            },
        },
        GroundTruthOutcome::Falsified {
            witness,
            difference,
        } => {
            let witness_region = locator_box(&witness, region);
            WholeFieldOutcome::Violates {
                witness,
                witness_region,
                difference,
            }
        }
        GroundTruthOutcome::Unknown { difference_bounds } => WholeFieldOutcome::Unknown {
            deviation_bounds: difference_bounds,
        },
    })
}

/// Build a small locator box around the witness point: each dimension is
/// `x*ᵢ ± (WITNESS_LOCATOR_FRACTION · widthᵢ)`, clamped to the region so it can
/// never leave it. A degenerate (zero-width) region dimension stays degenerate.
fn locator_box(witness: &[f32], region: &[Bound]) -> Vec<Bound> {
    witness
        .iter()
        .zip(region.iter())
        .map(|(&x, b)| {
            let half = (b.upper() - b.lower()) * WITNESS_LOCATOR_FRACTION;
            let lo = (x - half).max(b.lower());
            let hi = (x + half).min(b.upper());
            // Guard against FP making lo > hi at a clamped corner.
            Bound::new_allow_infinite(lo.min(hi), hi.max(lo))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signed_plane_distance;

    #[test]
    fn locator_box_stays_inside_region_and_hugs_witness() {
        let region = vec![Bound::new(0.0, 1.0), Bound::new(-2.0, 2.0)];
        let witness = vec![0.0, 2.0]; // both on a corner of the region
        let loc = locator_box(&witness, &region);
        for (b, r) in loc.iter().zip(region.iter()) {
            assert!(
                b.lower() >= r.lower() && b.upper() <= r.upper(),
                "inside region"
            );
            assert!(b.lower() <= b.upper(), "ordered");
        }
        // A witness in the interior yields a symmetric neighborhood.
        let inner = locator_box(&[0.5, 0.0], &region);
        assert!(inner[0].lower() < 0.5 && 0.5 < inner[0].upper());
        assert!(inner[1].lower() < 0.0 && 0.0 < inner[1].upper());
    }

    #[test]
    fn certificate_max_abs_deviation() {
        let cert = WholeFieldCertificate {
            deviation_bounds: vec![Bound::new(-0.03, 0.02), Bound::new(0.01, 0.04)],
            tol: 0.1,
        };
        assert!((cert.max_abs_deviation() - 0.04).abs() < 1e-6);
    }

    #[test]
    fn identical_field_and_nominal_conform_trivially() {
        // Deviation ≡ 0; any positive tolerance conforms, cert bound is tight.
        let g = signed_plane_distance([0.0, 0.0, 1.0], 0.0).unwrap();
        let f = signed_plane_distance([0.0, 0.0, 1.0], 0.0).unwrap();
        let region = vec![
            Bound::new(-1.0, 1.0),
            Bound::new(-1.0, 1.0),
            Bound::new(-0.5, 0.5),
        ];
        match verify_whole_field_tolerance(&f, &g, &region, 0.1).unwrap() {
            WholeFieldOutcome::Conforms { cert } => {
                assert!(cert.max_abs_deviation() <= 0.1);
            }
            other => panic!("expected Conforms, got {other:?}"),
        }
    }
}
