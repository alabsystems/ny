// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The seam between `ny-falsify` and the scored path.
//!
//! `ny-falsify` is deliberately dependency-free: it cannot name `VnncompResult`,
//! it cannot open an ONNX file, and it cannot read the process environment. All
//! three of those things happen HERE, in `ny-cli`, and nowhere else. This module
//! is the whole of the coupling, and it is a child module of `vnncomp` on
//! purpose — it reuses that module's already-audited primitives
//! (`build_search_box`, `property_margin`, `property_violated_f64`,
//! `format_smtlib_witness_f64`, `rehydrate_original_witness_outputs`) rather
//! than growing a second copy of any of them.
//!
//! # What can and cannot come out of here
//!
//! The only value this module returns to the caller is `Option<String>`: an
//! SMT-LIB witness, or nothing. It never returns a verdict. The witness is
//! rendered INPUT-ONLY and its `Y_j` coordinates are then supplied by
//! [`super::rehydrate_original_witness_outputs`] — a real ONNX Runtime forward
//! on the ORIGINAL graph — exactly as the STE-PGD lane does it. A failed
//! re-forward DROPS the candidate. The caller then routes whatever comes back
//! through the UNCHANGED `gate_sat_with_trusted_oracle`, and a candidate the
//! gate declines falls through to the normal verification path.
//!
//! # Dark by default
//!
//! [`portfolio_falsify_armed`] is the first thing that runs and the only thing
//! that runs when the lane is disarmed. On that arm nothing is parsed, no
//! search box is built, no ORT session is constructed and no clock is consulted
//! past the lever read — the path is byte-identical to the tree without this
//! module.

use ny_falsify::strategies::{SpecialPoints, Square};
use ny_falsify::{
    Admission, Decline, FactLadder, GraphFacts, ObjectiveQuality, Oracle, OracleError, Proposal,
    Receipt, Registry, Score, SearchBox, SpecFacts, SpecShape,
};
use std::path::Path;
use std::time::{Duration, Instant};

/// Time reserved below the scored deadline for the confirming ORT re-check.
///
/// The same three seconds `UPFRONT_ATTACK_SAFETY_MARGIN` reserves, and for the
/// same reason: the publication path costs one more ORT session plus a forward.
const PORTFOLIO_SAFETY_MARGIN: Duration = Duration::from_secs(3);

/// Fraction of the remaining instance budget the portfolio phase may take.
///
/// The upfront attack's own fraction. This lane is scheduled ahead of it and
/// pays out of the same `attack_start` subtraction, so taking more would move
/// budget from the BaB proof, which is where 96% of ny's measured deficit is.
const PORTFOLIO_BUDGET_FRACTION: f64 = 0.08;

/// Hard ceiling on the phase, whatever the fraction works out to.
///
/// Sixty seconds because that is the budget
/// `reports/falsification_audit/selftest_calibration.json` was measured at: 75
/// of 100 known-SAT rows refuted, `special` winning at 2% of it. A cap above
/// the number the evidence was collected at would be an extrapolation.
const PORTFOLIO_WALL_CAP: Duration = Duration::from_mins(1);

/// Below this the phase is not worth entering.
const PORTFOLIO_MIN_BUDGET: Duration = Duration::from_millis(800);

/// Whether the portfolio lane is armed.
///
/// Environment only, and deliberately: there is no typed `attack.*` preset key,
/// so a competition harness (which exports no `NY_*`) cannot reach this lane at
/// all. Exact `"1"` arms it, exact `"0"` disarms it, any other byte string is a
/// recorded rejection resolving to the declaration's `false` default. Fails
/// CLOSED.
pub(crate) fn portfolio_falsify_armed() -> bool {
    ny_levers::read(&ny_levers::decls::dark_probes::FALSIFY_PORTFOLIO_LANE)
        .value
        .as_bool()
}

/// The operator-supplied wall cap in seconds, or `None` for the derived rule.
///
/// Read only after [`portfolio_falsify_armed`] has already returned true, so on
/// a default run this declaration has no reader.
fn portfolio_wall_cap() -> Option<Duration> {
    let seconds = ny_levers::read(&ny_levers::decls::dark_probes::FALSIFY_PORTFOLIO_SECONDS)
        .value
        .as_u64()
        .unwrap_or(0);
    (seconds > 0).then(|| Duration::from_secs(seconds))
}

/// Size the phase from what is left of the instance budget.
///
/// Frozen once, before any parse or model load, so every subsequent stage is
/// charged against this one slice instead of rebasing a fresh duration — the
/// rule `try_upfront_falsify` already follows.
fn portfolio_budget(remaining: Option<Duration>, cap: Option<Duration>) -> Option<Duration> {
    let ceiling = cap.unwrap_or(PORTFOLIO_WALL_CAP);
    let budget = match remaining {
        Some(rem) => rem
            .checked_sub(PORTFOLIO_SAFETY_MARGIN)?
            .mul_f64(PORTFOLIO_BUDGET_FRACTION)
            .max(if cap.is_some() {
                // An explicit cap is an instrument: honour it up to whatever the
                // instance actually has left, rather than silently shrinking it
                // to 8%. It is still clamped by `bounded_work_deadline` below.
                ceiling.min(rem.saturating_sub(PORTFOLIO_SAFETY_MARGIN))
            } else {
                Duration::ZERO
            })
            .min(ceiling),
        None => ceiling,
    };
    (budget >= PORTFOLIO_MIN_BUDGET).then_some(budget)
}

/// The evidence ladder, over ny's own VNN-LIB parse.
///
/// Both ported strategies declare `AdmissionStage::SpecShape`, so
/// [`FactLadder::graph_facts`] and [`FactLadder::load_model`] are never reached.
/// They are implemented as typed refusals rather than `unreachable!()` so that a
/// future strategy which declares a deeper stage gets a receipt instead of a
/// panic on a scored run.
struct SpecLadder {
    facts: SpecFacts,
}

impl FactLadder for SpecLadder {
    fn spec_facts(&mut self) -> Result<SpecFacts, Decline> {
        Ok(self.facts.clone())
    }

    fn graph_facts(&mut self) -> Result<GraphFacts, Decline> {
        Err(Decline::FragmentRefusal {
            fragment: "ny-cli/falsify-portfolio".to_string(),
            reason: "no ported strategy declares GraphScan; the ONNX graph is deliberately \
                     not scanned so a refusal costs one VNN-LIB parse and nothing else"
                .to_string(),
        })
    }

    fn load_model(&mut self) -> Result<(), Decline> {
        Err(Decline::FragmentRefusal {
            fragment: "ny-cli/falsify-portfolio".to_string(),
            reason: "no ported strategy declares ModelLoad".to_string(),
        })
    }
}

/// Read the structural facts admission is allowed to see off the parsed spec.
///
/// Note what is NOT read: no path, no filename, no category, no preset key. The
/// only inputs are the parsed property and the box derived from it.
fn spec_facts_from(spec: &ny_onnx::vnnlib::VnnLibSpec, domain: &SearchBox) -> SpecFacts {
    // A property with per-clause input boxes does not have a box domain: ny's
    // parser widens `input_bounds` to the UNION of the clause boxes, so most of
    // the box a strategy would search satisfies no clause at all and
    // `property_margin` returns -inf there. That is a flat, uninformative
    // landscape by construction rather than a hard problem, so report it as the
    // non-box shape and let both strategies decline structurally instead of
    // spending the slice. (Soundness never depended on this: the per-clause box
    // gate inside `property_violated_f64` is enforced at zero tolerance either
    // way.)
    let non_box = spec
        .per_clause_input_bounds
        .iter()
        .filter(|map| !map.is_empty())
        .count();
    let shape = if non_box > 0 {
        SpecShape::NonBoxInputAssertions {
            non_box_assertions: non_box,
        }
    } else if super::is_low_dim_common_rhs_relational_conjunction(spec) {
        SpecShape::LowDimRelationalConjunction
    } else {
        SpecShape::BoxInputs
    };
    SpecFacts {
        free_dims: domain.free_dims(),
        pinned_dims: domain.pinned_dims(),
        shape,
        disjunct_targets: spec.output_constraint_clauses.len().max(1),
        // `ny_onnx::vnnlib::OutputConstraint` has eight variants and none of
        // them is an equality, so this is `false` by construction rather than
        // by inspection. It stays a field because the chassis's contract has
        // it, and a parser that grows `Eq` should not silently keep reporting
        // no equality atoms.
        has_equality_atoms: false,
    }
}

/// The evaluator seam: one trusted ORT forward per proposed point.
///
/// `steer` is the exact parsed-property margin the APGD and corner lanes
/// already hill-climb, and `holds` is the exact zero-tolerance
/// [`super::property_violated_f64`] predicate they already accept on. Nothing
/// here is a new acceptance rule; it is the existing one, called from a
/// different search.
struct OrtPortfolioOracle<'a> {
    forward: &'a mut ny_onnx::diff::OrtForward,
    spec: &'a ny_onnx::vnnlib::VnnLibSpec,
    deadline: Instant,
    forwards: usize,
}

impl Oracle for OrtPortfolioOracle<'_> {
    /// One. `OrtForward::run` takes a single flat input and there is no batched
    /// entry point, so claiming more would make the strategies size their work
    /// units against a batch that does not exist. The trait's own contract says
    /// a non-batching oracle returns 1 and pays one forward per point.
    fn batch_limit(&self) -> usize {
        1
    }

    fn evaluate_batch(&mut self, points: &[Vec<f64>]) -> Result<Vec<Score>, OracleError> {
        let mut scores = Vec::with_capacity(points.len());
        for point in points {
            if Instant::now() >= self.deadline {
                return Err(OracleError("portfolio slice expired".to_string()));
            }
            // The point arrives in the EMIT VIEW already: `SearchBox` snapped
            // every free coordinate onto the float32 grid inside the box and
            // left every pinned coordinate at its exact declared f64. So this
            // cast is lossless on the free dimensions and reproduces exactly
            // the tensor the other lanes feed ORT on the pinned ones.
            let point32: Vec<f32> = point.iter().map(|&value| value as f32).collect();
            let raw = self
                .forward
                .run(&point32)
                .map_err(|error| OracleError(error.to_string()))?;
            self.forwards += 1;
            if Instant::now() >= self.deadline {
                return Err(OracleError("portfolio slice expired".to_string()));
            }
            let outputs: Vec<f64> = raw
                .iter()
                .map(|&value| super::f32_to_f64_exact(value))
                .collect();
            scores.push(Score {
                steer: super::property_margin(self.spec, &point32, &outputs),
                holds: super::property_violated_f64(self.spec, point, &outputs),
            });
        }
        Ok(scores)
    }
}

/// One line per strategy, so a decline is a receipt rather than a silence.
fn describe(receipt: &Receipt, forwards: usize) -> String {
    let mut parts = Vec::new();
    for admission in &receipt.admissions {
        let verdict = match &admission.admission {
            Admission::Admitted(profile) => format!("admitted({} free)", profile.free_dims),
            Admission::Declined(decline) => format!("declined({decline:?})"),
        };
        parts.push(format!(
            "{}: {verdict} [{:.4}s]",
            admission.strategy,
            admission.elapsed.as_secs_f64()
        ));
    }
    for (name, proposal) in &receipt.proposals {
        let outcome = match proposal {
            Proposal::Candidate(candidate) => format!(
                "CANDIDATE after {} points / {} batches",
                candidate.effort().points,
                candidate.effort().batches
            ),
            Proposal::Exhausted(effort) => format!(
                "exhausted after {} points / {} batches (best steer {:?})",
                effort.points, effort.batches, effort.best_steer
            ),
            Proposal::Declined(decline) => format!("declined({decline:?})"),
        };
        parts.push(format!("{name} ran: {outcome}"));
    }
    parts.push(format!("{forwards} trusted ORT forwards"));
    parts.join("; ")
}

/// Consult the dark `ny-falsify` portfolio and render any candidate as an
/// ORIGINAL-MODEL SMT-LIB witness for the caller's trusted-oracle gate.
///
/// Byte-for-byte the same publication path as `try_ste_pgd_falsify`: the
/// candidate is emitted INPUT-ONLY, its `Y_j` values come from
/// [`super::rehydrate_original_witness_outputs`] (a real ONNX Runtime forward on
/// the ORIGINAL graph), and a failed re-forward DROPS the candidate rather than
/// publishing the search's own arithmetic. Returns `None` for every
/// non-candidate outcome, including the default disarmed one.
pub(crate) fn try_portfolio_falsify(
    onnx: &Path,
    vnnlib: &Path,
    instance_deadline: Option<Instant>,
) -> Option<String> {
    if !portfolio_falsify_armed() {
        // Unarmed arm: no receipt entry, no stderr, no parse, no session.
        return None;
    }
    // Freeze admission before any VNN-LIB / model / ORT setup, exactly as the
    // upfront attack does, so every stage below is charged to this one slice.
    let started = Instant::now();
    let remaining = instance_deadline.map(|d| d.saturating_duration_since(started));
    let budget = portfolio_budget(remaining, portfolio_wall_cap())?;
    let deadline = super::bounded_work_deadline(started, budget, instance_deadline)?;
    if Instant::now() >= deadline {
        return None;
    }

    let spec = ny_onnx::vnnlib::load_vnnlib(vnnlib).ok()?;
    if spec.dual_network.is_some() || spec.num_inputs != spec.input_bounds.len() {
        // A dual-network relation is not searchable with one forward session,
        // and a spec whose declared arity disagrees with its bounds cannot be
        // materialised into a tensor. Both are structural and both are cheap.
        eprintln!(
            "Falsification portfolio: declined structurally (dual-network or arity mismatch) \
             [{:.2}s]",
            started.elapsed().as_secs_f64()
        );
        return None;
    }
    // Reuse the SAME box construction the upfront attack uses, so every
    // coordinate the portfolio can propose is one the existing lanes could have
    // proposed: bounds rounded INWARD to float32, degenerate dimensions pinned
    // at their exact declared f64, sub-ULP intervals pinned at their midpoint.
    let (box_lo, box_hi, emit_pin) = super::build_search_box(&spec)?;
    let lo: Vec<f64> = box_lo
        .iter()
        .zip(&emit_pin)
        .map(|(&value, pin)| pin.unwrap_or_else(|| super::f32_to_f64_exact(value)))
        .collect();
    let hi: Vec<f64> = box_hi
        .iter()
        .zip(&emit_pin)
        .map(|(&value, pin)| pin.unwrap_or_else(|| super::f32_to_f64_exact(value)))
        .collect();
    let domain = SearchBox::new(&lo, &hi).ok()?;
    let mut ladder = SpecLadder {
        facts: spec_facts_from(&spec, &domain),
    };

    if Instant::now() >= deadline {
        return None;
    }
    let mut forward = match ny_onnx::diff::OrtForward::from_path(onnx, box_lo.len()) {
        Ok(forward) => forward,
        Err(error) => {
            eprintln!("Falsification portfolio: trusted forward unavailable ({error}); skipping");
            return None;
        }
    };

    let mut registry = Registry::new()
        .with(Box::new(SpecialPoints))
        .with(Box::new(Square::default()))
        .armed();
    let mut oracle = OrtPortfolioOracle {
        forward: &mut forward,
        spec: &spec,
        deadline,
        forwards: 0,
    };
    let phase = deadline.saturating_duration_since(Instant::now());
    if phase.is_zero() {
        return None;
    }
    // `ObjectiveQuality` is a scheduler input the chassis carries for the
    // estimated-gradient strategies that were NOT ported. Both `special` and
    // `square` declare `ObjectiveRequirement::ValueOnly` and neither declines on
    // a flat objective, and `Registry::schedule` orders by `CostClass` alone —
    // so measuring flatness here would spend scored ORT forwards on a value
    // nothing reads. Reported as `Informative` and left as an owed item for
    // whenever an estimated-gradient strategy joins the registry.
    let outcome = registry.run(
        &mut ladder,
        &domain,
        &mut oracle,
        phase,
        ObjectiveQuality::Informative,
    );
    let forwards = oracle.forwards;

    let receipt = match outcome {
        Ok(receipt) => receipt,
        Err(decline) => {
            eprintln!(
                "Falsification portfolio: {decline:?} [{:.2}s]",
                started.elapsed().as_secs_f64()
            );
            return None;
        }
    };
    // The elapsed time is on the REFUSAL lines too, on purpose, for the same
    // reason it is on the two BNN lanes': armed, this lane spends scored budget
    // before every other family's attack, and a structural decline is only
    // "free" if it is measured to be.
    eprintln!(
        "Falsification portfolio: {} [{:.2}s]",
        describe(&receipt, forwards),
        started.elapsed().as_secs_f64()
    );
    let candidate = receipt.candidate()?;
    // Input-only witness first: the organizer parses these decimals AS WRITTEN,
    // and the outputs must come from the ORIGINAL model, never from the search.
    let input_only = super::format_smtlib_witness_f64(candidate.inputs(), &[]);
    match super::rehydrate_original_witness_outputs(onnx, &input_only) {
        Ok(witness) => Some(witness),
        Err(error) => {
            crate::flight::note(
                "falsify_portfolio",
                crate::flight::FlightStatus::Ran,
                Some(format!(
                    "candidate dropped: original-model re-forward failed ({error})"
                )),
            );
            eprintln!(
                "Falsification portfolio: could not re-forward the candidate through the \
                 ORIGINAL model ({error}); dropping it and continuing on the normal path"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lane_is_dark_with_no_lever_set() {
        // The production reader, on the production declaration. `read` consults
        // the process environment, and the harness does not set this.
        assert!(
            !ny_levers::read_with(
                &ny_levers::decls::dark_probes::FALSIFY_PORTFOLIO_LANE,
                |_| None
            )
            .value
            .as_bool(),
            "the portfolio lane must be dark with no lever set"
        );
        for token in ["0", "true", "yes", "", "1 "] {
            assert!(
                !ny_levers::read_with(
                    &ny_levers::decls::dark_probes::FALSIFY_PORTFOLIO_LANE,
                    |_| Some(token.to_string())
                )
                .value
                .as_bool(),
                "{token:?} must not arm the portfolio lane"
            );
        }
        assert!(
            ny_levers::read_with(
                &ny_levers::decls::dark_probes::FALSIFY_PORTFOLIO_LANE,
                |_| Some("1".to_string())
            )
            .value
            .as_bool(),
            "exact \"1\" arms it"
        );
    }

    #[test]
    fn the_phase_never_outgrows_the_instance_and_never_starts_below_the_floor() {
        // Derived rule: 8% of (remaining - 3s), capped at 60s.
        let budget = portfolio_budget(Some(Duration::from_secs(303)), None).unwrap();
        assert!((budget.as_secs_f64() - 24.0).abs() < 1e-9, "{budget:?}");
        // The cap binds on a long instance.
        let budget = portfolio_budget(Some(Duration::from_secs(3603)), None).unwrap();
        assert_eq!(budget, PORTFOLIO_WALL_CAP);
        // Below the floor the phase is not entered at all.
        assert!(portfolio_budget(Some(Duration::from_secs(4)), None).is_none());
        assert!(portfolio_budget(Some(Duration::from_secs(1)), None).is_none());
        // An explicit cap is honoured up to what the instance has left, and
        // never past it.
        let budget =
            portfolio_budget(Some(Duration::from_secs(303)), Some(Duration::from_mins(2))).unwrap();
        assert_eq!(budget, Duration::from_mins(2));
        let budget =
            portfolio_budget(Some(Duration::from_secs(53)), Some(Duration::from_mins(10))).unwrap();
        assert_eq!(budget, Duration::from_secs(50));
    }

    #[test]
    fn a_per_clause_input_box_is_reported_as_the_non_box_shape() {
        // nn4sys/lindex shape: `input_bounds` is the UNION of the clause boxes,
        // so a box search over it is searching a region the property does not
        // describe. Both strategies decline on the shape rather than spending
        // the slice on an all-minus-infinity landscape.
        let mut spec = ny_onnx::vnnlib::VnnLibSpec {
            num_inputs: 1,
            num_outputs: 1,
            input_bounds: vec![(0.0, 10.0)],
            output_constraints: Vec::new(),
            output_constraint_clauses: vec![Vec::new(), Vec::new()],
            is_disjunction: true,
            version: None,
            per_clause_input_bounds: vec![Default::default(), Default::default()],
            declared_input_bounds: Vec::new(),
            dual_network: None,
        };
        let domain = SearchBox::new(&[0.0], &[10.0]).unwrap();
        assert_eq!(spec_facts_from(&spec, &domain).shape, SpecShape::BoxInputs);

        spec.per_clause_input_bounds[0].insert(0, (0.0, 1.0));
        assert_eq!(
            spec_facts_from(&spec, &domain).shape,
            SpecShape::NonBoxInputAssertions {
                non_box_assertions: 1
            }
        );
    }

    #[test]
    fn the_ladder_never_offers_graph_or_model_evidence() {
        // Both ported strategies stop at `SpecShape`, so these two arms are
        // unreachable today. They are typed refusals rather than panics so a
        // future deeper strategy gets a receipt on a scored run.
        let mut ladder = SpecLadder {
            facts: SpecFacts {
                free_dims: 4,
                pinned_dims: 0,
                shape: SpecShape::BoxInputs,
                disjunct_targets: 1,
                has_equality_atoms: false,
            },
        };
        assert!(matches!(
            ladder.graph_facts(),
            Err(Decline::FragmentRefusal { .. })
        ));
        assert!(matches!(
            ladder.load_model(),
            Err(Decline::FragmentRefusal { .. })
        ));
        assert!(ladder.spec_facts().is_ok());
    }
}
