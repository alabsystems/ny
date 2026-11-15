// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential model verification paths.
//!
//! Handles Network (non-graph) verification including:
//! - Reduced max-difference verification for targeted robustness
//! - Per-constraint iteration with timeout budgeting
//! - PGD attack integration

use anyhow::Result;
use ndarray::Array2;
use ny_core::GemmEngine;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    layers::{LinearLayer, MaxPool2dLayer, ReshapeLayer},
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, Layer as PropLayer,
    Network, PgdCancelToken, PgdConfig,
};
use ny_tensor::BoundedTensor;

use super::super::attack_concurrent::{self, ConcurrentAttackCtx, RacePlan, RaceSite};
use super::super::verification_outcome::VerificationOutcome;
use super::constraint_iter::{iterate_constraints, ConstraintIterConfig};
use super::disjunctive_pgd::beta_crown_pgd_config;
use super::phase_budget::PhaseBudgetLedger;
use super::{augment_network_with_spec, classify_constraints, AggregationMode};

/// Detect the same-LHS relational reduction pattern used by reduced max-diff
/// verification, and return its normalized form.
///
/// Normalizes all relational constraints to (lhs, rhs) pairs where the LHS index
/// is consistent across all constraints. LessEq(i, j) is equivalent to
/// GreaterEq(j, i), so when the literal same-LHS check fails, everything is
/// re-normalized to the "ge" direction (this handles ACAS Xu prop_2:
/// Y_1<=Y_0, Y_2<=Y_0, ... → Y_0>=Y_1, Y_0>=Y_2, ...).
///
/// Returns `Some((family, lhs_idx, sorted-deduped rhs_indices))` when the
/// reduction applies, `None` otherwise (e.g. any constant constraint present).
///
/// Multi-row conjunctive dispatch now preserves these rows in the graph
/// multi-objective lane. This detector remains the canonical normalization for
/// the scalar reduced fallback and for row-parity tests.
pub(super) fn normalize_same_lhs_reduction(
    vnnlib: &VnnLibSpec,
) -> Option<(&'static str, usize, Vec<usize>)> {
    let mut rel_family: Option<&'static str> = None; // "ge" or "le"
    let mut lhs_idx: Option<usize> = None;
    let mut rhs_indices: Vec<usize> = Vec::new();

    // First pass: try same-LHS directly
    let mut same_lhs_ok = true;
    for c in &vnnlib.output_constraints {
        match c {
            OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                if rel_family.is_none() {
                    rel_family = Some("ge");
                }
                if rel_family != Some("ge") {
                    same_lhs_ok = false;
                    break;
                }
                if lhs_idx.is_none() {
                    lhs_idx = Some(*i);
                }
                if lhs_idx != Some(*i) {
                    same_lhs_ok = false;
                    break;
                }
                rhs_indices.push(*j);
            }
            OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                if rel_family.is_none() {
                    rel_family = Some("le");
                }
                if rel_family != Some("le") {
                    same_lhs_ok = false;
                    break;
                }
                if lhs_idx.is_none() {
                    lhs_idx = Some(*i);
                }
                if lhs_idx != Some(*i) {
                    same_lhs_ok = false;
                    break;
                }
                rhs_indices.push(*j);
            }
            _ => {
                same_lhs_ok = false;
                break;
            }
        }
    }

    // Second pass: if same-LHS failed, try normalizing LessEq(i,j) → GreaterEq(j,i)
    if !same_lhs_ok {
        rel_family = None;
        lhs_idx = None;
        rhs_indices.clear();
        for c in &vnnlib.output_constraints {
            // Normalize: flip LessEq/LessThan to GreaterEq/GreaterThan
            let (normalized_family, i, j) = match c {
                OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                    ("ge", *i, *j)
                }
                OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                    ("ge", *j, *i) // Flip: LessEq(i,j) → GreaterEq(j,i)
                }
                _ => return None,
            };
            if rel_family.is_none() {
                rel_family = Some(normalized_family);
            }
            if rel_family != Some(normalized_family) {
                return None;
            }
            if lhs_idx.is_none() {
                lhs_idx = Some(i);
            }
            if lhs_idx != Some(i) {
                return None;
            }
            rhs_indices.push(j);
        }
    }

    rhs_indices.sort_unstable();
    rhs_indices.dedup();
    let (Some(family), Some(lhs)) = (rel_family, lhs_idx) else {
        return None;
    };
    if rhs_indices.is_empty() {
        return None;
    }
    Some((family, lhs, rhs_indices))
}

/// Verify sequential model with relational constraints.
// Justification: Sequential verification forwards the same parameter set as graph
// verification — network, bounds, constraints, config, BaB/attack flags, engine.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_sequential_relational(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    pgd_attack: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    ledger: &PhaseBudgetLedger,
    // #attack-concurrent: the dispatch's race context (`None` on every
    // nested / test caller). Consulted only when `config.attack_concurrent`.
    concurrent: Option<&ConcurrentAttackCtx<'_>>,
) -> Result<BetaCrownResult> {
    verify_sequential_relational_outcome(
        network,
        input,
        vnnlib,
        config,
        verifier,
        pgd_attack,
        pgd_restarts,
        pgd_steps,
        timeout,
        gemm_engine,
        json,
        ledger,
        false,
        concurrent,
    )?
    .into_ordinary()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_sequential_relational_outcome(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    pgd_attack: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    ledger: &PhaseBudgetLedger,
    allow_instance_hull: bool,
    // #attack-concurrent: the dispatch's race context (`None` on every
    // nested / test caller). Consulted only when `config.attack_concurrent`.
    concurrent: Option<&ConcurrentAttackCtx<'_>>,
) -> Result<VerificationOutcome> {
    let aggregation = classify_constraints(vnnlib).aggregation;
    let is_disjunction = aggregation == AggregationMode::Disjunctive;
    let start_time = std::time::Instant::now();
    let pgd_deadline = ledger.upfront_pgd_deadline();

    // Early PGD attack before any CROWN computation. For categories where most
    // instances are violated (e.g., soundnessbench), PGD finds counterexamples in
    // seconds, avoiding the expensive 40% reduced-verification CROWN pass on
    // Conv-heavy networks. This does NOT affect soundness: PGD only reports
    // violations, verified by an independent forward pass.
    //
    // Single-clause disjunctions (output_constraint_clauses.len() <= 1) are
    // semantically conjunctive — all constraints in the single clause must hold
    // simultaneously. PGD on the conjunctive interpretation is correct.
    //
    // Match the graph path's budget split: reserve 80% of the total timeout
    // for reduced verification + BaB so expensive sequential PGD cannot consume
    // the whole wall-clock budget before verification begins.
    let is_single_clause = vnnlib.output_constraint_clauses.len() <= 1;
    if pgd_attack && (!is_disjunction || is_single_clause) {
        // #attack-concurrent (dark; typed `attack.concurrent`): race this
        // IDENTICAL attack against the proof below instead of running it
        // first. With the key off this reads one bool; any admission refusal
        // falls through to the serial code unchanged.
        if config.attack_concurrent {
            let site_start = std::time::Instant::now();
            if let Some(admitted) = attack_concurrent::admit_site(
                RaceSite::SeqUpfront,
                concurrent,
                // The attack and the proof run on the SAME handle here.
                attack_concurrent::shared_device(gemm_engine, gemm_engine, false),
                attack_concurrent::network_has_sign(network),
                site_start,
                pgd_deadline,
                ledger.overall_deadline(),
            ) {
                let base = beta_crown_pgd_config(config, pgd_restarts, pgd_steps, pgd_deadline);
                let attack = |token: PgdCancelToken, deadline: std::time::Instant| {
                    let pgd = PgdConfig {
                        deadline: Some(deadline),
                        cancel: Some(token),
                        ..base.clone()
                    };
                    // The serial site's own engine: admission refused an
                    // accelerator (it would be the proof's too), so it is a
                    // CPU handle (or `None`).
                    super::pgd::try_conjunctive_pgd_attack_upfront_with_config(
                        network,
                        input,
                        vnnlib,
                        pgd,
                        gemm_engine,
                        json,
                    )
                };
                // The reduced lane's post-inconclusive PGD repeats THIS attack
                // (same classified attacker, same seed) and is bounded by the
                // same upfront deadline the serial order had already spent; the
                // race's own attack covers it, so the proof skips it.
                //
                // The race carries the proof's REPORT; a retained original-hull
                // proof (origin's VerificationOutcome) is held aside and rides
                // with the report only when the race returned the proof's own
                // `Verified` (an attack candidate returned instead drops it,
                // exactly as the serial order never reaches the hull after a
                // PGD hit).
                #[cfg(feature = "mip")]
                let hull = std::cell::RefCell::new(None);
                let proof = || {
                    let outcome = verify_sequential_relational_proof(
                        network,
                        input,
                        vnnlib,
                        config,
                        verifier,
                        aggregation,
                        false,
                        pgd_restarts,
                        pgd_steps,
                        pgd_deadline,
                        timeout,
                        gemm_engine,
                        json,
                        ledger,
                        start_time,
                        allow_instance_hull,
                    )?;
                    #[cfg(feature = "mip")]
                    {
                        let (report, proof) = outcome.into_parts();
                        *hull.borrow_mut() = proof;
                        Ok(report)
                    }
                    #[cfg(not(feature = "mip"))]
                    outcome.into_ordinary()
                };
                let report = attack_concurrent::run_race(
                    RacePlan {
                        site: RaceSite::SeqUpfront,
                        window: admitted.window,
                        site_start,
                        slice_deadline: pgd_deadline.unwrap_or(site_start),
                        overall_deadline: ledger.overall_deadline(),
                        lane_start: start_time,
                        publish: concurrent
                            .map(|ctx| ctx.publish)
                            .unwrap_or(&attack_concurrent::unarmed_publish),
                    },
                    admitted.pool,
                    attack,
                    proof,
                )?;
                #[cfg(feature = "mip")]
                if let Some(proof) = hull.into_inner() {
                    if matches!(report.result, BabVerificationStatus::Verified) {
                        return Ok(VerificationOutcome::hull(report, proof));
                    }
                }
                return Ok(VerificationOutcome::ordinary(report));
            }
        }
        let early_pgd = super::pgd::try_conjunctive_pgd_attack_upfront_with_config(
            network,
            input,
            vnnlib,
            beta_crown_pgd_config(config, pgd_restarts, pgd_steps, pgd_deadline),
            gemm_engine,
            json,
        )?;
        if let Some((counterexample, output)) = early_pgd {
            return Ok(VerificationOutcome::ordinary(BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample: counterexample.iter().copied().collect(),
                    output: output.iter().copied().collect(),
                },
                domains_explored: 0,
                domains_verified: 0,
                cuts_generated: 0,
                max_depth_reached: 0,
                time_elapsed: start_time.elapsed(),
                output_bounds: None,
            }));
        }
    }

    verify_sequential_relational_proof(
        network,
        input,
        vnnlib,
        config,
        verifier,
        aggregation,
        pgd_attack,
        pgd_restarts,
        pgd_steps,
        pgd_deadline,
        timeout,
        gemm_engine,
        json,
        ledger,
        start_time,
        allow_instance_hull,
    )
}

/// The proof half of [`verify_sequential_relational`]: everything after the
/// upfront attack, unchanged. #attack-concurrent extracted it so the serial
/// path (after the attack) and the concurrent race (alongside it) run the
/// IDENTICAL code under the identical ledger deadlines. `pgd_attack` reaches
/// only the reduced lane's post-inconclusive PGD (see the race site).
// Justification: the proof consumes the same independent inputs as the lane.
#[allow(clippy::too_many_arguments)]
fn verify_sequential_relational_proof(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    aggregation: AggregationMode,
    pgd_attack: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    pgd_deadline: Option<std::time::Instant>,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    ledger: &PhaseBudgetLedger,
    start_time: std::time::Instant,
    allow_instance_hull: bool,
) -> Result<VerificationOutcome> {
    #[cfg(not(feature = "mip"))]
    let _ = allow_instance_hull;
    let is_disjunction = aggregation == AggregationMode::Disjunctive;

    // #hull-root / #hull-bab (wave-6 C4; dark unless `bab.single_hidden_hull`
    // is armed): the CERTIFIED per-neuron hull — one root bound (`root`) or a
    // best-first ReLU-split tree over the concave neurons (`bab`) — after the
    // upfront attack and BEFORE the same-LHS reduction, whose input-split BaB
    // would otherwise own the whole deadline. `Verified` only on a strictly
    // positive closed-form certificate (for `bab`: on EVERY leaf); a decline
    // or an undecided run falls through with whatever wall it did not spend
    // (the reduction below re-derives its budget from the ledger's absolute
    // deadline). The lane prints `[hull-root] armed ...` / `[hull-bab] armed
    // ...` on every armed run.
    #[cfg(feature = "mip")]
    if config.single_hidden_hull.is_armed() {
        if config.single_hidden_hull == ny_propagate::SingleHiddenHullMode::Root
            && allow_instance_hull
        {
            let now = std::time::Instant::now();
            let deadline =
                super::super::single_hidden_hull::hull_root_deadline(now, ledger.remaining());
            let deadline = ledger
                .overall_deadline()
                .map_or(deadline, |d| d.min(deadline));
            if let Some(resources) = ledger.verification_resources() {
                let context = resources.context(deadline);
                match super::super::single_hidden_hull::resource::try_single_hidden_hull_root_in(
                    config.single_hidden_hull, network, input, vnnlib, context.query(), start_time,
                ) {
                    Ok(checked) => {
                        match super::super::hull_publication::HullPublicationProof::from_checked(checked, context) {
                            Ok((report, proof)) => return Ok(VerificationOutcome::hull(report, proof)),
                            Err(reason) => eprintln!("[hull-resource] publication_owner_refused reason={reason} legacy_fallback=unqualified"),
                        }
                    }
                    Err(reason) => {
                        eprintln!("[hull-resource] declined reason={reason} census=participating-hull-only legacy_fallback=unqualified");
                        crate::flight::note("hull_resource", crate::flight::FlightStatus::Ran,
                            Some(format!("outcome=declined reason={reason} census=participating-hull-only legacy_fallback=unqualified")));
                    }
                }
            }
            if ledger
                .verification_resources()
                .is_some_and(|owner| owner.is_cancelled())
            {
                return Ok(VerificationOutcome::ordinary(BetaCrownResult {
                    result: BabVerificationStatus::Unknown {
                        reason: "instance resource owner cancelled".to_owned(),
                    },
                    domains_explored: 0,
                    domains_verified: 0,
                    cuts_generated: 0,
                    max_depth_reached: 0,
                    time_elapsed: start_time.elapsed(),
                    output_bounds: None,
                }));
            }
            // Preserve AY LP/canonicalization and the same original Root slice.
            if let Some(result) = super::super::single_hidden_hull::try_single_hidden_hull_root(
                config.single_hidden_hull,
                network,
                input,
                vnnlib,
                deadline,
                start_time,
            ) {
                return Ok(VerificationOutcome::ordinary(result));
            }
        } else if let Some(result) = super::super::single_hidden_hull::try_single_hidden_hull(
            config.single_hidden_hull,
            network,
            input,
            vnnlib,
            ledger.remaining(),
            start_time,
        ) {
            return Ok(VerificationOutcome::ordinary(result));
        }
    }

    // Reduced max-difference verification is only sound for conjunctive aggregation.
    if !is_disjunction {
        // Allocate reduced_verification_fraction of total timeout for reduced
        // verification. The max-diff augmented network (with MaxPool2d) can be
        // harder to verify than individual constraints. If inconclusive, we need
        // remaining time for per-constraint fallback. Without this limit, reduced
        // verification consumes the entire timeout.
        // Source: PhaseBudgetConfig.reduced_verification_fraction (default 0.40).
        //
        // Direct-call fallback: when the same-LHS reduction pattern
        // applies, the max-diff objective semantically SUBSUMES every
        // per-constraint check — per-constraint-i success means min_x m_i(x) > 0,
        // which implies min_x max_j m_j(x) > 0, the max-diff success criterion —
        // so reserving budget for the per-constraint fallback cannot recover
        // anything the max-diff BaB misses. Give the max-diff BaB the full
        // remaining wall-clock instead. Thread that exact absolute authority
        // rather than rebasing `remaining()` at engine entry: the latter moves
        // the deadline later and breaks strict timeout attribution. This is
        // what acasxu prop_2 needs: its
        // verifying conjunct varies across the input box, so per-constraint
        // decomposition is provably useless there (each conjunct individually
        // falsifiable — PGD found Y_1-Y_0 = -3.7e-4 < 0 on 1_5), while the 0.4
        // sub-budget starved the one engine that can decide it (26400 domains
        // at timeout under the old split). The default-dark SafeNLP
        // shared-prefix treatment is the narrow exception: it supplies an
        // earlier absolute deadline below so this prefix yields to its reserved
        // MIP slice. Gate-off and unrelated same-LHS categories retain the full
        // historical wall clock via the exact overall deadline. Ordinary
        // multi-row conjunctions are routed to graph any-row verification
        // before reaching this fallback.
        let reduction_applies = normalize_same_lhs_reduction(vnnlib).is_some();
        let reduced_timeout = if reduction_applies {
            ledger
                .remaining()
                .unwrap_or(std::time::Duration::from_secs(timeout))
        } else {
            std::time::Duration::from_secs(timeout).mul_f32(
                ledger
                    .policy()
                    .reduced_verification_fraction
                    .clamp(0.0, 1.0),
            )
        };
        let mut reduced_config = config.clone();
        reduced_config.timeout = reduced_timeout;
        let reduced_verifier = verifier.with_config_from(reduced_config);

        let reduced_result = try_reduced_verification(
            network,
            input,
            vnnlib,
            config,
            &reduced_verifier,
            pgd_attack,
            pgd_restarts,
            pgd_steps,
            pgd_deadline,
            reduced_verification_deadline(ledger, reduction_applies),
            gemm_engine,
            json,
        )?;

        if let Some(result) = reduced_result {
            match &result.result {
                BabVerificationStatus::Verified | BabVerificationStatus::Violated { .. } => {
                    return Ok(VerificationOutcome::ordinary(result));
                }
                BabVerificationStatus::Unknown { .. } | BabVerificationStatus::Timeout
                    if reduction_applies =>
                {
                    // Same-LHS reduction: per-constraint verification is subsumed
                    // by the max-diff objective (see the budget note above), so
                    // running it after an inconclusive full-budget max-diff BaB
                    // only burns wall-clock on checks that cannot succeed where
                    // max-diff failed. MEASURED 2026-07-10 (1_6/prop_2): the
                    // fall-through leaked ~25s into 3 useless per-constraint
                    // rounds after the max-diff phase. Return directly.
                    // PotentialViolation still falls through: it carries a
                    // concrete counterexample lead worth chasing in the
                    // per-constraint attack phases.
                    if !json {
                        println!(
                            "\nReduced (max-diff) verification inconclusive; skipping per-constraint fallback (subsumed by max-diff for same-LHS conjunctions)."
                        );
                    }
                    return Ok(VerificationOutcome::ordinary(result));
                }
                _ => {
                    // Reduced verification returned Unknown/PotentialViolation —
                    // fall through to per-constraint path which may succeed with
                    // tighter per-output bounds.
                    if !json {
                        println!(
                            "\nReduced verification inconclusive, falling back to per-constraint verification."
                        );
                    }
                }
            }
        }
    } else if !json {
        println!(
            "\nSkipping reduced relational verification for disjunctive property (requires all constraints provably violated)."
        );
    }

    // Fall back to per-constraint verification with remaining time budget.
    // Check actual Duration (not truncated as_secs()) to avoid false timeout
    // when sub-second time remains (e.g., 0.999s → as_secs()=0 → false exhaust).
    let remaining_duration = ledger.remaining().unwrap_or(std::time::Duration::ZERO);
    if timeout > 0 && remaining_duration.is_zero() {
        return Ok(VerificationOutcome::ordinary(BetaCrownResult {
            result: BabVerificationStatus::Timeout,
            domains_explored: 0,
            domains_verified: 0,
            cuts_generated: 0,
            max_depth_reached: 0,
            time_elapsed: start_time.elapsed(),
            output_bounds: None,
        }));
    }
    let remaining_timeout = if timeout == 0 {
        0 // unbounded
    } else {
        remaining_duration.as_secs().max(1)
    };

    verify_sequential_per_constraint(
        network,
        input,
        vnnlib,
        aggregation,
        config,
        verifier,
        pgd_attack,
        pgd_restarts,
        pgd_steps,
        remaining_timeout,
        gemm_engine,
        json,
        ledger.bab_deadline(),
    )
    .map(VerificationOutcome::ordinary)
}

/// Preserve one absolute authority for the reduced same-LHS engine call.
///
/// The default ACAS route owns the parent ledger's full wall clock. The
/// SafeNLP shared-prefix treatment is the sole earlier authority and therefore
/// takes precedence when armed. Non-reduced calls preserve their historical
/// treatment-only behavior.
fn reduced_verification_deadline(
    ledger: &PhaseBudgetLedger,
    reduction_applies: bool,
) -> Option<std::time::Instant> {
    let treatment_deadline = ledger.safenlp_shared_prefix_bab_deadline();
    if reduction_applies {
        treatment_deadline.or_else(|| ledger.overall_deadline())
    } else {
        treatment_deadline
    }
}

/// Preserve the signed objective exactly while avoiding a singleton MaxPool.
/// A 1x1 MaxPool is an identity, but its presence selects the sequential
/// alpha-CROWN unsupported-layer fallback. Multiple rows still need the
/// historical pointwise maximum, including its exact reshape layout.
fn reduced_objective_networks(
    network: &Network,
    spec_weight: Array2<f32>,
) -> Result<(Network, Network)> {
    let k = spec_weight.nrows();
    anyhow::ensure!(k > 0, "reduced verification requires a nonempty objective");
    let mut truncated = network.clone();
    truncated.add_layer(PropLayer::Linear(LinearLayer::new(spec_weight, None)?));
    let mut augmented = truncated.clone();
    if k > 1 {
        augmented.add_layer(PropLayer::Reshape(ReshapeLayer::new(vec![1, 1, k as i64])));
        augmented.add_layer(PropLayer::MaxPool2d(MaxPool2dLayer::new(
            (1, k),
            (1, k),
            (0, 0),
        )));
        augmented.add_layer(PropLayer::Reshape(ReshapeLayer::new(vec![1])));
    }
    Ok((truncated, augmented))
}

/// Try reduced verification for common targeted-robustness properties.
// Justification: Reduced verification optimization requires the full verification
// context to fall back to standard verification if the reduction doesn't apply.
#[allow(clippy::too_many_arguments)]
fn try_reduced_verification(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    pgd_attack: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    pgd_deadline: Option<std::time::Instant>,
    bab_deadline: Option<std::time::Instant>,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<BetaCrownResult>> {
    let Some((family, lhs, rhs_indices)) = normalize_same_lhs_reduction(vnnlib) else {
        return Ok(None);
    };

    let num_outputs = vnnlib.num_outputs;
    let k = rhs_indices.len();

    if !json {
        let constraint_desc = match family {
            "ge" => format!("Y_{} >= Y_j for j in {:?}", lhs, rhs_indices),
            "le" => format!("Y_{} <= Y_j for j in {:?}", lhs, rhs_indices),
            _ => "unknown".to_string(),
        };
        println!(
            "\nRelational constraint reduction detected: {}",
            constraint_desc
        );
        println!(
            "Reducing to single objective: maxdiff = max_j(signed_diff_j), verify maxdiff > 0"
        );
    }

    // Build Linear layer computing signed differences, one per RHS:
    // - For unsafe (Y_lhs >= Y_rhs): signed_diff = Y_rhs - Y_lhs
    // - For unsafe (Y_lhs <= Y_rhs): signed_diff = Y_lhs - Y_rhs
    let mut weights = vec![0.0f32; k * num_outputs];
    for (row, &rhs) in rhs_indices.iter().enumerate() {
        let row_start = row * num_outputs;
        match family {
            "ge" => {
                weights[row_start + rhs] = 1.0;
                weights[row_start + lhs] = -1.0;
            }
            "le" => {
                weights[row_start + lhs] = 1.0;
                weights[row_start + rhs] = -1.0;
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown relational family '{}' (expected 'ge' or 'le')",
                    family
                ))
            }
        }
    }

    let spec_weight = Array2::from_shape_vec((k, num_outputs), weights).map_err(|e| {
        anyhow::anyhow!(
            "failed to build relational spec weight with shape ({k}, {num_outputs}): {e}"
        )
    })?;

    // Signed-difference sub-network: [orig] + Linear(k signed diffs), WITHOUT the
    // maxpool tail. Its k outputs ARE the per-conjunct signed diffs whose
    // pointwise max is the max-diff objective — the joint-margin closer runs one
    // CROWN-with-linear pass on it per domain (see JointMarginCloser).
    let (truncated, augmented) = reduced_objective_networks(network, spec_weight)?;

    // JOINT-MARGIN closer (task #40): this legacy scalar fallback routes the
    // MaxPool lower relaxation through a single conjunct. The closer certifies
    // a tighter JOINT lower bound over ALL conjuncts per domain (min_x max_j g_j
    // via a sound dual certificate). Sound (only ever raises a domain's lower
    // bound).
    //
    // OPT-IN (NY_JOINT_MARGIN_LP=1), default-OFF — MEASURED NEGATIVE (task #40,
    // 1_5/prop_2, contention-matched A/B): the joint over the per-conjunct
    // CROWN-*backward affine* bounds NEVER beats the MaxPool baseline
    // (computed=20000/20000 domains, raw_improved=0, flipped=0, and up to 3.466
    // LOOSER). Root cause: the baseline scalar bound is rescued by
    // `tighten_crown_output` intersecting with the IBP-*forward* bound
    // `max_j(IBP_lower_j)`, which for acasxu is far tighter than the raw
    // CROWN-backward affine bounds the closer extracts (plain-IBP intermediates,
    // no alpha → loose ReLU relaxations). The joint aggregation needs tight
    // per-conjunct AFFINE bounds to express the x-coupling; here only the IBP
    // CONSTANTS are tight, and constants cannot couple. So the bottleneck is
    // per-conjunct affine looseness, not the aggregation — closing it needs
    // alpha / crown-ibp per-conjunct bounds (contraindicated for acasxu),
    // not this lever. Default-OFF makes the pipeline byte-identical to baseline;
    // the sound machinery + `NY_JOINT_MARGIN_DIAG` accounting are kept for
    // reproducibility and a future tight-affine integration.
    //
    // DARK by design: no typed preset key and no `run_instance.sh` export,
    // because the measurement above is NEGATIVE — there is nothing to deliver.
    // Revisit only alongside tight per-conjunct affine bounds.
    let joint_lp_enabled = std::env::var("NY_JOINT_MARGIN_LP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("on"))
        .unwrap_or(false);
    let reduced_result = if joint_lp_enabled && k >= 2 {
        let closer = ny_propagate::JointMarginCloser::new(std::sync::Arc::new(truncated), k);
        verifier
            .with_config_from(verifier.config.clone())
            .with_joint_margin_closer(std::sync::Arc::new(closer))
            .verify_with_engine(&augmented, input, 0.0, gemm_engine, bab_deadline)?
    } else {
        verifier.verify_with_engine(&augmented, input, 0.0, gemm_engine, bab_deadline)?
    };

    // If reduced verification couldn't prove property and PGD is enabled,
    // try conjunctive PGD attack to find counterexample
    if matches!(
        reduced_result.result,
        BabVerificationStatus::PotentialViolation { .. }
            | BabVerificationStatus::Unknown { .. }
            | BabVerificationStatus::Timeout
    ) && pgd_attack
    {
        if let Some(result) = super::pgd::try_conjunctive_pgd_attack_with_config(
            network,
            input,
            vnnlib,
            &reduced_result,
            beta_crown_pgd_config(config, pgd_restarts, pgd_steps, pgd_deadline),
            gemm_engine,
            json,
        )? {
            return Ok(Some(result));
        }
    }

    Ok(Some(reduced_result))
}

#[cfg(test)]
#[path = "sequential_singleton_tail_tests.rs"]
mod singleton_tail_tests;

/// Verify sequential model with per-constraint verification.
// Justification: Per-constraint verification iterates each constraint with the full
// verification context — grouping into a struct would duplicate BetaCrownConfig.
#[allow(clippy::too_many_arguments)]
fn verify_sequential_per_constraint(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    aggregation: AggregationMode,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    _pgd_attack: bool,
    _pgd_restarts: usize,
    _pgd_steps: usize,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    deadline: Option<std::time::Instant>,
) -> Result<BetaCrownResult> {
    // Use total constraint count (not just relational) since constants are now included
    let total_constraint_count = vnnlib.output_constraints.len();
    let is_disjunction = aggregation == AggregationMode::Disjunctive;

    // Budget timeout across constraints:
    // - Conjunctive (SAFE if ANY violated): each constraint gets the full remaining time.
    //   Only ONE needs to succeed, so don't split the budget — try each with full time
    //   and stop on first success. Splitting penalizes an easy constraint that could
    //   be solved with more time.
    // - Disjunctive (SAFE if ALL violated): split evenly since all must be proved.
    //   The minimum loop threshold is 10ms (not 100ms) to handle specs with many
    //   constraints (e.g., lindex 400 constraints at 30s = 74ms each, sufficient
    //   for quick BaB/PGD counterexample search).
    let per_constraint_timeout = if !is_disjunction {
        // Conjunctive: use full timeout for each (early exit on first success)
        std::time::Duration::from_secs(timeout)
    } else if total_constraint_count > 0 {
        // Disjunctive: split evenly with buffer
        let n = total_constraint_count.min(u32::MAX as usize - 1) as u32;
        std::time::Duration::from_secs(timeout) / (n + 1)
    } else {
        std::time::Duration::from_secs(timeout)
    };

    if !json {
        println!("\nRunning β-CROWN with constraints...");
        if is_disjunction {
            println!("Disjunctive property: SAFE if ALL constraints are provably violated.");
            println!(
                "Timeout budget: {:.1}s per constraint ({} constraints)",
                per_constraint_timeout.as_secs_f64(),
                total_constraint_count
            );
        } else {
            println!("Conjunctive property: SAFE if ANY constraint is provably violated.");
            println!(
                "Early-exit strategy: full timeout ({:.1}s) per constraint, stop on first success ({} constraints)",
                per_constraint_timeout.as_secs_f64(),
                total_constraint_count
            );
        }
    }

    // Note: conjunctive PGD attack was already run in verify_sequential_relational
    // (line 52-75) before reduced verification. The upfront attack now includes
    // a generic sampling+SPSA phase that checks random start points and intermediate
    // steps against ALL constraints. Running it again here would duplicate
    // work with the same seed, so we skip straight to per-constraint BaB.

    let iter_config = ConstraintIterConfig {
        aggregation,
        overall_timeout: std::time::Duration::from_secs(timeout),
        per_constraint_timeout,
        min_timeout_ms: 10,
        total_constraint_count,
        num_outputs: vnnlib.num_outputs,
        base_config: config.clone(),
        parent_verifier: Some(verifier),
        engine: verifier.engine_arc(),
        json,
    };

    // Cross-validation closure: evaluate original (un-augmented) network at a
    // concrete counterexample input. When per-constraint BaB finds a counterexample
    // for constraint i, this checks if that same input satisfies ALL constraints —
    // making it a joint counterexample proving the property VIOLATED.
    let eval_original = |cx_input: &[f32]| -> Result<ndarray::ArrayD<f32>> {
        let point = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(input.lower().shape()),
            cx_input.to_vec(),
        )?;
        super::pgd::evaluate_network(network, &point, gemm_engine)
    };

    iterate_constraints(
        &vnnlib.output_constraints,
        &iter_config,
        |dispatch| {
            let augmented = augment_network_with_spec(network, dispatch.spec_coeffs.to_vec())?;
            Ok(dispatch.verifier.verify_with_engine(
                &augmented,
                input,
                dispatch.threshold,
                gemm_engine,
                deadline,
            )?)
        },
        Some(&eval_original),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ny_onnx::vnnlib::parse_vnnlib;
    use std::time::{Duration, Instant};

    #[test]
    fn same_lhs_deadline_preserves_exact_overall_authority_and_treatment_precedence() {
        let exact_overall = Instant::now() + Duration::from_secs(100);
        let policy = ny_propagate::PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            ..Default::default()
        };
        let ordinary = PhaseBudgetLedger::from_deadline(Some(exact_overall), policy.clone())
            .with_mip_escalation_allowed(false);
        assert_eq!(
            reduced_verification_deadline(&ordinary, true),
            Some(exact_overall),
            "same-LHS ACAS must receive the parent's exact overall Instant"
        );
        assert_eq!(reduced_verification_deadline(&ordinary, false), None);

        let treatment = PhaseBudgetLedger::from_deadline(Some(exact_overall), policy)
            .with_mip_reservation(true)
            .with_safenlp_shared_prefix_budget_repair(true);
        assert_eq!(
            reduced_verification_deadline(&treatment, true),
            treatment.safenlp_shared_prefix_bab_deadline(),
            "the explicitly armed shared-prefix handoff remains the earlier authority"
        );
        assert!(
            reduced_verification_deadline(&treatment, true) < Some(exact_overall),
            "the treatment deadline must precede the overall wall"
        );
    }

    /// The same-LHS reduction is the SafeNLP shared prefix before exact MIP.
    /// Its verifier must honor the caller's phase deadline even though its
    /// cloned config still carries a much larger relative timeout.
    #[test]
    fn reduced_verification_honors_authoritative_bab_deadline() {
        let mut network = Network::new();
        network.add_layer(PropLayer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32], [-1.0_f32]]), None)
                .expect("two-output linear layer"),
        ));
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("finite input box");
        let vnnlib = parse_vnnlib(
            r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (<= Y_0 Y_1))
"#,
        )
        .expect("single relational constraint");
        let config = BetaCrownConfig {
            timeout: Duration::from_mins(1),
            use_alpha_crown: false,
            use_crown_ibp: false,
            enable_cuts: false,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config.clone());

        let result = try_reduced_verification(
            &network,
            &input,
            &vnnlib,
            &config,
            &verifier,
            false,
            0,
            0,
            None,
            Some(Instant::now()),
            None,
            true,
        )
        .expect("reduced verification should complete")
        .expect("same-LHS property must take the reduction");

        assert!(
            matches!(result.result, BabVerificationStatus::Timeout),
            "expired authoritative deadline must stop the reduced prefix, got {:?}",
            result.result
        );
    }
}
