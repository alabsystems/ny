// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared per-constraint iteration engine.
//!
//! Extracts the common constraint iteration loop used by both graph and
//! sequential verification backends. The loop handles:
//! - Constraint objective building
//! - Timeout budgeting per constraint
//! - Status accumulation and display
//! - Early exit (conjunctive: first proved, disjunctive: first failure)
//! - Cross-validation of per-constraint BaB counterexamples (#3209)
//! - Final status computation
//!
//! Backends provide a dispatch closure for the actual verification call.
//! Part of #2215.

use std::sync::Arc;

use anyhow::Result;
use ndarray::ArrayD;
use ny_core::GemmEngine;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier};

use super::{
    build_constraint_objective, check_unsafe_counterexample, constraint_is_safety_proof,
    disjunctive_failure_to_final_status, finalize_relational_status, AggregationMode,
};

/// Configuration for the per-constraint iteration loop.
pub(super) struct ConstraintIterConfig<'a> {
    /// Aggregation mode (conjunctive vs disjunctive).
    pub aggregation: AggregationMode,
    /// Overall timeout for all constraints combined.
    pub overall_timeout: std::time::Duration,
    /// Pre-computed per-constraint timeout budget.
    pub per_constraint_timeout: std::time::Duration,
    /// Minimum viable timeout (ms) to attempt a constraint.
    pub min_timeout_ms: u64,
    /// Total constraint count for final status computation.
    pub total_constraint_count: usize,
    /// Number of model outputs (for objective building).
    pub num_outputs: usize,
    /// Base config to clone per constraint (timeout is overridden).
    pub base_config: BetaCrownConfig,
    /// Parent verifier to clone runtime-only resources from when present.
    pub parent_verifier: Option<&'a BetaCrownVerifier>,
    /// Stored GPU engine for per-constraint sub-verifiers (#3627).
    pub engine: Option<Arc<dyn GemmEngine>>,
    /// Suppress non-JSON output.
    pub json: bool,
}

/// Per-constraint dispatch context passed to the verification closure.
pub(super) struct ConstraintDispatch<'a> {
    /// Specification coefficients for this constraint.
    pub spec_coeffs: &'a [f32],
    /// Objective threshold.
    pub threshold: f32,
    /// Verifier with per-constraint timeout.
    pub verifier: &'a BetaCrownVerifier,
}

/// Run per-constraint iteration with a backend-specific dispatch closure.
///
/// `verify_fn` is called for each constraint with the dispatch context.
/// It returns a `BetaCrownResult` for that single constraint.
///
/// `eval_original` optionally evaluates the original (un-augmented) network at a
/// concrete input, returning the full output vector. When provided and in conjunctive
/// mode, per-constraint BaB counterexamples are cross-validated against ALL constraints.
/// If any single-constraint counterexample satisfies ALL constraints, the property is
/// immediately reported as Violated. This fixes #3209 where per-constraint BaB finds
/// counterexamples for each constraint independently but never checks if any input
/// satisfies all constraints simultaneously.
#[allow(clippy::type_complexity)]
pub(super) fn iterate_constraints<F>(
    constraints: &[OutputConstraint],
    config: &ConstraintIterConfig<'_>,
    mut verify_fn: F,
    eval_original: Option<&dyn Fn(&[f32]) -> Result<ArrayD<f32>>>,
) -> Result<BetaCrownResult>
where
    F: FnMut(&ConstraintDispatch<'_>) -> Result<BetaCrownResult>,
{
    let is_disjunction = config.aggregation == AggregationMode::Disjunctive;
    let overall_start = std::time::Instant::now();

    let mut proved_violated_count = 0usize;
    let mut total_domains = 0usize;
    let mut max_depth = 0usize;
    let mut total_time = std::time::Duration::ZERO;
    let mut constraint_results: Vec<(String, BetaCrownResult)> = Vec::new();
    let mut final_status_override: Option<BabVerificationStatus> = None;

    for (idx, constraint) in constraints.iter().enumerate() {
        let obj = build_constraint_objective(constraint, config.num_outputs)?;
        let spec_coeffs = obj.spec_coeffs().to_vec();
        let obj_threshold = obj.threshold();
        let constraint_desc = obj.constraint_desc().to_string();
        let diff_desc = obj.diff_desc();

        if overall_start.elapsed() >= config.overall_timeout {
            if !config.json {
                println!("\n  Overall timeout reached, stopping constraint iteration");
            }
            break;
        }

        let remaining = config
            .overall_timeout
            .saturating_sub(overall_start.elapsed());
        let this_timeout = config.per_constraint_timeout.min(remaining);
        if this_timeout.as_millis() < config.min_timeout_ms as u128 {
            break;
        }

        if !config.json {
            println!(
                "\n  Constraint {}: {} (verify {} > {}, timeout: {:.1}s)",
                idx + 1,
                constraint_desc,
                diff_desc,
                obj_threshold,
                this_timeout.as_secs_f64()
            );
        }

        let constraint_config = BetaCrownConfig {
            timeout: this_timeout,
            ..config.base_config.clone()
        };
        let constraint_verifier = match config.parent_verifier {
            Some(parent_verifier) => parent_verifier.with_config_from(constraint_config),
            None => match config.engine.clone() {
                Some(engine) => BetaCrownVerifier::new_with_engine(constraint_config, engine),
                None => BetaCrownVerifier::new(constraint_config),
            },
        };

        let dispatch = ConstraintDispatch {
            spec_coeffs: &spec_coeffs,
            threshold: obj_threshold,
            verifier: &constraint_verifier,
        };
        let constraint_result = verify_fn(&dispatch)?;

        total_domains += constraint_result.domains_explored;
        max_depth = max_depth.max(constraint_result.max_depth_reached);
        total_time += constraint_result.time_elapsed;

        if !config.json {
            let status_str = format_status(&constraint_result.result);
            println!(
                "    Result: {} ({} domains, {:.2}s)",
                status_str,
                constraint_result.domains_explored,
                constraint_result.time_elapsed.as_secs_f64()
            );
        }

        // Cross-validate per-constraint BaB counterexamples against ALL constraints.
        // In conjunctive mode, if BaB finds a concrete input violating constraint i,
        // that same input might satisfy ALL constraints — making it a joint counterexample
        // proving the property VIOLATED. This is essentially free (one forward pass).
        // Part of #3209.
        if !is_disjunction {
            if let BabVerificationStatus::Violated { counterexample, .. } =
                &constraint_result.result
            {
                if let Some(eval_fn) = &eval_original {
                    match eval_fn(counterexample) {
                        Ok(full_output) => {
                            if check_unsafe_counterexample(&full_output, constraints) {
                                if !config.json {
                                    println!(
                                        "\n  Cross-validation: constraint {} counterexample \
                                         satisfies ALL {} constraints! Property VIOLATED.",
                                        idx + 1,
                                        constraints.len()
                                    );
                                }
                                let output_vec: Vec<f32> = full_output.iter().copied().collect();
                                return Ok(BetaCrownResult {
                                    result: BabVerificationStatus::Violated {
                                        counterexample: counterexample.clone(),
                                        output: output_vec,
                                    },
                                    domains_explored: total_domains,
                                    domains_verified: 0,
                                    cuts_generated: 0,
                                    max_depth_reached: max_depth,
                                    time_elapsed: total_time,
                                    output_bounds: None,
                                });
                            }
                        }
                        Err(e) => {
                            tracing::debug!("Cross-validation forward pass failed: {e}");
                        }
                    }
                }
            }
        }

        constraint_results.push((constraint_desc.clone(), constraint_result.clone()));

        if constraint_is_safety_proof(&constraint_result.result) {
            proved_violated_count += 1;
            if !is_disjunction {
                if !config.json {
                    println!(
                        "\n  Early exit: constraint {} violated (conjunctive property satisfied)",
                        constraint_desc
                    );
                }
                break;
            }
        } else if is_disjunction {
            let disjunctive_status =
                disjunctive_failure_to_final_status(&constraint_result.result, &constraint_desc);
            if !config.json {
                match &disjunctive_status {
                    BabVerificationStatus::Violated { .. } => println!(
                        "\n  Early exit: counterexample found for constraint {} \
                         (disjunctive property violated)",
                        constraint_desc
                    ),
                    _ => println!(
                        "\n  Early exit: constraint {} not proved violated \
                         (disjunctive property requires ALL)",
                        constraint_desc
                    ),
                }
            }
            final_status_override = Some(disjunctive_status);
            break;
        }
    }

    let final_status = final_status_override.unwrap_or_else(|| {
        finalize_relational_status(
            config.aggregation,
            proved_violated_count,
            config.total_constraint_count,
            constraint_results.len(),
        )
    });

    Ok(BetaCrownResult {
        result: final_status,
        domains_explored: total_domains,
        domains_verified: proved_violated_count,
        cuts_generated: 0,
        max_depth_reached: max_depth,
        time_elapsed: total_time,
        output_bounds: None,
    })
}

/// Format a verification status for display.
fn format_status(status: &BabVerificationStatus) -> &'static str {
    match status {
        BabVerificationStatus::Verified => "VIOLATED (safe)",
        BabVerificationStatus::Violated { .. } => "VIOLATED (counterexample found)",
        BabVerificationStatus::PotentialViolation => "MAY HOLD",
        BabVerificationStatus::Unknown { .. } => "UNKNOWN",
        BabVerificationStatus::Timeout => "TIMEOUT",
    }
}
