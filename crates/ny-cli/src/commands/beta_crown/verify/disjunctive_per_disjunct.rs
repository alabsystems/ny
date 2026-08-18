// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-disjunct alpha multi-objective routing for disjunctive properties (#4355).
//!
//! When `optimize_disjuncts_separately` is enabled, routes to a shared BaB tree
//! with per-disjunct alpha optimization instead of splitting the time budget
//! across individual clauses.

use anyhow::Result;
use ny_core::GemmEngine;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, GraphNetwork};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};
use tracing::debug;

use super::disjunctive_unified::try_no_branchable_neuron_pgd_fallback;
use super::phase_budget::PhaseBudgetLedger;
use super::BetaCrownModel;

/// Route disjunctive property to multi-objective Graph BaB with per-disjunct alpha.
///
/// For Sequential models (MLP-only cora), converts to Graph first via
/// `GraphNetwork::from_sequential`.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_per_disjunct_multi_objective(
    model_net: &BetaCrownModel,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    clauses: &[Vec<OutputConstraint>],
    gemm_engine: Option<&dyn GemmEngine>,
    pgd_attack: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    json: bool,
    timeout: u64,
    overall_start: Instant,
    ledger: &PhaseBudgetLedger,
) -> Result<BetaCrownResult> {
    let graph = match model_net {
        BetaCrownModel::Graph(g) => std::borrow::Cow::Borrowed(g.as_ref()),
        BetaCrownModel::Sequential(network) => {
            let mut g = GraphNetwork::from_sequential(network)?;
            g.set_use_patches_mode(config.use_patches());
            std::borrow::Cow::Owned(g)
        }
    };

    let remaining_timeout = ledger.remaining_for_engine();
    if timeout > 0 && remaining_timeout.is_zero() {
        return Ok(BetaCrownResult {
            result: ny_propagate::BabVerificationStatus::Timeout,
            domains_explored: 0,
            time_elapsed: Duration::ZERO,
            max_depth_reached: 0,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        });
    }
    let remaining_config = BetaCrownConfig {
        timeout: remaining_timeout,
        ..config.clone()
    };

    debug!(
        clauses = clauses.len(),
        "Per-disjunct alpha fast-path (#4355): routing to multi-objective Graph BaB"
    );

    let remaining_verifier = verifier.with_config_from(remaining_config);
    let bab_deadline = ledger.bab_deadline();
    let (objectives, thresholds) = super::build_multi_objectives(vnnlib)?;

    if !json {
        println!("\nRunning multi-objective Graph β-CROWN (per-disjunct alpha) for disjunction...");
        println!(
            "Verifying {} constraints simultaneously (per-disjunct alpha optimization)",
            objectives.len()
        );
        println!("SAFE requires: ALL constraints provably violated");
    }

    let mut result = remaining_verifier.verify_graph_relu_split_multi_objective_with_engine(
        &graph,
        input,
        &objectives,
        &thresholds,
        gemm_engine,
        bab_deadline,
    )?;
    result.time_elapsed = overall_start.elapsed();

    if let Some(sat) = try_no_branchable_neuron_pgd_fallback(
        &result,
        pgd_attack,
        model_net,
        input,
        clauses,
        &vnnlib.per_clause_input_bounds,
        config,
        pgd_restarts,
        pgd_steps,
        gemm_engine,
        json,
        ledger,
    ) {
        return Ok(sat);
    }

    result.time_elapsed = overall_start.elapsed();
    Ok(result)
}
