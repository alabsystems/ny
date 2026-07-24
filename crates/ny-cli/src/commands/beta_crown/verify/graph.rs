// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph model per-constraint verification.
//!
//! Handles GraphNetwork verification with per-constraint iteration,
//! pre-computed α-CROWN bounds sharing, and multi-objective BaB for disjunctions.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::VnnLibSpec;
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, GraphNetwork,
    GraphPrecomputedBounds,
};
use ny_tensor::{next_down_f32, BoundedTensor};

use super::attack_budget::graph_upfront_pgd_budget;
use super::constraint_iter::{iterate_constraints, ConstraintIterConfig};
use super::disjunctive_pgd::beta_crown_pgd_config;
use super::dispatch_graph_constraint;
use super::graph_pgd::{evaluate_graph, try_graph_pgd_upfront_with_config};
use super::phase_budget::PhaseBudgetLedger;
use super::{build_multi_objectives, classify_constraints, AggregationMode};

/// Verify graph model with relational constraints.
// Justification: Graph verification requires model, bounds, constraints, config,
// verifier, BaB flags, attack parameters, and engine — all independent inputs.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_graph_relational(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    use_relu_split: bool,
    gpu_bab: bool,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<BetaCrownResult> {
    // Classify constraints via shared planning module (#1881)
    let classification = classify_constraints(vnnlib);
    let total_constraint_count = vnnlib.output_constraints.len();
    let is_disjunction = classification.aggregation == AggregationMode::Disjunctive;
    let ledger = PhaseBudgetLedger::new(timeout, config.phase_budget.clone());
    let start_time = std::time::Instant::now();

    // Early PGD attack for graph networks before expensive α-CROWN computation.
    // For Conv-heavy models (soundnessbench, malbeware conv), α-CROWN bounds
    // take 30-60s. Random sampling + SPSA PGD can find violations in <1s.
    //
    // Single-clause disjunctions (output_constraint_clauses.len() <= 1) are
    // semantically conjunctive — all constraints in the single clause must hold
    // simultaneously. PGD on the conjunctive interpretation is correct. (#3309)
    //
    // Reserve (1 - upfront_pgd_fraction) of timeout for CROWN + BaB (#3781).
    // Graph PGD can be expensive on large models.
    let is_single_clause = vnnlib.output_constraint_clauses.len() <= 1;
    if config.enable_pgd_attack && (!is_disjunction || is_single_clause) {
        let (pgd_restarts, pgd_steps) = graph_upfront_pgd_budget(config);
        let pgd_deadline = ledger.upfront_pgd_deadline();
        if let Some((counterexample, output)) = try_graph_pgd_upfront_with_config(
            graph,
            input,
            vnnlib,
            beta_crown_pgd_config(config, pgd_restarts, pgd_steps, pgd_deadline),
            gemm_engine,
            json,
        )? {
            return Ok(BetaCrownResult {
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
            });
        }
    }

    // Per-constraint timeout derived from ledger remaining time (#2206 Packet B).
    // Uses remaining wall-clock after PGD instead of the original raw timeout.
    // - Conjunctive: full remaining per constraint (early exit on first success).
    // - Disjunctive: splits remaining evenly, floored to min_clause_secs.
    let per_constraint_timeout = ledger
        .per_clause_timeout(total_constraint_count, is_disjunction)
        .unwrap_or(std::time::Duration::from_secs(timeout));

    // Certified sparse-input double-double zonotope (#dd-zonotope, dark
    // `NY_DD_ZONOTOPE=1`, default-OFF).
    //
    // This sits AHEAD of both the multi-objective and per-constraint branches
    // because `vggnet16_2022` specs carry exactly ONE output constraint, so the
    // multi-objective engine hook in
    // `beta_crown::engine::graph::multi_objective::root` is never reached for
    // them; and because the existing root pass on that category consumes the
    // whole budget, so an intersect placed after it would never run either.
    //
    // The pass is a pure ADDITION: it can only return "every objective is
    // certified", in which case the property is safe. Any refusal (gate off,
    // detector declined, unsupported op, cap exceeded, deadline, or the
    // self-policing precision gate) falls straight through to the code below
    // unchanged, so gate-off is byte-identical. See
    // `ny_propagate::dd_zonotope` for the soundness contract.
    if ny_propagate::dd_zonotope::dd_zonotope_enabled() {
        if let Some(result) = try_dd_zonotope_root(graph, input, vnnlib, &ledger, start_time, json)?
        {
            return Ok(result);
        }
    }

    // Multi-objective BaB for ≥2 constraints: conjunctive uses joint BaB (any_verified
    // semantics, strictly more powerful than per-constraint decomposition), disjunctive
    // uses all_verified semantics. Input splitting falls back to per-constraint for
    // disjunctive. See designs/2026-03-05-joint-conjunctive-bab.md.
    if total_constraint_count >= 2 && (use_relu_split || !is_disjunction) {
        let conjunctive = !is_disjunction;
        let (objectives, thresholds) = build_multi_objectives(vnnlib)?;

        if !json {
            let split_type = if use_relu_split {
                "ReLU splitting"
            } else {
                "input splitting"
            };
            let mode = if conjunctive {
                "conjunction (AND)"
            } else {
                "disjunction (OR)"
            };
            let safe_desc = if conjunctive {
                "ANY single constraint provably violated"
            } else {
                "ALL constraints provably violated"
            };
            println!("\nRunning multi-objective Graph β-CROWN ({split_type}) for {mode}...");
            println!(
                "Verifying {} constraints simultaneously (shared computation)",
                objectives.len()
            );
            println!("SAFE requires: {safe_desc}");
        }

        let multi_verifier = verifier.with_config_from(config.clone());
        let bab_deadline = ledger.bab_deadline();
        let result = if conjunctive {
            if use_relu_split {
                multi_verifier.verify_graph_relu_split_multi_objective_conjunctive_with_engine(
                    graph,
                    input,
                    &objectives,
                    &thresholds,
                    gemm_engine,
                    bab_deadline,
                )?
            } else {
                multi_verifier.verify_graph_input_split_multi_objective_conjunctive(
                    graph,
                    input,
                    &objectives,
                    &thresholds,
                    gemm_engine,
                    bab_deadline,
                )?
            }
        } else {
            multi_verifier.verify_graph_relu_split_multi_objective_with_engine(
                graph,
                input,
                &objectives,
                &thresholds,
                gemm_engine,
                bab_deadline,
            )?
        };

        if !json {
            println!("\n  Multi-objective result: {:?}", result.result);
            println!("    Domains explored: {}", result.domains_explored);
            println!("    Domains verified: {}", result.domains_verified);
            println!("    Max depth: {}", result.max_depth_reached);
            println!("    Time: {:.2}s", result.time_elapsed.as_secs_f64());
        }

        return Ok(result);
    }

    // Per-constraint verification (non-disjunction or input splitting)
    if !json {
        let split_type = if use_relu_split {
            "ReLU splitting"
        } else {
            "input splitting"
        };
        println!(
            "\nRunning Graph β-CROWN ({}) with constraints...",
            split_type
        );
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

    // Use remaining time from ledger rather than the original raw timeout,
    // so earlier phases (PGD, multi-objective BaB) reduce the per-constraint budget.
    let overall_timeout = ledger
        .remaining()
        .unwrap_or(std::time::Duration::from_secs(timeout));

    // Pre-compute α-CROWN bounds once for all constraints (major optimization)
    // This avoids re-computing bounds for each constraint, providing ~Nx speedup
    let precomputed_bounds = if use_relu_split && total_constraint_count > 1 {
        if !json {
            println!(
                "\n  Pre-computing α-CROWN bounds (shared across {} constraints)...",
                total_constraint_count
            );
        }
        let bounds_start = std::time::Instant::now();
        let bounds_verifier = verifier.with_config_from(config.clone());
        // Thread deadline from ledger so α-CROWN bails early if timeout budget
        // exhausted (#2698). Uses ledger.overall_deadline() instead of manually
        // computing overall_start + overall_timeout.
        let deadline = ledger.overall_deadline();
        let (node_bounds, output_bounds) =
            bounds_verifier.compute_initial_graph_bounds(graph, input, deadline)?;
        if !json {
            println!(
                "    Bounds computed in {:.2}s",
                bounds_start.elapsed().as_secs_f64()
            );
        }
        // Stash for a potential Graph-MIP escalation at BaB fall-through
        // (increment 5): the escalation reuses THIS per-property α-CROWN map
        // instead of recomputing it. The bounded stash is disabled when its
        // whole-network Graph-MIP consumer is explicitly off or this category
        // requests no MIP reservation.
        #[cfg(feature = "mip")]
        super::super::graph_mip_escalate::stash_graph_bounds_for_mip(
            graph,
            input,
            &config.phase_budget,
            &node_bounds,
        );
        Some((node_bounds, output_bounds))
    } else {
        None
    };

    let iter_config = ConstraintIterConfig {
        aggregation: classification.aggregation,
        overall_timeout,
        per_constraint_timeout,
        min_timeout_ms: 100,
        total_constraint_count,
        num_outputs: vnnlib.num_outputs,
        base_config: config.clone(),
        parent_verifier: Some(verifier),
        engine: verifier.engine_arc(),
        json,
    };

    // Cross-validation closure: evaluate original (un-augmented) graph network at a
    // concrete counterexample input. Part of #3209.
    let eval_original = |cx_input: &[f32]| -> Result<ArrayD<f32>> {
        let point = ArrayD::from_shape_vec(IxDyn(input.lower().shape()), cx_input.to_vec())?;
        evaluate_graph(graph, &point, gemm_engine)
    };

    let bab_deadline = ledger.bab_deadline();
    iterate_constraints(
        &vnnlib.output_constraints,
        &iter_config,
        |dispatch| {
            let bounds_ref = precomputed_bounds
                .as_ref()
                .map(|(nb, ob)| GraphPrecomputedBounds::new(nb, ob));
            dispatch_graph_constraint(
                dispatch.verifier,
                graph,
                input,
                dispatch.spec_coeffs,
                dispatch.threshold,
                use_relu_split,
                gpu_bab,
                bounds_ref.as_ref(),
                gemm_engine,
                bab_deadline,
            )
        },
        Some(&eval_original),
    )
}

/// Try to certify EVERY output constraint with the double-double zonotope
/// (`#dd-zonotope`). `Ok(None)` on any refusal — never a degraded verdict.
///
/// A `Verified` return here means the certified margin lower bound cleared the
/// threshold on every objective. Two independent enclosures are NOT combined:
/// this is the zonotope's own certificate, produced by a pass whose rounding
/// channel is computed rather than assumed and which refuses when that channel
/// is not far below the margin.
fn try_dd_zonotope_root(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    ledger: &PhaseBudgetLedger,
    start_time: std::time::Instant,
    json: bool,
) -> Result<Option<BetaCrownResult>> {
    use ny_propagate::dd_zonotope::{dd_zonotope_margins, DdZonoConfig, DdZonoPlan};

    let cfg = DdZonoConfig::from_env();
    let Some(plan) = DdZonoPlan::detect(graph, input, &cfg) else {
        if !json {
            println!("[dd-zonotope] detector declined; continuing with the standard pipeline");
        }
        return Ok(None);
    };
    let (objectives, thresholds) = build_multi_objectives(vnnlib)?;
    if objectives.is_empty() || objectives.len() != thresholds.len() {
        return Ok(None);
    }

    // Bounded slice of the remaining budget, so a refusal never costs the
    // whole timeout.
    let now = std::time::Instant::now();
    let cap = std::time::Duration::from_secs(
        std::env::var("NY_DD_ZONOTOPE_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(900),
    );
    let deadline = ledger.bab_deadline().map_or(now + cap, |d| {
        now + cap.min(d.saturating_duration_since(now).mul_f32(0.6))
    });
    if !json {
        println!(
            "[dd-zonotope] admitted: k={} input={} objectives={} budget={:.1}s",
            plan.k(),
            input.len(),
            objectives.len(),
            deadline.saturating_duration_since(now).as_secs_f32()
        );
    }

    let margin = match dd_zonotope_margins(graph, input, &objectives, &plan, &cfg, Some(deadline)) {
        Ok(Some(m)) => m,
        Ok(None) => {
            if !json {
                println!("[dd-zonotope] refused (fail-closed); bounds unchanged");
            }
            return Ok(None);
        }
        Err(e) => {
            if !json {
                println!("[dd-zonotope] error ({e}); bounds unchanged");
            }
            return Ok(None);
        }
    };
    if !json {
        for (i, ((&l, &u), (&hw, &rw))) in margin
            .lower
            .iter()
            .zip(margin.upper.iter())
            .zip(
                margin
                    .rounding_half_width
                    .iter()
                    .zip(margin.relax_half_width.iter()),
            )
            .enumerate()
        {
            println!(
                "[dd-zonotope] obj[{i}] certified=[{l:.9}, {u:.9}] relax_hw={rw:.4e} rounding_hw={hw:.4e} \
                 lower@{:.1}x_rounding={:.9} threshold={}",
                cfg.safety_factor,
                margin.lower_with_safety(i, cfg.safety_factor),
                thresholds[i]
            );
        }
        println!(
            "[dd-zonotope] gens={} wall={:.1}s",
            margin.n_generators,
            margin.wall.as_secs_f32()
        );
    }

    // SELF-POLICING PRECISION GATE: the certified ROUNDING half-width is
    // computed, not assumed. The ~2^66 amplification behind this method was
    // MEASURED on vgg16-7 only; a deeper or larger-weight network could exceed
    // double-double, and this turns that from an unsound assumption into an
    // ordinary decline.
    if !margin.precision_ok(cfg.precision_ratio) {
        if !json {
            println!(
                "[dd-zonotope] PRECISION GATE refused: the certified rounding half-width is \
                 not below {:.1e} x |margin|; bounds unchanged",
                cfg.precision_ratio
            );
        }
        return Ok(None);
    }

    // VERDICT with the explicit safety factor on the certified rounding
    // channel: the property must still hold when that channel is inflated by
    // `cfg.safety_factor` (default 2x). The `f64 -> f32` narrowing rounds
    // OUTWARD (`next_down_f32`) so a nearest-mode cast can never round a
    // certified lower bound UP across the threshold. `objectives.len() ==
    // thresholds.len()` was checked on entry, and `evaluate_objectives` emits
    // one margin per objective, so the three lengths agree here.
    let all_verified = margin.lower.len() == thresholds.len()
        && thresholds.iter().enumerate().all(|(i, &t)| {
            let lo = margin.lower_with_safety(i, cfg.safety_factor);
            lo.is_finite() && next_down_f32(lo as f32) > t
        });
    if !all_verified {
        if !json {
            println!("[dd-zonotope] certified margin does not clear every threshold; continuing");
        }
        return Ok(None);
    }

    let output_bounds = build_dd_output_bounds(&margin);
    if !json {
        println!(
            "[dd-zonotope] all {} objective(s) CERTIFIED at the root — property safe",
            objectives.len()
        );
    }
    Ok(Some(BetaCrownResult {
        result: BabVerificationStatus::Verified,
        domains_explored: 1,
        domains_verified: 1,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: start_time.elapsed(),
        output_bounds,
    }))
}

fn build_dd_output_bounds(
    margin: &ny_propagate::dd_zonotope::DdZonoMargin,
) -> Option<BoundedTensor> {
    if margin.output_lower.is_empty() || margin.output_shape.is_empty() {
        return None;
    }
    let shape = IxDyn(&margin.output_shape);
    let lower = ArrayD::from_shape_vec(shape.clone(), margin.output_lower.clone()).ok()?;
    let upper = ArrayD::from_shape_vec(shape, margin.output_upper.clone()).ok()?;
    BoundedTensor::new(lower, upper).ok()
}
