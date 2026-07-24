// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-clause disjunction verification.
//!
//! Handles VNN-LIB specs with `(assert (or ...))` that produce multiple
//! independent clauses, each verified separately with timeout budgeting.

use anyhow::Result;
use ny_core::GemmEngine;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, GraphNetwork, Layer,
};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};
use tracing::{debug, info};

use super::attack_budget::disjunctive_sampling_budget;
use super::disjunctive_pgd::{beta_crown_pgd_config, try_disjunctive_sampling_attack_with_config};
use super::disjunctive_precheck::{alpha_crown_precheck_clauses, crown_precheck_clauses};
use super::disjunctive_unified::{
    supports_grouped_disjunctive_contract, try_no_branchable_neuron_pgd_fallback,
    try_prechecked_unified_input_split_disjunctive_bab,
};
use super::phase_budget::PhaseBudgetLedger;
use super::BetaCrownModel;
use crate::commands::beta_crown::constraint_plan::build_grouped_disjunctive_objectives;
pub(super) fn finalize_disjunctive_result(
    aggregated: BetaCrownResult,
    overall_start: Instant,
    final_status: BabVerificationStatus,
) -> BetaCrownResult {
    let mut aggregated = super::normalize_result_wall_time(aggregated, overall_start);
    aggregated.result = final_status;
    aggregated
}
fn graph_has_conv2d_layers(model_net: &BetaCrownModel) -> bool {
    let BetaCrownModel::Graph(graph) = model_net else {
        return false;
    };

    graph.node_names().iter().any(|name| {
        graph.node(name).is_some_and(|node| {
            matches!(node.layer(), Layer::Conv2d(_) | Layer::ConvTranspose2d(_))
        })
    })
}

fn supports_graph_multi_objective_clause_batch(
    model_net: &BetaCrownModel,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    use_relu_split: bool,
    gpu_bab: bool,
) -> bool {
    graph_has_conv2d_layers(model_net)
        && use_relu_split
        && !gpu_bab
        && clauses.iter().all(|clause| clause.len() == 1)
        && per_clause_input_bounds
            .iter()
            .all(|bounds| bounds.is_empty())
}

/// Run the attack while optionally warming the forward-linear cache in a
/// scoped worker.  Keeping the optionality outside `thread::scope` makes the
/// short-budget contract testable: a refused warmer is never spawned, so there
/// is no worker for scope teardown to join.
pub(super) fn run_with_optional_forward_linear_warmer<W, A, R>(warmer: Option<W>, attack: A) -> R
where
    W: FnOnce() + Send,
    A: FnOnce() -> R,
{
    std::thread::scope(|scope| {
        if let Some(warmer) = warmer {
            scope.spawn(warmer);
        }
        attack()
    })
}

// Justification: Multi-clause disjunction requires the full verification context
// (model, bounds, spec, config, verifier, flags, engine) to check each clause.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_multi_clause_disjunction(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    use_relu_split: bool,
    gpu_bab: bool,
    pgd_attack: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<BetaCrownResult> {
    // LEVER 1: fresh IMB early-attempt flag per verify (scopes the in-lane suppression
    // to this instance; no-op today since ny runs one instance per process).
    ny_propagate::imb::reset_early_attempted();

    let clauses = if vnnlib.output_constraint_clauses.is_empty() {
        vec![vnnlib.output_constraints.clone()]
    } else {
        vnnlib.output_constraint_clauses.clone()
    };

    let num_clauses = clauses.len() as u64;

    let ledger = PhaseBudgetLedger::new(timeout, config.phase_budget.clone());
    let overall_timeout = Duration::from_secs(timeout);
    let overall_start = Instant::now();
    let overall_deadline = ledger.overall_deadline();
    let timeout_result = || BetaCrownResult {
        result: BabVerificationStatus::Timeout,
        domains_explored: 0,
        time_elapsed: overall_start.elapsed(),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };

    // #fit-100s (dark, `NY_SKIP_DISJ_PGD=1`, default OFF = byte-identical):
    // skip the disjunctive global PGD phase AND its #attack-extend retry
    // entirely — measured on cifar100_2024 prop7641 @100s the phase burned
    // 39.3s (30s PGD cap + 9.7s extension on a "promising" −0.022 margin)
    // finding nothing on a hold instance. The forward-linear reference map
    // warm (#w5-bab-throughput, ~22s, REQUIRED by the bootstrap + root margin
    // pass) still runs — synchronously, bounded by the overall deadline — so
    // only the dead falsification time is reclaimed for BaB. Measurement
    // lever for hold-heavy sweeps; competition keeps PGD (a missed sat is a
    // lost +10 — or worse, a wrong-verdict risk taken by BaB instead).
    let skip_disj_pgd = matches!(std::env::var("NY_SKIP_DISJ_PGD").ok().as_deref(), Some("1"));
    if pgd_attack && skip_disj_pgd {
        if let BetaCrownModel::Graph(g) = model_net {
            if graph_has_conv2d_layers(model_net)
                && GraphNetwork::forward_linear_reference_enabled()
            {
                let warm_start = Instant::now();
                match g.collect_forward_linear_bounds_dag_cached(
                    input,
                    gemm_engine,
                    overall_deadline,
                ) {
                    Ok(_) => info!(
                        elapsed_s = warm_start.elapsed().as_secs_f64(),
                        "Forward-linear reference map warmed; disjunctive PGD skipped (NY_SKIP_DISJ_PGD, #fit-100s)"
                    ),
                    Err(e) => debug!(
                        error = %e,
                        "Forward-linear warmer refused under NY_SKIP_DISJ_PGD (fail-closed; verify phase recomputes)"
                    ),
                }
            }
        }
    }

    // Global disjunctive PGD attack before per-clause BaB (#3218).
    // Reserve (1 - disjunctive_pgd_fraction) of timeout for CROWN precheck + BaB.
    // Source: PhaseBudgetConfig.disjunctive_pgd_fraction (default 0.50).
    if pgd_attack && !skip_disj_pgd {
        let (attack_restarts, attack_steps) = disjunctive_sampling_budget(pgd_restarts, pgd_steps);
        // Time-capped on tiny budgets so the attack cannot starve the CROWN
        // precheck + BaB phases (#four-walls; lsnc_relu burned 50% of a 20s
        // budget here before any bound computation ran), and clamped to the
        // optional absolute ceiling `disjunctive_pgd_max_secs` so a LARGE budget
        // cannot hand a hard-UNSAT (hold) falsifier a huge dead slice before BaB
        // (#reclaim-pgd; the fast-path BaB re-bases on `ledger.remaining()`, so
        // the reclaimed seconds flow straight to BaB).
        let pgd_deadline = ledger.disjunctive_pgd_deadline();
        // #w5-bab-throughput: warm the certified forward-linear reference map
        // CONCURRENTLY with the attack. On conv DAGs (cifar100-class), the
        // verify phase's bootstrap, spec-propagation setup, and root-margin
        // composition all consume this input-keyed map (~25s one-off on
        // cifar100), and it depends only on (graph, root input box) — both
        // known now. Computing it during the attack phase converts dead serial
        // time into BaB budget; `configured_graph_for_crown` carries the cache
        // into the verify clone. Sound: the warmer only populates a cache of
        // certified bounds through the exact machinery the verify phase would
        // run itself; on deadline refusal nothing is cached (status quo).
        let warm_graph: Option<&GraphNetwork> = match model_net {
            BetaCrownModel::Graph(g)
                if graph_has_conv2d_layers(model_net)
                    && GraphNetwork::forward_linear_reference_enabled() =>
            {
                Some(g)
            }
            _ => None,
        };
        const WARM_GRACE: Duration = Duration::from_secs(10);
        let warm_deadline = match (pgd_deadline, ledger.overall_deadline()) {
            (Some(p), Some(o)) => Some((p + WARM_GRACE).min(o)),
            (Some(p), None) => Some(p + WARM_GRACE),
            (None, o) => o,
        };
        let warm_graph = warm_graph.filter(|_| {
            let admitted = GraphNetwork::forward_linear_cold_build_admitted(warm_deadline);
            if !admitted {
                info!(
                    "Skipping optional forward-linear attack-phase warmer: insufficient cold-build headroom"
                );
            }
            admitted
        });
        let mut attack_feedback = super::disjunctive_pgd::DisjunctiveAttackFeedback::default();
        let warmer = warm_graph.map(|g| {
            move || {
                    let warm_start = Instant::now();
                    match g.collect_forward_linear_bounds_dag_cached(
                        input,
                        gemm_engine,
                        warm_deadline,
                    ) {
                        Ok(_) => info!(
                            elapsed_s = warm_start.elapsed().as_secs_f64(),
                            "Forward-linear reference map warmed during attack phase (#w5-bab-throughput)"
                        ),
                        Err(e) => debug!(
                            error = %e,
                            "Forward-linear attack-phase warmer refused (fail-closed; verify phase recomputes)"
                        ),
                    }
                }
        });
        let attack_outcome = run_with_optional_forward_linear_warmer(warmer, || {
            try_disjunctive_sampling_attack_with_config(
                model_net,
                input,
                &clauses,
                &vnnlib.per_clause_input_bounds,
                beta_crown_pgd_config(config, attack_restarts, attack_steps, pgd_deadline),
                gemm_engine,
                json,
                Some(&mut attack_feedback),
            )
        });
        match attack_outcome {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => {
                // #attack-extend: ONE bounded attack extension when the phase
                // ended without a counterexample but the closest-to-violation
                // margin was PROMISING (a hair under 0 ⇒ the ascent was cut by
                // the preset attack cap, and BaB can never prove a sat
                // instance). Hopeless margins hand off to BaB immediately, so
                // near-wall UNSAT proofs keep their slice. The retry CONTINUES
                // the restart seed sequence where the cap cut it (replaying the
                // cut restart from scratch), reproducing the measured longer
                // continuous run. Attack-only; NY_ATTACK_EXTEND=0 disables.
                if let Some(ext_slice) = super::attack_extension::attack_extension_slice(
                    ledger.remaining(),
                    &attack_feedback,
                    ledger.policy().attack_extension_fraction,
                ) {
                    let ext_deadline = {
                        let d = Instant::now() + ext_slice;
                        overall_deadline.map_or(d, |o| d.min(o))
                    };
                    let best_margin = attack_feedback.best_margin.unwrap_or(f32::NEG_INFINITY);
                    info!(
                        best_margin,
                        hit_deadline = attack_feedback.hit_deadline,
                        restarts_started = attack_feedback.restarts_started,
                        steps_taken = attack_feedback.steps_taken,
                        extension_s = ext_slice.as_secs_f64(),
                        "Attack extension granted: promising margin, one bounded retry (#attack-extend)"
                    );
                    if !json {
                        println!(
                            "  Attack extension: best margin {best_margin:.5} is promising (>= 0 is a counterexample) — one bounded retry (+{:.1}s, continuing the restart sequence)...",
                            ext_slice.as_secs_f64()
                        );
                    }
                    let ext_config = ny_propagate::PgdConfig {
                        // Continue the restart seed sequence at the restart the
                        // first run's cap cut (seed 42 + k ⇒ restart index k;
                        // see graph_pgd.rs restart_seed_offset).
                        seed:
                            42u64.wrapping_add(
                                attack_feedback.restarts_started.saturating_sub(1) as u64
                            ),
                        // The retry is deadline-bound; keep the preset's step
                        // depth but never let the restart count bind first.
                        num_restarts: attack_restarts.max(4096),
                        ..beta_crown_pgd_config(
                            config,
                            attack_restarts,
                            attack_steps,
                            Some(ext_deadline),
                        )
                    };
                    match try_disjunctive_sampling_attack_with_config(
                        model_net,
                        input,
                        &clauses,
                        &vnnlib.per_clause_input_bounds,
                        ext_config,
                        gemm_engine,
                        json,
                        None,
                    ) {
                        Ok(Some(result)) => return Ok(result),
                        Ok(None) => {
                            info!("Attack extension exhausted without a counterexample; handing off to BaB");
                        }
                        Err(e) => {
                            debug!(error = %e, "Attack extension failed, continuing to BaB");
                        }
                    }
                } else if attack_feedback.best_margin.is_some() {
                    debug!(
                        best_margin = attack_feedback.best_margin.unwrap_or(f32::NEG_INFINITY),
                        "No attack extension: margin not promising or budget too small (#attack-extend)"
                    );
                }
            }
            Err(e) => {
                debug!(error = %e, "Disjunctive sampling attack failed, continuing");
            }
        }
    }

    if timeout > 0 && overall_start.elapsed() >= overall_timeout {
        return Ok(timeout_result());
    }

    // Per-disjunct alpha fast-path (#4355).
    if config.optimize_disjuncts_separately && use_relu_split {
        return super::disjunctive_per_disjunct::try_per_disjunct_multi_objective(
            model_net,
            config,
            verifier,
            input,
            vnnlib,
            &clauses,
            gemm_engine,
            pgd_attack,
            pgd_restarts,
            pgd_steps,
            json,
            timeout,
            overall_start,
            &ledger,
        );
    }

    // #3813: Conv2d-heavy graph fast-path — skip precheck, route to multi-
    // objective BaB directly (precheck is redundant, BaB recomputes anyway).
    if supports_graph_multi_objective_clause_batch(
        model_net,
        &clauses,
        &vnnlib.per_clause_input_bounds,
        use_relu_split,
        gpu_bab,
    ) {
        let remaining_timeout = ledger.remaining().unwrap_or(Duration::from_secs(u64::MAX));
        if timeout > 0 && remaining_timeout.is_zero() {
            return Ok(timeout_result());
        }

        let remaining_timeout_secs = ledger.remaining_secs_clamped();
        let remaining_timeout_secs = if timeout == 0 {
            0
        } else {
            remaining_timeout_secs.max(1)
        };
        let remaining_config = BetaCrownConfig {
            timeout: if timeout == 0 {
                config.timeout
            } else {
                remaining_timeout
            },
            ..config.clone()
        };
        let remaining_verifier = verifier.with_config_from(remaining_config.clone());

        debug!(
            clauses = clauses.len(),
            timeout_s = remaining_timeout_secs,
            "Graph multi-objective fast-path: skipping precheck, routing all clauses to BaB"
        );

        let mut result = super::verify_relational_constraints_impl(
            model_net,
            input,
            vnnlib,
            &remaining_config,
            &remaining_verifier,
            use_relu_split,
            gpu_bab,
            false, // global disjunctive PGD already ran above
            pgd_restarts,
            pgd_steps,
            remaining_timeout_secs,
            gemm_engine,
            json,
        )?;
        result.time_elapsed = overall_start.elapsed();

        // Fallback PGD for no-branchable-neuron BaB results (#3769).
        if let Some(sat) = try_no_branchable_neuron_pgd_fallback(
            &result,
            pgd_attack,
            model_net,
            input,
            &clauses,
            &vnnlib.per_clause_input_bounds,
            config,
            pgd_restarts,
            pgd_steps,
            gemm_engine,
            json,
            &ledger,
        ) {
            return Ok(sat);
        }

        result.time_elapsed = overall_start.elapsed();
        return Ok(result);
    }

    // LEVER 1 — IMB EARLY FAST-PATH, AHEAD of the ~365 s CROWN-IBP per-output precheck.
    // The IMB certificate neither needs nor consumes that precheck, so running it here
    // — AFTER the PGD attack (so SAT instances are still falsified first) but BEFORE the
    // precheck — hands it the full remaining budget instead of the ~115 s that were left
    // downstream. On a clear ⇒ return `unsat` immediately; on a miss/disarmed/error ⇒
    // `None` and the UNCHANGED precheck + BaB run exactly as today (the downstream
    // in-lane `#imb-early` block is suppressed via `imb::early_attempted`, so the IMB
    // never repeats). Gated `NY_IMB=1 && NY_IMB_WIRE=1`; a no-op for every other model.
    if let BetaCrownModel::Graph(g) = model_net {
        if supports_grouped_disjunctive_contract(vnnlib, use_relu_split) {
            if let Ok(plan) = build_grouped_disjunctive_objectives(vnnlib) {
                if let Some(res) = verifier.try_imb_early_disjunctive(
                    g,
                    input,
                    &plan.objectives,
                    &plan.thresholds,
                    &plan.clause_sizes,
                    gemm_engine,
                    overall_deadline,
                ) {
                    return Ok(finalize_disjunctive_result(
                        res,
                        overall_start,
                        BabVerificationStatus::Verified,
                    ));
                }
            }
        }
    }

    // CROWN pre-check: screen clauses with a single pass before per-clause BaB.
    // #3813: Cap precheck budget to disjunctive_precheck_fraction of total timeout.
    // The slice is computed from NOW (not the ledger start): with the start
    // base, any attack phase longer than the precheck fraction left this
    // deadline already expired, so the root CROWN pass silently degraded to
    // IBP-only bounds (#four-walls; `phase_deadline_from_now` is capped at the
    // overall deadline, so this only reclaims the slice the precheck was
    // always meant to have).
    // Per-clause-box disjunctions (nn4sys mscn/lindex bands) get a LARGER
    // slice: for that shape the precheck is the batched box-refinement engine
    // — the only lane measured to make progress (999/1000 clauses on mscn
    // cardinality_0_500 in <1s), while the downstream serial per-clause BaB
    // was measured to burn its whole slice without converging (shared-root
    // intermediate bounds; 100k domains/87s, zero clauses closed). The
    // refinement returns as soon as every clause is decided, so the bigger
    // slice only spends time that the serial loop would otherwise waste;
    // the remaining fraction still reaches the loop for any stragglers.
    // 0.95 (was 0.75, then 0.9): the mscn `_dual` band instances are decided
    // ONLY by the box-refinement screen's f64 escalation — measured on
    // cardinality_1_240_128_dual, the downstream serial per-clause BaB closes
    // ZERO clauses (0.9s/clause of f32 input-split BaB against ±1e-5 margins
    // below the f32 rounding floor), so every second reserved for it after an
    // unfinished screen is wasted. The screen still returns the moment all
    // clauses are decided, so fast instances (lindex, non-dual mscn, <1s) are
    // unaffected, and the small reserve still reaches the downstream lanes
    // for shapes the screen cannot close.
    // The bump is GRAPH-ONLY and 0.95: it funds the batched box-refinement
    // screen, the only lane measured to progress on the nn4sys Mul-heavy DAGs
    // (per-clause BaB closes zero clauses there), and with the batched
    // multi-box screen (#f64-batch-boxes) cardinality_1_1_2048_dual's full
    // refinement tree costs ~12.8s at its 20s official budget — a 0.9 slice
    // finished the tree at the deadline's edge (verified=0 by <0.5s) while
    // the reserve went unused (per-clause BaB closes zero, MIP ineligible for
    // graph models). On SEQUENTIAL models with per-clause boxes (acasxu
    // prop_6-class input-box disjunctions) the screen is one cheap CROWN pass
    // per clause and the real solver is the per-clause input-split BaB loop
    // below — a bumped slice starved it, so sequential keeps the policy value.
    // Saturation-Escape Branching engagement (M1; gate `NY_SAT_ESCAPE_BRANCH=1`,
    // default OFF ⇒ this whole block is byte-identical). On the mscn `_dual`
    // per-clause-box graphs the box-refinement screen structurally CANNOT close
    // the near-closing multi-dim boxes (f64 enclosure below the f32 margin
    // floor), so the 0.95 precheck fraction burns the whole budget on a doomed
    // screen and the serial per-clause input-split BaB loop below is reached with
    // ~0 budget — it returns `unknown` WITHOUT ever branching the inconclusive
    // disjuncts (the measured engagement gap). With the gate on, cap the screen
    // at the default fraction so it still discharges the easy clauses but RESERVES
    // budget for the per-clause BaB loop, where the SEB scorer targets the
    // de-saturating dims. Soundness is unchanged: the serial loop still proves
    // HOLD only when every inconclusive disjunct is refuted on its exact box
    // partition (`multi_dim_split_boxes` is a complete cover), identical to today.
    // Matches `sat_escape::enabled()` in ny-propagate; read directly here to
    // avoid widening that crate-internal module's visibility across crates.
    let seb = matches!(
        std::env::var("NY_SAT_ESCAPE_BRANCH").ok().as_deref(),
        Some("1")
    );
    let has_per_clause_boxes = vnnlib.per_clause_input_bounds.iter().any(|b| !b.is_empty());
    let precheck_fraction = if has_per_clause_boxes && matches!(model_net, BetaCrownModel::Graph(_))
    {
        if seb {
            ledger.policy().disjunctive_precheck_fraction.min(0.5)
        } else {
            ledger.policy().disjunctive_precheck_fraction.max(0.95)
        }
    } else {
        ledger.policy().disjunctive_precheck_fraction
    };
    let precheck_deadline = {
        let phase = ledger.phase_deadline_from_now(precheck_fraction);
        match (phase, overall_deadline) {
            (Some(p), Some(o)) => Some(p.min(o)),
            (p, o) => p.or(o),
        }
    };
    let t_crown = Instant::now();
    let pre_verified = crown_precheck_clauses(
        model_net,
        input,
        &clauses,
        &vnnlib.per_clause_input_bounds,
        gemm_engine,
        precheck_deadline,
    );
    let pre_verified_count = pre_verified.iter().filter(|&&v| v).count();
    info!(
        verified = pre_verified_count,
        total = clauses.len(),
        elapsed_s = t_crown.elapsed().as_secs_f64(),
        "CROWN precheck phase complete"
    );

    if timeout > 0 && overall_start.elapsed() >= overall_timeout {
        return Ok(timeout_result());
    }

    if pre_verified_count > 0 {
        debug!(
            pre_verified = pre_verified_count,
            total = num_clauses,
            elapsed_s = overall_start.elapsed().as_secs_f64(),
            "CROWN pre-check verified clauses"
        );
    }

    // Fast path: if all clauses verified by CROWN pre-check, skip BaB entirely.
    if pre_verified_count == clauses.len() {
        return Ok(BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: 0,
            time_elapsed: overall_start.elapsed(),
            max_depth_reached: 0,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        });
    }

    // Alpha-CROWN pre-check (#3813). Gated on use_alpha_crown (#4258).
    // Slice computed from NOW for the same wrong-base reason as the CROWN
    // precheck above (#four-walls).
    let pre_verified = if config.use_alpha_crown {
        let deadline = {
            let phase =
                ledger.phase_deadline_from_now(ledger.policy().disjunctive_precheck_fraction);
            match (phase, overall_deadline) {
                (Some(p), Some(o)) => Some(p.min(o)),
                (p, o) => p.or(o),
            }
        };
        let t0 = Instant::now();
        let result = alpha_crown_precheck_clauses(
            model_net,
            input,
            &clauses,
            &pre_verified,
            &vnnlib.per_clause_input_bounds,
            gemm_engine,
            deadline,
        );
        info!(
            verified = result.iter().filter(|&&v| v).count(),
            total = clauses.len(),
            elapsed_s = t0.elapsed().as_secs_f64(),
            total_elapsed_s = overall_start.elapsed().as_secs_f64(),
            "Alpha-CROWN precheck phase complete"
        );
        result
    } else {
        debug!("Skipping alpha-CROWN precheck: use_alpha_crown=false");
        pre_verified
    };
    let pre_verified_count = pre_verified.iter().filter(|&&v| v).count();

    if timeout > 0 && overall_start.elapsed() >= overall_timeout {
        return Ok(timeout_result());
    }

    if pre_verified_count == clauses.len() {
        debug!(
            elapsed_s = overall_start.elapsed().as_secs_f64(),
            "Alpha-CROWN pre-check verified ALL clauses — skipping BaB"
        );
        return Ok(BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: 0,
            time_elapsed: overall_start.elapsed(),
            max_depth_reached: 0,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        });
    }

    // Box-grouped decomposition for input-box disjunctions (acasxu prop_6-class:
    // `(or <box_1> <box_2>) AND (or <out_1> ... <out_m>)` normalizes to K·m
    // clauses over K distinct boxes). Solving per CLAUSE re-pays the BaB tree m
    // times per box (measured on ACASXU_run2a_1_1/prop_6: 58s per clause — 8
    // clauses can never fit 111s), while ONE grouped multi-clause tree over a
    // box amortizes all m objectives (measured: 58s for all 4 — the same cost
    // as one). Group the unresolved clauses by bit-identical box and solve each
    // box as an independent sub-problem via the grouped unified lane.
    // Aggregation is EXACT: UNSAT iff every box verifies; SAT if any box yields
    // a confirmed counterexample (a witness for `box_k ∧ clause` is a witness
    // for the full property); anything else stays unknown/timeout.
    if let Some(result) = try_box_grouped_disjunctive(
        model_net,
        input,
        vnnlib,
        &pre_verified,
        &clauses,
        config,
        verifier,
        use_relu_split,
        gpu_bab,
        pgd_restarts,
        pgd_steps,
        timeout,
        gemm_engine,
        json,
        &ledger,
        overall_start,
    )? {
        return Ok(result);
    }

    // #3740 Packet B: unified input-split disjunctive BaB.
    if let Some(result) = try_prechecked_unified_input_split_disjunctive_bab(
        model_net,
        input,
        vnnlib,
        &pre_verified,
        config,
        verifier,
        &clauses,
        use_relu_split,
        gpu_bab,
        pgd_attack,
        pgd_restarts,
        pgd_steps,
        timeout,
        gemm_engine,
        json,
        &ledger,
        overall_start,
    )? {
        return Ok(result);
    }

    let mut aggregated = BetaCrownResult {
        result: BabVerificationStatus::Verified,
        domains_explored: 0,
        time_elapsed: Duration::from_secs(0),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };
    let mut saw_potential = false;
    let mut unknown_reason: Option<String> = None;

    for (idx, clause) in clauses.iter().enumerate() {
        // Skip clauses already verified by CROWN pre-check.
        if pre_verified[idx] {
            continue;
        }

        if timeout > 0 && overall_start.elapsed() >= overall_timeout {
            unknown_reason
                .get_or_insert_with(|| format!("Timeout before clause {} verification", idx + 1));
            break;
        }

        // Adaptive timeout: remaining time / remaining BaB clauses.
        // Easy clauses (verified by CROWN pre-check) are excluded from the
        // count so hard clauses get the full remaining budget.
        // Minimum 1s per clause to avoid integer truncation when remaining
        // seconds < clause count (e.g., CIFAR-100: 99 clauses, 7s remaining
        // → 7/99=0 in integer division → instant "timeout before clause 1").
        let clause_timeout = if timeout == 0 {
            0
        } else {
            let remaining = ledger.remaining().unwrap_or(Duration::ZERO);
            if remaining.is_zero() {
                unknown_reason.get_or_insert_with(|| {
                    format!("Timeout before clause {} verification", idx + 1)
                });
                break;
            }
            // Count remaining unverified clauses from this index onward.
            let remaining_bab = pre_verified[idx..].iter().filter(|&&v| !v).count().max(1) as u64;
            // Use ceiling division to ensure at least 1s per clause when
            // there is any remaining time.
            let remaining_secs = remaining.as_secs().max(1);
            let adaptive = remaining_secs / remaining_bab;
            adaptive.max(1)
        };

        debug!(
            clause = idx + 1,
            total = num_clauses,
            timeout_s = clause_timeout,
            elapsed_s = overall_start.elapsed().as_secs(),
            "Disjunctive clause verification"
        );

        let mut clause_spec = vnnlib.clone();
        clause_spec.output_constraints = clause.clone();
        clause_spec.output_constraint_clauses = Vec::new();
        clause_spec.is_disjunction = false;
        clause_spec.per_clause_input_bounds = Vec::new();

        // Apply per-clause input bounds if available (e.g., nn4sys lindex).
        // Tighten the input BoundedTensor to this clause's specific domain.
        let clause_input;
        let effective_input = if let Some(clause_bounds) = vnnlib.per_clause_input_bounds.get(idx) {
            if !clause_bounds.is_empty() {
                // Tighten spec's input bounds for this clause
                for (&var_idx, &(lower, upper)) in clause_bounds {
                    if var_idx < clause_spec.input_bounds.len() {
                        clause_spec.input_bounds[var_idx] = (lower, upper);
                    }
                }
                // Create tightened BoundedTensor for this clause's input domain
                let flat_len = input.lower().len();
                let mut new_lower = input.lower().clone();
                let mut new_upper = input.upper().clone();
                if let (Some(lo_slice), Some(up_slice)) =
                    (new_lower.as_slice_mut(), new_upper.as_slice_mut())
                {
                    for (&var_idx, &(lb, ub)) in clause_bounds {
                        if var_idx < flat_len {
                            // Directed f64→f32 rounding: only widen when the
                            // conversion is inexact (see f64_to_f32_floor —
                            // unconditional ULP widening pushed tight nn4sys
                            // band margins below the provable floor).
                            let lb_f32 = super::disjunctive_precheck::f64_to_f32_floor(lb);
                            let ub_f32 = super::disjunctive_precheck::f64_to_f32_ceil(ub);
                            lo_slice[var_idx] = lo_slice[var_idx].max(lb_f32);
                            up_slice[var_idx] = up_slice[var_idx].min(ub_f32);
                        }
                    }
                }
                clause_input =
                    BoundedTensor::new(new_lower, new_upper).unwrap_or_else(|_| input.clone());
                &clause_input
            } else {
                input
            }
        } else {
            input
        };

        // Disable per-clause PGD attacks: each clause's verify_sequential_relational
        // runs sampling+SPSA+PGD+reduced-PGD, taking ~7s per clause on MNIST FC models.
        // With N clauses (typically 9 for MNIST 10-class), PGD alone exceeds the timeout
        // budget (9 × 7s = 63s vs 30s timeout). Per-clause PGD is also redundant — each
        // clause checks "can class k beat the true class" individually, but a single global
        // PGD could check all classes at once. Disabling per-clause PGD lets each clause
        // use its full adaptive timeout budget for CROWN + BaB verification.
        let clause_result = super::verify_relational_constraints_impl(
            model_net,
            effective_input,
            &clause_spec,
            config,
            verifier,
            use_relu_split,
            gpu_bab,
            false, // pgd_attack disabled — per-clause PGD is too expensive
            pgd_restarts,
            pgd_steps,
            clause_timeout,
            gemm_engine,
            json,
        )?;

        aggregated.domains_explored += clause_result.domains_explored;
        aggregated.domains_verified += clause_result.domains_verified;
        aggregated.cuts_generated += clause_result.cuts_generated;
        aggregated.time_elapsed += clause_result.time_elapsed;
        aggregated.max_depth_reached = aggregated
            .max_depth_reached
            .max(clause_result.max_depth_reached);

        match clause_result.result {
            BabVerificationStatus::Verified => {}
            violation @ BabVerificationStatus::Violated { .. } => {
                return Ok(finalize_disjunctive_result(
                    aggregated,
                    overall_start,
                    violation,
                ));
            }
            BabVerificationStatus::PotentialViolation => {
                saw_potential = true;
            }
            BabVerificationStatus::Unknown { reason } => {
                if unknown_reason.is_none() {
                    unknown_reason = Some(format!("Clause {}: {}", idx + 1, reason));
                }
            }
            BabVerificationStatus::Timeout => {
                if unknown_reason.is_none() {
                    unknown_reason = Some(format!("Clause {}: timeout", idx + 1));
                }
            }
        }
    }

    let final_status = if saw_potential {
        BabVerificationStatus::PotentialViolation
    } else if let Some(reason) = unknown_reason {
        BabVerificationStatus::Unknown { reason }
    } else {
        BabVerificationStatus::Verified
    };

    Ok(finalize_disjunctive_result(
        aggregated,
        overall_start,
        final_status,
    ))
}

/// Maximum number of DISTINCT per-clause input boxes the box-grouped
/// decomposition will take on. Input-box disjunctions (acasxu prop_6-class)
/// have a handful of boxes; the nn4sys lindex/mscn band families have
/// hundreds-to-thousands of distinct boxes and are served by the batched
/// box-refinement screen instead — this cap keeps them there.
const MAX_BOX_GROUPS: usize = 8;

/// Bit-exact grouping key for a per-clause input box (f64 bounds have no Eq;
/// bit-identical grouping mirrors the box-refinement engine's box grouping).
fn clause_box_key(map: &std::collections::BTreeMap<usize, (f64, f64)>) -> Vec<(usize, u64, u64)> {
    map.iter()
        .map(|(&idx, &(lo, hi))| (idx, lo.to_bits(), hi.to_bits()))
        .collect()
}

/// Box-grouped decomposition of a per-clause-box disjunction: solve each
/// DISTINCT input box as one independent multi-clause disjunctive sub-problem
/// (its clauses over its box, boxes cleared), recursing into
/// [`verify_multi_clause_disjunction`] so the sub-problem routes through the
/// grouped unified input-split lane (one shared BaB tree amortizing all of the
/// box's objectives). Recursion terminates because the sub-spec has NO
/// per-clause boxes (this gate requires one).
///
/// Returns `Ok(None)` (decline — caller falls through to the legacy serial
/// per-clause loop) unless the spec is a disjunction whose clauses group into
/// `1..=MAX_BOX_GROUPS` distinct boxes.
///
/// SOUNDNESS / aggregation exactness: the property's unsafe region is
/// `∃k: x ∈ box_k ∧ clause_k(y)`. Grouping preserves it exactly — the group of
/// box B carries ALL clauses whose box is B, each sub-problem verifies its
/// disjunction over B tightened with directed rounding (`tighten_input_to_box`
/// encloses the real clause domain), and the verdict is Verified only when
/// EVERY group (and every pre-verified clause) is discharged. A group
/// counterexample satisfies `box_k ∧ clause_k` and therefore the full
/// property; it is still re-confirmed downstream (confirm_potential_violation
/// + the trusted-ORT vnncomp gate) before any `sat` is scored.
// Justification: forwards the full verification context of the caller plus the
// precheck results and budget ledger.
#[allow(clippy::too_many_arguments)]
fn try_box_grouped_disjunctive(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    pre_verified: &[bool],
    clauses: &[Vec<OutputConstraint>],
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    use_relu_split: bool,
    gpu_bab: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    ledger: &PhaseBudgetLedger,
    overall_start: Instant,
) -> Result<Option<BetaCrownResult>> {
    if !vnnlib.is_disjunction
        || vnnlib.per_clause_input_bounds.len() != clauses.len()
        || pre_verified.len() != clauses.len()
        || !vnnlib
            .per_clause_input_bounds
            .iter()
            .any(|bounds| !bounds.is_empty())
    {
        return Ok(None);
    }

    // Group UNRESOLVED clause indices by bit-identical box (order-preserving).
    // A clause with an EMPTY box map ranges over the global input box; it forms
    // its own (untightened) group, keeping the union semantics exact.
    let mut groups: Vec<(Vec<(usize, u64, u64)>, Vec<usize>)> = Vec::new();
    let mut unresolved = 0usize;
    for (idx, bounds) in vnnlib.per_clause_input_bounds.iter().enumerate() {
        if pre_verified[idx] {
            continue;
        }
        unresolved += 1;
        let key = clause_box_key(bounds);
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, indices)) => indices.push(idx),
            None => groups.push((key, vec![idx])),
        }
    }
    // Fire only when grouping actually amortizes (strictly more clauses than
    // boxes). One-clause-per-box shapes (nn4sys mscn `_dual` singletons) keep
    // their measured route: box-refinement screen + serial per-clause loop.
    if groups.is_empty() || groups.len() > MAX_BOX_GROUPS || unresolved <= groups.len() {
        return Ok(None);
    }

    info!(
        groups = groups.len(),
        clauses = clauses.len(),
        pre_verified = pre_verified.iter().filter(|&&v| v).count(),
        "Box-grouped disjunctive decomposition: one grouped sub-problem per distinct clause box"
    );

    let mut aggregated = BetaCrownResult {
        result: BabVerificationStatus::Verified,
        domains_explored: 0,
        time_elapsed: Duration::from_secs(0),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };
    let mut saw_potential = false;
    let mut unknown_reason: Option<String> = None;

    // DFS work stack of RE-ROOTED sub-problems: (box map, clause indices,
    // re-root depth). When a node's grouped BaB comes back undecided
    // (timeout / domain cap), its box is bisected on the widest effective axis
    // and both halves are re-solved with FRESH ROOTS. Measured on
    // ACASXU_run2a_1_1/prop_6 box 2: the engine's own tree reuses ROOT
    // intermediate bounds (#3453-class), so 100k in-tree domains fail to close
    // what a fresh-rooted half-box discharges in ~1.8k domains. Bisection
    // children exactly cover the parent (`[lo, mid] ∪ [mid, hi]`), so the
    // partition preserves the unsafe-region union and Verified still means
    // EVERY leaf of every group's box partition is discharged.
    let mut stack: Vec<(
        std::collections::BTreeMap<usize, (f64, f64)>,
        Vec<usize>,
        u16,
    )> = groups
        .iter()
        .rev()
        .map(|(_, clause_indices)| {
            (
                vnnlib.per_clause_input_bounds[clause_indices[0]].clone(),
                clause_indices.clone(),
                0u16,
            )
        })
        .collect();
    let mut nodes_processed = 0usize;

    while let Some((clause_bounds, clause_indices, depth)) = stack.pop() {
        let remaining = ledger.remaining().unwrap_or(Duration::from_secs(u64::MAX));
        if timeout > 0 && remaining.is_zero() {
            unknown_reason
                .get_or_insert_with(|| "Timeout before box sub-problem verification".to_string());
            break;
        }
        nodes_processed += 1;
        // Adaptive budget split: remaining time / open sub-problems (>= 1s
        // each, mirroring the serial per-clause loop's slice arithmetic).
        let group_timeout = if timeout == 0 {
            0
        } else {
            let open = (stack.len() + 1) as u64;
            (remaining.as_secs().max(1) / open).max(1)
        };

        // Sub-spec: this box's clauses over the box, per-clause boxes CLEARED
        // (every clause in the group shares the box, now enforced by the
        // tightened input tensor), disjunctive aggregation preserved.
        let mut sub_spec = vnnlib.clone();
        sub_spec.output_constraint_clauses = clause_indices
            .iter()
            .map(|&idx| clauses[idx].clone())
            .collect();
        sub_spec.output_constraints = sub_spec
            .output_constraint_clauses
            .iter()
            .flatten()
            .cloned()
            .collect();
        sub_spec.is_disjunction = true;
        sub_spec.per_clause_input_bounds = Vec::new();

        let group_input = if clause_bounds.is_empty() {
            input.clone()
        } else {
            // Mirror the box restriction into the sub-spec's own input bounds
            // (attack-box construction and witness gating read them).
            for (&var_idx, &(lb, ub)) in &clause_bounds {
                if var_idx < sub_spec.input_bounds.len() {
                    let global = sub_spec.input_bounds[var_idx];
                    sub_spec.input_bounds[var_idx] = (global.0.max(lb), global.1.min(ub));
                }
            }
            super::disjunctive_precheck::tighten_input_to_box(input, &clause_bounds)
        };

        // Fail-fast domain cap while re-rooting is still available: an
        // undecided verdict at REROOT_DOMAIN_CAP arrives in seconds and the
        // fresh-rooted halves converge orders of magnitude faster than more
        // in-tree domains would (root-bound staleness). The final depth gets
        // the caller's full cap as a last chance. SOUND: a smaller domain cap
        // can only turn would-be-verified into unknown, never the reverse.
        let can_reroot =
            depth < MAX_REROOT_DEPTH && nodes_processed + stack.len() < MAX_REROOT_NODES;
        let node_config = if can_reroot && config.max_domains > REROOT_DOMAIN_CAP {
            let mut c = config.clone();
            c.max_domains = REROOT_DOMAIN_CAP;
            std::borrow::Cow::Owned(c)
        } else {
            std::borrow::Cow::Borrowed(config)
        };

        debug!(
            depth,
            open = stack.len() + 1,
            group_clauses = clause_indices.len(),
            timeout_s = group_timeout,
            elapsed_s = overall_start.elapsed().as_secs_f64(),
            "Box-grouped disjunctive sub-problem"
        );

        // Recurse: the sub-spec has no per-clause boxes, so this call routes to
        // the grouped unified input-split lane (or the serial per-clause loop
        // for shapes the unified gate declines). The global disjunctive PGD
        // already ran over the full hull, so per-group attack stays off.
        let group_result = verify_multi_clause_disjunction(
            model_net,
            &group_input,
            &sub_spec,
            &node_config,
            verifier,
            use_relu_split,
            gpu_bab,
            false, // pgd_attack: the global attack phase already ran
            pgd_restarts,
            pgd_steps,
            group_timeout,
            gemm_engine,
            json,
        )?;

        aggregated.domains_explored += group_result.domains_explored;
        aggregated.domains_verified += group_result.domains_verified;
        aggregated.cuts_generated += group_result.cuts_generated;
        aggregated.max_depth_reached = aggregated
            .max_depth_reached
            .max(group_result.max_depth_reached);

        match group_result.result {
            BabVerificationStatus::Verified => {}
            violation @ BabVerificationStatus::Violated { .. } => {
                return Ok(Some(finalize_disjunctive_result(
                    aggregated,
                    overall_start,
                    violation,
                )));
            }
            BabVerificationStatus::PotentialViolation => {
                saw_potential = true;
            }
            BabVerificationStatus::Unknown { .. } | BabVerificationStatus::Timeout => {
                let reason = match &group_result.result {
                    BabVerificationStatus::Unknown { reason } => reason.clone(),
                    _ => "timeout".to_string(),
                };
                // Re-root: bisect this box on its widest effective axis and
                // re-solve both halves with fresh roots (budget permitting).
                let bisected = can_reroot
                    && (timeout == 0 || remaining.as_secs() >= 2)
                    && match bisect_effective_box(&group_input, &clause_bounds) {
                        Some((left, right)) => {
                            stack.push((right, clause_indices.clone(), depth + 1));
                            stack.push((left, clause_indices.clone(), depth + 1));
                            true
                        }
                        None => false,
                    };
                if !bisected {
                    unknown_reason.get_or_insert_with(|| {
                        format!("Box sub-problem (depth {}): {}", depth, reason)
                    });
                }
            }
        }
    }

    let final_status = if saw_potential {
        BabVerificationStatus::PotentialViolation
    } else if let Some(reason) = unknown_reason {
        BabVerificationStatus::Unknown { reason }
    } else {
        BabVerificationStatus::Verified
    };

    Ok(Some(finalize_disjunctive_result(
        aggregated,
        overall_start,
        final_status,
    )))
}

/// Maximum re-root bisection depth per box group (leaf width = axis width /
/// 2^16 in the worst case; the measured acasxu prop_6 hard slivers verify
/// within a handful of levels).
const MAX_REROOT_DEPTH: u16 = 16;

/// Hard cap on re-rooted sub-problems per property (bounds the fixed
/// per-sub-problem root-bound cost).
const MAX_REROOT_NODES: usize = 64;

/// Fail-fast domain cap for re-root-eligible sub-problems: verified boxes on
/// the measured acasxu prop_6 profile need <= ~5k domains; a box still open at
/// this many in-tree domains converges faster after a fresh-rooted bisection
/// than by exploring further against stale root intermediate bounds.
const REROOT_DOMAIN_CAP: usize = 25_000;

/// Bisect the EFFECTIVE box of a re-rooted sub-problem on its widest axis.
///
/// The effective box is the tightened input tensor (clause box ∩ global box);
/// the returned halves are expressed as clause-box maps whose split axis is
/// overridden with `[lo, mid]` / `[mid, hi]` (f64 midpoint of the effective
/// f32 bounds). The halves exactly cover the parent (mid shared — closed
/// intervals over-cover, which is sound). Returns `None` when no axis is
/// splittable (degenerate box), so the caller keeps the parent's undecided
/// verdict instead of looping.
fn bisect_effective_box(
    group_input: &BoundedTensor,
    clause_bounds: &std::collections::BTreeMap<usize, (f64, f64)>,
) -> Option<(
    std::collections::BTreeMap<usize, (f64, f64)>,
    std::collections::BTreeMap<usize, (f64, f64)>,
)> {
    let lower = group_input.lower();
    let upper = group_input.upper();
    let (lo, hi) = (lower.as_slice()?, upper.as_slice()?);
    let mut best: Option<(usize, f64, f64, f64)> = None;
    for (idx, (&l, &u)) in lo.iter().zip(hi.iter()).enumerate() {
        let (l, u) = (f64::from(l), f64::from(u));
        let width = u - l;
        if width.is_finite() && best.as_ref().is_none_or(|(_, _, _, w)| width > *w) {
            best = Some((idx, l, u, width));
        }
    }
    let (axis, l, u, _) = best?;
    let mid = l + (u - l) / 2.0;
    if !(mid > l && mid < u) {
        return None; // degenerate: no representable split point
    }
    let mut left = clause_bounds.clone();
    let mut right = clause_bounds.clone();
    left.insert(axis, (l, mid));
    right.insert(axis, (mid, u));
    Some((left, right))
}
