// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched exact-VJP graph **disjunctive** PGD (#attack-extend increment 2):
//! K parallel restarts per step from ONE wide GPU pass, for multi-clause
//! `(or ...)` properties.
//!
//! The sequential disjunctive loop in `graph_pgd.rs` pays one certified
//! forward + one backward per restart per step (~0.3s/step on metaroom's
//! 6cnn), so a preset-capped ~20s attack slice covers barely 2 restarts of a
//! 30-restart budget — measured on metaroom_2023 spec_idx_129/148 (GT=sat),
//! the attack is cut mid-exploration and the instance burns its remaining
//! budget in BaB, which can never prove a sat instance. This driver runs the
//! SAME best-disjunct margin ascent for K restarts simultaneously: per step,
//! one batched CPU template forward captures every restart's ReLU masks +
//! outputs, then ONE wide GPU pass returns all K exact margin gradients (each
//! restart's spec row is its OWN current best-disjunct target), then K
//! box-projected ascent steps. Wall-clock throughput ~K× the sequential loop,
//! so the whole preset restart budget fits INSIDE the existing attack cap —
//! near-wall UNSAT instances keep their BaB slice untouched.
//!
//! Lane strategy (measured on metaroom, 2026-07): waves INTERLEAVE two proven
//! attack modes — even restart indices are TARGETED lanes (fixed target class,
//! ranked once at the clean image, clean-init; the α,β-CROWN-style attack that
//! flips spec_idx_148, whose random-restart margin plateaus at -5.01 while the
//! ranked-target ascent reaches the counterexample), odd indices are DYNAMIC
//! lanes (uniform-random init, per-step best-disjunct re-picking; the legacy
//! strategy that flips spec_idx_129/144/101/43). Each wave therefore carries
//! both strategies, so neither class of instance regresses.
//!
//! DEFAULT ON when a GPU CROWN engine is present and the wide plan builds for
//! the graph shape; kill switch `NY_PGD_VJP_BATCH=0` (shared with the
//! conjunctive batched driver). Restart width: `NY_PGD_VJP_K` (default 32).
//! The experimental sequential-lane levers (`NY_PGD_TARGETED`,
//! `NY_PGD_CLEAN_INIT`, `NY_PGD_GAMA`) bypass this driver so they keep
//! working unchanged.
//!
//! ATTACK-ONLY: gradients steer PGD; a satisfied point is returned as a
//! CANDIDATE that the caller re-evaluates (`re_evaluate_and_confirm`) and the
//! vnncomp trusted-ORT gate re-confirms before any `sat` is emitted. Any
//! capability miss falls back to the byte-identical sequential loop. Nothing
//! here touches verdict/bound paths.

use anyhow::Result;
use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{GraphNetwork, PgdConfig, PgdStepState, PointVjpWavePlan};
use ny_tensor::BoundedTensor;
use std::time::Instant;

use super::attack_stall::AttackStallPolicy;
use super::graph_pgd::{
    best_clause_bottleneck_margin, best_disjunctive_target, constraint_target,
    emit_graph_pgd_status, ranked_disjunctive_targets, GraphDisjunctiveAttackOutcome,
    GraphPgdTarget,
};
use super::graph_pgd_exact_vjp_evidence::{
    ExactVjpExecution, ExactVjpInactiveReason as Inactive, ExactVjpOutcome as EvidenceOutcome,
};
use super::graph_pgd_init::{
    evaluate_graph, initialize_graph_point, sample_center_point, sample_uniform_point, SimpleRng,
};
use super::graph_pgd_vjp_disj_odi::{
    observe_clause_margins, odi_directions, run_focus_wave, run_odi_phase, DisjLaneEnv,
    DisjOdiPolicy, DisjOdiReceipt, FocusEnd, LanePhaseEnd,
};

pub(super) enum DisjVjpBatchedOutcome {
    /// The batched attack ran to completion (candidate, deadline, or budget
    /// exhausted) — the caller should NOT run the sequential loop.
    Completed,
    /// The batched lane is unavailable — run the sequential loop instead.
    FallbackToSequential,
}

/// Batched exact-VJP best-disjunct attack. Mutates `outcome` in place
/// (candidate, margin telemetry, budget accounting) and reports whether the
/// batched lane completed or the sequential loop should run. Returns
/// `FallbackToSequential` (never `Err`) on any capability miss.
///
/// `stall` (#attack-stall) is the adaptive cutoff policy for this phase;
/// [`AttackStallPolicy::disabled`] (the default) keeps the lane byte-identical
/// to before the cutoff existed.
///
/// #disj-odi: when `attack.pgd_disj_odi` or `attack.pgd_disj_focus_lanes` is
/// armed, one fixed-schema `[pgd-vjp-disj] disj-odi` line is printed on EVERY
/// return path (including `Err`); dark, nothing is printed and the driver is
/// byte-identical.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_graph_disjunctive_pgd_vjp_batched(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    pgd_config: &PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    stall: AttackStallPolicy,
    outcome: &mut GraphDisjunctiveAttackOutcome,
) -> Result<DisjVjpBatchedOutcome> {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::Attack);
    let mut receipt = DisjOdiReceipt::new(DisjOdiPolicy::from_config(pgd_config.exact_vjp));
    let result = run_disj_vjp_batched(
        graph,
        input,
        clauses,
        pgd_config,
        gemm_engine,
        json,
        stall,
        outcome,
        &mut receipt,
    );
    if receipt.policy().armed() {
        receipt.outcome = disj_odi_outcome_token(&result, outcome);
        eprintln!("{}", receipt.line());
    }
    result
}

/// The driver's terminal disposition for the #disj-odi receipt.
fn disj_odi_outcome_token(
    result: &Result<DisjVjpBatchedOutcome>,
    outcome: &GraphDisjunctiveAttackOutcome,
) -> &'static str {
    match result {
        Err(_) => "error",
        Ok(DisjVjpBatchedOutcome::FallbackToSequential) => "fallback_sequential",
        Ok(DisjVjpBatchedOutcome::Completed) => {
            if outcome.candidate.is_some() {
                "candidate"
            } else if outcome.stalled_out {
                "stalled"
            } else if outcome.hit_deadline {
                "deadline"
            } else {
                "exhausted"
            }
        }
    }
}

/// The driver body. `receipt` carries the #disj-odi policy and counters; with
/// both keys dark none of its phases run and this is the historical driver.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_disj_vjp_batched(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    pgd_config: &PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    stall: AttackStallPolicy,
    outcome: &mut GraphDisjunctiveAttackOutcome,
    receipt: &mut DisjOdiReceipt,
) -> Result<DisjVjpBatchedOutcome> {
    use DisjVjpBatchedOutcome::{Completed, FallbackToSequential};

    let evidence = ExactVjpExecution::disjunctive(
        pgd_config.exact_vjp,
        pgd_config.num_restarts.max(1),
        pgd_config.num_steps,
    );
    // A deferred/zero slice must not pay the potentially large exact-VJP plan
    // construction cost before discovering that no attack time exists.
    // #attack-setup-regrant: unless the phase offered a re-grant that still has
    // time under its cap — then the plan is built and the re-grant is applied
    // below, once the search is actually ready to start.
    let mut deadline = pgd_config.deadline;
    if exact_vjp_slice_expired(deadline, Instant::now())
        && pgd_config
            .setup_regrant
            .and_then(|r| r.regranted_deadline(deadline, Instant::now()))
            .is_none()
    {
        outcome.hit_deadline = true;
        evidence.finish(EvidenceOutcome::NotStarted, Inactive::DeadlineBeforeStart);
        return Ok(Completed);
    }
    let Some(gpu) = gemm_engine.and_then(|e| e.as_gpu_crown_backward()) else {
        evidence.finish(EvidenceOutcome::NotStarted, Inactive::GpuUnavailable);
        return Ok(FallbackToSequential);
    };
    if !clauses
        .iter()
        .flatten()
        .any(|constraint| constraint_target(constraint).is_some())
    {
        evidence.finish(EvidenceOutcome::NotStarted, Inactive::EmptyObjective);
        return Ok(FallbackToSequential);
    }
    // The wide plan builds ONCE per attack (Arc-shared weights); `None` means
    // the graph is outside both the pure-chain AND the chain+residual resnet
    // fragments (#batched-vjp-resnet) → sequential loop.
    let Some(plan) = PointVjpWavePlan::build(graph, input) else {
        evidence.finish(EvidenceOutcome::NotStarted, Inactive::UnsupportedPlan);
        return Ok(FallbackToSequential);
    };
    // Prove the disabled treatment had a live counterfactual route without
    // issuing an attack forward or a GPU VJP. Plan construction is deterministic
    // and may consume the remaining slice, hence the second deadline check.
    if exact_vjp_slice_expired(deadline, Instant::now()) {
        // #attack-setup-regrant: the setup (dispatch prelude + this plan) ate
        // the whole slice before the first search step. Re-grant, once, the
        // slice the phase declared from NOW, capped by the overall deadline.
        let now = Instant::now();
        let regranted = pgd_config
            .setup_regrant
            .and_then(|r| r.regranted_deadline(deadline, now));
        let Some(fresh) = regranted else {
            outcome.hit_deadline = true;
            evidence.finish(EvidenceOutcome::NotStarted, Inactive::DeadlineBeforeStart);
            return Ok(Completed);
        };
        eprintln!(
            "[attack-setup-regrant] regranted=1 route=disjunctive overrun_s={:.3} slice_s={:.3}",
            deadline.map_or(0.0, |d| now.saturating_duration_since(d).as_secs_f64()),
            fresh.saturating_duration_since(now).as_secs_f64(),
        );
        deadline = Some(fresh);
    }
    let deadline = deadline;
    if !pgd_config.exact_vjp.batch_enabled {
        evidence.finish(EvidenceOutcome::BatchDisabled, Inactive::BatchDisabled);
        return Ok(FallbackToSequential);
    }

    let satisfied = |output: &ArrayD<f32>| {
        clauses
            .iter()
            .any(|clause| super::check_unsafe_counterexample(output, clause))
    };
    // #disj-odi: the phases' view of this invocation. Only consulted when a
    // #disj-odi key is armed.
    let odi_policy = receipt.policy();
    let env = DisjLaneEnv {
        graph,
        input,
        clauses,
        plan: &plan,
        gpu,
        gemm_engine,
        deadline,
    };

    let diag = std::env::var("NY_PGD_DIAG").ok().as_deref() == Some("1");
    let attack_start = Instant::now();
    // #attack-stall: probe window sized from THIS phase's own slice
    // (deadline - attack_start), so the policy carries no absolute constant.
    let mut stall_monitor = stall.monitor(attack_start, deadline);
    let deadline_hit = || deadline.is_some_and(|d| Instant::now() >= d);
    let total_restarts = pgd_config.num_restarts.max(1);
    let width = pgd_config.exact_vjp.disjunctive_batch_width();
    // Restart-index seed offset: same continuation convention as the
    // sequential loop (`seed = 42 + k` ⇒ start the sequence at index k), so
    // the adaptive attack extension continues instead of replaying. Default
    // seed (42) ⇒ offset 0 ⇒ the conjunctive driver's historical `42 + i`.
    // Only the stochastic DYNAMIC lanes consume it — the TARGETED lanes are
    // deterministic (clean init, fixed target), so a retry re-running them
    // replays their cut prefix and CONTINUES past it, which is the point.
    let restart_seed_offset = pgd_config.seed.wrapping_sub(42);

    // Targeted-lane schedule: target classes ranked once at the clean image
    // (closest-to-violation first). Even restart indices walk this ranking
    // from clean init; odd indices are dynamic random-init lanes.
    let clean_point = sample_center_point(input);
    evidence.mark_work_started();
    // #disj-odi focus: per-clause running max of bottleneck margins over the
    // clean point and every restart-wave point (`None` unless focus lanes are
    // requested, so the dark driver never computes it).
    let mut focus_clause_best: Option<Vec<f32>> = None;
    let mut focus_observed_waves = false;
    let ranked: Vec<GraphPgdTarget> = {
        let clean_out = evaluate_graph(graph, &clean_point, gemm_engine)?;
        if let Some(m) = best_clause_bottleneck_margin(&clean_out, clauses) {
            outcome.best_margin = outcome.best_margin.max(m);
            // #attack-stall: anchor the first probe window at the clean point —
            // the start of the ascent. This first observation can never cut (no
            // window has elapsed yet); it only fixes the baseline the window is
            // judged against. `is_armed()` keeps the unarmed path free.
            if stall_monitor.is_armed() {
                stall_monitor.observe(
                    m,
                    super::disjunctive_pgd::noise_scaled_margin(&clean_out),
                    attack_start,
                );
            }
        }
        if satisfied(&clean_out) {
            // The clean image itself violates the property.
            outcome.candidate = Some(clean_point);
            evidence.finish(EvidenceOutcome::Candidate, Inactive::None);
            return Ok(Completed);
        }
        if odi_policy.tracks_focus() {
            let mut best = vec![f32::NEG_INFINITY; clauses.len()];
            observe_clause_margins(&mut best, &clean_out, clauses);
            focus_clause_best = Some(best);
        }
        ranked_disjunctive_targets(&clean_out, clauses)
    };
    // Lane strategy for a restart index (see module docs): `Some(target)` ⇒
    // targeted clean-init lane, `None` ⇒ dynamic random-init lane.
    let lane_target = |r: usize| -> Option<GraphPgdTarget> {
        if r.is_multiple_of(2) && r / 2 < ranked.len() {
            Some(ranked[r / 2])
        } else {
            None
        }
    };

    emit_graph_pgd_status(
        json,
        format_args!(
            "  Graph disjunctive PGD: batched exact-VJP attack ({} clauses, {} restarts in waves of {width} [interleaved targeted/dynamic], {} steps; NY_PGD_VJP_BATCH=0 disables)",
            clauses.len(),
            total_restarts,
            pgd_config.num_steps
        ),
    );

    let mut fwd_ms = 0.0f64;
    let mut vjp_ms = 0.0f64;
    let mut wave_steps = 0usize;

    let mut restart_base = 0usize;
    while restart_base < total_restarts {
        if deadline_hit() {
            outcome.hit_deadline = true;
            break;
        }
        let k = width.min(total_restarts - restart_base);

        let mut rngs: Vec<SimpleRng> = (0..k)
            .map(|i| {
                SimpleRng::new(42 + restart_seed_offset.wrapping_add((restart_base + i) as u64))
            })
            .collect();
        let fixed_targets: Vec<Option<GraphPgdTarget>> =
            (0..k).map(|i| lane_target(restart_base + i)).collect();
        let mut xs: Vec<ArrayD<f32>> = Vec::with_capacity(k);
        for (i, rng) in rngs.iter_mut().enumerate() {
            xs.push(if fixed_targets[i].is_some() {
                // Targeted lane: from the clean image toward its fixed target.
                clean_point.clone()
            } else {
                initialize_graph_point(pgd_config, graph, input, rng, gemm_engine)?
            });
        }
        outcome.restarts_started += k;
        evidence.start_wave(restart_base, k, 0);
        let mut states: Vec<PgdStepState> = (0..k)
            .map(|_| {
                PgdStepState::from_config(
                    pgd_config.optimizer,
                    pgd_config.alpha_mode,
                    pgd_config.step_size,
                    pgd_config.adam,
                    input,
                    input.shape(),
                )
            })
            .collect();

        // #disj-odi: output-diversified initialization of this wave's DYNAMIC
        // lanes (targeted lanes keep their clean init). Each dynamic lane
        // draws a fixed output direction from its own stream AFTER its init,
        // then the lanes take the configured ODI sign steps through one
        // batched exact VJP per step, every ODI point screened like any other.
        // Dark (`attack.pgd_disj_odi` unset) ⇒ zero steps ⇒ skipped entirely.
        let wave_odi_steps = odi_policy.wave_odi_steps();
        if wave_odi_steps > 0 {
            let dynamic: Vec<usize> = (0..k).filter(|&i| fixed_targets[i].is_none()).collect();
            let directions = odi_directions(&mut rngs, &dynamic, plan.output_dim());
            let observe = focus_clause_best.is_some() && !dynamic.is_empty();
            match run_odi_phase(
                &env,
                &mut xs,
                &dynamic,
                &directions,
                wave_odi_steps,
                odi_policy.odi_eta,
                focus_clause_best.as_deref_mut(),
                receipt,
                &evidence,
                outcome,
                &mut stall_monitor,
            )? {
                LanePhaseEnd::Continue => {}
                LanePhaseEnd::Candidate { lane, step } => {
                    if !json {
                        println!(
                            "  Graph disjunctive PGD (batched exact-VJP) found candidate at restart {}, ODI step {step}!",
                            restart_base + lane
                        );
                    }
                    evidence.finish(EvidenceOutcome::Candidate, Inactive::None);
                    return Ok(Completed);
                }
                LanePhaseEnd::Deadline => {
                    outcome.hit_deadline = true;
                    evidence.finish(EvidenceOutcome::Deadline, Inactive::None);
                    return Ok(Completed);
                }
                LanePhaseEnd::Stalled => {
                    outcome.stalled_out = true;
                    super::attack_stall::report_stall_cut(
                        json,
                        diag,
                        format_args!("wave {restart_base} ODI"),
                        stall_monitor.best_margin(),
                    );
                    evidence.finish(EvidenceOutcome::Stalled, Inactive::None);
                    return Ok(Completed);
                }
            }
            if observe {
                focus_observed_waves = true;
            }
            if diag {
                eprintln!(
                    "[pgd-vjp-disj] wave {restart_base}: ODI init done ({wave_odi_steps} steps, eta={}, dynamic_lanes={}, {:.1}s)",
                    odi_policy.odi_eta,
                    dynamic.len(),
                    attack_start.elapsed().as_secs_f64()
                );
            }
        }

        for step in 0..pgd_config.num_steps {
            if deadline_hit() {
                outcome.hit_deadline = true;
                if diag {
                    eprintln!(
                        "[pgd-vjp-disj] deadline at wave {restart_base} step {step} ({:.1}s): wave_steps={wave_steps} avg fwd={:.1}ms vjp={:.1}ms best_margin={:.5}",
                        attack_start.elapsed().as_secs_f64(),
                        fwd_ms / wave_steps.max(1) as f64,
                        vjp_ms / wave_steps.max(1) as f64,
                        outcome.best_margin
                    );
                }
                evidence.finish(EvidenceOutcome::Deadline, Inactive::None);
                return Ok(Completed);
            }

            // Batched template forward: every restart's ReLU masks (fold
            // order) + network outputs, one rayon-parallel CPU pass.
            let t0 = Instant::now();
            let points: Vec<Vec<f32>> = xs.iter().map(|x| x.iter().copied().collect()).collect();
            let (masks, outputs) = match plan.forward_masks(&points) {
                Ok(v) => v,
                Err(e) => {
                    if diag {
                        eprintln!("[pgd-vjp-disj] forward failed ({e}); sequential fallback");
                    }
                    evidence.finish(
                        EvidenceOutcome::FallbackSequential,
                        Inactive::ForwardFailure,
                    );
                    return Ok(FallbackToSequential);
                }
            };

            // Candidate screen + per-restart BEST-DISJUNCT spec rows.
            let mut spec_rows: Vec<f32> = Vec::with_capacity(k * plan.output_dim());
            // One clock read for the whole wave-step: every lane's margin is
            // observed at the same instant, which is what the running max means.
            // `None` when the cutoff is unarmed — the default path does not
            // even read the clock here.
            let step_now = stall_monitor.is_armed().then(Instant::now);
            for kk in 0..k {
                let out_arr =
                    ArrayD::from_shape_vec(IxDyn(&[plan.output_dim()]), outputs[kk].clone())?;
                if let Some(best) = focus_clause_best.as_deref_mut() {
                    observe_clause_margins(best, &out_arr, clauses);
                    focus_observed_waves = true;
                }
                if let Some(m) = best_clause_bottleneck_margin(&out_arr, clauses) {
                    outcome.best_margin = outcome.best_margin.max(m);
                    // #attack-stall: a full probe window in which the running
                    // max did not rise past the confirmation noise floor ⇒ the
                    // ascent has plateaued across every lane in flight. Stop
                    // GENERATING candidates; the caller sees the same "no
                    // counterexample" it sees from a finished attack, and BaB
                    // re-bases on the reclaimed remaining budget.
                    if let Some(now) = step_now {
                        if stall_monitor.observe(
                            m,
                            super::disjunctive_pgd::noise_scaled_margin(&out_arr),
                            now,
                        ) {
                            outcome.stalled_out = true;
                            super::attack_stall::report_stall_cut(
                                json,
                                diag,
                                format_args!("wave {restart_base} step {step} lane {kk}"),
                                stall_monitor.best_margin(),
                            );
                            evidence.finish(EvidenceOutcome::Stalled, Inactive::None);
                            return Ok(Completed);
                        }
                    }
                }
                if satisfied(&out_arr) {
                    // #batched-vjp-resnet screen hardening: the TEMPLATE forward
                    // that fired this screen is ny's own f32 fold; on deep conv
                    // resnets its output deviates from the exact/ORT forward by
                    // more than a razor-thin margin, so a barely-positive screen
                    // hit can fail the caller's `re_evaluate_and_confirm` — and
                    // returning here would ABORT the whole attack on a near-miss
                    // (measured: cifar100 4752, candidate at step 6, unconfirmed,
                    // attack over). Re-screen on the exact/ORT-routed forward with
                    // the SAME noise-scaled margin the caller's confirm uses;
                    // only a candidate that will actually confirm ends the
                    // attack — otherwise the lane keeps ascending (the margin is
                    // still improving; a true crossing confirms a few steps
                    // later). ATTACK-ONLY: this gates candidate hand-up, never a
                    // verdict (the ORT gate remains the sole sat authority).
                    let exact_out = evaluate_graph(graph, &xs[kk], gemm_engine)?;
                    let eps = super::disjunctive_pgd::noise_scaled_margin(&exact_out);
                    let exact_margin = best_clause_bottleneck_margin(&exact_out, clauses);
                    if satisfied(&exact_out) && exact_margin.is_some_and(|m| m >= eps) {
                        if !json {
                            println!(
                                "  Graph disjunctive PGD (batched exact-VJP) found candidate at restart {}, step {step}!",
                                restart_base + kk
                            );
                        }
                        outcome.candidate = Some(xs[kk].clone());
                        evidence.finish(EvidenceOutcome::Candidate, Inactive::None);
                        return Ok(Completed);
                    }
                    if diag {
                        eprintln!(
                            "[pgd-vjp-disj] near-miss at restart {} step {step}: template screen fired, exact margin {:?} < eps {eps:.2e} — lane continues",
                            restart_base + kk,
                            exact_margin
                        );
                    }
                }
                // Targeted lanes ascend their FIXED ranked target; dynamic
                // lanes re-pick the current best disjunct every step.
                let Some(row) = fixed_targets[kk]
                    .or_else(|| best_disjunctive_target(&out_arr, clauses))
                    .map(|t| t.to_spec_row(plan.output_dim()))
                else {
                    // No modeled constraint in any clause — nothing to ascend.
                    evidence.finish(
                        EvidenceOutcome::FallbackSequential,
                        Inactive::NoAttackTarget,
                    );
                    return Ok(FallbackToSequential);
                };
                spec_rows.extend(row.row(0).iter().copied());
            }
            let t1 = Instant::now();

            // ONE wide GPU pass: all K exact best-disjunct margin gradients.
            let grads = match plan.gpu_vjp(gpu, &masks, &spec_rows) {
                Ok(g) => g,
                Err(e) => {
                    if diag {
                        eprintln!("[pgd-vjp-disj] wide VJP failed ({e}); sequential fallback");
                    }
                    evidence.finish(EvidenceOutcome::FallbackSequential, Inactive::VjpFailure);
                    return Ok(FallbackToSequential);
                }
            };
            if grads.len() != k {
                evidence.finish(
                    EvidenceOutcome::FallbackSequential,
                    Inactive::GradientCountMismatch,
                );
                return Ok(FallbackToSequential);
            }
            evidence.successful_vjp(false);
            // #batched-vjp-resnet gradient cross-check (`NY_PGD_VJP_CHECK=1`,
            // diagnostics only): on the FIRST step, compare lane 0's wide-pass
            // gradient against the sequential exact point-Jacobian
            // (`attack_point_gradient`) at the same point/spec row. Attack-only
            // telemetry — a mismatch is printed, never acted on (the ORT gate
            // remains the sole sat authority either way).
            if restart_base == 0
                && step == 0
                && std::env::var("NY_PGD_VJP_CHECK").ok().as_deref() == Some("1")
            {
                let row = Array2::from_shape_vec(
                    (1, plan.output_dim()),
                    spec_rows[..plan.output_dim()].to_vec(),
                )?;
                match graph.attack_point_gradient(&xs[0], &row, gemm_engine, None) {
                    Ok(Some(seq)) => {
                        let g0 = &grads[0];
                        let (mut dot, mut na, mut nb, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
                        for (&a, &b) in seq.iter().zip(g0.iter()) {
                            dot += f64::from(a) * f64::from(b);
                            na += f64::from(a) * f64::from(a);
                            nb += f64::from(b) * f64::from(b);
                            max_abs = max_abs.max((f64::from(a) - f64::from(b)).abs());
                        }
                        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
                        eprintln!(
                            "[pgd-vjp-check] lane0 wide-vs-sequential: cos={cos:.9} max_abs_diff={max_abs:.3e} |seq|={:.3e} |wide|={:.3e}",
                            na.sqrt(),
                            nb.sqrt()
                        );
                    }
                    other => {
                        eprintln!(
                            "[pgd-vjp-check] sequential gradient unavailable ({})",
                            match other {
                                Ok(None) => "outside fragment/deadline".to_string(),
                                Err(e) => format!("{e}"),
                                Ok(Some(_)) => unreachable!(),
                            }
                        );
                    }
                }
            }
            let t2 = Instant::now();
            fwd_ms += (t1 - t0).as_secs_f64() * 1e3;
            vjp_ms += (t2 - t1).as_secs_f64() * 1e3;
            wave_steps += 1;
            outcome.steps_taken += k;

            // K ascent steps with box projection (+ restart-when-stuck).
            for kk in 0..k {
                let grad = ArrayD::from_shape_vec(IxDyn(xs[kk].shape()), grads[kk].clone())?;
                let previous_x = pgd_config.restart_when_stuck.then(|| xs[kk].clone());
                xs[kk] = states[kk].step(&grad, &xs[kk], input, true);
                if let Some(prev) = previous_x {
                    if prev.iter().zip(xs[kk].iter()).all(|(&a, &b)| a == b) {
                        xs[kk] = sample_uniform_point(input, &mut rngs[kk]);
                        states[kk].reset();
                    }
                }
            }

            if diag && (step + 1) % 100 == 0 {
                eprintln!(
                    "[pgd-vjp-disj] wave {restart_base} step {} K={k}: avg fwd={:.1}ms vjp={:.1}ms step_total={:.1}ms best_margin={:.5} (>=0 => CE)",
                    step + 1,
                    fwd_ms / wave_steps as f64,
                    vjp_ms / wave_steps as f64,
                    (fwd_ms + vjp_ms) / wave_steps as f64,
                    outcome.best_margin
                );
            }
        }

        // Post-wave final check at the last points (one more batched forward).
        let points: Vec<Vec<f32>> = xs.iter().map(|x| x.iter().copied().collect()).collect();
        if let Ok((_m, outputs)) = plan.forward_masks(&points) {
            let wave_end = stall_monitor.is_armed().then(Instant::now);
            for kk in 0..k {
                let out_arr =
                    ArrayD::from_shape_vec(IxDyn(&[plan.output_dim()]), outputs[kk].clone())?;
                if let Some(best) = focus_clause_best.as_deref_mut() {
                    observe_clause_margins(best, &out_arr, clauses);
                    focus_observed_waves = true;
                }
                if let Some(m) = best_clause_bottleneck_margin(&out_arr, clauses) {
                    outcome.best_margin = outcome.best_margin.max(m);
                    // #attack-stall: these points count too — the monitor's
                    // running max must see every point `outcome.best_margin`
                    // sees, or a missed improvement would make a cut MORE
                    // likely than the evidence supports.
                    if let Some(now) = wave_end {
                        if stall_monitor.observe(
                            m,
                            super::disjunctive_pgd::noise_scaled_margin(&out_arr),
                            now,
                        ) {
                            outcome.stalled_out = true;
                            super::attack_stall::report_stall_cut(
                                json,
                                diag,
                                format_args!("wave {restart_base} final check"),
                                stall_monitor.best_margin(),
                            );
                            evidence.finish(EvidenceOutcome::Stalled, Inactive::None);
                            return Ok(Completed);
                        }
                    }
                }
                if satisfied(&out_arr) {
                    // Same exact-forward re-screen as the in-loop candidate path
                    // (see above): only hand up a candidate that will confirm.
                    let exact_out = evaluate_graph(graph, &xs[kk], gemm_engine)?;
                    let eps = super::disjunctive_pgd::noise_scaled_margin(&exact_out);
                    let confirmed = satisfied(&exact_out)
                        && best_clause_bottleneck_margin(&exact_out, clauses)
                            .is_some_and(|m| m >= eps);
                    if confirmed {
                        if !json {
                            println!(
                                "  Graph disjunctive PGD (batched exact-VJP) found candidate at restart {} (final check)!",
                                restart_base + kk
                            );
                        }
                        outcome.candidate = Some(xs[kk].clone());
                        evidence.finish(EvidenceOutcome::Candidate, Inactive::None);
                        return Ok(Completed);
                    }
                }
            }
        }

        restart_base += k;
    }

    // #disj-odi focus stage: the restart waves are done without a candidate
    // and without hitting the deadline; spend the rest of the slice on the
    // closest clauses. A focus deadline ends the stage only — it never sets
    // `outcome.hit_deadline` (the restart-sequence flag the attack extension
    // reads), and focus lanes never count as restarts.
    let mut focus_deadline = false;
    if let Some(best) = focus_clause_best.as_deref() {
        if outcome.hit_deadline {
            receipt.focus_reason = "deadline_before_start";
        } else {
            match run_focus_wave(
                &env,
                best,
                focus_observed_waves,
                restart_seed_offset,
                receipt,
                &evidence,
                outcome,
                &mut stall_monitor,
            )? {
                FocusEnd::Done => {}
                FocusEnd::Candidate { lane, step, odi } => {
                    if !json {
                        println!(
                            "  Graph disjunctive PGD (batched exact-VJP) found candidate at focus lane {lane}, {}step {step}!",
                            if odi { "ODI " } else { "" }
                        );
                    }
                    evidence.finish(EvidenceOutcome::Candidate, Inactive::None);
                    return Ok(Completed);
                }
                FocusEnd::Deadline => focus_deadline = true,
                FocusEnd::Stalled => {
                    outcome.stalled_out = true;
                    super::attack_stall::report_stall_cut(
                        json,
                        diag,
                        format_args!("focus wave"),
                        stall_monitor.best_margin(),
                    );
                    evidence.finish(EvidenceOutcome::Stalled, Inactive::None);
                    return Ok(Completed);
                }
            }
        }
    }

    if diag {
        eprintln!(
            "[pgd-vjp-disj] {} ({:.1}s): wave_steps={wave_steps} avg fwd={:.1}ms vjp={:.1}ms best_margin={:.5}",
            if outcome.hit_deadline {
                "deadline"
            } else {
                "exhausted"
            },
            attack_start.elapsed().as_secs_f64(),
            fwd_ms / wave_steps.max(1) as f64,
            vjp_ms / wave_steps.max(1) as f64,
            outcome.best_margin
        );
    }
    emit_graph_pgd_status(
        json,
        format_args!(
            "  Graph disjunctive PGD (batched exact-VJP): no candidate found. ({:.2}s, {})",
            attack_start.elapsed().as_secs_f64(),
            if outcome.hit_deadline {
                "deadline"
            } else {
                "budget exhausted"
            }
        ),
    );
    evidence.finish(
        if outcome.hit_deadline || focus_deadline {
            EvidenceOutcome::Deadline
        } else {
            EvidenceOutcome::Exhausted
        },
        Inactive::None,
    );
    Ok(Completed)
}

fn exact_vjp_slice_expired(deadline: Option<Instant>, now: Instant) -> bool {
    deadline.is_some_and(|limit| now >= limit)
}

#[cfg(test)]
mod deadline_admission_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn expired_or_zero_slice_is_rejected_before_plan_construction() {
        let now = Instant::now();
        assert!(exact_vjp_slice_expired(Some(now), now));
        assert!(exact_vjp_slice_expired(
            Some(
                now.checked_sub(Duration::from_nanos(1))
                    .expect("an Instant one nanosecond earlier must be representable"),
            ),
            now
        ));
        assert!(!exact_vjp_slice_expired(None, now));
        assert!(!exact_vjp_slice_expired(
            Some(now + Duration::from_secs(1)),
            now
        ));
    }
}
