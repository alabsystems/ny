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

use super::graph_pgd::{
    best_clause_bottleneck_margin, best_disjunctive_target, emit_graph_pgd_status,
    ranked_disjunctive_targets, GraphDisjunctiveAttackOutcome, GraphPgdTarget,
};
use super::graph_pgd_init::{
    evaluate_graph, initialize_graph_point, sample_center_point, sample_uniform_point, SimpleRng,
};

pub(super) enum DisjVjpBatchedOutcome {
    /// The batched attack ran to completion (candidate, deadline, or budget
    /// exhausted) — the caller should NOT run the sequential loop.
    Completed,
    /// The batched lane is unavailable — run the sequential loop instead.
    FallbackToSequential,
}

/// Parallel restart width (shared knob with the conjunctive driver).
fn vjp_batch_width() -> usize {
    std::env::var("NY_PGD_VJP_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(32)
}

/// Batched exact-VJP best-disjunct attack. Mutates `outcome` in place
/// (candidate, margin telemetry, budget accounting) and reports whether the
/// batched lane completed or the sequential loop should run. Returns
/// `FallbackToSequential` (never `Err`) on any capability miss.
pub(super) fn try_graph_disjunctive_pgd_vjp_batched(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    pgd_config: &PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    outcome: &mut GraphDisjunctiveAttackOutcome,
) -> Result<DisjVjpBatchedOutcome> {
    use DisjVjpBatchedOutcome::{Completed, FallbackToSequential};

    if std::env::var("NY_PGD_VJP_BATCH").ok().as_deref() == Some("0") {
        return Ok(FallbackToSequential);
    }
    let Some(gpu) = gemm_engine.and_then(|e| e.as_gpu_crown_backward()) else {
        return Ok(FallbackToSequential);
    };
    // The wide plan builds ONCE per attack (Arc-shared weights); `None` means
    // the graph is outside both the pure-chain AND the chain+residual resnet
    // fragments (#batched-vjp-resnet) → sequential loop.
    let Some(plan) = PointVjpWavePlan::build(graph, input) else {
        return Ok(FallbackToSequential);
    };

    let satisfied = |output: &ArrayD<f32>| {
        clauses
            .iter()
            .any(|clause| super::check_unsafe_counterexample(output, clause))
    };

    let diag = std::env::var("NY_PGD_DIAG").ok().as_deref() == Some("1");
    let attack_start = Instant::now();
    let deadline_hit = || pgd_config.deadline.is_some_and(|d| Instant::now() >= d);
    let total_restarts = pgd_config.num_restarts.max(1);
    let width = vjp_batch_width();
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
    let ranked: Vec<GraphPgdTarget> = {
        let clean_out = evaluate_graph(graph, &clean_point, gemm_engine)?;
        if let Some(m) = best_clause_bottleneck_margin(&clean_out, clauses) {
            outcome.best_margin = outcome.best_margin.max(m);
        }
        if satisfied(&clean_out) {
            // The clean image itself violates the property.
            outcome.candidate = Some(clean_point);
            return Ok(Completed);
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
                    return Ok(FallbackToSequential);
                }
            };

            // Candidate screen + per-restart BEST-DISJUNCT spec rows.
            let mut spec_rows: Vec<f32> = Vec::with_capacity(k * plan.output_dim());
            for kk in 0..k {
                let out_arr =
                    ArrayD::from_shape_vec(IxDyn(&[plan.output_dim()]), outputs[kk].clone())?;
                if let Some(m) = best_clause_bottleneck_margin(&out_arr, clauses) {
                    outcome.best_margin = outcome.best_margin.max(m);
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
                    return Ok(FallbackToSequential);
                }
            };
            if grads.len() != k {
                return Ok(FallbackToSequential);
            }
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
            for kk in 0..k {
                let out_arr =
                    ArrayD::from_shape_vec(IxDyn(&[plan.output_dim()]), outputs[kk].clone())?;
                if let Some(m) = best_clause_bottleneck_margin(&out_arr, clauses) {
                    outcome.best_margin = outcome.best_margin.max(m);
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
                        return Ok(Completed);
                    }
                }
            }
        }

        restart_base += k;
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
    Ok(Completed)
}
