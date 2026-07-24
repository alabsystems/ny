// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched exact-VJP graph PGD (#batched-vjp): K deep restarts per step from
//! ONE wide GPU pass.
//!
//! The sequential joint-margin exact-gradient loop in `graph_pgd.rs` pays ~140 ms
//! per step per restart (one certified forward + one backward per point), so a
//! 121 s attack budget covers ~1 deep (~800-step) restart — while the
//! alpha-beta-CROWN reference runs 250 restarts × 1000 steps. This driver runs
//! the SAME joint AND-clause ascent, but for K restarts simultaneously:
//!
//! per step: one batched CPU template forward captures every restart's ReLU
//! masks + outputs ([`ny_propagate::point_vjp_forward_masks`]), then ONE wide
//! GPU pass ([`ny_core::GpuCrownBackward::crown_point_vjp_batched`]) returns all
//! K exact joint-margin gradients, then K signed-gradient/Adam box-projected
//! steps.
//!
//! DEFAULT ON when the wide plan builds for the graph shape (pure conv/linear
//! ReLU chain) and a GPU CROWN engine is present; kill switch
//! `NY_PGD_VJP_BATCH=0`. Restart width: `NY_PGD_VJP_K` (default 32, aiming
//! 32–64 parallel deep restarts).
//!
//! ATTACK-ONLY: gradients steer PGD; every counterexample candidate still flows
//! through `revalidate_graph_counterexample` (ORT / independent forward) before
//! a `sat` is ever claimed, and any assembly/GPU failure falls back to the
//! byte-identical sequential loop. Nothing here touches verdict/bound paths.

use anyhow::Result;
use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::VnnLibSpec;
use ny_propagate::gama::{gama_lambda_at, gama_lin_steps};
use ny_propagate::{GraphNetwork, PgdConfig, PgdStepState, PointVjpWavePlan};
use ny_tensor::BoundedTensor;
use std::time::Instant;

use super::graph_pgd::{
    add_gama_guidance_to_spec_row, constraint_target, emit_graph_pgd_status, gama_lambda_init,
    gama_reference_softmax, joint_hinge_loss, reset_gama_lane, GraphPgdTarget,
};
use super::graph_pgd_init::{
    initialize_graph_point, revalidate_graph_counterexample, sample_uniform_point, SimpleRng,
};

// Transient return value whose LARGE variant is the common case: boxing it
// would cost an allocation per attack outcome for no storage win.
#[allow(clippy::large_enum_variant)]
pub(super) enum VjpBatchedOutcome {
    /// The batched attack ran to completion (counterexample or budget spent).
    Completed(Option<(ArrayD<f32>, ArrayD<f32>)>),
    /// The batched lane is unavailable/failed — run the sequential loop instead.
    FallbackToSequential,
}

/// Parallel restart width; override with `NY_PGD_VJP_K` for A/B throughput
/// probes. Default 64: the wide GPU pass is ~72ms fixed + ~2ms/restart on the
/// soundnessbench conv chain (measured K=8/16/32/64/128 → 88/108/136/194/318
/// ms/step), so K=64 costs 3.0ms per restart-step vs 4.3ms at K=32 — 40% more
/// restart-steps per slice, and twice the ODI basin directions per wave. Needs
/// the big-bindings device limits (default ON); if the wide pass fails on this
/// adapter the wave HALVES its width and retries (never the sequential cliff).
fn vjp_batch_width() -> usize {
    std::env::var("NY_PGD_VJP_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(64)
}

/// #legacy-lanes (`NY_PGD_LEGACY_LANES`, default 20, 0 disables): in the FIRST
/// wave, this many trailing lanes are UNIFORM-init "legacy" lanes that REPLAY
/// the pre-ODI wave's restart seeds (42, 43, …) instead of taking fresh ODI
/// seeds — so counterexamples that the old uniform wave reached deterministically
/// stay reachable (measured: soundnessbench model_0's CE lands at uniform-seed-55
/// step 657, which pure-ODI K=64's 616-step wave lost). Per-lane trajectories
/// depend only on the lane's own seed/rng, so the leading ODI lanes keep the
/// trajectories that win models 6/8/16/26/30/35 and the legacy lanes keep the
/// old wave's. K=52 = 32 ODI + 20 legacy keeps the wave deep enough (~690
/// steps in the 121 s slice at 176 ms/step) for the step-657 legacy CE.
/// Legacy lanes are exempt from #exploit-recycle (a legacy climber can rank
/// low while still deterministically heading for its CE).
/// DEFAULT OFF: measured Pareto on soundnessbench — `NY_PGD_LEGACY_LANES=20
/// NY_PGD_VJP_K=52` recovers model_0's legacy CE but loses models 6/8/30
/// (their winning seeds come from ODI lanes 32-63, which verifiably do NOT
/// re-win from lanes 0-33 even with the deeper 706-step wave), a net -2 vs
/// the pure-ODI K=64 wave. No configuration keeps both: the legacy replay
/// needs >=657 steps (=> K<=54 in the 121s slice) while 6/8/30 need the full
/// 64-direction ODI fan. Kept as an opt-in probe lever.
fn legacy_lanes() -> usize {
    std::env::var("NY_PGD_LEGACY_LANES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
}

/// #odi steps: Output Diversified Initialization (Tashiro et al., CVPR 2020).
/// DEFAULT ON (`NY_PGD_ODI_STEPS=0` disables): before the joint-margin ascent,
/// each restart lane takes this many signed-gradient steps maximizing `<w_k,
/// f(x)>` for a FIXED random output-space direction `w_k` — reaching basins
/// that uniform input-noise inits provably miss (the soundnessbench hard-basin
/// diagnosis: the stubborn conjunct stalls in the WRONG basin; more raw steps
/// with uniform init were measured NEGATIVE). Costs `odi_steps` extra batched
/// VJP passes per wave (~1s for 8 steps at K=32) out of a ~121s slice.
/// ATTACK-ONLY: it only moves PGD start points inside the input box.
fn odi_steps() -> usize {
    std::env::var("NY_PGD_ODI_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8)
}

/// #odi step size, in units of the per-coordinate half-range `(hi-lo)/2`.
/// The reference ODI uses the full attack radius (`eta = eps`), i.e. 1.0.
fn odi_eta() -> f32 {
    std::env::var("NY_PGD_ODI_ETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.0)
}

/// #odi wave-steps override (`NY_PGD_VJP_STEPS`): cap the joint-margin steps per
/// wave so MORE waves (more ODI directions) fit the slice — the many-short-
/// restarts schedule. Unset ⇒ the preset `pgd_steps` (deep waves) is kept.
fn vjp_wave_steps(config_steps: usize) -> usize {
    std::env::var("NY_PGD_VJP_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(config_steps)
}

/// #lane-recycle window (`NY_PGD_LANE_WINDOW`, 0 disables, default 100 steps):
/// every window, a lane whose joint-margin gap-closing RATE cannot reach 0 by
/// the remaining budget (`progress_over_window < gap × window / remaining`) is
/// parked at a wrong-basin local max — resample it (fresh uniform point + fresh
/// ODI direction) INSIDE the running wave. Per-lane spec rows make the re-init
/// free; the lane holding the wave-best joint is protected (it feeds the
/// #postbab-seed export), and no lane is recycled when fewer than
/// [`LANE_RECYCLE_MIN_REMAINING`] steps remain (a fresh lane needs time to
/// climb). Adaptive many-short-restarts basin coverage without truncating
/// productive lanes — a plain no-improvement patience never fires here because
/// Adam keeps making sub-1e-7 micro-improvements at a parked local max.
fn lane_recycle_window() -> usize {
    std::env::var("NY_PGD_LANE_WINDOW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        // Default OFF: measured on soundnessbench model_29, rate-recycling
        // perturbed the productive lanes (best joint -5e-6 → -1.1e-4) without
        // flipping anything model 8/29/37. Opt-in probe lever.
        .unwrap_or(0)
}

/// Minimum steps a recycled lane needs to be worth starting (ODI + climb).
const LANE_RECYCLE_MIN_REMAINING: usize = 250;

/// #exploit-recycle (`NY_PGD_EXPLOIT=0` disables, default ON): in the ENDGAME
/// (projected remaining budget below [`EXPLOIT_ENDGAME_STEPS`] steps), when the
/// wave's best joint margin is within [`EXPLOIT_GAP`] of 0, every
/// [`EXPLOIT_INTERVAL`] steps the WORST quarter of lanes respawn as jittered
/// clones of the global best point (uniform noise, scales 1e-3..1e-5 of the
/// half-range — the measured scale of the near-boundary margin oscillation)
/// with fresh optimizer state. Breaks the deterministic Adam limit cycle that
/// parks lanes with 2-3 conjuncts oscillating around 0 (measured flip:
/// soundnessbench model_37, crossing within 17 steps of an endgame respawn).
/// ENDGAME-ONLY because deep-basin climber lanes look BAD mid-flight: firing
/// from step ~300 recycled the very lane that reaches -2e-5 by step 600 on
/// model_6 (wave stalled at -7e-5 instead — flip lost). By the endgame the
/// climbers are at the top of the ranking and protected. Attack-only.
fn exploit_enabled() -> bool {
    std::env::var("NY_PGD_EXPLOIT").ok().as_deref() != Some("0")
}
const EXPLOIT_INTERVAL: usize = 50;
const EXPLOIT_GAP: f32 = 1e-3;
// 150/quarter measured best: 200/half respawned the model_6 climber lane at the
// first endgame firing (still mid-ranked there) and lost that flip.
const EXPLOIT_ENDGAME_STEPS: usize = 150;
const EXPLOIT_SCALES: [f32; 5] = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5];

/// #tau-hinge (`NY_PGD_HINGE_TAU`, default 1e-3, 0 disables): loss ensemble
/// across lanes. ODD lanes ascend the tau-hinge `Σ_c min(margin_c, τ)` — every
/// conjunct with margin < τ stays in the gradient, so barely-positive margins
/// are HELD while the negatives climb. The plain hinge drops a conjunct from
/// the gradient the instant it crosses 0, and the next step knocks it back
/// negative: measured on soundnessbench model_8, the best lane parks with
/// THREE conjuncts oscillating in ±1.5e-4 around 0 forever. EVEN lanes keep
/// the plain hinge (interior-seeking can lose razor-thin CE regions whose
/// tau-interior is empty). Attack-only: the CE screen stays the raw margins.
fn hinge_tau() -> f32 {
    std::env::var("NY_PGD_HINGE_TAU")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        // Default OFF: measured on soundnessbench models 8/29, the τ ensemble
        // did not break the boundary jam and cost plain-hinge lanes (model_8
        // best -8e-5 → -1.3e-4; model_29 confounded regression). Opt-in lever.
        .unwrap_or(0.0)
}

/// Batched exact-VJP joint-margin attack. See the module docs. Returns
/// `FallbackToSequential` (never `Err`) on any capability miss so the caller's
/// sequential exact-gradient loop stays the proven fallback.
pub(super) fn try_graph_pgd_vjp_batched(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    pgd_config: &PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<VjpBatchedOutcome> {
    use VjpBatchedOutcome::{Completed, FallbackToSequential};

    if std::env::var("NY_PGD_VJP_BATCH").ok().as_deref() == Some("0") {
        return Ok(FallbackToSequential);
    }
    let Some(gpu) = gemm_engine.and_then(|e| e.as_gpu_crown_backward()) else {
        return Ok(FallbackToSequential);
    };
    let constraints = &vnnlib.output_constraints;
    let conjunct_targets: Vec<GraphPgdTarget> =
        constraints.iter().filter_map(constraint_target).collect();
    if conjunct_targets.is_empty() {
        return Ok(FallbackToSequential);
    }
    let gama_lambda0 = gama_lambda_init(pgd_config);
    let gama_lin = gama_lin_steps(pgd_config.num_steps);
    // The wide plan builds ONCE per attack (Arc-shared weights); `None` means the
    // graph is outside both the pure-chain AND the chain+residual resnet
    // fragments (#batched-vjp-resnet) → sequential loop.
    let Some(plan) = PointVjpWavePlan::build(graph, input) else {
        return Ok(FallbackToSequential);
    };

    let diag = std::env::var("NY_PGD_DIAG").ok().as_deref() == Some("1");
    let attack_start = Instant::now();
    let deadline_hit = || pgd_config.deadline.is_some_and(|d| Instant::now() >= d);
    let total_restarts = pgd_config.num_restarts.max(1);
    let mut width = vjp_batch_width();

    emit_graph_pgd_status(
        json,
        format_args!(
            "  Graph PGD: batched exact-VJP attack ({} conjuncts, {} restarts in waves of {width}, {} steps; NY_PGD_VJP_BATCH=0 disables)",
            conjunct_targets.len(),
            total_restarts,
            pgd_config.num_steps
        ),
    );

    let tau = hinge_tau();
    let exploit = exploit_enabled();
    let mut best_joint = f32::NEG_INFINITY;
    let mut best_x: Option<ArrayD<f32>> = None;
    let mut lanes_exploited = 0usize;
    let mut fwd_ms = 0.0f64;
    let mut vjp_ms = 0.0f64;
    let mut steps_done = 0usize;

    let mut restart_base = 0usize;
    'waves: while restart_base < total_restarts {
        if deadline_hit() {
            break;
        }
        let k = width.min(total_restarts - restart_base);

        // Per-restart seeds mirror the sequential loop (42 + restart index), so a
        // K=1 wave explores the same start point the serial path would.
        // #legacy-lanes: the trailing `n_legacy` lanes of the FIRST wave replay
        // the OLD uniform wave's seeds (42 + j) so its deterministic CEs stay
        // reachable (see `legacy_lanes`); the leading `odi_n` lanes keep the
        // fresh ODI seed sequence.
        let n_legacy = if restart_base == 0 {
            legacy_lanes().min(k.saturating_sub(1))
        } else {
            0
        };
        let odi_n = k - n_legacy;
        let mut rngs: Vec<SimpleRng> = (0..k)
            .map(|i| {
                if i < odi_n {
                    SimpleRng::new(42 + (restart_base + i) as u64)
                } else {
                    SimpleRng::new(42 + (i - odi_n) as u64)
                }
            })
            .collect();
        let mut xs: Vec<ArrayD<f32>> = Vec::with_capacity(k);
        for rng in &mut rngs {
            xs.push(initialize_graph_point(
                pgd_config,
                graph,
                input,
                rng,
                gemm_engine,
            )?);
        }
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

        // #odi phase: per-lane FIXED random output direction w_k, ascended for a
        // few full-radius signed steps via the SAME batched exact VJP (spec row =
        // w_k). Output-space diversity reaches basins uniform input noise misses.
        // Non-fatal on any failure: the lanes keep their uniform inits.
        // `odi_rows` stays alive: lane recycling below re-rolls a lane's row.
        let n_odi = odi_steps();
        let eta = odi_eta();
        let mut odi_rows: Vec<f32> = Vec::with_capacity(k * plan.output_dim());
        for rng in rngs.iter_mut() {
            for _ in 0..plan.output_dim() {
                odi_rows.push(rng.next_f32() * 2.0 - 1.0);
            }
        }
        if n_odi > 0 {
            'odi: for odi_step in 0..n_odi {
                if deadline_hit() {
                    if diag {
                        eprintln!(
                            "[pgd-vjp] deadline in ODI at wave {restart_base} step {odi_step} ({:.1}s)",
                            attack_start.elapsed().as_secs_f64()
                        );
                    }
                    return Ok(Completed(None));
                }
                let points: Vec<Vec<f32>> =
                    xs.iter().map(|x| x.iter().copied().collect()).collect();
                let Ok((masks, outputs)) = plan.forward_masks(&points) else {
                    break 'odi;
                };
                // Free CE screen on the ODI trajectory (outputs already computed).
                for kk in 0..k {
                    let out_arr =
                        ArrayD::from_shape_vec(IxDyn(&[plan.output_dim()]), outputs[kk].clone())?;
                    let joint = joint_hinge_loss(&conjunct_targets, &out_arr);
                    super::super::best_margin_export::record_best_margin_candidate(joint, &xs[kk]);
                    if joint > best_joint {
                        best_joint = joint;
                        best_x = Some(xs[kk].clone());
                    }
                    if super::check_unsafe_counterexample(&out_arr, constraints) {
                        let ctx = format!(
                            "vjp-batched ODI restart {}, step {odi_step}",
                            restart_base + kk
                        );
                        if let Some(pair) = revalidate_graph_counterexample(
                            graph,
                            xs[kk].clone(),
                            constraints,
                            &ctx,
                        )? {
                            emit_graph_pgd_status(
                                json,
                                format_args!("  Graph PGD found counterexample at {ctx}!"),
                            );
                            return Ok(Completed(Some(pair)));
                        }
                    }
                }
                let Ok(grads) = plan.gpu_vjp(gpu, &masks, &odi_rows) else {
                    break 'odi;
                };
                if grads.len() != k {
                    break 'odi;
                }
                let lower = input.lower();
                let upper = input.upper();
                // Legacy lanes (kk >= odi_n) sit out the ODI phase: they keep
                // their uniform init so their old-wave trajectory replays.
                for kk in 0..odi_n {
                    for ((xi, &g), (lo, hi)) in xs[kk]
                        .iter_mut()
                        .zip(grads[kk].iter())
                        .zip(lower.iter().zip(upper.iter()))
                    {
                        let step = eta * 0.5 * (hi - lo) * g.signum();
                        *xi = (*xi + step).clamp(*lo, *hi);
                    }
                }
            }
            if diag {
                eprintln!(
                    "[pgd-vjp] wave {restart_base}: ODI init done ({n_odi} steps, eta={eta}, odi_lanes={odi_n}, legacy={n_legacy}, {:.1}s)",
                    attack_start.elapsed().as_secs_f64()
                );
            }
        }

        // GAMA references are captured from the first point AFTER each lane's
        // initialization/ODI phase, matching the existing sequential GAMA
        // semantics. Recycling starts a fresh attack trajectory and therefore
        // resets both the reference and its annealing schedule.
        let mut gama_p_refs: Vec<Option<Vec<f32>>> = vec![None; k];
        let mut gama_steps: Vec<usize> = vec![0; k];

        // #lane-recycle state: per-lane best joint, previous-window gap snapshot,
        // and remaining per-lane ODI steps (a recycled lane re-runs its own ODI).
        let window = lane_recycle_window();
        let mut lane_best: Vec<f32> = vec![f32::NEG_INFINITY; k];
        let mut lane_prev_gap: Vec<f32> = vec![f32::INFINITY; k];
        let mut lane_odi_left: Vec<usize> = vec![0; k];
        let mut lanes_recycled = 0usize;
        // #exploit-recycle lineage: exploit-spawned lanes export to a SEPARATE
        // best-margin slot (their jam points differ from plain lanes', and the
        // post-BaB jitter is position-sensitive — see `best_margin_export`).
        let mut lane_exploit: Vec<bool> = vec![false; k];

        let wave_steps = vjp_wave_steps(pgd_config.num_steps);
        for step in 0..wave_steps {
            // #lane-recycle: every window, project each lane's gap-closing rate
            // onto the remaining budget; resample lanes that cannot reach 0 in
            // time (fresh uniform point + fresh ODI direction). The wave-best
            // lane is protected — it feeds the #postbab-seed export that
            // finishes near-boundary instances.
            if window > 0 && step > 0 && step % window == 0 {
                // Remaining budget: the wave's steps left, capped by a
                // deadline-projected step count from the measured step cost.
                let mut remaining = wave_steps - step;
                if let Some(d) = pgd_config.deadline {
                    let avg_step = attack_start.elapsed().as_secs_f64() / steps_done.max(1) as f64;
                    if avg_step > 0.0 {
                        let till_deadline =
                            d.saturating_duration_since(Instant::now()).as_secs_f64();
                        remaining = remaining.min((till_deadline / avg_step) as usize);
                    }
                }
                if remaining >= LANE_RECYCLE_MIN_REMAINING {
                    let best_lane = lane_best
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i);
                    for kk in 0..k {
                        let gap_now = -lane_best[kk]; // > 0 (0 crossing == CE)
                        let progress = lane_prev_gap[kk] - gap_now;
                        let needed = gap_now * (window as f32) / (remaining as f32);
                        if Some(kk) != best_lane
                            && lane_prev_gap[kk].is_finite()
                            && progress < needed
                        {
                            xs[kk] = sample_uniform_point(input, &mut rngs[kk]);
                            states[kk].reset();
                            for w in odi_rows[kk * plan.output_dim()..(kk + 1) * plan.output_dim()]
                                .iter_mut()
                            {
                                *w = rngs[kk].next_f32() * 2.0 - 1.0;
                            }
                            lane_odi_left[kk] = n_odi;
                            lane_best[kk] = f32::NEG_INFINITY;
                            lane_prev_gap[kk] = f32::INFINITY;
                            lane_exploit[kk] = false;
                            reset_gama_lane(&mut gama_p_refs[kk], &mut gama_steps[kk]);
                            lanes_recycled += 1;
                        } else {
                            lane_prev_gap[kk] = gap_now;
                        }
                    }
                }
            }
            // #exploit-recycle: in the ENDGAME, near the boundary, respawn the
            // worst quarter of lanes as jittered clones of the global best
            // point (see docs).
            let endgame = || -> bool {
                let mut remaining = wave_steps - step;
                if let Some(d) = pgd_config.deadline {
                    let avg_step = attack_start.elapsed().as_secs_f64() / steps_done.max(1) as f64;
                    if avg_step > 0.0 {
                        let till = d.saturating_duration_since(Instant::now()).as_secs_f64();
                        remaining = remaining.min((till / avg_step) as usize);
                    }
                }
                remaining <= EXPLOIT_ENDGAME_STEPS
            };
            if exploit
                && step > 0
                && step % EXPLOIT_INTERVAL == 0
                && best_joint > -EXPLOIT_GAP
                && best_joint < 0.0
                && endgame()
            {
                if let Some(bx) = &best_x {
                    // Legacy lanes are exempt: a legacy climber can rank low
                    // while deterministically heading for its old-wave CE.
                    let mut order: Vec<usize> =
                        (0..odi_n).filter(|&kk| lane_odi_left[kk] == 0).collect();
                    order.sort_by(|&a, &b| {
                        lane_best[a]
                            .partial_cmp(&lane_best[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let n_exploit = (k / 4).max(1).min(order.len().saturating_sub(1));
                    let lower = input.lower();
                    let upper = input.upper();
                    for (i, &kk) in order.iter().take(n_exploit).enumerate() {
                        let scale = EXPLOIT_SCALES[i % EXPLOIT_SCALES.len()];
                        let mut x = bx.clone();
                        for (xi, (lo, hi)) in x.iter_mut().zip(lower.iter().zip(upper.iter())) {
                            let s = scale * 0.5 * (hi - lo);
                            *xi = (*xi + (rngs[kk].next_f32() * 2.0 - 1.0) * s).clamp(*lo, *hi);
                        }
                        xs[kk] = x;
                        states[kk].reset();
                        lane_best[kk] = f32::NEG_INFINITY;
                        lane_prev_gap[kk] = f32::INFINITY;
                        lane_exploit[kk] = true;
                        reset_gama_lane(&mut gama_p_refs[kk], &mut gama_steps[kk]);
                        lanes_exploited += 1;
                    }
                }
            }
            if deadline_hit() {
                if diag {
                    eprintln!(
                        "[pgd-vjp] deadline at wave {restart_base} step {step} ({:.1}s): steps={steps_done} avg fwd={:.1}ms vjp={:.1}ms best_joint={best_joint:.5}",
                        attack_start.elapsed().as_secs_f64(),
                        fwd_ms / steps_done.max(1) as f64,
                        vjp_ms / steps_done.max(1) as f64
                    );
                }
                return Ok(Completed(None));
            }

            // Batched template forward: every restart's ReLU masks (fold order)
            // + network outputs, one rayon-parallel CPU pass.
            let t0 = Instant::now();
            let points: Vec<Vec<f32>> = xs.iter().map(|x| x.iter().copied().collect()).collect();
            let (masks, outputs) = match plan.forward_masks(&points) {
                Ok(v) => v,
                Err(e) => {
                    if diag {
                        eprintln!("[pgd-vjp] forward failed ({e}); sequential fallback");
                    }
                    return Ok(FallbackToSequential);
                }
            };

            // Counterexample screen + per-restart joint-margin spec rows.
            let mut spec_rows: Vec<f32> = Vec::with_capacity(k * plan.output_dim());
            // Diag-only: the per-conjunct margins of this step's best lane —
            // tells "one stubborn conjunct" from "several binding surfaces".
            let mut diag_best: Option<(f32, Vec<f32>)> = None;
            for kk in 0..k {
                let out_arr =
                    ArrayD::from_shape_vec(IxDyn(&[plan.output_dim()]), outputs[kk].clone())?;
                // #postbab-seed: track/export the best-margin point (guidance-only;
                // see `best_margin_export`). O(conjuncts) per restart per step —
                // noise next to the batched forward that produced the outputs.
                {
                    let joint = joint_hinge_loss(&conjunct_targets, &out_arr);
                    if lane_exploit[kk] {
                        super::super::best_margin_export::record_best_margin_candidate_exploit(
                            joint, &xs[kk],
                        );
                    } else {
                        super::super::best_margin_export::record_best_margin_candidate(
                            joint, &xs[kk],
                        );
                    }
                    if joint > best_joint {
                        best_joint = joint;
                        best_x = Some(xs[kk].clone());
                    }
                    // #lane-recycle bookkeeping.
                    if joint > lane_best[kk] {
                        lane_best[kk] = joint;
                    }
                    if diag && diag_best.as_ref().is_none_or(|(j, _)| joint > *j) {
                        diag_best = Some((
                            joint,
                            conjunct_targets
                                .iter()
                                .map(|t| t.margin(&out_arr))
                                .collect(),
                        ));
                    }
                }
                if super::check_unsafe_counterexample(&out_arr, constraints) {
                    let ctx = format!("vjp-batched restart {}, step {step}", restart_base + kk);
                    if let Some(pair) =
                        revalidate_graph_counterexample(graph, xs[kk].clone(), constraints, &ctx)?
                    {
                        if diag {
                            eprintln!(
                                "[pgd-vjp] CE FOUND at {ctx} elapsed={:.1}s",
                                attack_start.elapsed().as_secs_f64()
                            );
                        }
                        emit_graph_pgd_status(
                            json,
                            format_args!("  Graph PGD found counterexample at {ctx}!"),
                        );
                        return Ok(Completed(Some(pair)));
                    }
                }
                // #lane-recycle: a lane still in its ODI re-init phase ascends
                // its diversification direction, not the joint margin.
                if lane_odi_left[kk] > 0 {
                    spec_rows.extend_from_slice(
                        &odi_rows[kk * plan.output_dim()..(kk + 1) * plan.output_dim()],
                    );
                    continue;
                }
                if gama_lambda0.is_some() && gama_p_refs[kk].is_none() {
                    gama_p_refs[kk] = gama_reference_softmax(&out_arr);
                }
                // #tau-hinge spec row: conjuncts with margin < τ_lane contribute;
                // τ_lane = τ on odd lanes (hold-the-boundary ensemble), 0 on even
                // lanes (== `joint_unsat_spec_row`, the plain hinge). Fallback:
                // worst conjunct row when nothing is below τ (mirrors the plain
                // hinge's saturated case).
                let tau_kk = if kk % 2 == 1 { tau } else { 0.0 };
                let mut row = Array2::<f32>::zeros((1, plan.output_dim()));
                let mut active = 0usize;
                let mut worst: Option<(f32, &GraphPgdTarget)> = None;
                for t in &conjunct_targets {
                    let m = t.margin(&out_arr);
                    if worst.is_none_or(|(wm, _)| m < wm) {
                        worst = Some((m, t));
                    }
                    if m < tau_kk {
                        row += &t.to_spec_row(plan.output_dim());
                        active += 1;
                    }
                }
                if active == 0 {
                    let Some((_, t)) = worst else {
                        return Ok(FallbackToSequential);
                    };
                    row = t.to_spec_row(plan.output_dim());
                }
                if let (Some(lambda0), Some(p_ref)) = (gama_lambda0, gama_p_refs[kk].as_deref()) {
                    let lambda = gama_lambda_at(lambda0, gama_steps[kk], gama_lin);
                    row = add_gama_guidance_to_spec_row(row, &out_arr, p_ref, lambda);
                }
                spec_rows.extend(row.row(0).iter().copied());
            }
            let t1 = Instant::now();

            // ONE wide GPU pass: all K exact joint-margin gradients.
            let grads = match plan.gpu_vjp(gpu, &masks, &spec_rows) {
                Ok(g) => g,
                Err(e) => {
                    // Width fail-safe: if the very first wide pass of the attack
                    // does not fit this adapter (binding-limit validation), HALVE
                    // the wave width and rebuild the wave — never pay the
                    // sequential-loop cliff for a capacity miss. Mid-attack
                    // failures (device hiccup) keep the sequential fallback.
                    if steps_done == 0 && restart_base == 0 && width > 1 {
                        width = (width / 2).max(1);
                        if diag {
                            eprintln!(
                                "[pgd-vjp] wide VJP failed at K={k} ({e}); retrying wave at K={width}"
                            );
                        }
                        continue 'waves;
                    }
                    if diag {
                        eprintln!("[pgd-vjp] wide VJP failed ({e}); sequential fallback");
                    }
                    return Ok(FallbackToSequential);
                }
            };
            if grads.len() != k {
                return Ok(FallbackToSequential);
            }
            let t2 = Instant::now();
            fwd_ms += (t1 - t0).as_secs_f64() * 1e3;
            vjp_ms += (t2 - t1).as_secs_f64() * 1e3;
            steps_done += 1;

            // K signed-gradient/Adam ascent steps with box projection.
            for kk in 0..k {
                // #lane-recycle ODI re-init step: full-radius signed ascent of the
                // lane's diversification direction (same rule as the wave ODI).
                if lane_odi_left[kk] > 0 {
                    lane_odi_left[kk] -= 1;
                    let lower = input.lower();
                    let upper = input.upper();
                    for ((xi, &g), (lo, hi)) in xs[kk]
                        .iter_mut()
                        .zip(grads[kk].iter())
                        .zip(lower.iter().zip(upper.iter()))
                    {
                        let s = eta * 0.5 * (hi - lo) * g.signum();
                        *xi = (*xi + s).clamp(*lo, *hi);
                    }
                    continue;
                }
                let grad = ArrayD::from_shape_vec(IxDyn(xs[kk].shape()), grads[kk].clone())?;
                let previous_x = pgd_config.restart_when_stuck.then(|| xs[kk].clone());
                xs[kk] = states[kk].step(&grad, &xs[kk], input, true);
                gama_steps[kk] = gama_steps[kk].saturating_add(1);
                // Restart-when-stuck (mirrors the sequential loop): a projection
                // fixed point resamples this lane instead of running it dead.
                if let Some(prev) = previous_x {
                    if prev.iter().zip(xs[kk].iter()).all(|(&a, &b)| a == b) {
                        xs[kk] = sample_uniform_point(input, &mut rngs[kk]);
                        states[kk].reset();
                        lane_exploit[kk] = false;
                        reset_gama_lane(&mut gama_p_refs[kk], &mut gama_steps[kk]);
                    }
                }
            }

            if diag && (step + 1) % 100 == 0 {
                eprintln!(
                    "[pgd-vjp] wave {restart_base} step {} K={k}: avg fwd={:.1}ms vjp={:.1}ms step_total={:.1}ms best_joint={best_joint:.5} recycled={lanes_recycled} exploited={lanes_exploited} (0 => CE)",
                    step + 1,
                    fwd_ms / steps_done as f64,
                    vjp_ms / steps_done as f64,
                    (fwd_ms + vjp_ms) / steps_done as f64
                );
                if let Some((j, margins)) = &diag_best {
                    let s: Vec<String> = margins.iter().map(|m| format!("{m:.2e}")).collect();
                    eprintln!(
                        "[pgd-vjp]   step-best lane joint={j:.2e} margins=[{}]",
                        s.join(", ")
                    );
                }
            }
        }

        // Post-wave final check at the last points (mirrors the sequential
        // final-check): one more batched forward, screen, revalidate.
        let points: Vec<Vec<f32>> = xs.iter().map(|x| x.iter().copied().collect()).collect();
        if let Ok((_m, outputs)) = plan.forward_masks(&points) {
            for kk in 0..k {
                let out_arr =
                    ArrayD::from_shape_vec(IxDyn(&[plan.output_dim()]), outputs[kk].clone())?;
                // #postbab-seed: the post-wave points (after the final step update)
                // are the deepest of the wave — record them too.
                let joint = joint_hinge_loss(&conjunct_targets, &out_arr);
                if lane_exploit[kk] {
                    super::super::best_margin_export::record_best_margin_candidate_exploit(
                        joint, &xs[kk],
                    );
                } else {
                    super::super::best_margin_export::record_best_margin_candidate(joint, &xs[kk]);
                }
                if super::check_unsafe_counterexample(&out_arr, constraints) {
                    let ctx = format!("vjp-batched restart {} (final check)", restart_base + kk);
                    if let Some(pair) =
                        revalidate_graph_counterexample(graph, xs[kk].clone(), constraints, &ctx)?
                    {
                        emit_graph_pgd_status(
                            json,
                            format_args!("  Graph PGD found counterexample at {ctx}!"),
                        );
                        return Ok(Completed(Some(pair)));
                    }
                }
            }
        }

        restart_base += k;
    }

    if diag {
        eprintln!(
            "[pgd-vjp] exhausted ({:.1}s): steps={steps_done} avg fwd={:.1}ms vjp={:.1}ms best_joint={best_joint:.5}",
            attack_start.elapsed().as_secs_f64(),
            fwd_ms / steps_done.max(1) as f64,
            vjp_ms / steps_done.max(1) as f64
        );
    }
    emit_graph_pgd_status(
        json,
        format_args!(
            "  Graph PGD (batched exact-VJP): no counterexample found. ({:.2}s)",
            attack_start.elapsed().as_secs_f64()
        ),
    );
    Ok(Completed(None))
}
