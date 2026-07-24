// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Root evaluation and root-domain assembly for multi-objective graph BaB.
//!
//! Keeps the top-level coordinator in `verify.rs` focused on queue flow while
//! preserving the existing root semantics, including the `#3813` cached-lA
//! warm-start path.

use ny_core::{GemmEngine, Result};
use ny_tensor::{next_down_f32, BoundedTensor};
use tracing::{debug, info};

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::beta_crown::state::GraphDomainAlphaState;
use crate::network::SpecCrownRequest;
use crate::{BetaCrownConfig, GraphNetwork};

use super::super::super::BetaCrownVerifier;
use super::super::shared::init::{
    compute_graph_bab_bootstrap, compute_graph_root_output_bounds, GraphBabBootstrap,
};
use super::super::shared::setup::{
    build_graph_bab_setup, build_graph_cut_pool, build_initial_node_bounds_arc,
    build_root_alpha_state,
};
use super::dd_zono_root::{intersect_objective_bounds, run_dd_zono_root};
use super::per_disjunct::build_per_disjunct_alphas;
use super::post_c_survivor::{
    build_post_c_survivor_plan, run_post_c_survivor_candidate, PostCSurvivorAccepted,
};
use super::shared::{build_spec_matrix, spec_bounds_to_vec};

pub(super) enum MultiObjectiveRootOutcome {
    Finished(Box<BetaCrownResult>),
    Continue(Box<MultiObjectiveRootState>),
}

#[must_use]
pub(super) struct MultiObjectiveRootState {
    pub(super) root_domain: MultiObjectiveGraphBabDomain,
    pub(super) relu_nodes: Vec<String>,
    pub(super) cut_pool: GraphCutPool,
    pub(super) use_batched_gpu: bool,
}

#[must_use]
pub(super) struct MultiObjectiveRootRequest<'a> {
    pub(super) verifier: &'a BetaCrownVerifier,
    pub(super) graph: &'a GraphNetwork,
    pub(super) input: &'a BoundedTensor,
    pub(super) objectives: &'a [Vec<f32>],
    pub(super) thresholds: &'a [f32],
    pub(super) engine: Option<&'a dyn GemmEngine>,
    pub(super) conjunctive: bool,
    pub(super) deadline: Option<std::time::Instant>,
}

#[must_use]
pub(super) struct RootObjectiveEvaluation {
    pub(super) initial_output: BoundedTensor,
    pub(super) initial_obj_bounds: Vec<(f32, f32)>,
    pub(super) root_spec_cache: Option<CachedLinearBounds>,
    /// Full-objective indices represented by `root_spec_cache`'s rows.
    ///
    /// The default/full-spec path stores `0..objectives.len()`.  The dark
    /// `NY_ROOT_SPEC_PRUNE=1` path stores only still-active rows; attachment
    /// expands them back to a full `Vec<Option<_>>`, leaving certified-pruned
    /// objectives as `None` so a cache can never be applied to the wrong row.
    pub(super) root_spec_cache_active_indices: Vec<usize>,
}

/// A sound pre-CROWN compression plan for disjunctive root specifications.
///
/// `pre_bounds` comes exclusively from the bootstrap's certified full-output
/// enclosure.  Rows whose lower endpoint is already strictly above their
/// threshold need no optimized specification backward.  The output enclosure
/// remains valid even when a later root-tightening pass only shrank intermediate
/// boxes, so it is also the sound result source for an all-pruned plan.
#[derive(Debug)]
struct RootSpecPrunePlan {
    bootstrap_output: BoundedTensor,
    pre_bounds: Vec<(f32, f32)>,
    active_indices: Vec<usize>,
    active_spec_matrix: Option<ndarray::Array2<f32>>,
}

fn root_spec_prune_enabled() -> bool {
    std::env::var("NY_ROOT_SPEC_PRUNE").ok().as_deref() == Some("1")
}

/// Authority predicate for removing a row before optimized spec CROWN.
/// Every endpoint must be an ordinary, ordered finite enclosure; malformed or
/// unbounded intervals always stay active even if their lower endpoint alone
/// would compare above the threshold.
fn root_prebound_certifies(lower: f32, upper: f32, threshold: f32) -> bool {
    root_interval_is_finite_ordered(lower, upper) && threshold.is_finite() && lower > threshold
}

fn root_interval_is_finite_ordered(lower: f32, upper: f32) -> bool {
    lower.is_finite() && upper.is_finite() && lower <= upper
}

/// Resolve the exact output entry retained by the bootstrap.  Any graph-order
/// or lookup problem declines the optimization so the caller runs the historic
/// full-spec request unchanged.
fn bootstrap_output_bounds<'a>(
    graph: &GraphNetwork,
    bootstrap: &'a GraphBabBootstrap,
) -> Option<&'a BoundedTensor> {
    let output_name = if graph.output_name().is_empty() {
        graph.exec_order().ok()?.last()?.clone()
    } else {
        graph.output_name().to_string()
    };
    bootstrap.initial_node_bounds.get(&output_name)
}

/// Build a root-spec compression plan, returning `None` to fail closed to the
/// existing full request.  In particular, conjunctive semantics are never
/// compressed: their root stopping rule is "any row verified", unlike the
/// disjunctive all-rows rule this optimization targets.
fn build_root_spec_prune_plan(
    enabled: bool,
    conjunctive: bool,
    output: &BoundedTensor,
    full_spec_matrix: &ndarray::Array2<f32>,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
) -> Option<RootSpecPrunePlan> {
    if !enabled || conjunctive || objectives.is_empty() || objectives.len() != thresholds.len() {
        return None;
    }

    let output_dim = output.flatten().len();
    if full_spec_matrix.nrows() != objectives.len()
        || full_spec_matrix.ncols() != output_dim
        || objectives.iter().any(|objective| {
            objective.len() != output_dim || objective.iter().any(|value| !value.is_finite())
        })
        || thresholds.iter().any(|threshold| !threshold.is_finite())
    {
        return None;
    }

    let pre_bounds = BetaCrownVerifier::objective_bounds_multi(output, objectives).ok()?;
    if pre_bounds.len() != objectives.len() {
        return None;
    }

    // Strictly mirror the verifier's authority test (`lower > threshold`) while
    // requiring a finite, ordered enclosure. Malformed/degenerate arithmetic
    // must never create a shortcut.
    let active_indices: Vec<usize> = pre_bounds
        .iter()
        .zip(thresholds)
        .enumerate()
        .filter_map(|(idx, (&(lower, upper), &threshold))| {
            (!root_prebound_certifies(lower, upper, threshold)).then_some(idx)
        })
        .collect();

    let active_spec_matrix = if active_indices.is_empty() {
        None
    } else {
        let mut active = ndarray::Array2::zeros((active_indices.len(), output_dim));
        for (active_row, &full_row) in active_indices.iter().enumerate() {
            active
                .row_mut(active_row)
                .assign(&full_spec_matrix.row(full_row));
        }
        Some(active)
    };

    Some(RootSpecPrunePlan {
        bootstrap_output: output.clone(),
        pre_bounds,
        active_indices,
        active_spec_matrix,
    })
}

/// Restore active CROWN rows into the certified full pre-bound vector.
///
/// Both enclosures are independently sound, so take their intersection when it
/// is well formed.  This prevents compression from weakening an active row when
/// the optimized spec candidate happens to be looser than the bootstrap output
/// projection. A malformed or disjoint compact candidate rejects the whole
/// compact result so its cache cannot gain downstream authority; the caller
/// retries the full legacy request. A finite compact row may replace a malformed
/// bootstrap row because the former is an independent certified enclosure.
fn merge_root_spec_pruned_bounds(
    plan: &RootSpecPrunePlan,
    active_bounds: Vec<(f32, f32)>,
) -> Option<Vec<(f32, f32)>> {
    if active_bounds.len() != plan.active_indices.len() {
        return None;
    }
    let mut seen = vec![false; plan.pre_bounds.len()];
    let mut merged = plan.pre_bounds.clone();
    for (&full_idx, (active_lower, active_upper)) in plan.active_indices.iter().zip(active_bounds) {
        if full_idx >= merged.len() || seen[full_idx] {
            return None;
        }
        seen[full_idx] = true;
        let pre = merged[full_idx];
        let active = (active_lower, active_upper);
        match (
            root_interval_is_finite_ordered(pre.0, pre.1),
            root_interval_is_finite_ordered(active.0, active.1),
        ) {
            (true, true) => {
                let intersection = (pre.0.max(active.0), pre.1.min(active.1));
                if root_interval_is_finite_ordered(intersection.0, intersection.1) {
                    merged[full_idx] = intersection;
                } else {
                    // The compact run's bounds and bootstrap enclosure must
                    // describe the same reachable values. Reject the entire
                    // compact result (including its cache) on disagreement.
                    return None;
                }
            }
            // A malformed compact result also invalidates its captured linear
            // cache as downstream authority. Retry the full legacy request.
            (true, false) => return None,
            (false, true) => {
                // The row was deliberately kept active because its bootstrap
                // interval could not certify anything. A fresh finite CROWN
                // enclosure is independently sound and can replace it.
                merged[full_idx] = active;
            }
            (false, false) => return None,
        }
    }
    merged
        .iter()
        .all(|&(lower, upper)| root_interval_is_finite_ordered(lower, upper))
        .then_some(merged)
}

pub(super) fn validate_multi_objective_inputs(
    objectives: &[Vec<f32>],
    thresholds: &[f32],
) -> Result<()> {
    if objectives.is_empty() {
        return Err(ny_core::NyError::InvalidSpec(
            "empty objectives in multi-objective verification — nothing to verify".to_string(),
        ));
    }
    if objectives.len() != thresholds.len() {
        return Err(ny_core::NyError::InvalidSpec(format!(
            "objectives/thresholds length mismatch: {} objectives vs {} thresholds (#3383)",
            objectives.len(),
            thresholds.len()
        )));
    }
    Ok(())
}

pub(super) fn evaluate_root(
    request: MultiObjectiveRootRequest<'_>,
    lifecycle: &mut GraphBabLifecycle,
) -> Result<MultiObjectiveRootOutcome> {
    let MultiObjectiveRootRequest {
        verifier,
        graph,
        input,
        objectives,
        thresholds,
        engine,
        conjunctive,
        deadline,
    } = request;
    // Warmup cap (#2206 Packet C, #4095): initial bounds get at most
    // `initial_bounds_fraction` of the BaB timeout. Mirrors core.rs pattern.
    //
    // When a wall-clock deadline is provided (#4321), derive the effective
    // timeout from remaining time instead of the configured timeout.
    let pgd_frac = verifier
        .config
        .phase_budget
        .post_bab_pgd_fraction
        .clamp(0.0, 0.5);
    // #cora-double-reserve: Some(deadline) already carries the ledger's one-time
    // post_bab_pgd_fraction reservation — do not scale it again (see verify.rs).
    let bab_timeout = match deadline {
        Some(dl) => dl.saturating_duration_since(lifecycle.start_time),
        None => verifier.config.timeout.mul_f32(1.0 - pgd_frac),
    };
    // The mandatory foundational node-bounds sweep must reach every node, so it
    // gets the full global deadline (not the warmup fraction) — capping it choked
    // conv-heavy DAGs (yolo, tinyimagenet) into "deadline exceeded before node
    // 'Conv_0'" with most of the budget unused (#4321).
    let bab_deadline = lifecycle.start_time + bab_timeout;
    let initial_deadline = Some(bab_deadline);
    // #w4-root-alpha-opt WARMUP CAP: the alpha warmup otherwise runs to the
    // full BaB budget (the initial_bounds_fraction knob never capped this
    // path — W4-2 finding), leaving the root pass a grace slice smaller than
    // one measured forward-map rebuild (~22s grace vs ~25s rebuild at 95s,
    // measured), so the root alpha OPTIMIZER — the root-relaxation lever —
    // could never fire. When the lever is armed and the CLI attack-phase
    // warmer measured the fixed-map cost, reserve optimizer+rebuild budget
    // out of the warmup. The worst stragglers are BETA-INSENSITIVE (W4-6),
    // so trading warmup iterations for root tightness is the measured-win
    // direction. Applies only where the fixed forward map is warm (image
    // conv DAGs); everything else keeps the status quo. Sound: deadlines
    // only schedule work.
    let initial_deadline = {
        let alpha_lever_armed = graph.has_conv_layers()
            && !matches!(
                std::env::var("NY_SPEC_ROOT_ALPHA").ok().as_deref(),
                Some("0")
            );
        let reserved_deadline = if alpha_lever_armed {
            graph
                .forward_linear_fixed_pass_cost(input)
                .zip(deadline)
                .and_then(|(cost, global)| {
                    // Rebuild (~= fixed cost) with margin + optimizer sweeps
                    // + the root spec pass's own candidates.
                    let reserve = cost.mul_f64(1.3) + std::time::Duration::from_secs(10);
                    global.checked_sub(reserve)
                })
        } else {
            None
        };
        match (initial_deadline, reserved_deadline) {
            (Some(base), Some(reserved)) => {
                // Warmup floor: keep enough for the reference-bounds pass and
                // a few alpha iterations (the GPU root candidate still wants
                // a sane alpha state).
                let floor = std::time::Instant::now() + std::time::Duration::from_secs(10);
                Some(base.min(reserved.max(floor)))
            }
            (base, _) => base,
        }
    };
    // Certified sparse-input double-double zonotope (#dd-zonotope, dark
    // `NY_DD_ZONOTOPE=1`, default-OFF). Runs BEFORE the bootstrap because on
    // the category it targets (vggnet16_2022) the existing root pass consumes
    // the entire budget — an intersect placed after `compute_root_objective_bounds`
    // would never be reached. Gate-off short-circuits before any allocation, so
    // the arm is byte-identical when unset. See `dd_zono_root` for the full
    // soundness contract: conjunctive detector, self-policing precision gate,
    // fail-closed refusals, and INTERSECT-never-replace publication.
    let dd_zono = run_dd_zono_root(graph, input, objectives, deadline);
    if let Some(result) = dd_zono.as_ref() {
        // Certified-at-root fast exit. Requiring EVERY objective to clear its
        // threshold is sufficient for both the conjunctive (any) and the
        // disjunctive (all) verdict rules, so no rule-specific reasoning is
        // duplicated here.
        //
        // The three-way length equality is asserted rather than assumed. A
        // `zip` between a short `thresholds` and a long margin list would
        // silently check only the common prefix and then report `all` — i.e. it
        // would publish `Verified` for objectives it never examined. The
        // caller already rejects a mismatch (#3383) and `evaluate_objectives`
        // emits exactly one entry per objective, so this can only fire if one
        // of those invariants is later broken; it is a verdict site, so it
        // fails closed instead of trusting them.
        let safety = crate::dd_zonotope::DdZonoConfig::from_env().safety_factor;
        let lengths_agree = result.margin.lower.len() == thresholds.len()
            && result.margin.lower.len() == objectives.len();
        // Narrow OUTWARD (toward -inf) on the f64 -> f32 cast: a nearest-mode
        // cast could round a certified lower bound UP across the threshold.
        let all_verified = lengths_agree
            && thresholds.iter().enumerate().all(|(i, &t)| {
                let lo = result.margin.lower_with_safety(i, safety);
                lo.is_finite() && next_down_f32(lo as f32) > t
            });
        if all_verified {
            if let Some(output) = result.output.clone() {
                info!(
                    "#dd-zonotope: all {} objective(s) certified at the root by the \
                     double-double zonotope ({} generators, {:.1}s) — property safe",
                    objectives.len(),
                    result.margin.n_generators,
                    result.margin.wall.as_secs_f32()
                );
                lifecycle.domains_explored = 1;
                lifecycle.domains_verified = 1;
                return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
                    lifecycle.build_result_with_bounds(BabVerificationStatus::Verified, output),
                )));
            }
        }
    }

    // Root/output-bound passes now check the wall-clock deadline between nodes
    // (#4321). When one of them aborts because the deadline passed mid-phase, that
    // surfaces as DeadlineExceeded; convert it here into a graceful Timeout verdict
    // so the CLI emits a valid JSON result instead of being killed externally.
    // Sound: a Timeout never claims Verified.
    let mut bootstrap =
        match compute_graph_bab_bootstrap(graph, input, &verifier.config, engine, initial_deadline)
        {
            Ok(bootstrap) => bootstrap,
            Err(ny_core::NyError::DeadlineExceeded(_)) => {
                return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
                    lifecycle.timeout_result(),
                )));
            }
            Err(e) => return Err(e),
        };

    // STABILIZE-AND-FIX (#stabilize, dark `NY_STABILIZE=<budget_secs>`, default
    // OFF ⇒ byte-identical): spend a bounded root budget proving individual
    // unstable ReLU neurons stable (per-neuron α-CROWN backward tighten,
    // intersect-only into the stored pre-activation entry) and FIX the proven
    // ones. The tightened STORED entry (l≥0 / u≤0) is itself the proof artifact
    // that every relaxation consumer (constraints/backward/relu.rs, the GPU
    // Activation extraction) and every branching scan
    // (find_unstable_graph_neurons_*) reads, so fixed neurons lose their
    // triangle looseness and branching candidacy on EVERY descendant domain
    // with zero new trust surface. Runs BEFORE the MIP stash, the root
    // objective pass, and `build_graph_bab_setup`, so all of them inherit the
    // fixes. See `shared/stabilize.rs` for the loop + soundness invariants.
    if let Some(stabilize_report) = super::super::shared::stabilize::stabilize_and_fix_from_env(
        graph,
        input,
        objectives,
        engine,
        deadline,
        bootstrap.root_alpha_state.as_ref(),
        &mut bootstrap.initial_node_bounds,
    ) {
        info!(
            "stabilize-and-fix: {} round(s), {} neuron-fix(es)",
            stabilize_report.rounds,
            stabilize_report.fixed.len()
        );
    }

    // FC-head pre-activation tightening (#cifar100-fchead): the α-CROWN warmup
    // returns the forward-linear / IBP reference intermediate bounds unchanged
    // when fix_interm_bounds is set (the default deep-conv path). On deep conv
    // ResNets the dominant residual ReLU-relaxation slack at the output is
    // concentrated in the *dense* head pre-activation (cifar100 `Gemm_56`:
    // 2048→100, ~51/100 unstable, mean width 3.7 vs a <2%-unstable, tight conv
    // stack). Refine just those dense-fed ReLU pre-activations with a per-target
    // α-CROWN backward (reusing the warmup's optimized slopes) BEFORE the root
    // objective pass and the per-domain BaB setup consume the bounds, so both
    // benefit. The warmup deadline is already spent here, so this gets its own
    // small grace slice out of the remaining global budget (mirrors the root
    // spec pass grace). SOUND: `tighten_fc_head_preactivations` intersect-only —
    // it can only shrink a bound, never widen it, and every stored bound still
    // encloses the true reachable pre-activation set. Deadlines only schedule
    // work; on expiry each target keeps its sound reference bound.
    //
    // Gated OPT-IN (NY_FCHEAD_TIGHTEN=1): measured to close ~60% of the cifar100
    // resnet root-margin gap (helps deep classifiers with a loose dense FC head)
    // but adds ~1-2s at the BaB root on ANY net with a dense-fed unstable ReLU,
    // where it buys nothing — so it is OFF by default and enabled only where it
    // pays (the cifar100/tinyimagenet path / an explicit opt-in). It does not flip
    // cifar100 alone (the residual gap needs conv-stack tightening too).
    if std::env::var("NY_FCHEAD_TIGHTEN").ok().as_deref() == Some("1") {
        if let Some(alpha) = bootstrap.root_alpha_state.as_ref() {
            let now = std::time::Instant::now();
            // Grace cap (env-tunable for measurement; default 12s). The single
            // dense-head backward is ~1-2s on the sound GPU resnet path; cap it
            // and leave the bulk of the remaining budget for the root spec pass
            // and BaB. Only fires while the global wall-clock still has room.
            let grace_cap = std::env::var("NY_FCHEAD_GRACE_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(12);
            let global_remaining = deadline
                .map(|g| g.saturating_duration_since(now))
                .unwrap_or_else(|| std::time::Duration::from_secs(grace_cap));
            // Reserve at least half the remaining budget for root-spec + BaB.
            let slice =
                std::time::Duration::from_secs(grace_cap).min(global_remaining.mul_f32(0.5));
            if slice >= std::time::Duration::from_secs(2) {
                let fc_deadline = Some(now + slice);
                if let Ok(exec_order) = graph.exec_order() {
                    let exec_order = exec_order.to_vec();
                    graph.tighten_fc_head_preactivations(
                        input,
                        &exec_order,
                        alpha,
                        engine,
                        fc_deadline,
                        &mut bootstrap.initial_node_bounds,
                    );
                }
            }
        }
    }

    // Root intermediate-bound α tightening (#root-interm-alpha, dark
    // `NY_ROOT_INTERM_ALPHA=1`): the BROAD counterpart to NY_FCHEAD_TIGHTEN.
    // With `fix_interm_bounds=true` the α-CROWN warmup returns the heuristic-α
    // reference intermediate bounds unchanged — the α it optimized for the
    // output margin is never applied to the intermediate pre-activations, so
    // every crossing-ReLU triangle along the conv stack + FC head is relaxed
    // with heuristic (not optimized) slopes. auto_LiRPA instead optimizes the α
    // used to compute EACH intermediate bound. This pass recomputes ALL root
    // ReLU pre-activations (conv-stack BN/Add outputs AND the dense `Gemm_56`
    // head) with the warmup's OPTIMIZED α, BEFORE the root objective pass and
    // per-domain BaB setup consume the bounds (children inherit them as their
    // forward base), so the whole tree benefits. It measures whether the one
    // untested lever — optimized-α root intermediate bounds — moves the cifar100
    // worst-subdomain plateau. SOUND: `tighten_all_relu_preactivations` is
    // intersect-only (shrink a bound, never widen; α only tunes the ReLU lower
    // slope within the sound triangle); on deadline each target keeps its sound
    // reference bound. Default-OFF ⇒ byte-identical (no bound is ever touched).
    if std::env::var("NY_ROOT_INTERM_ALPHA").ok().as_deref() == Some("1") {
        eprintln!(
            "[root-interm-alpha] gate ON, root_alpha_state={}",
            if bootstrap.root_alpha_state.is_some() {
                "Some"
            } else {
                "None"
            }
        );
        if let Some(alpha) = bootstrap.root_alpha_state.as_ref() {
            let now = std::time::Instant::now();
            // Grace cap (env-tunable for measurement; default 120s — this pass is
            // the full O(L²) per-node root sweep over ~20 ReLU pre-activations on
            // a deep conv ResNet, far heavier than the single FC-head backward).
            let grace_cap = std::env::var("NY_ROOT_INTERM_ALPHA_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(120);
            let global_remaining = deadline
                .map(|g| g.saturating_duration_since(now))
                .unwrap_or_else(|| std::time::Duration::from_secs(grace_cap));
            // Reserve at least half the remaining budget for root-spec + BaB.
            let slice =
                std::time::Duration::from_secs(grace_cap).min(global_remaining.mul_f32(0.5));
            if slice >= std::time::Duration::from_secs(2) {
                let ria_deadline = Some(now + slice);
                if let Ok(exec_order) = graph.exec_order() {
                    let exec_order = exec_order.to_vec();
                    graph.tighten_all_relu_preactivations(
                        input,
                        &exec_order,
                        alpha,
                        engine,
                        ria_deadline,
                        &mut bootstrap.initial_node_bounds,
                    );
                }
            }
        }
    }

    // NY_ROOT_INTERM_ALPHA block's closing brace (before the next lever's comment).

    // Root JOINT per-target intermediate-bound α pass (#root-joint-interm-alpha,
    // dark `NY_ROOT_JOINT_INTERM_ALPHA=1`, default-OFF ⇒ byte-identical). The
    // auto_LiRPA `fix_intermediate_layer_bounds=False` root pass — the ONE α-family
    // variant never directly measured (docs/ROOT_JOINT_INTERM_ALPHA_PLAN.md §B).
    // Unlike the NY_ROOT_INTERM_ALPHA block above (which BORROWS the output-margin
    // warmup α and applies it ONCE per target — measured ZERO), this HOISTS the
    // per-target α′ ascent to the root: for each scoped target layer L it seeds
    // identity AT L's crossing rows and lets the gradient flow THROUGH L's own
    // intermediate-bound computation via the on-device joint adjoint
    // (`crown_joint_alpha_gradient_resident`), Adam-ascends the below-L α, scores
    // every iterate with the certified sound fold, and writes the element-wise
    // best box SHRINK-ONLY into `bootstrap.initial_node_bounds` HERE — before
    // `compute_root_objective_bounds` and the BaB-setup Arc — so the whole frozen
    // tree inherits it by pointer. Scope knobs:
    // NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM (default 2048 ⇒ head + last residual
    // block), _LAYERS (comma-list of ReLU node names), _ITERS (default 100),
    // _LR (default 0.1), _SECS grace cap (default 30), _MAX_SEL (identity-seed
    // row cap, default 512). SOUND: every kept bound comes from the sound fold;
    // shrink-only intersect with per-element union fallback; fail-closed on any
    // refusal. Default-OFF ⇒ no bound is ever touched.
    if std::env::var("NY_ROOT_JOINT_INTERM_ALPHA").ok().as_deref() == Some("1") {
        let now = std::time::Instant::now();
        let grace_cap = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_SECS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(30);
        let global_remaining = deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or_else(|| std::time::Duration::from_secs(grace_cap));
        // #root-joint-admission (promotion prerequisite): the pass's measured
        // cost is ~+18s bab-start — free at the 700s research tier, FATAL on
        // scored 100s rows banked at 90-96s runtime (the tax alone flips them
        // to timeout). Admission floor: run only when the remaining budget can
        // absorb the slice with room for BaB — default 240s, override with
        // NY_ROOT_JOINT_MIN_REMAINING_SECS (0 disables the floor; research
        // probes at 400-900s budgets are unaffected). Below the floor the
        // gate-ON path is byte-identical to gate-OFF (skip, no map change).
        let min_remaining = std::env::var("NY_ROOT_JOINT_MIN_REMAINING_SECS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(240);
        let admitted = !(deadline.is_some()
            && global_remaining < std::time::Duration::from_secs(min_remaining));
        if !admitted {
            eprintln!(
                "[root-joint-interm-alpha] admission floor: remaining {:.1}s < {min_remaining}s — pass skipped (byte-identical)",
                global_remaining.as_secs_f32()
            );
        }
        // Reserve at least half the remaining budget for root-spec + BaB.
        let slice = std::time::Duration::from_secs(grace_cap).min(global_remaining.mul_f32(0.5));
        let targets =
            crate::beta_crown::engine::graph::propagation::batched::interm_refine::
                scoped_joint_alpha_targets(graph, &bootstrap.initial_node_bounds);
        eprintln!(
            "[root-joint-interm-alpha] gate ON, targets={} slice={:.1}s",
            targets.len(),
            slice.as_secs_f32()
        );
        // #boxlift phase mirror (dark, NY_PHASE_TELEMETRY=1, print-only): the
        // same summary as a `[phase]` line on the shared epoch clock, so the
        // pass shows up in phase logs alongside `[frontier]` frames. The bare
        // eprintln above is unchanged; gate-off skips the format entirely.
        if crate::phase_telemetry::phase_telemetry_enabled() {
            crate::phase_telemetry::phase_marker(&format!(
                "root-joint-interm start targets={} slice={:.1}s",
                targets.len(),
                slice.as_secs_f32()
            ));
        }
        if admitted && slice >= std::time::Duration::from_secs(2) && !targets.is_empty() {
            let iters = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_ITERS")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(100);
            let lr = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_LR")
                .ok()
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.1);
            let n_tightened =
                crate::beta_crown::engine::graph::propagation::batched::interm_refine::
                    root_joint_tighten_relu_preactivations(
                    graph,
                    input,
                    &targets,
                    engine,
                    Some(now + slice),
                    iters,
                    lr,
                    &mut bootstrap.initial_node_bounds,
                );
            eprintln!(
                "[root-joint-interm-alpha] done: {}/{} target(s) tightened",
                n_tightened,
                targets.len()
            );
            // #boxlift phase mirror (dark, print-only): pass outcome + wall on
            // the shared epoch clock; the eprintln above is unchanged.
            if crate::phase_telemetry::phase_telemetry_enabled() {
                crate::phase_telemetry::phase_marker(&format!(
                    "root-joint-interm done tightened={}/{} wall={:.1}s",
                    n_tightened,
                    targets.len(),
                    now.elapsed().as_secs_f32()
                ));
            }
        }
    }

    // Typed sparse crossing-row intermediate CROWN (#root-sparse-interm-crown).
    // Unlike the research joint-α seam above, this production-shaped pass runs
    // only the certified BASE sound fold (zero optimization iterations), selects
    // convolutional/residual ReLU pre-activations structurally, excludes the
    // separately-owned dense head, and caps dimensions, selected rows, targets,
    // and wall time before allocation. Tightened boxes are shrink-only and are
    // inherited by the root objective and every BaB child. Default-OFF unless a
    // measured typed preset or the sealed force-on A/B gate enables it.
    let root_sparse_interm_tightened_targets = if let Some(policy) =
        root_sparse_interm_crown_policy_from_env(&verifier.config)
    {
        let now = std::time::Instant::now();
        if let Some(pass_deadline) =
            bounded_root_crown_interm_deadline(now, deadline, policy.max_secs)
        {
            let targets =
                crate::beta_crown::engine::graph::propagation::batched::interm_refine::
                    scoped_sparse_crown_targets(
                        graph,
                        &bootstrap.initial_node_bounds,
                        policy.max_dim,
                        policy.max_rows,
                        policy.max_targets,
                    );
            eprintln!(
                "[root-sparse-interm-crown] targets={} max_dim={} max_rows={} max_targets={} budget={:.3}s",
                targets.len(),
                policy.max_dim,
                policy.max_rows,
                policy.max_targets,
                pass_deadline.saturating_duration_since(now).as_secs_f32(),
            );
            crate::beta_crown::engine::graph::propagation::batched::interm_refine::
                root_sparse_tighten_relu_preactivations(
                    graph,
                    input,
                    &targets,
                    engine,
                    Some(pass_deadline),
                    policy.max_rows,
                    &mut bootstrap.initial_node_bounds,
                )
        } else {
            eprintln!(
                "[root-sparse-interm-crown] no safe deadline slice remains; skipping (bounds unchanged)"
            );
            0
        }
    } else {
        0
    };

    // Root CROWN-backward intermediate-bound INTERSECT (#root-crown-interm). At
    // the ROOT (before BaB), compute a SOUND heuristic-α CROWN BACKWARD box to
    // the input eps-box and INTERSECT it SHRINK-ONLY into the frozen
    // `initial_node_bounds`:
    //   l_new = max(l_fwd, l_crown),  u_new = min(u_fwd, u_crown).
    // SOUNDNESS: both boxes are sound enclosures of the reachable pre-activation
    // set, so their intersection is a sound enclosure (never drops a real point)
    // and can ONLY tighten. The tightened bounds are written into
    // `bootstrap.initial_node_bounds` HERE — before `compute_root_objective_bounds`
    // AND before the BaB-setup Arc (`build_graph_bab_setup`, below, which Arc-wraps
    // this same map) — so the root objective and EVERY BaB subdomain inherit the
    // tighter bounds by pointer; the CROWN pass runs ONCE at the root, never per
    // child. Production selection is structural: dense-fed ReLU pre-activations
    // only, armed by the typed benchmark preset. FAIL-CLOSED: the pass has its
    // own deadline, and any expiry/non-finite/disjoint/shape mismatch keeps the
    // forward-linear reference. Legacy env force-on/off and layer selection
    // remain available for A/B and rollback.
    let root_crown_interm_tightened_elements = if let Some(policy) =
        root_crown_interm_policy_from_env(&verifier.config)
    {
        let now = std::time::Instant::now();
        if let Some(pass_deadline) =
            bounded_root_crown_interm_deadline(now, deadline, policy.max_secs)
        {
            run_root_crown_interm_tighten(
                graph,
                input,
                engine,
                &mut bootstrap,
                &policy,
                pass_deadline,
            )
        } else {
            eprintln!(
                "[root-crown-interm-tighten] no safe deadline slice remains; skipping (bounds unchanged)"
            );
            0
        }
    } else {
        0
    };
    // Graph-MIP stash (FIX 1, `docs/GRAPH_MIP_LEAF_SOLVER.md`): the relational
    // multi-objective lane computes its per-property bounds HERE (not at the
    // ny-cli per-constraint precompute), so this is where the Graph-MIP
    // escalation's reuse mailbox must be filled — otherwise the escalation
    // falls back to a deadline-truncated recompute whose LOOSE bounds inflate
    // the unstable-ReLU eligibility count. Disabled when whole-net Graph-MIP
    // is explicitly off or the category requests no MIP reservation; the leaf
    // oracle consumes child bounds directly and remains independent.
    crate::beta_crown::graph_mip_leaf::stash_root_bounds_for_mip(
        graph,
        input,
        &verifier.config.phase_budget,
        &bootstrap.initial_node_bounds,
    );
    // DIAGNOSTIC (NY_LPOPT_DUMP=<path>): dump the EXACT root state feeding every BaB
    // subdomain — the input eps-box + the full per-node pre-activation `[l,u]`
    // (`bootstrap.initial_node_bounds`, AFTER the optional CROWN-interm tighten) +
    // the ReLU→pre-activation node-name map. This is the ground-truth data needed to
    // rebuild NY's OWN triangle-relaxation LP off-line (p*_LP) and check whether NY's
    // α/β-CROWN reaches its own relaxation's LP optimum. Read-only / print-only;
    // never mutates the bootstrap or any verdict. Default-OFF ⇒ byte-identical.
    if let Ok(path) = std::env::var("NY_LPOPT_DUMP") {
        run_lpopt_dump(graph, input, &bootstrap, &path);
    }
    // #phase-telemetry (dark, NY_PHASE_TELEMETRY=1, print-only): bracket the
    // root objective evaluation. A start without an end in a log means the
    // phase timed out (the DeadlineExceeded arm) or errored.
    crate::phase_telemetry::phase_marker("root-objective start");
    let RootObjectiveEvaluation {
        initial_output,
        mut initial_obj_bounds,
        root_spec_cache,
        root_spec_cache_active_indices,
    } = match compute_root_objective_bounds(
        verifier,
        graph,
        input,
        objectives,
        thresholds,
        conjunctive,
        engine,
        &bootstrap,
        deadline,
        root_crown_interm_tightened_elements > 0 || root_sparse_interm_tightened_targets > 0,
    ) {
        Ok(evaluation) => evaluation,
        Err(ny_core::NyError::DeadlineExceeded(_)) => {
            return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
                lifecycle.timeout_result(),
            )));
        }
        Err(e) => return Err(e),
    };
    crate::phase_telemetry::phase_marker("root-objective end");

    // #dd-zonotope INTERSECT: the certified zonotope margin and the CROWN
    // margin are two sound enclosures of the same objective values, so keeping
    // the tighter side of each is sound and can only raise a lower bound /
    // lower an upper bound. A refusal above leaves `dd_zono` as `None` and this
    // block is inert.
    if let Some(result) = dd_zono.as_ref() {
        let tightened = intersect_objective_bounds(&mut initial_obj_bounds, &result.margin);
        if tightened > 0 {
            info!(
                "#dd-zonotope: intersected {}/{} root objective bound(s)",
                tightened,
                initial_obj_bounds.len()
            );
        }
    }

    // Multi-neuron (k-ReLU) ROOT injection (increment 3, NY_MULTINEURON=1,
    // default-OFF). Runs the objective backward with sound coupling facets
    // injected at the head ReLU (§2.2) and combines the injected margin lower
    // bound with the baseline by a per-objective sound MAX — it can only RAISE
    // the certified lower bound, feeding both the root verdict and BaB. Byte-
    // identical when the gate is off. Given its own bounded grace slice out of
    // the remaining global budget (like the FC-head lever), so BaB keeps room.
    if crate::multineuron::root_inject::enabled() {
        let now = std::time::Instant::now();
        let grace_cap = std::time::Duration::from_secs(
            std::env::var("NY_MULTINEURON_GRACE_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(30),
        );
        let global_remaining = deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or(grace_cap);
        let slice = grace_cap.min(global_remaining.mul_f32(0.6));
        // Measurement override (NY_MULTINEURON_NODEADLINE=1): give the injected
        // backwards NO deadline so they complete against the tight acrown
        // intermediates (the ~-0.829 CPU backward) instead of IBP-falling-back
        // to the loose IBP intermediates (~-20). Deadlines only schedule work, so
        // this is sound; it may overrun the scored budget and is for A/B only.
        let mn_deadline = if matches!(
            std::env::var("NY_MULTINEURON_NODEADLINE").ok().as_deref(),
            Some("1")
        ) {
            None
        } else {
            Some(now + slice.max(std::time::Duration::from_secs(2)))
        };
        initial_obj_bounds = crate::multineuron::root_inject::tighten_root_objective_bounds(
            graph,
            input,
            objectives,
            engine,
            &bootstrap.initial_node_bounds,
            bootstrap.root_alpha_state.as_ref(),
            &initial_obj_bounds,
            mn_deadline,
        );
    }

    // STEM-RESIDENT research implementation. `stem_enabled()` is production-
    // authority quarantined until the facet support, fold reduction error, and
    // target/model binding have checker-backed certificates. This block is
    // therefore unreachable from environment requests today.
    if crate::multineuron::root_inject::stem_enabled() {
        let now = std::time::Instant::now();
        let grace_cap = std::time::Duration::from_secs(
            std::env::var("NY_MULTINEURON_STEM_GRACE_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(30),
        );
        let global_remaining = deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or(grace_cap);
        let stem_deadline = if matches!(
            std::env::var("NY_MULTINEURON_NODEADLINE").ok().as_deref(),
            Some("1")
        ) {
            None
        } else {
            Some(
                now + grace_cap
                    .min(global_remaining.mul_f32(0.6))
                    .max(std::time::Duration::from_secs(2)),
            )
        };
        initial_obj_bounds =
            crate::multineuron::root_inject::tighten_root_objective_bounds_stem_resident(
                graph,
                input,
                objectives,
                engine,
                &bootstrap.initial_node_bounds,
                bootstrap.root_alpha_state.as_ref(),
                &initial_obj_bounds,
                stem_deadline,
            );
    }

    // #mn-head-facet increment 1 (dark, NY_MN_HEAD_FACET=1, default-OFF). Build the
    // HEAD k-ReLU coupling-facet fold β-grid ONCE here at root (using the tight root
    // alpha-CROWN intermediates + node bounds) and register it in the shared
    // ny-core registry. Per-subdomain, the CPU f64 critical-row recovery
    // (`sound_f64_lower_bound`, itself armed by NY_F64_LINEAGE_RECOVER=1) reads the
    // registry and `max`-intersects each fold into best_lo[critical] — sound by
    // monotone max (β=0 / no-fold reproduces the recovery byte-for-byte). Does NOT
    // change the root objective bounds directly; only arms the per-domain lever.
    // The RESEARCH-MEASUREMENT gate (NY_MN_HEAD_F64_CERTIFIED_MEASURE=1, dark,
    // default-OFF) also fires this install, drawing EXACT-certified facets +
    // binding-select ranking into the SAME sound CPU-f64 masked fold. It is a
    // certifier-gated measurement lane on the f64 masked path ONLY (monotone max ⇒
    // can only RAISE the bound), NOT production verdict re-authorization.
    if crate::multineuron::root_inject::head_facet_enabled()
        || crate::multineuron::root_inject::head_f64_certified_measure_enabled()
    {
        crate::multineuron::root_inject::install_head_f64_fold(
            graph,
            input,
            objectives,
            &bootstrap.initial_node_bounds,
            bootstrap.root_alpha_state.as_ref(),
            engine,
        );
    }

    // #mn-head-resident (dark, NY_MN_HEAD_RESIDENT=1, default-OFF, byte-identical
    // when unset). The UNMASKED head lever: thread the HEAD coupling facets into
    // the OPTIMIZED resident GPU backward, RETARGETED from the stem to the head
    // act (fold index 0) — so the facet rides the tight GPU baseline itself rather
    // than the (masked) CPU f64 recovery. Single-global sound (root facet valid on
    // every subdomain); GUARD1 refuses any pool-node vs head-target mismatch. Sound
    // per-objective MAX with the baseline: it can only RAISE the certified lower
    // bound (INV-A: β=0 == baseline; INV-B: β>0 non-decreasing).
    if crate::multineuron::root_inject::head_resident_enabled() {
        let now = std::time::Instant::now();
        let grace_cap = std::time::Duration::from_secs(
            std::env::var("NY_MN_HEAD_RESIDENT_GRACE_SECS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(30),
        );
        let global_remaining = deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or(grace_cap);
        let head_deadline = if matches!(
            std::env::var("NY_MULTINEURON_NODEADLINE").ok().as_deref(),
            Some("1")
        ) {
            None
        } else {
            Some(
                now + grace_cap
                    .min(global_remaining.mul_f32(0.6))
                    .max(std::time::Duration::from_secs(2)),
            )
        };
        initial_obj_bounds =
            crate::multineuron::root_inject::tighten_root_objective_bounds_head_resident(
                graph,
                input,
                objectives,
                engine,
                &bootstrap.initial_node_bounds,
                bootstrap.root_alpha_state.as_ref(),
                &initial_obj_bounds,
                head_deadline,
            );
    }

    let verified_count = log_root_objective_bounds(&initial_obj_bounds, thresholds);
    // DIAGNOSTIC (NY_LPOPT_DUMP): also record NY's ROOT alpha-CROWN per-objective
    // (per-margin) lower/upper bounds + thresholds to `<path>.margins`, so the
    // off-line LP (p*_LP) can be compared to NY's own root bound WITHOUT needing
    // `-v` info tracing. One line per objective: `idx lower upper threshold`.
    if let Ok(path) = std::env::var("NY_LPOPT_DUMP") {
        let mpath = format!("{path}.margins");
        let mut s = String::new();
        for (idx, ((lo, up), th)) in initial_obj_bounds.iter().zip(thresholds.iter()).enumerate() {
            s.push_str(&format!("{idx} {lo} {up} {th}\n"));
        }
        match std::fs::write(&mpath, s) {
            Ok(()) => eprintln!(
                "[lpopt-dump] wrote {} root margins to {mpath}",
                initial_obj_bounds.len()
            ),
            Err(e) => eprintln!("[lpopt-dump] margins write error on {mpath}: {e}"),
        }
    }
    // DIAGNOSTIC (NY_ROOT_WIDTH_PROBE=1): per-layer bound-width profile + output-margin
    // looseness decomposition at the ROOT domain (no split). Read-only / print-only;
    // never mutates the bootstrap or feeds any verdict. See diag/cifar100-root-width.
    if std::env::var("NY_ROOT_WIDTH_PROBE").ok().as_deref() == Some("1") {
        run_root_width_probe(
            graph,
            input,
            objectives,
            thresholds,
            engine,
            &bootstrap,
            &initial_obj_bounds,
        );
    }
    // DIAGNOSTIC (NY_ROOT_CROWN_INTERM_PROBE=1): per-ReLU pre-activation total box
    // width, computed TWO ways at the ROOT (no BaB split): (a) NY's frozen
    // forward-linear reference bound (what every BaB subdomain inherits) vs (b) a
    // sound CROWN BACKWARD from that pre-activation node to the input eps-box
    // (heuristic α), and optionally (c) the same CROWN backward with the warmup's
    // OPTIMIZED α. Decides whether the frozen intermediate bounds have real CROWN
    // headroom or are already CROWN-tight. Read-only / print-only; never mutates
    // the bootstrap or any verdict. Default-OFF ⇒ byte-identical.
    if std::env::var("NY_ROOT_CROWN_INTERM_PROBE").ok().as_deref() == Some("1") {
        run_root_crown_interm_probe(graph, input, engine, &bootstrap);
    }
    if let Some(result) = maybe_finish_at_root(
        lifecycle,
        initial_output.clone(),
        &initial_obj_bounds,
        thresholds,
        conjunctive,
        verified_count,
    ) {
        return Ok(MultiObjectiveRootOutcome::Finished(Box::new(result)));
    }

    // Use bab_timeout so post-BaB PGD reservation is respected (#4095).
    if lifecycle.start_time.elapsed() > bab_timeout {
        return Ok(MultiObjectiveRootOutcome::Finished(Box::new(
            lifecycle.build_result_with_bounds(BabVerificationStatus::Timeout, initial_output),
        )));
    }

    let graph_setup = build_graph_bab_setup(graph, &bootstrap.initial_node_bounds);
    let cut_pool = build_graph_cut_pool(
        graph,
        &graph_setup.initial_node_bounds_arc,
        &graph_setup.relu_nodes,
        &verifier.config,
    )?;

    // Clone initial_obj_bounds before moving into root domain — needed by
    // per-disjunct alpha optimization below to identify unverified disjuncts.
    let initial_obj_bounds_ref = initial_obj_bounds.clone();
    let mut root_domain = MultiObjectiveGraphBabDomain::root(
        bootstrap.initial_node_bounds,
        initial_obj_bounds,
        input,
        thresholds,
        false,
    )?;
    root_domain.alpha_state = build_root_alpha_state(
        graph,
        input,
        &root_domain.history,
        &graph_setup.initial_node_bounds_arc,
        if root_crown_interm_tightened_elements > 0 || root_sparse_interm_tightened_targets > 0 {
            // The warmup alpha was optimized against the pre-tightening boxes.
            // It remains sound, but can be badly stale (prop1761: -1.026 vs
            // adaptive -0.496). Reinitialize the shared BaB state from the
            // inherited tightened boxes so children keep the root gain.
            None
        } else {
            bootstrap.root_alpha_state.as_ref()
        },
        verifier.config.beta_iterations > 0,
    );

    // Per-disjunct alpha optimization (#4355): when enabled and the property
    // is disjunctive, optimize alpha independently for each unverified disjunct.
    // Each disjunct gets a `GraphDomainAlphaState` with slopes specialized for
    // proving that specific output constraint (e.g., "prove Y_3 < Y_1").
    if let Some(root_alpha) = verifier
        .config
        .optimize_disjuncts_separately
        .then_some(bootstrap.root_alpha_state.as_ref())
        .flatten()
        .filter(|_| !conjunctive && objectives.len() > 1)
    {
        let per_disjunct = build_per_disjunct_alphas(
            graph,
            input,
            root_alpha,
            &bootstrap.alpha_config,
            bab_deadline,
            objectives,
            thresholds,
            &initial_obj_bounds_ref,
            &root_domain.history,
            &graph_setup.initial_node_bounds_arc,
            engine,
        )?;
        root_domain.set_per_disjunct_alphas(per_disjunct);
    }

    attach_root_spec_cache(
        &mut root_domain,
        root_spec_cache,
        &root_spec_cache_active_indices,
        objectives.len(),
    );

    let batch_size = verifier.config.batch_size.max(1);
    let use_batched_gpu = engine.is_some() && batch_size > 1 && !conjunctive;
    let mode_str = if conjunctive {
        "conjunctive"
    } else {
        "disjunctive"
    };
    info!(
        "Multi-objective BaB ({}): {} objectives, {} ReLU nodes, {} cuts, batch_size={}, gpu_batched={}, timeout {:?}",
        mode_str,
        objectives.len(),
        graph_setup.relu_nodes.len(),
        cut_pool.len(),
        batch_size,
        use_batched_gpu,
        verifier.config.timeout
    );

    Ok(MultiObjectiveRootOutcome::Continue(Box::new(
        MultiObjectiveRootState {
            root_domain,
            relu_nodes: graph_setup.relu_nodes,
            cut_pool,
            use_batched_gpu,
        },
    )))
}

/// Run the adaptive scalar/DAG coefficient backward, bypassing bounds-only
/// forward-linear/GPU root candidates while retaining its child-domain cache.
#[allow(clippy::too_many_arguments)]
fn run_adaptive_root_spec_candidate(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &ndarray::Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    alpha_state: Option<&crate::bounds::GraphAlphaState>,
    deadline: Option<std::time::Instant>,
    crown_backward_layers: Option<usize>,
) -> Result<(BoundedTensor, Option<CachedLinearBounds>)> {
    SpecCrownRequest::new(graph, input, spec_matrix, engine)
        .node_bounds(node_bounds)
        .alpha_state_opt(alpha_state)
        .deadline_opt(deadline)
        .truncate_after_opt(crown_backward_layers)
        .run_with_backward_cache()
}

/// Intersect two certified enclosures row by row. A disjoint element retains
/// `primary` while other overlapping rows still improve (never-worse fallback).
fn best_of_sound_root_spec_bounds(
    primary: BoundedTensor,
    secondary: &BoundedTensor,
) -> BoundedTensor {
    if primary.shape() != secondary.shape() {
        return primary;
    }
    let mut lower = primary.lower().clone();
    let mut upper = primary.upper().clone();
    ndarray::Zip::from(&mut lower)
        .and(&mut upper)
        .and(secondary.lower())
        .and(secondary.upper())
        .for_each(
            |primary_lower, primary_upper, &secondary_lower, &secondary_upper| {
                let intersect_lower = (*primary_lower).max(secondary_lower);
                let intersect_upper = (*primary_upper).min(secondary_upper);
                if intersect_lower <= intersect_upper {
                    *primary_lower = intersect_lower;
                    *primary_upper = intersect_upper;
                }
            },
        );
    BoundedTensor::new_allow_infinite(lower, upper).unwrap_or(primary)
}

#[cfg(test)]
mod post_tightening_alpha_tests {
    use super::{best_of_sound_root_spec_bounds, run_adaptive_root_spec_candidate};
    use crate::bounds::GraphAlphaState;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::{GraphNetwork, GraphNode};
    use ndarray::{arr1, arr2, Array1};
    use ny_tensor::BoundedTensor;

    fn stale_alpha_fixture() -> (GraphNetwork, BoundedTensor, Vec<f32>, Vec<f32>) {
        // All four hidden intervals cross zero. The adaptive slopes are
        // [1, 1, 0, 1]; a stale all-zero alpha loses >1.6 lower-margin units.
        let hidden_w = vec![2.361_429_2, 1.780_559_5, 1.406_410_1, 2.439_561_8];
        let hidden_b = vec![1.051_541_9, 1.158_990_5, -0.584_852_1, 1.923_906_3];
        let output_w = vec![2.771_405_7, -2.032_892, 1.524_024_4, 1.290_905_4];

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "hidden",
            Layer::Linear(
                LinearLayer::new(
                    ndarray::Array2::from_shape_vec((4, 1), hidden_w.clone()).unwrap(),
                    Some(Array1::from_vec(hidden_b.clone())),
                )
                .unwrap(),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["hidden".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(
                LinearLayer::new(
                    ndarray::Array2::from_shape_vec((1, 4), output_w).unwrap(),
                    None,
                )
                .unwrap(),
            ),
            vec!["relu".to_string()],
        ));
        graph.set_output("output");
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        (graph, input, hidden_w, hidden_b)
    }

    #[test]
    fn best_of_root_candidates_intersects_each_objective_row() {
        let adaptive = BoundedTensor::new(
            arr1(&[0.0_f32, 1.0, 2.0]).into_dyn(),
            arr1(&[10.0_f32, 11.0, 12.0]).into_dyn(),
        )
        .unwrap();
        let historical = BoundedTensor::new(
            arr1(&[1.0_f32, 0.0, 4.0]).into_dyn(),
            arr1(&[9.0_f32, 8.0, 13.0]).into_dyn(),
        )
        .unwrap();
        let best = best_of_sound_root_spec_bounds(adaptive, &historical);
        assert_eq!(best.lower(), &arr1(&[1.0_f32, 1.0, 4.0]).into_dyn());
        assert_eq!(best.upper(), &arr1(&[9.0_f32, 8.0, 12.0]).into_dyn());

        let adaptive = BoundedTensor::new(
            arr1(&[0.0_f32, 1.0, 2.0]).into_dyn(),
            arr1(&[10.0_f32, 11.0, 12.0]).into_dyn(),
        )
        .unwrap();
        let partially_disjoint = BoundedTensor::new(
            arr1(&[1.0_f32, 20.0, 4.0]).into_dyn(),
            arr1(&[9.0_f32, 21.0, 13.0]).into_dyn(),
        )
        .unwrap();
        let best = best_of_sound_root_spec_bounds(adaptive, &partially_disjoint);
        assert_eq!(best.lower(), &arr1(&[1.0_f32, 1.0, 4.0]).into_dyn());
        assert_eq!(best.upper(), &arr1(&[9.0_f32, 11.0, 12.0]).into_dyn());
    }

    #[test]
    fn post_tightening_best_of_stale_alpha_cannot_worsen_adaptive() {
        let (graph, input, hidden_w, hidden_b) = stale_alpha_fixture();
        let node_bounds = graph.collect_node_bounds(&input).unwrap();
        let spec = arr2(&[[1.0_f32]]);

        let (adaptive, _) = run_adaptive_root_spec_candidate(
            &graph,
            &input,
            &spec,
            None,
            &node_bounds,
            None,
            None,
            None,
        )
        .unwrap();
        let mut stale = GraphAlphaState::new();
        stale.alphas.insert("relu".to_string(), Array1::zeros(4));
        stale
            .alphas_upper
            .insert("relu".to_string(), Array1::zeros(4));
        stale
            .unstable_mask
            .insert("relu".to_string(), Array1::from_elem(4, true));
        let (stale_bounds, _) = run_adaptive_root_spec_candidate(
            &graph,
            &input,
            &spec,
            None,
            &node_bounds,
            Some(&stale),
            None,
            None,
        )
        .unwrap();

        let adaptive_lower = adaptive.lower()[[0]];
        let stale_lower = stale_bounds.lower()[[0]];
        assert!(
            adaptive_lower > stale_lower + 1.5,
            "fixture must discriminate adaptive from stale alpha: adaptive={adaptive_lower}, stale={stale_lower}"
        );
        let best = best_of_sound_root_spec_bounds(adaptive.clone(), &stale_bounds);
        assert!(best.lower()[[0]] >= adaptive_lower);
        assert!(best.upper()[[0]] <= adaptive.upper()[[0]]);

        let disjoint = BoundedTensor::new(
            arr1(&[adaptive.upper()[[0]] + 1.0]).into_dyn(),
            arr1(&[adaptive.upper()[[0]] + 2.0]).into_dyn(),
        )
        .unwrap();
        let preserved = best_of_sound_root_spec_bounds(adaptive.clone(), &disjoint);
        assert_eq!(preserved.lower(), adaptive.lower());
        assert_eq!(preserved.upper(), adaptive.upper());

        // The sound intersection must still enclose the concrete network.
        let output_w = [2.771_405_7, -2.032_892, 1.524_024_4, 1.290_905_4];
        for step in 0..=100 {
            let x = -1.0 + 2.0 * step as f32 / 100.0;
            let y: f32 = hidden_w
                .iter()
                .zip(&hidden_b)
                .zip(output_w)
                .map(|((&weight, &bias), output)| output * (weight * x + bias).max(0.0))
                .sum();
            assert!(
                y >= best.lower()[[0]] - 1e-5 && y <= best.upper()[[0]] + 1e-5,
                "best-of enclosure missed y={y} at x={x}: [{}, {}]",
                best.lower()[[0]],
                best.upper()[[0]]
            );
        }
    }
}

#[cfg(test)]
mod root_spec_prune_tests {
    use super::{
        build_root_spec_prune_plan, expand_root_spec_cache, merge_root_spec_pruned_bounds,
        root_prebound_certifies,
    };
    use crate::batched_domain::CachedLinearBounds;
    use ndarray::{arr1, arr2};
    use ny_tensor::BoundedTensor;

    fn fixture() -> (BoundedTensor, ndarray::Array2<f32>, Vec<Vec<f32>>, Vec<f32>) {
        let output = BoundedTensor::new(
            arr1(&[1.0_f32, 2.0]).into_dyn(),
            arr1(&[3.0_f32, 4.0]).into_dyn(),
        )
        .unwrap();
        let spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [-1.0, 0.0]]);
        let objectives = spec.outer_iter().map(|row| row.to_vec()).collect();
        // Rows 0 and 2 are certified by the output box; row 1 remains active.
        let thresholds = vec![0.0_f32, 2.5, -4.0];
        (output, spec, objectives, thresholds)
    }

    #[test]
    fn root_spec_prune_gate_off_preserves_full_request_exactly() {
        let (output, spec, objectives, thresholds) = fixture();
        let before = spec.clone();
        assert!(
            build_root_spec_prune_plan(false, false, &output, &spec, &objectives, &thresholds,)
                .is_none(),
            "gate-off must decline compression and leave the historical full request in force"
        );
        assert_eq!(
            spec, before,
            "planning must not mutate the full spec matrix"
        );

        assert!(
            build_root_spec_prune_plan(true, true, &output, &spec, &objectives, &thresholds,)
                .is_none(),
            "conjunctive root semantics must always retain the full request"
        );
    }

    #[test]
    fn root_spec_prune_active_matrix_keeps_only_unverified_rows_in_order() {
        let (output, spec, objectives, thresholds) = fixture();
        let plan =
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .expect("valid disjunctive fixture should produce a compression plan");

        assert_eq!(plan.active_indices, vec![1]);
        let active = plan
            .active_spec_matrix
            .expect("one unverified objective must produce one active row");
        assert_eq!(active.nrows(), 1);
        assert_eq!(active.ncols(), 2);
        assert_eq!(active.row(0), spec.row(1));
    }

    #[test]
    fn root_spec_prune_merge_restores_the_full_bound_vector_without_reordering() {
        let (output, spec, objectives, thresholds) = fixture();
        let plan =
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .unwrap();
        let active_result = (2.25_f32, 3.5_f32);
        let merged = merge_root_spec_pruned_bounds(&plan, vec![active_result])
            .expect("one active row should merge into a three-row full vector");

        assert_eq!(merged.len(), objectives.len());
        assert_eq!(merged[0], plan.pre_bounds[0]);
        assert_eq!(merged[1], active_result);
        assert_eq!(merged[2], plan.pre_bounds[2]);
        assert!(merge_root_spec_pruned_bounds(&plan, Vec::new()).is_none());

        let looser = merge_root_spec_pruned_bounds(&plan, vec![(1.0, 5.0)]).unwrap();
        assert_eq!(
            looser[1], plan.pre_bounds[1],
            "a looser active candidate must not weaken the bootstrap row"
        );
        assert!(
            merge_root_spec_pruned_bounds(&plan, vec![(10.0, 11.0)]).is_none(),
            "a disjoint active candidate must reject the compact result and cache"
        );
        assert!(
            merge_root_spec_pruned_bounds(&plan, vec![(f32::NAN, 3.5)]).is_none(),
            "a malformed active candidate must reject the compact result and cache"
        );

        for malformed_pre in [(f32::INFINITY, f32::INFINITY), (f32::NAN, f32::NAN)] {
            let mut malformed_plan =
                build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                    .unwrap();
            malformed_plan.pre_bounds[1] = malformed_pre;
            assert!(
                merge_root_spec_pruned_bounds(
                    &malformed_plan,
                    vec![(f32::INFINITY, f32::INFINITY)]
                )
                .is_none(),
                "an unusable prebound and unusable active result must retry the full request"
            );
            let recovered =
                merge_root_spec_pruned_bounds(&malformed_plan, vec![active_result]).unwrap();
            assert_eq!(
                recovered[1], active_result,
                "a fresh finite active enclosure must replace a malformed prebound"
            );
        }
    }

    #[test]
    fn root_spec_prune_all_pruned_skips_the_active_matrix() {
        let (output, spec, objectives, _) = fixture();
        let thresholds = vec![0.0_f32, 1.0, -4.0];
        let plan =
            build_root_spec_prune_plan(true, false, &output, &spec, &objectives, &thresholds)
                .expect("all-pruned fixture should still produce its certified full result");

        assert!(plan.active_indices.is_empty());
        assert!(plan.active_spec_matrix.is_none());
        assert_eq!(plan.pre_bounds.len(), objectives.len());
        for ((lower, _upper), threshold) in plan.pre_bounds.iter().zip(thresholds) {
            assert!(*lower > threshold);
        }
    }

    #[test]
    fn root_spec_prune_malformed_prebound_endpoints_never_certify() {
        assert!(root_prebound_certifies(1.0, 2.0, 0.0));
        assert!(!root_prebound_certifies(1.0, f32::NAN, 0.0));
        assert!(!root_prebound_certifies(1.0, f32::INFINITY, 0.0));
        assert!(!root_prebound_certifies(2.0, 1.0, 0.0));
        assert!(!root_prebound_certifies(f32::INFINITY, f32::INFINITY, 0.0));
    }

    #[test]
    fn root_spec_prune_cache_rows_expand_to_their_original_objective_slots() {
        let mut compact = CachedLinearBounds::default();
        compact
            .lower_a
            .insert("relu".to_string(), arr2(&[[10.0_f32, 11.0], [30.0, 31.0]]));
        compact
            .upper_a
            .insert("relu".to_string(), arr2(&[[12.0_f32, 13.0], [32.0, 33.0]]));
        compact
            .lower_b
            .insert("relu".to_string(), arr1(&[100.0_f32, 300.0]));
        compact
            .upper_b
            .insert("relu".to_string(), arr1(&[101.0_f32, 301.0]));

        let expanded = expand_root_spec_cache(&compact, &[1, 3], 4)
            .expect("two compact rows should map into four full objective slots");
        assert_eq!(expanded.len(), 4);
        assert!(expanded[0].is_none());
        assert!(expanded[2].is_none());
        assert_eq!(
            expanded[1]
                .as_ref()
                .and_then(|cache| cache.lower_a.get("relu"))
                .map(|a| a[[0, 0]]),
            Some(10.0)
        );
        assert_eq!(
            expanded[3]
                .as_ref()
                .and_then(|cache| cache.lower_a.get("relu"))
                .map(|a| a[[0, 0]]),
            Some(30.0)
        );
        assert!(expand_root_spec_cache(&compact, &[1, 1], 4).is_none());
        assert!(expand_root_spec_cache(&compact, &[1, 4], 4).is_none());
        assert!(
            expand_root_spec_cache(&compact, &[1], 4).is_none(),
            "a sparse one-row mapping must reject an unexpectedly two-row cache"
        );
    }
}

/// Choose the root objective authority as one atomic value. Keeping this seam
/// pure makes the fail-open contract explicit: when Stage B is disabled or
/// rejects, the exact Stage-A bounds, cache, and row map are moved through
/// unchanged. No per-row mutation occurs at this layer.
fn select_post_c_survivor_or_stage_a(
    stage_a_bounds: Vec<(f32, f32)>,
    stage_a_cache: Option<CachedLinearBounds>,
    stage_a_active_indices: Vec<usize>,
    post_c_survivor: Option<PostCSurvivorAccepted>,
) -> (Vec<(f32, f32)>, Option<CachedLinearBounds>, Vec<usize>) {
    match post_c_survivor {
        Some(post_c) => (
            post_c.merged_bounds,
            Some(post_c.compact_cache),
            post_c.active_indices,
        ),
        None => (stage_a_bounds, stage_a_cache, stage_a_active_indices),
    }
}

#[cfg(test)]
mod post_c_root_publication_tests {
    use super::select_post_c_survivor_or_stage_a;
    use crate::batched_domain::CachedLinearBounds;
    use ndarray::{arr1, arr2};

    #[test]
    fn disabled_or_refused_stage_b_preserves_stage_a_bounds_cache_and_map_bit_exactly() {
        let stage_a_bounds = vec![
            (f32::from_bits(0x3f80_0001), f32::from_bits(0x4000_0001)),
            (f32::from_bits(0xbf00_0001), f32::from_bits(0x4040_0001)),
        ];
        let expected_bits: Vec<_> = stage_a_bounds
            .iter()
            .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
            .collect();
        let mut stage_a_cache = CachedLinearBounds::default();
        stage_a_cache
            .lower_a
            .insert("relu".to_string(), arr2(&[[1.25_f32, -2.5], [3.75, -4.0]]));
        stage_a_cache
            .upper_a
            .insert("relu".to_string(), arr2(&[[5.25_f32, -6.5], [7.75, -8.0]]));
        stage_a_cache
            .lower_b
            .insert("relu".to_string(), arr1(&[0.125_f32, -0.25]));
        stage_a_cache
            .upper_b
            .insert("relu".to_string(), arr1(&[0.5_f32, -1.0]));

        let (bounds, cache, active_indices) = select_post_c_survivor_or_stage_a(
            stage_a_bounds,
            Some(stage_a_cache),
            vec![0, 1],
            None,
        );
        assert_eq!(
            bounds
                .iter()
                .map(|&(lower, upper)| (lower.to_bits(), upper.to_bits()))
                .collect::<Vec<_>>(),
            expected_bits
        );
        assert_eq!(active_indices, vec![0, 1]);
        let cache = cache.expect("Stage-A cache must survive a Stage-B refusal");
        assert_eq!(cache.lower_a["relu"], arr2(&[[1.25, -2.5], [3.75, -4.0]]));
        assert_eq!(cache.upper_a["relu"], arr2(&[[5.25, -6.5], [7.75, -8.0]]));
        assert_eq!(cache.lower_b["relu"], arr1(&[0.125, -0.25]));
        assert_eq!(cache.upper_b["relu"], arr1(&[0.5, -1.0]));
    }
}

pub(super) fn compute_root_objective_bounds(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    conjunctive: bool,
    engine: Option<&dyn GemmEngine>,
    bootstrap: &GraphBabBootstrap,
    global_deadline: Option<std::time::Instant>,
    root_intermediate_bounds_changed: bool,
) -> Result<RootObjectiveEvaluation> {
    let spec_matrix = build_spec_matrix(objectives);
    let mut root_spec_cache = None;
    let mut root_spec_cache_active_indices = Vec::new();
    // Keep gate-off and conjunctive execution on the exact historic path: do
    // not even resolve/inspect the bootstrap output unless compression is armed.
    let root_spec_prune_plan = if root_spec_prune_enabled() && !conjunctive {
        spec_matrix.as_ref().and_then(|full_spec_matrix| {
            let output = bootstrap_output_bounds(graph, bootstrap)?;
            build_root_spec_prune_plan(
                true,
                false,
                output,
                full_spec_matrix,
                objectives,
                thresholds,
            )
        })
    } else {
        None
    };

    // Root-pass grace slice (#w4-root-gpu): the alpha warmup runs to the FULL
    // initial deadline (measured on cifar100: `bootstrap.alpha_config.deadline`
    // is already EXPIRED by the time this root pass runs), so the C-matrix root
    // pass — the root-verification lever, <1s on the sound GPU resnet backward —
    // refused to start and the root objective bounds degraded to the per-logit
    // IBP projection every time. Grant the root spec pass a small grace slice
    // when its deadline has been consumed, capped by the caller's global
    // wall-clock deadline. Sound: deadlines only schedule work; the bounds
    // computed under the grace slice are the same certified machinery.
    //
    // #w4-root-alpha-opt: on conv DAGs the root pass additionally runs the
    // forward-map alpha OPTIMIZER (cheap surrogate sweeps) followed by ONE
    // alpha-fed rebuild of the forward-linear map — a full O(L) certified
    // pass (~22s on cifar100 release), the ROOT-relaxation lever. It needs no
    // warmup alphas (it starts from the adaptive slopes and optimizes the
    // forward objective directly). That work gets a larger grace: a slice of
    // the remaining global budget, capped at 30s. If it does not fit, the
    // optimizer fail-closes and the fixed-slope root candidates stand (never
    // uncapped work past the timeout — #4260 regression contract).
    const ROOT_SPEC_GRACE: std::time::Duration = std::time::Duration::from_secs(3);
    const ROOT_SPEC_ALPHA_GRACE_CAP: std::time::Duration = std::time::Duration::from_secs(40);
    let now = std::time::Instant::now();
    let alpha_rebuild_pending = graph.has_conv_layers()
        && !matches!(
            std::env::var("NY_SPEC_ROOT_ALPHA").ok().as_deref(),
            Some("0")
        );
    let grace_slice = if alpha_rebuild_pending {
        // 9/10 of the remaining global budget, cap 40s (measured at 95s with
        // the warmup cap above: remaining ≈ 35-43s at root; the optimizer +
        // rebuild need ~1.15x measured fixed cost + sweeps ≈ 30s+; the old
        // 0.8/30s grace left 22s and the optimizer could never fire). The
        // spec core skips the optimizer+rebuild entirely when the measured
        // cost does not fit this slice, so unused grace returns to BaB.
        let global_remaining = global_deadline
            .map(|g| g.saturating_duration_since(now))
            .unwrap_or(ROOT_SPEC_ALPHA_GRACE_CAP);
        global_remaining
            .mul_f32(0.9)
            .min(ROOT_SPEC_ALPHA_GRACE_CAP)
            .max(ROOT_SPEC_GRACE)
    } else {
        ROOT_SPEC_GRACE
    };
    let root_deadline = match bootstrap.alpha_config.deadline {
        // Live warmup deadline: keep it, but when the alpha rebuild is
        // pending make sure the root pass has at least the grace slice
        // (still capped by the global wall clock).
        Some(d) if d > now => {
            if alpha_rebuild_pending {
                let want = now + grace_slice;
                let capped = global_deadline.map_or(want, |g| want.min(g));
                Some(d.max(capped))
            } else {
                Some(d)
            }
        }
        Some(expired) => {
            let grace = now + grace_slice;
            let capped = global_deadline.map_or(grace, |g| grace.min(g));
            // Grace applies only while the GLOBAL wall-clock budget has room;
            // when everything is spent, keep the (expired) deadline so the
            // spec pass stays capped and bails to IBP immediately — never
            // uncapped work past the timeout (#4260 regression contract).
            Some(if capped > now { capped } else { expired })
        }
        None => None,
    };

    // NY_SLACK_PROBE: clear the per-row f32-slack accumulator so the report below
    // reflects only this root backward (dark; no-op when the gate is off).
    if crate::bounds::slack_probe_enabled() {
        let _ = crate::bounds::slack_probe_take();
    }
    let (initial_output, initial_obj_bounds) = if let Some(all_pruned) = root_spec_prune_plan
        .as_ref()
        .filter(|plan| plan.active_indices.is_empty())
    {
        info!(
            "Multi-objective: root spec pre-prune certified all {} objectives; skipping spec-guided CROWN",
            objectives.len(),
        );
        (
            all_pruned.bootstrap_output.clone(),
            all_pruned.pre_bounds.clone(),
        )
    } else if let Some(ref full_spec_mat) = spec_matrix {
        let mut applied_prune = root_spec_prune_plan.as_ref();
        let selected_spec_mat = applied_prune
            .and_then(|plan| plan.active_spec_matrix.as_ref())
            .unwrap_or(full_spec_mat);

        if let Some(plan) = applied_prune {
            info!(
                "Multi-objective: root spec pre-prune kept {} of {} objectives",
                plan.active_indices.len(),
                objectives.len(),
            );
        }

        // Tightening an intermediate box invalidates the QUALITY assumptions of
        // the warmup alpha (not its soundness). On prop1761 the inherited alpha
        // loses 0.53 margin after the dense-head box changes. Preserve the
        // historical request first, then—only with deadline left—sound-intersect
        // an adaptive full DAG backward. Unchanged boxes execute the original
        // request below exactly once.
        let run_spec_request = |spec_mat: &ndarray::Array2<f32>| {
            if root_intermediate_bounds_changed {
                let historical = SpecCrownRequest::new(graph, input, spec_mat, engine)
                    .node_bounds(&bootstrap.initial_node_bounds)
                    .alpha_state_opt(bootstrap.root_alpha_state.as_ref())
                    .deadline_opt(root_deadline)
                    .truncate_after_opt(verifier.config.crown_backward_layers)
                    .capture_cache()
                    .run_with_cache();
                // #root-adaptive-skip (dark, NY_ROOT_SKIP_ADAPTIVE_SPEC=1, default OFF
                // => byte-identical): skip the SECOND (adaptive, fresh-slope) full
                // coefficient backward and keep only the historical candidate (which
                // already consumes the tightened head node bounds). SOUND: the adaptive
                // pass only INTERSECTS extra tightening via best_of_sound_root_spec_bounds;
                // dropping it can only LOOSEN the root box, never emit a wrong verdict.
                // Probes whether the ~2x root-objective cost buys enough margin to justify
                // the warmup budget it steals from BaB.
                if root_deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
                    || matches!(
                        std::env::var("NY_ROOT_SKIP_ADAPTIVE_SPEC").ok().as_deref(),
                        Some("1")
                    )
                {
                    historical
                } else {
                    let adaptive = run_adaptive_root_spec_candidate(
                        graph,
                        input,
                        spec_mat,
                        engine,
                        &bootstrap.initial_node_bounds,
                        None,
                        root_deadline,
                        verifier.config.crown_backward_layers,
                    );
                    match (historical, adaptive) {
                        (
                            Ok((historical_bounds, _historical_cache)),
                            Ok((adaptive_bounds, cache)),
                        ) => {
                            // The cache is itself a certified linear enclosure and
                            // child consumers use it only as a same-row backward
                            // seed. It stays sound even when the independent
                            // historical candidate tightens the displayed root box.
                            Ok((
                                best_of_sound_root_spec_bounds(adaptive_bounds, &historical_bounds),
                                cache,
                            ))
                        }
                        (Ok(result), Err(error)) => {
                            debug!(
                                %error,
                                "Post-tightening adaptive root backward unavailable; retaining historical root candidate"
                            );
                            Ok(result)
                        }
                        (Err(error), Ok(result)) => {
                            debug!(
                                %error,
                                "Post-tightening historical root candidate unavailable; retaining adaptive backward"
                            );
                            Ok(result)
                        }
                        (Err(historical_error), Err(adaptive_error)) => {
                            debug!(
                                %adaptive_error,
                                "Post-tightening adaptive root backward also unavailable"
                            );
                            Err(historical_error)
                        }
                    }
                }
            } else {
                SpecCrownRequest::new(graph, input, spec_mat, engine)
                    .node_bounds(&bootstrap.initial_node_bounds)
                    .alpha_state_opt(bootstrap.root_alpha_state.as_ref())
                    .deadline_opt(root_deadline)
                    .truncate_after_opt(verifier.config.crown_backward_layers)
                    .capture_cache()
                    .run_with_cache()
            }
        };

        let mut spec_result = run_spec_request(selected_spec_mat);
        let mut merged_pruned_bounds = match (applied_prune, spec_result.as_ref()) {
            (Some(plan), Ok((bounds, _cache))) => {
                merge_root_spec_pruned_bounds(plan, spec_bounds_to_vec(bounds))
            }
            _ => None,
        };
        // A malformed compressed result is never authoritative. Retry the exact
        // historical full-spec request; if that also fails, the ordinary output-
        // bound fallback below remains in force.
        if applied_prune.is_some() && spec_result.is_ok() && merged_pruned_bounds.is_none() {
            debug!(
                "Root spec pre-prune produced a row-shape mismatch; retrying the full spec request"
            );
            spec_result = run_spec_request(full_spec_mat);
            applied_prune = None;
        }

        // #post-c-survivor Stage B (dark/default-OFF): Stage A above remains
        // the exact historic full-C request. A bounds-only fast candidate is
        // identifiable by its successful bounds plus absent backward cache.
        // Only then, and only for the uncompressed disjunctive root, offer its
        // <=16 unresolved rows to one generic full-DAG Patch-CROWN backward.
        // The helper has no alpha-state input and publishes bounds+cache only
        // after all rows validate, so every decline/fault retains Stage A
        // byte-for-byte and cannot leak a partially tightened row.
        let post_c_survivor: Option<PostCSurvivorAccepted> =
            if verifier.config.root_post_c_survivor && applied_prune.is_none() {
                spec_result
                    .as_ref()
                    .ok()
                    .filter(|(_stage_a_bounds, stage_a_cache)| stage_a_cache.is_none())
                    .and_then(|(stage_a_bounds, _stage_a_cache)| {
                        let plan = build_post_c_survivor_plan(
                            true,
                            conjunctive,
                            stage_a_bounds,
                            full_spec_mat,
                            thresholds,
                            input,
                            &bootstrap.initial_node_bounds,
                            std::time::Instant::now(),
                            global_deadline,
                        )?;
                        info!(
                            active_rows = plan.active_indices.len(),
                            total_rows = objectives.len(),
                            estimated_workspace_bytes = plan.estimated_workspace_bytes,
                            "Multi-objective: running bounded post-C survivor Patch-CROWN"
                        );
                        run_post_c_survivor_candidate(
                            graph,
                            input,
                            &bootstrap.initial_node_bounds,
                            engine,
                            plan,
                        )
                    })
            } else {
                None
            };

        match spec_result {
            Ok((bounds, cache)) => {
                let stage_a_obj_bounds = if let Some(plan) = applied_prune {
                    // Populated above from this exact `bounds` result.
                    merged_pruned_bounds
                        .take()
                        .unwrap_or_else(|| plan.pre_bounds.clone())
                } else {
                    spec_bounds_to_vec(&bounds)
                };
                if let Some(post_c) = post_c_survivor.as_ref() {
                    info!(
                        active_rows = post_c.active_indices.len(),
                        total_rows = objectives.len(),
                        "Multi-objective: post-C survivor Patch-CROWN accepted atomically"
                    );
                }
                let stage_a_active_indices = applied_prune.map_or_else(
                    || (0..objectives.len()).collect(),
                    |plan| plan.active_indices.clone(),
                );
                let (obj_bounds, selected_cache, selected_active_indices) =
                    select_post_c_survivor_or_stage_a(
                        stage_a_obj_bounds,
                        cache,
                        stage_a_active_indices,
                        post_c_survivor,
                    );
                root_spec_cache = selected_cache;
                root_spec_cache_active_indices = selected_active_indices;
                info!(
                    "Multi-objective: Using spec-guided CROWN ({} active / {} total objectives, cache_captured={})",
                    root_spec_cache_active_indices.len(),
                    obj_bounds.len(),
                    root_spec_cache.is_some(),
                );
                // Thread the deadline (#4321): this full IBP forward over a deep
                // conv DAG can itself overrun the verifier timeout. Aborting
                // between nodes lets the caller surface a graceful Timeout verdict.
                // Uses the same (possibly grace-extended) root deadline so a
                // successful spec pass is not immediately discarded by an
                // already-expired warmup deadline.
                let output =
                    graph.propagate_ibp_with_engine_and_deadline(input, engine, root_deadline)?;
                (output, obj_bounds)
            }
            Err(e) => {
                debug!(
                    "Spec-guided CROWN failed ({}), falling back to CROWN output bounds",
                    e
                );
                // Warmup cap (#4095) must survive fallback: this is a
                // continuation of the same bounded warmup, not a new phase.
                let output = compute_graph_root_output_bounds(
                    graph,
                    input,
                    &verifier.config,
                    engine,
                    bootstrap,
                    bootstrap.alpha_config.deadline,
                )?;
                let obj_bounds = BetaCrownVerifier::objective_bounds_multi(&output, objectives)?;
                (output, obj_bounds)
            }
        }
    } else {
        // Warmup cap (#4095) must survive the no-spec-matrix path:
        // same bounded warmup deadline as the spec-guided request.
        let output = compute_graph_root_output_bounds(
            graph,
            input,
            &verifier.config,
            engine,
            bootstrap,
            bootstrap.alpha_config.deadline,
        )?;
        let obj_bounds = BetaCrownVerifier::objective_bounds_multi(&output, objectives)?;
        (output, obj_bounds)
    };

    // NY_SLACK_PROBE report: how many margin-units the accumulated f32 soundness
    // rounding (`lower_a_err` folded over the box at every node) removed from the
    // BINDING (min-margin) objective. If this is ≪ the ~0.3 gap to α,β-CROWN, the
    // gap is relaxation looseness (f64 cannot help), not precision slack.
    if crate::bounds::slack_probe_enabled() && root_intermediate_bounds_changed {
        // Best-of evaluates two independent certified candidates, so summing
        // their diagnostic fold penalties would not describe the selected
        // enclosure. Drain rather than emit a misleading "exact" number.
        let _ = crate::bounds::slack_probe_take();
        eprintln!(
            "[NY_SLACK_PROBE] unavailable for post-tightening best-of root evaluation (multiple sound candidates)"
        );
    } else if crate::bounds::slack_probe_enabled() {
        let slack = crate::bounds::slack_probe_take();
        let (mut worst_row, mut worst_lb) = (0usize, f32::INFINITY);
        for (r, &(l, _)) in initial_obj_bounds.iter().enumerate() {
            if l < worst_lb {
                worst_lb = l;
                worst_row = r;
            }
        }
        let binding_slack = slack.get(worst_row).copied().unwrap_or(0.0);
        let max_slack = slack.iter().copied().fold(0.0f64, f64::max);
        let sum_slack: f64 = slack.iter().sum();
        eprintln!(
            "[NY_SLACK_PROBE] objectives={} binding_row={worst_row} binding_margin_lb={worst_lb:.6} \
             binding_f32_slack={binding_slack:.6} max_row_slack={max_slack:.6} total_slack_all_rows={sum_slack:.6}",
            initial_obj_bounds.len()
        );
        eprintln!(
            "[NY_SLACK_PROBE] => f32 soundness rounding removed {binding_slack:.6} margin-units from the \
             binding objective; an exact/f64 backward could recover AT MOST that much of any gap to 0."
        );
    }

    Ok(RootObjectiveEvaluation {
        initial_output,
        initial_obj_bounds,
        root_spec_cache,
        root_spec_cache_active_indices,
    })
}

fn log_root_objective_bounds(initial_obj_bounds: &[(f32, f32)], thresholds: &[f32]) -> usize {
    let verified_count = initial_obj_bounds
        .iter()
        .zip(thresholds.iter())
        .filter(|((lower, _upper), threshold)| *lower > **threshold)
        .count();

    for (idx, ((lower, upper), threshold)) in
        initial_obj_bounds.iter().zip(thresholds.iter()).enumerate()
    {
        info!(
            "Multi-objective obj[{}]: bounds=[{}, {}], threshold={}, verified={}",
            idx,
            lower,
            upper,
            threshold,
            *lower > *threshold
        );
    }
    info!(
        "Multi-objective initial: {}/{} objectives already verified",
        verified_count,
        initial_obj_bounds.len()
    );

    verified_count
}

fn maybe_finish_at_root(
    lifecycle: &mut GraphBabLifecycle,
    initial_output: BoundedTensor,
    initial_obj_bounds: &[(f32, f32)],
    thresholds: &[f32],
    conjunctive: bool,
    verified_count: usize,
) -> Option<BetaCrownResult> {
    let num_objectives = initial_obj_bounds.len();
    let initially_verified_all = verified_count == num_objectives;
    let initially_verified_any = verified_count > 0;

    if conjunctive && initially_verified_any {
        info!(
            "Multi-objective conjunctive: {}/{} objectives verified at root — property safe",
            verified_count, num_objectives
        );
        lifecycle.domains_explored = 1;
        lifecycle.domains_verified = 1;
        return Some(
            lifecycle.build_result_with_bounds(BabVerificationStatus::Verified, initial_output),
        );
    }
    if !conjunctive && initially_verified_all {
        lifecycle.domains_explored = 1;
        lifecycle.domains_verified = 1;
        return Some(
            lifecycle.build_result_with_bounds(BabVerificationStatus::Verified, initial_output),
        );
    }

    if conjunctive {
        let all_violated = initial_obj_bounds
            .iter()
            .zip(thresholds.iter())
            .all(|((_lower, upper), threshold)| *upper < *threshold);
        if all_violated {
            info!("Multi-objective conjunctive: ALL objectives conclusively violated at root");
            lifecycle.domains_explored = 1;
            return Some(lifecycle.build_result_with_bounds(
                BabVerificationStatus::Unknown {
                    reason:
                        "All objectives conclusively violated — conjunction may hold".to_string(),
                },
                initial_output,
            ));
        }
        return None;
    }

    for (idx, ((_lower, upper), threshold)) in
        initial_obj_bounds.iter().zip(thresholds.iter()).enumerate()
    {
        if *upper < *threshold {
            info!(
                "Multi-objective: objective {} is conclusively violated (upper={} < threshold={})",
                idx, upper, threshold
            );
            lifecycle.domains_explored = 1;
            return Some(lifecycle.build_result_with_bounds(
                BabVerificationStatus::Unknown {
                    reason: format!(
                        "Objective {} cannot be verified (upper {} < threshold {})",
                        idx, upper, threshold
                    ),
                },
                initial_output,
            ));
        }
    }

    None
}

fn attach_root_spec_cache(
    root_domain: &mut MultiObjectiveGraphBabDomain,
    root_spec_cache: Option<CachedLinearBounds>,
    active_indices: &[usize],
    num_objectives: usize,
) {
    let Some(multi_row_cache) = root_spec_cache else {
        return;
    };

    if let Some(per_objective_caches) =
        expand_root_spec_cache(&multi_row_cache, active_indices, num_objectives)
    {
        let captured_nodes = per_objective_caches
            .iter()
            .flatten()
            .next()
            .map_or(0, CachedLinearBounds::len);
        let captured_objectives = per_objective_caches.iter().flatten().count();
        if let Err(err) = root_domain.set_cached_las(per_objective_caches) {
            debug!("lA warm-start: failed to set cached_las on root: {err}");
        } else {
            info!(
                "lA warm-start: captured {} of {} per-objective cached_las on root domain across {} nodes",
                captured_objectives,
                num_objectives,
                captured_nodes,
            );
        }
    } else {
        debug!(
            "lA warm-start: cache row expansion returned None (index, linear-shape, or empty mismatch)"
        );
    }
}

/// Split compact cache rows and restore their full objective positions.  This
/// is deliberately total-order preserving and rejects duplicates/out-of-range
/// indices; a declined cache is only a performance loss, never a bound change.
fn expand_root_spec_cache(
    multi_row_cache: &CachedLinearBounds,
    active_indices: &[usize],
    num_objectives: usize,
) -> Option<Vec<Option<CachedLinearBounds>>> {
    if active_indices.is_empty() || num_objectives == 0 {
        return None;
    }
    let is_legacy_full_layout = active_indices.iter().copied().eq(0..num_objectives);
    // Legacy split_multi_row intentionally accepts "at least N" rows. Preserve
    // that exact behavior for the gate-off full layout, but require an exact
    // compact shape before mapping sparse rows: otherwise a mistakenly full
    // cache could silently attach its row 0 to active objective k.
    if !is_legacy_full_layout
        && (multi_row_cache
            .lower_a
            .values()
            .chain(multi_row_cache.upper_a.values())
            .any(|a| a.nrows() != active_indices.len())
            || multi_row_cache
                .lower_b
                .values()
                .chain(multi_row_cache.upper_b.values())
                .any(|b| b.len() != active_indices.len()))
    {
        return None;
    }

    let per_active = multi_row_cache.split_multi_row(active_indices.len())?;
    if per_active.len() != active_indices.len() {
        return None;
    }

    let mut seen = vec![false; num_objectives];
    let mut full: Vec<Option<CachedLinearBounds>> = vec![None; num_objectives];
    for (&full_idx, cache) in active_indices.iter().zip(per_active) {
        if full_idx >= num_objectives || seen[full_idx] {
            return None;
        }
        seen[full_idx] = true;
        full[full_idx] = Some(cache);
    }
    Some(full)
}

// ============================================================================
// DIAGNOSTIC-ONLY (NY_ROOT_WIDTH_PROBE=1) — cifar100 root looseness decomposition.
// Print-only. Not compiled out, but every effect is behind the env gate and never
// mutates the bootstrap or any verdict path. Branch: diag/cifar100-root-width.
// ============================================================================

fn probe_layer_kind(layer: &crate::Layer) -> &'static str {
    use crate::Layer;
    match layer {
        Layer::Conv2d(_) | Layer::Conv1d(_) => "conv",
        Layer::ConvTranspose2d(_) | Layer::ConvTranspose1d(_) => "convT",
        Layer::Linear(_) => "linear",
        Layer::ReLU(_) => "relu",
        Layer::BatchNorm(_) => "batchnorm",
        Layer::Add(_) => "add",
        Layer::Sub(_) => "sub",
        Layer::AveragePool(_) => "avgpool",
        Layer::MaxPool2d(_) => "maxpool",
        _ => "other",
    }
}

/// Width stats (max, mean) over `u_i - l_i` and numel for a BoundedTensor.
fn probe_width_stats(bt: &BoundedTensor) -> (f32, f32, usize) {
    let lo = bt.lower();
    let hi = bt.upper();
    let mut maxw = 0.0f32;
    let mut sumw = 0.0f64;
    let mut n = 0usize;
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        let w = u - l;
        if w.is_finite() {
            if w > maxw {
                maxw = w;
            }
            sumw += w as f64;
            n += 1;
        }
    }
    let mean = if n > 0 { (sumw / n as f64) as f32 } else { 0.0 };
    (maxw, mean, n)
}

/// Fraction of neurons with l<0<u (unstable — the only ReLUs whose relaxation has a gap).
fn probe_unstable_frac(bt: &BoundedTensor) -> (f32, usize, usize) {
    let lo = bt.lower();
    let hi = bt.upper();
    let mut unst = 0usize;
    let mut n = 0usize;
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        n += 1;
        if l < 0.0 && u > 0.0 {
            unst += 1;
        }
    }
    let frac = if n > 0 { unst as f32 / n as f32 } else { 0.0 };
    (frac, unst, n)
}

/// Build a copy of `map` with every non-input node's [l,u] shrunk toward its midpoint:
/// new half-width = `keep` * old half-width. `keep=1.0` is identity, `keep=0.0` collapses
/// to the midpoint. The graph input node is left untouched (the real property region).
fn probe_shrink_map(
    map: &std::collections::HashMap<String, BoundedTensor>,
    keep: f32,
    input_node: &str,
) -> std::collections::HashMap<String, BoundedTensor> {
    probe_shrink_filtered(map, keep, &|n| n != input_node)
}

/// Shrink toward midpoint only for nodes where `pred(name)` is true; others unchanged.
fn probe_shrink_filtered(
    map: &std::collections::HashMap<String, BoundedTensor>,
    keep: f32,
    pred: &dyn Fn(&str) -> bool,
) -> std::collections::HashMap<String, BoundedTensor> {
    map.iter()
        .map(|(name, bt)| {
            if !pred(name) {
                return (name.clone(), bt.clone());
            }
            let lo = bt.lower();
            let hi = bt.upper();
            let shape: Vec<usize> = lo.shape().to_vec();
            let new_lo_vec: Vec<f32> = lo
                .iter()
                .zip(hi.iter())
                .map(|(&l, &u)| {
                    // Bit-identical shrink center: f32::midpoint rounds differently at overflow/subnormal edges.
                    #[allow(clippy::manual_midpoint)]
                    let mid = (l + u) * 0.5f32;
                    let half = (u - l) * 0.5f32 * keep;
                    mid - half
                })
                .collect();
            let new_hi_vec: Vec<f32> = lo
                .iter()
                .zip(hi.iter())
                .map(|(&l, &u)| {
                    // Bit-identical shrink center: f32::midpoint rounds differently at overflow/subnormal edges.
                    #[allow(clippy::manual_midpoint)]
                    let mid = (l + u) * 0.5f32;
                    let half = (u - l) * 0.5f32 * keep;
                    mid + half
                })
                .collect();
            let new_lo = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), new_lo_vec);
            let new_hi = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), new_hi_vec);
            let shrunk = match (new_lo, new_hi) {
                (Ok(nl), Ok(nh)) => BoundedTensor::new(nl, nh).unwrap_or_else(|_| bt.clone()),
                _ => bt.clone(),
            };
            (name.clone(), shrunk)
        })
        .collect()
}

/// Min / mean of the per-objective margin LOWER bound + count verified (lower>threshold).
fn probe_margin_stats(out: &BoundedTensor, thresholds: &[f32]) -> (f32, f32, usize, usize) {
    let lo = out.lower();
    let vals: Vec<f32> = lo.iter().copied().collect();
    let mut min = f32::INFINITY;
    let mut sum = 0.0f64;
    let mut verified = 0usize;
    for (i, &v) in vals.iter().enumerate() {
        if v < min {
            min = v;
        }
        sum += v as f64;
        if v > thresholds.get(i).copied().unwrap_or(0.0) {
            verified += 1;
        }
    }
    let n = vals.len();
    let mean = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
    (min, mean, verified, n)
}

/// Run the SpecCrownRequest margin pass with a given intermediate-bound map (+ fixed root α).
fn probe_margin_with_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec: &ndarray::Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    alpha: Option<&crate::bounds::GraphAlphaState>,
    thresholds: &[f32],
) -> Option<(f32, f32, usize, usize)> {
    let out = SpecCrownRequest::new(graph, input, spec, engine)
        .node_bounds(node_bounds)
        .alpha_state_opt(alpha)
        .deadline_opt(None)
        .run()
        .ok()?;
    Some(probe_margin_stats(&out, thresholds))
}

#[allow(clippy::too_many_arguments)]
fn run_root_width_probe(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
    bootstrap: &GraphBabBootstrap,
    initial_obj_bounds: &[(f32, f32)],
) {
    let acrown = &bootstrap.initial_node_bounds;
    let alpha = bootstrap.root_alpha_state.as_ref();
    let out_name = if graph.output_name().is_empty() {
        graph
            .exec_order()
            .ok()
            .and_then(|o| o.last().cloned())
            .unwrap_or_default()
    } else {
        graph.output_name().to_string()
    };
    let input_node = graph
        .exec_order()
        .ok()
        .and_then(|o| o.first().cloned())
        .unwrap_or_default();

    // Plain IBP intermediate bounds for the A/B comparison.
    let ibp = graph.collect_node_bounds_with_engine(input, engine).ok();

    eprintln!(
        "[root-width] ===== ROOT LOOSENESS PROBE (out_node={out_name} input_node={input_node} n_obj={}) =====",
        objectives.len()
    );

    // ---- Per-node width profile in exec order (α-CROWN vs IBP intermediates) ----
    let order: Vec<String> = graph
        .exec_order()
        .map(|o| o.to_vec())
        .unwrap_or_else(|_| acrown.keys().cloned().collect());
    eprintln!("[root-width] --- per-node width (exec order): node kind numel | acrown[max mean] | ibp[max mean] | acrown/ibp ---");
    for name in &order {
        let Some(bt) = acrown.get(name) else { continue };
        let kind = graph
            .nodes
            .get(name)
            .map(|nd| probe_layer_kind(&nd.layer))
            .unwrap_or("?");
        let (amax, amean, an) = probe_width_stats(bt);
        let (imax, imean, ratio) = match ibp.as_ref().and_then(|m| m.get(name)) {
            Some(ib) => {
                let (im, imn, _) = probe_width_stats(ib);
                (im, imn, if imn > 0.0 { amean / imn } else { f32::NAN })
            }
            None => (f32::NAN, f32::NAN, f32::NAN),
        };
        eprintln!(
            "[root-width] node={name} kind={kind} numel={an} acrown_max={amax:.4} acrown_mean={amean:.4} ibp_max={imax:.4} ibp_mean={imean:.4} ratio={ratio:.3}"
        );
    }

    // ---- ReLU pre-activation unstable fractions (drives every triangle relaxation) ----
    eprintln!(
        "[root-width] --- ReLU pre-activation (input-node bounds): unstable fraction + width ---"
    );
    let mut tot_relu_neurons = 0usize;
    let mut tot_unstable_acrown = 0usize;
    let mut tot_unstable_ibp = 0usize;
    for name in &order {
        let Some(nd) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(nd.layer, crate::Layer::ReLU(_)) {
            continue;
        }
        let Some(in_name) = nd.inputs.first() else {
            continue;
        };
        let Some(pre) = acrown.get(in_name) else {
            continue;
        };
        let (afrac, aun, an) = probe_unstable_frac(pre);
        let (_amax, amean, _) = probe_width_stats(pre);
        tot_relu_neurons += an;
        tot_unstable_acrown += aun;
        let (ifrac, iun, imean) = match ibp.as_ref().and_then(|m| m.get(in_name)) {
            Some(ib) => {
                let (f, u, _) = probe_unstable_frac(ib);
                tot_unstable_ibp += u;
                let (_m, mn, _) = probe_width_stats(ib);
                (f, u, mn)
            }
            None => (f32::NAN, 0, f32::NAN),
        };
        eprintln!(
            "[root-width] relu={name} pre={in_name} numel={an} acrown_unstable={aun}({afrac:.3}) acrown_meanw={amean:.4} | ibp_unstable={iun}({ifrac:.3}) ibp_meanw={imean:.4}"
        );
    }
    eprintln!(
        "[root-width] RELU TOTALS: neurons={tot_relu_neurons} unstable_acrown={tot_unstable_acrown}({:.4}) unstable_ibp={tot_unstable_ibp}({:.4})",
        tot_unstable_acrown as f32 / tot_relu_neurons.max(1) as f32,
        tot_unstable_ibp as f32 / tot_relu_neurons.max(1) as f32,
    );

    // ---- Output-margin looseness decomposition ----
    let root_min = initial_obj_bounds
        .iter()
        .map(|(l, _)| *l)
        .fold(f32::INFINITY, f32::min);
    eprintln!(
        "[root-width] --- OUTPUT MARGIN DECOMPOSITION (min margin-lower; verify needs >0) ---"
    );
    eprintln!(
        "[root-width] real_root_margin_min={root_min:.5} (from compute_root_objective_bounds)"
    );

    // Pure IBP-concretized margin (no CROWN backward at all).
    if let Some(ibp_map) = ibp.as_ref() {
        if let Some(o) = ibp_map.get(&out_name) {
            let olo: Vec<f32> = o.lower().iter().copied().collect();
            let ohi: Vec<f32> = o.upper().iter().copied().collect();
            let mut min = f32::INFINITY;
            for obj in objectives {
                let mut lb = 0.0f32;
                for (j, &c) in obj.iter().enumerate() {
                    lb += if c >= 0.0 {
                        c * olo.get(j).copied().unwrap_or(0.0)
                    } else {
                        c * ohi.get(j).copied().unwrap_or(0.0)
                    };
                }
                if lb < min {
                    min = lb;
                }
            }
            eprintln!("[root-width] ibp_concretized_margin_min={min:.5} (IBP output box · objective, no backward)");
        }
    }

    let Some(spec) = build_spec_matrix(objectives) else {
        eprintln!("[root-width] build_spec_matrix failed; skipping CROWN-backward decomposition");
        return;
    };

    // Baseline: CROWN backward over α-CROWN intermediates + fixed root α.
    if let Some((mn, mean, ver, n)) =
        probe_margin_with_bounds(graph, input, &spec, engine, acrown, alpha, thresholds)
    {
        eprintln!("[root-width] CROWN[acrown_interm] margin_min={mn:.5} mean={mean:.5} verified={ver}/{n}");
    }
    // CROWN backward but with IBP intermediates (isolates: how much tighter intermediates buy).
    if let Some(ibp_map) = ibp.as_ref() {
        if let Some((mn, mean, ver, n)) =
            probe_margin_with_bounds(graph, input, &spec, engine, ibp_map, alpha, thresholds)
        {
            eprintln!("[root-width] CROWN[ibp_interm]    margin_min={mn:.5} mean={mean:.5} verified={ver}/{n}");
        }
    }
    // Artificially tightened intermediates: shrink each [l,u] toward its midpoint.
    // If margin barely moves => ReLU relaxation given these intermediates is the wall.
    // If margin jumps toward >0 => intermediate-bound looseness is the lever.
    for keep in [0.75f32, 0.5, 0.25, 0.1, 0.0] {
        let shrunk = probe_shrink_map(acrown, keep, &input_node);
        if let Some((mn, mean, ver, n)) =
            probe_margin_with_bounds(graph, input, &spec, engine, &shrunk, alpha, thresholds)
        {
            eprintln!(
                "[root-width] CROWN[shrink keep={keep:.2}] margin_min={mn:.5} mean={mean:.5} verified={ver}/{n}"
            );
        }
    }
    // TARGETED: isolate the FC head (the width-explosion layers) from the conv stack.
    // The pre-activation feeding the final ReLU (Relu_57) is Gemm_56. If shrinking ONLY
    // Gemm_56 closes most of the gap while shrinking everything-but-the-head barely moves,
    // the FC-head intermediate bounds are THE lever.
    let head_nodes = ["Gemm_56", "Relu_57", "Gemm_58"];
    let is_head = |n: &str| head_nodes.contains(&n);
    for keep in [0.5f32, 0.0] {
        let head_only = probe_shrink_filtered(acrown, keep, &|n| n == "Gemm_56");
        if let Some((mn, _mean, ver, n)) =
            probe_margin_with_bounds(graph, input, &spec, engine, &head_only, alpha, thresholds)
        {
            eprintln!("[root-width] CROWN[shrink Gemm_56-ONLY keep={keep:.2}] margin_min={mn:.5} verified={ver}/{n}");
        }
        let except_head = probe_shrink_filtered(acrown, keep, &|n| n != input_node && !is_head(n));
        if let Some((mn, _mean, ver, n)) =
            probe_margin_with_bounds(graph, input, &spec, engine, &except_head, alpha, thresholds)
        {
            eprintln!("[root-width] CROWN[shrink CONV-STACK-only(except head) keep={keep:.2}] margin_min={mn:.5} verified={ver}/{n}");
        }
    }
    eprintln!("[root-width] ===== END ROOT LOOSENESS PROBE =====");
}

/// Total finite box width Σ_j(u_j−l_j) and the finite-neuron count for a tensor.
fn probe_width_total(bt: &BoundedTensor) -> (f64, usize) {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (&l, &u) in bt.lower().iter().zip(bt.upper().iter()) {
        let w = (u - l) as f64;
        if w.is_finite() {
            sum += w;
            n += 1;
        }
    }
    (sum, n)
}

/// DIAGNOSTIC-ONLY (NY_ROOT_CROWN_INTERM_PROBE=1): compare, at the ROOT (no BaB
/// split), NY's FROZEN forward-linear pre-activation box width vs a sound
/// CROWN-backward pre-activation box width, per intermediate ReLU layer.
///
/// For every ReLU node, its pre-activation is the ReLU's single input node.
/// - (a) `fwd_linear_width` = Σ width of the frozen `initial_node_bounds[pre]`
///   (the forward-linear certified reference every BaB subdomain inherits);
/// - (b) `crown_heuristic_width` = Σ width of a sound CROWN BACKWARD from `pre`
///   to the input eps-box with heuristic α (`backward_input_relative_bounds_at_node`,
///   which folds the certified coeff error outward and refuses non-finite rows —
///   a valid enclosure) concretized via `LinearBounds::concretize_sound`;
/// - (c) `crown_optalpha_width` = the same sound backward with the warmup's
///   optimized α folded in (NY_ROOT_CROWN_INTERM_OPTALPHA≠0, when a warmup α exists).
///
/// `ratio_crown/fwd = (b)/(a)`: ≤0.5 ⇒ real headroom in the frozen intermediate
/// bounds (intersect the CROWN bounds in next); ~0.9–1.0 ⇒ the forward-linear
/// reference is already CROWN-tight (intermediate-bounds lever closed).
///
/// Print-only; never mutates any bound or verdict. See docs/ROOT_JOINT_INTERM_ALPHA_PLAN.md.
/// DIAGNOSTIC-ONLY (`NY_LPOPT_DUMP=<path>`): serialize the exact root state that
/// feeds every BaB subdomain, so the triangle-relaxation LP (`p*_LP`) can be
/// rebuilt off-line from NY's OWN bounds + relaxation.
///
/// Writes a plain-text file:
/// ```text
/// # ny lpopt dump v1
/// INPUT <n> <shape dims...>
/// L <l0> <l1> ...            # input eps-box lower, logical (row-major) order
/// U <u0> <u1> ...            # input eps-box upper
/// RELUMAP <k>
/// <relu_node_name> <pre_activation_node_name>   # one per ReLU, exec order
/// ...
/// NODE <name> <n> <shape dims...>
/// L <l0> ...                 # bootstrap.initial_node_bounds[name] lower
/// U <u0> ...
/// ...
/// ```
/// Floats use Rust's shortest round-trip `Display` (bit-exact reload of the f32
/// bounds NY's ReLU big-M / triangle relaxation actually casts to f64). Runs once
/// at the root, AFTER the optional CROWN-interm tighten, so it captures exactly the
/// frozen `initial_node_bounds` each subdomain inherits by pointer.
fn run_lpopt_dump(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bootstrap: &GraphBabBootstrap,
    path: &str,
) {
    use std::io::Write;
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[lpopt-dump] failed to create {path}: {e}");
            return;
        }
    };
    let mut w = std::io::BufWriter::new(file);
    let write_lu =
        |w: &mut std::io::BufWriter<std::fs::File>, bt: &BoundedTensor| -> std::io::Result<()> {
            let (lo, up) = bt.lower_upper();
            write!(w, "L")?;
            for v in lo.iter() {
                write!(w, " {v}")?;
            }
            writeln!(w)?;
            write!(w, "U")?;
            for v in up.iter() {
                write!(w, " {v}")?;
            }
            writeln!(w)
        };
    let res = (|| -> std::io::Result<()> {
        writeln!(w, "# ny lpopt dump v1")?;
        // Input eps-box.
        write!(w, "INPUT {}", input.len())?;
        for d in input.shape() {
            write!(w, " {d}")?;
        }
        writeln!(w)?;
        write_lu(&mut w, input)?;
        // ReLU -> pre-activation node-name map (exec order).
        let order = graph.exec_order().map(|o| o.to_vec()).unwrap_or_default();
        let relu_pairs: Vec<(String, String)> = order
            .iter()
            .filter_map(|name| {
                let node = graph.node(name)?;
                if !matches!(node.layer(), crate::Layer::ReLU(_)) {
                    return None;
                }
                let pre = node.inputs().first()?.clone();
                Some((name.clone(), pre))
            })
            .collect();
        writeln!(w, "RELUMAP {}", relu_pairs.len())?;
        for (relu, pre) in &relu_pairs {
            writeln!(w, "{relu} {pre}")?;
        }
        // Every node's frozen pre-activation box (keyed by node name).
        for (name, bt) in bootstrap.initial_node_bounds.iter() {
            write!(w, "NODE {name} {}", bt.len())?;
            for d in bt.shape() {
                write!(w, " {d}")?;
            }
            writeln!(w)?;
            write_lu(&mut w, bt)?;
        }
        w.flush()
    })();
    match res {
        Ok(()) => eprintln!(
            "[lpopt-dump] wrote {} nodes + input box to {path}",
            bootstrap.initial_node_bounds.len()
        ),
        Err(e) => eprintln!("[lpopt-dump] write error on {path}: {e}"),
    }
}

fn run_root_crown_interm_probe(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    bootstrap: &GraphBabBootstrap,
) {
    let Some(engine) = engine else {
        eprintln!("[root-crown-interm] no GPU engine (need the `ny vnncomp` GPU preset); skipping");
        return;
    };
    let acrown = &bootstrap.initial_node_bounds;
    // One-time Arc view of the frozen root map for the sound CROWN backward
    // (`backward_input_relative_bounds_at_node` reads the #cone-delta
    // increment 2 Arc-shared cache type). Diagnostic lane; values unchanged.
    let acrown_arc = build_initial_node_bounds_arc(acrown);
    let order: Vec<String> = match graph.exec_order() {
        Ok(o) => o.to_vec(),
        Err(_) => {
            eprintln!("[root-crown-interm] exec_order unavailable; skipping");
            return;
        }
    };
    // Bound the one-time probe cost: skip start-node seeds wider than this
    // (default 20000 = the whole cifar100 conv stack + head fits).
    let max_dim = std::env::var("NY_ROOT_CROWN_INTERM_MAXDIM")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(20000);
    // (c) optimized-α ceiling: build a root GraphDomainAlphaState from the warmup
    // α and fold it into the SAME sound backward. Default ON when a warmup α is
    // present; NY_ROOT_CROWN_INTERM_OPTALPHA=0 disables it (heuristic-α only).
    let want_optalpha = std::env::var("NY_ROOT_CROWN_INTERM_OPTALPHA")
        .ok()
        .as_deref()
        != Some("0");
    let domain_alpha: Option<GraphDomainAlphaState> = if want_optalpha {
        bootstrap.root_alpha_state.as_ref().map(|ra| {
            let arc = build_initial_node_bounds_arc(acrown);
            let hist = GraphSplitHistory::new();
            GraphDomainAlphaState::from_root_alpha_state(ra, graph, &arc, &hist, input)
        })
    } else {
        None
    };

    eprintln!(
        "[root-crown-interm] ===== ROOT CROWN-BACKWARD vs FROZEN FORWARD-LINEAR INTERM PROBE (max_dim={max_dim} optalpha={}) =====",
        domain_alpha.is_some()
    );
    let t0 = std::time::Instant::now();
    for name in &order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer, crate::Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs.first() else {
            continue;
        };
        let Some(ref_bt) = acrown.get(pre) else {
            continue;
        };
        let (fwd_sum, pre_dim) = probe_width_total(ref_bt);
        if pre_dim == 0 {
            continue;
        }
        if pre_dim > max_dim {
            eprintln!(
                "[root-crown-interm] relu={name} pre={pre} pre_dim={pre_dim} fwd_linear_width={fwd_sum:.4} crown_heuristic_width=SKIP(>max_dim) crown_optalpha_width=SKIP ratio_crown/fwd=SKIP"
            );
            continue;
        }
        // (b) heuristic-α sound CROWN backward from `pre` to the input eps-box.
        let crown_h = crate::beta_crown::engine::graph::propagation::batched::backward_input_relative_bounds_at_node(
            graph, pre, &acrown_arc, input, engine, None, None,
        )
        .map(|lb| probe_width_total(&lb.concretize_sound(input)).0);
        // (c) optimized-α sound CROWN backward (same lane, warmup α).
        let crown_o = domain_alpha.as_ref().and_then(|da| {
            crate::beta_crown::engine::graph::propagation::batched::backward_input_relative_bounds_at_node(
                graph, pre, &acrown_arc, input, engine, None, Some(da),
            )
            .map(|lb| probe_width_total(&lb.concretize_sound(input)).0)
        });
        let ch = crown_h
            .map(|w| format!("{w:.4}"))
            .unwrap_or_else(|| "REFUSED".into());
        let co = if domain_alpha.is_none() {
            "OFF".to_string()
        } else {
            crown_o
                .map(|w| format!("{w:.4}"))
                .unwrap_or_else(|| "REFUSED".into())
        };
        let ratio = crown_h
            .filter(|_| fwd_sum > 0.0)
            .map(|w| format!("{:.4}", w / fwd_sum))
            .unwrap_or_else(|| "NA".into());
        eprintln!(
            "[root-crown-interm] relu={name} pre={pre} pre_dim={pre_dim} fwd_linear_width={fwd_sum:.4} crown_heuristic_width={ch} crown_optalpha_width={co} ratio_crown/fwd={ratio}"
        );
    }
    eprintln!(
        "[root-crown-interm] ===== END ({:.1}s) =====",
        t0.elapsed().as_secs_f32()
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RootSparseIntermCrownPolicy {
    max_dim: usize,
    max_rows: usize,
    max_targets: usize,
    max_secs: u64,
}

const ROOT_SPARSE_INTERM_ABS_MAX_DIM: usize = 8_192;
const ROOT_SPARSE_INTERM_ABS_MAX_ROWS: usize = 512;
const ROOT_SPARSE_INTERM_ABS_MAX_TARGETS: usize = 4;
const ROOT_SPARSE_INTERM_ABS_MAX_SECS: u64 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootSparseGateEnv<'a> {
    Absent,
    Unicode(&'a str),
    NonUnicode,
}

fn root_sparse_gate_env(raw: Option<&std::ffi::OsStr>) -> RootSparseGateEnv<'_> {
    match raw {
        None => RootSparseGateEnv::Absent,
        Some(value) => value
            .to_str()
            .map_or(RootSparseGateEnv::NonUnicode, RootSparseGateEnv::Unicode),
    }
}

/// Resolve the typed sparse-row policy plus sealed environment overrides without
/// touching process-global state. Only an exact raw `"1"` enables and an exact
/// raw `"0"` disables. Any other present value fails closed, even when the typed
/// config is enabled. Explicit zero caps fail closed at the selector/deadline
/// boundary.
#[allow(clippy::too_many_arguments)]
fn resolve_root_sparse_interm_crown_policy(
    config: &BetaCrownConfig,
    gate_env: RootSparseGateEnv<'_>,
    max_dim_env: Option<&str>,
    max_rows_env: Option<&str>,
    max_targets_env: Option<&str>,
    max_secs_env: Option<&str>,
) -> Option<RootSparseIntermCrownPolicy> {
    let enabled = match gate_env {
        RootSparseGateEnv::Absent => config.root_sparse_interm_crown,
        RootSparseGateEnv::Unicode("1") => true,
        RootSparseGateEnv::Unicode(_) | RootSparseGateEnv::NonUnicode => false,
    };
    if !enabled {
        return None;
    }
    Some(RootSparseIntermCrownPolicy {
        max_dim: max_dim_env
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(config.root_sparse_interm_crown_max_dim)
            .min(ROOT_SPARSE_INTERM_ABS_MAX_DIM),
        max_rows: max_rows_env
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(config.root_sparse_interm_crown_max_rows)
            .min(ROOT_SPARSE_INTERM_ABS_MAX_ROWS),
        max_targets: max_targets_env
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(config.root_sparse_interm_crown_max_targets)
            .min(ROOT_SPARSE_INTERM_ABS_MAX_TARGETS),
        max_secs: max_secs_env
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(config.root_sparse_interm_crown_max_secs)
            .min(ROOT_SPARSE_INTERM_ABS_MAX_SECS),
    })
}

fn root_sparse_interm_crown_policy_from_env(
    config: &BetaCrownConfig,
) -> Option<RootSparseIntermCrownPolicy> {
    let gate = std::env::var_os("NY_ROOT_SPARSE_INTERM_CROWN");
    let max_dim = std::env::var("NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM").ok();
    let max_rows = std::env::var("NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS").ok();
    let max_targets = std::env::var("NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS").ok();
    let max_secs = std::env::var("NY_ROOT_SPARSE_INTERM_CROWN_SECS").ok();
    resolve_root_sparse_interm_crown_policy(
        config,
        root_sparse_gate_env(gate.as_deref()),
        max_dim.as_deref(),
        max_rows.as_deref(),
        max_targets.as_deref(),
        max_secs.as_deref(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RootCrownIntermSelection {
    /// Structural production scope: Linear/Gemm producers immediately feeding
    /// a ReLU. This finds cifar100's head without relying on ONNX node names.
    DenseHead,
    /// Legacy diagnostic scope retained for environment-driven experiments.
    All,
    /// Legacy measured high-Δ node-name set.
    Preset,
    /// Legacy explicit ReLU node-name list.
    Explicit(std::collections::HashSet<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RootCrownIntermPolicy {
    selection: RootCrownIntermSelection,
    max_dim: usize,
    max_secs: u64,
}

/// Resolve typed production policy plus legacy environment overrides without
/// reading process-global state. Kept pure so default-off and kill-switch
/// semantics are directly testable.
fn resolve_root_crown_interm_policy(
    config: &BetaCrownConfig,
    gate_env: Option<&str>,
    layers_env: Option<&str>,
    max_dim_env: Option<&str>,
    max_secs_env: Option<&str>,
) -> Option<RootCrownIntermPolicy> {
    let env_forced_on = gate_env.is_some_and(|value| value.trim() == "1");
    let enabled = match gate_env.map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        // Unknown inherited values are not interpreted as a force-on; the typed
        // preset remains authoritative.
        _ => config.root_crown_interm_dense_head,
    };
    if !enabled {
        return None;
    }

    let selection = match layers_env.map(str::trim) {
        Some(value) if value.is_empty() || value.eq_ignore_ascii_case("all") => {
            RootCrownIntermSelection::All
        }
        Some(value)
            if value.eq_ignore_ascii_case("dense-head")
                || value.eq_ignore_ascii_case("dense_head")
                || value.eq_ignore_ascii_case("head") =>
        {
            RootCrownIntermSelection::DenseHead
        }
        Some(value) if value.eq_ignore_ascii_case("preset") => RootCrownIntermSelection::Preset,
        Some(value) => RootCrownIntermSelection::Explicit(
            value
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .collect(),
        ),
        // Preserve the old `NY_ROOT_CROWN_INTERM=1` no-layers behavior (`all`)
        // when it alone force-enables an otherwise-off config. A typed preset
        // selects the production dense-head scope.
        None if env_forced_on && !config.root_crown_interm_dense_head => {
            RootCrownIntermSelection::All
        }
        None => RootCrownIntermSelection::DenseHead,
    };
    let max_dim = max_dim_env
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or({
            if env_forced_on && !config.root_crown_interm_dense_head {
                // Preserve the original env-only experiment contract. Before
                // the typed production preset existed, force-on without an
                // explicit MAXDIM admitted the complete CIFAR conv stack.
                20_000
            } else {
                config.root_crown_interm_max_dim
            }
        });
    let max_secs = max_secs_env
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(config.root_crown_interm_max_secs);
    Some(RootCrownIntermPolicy {
        selection,
        max_dim,
        max_secs,
    })
}

fn root_crown_interm_policy_from_env(config: &BetaCrownConfig) -> Option<RootCrownIntermPolicy> {
    let gate = std::env::var("NY_ROOT_CROWN_INTERM").ok();
    let layers = std::env::var("NY_ROOT_CROWN_INTERM_LAYERS").ok();
    let max_dim = std::env::var("NY_ROOT_CROWN_INTERM_MAXDIM").ok();
    let max_secs = std::env::var("NY_ROOT_CROWN_INTERM_SECS").ok();
    resolve_root_crown_interm_policy(
        config,
        gate.as_deref(),
        layers.as_deref(),
        max_dim.as_deref(),
        max_secs.as_deref(),
    )
}

/// Give the pass at most its configured cap and at most half of the remaining
/// global verifier budget. Expired/zero/tiny slices skip the pass, so the root
/// objective and BaB retain their original sound bounds and remaining wall time.
fn bounded_root_crown_interm_deadline(
    now: std::time::Instant,
    global_deadline: Option<std::time::Instant>,
    max_secs: u64,
) -> Option<std::time::Instant> {
    if max_secs == 0 {
        return None;
    }
    let cap = std::time::Duration::from_secs(max_secs);
    let slice = match global_deadline {
        Some(global) => {
            let remaining = global.checked_duration_since(now)?;
            cap.min(remaining.mul_f32(0.5))
        }
        None => cap,
    };
    // Below this floor a graph walk cannot usefully start. Skipping is the
    // fail-closed outcome: the existing sound box remains untouched.
    if slice < std::time::Duration::from_millis(100) {
        return None;
    }
    now.checked_add(slice)
}

/// High-Δ preset (`NY_ROOT_CROWN_INTERM_LAYERS=preset`): the deep ReLU
/// pre-activations the probe measured as materially looser than the CROWN
/// backward (Relu_13/19/25/31/57) plus the cheap 2048-wide deep blocks
/// (Relu_39/45/51). Names are matched against the ReLU node names in exec order.
const ROOT_CROWN_INTERM_PRESET: &[&str] = &[
    "Relu_13", "Relu_19", "Relu_25", "Relu_31", "Relu_39", "Relu_45", "Relu_51", "Relu_57",
];

/// SHRINK-ONLY per-element intersect of a frozen forward-linear reference box
/// `ref_bt` with a sound CROWN box `crown` (flat `[num_outputs]`, same element
/// count, iterated in matching logical order). For each element:
///   `l_new = max(l_fwd, l_crown)`, `u_new = min(u_fwd, u_crown)`.
/// FAIL-CLOSED and NEVER-WIDEN: if the CROWN endpoints are non-finite/inverted, or
/// the intersect would invert (`l_new > u_new`, disjoint enclosures ⇒ upstream
/// bug or an infeasible domain), the reference element is kept verbatim. The
/// result is therefore always `l_new ∈ [l_fwd, u_fwd]`, `u_new ∈ [l_fwd, u_fwd]`,
/// `l_new ≤ u_new` — a sound tightening of `ref_bt` that never drops a real point
/// (both inputs enclose the reachable set, so does their intersection).
///
/// Returns `(tightened, n_tightened_elems)`, or `None` on element-count mismatch
/// or if the rebuilt tensor is rejected (⇒ caller keeps the reference).
fn shrink_only_intersect(
    ref_bt: &BoundedTensor,
    crown: &BoundedTensor,
) -> Option<(BoundedTensor, usize)> {
    if ref_bt.len() != crown.len() {
        return None;
    }
    let (mut nl, mut nu) = ref_bt.clone().into_parts();
    let mut n_tightened = 0usize;
    for ((l, u), (&cl, &cu)) in nl
        .iter_mut()
        .zip(nu.iter_mut())
        .zip(crown.lower().iter().zip(crown.upper().iter()))
    {
        // Fail-closed: only tighten from a finite, valid CROWN endpoint.
        if !cl.is_finite() || !cu.is_finite() || cl > cu {
            continue;
        }
        let lf = *l;
        let uf = *u;
        let cand_l = lf.max(cl);
        let cand_u = uf.min(cu);
        // Never invert (disjoint boxes ⇒ keep the reference; never widen).
        if cand_l <= cand_u {
            // Shrink-only invariant (both hold by construction of max/min).
            debug_assert!(
                cand_l >= lf && cand_u <= uf,
                "root-crown-interm widened a bound"
            );
            if cand_l > lf || cand_u < uf {
                n_tightened += 1;
            }
            *l = cand_l;
            *u = cand_u;
        }
    }
    BoundedTensor::new_allow_infinite(nl, nu)
        .ok()
        .map(|bt| (bt, n_tightened))
}

/// Count crossing-ReLU (unstable) pre-activation neurons across all ReLU nodes: a
/// neuron is unstable when its pre-activation box straddles 0 (`l < 0 < u`). Used
/// only for the before/after diagnostic log (baseline root count ≈ 1008 / 970).
fn count_unstable_relu_preacts(
    graph: &GraphNetwork,
    bounds: &std::collections::HashMap<String, BoundedTensor>,
) -> usize {
    let order = match graph.exec_order() {
        Ok(o) => o.to_vec(),
        Err(_) => return 0,
    };
    let mut n = 0usize;
    for name in &order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer, crate::Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs.first() else {
            continue;
        };
        let Some(bt) = bounds.get(pre) else {
            continue;
        };
        for (&l, &u) in bt.lower().iter().zip(bt.upper().iter()) {
            if l < 0.0 && u > 0.0 {
                n += 1;
            }
        }
    }
    n
}

/// `#root-crown-interm`: at the ROOT, tighten the frozen forward-linear
/// pre-activation bounds by SHRINK-ONLY intersecting a sound heuristic-α CROWN
/// backward box into each selected intermediate ReLU pre-activation of
/// `bootstrap.initial_node_bounds`. Runs ONCE at the root; the typed production
/// scope is dense-head only, and the tightened map is then Arc-shared to every
/// BaB subdomain by the caller.
///
/// Two-phase (compute-then-apply) so every CROWN backward reads the ORIGINAL frozen
/// bounds (matching the probe's measured widths; avoids in-pass feedback and the
/// immutable/mutable borrow overlap): phase 1 computes each node's CROWN box from
/// the frozen cache; phase 2 shrink-only intersects them in. SOUNDNESS: see
/// `shrink_only_intersect` — never widens, never drops a real point, fail-closed.
/// Returns the number of elements whose lower or upper endpoint actually shrank;
/// zero is the clean signal for no target/no improvement/deadline/refusal.
fn run_root_crown_interm_tighten(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    bootstrap: &mut GraphBabBootstrap,
    policy: &RootCrownIntermPolicy,
    pass_deadline: std::time::Instant,
) -> usize {
    let Some(engine) = engine else {
        eprintln!("[root-crown-interm-tighten] no GPU engine (need the `ny vnncomp` GPU preset); skipping (bounds unchanged)");
        return 0;
    };
    let order: Vec<String> = match graph.exec_order() {
        Ok(o) => o.to_vec(),
        Err(_) => {
            eprintln!(
                "[root-crown-interm-tighten] exec_order unavailable; skipping (bounds unchanged)"
            );
            return 0;
        }
    };
    let max_dim = policy.max_dim;
    let dense_head_targets: std::collections::HashSet<String> = graph
        .fc_head_preactivation_targets(&order)
        .into_iter()
        .collect();
    let want = |relu_name: &str, pre_name: &str| -> bool {
        match &policy.selection {
            RootCrownIntermSelection::DenseHead => dense_head_targets.contains(pre_name),
            RootCrownIntermSelection::All => true,
            RootCrownIntermSelection::Preset => ROOT_CROWN_INTERM_PRESET.contains(&relu_name),
            RootCrownIntermSelection::Explicit(names) => names.contains(relu_name),
        }
    };
    let selection_label = match &policy.selection {
        RootCrownIntermSelection::DenseHead => "dense-head".to_string(),
        RootCrownIntermSelection::All => "all".to_string(),
        RootCrownIntermSelection::Preset => "preset".to_string(),
        RootCrownIntermSelection::Explicit(names) => {
            let mut names: Vec<&str> = names.iter().map(String::as_str).collect();
            names.sort_unstable();
            names.join(",")
        }
    };

    let unstable_before = count_unstable_relu_preacts(graph, &bootstrap.initial_node_bounds);
    eprintln!(
        "[root-crown-interm-tighten] ===== ROOT SHRINK-ONLY CROWN INTERSECT (layers={selection_label} max_dim={max_dim} budget={:.3}s) unstable_before={unstable_before} =====",
        pass_deadline.saturating_duration_since(std::time::Instant::now()).as_secs_f32(),
    );
    let t0 = std::time::Instant::now();

    // Phase 1 — compute every selected node's sound CROWN box from the ORIGINAL
    // frozen bounds (immutable borrow only).
    let mut computed: Vec<(String, BoundedTensor)> = Vec::new();
    {
        let acrown = &bootstrap.initial_node_bounds;
        // One-time Arc view for the sound CROWN backward (see probe above).
        let acrown_arc = build_initial_node_bounds_arc(acrown);
        for name in &order {
            if std::time::Instant::now() >= pass_deadline {
                eprintln!(
                    "[root-crown-interm-tighten] deadline reached before next target -> keep remaining fwd_linear bounds"
                );
                break;
            }
            let Some(node) = graph.nodes.get(name) else {
                continue;
            };
            if !matches!(node.layer, crate::Layer::ReLU(_)) {
                continue;
            }
            let Some(pre) = node.inputs.first() else {
                continue;
            };
            if !want(name, pre) {
                continue;
            }
            let Some(ref_bt) = acrown.get(pre) else {
                continue;
            };
            let pre_dim = ref_bt.len();
            if pre_dim == 0 {
                continue;
            }
            if pre_dim > max_dim {
                eprintln!(
                    "[root-crown-interm-tighten] relu={name} pre={pre} pre_dim={pre_dim} SKIP(>max_dim)"
                );
                continue;
            }
            // Sound heuristic-α CROWN backward (α=None), certified error folded
            // outward; refuses non-finite ⇒ Option. Concretize soundly over the box.
            let crown = crate::beta_crown::engine::graph::propagation::batched::backward_input_relative_bounds_at_node(
                graph, pre, &acrown_arc, input, engine, Some(pass_deadline), None,
            )
            .map(|lb| lb.concretize_sound(input));
            match crown {
                Some(cbox) => computed.push((pre.clone(), cbox)),
                None if std::time::Instant::now() >= pass_deadline => {
                    eprintln!(
                        "[root-crown-interm-tighten] relu={name} pre={pre} deadline/refusal -> keep fwd_linear and stop"
                    );
                    break;
                }
                None => eprintln!(
                    "[root-crown-interm-tighten] relu={name} pre={pre} pre_dim={pre_dim} CROWN REFUSED -> keep fwd_linear"
                ),
            }
        }
    }

    // Phase 2 — shrink-only intersect each CROWN box into the frozen map.
    let mut nodes_tightened = 0usize;
    let mut elements_tightened = 0usize;
    let mut total_fwd_w = 0.0f64;
    let mut total_new_w = 0.0f64;
    for (pre, cbox) in &computed {
        let Some(ref_bt) = bootstrap.initial_node_bounds.get(pre) else {
            continue;
        };
        let (fwd_w, _) = probe_width_total(ref_bt);
        match shrink_only_intersect(ref_bt, cbox) {
            Some((tightened, n_elems)) => {
                let (new_w, _) = probe_width_total(&tightened);
                total_fwd_w += fwd_w;
                total_new_w += new_w;
                if n_elems > 0 {
                    nodes_tightened += 1;
                    elements_tightened += n_elems;
                }
                eprintln!(
                    "[root-crown-interm-tighten] pre={pre} fwd_linear_width={fwd_w:.4} -> intersected_width={new_w:.4} tightened_elems={n_elems}"
                );
                bootstrap.initial_node_bounds.insert(pre.clone(), tightened);
            }
            None => eprintln!(
                "[root-crown-interm-tighten] pre={pre} shape/len mismatch or rebuild rejected -> keep fwd_linear"
            ),
        }
    }

    let unstable_after = count_unstable_relu_preacts(graph, &bootstrap.initial_node_bounds);
    let elapsed = t0.elapsed().as_secs_f32();
    let reduction = total_fwd_w - total_new_w;
    let unstable_delta = (unstable_before as i64) - (unstable_after as i64);
    let n_computed = computed.len();
    eprintln!(
        "[root-crown-interm-tighten] ===== END ({elapsed:.1}s) nodes_tightened={nodes_tightened}/{n_computed} total_width {total_fwd_w:.4} -> {total_new_w:.4} (reduction {reduction:.4}) unstable {unstable_before} -> {unstable_after} (Δ {unstable_delta}) ====="
    );
    elements_tightened
}

#[cfg(test)]
mod root_crown_interm_tests {
    use super::{
        bounded_root_crown_interm_deadline, resolve_root_crown_interm_policy,
        resolve_root_sparse_interm_crown_policy, root_sparse_gate_env, shrink_only_intersect,
        RootCrownIntermSelection, RootSparseGateEnv,
    };
    use crate::BetaCrownConfig;
    use ny_tensor::BoundedTensor;

    fn bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        use ndarray::{ArrayD, IxDyn};
        BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn typed_root_crown_interm_is_default_off_and_dense_head_when_armed() {
        let default = BetaCrownConfig::default();
        assert!(
            resolve_root_crown_interm_policy(&default, None, None, None, None).is_none(),
            "missing preset/env must leave the new root pass off"
        );

        let armed = BetaCrownConfig {
            root_crown_interm_dense_head: true,
            root_crown_interm_max_secs: 3,
            root_crown_interm_max_dim: 321,
            ..BetaCrownConfig::default()
        };
        let policy = resolve_root_crown_interm_policy(&armed, None, None, None, None).unwrap();
        assert_eq!(policy.selection, RootCrownIntermSelection::DenseHead);
        assert_eq!(policy.max_secs, 3);
        assert_eq!(policy.max_dim, 321);
    }

    #[test]
    fn typed_root_sparse_interm_is_default_off_bounded_and_killable() {
        let default = BetaCrownConfig::default();
        assert!(resolve_root_sparse_interm_crown_policy(
            &default,
            RootSparseGateEnv::Absent,
            None,
            None,
            None,
            None,
        )
        .is_none());

        let armed = BetaCrownConfig {
            root_sparse_interm_crown: true,
            root_sparse_interm_crown_max_secs: 3,
            root_sparse_interm_crown_max_dim: 4_096,
            root_sparse_interm_crown_max_rows: 96,
            root_sparse_interm_crown_max_targets: 2,
            ..BetaCrownConfig::default()
        };
        assert!(resolve_root_sparse_interm_crown_policy(
            &armed,
            RootSparseGateEnv::Unicode("0"),
            None,
            None,
            None,
            None,
        )
        .is_none());
        let policy = resolve_root_sparse_interm_crown_policy(
            &armed,
            RootSparseGateEnv::Absent,
            Some("2048"),
            Some("32"),
            Some("1"),
            Some("4"),
        )
        .unwrap();
        assert_eq!(policy.max_dim, 2_048);
        assert_eq!(policy.max_rows, 32);
        assert_eq!(policy.max_targets, 1);
        assert_eq!(policy.max_secs, 4);

        let forced = resolve_root_sparse_interm_crown_policy(
            &default,
            RootSparseGateEnv::Unicode("1"),
            Some("0"),
            Some("0"),
            Some("0"),
            Some("0"),
        )
        .unwrap();
        assert_eq!(forced.max_dim, 0);
        assert_eq!(forced.max_rows, 0);
        assert_eq!(forced.max_targets, 0);
        assert_eq!(forced.max_secs, 0);

        let clamped = resolve_root_sparse_interm_crown_policy(
            &default,
            RootSparseGateEnv::Unicode("1"),
            Some("999999"),
            Some("999999"),
            Some("999999"),
            Some("999999"),
        )
        .unwrap();
        assert_eq!(clamped.max_dim, 8_192);
        assert_eq!(clamped.max_rows, 512);
        assert_eq!(clamped.max_targets, 4);
        assert_eq!(clamped.max_secs, 8);

        for malformed in [" 1", "1 ", "true", "yes", "01", ""] {
            assert!(
                resolve_root_sparse_interm_crown_policy(
                    &armed,
                    RootSparseGateEnv::Unicode(malformed),
                    None,
                    None,
                    None,
                    None,
                )
                .is_none(),
                "present malformed gate {malformed:?} must disable even a typed-on config"
            );
        }
        assert!(
            resolve_root_sparse_interm_crown_policy(
                &armed,
                RootSparseGateEnv::NonUnicode,
                None,
                None,
                None,
                None,
            )
            .is_none(),
            "a present non-Unicode gate must disable even a typed-on config"
        );
        assert_eq!(
            root_sparse_gate_env(None),
            RootSparseGateEnv::Absent,
            "an absent environment gate must defer to typed config"
        );
    }

    #[test]
    fn root_crown_interm_env_retains_force_and_selection_overrides() {
        let armed = BetaCrownConfig {
            root_crown_interm_dense_head: true,
            ..BetaCrownConfig::default()
        };
        assert!(
            resolve_root_crown_interm_policy(&armed, Some("0"), None, None, None).is_none(),
            "NY_ROOT_CROWN_INTERM=0 must remain a production kill switch"
        );

        let off = BetaCrownConfig::default();
        let legacy =
            resolve_root_crown_interm_policy(&off, Some("1"), None, Some("42"), Some("7")).unwrap();
        assert_eq!(legacy.selection, RootCrownIntermSelection::All);
        assert_eq!(legacy.max_dim, 42);
        assert_eq!(legacy.max_secs, 7);

        let legacy_implicit =
            resolve_root_crown_interm_policy(&off, Some("1"), None, None, None).unwrap();
        assert_eq!(legacy_implicit.selection, RootCrownIntermSelection::All);
        assert_eq!(legacy_implicit.max_dim, 20_000);

        let explicit = resolve_root_crown_interm_policy(
            &armed,
            Some("1"),
            Some("Relu_57, custom_relu"),
            None,
            None,
        )
        .unwrap();
        let RootCrownIntermSelection::Explicit(names) = explicit.selection else {
            panic!("explicit env layer list must remain supported")
        };
        assert!(names.contains("Relu_57"));
        assert!(names.contains("custom_relu"));
    }

    #[test]
    fn root_crown_interm_deadline_is_capped_and_failclosed() {
        let now = std::time::Instant::now();
        assert_eq!(bounded_root_crown_interm_deadline(now, None, 0), None);
        assert_eq!(
            bounded_root_crown_interm_deadline(now, Some(now), 2),
            None,
            "expired global deadline must not start the pass"
        );
        assert_eq!(
            bounded_root_crown_interm_deadline(
                now,
                Some(now + std::time::Duration::from_secs(1)),
                2,
            ),
            Some(now + std::time::Duration::from_millis(500)),
            "pass gets at most half the remaining global budget"
        );
        assert_eq!(
            bounded_root_crown_interm_deadline(now, None, 2),
            Some(now + std::time::Duration::from_secs(2)),
            "without a global deadline the typed cap remains authoritative"
        );
    }

    /// Core soundness contract: the intersect is SHRINK-ONLY — every output element
    /// satisfies l_fwd ≤ l_new ≤ u_new ≤ u_fwd (never widens, never inverts).
    #[test]
    fn shrink_only_intersect_never_widens() {
        let fwd = bt(&[-5.0, -5.0, -1.0, 0.0], &[5.0, 5.0, 3.0, 10.0]);
        // CROWN box: tighter on some elems, looser on others (must be ignored when looser).
        let crown = bt(&[-2.0, -9.0, 2.0, 1.0], &[2.0, 9.0, 4.0, 6.0]);
        let (out, n) = shrink_only_intersect(&fwd, &crown).unwrap();
        let (fl, fu) = fwd.lower_upper();
        let (ol, ou) = out.lower_upper();
        for i in 0..4 {
            // never widened past the forward reference
            assert!(ol[i] >= fl[i] - 0.0, "elem {i} lower widened");
            assert!(ou[i] <= fu[i] + 0.0, "elem {i} upper widened");
            assert!(ol[i] <= ou[i], "elem {i} inverted");
        }
        // elem0: [-5,5]∩[-2,2] = [-2,2] tightened both sides
        assert_eq!(ol[0], -2.0);
        assert_eq!(ou[0], 2.0);
        // elem1: crown [-9,9] looser => keep forward [-5,5]
        assert_eq!(ol[1], -5.0);
        assert_eq!(ou[1], 5.0);
        // elem2: [-1,3]∩[2,4] = [2,3] lower tightened
        assert_eq!(ol[2], 2.0);
        assert_eq!(ou[2], 3.0);
        // elem3: [0,10]∩[1,6] = [1,6]
        assert_eq!(ol[3], 1.0);
        assert_eq!(ou[3], 6.0);
        assert_eq!(n, 3); // elems 0,2,3 moved; elem1 unchanged
    }

    /// Fail-closed: non-finite CROWN endpoints keep the forward reference verbatim.
    #[test]
    fn shrink_only_intersect_failclosed_on_nonfinite() {
        let fwd = bt(&[-5.0, -5.0], &[5.0, 5.0]);
        let crown = bt(&[f32::NEG_INFINITY, 1.0], &[f32::INFINITY, 2.0]);
        let (out, n) = shrink_only_intersect(&fwd, &crown).unwrap();
        let (ol, ou) = out.lower_upper();
        // elem0: crown non-finite => keep forward
        assert_eq!(ol[0], -5.0);
        assert_eq!(ou[0], 5.0);
        // elem1: finite crown [1,2] tightens
        assert_eq!(ol[1], 1.0);
        assert_eq!(ou[1], 2.0);
        assert_eq!(n, 1);
    }

    /// Fail-closed: disjoint boxes keep the forward reference (never widen/invert).
    #[test]
    fn shrink_only_intersect_failclosed_on_disjoint() {
        let fwd = bt(&[-5.0], &[-1.0]);
        let crown = bt(&[2.0], &[4.0]); // disjoint from [-5,-1]
        let (out, n) = shrink_only_intersect(&fwd, &crown).unwrap();
        let (ol, ou) = out.lower_upper();
        assert_eq!(ol[0], -5.0);
        assert_eq!(ou[0], -1.0);
        assert_eq!(n, 0);
    }

    /// Element-count mismatch => None (caller keeps the reference).
    #[test]
    fn shrink_only_intersect_len_mismatch_is_none() {
        let fwd = bt(&[-5.0, 1.0], &[5.0, 2.0]);
        let crown = bt(&[0.0], &[1.0]);
        assert!(shrink_only_intersect(&fwd, &crown).is_none());
    }
}
