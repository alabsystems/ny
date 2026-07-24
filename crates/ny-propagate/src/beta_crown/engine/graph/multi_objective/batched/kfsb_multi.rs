// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Wave-batched kFSB branch selection for the multi-objective graph BaB lane
//! (#kfsb-multi, barrier 2 — dark, `NY_MO_KFSB=1`, default OFF).
//!
//! MEASURED MOTIVATION (prop1498 d5 worst child, LP-exact min-of-2-children on
//! 48 candidate splits): exact child evaluation SEPARATES candidates — best
//! split lift +0.012 vs the intercept-argmax's ~0.0 (the depth-invariance
//! plateau). ONE-SIDED verifiers (a child fully verified / infeasible) are
//! scored correctly by the `Min` metric this lane pins: the empty half is
//! `+inf`, so `min(children)` ranks the split by its surviving child (the
//! worst-child descent that measurement selected; see step 4). The
//! multi-objective lane's
//! advisory selector (`select_graph_branch_multi`) never child-evaluates; this
//! module brings the single-objective GPU-kFSB discipline (score → filter →
//! SIMULATE both children → pick by reduce-op) to the wave level:
//!
//! 1. PRE-SCORE each wave domain's unstable neurons with
//!    `compute_graph_babsr_scores_from_bounds`, seeded with that domain's worst
//!    unverified straggler's margin row (objective-directed BaBSR).
//! 2. FILTER to top-k by main score ∪ top-k by backup intercept
//!    (`kfsb_shared::select_graph_kfsb_eval_candidates`, k = `fsb_candidates`),
//!    plus an optional stratified top-1-per-unstable-ReLU-layer quota
//!    (`NY_KFSB_LAYER_QUOTA=1` — the probe showed stem-layer candidates like
//!    Relu_13/Relu_5 matter but never crack the global top-k).
//! 3. SIMULATE: both children per candidate via `with_constraint`, bounded for
//!    the WHOLE WAVE through the existing dense-spec domain-batched backward
//!    (`propagate_crown_with_batched_domains_full_specs`) — single-shot, no
//!    β-opt, ONE spec row per call (children are bucketed by their domain's
//!    straggler row, so a chunk of C children costs one C-domain × 1-spec
//!    backward). The `clip_child_node_bounds` research hook is retained, but its
//!    shared production authority gate is quarantined; simulations currently use
//!    inherited bounds unchanged.
//! 4. PICK per domain: argmax of `kfsb_reduce_op(active_lb, inactive_lb)` on
//!    that domain's straggler row (PINNED to `Min` — a multi-objective domain
//!    verifies iff ALL children verify, so the post-split bound is
//!    `min(children)` and maximizing it is the correct metric; see
//!    `kfsb_multi_reduce_op`), main-score tiebreak. An INFEASIBLE child counts
//!    as `+inf` (that side is empty ⇒ the split is one-sided and free); an
//!    EVAL-FAILED child counts as `-inf`; a candidate with both sides `-inf`
//!    is skipped.
//! 5. COMMIT the winner's already-built children directly (no rebuild) into
//!    the normal child pipeline.
//!
//! M27 optionally reranks the first three one-step roots with complete,
//! branch-specific depth-2 lookahead under
//! `NY_MO_ADAPTIVE_DEPTH_SELECT=1`. It can return only a revalidated root
//! identity; any incomplete metric or identity fault preserves step 4's
//! one-step winner, and private depth-2 leaves never enter the child pipeline.
//!
//! ADVISORY-ONLY ⇒ SOUNDNESS-FREE: everything here only chooses WHICH neuron
//! to split; the committed children flow through the same bounding/verdict
//! pipeline as the advisory path. Any error, miss, or deadline expiry simply
//! drops the domain from the returned map and the caller falls back to
//! `select_graph_branch_multi` — never fail the run.
//!
//! COST per wave: `Σ_d 2·|candidates_d|` simulated children (≤ `2·(2k + L)`
//! per domain, L = unstable-layer count under the quota), each bounded by a
//! 1-spec-row share of a chunked batched backward — vs the main pipeline's
//! `2·|domains|` children × |union unverified specs| rows. With k=7 the
//! selection pass costs roughly `k×(1/S)` of the main child pass (S = live
//! spec rows), i.e. comparable to it on cifar100 where S ≈ 8.

use std::collections::HashMap;
use std::sync::Arc;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::batched_domain::{BatchedDomainOptions, BatchedDomains};
use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
use crate::beta_crown::domain::{
    GraphBabDomain, MultiObjDomainWithUnstable, MultiObjectiveGraphBabDomain,
};
use crate::GraphNetwork;

use super::super::super::super::branching::kfsb_shared::{
    kfsb_reduce, select_graph_kfsb_eval_candidates, GraphKfsbCandidate,
};
use super::super::super::super::BetaCrownVerifier;
use super::super::shared::{build_spec_matrix, spec_bounds_to_vec};
use super::batched_dense_specs::{clip_interm_resnet_enabled, graph_bab_domain_shim};

/// Master gate ENV override (kill switch): `NY_MO_KFSB=1` force-ARMS the
/// wave-batched kFSB selector on the multi-objective lane, `NY_MO_KFSB=0`
/// force-DISARMS it. The self-contained alias `NY_BRANCH_KFSB_CHILDSIM=1` also
/// force-ARMS (so the child-sim scoring at the selector actually runs). Any
/// other value (or unset, no childsim) ⇒ `None`, so the gate falls back to
/// `config.use_kfsb_multi_branching` (the preset opt-in). The env thus
/// overrides the preset in EITHER direction, preserving the A/B kill switch.
fn kfsb_multi_env_override() -> Option<bool> {
    match std::env::var("NY_MO_KFSB").ok().as_deref() {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ if kfsb_childsim_gate_enabled() => Some(true),
        _ => None,
    }
}

/// #branch-kfsb-childsim: a self-contained single switch for the wave-batched
/// kFSB CHILD-SIMULATION branch selector. `NY_BRANCH_KFSB_CHILDSIM=1` arms the
/// exact same lane as `NY_MO_KFSB=1` — it scores each candidate split by the
/// ACTUAL bound on BOTH children (simulated in one wave-batched backward) and
/// picks the argmax worst-child bound, instead of the objective-blind intercept
/// `(-l·u)/(u-l)` proxy the default `select_graph_branch_multi` falls back to
/// under the auto-selected `Kfsb` heuristic.
///
/// MEASURED (cifar100 CIFAR100_resnet_medium prop_4429 @100 s, cuda): the
/// intercept default splits ONE layer (Relu_57) for all 1023 domains, wastes
/// 40 % of splits (lift < 1e-3), and verifies 0 sub-domains; this lane wastes
/// 0 % and verifies 10 — the frontier-worst climbs from −0.965 to −0.79. When
/// the candidate budget is not explicitly overridden (`NY_MO_KFSB_K` unset), it
/// pins `k = 2` — the measured throughput/quality sweet spot (k=2 explores 4×
/// the domains of k=7 at an identical verified count and frontier-worst). It
/// only arms where a kFSB heuristic is active (auto-branching's choice for every
/// high-dim conv net: cifar100 / tinyimagenet / traffic_signs / …). Default OFF
/// ⇒ byte-identical. Advisory-only (chooses WHICH neuron to split; children flow
/// through the same sound bounding/verdict pipeline) ⇒ soundness-free.
fn kfsb_childsim_gate_enabled() -> bool {
    std::env::var("NY_BRANCH_KFSB_CHILDSIM").ok().as_deref() == Some("1")
}

/// Stratified layer quota (`NY_KFSB_LAYER_QUOTA=1`, dark): additionally admit
/// each unstable ReLU layer's top-1 main-score candidate to the eval set.
fn kfsb_layer_quota_enabled() -> bool {
    std::env::var("NY_KFSB_LAYER_QUOTA").ok().as_deref() == Some("1")
}

/// One-line per-wave probe (`NY_MO_KFSB_PROBE=1`).
fn kfsb_probe_enabled() -> bool {
    std::env::var("NY_MO_KFSB_PROBE").ok().as_deref() == Some("1")
}

/// Winner-parity prescore (`NY_MO_KFSB_CACHED_LA=1`, dark): use the exact
/// per-objective lower-A coefficients captured by the domain's preceding CROWN
/// pass.  Unset/`0` preserves the historical fixed-slope proxy byte-for-byte.
fn kfsb_cached_la_enabled() -> bool {
    resolve_kfsb_cached_la_enabled(std::env::var("NY_MO_KFSB_CACHED_LA").ok().as_deref())
}

pub(super) fn resolve_kfsb_cached_la_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Observation-only winner oracle.  It reports every evaluated candidate's
/// direct-C child values plus the `Min` and `Max` winners without changing the
/// configured pick.
fn kfsb_winner_probe_enabled() -> bool {
    std::env::var("NY_MO_KFSB_WINNER_PROBE").ok().as_deref() == Some("1")
}

fn kfsb_winner_probe_domains() -> usize {
    std::env::var("NY_MO_KFSB_WINNER_PROBE_DOMAINS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1)
}

/// Effective reduce op for the wave-batched multi-objective kFSB lane, given
/// the optional `NY_MO_KFSB_REDUCE` A/B override.
///
/// PINNED to `Min`, DELIBERATELY decoupled from `config.kfsb_reduce_op`. A
/// multi-objective domain verifies iff ALL of its children verify, so the
/// post-split bound is `min(children)` and MAXIMIZING that min is the correct
/// branching objective (measured: worst-child descent +0.025/depth under `Min`
/// vs +0.008 for the advisory selector). `config.kfsb_reduce_op` is the
/// α,β-CROWN SINGLE-objective parity knob — every cifar100/relational preset
/// sets `reduceop: max` — so inheriting it here would optimize the WRONG
/// metric; the multi-objective lane ignores it. `NY_MO_KFSB_REDUCE=min|max`
/// still overrides for A/B measurement (any other value ⇒ the pinned `Min`).
pub(super) fn resolve_kfsb_multi_reduce_op(
    env_override: Option<&str>,
) -> crate::beta_crown::KfsbReduceOp {
    match env_override {
        Some("max") => crate::beta_crown::KfsbReduceOp::Max,
        Some("min") => crate::beta_crown::KfsbReduceOp::Min,
        _ => crate::beta_crown::KfsbReduceOp::Min,
    }
}

/// The lane's effective reduce op, reading the `NY_MO_KFSB_REDUCE` A/B env.
fn kfsb_multi_reduce_op() -> crate::beta_crown::KfsbReduceOp {
    resolve_kfsb_multi_reduce_op(std::env::var("NY_MO_KFSB_REDUCE").ok().as_deref())
}

/// Chunk width for the simulation's batched backward calls
/// (`NY_MO_KFSB_CHUNK=<n>`, default 64 — the same width as the main lane's
/// GPU single-pass chunk, bounding both memory and deadline overrun).
fn kfsb_sim_chunk() -> usize {
    std::env::var("NY_MO_KFSB_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

/// Default-off, observation-only adaptive-depth probe. The exact `"1"`
/// spelling is intentional: malformed values fail closed to the historical
/// path.
fn adaptive_depth_shadow_enabled() -> bool {
    resolve_adaptive_depth_shadow_enabled(
        std::env::var("NY_MO_ADAPTIVE_DEPTH_SHADOW").ok().as_deref(),
    )
}

pub(super) fn resolve_adaptive_depth_shadow_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Default-off promotion gate for branch-specific depth-2 selection
/// authority. Only the exact `"1"` spelling arms it; malformed values retain
/// the historical one-step winner.
fn adaptive_depth_select_enabled() -> bool {
    resolve_adaptive_depth_select_enabled(
        std::env::var("NY_MO_ADAPTIVE_DEPTH_SELECT").ok().as_deref(),
    )
}

pub(super) fn resolve_adaptive_depth_select_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Selection authority implies the same bounded private evaluation used by
/// the shadow observer. Shadow-only mode remains telemetry-only.
fn adaptive_depth_evaluation_enabled() -> bool {
    resolve_adaptive_depth_evaluation_enabled(
        std::env::var("NY_MO_ADAPTIVE_DEPTH_SHADOW").ok().as_deref(),
        std::env::var("NY_MO_ADAPTIVE_DEPTH_SELECT").ok().as_deref(),
    )
}

pub(super) fn resolve_adaptive_depth_evaluation_enabled(
    shadow: Option<&str>,
    selection: Option<&str>,
) -> bool {
    resolve_adaptive_depth_shadow_enabled(shadow)
        || resolve_adaptive_depth_select_enabled(selection)
}

/// The whole shadow observation gets at most one second and may start only
/// while five seconds remain reserved for the authoritative BaB loop. Both its
/// per-child BaBSR rescoring and dense backward receive this separate deadline
/// through a fresh observer verifier; it never changes
/// `self.config.alpha_config.deadline`.
const ADAPTIVE_DEPTH_SHADOW_BUDGET: std::time::Duration = std::time::Duration::from_secs(1);
const ADAPTIVE_DEPTH_AUTHORITY_RESERVE: std::time::Duration = std::time::Duration::from_secs(5);

pub(super) fn adaptive_depth_shadow_deadline(
    now: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
) -> Option<std::time::Instant> {
    let shadow_deadline = now.checked_add(ADAPTIVE_DEPTH_SHADOW_BUDGET)?;
    if let Some(authority_deadline) = authority_deadline {
        let latest_finish = authority_deadline.checked_sub(ADAPTIVE_DEPTH_AUTHORITY_RESERVE)?;
        if shadow_deadline > latest_finish {
            return None;
        }
    }
    Some(shadow_deadline)
}

/// Whether optional shadow work may continue right now.  The private
/// one-second deadline bounds the whole observer (including BaBSR rescoring),
/// while the second condition prevents any new side from consuming the five
/// seconds reserved for authoritative search.
pub(super) fn adaptive_depth_shadow_budget_available(
    now: std::time::Instant,
    shadow_deadline: std::time::Instant,
    authority_deadline: Option<std::time::Instant>,
) -> bool {
    now < shadow_deadline
        && authority_deadline.is_none_or(|deadline| {
            now.checked_add(ADAPTIVE_DEPTH_AUTHORITY_RESERVE)
                .is_some_and(|reserved_until| reserved_until < deadline)
        })
}

/// Deterministically rank unique candidates by their already-computed one-step
/// child score. No new bounds are produced here: this is a pure view over the
/// scored kFSB simulations. Exact score, main prescore, node name, neuron index,
/// then original position form a total deterministic ordering.
pub(super) fn rank_adaptive_depth_candidates(
    candidates: &[GraphKfsbCandidate],
    side_values: &[(f32, f32)],
    reduce_op: crate::beta_crown::KfsbReduceOp,
) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = candidates
        .iter()
        .zip(side_values)
        .enumerate()
        .filter_map(|(idx, (_candidate, &(active, inactive)))| {
            if active == f32::NEG_INFINITY && inactive == f32::NEG_INFINITY {
                return None;
            }
            let score = kfsb_reduce(reduce_op, active, inactive);
            (!score.is_nan()).then_some((idx, score))
        })
        .collect();
    ranked.sort_by(|(a_idx, a_score), (b_idx, b_score)| {
        let a = &candidates[*a_idx];
        let b = &candidates[*b_idx];
        let a_main = if a.main_score.is_nan() {
            f32::NEG_INFINITY
        } else {
            a.main_score
        };
        let b_main = if b.main_score.is_nan() {
            f32::NEG_INFINITY
        } else {
            b.main_score
        };
        b_score
            .total_cmp(a_score)
            .then_with(|| b_main.total_cmp(&a_main))
            .then_with(|| a.node_name.cmp(&b.node_name))
            .then_with(|| a.neuron_idx.cmp(&b.neuron_idx))
            .then_with(|| a_idx.cmp(b_idx))
    });

    let mut seen: std::collections::HashSet<(String, usize)> = std::collections::HashSet::new();
    ranked.retain(|(idx, _)| {
        let candidate = &candidates[*idx];
        seen.insert((candidate.node_name.clone(), candidate.neuron_idx))
    });
    ranked
}

/// Build the fixed three-root authority portfolio without changing the
/// historical one-step tie contract. The actual all-candidate
/// `pick_kfsb_candidate` winner must be one of the captured first three roots
/// and fully eligible; it is placed at rank zero. The remaining ranks follow
/// the exact deterministic order used by shadow telemetry. If the historical
/// winner is outside capture, or any three-root portfolio cannot be completed,
/// authority declines.
pub(super) fn rank_adaptive_depth_authority_portfolio(
    candidates: &[GraphKfsbCandidate],
    side_values: &[(f32, f32)],
    captured_eligible: &[bool],
    reduce_op: crate::beta_crown::KfsbReduceOp,
) -> Option<Vec<(usize, f32)>> {
    if candidates.len() != side_values.len()
        || candidates.len() < ADAPTIVE_DEPTH_SHADOW_ROOTS
        || captured_eligible.len() != ADAPTIVE_DEPTH_SHADOW_ROOTS
    {
        return None;
    }
    let (historical_idx, historical_score, _) =
        pick_kfsb_candidate(candidates, side_values.iter().copied(), reduce_op)?;
    if historical_idx >= ADAPTIVE_DEPTH_SHADOW_ROOTS || !captured_eligible[historical_idx] {
        return None;
    }

    let mut portfolio = Vec::with_capacity(ADAPTIVE_DEPTH_SHADOW_ROOTS);
    portfolio.push((historical_idx, historical_score));
    let historical = &candidates[historical_idx];
    for (candidate_idx, score) in rank_adaptive_depth_candidates(
        &candidates[..ADAPTIVE_DEPTH_SHADOW_ROOTS],
        &side_values[..ADAPTIVE_DEPTH_SHADOW_ROOTS],
        reduce_op,
    ) {
        let candidate = &candidates[candidate_idx];
        if !captured_eligible[candidate_idx]
            || (candidate.node_name == historical.node_name
                && candidate.neuron_idx == historical.neuron_idx)
        {
            continue;
        }
        portfolio.push((candidate_idx, score));
        if portfolio.len() == ADAPTIVE_DEPTH_SHADOW_ROOTS {
            break;
        }
    }
    (portfolio.len() == ADAPTIVE_DEPTH_SHADOW_ROOTS).then_some(portfolio)
}

/// A truth-table leaf exists only inside the observer. It is never inserted in
/// the authoritative domain map or queue.
enum AdaptiveDepthShadowLeaf {
    Feasible(Box<MultiObjectiveGraphBabDomain>),
    Infeasible,
    Failed,
}

pub(super) type AdaptiveDepthShadowNodeBounds = HashMap<String, Arc<BoundedTensor>>;

const ADAPTIVE_DEPTH_SHADOW_ROOTS: usize = 3;
const ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS: usize = 2 * ADAPTIVE_DEPTH_SHADOW_ROOTS;

struct AdaptiveDepthShadowCaptureSlot {
    sim_index: usize,
    node_bounds: Option<AdaptiveDepthShadowNodeBounds>,
}

/// Fixed-size capture store: both its index metadata and retained bound maps
/// are O(6), independent of the authoritative wave/frontier size.
pub(super) struct AdaptiveDepthShadowCapture {
    prep_index: usize,
    slots: [Option<AdaptiveDepthShadowCaptureSlot>; ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS],
}

impl AdaptiveDepthShadowCapture {
    pub(super) fn from_sim_indices<I>(prep_index: usize, sim_indices: I, sims_len: usize) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        let mut slots: [Option<AdaptiveDepthShadowCaptureSlot>;
            ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS] = std::array::from_fn(|_| None);
        let mut slot_index = 0;
        let mut sim_indices = sim_indices.into_iter();
        while slot_index < ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS {
            let Some(sim_index) = sim_indices.next() else {
                break;
            };
            if sim_index >= sims_len
                || slots[..slot_index].iter().any(|slot| {
                    slot.as_ref()
                        .is_some_and(|slot| slot.sim_index == sim_index)
                })
            {
                continue;
            }
            slots[slot_index] = Some(AdaptiveDepthShadowCaptureSlot {
                sim_index,
                node_bounds: None,
            });
            slot_index += 1;
        }
        Self { prep_index, slots }
    }

    #[cfg(test)]
    pub(super) fn planned_slot_count(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    #[cfg(test)]
    pub(super) const fn slot_capacity() -> usize {
        ADAPTIVE_DEPTH_SHADOW_CAPTURE_SLOTS
    }

    #[cfg(test)]
    pub(super) fn captured_map_count(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|slot| slot.node_bounds.is_some())
            .count()
    }

    pub(super) fn contains_sim(&self, sim_index: usize) -> bool {
        self.slots
            .iter()
            .flatten()
            .any(|slot| slot.sim_index == sim_index)
    }

    pub(super) fn insert_node_bounds(
        &mut self,
        sim_index: usize,
        node_bounds: AdaptiveDepthShadowNodeBounds,
    ) {
        if let Some(slot) = self
            .slots
            .iter_mut()
            .flatten()
            .find(|slot| slot.sim_index == sim_index)
        {
            slot.node_bounds = Some(node_bounds);
        }
    }

    fn node_bounds(&self, sim_index: usize) -> Option<&AdaptiveDepthShadowNodeBounds> {
        self.slots
            .iter()
            .flatten()
            .find(|slot| slot.sim_index == sim_index)
            .and_then(|slot| slot.node_bounds.as_ref())
    }
}

pub(super) fn clear_shadow_cached_las(child: &mut MultiObjectiveGraphBabDomain) -> bool {
    let count = child.objective_bounds().len();
    child.set_cached_las(vec![None; count]).is_ok()
}

fn split_adaptive_depth_shadow_leaf(
    graph: &GraphNetwork,
    domain: &MultiObjectiveGraphBabDomain,
    candidate: &GraphKfsbCandidate,
    score: f32,
    verify_upper: bool,
    thresholds: &[f32],
) -> [AdaptiveDepthShadowLeaf; 2] {
    let build = |is_active| {
        let constraint = GraphNeuronConstraint {
            node_name: candidate.node_name.clone(),
            neuron_idx: candidate.neuron_idx,
            is_active,
            score,
        };
        match domain.with_constraint(graph, constraint, verify_upper, thresholds) {
            Ok(Some(mut child)) => {
                // `with_constraint` inherits lA. Drop the potentially ~44 MiB
                // cache immediately; the shadow dense path never consumes it.
                if clear_shadow_cached_las(&mut child) {
                    AdaptiveDepthShadowLeaf::Feasible(Box::new(child))
                } else {
                    AdaptiveDepthShadowLeaf::Failed
                }
            }
            Ok(None) => AdaptiveDepthShadowLeaf::Infeasible,
            Err(ref error) if error.is_infeasible_domain() => AdaptiveDepthShadowLeaf::Infeasible,
            Err(_) => AdaptiveDepthShadowLeaf::Failed,
        }
    };
    [build(true), build(false)]
}

fn adaptive_depth_shadow_leaf_from_side(
    side: &SideSlot,
    sims: &[Option<MultiObjectiveGraphBabDomain>],
    capture: &AdaptiveDepthShadowCapture,
) -> AdaptiveDepthShadowLeaf {
    match side {
        SideSlot::Infeasible => AdaptiveDepthShadowLeaf::Infeasible,
        SideSlot::Failed => AdaptiveDepthShadowLeaf::Failed,
        SideSlot::Sim(index) => {
            let Some(mut child) = sims.get(*index).and_then(Option::as_ref).cloned() else {
                return AdaptiveDepthShadowLeaf::Failed;
            };
            let Some(node_bounds) = capture.node_bounds(*index) else {
                return AdaptiveDepthShadowLeaf::Failed;
            };

            // The one-step kFSB simulation already paid for a constrained
            // forward fixpoint. Install that cache only on this PRIVATE clone;
            // the authoritative child in `sims` remains byte-for-byte
            // untouched and is still what the normal selector may commit.
            child.node_bounds = node_bounds.clone();
            child.delta_pre_nodes.clear();
            if clear_shadow_cached_las(&mut child) {
                AdaptiveDepthShadowLeaf::Feasible(Box::new(child))
            } else {
                AdaptiveDepthShadowLeaf::Failed
            }
        }
    }
}

fn expand_adaptive_depth_shadow_leaves_branch_specific<F, B>(
    graph: &GraphNetwork,
    leaves: Vec<AdaptiveDepthShadowLeaf>,
    verify_upper: bool,
    thresholds: &[f32],
    mut select: F,
    mut budget_available: B,
) -> (Vec<AdaptiveDepthShadowLeaf>, Vec<String>)
where
    F: FnMut(&MultiObjectiveGraphBabDomain) -> ny_core::Result<Option<GraphKfsbCandidate>>,
    B: FnMut() -> bool,
{
    let mut next = Vec::with_capacity(leaves.len() * 2);
    let mut choices = Vec::with_capacity(leaves.len());
    for (side, leaf) in leaves.into_iter().enumerate() {
        let label = if side == 0 { "active" } else { "inactive" };
        match leaf {
            AdaptiveDepthShadowLeaf::Feasible(domain) => {
                if !budget_available() {
                    choices.push(format!("{label}=budget-expired"));
                    next.push(AdaptiveDepthShadowLeaf::Failed);
                    continue;
                }
                match select(&domain) {
                    Ok(Some(candidate)) => {
                        if !budget_available() {
                            choices.push(format!("{label}=budget-expired"));
                            next.push(AdaptiveDepthShadowLeaf::Failed);
                            continue;
                        }
                        choices.push(format!(
                            "{label}={}:{}@{:.6}",
                            candidate.node_name, candidate.neuron_idx, candidate.main_score
                        ));
                        let children = split_adaptive_depth_shadow_leaf(
                            graph,
                            &domain,
                            &candidate,
                            candidate.main_score,
                            verify_upper,
                            thresholds,
                        );
                        if budget_available() {
                            next.extend(children);
                        } else {
                            // Child construction crossed the private deadline.
                            // Discard both private children and poison the side
                            // rather than publishing an incomplete metric.
                            if let Some(choice) = choices.last_mut() {
                                *choice = format!("{label}=budget-expired");
                            }
                            next.push(AdaptiveDepthShadowLeaf::Failed);
                        }
                    }
                    Ok(None) if budget_available() => {
                        // No remaining unstable ReLU: this side is already a
                        // terminal depth-1 leaf. Keeping it once (rather than
                        // fabricating duplicate children) preserves its bound
                        // in the depth-2 aggregate.
                        choices.push(format!("{label}=terminal"));
                        next.push(AdaptiveDepthShadowLeaf::Feasible(domain));
                    }
                    Ok(None) => {
                        choices.push(format!("{label}=budget-expired"));
                        next.push(AdaptiveDepthShadowLeaf::Failed);
                    }
                    Err(ref error) if error.is_deadline_exceeded() => {
                        choices.push(format!("{label}=budget-expired"));
                        next.push(AdaptiveDepthShadowLeaf::Failed);
                    }
                    Err(_) => {
                        // A scoring failure is not evidence that a branch is
                        // terminal. Poison this root metric so the observer
                        // cannot mistake an incomplete depth-2 tree for an
                        // improvement.
                        choices.push(format!("{label}=score-failed"));
                        next.push(AdaptiveDepthShadowLeaf::Failed);
                    }
                }
            }
            AdaptiveDepthShadowLeaf::Infeasible => {
                choices.push(format!("{label}=empty"));
                next.push(AdaptiveDepthShadowLeaf::Infeasible);
            }
            AdaptiveDepthShadowLeaf::Failed => {
                choices.push(format!("{label}=failed"));
                next.push(AdaptiveDepthShadowLeaf::Failed);
            }
        }
    }
    (next, choices)
}

#[derive(Default)]
pub(super) struct AdaptiveDepthShadowMetrics {
    pub(super) expected: usize,
    pub(super) infeasible: usize,
    pub(super) bounded: usize,
    pub(super) verified: usize,
    pub(super) surviving: usize,
    pub(super) failures: usize,
    pub(super) post_min: f32,
}

impl AdaptiveDepthShadowMetrics {
    fn from_leaves(leaves: &[AdaptiveDepthShadowLeaf]) -> Self {
        let infeasible = leaves
            .iter()
            .filter(|leaf| matches!(leaf, AdaptiveDepthShadowLeaf::Infeasible))
            .count();
        let failures = leaves
            .iter()
            .filter(|leaf| matches!(leaf, AdaptiveDepthShadowLeaf::Failed))
            .count();
        Self {
            expected: leaves.len(),
            infeasible,
            // Empty leaves are resolved/verified by construction.
            verified: infeasible,
            failures,
            post_min: f32::INFINITY,
            ..Self::default()
        }
    }

    fn record_bounds(
        &mut self,
        bounds: Option<(f32, f32)>,
        config: &crate::beta_crown::BetaCrownConfig,
        threshold: f32,
    ) {
        let Some((lower, upper)) = bounds.filter(|(lower, upper)| {
            lower.is_finite() && upper.is_finite() && threshold.is_finite()
        }) else {
            self.failures += 1;
            return;
        };
        self.bounded += 1;
        self.post_min = self
            .post_min
            .min(config.child_bound_value(Some((lower, upper))));
        if config.domain_is_verified(lower, upper, threshold) {
            self.verified += 1;
        } else {
            self.surviving += 1;
        }
    }

    fn post_min_and_lift(&self, parent: f32) -> (f32, f32) {
        if self.failures > 0 || self.bounded + self.infeasible + self.failures != self.expected {
            return (f32::NAN, f32::NAN);
        }
        if self.bounded == 0 && self.infeasible == self.expected {
            return (f32::INFINITY, f32::INFINITY);
        }
        let lift = if self.post_min.is_finite() && parent.is_finite() {
            self.post_min - parent
        } else {
            f32::NAN
        };
        (self.post_min, lift)
    }

    /// Return the authoritative max-worst-leaf score only for a structurally
    /// complete root tree. Infinite scores are accepted solely for the exact
    /// all-infeasible certificate; every other accepted score is finite.
    fn complete_authority_score(&self) -> Option<f32> {
        let accounted = self.bounded.checked_add(self.infeasible)?;
        let classified = self.verified.checked_add(self.surviving)?;
        if self.expected == 0
            || self.failures != 0
            || accounted != self.expected
            || classified != self.expected
        {
            return None;
        }
        if self.bounded == 0 && self.infeasible == self.expected && self.post_min == f32::INFINITY {
            return Some(f32::INFINITY);
        }
        (self.bounded > 0 && self.post_min.is_finite()).then_some(self.post_min)
    }
}

/// Select one root rank only when the fixed three-root portfolio is complete.
/// Exact score ties deliberately retain the earlier one-step rank.
pub(super) fn select_complete_adaptive_depth_rank(
    metrics: &[AdaptiveDepthShadowMetrics],
) -> Option<usize> {
    if metrics.len() != ADAPTIVE_DEPTH_SHADOW_ROOTS {
        return None;
    }
    let mut scores = metrics
        .iter()
        .map(AdaptiveDepthShadowMetrics::complete_authority_score);
    let mut best_rank = 0;
    let mut best_score = scores.next()??;
    for (rank, score) in scores.enumerate() {
        let score = score?;
        if score > best_score {
            best_rank = rank + 1;
            best_score = score;
        }
    }
    Some(best_rank)
}

fn mark_adaptive_depth_dense_failure(metrics: &mut [AdaptiveDepthShadowMetrics], owners: &[usize]) {
    for &owner in owners {
        if let Some(metric) = metrics.get_mut(owner) {
            metric.failures += 1;
        }
    }
}

/// The committed selection for one wave domain: the winner split's children,
/// already built via `with_constraint` (0..=2 entries — an infeasible half is
/// simply absent, mirroring the advisory path's `Ok(None)` skip).
pub(in crate::beta_crown::engine::graph) type KfsbMultiChildren =
    Vec<(MultiObjectiveGraphBabDomain, bool)>;

/// Where a candidate side's value comes from during simulation.
pub(super) enum SideSlot {
    /// `with_constraint` proved the half-space empty — the side is resolved.
    Infeasible,
    /// Simulated child at `sims[i]`; its bound lands in `sim_values[i]`.
    Sim(usize),
    /// Child construction failed — the side counts as `-inf` at pick time.
    Failed,
}

/// Per-domain preparation: straggler row + filtered candidates + built children.
pub(super) struct DomainPrep {
    /// Position in `domains_with_unstable` (NOT the parent result index).
    pub(super) slot: usize,
    /// Straggler objective index (drives both the score seed and the pick row).
    pub(super) straggler: usize,
    /// Number of unstable candidates whose pre-score came from cached lA.
    pub(super) cached_score_candidates: usize,
    pub(super) candidates: Vec<GraphKfsbCandidate>,
    /// Per candidate: [active, inactive] side slots.
    pub(super) sides: Vec<[SideSlot; 2]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaptiveDepthAuthoritySide {
    Infeasible,
    Sim(usize),
}

/// The only value allowed to cross from private depth-2 evaluation back to
/// authoritative branch selection. It identifies one already-built root and
/// its exact preparation; it contains no private leaf or private bound.
#[derive(Debug)]
pub(super) struct AdaptiveDepthAuthoritySelection {
    pub(super) prep_index: usize,
    pub(super) parent_index: usize,
    pub(super) slot: usize,
    pub(super) straggler: usize,
    pub(super) candidates_len: usize,
    pub(super) sides_len: usize,
    pub(super) candidate_index: usize,
    pub(super) node_name: String,
    pub(super) neuron_idx: usize,
    pub(super) main_score_bits: u32,
    pub(super) backup_score_bits: u32,
    sides: [AdaptiveDepthAuthoritySide; 2],
}

fn adaptive_depth_authority_side(side: &SideSlot) -> Option<AdaptiveDepthAuthoritySide> {
    match side {
        SideSlot::Infeasible => Some(AdaptiveDepthAuthoritySide::Infeasible),
        SideSlot::Sim(index) => Some(AdaptiveDepthAuthoritySide::Sim(*index)),
        SideSlot::Failed => None,
    }
}

pub(super) fn adaptive_depth_authority_identity(
    prep_index: usize,
    parent_index: usize,
    prep: &DomainPrep,
    candidate_index: usize,
) -> Option<AdaptiveDepthAuthoritySelection> {
    let candidate = prep.candidates.get(candidate_index)?;
    let sides = prep.sides.get(candidate_index)?;
    Some(AdaptiveDepthAuthoritySelection {
        prep_index,
        parent_index,
        slot: prep.slot,
        straggler: prep.straggler,
        candidates_len: prep.candidates.len(),
        sides_len: prep.sides.len(),
        candidate_index,
        node_name: candidate.node_name.clone(),
        neuron_idx: candidate.neuron_idx,
        main_score_bits: candidate.main_score.to_bits(),
        backup_score_bits: candidate.backup_score.to_bits(),
        sides: [
            adaptive_depth_authority_side(&sides[0])?,
            adaptive_depth_authority_side(&sides[1])?,
        ],
    })
}

/// Validate the private evaluator's identity against the untouched current
/// preparation and require every authoritative first-level child to remain
/// available. This function does not take or mutate any simulation domain.
pub(super) fn resolve_adaptive_depth_authority_candidate(
    selection: &AdaptiveDepthAuthoritySelection,
    prep_index: usize,
    parent_index: usize,
    prep: &DomainPrep,
    sim_values: &[Option<f32>],
    sims: &[Option<MultiObjectiveGraphBabDomain>],
    reduce_op: crate::beta_crown::KfsbReduceOp,
) -> Option<(usize, f32)> {
    if selection.prep_index != prep_index
        || selection.parent_index != parent_index
        || selection.slot != prep.slot
        || selection.straggler != prep.straggler
        || selection.candidates_len != prep.candidates.len()
        || selection.sides_len != prep.sides.len()
    {
        return None;
    }
    let candidate = prep.candidates.get(selection.candidate_index)?;
    let sides = prep.sides.get(selection.candidate_index)?;
    if selection.node_name != candidate.node_name
        || selection.neuron_idx != candidate.neuron_idx
        || selection.main_score_bits != candidate.main_score.to_bits()
        || selection.backup_score_bits != candidate.backup_score.to_bits()
        || selection.sides[0] != adaptive_depth_authority_side(&sides[0])?
        || selection.sides[1] != adaptive_depth_authority_side(&sides[1])?
    {
        return None;
    }

    let side_value = |side: AdaptiveDepthAuthoritySide| -> Option<f32> {
        match side {
            AdaptiveDepthAuthoritySide::Infeasible => Some(f32::INFINITY),
            AdaptiveDepthAuthoritySide::Sim(index) => {
                sims.get(index)?.as_ref()?;
                sim_values.get(index).copied().flatten()
            }
        }
    };
    let active = side_value(selection.sides[0])?;
    let inactive = side_value(selection.sides[1])?;
    if active.is_nan()
        || inactive.is_nan()
        || (active == f32::NEG_INFINITY && inactive == f32::NEG_INFINITY)
    {
        return None;
    }
    let score = kfsb_reduce(reduce_op, active, inactive);
    (!score.is_nan()).then_some((selection.candidate_index, score))
}

/// Pick one child-evaluated kFSB candidate with the historical deterministic
/// main-score tiebreak.  Kept pure so the winner oracle can price `Min` and
/// winner-compatible `Max` on the exact same child values.
pub(super) fn pick_kfsb_candidate<I>(
    candidates: &[GraphKfsbCandidate],
    side_values: I,
    reduce_op: crate::beta_crown::KfsbReduceOp,
) -> Option<(usize, f32, f32)>
where
    I: IntoIterator<Item = (f32, f32)>,
{
    let mut best: Option<(usize, f32, f32)> = None; // (candidate, score, main)
    for (ci, ((active_val, inactive_val), candidate)) in
        side_values.into_iter().zip(candidates).enumerate()
    {
        if active_val == f32::NEG_INFINITY && inactive_val == f32::NEG_INFINITY {
            continue;
        }
        let score = kfsb_reduce(reduce_op, active_val, inactive_val);
        if score.is_nan() {
            continue;
        }
        let main = candidate.main_score;
        let is_better = best
            .as_ref()
            .map(|(_, best_score, best_main)| {
                score > *best_score + 1e-6
                    || ((score - *best_score).abs() <= 1e-6
                        && !main.is_nan()
                        && (best_main.is_nan() || main > *best_main))
            })
            .unwrap_or(true);
        if is_better {
            best = Some((ci, score, main));
        }
    }
    best
}

impl BetaCrownVerifier {
    /// Whether the wave-batched kFSB selector should run for this config:
    /// ARMED (env `NY_MO_KFSB` override, else `config.use_kfsb_multi_branching`)
    /// AND a kFSB heuristic AND a nonzero candidate budget.
    ///
    /// Tri-state arming: `NY_MO_KFSB=1` forces on, `NY_MO_KFSB=0` forces off
    /// (kill switch), and when the env is unset the preset field
    /// `use_kfsb_multi_branching` decides (default false ⇒ byte-identical to the
    /// pre-#kfsb-multi advisory path everywhere it is not a cifar100 preset).
    pub(in crate::beta_crown::engine::graph) fn kfsb_multi_wave_enabled(&self) -> bool {
        let armed = kfsb_multi_env_override().unwrap_or(self.config.use_kfsb_multi_branching);
        armed
            && matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::Kfsb | BranchingHeuristic::KfsbInterceptOnly
            )
            && self.config.fsb_candidates > 0
    }

    /// Wave-batched kFSB branch selection + child commit (#kfsb-multi).
    ///
    /// Returns `parent_idx → committed children` for every wave domain whose
    /// selection completed; misses fall back to the advisory path in the
    /// caller. INFALLIBLE by design (never fails the run): every internal
    /// error just drops the affected domain from the map.
    // Justification: the selector threads the same verification context as the
    // caller (graph, wave, relu nodes, objectives, thresholds, engine).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn select_graph_branch_kfsb_multi_batched(
        &self,
        graph: &GraphNetwork,
        domains_with_unstable: &[MultiObjDomainWithUnstable<'_>],
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: &dyn GemmEngine,
    ) -> HashMap<usize, KfsbMultiChildren> {
        // A/B isolation overrides (measurement-only): NY_MO_KFSB_K trims the
        // candidate count (simulation cost scales with it) and NY_MO_KFSB_REDUCE
        // (`min`|`max`) overrides the lane's PINNED `Min` reduce op (see
        // `resolve_kfsb_multi_reduce_op`) — the worst-child (`Min`) metric is
        // both theoretically correct and measured-best, so the knob only exists
        // to isolate selection quality from simulation cost without a preset edit.
        // Candidate budget: explicit `NY_MO_KFSB_K` always wins. Otherwise the
        // self-contained `NY_BRANCH_KFSB_CHILDSIM` switch pins the measured
        // throughput/quality sweet spot (k=2 — 4× the domains of k=7 at an
        // identical verified count on cifar100), while the legacy `NY_MO_KFSB`
        // path keeps `fsb_candidates` (byte-identical to before this gate).
        let k = std::env::var("NY_MO_KFSB_K")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| {
                if kfsb_childsim_gate_enabled() {
                    2
                } else {
                    self.config.fsb_candidates
                }
            });
        if k == 0 || domains_with_unstable.is_empty() {
            return HashMap::new();
        }
        let layer_quota = kfsb_layer_quota_enabled();
        // Read the dark gate once per wave.  In particular, the disabled hot
        // path below enters the historical proxy computation directly; it
        // neither consults a domain cache nor rebuilds the proxy score map.
        let cached_la_enabled = kfsb_cached_la_enabled();

        // ── 1+2+3a: per-domain pre-score, filter, and child construction ──
        // (parallel; the score backward dominates this stage's cost).
        let mut sims: Vec<Option<MultiObjectiveGraphBabDomain>> = Vec::new();
        let mut sim_owner: Vec<(usize, usize)> = Vec::new(); // (prep index, straggler row)
        let preps_raw: Vec<Option<(DomainPrep, Vec<(usize, MultiObjectiveGraphBabDomain)>)>> =
            domains_with_unstable
                .par_iter()
                .enumerate()
                .map(|(slot, (_idx, domain, unstable))| {
                    self.kfsb_multi_prepare_domain(
                        graph,
                        slot,
                        domain,
                        unstable,
                        objectives,
                        thresholds,
                        k,
                        layer_quota,
                        cached_la_enabled,
                    )
                })
                .collect();
        let mut preps: Vec<DomainPrep> = Vec::new();
        for prep in preps_raw.into_iter().flatten() {
            let (mut prep, children) = prep;
            // Renumber this prep's local sim indices into the wave-global list.
            let base = sims.len();
            for sides in &mut prep.sides {
                for side in sides.iter_mut() {
                    if let SideSlot::Sim(local) = side {
                        *local += base;
                    }
                }
            }
            let prep_index = preps.len();
            for (_local, child) in children {
                sim_owner.push((prep_index, prep.straggler));
                sims.push(Some(child));
            }
            preps.push(prep);
        }
        if preps.is_empty() {
            return HashMap::new();
        }

        // ── 3b: wave-batched simulation, bucketed by straggler row so every
        // batched call carries exactly ONE spec row. ──
        let mut sim_values: Vec<Option<f32>> = vec![None; sims.len()];
        // The branch-specific depth-2 observer needs each shortlisted
        // one-step child's constrained-forward fixpoint to make a genuinely
        // child-dependent second choice. Pick ONE deterministic wave domain
        // and retain ONLY its first three BaBSR-preselected roots (at most six
        // child maps). Retaining every result map in a large frontier would be
        // an unnecessary gate-only memory spike. These maps are never installed
        // on `sims`, whose entries remain the authoritative children committed
        // below.
        let capture_adaptive_depth = adaptive_depth_evaluation_enabled()
            && !self
                .adaptive_depth_shadow_fired
                .load(std::sync::atomic::Ordering::Relaxed);
        let mut adaptive_depth_capture = if capture_adaptive_depth {
            preps
                .iter()
                .position(|prep| {
                    prep.candidates.len() >= ADAPTIVE_DEPTH_SHADOW_ROOTS
                        && prep.sides.len() >= ADAPTIVE_DEPTH_SHADOW_ROOTS
                })
                .map(|prep_index| {
                    let sim_indices = preps[prep_index]
                        .sides
                        .iter()
                        .take(ADAPTIVE_DEPTH_SHADOW_ROOTS)
                        .flat_map(|sides| sides.iter())
                        .filter_map(|side| match side {
                            SideSlot::Sim(index) => Some(*index),
                            SideSlot::Infeasible | SideSlot::Failed => None,
                        });
                    AdaptiveDepthShadowCapture::from_sim_indices(
                        prep_index,
                        sim_indices,
                        sims.len(),
                    )
                })
        } else {
            None
        };
        let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &(_, row)) in sim_owner.iter().enumerate() {
            buckets.entry(row).or_default().push(i);
        }
        let chunk_size = kfsb_sim_chunk();
        let clip = clip_interm_resnet_enabled();
        'buckets: for (row, members) in buckets {
            let Some(spec_matrix) = build_spec_matrix(&[objectives[row].clone()]) else {
                continue;
            };
            for chunk in members.chunks(chunk_size) {
                // Deadline between chunks: unprocessed children stay failed
                // (-inf) and their domains fall back to the advisory path.
                if self.config.alpha_config.past_deadline() {
                    break 'buckets;
                }
                let __clip_t = std::time::Instant::now();
                let build_shim = |i: usize| -> GraphBabDomain {
                    let child = sims[i].as_ref().expect("sim child pending");
                    let mut shim = graph_bab_domain_shim(child);
                    if clip {
                        if let Some(clipped) = self.clip_child_node_bounds(graph, child, engine) {
                            shim.node_bounds = clipped;
                            // #cone-delta: the clip replaced the map the delta
                            // was tracked against — fail closed to full-history
                            // seeding for this shim.
                            shim.delta_pre_nodes =
                                crate::beta_crown::domain::delta_pre_nodes_unknown();
                        }
                    }
                    shim
                };
                // #clip-interm-par (M1): parallelize the per-child clip when armed.
                // `collect` preserves chunk order => the per-bucket ONE-spec-row
                // invariant (buckets above) is untouched.
                let shims: Vec<GraphBabDomain> =
                    if clip && super::batched_dense_specs::clip_interm_par_enabled() {
                        chunk
                            .par_iter()
                            .map(|&i| {
                                let _g = crate::faer_parallelism::RayonTaskGuard::new();
                                build_shim(i)
                            })
                            .collect()
                    } else {
                        chunk.iter().map(|&i| build_shim(i)).collect()
                    };
                let shim_refs: Vec<&GraphBabDomain> = shims.iter().collect();
                if std::env::var("NY_CLIP_INTERM_RESNET_PROBE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "[clip-resnet] stage=kfsb n={} par={} secs={:.3}",
                        chunk.len(),
                        (clip && super::batched_dense_specs::clip_interm_par_enabled()) as u8,
                        __clip_t.elapsed().as_secs_f64()
                    );
                }
                let batched = match BatchedDomains::from_graph_domains_with_options(
                    &shim_refs,
                    relu_nodes,
                    BatchedDomainOptions {
                        enable_interm_transfer: self.config.enable_interm_transfer,
                    },
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::debug!(
                            "kfsb-multi: BatchedDomains build failed ({e}); chunk falls back"
                        );
                        continue;
                    }
                };
                match self.propagate_crown_with_batched_domains_full_specs(
                    graph,
                    &shim_refs,
                    &batched,
                    &spec_matrix,
                    engine,
                ) {
                    Ok(results) if results.len() == chunk.len() => {
                        for (&i, result) in chunk.iter().zip(results) {
                            let bounds = spec_bounds_to_vec(&result.output_bounds);
                            if let Some(&(l, u)) = bounds.first() {
                                sim_values[i] = Some(self.config.child_bound_value(Some((l, u))));
                            }
                            let should_capture = adaptive_depth_capture
                                .as_ref()
                                .is_some_and(|capture| capture.contains_sim(i));
                            if should_capture {
                                if let Some(capture) = adaptive_depth_capture.as_mut() {
                                    capture.insert_node_bounds(i, result.node_bounds);
                                }
                            }
                        }
                    }
                    Ok(results) => {
                        tracing::debug!(
                            "kfsb-multi: result count {} != chunk {} — chunk dropped",
                            results.len(),
                            chunk.len()
                        );
                    }
                    Err(e) => {
                        tracing::debug!(
                            "kfsb-multi: dense-spec backward failed ({e}); chunk dropped"
                        );
                    }
                }
            }
        }

        // Observation only: price three root candidates with a DIFFERENT
        // BaBSR-selected second split in each root child. The helper receives
        // only immutable authoritative state and uses a fresh,
        // separately-deadlined verifier for its single 12-leaf-at-most dense
        // batch.
        let adaptive_depth_selection = self.evaluate_adaptive_depth_branch_specific(
            graph,
            domains_with_unstable,
            relu_nodes,
            objectives,
            thresholds,
            engine,
            &preps,
            &sim_values,
            &sims,
            adaptive_depth_capture.as_ref(),
        );

        // ── 4+5: per-domain pick + commit of the already-built children. ──
        let mut committed: HashMap<usize, KfsbMultiChildren> = HashMap::new();
        let mut probe_lines: Vec<String> = Vec::new();
        let winner_probe = kfsb_winner_probe_enabled();
        let winner_probe_domains = winner_probe.then(kfsb_winner_probe_domains).unwrap_or(0);
        let mut winner_probe_lines: Vec<String> = Vec::new();
        for (prep_index, prep) in preps.into_iter().enumerate() {
            let side_value = |side: &SideSlot| -> f32 {
                match side {
                    SideSlot::Infeasible => f32::INFINITY,
                    SideSlot::Sim(i) => sim_values
                        .get(*i)
                        .copied()
                        .flatten()
                        .unwrap_or(f32::NEG_INFINITY),
                    SideSlot::Failed => f32::NEG_INFINITY,
                }
            };
            let best = pick_kfsb_candidate(
                &prep.candidates,
                prep.sides
                    .iter()
                    .map(|sides| (side_value(&sides[0]), side_value(&sides[1]))),
                kfsb_multi_reduce_op(),
            );
            if winner_probe && prep.slot < winner_probe_domains {
                let format_pick = |pick: Option<(usize, f32, f32)>| {
                    pick.map(|(idx, score, _)| {
                        format!(
                            "{}:{}@{score:.5}",
                            prep.candidates[idx].node_name, prep.candidates[idx].neuron_idx
                        )
                    })
                    .unwrap_or_else(|| "none".to_string())
                };
                let min_pick = pick_kfsb_candidate(
                    &prep.candidates,
                    prep.sides
                        .iter()
                        .map(|sides| (side_value(&sides[0]), side_value(&sides[1]))),
                    crate::beta_crown::KfsbReduceOp::Min,
                );
                let max_pick = pick_kfsb_candidate(
                    &prep.candidates,
                    prep.sides
                        .iter()
                        .map(|sides| (side_value(&sides[0]), side_value(&sides[1]))),
                    crate::beta_crown::KfsbReduceOp::Max,
                );
                let candidate_values = prep
                    .candidates
                    .iter()
                    .zip(&prep.sides)
                    .map(|(candidate, sides)| {
                        let active = side_value(&sides[0]);
                        let inactive = side_value(&sides[1]);
                        format!(
                            "{}:{}(a={active:.5},i={inactive:.5},min={:.5},max={:.5})",
                            candidate.node_name,
                            candidate.neuron_idx,
                            active.min(inactive),
                            active.max(inactive),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                winner_probe_lines.push(format!(
                    "slot={} straggler={} cached_gate={} cached_candidates={}/{} min={} max={} diverged={} candidates=[{}]",
                    prep.slot,
                    prep.straggler,
                    cached_la_enabled as u8,
                    prep.cached_score_candidates,
                    prep.candidates.len(),
                    format_pick(min_pick),
                    format_pick(max_pick),
                    min_pick.map(|pick| pick.0) != max_pick.map(|pick| pick.0),
                    candidate_values,
                ));
            }
            let Some((one_step_winner, one_step_score, _)) = best else {
                continue; // no evaluable candidate — advisory fallback
            };
            let (parent_idx, domain, _) = &domains_with_unstable[prep.slot];
            let (winner, score) = adaptive_depth_selection
                .as_ref()
                .and_then(|selection| {
                    resolve_adaptive_depth_authority_candidate(
                        selection,
                        prep_index,
                        *parent_idx,
                        &prep,
                        &sim_values,
                        &sims,
                        kfsb_multi_reduce_op(),
                    )
                })
                .unwrap_or((one_step_winner, one_step_score));
            let mut children: KfsbMultiChildren = Vec::with_capacity(2);
            for (side_pos, is_active) in [(0usize, true), (1usize, false)] {
                if let SideSlot::Sim(i) = &prep.sides[winner][side_pos] {
                    if let Some(child) = sims[*i].take() {
                        children.push((child, is_active));
                    }
                }
            }
            if kfsb_probe_enabled() {
                let parent_lb = domain
                    .objective_bounds
                    .get(prep.straggler)
                    .map(|(l, _)| *l)
                    .unwrap_or(f32::NAN);
                probe_lines.push(format!(
                    "slot={} cand={}:{} score={:.5} lift={:.5} n_cands={}",
                    prep.slot,
                    prep.candidates[winner].node_name,
                    prep.candidates[winner].neuron_idx,
                    score,
                    score - parent_lb,
                    prep.candidates.len(),
                ));
            }
            committed.insert(*parent_idx, children);
        }
        if kfsb_probe_enabled() {
            eprintln!(
                "[kfsb-multi] wave: domains={} sims={} committed={} | {}",
                domains_with_unstable.len(),
                sim_owner.len(),
                committed.len(),
                probe_lines.join(" ; ")
            );
        }
        if winner_probe {
            eprintln!(
                "[kfsb-winner-oracle] domains={} logged={} | {}",
                domains_with_unstable.len(),
                winner_probe_lines.len(),
                winner_probe_lines.join(" ; ")
            );
        }
        committed
    }

    /// Select the second-level BaBSR branch independently for one private
    /// lookahead child.
    ///
    /// The child's `node_bounds` must already be the constrained-forward
    /// fixpoint captured by its one-step kFSB simulation. Recomputing the
    /// objective-directed score against THAT cache is the critical difference
    /// from the historical prefix probe, which reused one global second
    /// candidate for both root children.
    #[cfg(test)]
    pub(super) fn select_adaptive_depth_base_candidate(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        relu_nodes: &[String],
        objective: &[f32],
    ) -> ny_core::Result<Option<GraphKfsbCandidate>> {
        self.select_adaptive_depth_base_candidate_impl(graph, relu_nodes, domain, objective, None)
    }

    /// Private-deadline form used by the adaptive-depth observer.  Budget
    /// expiry before or after any per-child scoring pass returns a structural
    /// deadline error, which the tree builder records as one failed side.
    pub(super) fn select_adaptive_depth_base_candidate_with_budget(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        relu_nodes: &[String],
        objective: &[f32],
        shadow_deadline: std::time::Instant,
        authority_deadline: Option<std::time::Instant>,
    ) -> ny_core::Result<Option<GraphKfsbCandidate>> {
        self.select_adaptive_depth_base_candidate_impl(
            graph,
            relu_nodes,
            domain,
            objective,
            Some((shadow_deadline, authority_deadline)),
        )
    }

    fn select_adaptive_depth_base_candidate_impl(
        &self,
        graph: &GraphNetwork,
        relu_nodes: &[String],
        domain: &MultiObjectiveGraphBabDomain,
        objective: &[f32],
        budget: Option<(std::time::Instant, Option<std::time::Instant>)>,
    ) -> ny_core::Result<Option<GraphKfsbCandidate>> {
        let check_budget = || {
            if budget.is_some_and(|(shadow_deadline, authority_deadline)| {
                !adaptive_depth_shadow_budget_available(
                    std::time::Instant::now(),
                    shadow_deadline,
                    authority_deadline,
                )
            }) {
                Err(ny_core::NyError::DeadlineExceeded(
                    "branch-specific BaBSR shadow side exhausted its private budget".to_string(),
                ))
            } else {
                Ok(())
            }
        };
        check_budget()?;
        let unstable = self.find_unstable_graph_neurons_multi(graph, domain, relu_nodes);
        check_budget()?;
        if unstable.is_empty() {
            return Ok(None);
        }
        let unstable_nodes: std::collections::HashSet<String> =
            unstable.iter().map(|(name, _)| name.clone()).collect();
        let score_parts = if let Some((shadow_deadline, _)) = budget {
            self.compute_graph_babsr_scores_from_bounds_until(
                graph,
                domain.node_bounds(),
                domain.input_bounds(),
                kfsb_multi_reduce_op(),
                Some(objective),
                Some(&unstable_nodes),
                shadow_deadline,
            )?
        } else {
            self.compute_graph_babsr_scores_from_bounds(
                graph,
                domain.node_bounds(),
                domain.input_bounds(),
                kfsb_multi_reduce_op(),
                Some(objective),
                Some(&unstable_nodes),
            )?
        };
        check_budget()?;
        let mut candidates: Vec<GraphKfsbCandidate> = unstable
            .into_iter()
            .filter_map(|(node_name, neuron_idx)| {
                let parts = score_parts.get(&(node_name.clone(), neuron_idx))?;
                parts.main_score.is_finite().then_some(GraphKfsbCandidate {
                    node_name,
                    neuron_idx,
                    main_score: parts.main_score,
                    backup_score: parts.backup_score,
                })
            })
            .collect();
        candidates.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.main_score, &b.main_score)
                .then_with(|| {
                    crate::cmp_utils::nan_last_descending_cmp(&a.backup_score, &b.backup_score)
                })
                .then_with(|| a.node_name.cmp(&b.node_name))
                .then_with(|| a.neuron_idx.cmp(&b.neuron_idx))
        });
        check_budget()?;
        candidates.into_iter().next().map(Some).ok_or_else(|| {
            ny_core::NyError::InternalError(
                "branch-specific lookahead found unstable neurons but no finite BaBSR score"
                    .to_string(),
            )
        })
    }

    /// Measure true branch-specific depth-2 lookahead for one domain's first
    /// three BaBSR-preselected roots. Each root child's second split is selected
    /// independently from that child's refreshed bounds, matching the key
    /// recursive `baseSelect` property of Lookahead Branching (Davis et al.,
    /// arXiv:2607.17290) while retaining a hard twelve-leaf cap and six captured
    /// one-step bound maps.
    ///
    /// It receives immutable authoritative inputs, builds private domains, and
    /// runs separately-deadlined child scoring plus one dense batch. Shadow-only
    /// mode prints metrics and returns no authority. The separately gated
    /// selection mode may return only an exact identity for one already-built
    /// first-level root; no private leaf or bound crosses this boundary.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_adaptive_depth_branch_specific(
        &self,
        graph: &GraphNetwork,
        domains_with_unstable: &[MultiObjDomainWithUnstable<'_>],
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: &dyn GemmEngine,
        preps: &[DomainPrep],
        sim_values: &[Option<f32>],
        sims: &[Option<MultiObjectiveGraphBabDomain>],
        capture: Option<&AdaptiveDepthShadowCapture>,
    ) -> Option<AdaptiveDepthAuthoritySelection> {
        use std::sync::atomic::Ordering;

        let shadow_enabled = adaptive_depth_shadow_enabled();
        let authority_enabled = adaptive_depth_select_enabled();
        if (!shadow_enabled && !authority_enabled)
            || self.adaptive_depth_shadow_fired.load(Ordering::Relaxed)
        {
            return None;
        }

        // The capture plan fixed one domain and three BaBSR roots before the
        // dense simulation, allowing a strict six-map memory cap. Re-rank that
        // fixed shortlist by the already-computed one-step child values; do not
        // fall through to an uncaptured domain if one simulation failed.
        let capture = capture?;
        let prep = preps.get(capture.prep_index)?;
        let side_value = |side: &SideSlot| -> f32 {
            match side {
                SideSlot::Infeasible => f32::INFINITY,
                SideSlot::Sim(i) => sim_values
                    .get(*i)
                    .copied()
                    .flatten()
                    .unwrap_or(f32::NEG_INFINITY),
                SideSlot::Failed => f32::NEG_INFINITY,
            }
        };
        let values: Vec<(f32, f32)> = prep
            .sides
            .iter()
            .map(|sides| (side_value(&sides[0]), side_value(&sides[1])))
            .collect();
        let captured_side_eligible = |side: &SideSlot| match side {
            SideSlot::Infeasible => true,
            SideSlot::Failed => false,
            SideSlot::Sim(index) => {
                capture.contains_sim(*index)
                    && capture.node_bounds(*index).is_some()
                    && sims.get(*index).is_some_and(Option::is_some)
                    && sim_values
                        .get(*index)
                        .copied()
                        .flatten()
                        .is_some_and(|value| !value.is_nan())
            }
        };
        let captured_eligible: Vec<bool> = prep
            .sides
            .iter()
            .take(ADAPTIVE_DEPTH_SHADOW_ROOTS)
            .map(|sides| captured_side_eligible(&sides[0]) && captured_side_eligible(&sides[1]))
            .collect();
        let top = if authority_enabled {
            rank_adaptive_depth_authority_portfolio(
                &prep.candidates,
                &values,
                &captured_eligible,
                kfsb_multi_reduce_op(),
            )?
        } else {
            // Preserve M1 shadow telemetry exactly: it ranks the fixed captured
            // shortlist by exact child score. Only the M27 authority path must
            // pin the historical all-candidate winner at rank zero.
            if prep.candidates.len() < ADAPTIVE_DEPTH_SHADOW_ROOTS
                || values.len() < ADAPTIVE_DEPTH_SHADOW_ROOTS
            {
                return None;
            }
            let ranked = rank_adaptive_depth_candidates(
                &prep.candidates[..ADAPTIVE_DEPTH_SHADOW_ROOTS],
                &values[..ADAPTIVE_DEPTH_SHADOW_ROOTS],
                kfsb_multi_reduce_op(),
            );
            if ranked.len() < ADAPTIVE_DEPTH_SHADOW_ROOTS {
                return None;
            }
            ranked
        };

        let started = std::time::Instant::now();
        let authority_deadline = self.config.alpha_config.deadline;
        let Some(shadow_deadline) = adaptive_depth_shadow_deadline(started, authority_deadline)
        else {
            if self
                .adaptive_depth_shadow_fired
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                eprintln!(
                    "[mo-adaptive-depth-shadow] skipped=authority_reserve reserve_ms={}",
                    ADAPTIVE_DEPTH_AUTHORITY_RESERVE.as_millis()
                );
            }
            return None;
        };
        if self
            .adaptive_depth_shadow_fired
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }

        let Some((parent_idx, domain, _unstable)) = domains_with_unstable.get(prep.slot) else {
            eprintln!(
                "[mo-adaptive-depth-shadow] skipped=missing_parent slot={}",
                prep.slot
            );
            return None;
        };
        let Some(objective) = objectives.get(prep.straggler) else {
            eprintln!(
                "[mo-adaptive-depth-shadow] skipped=missing_objective straggler={}",
                prep.straggler
            );
            return None;
        };
        let Some(&threshold) = thresholds.get(prep.straggler) else {
            eprintln!(
                "[mo-adaptive-depth-shadow] skipped=missing_threshold straggler={}",
                prep.straggler
            );
            return None;
        };
        let parent_value = self
            .config
            .child_bound_value(domain.objective_bounds().get(prep.straggler).copied());
        let mut shadow_config = self.config.clone();
        shadow_config.timeout = ADAPTIVE_DEPTH_SHADOW_BUDGET;
        let mut shadow_verifier = BetaCrownVerifier::new(shadow_config);
        // `new` derives a deadline from `timeout` at construction time. Pin it
        // back to the exact observer deadline created before any branch-specific
        // scoring so setup latency cannot extend the one-second budget.
        shadow_verifier.config.alpha_config.deadline = Some(shadow_deadline);

        // Evaluate EACH shortlisted root candidate. Its active and inactive
        // children carry independently refreshed bounds, so `baseSelect`
        // below is genuinely branch-specific. Three roots × two root sides ×
        // two second-level sides gives the same hard twelve-leaf ceiling as
        // the historical d2+d3 prefix probe, but now prices the published
        // depth-2 decision rather than one arbitrary shared prefix.
        let mut trees: Vec<Vec<AdaptiveDepthShadowLeaf>> = Vec::with_capacity(top.len());
        let mut second_choices: Vec<Vec<String>> = Vec::with_capacity(top.len());
        for (candidate_idx, _one_step_score) in &top {
            if !adaptive_depth_shadow_budget_available(
                std::time::Instant::now(),
                shadow_deadline,
                authority_deadline,
            ) {
                trees.push(vec![AdaptiveDepthShadowLeaf::Failed]);
                second_choices.push(vec!["root=budget-expired".to_string()]);
                continue;
            }
            let Some(sides) = prep.sides.get(*candidate_idx) else {
                trees.push(vec![AdaptiveDepthShadowLeaf::Failed]);
                second_choices.push(vec!["root=missing".to_string()]);
                continue;
            };
            let root_leaves = sides
                .iter()
                .map(|side| adaptive_depth_shadow_leaf_from_side(side, sims, capture))
                .collect();
            if !adaptive_depth_shadow_budget_available(
                std::time::Instant::now(),
                shadow_deadline,
                authority_deadline,
            ) {
                trees.push(vec![AdaptiveDepthShadowLeaf::Failed]);
                second_choices.push(vec!["root=budget-expired".to_string()]);
                continue;
            }
            let (leaves, choices) = expand_adaptive_depth_shadow_leaves_branch_specific(
                graph,
                root_leaves,
                self.config.verify_upper_bound,
                thresholds,
                |child| {
                    shadow_verifier.select_adaptive_depth_base_candidate_with_budget(
                        graph,
                        child,
                        relu_nodes,
                        objective,
                        shadow_deadline,
                        authority_deadline,
                    )
                },
                || {
                    adaptive_depth_shadow_budget_available(
                        std::time::Instant::now(),
                        shadow_deadline,
                        authority_deadline,
                    )
                },
            );
            trees.push(leaves);
            second_choices.push(choices);
        }

        let mut tree_metrics: Vec<AdaptiveDepthShadowMetrics> = trees
            .iter()
            .map(|leaves| AdaptiveDepthShadowMetrics::from_leaves(leaves))
            .collect();
        let mut shims: Vec<GraphBabDomain> = Vec::with_capacity(12);
        let mut owners: Vec<usize> = Vec::with_capacity(12);
        for (owner, leaves) in trees.iter().enumerate() {
            for leaf in leaves {
                if let AdaptiveDepthShadowLeaf::Feasible(child) = leaf {
                    shims.push(graph_bab_domain_shim(child));
                    owners.push(owner);
                }
            }
        }

        if !shims.is_empty() {
            if !adaptive_depth_shadow_budget_available(
                std::time::Instant::now(),
                shadow_deadline,
                authority_deadline,
            ) {
                mark_adaptive_depth_dense_failure(&mut tree_metrics, &owners);
            } else {
                let refs: Vec<&GraphBabDomain> = shims.iter().collect();
                let batched = BatchedDomains::from_graph_domains_with_options(
                    &refs,
                    relu_nodes,
                    BatchedDomainOptions {
                        enable_interm_transfer: self.config.enable_interm_transfer,
                    },
                );
                let spec_matrix = build_spec_matrix(&[objective.clone()]);
                match (batched, spec_matrix) {
                    (Ok(batched), Some(spec_matrix)) => {
                        if !adaptive_depth_shadow_budget_available(
                            std::time::Instant::now(),
                            shadow_deadline,
                            authority_deadline,
                        ) {
                            mark_adaptive_depth_dense_failure(&mut tree_metrics, &owners);
                        } else {
                            match shadow_verifier.propagate_crown_with_batched_domains_full_specs(
                                graph,
                                &refs,
                                &batched,
                                &spec_matrix,
                                engine,
                            ) {
                                Ok(results)
                                    if results.len() == owners.len()
                                        && adaptive_depth_shadow_budget_available(
                                            std::time::Instant::now(),
                                            shadow_deadline,
                                            authority_deadline,
                                        ) =>
                                {
                                    for (&owner, result) in owners.iter().zip(&results) {
                                        if let Some(metrics) = tree_metrics.get_mut(owner) {
                                            if adaptive_depth_shadow_budget_available(
                                                std::time::Instant::now(),
                                                shadow_deadline,
                                                authority_deadline,
                                            ) {
                                                let bounds =
                                                    spec_bounds_to_vec(&result.output_bounds)
                                                        .first()
                                                        .copied();
                                                metrics.record_bounds(
                                                    bounds,
                                                    &self.config,
                                                    threshold,
                                                );
                                            } else {
                                                metrics.failures += 1;
                                            }
                                        }
                                    }
                                }
                                Ok(_) | Err(_) => {
                                    mark_adaptive_depth_dense_failure(&mut tree_metrics, &owners);
                                }
                            }
                        }
                    }
                    _ => mark_adaptive_depth_dense_failure(&mut tree_metrics, &owners),
                }
            }
        }

        // Deterministic fault injection for end-to-end authority regressions.
        // This block is absent from production builds; every injected metric
        // still flows through the same completeness helper as real results.
        #[cfg(test)]
        let adaptive_depth_test_fault = std::env::var("NY_TEST_MO_ADAPTIVE_DEPTH_FAULT").ok();
        #[cfg(test)]
        if let Some(fault) = adaptive_depth_test_fault.as_deref() {
            let complete = |score| AdaptiveDepthShadowMetrics {
                expected: 4,
                bounded: 4,
                surviving: 4,
                post_min: score,
                ..AdaptiveDepthShadowMetrics::default()
            };
            tree_metrics = vec![complete(1.0), complete(2.0), complete(3.0)];
            match fault {
                "force-third" | "timeout" | "missing-side" | "identity-mismatch" => {}
                "nan-bounds" => tree_metrics[1].post_min = f32::NAN,
                "failed-leaf" | "construction-error" | "shape-error" => {
                    tree_metrics[1].failures = 1;
                }
                "malformed-counts" => {
                    tree_metrics[1].bounded = 3;
                    tree_metrics[1].surviving = 3;
                }
                "partial-metrics" => {
                    tree_metrics.pop();
                }
                _ => tree_metrics.clear(),
            }
        }

        let outcomes: Vec<(f32, f32)> = tree_metrics
            .iter()
            .map(|metrics| metrics.post_min_and_lift(parent_value))
            .collect();
        let final_budget_available = adaptive_depth_shadow_budget_available(
            std::time::Instant::now(),
            shadow_deadline,
            authority_deadline,
        ) && parent_value.is_finite();
        #[cfg(test)]
        let final_budget_available =
            final_budget_available && adaptive_depth_test_fault.as_deref() != Some("timeout");
        let best_rank = final_budget_available
            .then(|| select_complete_adaptive_depth_rank(&tree_metrics))
            .flatten();
        let candidate_text = top
            .iter()
            .enumerate()
            .map(|(rank, (idx, score))| {
                let candidate = &prep.candidates[*idx];
                match (tree_metrics.get(rank), outcomes.get(rank)) {
                    (Some(metrics), Some(&(post_min, lift))) => format!(
                        "{}:{}@one={score:.6}/second=[{}]/post_min={post_min:.6}/lift={lift:.6}/v={}/s={}/e={}/b={}/f={}/{}",
                        candidate.node_name,
                        candidate.neuron_idx,
                        second_choices
                            .get(rank)
                            .map(|choices| choices.join("|"))
                            .unwrap_or_else(|| "missing".to_string()),
                        metrics.verified,
                        metrics.surviving,
                        metrics.infeasible,
                        metrics.bounded,
                        metrics.failures,
                        metrics.expected,
                    ),
                    _ => format!(
                        "{}:{}@one={score:.6}/metric=missing",
                        candidate.node_name, candidate.neuron_idx,
                    ),
                }
            })
            .collect::<Vec<_>>()
            .join(";");
        let selected_text = best_rank
            .map(|rank| {
                let candidate = &prep.candidates[top[rank].0];
                format!("{}:{}", candidate.node_name, candidate.neuron_idx)
            })
            .unwrap_or_else(|| "none".to_string());
        let authority_selection = if authority_enabled {
            best_rank.and_then(|rank| {
                adaptive_depth_authority_identity(
                    capture.prep_index,
                    *parent_idx,
                    prep,
                    top.get(rank)?.0,
                )
            })
        } else {
            None
        };
        #[cfg(test)]
        let authority_selection = {
            let mut selection = authority_selection;
            if let Some(selection) = selection.as_mut() {
                match adaptive_depth_test_fault.as_deref() {
                    Some("missing-side") => selection.sides_len = usize::MAX,
                    Some("identity-mismatch") => selection.neuron_idx ^= 1,
                    _ => {}
                }
            }
            selection
        };
        eprintln!(
            "[mo-adaptive-depth-shadow] mode=branch-specific-d2 slot={} straggler={} parent={:.6} roots={} selected={} changed={} authority={} candidates=[{}] batched={} budget_ms={} elapsed_ms={} cap_hit={}",
            prep.slot,
            prep.straggler,
            parent_value,
            top.len(),
            selected_text,
            best_rank.is_some_and(|rank| rank != 0) as u8,
            authority_selection.is_some() as u8,
            candidate_text,
            owners.len(),
            ADAPTIVE_DEPTH_SHADOW_BUDGET.as_millis(),
            started.elapsed().as_millis(),
            (std::time::Instant::now() >= shadow_deadline) as u8,
        );
        authority_selection
    }

    /// Steps 1+2+3a for one wave domain: straggler row, objective-directed
    /// pre-scores, top-k ∪ backup ∪ layer-quota filter, and both children per
    /// candidate. Returns `None` (advisory fallback) when the domain has no
    /// unverified objective, the score backward fails, or no candidate yields
    /// any feasible-or-infeasible side.
    // Justification: mirrors the caller's context; splitting further would
    // just add a one-use struct.
    #[allow(clippy::too_many_arguments)]
    fn kfsb_multi_prepare_domain(
        &self,
        graph: &GraphNetwork,
        slot: usize,
        domain: &MultiObjectiveGraphBabDomain,
        unstable: &[(String, usize)],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        k: usize,
        layer_quota: bool,
        cached_la_enabled: bool,
    ) -> Option<(DomainPrep, Vec<(usize, MultiObjectiveGraphBabDomain)>)> {
        if unstable.is_empty() {
            return None;
        }
        // Worst unverified straggler (mirrors branching/graph.rs).
        let mut straggler: Option<(usize, f32)> = None;
        for (i, (lo, _)) in domain.objective_bounds.iter().enumerate() {
            if domain.verified.get(i).copied().unwrap_or(false) {
                continue;
            }
            let lo = if lo.is_nan() { f32::NEG_INFINITY } else { *lo };
            if straggler.is_none_or(|(_, w)| lo < w) {
                straggler = Some((i, lo));
            }
        }
        let (straggler, _) = straggler?;
        let seed_row = objectives.get(straggler)?.as_slice();

        // Objective-directed BaBSR pre-scores, stopped early at the unstable set.
        let unstable_nodes: std::collections::HashSet<String> =
            unstable.iter().map(|(n, _)| n.clone()).collect();
        let reduce_op = kfsb_multi_reduce_op();
        let (score_parts, cached_score_keys) = if cached_la_enabled {
            let mut cached_parts = domain
                .cached_la_for_objective(straggler)
                .map(|cached_la| {
                    self.compute_graph_babsr_scores_from_cached_la(
                        graph,
                        &domain.node_bounds,
                        &domain.input_bounds,
                        cached_la,
                        reduce_op,
                        &unstable_nodes,
                    )
                })
                .unwrap_or_default();
            let cached_score_keys: std::collections::HashSet<(&str, usize)> = unstable
                .iter()
                .filter(|candidate| cached_parts.contains_key(*candidate))
                .map(|(node_name, neuron_idx)| (node_name.as_str(), *neuron_idx))
                .collect();
            let needs_proxy = unstable
                .iter()
                .any(|candidate| !cached_parts.contains_key(candidate));
            if needs_proxy {
                let proxy_parts = self
                    .compute_graph_babsr_scores_from_bounds(
                        graph,
                        &domain.node_bounds,
                        &domain.input_bounds,
                        reduce_op,
                        Some(seed_row),
                        Some(&unstable_nodes),
                    )
                    .ok()?;
                for candidate in unstable {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        cached_parts.entry(candidate.clone())
                    {
                        if let Some(parts) = proxy_parts.get(candidate) {
                            entry.insert(*parts);
                        }
                    }
                }
            }
            (cached_parts, cached_score_keys)
        } else {
            // Exact off-path anchor: this is the pre-gate expression and its
            // returned map is consumed directly, without an O(unstable) copy.
            (
                self.compute_graph_babsr_scores_from_bounds(
                    graph,
                    &domain.node_bounds,
                    &domain.input_bounds,
                    reduce_op,
                    Some(seed_row),
                    Some(&unstable_nodes),
                )
                .ok()?,
                std::collections::HashSet::new(),
            )
        };
        let mut scored: Vec<GraphKfsbCandidate> = unstable
            .iter()
            .map(|(node_name, neuron_idx)| {
                let parts = score_parts
                    .get(&(node_name.clone(), *neuron_idx))
                    .copied()
                    .unwrap_or_default();
                GraphKfsbCandidate {
                    node_name: node_name.clone(),
                    neuron_idx: *neuron_idx,
                    main_score: parts.main_score,
                    backup_score: parts.backup_score,
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.main_score, &b.main_score)
        });

        // Top-k by main ∪ top-k by backup (this lane always uses the backup
        // channel — the probe's rank-2 candidate was backup-admitted), plus
        // the optional stratified per-layer quota.
        let mut candidates = select_graph_kfsb_eval_candidates(&scored, k, true);
        if layer_quota {
            append_layer_quota_candidates(&scored, &mut candidates);
        }
        if candidates.is_empty() {
            return None;
        }

        // Both children per candidate. `Ok(None)` / infeasible-domain errors
        // mean the half-space is EMPTY (side resolved); other errors mark the
        // side failed (-inf) — a candidate with any usable side stays ranked.
        let mut sides: Vec<[SideSlot; 2]> = Vec::with_capacity(candidates.len());
        let mut children: Vec<(usize, MultiObjectiveGraphBabDomain)> = Vec::new();
        let mut kept: Vec<GraphKfsbCandidate> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let mut pair: Vec<SideSlot> = Vec::with_capacity(2);
            let mut usable = false;
            for is_active in [true, false] {
                let constraint = GraphNeuronConstraint {
                    node_name: candidate.node_name.clone(),
                    neuron_idx: candidate.neuron_idx,
                    is_active,
                    score: candidate.main_score,
                };
                match domain.with_constraint(graph, constraint, false, thresholds) {
                    Ok(Some(child)) => {
                        let local = children.len();
                        children.push((local, child));
                        pair.push(SideSlot::Sim(local));
                        usable = true;
                    }
                    Ok(None) => {
                        pair.push(SideSlot::Infeasible);
                        usable = true;
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        pair.push(SideSlot::Infeasible);
                        usable = true;
                    }
                    Err(_) => {
                        // Reusing Infeasible here would over-reward a broken
                        // side; Failed counts as -inf at pick time so the
                        // candidate survives only on its other side's merit.
                        pair.push(SideSlot::Failed);
                    }
                }
            }
            if usable {
                let pair: [SideSlot; 2] = match pair.try_into() {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                sides.push(pair);
                kept.push(candidate);
            }
        }
        if kept.is_empty() {
            return None;
        }
        let cached_score_candidates = kept
            .iter()
            .filter(|candidate| {
                cached_score_keys.contains(&(candidate.node_name.as_str(), candidate.neuron_idx))
            })
            .count();
        Some((
            DomainPrep {
                slot,
                straggler,
                cached_score_candidates,
                candidates: kept,
                sides,
            },
            children,
        ))
    }
}

/// Stratified layer quota: admit each unstable ReLU layer's top-1 main-score
/// candidate (dedup against the already-selected set). `scored` is sorted by
/// main score descending, so the first sighting of a layer is its best.
pub(super) fn append_layer_quota_candidates(
    scored: &[GraphKfsbCandidate],
    candidates: &mut Vec<GraphKfsbCandidate>,
) {
    let mut seen: std::collections::HashSet<(String, usize)> = candidates
        .iter()
        .map(|c| (c.node_name.clone(), c.neuron_idx))
        .collect();
    let mut layers_done: std::collections::HashSet<String> =
        candidates.iter().map(|c| c.node_name.clone()).collect();
    for candidate in scored {
        if !layers_done.insert(candidate.node_name.clone()) {
            continue;
        }
        if seen.insert((candidate.node_name.clone(), candidate.neuron_idx)) {
            candidates.push(candidate.clone());
        }
    }
}
