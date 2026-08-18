// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Increment 3 — wire the proven multi-neuron facets into the cifar100 BaB ROOT
//! margin (`docs/MULTI_NEURON_RELAXATION_DESIGN.md` §5, increment 3).
//!
//! The root margin is computed by a spec-guided CROWN backward over the output
//! objective spec. This module:
//!
//! 1. selects both-unstable neuron pairs at a chosen ReLU node (§3, ranked by
//!    [`Octahedron2::excluded_corner_score`]);
//! 2. builds a sound [`MultiNeuronPool`] of coupling facets via the LIVE joint
//!    producer ([`combined_rows_octahedra`], batched into ONE backward);
//! 3. runs the objective backward with the pool injected (§2.2) over a `β_c`
//!    line-search + Adam ascent, tracking the per-objective MAX lower bound; and
//! 4. returns per-objective bounds whose lower is the sound max of the baseline
//!    and every injected candidate.
//!
//! # Soundness (Invariant MN, the deliverable's core)
//!
//! Every injected margin lower bound is a *valid* lower bound on the true min
//! margin for ANY `β_c ≥ 0`, because each facet is a proven-superset half-space
//! (Invariant P1 outward-rounded producer + Invariant P2 certified-outward RHS)
//! and the §2.2 injection is the proven Lagrangian embedding. Taking, per
//! objective, the MAX over the baseline and every evaluated `β_c` is therefore
//! sound: `max_k L_o(β_k) ≤ min_u margin_o(u)`. A looser/degenerate facet only
//! fails to tighten; it can never raise the LB above the true min. The path is
//! DEFAULT-OFF (`NY_MULTINEURON=1` to arm) and scoped to conv nets.

use std::collections::HashMap;
use std::time::Instant;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

use crate::bounds::GraphAlphaState;
use crate::layers::Layer;
use crate::network::SpecCrownRequest;
use crate::GraphNetwork;

use super::producer::combined_rows_octahedra;
use super::{coupling_facets, MnVar, MultiNeuronConstraint, MultiNeuronPool, Octahedron2};

/// Production authority gate for multi-neuron root injection.
///
/// Hard-quarantined even when `NY_MULTINEURON=1` is present. The current facet
/// generator uses tolerance-based vertex deduplication and lacks a directed or
/// exact support checker for its stored f32 RHS. Finite sampling can falsify a
/// bad facet but cannot authorize a verdict. The implementation remains for
/// research and repair; an environment variable cannot enable it.
pub fn enabled() -> bool {
    false
}

/// Env gate: permit multi-neuron root injection on conv-FREE (dense/MLP) nets —
/// e.g. relational ACAS difference nets — default-OFF (`NY_MULTINEURON_MLP=1`).
/// The main-path both-unstable-pair selection ([`target_relu_nodes`] picks
/// dense-fed ReLUs) and the sound per-objective MAX fold are layer-agnostic; the
/// only NCHW-layout assumption lives in the separate STEM-resident lever. So
/// opening the conv scope for the main path is sound-by-construction (can only
/// tighten, never over-claim). Byte-identical when unset.
pub fn mlp_enabled() -> bool {
    matches!(
        std::env::var("NY_MULTINEURON_MLP").ok().as_deref(),
        Some("1")
    )
}

/// Production authority gate for the proposed STEM-RESIDENT lever.
///
/// Hard-quarantined: it shares the uncertified facet producer, its f64→f32 fold
/// reduction drops build error, and its process-global entry carries no target
/// or model identity. The true-network sample guard is diagnostic, not a proof.
pub fn stem_enabled() -> bool {
    false
}

/// Production authority gate for the proposed HEAD-RESIDENT lever.
///
/// Currently hard-quarantined in `ny-core`: finite true-network sampling cannot
/// certify a verdict-authoritative lower bound, and the f32 resident reduction
/// does not carry the facet build-error certificate. The research implementation
/// remains available for repair, but `NY_MN_HEAD_RESIDENT=1` cannot arm it.
pub fn head_resident_enabled() -> bool {
    ny_core::resident_cut_fold::head_resident_retarget_enabled()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

/// Configuration knobs (all env-tunable for measurement).
struct MnConfig {
    /// Candidate cap: top-K widest both-unstable neurons per ReLU node.
    top_k: usize,
    /// Max facet-carrying groups kept per ReLU node (by score).
    max_groups: usize,
    /// Number of shared-`β` Adam ascent steps after the grid.
    adam_steps: usize,
    /// Whether to also inject conv-fed ReLU groups (default OFF this increment —
    /// the head holds the hard-objective slack; conv groups use the validated
    /// [`super::chw_to_flat`] mapping and land next).
    conv_groups: bool,
}

impl MnConfig {
    fn from_env() -> Self {
        Self {
            top_k: env_usize("NY_MULTINEURON_TOPK", 12),
            max_groups: env_usize("NY_MULTINEURON_MAXGROUPS", 24),
            adam_steps: env_usize("NY_MULTINEURON_ADAM_STEPS", 2),
            conv_groups: matches!(
                std::env::var("NY_MULTINEURON_CONV").ok().as_deref(),
                Some("1")
            ),
        }
    }
}

/// A both-unstable candidate pair with its assembled octahedron and score.
struct ScoredPair {
    i: usize,
    j: usize,
    p: Octahedron2,
    score: f64,
}

/// Row-major flat width of the pre-activation node.
fn pre_width(node_bounds: &HashMap<String, BoundedTensor>, pre_node: &str) -> Option<usize> {
    node_bounds.get(pre_node).map(|bt| bt.flatten().len())
}

/// Top-K widest both-unstable neuron indices at `pre_node` (§3 candidate gate).
fn top_unstable(
    node_bounds: &HashMap<String, BoundedTensor>,
    pre_node: &str,
    top_k: usize,
) -> Vec<usize> {
    let Some(bt) = node_bounds.get(pre_node) else {
        return Vec::new();
    };
    let flat = bt.flatten();
    let lo = flat.lower();
    let hi = flat.upper();
    let mut unstable: Vec<(usize, f32)> = Vec::new();
    for (idx, (&l, &u)) in lo.iter().zip(hi.iter()).enumerate() {
        if l < 0.0 && u > 0.0 && (u - l).is_finite() {
            unstable.push((idx, u - l));
        }
    }
    unstable.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    unstable.truncate(top_k.max(2));
    unstable.into_iter().map(|(idx, _)| idx).collect()
}

/// Build the scored, facet-carrying pool for one ReLU node from a single batched
/// producer backward. Returns the pool and the kept scored pairs (for logging).
///
/// `binding_coeffs` (#mn-binding-select): when `Some(c)`, `c[k]` is the objective
/// sensitivity of the binding worst-child spec row to head post-activation neuron
/// `k` (see [`head_objective_coeffs`]); candidate pairs are then RANKED by their
/// JOINT effect on that margin ([`binding_rank_score`]) instead of the objective-
/// blind geometric heuristic. `None` (every caller but the armed head-resident
/// lever) reproduces the heuristic ranking BYTE-IDENTICALLY. The pair-inclusion
/// gate (`excluded_corner_score > 0`, Lemma 1) and every downstream step
/// (`coupling_facets`, facet build) are UNCHANGED — selection only reorders which
/// valid facets are kept, so soundness is independent of the ranking key.
#[allow(clippy::too_many_arguments)]
fn build_pool_for_node(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: &HashMap<String, BoundedTensor>,
    relu_node: &str,
    pre_node: &str,
    engine: Option<&dyn GemmEngine>,
    cfg: &MnConfig,
    binding_coeffs: Option<&[f64]>,
    use_certified: bool,
) -> Option<(MultiNeuronPool, Vec<ScoredPair>)> {
    let n_pre = pre_width(node_bounds, pre_node)?;
    let candidates = top_unstable(node_bounds, pre_node, cfg.top_k);
    if candidates.len() < 2 {
        return None;
    }
    // All pairs among the top-K widest both-unstable neurons.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for a in 0..candidates.len() {
        for b in (a + 1)..candidates.len() {
            let (i, j) = (candidates[a], candidates[b]);
            if i < n_pre && j < n_pre {
                pairs.push((i, j));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }

    // ONE batched sound producer backward for every candidate pair (§5.3).
    let octahedra = combined_rows_octahedra(
        graph,
        input,
        alpha_state,
        Some(node_bounds),
        pre_node,
        &pairs,
        engine,
    )
    .ok()?;

    let mut scored: Vec<ScoredPair> = Vec::new();
    for (&(i, j), p) in pairs.iter().zip(octahedra) {
        if !p.both_unstable() {
            continue;
        }
        // Pair-INCLUSION gate (UNCHANGED, ranking-independent): a pair with
        // `P == box` has excluded_corner_score == 0 ⇒ no coupling facet to build
        // (Lemma 1) ⇒ skipped, whatever the ranking key.
        let geo = p.excluded_corner_score();
        if geo <= 0.0 {
            continue;
        }
        // RANKING key. Default (`binding_coeffs == None`): the objective-BLIND
        // geometric score — BYTE-IDENTICAL to the pre-#mn-binding-select path.
        // Armed (`Some(coeffs)`, NY_MN_BINDING_SELECT on the head-resident lever):
        // the objective-gradient-informed score, so the kept `max_groups` are the
        // pairs whose JOINT coupling most tightens the binding worst-child margin.
        let score = match binding_coeffs {
            Some(coeffs) => binding_rank_score(&p, i, j, coeffs),
            None => geo,
        };
        scored.push(ScoredPair { i, j, p, score });
    }
    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(cfg.max_groups);

    let mut pool = MultiNeuronPool::new(0);
    let mut kept: Vec<ScoredPair> = Vec::new();
    for sp in scored {
        // Facet SOURCE. Default (`use_certified == false`): the legacy research
        // producer `coupling_facets` — BYTE-IDENTICAL to the pre-#mn-head-f64-
        // certified-measure path. Armed (`use_certified == true`,
        // NY_MN_HEAD_F64_CERTIFIED_MEASURE on the f64 masked lever):
        // `certified_coupling_facets_exact` — every stored RHS is rebuilt by exact
        // rational maximization over `P ∩` (all four ReLU orthants), so the pool
        // carries only certifier-approved proven-superset half-spaces. Either
        // source only feeds VALID facets into the sound monotone-max f64 fold.
        let facets = if use_certified {
            crate::multineuron::certified_coupling_facets_exact(&sp.p)
        } else {
            coupling_facets(&sp.p)
        };
        let mut any = false;
        for f in &facets {
            if let Ok(group) =
                MultiNeuronConstraint::from_facet_for_group(f, relu_node, sp.i, relu_node, sp.j)
            {
                if pool.push(group) {
                    any = true;
                }
            }
        }
        if any {
            kept.push(sp);
        }
    }
    if pool.is_empty() {
        return None;
    }
    Some((pool, kept))
}

// ===========================================================================
// #mn-binding-select (NY_MN_BINDING_SELECT) — objective-gradient-informed head
// facet pair SELECTION. The heuristic ranks pairs by the objective-BLIND
// `excluded_corner_score` (pure coupling geometry). A k=2 coupling facet only
// helps if the pair's correlation is BINDING for the worst-child margin, so this
// re-ranks candidate pairs by their JOINT effect on the binding spec row. Rides
// the head-resident path; default OFF ⇒ heuristic ranking, byte-identical.
// SOUNDNESS: this changes only WHICH valid facets are kept — `coupling_facets`
// and the facet build are untouched, so every kept facet is still a proven
// superset half-space (the existing enclosure/MC oracles apply unchanged).
// ===========================================================================

/// Env gate for objective-gradient-informed head facet pair selection
/// (`NY_MN_BINDING_SELECT=1`, default OFF). Consulted ONLY by the head-resident
/// lever, which then passes `binding_coeffs` into [`build_pool_for_node`]. When
/// unset the lever passes `None` ⇒ the heuristic ranking is reproduced
/// byte-identically.
pub fn binding_select_enabled() -> bool {
    matches!(
        std::env::var("NY_MN_BINDING_SELECT").ok().as_deref(),
        Some("1")
    )
}

/// #mn-binding-select — objective-gradient-informed RANKING key for a candidate
/// head pair `(i, j)`. `coeffs[k]` is the objective's linear sensitivity to head
/// POST-activation neuron `k` (`c_k = Σ_o spec[o]·W_out[o,k]`, see
/// [`head_objective_coeffs`]). The k=2 coupling facet tightens the binding margin
/// most when BOTH neurons carry a large objective coefficient AND the octahedron
/// clips exactly the box corner the objective's minimization drives toward:
///
/// * minimizing `Σ c_k·y_k` with `y_k = ReLU(x_k) ≥ 0` pushes `y_k` HIGH when
///   `c_k < 0` (⇒ `x_k` toward its UPPER bound) and toward 0 when `c_k > 0`
///   (⇒ `x_k` toward its LOWER bound);
/// * the octahedral SUM bound (`s_hi`/`s_lo`) clips the both-high / both-low
///   corners; the DIFFERENCE bound (`d_hi`/`d_lo`) clips the mixed (one-high,
///   one-low) corners.
///
/// So the score is `|c_i|·|c_j|` (BOTH neurons must matter to the objective —
/// this is where 2-neuron coupling beats the product of single-neuron
/// relaxations) times the single corner gap the worst-case objective corner falls
/// in. Objective-IRRELEVANT pairs (`c_i≈0` or `c_j≈0`) score 0 and rank last.
/// Never negative. Purely a ranking key — it cannot affect facet validity.
fn binding_rank_score(p: &Octahedron2, i: usize, j: usize, coeffs: &[f64]) -> f64 {
    let ci = coeffs.get(i).copied().unwrap_or(0.0);
    let cj = coeffs.get(j).copied().unwrap_or(0.0);
    let mag = ci.abs() * cj.abs();
    if !mag.is_finite() || mag == 0.0 {
        return 0.0;
    }
    // The four octahedral corner gaps (mirrors `excluded_corner_score`'s terms).
    let sum_hi_gap = ((p.u1 + p.u2) - p.s_hi).max(0.0);
    let sum_lo_gap = (p.s_lo - (p.l1 + p.l2)).max(0.0);
    let dif_hi_gap = ((p.u1 - p.l2) - p.d_hi).max(0.0);
    let dif_lo_gap = (p.d_lo - (p.l1 - p.u2)).max(0.0);
    // Which box corner the objective's minimization drives toward (want y HIGH ⇔ c<0).
    let want_high_i = ci < 0.0;
    let want_high_j = cj < 0.0;
    let gap = match (want_high_i, want_high_j) {
        (true, true) => sum_hi_gap,   // both high     ⇒ x_i + x_j clipped above
        (false, false) => sum_lo_gap, // both low      ⇒ x_i + x_j clipped below
        (true, false) => dif_hi_gap,  // i high, j low ⇒ x_i − x_j clipped above
        (false, true) => dif_lo_gap,  // i low, j high ⇒ x_i − x_j clipped below
    };
    mag * gap
}

/// #mn-binding-select — per-head-neuron sensitivity of the BINDING worst-child
/// objective row. The dense head ReLU output `y` feeds the network's output
/// `Linear` (`output = W_out·y + b_out`), so the objective `Σ_o spec[o]·output[o]`
/// is linear in `y` with coefficient `c_k = Σ_o spec[o]·W_out[o,k]` on head neuron
/// `k`. The binding worst-child row is the objective with the SMALLEST baseline
/// lower bound (the closest-to-violating margin = the domain's critical row).
///
/// Returns `None` (⇒ the caller falls back to the objective-blind heuristic) when
/// the output `Linear` consuming the head cannot be UNIQUELY resolved (`in_features
/// == head_width && out_features == spec-row width`). That fallback is sound:
/// selection only changes which VALID facets are built, never their validity.
fn head_objective_coeffs(
    graph: &GraphNetwork,
    relu_node: &str,
    head_width: usize,
    objectives: &[Vec<f32>],
    baseline: &[(f32, f32)],
) -> Option<Vec<f64>> {
    // Binding worst-child row = argmin over finite baseline lower bounds.
    let worst = baseline
        .iter()
        .enumerate()
        .filter(|(_, &(l, _))| l.is_finite())
        .min_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(o, _)| o)?;
    let spec_row = objectives.get(worst)?;
    if spec_row.is_empty() {
        return None;
    }
    // The output Linear consuming the head ReLU: it takes `relu_node` as an input,
    // its in_features == head_width, and its out_features == the objective width
    // (num output classes). Require a UNIQUE such consumer (fail-closed otherwise).
    let mut matching = graph.nodes.values().filter(|n| {
        n.inputs.iter().any(|inp| inp == relu_node)
            && matches!(
                &n.layer,
                Layer::Linear(l)
                    if l.in_features() == head_width && l.out_features() == spec_row.len()
            )
    });
    let out_node = matching.next()?;
    if matching.next().is_some() {
        return None; // ambiguous output consumer ⇒ heuristic fallback
    }
    let Layer::Linear(lin) = &out_node.layer else {
        return None;
    };
    let w = &lin.weight; // shape [out_features, in_features] = [classes, head_width]
    let mut coeffs = vec![0.0f64; head_width];
    for (k, ck) in coeffs.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (o, &s) in spec_row.iter().enumerate() {
            acc += f64::from(s) * f64::from(w[[o, k]]);
        }
        *ck = acc;
    }
    Some(coeffs)
}

/// Run the objective backward with the pool injected at a fixed shared `β`, and
/// return the per-objective lower bounds (a sound LB vector for any `β ≥ 0`).
fn injected_lowers(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec: &ndarray::Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: Option<&GraphAlphaState>,
    pool: &MultiNeuronPool,
    deadline: Option<Instant>,
) -> Option<Vec<f32>> {
    let out = SpecCrownRequest::new(graph, input, spec, engine)
        .node_bounds(node_bounds)
        .alpha_state_opt(alpha_state)
        .deadline_opt(deadline)
        .mn_pool_opt(Some(pool))
        .run()
        .ok()?;
    Some(out.lower().iter().copied().collect())
}

/// Set the same `β` on every group in the pool.
fn set_shared_beta(pool: &mut MultiNeuronPool, beta: f32) {
    for g in pool.groups_mut() {
        let _ = g.set_beta(beta.max(0.0));
    }
}

/// Elementwise sound max: `acc[o] = max(acc[o], cand[o])`. Every operand is a
/// valid LB of objective `o`, so the max is a valid (tighter) LB.
fn max_into(acc: &mut [f32], cand: &[f32]) {
    for (a, &c) in acc.iter_mut().zip(cand.iter()) {
        if c.is_finite() && c > *a {
            *a = c;
        }
    }
}

fn min_of(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Increment-3 entry point: return per-objective bounds whose LOWER is the sound
/// max of the baseline and every multi-neuron-injected candidate. `upper` is
/// carried through unchanged (the injection only tightens the verification-side
/// lower bound). Returns `baseline` verbatim on any gate-off / unavailable /
/// error path (byte-identical when `NY_MULTINEURON != 1`).
#[allow(clippy::too_many_arguments)]
pub fn tighten_root_objective_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    engine: Option<&dyn GemmEngine>,
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: Option<&GraphAlphaState>,
    baseline: &[(f32, f32)],
    deadline: Option<Instant>,
) -> Vec<(f32, f32)> {
    if !enabled() {
        return baseline.to_vec();
    }
    // Scope to conv nets (the cifar100/tinyimagenet surface) by default; the
    // dense-fed pair machinery is layer-agnostic, so NY_MULTINEURON_MLP=1 opens
    // it to conv-free nets (relational difference nets). Default-OFF ⇒ inert.
    if !graph.has_conv_layers() && !mlp_enabled() {
        return baseline.to_vec();
    }
    let Some(alpha) = alpha_state else {
        return baseline.to_vec();
    };
    let Some(spec) = build_spec(objectives) else {
        return baseline.to_vec();
    };
    if spec.nrows() != baseline.len() {
        return baseline.to_vec();
    }
    let cfg = MnConfig::from_env();

    // Target ReLU nodes: the dense-fed head ReLUs (§5 increment 1 — the hardest
    // cifar100 objective's residual slack sits in the head). Conv-fed ReLUs are
    // opt-in (NY_MULTINEURON_CONV=1) this increment.
    let targets = target_relu_nodes(graph, &cfg);
    if targets.is_empty() {
        return baseline.to_vec();
    }

    // Accumulate the sound tightened lower vector across all targets.
    let mut acc_lower: Vec<f32> = baseline.iter().map(|&(l, _)| l).collect();
    let baseline_min = min_of(&acc_lower);
    eprintln!(
        "[multineuron] ===== ROOT INJECTION (NY_MULTINEURON=1) baseline margin_min={baseline_min:.5} targets={:?} =====",
        targets
    );

    for (relu_node, pre_node) in &targets {
        let Some((mut pool, kept)) = build_pool_for_node(
            graph,
            input,
            alpha,
            node_bounds,
            relu_node,
            pre_node,
            engine,
            &cfg,
            None,  // #mn-binding-select: heuristic ranking on the root (Increment 3) path
            false, // #mn-head-f64-certified-measure: legacy coupling_facets source (byte-identical)
        ) else {
            eprintln!(
                "[multineuron] node={relu_node} pre={pre_node}: no facet-carrying group (skip)"
            );
            continue;
        };
        eprintln!(
            "[multineuron] node={relu_node} pre={pre_node}: {} facets over {} pairs (top score={:.4})",
            pool.len(),
            kept.len(),
            kept.first().map(|s| s.score).unwrap_or(0.0),
        );

        // --- β line-search (sound: track per-objective max) ---
        let betas = [0.0f32, 0.5, 1.0, 2.0, 4.0];
        let mut best_min = min_of(&acc_lower);
        let mut best_beta = 0.0f32;
        for &b in &betas {
            set_shared_beta(&mut pool, b);
            if let Some(lows) = injected_lowers(
                graph,
                input,
                &spec,
                engine,
                node_bounds,
                Some(alpha),
                &pool,
                deadline,
            ) {
                max_into(&mut acc_lower, &lows);
                let this_min = min_of(&lows);
                if this_min > best_min {
                    best_min = this_min;
                    best_beta = b;
                }
                eprintln!(
                    "[multineuron]   beta={b:.2} -> injected margin_min={this_min:.5} (acc margin_min={:.5})",
                    min_of(&acc_lower)
                );
            }
        }

        // --- shared-β Adam ascent around the best grid point (§2.3/§2.4) ---
        // Gradient of min-margin w.r.t. β by central finite difference; EVERY
        // evaluated β is folded into the sound per-objective max, so Adam quality
        // only affects tightness, never soundness. 3 backwards/step (β±h + step).
        if cfg.adam_steps > 0 {
            let mut beta = best_beta.max(0.25);
            let (mut m, mut v) = (0.0f32, 0.0f32);
            let (b1, b2, eps, lr, h) = (0.9f32, 0.999f32, 1e-8f32, 0.5f32, 0.25f32);
            // Evaluate at β, fold into acc, and return this-β min-margin.
            let eval_fold = |beta: f32, pool: &mut MultiNeuronPool, acc: &mut Vec<f32>| -> f32 {
                set_shared_beta(pool, beta.max(0.0));
                match injected_lowers(
                    graph,
                    input,
                    &spec,
                    engine,
                    node_bounds,
                    Some(alpha),
                    pool,
                    deadline,
                ) {
                    Some(l) => {
                        max_into(acc, &l);
                        min_of(&l)
                    }
                    None => f32::NEG_INFINITY,
                }
            };
            for t in 1..=cfg.adam_steps {
                let f_plus = eval_fold((beta + h).max(0.0), &mut pool, &mut acc_lower);
                let f_minus = eval_fold((beta - h).max(0.0), &mut pool, &mut acc_lower);
                let grad = (f_plus - f_minus) / (2.0 * h); // ascend min-margin
                if !grad.is_finite() {
                    break;
                }
                m = b1 * m + (1.0 - b1) * grad;
                v = b2 * v + (1.0 - b2) * grad * grad;
                let mhat = m / (1.0 - b1.powi(t as i32)).max(f32::EPSILON);
                let vhat = v / (1.0 - b2.powi(t as i32)).max(f32::EPSILON);
                beta = (beta + lr * mhat / (vhat.sqrt() + eps)).clamp(0.0, 10.0);
                let _ = eval_fold(beta, &mut pool, &mut acc_lower);
                eprintln!(
                    "[multineuron]   adam[{t}] beta={beta:.3} grad={grad:.4} -> acc margin_min={:.5}",
                    min_of(&acc_lower)
                );
            }
        }
    }

    let final_min = min_of(&acc_lower);
    eprintln!(
        "[multineuron] ===== END ROOT INJECTION: margin_min {baseline_min:.5} -> {final_min:.5} (Δ={:+.5}) =====",
        final_min - baseline_min
    );

    // SOUNDNESS SELF-CHECK (NY_MULTINEURON_TRUESAT_CHECK=1): the tightened
    // certified per-objective LOWER bound must never exceed the TRUE margin at
    // any real input in the box (a false UNSAT catch — guard (a)/(d)). Sample
    // inputs, forward through the REAL network, and assert `acc_lower[o] ≤
    // true_margin_o + tol`. A violation means an unsound facet — reported loudly.
    if matches!(
        std::env::var("NY_MULTINEURON_TRUESAT_CHECK")
            .ok()
            .as_deref(),
        Some("1")
    ) {
        run_truesat_lb_check(graph, input, objectives, engine, &acc_lower);
    }

    // SOUNDNESS: acc_lower[o] is the max of the baseline lower and valid injected
    // LBs — never below baseline (max is monotone), so this can only tighten.
    baseline
        .iter()
        .zip(acc_lower.iter())
        .map(|(&(bl, bu), &nl)| (bl.max(nl), bu))
        .collect()
}

/// Dense-fed (and optionally conv-fed) ReLU nodes with their pre-activation node.
fn target_relu_nodes(graph: &GraphNetwork, cfg: &MnConfig) -> Vec<(String, String)> {
    let Ok(order) = graph.exec_order() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for name in order.iter() {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer, Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs.first() else {
            continue;
        };
        let pre_kind = graph.nodes.get(pre).map(|n| &n.layer);
        let is_dense = matches!(pre_kind, Some(Layer::Linear(_)));
        let is_conv = matches!(pre_kind, Some(Layer::Conv2d(_)));
        // MLP scope (NY_MULTINEURON_MLP): the ReLU's DIRECT input node IS its
        // pre-activation regardless of the producing op — the pool machinery is
        // node_bounds + generic graph-backward driven (combined_rows_octahedra
        // does `set_output(pre_node)`, capturing the full upstream linear map),
        // so it is layer-agnostic. Accept the relational ACAS pattern
        // MatMul→Add→ReLU (and unfused MatMul→ReLU / const-bias adds).
        let is_mlp_pre = mlp_enabled()
            && matches!(
                pre_kind,
                Some(Layer::Linear(_) | Layer::Add(_) | Layer::MatMul(_) | Layer::AddConstant(_))
            );
        if mlp_enabled() {
            eprintln!(
                "[multineuron] relu={name} pre={pre} pre_kind={} accepted={}",
                pre_kind.map(layer_kind).unwrap_or("<none>"),
                is_dense || (cfg.conv_groups && is_conv) || is_mlp_pre
            );
        }
        if is_dense || (cfg.conv_groups && is_conv) || is_mlp_pre {
            out.push((name.clone(), pre.clone()));
        }
    }
    out
}

/// Short discriminant string for the MLP-scope diagnostic trace.
fn layer_kind(l: &Layer) -> &'static str {
    match l {
        Layer::Linear(_) => "Linear",
        Layer::MatMul(_) => "MatMul",
        Layer::Add(_) => "Add",
        Layer::AddConstant(_) => "AddConstant",
        Layer::Conv2d(_) => "Conv2d",
        Layer::Sub(_) => "Sub",
        Layer::ReLU(_) => "ReLU",
        _ => "other",
    }
}

/// Sample real inputs in the box, forward them through the TRUE (ONNX-matching)
/// network, and return the WORST (smallest) true margin observed per objective —
/// i.e. `min_true[o] = min_s Σ_j obj_o[j]·f(u_s)[j]`, a sound *upper* estimate of
/// the true min margin (sampling only sees a subset of the box). Returns `None`
/// if NO sample could be forwarded (⇒ the caller must degrade to baseline: the
/// injected tightening is UNVERIFIABLE and therefore not acceptable — thesis).
///
/// Boundary-biased sampling (vertices / uniform / midpoint-perturbed) because the
/// tightest true margins sit at corners. FAITHFUL forward, not IBP, which widens
/// even a point box (BatchNorm soundness) and would understate the true margin.
fn sample_true_margins(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    engine: Option<&dyn GemmEngine>,
    n_samples: usize,
) -> Option<Vec<f32>> {
    let lo = input.flatten();
    let lo_v: Vec<f32> = lo.lower().iter().copied().collect();
    let hi_v: Vec<f32> = lo.upper().iter().copied().collect();
    let n = lo_v.len();
    let shape = input.lower().shape().to_vec();
    let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut min_true: Vec<f32> = vec![f32::INFINITY; objectives.len()];
    let mut forwarded = 0usize;

    for s in 0..n_samples {
        let mut u = Vec::with_capacity(n);
        let mode = s % 3;
        for k in 0..n {
            let val = match mode {
                0 => {
                    if next() < 0.5 {
                        lo_v[k]
                    } else {
                        hi_v[k]
                    }
                } // vertex
                1 => lo_v[k] + (hi_v[k] - lo_v[k]) as f64 as f32 * (next() as f32), // uniform
                _ => {
                    // Bit-identical sample anchor: f32::midpoint rounds differently at overflow/subnormal edges.
                    #[allow(clippy::manual_midpoint)]
                    let mid = 0.5 * (lo_v[k] + hi_v[k]);
                    let half = 0.5 * (hi_v[k] - lo_v[k]);
                    mid + half * (2.0 * next() as f32 - 1.0)
                }
            };
            u.push(val.clamp(lo_v[k], hi_v[k]));
        }
        let Ok(arr) = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), u) else {
            continue;
        };
        let Ok(pt) = BoundedTensor::new(arr.clone(), arr) else {
            continue;
        };
        let Ok(out) = graph.propagate_concrete_point(&pt, engine, None) else {
            continue;
        };
        forwarded += 1;
        let out_flat = out.flatten();
        let ov = out_flat.lower();
        for (o, obj) in objectives.iter().enumerate() {
            let mut m = 0.0f32;
            for (j, &c) in obj.iter().enumerate() {
                m += c * ov.get(j).copied().unwrap_or(0.0);
            }
            if m < min_true[o] {
                min_true[o] = m;
            }
        }
    }
    if forwarded == 0 {
        return None;
    }
    Some(min_true)
}

/// GUARD 2 — per-objective TRUE-SAT reject mask + sound fold. For each objective
/// `o`, fold the candidate injected LB `cand[o]` into `acc[o]` (via the sound max)
/// **only if** it passes the sampled-true-margin ceiling: `cand[o] ≤ min_true[o] +
/// tol`. Any candidate that EXCEEDS the sampled true margin (a would-be false
/// UNSAT) is REJECTED — `acc[o]` keeps its baseline/previously-accepted value.
/// Returns the number of REJECTED objectives (0 ⇒ all injected values passed).
///
/// `tol` is OUTWARD (favors rejection): the default is `0.0`, so ANY exceedance of
/// the sampled true margin rejects; a larger `tol` only LOOSENS acceptance and is
/// never required for tightness. This is the invariant that makes a layout/index/
/// node error (RISK 1 / RISK 2) degrade to "no tightening", never a lifted LB.
pub(crate) fn max_into_true_sat_masked(
    acc: &mut [f32],
    cand: &[f32],
    min_true: &[f32],
    tol: f32,
) -> usize {
    let mut rejected = 0usize;
    for (o, (a, &c)) in acc.iter_mut().zip(cand.iter()).enumerate() {
        if !c.is_finite() || c <= *a {
            continue; // no candidate gain to accept (baseline/prev already ≥ it).
        }
        let ceil = min_true.get(o).copied().unwrap_or(f32::NEG_INFINITY);
        if c <= ceil + tol {
            *a = c; // sound: passed the true-net sample check.
        } else {
            rejected += 1;
            if rejected <= 8 {
                eprintln!(
                    "[multineuron-stem TRUESAT] REJECT obj[{o}]: injected_LB={c} > sampled_true_margin={ceil} (+tol={tol}) — kept baseline (false-UNSAT catch)"
                );
            }
        }
    }
    rejected
}

/// Sound-LB self-check (diagnostic, non-stem path): confirm the tightened
/// certified lower bound stays ≤ the sampled true margin per objective. Prints
/// the tightest observed slack; SHOUTS on any violation (an unsound facet → a
/// potential false UNSAT). Purely observational — it does NOT alter bounds.
fn run_truesat_lb_check(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    engine: Option<&dyn GemmEngine>,
    acc_lower: &[f32],
) {
    let n_samples = env_usize("NY_MULTINEURON_TRUESAT_SAMPLES", 2000);
    let Some(min_true) = sample_true_margins(graph, input, objectives, engine, n_samples) else {
        eprintln!("[multineuron TRUESAT] no sample could be forwarded — check SKIPPED");
        return;
    };
    let mut violations = 0usize;
    let mut worst_slack = f32::INFINITY;
    for (o, &m) in min_true.iter().enumerate() {
        let slack = m - acc_lower.get(o).copied().unwrap_or(f32::NEG_INFINITY);
        if slack < worst_slack {
            worst_slack = slack;
        }
        if slack < -1e-3 {
            violations += 1;
            if violations <= 5 {
                eprintln!(
                    "[multineuron TRUESAT] VIOLATION obj[{o}]: certified_LB={} > true_margin={m} (slack={slack:.5})",
                    acc_lower.get(o).copied().unwrap_or(f32::NAN)
                );
            }
        }
    }
    let min_true_min = min_of(&min_true);
    let acc_min = min_of(acc_lower);
    eprintln!(
        "[multineuron TRUESAT] samples={n_samples} min_sampled_true_margin={min_true_min:.5} certified_LB_min={acc_min:.5} worst_slack={worst_slack:.5} violations={violations}"
    );
    if violations == 0 {
        eprintln!("[multineuron TRUESAT] PASS — certified LB stays <= true margin on all samples (no false UNSAT)");
    } else {
        eprintln!("[multineuron TRUESAT] *** FAIL *** {violations} sampled inputs fell below the certified LB — UNSOUND facet");
    }
}

// ===========================================================================
// STEM-RESIDENT lever (`NY_MULTINEURON_STEM`) — thread the coupling facets into
// the OPTIMIZED resident GPU backward at the stem ReLU (Relu_2), not the loose
// fresh side-backward. See `docs/CERTIFIED_CUT_CROWN_DESIGN.md`.
// ===========================================================================

/// Convert a stem `MultiNeuronPool` into a resident cut-fold entry at shared
/// `beta ≥ 0` (the resident mirror of `inject_post_terms_before_relu` +
/// `inject_pre_terms_after_relu`). PostActivation terms feed `coeffs` (the
/// `+β·g_i` ReLU-OUTPUT channel), PreActivation terms feed `pre_coeffs` (the
/// `+β·a_i` ReLU-INPUT channel), and each group contributes `−β·b_c` to the
/// lower bias. Reuses the same finiteness guards (`c.is_finite() && c != 0.0`)
/// and the `β ≥ 0` clamp. `sound_round = true` selects the outward-rounded,
/// error-widened resident fold (production soundness).
pub(crate) fn pool_to_resident_fold(
    pool: &MultiNeuronPool,
    beta: f32,
) -> ny_core::resident_cut_fold::ResidentCutFold {
    use std::collections::HashMap;
    let beta = beta.max(0.0);
    let mut post: HashMap<u32, f64> = HashMap::new();
    let mut pre: HashMap<u32, f64> = HashMap::new();
    let mut bias_shift = 0.0f64;
    if beta.is_finite() && beta != 0.0 {
        for g in pool.groups() {
            for t in g.terms() {
                let c = beta * t.coefficient;
                if !c.is_finite() || c == 0.0 {
                    continue;
                }
                let idx = t.neuron_idx as u32;
                match t.var {
                    MnVar::PostActivation => {
                        *post.entry(idx).or_insert(0.0) += f64::from(c);
                    }
                    MnVar::PreActivation => {
                        *pre.entry(idx).or_insert(0.0) += f64::from(c);
                    }
                }
            }
            let bd = -beta * g.bias();
            if bd.is_finite() {
                bias_shift += f64::from(bd);
            }
        }
    }
    let mut coeffs: Vec<(u32, f32)> = post.into_iter().map(|(n, c)| (n, c as f32)).collect();
    coeffs.sort_unstable_by_key(|&(n, _)| n);
    let mut pre_coeffs: Vec<(u32, f32)> = pre.into_iter().map(|(n, c)| (n, c as f32)).collect();
    pre_coeffs.sort_unstable_by_key(|&(n, _)| n);
    ny_core::resident_cut_fold::ResidentCutFold {
        coeffs,
        bias_shift: bias_shift as f32,
        pre_coeffs,
        sound_round: true,
    }
}

// ===========================================================================
// Retired #mn-head-facet research construction. Build the head pool once and
// retain the f64 `HeadF64Fold` representation for non-authoritative inspection.
// The shared ny-core proof-path reader is hard-quarantined and never publishes
// these registry entries into critical-row recovery.
// ===========================================================================

/// f64 unit roundoff `u = 2⁻⁵³` — the head-fold build-error twin of the recovery's
/// `U_F64`, used to track the OUTWARD accumulation rounding of each fold coeff.
const HEAD_U_F64: f64 = f64::from_bits(0x3CA0_0000_0000_0000);

/// Env gate: the HEAD coupling-facet lever (`NY_MN_HEAD_FACET=1`, default-OFF).
/// Rides `NY_F64_LINEAGE_RECOVER=1` (the fold only enters the armed f64 recovery).
/// Byte-identical when unset (nothing registered ⇒ the recovery arm is untouched).
pub fn head_facet_enabled() -> bool {
    ny_core::head_f64_fold::head_f64_fold_enabled()
}

/// Legacy #mn-head-f64-certified-measure request (research only).
///
/// It may build an exact-facet registry entry for inspection, but cannot arm
/// the hard-quarantined CPU-f64 proof-path reader or affect a verdict.
pub fn head_f64_certified_measure_enabled() -> bool {
    matches!(
        std::env::var("NY_MN_HEAD_F64_CERTIFIED_MEASURE")
            .ok()
            .as_deref(),
        Some("1")
    )
}

/// The β-grid for the head fold (default `0.5, 1.0, 2.0`; each independently
/// sound, the recovery `max`es over them). Overridable via `NY_MN_HEAD_FACET_BETAS`
/// (comma-separated, positive finite).
fn head_facet_betas() -> Vec<f32> {
    std::env::var("NY_MN_HEAD_FACET_BETAS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .collect::<Vec<f32>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![0.5, 1.0, 2.0])
}

/// The graph's ReLU node names in f64-recovery FOLD order (output→input): the
/// reverse of `exec_order` filtered to ReLU. Best-effort for locating the head
/// ReLU (fold index ~0, right below the output Gemm); the recovery site
/// re-validates the resolved `target_act` against the lane's real `relu_names`
/// (by name) and fails closed on any mismatch — so a mis-derivation here can only
/// cost the tightening, never soundness.
fn relu_names_fold_order(graph: &GraphNetwork) -> Vec<String> {
    let Ok(order) = graph.exec_order() else {
        return Vec::new();
    };
    let mut relus: Vec<String> = order
        .iter()
        .filter(|n| {
            matches!(
                graph.nodes.get(n.as_str()).map(|nd| &nd.layer),
                Some(Layer::ReLU(_))
            )
        })
        .cloned()
        .collect();
    relus.reverse();
    relus
}

/// The dense (`Linear`-fed) head ReLU + its pre-activation node. The dense head
/// `Gemm→ReLU→Gemm` is the ONLY `Linear`-fed ReLU in a conv resnet ⇒ the FIRST
/// dense-fed target is the head. Overridable by ReLU name via
/// `NY_MN_HEAD_FACET_NODE`. Only dense-fed heads are eligible: the pool
/// `neuron_idx` (row-major over the pre-node) and the f64 recovery row column
/// share a C-order flat index ONLY when the pre-node is dense (identity flatten),
/// so the GUARD3 col2im permutation hazard cannot arise (charter §1).
fn resolve_head_dense_relu(graph: &GraphNetwork, cfg: &MnConfig) -> Option<(String, String)> {
    let override_name = std::env::var("NY_MN_HEAD_FACET_NODE")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let mut dense = target_relu_nodes(graph, cfg)
        .into_iter()
        .filter(|(_, pre)| {
            matches!(
                graph.nodes.get(pre).map(|n| &n.layer),
                Some(Layer::Linear(_))
            )
        });
    match override_name {
        Some(name) => dense.find(|(relu, _)| relu == &name),
        None => dense.next(),
    }
}

/// Reduce a head `MultiNeuronPool` to a single f64 [`HeadF64Fold`] at shared
/// `β ≥ 0` (the head mirror of `pool_to_resident_fold`), keeping products in f64
/// (EXACT for `f32×f32`: ≤48 mantissa bits ≤ 53) and tracking the f64 accumulation
/// rounding OUTWARD per key (the moat — injected into the recovery's certified err
/// channel so the fold LB stays rigorous w.r.t. the EXACT `β·g` multipliers).
/// PostActivation terms → `post` (`+β·g_i`), PreActivation → `pre` (`+β·a_i`), and
/// each group contributes `−β·b_c` to `bias_shift`. Out-of-range / non-finite /
/// exact-zero terms are dropped (fail-closed).
fn pool_to_head_f64_fold(
    pool: &MultiNeuronPool,
    beta: f32,
    relu_node: &str,
    head_width: usize,
    target_act: usize,
) -> ny_core::head_f64_fold::HeadF64Fold {
    let beta = beta.max(0.0);
    let mut post: HashMap<u32, (f64, f64)> = HashMap::new();
    let mut pre: HashMap<u32, (f64, f64)> = HashMap::new();
    let mut bias_shift = 0.0f64;
    let mut bias_err = 0.0f64;
    if beta.is_finite() && beta != 0.0 {
        let bf = f64::from(beta);
        for g in pool.groups() {
            for t in g.terms() {
                if t.neuron_idx >= head_width {
                    continue; // out-of-range never applied (fail-closed)
                }
                // EXACT product (f32×f32 in f64) — no representation error here; the
                // only rounding is the per-key accumulation `+=` below.
                let p = bf * f64::from(t.coefficient);
                if !p.is_finite() || p == 0.0 {
                    continue;
                }
                let idx = t.neuron_idx as u32;
                let slot = match t.var {
                    MnVar::PostActivation => post.entry(idx),
                    MnVar::PreActivation => pre.entry(idx),
                }
                .or_insert((0.0, 0.0));
                let ns = slot.0 + p;
                // |fl(s+p) − (s+p)| ≤ u·|fl(s+p)|; Σ over adds bounds |f64 − exact|.
                slot.1 += HEAD_U_F64 * ns.abs();
                slot.0 = ns;
            }
            let bd = -bf * f64::from(g.bias());
            if bd.is_finite() {
                let nb = bias_shift + bd;
                bias_err += HEAD_U_F64 * nb.abs();
                bias_shift = nb;
            }
        }
    }
    // Drop keys that reduced to an exact-zero coeff with zero error (no-op adds).
    post.retain(|_, v| v.0 != 0.0 || v.1 != 0.0);
    pre.retain(|_, v| v.0 != 0.0 || v.1 != 0.0);
    ny_core::head_f64_fold::HeadF64Fold {
        target_act,
        relu_name: relu_node.to_string(),
        head_width,
        post,
        pre,
        bias_shift,
        bias_err,
    }
}

/// #mn-head-facet increment 1 — build the HEAD coupling-facet β-grid ONCE at root
/// and reduce each β to an f64 [`HeadF64Fold`]. Reuses the EXISTING pool builder
/// (`build_pool_for_node` → `coupling_facets`) and mirrors the stem driver's
/// producer. `None` when there is no dense head or no facet-carrying group (⇒
/// nothing registered ⇒ byte-identical). Sound by construction: every facet is a
/// proven superset half-space; the fold is the Lagrangian embedding at β ≥ 0.
///
/// `binding_coeffs` / `use_certified` (#mn-head-f64-certified-measure): when the
/// research measurement gate is armed the caller passes the binding worst-child
/// objective coeffs (heuristic-vs-binding pair RANKING) and `use_certified = true`
/// (draw facets from the EXACT-certified `certified_coupling_facets_exact` source).
/// `None` + `false` (every non-measurement caller) reproduces the heuristic +
/// legacy-`coupling_facets` path byte-identically. Both only affect WHICH valid
/// facets enter the sound monotone-max fold, never their validity.
pub fn build_head_f64_fold(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: &GraphAlphaState,
    engine: Option<&dyn GemmEngine>,
    binding_coeffs: Option<&[f64]>,
    use_certified: bool,
) -> Option<Vec<ny_core::head_f64_fold::HeadF64Fold>> {
    let cfg = MnConfig::from_env();
    let (relu_node, pre_node) = resolve_head_dense_relu(graph, &cfg)?;
    let head_width = pre_width(node_bounds, &pre_node)?;
    let fold_order = relu_names_fold_order(graph);
    let target_act = fold_order.iter().position(|n| n == &relu_node)?;
    let (pool, kept) = build_pool_for_node(
        graph,
        input,
        alpha_state,
        node_bounds,
        &relu_node,
        &pre_node,
        engine,
        &cfg,
        binding_coeffs, // #mn-head-f64-certified-measure: binding-select ranking (or None ⇒ heuristic)
        use_certified, // #mn-head-f64-certified-measure: exact-certified source (or false ⇒ coupling_facets)
    )?;
    let betas = head_facet_betas();
    let mut folds = Vec::new();
    for &b in &betas {
        let fold = pool_to_head_f64_fold(&pool, b, &relu_node, head_width, target_act);
        if !fold.is_empty() {
            folds.push(fold);
        }
    }
    if folds.is_empty() {
        return None;
    }
    eprintln!(
        "[mn-head-facet] built {} fold(s): head relu={relu_node} pre={pre_node} width={head_width} target_act={target_act} from {} facets over {} pairs betas={betas:?}",
        folds.len(),
        pool.len(),
        kept.len(),
    );
    Some(folds)
}

/// Build + register the retired HEAD coupling-facet β-grid for raw research
/// inspection. The shared `ny-core` proof-path reader is hard-quarantined, so a
/// populated registry cannot affect per-subdomain f64 recovery. Always clears
/// any stale research entry first.
///
/// `objectives` threads the output objective spec so the #mn-head-f64-certified-
/// measure research lane (`NY_MN_HEAD_F64_CERTIFIED_MEASURE=1`) can rank candidate
/// head pairs by their joint effect on the binding worst-child margin (via the
/// existing [`head_objective_coeffs`]) and draw facets from the exact-certified
/// source. This changes only the inert research registry; proof-bearing recovery
/// remains unchanged.
pub fn install_head_f64_fold(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: Option<&GraphAlphaState>,
    engine: Option<&dyn GemmEngine>,
) {
    ny_core::head_f64_fold::clear_head_f64_folds();
    let Some(alpha) = alpha_state else {
        eprintln!("[mn-head-facet] no root alpha state (skip; recovery stays baseline)");
        return;
    };

    // #mn-head-f64-certified-measure (NY_MN_HEAD_F64_CERTIFIED_MEASURE=1, dark,
    // default-OFF, RESEARCH ONLY — NOT verdict re-authorization). When armed:
    //  * draw facets from the EXACT rational-arithmetic certifier
    //    (`certified_coupling_facets_exact`) instead of the legacy research
    //    producer — every pool facet is a proven-superset half-space over all four
    //    ReLU orthants; and
    //  * RANK candidate pairs by their joint effect on the binding worst-child
    //    margin (the same #mn-binding-select coeffs the head-resident lever uses),
    //    computed here from a plain root backward baseline (argmin-lower row).
    // Both only reorder / re-source WHICH valid facets enter the sound f64
    // monotone-max fold; soundness is independent of either choice. When the gate
    // is OFF this whole block is skipped ⇒ `None` + `false` ⇒ byte-identical.
    let use_certified = head_f64_certified_measure_enabled();
    let binding_coeffs: Option<Vec<f64>> = if use_certified {
        let cfg = MnConfig::from_env();
        match resolve_head_dense_relu(graph, &cfg).and_then(|(relu_node, pre_node)| {
            let head_width = pre_width(node_bounds, &pre_node)?;
            // Baseline per-objective lowers via a plain optimized root backward (no
            // fold registered) — used ONLY to pick the binding worst-child row
            // (argmin lower). Selection-only ⇒ cannot affect facet validity.
            let spec = build_spec(objectives)?;
            let lows = injected_lowers_resident(
                graph,
                input,
                &spec,
                engine,
                node_bounds,
                Some(alpha),
                None,
            )?;
            let baseline: Vec<(f32, f32)> = lows.iter().map(|&l| (l, f32::INFINITY)).collect();
            head_objective_coeffs(graph, &relu_node, head_width, objectives, &baseline)
        }) {
            Some(c) => {
                eprintln!(
                    "[mn-head-f64-certified-measure] ON: {} head objective coeffs from the binding worst-child row; ranking pairs by |c_i|·|c_j|·directional-corner-gap; facet source = certified_coupling_facets_exact",
                    c.len()
                );
                Some(c)
            }
            None => {
                eprintln!("[mn-head-f64-certified-measure] ON but could not resolve head objective coeffs (no unique output Linear consumer) — falling back to the heuristic ranking (still exact-certified facets; sound either way)");
                None
            }
        }
    } else {
        None
    };

    match build_head_f64_fold(
        graph,
        input,
        node_bounds,
        alpha,
        engine,
        binding_coeffs.as_deref(),
        use_certified,
    ) {
        Some(folds) => {
            eprintln!(
                "[mn-head-facet] registered {} head fold(s) (certified_measure={use_certified})",
                folds.len()
            );
            ny_core::head_f64_fold::set_head_f64_folds(folds);
        }
        None => {
            eprintln!(
                "[mn-head-facet] no dense head / facet-carrying group (recovery stays baseline)"
            )
        }
    }
}

/// Run the OPTIMIZED objective backward (NO `mn_pool` — so the resident fast
/// path is NOT disabled) with whatever cut fold is currently registered in the
/// `ny-gpu` resident registry, and return the per-objective lower bounds. This
/// is the same optimized pass that produces the competitive baseline; the fold
/// (if any) is read inside the resident lane via `active_resident_cut_fold`.
fn injected_lowers_resident(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec: &ndarray::Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: Option<&GraphAlphaState>,
    deadline: Option<Instant>,
) -> Option<Vec<f32>> {
    let out = SpecCrownRequest::new(graph, input, spec, engine)
        .node_bounds(node_bounds)
        .alpha_state_opt(alpha_state)
        .deadline_opt(deadline)
        .run()
        .ok()?;
    Some(out.lower().iter().copied().collect())
}

/// Resolve the stem ReLU node and its pre-activation producer. Bypasses
/// `target_relu_nodes` (which only returns dense-fed head ReLUs) — the stem
/// ReLU is conv/BatchNorm-fed. The ReLU node name is `NY_MULTINEURON_STEM_NODES`
/// (default `Relu_2`); the pre-node is that ReLU's first graph input.
fn resolve_stem_node(graph: &GraphNetwork) -> Option<(String, String)> {
    let relu = std::env::var("NY_MULTINEURON_STEM_NODES")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Relu_2".to_string());
    let node = graph.nodes.get(&relu)?;
    if !matches!(node.layer, Layer::ReLU(_)) {
        return None;
    }
    let pre = node.inputs.first()?.clone();
    Some((relu, pre))
}

/// GUARD 1 — the node the RESIDENT lane actually folds at: `cut_fold_target =
/// total-1` = the LAST `Activation` in resident fold order, which (the resident
/// backward runs output→input) is the network's FIRST ReLU in EXECUTION order.
/// We recompute that name independently from the graph's topological `exec_order`
/// (dependencies precede dependents ⇒ the first ReLU encountered is the earliest-
/// computed one). The stem driver ASSERTS `resolve_stem_node`'s ReLU equals this;
/// on any mismatch it refuses to register a fold (RISK 1 → baseline, not a fold
/// of one ReLU's cut coefficients applied to a DIFFERENT ReLU's neurons).
fn first_relu_in_fold_order(graph: &GraphNetwork) -> Option<String> {
    let order = graph.exec_order().ok()?;
    for name in order.iter() {
        if let Some(node) = graph.nodes.get(name) {
            if matches!(node.layer, Layer::ReLU(_)) {
                return Some(name.clone());
            }
        }
    }
    None
}

/// GUARD 1 (HEAD variant) — the node the HEAD-RESIDENT lane actually folds at:
/// `cut_fold_target = 0` = the FIRST `Activation` in resident fold order, which
/// (the resident backward runs output→input) is the network's LAST ReLU in
/// EXECUTION order (the dense `Gemm→ReLU→Gemm` head). We recompute that name
/// independently from the graph's topological `exec_order` (dependencies precede
/// dependents ⇒ the LAST ReLU encountered is the latest-computed = the head).
/// The head driver ASSERTS `resolve_head_dense_relu`'s ReLU equals this; on any
/// mismatch it refuses to register a fold (RISK 1 → baseline, never a fold of one
/// ReLU's cut coefficients applied to a DIFFERENT ReLU's neurons = false UNSAT).
fn head_relu_in_fold_order(graph: &GraphNetwork) -> Option<String> {
    let order = graph.exec_order().ok()?;
    order.iter().rev().find_map(|name| {
        graph
            .nodes
            .get(name)
            .filter(|node| matches!(node.layer, Layer::ReLU(_)))
            .map(|_| name.clone())
    })
}

/// GUARD 3 (HEAD layout precondition): the CPU pool `neuron_idx` is a C-order
/// flat column over the head `pre_node`'s LOGICAL shape; the GPU-resident fold
/// applies that SAME index to the head ReLU's post-activation frontier column.
/// For a DENSE (`Linear`-fed) head the flatten is the IDENTITY (the pre-node
/// bounds are rank-1 `[width]`), so the two indices are trivially equal — the
/// GUARD3 col2im permutation hazard that makes the conv/stem port hard CANNOT
/// arise (charter §1). We enforce that precondition (Linear producer + rank-1
/// bounds whose width matches the frontier) so a non-dense head can never fold a
/// permuted cut onto the wrong neurons.
fn head_layout_precondition_ok(
    graph: &GraphNetwork,
    node_bounds: &HashMap<String, BoundedTensor>,
    pre_node: &str,
) -> bool {
    let producer_ok = graph
        .nodes
        .get(pre_node)
        .map(|n| matches!(n.layer, Layer::Linear(_)))
        .unwrap_or(false);
    let shape_ok = node_bounds
        .get(pre_node)
        .map(|bt| matches!(bt.shape(), [_w] | [1, _w]))
        .unwrap_or(false);
    producer_ok && shape_ok
}

/// GUARD 3 (layout precondition — RISK 2): the CPU pool `neuron_idx` is a C-order
/// flat column over `pre_node`'s LOGICAL shape (`top_unstable` enumerates
/// `bt.flatten().iter()`, which walks logical row-major); the GPU-resident fold
/// applies that SAME index to its frontier column, which the resident conv reshape
/// decodes channel-major-then-row-major over `(C,H,W)` (`CONV_RESHAPE_SHADER` +
/// col2im). The layout proof shows those two formulas are IDENTICAL
/// (`i = c·H·W + h·W + w`) **iff** `pre_node` is a genuine NCHW channel-first conv
/// producer — `Conv2d` or (NY does not fuse Conv+BN) `BatchNorm` — whose bounds are
/// rank-3 `(C,H,W)` (or rank-4 `(1,C,H,W)`). For any other producer/shape (e.g. a
/// `Gemm`/`Linear`-fed ReLU, or a non-NCHW reshape) the two flattens can differ by
/// a permutation ⇒ the cut lands on the wrong neurons ⇒ invalid Lagrangian ⇒ a
/// potential false UNSAT. We enforce the proof's precondition in code so it can
/// never be silently violated by an override or a future importer.
pub(crate) fn stem_layout_precondition_ok(
    graph: &GraphNetwork,
    node_bounds: &HashMap<String, BoundedTensor>,
    pre_node: &str,
) -> bool {
    let producer_ok = graph
        .nodes
        .get(pre_node)
        .map(|n| matches!(n.layer, Layer::Conv2d(_) | Layer::BatchNorm(_)))
        .unwrap_or(false);
    let shape_ok = node_bounds
        .get(pre_node)
        .map(|bt| matches!(bt.shape(), [_c, _h, _w] | [1, _c, _h, _w]))
        .unwrap_or(false);
    producer_ok && shape_ok
}

/// STEM-RESIDENT entry point (`NY_MULTINEURON_STEM=1`, default-OFF): tighten the
/// per-objective LOWER bounds by threading the stem coupling facets into the
/// OPTIMIZED resident backward via the `ny-gpu` cut-fold registry.
///
/// # Invariants
/// * **INV-A (β=0 reproduces baseline exactly):** `β = 0` is NEVER re-run — the
///   accumulator starts AS the baseline lower vector and only ever takes a sound
///   `max` with an injected candidate. The final combine is `baseline.max(acc)`,
///   so with the gate off, no facets, or an all-loose grid the result is the
///   baseline verbatim.
/// * **INV-B (β>0 only tightens):** every injected candidate is a valid LB for
///   ANY `β ≥ 0` (each facet is a proven superset half-space); the per-objective
///   `max` is monotone, so the result can only rise. A looser β loses the max.
/// * **INV-C (outward rounding):** the resident fold uses `sound_round = true`
///   (outward coeff/bias rounding + error widening in `ny-gpu`).
#[allow(clippy::too_many_arguments)]
pub fn tighten_root_objective_bounds_stem_resident(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    engine: Option<&dyn GemmEngine>,
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: Option<&GraphAlphaState>,
    baseline: &[(f32, f32)],
    deadline: Option<Instant>,
) -> Vec<(f32, f32)> {
    if !stem_enabled() {
        return baseline.to_vec();
    }
    if !graph.has_conv_layers() {
        return baseline.to_vec();
    }
    let Some(alpha) = alpha_state else {
        return baseline.to_vec();
    };
    let Some(spec) = build_spec(objectives) else {
        return baseline.to_vec();
    };
    if spec.nrows() != baseline.len() {
        return baseline.to_vec();
    }
    let Some((relu_node, pre_node)) = resolve_stem_node(graph) else {
        eprintln!("[multineuron-stem] no stem ReLU node resolved (skip)");
        return baseline.to_vec();
    };

    // GUARD 1 (node identity — RISK 1): the resident lane HARDWIRES its fold to
    // `cut_fold_target = total-1` = the network's FIRST ReLU in fold order. The
    // pool + coeffs here are built for `resolve_stem_node`'s ReLU. If those two
    // ReLUs DIFFER (e.g. an `NY_MULTINEURON_STEM_NODES` override that is not the
    // first ReLU), the fold would apply THIS ReLU's cut coefficients to a
    // DIFFERENT ReLU's neurons ⇒ invalid Lagrangian ⇒ potential false UNSAT. So
    // we refuse to register any fold on mismatch — degrade to baseline, loudly.
    match first_relu_in_fold_order(graph) {
        Some(fold_relu) if fold_relu == relu_node => {
            eprintln!("[multineuron-stem] GUARD1 OK: stem node={relu_node} IS the resident fold target (first ReLU in fold order)");
        }
        Some(fold_relu) => {
            eprintln!(
                "[multineuron-stem] *** GUARD1 REFUSE *** stem node={relu_node} != resident fold target={fold_relu} (first ReLU in fold order); the resident lane would fold {relu_node}'s coefficients onto {fold_relu}'s neurons — returning baseline (no tightening)"
            );
            return baseline.to_vec();
        }
        None => {
            eprintln!("[multineuron-stem] *** GUARD1 REFUSE *** could not resolve the network's first ReLU in fold order — returning baseline (no tightening)");
            return baseline.to_vec();
        }
    }

    // GUARD 3 (layout precondition — RISK 2): the CPU pool index and the GPU-resident
    // frontier column are PROVEN-EQUAL only when `pre_node` is a genuine NCHW
    // channel-first conv producer (`Conv2d`/`BatchNorm`) with rank-3 `(C,H,W)` bounds.
    // Enforce that precondition; refuse (baseline) on anything else so a non-NCHW
    // stem can never fold a permuted cut onto the wrong neurons.
    if !stem_layout_precondition_ok(graph, node_bounds, &pre_node) {
        let shape = node_bounds.get(&pre_node).map(|bt| bt.shape().to_vec());
        eprintln!(
            "[multineuron-stem] *** GUARD3 REFUSE *** pre={pre_node} (bounds shape={shape:?}) is not a rank-3 NCHW (C,H,W) Conv2d/BatchNorm output; the CPU pool index and the GPU-resident frontier column can differ by a permutation (RISK 2) — returning baseline (no tightening)"
        );
        return baseline.to_vec();
    }
    eprintln!("[multineuron-stem] GUARD3 OK: pre={pre_node} is a rank-3 NCHW (C,H,W) conv producer — CPU↔GPU-resident column layout PROVEN-EQUAL");

    let cfg = MnConfig::from_env();

    // GUARD 2 (MANDATORY true-SAT reject-guard — RISK 2 + everything else): with
    // the stem gate ON we ALWAYS sample the TRUE network up front and only fold an
    // injected value that stays ≤ the sampled true margin (per objective). If NO
    // sample can be forwarded, the tightening is UNVERIFIABLE ⇒ baseline. This is
    // NOT behind `NY_MULTINEURON_TRUESAT_CHECK`; it is unconditional on this lever.
    let truesat_samples = env_usize("NY_MULTINEURON_STEM_TRUESAT_SAMPLES", 4000);
    let truesat_tol = env_f32("NY_MULTINEURON_STEM_TRUESAT_TOL", 0.0);
    let Some(min_true) = sample_true_margins(graph, input, objectives, engine, truesat_samples)
    else {
        eprintln!("[multineuron-stem] *** GUARD2 REFUSE *** true-SAT sampling forwarded 0 inputs (unverifiable) — returning baseline (no tightening)");
        return baseline.to_vec();
    };

    // INV-A: acc starts AS the baseline lower vector — β=0 is never re-run.
    let mut acc_lower: Vec<f32> = baseline.iter().map(|&(l, _)| l).collect();
    let baseline_min = min_of(&acc_lower);
    eprintln!(
        "[multineuron-stem] ===== STEM-RESIDENT INJECTION (NY_MULTINEURON_STEM=1) baseline margin_min={baseline_min:.5} node={relu_node} pre={pre_node} truesat_samples={truesat_samples} truesat_tol={truesat_tol} min_sampled_true_margin={:.5} ====="
        , min_of(&min_true)
    );

    let Some((mut pool, kept)) = build_pool_for_node(
        graph,
        input,
        alpha,
        node_bounds,
        &relu_node,
        &pre_node,
        engine,
        &cfg,
        None, // #mn-binding-select: stem lever keeps the heuristic (binding-select rides the head)
        false, // #mn-head-f64-certified-measure: stem lever keeps legacy coupling_facets (byte-identical)
    ) else {
        eprintln!("[multineuron-stem] node={relu_node}: no facet-carrying group (skip)");
        return baseline.to_vec();
    };
    eprintln!(
        "[multineuron-stem] node={relu_node}: {} facets over {} pairs (top score={:.4})",
        pool.len(),
        kept.len(),
        kept.first().map(|s| s.score).unwrap_or(0.0),
    );

    // β grid EXCLUDING 0 (β=0 == baseline, already folded in as `acc`). Each β>0
    // registers the resident fold, re-runs the OPTIMIZED backward, and folds the
    // sound per-objective max. `set_shared_beta` keeps the pool β in sync (used
    // for logging / potential future ascent); the fold uses the explicit β.
    let betas = [0.5f32, 1.0, 2.0, 4.0];
    for &b in &betas {
        set_shared_beta(&mut pool, b);
        let fold = pool_to_resident_fold(&pool, b);
        ny_core::resident_cut_fold::set_resident_cut_fold(fold);
        let lows = injected_lowers_resident(
            graph,
            input,
            &spec,
            engine,
            node_bounds,
            Some(alpha),
            deadline,
        );
        ny_core::resident_cut_fold::clear_resident_cut_fold();
        if let Some(lows) = lows {
            // GUARD 2: fold ONLY the per-objective injected values that pass the
            // true-net sample ceiling; a too-high value (RISK 2 layout/index
            // error) is REJECTED and `acc[o]` keeps its baseline. Every value now
            // in `acc` has passed a true-net sample check ⇒ never a lifted LB.
            let rejected = max_into_true_sat_masked(&mut acc_lower, &lows, &min_true, truesat_tol);
            eprintln!(
                "[multineuron-stem]   beta={b:.2} -> injected margin_min={:.5} rejected={rejected} (acc margin_min={:.5})",
                min_of(&lows),
                min_of(&acc_lower)
            );
        }
    }

    let final_min = min_of(&acc_lower);
    eprintln!(
        "[multineuron-stem] ===== END STEM-RESIDENT INJECTION: margin_min {baseline_min:.5} -> {final_min:.5} (Δ={:+.5}) =====",
        final_min - baseline_min
    );

    // Post-hoc diagnostic (already GUARD-2-protected above; this only re-reports).
    if matches!(
        std::env::var("NY_MULTINEURON_TRUESAT_CHECK")
            .ok()
            .as_deref(),
        Some("1")
    ) {
        run_truesat_lb_check(graph, input, objectives, engine, &acc_lower);
    }

    // INV-A/INV-B: `bl.max(nl)` — never below the baseline lower (monotone max).
    baseline
        .iter()
        .zip(acc_lower.iter())
        .map(|(&(bl, bu), &nl)| (bl.max(nl), bu))
        .collect()
}

// ===========================================================================
// #mn-head-resident — the UNMASKED head lever (`NY_MN_HEAD_RESIDENT`). Thread
// the HEAD coupling facets into the OPTIMIZED resident GPU backward, RETARGETED
// from the stem (`cut_fold_target = total-1`) to the head (fold index `0`), so
// the facet rides the tight GPU baseline itself instead of the (masked) CPU f64
// recovery. Single-global sound; GUARD1 refuses the false-UNSAT node mismatch.
// ===========================================================================

/// HEAD-RESIDENT entry point (`NY_MN_HEAD_RESIDENT=1`, default-OFF): tighten the
/// per-objective LOWER bounds by threading the HEAD coupling facets into the
/// OPTIMIZED resident backward via the `ny-gpu` cut-fold registry, retargeted to
/// the head act. The mirror of [`tighten_root_objective_bounds_stem_resident`]
/// with the HEAD resolver + HEAD GUARD1/GUARD3.
///
/// # Invariants (identical to the stem lever's; the fold math is unchanged, only
/// the TARGET act moves from `total-1` to `0` — see `head_resident_retarget_enabled`)
/// * **INV-A (β=0 == baseline):** `acc` starts AS the baseline lower vector; β=0
///   is never re-run; the final combine is `baseline.max(acc)`.
/// * **INV-B (β>0 only tightens):** every injected candidate is a valid LB for any
///   `β ≥ 0` (each head facet is a proven superset half-space, valid at root ⇒
///   valid on every subdomain); the per-objective `max` is monotone.
/// * **INV-C (outward rounding):** the resident fold uses `sound_round = true`.
/// * **GUARD1 (node identity):** `resolve_head_dense_relu`'s ReLU MUST equal the
///   network's LAST ReLU in exec order (= the retarget's fold-index-0 act); on
///   mismatch we register NO fold (baseline) — the false-UNSAT hazard is refused.
/// * **GUARD2 (true-SAT mask):** every accepted injected value passed a true-net
///   sample ceiling ⇒ a layout/index error can only fail to tighten, never lift.
#[allow(clippy::too_many_arguments)]
pub fn tighten_root_objective_bounds_head_resident(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    engine: Option<&dyn GemmEngine>,
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha_state: Option<&GraphAlphaState>,
    baseline: &[(f32, f32)],
    deadline: Option<Instant>,
) -> Vec<(f32, f32)> {
    if !head_resident_enabled() {
        return baseline.to_vec();
    }
    if !graph.has_conv_layers() {
        return baseline.to_vec();
    }
    let Some(alpha) = alpha_state else {
        return baseline.to_vec();
    };
    let Some(spec) = build_spec(objectives) else {
        return baseline.to_vec();
    };
    if spec.nrows() != baseline.len() {
        return baseline.to_vec();
    }
    let cfg = MnConfig::from_env();
    let Some((relu_node, pre_node)) = resolve_head_dense_relu(graph, &cfg) else {
        eprintln!("[mn-head-resident] no dense head ReLU resolved (skip)");
        return baseline.to_vec();
    };

    // GUARD 1 (node identity — RISK 1): the resident lane folds at fold-order
    // index 0 = the network's LAST ReLU in exec order (the head). The pool +
    // coeffs here are built for `resolve_head_dense_relu`'s ReLU. If those two
    // ReLUs DIFFER, the fold would apply THIS ReLU's cut coefficients to a
    // DIFFERENT ReLU's neurons ⇒ invalid Lagrangian ⇒ potential false UNSAT.
    // Refuse to register any fold on mismatch — degrade to baseline, loudly.
    match head_relu_in_fold_order(graph) {
        Some(fold_relu) if fold_relu == relu_node => {
            eprintln!("[mn-head-resident] GUARD1 OK: head node={relu_node} IS the resident fold target (last ReLU in exec order = fold index 0)");
        }
        Some(fold_relu) => {
            eprintln!(
                "[mn-head-resident] *** GUARD1 REFUSE *** head node={relu_node} != resident fold target={fold_relu} (last ReLU in exec order); the resident lane would fold {relu_node}'s coefficients onto {fold_relu}'s neurons — returning baseline (no tightening)"
            );
            return baseline.to_vec();
        }
        None => {
            eprintln!("[mn-head-resident] *** GUARD1 REFUSE *** could not resolve the network's last ReLU in fold order — returning baseline (no tightening)");
            return baseline.to_vec();
        }
    }

    // GUARD 3 (HEAD layout precondition): the head must be DENSE (`Linear`-fed,
    // rank-1 bounds) so the CPU pool index and the GPU-resident frontier column
    // share the same C-order flat index (identity flatten). Refuse otherwise.
    if !head_layout_precondition_ok(graph, node_bounds, &pre_node) {
        let shape = node_bounds.get(&pre_node).map(|bt| bt.shape().to_vec());
        eprintln!(
            "[mn-head-resident] *** GUARD3 REFUSE *** pre={pre_node} (bounds shape={shape:?}) is not a rank-1 dense `Linear` head output; the CPU pool index and the GPU-resident frontier column can differ by a permutation — returning baseline (no tightening)"
        );
        return baseline.to_vec();
    }
    eprintln!("[mn-head-resident] GUARD3 OK: pre={pre_node} is a rank-1 dense Linear head — CPU↔GPU-resident column layout PROVEN-EQUAL (identity flatten)");

    // GUARD 2 (MANDATORY true-SAT reject-guard): sample the TRUE network up front
    // and only fold an injected value that stays ≤ the sampled true margin (per
    // objective). No sample ⇒ unverifiable ⇒ baseline. Unconditional on this lever.
    let truesat_samples = env_usize("NY_MN_HEAD_RESIDENT_TRUESAT_SAMPLES", 4000);
    let truesat_tol = env_f32("NY_MN_HEAD_RESIDENT_TRUESAT_TOL", 0.0);
    let Some(min_true) = sample_true_margins(graph, input, objectives, engine, truesat_samples)
    else {
        eprintln!("[mn-head-resident] *** GUARD2 REFUSE *** true-SAT sampling forwarded 0 inputs (unverifiable) — returning baseline (no tightening)");
        return baseline.to_vec();
    };

    // INV-A: acc starts AS the baseline lower vector — β=0 is never re-run.
    let mut acc_lower: Vec<f32> = baseline.iter().map(|&(l, _)| l).collect();
    let baseline_min = min_of(&acc_lower);
    eprintln!(
        "[mn-head-resident] ===== HEAD-RESIDENT INJECTION (NY_MN_HEAD_RESIDENT=1) baseline margin_min={baseline_min:.5} node={relu_node} pre={pre_node} truesat_samples={truesat_samples} truesat_tol={truesat_tol} min_sampled_true_margin={:.5} ====="
        , min_of(&min_true)
    );

    // #mn-binding-select (NY_MN_BINDING_SELECT=1, rides this head-resident lever):
    // rank candidate pairs by their JOINT effect on the BINDING worst-child margin
    // instead of the objective-blind excluded-corner heuristic. Compute the per-
    // head-neuron objective sensitivity of the binding row (argmin baseline lower)
    // and pass it into `build_pool_for_node`. Default OFF ⇒ `None` ⇒ byte-identical
    // heuristic selection. This only reorders WHICH valid facets are kept, so the
    // soundness argument (every kept facet is a proven superset half-space,
    // validated by the enclosure/MC oracles) is UNCHANGED.
    let binding_coeffs: Option<Vec<f64>> = if binding_select_enabled() {
        match pre_width(node_bounds, &pre_node)
            .and_then(|w| head_objective_coeffs(graph, &relu_node, w, objectives, baseline))
        {
            Some(c) => {
                eprintln!(
                    "[mn-binding-select] ON: {} head objective coeffs from the binding worst-child row (argmin baseline lower); ranking pairs by |c_i|·|c_j|·directional-corner-gap",
                    c.len()
                );
                Some(c)
            }
            None => {
                eprintln!("[mn-binding-select] ON but could not resolve head objective coeffs (no unique output Linear consumer) — falling back to the excluded-corner heuristic (sound: valid facets either way)");
                None
            }
        }
    } else {
        None
    };

    let Some((mut pool, kept)) = build_pool_for_node(
        graph,
        input,
        alpha,
        node_bounds,
        &relu_node,
        &pre_node,
        engine,
        &cfg,
        binding_coeffs.as_deref(),
        false, // #mn-head-f64-certified-measure: the UNMASKED resident lever keeps legacy coupling_facets
    ) else {
        eprintln!("[mn-head-resident] node={relu_node}: no facet-carrying group (skip)");
        return baseline.to_vec();
    };
    eprintln!(
        "[mn-head-resident] node={relu_node}: {} facets over {} pairs (top score={:.4})",
        pool.len(),
        kept.len(),
        kept.first().map(|s| s.score).unwrap_or(0.0),
    );

    // β grid EXCLUDING 0 (β=0 == baseline). Each β>0 registers the resident fold
    // (reduced via the EXISTING `pool_to_resident_fold`), re-runs the OPTIMIZED
    // backward (which retargets the fold to the head act while NY_MN_HEAD_RESIDENT
    // is armed), and folds the sound per-objective max.
    let betas = [0.5f32, 1.0, 2.0, 4.0];
    for &b in &betas {
        set_shared_beta(&mut pool, b);
        let fold = pool_to_resident_fold(&pool, b);
        ny_core::resident_cut_fold::set_resident_cut_fold(fold);
        let lows = injected_lowers_resident(
            graph,
            input,
            &spec,
            engine,
            node_bounds,
            Some(alpha),
            deadline,
        );
        ny_core::resident_cut_fold::clear_resident_cut_fold();
        if let Some(lows) = lows {
            let rejected = max_into_true_sat_masked(&mut acc_lower, &lows, &min_true, truesat_tol);
            eprintln!(
                "[mn-head-resident]   beta={b:.2} -> injected margin_min={:.5} rejected={rejected} (acc margin_min={:.5})",
                min_of(&lows),
                min_of(&acc_lower)
            );
        }
    }

    let final_min = min_of(&acc_lower);
    eprintln!(
        "[mn-head-resident] ===== END HEAD-RESIDENT INJECTION: margin_min {baseline_min:.5} -> {final_min:.5} (Δ={:+.5}) =====",
        final_min - baseline_min
    );

    // INV-A/INV-B: `bl.max(nl)` — never below the baseline lower (monotone max).
    baseline
        .iter()
        .zip(acc_lower.iter())
        .map(|(&(bl, bu), &nl)| (bl.max(nl), bu))
        .collect()
}

/// Test accessor: the ReLU node `resolve_head_dense_relu` would pick (default cfg).
#[cfg(test)]
pub(crate) fn resolve_head_dense_relu_for_test(graph: &GraphNetwork) -> Option<String> {
    let cfg = MnConfig::from_env();
    resolve_head_dense_relu(graph, &cfg).map(|(relu, _)| relu)
}

/// Test accessor: the resident head fold target (last ReLU in exec order = fold 0).
#[cfg(test)]
pub(crate) fn head_relu_in_fold_order_for_test(graph: &GraphNetwork) -> Option<String> {
    head_relu_in_fold_order(graph)
}

/// Test accessor (#mn-binding-select): the kept `(i, j, ranking_score)` triples in
/// RANKED order for one ReLU node. `binding_coeffs = None` reproduces the heuristic
/// (excluded-corner) ranking — its `ranking_score` IS `excluded_corner_score`;
/// `Some(c)` applies the objective-gradient-informed ranking. Exercises the exact
/// selection `build_pool_for_node` performs.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pool_pairs_for_test(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: &HashMap<String, BoundedTensor>,
    relu_node: &str,
    pre_node: &str,
    binding_coeffs: Option<&[f64]>,
    use_certified: bool,
) -> Vec<(usize, usize, f64)> {
    let cfg = MnConfig::from_env();
    build_pool_for_node(
        graph,
        input,
        alpha_state,
        node_bounds,
        relu_node,
        pre_node,
        None,
        &cfg,
        binding_coeffs,
        use_certified,
    )
    .map(|(_, kept)| kept.iter().map(|s| (s.i, s.j, s.score)).collect())
    .unwrap_or_default()
}

/// Test accessor (#mn-head-f64-certified-measure): the kept `(i, j, octahedron)`
/// triples in RANKED order for one ReLU node, exercising the EXACT selection +
/// facet-SOURCE (`use_certified`) `build_pool_for_node` performs. Returning the
/// producer octahedron lets a test recompute the pool's facet source
/// (`certified_coupling_facets_exact` when `use_certified`, else `coupling_facets`)
/// and run the enclosure oracle against it.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pool_octahedra_for_test(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: &HashMap<String, BoundedTensor>,
    relu_node: &str,
    pre_node: &str,
    binding_coeffs: Option<&[f64]>,
    use_certified: bool,
) -> Vec<(usize, usize, Octahedron2)> {
    let cfg = MnConfig::from_env();
    build_pool_for_node(
        graph,
        input,
        alpha_state,
        node_bounds,
        relu_node,
        pre_node,
        None,
        &cfg,
        binding_coeffs,
        use_certified,
    )
    .map(|(_, kept)| kept.iter().map(|s| (s.i, s.j, s.p.clone())).collect())
    .unwrap_or_default()
}

/// Test accessor (#mn-binding-select): the objective-gradient-informed ranking
/// key for a pair, exposed for the constructed-fixture unit test.
#[cfg(test)]
pub(crate) fn binding_rank_score_for_test(
    p: &Octahedron2,
    i: usize,
    j: usize,
    coeffs: &[f64],
) -> f64 {
    binding_rank_score(p, i, j, coeffs)
}

/// Test accessor (#mn-binding-select): the per-head-neuron objective coefficient
/// vector for the binding worst-child row (or `None` if the output Linear consumer
/// is not uniquely resolvable).
#[cfg(test)]
pub(crate) fn head_objective_coeffs_for_test(
    graph: &GraphNetwork,
    relu_node: &str,
    head_width: usize,
    objectives: &[Vec<f32>],
    baseline: &[(f32, f32)],
) -> Option<Vec<f64>> {
    head_objective_coeffs(graph, relu_node, head_width, objectives, baseline)
}

fn build_spec(objectives: &[Vec<f32>]) -> Option<ndarray::Array2<f32>> {
    if objectives.is_empty() {
        return None;
    }
    let cols = objectives[0].len();
    let mut data = Vec::with_capacity(objectives.len() * cols);
    for o in objectives {
        if o.len() != cols {
            return None;
        }
        data.extend_from_slice(o);
    }
    ndarray::Array2::from_shape_vec((objectives.len(), cols), data).ok()
}
