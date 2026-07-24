// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STABILIZE-AND-FIX (#stabilize, dark `NY_STABILIZE=<budget_secs>`, default
//! OFF ⇒ byte-identical).
//!
//! On deep-ResNet roots many ReLUs are *unstable* (stored pre-activation bounds
//! straddle 0), and each one forces a relaxation triangle into every downstream
//! bound plus a branching candidate. This module spends a bounded root budget
//! proving individual unstable neurons actually STABLE over the root box: it
//! ranks unstable neurons by expected impact, recomputes the top pre-activation
//! nodes with a per-target α-CROWN backward (the existing certified
//! directed-rounding machinery), intersects shrink-only into the stored
//! `initial_node_bounds` entry, and treats a neuron as FIXED iff the STORED
//! post-merge pair satisfies `l ≥ 0` (always-active) or `u ≤ 0`
//! (always-inactive). Because every relaxation consumer
//! (`constraints/backward/relu.rs`, the GPU Activation extraction) and every
//! branching scan (`find_unstable_graph_neurons*`) reads that same stored
//! entry, a fixed neuron automatically gets the exact identity/zero
//! linearization and drops out of branching candidacy on every descendant
//! domain — there is no side-channel "fixed list" that could desynchronize
//! from the proof: the stored bound IS the proof artifact.
//!
//! Loop (≤ `NY_STABILIZE_ROUNDS` rounds, while budget remains and the last
//! round converted > 0 neurons):
//!   1. SCAN unstable neurons per ReLU pre-activation (`l < 0 < u`, the same
//!      predicate as `branching/graph.rs`).
//!   2. RANK pre-activation nodes by `S_p = Σ g_i`, `g_i = m_i · intercept_i`
//!      with `intercept_i = (−l·u)/(u−l)` (the triangle looseness a fix
//!      deletes) and `m_i` the margin-weight proxy for the last ReLU
//!      (`compute_margin_weights`) or 1 upstream; filter to plausibly-stable
//!      neurons `min(|l|,u) ≤ κ·(u−l)` (`NY_STABILIZE_KAPPA`, default 0.25).
//!      Ranking is cost-only — it chooses WHERE to spend budget, never touches
//!      a bound.
//!   3. TIGHTEN the ranked nodes via
//!      `GraphNetwork::stabilize_tighten_targets_with_alpha` (frozen reference
//!      snapshot, equal-share per-node deadlines, intersect-only merge, the
//!      `tighten_preactivations_with_alpha` error taxonomy).
//!   4. FIX: read back the stored post-merge pairs; record conversions; tighten
//!      each candidate ReLU's stored POST-activation entry to the exact
//!      monotone image `[relu(l), relu(u)]` ∩ inherited.
//!   5. RECOMPUTE DOWNSTREAM: one forward IBP resweep intersecting the stored
//!      map (`resweep_stored_bounds_ibp`), or per-target CROWN on downstream
//!      ReLU pre-activations with `NY_STABILIZE_RECOMPUTE=crown`.
//!
//! SOUNDNESS (zero new trust surface): every bound written is (a) the output
//! of the existing certified directed-rounding α-CROWN backward, (b) a
//! per-element intersection of two sound enclosures of the same reachable set
//! (shrink-only, NaN/disjoint ⇒ keep/union), or (c) the exact monotone ReLU
//! image of a sound pre-activation enclosure. Fixing a proven-stable neuron
//! does not change the true function on the box, so no verdict can be
//! corrupted by the fix itself. A NOT-proven neuron keeps a straddling stored
//! bound and therefore keeps its triangle and its branching candidacy —
//! structurally impossible to fix it. Budget/deadline expiry anywhere yields
//! partial work with every completed merge sound.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::Array2;
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;

use crate::bounds::GraphAlphaState;
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

use super::super::propagation::batched::interm_refine::{
    compute_margin_weights, find_last_relu_seed,
};

/// Which side a fixed neuron was proven stable on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StableSide {
    /// `l ≥ 0` over the root box: ReLU == identity there.
    Active,
    /// `u ≤ 0` over the root box: ReLU == zero there.
    Inactive,
}

/// Parsed gate + knobs snapshot (the `IntermRefineOptions` pattern): read the
/// environment ONCE at the call site so the loop never re-parses mid-flight.
#[derive(Debug, Clone)]
pub(crate) struct StabilizeOptions {
    /// Total wall-clock budget for the whole loop (`NY_STABILIZE`, seconds).
    pub(crate) budget: Duration,
    /// Max loop rounds (`NY_STABILIZE_ROUNDS`, default 3).
    pub(crate) max_rounds: usize,
    /// Plausibly-stable filter `min(|l|,u) ≤ κ·(u−l)` (`NY_STABILIZE_KAPPA`,
    /// default 0.25). Cost-only: chooses WHERE to spend budget.
    pub(crate) kappa: f32,
    /// `NY_STABILIZE_PROBE=1`: per-round stderr telemetry.
    pub(crate) probe: bool,
    /// `NY_STABILIZE_RECOMPUTE=crown`: downstream recompute via per-target
    /// CROWN instead of the default IBP resweep.
    pub(crate) recompute_crown: bool,
}

impl StabilizeOptions {
    /// Parse the primary gate value. `None` (absent), `0`, negative, or
    /// unparseable ⇒ gate OFF (the caller must not touch anything).
    pub(crate) fn parse(raw: Option<&str>) -> Option<Self> {
        let secs = raw?.trim().parse::<f64>().ok()?;
        if !secs.is_finite() || secs <= 0.0 {
            return None;
        }
        Some(Self {
            budget: Duration::from_secs_f64(secs),
            max_rounds: std::env::var("NY_STABILIZE_ROUNDS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&r| r >= 1)
                .unwrap_or(3),
            kappa: std::env::var("NY_STABILIZE_KAPPA")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|k| k.is_finite() && *k > 0.0)
                .unwrap_or(0.25),
            probe: std::env::var("NY_STABILIZE_PROBE").ok().as_deref() == Some("1"),
            recompute_crown: std::env::var("NY_STABILIZE_RECOMPUTE").ok().as_deref()
                == Some("crown"),
        })
    }

    fn from_env() -> Option<Self> {
        Self::parse(std::env::var("NY_STABILIZE").ok().as_deref())
    }
}

/// What the loop did — feeds the probe and the root-pass log line. The FIXES
/// themselves live in the stored bounds map (the proof artifact); this report
/// is telemetry only and no relaxation/branching consumer ever reads it.
#[derive(Debug, Default)]
pub(crate) struct StabilizeReport {
    /// Rounds that ran (scan + tighten + fix + recompute).
    pub(crate) rounds: usize,
    /// `(relu_node, neuron_idx, side)` for every neuron whose STORED
    /// pre-activation pair satisfied the stable predicate after a merge.
    pub(crate) fixed: Vec<(String, usize, StableSide)>,
}

/// One ranked pre-activation node: the tighten target plus its ReLU
/// consumer(s) and the unstable-neuron indices observed at scan time.
struct StabilizeCandidate {
    pre_node: String,
    relu_nodes: Vec<String>,
    unstable: Vec<usize>,
    score: f64,
}

/// Env-gated entry point (`NY_STABILIZE=<budget_secs>`). Absent / 0 /
/// unparseable ⇒ returns `None` WITHOUT touching `bounds` (byte-identical).
#[allow(clippy::too_many_arguments)]
pub(crate) fn stabilize_and_fix_from_env(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    engine: Option<&dyn GemmEngine>,
    global_deadline: Option<Instant>,
    root_alpha_state: Option<&GraphAlphaState>,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> Option<StabilizeReport> {
    let opts = StabilizeOptions::from_env()?;
    Some(stabilize_and_fix(
        graph,
        input,
        objectives,
        engine,
        global_deadline,
        root_alpha_state,
        bounds,
        &opts,
    ))
}

/// The stabilize-and-fix loop. See the module doc for the algorithm and the
/// soundness invariants. A pre-expired budget/global deadline returns with
/// `bounds` untouched; mid-loop expiry yields partial (sound) work.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stabilize_and_fix(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    engine: Option<&dyn GemmEngine>,
    global_deadline: Option<Instant>,
    root_alpha_state: Option<&GraphAlphaState>,
    bounds: &mut HashMap<String, BoundedTensor>,
    opts: &StabilizeOptions,
) -> StabilizeReport {
    let mut report = StabilizeReport::default();
    let start = Instant::now();
    let mut deadline = start + opts.budget;
    if let Some(g) = global_deadline {
        deadline = deadline.min(g);
    }
    if deadline <= start {
        // Pre-expired: touch nothing (oracle 6 / byte-identical contract).
        return report;
    }
    // The tighten lane is the α-CROWN backward; without a root alpha state
    // (non-alpha routes) there is nothing sound to run — skip untouched.
    let Some(alpha) = root_alpha_state else {
        if opts.probe {
            eprintln!("[stabilize] no root alpha state (non-α route) — skipping");
        }
        return report;
    };
    let Ok(exec_order) = graph.exec_order() else {
        return report;
    };
    let exec_order: Vec<String> = exec_order.to_vec();

    // m_i proxy: margin weights apply only to the LAST ReLU's pre-activation
    // (the `compute_margin_weights` tail-linear contract); upstream ReLUs use
    // m_i = 1 (width-intercept ranking). Cost-only — never touches a bound.
    let output_node = if graph.output_name().is_empty() {
        exec_order.last().cloned().unwrap_or_default()
    } else {
        graph.output_name().to_string()
    };
    let last_pre = find_last_relu_seed(graph, &output_node).map(|(_relu, seed)| seed);
    let margin_w = match (&last_pre, build_spec_matrix_local(objectives)) {
        (Some(_), Some(spec)) => compute_margin_weights(graph, &output_node, &spec),
        _ => None,
    };

    for _round in 0..opts.max_rounds {
        if Instant::now() >= deadline {
            break;
        }
        let candidates = scan_and_rank(
            graph,
            &exec_order,
            bounds,
            opts.kappa,
            last_pre.as_deref(),
            margin_w.as_deref(),
        );
        if candidates.is_empty() {
            break;
        }
        report.rounds += 1;
        let round_start = Instant::now();
        let before: Vec<(String, usize, usize)> = if opts.probe {
            candidates
                .iter()
                .map(|c| {
                    let (unst, total) = unstable_count(bounds.get(&c.pre_node));
                    (c.pre_node.clone(), unst, total)
                })
                .collect()
        } else {
            Vec::new()
        };

        // [3] TIGHTEN the ranked pre-activation nodes (one backward per node
        // refreshes all of its neurons at once) under equal-share windows.
        let targets: Vec<String> = candidates.iter().map(|c| c.pre_node.clone()).collect();
        graph
            .stabilize_tighten_targets_with_alpha(input, &targets, alpha, engine, deadline, bounds);

        // [4] FIX: a neuron is fixed iff the STORED post-merge pair proves it
        // stable — no clamp beyond what was proven; the stored bound IS the
        // proof artifact every downstream consumer reads.
        let mut conversions = 0usize;
        for cand in &candidates {
            let Some(pre_bt) = bounds.get(&cand.pre_node) else {
                continue;
            };
            let lows: Vec<f32> = pre_bt.lower().iter().copied().collect();
            let ups: Vec<f32> = pre_bt.upper().iter().copied().collect();
            for &idx in &cand.unstable {
                let (Some(&l), Some(&u)) = (lows.get(idx), ups.get(idx)) else {
                    continue;
                };
                let side = if l >= 0.0 {
                    Some(StableSide::Active)
                } else if u <= 0.0 {
                    Some(StableSide::Inactive)
                } else {
                    None
                };
                if let Some(side) = side {
                    conversions += 1;
                    for relu_name in &cand.relu_nodes {
                        report.fixed.push((relu_name.clone(), idx, side));
                    }
                }
            }
            // Exact monotone post-activation image [relu(l), relu(u)] ∩
            // inherited (the interm_refine module-doc rule): max(·,0) is exact
            // in f32, and ReLU is monotone, so the image encloses the true
            // post-activation set whenever the pre-activation enclosure does.
            let post_l = pre_bt.lower().mapv(|v| v.max(0.0));
            let post_u = pre_bt.upper().mapv(|v| v.max(0.0));
            let Ok(post_img) = BoundedTensor::new(post_l, post_u) else {
                continue;
            };
            for relu_name in &cand.relu_nodes {
                let Some(stored_post) = bounds.get(relu_name) else {
                    continue;
                };
                if stored_post.shape() != post_img.shape() {
                    continue;
                }
                if let Some((tightened, _disjoint)) =
                    stored_post.intersection_per_element(&post_img)
                {
                    bounds.insert(relu_name.clone(), tightened);
                }
            }
        }

        // [5] RECOMPUTE DOWNSTREAM: propagate the deleted triangle looseness
        // into every downstream stored entry so the next round's scan sees
        // newly-stabilizable neurons.
        if opts.recompute_crown {
            // NY_STABILIZE_RECOMPUTE=crown: per-target CROWN on every ReLU
            // pre-activation downstream of the earliest tightened target
            // (same intersect-only harness as step [3]).
            let downstream = downstream_relu_preactivations(graph, &exec_order, &targets);
            if !downstream.is_empty() {
                graph.stabilize_tighten_targets_with_alpha(
                    input,
                    &downstream,
                    alpha,
                    engine,
                    deadline,
                    bounds,
                );
            }
        } else {
            let _ = graph.resweep_stored_bounds_ibp(input, engine, Some(deadline), bounds);
        }

        if opts.probe {
            for (pre_node, unst_before, total) in &before {
                let (unst_after, _) = unstable_count(bounds.get(pre_node.as_str()));
                eprintln!(
                    "[stabilize] round={} node={} unstable {} -> {} (of {})",
                    report.rounds, pre_node, unst_before, unst_after, total
                );
            }
            eprintln!(
                "[stabilize] round={} candidates={} conversions={} fixed_total={} round_secs={:.3}",
                report.rounds,
                candidates.len(),
                conversions,
                report.fixed.len(),
                round_start.elapsed().as_secs_f64()
            );
        }
        // [6] ITERATE while conversions occur (and budget remains).
        if conversions == 0 {
            break;
        }
    }
    if opts.probe {
        eprintln!(
            "[stabilize] done rounds={} fixed={} total_secs={:.3}",
            report.rounds,
            report.fixed.len(),
            start.elapsed().as_secs_f64()
        );
    }
    report
}

/// Steps [1]+[2]: scan exec-order ReLU pre-activations for unstable neurons
/// and rank the pre-activation NODES by aggregated gap attribution
/// `S_p = Σ m_i · intercept_i` over the plausibly-stable subset. Read-only.
fn scan_and_rank(
    graph: &GraphNetwork,
    exec_order: &[String],
    bounds: &HashMap<String, BoundedTensor>,
    kappa: f32,
    last_pre: Option<&str>,
    margin_w: Option<&[f32]>,
) -> Vec<StabilizeCandidate> {
    let mut out: Vec<StabilizeCandidate> = Vec::new();
    for name in exec_order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer, Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs.first() else {
            continue;
        };
        // The network input box is the property region — never a tighten target.
        if pre == NETWORK_INPUT {
            continue;
        }
        // Two ReLUs sharing one pre-activation: extend the existing candidate.
        if let Some(existing) = out.iter_mut().find(|c| &c.pre_node == pre) {
            existing.relu_nodes.push(name.clone());
            continue;
        }
        let Some(bt) = bounds.get(pre) else {
            continue;
        };
        let mut unstable = Vec::new();
        let mut score = 0.0f64;
        for (i, (&l, &u)) in bt.lower().iter().zip(bt.upper().iter()).enumerate() {
            if !(l < 0.0 && u > 0.0) {
                continue; // same predicate as branching/graph.rs:59
            }
            unstable.push(i);
            let width = u - l;
            if !width.is_finite() || width <= 0.0 {
                continue;
            }
            // Plausibly-stable filter: a bound that barely crosses 0 is the
            // conversion candidate. Cost-only (chooses WHERE to spend budget).
            if l.abs().min(u) > kappa * width {
                continue;
            }
            let intercept = (-(l as f64) * (u as f64)) / width as f64;
            let m = match (last_pre, margin_w) {
                (Some(lp), Some(w)) if lp == pre.as_str() => {
                    w.get(i).copied().unwrap_or(1.0) as f64
                }
                _ => 1.0,
            };
            score += m * intercept;
        }
        if unstable.is_empty() || score <= 0.0 {
            continue;
        }
        out.push(StabilizeCandidate {
            pre_node: pre.clone(),
            relu_nodes: vec![name.clone()],
            unstable,
            score,
        });
    }
    // Deterministic ranking: descending score, name tie-break.
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pre_node.cmp(&b.pre_node))
    });
    out
}

/// ReLU pre-activation nodes strictly AFTER the earliest tightened target in
/// exec order (the downstream cone approximation for the `crown` recompute),
/// excluding the targets themselves (already tightened this round).
fn downstream_relu_preactivations(
    graph: &GraphNetwork,
    exec_order: &[String],
    targets: &[String],
) -> Vec<String> {
    let earliest = exec_order
        .iter()
        .position(|n| targets.iter().any(|t| t == n));
    let Some(earliest) = earliest else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for name in exec_order.iter().skip(earliest + 1) {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer, Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs.first() else {
            continue;
        };
        if pre == NETWORK_INPUT || targets.iter().any(|t| t == pre) || out.contains(pre) {
            continue;
        }
        out.push(pre.clone());
    }
    out
}

/// Unstable-neuron count (`l < 0 < u`, the `probe_unstable_frac` predicate).
fn unstable_count(bt: Option<&BoundedTensor>) -> (usize, usize) {
    let Some(bt) = bt else {
        return (0, 0);
    };
    let mut unst = 0usize;
    let mut n = 0usize;
    for (&l, &u) in bt.lower().iter().zip(bt.upper().iter()) {
        n += 1;
        if l < 0.0 && u > 0.0 {
            unst += 1;
        }
    }
    (unst, n)
}

/// Local copy of the multi-objective spec-matrix builder (that one is scoped
/// `pub(in ...::graph)` inside a private `multi_objective::shared` module).
/// Rows = objectives; `None` on empty/ragged input.
fn build_spec_matrix_local(objectives: &[Vec<f32>]) -> Option<Array2<f32>> {
    if objectives.is_empty() {
        return None;
    }
    let output_dim = objectives[0].len();
    let mut data = Vec::with_capacity(objectives.len() * output_dim);
    for obj in objectives {
        if obj.len() != output_dim {
            return None;
        }
        data.extend_from_slice(obj);
    }
    Array2::from_shape_vec((objectives.len(), output_dim), data).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    use crate::beta_crown::config::BetaCrownConfig;
    use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::network::GraphNode;
    use crate::{AlphaCrownConfig, BetaCrownVerifier};

    fn test_opts() -> StabilizeOptions {
        StabilizeOptions {
            budget: Duration::from_secs(30),
            max_rounds: 3,
            kappa: 0.25,
            probe: false,
            recompute_crown: false,
        }
    }

    /// Fixture A (convertible): x ∈ [-1,1] →
    ///   lin1 (W=[[1],[1]], b=[3,3])  → pre1 = (x+3, x+3) ∈ [2,4]² (stable-active)
    ///   relu1
    ///   lin2 (W=[[1,-1]], b=[1.9])   → y = 1.9 EXACTLY (cancellation);
    ///                                  IBP sees [-0.1, 3.9] (straddles 0)
    ///   relu2
    ///   lin3 (W=[[1]])               → output
    /// The α-CROWN backward for lin2 cancels the coefficients through the
    /// stable-active relu1 and proves y ≥ ~1.9 > 0 ⇒ relu2/0 is fixable.
    fn build_convertible_graph() -> (GraphNetwork, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "lin1",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0_f32], [1.0]]), Some(arr1(&[3.0_f32, 3.0]))).unwrap(),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["lin1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "lin2",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0_f32, -1.0]]), Some(arr1(&[1.9_f32]))).unwrap(),
            ),
            vec!["relu1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["lin2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "lin3",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
            vec!["relu2".to_string()],
        ));
        graph.set_output("lin3");

        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid input box");
        (graph, input)
    }

    /// Fixture B (genuinely unstable): x ∈ [-1,1] →
    ///   lin1 (W=[[1],[-1]]) → pre1 = (x, -x)
    ///   relu1
    ///   lin2 (W=[[1,-1]])   → y = relu(x) − relu(−x) = x, TRUE range [-1,1]
    ///   relu2
    ///   lin3 (W=[[1]])
    /// relu2/0 truly straddles 0 — no sound tightening can ever fix it.
    fn build_straddling_graph() -> (GraphNetwork, BoundedTensor) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "lin1",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), None).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["lin1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "lin2",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32, -1.0]]), None).unwrap()),
            vec!["relu1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["lin2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "lin3",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
            vec!["relu2".to_string()],
        ));
        graph.set_output("lin3");

        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid input box");
        (graph, input)
    }

    fn root_alpha_for(graph: &GraphNetwork, input: &BoundedTensor) -> GraphAlphaState {
        let cfg = AlphaCrownConfig {
            iterations: 1,
            adaptive_skip: false,
            adaptive_skip_pilot: false,
            ..Default::default()
        };
        let (_bounds, alpha) = graph
            .collect_alpha_crown_bounds_dag_with_engine(input, &cfg, None)
            .expect("alpha warmup should succeed on the toy graph");
        alpha
    }

    fn clone_bits(map: &HashMap<String, BoundedTensor>) -> HashMap<String, (Vec<u32>, Vec<u32>)> {
        map.iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    (
                        v.lower().iter().map(|f| f.to_bits()).collect(),
                        v.upper().iter().map(|f| f.to_bits()).collect(),
                    ),
                )
            })
            .collect()
    }

    fn assert_bits_identical(
        before: &HashMap<String, (Vec<u32>, Vec<u32>)>,
        after: &HashMap<String, BoundedTensor>,
        label: &str,
    ) {
        assert_eq!(before.len(), after.len(), "{label}: key count changed");
        for (k, (lo_bits, up_bits)) in before {
            let bt = after
                .get(k)
                .unwrap_or_else(|| panic!("{label}: key '{k}' vanished"));
            let lo_now: Vec<u32> = bt.lower().iter().map(|f| f.to_bits()).collect();
            let up_now: Vec<u32> = bt.upper().iter().map(|f| f.to_bits()).collect();
            assert_eq!(lo_bits, &lo_now, "{label}: '{k}' lower bits changed");
            assert_eq!(up_bits, &up_now, "{label}: '{k}' upper bits changed");
        }
    }

    /// Elementwise containment: `inner ⊆ outer` for every shared key
    /// (oracle 5: shrink-only monotonicity).
    fn assert_contained(
        outer: &HashMap<String, BoundedTensor>,
        inner: &HashMap<String, BoundedTensor>,
        label: &str,
    ) {
        for (k, pre) in outer {
            let post = inner
                .get(k)
                .unwrap_or_else(|| panic!("{label}: key '{k}' vanished"));
            assert_eq!(pre.shape(), post.shape(), "{label}: '{k}' shape changed");
            for ((&pl, &pu), (&nl, &nu)) in pre
                .lower()
                .iter()
                .zip(pre.upper().iter())
                .zip(post.lower().iter().zip(post.upper().iter()))
            {
                assert!(
                    nl >= pl && nu <= pu,
                    "{label}: '{k}' widened: pre=[{pl},{pu}] post=[{nl},{nu}]"
                );
            }
        }
    }

    /// Oracle 4: every stored [l,u] must enclose the true per-node activation
    /// at sampled points of the root box.
    fn assert_enclosure_under_sampling(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        bounds: &HashMap<String, BoundedTensor>,
    ) {
        let lo = input.lower()[[0]];
        let hi = input.upper()[[0]];
        for step in 0..=20 {
            let x = lo + (hi - lo) * (step as f32 / 20.0);
            let point = BoundedTensor::concrete(arr1(&[x]).into_dyn()).unwrap();
            let acts = graph
                .collect_node_activations_pointwise(&point, None)
                .expect("pointwise forward should succeed");
            for (node, bt) in bounds {
                let Some(act) = acts.get(node) else { continue };
                for ((&l, &u), &v) in bt
                    .lower()
                    .iter()
                    .zip(bt.upper().iter())
                    .zip(act.center().iter())
                {
                    assert!(
                        v >= l - 1e-4 && v <= u + 1e-4,
                        "sample x={x}: node '{node}' activation {v} outside stored [{l},{u}]"
                    );
                }
            }
        }
    }

    #[ntest::timeout(60000)]
    #[test]
    fn test_stabilize_fixes_provably_stable_neuron_and_removes_branching_candidacy() {
        let (graph, input) = build_convertible_graph();
        let mut bounds = graph.collect_node_bounds(&input).unwrap();
        let pre_loop = bounds.clone();

        // Precondition: IBP sees lin2 straddling 0 and relu2/0 is a branching
        // candidate.
        let lin2 = bounds.get("lin2").unwrap();
        assert!(
            lin2.lower()[[0]] < 0.0 && lin2.upper()[[0]] > 0.0,
            "fixture: lin2 must straddle 0 under IBP, got [{}, {}]",
            lin2.lower()[[0]],
            lin2.upper()[[0]]
        );
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let relu_nodes = vec!["relu1".to_string(), "relu2".to_string()];
        let domain_before = MultiObjectiveGraphBabDomain::root(
            bounds.clone(),
            vec![(-1.0, 1.0)],
            &input,
            &[0.0],
            false,
        )
        .unwrap();
        let unstable_before =
            verifier.find_unstable_graph_neurons_multi(&graph, &domain_before, &relu_nodes);
        assert!(
            unstable_before.contains(&("relu2".to_string(), 0)),
            "fixture: relu2/0 must start as a branching candidate"
        );

        let alpha = root_alpha_for(&graph, &input);
        let report = stabilize_and_fix(
            &graph,
            &input,
            &[vec![1.0]],
            None,
            None,
            Some(&alpha),
            &mut bounds,
            &test_opts(),
        );

        // The provably-stable neuron is FIXED (always-active).
        assert!(
            report
                .fixed
                .iter()
                .any(|(n, i, s)| n == "relu2" && *i == 0 && *s == StableSide::Active),
            "relu2/0 should be fixed always-active; report: {:?}",
            report.fixed
        );
        // Property (oracle 2b): every reported fix is backed by the STORED
        // post-merge pair — no fix without l>=0 || u<=0 in storage.
        for (relu_name, idx, side) in &report.fixed {
            let pre_name = graph.nodes.get(relu_name).unwrap().inputs[0].clone();
            let bt = bounds.get(&pre_name).unwrap();
            let l: Vec<f32> = bt.lower().iter().copied().collect();
            let u: Vec<f32> = bt.upper().iter().copied().collect();
            match side {
                StableSide::Active => assert!(
                    l[*idx] >= 0.0,
                    "fix {relu_name}/{idx} Active but stored l={} < 0",
                    l[*idx]
                ),
                StableSide::Inactive => assert!(
                    u[*idx] <= 0.0,
                    "fix {relu_name}/{idx} Inactive but stored u={} > 0",
                    u[*idx]
                ),
            }
        }
        // Oracle 5: shrink-only everywhere (including the downstream resweep).
        assert_contained(&pre_loop, &bounds, "post-stabilize");
        // Oracle 4: containment under sampling (catches wrong-key writes).
        assert_enclosure_under_sampling(&graph, &input, &bounds);
        // Downstream recompute propagated the fix: lin3 inherits the tightened
        // relu2 output (true value 1.9), so its stored lower must have moved
        // well above the pre-loop 0.0.
        assert!(
            bounds.get("lin3").unwrap().lower()[[0]] > 1.0,
            "downstream resweep should propagate the fix into lin3, got lower={}",
            bounds.get("lin3").unwrap().lower()[[0]]
        );
        // Oracle 7: the fixed neuron drops out of branching candidacy purely
        // via the stored bound (no side channel).
        let domain_after =
            MultiObjectiveGraphBabDomain::root(bounds, vec![(-1.0, 1.0)], &input, &[0.0], false)
                .unwrap();
        let unstable_after =
            verifier.find_unstable_graph_neurons_multi(&graph, &domain_after, &relu_nodes);
        assert!(
            !unstable_after.contains(&("relu2".to_string(), 0)),
            "fixed neuron relu2/0 must no longer be a branching candidate"
        );
    }

    #[ntest::timeout(60000)]
    #[test]
    fn test_stabilize_never_fixes_genuinely_unstable_neuron() {
        let (graph, input) = build_straddling_graph();
        let mut bounds = graph.collect_node_bounds(&input).unwrap();
        let pre_loop = bounds.clone();
        let alpha = root_alpha_for(&graph, &input);

        // kappa = 1.0 admits every unstable neuron as a tighten candidate, so
        // the loop genuinely ATTEMPTS the conversion and must fail it.
        let mut opts = test_opts();
        opts.kappa = 1.0;
        let report = stabilize_and_fix(
            &graph,
            &input,
            &[vec![1.0]],
            None,
            None,
            Some(&alpha),
            &mut bounds,
            &opts,
        );

        // The true range of lin2 is [-1,1]: a sound bound can NEVER stop
        // straddling 0, so no fix may be reported and the stored bound must
        // still straddle.
        assert!(
            report.fixed.is_empty(),
            "genuinely unstable neurons must never be fixed; report: {:?}",
            report.fixed
        );
        let lin2 = bounds.get("lin2").unwrap();
        assert!(
            lin2.lower()[[0]] < 0.0 && lin2.upper()[[0]] > 0.0,
            "lin2 stored bound must still straddle 0, got [{}, {}]",
            lin2.lower()[[0]],
            lin2.upper()[[0]]
        );
        // Still a branching candidate.
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let relu_nodes = vec!["relu1".to_string(), "relu2".to_string()];
        let domain = MultiObjectiveGraphBabDomain::root(
            bounds.clone(),
            vec![(-1.0, 1.0)],
            &input,
            &[0.0],
            false,
        )
        .unwrap();
        let unstable = verifier.find_unstable_graph_neurons_multi(&graph, &domain, &relu_nodes);
        assert!(
            unstable.contains(&("relu2".to_string(), 0)),
            "not-proven neuron must keep its branching candidacy"
        );
        // Sound partial work only: shrink-only + containment under sampling.
        assert_contained(&pre_loop, &bounds, "straddling post-stabilize");
        assert_enclosure_under_sampling(&graph, &input, &bounds);
    }

    #[ntest::timeout(60000)]
    #[test]
    fn test_gate_off_is_byte_identical_and_parse_rejects_invalid() {
        // Oracle 3: with NY_STABILIZE unset the env entry point must return
        // None and leave the map bit-identical. (Serialized + restored via
        // the blessed env choke point.)
        let _env_lock = ny_test_utils::env::lock_env();
        let _unset = ny_test_utils::env::ScopedEnvVar::unset("NY_STABILIZE");
        let (graph, input) = build_convertible_graph();
        let mut bounds = graph.collect_node_bounds(&input).unwrap();
        let bits = clone_bits(&bounds);
        let alpha = root_alpha_for(&graph, &input);
        let out = stabilize_and_fix_from_env(
            &graph,
            &input,
            &[vec![1.0]],
            None,
            None,
            Some(&alpha),
            &mut bounds,
        );
        assert!(out.is_none(), "gate off must return None");
        assert_bits_identical(&bits, &bounds, "gate-off");

        // Gate parsing: absent/0/negative/unparseable ⇒ OFF.
        assert!(StabilizeOptions::parse(None).is_none());
        assert!(StabilizeOptions::parse(Some("0")).is_none());
        assert!(StabilizeOptions::parse(Some("-3")).is_none());
        assert!(StabilizeOptions::parse(Some("abc")).is_none());
        assert!(StabilizeOptions::parse(Some("nan")).is_none());
        let opts = StabilizeOptions::parse(Some("2.5")).expect("valid budget parses");
        assert_eq!(opts.budget, Duration::from_secs_f64(2.5));
    }

    #[ntest::timeout(60000)]
    #[test]
    fn test_pre_expired_deadline_leaves_map_untouched_and_tiny_budget_stays_sound() {
        let (graph, input) = build_convertible_graph();
        let alpha = root_alpha_for(&graph, &input);

        // Oracle 6a: pre-expired GLOBAL deadline ⇒ untouched, zero rounds.
        let mut bounds = graph.collect_node_bounds(&input).unwrap();
        let bits = clone_bits(&bounds);
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(5))
            .unwrap();
        let report = stabilize_and_fix(
            &graph,
            &input,
            &[vec![1.0]],
            None,
            Some(expired),
            Some(&alpha),
            &mut bounds,
            &test_opts(),
        );
        assert_eq!(report.rounds, 0, "pre-expired deadline must run no rounds");
        assert!(report.fixed.is_empty());
        assert_bits_identical(&bits, &bounds, "pre-expired");

        // Oracle 6b: a ~zero budget exits early with PARTIAL work; whatever
        // completed must still satisfy shrink-only + sampling containment.
        let mut bounds = graph.collect_node_bounds(&input).unwrap();
        let pre_loop = bounds.clone();
        let mut opts = test_opts();
        opts.budget = Duration::from_nanos(1);
        let _ = stabilize_and_fix(
            &graph,
            &input,
            &[vec![1.0]],
            None,
            None,
            Some(&alpha),
            &mut bounds,
            &opts,
        );
        assert_contained(&pre_loop, &bounds, "tiny-budget");
        assert_enclosure_under_sampling(&graph, &input, &bounds);
    }

    #[ntest::timeout(60000)]
    #[test]
    fn test_missing_alpha_state_is_a_no_op() {
        let (graph, input) = build_convertible_graph();
        let mut bounds = graph.collect_node_bounds(&input).unwrap();
        let bits = clone_bits(&bounds);
        let report = stabilize_and_fix(
            &graph,
            &input,
            &[vec![1.0]],
            None,
            None,
            None,
            &mut bounds,
            &test_opts(),
        );
        assert_eq!(report.rounds, 0);
        assert!(report.fixed.is_empty());
        assert_bits_identical(&bits, &bounds, "no-alpha");
    }

    #[ntest::timeout(60000)]
    #[test]
    fn test_crown_recompute_variant_stays_sound() {
        let (graph, input) = build_convertible_graph();
        let mut bounds = graph.collect_node_bounds(&input).unwrap();
        let pre_loop = bounds.clone();
        let alpha = root_alpha_for(&graph, &input);
        let mut opts = test_opts();
        opts.recompute_crown = true;
        let report = stabilize_and_fix(
            &graph,
            &input,
            &[vec![1.0]],
            None,
            None,
            Some(&alpha),
            &mut bounds,
            &opts,
        );
        assert!(
            report.fixed.iter().any(|(n, i, _)| n == "relu2" && *i == 0),
            "crown-recompute arm should still fix relu2/0"
        );
        assert_contained(&pre_loop, &bounds, "crown-recompute");
        assert_enclosure_under_sampling(&graph, &input, &bounds);
    }

    /// Oracle 1 (verdict preservation, unit scale): the full multi-objective
    /// verifier with the gate ON must not contradict the gate-OFF verdict on
    /// the same instance. Env-based because the gate is read at the root pass.
    #[ntest::timeout(120000)]
    #[test]
    fn test_verifier_ab_gate_on_off_no_verdict_contradiction() {
        use crate::beta_crown::result::BabVerificationStatus;

        let (graph, input) = build_convertible_graph();
        let mut config = BetaCrownConfig {
            use_alpha_crown: true,
            timeout: Duration::from_secs(20),
            ..Default::default()
        };
        config.alpha_config.iterations = 1;
        config.alpha_config.adaptive_skip = false;
        config.alpha_config.adaptive_skip_pilot = false;
        let verifier = BetaCrownVerifier::new(config);
        let objectives = vec![vec![1.0_f32]];
        let thresholds = vec![0.0_f32];

        // Serialized env scope (clippy env wall): OFF arm then ON arm.
        let (off, on) = ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_STABILIZE");
            let off = verifier
                .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
                .expect("gate-off verify should succeed");

            env.set("NY_STABILIZE", "5");
            let on = verifier
                .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
                .expect("gate-on verify should succeed");
            (off, on)
        });

        // The property (output = 1.9 > 0) is true; both arms must verify it —
        // and in no case may one arm report Verified while the other reports a
        // counterexample.
        assert!(
            matches!(off.result, BabVerificationStatus::Verified),
            "gate-off arm should verify the true property, got {:?}",
            off.result
        );
        assert!(
            matches!(on.result, BabVerificationStatus::Verified),
            "gate-on arm should verify the true property, got {:?}",
            on.result
        );
    }
}
