// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Graph-MIP LEAF oracle (increment 6, `docs/GRAPH_MIP_LEAF_SOLVER.md`): the
// ny-cli implementation of `ny_propagate::beta_crown::graph_mip_leaf::
// GraphMipLeafOracle`, attached to the verifier by default
// (`NY_GRAPH_MIP_LEAF=0` disables it, independently of the whole-net
// `NY_GRAPH_MIP` escalation).
//
// Per undecided BaB child: (1) eligibility gates (depth, free-binary budget,
// time budget), (2) PIN the split premises — clamp each premise's
// pre-activation box (emit_hard_six `clamp()` parity) and, in the default
// `fix` mode, additionally fix the premise's ReLU indicator column (the lpopt
// LP-A "piece-fix" form: `z=1 ⇒ y=x ∧ x≥0`, `z=0 ⇒ y=0 ∧ x≤0` — the premise
// enforced EXACTLY, no Δ-hole), (3) one decision MIP per undecided spec row
// (`objective · y <= threshold`; infeasible ⇔ row verified on the subdomain),
// (4) certified-UNSAT-only admission + graph-forward witness revalidation.
//
// SOUNDNESS: the MIP feasible set contains the subdomain's reachable set
// (exact affine rows; the domain's own sound boxes ±DELTA as big-M ranges;
// the domain's exact input box; premise pieces implied by the premises that
// DEFINE the subdomain). `VerifiedAllRows` requires a VERIFIED Farkas
// certificate on every row; every other outcome degrades to `Undecided`
// (the child is requeued unchanged) — strictly additive.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ny_core::Bound;
use ny_mip::{obbt_relaxation_bounds, MipBackend, MipConfig, MipResult, MipSolver};
use ny_propagate::beta_crown::graph_mip_leaf::{
    GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict,
};
use ny_propagate::{AlphaCrownConfig, GraphNetwork, Layer};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::{debug, info, warn};

use super::graph_mip::GraphMipEncoding;
use super::graph_mip_diff_coupling::diff_coupling_enabled;
use super::mip_highs::clamp_witness_to_box;
use super::mip_preprocess::bounded_tensor_to_bounds;

/// Master gate for the leaf oracle (independent of `NY_GRAPH_MIP`).
///
/// DEFAULT-ON (2026-07-21): the leaf oracle escalates stuck deep BaB leaves
/// (MIN_DEPTH>=4) to an exact ay-milp solve, admitting a child `verified` only on
/// a certified-UNSAT Farkas skeleton (0-wrong moat) and requeueing it to normal
/// BaB otherwise, all under a time slice (TOTAL_FRAC 0.5 / SLICE_S 10s). Sound by
/// construction; bounded downside. Armed now that ay-milp is fast enough to close
/// depth-5..8 leaves in ~1-10s. `NY_GRAPH_MIP_LEAF=0` restores the old off path.
fn graph_mip_leaf_enabled_from_value(value: Option<&str>) -> bool {
    value != Some("0")
}

pub(super) fn graph_mip_leaf_enabled() -> bool {
    graph_mip_leaf_enabled_from_value(std::env::var("NY_GRAPH_MIP_LEAF").ok().as_deref())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

/// Pin mode: `fix` (default — clamp + fix the premise binaries, the LP-A
/// piece-fix form) or `clamp` (emit_hard_six byte-parity; premises enforced
/// only through the Δ-weakened box clamp).
fn pin_mode_is_fix() -> bool {
    !matches!(
        std::env::var("NY_GRAPH_MIP_LEAF_PIN").ok().as_deref(),
        Some("clamp")
    )
}

/// LEAF-scale binary budget (defect 1): the leaf lane exists for the w2/w5
/// rungs of the hard-six ladder (~16-96 free binaries), NOT the full rung —
/// its cap is therefore separate from (and far below) the whole-net
/// escalation's `NY_GRAPH_MIP_MAX_BINARIES`. Default 96.
fn leaf_max_binaries() -> usize {
    env_usize("NY_GRAPH_MIP_LEAF_MAX_BINARIES", 96)
}

/// Certified phase-enumeration cap (#relational-bab): a row whose encoding
/// carries at most this many FREE binaries is decided by enumerating the
/// `2^k` phase-fixed LP leaves, each requiring the verified-Farkas
/// `Unsat {{ certified: true }}` admission. SUPERSEDED as the default lane by
/// ay's P2 `tree_cert` (the plain solve now carries a VERIFIED case-split
/// certificate for MIP infeasibility at ANY k), so the default is 0
/// (disabled — the full slice goes to ay); `NY_GRAPH_MIP_LEAF_ENUM_CAP=k`
/// restores the enumeration as an independent fallback lane.
fn leaf_certified_enum_cap() -> usize {
    env_usize("NY_GRAPH_MIP_LEAF_ENUM_CAP", 0)
}

/// LEAF-scale nnz budget (defect 1): a pre-encode estimate of the encoded
/// matrix's nonzeros gates the leaf BEFORE the (memory-heavy) encode + exact
/// rational conversion happen — the measured cifar100 full-net leaf encode is
/// ~44M nnz × ay's BigRational model, the 24GB-box memory bomb. Default 5M
/// (well below the whole-net caps).
fn leaf_max_nnz() -> usize {
    env_usize("NY_GRAPH_MIP_LEAF_MAX_NNZ", 5_000_000)
}

/// Cheap OVER-estimate of the encoded problem's nonzeros, from layer shapes +
/// the flattened bounds (no allocation): Linear `out×in`, Conv2d
/// `out_len × (in_c/groups × kh × kw + 1)` (the im2col row density),
/// BatchNorm `2×len`, Add `3×len`, ReLU `~5×unstable`. `None` fails closed
/// (unknown layer / missing box → the encoder would bail anyway).
pub(super) fn estimate_encode_nnz(
    graph: &GraphNetwork,
    flat_bounds: &HashMap<String, Vec<Bound>>,
) -> Option<usize> {
    let exec = graph.exec_order().ok()?;
    let mut nnz: usize = 0;
    for name in exec {
        let node = graph.node(name)?;
        let out_len = |n: &str| flat_bounds.get(n).map(Vec::len);
        match node.layer() {
            Layer::Linear(lin) => {
                let (out_dim, in_dim) = lin.weight.dim();
                nnz = nnz.saturating_add(out_dim.saturating_mul(in_dim + 1));
            }
            Layer::Conv2d(conv) => {
                let k = conv.kernel.shape();
                let row = k
                    .get(1)
                    .copied()
                    .unwrap_or(1)
                    .saturating_mul(k.get(2).copied().unwrap_or(1))
                    .saturating_mul(k.get(3).copied().unwrap_or(1))
                    + 1;
                let out = out_len(name)?;
                nnz = nnz.saturating_add(out.saturating_mul(row));
            }
            Layer::BatchNorm(_) => {
                nnz = nnz.saturating_add(out_len(name)?.saturating_mul(2));
            }
            Layer::Add(_) => {
                nnz = nnz.saturating_add(out_len(name)?.saturating_mul(3));
            }
            Layer::ReLU(_) => {
                // INPUT-node entry first (the true pre-activation; the relu's
                // own entry is the post box in live maps) — mirrors the
                // encoder's lookup order (#relational-bab soundness fix).
                let pre = node
                    .inputs()
                    .first()
                    .and_then(|i| flat_bounds.get(i))
                    .or_else(|| flat_bounds.get(name))?;
                let unstable = pre
                    .iter()
                    .filter(|b| b.lower() < 0.0 && b.upper() > 0.0)
                    .count();
                nnz = nnz.saturating_add(unstable.saturating_mul(5));
            }
            Layer::Flatten(_) | Layer::Reshape(_) => {}
            // #relational-bab: the encoder's remaining exact ops (the mscn
            // increment + the relational difference nets — Sub of two towers,
            // AddConstant/SubConstant chains). Without these the estimator
            // returned None and the leaf lane DECLINED every relational edge
            // consult despite the encoder supporting the graph.
            Layer::Sub(_) => {
                nnz = nnz.saturating_add(out_len(name)?.saturating_mul(3));
            }
            Layer::AddConstant(_)
            | Layer::SubConstant(_)
            | Layer::MulConstant(_)
            | Layer::DivConstant(_) => {
                nnz = nnz.saturating_add(out_len(name)?.saturating_mul(2));
            }
            // Pure index plumbing: aliases, no rows.
            Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
            | Layer::Slice(_)
            | Layer::Gather(_)
            | Layer::Concat(_) => {}
            Layer::ReduceSum(_) => {
                let in_len = node
                    .inputs()
                    .first()
                    .and_then(|i| flat_bounds.get(i.as_str()))
                    .map(Vec::len)?;
                nnz = nnz.saturating_add(in_len.saturating_add(out_len(name)?));
            }
            _ => return None,
        }
    }
    Some(nnz)
}

/// Rate-limited info logging for eligibility declines (defect 4): the first
/// few declines per run log at INFO (the visibility the live measurement
/// lacked), the rest at DEBUG (a deep BaB consults thousands of leaves).
fn decline_info_budget() -> bool {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DECLINES_LOGGED: AtomicUsize = AtomicUsize::new(0);
    DECLINES_LOGGED.fetch_add(1, Ordering::Relaxed) < 5
}

/// Cumulative leaf-solve budget state (see the design doc §3).
#[derive(Default)]
struct LeafBudgetState {
    /// Seconds remaining to the deadline at the FIRST oracle attempt.
    first_remaining: Option<f64>,
    /// Cumulative seconds spent inside leaf solves.
    spent: f64,
}

/// The leaf oracle: eligibility gates → pinned encoding → per-row decision
/// solves → certified admission. All internal failures map to `Undecided`.
pub(super) struct GraphMipLeafSolver {
    backend: MipBackend,
    budget: Mutex<LeafBudgetState>,
    /// Latched after a graph-forward-CONFIRMED SAT witness (defect 3): a real
    /// margin≤threshold point means certified-UNSAT is unreachable in this
    /// region of the tree — further leaf solves would only burn budget re-
    /// discovering it, so the oracle disables itself for the rest of the run
    /// (the witness is logged loudly; sat-side reporting belongs to the
    /// PGD/ORT lanes until the BaB lanes grow witness plumbing).
    sat_latch: std::sync::atomic::AtomicBool,
}

impl GraphMipLeafSolver {
    pub(super) fn new(backend: MipBackend) -> Self {
        Self {
            backend,
            budget: Mutex::new(LeafBudgetState::default()),
            sat_latch: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Per-leaf wall slice (design doc §3): declines (`None`) when the
    /// cumulative cap is exhausted or the remaining slice is uselessly small.
    fn admit_slice(&self, deadline: Option<Instant>) -> Option<f64> {
        let remaining = match deadline {
            Some(d) => d.saturating_duration_since(Instant::now()).as_secs_f64(),
            None => f64::INFINITY,
        };
        let mut budget = self.budget.lock().ok()?;
        let first = *budget.first_remaining.get_or_insert(remaining);
        let frac = env_f64("NY_GRAPH_MIP_LEAF_TOTAL_FRAC", 0.5).clamp(0.0, 1.0);
        if first.is_finite() && budget.spent >= first * frac {
            debug!(
                spent_s = budget.spent,
                cap_s = first * frac,
                "Graph-MIP leaf: cumulative budget exhausted; declining"
            );
            return None;
        }
        let slice = env_f64("NY_GRAPH_MIP_LEAF_SLICE_S", 10.0).min(remaining / 4.0);
        if slice < 1.0 {
            return None;
        }
        Some(slice)
    }

    fn record_spent(&self, secs: f64) {
        if let Ok(mut budget) = self.budget.lock() {
            budget.spent += secs;
        }
    }
}

/// Flatten the domain's node bounds and CLAMP the split premises into them —
/// the emit_hard_six `clamp()` step: an active premise raises the ReLU's
/// pre-activation lower bound to 0, an inactive one lowers its upper bound to
/// 0 (both SOUND: every point of the subdomain satisfies the premise, so the
/// clamped box still encloses the subdomain's reachable pre-activations).
/// Returns `None` when a premise's ReLU/pre-node/neuron cannot be resolved
/// (fail-closed: the caller degrades to `Undecided`).
fn clamped_flat_bounds(
    graph: &GraphNetwork,
    node_bounds: &HashMap<String, Arc<BoundedTensor>>,
    splits: &[ny_propagate::beta_crown::graph_mip_leaf::LeafSplit],
) -> Option<HashMap<String, Vec<Bound>>> {
    let mut flat: HashMap<String, Vec<Bound>> = HashMap::with_capacity(node_bounds.len());
    for (name, bt) in node_bounds {
        let bounds = bounded_tensor_to_bounds(bt).ok()?;
        flat.insert(name.clone(), bounds);
    }
    for split in splits {
        let relu = graph.node(&split.relu_node)?;
        if !matches!(relu.layer(), Layer::ReLU(_)) {
            return None;
        }
        let pre_name = relu.inputs().first()?;
        let pre = flat.get_mut(pre_name)?;
        let b = pre.get_mut(split.neuron_idx)?;
        let (lo, hi) = if split.is_active {
            (b.lower().max(0.0), b.upper())
        } else {
            (b.lower(), b.upper().min(0.0))
        };
        if lo > hi {
            // Premise ∧ box infeasible ⇒ the subdomain is empty. The BaB's own
            // feasibility screen normally prevents this; degrade rather than
            // claim (fail-closed). Checked BEFORE constructing the Bound —
            // `new_allow_infinite` panics on inverted bounds.
            return None;
        }
        *b = Bound::new_allow_infinite(lo, hi);
    }
    Some(flat)
}

/// Free-binary estimate on the CLAMPED (un-inflated) bounds: unstable ReLU
/// pre-activation entries (`l < 0 < u`). Premise-clamped neurons sit at
/// exactly 0 on one side, so they do NOT count — this is the count of
/// binaries the solver must actually branch on. Fail-closed `None` when any
/// ReLU lacks a box.
fn free_binary_count(
    graph: &GraphNetwork,
    flat_bounds: &HashMap<String, Vec<Bound>>,
) -> Option<usize> {
    let exec = graph.exec_order().ok()?;
    let mut count = 0usize;
    for name in exec {
        let node = graph.node(name)?;
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        // INPUT-node entry first — see the encoder's lookup-order note.
        let pre = node
            .inputs()
            .first()
            .and_then(|i| flat_bounds.get(i))
            .or_else(|| flat_bounds.get(name))?;
        count += pre
            .iter()
            .filter(|b| b.lower() < 0.0 && b.upper() > 0.0)
            .count();
    }
    Some(count)
}

/// Fix the split premises' indicator columns in the encoding (the LP-A
/// piece-fix form). A premise whose neuron produced no binary (already stable
/// in the clamped-then-inflated box) is skipped — its piece is enforced by the
/// stable encoding itself. Returns the number of binaries pinned.
fn fix_split_binaries(
    graph: &GraphNetwork,
    enc: &mut GraphMipEncoding,
    splits: &[ny_propagate::beta_crown::graph_mip_leaf::LeafSplit],
) -> usize {
    let mut pinned = 0usize;
    for split in splits {
        let Some(pos) = enc
            .binary_keys
            .iter()
            .position(|(node, idx)| node == &split.relu_node && *idx == split.neuron_idx)
        else {
            continue; // stable in the encoding: the piece is already forced
        };
        let col = enc.binary_vars[pos];
        enc.problem
            .fix_col(col, if split.is_active { 1.0 } else { 0.0 });
        pinned += 1;
    }
    let _ = graph;
    pinned
}

/// Certified MIP infeasibility by PHASE ENUMERATION: fix the `k` free
/// binaries to every assignment in `{0,1}^k` and prove EACH resulting pure LP
/// infeasible through the exact, self-verified Farkas lane. The enumerated
/// leaves exactly cover the MIP's feasible set (binaries range over `{0,1}`),
/// so all-leaves-certified-infeasible ⇒ the MIP is infeasible, with a
/// certificate chain at the SAME trust level as the admitted
/// `Unsat {{ certified: true }}` path (every leaf cert verified against its
/// model inside `prove_infeasible_with_row_farkas`).
///
/// Returns `Some(true)` = certified infeasible; `Some(false)` = some leaf not
/// proven infeasible (feasible, uncertified, or per-leaf failure — the caller
/// falls through to the plain MIP solve, e.g. for the witness path); `None` =
/// deadline expired mid-enumeration (fall through likewise).
fn certified_phase_enumeration(
    enc: &GraphMipEncoding,
    backend: MipBackend,
    deadline: Instant,
) -> Option<bool> {
    let k = enc.binary_vars.len();
    if k > 63 {
        return Some(false); // 2^k would overflow; the plain solve takes over
    }
    let leaves: u64 = 1u64 << k;
    for assignment in 0..leaves {
        if Instant::now() >= deadline {
            return None;
        }
        let mut leaf = enc.problem.clone();
        for (bit, &col) in enc.binary_vars.iter().enumerate() {
            leaf.fix_col(col, ((assignment >> bit) & 1) as f64);
        }
        // Pure LP: every binary fixed to a constant, integrality dropped.
        // ay emits (and `check_feasibility` VERIFIES) the exact Farkas
        // certificate for relaxation-level infeasibility — the same
        // `Unsat { certified: true }` admission seam as the plain path.
        leaf.relax_integrality();
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        let num_cols = leaf.num_cols();
        let solver = MipSolver::new(
            ny_mip::MipParts {
                problem: leaf,
                input_vars: enc.input_vars.clone(),
                output_vars: enc.output_vars.clone(),
                binary_vars: Vec::new(),
                binary_widths: Vec::new(),
                num_cols,
            },
            MipConfig {
                backend,
                timeout_secs: remaining.max(0.1),
                parallel_split: 1,
                ..Default::default()
            },
        );
        let leaf_result = solver.check_feasibility();
        if std::env::var("NY_LEAF_ENUM_TRACE").is_ok() {
            eprintln!("[enum-trace] leaf {assignment}/{leaves}: {leaf_result:?}");
        }
        match leaf_result {
            Ok(MipResult::Unsat { certified: true }) => {} // leaf certified
            Ok(_) | Err(_) => return Some(false),
        }
    }
    Some(true)
}

/// ORACLE BOUNDARY CONTAINMENT (defect 3): the trait contract is infallible —
/// a panic anywhere inside the leaf solve (encoder, ny-mip, ay, a solver
/// worker's join) must degrade THIS LEAF to `Undecided` and let BaB continue,
/// never end the run. `catch_unwind` is the audit-proof backstop for the
/// error paths already mapped inside the solve.
fn contain_leaf_panics(f: impl FnOnce() -> GraphMipLeafVerdict) -> GraphMipLeafVerdict {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(verdict) => verdict,
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            warn!(
                "Graph-MIP leaf: PANIC contained at the oracle boundary ({msg}); leaf degrades to Undecided, BaB continues"
            );
            GraphMipLeafVerdict::Undecided
        }
    }
}

impl GraphMipLeafOracle for GraphMipLeafSolver {
    fn solve_leaf(&self, req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
        contain_leaf_panics(|| self.solve_leaf_gated(req))
    }
}

impl GraphMipLeafSolver {
    fn solve_leaf_gated(&self, req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
        // --- eligibility gates (cheap, before any encoding) ---
        if self.sat_latch.load(std::sync::atomic::Ordering::Relaxed) {
            debug!("Graph-MIP leaf: disabled by earlier confirmed-SAT latch");
            return GraphMipLeafVerdict::Undecided;
        }
        let min_depth = env_usize("NY_GRAPH_MIP_LEAF_MIN_DEPTH", 4);
        if req.depth < min_depth || req.rows.is_empty() {
            return GraphMipLeafVerdict::Undecided;
        }
        let Some(flat_bounds) = clamped_flat_bounds(req.graph, req.node_bounds, &req.splits) else {
            debug!("Graph-MIP leaf: premise clamp unresolved/infeasible; declined");
            return GraphMipLeafVerdict::Undecided;
        };
        // LEAF-scale binary cap (defect 1): the leaf lane's OWN budget — far
        // below the whole-net cap; a depth-5 cifar100 child (~490 free) is the
        // FULL rung and must decline here without encoding.
        let max_binaries = leaf_max_binaries();
        let free = match free_binary_count(req.graph, &flat_bounds) {
            Some(free) if free <= max_binaries => free,
            Some(free) => {
                if decline_info_budget() {
                    info!(
                        "Graph-MIP leaf: declined (free_binaries={free} > leaf  budget {max_binaries}, depth={})",
                        req.depth
                    );
                } else {
                    debug!(
                        "Graph-MIP leaf: declined (free_binaries={free} > leaf  budget {max_binaries})"
                    );
                }
                return GraphMipLeafVerdict::Undecided;
            }
            None => {
                debug!("Graph-MIP leaf: declined (unsupported layer or missing box)");
                return GraphMipLeafVerdict::Undecided;
            }
        };
        // LEAF-scale nnz cap (defect 1): estimated BEFORE the encode so the
        // encode + exact-rational conversion (the measured memory bomb at
        // ~44M nnz full-net) never start on an over-scale leaf.
        let max_nnz = leaf_max_nnz();
        match estimate_encode_nnz(req.graph, &flat_bounds) {
            Some(nnz) if nnz <= max_nnz => {}
            Some(nnz) => {
                if decline_info_budget() {
                    info!(
                        "Graph-MIP leaf: declined (estimated nnz {nnz} > leaf cap {max_nnz};                          full-rung encode — the leaf lane targets the w2/w5 scale)"
                    );
                } else {
                    debug!("Graph-MIP leaf: declined (estimated nnz {nnz} > cap {max_nnz})");
                }
                return GraphMipLeafVerdict::Undecided;
            }
            None => {
                debug!("Graph-MIP leaf: declined (nnz estimate unavailable)");
                return GraphMipLeafVerdict::Undecided;
            }
        }
        let Some(slice) = self.admit_slice(req.deadline) else {
            debug!("Graph-MIP leaf: declined (budget exhausted or slice too small)");
            return GraphMipLeafVerdict::Undecided;
        };
        // CONSULT visibility (defect 4).
        info!(
            depth = req.depth,
            free_binaries = free,
            rows = req.rows.len(),
            slice_s = slice,
            "Graph-MIP leaf: consulting the exact solver on an eligible subdomain"
        );
        let start = Instant::now();
        let verdict = self.solve_leaf_inner(req, &flat_bounds, slice);
        self.record_spent(start.elapsed().as_secs_f64());
        // VERDICT visibility (defect 4).
        match &verdict {
            GraphMipLeafVerdict::VerifiedAllRows => info!(
                depth = req.depth,
                wall_s = start.elapsed().as_secs_f64(),
                "Graph-MIP leaf verdict: VERIFIED (all rows certified-UNSAT)"
            ),
            GraphMipLeafVerdict::Violated { .. } => info!(
                depth = req.depth,
                wall_s = start.elapsed().as_secs_f64(),
                "Graph-MIP leaf verdict: confirmed SAT witness (latching oracle off)"
            ),
            GraphMipLeafVerdict::Undecided => info!(
                depth = req.depth,
                wall_s = start.elapsed().as_secs_f64(),
                "Graph-MIP leaf verdict: undecided (leaf stays in BaB)"
            ),
        }
        verdict
    }
}

impl GraphMipLeafSolver {
    fn solve_leaf_inner(
        &self,
        req: &GraphMipLeafRequest<'_>,
        flat_bounds: &HashMap<String, Vec<Bound>>,
        slice: f64,
    ) -> GraphMipLeafVerdict {
        // --- encode ONCE (exact rows + DELTA inflation), pin the premises ---
        let Ok(input_bounds) = bounded_tensor_to_bounds(req.input_bounds) else {
            return GraphMipLeafVerdict::Undecided;
        };
        let encode_deadline = Instant::now() + std::time::Duration::from_secs_f64(slice.max(1.0));
        // DECLARED SHAPES for the encoder's exact index/broadcast math: the
        // ONNX loader never populates `declared_shapes`, and stitched
        // in-memory graphs (the relational difference nets) have none either
        // — so the const-op/plumbing arms fail closed on every real graph
        // ("no declared shape for '_input'", the measured 1µs live decline).
        // The request's per-node bounds carry the TRUE forward shapes; stamp
        // them (and the input box's shape) onto a shadow clone, never
        // overriding a shape the graph already declares. Exact — no guessing.
        let mut shaped_graph = req.graph.clone();
        if shaped_graph
            .declared_shape(ny_propagate::NETWORK_INPUT)
            .is_none()
        {
            shaped_graph.set_declared_shape(
                ny_propagate::NETWORK_INPUT,
                req.input_bounds.shape().to_vec(),
            );
        }
        for (name, bt) in req.node_bounds {
            if shaped_graph.declared_shape(name).is_none() {
                shaped_graph.set_declared_shape(name.clone(), bt.shape().to_vec());
            }
        }
        let graph = &shaped_graph;
        let mut base = match super::graph_mip::encode_graph_with_deadline(
            graph,
            &input_bounds,
            flat_bounds,
            Some(encode_deadline),
        ) {
            Ok(enc) => enc,
            Err(e) => {
                debug!("Graph-MIP leaf: encoding failed ({e:#}); undecided");
                return GraphMipLeafVerdict::Undecided;
            }
        };
        if pin_mode_is_fix() {
            let pinned = fix_split_binaries(req.graph, &mut base, &req.splits);
            debug!(
                pinned,
                total_binaries = base.binary_vars.len(),
                "Graph-MIP leaf: premises pinned (fix mode)"
            );
        }

        // --- one decision MIP per undecided row ---
        let leaf_start = Instant::now();
        let per_row = (slice / req.rows.len() as f64).max(0.5);
        for (row_idx, (coeffs, threshold)) in req.rows.iter().enumerate() {
            // Per-leaf wall guard (defect 2): the TOTAL leaf spend respects the
            // slice even if an earlier row overran its share (model conversion
            // is not deadline-checked inside the solver).
            if leaf_start.elapsed().as_secs_f64() >= slice {
                debug!("Graph-MIP leaf: slice exhausted before row {row_idx}; undecided");
                return GraphMipLeafVerdict::Undecided;
            }
            let mut enc = base.clone();
            if let Err(e) = enc.add_violation_row(coeffs, *threshold as f64) {
                debug!("Graph-MIP leaf: row {row_idx} emission failed ({e:#}); undecided");
                return GraphMipLeafVerdict::Undecided;
            }
            // CERTIFIED PHASE-ENUMERATION lane (#relational-bab): small free
            // -binary rows are decided as 2^k Farkas-certified LP leaves —
            // the ONLY certified route for case-split infeasibility (ay's
            // MIP-BaB verdict carries no certificate and is never admitted).
            // Any miss falls through to the plain solve below unchanged.
            if enc.binary_vars.len() <= leaf_certified_enum_cap() {
                let row_deadline = Instant::now()
                    + std::time::Duration::from_secs_f64(
                        per_row
                            .min(slice - leaf_start.elapsed().as_secs_f64())
                            .max(0.1),
                    );
                match certified_phase_enumeration(&enc, self.backend, row_deadline) {
                    Some(true) => {
                        debug!(
                            "Graph-MIP leaf: row {row_idx} certified-UNSAT by phase \
                             enumeration ({} leaves)",
                            1u64 << enc.binary_vars.len()
                        );
                        continue;
                    }
                    Some(false) | None => {
                        // Not refuted by enumeration — the plain solve still
                        // gets its chance (witness search / certified LP).
                    }
                }
            }
            let solver = MipSolver::new(
                enc.into_parts(),
                MipConfig {
                    backend: self.backend,
                    timeout_secs: per_row,
                    // NO phase-split racing for LEAF solves (defect 2): the
                    // measured racing fan-out (2^4 = 16 concurrent subproblems,
                    // each an exact-rational copy of the model) is the 24GB-box
                    // memory bomb. `1` is the solver's explicit disable — one
                    // sequential subproblem, one model, the ay time limit
                    // bounding it.
                    parallel_split: 1,
                    ..Default::default()
                },
            );
            let result = match solver.check_feasibility() {
                Ok(r) => r,
                Err(e) => {
                    debug!("Graph-MIP leaf: row {row_idx} solve failed ({e}); undecided");
                    return GraphMipLeafVerdict::Undecided;
                }
            };
            match result {
                // (f) 0-wrong moat: certified evidence only.
                MipResult::Unsat { certified: true } => {
                    debug!("Graph-MIP leaf: row {row_idx} certified-UNSAT (verified here)");
                }
                MipResult::Unsat { certified: false } => {
                    debug!(
                        "Graph-MIP leaf: row {row_idx} UNSAT but uncertified — not admitted; \
                         undecided"
                    );
                    return GraphMipLeafVerdict::Undecided;
                }
                MipResult::Sat { input_values, .. } => {
                    // Witness gate: clamp into the DOMAIN's input box, re-run
                    // the ORIGINAL graph forward, re-check THIS row's margin.
                    return match revalidate_leaf_witness(
                        req.graph,
                        req.input_bounds,
                        &input_values,
                        coeffs,
                        *threshold,
                    ) {
                        Some((witness, output)) => {
                            // LATCH OFF (defect 3): a real margin≤threshold
                            // point means certified-UNSAT is unreachable here;
                            // repeating leaf solves would only re-discover it.
                            // The loop REQUEUES the child (never drops it), so
                            // this can never end the run.
                            self.sat_latch
                                .store(true, std::sync::atomic::Ordering::Relaxed);
                            warn!(
                                "Graph-MIP leaf: row {row_idx} SAT witness CONFIRMED in-box \
                                 (margin <= threshold at the graph forward); oracle latched off"
                            );
                            GraphMipLeafVerdict::Violated { witness, output }
                        }
                        None => {
                            debug!(
                                "Graph-MIP leaf: row {row_idx} SAT witness unconfirmed; undecided"
                            );
                            GraphMipLeafVerdict::Undecided
                        }
                    };
                }
                MipResult::Timeout => {
                    debug!("Graph-MIP leaf: row {row_idx} timeout; undecided");
                    return GraphMipLeafVerdict::Undecided;
                }
                MipResult::Error(e) => {
                    debug!("Graph-MIP leaf: row {row_idx} error ({e}); undecided");
                    return GraphMipLeafVerdict::Undecided;
                }
            }
        }
        info!(
            rows = req.rows.len(),
            depth = req.depth,
            "Graph-MIP leaf: ALL rows certified-UNSAT — subdomain verified"
        );
        GraphMipLeafVerdict::VerifiedAllRows
    }
}

/// Clamp a leaf witness into the domain's input box, re-run the ORIGINAL
/// graph forward (engine=None, point forward), and confirm the row's margin
/// (`coeffs · output <= threshold` — the decision row the MIP satisfied).
/// `Some((witness, output))` only on confirmation.
fn revalidate_leaf_witness(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    raw_input: &[f64],
    coeffs: &[f32],
    threshold: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let clamped = clamp_witness_to_box(raw_input, input);
    let pt = BoundedTensor::concrete(clamped.clone()).ok()?;
    let out = graph.propagate_concrete_point(&pt, None, None).ok()?;
    let output = out.center();
    let margin: f64 = coeffs
        .iter()
        .zip(output.iter())
        .map(|(c, y)| *c as f64 * *y as f64)
        .sum();
    if margin <= threshold as f64 {
        Some((
            clamped.iter().copied().collect(),
            output.iter().copied().collect(),
        ))
    } else {
        None
    }
}

/// Oracle for the RELATIONAL edge-domain escalation (#relational-bab): the
/// same certified-UNSAT-only leaf solver, armed by DEFAULT for the relational
/// input-split lane (its own consult gates live in `BetaCrownConfig::
/// input_split_edge_milp*`); `NY_REL_EDGE_MILP=0` disarms. Independent of the
/// `NY_GRAPH_MIP_LEAF` gate, which governs the ReLU-split leaf lane.
pub(crate) fn relational_edge_milp_oracle() -> Option<Arc<dyn GraphMipLeafOracle>> {
    if std::env::var("NY_REL_EDGE_MILP").ok().as_deref() == Some("0") {
        return None;
    }
    info!("Graph-MIP edge oracle armed for the relational input-split lane");
    Some(Arc::new(GraphMipLeafSolver::new(MipBackend::Ay)))
}

/// Whether the SOUND α-CROWN big-M tightening runs (default ON; `=0` disables,
/// leaving the pre-reframe pure-CROWN-IBP boxes for A/B measurement).
fn whole_net_tighten_enabled() -> bool {
    !matches!(
        std::env::var("NY_REL_WHOLE_MIP_TIGHTEN").ok().as_deref(),
        Some("0")
    )
}

/// α-CROWN optimization iterations for the tightening pass (`NY_REL_WHOLE_MIP_TIGHTEN_ITERS`).
fn alpha_tighten_iterations() -> usize {
    std::env::var("NY_REL_WHOLE_MIP_TIGHTEN_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20)
}

/// Seconds to spend on α-CROWN tightening, reserving the rest of the slice for
/// the solve. `NY_REL_WHOLE_MIP_TIGHTEN_S` sets an absolute cap; otherwise 40%
/// of the remaining budget, capped at 30s.
fn alpha_tighten_budget_secs(remaining: f64) -> f64 {
    match std::env::var("NY_REL_WHOLE_MIP_TIGHTEN_S")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
    {
        Some(cap) => cap.min(remaining),
        None => (remaining * 0.4).min(30.0),
    }
}

/// Summary stats over the finite per-neuron box widths: `(max, median,
/// count>1000, total)`. Drives the before/after big-M looseness measurement.
fn box_width_stats(bounds: &HashMap<String, Vec<Bound>>) -> (f64, f64, usize, usize) {
    let mut widths: Vec<f64> = Vec::new();
    for bs in bounds.values() {
        for b in bs {
            let w = f64::from(b.upper() - b.lower());
            if w.is_finite() {
                widths.push(w);
            }
        }
    }
    if widths.is_empty() {
        return (0.0, 0.0, 0, 0);
    }
    let total = widths.len();
    let over = widths.iter().filter(|&&w| w > 1000.0).count();
    let max = widths.iter().copied().fold(0.0_f64, f64::max);
    widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = widths[total / 2];
    (max, median, over, total)
}

/// SOUND per-neuron big-M tightening: run α-CROWN with CROWN-tight INTERMEDIATE
/// bounds and INTERSECT it, neuron-by-neuron, into the incoming CROWN-IBP boxes
/// (`lo := max(ibp.lo, α.lo)`, `hi := min(ibp.hi, α.hi)`).
///
/// SOUNDNESS: both α-CROWN and CROWN-IBP are certified SOUND outer bounds of the
/// true reachable pre-activation set (each is a valid over-approximation), so
/// their intersection still contains that set — it is a valid outer bound, never
/// looser than CROWN-IBP, and can only shrink the big-M. A too-tight bound would
/// be a wrong verdict; intersection cannot produce one. The `lo <= hi` guard
/// keeps the original box if directed rounding ever crosses the bounds (never
/// widen, never invert). Any α-CROWN failure (unsupported op, deadline, shape
/// mismatch) leaves the CROWN-IBP boxes untouched — fail-open to the
/// looser-but-sound relaxation.
fn intersect_alpha_crown_tightening(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    flat_bounds: &mut HashMap<String, Vec<Bound>>,
    deadline: Instant,
) {
    let cfg = AlphaCrownConfig {
        // The lever: CROWN-tightened intermediate pre-activation bounds (captures
        // the difference-net correlation IBP throws away), not IBP intermediates.
        fix_interm_bounds: false,
        iterations: alpha_tighten_iterations(),
        deadline: Some(deadline),
        ..AlphaCrownConfig::default()
    };
    let alpha = match graph.collect_alpha_crown_bounds_dag(input, &cfg) {
        Ok((bounds, _state)) => bounds,
        Err(e) => {
            debug!("rel whole-net MIP: α-CROWN unavailable ({e}); keeping CROWN-IBP boxes");
            return;
        }
    };
    let mut nodes = 0usize;
    let mut neurons = 0usize;
    for (name, ibp_bounds) in flat_bounds.iter_mut() {
        let Some(a_bt) = alpha.get(name) else {
            continue;
        };
        let Ok(a_bounds) = bounded_tensor_to_bounds(a_bt) else {
            continue;
        };
        if a_bounds.len() != ibp_bounds.len() {
            continue;
        }
        let mut touched = false;
        for (b, a) in ibp_bounds.iter_mut().zip(a_bounds.iter()) {
            let lo = b.lower().max(a.lower());
            let hi = b.upper().min(a.upper());
            // Intersection of two sound outer bounds; defensive inversion guard.
            if lo <= hi && (lo > b.lower() || hi < b.upper()) {
                *b = Bound::new(lo, hi);
                touched = true;
                neurons += 1;
            }
        }
        if touched {
            nodes += 1;
        }
    }
    debug!(
        nodes,
        neurons, "rel whole-net MIP: α-CROWN ∩ CROWN-IBP tightened per-neuron boxes"
    );
}

/// Whether the SOUND per-neuron OBBT pass runs after the α-CROWN intersect.
///
/// DEFAULT-OFF (opt-in via `NY_REL_WHOLE_MIP_OBBT=1`). MEASURED on
/// isomorphic_acasxu_2026/2.0 (instances 0/1/6): OBBT over the difference-net
/// LP (triangle) relaxation tightened ZERO of the loosest pre-activation columns
/// (`tightened_cols=0`) beyond both the raw CROWN-IBP boxes AND the α-CROWN ∩
/// boxes — the max/median widths are byte-identical before and after. The cause
/// is structural: the finisher encodes the net with NO output/property
/// constraint, so an LP min/max of an intermediate is just its forward-reachable
/// range, which α-CROWN (free intermediate bounds) already attains (by LP
/// duality the triangle-relaxation LP bound is the optimal-α CROWN bound). The
/// pass costs ~0.1–0.4s per neuron, so defaulting it ON would burn the finisher
/// slice for no shrink. Kept, sound, and unit-tested (`ny-mip`
/// `obbt_relaxation_tightens_coupled_box`) as a reusable capability; the lever
/// that WOULD bite is property-CONDITIONED OBBT (the violation row in the LP),
/// which is a larger, per-band-row change.
fn whole_net_obbt_enabled() -> bool {
    matches!(
        std::env::var("NY_REL_WHOLE_MIP_OBBT").ok().as_deref(),
        Some("1")
    )
}

/// Whether PROPERTY-CONDITIONED OBBT runs (opt-in, `NY_REL_WHOLE_MIP_OBBT_COND=1`).
///
/// Unlike the unconditioned pass (a measured no-op — α-CROWN already attains the
/// forward-reachable LP bound), this adds the per-band-row VIOLATION constraint
/// into the OBBT LP, so each intermediate is bounded within that row's violation
/// region (a strict subset of the forward-reachable set). It is PER-ROW: each
/// band row is conditioned, re-encoded and solved against its OWN tightened
/// boxes, because a box valid over one row's violation region is NOT a valid
/// outer bound over another's (their regions differ). DEFAULT-OFF: it costs
/// `2·targets` extra LP solves per row; armed for measurement / when the extra
/// shrink closes an instance the shared big-M cannot.
fn whole_net_obbt_cond_enabled() -> bool {
    matches!(
        std::env::var("NY_REL_WHOLE_MIP_OBBT_COND").ok().as_deref(),
        Some("1")
    )
}

/// Fraction of a band row's per-row slice spent on the conditioned OBBT pass;
/// the rest goes to that row's decision solve. `NY_REL_WHOLE_MIP_OBBT_COND_FRAC`
/// (default 0.5), clamped to `[0, 0.9]`.
fn obbt_cond_frac() -> f64 {
    obbt_env_f64("NY_REL_WHOLE_MIP_OBBT_COND_FRAC", 0.5).clamp(0.0, 0.9)
}

fn obbt_env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn obbt_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Seconds to spend on the OBBT pass. `NY_REL_WHOLE_MIP_OBBT_S` caps it
/// absolutely; otherwise 60% of the budget still remaining after α-CROWN,
/// capped at 40s (the finisher keeps the rest for the row solves).
fn obbt_budget_secs(remaining: f64) -> f64 {
    match std::env::var("NY_REL_WHOLE_MIP_OBBT_S")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
    {
        Some(cap) => cap.min(remaining),
        None => (remaining * 0.6).min(40.0),
    }
}

/// SOUND per-neuron OBBT with FULL LP coupling — the lever α-CROWN's per-node
/// intersect cannot reach. AFTER the α-CROWN ∩ CROWN-IBP intersect and BEFORE
/// the big-M rows are emitted, encode the whole difference net at the current
/// boxes, relax it to its LP (triangle) relaxation, and rigorously min/max each
/// still-loose pre-activation column over that relaxation with EVERY other
/// neuron's current bound in force. Intersect the proven `[lp_min, lp_max]` into
/// each box (`lo := max(box.lo, lp_min)`, `hi := min(box.hi, lp_max)`). Re-encode
/// so the tighter boxes tighten the big-M slopes, then repeat.
///
/// SOUNDNESS: the LP relaxation's feasible set CONTAINS the true reachable set
/// (both the triangle relaxation of each ReLU and the affine rows are valid
/// over-approximations), so each column's rigorous LP min/max is a valid OUTER
/// bound on its reachable pre-activation — intersecting can only SHRINK a box to
/// a still-valid range, never exclude a reachable state, never produce a wrong
/// verdict. Every committed OBBT bound is a rigorous dual bound, outward-rounded
/// to f64 by ay and then to f32 here (`next_down_f32` for lows, `next_up_f32`
/// for highs), clamped so it never widens the incoming box. The invariant "every
/// box in `flat_bounds` is a valid outer bound" holds before the pass and each
/// intersect preserves it, so re-encoding at the tightened boxes stays a valid
/// relaxation across rounds. FAIL-OPEN: any encode/solve error, or an infeasible
/// relaxation, leaves the current boxes untouched (the α-CROWN result stands).
/// Coverage/effect diagnostics from one [`obbt_tighten_boxes`] pass.
#[derive(Debug, Default, Clone, Copy)]
struct ObbtDiag {
    /// Columns wider than the threshold that node_cols can reach (selectable).
    selectable: usize,
    /// Columns actually optimized (top-K of `selectable`).
    targets: usize,
    /// Target columns whose LP box shrank at least once (from ay's report).
    tightened_cols: usize,
    /// `flat_bounds` entries the shrink was written back into.
    boxes_shrunk: usize,
}

fn obbt_tighten_boxes(
    shaped: &GraphNetwork,
    input_bounds: &[Bound],
    flat_bounds: &mut HashMap<String, Vec<Bound>>,
    deadline: Instant,
    violation: Option<(&[f32], f64)>,
) -> ObbtDiag {
    let mut diag = ObbtDiag::default();
    let width_thresh = obbt_env_f64("NY_REL_WHOLE_MIP_OBBT_WIDTH", 1000.0);
    // Targeted top-K: each rigorous LP min/max on the diff-net relaxation is
    // ~0.1s (the exact rim engages on the ill-conditioned tower rows), so full
    // OBBT over all ~500 loose neurons blows any slice. Tighten the loosest K.
    let max_n = obbt_env_usize("NY_REL_WHOLE_MIP_OBBT_MAXN", 128);
    let inner_rounds = obbt_env_usize("NY_REL_WHOLE_MIP_OBBT_ROUNDS", 2).max(1);
    let outer_rounds = obbt_env_usize("NY_REL_WHOLE_MIP_OBBT_OUTER", 1).max(1);

    for outer in 0..outer_rounds {
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        if remaining < 1.0 {
            debug!("rel whole-net OBBT: <1s remains at outer round {outer}; stopping");
            break;
        }
        // Encode at the CURRENT boxes so the LP relaxation's big-M slopes reflect
        // every tightening from the previous round.
        let mut enc = match super::graph_mip::encode_graph_with_deadline(
            shaped,
            input_bounds,
            flat_bounds,
            Some(deadline),
        ) {
            Ok(enc) => enc,
            Err(e) => {
                debug!("rel whole-net OBBT: encode failed ({e:#}); fail-open");
                break;
            }
        };
        // PROPERTY-CONDITIONED OBBT (#rel-whole-mip): add the band-VIOLATION row
        // (`Σ coeffs·y <= threshold`, the exact linear row the whole-net MILP
        // asserts for this band direction) into the LP BEFORE min/max-ing each
        // intermediate. Each intermediate is then bounded WITHIN this row's
        // violation region — a strict subset of the forward-reachable set —
        // rather than over the whole forward range (which α-CROWN already
        // attains). SOUNDNESS: the OBBT LP is still a RELAXATION of the exact
        // violation region (triangle ReLU ⊇ exact ReLU; the violation row is
        // exact/linear and present in both), so each column's rigorous min/max
        // stays a valid OUTER bound over the violation region — the tightened
        // box never cuts off a point that violates THIS row. Fail-open: a row
        // that cannot be emitted keeps the unconditioned (looser) boxes.
        if let Some((coeffs, threshold)) = violation {
            if let Err(e) = enc.add_violation_row(coeffs, threshold) {
                debug!(
                    "rel whole-net OBBT: conditioning violation row emit failed ({e:#}); \
                     keeping unconditioned boxes for this round"
                );
                break;
            }
        }

        // Collect still-loose pre-activation columns. `node_cols[name][i]` is the
        // problem column carrying node `name`'s i-th output — the pre-activation
        // of any ReLU fed by `name`. Dedupe by column index (pass-throughs alias
        // the same column across nodes) so a column is optimized once and written
        // back to every box that reads it.
        let mut by_col: HashMap<usize, (f64, Vec<(String, usize)>)> = HashMap::new();
        for (name, cols) in &enc.node_cols {
            let Some(bounds) = flat_bounds.get(name) else {
                continue;
            };
            if cols.len() != bounds.len() {
                continue;
            }
            for (i, (&col, b)) in cols.iter().zip(bounds.iter()).enumerate() {
                let w = f64::from(b.upper() - b.lower());
                if w.is_finite() && w > width_thresh {
                    let entry = by_col.entry(col.0).or_insert_with(|| (w, Vec::new()));
                    entry.0 = entry.0.max(w);
                    entry.1.push((name.clone(), i));
                }
            }
        }
        if by_col.is_empty() {
            debug!(
                "rel whole-net OBBT: no columns wider than {width_thresh} at outer round {outer}"
            );
            break;
        }
        // Rank loosest-first and keep the top-K (the diff-net pre-activations, the
        // big-M drivers, dominate the width distribution — top-K targets them).
        let mut ranked: Vec<(usize, f64, Vec<(String, usize)>)> = by_col
            .into_iter()
            .map(|(col, (w, uses))| (col, w, uses))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        diag.selectable = ranked.len();
        if max_n > 0 && ranked.len() > max_n {
            ranked.truncate(max_n);
        }
        diag.targets = ranked.len();
        let targets: Vec<ny_mip::ir::Col> = ranked
            .iter()
            .map(|(col, _, _)| ny_mip::ir::Col(*col))
            .collect();

        let now = Instant::now();
        if now >= deadline {
            break;
        }
        // Per-solve advisory cap: a single warm continuous LP min/max is ms-scale;
        // a solve that grinds past this bails to a non-rigorous verdict (tightens
        // nothing) rather than eating the slice. The `deadline`, checked between
        // chunks inside `obbt_relaxation_bounds`, is the hard authority.
        let per_solve = std::time::Duration::from_secs_f64(1.0);
        let chunk = obbt_env_usize("NY_REL_WHOLE_MIP_OBBT_CHUNK", 8).max(1);
        let report = match obbt_relaxation_bounds(
            &enc.problem,
            &targets,
            inner_rounds,
            per_solve,
            deadline,
            chunk,
        ) {
            Ok(r) => r,
            Err(e) => {
                debug!("rel whole-net OBBT: solve error ({e}); fail-open");
                break;
            }
        };
        if report.infeasible {
            // The RELAXATION is infeasible — a strictly stronger fact than any
            // box tightening, but this path only tightens boxes; the certified
            // decision path re-derives (and certifies) infeasibility itself.
            // Keep the current boxes and stop.
            debug!("rel whole-net OBBT: relaxation infeasible; keeping boxes, stopping");
            break;
        }

        let mut shrunk = 0usize;
        for ((_, _, uses), (lp_lo, lp_hi)) in ranked.iter().zip(report.bounds.iter()) {
            // f64→f32: lows round DOWN, highs round UP; clamp so the LP result
            // never widens the incoming box (defensive inversion guard too).
            let cand_lo = next_down_f32(*lp_lo as f32);
            let cand_hi = next_up_f32(*lp_hi as f32);
            for (name, i) in uses {
                if let Some(bs) = flat_bounds.get_mut(name) {
                    let cur = bs[*i];
                    let lo = cur.lower().max(cand_lo);
                    let hi = cur.upper().min(cand_hi);
                    if lo <= hi && (lo > cur.lower() || hi < cur.upper()) {
                        bs[*i] = Bound::new(lo, hi);
                        shrunk += 1;
                    }
                }
            }
        }
        diag.tightened_cols = report.tightened;
        diag.boxes_shrunk += shrunk;
        debug!(
            outer,
            targets = targets.len(),
            rounds = report.rounds,
            tightened_cols = report.tightened,
            boxes_shrunk = shrunk,
            "rel whole-net OBBT: coupled LP pass"
        );
        if shrunk == 0 {
            break; // fixpoint: nothing moved this round
        }
    }
    diag
}

/// #rel-whole-mip — WHOLE-NET certified-UNSAT MILP on a DIFFERENCE network.
///
/// The last-resort finisher for the relational unsat lane: when input-split
/// BaB stalls at the paired-ReLU relaxation floor (measured-dead for every
/// generic lever), encode the WHOLE difference net (Sub of two towers — all
/// encodable ops) as ONE exact MIP and require EVERY band row's decision MIP
/// to be CERTIFIED-UNSAT. `tree_cert` (ay_lib.rs) now admits case-split UNSAT
/// at ANY k with a verified per-leaf Farkas skeleton, so a whole-net UNSAT is
/// admissible the instant ay can solve the model; ay's UNCERTIFIED BaB UNSAT
/// is never admitted — the 0-wrong moat.
///
/// Returns `true` iff the whole band is certified verified. FAIL-OPEN: any
/// miss (encode failure, uncertified verdict, timeout, solver error, panic)
/// → `false`, and the caller keeps its inconclusive BaB verdict.
///
/// READINESS POSTURE: the 530-binary diff-net root MILP is (as of today)
/// beyond ay's w2/w5 ladder, so this times out and returns `false` (harmless).
/// It is the consumption path that closes the last iso holdouts the instant ay
/// can solve full-rung — the INFO line reports the exact (cols, binaries) so
/// each ay bump re-measures whether the instances are now in-budget.
pub(crate) fn whole_net_certified_band_unsat(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    rows: &[(Vec<f32>, f32)],
    slice_secs: f64,
    deadline: Option<Instant>,
) -> bool {
    contain_leaf_panics(|| {
        whole_net_certified_band_unsat_inner(graph, input_bounds, rows, slice_secs, deadline)
    })
    .is_verified()
}

/// Panic-contained core (returns a leaf verdict so the boundary containment
/// helper — infallible by contract — can wrap it).
fn whole_net_certified_band_unsat_inner(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    rows: &[(Vec<f32>, f32)],
    slice_secs: f64,
    deadline: Option<Instant>,
) -> GraphMipLeafVerdict {
    use std::time::Duration;
    if rows.is_empty() {
        return GraphMipLeafVerdict::Undecided;
    }
    if !slice_secs.is_finite() || slice_secs <= 0.0 {
        debug!("rel whole-net MIP: invalid slice {slice_secs}; fail-open");
        return GraphMipLeafVerdict::Undecided;
    }
    let slice = slice_secs.max(1.0);
    let Ok(slice_duration) = Duration::try_from_secs_f64(slice) else {
        debug!("rel whole-net MIP: unrepresentable slice {slice}; fail-open");
        return GraphMipLeafVerdict::Undecided;
    };
    let start = Instant::now();
    let Some(slice_deadline) = start.checked_add(slice_duration) else {
        debug!("rel whole-net MIP: slice deadline overflow; fail-open");
        return GraphMipLeafVerdict::Undecided;
    };
    // One authority caps every finisher phase: the original caller deadline
    // (which accounts for BaB overshoot) AND this invocation's normalized
    // slice. Direct callers without an outer deadline remain slice-bounded.
    let work_deadline = deadline.unwrap_or(slice_deadline).min(slice_deadline);
    let work_slice_secs = work_deadline.saturating_duration_since(start).as_secs_f64();

    // Input box → tensor.
    let mut lo = Vec::with_capacity(input_bounds.len());
    let mut hi = Vec::with_capacity(input_bounds.len());
    for b in input_bounds {
        lo.push(b.lower());
        hi.push(b.upper());
    }
    let Ok(input) = BoundedTensor::new(
        ndarray::Array1::from(lo).into_dyn(),
        ndarray::Array1::from(hi).into_dyn(),
    ) else {
        return GraphMipLeafVerdict::Undecided;
    };
    if Instant::now() >= work_deadline {
        debug!("rel whole-net MIP: deadline exhausted before node-bound collection; fail-open");
        return GraphMipLeafVerdict::Undecided;
    }
    // Per-node big-M boxes over the input box (per-node CROWN-IBP, tight).
    let Ok(node_bounds_bt) = graph.collect_crown_ibp_bounds_dag_with_deadline_and_engine(
        &input,
        Some(work_deadline),
        None,
    ) else {
        debug!("rel whole-net MIP: node-bound collection failed; fail-open");
        return GraphMipLeafVerdict::Undecided;
    };
    if Instant::now() >= work_deadline {
        debug!("rel whole-net MIP: slice exhausted during node-bound collection; fail-open");
        return GraphMipLeafVerdict::Undecided;
    }
    let mut flat_bounds: HashMap<String, Vec<Bound>> = HashMap::with_capacity(node_bounds_bt.len());
    for (name, bt) in &node_bounds_bt {
        match bounded_tensor_to_bounds(bt) {
            Ok(b) => {
                flat_bounds.insert(name.clone(), b);
            }
            Err(_) => return GraphMipLeafVerdict::Undecided,
        }
    }
    // BIG-M TIGHTENING (Part B): the difference net's big-M constants ARE the
    // per-neuron pre-activation box widths (`encode_relu_node`: `y <= u·z`,
    // `y <= x - l(1-z)`). CROWN-IBP discards the two isomorphic subnets'
    // correlation, leaving those widths loose by orders of magnitude vs the
    // verification band — no MIP engine closes that gap. SOUNDLY shrink them
    // BEFORE the big-M rows are emitted by intersecting an α-CROWN pass (with
    // CROWN-tight INTERMEDIATE bounds, `fix_interm_bounds=false`) into the
    // CROWN-IBP boxes. Gate: `NY_REL_WHOLE_MIP_TIGHTEN=0` keeps pure CROWN-IBP.
    let (w_max0, w_med0, w_over0, w_tot0) = box_width_stats(&flat_bounds);
    info!(
        max_width = w_max0,
        median_width = w_med0,
        wide_gt1000 = w_over0,
        continuous = w_tot0,
        "rel whole-net MIP: pre-activation box widths BEFORE tightening"
    );
    // Declared shapes on a clone (stitched diff nets carry none — the encoder's
    // const-op/plumbing arms fail closed without them). Built BEFORE tightening
    // so the OBBT pass can re-encode the difference net at the current boxes.
    let mut shaped = graph.clone();
    if shaped.declared_shape(ny_propagate::NETWORK_INPUT).is_none() {
        shaped.set_declared_shape(ny_propagate::NETWORK_INPUT, input.shape().to_vec());
    }
    for (name, bt) in &node_bounds_bt {
        if shaped.declared_shape(name).is_none() {
            shaped.set_declared_shape(name.clone(), bt.shape().to_vec());
        }
    }
    if whole_net_tighten_enabled() {
        let remaining_for_tighten = work_deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        let alpha_budget = alpha_tighten_budget_secs(remaining_for_tighten);
        if alpha_budget >= 1.0 {
            if let Some(alpha_deadline) = Instant::now()
                .checked_add(Duration::from_secs_f64(alpha_budget))
                .map(|d| d.min(work_deadline))
            {
                intersect_alpha_crown_tightening(graph, &input, &mut flat_bounds, alpha_deadline);
                let (w_max1, w_med1, w_over1, _) = box_width_stats(&flat_bounds);
                info!(
                    max_width = w_max1,
                    median_width = w_med1,
                    wide_gt1000 = w_over1,
                    "rel whole-net MIP: pre-activation box widths AFTER α-CROWN ∩ CROWN-IBP"
                );
            }
        } else {
            debug!(
                "rel whole-net MIP: too little budget ({alpha_budget:.1}s) for α-CROWN tighten; \
                 keeping CROWN-IBP boxes"
            );
        }
        // PER-NEURON OBBT WITH FULL LP COUPLING (opt-in, `NY_REL_WHOLE_MIP_OBBT=1`;
        // see `whole_net_obbt_enabled` for the measured no-op finding). Rigorously
        // min/max each still-loose pre-activation over the whole difference net's
        // LP relaxation with every other neuron's bound in force, and intersect
        // the proven range. Sound but DEFAULT-OFF: on the real iso diff nets it
        // shrinks the big-M by zero (α-CROWN already attains the LP bound).
        if whole_net_obbt_enabled() {
            let remaining_for_obbt = work_deadline
                .saturating_duration_since(Instant::now())
                .as_secs_f64();
            let obbt_budget = obbt_budget_secs(remaining_for_obbt);
            if obbt_budget >= 1.0 {
                if let Some(obbt_deadline) = Instant::now()
                    .checked_add(Duration::from_secs_f64(obbt_budget))
                    .map(|d| d.min(work_deadline))
                {
                    let diag = obbt_tighten_boxes(
                        &shaped,
                        input_bounds,
                        &mut flat_bounds,
                        obbt_deadline,
                        None,
                    );
                    let (w_max2, w_med2, w_over2, _) = box_width_stats(&flat_bounds);
                    info!(
                        max_width = w_max2,
                        median_width = w_med2,
                        wide_gt1000 = w_over2,
                        selectable = diag.selectable,
                        targets = diag.targets,
                        tightened_cols = diag.tightened_cols,
                        boxes_shrunk = diag.boxes_shrunk,
                        "rel whole-net MIP: pre-activation box widths AFTER coupled OBBT"
                    );
                }
            } else {
                debug!(
                    "rel whole-net MIP: too little budget ({obbt_budget:.1}s) for OBBT; \
                     keeping α-CROWN boxes"
                );
            }
        }
    }
    // PROPERTY-CONDITIONED OBBT lane (opt-in). The boxes tightened against a
    // band row's violation region are sound ONLY for THAT row's decision MIP, so
    // this lane runs per-row (condition → re-encode → add the row → solve) inside
    // its own helper rather than sharing one big-M `base` across every row.
    if whole_net_obbt_cond_enabled() {
        return whole_net_certified_band_unsat_conditioned(
            &shaped,
            input_bounds,
            &flat_bounds,
            rows,
            work_deadline,
            work_slice_secs,
        );
    }
    let mut base = match super::graph_mip::encode_graph_with_deadline(
        &shaped,
        input_bounds,
        &flat_bounds,
        Some(work_deadline),
    ) {
        Ok(enc) => enc,
        Err(e) => {
            debug!("rel whole-net MIP: encode failed ({e:#}); fail-open");
            return GraphMipLeafVerdict::Undecided;
        }
    };
    // #rel-diff-coupling (opt-in, `NY_REL_DIFF_COUPLING=1`): strengthen the
    // per-node triangle relaxation with SOUND paired-neuron difference-coupling
    // rows (`-δ_i <= a_X_i - b_X_i <= δ_i`), carrying the cross-tower
    // correlation the triangle relaxation discards. Every δ_i is an outward
    // upper bound on |f_X_i - g_X_i|, so the rows only remove spurious relaxed
    // points — never a real one. Fail-open: 0 rows leaves the sound baseline.
    if diff_coupling_enabled() {
        let (added, out_delta) = super::graph_mip_diff_coupling::attach_diff_coupling(
            &mut base,
            &shaped,
            input_bounds,
            &flat_bounds,
        );
        let out_delta_max = out_delta
            .as_ref()
            .map(|d| d.iter().copied().fold(0.0_f64, f64::max));
        info!(
            coupling_rows = added,
            output_delta_max = out_delta_max,
            "rel whole-net MIP: difference-coupling rows added (relaxation strengthening)"
        );
    }
    // #rel-joint-relu-cuts (opt-in, `NY_REL_JOINT_RELU_CUTS=1`): the north-star
    // lever — JOINT paired-ReLU multi-neuron cuts. For each paired UNSTABLE ReLU
    // (zf, zg), emit the concave/convex ENVELOPE facets of relu(zf)−relu(zg)
    // (and relu(zf)+relu(zg)) over P = box ∩ {|zf−zg| ≤ δ}, the diagonal-coupling
    // inequalities the two independent triangles cannot express. Each is a valid
    // inequality of the exact paired-ReLU set ∩ P (γ = rigorous extremum over P's
    // subdivision vertices, outward-rounded), so it only removes SPURIOUS relaxed
    // points. Fail-open: 0 cuts leaves the sound baseline.
    if super::graph_mip_joint_relu_cuts::joint_relu_cuts_enabled() {
        let (added, diag) = super::graph_mip_joint_relu_cuts::attach_joint_relu_cuts(
            &mut base,
            &shaped,
            input_bounds,
            &flat_bounds,
        );
        info!(
            joint_cut_rows = added,
            paired_unstable = diag.paired_unstable,
            diff_rows = diag.diff_rows,
            sum_rows = diag.sum_rows,
            delta_box_rows = diag.delta_box_rows,
            max_diff_tighten = diag.max_diff_tighten,
            "rel whole-net MIP: JOINT paired-ReLU cuts added (diagonal-coupling relaxation strengthening)"
        );
    }
    info!(
        cols = base.problem.num_cols(),
        binaries = base.binary_vars.len(),
        rows = rows.len(),
        slice_s = work_slice_secs,
        "rel whole-net MIP: encoded difference network (readiness: certified-UNSAT closes when ay solves this size)"
    );
    let per_row = (work_slice_secs / rows.len() as f64).max(0.5);
    for (ri, (coeffs, threshold)) in rows.iter().enumerate() {
        if Instant::now() >= work_deadline {
            debug!("rel whole-net MIP: slice exhausted before row {ri}; fail-open");
            return GraphMipLeafVerdict::Undecided;
        }
        let mut enc = base.clone();
        if enc
            .add_violation_row(coeffs, f64::from(*threshold))
            .is_err()
        {
            return GraphMipLeafVerdict::Undecided;
        }
        let remaining = work_deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        if remaining < 0.1 {
            debug!("rel whole-net MIP: less than 0.1s remains before row {ri}; fail-open");
            return GraphMipLeafVerdict::Undecided;
        }
        let solver = MipSolver::new(
            enc.into_parts(),
            MipConfig {
                backend: MipBackend::Ay,
                timeout_secs: per_row.min(remaining),
                parallel_split: 1,
                ..Default::default()
            },
        );
        match solver.check_feasibility() {
            // 0-wrong moat: certified evidence only (tree_cert admits any-k).
            Ok(MipResult::Unsat { certified: true }) => {}
            Ok(MipResult::Unsat { certified: false }) => {
                debug!(
                    "rel whole-net MIP: row {ri} UNSAT but uncertified — not admitted; fail-open"
                );
                return GraphMipLeafVerdict::Undecided;
            }
            other => {
                debug!("rel whole-net MIP: row {ri} inconclusive ({other:?}); fail-open");
                return GraphMipLeafVerdict::Undecided;
            }
        }
    }
    info!(
        rows = rows.len(),
        "rel whole-net MIP: ALL band rows certified-UNSAT — difference network verified"
    );
    GraphMipLeafVerdict::VerifiedAllRows
}

/// PROPERTY-CONDITIONED per-row finisher (`NY_REL_WHOLE_MIP_OBBT_COND=1`).
///
/// For EACH band row: (1) clone the shared (α-CROWN ∩ CROWN-IBP) boxes, (2) run
/// OBBT with that row's violation constraint in the LP so every intermediate is
/// bounded within THIS row's violation region, (3) re-encode the difference net
/// at those tighter boxes (tighter big-M), (4) assert the same violation row and
/// require CERTIFIED-UNSAT. Every row must certify for the band to be verified.
///
/// SOUNDNESS: box_r is a rigorous outer bound over row r's violation region
/// (the OBBT LP is the triangle relaxation of {exact ReLU ∧ affine ∧ the exact
/// linear row}, which CONTAINS that region), so re-encoding row r's MILP with
/// box_r as big-M cannot cut off any point that violates row r — infeasible ⇒
/// row r's violation region is genuinely empty. Because box_r is used ONLY for
/// row r's own MILP (never shared across rows), a point that violates a
/// DIFFERENT row can never be wrongly excluded. FAIL-OPEN throughout: any
/// encode/solve/emit failure or uncertified verdict returns `Undecided`, and the
/// conditioning only ever shrinks boxes from the sound α-CROWN baseline.
fn whole_net_certified_band_unsat_conditioned(
    shaped: &GraphNetwork,
    input_bounds: &[Bound],
    base_bounds: &HashMap<String, Vec<Bound>>,
    rows: &[(Vec<f32>, f32)],
    work_deadline: Instant,
    work_slice_secs: f64,
) -> GraphMipLeafVerdict {
    use std::time::Duration;
    let per_row_total = (work_slice_secs / rows.len() as f64).max(0.5);
    let frac = obbt_cond_frac();
    for (ri, (coeffs, threshold)) in rows.iter().enumerate() {
        if Instant::now() >= work_deadline {
            debug!("rel whole-net MIP (cond): slice exhausted before row {ri}; fail-open");
            return GraphMipLeafVerdict::Undecided;
        }
        // Per-row conditioned boxes start from the shared α-CROWN baseline.
        let mut row_bounds = base_bounds.clone();
        let thr = f64::from(*threshold);

        // (a) violation-conditioned OBBT for THIS row.
        let obbt_budget = (per_row_total * frac).min(
            work_deadline
                .saturating_duration_since(Instant::now())
                .as_secs_f64(),
        );
        if obbt_budget >= 1.0 {
            if let Some(obbt_deadline) = Instant::now()
                .checked_add(Duration::from_secs_f64(obbt_budget))
                .map(|d| d.min(work_deadline))
            {
                let diag = obbt_tighten_boxes(
                    shaped,
                    input_bounds,
                    &mut row_bounds,
                    obbt_deadline,
                    Some((coeffs, thr)),
                );
                let (w_max, w_med, w_over, _) = box_width_stats(&row_bounds);
                info!(
                    row = ri,
                    max_width = w_max,
                    median_width = w_med,
                    wide_gt1000 = w_over,
                    selectable = diag.selectable,
                    targets = diag.targets,
                    tightened_cols = diag.tightened_cols,
                    boxes_shrunk = diag.boxes_shrunk,
                    "rel whole-net MIP: box widths AFTER property-conditioned OBBT"
                );
            }
        }

        // (b) re-encode at the row-conditioned boxes (tighter big-M).
        let mut enc = match super::graph_mip::encode_graph_with_deadline(
            shaped,
            input_bounds,
            &row_bounds,
            Some(work_deadline),
        ) {
            Ok(enc) => enc,
            Err(e) => {
                debug!("rel whole-net MIP (cond): row {ri} encode failed ({e:#}); fail-open");
                return GraphMipLeafVerdict::Undecided;
            }
        };
        // (c') optional difference-coupling rows on THIS row's boxes.
        if diff_coupling_enabled() {
            let (added, _) = super::graph_mip_diff_coupling::attach_diff_coupling(
                &mut enc,
                shaped,
                input_bounds,
                &row_bounds,
            );
            debug!(
                row = ri,
                coupling_rows = added,
                "rel whole-net MIP (cond): coupling rows"
            );
        }
        // (c'') optional JOINT paired-ReLU cuts on THIS row's conditioned boxes.
        if super::graph_mip_joint_relu_cuts::joint_relu_cuts_enabled() {
            let (added, diag) = super::graph_mip_joint_relu_cuts::attach_joint_relu_cuts(
                &mut enc,
                shaped,
                input_bounds,
                &row_bounds,
            );
            debug!(
                row = ri,
                joint_cut_rows = added,
                paired_unstable = diag.paired_unstable,
                "rel whole-net MIP (cond): joint paired-ReLU cuts"
            );
        }
        // (c) assert the SAME violation row this box was conditioned on.
        if enc.add_violation_row(coeffs, thr).is_err() {
            return GraphMipLeafVerdict::Undecided;
        }
        let remaining = work_deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        if remaining < 0.1 {
            debug!("rel whole-net MIP (cond): <0.1s remains before row {ri} solve; fail-open");
            return GraphMipLeafVerdict::Undecided;
        }
        let solve_budget = (per_row_total * (1.0 - frac)).max(0.1).min(remaining);
        let solver = MipSolver::new(
            enc.into_parts(),
            MipConfig {
                backend: MipBackend::Ay,
                timeout_secs: solve_budget,
                parallel_split: 1,
                ..Default::default()
            },
        );
        match solver.check_feasibility() {
            // 0-wrong moat: certified evidence only (tree_cert admits any-k).
            Ok(MipResult::Unsat { certified: true }) => {}
            Ok(MipResult::Unsat { certified: false }) => {
                debug!(
                    "rel whole-net MIP (cond): row {ri} UNSAT but uncertified — not admitted; \
                     fail-open"
                );
                return GraphMipLeafVerdict::Undecided;
            }
            other => {
                debug!("rel whole-net MIP (cond): row {ri} inconclusive ({other:?}); fail-open");
                return GraphMipLeafVerdict::Undecided;
            }
        }
    }
    info!(
        rows = rows.len(),
        "rel whole-net MIP (cond): ALL band rows certified-UNSAT under property-conditioned \
         big-M — difference network verified"
    );
    GraphMipLeafVerdict::VerifiedAllRows
}

trait IsVerified {
    fn is_verified(&self) -> bool;
}
impl IsVerified for GraphMipLeafVerdict {
    fn is_verified(&self) -> bool {
        matches!(self, GraphMipLeafVerdict::VerifiedAllRows)
    }
}

/// Build the default-on ReLU-split leaf oracle. The caller attaches it to the
/// verifier; exact `NY_GRAPH_MIP_LEAF=0` returns `None` and leaves BaB unchanged.
pub(super) fn maybe_graph_mip_leaf_oracle(
    backend: MipBackend,
) -> Option<Arc<dyn GraphMipLeafOracle>> {
    if !graph_mip_leaf_enabled() {
        return None;
    }
    info!("Graph-MIP LEAF oracle armed (default-on; NY_GRAPH_MIP_LEAF=0 disables)");
    Some(Arc::new(GraphMipLeafSolver::new(backend)))
}

#[cfg(test)]
#[path = "graph_mip_leaf_tests.rs"]
mod tests;
