// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#vit-zono-root` (VIBECHECK plan I1 — "vit certified forward-zonotope root
//! map (s0 I1)", the critical path to the vit_2023 pool): ONE certified
//! correlation-preserving zonotope forward pass over the ROOT input box, whose
//! per-node boxes are intersected SHRINK-ONLY into the alpha reference map the
//! root CROWN / alpha-margin pass reads.
//!
//! # Why
//!
//! ny's vit reference map is IBP, or a CROWN-IBP collection that truncates on
//! vit (`crown_done=1 ibp_fallback=18`, diag6/vit.md). IBP loses the input
//! dependence at every affine stage, so the fixed-index softmax's relative
//! logits `x_i - x_{j*}` widen to `u_i - l_{j*}` (the measured `-0.07` of
//! `softmax: complex_fixed_index` on ibp) and the attention products blow up
//! on pgd. A zonotope keeps every symbol, so the shared query-row term
//! cancels EXACTLY in `x_i - x_{j*}` and in `Q K^T`. ny's own independent DeepZ
//! oracle (`scratchpad/vit_frontier_2026-08-06/RESULTS.txt` §2) reached the
//! full-collection root on ibp (-0.21/-0.20) for one forward pass, and -7..-17
//! on pgd against the partial collection's -1.3e4.
//!
//! # How it is consumed
//!
//! [`crate::GraphNetwork`] carries a graph-local [`VitZonoRootPlan`] (stamped
//! by the CLI from the typed key and, sticky-on, by
//! `BetaCrownVerifier::configured_graph_for_crown`). When it is present, the
//! alpha reference collector (`graph_alpha/bounds/alpha.rs`):
//!
//! 1. computes (or serves from an exact-box cache) this map;
//! 2. hands the CROWN-IBP collector ny's IBP map with every zonotope entry
//!    substituted as its baseline (`precomputed_ibp`), so every CROWN target
//!    is intersected with — and every truncated target keeps — the zonotope
//!    box (the `fix_interm_bounds: false` typed route);
//! 3. intersects the returned map (any route: CROWN-IBP, forward-linear, IBP)
//!    shrink-only with the zonotope boxes.
//!
//! That map is what the DAG alpha state is initialised from, what the initial
//! CROWN bound reuses on the capped typed route, and what the alpha-margin
//! lane and the published root map read.
//!
//! # Soundness
//!
//! * Every zonotope transformer is certified (binary64, explicit `gamma`
//!   charges, outward endpoints, explicit symbol IDs; `vit_ops.rs`,
//!   `dag.rs`). Each node box is the zonotope's concretisation ∩ ny's IBP box
//!   over the SAME input box; a crossed intersection REFUSES the whole pass.
//! * A runtime SELF-CHECK runs before anything is published: at three points
//!   of the box (the centre and two vertices) the compiled program's own point
//!   semantics must agree with ny's pointwise forward at EVERY node (a
//!   mis-modelled layer refuses), and every sampled value must lie in its
//!   published box (a transformer that encloses the wrong function refuses).
//! * The merge is shrink-only per element; a disjoint element (which only a
//!   defect can produce) keeps the CURRENT value, never the zonotope's, and a
//!   shape mismatch keeps the whole current entry.
//! * Any refusal (unsupported op, NaN/Inf, the byte budget, the deadline, the
//!   self-check) publishes NOTHING: the reference collector runs exactly its
//!   historical route.
//! * Dark (`bab.vit_zono_root` absent/false, every shipped preset): no plan is
//!   stamped, the collector's wrapper takes its first branch, and the
//!   historical path runs byte-for-byte with no output.
//!
//! # Observables (fixed schema)
//!
//! * `[vit-zono-root] armed=1 admitted=<0|1> reason=<token> nodes=… census=…
//!   symbols=… …` — once per distinct (graph, box) computation.
//! * `[vit-zono-root] stage=<route> nodes=… nodes_merged=… elems_tightened=…
//!   disjoint_kept=… shape_skipped=… cache=<hit|miss>` — at every reference
//!   collection (first 16, then powers of two).
//! * The CLI prints `[vit-zono-root] armed=1 stage=stamp …` once per armed run.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_levers::Source;
use ny_tensor::BoundedTensor;

use super::concretize::to_f32_outward;
use super::dag::{DagForwardOptions, DagForwardStats, DagNodeView, DagOpSet, DagProgram};
use super::fwd_root::admitted_byte_budget;
use super::ZonoRefusal;
use crate::beta_crown::bab_cuts::CutFoldScope;
use crate::beta_crown::config::BetaCrownConfig;
use crate::network::NETWORK_INPUT;
use crate::GraphNetwork;

/// Default wall cap of one root map computation (`bab.vit_zono_root_max_secs`).
pub const VIT_ZONO_ROOT_DEFAULT_MAX_SECS: f64 = 30.0;

/// Distinct (graph, box) computations one graph identity may pay for. The
/// root map is a ROOT object: a caller that asks for many different boxes
/// (an input-split lane) gets its historical route after this many.
pub const VIT_ZONO_ROOT_MAX_RUNS_PER_GRAPH: usize = 4;

/// Exact-key cache slots (process-wide). A root map is a handful of `f32`
/// node boxes (~MBs on vit), so eight slots cost little; they also keep the
/// unit tests, which share this process-wide cache across threads, from
/// evicting one another's entries mid-assertion.
const CACHE_SLOTS: usize = 8;

/// Resolved `bab.vit_zono_root` for one run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VitZonoRootPlan {
    /// Configured live-state ceiling (`bab.vit_zono_root_max_mb`).
    pub max_bytes: u64,
    /// Wall cap of one computation (`bab.vit_zono_root_max_secs`).
    pub max_secs: f64,
    /// Where the arming came from; always [`Source::Config`] (no env reader).
    pub source: Source,
}

/// The plan for `config`, or `None` when the key is dark (the default).
#[must_use]
pub fn vit_zono_root_plan(config: &BetaCrownConfig) -> Option<VitZonoRootPlan> {
    if !config.vit_zono_root {
        return None;
    }
    let max_secs = f64::from(config.vit_zono_root_max_secs);
    Some(VitZonoRootPlan {
        max_bytes: u64::from(config.vit_zono_root_max_mb.max(1)) << 20,
        max_secs: if max_secs.is_finite() && max_secs > 0.0 {
            max_secs
        } else {
            VIT_ZONO_ROOT_DEFAULT_MAX_SECS
        },
        source: Source::Config,
    })
}

/// A completed, self-checked root map.
#[derive(Debug, Clone)]
pub struct VitZonoRootMap {
    /// Certified per-node boxes (zonotope ∩ IBP, `f32` outward, ny's shapes)
    /// for every node the program computed.
    pub bounds: HashMap<String, BoundedTensor>,
    /// ny's COMPLETE IBP map over the same box with every [`Self::bounds`]
    /// entry substituted (each already inside its IBP box): the CROWN-IBP
    /// collector's baseline.
    pub baseline: HashMap<String, BoundedTensor>,
    /// Compiled op census (`kind:count,...`).
    pub census: String,
    /// Forward-pass statistics.
    pub stats: DagForwardStats,
    /// `f32` elements strictly tighter than ny's IBP box.
    pub tightened_elems: usize,
    /// Node-elements the runtime self-check compared.
    pub self_check_elems: usize,
    /// Admitted live-state budget.
    pub budget_bytes: u64,
    /// Wall time, compilation, IBP pass and self-check included.
    pub elapsed: Duration,
}

/// A declined computation (fail closed: nothing is published).
#[derive(Debug, Clone)]
pub struct VitZonoDecline {
    /// Why.
    pub reason: ZonoRefusal,
    /// Compiled op census, or `-` when compilation itself refused.
    pub census: String,
    /// Admitted live-state budget.
    pub budget_bytes: u64,
    /// Wall time until the refusal.
    pub elapsed: Duration,
    /// True only for a [`ZonoRefusal::Deadline`] that the CALLER's deadline
    /// imposed (it was tighter than the plan's own `max_secs` cap). Such a
    /// refusal depends on how much budget that caller had left, not on the
    /// (graph, box, plan), so it is never cached: a later collection with
    /// more time may retry. A deadline refusal under the plan's OWN cap is a
    /// property of (graph, box, plan) and is cached like every other
    /// refusal — retrying it would only burn another `max_secs` of the row's
    /// budget per attempt (up to the per-graph run cap) for the same answer.
    pub caller_budget_bound: bool,
}

/// Per-node `f64` boxes the observer records (node -> (shape, lo, hi)).
type NodeBoxes = HashMap<String, (Vec<usize>, Vec<f64>, Vec<f64>)>;

/// Compute the root map for `input` WITHOUT the cache (tests, the real-ONNX
/// gate and [`vit_zono_root_map`] call this).
///
/// Wall: `min(deadline, now + plan.max_secs)`. Memory: the configured ceiling
/// clamped to half the process headroom (`admitted_byte_budget`; competition
/// runs without a cgroup, so this is the only RSS guard).
pub fn compute_vit_zono_root_map(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    plan: &VitZonoRootPlan,
    deadline: Option<Instant>,
) -> Result<VitZonoRootMap, VitZonoDecline> {
    let started = Instant::now();
    let budget_bytes = admitted_byte_budget(plan.max_bytes);
    let window = Duration::try_from_secs_f64(plan.max_secs)
        .unwrap_or(Duration::from_secs(VIT_ZONO_ROOT_DEFAULT_MAX_SECS as u64));
    let cap = started.checked_add(window).unwrap_or(started);
    let local_deadline = Some(deadline.map_or(cap, |outer| outer.min(cap)));
    // Whose deadline binds: the caller's (strictly tighter than the cap) or
    // the plan's own wall cap. Only the former makes a Deadline refusal
    // budget-dependent (see `VitZonoDecline::caller_budget_bound`).
    let caller_deadline_binds = deadline.is_some_and(|outer| outer < cap);
    let decline = |reason: ZonoRefusal, census: &str| VitZonoDecline {
        caller_budget_bound: caller_deadline_binds && reason == ZonoRefusal::Deadline,
        reason,
        census: census.to_string(),
        budget_bytes,
        elapsed: started.elapsed(),
    };
    let (program, ibp) =
        DagProgram::compile_with_ibp_opset(graph, input, local_deadline, DagOpSet::Vit)
            .map_err(|reason| decline(reason, "-"))?;
    let census = program.census();
    let opts = DagForwardOptions {
        intersect_ibp: true,
        byte_budget: budget_bytes,
    };
    let mut boxes: NodeBoxes = HashMap::new();
    let mut observer = |view: &DagNodeView<'_>| {
        if view.node != NETWORK_INPUT {
            boxes.insert(
                view.node.to_string(),
                (view.shape.to_vec(), view.lo.to_vec(), view.hi.to_vec()),
            );
        }
    };
    let outcome = program
        .forward(input, &opts, &ibp, local_deadline, Some(&mut observer))
        .map_err(|reason| decline(reason, &census))?;
    let self_check_elems = runtime_self_check(graph, &program, input, &boxes, local_deadline)
        .map_err(|reason| decline(reason, &census))?;

    let mut bounds = HashMap::with_capacity(boxes.len());
    let mut tightened_elems = 0usize;
    for (node, (_shape, lo, hi)) in &boxes {
        let Some(entry) = ibp.get(node) else {
            continue;
        };
        if entry.len() != lo.len() {
            continue;
        }
        let (lo32, hi32) = to_f32_outward(lo, hi);
        let mut lower: Vec<f32> = Vec::with_capacity(lo32.len());
        let mut upper: Vec<f32> = Vec::with_capacity(hi32.len());
        for (((&zl, &zu), &il), &iu) in lo32
            .iter()
            .zip(&hi32)
            .zip(entry.lower().iter())
            .zip(entry.upper().iter())
        {
            // Shrink-only against the IBP entry (the f64 box is already
            // inside it; the outward f32 store stays inside it too, this only
            // makes that a local fact).
            let (nl, nu) = (zl.max(il), zu.min(iu));
            if nl > nu || nl.is_nan() || nu.is_nan() {
                lower.push(il);
                upper.push(iu);
            } else {
                if nl > il || nu < iu {
                    tightened_elems += 1;
                }
                lower.push(nl);
                upper.push(nu);
            }
        }
        let shape_ibp = entry.shape().to_vec();
        let lower = ArrayD::from_shape_vec(IxDyn(&shape_ibp), lower).map_err(|e| {
            decline(
                ZonoRefusal::Internal(format!("publish `{node}`: {e}")),
                &census,
            )
        })?;
        let upper = ArrayD::from_shape_vec(IxDyn(&shape_ibp), upper).map_err(|e| {
            decline(
                ZonoRefusal::Internal(format!("publish `{node}`: {e}")),
                &census,
            )
        })?;
        let published = BoundedTensor::new_allow_infinite(lower, upper).map_err(|e| {
            decline(
                ZonoRefusal::Internal(format!("publish `{node}`: {e}")),
                &census,
            )
        })?;
        bounds.insert(node.clone(), published);
    }
    if bounds.is_empty() {
        return Err(decline(
            ZonoRefusal::Internal("no node box to publish".to_string()),
            &census,
        ));
    }
    let mut baseline = ibp;
    for (node, entry) in &bounds {
        baseline.insert(node.clone(), entry.clone());
    }
    Ok(VitZonoRootMap {
        bounds,
        baseline,
        census,
        stats: outcome.stats,
        tightened_elems,
        self_check_elems,
        budget_bytes,
        elapsed: started.elapsed(),
    })
}

/// SplitMix64 (deterministic sample vertices).
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The self-check's sample points: the box centre and two vertices, all
/// exact `f32` values inside the box.
fn self_check_points(input: &BoundedTensor) -> Vec<Vec<f32>> {
    let lower: Vec<f32> = input.lower().iter().copied().collect();
    let upper: Vec<f32> = input.upper().iter().copied().collect();
    let mut points = Vec::with_capacity(3);
    points.push(
        lower
            .iter()
            .zip(&upper)
            .map(|(&l, &u)| {
                // The exact f64 midpoint of two f32 values, rounded to the
                // nearest f32: never outside [l, u] (both are f32 values).
                #[allow(clippy::cast_possible_truncation)]
                let mid = (0.5 * f64::from(l) + 0.5 * f64::from(u)) as f32;
                mid.clamp(l, u)
            })
            .collect(),
    );
    for seed in [0x5EED_0001u64, 0x5EED_0002] {
        points.push(
            lower
                .iter()
                .zip(&upper)
                .enumerate()
                .map(|(i, (&l, &u))| {
                    if splitmix64(seed ^ (i as u64)) & 1 == 0 {
                        l
                    } else {
                        u
                    }
                })
                .collect(),
        );
    }
    points
}

/// The runtime self-check (see the module docs). Returns the number of
/// node-elements compared; zero compared refuses (a check that looked at
/// nothing is void, not a pass).
fn runtime_self_check(
    graph: &GraphNetwork,
    program: &DagProgram,
    input: &BoundedTensor,
    boxes: &NodeBoxes,
    deadline: Option<Instant>,
) -> Result<usize, ZonoRefusal> {
    let mut compared = 0usize;
    for point in self_check_points(input) {
        super::kernels::check_deadline(deadline)?;
        let array = ArrayD::from_shape_vec(IxDyn(input.shape()), point.clone())
            .map_err(|e| ZonoRefusal::Internal(format!("self-check point: {e}")))?;
        let probe = BoundedTensor::concrete(array)
            .map_err(|e| ZonoRefusal::Internal(format!("self-check point: {e}")))?;
        let reference = graph
            .collect_node_activations_pointwise(&probe, None)
            .map_err(|e| ZonoRefusal::Internal(format!("self-check: ny pointwise forward: {e}")))?;
        let exact: Vec<f64> = point.iter().map(|&v| f64::from(v)).collect();
        let mut failure: Option<String> = None;
        program.evaluate_points(&exact, 1, |_, name, values| {
            if failure.is_some() || name == NETWORK_INPUT {
                return;
            }
            if let Some(theirs) = reference.get(name) {
                if theirs.len() == values.len() {
                    let scale = theirs
                        .lower()
                        .iter()
                        .chain(theirs.upper().iter())
                        .fold(1.0f64, |acc, v| acc.max(f64::from(v.abs())));
                    for (j, ((&mine, &l), &u)) in values
                        .iter()
                        .zip(theirs.lower().iter())
                        .zip(theirs.upper().iter())
                        .enumerate()
                    {
                        let (l, u) = (f64::from(l), f64::from(u));
                        let tolerance = 1e-4 * scale + (u - l).abs();
                        let distance = if mine < l {
                            l - mine
                        } else if mine > u {
                            mine - u
                        } else {
                            0.0
                        };
                        if !mine.is_finite() || distance > tolerance {
                            failure = Some(format!(
                                "semantics: node `{name}`[{j}] program {mine:e} vs ny [{l:e}, {u:e}]"
                            ));
                            return;
                        }
                        compared += 1;
                    }
                } else {
                    failure = Some(format!(
                        "semantics: node `{name}` has {} program values vs {} in ny",
                        values.len(),
                        theirs.len()
                    ));
                    return;
                }
            }
            if let Some((_, lo, hi)) = boxes.get(name) {
                if lo.len() != values.len() {
                    failure = Some(format!("containment: node `{name}` length"));
                    return;
                }
                for (j, ((&v, &l), &h)) in values.iter().zip(lo).zip(hi).enumerate() {
                    let slack = 1e-9 * (1.0 + v.abs());
                    if !(v >= l - slack && v <= h + slack) {
                        failure = Some(format!(
                            "containment: node `{name}`[{j}] value {v:e} outside [{l:e}, {h:e}]"
                        ));
                        return;
                    }
                }
            }
        })?;
        if let Some(why) = failure {
            return Err(ZonoRefusal::Internal(format!("self-check failed: {why}")));
        }
    }
    if compared == 0 {
        return Err(ZonoRefusal::Internal(
            "self-check compared nothing (VOID)".to_string(),
        ));
    }
    Ok(compared)
}

// ---------------------------------------------------------------------------
// The exact-box cache and the lane entry point
// ---------------------------------------------------------------------------

/// Exact cache key: the graph's semantic identity token (fresh on every
/// structural mutation), its node count, the input box's shape and bit
/// patterns, and the plan's two ceilings (a cached refusal under the plan's
/// own wall cap is a property of that cap). A map computed for one box is an
/// enclosure over that box ONLY, with that box's node shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheKey {
    scope: CutFoldScope,
    nodes: usize,
    shape: Vec<usize>,
    lower: Vec<u32>,
    upper: Vec<u32>,
    max_bytes: u64,
    max_secs_bits: u64,
}

struct CacheEntry {
    key: CacheKey,
    outcome: Result<Arc<VitZonoRootMap>, VitZonoDecline>,
}

static CACHE: Mutex<Vec<CacheEntry>> = Mutex::new(Vec::new());
static RUNS: Mutex<Vec<(CutFoldScope, usize)>> = Mutex::new(Vec::new());

fn cache_key(graph: &GraphNetwork, input: &BoundedTensor, plan: &VitZonoRootPlan) -> CacheKey {
    CacheKey {
        scope: graph.cut_fold_scope,
        nodes: graph.nodes.len(),
        shape: input.shape().to_vec(),
        lower: input.lower().iter().map(|v| v.to_bits()).collect(),
        upper: input.upper().iter().map(|v| v.to_bits()).collect(),
        max_bytes: plan.max_bytes,
        max_secs_bits: plan.max_secs.to_bits(),
    }
}

/// Whether a computation for this graph identity and box is cached (test
/// probe for the dark-default pin).
#[cfg(test)]
pub(crate) fn cached_for(graph: &GraphNetwork, input: &BoundedTensor) -> bool {
    let lower: Vec<u32> = input.lower().iter().map(|v| v.to_bits()).collect();
    let upper: Vec<u32> = input.upper().iter().map(|v| v.to_bits()).collect();
    CACHE.lock().map_or(false, |cache| {
        cache.iter().any(|entry| {
            entry.key.scope == graph.cut_fold_scope
                && entry.key.lower == lower
                && entry.key.upper == upper
        })
    })
}

/// The lane entry point: the root map for `(graph, input)`, served from the
/// exact-key cache when present, computed (and cached, refusals included)
/// otherwise. Returns the map (or `None` on any decline) and whether it was a
/// cache hit. Prints the fixed-schema line once per computation.
pub fn vit_zono_root_map(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    plan: &VitZonoRootPlan,
    deadline: Option<Instant>,
) -> (Option<Arc<VitZonoRootMap>>, bool) {
    let key = cache_key(graph, input, plan);
    if let Ok(cache) = CACHE.lock() {
        if let Some(entry) = cache.iter().find(|entry| entry.key == key) {
            return (entry.outcome.as_ref().ok().cloned(), true);
        }
    }
    let runs = RUNS.lock().ok().map_or(usize::MAX, |mut runs| {
        match runs.iter_mut().find(|(scope, _)| *scope == key.scope) {
            Some((_, count)) => {
                *count += 1;
                *count
            }
            None => {
                if runs.len() >= 64 {
                    runs.remove(0);
                }
                runs.push((key.scope, 1));
                1
            }
        }
    });
    if runs > VIT_ZONO_ROOT_MAX_RUNS_PER_GRAPH {
        static CAPPED: AtomicUsize = AtomicUsize::new(0);
        if should_print(&CAPPED) {
            eprintln!(
                "[vit-zono-root] armed=1 admitted=0 reason=run_cap runs={runs} \
                 cap={VIT_ZONO_ROOT_MAX_RUNS_PER_GRAPH} source=config key=bab.vit_zono_root"
            );
        }
        return (None, false);
    }
    let outcome = compute_vit_zono_root_map(graph, input, plan, deadline).map(Arc::new);
    emit_computation_line(&outcome, plan);
    let served = outcome.as_ref().ok().cloned();
    // A deadline refusal imposed by the CALLER's deadline depends on that
    // caller's remaining budget, not on the (graph, box, plan): it is not
    // cached, so a later collection with more time retries (bounded by the
    // per-graph run cap). Every other outcome — a deadline refusal under the
    // plan's own `max_secs` cap included — is a property of the (graph, box,
    // plan) and is cached, refusals included: re-running a cap-bound pass
    // would burn another `max_secs` of the row's budget for the same answer.
    let budget_dependent = matches!(&outcome, Err(decline) if decline.caller_budget_bound);
    if !budget_dependent {
        if let Ok(mut cache) = CACHE.lock() {
            if cache.len() >= CACHE_SLOTS {
                cache.remove(0);
            }
            cache.push(CacheEntry { key, outcome });
        }
    }
    (served, false)
}

fn should_print(counter: &AtomicUsize) -> bool {
    let n = counter.fetch_add(1, Ordering::Relaxed).saturating_add(1);
    n <= 16 || n.is_power_of_two()
}

/// Process VmHWM in MiB (Linux), for the observable.
fn peak_rss_mb() -> Option<u64> {
    super::cgan::self_rss_mb().1
}

fn emit_computation_line(
    outcome: &Result<Arc<VitZonoRootMap>, VitZonoDecline>,
    plan: &VitZonoRootPlan,
) {
    match outcome {
        Ok(map) => {
            let stats = &map.stats;
            eprintln!(
                "[vit-zono-root] armed=1 admitted=1 reason=none nodes={} census={} symbols={} \
                 input_symbols={} fresh_relu={} fresh_bilinear={} fresh_mul={} fresh_band={} \
                 fresh_softmax={} gens_peak={} live_mb_peak={} max_remainder={:e} \
                 ibp_tightened={} tightened_elems={} self_check=pass self_check_elems={} \
                 secs={:.3} budget_mb={} max_secs={} peak_rss_mb={} source=config \
                 key=bab.vit_zono_root",
                map.bounds.len(),
                map.census,
                stats.symbols,
                stats.input_symbols,
                stats.fresh_relu,
                stats.fresh_bilinear,
                stats.fresh_mul,
                stats.fresh_band,
                stats.fresh_softmax,
                stats.gens_peak,
                stats.live_bytes_peak >> 20,
                stats.max_remainder,
                stats.ibp_tightened,
                map.tightened_elems,
                map.self_check_elems,
                map.elapsed.as_secs_f64(),
                map.budget_bytes >> 20,
                plan.max_secs,
                peak_rss_mb().map_or_else(|| "na".to_string(), |mb| mb.to_string()),
            );
        }
        Err(decline) => {
            let detail: String = decline
                .reason
                .to_string()
                .chars()
                .map(|c| if c.is_whitespace() { '_' } else { c })
                .take(240)
                .collect();
            eprintln!(
                "[vit-zono-root] armed=1 admitted=0 reason={} detail={detail} census={} \
                 secs={:.3} budget_mb={} max_secs={} source=config key=bab.vit_zono_root",
                decline.reason.token(),
                decline.census,
                decline.elapsed.as_secs_f64(),
                decline.budget_bytes >> 20,
                plan.max_secs,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Shrink-only merge into a reference map
// ---------------------------------------------------------------------------

/// What one merge did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VitZonoMergeStats {
    /// Entries of the reference map.
    pub nodes: usize,
    /// Entries a zonotope box was merged into.
    pub nodes_merged: usize,
    /// Elements that strictly shrank.
    pub elems_tightened: usize,
    /// Elements whose intersection was empty (kept CURRENT).
    pub disjoint_kept: usize,
    /// Entries skipped for a shape mismatch (kept CURRENT).
    pub shape_skipped: usize,
}

/// Intersect `map` shrink-only with `zono` (entries `map` already has only).
///
/// SOUND: both are certified enclosures of the same node over the same input
/// box, so their elementwise intersection is one too. A disjoint element can
/// only come from a defect in one of them, and nothing says which, so it
/// keeps the CURRENT value (never the zonotope's, never a collapse); a shape
/// mismatch keeps the whole current entry. Nothing is ever widened.
pub fn merge_shrink_only(
    map: &mut HashMap<String, BoundedTensor>,
    zono: &HashMap<String, BoundedTensor>,
) -> VitZonoMergeStats {
    let mut stats = VitZonoMergeStats {
        nodes: map.len(),
        ..VitZonoMergeStats::default()
    };
    for (name, candidate) in zono {
        let Some(current) = map.get(name) else {
            continue;
        };
        if current.shape() != candidate.shape() {
            stats.shape_skipped += 1;
            continue;
        }
        let mut lower = current.lower().clone();
        let mut upper = current.upper().clone();
        let mut tightened = 0usize;
        let mut disjoint = 0usize;
        for (((l, u), &zl), &zu) in lower
            .iter_mut()
            .zip(upper.iter_mut())
            .zip(candidate.lower().iter())
            .zip(candidate.upper().iter())
        {
            if zl.is_nan() || zu.is_nan() {
                continue;
            }
            let (nl, nu) = (l.max(zl), u.min(zu));
            if nl > nu {
                disjoint += 1;
                continue;
            }
            if nl > *l || nu < *u {
                tightened += 1;
            }
            *l = nl;
            *u = nu;
        }
        stats.disjoint_kept += disjoint;
        if tightened == 0 {
            continue;
        }
        if let Ok(merged) = BoundedTensor::new_allow_infinite(lower, upper) {
            stats.nodes_merged += 1;
            stats.elems_tightened += tightened;
            map.insert(name.clone(), merged);
        }
    }
    stats
}

/// `[vit-zono-root] stage=…` for one consumption (rate-limited per process).
pub fn emit_stage(stage: &str, route: &str, stats: &VitZonoMergeStats, cache_hit: bool) {
    static EVENTS: AtomicUsize = AtomicUsize::new(0);
    if !should_print(&EVENTS) {
        return;
    }
    eprintln!(
        "[vit-zono-root] stage={stage} route={route} nodes={} nodes_merged={} \
         elems_tightened={} disjoint_kept={} shape_skipped={} cache={}",
        stats.nodes,
        stats.nodes_merged,
        stats.elems_tightened,
        stats.disjoint_kept,
        stats.shape_skipped,
        if cache_hit { "hit" } else { "miss" },
    );
}

/// `[vit-zono-root] stage=unavailable …` when an armed collection proceeds
/// without a map (declined, capped or cached refusal).
pub fn emit_unavailable(route_hint: &str, cache_hit: bool) {
    static EVENTS: AtomicUsize = AtomicUsize::new(0);
    if !should_print(&EVENTS) {
        return;
    }
    eprintln!(
        "[vit-zono-root] stage=unavailable route={route_hint} cache={} (historical route, \
         nothing published)",
        if cache_hit { "hit" } else { "miss" },
    );
}
