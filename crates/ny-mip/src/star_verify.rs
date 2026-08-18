// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Structurally exact star-set candidate search: overapproximate first, refine as needed.
//!
//! This is the third piece of the star path. [`ny_tensor::zonotope::Star`] supplies the
//! structurally exact ReLU split and [`crate::star_lp`] supplies the predicate LP; this module is the
//! search that turns them into candidate evidence. "Exact" in this module means exact set
//! partitioning over the stored affine forms, not certified floating-point enclosure.
//!
//! ## Why not enumerate
//!
//! Blind enumeration is `2^unstable`, and a network like ACAS Xu has ~300 ReLUs — `2^300`
//! is not a plan. `nnenum`, the tool that actually wins these rows, does NOT enumerate
//! blindly: it over-approximates first and refines only where the over-approximation fails
//! to decide. This driver has the same shape:
//!
//! 1. Propagate the stored affine forms through affine layers (structurally exact — affine
//!    maps never touch α space).
//! 2. At a ReLU, split ONLY unstable neurons, one at a time, cheapest-first.
//! 3. Prune branches the LP proves empty.
//! 4. At full depth the internal star model is exact, so the LP's answer is decisive for
//!    that model in both directions.
//!
//! ## The property that makes this work
//!
//! Neither an affine map nor a structural ReLU split introduces a new error symbol. So α
//! always indexes the ORIGINAL input box of the internal f32 star model, at every depth. A
//! predicate `A·α ≤ b` therefore describes that model's input points directly, which lets a
//! leaf produce candidate evidence rather than merely a relaxation of the internal model.
//!
//! ## Verdict discipline
//!
//! [`StarVerdict::CandidateCounterexampleExists`] deliberately carries NO witness. At a leaf
//! the structural star model contains a violating point, but VNN-COMP requires a *validated*
//! counterexample, and NY's moat forbids claiming `sat` without one. Callers must materialise
//! and check a witness before emitting `sat`; extracting the LP's argmin is tracked as
//! #star-witness.
//!
//! The `Candidate` prefix is also load-bearing in the UNSAT direction: affine coefficients
//! are currently stored and transformed in `f32` without a certified roundoff enclosure.
//! The search is structurally exact over its internal star model, but neither positive
//! candidate may authorize a concrete-model verdict until exact replay or enclosing
//! arithmetic discharges that obligation. Budget exhaustion always returns
//! [`StarVerdict::Unknown`].

use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use ny_tensor::zonotope::{Star, StarReluSplit};

use crate::star_dual::{dual_certifies_empty, dual_coordinate_bounds};
use crate::star_lp::{star_predicate_bounds, StarLpReport, StarLpRequest, StarLpSession};
use crate::{MipError, Result};

/// One layer of a ReLU feed-forward network.
#[derive(Debug, Clone)]
// `Gemm` keeps its weight inline. A network is one `Vec<StarLayer>` built once, one entry
// per layer (tens of them), so the padding the dataless `Relu` variant pays is well under
// a KB; boxing would instead add a pointer chase to the `layers[work.layer]` match that
// runs on every star pop, and to the tail-bound sweep under it.
#[allow(clippy::large_enum_variant)]
pub enum StarLayer {
    /// `y = W·x + b`.
    Gemm {
        /// Weight matrix `(out, in)`.
        weight: Array2<f32>,
        /// Optional bias `(out,)`.
        bias: Option<Array1<f32>>,
    },
    /// Element-wise `ReLU`.
    Relu,
}

/// A conjunctive specification: EVERY row must hold for the property to be verified.
///
/// Row `(c, t)` asserts `c·y > t` over the network output `y`.
#[derive(Debug, Clone)]
pub struct StarSpec {
    /// `(coefficients, threshold)` pairs, all of which must hold.
    pub rows: Vec<(Vec<f64>, f64)>,
}

/// One half-space in a candidate unsafe region.
///
/// The row is `coefficients·y <= threshold` when `strict == false`, and
/// `coefficients·y < threshold` when `strict == true`.
#[derive(Debug, Clone)]
pub struct StarUnsafeRow {
    /// Output coefficients, in flattened output order.
    pub coefficients: Vec<f64>,
    /// Right-hand side of the unsafe comparison.
    pub threshold: f64,
    /// Whether the comparison is strict (`<` rather than `<=`).
    pub strict: bool,
}

/// A CONJUNCTION of output half-spaces describing one candidate unsafe region.
///
/// This is deliberately separate from [`StarSpec`]. A VNN-LIB conjunction such as
/// ACAS Xu property 2 is unsafe only when *all* of its atoms hold at one point; its safe
/// complement is therefore a disjunction. Treating those complement rows as a
/// [`StarSpec`] would require all of them to hold and would prove the wrong property.
#[derive(Debug, Clone)]
pub struct StarUnsafeConjunction {
    /// Every row must hold simultaneously for the output to be in this unsafe region.
    pub rows: Vec<StarUnsafeRow>,
}

/// Verdict from [`star_search_unsafe_conjunction`].
///
/// Every positive-sounding variant contains `Candidate` in its name on purpose. The
/// current star algebra stores affine coefficients in `f32` without a certified
/// roundoff enclosure. These results are useful search evidence and performance
/// measurements, but MUST NOT authorize a VNN-COMP verdict. A production authority
/// needs either roundoff-enclosing star arithmetic or an independent certificate over
/// the concrete model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StarUnsafeCandidateVerdict {
    /// Every searched leaf made the CLOSED unsafe relaxation infeasible.
    CandidateUnsafeClosureEmpty,
    /// A completed leaf query did not prove the CLOSED unsafe relaxation empty.
    /// This is deliberately not named `Feasible`: the current LP report carries rigorous
    /// bounds and an infeasibility proof bit, but no primal-feasibility certificate or
    /// validated witness. For a strict row, even a genuinely feasible closure may consist
    /// only of boundary points.
    CandidateUnsafeClosureNotProvedEmpty,
    /// Budget, depth, or solver limitations prevented a candidate conclusion.
    Unknown(String),
}

/// Search limits. Exhausting any of them yields [`StarVerdict::Unknown`], never a claim.
#[derive(Debug, Clone)]
pub struct StarBudget {
    /// Maximum stars popped from the queue.
    pub max_stars: usize,
    /// Maximum split depth along any single branch.
    pub max_depth: usize,
    /// Wall-clock deadline for the whole search.
    pub deadline: Instant,
    /// Per-LP time limit (exact solver, consulted only off the hot path).
    pub lp_time_limit: Duration,
    /// Projected dual-ascent steps for the solver-free bound. More is tighter, never
    /// less sound; 0 degrades to the plain interval bound.
    pub dual_iters: usize,
    /// Below this many unstable neurons at a node, branch EXACTLY (neuron split) instead of
    /// bisecting, even when `prefer_input_split` is set.
    ///
    /// Bisection tightens the input-symbol part of the bound but NOT `TailForm`'s
    /// relaxation radius, which accumulates one `μ` per unstable neuron and shrinks only when
    /// a neuron becomes stable. Deep in the tree that radius is what stalls the discharge
    /// rate. Once only a handful of neurons remain unstable, splitting them exactly removes
    /// the radius outright — the hybrid nnenum uses. `0` disables the switch.
    pub exact_below_unstable: usize,

    /// Branch by bisecting the INPUT box rather than splitting a neuron.
    ///
    /// The right choice whenever the input dimension is far below the unstable-neuron
    /// count, which is the usual case for control-style networks (ACAS Xu: 5 inputs,
    /// ~300 neurons).
    pub prefer_input_split: bool,
}

impl StarBudget {
    /// A budget with sane defaults for small networks.
    #[must_use]
    pub fn new(max_stars: usize, max_depth: usize, deadline: Instant) -> Self {
        Self {
            max_stars,
            max_depth,
            deadline,
            lp_time_limit: Duration::from_secs(2),
            dual_iters: 32,
            prefer_input_split: false,
            exact_below_unstable: 0,
        }
    }
}

/// Outcome of the search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StarVerdict {
    /// Every reachable output in the internal star model satisfies every spec row. This is
    /// NOT concrete-model UNSAT authority; see the verdict discipline in the module docs.
    CandidateVerified,
    /// A leaf star contains a violating point in the internal model. NOT a `sat` claim on
    /// its own; see the verdict discipline in the module docs.
    CandidateCounterexampleExists,
    /// Budget, depth, or star cap exhausted; nothing is claimed.
    Unknown(String),
}

/// Diagnostics for a completed search. Counts only — nothing here feeds a verdict.
#[derive(Debug, Clone, Default)]
pub struct StarStats {
    /// Stars popped from the queue.
    pub popped: usize,
    /// Branches the LP proved empty and dropped.
    pub pruned_infeasible: usize,
    /// Leaves discharged as safe.
    pub leaves_verified: usize,
    /// Exact ReLU splits performed.
    pub splits: usize,
    /// Neurons the LP reclassified as stable that the interval test called unstable.
    pub lp_reclaimed_stable: usize,
    /// Nodes discharged by the interval over-approximation without any splitting.
    pub discharged_by_overapprox: usize,
    /// Exact-LP calls made only where they could avert a split.
    pub exact_lp_calls: usize,
    /// Nanoseconds inside the exact LP.
    pub ns_exact_lp: u128,
    /// Nanoseconds inside the solver-free dual classification.
    pub ns_dual: u128,
    /// Nanoseconds inside the interval tail walk.
    pub ns_tail: u128,
    /// Nanoseconds inside star algebra (gemm / relu_split / clones).
    pub ns_star: u128,
    /// Times the NS expression bound declined and the column path was used.
    pub ns_declined: usize,
    /// Queries answered by the untrusted-simplex / trusted-verifier path.
    pub verified_float_hits: usize,
    /// Input-box bisections performed instead of neuron splits.
    pub input_bisections: usize,
    /// Nodes whose tail bound came from star propagation rather than intervals.
    pub star_tail_hits: usize,
    /// Per-depth `(popped, discharged)`, so the discharge rate can be read as a function of
    /// how far the box has been refined. A rate that does NOT rise with depth means
    /// bisection is not buying tightness, which is a different problem from being slow.
    pub depth_histogram: Vec<(usize, usize)>,
}

struct Work {
    star: Star,
    /// The INPUT-space star this node stands for, kept in lockstep with `star`.
    ///
    /// `star` is mid-network: past the first `Gemm` its value shape is a hidden
    /// width, not the input width. Leaf refinement restarts at layer 0 and therefore
    /// needs the input box, not the network's own output. Cheap to carry — it lives in
    /// input dimensions (5 for ACAS Xu), not hidden ones.
    input: Star,
    /// Index into `layers` at which to resume.
    layer: usize,
    /// Next coordinate to examine within the current `Relu` layer.
    coord: usize,
    depth: usize,
}

#[derive(Clone, Copy)]
enum SearchProperty<'a> {
    SafeConjunction(&'a StarSpec),
    UnsafeConjunction(&'a StarUnsafeConjunction),
}

impl SearchProperty<'_> {
    fn is_empty(self) -> bool {
        match self {
            Self::SafeConjunction(spec) => spec.rows.is_empty(),
            Self::UnsafeConjunction(spec) => spec.rows.is_empty(),
        }
    }
}

/// Build the LP request for a set of target affine forms over a star.
fn lp_request(star: &Star, targets: Vec<(f64, Vec<f64>)>) -> Result<StarLpRequest> {
    let (a, b) = star.constraints();
    let a_rows = (0..a.nrows())
        .map(|r| (0..a.ncols()).map(|c| f64::from(a[[r, c]])).collect())
        .collect();
    Ok(StarLpRequest {
        alpha_dim: star.alpha_dim(),
        a_rows,
        b: b.iter().map(|v| f64::from(*v)).collect(),
        targets,
    })
}

/// Resolve a neuron from a sound `(lower, upper)` on its pre-activation, or `None` when the
/// bound genuinely straddles zero and a split is unavoidable.
fn classify(lo: f64, hi: f64, active: &Star, idx: usize) -> Option<Star> {
    if !lo.is_finite() || !hi.is_finite() {
        None
    } else if lo >= 0.0 {
        Some(active.clone())
    } else if hi <= 0.0 {
        // PROVABLY inactive: zero the coordinate WITHOUT recording `g_i·α ≤ -c_i`. That row
        // is what `relu_split`'s Split arm uses to ENFORCE an assumed sign; once the bound
        // has proven the sign it is implied, and keeping it only grows the predicate that
        // every later LP and bound must carry. Measured on ACAS Xu prop_2: the redundant
        // rows left a leaf with 178 predicate rows over 5 α variables.
        active.zero_coordinate(idx).ok()
    } else {
        None
    }
}

/// The star's predicate `(A, b)` widened to f64.
fn predicate_of(star: &Star) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (a, b) = star.constraints();
    let a_rows = (0..a.nrows())
        .map(|r| (0..a.ncols()).map(|c| f64::from(a[[r, c]])).collect())
        .collect();
    (a_rows, b.iter().map(|v| f64::from(*v)).collect())
}

/// Affine form of one coordinate, widened to f64 for the LP.
fn target_of(star: &Star, idx: usize) -> Result<(f64, Vec<f64>)> {
    let (c, g) = star
        .coordinate_form(idx)
        .map_err(|e| MipError::Encoding(format!("star_verify: coordinate_form: {e}")))?;
    Ok((f64::from(c), g.iter().map(|v| f64::from(*v)).collect()))
}

/// Search a conjunctive spec over a ReLU network by structurally exact star splitting.
///
/// Returns [`StarVerdict::CandidateVerified`] only when every reachable output in the
/// internal star model satisfies every row.
/// Any exhaustion returns [`StarVerdict::Unknown`]; no budget path can produce a claim.
///
/// # Errors
/// [`MipError`] on a malformed network/spec or an LP backend failure.
pub fn star_verify(
    layers: &[StarLayer],
    input: &Star,
    spec: &StarSpec,
    budget: &StarBudget,
) -> Result<(StarVerdict, StarStats)> {
    star_search(layers, input, SearchProperty::SafeConjunction(spec), budget)
}

/// Search one CONJUNCTIVE unsafe output region over a ReLU network.
///
/// The conjunction is handled JOINTLY at a leaf: its output half-spaces are composed
/// into the star's alpha space and one LP asks whether they can hold simultaneously.
/// This is the semantics needed by ACAS Xu property 2. Bounding every atom separately
/// and requiring every safe complement would solve a different, stronger property.
///
/// # Verdict authority
/// This API is deliberately candidate-only; see [`StarUnsafeCandidateVerdict`]. In
/// particular, `CandidateUnsafeClosureNotProvedEmpty` is neither feasibility evidence, a
/// witness, nor a `sat` verdict, and `CandidateUnsafeClosureEmpty` is not an `unsat` verdict
/// over the concrete ONNX model until the star arithmetic's floating-point enclosure
/// obligation is discharged.
///
/// # Errors
/// [`MipError`] on a malformed network/spec or an LP backend failure.
pub fn star_search_unsafe_conjunction(
    layers: &[StarLayer],
    input: &Star,
    spec: &StarUnsafeConjunction,
    budget: &StarBudget,
) -> Result<(StarUnsafeCandidateVerdict, StarStats)> {
    let (verdict, stats) = star_search(
        layers,
        input,
        SearchProperty::UnsafeConjunction(spec),
        budget,
    )?;
    let candidate = match verdict {
        StarVerdict::CandidateVerified => StarUnsafeCandidateVerdict::CandidateUnsafeClosureEmpty,
        StarVerdict::CandidateCounterexampleExists => {
            StarUnsafeCandidateVerdict::CandidateUnsafeClosureNotProvedEmpty
        }
        StarVerdict::Unknown(reason) => StarUnsafeCandidateVerdict::Unknown(reason),
    };
    Ok((candidate, stats))
}

fn star_search(
    layers: &[StarLayer],
    input: &Star,
    property: SearchProperty<'_>,
    budget: &StarBudget,
) -> Result<(StarVerdict, StarStats)> {
    star_search_with_clock(layers, input, property, budget, |_| Instant::now())
}

fn star_search_with_clock<N>(
    layers: &[StarLayer],
    input: &Star,
    property: SearchProperty<'_>,
    budget: &StarBudget,
    mut now: N,
) -> Result<(StarVerdict, StarStats)>
where
    N: FnMut(&'static str) -> Instant,
{
    if property.is_empty() {
        return Err(MipError::Encoding("star_verify: empty spec".into()));
    }
    let mut stats = StarStats::default();
    let mut queue = vec![Work {
        star: input.clone(),
        input: input.clone(),
        layer: 0,
        coord: 0,
        depth: 0,
    }];

    while let Some(mut work) = queue.pop() {
        if now("star work-item admission") >= budget.deadline {
            return Ok((StarVerdict::Unknown("deadline exceeded".into()), stats));
        }
        stats.popped += 1;
        if stats.depth_histogram.len() <= work.depth {
            stats.depth_histogram.resize(work.depth + 1, (0, 0));
        }
        stats.depth_histogram[work.depth].0 += 1;
        if stats.popped > budget.max_stars {
            return Ok((
                StarVerdict::Unknown(format!("star cap {} exceeded", budget.max_stars)),
                stats,
            ));
        }

        // Run affine layers until the next ReLU (or the end of the network).
        let mut advanced = true;
        while advanced && work.layer < layers.len() {
            advanced = false;
            if let StarLayer::Gemm { weight, bias } = &layers[work.layer] {
                work.star = work
                    .star
                    .gemm(weight, bias.as_ref())
                    .map_err(|e| MipError::Encoding(format!("star_verify: gemm: {e}")))?;
                work.layer += 1;
                work.coord = 0;
                advanced = true;
            }
        }

        // End of network: discharge the spec on an exact leaf of the INTERNAL model.
        if work.layer >= layers.len() {
            let leaf_outcome = discharge_leaf(&work.star, property, budget, &mut stats)?;
            // A leaf solver may return a valid proof just after its own last deadline
            // poll. It remains mathematically useful, but cannot make this bounded search
            // look completed inside a wall budget that has already expired.
            if now("star leaf-discharge publication") >= budget.deadline {
                return Ok((StarVerdict::Unknown("deadline exceeded".into()), stats));
            }
            match leaf_outcome {
                LeafOutcome::Safe => {
                    stats.leaves_verified += 1;
                    continue;
                }
                LeafOutcome::Empty => {
                    stats.pruned_infeasible += 1;
                    continue;
                }
                LeafOutcome::Violated => {
                    // A leaf that could not be discharged is not automatically a
                    // counterexample. Under input bisection the star's box may simply be too
                    // coarse for the LP to separate, and a finer box often settles it — so
                    // REFINE rather than conclude. Measured on ACAS Xu prop_2: without this
                    // the search returned "not proved empty" in 96ms after touching one
                    // stubborn leaf, having explored 425 nodes it never came back to.
                    //
                    // Only when refinement is unavailable (bisection off, no symbol left, or
                    // the depth cap reached) does this stand as the leaf's verdict.
                    if std::env::var("NY_STAR_LEAF_PROBE").as_deref() == Ok("1") {
                        eprintln!(
                            "TERMPROBE leaf-violated: prefer_split={} depth={}/{} widest_sym={:?}",
                            budget.prefer_input_split,
                            work.depth,
                            budget.max_depth,
                            work.input.widest_input_symbol()
                        );
                    }
                    if budget.prefer_input_split
                        && work.depth < budget.max_depth
                        && !few_unstable_remaining(&work.input, layers, 0, 0, budget)
                    {
                        // Refine the INPUT box and re-propagate from layer 0. This must
                        // split `work.input`, NOT `work.star`: at a leaf `work.star` is the
                        // network's OUTPUT, so bisecting it and re-queueing at layer 0 fed
                        // the network its own output — a shape error whenever the input and
                        // output widths differ, and silently the network applied twice when
                        // they happen to match (ACAS Xu is 5->5, which is why this survived).
                        // No free sensitivity here: the leaf path discharges before the
                        // tail is propagated, so this keeps the width heuristic.
                        if let Some(sym) = work.input.widest_input_symbol() {
                            let (lo_half, hi_half) =
                                work.input.split_input_symbol(sym).map_err(|e| {
                                    MipError::Encoding(format!("star_verify: leaf bisect: {e}"))
                                })?;
                            stats.input_bisections += 1;
                            for branch in [lo_half, hi_half] {
                                queue.push(Work {
                                    star: branch.clone(),
                                    input: branch,
                                    layer: 0,
                                    coord: 0,
                                    depth: work.depth + 1,
                                });
                            }
                            continue;
                        }
                    }
                    return Ok((StarVerdict::CandidateCounterexampleExists, stats));
                }
                LeafOutcome::Undecided => {
                    return Ok((
                        StarVerdict::Unknown(
                            "leaf star undecided: the LP could not separate the spec".into(),
                        ),
                        stats,
                    ));
                }
            }
        }

        // BOUND AND PRUNE. Before splitting anything, ask whether a cheap sound
        // over-approximation of the remaining network already proves the spec. Interval
        // arithmetic is a relaxation, so a positive answer is decisive and a negative one
        // costs only the walk. This is what stops the search enumerating a property that
        // was never in doubt.
        let t_tail = Instant::now();
        // Prefer the STAR tail (zonotope relaxation, keeps correlations) and fall back to
        // the interval tail if it cannot be built. The interval version alone pruned nothing
        // on ACAS Xu prop_2.
        // Fixed-size tail first: same relaxation, but the representation does not grow with
        // the neuron count, so it stays cheap on deep networks.
        let (discharged, _branch_sensitivity) =
            match tail_form_discharges(&work.star, layers, work.layer, work.coord, property) {
                Some((result, alpha)) => {
                    stats.star_tail_hits += 1;
                    (result, alpha)
                }
                None => {
                    let bounds = tail_interval_bounds(&work.star, layers, work.layer, work.coord)?;
                    (tail_discharges_property(&bounds, property), Vec::new())
                }
            };
        stats.ns_tail += t_tail.elapsed().as_nanos();
        if now("star tail-discharge publication") >= budget.deadline {
            return Ok((StarVerdict::Unknown("deadline exceeded".into()), stats));
        }
        if discharged {
            stats.discharged_by_overapprox += 1;
            stats.depth_histogram[work.depth].1 += 1;
            continue;
        }

        // At a ReLU layer: classify EVERY coordinate with ONE LP, then apply the stable
        // ones and split the first genuinely unstable one.
        debug_assert!(matches!(layers[work.layer], StarLayer::Relu));
        let width = work.star.zonotope().len();
        let mut star = work.star;
        let mut idx = work.coord;
        let mut split_here: Option<(Star, Star)> = None;
        let mut node_proved_empty = false;
        // Set when this node's whole subtree has been handed to freshly queued
        // children (input bisection), so the parent itself must NOT be advanced.
        let mut node_delegated = false;

        // One LP for the whole layer. Doing this per neuron dominated the runtime
        // (measured: 35 stars in 120s on a 100-ReLU net, ~0.5s of LP per neuron).
        let t_dual = Instant::now();
        let lp_layer = if star.num_constraints() > 0 {
            let (a_rows, b) = predicate_of(&star);
            if dual_certifies_empty(&a_rows, &b, budget.dual_iters) {
                if now("star dual-infeasibility publication") >= budget.deadline {
                    return Ok((StarVerdict::Unknown("deadline exceeded".into()), stats));
                }
                stats.pruned_infeasible += 1;
                continue;
            }
            let bounds = (idx..width)
                .map(|i| {
                    let (c, g) = target_of(&star, i)?;
                    Ok(dual_coordinate_bounds(
                        c,
                        &g,
                        &a_rows,
                        &b,
                        budget.dual_iters,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            if now("star dual-bounds publication") >= budget.deadline {
                return Ok((StarVerdict::Unknown("deadline exceeded".into()), stats));
            }
            Some(bounds)
        } else {
            None
        };
        stats.ns_dual += t_dual.elapsed().as_nanos();
        let layer_base = idx;
        // Built lazily: most nodes never reach the exact solver at all.
        let mut node_session: Option<StarLpSession> = None;

        while idx < width {
            let t_star = Instant::now();
            let split_out = star
                .relu_split(idx)
                .map_err(|e| MipError::Encoding(format!("star_verify: relu_split: {e}")))?;
            stats.ns_star += t_star.elapsed().as_nanos();
            match split_out {
                StarReluSplit::Active(s) | StarReluSplit::Inactive(s) => {
                    star = s;
                    idx += 1;
                }
                StarReluSplit::Split { inactive, active } => {
                    // The interval test says unstable. The predicate-aware range is a
                    // subset, so the LP often proves stability the box cannot. Bounds are
                    // outward-rounded, so a "stable" answer here is genuinely stable.
                    // Stage 1 — the solver-free dual. Cheap, sound, and it resolves the
                    // easy neurons without touching the solver.
                    let resolved = lp_layer.as_ref().and_then(|bounds| {
                        let (lo, hi) = *bounds.get(idx - layer_base)?;
                        classify(lo, hi, &star, idx)
                    });
                    // Stage 2 — the dual could not decide, and the only alternative is a
                    // SPLIT, which doubles the subtree. An exact LP on this one coordinate
                    // is far cheaper than the branch it avoids, so it is worth paying for
                    // here and nowhere else.
                    let resolved = match resolved {
                        Some(s) => Some(s),
                        None if star.num_constraints() > 0 => {
                            // ONE AY session per node, not per neuron. ay_milp is built for
                            // reuse: rigorous_bound tries Neumaier-Shcherbina off a single
                            // float solve before touching the exact rim, so the cost is in
                            // the model build — pay it once and query every coordinate.
                            //
                            // Applying a STABLE neuron never changes the predicate and never
                            // touches another coordinate's affine form, so the session stays
                            // valid across this walk. A split ends the walk.
                            // Encode ONLY this coordinate. Every target costs a column AND
                            // an equality row, and query time scales with model size:
                            // measured 29ms for a 1-target model versus 138ms for a
                            // 50-target one, and 23.5s through the batch wrapper. Building
                            // a fresh session is 70us, so a narrow model per query beats a
                            // wide model reused.
                            let t_lp = Instant::now();
                            // STAGE A — alpha-only expression bound (ay_milp's
                            // rigorous_bound_expr). The model is the star's polytope alone,
                            // so it never grows with the number of coordinates queried.
                            //
                            // It is very slightly LOOSER than the column encoding (~2e-15)
                            // because the star's constant `c` is added outside the LP rather
                            // than living in the model. That is irrelevant almost everywhere,
                            // but it is decisive for a neuron whose true bound sits exactly
                            // on 0: an outward-rounded -5.6e-16 fails `lo >= 0` and the
                            // neuron splits when it did not need to. So when this cannot
                            // classify, fall through to the tighter column encoding BEFORE
                            // paying for a split.
                            if node_session.is_none() {
                                let (a_rows, b) = predicate_of(&star);
                                let req = StarLpRequest {
                                    alpha_dim: star.alpha_dim(),
                                    a_rows,
                                    b,
                                    targets: Vec::new(),
                                };
                                node_session = Some(StarLpSession::new_alpha_only(
                                    &req,
                                    budget.lp_time_limit,
                                    budget.deadline,
                                )?);
                            }
                            let (c_i, g_i) = target_of(&star, idx)?;
                            let (a_rows, b_rhs) = predicate_of(&star);
                            let sess = node_session.as_mut().expect("just built");
                            stats.exact_lp_calls += 1;
                            // STAGE A0 — AY's FLOAT simplex for the multipliers, this crate's
                            // Lagrangian for the bound. Weak duality makes it sound for any
                            // lambda >= 0, so the float lane is never trusted: a bad lambda
                            // costs tightness, not correctness. Measured ~30-47us against the
                            // exact rim's ~300ms, and the rim is what the driver was paying
                            // on ~90% of queries.
                            let quick = sess.verified_float_bounds(c_i, &g_i, &a_rows, &b_rhs)?;
                            let classified =
                                quick.filter(|(lo, hi)| classify(*lo, *hi, &star, idx).is_some());
                            let fast = match classified {
                                Some(bounds) => {
                                    stats.verified_float_hits += 1;
                                    bounds
                                }
                                // STAGE A1 — the rigorous expression path (NS, then the exact
                                // rim). Reached only when the float-verified bound could not
                                // decide the neuron, which is precisely where the extra
                                // tightness earns its cost: a split doubles the subtree.
                                None => sess.expr_bounds(c_i, &g_i)?,
                            };
                            // Fall back ONLY where the column path could plausibly differ:
                            // a non-finite answer, or a straddle so narrow that alpha-only's
                            // ~2e-15 extra looseness (the star's constant `c` is added
                            // outside the LP) could be what is causing it. A WIDE straddle
                            // is a genuinely unstable neuron that no bound will resolve, and
                            // paying a second solve before splitting it is pure waste — that
                            // mistake made every query cost both paths.
                            let marginal = {
                                let scale = 1.0 + fast.0.abs().max(fast.1.abs());
                                fast.0.abs() <= 1e-9 * scale || fast.1.abs() <= 1e-9 * scale
                            };
                            let got = if fast.0.is_nan()
                                || classify(fast.0, fast.1, &star, idx).is_some()
                                || !marginal
                            {
                                Some(fast)
                            } else {
                                // STAGE B — the tighter column encoding, only for coordinates
                                // the cheap bound left straddling zero.
                                stats.ns_declined += 1;
                                let req = lp_request(&star, vec![(c_i, g_i.clone())])?;
                                let mut col_sess = StarLpSession::new(
                                    &req,
                                    budget.lp_time_limit,
                                    budget.deadline,
                                )?;
                                col_sess.bounds(0)?
                            };
                            stats.ns_exact_lp += t_lp.elapsed().as_nanos();
                            if now("star exact-LP publication") >= budget.deadline {
                                return Ok((
                                    StarVerdict::Unknown("deadline exceeded".into()),
                                    stats,
                                ));
                            }
                            match got {
                                // NaN pair is the session's infeasibility signal.
                                Some((lo, hi)) if lo.is_nan() || hi.is_nan() => {
                                    stats.pruned_infeasible += 1;
                                    node_proved_empty = true;
                                    break;
                                }
                                Some((lo, hi)) => classify(lo, hi, &star, idx),
                                None => None,
                            }
                        }
                        None => None,
                    };
                    if let Some(s) = resolved {
                        stats.lp_reclaimed_stable += 1;
                        star = s;
                        idx += 1;
                        continue;
                    }
                    // INPUT BISECTION instead of a neuron split, when the input space is
                    // the smaller tree. A ReLU split branches on neuron sign, so its tree is
                    // exponential in UNSTABLE NEURONS (~300 on ACAS Xu); bisection branches
                    // on the INPUT box (5 dims there). Measured: the neuron search closes up
                    // to 89% of the relaxation gap below ~16 neurons and 0% beyond, because
                    // the neuron tree outruns any budget.
                    //
                    // Bisection also tightens for free: it is a substitution, so the box
                    // shrinks in the representation and every downstream interval bound
                    // improves — where a ReLU split only adds a predicate row that the cheap
                    // bound ignores.
                    // Per-LAYER trigger: when only a handful of neurons in this layer are
                    // still unstable, split them EXACTLY. Bisection never shrinks the
                    // relaxation radius — only a neuron becoming stable does — so on a layer
                    // that is nearly resolved, exact splitting removes radius that no amount
                    // of box-halving would.
                    if budget.prefer_input_split
                        && work.depth < budget.max_depth
                        && !few_unstable_remaining(&star, layers, work.layer, idx, budget)
                    {
                        if let Some(sym) = star.widest_input_symbol() {
                            let (lo_half, hi_half) = star.split_input_symbol(sym).map_err(|e| {
                                MipError::Encoding(format!("star_verify: bisect: {e}"))
                            })?;
                            // Same substitution on the input-space twin, so a later leaf
                            // refinement restarts from the box this node actually stands for.
                            let (lo_in, hi_in) =
                                work.input.split_input_symbol(sym).map_err(|e| {
                                    MipError::Encoding(format!("star_verify: bisect input: {e}"))
                                })?;
                            stats.input_bisections += 1;
                            for (branch, branch_input) in [(lo_half, lo_in), (hi_half, hi_in)] {
                                queue.push(Work {
                                    star: branch,
                                    input: branch_input,
                                    layer: work.layer,
                                    // Re-examine the layer from the start: a tighter box can
                                    // make EARLIER neurons stable too.
                                    coord: layer_base,
                                    depth: work.depth + 1,
                                });
                            }
                            node_delegated = true;
                            break;
                        }
                    }
                    if work.depth >= budget.max_depth {
                        return Ok((
                            StarVerdict::Unknown(format!(
                                "max split depth {} reached",
                                budget.max_depth
                            )),
                            stats,
                        ));
                    }
                    split_here = Some((*inactive, *active));
                    break;
                }
            }
        }

        if node_proved_empty {
            // The exact LP proved THIS node's predicate empty. Advancing an infeasible
            // star to the next layer only makes later passes rediscover the same fact and
            // was a material throughput leak on deep exact-split trees.
            continue;
        }

        if node_delegated {
            // Input bisection queued two halves whose union is exactly this node, each
            // re-entering at `layer` from `layer_base`. The parent must stop here.
            // Falling through to the `split_here == None` arm below would ALSO advance
            // this star to `layer + 1` with the ReLU never applied to the unstable
            // coordinate — nor to any coordinate after it — which is not a relaxation
            // of the true image but a different function, so it could manufacture a
            // counterexample that does not exist. It would also re-queue the parent at
            // the ORIGINAL depth, letting that lineage bisect forever without ever
            // charging `budget.max_depth`.
            continue;
        }

        match split_here {
            None => {
                // Whole layer resolved without branching.
                queue.push(Work {
                    star,
                    input: work.input,
                    layer: work.layer + 1,
                    coord: 0,
                    depth: work.depth,
                });
            }
            Some((inactive, active)) => {
                stats.splits += 1;
                for branch in [inactive, active] {
                    // Drop provably-empty branches before they spawn descendants.
                    let branch_empty = lp_is_empty(&branch, budget)?;
                    if now("star branch-infeasibility publication") >= budget.deadline {
                        return Ok((StarVerdict::Unknown("deadline exceeded".into()), stats));
                    }
                    if branch_empty {
                        stats.pruned_infeasible += 1;
                        continue;
                    }
                    queue.push(Work {
                        star: branch,
                        // A ReLU split does not move the input box, only the predicate.
                        input: work.input.clone(),
                        layer: work.layer,
                        // Resume AT this coordinate: the split fixed its sign, so the
                        // next relu_split call on it now returns a stable arm.
                        coord: idx,
                        depth: work.depth + 1,
                    });
                }
            }
        }
    }

    // The last item can be discharged by tail/dual/leaf work after its admission
    // check. Never let queue exhaustion publish a late candidate conclusion.
    if now("star search-completion publication") >= budget.deadline {
        return Ok((StarVerdict::Unknown("deadline exceeded".into()), stats));
    }
    Ok((StarVerdict::CandidateVerified, stats))
}

/// Sound interval over-approximation of the network TAIL, used to discharge a node before
/// splitting it.
///
/// This is the "over-approximate first" half of the algorithm. Without it the search splits
/// every unstable neuron before ever looking at the property, which makes even a trivially
/// true spec cost a full enumeration (measured: a `y > -1e6` spec did not finish in 120s on
/// a 100-ReLU net).
///
/// Interval arithmetic is a RELAXATION, so it can only fail to decide — never decide wrongly.
/// If it proves every spec row, the node is genuinely safe and needs no refinement.
fn tail_interval_bounds(
    star: &Star,
    layers: &[StarLayer],
    from: usize,
    relu_from_coord: usize,
) -> Result<Vec<(f64, f64)>> {
    let bt = star
        .interval_bounds()
        .map_err(|e| MipError::Encoding(format!("star_verify: interval_bounds: {e}")))?;
    let mut lo: Vec<f64> = bt.lower().iter().map(|v| f64::from(*v)).collect();
    let mut hi: Vec<f64> = bt.upper().iter().map(|v| f64::from(*v)).collect();

    for (li, layer) in layers.iter().enumerate().skip(from) {
        match layer {
            StarLayer::Gemm { weight, bias } => {
                let (rows, cols) = (weight.nrows(), weight.ncols());
                if cols != lo.len() {
                    return Err(MipError::Encoding(format!(
                        "star_verify: tail gemm expects {cols} inputs, have {}",
                        lo.len()
                    )));
                }
                let mut nlo = vec![0.0f64; rows];
                let mut nhi = vec![0.0f64; rows];
                for r in 0..rows {
                    let (mut l, mut h) = (0.0f64, 0.0f64);
                    for c in 0..cols {
                        let w = f64::from(weight[[r, c]]);
                        if w >= 0.0 {
                            l += w * lo[c];
                            h += w * hi[c];
                        } else {
                            l += w * hi[c];
                            h += w * lo[c];
                        }
                    }
                    if let Some(b) = bias {
                        let bv = f64::from(b[r]);
                        l += bv;
                        h += bv;
                    }
                    nlo[r] = l;
                    nhi[r] = h;
                }
                lo = nlo;
                hi = nhi;
            }
            StarLayer::Relu => {
                // Coordinates before `relu_from_coord` in the CURRENT layer are already
                // applied in the star itself; re-applying max(0,.) is idempotent and sound.
                let skip = if li == from { relu_from_coord } else { 0 };
                for i in 0..lo.len() {
                    if i < skip {
                        continue;
                    }
                    lo[i] = lo[i].max(0.0);
                    hi[i] = hi[i].max(0.0);
                }
            }
        }
    }
    Ok(lo.into_iter().zip(hi).collect())
}

/// Minimum of one output affine form over an interval box.
fn row_min_over_box(coeffs: &[f64], bounds: &[(f64, f64)]) -> Option<f64> {
    if coeffs.len() != bounds.len() {
        return None;
    }
    let mut min = 0.0f64;
    for (c, &(l, h)) in coeffs.iter().zip(bounds) {
        min += if *c >= 0.0 { c * l } else { c * h };
    }
    min.is_finite().then_some(min)
}

/// Tail bound by STAR propagation: affine layers exactly, remaining ReLUs by the zonotope
/// relaxation. Strictly tighter than interval propagation because it keeps the correlations
/// between coordinates that intervals discard.
///
/// Measured motivation: with the interval tail, ACAS Xu prop_2 ran 54,055 bisections and
/// pruned ZERO nodes — six layers of IBP cannot resolve a 0.001 margin, so nothing was ever
/// discharged and the search just subdivided.
///
/// Returns `None` if the relaxation fails, so callers fall back to the interval tail.
// Superseded on the shipped path by the fixed-size tail below, which documents this one as
// the tighter-but-unaffordable alternative and links to it by name. Kept so that comparison
// stays checkable rather than becoming a claim about deleted code.
#[allow(dead_code)]
fn tail_star_bounds(
    star: &Star,
    layers: &[StarLayer],
    from: usize,
    relu_from_coord: usize,
) -> Option<Star> {
    let mut cur = star.clone();
    for (li, layer) in layers.iter().enumerate().skip(from) {
        match layer {
            StarLayer::Gemm { weight, bias } => {
                cur = cur.gemm(weight, bias.as_ref()).ok()?;
            }
            StarLayer::Relu => {
                let skip = if li == from { relu_from_coord } else { 0 };
                let width = cur.zonotope().len();
                for i in skip..width {
                    cur = cur.relu_overapprox(i).ok()?;
                }
            }
        }
    }
    Some(cur)
}

/// Tail bound that keeps the INPUT symbols exact and folds every relaxation error into a
/// per-coordinate radius — a zonotope with an interval remainder.
///
/// [`tail_star_bounds`] is tighter but grows the star by one symbol per unstable neuron, so
/// on a 300-neuron network the α dimension balloons and every later operation slows with it.
/// Measured: 256 nodes/s, with the tail dominating.
///
/// Here the representation is fixed-size: `center` and one coefficient row per ORIGINAL input
/// symbol, plus `radius`, the accumulated relaxation error per coordinate. Affine layers map
/// all three; the ReLU relaxation scales them by λ and adds μ to the radius. Cost is O(m·n)
/// per layer with NO growth.
///
/// The margin correlation that matters — between output coordinates, through shared input
/// symbols — is preserved exactly. Only the correlation BETWEEN relaxation errors is dropped,
/// which is the part a zonotope pays most to keep and gains least from.
struct TailForm {
    /// Per-coordinate constant.
    center: Vec<f64>,
    /// `coeffs[j][i]` — coefficient of input symbol `j` on coordinate `i`.
    coeffs: Vec<Vec<f64>>,
    /// Per-coordinate accumulated relaxation radius (independent, non-negative).
    radius: Vec<f64>,
}

impl TailForm {
    fn from_star(star: &Star) -> Option<Self> {
        let n = star.zonotope().len();
        let m = star.alpha_dim();
        let mut center = vec![0.0f64; n];
        let mut coeffs = vec![vec![0.0f64; n]; m];
        for i in 0..n {
            let (c, g) = star.coordinate_form(i).ok()?;
            center[i] = f64::from(c);
            for j in 0..m {
                coeffs[j][i] = f64::from(g[j]);
            }
        }
        Some(Self {
            center,
            coeffs,
            radius: vec![0.0f64; n],
        })
    }

    fn gemm(&self, weight: &Array2<f32>, bias: Option<&Array1<f32>>) -> Option<Self> {
        let (rows, cols) = (weight.nrows(), weight.ncols());
        if cols != self.center.len() {
            return None;
        }
        let mut center = vec![0.0f64; rows];
        let mut radius = vec![0.0f64; rows];
        let mut coeffs = vec![vec![0.0f64; rows]; self.coeffs.len()];
        for r in 0..rows {
            let mut c_acc = 0.0f64;
            let mut rad = 0.0f64;
            for k in 0..cols {
                let w = f64::from(weight[[r, k]]);
                if w == 0.0 {
                    continue;
                }
                c_acc += w * self.center[k];
                rad += w.abs() * self.radius[k];
                for (j, plane) in self.coeffs.iter().enumerate() {
                    coeffs[j][r] += w * plane[k];
                }
            }
            if let Some(b) = bias {
                c_acc += f64::from(b[r]);
            }
            center[r] = c_acc;
            radius[r] = rad;
        }
        Some(Self {
            center,
            coeffs,
            radius,
        })
    }

    /// Range of one coordinate over the α box and its radius.
    fn coordinate_range(&self, i: usize) -> (f64, f64) {
        let spread: f64 = self.coeffs.iter().map(|p| p[i].abs()).sum::<f64>() + self.radius[i];
        (self.center[i] - spread, self.center[i] + spread)
    }

    /// Zonotope ReLU relaxation applied in place, from coordinate `skip` onward.
    fn relu_from(&mut self, skip: usize) {
        for i in skip..self.center.len() {
            let (lo, hi) = self.coordinate_range(i);
            if lo >= 0.0 {
                continue;
            }
            if hi <= 0.0 {
                self.center[i] = 0.0;
                self.radius[i] = 0.0;
                for plane in &mut self.coeffs {
                    plane[i] = 0.0;
                }
                continue;
            }
            let lambda = hi / (hi - lo);
            let mu = -lambda * lo * 0.5;
            self.center[i] = lambda * self.center[i] + mu;
            self.radius[i] = lambda * self.radius[i] + mu;
            for plane in &mut self.coeffs {
                plane[i] *= lambda;
            }
        }
    }

    /// As [`Self::row_min`], also returning the row's α sensitivity so a caller can pick a
    /// branch symbol without a second propagation.
    fn row_min_with_alpha(&self, coefficients: &[f64]) -> Option<(f64, Vec<f64>)> {
        if coefficients.len() != self.center.len() {
            return None;
        }
        let mut constant = 0.0f64;
        let mut alpha = vec![0.0f64; self.coeffs.len()];
        let mut rad = 0.0f64;
        for (i, &w) in coefficients.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            constant += w * self.center[i];
            rad += w.abs() * self.radius[i];
            for (j, plane) in self.coeffs.iter().enumerate() {
                alpha[j] += w * plane[i];
            }
        }
        let spread: f64 = alpha.iter().map(|v| v.abs()).sum::<f64>() + rad;
        let min = constant - spread;
        min.is_finite().then_some((min, alpha))
    }

    /// Minimum of `coefficients · output` over the α box, keeping input correlations.
    #[allow(dead_code)]
    fn row_min(&self, coefficients: &[f64]) -> Option<f64> {
        if coefficients.len() != self.center.len() {
            return None;
        }
        let mut constant = 0.0f64;
        let mut alpha = vec![0.0f64; self.coeffs.len()];
        let mut rad = 0.0f64;
        for (i, &w) in coefficients.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            constant += w * self.center[i];
            rad += w.abs() * self.radius[i];
            for (j, plane) in self.coeffs.iter().enumerate() {
                alpha[j] += w * plane[i];
            }
        }
        let spread: f64 = alpha.iter().map(|v| v.abs()).sum::<f64>() + rad;
        let min = constant - spread;
        min.is_finite().then_some(min)
    }
}

/// Pick the branch symbol from an already-computed margin sensitivity.
///
/// MEASURED NEGATIVE, TWICE — not wired. Tried expensively (recomputing the tail per branch
/// decision: 45,698 -> 30,280 nodes) and then for FREE, reusing the sensitivity the discharge
/// check already produces. Both lost decisiveness on a small case that width-bisection
/// decides, and `input_bisection_never_changes_the_verdict` caught both. The depth histogram
/// shows the failure mode: the heuristic re-picks the SAME symbol into a narrow chain —
/// (1,0),(2,1),(2,1)... to the depth cap — instead of spreading refinement across symbols.
///
/// Twice-refuted is enough. The oscillation in the discharge rate is real, but attacking it
/// through branch CHOICE is not the fix; the residual relaxation radius, which no bisection
/// shrinks, is the better suspect.
///
/// `alpha[j]` is the margin row's coefficient on symbol `j`, so halving symbol `j` halves
/// its contribution to that row's spread. The largest |coefficient| therefore buys the most
/// bound per bisection — and because the caller already computed this vector while
/// discharging, it costs nothing. The earlier attempt at this heuristic recomputed the tail
/// per decision and lost on throughput; that was the implementation, not the idea.
///
/// Falls back to `None` when no symbol carries a usable coefficient.
#[allow(dead_code)] // measured-negative branch-choice reference retained above
fn symbol_from_sensitivity(alpha: &[f64]) -> Option<usize> {
    alpha
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite() && v.abs() > 0.0)
        .max_by(|a, b| {
            a.1.abs()
                .partial_cmp(&b.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(j, _)| j)
}

/// Are few enough neurons still unstable that EXACT splitting beats more bisection?
///
/// Bisection never shrinks `TailForm`'s relaxation radius; only a neuron becoming stable
/// does. When the remaining unstable count is small, splitting them exactly removes the
/// radius entirely and is cheaper than halving boxes that no longer tighten the bound.
/// `star` is already at `layers[from]`, with coordinates before
/// `relu_from_coord` in that ReLU already applied, so only that exact suffix may
/// be replayed. Reapplying the prefix would either dimension-mismatch or silently
/// transform an equal-width network twice.
fn few_unstable_remaining(
    star: &Star,
    layers: &[StarLayer],
    from: usize,
    relu_from_coord: usize,
    budget: &StarBudget,
) -> bool {
    if budget.exact_below_unstable == 0 {
        return false;
    }
    if from > layers.len() {
        return false;
    }
    let Some(mut form) = TailForm::from_star(star) else {
        return false;
    };
    let mut unstable = 0usize;
    for (layer_index, layer) in layers.iter().enumerate().skip(from) {
        match layer {
            StarLayer::Gemm { weight, bias } => {
                let Some(next) = form.gemm(weight, bias.as_ref()) else {
                    return false;
                };
                form = next;
            }
            StarLayer::Relu => {
                let first_coordinate = if layer_index == from {
                    relu_from_coord
                } else {
                    0
                };
                if first_coordinate > form.center.len() {
                    return false;
                }
                for i in first_coordinate..form.center.len() {
                    let (lo, hi) = form.coordinate_range(i);
                    if lo < 0.0 && hi > 0.0 {
                        unstable += 1;
                        if unstable >= budget.exact_below_unstable {
                            return false;
                        }
                    }
                }
                form.relu_from(first_coordinate);
            }
        }
    }
    true
}

/// Which input symbol most reduces the PROPERTY's bound if bisected?
///
/// MEASURED NEGATIVE — not wired. Attacking the margin row's largest-|coefficient| symbol
/// sounds better than bisecting the widest generator, but it costs a full tail propagation
/// per branch DECISION and did not improve the discharge rate: on ACAS Xu prop_2 it took
/// 45,698 -> 30,280 nodes in 60s at an unchanged ~50% discharge, and it cost decisiveness on
/// a small case (Unknown where width-bisection concluded), which
/// `input_bisection_never_changes_the_verdict` caught. Kept for the record so the next
/// attempt does not re-derive it; a cheaper sensitivity proxy might still win.
#[allow(dead_code)]
///
/// Bisecting the widest generator is a proxy for "biggest box", not "biggest obstacle". What
/// matters is the composed margin row: halving a symbol halves its contribution to that row's
/// spread, so the symbol with the largest |coefficient| in the row nearest to deciding is the
/// one that buys the most.
///
/// Measured motivation: bisecting by width held the discharge rate at ~50%, so the frontier
/// neither grew nor shrank and the search just went deeper.
///
/// Returns `None` when no symbol carries a finite non-zero coefficient, i.e. bisecting could
/// not tighten the property at all.
fn best_symbol_for_property(
    star: &Star,
    layers: &[StarLayer],
    from: usize,
    relu_from_coord: usize,
    property: SearchProperty<'_>,
) -> Option<usize> {
    let mut form = TailForm::from_star(star)?;
    for (li, layer) in layers.iter().enumerate().skip(from) {
        match layer {
            StarLayer::Gemm { weight, bias } => form = form.gemm(weight, bias.as_ref())?,
            StarLayer::Relu => form.relu_from(if li == from { relu_from_coord } else { 0 }),
        }
    }
    // The row closest to discharging is the one worth attacking.
    let rows: Vec<&[f64]> = match property {
        SearchProperty::SafeConjunction(spec) => {
            spec.rows.iter().map(|(c, _)| c.as_slice()).collect()
        }
        SearchProperty::UnsafeConjunction(spec) => spec
            .rows
            .iter()
            .map(|r| r.coefficients.as_slice())
            .collect(),
    };
    let mut best_row: Option<(f64, Vec<f64>)> = None;
    for coefficients in rows {
        let mut alpha = vec![0.0f64; form.coeffs.len()];
        let mut constant = 0.0f64;
        let mut rad = 0.0f64;
        for (i, &w) in coefficients.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            constant += w * form.center[i];
            rad += w.abs() * form.radius[i];
            for (j, plane) in form.coeffs.iter().enumerate() {
                alpha[j] += w * plane[i];
            }
        }
        let spread: f64 = alpha.iter().map(|v| v.abs()).sum::<f64>() + rad;
        let min = constant - spread;
        if !min.is_finite() {
            continue;
        }
        // Highest min = closest to clearing its threshold.
        if best_row.as_ref().is_none_or(|(m, _)| min > *m) {
            best_row = Some((min, alpha));
        }
    }
    let (_, alpha) = best_row?;
    alpha
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite() && v.abs() > 0.0)
        .max_by(|a, b| {
            a.1.abs()
                .partial_cmp(&b.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(j, _)| j)
}

/// Fixed-size tail bound: propagate a [`TailForm`] through the remaining layers.
fn tail_form_discharges(
    star: &Star,
    layers: &[StarLayer],
    from: usize,
    relu_from_coord: usize,
    property: SearchProperty<'_>,
) -> Option<(bool, Vec<f64>)> {
    let mut form = TailForm::from_star(star)?;
    for (li, layer) in layers.iter().enumerate().skip(from) {
        match layer {
            StarLayer::Gemm { weight, bias } => {
                form = form.gemm(weight, bias.as_ref())?;
            }
            StarLayer::Relu => {
                form.relu_from(if li == from { relu_from_coord } else { 0 });
            }
        }
    }
    // Discharge, AND hand back the α sensitivity of the row nearest to deciding, so the
    // branch choice costs nothing extra. Bisecting the widest GENERATOR only improves the
    // margin once a full round over every symbol completes — exactly the oscillation the
    // depth histogram showed, where the rate spikes at the end of a round and collapses at
    // the start of the next. Bisecting the symbol the margin is most sensitive to attacks
    // the bound at every step instead.
    let rows: Vec<(&[f64], f64)> = match property {
        SearchProperty::SafeConjunction(spec) => {
            spec.rows.iter().map(|(c, t)| (c.as_slice(), *t)).collect()
        }
        SearchProperty::UnsafeConjunction(spec) => spec
            .rows
            .iter()
            .map(|r| (r.coefficients.as_slice(), r.threshold))
            .collect(),
    };
    let mut discharged = matches!(property, SearchProperty::SafeConjunction(_));
    let mut best: Option<(f64, Vec<f64>)> = None;
    for (coefficients, threshold) in rows {
        let Some((min, alpha)) = form.row_min_with_alpha(coefficients) else {
            if matches!(property, SearchProperty::SafeConjunction(_)) {
                discharged = false;
            }
            continue;
        };
        match property {
            SearchProperty::SafeConjunction(_) => {
                if min <= threshold {
                    discharged = false;
                }
            }
            SearchProperty::UnsafeConjunction(_) => {
                if min > threshold {
                    discharged = true;
                }
            }
        }
        let slack = min - threshold;
        if best.as_ref().is_none_or(|(s, _)| slack > *s) {
            best = Some((slack, alpha));
        }
    }
    Some((discharged, best.map_or_else(Vec::new, |(_, a)| a)))
}

/// Does the propagated star discharge the property, using the star's OWN affine forms?
///
/// Bounding a margin row from per-coordinate boxes computes `min(Y_i) − max(Y_0)`, which
/// throws away exactly the correlation the star exists to keep — the same error class as
/// bounding a classification margin by summing independent logit bounds. Composing the row
/// against the star first and bounding the single resulting affine form is dramatically
/// tighter, and it is the difference between pruning and not.
// Superseded on the shipped path by the fixed-size-form sibling above; kept as the exact-star
// reference the argument in this doc comment is about.
#[allow(dead_code)]
fn star_discharges_property(star: &Star, property: SearchProperty<'_>) -> bool {
    let row_min = |coefficients: &[f64]| -> Option<f64> {
        let (constant, alpha) = compose_output_row(star, coefficients).ok()?;
        // min over the alpha box; the predicate can only shrink this further.
        let radius: f64 = alpha.iter().map(|v| v.abs()).sum();
        let min = constant - radius;
        min.is_finite().then_some(min)
    };
    match property {
        SearchProperty::SafeConjunction(spec) => {
            spec.rows.iter().all(|(coefficients, threshold)| {
                row_min(coefficients).is_some_and(|min| min > *threshold)
            })
        }
        SearchProperty::UnsafeConjunction(spec) => spec
            .rows
            .iter()
            .any(|row| row_min(&row.coefficients).is_some_and(|min| min > row.threshold)),
    }
}

/// Does the interval over-approximation already discharge this search node?
fn tail_discharges_property(bounds: &[(f64, f64)], property: SearchProperty<'_>) -> bool {
    match property {
        SearchProperty::SafeConjunction(spec) => spec.rows.iter().all(|(coeffs, threshold)| {
            row_min_over_box(coeffs, bounds).is_some_and(|min| min > *threshold)
        }),
        SearchProperty::UnsafeConjunction(spec) => {
            // The unsafe region is a conjunction. If even ONE row cannot hold over this
            // output box, the whole conjunction is empty on this node.
            spec.rows.iter().any(|row| {
                row_min_over_box(&row.coefficients, bounds).is_some_and(|min| min > row.threshold)
            })
        }
    }
}

enum LeafOutcome {
    Safe,
    Empty,
    Violated,
    Undecided,
}

/// Interpret the rigorous dummy-target query used for unsafe-conjunction feasibility.
///
/// `infeasible=true` is mathematically useful proof evidence, but an expired bounded search
/// may not publish it as a completed candidate result. The converse is not true:
/// `infeasible=false` means only "not proved empty". Require a completed finite bound on the
/// identically-zero target before returning that candidate result; a deadline cut, partial
/// `(-inf,+inf)` result, malformed target count, or a bound that excludes zero is undecided.
fn unsafe_closure_report_outcome(report: &StarLpReport, deadline: Instant) -> LeafOutcome {
    unsafe_closure_report_outcome_at(report, deadline, Instant::now())
}

fn unsafe_closure_report_outcome_at(
    report: &StarLpReport,
    deadline: Instant,
    now: Instant,
) -> LeafOutcome {
    if now >= deadline {
        return LeafOutcome::Undecided;
    }
    if report.infeasible {
        return LeafOutcome::Safe;
    }
    match report.lp_bounds.as_slice() {
        [(lo, hi)] if lo.is_finite() && hi.is_finite() && *lo <= 0.0 && *hi >= 0.0 => {
            LeafOutcome::Violated
        }
        _ => LeafOutcome::Undecided,
    }
}

/// Decide a leaf star against the requested property semantics.
fn discharge_leaf(
    star: &Star,
    property: SearchProperty<'_>,
    budget: &StarBudget,
    stats: &mut StarStats,
) -> Result<LeafOutcome> {
    match property {
        SearchProperty::SafeConjunction(spec) => discharge_safe_leaf(star, spec, budget, stats),
        SearchProperty::UnsafeConjunction(spec) => {
            discharge_unsafe_conjunction_leaf(star, spec, budget, stats)
        }
    }
}

/// Compose `coefficients·star_output` into `(constant, alpha_coefficients)`.
fn compose_output_row(star: &Star, coefficients: &[f64]) -> Result<(f64, Vec<f64>)> {
    let width = star.zonotope().len();
    if coefficients.len() != width {
        return Err(MipError::Encoding(format!(
            "star_verify: spec row width {} != output width {width}",
            coefficients.len()
        )));
    }
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(MipError::Encoding(
            "star_verify: non-finite output coefficient".into(),
        ));
    }
    let mut c_acc = 0.0f64;
    let mut g_acc = vec![0.0f64; star.alpha_dim()];
    for (i, &weight) in coefficients.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        let (c_i, g_i) = target_of(star, i)?;
        c_acc += weight * c_i;
        for (acc, gi) in g_acc.iter_mut().zip(&g_i) {
            *acc += weight * gi;
        }
    }
    if !c_acc.is_finite() || g_acc.iter().any(|value| !value.is_finite()) {
        return Err(MipError::Encoding(
            "star_verify: composed output row is non-finite".into(),
        ));
    }
    Ok((c_acc, g_acc))
}

/// Decide a leaf star against the original conjunction of safe margins.
fn discharge_safe_leaf(
    star: &Star,
    spec: &StarSpec,
    budget: &StarBudget,
    _stats: &mut StarStats,
) -> Result<LeafOutcome> {
    // Compose each spec row with the star's output coordinates into a single affine form
    // over α, so one LP bounds every row.
    let targets = spec
        .rows
        .iter()
        .map(|(coefficients, threshold)| {
            if !threshold.is_finite() {
                return Err(MipError::Encoding(
                    "star_verify: non-finite spec threshold".into(),
                ));
            }
            compose_output_row(star, coefficients)
        })
        .collect::<Result<Vec<_>>>()?;

    // A concrete star has no alpha variables and therefore no LP model to build.
    if star.alpha_dim() == 0 {
        return Ok(
            if spec
                .rows
                .iter()
                .zip(&targets)
                .all(|((_, threshold), (constant, _))| *constant > *threshold)
            {
                LeafOutcome::Safe
            } else {
                LeafOutcome::Violated
            },
        );
    }

    // Cheap sound bound first. It decides most leaves outright; the exact LP is only
    // consulted when the dual leaves the spec genuinely undecided, which keeps the
    // heavyweight solver off the hot path.
    let (a_rows, b) = predicate_of(star);
    if dual_certifies_empty(&a_rows, &b, budget.dual_iters) {
        return Ok(LeafOutcome::Empty);
    }
    let dual: Vec<(f64, f64)> = targets
        .iter()
        .map(|(c, g)| dual_coordinate_bounds(*c, g, &a_rows, &b, budget.dual_iters))
        .collect();
    if spec
        .rows
        .iter()
        .zip(&dual)
        .all(|((_, t), (lo, _))| lo.is_finite() && lo > t)
    {
        return Ok(LeafOutcome::Safe);
    }

    let req = lp_request(star, targets)?;
    let report = star_predicate_bounds(&req, budget.lp_time_limit, budget.deadline)?;
    if report.infeasible {
        return Ok(LeafOutcome::Empty);
    }

    // Candidate-verified iff EVERY row's minimum strictly clears its threshold.
    for (i, (_c, t)) in spec.rows.iter().enumerate() {
        let (lo, _hi) = report.lp_bounds[i];
        if !lo.is_finite() {
            return Ok(LeafOutcome::Undecided);
        }
        if lo <= *t {
            // This leaf is exact in the internal star model, so a minimum at or below the
            // threshold means a violating point exists in that model.
            return Ok(LeafOutcome::Violated);
        }
    }
    Ok(LeafOutcome::Safe)
}

/// Decide whether one CLOSED unsafe conjunction intersects a leaf star.
///
/// Strict rows are intentionally relaxed from `<` to `<=`. Empty closure is enough to
/// establish candidate emptiness. Failure to prove the closure empty is reported only as
/// such: the current LP report does not carry a primal-feasibility certificate, and a
/// genuinely feasible closure could still contain only boundary points, not a strict point.
fn discharge_unsafe_conjunction_leaf(
    star: &Star,
    spec: &StarUnsafeConjunction,
    budget: &StarBudget,
    _stats: &mut StarStats,
) -> Result<LeafOutcome> {
    let forms = spec
        .rows
        .iter()
        .map(|row| {
            if !row.threshold.is_finite() {
                return Err(MipError::Encoding(
                    "star_verify: non-finite unsafe threshold".into(),
                ));
            }
            compose_output_row(star, &row.coefficients)
        })
        .collect::<Result<Vec<_>>>()?;

    // A concrete (zero-alpha) star needs no solver.
    if star.alpha_dim() == 0 {
        let closure_feasible = spec
            .rows
            .iter()
            .zip(&forms)
            .all(|(row, (constant, _))| *constant <= row.threshold);
        return Ok(if closure_feasible {
            LeafOutcome::Violated
        } else {
            LeafOutcome::Safe
        });
    }

    let (base_a, base_b) = predicate_of(star);
    if dual_certifies_empty(&base_a, &base_b, budget.dual_iters) {
        return Ok(LeafOutcome::Empty);
    }

    let base_len = base_b.len();
    let mut unsafe_a = base_a;
    let mut unsafe_b = base_b;
    unsafe_a.reserve(forms.len());
    unsafe_b.reserve(forms.len());
    for (row, (constant, alpha_coefficients)) in spec.rows.iter().zip(forms) {
        // c + g·alpha <= t  ==>  g·alpha <= t - c.
        let rhs = row.threshold - constant;
        if !rhs.is_finite() {
            return Err(MipError::Encoding(
                "star_verify: composed unsafe rhs is non-finite".into(),
            ));
        }
        unsafe_a.push(alpha_coefficients);
        unsafe_b.push(rhs);
    }

    if dual_certifies_empty(&unsafe_a, &unsafe_b, budget.dual_iters) {
        return Ok(LeafOutcome::Safe);
    }
    // #star-leaf-probe (dark): why did a leaf fail to discharge? Prints the per-row slack
    // and the residual box width, which distinguishes "the box is still too coarse" from
    // "the arithmetic cannot resolve the margin".
    if std::env::var("NY_STAR_LEAF_PROBE").as_deref() == Ok("1") {
        let widths: Vec<f64> = (0..star.alpha_dim())
            .map(|j| unsafe_a.iter().map(|r| r[j].abs()).sum::<f64>())
            .collect();
        let slacks: Vec<f64> = unsafe_b.iter().skip(base_len).copied().collect();
        eprintln!(
            "LEAFPROBE alpha_dim={} rows={} min_rhs={:?} row_widths={:?}",
            star.alpha_dim(),
            unsafe_a.len(),
            slacks.iter().cloned().fold(f64::INFINITY, f64::min),
            &widths[..widths.len().min(5)]
        );
    }

    // `star_predicate_bounds` runs rigorous feasibility as part of bounding a target.
    // Materialise one identically-zero target so it cannot return early on an empty
    // target list; the target adds x=0 and does not change feasibility.
    let request = StarLpRequest {
        alpha_dim: star.alpha_dim(),
        a_rows: unsafe_a,
        b: unsafe_b,
        targets: vec![(0.0, vec![0.0; star.alpha_dim()])],
    };
    let report = star_predicate_bounds(&request, budget.lp_time_limit, budget.deadline)?;
    Ok(unsafe_closure_report_outcome(&report, budget.deadline))
}

/// Is this branch's predicate polytope provably empty?
fn lp_is_empty(star: &Star, budget: &StarBudget) -> Result<bool> {
    if star.num_constraints() == 0 {
        return Ok(false);
    }
    let (a_rows, b) = predicate_of(star);
    Ok(dual_certifies_empty(&a_rows, &b, budget.dual_iters))
}

#[cfg(test)]
#[path = "star_verify_tests.rs"]
mod tests;
