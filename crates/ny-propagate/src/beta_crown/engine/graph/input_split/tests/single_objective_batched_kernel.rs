// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched single-objective deferred-rebound kernel tests (#4210).
//!
//! # What these four tests observe, and why it changed
//!
//! They used to measure "the batched kernel fired" by counting `gemm_f32` calls
//! on an injected [`CountingGemmEngine`]. `9f06e35b2` moved
//! `dispatch_node_backward`'s `Layer::Linear` arm onto
//! `propagate_linear_pollable_cpu` (a `#deadline-parity` fix that stays: it
//! removes an arithmetic split between the always-deadlined competition binary
//! and the never-deadlined test suite). The batched lane therefore issues NO
//! engine GEMM at all, and that counter can only ever read 0 — an instrument
//! that can only produce one reading is not a test.
//!
//! The observable is now `backward_core`'s Linear-node dispatch probe, and the
//! property asserted is the one those counts were a proxy for: **one shared
//! backward walk serves every domain in the batch**, so the dispatch count is
//! INVARIANT to the number of domains, and therefore strictly below what the
//! same domains cost one at a time.
//!
//! # And why the bound tolerances changed
//!
//! The same commit split the two lanes' arithmetic: the per-domain baseline runs
//! with a `GemmEngine` and publishes `γ_n^{f32}·S`, the batched lane publishes
//! `γ_n^{f64}·S` (~2^29 tighter). The old flat `1e-6` was calibrated when both
//! shared one f32 GEMM. Tolerances here are now DERIVED from that certificate
//! algebra for this fixture — see [`super::lane_gap`] — and the equality claims
//! are replaced by the two properties actually owed:
//!
//! * **containment in the reference lane**: the batched box may never be LOOSER
//!   than the per-domain baseline (beyond the derived certificate gap), and
//! * **soundness against an independently computed value**: the batched box must
//!   contain the EXACT range of this piecewise-linear fixture over the domain,
//!   computed here by vertex enumeration and never by the code under test.

use super::*;
use crate::beta_crown::engine::graph::propagation::batched::{
    linear_dispatches_on_thread, soa_backward_engaged_on_thread,
};
use crate::beta_crown::engine::GraphDomainBatchMetricsSink;
use crate::beta_crown::GraphDomainBatchRecord;
use crate::bounds::GraphAlphaState;
use crate::layers::linear::crown_single_gamma_n_f64;
use ny_core::Result as NyResult;
use ny_test_utils::CountingGemmEngine;
use std::sync::Mutex;

use super::lane_gap::{
    assert_linear_bounds_within_gap, box_magnitudes, cross_lane_gap, LaneGap, LaneStep,
};

/// The fixture's weights, in ONE place. The graph builder, the derived
/// tolerance, and the independent ground-truth evaluator all read these, so a
/// fixture edit cannot silently invalidate the derivation.
const W1_4210: [[f32; 2]; 2] = [[1.1, -0.4], [0.3, 0.8]];
const W2_4210: [[f32; 2]; 1] = [[0.7, -0.2]];
const B2_4210: f32 = 0.05;
/// The root input box the fixture splits. Both child domains are subsets of it.
const ROOT_BOX_4210: (f32, f32) = (-1.0, 1.0);

fn w1_4210() -> Array2<f32> {
    arr2(&W1_4210)
}

fn w2_4210() -> Array2<f32> {
    arr2(&W2_4210)
}

fn build_single_objective_batching_graph_4210() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1_4210(), None).expect("valid linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2_4210(), Some(arr1(&[B2_4210]))).expect("valid linear2")),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

fn root_input_4210() -> BoundedTensor {
    let (lo, hi) = ROOT_BOX_4210;
    BoundedTensor::new(arr1(&[lo, lo]).into_dyn(), arr1(&[hi, hi]).into_dyn())
        .expect("valid root input")
}

fn build_batched_kernel_fixture_4210() -> (
    GraphNetwork,
    HashMap<String, BoundedTensor>,
    Array2<f32>,
    BoundedTensor,
    BoundedTensor,
) {
    let graph = build_single_objective_batching_graph_4210();
    let root_input = root_input_4210();
    let root_node_bounds = graph
        .collect_node_bounds_with_engine(&root_input, None)
        .expect("root node bounds");
    let spec_matrix = arr2(&[[1.0_f32]]);
    let child_a = BoundedTensor::new(
        arr1(&[-0.8_f32, -0.5_f32]).into_dyn(),
        arr1(&[0.4_f32, 0.9_f32]).into_dyn(),
    )
    .expect("valid child a");
    let child_b = BoundedTensor::new(
        arr1(&[-0.2_f32, -0.7_f32]).into_dyn(),
        arr1(&[0.9_f32, 0.3_f32]).into_dyn(),
    )
    .expect("valid child b");
    (graph, root_node_bounds, spec_matrix, child_a, child_b)
}

// ---------------------------------------------------------------------------
// The derived cross-lane tolerance
// ---------------------------------------------------------------------------

/// Cross-lane certificate gap for this fixture over `domain`.
///
/// Chain, output-first: `linear2` (contraction width `n = rows(W2) = 1`, input
/// node `relu1`) → `relu1` (slope in `[0,1]`, contributes nothing) → `linear1`
/// (`n = rows(W1) = 2`, input node = the network input). See [`super::lane_gap`]
/// for the algebra; this function only supplies the fixture's `|W|` matrices and
/// the box magnitudes.
///
/// `MAG(relu1)` is taken over the ROOT box, not the child domain. Tests 2–4 hand
/// the run root node bounds, so that IS the box its relaxation was built from;
/// test 1 lets the run derive a tighter box from the child domain, and a wider
/// `MAG` only makes the tolerance more conservative.
fn lane_gap_4210(
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    domain: &BoundedTensor,
) -> LaneGap {
    let root_node_bounds = graph
        .collect_node_bounds_with_engine(&root_input_4210(), None)
        .expect("root node bounds for the lane-gap derivation");
    let relu1 = root_node_bounds
        .get("relu1")
        .expect("fixture must expose a 'relu1' node — the lane-gap chain names it");
    let relu1_mag = box_magnitudes(relu1);
    let input_mag = box_magnitudes(domain);

    let spec_abs = spec_matrix.mapv(|v| f64::from(v.abs()));
    cross_lane_gap(
        &spec_abs,
        &[
            LaneStep::new(&w2_4210(), &relu1_mag),
            LaneStep::new(&w1_4210(), &input_mag),
        ],
    )
}

// ---------------------------------------------------------------------------
// Independently computed ground truth
// ---------------------------------------------------------------------------

/// EXACT range of `spec · f(x)` over the box, by vertex enumeration.
///
/// `f(x) = W2·relu(W1·x) + b2` is piecewise linear on a 2-D box cut by the two
/// hyperplanes `(W1·x)_k = 0`. A piecewise-linear function attains its extrema
/// over a convex polygon at a vertex of the induced subdivision, so the exact
/// range is the min/max over: the four box corners, every intersection of a
/// `(W1·x)_k = 0` line with a box edge, and the intersection of the two lines
/// (both pass through the origin here, so that is the origin).
///
/// This is computed from the fixture constants in f64 and NEVER from the code
/// under test — it is the independent value the certified box must contain.
fn exact_range_4210(domain: &BoundedTensor, spec: f32) -> (f64, f64) {
    let lower: Vec<f64> = domain.lower().iter().map(|&v| f64::from(v)).collect();
    let upper: Vec<f64> = domain.upper().iter().map(|&v| f64::from(v)).collect();
    assert_eq!(
        lower.len(),
        2,
        "vertex enumeration is written for a 2-D box"
    );

    let mut points: Vec<[f64; 2]> = Vec::new();
    for &x0 in &[lower[0], upper[0]] {
        for &x1 in &[lower[1], upper[1]] {
            points.push([x0, x1]);
        }
    }
    let inside =
        |p: [f64; 2]| p[0] >= lower[0] && p[0] <= upper[0] && p[1] >= lower[1] && p[1] <= upper[1];
    for row in W1_4210 {
        let (a, b) = (f64::from(row[0]), f64::from(row[1]));
        // a·x0 + b·x1 = 0 against each of the four box edges.
        if b != 0.0 {
            for &x0 in &[lower[0], upper[0]] {
                let p = [x0, -a * x0 / b];
                if inside(p) {
                    points.push(p);
                }
            }
        }
        if a != 0.0 {
            for &x1 in &[lower[1], upper[1]] {
                let p = [-b * x1 / a, x1];
                if inside(p) {
                    points.push(p);
                }
            }
        }
    }
    // Both relu boundaries pass through the origin (W1 has no bias).
    if inside([0.0, 0.0]) {
        points.push([0.0, 0.0]);
    }

    let eval = |p: [f64; 2]| -> f64 {
        let hidden: Vec<f64> = W1_4210
            .iter()
            .map(|row| (f64::from(row[0]) * p[0] + f64::from(row[1]) * p[1]).max(0.0))
            .collect();
        let out: f64 = W2_4210[0]
            .iter()
            .zip(hidden.iter())
            .map(|(&w, &h)| f64::from(w) * h)
            .sum::<f64>()
            + f64::from(B2_4210);
        f64::from(spec) * out
    };

    let values: Vec<f64> = points.into_iter().map(eval).collect();
    assert!(
        values.len() >= 4,
        "vertex enumeration produced no candidates — the ground truth would be vacuous"
    );
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

/// Rounding allowance for [`exact_range_4210`]'s own f64 evaluation.
///
/// Each hidden unit costs 2 products + 1 sum, the output row 2 products + 2 sums:
/// at most 12 f64 roundings on a value whose intermediate magnitudes are bounded
/// by `scale`. `γ_12^{f64}·scale` covers them (Higham Thm 3.1).
fn exact_range_eval_slack_4210(domain: &BoundedTensor) -> f64 {
    let mag: f64 = domain
        .lower()
        .iter()
        .zip(domain.upper().iter())
        .map(|(&l, &u)| f64::from(l.abs().max(u.abs())))
        .sum();
    let w_mass: f64 = W1_4210
        .iter()
        .flat_map(|row| row.iter())
        .map(|&w| f64::from(w.abs()))
        .sum::<f64>()
        * W2_4210[0].iter().map(|&w| f64::from(w.abs())).sum::<f64>();
    crown_single_gamma_n_f64(12) * (mag * w_mass + f64::from(B2_4210.abs())).max(1.0)
}

/// The certified box must contain the exact range. This is the assertion that
/// makes every tolerance above non-vacuous: a batched lane that "agrees with the
/// baseline" while publishing an unsound box fails HERE.
fn assert_contains_exact_range_4210(
    label: &str,
    domain_lower: f32,
    domain_upper: f32,
    domain: &BoundedTensor,
    spec: f32,
) {
    let (min, max) = exact_range_4210(domain, spec);
    let slack = exact_range_eval_slack_4210(domain);
    let mut failures = Vec::new();
    if f64::from(domain_lower) > min + slack {
        failures.push(format!(
            "lower bound {domain_lower} EXCEEDS the exact minimum {min} (slack {slack:e}) — \
             the certified box excludes a reachable output, i.e. a FALSE PROOF"
        ));
    }
    if f64::from(domain_upper) < max - slack {
        failures.push(format!(
            "upper bound {domain_upper} is BELOW the exact maximum {max} (slack {slack:e}) — \
             the certified box excludes a reachable output, i.e. a FALSE PROOF"
        ));
    }
    assert!(
        failures.is_empty(),
        "{label}: soundness violated against the independently enumerated exact range:\n  {}",
        failures.join("\n  ")
    );
    assert!(
        domain_upper - domain_lower > 0.0 && domain_lower.is_finite() && domain_upper.is_finite(),
        "{label}: degenerate certified box [{domain_lower}, {domain_upper}] would make every \
         comparison above vacuous"
    );
}

// ---------------------------------------------------------------------------
// The batched-lane engagement instruments
// ---------------------------------------------------------------------------

/// Collecting sink for the PRODUCTION `GraphDomainBatchRecord` the deferred
/// rebound already emits. `domains_batched` / `domains_fallback` are that
/// record's own account of how many domains the shared batched executor served
/// versus how many fell back to per-domain work.
#[derive(Default)]
struct CollectingBatchSink4210 {
    records: Mutex<Vec<GraphDomainBatchRecord>>,
}

impl GraphDomainBatchMetricsSink for CollectingBatchSink4210 {
    fn record_batch_summary(&self, record: &GraphDomainBatchRecord) -> NyResult<()> {
        self.records
            .lock()
            .expect("4210 batch sink mutex")
            .push(record.clone());
        Ok(())
    }
}

/// Thread-local engagement counts for BOTH legs of the domain-batched backward.
///
/// The batched backward is not one code path. With no deadline set — which is
/// every test — `batched_bwd::try_backward_soa` accepts and returns before the
/// per-node reference loop is reached, so `linear_dispatches_on_thread` reads 0
/// even though the batched lane fired. Counting one leg is how the previous
/// instrument (`CountingGemmEngine::gemm_calls`) became unable to produce any
/// reading but 0. Both legs are read here and either one satisfies "the batched
/// lane fired"; what must NEVER satisfy it is the rayon per-domain fallback,
/// which runs the scalar single-domain CROWN on worker threads and therefore
/// leaves both counters at 0.
#[derive(Debug, Clone, Copy)]
struct BatchedLegEngagement4210 {
    soa: usize,
    reference_linear_nodes: usize,
}

impl BatchedLegEngagement4210 {
    fn total(self) -> usize {
        self.soa + self.reference_linear_nodes
    }
}

fn batched_leg_engagement_4210<R>(f: impl FnOnce() -> R) -> (R, BatchedLegEngagement4210) {
    let soa_before = soa_backward_engaged_on_thread();
    let linear_before = linear_dispatches_on_thread();
    let result = f();
    (
        result,
        BatchedLegEngagement4210 {
            soa: soa_backward_engaged_on_thread() - soa_before,
            reference_linear_nodes: linear_dispatches_on_thread() - linear_before,
        },
    )
}

/// Run the deferred rebound under both instruments.
#[allow(clippy::too_many_arguments)]
fn rebound_with_instruments_4210(
    domains: &mut [GraphInputDomain],
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    engine: &CountingGemmEngine,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    config: &BetaCrownConfig,
) -> (BatchedLegEngagement4210, Vec<GraphDomainBatchRecord>) {
    let sink = CollectingBatchSink4210::default();
    let (result, engagement) = batched_leg_engagement_4210(|| {
        bound_deferred_domains_batch_with_metrics(
            domains,
            graph,
            spec_matrix,
            Some(engine),
            root_node_bounds,
            alpha_state,
            None,
            None,
            None,
            config,
            Some(&sink),
            0,
        )
    });
    result.expect("deferred rebound should succeed");
    let records = sink.records.lock().expect("4210 batch sink mutex").clone();
    (engagement, records)
}

/// The batched lane must have served the WHOLE batch in one shared executor.
///
/// Two independent observables, because either alone has already failed this
/// project once:
///
/// * the production `GraphDomainBatchRecord` — `domains_batched` counts the
///   domains the shared executor took and `domains_fallback` the ones it did
///   not, so `domains_fallback > 0` is the rayon-per-domain fallback this test
///   exists to exclude; and
/// * a thread-local engagement counter on each of the two batched-backward legs,
///   which fails if the work silently moved off the caller's thread (the rayon
///   fallback's signature) even should the record be emitted anyway.
fn assert_batched_lane_served_whole_batch_4210(
    engagement: BatchedLegEngagement4210,
    records: &[GraphDomainBatchRecord],
    expected_domains: usize,
) {
    assert_eq!(
        records.len(),
        1,
        "one deferred rebound should emit exactly one batch record, got {records:?}"
    );
    let record = &records[0];
    assert_eq!(
        record.domains_batched, expected_domains,
        "the shared batched executor served {} of {expected_domains} domains \
         (fallback={}, reasons={:?}) — the batched kernel did not fire for the whole batch",
        record.domains_batched, record.domains_fallback, record.fallback_reason_counts
    );
    assert_eq!(
        record.domains_fallback, 0,
        "no domain may fall back to per-domain rebound here; reasons={:?}",
        record.fallback_reason_counts
    );
    assert!(
        engagement.total() > 0,
        "neither batched-backward leg engaged on the calling thread \
         (soa={}, reference Linear-node dispatches={}). The rayon per-domain \
         fallback runs the scalar CROWN on worker threads and leaves both at 0, \
         so this is that fallback — or the backward stopped running synchronously, \
         in which case re-point these probes rather than deleting them.",
        engagement.soa,
        engagement.reference_linear_nodes
    );
}

fn baseline_single_objective_bounds_4210(
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    child_a: &BoundedTensor,
    child_b: &BoundedTensor,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
) -> (
    CountingGemmEngine,
    (BoundedTensor, Option<LinearBounds>),
    (BoundedTensor, Option<LinearBounds>),
) {
    baseline_bounds_with_alpha_4210(graph, spec_matrix, child_a, child_b, root_node_bounds, None)
}

fn baseline_bounds_with_alpha_4210(
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    child_a: &BoundedTensor,
    child_b: &BoundedTensor,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
) -> (
    CountingGemmEngine,
    (BoundedTensor, Option<LinearBounds>),
    (BoundedTensor, Option<LinearBounds>),
) {
    let baseline_engine = CountingGemmEngine::new();
    let baseline_a = compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        child_a,
        spec_matrix,
        Some(&baseline_engine),
        root_node_bounds,
        None,
        alpha_state,
        None,
        None,
        None,
        false,
    )
    .expect("baseline child_a");
    let baseline_b = compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        child_b,
        spec_matrix,
        Some(&baseline_engine),
        root_node_bounds,
        None,
        alpha_state,
        None,
        None,
        None,
        false,
    )
    .expect("baseline child_b");
    (baseline_engine, baseline_a, baseline_b)
}

/// The batched box must be contained in the per-domain baseline box, up to the
/// DERIVED cross-lane certificate gap.
///
/// Direction: the batched lane's certificate is `γ_n^{f64}·S` against the
/// baseline's `γ_n^{f32}·S`, and (when root bounds are supplied) it additionally
/// re-propagates IBP through the sub-domain region, so it can only be TIGHTER.
/// A batched box that pokes out of the baseline box by more than the certificate
/// gap means the batched lane LOST a tightening it used to have.
///
/// Divergences are COLLECTED: an assertion that aborts at the first mismatch
/// leaves every later element unchecked.
fn assert_batched_within_baseline_4210(
    domains: &[GraphInputDomain],
    baseline_a: &(BoundedTensor, Option<LinearBounds>),
    baseline_b: &(BoundedTensor, Option<LinearBounds>),
    gap: &[LaneGap; 2],
) {
    assert!(!domains[0].needs_bounding && !domains[1].needs_bounding);
    let mut failures = Vec::new();
    for (idx, (label, baseline)) in [("child_a", baseline_a), ("child_b", baseline_b)]
        .into_iter()
        .enumerate()
    {
        let tol = gap[idx].affine;
        let (bl, bu) = (baseline.0.lower_scalar(), baseline.0.upper_scalar());
        let (dl, du) = (domains[idx].lower_bound, domains[idx].upper_bound);
        if dl < bl - tol {
            failures.push(format!(
                "{label} batched lower {dl} is below baseline lower {bl} by more than the \
                 derived cross-lane gap {tol:e}: the batched lane is LOOSER than the \
                 per-domain reference"
            ));
        }
        if du > bu + tol {
            failures.push(format!(
                "{label} batched upper {du} is above baseline upper {bu} by more than the \
                 derived cross-lane gap {tol:e}: the batched lane is LOOSER than the \
                 per-domain reference"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn deferred_domains_4210(child_a: BoundedTensor, child_b: BoundedTensor) -> Vec<GraphInputDomain> {
    vec![deferred_domain_4210(child_a), deferred_domain_4210(child_b)]
}

fn deferred_domain_4210(child: BoundedTensor) -> GraphInputDomain {
    GraphInputDomain {
        input_bounds: Arc::new(child),
        lower_bound: f32::NEG_INFINITY,
        upper_bound: f32::INFINITY,
        depth: 1,
        priority: 0.0,
        linear_bounds: None,
        needs_bounding: true,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

#[test]
fn test_bound_deferred_domains_batch_uses_batched_backward_kernel_4210() {
    let (graph, _root_node_bounds, spec_matrix, child_a, child_b) =
        build_batched_kernel_fixture_4210();
    let (_baseline_engine, baseline_a, baseline_b) =
        baseline_single_objective_bounds_4210(&graph, &spec_matrix, &child_a, &child_b, None);
    let gap = [
        lane_gap_4210(&graph, &spec_matrix, &child_a),
        lane_gap_4210(&graph, &spec_matrix, &child_b),
    ];
    let config = BetaCrownConfig::default();
    let mut domains = deferred_domains_4210(child_a.clone(), child_b.clone());
    let batch_engine = CountingGemmEngine::new();
    let (engagement, records) = rebound_with_instruments_4210(
        &mut domains,
        &graph,
        &spec_matrix,
        &batch_engine,
        None,
        None,
        &config,
    );

    assert_batched_lane_served_whole_batch_4210(engagement, &records, domains.len());
    assert_batched_within_baseline_4210(&domains, &baseline_a, &baseline_b, &gap);
    assert_contains_exact_range_4210(
        "child_a",
        domains[0].lower_bound,
        domains[0].upper_bound,
        &child_a,
        spec_matrix[[0, 0]],
    );
    assert_contains_exact_range_4210(
        "child_b",
        domains[1].lower_bound,
        domains[1].upper_bound,
        &child_b,
        spec_matrix[[0, 0]],
    );

    for (idx, (label, baseline, child)) in [
        ("child_a", &baseline_a, &child_a),
        ("child_b", &baseline_b, &child_b),
    ]
    .into_iter()
    .enumerate()
    {
        match (&domains[idx].linear_bounds, &baseline.1) {
            (Some(actual), Some(expected)) => {
                assert_linear_bounds_within_gap(label, actual, expected, &gap[idx], child);
            }
            (None, None) => {}
            (actual, expected) => panic!(
                "{label} linear bounds diverged: batched={} baseline={}",
                actual.is_some(),
                expected.is_some()
            ),
        }
    }
}

/// The batched backward path re-propagates IBP through the sub-domain input region
/// during `compute_constrained_forward_bounds`, producing tighter intermediate bounds
/// than the per-domain baseline (Mode A, ibp_enhancement=false) which uses root-level
/// alpha_node_bounds directly. Verify the batched path produces bounds at least as
/// tight as the per-domain baseline — and that the result is still SOUND.
#[test]
fn test_bound_deferred_domains_batch_tightens_with_root_bounds_4210() {
    let (graph, root_node_bounds, spec_matrix, child_a, child_b) =
        build_batched_kernel_fixture_4210();
    let (_baseline_engine, baseline_a, baseline_b) = baseline_single_objective_bounds_4210(
        &graph,
        &spec_matrix,
        &child_a,
        &child_b,
        Some(&root_node_bounds),
    );
    let gap = [
        lane_gap_4210(&graph, &spec_matrix, &child_a),
        lane_gap_4210(&graph, &spec_matrix, &child_b),
    ];
    let config = BetaCrownConfig::default();
    let mut domains = deferred_domains_4210(child_a.clone(), child_b.clone());
    let batch_engine = CountingGemmEngine::new();
    let (engagement, records) = rebound_with_instruments_4210(
        &mut domains,
        &graph,
        &spec_matrix,
        &batch_engine,
        Some(&root_node_bounds),
        None,
        &config,
    );

    assert_batched_lane_served_whole_batch_4210(engagement, &records, domains.len());
    assert_batched_within_baseline_4210(&domains, &baseline_a, &baseline_b, &gap);
    assert_contains_exact_range_4210(
        "child_a",
        domains[0].lower_bound,
        domains[0].upper_bound,
        &child_a,
        spec_matrix[[0, 0]],
    );
    assert_contains_exact_range_4210(
        "child_b",
        domains[1].lower_bound,
        domains[1].upper_bound,
        &child_b,
        spec_matrix[[0, 0]],
    );
}

/// Verify the batched kernel fires with alpha state. Part of #4210.
/// Before this fix, `alpha_state.is_some()` forced rayon par_iter fallback.
#[test]
fn test_bound_deferred_domains_batch_uses_batched_kernel_with_alpha_state_4210() {
    let (graph, root_node_bounds, spec_matrix, child_a, child_b) =
        build_batched_kernel_fixture_4210();

    // Build GraphAlphaState simulating warmup α-CROWN pass.
    let pre_relu = root_node_bounds.get("linear1").expect("pre-activation");
    let mut alpha_state = GraphAlphaState::new();
    alpha_state
        .add_relu_node("relu1", pre_relu, false)
        .expect("alpha");

    let (_baseline_engine, baseline_a, baseline_b) = baseline_bounds_with_alpha_4210(
        &graph,
        &spec_matrix,
        &child_a,
        &child_b,
        Some(&root_node_bounds),
        Some(&alpha_state),
    );
    let gap = [
        lane_gap_4210(&graph, &spec_matrix, &child_a),
        lane_gap_4210(&graph, &spec_matrix, &child_b),
    ];
    let config = BetaCrownConfig::default();
    let mut domains = deferred_domains_4210(child_a.clone(), child_b.clone());
    let batch_engine = CountingGemmEngine::new();
    let (engagement, records) = rebound_with_instruments_4210(
        &mut domains,
        &graph,
        &spec_matrix,
        &batch_engine,
        Some(&root_node_bounds),
        Some(&alpha_state),
        &config,
    );

    assert_batched_lane_served_whole_batch_4210(engagement, &records, domains.len());
    assert_batched_within_baseline_4210(&domains, &baseline_a, &baseline_b, &gap);
    assert_contains_exact_range_4210(
        "child_a",
        domains[0].lower_bound,
        domains[0].upper_bound,
        &child_a,
        spec_matrix[[0, 0]],
    );
    assert_contains_exact_range_4210(
        "child_b",
        domains[1].lower_bound,
        domains[1].upper_bound,
        &child_b,
        spec_matrix[[0, 0]],
    );

    for (idx, (label, baseline, child)) in [
        ("child_a", &baseline_a, &child_a),
        ("child_b", &baseline_b, &child_b),
    ]
    .into_iter()
    .enumerate()
    {
        match (&domains[idx].linear_bounds, &baseline.1) {
            (Some(actual), Some(expected)) => {
                assert_linear_bounds_within_gap(label, actual, expected, &gap[idx], child);
            }
            (actual, expected) => panic!(
                "{label} linear bounds diverged: batched={} baseline={}",
                actual.is_some(),
                expected.is_some()
            ),
        }
    }
}

/// Verify the batched kernel fires when ibp_enhancement=true. Part of #4210.
/// Before this fix, `ibp_enhancement` forced rayon par_iter fallback.
#[test]
fn test_bound_deferred_domains_batch_fires_with_ibp_enhancement_4210() {
    let (graph, root_node_bounds, spec_matrix, child_a, child_b) =
        build_batched_kernel_fixture_4210();

    // Non-ibp_enhancement baseline (the batched path uses warmup base_bounds
    // without fresh IBP forward, matching this baseline's mechanism).
    let (_baseline_engine, baseline_a, baseline_b) = baseline_single_objective_bounds_4210(
        &graph,
        &spec_matrix,
        &child_a,
        &child_b,
        Some(&root_node_bounds),
    );
    let gap = [
        lane_gap_4210(&graph, &spec_matrix, &child_a),
        lane_gap_4210(&graph, &spec_matrix, &child_b),
    ];
    let config = BetaCrownConfig {
        input_split_ibp_enhancement: true,
        ..BetaCrownConfig::default()
    };
    let mut domains = deferred_domains_4210(child_a.clone(), child_b.clone());
    let batch_engine = CountingGemmEngine::new();
    let (engagement, records) = rebound_with_instruments_4210(
        &mut domains,
        &graph,
        &spec_matrix,
        &batch_engine,
        Some(&root_node_bounds),
        None,
        &config,
    );

    assert_batched_lane_served_whole_batch_4210(engagement, &records, domains.len());
    assert!(!domains[0].needs_bounding && !domains[1].needs_bounding);
    // The batched path with ibp_enhancement=true still uses warmup base_bounds
    // (not fresh subdomain IBP), so bounds match the non-ibp_enhancement
    // baseline with root_node_bounds (at-least-as-tight due to constrained
    // forward bounds).
    assert_batched_within_baseline_4210(&domains, &baseline_a, &baseline_b, &gap);
    assert_contains_exact_range_4210(
        "child_a",
        domains[0].lower_bound,
        domains[0].upper_bound,
        &child_a,
        spec_matrix[[0, 0]],
    );
    assert_contains_exact_range_4210(
        "child_b",
        domains[1].lower_bound,
        domains[1].upper_bound,
        &child_b,
        spec_matrix[[0, 0]],
    );
}
