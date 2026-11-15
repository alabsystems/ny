// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-root-alpha (ENGINE_PROGRAM §2 V2) — the lane side of the joint search,
//! on RANDOM attention blocks (q/k/v projections, `q·kᵀ`, a softmax — both the
//! monolithic LSE form and the fixed-index decomposition with its MulBinary —
//! `attn·v`, a projection, a ReLU and a head), with FAULT-INJECTED backends
//! (this crate cannot name the real search; ny-cli composes it).
//!
//! * `joint_search_fault_injection_never_loosens_and_never_falsely_tightens` —
//!   garbage hints (NaN, ±inf, out-of-range, wrong ops, wrong lengths, family
//!   mismatches) and extreme valid ones: every value that reached the certified
//!   fold is finite and in `[0, 1]`, the never-read planes `[1]`/`[3]` never
//!   move, the publication is the per-row intersection with the incumbent
//!   (never looser on any row), a kept state is strictly better, the candidate
//!   bound is re-derived bit for bit by an independent certified pass over the
//!   recorded ingested state, and the published bound encloses the network's
//!   margin at sampled points (containment). Valid uniform hints must also
//!   ARRIVE: every entry matches a lane site and its value is what the
//!   certified fold used (the delivery half — without it an ingest gate that
//!   matched nothing would pass every other assertion).
//! * `joint_search_echo_of_the_incumbent_is_never_kept` — the strict keep rule
//!   on the real lane: a hint equal to the incumbent certifies bit-identically
//!   and is NOT kept.
//! * `joint_search_refusals_publish_the_incumbent` — a panicking, a refusing
//!   and a missing backend each publish exactly the incumbent.
//! * `joint_search_is_dark_by_default_and_the_ascent_is_bit_identical`.
//! * `joint_search_lowering_self_check_passes_and_catches_a_corrupted_copy`
//!   (G1 plus a mutation of the copy).
//! * `joint_search_keep_rule_and_intersection_pins` (mutation pins, including
//!   the publication rule: a disjoint pair WITHDRAWS the publication).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use ny_core::search::{
    BackendCaps, GraphAlphaHint, GraphRootOutcome, GraphRootRequest, GraphRootStats,
    GraphSearchNet, GraphSearchOp, Refusal, SearchBackend, SearchPrecision, Untrusted,
};
use ny_tensor::BoundedTensor;

use super::joint_search::{
    intersect_rows, keep_candidate, resolve_publication, JointSearchRecord, LAST_JOINT_SEARCH,
};
use super::search_lowering::{lower_graph, self_check, SiteKind};
use super::{score_margins, LAST_MARGIN_COMMIT};
use crate::bounds::{AlphaCrownConfig, AlphaSpecAscent, AlphaSpecEarlyExit, MarginAlphaSearch};
use crate::layers::{AbsLayer, BilinearCrownLayer, Layer, LinearLayer, ReLULayer, SoftmaxLayer};
use crate::network::core::GraphNetwork;
use crate::network::graph_root_search::with_test_backend;
use crate::network::GraphNode;

// ---------------------------------------------------------------------------
//  Fixtures
// ---------------------------------------------------------------------------

/// Tokens and model width of the attention block; the head has 4 classes per
/// token, so the output is `[T, 4]` (12 values).
const T: usize = 3;
const D: usize = 2;
const OUT: usize = T * 4;

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn values(&mut self, n: usize, scale: f64) -> Vec<f32> {
        (0..n).map(|_| self.uniform(-scale, scale) as f32).collect()
    }
}

fn linear(rng: &mut Lcg, n_out: usize, n_in: usize, scale: f64) -> Layer {
    let weight = Array2::from_shape_vec((n_out, n_in), rng.values(n_out * n_in, scale))
        .expect("weight shape");
    let bias = Array1::from(rng.values(n_out, 0.2));
    Layer::Linear(LinearLayer::new(weight, Some(bias)).expect("linear layer"))
}

fn input_box(seed: u64) -> BoundedTensor {
    let mut rng = Lcg(seed ^ 0xb0c5);
    let centre = rng.values(T * D, 1.0);
    let lower: Vec<f32> = centre.iter().map(|c| c - 0.3).collect();
    let upper: Vec<f32> = centre.iter().map(|c| c + 0.3).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[T, D]), lower).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[T, D]), upper).expect("upper"),
    )
    .expect("input box")
}

/// A random attention block over a `[T, D]` token box. `fixed_index` rewrites
/// the softmax into the fixed-index decomposition (Exp, ReduceSum, Reciprocal,
/// MulBinary, Clip), the vit `softmax: complex_fixed_index` graph.
fn attention_block(seed: u64, input: &BoundedTensor, fixed_index: bool) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("q", linear(&mut rng, D, D, 0.9)));
    graph.add_node(GraphNode::from_input("k", linear(&mut rng, D, D, 0.9)));
    graph.add_node(GraphNode::from_input("v", linear(&mut rng, D, D, 0.9)));
    graph.add_node(GraphNode::binary(
        "qk",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(0.7))),
        "q",
        "k",
    ));
    graph.add_node(GraphNode::new(
        "sm",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["qk".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "av",
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        "sm",
        "v",
    ));
    graph.add_node(GraphNode::new(
        "proj",
        linear(&mut rng, 3, D, 0.9),
        vec!["av".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "act",
        Layer::ReLU(ReLULayer),
        vec!["proj".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "head",
        linear(&mut rng, 4, 3, 1.0),
        vec!["act".to_string()],
    ));
    graph.set_output("head");
    if fixed_index {
        let report = graph
            .decompose_softmax_complex_fixed_index(input)
            .expect("the fixed-index rewrite runs");
        assert_eq!(
            report.decomposed,
            vec!["sm".to_string()],
            "the softmax must be decomposed (skipped: {:?})",
            report.skipped
        );
    }
    graph
}

/// `y[0] - y[j]` on the first token, thresholds far above reach: every row
/// stays unverified, so the search and the keep rule always engage.
fn margin_rows() -> Vec<AlphaSpecEarlyExit> {
    (1..4)
        .map(|j| {
            let mut objective = vec![0.0f32; OUT];
            objective[0] = 1.0;
            objective[j] = -1.0;
            AlphaSpecEarlyExit {
                objective,
                threshold: 50.0,
                verify_upper_bound: false,
            }
        })
        .collect()
}

fn lane_config(fixed_index: bool, search: MarginAlphaSearch) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations: 12,
        learning_rate: 0.3,
        early_stop_patience: 1000,
        adaptive_skip: false,
        root_alpha_margin: true,
        root_alpha_margin_gradient: true,
        matmul_alpha: true,
        alpha_spec_slots: 0,
        spec_ascent: AlphaSpecAscent::new(margin_rows()),
        margin_alpha_freeze_mul_reciprocal: fixed_index,
        margin_alpha_search: search,
        ..AlphaCrownConfig::default()
    }
}

// ---------------------------------------------------------------------------
//  Fault-injected backends
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// NaN, ±inf, out-of-range and random values, plus entries the gate must
    /// ignore (an unknown op, a wrong length, a family mismatch).
    Garbage(u64),
    /// Every parameter at one valid value.
    Uniform(f32),
    /// The request's own starting parameters (the incumbent), unchanged.
    Echo,
    Panic,
    Refuse,
}

struct Mock {
    mode: Mode,
    calls: AtomicUsize,
}

impl Mock {
    fn arc(mode: Mode) -> Arc<Self> {
        Arc::new(Self {
            mode,
            calls: AtomicUsize::new(0),
        })
    }
}

const GARBAGE: [f32; 12] = [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    -5.0,
    7.0,
    1e30,
    -0.0,
    0.0,
    1.0,
    0.5,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
];

impl SearchBackend for Mock {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps::Cpu {
            threads: 1,
            lanes: 1,
            precision: SearchPrecision::F32,
        }
    }

    fn graph_root_alpha(
        &self,
        net: &GraphSearchNet,
        request: &GraphRootRequest,
    ) -> Result<GraphRootOutcome, Refusal> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        match self.mode {
            Mode::Panic => panic!("fault-injected search backend panic"),
            Mode::Refuse => return Err(Refusal::Nothing("fault-injected refusal")),
            _ => {}
        }
        let sizes = net.tensor_sizes().map_err(Refusal::Invalid)?;
        let mut hint = GraphAlphaHint::default();
        let mut counter = 0u64;
        for (op, node) in net.ops.iter().enumerate() {
            let (family, len) = match node {
                GraphSearchOp::Relu { input } => (0usize, sizes[*input]),
                GraphSearchOp::Bilinear { m, n, k, .. } => (1, 4 * m * n * k),
                GraphSearchOp::MulBinary { a, .. } => (2, 2 * sizes[*a]),
                _ => continue,
            };
            let values: Vec<f32> = match self.mode {
                Mode::Uniform(r) => vec![r; len],
                Mode::Echo => {
                    let given = match family {
                        0 => &request.init.relu,
                        1 => &request.init.bilinear,
                        _ => &request.init.mul_binary,
                    };
                    given
                        .iter()
                        .find(|(o, v)| *o == op && v.len() == len)
                        .map(|(_, v)| v.clone())
                        .unwrap_or_default()
                }
                Mode::Garbage(seed) => (0..len)
                    .map(|_| {
                        counter += 1;
                        let h = (seed ^ counter).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
                        if h.is_multiple_of(3) {
                            GARBAGE[(h as usize / 3) % GARBAGE.len()]
                        } else {
                            (h as f32 / (1u64 << 24) as f32) * 7.0 - 3.0
                        }
                    })
                    .collect(),
                Mode::Panic | Mode::Refuse => unreachable!("handled above"),
            };
            if values.is_empty() {
                continue;
            }
            match family {
                0 => hint.relu.push((op, values)),
                1 => hint.bilinear.push((op, values)),
                _ => hint.mul_binary.push((op, values)),
            }
        }
        if matches!(self.mode, Mode::Garbage(_)) {
            hint.relu.push((usize::MAX, vec![0.5; 3]));
            if let Some((op, values)) = hint.bilinear.first().cloned() {
                hint.bilinear
                    .push((op, values[..values.len() - 1].to_vec()));
                hint.relu.push((op, values));
            }
        }
        Ok(GraphRootOutcome {
            alpha: Untrusted::new(hint),
            stats: GraphRootStats {
                iters: 1,
                evals: 2,
                search_secs: 0.0,
                init_slack: Untrusted::new(0.0),
                best_slack: Untrusted::new(0.0),
                rows: request.rows.len(),
                verified_init: 0,
                verified_best: 0,
                relu_sites: 0,
                bilinear_sites: 0,
                mul_sites: 0,
                threads: 1,
                stop: "iters",
            },
        })
    }
}

// ---------------------------------------------------------------------------
//  Driving the lane
// ---------------------------------------------------------------------------

struct Run {
    record: Option<JointSearchRecord>,
    artifact: super::super::DagAlphaCollectionArtifact,
}

fn run_lane(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &AlphaCrownConfig,
    backend: Option<Arc<dyn SearchBackend>>,
) -> Run {
    LAST_JOINT_SEARCH.with(|cell| cell.borrow_mut().take());
    let artifact = with_test_backend(backend, || {
        graph
            .propagate_dag_alpha_crown_collect_with_engine(input, config, None)
            .expect("the margin lane runs")
            .expect("the DAG optimizer path, not an early return")
    });
    Run {
        record: LAST_JOINT_SEARCH.with(|cell| cell.borrow_mut().take()),
        artifact,
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Every corner of the input box, its centre and a few interior points.
fn sample_points(input: &BoundedTensor, seed: u64) -> Vec<Vec<f32>> {
    let lo: Vec<f32> = input.lower().iter().copied().collect();
    let hi: Vec<f32> = input.upper().iter().copied().collect();
    let n = lo.len();
    let mut points: Vec<Vec<f32>> = (0..(1usize << n))
        .map(|mask| {
            (0..n)
                .map(|i| if (mask >> i) & 1 == 1 { hi[i] } else { lo[i] })
                .collect()
        })
        .collect();
    points.push(lo.iter().zip(&hi).map(|(l, u)| 0.5 * (l + u)).collect());
    let mut rng = Lcg(seed);
    for _ in 0..20 {
        points.push(
            lo.iter()
                .zip(&hi)
                .map(|(&l, &u)| {
                    let v = rng.uniform(f64::from(l), f64::from(u)) as f32;
                    v.clamp(l, u)
                })
                .collect(),
        );
    }
    points
}

/// The published row bounds enclose the network's own margin at every sample
/// (the rows are lower objectives, so the publication's bounds ARE the rows').
fn assert_contained(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    lower: &[f32],
    upper: &[f32],
    context: &str,
) -> usize {
    let rows = margin_rows();
    let mut compared = 0usize;
    for point in sample_points(input, 99) {
        let concrete = BoundedTensor::concrete(
            ArrayD::from_shape_vec(input.lower().raw_dim(), point).expect("point"),
        )
        .expect("concrete point");
        let exact = graph.propagate_ibp(&concrete).expect("point forward");
        let y_lo: Vec<f64> = exact.lower().iter().map(|v| f64::from(*v)).collect();
        let y_hi: Vec<f64> = exact.upper().iter().map(|v| f64::from(*v)).collect();
        for (r, row) in rows.iter().enumerate() {
            let (mut v_lo, mut v_hi, mut magnitude) = (0.0f64, 0.0f64, 0.0f64);
            for (c, &coefficient) in row.objective.iter().enumerate() {
                let c64 = f64::from(coefficient);
                if c64 >= 0.0 {
                    v_lo += c64 * y_lo[c];
                    v_hi += c64 * y_hi[c];
                } else {
                    v_lo += c64 * y_hi[c];
                    v_hi += c64 * y_lo[c];
                }
                magnitude += c64.abs() * y_lo[c].abs().max(y_hi[c].abs());
            }
            let tol = 1e-12 * (1.0 + magnitude);
            assert!(
                f64::from(lower[r]) <= v_hi + tol,
                "{context} row {r}: published lower {} above the network's margin \
                 [{v_lo}, {v_hi}] — a false-proof mechanism",
                lower[r]
            );
            assert!(
                f64::from(upper[r]) >= v_lo - tol,
                "{context} row {r}: published upper {} below the network's margin \
                 [{v_lo}, {v_hi}]",
                upper[r]
            );
            compared += 1;
        }
    }
    compared
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[test]
fn joint_search_fault_injection_never_loosens_and_never_falsely_tightens() {
    let _env = ny_test_utils::env::lock_env_shared();
    let mut kept_any = 0usize;
    let mut compared = 0usize;
    let mut relu_delivered = 0usize;
    for fixed_index in [false, true] {
        for seed in 1..=2u64 {
            let input = input_box(seed);
            let graph = attention_block(seed, &input, fixed_index);
            let exec_order: Vec<String> = graph.exec_order().expect("exec order").to_vec();
            for mode in [
                Mode::Garbage(seed),
                Mode::Uniform(0.0),
                Mode::Uniform(1.0),
                Mode::Uniform(0.37),
            ] {
                let context = format!("fixed_index={fixed_index} seed={seed} mode={mode:?}");
                let mock = Mock::arc(mode);
                let config = lane_config(fixed_index, MarginAlphaSearch::JointF32);
                let run = run_lane(
                    &graph,
                    &input,
                    &config,
                    Some(Arc::clone(&mock) as Arc<dyn SearchBackend>),
                );
                let record = run.record.unwrap_or_else(|| {
                    panic!("{context}: the joint phase must run (no record = never entered)")
                });
                assert_eq!(
                    record.refused, None,
                    "{context}: the phase refused — a lowering or self-check refusal on an \
                     attention block is a finding, not a pass"
                );
                assert!(record.backend_called && mock.calls.load(Ordering::Relaxed) == 1);
                assert!(
                    record.selfcheck_compared > 0,
                    "{context}: G1 compared nodes"
                );

                let (inc_lo, inc_hi) = (&record.incumbent_lower, &record.incumbent_upper);
                let cand_lo = record
                    .candidate_lower
                    .as_ref()
                    .unwrap_or_else(|| panic!("{context}: one certified candidate fold"));
                let cand_hi = record.candidate_upper.as_ref().expect("candidate upper");
                let pub_lo = record.published_lower.as_ref().expect("a publication");
                let pub_hi = record.published_upper.as_ref().expect("a publication");

                // Never looser than the incumbent, row by row; exactly the
                // intersection of the two certified folds.
                for r in 0..pub_lo.len() {
                    assert!(pub_lo[r] >= inc_lo[r], "{context} row {r}: lower loosened");
                    assert!(pub_hi[r] <= inc_hi[r], "{context} row {r}: upper loosened");
                }
                let (want_lo, want_hi) = intersect_rows(inc_lo, inc_hi, cand_lo, cand_hi)
                    .expect("two certified enclosures of the same rows intersect");
                assert_eq!(bits(pub_lo), bits(&want_lo), "{context}");
                assert_eq!(bits(pub_hi), bits(&want_hi), "{context}");
                let publication = run
                    .artifact
                    .alpha_state
                    .margin_alpha_publication()
                    .unwrap_or_else(|| panic!("{context}: publication attached"));
                assert_eq!(bits(publication.lower()), bits(pub_lo), "{context}");
                assert_eq!(bits(publication.upper()), bits(pub_hi), "{context}");

                // The keep rule is strict.
                let thresholds: Vec<f32> = margin_rows().iter().map(|r| r.threshold).collect();
                let inc_score = score_margins(inc_lo, &thresholds);
                let cand_score = score_margins(cand_lo, &thresholds);
                assert_eq!(
                    record.kept,
                    keep_candidate(cand_score, inc_score),
                    "{context}: kept iff strictly better"
                );
                kept_any += usize::from(record.kept);

                // Only finite values in [0, 1] reached the certified fold; the
                // upper-direction planes never moved.
                let relu = record
                    .candidate_relu
                    .as_ref()
                    .expect("candidate relu state");
                for (name, alphas) in &relu.alphas {
                    assert!(
                        alphas
                            .iter()
                            .all(|a| a.is_finite() && (0.0..=1.0).contains(a)),
                        "{context}: relu α of {name} left [0, 1]"
                    );
                }
                let cand_bil = record.candidate_bilinear.as_ref().expect("product planes");
                let inc_bil = record
                    .incumbent_bilinear
                    .as_ref()
                    .expect("incumbent planes");
                assert_eq!(cand_bil.len(), 2, "{context}: both attention products");
                for (name, alphas) in cand_bil {
                    assert!(
                        alphas
                            .iter()
                            .all(|a| a.is_finite() && (0.0..=1.0).contains(a)),
                        "{context}: product planes of {name} left [0, 1]"
                    );
                    let before = &inc_bil[name];
                    for plane in [1usize, 3] {
                        let now: Vec<u32> = alphas
                            .index_axis(Axis(0), plane)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect();
                        let was: Vec<u32> = before
                            .index_axis(Axis(0), plane)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect();
                        assert_eq!(now, was, "{context}: plane {plane} of {name} moved");
                    }
                }
                if fixed_index {
                    let mul = record.candidate_mul.as_ref().expect("MulBinary planes");
                    assert!(
                        !mul.is_empty(),
                        "{context}: the softmax MulBinary is searched"
                    );
                    for (name, alphas) in mul {
                        assert!(
                            alphas
                                .iter()
                                .all(|a| a.is_finite() && (0.0..=1.0).contains(a)),
                            "{context}: MulBinary planes of {name} left [0, 1]"
                        );
                    }
                } else {
                    assert!(record.candidate_mul.is_none(), "{context}: no MulBinary");
                }
                if let Mode::Garbage(_) = mode {
                    let c = record.counts;
                    assert!(
                        c.relu_refused + c.bilinear_refused + c.mul_refused > 0,
                        "{context}: the non-finite values were refused: {c:?}"
                    );
                    assert!(
                        c.unmatched >= 3,
                        "{context}: unknown / short / mismatched entries were ignored: {c:?}"
                    );
                }
                // DELIVERY (review 2026-09-29): a valid hint must actually
                // reach the certified fold. Without this every assertion above
                // also holds when the ingest gate matches NOTHING (an op-index
                // or site-name drift turns every entry `unmatched`, the
                // candidate is the incumbent and the publication is the
                // incumbent) — a VOID arm that reads as a pass.
                if let Mode::Uniform(r) = mode {
                    let want = r.clamp(0.0, 1.0).to_bits();
                    let c = record.counts;
                    assert_eq!(
                        (
                            c.unmatched,
                            c.relu_refused,
                            c.bilinear_refused,
                            c.mul_refused
                        ),
                        (0, 0, 0, 0),
                        "{context}: every valid entry matched a lane site: {c:?}"
                    );
                    let product_planes: usize =
                        cand_bil.values().map(|alphas| alphas.len() / 2).sum();
                    assert_eq!(
                        c.bilinear_written, product_planes,
                        "{context}: planes [0]/[2] of every product were written: {c:?}"
                    );
                    for (name, alphas) in cand_bil {
                        for plane in [0usize, 2] {
                            assert!(
                                alphas
                                    .index_axis(Axis(0), plane)
                                    .iter()
                                    .all(|v| v.to_bits() == want),
                                "{context}: plane {plane} of {name} carries the hint"
                            );
                        }
                    }
                    if let Some(mul) = record.candidate_mul.as_ref() {
                        let entries: usize = mul.values().map(|alphas| alphas.len()).sum();
                        assert_eq!(c.mul_written, entries, "{context}: {c:?}");
                        assert!(
                            mul.values().flatten().all(|v| v.to_bits() == want),
                            "{context}: both MulBinary planes carry the hint"
                        );
                    }
                    // ReLU: exactly the neurons unstable in the certified box
                    // of the ReLU's input carry the hint.
                    let pre = run
                        .artifact
                        .reference_bounds
                        .get("proj")
                        .unwrap_or_else(|| panic!("{context}: certified box of 'proj'"));
                    let slopes = relu
                        .alphas
                        .get("act")
                        .unwrap_or_else(|| panic!("{context}: the ReLU's α"));
                    let unstable: Vec<usize> = pre
                        .lower()
                        .iter()
                        .zip(pre.upper().iter())
                        .enumerate()
                        .filter(|(_, (l, u))| **l < 0.0 && **u > 0.0)
                        .map(|(j, _)| j)
                        .collect();
                    assert_eq!(c.relu_written, unstable.len(), "{context}: {c:?}");
                    for &j in &unstable {
                        assert_eq!(
                            slopes[j].to_bits(),
                            want,
                            "{context}: unstable neuron {j} carries the hint"
                        );
                    }
                    relu_delivered += unstable.len();
                }

                // The candidate bound is ONE certified pass of exactly the
                // recorded ingested state (nothing untrusted leaked into it).
                let (fresh, _) = graph
                    .dag_alpha_margin_rows_pass(
                        &input,
                        &run.artifact.reference_bounds,
                        &exec_order,
                        run.artifact.output_bounds.len(),
                        input.len(),
                        &record.relu_name_to_idx,
                        relu,
                        None,
                        record.candidate_bilinear.as_ref(),
                        record.candidate_mul.as_ref(),
                        None,
                        &record.seed,
                        true,
                    )
                    .expect("independent certified pass");
                let fresh_lo: Vec<f32> = fresh.lower().iter().copied().collect();
                let fresh_hi: Vec<f32> = fresh.upper().iter().copied().collect();
                assert_eq!(bits(&fresh_lo), bits(cand_lo), "{context}: recompute");
                assert_eq!(bits(&fresh_hi), bits(cand_hi), "{context}: recompute");

                // Containment against the network itself.
                compared += assert_contained(&graph, &input, pub_lo, pub_hi, &context);
            }
        }
    }
    assert!(compared > 0, "containment compared nothing");
    assert!(
        relu_delivered > 0,
        "no unstable ReLU neuron received a hinted slope on any block: the ReLU family's \
         delivery went untested"
    );
    eprintln!(
        "[joint-search-test] kept={kept_any} containment_checks={compared} \
         relu_delivered={relu_delivered}"
    );
}

#[test]
fn joint_search_echo_of_the_incumbent_is_never_kept() {
    let _env = ny_test_utils::env::lock_env_shared();
    for fixed_index in [false, true] {
        let input = input_box(7);
        let graph = attention_block(7, &input, fixed_index);
        let config = lane_config(fixed_index, MarginAlphaSearch::JointF32);
        let run = run_lane(&graph, &input, &config, Some(Mock::arc(Mode::Echo)));
        let record = run.record.expect("the joint phase ran");
        assert_eq!(record.refused, None, "fixed_index={fixed_index}");
        let cand_lo = record.candidate_lower.as_ref().expect("a candidate fold");
        let cand_hi = record.candidate_upper.as_ref().expect("a candidate fold");
        assert_eq!(
            bits(cand_lo),
            bits(&record.incumbent_lower),
            "fixed_index={fixed_index}: the incumbent's own parameters certify bit-identically"
        );
        assert_eq!(bits(cand_hi), bits(&record.incumbent_upper));
        assert!(
            !record.kept,
            "fixed_index={fixed_index}: an equal score is NOT strictly better"
        );
        assert_eq!(
            bits(record.published_lower.as_ref().expect("published")),
            bits(&record.incumbent_lower)
        );
    }
}

#[test]
fn joint_search_refusals_publish_the_incumbent() {
    let _env = ny_test_utils::env::lock_env_shared();
    let input = input_box(11);
    let graph = attention_block(11, &input, true);
    let config = lane_config(true, MarginAlphaSearch::JointF32);
    let cases: [(Option<Arc<dyn SearchBackend>>, &str); 3] = [
        (Some(Mock::arc(Mode::Panic)), "the search backend panicked"),
        (Some(Mock::arc(Mode::Refuse)), "backend: nothing to search"),
        (None, "no_backend"),
    ];
    for (backend, token) in cases {
        let run = run_lane(&graph, &input, &config, backend);
        let record = run.record.expect("the joint phase ran");
        let refused = record.refused.clone().unwrap_or_default();
        assert!(
            refused.starts_with(token),
            "expected a `{token}` refusal, got {refused:?}"
        );
        assert!(!record.kept && record.candidate_lower.is_none());
        assert_eq!(
            bits(record.published_lower.as_ref().expect("published")),
            bits(&record.incumbent_lower),
            "{token}: the incumbent stands"
        );
        assert_eq!(
            bits(record.published_upper.as_ref().expect("published")),
            bits(&record.incumbent_upper)
        );
        let publication = run
            .artifact
            .alpha_state
            .margin_alpha_publication()
            .expect("the incumbent is still published");
        assert_eq!(bits(publication.lower()), bits(&record.incumbent_lower));
    }
}

#[test]
fn joint_search_is_dark_by_default_and_the_ascent_is_bit_identical() {
    let _env = ny_test_utils::env::lock_env_shared();
    assert_eq!(
        AlphaCrownConfig::default().margin_alpha_search,
        MarginAlphaSearch::Ascent,
        "V2 must be default-dark"
    );
    for fixed_index in [false, true] {
        let input = input_box(13);
        let graph = attention_block(13, &input, fixed_index);
        let config = lane_config(fixed_index, MarginAlphaSearch::Ascent);
        let panicking = Mock::arc(Mode::Panic);
        LAST_MARGIN_COMMIT.with(|cell| cell.borrow_mut().take());
        let with_backend = run_lane(
            &graph,
            &input,
            &config,
            Some(Arc::clone(&panicking) as Arc<dyn SearchBackend>),
        );
        assert!(
            with_backend.record.is_none(),
            "fixed_index={fixed_index}: the dark key never enters the joint phase"
        );
        assert_eq!(
            panicking.calls.load(Ordering::Relaxed),
            0,
            "the dark key never consults a backend"
        );
        assert!(
            LAST_MARGIN_COMMIT
                .with(|cell| cell.borrow_mut().take())
                .is_some(),
            "fixed_index={fixed_index}: the certified ascent ran"
        );
        let without = run_lane(&graph, &input, &config, None);
        let a = with_backend
            .artifact
            .alpha_state
            .margin_alpha_publication()
            .expect("the ascent publishes")
            .clone();
        let b = without
            .artifact
            .alpha_state
            .margin_alpha_publication()
            .expect("the ascent publishes")
            .clone();
        assert_eq!(
            bits(a.lower()),
            bits(b.lower()),
            "a registered backend changes nothing"
        );
        assert_eq!(bits(a.upper()), bits(b.upper()));
        assert_eq!(a.committed_iteration(), b.committed_iteration());
    }
}

#[test]
fn joint_search_lowering_self_check_passes_and_catches_a_corrupted_copy() {
    let _env = ny_test_utils::env::lock_env_shared();
    for fixed_index in [false, true] {
        let input = input_box(17);
        let graph = attention_block(17, &input, fixed_index);
        let node_bounds = graph.collect_node_bounds(&input).expect("IBP boxes");
        let exec_order: Vec<String> = graph.exec_order().expect("exec order").to_vec();
        let lowering = lower_graph(&graph, &exec_order, &input, &node_bounds)
            .unwrap_or_else(|e| panic!("fixed_index={fixed_index}: lowering refused: {e}"));
        let relu = lowering
            .sites
            .iter()
            .filter(|s| s.kind == SiteKind::Relu)
            .count();
        let products = lowering
            .sites
            .iter()
            .filter(|s| matches!(s.kind, SiteKind::Bilinear { .. }))
            .count();
        let muls = lowering
            .sites
            .iter()
            .filter(|s| matches!(s.kind, SiteKind::MulBinary { .. }))
            .count();
        assert_eq!((relu, products), (1, 2), "fixed_index={fixed_index}");
        assert_eq!(muls, usize::from(fixed_index), "fixed_index={fixed_index}");
        let check = self_check(&lowering, &graph, &input, None)
            .unwrap_or_else(|e| panic!("fixed_index={fixed_index}: G1 failed: {e}"));
        assert_eq!(
            check.compared,
            2 * lowering.tensor_of.len(),
            "every lowered node is compared at both points"
        );

        // Corrupt the copy: G1 must refuse it.
        let mut corrupted = lowering;
        let dense = corrupted
            .net
            .ops
            .iter_mut()
            .find_map(|op| match op {
                GraphSearchOp::Dense { weight, .. } => Some(weight),
                _ => None,
            })
            .expect("the copy has a dense layer");
        dense[0] += 0.5;
        assert!(
            self_check(&corrupted, &graph, &input, None).is_err(),
            "fixed_index={fixed_index}: a corrupted copy must fail the self-check"
        );
    }

    // An unsupported layer on the output's ancestry refuses the lowering.
    let input = input_box(19);
    let mut graph = attention_block(19, &input, false);
    graph.add_node(GraphNode::new(
        "abs",
        Layer::Abs(AbsLayer),
        vec!["head".to_string()],
    ));
    graph.set_output("abs");
    let node_bounds = graph.collect_node_bounds(&input).expect("IBP boxes");
    let exec_order: Vec<String> = graph.exec_order().expect("exec order").to_vec();
    let refused = lower_graph(&graph, &exec_order, &input, &node_bounds)
        .err()
        .expect("an unsupported layer must refuse the lowering");
    assert!(refused.contains("unsupported"), "{refused}");
}

#[test]
fn joint_search_keep_rule_and_intersection_pins() {
    let thresholds = [0.0f32, 0.0];
    let base = score_margins(&[-1.0, -2.0], &thresholds);
    assert!(!keep_candidate(base, base), "an equal score is never kept");
    assert!(keep_candidate(
        score_margins(&[-0.5, -2.0], &thresholds),
        base
    ));
    assert!(!keep_candidate(
        score_margins(&[-1.5, -2.0], &thresholds),
        base
    ));
    assert!(
        !keep_candidate(score_margins(&[f32::NAN, 5.0], &thresholds), base),
        "a NaN row is the worst row"
    );

    let (lo, hi) = intersect_rows(
        &[-1.0, -2.0, -3.0],
        &[4.0, 5.0, 6.0],
        &[-0.5, f32::NAN, -9.0],
        &[3.0, f32::NAN, 9.0],
    )
    .expect("overlapping enclosures");
    assert_eq!(
        lo,
        vec![-0.5, -2.0, -3.0],
        "tighter kept, NaN and looser ignored"
    );
    assert_eq!(hi, vec![3.0, 5.0, 6.0]);
    assert!(
        intersect_rows(&[0.0], &[1.0], &[2.0], &[3.0]).is_none(),
        "disjoint certified enclosures are a soundness alarm, never merged"
    );

    // The publication rule (review 2026-09-29): no candidate, or a malformed
    // one, leaves the incumbent; an overlapping one is intersected; a DISJOINT
    // one WITHDRAWS the publication instead of guessing that the incumbent is
    // the sound side.
    let (inc_lo, inc_hi) = ([-1.0f32, -2.0], [4.0f32, 5.0]);
    assert_eq!(
        resolve_publication(&inc_lo, &inc_hi, None),
        Some((inc_lo.to_vec(), inc_hi.to_vec()))
    );
    assert_eq!(
        resolve_publication(&inc_lo, &inc_hi, Some((&[9.0], &[9.0]))),
        Some((inc_lo.to_vec(), inc_hi.to_vec())),
        "a candidate of the wrong shape is ignored, never truncated into the publication"
    );
    assert_eq!(
        resolve_publication(&inc_lo, &inc_hi, Some((&[-0.5, -3.0], &[3.0, 6.0]))),
        Some((vec![-0.5, -2.0], vec![3.0, 5.0]))
    );
    assert_eq!(
        resolve_publication(&inc_lo, &inc_hi, Some((&[-0.5, 6.0], &[3.0, 7.0]))),
        None,
        "a disjoint row withdraws the whole publication"
    );
}
