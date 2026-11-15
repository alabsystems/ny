// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-lane-search (ENGINE_PROGRAM §2 M7) — the LANE side, with
//! FAULT-INJECTED backends (this crate cannot name the real search; ny-cli's
//! `real_cpu_f32_graph_lane_search_*` test composes it with `ny_bbe`).
//!
//! * `graph_lane_search_root_keep_rule_pins` / `graph_lane_search_child_keep_rule_pins`
//!   — the pure keep rules (mutation pins: a looser, NaN, shape-mismatched or
//!   disjoint candidate never moves a bound; an INVERTED root candidate is an
//!   alarm, never a raise of its lower alone; an upper never moves at the root
//!   and is never lowered on a child).
//! * `graph_lane_search_slope_gate_writes_only_finite_clamped_values_on_certified_unstable_neurons`
//!   and `graph_lane_search_multiplier_gate_writes_only_the_domains_own_literals` —
//!   the two ingest gates, value by value.
//! * `graph_lane_search_fault_injection_never_loosens_and_never_falsely_tightens` —
//!   the ROOT phase on random ReLU nets (MLP, residual) under garbage, extreme,
//!   echo and random hints: never looser than the incumbent on any row, the
//!   upper ends untouched, every raised lower re-derived BIT FOR BIT by an
//!   independent certified pass at the recorded ingested state, and every
//!   published lower below the network's true margin at sampled points
//!   (containment). Valid hints must ARRIVE (a gate that matched nothing would
//!   pass every other assertion).
//! * `graph_lane_search_waves_never_verify_a_sat_instance_and_engage` — the full
//!   multi-objective lane on the shared executor (CPU engine) with random and
//!   garbage hints: a property violated at a known point is never `Verified`,
//!   and the waves did search and certify (not VOID).
//! * `graph_lane_search_wave_candidates_are_contained_in_their_child_regions` —
//!   CHILD containment: every wave candidate the lane certified (β ingested)
//!   lower-bounds its row at every sampled point of the child's sub-region.
//! * `graph_lane_search_refusals_leave_the_lane_bit_identical` — no backend, a
//!   refusing backend, an empty hint and a PANICKING backend (contained to its
//!   phase) each reproduce the dark run exactly.
//! * `graph_lane_search_is_dark_by_default_and_off_never_consults_a_backend`.
//! * `graph_lane_search_lowering_self_check_passes_on_relu_graphs` (G1 on the
//!   graph lane's own graphs, residual included).
//! * `graph_lane_search_refuses_a_graph_too_large_for_the_search_copy` (the size
//!   guard runs before any lowering work).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::search::{
    BackendCaps, GraphAlphaHint, GraphBabOutcome, GraphBabRequest, GraphBabStats, GraphDomainHint,
    GraphSearchNet, GraphSearchOp, Refusal, SearchBackend, SearchPrecision, Untrusted,
};
use ny_core::NaiveCpuGemmEngine;
use ny_tensor::BoundedTensor;

use super::graph_search::{
    ingest_relu_slopes, ingest_split_multipliers, lower_and_check, merge_child_rows,
    merge_root_rows, run_root_phase, IngestCounts, RootMerge, RootPhase, CERTIFIED_CHILDREN,
    LAST_ROOT_CANDIDATE, MAX_LOWERED_ELEMENTS,
};
use super::shared::{build_spec_matrix, spec_bounds_to_vec};
use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::config::{BetaCrownConfig, GraphLaneSearch};
use crate::beta_crown::result::BabVerificationStatus;
use crate::beta_crown::state::GraphBetaState;
use crate::beta_crown::BetaCrownVerifier;
use crate::bounds::GraphAlphaState;
use crate::layers::{AddLayer, Layer, LinearLayer, ReLULayer};
use crate::network::graph_root_search::{graph_lane_search_counts, with_test_backend};
use crate::network::graph_search_lowering::{lower_graph, self_check, SiteKind};
use crate::network::{GraphNetwork, GraphNode, SpecCrownRequest};

/// The M7 counters are process-wide: serialize the tests that read them.
static COUNTERS: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
//  Fixtures
// ---------------------------------------------------------------------------

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

const IN: usize = 3;
const HIDDEN: usize = 6;
const CLASSES: usize = 3;

fn linear(rng: &mut Lcg, n_out: usize, n_in: usize, scale: f64) -> Layer {
    let weight = Array2::from_shape_vec((n_out, n_in), rng.values(n_out * n_in, scale))
        .expect("weight shape");
    let bias = Array1::from(rng.values(n_out, 0.2));
    Layer::Linear(LinearLayer::new(weight, Some(bias)).expect("linear layer"))
}

/// `x[3] → lin → relu → lin → relu (→ + relu0 when residual) → lin[3]`.
fn relu_net(seed: u64, residual: bool) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l0",
        linear(&mut rng, HIDDEN, IN, 1.0),
    ));
    graph.add_node(GraphNode::new(
        "r0",
        Layer::ReLU(ReLULayer),
        vec!["l0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, HIDDEN, HIDDEN, 0.8),
        vec!["r0".to_string()],
    ));
    let pre = if residual {
        graph.add_node(GraphNode::binary("skip", Layer::Add(AddLayer), "l1", "r0"));
        "skip"
    } else {
        "l1"
    };
    graph.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer),
        vec![pre.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        linear(&mut rng, CLASSES, HIDDEN, 0.8),
        vec!["r1".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn input_box(seed: u64, eps: f32) -> BoundedTensor {
    let mut rng = Lcg(seed ^ 0x1b0c);
    let centre = rng.values(IN, 1.0);
    let lower: Vec<f32> = centre.iter().map(|c| c - eps).collect();
    let upper: Vec<f32> = centre.iter().map(|c| c + eps).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[IN]), lower).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[IN]), upper).expect("upper"),
    )
    .expect("input box")
}

/// `y_target − y_c` margin rows.
fn margin_rows(target: usize) -> Vec<Vec<f32>> {
    (0..CLASSES)
        .filter(|&c| c != target)
        .map(|c| {
            let mut row = vec![0.0f32; CLASSES];
            row[target] = 1.0;
            row[c] = -1.0;
            row
        })
        .collect()
}

fn point_output(graph: &GraphNetwork, point: &[f32]) -> Vec<f64> {
    let concrete = BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[point.len()]), point.to_vec()).expect("point"),
    )
    .expect("concrete point");
    let out = graph.propagate_ibp(&concrete).expect("point forward");
    out.lower()
        .iter()
        .zip(out.upper().iter())
        .map(|(&l, &u)| f64::midpoint(f64::from(l), f64::from(u)))
        .collect()
}

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
    let mut rng = Lcg(seed);
    for _ in 0..200 {
        points.push(
            lo.iter()
                .zip(&hi)
                .map(|(&l, &u)| (rng.uniform(f64::from(l), f64::from(u)) as f32).clamp(l, u))
                .collect(),
        );
    }
    points
}

fn bits(v: &[(f32, f32)]) -> Vec<(u32, u32)> {
    v.iter().map(|(l, u)| (l.to_bits(), u.to_bits())).collect()
}

// ---------------------------------------------------------------------------
//  Fault-injected backends
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// NaN / ±inf / out-of-range slopes; NaN / negative / +inf / huge β.
    Garbage,
    /// Every slope 1e9 and -1e9 alternately (clamped to the ends), β 1e30.
    Extreme,
    /// Uniform random valid slopes and β in [0, 2].
    Random(u64),
    /// Slopes 0.5, β 0.
    Half,
    /// A hint with wrong ops and wrong widths only.
    Mismatched,
    /// `Err(Refusal)`.
    Refuse,
    /// `Ok` with every hint slot empty.
    Empty,
    /// Panics if consulted.
    Panic,
    /// Panics mid-search on an ARMED lane (the lane must contain it).
    Crash,
}

struct FaultBackend {
    mode: Mode,
    calls: AtomicUsize,
}

impl FaultBackend {
    fn arc(mode: Mode) -> Arc<Self> {
        Arc::new(Self {
            mode,
            calls: AtomicUsize::new(0),
        })
    }
}

fn relu_widths(net: &GraphSearchNet) -> Vec<(usize, usize)> {
    let sizes = net.tensor_sizes().expect("valid net");
    net.ops
        .iter()
        .enumerate()
        .filter_map(|(k, op)| match op {
            GraphSearchOp::Relu { input } => Some((k, sizes[*input])),
            _ => None,
        })
        .collect()
}

impl SearchBackend for FaultBackend {
    fn name(&self) -> &'static str {
        "fault"
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps::Cpu {
            threads: 1,
            lanes: 1,
            precision: SearchPrecision::F32,
        }
    }

    fn graph_bab_batch(
        &self,
        net: &GraphSearchNet,
        request: &GraphBabRequest,
    ) -> Result<GraphBabOutcome, Refusal> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let widths = relu_widths(net);
        let mut rng = Lcg(match self.mode {
            Mode::Random(seed) => seed,
            _ => 7,
        });
        let hint = |rng: &mut Lcg, n_splits: usize| -> Option<GraphDomainHint> {
            let slopes = |n: usize, rng: &mut Lcg| -> Vec<f32> {
                (0..n)
                    .map(|j| match self.mode {
                        Mode::Garbage => {
                            [f32::NAN, f32::INFINITY, -7.0, 3.5, f32::NEG_INFINITY][j % 5]
                        }
                        Mode::Extreme => {
                            if j % 2 == 0 {
                                1e9
                            } else {
                                -1e9
                            }
                        }
                        Mode::Random(_) => rng.unit() as f32,
                        _ => 0.5,
                    })
                    .collect()
            };
            match self.mode {
                Mode::Panic => panic!("the dark lane consulted the search backend"),
                Mode::Crash => {
                    panic!("fault backend crashes mid-search (the lane must contain it)")
                }
                Mode::Empty | Mode::Refuse => None,
                Mode::Mismatched => Some(GraphDomainHint {
                    alpha: GraphAlphaHint {
                        relu: widths
                            .iter()
                            .map(|&(op, n)| (op + 1000, vec![0.3; n]))
                            .chain(widths.iter().map(|&(op, n)| (op, vec![0.3; n + 1])))
                            .collect(),
                        ..GraphAlphaHint::default()
                    },
                    beta: vec![0.4; n_splits + 2],
                }),
                _ => Some(GraphDomainHint {
                    alpha: GraphAlphaHint {
                        relu: widths
                            .iter()
                            .map(|&(op, n)| (op, slopes(n, &mut *rng)))
                            .collect(),
                        ..GraphAlphaHint::default()
                    },
                    beta: (0..n_splits)
                        .map(|k| match self.mode {
                            Mode::Garbage => [f32::NAN, -5.0, f32::INFINITY, 1e30][k % 4],
                            Mode::Extreme => 1e30,
                            Mode::Random(_) => 2.0 * rng.unit() as f32,
                            _ => 0.0,
                        })
                        .collect(),
                }),
            }
        };
        if self.mode == Mode::Refuse {
            return Err(Refusal::Nothing("fault backend refuses"));
        }
        let hints: Vec<Option<GraphDomainHint>> = request
            .domains
            .iter()
            .map(|d| hint(&mut rng, d.splits.len()))
            .collect();
        let searched = hints.iter().filter(|h| h.is_some()).count();
        Ok(GraphBabOutcome {
            hints: Untrusted::new(hints),
            stats: GraphBabStats {
                domains: request.domains.len(),
                searched,
                refused: request.domains.len() - searched,
                late: 0,
                rows: request.rows.len(),
                relu_sites: widths.len(),
                splits: request.domains.iter().map(|d| d.splits.len()).sum(),
                threads: 1,
                search_secs: 0.0,
                per_domain: request.domains.iter().map(|_| None).collect(),
            },
        })
    }
}

fn lane_verifier(search: GraphLaneSearch) -> BetaCrownVerifier {
    BetaCrownVerifier::new(BetaCrownConfig {
        graph_search: search,
        timeout: std::time::Duration::from_secs(20),
        max_domains: 400,
        max_depth: 12,
        batch_size: 4,
        ..BetaCrownConfig::default()
    })
}

/// The lane's own heuristic-slope root incumbent (a certified spec CROWN pass).
fn incumbent(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    rows: &[Vec<f32>],
) -> Vec<(f32, f32)> {
    let spec = build_spec_matrix(rows).expect("spec");
    let bounds = SpecCrownRequest::new(graph, input, &spec, None)
        .node_bounds(node_bounds)
        .run()
        .expect("incumbent pass");
    spec_bounds_to_vec(&bounds)
}

// ---------------------------------------------------------------------------
//  Pure rules
// ---------------------------------------------------------------------------

#[test]
fn graph_lane_search_root_keep_rule_pins() {
    let base = vec![(-1.0f32, 2.0f32), (-0.5, 1.0), (0.3, 0.9)];
    // Shape mismatch: ignored, untouched.
    let mut b = base.clone();
    assert_eq!(
        merge_root_rows(&mut b, &[0, 1], &[(0.0, 1.0)]),
        RootMerge::Ignored
    );
    assert_eq!(
        merge_root_rows(&mut b, &[5], &[(0.0, 1.0)]),
        RootMerge::Ignored
    );
    assert_eq!(bits(&b), bits(&base));
    // Equal and looser candidates: not better, bit-identical.
    let mut b = base.clone();
    assert_eq!(
        merge_root_rows(&mut b, &[0, 1], &[(-1.0, 2.0), (-0.9, 1.5)]),
        RootMerge::NotBetter
    );
    assert_eq!(bits(&b), bits(&base));
    // NaN lower: never written.
    let mut b = base.clone();
    assert_eq!(
        merge_root_rows(&mut b, &[0], &[(f32::NAN, 1.0)]),
        RootMerge::NotBetter
    );
    assert_eq!(bits(&b), bits(&base));
    // NaN UPPER: the row is not an enclosure, so its finite, strictly better
    // lower is NOT read either (mutation pin: dropping the `cu` NaN check
    // raises row 0 to 0.5).
    let mut b = base.clone();
    assert_eq!(
        merge_root_rows(&mut b, &[0], &[(0.5, f32::NAN)]),
        RootMerge::NotBetter
    );
    assert_eq!(bits(&b), bits(&base), "a NaN-upper row is never read");
    // An infinite upper is a valid (vacuous) certified upper: the lower counts.
    let mut b = base.clone();
    assert_eq!(
        merge_root_rows(&mut b, &[0], &[(0.5, f32::INFINITY)]),
        RootMerge::Kept { raised: 1 }
    );
    assert_eq!(b[0], (0.5, 2.0));
    // An INVERTED candidate lying inside the incumbent (overlapping it on both
    // sides) is an EMPTY certified enclosure: an alarm, all-or-nothing — never
    // a raise of its lower alone, which would publish the ordered interval
    // [0.5, 2.0] past the lane's inverted-interval guard (mutation pin: drop
    // `cl > cu` and row 0 is raised to 0.5 with `Kept`).
    let mut b = base.clone();
    assert_eq!(
        merge_root_rows(&mut b, &[1, 0], &[(0.0, 0.5), (0.5, 0.3)]),
        RootMerge::Alarm { row: 0 }
    );
    assert_eq!(
        bits(&b),
        bits(&base),
        "an inverted candidate writes nothing"
    );
    // Strictly better rows raised, upper ends untouched.
    let mut b = base.clone();
    assert_eq!(
        merge_root_rows(&mut b, &[0, 1], &[(-0.25, 0.5), (-0.6, 0.8)]),
        RootMerge::Kept { raised: 1 }
    );
    assert_eq!(b[0], (-0.25, 2.0), "lower raised, upper kept");
    assert_eq!(b[1], base[1], "a looser lower never moves");
    assert_eq!(b[2], base[2], "a row outside `rows` never moves");
    // Disjoint pairs (candidate lower above the incumbent's certified upper,
    // candidate upper below its certified lower, an infinite lower) are alarms
    // and ALL-OR-NOTHING: the improving row 0 is not written either.
    for bad in [
        (2.5f32, 3.0f32),
        (-3.0, -2.0),
        (f32::INFINITY, f32::INFINITY),
    ] {
        let mut b = base.clone();
        assert_eq!(
            merge_root_rows(&mut b, &[0, 1], &[(0.0, 1.0), bad]),
            RootMerge::Alarm { row: 1 },
            "{bad:?}"
        );
        assert_eq!(bits(&b), bits(&base), "an alarm writes nothing");
    }
}

#[test]
fn graph_lane_search_child_keep_rule_pins() {
    let base = vec![(-1.0f32, 2.0f32), (-0.5, -0.2), (0.3, 0.9)];
    assert_eq!(merge_child_rows(&base, &[0, 1], &[(0.0, 1.0)]), None);
    assert_eq!(merge_child_rows(&base, &[7], &[(0.0, 1.0)]), None);
    assert_eq!(
        merge_child_rows(&base, &[0, 1], &[(-2.0, 5.0), (f32::NAN, 0.0)]),
        None,
        "looser and NaN lowers never move"
    );
    assert_eq!(
        merge_child_rows(&base, &[0], &[(f32::INFINITY, f32::INFINITY)]),
        None,
        "a non-finite lower is never written"
    );
    let (merged, raised) =
        merge_child_rows(&base, &[0, 1], &[(-0.5, -3.0), (0.1, -9.0)]).expect("better");
    assert_eq!(raised, 2);
    assert_eq!(merged[0], (-0.5, 2.0), "a child's upper is never lowered");
    assert_eq!(
        merged[1],
        (0.1, 0.1),
        "a raised lower past the uncertified upper lifts that upper to it"
    );
    assert_eq!(merged[2], base[2]);
}

// ---------------------------------------------------------------------------
//  The ingest gates
// ---------------------------------------------------------------------------

#[test]
fn graph_lane_search_slope_gate_writes_only_finite_clamped_values_on_certified_unstable_neurons() {
    let input = input_box(3, 0.6);
    let graph = relu_net(3, false);
    let node_bounds = graph.collect_node_bounds(&input).expect("IBP boxes");
    let order: Vec<String> = graph.exec_order().expect("order").to_vec();
    let lowering = lower_graph(&graph, &order, &input, &node_bounds).expect("lowered");
    let relu: Vec<_> = lowering
        .sites
        .iter()
        .filter(|s| s.kind == SiteKind::Relu)
        .collect();
    assert_eq!(relu.len(), 2, "both ReLUs are α sites");
    let site = relu[0];
    let pre = &node_bounds[&site.input_node];
    let (lo, hi): (Vec<f32>, Vec<f32>) = (
        pre.lower().iter().copied().collect(),
        pre.upper().iter().copied().collect(),
    );
    let unstable: Vec<bool> = lo
        .iter()
        .zip(&hi)
        .map(|(&l, &u)| l < 0.0 && u > 0.0)
        .collect();
    assert!(
        unstable.iter().any(|&u| u),
        "fixture must have an unstable neuron"
    );
    let values: Vec<f32> = (0..lo.len())
        .map(|j| [f32::NAN, 7.0, -3.0, 0.25, f32::INFINITY, 0.75][j % 6])
        .collect();
    let mut state = GraphAlphaState::new();
    state
        .add_relu_node(&site.node, pre, false)
        .expect("relu node");
    let before = state
        .relu_alpha_pair(&site.node)
        .map(|(l, u)| (l.to_vec(), u.to_vec()));
    let (before_lower, before_upper) = before.expect("pair");
    let counts = ingest_relu_slopes(
        &mut state,
        &lowering.sites,
        &[
            (site.op, values.clone()),
            (site.op + 999, vec![0.5; lo.len()]),
            (relu[1].op, vec![0.5; 1]),
        ],
        &input,
        |n: &str| node_bounds.get(n),
    );
    assert_eq!(
        counts.unmatched, 2,
        "a wrong op and a wrong width are unmatched"
    );
    let (after_lower, after_upper) = state.relu_alpha_pair(&site.node).expect("pair");
    let mut written = 0usize;
    let mut refused = 0usize;
    for j in 0..lo.len() {
        if !unstable[j] {
            assert_eq!(
                after_lower[j].to_bits(),
                before_lower[j].to_bits(),
                "a neuron stable in the certified box is never written"
            );
            continue;
        }
        if values[j].is_finite() {
            assert_eq!(after_lower[j], values[j].clamp(0.0, 1.0));
            written += 1;
        } else {
            assert_eq!(after_lower[j].to_bits(), before_lower[j].to_bits());
            refused += 1;
        }
    }
    assert_eq!((counts.written, counts.refused), (written, refused));
    assert_eq!(
        after_upper.to_vec(),
        before_upper,
        "the upper path is never written"
    );
    assert!(after_lower.iter().all(|a| (0.0..=1.0).contains(a)));

    // A channel-only (narrower) entry is replaced by a full-width one first.
    let mut narrow = GraphAlphaState::new();
    narrow.add_relu_node(&site.node, pre, false).expect("node");
    narrow
        .spatial_shapes
        .insert(site.node.clone(), vec![1, 1, lo.len()]);
    let counts = ingest_relu_slopes(
        &mut narrow,
        &lowering.sites,
        &[(site.op, vec![0.5; lo.len()])],
        &input,
        |n: &str| node_bounds.get(n),
    );
    assert_eq!(counts.unmatched, 0);
    assert!(!narrow.spatial_shapes.contains_key(&site.node));
    assert_eq!(narrow.relu_len(&site.node), Some(lo.len()));
}

#[test]
fn graph_lane_search_multiplier_gate_writes_only_the_domains_own_literals() {
    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint::new("r0".into(), 1, true, 0.0).expect("c"));
    history.add_constraint(GraphNeuronConstraint::new("r1".into(), 2, false, 0.0).expect("c"));
    history.add_constraint(GraphNeuronConstraint::new("r1".into(), 4, true, 0.0).expect("c"));
    let mut beta = GraphBetaState::from_history(&history).expect("beta");
    let counts = ingest_split_multipliers(
        &mut beta,
        &history.constraints,
        &[-5.0, f32::NAN, 0.75, 9.0],
    );
    assert_eq!(
        counts,
        IngestCounts {
            written: 2,
            refused: 1,
            unmatched: 1,
        }
    );
    assert_eq!(
        beta.entry("r0", 1).expect("r0").value(),
        0.0,
        "clamped to ≥ 0"
    );
    assert_eq!(
        beta.entry("r1", 2).expect("r1").value(),
        0.0,
        "NaN keeps the lane's"
    );
    assert_eq!(beta.entry("r1", 4).expect("r1").value(), 0.75);
    assert!(beta.entry("r0", 0).is_none(), "no literal, no entry");
}

// ---------------------------------------------------------------------------
//  The root phase under fault injection
// ---------------------------------------------------------------------------

#[test]
fn graph_lane_search_fault_injection_never_loosens_and_never_falsely_tightens() {
    let _env = ny_test_utils::env::lock_env_shared();
    let _serial = COUNTERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let modes = [
        Mode::Garbage,
        Mode::Extreme,
        Mode::Half,
        Mode::Random(11),
        Mode::Random(12),
        Mode::Mismatched,
    ];
    let mut delivered = 0usize;
    let mut kept_rows = 0usize;
    let mut compared = 0usize;
    for seed in 0..6u64 {
        for residual in [false, true] {
            let input = input_box(seed, 0.5);
            let graph = relu_net(40 + seed, residual);
            let node_bounds = graph.collect_node_bounds(&input).expect("IBP boxes");
            let rows = margin_rows((seed % 3) as usize);
            let thresholds = vec![0.0f32; rows.len()];
            let incumbent = incumbent(&graph, &input, &node_bounds, &rows);
            let verifier = lane_verifier(GraphLaneSearch::CpuF32);
            for mode in modes {
                let before = graph_lane_search_counts();
                let mut published = incumbent.clone();
                let backend = FaultBackend::arc(mode) as Arc<dyn SearchBackend>;
                let outcome = with_test_backend(Some(backend), || {
                    run_root_phase(
                        &RootPhase {
                            verifier: &verifier,
                            graph: &graph,
                            input: &input,
                            objectives: &rows,
                            thresholds: &thresholds,
                            node_bounds: &node_bounds,
                            incumbent_alpha: None,
                            deadline: None,
                        },
                        &mut published,
                    )
                });
                let ctx = format!("seed {seed} residual {residual} {mode:?}");
                assert_eq!(outcome.alarm_row(), None, "{ctx}: no alarm on a sound lane");
                assert_eq!(
                    graph_lane_search_counts().root_phases,
                    before.root_phases + 1
                );
                for (r, (&(pl, pu), &(il, iu))) in published.iter().zip(&incumbent).enumerate() {
                    assert!(
                        pl >= il,
                        "{ctx} row {r}: looser than the incumbent ({pl} < {il})"
                    );
                    assert_eq!(pu.to_bits(), iu.to_bits(), "{ctx} row {r}: an upper moved");
                }
                let open = incumbent
                    .iter()
                    .zip(&thresholds)
                    .filter(|((l, _), t)| l.partial_cmp(t) != Some(std::cmp::Ordering::Greater))
                    .count();
                let record = LAST_ROOT_CANDIDATE.with(|cell| cell.borrow_mut().take());
                if open > 0 && mode != Mode::Mismatched {
                    let Some(record) = record else {
                        // Only a hint with no finite value on any certified
                        // unstable neuron may skip the certified pass.
                        assert_eq!(
                            outcome.refused.as_deref(),
                            Some("nothing_ingested"),
                            "{ctx}: no certified candidate"
                        );
                        assert_eq!(bits(&published), bits(&incumbent), "{ctx}: refused");
                        continue;
                    };
                    // Valid positions must ARRIVE (the gate matched the sites).
                    assert_eq!(
                        record.ingest.unmatched, 0,
                        "{ctx}: the hint matched no site"
                    );
                    delivered += usize::from(record.ingest.written > 0);
                    // Independent certified pass at the recorded ingested state.
                    let open_rows: Vec<Vec<f32>> =
                        record.rows.iter().map(|&r| rows[r].clone()).collect();
                    let spec = build_spec_matrix(&open_rows).expect("spec");
                    // Same route as the phase (a live deadline keeps the
                    // bounds-only fast candidates off, as it did there).
                    let again = SpecCrownRequest::new(&graph, &input, &spec, None)
                        .node_bounds(&node_bounds)
                        .alpha_state_opt(Some(&record.state))
                        .deadline_opt(Some(
                            std::time::Instant::now() + std::time::Duration::from_secs(120),
                        ))
                        .run()
                        .map(|b| spec_bounds_to_vec(&b))
                        .expect("independent pass");
                    assert_eq!(
                        bits(&again),
                        bits(&record.certified),
                        "{ctx}: re-derivation"
                    );
                    for (k, &r) in record.rows.iter().enumerate() {
                        let want = incumbent[r].0.max(record.certified[k].0);
                        assert_eq!(
                            published[r].0.to_bits(),
                            want.to_bits(),
                            "{ctx} row {r}: publication is max(incumbent, certified)"
                        );
                        kept_rows += usize::from(record.certified[k].0 > incumbent[r].0);
                    }
                } else if mode == Mode::Mismatched {
                    assert!(record.is_none(), "{ctx}: a mismatched hint was certified");
                    assert_eq!(bits(&published), bits(&incumbent), "{ctx}: nothing matched");
                }
                // Containment: every published lower is below the true margin.
                for point in sample_points(&input, seed ^ 0x5a) {
                    let y = point_output(&graph, &point);
                    for (r, row) in rows.iter().enumerate() {
                        let m: f64 = row.iter().zip(&y).map(|(&c, &v)| f64::from(c) * v).sum();
                        assert!(
                            f64::from(published[r].0) <= m + 1e-5 * (1.0 + m.abs()),
                            "{ctx} row {r}: published lower {} above the true margin {m} — \
                             a false-proof mechanism",
                            published[r].0
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(
        delivered > 20,
        "valid hints must reach the certified pass: {delivered}"
    );
    assert!(
        compared > 10_000,
        "containment compared too little: {compared}"
    );
    eprintln!("[graph-search-test] delivered={delivered} kept_rows={kept_rows}");
}

// ---------------------------------------------------------------------------
//  The full lane (shared executor on the CPU engine)
// ---------------------------------------------------------------------------

/// A property violated at the box centre: every row's threshold sits above the
/// row's value there, so no sound run may return `Verified`.
fn sat_instance(seed: u64) -> (GraphNetwork, BoundedTensor, Vec<Vec<f32>>, Vec<f32>) {
    let input = input_box(seed, 0.4);
    let graph = relu_net(90 + seed, seed % 2 == 1);
    let rows = margin_rows((seed % 3) as usize);
    let centre: Vec<f32> = input
        .lower()
        .iter()
        .zip(input.upper().iter())
        .map(|(l, u)| 0.5 * (l + u))
        .collect();
    let y = point_output(&graph, &centre);
    let thresholds = rows
        .iter()
        .map(|row| {
            let v: f64 = row.iter().zip(&y).map(|(&c, &v)| f64::from(c) * v).sum();
            (v + 0.05) as f32
        })
        .collect();
    (graph, input, rows, thresholds)
}

#[test]
fn graph_lane_search_waves_never_verify_a_sat_instance_and_engage() {
    let _env = ny_test_utils::env::lock_env_shared();
    let _serial = COUNTERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let engine = NaiveCpuGemmEngine;
    let before = graph_lane_search_counts();
    let mut runs = 0usize;
    for seed in 0..6u64 {
        let (graph, input, rows, thresholds) = sat_instance(seed);
        for mode in [Mode::Random(seed), Mode::Garbage, Mode::Extreme] {
            let verifier = lane_verifier(GraphLaneSearch::CpuF32);
            let backend = FaultBackend::arc(mode) as Arc<dyn SearchBackend>;
            let result = with_test_backend(Some(backend), || {
                verifier.verify_graph_relu_split_multi_objective_with_engine(
                    &graph,
                    &input,
                    &rows,
                    &thresholds,
                    Some(&engine),
                    None,
                )
            })
            .expect("the lane completes");
            assert!(
                !matches!(result.result, BabVerificationStatus::Verified),
                "seed {seed} {mode:?}: a property violated at the box centre was VERIFIED"
            );
            runs += 1;
        }
    }
    let after = graph_lane_search_counts();
    assert_eq!(runs, 18);
    assert!(
        after.root_phases > before.root_phases,
        "the root phase never ran"
    );
    assert!(after.waves > before.waves, "no armed wave ran");
    assert!(
        after.domains_certified > before.domains_certified,
        "no wave candidate was certified (VOID): {before:?} -> {after:?}"
    );
    assert_eq!(after.alarms, before.alarms, "no alarm on a sound lane");
}

/// The network output at `point` when `point` lies STRICTLY inside the
/// sub-region of `history` (every ReLU literal holds with a margin, so a
/// rounding tie is never counted), else `None`.
fn in_region_output(
    graph: &GraphNetwork,
    history: &GraphSplitHistory,
    point: &[f32],
) -> Option<Vec<f64>> {
    let concrete = BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[point.len()]), point.to_vec()).expect("point"),
    )
    .expect("concrete point");
    let values = graph.collect_node_bounds(&concrete).expect("point forward");
    for constraint in &history.constraints {
        let relu = graph.nodes.get(constraint.node_name())?;
        let pre = values.get(relu.inputs.first()?)?;
        let j = constraint.neuron_idx();
        let v = f64::midpoint(
            f64::from(*pre.lower().iter().nth(j)?),
            f64::from(*pre.upper().iter().nth(j)?),
        );
        let holds = if constraint.is_active() {
            v > 1e-6
        } else {
            v < -1e-6
        };
        if !holds {
            return None;
        }
    }
    Some(point_output(graph, point))
}

/// CHILD CONTAINMENT. Every candidate a wave certified (kept or not) must be
/// a sound LOWER bound of its row over the child's sub-region — the input box
/// intersected with the child's split literals — at every sampled point that
/// lies in that sub-region. The root phase has its own containment check
/// (`graph_lane_search_fault_injection_never_loosens_and_never_falsely_tightens`);
/// this is the wave's, where β multipliers are ingested. Random and extreme
/// hints; both the plain and the residual net. VOID guards: split-child
/// candidates must have been certified and in-region points compared.
#[test]
fn graph_lane_search_wave_candidates_are_contained_in_their_child_regions() {
    let _env = ny_test_utils::env::lock_env_shared();
    let _serial = COUNTERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let engine = NaiveCpuGemmEngine;
    let mut certified = 0usize;
    let mut split_records = 0usize;
    let mut compared = 0usize;
    for seed in 0..4u64 {
        let (graph, input, rows, thresholds) = sat_instance(seed);
        for mode in [Mode::Random(seed ^ 0x3c), Mode::Extreme] {
            CERTIFIED_CHILDREN.with(|cell| cell.borrow_mut().clear());
            let verifier = lane_verifier(GraphLaneSearch::CpuF32);
            let backend = FaultBackend::arc(mode) as Arc<dyn SearchBackend>;
            let _result = with_test_backend(Some(backend), || {
                verifier.verify_graph_relu_split_multi_objective_with_engine(
                    &graph,
                    &input,
                    &rows,
                    &thresholds,
                    Some(&engine),
                    None,
                )
            })
            .expect("the lane completes");
            let records = CERTIFIED_CHILDREN.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
            for record in &records {
                certified += 1;
                split_records += usize::from(!record.history.constraints.is_empty());
                // A shape mismatch is ignored by the keep rule (never
                // published), so only well-shaped candidates are checked.
                if record.rows.len() != record.certified.len() {
                    continue;
                }
                for point in sample_points(&record.input, seed ^ 0x77) {
                    let Some(y) = in_region_output(&graph, &record.history, &point) else {
                        continue;
                    };
                    for (k, &r) in record.rows.iter().enumerate() {
                        let m: f64 = rows[r]
                            .iter()
                            .zip(&y)
                            .map(|(&c, &v)| f64::from(c) * v)
                            .sum();
                        let lower = f64::from(record.certified[k].0);
                        assert!(
                            lower <= m + 1e-4 * (1.0 + m.abs()),
                            "seed {seed} {mode:?} row {r}: certified child lower {lower} above \
                             the true margin {m} inside the child's sub-region \
                             ({} literals) — a false-proof mechanism",
                            record.history.constraints.len()
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(
        certified > 0 && split_records > 0,
        "no split child candidate was certified (VOID): certified={certified} \
         split={split_records}"
    );
    assert!(
        compared >= 100,
        "containment compared too few in-region points (VOID): {compared}"
    );
    eprintln!(
        "[graph-lane-search-test] child containment certified={certified} \
         split={split_records} compared={compared}"
    );
}

fn run_lane(
    search: GraphLaneSearch,
    backend: Option<Arc<dyn SearchBackend>>,
    seed: u64,
) -> crate::beta_crown::result::BetaCrownResult {
    let engine = NaiveCpuGemmEngine;
    let input = input_box(seed, 0.35);
    let graph = relu_net(200 + seed, seed.is_multiple_of(2));
    let rows = margin_rows(1);
    let thresholds = vec![-0.4f32; rows.len()];
    let verifier = lane_verifier(search);
    with_test_backend(backend, || {
        verifier.verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &rows,
            &thresholds,
            Some(&engine),
            None,
        )
    })
    .expect("the lane completes")
}

fn same_run(
    a: &crate::beta_crown::result::BetaCrownResult,
    b: &crate::beta_crown::result::BetaCrownResult,
) -> bool {
    std::mem::discriminant(&a.result) == std::mem::discriminant(&b.result)
        && a.domains_explored == b.domains_explored
        && a.domains_verified == b.domains_verified
}

#[test]
fn graph_lane_search_refusals_leave_the_lane_bit_identical() {
    let _env = ny_test_utils::env::lock_env_shared();
    let _serial = COUNTERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for seed in 0..4u64 {
        let dark = run_lane(GraphLaneSearch::Off, None, seed);
        let before = graph_lane_search_counts();
        for (label, backend) in [
            ("no_backend", None),
            (
                "refuse",
                Some(FaultBackend::arc(Mode::Refuse) as Arc<dyn SearchBackend>),
            ),
            (
                "empty",
                Some(FaultBackend::arc(Mode::Empty) as Arc<dyn SearchBackend>),
            ),
            // A PANICKING backend is contained to its phase (`backend_panicked`)
            // and must leave the lane exactly as a refusal does — it may not
            // unwind through the lane and take the lane's own verdict with it.
            (
                "crash",
                Some(FaultBackend::arc(Mode::Crash) as Arc<dyn SearchBackend>),
            ),
        ] {
            let armed = run_lane(GraphLaneSearch::CpuF32, backend, seed);
            assert!(
                same_run(&dark, &armed),
                "seed {seed} {label}: a refusal changed the run: dark {:?}/{}/{} vs armed \
                 {:?}/{}/{}",
                dark.result,
                dark.domains_explored,
                dark.domains_verified,
                armed.result,
                armed.domains_explored,
                armed.domains_verified
            );
        }
        let after = graph_lane_search_counts();
        assert_eq!(after.root_kept, before.root_kept, "a refusal kept nothing");
        assert_eq!(
            after.domains_kept, before.domains_kept,
            "a refusal kept nothing"
        );
    }
}

#[test]
fn graph_lane_search_is_dark_by_default_and_off_never_consults_a_backend() {
    let _env = ny_test_utils::env::lock_env_shared();
    let _serial = COUNTERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = BetaCrownConfig::default();
    assert_eq!(config.graph_search, GraphLaneSearch::Off);
    assert!(!config.graph_search.is_armed());
    assert_eq!(GraphLaneSearch::TOKENS, ["off", "cpu_f32"]);
    assert_eq!(GraphLaneSearch::CpuF32.label(), "cpu_f32");
    let absent: BetaCrownConfig =
        serde_json::from_value(serde_json::json!({})).expect("empty config deserializes");
    assert_eq!(absent.graph_search, GraphLaneSearch::Off);
    let armed: GraphLaneSearch =
        serde_json::from_value(serde_json::json!("cpu_f32")).expect("token");
    assert_eq!(armed, GraphLaneSearch::CpuF32);
    for bad in ["CpuF32", "cpu", "on", "true"] {
        assert!(
            serde_json::from_value::<GraphLaneSearch>(serde_json::json!(bad)).is_err(),
            "`{bad}` must not parse"
        );
    }
    let before = graph_lane_search_counts();
    for seed in 0..3u64 {
        let plain = run_lane(GraphLaneSearch::Off, None, seed);
        let panicking = FaultBackend::arc(Mode::Panic);
        let with_backend = run_lane(
            GraphLaneSearch::Off,
            Some(Arc::clone(&panicking) as Arc<dyn SearchBackend>),
            seed,
        );
        assert_eq!(
            panicking.calls.load(Ordering::Relaxed),
            0,
            "`off` consulted the backend"
        );
        assert!(
            same_run(&plain, &with_backend),
            "seed {seed}: `off` is not bit-identical"
        );
    }
    assert_eq!(
        graph_lane_search_counts(),
        before,
        "`off` entered the M7 lane"
    );
}

#[test]
fn graph_lane_search_lowering_self_check_passes_on_relu_graphs() {
    let _env = ny_test_utils::env::lock_env_shared();
    for residual in [false, true] {
        let input = input_box(5, 0.5);
        let graph = relu_net(5, residual);
        let node_bounds = graph.collect_node_bounds(&input).expect("IBP boxes");
        let order: Vec<String> = graph.exec_order().expect("order").to_vec();
        let lowering = lower_graph(&graph, &order, &input, &node_bounds)
            .unwrap_or_else(|e| panic!("residual={residual}: lowering refused: {e}"));
        let check = self_check(&lowering, &graph, &input, None).expect("G1 passes");
        assert!(check.compared > 0 && check.worst < 1e-4, "{check:?}");
        assert_eq!(
            lowering
                .sites
                .iter()
                .filter(|s| s.kind == SiteKind::Relu)
                .count(),
            2
        );
    }
}

#[test]
fn graph_lane_search_refuses_a_graph_too_large_for_the_search_copy() {
    let input = input_box(9, 0.5);
    let graph = relu_net(9, false);
    let mut node_bounds = graph.collect_node_bounds(&input).expect("IBP boxes");
    assert!(
        lower_and_check(&graph, &input, &node_bounds).is_ok(),
        "the small graph lowers"
    );
    // A box map past the cap (the size guard runs before any lowering work).
    let big = MAX_LOWERED_ELEMENTS + 1;
    let zeros = ArrayD::<f32>::zeros(IxDyn(&[big]));
    node_bounds.insert(
        "huge".to_string(),
        BoundedTensor::new(zeros.clone(), zeros).expect("big box"),
    );
    let refused = lower_and_check(&graph, &input, &node_bounds)
        .err()
        .expect("a graph past the cap is refused");
    assert!(
        refused.starts_with("graph_too_large_for_search_copy"),
        "{refused}"
    );
}
