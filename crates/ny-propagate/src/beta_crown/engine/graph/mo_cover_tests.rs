// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-v3-bankable: the multi-objective cover certification system, end to
//! end and at its seams.
//!
//! * the policy is dark by default and reads the typed value through its
//!   governed declaration;
//! * `off` is the historical lane (same verdicts and counters as `ledger` on
//!   every run the ledger proves, no frame armed, nothing captured);
//! * `ledger` on the REAL multi-objective driver: never refuses a genuine run,
//!   and every batched `Verified` leaves one cover per row that the instance
//!   door accepts for exactly the property; the sequential lane is UNBUILT
//!   once it branches (a root-terminal close still proves);
//! * RED-first, the dropped-child class: a lane `Verified` resting on one
//!   child of a split is WITHDRAWN (the legacy door published it);
//! * RED-first, the reuse class (`#warm-dag`, `#2904`): a leaf whose lane
//!   bound was tighter than certified passes the ledger (it trusts lane
//!   arithmetic) and is WITHDRAWN by the certify-once replay;
//! * the construction-time emptiness slack is exactly `with_constraint`'s
//!   decision;
//! * SAMPLING containment on a miniature vit (Conv patch embedding, CLS
//!   Concat, BatchNorm, Q/K/V, `Q K^T`, Softmax, `attn V`, residuals, ReLU MLP,
//!   ReduceMean, head): every replay bound on a random split region, under the
//!   default and under random `β >= 0`, lower-bounds the objective at every
//!   sampled point of the region, and no region containing a sample is ever
//!   certified empty.

use std::time::Duration;

use ndarray::{arr1, Array1, Array2};

use super::*;
use crate::beta_crown::{BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier};
use crate::certify::graph::CAPTURE_TEST_LOCK;
use crate::certify::{
    authorize_graph_instance, reset_graph_cover_capture, take_graph_cover_capture,
    GraphInstanceProperty, TokenScope,
};
use crate::zono_core::dag::{DagOpSet, DagProgram};
use crate::zono_core::tests::Rng;
use crate::zono_core::vit_root_tests::{mini_vit, mini_vit_input};
use crate::{AddLayer, BoundedTensor, GraphNode, Layer, LinearLayer, ReLULayer};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn unit_box() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite box")
}

/// An MLP `2 -> widths... -> 2` with f64 weights for enumeration.
struct Mlp2 {
    graph: GraphNetwork,
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl Mlp2 {
    fn from_layers(drawn: Vec<(Vec<Vec<f32>>, Vec<f32>)>) -> Self {
        let mut graph = GraphNetwork::new();
        let mut layers = Vec::new();
        let mut fan_in = 2usize;
        let mut prev: Option<String> = None;
        let count = drawn.len();
        for (i, (w, b)) in drawn.into_iter().enumerate() {
            let out = w.len();
            let flat: Vec<f32> = w.iter().flatten().copied().collect();
            let linear = LinearLayer::new(
                Array2::from_shape_vec((out, fan_in), flat).expect("weight shape"),
                Some(Array1::from(b.clone())),
            )
            .expect("finite linear layer");
            let name = format!("fc{i}");
            match &prev {
                None => graph.add_node(GraphNode::from_input(name.clone(), Layer::Linear(linear))),
                Some(p) => graph.add_node(GraphNode::new(
                    name.clone(),
                    Layer::Linear(linear),
                    vec![p.clone()],
                )),
            }
            layers.push((
                w.iter()
                    .map(|row| row.iter().map(|&v| f64::from(v)).collect())
                    .collect(),
                b.iter().map(|&v| f64::from(v)).collect(),
            ));
            prev = Some(name.clone());
            if i + 1 < count {
                let relu = format!("relu{i}");
                graph.add_node(GraphNode::new(
                    relu.clone(),
                    Layer::ReLU(ReLULayer),
                    vec![name],
                ));
                prev = Some(relu);
            }
            fan_in = out;
        }
        graph.set_output(prev.expect("at least one layer"));
        Self { graph, layers }
    }

    /// A random `2 -> widths -> 2` net (1/1024-grid weights).
    fn random(rng: &mut Rng, widths: &[usize]) -> Self {
        let mut fan_in = 2usize;
        let dims: Vec<usize> = widths.iter().copied().chain(std::iter::once(2)).collect();
        let mut drawn = Vec::new();
        #[allow(clippy::cast_possible_truncation)]
        let weight = |rng: &mut Rng| ((rng.sym() * 1024.0).round() / 1024.0) as f32;
        for &out in &dims {
            let w: Vec<Vec<f32>> = (0..out)
                .map(|_| (0..fan_in).map(|_| weight(rng)).collect())
                .collect();
            let b: Vec<f32> = (0..out).map(|_| weight(rng) * 0.5).collect();
            drawn.push((w, b));
            fan_in = out;
        }
        Self::from_layers(drawn)
    }

    /// Two outputs over three ReLUs whose zero lines cross inside the box:
    /// `y0 = 1 - relu(x1 + x2) - relu(x1 - x2)` (true min -1, root chord bound
    /// -2) and `y1 = 1 - relu(x1 - x2) - relu(-x1 + 0.5 x2)` (true min -1). A
    /// threshold of -1.5 on both rows needs a split tree.
    fn crossing2() -> Self {
        Self::from_layers(vec![
            (
                vec![vec![1.0, 1.0], vec![1.0, -1.0], vec![-1.0, 0.5]],
                vec![0.0, 0.0, 0.0],
            ),
            (
                vec![vec![-1.0, -1.0, 0.0], vec![0.0, -1.0, -1.0]],
                vec![1.0, 1.0],
            ),
        ])
    }

    fn eval(&self, x: [f64; 2]) -> Vec<f64> {
        let mut h: Vec<f64> = x.to_vec();
        let last = self.layers.len() - 1;
        for (i, (w, b)) in self.layers.iter().enumerate() {
            let mut next: Vec<f64> = w
                .iter()
                .zip(b)
                .map(|(row, bias)| row.iter().zip(&h).map(|(a, v)| a * v).sum::<f64>() + bias)
                .collect();
            if i < last {
                for v in &mut next {
                    *v = v.max(0.0);
                }
            }
            h = next;
        }
        h
    }

    /// Per output: a threshold STRICTLY below its minimum over `[-1, 1]^2`
    /// (grid minimum minus a Lipschitz bound on the grid error, minus `slack`
    /// times the output range).
    fn true_thresholds(&self, slack: f64) -> Vec<f32> {
        let n = 80;
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for i in 0..=n {
            for j in 0..=n {
                let y = self.eval([
                    -1.0 + 2.0 * f64::from(i) / f64::from(n),
                    -1.0 + 2.0 * f64::from(j) / f64::from(n),
                ]);
                for ((lo, hi), &v) in lo.iter_mut().zip(hi.iter_mut()).zip(&y) {
                    *lo = lo.min(v);
                    *hi = hi.max(v);
                }
            }
        }
        let lipschitz: f64 = self
            .layers
            .iter()
            .map(|(w, _)| {
                w.iter()
                    .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
                    .fold(0.0, f64::max)
            })
            .product();
        let grid = lipschitz * 2.0 / f64::from(n);
        #[allow(clippy::cast_possible_truncation)]
        lo.iter()
            .zip(&hi)
            .map(|(&lo, &hi)| (lo - grid - slack * (hi - lo) - 1e-3) as f32)
            .collect()
    }
}

/// The two rows `y0 > t0`, `y1 > t1` (disjunctive unsafe set `y0 <= t0 OR
/// y1 <= t1`), scaled by 0.625 (exact) so no other test's rows coincide.
fn two_rows() -> Vec<Vec<f32>> {
    vec![vec![0.625, 0.0], vec![0.0, 0.625]]
}

fn scaled(thresholds: &[f32]) -> Vec<f32> {
    thresholds.iter().map(|t| t * 0.625).collect()
}

fn base_config(mode: MoCoverCertify, batched: bool) -> BetaCrownConfig {
    BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 2000,
        max_depth: 24,
        timeout: Duration::from_secs(20),
        batch_size: if batched { 4 } else { 1 },
        mo_cover_certify: mode,
        ..Default::default()
    }
}

fn run_driver(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    mode: MoCoverCertify,
    batched: bool,
) -> (BetaCrownResult, BetaCrownVerifier) {
    let verifier = BetaCrownVerifier::new(base_config(mode, batched));
    let naive = ny_core::NaiveCpuGemmEngine;
    let engine: Option<&dyn ny_core::GemmEngine> = if batched { Some(&naive) } else { None };
    let result = verifier
        .verify_graph_relu_split_multi_objective_with_engine(
            graph, input, objectives, thresholds, engine, None,
        )
        .expect("the multi-objective driver runs");
    (result, verifier)
}

fn is_withdrawn(result: &BabVerificationStatus) -> bool {
    matches!(result, BabVerificationStatus::Unknown { reason }
        if reason.contains("graph cover gate refused") || reason.contains("mo-cover replay"))
}

fn property(objectives: &[Vec<f32>], thresholds: &[f32]) -> GraphInstanceProperty {
    let clauses = objectives
        .iter()
        .zip(thresholds)
        .map(|(row, &t)| vec![GraphObjective::new(row, t, false).expect("finite atom")])
        .collect();
    GraphInstanceProperty::new(vec![-1.0, -1.0], vec![1.0, 1.0], clauses).expect("well formed")
}

use crate::certify::GraphObjective;

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// Dark by default; the typed value is read through the governed declaration
/// (source `config`); the governed override wins in both directions; a
/// malformed override resolves to the declaration default (`off`), never to
/// the typed value; the vocabulary is closed and round-trips.
#[test]
fn mo_cover_policy_is_dark_by_default_and_reads_the_typed_value() {
    let decl = &ny_levers::decls::mo_cover::MO_COVER_CERTIFY;
    let resolve = |config: &BetaCrownConfig, raw: Option<&str>| {
        let raw = raw.map(str::to_string);
        policy_from_resolution(
            config,
            ny_levers::read_over_config_with(decl, |_| raw, configured_value(config)),
        )
    };
    let dark = BetaCrownConfig::default();
    assert_eq!(dark.mo_cover_certify, MoCoverCertify::Off);
    let policy = resolve(&dark, None);
    assert_eq!(
        (policy.mode, policy.source),
        (MoCoverCertify::Off, "default")
    );
    for mode in [MoCoverCertify::Ledger, MoCoverCertify::Replay] {
        let config = BetaCrownConfig {
            mo_cover_certify: mode,
            ..Default::default()
        };
        let policy = resolve(&config, None);
        assert_eq!((policy.mode, policy.source), (mode, "config"));
        // The governed override wins, in both directions.
        let policy = resolve(&config, Some("off"));
        assert_eq!(policy.mode, MoCoverCertify::Off);
        // A malformed override is a kill switch, not a pass-through.
        let policy = resolve(&config, Some("Ledger"));
        assert_eq!(
            (policy.mode, policy.source),
            (MoCoverCertify::Off, "legacy_env_rejected")
        );
    }
    let policy = resolve(&dark, Some("replay"));
    assert_eq!(
        (policy.mode, policy.source),
        (MoCoverCertify::Replay, "legacy_env")
    );
    for token in MoCoverCertify::TOKENS {
        let mode = MoCoverCertify::parse(token).expect("vocabulary");
        assert_eq!(mode.label(), token);
    }
    assert!(MoCoverCertify::parse("Replay").is_none());
    assert!(MoCoverCertify::parse("").is_none());
    assert!(!MoCoverCertify::Off.is_armed() && !MoCoverCertify::Off.replays());
    assert!(MoCoverCertify::Ledger.is_armed() && !MoCoverCertify::Ledger.replays());
    assert!(MoCoverCertify::Replay.is_armed() && MoCoverCertify::Replay.replays());
}

// ---------------------------------------------------------------------------
// The real driver: bit-identity of `off`, completeness of `ledger`, replay
// ---------------------------------------------------------------------------

/// `off` is the historical lane and `ledger` changes nothing a proof does:
/// on a family of small multi-row nets (random MLPs and the crafted
/// `crossing2`, which needs a split tree), the batched lane under `off` and
/// under `ledger` return the SAME status and the SAME counters on every run,
/// `off` arms no frame and captures nothing, `ledger` never refuses a genuine
/// run and every `Verified` leaves one cover per row that the instance door
/// accepts for exactly the property. The sequential lane is UNBUILT (nothing
/// captured, verdict unchanged). `replay` never publishes a `Verified` that
/// `off` did not, and confirms at least one split tree below the root.
#[ntest::timeout(600000)]
#[test]
fn mo_cover_off_is_bit_identical_and_ledger_proves_every_verified_run() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut rng = Rng::new(0x0A11_C0FE_E5EE_D001);
    let input = unit_box();
    let objectives = two_rows();
    let mut verified = 0usize;
    let mut deep = 0usize;
    let mut replay_confirmed_deep = 0usize;
    let mut replay_withdrawn = 0usize;
    let mut sequential_unbuilt = 0usize;
    let mut runs = 0usize;
    for net in 0..5 {
        let mlp = match net {
            0 | 2 => Mlp2::random(&mut rng, &[4]),
            1 | 3 => Mlp2::random(&mut rng, &[3, 3]),
            _ => Mlp2::crossing2(),
        };
        let ladders: Vec<Vec<f32>> = if net == 4 {
            vec![vec![-1.5, -1.5], vec![-1.25, -1.75]]
        } else {
            [0.02, 0.1, 0.3]
                .iter()
                .map(|&slack| mlp.true_thresholds(slack))
                .collect()
        };
        for raw in ladders {
            let thresholds = scaled(&raw);
            for batched in [true, false] {
                runs += 1;
                reset_graph_cover_capture();
                let (off, off_verifier) = run_driver(
                    &mlp.graph,
                    &input,
                    &objectives,
                    &thresholds,
                    MoCoverCertify::Off,
                    batched,
                );
                assert!(!off_verifier.graph_cover.is_armed());
                assert!(!off_verifier.mo_cover.is_armed());
                assert!(
                    take_graph_cover_capture().is_empty(),
                    "net={net} batched={batched}: `off` must capture nothing"
                );

                reset_graph_cover_capture();
                let (ledger, _) = run_driver(
                    &mlp.graph,
                    &input,
                    &objectives,
                    &thresholds,
                    MoCoverCertify::Ledger,
                    batched,
                );
                let capture = take_graph_cover_capture();
                assert!(
                    !is_withdrawn(&ledger.result),
                    "net={net} batched={batched} t={thresholds:?}: the ledger WITHDREW a genuine \
                     run: {:?}",
                    ledger.result
                );
                assert_eq!(
                    std::mem::discriminant(&off.result),
                    std::mem::discriminant(&ledger.result),
                    "net={net} batched={batched}: status changed: {:?} vs {:?}",
                    off.result,
                    ledger.result
                );
                assert_eq!(
                    (
                        off.domains_explored,
                        off.domains_verified,
                        off.max_depth_reached,
                        off.cuts_generated
                    ),
                    (
                        ledger.domains_explored,
                        ledger.domains_verified,
                        ledger.max_depth_reached,
                        ledger.cuts_generated
                    ),
                    "net={net} batched={batched}: counters changed"
                );
                if matches!(ledger.result, BabVerificationStatus::Verified) {
                    if batched {
                        verified += 1;
                        assert_eq!(capture.len(), 2, "one cover per row");
                        let token =
                            authorize_graph_instance(&property(&objectives, &thresholds), &capture)
                                .unwrap_or_else(|e| {
                                    panic!("net={net}: Verified but the instance door refused: {e}")
                                });
                        assert_eq!(token.scope(), TokenScope::Instance);
                        let atom = GraphObjective::new(&objectives[1], thresholds[1], false)
                            .expect("finite");
                        if capture.leaves_for(&atom).is_some_and(|n| n > 1) {
                            deep += 1;
                        }
                    } else if ledger.max_depth_reached > 0 {
                        // The sequential lane folds children without the
                        // frame: once it branched, the frame is UNBUILT.
                        assert!(
                            capture.is_empty(),
                            "net={net}: the sequential lane branched but a cover was captured"
                        );
                        sequential_unbuilt += 1;
                    } else if !capture.is_empty() {
                        // A ROOT-terminal close never reaches the sequential
                        // lane: it is recorded on its objective lowers and
                        // proved like any root close.
                        assert_eq!(capture.len(), 2, "one cover per row");
                        authorize_graph_instance(&property(&objectives, &thresholds), &capture)
                            .unwrap_or_else(|e| {
                                panic!("net={net}: root-terminal cover refused by the door: {e}")
                            });
                    }
                }

                if batched {
                    let (replay, replay_verifier) = run_driver(
                        &mlp.graph,
                        &input,
                        &objectives,
                        &thresholds,
                        MoCoverCertify::Replay,
                        true,
                    );
                    if matches!(replay.result, BabVerificationStatus::Verified) {
                        assert!(
                            matches!(off.result, BabVerificationStatus::Verified),
                            "net={net}: replay published a Verified the lane did not"
                        );
                        let report = replay_verifier
                            .mo_cover
                            .last_replay
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .clone()
                            .expect("a replay report for a proved cover");
                        assert!(report.all_confirmed(), "{report:?}");
                        if report.replayed > 0 {
                            replay_confirmed_deep += 1;
                        }
                    } else if is_withdrawn(&replay.result) {
                        replay_withdrawn += 1;
                        eprintln!(
                            "[mo-cover-family] net={net} t={thresholds:?} replay withdrew: {:?}",
                            replay.result
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "[mo-cover-family] runs={runs} batched_verified={verified} multi_leaf={deep} \
         replay_confirmed_below_root={replay_confirmed_deep} replay_withdrawn={replay_withdrawn} \
         sequential_unbuilt={sequential_unbuilt}"
    );
    assert!(verified >= 3, "only {verified} batched runs verified: VOID");
    assert!(
        deep >= 1,
        "no verified run built a split tree: the recording sites are VOID"
    );
    assert!(
        replay_confirmed_deep >= 1,
        "the replay never re-certified a region below the root: VOID"
    );
}

// ---------------------------------------------------------------------------
// RED-first defect injections
// ---------------------------------------------------------------------------

/// `out = Σ_i relu(x_i) + x_i` on `[-1, 1]^2` (range `[-2, 4]`): the ReLU sits
/// inside a residual block, the #warm-dag shape.
fn residual_graph() -> GraphNetwork {
    let linear = |w: Array2<f32>| Layer::Linear(LinearLayer::new(w, None).expect("linear"));
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        linear(ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0]])),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "add1",
        Layer::Add(AddLayer),
        vec!["relu1".to_string(), "linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        linear(ndarray::arr2(&[[1.0, 1.0]])),
        vec!["add1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

fn residual_root(
    graph: &GraphNetwork,
    lane_bounds: (f32, f32),
    thresholds: &[f32],
) -> MultiObjectiveGraphBabDomain {
    let node_bounds = graph
        .collect_node_bounds_with_engine(&unit_box(), None)
        .expect("ibp");
    MultiObjectiveGraphBabDomain::root(
        node_bounds,
        vec![lane_bounds],
        &unit_box(),
        thresholds,
        false,
    )
    .expect("root domain")
}

fn child(
    graph: &GraphNetwork,
    parent: &MultiObjectiveGraphBabDomain,
    active: bool,
    lane_bounds: (f32, f32),
    thresholds: &[f32],
) -> MultiObjectiveGraphBabDomain {
    let mut child = parent
        .with_constraint(
            graph,
            GraphNeuronConstraint {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                is_active: active,
                score: 1.0,
            },
            false,
            thresholds,
        )
        .expect("split")
        .expect("feasible half-space");
    child
        .update_bounds(vec![lane_bounds], thresholds, false)
        .expect("finite bounds");
    child
}

fn verified_result() -> BetaCrownResult {
    BetaCrownResult {
        result: BabVerificationStatus::Verified,
        domains_explored: 3,
        time_elapsed: Duration::ZERO,
        max_depth_reached: 1,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 2,
    }
}

fn policy(mode: MoCoverCertify) -> MoCoverPolicy {
    MoCoverPolicy {
        mode,
        source: "test",
    }
}

/// RED-first, the dropped-child class. The lane decided `Verified` on a TRUE
/// property (`out < 4.5`) while only ONE child of its root split was ever
/// closed on the record (the sibling was dropped by a bookkeeping path). The
/// legacy door published that `Verified`; the ledger WITHDRAWS it. With both
/// children recorded the same run proves and deposits its cover.
#[test]
fn mo_cover_ledger_withdraws_a_verified_whose_child_was_dropped() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let graph = residual_graph();
    let objectives = vec![vec![-1.0_f32]];
    let thresholds = vec![-4.5_f32];
    let root = residual_root(&graph, (-5.0, 5.0), &thresholds);
    assert!(!root.all_verified(), "precondition: the root is open");
    let active = child(&graph, &root, true, (-4.25, 2.0), &thresholds);
    let inactive = child(&graph, &root, false, (-4.25, 2.0), &thresholds);
    assert!(active.all_verified() && inactive.all_verified());

    for (with_sibling, expect_withdrawn) in [(false, true), (true, false)] {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            mo_cover_certify: MoCoverCertify::Ledger,
            ..Default::default()
        });
        let scope = verifier
            .mo_cover_arm(
                policy(MoCoverCertify::Ledger),
                &graph,
                &unit_box(),
                &objectives,
                &thresholds,
                false,
            )
            .expect("a disjunctive call arms");
        verifier.mo_cover_note_root(&graph, &root);
        verifier.mo_cover_note_domain(&active);
        if with_sibling {
            verifier.mo_cover_note_domain(&inactive);
        }
        reset_graph_cover_capture();
        let settled = verifier.mo_cover_settle(
            policy(MoCoverCertify::Ledger),
            scope,
            verified_result(),
            &objectives,
            &thresholds,
            None,
        );
        let capture = take_graph_cover_capture();
        assert!(!verifier.mo_cover.is_armed() && !verifier.graph_cover.is_armed());
        if expect_withdrawn {
            assert!(
                matches!(&settled.result, BabVerificationStatus::Unknown { reason }
                    if reason.contains("graph cover gate refused") && reason.contains("row 0")),
                "a dropped child must withdraw the verdict: {:?}",
                settled.result
            );
            assert!(capture.is_empty());
        } else {
            assert!(matches!(settled.result, BabVerificationStatus::Verified));
            assert_eq!(capture.len(), 1);
        }
    }
    // Conjunctive calls do not arm (a different aggregation).
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    assert!(verifier
        .mo_cover_arm(
            policy(MoCoverCertify::Ledger),
            &graph,
            &unit_box(),
            &objectives,
            &thresholds,
            true,
        )
        .is_none());
}

/// RED-first, the reuse class (`#warm-dag`: a seeded child bounded a
/// different function; `#2904`: an objective intersected with a raw box).
/// The lane "proved" `out < 3` (FALSE: `out` reaches 4 at `x = (1, 1)`) with
/// both children of `relu1[0]` carrying a lower bound on `-out` of -2.5 — the
/// active child's is tighter than the truth (-4). The ledger ACCEPTS the cover
/// (it trusts lane arithmetic: this is why the replay exists) and the verdict
/// would have published; the certify-once replay re-bounds both children from
/// the root with no reused state, confirms the inactive child (truly
/// `-out >= -2 > -3`) and WITHDRAWS the verdict on the active one.
#[test]
fn mo_cover_replay_withdraws_a_leaf_tighter_than_certified() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let graph = residual_graph();
    let objectives = vec![vec![-1.0_f32]];
    let thresholds = vec![-3.0_f32];
    let root = residual_root(&graph, (-4.0, 2.0), &thresholds);
    assert!(!root.all_verified());
    let active = child(&graph, &root, true, (-2.5, 0.0), &thresholds);
    let inactive = child(&graph, &root, false, (-2.5, 0.0), &thresholds);

    let settle = |mode: MoCoverCertify| {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            mo_cover_certify: mode,
            ..Default::default()
        });
        let scope = verifier
            .mo_cover_arm(
                policy(mode),
                &graph,
                &unit_box(),
                &objectives,
                &thresholds,
                false,
            )
            .expect("arms");
        verifier.mo_cover_note_root(&graph, &root);
        verifier.mo_cover_note_domain(&active);
        verifier.mo_cover_note_domain(&inactive);
        reset_graph_cover_capture();
        let settled = verifier.mo_cover_settle(
            policy(mode),
            scope,
            verified_result(),
            &objectives,
            &thresholds,
            None,
        );
        let capture = take_graph_cover_capture();
        let report = verifier
            .mo_cover
            .last_replay
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        (settled, capture.len(), report)
    };

    // RED: the ledger alone keeps the false Verified (and deposits a cover).
    let (ledger, deposited, _) = settle(MoCoverCertify::Ledger);
    assert!(matches!(ledger.result, BabVerificationStatus::Verified));
    assert_eq!(deposited, 1);

    // The replay withdraws it: one region confirmed, one not.
    let (replay, deposited, report) = settle(MoCoverCertify::Replay);
    let report = report.expect("a replay ran");
    assert!(
        matches!(&replay.result, BabVerificationStatus::Unknown { reason }
            if reason.contains("mo-cover replay")),
        "the replay must withdraw a leaf tighter than certified: {:?} {report:?}",
        replay.result
    );
    assert_eq!(deposited, 0, "a withdrawn verdict deposits nothing");
    assert_eq!(report.regions, 2);
    assert_eq!(report.replayed, 2);
    assert_eq!(
        report.confirmed, 1,
        "the inactive child is genuinely closed"
    );
    assert_eq!(report.unconfirmed.len(), 1);
    assert!(report.unconfirmed[0].contains("replay_lower"), "{report:?}");
}

/// A TRUE property on the same residual net (`out < 4.5`, true max 4) whose
/// lane leaves are honest: the replay confirms both children and the verdict
/// stands with its cover deposited (the replay is not a blanket refusal).
#[test]
fn mo_cover_replay_confirms_honest_leaves_and_keeps_the_verdict() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let graph = residual_graph();
    let objectives = vec![vec![-1.0_f32]];
    let thresholds = vec![-4.5_f32];
    let root = residual_root(&graph, (-5.0, 5.0), &thresholds);
    let active = child(&graph, &root, true, (-4.25, 2.0), &thresholds);
    let inactive = child(&graph, &root, false, (-4.25, 2.0), &thresholds);
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        mo_cover_certify: MoCoverCertify::Replay,
        ..Default::default()
    });
    let scope = verifier
        .mo_cover_arm(
            policy(MoCoverCertify::Replay),
            &graph,
            &unit_box(),
            &objectives,
            &thresholds,
            false,
        )
        .expect("arms");
    verifier.mo_cover_note_root(&graph, &root);
    verifier.mo_cover_note_domain(&active);
    verifier.mo_cover_note_domain(&inactive);
    reset_graph_cover_capture();
    let settled = verifier.mo_cover_settle(
        policy(MoCoverCertify::Replay),
        scope,
        verified_result(),
        &objectives,
        &thresholds,
        None,
    );
    let capture = take_graph_cover_capture();
    let report = verifier
        .mo_cover
        .last_replay
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("a replay ran");
    assert!(
        matches!(settled.result, BabVerificationStatus::Verified),
        "{:?} {report:?}",
        settled.result
    );
    assert_eq!((report.replayed, report.confirmed), (2, 2), "{report:?}");
    assert!(report.all_confirmed());
    assert_eq!(capture.len(), 1);
}

// ---------------------------------------------------------------------------
// Emptiness slack
// ---------------------------------------------------------------------------

/// The construction-time emptiness slack is exactly `with_constraint`'s
/// decision: for every neuron and side of a random root (IBP boxes of random
/// nets, both stable and unstable neurons), `with_constraint` returns
/// `Ok(None)` iff `split_emptiness_slack` is `Some`, and the slack is `-u` /
/// `l` of the certified pre-activation box.
#[test]
fn mo_cover_split_emptiness_slack_matches_with_constraint() {
    let mut rng = Rng::new(0x005E_A70F_E377);
    let mut empties = 0usize;
    let mut feasible = 0usize;
    for _ in 0..48 {
        let mlp = Mlp2::random(&mut rng, &[5, 4]);
        let input = unit_box();
        let node_bounds = mlp
            .graph
            .collect_node_bounds_with_engine(&input, None)
            .expect("ibp");
        let thresholds = vec![0.0_f32, 0.0];
        let root = MultiObjectiveGraphBabDomain::root(
            node_bounds.clone(),
            vec![(-1.0, 1.0), (-1.0, 1.0)],
            &input,
            &thresholds,
            false,
        )
        .expect("root");
        for (relu, pre) in [("relu0", "fc0"), ("relu1", "fc1")] {
            let pre_box = node_bounds[pre].flatten();
            for neuron in 0..pre_box.len() {
                let (l, u) = (pre_box.lower()[[neuron]], pre_box.upper()[[neuron]]);
                for active in [true, false] {
                    let constraint = GraphNeuronConstraint {
                        node_name: relu.to_string(),
                        neuron_idx: neuron,
                        is_active: active,
                        score: 1.0,
                    };
                    let slack = root.split_emptiness_slack(&mlp.graph, &constraint);
                    let built = root
                        .with_constraint(&mlp.graph, constraint, false, &thresholds)
                        .expect("no structural error");
                    assert_eq!(
                        built.is_none(),
                        slack.is_some(),
                        "{relu}[{neuron}] active={active}"
                    );
                    if let Some(slack) = slack {
                        empties += 1;
                        let expected = if active { -f64::from(u) } else { f64::from(l) };
                        assert_eq!(slack, expected);
                        assert!(slack > 0.0);
                    } else {
                        feasible += 1;
                    }
                }
            }
        }
    }
    assert!(
        empties >= 10 && feasible >= 10,
        "empties={empties} feasible={feasible}"
    );
}

// ---------------------------------------------------------------------------
// Sampling containment on a miniature vit
// ---------------------------------------------------------------------------

/// The replay's certified pass on random split regions of a miniature vit
/// (the vit_2023 op set as ny lowers it: Conv patch embedding, CLS Concat,
/// positional add, BatchNorm, Q/K/V, `Q K^T` and `attn V` BilinearCrown,
/// Softmax, residual Adds, ReLU MLP, ReduceMean, head). Each region is built
/// from literals a sampled point satisfies, so it is NON-EMPTY:
///
/// * the replay must never certify it empty (neither by a root-box slack nor
///   by a slack-bearing `InfeasibleDomain`);
/// * under the default β and under random `β >= 0` (any is sound by weak
///   duality), every published objective LOWER bound must sit at or below the
///   objective at every sampled point of the region (f64 point evaluation,
///   relative tolerance 1e-9).
#[ntest::timeout(600000)]
#[test]
fn mo_cover_mini_vit_replay_bounds_contain_samples_on_split_regions() {
    let graph = mini_vit(0x517E_0101);
    let input = mini_vit_input(0x517E_0102, 0.03);
    let (program, _) = DagProgram::compile_with_ibp_opset(&graph, &input, None, DagOpSet::Vit)
        .expect("the vit program compiles");
    let node_bounds = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("ibp");
    // Disjunctive rows y0 - y1, y0 - y2 (thresholds only steer priority).
    let objectives = vec![vec![1.0_f32, -1.0, 0.0], vec![1.0, 0.0, -1.0]];
    let thresholds = vec![0.0_f32, 0.0];
    let root = MultiObjectiveGraphBabDomain::root(
        node_bounds,
        vec![(-10.0, 10.0), (-10.0, 10.0)],
        &input,
        &thresholds,
        false,
    )
    .expect("root");
    let base = MoReplayBase { graph, root };
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let lo: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let hi: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    let mut rng = Rng::new(0x517E_0103);
    let sample = |rng: &mut Rng| -> Vec<f64> {
        lo.iter()
            .zip(&hi)
            .map(|(&l, &h)| (l + (h - l) * rng.unit()).clamp(l, h))
            .collect()
    };
    let evaluate = |point: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let mut fc1 = Vec::new();
        let mut head = Vec::new();
        program
            .evaluate_points(point, 1, |_, name, values| {
                if name == "fc1" {
                    fc1 = values.to_vec();
                } else if name == "head" {
                    head = values.to_vec();
                }
            })
            .expect("evaluate");
        (fc1, head)
    };
    let mut regions = 0usize;
    let mut checked = 0usize;
    let mut upper_misses = 0usize;
    for trial in 0..24 {
        let anchor = sample(&mut rng);
        let (fc1, _) = evaluate(&anchor);
        assert_eq!(fc1.len(), 40, "precondition: fc1 is [5, 8]");
        let depth = 1 + trial % 3;
        let mut key: Vec<(String, usize, bool)> = Vec::new();
        for _ in 0..depth {
            #[allow(clippy::cast_possible_truncation)]
            let neuron = (rng.next_u64() % 40) as usize;
            if fc1[neuron].abs() < 1e-6 || key.iter().any(|(_, n, _)| *n == neuron) {
                continue;
            }
            key.push(("act".to_string(), neuron, fc1[neuron] > 0.0));
        }
        if key.is_empty() {
            continue;
        }
        key.sort();
        let prepared = match verifier.mo_cover_replay_prepare(&base, &key, &thresholds) {
            Ok(prepared) => prepared,
            Err(RegionOutcome::ConfirmedEmpty) => {
                panic!("trial {trial}: a region containing a sample was certified EMPTY: {key:?}")
            }
            Err(other) => panic!("trial {trial}: the region did not build: {other:?}"),
        };
        regions += 1;
        let mut betas = vec![replay_beta(&prepared, None)];
        let mut random = replay_beta(&prepared, None);
        for entry in &mut random.entries {
            entry.set_value((rng.unit() * 2.0) as f32);
        }
        betas.push(random);
        for beta in &betas {
            let bounds = match verifier.mo_cover_replay_pass(
                &base,
                &prepared,
                beta,
                &objectives,
                &thresholds,
            ) {
                Ok(bounds) => bounds,
                Err(error) => {
                    assert!(
                        !error.is_infeasible_domain() || infeasibility_slack(&error).is_none(),
                        "trial {trial}: a region containing a sample was certified EMPTY \
                         by the pass: {error}"
                    );
                    panic!("trial {trial}: the replay pass failed: {error}");
                }
            };
            assert_eq!(bounds.len(), 2);
            for _ in 0..48 {
                let point = sample(&mut rng);
                let (fc1, head) = evaluate(&point);
                let inside = key.iter().all(|(_, neuron, active)| {
                    let z = fc1[*neuron];
                    if *active {
                        z >= 1e-9
                    } else {
                        z <= -1e-9
                    }
                });
                if !inside {
                    continue;
                }
                for (row, &(lower, upper)) in objectives.iter().zip(&bounds) {
                    let value: f64 = row.iter().zip(&head).map(|(&c, &y)| f64::from(c) * y).sum();
                    let tol = 1e-9 * (1.0 + value.abs());
                    assert!(
                        f64::from(lower) <= value + tol,
                        "trial {trial} region {key:?}: certified lower {lower:e} above the \
                         objective {value:e} at a sampled point of the region"
                    );
                    if f64::from(upper) + tol < value {
                        upper_misses += 1;
                    }
                    checked += 1;
                }
            }
        }
    }
    eprintln!(
        "[mo-cover-mini-vit] regions={regions} checked={checked} upper_misses={upper_misses}"
    );
    assert!(regions >= 12, "only {regions} regions built: VOID");
    assert!(
        checked >= 500,
        "only {checked} sampled values checked: VOID"
    );
}

// ---------------------------------------------------------------------------
// review(vit-v3): adversarial additions
// ---------------------------------------------------------------------------

/// A two-row root of [`residual_graph`] with the lane bounds `lane_bounds`.
fn residual_root_rows(
    graph: &GraphNetwork,
    lane_bounds: Vec<(f32, f32)>,
    thresholds: &[f32],
) -> MultiObjectiveGraphBabDomain {
    let node_bounds = graph
        .collect_node_bounds_with_engine(&unit_box(), None)
        .expect("ibp");
    MultiObjectiveGraphBabDomain::root(node_bounds, lane_bounds, &unit_box(), thresholds, false)
        .expect("root domain")
}

/// A child of `parent` on `relu1[0]` carrying the lane bounds `lane_bounds`.
fn residual_child_rows(
    graph: &GraphNetwork,
    parent: &MultiObjectiveGraphBabDomain,
    active: bool,
    lane_bounds: Vec<(f32, f32)>,
    thresholds: &[f32],
) -> MultiObjectiveGraphBabDomain {
    let mut child = parent
        .with_constraint(
            graph,
            GraphNeuronConstraint {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                is_active: active,
                score: 1.0,
            },
            false,
            thresholds,
        )
        .expect("split")
        .expect("feasible half-space");
    child
        .update_bounds(lane_bounds, thresholds, false)
        .expect("finite bounds");
    child
}

/// Arm, record the root and `children`, and settle one call under `mode`.
/// With `deadline`, it is BOTH the verifier's own deadline and the caller's:
/// the replay runs under the verifier's `alpha_config.deadline` (the
/// instance's internal budget, which `BetaCrownVerifier::new` always sets) and
/// falls back to the caller's BaB slice only without one. The settled result,
/// the number of covers it deposited, and the replay report (if one ran).
fn settle_tree(
    graph: &GraphNetwork,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    root: &MultiObjectiveGraphBabDomain,
    children: &[&MultiObjectiveGraphBabDomain],
    mode: MoCoverCertify,
    deadline: Option<Instant>,
) -> (BetaCrownResult, usize, Option<MoReplayReport>) {
    let mut verifier = BetaCrownVerifier::new(BetaCrownConfig {
        mo_cover_certify: mode,
        ..Default::default()
    });
    if deadline.is_some() {
        verifier.config.alpha_config.deadline = deadline;
    }
    let scope = verifier
        .mo_cover_arm(
            policy(mode),
            graph,
            &unit_box(),
            objectives,
            thresholds,
            false,
        )
        .expect("a disjunctive call arms");
    verifier.mo_cover_note_root(graph, root);
    for child in children {
        verifier.mo_cover_note_domain(child);
    }
    reset_graph_cover_capture();
    let settled = verifier.mo_cover_settle(
        policy(mode),
        scope,
        verified_result(),
        objectives,
        thresholds,
        deadline,
    );
    let deposited = take_graph_cover_capture().len();
    let report = verifier
        .mo_cover
        .last_replay
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    (settled, deposited, report)
}

/// review(vit-v3), RED under the mutant that replays only the FIRST row a
/// witness region closes. Two rows over the residual net: row 0 `-out >
/// -4.5` is TRUE (max `out` = 4), row 1 `-out/2 > -1.5` (`out < 3`) is FALSE
/// (`out = 4` at `x = (1, 1)`, inside the ACTIVE child of `relu1[0]`). Both
/// children claim both rows; the active child's row-1 lower (-1.25) is tighter
/// than the truth (-2). The ledger keeps the false `Verified` (lane
/// arithmetic is trusted); the replay must re-certify EVERY row of the
/// region and withdraw on row 1 while confirming the honest inactive child.
#[test]
fn mo_cover_replay_withdraws_a_region_false_on_its_second_row() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let graph = residual_graph();
    let objectives = vec![vec![-1.0_f32], vec![-0.5_f32]];
    let thresholds = vec![-4.5_f32, -1.5];
    let root = residual_root_rows(&graph, vec![(-5.0, 5.0), (-2.5, 2.5)], &thresholds);
    assert!(
        root.verified().iter().all(|&v| !v),
        "precondition: the root closes no row"
    );
    let lane = vec![(-4.25_f32, 2.0_f32), (-1.25, 1.0)];
    let active = residual_child_rows(&graph, &root, true, lane.clone(), &thresholds);
    let inactive = residual_child_rows(&graph, &root, false, lane, &thresholds);
    assert!(active.all_verified() && inactive.all_verified());

    let (ledger, deposited, _) = settle_tree(
        &graph,
        &objectives,
        &thresholds,
        &root,
        &[&active, &inactive],
        MoCoverCertify::Ledger,
        None,
    );
    assert!(matches!(ledger.result, BabVerificationStatus::Verified));
    assert_eq!(deposited, 2, "one cover per row");

    let (replay, deposited, report) = settle_tree(
        &graph,
        &objectives,
        &thresholds,
        &root,
        &[&active, &inactive],
        MoCoverCertify::Replay,
        None,
    );
    let report = report.expect("a replay ran");
    assert!(
        is_withdrawn(&replay.result),
        "a region false on its SECOND row must be withdrawn: {:?} {report:?}",
        replay.result
    );
    assert_eq!(deposited, 0);
    assert_eq!((report.regions, report.replayed), (2, 2), "{report:?}");
    assert_eq!(
        report.confirmed, 1,
        "the inactive child is honest on both rows"
    );
    assert_eq!(report.unconfirmed.len(), 1, "{report:?}");
    assert!(
        report.unconfirmed[0].contains("rows=[0, 1]")
            && report.unconfirmed[0].contains("row_slot=1"),
        "{report:?}"
    );
}

/// review(vit-v3), RED under the mutant that counts a deadline-cut region
/// as needing no replay: honest leaves whose replay the deadline cut must
/// WITHDRAW the verdict (`deadline_hit=1`), never keep it unreplayed.
#[test]
fn mo_cover_replay_withdraws_when_its_deadline_cuts_a_region() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let graph = residual_graph();
    let objectives = vec![vec![-1.0_f32]];
    let thresholds = vec![-4.5_f32];
    let root = residual_root(&graph, (-5.0, 5.0), &thresholds);
    let active = child(&graph, &root, true, (-4.25, 2.0), &thresholds);
    let inactive = child(&graph, &root, false, (-4.25, 2.0), &thresholds);

    // Control: the same honest tree with no deadline is confirmed.
    let (kept, deposited, report) = settle_tree(
        &graph,
        &objectives,
        &thresholds,
        &root,
        &[&active, &inactive],
        MoCoverCertify::Replay,
        None,
    );
    assert!(matches!(kept.result, BabVerificationStatus::Verified));
    assert_eq!(deposited, 1);
    assert!(report.expect("a replay ran").all_confirmed());

    // An already-expired replay deadline — the VERIFIER's own (review gate 1
    // measured that a caller-only past deadline is not the replay's: the
    // replay kept confirming under the constructor's). The tiny cover check
    // polls its deadline only every 2^16 work units, so it still proves.
    let (cut, deposited, report) = settle_tree(
        &graph,
        &objectives,
        &thresholds,
        &root,
        &[&active, &inactive],
        MoCoverCertify::Replay,
        Some(Instant::now()),
    );
    let report = report.expect("the cover proved, so a replay ran");
    assert!(
        is_withdrawn(&cut.result),
        "a replay the deadline cut must withdraw: {:?} {report:?}",
        cut.result
    );
    assert_eq!(deposited, 0);
    assert!(report.deadline_hit, "{report:?}");
    assert_eq!((report.replayed, report.confirmed), (2, 0), "{report:?}");
    assert_eq!(report.root_anchored, 0, "{report:?}");
    assert!(
        report.unconfirmed.iter().all(|u| u.contains("deadline")),
        "{report:?}"
    );
}

/// review(vit-v3), RED on the builder's head (it read the LOWER alone): the
/// replay confirms a row exactly when the lane's own closure rule
/// (`domain_is_verified_for_mode`, the rule the ledger re-applies through
/// `graph_bound_evidence`) accepts the replayed PAIR — a NaN or infinite
/// upper (the #2993 corruption signal) or a crossed interval is not a
/// re-certification — and EVERY row must pass.
#[test]
fn mo_cover_replay_confirmation_applies_the_lane_closure_rule() {
    use crate::certify::graph::graph_bound_evidence;
    assert!(rows_confirmed(&[(0.5, 1.0)], &[0.0]).is_ok());
    assert!(rows_confirmed(&[(0.0, 1.0)], &[0.0]).is_err(), "strict");
    for upper in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            rows_confirmed(&[(0.5, upper)], &[0.0]).is_err(),
            "a non-finite upper (#2993) is not a re-certification: {upper}"
        );
    }
    assert!(
        rows_confirmed(&[(0.5, 0.25)], &[0.0]).is_err(),
        "a crossed interval is not a re-certification"
    );
    assert!(rows_confirmed(&[(f32::NAN, 1.0)], &[0.0]).is_err());
    assert!(rows_confirmed(&[(0.5, 1.0)], &[f32::NEG_INFINITY]).is_err());
    assert!(rows_confirmed(&[(0.5, 1.0), (0.5, 1.0)], &[0.0]).is_err());
    let err = rows_confirmed(&[(0.5, 1.0), (-0.5, 1.0)], &[0.0, 0.0]).expect_err("row 1 open");
    assert!(
        err.contains("row_slot=1") && err.contains("replay_lower"),
        "{err}"
    );
    // Exactly the ledger's evidence rule, on a grid with the edge values.
    let values = [
        f32::NEG_INFINITY,
        -1.0,
        -f32::MIN_POSITIVE,
        0.0,
        f32::MIN_POSITIVE,
        0.25,
        1.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let mut agreed = 0usize;
    for &lower in &values {
        for &upper in &values {
            for &threshold in &values {
                assert_eq!(
                    rows_confirmed(&[(lower, upper)], &[threshold]).is_ok(),
                    graph_bound_evidence(false, lower, upper, threshold).is_some(),
                    "lower={lower} upper={upper} threshold={threshold}"
                );
                agreed += 1;
            }
        }
    }
    assert_eq!(agreed, 729);
}

// ---------------------------------------------------------------------------
// review(vit-v3): ENUMERATION-EXACT checks on a residual DAG
// ---------------------------------------------------------------------------

/// `y = W2 relu(W1 x + b1) + b2 + S x` on `[-1, 1]^2`: two outputs, `h` hidden
/// ReLUs, and a residual Linear skip joined by an `Add` (the #warm-dag DAG
/// shape: the input feeds two consumers). Every weight is on the 1/1024 grid,
/// so the f32 graph and the f64 reference compute the same function.
///
/// The function is affine on every cell of the arrangement of the `h` lines
/// `W1[i] x + b1[i] = 0` and the four box edges, and a region cut out by ReLU
/// literals is a convex union of closed cells, so the EXACT minimum of any row
/// `c · y` over the box or over such a region is attained at an arrangement
/// vertex inside it — which this struct enumerates.
struct ResNet2 {
    graph: GraphNetwork,
    w1: Vec<[f64; 2]>,
    b1: Vec<f64>,
    w2: [Vec<f64>; 2],
    b2: [f64; 2],
    s: [[f64; 2]; 2],
}

impl ResNet2 {
    #[allow(clippy::cast_possible_truncation)]
    fn random(rng: &mut Rng, h: usize) -> Self {
        let mut draw = |scale: f64| ((rng.sym() * scale * 1024.0).round() / 1024.0) as f32;
        let w1: Vec<[f32; 2]> = (0..h).map(|_| [draw(1.0), draw(1.0)]).collect();
        let b1: Vec<f32> = (0..h).map(|_| draw(0.5)).collect();
        let w2: Vec<Vec<f32>> = (0..2)
            .map(|_| (0..h).map(|_| draw(1.0)).collect())
            .collect();
        let b2: Vec<f32> = (0..2).map(|_| draw(0.5)).collect();
        let s: Vec<[f32; 2]> = (0..2).map(|_| [draw(0.5), draw(0.5)]).collect();

        let linear = |w: Vec<f32>, rows: usize, cols: usize, b: Option<Vec<f32>>| {
            Layer::Linear(
                LinearLayer::new(
                    Array2::from_shape_vec((rows, cols), w).expect("weight shape"),
                    b.map(Array1::from),
                )
                .expect("finite linear layer"),
            )
        };
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "fc0",
            linear(
                w1.iter().flatten().copied().collect(),
                h,
                2,
                Some(b1.clone()),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["fc0".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "fc1",
            linear(
                w2.iter().flatten().copied().collect(),
                2,
                h,
                Some(b2.clone()),
            ),
            vec!["relu0".to_string()],
        ));
        graph.add_node(GraphNode::from_input(
            "skip",
            linear(s.iter().flatten().copied().collect(), 2, 2, None),
        ));
        graph.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["fc1".to_string(), "skip".to_string()],
        ));
        graph.set_output("add");
        let f = |v: f32| f64::from(v);
        Self {
            graph,
            w1: w1.iter().map(|r| [f(r[0]), f(r[1])]).collect(),
            b1: b1.iter().copied().map(f).collect(),
            w2: [
                w2[0].iter().copied().map(f).collect(),
                w2[1].iter().copied().map(f).collect(),
            ],
            b2: [f(b2[0]), f(b2[1])],
            s: [[f(s[0][0]), f(s[0][1])], [f(s[1][0]), f(s[1][1])]],
        }
    }

    fn pre(&self, i: usize, x: [f64; 2]) -> f64 {
        self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
    }

    fn eval(&self, x: [f64; 2]) -> [f64; 2] {
        let hidden: Vec<f64> = (0..self.w1.len())
            .map(|i| self.pre(i, x).max(0.0))
            .collect();
        let out = |o: usize| {
            self.w2[o]
                .iter()
                .zip(&hidden)
                .map(|(w, a)| w * a)
                .sum::<f64>()
                + self.b2[o]
                + self.s[o][0] * x[0]
                + self.s[o][1] * x[1]
        };
        [out(0), out(1)]
    }

    /// Every vertex of the arrangement (hidden lines + box edges) in the box.
    fn vertices(&self) -> Vec<[f64; 2]> {
        let mut lines: Vec<([f64; 2], f64)> = self
            .w1
            .iter()
            .copied()
            .zip(self.b1.iter().copied())
            .collect();
        lines.extend([
            ([1.0, 0.0], 1.0),
            ([1.0, 0.0], -1.0),
            ([0.0, 1.0], 1.0),
            ([0.0, 1.0], -1.0),
        ]);
        let mut out = Vec::new();
        for i in 0..lines.len() {
            for j in (i + 1)..lines.len() {
                let ((a, b), (c, d)) = (lines[i], lines[j]);
                let det = a[0] * c[1] - a[1] * c[0];
                if det.abs() < 1e-12 {
                    continue;
                }
                let x0 = (-b * c[1] + a[1] * d) / det;
                let x1 = (-a[0] * d + b * c[0]) / det;
                if x0.abs() <= 1.0 + 1e-12 && x1.abs() <= 1.0 + 1e-12 {
                    out.push([x0.clamp(-1.0, 1.0), x1.clamp(-1.0, 1.0)]);
                }
            }
        }
        out
    }

    /// Whether `x` lies in the region of `literals` (`(neuron, active)`),
    /// with `margin` (> 0: strictly inside; 0: the closed region up to
    /// rounding).
    fn inside(&self, literals: &[(usize, bool)], x: [f64; 2], margin: f64) -> bool {
        literals.iter().all(|&(i, active)| {
            let z = self.pre(i, x);
            let tol = 1e-9 * (1.0 + self.w1[i][0].abs() + self.w1[i][1].abs() + self.b1[i].abs());
            if active {
                z >= margin - tol
            } else {
                z <= -margin + tol
            }
        })
    }

    /// The EXACT minimum of `row · y` over the region of `literals` (`None`:
    /// the region has no vertex, i.e. it is empty).
    fn exact_min(&self, row: &[f32], literals: &[(usize, bool)]) -> Option<f64> {
        self.vertices()
            .into_iter()
            .filter(|&x| self.inside(literals, x, 0.0))
            .map(|x| {
                let y = self.eval(x);
                f64::from(row[0]) * y[0] + f64::from(row[1]) * y[1]
            })
            .reduce(f64::min)
    }

    /// The EXACT maximum of `row · y` over the box.
    fn exact_max(&self, row: &[f32]) -> f64 {
        self.vertices()
            .into_iter()
            .map(|x| {
                let y = self.eval(x);
                f64::from(row[0]) * y[0] + f64::from(row[1]) * y[1]
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// A point strictly inside the region of `literals`, if its vertices'
    /// centroid is one (the region is convex, so a centroid strictly inside
    /// every literal proves the region NON-EMPTY with interior).
    fn interior_point(&self, literals: &[(usize, bool)]) -> Option<[f64; 2]> {
        let inside: Vec<[f64; 2]> = self
            .vertices()
            .into_iter()
            .filter(|&x| self.inside(literals, x, 0.0))
            .collect();
        if inside.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = inside.len() as f64;
        let c = [
            inside.iter().map(|x| x[0]).sum::<f64>() / n,
            inside.iter().map(|x| x[1]).sum::<f64>() / n,
        ];
        self.inside(literals, c, 1e-6).then_some(c)
    }
}

/// The largest binary32 value STRICTLY below `v`: a threshold on the TRUE
/// side (the exact minimum is above it).
fn f32_strictly_below(v: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    let t = v as f32;
    if f64::from(t) < v {
        t
    } else {
        f32_step(t, false)
    }
}

/// The smallest binary32 value STRICTLY above `v`: a threshold on the FALSE
/// side (the exact minimum is below it, so the property is violated).
fn f32_strictly_above(v: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    let t = v as f32;
    if f64::from(t) > v {
        t
    } else {
        f32_step(t, true)
    }
}

/// One binary32 ulp toward `+inf` (`up`) or `-inf`.
fn f32_step(t: f32, up: bool) -> f32 {
    if t == 0.0 {
        let tiny = f32::from_bits(1);
        return if up { tiny } else { -tiny };
    }
    let bits = t.to_bits();
    if (t > 0.0) == up {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

/// review(vit-v3): ENUMERATION-EXACT soundness of the whole armed system on
/// the REAL multi-objective driver. On random residual DAGs (input feeding a
/// ReLU branch and a Linear skip joined by `Add`), each row's EXACT minimum
/// over the box is enumerated, and threshold ladders are placed on both
/// sides of it — TRUE (every row's exact minimum strictly above its f32
/// threshold) and FALSE (some row's exact minimum strictly below it), down to
/// 0.2% of the row's range. For every run, batched `off` / `ledger` /
/// `replay` and sequential `ledger`:
///
/// * a `Verified` in ANY mode is a TRUE property (the exact minimum decides);
/// * `ledger` / `replay` never keep a `Verified` that `off` did not
///   (downgrade-only), and neither ever REFUSES a genuine run;
/// * every deposited capture authorizes the instance door for exactly the
///   property, and the door REFUSES the same covers for the FALSE property
///   one ulp past a row's exact minimum (a cover never entails more than it
///   proved).
#[ntest::timeout(900000)]
#[test]
fn mo_cover_exact_enumeration_no_mode_verifies_a_false_property() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut rng = Rng::new(0x0E8A_C7E4_0001_2026);
    let input = unit_box();
    // Margin-shaped rows (`y0 - y1/2`, `y1 - 5 y0 / 16`), dyadic so they are
    // exact, and used by no other test (the capture is process-global).
    let objectives = vec![vec![0.875_f32, -0.4375], vec![-0.3125_f32, 0.875]];
    let ladders: [(f64, f64); 7] = [
        (0.2, 0.2),
        (0.02, 0.05),
        (0.002, 0.3),
        (-0.002, 0.3),
        (0.3, -0.02),
        (-0.2, -0.2),
        (0.002, 0.002),
    ];
    let mut true_runs = 0usize;
    let mut false_runs = 0usize;
    let mut verified = [0usize; 3];
    let mut door_checked = 0usize;
    let mut replay_withdrawn = 0usize;
    for net in 0..12 {
        let res = ResNet2::random(&mut rng, 3 + net % 3);
        let mins: Vec<f64> = objectives
            .iter()
            .map(|row| res.exact_min(row, &[]).expect("the box has vertices"))
            .collect();
        let ranges: Vec<f64> = objectives
            .iter()
            .zip(&mins)
            .map(|(row, &lo)| (res.exact_max(row) - lo).max(1e-3))
            .collect();
        for &(d0, d1) in &ladders {
            let thresholds: Vec<f32> = [d0, d1]
                .iter()
                .enumerate()
                .map(|(j, &d)| {
                    let t = mins[j] - d * ranges[j];
                    if d > 0.0 {
                        f32_strictly_below(t)
                    } else {
                        f32_strictly_above(t)
                    }
                })
                .collect();
            let truth = mins
                .iter()
                .zip(&thresholds)
                .all(|(&m, &t)| m > f64::from(t));
            assert_eq!(truth, d0 > 0.0 && d1 > 0.0, "ladder side construction");
            if truth {
                true_runs += 1;
            } else {
                false_runs += 1;
            }
            let mut off_verified = false;
            for (slot, mode) in [
                MoCoverCertify::Off,
                MoCoverCertify::Ledger,
                MoCoverCertify::Replay,
            ]
            .into_iter()
            .enumerate()
            {
                reset_graph_cover_capture();
                let (result, _) =
                    run_driver(&res.graph, &input, &objectives, &thresholds, mode, true);
                let capture = take_graph_cover_capture();
                let is_verified = matches!(result.result, BabVerificationStatus::Verified);
                assert!(
                    !is_verified || truth,
                    "net={net} mode={mode:?} t={thresholds:?} exact_min={mins:?}: VERIFIED a \
                     FALSE property"
                );
                if mode == MoCoverCertify::Off {
                    off_verified = is_verified;
                    assert!(capture.is_empty(), "`off` captures nothing");
                } else {
                    assert!(
                        !is_verified || off_verified,
                        "net={net} mode={mode:?}: kept a Verified the lane did not decide"
                    );
                    if matches!(&result.result, BabVerificationStatus::Unknown { reason }
                        if reason.contains("graph cover gate refused"))
                    {
                        panic!(
                            "net={net} mode={mode:?}: REFUSED a genuine run: {:?}",
                            result.result
                        );
                    }
                    if mode == MoCoverCertify::Replay && off_verified && !is_verified {
                        replay_withdrawn += 1;
                    }
                }
                if is_verified {
                    verified[slot] += 1;
                }
                if is_verified && !capture.is_empty() {
                    door_checked += 1;
                    authorize_graph_instance(&property(&objectives, &thresholds), &capture)
                        .unwrap_or_else(|e| {
                            panic!("net={net} mode={mode:?}: Verified, door refused: {e}")
                        });
                    for j in 0..objectives.len() {
                        let mut false_side = thresholds.clone();
                        false_side[j] = f32_strictly_above(mins[j]);
                        assert!(
                            authorize_graph_instance(&property(&objectives, &false_side), &capture)
                                .is_err(),
                            "net={net} mode={mode:?} row={j}: the door minted a FALSE property"
                        );
                    }
                }
            }
            // The sequential lane, for soundness only (UNBUILT once it
            // branches; a root-terminal close still proves).
            reset_graph_cover_capture();
            let (sequential, _) = run_driver(
                &res.graph,
                &input,
                &objectives,
                &thresholds,
                MoCoverCertify::Ledger,
                false,
            );
            let _ = take_graph_cover_capture();
            assert!(
                !matches!(sequential.result, BabVerificationStatus::Verified) || truth,
                "net={net} sequential ledger VERIFIED a FALSE property"
            );
        }
    }
    eprintln!(
        "[mo-cover-exact] true_runs={true_runs} false_runs={false_runs} verified(off,ledger,\
         replay)={verified:?} door_checked={door_checked} replay_withdrawn={replay_withdrawn}"
    );
    assert!(true_runs >= 40 && false_runs >= 30, "ladders VOID");
    assert!(
        verified[1] >= 8 && verified[2] >= 3,
        "too few armed Verified runs to mean anything: {verified:?}"
    );
    assert!(door_checked >= 8, "door never exercised: {door_checked}");
}

/// review(vit-v3): ENUMERATION-EXACT containment of the certify-once replay
/// pass. On random split regions of random residual DAGs, the replay's
/// certified LOWER bound for each row never exceeds that row's EXACT minimum
/// over the region (enumerated), under the default β and a random `β >= 0`;
/// and the replay never certifies EMPTY a region with an interior point.
#[ntest::timeout(600000)]
#[test]
fn mo_cover_replay_bounds_contain_the_exact_minimum_on_residual_regions() {
    let mut rng = Rng::new(0x0E8A_C7E4_0002_2026);
    let input = unit_box();
    let objectives = vec![vec![0.875_f32, -0.4375], vec![-0.3125_f32, 0.875]];
    let thresholds = vec![0.0_f32, 0.0];
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut regions = 0usize;
    let mut checked = 0usize;
    let mut empties = 0usize;
    let mut tightest_gap = f64::INFINITY;
    for net in 0..16 {
        let h = 3 + net % 4;
        let res = ResNet2::random(&mut rng, h);
        let node_bounds = res
            .graph
            .collect_node_bounds_with_engine(&input, None)
            .expect("ibp");
        let root = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(-10.0, 10.0); 2],
            &input,
            &thresholds,
            false,
        )
        .expect("root");
        let base = MoReplayBase {
            graph: res.graph.clone(),
            root,
        };
        for _ in 0..8 {
            #[allow(clippy::cast_possible_truncation)]
            let depth = 1 + (rng.next_u64() % 3) as usize;
            let mut literals: Vec<(usize, bool)> = Vec::new();
            for _ in 0..depth {
                #[allow(clippy::cast_possible_truncation)]
                let neuron = (rng.next_u64() % h as u64) as usize;
                if literals.iter().any(|&(n, _)| n == neuron) {
                    continue;
                }
                literals.push((neuron, rng.next_u64().is_multiple_of(2)));
            }
            let mut key: Vec<(String, usize, bool)> = literals
                .iter()
                .map(|&(n, a)| ("relu0".to_string(), n, a))
                .collect();
            key.sort();
            let exact: Vec<Option<f64>> = objectives
                .iter()
                .map(|row| res.exact_min(row, &literals))
                .collect();
            let interior = res.interior_point(&literals);
            let prepared = match verifier.mo_cover_replay_prepare(&base, &key, &thresholds) {
                Ok(prepared) => prepared,
                Err(RegionOutcome::ConfirmedEmpty) => {
                    assert!(
                        interior.is_none(),
                        "net={net} {literals:?}: certified EMPTY a region with interior \
                         point {interior:?}"
                    );
                    empties += 1;
                    continue;
                }
                Err(other) => panic!("net={net} {literals:?}: the region did not build: {other:?}"),
            };
            regions += 1;
            let mut random = replay_beta(&prepared, None);
            for entry in &mut random.entries {
                #[allow(clippy::cast_possible_truncation)]
                entry.set_value((rng.unit() * 3.0) as f32);
            }
            for beta in [replay_beta(&prepared, None), random] {
                match verifier.mo_cover_replay_pass(
                    &base,
                    &prepared,
                    &beta,
                    &objectives,
                    &thresholds,
                ) {
                    Ok(bounds) => {
                        assert_eq!(bounds.len(), 2);
                        for (j, &(lower, _)) in bounds.iter().enumerate() {
                            let Some(m) = exact[j] else { continue };
                            let tol = 1e-6 * (1.0 + m.abs());
                            assert!(
                                f64::from(lower) <= m + tol,
                                "net={net} {literals:?} row {j}: certified lower {lower:e} \
                                 above the EXACT minimum {m:e}"
                            );
                            tightest_gap = tightest_gap.min(m - f64::from(lower));
                            checked += 1;
                        }
                    }
                    Err(error) => {
                        assert!(
                            infeasibility_slack(&error).is_some() && interior.is_none(),
                            "net={net} {literals:?}: the replay pass failed or certified EMPTY \
                             a region with interior point {interior:?}: {error}"
                        );
                        empties += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "[mo-cover-exact-replay] regions={regions} checked={checked} empties={empties} \
         tightest_gap={tightest_gap:e}"
    );
    assert!(regions >= 60, "only {regions} regions built: VOID");
    assert!(checked >= 200, "only {checked} rows checked: VOID");
}

// ---------------------------------------------------------------------------
// review(eng8): the integration's composition with the B2.2 proof tape
// ---------------------------------------------------------------------------

/// review(eng8): int/eng8 composed B2.2 and this system in
/// `verify_graph_relu_split_multi_objective_core`: the B2.2 proof tape WRAPS
/// the cover certification (`_certified`), which wraps the containment fence
/// (`_fenced`) and the search (`_unfenced`). Neither branch pins the
/// composition (B2.2's driver test runs with this system `off`; this file
/// never arms the lane proof capture). With BOTH armed on the real batched
/// driver, in every mode:
///
/// * the verdict and counters are the unarmed `off` run's (`replay` may only
///   withdraw a `Verified`, never add one);
/// * a run that ends `Verified` deposits exactly ONE tape for its rows into
///   the LANE-PROOF capture, and that tape streams more than its root leaf
///   (the tape was threaded through the settlement into the search);
/// * an armed mode still deposits one cover per row into the GRAPH-COVER
///   capture (two slots, no cross-talk), and `off` deposits none;
/// * a run that does not end `Verified` (a false row, or a verdict the
///   settlement withdrew) deposits NO tape.
#[ntest::timeout(600000)]
#[test]
fn mo_cover_composes_with_the_proof_tape_in_two_slots() {
    use crate::certify::lane_proof::{
        arm_lane_proof_capture_for_test, reset_lane_proof_capture, stream_lane_proofs,
        take_lane_proof_capture, LaneProofCapture, LaneProofInstance,
    };
    use crate::certify::{force_hint_capture_for_test, RecordingSink};

    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mlp = Mlp2::crossing2();
    let input = unit_box();
    let objectives = two_rows();
    let own = |thresholds: &[f32]| -> LaneProofCapture {
        let mut capture = take_lane_proof_capture();
        capture.retain_objectives(&[
            (objectives[0].as_slice(), thresholds[0]),
            (objectives[1].as_slice(), thresholds[1]),
        ]);
        capture
    };
    let counters = |r: &BetaCrownResult| {
        (
            r.domains_explored,
            r.domains_verified,
            r.max_depth_reached,
            r.cuts_generated,
        )
    };
    // `crossing2`: true minimum -1 on both rows, root chord bound -2.
    // -1.5 on both rows needs a split tree; -0.5 on row 0 is FALSE.
    let mut verified_taped = 0usize;
    let mut untaped = 0usize;
    for (raw, must_verify) in [(vec![-1.5f32, -1.5], true), (vec![-0.5f32, -1.5], false)] {
        let thresholds = scaled(&raw);
        // The unarmed baseline (the Competition profile, this system dark).
        force_hint_capture_for_test(Some(false));
        reset_lane_proof_capture();
        reset_graph_cover_capture();
        let (dark, _) = run_driver(
            &mlp.graph,
            &input,
            &objectives,
            &thresholds,
            MoCoverCertify::Off,
            true,
        );
        force_hint_capture_for_test(None);
        assert!(
            own(&thresholds).is_empty(),
            "t={raw:?}: the unarmed run deposited a tape"
        );
        drop(take_graph_cover_capture());
        assert_eq!(
            matches!(dark.result, BabVerificationStatus::Verified),
            must_verify,
            "t={raw:?}: fixture premise: {:?}",
            dark.result
        );
        for mode in [
            MoCoverCertify::Off,
            MoCoverCertify::Ledger,
            MoCoverCertify::Replay,
        ] {
            arm_lane_proof_capture_for_test();
            reset_graph_cover_capture();
            let (armed, _) = run_driver(&mlp.graph, &input, &objectives, &thresholds, mode, true);
            let tapes = own(&thresholds);
            let covers = take_graph_cover_capture();
            let label = mode.label();
            let withdrawn = mode.replays() && is_withdrawn(&armed.result);
            if !withdrawn {
                assert_eq!(
                    std::mem::discriminant(&dark.result),
                    std::mem::discriminant(&armed.result),
                    "t={raw:?} mode={label}: status changed: {:?} vs {:?}",
                    dark.result,
                    armed.result
                );
                assert_eq!(
                    counters(&dark),
                    counters(&armed),
                    "t={raw:?} mode={label}: counters changed"
                );
            }
            if matches!(armed.result, BabVerificationStatus::Verified) {
                assert_eq!(
                    tapes.len(),
                    1,
                    "t={raw:?} mode={label}: a Verified invocation deposits exactly one tape"
                );
                let clauses = objectives
                    .iter()
                    .zip(&thresholds)
                    .map(|(row, &t)| vec![GraphObjective::new(row, t, false).expect("finite")])
                    .collect();
                let instance_property =
                    GraphInstanceProperty::new(vec![-1.0, -1.0], vec![1.0, 1.0], clauses)
                        .expect("well formed");
                let mut sites = HashMap::new();
                sites.insert("relu0".to_string(), "z".to_string());
                let inst = LaneProofInstance::new(&instance_property, sites);
                let mut sink = RecordingSink::default();
                let summary = stream_lane_proofs(&tapes, &inst, &mut sink);
                // The two `lane_claim` fallbacks stream whether or not the
                // tape holds a leaf (a multi-objective tape is never
                // complete), so the evidence is the lane's own split leaves.
                let split_leaves = sink
                    .nodes
                    .iter()
                    .filter(|n| {
                        n.close == crate::certify::NodeClose::Bound
                            && n.src == "graph_mo_disjunctive"
                            && !n.lits.is_empty()
                    })
                    .count();
                assert!(
                    summary.tapes == 1 && split_leaves >= 2,
                    "t={raw:?} mode={label}: the tape recorded no search below the root \
                     (split_leaves={split_leaves}): {summary:?} {:?}",
                    sink.nodes
                );
                if mode.is_armed() {
                    assert_eq!(
                        covers.len(),
                        2,
                        "t={raw:?} mode={label}: one cover per row in the graph-cover slot"
                    );
                } else {
                    assert!(covers.is_empty(), "t={raw:?}: `off` captured a cover");
                }
                verified_taped += 1;
            } else {
                assert!(
                    tapes.is_empty(),
                    "t={raw:?} mode={label}: a run that did not end Verified ({:?}) deposited \
                     a tape",
                    armed.result
                );
                assert!(
                    covers.is_empty(),
                    "t={raw:?} mode={label}: a run that did not end Verified deposited a cover"
                );
                untaped += 1;
            }
            drop(take_lane_proof_capture());
        }
    }
    eprintln!("[mo-cover-tape] verified_taped={verified_taped} untaped={untaped}");
    assert!(
        verified_taped >= 2,
        "only {verified_taped} taped Verified runs: VOID"
    );
    assert!(untaped >= 3, "only {untaped} untaped runs: VOID");
}
