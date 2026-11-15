// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-bab-cover-ledger, end to end: the REAL ledgered drivers on a family
//! of small random ReLU networks with TRUE properties, across the execution
//! modes the drivers choose between (sequential, parallel, shared executor,
//! multi-depth splits; input split with and without reorder, IBP prescreen
//! and clipping).
//!
//! What is pinned is instrumentation COMPLETENESS: a genuine search that the
//! lane itself decides `Verified` must never be refused by its own ledger (a
//! refusal here would be a close site the ledger does not record — the false
//! refusal that would cost a banked row), and every such `Verified` must leave
//! a captured cover that the instance door accepts for exactly that
//! property. The family must contain at least one verified run whose cover
//! has more than one leaf, or the pin is VOID.

use std::sync::Arc;
use std::time::Duration;

use ndarray::{arr1, Array1, Array2};

use crate::beta_crown::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, GraphPrecomputedBounds,
};
use crate::certify::graph::CAPTURE_TEST_LOCK;
use crate::certify::{
    authorize_graph_instance, reset_graph_cover_capture, take_graph_cover_capture,
    GraphInstanceProperty, GraphObjective, TokenScope,
};
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A value in `[-1, 1)` on a 1/1024 grid (exact in binary32).
    fn weight(&mut self) -> f32 {
        let k = (self.next() % 2048) as i32 - 1024;
        k as f32 / 1024.0
    }
}

/// One MLP `2 -> widths... -> 1`: the graph and its f64 weights.
struct Mlp {
    graph: GraphNetwork,
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl Mlp {
    /// A random MLP `2 -> widths... -> 1` (weights drawn layer by layer, `w`
    /// then `b`, on the 1/1024 grid).
    fn random(rng: &mut Rng, widths: &[usize]) -> Self {
        let mut fan_in = 2usize;
        let dims: Vec<usize> = widths.iter().copied().chain(std::iter::once(1)).collect();
        let mut drawn = Vec::with_capacity(dims.len());
        for &out in &dims {
            let w: Vec<Vec<f32>> = (0..out)
                .map(|_| (0..fan_in).map(|_| rng.weight()).collect())
                .collect();
            let b: Vec<f32> = (0..out).map(|_| rng.weight() * 0.5).collect();
            drawn.push((w, b));
            fan_in = out;
        }
        Self::from_layers(drawn)
    }

    /// `y = 1 - relu(x1 + x2) - relu(x1 - x2)` on `[-1, 1]^2`: two ReLUs whose
    /// zero lines cross at the box centre, so all four activation patterns are
    /// feasible (no jointly-empty split combination). True minimum `-1` (at
    /// `x1 = 1`); the root CROWN bound (chord relaxation of both ReLUs) is
    /// `-2`, below every family threshold, while each full pattern's exact
    /// linear bound is `>= -1`, above every one. So every ReLU-split mode must
    /// close a split tree with more than one leaf on it — added at the ENG7
    /// gate because the random family closed NO split tree in
    /// `multi_depth_parallel` (verified=9 multi_leaf=0), which left that
    /// mode's recording sites VOID.
    fn crossing() -> Self {
        Self::from_layers(vec![
            (vec![vec![1.0, 1.0], vec![1.0, -1.0]], vec![0.0, 0.0]),
            (vec![vec![-1.0, -1.0]], vec![1.0]),
        ])
    }

    /// The graph `fc0 -> relu0 -> ... -> fcN` of `layers` (input width 2,
    /// output width 1) and its f64 weights.
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

    fn eval(&self, x: [f64; 2]) -> f64 {
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
        h[0]
    }

    /// A threshold STRICTLY below the network's minimum over `[-1, 1]^2`:
    /// the grid minimum minus a Lipschitz bound on the grid error, minus
    /// `slack` times the output range.
    fn true_threshold(&self, slack: f64) -> f32 {
        let n = 100;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..=n {
            for j in 0..=n {
                let x = [
                    -1.0 + 2.0 * f64::from(i) / f64::from(n),
                    -1.0 + 2.0 * f64::from(j) / f64::from(n),
                ];
                let v = self.eval(x);
                lo = lo.min(v);
                hi = hi.max(v);
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
        let grid_error = lipschitz * (1.0 / f64::from(n));
        (lo - grid_error - slack * (hi - lo) - 1e-3) as f32
    }
}

fn unit_box() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite box")
}

fn refused(result: &BabVerificationStatus) -> bool {
    matches!(result, BabVerificationStatus::Unknown { reason } if reason.contains("graph cover gate refused"))
}

/// The family's objective: `0.5 · y`, a row no other test uses, so a cover a
/// concurrently running test deposits can never be mistaken for this one's.
/// Scaling by `0.5` is exact, so `0.5·y > 0.5·t` is the property `y > t`.
const OBJECTIVE: [f32; 1] = [0.5];

/// Run `verify` on every (network, slack, config) of the family under an
/// armed capture; pin completeness and the instance door on every verdict.
fn run_family(
    lane: &str,
    configs: &[(&str, BetaCrownConfig, bool)],
    verify: impl Fn(&BetaCrownVerifier, &GraphNetwork, &BoundedTensor, f32) -> BabVerificationStatus,
) {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut rng = Rng(0x2F69_3A1C_7B45_9E0D);
    let input = unit_box();
    let mut verified = 0usize;
    let mut deep = 0usize;
    let mut runs = 0usize;
    // Per configuration: (verified runs, verified runs whose cover has more
    // than one leaf). Each configuration names an execution mode; an
    // aggregate count lets one mode pass on another mode's evidence.
    let mut per_config = vec![(0usize, 0usize); configs.len()];
    // Per configuration: witness-less `PotentialViolation` early exits on a
    // TRUE property (reported in the summary; see the assertion below).
    let mut per_config_potential = vec![0usize; configs.len()];
    // Six random nets, then the crafted `crossing` instance (see its doc).
    for net in 0..7 {
        let mlp = if net < 6 {
            let widths: &[usize] = if net % 2 == 0 { &[4] } else { &[3, 3] };
            Mlp::random(&mut rng, widths)
        } else {
            Mlp::crossing()
        };
        for slack in [0.02, 0.1, 0.3] {
            let threshold = mlp.true_threshold(slack) * 0.5;
            for (config_idx, (name, config, with_engine)) in configs.iter().enumerate() {
                let verifier = if *with_engine {
                    BetaCrownVerifier::new_with_engine(
                        config.clone(),
                        Arc::new(ny_core::NaiveCpuGemmEngine),
                    )
                } else {
                    BetaCrownVerifier::new(config.clone())
                };
                reset_graph_cover_capture();
                let result = verify(&verifier, &mlp.graph, &input, threshold);
                let capture = take_graph_cover_capture();
                runs += 1;
                assert!(
                    !refused(&result),
                    "{lane}/{name} net={net} slack={slack}: the ledger REFUSED a genuine run: \
                     {result:?}"
                );
                // A witness-CARRYING claim on a TRUE property is a wrong answer
                // and fails here. A witness-LESS `PotentialViolation` is the
                // ReLU-split drivers' "counterexample REGION" early exit
                // (`domain_is_violation`: an ordered interval entirely below the
                // threshold), which they report without re-checking that the
                // domain's split combination is jointly feasible; on a TRUE
                // property a SOUND interval can only lie below the threshold on
                // an EMPTY region. The split leaf's UPPER end is sound (settled
                // 2026-09-30, #warm-dag): `apply_beta_split_to_column` applies
                // the Lagrangian signs (`lower_a -= s*beta`, `upper_a += s*beta`,
                // every beta write clamped `>= 0`), which is weak duality for
                // `s*z >= 0` in both directions. The #violdrop measurement
                // (`multi_objective/shared.rs`) was the lA warm start seeding a
                // child with an incomplete block on a RESIDUAL graph; these
                // MLPs are chains, where the seed is complete, which is why an
                // ENG7 review probe (1800 runs of this driver, 6 modes, both
                // directions) verified 0 of 900 FALSE properties. The residual
                // case is pinned by `graph/tests/warm_start_seed.rs`. In
                // production a witness-less potential is re-searched for a
                // CONCRETE point (`confirm_potential_violation`; a found point
                // must still pass the ORT sat gate) and otherwise becomes
                // `Unknown`, and a residual `potential_violation` status
                // translates to `Unknown` (ny-cli `commands/vnncomp.rs`
                // `translate_status_with`), so it can cost a +10 and never
                // publishes an unvalidated `sat`. It is an incompleteness of the
                // driver, not of the ledger this family pins, so it is counted,
                // not failed (measured at the ENG7 gate: `multi_depth_shared`,
                // net 0, slack 0.02, where `{z0 <= 0, z1 <= 0}` is jointly
                // infeasible on the box and no beta=0 pattern interval lies below
                // the threshold).
                assert!(
                    !matches!(
                        result,
                        BabVerificationStatus::Violated { .. }
                            | BabVerificationStatus::PotentialViolation { witness: Some(_) }
                    ),
                    "{lane}/{name} net={net} slack={slack}: a TRUE property reported violated \
                     with a witness: {result:?}"
                );
                if matches!(
                    result,
                    BabVerificationStatus::PotentialViolation { witness: None }
                ) {
                    per_config_potential[config_idx] += 1;
                }
                if matches!(result, BabVerificationStatus::Verified) {
                    verified += 1;
                    per_config[config_idx].0 += 1;
                    let atom = GraphObjective::new(&OBJECTIVE, threshold, false).expect("finite");
                    let property = GraphInstanceProperty::new(
                        vec![-1.0, -1.0],
                        vec![1.0, 1.0],
                        vec![vec![atom]],
                    )
                    .expect("well formed");
                    let token = authorize_graph_instance(&property, &capture).unwrap_or_else(|e| {
                        panic!(
                            "{lane}/{name} net={net} slack={slack}: Verified but the instance \
                             door refused: {e}"
                        )
                    });
                    assert_eq!(token.scope(), TokenScope::Instance);
                    let atom = GraphObjective::new(&OBJECTIVE, threshold, false).expect("finite");
                    if capture.leaves_for(&atom).is_some_and(|n| n > 1) {
                        deep += 1;
                        per_config[config_idx].1 += 1;
                    }
                }
            }
        }
    }
    // 7 nets (6 random + `crossing`) x 3 slacks per configuration.
    assert_eq!(runs, 21 * configs.len(), "the family ran incompletely");
    assert!(
        verified >= 4,
        "{lane}: only {verified} of {runs} runs verified: the completeness pin is VOID"
    );
    assert!(
        deep >= 1,
        "{lane}: no verified run built a multi-leaf cover: the pin never exercised a split tree"
    );
    let summary: Vec<String> = configs
        .iter()
        .zip(&per_config)
        .zip(&per_config_potential)
        .map(|(((name, _, _), (v, d)), p)| {
            format!("{name}: verified={v} multi_leaf={d} potential_no_witness={p}")
        })
        .collect();
    eprintln!("[cover-ledger-family] lane={lane} {summary:?}");
    for ((name, _, _), &(v, d)) in configs.iter().zip(&per_config) {
        assert!(
            v >= 1 && d >= 1,
            "{lane}/{name}: verified={v} multi_leaf={d}: this execution mode never closed a \
             split tree, so its recording sites are VOID, not pinned ({summary:?})"
        );
    }
}

/// The single-objective ReLU-split driver in every execution mode it picks
/// (sequential, parallel, shared executor) and with multi-depth splits.
#[ntest::timeout(300000)]
#[test]
fn graph_cover_ledger_relu_split_runs_never_refuse_and_prove_their_verdicts() {
    let base = BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 2000,
        max_depth: 24,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let configs = [
        (
            "sequential",
            BetaCrownConfig {
                batch_size: 1,
                ..base.clone()
            },
            false,
        ),
        (
            "parallel",
            BetaCrownConfig {
                batch_size: 4,
                ..base.clone()
            },
            false,
        ),
        (
            "shared_executor",
            BetaCrownConfig {
                batch_size: 4,
                ..base.clone()
            },
            true,
        ),
        (
            "multi_depth_parallel",
            BetaCrownConfig {
                batch_size: 16,
                max_relu_split_depth: 3,
                min_batch_fill_ratio: 0.5,
                ..base.clone()
            },
            false,
        ),
        (
            "multi_depth_shared",
            BetaCrownConfig {
                batch_size: 16,
                max_relu_split_depth: 3,
                min_batch_fill_ratio: 0.5,
                ..base
            },
            true,
        ),
    ];
    run_family(
        "relu_split",
        &configs,
        |verifier, graph, input, threshold| {
            verifier
                .verify_graph_relu_split(graph, input, &OBJECTIVE, threshold)
                .expect("the relu-split driver runs")
                .result
        },
    );
}

/// The single-objective input-split driver: per-child and reorder batching,
/// IBP prescreen, relaxed clipping (whose excluded slabs the ledger records),
/// multi-dimension splits.
#[ntest::timeout(300000)]
#[test]
fn graph_cover_ledger_input_split_runs_never_refuse_and_prove_their_verdicts() {
    let base = BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 3000,
        max_depth: 40,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let configs = [
        ("per_child", base.clone(), false),
        (
            "ibp_prescreen",
            BetaCrownConfig {
                input_split_ibp_enhancement: true,
                ..base.clone()
            },
            false,
        ),
        (
            "reorder_prescreen",
            BetaCrownConfig {
                reorder_bab: true,
                input_split_ibp_enhancement: true,
                batch_size: 8,
                ..base.clone()
            },
            false,
        ),
        (
            "relaxed_clip",
            BetaCrownConfig {
                enable_relaxed_clip: true,
                ..base.clone()
            },
            false,
        ),
        (
            "multi_dim",
            BetaCrownConfig {
                input_split_depth: 2,
                ..base
            },
            false,
        ),
    ];
    run_family(
        "input_split",
        &configs,
        |verifier, graph, input, threshold| {
            verifier
                .verify_graph_input_split(graph, input, &OBJECTIVE, threshold)
                .expect("the input-split driver runs")
                .result
        },
    );
}

/// The precomputed-bounds ReLU-split driver: the lane the serial multi-clause
/// disjunctive loop hands one clause at a time (explicit active/inactive
/// children, `Ok(None)` half-spaces, no-unstable leaves).
#[ntest::timeout(300000)]
#[test]
fn graph_cover_ledger_relu_split_with_bounds_runs_never_refuse_and_prove_their_verdicts() {
    let base = BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 2000,
        max_depth: 24,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let configs = [
        ("precomputed", base.clone(), false),
        (
            "precomputed_alpha",
            BetaCrownConfig {
                use_alpha_crown: true,
                ..base
            },
            false,
        ),
    ];
    run_family(
        "relu_split_with_bounds",
        &configs,
        |verifier, graph, input, threshold| {
            let (node_bounds, output_bounds) = verifier
                .compute_initial_graph_bounds(graph, input, None)
                .expect("the root bounds compute");
            let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);
            verifier
                .verify_graph_relu_split_with_bounds(
                    graph,
                    input,
                    &OBJECTIVE,
                    threshold,
                    &precomputed,
                )
                .expect("the precomputed-bounds driver runs")
                .result
        },
    );
}
