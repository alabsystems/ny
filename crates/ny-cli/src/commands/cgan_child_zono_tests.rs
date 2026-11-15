// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#cgan-child-zono` REAL-MODEL soundness pins (V2 brief §3.5):
//!
//! * `child_zono_contains_onnx_samples` — the four ConvT cGANs, random child
//!   boxes at depths 0-12, 1e4 samples plus every corner per box, BOX and
//!   POINTWISE FORM containment at every node, published entries included,
//!   half the boxes intersected with ny's own IBP map of the root box;
//! * `child_zono_replica_parity_row7` — the root unstable counts and zonotope +
//!   CROWN margins of the Z1 replica on row 7: every BatchNorm charged with
//!   the binary64 enclosure of its exact-real coefficients (#zono-exact-bn),
//!   margins pinned to Z1's own binary64-BatchNorm numbers (see the test);
//! * `child_zono_sat_leaf_never_closes` — rows 4 and 9, on leaves containing
//!   the banked counterexample: every certified lower bound stays at or below
//!   the counterexample's own query value;
//! * `child_zono_refuses_unsupported_ops` — the non-ConvT cGAN variants refuse;
//!   the small-transformer model (row 19) refuses one boundary EARLIER, at
//!   graph construction, so no lane is ever handed it.
//!
//! The corpus is resolved in this checkout or, from a git worktree (which has
//! no `benchmarks/`), in the main checkout the worktree belongs to. A missing
//! corpus is VOID and FAILS the test.
//!
//! Every program is ALSO cross-checked node by node against ny's own pointwise
//! forward (`collect_node_activations_pointwise`), an independent evaluator: a
//! layout, padding or transposition mistake in the zonotope kernels would make
//! the sampled "truth" wrong in the same way as the enclosure, and this is
//! what catches it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::{
    load_onnx_with_config, BatchNormFoldingPolicy, CompoundNodePolicy, GraphNetworkOptions,
    OnnxLoadConfig,
};
use ny_propagate::zono_core::form_check::{check_point, summarize, ContainmentReport, FormSummary};
use ny_propagate::zono_core::{
    spec_lower_bounds, BatchNormCoefficients, RowPar, SpecQuery, ZonoForwardOptions, ZonoNodeView,
    ZonoProgram, ZonoRefusal,
};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

const CORPUS: &str = "benchmarks/vnncomp2026/benchmarks/cgan2026/1.0/onnx";

/// Resolve the cgan2026 ONNX corpus (see the module docs).
fn corpus_dir() -> PathBuf {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let direct = workspace.join(CORPUS);
    if direct.is_dir() {
        return direct;
    }
    if let Ok(text) = std::fs::read_to_string(workspace.join(".git")) {
        if let Some(gitdir) = text.trim().strip_prefix("gitdir:") {
            let gitdir = PathBuf::from(gitdir.trim());
            if let Some(main_git) = gitdir
                .ancestors()
                .find(|path| path.file_name().is_some_and(|name| name == ".git"))
            {
                if let Some(main) = main_git.parent() {
                    let candidate = main.join(CORPUS);
                    if candidate.is_dir() {
                        return candidate;
                    }
                }
            }
        }
    }
    panic!(
        "VOID: the cgan2026 ONNX corpus ({CORPUS}) is missing under {} and under the main \
         checkout; this soundness test cannot pass without it",
        workspace.display()
    );
}

/// Load a cgan2026 model the way the scored preset does (BatchNorm preserved
/// raw, normalization decomposition policy of the CLI loader). The preset's
/// loader (`preset::build_onnx_load_config`) additionally sets
/// `require_authored_float32_initializers`, an admission check that can only
/// REFUSE a load, never change the graph it converts; no cGAN preset sets an
/// ONNX optimization flag.
fn load_graph(name: &str) -> (GraphNetwork, Vec<usize>) {
    let (graph, input_shape) = try_load_graph(name);
    let graph = graph.unwrap_or_else(|error| panic!("graph {name}: {error}"));
    (graph, input_shape)
}

/// [`load_graph`] with the GRAPH CONSTRUCTION result returned instead of
/// unwrapped. The ONNX parse itself must still succeed: a model that does not
/// even parse is a different fact from one whose graph ny refuses to build.
fn try_load_graph(name: &str) -> (Result<GraphNetwork, String>, Vec<usize>) {
    let path = corpus_dir().join(name);
    assert!(path.is_file(), "VOID: model {} is missing", path.display());
    let config = OnnxLoadConfig::default()
        .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw);
    let model = load_onnx_with_config(&path, &config)
        .unwrap_or_else(|error| panic!("load {name}: {error}"));
    let input_shape = model
        .network
        .inputs
        .first()
        .map(|input| ny_onnx::resolve_dynamic_shape(&input.shape, 1))
        .unwrap_or_else(|| vec![1]);
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        })
        .map_err(|error| error.to_string());
    (graph, input_shape)
}

fn boxed(lo: &[f32], hi: &[f32], shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lo.to_vec()).expect("box lower"),
        ArrayD::from_shape_vec(IxDyn(shape), hi.to_vec()).expect("box upper"),
    )
    .expect("box")
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    #[allow(clippy::cast_precision_loss)]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A random BaB child of `parent`: `depth` midpoint halvings of free dims.
#[allow(clippy::cast_possible_truncation)]
fn random_child(parent: &BoundedTensor, depth: usize, rng: &mut Rng) -> BoundedTensor {
    let mut lo: Vec<f32> = parent.lower().iter().copied().collect();
    let mut hi: Vec<f32> = parent.upper().iter().copied().collect();
    let free: Vec<usize> = (0..lo.len()).filter(|&d| hi[d] > lo[d]).collect();
    for _ in 0..depth {
        let dim = free[(rng.next_u64() % free.len() as u64) as usize];
        let mid = lo[dim] + (hi[dim] - lo[dim]) / 2.0;
        if rng.unit() < 0.5 {
            hi[dim] = mid;
        } else {
            lo[dim] = mid;
        }
    }
    boxed(&lo, &hi, parent.shape())
}

/// `samples` uniform points plus every corner, flat row-major.
fn sample_points(domain: &BoundedTensor, samples: usize, seed: u64) -> Vec<f64> {
    let lo: Vec<f64> = domain.lower().iter().map(|&v| f64::from(v)).collect();
    let hi: Vec<f64> = domain.upper().iter().map(|&v| f64::from(v)).collect();
    let n = lo.len();
    let mut points = Vec::with_capacity((samples + (1 << n)) * n);
    for corner in 0..(1usize << n) {
        for d in 0..n {
            points.push(if corner >> d & 1 == 1 { hi[d] } else { lo[d] });
        }
    }
    let mut rng = Rng::new(seed);
    for _ in 0..samples {
        for d in 0..n {
            points.push((lo[d] + rng.unit() * (hi[d] - lo[d])).clamp(lo[d], hi[d]));
        }
    }
    points
}

/// Forward pass with an observer, then box + pointwise-form containment of
/// every node (and every published entry) at every sampled point, in chunks.
fn check_box(
    program: &ZonoProgram,
    child: &BoundedTensor,
    inherited: &[&HashMap<String, BoundedTensor>],
    samples: usize,
    seed: u64,
) -> ContainmentReport {
    let opts = ZonoForwardOptions {
        max_generators: 16_384,
        record_trace: false,
        row_par: RowPar::Seq,
    };
    let mut summaries: Vec<FormSummary> = Vec::new();
    let mut observer = |view: &ZonoNodeView<'_>| summaries.push(summarize(view));
    let outcome = program
        .forward(child, &opts, inherited, None, Some(&mut observer))
        .expect("certified forward pass on a cGAN child");
    assert_eq!(summaries.len(), program.steps().len() + 1);
    let steps = program.steps();
    let published: Vec<Option<(Vec<f32>, Vec<f32>)>> = steps
        .iter()
        .map(|step| {
            outcome.published.get(&step.node).map(|entry| {
                (
                    entry.lower().iter().copied().collect(),
                    entry.upper().iter().copied().collect(),
                )
            })
        })
        .collect();
    let points = sample_points(child, samples, seed);
    let n0 = program.input_len();
    let rows_total = points.len() / n0;
    let mut report = ContainmentReport::default();
    let chunk = 256usize;
    for start in (0..rows_total).step_by(chunk) {
        let rows = chunk.min(rows_total - start);
        let block = &points[start * n0..(start + rows) * n0];
        program
            .evaluate_points(block, rows, |step, values| {
                let summary = &summaries[step.map_or(0, |s| s + 1)];
                let n = summary.center.len();
                for r in 0..rows {
                    check_point(
                        summary,
                        &block[r * n0..(r + 1) * n0],
                        &values[r * n..(r + 1) * n],
                        &mut report,
                    );
                }
                if let Some((lo, hi)) = step.and_then(|s| published[s].as_ref()) {
                    for r in 0..rows {
                        for (j, &v) in values[r * n..(r + 1) * n].iter().enumerate() {
                            report.checked += 1;
                            if v < f64::from(lo[j]) || v > f64::from(hi[j]) {
                                report.box_violations += 1;
                            }
                        }
                    }
                }
            })
            .expect("concrete evaluation");
    }
    assert!(report.checked > 0, "nothing was checked: VOID");
    report
}

/// Node-by-node agreement of the compiled program's binary64 forward with
/// ny's own pointwise forward at a few points of `domain`.
fn cross_check_semantics(graph: &GraphNetwork, program: &ZonoProgram, domain: &BoundedTensor) {
    let points = sample_points(domain, 8, 99);
    let n0 = program.input_len();
    let rows = points.len() / n0;
    let steps = program.steps();
    let mut compared = 0usize;
    for r in 0..rows {
        let x = &points[r * n0..(r + 1) * n0];
        #[allow(clippy::cast_possible_truncation)]
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let degenerate = boxed(&x32, &x32, domain.shape());
        let ny_values = graph
            .collect_node_activations_pointwise(&degenerate, None)
            .expect("ny pointwise forward");
        let x_exact: Vec<f64> = x32.iter().map(|&v| f64::from(v)).collect();
        program
            .evaluate_points(&x_exact, 1, |step, values| {
                let Some(s) = step else { return };
                let reference = ny_values
                    .get(&steps[s].node)
                    .unwrap_or_else(|| panic!("ny has no value for `{}`", steps[s].node));
                let reference: Vec<f32> = reference.lower().iter().copied().collect();
                assert_eq!(reference.len(), values.len(), "`{}` length", steps[s].node);
                let scale = reference
                    .iter()
                    .fold(1.0f64, |m, v| m.max(f64::from(v.abs())));
                for (j, (&mine, &theirs)) in values.iter().zip(&reference).enumerate() {
                    assert!(
                        (mine - f64::from(theirs)).abs() <= 1e-3 * scale,
                        "`{}`[{j}]: zonotope program {mine} vs ny {theirs}",
                        steps[s].node
                    );
                    compared += 1;
                }
            })
            .expect("program evaluation");
    }
    assert!(compared > 0, "semantic cross-check compared nothing: VOID");
}

struct Row {
    model: &'static str,
    lo: [f32; 5],
    hi: [f32; 5],
}

/// One representative official box per ConvT cGAN (vnnlib literals, which
/// are exact binary32 values).
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
const CONTAINMENT_ROWS: [Row; 4] = [
    // row 2: cGAN_imgSz32_nCh_1 prop_2
    Row {
        model: "cGAN_imgSz32_nCh_1.onnx",
        lo: [
            0.3720855414867401,
            0.21389585733413696,
            0.8399999737739563,
            -1.0,
            0.3964410424232483,
        ],
        hi: [
            0.5320855379104614,
            0.3738958537578583,
            1.0,
            -0.8399999737739563,
            0.5564410090446472,
        ],
    },
    // row 7: cGAN_imgSz32_nCh_3 prop_3
    Row {
        model: "cGAN_imgSz32_nCh_3.onnx",
        lo: [
            0.5669031143188477,
            -1.0,
            -0.16821321845054626,
            -0.9540205001831055,
            -0.1360306590795517,
        ],
        hi: [
            0.7069031000137329,
            -0.8600000143051147,
            -0.028213215991854668,
            -0.8140205144882202,
            0.003969347104430199,
        ],
    },
    // row 10: cGAN_imgSz64_nCh_1 prop_2
    Row {
        model: "cGAN_imgSz64_nCh_1.onnx",
        lo: [
            0.6692480444908142,
            -0.6740142703056335,
            -0.5373633503913879,
            0.949999988079071,
            0.8862711191177368,
        ],
        hi: [
            0.7192479968070984,
            -0.6240142583847046,
            -0.5373633503913879,
            1.0,
            0.9362711310386658,
        ],
    },
    // row 12: cGAN_imgSz64_nCh_3 prop_0
    Row {
        model: "cGAN_imgSz64_nCh_3.onnx",
        lo: [
            0.578574001789093,
            0.668132483959198,
            -0.9700000286102295,
            -0.9700000286102295,
            -0.9262236952781677,
        ],
        hi: [
            0.6385740041732788,
            0.7281324863433838,
            -0.9700000286102295,
            -0.9700000286102295,
            -0.8662236928939819,
        ],
    },
];

#[test]
fn child_zono_contains_onnx_samples() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test pool");
    for (row_index, row) in CONTAINMENT_ROWS.iter().enumerate() {
        let (graph, shape) = load_graph(row.model);
        let root = boxed(&row.lo, &row.hi, &shape);
        let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root)
            .unwrap_or_else(|refusal| panic!("{}: program refused: {refusal}", row.model));
        cross_check_semantics(&graph, &program, &root);
        let root_ibp = graph
            .collect_node_bounds_with_engine(&root, None)
            .expect("root IBP map");
        let mut rng = Rng::new(1_000 + row_index as u64);
        let children: Vec<BoundedTensor> = (0..=12)
            .map(|depth| random_child(&root, depth, &mut rng))
            .collect();
        let reports: Vec<(usize, ContainmentReport)> = pool.install(|| {
            use rayon::prelude::*;
            children
                .par_iter()
                .enumerate()
                .map(|(depth, child)| {
                    let inherited: Vec<&HashMap<String, BoundedTensor>> = if depth % 2 == 0 {
                        vec![&root_ibp]
                    } else {
                        Vec::new()
                    };
                    (
                        depth,
                        check_box(&program, child, &inherited, 10_000, depth as u64),
                    )
                })
                .collect()
        });
        for (depth, report) in reports {
            assert_eq!(
                report.box_violations, 0,
                "{} depth {depth}: box containment violated: {report:?}",
                row.model
            );
            assert_eq!(
                report.form_violations, 0,
                "{} depth {depth}: pointwise form containment violated: {report:?}",
                row.model
            );
        }
    }
}

#[test]
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
fn child_zono_replica_parity_row7() {
    let row = &CONTAINMENT_ROWS[1];
    let (graph, shape) = load_graph(row.model);
    let root = boxed(&row.lo, &row.hi, &shape);
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile row 7");
    // The lane's own observable (#zono-exact-bn): all seven BatchNorms of the
    // ONNX-converted model charge the binary64 enclosure of their exact-real
    // coefficients. A binary32 BatchNorm here makes the pins below VOID.
    let batch_norms = program.batch_norm_coefficients();
    assert_eq!(
        batch_norms.len(),
        7,
        "row 7 has seven BatchNorms: {batch_norms:?}"
    );
    assert!(
        batch_norms
            .iter()
            .all(|(_, source)| *source == BatchNormCoefficients::ExactF64),
        "every row-7 BatchNorm must take the exact-f64 path: {batch_norms:?}"
    );
    let opts = ZonoForwardOptions {
        max_generators: 16_384,
        record_trace: true,
        row_par: RowPar::Seq,
    };
    let outcome = program
        .forward(&root, &opts, &[], None, None)
        .expect("root zonotope");
    // Z1 root_r7.log: BN_5, BN_8, BN_11, Conv_14, Conv_16, Conv_19, Conv_22
    // (the exact-f64 replica arm reproduces them exactly, and 4153 generators).
    let expected = [260usize, 956, 1993, 406, 183, 193, 157];
    let counts: Vec<usize> = outcome.relus.iter().map(|r| r.zono_unstable).collect();
    assert_eq!(
        counts.len(),
        expected.len(),
        "ReLU count differs: {counts:?}"
    );
    for (layer, (&got, &want)) in counts.iter().zip(&expected).enumerate() {
        let tolerance = (want / 50).max(2);
        assert!(
            got.abs_diff(want) <= tolerance,
            "layer {layer}: {got} unstable vs Z1 {want} (+/- {tolerance}); all {counts:?}"
        );
    }
    assert_eq!(
        outcome.stats.folded, 0,
        "row 7's root must not fold at 16384"
    );
    assert!(
        outcome.stats.gens_final.abs_diff(4153) <= 4153 / 50,
        "final generators {} vs Z1 4153",
        outcome.stats.gens_final
    );
    let trace = outcome.trace.as_ref().expect("trace");
    let queries = [
        // Y >= 0.7149031162261963 is refuted by (0.7149031162261963 - Y) > 0.
        SpecQuery {
            coeffs: vec![-1.0],
            constant: 0.7149031162261963,
        },
        // Y <= 0.5589030981063843 is refuted by (Y - 0.5589030981063843) > 0.
        SpecQuery {
            coeffs: vec![1.0],
            constant: -0.5589030981063843,
        },
    ];
    let lbs = spec_lower_bounds(&program, trace, &queries).expect("zonotope + CROWN margins");
    // WHICH REPLICA NUMBERS THIS PINS (#zono-exact-bn).
    //
    // Z1 (`root_r7.log`: -0.297276 / -0.178076) applies each BatchNorm as a
    // binary64 affine it recomputes from the ONNX statistics, radius 4u|s|
    // (`z1lib.load_model`). This program consumes ny's own `BatchNormLayer`,
    // which since #zono-exact-bn hands the zonotope core the binary64
    // enclosure `BatchNormLayer::new` computes of the EXACT-REAL coefficients
    // (outward-rounded intervals, centre +/- a radius of ~1e-15 relative) in
    // place of its binary32 scale/bias and their certified ~1e-7 radii.
    //
    // Replica evidence, `scratchpad/vbs/diag14/z1_exactbn_parity.py` (main
    // checkout; log beside it): the unchanged Z1 forward + CROWN with ONLY the
    // BatchNorm (s, t, es, et) swapped for this exact representation,
    // reproduced bit for bit in binary64 (arm `ny_exact_f64`), gives
    // -0.2972756931391007 / -0.17807597265806638: 3.7e-11 / 1.5e-11 below Z1's
    // own binary64-BatchNorm numbers (its radius is a few u wider than Z1's
    // 4u), counts 260/956/1993/406/183/193/157 and 4153 generators exactly as
    // Z1. The binary32 representation this test pinned before
    // (`z1_nybn_parity.py` arm `ny_bn`, Rust query 0 observed within 1.3e-10
    // of it) gave -0.29998938733419667 / -0.1791655373991373: the exact path
    // recovers +2.7137e-3 / +1.0896e-3 of margin at the root.
    //
    // So: the margin must match BOTH the exact-f64 replica arm and Z1's own
    // binary64-BatchNorm numbers to 1e-6 (the replicas agree to ~4e-11); the
    // gain over the binary32 arm is asserted explicitly. The Rust values have
    // not been observed on this branch yet: the pins are the replica's
    // PREDICTION, and a failure is new evidence, not a tolerance to widen.
    let z1_f64_batch_norm = [-0.297_275_693_101_987_26f64, -0.178_075_972_643_201_85];
    let z1_exact_f64_arm = [-0.297_275_693_139_100_7f64, -0.178_075_972_658_066_38];
    let z1_binary32_arm = [-0.299_989_387_334_196_67f64, -0.179_165_537_399_137_3];
    assert_eq!(lbs.len(), 2, "two queries in, {} margins out", lbs.len());
    for (q, &got) in lbs.iter().enumerate() {
        assert!(got.is_finite(), "query {q}: non-finite margin {got}");
        assert!(
            (got - z1_exact_f64_arm[q]).abs() <= 1e-6,
            "query {q}: margin {got} vs the exact-f64-BatchNorm replica arm {} \
             (Z1 binary64 BatchNorm {}, binary32 arm {})",
            z1_exact_f64_arm[q],
            z1_f64_batch_norm[q],
            z1_binary32_arm[q]
        );
        assert!(
            (got - z1_f64_batch_norm[q]).abs() <= 1e-6,
            "query {q}: margin {got} vs Z1's binary64-BatchNorm replica {}",
            z1_f64_batch_norm[q]
        );
        let replica_gain = z1_exact_f64_arm[q] - z1_binary32_arm[q];
        assert!(
            got - z1_binary32_arm[q] >= 0.9 * replica_gain,
            "query {q}: margin {got} recovers less than 90% of the replica's {replica_gain:e} \
             gain over the binary32 BatchNorm arm {}",
            z1_binary32_arm[q]
        );
    }
}

/// A depth-`depth` leaf of `root` that still contains `witness`: halve the
/// widest free dimension, keeping the half the witness lies in.
fn leaf_containing(root: &BoundedTensor, witness: &[f32], depth: usize) -> BoundedTensor {
    let mut lo: Vec<f32> = root.lower().iter().copied().collect();
    let mut hi: Vec<f32> = root.upper().iter().copied().collect();
    for _ in 0..depth {
        let dim = (0..lo.len())
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .expect("dimension");
        if hi[dim] <= lo[dim] {
            break;
        }
        let mid = lo[dim] + (hi[dim] - lo[dim]) / 2.0;
        if witness[dim] <= mid {
            hi[dim] = mid;
        } else {
            lo[dim] = mid;
        }
    }
    for d in 0..lo.len() {
        assert!(
            lo[d] <= witness[d] && witness[d] <= hi[d],
            "leaf lost the witness"
        );
    }
    boxed(&lo, &hi, root.shape())
}

#[test]
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
fn child_zono_sat_leaf_never_closes() {
    // (model, box lo, box hi, witness from the banked sat counterexample,
    //  thresholds a (Y >= a) and b (Y <= b)).
    let rows: [(&str, [f32; 5], [f32; 5], [f32; 5], f64, f64); 2] = [
        (
            // row 4: cGAN_imgSz32_nCh_3 prop_0 (witness Y = 0.51617 >= a)
            "cGAN_imgSz32_nCh_3.onnx",
            [
                0.3641427159309387,
                -1.0,
                -0.656203031539917,
                0.5323579907417297,
                -1.0,
            ],
            [
                0.504142701625824,
                -0.8600000143051147,
                -0.5162030458450317,
                0.6723580360412598,
                -0.8600000143051147,
            ],
            [
                0.504142701625824,
                -1.0,
                -0.656203031539917,
                0.5323579907417297,
                -1.0,
            ],
            0.5121427178382874,
            0.35614272952079773,
        ),
        (
            // row 9: cGAN_imgSz64_nCh_1 prop_1 (witness Y = 0.599656 <= b)
            "cGAN_imgSz64_nCh_1.onnx",
            [
                0.6046702265739441,
                -1.0,
                -0.41744861006736755,
                -0.9750000238418579,
                0.27076369524002075,
            ],
            [
                0.6546701788902283,
                -0.949999988079071,
                -0.3674485981464386,
                -0.9750000238418579,
                0.3207637071609497,
            ],
            [
                0.6046702265739441,
                -0.949999988079071,
                -0.3674485981464386,
                -0.9750000238418579,
                0.3207637071609497,
            ],
            0.6596702337265015,
            0.5996702313423157,
        ),
    ];
    for (model, lo, hi, witness, a, b) in rows {
        let (graph, shape) = load_graph(model);
        let root = boxed(&lo, &hi, &shape);
        let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
        let root_ibp = graph
            .collect_node_bounds_with_engine(&root, None)
            .expect("root IBP map");
        // The witness must be a GENUINE counterexample under the program's
        // own semantics, or this control is VOID.
        let x: Vec<f64> = witness.iter().map(|&v| f64::from(v)).collect();
        let mut y = f64::NAN;
        let last = program.steps().len() - 1;
        program
            .evaluate_points(&x, 1, |step, values| {
                if step == Some(last) {
                    y = values[0];
                }
            })
            .expect("witness evaluation");
        let query_values = [a - y, y - b];
        assert!(
            query_values.iter().any(|&v| v <= 0.0),
            "{model}: the banked witness (Y = {y}) violates no disjunct: VOID"
        );
        let queries = [
            SpecQuery {
                coeffs: vec![-1.0],
                constant: a,
            },
            SpecQuery {
                coeffs: vec![1.0],
                constant: -b,
            },
        ];
        let opts = ZonoForwardOptions {
            max_generators: 16_384,
            record_trace: true,
            row_par: RowPar::Seq,
        };
        for depth in [0usize, 4, 8, 12] {
            let leaf = leaf_containing(&root, &witness, depth);
            for inherited in [Vec::new(), vec![&root_ibp]] {
                let outcome = program
                    .forward(&leaf, &opts, &inherited, None, None)
                    .expect("leaf zonotope");
                let lbs =
                    spec_lower_bounds(&program, outcome.trace.as_ref().expect("trace"), &queries)
                        .expect("leaf margins");
                for q in 0..2 {
                    assert!(
                        lbs[q] <= query_values[q],
                        "{model} depth {depth}: certified lb {} exceeds the witness's own \
                         query value {} — a false bound",
                        lbs[q],
                        query_values[q]
                    );
                }
                assert!(
                    lbs.iter().any(|&lb| lb <= 0.0),
                    "{model} depth {depth}: a leaf containing a counterexample closed"
                );
            }
        }
    }
}

/// Row 19's model. It never reaches [`ZonoProgram`]: ny cannot build its
/// graph at all. Its spectral-norm cones compute `W_orig / sigma` with
/// `sigma = u . (W v)` from INITIALIZERS through `MatMul`, and ny's constant
/// folder publishes a both-constant `MatMul` only when every entry is EXACTLY
/// a binary32 number (`const_fold::ops::elementwise::try_fold_matmul_exact`);
/// these products are not, so the cone lowers to a sound skip and `Div_2`
/// (`gen.snlinear0.weight_orig / onnx::Div_245`) is a binary `Div` with ONE
/// activation operand, which `ny-build` refuses ("expects 2 activation inputs
/// for binary op"). The scored path builds its graph the same way (read, not
/// run: `beta_crown::model_load::load_model`, same graph options, the preset
/// loader config), so it refuses there too — the banked row-20 `sat` comes
/// from the ORT-only upfront attack, which runs before any graph is built,
/// and `census_board_complete_2026-08-25/BOARD_TABLE.md` records the same
/// `GRAPH_FAIL`.
const SMALL_TRANSFORMER: &str = "cGAN_imgSz32_nCh_3_small_transformer.onnx";

#[test]
fn child_zono_refuses_unsupported_ops() {
    let mut refused_at_compile = 0usize;
    for model in [
        SMALL_TRANSFORMER,
        "cGAN_imgSz32_nCh_3_nonlinear_activations.onnx",
        "cGAN_imgSz32_nCh_3_upsample.onnx",
    ] {
        let (graph, shape) = try_load_graph(model);
        let graph = match graph {
            Ok(graph) => graph,
            Err(error) => {
                // Refused at the LOAD boundary: no GraphNetwork exists, so no
                // lane — this one included — can be handed the model. Only the
                // diagnosed refusal of the one model known to hit it passes;
                // anything else is a new fact and fails here.
                assert!(
                    model == SMALL_TRANSFORMER
                        && error.contains("(Div)")
                        && error.contains("expects 2 activation inputs for binary op"),
                    "{model}: graph construction refused for an unexpected reason: {error}"
                );
                continue;
            }
        };
        let root = boxed(&[0.0; 5], &[0.01; 5], &shape);
        match ZonoProgram::compile_with_ibp_shapes(&graph, &root) {
            Err(ZonoRefusal::UnsupportedLayer { .. } | ZonoRefusal::UnsupportedGraph(_)) => {}
            Err(other) => panic!("{model}: refused for the wrong reason: {other}"),
            Ok(_) => panic!("{model}: an unsupported model compiled"),
        }
        refused_at_compile += 1;
    }
    // nonlinear_activations (Tanh, Sigmoid) and upsample (Upsample) build a
    // graph on the scored path, so BOTH must reach — and fail — the program's
    // own op-set check; a refusal test that refused nothing itself is VOID.
    assert!(
        refused_at_compile >= 2,
        "only {refused_at_compile} model(s) reached ZonoProgram::compile: VOID"
    );
    // Positive control: every ConvT cGAN (padding variant included) compiles.
    for model in [
        "cGAN_imgSz32_nCh_1.onnx",
        "cGAN_imgSz32_nCh_3.onnx",
        "cGAN_imgSz64_nCh_1.onnx",
        "cGAN_imgSz64_nCh_3.onnx",
        "cGAN_imgSz32_nCh_1_transposedConvPadding_1.onnx",
    ] {
        let (graph, shape) = load_graph(model);
        let root = boxed(&[0.0; 5], &[0.01; 5], &shape);
        let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root)
            .unwrap_or_else(|refusal| panic!("{model}: must compile: {refusal}"));
        assert!(program.steps().len() > 10);
        if model.contains("Padding") {
            cross_check_semantics(&graph, &program, &root);
        }
    }
}

// ------------------------------------------ review(m8b): the batch evaluator on real cGANs

/// Every verdict-bearing field of two real-model outcomes is bitwise equal.
fn assert_outcomes_bitwise(
    a: &ny_propagate::zono_core::ZonoForwardOutcome,
    b: &ny_propagate::zono_core::ZonoForwardOutcome,
    what: &str,
) {
    let mut keys: Vec<&String> = a.published.keys().collect();
    keys.sort();
    let mut keys_b: Vec<&String> = b.published.keys().collect();
    keys_b.sort();
    assert_eq!(keys, keys_b, "{what}: published nodes differ");
    assert!(!keys.is_empty(), "{what}: nothing published (VOID)");
    for key in keys {
        let (x, y) = (&a.published[key], &b.published[key]);
        assert_eq!(x.shape(), y.shape(), "{what}: `{key}` shape");
        let bits = |t: &BoundedTensor| -> (Vec<u32>, Vec<u32>) {
            (
                t.lower().iter().map(|v| v.to_bits()).collect(),
                t.upper().iter().map(|v| v.to_bits()).collect(),
            )
        };
        assert_eq!(bits(x), bits(y), "{what}: `{key}` differs");
    }
    let bits64 = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(
        bits64(&a.output_lo),
        bits64(&b.output_lo),
        "{what}: output lower"
    );
    assert_eq!(
        bits64(&a.output_hi),
        bits64(&b.output_hi),
        "{what}: output upper"
    );
    assert_eq!(a.relus, b.relus, "{what}: relu summaries");
    assert_eq!(
        (a.stats.gens_final, a.stats.gens_peak, a.stats.folded),
        (b.stats.gens_final, b.stats.gens_peak, b.stats.folded),
        "{what}: generator stats"
    );
    assert_eq!(
        a.stats.max_remainder.to_bits(),
        b.stats.max_remainder.to_bits(),
        "{what}: max remainder"
    );
}

/// review(m8b): the batch evaluator on the REAL ConvT cGANs (rows 2, 7, 10,
/// 12). A tranche of three children of each official box (depths 0, 4 and 8;
/// the middle one inherits the root IBP map) runs through
/// `ZonoProgram::forward_batch`, and each child is compared BITWISE with its
/// own sequential `forward` (published maps, output box, ReLU summaries,
/// generator stats). The builder's bit-identity pins are synthetic fixtures;
/// this is the M8b claim on the production layer shapes (large ConvT /
/// Conv patch grids, thousands of generator rows, column blocks). VOID unless
/// a multi-block row task actually ran.
#[test]
fn review_child_zono_forward_batch_is_bitwise_the_serial_forward_on_real_cgans() {
    use ny_propagate::zono_core::{StepOp, ZonoBatchChild};
    let opts = ZonoForwardOptions {
        max_generators: 16_384,
        record_trace: false,
        row_par: RowPar::Seq,
    };
    for (row_index, row) in CONTAINMENT_ROWS.iter().enumerate() {
        let (graph, shape) = load_graph(row.model);
        let root = boxed(&row.lo, &row.hi, &shape);
        let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root)
            .unwrap_or_else(|refusal| panic!("{}: program refused: {refusal}", row.model));
        let root_ibp = graph
            .collect_node_bounds_with_engine(&root, None)
            .expect("root IBP map");
        let affine_steps = program
            .steps()
            .iter()
            .filter(|s| matches!(s.op, StepOp::Affine(_)))
            .count();
        let mut rng = Rng::new(7_000 + row_index as u64);
        let boxes: Vec<BoundedTensor> = [0usize, 4, 8]
            .iter()
            .map(|&depth| random_child(&root, depth, &mut rng))
            .collect();
        let children: Vec<ZonoBatchChild<'_>> = boxes
            .iter()
            .enumerate()
            .map(|(c, input)| ZonoBatchChild {
                input,
                inherited: if c == 1 { vec![&root_ibp] } else { Vec::new() },
                deadline: None,
            })
            .collect();
        let batch = program.forward_batch(&children, &opts);
        assert_eq!(batch.len(), children.len());
        let mut multi_block = false;
        for (c, (child, (result, _nanos))) in children.iter().zip(&batch).enumerate() {
            let what = format!("{} child {c}", row.model);
            let got = result
                .as_ref()
                .unwrap_or_else(|refusal| panic!("{what}: batched forward refused: {refusal}"));
            let reference = program
                .forward(child.input, &opts, &child.inherited, None, None)
                .unwrap_or_else(|refusal| panic!("{what}: sequential forward refused: {refusal}"));
            assert_eq!(
                reference.stats.row_tasks, 0,
                "{what}: the reference ran row tasks"
            );
            multi_block |= got.stats.row_tasks > affine_steps;
            assert_outcomes_bitwise(&reference, got, &what);
        }
        assert!(
            multi_block,
            "{}: no child ran a multi-block row task ({affine_steps} affine steps): VOID",
            row.model
        );
    }
}
