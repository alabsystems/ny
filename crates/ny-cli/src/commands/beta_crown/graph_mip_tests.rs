// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the DAG-aware MIP encoder (`encode_graph`):
//   * increment 1 — a Linear→ReLU→Linear→ReLU→Linear chain must be
//     BYTE-FOR-BYTE identical to `ny_mip::encode_feedforward`;
//   * increment 2 — BatchNorm rows equal the hand-computed per-channel affine,
//     and the MIP feasible set contains the true (x, BN(x)) point / rejects a
//     perturbed one;
//   * increment 3 — a residual `out = F(in) + in` Add row references BOTH the
//     F-branch cols and the input cols, and accepts the true point / rejects a
//     violating twin.
//   * increment 4 — a Conv2d node unfolds (im2col) to the SAME Linear equality
//     rows and its feasible set contains the true (x, conv(x)) forward pass; and
//     the DELTA box inflation bounds every non-input affine column by its node
//     box ±1e-4 while keeping a box-face point feasible.

use super::*;

use std::path::Path;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::Bound;
use ny_mip::{encode_feedforward, CertifiedLinearLowerProofRoute, MilpProblem};
use ny_propagate::layers::{
    AddConstantLayer, AddLayer, BatchNormLayer, ConcatLayer, Conv2dLayer, DivConstantLayer,
    DivLayer, FlattenLayer, GatherLayer, LinearLayer, MulBinaryLayer, MulConstantLayer, ReLULayer,
    ReduceSumLayer, SigmoidLayer, SliceLayer, SubConstantLayer, SubLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer, Verifier, NETWORK_INPUT};

static IMB_AY_TAIL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn imb_ay_tail_encoding_cap_admits_measured_cgan_tail_only_within_hard_limits() {
    // Exact sealed row-5 observation at the Relu_17 seam.  This regression
    // guards against silently restoring the pre-AY 64-binary rejection.
    assert!(imb_ay_tail_encoding_within_caps(7_716, 4_945, 112));
    assert!(imb_ay_tail_encoding_within_caps(
        IMB_AY_TAIL_MAX_COLS,
        IMB_AY_TAIL_MAX_ROWS,
        IMB_AY_TAIL_MAX_BINARIES,
    ));
    assert!(!imb_ay_tail_encoding_within_caps(7_716, 4_945, 129));
    assert!(!imb_ay_tail_encoding_within_caps(
        IMB_AY_TAIL_MAX_COLS + 1,
        4_945,
        112,
    ));
    assert!(!imb_ay_tail_encoding_within_caps(
        7_716,
        IMB_AY_TAIL_MAX_ROWS + 1,
        112,
    ));
    assert!(!imb_ay_tail_input_within_caps(0));
    assert!(imb_ay_tail_input_within_caps(IMB_AY_TAIL_MAX_INPUTS));
    assert!(!imb_ay_tail_input_within_caps(IMB_AY_TAIL_MAX_INPUTS + 1));
    assert_eq!(AY_TAIL_AFFINE_REACHABILITY_ROWS, 2);
    assert_eq!(IMB_AY_TAIL_AFFINE_ROWS, 4);
    assert_eq!(IMB_AY_TAIL_AFFINE_MAX_LATENT_INPUTS, 16);
    assert_eq!(IMB_AY_TAIL_AFFINE_MAX_NNZ, 16_384);
    // Dense K=2 support rows at the measured 2,048-wide cGAN seam plus all
    // five regional input coordinates consume exactly this many new nonzeros:
    // P appears in both sides, while A^- and A^+ each appear once.
    assert_eq!(2 * 2 * 2_048 + 2 * 2 * 5, 8_212);
    // Deliberate const assert: documents that the measured seam fits the cap.
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(8_212 <= IMB_AY_TAIL_AFFINE_MAX_NNZ);
    }
}

#[test]
fn imb_ay_affine_reachability_rows_have_exact_signs_bounds_and_shared_input() {
    let mut problem = MilpProblem::new();
    let seam = vec![
        problem.add_col(0.0, -10.0, 10.0),
        problem.add_col(0.0, -20.0, 20.0),
    ];
    let directions = Array2::from_shape_vec((2, 2), vec![1.0, -2.0, 0.5, 3.0]).unwrap();
    let lower_a = Array2::from_shape_vec((2, 2), vec![4.0, -5.0, 0.0, 6.0]).unwrap();
    let lower_b = Array1::from_vec(vec![7.0, -8.0]);
    let upper_a = Array2::from_shape_vec((2, 2), vec![-9.0, 10.0, 11.0, 0.0]).unwrap();
    let upper_b = Array1::from_vec(vec![12.0, -13.0]);

    let (latent, nnz) = add_affine_reachability_rows(
        &mut problem,
        &seam,
        &[-1.0, -2.0],
        &[3.0, 4.0],
        &directions,
        &lower_a,
        &lower_b,
        &upper_a,
        &upper_b,
    )
    .expect("well-formed K=2 rows");

    assert_eq!(latent, vec![Col(2), Col(3)]);
    assert_eq!(problem.num_cols(), 4);
    assert_eq!(problem.cols()[2].lb, -1.0);
    assert_eq!(problem.cols()[2].ub, 3.0);
    assert_eq!(problem.cols()[3].lb, -2.0);
    assert_eq!(problem.cols()[3].ub, 4.0);
    assert_eq!(problem.num_rows(), 4);
    assert_eq!(nnz, 14);

    let rows = problem.rows();
    // P_0 y - A^-_0 x >= b^-_0.
    assert_eq!(rows[0].lb, 7.0);
    assert_eq!(rows[0].ub, f64::INFINITY);
    assert_eq!(
        rows[0].coeffs,
        vec![(0, 1.0), (1, -2.0), (2, -4.0), (3, 5.0)]
    );
    // P_0 y - A^+_0 x <= b^+_0.
    assert_eq!(rows[1].lb, f64::NEG_INFINITY);
    assert_eq!(rows[1].ub, 12.0);
    assert_eq!(
        rows[1].coeffs,
        vec![(0, 1.0), (1, -2.0), (2, 9.0), (3, -10.0)]
    );
    // Both second-support rows reuse the exact same latent x columns.
    assert_eq!(rows[2].lb, -8.0);
    assert_eq!(rows[2].ub, f64::INFINITY);
    assert_eq!(rows[2].coeffs, vec![(0, 0.5), (1, 3.0), (3, -6.0)]);
    assert_eq!(rows[3].lb, f64::NEG_INFINITY);
    assert_eq!(rows[3].ub, -13.0);
    assert_eq!(rows[3].coeffs, vec![(0, 0.5), (1, 3.0), (2, -11.0)]);
}

#[test]
fn imb_ay_affine_reachability_rows_reject_nonfinite_or_malformed_inputs() {
    let mut problem = MilpProblem::new();
    let seam = vec![problem.add_col(0.0, -1.0, 1.0)];
    let directions = Array2::from_shape_vec((2, 1), vec![1.0, f32::NAN]).unwrap();
    let a = Array2::zeros((2, 1));
    let b = Array1::zeros(2);
    let before = (problem.num_cols(), problem.num_rows());
    assert!(add_affine_reachability_rows(
        &mut problem,
        &seam,
        &[-1.0],
        &[1.0],
        &directions,
        &a,
        &b,
        &a,
        &b,
    )
    .is_none());
    assert_eq!(
        (problem.num_cols(), problem.num_rows()),
        before,
        "preflight rejection must not partially mutate the exact model"
    );
}

/// Flatten an f32 weight matrix (row-major) to the `Vec<f64>` shape
/// `encode_feedforward` expects — the SAME `as f64` cast the graph encoder
/// applies to `LinearLayer::weight`, so the two stay bit-identical.
fn weight_to_f64(w: &Array2<f32>) -> Vec<f64> {
    w.iter().map(|&x| x as f64).collect()
}

fn bias_to_f64(b: &Array1<f32>) -> Vec<f64> {
    b.iter().map(|&x| x as f64).collect()
}

#[test]
fn imb_ay_tail_oracle_certifies_fixed_affine_residual() {
    let _test_guard = IMB_AY_TAIL_TEST_LOCK.lock().unwrap();
    // tail(x)=2x+1 over x∈[0,1], objective=tail and p=x:
    // objective·tail(x)-p·x = x+1, whose exact minimum is 1.
    let mut tail = GraphNetwork::new();
    let affine = LinearLayer::new(
        Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
        Some(Array1::from_vec(vec![1.0])),
    )
    .unwrap();
    tail.add_node(GraphNode::from_input("out", Layer::Linear(affine)));
    tail.set_output("out");
    let seam = ny_tensor::BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    // A caller-supplied output singleton at x=1 would raise the restricted
    // residual minimum to 2. It is sound only for that reachable subset, not
    // for every point in the seam box, so the universal authority path must
    // ignore it and still prove the true seam-box minimum near 1.
    let supplied_output = ny_tensor::BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    )
    .unwrap();
    let supplied = HashMap::from([("out".to_string(), supplied_output)]);
    let certificate = imb_ay_tail_certificate_exact_result(
        &tail,
        &seam,
        &supplied,
        &[1.0],
        &[1.0],
        None,
        Instant::now() + Duration::from_secs(10),
    )
    .expect("affine tail should have an exact linear certificate");
    assert!(certificate.lower < 1.0);
    assert!(certificate.lower > 0.99);
    assert_eq!(
        certificate.proof_route,
        CertifiedLinearLowerProofRoute::RelaxationEntailment
    );
    assert_eq!(certificate.ay_tree_leaves, 0);
    assert_eq!(certificate.ny_cert_farkas_replays, 1);

    let selected = imb_ay_tail_certificate_exact_result(
        &tail,
        &seam,
        &supplied,
        &[1.0],
        &[1.0],
        Some(0.5),
        Instant::now() + Duration::from_secs(10),
    )
    .expect("decision-only tail threshold should have an exact linear certificate");
    assert_eq!(selected.lower, 0.5);
    assert_eq!(
        selected.proof_route,
        CertifiedLinearLowerProofRoute::RelaxationEntailment
    );
    assert_eq!(selected.ay_tree_leaves, 0);
    assert_eq!(selected.ny_cert_farkas_replays, 1);
}

#[test]
fn imb_ay_tail_reachability_oracle_uses_the_exact_prefix_premise() {
    let _test_guard = IMB_AY_TAIL_TEST_LOCK.lock().unwrap();
    // tail(y)=y over y∈[-1,1]. The root seam alone contains counterexamples to
    // y>0.25, while the certified regional fact y>=0.5 excludes all of them.
    let mut tail = GraphNetwork::new();
    let identity =
        LinearLayer::new(Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(), None).unwrap();
    tail.add_node(GraphNode::from_input("out", Layer::Linear(identity)));
    tail.set_output("out");
    let seam = ny_tensor::BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    // The reachability oracle API deliberately accepts no caller-supplied
    // full-graph node boxes: only the explicit affine premise may restrict the
    // fresh seam-box model.
    let certificate = imb_ay_tail_reachability_certificate_exact_result(
        &tail,
        &seam,
        &[1.0],
        &[1.0],
        0.5,
        0.25,
        Instant::now() + Duration::from_secs(10),
    )
    .expect("the exact reachability row should admit an exact linear proof");
    assert_eq!(certificate.lower, 0.25);
    assert_eq!(
        certificate.proof_route,
        CertifiedLinearLowerProofRoute::RelaxationEntailment
    );
    assert_eq!(certificate.ay_tree_leaves, 0);
    assert_eq!(certificate.ny_cert_farkas_replays, 1);

    assert!(
        imb_ay_tail_reachability_certificate_exact_result(
            &tail,
            &seam,
            &[1.0],
            &[1.0],
            2.0,
            0.25,
            Instant::now() + Duration::from_secs(10),
        )
        .is_none(),
        "a premise above the directed seam-box maximum must decline before proof"
    );
    assert!(
        imb_ay_tail_reachability_certificate_exact_result(
            &tail,
            &seam,
            &[1.0],
            &[0.0],
            0.0,
            0.25,
            Instant::now() + Duration::from_secs(10),
        )
        .is_none(),
        "an identically-zero reachability direction is deliberately non-authoritative"
    );
    assert!(
        imb_ay_tail_reachability_certificate_exact_result(
            &tail,
            &seam,
            &[1.0],
            &[1.0],
            -1.0,
            0.25,
            Instant::now() + Duration::from_secs(10),
        )
        .is_none(),
        "a weak premise leaves a real counterexample and cannot certify"
    );
    assert!(
        imb_ay_tail_reachability_certificate_exact_result(
            &tail,
            &seam,
            &[1.0],
            &[1.0],
            0.5,
            0.5,
            Instant::now() + Duration::from_secs(10),
        )
        .is_none(),
        "the non-strict premise and decision rows meet at equality, so strict proof must fail"
    );
}

#[test]
fn imb_ay_tail_reachability_affine_box_guard_is_directed_and_fail_closed() {
    let seam = ny_tensor::BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-2.0, -3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0, 5.0]).unwrap(),
    )
    .unwrap();
    let upper = imb_affine_box_upper(&[2.0, -1.0], &seam).expect("finite affine range");
    assert!(upper >= 11.0);
    assert!(imb_affine_box_upper(&[1.0], &seam).is_none());
    assert!(imb_affine_box_upper(&[f32::NAN, 1.0], &seam).is_none());
}

#[test]
fn imb_ay_tail_oracle_rejects_dimension_mismatch_before_solve() {
    let _test_guard = IMB_AY_TAIL_TEST_LOCK.lock().unwrap();
    let mut tail = GraphNetwork::new();
    let affine =
        LinearLayer::new(Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(), None).unwrap();
    tail.add_node(GraphNode::from_input("out", Layer::Linear(affine)));
    tail.set_output("out");
    let seam = ny_tensor::BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();
    assert!(imb_ay_tail_certificate_exact_result(
        &tail,
        &seam,
        &HashMap::new(),
        &[1.0],
        &[1.0, 2.0],
        None,
        Instant::now() + Duration::from_secs(1),
    )
    .is_none());
}

/// Build the three (W, b) layers shared by both encodings. Deliberately mixes
/// signs and includes explicit `0.0` weights to exercise the identical
/// zero-skip path in both encoders.
fn build_layers() -> (
    (Array2<f32>, Array1<f32>),
    (Array2<f32>, Array1<f32>),
    (Array2<f32>, Array1<f32>),
) {
    // W1: 4 x 3 (two exact zeros).
    let w1 = Array2::from_shape_vec(
        (4, 3),
        vec![
            0.5, -0.3, 0.0, //
            0.2, 0.1, -0.4, //
            -0.1, 0.6, 0.3, //
            0.7, -0.2, 0.0,
        ],
    )
    .unwrap();
    let b1 = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.3]);

    // W2: 5 x 4 (several zeros).
    let w2 = Array2::from_shape_vec(
        (5, 4),
        vec![
            0.1, -0.2, 0.3, 0.0, //
            -0.4, 0.5, 0.0, 0.2, //
            0.6, 0.1, -0.3, 0.4, //
            0.0, -0.5, 0.2, 0.1, //
            0.3, 0.3, -0.1, -0.2,
        ],
    )
    .unwrap();
    let b2 = Array1::from_vec(vec![0.0, 0.1, -0.1, 0.2, -0.3]);

    // W3: 2 x 5 (output layer, no ReLU after).
    let w3 = Array2::from_shape_vec(
        (2, 5),
        vec![
            0.2, -0.1, 0.4, 0.0, 0.3, //
            -0.3, 0.5, 0.1, -0.2, 0.0,
        ],
    )
    .unwrap();
    let b3 = Array1::from_vec(vec![0.05, -0.05]);

    ((w1, b1), (w2, b2), (w3, b3))
}

/// Pre-activation bounds for the two ReLUs, chosen to hit ALL three big-M
/// branches: `l >= 0` (stable active), `u <= 0` (stable inactive), and unstable.
fn relu_bounds() -> (Vec<Bound>, Vec<Bound>) {
    let relu1 = vec![
        Bound::new(-1.0, 2.0),  // unstable
        Bound::new(0.5, 3.0),   // active   (l >= 0)
        Bound::new(-3.0, -1.0), // inactive (u <= 0)
        Bound::new(-2.0, 1.5),  // unstable
    ];
    let relu2 = vec![
        Bound::new(-1.5, 0.5),  // unstable
        Bound::new(-4.0, -2.0), // inactive
        Bound::new(0.2, 2.0),   // active
        Bound::new(-0.5, 0.5),  // unstable
        Bound::new(-2.0, 3.0),  // unstable
    ];
    (relu1, relu2)
}

#[test]
fn graph_chain_encoding_matches_feedforward() {
    let ((w1, b1), (w2, b2), (w3, b3)) = build_layers();
    let (relu1_bounds, relu2_bounds) = relu_bounds();

    // 3 network inputs, each in [-1, 1].
    let input_bounds = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
    ];

    // --- Reference: the flat chain encoder. ---
    let flat_weights = vec![weight_to_f64(&w1), weight_to_f64(&w2), weight_to_f64(&w3)];
    let flat_biases = vec![bias_to_f64(&b1), bias_to_f64(&b2), bias_to_f64(&b3)];
    let layer_dims = vec![3usize, 4, 5, 2];
    let intermediate_bounds = vec![relu1_bounds.clone(), relu2_bounds.clone()];

    let ff = encode_feedforward(
        &flat_weights,
        &flat_biases,
        &layer_dims,
        &input_bounds,
        &intermediate_bounds,
    )
    .expect("encode_feedforward");
    let ff_parts = ff.into_parts();
    let ff_problem = ff_parts.problem;

    // --- DAG encoder: build the equivalent GraphNetwork. ---
    // A leading Flatten node exercises the identity path AND must leave the
    // encoding untouched (flatten of a 1-D input is a no-op on columns).
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "flatten0",
        Layer::Flatten(FlattenLayer::new(1)),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
        vec!["flatten0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let mut node_bounds = HashMap::new();
    node_bounds.insert("relu1".to_string(), relu1_bounds);
    node_bounds.insert("relu2".to_string(), relu2_bounds);

    // delta = 0.0: the increment-1 byte-identical invariant is only defined
    // WITHOUT DELTA box inflation (production uses `encode_graph`, delta = 1e-4).
    let g = encode_graph_with_delta(&graph, &input_bounds, &node_bounds, 0.0)
        .expect("encode_graph_with_delta");
    let g_problem = g.problem;

    // 1. Identical column count.
    assert_eq!(
        g_problem.num_cols(),
        ff_problem.num_cols(),
        "column count mismatch"
    );

    // 2. Identical integer (binary ReLU indicator) count.
    let ff_int = ff_problem.cols().iter().filter(|c| c.integer).count();
    let g_int = g_problem.cols().iter().filter(|c| c.integer).count();
    assert_eq!(g_int, ff_int, "integer-column count mismatch");
    // Two ReLU layers with 2 + 3 unstable neurons => 5 binaries.
    assert_eq!(g_int, 5, "expected 5 binary indicators");
    assert_eq!(g.binary_vars.len(), 5, "binary_vars length");
    assert_eq!(g.binary_widths.len(), 5, "binary_widths length");

    // 3. Identical row count.
    assert_eq!(
        g_problem.num_rows(),
        ff_problem.num_rows(),
        "row count mismatch"
    );

    // 4. Byte-for-byte identical columns (bounds + obj + integrality) — the
    //    strongest form of "same row set modulo ordering": since the insertion
    //    order is identical, so are the specs, so any solver yields the same
    //    feasibility verdict on any spec.
    assert_eq!(
        g_problem.cols(),
        ff_problem.cols(),
        "column specs differ from encode_feedforward"
    );

    // 5. Byte-for-byte identical rows (bounds + sparse coeffs, in order).
    assert_eq!(
        g_problem.rows(),
        ff_problem.rows(),
        "row specs differ from encode_feedforward"
    );

    // 6. Output columns match the reference output frontier.
    assert_eq!(
        g.output_vars, ff_parts.output_vars,
        "output columns differ from encode_feedforward"
    );
    assert_eq!(g.output_vars.len(), 2, "expected 2 output columns");

    // 7. Input columns are the first 3 columns (the `_input` block).
    assert_eq!(g.input_vars, ff_parts.input_vars, "input columns differ");
    assert_eq!(g.input_vars.len(), 3, "expected 3 input columns");
}

// ── nn4sys mscn: empirical layer-variant introspection ──────────────────────

/// Path to the vnncomp2026 nn4sys onnx directory (skip the test when absent —
/// benchmark checkouts are not part of the repo).
const NN4SYS_ONNX_DIR: &str = "benchmarks/vnncomp2026/benchmarks/nn4sys/1.0/onnx";

fn nn4sys_onnx_path(file: &str) -> Option<std::path::PathBuf> {
    // Tests run with cwd = crate root (crates/ny-cli); the benchmarks live at
    // the workspace root. Try both.
    for base in ["../..", "."] {
        let p = Path::new(base).join(NN4SYS_ONNX_DIR).join(file);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// EMPIRICAL introspection (not an assertion test): prints every node's Layer
/// variant + inputs + declared shape for the two mscn cardinality models, so
/// the encoder's op coverage is grounded in what ny's loader ACTUALLY produces
/// rather than in the raw ONNX op list. Run with `-- --nocapture`.
#[test]
fn mscn_introspect_layer_variants() {
    for file in ["mscn_128d.onnx", "mscn_128d_dual.onnx"] {
        let Some(path) = nn4sys_onnx_path(file) else {
            eprintln!("mscn_introspect_layer_variants: {file} not found; skipping");
            continue;
        };
        let graph = crate::commands::vnncomp::load_graph_network(&path)
            .unwrap_or_else(|e| panic!("load_graph_network({file}): {e}"));
        eprintln!("==== {file} ====");
        eprintln!(
            "input declared shape: {:?}; output node: {}",
            graph.declared_shape(NETWORK_INPUT),
            graph.output_name()
        );
        let exec = graph.exec_order().expect("exec_order");
        for name in exec {
            let node = graph.node(name).expect("node");
            let layer = node.layer();
            let detail = match layer {
                Layer::Linear(l) => {
                    format!(
                        "in={} out={} bias={}",
                        l.in_features(),
                        l.out_features(),
                        l.bias.is_some()
                    )
                }
                Layer::Slice(s) => format!("axis={} start={} end={}", s.axis, s.start, s.end),
                Layer::Concat(c) => format!("axis={} input_shapes={:?}", c.axis, c.input_shapes),
                Layer::ReduceSum(r) => format!("axes={:?} keepdims={}", r.axes, r.keepdims),
                Layer::MatMul(m) => format!("transpose_b={}", m.transpose_b()),
                Layer::MulConstant(m) => format!("const shape={:?}", m.constant().shape()),
                Layer::DivConstant(d) => format!("const shape={:?}", d.constant().shape()),
                Layer::AddConstant(a) => format!("const shape={:?}", a.constant().shape()),
                Layer::SubConstant(s) => {
                    format!(
                        "const shape={:?} reverse={}",
                        s.constant().shape(),
                        s.reverse
                    )
                }
                Layer::Gather(g) => format!("{g:?}"),
                Layer::Reshape(r) => format!("{r:?}"),
                Layer::Flatten(f) => format!("{f:?}"),
                Layer::Squeeze(s) => format!("{s:?}"),
                Layer::Unsqueeze(u) => format!("{u:?}"),
                _ => String::new(),
            };
            eprintln!(
                "{name}: {} inputs={:?} declared_shape={:?} {detail}",
                layer.layer_type(),
                node.inputs(),
                graph.declared_shape(name)
            );
        }
    }
}

#[test]
fn graph_mip_gate_defaults_on() {
    // The whole-net Graph-MIP escalation is DEFAULT-ON (2026-07-21):
    // verdict-sound + memory-safe encode-nnz cap (5M) + ay-ladder binary cap
    // (1024), so over-scale graph instances decline before encoding.
    // `NY_GRAPH_MIP=0` is the kill switch. Exercise the pure gate parser so
    // parallel tests never race through process-global environment mutation.
    assert!(graph_mip_enabled_from_value(None), "unset defaults on");
    assert!(
        graph_mip_enabled_from_value(Some("1")),
        "explicit on stays on"
    );
    assert!(
        !graph_mip_enabled_from_value(Some("0")),
        "explicit zero is the off switch"
    );
}

fn graph_mip_manual_probe_enabled() -> bool {
    std::env::var("NY_GRAPH_MIP").ok().as_deref() == Some("1")
}

#[test]
fn graph_mip_gate_value_contract() {
    assert!(graph_mip_enabled_from_value(None), "unset is default-on");
    assert!(
        graph_mip_enabled_from_value(Some("1")),
        "explicit 1 remains enabled"
    );
    assert!(
        graph_mip_enabled_from_value(Some("")),
        "only exact 0 disables"
    );
    assert!(
        graph_mip_enabled_from_value(Some("false")),
        "only exact 0 disables"
    );
    assert!(!graph_mip_enabled_from_value(Some("0")), "0 is kill switch");
}

// ── increment 2 / 3 test helpers ───────────────────────────────────────────

/// Self-contained feasibility check on the solver-neutral IR: a full column
/// assignment is IN the MIP feasible set iff every column bound and every row
/// bound holds (within `tol`). No solver needed — the feasible set is exactly
/// the set of points satisfying all rows + column bounds, so this directly
/// answers "does the MIP contain this point?".
fn point_is_feasible(problem: &MilpProblem, assign: &[f64], tol: f64) -> bool {
    assert_eq!(assign.len(), problem.num_cols(), "assignment length");
    for (i, spec) in problem.cols().iter().enumerate() {
        if assign[i] < spec.lb - tol || assign[i] > spec.ub + tol {
            return false;
        }
    }
    for row in problem.rows() {
        let s: f64 = row.coeffs.iter().map(|&(col, w)| w * assign[col]).sum();
        if s < row.lb - tol || s > row.ub + tol {
            return false;
        }
    }
    true
}

/// Forward one Linear layer in f64 with the SAME `as f64` casts the encoder
/// applies to the f32 weight/bias, so the produced values satisfy the encoder's
/// equality rows to machine precision.
fn linear_fwd(w: &Array2<f32>, b: &Array1<f32>, x: &[f64]) -> Vec<f64> {
    let (out_f, in_f) = w.dim();
    (0..out_f)
        .map(|i| {
            let mut s = b[i] as f64;
            for j in 0..in_f {
                s += (w[[i, j]] as f64) * x[j];
            }
            s
        })
        .collect()
}

/// Forward the exact per-channel BatchNorm affine `y_i = scale_c·x_i + bias_c`
/// with `elements_per_channel = 1` (each element is its own channel — the
/// Linear→BatchNorm case), mirroring `encode_batchnorm_node`.
fn bn_fwd(scale: &[f32], bias: &[f32], x: &[f64]) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(c, &xi)| (scale[c] as f64) * xi + (bias[c] as f64))
        .collect()
}

// ── increment 2: BatchNorm ─────────────────────────────────────────────────

#[test]
fn graph_batchnorm_rows_are_exact_affine_and_feasible() {
    // Linear1 (2->3) -> BatchNorm(3 channels) -> ReLU -> Linear2 (3->2).
    let w1 = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]).unwrap();
    let b1 = Array1::from_vec(vec![0.5, 0.5, 0.5]);
    // Per-channel BatchNorm affine coefficients (already baked scale/bias).
    let scale = [2.0f32, 0.5, 1.5];
    let bias = [0.1f32, -0.2, 0.3];
    let w2 = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 0.0, 1.0, 1.0]).unwrap();
    let b2 = Array1::from_vec(vec![0.0, 0.0]);

    // Input box [1, 2]^2 keeps every BatchNorm output strictly positive, so the
    // ReLU is stable-active (pass-through) => no ReLU columns/rows/binaries and a
    // fully deterministic column layout: [input(2) | linear1(3) | bn(3) | linear2(2)].
    let input_bounds = vec![Bound::new(1.0, 2.0), Bound::new(1.0, 2.0)];
    // Sound pre-activation bounds on the BN output (all l > 0 => stable active).
    let relu_bounds = vec![
        Bound::new(3.1, 5.1),
        Bound::new(0.55, 1.05),
        Bound::new(2.55, 4.05),
    ];

    let bn_layer = BatchNormLayer::from_scale_bias(
        Array1::from_vec(scale.to_vec()).into_dyn(),
        Array1::from_vec(bias.to_vec()).into_dyn(),
    )
    .unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn_layer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["bn".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");

    let mut node_bounds = HashMap::new();
    node_bounds.insert("relu".to_string(), relu_bounds);

    let g = encode_graph(&graph, &input_bounds, &node_bounds).expect("encode_graph");

    // Stable ReLU => no binaries and the exact 2+3+3+2 = 10 column layout.
    assert_eq!(g.binary_vars.len(), 0, "stable ReLU must add no binaries");
    assert_eq!(g.problem.num_cols(), 10, "input+linear1+bn+linear2 columns");
    assert_eq!(g.input_vars.len(), 2);
    assert_eq!(g.output_vars.len(), 2);

    // (1) The three BatchNorm rows equal the hand-computed per-channel affine.
    // Row order: linear1 (3 rows) then bn (3 rows) then linear2 (2 rows).
    let n1 = 3usize;
    let d_in = 2usize;
    for i in 0..n1 {
        let row = &g.problem.rows()[n1 + i];
        assert_eq!(row.coeffs.len(), 2, "BN row {i}: out & in coeff only");
        // out_i coefficient is +1, referencing the bn output column (after
        // input(2) + linear1(3) => column d_in + n1 + i).
        assert_eq!(row.coeffs[0], (d_in + n1 + i, 1.0), "BN row {i} out coeff");
        // in_i coefficient is -scale_c, referencing linear1's output column
        // (column d_in + i).
        assert_eq!(
            row.coeffs[1],
            (d_in + i, -(scale[i] as f64)),
            "BN row {i} in coeff = -scale"
        );
        // Equality RHS: lb == ub == bias_c.
        assert_eq!(row.lb, row.ub, "BN row {i} must be an equality");
        assert_eq!(row.lb, bias[i] as f64, "BN row {i} rhs = bias");
    }

    // (2) The MIP feasible set contains the true (x, BN(x)) forward pass and
    // rejects a perturbed BN output.
    let x = vec![1.5f64, 1.5];
    let lin1 = linear_fwd(&w1, &b1, &x);
    let bn = bn_fwd(&scale, &bias, &lin1); // ReLU is pass-through here.
    let lin2 = linear_fwd(&w2, &b2, &bn);
    let mut assign = Vec::new();
    assign.extend_from_slice(&x);
    assign.extend_from_slice(&lin1);
    assign.extend_from_slice(&bn);
    assign.extend_from_slice(&lin2);
    assert_eq!(assign.len(), g.problem.num_cols());
    assert!(
        point_is_feasible(&g.problem, &assign, 1e-6),
        "true (x, BN(x)) forward pass must be feasible"
    );

    // Perturb the channel-0 BatchNorm output => breaks `out_0 - scale_0*in_0 = bias_0`.
    let mut bad = assign.clone();
    bad[d_in + n1] += 0.5;
    assert!(
        !point_is_feasible(&g.problem, &bad, 1e-6),
        "a perturbed BN output must be rejected"
    );
}

// ── increment 3: residual Add (the DAG piece) ───────────────────────────────

#[test]
fn graph_residual_add_wiring_and_feasible_twin() {
    // out = F(in) + in, with F = Linear(2->3) -> ReLU -> Linear(3->2). The Add's
    // two inputs are the F-branch output ("f_lin2") and the network input ("in"),
    // a real skip connection: the encoder must resolve BOTH upstream blocks.
    let w1 = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let b1 = Array1::from_vec(vec![0.5, 0.5, 0.5]);
    let w2 = Array2::from_shape_vec((2, 3), vec![0.5, -0.5, 1.0, 1.0, 0.5, -0.5]).unwrap();
    let b2 = Array1::from_vec(vec![0.1, -0.2]);

    // Input box [1, 2]^2 keeps F's pre-activations positive => ReLU stable-active
    // (pass-through), so the column layout is deterministic:
    // [input(2) | f_lin1(3) | f_lin2(2) | add(2)].
    let input_bounds = vec![Bound::new(1.0, 2.0), Bound::new(1.0, 2.0)];
    let relu_bounds = vec![
        Bound::new(1.5, 2.5),
        Bound::new(1.5, 2.5),
        Bound::new(2.5, 4.5),
    ];

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "f_lin1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "f_relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["f_lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "f_lin2",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
        vec!["f_relu".to_string()],
    ));
    // Residual sum: inputs = [F-branch, skip]. AddLayer is element-wise A + B.
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["f_lin2".to_string(), NETWORK_INPUT.to_string()],
    ));
    graph.set_output("add");

    let mut node_bounds = HashMap::new();
    node_bounds.insert("f_relu".to_string(), relu_bounds);

    let g = encode_graph(&graph, &input_bounds, &node_bounds).expect("encode_graph");

    assert_eq!(g.binary_vars.len(), 0, "stable ReLU must add no binaries");
    assert_eq!(g.problem.num_cols(), 9, "input+f_lin1+f_lin2+add columns");
    let d = 2usize;
    assert_eq!(g.input_vars.len(), d);
    assert_eq!(g.output_vars.len(), d);

    // (1) Each residual Add row wires the output col to BOTH parents: the input
    // (skip) col AND a DISTINCT F-branch col — the DAG-awareness this exercises.
    // Add rows are the last `d` rows.
    let num_rows = g.problem.num_rows();
    let input_set: std::collections::HashSet<usize> = g.input_vars.iter().map(|c| c.0).collect();
    for i in 0..d {
        let row = &g.problem.rows()[num_rows - d + i];
        assert_eq!(row.coeffs.len(), 3, "Add row {i}: out + two parents");
        assert_eq!(row.lb, 0.0, "Add row {i} equality lb");
        assert_eq!(row.ub, 0.0, "Add row {i} equality ub");

        // The +1.0 coefficient is the Add output column.
        let out_col = g.output_vars[i].0;
        let plus: Vec<usize> = row
            .coeffs
            .iter()
            .filter(|&&(_, w)| w == 1.0)
            .map(|&(c, _)| c)
            .collect();
        assert_eq!(plus, vec![out_col], "Add row {i}: +1 coeff is the out col");

        // The two -1.0 coefficients are the two parents: one is the skip (input)
        // col, the other is an F-branch col (distinct, and NOT an input col).
        let minus: Vec<usize> = row
            .coeffs
            .iter()
            .filter(|&&(_, w)| w == -1.0)
            .map(|&(c, _)| c)
            .collect();
        assert_eq!(minus.len(), 2, "Add row {i}: two -1 parent coeffs");
        assert!(
            minus.contains(&g.input_vars[i].0),
            "Add row {i} must reference the skip/input col"
        );
        let f_branch: Vec<usize> = minus
            .iter()
            .copied()
            .filter(|c| !input_set.contains(c))
            .collect();
        assert_eq!(
            f_branch.len(),
            1,
            "Add row {i} must reference exactly one (distinct) F-branch col"
        );
        assert_ne!(f_branch[0], out_col, "F-branch col distinct from out col");
    }

    // (2) The true residual forward pass out = F(in) + in is feasible; a twin that
    // violates it (perturbed output) is rejected.
    let x = vec![1.5f64, 1.5];
    let flin1 = linear_fwd(&w1, &b1, &x);
    let flin2 = linear_fwd(&w2, &b2, &flin1); // ReLU pass-through.
    let add: Vec<f64> = (0..d).map(|i| flin2[i] + x[i]).collect();
    let mut assign = Vec::new();
    assign.extend_from_slice(&x);
    assign.extend_from_slice(&flin1);
    assign.extend_from_slice(&flin2);
    assign.extend_from_slice(&add);
    assert_eq!(assign.len(), g.problem.num_cols());
    assert!(
        point_is_feasible(&g.problem, &assign, 1e-6),
        "true out = F(in) + in must be feasible"
    );

    // Break the skip sum at output 0 => `out_0 - f_0 - in_0 = 0` is violated.
    let mut bad = assign.clone();
    bad[g.output_vars[0].0] += 0.7;
    assert!(
        !point_is_feasible(&g.problem, &bad, 1e-6),
        "a point violating out = F(in) + in must be rejected"
    );
}

// ── increment 4: Conv2d (im2col) ─────────────────────────────────────────────

/// A tiny Conv→ReLU→Linear graph. Asserts (1) the encoded MIP's conv rows are
/// BYTE-IDENTICAL to `unfold_conv2d_to_linear` applied then encoded as a Linear
/// (structural), and (2) the MIP feasible set contains the true (x, conv(x))
/// forward pass at multiple sample points and excludes a perturbed-output twin.
#[test]
fn graph_conv2d_unfolds_to_linear_rows_and_is_feasible() {
    // Conv: IC=1, OC=1, 2x2 kernel of 0.5, no bias-shift, stride 1, no padding,
    // over a 1x3x3 input => 1x2x2 output (4 elements). Each output is
    // 0.5*(sum of a 2x2 input patch).
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5f32, 0.5, 0.5, 0.5]).unwrap();
    let conv = Conv2dLayer::with_input_shape(
        kernel,
        Some(Array1::from_vec(vec![0.0f32])),
        (1, 1),
        (0, 0),
        3,
        3,
    )
    .unwrap();

    // Linear head 4->2.
    let w_lin =
        Array2::from_shape_vec((2, 4), vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let b_lin = Array1::from_vec(vec![0.0f32, 0.0]);

    // Input box [0.5, 1]^9 keeps every conv output in [1, 2] => ReLU stable-active
    // (pass-through) => deterministic layout [input(9) | conv(4) | linear(2)].
    let input_bounds = vec![Bound::new(0.5, 1.0); 9];
    let relu_bounds = vec![Bound::new(1.0, 2.0); 4];

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "conv",
        Layer::Conv2d(conv.clone()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["conv".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(LinearLayer::new(w_lin.clone(), Some(b_lin.clone())).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear");

    let mut node_bounds = HashMap::new();
    node_bounds.insert("relu".to_string(), relu_bounds);

    // Production path (delta = DELTA); the conv node carries no box => free conv
    // columns, so the conv rows are delta-independent.
    let g = encode_graph(&graph, &input_bounds, &node_bounds).expect("encode_graph");

    // Stable ReLU => no binaries; 9 + 4 + 2 = 15 columns; conv occupies rows 0..4.
    assert_eq!(g.binary_vars.len(), 0, "stable ReLU must add no binaries");
    assert_eq!(g.problem.num_cols(), 15, "input+conv+linear columns");
    assert_eq!(g.input_vars.len(), 9);
    assert_eq!(g.output_vars.len(), 2);

    // (1) STRUCTURAL: the conv's 4 rows equal a fresh `unfold_conv2d_to_linear`
    // (via `unfold_conv_node`) encoded through `encode_linear_node` with the SAME
    // column indices (input 0..9, conv 9..13) — hence byte-identical rows.
    let lin_ref = unfold_conv_node(&conv, 3, 3, "conv").expect("unfold_conv_node");
    let mut p2 = MilpProblem::new();
    let in_cols2: Vec<_> = (0..9)
        .map(|_| p2.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY))
        .collect();
    let _ = encode_linear_node(&mut p2, &lin_ref, &in_cols2, None, 0.0, "conv")
        .expect("encode_linear_node reference");
    assert_eq!(p2.num_rows(), 4, "reference conv unfold has 4 rows");
    assert_eq!(
        &g.problem.rows()[0..4],
        p2.rows(),
        "graph conv rows must be byte-identical to unfold-then-encode-as-Linear"
    );

    // (2) FEASIBILITY at multiple sample points: the true (x, conv(x), linear)
    // forward pass is feasible; a perturbed conv output is rejected. conv(x) is
    // the unfolded-Linear forward (im2col correctness is covered by
    // mip_preprocess::tests::test_unfold_conv2d_identity_no_padding).
    let ref_bias = lin_ref.bias.clone().unwrap_or_else(|| Array1::zeros(4));
    let samples: [Vec<f64>; 3] = [
        vec![0.75; 9],
        vec![0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
        vec![1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0],
    ];
    for x in samples.iter() {
        let conv_out = linear_fwd(&lin_ref.weight, &ref_bias, x); // 4 = conv(x)
        let head = linear_fwd(&w_lin, &b_lin, &conv_out); // ReLU pass-through
        let mut assign = Vec::new();
        assign.extend_from_slice(x);
        assign.extend_from_slice(&conv_out);
        assign.extend_from_slice(&head);
        assert_eq!(assign.len(), g.problem.num_cols());
        assert!(
            point_is_feasible(&g.problem, &assign, 1e-6),
            "true (x, conv(x), head) forward pass must be feasible"
        );

        // Perturb the first conv output => breaks its im2col equality row.
        let mut bad = assign.clone();
        bad[9] += 0.5;
        assert!(
            !point_is_feasible(&g.problem, &bad, 1e-6),
            "a perturbed conv output must be rejected"
        );
    }
}

// ── increment 4: DELTA box inflation ─────────────────────────────────────────

/// Assert that every non-input node column's `[lo, hi]` equals its supplied node
/// box inflated by `±DELTA`, that the network INPUT box is used EXACTLY (not
/// inflated), and that a point sitting on the exact (pre-inflation) box FACE is
/// feasible (the inflation pushes the real column bound strictly past the face).
#[test]
fn graph_delta_inflates_node_boxes_and_keeps_face_feasible() {
    // input(2) -> linear1(2->3) -> relu -> linear2(3->2).
    let w1 = Array2::from_shape_vec((3, 2), vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let b1 = Array1::from_vec(vec![0.0f32, 0.0, 0.0]);
    let w2 = Array2::from_shape_vec((2, 3), vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let b2 = Array1::from_vec(vec![0.0f32, 0.0]);

    // Input box [0.5, 1]^2: linear1 = [x0, x1, x0+x1] in [0.5,1]x[0.5,1]x[1,2];
    // linear2 = [linear1_0, linear1_2] in [0.5,1]x[1,2]. All pre-activations
    // stay >= 0.5 > DELTA => ReLU stable-active even after inflation.
    let input_bounds = vec![Bound::new(0.5, 1.0), Bound::new(0.5, 1.0)];
    let lin1_box = vec![
        Bound::new(0.5, 1.0),
        Bound::new(0.5, 1.0),
        Bound::new(1.0, 2.0),
    ];
    let lin2_box = vec![Bound::new(0.5, 1.0), Bound::new(1.0, 2.0)];

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");

    // Output-box semantics: every affine node carries its own box; the ReLU reads
    // its input node's box as the pre-activation (encode_graph tries "relu" first,
    // then falls back to "linear1").
    let mut node_bounds = HashMap::new();
    node_bounds.insert("linear1".to_string(), lin1_box.clone());
    node_bounds.insert("linear2".to_string(), lin2_box.clone());

    // Production path: delta = DELTA (1e-4).
    let g = encode_graph(&graph, &input_bounds, &node_bounds).expect("encode_graph");

    assert_eq!(g.binary_vars.len(), 0, "stable ReLU must add no binaries");
    assert_eq!(g.problem.num_cols(), 7, "input(2)+linear1(3)+linear2(2)");
    let cols = g.problem.cols();

    // (1) Input columns: EXACT box (never inflated).
    let approx = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(
        approx(cols[0].lb, 0.5) && approx(cols[0].ub, 1.0),
        "input col 0 exact"
    );
    assert!(
        approx(cols[1].lb, 0.5) && approx(cols[1].ub, 1.0),
        "input col 1 exact"
    );

    // (2) Every non-input node column == its node box inflated by +/-DELTA.
    // Columns 2..5 = linear1 (box = lin1_box); columns 5..7 = linear2 (lin2_box).
    let expected: Vec<Bound> = lin1_box.iter().chain(lin2_box.iter()).copied().collect();
    for (k, bnd) in expected.iter().enumerate() {
        let c = &cols[2 + k];
        let lo = bnd.lower() as f64 - DELTA;
        let hi = bnd.upper() as f64 + DELTA;
        assert!(
            approx(c.lb, lo) && approx(c.ub, hi),
            "non-input col {} must equal box [{}, {}] +/- DELTA, got [{}, {}]",
            2 + k,
            bnd.lower(),
            bnd.upper(),
            c.lb,
            c.ub
        );
    }

    // (3) A point on the exact pre-inflation box FACE is feasible. Input (1, 1)
    // drives linear1 = [1, 1, 2] and linear2 = [1, 2] — each intermediate sits on
    // its box's UPPER face; feasible because the column ub = face + DELTA.
    let x = vec![1.0f64, 1.0];
    let flin1 = linear_fwd(&w1, &b1, &x); // = [1, 1, 2] (ReLU pass-through)
    let flin2 = linear_fwd(&w2, &b2, &flin1); // = [1, 2]
    let mut assign = Vec::new();
    assign.extend_from_slice(&x);
    assign.extend_from_slice(&flin1);
    assign.extend_from_slice(&flin2);
    assert_eq!(assign.len(), g.problem.num_cols());
    // Sanity: the face values equal the declared box upper bounds.
    assert!(
        approx(flin1[2], 2.0) && approx(flin2[1], 2.0),
        "face values hit box.hi"
    );
    assert!(
        point_is_feasible(&g.problem, &assign, 1e-9),
        "a point on the exact box face must be feasible under DELTA inflation"
    );
}

// ===========================================================================
// Increment 5 (wiring) tests: MipParts conversion, spec emission, node-bounds
// shim, and the graph encodability predicate.
// ===========================================================================

/// Rebuild the increment-1 chain fixture (graph + input box + node bounds).
fn chain_fixture() -> (GraphNetwork, Vec<Bound>, HashMap<String, Vec<Bound>>) {
    let ((w1, b1), (w2, b2), (w3, b3)) = build_layers();
    let (relu1_bounds, relu2_bounds) = relu_bounds();
    let input_bounds = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
    ];

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let mut node_bounds = HashMap::new();
    node_bounds.insert("relu1".to_string(), relu1_bounds);
    node_bounds.insert("relu2".to_string(), relu2_bounds);
    (graph, input_bounds, node_bounds)
}

/// The reference feedforward encoder for the same chain.
fn chain_reference_encoder() -> ny_mip::MipEncoder {
    let ((w1, b1), (w2, b2), (w3, b3)) = build_layers();
    let (relu1_bounds, relu2_bounds) = relu_bounds();
    let input_bounds = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
    ];
    encode_feedforward(
        &[weight_to_f64(&w1), weight_to_f64(&w2), weight_to_f64(&w3)],
        &[bias_to_f64(&b1), bias_to_f64(&b2), bias_to_f64(&b3)],
        &[3usize, 4, 5, 2],
        &input_bounds,
        &[relu1_bounds, relu2_bounds],
    )
    .expect("encode_feedforward")
}

/// (b) `into_parts` preserves every field and sizes `num_cols` off the problem
/// — so `MipSolver`'s warm-start/phase-split machinery sees the same shape it
/// gets from `MipEncoder::into_parts`.
#[test]
fn graph_encoding_into_parts_matches_fields() {
    let (graph, input_bounds, node_bounds) = chain_fixture();
    let g = encode_graph_with_delta(&graph, &input_bounds, &node_bounds, 0.0)
        .expect("encode_graph_with_delta");
    let (in_vars, out_vars, bin_vars, bin_widths) = (
        g.input_vars.clone(),
        g.output_vars.clone(),
        g.binary_vars.clone(),
        g.binary_widths.clone(),
    );
    let expected_cols = g.problem.num_cols();
    let parts = g.into_parts();
    assert_eq!(parts.input_vars, in_vars, "input_vars must move unchanged");
    assert_eq!(
        parts.output_vars, out_vars,
        "output_vars must move unchanged"
    );
    assert_eq!(
        parts.binary_vars, bin_vars,
        "binary_vars must move unchanged"
    );
    assert_eq!(
        parts.binary_widths, bin_widths,
        "binary_widths must move unchanged"
    );
    assert_eq!(
        parts.num_cols, expected_cols,
        "num_cols must equal the problem's"
    );
    assert_eq!(
        parts.problem.num_cols(),
        expected_cols,
        "problem must move unchanged"
    );
    assert_eq!(
        parts.problem.margin_row(),
        None,
        "encoding shape alone must not infer a decision margin"
    );
}

/// The leaf/whole-net decision-row API is the sole Graph-MIP caller gate for
/// AY's margin reframe. Unset and `0` must append the same unmarked row; exact
/// `1` marks only that row, and its identity survives the move into `MipParts`.
/// The explicit value seam avoids mutating process-global environment state in
/// this regression test.
#[test]
fn graph_violation_row_margin_reframe_gate_is_exact() {
    for (raw, expected_marked) in [(None, false), (Some("0"), false), (Some("1"), true)] {
        let (graph, input_bounds, node_bounds) = chain_fixture();
        let mut g = encode_graph_with_delta(&graph, &input_bounds, &node_bounds, 0.0)
            .expect("encode_graph_with_delta");
        let expected_row = ny_mip::ir::Row(g.problem.num_rows());

        g.add_violation_row_with_margin_reframe(
            &[1.0, -1.0],
            0.25,
            ay_margin_reframe_enabled_from_value(raw),
        )
        .expect("valid decision row");
        let expected_margin = expected_marked.then_some(expected_row);
        assert_eq!(
            g.problem.margin_row(),
            expected_margin,
            "unexpected marker for NY_AY_MARGIN_REFRAME={raw:?}"
        );

        let parts = g.into_parts();
        assert_eq!(parts.problem.margin_row(), expected_margin);
        let row = &parts.problem.rows()[expected_row.0];
        assert_eq!(row.lb, f64::NEG_INFINITY);
        assert_eq!(row.ub, 0.25);
        assert_eq!(row.coeffs.len(), 2);
    }

    for malformed in ["", "01", "true", "yes", " 1"] {
        assert!(
            !ay_margin_reframe_enabled_from_value(Some(malformed)),
            "non-exact force value {malformed:?} must fail closed"
        );
    }
}

/// A failed second force-on append must preserve the atomic `add_margin_row`
/// contract: no duplicate decision row may remain after the error.
#[test]
fn graph_violation_row_duplicate_force_is_atomic() {
    let (graph, input_bounds, node_bounds) = chain_fixture();
    let mut g = encode_graph_with_delta(&graph, &input_bounds, &node_bounds, 0.0)
        .expect("encode_graph_with_delta");
    g.add_violation_row_with_margin_reframe(&[1.0, -1.0], 0.25, true)
        .expect("first force-on decision row");
    let rows_before = g.problem.rows().to_vec();
    let marker_before = g.problem.margin_row();

    let error = g
        .add_violation_row_with_margin_reframe(&[1.0, -1.0], 0.5, true)
        .expect_err("a second distinct decision row must be rejected");

    assert!(error.to_string().contains("already marked"));
    assert_eq!(g.problem.rows(), rows_before);
    assert_eq!(g.problem.margin_row(), marker_before);
}

/// (c) Spec emission byte-equality: stamping the same VNN-LIB constraints via
/// `GraphMipEncoding::add_output_constraint` and via `MipEncoder::constrain_*`
/// yields byte-identical problems (rows in the same order with the same bounds
/// and coefficients) — extending the increment-1 invariant to the spec surface.
/// Strict variants must encode identically to their non-strict twins.
#[test]
fn graph_spec_emission_matches_feedforward_constraints() {
    use ny_onnx::vnnlib::OutputConstraint;

    let (graph, input_bounds, node_bounds) = chain_fixture();
    let mut g = encode_graph_with_delta(&graph, &input_bounds, &node_bounds, 0.0)
        .expect("encode_graph_with_delta");
    let mut ff = chain_reference_encoder();

    // One of each supported shape, plus a strict twin (encodes identically).
    let constraints = [
        OutputConstraint::LessEq(0, 1),
        OutputConstraint::GreaterEq(1, 0),
        OutputConstraint::LessEqConst(0, 0.5),
        OutputConstraint::GreaterEqConst(1, -0.25),
        OutputConstraint::LessThan(0, 1),
        OutputConstraint::GreaterThanConst(0, 0.125),
    ];
    for c in &constraints {
        g.add_output_constraint(c).expect("graph spec emission");
    }
    assert_eq!(
        g.problem.margin_row(),
        None,
        "ordinary one-sided output constraints are not decision margins"
    );
    ff.constrain_output_leq(0, 1).unwrap();
    ff.constrain_output_geq(1, 0).unwrap();
    ff.constrain_output_leq_const(0, 0.5).unwrap();
    ff.constrain_output_geq_const(1, -0.25).unwrap();
    ff.constrain_output_leq(0, 1).unwrap(); // strict LessThan twin
    ff.constrain_output_geq_const(0, 0.125).unwrap(); // strict GreaterThanConst twin

    let ff_problem = ff.into_parts().problem;
    assert_eq!(
        g.problem.rows(),
        ff_problem.rows(),
        "constraint rows differ from MipEncoder::constrain_*"
    );
    assert_eq!(
        g.problem.cols(),
        ff_problem.cols(),
        "constraint stamping must not touch columns"
    );
}

/// (a) The node-bounds shim flattens each `BoundedTensor` to `Vec<Bound>` in
/// flattened element order, keyed by the same node names.
#[test]
fn flatten_node_bounds_preserves_order_and_keys() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0f32, -2.0, -3.0, -4.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let bt = ny_tensor::BoundedTensor::new(lower, upper).unwrap();
    let mut map = HashMap::new();
    map.insert("conv1".to_string(), bt);

    let flat = crate::commands::beta_crown::graph_mip_escalate::flatten_node_bounds(&map)
        .expect("flatten");
    let b = flat.get("conv1").expect("key preserved");
    assert_eq!(b.len(), 4, "flattened length");
    for (i, (lo, hi)) in [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0)]
        .iter()
        .enumerate()
    {
        assert_eq!(b[i].lower(), *lo as f32, "lower[{i}] in flattened order");
        assert_eq!(b[i].upper(), *hi as f32, "upper[{i}] in flattened order");
    }
}

/// (d) Graph encodability: the chain fixture has 5 unstable ReLUs, so it is
/// encodable at a budget of 5 and NOT at 4; a graph with an unsupported layer
/// is never encodable; a ReLU without a pre-activation box is rejected
/// (fail-closed — the encoder would bail).
#[test]
fn graph_encodability_counts_unstable_relus() {
    use crate::commands::beta_crown::dispatch::is_mip_encodable_graph;

    let (graph, _input_bounds, node_bounds) = chain_fixture();
    assert!(
        is_mip_encodable_graph(&graph, &node_bounds, 5),
        "5 unstable ReLUs fit a budget of 5"
    );
    assert!(
        !is_mip_encodable_graph(&graph, &node_bounds, 4),
        "5 unstable ReLUs exceed a budget of 4"
    );

    // No pre-activation box for relu2 -> fail closed.
    let mut missing = node_bounds;
    missing.remove("relu2");
    assert!(
        !is_mip_encodable_graph(&graph, &missing, 100),
        "a ReLU without a box must be rejected"
    );

    // Unsupported layer -> never encodable.
    let mut bad = GraphNetwork::new();
    bad.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer),
        vec![NETWORK_INPUT.to_string()],
    ));
    bad.set_output("sig");
    assert!(
        !is_mip_encodable_graph(&bad, &HashMap::new(), 100),
        "unsupported layers must be rejected"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Increment 5 — nn4sys mscn ops (Slice / Gather / Concat / row-batched Linear /
// *Constant / ReduceSum / Sub / pinned MulBinary / pinned Div / Sigmoid peel).
// ═════════════════════════════════════════════════════════════════════════════

fn free_box(n: usize) -> Vec<Bound> {
    vec![Bound::new(-1.0, 1.0); n]
}

/// Slice is pure index aliasing: no new columns/rows, output columns ARE the
/// exact input columns per the forward index math — including the
/// input-shape-reconstruction path when the producer is the (shapeless)
/// `_input` sentinel.
#[test]
fn graph_slice_is_pure_index_aliasing() {
    // _input is logically [2, 4] (flat 8), but carries NO declared shape.
    // s1 = rows 0..1  (axis -2)  => [1, 4]  => input cols 0..4
    // s2 = cols 1..3 of s1 (axis -1) => [1, 2] => input cols 1, 2
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "s1",
        Layer::Slice(SliceLayer::new(-2, 0, 1)),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "s2",
        Layer::Slice(SliceLayer::new(-1, 1, 3)),
        vec!["s1".to_string()],
    ));
    graph.set_output("s2");
    graph.set_declared_shape("s1", vec![1, 4]);
    graph.set_declared_shape("s2", vec![1, 2]);

    let g = encode_graph(&graph, &free_box(8), &HashMap::new()).expect("encode_graph");
    assert_eq!(g.problem.num_cols(), 8, "aliasing must add no columns");
    assert_eq!(g.problem.num_rows(), 0, "aliasing must add no rows");
    assert_eq!(g.output_vars, vec![g.input_vars[1], g.input_vars[2]]);
    // Intermediate aliasing (s1 = row 0 = first 4 input cols).
    assert_eq!(g.node_cols["s1"], g.input_vars[0..4].to_vec());

    // Fail-closed: no declared shape anywhere => cannot derive the index map.
    let mut bare = GraphNetwork::new();
    bare.add_node(GraphNode::new(
        "s",
        Layer::Slice(SliceLayer::new(-1, 0, 2)),
        vec![NETWORK_INPUT.to_string()],
    ));
    bare.set_output("s");
    let err = encode_graph(&bare, &free_box(8), &HashMap::new());
    assert!(err.is_err(), "shapeless Slice must fail closed");
}

/// Concat is pure index aliasing with exact axis interleaving (3-ary, 2-D,
/// non-identity order so a block-order bug cannot pass).
#[test]
fn graph_concat_axis_interleaving_is_exact() {
    // _input [2, 3]: a = cols 0..2 => [2, 2] (cols 0,1,3,4); b = col 2 => [2, 1]
    // (cols 2,5). cat = Concat([b, a], axis=-1) => [2, 3]:
    //   out(r, 0) = b(r, 0); out(r, 1..3) = a(r, 0..2)
    // => cols [2, 0, 1, 5, 3, 4].
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "a",
        Layer::Slice(SliceLayer::new(-1, 0, 2)),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "b",
        Layer::Slice(SliceLayer::new(-1, 2, 3)),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cat",
        Layer::Concat(ConcatLayer::new(-1)),
        vec!["b".to_string(), "a".to_string()],
    ));
    graph.set_output("cat");
    graph.set_declared_shape("a", vec![2, 2]);
    graph.set_declared_shape("b", vec![2, 1]);
    graph.set_declared_shape("cat", vec![2, 3]);

    let g = encode_graph(&graph, &free_box(6), &HashMap::new()).expect("encode_graph");
    assert_eq!(g.problem.num_cols(), 6, "aliasing must add no columns");
    assert_eq!(g.problem.num_rows(), 0, "aliasing must add no rows");
    let iv = &g.input_vars;
    assert_eq!(
        g.output_vars,
        vec![iv[2], iv[0], iv[1], iv[5], iv[3], iv[4]],
        "concat must interleave the two blocks exactly along axis -1"
    );

    // 3-ary 1-D concat in permuted order.
    let mut g3 = GraphNetwork::new();
    for (name, (s, e)) in [("p", (0usize, 2usize)), ("q", (2, 3)), ("r", (3, 6))] {
        g3.add_node(GraphNode::new(
            name,
            Layer::Slice(SliceLayer::new(0, s, e)),
            vec![NETWORK_INPUT.to_string()],
        ));
        g3.set_declared_shape(name, vec![e - s]);
    }
    g3.add_node(GraphNode::new(
        "cat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["r".to_string(), "p".to_string(), "q".to_string()],
    ));
    g3.set_output("cat");
    g3.set_declared_shape("cat", vec![6]);
    g3.set_declared_shape(NETWORK_INPUT, vec![6]);
    let g3e = encode_graph(&g3, &free_box(6), &HashMap::new()).expect("encode_graph");
    let iv = &g3e.input_vars;
    assert_eq!(
        g3e.output_vars,
        vec![iv[3], iv[4], iv[5], iv[0], iv[1], iv[2]],
        "3-ary concat must append blocks in the node's input order"
    );
}

/// Gather with embedded constant indices is pure index aliasing, including
/// negative-index resolution and repeated indices (repeated aliases).
#[test]
fn graph_gather_constant_indices_alias_exactly() {
    // _input [2, 3]; gather axis=1, indices [2, 0, -1] => out [2, 3]:
    // out(r, k) = in(r, [2, 0, 2][k]) => cols [2, 0, 2, 5, 3, 5].
    let indices = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2i64, 0, -1]).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "gather",
        Layer::Gather(GatherLayer::new(1, Some(indices), vec![])),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.set_output("gather");
    graph.set_declared_shape(NETWORK_INPUT, vec![2, 3]);

    let g = encode_graph(&graph, &free_box(6), &HashMap::new()).expect("encode_graph");
    assert_eq!(g.problem.num_cols(), 6, "aliasing must add no columns");
    assert_eq!(g.problem.num_rows(), 0, "aliasing must add no rows");
    let iv = &g.input_vars;
    assert_eq!(
        g.output_vars,
        vec![iv[2], iv[0], iv[2], iv[5], iv[3], iv[5]],
        "gather must alias (with repeats) via the resolved indices"
    );

    // Fail-closed: out-of-range index.
    let bad = ArrayD::from_shape_vec(IxDyn(&[1]), vec![3i64]).unwrap();
    let mut gb = GraphNetwork::new();
    gb.add_node(GraphNode::new(
        "gather",
        Layer::Gather(GatherLayer::new(1, Some(bad), vec![])),
        vec![NETWORK_INPUT.to_string()],
    ));
    gb.set_output("gather");
    gb.set_declared_shape(NETWORK_INPUT, vec![2, 3]);
    assert!(encode_graph(&gb, &free_box(6), &HashMap::new()).is_err());
}

/// Row-batched Linear (`[R, in] -> [R, out]`, LinearLayer's last-axis
/// contraction): the emitted rows are the exact per-(r, i) affine equalities,
/// with the zero-weight skip and `-b_i` right-hand side.
#[test]
fn graph_row_batched_linear_rows_are_exact() {
    // in = [2, 3] (R = 2), W: 2x3 with one exact zero, b = [0.5, -1.0].
    let w = Array2::from_shape_vec((2, 3), vec![1.0f32, 0.0, 2.0, -0.5, 3.0, 0.25]).unwrap();
    let b = Array1::from_vec(vec![0.5f32, -1.0]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "lin",
        Layer::Linear(LinearLayer::new(w.clone(), Some(b.clone())).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.set_output("lin");

    let g = encode_graph(&graph, &free_box(6), &HashMap::new()).expect("encode_graph");
    assert_eq!(g.problem.num_cols(), 6 + 4, "input(6) + out(2x2)");
    assert_eq!(g.problem.num_rows(), 4, "one equality row per (r, i)");
    assert_eq!(g.output_vars.len(), 4);

    for r in 0..2usize {
        for i in 0..2usize {
            let row = &g.problem.rows()[r * 2 + i];
            let y = g.output_vars[r * 2 + i].0;
            // Expected: nonzero (x_{3r+j}, w_ij) then (y, -1); rhs = -b_i.
            let mut expect: Vec<(usize, f64)> = Vec::new();
            for j in 0..3usize {
                let wij = w[[i, j]] as f64;
                if wij != 0.0 {
                    expect.push((g.input_vars[3 * r + j].0, wij));
                }
            }
            expect.push((y, -1.0));
            assert_eq!(row.coeffs, expect, "row ({r},{i}) coefficients");
            assert_eq!(row.lb, row.ub, "equality");
            assert_eq!(row.lb, -(b[i] as f64), "rhs = -b_i");
        }
    }

    // Fail-closed: a column count that is not a multiple of in_features.
    let g_bad = encode_graph(&graph, &free_box(7), &HashMap::new());
    assert!(g_bad.is_err(), "non-divisible input width must fail closed");
}

/// ReduceSum emits one exact ones-row per output element over the reduced
/// axes (negative axis resolution, keepdims both ways).
#[test]
fn graph_reduce_sum_rows_are_exact_ones_rows() {
    // _input [3, 2]; axes = [-2] (rows), keepdims = false => out [2]:
    //   out_c = x[0,c] + x[1,c] + x[2,c] (flat cols c, c+2, c+4).
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "rs",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-2], false)),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.set_output("rs");
    graph.set_declared_shape(NETWORK_INPUT, vec![3, 2]);

    let g = encode_graph(&graph, &free_box(6), &HashMap::new()).expect("encode_graph");
    assert_eq!(g.problem.num_cols(), 8, "input(6) + out(2)");
    assert_eq!(g.problem.num_rows(), 2);
    for c in 0..2usize {
        let row = &g.problem.rows()[c];
        let expect = vec![
            (g.input_vars[c].0, 1.0),
            (g.input_vars[c + 2].0, 1.0),
            (g.input_vars[c + 4].0, 1.0),
            (g.output_vars[c].0, -1.0),
        ];
        assert_eq!(row.coeffs, expect, "ReduceSum row {c}");
        assert_eq!((row.lb, row.ub), (0.0, 0.0), "equality with rhs 0");
    }

    // keepdims = true reduces to the SAME rows (shape metadata only).
    let mut gk = GraphNetwork::new();
    gk.add_node(GraphNode::new(
        "rs",
        Layer::ReduceSum(ReduceSumLayer::new(vec![1], true)),
        vec![NETWORK_INPUT.to_string()],
    ));
    gk.set_output("rs");
    gk.set_declared_shape(NETWORK_INPUT, vec![3, 2]);
    let gke = encode_graph(&gk, &free_box(6), &HashMap::new()).expect("encode_graph");
    assert_eq!(
        gke.problem.num_rows(),
        3,
        "3 kept rows for axis=1 reduction"
    );
    for r in 0..3usize {
        let row = &gke.problem.rows()[r];
        let expect = vec![
            (gke.input_vars[2 * r].0, 1.0),
            (gke.input_vars[2 * r + 1].0, 1.0),
            (gke.output_vars[r].0, -1.0),
        ];
        assert_eq!(row.coeffs, expect, "keepdims ReduceSum row {r}");
    }
}

/// AddConstant / SubConstant / MulConstant / DivConstant emit exact affine
/// rows, mirroring each layer's broadcast; DivConstant multiplies through by
/// the divisor (`x - c·y = 0`, no reciprocal rounding).
#[test]
fn graph_constant_op_rows_are_exact() {
    // AddConstant with trailing broadcast: input [2, 3], const [3].
    let c3 = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.25f32, -0.5, 1.5]).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "addc",
        Layer::AddConstant(AddConstantLayer::new(c3.clone())),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.set_output("addc");
    graph.set_declared_shape(NETWORK_INPUT, vec![2, 3]);
    let g = encode_graph(&graph, &free_box(6), &HashMap::new()).expect("encode_graph");
    assert_eq!(g.problem.num_rows(), 6);
    for i in 0..6usize {
        let row = &g.problem.rows()[i];
        let expect = vec![(g.input_vars[i].0, 1.0), (g.output_vars[i].0, -1.0)];
        assert_eq!(row.coeffs, expect, "AddConstant row {i}");
        let c = c3[[i % 3]] as f64;
        assert_eq!(
            (row.lb, row.ub),
            (-c, -c),
            "AddConstant rhs = -c (broadcast)"
        );
    }

    // Fail-closed: non-scalar constant with NO declared input shape.
    let mut bare = GraphNetwork::new();
    bare.add_node(GraphNode::new(
        "addc",
        Layer::AddConstant(AddConstantLayer::new(c3)),
        vec![NETWORK_INPUT.to_string()],
    ));
    bare.set_output("addc");
    assert!(
        encode_graph(&bare, &free_box(6), &HashMap::new()).is_err(),
        "broadcast without a declared shape must fail closed"
    );

    // Scalar Mul / Div / Sub(reverse) constants need no shape.
    let scalar = |v: f32| ArrayD::from_elem(IxDyn(&[1]), v);
    let mut g2 = GraphNetwork::new();
    g2.add_node(GraphNode::new(
        "mulc",
        Layer::MulConstant(MulConstantLayer::new(scalar(2.5))),
        vec![NETWORK_INPUT.to_string()],
    ));
    g2.add_node(GraphNode::new(
        "divc",
        Layer::DivConstant(DivConstantLayer::new(scalar(4.0))),
        vec!["mulc".to_string()],
    ));
    g2.add_node(GraphNode::new(
        "subc",
        Layer::SubConstant(SubConstantLayer::new_reverse(scalar(1.25))),
        vec!["divc".to_string()],
    ));
    g2.set_output("subc");
    let e2 = encode_graph(&g2, &free_box(2), &HashMap::new()).expect("encode_graph");
    assert_eq!(e2.problem.num_rows(), 6);
    let rows = e2.problem.rows();
    // mulc rows: 2.5·x - y = 0.
    for i in 0..2usize {
        assert_eq!(rows[i].coeffs[0].1, 2.5, "MulConstant coefficient");
        assert_eq!((rows[i].lb, rows[i].ub), (0.0, 0.0));
    }
    // divc rows: x - 4·y = 0 — the coefficient is the DIVISOR, not 1/4.
    for i in 2..4usize {
        assert_eq!(rows[i].coeffs[0].1, 1.0);
        assert_eq!(
            rows[i].coeffs[1].1, -4.0,
            "DivConstant multiplies through by c"
        );
        assert_eq!((rows[i].lb, rows[i].ub), (0.0, 0.0));
    }
    // subc (reverse) rows: x + y = 1.25.
    for i in 4..6usize {
        assert_eq!(rows[i].coeffs[0].1, 1.0);
        assert_eq!(rows[i].coeffs[1].1, 1.0, "reverse Sub adds the two");
        assert_eq!((rows[i].lb, rows[i].ub), (1.25, 1.25));
    }
}

/// Element-wise Sub emits exact `A - B - out = 0` rows.
#[test]
fn graph_sub_rows_are_exact() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "a",
        Layer::Slice(SliceLayer::new(0, 0, 1)),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "b",
        Layer::Slice(SliceLayer::new(0, 1, 2)),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::binary("sub", Layer::Sub(SubLayer), "a", "b"));
    graph.set_output("sub");
    graph.set_declared_shape(NETWORK_INPUT, vec![2]);
    graph.set_declared_shape("a", vec![1]);
    graph.set_declared_shape("b", vec![1]);

    let g = encode_graph(&graph, &free_box(2), &HashMap::new()).expect("encode_graph");
    assert_eq!(g.problem.num_rows(), 1);
    let row = &g.problem.rows()[0];
    assert_eq!(
        row.coeffs,
        vec![
            (g.input_vars[0].0, 1.0),
            (g.input_vars[1].0, -1.0),
            (g.output_vars[0].0, -1.0)
        ],
        "Sub row must be A - B - out = 0"
    );
    assert_eq!((row.lb, row.ub), (0.0, 0.0));
}

/// Build the mscn-style masked-branch micro-graph shared by the MulBinary /
/// Div tests: `_input` [2, 4] = hidden [2, 3] ++ mask [2, 1] (last column).
/// The mask columns' pinnedness is controlled by the input box.
fn masked_micro_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "h",
        Layer::Slice(SliceLayer::new(-1, 0, 3)),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "m",
        Layer::Slice(SliceLayer::new(-1, 3, 4)),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        "h",
        "m",
    ));
    graph.set_declared_shape(NETWORK_INPUT, vec![2, 4]);
    graph.set_declared_shape("h", vec![2, 3]);
    graph.set_declared_shape("m", vec![2, 1]);
    graph.set_declared_shape("mul", vec![2, 3]);
    graph
}

/// Input box for [`masked_micro_graph`]: hidden columns free, mask columns
/// (flat 3 and 7) fixed to `m0` / `m1`.
fn masked_box(m0: f32, m1: f32) -> Vec<Bound> {
    let mut bounds = free_box(8);
    bounds[3] = Bound::new(m0, m0);
    bounds[7] = Bound::new(m1, m1);
    bounds
}

/// MulBinary with a mask PINNED by the instance box encodes the exact affine
/// rows `m_r·h(r,c) - out(r,c) = 0` (broadcast [2,3] × [2,1]); an unpinned
/// mask fails closed.
#[test]
fn graph_mul_binary_pinned_mask_is_exact_affine() {
    let mut graph = masked_micro_graph();
    graph.set_output("mul");

    // m0 = 2.0 (live rows), m1 = 0.0 (zeroed rows — the y = 0 special case).
    let g = encode_graph(&graph, &masked_box(2.0, 0.0), &HashMap::new()).expect("encode_graph");
    assert_eq!(
        g.problem.num_rows(),
        6,
        "one row per broadcast output element"
    );
    for r in 0..2usize {
        for c in 0..3usize {
            let row = &g.problem.rows()[r * 3 + c];
            let y = g.output_vars[r * 3 + c].0;
            let h_col = g.input_vars[4 * r + c].0; // h aliases input cols (row r, col c)
            if r == 0 {
                assert_eq!(
                    row.coeffs,
                    vec![(h_col, 2.0), (y, -1.0)],
                    "pinned 2.0 factor"
                );
            } else {
                assert_eq!(row.coeffs, vec![(y, -1.0)], "pinned 0.0 factor => out = 0");
            }
            assert_eq!((row.lb, row.ub), (0.0, 0.0));
        }
    }

    // Fail-closed: mask free in the box => bilinear => refuse.
    let err = encode_graph(&graph, &free_box(8), &HashMap::new());
    assert!(err.is_err(), "unpinned MulBinary must fail closed");
    assert!(
        format!("{:#}", err.unwrap_err()).contains("pinned"),
        "error must say why (pinning)"
    );
}

/// Div by a ReduceSum-of-pinned-masks denominator encodes the exact rows
/// `a_i - β·out_i = 0` (β = the verified-exact mask count, NEVER 1/β); an
/// unpinned or zero denominator fails closed.
#[test]
fn graph_div_by_pinned_mask_count_is_exact_affine() {
    let build = |m0: f32, m1: f32| -> (GraphNetwork, Vec<Bound>) {
        let mut graph = masked_micro_graph();
        // num = column sums of mul => [3]; den = mask count => [1]; div = num/den.
        graph.add_node(GraphNode::new(
            "num",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-2], false)),
            vec!["mul".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "den",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-2], false)),
            vec!["m".to_string()],
        ));
        graph.add_node(GraphNode::binary("div", Layer::Div(DivLayer), "num", "den"));
        graph.set_output("div");
        graph.set_declared_shape("num", vec![3]);
        graph.set_declared_shape("den", vec![1]);
        graph.set_declared_shape("div", vec![3]);
        (graph, masked_box(m0, m1))
    };

    // β = 1.0 + 2.0 = 3.0 (exact f64 sum of pinned masks).
    let (graph, bounds) = build(1.0, 2.0);
    let g = encode_graph(&graph, &bounds, &HashMap::new()).expect("encode_graph");
    // Rows: mul(6) + num(3) + den(1) + div(3) = 13 (order-robust: match by
    // signature rather than position).
    assert_eq!(g.problem.num_rows(), 13);
    let den_col = g.node_cols["den"][0];
    let num_cols = &g.node_cols["num"];
    let has_row = |expect: &Vec<(usize, f64)>| {
        g.problem
            .rows()
            .iter()
            .any(|r| r.coeffs == *expect && (r.lb, r.ub) == (0.0, 0.0))
    };
    for i in 0..3usize {
        let expect = vec![(num_cols[i].0, 1.0), (g.output_vars[i].0, -3.0)];
        assert!(
            has_row(&expect),
            "Div row {i} must be a - β·y = 0 with β = 3 (NOT 1/3): {expect:?}"
        );
    }
    // The den row itself is still the plain ReduceSum equality.
    let den_expect = vec![
        (g.input_vars[3].0, 1.0),
        (g.input_vars[7].0, 1.0),
        (den_col.0, -1.0),
    ];
    assert!(has_row(&den_expect), "den = mask-count ReduceSum row");

    // Fail-closed: β = 0 (all masks zero).
    let (graph0, bounds0) = build(0.0, 0.0);
    let err0 = encode_graph(&graph0, &bounds0, &HashMap::new());
    assert!(err0.is_err(), "zero pinned denominator must fail closed");

    // Fail-closed: mask not pinned (free box).
    let (graph_f, _) = build(1.0, 1.0);
    let err_f = encode_graph(&graph_f, &free_box(8), &HashMap::new());
    assert!(err_f.is_err(), "unpinned denominator must fail closed");
}

/// The final-Sigmoid peel: `encode_graph_peel_final_sigmoid` encodes up to the
/// logit and reports `peeled = true`; plain `encode_graph` refuses Sigmoid;
/// a NON-final Sigmoid refuses even with peeling requested.
#[test]
fn graph_final_sigmoid_peels_to_logit() {
    let w = Array2::from_shape_vec((1, 2), vec![0.5f32, -0.25]).unwrap();
    let b = Array1::from_vec(vec![0.1f32]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "lin",
        Layer::Linear(LinearLayer::new(w.clone(), Some(b.clone())).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["lin".to_string()],
    ));
    graph.set_output("sig");

    let (g, peeled) = encode_graph_peel_final_sigmoid(&graph, &free_box(2), &HashMap::new())
        .expect("peel encode");
    assert!(peeled, "final Sigmoid must be peeled");
    assert_eq!(
        g.problem.num_cols(),
        3,
        "input(2) + logit(1); NO sigmoid columns"
    );
    assert_eq!(g.problem.num_rows(), 1, "only the Linear equality row");
    assert_eq!(
        g.output_vars, g.node_cols["lin"],
        "output = the LOGIT columns"
    );
    assert!(
        !g.node_cols.contains_key("sig"),
        "peeled node has no columns"
    );

    // Plain encode_graph stays sigmoid-free (fail closed).
    let plain = encode_graph(&graph, &free_box(2), &HashMap::new());
    assert!(plain.is_err());
    assert!(format!("{:#}", plain.unwrap_err()).contains("Sigmoid"));

    // Mid-graph Sigmoid fails closed even when peeling is requested.
    let mut mid = GraphNetwork::new();
    mid.add_node(GraphNode::new(
        "lin",
        Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    mid.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["lin".to_string()],
    ));
    let w2 = Array2::from_shape_vec((1, 1), vec![2.0f32]).unwrap();
    mid.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2, None).unwrap()),
        vec!["sig".to_string()],
    ));
    mid.set_output("lin2");
    let err = encode_graph_peel_final_sigmoid(&mid, &free_box(2), &HashMap::new());
    assert!(err.is_err(), "non-final Sigmoid must fail closed");
    assert!(format!("{:#}", err.unwrap_err()).contains("Sigmoid"));
}

/// The σ-space → logit-space threshold transform: outward (never-stricter)
/// rounding, tight enclosures, and exact edge handling.
#[test]
fn logit_threshold_transform_is_outward_and_tight() {
    let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());

    // Edges (exact real-arithmetic semantics; σ(z) ∈ (0,1) for finite z).
    assert_eq!(
        logit_upper_threshold(0.0).unwrap(),
        LogitThreshold::Infeasible
    );
    assert_eq!(
        logit_upper_threshold(-0.5).unwrap(),
        LogitThreshold::Infeasible
    );
    assert_eq!(logit_upper_threshold(1.0).unwrap(), LogitThreshold::Vacuous);
    assert_eq!(logit_upper_threshold(1.5).unwrap(), LogitThreshold::Vacuous);
    assert_eq!(logit_lower_threshold(0.0).unwrap(), LogitThreshold::Vacuous);
    assert_eq!(
        logit_lower_threshold(-0.5).unwrap(),
        LogitThreshold::Vacuous
    );
    assert_eq!(
        logit_lower_threshold(1.0).unwrap(),
        LogitThreshold::Infeasible
    );
    assert!(logit_upper_threshold(f64::NAN).is_err(), "NaN fails closed");
    assert!(logit_lower_threshold(f64::NAN).is_err(), "NaN fails closed");

    // Sweep incl. extremes, the exact 0.5 pivot, and the REAL mscn thresholds.
    let mut ts: Vec<f64> = vec![
        1e-12,
        1e-6,
        1e-3,
        0.1,
        0.25,
        0.5,
        0.4999999999,
        0.5000000001,
        0.75,
        0.9,
        0.999,
        0.999999,
        1.0 - 1e-12,
        0.7206605449954523,
        0.765354006704809,
    ];
    for k in 1..1000 {
        ts.push(k as f64 / 1000.0);
    }
    for &t in &ts {
        let LogitThreshold::Bound(hi) = logit_upper_threshold(t).unwrap() else {
            panic!("t = {t} must transform to a bound");
        };
        let LogitThreshold::Bound(lo) = logit_lower_threshold(t).unwrap() else {
            panic!("t = {t} must transform to a bound");
        };
        // (1) The naive logit lies INSIDE the enclosure (outwardness by
        //     construction) and the enclosure is properly ordered.
        let naive = (t / (1.0 - t)).ln();
        assert!(
            lo <= naive && naive <= hi,
            "t = {t}: enclosure must contain ln(t/(1-t))"
        );
        // (2) Never-stricter, checked through σ with a tiny eval-error margin:
        //     σ(hi) must not be BELOW t, σ(lo) must not be ABOVE t.
        assert!(
            sigmoid(hi) >= t * (1.0 - 1e-13) - 1e-300,
            "t = {t}: σ(upper bound) must reach t (never stricter)"
        );
        assert!(
            sigmoid(lo) <= t * (1.0 + 1e-13) + 1e-300,
            "t = {t}: σ(lower bound) must not exceed t (never stricter)"
        );
        // (3) Tightness: the outward rounding is ULP-scale, not a real loss.
        let width = hi - lo;
        assert!(
            width <= 1e-11 * hi.abs().max(lo.abs()).max(1.0),
            "t = {t}: enclosure width {width} must be ULP-scale"
        );
    }
}

// ── nn4sys mscn: point-eval parity (the critical oracle) ────────────────────

/// Parse a nn4sys cardinality vnnlib: the (shared) input box from the
/// `(>= X_i v)` / `(<= X_i v)` atoms and the two Y_0 thresholds from
/// `(<= Y_0 t)` (upper) / `(>= Y_0 t')` (lower). The instance encodes the
/// UNSAFE set as a disjunction of two conjuncts with IDENTICAL X boxes, so
/// overwriting per-variable bounds across conjuncts is exact.
fn parse_cardinality_vnnlib(
    path: &Path,
    n_inputs: usize,
) -> (Vec<(f64, f64)>, Option<f64>, Option<f64>) {
    let text = std::fs::read_to_string(path).expect("read vnnlib");
    let mut lo = vec![f64::NEG_INFINITY; n_inputs];
    let mut hi = vec![f64::INFINITY; n_inputs];
    let (mut y_upper, mut y_lower) = (None, None);
    for frag in text.split('(') {
        let frag = frag.trim();
        let (ge, rest) = if let Some(r) = frag.strip_prefix(">=") {
            (true, r)
        } else if let Some(r) = frag.strip_prefix("<=") {
            (false, r)
        } else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let (Some(var), Some(raw_val)) = (it.next(), it.next()) else {
            continue;
        };
        let Ok(val) = raw_val.trim_end_matches(')').parse::<f64>() else {
            continue;
        };
        if let Some(idx_s) = var.strip_prefix("X_") {
            let idx: usize = idx_s.parse().expect("X index");
            assert!(idx < n_inputs, "X_{idx} out of range");
            if ge {
                lo[idx] = val;
            } else {
                hi[idx] = val;
            }
        } else if var == "Y_0" {
            if ge {
                y_lower = Some(val);
            } else {
                y_upper = Some(val);
            }
        }
    }
    let bounds: Vec<(f64, f64)> = lo.into_iter().zip(hi).collect();
    assert!(
        bounds
            .iter()
            .all(|&(l, h)| l.is_finite() && h.is_finite() && l <= h),
        "input box must be a finite box"
    );
    (bounds, y_upper, y_lower)
}

/// The declared shape of `name`, with the (shapeless) `_input` sentinel mapped
/// to the model's known input shape.
fn walk_shape(graph: &GraphNetwork, name: &str, input_shape: &[usize]) -> Vec<usize> {
    if name == NETWORK_INPUT {
        input_shape.to_vec()
    } else {
        graph
            .declared_shape(name)
            .unwrap_or_else(|| panic!("walk: no declared shape for '{name}'"))
            .to_vec()
    }
}

fn walk_unflatten(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut multi = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        multi[d] = flat % shape[d];
        flat /= shape[d];
    }
    multi
}

/// NumPy trailing-axis broadcast source index (test-side reimplementation —
/// written from the broadcasting SPEC, not shared with the encoder).
fn walk_broadcast_idx(out_multi: &[usize], shape: &[usize]) -> usize {
    let off = out_multi.len() - shape.len();
    let mut flat = 0usize;
    for (d, &s) in shape.iter().enumerate() {
        flat = flat * s + if s == 1 { 0 } else { out_multi[d + off] };
    }
    flat
}

/// Test-side f64 forward walk of the mscn graph, written from each layer's
/// documented forward semantics (slice index math, last-axis Linear
/// contraction, trailing broadcast, reduction, division, 1-D concat, σ).
/// Independent of the encoder; validated against ny's own forward in the
/// parity test, then used to certify the encoder's rows via feasibility.
fn mscn_f64_walk(
    graph: &GraphNetwork,
    x: &[f64],
    input_shape: &[usize],
) -> HashMap<String, Vec<f64>> {
    let mut vals: HashMap<String, Vec<f64>> = HashMap::new();
    vals.insert(NETWORK_INPUT.to_string(), x.to_vec());
    let exec: Vec<String> = graph.exec_order().expect("exec order").to_vec();
    for name in &exec {
        let node = graph.node(name).unwrap();
        let ins = node.inputs();
        let out: Vec<f64> = match node.layer() {
            Layer::Slice(s) => {
                let in_shape = walk_shape(graph, &ins[0], input_shape);
                let v = &vals[&ins[0]];
                assert_eq!(v.len(), in_shape.iter().product::<usize>());
                let out_shape = s.compute_output_shape(&in_shape).expect("slice shape");
                let ndim = in_shape.len();
                let a = s.axis as i64;
                let axis = (if a < 0 { a + ndim as i64 } else { a }) as usize;
                let start = s.start.min(in_shape[axis]);
                let out_size: usize = out_shape.iter().product();
                (0..out_size)
                    .map(|of| {
                        let mut multi = walk_unflatten(of, &out_shape);
                        multi[axis] += start;
                        let mut flat = 0usize;
                        for d in 0..ndim {
                            flat = flat * in_shape[d] + multi[d];
                        }
                        v[flat]
                    })
                    .collect()
            }
            Layer::Linear(l) => {
                let v = &vals[&ins[0]];
                let (n_out, n_in) = (l.out_features(), l.in_features());
                assert_eq!(v.len() % n_in, 0);
                let rows = v.len() / n_in;
                let mut out = Vec::with_capacity(rows * n_out);
                for r in 0..rows {
                    for i in 0..n_out {
                        let mut s = l.bias.as_ref().map(|b| b[i] as f64).unwrap_or(0.0);
                        for j in 0..n_in {
                            s += (l.weight[[i, j]] as f64) * v[r * n_in + j];
                        }
                        out.push(s);
                    }
                }
                out
            }
            Layer::AddConstant(a) => {
                let v = &vals[&ins[0]];
                let c: Vec<f64> = a.constant().iter().map(|&w| w as f64).collect();
                assert_eq!(v.len() % c.len(), 0, "mscn bias is a trailing broadcast");
                v.iter()
                    .enumerate()
                    .map(|(i, &vi)| vi + c[i % c.len()])
                    .collect()
            }
            Layer::ReLU(_) => vals[&ins[0]].iter().map(|&v| v.max(0.0)).collect(),
            Layer::MulBinary(_) | Layer::Div(_) => {
                let a_shape = walk_shape(graph, &ins[0], input_shape);
                let b_shape = walk_shape(graph, &ins[1], input_shape);
                let out_shape = broadcast_shapes(&a_shape, &b_shape).expect("broadcast");
                let (a, b) = (&vals[&ins[0]], &vals[&ins[1]]);
                let is_mul = matches!(node.layer(), Layer::MulBinary(_));
                (0..out_shape.iter().product::<usize>())
                    .map(|of| {
                        let multi = walk_unflatten(of, &out_shape);
                        let av = a[walk_broadcast_idx(&multi, &a_shape)];
                        let bv = b[walk_broadcast_idx(&multi, &b_shape)];
                        if is_mul {
                            av * bv
                        } else {
                            av / bv
                        }
                    })
                    .collect()
            }
            Layer::ReduceSum(rs) => {
                let in_shape = walk_shape(graph, &ins[0], input_shape);
                let v = &vals[&ins[0]];
                let ndim = in_shape.len();
                let mut reduced = vec![rs.axes.is_empty(); ndim];
                for &a in &rs.axes {
                    reduced[(if a < 0 { a + ndim as i64 } else { a }) as usize] = true;
                }
                let out_size: usize = in_shape
                    .iter()
                    .enumerate()
                    .filter(|(d, _)| !reduced[*d])
                    .map(|(_, &s)| s)
                    .product();
                let mut out = vec![0.0f64; out_size.max(1)];
                for (flat, &vi) in v.iter().enumerate() {
                    let multi = walk_unflatten(flat, &in_shape);
                    let mut of = 0usize;
                    for d in 0..ndim {
                        if !reduced[d] {
                            of = of * in_shape[d] + multi[d];
                        }
                    }
                    out[of] += vi;
                }
                out
            }
            Layer::Concat(_) => {
                for inp in ins {
                    assert_eq!(
                        walk_shape(graph, inp, input_shape).len(),
                        1,
                        "walk supports the mscn 1-D concat"
                    );
                }
                ins.iter().flat_map(|inp| vals[inp].clone()).collect()
            }
            Layer::Sigmoid(_) => vals[&ins[0]]
                .iter()
                .map(|&z| 1.0 / (1.0 + (-z).exp()))
                .collect(),
            other => panic!("walk: unexpected mscn layer {}", other.layer_type()),
        };
        vals.insert(name.clone(), out);
    }
    vals
}

/// ny's own forward at a point: exact point IBP (degenerate box), midpoint of
/// the (tiny) output interval — the same oracle `revalidate_monotonic_witness`
/// trusts in production.
fn ny_point_forward(graph: &GraphNetwork, x: &[f64], shape: &[usize]) -> Vec<f64> {
    let degenerate: Vec<Bound> = x.iter().map(|&v| Bound::new(v as f32, v as f32)).collect();
    let t = Verifier::bounds_to_tensor(&degenerate, Some(shape)).expect("tensor");
    let out = graph.propagate_ibp(&t).expect("point IBP");
    out.lower()
        .iter()
        .zip(out.upper().iter())
        .map(|(&l, &u)| f64::midpoint(l as f64, u as f64))
        .collect()
}

/// Build a full MIP column assignment from the f64 walk values: every node's
/// columns take its walk values; ReLU indicator columns are re-derived by
/// replaying the encoder's stability triage in exec order (binaries are
/// created in that order) with `z = [y > 0]`.
fn build_assignment(
    enc: &GraphMipEncoding,
    graph: &GraphNetwork,
    vals: &HashMap<String, Vec<f64>>,
    node_bounds: &HashMap<String, Vec<Bound>>,
) -> Vec<f64> {
    let mut assign = vec![f64::NAN; enc.problem.num_cols()];
    for (name, cols) in &enc.node_cols {
        let v = &vals[name];
        assert_eq!(v.len(), cols.len(), "node '{name}' column count");
        for (val, col) in v.iter().zip(cols) {
            assign[col.0] = *val;
        }
    }
    let mut bins = enc.binary_vars.iter();
    for name in graph.exec_order().unwrap() {
        let node = graph.node(name).unwrap();
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        let pre = node_bounds
            .get(name)
            .or_else(|| node_bounds.get(&node.inputs()[0]))
            .expect("relu pre-activation box");
        let outs = &vals[name];
        for (i, b) in pre.iter().enumerate() {
            let lb = b.lower() as f64 - DELTA;
            let ub = b.upper() as f64 + DELTA;
            if lb >= 0.0 || ub <= 0.0 {
                continue; // stable neuron: no binary
            }
            let z = bins.next().expect("binary under-run: triage misaligned");
            assign[z.0] = if outs[i] > 0.0 { 1.0 } else { 0.0 };
        }
    }
    assert!(bins.next().is_none(), "binary over-run: triage misaligned");
    for (i, v) in assign.iter().enumerate() {
        assert!(!v.is_nan(), "column {i} unassigned");
    }
    assign
}

/// Per-node sound IBP boxes over the instance box, via ny's own IBP on
/// output-truncated copies of the graph (best-effort: nodes whose truncated
/// propagation fails are skipped — only ReLU pre-activations strictly need a
/// box). ReLU nodes themselves get NO entry: the encoder's contract keys a
/// ReLU-named entry as that ReLU's PRE-activation box (legacy/explicit), while
/// this helper computes OUTPUT boxes — so ReLU entries are omitted and the
/// encoder falls back to the input node's (true pre-activation) box.
fn ibp_node_boxes(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    input_shape: &[usize],
) -> HashMap<String, Vec<Bound>> {
    let tensor = Verifier::bounds_to_tensor(input_bounds, Some(input_shape)).expect("box tensor");
    let exec: Vec<String> = graph.exec_order().expect("exec order").to_vec();
    let mut node_bounds = HashMap::new();
    for name in &exec {
        if matches!(graph.node(name).unwrap().layer(), Layer::ReLU(_)) {
            continue; // see doc comment: OUTPUT box ≠ the encoder's pre-activation key
        }
        let mut sub = graph.clone();
        sub.set_output(name.clone());
        let Ok(out) = sub.propagate_ibp(&tensor) else {
            continue;
        };
        let bounds: Vec<Bound> = out
            .lower()
            .iter()
            .zip(out.upper().iter())
            .map(|(&l, &u)| Bound::new(l, u))
            .collect();
        node_bounds.insert(name.clone(), bounds);
    }
    node_bounds
}

/// Human-readable first bound/row violation of an assignment, for test
/// diagnostics (mirrors [`point_is_feasible`]'s checks).
fn first_violation(problem: &MilpProblem, assign: &[f64], tol: f64) -> Option<String> {
    for (i, spec) in problem.cols().iter().enumerate() {
        if assign[i] < spec.lb - tol || assign[i] > spec.ub + tol {
            return Some(format!(
                "col {i}: value {} outside [{}, {}]",
                assign[i], spec.lb, spec.ub
            ));
        }
    }
    for (r, row) in problem.rows().iter().enumerate() {
        let s: f64 = row.coeffs.iter().map(|&(col, w)| w * assign[col]).sum();
        if s < row.lb - tol || s > row.ub + tol {
            return Some(format!(
                "row {r}: activity {s} outside [{}, {}] (coeffs {:?})",
                row.lb,
                row.ub,
                &row.coeffs[..row.coeffs.len().min(8)]
            ));
        }
    }
    None
}

/// THE CRITICAL ORACLE (an all-UNSAT benchmark hides a broken encoder, so we
/// prove the encoding faithful at concrete points): load the REAL mscn_128d
/// model and a REAL instance box, then for >= 100 sampled points assert
///   (1) the test-side f64 affine walk matches ny's own forward logit within
///       DELTA (validates the walk semantics against ny's ground truth);
///   (2) the true forward trajectory is FEASIBLE in the encoded MIP and a
///       perturbed logit is NOT (validates every emitted row against the
///       walk — false-UNSAT risk is a trajectory the MIP excludes);
///   (3) + (4) the σ-space property and the logit-space transformed property
///       classify the point identically (validates the sigmoid peel rewrite).
#[test]
fn mscn_point_eval_parity_128d() {
    let Some(onnx) = nn4sys_onnx_path("mscn_128d.onnx") else {
        eprintln!("mscn_point_eval_parity_128d: model not found; skipping");
        return;
    };
    let vnnlib = onnx
        .parent()
        .and_then(Path::parent)
        .map(|d| d.join("vnnlib/cardinality_0_1_128.vnnlib"))
        .expect("vnnlib path");
    if !vnnlib.exists() {
        eprintln!("mscn_point_eval_parity_128d: vnnlib not found; skipping");
        return;
    }
    let graph = crate::commands::vnncomp::load_graph_network(&onnx).expect("load mscn_128d");
    let input_shape = [11usize, 14];
    let (raw_box, y_upper_t, y_lower_t) = parse_cardinality_vnnlib(&vnnlib, 11 * 14);
    let (y_upper_t, y_lower_t) = (
        y_upper_t.expect("(<= Y_0 t)"),
        y_lower_t.expect("(>= Y_0 t)"),
    );
    assert!(
        y_upper_t < y_lower_t,
        "cardinality band: UNSAT iff Y_0 stays strictly inside ({y_upper_t}, {y_lower_t})"
    );
    let input_bounds: Vec<Bound> = raw_box
        .iter()
        .map(|&(l, u)| Bound::new(l as f32, u as f32))
        .collect();

    let node_bounds = ibp_node_boxes(&graph, &input_bounds, &input_shape);

    // The headline: the real mscn model must ENCODE (peeling its Sigmoid).
    let (enc, peeled) = encode_graph_peel_final_sigmoid(&graph, &input_bounds, &node_bounds)
        .expect("mscn_128d must encode exactly");
    assert!(peeled, "final Sigmoid must be peeled");
    assert_eq!(enc.output_vars.len(), 1, "single logit output");
    assert!(enc.problem.num_rows() > 0);

    let LogitThreshold::Bound(z_upper) = logit_upper_threshold(y_upper_t).unwrap() else {
        panic!("upper threshold must transform to a bound");
    };
    let LogitThreshold::Bound(z_lower) = logit_lower_threshold(y_lower_t).unwrap() else {
        panic!("lower threshold must transform to a bound");
    };

    let logit_name = graph.node(graph.output_name()).unwrap().inputs()[0].clone();
    let mut logit_graph = graph.clone();
    logit_graph.set_output(logit_name.clone());

    // Deterministic LCG sampling across the box (only free dims vary; this
    // instance pins everything except one selectivity input).
    let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_unit = move || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let n_samples = 120;
    for k in 0..n_samples {
        let x: Vec<f64> = input_bounds
            .iter()
            .map(|b| {
                let (l, u) = (b.lower() as f64, b.upper() as f64);
                if l == u {
                    l
                } else {
                    // Include both box endpoints, then random interior points.
                    let t = match k {
                        0 => 0.0,
                        1 => 1.0,
                        _ => next_unit(),
                    };
                    l + t * (u - l)
                }
            })
            .collect();

        // (1) Test-side f64 walk vs ny's own forward: logit within DELTA.
        let vals = mscn_f64_walk(&graph, &x, &input_shape);
        let z_walk = vals[&logit_name][0];
        let z_ny = ny_point_forward(&logit_graph, &x, &input_shape)[0];
        assert!(
            (z_walk - z_ny).abs() <= DELTA,
            "sample {k}: walk logit {z_walk} vs ny logit {z_ny} differ beyond DELTA"
        );

        // (2) True trajectory feasible; perturbed logit rejected.
        let assign = build_assignment(&enc, &graph, &vals, &node_bounds);
        assert!(
            point_is_feasible(&enc.problem, &assign, 1e-6),
            "sample {k}: the true forward trajectory must be MIP-feasible; first violation: {:?}",
            first_violation(&enc.problem, &assign, 1e-6)
        );
        let mut bad = assign.clone();
        bad[enc.output_vars[0].0] += 0.05;
        assert!(
            !point_is_feasible(&enc.problem, &bad, 1e-6),
            "sample {k}: a perturbed logit must be MIP-infeasible"
        );

        // (3) Sharp transform parity on the SAME value: σ(z) vs t agrees with
        // z vs logit(t) (only ULP-scale outwardness in between).
        let y_walk = 1.0 / (1.0 + (-z_walk).exp());
        for (t, z_t, upper) in [(y_upper_t, z_upper, true), (y_lower_t, z_lower, false)] {
            if (y_walk - t).abs() < 1e-12 {
                continue;
            }
            let sig_sat = if upper { y_walk <= t } else { y_walk >= t };
            let z_sat = if upper { z_walk <= z_t } else { z_walk >= z_t };
            assert_eq!(
                sig_sat, z_sat,
                "sample {k}: σ-space vs transformed logit-space classification (t = {t})"
            );
        }

        // (4) ny's f32 sigmoid output classifies like the transformed logit
        // property away from the float-gap guard band (|y - t| > 1e-3).
        let y_ny = ny_point_forward(&graph, &x, &input_shape)[0];
        for (t, z_t, upper) in [(y_upper_t, z_upper, true), (y_lower_t, z_lower, false)] {
            if (y_ny - t).abs() < 1e-3 {
                continue;
            }
            let sig_sat = if upper { y_ny <= t } else { y_ny >= t };
            let z_sat = if upper { z_walk <= z_t } else { z_walk >= z_t };
            assert_eq!(
                sig_sat, z_sat,
                "sample {k}: ny σ output vs transformed property (t = {t})"
            );
        }
    }
}

/// Increment 5c — THE FOLD PARITY ORACLE on the REAL mscn_128d model (the
/// direct anti-false-UNSAT check for the pinned-column fold): fold the real
/// clause encoding, then for sampled points of the instance box assert
///   (1) every column the fold PINNED agrees with the true forward
///       trajectory's value at that column (a wrong pin would be a fold that
///       excludes the true trajectory — the false-UNSAT mechanism);
///   (2) the true trajectory RESTRICTED to the surviving columns is FEASIBLE
///       in the folded MIP (tol 1e-6);
///   (3) a perturbed logit is INFEASIBLE in the folded MIP (the fold must
///       not have weakened the system into vacuity).
#[test]
fn mscn_fold_parity_128d() {
    use super::super::graph_mip_fold::{fold_pinned_columns, FoldOutcome};

    let Some(onnx) = nn4sys_onnx_path("mscn_128d.onnx") else {
        eprintln!("mscn_fold_parity_128d: model not found; skipping");
        return;
    };
    let vnnlib = onnx
        .parent()
        .and_then(Path::parent)
        .map(|d| d.join("vnnlib/cardinality_0_1_128.vnnlib"))
        .expect("vnnlib path");
    if !vnnlib.exists() {
        eprintln!("mscn_fold_parity_128d: vnnlib not found; skipping");
        return;
    }
    let graph = crate::commands::vnncomp::load_graph_network(&onnx).expect("load mscn_128d");
    let input_shape = [11usize, 14];
    let (raw_box, _, _) = parse_cardinality_vnnlib(&vnnlib, 11 * 14);
    let input_bounds: Vec<Bound> = raw_box
        .iter()
        .map(|&(l, u)| Bound::new(l as f32, u as f32))
        .collect();
    let node_bounds = ibp_node_boxes(&graph, &input_bounds, &input_shape);
    let (enc, peeled) = encode_graph_peel_final_sigmoid(&graph, &input_bounds, &node_bounds)
        .expect("mscn_128d must encode exactly");
    assert!(peeled);

    let folded = match fold_pinned_columns(&enc).expect("fold must not error") {
        FoldOutcome::Folded(folded) => *folded,
        FoldOutcome::ProvedInfeasible { row } => panic!(
            "the real mscn instance must not fold-infeasible (the true trajectory is feasible); \
             violated row {row}"
        ),
    };
    let s = folded.stats;
    eprintln!(
        "mscn_fold_parity_128d: fold {}→{} cols, {}→{} rows, {}→{} binaries ({} bound-pinned + \
         {} derived-pinned, {} constant rows dropped, {} rows kept unfolded)",
        s.cols_before,
        s.cols_after,
        s.rows_before,
        s.rows_after,
        s.binaries_before,
        s.binaries_after,
        s.pinned_from_bounds,
        s.pinned_derived,
        s.rows_dropped_constant,
        s.rows_kept_unfolded,
    );
    // The headline shrink: the fold must remove a large majority of columns.
    assert!(
        s.cols_after * 2 < s.cols_before,
        "fold barely shrank the problem ({} -> {} cols)",
        s.cols_before,
        s.cols_after
    );
    // The (never-folded) logit output column must survive the remap.
    let folded_logit = folded.col_map[enc.output_vars[0].0].expect("output col survives");
    assert_eq!(folded.encoding.output_vars, vec![folded_logit]);

    // Deterministic LCG sampling (same generator as the unfolded parity test).
    let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_unit = move || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let n_samples = 40;
    for k in 0..n_samples {
        let x: Vec<f64> = input_bounds
            .iter()
            .map(|b| {
                let (l, u) = (b.lower() as f64, b.upper() as f64);
                if l == u {
                    l
                } else {
                    let t = match k {
                        0 => 0.0,
                        1 => 1.0,
                        _ => next_unit(),
                    };
                    l + t * (u - l)
                }
            })
            .collect();
        let vals = mscn_f64_walk(&graph, &x, &input_shape);
        let assign = build_assignment(&enc, &graph, &vals, &node_bounds);

        // (1) Every pin agrees with the true trajectory (tol: the pins are
        // EXACT real-arithmetic values, the walk is f64-rounded).
        for (old, pin) in folded.pins.iter().enumerate() {
            if let Some(v) = pin {
                assert!(
                    (assign[old] - v).abs() <= 1e-6,
                    "sample {k}: fold pinned col {old} to {v} but the true trajectory has {}",
                    assign[old]
                );
            }
        }

        // (2) The trajectory restricted to surviving columns is feasible.
        let mut fassign = vec![f64::NAN; folded.encoding.problem.num_cols()];
        for (old, mapped) in folded.col_map.iter().enumerate() {
            if let Some(nc) = mapped {
                fassign[nc.0] = assign[old];
            }
        }
        assert!(
            fassign.iter().all(|v| !v.is_nan()),
            "dense remap must cover"
        );
        assert!(
            point_is_feasible(&folded.encoding.problem, &fassign, 1e-6),
            "sample {k}: the true trajectory must stay feasible in the FOLDED MIP; first \
             violation: {:?}",
            first_violation(&folded.encoding.problem, &fassign, 1e-6)
        );

        // (3) A perturbed logit is infeasible in the folded MIP.
        let mut bad = fassign.clone();
        bad[folded_logit.0] += 0.05;
        assert!(
            !point_is_feasible(&folded.encoding.problem, &bad, 1e-6),
            "sample {k}: a perturbed logit must be infeasible in the FOLDED MIP"
        );
    }
}

// ── increment 5d: CROWN-IBP-tightened boxes for the escalation encoder ──────

/// inc5d fail-closed contract: with an ALREADY-EXPIRED deadline the CROWN-IBP
/// collector tightens nothing and every node must come back with its plain IBP
/// bound — the helper never loses a bound and never returns anything looser or
/// tighter than the IBP map it was given.
#[test]
fn crown_tightened_node_bounds_expired_deadline_is_ibp_identity() {
    let w = Array2::from_shape_vec((2, 2), vec![1.0f32, -0.5, 0.25, 1.5]).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "lin",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["lin".to_string()],
    ));
    graph.set_output("relu");
    let input_bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-2.0, 2.0)];
    let tensor = Verifier::bounds_to_tensor(&input_bounds, Some(&[2])).expect("tensor");
    let ibp = graph
        .collect_node_bounds_with_engine_and_deadline(&tensor, None, None)
        .expect("ibp");

    // Deadline already reached when the collector starts ⇒ zero tightening.
    let (out, crown_nodes) =
        crown_tightened_node_bounds(&graph, &tensor, ibp.clone(), Instant::now());
    assert_eq!(
        crown_nodes,
        Some(0),
        "no node may be CROWN-tightened past the deadline"
    );
    assert_eq!(out.len(), ibp.len(), "no bound may be lost or invented");
    for (name, bt) in &ibp {
        let got = out
            .get(name)
            .unwrap_or_else(|| panic!("node '{name}' lost its bound"));
        assert_eq!(got.lower(), bt.lower(), "node '{name}' lower changed");
        assert_eq!(got.upper(), bt.upper(), "node '{name}' upper changed");
    }
}

/// inc5d — THE CRITICAL ORACLE with CROWN-tightened boxes (the direct
/// anti-false-UNSAT check): on the REAL mscn_128d model + REAL instance box,
/// encode with the CROWN-IBP boxes the escalation now hands the encoder and
/// assert for sampled points that
///   (1) the true forward trajectory stays FEASIBLE (tol 1e-6) — a tightened
///       box that excludes the true trajectory IS the false-UNSAT mechanism,
///       so any violation here means the bounds or the ReLU-name convention
///       are WRONG;
///   (2) a perturbed logit stays INFEASIBLE (tightening must not have
///       weakened the system into vacuity);
///   (3) the encoder-visible CROWN boxes are TIGHTER in aggregate than the
///       plain-IBP boxes from the same collection seed (the entire point of
///       inc5d: smaller big-M ⇒ root-LP-certifiable clauses), and every
///       ReLU pre-activation key the IBP path supplied survives.
#[test]
fn mscn_crown_bounds_parity_128d() {
    let Some(onnx) = nn4sys_onnx_path("mscn_128d.onnx") else {
        eprintln!("mscn_crown_bounds_parity_128d: model not found; skipping");
        return;
    };
    let vnnlib = onnx
        .parent()
        .and_then(Path::parent)
        .map(|d| d.join("vnnlib/cardinality_0_1_128.vnnlib"))
        .expect("vnnlib path");
    if !vnnlib.exists() {
        eprintln!("mscn_crown_bounds_parity_128d: vnnlib not found; skipping");
        return;
    }
    let graph = crate::commands::vnncomp::load_graph_network(&onnx).expect("load mscn_128d");
    let input_shape = [11usize, 14];
    let (raw_box, _, _) = parse_cardinality_vnnlib(&vnnlib, 11 * 14);
    let input_bounds: Vec<Bound> = raw_box
        .iter()
        .map(|&(l, u)| Bound::new(l as f32, u as f32))
        .collect();
    let tensor = Verifier::bounds_to_tensor(&input_bounds, Some(&input_shape)).expect("tensor");

    // The escalation's exact bound pipeline: plain IBP map, then the CROWN-IBP
    // tightening helper (generous deadline — this is the quality probe).
    let ibp_bt = graph
        .collect_node_bounds_with_engine_and_deadline(&tensor, None, None)
        .expect("ibp map");
    let deadline = Instant::now() + Duration::from_mins(1);
    let (crown_bt, crown_nodes) =
        crown_tightened_node_bounds(&graph, &tensor, ibp_bt.clone(), deadline);
    let crown_nodes = crown_nodes.expect("CROWN-IBP collection must succeed on mscn_128d");

    let ibp_boxes = ibp_boxes_for_encoder(&graph, &ibp_bt);
    let crown_boxes = ibp_boxes_for_encoder(&graph, &crown_bt);

    // (3a) Key coverage: every encoder-visible IBP key survives (in particular
    // every ReLU's INPUT node — the pre-activation source for big-M).
    for name in ibp_boxes.keys() {
        assert!(
            crown_boxes.contains_key(name),
            "node '{name}' lost its encoder box in the CROWN path"
        );
    }

    // (3b) Aggregate tightness over finite widths (CROWN∩IBP ⊆ IBP per node).
    let width_sum = |m: &HashMap<String, Vec<Bound>>| -> f64 {
        m.values()
            .flat_map(|v| v.iter())
            .map(|b| {
                let w = (b.upper() as f64) - (b.lower() as f64);
                if w.is_finite() {
                    w
                } else {
                    0.0
                }
            })
            .sum()
    };
    let (ibp_w, crown_w) = (width_sum(&ibp_boxes), width_sum(&crown_boxes));
    eprintln!(
        "mscn_crown_bounds_parity_128d: {crown_nodes} nodes CROWN-tightened; encoder box width \
         sum {ibp_w:.4} (ibp) -> {crown_w:.4} (crown-ibp)"
    );
    assert!(
        crown_w <= ibp_w + 1e-9,
        "CROWN boxes must never be looser in aggregate than IBP ({crown_w} > {ibp_w})"
    );
    assert!(
        crown_w < ibp_w,
        "CROWN must strictly tighten the encoder boxes on mscn_128d (the inc5d lever); \
         got {crown_w} vs {ibp_w}"
    );

    // Encode with the CROWN boxes; the big-M shrink must never ADD binaries.
    let (enc, peeled) = encode_graph_peel_final_sigmoid(&graph, &input_bounds, &crown_boxes)
        .expect("mscn_128d must encode with CROWN boxes");
    assert!(peeled, "final Sigmoid must be peeled");
    let (enc_ibp, _) = encode_graph_peel_final_sigmoid(&graph, &input_bounds, &ibp_boxes)
        .expect("mscn_128d must encode with IBP boxes");
    eprintln!(
        "mscn_crown_bounds_parity_128d: ReLU binaries {} (ibp) -> {} (crown-ibp)",
        enc_ibp.binary_vars.len(),
        enc.binary_vars.len()
    );
    assert!(
        enc.binary_vars.len() <= enc_ibp.binary_vars.len(),
        "tighter pre-activation boxes must never create MORE unstable ReLUs"
    );

    // (1) + (2): sampled-point trajectory oracle against the CROWN encoding.
    let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_unit = move || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let n_samples = 60;
    for k in 0..n_samples {
        let x: Vec<f64> = input_bounds
            .iter()
            .map(|b| {
                let (l, u) = (b.lower() as f64, b.upper() as f64);
                if l == u {
                    l
                } else {
                    let t = match k {
                        0 => 0.0,
                        1 => 1.0,
                        _ => next_unit(),
                    };
                    l + t * (u - l)
                }
            })
            .collect();
        let vals = mscn_f64_walk(&graph, &x, &input_shape);
        let assign = build_assignment(&enc, &graph, &vals, &crown_boxes);
        assert!(
            point_is_feasible(&enc.problem, &assign, 1e-6),
            "sample {k}: the true forward trajectory must stay FEASIBLE under CROWN boxes \
             (a violation = unsound/miskeyed bounds = the false-UNSAT mechanism); first \
             violation: {:?}",
            first_violation(&enc.problem, &assign, 1e-6)
        );
        let mut bad = assign.clone();
        bad[enc.output_vars[0].0] += 0.05;
        assert!(
            !point_is_feasible(&enc.problem, &bad, 1e-6),
            "sample {k}: a perturbed logit must stay MIP-infeasible under CROWN boxes"
        );
    }
}

// ── increment 5e: α-CROWN output bound row for the escalation ────────────────

/// inc5e — THE CRITICAL ORACLE for the α output row (the direct
/// anti-false-UNSAT check): on the REAL mscn_128d model + REAL instance box,
/// compute the α-CROWN output bound exactly the way the escalation does
/// (retargeted clone → `alpha_output_bound`), add the α row to the encoding
/// via `add_alpha_output_rows`, and assert for sampled points that
///   (1) the TRUE forward trajectory stays FEASIBLE (tol 1e-6) — if the α row
///       cuts off the true point, the bound or the node mapping is WRONG (the
///       false-UNSAT mechanism) and this increment must not ship;
///   (2) a perturbed logit stays INFEASIBLE (the row must not have weakened
///       the system into vacuity);
///   (3) the α machinery produced a bound at all and it is consistent with
///       (non-disjoint from) the encoder's own output column bound.
#[test]
fn mscn_alpha_row_parity_128d() {
    let Some(onnx) = nn4sys_onnx_path("mscn_128d.onnx") else {
        eprintln!("mscn_alpha_row_parity_128d: model not found; skipping");
        return;
    };
    let vnnlib = onnx
        .parent()
        .and_then(Path::parent)
        .map(|d| d.join("vnnlib/cardinality_0_1_128.vnnlib"))
        .expect("vnnlib path");
    if !vnnlib.exists() {
        eprintln!("mscn_alpha_row_parity_128d: vnnlib not found; skipping");
        return;
    }
    let graph = crate::commands::vnncomp::load_graph_network(&onnx).expect("load mscn_128d");
    let input_shape = [11usize, 14];
    let (raw_box, _, _) = parse_cardinality_vnnlib(&vnnlib, 11 * 14);
    let input_bounds: Vec<Bound> = raw_box
        .iter()
        .map(|&(l, u)| Bound::new(l as f32, u as f32))
        .collect();
    let node_bounds = ibp_node_boxes(&graph, &input_bounds, &input_shape);
    let (mut enc, peeled) = encode_graph_peel_final_sigmoid(&graph, &input_bounds, &node_bounds)
        .expect("mscn_128d must encode exactly");
    assert!(peeled, "final Sigmoid must be peeled");
    assert_eq!(enc.output_vars.len(), 1, "single logit output");

    // The escalation's exact α pipeline: peel-mirroring target, retargeted
    // clone, deadline-capped main-path α-CROWN.
    let target = alpha_output_target(&graph).expect("α target");
    assert_ne!(
        target,
        graph.output_name(),
        "mscn peels the final sigmoid: the α target must be the logit node"
    );
    let mut alpha_graph = graph.clone();
    alpha_graph.set_output(target.clone());
    let tensor = Verifier::bounds_to_tensor(&input_bounds, Some(&input_shape)).expect("box tensor");
    let deadline = Instant::now() + Duration::from_mins(1);
    let alpha = alpha_output_bound(&alpha_graph, &tensor, deadline, 0)
        .expect("α-CROWN must produce a well-formed logit bound on mscn_128d");
    let col = enc.output_vars[0];
    let (old_lb, old_ub) = (enc.problem.cols()[col.0].lb, enc.problem.cols()[col.0].ub);
    let a_lo = alpha.lower().iter().copied().next().expect("1 element");
    let a_hi = alpha.upper().iter().copied().next().expect("1 element");
    eprintln!(
        "mscn_alpha_row_parity_128d: α logit bound [{a_lo}, {a_hi}] vs encoder output col \
         [{old_lb}, {old_ub}] (target '{target}')"
    );

    // (3) The two sound enclosures must intersect (disjoint = wrong bound or
    // wrong node = the exact failure this oracle exists to catch).
    let rows = add_alpha_output_rows(&mut enc, &alpha, 0)
        .expect("α bound must be consistent with the encoder's output column bound");
    eprintln!(
        "mscn_alpha_row_parity_128d: α row effective [{}, {}], {} row(s) added",
        rows.effective[0].0, rows.effective[0].1, rows.rows_added
    );

    // (1) + (2): sampled-point trajectory oracle with the α row present.
    let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_unit = move || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let n_samples = 60;
    for k in 0..n_samples {
        let x: Vec<f64> = input_bounds
            .iter()
            .map(|b| {
                let (l, u) = (b.lower() as f64, b.upper() as f64);
                if l == u {
                    l
                } else {
                    let t = match k {
                        0 => 0.0,
                        1 => 1.0,
                        _ => next_unit(),
                    };
                    l + t * (u - l)
                }
            })
            .collect();
        let vals = mscn_f64_walk(&graph, &x, &input_shape);
        let assign = build_assignment(&enc, &graph, &vals, &node_bounds);
        assert!(
            point_is_feasible(&enc.problem, &assign, 1e-6),
            "sample {k}: the true forward trajectory must stay FEASIBLE with the α row (a \
             violation = the α bound or node mapping is WRONG = the false-UNSAT mechanism); \
             first violation: {:?}",
            first_violation(&enc.problem, &assign, 1e-6)
        );
        let mut bad = assign.clone();
        bad[col.0] += 0.05;
        assert!(
            !point_is_feasible(&enc.problem, &bad, 1e-6),
            "sample {k}: a perturbed logit must stay MIP-infeasible with the α row"
        );
    }
}

/// The dual model's output is Sub(Sigmoid, Sigmoid) — the Sigmoids are NOT
/// final, so the exact encoder must FAIL CLOSED on them (this is precisely
/// what still blocks mscn_128d_dual coverage; documented in the module
/// header).
#[test]
fn mscn_dual_fails_closed_on_nonfinal_sigmoid() {
    let Some(onnx) = nn4sys_onnx_path("mscn_128d_dual.onnx") else {
        eprintln!("mscn_dual_fails_closed_on_nonfinal_sigmoid: model not found; skipping");
        return;
    };
    let vnnlib = onnx
        .parent()
        .and_then(Path::parent)
        .map(|d| d.join("vnnlib/cardinality_1_10450_128_dual.vnnlib"))
        .expect("vnnlib path");
    if !vnnlib.exists() {
        eprintln!("mscn_dual_fails_closed_on_nonfinal_sigmoid: vnnlib not found; skipping");
        return;
    }
    let graph = crate::commands::vnncomp::load_graph_network(&onnx).expect("load dual");
    let input_shape = [22usize, 14];
    let (raw_box, _, _) = parse_cardinality_vnnlib(&vnnlib, 22 * 14);
    let input_bounds: Vec<Bound> = raw_box
        .iter()
        .map(|&(l, u)| Bound::new(l as f32, u as f32))
        .collect();
    let node_bounds = ibp_node_boxes(&graph, &input_bounds, &input_shape);

    let err = encode_graph_peel_final_sigmoid(&graph, &input_bounds, &node_bounds);
    assert!(err.is_err(), "the dual model must fail closed");
    let msg = format!("{:#}", err.unwrap_err());
    assert!(
        msg.contains("Sigmoid"),
        "the blocker must be the non-final Sigmoid, got: {msg}"
    );
}

// ── increment 5b: escalation-helper tests ───────────────────────────────────

/// Clause-box f64→f32 conversion: pins are preserved (degenerate bounds stay
/// degenerate — the encoder's exact MulBinary/Div lane depends on it), and
/// non-degenerate endpoints only ever move OUTWARD (soundness: the f32 box
/// must enclose every f32 point of the real interval).
#[test]
fn clause_bound_conversion_preserves_pins_and_rounds_outward() {
    // Exactly representable pin (the mscn mask case): stays a pin, exact.
    let b = clause_bound_to_f32(1.0, 1.0).expect("pin");
    assert_eq!((b.lower(), b.upper()), (1.0, 1.0));
    let b = clause_bound_to_f32(0.0, 0.0).expect("pin");
    assert_eq!((b.lower(), b.upper()), (0.0, 0.0));

    // Non-representable pin: still a pin (nearest f32; sound — no f32 point
    // lies in the true degenerate interval, so any superset certifies).
    let v = 0.1_f64;
    let b = clause_bound_to_f32(v, v).expect("pin");
    assert_eq!(b.lower(), b.upper());
    assert_eq!(b.lower(), v as f32);

    // Non-degenerate interval with non-representable endpoints: outward only.
    let (l, u) = (0.1_f64, 0.3_f64);
    let b = clause_bound_to_f32(l, u).expect("interval");
    assert!(
        (b.lower() as f64) <= l && (b.upper() as f64) >= u,
        "converted box [{}, {}] must enclose [{l}, {u}]",
        b.lower(),
        b.upper()
    );

    // Exactly representable non-degenerate endpoints: unchanged (no widening).
    let b = clause_bound_to_f32(0.25, 0.5).expect("interval");
    assert_eq!((b.lower(), b.upper()), (0.25, 0.5));

    // NaN / inverted bounds fail closed.
    assert!(clause_bound_to_f32(f64::NAN, 1.0).is_none());
    assert!(clause_bound_to_f32(0.0, f64::NAN).is_none());
    assert!(clause_bound_to_f32(1.0, 0.0).is_none());
}

/// A tiny Linear→Sigmoid graph whose encoding peels the final Sigmoid — the
/// fixture for the violation-constraint transform tests.
fn peeled_logit_encoding() -> GraphMipEncoding {
    let w = Array2::from_shape_vec((1, 2), vec![1.0f32, -1.0]).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "gemm",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["gemm".to_string()],
    ));
    graph.set_output("sig");
    let input_bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
    let node_bounds: HashMap<String, Vec<Bound>> =
        [("gemm".to_string(), vec![Bound::new(-2.0, 2.0)])].into();
    let (enc, peeled) = encode_graph_peel_final_sigmoid(&graph, &input_bounds, &node_bounds)
        .expect("tiny sigmoid graph must encode");
    assert!(peeled, "final Sigmoid must be peeled");
    enc
}

/// σ-space violation thresholds map onto the logit with OUTWARD slack; the
/// vacuous/infeasible edges behave per the fail-closed contract:
///  * vacuous (σ <= t, t >= 1): dropped — dropping only weakens, sound;
///  * infeasible (σ <= t, t <= 0): `Err` — the f32 sigmoid can saturate to
///    exactly 0.0/1.0, so certifying the clause impossible would risk a false
///    unsat;
///  * non-const constraint shapes: `Err` (V1 supports only const bounds).
#[test]
fn violation_constraint_transform_edges_fail_closed() {
    use ny_onnx::vnnlib::OutputConstraint as OC;

    // Bound case: a row is added, weakened outward by DELTA + σ-slack.
    let mut enc = peeled_logit_encoding();
    let rows_before = enc.problem.num_rows();
    let outcome = add_violation_constraint(&mut enc, &OC::LessEqConst(0, 0.5), true)
        .expect("mid-range threshold must transform");
    assert_eq!(outcome, ClauseConstraintOutcome::Added);
    assert_eq!(enc.problem.num_rows(), rows_before + 1);
    let row = enc.problem.rows().last().unwrap();
    assert_eq!(row.coeffs, vec![(enc.output_vars[0].0, 1.0)]);
    // logit(0.5) = 0; the ub must sit ABOVE it by the outward slack (weaker).
    assert!(row.lb == f64::NEG_INFINITY && row.ub > 0.0 && row.ub < 0.1);

    // Lower-threshold case is symmetric (>=): lb BELOW logit(0.5) = 0.
    let mut enc = peeled_logit_encoding();
    add_violation_constraint(&mut enc, &OC::GreaterEqConst(0, 0.5), true).expect("lower");
    let row = enc.problem.rows().last().unwrap();
    assert!(row.ub == f64::INFINITY && row.lb < 0.0 && row.lb > -0.1);

    // Vacuous: dropped, no row added.
    let mut enc = peeled_logit_encoding();
    let rows_before = enc.problem.num_rows();
    let outcome = add_violation_constraint(&mut enc, &OC::LessEqConst(0, 1.5), true)
        .expect("vacuous transforms fine");
    assert_eq!(outcome, ClauseConstraintOutcome::DroppedVacuous);
    assert_eq!(enc.problem.num_rows(), rows_before);

    // Infeasible edge (σ <= 0): FAIL CLOSED, never "clause impossible".
    let mut enc = peeled_logit_encoding();
    assert!(add_violation_constraint(&mut enc, &OC::LessEqConst(0, 0.0), true).is_err());
    assert!(add_violation_constraint(&mut enc, &OC::GreaterEqConst(0, 1.0), true).is_err());

    // Saturation guard band: thresholds hugging 0/1 fail closed too.
    let mut enc = peeled_logit_encoding();
    assert!(add_violation_constraint(&mut enc, &OC::LessEqConst(0, 1e-6), true).is_err());
    assert!(add_violation_constraint(&mut enc, &OC::GreaterEqConst(0, 1.0 - 1e-9), true).is_err());

    // Non-const constraint shape: fail closed (V1 contract).
    let mut enc = peeled_logit_encoding();
    assert!(add_violation_constraint(&mut enc, &OC::LessEq(0, 1), true).is_err());

    // Out-of-range output index: fail closed.
    let mut enc = peeled_logit_encoding();
    assert!(add_violation_constraint(&mut enc, &OC::LessEqConst(7, 0.5), true).is_err());
}

/// 1-D `BoundedTensor` helper for the α-row unit tests.
fn alpha_bt(lower: &[f32], upper: &[f32]) -> ny_tensor::BoundedTensor {
    ny_tensor::BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// inc5e — the α target mirrors the encoder's peel decision exactly: the
/// logit (Sigmoid input) for a final-Sigmoid graph, the output node itself
/// otherwise.
#[test]
fn alpha_output_target_mirrors_encoder_peel() {
    // Final-Sigmoid graph (the peel case): target = the logit node.
    let w = Array2::from_shape_vec((1, 2), vec![1.0f32, -1.0]).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "gemm",
        Layer::Linear(LinearLayer::new(w.clone(), None).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["gemm".to_string()],
    ));
    graph.set_output("sig");
    assert_eq!(alpha_output_target(&graph).as_deref(), Some("gemm"));

    // Non-Sigmoid output (no peel): target = the output node itself.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "gemm",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.set_output("gemm");
    assert_eq!(alpha_output_target(&graph).as_deref(), Some("gemm"));
}

/// inc5e — the α row is INTERSECTED into the encoding with the encoder's own
/// `±DELTA` outward discipline: never tighter than `α ∓ DELTA`, never looser
/// than the existing column bound, and the column bound itself is untouched
/// (a row is added, never a replacement).
#[test]
fn alpha_output_rows_tighten_with_delta_discipline() {
    let mut enc = peeled_logit_encoding();
    let col = enc.output_vars[0];
    let (old_lb, old_ub) = (enc.problem.cols()[col.0].lb, enc.problem.cols()[col.0].ub);
    let rows_before = enc.problem.num_rows();

    let rows = add_alpha_output_rows(&mut enc, &alpha_bt(&[-1.0], &[1.0]), 0)
        .expect("overlapping enclosures must be accepted");
    assert_eq!(rows.rows_added, 1);
    assert_eq!(enc.problem.num_rows(), rows_before + 1);
    let row = enc.problem.rows().last().unwrap();
    assert_eq!(row.coeffs, vec![(col.0, 1.0)]);
    // DELTA discipline: the row must sit OUTSIDE α∓DELTA (the f32-net vs
    // f64-affine absorber `out_col_bounds` applies to every other collector
    // box), and within 2·DELTA of it (i.e. the inflation really is ~DELTA).
    assert!(
        row.lb <= -1.0 - DELTA,
        "row lb {} must be <= -1 - DELTA",
        row.lb
    );
    assert!(row.lb >= -1.0 - 2.0 * DELTA);
    assert!(
        row.ub >= 1.0 + DELTA,
        "row ub {} must be >= 1 + DELTA",
        row.ub
    );
    assert!(row.ub <= 1.0 + 2.0 * DELTA);
    // Intersected with (never looser than) the existing column bound.
    assert!(row.lb >= old_lb && row.ub <= old_ub);
    assert_eq!(rows.effective[0], (row.lb, row.ub));
    // The column bound itself is untouched.
    assert_eq!(enc.problem.cols()[col.0].lb, old_lb);
    assert_eq!(enc.problem.cols()[col.0].ub, old_ub);
}

/// inc5e — a LOOSER-than-column α bound adds no row (nothing to tighten) and
/// reports the column bound as the effective interval; a one-sided
/// improvement adds exactly one row whose other side is the column bound.
#[test]
fn alpha_output_rows_skip_non_tightening_and_intersect_one_sided() {
    // Looser than the column bound on both sides: no row.
    let mut enc = peeled_logit_encoding();
    let col = enc.output_vars[0];
    let (old_lb, old_ub) = (enc.problem.cols()[col.0].lb, enc.problem.cols()[col.0].ub);
    let rows_before = enc.problem.num_rows();
    let rows = add_alpha_output_rows(&mut enc, &alpha_bt(&[-100.0], &[100.0]), 0)
        .expect("looser enclosures overlap");
    assert_eq!(rows.rows_added, 0);
    assert_eq!(enc.problem.num_rows(), rows_before);
    assert_eq!(rows.effective[0], (old_lb, old_ub));

    // One-sided improvement: the row's loose side is the COLUMN bound (the
    // intersection), not the α value.
    let mut enc = peeled_logit_encoding();
    let rows = add_alpha_output_rows(&mut enc, &alpha_bt(&[-1.0], &[100.0]), 0)
        .expect("one-sided improvement overlaps");
    assert_eq!(rows.rows_added, 1);
    let row = enc.problem.rows().last().unwrap();
    assert!(row.lb <= -1.0 - DELTA && row.lb >= -1.0 - 2.0 * DELTA);
    assert_eq!(
        row.ub, old_ub,
        "the non-improving side must stay the column bound"
    );
}

/// inc5e — fail-open contract: a DISJOINT α bound (evidence of a wrong bound
/// or wrong node mapping upstream — the false-UNSAT mechanism) and a
/// length-mismatched α bound both add NOTHING and return `None`.
#[test]
fn alpha_output_rows_fail_open_on_disjoint_and_mismatch() {
    // Disjoint from the column bound ([-2-DELTA, 2+DELTA] on the fixture).
    let mut enc = peeled_logit_encoding();
    let rows_before = enc.problem.num_rows();
    assert!(add_alpha_output_rows(&mut enc, &alpha_bt(&[5.0], &[6.0]), 0).is_none());
    assert_eq!(enc.problem.num_rows(), rows_before, "no row on disjoint");

    // Length mismatch (2 α elements vs 1 output column).
    let mut enc = peeled_logit_encoding();
    let rows_before = enc.problem.num_rows();
    assert!(add_alpha_output_rows(&mut enc, &alpha_bt(&[0.0, 0.0], &[1.0, 1.0]), 0).is_none());
    assert_eq!(enc.problem.num_rows(), rows_before, "no row on mismatch");
}

/// The legacy escalation entry point itself stays inert while the explicit
/// kill switch is set. Production dispatch uses the strict planner instead.
#[test]
fn graph_mip_escalation_noop_when_gate_off() {
    if graph_mip_enabled() {
        eprintln!(
            "graph_mip_escalation_noop_when_gate_off: set NY_GRAPH_MIP=0 to exercise; skipping"
        );
        return;
    }
    let spec = VnnLibSpec::new();
    let out = try_graph_mip_escalation(
        Path::new("/nonexistent/model.onnx"),
        &OnnxLoadConfig::default(),
        &[1],
        Some(&spec),
        MipBackend::Ay,
        60,
    );
    assert!(out.is_none(), "gate OFF must be a no-op");
}

/// END-TO-END certify-path validation on the REAL mscn_128d model + a REAL
/// all-unsat cardinality instance (abc ground truth: every mscn instance is
/// unsat): the escalation must reload the graph, build both per-clause boxes,
/// run per-clause IBP, encode, solve on ay, and return the certified-unsat
/// marker ONLY if BOTH clauses came back `Unsat { certified: true }`.
///
/// MANUAL-PROBE GATE — runs only with explicit `NY_GRAPH_MIP=1`, so the
/// default-on production policy does not make CI consume a local corpus.
/// Run manually:
///   NY_GRAPH_MIP=1 cargo test --release -p ny-cli --features mip \
///     mscn_escalation_certifies_real_instance -- --nocapture
#[test]
fn mscn_escalation_certifies_real_instance() {
    if !graph_mip_manual_probe_enabled() {
        eprintln!("mscn_escalation_certifies_real_instance: NY_GRAPH_MIP != 1; skipping");
        return;
    }
    let Some(onnx) = nn4sys_onnx_path("mscn_128d.onnx") else {
        eprintln!("mscn_escalation_certifies_real_instance: model not found; skipping");
        return;
    };
    let vnnlib_path = onnx
        .parent()
        .and_then(Path::parent)
        .map(|d| d.join("vnnlib/cardinality_0_1_128.vnnlib"))
        .expect("vnnlib path");
    if !vnnlib_path.exists() {
        eprintln!("mscn_escalation_certifies_real_instance: vnnlib not found; skipping");
        return;
    }
    let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib_path).expect("parse vnnlib");
    assert!(
        !spec.output_constraint_clauses.is_empty(),
        "cardinality instance must parse to violation clauses"
    );
    // Surface the escalation's info! decision trail in `--nocapture` runs.
    let _ = tracing_subscriber::fmt()
        .with_max_level(
            if std::env::var("NY_GRAPH_MIP_TEST_DEBUG").ok().as_deref() == Some("1") {
                tracing::Level::DEBUG
            } else {
                tracing::Level::INFO
            },
        )
        .with_test_writer()
        .try_init();

    // Budget override for manual probing (default 60s → 30s per clause).
    let budget: u64 = std::env::var("NY_GRAPH_MIP_TEST_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let start = Instant::now();
    let out = try_graph_mip_escalation(
        &onnx,
        &OnnxLoadConfig::default(),
        &[11, 14],
        Some(&spec),
        MipBackend::Ay,
        budget,
    );
    let elapsed = start.elapsed();
    eprintln!(
        "mscn_escalation_certifies_real_instance: {} clauses, budget {budget}s, {:.3}s -> {:?}",
        spec.output_constraint_clauses.len(),
        elapsed.as_secs_f64(),
        out
    );
    // The path must never panic and never fabricate a sat. `None` is the
    // fail-closed outcome when ay cannot decide a clause within its slice
    // (measured 2026-07: one mscn_128d clause exceeds a 30s ay slice — see
    // the info! trail); `Some` is the certified-unsat outcome and must carry
    // the spec's output arity.
    match out {
        Some(unsat) => assert_eq!(unsat.num_outputs, 1),
        None => eprintln!(
            "mscn_escalation_certifies_real_instance: escalation fell back (ay did not certify \
             within budget) — fail-closed outcome, see log for the stalling clause"
        ),
    }
}

/// inc5d probe — the SAME end-to-end escalation contract on the larger
/// mscn_2048d model + its smallest-clause-count instance
/// (cardinality_0_1_2048, 2 clauses, official budget 20s). ENV-GATED like the
/// 128d probe; run manually:
///   NY_GRAPH_MIP=1 cargo test --release -p ny-cli --features mip \
///     mscn_escalation_2048d_probe -- --nocapture
#[test]
fn mscn_escalation_2048d_probe() {
    if !graph_mip_manual_probe_enabled() {
        eprintln!("mscn_escalation_2048d_probe: NY_GRAPH_MIP != 1; skipping");
        return;
    }
    let Some(onnx) = nn4sys_onnx_path("mscn_2048d.onnx") else {
        eprintln!("mscn_escalation_2048d_probe: model not found; skipping");
        return;
    };
    let vnnlib_path = onnx
        .parent()
        .and_then(Path::parent)
        .map(|d| d.join("vnnlib/cardinality_0_1_2048.vnnlib"))
        .expect("vnnlib path");
    if !vnnlib_path.exists() {
        eprintln!("mscn_escalation_2048d_probe: vnnlib not found; skipping");
        return;
    }
    let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib_path).expect("parse vnnlib");
    assert!(
        !spec.output_constraint_clauses.is_empty(),
        "cardinality instance must parse to violation clauses"
    );
    let _ = tracing_subscriber::fmt()
        .with_max_level(
            if std::env::var("NY_GRAPH_MIP_TEST_DEBUG").ok().as_deref() == Some("1") {
                tracing::Level::DEBUG
            } else {
                tracing::Level::INFO
            },
        )
        .with_test_writer()
        .try_init();
    let budget: u64 = std::env::var("NY_GRAPH_MIP_TEST_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20); // the official instances.csv budget for this row
                        // The 2048d model input is [11, 224] (11 rows x (2048/32 + extras)); derive
                        // the flat size from the spec instead of hardcoding a possibly-wrong shape.
    let flat = spec.input_bounds.len();
    assert_eq!(flat % 11, 0, "mscn input rows");
    let start = Instant::now();
    let out = try_graph_mip_escalation(
        &onnx,
        &OnnxLoadConfig::default(),
        &[11, flat / 11],
        Some(&spec),
        MipBackend::Ay,
        budget,
    );
    let elapsed = start.elapsed();
    eprintln!(
        "mscn_escalation_2048d_probe: {} clauses, budget {budget}s, {:.3}s -> {:?}",
        spec.output_constraint_clauses.len(),
        elapsed.as_secs_f64(),
        out
    );
    match out {
        Some(unsat) => assert_eq!(unsat.num_outputs, 1),
        None => eprintln!(
            "mscn_escalation_2048d_probe: escalation fell back (fail-closed outcome; see the \
             info! trail for the stalling clause)"
        ),
    }
}

/// inc5d measurement probe — the PER-CLAUSE outcome table for mscn_128d +
/// cardinality_0_1_128. The production escalation fail-closes at the FIRST
/// non-certified clause, so later clauses never get measured; this probe runs
/// each clause as its own single-clause escalation to attribute
/// root-certified vs branched-uncertified vs timeout per clause (the decisive
/// inc5d metric). Measurement-only: single-clause escalation outcomes are
/// never combined into a verdict here. ENV-GATED like the other probes:
///   NY_GRAPH_MIP=1 cargo test --release -p ny-cli --features mip \
///     mscn_escalation_per_clause_table_128d -- --nocapture
/// `NY_GRAPH_MIP_TEST_BUDGET` = per-clause budget seconds (default 10 = the
/// official 20s instance budget split over its 2 clauses).
#[test]
fn mscn_escalation_per_clause_table_128d() {
    if !graph_mip_manual_probe_enabled() {
        eprintln!("mscn_escalation_per_clause_table_128d: NY_GRAPH_MIP != 1; skipping");
        return;
    }
    let Some(onnx) = nn4sys_onnx_path("mscn_128d.onnx") else {
        eprintln!("mscn_escalation_per_clause_table_128d: model not found; skipping");
        return;
    };
    let vnnlib_path = onnx
        .parent()
        .and_then(Path::parent)
        .map(|d| d.join("vnnlib/cardinality_0_1_128.vnnlib"))
        .expect("vnnlib path");
    if !vnnlib_path.exists() {
        eprintln!("mscn_escalation_per_clause_table_128d: vnnlib not found; skipping");
        return;
    }
    let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib_path).expect("parse vnnlib");
    let n = spec.output_constraint_clauses.len();
    assert!(
        n > 0,
        "cardinality instance must parse to violation clauses"
    );
    let _ = tracing_subscriber::fmt()
        .with_max_level(
            if std::env::var("NY_GRAPH_MIP_TEST_DEBUG").ok().as_deref() == Some("1") {
                tracing::Level::DEBUG
            } else {
                tracing::Level::INFO
            },
        )
        .with_test_writer()
        .try_init();
    let budget: u64 = std::env::var("NY_GRAPH_MIP_TEST_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    for k in 0..n {
        let mut single = spec.clone();
        single.output_constraint_clauses = vec![spec.output_constraint_clauses[k].clone()];
        if !spec.per_clause_input_bounds.is_empty() {
            single.per_clause_input_bounds = vec![spec.per_clause_input_bounds[k].clone()];
        }
        let start = Instant::now();
        let out = try_graph_mip_escalation(
            &onnx,
            &OnnxLoadConfig::default(),
            &[11, 14],
            Some(&single),
            MipBackend::Ay,
            budget,
        );
        eprintln!(
            "per-clause table: clause {}/{n} -> {} in {:.2}s (budget {budget}s)",
            k + 1,
            if out.is_some() {
                "ROOT-CERTIFIED unsat"
            } else {
                "not certified (see info trail)"
            },
            start.elapsed().as_secs_f64()
        );
    }
}
