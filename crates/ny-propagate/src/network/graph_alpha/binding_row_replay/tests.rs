// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #binding-row-replay gate tests (consult #6 §4 arbitration):
//!
//! 1. CORRECTNESS — central finite differences of the certified DAG alpha
//!    fold on a conv+residual fixture (the `margin_resnet_fixture` idiom from
//!    `propagate_dag/iter0_parity_tests.rs`) agree with the replay gradient:
//!    sign agreement above a scale floor + cosine > 0.99, plus a whole-replay
//!    parity check (replayed row value vs the fold's certified row lower).
//! 2. TIMING — 20-rep manual timer on a synthetic net with cifar100
//!    `resnet_medium`'s REAL topology (extracted from
//!    `CIFAR100_resnet_medium.onnx`: 16 convs, 10 ReLUs, 55,460 ReLU neurons,
//!    100 seed rows), reporting p50/p95 per binding row.

use std::collections::HashMap;
use std::time::Instant;

use ndarray::{Array, Array1, Array2, IxDyn};
use ny_tensor::BoundedTensor;

use crate::bounds::{GraphAlphaCrownIntermediate, GraphAlphaState, LinearBounds};
use crate::layers::{FlattenLayer, Layer, LinearLayer};
use crate::network::core::{GraphNetwork, GraphNode, NETWORK_INPUT};
use crate::network::graph_alpha::resnet_skeleton::test_support::{add, box_input, conv, lcg, relu};

/// The iter0-parity conv+residual fixture verbatim (identity skip +
/// projection skip + 10 margin rows through one shared alpha per ReLU).
fn margin_resnet_fixture() -> GraphNetwork {
    let mut rng = lcg(0x17E2_0A1F_A171_7E57);
    let mut g = GraphNetwork::new();
    g.add_node(conv(
        &mut rng,
        "conv0",
        NETWORK_INPUT,
        (2, 4),
        3,
        1,
        1,
        true,
    ));
    g.add_node(relu("relu0", "conv0"));
    g.add_node(conv(&mut rng, "b1c1", "relu0", (4, 4), 3, 1, 1, false));
    g.add_node(relu("b1r1", "b1c1"));
    g.add_node(conv(&mut rng, "b1c2", "b1r1", (4, 4), 3, 1, 1, true));
    g.add_node(add("add1", "b1c2", "relu0"));
    g.add_node(relu("b2r1", "add1"));
    g.add_node(conv(&mut rng, "b2c1", "b2r1", (4, 8), 3, 1, 1, true));
    g.add_node(conv(&mut rng, "p2c1", "add1", (4, 8), 1, 1, 0, false));
    g.add_node(add("add2", "b2c1", "p2c1"));
    g.add_node(relu("relu_out", "add2"));
    g.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["relu_out".to_string()],
    ));
    let in_features = 8 * 6 * 6;
    let out_rows = 10;
    let w = Array2::from_shape_fn((out_rows, in_features), |_| rng() * 0.2);
    let b = Array1::from_shape_fn(out_rows, |_| rng() * 0.1);
    g.add_node(GraphNode::new(
        "margins",
        Layer::Linear(LinearLayer::new(w, Some(b)).expect("valid margin gemm")),
        vec!["flatten".to_string()],
    ));
    g.set_output("margins");
    g
}

const FIXTURE_RELUS: [&str; 4] = ["relu0", "b1r1", "b2r1", "relu_out"];

struct Harness {
    graph: GraphNetwork,
    input: BoundedTensor,
    node_bounds: HashMap<String, BoundedTensor>,
    exec_order: Vec<String>,
    relu_name_to_idx: HashMap<String, usize>,
    output_dim: usize,
    input_dim: usize,
}

impl Harness {
    fn new() -> Self {
        let graph = margin_resnet_fixture();
        let input = box_input(&[2, 6, 6], -0.6, 0.6);
        let node_bounds = graph.collect_node_bounds(&input).expect("IBP forward");
        let exec_order = graph.exec_order().expect("exec order").to_vec();
        let relu_name_to_idx: HashMap<String, usize> = FIXTURE_RELUS
            .iter()
            .enumerate()
            .map(|(i, n)| (n.to_string(), i))
            .collect();
        Self {
            graph,
            input,
            node_bounds,
            exec_order,
            relu_name_to_idx,
            output_dim: 10,
            input_dim: 2 * 6 * 6,
        }
    }

    /// Alpha state with LCG-interior values in (0.2, 0.8) so central FD
    /// probes at h = 5e-3 stay inside [0, 1].
    fn interior_alpha_state(&self) -> GraphAlphaState {
        let mut st = GraphAlphaState::new();
        let mut rng = lcg(0xB1D1_0A11_5EED);
        for name in FIXTURE_RELUS {
            let pre = self
                .graph
                .relu_preactivation_bounds(name, &self.input, &self.node_bounds, "brr-test")
                .expect("pre bounds");
            st.add_relu_node(name, pre, false).expect("alpha init");
            let alpha = st.alphas.get_mut(name).expect("lower alpha");
            for a in alpha.iter_mut() {
                *a = 0.5 + 0.3 * rng(); // in (0.2, 0.8)
            }
        }
        st
    }

    /// CHANNEL-SHARED alpha state (#channel-alpha-grad): `add_relu_node` with
    /// `channel_only_alpha=true` — the production `full_conv_alpha: false`
    /// wiring (`propagate_dag/init.rs:162` passes `!config.full_conv_alpha`) —
    /// so each conv ReLU carries one α_c per channel (length C, spatial_shapes
    /// recorded). Interior values keep FD probes inside [0, 1].
    fn channel_alpha_state(&self) -> GraphAlphaState {
        let mut st = GraphAlphaState::new();
        let mut rng = lcg(0xC4A2_B1D1_5EED);
        for name in FIXTURE_RELUS {
            let pre = self
                .graph
                .relu_preactivation_bounds(name, &self.input, &self.node_bounds, "brr-test")
                .expect("pre bounds");
            st.add_relu_node(name, pre, true).expect("alpha init");
            assert!(
                st.spatial_shapes.contains_key(name),
                "fixture ReLU '{name}' must actually take the channel-only path"
            );
            let channels = st.spatial_shapes[name][0];
            let alpha = st.alphas.get_mut(name).expect("lower alpha");
            assert_eq!(alpha.len(), channels, "channel-only alpha is length C");
            for a in alpha.iter_mut() {
                *a = 0.5 + 0.3 * rng(); // in (0.2, 0.8)
            }
        }
        st
    }

    fn zero_grad_buffers(&self) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let g: Vec<Array1<f32>> = FIXTURE_RELUS
            .iter()
            .map(|name| {
                let pre = self
                    .graph
                    .relu_preactivation_bounds(name, &self.input, &self.node_bounds, "brr-test")
                    .expect("pre bounds");
                Array1::zeros(pre.len())
            })
            .collect();
        let gu = g.iter().map(|v| Array1::zeros(v.len())).collect();
        (g, gu)
    }

    /// Certified fold, bounds only (the FD oracle's function value).
    fn fold_lower(&self, st: &GraphAlphaState, row: usize) -> f32 {
        let (mut g, mut gu) = self.zero_grad_buffers();
        let bounds = self
            .graph
            .dag_alpha_backward_pass_with_engine(
                &self.input,
                &self.node_bounds,
                &self.exec_order,
                self.output_dim,
                self.input_dim,
                &self.relu_name_to_idx,
                st,
                None,
                &mut g,
                &mut gu,
                None,
                None,
                None,
                None,
            )
            .expect("alpha fold");
        bounds.lower()[[row]]
    }

    fn fold_with_intermediates(
        &self,
        st: &GraphAlphaState,
    ) -> (BoundedTensor, GraphAlphaCrownIntermediate) {
        let (mut g, mut gu) = self.zero_grad_buffers();
        self.graph
            .dag_alpha_backward_pass_with_intermediates(
                &self.input,
                &self.node_bounds,
                &self.exec_order,
                self.output_dim,
                self.input_dim,
                &self.relu_name_to_idx,
                st,
                None,
                &mut g,
                &mut gu,
                None,
                None,
                None,
                None,
            )
            .expect("alpha fold with intermediates")
    }

    fn fd_grad(&self, st: &GraphAlphaState, relu: &str, i: usize, row: usize, h: f32) -> f32 {
        let base = st.alphas.get(relu).expect("alpha")[i];
        assert!(
            base - h > 0.0 && base + h < 1.0,
            "FD probe must stay interior to [0,1]"
        );
        let mut plus = st.clone();
        plus.alphas.get_mut(relu).expect("alpha")[i] = base + h;
        let mut minus = st.clone();
        minus.alphas.get_mut(relu).expect("alpha")[i] = base - h;
        (self.fold_lower(&plus, row) - self.fold_lower(&minus, row)) / (2.0 * h)
    }
}

/// GATE (a): the replay gradient IS the derivative of the certified fold —
/// central finite differences on the conv+residual fixture, all four ReLU
/// layers sampled, sign agreement above a scale floor + cosine > 0.99, plus
/// the whole-replay row-value parity check.
#[ntest::timeout(300000)]
#[test]
fn replay_gradient_matches_fold_finite_differences_on_conv_residual_fixture() {
    let h = Harness::new();
    let st = h.interior_alpha_state();

    let (bounds, inter) = h.fold_with_intermediates(&st);
    assert!(
        bounds.lower().iter().all(|v| v.is_finite()),
        "fixture fold must be finite"
    );

    // Binding row: the fold's weakest (minimum) lower-bound row — the row the
    // margin objective actually ascends.
    let binding_row = bounds
        .lower()
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
        .map(|(i, _)| i)
        .expect("nonempty output");
    let fold_row_lower = bounds.lower()[[binding_row]] as f64;

    let replay = h
        .graph
        .binding_row_true_alpha_grads(&h.input, &st, &inter, binding_row)
        .expect("replay succeeds");

    // Whole-replay parity: replayed relaxed row value == certified row lower
    // (modulo the fold's directed rounding / f64 bias accumulation).
    let parity_err = (replay.replayed_row_value - fold_row_lower).abs();
    let parity_tol = 1e-3 + 1e-4 * fold_row_lower.abs();
    eprintln!(
        "[brr] row={binding_row} fold_lower={fold_row_lower:+.6} replayed={:+.6} |err|={parity_err:.3e} (tol {parity_tol:.3e})",
        replay.replayed_row_value
    );
    assert!(
        parity_err <= parity_tol,
        "#binding-row-replay: replayed row value {} != certified fold lower {fold_row_lower} \
         (err {parity_err:.3e} > tol {parity_tol:.3e}) — the replay is NOT reproducing the walk",
        replay.replayed_row_value
    );

    // FD sweep: every 5th neuron of every ReLU layer (deterministic sample).
    let fd_h = 5e-3f32;
    let mut fd_vals: Vec<f64> = Vec::new();
    let mut true_vals: Vec<f64> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    for name in FIXTURE_RELUS {
        let g = replay.grads.get(name).expect("replay grad for relu");
        for i in (0..g.len()).step_by(5) {
            let fd = h.fd_grad(&st, name, i, binding_row, fd_h) as f64;
            fd_vals.push(fd);
            true_vals.push(g[i] as f64);
            labels.push(format!("{name}[{i}]"));
        }
    }
    let n = fd_vals.len();
    assert!(n >= 100, "sample must be substantial, got {n}");

    let dot: f64 = fd_vals.iter().zip(&true_vals).map(|(a, b)| a * b).sum();
    let na: f64 = fd_vals.iter().map(|a| a * a).sum::<f64>().sqrt();
    let nb: f64 = true_vals.iter().map(|b| b * b).sum::<f64>().sqrt();
    assert!(na > 0.0 && nb > 0.0, "degenerate gradient sample");
    let cosine = dot / (na * nb);

    let max_abs_fd = fd_vals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    // Scale floor per consult #6: measure sign agreement only above it; tiny
    // components are inside the FD/f32 noise envelope.
    let floor = 1e-4f64.max(0.02 * max_abs_fd);
    let mut above = 0usize;
    let mut sign_ok = 0usize;
    let mut max_err = 0.0f64;
    let mut worst = String::new();
    for k in 0..n {
        let err = (fd_vals[k] - true_vals[k]).abs();
        if err > max_err {
            max_err = err;
            worst = labels[k].clone();
        }
        if fd_vals[k].abs() > floor {
            above += 1;
            if fd_vals[k].signum() == true_vals[k].signum() {
                sign_ok += 1;
            } else {
                eprintln!(
                    "[brr] SIGN MISMATCH {}: fd={:+.6} true={:+.6}",
                    labels[k], fd_vals[k], true_vals[k]
                );
            }
        }
    }
    eprintln!(
        "[brr] MEASURED n={n} cosine={cosine:.6} sign_ok={sign_ok}/{above} (floor {floor:.2e}) \
         max|fd-true|={max_err:.3e} at {worst} max|fd|={max_abs_fd:.3e}"
    );
    assert!(
        cosine > 0.99,
        "#binding-row-replay gate: cosine {cosine} <= 0.99"
    );
    assert!(above >= 20, "scale floor left too few components ({above})");
    assert_eq!(
        sign_ok, above,
        "#binding-row-replay gate: sign disagreement above the scale floor"
    );
}

/// #channel-alpha-grad FD gate: with CHANNEL-SHARED α (production
/// `full_conv_alpha: false`), moving α_c moves EVERY spatial position of
/// channel c in the certified fold, so the true derivative is the spatial sum
/// `dL/dα_c = Σ_{h,w} ν_{c,h,w}·ĥ_{c,h,w}(x*)` — exactly what the replay now
/// returns at channel width. Central FD of the certified fold on every
/// channel of all four conv ReLU layers: cosine > 0.99 + sign agreement
/// above a scale floor, spanning ≥ 2 conv ReLU layers, plus the whole-replay
/// row-value parity check.
#[ntest::timeout(300000)]
#[test]
fn channel_shared_replay_gradient_matches_fold_finite_differences() {
    let h = Harness::new();
    let st = h.channel_alpha_state();

    let (bounds, inter) = h.fold_with_intermediates(&st);
    assert!(
        bounds.lower().iter().all(|v| v.is_finite()),
        "fixture fold must be finite"
    );
    let binding_row = bounds
        .lower()
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
        .map(|(i, _)| i)
        .expect("nonempty output");
    let fold_row_lower = bounds.lower()[[binding_row]] as f64;

    let replay = h
        .graph
        .binding_row_true_alpha_grads(&h.input, &st, &inter, binding_row)
        .expect("channel-shared replay succeeds");

    // Whole-replay parity: replayed relaxed row value == certified row lower.
    let parity_err = (replay.replayed_row_value - fold_row_lower).abs();
    let parity_tol = 1e-3 + 1e-4 * fold_row_lower.abs();
    eprintln!(
        "[brr-ch] row={binding_row} fold_lower={fold_row_lower:+.6} replayed={:+.6} |err|={parity_err:.3e} (tol {parity_tol:.3e})",
        replay.replayed_row_value
    );
    assert!(
        parity_err <= parity_tol,
        "#channel-alpha-grad: replayed row value {} != certified fold lower {fold_row_lower} \
         (err {parity_err:.3e} > tol {parity_tol:.3e})",
        replay.replayed_row_value
    );

    // FD sweep over EVERY channel of every ReLU layer (4+4+4+8 = 20 α_c).
    let fd_h = 5e-3f32;
    let mut fd_vals: Vec<f64> = Vec::new();
    let mut true_vals: Vec<f64> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut layer_of: Vec<&str> = Vec::new();
    for name in FIXTURE_RELUS {
        let g = replay.grads.get(name).expect("replay grad for relu");
        let channels = st.spatial_shapes[name][0];
        assert_eq!(
            g.len(),
            channels,
            "replay gradient at '{name}' must be at ALPHA width (C), the layout \
             update_all_alphas consumes"
        );
        for c in 0..channels {
            let fd = h.fd_grad(&st, name, c, binding_row, fd_h) as f64;
            fd_vals.push(fd);
            true_vals.push(g[c] as f64);
            labels.push(format!("{name}[c={c}]"));
            layer_of.push(name);
        }
    }
    let n = fd_vals.len();
    assert_eq!(n, 20, "4+4+4+8 channels across the four conv ReLUs");

    let dot: f64 = fd_vals.iter().zip(&true_vals).map(|(a, b)| a * b).sum();
    let na: f64 = fd_vals.iter().map(|a| a * a).sum::<f64>().sqrt();
    let nb: f64 = true_vals.iter().map(|b| b * b).sum::<f64>().sqrt();
    assert!(na > 0.0 && nb > 0.0, "degenerate gradient sample");
    let cosine = dot / (na * nb);

    let max_abs_fd = fd_vals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let floor = 1e-4f64.max(0.02 * max_abs_fd);
    let mut above = 0usize;
    let mut sign_ok = 0usize;
    let mut layers_above: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for k in 0..n {
        if fd_vals[k].abs() > floor {
            above += 1;
            layers_above.insert(layer_of[k]);
            if fd_vals[k].signum() == true_vals[k].signum() {
                sign_ok += 1;
            } else {
                eprintln!(
                    "[brr-ch] SIGN MISMATCH {}: fd={:+.6} true={:+.6}",
                    labels[k], fd_vals[k], true_vals[k]
                );
            }
        }
    }
    eprintln!(
        "[brr-ch] MEASURED n={n} cosine={cosine:.6} sign_ok={sign_ok}/{above} (floor {floor:.2e}) \
         layers_above={layers_above:?} max|fd|={max_abs_fd:.3e}"
    );
    assert!(
        cosine > 0.99,
        "#channel-alpha-grad gate: cosine {cosine} <= 0.99"
    );
    assert!(
        layers_above.len() >= 2,
        "FD proof must span >= 2 conv ReLU layers, got {layers_above:?}"
    );
    assert_eq!(
        sign_ok, above,
        "#channel-alpha-grad gate: sign disagreement above the scale floor"
    );
}

/// #channel-alpha-grad refusal: an α width that is neither per-neuron nor
/// channel-reconcilable against the node's recorded geometry stays a TYPED
/// error (here: channel α whose spatial_shapes record was dropped).
#[ntest::timeout(60000)]
#[test]
fn channel_replay_refuses_irreconcilable_alpha_geometry() {
    let h = Harness::new();
    let mut st = h.channel_alpha_state();
    let (_bounds, inter) = h.fold_with_intermediates(&st);
    // Alpha stays length C, but the measured geometry key is gone: nothing
    // may guess a layout.
    st.spatial_shapes.remove("b1r1");
    let err = h
        .graph
        .binding_row_true_alpha_grads(&h.input, &st, &inter, 0)
        .expect_err("must refuse an unreconcilable alpha layout");
    assert!(
        err.to_string().contains("b1r1"),
        "error must name the offending node: {err}"
    );
}

/// The replay refuses partial state instead of guessing: missing dense
/// a_at_relu for a walked ReLU is a typed error, not a zero gradient.
#[ntest::timeout(60000)]
#[test]
fn replay_refuses_missing_dense_a_matrix() {
    let h = Harness::new();
    let st = h.interior_alpha_state();
    let (_bounds, mut inter) = h.fold_with_intermediates(&st);
    inter.a_at_relu.remove("b1r1");
    let err = h
        .graph
        .binding_row_true_alpha_grads(&h.input, &st, &inter, 0)
        .expect_err("must refuse partial intermediates");
    assert!(
        err.to_string().contains("b1r1"),
        "error must name the offending node: {err}"
    );
}

// ===================== resnet_medium-scale timing =====================

/// Conv builder with fan-in scaling so a 16-conv trunk keeps activations and
/// IBP boxes finite (the test_support builder's fixed 0.35 amplitude
/// overflows f32 at 128-channel depth).
fn scaled_conv(
    rng: &mut impl FnMut() -> f32,
    name: &str,
    input: &str,
    (ic, oc): (usize, usize),
    k: usize,
    s: usize,
    p: usize,
) -> GraphNode {
    let scale = 0.5 / ((ic * k * k) as f32).sqrt();
    let kernel = Array::from_shape_vec(
        IxDyn(&[oc, ic, k, k]),
        (0..oc * ic * k * k).map(|_| rng() * scale).collect(),
    )
    .expect("kernel");
    let bias = Array1::from_shape_fn(oc, |_| rng() * 0.05);
    let layer = Layer::Conv2d(
        crate::layers::Conv2dLayer::new(kernel, Some(bias), (s, s), (p, p)).expect("conv"),
    );
    if input == NETWORK_INPUT {
        GraphNode::from_input(name, layer)
    } else {
        GraphNode::new(name, layer, vec![input.to_string()])
    }
}

/// Synthetic clone of `CIFAR100_resnet_medium.onnx` topology at REAL widths
/// (BN folded into conv bias): input [3,32,32];
/// conv0 3->64 k3 s2 p0 (64x15x15) -> relu_a;
/// conv1 64->128 k3 s2 p1 (128x8x8) -> relu_b;
/// stage-entry projection add (conv from relu_b + 1x1 s2 shortcut from relu_a);
/// 3 identity blocks at 128x8x8; stride-2 projection block to 128x4x4;
/// 3 identity blocks at 128x4x4; flatten -> gemm 2048->100 -> relu -> gemm 100->100.
/// 10 ReLUs, 55,460 ReLU neurons, 16 convs — matching the onnx dump.
fn resnet_medium_scale_fixture() -> (GraphNetwork, Vec<&'static str>) {
    let mut rng = lcg(0x0005_CA1E_C1FA_9100);
    let mut g = GraphNetwork::new();
    g.add_node(scaled_conv(
        &mut rng,
        "conv0",
        NETWORK_INPUT,
        (3, 64),
        3,
        2,
        0,
    ));
    g.add_node(relu("relu_a", "conv0"));
    g.add_node(scaled_conv(&mut rng, "conv1", "relu_a", (64, 128), 3, 2, 1));
    g.add_node(relu("relu_b", "conv1"));
    // Stage-entry projection add (Add_10): conv(relu_b) + 1x1-s2 shortcut(relu_a).
    g.add_node(scaled_conv(&mut rng, "s1c", "relu_b", (128, 128), 3, 1, 1));
    g.add_node(scaled_conv(&mut rng, "s1p", "relu_a", (64, 128), 1, 2, 0));
    g.add_node(add("add10", "s1c", "s1p"));
    // Three identity blocks at 128x8x8 (Add_16/22/28).
    let mut prev = "add10".to_string();
    let mut relu_names: Vec<&'static str> = vec!["relu_a", "relu_b"];
    for (bi, rn) in [(1usize, "relu_c"), (2, "relu_d"), (3, "relu_e")] {
        let c1 = format!("b{bi}c1");
        let c2 = format!("b{bi}c2");
        let an = format!("badd{bi}");
        g.add_node(scaled_conv(&mut rng, &c1, &prev, (128, 128), 3, 1, 1));
        g.add_node(relu(rn, &c1));
        g.add_node(scaled_conv(&mut rng, &c2, rn, (128, 128), 3, 1, 1));
        g.add_node(add(&an, &c2, &prev));
        relu_names.push(rn);
        prev = an;
    }
    // Stride-2 projection block to 128x4x4 (Add_36).
    g.add_node(scaled_conv(&mut rng, "s2c1", &prev, (128, 128), 3, 2, 1));
    g.add_node(relu("relu_f", "s2c1"));
    g.add_node(scaled_conv(&mut rng, "s2c2", "relu_f", (128, 128), 3, 1, 1));
    g.add_node(scaled_conv(&mut rng, "s2p", &prev, (128, 128), 1, 2, 0));
    g.add_node(add("add36", "s2c2", "s2p"));
    relu_names.push("relu_f");
    // Three identity blocks at 128x4x4 (Add_42/48/54).
    prev = "add36".to_string();
    for (bi, rn) in [(4usize, "relu_g"), (5, "relu_h"), (6, "relu_i")] {
        let c1 = format!("b{bi}c1");
        let c2 = format!("b{bi}c2");
        let an = format!("badd{bi}");
        g.add_node(scaled_conv(&mut rng, &c1, &prev, (128, 128), 3, 1, 1));
        g.add_node(relu(rn, &c1));
        g.add_node(scaled_conv(&mut rng, &c2, rn, (128, 128), 3, 1, 1));
        g.add_node(add(&an, &c2, &prev));
        relu_names.push(rn);
        prev = an;
    }
    // Head: flatten -> gemm 2048->100 -> relu -> gemm 100->100.
    g.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::new(0)),
        vec![prev],
    ));
    let w1 = Array2::from_shape_fn((100, 2048), |_| rng() * 0.02);
    let b1 = Array1::from_shape_fn(100, |_| rng() * 0.05);
    g.add_node(GraphNode::new(
        "fc1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("fc1")),
        vec!["flatten".to_string()],
    ));
    g.add_node(relu("relu_j", "fc1"));
    relu_names.push("relu_j");
    let w2 = Array2::from_shape_fn((100, 100), |_| rng() * 0.1);
    let b2 = Array1::from_shape_fn(100, |_| rng() * 0.05);
    g.add_node(GraphNode::new(
        "fc2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("fc2")),
        vec!["relu_j".to_string()],
    ));
    g.set_output("fc2");
    (g, relu_names)
}

/// GATE (b): p95 replay latency per binding row at cifar100 resnet_medium
/// scale — 20-rep manual timer, MEASURED numbers printed for the consult #6
/// decision. The intermediate state (dense a_at_relu rows, final_bounds,
/// pre-ReLU bounds) is synthesized at the REAL shapes: the replay's cost is a
/// function of shapes, not values, and synthesizing skips a multi-minute
/// dense CPU backward in the debug-build test environment.
#[ntest::timeout(600000)]
#[test]
fn replay_p95_latency_at_resnet_medium_scale() {
    let (graph, relu_names) = resnet_medium_scale_fixture();
    let input = box_input(&[3, 32, 32], 0.3, 0.3 + 2.0 * 0.0039); // eps 0.0039 box
    let t0 = Instant::now();
    let node_bounds = graph.collect_node_bounds(&input).expect("IBP forward");
    let ibp_ms = t0.elapsed().as_secs_f64() * 1e3;

    let seed_rows = 100usize;
    let input_dim = 3 * 32 * 32;
    let mut rng = lcg(0xD00D_F00D_5EED);

    // Alpha state + synthesized intermediates at real widths.
    let mut st = GraphAlphaState::new();
    let mut inter = GraphAlphaCrownIntermediate::new();
    let mut total_neurons = 0usize;
    for name in &relu_names {
        let pre = graph
            .relu_preactivation_bounds(name, &input, &node_bounds, "brr-scale")
            .expect("pre bounds");
        st.add_relu_node(name, pre, false).expect("alpha init");
        let alpha = st.alphas.get_mut(*name).expect("alpha");
        for a in alpha.iter_mut() {
            *a = 0.5 + 0.3 * rng();
        }
        let n = pre.len();
        total_neurons += n;
        let flat = pre.flatten();
        let pl = Array1::from_iter(flat.lower().iter().copied());
        let pu = Array1::from_iter(flat.upper().iter().copied());
        inter.pre_relu_bounds.insert(name.to_string(), (pl, pu));
        inter.a_at_relu.insert(
            name.to_string(),
            Array2::from_shape_fn((seed_rows, n), |_| rng() * 0.1),
        );
    }
    eprintln!(
        "[brr-scale] MEASURED setup: IBP collect {ibp_ms:.1} ms, {} relus, {total_neurons} relu neurons",
        relu_names.len()
    );
    assert_eq!(total_neurons, 55_460, "real resnet_medium ReLU census");

    inter.final_bounds = LinearBounds::new(
        Array2::from_shape_fn((seed_rows, input_dim), |_| rng()),
        Array1::zeros(seed_rows),
        Array2::from_shape_fn((seed_rows, input_dim), |_| rng()),
        Array1::zeros(seed_rows),
    )
    .expect("final bounds");

    // Warmup (allocator, caches), then 20 timed reps over varying rows.
    for row in 0..2 {
        let r = graph
            .binding_row_true_alpha_grads(&input, &st, &inter, row)
            .expect("replay");
        assert!(r.grads.values().all(|g| g.iter().all(|v| v.is_finite())));
    }
    let reps = 20usize;
    let mut times_ms: Vec<f64> = Vec::with_capacity(reps);
    let mut nonzero_seen = false;
    for rep in 0..reps {
        let row = (rep * 5) % seed_rows;
        let t = Instant::now();
        let r = graph
            .binding_row_true_alpha_grads(&input, &st, &inter, row)
            .expect("replay");
        times_ms.push(t.elapsed().as_secs_f64() * 1e3);
        nonzero_seen |= r.grads.values().any(|g| g.iter().any(|v| *v != 0.0));
        assert_eq!(r.grads.len(), relu_names.len());
    }
    assert!(nonzero_seen, "scale replay must produce live gradients");
    times_ms.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let p50 = times_ms[reps / 2];
    let p95 = times_ms[(reps * 95).div_ceil(100) - 1];
    eprintln!(
        "[brr-scale] MEASURED replay per binding row over {reps} reps: p50={p50:.1} ms p95={p95:.1} ms min={:.1} max={:.1} (build profile: {})",
        times_ms[0],
        times_ms[reps - 1],
        if cfg!(debug_assertions) { "debug" } else { "release" },
    );
    // Collapse tripwire only — the consult #6 <100 ms decision is made from
    // the MEASURED release-profile print, not asserted here (debug-profile CI
    // would flake a hard 100 ms gate).
    assert!(
        p95 < 30_000.0,
        "#binding-row-replay: p95 {p95:.1} ms is collapse-class even for a debug build"
    );
}

// ===================== ADVERSARIAL VERIFY (independent) =====================
// Independent verifier tests — OWN fixture with C, H, W pairwise distinct
// (3x5x7 and 4x5x7) so a stride/axis confusion cannot cancel against the
// production fixture's H == W == 6.

struct VerifyHarness {
    graph: GraphNetwork,
    input: BoundedTensor,
    node_bounds: HashMap<String, BoundedTensor>,
    exec_order: Vec<String>,
    relu_name_to_idx: HashMap<String, usize>,
    output_dim: usize,
    input_dim: usize,
}

const VERIFY_RELUS: [&str; 2] = ["vr0", "vr1"];

impl VerifyHarness {
    fn new() -> Self {
        let mut rng = lcg(0xAD7E_2026_0801);
        let mut g = GraphNetwork::new();
        // input [2,5,7] -> conv -> [3,5,7] (C=3,H=5,W=7 pairwise distinct)
        g.add_node(conv(&mut rng, "vc0", NETWORK_INPUT, (2, 3), 3, 1, 1, true));
        g.add_node(relu("vr0", "vc0"));
        // [3,5,7] -> conv -> [4,5,7]
        g.add_node(conv(&mut rng, "vc1", "vr0", (3, 4), 3, 1, 1, true));
        g.add_node(relu("vr1", "vc1"));
        g.add_node(GraphNode::new(
            "vflat",
            Layer::Flatten(FlattenLayer::new(0)),
            vec!["vr1".to_string()],
        ));
        let in_features = 4 * 5 * 7;
        let out_rows = 6;
        let w = Array2::from_shape_fn((out_rows, in_features), |_| rng() * 0.2);
        let b = Array1::from_shape_fn(out_rows, |_| rng() * 0.1);
        g.add_node(GraphNode::new(
            "vm",
            Layer::Linear(LinearLayer::new(w, Some(b)).expect("margin gemm")),
            vec!["vflat".to_string()],
        ));
        g.set_output("vm");
        let input = box_input(&[2, 5, 7], -0.5, 0.5);
        let node_bounds = g.collect_node_bounds(&input).expect("IBP forward");
        let exec_order = g.exec_order().expect("exec order").to_vec();
        let relu_name_to_idx: HashMap<String, usize> = VERIFY_RELUS
            .iter()
            .enumerate()
            .map(|(i, n)| (n.to_string(), i))
            .collect();
        Self {
            graph: g,
            input,
            node_bounds,
            exec_order,
            relu_name_to_idx,
            output_dim: 6,
            input_dim: 2 * 5 * 7,
        }
    }

    fn channel_alpha(&self) -> GraphAlphaState {
        let mut st = GraphAlphaState::new();
        let mut rng = lcg(0x5EED_ADE5_A170);
        for name in VERIFY_RELUS {
            let pre = self
                .graph
                .relu_preactivation_bounds(name, &self.input, &self.node_bounds, "verify")
                .expect("pre bounds");
            st.add_relu_node(name, pre, true).expect("alpha init");
            let shape = st.spatial_shapes.get(name).expect("channel path").clone();
            assert_eq!(shape.len(), 3, "conv geometry [C,H,W]");
            let alpha = st.alphas.get_mut(name).expect("lower alpha");
            assert_eq!(alpha.len(), shape[0], "alpha is length C");
            for a in alpha.iter_mut() {
                *a = 0.5 + 0.25 * rng(); // interior (0.25, 0.75)
            }
        }
        assert_eq!(st.spatial_shapes["vr0"], vec![3, 5, 7]);
        assert_eq!(st.spatial_shapes["vr1"], vec![4, 5, 7]);
        st
    }

    fn zero_grads(&self) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let g: Vec<Array1<f32>> = VERIFY_RELUS
            .iter()
            .map(|name| {
                let pre = self
                    .graph
                    .relu_preactivation_bounds(name, &self.input, &self.node_bounds, "verify")
                    .expect("pre bounds");
                Array1::zeros(pre.len())
            })
            .collect();
        let gu = g.iter().map(|v| Array1::zeros(v.len())).collect();
        (g, gu)
    }

    fn fold_lower(&self, st: &GraphAlphaState, row: usize) -> f32 {
        let (mut g, mut gu) = self.zero_grads();
        let bounds = self
            .graph
            .dag_alpha_backward_pass_with_engine(
                &self.input,
                &self.node_bounds,
                &self.exec_order,
                self.output_dim,
                self.input_dim,
                &self.relu_name_to_idx,
                st,
                None,
                &mut g,
                &mut gu,
                None,
                None,
                None,
                None,
            )
            .expect("alpha fold");
        bounds.lower()[[row]]
    }

    fn fold_with_intermediates(
        &self,
        st: &GraphAlphaState,
    ) -> (BoundedTensor, GraphAlphaCrownIntermediate) {
        let (mut g, mut gu) = self.zero_grads();
        self.graph
            .dag_alpha_backward_pass_with_intermediates(
                &self.input,
                &self.node_bounds,
                &self.exec_order,
                self.output_dim,
                self.input_dim,
                &self.relu_name_to_idx,
                st,
                None,
                &mut g,
                &mut gu,
                None,
                None,
                None,
                None,
            )
            .expect("alpha fold with intermediates")
    }

    fn fd(&self, st: &GraphAlphaState, relu: &str, c: usize, row: usize, h: f32) -> f64 {
        let base = st.alphas[relu][c];
        assert!(base - h > 0.0 && base + h < 1.0, "interior probe");
        let mut plus = st.clone();
        plus.alphas.get_mut(relu).expect("alpha")[c] = base + h;
        let mut minus = st.clone();
        minus.alphas.get_mut(relu).expect("alpha")[c] = base - h;
        f64::from(self.fold_lower(&plus, row) - self.fold_lower(&minus, row)) / (2.0 * f64::from(h))
    }
}

/// VERIFIER attack 1 (wrong-reduction / stride confusion): on a fixture with
/// C,H,W pairwise distinct, central FD of the certified fold per alpha_c must
/// match the replay's channel gradient, the channel-major (i/spatial)
/// reduction of the per-position contributions must equal the replay grads,
/// and the channel-MINOR (i % C) mis-reduction must measurably disagree with
/// FD — proving the test could have caught a stride bug.
#[ntest::timeout(300000)]
#[test]
fn verify_channel_reduction_is_true_derivative_on_distinct_dims_fixture() {
    let h = VerifyHarness::new();
    let st = h.channel_alpha();

    let (bounds, inter) = h.fold_with_intermediates(&st);
    assert!(bounds.lower().iter().all(|v| v.is_finite()));
    let binding_row = bounds
        .lower()
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
        .map(|(i, _)| i)
        .expect("nonempty");

    let replay = h
        .graph
        .binding_row_true_alpha_grads(&h.input, &st, &inter, binding_row)
        .expect("channel replay");

    let mut fd_all: Vec<f64> = Vec::new();
    let mut true_all: Vec<f64> = Vec::new();
    let mut wrong_all: Vec<f64> = Vec::new();
    for name in VERIFY_RELUS {
        let shape = &st.spatial_shapes[name];
        let (channels, spatial) = (shape[0], shape[1] * shape[2]);
        let n = channels * spatial;
        let g = replay.grads.get(name).expect("replay grad");
        assert_eq!(
            g.len(),
            channels,
            "replay grad at ALPHA width C at '{name}'"
        );

        // Recompute per-position contributions p_i = nu_i * z_i under the
        // combine's masks, then reduce with BOTH conventions.
        let (pre_l, pre_u) = inter.pre_relu_bounds(name).expect("pre bounds");
        let a_mat = inter.a_at_relu(name).expect("dense a_at_relu");
        let z = replay.hhat.get(name).expect("hhat z values");
        assert_eq!(z.len(), n);
        let mut right = vec![0.0f64; channels];
        let mut wrong = vec![0.0f64; channels];
        for i in 0..n {
            let (l, u) = (pre_l[i], pre_u[i]);
            if !l.is_finite() || !u.is_finite() || l >= 0.0 || u <= 0.0 {
                continue;
            }
            let nu = a_mat[[binding_row, i]];
            if !nu.is_finite() || nu <= 0.0 {
                continue;
            }
            let p = f64::from(nu) * z[i];
            right[i / spatial] += p; // channel-major NCHW (the fix's claim)
            wrong[i % channels] += p; // channel-minor (the stride bug)
        }
        for c in 0..channels {
            let fd = h.fd(&st, name, c, binding_row, 5e-3);
            fd_all.push(fd);
            true_all.push(f64::from(g[c]));
            wrong_all.push(wrong[c]);
            // replay grad must BE the channel-major reduction (same masks).
            assert!(
                (f64::from(g[c]) - right[c]).abs() <= 1e-4 + 1e-3 * right[c].abs(),
                "'{name}'[c={c}]: replay {} vs recomputed channel-major {}",
                g[c],
                right[c]
            );
        }
    }
    let n = fd_all.len();
    assert_eq!(n, 7, "3 + 4 channels");
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let nrm = |a: &[f64]| dot(a, a).sqrt();
    let cos_right = dot(&fd_all, &true_all) / (nrm(&fd_all) * nrm(&true_all));
    let rel = |a: &[f64], b: &[f64]| {
        (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>()
            / b.iter().map(|y| y * y).sum::<f64>().max(1e-30))
        .sqrt()
    };
    let rel_right = rel(&true_all, &fd_all);
    let rel_wrong = rel(&wrong_all, &fd_all);
    let rel_wr = rel(&wrong_all, &true_all);
    eprintln!(
        "[verify-ch] MEASURED n={n} cosine(fd,replay)={cos_right:.6} relL2(replay,fd)={rel_right:.3e} \
         relL2(wrongstride,fd)={rel_wrong:.3e} relL2(wrongstride,replay)={rel_wr:.3e}"
    );
    assert!(cos_right > 0.999, "cosine {cos_right} <= 0.999");
    assert!(rel_right < 0.05, "replay vs FD rel L2 {rel_right} >= 0.05");
    assert!(
        rel_wr > 0.05,
        "fixture degenerate: wrong-stride reduction indistinguishable ({rel_wr})"
    );
    assert!(
        rel_wrong > 10.0 * rel_right.max(1e-6),
        "wrong-stride reduction is not rejected by FD: {rel_wrong} vs {rel_right}"
    );
}

/// VERIFIER attack 2 (shape keying): divisibility must NOT trigger reduction.
/// The key must be the node's recorded channel count AND the stored alpha
/// vector's width — alpha widths that merely divide the gradient length
/// (including the spatial count itself) must refuse.
#[ntest::timeout(300000)]
#[test]
fn verify_shape_keying_rejects_divisible_but_wrong_widths() {
    let h = VerifyHarness::new();
    let st = h.channel_alpha();
    // vr0: C=3, spatial=35, n=105. Divisors of 105 that are NOT C: 5, 7, 15,
    // 21, 35 (=spatial), 105 (identity).
    assert_eq!(st.channel_reduction_geometry("vr0", 3, 105), Some((3, 35)));
    for bad in [1usize, 5, 7, 15, 21, 35, 105] {
        assert_eq!(
            st.channel_reduction_geometry("vr0", bad, 105),
            None,
            "alpha_len {bad} divides 105 but must refuse (C=3)"
        );
    }
    // grad_len divisible by C but not C*spatial must refuse (210 = 2*105).
    assert_eq!(st.channel_reduction_geometry("vr0", 3, 210), None);
    assert_eq!(st.channel_reduction_geometry("vr0", 3, 3), None, "identity");
    // vr1: C=4, n=140. alpha_len 2 divides 140 and divides C — the exact
    // attack shape (alpha_len 2, C=4, spatial even in total) must refuse.
    assert_eq!(st.channel_reduction_geometry("vr1", 4, 140), Some((4, 35)));
    assert_eq!(st.channel_reduction_geometry("vr1", 2, 140), None);
    // Stored-alpha key: same lengths, but the stored alpha vector is expanded
    // to per-neuron width -> must refuse even though 3 == shape[0].
    let mut tampered = st.clone();
    tampered
        .alphas
        .insert("vr0".to_string(), Array1::from_elem(105, 0.5));
    assert_eq!(tampered.channel_reduction_geometry("vr0", 3, 105), None);
    // Tampered channel count in recorded geometry: 3 divides 105 but the
    // recorded shape now says C=5 -> refuse.
    let mut tampered2 = st.clone();
    tampered2
        .spatial_shapes
        .insert("vr0".to_string(), vec![5, 3, 7]);
    assert_eq!(tampered2.channel_reduction_geometry("vr0", 3, 105), None);

    // End-to-end replay refusal: stored alpha width 5 (divides 105) with
    // intact geometry must be a typed error naming the node, not a guess.
    let (_bounds, inter) = h.fold_with_intermediates(&st);
    let mut bad_st = st;
    bad_st
        .alphas
        .insert("vr0".to_string(), Array1::from_elem(5, 0.5));
    let err = h
        .graph
        .binding_row_true_alpha_grads(&h.input, &bad_st, &inter, 0)
        .expect_err("divisible-but-wrong alpha width must refuse");
    let msg = err.to_string();
    assert!(msg.contains("vr0"), "error must name the node: {msg}");
}
