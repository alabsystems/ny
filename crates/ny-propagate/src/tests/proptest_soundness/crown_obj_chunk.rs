// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness tests for the two conv-CROWN improvements (#patches-obj-chunk,
//! #patches-backward-oom):
//!
//! 1. PRIMARY — objective-chunking streaming (`NY_CROWN_OBJ_CHUNK`):
//!    - equivalence: chunked (C = 1 and 7) vs single-pass concretized bounds.
//!      Widening-only is OK (`streamed_lower <= single_lower + tol` AND
//!      `streamed_upper >= single_upper - tol`); narrowing is a FAIL.
//!    - containment: both streams contain the true conv output on ~20 sampled
//!      in-box points.
//! 2. SECONDARY — re-route mid-size conv targets to EXACT patches: a target
//!    whose identity pair fits the budget but whose backward pair does not now
//!    reports `is_patches_target = true`, and its CROWN bound matches the dense
//!    `to_dense()` bound (NOT the loose IBP fallback).
//!
//! Helpers (`make_kernel`, `make_bias`) are cloned from `crown_patches.rs`.

use crate::layers::{Conv2dLayer, ReLULayer};
use crate::network::{GraphNetwork, GraphNode};
use crate::tests::with_serialized_env_vars;
use crate::Layer;
use ndarray::{Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;
use std::collections::HashMap;

/// Generate a random kernel of shape (out_c, in_c, kh, kw) with values in
/// [-2.0, 2.0], using a seed for reproducibility. (Cloned from crown_patches.rs.)
fn make_kernel(out_c: usize, in_c: usize, kh: usize, kw: usize, seed: u64) -> ArrayD<f32> {
    let len = out_c * in_c * kh * kw;
    let mut rng = seed;
    let values: Vec<f32> = (0..len)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 4.0 - 2.0
        })
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), values).expect("kernel shape mismatch")
}

/// Generate a random bias of shape (out_c,) with values in [-1.0, 1.0].
/// (Cloned from crown_patches.rs.)
fn make_bias(out_c: usize, seed: u64) -> Array1<f32> {
    let mut rng = seed.wrapping_add(12345);
    let values: Vec<f32> = (0..out_c)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 2.0 - 1.0
        })
        .collect();
    Array1::from_vec(values)
}

/// Build a Conv2d -> ReLU graph in patches mode (so a spatial conv target is
/// patches-eligible). Returns the graph and the conv output spatial shape.
fn build_conv_relu_graph(conv: Conv2dLayer, use_patches: bool) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".into()],
    ));
    graph.set_output("relu");
    graph.set_use_patches_mode(use_patches);
    graph
}

/// Run CROWN-IBP backward to the output node under a specific objective-chunk
/// setting. `chunk` of `None` runs the single pass (env unset); `Some(c)` sets
/// `NY_CROWN_OBJ_CHUNK = c`. `NY_DENSE_BUDGET_MB` is pinned so the budget gate
/// behaves identically across all runs in a comparison.
fn crown_bounds_with_chunk(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    budget_mb: &str,
    chunk: Option<usize>,
) -> BoundedTensor {
    let chunk_val = chunk.map(|c| c.to_string());
    let vars: Vec<(&'static str, &str)> = match &chunk_val {
        Some(v) => vec![
            ("NY_DENSE_BUDGET_MB", budget_mb),
            ("NY_CROWN_OBJ_CHUNK", v.as_str()),
        ],
        // Explicitly pin chunk to "0" (disabled) for the single-pass baseline so
        // a stray env value from another test cannot leak in.
        None => vec![
            ("NY_DENSE_BUDGET_MB", budget_mb),
            ("NY_CROWN_OBJ_CHUNK", "0"),
        ],
    };
    with_serialized_env_vars(&vars, || {
        graph
            .propagate_crown_to_node(
                input,
                graph.output_name(),
                &HashMap::new(),
                ibp_bounds,
                None,
                None,
                None,
                None,
            )
            .expect("propagate_crown_to_node failed")
    })
}

/// Build input bounds: random centers in [-1, 1), width 0.2 per element.
fn make_input_box(
    shape: &[usize],
    in_dim: usize,
    seed: u64,
) -> (BoundedTensor, Vec<f32>, Vec<f32>) {
    let lower_vals: Vec<f32> = (0..in_dim)
        .map(|i| {
            let mut rng = seed.wrapping_add(i as u64 + 99999);
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 2.0 - 1.0
        })
        .collect();
    let upper_vals: Vec<f32> = lower_vals.iter().map(|&l| l + 0.2).collect();
    let lower_nd = ArrayD::from_shape_vec(IxDyn(shape), lower_vals.clone()).unwrap();
    let upper_nd = ArrayD::from_shape_vec(IxDyn(shape), upper_vals.clone()).unwrap();
    let bt = BoundedTensor::new(lower_nd, upper_nd).expect("input box");
    (bt, lower_vals, upper_vals)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(120) })]

    /// PRIMARY equivalence + containment: chunked (C = 1, 7) vs single-pass.
    ///
    /// Equivalence (widening-only is OK, narrowing is a FAIL):
    ///   streamed_lower <= single_lower + tol AND streamed_upper >= single_upper - tol.
    /// Containment: both streams contain the true conv->relu output on 20
    /// sampled in-box points.
    #[ntest::timeout(30000)]
    #[test]
    fn proptest_obj_chunk_equivalence_and_containment(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        kh in 1usize..=3,
        kw in 1usize..=3,
        // Keep every generated output at least 2x2, so C=1 actually streams
        // more than one objective without rejecting cases at runtime.
        in_h in 5usize..=8,
        in_w in 5usize..=8,
        stride_h in 1usize..=2,
        stride_w in 1usize..=2,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        use_bias in proptest::bool::ANY,
        use_patches in proptest::bool::ANY,
        seed in any::<u64>(),
    ) {
        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;
        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;

        let in_dim = in_c * in_h * in_w;
        debug_assert!(out_c * out_h * out_w >= 4);

        let kernel = make_kernel(out_c, in_c, kh, kw, seed);
        let bias = if use_bias { Some(make_bias(out_c, seed)) } else { None };
        let conv = Conv2dLayer::with_input_shape(
            kernel, bias, (stride_h, stride_w), (pad_h, pad_w), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!(
            "generated valid Conv2d configuration was rejected: {e}"
        )))?;
        let graph = build_conv_relu_graph(conv, use_patches);

        let in_shape = [in_c, in_h, in_w];
        let (input_bt, lower_vals, upper_vals) = make_input_box(&in_shape, in_dim, seed);
        let ibp_bounds = graph.collect_node_bounds(&input_bt)
            .map_err(|e| TestCaseError::fail(format!("collect_node_bounds failed: {e}")))?;

        // Use a large budget so chunking is the only variable under test.
        let single = crown_bounds_with_chunk(&graph, &input_bt, &ibp_bounds, "4096", None);
        let chunk1 = crown_bounds_with_chunk(&graph, &input_bt, &ibp_bounds, "4096", Some(1));
        let chunk7 = crown_bounds_with_chunk(&graph, &input_bt, &ibp_bounds, "4096", Some(7));

        let single_l = single.lower();
        let single_u = single.upper();

        // tol 1e-5 single op (this is a single conv-backward + ReLU pass).
        let tol = 1e-5_f32;
        for (label, streamed) in [("C=1", &chunk1), ("C=7", &chunk7)] {
            prop_assert_eq!(
                streamed.shape(), single.shape(),
                "{}: shape mismatch streamed={:?} single={:?}",
                label, streamed.shape(), single.shape()
            );
            let sl = streamed.lower();
            let su = streamed.upper();
            for (idx, &one_lo) in single_l.indexed_iter() {
                let one_up = single_u[&idx];
                let s_lo = sl[&idx];
                let s_up = su[&idx];
                let scale = one_lo.abs().max(one_up.abs()).max(1.0);
                // Widening-only: streamed lower may not exceed single lower;
                // streamed upper may not fall below single upper.
                prop_assert!(
                    s_lo <= one_lo + tol * scale,
                    "{} NARROWED lower at {:?}: streamed={} > single={}",
                    label, idx, s_lo, one_lo,
                );
                prop_assert!(
                    s_up >= one_up - tol * scale,
                    "{} NARROWED upper at {:?}: streamed={} < single={}",
                    label, idx, s_up, one_up,
                );
            }
        }

        // Containment: 20 sampled in-box points; both streams (and the single
        // pass) must contain the true conv->relu output at every coordinate.
        let containment_tol = 1e-4_f32;
        for s in 0..20 {
            let sample_vals: Vec<f32> = lower_vals.iter().zip(upper_vals.iter()).enumerate()
                .map(|(i, (&l, &u))| {
                    let t = ((s as f32 * 0.618_034) + (i as f32 * 0.414_213)) % 1.0;
                    l + (u - l) * t
                })
                .collect();
            let sample_nd = ArrayD::from_shape_vec(IxDyn(&in_shape), sample_vals).unwrap();
            let sample_pt = BoundedTensor::new(sample_nd.clone(), sample_nd).unwrap();
            let true_out = graph.collect_node_bounds(&sample_pt)
                .map_err(|e| TestCaseError::fail(format!("eval point failed: {e}")))?;
            let true_relu = true_out.get("relu").unwrap();
            let tl = true_relu.lower();

            for (which, bounds) in [("single", &single), ("C=1", &chunk1), ("C=7", &chunk7)] {
                let bl = bounds.lower();
                let bu = bounds.upper();
                for (idx, &tv) in tl.indexed_iter() {
                    prop_assert!(
                        bl[&idx] <= tv + containment_tol,
                        "{} lower bound violation at {:?}: bound={} > true={}",
                        which, idx, bl[&idx], tv,
                    );
                    prop_assert!(
                        bu[&idx] >= tv - containment_tol,
                        "{} upper bound violation at {:?}: bound={} < true={}",
                        which, idx, bu[&idx], tv,
                    );
                }
            }
        }
    }
}

/// SECONDARY (#patches-backward-oom): a mid-size conv target whose identity pair
/// fits the budget but whose [target_dim x conv_in_size] BACKWARD pair does not
/// now reports `is_patches_target = true`, and its CROWN bound matches the dense
/// `to_dense()` bound (NOT the loose IBP fallback).
///
/// Sizing (NY_DENSE_BUDGET_MB = 1 => budget = 1,048,576 bytes):
///   input  (4, 20, 25)  -> conv_in_size = 4*20*25 = 2000
///   conv 4->1ch, 2x2, stride 2, pad 0 -> output (1, 10, 12), target_dim = 120
///   identity pair = 2 * 120^2 * 4 =   115,200 bytes  (< 1 MiB, FITS)
///   backward pair = 2 * 120*2000 * 4 = 1,920,000 bytes (> 1 MiB, EXCEEDS)
#[test]
fn secondary_midsize_conv_target_routes_to_exact_patches() {
    // Network: conv1 -> relu -> conv2, OUTPUT = conv2.
    //   conv1: input (4, 20, 25) [in_size 2000] -> 4->2ch, 2x2, stride 2
    //          -> (2, 10, 12)
    //   relu over (2, 10, 12)
    //   conv2: (2, 10, 12) -> 2->1ch, 3x3, stride 1, pad 0 -> (1, 8, 10) = 80
    //
    // The deepest Conv2d ancestor of the conv2 target is conv1, so the gated
    // backward pair is [conv2_out_dim(80) x conv1_in_size(2000)]. With the ReLU
    // between them, dense CROWN is strictly tighter than IBP at conv2 — so a
    // patches bound that MATCHES dense CROWN cannot be the loose IBP fallback.
    let in_c = 4usize;
    let in_h = 20usize;
    let in_w = 25usize;
    let in_dim = in_c * in_h * in_w; // 2000

    let k1 = make_kernel(2, in_c, 2, 2, 0xC0FFEE);
    let b1 = Some(make_bias(2, 0xC0FFEE));
    let conv1 =
        Conv2dLayer::with_input_shape(k1, b1, (2, 2), (0, 0), in_h, in_w).expect("valid conv1");
    let (mid_c, mid_h, mid_w) = (2usize, 10usize, 12usize);

    let k2 = make_kernel(1, mid_c, 3, 3, 0xBEEF);
    let b2 = Some(make_bias(1, 0xBEEF));
    let conv2 =
        Conv2dLayer::with_input_shape(k2, b2, (1, 1), (0, 0), mid_h, mid_w).expect("valid conv2");
    let (out_c, out_h, out_w) = (1usize, 8usize, 10usize);
    let out_dim = out_c * out_h * out_w; // 80

    // Sanity-check the byte sizing the test relies on (budget = 1 MiB).
    let budget_bytes = 1024 * 1024;
    let identity_pair = 2 * out_dim * out_dim * 4; // [80 x 80]
    let backward_pair = 2 * out_dim * in_dim * 4; // [80 x 2000]
    assert!(
        identity_pair < budget_bytes,
        "identity pair {identity_pair} must fit the 1 MiB budget"
    );
    assert!(
        backward_pair > budget_bytes,
        "backward pair {backward_pair} must exceed the 1 MiB budget"
    );

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv1".into()],
    ));
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv2),
        vec!["relu".into()],
    ));
    graph.set_output("conv2");
    graph.set_use_patches_mode(true);
    // Silence the unused mid-shape warning while keeping the comment accurate.
    let _ = (mid_c, mid_h, mid_w);

    let in_shape = [in_c, in_h, in_w];
    // Symmetric box [-0.5, 0.5] per element: conv1 outputs straddle zero, so the
    // ReLU has genuinely unstable neurons and CROWN beats IBP at conv2.
    let lower_vals = vec![-0.5_f32; in_dim];
    let upper_vals = vec![0.5_f32; in_dim];
    let input_bt = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&in_shape), lower_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&in_shape), upper_vals.clone()).unwrap(),
    )
    .expect("symmetric input box");

    let ibp_bounds = graph.collect_node_bounds(&input_bt).expect("ibp bounds");
    let target_ibp = ibp_bounds.get("conv2").expect("conv2 ibp bounds");

    // 1. With budget = 1 MiB, the predicate must classify the conv2 target as a
    //    patches target (the [80 x 2000] backward pair exceeds budget even though
    //    the [80 x 80] identity fits) — this is exactly the #patches-backward-oom
    //    re-route.
    let is_patches_target = with_serialized_env_vars(&[("NY_DENSE_BUDGET_MB", "1")], || {
        graph.crown_ibp_target_can_start_in_patches_for_test("conv2", target_ibp)
    });
    assert!(
        is_patches_target,
        "#patches-backward-oom: mid-size conv target (identity fits, backward pair exceeds) \
         must be classified as a patches target"
    );

    // With the OLD identity-only predicate this target would NOT be patches-
    // eligible (identity pair fits the budget). Confirm that distinction.
    let identity_only_would_skip = identity_pair < budget_bytes;
    assert!(
        identity_only_would_skip,
        "test precondition: identity pair must fit (so the old gate would skip patches)"
    );

    // 2. The patches-mode CROWN bound (budget = 1 MiB) must match the dense
    //    CROWN bound (huge budget, pure dense path) — proving it is the EXACT
    //    patches bound, NOT the loose IBP fallback.
    let patches_bound = with_serialized_env_vars(&[("NY_DENSE_BUDGET_MB", "1")], || {
        graph
            .propagate_crown_to_node(
                &input_bt,
                graph.output_name(),
                &HashMap::new(),
                &ibp_bounds,
                None,
                None,
                None,
                None,
            )
            .expect("patches CROWN failed")
    });
    let dense_bound = with_serialized_env_vars(&[("NY_DENSE_BUDGET_MB", "65536")], || {
        graph
            .propagate_crown_to_node(
                &input_bt,
                graph.output_name(),
                &HashMap::new(),
                &ibp_bounds,
                None,
                None,
                None,
                None,
            )
            .expect("dense CROWN failed")
    });

    let exact_tol = 1e-4_f32;
    let pl = patches_bound.lower();
    let pu = patches_bound.upper();
    let dl = dense_bound.lower();
    let du = dense_bound.upper();
    let il = target_ibp.lower();
    let iu = target_ibp.upper();

    let mut strictly_tighter_than_ibp = false;
    for (idx, &dlv) in dl.indexed_iter() {
        let duv = du[&idx];
        let plv = pl[&idx];
        let puv = pu[&idx];
        let scale = dlv.abs().max(duv.abs()).max(1.0);
        // Patches CROWN == dense CROWN (exact, within tol).
        assert!(
            (plv - dlv).abs() <= exact_tol * scale,
            "patches lower {plv} != dense lower {dlv} at {idx:?} (would be loose IBP)"
        );
        assert!(
            (puv - duv).abs() <= exact_tol * scale,
            "patches upper {puv} != dense upper {duv} at {idx:?} (would be loose IBP)"
        );
        // Patches CROWN is tighter-or-equal to the loose IBP interval; record
        // at least one strictly-tighter coordinate to prove it is NOT the IBP
        // fallback.
        let ilv = il[&idx];
        let iuv = iu[&idx];
        // CROWN must never be wider than IBP.
        assert!(
            plv >= ilv - exact_tol * scale && puv <= iuv + exact_tol * scale,
            "patches CROWN [{plv},{puv}] not contained in IBP [{ilv},{iuv}] at {idx:?}"
        );
        if plv > ilv + exact_tol * scale || puv < iuv - exact_tol * scale {
            strictly_tighter_than_ibp = true;
        }
    }
    assert!(
        strictly_tighter_than_ibp,
        "#patches-backward-oom: exact patches bound must be strictly tighter than loose IBP \
         on at least one coordinate (otherwise it is indistinguishable from the IBP fallback)"
    );

    // Containment of the true output at the box corners (defensive).
    let corner_lo = ArrayD::from_shape_vec(IxDyn(&in_shape), lower_vals).unwrap();
    let corner_hi = ArrayD::from_shape_vec(IxDyn(&in_shape), upper_vals).unwrap();
    for corner in [corner_lo, corner_hi] {
        let pt = BoundedTensor::new(corner.clone(), corner).unwrap();
        let out = graph.collect_node_bounds(&pt).unwrap();
        let conv2_out = out.get("conv2").unwrap();
        let tl = conv2_out.lower();
        for (idx, &tv) in tl.indexed_iter() {
            assert!(
                pl[&idx] <= tv + exact_tol && pu[&idx] >= tv - exact_tol,
                "patches bound [{},{}] does not contain true {} at {:?}",
                pl[&idx],
                pu[&idx],
                tv,
                idx
            );
        }
    }
}

/// #cgan-bn11-chunk: a target whose dense identity pair exceeds the CPU dense
/// budget — and that CANNOT start in patches (ConvTranspose2d ancestors only,
/// exactly the cgan_2023 BatchNormalization_11 shape) — must no longer degrade
/// to IBP in the CROWN-IBP collector. It reroutes through the auto-chunked
/// objective streaming backward and produces bounds EQUAL to the unbudgeted
/// single-pass CROWN bounds, strictly tighter than IBP.
///
/// Mini cgan generator chain (all affine before the pre-ReLU target, so CROWN
/// is the exact affine map and IBP is strictly looser):
///   lin (5 -> 128) -> reshape [2,8,8] -> bn_a -> convt (2->2, k2, s2)
///     -> bn_b [2,16,16] = 512 dims -> relu
///
/// Sizing (NY_DENSE_BUDGET_MB = 1 => budget = 1,048,576 bytes):
///   bn_b identity pair = 2 * 512^2 * 4 = 2,097,152 bytes  (> 1 MiB, EXCEEDS)
///   auto chunk C = dim * budget / pair = 512 / 2 = 256 rows
///   chunk seed pair = 2 * 256 * 512 * 4 = 1,048,576 bytes (== budget, FITS)
/// Build the mini cgan generator chain used by the over-budget chunk-reroute
/// tests: `lin (5->128) -> reshape [2,8,8] -> bn_a -> convt (2->2, k2, s2)
/// -> bn_b [2,16,16] = 512 dims -> relu`. All layers before the pre-ReLU `bn_b`
/// target are affine, so CROWN is the exact affine map and IBP is strictly
/// looser; `bn_b` has NO Conv2d ancestor (ConvTranspose2d only), so it cannot
/// start in patches — exactly the cgan_2023 BatchNormalization_11 shape.
fn build_mini_cgan_over_budget() -> (GraphNetwork, BoundedTensor) {
    use crate::layers::{BatchNormLayer, ConvTranspose2dLayer, LinearLayer, ReshapeLayer};
    use crate::{GraphNode, Layer};

    let in_dim = 5usize;
    let hidden = 128usize;
    let w = ndarray::Array2::from_shape_fn((hidden, in_dim), |(i, j)| {
        (((i * 7 + j * 3) % 11) as f32 * 0.21 - 1.0) * if (i + j) % 2 == 0 { 1.0 } else { -1.0 }
    });
    let lin = LinearLayer::new(w, None).expect("lin");
    let reshape = ReshapeLayer {
        target_shape: vec![2, 8, 8],
    };
    let kernel = ArrayD::from_shape_fn(IxDyn(&[2, 2, 2, 2]), |d| {
        (((d[0] * 5 + d[1] * 3 + d[2] * 2 + d[3]) % 7) as f32 * 0.33 - 1.0)
            * if (d[0] + d[1] + d[2] + d[3]) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
    });
    let convt = ConvTranspose2dLayer::new_full(kernel, None, (2, 2), (0, 0), (1, 1), (0, 0))
        .expect("convt");
    let mk_bn = |ch: usize, seed: usize| -> BatchNormLayer {
        let g = ArrayD::from_shape_fn(IxDyn(&[ch]), |d| 0.5 + ((d[0] * 3 + seed) % 5) as f32 * 0.3);
        let b = ArrayD::from_shape_fn(IxDyn(&[ch]), |d| ((d[0] + seed) % 3) as f32 * 0.1 - 0.1);
        let m = ArrayD::from_shape_fn(IxDyn(&[ch]), |d| ((d[0] * 2 + seed) % 4) as f32 * 0.2 - 0.3);
        let v = ArrayD::from_shape_fn(IxDyn(&[ch]), |d| 0.5 + ((d[0] + seed) % 3) as f32 * 0.4);
        BatchNormLayer::new(&g, &b, &m, &v, 1e-5).expect("bn")
    };

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("lin", Layer::Linear(lin)));
    graph.add_node(GraphNode::new(
        "reshape",
        Layer::Reshape(reshape),
        vec!["lin".into()],
    ));
    graph.add_node(GraphNode::new(
        "bn_a",
        Layer::BatchNorm(mk_bn(2, 1)),
        vec!["reshape".into()],
    ));
    graph.add_node(GraphNode::new(
        "convt",
        Layer::ConvTranspose2d(convt),
        vec!["bn_a".into()],
    ));
    graph.add_node(GraphNode::new(
        "bn_b",
        Layer::BatchNorm(mk_bn(2, 2)),
        vec!["convt".into()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["bn_b".into()],
    ));
    graph.set_output("relu");

    let lower = Array1::from_elem(in_dim, 0.3f32) - 0.01f32;
    let upper = Array1::from_elem(in_dim, 0.3f32) + 0.01f32;
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).expect("input box");
    (graph, input)
}

#[test]
fn auto_chunk_reroutes_over_budget_target_and_matches_single_pass() {
    use crate::types::BoundsProvenance;

    let (graph, input) = build_mini_cgan_over_budget();
    let ibp_bounds = graph.collect_node_bounds(&input).expect("ibp bounds");
    let target_ibp = ibp_bounds.get("bn_b").expect("bn_b ibp");
    let node_dim = target_ibp.len();
    assert_eq!(node_dim, 512, "test sizing: bn_b must have 512 dims");

    // Sanity: at 1 MiB the identity pair exceeds the budget, and the target has
    // no Conv2d ancestor, so before #cgan-bn11-chunk the collector skipped it
    // to IBP with MemoryBudgetExceeded.
    let budget_bytes = 1024 * 1024;
    assert!(2 * 4 * node_dim * node_dim > budget_bytes);

    // Collector run under the tiny budget. NY_CROWN_OBJ_CHUNK is pinned to 0 to
    // prove the reroute comes from the collector's chunk_override, not the env.
    let collected = with_serialized_env_vars(
        &[("NY_DENSE_BUDGET_MB", "1"), ("NY_CROWN_OBJ_CHUNK", "0")],
        || {
            graph
                .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
                .expect("crown-ibp collection")
        },
    );

    // 1. Provenance must be Crown — a chunk-routed target is NOT a fallback.
    assert_eq!(
        collected.provenance.get("bn_b").copied(),
        Some(BoundsProvenance::Crown),
        "#cgan-bn11-chunk: over-budget target must be chunk-routed with Crown \
         provenance, not an IBP fallback (events: {:?})",
        collected.fallback_events,
    );
    assert!(
        !collected
            .fallback_events
            .iter()
            .any(|ev| ev.details.contains("bn_b")),
        "#cgan-bn11-chunk: no fallback event may be recorded for the chunk-routed \
         target (events: {:?})",
        collected.fallback_events,
    );

    // 2. Chunk-routed bounds must EQUAL the unbudgeted single-pass CROWN bounds
    //    (after the collector's IBP intersection).
    let single = with_serialized_env_vars(
        &[("NY_DENSE_BUDGET_MB", "4096"), ("NY_CROWN_OBJ_CHUNK", "0")],
        || {
            graph
                .propagate_crown_to_node(
                    &input,
                    "bn_b",
                    &HashMap::new(),
                    &ibp_bounds,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("single-pass CROWN")
        },
    );
    let routed = collected.bounds.get("bn_b").expect("bn_b collected bounds");
    assert_eq!(routed.shape(), target_ibp.shape());

    let tol = 1e-5_f32;
    let il = target_ibp.lower();
    let iu = target_ibp.upper();
    let sl = single.lower();
    let su = single.upper();
    let rl = routed.lower();
    let ru = routed.upper();
    let mut strictly_tighter = false;
    for (idx, &ilv) in il.indexed_iter() {
        let iuv = iu[&idx];
        // Expected collector output: per-element intersection of IBP and the
        // single-pass CROWN bound.
        let elv = sl[&idx].max(ilv);
        let euv = su[&idx].min(iuv);
        let rlv = rl[&idx];
        let ruv = ru[&idx];
        let scale = elv.abs().max(euv.abs()).max(1.0);
        assert!(
            (rlv - elv).abs() <= tol * scale,
            "chunk-routed lower {rlv} != single-pass lower {elv} at {idx:?}"
        );
        assert!(
            (ruv - euv).abs() <= tol * scale,
            "chunk-routed upper {ruv} != single-pass upper {euv} at {idx:?}"
        );
        if rlv > ilv + tol * scale || ruv < iuv - tol * scale {
            strictly_tighter = true;
        }
    }
    // 3. Strictly tighter than IBP somewhere — proves this is genuine CROWN
    //    output, not the pre-fix IBP degradation.
    assert!(
        strictly_tighter,
        "#cgan-bn11-chunk: chunk-routed CROWN must beat IBP on at least one \
         coordinate (otherwise it is indistinguishable from the IBP fallback)"
    );
}

/// #cgan-alpha-chunk: the ALPHA-CROWN intermediate collector must apply the SAME
/// over-budget objective-chunk reroute as the CROWN-IBP collector above. Before
/// this fix `propagate_crown_to_node_with_alpha` hard-coded `chunk_override = None`,
/// so an over-budget generator target (ConvTranspose/BN with no Conv2d ancestor —
/// the cgan_2023 BatchNormalization_11 shape) raised `CpuMemoryExceeded` and the
/// α collector degraded it to the LOOSER reference bound (the observed
/// "α-CROWN: selected node ... unsupported/shape mismatch, falling back to
/// reference bounds" warning). It must now reroute through the bound-equivalent
/// objective-chunked backward and produce bounds EQUAL to the unbudgeted
/// single-pass α CROWN, strictly tighter than IBP.
///
/// The backward from `bn_b` to the input crosses only affine layers (no ReLU),
/// so an empty α state yields the exact affine map — identical single-pass vs
/// chunked (row-independence), and strictly tighter than the IBP interval.
#[test]
fn alpha_chunk_reroutes_over_budget_target_and_matches_single_pass() {
    use crate::bounds::GraphAlphaState;

    let (graph, input) = build_mini_cgan_over_budget();
    let ibp_bounds = graph.collect_node_bounds(&input).expect("ibp bounds");
    let target_ibp = ibp_bounds.get("bn_b").expect("bn_b ibp");
    assert_eq!(
        target_ibp.len(),
        512,
        "test sizing: bn_b must have 512 dims"
    );
    // Sanity: at 1 MiB the identity pair exceeds the budget (compile-time sizes).
    // Static assert: "always true" is the point — it breaks the build if the
    // test's sizing constants stop exceeding the budget.
    #[allow(clippy::assertions_on_constants)]
    const _: () = assert!(2 * 4 * 512 * 512 > 1024 * 1024);

    let alpha_state = GraphAlphaState::new();

    // Over-budget α backward (budget 1 MiB): must NOT error — the new
    // chunk_override reroute streams it instead of raising CpuMemoryExceeded.
    // NY_CROWN_OBJ_CHUNK pinned to 0 to prove the reroute comes from the α
    // collector's chunk_override, not the env knob.
    let routed = with_serialized_env_vars(
        &[("NY_DENSE_BUDGET_MB", "1"), ("NY_CROWN_OBJ_CHUNK", "0")],
        || {
            graph
                .propagate_crown_to_node_with_alpha_for_test(
                    &input,
                    "bn_b",
                    &HashMap::new(),
                    &ibp_bounds,
                    &alpha_state,
                    None,
                    None,
                )
                .expect(
                    "#cgan-alpha-chunk: over-budget α target must reroute to the chunked \
                     backward, not error out to a reference fallback",
                )
        },
    );

    // Single-pass α CROWN at a large budget (no chunking; chunk_override = None).
    let single = with_serialized_env_vars(
        &[("NY_DENSE_BUDGET_MB", "4096"), ("NY_CROWN_OBJ_CHUNK", "0")],
        || {
            graph
                .propagate_crown_to_node_with_alpha_for_test(
                    &input,
                    "bn_b",
                    &HashMap::new(),
                    &ibp_bounds,
                    &alpha_state,
                    None,
                    None,
                )
                .expect("single-pass α CROWN")
        },
    );

    assert_eq!(routed.shape(), target_ibp.shape());
    let tol = 1e-5_f32;
    let il = target_ibp.lower();
    let iu = target_ibp.upper();
    let sl = single.lower();
    let su = single.upper();
    let rl = routed.lower();
    let ru = routed.upper();
    let mut strictly_tighter = false;
    for (idx, &slv) in sl.indexed_iter() {
        let suv = su[&idx];
        let rlv = rl[&idx];
        let ruv = ru[&idx];
        let scale = slv.abs().max(suv.abs()).max(1.0);
        // Chunked α CROWN == single-pass α CROWN (bound-equivalent by
        // row-independence; α slopes threaded identically per chunk).
        assert!(
            (rlv - slv).abs() <= tol * scale,
            "α chunk-routed lower {rlv} != single-pass lower {slv} at {idx:?}"
        );
        assert!(
            (ruv - suv).abs() <= tol * scale,
            "α chunk-routed upper {ruv} != single-pass upper {suv} at {idx:?}"
        );
        // Never wider than IBP (sound enclosure).
        let ilv = il[&idx];
        let iuv = iu[&idx];
        assert!(
            rlv >= ilv - tol * scale && ruv <= iuv + tol * scale,
            "α chunk-routed CROWN [{rlv},{ruv}] not contained in IBP [{ilv},{iuv}] at {idx:?}"
        );
        if rlv > ilv + tol * scale || ruv < iuv - tol * scale {
            strictly_tighter = true;
        }
    }
    assert!(
        strictly_tighter,
        "#cgan-alpha-chunk: α chunk-routed CROWN must beat IBP on at least one \
         coordinate (otherwise it is indistinguishable from the reference fallback)"
    );
}

/// STEP 2b (#cgan-alpha-refresh-budget): the per-node refresh budget must engage
/// ONLY for cgan-like generators (ConvTranspose present AND an over-budget refresh
/// target). Conv2d-only graphs (cifar100 resnet class) and no-over-budget-target
/// graphs must keep the flat shared refresh deadline byte-for-byte, so their BaB
/// budget is untouched.
#[test]
fn alpha_refresh_gate_selects_per_node_only_for_cgan_like() {
    // cgan-like: ConvTranspose present AND bn_b (512-dim) over-budget under 1 MiB.
    let (cgan, cgan_input) = build_mini_cgan_over_budget();
    let cgan_ibp = cgan.collect_node_bounds(&cgan_input).expect("cgan ibp");
    let cgan_targets = vec!["bn_b".to_string()];
    let gate_small = with_serialized_env_vars(&[("NY_DENSE_BUDGET_MB", "1")], || {
        cgan.alpha_refresh_uses_per_node_budget_for_test(&cgan_targets, &cgan_ibp)
    });
    assert!(
        gate_small,
        "cgan-like graph (ConvTranspose + over-budget target) must use the per-node refresh budget"
    );

    // Same graph, large budget: bn_b is no longer over-budget -> flat path.
    let gate_big = with_serialized_env_vars(&[("NY_DENSE_BUDGET_MB", "4096")], || {
        cgan.alpha_refresh_uses_per_node_budget_for_test(&cgan_targets, &cgan_ibp)
    });
    assert!(
        !gate_big,
        "no over-budget target => flat refresh (byte-identical), even with ConvTranspose present"
    );

    // Conv2d-only graph: no ConvTranspose -> flat path regardless of budget.
    let kernel = make_kernel(2, 2, 3, 3, 0xABCD);
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 8, 8).expect("conv");
    let conv_graph = build_conv_relu_graph(conv, false);
    let (conv_in, _l, _u) = make_input_box(&[2, 8, 8], 2 * 8 * 8, 0x1234);
    let conv_ibp = conv_graph.collect_node_bounds(&conv_in).expect("conv ibp");
    let conv_targets = vec!["conv".to_string()];
    let gate_conv = with_serialized_env_vars(&[("NY_DENSE_BUDGET_MB", "1")], || {
        conv_graph.alpha_refresh_uses_per_node_budget_for_test(&conv_targets, &conv_ibp)
    });
    assert!(
        !gate_conv,
        "Conv2d-only graph must never use the per-node refresh budget (no ConvTranspose)"
    );
}
