// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the certified image forward-linear surface
//! (#vnncomp-image-forward-linear): dense-matrix oracles for Conv2d/Add
//! composition, Monte-Carlo containment (soundness), IBP tightness, and the
//! alpha reference-bounds wiring.

use super::*;
use crate::bounds::AlphaCrownConfig;
use crate::layers::{
    BatchNormLayer, Conv2dLayer, ConvTranspose2dLayer, LinearLayer, ReLULayer, SigmoidLayer,
};
use crate::network::GraphNode;
use ndarray::{Array1, Array2, ArrayD, IxDyn};

/// Deterministic pseudo-random generator (SplitMix64-style) so fixtures are
/// reproducible without a rand dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-scale, scale].
    fn next_f32(&mut self, scale: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        (unit * 2.0 - 1.0) * scale
    }
}

fn random_kernel(
    rng: &mut Lcg,
    out_c: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    scale: f32,
) -> ArrayD<f32> {
    let mut v = Vec::with_capacity(out_c * in_c * kh * kw);
    for _ in 0..out_c * in_c * kh * kw {
        v.push(rng.next_f32(scale));
    }
    ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), v).expect("kernel shape")
}

fn random_bias(rng: &mut Lcg, n: usize, scale: f32) -> Array1<f32> {
    Array1::from_iter((0..n).map(|_| rng.next_f32(scale)))
}

fn random_box(rng: &mut Lcg, shape: &[usize], center_scale: f32, radius: f32) -> BoundedTensor {
    let n: usize = shape.iter().product();
    let mut lo = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    for _ in 0..n {
        let c = rng.next_f32(center_scale);
        let r = rng.next_f32(radius).abs() + radius * 0.1;
        lo.push(c - r);
        hi.push(c + r);
    }
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lo).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(shape), hi).expect("upper shape"),
    )
    .expect("valid box")
}

/// Dense (out_dim x in_dim) matrix of a Conv2d in f64 — an independent oracle
/// implementation (plain loops, no im2col/GEMM code shared with production).
struct DenseConv {
    matrix: Vec<f64>, // row-major (out_dim, in_dim)
    bias: Vec<f64>,   // out_dim
    out_dim: usize,
    in_dim: usize,
}

fn dense_conv_oracle(layer: &Conv2dLayer, in_c: usize, in_h: usize, in_w: usize) -> DenseConv {
    let kshape = layer.kernel.shape();
    let (out_c, kin_c, kh, kw) = (kshape[0], kshape[1], kshape[2], kshape[3]);
    assert_eq!(kin_c, in_c, "oracle expects groups=1");
    let (sh, sw) = layer.stride;
    let (ph, pw) = layer.padding;
    let out_h = (in_h + 2 * ph - kh) / sh + 1;
    let out_w = (in_w + 2 * pw - kw) / sw + 1;
    let in_dim = in_c * in_h * in_w;
    let out_dim = out_c * out_h * out_w;
    let mut matrix = vec![0.0f64; out_dim * in_dim];
    let mut bias = vec![0.0f64; out_dim];
    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let p = oc * out_h * out_w + oh * out_w + ow;
                bias[p] = layer.bias.as_ref().map_or(0.0, |b| b[oc] as f64);
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = (oh * sh + ki) as isize - ph as isize;
                            let iw = (ow * sw + kj) as isize - pw as isize;
                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let j = ic * in_h * in_w + ih as usize * in_w + iw as usize;
                                matrix[p * in_dim + j] = layer.kernel[[oc, ic, ki, kj]] as f64;
                            }
                        }
                    }
                }
            }
        }
    }
    DenseConv {
        matrix,
        bias,
        out_dim,
        in_dim,
    }
}

/// Dense f64 matrix for ConvTranspose2d, independently assembled by scattering
/// every `(input, kernel-tap)` product into its output coordinate.  This shares
/// no indexing code with the production transposed-convolution interval kernel.
fn dense_conv_transpose_oracle(
    layer: &ConvTranspose2dLayer,
    in_c: usize,
    in_h: usize,
    in_w: usize,
) -> DenseConv {
    let kshape = layer.kernel.shape();
    let (kin_c, out_c, kh, kw) = (kshape[0], kshape[1], kshape[2], kshape[3]);
    assert_eq!(kin_c, in_c);
    let out_h = (in_h - 1) * layer.stride.0 - 2 * layer.padding.0
        + layer.dilation.0 * (kh - 1)
        + layer.output_padding.0
        + 1;
    let out_w = (in_w - 1) * layer.stride.1 - 2 * layer.padding.1
        + layer.dilation.1 * (kw - 1)
        + layer.output_padding.1
        + 1;
    let in_dim = in_c * in_h * in_w;
    let out_dim = out_c * out_h * out_w;
    let mut matrix = vec![0.0f64; out_dim * in_dim];
    let mut bias = vec![0.0f64; out_dim];
    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                bias[oc * out_h * out_w + oh * out_w + ow] =
                    layer.bias.as_ref().map_or(0.0, |b| b[oc] as f64);
            }
        }
    }
    for ic in 0..in_c {
        for ih in 0..in_h {
            for iw in 0..in_w {
                let input_idx = ic * in_h * in_w + ih * in_w + iw;
                for oc in 0..out_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let oh = ih * layer.stride.0 + ki * layer.dilation.0;
                            let ow = iw * layer.stride.1 + kj * layer.dilation.1;
                            if oh < layer.padding.0 || ow < layer.padding.1 {
                                continue;
                            }
                            let oh = oh - layer.padding.0;
                            let ow = ow - layer.padding.1;
                            if oh >= out_h || ow >= out_w {
                                continue;
                            }
                            let output_idx = oc * out_h * out_w + oh * out_w + ow;
                            matrix[output_idx * in_dim + input_idx] +=
                                layer.kernel[[ic, oc, ki, kj]] as f64;
                        }
                    }
                }
            }
        }
    }
    DenseConv {
        matrix,
        bias,
        out_dim,
        in_dim,
    }
}

fn dense_channel_affine_after_conv(
    conv: &DenseConv,
    channels: usize,
    scale: &[f64],
    bias: &[f64],
) -> DenseConv {
    assert_eq!(scale.len(), channels);
    assert_eq!(bias.len(), channels);
    assert_eq!(conv.out_dim % channels, 0);
    let spatial = conv.out_dim / channels;
    let mut matrix = conv.matrix.clone();
    let mut out_bias = conv.bias.clone();
    for p in 0..conv.out_dim {
        let channel = p / spatial;
        for j in 0..conv.in_dim {
            matrix[p * conv.in_dim + j] *= scale[channel];
        }
        out_bias[p] = out_bias[p] * scale[channel] + bias[channel];
    }
    DenseConv {
        matrix,
        bias: out_bias,
        out_dim: conv.out_dim,
        in_dim: conv.in_dim,
    }
}

/// Exact range of `M x + b` over the box, in f64 (affine over a box is exact).
fn affine_exact_range(
    matrix: &[f64],
    bias: &[f64],
    out_dim: usize,
    in_dim: usize,
    input: &BoundedTensor,
) -> (Vec<f64>, Vec<f64>) {
    let flat = input.flatten();
    let lo: Vec<f64> = flat.lower().iter().map(|&v| v as f64).collect();
    let hi: Vec<f64> = flat.upper().iter().map(|&v| v as f64).collect();
    assert_eq!(lo.len(), in_dim);
    let mut out_lo = vec![0.0f64; out_dim];
    let mut out_hi = vec![0.0f64; out_dim];
    for p in 0..out_dim {
        let mut a = bias[p];
        let mut b = bias[p];
        for j in 0..in_dim {
            let m = matrix[p * in_dim + j];
            if m >= 0.0 {
                a += m * lo[j];
                b += m * hi[j];
            } else {
                a += m * hi[j];
                b += m * lo[j];
            }
        }
        out_lo[p] = a;
        out_hi[p] = b;
    }
    (out_lo, out_hi)
}

/// Sample a deterministic pseudo-random point inside the box.
fn sample_point(rng: &mut Lcg, input: &BoundedTensor) -> BoundedTensor {
    let flat = input.flatten();
    let shape = input.shape().to_vec();
    let vals: Vec<f32> = flat
        .lower()
        .iter()
        .zip(flat.upper().iter())
        .map(|(&l, &u)| {
            let t = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            l + (u - l) * t
        })
        .collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), vals).expect("point shape");
    BoundedTensor::new(arr.clone(), arr).expect("point bounds")
}

/// Assert every node's concrete activation (evaluated via a degenerate-box
/// forward pass) lies inside the claimed forward-linear bounds.
fn assert_mc_containment(graph: &GraphNetwork, input: &BoundedTensor, samples: usize, seed: u64) {
    assert_mc_containment_with_alphas(graph, input, samples, seed, None);
}

/// [`assert_mc_containment`] against the ALPHA-FED forward-linear map
/// (#w4-root-alpha) when `relu_alphas` is provided.
fn assert_mc_containment_with_alphas(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    samples: usize,
    seed: u64,
    relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
) {
    let forward = match relu_alphas {
        Some(alphas) => graph
            .collect_forward_linear_bounds_dag_with_alphas(input, alphas, None)
            .expect("alpha-fed forward-linear collection should succeed"),
        None if graph
            .nodes
            .values()
            .any(|node| matches!(node.layer, Layer::ConvTranspose2d(_))) =>
        {
            graph
                .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(input, None)
                .expect("dark ConvTranspose forward-linear collection should succeed in test")
        }
        None => graph
            .collect_forward_linear_bounds_dag_with_engine(input, None)
            .expect("forward-linear collection should succeed"),
    };
    let mut rng = Lcg::new(seed);
    let mut points: Vec<BoundedTensor> = (0..samples)
        .map(|_| sample_point(&mut rng, input))
        .collect();
    // Include the box corners spanned by all-lower / all-upper.
    points.push(BoundedTensor::new(input.lower().clone(), input.lower().clone()).unwrap());
    points.push(BoundedTensor::new(input.upper().clone(), input.upper().clone()).unwrap());

    for (pi, point) in points.iter().enumerate() {
        let exact = graph
            .collect_node_bounds_with_engine(point, None)
            .expect("point evaluation should succeed");
        for (node_name, claimed) in &forward {
            let concrete = exact.get(node_name).expect("point map should include node");
            for ((&lo, &hi), (&cl, &cu)) in claimed
                .lower()
                .iter()
                .zip(claimed.upper().iter())
                .zip(concrete.lower().iter().zip(concrete.upper().iter()))
            {
                // The point-eval is itself an outward-rounded degenerate IBP;
                // its midpoint is within a hair of the true activation.
                let value = f64::midpoint(cl as f64, cu as f64);
                let slack = 1e-4 * (1.0 + value.abs());
                assert!(
                    (lo as f64) - slack <= value && value <= (hi as f64) + slack,
                    "MC containment violated at node '{node_name}' sample {pi}: \
                     value={value}, claimed=[{lo}, {hi}]"
                );
            }
        }
    }
}

fn tensor_width_sum(bounds: &BoundedTensor) -> f64 {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(&l, &u)| (u as f64) - (l as f64))
        .sum()
}

/// conv1 (2ch 4x4 -> 4ch 4x4, pad 1) -> conv2 (4ch 4x4 -> 3ch 2x2, pad 0,
/// stride 1, kernel 3x3): pure affine chain with a dense f64 oracle.
fn build_conv_chain() -> (GraphNetwork, BoundedTensor, Conv2dLayer, Conv2dLayer) {
    let mut rng = Lcg::new(7);
    let conv1 = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 4, 2, 3, 3, 0.6),
        Some(random_bias(&mut rng, 4, 0.3)),
        (1, 1),
        (1, 1),
        4,
        4,
    )
    .expect("conv1");
    let conv2 = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 3, 4, 3, 3, 0.6),
        Some(random_bias(&mut rng, 3, 0.3)),
        (1, 1),
        (0, 0),
        4,
        4,
    )
    .expect("conv2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1.clone())));
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv2.clone()),
        vec!["conv1".to_string()],
    ));
    graph.set_output("conv2");

    let input = random_box(&mut rng, &[2, 4, 4], 0.5, 0.4);
    (graph, input, conv1, conv2)
}

/// Residual DAG mirroring a cifar100 ResNet block pair:
/// conv1 -> relu1 -> conv2 -> Add(conv2, shortcut(relu1)) -> relu2
///        -> conv3 -> Add(conv3, prev) -> relu3 -> flatten -> linear.
fn build_residual_dag(seed: u64, weight_scale: f32) -> (GraphNetwork, BoundedTensor) {
    let mut rng = Lcg::new(seed);
    let mut graph = GraphNetwork::new();

    let conv1 = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 4, 2, 3, 3, weight_scale),
        Some(random_bias(&mut rng, 4, 0.2)),
        (1, 1),
        (1, 1),
        4,
        4,
    )
    .expect("conv1");
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));

    let conv2 = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 4, 4, 3, 3, weight_scale),
        Some(random_bias(&mut rng, 4, 0.2)),
        (1, 1),
        (1, 1),
        4,
        4,
    )
    .expect("conv2");
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv2),
        vec!["relu1".to_string()],
    ));
    // 1x1 shortcut conv (like the cifar100 ResNet shortcut branch).
    let shortcut = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 4, 4, 1, 1, weight_scale),
        None,
        (1, 1),
        (0, 0),
        4,
        4,
    )
    .expect("shortcut");
    graph.add_node(GraphNode::new(
        "shortcut",
        Layer::Conv2d(shortcut),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "add1",
        Layer::Add(crate::layers::AddLayer),
        vec!["conv2".to_string(), "shortcut".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add1".to_string()],
    ));

    let conv3 = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 4, 4, 3, 3, weight_scale),
        Some(random_bias(&mut rng, 4, 0.2)),
        (1, 1),
        (1, 1),
        4,
        4,
    )
    .expect("conv3");
    graph.add_node(GraphNode::new(
        "conv3",
        Layer::Conv2d(conv3),
        vec!["relu2".to_string()],
    ));
    // Identity residual skip (second block).
    graph.add_node(GraphNode::new(
        "add2",
        Layer::Add(crate::layers::AddLayer),
        vec!["conv3".to_string(), "add1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["add2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(crate::layers::FlattenLayer::new(0)),
        vec!["relu3".to_string()],
    ));

    let mut w = Array2::<f32>::zeros((3, 64));
    for v in w.iter_mut() {
        *v = rng.next_f32(0.4);
    }
    let linear = LinearLayer::new(w, Some(random_bias(&mut rng, 3, 0.2))).expect("linear");
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(linear),
        vec!["flatten".to_string()],
    ));
    graph.set_output("out");

    // eps-style small box (like a VNN-COMP image perturbation): early layers
    // stay mostly stable, which is where forward substitution beats IBP.
    let input = random_box(&mut rng, &[2, 4, 4], 0.4, 0.02);
    (graph, input)
}

#[test]
fn test_image_conv_chain_matches_dense_oracle() {
    let (graph, input, conv1, conv2) = build_conv_chain();
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("conv chain should be supported in image mode");

    // Dense oracle: M = C2 @ C1, b = C2 @ b1 + b2 in f64.
    let d1 = dense_conv_oracle(&conv1, 2, 4, 4);
    let d2 = dense_conv_oracle(&conv2, 4, 4, 4);
    let mut m = vec![0.0f64; d2.out_dim * d1.in_dim];
    let mut b = d2.bias.clone();
    for p in 0..d2.out_dim {
        for k in 0..d2.in_dim {
            let w = d2.matrix[p * d2.in_dim + k];
            if w == 0.0 {
                continue;
            }
            b[p] += w * d1.bias[k];
            for j in 0..d1.in_dim {
                m[p * d1.in_dim + j] += w * d1.matrix[k * d1.in_dim + j];
            }
        }
    }
    let (ref_lo, ref_hi) = affine_exact_range(&m, &b, d2.out_dim, d1.in_dim, &input);

    let got = &forward["conv2"];
    assert_eq!(got.len(), d2.out_dim);
    for (p, (&lo, &hi)) in got.lower().iter().zip(got.upper().iter()).enumerate() {
        let tol = 1e-4 * (1.0 + ref_lo[p].abs().max(ref_hi[p].abs()));
        // Sound: claimed interval must CONTAIN the exact range.
        assert!(
            (lo as f64) <= ref_lo[p] + 1e-9 && (hi as f64) >= ref_hi[p] - 1e-9,
            "conv2[{p}]: claimed [{lo}, {hi}] must contain exact [{}, {}]",
            ref_lo[p],
            ref_hi[p]
        );
        // Tight: within tolerance of the exact affine range (the certified
        // penalties are ~1e-7-relative at this scale).
        assert!(
            (lo as f64 - ref_lo[p]).abs() <= tol && (hi as f64 - ref_hi[p]).abs() <= tol,
            "conv2[{p}]: claimed [{lo}, {hi}] too loose vs exact [{}, {}]",
            ref_lo[p],
            ref_hi[p]
        );
    }
    assert_mc_containment(&graph, &input, 20, 11);
}

#[test]
fn test_image_add_two_branch_matches_dense_oracle() {
    let mut rng = Lcg::new(21);
    let conv_a = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 3, 2, 3, 3, 0.7),
        Some(random_bias(&mut rng, 3, 0.3)),
        (1, 1),
        (1, 1),
        4,
        4,
    )
    .expect("conv_a");
    let conv_b = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 3, 2, 1, 1, 0.7),
        None,
        (1, 1),
        (0, 0),
        4,
        4,
    )
    .expect("conv_b");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "branch_a",
        Layer::Conv2d(conv_a.clone()),
    ));
    graph.add_node(GraphNode::from_input(
        "branch_b",
        Layer::Conv2d(conv_b.clone()),
    ));
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(crate::layers::AddLayer),
        vec!["branch_a".to_string(), "branch_b".to_string()],
    ));
    graph.set_output("add");
    let input = random_box(&mut rng, &[2, 4, 4], 0.5, 0.35);

    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("two-branch add should be supported in image mode");

    let da = dense_conv_oracle(&conv_a, 2, 4, 4);
    let db = dense_conv_oracle(&conv_b, 2, 4, 4);
    assert_eq!(da.out_dim, db.out_dim);
    let m: Vec<f64> = da
        .matrix
        .iter()
        .zip(db.matrix.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    let b: Vec<f64> = da
        .bias
        .iter()
        .zip(db.bias.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    let (ref_lo, ref_hi) = affine_exact_range(&m, &b, da.out_dim, da.in_dim, &input);

    let got = &forward["add"];
    for (p, (&lo, &hi)) in got.lower().iter().zip(got.upper().iter()).enumerate() {
        let tol = 1e-4 * (1.0 + ref_lo[p].abs().max(ref_hi[p].abs()));
        assert!(
            (lo as f64) <= ref_lo[p] + 1e-9 && (hi as f64) >= ref_hi[p] - 1e-9,
            "add[{p}]: claimed [{lo}, {hi}] must contain exact [{}, {}]",
            ref_lo[p],
            ref_hi[p]
        );
        assert!(
            (lo as f64 - ref_lo[p]).abs() <= tol && (hi as f64 - ref_hi[p]).abs() <= tol,
            "add[{p}]: claimed [{lo}, {hi}] too loose vs exact [{}, {}]",
            ref_lo[p],
            ref_hi[p]
        );
    }
    assert_mc_containment(&graph, &input, 20, 22);
}

#[test]
fn test_image_residual_dag_mc_containment_and_tighter_than_ibp() {
    let (graph, input) = build_residual_dag(42, 0.5);
    assert_mc_containment(&graph, &input, 40, 33);

    let ibp = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("IBP collection");
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear collection");

    // Contained in IBP everywhere (tighten_with_ibp invariant)...
    for (name, fw) in &forward {
        let ib = &ibp[name];
        for ((&fl, &fu), (&il, &iu)) in fw
            .lower()
            .iter()
            .zip(fw.upper().iter())
            .zip(ib.lower().iter().zip(ib.upper().iter()))
        {
            assert!(
                fl >= il - 1e-6 && fu <= iu + 1e-6,
                "node '{name}': forward [{fl}, {fu}] must be within IBP [{il}, {iu}]"
            );
        }
    }
    // ...and strictly tighter at the output (the whole point of the pass).
    let fw_w = tensor_width_sum(&forward["out"]);
    let ibp_w = tensor_width_sum(&ibp["out"]);
    assert!(
        fw_w < ibp_w * 0.95,
        "forward-linear output width {fw_w} should be well below IBP width {ibp_w}"
    );
}

#[test]
fn test_image_deep_conv_stack_stays_much_tighter_than_ibp() {
    // 6 conv+relu layers: IBP compounds unsigned row-norm growth per layer,
    // forward substitution keeps signed cancellation. This is the cifar100
    // failure mode in miniature.
    let mut rng = Lcg::new(77);
    let mut graph = GraphNetwork::new();
    let mut prev = String::new();
    for layer_idx in 0..6 {
        let conv = Conv2dLayer::with_input_shape(
            random_kernel(&mut rng, 4, if layer_idx == 0 { 2 } else { 4 }, 3, 3, 0.7),
            Some(random_bias(&mut rng, 4, 0.1)),
            (1, 1),
            (1, 1),
            4,
            4,
        )
        .expect("conv");
        let conv_name = format!("conv{layer_idx}");
        if layer_idx == 0 {
            graph.add_node(GraphNode::from_input(
                conv_name.clone(),
                Layer::Conv2d(conv),
            ));
        } else {
            graph.add_node(GraphNode::new(
                conv_name.clone(),
                Layer::Conv2d(conv),
                vec![prev.clone()],
            ));
        }
        let relu_name = format!("relu{layer_idx}");
        graph.add_node(GraphNode::new(
            relu_name.clone(),
            Layer::ReLU(ReLULayer),
            vec![conv_name],
        ));
        prev = relu_name;
    }
    graph.set_output(&prev);

    // Small box around a point: forward-substitution stays near-exact while
    // IBP explodes.
    let mut center = Vec::with_capacity(32);
    for _ in 0..32 {
        center.push(rng.next_f32(0.5));
    }
    let lo: Vec<f32> = center.iter().map(|&c| c - 0.005).collect();
    let hi: Vec<f32> = center.iter().map(|&c| c + 0.005).collect();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4, 4]), lo).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 4, 4]), hi).unwrap(),
    )
    .unwrap();

    assert_mc_containment(&graph, &input, 30, 55);

    let ibp = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .unwrap();
    let fw_w = tensor_width_sum(&forward[&prev]);
    let ibp_w = tensor_width_sum(&ibp[&prev]);
    assert!(
        fw_w.is_finite() && fw_w > 0.0,
        "deep-stack forward width must be finite and positive, got {fw_w}"
    );
    assert!(
        fw_w < ibp_w * 0.5,
        "deep-stack forward width {fw_w} should be far below IBP width {ibp_w}"
    );
}

/// Random-geometry Monte-Carlo sweep: strides, paddings, kernel sizes,
/// non-square inputs. Every claimed bound must contain sampled activations.
#[test]
fn test_image_conv_random_geometry_soundness_sweep() {
    for seed in 0..8u64 {
        let mut rng = Lcg::new(1000 + seed);
        let (kh, kw) = if seed % 3 == 0 { (1, 1) } else { (3, 3) };
        let stride = if seed % 2 == 0 { (1, 1) } else { (2, 2) };
        let padding = if kh == 1 {
            (0, 0)
        } else {
            ((seed % 2) as usize, (seed % 2) as usize)
        };
        let (in_h, in_w) = (4 + (seed % 2) as usize * 2, 4);
        let in_c = 2;
        let out_c = 3;
        let conv = Conv2dLayer::with_input_shape(
            random_kernel(&mut rng, out_c, in_c, kh, kw, 0.8),
            Some(random_bias(&mut rng, out_c, 0.3)),
            stride,
            padding,
            in_h,
            in_w,
        )
        .expect("conv");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv".to_string()],
        ));
        graph.set_output("relu");
        let input = random_box(&mut rng, &[in_c, in_h, in_w], 0.5, 0.4);
        assert_mc_containment(&graph, &input, 15, 2000 + seed);
    }
}

/// ConvTranspose2d + BatchNorm differential regression for the cGAN operator
/// seam.  Covers mixed-sign kernels, asymmetric stride/padding/dilation,
/// nonzero output_padding, negative BN scale, and nonzero certified BN
/// precompute errors.  An independent dense f64 matrix is the oracle.
#[test]
fn test_image_conv_transpose_batch_norm_dense_f64_enclosure() {
    let in_c = 2usize;
    let out_c = 3usize;
    let (in_h, in_w) = (2usize, 3usize);
    let kernel = ArrayD::from_shape_fn(IxDyn(&[in_c, out_c, 3, 2]), |idx| {
        let raw = ((idx[0] * 29 + idx[1] * 17 + idx[2] * 7 + idx[3] * 3) % 19) as f32;
        (raw - 9.0) * 0.137
    });
    assert!(kernel.iter().any(|&w| w < 0.0));
    assert!(kernel.iter().any(|&w| w > 0.0));
    let conv = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![0.13, -0.27, 0.08])),
        (2, 1),
        (1, 0),
        (1, 2),
        (1, 0),
    )
    .expect("valid ConvTranspose geometry");
    let bn = BatchNormLayer {
        scale: Array1::from_vec(vec![-1.35, 0.62, 1.91]).into_dyn(),
        bias: Array1::from_vec(vec![0.17, -0.09, 0.31]).into_dyn(),
        scale_err: Array1::from_vec(vec![1.5e-4, 3.0e-5, 8.0e-5]).into_dyn(),
        bias_err: Array1::from_vec(vec![2.0e-4, 7.0e-6, 4.5e-5]).into_dyn(),
        num_channels: out_c,
    };

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "convt",
        Layer::ConvTranspose2d(conv.clone()),
    ));
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn.clone()),
        vec!["convt".to_string()],
    ));
    graph.set_output("bn");

    let mut rng = Lcg::new(0xC6A4_2025);
    let input = random_box(&mut rng, &[in_c, in_h, in_w], 0.8, 0.17);
    let forward = graph
        .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
        .expect("ConvTranspose+BN must be on the certified image surface");
    let claimed = forward["bn"].flatten();

    let dense_conv = dense_conv_transpose_oracle(&conv, in_c, in_h, in_w);
    let stored_scale: Vec<f64> = bn.scale.iter().map(|&x| x as f64).collect();
    let stored_bias: Vec<f64> = bn.bias.iter().map(|&x| x as f64).collect();
    let scale_err: Vec<f64> = bn.scale_err.iter().map(|&x| x as f64).collect();
    let bias_err: Vec<f64> = bn.bias_err.iter().map(|&x| x as f64).collect();

    // Exercise nominal parameters and multiple admissible exact-real
    // parameters at the certified precompute-error extremes.
    for (case, (scale_direction, bias_direction)) in [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([1.0, -1.0, 0.75], [-1.0, 1.0, -0.5]),
        ([-0.8, 0.6, -1.0], [0.4, -0.7, 1.0]),
    ]
    .into_iter()
    .enumerate()
    {
        let real_scale: Vec<f64> = (0..out_c)
            .map(|c| stored_scale[c] + scale_direction[c] * scale_err[c])
            .collect();
        let real_bias: Vec<f64> = (0..out_c)
            .map(|c| stored_bias[c] + bias_direction[c] * bias_err[c])
            .collect();
        let dense = dense_channel_affine_after_conv(&dense_conv, out_c, &real_scale, &real_bias);
        let (exact_lo, exact_hi) = affine_exact_range(
            &dense.matrix,
            &dense.bias,
            dense.out_dim,
            dense.in_dim,
            &input,
        );
        for p in 0..dense.out_dim {
            assert!(
                (claimed.lower()[p] as f64) <= exact_lo[p]
                    && (claimed.upper()[p] as f64) >= exact_hi[p],
                "case {case} output {p}: claimed [{}, {}] excludes dense f64 exact range [{}, {}]",
                claimed.lower()[p],
                claimed.upper()[p],
                exact_lo[p],
                exact_hi[p]
            );
        }

        // F64 interior-point enclosure, independent of the graph's f32 point
        // evaluator.  (The exact box range above is stronger; samples make a
        // regression easier to localize to individual arithmetic paths.)
        let flat = input.flatten();
        for sample in 0..32 {
            let x: Vec<f64> = flat
                .lower()
                .iter()
                .zip(flat.upper().iter())
                .map(|(&l, &u)| {
                    let t = (rng.next_u64() >> 40) as f64 / (1u64 << 24) as f64;
                    (l as f64) + ((u as f64) - (l as f64)) * t
                })
                .collect();
            for p in 0..dense.out_dim {
                let mut value = dense.bias[p];
                for (j, &xj) in x.iter().enumerate() {
                    value += dense.matrix[p * dense.in_dim + j] * xj;
                }
                assert!(
                    (claimed.lower()[p] as f64) <= value && value <= (claimed.upper()[p] as f64),
                    "case {case} sample {sample} output {p}: f64 value {value} outside [{}, {}]",
                    claimed.lower()[p],
                    claimed.upper()[p]
                );
            }
        }
    }

    // Also compare against the network's production point evaluator at every
    // node, including the ConvTranspose pre-BN node.
    assert_mc_containment(&graph, &input, 32, 0xC6A4_2026);

    // Malformed public struct metadata must fail closed instead of reducing a
    // directed widening penalty.
    let mut bad_bn = bn;
    bad_bn.scale_err[[0]] = -1.0e-4;
    let mut bad_graph = GraphNetwork::new();
    bad_graph.add_node(GraphNode::from_input("convt", Layer::ConvTranspose2d(conv)));
    bad_graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bad_bn),
        vec!["convt".to_string()],
    ));
    bad_graph.set_output("bn");
    let err = bad_graph
        .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
        .expect_err("negative BN error metadata must fail closed");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

/// Default-off compatibility for the dark cGAN surface:
/// ConvTranspose-only graphs retain the pre-existing generic route/refusal,
/// while a Conv2d image graph containing BatchNorm still fails at the old
/// image allowlist.
#[test]
fn test_image_conv_transpose_dark_gate_preserves_legacy_routing() {
    let convt = ConvTranspose2dLayer::with_input_shape(
        ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), -0.75),
        Some(Array1::from_vec(vec![0.2])),
        (1, 1),
        (0, 0),
        1,
        1,
    )
    .expect("tiny ConvTranspose");
    let mut convt_only = GraphNetwork::new();
    convt_only.add_node(GraphNode::from_input(
        "convt",
        Layer::ConvTranspose2d(convt),
    ));
    convt_only.set_output("convt");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), -0.3),
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.4),
    )
    .unwrap();
    let err = convt_only
        .collect_forward_linear_bounds_dag_without_conv_transpose_for_test(&input, None)
        .expect_err("legacy generic packet refuses ConvTranspose");
    let message = err.to_string();
    assert!(
        message.contains("operator is outside the forward-linear packet surface")
            && !message.contains("dark"),
        "dark-off ConvTranspose-only graph must retain the generic legacy refusal, got: {message}"
    );

    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 0.5),
        None,
        (1, 1),
        (0, 0),
        1,
        1,
    )
    .expect("tiny Conv2d");
    let bn = BatchNormLayer::from_scale_bias(
        Array1::from_vec(vec![1.2]).into_dyn(),
        Array1::from_vec(vec![-0.1]).into_dyn(),
    )
    .expect("tiny BatchNorm");
    let mut conv_bn = GraphNetwork::new();
    conv_bn.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    conv_bn.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["conv".to_string()],
    ));
    conv_bn.set_output("bn");
    let err = conv_bn
        .collect_forward_linear_bounds_dag_without_conv_transpose_for_test(&input, None)
        .expect_err("dark-off Conv2d+BatchNorm must retain old fail-closed image surface");
    assert!(matches!(err, NyError::UnsupportedConfiguration(_)));
}

#[test]
fn test_alpha_reference_bounds_use_forward_linear_for_conv_dag() {
    let (graph, input) = build_residual_dag(42, 0.5);
    let config = AlphaCrownConfig {
        fix_interm_bounds: true,
        ..AlphaCrownConfig::default()
    };
    let exec_order = graph.exec_order().expect("exec order").to_vec();
    let reference = graph
        .collect_alpha_reference_bounds_with_engine(&input, &config, None, &exec_order)
        .expect("reference bounds");
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear bounds");

    // The conv-DAG reference source must BE the forward-linear map (default ON).
    assert_eq!(reference.len(), forward.len());
    for (name, fw) in &forward {
        let rf = reference.get(name).expect("reference includes node");
        assert_eq!(rf.lower(), fw.lower(), "node '{name}' lower mismatch");
        assert_eq!(rf.upper(), fw.upper(), "node '{name}' upper mismatch");
    }
}

/// #cgan-fwdlin-ref: a cgan-class SEQUENTIAL ConvTranspose chain (is_dag =
/// false) never reached the conv-DAG forward-linear branch, so the certified
/// ConvTranspose/BatchNorm surface was unreachable exactly on the graphs it
/// was built for. Under the dark surface gate the α reference collection must
/// serve the forward-linear map for such chains.
#[test]
fn test_alpha_reference_bounds_use_forward_linear_for_sequential_conv_transpose_chain() {
    let in_c = 2usize;
    let out_c = 3usize;
    let (in_h, in_w) = (2usize, 3usize);
    let kernel = ArrayD::from_shape_fn(IxDyn(&[in_c, out_c, 2, 2]), |idx| {
        let raw = ((idx[0] * 13 + idx[1] * 7 + idx[2] * 5 + idx[3] * 3) % 11) as f32;
        (raw - 5.0) * 0.21
    });
    let conv = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![0.05, -0.11, 0.02])),
        (1, 1),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .expect("valid ConvTranspose geometry");
    let bn = BatchNormLayer::from_scale_bias(
        Array1::from_vec(vec![1.4, -0.6, 0.9]).into_dyn(),
        Array1::from_vec(vec![-0.2, 0.1, 0.3]).into_dyn(),
    )
    .expect("BatchNorm");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("convt", Layer::ConvTranspose2d(conv)));
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["convt".to_string()],
    ));
    graph.set_output("bn");

    let mut rng = Lcg::new(0xC6A4_2126);
    let input = random_box(&mut rng, &[in_c, in_h, in_w], 0.6, 0.12);
    let exec_order = graph.exec_order().expect("exec order").to_vec();
    assert!(
        graph.is_sequential_graph(&exec_order),
        "fixture must be a sequential chain to cover the is_dag=false route"
    );

    crate::tests::with_serialized_env_vars(
        &[("NY_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "1")],
        || {
            let config = AlphaCrownConfig {
                fix_interm_bounds: true,
                ..AlphaCrownConfig::default()
            };
            let reference = graph
                .collect_alpha_reference_bounds_with_engine(&input, &config, None, &exec_order)
                .expect("reference bounds");
            let forward = graph
                .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
                .expect("forward-linear chain collection");

            assert_eq!(reference.len(), forward.len());
            for (name, fw) in &forward {
                let rf = reference.get(name).expect("reference includes node");
                assert_eq!(rf.lower(), fw.lower(), "node '{name}' lower mismatch");
                assert_eq!(rf.upper(), fw.upper(), "node '{name}' upper mismatch");
            }
        },
    );
}

#[test]
fn test_alpha_reference_bounds_fail_closed_to_ibp_on_unsupported_op() {
    // Conv DAG with a Sigmoid (outside the certified image allowlist): the
    // forward-linear attempt must fail closed and the reference bounds must
    // equal plain IBP.
    let (mut graph, input) = build_residual_dag(42, 0.5);
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["out".to_string()],
    ));
    graph.set_output("sigmoid");

    let err = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect_err("sigmoid must be outside the image allowlist");
    assert!(
        matches!(err, NyError::UnsupportedConfiguration(_)),
        "expected fail-closed UnsupportedConfiguration, got {err:?}"
    );

    let config = AlphaCrownConfig {
        fix_interm_bounds: true,
        ..AlphaCrownConfig::default()
    };
    let exec_order = graph.exec_order().expect("exec order").to_vec();
    let reference = graph
        .collect_alpha_reference_bounds_with_engine(&input, &config, None, &exec_order)
        .expect("reference bounds fall back to IBP");
    let ibp = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("IBP bounds");
    for (name, ib) in &ibp {
        let rf = reference.get(name).expect("reference includes node");
        assert_eq!(
            rf.lower(),
            ib.lower(),
            "node '{name}' IBP-fallback lower mismatch"
        );
        assert_eq!(
            rf.upper(),
            ib.upper(),
            "node '{name}' IBP-fallback upper mismatch"
        );
    }
}

/// Build a per-ReLU alpha map with uniformly random values in [0, 1]
/// (lengths taken from the concretized node-bounds map).
fn random_alpha_map(
    graph: &GraphNetwork,
    forward: &HashMap<String, BoundedTensor>,
    rng: &mut Lcg,
) -> std::collections::BTreeMap<String, Array1<f32>> {
    let mut alphas = std::collections::BTreeMap::new();
    for (name, node) in &graph.nodes {
        if matches!(node.layer, Layer::ReLU(_)) {
            let len = forward[name].len();
            let alpha = Array1::from_iter(
                (0..len).map(|_| (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32),
            );
            alphas.insert(name.clone(), alpha);
        }
    }
    alphas
}

/// #w4-root-alpha soundness: the ALPHA-FED forward-linear map must contain
/// every concrete activation for RANDOM per-neuron alphas in [0, 1], on the
/// residual conv DAG (mirrors the fixed-slope MC containment suite).
#[test]
fn test_image_alpha_fed_residual_dag_mc_containment_random_alphas() {
    for seed in [42u64, 43, 44] {
        let (graph, input) = build_residual_dag(seed, 0.5);
        let forward = graph
            .collect_forward_linear_bounds_dag_with_engine(&input, None)
            .expect("fixed-slope collection");
        let mut rng = Lcg::new(9000 + seed);
        let alphas = random_alpha_map(&graph, &forward, &mut rng);
        assert!(!alphas.is_empty(), "residual DAG must have ReLU nodes");
        assert_mc_containment_with_alphas(&graph, &input, 30, 7000 + seed, Some(&alphas));
    }
}

/// #w4-root-alpha soundness: random alphas on the deep conv stack (the
/// cifar100 failure mode in miniature), including the extreme all-zero and
/// all-one alpha maps.
#[test]
fn test_image_alpha_fed_deep_stack_mc_containment_extremes() {
    let (graph, input) = build_residual_dag(77, 0.7);
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("fixed-slope collection");

    for fill in [0.0f32, 1.0f32] {
        let mut alphas = std::collections::BTreeMap::new();
        for (name, node) in &graph.nodes {
            if matches!(node.layer, Layer::ReLU(_)) {
                alphas.insert(name.clone(), Array1::from_elem(forward[name].len(), fill));
            }
        }
        assert_mc_containment_with_alphas(&graph, &input, 25, 8100, Some(&alphas));
    }

    // Out-of-range and NaN values must be handled (clamped / adaptive
    // fallback) without soundness loss.
    let mut rng = Lcg::new(8200);
    let mut alphas = random_alpha_map(&graph, &forward, &mut rng);
    for alpha in alphas.values_mut() {
        if alpha.len() > 2 {
            alpha[0] = -0.7; // clamps to 0
            alpha[1] = 1.9; // clamps to 1
            alpha[2] = f32::NAN; // adaptive fallback
        }
    }
    assert_mc_containment_with_alphas(&graph, &input, 25, 8300, Some(&alphas));
}

/// #w4-root-alpha regression: feeding the ADAPTIVE alpha values (1 if
/// u > −l else 0, computed from the pass's own running pre-activation
/// bounds) reproduces the fixed-slope map BYTE-IDENTICALLY. Stable neurons
/// get a sentinel 0.5 to prove alpha is ignored outside the crossing region.
#[test]
fn test_image_alpha_fed_adaptive_alphas_byte_identical() {
    let (graph, input) = build_residual_dag(42, 0.5);
    let baseline = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("fixed-slope collection");

    // Reconstruct the adaptive per-neuron lower slopes from the RUNNING
    // tightened pre-activation bounds (the ReLU predecessor's entry in the
    // returned map — exactly what compose_relu_diag_forward consumed).
    let mut alphas = std::collections::BTreeMap::new();
    for (name, node) in &graph.nodes {
        if !matches!(node.layer, Layer::ReLU(_)) {
            continue;
        }
        let pred = node.inputs.first().expect("ReLU has one input");
        let pre = baseline
            .get(pred)
            .expect("predecessor bounds present")
            .flatten();
        let alpha =
            Array1::from_iter(pre.lower().iter().zip(pre.upper().iter()).map(|(&l, &u)| {
                if l < 0.0 && u > 0.0 {
                    if u > -l {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    0.5 // sentinel: must be ignored on stable neurons
                }
            }));
        alphas.insert(name.clone(), alpha);
    }
    assert!(!alphas.is_empty());

    let alpha_fed = graph
        .collect_forward_linear_bounds_dag_with_alphas(&input, &alphas, None)
        .expect("alpha-fed collection");

    assert_eq!(baseline.len(), alpha_fed.len());
    for (name, base) in &baseline {
        let got = &alpha_fed[name];
        assert_eq!(base.lower(), got.lower(), "node '{name}' lower bits differ");
        assert_eq!(base.upper(), got.upper(), "node '{name}' upper bits differ");
    }
}

/// #w4-root-alpha: a wrong-length alpha vector fails OPEN to the adaptive
/// rule (byte-identical to the fixed-slope map) instead of erroring or
/// silently mis-indexing.
#[test]
fn test_image_alpha_fed_wrong_length_ignored() {
    let (graph, input) = build_residual_dag(42, 0.5);
    let baseline = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("fixed-slope collection");

    let mut alphas = std::collections::BTreeMap::new();
    alphas.insert("relu1".to_string(), Array1::from_elem(3, 0.25)); // wrong length
    let alpha_fed = graph
        .collect_forward_linear_bounds_dag_with_alphas(&input, &alphas, None)
        .expect("alpha-fed collection");
    for (name, base) in &baseline {
        let got = &alpha_fed[name];
        assert_eq!(base.lower(), got.lower(), "node '{name}' lower bits differ");
        assert_eq!(base.upper(), got.upper(), "node '{name}' upper bits differ");
    }
}

/// #w4-root-alpha: the alpha-fed C-margin spec route is a sound enclosure of
/// the spec values (contains the margin evaluated at sampled points) and the
/// two cache slots never clobber each other — the fixed-slope route returns
/// the same bits before and after an alpha-fed call on the same input.
#[test]
fn test_image_alpha_fed_spec_margin_sound_and_cache_isolated() {
    let (graph, input) = build_residual_dag(42, 0.5);
    // 2 margin rows over the 3 outputs: y0 − y1, y2 − y0.
    let mut spec = Array2::<f32>::zeros((2, 3));
    spec[[0, 0]] = 1.0;
    spec[[0, 1]] = -1.0;
    spec[[1, 2]] = 1.0;
    spec[[1, 0]] = -1.0;

    let fixed_before = graph
        .forward_linear_spec_margin_bounds(&input, &spec, None, None)
        .expect("fixed-slope margin");

    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("fixed-slope collection");
    let mut rng = Lcg::new(6100);
    let alphas = random_alpha_map(&graph, &forward, &mut rng);
    let alpha_margin = graph
        .forward_linear_spec_margin_bounds_with_alphas(&input, &spec, &alphas, None, None)
        .expect("alpha-fed margin");

    // Soundness: sampled concrete margins lie inside BOTH enclosures.
    let mut rng = Lcg::new(6200);
    for _ in 0..25 {
        let point = sample_point(&mut rng, &input);
        let exact = graph
            .collect_node_bounds_with_engine(&point, None)
            .expect("point eval");
        let out = exact["out"].flatten();
        for row in 0..2 {
            let mut val = 0.0f64;
            for col in 0..3 {
                let y = f64::midpoint(out.lower()[col] as f64, out.upper()[col] as f64);
                val += spec[[row, col]] as f64 * y;
            }
            let slack = 1e-4 * (1.0 + val.abs());
            for (label, margin) in [("fixed", &fixed_before), ("alpha", &alpha_margin)] {
                let lo = margin.lower()[row] as f64;
                let hi = margin.upper()[row] as f64;
                assert!(
                    lo - slack <= val && val <= hi + slack,
                    "{label} margin row {row}: value {val} outside [{lo}, {hi}]"
                );
            }
        }
    }

    // Cache isolation: the fixed-slope entry must be untouched by the
    // alpha-fed pass (same bits, served from the fixed slot).
    let fixed_after = graph
        .forward_linear_spec_margin_bounds(&input, &spec, None, None)
        .expect("fixed-slope margin after alpha pass");
    assert_eq!(fixed_before.lower(), fixed_after.lower());
    assert_eq!(fixed_before.upper(), fixed_after.upper());
}

/// Fixture accessor for the alpha-opt gradient finite-difference test
/// (`alpha_opt::grad_tests`): a residual DAG with a wider box so several
/// neurons cross.
pub(super) fn build_residual_dag_for_grad_test() -> (GraphNetwork, BoundedTensor) {
    build_residual_dag(42, 0.7)
}

/// Margin spec fixture: two margin rows over the residual DAG's 3 outputs
/// (`y0 − y1`, `y2 − y0`).
fn margin_spec() -> Array2<f32> {
    let mut spec = Array2::<f32>::zeros((2, 3));
    spec[[0, 0]] = 1.0;
    spec[[0, 1]] = -1.0;
    spec[[1, 2]] = 1.0;
    spec[[1, 0]] = -1.0;
    spec
}

/// Assert sampled concrete margin values lie inside the claimed enclosure.
fn assert_margin_mc_containment(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec: &Array2<f32>,
    margin: &BoundedTensor,
    seed: u64,
    label: &str,
) {
    let mut rng = Lcg::new(seed);
    for _ in 0..25 {
        let point = sample_point(&mut rng, input);
        let exact = graph
            .collect_node_bounds_with_engine(&point, None)
            .expect("point eval");
        let out = exact["out"].flatten();
        for row in 0..spec.nrows() {
            let mut val = 0.0f64;
            for col in 0..spec.ncols() {
                let y = f64::midpoint(out.lower()[col] as f64, out.upper()[col] as f64);
                val += spec[[row, col]] as f64 * y;
            }
            let slack = 1e-4 * (1.0 + val.abs());
            let lo = margin.lower()[row] as f64;
            let hi = margin.upper()[row] as f64;
            assert!(
                lo - slack <= val && val <= hi + slack,
                "{label} margin row {row}: value {val} outside [{lo}, {hi}]"
            );
        }
    }
}

/// #w4-root-alpha-opt soundness: the OPTIMIZER's output alphas produce a
/// sound forward-linear map (MC containment at every node) on 3 random conv
/// DAGs, and never a worse surrogate objective than the adaptive start.
#[test]
fn test_alpha_optimizer_output_sound_on_random_dags() {
    let spec = margin_spec();
    let mut engaged = 0usize;
    for seed in [42u64, 43, 77] {
        let (graph, input) = build_residual_dag(seed, 0.6);
        // Warm the fixed cache (the optimizer only runs where the fixed pass ran).
        graph
            .forward_linear_spec_margin_bounds(&input, &spec, None, None)
            .expect("fixed margin warms the cache");
        let (map, output_lb) = graph
            .collect_forward_linear_state_cached(&input, None, None)
            .expect("cached fixed state");
        let output_lb = output_lb.expect("output map retained");

        let outcome = alpha_opt::optimize_margin_alphas(
            &graph, &input, &spec, None, &map, &output_lb, None, None,
        )
        .expect("optimizer must not error");
        let Some((alphas, stats)) = outcome else {
            continue;
        };
        engaged += 1;

        // Improvement invariant: the optimizer starts from the adaptive
        // slopes and only returns configurations its surrogate scores BETTER.
        assert!(
            stats.predicted_min > stats.baseline_min,
            "seed {seed}: predicted {} must beat baseline {}",
            stats.predicted_min,
            stats.baseline_min
        );
        assert!(stats.sweeps >= 1 && stats.moved >= 1);

        // Soundness: the optimizer's alphas produce a sound map...
        assert_mc_containment_with_alphas(&graph, &input, 30, 5200 + seed, Some(&alphas));
        // ...and a sound margin enclosure through the certified rebuild.
        let alpha_margin = graph
            .forward_linear_spec_margin_bounds_with_alphas(&input, &spec, &alphas, None, None)
            .expect("alpha-fed margin");
        assert_margin_mc_containment(
            &graph,
            &input,
            &spec,
            &alpha_margin,
            5300 + seed,
            "optimized",
        );
    }
    assert!(
        engaged >= 1,
        "the optimizer should engage on at least one of the 3 fixtures"
    );
}

/// #w4-root-alpha-opt: the public entry (optimizer + certified rebuild)
/// fail-opens without a warm fixed cache, produces sound margin bounds whose
/// intersection with the fixed route is never worse, and memoizes.
#[test]
fn test_alpha_optimizer_entry_cache_gate_and_never_worse() {
    let spec = margin_spec();
    let (graph, input) = build_residual_dag(42, 0.6);

    // Cold cache: the entry must decline instead of paying the fresh pass.
    let cold = graph
        .forward_linear_alpha_optimized_spec_margin_bounds(&input, &spec, None, None, None)
        .expect("entry must not error on cold cache");
    assert!(cold.is_none(), "cold fixed cache must fail open");

    let fixed = graph
        .forward_linear_spec_margin_bounds(&input, &spec, None, None)
        .expect("fixed margin");
    let first = graph
        .forward_linear_alpha_optimized_spec_margin_bounds(&input, &spec, None, None, None)
        .expect("entry must not error");
    let Some((bounds, stats)) = first else {
        // Optimizer declined (no straggler rows / no predicted improvement) —
        // acceptable; nothing further to assert on this fixture.
        return;
    };
    assert!(stats.predicted_min > stats.baseline_min);
    assert_margin_mc_containment(&graph, &input, &spec, &bounds, 6400, "entry");

    // Intersection with the fixed route is sound and never worse than fixed.
    let (isect, _) = fixed
        .intersection_per_element(&bounds)
        .expect("intersection");
    for row in 0..spec.nrows() {
        assert!(
            isect.lower()[row] >= fixed.lower()[row] && isect.upper()[row] <= fixed.upper()[row],
            "row {row}: intersection must never be worse than fixed"
        );
    }
    assert_margin_mc_containment(&graph, &input, &spec, &isect, 6500, "intersected");

    // Memo: a second call reproduces the same bounds bit-for-bit.
    let second = graph
        .forward_linear_alpha_optimized_spec_margin_bounds(&input, &spec, None, None, None)
        .expect("memoized entry")
        .expect("memo must return the optimized result");
    assert_eq!(bounds.lower(), second.0.lower());
    assert_eq!(bounds.upper(), second.0.upper());
}

// ---------------------------------------------------------------------------
// Sound f32 forward-linear composition seam (NY_FORWARD_LINEAR_F32) oracles.
// ---------------------------------------------------------------------------

/// Wide-magnitude mixed-sign f32 stream: stresses cancellation so the f32
/// accumulation relative error approaches the Higham `γ_k^f32` worst case.
fn f32_seam_stream(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let e = ((s >> 40) % 30) as i32 - 15; // exponent in [-15, 14]
        let mant = ((s >> 12) & 0x7f_ffff) as f32 / (1u32 << 23) as f32; // [0,1)
        let sign = if (s >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        sign * (1.0 + mant) * 2f32.powi(e)
    }
}

/// DECISIVE soundness bound for the value seam: an IEEE round-to-nearest **f32**
/// GEMM (what cuBLAS `Sgemm` does, modeled here by the f32-accumulating
/// `NaiveCpuGemmEngine`) has coefficient error `≤ γ_{k+4}^f32 · S`, the exact
/// factor `compose_conv2d_forward` charges to the bias. If the seam used the f64
/// factor (`≈ 2^29×` smaller) the assertion would fail — proving the swap to
/// `gamma_n_f32` is mandatory, not cosmetic. The `gemm_f64` reference is
/// exact-widened (its own error `≤ γ_k^f64·S ≪ γ_{k+4}^f32·S`).
#[test]
fn test_forward_f32_value_gemm_error_within_gamma_f32_bound() {
    use crate::layers::linear::crown_single_gamma_n_f32 as gamma_n_f32;
    use ny_core::{GemmEngine, NaiveCpuGemmEngine};

    let mut next = f32_seam_stream(0xC0FFEE);
    let (m, n) = (3usize, 2usize);
    for &k in &[1usize, 4, 27, 64, 256, 512, 2304] {
        let a32: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b32: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let a64: Vec<f64> = a32.iter().map(|&x| f64::from(x)).collect();
        let b64: Vec<f64> = b32.iter().map(|&x| f64::from(x)).collect();

        let r32 = NaiveCpuGemmEngine.gemm_f32(m, k, n, &a32, &b32).unwrap();
        let r64 = NaiveCpuGemmEngine.gemm_f64(m, k, n, &a64, &b64).unwrap();
        let gamma = gamma_n_f32(k + 4);

        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64; // S = Σ_k |a|·|b| (nonneg → f64 sum ≥ true·(1−ε))
                for kk in 0..k {
                    s += a64[i * k + kk].abs() * b64[kk * n + j].abs();
                }
                let err = (f64::from(r32[i * n + j]) - r64[i * n + j]).abs();
                let bound = gamma * s * (1.0 + 1e-6); // tiny slack for the f64-ref/S roundings
                assert!(
                    err <= bound,
                    "UNSOUND: f32 GEMM error {err} > γ_{{k+4}}^f32·S = {bound} (k={k}, i={i}, j={j})"
                );
            }
        }
    }
}

/// End-to-end (wiring) soundness: the f32-seam Conv2d composition must produce
/// bounds that CONTAIN the exact affine conv range and are never TIGHTER than
/// the default f64 composition. Driven through `NaiveCpuGemmEngine` with an
/// explicit `use_f32_override` (no env / no process-global — race-free), so the
/// f32 value GEMMs genuinely run and their `γ^f32·S` + FTZ penalty is exercised.
#[test]
fn test_forward_f32_seam_contains_exact_conv_range_and_covers_f64() {
    use crate::bounds::LinearBounds;
    use ny_core::NaiveCpuGemmEngine;

    let mut rng = Lcg::new(4242);
    // in_c=3, 5×5, 3×3 kernel, stride 1, pad 1, out_c=4 → contraction=27.
    let (in_c, in_h, in_w) = (3usize, 5usize, 5usize);
    let conv = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 4, in_c, 3, 3, 1.5),
        Some(random_bias(&mut rng, 4, 0.5)),
        (1, 1),
        (1, 1),
        in_h,
        in_w,
    )
    .expect("conv");
    let input = random_box(&mut rng, &[in_c, in_h, in_w], 1.0, 0.8);
    let in_dim = in_c * in_h * in_w;

    // DENSE point-map upstream `M` (lower_a = upper_a, no radius): the conv GEMM
    // then accumulates over `in_c·9 = 27` dense coefficients per output, so the
    // f32 seam's accumulation error is genuinely present (an identity upstream
    // would make every output a single 1-hot GEMM row with zero accumulation
    // error, and the seam would be indistinguishable from f64).
    let m: Array2<f32> = Array2::from_shape_fn((in_dim, in_dim), |_| rng.next_f32(0.5));
    let zeros = Array1::<f32>::zeros(in_dim);
    let upstream = LinearBounds::new(m.clone(), zeros.clone(), m.clone(), zeros)
        .expect("dense point-map upstream");

    let flat = input.flatten();
    let input_mag: Vec<f64> = flat
        .lower()
        .iter()
        .zip(flat.upper().iter())
        .map(|(&l, &u)| f64::from(l).abs().max(f64::from(u).abs()))
        .collect();

    // Exact composed affine map `y = conv(M·x) = (conv_dense @ M)·x + conv_bias`.
    let dense = dense_conv_oracle(&conv, in_c, in_h, in_w);
    let out_dim = dense.out_dim;
    let mut cm = vec![0.0f64; out_dim * in_dim];
    for p in 0..out_dim {
        for j in 0..in_dim {
            let mut acc = 0.0f64;
            for kk in 0..in_dim {
                acc += dense.matrix[p * in_dim + kk] * f64::from(m[[kk, j]]);
            }
            cm[p * in_dim + j] = acc;
        }
    }
    let (ex_lo, ex_hi) = affine_exact_range(&cm, &dense.bias, out_dim, in_dim, &input);

    let compose = |use_f32: bool| {
        image::compose_conv2d_forward(
            "conv",
            &conv,
            &upstream,
            &[in_c, in_h, in_w],
            out_dim,
            &input_mag,
            Some(&NaiveCpuGemmEngine),
            None,
            Some(use_f32),
        )
        .expect("compose")
        .concretize_sound(&input)
    };
    let f64_bounds = compose(false);
    let f32_bounds = compose(true);

    let (fl, fu) = (f32_bounds.lower(), f32_bounds.upper());
    let (dl, du) = (f64_bounds.lower(), f64_bounds.upper());
    let mut widened = false;
    for p in 0..out_dim {
        // (1) SOUND: the f32 interval must contain the exact affine range.
        assert!(
            (fl[p] as f64) <= ex_lo[p] + 1e-6 && (fu[p] as f64) >= ex_hi[p] - 1e-6,
            "UNSOUND f32 seam: conv[{p}] claimed [{}, {}] excludes exact [{}, {}]",
            fl[p],
            fu[p],
            ex_lo[p],
            ex_hi[p]
        );
        // (2) NEVER TIGHTER than the f64 composition (widens outward).
        assert!(
            (fl[p] as f64) <= (dl[p] as f64) + 1e-6 && (fu[p] as f64) >= (du[p] as f64) - 1e-6,
            "f32 seam TIGHTER than f64 at conv[{p}]: f32 [{}, {}] vs f64 [{}, {}]",
            fl[p],
            fu[p],
            dl[p],
            du[p]
        );
        if (fu[p] - fl[p]) as f64 > (du[p] - dl[p]) as f64 + 1e-12 {
            widened = true;
        }
    }
    // The f32 path genuinely ran (looser somewhere), confirming it was not a
    // silent fall-through to the f64 path.
    assert!(
        widened,
        "f32 seam produced bounds identical to f64 — the f32 path did not engage"
    );
}
