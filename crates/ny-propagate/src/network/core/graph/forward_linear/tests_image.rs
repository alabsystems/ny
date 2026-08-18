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
    TanhLayer,
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
    assert_forward_map_mc_containment(graph, input, samples, seed, &forward);
}

fn assert_forward_map_mc_containment(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    samples: usize,
    seed: u64,
    forward: &HashMap<String, BoundedTensor>,
) {
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
        for (node_name, claimed) in forward {
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
        channel_axis_hint: None,
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
    assert!(
        matches!(err, NyError::InvalidSpec(ref message) if message.contains("BatchNorm")),
        "malformed public BatchNorm metadata must be rejected as an invalid spec, got {err:?}"
    );
}

/// Kill-switch compatibility for the default-on certified cGAN surface:
/// explicitly disabling ConvTranspose composition keeps ConvTranspose-only
/// graphs on the pre-existing generic refusal, while a Conv2d image graph
/// containing BatchNorm still fails at the old image allowlist.
#[test]
fn test_image_conv_transpose_kill_switch_preserves_legacy_routing() {
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
        "kill-switched ConvTranspose-only graph must retain the generic legacy refusal, got: {message}"
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
        .expect_err("kill-switched Conv2d+BatchNorm must retain old fail-closed image surface");
    assert!(matches!(err, NyError::UnsupportedConfiguration(_)));
}

#[test]
fn test_alpha_reference_bounds_use_forward_linear_for_conv_dag() {
    let (graph, input) = build_residual_dag(42, 0.5);
    let exec_order = graph.exec_order().expect("exec order").to_vec();
    crate::tests::with_env_edits(|env| {
        env.remove("NY_NO_FORWARD_LINEAR_REF");
        env.set("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "1");
        assert!(
            graph.should_collect_forward_linear_intermediate_reference(),
            "historical non-sequential Conv2d route must remain enabled"
        );
        assert!(
            graph.should_collect_forward_linear_image_reference(),
            "Conv2d DAG must remain image-eligible independently of the ConvTranspose gate"
        );

        let config = AlphaCrownConfig {
            fix_interm_bounds: true,
            ..AlphaCrownConfig::default()
        };
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
    });
}

/// #cgan-fwdlin-ref: a cgan-class SEQUENTIAL ConvTranspose chain (is_dag =
/// false) never reached the conv-DAG forward-linear branch, so the certified
/// ConvTranspose/BatchNorm surface was unreachable exactly on the graphs it
/// was built for. The default image-reference policy must now serve the
/// forward-linear map for such chains; either disable flag restores fallback.
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

    crate::tests::with_env_edits(|env| {
        env.remove("NY_NO_FORWARD_LINEAR_REF");
        env.remove("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF");
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
    });
}

#[test]
fn test_typed_tanh_diagonal_grid_encloses_endpoints_zero_and_crossing() {
    let upstream = LinearBounds::identity(1);
    for (label, lower, upper) in [
        ("negative", -3.0_f32, -0.125_f32),
        ("positive", 0.125_f32, 3.0_f32),
        ("crossing", -2.75_f32, 3.25_f32),
    ] {
        let input = BoundedTensor::new(
            Array1::from_vec(vec![lower]).into_dyn(),
            Array1::from_vec(vec![upper]).into_dyn(),
        )
        .expect("scalar pre-activation box");
        let input_mag = [f64::from(lower).abs().max(f64::from(upper).abs())];
        let composed = image::compose_tanh_diag_forward(
            &format!("tanh-{label}"),
            &upstream,
            &input,
            &input_mag,
        )
        .expect("certified diagonal Tanh composition");
        let concrete = composed
            .concretize_checked(&input)
            .expect("concretized Tanh row");
        assert!(
            concrete.lower()[0].is_finite() && concrete.upper()[0].is_finite(),
            "{label}: finite scalar input must retain finite Tanh enclosure"
        );

        let mut grid: Vec<f64> = (0..=1024)
            .map(|index| {
                f64::from(lower) + (f64::from(upper) - f64::from(lower)) * index as f64 / 1024.0
            })
            .collect();
        // Explicitly retain the exact activation endpoints and inflection point;
        // the evenly-spaced grid is not relied on to happen to hit zero.
        grid.extend([f64::from(lower), f64::from(upper)]);
        if lower <= 0.0 && upper >= 0.0 {
            grid.push(0.0);
        }
        for x in grid {
            let row_lower =
                f64::from(composed.lower_a()[[0, 0]]) * x + f64::from(composed.lower_b()[0]);
            let row_upper =
                f64::from(composed.upper_a()[[0, 0]]) * x + f64::from(composed.upper_b()[0]);
            let actual = x.tanh();
            assert!(
                row_lower <= actual && actual <= row_upper,
                "{label}: x={x} tanh={actual} escaped row [{row_lower}, {row_upper}]"
            );
            assert!(
                f64::from(concrete.lower()[0]) <= actual
                    && actual <= f64::from(concrete.upper()[0]),
                "{label}: x={x} tanh={actual} escaped concretized [{}, {}]",
                concrete.lower()[0],
                concrete.upper()[0]
            );
        }
    }
}

#[test]
fn test_typed_tanh_diagonal_large_cancellation_discharges_cast_gap() {
    let c0 = 8_388_607.0_f32;
    let c1 = -8_388_605.0_f32;
    let bias = 0.03125_f32;
    let coefficients = Array2::from_shape_vec((1, 2), vec![c0, c1]).expect("coefficient row");
    let biases = Array1::from_vec(vec![bias]);
    let upstream = LinearBounds::new(coefficients.clone(), biases.clone(), coefficients, biases)
        .expect("exact large-cancellation upstream row");
    let radius = 2.0e-7_f32;
    let input = BoundedTensor::new(
        Array1::from_vec(vec![-radius, -radius]).into_dyn(),
        Array1::from_vec(vec![radius, radius]).into_dyn(),
    )
    .expect("non-point cancellation box");
    assert!(
        input
            .lower()
            .iter()
            .zip(input.upper().iter())
            .all(|(&lower, &upper)| lower < upper),
        "the cast-gap regression must exercise a non-point input"
    );
    let pre_activation = upstream
        .concretize_checked(&input)
        .expect("certified upstream pre-activation");
    let input_mag = [f64::from(radius), f64::from(radius)];
    let relax = crate::layers::trigonometric::tanh_linear_relaxation(
        pre_activation.lower()[0],
        pre_activation.upper()[0],
    );
    let has_measured_cast_gap = [relax.lower_slope, relax.upper_slope]
        .into_iter()
        .flat_map(|slope| [c0, c1].map(|coefficient| (slope, coefficient)))
        .any(|(slope, coefficient)| {
            let exact = f64::from(slope) * f64::from(coefficient);
            exact.is_finite() && f64::from(exact as f32) != exact
        });
    assert!(
        has_measured_cast_gap,
        "fixture must force a nonzero f64-to-f32 coefficient cast gap"
    );

    let composed = image::compose_tanh_diag_forward(
        "tanh-large-cancellation",
        &upstream,
        &pre_activation,
        &input_mag,
    )
    .expect("large-cancellation Tanh composition");
    assert!(
        composed
            .lower_a()
            .iter()
            .chain(composed.upper_a().iter())
            .chain(composed.lower_b().iter())
            .chain(composed.upper_b().iter())
            .all(|value| value.is_finite()),
        "cast-gap discharge must remain finite"
    );
    let concrete = composed
        .concretize_checked(&input)
        .expect("finite cancellation enclosure");
    assert!(concrete.lower()[0].is_finite() && concrete.upper()[0].is_finite());

    for i in 0..=40 {
        let x0 = -f64::from(radius) + 2.0 * f64::from(radius) * i as f64 / 40.0;
        for j in 0..=40 {
            let x1 = -f64::from(radius) + 2.0 * f64::from(radius) * j as f64 / 40.0;
            let pre = f64::from(c0) * x0 + f64::from(c1) * x1 + f64::from(bias);
            let actual = pre.tanh();
            let row_lower = f64::from(composed.lower_a()[[0, 0]]) * x0
                + f64::from(composed.lower_a()[[0, 1]]) * x1
                + f64::from(composed.lower_b()[0]);
            let row_upper = f64::from(composed.upper_a()[[0, 0]]) * x0
                + f64::from(composed.upper_a()[[0, 1]]) * x1
                + f64::from(composed.upper_b()[0]);
            assert!(
                row_lower <= actual && actual <= row_upper,
                "x=({x0},{x1}) tanh={actual} escaped row [{row_lower},{row_upper}]"
            );
            assert!(
                f64::from(concrete.lower()[0]) <= actual
                    && actual <= f64::from(concrete.upper()[0]),
                "x=({x0},{x1}) tanh={actual} escaped concretized [{},{}]",
                concrete.lower()[0],
                concrete.upper()[0]
            );
        }
    }
}

#[test]
fn test_image_forward_linear_tanh_sampled_enclosure() {
    let pre = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![1.25_f32, -0.8])
            .expect("pre-Tanh kernel"),
        Some(Array1::from_vec(vec![-0.1_f32, 0.2])),
        (1, 1),
        (0, 0),
        2,
        2,
    )
    .expect("pre-Tanh conv");
    let post = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 1, 1]), vec![-1.1_f32, 0.7])
            .expect("post-Tanh kernel"),
        Some(Array1::from_vec(vec![0.03_f32])),
        (1, 1),
        (0, 0),
        2,
        2,
    )
    .expect("post-Tanh conv");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("pre", Layer::Conv2d(pre)));
    graph.add_node(GraphNode::new(
        "tanh",
        Layer::Tanh(TanhLayer),
        vec!["pre".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "post",
        Layer::Conv2d(post),
        vec!["tanh".to_string()],
    ));
    graph.set_output("post");
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-1.0, -0.4, 0.2, -0.7])
            .expect("input lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.8, 0.9, 1.1, 0.5]).expect("input upper"),
    )
    .expect("input box");

    let ordinary_error = graph
        .collect_forward_linear_bounds_dag_cached(&input, None, None)
        .expect_err("ordinary image requests must retain the historical Tanh refusal");
    assert!(matches!(
        ordinary_error,
        NyError::UnsupportedConfiguration(_)
    ));
    let spoof_config = AlphaCrownConfig {
        fix_interm_bounds: true,
        cgan_complete_crown_ibp_root: true,
        ..AlphaCrownConfig::default()
    };
    let spoof_error = graph
        .collect_forward_linear_bounds_dag_cached_for_typed_cgan(&input, &spoof_config, None, None)
        .expect_err("an arbitrary Conv2d+Tanh graph must not claim typed cGAN authority");
    assert!(matches!(spoof_error, NyError::UnsupportedConfiguration(_)));
    let (forward, _) =
        collect_forward_linear_state_dag(&graph, &input, None, None, None, false, true)
            .expect("internal typed Tanh composition should succeed for enclosure testing");
    let tanh = forward.get("tanh").expect("Tanh map entry");
    assert!(
        tanh.lower().iter().all(|value| value.is_finite())
            && tanh.upper().iter().all(|value| value.is_finite()),
        "finite pre-activation bounds must produce finite Tanh bounds"
    );
    assert_forward_map_mc_containment(&graph, &input, 64, 0x7A4A_2026, &forward);
}

/// The ConvTranspose surface salts the fixed-cache identity. The warmer and
/// alpha optimizer must consult the same key; otherwise official cGAN warms a
/// map that the candidate can never observe.
#[test]
fn test_conv_transpose_warm_cache_reaches_typed_alpha_entry() {
    crate::tests::with_serialized_env_vars(
        &[("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "0")],
        || {
            let convt = ConvTranspose2dLayer::with_input_shape(
                ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 0.75),
                Some(Array1::from_vec(vec![0.1])),
                (1, 1),
                (0, 0),
                1,
                1,
            )
            .expect("tiny ConvTranspose");
            let mut graph = GraphNetwork::new();
            graph.add_node(GraphNode::from_input(
                "convt",
                Layer::ConvTranspose2d(convt),
            ));
            graph.set_output("convt");
            let input = BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), -0.4),
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.6),
            )
            .expect("input box");

            // The typed setter invalidates every forward-linear cache. Arm the
            // lane through the legacy cGAN alias before warming, so this test
            // also proves that old callers reach the generic authority check.
            graph.set_cgan_forward_alpha_surrogate(true);
            assert!(graph.forward_linear_spec_alpha_enabled());
            assert!(graph.forward_linear_fixed_state_if_cached(&input).is_none());
            graph
                .collect_forward_linear_state_cached(&input, None, None)
                .expect("ConvTranspose fixed map warm");
            assert!(
                graph.forward_linear_fixed_state_if_cached(&input).is_some(),
                "the lookup must include the same ConvTranspose policy salt as the warmer"
            );

            let spec = Array2::from_shape_vec((2, 1), vec![1.0, -1.0]).expect("two-row spec");
            let outcome = graph
                .forward_linear_alpha_optimized_spec_margin_bounds(&input, &spec, None, None, None)
                .expect("typed alpha entry");
            assert!(
                outcome.is_none(),
                "no-ReLU fixture should decline after admission"
            );
            assert!(
                graph
                    .cached_forward_linear_map
                    .alpha_opt
                    .read()
                    .expect("alpha-opt cache lock")
                    .is_some(),
                "memoized decline proves the typed entry passed the warm-cache gate"
            );
            let (memo_hash, memo_canonical) = {
                let guard = graph
                    .cached_forward_linear_map
                    .alpha_opt
                    .read()
                    .expect("alpha-opt cache lock");
                let (fingerprint, value) = guard.as_ref().expect("memoized decline");
                assert!(value.is_none());
                (fingerprint.hash, fingerprint.canonical.clone())
            };
            assert!(graph
                .forward_linear_alpha_optimized_spec_margin_bounds(&input, &spec, None, None, None,)
                .expect("memoized decline hit")
                .is_none());
            let guard = graph
                .cached_forward_linear_map
                .alpha_opt
                .read()
                .expect("alpha-opt cache lock");
            let (fingerprint, value) = guard.as_ref().expect("republished decline");
            assert!(value.is_none());
            assert_eq!(fingerprint.hash, memo_hash);
            assert_eq!(fingerprint.canonical, memo_canonical);
            drop(guard);

            graph.set_forward_linear_spec_alpha(false);
            assert!(
                graph
                    .cached_forward_linear_map
                    .alpha_opt
                    .read()
                    .expect("alpha-opt cache lock")
                    .is_none(),
                "disarming the typed lane must invalidate a memoized decline"
            );
        },
    );
}

/// Proof-cache hits require exact canonical equality after the hash
/// accelerator. Same endpoint bits under a different shape and a deliberately
/// forced u64 collision must both miss.
#[test]
fn test_forward_linear_cache_rejects_shape_alias_and_forced_hash_collision() {
    let lower = vec![-1.0_f32, -0.5, 0.25, 0.75];
    let upper = vec![1.0_f32, 0.5, 1.25, 1.75];
    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), lower.clone()).expect("shape a lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), upper.clone()).expect("shape a upper"),
    )
    .expect("shape a box");
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), lower).expect("shape b lower"),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), upper).expect("shape b upper"),
    )
    .expect("shape b box");

    let policy = GraphNetwork::forward_linear_conv_transpose_reference_enabled();
    let fingerprint_a = forward_linear_cache_fingerprint(&a, None, policy, false);
    let fingerprint_b = forward_linear_cache_fingerprint(&b, None, policy, false);
    assert_ne!(
        fingerprint_a.canonical, fingerprint_b.canonical,
        "shape is part of exact proof-cache authority"
    );
    assert_ne!(
        forward_linear_cache_fingerprint(&a, None, !policy, false).canonical,
        fingerprint_a.canonical,
        "ConvTranspose policy is part of exact proof-cache authority"
    );

    let graph = GraphNetwork::new();
    *graph
        .cached_forward_linear_map
        .fixed
        .write()
        .expect("fixed cache lock") = Some(ForwardLinearCacheEntry {
        fingerprint: fingerprint_a,
        map: std::sync::Arc::new(HashMap::new()),
        output_lb: None,
        build_cost: std::time::Duration::ZERO,
    });
    assert!(graph
        .forward_linear_fixed_state_if_cached_with_policy(&a, policy)
        .is_some());
    let mut pure_clone = graph.clone();
    pure_clone.adopt_forward_linear_cache_from(&graph);
    assert!(
        pure_clone
            .forward_linear_fixed_state_if_cached_with_policy(&a, policy)
            .is_some(),
        "a pure clone sharing graph scope may adopt the exact forward cache"
    );
    let mut foreign_graph = GraphNetwork::new();
    foreign_graph.adopt_forward_linear_cache_from(&graph);
    assert!(
        foreign_graph
            .forward_linear_fixed_state_if_cached_with_policy(&a, policy)
            .is_none(),
        "a separately constructed graph must reject forward-cache adoption"
    );
    assert!(
        graph
            .forward_linear_fixed_state_if_cached_with_policy(&b, policy)
            .is_none(),
        "same flat bits under a different shape must miss"
    );

    // Force the accelerator to collide with request B while retaining A's
    // canonical bytes. Exact comparison must still reject the entry.
    graph
        .cached_forward_linear_map
        .fixed
        .write()
        .expect("fixed cache lock")
        .as_mut()
        .expect("seeded entry")
        .fingerprint
        .hash = fingerprint_b.hash;
    assert!(
        graph
            .forward_linear_fixed_state_if_cached_with_policy(&b, policy)
            .is_none(),
        "u64 equality alone must never authorize a proof-cache hit"
    );
}

/// Alpha names and payloads are sorted and length-delimited in the exact
/// fingerprint, so ambiguous concatenations cannot alias.
#[test]
fn test_alpha_cache_fingerprint_is_sorted_and_length_delimited() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -1.0),
        ArrayD::from_elem(IxDyn(&[1]), 1.0),
    )
    .expect("input");
    let mut left = std::collections::BTreeMap::new();
    left.insert("a".to_string(), Array1::from_vec(vec![0.25]));
    left.insert("bc".to_string(), Array1::from_vec(vec![0.75]));
    let mut right = std::collections::BTreeMap::new();
    right.insert("ab".to_string(), Array1::from_vec(vec![0.25]));
    right.insert("c".to_string(), Array1::from_vec(vec![0.75]));
    let mut reverse_insert = std::collections::BTreeMap::new();
    reverse_insert.insert("bc".to_string(), Array1::from_vec(vec![0.75]));
    reverse_insert.insert("a".to_string(), Array1::from_vec(vec![0.25]));

    let policy = GraphNetwork::forward_linear_conv_transpose_reference_enabled();
    let left_fingerprint = forward_linear_cache_fingerprint(&input, Some(&left), policy, false);
    let right_fingerprint = forward_linear_cache_fingerprint(&input, Some(&right), policy, false);
    let reverse_fingerprint =
        forward_linear_cache_fingerprint(&input, Some(&reverse_insert), policy, false);
    assert_eq!(
        left_fingerprint.canonical, reverse_fingerprint.canonical,
        "BTreeMap order makes alpha fingerprint canonical"
    );
    assert_ne!(left_fingerprint.canonical, right_fingerprint.canonical);

    let graph = GraphNetwork::new();
    *graph
        .cached_forward_linear_map
        .alpha
        .write()
        .expect("alpha cache lock") = Some(ForwardLinearCacheEntry {
        fingerprint: left_fingerprint,
        map: std::sync::Arc::new(HashMap::new()),
        output_lb: None,
        build_cost: std::time::Duration::ZERO,
    });
    graph
        .cached_forward_linear_map
        .alpha
        .write()
        .expect("alpha cache lock")
        .as_mut()
        .expect("seeded entry")
        .fingerprint
        .hash = right_fingerprint.hash;
    assert!(
        graph
            .forward_linear_alpha_state_if_cached_with_policy(&input, &right, policy)
            .is_none(),
        "forced alpha hash collision must miss on canonical bytes"
    );
}

#[test]
fn test_alpha_memo_fingerprint_discriminates_incumbent_and_policy() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0),
        ArrayD::from_elem(IxDyn(&[2]), 1.0),
    )
    .expect("input");
    let spec = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("spec");
    let incumbent = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.5, -0.25]).expect("incumbent lower"),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.75]).expect("incumbent upper"),
    )
    .expect("incumbent");
    let mut changed_lower = incumbent.lower().clone();
    let changed_bit = ny_tensor::next_up_f32(changed_lower[0]);
    changed_lower.as_slice_mut().expect("contiguous")[0] = changed_bit;
    let changed =
        BoundedTensor::new(changed_lower, incumbent.upper().clone()).expect("changed incumbent");

    let base = margin_opt_memo_fingerprint(
        &input,
        &spec,
        Some(&incumbent),
        true,
        true,
        None,
        alpha_opt::MAX_SURROGATE_BYTES,
    )
    .expect("fingerprint build")
    .expect("fingerprint admitted");
    let mut changed_fingerprint = margin_opt_memo_fingerprint(
        &input,
        &spec,
        Some(&changed),
        true,
        true,
        None,
        alpha_opt::MAX_SURROGATE_BYTES,
    )
    .expect("fingerprint build")
    .expect("fingerprint admitted");
    assert_ne!(
        base.canonical, changed_fingerprint.canonical,
        "incumbent endpoint bits affect optimizer selection and memo identity"
    );
    changed_fingerprint.hash = base.hash;
    assert!(
        !base.exact_match(&changed_fingerprint),
        "a forced hash collision must still miss on exact canonical bytes"
    );
    assert_ne!(
        base.canonical,
        margin_opt_memo_fingerprint(
            &input,
            &spec,
            Some(&incumbent),
            false,
            true,
            None,
            alpha_opt::MAX_SURROGATE_BYTES,
        )
        .expect("fingerprint build")
        .expect("fingerprint admitted")
        .canonical,
        "ConvTranspose policy must discriminate optimizer memos"
    );
    assert_ne!(
        base.canonical,
        margin_opt_memo_fingerprint(
            &input,
            &spec,
            Some(&incumbent),
            true,
            false,
            None,
            alpha_opt::MAX_SURROGATE_BYTES,
        )
        .expect("fingerprint build")
        .expect("fingerprint admitted")
        .canonical,
        "typed candidate policy must discriminate optimizer memos"
    );
}

#[test]
fn test_alpha_memo_fingerprint_refuses_oversized_and_overflowed_layouts() {
    let oversized_spec_len = alpha_opt::MAX_SURROGATE_BYTES / size_of::<f32>() + 1;
    assert!(
        margin_opt_fingerprint_layout(
            1,
            1,
            1,
            1,
            oversized_spec_len,
            oversized_spec_len,
            None,
            alpha_opt::MAX_SURROGATE_BYTES,
        )
        .is_none(),
        "an oversized spec must be rejected from scalar lengths before allocation or scanning"
    );
    assert!(
        margin_opt_fingerprint_layout(
            usize::MAX,
            usize::MAX,
            usize::MAX,
            1,
            1,
            1,
            Some((usize::MAX, usize::MAX, usize::MAX)),
            alpha_opt::MAX_SURROGATE_BYTES,
        )
        .is_none(),
        "checked byte arithmetic must reject overflow"
    );
    assert!(
        margin_opt_fingerprint_layout(1, 1, 1, usize::MAX, 2, usize::MAX, None, usize::MAX)
            .is_none(),
        "checked spec-shape work must reject multiplication overflow"
    );
}

#[test]
fn test_alpha_memo_fingerprint_cap_and_mid_scan_deadline_are_retryable() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0),
        ArrayD::from_elem(IxDyn(&[2]), 1.0),
    )
    .expect("input");
    let columns = alpha_opt::DEADLINE_POLL_WORK as usize + 1;
    let spec = Array2::from_shape_vec(
        (2, columns),
        (0..2 * columns).map(|index| index as f32 * 0.25).collect(),
    )
    .expect("spec");
    let admitted = margin_opt_memo_fingerprint(
        &input,
        &spec,
        None,
        true,
        true,
        None,
        alpha_opt::MAX_SURROGATE_BYTES,
    )
    .expect("fingerprint build")
    .expect("fingerprint admitted");
    assert!(admitted.retained_bytes() >= admitted.canonical.len());

    let expired = margin_opt_memo_fingerprint(
        &input,
        &spec,
        None,
        true,
        true,
        Some(Instant::now()),
        alpha_opt::MAX_SURROGATE_BYTES,
    )
    .expect_err("an already-expired deadline must stop before allocation");
    assert!(matches!(expired, NyError::DeadlineExceeded(_)));

    let capped = margin_opt_memo_fingerprint_with(
        &input,
        &spec,
        None,
        true,
        true,
        admitted.retained_bytes() - 1,
        |_| Ok(()),
    )
    .expect("cap refusal is not an error");
    assert!(
        capped.is_none(),
        "one byte under the exact plan must refuse"
    );

    let incumbent_bytes = alpha_opt::MAX_SURROGATE_BYTES
        .checked_sub(admitted.retained_bytes())
        .expect("ordinary fingerprint is below cap")
        + 1;
    let replacement_budget = alpha_opt::MAX_SURROGATE_BYTES.saturating_sub(incumbent_bytes);
    assert!(
        margin_opt_memo_fingerprint_with(
            &input,
            &spec,
            None,
            true,
            true,
            replacement_budget,
            |_| Ok(()),
        )
        .expect("aggregate incumbent/replacement cap refusal")
        .is_none(),
        "the incumbent and replacement canonical Vecs must share one 256 MiB envelope"
    );

    let mut payload_polls = 0usize;
    let deadline = margin_opt_memo_fingerprint_with(
        &input,
        &spec,
        None,
        true,
        true,
        alpha_opt::MAX_SURROGATE_BYTES,
        |context| {
            if context == "optimizer memo payload fingerprint" {
                payload_polls += 1;
                if payload_polls == 2 {
                    return Err(NyError::DeadlineExceeded(
                        "injected optimizer memo fingerprint deadline".to_string(),
                    ));
                }
            }
            Ok(())
        },
    )
    .expect_err("the second bounded payload poll must cancel fingerprinting");
    assert!(matches!(deadline, NyError::DeadlineExceeded(_)));
    assert_eq!(payload_polls, 2);

    let retry = margin_opt_memo_fingerprint(
        &input,
        &spec,
        None,
        true,
        true,
        None,
        alpha_opt::MAX_SURROGATE_BYTES,
    )
    .expect("retry fingerprint build")
    .expect("retry fingerprint admitted");
    assert_eq!(
        retry.canonical, admitted.canonical,
        "a cap/deadline refusal must leave no state that poisons an ordinary retry"
    );
}

/// An expired deadline must terminate the certified ConvTranspose path without
/// publishing partial state. A later unbudgeted retry must be able to compute
/// and publish the same request.
#[test]
fn test_conv_transpose_expired_deadline_does_not_publish_and_retry_succeeds() {
    crate::tests::with_serialized_env_vars(
        &[("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "0")],
        || {
            let convt = ConvTranspose2dLayer::with_input_shape(
                ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 0.75),
                Some(Array1::from_vec(vec![0.1])),
                (1, 1),
                (0, 0),
                1,
                1,
            )
            .expect("tiny ConvTranspose");
            let mut graph = GraphNetwork::new();
            graph.add_node(GraphNode::from_input(
                "convt",
                Layer::ConvTranspose2d(convt),
            ));
            graph.set_output("convt");
            let input = BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), -0.4),
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.6),
            )
            .expect("input box");

            let expired = Some(Instant::now());
            let err = graph
                .collect_forward_linear_state_cached(&input, None, expired)
                .expect_err("expired build must cancel");
            assert!(matches!(err, NyError::DeadlineExceeded(_)));
            assert!(
                graph
                    .cached_forward_linear_map
                    .fixed
                    .read()
                    .expect("fixed cache lock")
                    .is_none(),
                "cancelled work must not publish a partial proof cache"
            );

            graph
                .collect_forward_linear_state_cached(&input, None, None)
                .expect("larger-budget retry");
            assert!(
                graph.forward_linear_fixed_state_if_cached(&input).is_some(),
                "deadline refusal must not poison a later retry"
            );
        },
    );
}

/// The deadline reaches the ConvTranspose composition itself, not merely the
/// outer graph loop.
#[test]
fn test_conv_transpose_composition_expired_deadline_then_retry() {
    let convt = ConvTranspose2dLayer::with_input_shape(
        ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 0.75),
        Some(Array1::from_vec(vec![0.1])),
        (1, 1),
        (0, 0),
        1,
        1,
    )
    .expect("tiny ConvTranspose");
    let upstream = LinearBounds::identity(1);
    let err = image::compose_conv_transpose2d_forward(
        "convt",
        &convt,
        &upstream,
        &[1, 1, 1],
        1,
        &[1.0],
        None,
        Some(Instant::now()),
    )
    .expect_err("expired composition must cancel");
    assert!(matches!(err, NyError::DeadlineExceeded(_)));
    let _ = image::compose_conv_transpose2d_forward(
        "convt",
        &convt,
        &upstream,
        &[1, 1, 1],
        1,
        &[1.0],
        None,
        None,
    )
    .expect("unbudgeted composition retry");
}

/// The final coefficient validation/repair pass is itself interruptible. This
/// uses an injected poller so cancellation after one full work quantum is
/// deterministic rather than dependent on wall-clock scheduling.
#[test]
fn test_conv_transpose_tail_validation_polls_and_cancels() {
    let columns = image::CONV_TRANSPOSE_DEADLINE_POLL_WORK + 1;
    let lower_a = Array2::<f32>::zeros((1, columns));
    let upper_a = Array2::<f32>::zeros((1, columns));
    let lower_b = Array1::<f32>::zeros(1);
    let upper_b = Array1::<f32>::zeros(1);
    let mut polls = 0usize;
    let err = image::finish_conv_transpose_bounds_with_poll(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        columns,
        "tail-poll test",
        |_| {
            polls += 1;
            if polls == 2 {
                Err(NyError::DeadlineExceeded(
                    "injected ConvTranspose tail deadline".into(),
                ))
            } else {
                Ok(())
            }
        },
    )
    .expect_err("the second bounded-work poll must cancel validation");
    assert!(matches!(err, NyError::DeadlineExceeded(_)));
    assert_eq!(
        polls, 2,
        "validation must poll again after at most one work quantum"
    );
}

#[test]
fn test_timeout_derived_alpha_decline_remains_retryable() {
    let now = Instant::now();
    assert!(
        !alpha_decline_is_memoizable(Some(now), now),
        "a timeout-derived None must leave the memo cold for a later retry"
    );
    assert!(
        alpha_decline_is_memoizable(Some(now + std::time::Duration::from_secs(1)), now),
        "a budget-independent decline may still be memoized"
    );
    assert!(alpha_decline_is_memoizable(None, now));
}

#[test]
fn test_alpha_optimizer_deadline_is_not_a_memoizable_decline() {
    let spec = margin_spec();
    let (graph, input) = build_residual_dag(42, 0.6);
    let (map, output_lb) = graph
        .collect_forward_linear_state_cached(&input, None, None)
        .expect("warm fixed state");
    let output_lb = output_lb.expect("retained output map");
    let err = alpha_opt::optimize_margin_alphas(
        &graph,
        &input,
        &spec,
        None,
        &map,
        &output_lb,
        None,
        Some(Instant::now()),
        0,
    )
    .expect_err("expired optimizer budget must remain a structured deadline");
    assert!(matches!(err, NyError::DeadlineExceeded(_)));
    assert!(
        graph
            .cached_forward_linear_map
            .alpha_opt
            .read()
            .expect("memo lock")
            .is_none(),
        "a deadline error must leave the outer memo cold for a later retry"
    );
}

#[test]
fn test_cold_memo_rebuild_requires_measured_reserve_before_work() {
    let now = Instant::now();
    let reserve = std::time::Duration::from_secs(3);
    assert!(!alpha_rebuild_fits(
        Some(
            (now + reserve)
                .checked_sub(std::time::Duration::from_nanos(1))
                .expect("now + 3s is well past the monotonic origin, so 1ns cannot underflow"),
        ),
        now,
        reserve,
        std::time::Duration::ZERO,
    ));
    assert!(alpha_rebuild_fits(
        Some(now + reserve),
        now,
        reserve,
        std::time::Duration::ZERO,
    ));
    assert!(alpha_rebuild_fits(
        None,
        now,
        reserve,
        std::time::Duration::from_secs(10),
    ));
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

/// ConvTranspose2d + mixed-sign BatchNorm fixture for the alpha surrogate's
/// forward/adjoint finite-difference test. Geometry deliberately exercises
/// asymmetric stride/dilation and nonzero output_padding.
pub(super) fn build_convt_signed_bn_for_grad_test() -> (GraphNetwork, BoundedTensor) {
    let convt = ConvTranspose2dLayer::new_full(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 2, 2]),
            vec![0.8, -0.4, 0.3, 0.6, -0.5, 0.7, -0.2, 0.9],
        )
        .expect("first ConvTranspose kernel"),
        Some(Array1::from_vec(vec![0.05, -0.1])),
        (2, 1),
        (1, 0),
        (2, 1),
        (1, 0),
    )
    .expect("valid asymmetric ConvTranspose geometry");
    let batch_norm = BatchNormLayer::from_scale_bias(
        Array1::from_vec(vec![-1.25, 0.75]).into_dyn(),
        Array1::from_vec(vec![0.15, -0.1]).into_dyn(),
    )
    .expect("mixed-sign BatchNorm");
    let tail = ConvTranspose2dLayer::new_full(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![0.7, -0.6])
            .expect("tail ConvTranspose kernel"),
        Some(Array1::from_vec(vec![0.02])),
        (1, 1),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .expect("tail ConvTranspose");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "convt",
        Layer::ConvTranspose2d(convt),
    ));
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(batch_norm),
        vec!["convt".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["bn".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "tail",
        Layer::ConvTranspose2d(tail),
        vec!["relu".to_string()],
    ));
    graph.set_output("tail");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0),
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0),
    )
    .expect("valid symmetric input box");
    (graph, input)
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
            &graph, &input, &spec, None, &map, &output_lb, None, None, 0,
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

/// #w4-root-alpha-opt: the generic public entry (optimizer + certified rebuild)
/// reaches the existing method on a small CIFAR-like residual DAG, fail-opens
/// without a warm fixed cache, produces sound margin bounds whose intersection
/// with the fixed route is never worse, and memoizes.
#[test]
fn test_generic_alpha_optimizer_reaches_non_cgan_dag_and_is_never_worse() {
    let spec = margin_spec();
    let (mut graph, input) = build_residual_dag(42, 0.6);
    assert!(graph.has_conv2d_layers());
    assert!(!graph.has_conv_transpose2d_layers());

    // Typed lane is default OFF. Even a direct internal call must be inert and
    // must not publish a memoized refusal that could leak into a later canary.
    assert!(!graph.forward_linear_spec_alpha_enabled());
    assert!(
        !graph.cgan_forward_alpha_surrogate_enabled(),
        "the compatibility query must read the same default-dark bit"
    );
    let dark = graph
        .forward_linear_alpha_optimized_spec_margin_bounds(&input, &spec, None, None, None)
        .expect("dark entry must not error");
    assert!(dark.is_none());
    assert!(
        graph
            .cached_forward_linear_map
            .alpha_opt
            .read()
            .expect("alpha-opt cache lock")
            .is_none(),
        "dark lane must not publish optimizer state"
    );
    graph.set_forward_linear_spec_alpha(true);
    assert!(graph.cgan_forward_alpha_surrogate_enabled());

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
    assert!(
        graph
            .cached_forward_linear_map
            .alpha_opt
            .read()
            .expect("alpha-opt cache lock")
            .is_some(),
        "the generic opt-in must reach the existing optimizer method on the non-cGAN DAG"
    );
    let (bounds, stats) = first.expect(
        "the advertised generic optimizer fixture must engage and publish an improved candidate",
    );
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

    graph.set_forward_linear_spec_alpha(false);
    assert!(
        graph
            .cached_forward_linear_map
            .alpha_opt
            .read()
            .expect("alpha-opt cache lock")
            .is_none(),
        "changing the typed lane must invalidate its memo identity"
    );
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

/// Dense f64 product `left · right` of two affine oracles (`left` consumes
/// `right`'s output), so a ConvTranspose CHAIN has an exact composed matrix.
fn dense_compose(left: &DenseConv, right: &DenseConv) -> DenseConv {
    assert_eq!(left.in_dim, right.out_dim);
    let mut matrix = vec![0.0f64; left.out_dim * right.in_dim];
    let mut bias = left.bias.clone();
    for p in 0..left.out_dim {
        for k in 0..left.in_dim {
            let m = left.matrix[p * left.in_dim + k];
            if m == 0.0 {
                continue;
            }
            bias[p] += m * right.bias[k];
            for j in 0..right.in_dim {
                matrix[p * right.in_dim + j] += m * right.matrix[k * right.in_dim + j];
            }
        }
    }
    DenseConv {
        matrix,
        bias,
        out_dim: left.out_dim,
        in_dim: right.in_dim,
    }
}

/// cGAN-shaped chain fixture: `ConvTranspose → BatchNorm → ReLU` blocks ending
/// in a ConvTranspose. Half the BatchNorm channels are shifted negative so a
/// large fraction of each ReLU is stably INACTIVE (the case that manufactures a
/// binary32-subnormal affine bias, see the `compose_conv_transpose2d_forward`
/// docs) while the rest stay live, so the output is not a constant.
fn build_conv_transpose_relu_chain(
    seed: u64,
    blocks: usize,
    bn_shift: f32,
) -> (GraphNetwork, BoundedTensor) {
    let mut rng = Lcg::new(seed);
    let in_c = 2usize;
    let (in_h, in_w) = (3usize, 3usize);
    let mut graph = GraphNetwork::new();
    let mut prev: Option<String> = None;
    for block in 0..blocks {
        let out_c = 3 + (block % 2);
        // ConvTranspose kernel layout is (in_c, out_c, kh, kw).
        let channels = if block == 0 {
            in_c
        } else {
            3 + ((block - 1) % 2)
        };
        let kernel = random_kernel(&mut rng, channels, out_c, 3, 3, 0.6);
        let convt = ConvTranspose2dLayer::new_full(
            kernel,
            Some(random_bias(&mut rng, out_c, 0.2)),
            (1, 1),
            (0, 0),
            (1, 1),
            (0, 0),
        )
        .expect("valid ConvTranspose geometry");
        let convt_name = format!("convt{block}");
        graph.add_node(match &prev {
            None => GraphNode::from_input(convt_name.as_str(), Layer::ConvTranspose2d(convt)),
            Some(p) => GraphNode::new(
                convt_name.as_str(),
                Layer::ConvTranspose2d(convt),
                vec![p.clone()],
            ),
        });
        prev = Some(convt_name.clone());
        if block + 1 == blocks {
            break;
        }
        // Alternate the shift so each ReLU carries BOTH stably-inactive
        // channels (subnormal intercepts) and live channels (real signal).
        let bn = BatchNormLayer::from_scale_bias(
            Array1::from_iter((0..out_c).map(|_| rng.next_f32(0.9) + 1.3)).into_dyn(),
            Array1::from_iter(
                (0..out_c)
                    .map(|c| rng.next_f32(0.2) + if c % 2 == 0 { -bn_shift } else { bn_shift }),
            )
            .into_dyn(),
        )
        .expect("BatchNorm");
        let bn_name = format!("bn{block}");
        graph.add_node(GraphNode::new(
            bn_name.as_str(),
            Layer::BatchNorm(bn),
            vec![convt_name],
        ));
        let relu_name = format!("relu{block}");
        graph.add_node(GraphNode::new(
            relu_name.as_str(),
            Layer::ReLU(ReLULayer::new()),
            vec![bn_name.clone()],
        ));
        prev = Some(relu_name);
    }
    graph.set_output(prev.as_deref().expect("chain has an output"));
    let input = random_box(&mut rng, &[in_c, in_h, in_w], 0.7, 0.06);
    (graph, input)
}

/// REGRESSION PIN for the measured cGAN forward-linear correlation loss.
///
/// The ConvTranspose2d composition used to route the packed affine coefficient
/// columns through `ConvTranspose2dLayer::propagate_ibp_sound_with_engine`, the
/// certified f32 INTERVAL kernel. That cost the pass twice over:
///
/// 1. **Fail-open.** The kernel returns `[-inf, +inf]` for the WHOLE node as
///    soon as any input endpoint is a binary32 subnormal (its DAZ-independence
///    guard). The forward-linear ReLU composition commits a stable-inactive
///    neuron's exact `0` intercept through `next_down_f32(0.0)` /
///    `next_up_f32(0.0)`, which BY CONSTRUCTION returns `∓1.4e-45`. So every
///    ConvTranspose downstream of the first ReLU returned the universal
///    interval, and `tighten_with_ibp` handed back plain IBP.
/// 2. **Precision.** Its Higham widening is `γ_{K+2}^f32` (≈1.2e-4 at cGAN's
///    `K = 2048`), nine orders coarser than the `γ^f64 ≈ 2.3e-13` the Conv2d
///    seam uses, and it scales with `S = Σ|W||A|` — an IBP-like quantity that
///    does not shrink under cancellation, so it compounds through the chain.
///
/// Part 1 documents the fail-open mechanism against the layer API directly.
/// Part 2 pins the graph-level consequence of (1): finite bounds that stay
/// materially tighter than IBP. Part 3 pins (2): on a PURELY AFFINE
/// ConvTranspose chain the composition must reproduce the exact dense-f64
/// affine range to near machine precision.
#[test]
fn test_image_conv_transpose_after_relu_keeps_input_correlation_daz_packet() {
    // ---- Part 1: the fail-open mechanism. -------------------------------
    let zero_lower = ny_tensor::next_down_f32(0.0);
    let zero_upper = ny_tensor::next_up_f32(0.0);
    for v in [zero_lower, zero_upper] {
        assert!(
            v != 0.0 && v.abs() < f32::MIN_POSITIVE,
            "the ReLU zero-intercept commit must be a binary32 subnormal, got {v:?}"
        );
    }
    let probe = ConvTranspose2dLayer::new_full(
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 0.5f32),
        None,
        (1, 1),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .expect("probe geometry");
    let subnormal_packet =
        BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[1, 2, 2]), zero_lower))
            .expect("subnormal packet");
    let failed_open = probe
        .propagate_ibp_sound_with_engine(&subnormal_packet, None)
        .expect("certified ConvTranspose IBP");
    assert!(
        failed_open
            .lower()
            .iter()
            .zip(failed_open.upper().iter())
            .all(|(&l, &u)| l == f32::NEG_INFINITY && u == f32::INFINITY),
        "fixture assumption broken: the certified interval ConvTranspose kernel no \
         longer fails open on a subnormal source operand"
    );

    // ---- Part 2: no collapse to IBP on a ReLU-fed chain. ----------------
    for (seed, blocks, shift) in [
        (0xDA2_0001u64, 3usize, 1.2f32),
        (0xDA2_0002, 3, 0.6),
        (0xDA2_0003, 4, 1.5),
        (0xDA2_0004, 2, 2.0),
    ] {
        let (graph, input) = build_conv_transpose_relu_chain(seed, blocks, shift);
        let exec_order = graph.exec_order().expect("exec order").to_vec();
        assert!(
            graph.is_sequential_graph(&exec_order),
            "fixture must be a cgan-class sequential chain"
        );
        let forward = graph
            .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
            .expect("ConvTranspose+BN+ReLU chain is on the certified image surface");
        let ibp = graph
            .collect_node_bounds_with_engine(&input, None)
            .expect("IBP bounds");

        for (name, bounds) in &forward {
            assert!(
                bounds
                    .lower()
                    .iter()
                    .chain(bounds.upper().iter())
                    .all(|v| v.is_finite()),
                "seed {seed:x}: forward-linear node '{name}' returned a non-finite bound — \
                 the ConvTranspose composition failed open"
            );
        }

        // A chain whose ReLU-fed ConvTransposes all fail open is BIT-IDENTICAL
        // to IBP (the `tighten_with_ibp` intersection keeps the IBP side).
        let output = graph.output_node.clone();
        let fw_width = tensor_width_sum(&forward[&output]);
        let ibp_width = tensor_width_sum(&ibp[&output]);
        assert!(
            fw_width > 0.0 && fw_width * 1.5 < ibp_width,
            "seed {seed:x}: forward-linear output width {fw_width} is not materially \
             tighter than IBP {ibp_width} — input correlation is being lost"
        );

        // SOUNDNESS: the map must still enclose every sampled point.
        assert_mc_containment(&graph, &input, 64, seed ^ 0xA5A5);
    }

    // ---- Part 3: exact-affine precision on a pure ConvTranspose chain. ---
    // No ReLU, so the composed map is EXACTLY affine and the dense f64 oracle
    // is the ground truth. The f32 interval route inflated this by ~1e-3
    // relative on this fixture (and 2.7% on cGAN's ConvTranspose_4); the f64
    // composition must stay within a few ULPs of exact.
    let mut rng = Lcg::new(0xC0FF_EE11);
    let (in_c, in_h, in_w) = (4usize, 4usize, 4usize);
    let ct1 = ConvTranspose2dLayer::new_full(
        random_kernel(&mut rng, in_c, 8, 4, 4, 0.5),
        Some(random_bias(&mut rng, 8, 0.2)),
        (2, 2),
        (1, 1),
        (1, 1),
        (0, 0),
    )
    .expect("ct1 geometry");
    let ct2 = ConvTranspose2dLayer::new_full(
        random_kernel(&mut rng, 8, 3, 4, 4, 0.5),
        Some(random_bias(&mut rng, 3, 0.2)),
        (2, 2),
        (1, 1),
        (1, 1),
        (0, 0),
    )
    .expect("ct2 geometry");
    let mut affine = GraphNetwork::new();
    affine.add_node(GraphNode::from_input(
        "ct1",
        Layer::ConvTranspose2d(ct1.clone()),
    ));
    affine.add_node(GraphNode::new(
        "ct2",
        Layer::ConvTranspose2d(ct2.clone()),
        vec!["ct1".to_string()],
    ));
    affine.set_output("ct2");
    let input = random_box(&mut rng, &[in_c, in_h, in_w], 0.8, 0.05);

    let d1 = dense_conv_transpose_oracle(&ct1, in_c, in_h, in_w);
    let (mid_h, mid_w) = ct1.output_size(in_h, in_w).expect("ct1 out size");
    let d2 = dense_conv_transpose_oracle(&ct2, 8, mid_h, mid_w);
    let dense = dense_compose(&d2, &d1);
    let (exact_lo, exact_hi) = affine_exact_range(
        &dense.matrix,
        &dense.bias,
        dense.out_dim,
        dense.in_dim,
        &input,
    );

    let forward = affine
        .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
        .expect("pure ConvTranspose chain forward-linear");
    let claimed = forward["ct2"].flatten();
    assert_eq!(claimed.len(), dense.out_dim);
    let mut worst_ratio = 0.0f64;
    for p in 0..dense.out_dim {
        let (cl, cu) = (claimed.lower()[p] as f64, claimed.upper()[p] as f64);
        // SOUND: never excludes the exact affine range.
        assert!(
            cl <= exact_lo[p] && cu >= exact_hi[p],
            "output {p}: claimed [{cl}, {cu}] excludes exact [{}, {}]",
            exact_lo[p],
            exact_hi[p]
        );
        let exact_width = exact_hi[p] - exact_lo[p];
        if exact_width > 1e-6 {
            worst_ratio = worst_ratio.max((cu - cl) / exact_width);
        }
    }
    assert!(
        worst_ratio < 1.000_01,
        "pure-affine ConvTranspose chain is {worst_ratio}x the exact affine width — the \
         composition is charging an f32-scale certified-error penalty (expected < 1.00001x \
         for the certified f64 seam)"
    );
}

/// Randomized ENCLOSURE sweep over ConvTranspose2d geometries with BatchNorm
/// and ReLU in the chain — the operator mix the cGAN generator uses. Covers
/// asymmetric stride/padding/dilation/output_padding and both BN scale signs.
/// Every node's claimed forward-linear bound must contain the brute-force
/// point evaluation at 24 interior samples plus both box corners, and the
/// composed map must additionally enclose the exact dense-f64 affine range of
/// the leading (pre-ReLU) ConvTranspose+BatchNorm.
#[test]
fn test_image_conv_transpose_random_geometry_enclosure_sweep() {
    for seed in 0..12u64 {
        let mut rng = Lcg::new(4_000 + seed);
        let in_c = 2 + (seed % 2) as usize;
        let out_c = 2 + ((seed / 2) % 3) as usize;
        let (kh, kw) = match seed % 3 {
            0 => (1, 1),
            1 => (3, 2),
            _ => (2, 3),
        };
        let stride = (1 + (seed % 2) as usize, 1 + ((seed / 3) % 2) as usize);
        let dilation = (1 + ((seed / 2) % 2) as usize, 1);
        // ConvTranspose padding must not exceed the effective kernel span.
        let padding = (
            ((seed / 4) % 2) as usize * (dilation.0 * (kh - 1) + 1).saturating_sub(1).min(1),
            0,
        );
        // ONNX/PyTorch require output_padding < stride.
        let output_padding = ((seed as usize / 5) % stride.0.max(1), 0);
        let (in_h, in_w) = (2 + (seed % 3) as usize, 3);

        let convt = ConvTranspose2dLayer::new_full(
            random_kernel(&mut rng, in_c, out_c, kh, kw, 0.7),
            Some(random_bias(&mut rng, out_c, 0.3)),
            stride,
            padding,
            dilation,
            output_padding,
        )
        .expect("valid ConvTranspose geometry");
        // Mixed BN scale signs (a negative scale swaps the affine sides).
        let bn = BatchNormLayer::from_scale_bias(
            Array1::from_iter((0..out_c).map(|c| {
                let m = rng.next_f32(0.7).abs() + 0.4;
                if c % 2 == 0 {
                    m
                } else {
                    -m
                }
            }))
            .into_dyn(),
            random_bias(&mut rng, out_c, 0.5).into_dyn(),
        )
        .expect("BatchNorm");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "convt",
            Layer::ConvTranspose2d(convt.clone()),
        ));
        graph.add_node(GraphNode::new(
            "bn",
            Layer::BatchNorm(bn.clone()),
            vec!["convt".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["bn".to_string()],
        ));
        // A second ConvTranspose downstream of the ReLU: the placement that
        // used to fail open on the ReLU's subnormal zero-intercepts.
        let (mid_h, mid_w) = convt.output_size(in_h, in_w).expect("convt out size");
        let convt2 = ConvTranspose2dLayer::new_full(
            random_kernel(&mut rng, out_c, 2, 2, 2, 0.7),
            Some(random_bias(&mut rng, 2, 0.3)),
            (1, 1),
            (0, 0),
            (1, 1),
            (0, 0),
        )
        .expect("valid ConvTranspose geometry");
        graph.add_node(GraphNode::new(
            "convt2",
            Layer::ConvTranspose2d(convt2),
            vec!["relu".to_string()],
        ));
        graph.set_output("convt2");
        let _ = (mid_h, mid_w);

        let input = random_box(&mut rng, &[in_c, in_h, in_w], 0.6, 0.25);

        // (1) Exact dense-f64 affine enclosure at the pre-ReLU BatchNorm.
        let forward = graph
            .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
            .expect("ConvTranspose sweep forward-linear");
        let dense_conv = dense_conv_transpose_oracle(&convt, in_c, in_h, in_w);
        let scale: Vec<f64> = bn.scale.iter().map(|&x| x as f64).collect();
        let bias: Vec<f64> = bn.bias.iter().map(|&x| x as f64).collect();
        let dense = dense_channel_affine_after_conv(&dense_conv, out_c, &scale, &bias);
        let (ex_lo, ex_hi) = affine_exact_range(
            &dense.matrix,
            &dense.bias,
            dense.out_dim,
            dense.in_dim,
            &input,
        );
        let claimed = forward["bn"].flatten();
        for p in 0..dense.out_dim {
            assert!(
                (claimed.lower()[p] as f64) <= ex_lo[p] && (claimed.upper()[p] as f64) >= ex_hi[p],
                "seed {seed}: bn[{p}] claimed [{}, {}] excludes exact [{}, {}]",
                claimed.lower()[p],
                claimed.upper()[p],
                ex_lo[p],
                ex_hi[p]
            );
        }

        // (2) Brute-force sampled containment at EVERY node, including the
        // post-ReLU ConvTranspose.
        assert_mc_containment(&graph, &input, 24, 5_000 + seed);

        // (3) Finite everywhere: no fail-open collapse.
        for (name, bounds) in &forward {
            assert!(
                bounds
                    .lower()
                    .iter()
                    .chain(bounds.upper().iter())
                    .all(|v| v.is_finite()),
                "seed {seed}: node '{name}' returned a non-finite forward-linear bound"
            );
        }
    }
}

// ===========================================================================
// L3: does the cGAN ConvTranspose+BatchNorm differential regression actually
// EXONERATE the conv algebra?  See the two tests below.
// ===========================================================================

/// L3-A — the cited cGAN differential fixture has NO power to detect a
/// correlation loss, and this test proves it by measurement.
///
/// `test_image_conv_transpose_batch_norm_dense_f64_enclosure` builds
/// `input -> ConvTranspose2d -> BatchNorm` and asserts only
/// `claimed ⊇ exact` (plus sampled containment). Two structural facts make
/// that fixture blind:
///
/// 1. The ConvTranspose's upstream map is `LinearBounds::identity(input_dim)`
///    (`resolve_upstream_linear_ref` for `NETWORK_INPUT`), so `lower_a ==
///    upper_a` and the radius channel `A_r = (A_u - A_l)/2` is IDENTICALLY
///    ZERO. The certified sign-split `W⁺A_l + W⁻A_u = W·A_c ∓ |W|·A_r` is
///    never exercised with a nonzero radius — exactly the `LinearBounds::
///    identity` blind spot that hid the CROWN `Sub` false bound.
/// 2. The chain is PURELY AFFINE and the domain is a box, so plain IBP is
///    already EXACT on it. A pass that concretized at every node would produce
///    the same numbers.
///
/// Assertion: on that fixture the forward-linear map is bit-comparable to IBP.
#[test]
fn test_image_cgan_dense_f64_fixture_is_ibp_indistinguishable() {
    let in_c = 2usize;
    let out_c = 3usize;
    let (in_h, in_w) = (2usize, 3usize);
    let kernel = ArrayD::from_shape_fn(IxDyn(&[in_c, out_c, 3, 2]), |idx| {
        let raw = ((idx[0] * 29 + idx[1] * 17 + idx[2] * 7 + idx[3] * 3) % 19) as f32;
        (raw - 9.0) * 0.137
    });
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
        channel_axis_hint: None,
    };
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("convt", Layer::ConvTranspose2d(conv)));
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["convt".to_string()],
    ));
    graph.set_output("bn");

    let mut rng = Lcg::new(0xC6A4_2025);
    let input = random_box(&mut rng, &[in_c, in_h, in_w], 0.8, 0.17);

    let forward = graph
        .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
        .expect("forward-linear");
    let ibp = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("IBP");

    for node in ["convt", "bn"] {
        let fw = tensor_width_sum(&forward[node]);
        let ib = tensor_width_sum(&ibp[node]);
        eprintln!(
            "[L3-A] node {node}: forward width {fw:.9}, IBP width {ib:.9}, ratio {:.9}",
            fw / ib
        );
        // MEASURED: ratio 0.999996 at 'convt', 0.999994 at 'bn'. The residual
        // is f64-vs-f32 accumulation rounding, not correlation.
        assert!(
            (fw - ib).abs() <= 1e-4 * ib,
            "fixture assumption changed: forward ({fw}) and IBP ({ib}) now differ at '{node}'"
        );
    }
}

/// L3-B — the STRONGER variant the cited test is missing: a real box, an
/// upstream forward map with `lower_a != upper_a`, and a TIGHTNESS oracle.
///
/// A crossing ReLU sits in front of the ConvTranspose so the composed rows
/// genuinely split (`A_r != 0`), and the leading Conv2d gives the
/// ConvTranspose a NON-identity, input-correlated upstream map. The oracle is
/// an independent dense-f64 DeepPoly: `M1` and `W` come from the scattered
/// dense oracles (no production indexing code), the ReLU diagonal is taken
/// from `relu_linear_relaxation` at ny's own pre-activation box so the ONLY
/// thing under test is the ConvTranspose/BatchNorm composition, and the
/// sign-split row selection + concretization are re-derived here.
///
/// Three assertions:
///   * ENCLOSURE   — ny never excludes the oracle's own DeepPoly box.
///   * TIGHTNESS   — ny is within a hair of the oracle. A composition that
///     concretized its upstream (IBP-at-every-node) would blow past this.
///   * TEETH       — the oracle is itself far tighter than IBP on this
///     fixture, so the tightness assertion can actually fail.
#[test]
fn test_image_conv_transpose_batch_norm_preserves_split_correlated_rows() {
    use crate::layers::activations::relu::relu_linear_relaxation;

    for seed in 0..6u64 {
        let mut rng = Lcg::new(0x5D3A_0000 + seed);
        let (in_c, in_h, in_w) = (2usize, 5usize, 5usize);
        let mid_c = 3usize;
        let out_c = 2usize;
        let in_dim = in_c * in_h * in_w;

        // Leading Conv2d: gives the ConvTranspose a NON-identity upstream map.
        // Channel biases are chosen so the ReLU carries all three regimes:
        // stably-active (D = I), stably-inactive (D = 0) and crossing
        // (D_l != D_u -> nonzero radius A_r into the ConvTranspose).
        let conv1 = Conv2dLayer::with_input_shape(
            random_kernel(&mut rng, mid_c, in_c, 3, 3, 0.8),
            Some(Array1::from_vec(vec![2.5, -2.5, 0.05])),
            (1, 1),
            (1, 1),
            in_h,
            in_w,
        )
        .expect("conv1");
        let convt = ConvTranspose2dLayer::new_full(
            random_kernel(&mut rng, mid_c, out_c, 3, 3, 0.7),
            Some(random_bias(&mut rng, out_c, 0.3)),
            (1, 1),
            (1, 1),
            (1, 1),
            (0, 0),
        )
        .expect("convt");
        // Negative scale on channel 0 exercises the BatchNorm side swap.
        let bn = BatchNormLayer::from_scale_bias(
            Array1::from_vec(vec![-1.35, 0.62]).into_dyn(),
            Array1::from_vec(vec![0.17, -0.09]).into_dyn(),
        )
        .expect("bn");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1.clone())));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["conv1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "convt",
            Layer::ConvTranspose2d(convt.clone()),
            vec!["relu".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "bn",
            Layer::BatchNorm(bn.clone()),
            vec!["convt".to_string()],
        ));
        graph.set_output("bn");

        let input = random_box(&mut rng, &[in_c, in_h, in_w], 0.6, 0.22);
        let forward = graph
            .collect_forward_linear_bounds_dag_with_conv_transpose_for_test(&input, None)
            .expect("forward-linear");
        let ibp = graph
            .collect_node_bounds_with_engine(&input, None)
            .expect("IBP");

        // ---- independent dense-f64 DeepPoly oracle -----------------------
        let m1 = dense_conv_oracle(&conv1, in_c, in_h, in_w);
        assert_eq!(m1.in_dim, in_dim);
        let mid_dim = m1.out_dim;

        // ReLU diagonal at ny's OWN pre-activation box (isolates the conv).
        let pre = forward["conv1"].flatten();
        let mut post_l_a = vec![0.0f64; mid_dim * in_dim];
        let mut post_u_a = vec![0.0f64; mid_dim * in_dim];
        let mut post_l_b = vec![0.0f64; mid_dim];
        let mut post_u_b = vec![0.0f64; mid_dim];
        let mut regimes = [0usize; 3];
        for i in 0..mid_dim {
            let (l, u) = (pre.lower()[i], pre.upper()[i]);
            if l >= 0.0 {
                regimes[0] += 1;
            } else if u <= 0.0 {
                regimes[1] += 1;
            } else {
                regimes[2] += 1;
            }
            let r = relu_linear_relaxation(l, u);
            let (dl, cl) = (r.lower_slope as f64, r.lower_intercept as f64);
            let (du, cu) = (r.upper_slope as f64, r.upper_intercept as f64);
            // conv1's upstream is the input identity, so its lower_a == upper_a
            // == m1.matrix; the ReLU's own source selection is a no-op here.
            for j in 0..in_dim {
                post_l_a[i * in_dim + j] = dl * m1.matrix[i * in_dim + j];
                post_u_a[i * in_dim + j] = du * m1.matrix[i * in_dim + j];
            }
            post_l_b[i] = dl * m1.bias[i] + cl;
            post_u_b[i] = du * m1.bias[i] + cu;
        }
        assert!(
            regimes.iter().all(|&c| c > 0),
            "seed {seed}: fixture must mix stable-active/inactive/crossing ReLUs, got {regimes:?}"
        );
        // The rows entering the ConvTranspose MUST be split.
        let split_rows = (0..mid_dim)
            .filter(|&i| (0..in_dim).any(|j| post_l_a[i * in_dim + j] != post_u_a[i * in_dim + j]))
            .count();
        assert!(
            split_rows > 0,
            "seed {seed}: upstream map has lower_a == upper_a — the degenerate case"
        );

        // ConvTranspose: sign-split row selection, re-derived independently.
        let wt = dense_conv_transpose_oracle(&convt, mid_c, in_h, in_w);
        assert_eq!(wt.in_dim, mid_dim);
        let out_dim = wt.out_dim;
        let mut ct_l_a = vec![0.0f64; out_dim * in_dim];
        let mut ct_u_a = vec![0.0f64; out_dim * in_dim];
        let mut ct_l_b = vec![0.0f64; out_dim];
        let mut ct_u_b = vec![0.0f64; out_dim];
        for p in 0..out_dim {
            let mut lb = wt.bias[p];
            let mut ub = wt.bias[p];
            for i in 0..mid_dim {
                let w = wt.matrix[p * mid_dim + i];
                if w == 0.0 {
                    continue;
                }
                if w >= 0.0 {
                    for j in 0..in_dim {
                        ct_l_a[p * in_dim + j] += w * post_l_a[i * in_dim + j];
                        ct_u_a[p * in_dim + j] += w * post_u_a[i * in_dim + j];
                    }
                    lb += w * post_l_b[i];
                    ub += w * post_u_b[i];
                } else {
                    for j in 0..in_dim {
                        ct_l_a[p * in_dim + j] += w * post_u_a[i * in_dim + j];
                        ct_u_a[p * in_dim + j] += w * post_l_a[i * in_dim + j];
                    }
                    lb += w * post_u_b[i];
                    ub += w * post_l_b[i];
                }
            }
            ct_l_b[p] = lb;
            ct_u_b[p] = ub;
        }

        // BatchNorm: per-channel diagonal, negative scale swaps the sides.
        let spatial = out_dim / out_c;
        let mut bn_l_a = vec![0.0f64; out_dim * in_dim];
        let mut bn_u_a = vec![0.0f64; out_dim * in_dim];
        let mut bn_l_b = vec![0.0f64; out_dim];
        let mut bn_u_b = vec![0.0f64; out_dim];
        for p in 0..out_dim {
            let c = p / spatial;
            let s = bn.scale[c] as f64;
            let b = bn.bias[c] as f64;
            let (src_l, src_lb, src_u, src_ub): (&[f64], f64, &[f64], f64) = if s >= 0.0 {
                (&ct_l_a, ct_l_b[p], &ct_u_a, ct_u_b[p])
            } else {
                (&ct_u_a, ct_u_b[p], &ct_l_a, ct_l_b[p])
            };
            for j in 0..in_dim {
                bn_l_a[p * in_dim + j] = s * src_l[p * in_dim + j];
                bn_u_a[p * in_dim + j] = s * src_u[p * in_dim + j];
            }
            bn_l_b[p] = s * src_lb + b;
            bn_u_b[p] = s * src_ub + b;
        }

        // Per-node divergence report: where (if anywhere) does ny's width
        // first depart from the dense-f64 DeepPoly reference?
        {
            let flat = input.flatten();
            let xlo: Vec<f64> = flat.lower().iter().map(|&v| v as f64).collect();
            let xhi: Vec<f64> = flat.upper().iter().map(|&v| v as f64).collect();
            let concretize = |la: &[f64], lb: &[f64], ua: &[f64], ub: &[f64], rows: usize| {
                let mut w = 0.0f64;
                for p in 0..rows {
                    let (mut lo, mut hi) = (lb[p], ub[p]);
                    for j in 0..in_dim {
                        let (al, au) = (la[p * in_dim + j], ua[p * in_dim + j]);
                        lo += if al >= 0.0 { al * xlo[j] } else { al * xhi[j] };
                        hi += if au >= 0.0 { au * xhi[j] } else { au * xlo[j] };
                    }
                    w += hi - lo;
                }
                w
            };
            let ref_relu = concretize(&post_l_a, &post_l_b, &post_u_a, &post_u_b, mid_dim);
            let ref_convt = concretize(&ct_l_a, &ct_l_b, &ct_u_a, &ct_u_b, out_dim);
            for (node, refw) in [("relu", ref_relu), ("convt", ref_convt)] {
                let nyw = tensor_width_sum(&forward[node]);
                let ibpw = tensor_width_sum(&ibp[node]);
                eprintln!(
                    "[L3-B]   seed {seed} node {node}: ny={nyw:.6} deeppoly_ref={refw:.6} \
                     ibp={ibpw:.6} ny/ref={:.6}",
                    nyw / refw
                );
                assert!(
                    nyw <= refw * 1.000_1,
                    "seed {seed}: node '{node}' ny width {nyw} exceeds the DeepPoly \
                     reference {refw} — correlation lost at this node"
                );
            }
        }

        // Concretize the oracle over the input box.
        let flat = input.flatten();
        let xlo: Vec<f64> = flat.lower().iter().map(|&v| v as f64).collect();
        let xhi: Vec<f64> = flat.upper().iter().map(|&v| v as f64).collect();
        let mut ref_lo = vec![0.0f64; out_dim];
        let mut ref_hi = vec![0.0f64; out_dim];
        for p in 0..out_dim {
            let mut lo = bn_l_b[p];
            let mut hi = bn_u_b[p];
            for j in 0..in_dim {
                let al = bn_l_a[p * in_dim + j];
                let au = bn_u_a[p * in_dim + j];
                lo += if al >= 0.0 { al * xlo[j] } else { al * xhi[j] };
                hi += if au >= 0.0 { au * xhi[j] } else { au * xlo[j] };
            }
            ref_lo[p] = lo;
            ref_hi[p] = hi;
        }

        // ---- compare ------------------------------------------------------
        let claimed = forward["bn"].flatten();
        assert_eq!(claimed.len(), out_dim);
        let ref_width: f64 = (0..out_dim).map(|p| ref_hi[p] - ref_lo[p]).sum();
        let ny_width = tensor_width_sum(&forward["bn"]);
        let ibp_width = tensor_width_sum(&ibp["bn"]);
        eprintln!(
            "[L3-B] seed {seed}: regimes(active,inactive,crossing)={regimes:?} split_rows={split_rows}/{mid_dim} \
             ny_width={ny_width:.6} deeppoly_ref_width={ref_width:.6} ibp_width={ibp_width:.6} \
             ny/ref={:.6} ibp/ref={:.6}",
            ny_width / ref_width,
            ibp_width / ref_width
        );

        // TEETH: the oracle must be materially tighter than IBP, otherwise the
        // tightness assertion below could not distinguish the two.
        assert!(
            ibp_width > ref_width * 1.5,
            "seed {seed}: fixture is not discriminating — IBP width {ibp_width} is not \
             materially above the DeepPoly reference {ref_width}"
        );

        let mut worst_slack = 0.0f64;
        for p in 0..out_dim {
            let (cl, cu) = (claimed.lower()[p] as f64, claimed.upper()[p] as f64);
            let scale = 1.0 + ref_lo[p].abs().max(ref_hi[p].abs());
            // ENCLOSURE: ny never cuts inside the oracle's DeepPoly box beyond
            // what the (sound) IBP intersection can legitimately shave. The
            // sampled containment below is the real soundness check.
            // TIGHTNESS: ny may not be materially LOOSER than the oracle.
            worst_slack = worst_slack.max((ref_lo[p] - cl) / scale);
            worst_slack = worst_slack.max((cu - ref_hi[p]) / scale);
        }
        assert!(
            worst_slack < 1e-4,
            "seed {seed}: ny's ConvTranspose+BatchNorm output is {worst_slack} (relative) LOOSER \
             than the independent dense-f64 DeepPoly oracle — input correlation is being lost in \
             the composition"
        );
        assert!(
            ny_width <= ref_width * 1.000_1,
            "seed {seed}: ny total width {ny_width} exceeds the DeepPoly reference {ref_width}"
        );

        // SOUNDNESS: brute-force containment at every node.
        assert_mc_containment(&graph, &input, 48, 0x5D3A_1000 + seed);
    }
}
