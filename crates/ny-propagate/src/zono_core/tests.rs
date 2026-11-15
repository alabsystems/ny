// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness pins for the certified zonotope core, on synthetic graphs.
//!
//! Real-model containment on the four cGANs, the Z1 replica parity, the sat
//! control leaves and the transformer refusal live in ny-cli
//! (`commands/cgan_child_zono_tests.rs`), where ONNX loading is available.
//!
//! Every check is POINTWISE: box containment AND pointwise form containment
//! (`form_check`), at uniform samples plus every corner. A test that samples
//! nothing, or whose precondition (unstable ReLUs, folding engaged, an
//! inherited map consulted) does not hold, fails instead of passing vacuously.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::form_check::{check_point, summarize, ContainmentReport, FormSummary};
use super::*;
use crate::layers::{
    BatchNormLayer, Conv2dLayer, ConvTranspose2dLayer, FlattenLayer, Layer, LinearLayer, ReLULayer,
    ReshapeLayer, SigmoidLayer,
};
use crate::{GraphNetwork, GraphNode};

/// Deterministic xorshift generator.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[0, 1)`.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[-1, 1)`.
    pub(crate) fn sym(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }
}

#[allow(clippy::cast_possible_truncation)]
fn rand_f32(rng: &mut Rng, n: usize, scale: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.sym() * scale) as f32).collect()
}

#[allow(clippy::cast_possible_truncation)]
fn bn_layer(rng: &mut Rng, channels: usize) -> BatchNormLayer {
    let gamma: Vec<f32> = (0..channels).map(|_| (0.5 + rng.unit()) as f32).collect();
    let beta = rand_f32(rng, channels, 0.3);
    let mean = rand_f32(rng, channels, 0.3);
    let var: Vec<f32> = (0..channels).map(|_| (0.3 + rng.unit()) as f32).collect();
    let arr = |v: Vec<f32>| ArrayD::from_shape_vec(IxDyn(&[channels]), v).expect("bn vector");
    BatchNormLayer::new(&arr(gamma), &arr(beta), &arr(mean), &arr(var), 1e-5).expect("bn layer")
}

/// A miniature cGAN: Gemm -> Reshape -> BN -> ConvTranspose -> BN -> ReLU ->
/// Conv -> ReLU -> BN -> Flatten -> Gemm, with every op of the cGAN set.
/// `convt_pad` exercises the cropping ConvTranspose variant.
pub(crate) fn small_cgan_graph(seed: u64, convt_pad: usize) -> GraphNetwork {
    let mut rng = Rng::new(seed);
    let w1 = Array2::from_shape_vec((16, 5), rand_f32(&mut rng, 80, 0.9)).expect("w1");
    let b1 = Array1::from_vec(rand_f32(&mut rng, 16, 0.2));
    let lin1 = LinearLayer::new(w1, Some(b1)).expect("lin1");
    let bn1 = bn_layer(&mut rng, 4);
    let (kt, st) = if convt_pad == 0 { (3, 2) } else { (4, 2) };
    let convt_out = st + kt - 2 * convt_pad;
    let convt = ConvTranspose2dLayer::with_input_shape(
        ArrayD::from_shape_vec(
            IxDyn(&[4, 3, kt, kt]),
            rand_f32(&mut rng, 4 * 3 * kt * kt, 0.6),
        )
        .expect("convt kernel"),
        Some(Array1::from_vec(rand_f32(&mut rng, 3, 0.1))),
        (st, st),
        (convt_pad, convt_pad),
        2,
        2,
    )
    .expect("convt");
    let bn2 = bn_layer(&mut rng, 3);
    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[4, 3, 3, 3]), rand_f32(&mut rng, 108, 0.5))
            .expect("conv kernel"),
        Some(Array1::from_vec(rand_f32(&mut rng, 4, 0.1))),
        (2, 2),
        (1, 1),
        convt_out,
        convt_out,
    )
    .expect("conv");
    let conv_out = (convt_out + 2 - 3) / 2 + 1;
    let bn3 = bn_layer(&mut rng, 4);
    let flat = 4 * conv_out * conv_out;
    let w2 = Array2::from_shape_vec((1, flat), rand_f32(&mut rng, flat, 0.5)).expect("w2");
    let lin2 = LinearLayer::new(w2, Some(Array1::from_vec(vec![0.05]))).expect("lin2");

    let mut graph = GraphNetwork::new();
    let chain: Vec<(&str, Layer)> = vec![
        ("gemm0", Layer::Linear(lin1)),
        ("reshape", Layer::Reshape(ReshapeLayer::new(vec![4, 2, 2]))),
        ("bn1", Layer::BatchNorm(bn1)),
        ("convt", Layer::ConvTranspose2d(convt)),
        ("bn2", Layer::BatchNorm(bn2)),
        ("relu1", Layer::ReLU(ReLULayer)),
        ("conv", Layer::Conv2d(conv)),
        ("relu2", Layer::ReLU(ReLULayer)),
        ("bn3", Layer::BatchNorm(bn3)),
        ("flat", Layer::Flatten(FlattenLayer::new(0))),
        ("gemm1", Layer::Linear(lin2)),
    ];
    let mut prev: Option<String> = None;
    for (name, layer) in chain {
        match prev {
            None => graph.add_node(GraphNode::from_input(name, layer)),
            Some(ref input) => graph.add_node(GraphNode::new(name, layer, vec![input.clone()])),
        }
        prev = Some(name.to_string());
    }
    graph.set_output("gemm1");
    graph
}

/// A wide ReLU MLP whose unstable count exceeds 1024 on `wide_box`, so every
/// budget of the fold test folds.
fn wide_mlp_graph(seed: u64) -> GraphNetwork {
    let mut rng = Rng::new(seed);
    let dims = [5usize, 1024, 1024, 512, 1];
    let mut graph = GraphNetwork::new();
    let mut prev: Option<String> = None;
    for layer_idx in 0..dims.len() - 1 {
        let (fan_in, fan_out) = (dims[layer_idx], dims[layer_idx + 1]);
        #[allow(clippy::cast_precision_loss)]
        let scale = 2.0 / (fan_in as f64).sqrt();
        let w = Array2::from_shape_vec(
            (fan_out, fan_in),
            rand_f32(&mut rng, fan_in * fan_out, scale),
        )
        .expect("mlp weight");
        let b = Array1::from_vec(rand_f32(&mut rng, fan_out, 0.1));
        let name = format!("fc{layer_idx}");
        let layer = Layer::Linear(LinearLayer::new(w, Some(b)).expect("mlp layer"));
        match prev {
            None => graph.add_node(GraphNode::from_input(name.clone(), layer)),
            Some(ref input) => {
                graph.add_node(GraphNode::new(name.clone(), layer, vec![input.clone()]));
            }
        }
        prev = Some(name.clone());
        if layer_idx + 2 < dims.len() {
            let relu = format!("relu{layer_idx}");
            graph.add_node(GraphNode::new(
                relu.clone(),
                Layer::ReLU(ReLULayer),
                vec![name],
            ));
            prev = Some(relu);
        }
    }
    graph.set_output(prev.expect("mlp output"));
    graph
}

fn input_box(lo: &[f32], hi: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lo.len()]), lo.to_vec()).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[hi.len()]), hi.to_vec()).expect("hi"),
    )
    .expect("input box")
}

fn root_box() -> BoundedTensor {
    input_box(&[-0.6, -0.4, 0.1, -0.9, 0.3], &[0.5, 0.35, 0.8, -0.2, 0.9])
}

/// A random sub-box of `parent` (a legitimate BaB child), halved `depth` times.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn random_child(parent: &BoundedTensor, depth: usize, rng: &mut Rng) -> BoundedTensor {
    let mut lo: Vec<f32> = parent.lower().iter().copied().collect();
    let mut hi: Vec<f32> = parent.upper().iter().copied().collect();
    for _ in 0..depth {
        let dim = (rng.next_u64() % lo.len() as u64) as usize;
        let mid = lo[dim] + (hi[dim] - lo[dim]) / 2.0;
        if rng.unit() < 0.5 {
            hi[dim] = mid;
        } else {
            lo[dim] = mid;
        }
    }
    BoundedTensor::new(
        ArrayD::from_shape_vec(parent.lower().raw_dim(), lo).expect("child lo"),
        ArrayD::from_shape_vec(parent.upper().raw_dim(), hi).expect("child hi"),
    )
    .expect("child box")
}

/// `samples` uniform points plus every corner of `domain`, flat row-major.
pub(crate) fn sample_points(
    domain: &BoundedTensor,
    samples: usize,
    seed: u64,
) -> (Vec<f64>, usize) {
    let lo: Vec<f64> = domain.lower().iter().map(|&v| f64::from(v)).collect();
    let hi: Vec<f64> = domain.upper().iter().map(|&v| f64::from(v)).collect();
    let n = lo.len();
    let corners = 1usize << n.min(12);
    let mut points = Vec::with_capacity((samples + corners) * n);
    for corner in 0..corners {
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
    let rows = points.len() / n.max(1);
    (points, rows)
}

/// Run the forward pass with an observer and check every node of every sampled
/// point (box + pointwise form), plus every published entry (box).
pub(crate) fn containment(
    program: &ZonoProgram,
    input: &BoundedTensor,
    opts: &ZonoForwardOptions,
    inherited: &[&HashMap<String, BoundedTensor>],
    samples: usize,
    seed: u64,
) -> (ContainmentReport, ZonoForwardOutcome) {
    let mut summaries: Vec<FormSummary> = Vec::new();
    let mut observer = |view: &ZonoNodeView<'_>| summaries.push(summarize(view));
    let outcome = program
        .forward(input, opts, inherited, None, Some(&mut observer))
        .expect("certified forward pass");
    assert_eq!(summaries.len(), program.steps().len() + 1);
    let (points, rows) = sample_points(input, samples, seed);
    let n0 = program.input_len();
    let mut report = ContainmentReport::default();
    program
        .evaluate_points(&points, rows, |step, values| {
            let summary = &summaries[step.map_or(0, |s| s + 1)];
            let n = summary.center.len();
            for r in 0..rows {
                check_point(
                    summary,
                    &points[r * n0..(r + 1) * n0],
                    &values[r * n..(r + 1) * n],
                    &mut report,
                );
            }
            if let Some(s) = step {
                if let Some(entry) = outcome.published.get(&program.steps()[s].node) {
                    let lo: Vec<f32> = entry.lower().iter().copied().collect();
                    let hi: Vec<f32> = entry.upper().iter().copied().collect();
                    for r in 0..rows {
                        for (j, &v) in values[r * n..(r + 1) * n].iter().enumerate() {
                            report.checked += 1;
                            if v < f64::from(lo[j]) || v > f64::from(hi[j]) {
                                report.box_violations += 1;
                            }
                        }
                    }
                }
            }
        })
        .expect("concrete evaluation");
    assert!(report.checked > 0, "no value was checked: the test is VOID");
    (report, outcome)
}

fn default_opts() -> ZonoForwardOptions {
    ZonoForwardOptions {
        max_generators: 16_384,
        record_trace: false,
        row_par: RowPar::Seq,
    }
}

// ---------------------------------------------------------------------------
// Kernels against an independent naive reference
// ---------------------------------------------------------------------------

fn naive_conv(x: &[f64], g: &ConvGeom, co: usize, k: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; co * g.out_h * g.out_w];
    for o in 0..co {
        for oy in 0..g.out_h {
            for ox in 0..g.out_w {
                let mut acc = 0.0;
                for ci in 0..g.channels {
                    for a in 0..g.kernel_h {
                        for b in 0..g.kernel_w {
                            let iy = (oy * g.stride_h + a) as isize - g.pad_h as isize;
                            let ix = (ox * g.stride_w + b) as isize - g.pad_w as isize;
                            if iy < 0 || ix < 0 || iy >= g.height as isize || ix >= g.width as isize
                            {
                                continue;
                            }
                            let (iy, ix) = (iy as usize, ix as usize);
                            acc += k[((o * g.channels + ci) * g.kernel_h + a) * g.kernel_w + b]
                                * x[(ci * g.height + iy) * g.width + ix];
                        }
                    }
                }
                out[(o * g.out_h + oy) * g.out_w + ox] = acc;
            }
        }
    }
    out
}

/// ONNX ConvTranspose with symmetric pads: `y[co, i*s + a - p, j*s + b - p]
/// += x[ci, i, j] * K[ci, co, a, b]`.
#[allow(clippy::too_many_arguments)]
fn naive_convt(
    x: &[f64],
    ci_n: usize,
    h: usize,
    w: usize,
    co_n: usize,
    kh: usize,
    kw: usize,
    s: usize,
    p: usize,
    k: &[f64],
) -> (Vec<f64>, usize, usize) {
    let oh = (h - 1) * s + kh - 2 * p;
    let ow = (w - 1) * s + kw - 2 * p;
    let mut out = vec![0.0; co_n * oh * ow];
    for ci in 0..ci_n {
        for i in 0..h {
            for j in 0..w {
                for co in 0..co_n {
                    for a in 0..kh {
                        for b in 0..kw {
                            let y = (i * s + a) as isize - p as isize;
                            let z = (j * s + b) as isize - p as isize;
                            if y < 0 || z < 0 || y >= oh as isize || z >= ow as isize {
                                continue;
                            }
                            out[(co * oh + y as usize) * ow + z as usize] +=
                                x[(ci * h + i) * w + j] * k[((ci * co_n + co) * kh + a) * kw + b];
                        }
                    }
                }
            }
        }
    }
    (out, oh, ow)
}

fn assert_close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (j, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-12 * (1.0 + x.abs() + y.abs()),
            "{what}: element {j} differs: {x} vs {y}"
        );
    }
}

#[test]
fn zono_kernels_match_naive_reference_and_adjoints_are_transposes() {
    let mut rng = Rng::new(3);
    // Conv2d with padding and stride.
    let geom = ConvGeom::new(3, 7, 6, 3, 3, 2, 2, 1, 1).expect("conv geom");
    let co = 4;
    let k: Vec<f64> = (0..co * geom.patch_len()).map(|_| rng.sym()).collect();
    let conv = LinearKernel::Conv2d {
        geom,
        out_channels: co,
        k: k.clone(),
        k_abs: k.iter().map(|v| v.abs()).collect(),
    };
    // ConvTranspose2d with cropping padding.
    let (ci_n, h, w, co_n, kh, s, p) = (3usize, 3usize, 4usize, 2usize, 4usize, 2usize, 1usize);
    let kt: Vec<f64> = (0..ci_n * co_n * kh * kh).map(|_| rng.sym()).collect();
    let (oh, ow) = ((h - 1) * s + kh - 2 * p, (w - 1) * s + kh - 2 * p);
    let convt = LinearKernel::ConvTranspose2d {
        geom: ConvGeom::new(co_n, oh, ow, kh, kh, s, s, p, p).expect("convt geom"),
        in_channels: ci_n,
        k: kt.clone(),
        k_abs: kt.iter().map(|v| v.abs()).collect(),
    };
    let wd: Vec<f64> = (0..6 * 5).map(|_| rng.sym()).collect();
    let dense = LinearKernel::Dense {
        n_in: 5,
        n_out: 6,
        w: wd.clone(),
        w_abs: wd.iter().map(|v| v.abs()).collect(),
    };

    let rows = 3;
    for kernel in [&conv, &convt, &dense] {
        let (n_in, n_out) = (kernel.n_in(), kernel.n_out());
        let x: Vec<f64> = (0..rows * n_in).map(|_| rng.sym()).collect();
        let mut y = vec![0.0; rows * n_out];
        kernel
            .forward_rows(&x, rows, false, &mut y, None)
            .expect("forward");
        for r in 0..rows {
            let xr = &x[r * n_in..(r + 1) * n_in];
            let reference = match kernel {
                LinearKernel::Conv2d { geom, .. } => naive_conv(xr, geom, co, &k),
                LinearKernel::ConvTranspose2d { .. } => {
                    naive_convt(xr, ci_n, h, w, co_n, kh, kh, s, p, &kt).0
                }
                LinearKernel::Dense { .. } => (0..6)
                    .map(|o| (0..5).map(|i| wd[o * 5 + i] * xr[i]).sum())
                    .collect(),
            };
            assert_close(
                &y[r * n_out..(r + 1) * n_out],
                &reference,
                "forward vs naive",
            );
        }
        // Adjoint: <A x, a> == <x, A^T a>.
        let a: Vec<f64> = (0..rows * n_out).map(|_| rng.sym()).collect();
        let mut at = vec![0.0; rows * n_in];
        kernel
            .adjoint_rows(&a, rows, false, &mut at, None)
            .expect("adjoint");
        for r in 0..rows {
            let lhs: f64 = y[r * n_out..(r + 1) * n_out]
                .iter()
                .zip(&a[r * n_out..(r + 1) * n_out])
                .map(|(p, q)| p * q)
                .sum();
            let rhs: f64 = x[r * n_in..(r + 1) * n_in]
                .iter()
                .zip(&at[r * n_in..(r + 1) * n_in])
                .map(|(p, q)| p * q)
                .sum();
            assert!(
                (lhs - rhs).abs() <= 1e-10 * (1.0 + lhs.abs()),
                "adjoint identity: {lhs} vs {rhs}"
            );
        }
        // The abs kernel on a non-negative input dominates |A x|.
        let xa: Vec<f64> = x.iter().map(|v| v.abs()).collect();
        let mut ya = vec![0.0; rows * n_out];
        kernel
            .forward_rows(&xa, rows, true, &mut ya, None)
            .expect("abs forward");
        for (signed, bound) in y.iter().zip(&ya) {
            assert!(signed.abs() <= bound * (1.0 + 1e-12) + 1e-300);
        }
    }
}

// ---------------------------------------------------------------------------
// Containment
// ---------------------------------------------------------------------------

#[test]
fn from_box_encloses_the_box_and_skips_fixed_dimensions() {
    let lo = [0.25f64, -1.0, 3.0];
    let hi = [0.5f64, -1.0, 3.5];
    let mut alloc = SymbolAllocator::new();
    let (state, symbols) = ZonoState::from_box(&lo, &hi, &mut alloc).expect("box");
    assert_eq!(symbols.len(), 2, "the fixed dimension gets no symbol");
    assert_eq!(state.num_generators(), 2);
    let (clo, chi) = concretize(&state, &state.abs_generator_sum());
    for j in 0..3 {
        assert!(
            clo[j] <= lo[j] && chi[j] >= hi[j],
            "dimension {j} not enclosed"
        );
    }
    assert!(ZonoState::from_box(&[1.0], &[0.0], &mut alloc).is_err());
    assert!(ZonoState::from_box(&[f64::NAN], &[0.0], &mut alloc).is_err());
}

#[test]
fn child_zono_synthetic_cgan_contains_samples_at_every_depth() {
    for convt_pad in [0usize, 1] {
        let graph = small_cgan_graph(11 + convt_pad as u64, convt_pad);
        let root = root_box();
        let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
        let inherited_root = graph
            .collect_node_bounds_with_engine(&root, None)
            .expect("root IBP map");
        let mut rng = Rng::new(17);
        let mut unstable_seen = 0usize;
        let mut inherited_seen = 0usize;
        for depth in 0..=12 {
            let child = random_child(&root, depth, &mut rng);
            let inherited: Vec<&HashMap<String, BoundedTensor>> = if depth % 2 == 0 {
                vec![&inherited_root]
            } else {
                Vec::new()
            };
            let (report, outcome) = containment(
                &program,
                &child,
                &default_opts(),
                &inherited,
                2_000,
                depth as u64,
            );
            assert_eq!(
                report.box_violations, 0,
                "pad={convt_pad} depth={depth}: box violations {report:?}"
            );
            assert_eq!(
                report.form_violations, 0,
                "pad={convt_pad} depth={depth}: form violations {report:?}"
            );
            unstable_seen += outcome.relus.iter().map(|r| r.unstable).sum::<usize>();
            inherited_seen += outcome
                .relus
                .iter()
                .filter(|r| r.inherited_unstable.is_some())
                .count();
            assert_eq!(outcome.published.len(), 2, "both ReLU inputs are published");
        }
        assert!(unstable_seen > 0, "no ReLU was ever unstable: VOID");
        assert!(
            inherited_seen > 0,
            "the inherited map was never consulted: VOID"
        );
    }
}

/// Test fault guard: resets the thread-local mutation on drop, even on panic.
struct FaultGuard;

impl Drop for FaultGuard {
    fn drop(&mut self) {
        relu::fault::set(relu::fault::ReluFault::None);
        ids::fault::set_positional_alignment(false);
    }
}

#[test]
fn child_zono_form_mutation_fails() {
    let graph = small_cgan_graph(23, 0);
    let root = root_box();
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
    let _guard = FaultGuard;

    // Positive controls first: the unmutated core is clean on both settings.
    let tight = ZonoForwardOptions {
        max_generators: 8,
        record_trace: false,
        row_par: RowPar::Seq,
    };
    for opts in [default_opts(), tight] {
        let (report, outcome) = containment(&program, &root, &opts, &[], 3_000, 5);
        assert_eq!(
            report.box_violations + report.form_violations,
            0,
            "{report:?}"
        );
        assert!(
            outcome.relus.iter().any(|r| r.unstable > 0),
            "no unstable ReLU: the mutation pin would be VOID"
        );
    }
    let (_, folded) = containment(&program, &root, &tight, &[], 10, 5);
    assert!(
        folded.stats.folded > 0,
        "budget 8 must fold for the r-drop pin"
    );

    // Flip the DeepZ offset: the form must go red.
    relu::fault::set(relu::fault::ReluFault::FlipMuSign);
    let (flipped, _) = containment(&program, &root, &default_opts(), &[], 3_000, 5);
    relu::fault::set(relu::fault::ReluFault::None);
    assert!(
        flipped.form_violations > 0,
        "a flipped mu must violate the pointwise form: {flipped:?}"
    );

    // Drop the remainder after each ReLU while folding put real mass into it.
    relu::fault::set(relu::fault::ReluFault::DropRemainder);
    let (dropped, _) = containment(&program, &root, &tight, &[], 3_000, 5);
    relu::fault::set(relu::fault::ReluFault::None);
    assert!(
        dropped.form_violations > 0,
        "a dropped remainder must violate the pointwise form: {dropped:?}"
    );
}

#[test]
fn child_zono_fold_budget_contains_samples() {
    let graph = wide_mlp_graph(31);
    let wide = input_box(&[-1.0, -1.0, -1.0, -1.0, -1.0], &[1.0, 1.0, 1.0, 1.0, 1.0]);
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &wide).expect("compile");
    for budget in [64usize, 256, 1024] {
        let opts = ZonoForwardOptions {
            max_generators: budget,
            record_trace: false,
            row_par: RowPar::Seq,
        };
        let (report, outcome) = containment(&program, &wide, &opts, &[], 1_000, 9);
        assert!(
            outcome.stats.folded > 0,
            "budget {budget}: nothing folded (gens peak {}), the test is VOID",
            outcome.stats.gens_peak
        );
        assert!(outcome.stats.gens_final <= budget.max(5));
        assert_eq!(report.box_violations, 0, "budget {budget}: {report:?}");
        assert_eq!(report.form_violations, 0, "budget {budget}: {report:?}");
    }
}

#[test]
fn child_zono_symbol_ids_align_by_id() {
    let n = 4;
    let mut rng = Rng::new(41);
    let make = |ids: &[u64], rng: &mut Rng| {
        let gens: Vec<f64> = (0..ids.len() * n).map(|_| rng.sym()).collect();
        let center: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
        let rem: Vec<f64> = (0..n).map(|_| 0.01 * rng.unit()).collect();
        ZonoState::from_parts(
            n,
            center,
            gens,
            ids.iter().map(|&i| SymbolId(i)).collect(),
            rem,
        )
        .expect("form")
    };
    // Permuted AND disjoint IDs.
    let a = make(&[1, 2, 3], &mut rng);
    let b = make(&[3, 1, 7], &mut rng);
    let _guard = FaultGuard;
    for subtract in [false, true] {
        let sum = add_aligned(&a, &b, subtract).expect("aligned add");
        // Reference by ID dictionary.
        let sign = if subtract { -1.0 } else { 1.0 };
        let mut by_id: HashMap<SymbolId, Vec<f64>> = HashMap::new();
        for (k, id) in a.ids().iter().enumerate() {
            by_id.insert(*id, a.generator_row(k).to_vec());
        }
        for (k, id) in b.ids().iter().enumerate() {
            let row = by_id.entry(*id).or_insert_with(|| vec![0.0; n]);
            for (slot, v) in row.iter_mut().zip(b.generator_row(k)) {
                *slot += sign * v;
            }
        }
        assert_eq!(sum.num_generators(), by_id.len());
        for (k, id) in sum.ids().iter().enumerate() {
            assert_eq!(sum.generator_row(k), by_id[id].as_slice(), "row of {id:?}");
        }
        // Pointwise: for random symbol assignments BY ID, the exact point of
        // a (+/-) b lies in the aligned result.
        let check = |result: &ZonoState, rng: &mut Rng| -> usize {
            let mut violations = 0;
            for _ in 0..500 {
                let e: HashMap<SymbolId, f64> = [1u64, 2, 3, 7]
                    .iter()
                    .map(|&i| (SymbolId(i), rng.sym()))
                    .collect();
                let eval = |z: &ZonoState, rho_scale: f64| -> Vec<f64> {
                    (0..n)
                        .map(|j| {
                            let mut v = z.center()[j] + rho_scale * z.remainder()[j];
                            for (k, id) in z.ids().iter().enumerate() {
                                v += z.generator_row(k)[j] * e[id];
                            }
                            v
                        })
                        .collect()
                };
                let rho = rng.sym();
                let pa = eval(&a, rho);
                let pb = eval(&b, -rho);
                let center_only = eval(result, 0.0);
                for j in 0..n {
                    let exact = pa[j] + sign * pb[j];
                    if (exact - center_only[j]).abs()
                        > result.remainder()[j] * (1.0 + 1e-12) + 1e-12
                    {
                        violations += 1;
                    }
                }
            }
            violations
        };
        assert_eq!(
            check(&sum, &mut rng),
            0,
            "ID-aligned result must contain a (+/-) b"
        );
        ids::fault::set_positional_alignment(true);
        let positional = add_aligned(&a, &b, subtract).expect("positional add");
        ids::fault::set_positional_alignment(false);
        assert!(
            check(&positional, &mut rng) > 0,
            "positional alignment must be caught by the by-ID containment check"
        );
    }
    // Duplicate IDs are refused at construction.
    assert!(ZonoState::from_parts(
        1,
        vec![0.0],
        vec![1.0, 2.0],
        vec![SymbolId(0), SymbolId(0)],
        vec![0.0]
    )
    .is_err());
}

#[test]
fn child_zono_program_refuses_unsupported_synthetic_graphs() {
    // An S-shaped activation is outside the op set.
    let mut rng = Rng::new(51);
    let w = Array2::from_shape_vec((3, 5), rand_f32(&mut rng, 15, 0.5)).expect("w");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "fc",
        Layer::Linear(LinearLayer::new(w, None).expect("fc")),
    ));
    graph.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["fc".to_string()],
    ));
    graph.set_output("sig");
    let refusal = ZonoProgram::compile_with_ibp_shapes(&graph, &root_box()).expect_err("refuse");
    assert!(
        matches!(refusal, ZonoRefusal::UnsupportedLayer { .. }),
        "{refusal:?}"
    );
    assert_eq!(refusal.token(), "unsupported_layer");
}

#[test]
fn spec_crown_bounds_stay_below_sampled_outputs() {
    let graph = small_cgan_graph(61, 0);
    let root = root_box();
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
    let opts = ZonoForwardOptions {
        max_generators: 16_384,
        record_trace: true,
        row_par: RowPar::Seq,
    };
    let mut rng = Rng::new(62);
    for depth in [0usize, 4, 8] {
        let child = random_child(&root, depth, &mut rng);
        let outcome = program
            .forward(&child, &opts, &[], None, None)
            .expect("forward");
        let trace = outcome.trace.as_ref().expect("trace");
        let queries = [
            SpecQuery {
                coeffs: vec![1.0],
                constant: 0.0,
            },
            SpecQuery {
                coeffs: vec![-1.0],
                constant: 0.25,
            },
        ];
        let lbs = spec_lower_bounds(&program, trace, &queries).expect("spec crown");
        let (points, rows) = sample_points(&child, 2_000, 63);
        let mut min_values = [f64::INFINITY; 2];
        program
            .evaluate_points(&points, rows, |step, values| {
                if step == Some(program.steps().len() - 1) {
                    for &y in values {
                        min_values[0] = min_values[0].min(y);
                        min_values[1] = min_values[1].min(0.25 - y);
                    }
                }
            })
            .expect("evaluate");
        for q in 0..2 {
            assert!(
                lbs[q] <= min_values[q],
                "depth {depth} query {q}: certified lb {} above sampled min {}",
                lbs[q],
                min_values[q]
            );
        }
        // The output box of the zonotope contains the samples too.
        assert!(outcome.output_lo[0] <= min_values[0]);
    }
}

#[test]
fn relu_transformer_holds_in_exact_rational_arithmetic() {
    use num_rational::BigRational;
    use num_traits::Zero;
    let rat = |v: f64| BigRational::from_float(v).expect("finite");
    let mut rng = Rng::new(71);
    let mut checked = 0usize;
    for trial in 0..3_000 {
        let scale = 10f64.powi(trial % 9 - 4);
        let c = rng.sym() * scale;
        let g = (0.05 + rng.unit()) * scale;
        let state = ZonoState::from_parts(1, vec![c], vec![g], vec![SymbolId(0)], vec![0.0])
            .expect("state");
        let abs_g = state.abs_generator_sum();
        let (lo, hi) = concretize(&state, &abs_g);
        let mut alloc = SymbolAllocator::new();
        let _ = alloc.fresh();
        let (out, _) = apply_relu(&state, &abs_g, &lo, &hi, &mut alloc).expect("relu");
        // relu(c + g e) must lie within  c' + G'_0 e +/- (sum_{new} |G'| + r').
        let c_out = rat(out.center()[0]);
        let g0 = rat(out.generator_row(0)[0]);
        let mut radius = rat(out.remainder()[0]);
        for k in 1..out.num_generators() {
            radius += rat(out.generator_row(k)[0].abs());
        }
        let (rc, rg) = (rat(c), rat(g));
        let mut es = vec![rat(-1.0), rat(1.0)];
        if lo[0] < 0.0 && hi[0] > 0.0 {
            es.push(-rc.clone() / rg.clone()); // the kink x = 0
        }
        for e in es {
            if e < rat(-1.0) || e > rat(1.0) {
                continue;
            }
            let x = rc.clone() + rg.clone() * e.clone();
            let relu = if x > BigRational::zero() {
                x
            } else {
                BigRational::zero()
            };
            let residual = relu - c_out.clone() - g0.clone() * e;
            let residual = if residual < BigRational::zero() {
                -residual
            } else {
                residual
            };
            assert!(
                residual <= radius,
                "trial {trial}: exact residual exceeds the certified radius (c={c}, g={g})"
            );
            checked += 1;
        }
    }
    assert!(checked > 6_000, "too few exact checks: {checked}");
}

#[test]
fn inherited_intersection_keeps_the_looser_side_when_crossed() {
    let mut lo = vec![-1.0, -1.0, -1.0];
    let mut hi = vec![1.0, 1.0, 1.0];
    // Element 0 tightens, element 1 is disjoint (kept), element 2 has NaN (ignored).
    let tightened =
        intersect_keep_looser(&mut lo, &mut hi, &[-0.5, 2.0, f64::NAN], &[0.5, 3.0, 0.0]);
    assert_eq!(tightened, 1);
    assert_eq!((lo[0], hi[0]), (-0.5, 0.5));
    assert_eq!(
        (lo[1], hi[1]),
        (-1.0, 1.0),
        "a crossed pair keeps the looser side"
    );
    assert_eq!((lo[2], hi[2]), (-1.0, 1.0));
}
