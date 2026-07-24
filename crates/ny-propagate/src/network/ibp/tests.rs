// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for IBP/CROWN-IBP propagation.

use super::forward::try_lower_dense_chain;
use super::helpers::{is_all_relu_stable, layer_output_needs_partial_crown};
use crate::layers::{
    Conv1dLayer, Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer, ReshapeLayer, SignLayer,
    SoftmaxLayer, SqrtLayer,
};
use crate::{Layer, Network};
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::{
    GemmEngine, GpuIbpForward, GpuIbpLayer, GpuIbpResult, NaiveCpuGemmEngine, NyError,
    Result as NyResult,
};
use ny_tensor::BoundedTensor;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ResidentOnlyEngine {
    resident_calls: AtomicUsize,
}

impl ResidentOnlyEngine {
    fn new() -> Self {
        Self {
            resident_calls: AtomicUsize::new(0),
        }
    }

    fn resident_calls(&self) -> usize {
        self.resident_calls.load(Ordering::SeqCst)
    }
}

impl GpuIbpForward for ResidentOnlyEngine {
    fn ibp_forward_gpu(
        &self,
        layers: &[GpuIbpLayer],
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> NyResult<GpuIbpResult> {
        self.resident_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(input_shape, &[4, 2]);
        assert_eq!(input_lower.len(), 8);
        assert_eq!(input_upper.len(), 8);
        assert_eq!(layers.len(), 3);
        assert!(matches!(
            layers[0],
            GpuIbpLayer::Linear {
                out_features: 3,
                in_features: 2,
                ..
            }
        ));
        assert!(matches!(layers[1], GpuIbpLayer::ReLU { num_elements: 12 }));
        assert!(matches!(
            layers[2],
            GpuIbpLayer::Linear {
                out_features: 1,
                in_features: 3,
                ..
            }
        ));

        Ok(GpuIbpResult {
            lower_bounds: vec![-1.0, -0.5, 0.0, 0.5],
            upper_bounds: vec![0.0, 0.5, 1.0, 1.5],
            output_shape: vec![4, 1],
        })
    }
}

impl GemmEngine for ResidentOnlyEngine {
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> NyResult<Vec<f32>> {
        panic!("resident dense-chain fast path should bypass per-layer GEMM")
    }

    fn as_gpu_ibp_forward(&self) -> Option<&dyn GpuIbpForward> {
        Some(self)
    }
}

struct ResidentConv2dEngine {
    resident_calls: AtomicUsize,
}

impl ResidentConv2dEngine {
    fn new() -> Self {
        Self {
            resident_calls: AtomicUsize::new(0),
        }
    }

    fn resident_calls(&self) -> usize {
        self.resident_calls.load(Ordering::SeqCst)
    }
}

impl GpuIbpForward for ResidentConv2dEngine {
    fn ibp_forward_gpu(
        &self,
        layers: &[GpuIbpLayer],
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> NyResult<GpuIbpResult> {
        self.resident_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(input_shape, &[1, 1, 2, 2]);
        assert_eq!(input_lower.len(), 4);
        assert_eq!(input_upper.len(), 4);
        assert_eq!(layers.len(), 2);
        assert!(matches!(
            layers[0],
            GpuIbpLayer::Conv2d {
                out_channels: 1,
                in_channels: 1,
                kernel_h: 1,
                kernel_w: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                groups: 1,
                input_h: 2,
                input_w: 2,
                ..
            }
        ));
        assert!(matches!(layers[1], GpuIbpLayer::ReLU { num_elements: 4 }));

        Ok(GpuIbpResult {
            lower_bounds: vec![-0.25, 0.0, 0.5, 1.0],
            upper_bounds: vec![0.25, 0.5, 1.0, 1.5],
            output_shape: vec![1, 1, 2, 2],
        })
    }
}

impl GemmEngine for ResidentConv2dEngine {
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> NyResult<Vec<f32>> {
        panic!("resident Conv2d fast path should bypass per-layer GEMM")
    }

    fn as_gpu_ibp_forward(&self) -> Option<&dyn GpuIbpForward> {
        Some(self)
    }
}

struct ResidentFallbackEngine {
    resident_calls: AtomicUsize,
    gemm_calls: AtomicUsize,
}

impl ResidentFallbackEngine {
    fn new() -> Self {
        Self {
            resident_calls: AtomicUsize::new(0),
            gemm_calls: AtomicUsize::new(0),
        }
    }

    fn resident_calls(&self) -> usize {
        self.resident_calls.load(Ordering::SeqCst)
    }

    fn gemm_calls(&self) -> usize {
        self.gemm_calls.load(Ordering::SeqCst)
    }
}

impl GpuIbpForward for ResidentFallbackEngine {
    fn ibp_forward_gpu(
        &self,
        _layers: &[GpuIbpLayer],
        _input_lower: &[f32],
        _input_upper: &[f32],
        _input_shape: &[usize],
    ) -> NyResult<GpuIbpResult> {
        self.resident_calls.fetch_add(1, Ordering::SeqCst);
        Err(NyError::InternalError(
            "unsupported network should stay on per-layer IBP fallback".into(),
        ))
    }
}

impl GemmEngine for ResidentFallbackEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> NyResult<Vec<f32>> {
        self.gemm_calls.fetch_add(1, Ordering::SeqCst);
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_ibp_forward(&self) -> Option<&dyn GpuIbpForward> {
        Some(self)
    }
}

fn linear_layer() -> Layer {
    Layer::Linear(LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap())
}

#[test]
fn test_layer_output_needs_partial_crown_skips_relu_before_linear() {
    let layers = vec![linear_layer(), Layer::ReLU(ReLULayer), linear_layer()];

    assert!(!layer_output_needs_partial_crown(&layers, 1));
}

#[test]
fn test_layer_output_needs_partial_crown_keeps_exact_layer_before_relu() {
    let layers = vec![linear_layer(), Layer::ReLU(ReLULayer), linear_layer()];

    assert!(layer_output_needs_partial_crown(&layers, 0));
}

#[test]
fn test_layer_output_needs_partial_crown_keeps_softmax_before_linear() {
    let layers = vec![Layer::Softmax(SoftmaxLayer::new(-1)), linear_layer()];

    assert!(layer_output_needs_partial_crown(&layers, 0));
}

#[test]
fn test_is_all_relu_stable_positive_region() {
    // All elements have lower >= 0 → all in identity region
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.5, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 0.5]).unwrap(),
    )
    .unwrap();
    assert!(is_all_relu_stable(&bounds));
}

#[test]
fn test_is_all_relu_stable_negative_region() {
    // All elements have upper <= 0 → all in zero region
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -1.0, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.1, -0.01, 0.0]).unwrap(),
    )
    .unwrap();
    assert!(is_all_relu_stable(&bounds));
}

#[test]
fn test_is_all_relu_stable_mixed_stable() {
    // Mix of positive-stable and negative-stable neurons
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.1, -2.0, 0.5, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, -0.1, 2.0, 0.0]).unwrap(),
    )
    .unwrap();
    assert!(is_all_relu_stable(&bounds));
}

#[test]
fn test_is_all_relu_stable_one_unstable() {
    // One element crosses zero → not all stable
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, -0.5, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 0.5, 2.0]).unwrap(),
    )
    .unwrap();
    assert!(!is_all_relu_stable(&bounds));
}

#[test]
fn test_is_all_relu_stable_empty() {
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
    )
    .unwrap();
    assert!(is_all_relu_stable(&bounds));
}

#[test]
fn test_relu_stable_skip_does_not_apply_to_sqrt_successor() {
    // x -> [x, -x] -> ReLU -> sum + 1 = |x| + 1.
    //
    // The pre-Sqrt interval is strictly positive, so a generic "all-stable"
    // shortcut would wrongly skip the partial CROWN pass and keep the loose
    // IBP upper bound 3.0. The correct CROWN-IBP intermediate is 2.0.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [-1.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[1.0]))).unwrap(),
    ));
    network.add_layer(Layer::Sqrt(SqrtLayer::new()));

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
    let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    assert!(
        (ibp_bounds[2].upper()[[0]] - 3.0).abs() < 1e-5,
        "sanity check: IBP upper bound should reflect independent ReLU relaxation"
    );
    assert!(
        crown_ibp_bounds[2].upper()[[0]] <= 2.0 + 1e-4,
        "positive pre-Sqrt interval still needs partial CROWN tightening"
    );
    assert!(
        crown_ibp_bounds[2].upper()[[0]] + 1e-4 < ibp_bounds[2].upper()[[0]],
        "CROWN-IBP should tighten the pre-Sqrt bound even though it is one-sign"
    );
}

// #tll-nested-collect-par: the nested-faer-parallel input-split collection must
// stay a SOUND OUTWARD enclosure. This runs the CROWN-IBP collection on a WIDE
// ReLU net (matrices big enough to hit the faer f64 aw path) from INSIDE a Rayon
// worker with `NestedFaerParGuard` active — i.e. the exact `Par::Rayon` nested
// path the fix enables — and asserts every collected intermediate pre-activation
// bound encloses the TRUE forward pre-activation at many sampled input points
// (a degenerate [p,p] box makes IBP exact for affine+ReLU, so its forward value
// IS the true value). A too-tight (non-enclosing) intermediate would be the
// catastrophic false-UNSAT failure mode; this catches it.
#[test]
fn test_nested_par_collection_encloses_true_forward_wide_relu() {
    let _env_lock = ny_test_utils::env::lock_env();
    let _budget = ny_test_utils::env::ScopedEnvVar::set("NY_DENSE_BUDGET_MB", "2048");
    use crate::faer_parallelism::NestedFaerParGuard;
    use ny_tensor::BoundedTensor;
    use rayon::prelude::*;

    // 2 -> 64 -> ReLU -> 64 -> ReLU -> 1 wide net with mixed-sign weights so the
    // pre-activation intervals straddle zero (unstable ReLUs => real CROWN work).
    let w = 64usize;
    let build = || {
        let mut network = Network::new();
        let l1 = ndarray::Array2::from_shape_fn((w, 2), |(i, j)| {
            (((i * 7 + j * 3) % 11) as f32 - 5.0) * 0.1
        });
        let b1 = ndarray::Array1::from_shape_fn(w, |i| ((i % 5) as f32 - 2.0) * 0.05);
        network.add_layer(Layer::Linear(LinearLayer::new(l1, Some(b1)).unwrap()));
        network.add_layer(Layer::ReLU(ReLULayer));
        let l2 = ndarray::Array2::from_shape_fn((w, w), |(i, j)| {
            (((i * 3 + j * 5) % 13) as f32 - 6.0) * 0.05
        });
        let b2 = ndarray::Array1::from_shape_fn(w, |i| ((i % 7) as f32 - 3.0) * 0.03);
        network.add_layer(Layer::Linear(LinearLayer::new(l2, Some(b2)).unwrap()));
        network.add_layer(Layer::ReLU(ReLULayer));
        let l3 = ndarray::Array2::from_shape_fn((1, w), |(_, j)| ((j % 3) as f32 - 1.0) * 0.1);
        network.add_layer(Layer::Linear(
            LinearLayer::new(l3, Some(arr1(&[0.25]))).unwrap(),
        ));
        network
    };
    let network = build();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Collect from INSIDE Rayon workers with the nested-par guard active (the
    // fix's path). Also collect on the main thread (guard inactive => historical
    // path) for a per-element parity/enclosure cross-check.
    let nested_bounds = (0..4usize)
        .into_par_iter()
        .map(|_| {
            let _g = NestedFaerParGuard::new();
            network.collect_crown_ibp_bounds(&input).unwrap()
        })
        .collect::<Vec<_>>();
    let main_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    // Enclosure: sample the box, forward each point exactly (degenerate box),
    // and require the collected intermediate bounds to enclose the true value.
    let samples = [
        [-1.0f32, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [0.5, -0.3],
        [-0.7, 0.9],
        [0.25, 0.75],
    ];
    let collected = &nested_bounds[0];
    for p in samples.iter() {
        let point = BoundedTensor::new(arr1(p).into_dyn(), arr1(p).into_dyn()).unwrap();
        // Exact per-layer forward pre-activations at this point.
        let exact = network.collect_ibp_bounds(&point).unwrap();
        for (k, ex) in exact.iter().enumerate() {
            let lo = collected[k].lower();
            let hi = collected[k].upper();
            for idx in 0..ex.len() {
                let tv = ex.lower().as_slice().unwrap()[idx]; // exact (lower==upper)
                let l = lo.as_slice().unwrap()[idx];
                let h = hi.as_slice().unwrap()[idx];
                assert!(
                    l <= tv + 1e-3 && tv <= h + 1e-3,
                    "nested-par collection NOT enclosing at layer {k} elem {idx}: \
                     true={tv} not in [{l}, {h}]"
                );
            }
        }
    }

    // Cross-check nested vs main: same shapes, and nested bounds enclose within a
    // tiny f32 tolerance (summation-order-only difference; γ_n·S is
    // order-independent so both are valid, near-identical enclosures).
    for (k, mb) in main_bounds.iter().enumerate() {
        assert_eq!(collected[k].shape(), mb.shape(), "layer {k} shape mismatch");
        for idx in 0..mb.len() {
            let ml = mb.lower().as_slice().unwrap()[idx];
            let mh = mb.upper().as_slice().unwrap()[idx];
            let nl = collected[k].lower().as_slice().unwrap()[idx];
            let nh = collected[k].upper().as_slice().unwrap()[idx];
            assert!(
                (ml - nl).abs() < 1e-2 && (mh - nh).abs() < 1e-2,
                "layer {k} elem {idx}: nested [{nl},{nh}] vs main [{ml},{mh}] diverge"
            );
        }
    }
}

// ── try_lower_dense_chain tests ──────────────────────────────────────────────

#[test]
fn test_try_lower_dense_chain_linear_relu_linear() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]), Some(arr1(&[0.5, -0.5]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0]]), None).unwrap(),
    ));

    let input_shape = &[2usize];
    let gpu_layers = try_lower_dense_chain(&network, input_shape);
    let gpu_layers = gpu_layers.expect("pure dense chain should lower");
    assert_eq!(gpu_layers.len(), 3);
    assert!(matches!(
        gpu_layers[0],
        GpuIbpLayer::Linear {
            out_features: 2,
            in_features: 2,
            ..
        }
    ));
    assert!(matches!(
        gpu_layers[1],
        GpuIbpLayer::ReLU { num_elements: 2 }
    ));
    assert!(matches!(
        gpu_layers[2],
        GpuIbpLayer::Linear {
            out_features: 1,
            in_features: 2,
            ..
        }
    ));
}

#[test]
fn test_try_lower_dense_chain_rejects_softmax() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), None).unwrap(),
    ));
    network.add_layer(Layer::Softmax(SoftmaxLayer::new(-1)));

    let input_shape = &[1usize];
    let result = try_lower_dense_chain(&network, input_shape);
    assert!(result.is_none(), "unsupported layer should cause fallback");
}

#[test]
fn test_try_lower_dense_chain_rejects_sign_4081() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), None).unwrap(),
    ));
    network.add_layer(Layer::Sign(SignLayer::new()));

    let result = try_lower_dense_chain(&network, &[1usize]);
    assert!(result.is_none(), "Sign must stay on the fallback path");
}

#[test]
fn test_try_lower_dense_chain_accepts_conv2d_4275() {
    let mut network = Network::new();
    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0]).unwrap(),
        None,
        (1, 1),
        (0, 0),
        2,
        2,
    )
    .unwrap();
    network.add_layer(Layer::Conv2d(conv));

    let gpu_layers = try_lower_dense_chain(&network, &[1usize, 2, 2])
        .expect("groups=1 Conv2d should lower into the resident subset");
    assert_eq!(gpu_layers.len(), 1);
    assert!(matches!(
        gpu_layers[0],
        GpuIbpLayer::Conv2d {
            out_channels: 1,
            in_channels: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            groups: 1,
            input_h: 2,
            input_w: 2,
            ..
        }
    ));
}

#[test]
fn test_try_lower_dense_chain_rejects_linear_input_feature_mismatch() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0]]), None).unwrap(),
    ));

    let input_shape = &[3usize];
    let result = try_lower_dense_chain(&network, input_shape);
    assert!(
        result.is_none(),
        "shape-mismatched linear chain should fall back"
    );
}

#[test]
fn test_try_lower_dense_chain_with_flatten() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), None).unwrap(),
    ));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0]]), None).unwrap(),
    ));

    let input_shape = &[2usize];
    let gpu_layers = try_lower_dense_chain(&network, input_shape)
        .expect("Linear + Flatten + Linear should lower");
    assert_eq!(gpu_layers.len(), 3);
    assert!(matches!(gpu_layers[1], GpuIbpLayer::View { .. }));
}

#[test]
fn test_try_lower_dense_chain_with_reshape() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), None).unwrap(),
    ));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![2])));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input_shape = &[2usize];
    let gpu_layers =
        try_lower_dense_chain(&network, input_shape).expect("Linear + Reshape + ReLU should lower");
    assert_eq!(gpu_layers.len(), 3);
    assert!(matches!(gpu_layers[1], GpuIbpLayer::View { .. }));
    assert!(matches!(
        gpu_layers[2],
        GpuIbpLayer::ReLU { num_elements: 2 }
    ));
}

#[test]
fn test_propagate_ibp_with_engine_prefers_resident_dense_chain_for_batched_inputs() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
            Some(arr1(&[0.0, 0.0, 0.0])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0, 0.0]]), None).unwrap(),
    ));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![-1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0; 8]).unwrap(),
    )
    .unwrap();

    let engine = ResidentOnlyEngine::new();
    let output = network
        .propagate_ibp_with_engine(&input, Some(&engine))
        .expect("resident engine should handle dense batched chains");

    assert_eq!(engine.resident_calls(), 1);
    assert_eq!(output.shape(), vec![4, 1]);
    assert_eq!(
        output.lower(),
        &ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![-1.0, -0.5, 0.0, 0.5]).unwrap()
    );
    assert_eq!(
        output.upper(),
        &ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![0.0, 0.5, 1.0, 1.5]).unwrap()
    );
}

#[test]
fn test_propagate_ibp_with_engine_falls_back_for_sign_network_4081() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, -1.0]]), None).unwrap(),
    ));
    network.add_layer(Layer::Sign(SignLayer::new()));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.5, 0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.75]).unwrap(),
    )
    .unwrap();

    let engine = ResidentFallbackEngine::new();
    let output = network
        .propagate_ibp_with_engine(&input, Some(&engine))
        .expect("Sign network should use per-layer fallback");

    assert_eq!(engine.resident_calls(), 0);
    assert!(
        engine.gemm_calls() > 0,
        "fallback path should still use linear GEMM, got {} calls",
        engine.gemm_calls()
    );
    assert_eq!(output.shape(), vec![1]);
}

#[test]
fn test_propagate_ibp_with_engine_falls_back_for_conv1d_network_4081() {
    let mut network = Network::new();
    let conv = Conv1dLayer::with_input_length(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0]).unwrap(),
        None,
        1,
        0,
        4,
    )
    .unwrap();
    network.add_layer(Layer::Conv1d(conv));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![-1.0, -0.5, 0.0, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![0.0, 0.5, 1.0, 1.5]).unwrap(),
    )
    .unwrap();

    let engine = ResidentFallbackEngine::new();
    let output = network
        .propagate_ibp_with_engine(&input, Some(&engine))
        .expect("Conv1d network should use per-layer fallback");

    assert_eq!(engine.resident_calls(), 0);
    assert!(
        engine.gemm_calls() > 0,
        "fallback path should still use Conv1d GEMM kernels"
    );
    assert_eq!(output.shape(), vec![1, 1, 4]);
}

#[test]
fn test_propagate_ibp_with_engine_prefers_resident_for_conv2d_network_4275() {
    let mut network = Network::new();
    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0]).unwrap(),
        None,
        (1, 1),
        (0, 0),
        2,
        2,
    )
    .unwrap();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![-1.0, -0.5, 0.0, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.0, 0.5, 1.0, 1.5]).unwrap(),
    )
    .unwrap();

    let engine = ResidentConv2dEngine::new();
    let output = network
        .propagate_ibp_with_engine(&input, Some(&engine))
        .expect("groups=1 Conv2d network should use the resident path");

    assert_eq!(engine.resident_calls(), 1);
    assert_eq!(output.shape(), vec![1, 1, 2, 2]);
    assert_eq!(
        output.lower(),
        &ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![-0.25, 0.0, 0.5, 1.0]).unwrap()
    );
    assert_eq!(
        output.upper(),
        &ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.25, 0.5, 1.0, 1.5]).unwrap()
    );
}

#[test]
fn test_propagate_ibp_with_engine_falls_back_for_grouped_conv2d_network_4275() {
    let mut network = Network::new();
    let conv = Conv2dLayer::with_input_shape_full(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![1.0, -1.0]).unwrap(),
        None,
        (1, 1),
        (0, 0),
        2,
        2,
        2,
    )
    .unwrap();
    network.add_layer(Layer::Conv2d(conv));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 2]), vec![-1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 2]), vec![1.0; 8]).unwrap(),
    )
    .unwrap();

    // #4275 (updated): grouped Conv2d IBP forward now ROUTES through the injected
    // GemmEngine via per-group 2D GEMMs (so --backend wgpu / Metal can accelerate
    // the dominant conv GEMMs). The old contract (gemm_calls()==0, CPU-only
    // fallback) is obsolete. SOUNDNESS: the engine-routed result must be
    // numerically identical to the CPU faer path — assert that here.
    let engine = ResidentFallbackEngine::new();
    let output = network
        .propagate_ibp_with_engine(&input, Some(&engine))
        .expect("grouped Conv2d should propagate via the engine GEMM path");
    let output_cpu = network
        .propagate_ibp_with_engine(&input, None)
        .expect("grouped Conv2d CPU reference");

    assert!(
        engine.gemm_calls() > 0,
        "grouped Conv2d should now route per-group GEMMs through the engine, got {} calls",
        engine.gemm_calls()
    );
    assert_eq!(output.shape(), vec![1, 2, 2, 2]);
    // Engine-routed bounds must equal the CPU faer bounds (numerically faithful reroute).
    assert_eq!(output.lower(), output_cpu.lower());
    assert_eq!(output.upper(), output_cpu.upper());
}

#[test]
fn test_try_lower_dense_chain_empty_network() {
    let network = Network::new();
    let input_shape = &[4usize];
    let gpu_layers = try_lower_dense_chain(&network, input_shape)
        .expect("empty network should lower to empty list");
    assert!(gpu_layers.is_empty());
}

/// Verify that `try_lower_dense_chain` rejects networks with Flatten(0) when the
/// input has a prepended restart batch axis, because Flatten(0) folds the batch dim
/// into features, creating a shape mismatch with the next Linear layer.
///
/// This invariant is the safety guarantee that allows the GPU resident IBP path to
/// fire for both Plain and PreserveLeadingAxis modes without an explicit mode guard
/// (#4345). Without this rejection, the GPU path would apply regular Flatten
/// semantics and produce incorrect bounds for PreserveLeadingAxis inputs.
#[test]
fn test_try_lower_dense_chain_rejects_flatten0_with_restart_batch_dim_4345() {
    // Network: Linear(2->3) -> ReLU -> Flatten(0) -> Linear(3->1)
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]), None).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0, 0.0]]), None).unwrap(),
    ));

    // With a single-sample input [2], the lowerer succeeds (Flatten(0) is a no-op
    // on 1D, shape stays [3], and the next Linear's in_features=3 matches).
    let single = try_lower_dense_chain(&network, &[2usize]);
    assert!(
        single.is_some(),
        "single-sample input should lower (Flatten(0) is identity on 1D)"
    );

    // With a batched input [R, 2] where R > 1, Flatten(0) produces shape [R*3]
    // but the next Linear expects in_features=3. Shape mismatch → None.
    let batched = try_lower_dense_chain(&network, &[4, 2]);
    assert!(
        batched.is_none(),
        "batched input with Flatten(0) must reject: Flatten folds restart axis \
         into features (shape [12] vs next Linear in_features=3)"
    );
}

/// End-to-end test: `propagate_ibp_with_engine_preserve_leading_axis` falls back
/// to the per-layer loop (not the GPU resident path) when the network contains
/// Flatten(0) and the input has a restart batch dim (#4345).
#[test]
fn test_preserve_leading_axis_flatten0_falls_back_from_resident_4345() {
    // Network: Linear(2->3) -> ReLU -> Flatten(0) -> Linear(3->1)
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]), None).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0, 0.0]]), None).unwrap(),
    ));

    // Batched input [2, 2] simulating 2 restart domains over 2-feature input.
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -1.0, 0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 1.0, 0.5, 0.5]).unwrap(),
    )
    .unwrap();

    let engine = ResidentFallbackEngine::new();
    let output = network
        .propagate_ibp_with_engine_preserve_leading_axis(&input, Some(&engine))
        .expect("PreserveLeadingAxis with Flatten(0) should use per-layer fallback");

    // Resident path must NOT fire — Flatten(0) with batched input is rejected.
    assert_eq!(
        engine.resident_calls(),
        0,
        "GPU resident path must not fire for Flatten(0) with restart batch dim"
    );
    // Per-layer GEMM should fire (at least for the two Linear layers).
    assert!(
        engine.gemm_calls() >= 2,
        "per-layer fallback should use GEMM for Linear layers, got {} calls",
        engine.gemm_calls()
    );
    // Output should preserve the leading restart axis.
    assert_eq!(
        output.shape(),
        vec![2, 1],
        "output must preserve leading restart axis [2, 1], not flatten to [2]"
    );
}
