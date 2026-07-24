// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared test builders and fake engine for attacker-local regressions.

use ndarray::{arr1, arr2};
use ny_core::{
    GemmEngine, GpuIbpForward, GpuIbpForwardExt, GpuIbpLayer, GpuIbpModelPlan, GpuIbpResult,
    NaiveCpuGemmEngine, Result as NyResult,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::layers::*;
use crate::Network;

pub(super) fn sign_threshold_network(threshold: f32) -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[-threshold]))).unwrap(),
    ));
    network.add_layer(Layer::Sign(SignLayer::new()));
    network
}

pub(super) struct ResidentCountingEngine {
    resident_calls: AtomicUsize,
    gemm_calls: AtomicUsize,
}

impl ResidentCountingEngine {
    pub(super) fn new() -> Self {
        Self {
            resident_calls: AtomicUsize::new(0),
            gemm_calls: AtomicUsize::new(0),
        }
    }

    pub(super) fn resident_calls(&self) -> usize {
        self.resident_calls.load(Ordering::SeqCst)
    }

    pub(super) fn gemm_calls(&self) -> usize {
        self.gemm_calls.load(Ordering::SeqCst)
    }
}

impl GpuIbpForward for ResidentCountingEngine {
    fn ibp_forward_gpu(
        &self,
        layers: &[GpuIbpLayer],
        input_lower: &[f32],
        input_upper: &[f32],
        input_shape: &[usize],
    ) -> NyResult<GpuIbpResult> {
        self.resident_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(input_shape, &[4, 2]);
        assert_eq!(input_lower, input_upper, "PGD batch uses concrete inputs");
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
            lower_bounds: vec![0.25, 0.5, 0.75, 1.0],
            upper_bounds: vec![0.25, 0.5, 0.75, 1.0],
            output_shape: vec![4, 1],
        })
    }
}

impl GemmEngine for ResidentCountingEngine {
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> NyResult<Vec<f32>> {
        self.gemm_calls.fetch_add(1, Ordering::SeqCst);
        panic!("resident dense-chain PGD path should bypass per-layer GEMM");
    }

    fn as_gpu_ibp_forward(&self) -> Option<&dyn GpuIbpForward> {
        Some(self)
    }
}

#[derive(Default)]
struct CachedPlanCounters {
    plan_preparations: AtomicUsize,
    cached_calls: AtomicUsize,
    resident_calls: AtomicUsize,
    gemm_calls: AtomicUsize,
}

struct CachedResidentPlan {
    counters: Arc<CachedPlanCounters>,
    input_shape: Vec<usize>,
}

impl GpuIbpModelPlan for CachedResidentPlan {
    fn ibp_forward_cached(
        &self,
        _input_lower: &[f32],
        _input_upper: &[f32],
        input_shape: &[usize],
    ) -> NyResult<GpuIbpResult> {
        self.counters.cached_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(input_shape, self.input_shape.as_slice());
        Ok(GpuIbpResult {
            lower_bounds: vec![0.25, 0.5, 0.75, 1.0],
            upper_bounds: vec![0.25, 0.5, 0.75, 1.0],
            output_shape: vec![4, 1],
        })
    }
}

pub(super) struct CachedPlanCountingEngine {
    counters: Arc<CachedPlanCounters>,
}

impl CachedPlanCountingEngine {
    pub(super) fn new() -> Self {
        Self {
            counters: Arc::new(CachedPlanCounters::default()),
        }
    }

    pub(super) fn plan_preparations(&self) -> usize {
        self.counters.plan_preparations.load(Ordering::SeqCst)
    }

    pub(super) fn cached_calls(&self) -> usize {
        self.counters.cached_calls.load(Ordering::SeqCst)
    }

    pub(super) fn resident_calls(&self) -> usize {
        self.counters.resident_calls.load(Ordering::SeqCst)
    }

    pub(super) fn gemm_calls(&self) -> usize {
        self.counters.gemm_calls.load(Ordering::SeqCst)
    }
}

impl GpuIbpForward for CachedPlanCountingEngine {
    fn ibp_forward_gpu(
        &self,
        _layers: &[GpuIbpLayer],
        _input_lower: &[f32],
        _input_upper: &[f32],
        _input_shape: &[usize],
    ) -> NyResult<GpuIbpResult> {
        self.counters.resident_calls.fetch_add(1, Ordering::SeqCst);
        panic!("cached PGD model-plan path should bypass one-shot resident IBP");
    }
}

impl GpuIbpForwardExt for CachedPlanCountingEngine {
    fn prepare_model_plan(
        &self,
        layers: &[GpuIbpLayer],
        input_shape: &[usize],
    ) -> NyResult<Option<Box<dyn GpuIbpModelPlan>>> {
        self.counters
            .plan_preparations
            .fetch_add(1, Ordering::SeqCst);
        assert_eq!(input_shape, &[4, 2]);
        assert_eq!(layers.len(), 3);
        Ok(Some(Box::new(CachedResidentPlan {
            counters: Arc::clone(&self.counters),
            input_shape: input_shape.to_vec(),
        })))
    }
}

impl GemmEngine for CachedPlanCountingEngine {
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> NyResult<Vec<f32>> {
        self.counters.gemm_calls.fetch_add(1, Ordering::SeqCst);
        panic!("cached PGD model-plan path should bypass per-layer GEMM");
    }

    fn as_gpu_ibp_forward(&self) -> Option<&dyn GpuIbpForward> {
        Some(self)
    }

    fn as_gpu_ibp_forward_ext(&self) -> Option<&dyn GpuIbpForwardExt> {
        Some(self)
    }
}

pub(super) struct UnsupportedModelFallbackEngine {
    plan_preparations: AtomicUsize,
    resident_calls: AtomicUsize,
    gemm_calls: AtomicUsize,
}

impl UnsupportedModelFallbackEngine {
    pub(super) fn new() -> Self {
        Self {
            plan_preparations: AtomicUsize::new(0),
            resident_calls: AtomicUsize::new(0),
            gemm_calls: AtomicUsize::new(0),
        }
    }

    pub(super) fn plan_preparations(&self) -> usize {
        self.plan_preparations.load(Ordering::SeqCst)
    }

    pub(super) fn resident_calls(&self) -> usize {
        self.resident_calls.load(Ordering::SeqCst)
    }

    pub(super) fn gemm_calls(&self) -> usize {
        self.gemm_calls.load(Ordering::SeqCst)
    }
}

impl GpuIbpForward for UnsupportedModelFallbackEngine {
    fn ibp_forward_gpu(
        &self,
        _layers: &[GpuIbpLayer],
        _input_lower: &[f32],
        _input_upper: &[f32],
        _input_shape: &[usize],
    ) -> NyResult<GpuIbpResult> {
        self.resident_calls.fetch_add(1, Ordering::SeqCst);
        panic!("unsupported PGD network should skip the resident dense-chain path");
    }
}

impl GpuIbpForwardExt for UnsupportedModelFallbackEngine {
    fn prepare_model_plan(
        &self,
        _layers: &[GpuIbpLayer],
        _input_shape: &[usize],
    ) -> NyResult<Option<Box<dyn GpuIbpModelPlan>>> {
        self.plan_preparations.fetch_add(1, Ordering::SeqCst);
        panic!("unsupported PGD network should skip cached-plan preparation");
    }
}

impl GemmEngine for UnsupportedModelFallbackEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> NyResult<Vec<f32>> {
        self.gemm_calls.fetch_add(1, Ordering::SeqCst);
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_ibp_forward(&self) -> Option<&dyn GpuIbpForward> {
        Some(self)
    }

    fn as_gpu_ibp_forward_ext(&self) -> Option<&dyn GpuIbpForwardExt> {
        Some(self)
    }
}
