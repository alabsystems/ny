// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression: prove that `adv_check` selects the graph-DAG cached-plan path
//! when a DAG-capable engine is present, and avoids it on unsupported graphs.
//!
//! Part of #4276 — mirrors the sequential PGD counting-engine pattern from
//! `pgd_attack/attacker/tests/common.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::{
    GemmEngine, GpuDagIbpForwardExt, GpuDagIbpModelPlan, GpuDagIbpPlanDesc, GpuIbpResult,
    NaiveCpuGemmEngine, Result as NyResult,
};
use ny_tensor::BoundedTensor;

use crate::layers::activations::ReLULayer;
use crate::layers::binary_ops::AddLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::misc::SignLayer;
use crate::layers::Layer;
use crate::{GraphNetwork, GraphNode};

use super::super::adv_check::try_adv_check_on_domain;

// ---------------------------------------------------------------------------
// Counting engine: DAG cached-plan path
// ---------------------------------------------------------------------------

#[derive(Default)]
struct DagCachedPlanCounters {
    plan_preparations: AtomicUsize,
    cached_calls: AtomicUsize,
}

struct DagCachedPlan {
    counters: Arc<DagCachedPlanCounters>,
    output_shape: Vec<usize>,
}

impl GpuDagIbpModelPlan for DagCachedPlan {
    fn dag_ibp_forward_cached(
        &self,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _input_shape: &[usize],
    ) -> NyResult<GpuIbpResult> {
        self.counters.cached_calls.fetch_add(1, Ordering::SeqCst);
        let n = self.output_shape.iter().product::<usize>();
        Ok(GpuIbpResult {
            lower_bounds: vec![0.1; n],
            upper_bounds: vec![0.2; n],
            output_shape: self.output_shape.clone(),
        })
    }
}

struct DagCachedPlanCountingEngine {
    counters: Arc<DagCachedPlanCounters>,
}

impl DagCachedPlanCountingEngine {
    fn new() -> Self {
        Self {
            counters: Arc::new(DagCachedPlanCounters::default()),
        }
    }

    fn plan_preparations(&self) -> usize {
        self.counters.plan_preparations.load(Ordering::SeqCst)
    }

    fn cached_calls(&self) -> usize {
        self.counters.cached_calls.load(Ordering::SeqCst)
    }
}

impl GpuDagIbpForwardExt for DagCachedPlanCountingEngine {
    fn prepare_dag_model_plan(
        &self,
        plan: &GpuDagIbpPlanDesc,
    ) -> NyResult<Option<Box<dyn GpuDagIbpModelPlan>>> {
        self.counters
            .plan_preparations
            .fetch_add(1, Ordering::SeqCst);
        // Compute output shape from the plan descriptor's output op.
        let output_shape = dag_output_shape(plan);
        Ok(Some(Box::new(DagCachedPlan {
            counters: Arc::clone(&self.counters),
            output_shape,
        })))
    }
}

impl GemmEngine for DagCachedPlanCountingEngine {
    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> NyResult<Vec<f32>> {
        panic!("DAG cached-plan path should bypass per-layer GEMM");
    }

    fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn GpuDagIbpForwardExt> {
        Some(self)
    }
}

/// Derive output shape from a plan descriptor. Simple heuristic for test use:
/// inspects the output op to determine element count.
fn dag_output_shape(plan: &GpuDagIbpPlanDesc) -> Vec<usize> {
    use ny_core::GpuDagIbpOp;
    match &plan.ops[plan.output_op_idx] {
        GpuDagIbpOp::Linear { out_features, .. } => vec![*out_features],
        GpuDagIbpOp::ReLU { num_elements, .. } => vec![*num_elements],
        GpuDagIbpOp::Add { num_elements, .. } => vec![*num_elements],
        GpuDagIbpOp::View { output_shape, .. } => output_shape.to_vec(),
        GpuDagIbpOp::Conv2d {
            out_channels,
            input_h,
            input_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            kernel_h,
            kernel_w,
            ..
        } => {
            let oh = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
            let ow = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
            vec![*out_channels, oh, ow]
        }
        GpuDagIbpOp::AveragePool {
            channels,
            output_h,
            output_w,
            ..
        } => vec![*channels, *output_h, *output_w],
    }
}

// ---------------------------------------------------------------------------
// Counting engine: unsupported graph (GEMM delegates to CPU)
// ---------------------------------------------------------------------------

/// Engine that advertises DAG capability but delegates per-layer GEMM to
/// NaiveCpuGemmEngine. Used for graphs where `try_lower_graph_dag` returns
/// None — the DAG methods should never be called, but the CPU graph IBP
/// fallback loop will use `gemm_f32` for Linear/Conv layers.
struct DagUnsupportedFallbackEngine {
    counters: Arc<DagCachedPlanCounters>,
}

impl DagUnsupportedFallbackEngine {
    fn new() -> Self {
        Self {
            counters: Arc::new(DagCachedPlanCounters::default()),
        }
    }

    fn plan_preparations(&self) -> usize {
        self.counters.plan_preparations.load(Ordering::SeqCst)
    }

    fn cached_calls(&self) -> usize {
        self.counters.cached_calls.load(Ordering::SeqCst)
    }
}

impl GpuDagIbpForwardExt for DagUnsupportedFallbackEngine {
    fn prepare_dag_model_plan(
        &self,
        _plan: &GpuDagIbpPlanDesc,
    ) -> NyResult<Option<Box<dyn GpuDagIbpModelPlan>>> {
        self.counters
            .plan_preparations
            .fetch_add(1, Ordering::SeqCst);
        panic!("unsupported graph should never reach prepare_dag_model_plan");
    }
}

impl GemmEngine for DagUnsupportedFallbackEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> NyResult<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }

    fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn GpuDagIbpForwardExt> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

/// Residual-add DAG: input → linear1 → relu → linear2 → Add(input, linear2).
/// This graph is DAG-lowerable (all ops supported).
fn build_dag_lowerable_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]);
    let b1 = arr1(&[0.1_f32, -0.05]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.6_f32, -0.2], [-0.4, 0.7]]);
    let b2 = arr1(&[0.0_f32, 0.0]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));

    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        "_input",
        "linear2",
    ));
    graph.set_output("residual");
    graph
}

/// Graph with Sign layer — unsupported by DAG lowering.
/// Sign is a non-linear unary op not in the DAG plan enum.
fn build_unsupported_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[0.5_f32, -0.2], [0.3, 0.8]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "sign",
        Layer::Sign(SignLayer::new()),
        vec!["linear1".to_string()],
    ));
    graph.set_output("sign");
    graph
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Prove that `try_adv_check_on_domain` reaches the DAG cached-plan fast path
/// (prepare_dag_model_plan + dag_ibp_forward_cached) when a DAG-capable engine
/// is present and the graph is DAG-lowerable.
///
/// Regression for #4276.
#[test]
fn test_adv_check_dag_lowerable_graph_hits_cached_plan_path_4276() {
    let graph = build_dag_lowerable_graph();
    let engine = DagCachedPlanCountingEngine::new();

    // Wide domain so SPSA steps don't immediately violate, forcing multiple IBP calls.
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    // Objective and threshold chosen so no violation is found (path-selection
    // proof, not SAT/UNSAT correctness).
    let objective = [1.0_f32, 0.0];
    let threshold = -1000.0;

    let result = try_adv_check_on_domain(
        &graph,
        &input_bounds,
        &objective,
        threshold,
        false,
        None,
        0,
        Some(&engine),
    );

    assert!(
        result.is_ok(),
        "adv_check should succeed with DAG engine: {:?}",
        result.err()
    );

    // Plan is prepared once (cached across all SPSA evaluations within the call).
    // The current implementation in graph_ibp.rs prepares a fresh plan per
    // propagate_ibp_with_engine call, so plan_preparations >= 1.
    assert!(
        engine.plan_preparations() >= 1,
        "DAG engine should have prepared at least 1 plan, got {}",
        engine.plan_preparations()
    );

    assert!(
        engine.cached_calls() > 0,
        "DAG engine should have executed cached forward at least once, got {}",
        engine.cached_calls()
    );
}

/// Prove that `try_adv_check_on_domain` does NOT call prepare_dag_model_plan
/// or dag_ibp_forward_cached when the graph contains an unsupported op (Sign).
/// The DAG lowering returns None, so the engine's DAG methods are never reached.
///
/// Regression for #4276.
#[test]
fn test_adv_check_unsupported_graph_skips_dag_plan_4276() {
    let graph = build_unsupported_graph();
    let engine = DagUnsupportedFallbackEngine::new();

    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.5, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap(),
    )
    .unwrap();

    let objective = [1.0_f32, 0.0];
    let threshold = -1000.0;

    let result = try_adv_check_on_domain(
        &graph,
        &input_bounds,
        &objective,
        threshold,
        false,
        None,
        0,
        Some(&engine),
    );

    assert!(
        result.is_ok(),
        "adv_check should succeed on unsupported graph via CPU fallback: {:?}",
        result.err()
    );

    assert_eq!(
        engine.plan_preparations(),
        0,
        "unsupported graph should never reach prepare_dag_model_plan"
    );
    assert_eq!(
        engine.cached_calls(),
        0,
        "unsupported graph should never reach dag_ibp_forward_cached"
    );
}

/// Value correctness for the DAG-lowerable fast path of
/// `GraphNetwork::propagate_concrete_point` (#cgan-eval / #4276): on a
/// widening-free DAG a point input must stay degenerate and the returned value
/// must equal the TRUE network forward — NOT a widened box (the whole bug this
/// routine exists to avoid). We check this independently of the IBP machinery by
/// hand-computing `residual = x + W2·relu(W1·x + b1)` for the residual-add graph.
#[test]
fn test_propagate_concrete_point_dag_fast_path_returns_true_forward() {
    let graph = build_dag_lowerable_graph();

    // A concrete (degenerate) point input.
    let x = [0.37_f32, -0.21_f32];
    let point = BoundedTensor::concrete(arr1(&x).into_dyn()).unwrap();

    // Hand-computed forward for build_dag_lowerable_graph():
    //   linear1: W1=[[0.8,-0.3],[0.4,0.9]], b1=[0.1,-0.05]
    //   relu; linear2: W2=[[0.6,-0.2],[-0.4,0.7]], b2=[0,0]; residual = x + linear2
    let l1 = [
        0.8 * x[0] - 0.3 * x[1] + 0.1,
        0.4 * x[0] + 0.9 * x[1] - 0.05,
    ];
    let r = [l1[0].max(0.0), l1[1].max(0.0)];
    let l2 = [0.6 * r[0] - 0.2 * r[1], -0.4 * r[0] + 0.7 * r[1]];
    let expected = [x[0] + l2[0], x[1] + l2[1]];

    // engine = None exercises the fast path's CPU box-forward delegation; the result
    // must be degenerate (lower == upper) and equal the true forward.
    let out = graph
        .propagate_concrete_point(&point, None, None)
        .expect("concrete point forward on DAG-lowerable graph should succeed");
    let lo = out.lower();
    let hi = out.upper();
    for i in 0..2 {
        let l = lo.iter().nth(i).copied().unwrap();
        let u = hi.iter().nth(i).copied().unwrap();
        assert!(
            (u - l).abs() <= 1e-6,
            "output[{i}] must be degenerate (point), got [{l}, {u}] (width {})",
            u - l
        );
        let c = f32::midpoint(l, u);
        assert!(
            (c - expected[i]).abs() <= 1e-5,
            "output[{i}] center {c} should equal the true forward {} (diff {})",
            expected[i],
            (c - expected[i]).abs()
        );
    }

    // Same via the GEMM-engine entry: a non-DAG CPU engine still delegates to the
    // box forward (no DAG ext) and must produce the identical true forward.
    let out_engine = graph
        .propagate_concrete_point(&point, Some(&NaiveCpuGemmEngine), None)
        .expect("concrete point forward with engine should succeed");
    for (i, (&a, &b)) in out
        .center()
        .iter()
        .zip(out_engine.center().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() <= 1e-6,
            "engine vs no-engine concrete point diverged at {i}: {a} vs {b}"
        );
    }
}
