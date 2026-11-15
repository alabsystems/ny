// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_core::{
    GemmEngine, GpuDagIbpForwardExt, GpuDagIbpModelPlan, GpuDagIbpPlanDesc, GpuIbpResult, NyError,
    Result,
};
use ny_tensor::BoundedTensor;

use crate::layers::softmax::GeluApproximation;
use crate::layers::{GELULayer, Layer, LinearLayer};
use crate::{GraphNetwork, GraphNode};

fn binade_graph(nonlinear: bool) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "unit_sum",
        Layer::Linear(LinearLayer::from_flat(vec![1.0 / 512.0; 512], 1, 512, None).unwrap()),
    ));
    if nonlinear {
        graph.add_node(GraphNode::new(
            "gelu",
            Layer::GELU(GELULayer::new(GeluApproximation::Erf)),
            vec!["unit_sum".into()],
        ));
        graph.set_output("gelu");
    } else {
        graph.set_output("unit_sum");
    }
    graph
}

#[test]
fn graph_point_lowerable_cpu_linear_does_not_center_certificate_padding() {
    let graph = binade_graph(false);
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[2, 512]), 1.0)).unwrap();
    let point = graph.propagate_concrete_point(&input, None, None).unwrap();
    assert_eq!(point.shape(), &[2, 1]);
    assert!(point.lower().iter().all(|&v| v == 1.0));
    assert_eq!(point.lower(), point.upper());
    let interval = graph.propagate_ibp_sound(&input).unwrap();
    assert!(interval.lower().iter().all(|&v| v < 1.0));
    assert!(interval.upper().iter().all(|&v| v > 1.0));
    assert!(interval.center().iter().any(|&v| v != 1.0));
}

#[test]
fn graph_point_nonlinear_linear_dispatch_uses_the_unpadded_value() {
    let graph = binade_graph(true);
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[1, 512]), 1.0)).unwrap();
    let expected = f64::midpoint(1.0, libm::erf(1.0 / 2.0_f64.sqrt())) as f32;
    for deadline in [None, Some(Instant::now() + Duration::from_secs(30))] {
        let point = graph
            .propagate_concrete_point(&input, None, deadline)
            .unwrap();
        assert_eq!(point.shape(), &[1, 1]);
        assert_eq!(point.lower()[[0, 0]], expected);
        assert_eq!(point.lower(), point.upper());
    }
    let enclosure = graph.propagate_ibp_sound(&input).unwrap();
    assert!(enclosure.lower()[[0, 0]] <= expected && enclosure.upper()[[0, 0]] >= expected);
}

#[test]
fn graph_point_linear_preserves_shape_and_deadline_refusals() {
    let graph = binade_graph(false);
    let wrong = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[1, 511]), 1.0)).unwrap();
    assert!(matches!(
        graph.propagate_concrete_point(&wrong, None, None),
        Err(NyError::ShapeMismatch { .. })
    ));
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[1, 512]), 1.0)).unwrap();
    assert!(matches!(
        graph.propagate_concrete_point(&input, None, Some(Instant::now())),
        Err(NyError::DeadlineExceeded(_))
    ));
}

pub(super) struct ToyGpu {
    mode: usize,
    preparations: AtomicUsize,
    gemms: AtomicUsize,
    executions: Arc<AtomicUsize>,
}

impl ToyGpu {
    pub(super) fn new(mode: usize) -> Self {
        Self {
            mode,
            preparations: AtomicUsize::new(0),
            gemms: AtomicUsize::new(0),
            executions: Arc::new(AtomicUsize::new(0)),
        }
    }
}

struct ToyPlan {
    fail: bool,
    executions: Arc<AtomicUsize>,
}

impl GpuDagIbpModelPlan for ToyPlan {
    fn dag_ibp_forward_cached(
        &self,
        lower: &[f32],
        upper: &[f32],
        shape: &[usize],
    ) -> Result<GpuIbpResult> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err(NyError::UnsupportedOp(
                "injected resident execution refusal".into(),
            ));
        }
        assert_eq!(shape, [2, 512]);
        assert!(lower.iter().chain(upper).all(|&v| v == 1.0));
        Ok(GpuIbpResult {
            // A valid enclosure of the exact unit sum, with a different
            // midpoint. Successful interval execution is not a point kernel.
            lower_bounds: vec![0.75; 2],
            upper_bounds: vec![1.5; 2],
            output_shape: vec![2, 1],
        })
    }
}

impl GpuDagIbpForwardExt for ToyGpu {
    fn prepare_dag_model_plan(
        &self,
        plan: &GpuDagIbpPlanDesc,
    ) -> Result<Option<Box<dyn GpuDagIbpModelPlan>>> {
        self.preparations.fetch_add(1, Ordering::SeqCst);
        assert_eq!(plan.ops.len(), 1);
        if self.mode == 1 {
            return Ok(None);
        }
        Ok(Some(Box::new(ToyPlan {
            fail: self.mode == 2,
            executions: self.executions.clone(),
        })))
    }
}

impl GemmEngine for ToyGpu {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.gemms.fetch_add(1, Ordering::SeqCst);
        let mut values = vec![0.0; m * n];
        for row in 0..m {
            for out in 0..n {
                for column in 0..k {
                    values[row * n + out] += a[row * k + column] * b[column * n + out];
                }
            }
        }
        Ok(values)
    }
    fn as_gpu_dag_ibp_forward_ext(&self) -> Option<&dyn GpuDagIbpForwardExt> {
        Some(self)
    }
}

#[test]
fn graph_point_declines_resident_interval_midpoints_and_uses_original_gemm() {
    let graph = binade_graph(false);
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[2, 512]), 1.0)).unwrap();
    for mode in 0..3 {
        let engine = ToyGpu {
            mode,
            preparations: AtomicUsize::new(0),
            gemms: AtomicUsize::new(0),
            executions: Arc::new(AtomicUsize::new(0)),
        };
        let point = graph
            .propagate_concrete_point(&input, Some(&engine), None)
            .unwrap();
        assert!(point.lower().iter().all(|&v| v == 1.0));
        assert_eq!(point.lower(), point.upper());
        assert_eq!(engine.preparations.load(Ordering::SeqCst), 0);
        assert_eq!(engine.executions.load(Ordering::SeqCst), 0);
        assert_eq!(engine.gemms.load(Ordering::SeqCst), 1);
    }
}
