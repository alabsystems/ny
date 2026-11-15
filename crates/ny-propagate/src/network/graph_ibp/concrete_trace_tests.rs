// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::layers::{CausalSoftmaxLayer, Layer, LinearLayer};
use crate::network::{GraphPointArm, LinearPointArm};
use crate::{GraphNetwork, GraphNode};
use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, NaiveCpuGemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};
use std::time::{Duration, Instant};

fn graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "probability",
        Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
    ));
    graph.add_node(GraphNode::new(
        "projected",
        Layer::Linear(LinearLayer::from_flat(vec![2.0, -1.0], 1, 2, Some(vec![0.25])).unwrap()),
        vec!["probability".into()],
    ));
    graph.set_output("projected");
    graph
}

#[test]
fn graph_point_trace_records_completed_causal_and_actual_linear_arms() {
    let graph = graph();
    let input = BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 9.0, 0.0, 0.0]).unwrap(),
    )
    .unwrap();
    let admission = graph.point_trace_admission(None, None).unwrap();
    let plain = graph.propagate_concrete_point(&input, None, None).unwrap();
    let (traced, trace) = graph
        .propagate_concrete_point_with_trace(&input, None, None)
        .unwrap();
    assert_eq!(traced.lower(), plain.lower());
    assert_eq!(trace.steps().len(), 2);
    assert_eq!(trace.steps()[0].node, "probability");
    assert_eq!(trace.steps()[0].arm, GraphPointArm::CausalSoftmaxF64Libm);
    assert!(
        matches!(trace.steps()[1].arm,GraphPointArm::Linear {receipt} if receipt.arm==LinearPointArm::CpuFaerF32)
    );
    assert!(trace.retained_bytes().unwrap() <= admission.persistent_bytes);
    let (finite, trace) = graph
        .propagate_concrete_point_with_trace(
            &input,
            None,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .unwrap();
    assert_eq!(finite.lower(), plain.lower());
    assert!(
        matches!(trace.steps()[1].arm,GraphPointArm::Linear {receipt} if receipt.arm==LinearPointArm::ScalarF64)
    );
    assert!(trace.qualification.contains("unqualified"));
}

#[test]
fn graph_point_trace_admission_and_deadline_fail_before_any_receipt_is_returned() {
    let graph = graph();
    let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[2, 2]))).unwrap();
    assert!(matches!(
        graph.propagate_concrete_point_with_trace(&input, None, Some(Instant::now())),
        Err(NyError::DeadlineExceeded(_))
    ));
    let mut oversized = GraphNetwork::new();
    let name = "z".repeat(65_537);
    oversized.add_node(GraphNode::from_input(
        &name,
        Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
    ));
    oversized.set_output(&name);
    assert!(oversized.point_trace_admission(None, None).is_err());
    // Opting into receipts doesn't impose their metadata cap on the original
    // no-trace concrete route.
    assert!(oversized
        .propagate_concrete_point(&input, None, None)
        .is_ok());
}

#[test]
fn graph_point_trace_records_original_gemm_arm_without_resident_interval_authority() {
    use super::concrete_linear_tests::ToyGpu;
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear",
        Layer::Linear(LinearLayer::from_flat(vec![1.0 / 512.0; 512], 1, 512, None).unwrap()),
    ));
    graph.set_output("linear");
    let input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[2, 512]), 1.0)).unwrap();
    for mode in 0..3 {
        let engine = ToyGpu::new(mode);
        let (point, trace) = graph
            .propagate_concrete_point_with_trace(&input, Some(&engine), None)
            .unwrap();
        assert!(point.lower().iter().all(|&v| v == 1.0));
        assert_eq!(trace.steps().len(), 1);
        assert_eq!(trace.steps()[0].node, "linear");
        assert!(
            matches!(trace.steps()[0].arm,GraphPointArm::Linear {receipt} if receipt.arm==LinearPointArm::EngineF32 && receipt.backend=="unspecified-custom-gemm-engine")
        );
    }
}

struct ChangingIdentity {
    calls: AtomicUsize,
    decline: bool,
}
impl GemmEngine for ChangingIdentity {
    fn backend_provenance(&self) -> &'static str {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return "initially-short-test-backend";
        }
        static LONG: OnceLock<String> = OnceLock::new();
        LONG.get_or_init(|| "b".repeat(4097)).as_str()
    }
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if self.decline {
            Err(NyError::UnsupportedOp(
                "injected identity-changing refusal".into(),
            ))
        } else {
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }
    }
}

#[test]
fn graph_point_trace_rechecks_successful_and_declined_backend_metadata() {
    let graph = graph();
    let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[2, 2]))).unwrap();
    for decline in [false, true] {
        let engine = ChangingIdentity {
            calls: AtomicUsize::new(0),
            decline,
        };
        let result = graph.propagate_concrete_point_with_trace(&input, Some(&engine), None);
        assert!(
            matches!(result, Err(NyError::UnsupportedOp(_))),
            "an engine cannot enlarge its completed/declined receipt beyond preflight"
        );
        assert!(engine.calls.load(Ordering::SeqCst) > 1);
    }
    // Price the full allowed provenance size rather than trusting a trait
    // implementation's first returned name to remain stable.
    let engine = ChangingIdentity {
        calls: AtomicUsize::new(0),
        decline: false,
    };
    let configured = graph.point_trace_admission(Some(&engine), None).unwrap();
    let cpu = graph.point_trace_admission(None, None).unwrap();
    assert!(configured.work - cpu.work >= graph.node_names().len() as u64 * 8192 * 8);
}
