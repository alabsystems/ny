// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::propagation::{
    dispatch_plain_patches_or_fallback, plain_dense_retry_is_authorized,
    plain_memory_fallback_or_error, prepare_plain_dense_boundary,
    with_plain_patches_deadline_capture, PlainPatchesDispatchOutcome,
};
use super::spec_propagation::SpecCrownRequest;
use super::GraphNetworkCrownExt;
use crate::layers::{Layer, LinearLayer};
use crate::network::{GraphNetwork, GraphNode};
use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
use crate::MulBinaryRelaxationMode;
use ndarray::{arr1, arr2, array};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

fn test_input() -> BoundedTensor {
    BoundedTensor::new(
        array![-1.0_f32, -0.5].into_dyn(),
        array![0.25_f32, 1.0].into_dyn(),
    )
    .expect("bounded tensor should construct")
}

fn single_linear_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.5_f32, -0.25]]), Some(arr1(&[0.75_f32])))
        .expect("linear layer should construct");
    graph.add_node(GraphNode::from_input("lin", Layer::Linear(linear)));
    graph.set_output("lin");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_backward_empty_graph_fast_paths_preserve_input_4205() {
    let graph = GraphNetwork::new();
    let input = test_input();

    let plain = GraphNetworkCrownExt::crown_backward_with_relaxation(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
    )
    .expect("empty graph CROWN should succeed");
    assert_eq!(plain.lower(), input.lower());
    assert_eq!(plain.upper(), input.upper());

    let with_provenance = GraphNetworkCrownExt::crown_backward_with_relaxation_and_provenance(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
    )
    .expect("empty graph provenance CROWN should succeed");
    assert_eq!(with_provenance.bounds.lower(), input.lower());
    assert_eq!(with_provenance.bounds.upper(), input.upper());
    assert_eq!(with_provenance.provenance, BoundsProvenance::Crown);

    let expired = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("system uptime exceeds 1ms"),
        ),
        Some(0),
    )
    .expect_err("expired finite authority must refuse before cloning the input");
    assert!(matches!(expired, ny_core::NyError::DeadlineExceeded(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_backward_expired_before_collection_returns_typed_deadline_4205() {
    let graph = single_linear_graph();
    let input = test_input();

    let error = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("system uptime exceeds 1ms"),
        ),
    )
    .expect_err("expired authority cannot launch a fresh no-deadline IBP pass");
    assert!(matches!(error, ny_core::NyError::DeadlineExceeded(_)));
}

#[ntest::timeout(10000)]
#[test]
fn finite_dense_relu_publishes_precollected_output_before_legacy_dispatch() {
    use crate::layers::ReLULayer;

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer::new())));
    graph.set_output("relu");
    let input = test_input();
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("precollected forward bounds");
    let expected = node_bounds.get("relu").expect("output bounds");

    let result = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        Some(Instant::now() + Duration::from_secs(30)),
        None,
        Some(&node_bounds),
        None,
    )
    .expect("finite custom Dense route must publish its collected enclosure");

    assert_eq!(
        result.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownPropagationError)
    );
    assert_eq!(result.bounds.lower(), expected.lower());
    assert_eq!(result.bounds.upper(), expected.upper());
}

/// Truncation after the stride-2 ConvTranspose step leaves an Anchored
/// relation on the preceding Conv2d node.  Its compact carrier fits this
/// budget, while collapsing the frontier to a full Dense pair does not.
#[ntest::timeout(10000)]
#[test]
fn truncated_anchored_frontier_memory_refusal_uses_exact_ibp() {
    use crate::layers::{Conv2dLayer, ConvTranspose2dLayer};
    use ndarray::{ArrayD, IxDyn};

    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "2");

        let conv = Conv2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0),
            None,
            (1, 1),
            (0, 0),
            1,
            400,
        )
        .expect("valid identity Conv2d");
        let conv_transpose = ConvTranspose2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 0.5),
            None,
            (2, 2),
            (0, 0),
            1,
            400,
        )
        .expect("valid stride-2 ConvTranspose2d");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "conv_transpose",
            Layer::ConvTranspose2d(conv_transpose),
            vec!["conv".into()],
        ));
        graph.set_output("conv_transpose");

        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 400]), -1.0),
            ArrayD::from_elem(IxDyn(&[1, 1, 400]), 1.0),
        )
        .expect("valid spatial input");
        let ibp = graph.propagate_ibp(&input).expect("IBP baseline");
        let node_bounds = graph
            .collect_node_bounds(&input)
            .expect("precollected IBP node bounds");

        let result = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
                &graph,
                &input,
                None,
                MulBinaryRelaxationMode::default(),
                None,
                Some(1),
                Some(&node_bounds),
                None,
            )
            .expect("truncated Anchored materialization must degrade to IBP");

        assert_eq!(
            result.provenance,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::MemoryBudgetExceeded)
        );
        assert_eq!(result.bounds.lower(), ibp.lower());
        assert_eq!(result.bounds.upper(), ibp.upper());
    });
}

/// A node-local deadline is terminal verifier authority for the plain
/// Graph-CROWN Patches path. The unchanged Patches carrier must not be
/// materialized as Dense and retried after the worker reports expiry.
#[ntest::timeout(10000)]
#[test]
fn test_plain_patches_deadline_is_terminal_ibp_without_dense_retry() {
    use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
    use crate::layers::Conv2dLayer;
    use ndarray::{ArrayD, IxDyn};

    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).expect("valid Conv2d kernel");
    let layer = Layer::Conv2d(
        Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 2, 2)
            .expect("valid Conv2d layer"),
    );
    let mut node_cb = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
        (1, 2, 2),
        (1, 2, 2),
    )));
    let pre_activation = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0),
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0),
    )
    .expect("valid pre-activation bounds");
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds 1ms");

    let outcome = dispatch_plain_patches_or_fallback(
        &mut node_cb,
        &layer,
        &pre_activation,
        None,
        Some(expired),
        "conv",
        layer.layer_type(),
    )
    .expect("deadline refusal is a typed Patches outcome");

    assert_eq!(
        outcome,
        PlainPatchesDispatchOutcome::IbpFallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded)
    );
    assert!(
        matches!(node_cb, CrownBounds::Patches(_)),
        "deadline expiry must not trigger the historical Patches-to-Dense retry"
    );
}

/// A live finite node authority keeps the exact stride-2 ConvTranspose route
/// native. No Patches-to-Dense retry may occur on this supported path.
#[ntest::timeout(10000)]
#[test]
fn test_plain_convtranspose_live_deadline_continues_anchored_without_dense_retry() {
    use crate::bounds::patches::{
        patches_to_dense_call_sites, reset_patches_to_dense_call_count, CrownBounds, PatchGeometry,
        PatchesLinearBounds,
    };
    use crate::layers::ConvTranspose2dLayer;
    use ndarray::{ArrayD, IxDyn};

    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0_f32])
        .expect("valid ConvTranspose2d kernel");
    let layer = Layer::ConvTranspose2d(
        ConvTranspose2dLayer::with_input_shape(kernel, None, (2, 2), (0, 0), 2, 2)
            .expect("valid stride-2 ConvTranspose2d layer"),
    );
    let mut node_cb = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
        (1, 3, 3),
        (1, 3, 3),
    )));
    let pre_activation = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0),
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0),
    )
    .expect("valid ConvTranspose2d pre-activation bounds");

    reset_patches_to_dense_call_count();
    let outcome = dispatch_plain_patches_or_fallback(
        &mut node_cb,
        &layer,
        &pre_activation,
        None,
        Some(Instant::now() + Duration::from_secs(30)),
        "conv_transpose",
        layer.layer_type(),
    )
    .expect("live finite ConvTranspose Patches dispatch must succeed");

    assert_eq!(outcome, PlainPatchesDispatchOutcome::AccumulateToInput);
    let CrownBounds::Patches(bounds) = node_cb else {
        panic!("supported finite ConvTranspose route must remain Patches");
    };
    assert!(matches!(
        bounds.lower_a.geometry,
        PatchGeometry::Anchored(_)
    ));
    assert!(matches!(
        bounds.upper_a.geometry,
        PatchGeometry::Anchored(_)
    ));
    assert!(patches_to_dense_call_sites().is_empty());
}

fn graph_cgan_6_14_30_fixture() -> (GraphNetwork, BoundedTensor, ndarray::ArrayD<f32>) {
    use crate::layers::{BatchNormLayer, ConvTranspose2dLayer, ReLULayer};
    use ndarray::{Array1, ArrayD, IxDyn};

    let input_kernel = ArrayD::from_shape_fn(IxDyn(&[1, 2, 4, 4]), |index| {
        if (index[1] + index[2] + index[3]).is_multiple_of(3) {
            -0.0625
        } else {
            0.0625
        }
    });
    let input_convt = ConvTranspose2dLayer::new_full(
        input_kernel,
        Some(Array1::from_vec(vec![0.0625, -0.0625])),
        (2, 2),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .expect("valid official 6-to-14 ConvTranspose");
    assert_eq!(input_convt.input_shape, None);
    assert_eq!(input_convt.output_size(6, 6).unwrap(), (14, 14));

    let batch_norm = BatchNormLayer::from_scale_bias(
        Array1::from_vec(vec![0.5, -0.5]).into_dyn(),
        Array1::from_vec(vec![0.125, -0.125]).into_dyn(),
    )
    .expect("valid two-channel BatchNorm");

    let output_kernel = ArrayD::from_shape_fn(IxDyn(&[2, 1, 4, 4]), |index| {
        if (index[0] + index[2] + index[3]) % 5 < 2 {
            -0.125
        } else {
            0.125
        }
    });
    let output_convt = ConvTranspose2dLayer::new_full(
        output_kernel.clone(),
        Some(Array1::from_vec(vec![0.03125])),
        (2, 2),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .expect("valid official 14-to-30 ConvTranspose");
    assert_eq!(output_convt.input_shape, None);
    assert_eq!(output_convt.output_size(14, 14).unwrap(), (30, 30));

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "input_convt",
        Layer::ConvTranspose2d(input_convt),
    ));
    graph.add_node(GraphNode::new(
        "batch_norm",
        Layer::BatchNorm(batch_norm),
        vec!["input_convt".into()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["batch_norm".into()],
    ));
    graph.add_node(GraphNode::new(
        "output_convt",
        Layer::ConvTranspose2d(output_convt),
        vec!["relu".into()],
    ));
    graph.set_output("output_convt");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 6, 6]), -0.25),
        ArrayD::from_elem(IxDyn(&[1, 6, 6]), 0.25),
    )
    .expect("valid cGAN input box");
    (graph, input, output_kernel)
}

/// Production Graph-CROWN regression for the first official cGAN spatial seam
/// whose composed relation remains below the deliberate Dense crossover:
///
///   input(6x6) -> ConvTranspose(14x14) -> BatchNorm -> ReLU
///               -> ConvTranspose(30x30)
///
/// Backward propagation must keep the identity CT, ReLU, BatchNorm, and
/// general CT relation native under one live finite authority.  The only
/// Patches materialization is the intentional terminal conversion at
/// NETWORK_INPUT.
#[ntest::timeout(30000)]
#[test]
fn finite_graph_cgan_6_14_30_chain_is_native_and_matches_legacy_crown() {
    use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};

    let (graph, input, output_kernel) = graph_cgan_6_14_30_fixture();
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("forward bounds for both CROWN runs");

    let legacy = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        None,
        None,
        Some(&node_bounds),
        None,
    )
    .expect("no-deadline Graph-CROWN baseline");
    assert_eq!(legacy.provenance, BoundsProvenance::Crown);

    reset_patches_to_dense_call_count();
    let finite = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        Some(Instant::now() + Duration::from_secs(30)),
        None,
        Some(&node_bounds),
        None,
    )
    .expect("finite Graph-CROWN cGAN chain");

    assert_eq!(finite.provenance, BoundsProvenance::Crown);
    assert_eq!(finite.bounds.lower(), legacy.bounds.lower());
    assert_eq!(finite.bounds.upper(), legacy.bounds.upper());
    assert!(finite
        .bounds
        .lower()
        .iter()
        .zip(finite.bounds.upper().iter())
        .any(|(&lower, &upper)| lower < upper));

    let dense_sites = patches_to_dense_call_sites();
    assert_eq!(
        dense_sites.len(),
        1,
        "native cGAN chain must materialize only at NETWORK_INPUT: {dense_sites:?}"
    );
    // Normalize separators before matching: `file!()` records the HOST
    // separator, so the recorded site reads
    // `...network\graph_crown\propagation.rs:67` on Windows and a POSIX-only
    // needle made this assertion unsatisfiable there — a hard red on the very
    // property it certifies, with the correct call site sitting in the message.
    assert!(
        dense_sites[0]
            .replace('\\', "/")
            .contains("network/graph_crown/propagation.rs:"),
        "the sole materialization must be the terminal NETWORK_INPUT frontier conversion, not a hidden operator fallback: {dense_sites:?}"
    );

    // Independent exact-f64 forward witness for the all-zero input.  The first
    // CT contributes only its channel biases; BatchNorm+ReLU therefore leaves
    // channel 0 at 0.15625 and channel 1 at zero.  Scatter that constant through
    // the final CT without calling any CROWN/IBP implementation.
    let mut exact_zero_output = vec![0.03125_f64; 30 * 30];
    for input_row in 0..14 {
        for input_column in 0..14 {
            for kernel_row in 0..4 {
                for kernel_column in 0..4 {
                    let output_row = input_row * 2 + kernel_row;
                    let output_column = input_column * 2 + kernel_column;
                    exact_zero_output[output_row * 30 + output_column] += 0.15625
                        * ny_core::f32_to_f64_exact(
                            output_kernel[[0, 0, kernel_row, kernel_column]],
                        );
                }
            }
        }
    }
    for (index, ((&lower, &upper), &exact)) in finite
        .bounds
        .lower()
        .iter()
        .zip(finite.bounds.upper().iter())
        .zip(exact_zero_output.iter())
        .enumerate()
    {
        let lower = ny_core::f32_to_f64_exact(lower);
        let upper = ny_core::f32_to_f64_exact(upper);
        assert!(
            lower <= exact && exact <= upper,
            "finite CROWN output {index} misses exact zero witness {exact}: [{lower}, {upper}]"
        );
    }
}

/// A cooperative node share is deliberately shorter than the outer request
/// authority. If the native ConvTranspose worker exhausts that share, the
/// coordinator may publish the already-collected output enclosure while the
/// outer authority remains live. It must not retry the carrier as Dense.
#[ntest::timeout(30000)]
#[test]
fn finite_graph_cgan_node_deadline_publishes_collected_forward_fallback() {
    use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};
    use crate::layers::convolution::conv2d::ConvTransposePatchesDeadlineFailpoint;

    let (graph, input, _) = graph_cgan_6_14_30_fixture();
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("precollected cGAN forward bounds");
    let expected = node_bounds
        .get("output_convt")
        .expect("cGAN output enclosure");

    reset_patches_to_dense_call_count();
    let outer_deadline = Instant::now() + Duration::from_secs(30);
    let _failpoint = ConvTransposePatchesDeadlineFailpoint::after_successful_polls(7);
    let fallback = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        Some(outer_deadline),
        None,
        Some(&node_bounds),
        None,
    )
    .expect("live outer authority must publish the collected fallback");

    assert!(
        Instant::now() < outer_deadline,
        "the deterministic worker failpoint must not consume outer authority"
    );
    assert_eq!(
        fallback.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded)
    );
    assert_eq!(fallback.bounds.lower(), expected.lower());
    assert_eq!(fallback.bounds.upper(), expected.upper());
    assert!(
        patches_to_dense_call_sites().is_empty(),
        "node-local expiry must not trigger terminal Patches materialization"
    );
}

/// The coordinator must hand the Patches fast path its computed node share,
/// not the later outer verification deadline.
#[ntest::timeout(10000)]
#[test]
fn test_plain_patches_receives_computed_node_deadline() {
    use crate::layers::{Conv2dLayer, ReLULayer};
    use ndarray::{ArrayD, IxDyn};

    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).expect("valid Conv2d kernel");
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 2, 2)
        .expect("valid Conv2d layer");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["conv".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0),
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0),
    )
    .expect("valid input bounds");
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("precollected node bounds");
    let outer_deadline = Instant::now() + Duration::from_secs(30);

    let (result, observed) = with_plain_patches_deadline_capture(|| {
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
            Some(outer_deadline),
            None,
            Some(&node_bounds),
            None,
        )
    });
    result.expect("plain Graph-CROWN Patches propagation should succeed");

    let first_node_deadline = observed
        .first()
        .copied()
        .flatten()
        .expect("Patches dispatch must observe a finite deadline");
    assert!(
        first_node_deadline < outer_deadline,
        "the first of two backward nodes must receive its computed share, not the outer deadline"
    );
    assert!(
        outer_deadline.saturating_duration_since(first_node_deadline) > Duration::from_secs(5),
        "the two-node fixture must make the node share observably earlier than the outer deadline"
    );
}

/// A typed IBP outcome is already policy, not an invitation to reinterpret the
/// same Patches relation through Dense. Only semantic Unsupported may retry.
#[ntest::timeout(10000)]
#[test]
fn test_plain_patches_non_deadline_ibp_outcome_keeps_patches() {
    use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
    use crate::layers::Conv2dLayer;
    use ndarray::{ArrayD, IxDyn};

    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).expect("valid Conv2d kernel");
    let layer = Layer::Conv2d(
        Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 2, 2)
            .expect("valid Conv2d layer"),
    );
    let mut node_cb = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
        (1, 2, 2),
        (1, 2, 2),
    )));
    // A 1D pre-activation makes the Conv2d Patches arm request its ordinary
    // CrownPropagationError fallback without involving a deadline.
    let pre_activation = test_input();

    let outcome = dispatch_plain_patches_or_fallback(
        &mut node_cb,
        &layer,
        &pre_activation,
        None,
        None,
        "conv",
        layer.layer_type(),
    )
    .expect("typed IBP refusal is policy, not a semantic error");

    assert_eq!(
        outcome,
        PlainPatchesDispatchOutcome::IbpFallback(CrownIbpFallbackReason::CrownPropagationError)
    );
    assert!(
        matches!(node_cb, CrownBounds::Patches(_)),
        "typed IBP fallback must not rewrite the pending Patches carrier"
    );
}

#[test]
fn plain_dense_retry_classifier_accepts_only_unsupported() {
    assert!(plain_dense_retry_is_authorized(
        &ny_core::NyError::UnsupportedOp("dense retry".into())
    ));
    assert!(plain_dense_retry_is_authorized(
        &ny_core::NyError::UnsupportedConfiguration("dense retry".into())
    ));
    assert!(!plain_dense_retry_is_authorized(
        &ny_core::NyError::InvalidSpec("terminal".into())
    ));
    assert!(!plain_dense_retry_is_authorized(
        &ny_core::NyError::NumericalInstability("terminal".into())
    ));
}

/// The graph shell must honor the materializer's full live-peak receipt, not
/// the smaller coefficient-pair estimate, and must leave the borrowed carrier
/// byte-for-byte intact when that receipt is refused.
#[test]
fn plain_dense_boundary_maps_only_anchored_full_peak_refusal_to_memory_ibp() {
    use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds};
    use ndarray::{Array1, ArrayD, IxDyn};
    use std::mem::size_of;

    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "1");

        const ROWS: usize = 128;
        const INPUTS: usize = 400;
        let geometry = PatchGeometry::anchored(
            vec![0],
            (0..ROWS)
                .map(|column| i128::try_from(column).expect("small fixture coordinate"))
                .collect(),
        )
        .expect("fixture axes are non-empty");
        let side = PatchesData {
            coeff_err: Some(Array1::from_elem(ROWS, 1.0e-6)),
            patches: Some(ArrayD::from_elem(IxDyn(&[1, 1, ROWS, 1, 1, 1]), 1.0)),
            geometry,
            identity: false,
            output_shape: (1, 1, ROWS),
            input_shape: (1, 1, INPUTS),
            unstable_idx: None,
        };
        let expected = PatchesLinearBounds {
            row_count: ROWS,
            lower_a: side.clone(),
            lower_b: Array1::from_elem(ROWS, -0.25),
            upper_a: side,
            upper_b: Array1::from_elem(ROWS, 0.5),
        };
        let mut carrier = CrownBounds::Patches(Box::new(expected.clone()));

        let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        let coefficient_pair_bytes = ROWS * INPUTS * 2 * size_of::<f32>();
        assert!(
            coefficient_pair_bytes <= budget,
            "the old pair-only estimate must admit this fixture"
        );

        let disposition = prepare_plain_dense_boundary(&mut carrier, None)
            .expect("a resource refusal is policy, not a semantic error");
        assert_eq!(
            disposition,
            Some(CrownIbpFallbackReason::MemoryBudgetExceeded),
            "the larger certified full peak must select the established memory IBP fallback"
        );

        let CrownBounds::Patches(actual) = &carrier else {
            panic!("failed admission must not publish a partial Dense carrier");
        };
        let assert_side_exact = |actual: &PatchesData, expected: &PatchesData| {
            assert_eq!(actual.coeff_err, expected.coeff_err);
            assert_eq!(actual.patches, expected.patches);
            assert_eq!(actual.geometry, expected.geometry);
            assert_eq!(actual.identity, expected.identity);
            assert_eq!(actual.output_shape, expected.output_shape);
            assert_eq!(actual.input_shape, expected.input_shape);
            assert_eq!(actual.unstable_idx, expected.unstable_idx);
        };
        assert_eq!(actual.row_count, expected.row_count);
        assert_side_exact(&actual.lower_a, &expected.lower_a);
        assert_eq!(actual.lower_b, expected.lower_b);
        assert_side_exact(&actual.upper_a, &expected.upper_a);
        assert_eq!(actual.upper_b, expected.upper_b);

        let mut malformed = expected;
        let malformed_geometry = PatchGeometry::anchored(
            vec![0, 1],
            (0..ROWS)
                .map(|column| i128::try_from(column).expect("small fixture coordinate"))
                .collect(),
        )
        .expect("non-empty malformed axes still construct");
        malformed.lower_a.geometry = malformed_geometry.clone();
        malformed.upper_a.geometry = malformed_geometry;
        let malformed_expected = malformed.clone();
        let mut malformed_carrier = CrownBounds::Patches(Box::new(malformed));
        let error = prepare_plain_dense_boundary(&mut malformed_carrier, None)
            .expect_err("semantic geometry failures must not be reclassified as memory IBP");
        assert!(matches!(error, ny_core::NyError::ShapeMismatch { .. }));
        let CrownBounds::Patches(actual) = &malformed_carrier else {
            panic!("semantic failure must preserve the original Patches carrier");
        };
        assert_side_exact(&actual.lower_a, &malformed_expected.lower_a);
        assert_eq!(actual.lower_b, malformed_expected.lower_b);
        assert_side_exact(&actual.upper_a, &malformed_expected.upper_a);
        assert_eq!(actual.upper_b, malformed_expected.upper_b);
    });
}

#[test]
fn plain_dense_boundary_deadline_is_terminal_and_atomic() {
    use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};

    let expected = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
    let mut carrier = CrownBounds::Patches(Box::new(expected.clone()));
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");

    let disposition = prepare_plain_dense_boundary(&mut carrier, Some(expired))
        .expect("deadline is a typed terminal fallback policy");
    assert_eq!(
        disposition,
        Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded)
    );
    let CrownBounds::Patches(actual) = carrier else {
        panic!("deadline-refused dense boundary must retain Patches");
    };
    assert_eq!(actual.row_count, expected.row_count);
    assert_eq!(actual.lower_a.identity, expected.lower_a.identity);
    assert_eq!(actual.upper_a.identity, expected.upper_a.identity);
    assert_eq!(actual.lower_b, expected.lower_b);
    assert_eq!(actual.upper_b, expected.upper_b);
}

#[test]
fn plain_merge_boundary_uses_collected_forward_bounds_under_finite_authority() {
    use crate::layers::ReLULayer;

    // This graph cannot run IBP because its only input name is unresolved.  A
    // successful finite fallback therefore proves the helper did not launch a
    // fresh graph pass and published only the supplied enclosure.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "poison",
        Layer::ReLU(ReLULayer::new()),
        vec!["missing".into()],
    ));
    graph.set_output("poison");
    let input = test_input();
    let forward = BoundedTensor::new(
        array![4.0_f32, 5.0].into_dyn(),
        array![6.0_f32, 7.0].into_dyn(),
    )
    .expect("valid collected forward bounds");
    let live = Some(Instant::now() + Duration::from_secs(30));
    let fallback = plain_memory_fallback_or_error(
        &graph,
        &input,
        &forward,
        live,
        ny_core::NyError::CpuMemoryExceeded {
            required_bytes: 2,
            budget_bytes: 1,
            site: "pending Anchored merge test",
        },
    )
    .expect("CPU memory refusal must degrade to sound IBP");
    assert_eq!(
        fallback.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::MemoryBudgetExceeded)
    );
    assert_eq!(fallback.bounds.lower(), forward.lower());
    assert_eq!(fallback.bounds.upper(), forward.upper());

    let historical = plain_memory_fallback_or_error(
        &GraphNetwork::new(),
        &input,
        &forward,
        None,
        ny_core::NyError::CpuMemoryExceeded {
            required_bytes: 2,
            budget_bytes: 1,
            site: "ordinary no-deadline fallback test",
        },
    )
    .expect("no-deadline fallback must retain the historical graph IBP path");
    assert_eq!(historical.bounds.lower(), input.lower());
    assert_eq!(historical.bounds.upper(), input.upper());

    let deadline_fallback = plain_memory_fallback_or_error(
        &graph,
        &input,
        &forward,
        live,
        ny_core::NyError::DeadlineExceeded("atomic deadline test".into()),
    )
    .expect("a live authority may publish the checked collected enclosure");
    assert_eq!(
        deadline_fallback.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded)
    );
    assert_eq!(deadline_fallback.bounds.lower(), forward.lower());
    assert_eq!(deadline_fallback.bounds.upper(), forward.upper());

    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    let expired_error = plain_memory_fallback_or_error(
        &graph,
        &input,
        &forward,
        Some(expired),
        ny_core::NyError::DeadlineExceeded("expired transaction".into()),
    )
    .expect_err("expired authority must not copy or recompute fallback bounds");
    assert!(matches!(
        expired_error,
        ny_core::NyError::DeadlineExceeded(_)
    ));

    let semantic = plain_memory_fallback_or_error(
        &graph,
        &input,
        &forward,
        live,
        ny_core::NyError::InvalidSpec("semantic".into()),
    )
    .expect_err("semantic errors must remain typed");
    assert!(matches!(semantic, ny_core::NyError::InvalidSpec(_)));
}

/// #dedup-root-collections Fix B: the `_with_node_bounds` variant fed the SAME
/// map the internal Step-1 collection would produce must yield bit-identical
/// output bounds and provenance to the legacy entry point — the precollected
/// path only skips the redundant collection, never changes the backward math.
/// An extra NETWORK_INPUT entry (inserted by the DAG alpha init wiring) must
/// be ignored.
#[ntest::timeout(10000)]
#[test]
fn test_crown_backward_with_precollected_node_bounds_bit_identical() {
    use crate::layers::ReLULayer;
    use crate::network::core::NETWORK_INPUT;

    let mut graph = GraphNetwork::new();
    let lin1 = LinearLayer::new(
        arr2(&[[1.0_f32, -0.5], [0.25, 0.75]]),
        Some(arr1(&[0.1_f32, -0.2])),
    )
    .expect("linear layer should construct");
    let lin2 = LinearLayer::new(arr2(&[[0.5_f32, -1.0]]), Some(arr1(&[0.3_f32])))
        .expect("linear layer should construct");
    graph.add_node(GraphNode::from_input("lin1", Layer::Linear(lin1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer::new()),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(lin2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("lin2");

    let input = test_input();

    // This small ReLU graph selects the per-node CROWN-IBP Step-1 collection;
    // precollect the identical map the legacy path would collect internally.
    assert!(graph.should_collect_per_node_crown_ibp_intermediates());
    let mut precollected = graph
        .collect_crown_ibp_bounds_dag_with_status_and_deadline(&input, None, None)
        .expect("CROWN-IBP collection should succeed")
        .bounds;

    let legacy = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        None,
        None,
    )
    .expect("legacy path should succeed");

    let with_bounds =
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
            None,
            None,
            Some(&precollected),
            None,
        )
        .expect("precollected-bounds path should succeed");

    assert_eq!(with_bounds.provenance, legacy.provenance);
    assert_eq!(with_bounds.bounds.lower(), legacy.bounds.lower());
    assert_eq!(with_bounds.bounds.upper(), legacy.bounds.upper());

    // The DAG alpha init map also carries a NETWORK_INPUT entry — must be inert.
    precollected.insert(NETWORK_INPUT.to_string(), input.clone());
    let with_input_entry =
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
            None,
            None,
            Some(&precollected),
            None,
        )
        .expect("precollected-bounds path with NETWORK_INPUT entry should succeed");
    assert_eq!(with_input_entry.provenance, legacy.provenance);
    assert_eq!(with_input_entry.bounds.lower(), legacy.bounds.lower());
    assert_eq!(with_input_entry.bounds.upper(), legacy.bounds.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_backward_spec_wrappers_match_spec_request_builder_4205() {
    let graph = single_linear_graph();
    let input = test_input();
    let spec_matrix = arr2(&[[1.0_f32]]);

    let expected = SpecCrownRequest::new(&graph, &input, &spec_matrix, None)
        .mul_binary_relaxation(MulBinaryRelaxationMode::default())
        .run()
        .expect("spec request builder should succeed");
    let actual = GraphNetworkCrownExt::crown_backward_specs_with_relaxation(
        &graph,
        &input,
        &spec_matrix,
        None,
        MulBinaryRelaxationMode::default(),
    )
    .expect("trait spec wrapper should succeed");

    assert_eq!(actual.lower(), expected.lower());
    assert_eq!(actual.upper(), expected.upper());

    let (expected_bounds, expected_linear) =
        SpecCrownRequest::new(&graph, &input, &spec_matrix, None)
            .mul_binary_relaxation(MulBinaryRelaxationMode::default())
            .run_with_linear()
            .expect("spec request builder should return linear bounds");
    let (actual_bounds, actual_linear) =
        GraphNetworkCrownExt::crown_backward_specs_linear_with_relaxation(
            &graph,
            &input,
            &spec_matrix,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("trait spec-linear wrapper should succeed");

    assert_eq!(actual_bounds.lower(), expected_bounds.lower());
    assert_eq!(actual_bounds.upper(), expected_bounds.upper());

    let expected_linear = expected_linear.expect("builder should capture linear bounds");
    let actual_linear = actual_linear.expect("wrapper should capture linear bounds");
    assert_eq!(actual_linear.lower_a(), expected_linear.lower_a());
    assert_eq!(actual_linear.lower_b(), expected_linear.lower_b());
    assert_eq!(actual_linear.upper_a(), expected_linear.upper_a());
    assert_eq!(actual_linear.upper_b(), expected_linear.upper_b());
}

// #margin-subset-alpha: root CROWN backward margin-subset seeding tests.
mod margin_subset_root_backward {
    use super::GraphNetworkCrownExt;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::{GraphNetwork, GraphNode};
    use crate::output_margin_seed::MarginOutputSeedGuard;
    use crate::MulBinaryRelaxationMode;
    use ndarray::{arr1, arr2, Array2};
    use ny_tensor::BoundedTensor;

    /// input(2) -> Linear(2->3) "pre" -> ReLU "act" -> Linear(3->600) "out".
    /// 600 outputs put the OUTPUT node at/above the margin-subset engagement
    /// width; the unstable ReLUs make CROWN strictly tighter than IBP.
    fn wide_output_net() -> (GraphNetwork, BoundedTensor) {
        let pre = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.25, 0.75], [-0.6, 0.4]]),
            Some(arr1(&[0.05_f32, -0.1, 0.02])),
        )
        .expect("pre");
        let weights = Array2::from_shape_fn((600, 3), |(i, j)| {
            let v = ((i * 7 + j * 13) % 11) as f32 / 11.0 - 0.5;
            if v == 0.0 {
                0.3
            } else {
                v
            }
        });
        let out = LinearLayer::new(weights, None).expect("out");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
        graph.add_node(GraphNode::new(
            "act",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(out),
            vec!["act".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("input");
        (graph, input)
    }

    /// With published indices the root CROWN backward's referenced rows are
    /// BIT-IDENTICAL to the full-width backward; every unreferenced row keeps
    /// a sound (equal-or-looser) enclosure of the full-width row.
    #[ntest::timeout(30000)]
    #[test]
    fn root_backward_scatters_published_margin_rows() {
        let (graph, input) = wide_output_net();

        // Full-width reference (no publication on this thread).
        let full = GraphNetworkCrownExt::crown_backward_with_relaxation(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("full-width root backward");

        let _guard = MarginOutputSeedGuard::publish(vec![200, 5]);
        let subset = GraphNetworkCrownExt::crown_backward_with_relaxation(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("margin-subset root backward");

        assert_eq!(subset.shape(), full.shape());
        for i in 0..600 {
            if i == 5 || i == 200 {
                assert_eq!(
                    subset.lower()[[i]],
                    full.lower()[[i]],
                    "referenced lower row {i} must match the full-width backward"
                );
                assert_eq!(
                    subset.upper()[[i]],
                    full.upper()[[i]],
                    "referenced upper row {i} must match the full-width backward"
                );
            } else {
                // Unreferenced rows keep the node's sound forward enclosure —
                // never tighter than the full-width CROWN row.
                assert!(
                    subset.lower()[[i]] <= full.lower()[[i]],
                    "unreferenced lower row {i} must enclose the full-width row"
                );
                assert!(
                    subset.upper()[[i]] >= full.upper()[[i]],
                    "unreferenced upper row {i} must enclose the full-width row"
                );
            }
        }
    }

    /// Fail-closed: without a publication the behavior is full-width even at
    /// engagement width (no accidental engagement from a stale thread-local).
    #[ntest::timeout(30000)]
    #[test]
    fn root_backward_unpublished_is_full_width() {
        let (graph, input) = wide_output_net();
        let a = GraphNetworkCrownExt::crown_backward_with_relaxation(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("run a");
        let b = GraphNetworkCrownExt::crown_backward_with_relaxation(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("run b");
        assert_eq!(a.lower(), b.lower());
        assert_eq!(a.upper(), b.upper());
    }
}
