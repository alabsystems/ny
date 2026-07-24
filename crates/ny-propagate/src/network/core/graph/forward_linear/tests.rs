// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{
    ConcatLayer, Conv1dLayer, DivLayer, GatherLayer, LinearLayer, MulBinaryLayer, PowConstantLayer,
    ReLULayer, ReduceSumLayer, SigmoidLayer, SliceLayer, SubLayer, TransposeLayer,
};
use crate::network::GraphNode;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

fn add_affine_relu_prefix(graph: &mut GraphNetwork) {
    let linear = LinearLayer::new(
        arr2(&[[1.0_f32, -0.75], [-0.5_f32, 1.5], [0.8_f32, 0.6]]),
        Some(arr1(&[0.1_f32, -0.2, 0.05])),
    )
    .expect("fixture linear should be valid");
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    // Second affine layer: forward-linear preserves input correlations while IBP
    // loses them at the linear→linear2 boundary.
    let linear2 = LinearLayer::new(
        arr2(&[
            [0.6_f32, -0.3, 0.5],
            [0.4_f32, 0.7, -0.2],
            [-0.5_f32, 0.1, 0.9],
        ]),
        Some(arr1(&[-0.1_f32, 0.15, 0.0])),
    )
    .expect("fixture linear2 should be valid");
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["linear".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
}

fn build_forward_linear_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    add_affine_relu_prefix(&mut graph);

    let gather = GatherLayer::new(
        0,
        Some(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![2_i64, 0_i64, 1_i64])
                .expect("fixture gather indices should shape"),
        ),
        vec![3],
    );
    graph.add_node(GraphNode::new(
        "gather",
        Layer::Gather(gather),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "slice_head",
        Layer::Slice(SliceLayer::new(0, 0, 2)),
        vec!["gather".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "slice_tail",
        Layer::Slice(SliceLayer::new(0, 1, 2)),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["slice_head".to_string(), "slice_tail".to_string()],
    ));

    let output = LinearLayer::new(
        arr2(&[[0.5_f32, -1.25, 1.0], [-0.3_f32, 0.8, 0.4]]),
        Some(arr1(&[0.2_f32, -0.1])),
    )
    .expect("fixture output linear should be valid");
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(output),
        vec!["concat".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.25]).into_dyn(),
        arr1(&[1.5_f32, 0.75]).into_dyn(),
    )
    .expect("fixture input bounds should be valid");

    (graph, input)
}

fn build_conv_reduce_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let conv = Conv1dLayer::with_input_length(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.75_f32, -0.25])
            .expect("conv fixture kernel should shape"),
        Some(arr1(&[0.1_f32])),
        1,
        0,
        4,
    )
    .expect("conv fixture should be valid");
    graph.add_node(GraphNode::from_input("conv", Layer::Conv1d(conv)));
    graph.add_node(GraphNode::new(
        "pow",
        Layer::PowConstant(PowConstantLayer::square()),
        vec!["conv".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "energy",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], false)),
        vec!["pow".to_string()],
    ));
    graph.set_output("energy");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0_f32, 0.1, 0.2, 0.3])
            .expect("conv fixture lower should shape"),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.5_f32, 0.6, 0.7, 0.8])
            .expect("conv fixture upper should shape"),
    )
    .expect("conv fixture input bounds should be valid");

    (graph, input)
}

fn build_div_sub_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "pow",
        Layer::PowConstant(PowConstantLayer::new(3.0)),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "energy",
        Layer::ReduceSum(ReduceSumLayer::new(vec![0], false)),
        vec!["pow".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "norm",
        Layer::Div(DivLayer),
        vec!["pow".to_string(), "energy".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "diff",
        Layer::Sub(SubLayer),
        vec!["norm".to_string(), "relu".to_string()],
    ));
    graph.set_output("diff");

    let input = BoundedTensor::new(
        arr1(&[0.5_f32, 0.8]).into_dyn(),
        arr1(&[1.0_f32, 1.2]).into_dyn(),
    )
    .expect("div/sub fixture input bounds should be valid");

    (graph, input)
}

fn build_mul_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let left = LinearLayer::new(
        arr2(&[[1.2_f32, -0.4], [0.3_f32, 0.9]]),
        Some(arr1(&[0.1_f32, -0.2])),
    )
    .expect("mul fixture left branch should be valid");
    let right_pre = LinearLayer::new(
        arr2(&[[0.5_f32, 1.1], [-0.7_f32, 0.8]]),
        Some(arr1(&[-0.1_f32, 0.2])),
    )
    .expect("mul fixture right branch should be valid");

    graph.add_node(GraphNode::from_input("left", Layer::Linear(left)));
    graph.add_node(GraphNode::from_input("right_pre", Layer::Linear(right_pre)));
    graph.add_node(GraphNode::new(
        "right",
        Layer::ReLU(ReLULayer),
        vec!["right_pre".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        vec!["left".to_string(), "right".to_string()],
    ));
    graph.set_output("mul");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, 0.1]).into_dyn(),
        arr1(&[1.0_f32, 0.9]).into_dyn(),
    )
    .expect("mul fixture input bounds should be valid");

    (graph, input)
}

fn build_mul_sigmoid_fixture() -> (GraphNetwork, BoundedTensor) {
    let (mut graph, input) = build_mul_fixture();
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["mul".to_string()],
    ));
    graph.set_output("sigmoid");
    (graph, input)
}

fn point_bounds(values: [f32; 2]) -> BoundedTensor {
    let point = arr1(&values).into_dyn();
    BoundedTensor::new(point.clone(), point).expect("point bounds should be valid")
}

fn tensor_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(lower, upper)| upper - lower)
        .sum()
}

#[test]
fn test_collect_forward_linear_bounds_dag_contains_corner_outputs() {
    let (graph, input) = build_forward_linear_fixture();
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear collection should succeed on fixture");

    for point in [
        point_bounds([-1.0, -0.25]),
        point_bounds([-1.0, 0.75]),
        point_bounds([1.5, -0.25]),
        point_bounds([1.5, 0.75]),
    ] {
        let exact = graph
            .collect_node_bounds_with_engine(&point, None)
            .expect("point evaluation should succeed");

        for node_name in ["gather", "slice_head", "concat", "out"] {
            let exact_bounds = exact
                .get(node_name)
                .expect("exact point map should include all fixture nodes");
            let forward_bounds = forward
                .get(node_name)
                .expect("forward-linear map should include all fixture nodes");

            for ((lower, upper), value) in forward_bounds
                .lower()
                .iter()
                .zip(forward_bounds.upper().iter())
                .zip(exact_bounds.lower().iter())
            {
                assert!(
                    lower - 1e-5 <= *value && *value <= upper + 1e-5,
                    "forward-linear bounds for node '{node_name}' should contain exact corner output: \
                     lower={lower}, value={value}, upper={upper}"
                );
            }
        }
    }
}

#[test]
fn test_collect_forward_linear_bounds_dag_tightens_some_fixture_node() {
    let (graph, input) = build_forward_linear_fixture();
    let ibp = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("IBP node collection should succeed");
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear collection should succeed");

    let tightened = ["gather", "slice_head", "concat", "out"]
        .into_iter()
        .any(|node_name| {
            let ibp_width = tensor_width(&ibp[node_name]);
            let forward_width = tensor_width(&forward[node_name]);
            forward_width + 1e-5 < ibp_width
        });

    assert!(
        tightened,
        "forward-linear collection should tighten at least one fixture node beyond plain IBP"
    );
}

#[test]
fn test_collect_forward_linear_bounds_dag_supports_conv1d_pow_reduce_sum_4354() {
    let (graph, input) = build_conv_reduce_fixture();
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear collection should support conv1d/pow/reduce-sum fixture");

    let lower = [0.0_f32, 0.1, 0.2, 0.3];
    let upper = [0.5_f32, 0.6, 0.7, 0.8];

    for mask in 0..(1 << lower.len()) {
        let values = (0..lower.len())
            .map(|index| {
                if (mask & (1 << index)) != 0 {
                    upper[index]
                } else {
                    lower[index]
                }
            })
            .collect::<Vec<_>>();
        let point_array =
            ArrayD::from_shape_vec(IxDyn(&[1, 4]), values).expect("corner point should shape");
        let point = BoundedTensor::new(point_array.clone(), point_array)
            .expect("corner point bounds should be valid");
        let exact = graph
            .collect_node_bounds_with_engine(&point, None)
            .expect("point evaluation should succeed");

        for node_name in ["conv", "pow", "energy"] {
            let exact_bounds = exact
                .get(node_name)
                .expect("exact point map should include all conv fixture nodes");
            let forward_bounds = forward
                .get(node_name)
                .expect("forward-linear map should include all conv fixture nodes");

            for ((lower, upper), value) in forward_bounds
                .lower()
                .iter()
                .zip(forward_bounds.upper().iter())
                .zip(exact_bounds.lower().iter())
            {
                assert!(
                    lower - 1e-5 <= *value && *value <= upper + 1e-5,
                    "forward-linear bounds for node '{node_name}' should contain exact corner output: \
                     lower={lower}, value={value}, upper={upper}"
                );
            }
        }
    }
}

#[test]
fn test_collect_forward_linear_bounds_dag_supports_div_and_sub_4354() {
    let (graph, input) = build_div_sub_fixture();
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear collection should support div/sub fixture");

    let lower = [0.5_f32, 0.8];
    let upper = [1.0_f32, 1.2];

    for mask in 0..(1 << lower.len()) {
        let values = (0..lower.len())
            .map(|index| {
                if (mask & (1 << index)) != 0 {
                    upper[index]
                } else {
                    lower[index]
                }
            })
            .collect::<Vec<_>>();
        let point_array = arr1(&values).into_dyn();
        let point =
            BoundedTensor::new(point_array.clone(), point_array).expect("point bounds should work");
        let exact = graph
            .collect_node_bounds_with_engine(&point, None)
            .expect("point evaluation should succeed");

        for node_name in ["pow", "energy", "norm", "diff"] {
            let exact_bounds = exact
                .get(node_name)
                .expect("exact point map should include all div/sub fixture nodes");
            let forward_bounds = forward
                .get(node_name)
                .expect("forward-linear map should include all div/sub fixture nodes");

            for ((lower, upper), value) in forward_bounds
                .lower()
                .iter()
                .zip(forward_bounds.upper().iter())
                .zip(exact_bounds.lower().iter())
            {
                assert!(
                    lower - 1e-5 <= *value && *value <= upper + 1e-5,
                    "forward-linear bounds for node '{node_name}' should contain exact corner output: \
                     lower={lower}, value={value}, upper={upper}"
                );
            }
        }
    }
}

#[test]
fn test_collect_forward_linear_bounds_dag_supports_mulbinary_4354() {
    let (graph, input) = build_mul_fixture();
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear collection should support MulBinary fixture");

    for point in [
        point_bounds([-0.5, 0.1]),
        point_bounds([-0.5, 0.9]),
        point_bounds([1.0, 0.1]),
        point_bounds([1.0, 0.9]),
    ] {
        let exact = graph
            .collect_node_bounds_with_engine(&point, None)
            .expect("point evaluation should succeed");

        for node_name in ["left", "right", "mul"] {
            let exact_bounds = exact
                .get(node_name)
                .expect("exact point map should include all mul fixture nodes");
            let forward_bounds = forward
                .get(node_name)
                .expect("forward-linear map should include all mul fixture nodes");

            for ((lower, upper), value) in forward_bounds
                .lower()
                .iter()
                .zip(forward_bounds.upper().iter())
                .zip(exact_bounds.lower().iter())
            {
                assert!(
                    lower - 1e-5 <= *value && *value <= upper + 1e-5,
                    "forward-linear bounds for node '{node_name}' should contain exact corner output: \
                     lower={lower}, value={value}, upper={upper}"
                );
            }
        }
    }
}

#[test]
fn test_collect_forward_linear_bounds_dag_supports_mulbinary_sigmoid_4354() {
    let (graph, input) = build_mul_sigmoid_fixture();
    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear collection should support MulBinary->Sigmoid fixture");

    for point in [
        point_bounds([-0.5, 0.1]),
        point_bounds([-0.5, 0.9]),
        point_bounds([1.0, 0.1]),
        point_bounds([1.0, 0.9]),
    ] {
        let exact = graph
            .collect_node_bounds_with_engine(&point, None)
            .expect("point evaluation should succeed");

        for node_name in ["mul", "sigmoid"] {
            let exact_bounds = exact
                .get(node_name)
                .expect("exact point map should include all MulBinary->Sigmoid nodes");
            let forward_bounds = forward
                .get(node_name)
                .expect("forward-linear map should include all MulBinary->Sigmoid nodes");

            for ((lower, upper), value) in forward_bounds
                .lower()
                .iter()
                .zip(forward_bounds.upper().iter())
                .zip(exact_bounds.lower().iter())
            {
                assert!(
                    lower - 1e-5 <= *value && *value <= upper + 1e-5,
                    "forward-linear bounds for node '{node_name}' should contain exact corner output: \
                     lower={lower}, value={value}, upper={upper}"
                );
            }
        }
    }
}

#[test]
fn test_collect_forward_linear_bounds_dag_rejects_unsupported_nodes() {
    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.0_f32, -1.0], [0.5_f32, 0.25]]), None)
        .expect("unsupported fixture linear should be valid");
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "softmax",
        Layer::Softmax(crate::layers::SoftmaxLayer::new(0)),
        vec!["linear".to_string()],
    ));
    graph.set_output("softmax");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .expect("unsupported fixture input should be valid");

    let error = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect_err("unsupported forward-linear node should fail closed");
    let message = error.to_string();
    assert!(
        message.contains("softmax"),
        "unsupported-node error should name the offending node: {message}"
    );
    assert!(
        message.contains("Softmax"),
        "unsupported-node error should name the offending operator: {message}"
    );
}

#[test]
fn test_collect_forward_linear_bounds_dag_seeds_transpose_input_shape() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "transpose",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
    ));
    graph.set_output("transpose");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0_f32, 0.0, 0.5, -0.25])
            .expect("transpose fixture lower should shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.25_f32, 1.0, 1.5, 0.75])
            .expect("transpose fixture upper should shape"),
    )
    .expect("transpose fixture input should be valid");

    let forward = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear transpose collection should succeed");
    let transposed = forward
        .get("transpose")
        .expect("forward-linear map should include transpose node");

    assert_eq!(
        transposed.shape(),
        &[2, 2],
        "transpose output should preserve the expected rank and extents"
    );

    let point = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5_f32, 0.2, 1.0, 0.5])
            .expect("transpose point lower should shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5_f32, 0.2, 1.0, 0.5])
            .expect("transpose point upper should shape"),
    )
    .expect("transpose point bounds should be valid");
    let exact = graph
        .collect_node_bounds_with_engine(&point, None)
        .expect("point evaluation should succeed");
    let exact_bounds = exact
        .get("transpose")
        .expect("exact point map should include transpose node");

    for ((lower, upper), value) in transposed
        .lower()
        .iter()
        .zip(transposed.upper().iter())
        .zip(exact_bounds.lower().iter())
    {
        assert!(
            lower - 1e-5 <= *value && *value <= upper + 1e-5,
            "forward-linear transpose bounds should contain exact point output: \
             lower={lower}, value={value}, upper={upper}"
        );
    }
}

/// Cache contract (#vnncomp-image-forward-linear): same input hits (same Arc),
/// different input misses (recompute), semantic mutation invalidates, and a
/// clone starts cold — a stale map across mutation/clone would be a soundness
/// hazard.
#[test]
fn forward_linear_cache_hit_miss_invalidate_clone() {
    // Serialized + gate pinned: the cache key is salted with the dark
    // ConvTranspose surface gate, so an unsynchronized mid-test env flip (a
    // concurrent gate-mutating test) would break the Arc-identity assertions.
    crate::tests::with_serialized_env_vars(
        &[("NY_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "0")],
        forward_linear_cache_hit_miss_invalidate_clone_body,
    );
}

fn forward_linear_cache_hit_miss_invalidate_clone_body() {
    let (graph, input) = build_forward_linear_fixture();

    let first = graph
        .collect_forward_linear_bounds_dag_cached(&input, None, None)
        .expect("first collection");
    let second = graph
        .collect_forward_linear_bounds_dag_cached(&input, None, None)
        .expect("second collection");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "identical input must hit the cache (same Arc)"
    );

    // A different input box must miss and recompute.
    let shifted = BoundedTensor::new(
        input.lower().mapv(|v| v - 0.25).into_dyn(),
        input.upper().mapv(|v| v + 0.25).into_dyn(),
    )
    .expect("shifted box");
    let third = graph
        .collect_forward_linear_bounds_dag_cached(&shifted, None, None)
        .expect("shifted collection");
    assert!(
        !std::sync::Arc::ptr_eq(&second, &third),
        "different input must recompute"
    );

    // Semantic mutation invalidates.
    let mut mutated = graph.clone();
    let cold = mutated
        .collect_forward_linear_bounds_dag_cached(&input, None, None)
        .expect("clone collection");
    assert!(
        !std::sync::Arc::ptr_eq(&first, &cold),
        "a clone must start with a cold cache"
    );
    mutated.set_use_patches_mode(false);
    let after_mutation = mutated
        .collect_forward_linear_bounds_dag_cached(&input, None, None)
        .expect("post-mutation collection");
    assert!(
        !std::sync::Arc::ptr_eq(&cold, &after_mutation),
        "a semantic mutation must invalidate the cache"
    );

    // Cached and uncached maps agree exactly.
    let uncached = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("uncached collection");
    assert_eq!(first.len(), uncached.len());
    for (name, bounds) in uncached {
        let cached_bounds = first.get(&name).expect("node present in cached map");
        assert_eq!(cached_bounds.lower(), bounds.lower(), "{name} lower");
        assert_eq!(cached_bounds.upper(), bounds.upper(), "{name} upper");
    }
}

#[test]
fn forward_linear_cold_build_admission_preserves_cached_hits() {
    // Serialized + gate pinned: see forward_linear_cache_hit_miss_invalidate_clone.
    crate::tests::with_serialized_env_vars(
        &[("NY_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "0")],
        forward_linear_cold_build_admission_preserves_cached_hits_body,
    );
}

fn forward_linear_cold_build_admission_preserves_cached_hits_body() {
    let now = Instant::now();
    assert!(!forward_linear_cold_build_admitted_at(
        Some(now + std::time::Duration::from_secs(10)),
        now,
    ));
    assert!(forward_linear_cold_build_admitted_at(
        Some(now + FORWARD_LINEAR_COLD_BUILD_MIN_HEADROOM),
        now,
    ));
    assert!(forward_linear_cold_build_admitted_at(None, now));

    let (graph, input) = build_forward_linear_fixture();
    let short = graph
        .collect_forward_linear_bounds_dag_cached(
            &input,
            None,
            Some(Instant::now() + std::time::Duration::from_secs(10)),
        )
        .expect_err("a cold cache must refuse work that cannot fit the deadline");
    assert!(matches!(short, NyError::DeadlineExceeded(_)));

    let warmed = graph
        .collect_forward_linear_bounds_dag_cached(&input, None, None)
        .expect("unbounded warmup");
    let expired = Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .unwrap();
    let cached = graph
        .collect_forward_linear_bounds_dag_cached(&input, None, Some(expired))
        .expect("an existing cache hit is cheap and remains admissible");
    assert!(
        std::sync::Arc::ptr_eq(&warmed, &cached),
        "admission must never hide an existing cache entry"
    );
}

/// #w5-bab-throughput: `run_with_linear` callers must receive the input
/// `LinearBounds` even on conv DAGs where the bounds-only root candidates
/// (forward-linear C-margin / GPU resnet root pass) apply.
///
/// Before the fix, the root-candidate block intercepted the request, returned
/// early with `None` linear, and silently defeated every linear-extraction
/// caller — measured on cifar100: the PGD exact-gradient path (#4274) paid a
/// full certified forward-linear walk at a concrete point per step only to
/// get `None` back. The linear-consuming route must take the CPU backward
/// loop, the only producer of the linear map.
#[test]
fn run_with_linear_bypasses_root_candidates_on_conv_dag_w5() {
    use crate::layers::{AddLayer, Conv2dLayer, FlattenLayer};

    // Conv residual DAG: conv -> relu -> add(conv, relu) -> flatten -> linear.
    // Non-sequential (binary Add) + Conv2d = the root-candidate surface.
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, -0.25, 0.75, 0.4]).unwrap();
    let conv = Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.1_f32])), (1, 1), (0, 0), 4, 4)
        .unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        "relu",
        "conv",
    ));
    graph.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["residual".to_string()],
    ));
    let head = LinearLayer::new(
        arr2(&[
            [0.25_f32, -0.5, 0.75, 0.1, 0.0, 0.5, -0.2, 0.4, 0.3],
            [-0.4, 0.3, 0.2, -0.6, 0.5, -0.1, 0.7, -0.2, 0.15],
        ]),
        Some(arr1(&[0.05_f32, -0.1])),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(head),
        vec!["flat".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.2_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.6_f32),
    )
    .unwrap();

    // Sanity: this graph IS the root-candidate surface (conv + non-sequential).
    let exec = graph.exec_order().expect("exec order");
    assert!(
        graph.has_conv_layers() && !graph.is_sequential_graph(exec),
        "fixture must be a conv DAG so the root candidates would fire"
    );

    let node_bounds = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("IBP node bounds");
    let spec = arr2(&[[1.0_f32, -1.0]]);

    // Non-vacuousness: the C-margin root candidate MUST be computable on this
    // fixture — otherwise the root-candidate block would have fallen through to
    // the CPU loop anyway and this test could not distinguish the fix.
    graph
        .forward_linear_spec_margin_bounds(
            &input,
            &spec,
            None,
            // The cold-build admission floor is 30s; retain real headroom
            // after the callee takes its own wall-clock sample.
            Some(Instant::now() + std::time::Duration::from_mins(1)),
        )
        .expect("fixture must support the C-margin root candidate (test would be vacuous)");

    let (bounds, linear) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &input,
            &spec,
            None,
            &node_bounds,
            Some(Instant::now() + std::time::Duration::from_mins(1)),
        )
        .expect("spec-guided CROWN with linear should succeed on the toy conv DAG");

    assert!(
        bounds.lower()[[0]].is_finite() && bounds.upper()[[0]].is_finite(),
        "spec bounds must be finite"
    );
    assert!(
        linear.is_some(),
        "#w5: run_with_linear must produce the input LinearBounds — the bounds-only \
         root candidates must not intercept linear-extraction requests"
    );
}
