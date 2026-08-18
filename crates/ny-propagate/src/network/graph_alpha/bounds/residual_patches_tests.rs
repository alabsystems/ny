// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for patches CROWN-backward across a residual `Add`
//! (#conv-crown-residual).
//!
//! Before the fix, two independent `inputs.len() == 1` guards forced every
//! ResNet target onto the dense path:
//!
//! 1. [`GraphNetwork::crown_ibp_target_can_start_in_patches`] refused to *seed*
//!    in patches when the target node itself had two inputs — and on a ResNet
//!    the demanded pre-activation targets frequently ARE the residual `Add`s.
//! 2. The target-backward step gate refused to *cross* a two-input node, so a
//!    walk densified at the first residual `Add` regardless of how small the
//!    composed receptive field was.
//!
//! IBP width compounds multiplicatively through convolutions, so a densified
//! (and therefore budget-refused) target degrades to IBP and the whole
//! downstream bound collapses. See
//! `docs/PATCHES_RESIDUAL_ADD_ROOT_CAUSE_2026-07-27.md`.

use super::*;
use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds};
use crate::layers::{AddLayer, Conv2dLayer, ReLULayer};
use crate::network::{CrownMergeAccumulator, GraphNode};
use crate::types::BoundsProvenance;
use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::NyError;

const SPATIAL: usize = 32;

#[test]
fn anchored_residual_clone_budget_refusal_is_atomic() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_DENSE_BUDGET_MB", "0");

        let geometry =
            PatchGeometry::anchored(vec![0, 1], vec![0, 1]).expect("fixture axes are non-empty");
        let side = |fill| PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(IxDyn(&[1, 2, 2, 1, 1, 1]), fill)),
            geometry: geometry.clone(),
            identity: false,
            output_shape: (1, 2, 2),
            input_shape: (1, 2, 2),
            unstable_idx: None,
        };
        let expected = PatchesLinearBounds {
            row_count: 4,
            lower_a: side(0.25),
            lower_b: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            upper_a: side(0.75),
            upper_b: Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]),
        };
        let mut node_cb = CrownBounds::Patches(Box::new(expected.clone()));
        let node = GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["left".to_string(), "right".to_string()],
        );
        let branch_box = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0),
            ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0),
        )
        .expect("valid branch box");
        let node_bounds = std::collections::HashMap::from([
            ("left".to_string(), branch_box.clone()),
            ("right".to_string(), branch_box),
        ]);
        let mut accumulator = CrownMergeAccumulator::new();
        let mut input_accumulated = false;
        let error =
            crate::network::core::graph::backward_helpers::try_apply_patches_residual_passthrough(
                &GraphNetwork::new(),
                &node,
                &mut node_cb,
                &node_bounds,
                &mut accumulator,
                4,
                4,
                &mut input_accumulated,
                "test",
            )
            .expect_err("zero budget must refuse the one required residual branch clone");
        assert!(matches!(error, NyError::CpuMemoryExceeded { .. }));
        let CrownBounds::Patches(actual) = node_cb else {
            panic!("clone refusal changed the source carrier type");
        };
        assert_eq!(actual.lower_a.patches, expected.lower_a.patches);
        assert_eq!(actual.upper_a.patches, expected.upper_a.patches);
        assert_eq!(actual.lower_a.geometry, expected.lower_a.geometry);
        assert_eq!(actual.upper_a.geometry, expected.upper_a.geometry);
        assert_eq!(actual.lower_b, expected.lower_b);
        assert_eq!(actual.upper_b, expected.upper_b);
        assert!(accumulator.is_empty());
        assert!(!input_accumulated);
    });
}

/// A 3x3 stride-1 same-padding single-channel conv over a `SPATIAL x SPATIAL`
/// grid, with deterministic non-symmetric weights.
fn residual_conv(seed: f32) -> Conv2dLayer {
    let mut kernel = ArrayD::zeros(IxDyn(&[1, 1, 3, 3]));
    for i in 0..3 {
        for j in 0..3 {
            // Deterministic, mixed-sign, magnitude < 1 so the stack stays bounded.
            kernel[[0, 0, i, j]] = seed * (((i * 3 + j) as f32) - 4.0) / 10.0;
        }
    }
    Conv2dLayer::with_input_shape(
        kernel,
        Some(Array1::from_vec(vec![0.01_f32])),
        (1, 1),
        (1, 1),
        SPATIAL,
        SPATIAL,
    )
    .expect("valid residual conv")
}

/// A minimal ResNet block:
///
/// ```text
/// input -> conv1 -> relu1 -> conv2 -> add(conv2, conv1) -> relu2 -> conv3
/// ```
///
/// The demanded CROWN-IBP targets are the pre-activation nodes `conv1`, `add`
/// and the output `conv3`. That covers both defects: `add` is a target whose
/// own layer is a two-input `Add` (gate 1), and `conv3`'s backward walk crosses
/// that same `Add` mid-walk (gate 2).
fn build_resnet_block_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "conv1",
        Layer::Conv2d(residual_conv(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(residual_conv(0.7)),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["conv2".to_string(), "conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "conv3",
        Layer::Conv2d(residual_conv(0.5)),
        vec!["relu2".to_string()],
    ));
    graph.set_output("conv3");

    let center = ArrayD::from_elem(IxDyn(&[1, SPATIAL, SPATIAL]), 0.3_f32);
    let input = BoundedTensor::from_epsilon(center, 0.02).expect("valid input box");
    (graph, input)
}

/// Gate 1: a two-input elementwise `Add` target must be admitted as a
/// patches start. `ancestors()` is inclusive (`traversal.rs:60`), so the walk's
/// first backward step crosses the target's own layer — which the patches
/// residual passthrough consumes natively.
#[ntest::timeout(60000)]
#[test]
fn residual_add_target_can_start_in_patches() {
    crate::tests::with_crown_dense_budget_mb("1", || {
        let (graph, _input) = build_resnet_block_graph();
        let add_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, SPATIAL, SPATIAL]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, SPATIAL, SPATIAL]), 1.0_f32),
        )
        .expect("valid add bounds");

        assert!(
            graph.crown_ibp_target_can_start_in_patches("add", &add_bounds),
            "#conv-crown-residual: a same-shape 2-input Add target must be patches-startable; \
             refusing it densifies every ResNet pre-activation target"
        );
    });
}

/// Gate 3, pinned to the real benchmark geometry: on `CIFAR100_resnet_medium`
/// the largest demanded target is `target_dim` 14400 over a 3x32x32 = 3072
/// input, so the dense identity pair is 1.659 GB and the dense backward pair
/// 0.354 GB — BOTH under the 2 GiB default budget. Admitting patches only when
/// dense *OOMs* therefore left every one of that model's targets on the slow
/// dense path, which made the residual-`Add` route above dead code on all 100
/// of its instances.
#[ntest::timeout(60000)]
#[test]
fn expensive_but_fitting_dense_backward_admits_patches() {
    // 3x32x32 input -> 16 channels, 3x3 valid conv -> 16x30x30 = 14400.
    let mut kernel = ArrayD::zeros(IxDyn(&[16, 3, 3, 3]));
    for (i, v) in kernel.iter_mut().enumerate() {
        *v = ((i % 7) as f32 - 3.0) / 20.0;
    }
    let conv =
        Conv2dLayer::with_input_shape(kernel, Some(Array1::zeros(16)), (1, 1), (0, 0), 32, 32)
            .expect("valid conv");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.set_output("conv");

    let target = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[16, 30, 30]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[16, 30, 30]), 1.0_f32),
    )
    .expect("valid target bounds");
    assert_eq!(
        target.len(),
        14400,
        "pinning the CIFAR100_resnet_medium shape"
    );

    // At the shipped 2 GiB default neither OOM condition fires (1.659 GB and
    // 0.354 GB both fit), so this target is admitted purely on cost.
    crate::tests::with_crown_dense_budget_mb("2048", || {
        assert!(
            graph.crown_ibp_target_can_start_in_patches("conv", &target),
            "#conv-crown-residual: a 14400-dim conv target whose dense backward pair is \
             0.354 GB must prefer patches; admitting only on OOM leaves every \
             CIFAR100_resnet_medium target on the dense path"
        );
    });
}

/// The cost admission must NOT drag small conv targets off their existing dense
/// route: those are cheap dense, and dense is tighter for thin seeds through
/// overlapping receptive fields (#cgan-alpha-on-tight-refs).
#[ntest::timeout(60000)]
#[test]
fn cheap_dense_backward_keeps_dense_route() {
    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_elem(IxDyn(&[1, 1, 3, 3]), 0.1_f32),
        Some(Array1::zeros(1)),
        (1, 1),
        (0, 0),
        8,
        8,
    )
    .expect("valid conv");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.set_output("conv");

    let target = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 6, 6]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 6, 6]), 1.0_f32),
    )
    .expect("valid target bounds");

    crate::tests::with_crown_dense_budget_mb("2048", || {
        assert!(
            !graph.crown_ibp_target_can_start_in_patches("conv", &target),
            "#conv-crown-residual: a 36-dim target's dense pair is kilobytes; it must keep \
             the existing dense route"
        );
    });
}

/// Non-residual multi-input nodes must stay excluded. `Concat` has no patches
/// branch/merge rule, and the generic single-input patches step would
/// misattribute the whole relation to `inputs[0]` — unsound, not merely loose.
#[ntest::timeout(60000)]
#[test]
fn non_elementwise_multi_input_target_still_refuses_patches_start() {
    use crate::layers::ConcatLayer;

    crate::tests::with_crown_dense_budget_mb("1", || {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "conv1",
            Layer::Conv2d(residual_conv(1.0)),
        ));
        graph.add_node(GraphNode::new(
            "conv2",
            Layer::Conv2d(residual_conv(0.7)),
            vec!["conv1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "cat",
            Layer::Concat(ConcatLayer::new(0)),
            vec!["conv1".to_string(), "conv2".to_string()],
        ));
        graph.set_output("cat");

        let cat_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[2, SPATIAL, SPATIAL]), 1.0_f32),
        )
        .expect("valid concat bounds");

        assert!(
            !graph.crown_ibp_target_can_start_in_patches("cat", &cat_bounds),
            "#conv-crown-residual: only elementwise Add/Sub may seed in patches; \
             Concat has no patches branch/merge rule"
        );
    });
}

/// Gates 1 and 2 end to end: with a dense budget too small for the dense
/// identity pair, every target of the ResNet block must still reach a CROWN
/// bound rather than degrading to IBP.
///
/// This is the miniature of the TinyYOLO measurement in
/// `docs/PATCHES_RESIDUAL_ADD_ROOT_CAUSE_2026-07-27.md`: there, `Conv_1` (zero
/// residual `Add`s on its backward path) was the ONLY target of eight to keep a
/// CROWN bound. Here `conv1` is that same node, and `add`/`conv3` are the ones
/// the fix recovers.
#[ntest::timeout(120000)]
#[test]
fn resnet_block_targets_keep_crown_provenance_across_residual() {
    let (graph, input) = build_resnet_block_graph();

    // The authenticated source-plus-unfold receipt is 1,429,512 bytes for this
    // fixture. Two MiB is therefore the smallest whole-MiB cap that admits the
    // Patches route, while remaining far below its >8 MiB dense identity pair.
    let result = crate::tests::with_crown_dense_budget_mb("2", || {
        graph.collect_crown_ibp_bounds_dag_with_status(&input)
    })
    .expect("collection succeeds");

    // Surface why any target degraded — an opaque `ForwardFallback(..)` is not
    // actionable, and the fallback details name the exact refusing site.
    let details: Vec<String> = result
        .fallback_events
        .iter()
        .map(|e| format!("{:?}: {}", e.reason, e.details))
        .collect();

    // `conv1` is the control: no residual Add on its backward path, so it kept a
    // CROWN bound even before the fix. If this regresses, the failure is in the
    // patches path generally, not in the residual handling.
    assert_eq!(
        result.provenance.get("conv1"),
        Some(&BoundsProvenance::Crown),
        "control target (no residual Add on its backward path) must keep CROWN"
    );

    // `add` exercises gate 1 (the target IS the two-input node).
    assert_eq!(
        result.provenance.get("add"),
        Some(&BoundsProvenance::Crown),
        "#conv-crown-residual: a residual Add target degraded to {:?}; the patches \
         start gate is refusing two-input targets again. fallbacks: {:#?}",
        result.provenance.get("add"),
        details
    );

    // `conv3` exercises gate 2 (the walk crosses the Add mid-walk).
    assert_eq!(
        result.provenance.get("conv3"),
        Some(&BoundsProvenance::Crown),
        "#conv-crown-residual: a target downstream of a residual Add degraded to \
         {:?}; the backward walk is densifying at the Add again. fallbacks: {:#?}",
        result.provenance.get("conv3"),
        details
    );
}

/// Soundness: the patches route across the residual must ENCLOSE the network's
/// true outputs. The duplication down both branches is only sound if the join
/// sums the two contributions; a dropped branch shows up here as a bound that
/// fails to contain a sampled point.
#[ntest::timeout(120000)]
#[test]
fn residual_patches_bounds_enclose_sampled_outputs() {
    let (graph, input) = build_resnet_block_graph();

    // Match the provenance regression above: 2 MiB admits the 1,429,512-byte
    // authenticated Patches receipt but still refuses the >8 MiB dense pair.
    let result = crate::tests::with_crown_dense_budget_mb("2", || {
        graph.collect_crown_ibp_bounds_dag_with_status(&input)
    })
    .expect("collection succeeds");

    // Sample the input box (including both corners and the centre) and require
    // every node's collected bound to contain the concrete forward value.
    for &t in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let point = ndarray::Zip::from(input.lower())
            .and(input.upper())
            .map_collect(|&l, &u| l + t * (u - l));
        let point_input = BoundedTensor::concrete(point).expect("valid concrete point");

        // Collecting over a DEGENERATE box evaluates the network exactly: every
        // node's interval collapses to its concrete forward value.
        let exact = crate::tests::with_crown_dense_budget_mb("2", || {
            graph.collect_crown_ibp_bounds_dag_with_status(&point_input)
        })
        .expect("point collection succeeds");

        for name in ["conv1", "add", "conv3"] {
            let bound = result
                .bounds
                .get(name)
                .unwrap_or_else(|| panic!("missing bounds for {name}"));
            let concrete = exact
                .bounds
                .get(name)
                .unwrap_or_else(|| panic!("missing point bounds for {name}"));

            for (idx, ((&value, &lower), &upper)) in concrete
                .lower()
                .iter()
                .zip(bound.lower().iter())
                .zip(bound.upper().iter())
                .enumerate()
            {
                assert!(
                    value >= lower - 1e-4,
                    "#conv-crown-residual soundness: {name}[{idx}] = {value} < lower {lower} \
                     at t={t}"
                );
                assert!(
                    value <= upper + 1e-4,
                    "#conv-crown-residual soundness: {name}[{idx}] = {value} > upper {upper} \
                     at t={t}"
                );
            }
        }
    }
}
/// FALSE-BOUND regression for `Sub` CROWN backward.
///
/// `SubLayer::propagate_linear_binary` used to give the right operand
/// `lower_a' = -upper_a`, `upper_a' = -lower_a` — a "negate and swap". Swapping
/// is the rule for negating a bounded QUANTITY (`l <= x <= u` gives
/// `-u <= -x <= -l`), not for negating the coefficients of a linear relation:
/// CROWN composes `y = u - v` by substitution, so each relation keeps its own
/// coefficients negated.
///
/// The bug is invisible whenever `lower_a == upper_a` (e.g. an identity seed at
/// the output node), which is why the existing unit tests missed it. It needs
/// BOTH a relaxation downstream of the `Sub` (to make the two coefficient
/// matrices differ) AND a right operand that can go negative. This graph has
/// both: `u = x`, `v = -x`, `y = u - v = 2x`, then a ReLU that genuinely
/// crosses zero over `x in [-1, 3]`.
///
/// Measured before the fix: CROWN returned an upper bound of **4.2562** while
/// `relu(2 * 2.2) = 4.4` is reachable — a FALSE bound, i.e. a verdict of
/// VERIFIED on a violated property.
#[ntest::timeout(60000)]
#[test]
fn sub_backward_encloses_when_right_operand_can_be_negative() {
    use crate::layers::{Layer, LinearLayer, ReLULayer, SubLayer};
    use crate::network::core::{GraphNetwork, GraphNode};
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    // u = x, v = -x  =>  y = u - v = 2x.  v is NEGATIVE whenever x > 0.
    // A ReLU after the Sub makes the backward relation at the Sub have
    // lower_a != upper_a, which is what the negate-and-swap actually affects.
    let u = LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap();
    let v = LinearLayer::new(arr2(&[[-1.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap();
    let out = LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("u", Layer::Linear(u)));
    graph.add_node(GraphNode::from_input("v", Layer::Linear(v)));
    graph.add_node(GraphNode::new(
        "y",
        Layer::Sub(SubLayer),
        vec!["u".to_string(), "v".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "z",
        Layer::ReLU(ReLULayer),
        vec!["y".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(out),
        vec!["z".to_string()],
    ));
    graph.set_output("out");

    // x in [-1, 3] => y = 2x in [-2, 6], so the ReLU genuinely crosses zero.
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0_f32]).unwrap(),
    )
    .unwrap();

    let bounds = graph.propagate_crown(&input).unwrap();
    let (lo, hi) = (bounds.lower()[[0]], bounds.upper()[[0]]);

    for i in 0..=40 {
        let x = -1.0 + (i as f32) * 0.1;
        let y = (2.0_f32 * x).max(0.0);
        assert!(
            y >= lo - 1e-4,
            "FALSE LOWER BOUND: relu(2*{x})={y} < CROWN lower {lo}"
        );
        assert!(
            y <= hi + 1e-4,
            "FALSE UPPER BOUND: relu(2*{x})={y} > CROWN upper {hi}"
        );
    }
}

/// The RAM-adaptive Conv2d CROWN memory cap must not sit at its floor on hosts
/// whose total RAM is readable.
///
/// `total_system_ram_bytes()` read only `/proc/meminfo`, which does not exist on
/// Darwin, so every Mac fell back to the fixed 512 MiB floor. Measured cost on
/// the path this change opens: a TinyYOLO target needing 549,755,648 bytes was
/// refused by the 536,870,912-byte floor — short by 2.4% — and kept its IBP
/// bound. This host reports 36 GB, for which the policy grants
/// `clamp(36 GB / 16, 512 MiB, 16 GiB)` = 2.25 GiB.
#[cfg(target_os = "macos")]
#[ntest::timeout(30000)]
#[test]
fn macos_crown_mem_cap_is_not_pinned_at_the_floor() {
    let cap_mb = crate::layers::convolution::conv2d::effective_crown_mem_cap_mb_for_test();
    assert!(
        cap_mb > 512,
        "#conv-crown-residual: the RAM-adaptive Conv2d CROWN cap is still at the \
         512 MiB floor on macOS ({cap_mb} MiB); the sysctl hw.memsize probe is not \
         reaching the policy"
    );
}

/// The forward-linear cold-build admission gate must be GRAPH-SIZE AWARE
/// (#forward-linear-cost-gate).
///
/// The gate was a fixed 30 s floor whose comment calibrated it against a pass
/// "measured at roughly 22--25 seconds". `CIFAR100_resnet_medium`'s cold build
/// is 559.4 G f64 MACs and takes ~102 s on this host, so at the scored 100 s
/// budget the pass STARTED, burned 45 s (45% of the whole budget), hit its
/// deadline mid-GEMM and returned nothing — the run fell back to plain IBP
/// exactly as if the pass had never run, minus the budget.
///
/// This builds a conv stack of the same order (3x32x32 input, 3x3 pad-1 convs)
/// and asserts the estimator sees enough work to refuse a short budget while
/// still admitting a long one.
#[ntest::timeout(60000)]
#[test]
fn forward_linear_cold_build_gate_is_graph_size_aware() {
    use std::time::{Duration, Instant};

    let mut graph = GraphNetwork::new();
    let mut prev: Option<String> = None;
    for i in 0..8 {
        let name = format!("conv{i}");
        // 16 channels, 3x3 pad-1 stride-1 over 32x32 — a CIFAR-ResNet conv.
        let mut kernel = ArrayD::zeros(IxDyn(&[16, 16, 3, 3]));
        for (j, v) in kernel.iter_mut().enumerate() {
            *v = ((j % 5) as f32 - 2.0) / 50.0;
        }
        let conv =
            Conv2dLayer::with_input_shape(kernel, Some(Array1::zeros(16)), (1, 1), (1, 1), 32, 32)
                .expect("valid conv");
        let node = match prev {
            None => GraphNode::from_input(&name, Layer::Conv2d(conv)),
            Some(p) => GraphNode::new(&name, Layer::Conv2d(conv), vec![p]),
        };
        graph.add_node(node);
        prev = Some(name);
    }
    graph.set_output(&prev.expect("at least one conv"));

    let input_numel = 3 * 32 * 32;
    let macs = graph
        .forward_linear_cold_build_macs(input_numel)
        .expect("conv graph must be estimable");

    // 8 convs x (32*32 out) x (16*3*3 contraction) x 16 out-c x 3072 inputs x 2
    // passes = 7.4e11 MACs. Anything near zero means the walk missed the convs.
    assert!(
        macs > 100_000_000_000,
        "#forward-linear-cost-gate: estimator saw only {macs} MACs for an 8-conv \
         CIFAR-scale stack; it is not seeing the conv geometry"
    );

    // Predicted cost at an EXPLICIT rate (the old shipped constant). The
    // production gate now measures its rate per-process, so admission logic is
    // tested through the pure `_with_rate` core — otherwise this test would
    // depend on whatever this host happens to calibrate.
    const TEST_RATE: u128 = 5_500_000_000;
    let predicted_secs = (macs / TEST_RATE) as u64;
    assert!(
        predicted_secs >= 10,
        "#forward-linear-cost-gate: an 8-conv CIFAR-scale stack should predict at \
         least ~10 s, got {predicted_secs}s"
    );

    // A budget well below the predicted cost must be refused — that case is
    // exactly how 45% of cifar100's budget was spent on a pass that returned
    // nothing. (Half the prediction is also below the x5/4 admission margin.)
    let now = Instant::now();
    assert!(
        !GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            Some(macs),
            Some(now + Duration::from_secs(predicted_secs / 2)),
            now,
            TEST_RATE,
        ),
        "#forward-linear-cost-gate: half the predicted {predicted_secs}s must not be \
         admitted"
    );

    // ...and a budget comfortably above it (and above the x5/4 margin) must
    // still be admitted, so this gate never refuses work the budget could
    // actually complete.
    assert!(
        GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            Some(macs),
            Some(now + Duration::from_secs(predicted_secs * 4 + 60)),
            now,
            TEST_RATE,
        ),
        "#forward-linear-cost-gate: 4x the predicted {predicted_secs}s must still be \
         admitted"
    );

    // No deadline is always affordable (offline/analysis use).
    assert!(
        GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            Some(macs),
            None,
            now,
            TEST_RATE,
        )
    );

    // No MAC estimate keeps the caller on the fixed floor.
    assert!(
        GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            None,
            Some(now + Duration::from_secs(1)),
            now,
            TEST_RATE,
        )
    );
}

/// #forward-linear-cost-gate: admission must respond to the CALIBRATED rate
/// in both directions — a fast measured rate admits a build at the scored
/// 100 s tier that a stale slow rate refuses (the 0/60 lockout shape), and a
/// slow measured rate refuses what a fast one admits (the floor-smoke
/// over-admission shape). Also: admitted sets are monotone in the rate.
#[ntest::timeout(60000)]
#[test]
fn forward_linear_admission_tracks_calibrated_rate_both_directions() {
    use std::time::{Duration, Instant};

    // cifar100_resnet_medium's measured cold build: 559.4 G f64 MACs.
    let macs: u128 = 559_400_000_000;
    let now = Instant::now();
    let deadline_100s = Some(now + Duration::from_secs(100));

    // Stale 5.5 GMAC/s: predicted ~101 s -> refused at 100 s (tonight's chain).
    assert!(
        !GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            Some(macs),
            deadline_100s,
            now,
            5_500_000_000,
        ),
        "stale 5.5 GMAC/s must refuse the 559 GMAC build at 100 s"
    );

    // Measured ~23 GMAC/s (559.4 GMAC / ~24 s real): predicted ~24 s, x5/4 =
    // ~30 s -> admitted at 100 s.
    assert!(
        GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            Some(macs),
            deadline_100s,
            now,
            23_000_000_000,
        ),
        "the measured ~23 GMAC/s rate must admit the 559 GMAC build at 100 s"
    );

    // Slow direction: a rate whose prediction crosses the x5/4 margin refuses.
    // predicted = 559.4/7 = ~79 s, padded ~99 s -> admitted at 100 s...
    assert!(
        GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            Some(macs),
            deadline_100s,
            now,
            7_000_000_000,
        )
    );
    // ...but 6.5 GMAC/s predicts ~86 s, padded ~107 s -> refused: the margin
    // keeps a marginal admit from starving BaB.
    assert!(
        !GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            Some(macs),
            deadline_100s,
            now,
            6_500_000_000,
        ),
        "the x5/4 admission margin must refuse a build that would consume ~86 of 100 s"
    );

    // Monotonicity: any deadline admitted at rate r is admitted at rate > r.
    for secs in [10u64, 30, 50, 100, 200, 400] {
        let deadline = Some(now + Duration::from_secs(secs));
        let slow = GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            Some(macs),
            deadline,
            now,
            5_500_000_000,
        );
        let fast = GraphNetwork::forward_linear_cold_build_affordable_with_rate(
            Some(macs),
            deadline,
            now,
            23_000_000_000,
        );
        assert!(
            !slow || fast,
            "admission must be monotone in the rate (deadline {secs}s: slow admitted \
             but fast refused)"
        );
    }
}

/// #deadline-gemm micro-measurement: a finite deadline must not cost the GEMM.
///
/// Before the fix, `conv2d_transpose_pair_batched_gemm_grouped_with_deadline`
/// returned the certified SCALAR f64 contraction for every finite-deadline call,
/// so every scored run (which always sets a deadline) executed the scalar route
/// while tests and benchmarks (no deadline) got the GEMM. This times both and
/// asserts the deadline path is within a small factor of the deadline-free one.
#[ntest::timeout(300000)]
#[test]
fn finite_deadline_conv_backward_is_not_orders_slower_than_gemm() {
    use crate::layers::convolution::conv2d::conv2d_transpose_pair_batched_gemm_grouped_with_deadline as pair;
    use ndarray::{Array2, ArrayD, IxDyn};
    use std::time::{Duration, Instant};

    // oval21/CIFAR-scale: 16 out-channels over a 16x16 grid, 3x3 kernel.
    let (objs, out_c, in_c, kh, kw) = (64usize, 16usize, 16usize, 3usize, 3usize);
    let (gh, gw) = (16usize, 16usize);
    let mut kernel = ArrayD::zeros(IxDyn(&[out_c, in_c, kh, kw]));
    for (i, v) in kernel.iter_mut().enumerate() {
        *v = ((i % 11) as f32 - 5.0) / 37.0;
    }
    let mk = |seed: f32| {
        Array2::<f32>::from_shape_fn((objs, out_c * gh * gw), |(r, c)| {
            seed * (((r * 7 + c * 13) % 17) as f32 - 8.0) / 23.0
        })
    };
    let (lo, up) = (mk(1.0), mk(1.1));

    // Pin the CPU dense budget: sibling tests in this process set
    // NY_DENSE_BUDGET_MB to 1 MiB, and env is process-global, so without this
    // the pair below fails with CpuMemoryExceeded under a parallel run. The
    // helper also serialises against those tests.
    crate::tests::with_crown_dense_budget_mb("2048", || {
        let run = |deadline: Option<Instant>| {
            pair(
                &lo,
                &up,
                &kernel,
                (1, 1),
                (1, 1),
                (1, 1),
                (gh, gw),
                (gh, gw),
                out_c,
                1,
                None,
                deadline,
            )
        };

        // Warm up, then time each route.
        let _ = run(None).expect("no-deadline pair");
        let t0 = Instant::now();
        let free = run(None).expect("no-deadline pair");
        let free_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let timed = run(Some(Instant::now() + Duration::from_mins(2))).expect("deadline pair");
        let timed_ms = t1.elapsed().as_secs_f64() * 1e3;

        eprintln!(
            "[deadline-gemm] no-deadline {free_ms:.1} ms, finite-deadline {timed_ms:.1} ms, ratio {:.2}x",
            timed_ms / free_ms.max(1e-9)
        );

        // Same shape either way; both routes are sound (the GEMM is the one the
        // caller's certified error channel is sized for).
        assert_eq!(free.0.dim(), timed.0.dim());

        // Deliberately loose: the regression this guards is 62.8x, and a
        // wall-clock ratio measured inside a parallel test run is load-sensitive.
        // 20x still catches the scalar-route regression with 3x margin.
        assert!(
            timed_ms < free_ms * 20.0 + 200.0,
            "#deadline-gemm: the finite-deadline conv backward took {timed_ms:.1} ms vs \
             {free_ms:.1} ms without a deadline ({:.1}x). A deadline must not drop the \
             GEMM — every scored run sets one.",
            timed_ms / free_ms.max(1e-9)
        );
    });
}

/// Micro-measurement for the CPU GEMM adapter's row-major handling.
///
/// `FaerCpuGemmEngine::gemm_f32` used to repack both operands into owned
/// column-major `Mat`s with `Mat::from_fn(..., |i, j| a[i*k+j])`, which walks a
/// row-major source with stride `k` — one cache line per element. Profiling the
/// patches compose seam attributed 52.8% of its samples to that repack.
///
/// MEASURE THIS IN RELEASE (`cargo test --release`). In the dev profile faer is
/// unoptimised and the whole GEMM runs at ~2.9 GFLOP/s, where the repack is
/// invisible and the comparison INVERTS — dev showed the repack "faster"
/// (149 vs 162 ms) while release shows the opposite:
///   old repack      1.80 ms/call, 262.3 GFLOP/s
///   row-major borrow 1.48 ms/call, 319.9 GFLOP/s   (1.22x, identical checksum)
#[ntest::timeout(300000)]
#[test]
fn cpu_gemm_f32_row_major_throughput() {
    use ny_core::GemmEngine;
    use std::time::Instant;

    // Shape representative of the patches compose seam on a cifar100 conv
    // target: [num_positions x k_contraction] @ [k_contraction x n_dim].
    let (m, k, n) = (1024usize, 576usize, 400usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) / 17.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 23.0).collect();

    let engine = crate::faer_parallelism::FaerCpuGemmEngine;
    let _ = engine.gemm_f32(m, k, n, &a, &b).expect("warmup");

    let reps = 5;
    let t0 = Instant::now();
    let mut checksum = 0.0f64;
    for _ in 0..reps {
        let c = engine.gemm_f32(m, k, n, &a, &b).expect("gemm");
        checksum += f64::from(c[0]);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
    let gflops = 2.0 * (m * k * n) as f64 / (ms * 1e-3) / 1e9;
    eprintln!(
        "[cpu-gemm-f32] {m}x{k}x{n}: {ms:.2} ms/call, {gflops:.2} GFLOP/s (checksum {checksum:.3})"
    );
    assert!(ms > 0.0);
}
