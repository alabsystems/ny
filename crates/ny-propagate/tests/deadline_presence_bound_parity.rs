// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CI guard for the `deadline`-PRESENCE degradation class.
//!
//! # The defect this exists to catch
//!
//! ny has a recurring, audited bug pattern: a FAST or EXACT or MORE-PRECISE code
//! path is guarded on the ABSENCE of a deadline —
//!
//! ```text
//! if deadline.is_none()      { /* good path */ }        // or
//! if !deadline_is_hard       { /* good path */ }        // or
//! if deadline.is_some()      { /* DEGRADED */ } else { /* good path */ }
//! ```
//!
//! Every scored competition run carries a deadline. So the degraded path is
//! taken **always in competition, and only in competition**. A 2026-08-17 audit
//! (30 candidates, 8 confirmed, adversarially verified) collapsed the known
//! instances into four root causes; the largest is the residual `Add` fan-out
//! refusing to clone a structured Patches carrier, which densifies the walk and
//! was measured at 257.56s CPU against 12.97s GPU for the resulting dense conv
//! backward. Two of the four do not merely slow the walk down — they DISCARD the
//! optimized alpha and revert the node to its heuristic slope.
//!
//! # Why the existing suite could never catch it
//!
//! Because tests do not set deadlines. Every instance in the class was invisible
//! for exactly that reason, and each was eventually found by hand, one at a time,
//! over multiple sessions. This test closes that hole by construction: it runs
//! the SAME workload twice, once with `deadline: None` and once with a deadline
//! so generous it cannot expire, and asserts the two bounds agree.
//!
//! A deadline that never expires must not change the answer. If it does, some
//! path keyed on deadline PRESENCE rather than EXPIRY, and that is the bug.
//!
//! # What a failure means
//!
//! `bounded` is looser than `unbounded` at some output index ⇒ a good path was
//! skipped merely because a deadline existed. Find the guard, and make it test
//! expiry (`Instant::now() >= limit`) instead of presence. Note that some guards
//! protect a real invariant — an unpollable kernel, or an admission that does not
//! charge its true retained footprint — so the fix is usually "make the path
//! cooperative", not "delete the guard".
//!
//! # Scope, stated honestly
//!
//! This covers the plain graph-CROWN backward, which is where the residual
//! fan-out and carrier-densification root causes live. It does NOT cover the
//! alpha-discard root causes: plain CROWN carries no optimized alpha, so a test
//! that exercises those needs an alpha ascent and belongs in a companion guard.
//! The graph below deliberately includes a residual `Add`, because that is the
//! node where a cifar100-style resnet's carrier actually dies.

use std::time::{Duration, Instant};

use ndarray::{Array1, ArrayD, IxDyn};
use ny_propagate::layers::{Conv2dLayer, ReLULayer, ReduceSumLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

/// A small residual conv graph: `conv -> relu -> conv -> Add(skip) -> relu`,
/// twice, then a 1x1 head and a spatial ReduceSum.
///
/// The `Add` is the point of the fixture. `backward_helpers.rs` declines to
/// clone a structured Patches carrier at a residual fan-out whenever a deadline
/// is present, and on a real resnet the demanded pre-activation targets
/// frequently ARE those Adds — so the decline fires at step 1 of the walk and
/// every later patches gate is dead code behind an already-Dense carrier.
fn build_residual_graph(
    channels: usize,
    hw: usize,
    blocks: usize,
) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let kernel = |out_ch: usize, in_ch: usize, k: usize| -> ArrayD<f32> {
        let numel = out_ch * in_ch * k * k;
        // Small alternating weights: finite bounds, while still exercising the
        // full conv backward. Values do not matter, shapes do.
        let data: Vec<f32> = (0..numel)
            .map(|i| if i % 2 == 0 { 0.04 } else { -0.031 })
            .collect();
        ArrayD::from_shape_vec(IxDyn(&[out_ch, in_ch, k, k]), data).expect("valid conv kernel")
    };
    let same_conv = |ch: usize| {
        Conv2dLayer::with_input_shape(
            kernel(ch, ch, 3),
            Some(Array1::zeros(ch)),
            (1, 1),
            (1, 1),
            hw,
            hw,
        )
        .expect("valid same-padding conv")
    };

    let mut prev = String::from(ny_propagate::NETWORK_INPUT);
    for block in 0..blocks {
        let skip = prev.clone();
        let (a, b, add, out) = (
            format!("conv_a_{block}"),
            format!("conv_b_{block}"),
            format!("add_{block}"),
            format!("relu_out_{block}"),
        );
        graph.add_node(GraphNode::new(
            &a,
            Layer::Conv2d(same_conv(channels)),
            vec![prev],
        ));
        graph.add_node(GraphNode::new(
            &format!("relu_a_{block}"),
            Layer::ReLU(ReLULayer),
            vec![a.clone()],
        ));
        graph.add_node(GraphNode::new(
            &b,
            Layer::Conv2d(same_conv(channels)),
            vec![format!("relu_a_{block}")],
        ));
        // The residual fan-out.
        graph.add_node(GraphNode::new(
            &add,
            Layer::Add(ny_propagate::layers::AddLayer),
            vec![b.clone(), skip],
        ));
        graph.add_node(GraphNode::new(&out, Layer::ReLU(ReLULayer), vec![add]));
        prev = out;
    }

    let head = Conv2dLayer::with_input_shape(
        kernel(2, channels, 1),
        Some(Array1::zeros(2)),
        (1, 1),
        (0, 0),
        hw,
        hw,
    )
    .expect("valid 1x1 head conv");
    graph.add_node(GraphNode::new("head", Layer::Conv2d(head), vec![prev]));
    graph.add_node(GraphNode::new(
        "out",
        Layer::ReduceSum(ReduceSumLayer::new(vec![1, 2], false)),
        vec!["head".to_string()],
    ));
    graph.set_output("out");

    let numel = channels * hw * hw;
    let lower = ArrayD::from_shape_vec(IxDyn(&[channels, hw, hw]), vec![-0.05_f32; numel])
        .expect("valid lower input");
    let upper = ArrayD::from_shape_vec(IxDyn(&[channels, hw, hw]), vec![0.05_f32; numel])
        .expect("valid upper input");
    let input = BoundedTensor::new(lower, upper).expect("valid input bounds");

    (graph, input)
}

// Anti-hang bound, not a benchmark. Debug builds run this residual CROWN
// unoptimized: 74 s in release here and 124 s in debug — a MARGINAL overshoot,
// so the flat bound failed the test for the BUILD PROFILE rather than for a
// hang. 300 s keeps roughly 2.4x headroom over the measured debug cost while
// still catching a real hang promptly.
#[cfg_attr(not(debug_assertions), ntest::timeout(120_000))]
#[cfg_attr(debug_assertions, ntest::timeout(300_000))]
#[test]
fn a_deadline_that_never_expires_does_not_change_the_bound() {
    let (graph, input) = build_residual_graph(8, 16, 2);

    let unbounded = graph
        .propagate_crown_with_engine_and_deadline(&input, None, None)
        .expect("CROWN backward without a deadline");

    // Generous enough that no cooperative check can fire. Any difference is
    // therefore attributable to deadline PRESENCE, not to expiry.
    let far_future = Instant::now() + Duration::from_hours(1);
    let bounded = graph
        .propagate_crown_with_engine_and_deadline(&input, None, Some(far_future))
        .expect("CROWN backward with a non-expiring deadline");

    let (ul, uu) = (unbounded.bounds.lower(), unbounded.bounds.upper());
    let (bl, bu) = (bounded.bounds.lower(), bounded.bounds.upper());
    assert_eq!(ul.len(), bl.len(), "output width differs between arms");

    // Both arms must be sound; we are testing TIGHTNESS parity, so compare with a
    // relative tolerance rather than bit equality — the two routes may associate
    // floating-point operations differently even when computing the same value.
    let tol = |a: f32, b: f32| {
        let scale = a.abs().max(b.abs()).max(1.0);
        (a - b).abs() <= 1e-4 * scale
    };

    for i in 0..ul.len() {
        assert!(
            tol(ul[i], bl[i]) && tol(uu[i], bu[i]),
            "DEADLINE-PRESENCE DEGRADATION at output {i}:\n  \
             no deadline: [{:.6}, {:.6}]\n  \
             non-expiring deadline: [{:.6}, {:.6}]\n\
             A deadline that cannot expire changed the bound, so some path is \
             guarded on deadline PRESENCE instead of EXPIRY. Grep for \
             `deadline.is_none()`, `!deadline_is_hard`, and \
             `if deadline.is_some() {{ .. }} else {{ .. }}`. See \
             docs/ and the audit notes on the four root causes; the fix is \
             normally to make the path cooperative, not to delete the guard.",
            ul[i],
            uu[i],
            bl[i],
            bu[i],
        );
    }

    // Guard the guard: a fixture that produced degenerate (all-equal or
    // non-finite) bounds would pass vacuously no matter how badly the walk
    // degraded. This project has lost real measurements to exactly that shape.
    assert!(
        ul.iter().chain(uu.iter()).all(|v| v.is_finite()),
        "fixture produced non-finite bounds — the parity assertion above would be vacuous"
    );
    assert!(
        (0..ul.len()).any(|i| uu[i] - ul[i] > 1e-6),
        "fixture produced a zero-width box — the parity assertion above would be vacuous"
    );
}
