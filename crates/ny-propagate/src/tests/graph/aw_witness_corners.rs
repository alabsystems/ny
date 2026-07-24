// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ADVERSARIAL WITNESS-CORNER SOUNDNESS (#vnncomp-aw-soundness audit).
//!
//! Build a small GRAPH net mixing add / concat / residual / slice + ReLU, run
//! the production CROWN verdict path (`propagate_crown`), then assert at EVERY
//! corner of the input box that the concretized output bounds enclose the exact
//! f32 forward value with ZERO tolerance, AND that the bounds are finite (no
//! spurious -inf from the coeff-error machinery degrading a verdict-relevant row).

use crate::*;
use ndarray::{arr1, arr2};

/// Graph:
/// ```text
///            /-> linA --\
/// input -> relu0          Add -> relu1 -\
///            \-> linB --/                Concat(axis0) -> lin_out
///         skip = Slice(relu0)[0..2] -----/
/// ```
/// All ops on the residual path (Add, Concat, Slice) are EXACT linear carriers
/// of the certified A·W coefficient error; this exercises carrier propagation
/// through every one of them in a single backward pass.
fn build_residual_concat_relu_witness_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let w0 = arr2(&[[1.3_f32, -0.7, 0.4], [0.5, 1.1, -0.9], [-0.8, 0.6, 1.2]]);
    let b0 = arr1(&[0.07_f32, -0.05, 0.11]);
    graph.add_node(GraphNode::from_input(
        "linear0",
        Layer::Linear(LinearLayer::new(w0, Some(b0)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer),
        vec!["linear0".to_string()],
    ));

    let wa = arr2(&[[0.9_f32, -0.4, 0.2], [-0.3, 0.8, 0.5], [0.6, 0.1, -0.7]]);
    graph.add_node(GraphNode::new(
        "linA",
        Layer::Linear(LinearLayer::new(wa, None).unwrap()),
        vec!["relu0".to_string()],
    ));
    let wb = arr2(&[[-0.5_f32, 0.7, 0.3], [0.4, -0.6, 0.9], [0.2, 0.5, -0.1]]);
    graph.add_node(GraphNode::new(
        "linB",
        Layer::Linear(LinearLayer::new(wb, None).unwrap()),
        vec!["relu0".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["linA".to_string(), "linB".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "skip",
        Layer::Slice(SliceLayer::new(0, 0, 2)),
        vec!["relu0".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["relu1".to_string(), "skip".to_string()],
    ));

    let wo = arr2(&[[0.8_f32, -0.3, 0.5, 0.2, -0.6], [-0.4, 0.7, -0.2, 0.9, 0.1]]);
    let bo = arr1(&[0.03_f32, -0.02]);
    graph.add_node(GraphNode::new(
        "lin_out",
        Layer::Linear(LinearLayer::new(wo, Some(bo)).unwrap()),
        vec!["concat".to_string()],
    ));
    graph.set_output("lin_out");

    let input = BoundedTensor::new(
        arr1(&[-0.6_f32, -0.4, -0.5]).into_dyn(),
        arr1(&[0.7_f32, 0.5, 0.6]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

/// Exact f32 forward evaluation of the residual/concat/slice/relu witness graph.
fn residual_concat_forward(x: &[f32; 3]) -> [f32; 2] {
    let w0 = arr2(&[[1.3_f32, -0.7, 0.4], [0.5, 1.1, -0.9], [-0.8, 0.6, 1.2]]);
    let b0 = arr1(&[0.07_f32, -0.05, 0.11]);
    let wa = arr2(&[[0.9_f32, -0.4, 0.2], [-0.3, 0.8, 0.5], [0.6, 0.1, -0.7]]);
    let wb = arr2(&[[-0.5_f32, 0.7, 0.3], [0.4, -0.6, 0.9], [0.2, 0.5, -0.1]]);
    let wo = arr2(&[[0.8_f32, -0.3, 0.5, 0.2, -0.6], [-0.4, 0.7, -0.2, 0.9, 0.1]]);
    let bo = arr1(&[0.03_f32, -0.02]);

    let xv = arr1(&[x[0], x[1], x[2]]);
    let h0 = w0.dot(&xv) + &b0;
    let r0 = h0.mapv(|v| v.max(0.0));
    let a = wa.dot(&r0);
    let b = wb.dot(&r0);
    let add = &a + &b;
    let r1 = add.mapv(|v| v.max(0.0));
    let cat = arr1(&[r1[0], r1[1], r1[2], r0[0], r0[1]]);
    let out = wo.dot(&cat) + &bo;
    [out[0], out[1]]
}

#[test]
fn aw_residual_concat_slice_relu_witness_corners_sound_and_finite() {
    tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_residual_concat_relu_witness_graph();

        let bounds = graph
            .propagate_crown(&input)
            .expect("residual/concat/slice CROWN should produce bounds");

        let lo = bounds.lower();
        let hi = bounds.upper();

        // (1) FINITENESS / NO SPURIOUS -inf (tightness-regression guard).
        for k in 0..2usize {
            assert!(
                lo[[k]].is_finite(),
                "spurious non-finite LOWER bound at out[{k}]: {} (coeff-error machinery \
                 degraded a verdict-relevant row to -inf)",
                lo[[k]]
            );
            assert!(
                hi[[k]].is_finite(),
                "spurious non-finite UPPER bound at out[{k}]: {}",
                hi[[k]]
            );
            assert!(
                lo[[k]] <= hi[[k]],
                "inverted bound at out[{k}]: lower={} > upper={}",
                lo[[k]],
                hi[[k]]
            );
        }

        // (2) ZERO-TOLERANCE ENCLOSURE at every corner of the input box.
        let l = input.lower();
        let u = input.upper();
        let pick = |bit: usize, j: usize| if (bit >> j) & 1 == 0 { l[[j]] } else { u[[j]] };
        for corner in 0u32..(1 << 3) {
            let x = [
                pick(corner as usize, 0),
                pick(corner as usize, 1),
                pick(corner as usize, 2),
            ];
            let y = residual_concat_forward(&x);
            for k in 0..2usize {
                assert!(
                    lo[[k]] <= y[k] && y[k] <= hi[[k]],
                    "ZERO-TOLERANCE soundness violation at corner {corner} out[{k}]: \
                     exact={} not in [{}, {}] (x={:?})",
                    y[k],
                    lo[[k]],
                    hi[[k]],
                    x
                );
            }
        }

        // (3) NON-TRIVIALITY: bounds must not be implausibly wide (carrier did
        // not silently blow up the verdict path).
        for k in 0..2usize {
            let width = hi[[k]] - lo[[k]];
            assert!(
                width < 50.0,
                "bound at out[{k}] is implausibly wide ({width}) — error machinery \
                 likely over-widened the verdict path",
            );
        }

        // Surface the actual numbers for the audit log.
        println!(
            "AW-WITNESS bounds: out0=[{}, {}] out1=[{}, {}]",
            lo[[0]],
            hi[[0]],
            lo[[1]],
            hi[[1]]
        );
    });
}
