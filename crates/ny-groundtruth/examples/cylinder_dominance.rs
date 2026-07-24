// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end ground-truth verification demo on the a3d acceptance shape:
//! an axis-aligned cylinder residual (`docs/GEOMETRIC_GROUND_TRUTH_PLAN.md`).
//!
//! This example stands in for the deferred `ny gt` CLI (M2): it builds the
//! ground truth, evaluates it at a point (`gt eval`), verifies a dominant
//! surrogate (`gt verify`), falsifies a violating one with a concrete
//! witness, and shows the §2.3 exact-constant rejection.
//!
//! Run with: `cargo run -p ny-groundtruth --example cylinder_dominance`

use ndarray::arr1;
use ny_core::Bound;
use ny_groundtruth::{
    cylinder_residual, reference, verify_against_ground_truth, GroundTruthOutcome, Relation,
};
use ny_propagate::layers::AddConstantLayer;
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

/// `f = g + k`: an exactly representable constant shift of a graph.
fn plus_constant(mut g: GraphNetwork, k: f32) -> GraphNetwork {
    let out = g.output_name().to_string();
    g.add_node(GraphNode::new(
        "surrogate_shift",
        Layer::AddConstant(AddConstantLayer::new(arr1(&[k]).into_dyn())),
        vec![out],
    ));
    g.set_output("surrogate_shift");
    g
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ground truth: cylinder along z through (1, -2, 0.5), radius 1.5 —
    // residual ||(x - p) - ((x - p) . a) a||^2 - r^2.
    let axis = [0.0, 0.0, 1.0];
    let point = [1.0, -2.0, 0.5];
    let radius = 1.5;
    let build = || cylinder_residual(axis, point, radius).expect("exact parameters");
    let g = build();

    // "gt eval": sound point evaluation via zero-width IBP vs the exact
    // rational reference.
    let x = [2.5_f32, -2.0, 7.0]; // on the surface
    let arr = arr1(&x).into_dyn();
    let enclosure = g.propagate_ibp(&BoundedTensor::new(arr.clone(), arr)?)?;
    println!(
        "g({x:?}) in [{}, {}] (exact reference: {})",
        enclosure.lower()[0],
        enclosure.upper()[0],
        reference::cylinder_residual(axis, point, radius, [2.5, -2.0, 7.0]),
    );

    // "gt verify": dominance of f = g + 10 over a box on the surface.
    let region = vec![
        Bound::new(2.0, 3.0),
        Bound::new(-2.5, -1.5),
        Bound::new(-0.5, 0.5),
    ];
    let dominant = plus_constant(build(), 10.0);
    println!(
        "f = g + 10 vs g: {:?}",
        verify_against_ground_truth(&dominant, &g, Relation::Dominates, &region)?
    );

    // Falsified direction: f2 = g - 10 with a concrete witness.
    let violating = plus_constant(build(), -10.0);
    match verify_against_ground_truth(&violating, &g, Relation::Dominates, &region)? {
        GroundTruthOutcome::Falsified {
            witness,
            difference,
        } => println!("f2 = g - 10 vs g: Falsified at {witness:?}, f2 - g in {difference:?}"),
        other => println!("f2 = g - 10 vs g: unexpected {other:?}"),
    }

    // Plan §2.3 contract: constants that would have to be rounded are
    // rejected, never silently rounded.
    println!(
        "cylinder with r = 0.1: {}",
        cylinder_residual(axis, point, 0.1).unwrap_err()
    );
    Ok(())
}
