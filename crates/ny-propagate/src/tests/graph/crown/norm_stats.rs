// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for graph-level normalization validation stats.

use ndarray::{arr1, arr2, ArrayD};
use ny_core::Result;
use ny_tensor::BoundedTensor;

/// Assert two bound arrays match element-wise within an absolute tolerance.
///
/// These regression tests compare two *mathematically equivalent* CROWN paths
/// (graph dispatch vs. the direct decomposed helper for #3892; AdaIN decomposed
/// path vs. its effective-InstanceNorm equivalent for #3912). The paths agree to
/// within f32 rounding but differ by a few ULP because they accumulate the same
/// reductions in a different order, so bit-exact `assert_eq!` is too strict. A
/// tight 1e-4 absolute tolerance still detects any genuine path divergence
/// (which would be orders of magnitude larger) while tolerating ULP-level noise.
fn assert_bounds_close(actual: &ArrayD<f32>, expected: &ArrayD<f32>, context: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{context}: bound shapes differ ({:?} vs {:?})",
        actual.shape(),
        expected.shape(),
    );
    const TOL: f32 = 1e-4;
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() <= TOL,
            "{context}: bound element diverged ({a} vs {e}, diff={:.2e}, tol={TOL})\n\
             actual={actual:?}\nexpected={expected:?}",
            (a - e).abs(),
        );
    }
}

use crate::bounds::BatchedLinearBounds;
use crate::layers::normalization::decomposed::decomposed_rms_norm_crown_backward;
use crate::layers::normalization::LayerNormCrownMode;
use crate::layers::{AdaIN1dLayer, InstanceNorm1dLayer, RmsNormLayer};
use crate::*;

/// `#3892` regression: graph CROWN-with-stats must surface the row-validation
/// payload from the decomposed RmsNorm helper instead of silently dropping it.
#[ntest::timeout(10000)]
#[test]
fn test_crown_within_graph_with_stats_surfaces_rmsnorm_validation_3892() -> Result<()> {
    // Pin NY_DENSE_BUDGET_MB (holding the shared env lock): within-block CROWN
    // reads the budget per call, and a concurrently-running zero-budget test's
    // window would otherwise fail this test with a spurious CpuMemoryExceeded.
    tests::with_crown_dense_budget_mb("2048", || {
        test_crown_within_graph_with_stats_surfaces_rmsnorm_validation_3892_body()
    })
}

fn test_crown_within_graph_with_stats_surfaces_rmsnorm_validation_3892_body() -> Result<()> {
    let ny = arr1(&[1.5_f32, -0.75, 0.25]);
    let eps = 1e-5_f32;

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "rmsnorm",
        Layer::RmsNorm(RmsNormLayer::new(ny.clone(), eps)?),
    ));
    graph.set_output("rmsnorm");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.25, 0.5]).into_dyn(),
        arr1(&[0.5_f32, 1.5, 2.0]).into_dyn(),
    )?;

    let (graph_bounds, graph_stats) = graph.propagate_crown_within_graph_with_stats(&input)?;

    let helper = decomposed_rms_norm_crown_backward(
        &BatchedLinearBounds::identity(&[3])?,
        &ny,
        eps,
        &input,
    )?;
    let helper_bounds = helper.bounds.concretize_sound(&input)?;

    assert_bounds_close(
        graph_bounds.lower(),
        helper_bounds.lower(),
        "#3892 regression: graph CROWN lower bounds diverged from direct RmsNorm helper",
    );
    assert_bounds_close(
        graph_bounds.upper(),
        helper_bounds.upper(),
        "#3892 regression: graph CROWN upper bounds diverged from direct RmsNorm helper",
    );

    assert_eq!(
        graph_stats.len(),
        1,
        "#3892 regression: expected exactly one normalization stat entry for a single RmsNorm node",
    );
    let stat = &graph_stats[0];
    assert_eq!(
        stat.node_name, "rmsnorm",
        "#3892 regression: RmsNorm stat should be keyed by the graph node name",
    );
    assert_eq!(
        stat.fallback_rows, helper.validation.fallback_rows,
        "#3892 regression: graph CROWN dropped the helper's RmsNorm fallback-row count",
    );
    assert_eq!(
        stat.total_rows, helper.validation.total_rows,
        "#3892 regression: graph CROWN dropped the helper's RmsNorm total-row count",
    );

    Ok(())
}

fn make_adain_and_effective_graphs_3912() -> Result<(GraphNetwork, GraphNetwork)> {
    let adain = AdaIN1dLayer::new(
        InstanceNorm1dLayer::new(arr1(&[1.5_f32, -0.75]), arr1(&[0.1_f32, -0.25]), 1e-5)?,
        arr1(&[0.5_f32, -1.2]),
        arr1(&[0.2_f32, 0.4]),
    )?
    .with_crown_mode(LayerNormCrownMode::IbpValidated);
    let effective = adain.effective_instance_norm()?;

    let mut adain_graph = GraphNetwork::new();
    adain_graph.add_node(GraphNode::from_input("adain", Layer::AdaIN1d(adain)));
    adain_graph.set_output("adain");

    let mut effective_graph = GraphNetwork::new();
    effective_graph.add_node(GraphNode::from_input(
        "inst",
        Layer::InstanceNorm1d(effective),
    ));
    effective_graph.set_output("inst");

    Ok((adain_graph, effective_graph))
}

fn adain_graph_input_3912() -> Result<BoundedTensor> {
    BoundedTensor::new(
        arr2(&[[-1.0_f32, 0.25, 0.5], [-0.75, 0.0, 1.0]]).into_dyn(),
        arr2(&[[0.5_f32, 1.5, 2.0], [0.25, 1.0, 2.5]]).into_dyn(),
    )
}

/// `#3912` regression: block-wise graph CROWN should route AdaIN through the
/// same decomposed validation path as its effective InstanceNorm equivalent
/// instead of silently partial-falling back to plain IBP.
#[ntest::timeout(10000)]
#[test]
fn test_crown_within_graph_with_stats_adain_matches_effective_instance_norm_3912() -> Result<()> {
    let (adain_graph, effective_graph) = make_adain_and_effective_graphs_3912()?;
    let input = adain_graph_input_3912()?;

    // Pin NY_DENSE_BUDGET_MB (holding the shared env lock): within-block CROWN
    // reads the budget per call, and a concurrently-running zero-budget test's
    // window would otherwise fail this test with a spurious CpuMemoryExceeded.
    let (adain_bounds, adain_stats, effective_bounds, effective_stats) =
        tests::with_crown_dense_budget_mb("2048", || -> Result<_> {
            let (adain_bounds, adain_stats) =
                adain_graph.propagate_crown_within_graph_with_stats(&input)?;
            let (effective_bounds, effective_stats) =
                effective_graph.propagate_crown_within_graph_with_stats(&input)?;
            Ok((adain_bounds, adain_stats, effective_bounds, effective_stats))
        })?;

    assert_bounds_close(
        adain_bounds.lower(),
        effective_bounds.lower(),
        "#3912 regression: AdaIN graph lower bounds diverged from effective InstanceNorm graph bounds",
    );
    assert_bounds_close(
        adain_bounds.upper(),
        effective_bounds.upper(),
        "#3912 regression: AdaIN graph upper bounds diverged from effective InstanceNorm graph bounds",
    );

    assert_eq!(
        adain_stats.len(),
        1,
        "#3912 regression: expected exactly one AdaIN stat entry in a single-node graph",
    );
    assert_eq!(
        adain_stats[0].node_name, "adain",
        "#3912 regression: AdaIN stat should be keyed by the graph node name",
    );
    assert_eq!(
        effective_stats.len(),
        1,
        "#3912 regression: expected exactly one InstanceNorm stat entry in the effective graph",
    );
    assert_eq!(
        adain_stats[0].fallback_rows, effective_stats[0].fallback_rows,
        "#3912 regression: AdaIN graph dropped the effective InstanceNorm fallback-row count",
    );
    assert_eq!(
        adain_stats[0].total_rows, effective_stats[0].total_rows,
        "#3912 regression: AdaIN graph dropped the effective InstanceNorm total-row count",
    );

    Ok(())
}
