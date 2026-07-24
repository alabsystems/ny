// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::boundary::EcapaCompositionBoundary;
use super::*;

/// Concrete-point containment check (#3683): evaluate the encoder at the
/// center of the epsilon ball and verify each stage output falls within
/// the tightened bounds.
pub(super) fn assert_stage_outputs_contain_center(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    boundary: &EcapaCompositionBoundary,
    stage_result: &EcapaStageResult,
) {
    let center = input.center();
    let point_input = BoundedTensor::new(center.clone(), center)
        .expect("center point should form a valid BoundedTensor");
    let concrete_node_bounds = graph
        .collect_node_bounds(&point_input)
        .expect("full encoder IBP at center should succeed");
    for (label, output_name, tightened) in [
        ("x2", &boundary.block_outputs[0], &stage_result.x2_bounds),
        ("x3", &boundary.block_outputs[1], &stage_result.x3_bounds),
        ("x4", &boundary.block_outputs[2], &stage_result.x4_bounds),
    ] {
        let concrete = concrete_node_bounds
            .get(output_name)
            .unwrap_or_else(|| panic!("concrete node bounds missing '{output_name}'"));
        assert_concrete_contained_in_bounds(
            concrete,
            tightened,
            &format!("{label} stage CROWN-IBP containment"),
        );
    }
}
