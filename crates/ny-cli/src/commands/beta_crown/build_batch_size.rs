// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_propagate::BetaCrownConfig;
use tracing::info;

const LARGE_GRAPH_INPUT_SPLIT_PARAM_THRESHOLD_NY: usize = 10_000_000;
const LARGE_GRAPH_INPUT_SPLIT_SPEC_THRESHOLD_NY: usize = 1_000;
const LARGE_GRAPH_INPUT_SPLIT_BUILD_BATCH_SIZE_NY: usize = 1_000;

pub(super) fn auto_build_batch_size_override_4354(
    param_count: usize,
    spec_count: usize,
    is_graph_model: bool,
    is_input_split: bool,
) -> Option<usize> {
    (is_graph_model
        && is_input_split
        && param_count > LARGE_GRAPH_INPUT_SPLIT_PARAM_THRESHOLD_NY
        && spec_count > LARGE_GRAPH_INPUT_SPLIT_SPEC_THRESHOLD_NY)
        .then_some(LARGE_GRAPH_INPUT_SPLIT_BUILD_BATCH_SIZE_NY)
}

pub(super) fn maybe_apply_build_batch_size_autotune_4354(
    config: &mut BetaCrownConfig,
    param_count: usize,
    vnnlib_spec: Option<&ny_onnx::vnnlib::VnnLibSpec>,
    is_graph_model: bool,
    is_input_split: bool,
) {
    if config.build_batch_size.is_some() {
        return;
    }

    let Some(spec_count) = vnnlib_spec.map(|spec| spec.output_constraints.len()) else {
        return;
    };
    let Some(build_batch_size) = auto_build_batch_size_override_4354(
        param_count,
        spec_count,
        is_graph_model,
        is_input_split,
    ) else {
        return;
    };

    info!(
        param_count,
        spec_count,
        build_batch_size,
        "Applying alpha-beta-CROWN-style build_batch_size auto-tuning for large graph input-split verification"
    );
    config.build_batch_size = Some(build_batch_size);
}
