// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::super::common::{
    assert_concrete_contained_in_bounds, assert_finite_and_ordered, evaluate_graph_at_center,
};
use super::alpha::{
    alpha_crown_config_for_stage, alpha_crown_stage_deadline_budget,
    refreshed_alpha_crown_stage_config, run_ecapa_stage_local_alpha_crown,
};
use super::core::{
    extract_ecapa_stage_graphs, output_bounds_from_crown_result, run_monolithic_ibp_cosine_bounds,
    stage_provenance, width_reduction_pct, EcapaStageResult,
};
use super::cosine::{
    run_ecapa_alpha_compositional_cosine_bounds, run_ecapa_compositional_cosine_bounds,
};
use super::stage_local::{collect_ecapa_stage_local_crown_ibp, run_ecapa_stage_local_crown_ibp};
use super::*;
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

mod alpha;
mod baseline;
mod diagnostics;
mod gpu_measure;
mod support;
