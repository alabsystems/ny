// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Receipt-bearing, diagnostic-only complete-cover pilot for the exact
//! `cGAN_imgSz32_nCh_1` constrained-zonotope lane.
//!
//! The production depth-two replay remains disabled and is reported as such;
//! the dormant implementation is not exercised by this pilot.

#![deny(unsafe_code)]

#[path = "../src/commands/cgan_status.rs"]
mod cgan_status;
#[path = "../src/commands/cz_cgan_sequential_unwired.rs"]
#[allow(dead_code)]
mod cz_cgan_sequential_unwired;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cgan_status::CGAN_DEPTH_TWO_PRODUCTION_MODE;
use cz_cgan_sequential_unwired::{
    cgan_nch1_generator_discriminator_handoff_qualification_limits,
    probe_cgan_nch1_protected_alpha_plan_unwired, CganCzM17CandidateTelemetry,
    CganCzM24Measurement, CganCzProtectedAlphaCoverLimits, CganCzProtectedAlphaProbeLimits,
    CganCzProtectedAlphaProbeStatus,
};
use ny_mip::{ConstrainedZonotopeAlphaBisectionLimits, ConstrainedZonotopeCallBudget};
use ny_onnx::vnnlib::load_vnnlib_with_certified_scalar_moat;
use ny_onnx::{load_onnx_with_config, BatchNormFoldingPolicy, OnnxLoadConfig};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const DEFAULT_PROPERTY: &str = "cGAN_imgSz32_nCh_1_prop_3_input_eps_0.010_output_eps_0.015.vnnlib";
fn benchmark_root() -> PathBuf {
    std::env::var_os("NY_CGAN_NCH1_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../benchmarks/vnncomp2025/benchmarks/cgan_2023")
        })
}

fn parse_env_u64(name: &str, fallback: u64) -> anyhow::Result<u64> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .map_err(|error| anyhow::anyhow!("{name} must be an unsigned integer: {error}")),
        Err(std::env::VarError::NotPresent) => Ok(fallback),
        Err(error) => Err(anyhow::anyhow!("unable to read {name}: {error}")),
    }
}

fn split_axes() -> anyhow::Result<Vec<usize>> {
    let raw = std::env::var("NY_CGAN_COVER_SPLIT_AXES").unwrap_or_else(|_| "0,1,2,3,4".to_string());
    anyhow::ensure!(!raw.is_empty(), "split-axis plan must be nonempty");
    let axes = raw
        .split(',')
        .map(|axis| {
            axis.parse::<usize>()
                .map_err(|error| anyhow::anyhow!("invalid split axis '{axis}': {error}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(
        axes.iter().all(|&axis| axis < 5),
        "split axes must be in 0..5"
    );
    anyhow::ensure!(
        axes.len() < usize::BITS as usize,
        "split-axis plan is too deep"
    );
    Ok(axes)
}

fn sha256(path: &Path) -> anyhow::Result<String> {
    let bytes = fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn canonical_limits(split_levels: usize) -> anyhow::Result<CganCzProtectedAlphaProbeLimits> {
    let leaf_domains = 1_usize
        .checked_shl(u32::try_from(split_levels)?)
        .ok_or_else(|| anyhow::anyhow!("leaf-domain count overflows usize"))?;
    let tree_nodes = leaf_domains
        .checked_mul(2)
        .and_then(|nodes| nodes.checked_sub(1))
        .ok_or_else(|| anyhow::anyhow!("tree-node count overflows usize"))?;
    Ok(CganCzProtectedAlphaProbeLimits {
        sequential: cgan_nch1_generator_discriminator_handoff_qualification_limits(),
        cover: CganCzProtectedAlphaCoverLimits {
            protected_alpha_dim: 5,
            max_split_levels: split_levels,
            max_tree_nodes: tree_nodes,
            max_leaf_domains: leaf_domains,
            bisection: ConstrainedZonotopeAlphaBisectionLimits {
                max_value_dim: 5,
                max_alpha_dim: 5,
                max_generator_nonzeros: 5,
                max_constraint_count: 0,
                max_constraint_elements: 0,
            },
        },
        max_leaf_propagations: leaf_domains,
    })
}

fn limits_receipt(limits: CganCzProtectedAlphaProbeLimits, split_axes: &[usize]) -> Value {
    let sequential = limits.sequential;
    let cover = limits.cover;
    json!({
        "sequential": {
            "max_graph_nodes": sequential.max_graph_nodes,
            "max_graph_edges": sequential.max_graph_edges,
            "max_topology_work_items": sequential.max_topology_work_items,
            "max_parameter_elements": sequential.max_parameter_elements,
            "max_value_dim": sequential.max_value_dim,
            "max_transient_alpha_dim": sequential.max_transient_alpha_dim,
            "retained_alpha_dim": sequential.retained_alpha_dim,
            "max_generator_nonzeros": sequential.max_generator_nonzeros,
            "max_interval_products_per_stage": sequential.max_interval_products_per_stage,
            "max_exact_terms_per_relu": sequential.max_exact_terms_per_relu,
            "max_m17_iterations": sequential.max_m17_iterations,
            "max_m17_search_work": sequential.max_m17_search_work,
        },
        "cover": {
            "split_axes": split_axes,
            "protected_alpha_dim": cover.protected_alpha_dim,
            "max_split_levels": cover.max_split_levels,
            "max_tree_nodes": cover.max_tree_nodes,
            "max_leaf_domains": cover.max_leaf_domains,
            "bisection": {
                "max_value_dim": cover.bisection.max_value_dim,
                "max_alpha_dim": cover.bisection.max_alpha_dim,
                "max_generator_nonzeros": cover.bisection.max_generator_nonzeros,
                "max_constraint_count": cover.bisection.max_constraint_count,
                "max_constraint_elements": cover.bisection.max_constraint_elements,
            },
        },
        "max_leaf_propagations": limits.max_leaf_propagations,
    })
}

fn m17_candidate_receipt(telemetry: CganCzM17CandidateTelemetry) -> Value {
    json!({
        "selected_lower_bound": telemetry.selected_lower_bound,
        "zero_positive_slope_lower_bound": telemetry.zero_positive_slope_lower_bound,
        "upper_endpoint_lower_bound": telemetry.upper_endpoint_lower_bound,
        "canonical_lower_bound": telemetry.canonical_lower_bound,
        "optimized_lower_bound": telemetry.optimized_lower_bound,
        "best_nonoptimized_lower_bound": telemetry.best_nonoptimized_lower_bound,
        "optimized_improvement": telemetry.optimized_improvement,
        "optimizable_slopes": telemetry.optimizable_slopes,
        "candidates_replayed": telemetry.candidates_replayed,
        "iterations_completed": telemetry.iterations_completed,
        "status": format!("{:?}", telemetry.status),
    })
}

fn m24_measurement_receipt(measurement: Option<&CganCzM24Measurement>) -> Value {
    let Some(measurement) = measurement else {
        return json!({ "status": "not_requested" });
    };
    let search_plan = measurement.search_plan.map(|plan| {
        json!({
            "value_dim": plan.value_dim,
            "alpha_dim": plan.alpha_dim,
            "generator_nonzeros": plan.generator_nonzeros,
            "box_variables": plan.box_variables,
            "restarts": plan.restarts,
            "total_iterations": plan.total_iterations,
            "exact_replays": plan.exact_replays,
            "search_work": plan.search_work,
        })
    });
    json!({
        "status": "measured_verdict_neutral",
        "exact_box_cut_lower_bound": measurement.exact_box_cut_lower_bound,
        "counterfactual_lower_bound": measurement.counterfactual_lower_bound,
        "counterfactual_selection": format!("{:?}", measurement.counterfactual_selection),
        "replay_status": format!("{:?}", measurement.replay_status),
        "search_status": format!("{:?}", measurement.search_status),
        "search_plan": search_plan,
        "iterations_completed": measurement.iterations_completed,
        "restarts_completed": measurement.restarts_completed,
        "candidates_scored": measurement.candidates_scored,
        "exact_replays": measurement.exact_replays,
        "optional_budget_error": measurement
            .optional_budget_error
            .as_ref()
            .map(|error| error.to_string()),
    })
}

fn main() -> anyhow::Result<()> {
    let root = benchmark_root();
    let property_name =
        std::env::var("NY_CGAN_NCH1_PROPERTY").unwrap_or_else(|_| DEFAULT_PROPERTY.to_string());
    anyhow::ensure!(
        !property_name.contains('/') && !property_name.contains('\\'),
        "NY_CGAN_NCH1_PROPERTY must be a basename"
    );
    let model_path = root.join("onnx/cGAN_imgSz32_nCh_1.onnx");
    let property_path = root.join("vnnlib").join(property_name);
    anyhow::ensure!(
        model_path.is_file(),
        "missing model {}",
        model_path.display()
    );
    anyhow::ensure!(
        property_path.is_file(),
        "missing property {}",
        property_path.display()
    );

    let deadline_secs = parse_env_u64("NY_CGAN_COVER_DEADLINE_SECS", 300)?;
    let peak_mib = parse_env_u64("NY_CGAN_COVER_PEAK_MIB", 4096)?;
    anyhow::ensure!(deadline_secs > 0, "deadline must be nonzero");
    anyhow::ensure!(peak_mib > 0, "peak-memory limit must be nonzero");
    let max_peak_live_bytes = usize::try_from(peak_mib)
        .ok()
        .and_then(|value| value.checked_mul(1 << 20))
        .ok_or_else(|| anyhow::anyhow!("peak-memory limit does not fit usize"))?;

    let executable = std::env::current_exe()?;
    let model_sha256 = sha256(&model_path)?;
    let property_sha256 = sha256(&property_path)?;
    let executable_sha256 = sha256(&executable)?;
    let split_axes = split_axes()?;
    let mut limits = canonical_limits(split_axes.len())?;
    limits.sequential.max_m17_iterations = usize::try_from(parse_env_u64(
        "NY_CGAN_M17_ITERATIONS",
        u64::try_from(limits.sequential.max_m17_iterations)?,
    )?)?;
    limits.sequential.max_m17_search_work = parse_env_u64(
        "NY_CGAN_M17_MAX_SEARCH_WORK",
        limits.sequential.max_m17_search_work,
    )?;
    let limits_json = limits_receipt(limits, &split_axes);
    let limits_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&limits_json)?));

    let load_config = OnnxLoadConfig::default()
        .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
        .with_raw_float32_initializer_provenance(true);
    let model = load_onnx_with_config(&model_path, &load_config)?;
    let graph = model.to_graph_network()?;
    let (spec, input, moat) = load_vnnlib_with_certified_scalar_moat(&property_path)?;
    anyhow::ensure!(
        spec.num_inputs == 5 && spec.num_outputs == 1,
        "unexpected property boundary"
    );

    let started = Instant::now();
    let budget = ConstrainedZonotopeCallBudget::new(
        started + Duration::from_secs(deadline_secs),
        64 << 20,
        max_peak_live_bytes,
    );
    let report = probe_cgan_nch1_protected_alpha_plan_unwired(
        &model,
        &graph,
        &input,
        moat,
        &split_axes,
        limits,
        budget,
    );
    let elapsed_seconds = started.elapsed().as_secs_f64();

    let status = match &report.status {
        CganCzProtectedAlphaProbeStatus::Completed(completed) => {
            let leaves = completed
                .leaf_completions
                .iter()
                .map(|leaf| {
                    json!({
                        "leaf_index": leaf.leaf_index,
                        "lower_bound": leaf.bounds.lower_bound,
                        "upper_bound": leaf.bounds.upper_bound,
                        "separates_unsafe_moat": leaf.bounds.separates_unsafe_moat,
                        "bn_tail_correction_upper": leaf.bounds.bn_tail_correction_upper,
                        "lower_m17_status": format!("{:?}", leaf.bounds.lower_m17_status),
                        "upper_m17_status": format!("{:?}", leaf.bounds.upper_m17_status),
                        "lower_m17_candidates": m17_candidate_receipt(
                            leaf.bounds.lower_m17_candidates,
                        ),
                        "negated_upper_m17_candidates": m17_candidate_receipt(
                            leaf.bounds.negated_upper_m17_candidates,
                        ),
                        "lower_m20_lower_bound": leaf.bounds.lower_m20_lower_bound,
                        "negated_upper_m20_lower_bound": leaf.bounds.negated_upper_m20_lower_bound,
                        "lower_m20_status": format!("{:?}", leaf.bounds.lower_m20_status),
                        "negated_upper_m20_status": format!(
                            "{:?}", leaf.bounds.negated_upper_m20_status,
                        ),
                        "lower_m24_measurement": m24_measurement_receipt(
                            leaf.bounds.lower_m24_measurement.as_ref(),
                        ),
                        "negated_upper_m24_measurement": m24_measurement_receipt(
                            leaf.bounds.negated_upper_m24_measurement.as_ref(),
                        ),
                        "completed_stages": leaf.completed_stages,
                        "peak_live_bytes": leaf.peak_live_bytes,
                        "charged_items": leaf.charged_items,
                        "deadline_polls": leaf.deadline_polls,
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "kind": "completed_diagnostic_only",
                "lower_bound": completed.lower_bound,
                "upper_bound": completed.upper_bound,
                "low_unsafe_threshold": completed.low_unsafe_threshold,
                "high_unsafe_threshold": completed.high_unsafe_threshold,
                "separates_unsafe_moat": completed.separates_unsafe_moat,
                "cover": {
                    "split_levels": completed.cover.split_levels(),
                    "tree_nodes": completed.cover.tree_nodes(),
                    "split_calls": completed.cover.split_calls(),
                    "leaf_domains": completed.cover.leaf_domains(),
                    "peak_live_bytes": completed.cover.peak_live_bytes(),
                    "charged_items": completed.cover.charged_items(),
                    "deadline_polls": completed.cover.deadline_polls(),
                },
                "leaves": leaves,
            })
        }
        CganCzProtectedAlphaProbeStatus::Declined {
            leaf_index,
            node,
            reason,
        } => json!({
            "kind": "declined",
            "leaf_index": leaf_index,
            "node": node,
            "reason": reason.to_string(),
        }),
    };

    let receipt = json!({
        "schema": "ny.cgan-cz-protected-cover-pilot.v4",
        "authority": format!("{:?}", report.authority),
        "verdict": "unknown",
        "depth_two": {
            "production_mode": CGAN_DEPTH_TWO_PRODUCTION_MODE,
            "pilot_status": "not_exercised",
            "production_leaf_row_receipt_status": "not_requested",
            "affects_published_bounds": false,
        },
        "model": {
            "path": model_path,
            "sha256": model_sha256,
        },
        "property": {
            "path": property_path,
            "sha256": property_sha256,
        },
        "executable": {
            "path": executable,
            "sha256": executable_sha256,
        },
        "limits": limits_json,
        "limits_sha256": limits_sha256,
        "budget": {
            "deadline_seconds": deadline_secs,
            "baseline_live_bytes": 64 << 20,
            "max_peak_live_bytes": max_peak_live_bytes,
        },
        "elapsed_seconds": elapsed_seconds,
        "report": {
            "topology_work_items": report.topology_work_items,
            "parameter_elements": report.parameter_elements,
            "protected_latent_symbols": report.protected_latent_symbols,
            "requested_leaf_domains": report.requested_leaf_domains,
            "peak_live_bytes": report.peak_live_bytes,
            "charged_items": report.charged_items,
            "deadline_polls": report.deadline_polls,
            "status": status,
        },
    });
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
