// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default-dark M0 planner for a cross-target resident CUDA implicit-Patches
//! root transaction.
//!
//! This increment is intentionally observation-only. It binds a bounded set of
//! demanded Patches targets to exact graph/input/bound transcripts, passes the
//! typed plan to an already-created GPU backend, and emits a kill-gate record.
//! The backend return type contains no coefficients or bounds, and this module
//! never mutates the collector's maps or provenance.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::{
    GemmEngine, GpuResidentPatchesRootObservation, GpuResidentPatchesRootPlan,
    GpuResidentPatchesRootTargetPlan, GPU_RESIDENT_PATCHES_ROOT_MAX_ROWS,
    GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS,
};
use ny_tensor::BoundedTensor;
use sha2::{Digest, Sha256};

use crate::layers::Layer;
use crate::network::core::graph::NETWORK_INPUT;
use crate::network::core::GraphNetwork;

pub(super) const RESIDENT_PATCHES_ROOT_ENV: &str = "NY_CUDA_RESIDENT_PATCHES_ROOT";
const TELEMETRY_MARKER: &str = "NY_CUDA_RESIDENT_PATCHES_ROOT_V1";
const HOST_PLAN_BUDGET: Duration = Duration::from_millis(50);
const FUTURE_TRANSACTION_BUDGET: Duration = Duration::from_secs(2);
const FUTURE_DEVICE_CAP_BYTES: usize = 512 * 1024 * 1024;
const MAX_PRESELECTION_GRAPH_NODES: usize = 4_096;
const MAX_PRESELECTION_BOUND_PAIRS: usize = 4 * 1024 * 1024;
const MAX_PRESELECTION_ANCESTOR_VISITS: usize = 131_072;

#[derive(Clone, Copy, Debug)]
pub(super) struct PlanningWindow {
    host_deadline: Instant,
    transaction_deadline: Instant,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PlanningCounters {
    demand_nodes_scanned: usize,
    bound_pairs_scanned: usize,
    candidate_nodes_scanned: usize,
    ancestor_nodes_scanned: usize,
    patches_candidates: usize,
    candidate_storage_peak: usize,
    preselection_timeouts: usize,
    preselection_cap_declines: usize,
    observer_calls: usize,
    telemetry_lines: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreselectionDecline {
    Deadline,
    Cap,
}

pub(super) fn enabled_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub(super) fn enabled() -> bool {
    enabled_from_raw(std::env::var(RESIDENT_PATCHES_ROOT_ENV).ok().as_deref())
}

/// Start the one absolute planning window before any gate-on demand or target
/// work. Missing/expired collection deadlines fail dark: no observer call and
/// no telemetry line can be attributed to a plan that never had a budget.
pub(super) fn begin_planning_if_enabled(
    armed: bool,
    collection_deadline: Option<Instant>,
) -> Option<PlanningWindow> {
    if !armed {
        return None;
    }
    let collection_deadline = collection_deadline?;
    let started = Instant::now();
    if started >= collection_deadline {
        return None;
    }
    let host_deadline = started
        .checked_add(HOST_PLAN_BUDGET)
        .map_or(collection_deadline, |deadline| {
            deadline.min(collection_deadline)
        });
    let transaction_deadline = started
        .checked_add(FUTURE_TRANSACTION_BUDGET)
        .map_or(collection_deadline, |deadline| {
            deadline.min(collection_deadline)
        });
    (started < host_deadline).then_some(PlanningWindow {
        host_deadline,
        transaction_deadline,
    })
}

fn hash_usize(hasher: &mut Sha256, value: usize) {
    hasher.update(u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hash_usize(hasher, bytes.len());
    hasher.update(bytes);
}

fn hash_array(hasher: &mut Sha256, array: &ndarray::ArrayD<f32>, deadline: Instant) -> bool {
    hash_usize(hasher, array.ndim());
    for &dimension in array.shape() {
        hash_usize(hasher, dimension);
    }
    hash_usize(hasher, array.len());
    for (index, &value) in array.iter().enumerate() {
        if index.is_multiple_of(4096) && Instant::now() >= deadline {
            return false;
        }
        hasher.update(value.to_bits().to_le_bytes());
    }
    true
}

fn hash_bounded_tensor(hasher: &mut Sha256, bounds: &BoundedTensor, deadline: Instant) -> bool {
    if !hash_array(hasher, bounds.lower(), deadline)
        || !hash_array(hasher, bounds.upper(), deadline)
    {
        return false;
    }
    match bounds.l2_constraint() {
        Some(l2) => {
            hasher.update([1]);
            hash_usize(hasher, l2.axis());
            hash_array(hasher, l2.center(), deadline) && hash_array(hasher, l2.radius(), deadline)
        }
        None => {
            hasher.update([0]);
            true
        }
    }
}

fn graph_identity(
    graph: &GraphNetwork,
    exec_order: &[String],
    deadline: Instant,
) -> Option<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"ny.cuda-resident-patches-root.graph.v1\0");
    // CutFoldScope is process-unique across independently built graphs and is
    // retained only by semantic clones; structural mutation mints a new scope.
    // The topology transcript below keeps the identity independently auditable.
    hash_bytes(
        &mut hasher,
        format!("{:?}", graph.cut_fold_scope()).as_bytes(),
    );
    hash_bytes(&mut hasher, graph.output_name().as_bytes());
    hasher.update([u8::from(graph.use_patches_mode)]);
    hash_usize(&mut hasher, exec_order.len());
    for (index, name) in exec_order.iter().enumerate() {
        if index.is_multiple_of(8) && Instant::now() >= deadline {
            return None;
        }
        let node = graph.node(name)?;
        hash_bytes(&mut hasher, name.as_bytes());
        hash_bytes(&mut hasher, node.layer().layer_type().as_bytes());
        hash_usize(&mut hasher, node.inputs().len());
        for input in node.inputs() {
            hash_bytes(&mut hasher, input.as_bytes());
        }
        match graph.declared_shape(name) {
            Some(shape) => {
                hasher.update([1]);
                hash_usize(&mut hasher, shape.len());
                for &dimension in shape {
                    hash_usize(&mut hasher, dimension);
                }
            }
            None => hasher.update([0]),
        }
    }
    Some(hasher.finalize().into())
}

fn input_identity(input: &BoundedTensor, deadline: Instant) -> Option<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"ny.cuda-resident-patches-root.input.v1\0");
    hash_bounded_tensor(&mut hasher, input, deadline).then(|| hasher.finalize().into())
}

fn bounds_identity(
    targets: &[GpuResidentPatchesRootTargetPlan],
    ibp_bounds: &HashMap<String, BoundedTensor>,
    deadline: Instant,
) -> Option<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"ny.cuda-resident-patches-root.bounds.v1\0");
    hash_usize(&mut hasher, targets.len());
    for target in targets {
        if Instant::now() >= deadline {
            return None;
        }
        hash_usize(&mut hasher, target.rank);
        hash_bytes(&mut hasher, target.node_name.as_bytes());
        hash_usize(&mut hasher, target.target_shape.len());
        for &dimension in target.target_shape.iter() {
            hash_usize(&mut hasher, dimension);
        }
        hash_usize(&mut hasher, target.target_rows);
        hash_usize(&mut hasher, target.conv_input_cols);
        hash_usize(&mut hasher, target.dense_pair_bytes);
        hash_usize(&mut hasher, target.bound_endpoint_bytes);
        let bounds = ibp_bounds.get(target.node_name.as_ref())?;
        if !hash_bounded_tensor(&mut hasher, bounds, deadline) {
            return None;
        }
    }
    Some(hasher.finalize().into())
}

#[derive(Clone, Copy)]
struct Candidate<'a> {
    layer_index: usize,
    node_name: &'a str,
    target_shape: &'a [usize],
    target_rows: usize,
    conv_input_cols: usize,
    dense_pair_bytes: usize,
    bound_endpoint_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SelectionStats {
    demanded_targets: usize,
    omitted_target_cap: usize,
    omitted_row_cap: usize,
}

fn poll_preselection(deadline: Instant) -> Result<(), PreselectionDecline> {
    if Instant::now() >= deadline {
        Err(PreselectionDecline::Deadline)
    } else {
        Ok(())
    }
}

fn record_decline(counters: &mut PlanningCounters, decline: PreselectionDecline) {
    match decline {
        PreselectionDecline::Deadline => counters.preselection_timeouts += 1,
        PreselectionDecline::Cap => counters.preselection_cap_declines += 1,
    }
}

fn bounded_is_concrete(
    bounds: &BoundedTensor,
    deadline: Instant,
    counters: &mut PlanningCounters,
) -> Result<bool, PreselectionDecline> {
    for (index, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        if counters.bound_pairs_scanned >= MAX_PRESELECTION_BOUND_PAIRS {
            return Err(PreselectionDecline::Cap);
        }
        counters.bound_pairs_scanned += 1;
        if index.is_multiple_of(1_024) {
            poll_preselection(deadline)?;
        }
        if lower != upper {
            return Ok(false);
        }
    }
    poll_preselection(deadline)?;
    Ok(true)
}

/// Deadline-aware equivalent of `nodes_requiring_crown_tightening` for this
/// observation only. Borrowed names and a hard graph cap bound storage; the
/// ordinary collector still computes its unchanged demand set later.
fn bounded_demand_set<'a>(
    graph: &'a GraphNetwork,
    exec_order: &'a [String],
    ibp_bounds: &HashMap<String, BoundedTensor>,
    deadline: Instant,
    counters: &mut PlanningCounters,
) -> Result<HashSet<&'a str>, PreselectionDecline> {
    poll_preselection(deadline)?;
    if exec_order.len() > MAX_PRESELECTION_GRAPH_NODES {
        return Err(PreselectionDecline::Cap);
    }

    let mut needs_bounds = HashSet::new();
    let output_name = if graph.output_name().is_empty() {
        exec_order.last().map(String::as_str)
    } else {
        Some(graph.output_name())
    };
    if let Some(output_name) = output_name.filter(|name| *name != NETWORK_INPUT) {
        needs_bounds.insert(output_name);
    }

    for node_name in exec_order {
        poll_preselection(deadline)?;
        counters.demand_nodes_scanned += 1;
        let Some(node) = graph.node(node_name) else {
            continue;
        };
        for &index in node.layer().required_input_bound_indices() {
            poll_preselection(deadline)?;
            let Some(input_name) = node.inputs().get(index) else {
                continue;
            };
            if input_name == NETWORK_INPUT {
                continue;
            }
            if let Some(bounds) = ibp_bounds.get(input_name) {
                if bounded_is_concrete(bounds, deadline, counters)? {
                    continue;
                }
            }
            if !needs_bounds.contains(input_name.as_str())
                && needs_bounds.len() >= MAX_PRESELECTION_GRAPH_NODES
            {
                return Err(PreselectionDecline::Cap);
            }
            needs_bounds.insert(input_name.as_str());
        }
    }
    poll_preselection(deadline)?;
    Ok(needs_bounds)
}

fn count_ancestor_scan(counters: &mut PlanningCounters) -> Result<(), PreselectionDecline> {
    if counters.ancestor_nodes_scanned >= MAX_PRESELECTION_ANCESTOR_VISITS {
        return Err(PreselectionDecline::Cap);
    }
    counters.ancestor_nodes_scanned += 1;
    Ok(())
}

/// Find the same deepest eligible convolution geometry used by production,
/// without populating the graph-wide ancestor cache. Both the discovered set
/// and the work performed across all candidates are hard-capped.
fn bounded_conv_ancestor_input_cols<'a>(
    graph: &'a GraphNetwork,
    exec_order: &'a [String],
    target: &'a str,
    deadline: Instant,
    counters: &mut PlanningCounters,
) -> Result<Option<usize>, PreselectionDecline> {
    poll_preselection(deadline)?;
    if graph.node(target).is_none() {
        return Ok(None);
    }

    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    visited.insert(target);
    stack.push(target);
    while let Some(name) = stack.pop() {
        poll_preselection(deadline)?;
        count_ancestor_scan(counters)?;
        let Some(node) = graph.node(name) else {
            return Ok(None);
        };
        for input in node.inputs() {
            if input == NETWORK_INPUT || visited.contains(input.as_str()) {
                continue;
            }
            if visited.len() >= MAX_PRESELECTION_GRAPH_NODES {
                return Err(PreselectionDecline::Cap);
            }
            visited.insert(input.as_str());
            stack.push(input.as_str());
        }
    }

    // Production chooses the first eligible convolution in topological order.
    for name in exec_order {
        poll_preselection(deadline)?;
        count_ancestor_scan(counters)?;
        if !visited.contains(name.as_str()) {
            continue;
        }
        let Some(node) = graph.node(name) else {
            return Ok(None);
        };
        match node.layer() {
            Layer::Conv2d(conv) => {
                let Some((input_h, input_w)) = conv.input_shape else {
                    return Ok(None);
                };
                return Ok(conv
                    .in_channels()
                    .checked_mul(input_h)
                    .and_then(|channels_height| channels_height.checked_mul(input_w)));
            }
            Layer::ConvTranspose2d(conv) if conv.stride == (1, 1) => {
                let Some((input_h, input_w)) = conv.input_shape else {
                    return Ok(None);
                };
                return Ok(conv
                    .in_channels()
                    .checked_mul(input_h)
                    .and_then(|channels_height| channels_height.checked_mul(input_w)));
            }
            _ => {}
        }
    }
    Ok(None)
}

/// Scalar remainder of the production Patches admission after a bounded walk
/// has established one eligible convolution and its exact input width.
fn resident_admitted_with_conv_input(
    graph: &GraphNetwork,
    node_name: &str,
    bounds: &BoundedTensor,
    conv_input_cols: usize,
) -> bool {
    let dense_budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    let target_rows = bounds.len();
    let dense_identity_exceeds_budget =
        crate::network::crown_memory::identity_pair_bytes(target_rows)
            .map_or(true, |required| required > dense_budget);
    let dense_pair_bytes =
        crate::network::crown_memory::dense_pair_bytes(target_rows, conv_input_cols);
    let dense_backward_exceeds_budget =
        dense_pair_bytes.is_some_and(|required| required > dense_budget);
    let dense_backward_cost_prefers_patches = target_rows
        >= crate::network::core::graph::backward_helpers::patches_reentry_min_rows()
        && dense_pair_bytes.is_some_and(|required| {
            required > super::patches_target::patches_cost_admission_threshold_bytes()
        });

    (dense_identity_exceeds_budget
        || dense_backward_exceeds_budget
        || dense_backward_cost_prefers_patches)
        && bounds.shape().len() == 3
        && graph.node(node_name).is_some_and(
            crate::network::core::graph::backward_helpers::node_admits_patches_backward_step,
        )
}

fn compare_candidates(left: &Candidate<'_>, right: &Candidate<'_>) -> std::cmp::Ordering {
    right
        .dense_pair_bytes
        .cmp(&left.dense_pair_bytes)
        .then_with(|| left.layer_index.cmp(&right.layer_index))
        .then_with(|| left.node_name.cmp(right.node_name))
}

fn select_targets<'a>(
    graph: &'a GraphNetwork,
    exec_order: &'a [String],
    demand_set: &HashSet<&str>,
    ibp_bounds: &'a HashMap<String, BoundedTensor>,
    deadline: Instant,
    counters: &mut PlanningCounters,
) -> Result<(Vec<GpuResidentPatchesRootTargetPlan>, SelectionStats), PreselectionDecline> {
    poll_preselection(deadline)?;
    if exec_order.len() > MAX_PRESELECTION_GRAPH_NODES {
        return Err(PreselectionDecline::Cap);
    }
    let mut stats = SelectionStats {
        demanded_targets: demand_set.len(),
        ..SelectionStats::default()
    };
    // This Vec can never grow past the public transaction target cap. Candidate
    // payloads borrow graph/bound metadata; only the final selected plans clone
    // names/shapes.
    let mut candidates: Vec<Candidate<'a>> =
        Vec::with_capacity(GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS);
    for (layer_index, node_name) in exec_order.iter().enumerate() {
        poll_preselection(deadline)?;
        counters.candidate_nodes_scanned += 1;
        if !demand_set.contains(node_name.as_str()) {
            continue;
        }
        let Some(bounds) = ibp_bounds.get(node_name) else {
            continue;
        };
        if bounds.shape().len() != 3
            || !graph.node(node_name).is_some_and(
                crate::network::core::graph::backward_helpers::node_admits_patches_backward_step,
            )
        {
            continue;
        }
        let Some(conv_input_cols) =
            bounded_conv_ancestor_input_cols(graph, exec_order, node_name, deadline, counters)?
        else {
            continue;
        };
        poll_preselection(deadline)?;
        if !resident_admitted_with_conv_input(graph, node_name, bounds, conv_input_cols) {
            continue;
        }
        let target_rows = bounds.len();
        let Some(dense_pair_bytes) =
            crate::network::crown_memory::dense_pair_bytes(target_rows, conv_input_cols)
        else {
            continue;
        };
        let Some(bound_endpoint_bytes) = target_rows.checked_mul(2 * size_of::<f32>()) else {
            continue;
        };
        counters.patches_candidates = counters.patches_candidates.saturating_add(1);
        let candidate = Candidate {
            layer_index,
            node_name,
            target_shape: bounds.shape(),
            target_rows,
            conv_input_cols,
            dense_pair_bytes,
            bound_endpoint_bytes,
        };
        if candidates.len() < GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS {
            candidates.push(candidate);
            counters.candidate_storage_peak = counters.candidate_storage_peak.max(candidates.len());
            continue;
        }
        let worst_index = candidates
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| compare_candidates(left, right))
            .map(|(index, _)| index)
            .expect("the bounded top-K is full");
        if compare_candidates(&candidate, &candidates[worst_index]).is_lt() {
            candidates[worst_index] = candidate;
        }
    }
    poll_preselection(deadline)?;
    stats.omitted_target_cap = counters.patches_candidates.saturating_sub(candidates.len());
    // Prioritize the largest dense pair avoided; source/topological order is
    // the deterministic tie-break. This makes the prop1761 record answer the
    // key M1 question: whether enough high-cost targets share one root plan.
    candidates.sort_by(compare_candidates);

    let mut selected = Vec::with_capacity(GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS);
    let mut total_rows = 0usize;
    for candidate in candidates {
        poll_preselection(deadline)?;
        let Some(next_rows) = total_rows.checked_add(candidate.target_rows) else {
            stats.omitted_row_cap += 1;
            continue;
        };
        if next_rows > GPU_RESIDENT_PATCHES_ROOT_MAX_ROWS {
            stats.omitted_row_cap += 1;
            continue;
        }
        let rank = selected.len();
        selected.push(GpuResidentPatchesRootTargetPlan {
            rank,
            node_name: Arc::from(candidate.node_name),
            target_shape: Arc::from(candidate.target_shape),
            target_rows: candidate.target_rows,
            conv_input_cols: candidate.conv_input_cols,
            dense_pair_bytes: candidate.dense_pair_bytes,
            bound_endpoint_bytes: candidate.bound_endpoint_bytes,
        });
        total_rows = next_rows;
    }
    poll_preselection(deadline)?;
    Ok((selected, stats))
}

fn hex(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(TABLE[usize::from(byte >> 4)] as char);
        encoded.push(TABLE[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn emit_declined(reason: &str, counters: &mut PlanningCounters) {
    eprintln!(
        "{TELEMETRY_MARKER} stage=declined reason_hex={} authority=false \
         device_allocations=0 cuda_dispatches=0 bounds_published=0 verdict_mutations=0",
        hex(reason.as_bytes()),
    );
    counters.telemetry_lines += 1;
}

fn observation_is_valid(
    observation: GpuResidentPatchesRootObservation,
    plan: &GpuResidentPatchesRootPlan,
) -> bool {
    observation.is_zero_authority()
        && if observation.backend_ready {
            observation.accepted_targets == plan.targets.len()
                && observation.accepted_rows == plan.total_rows()
        } else {
            observation.accepted_targets == 0 && observation.accepted_rows == 0
        }
}

/// Attempt one observation-only plan inside the absolute window created at
/// the gate site. The return value is counter telemetry for focused tests only;
/// it cannot carry coefficients, bounds, or verdict state.
pub(super) fn observe_in_window(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    ibp_bounds: &HashMap<String, BoundedTensor>,
    exec_order: &[String],
    engine: Option<&dyn GemmEngine>,
    window: PlanningWindow,
) -> PlanningCounters {
    let mut counters = PlanningCounters::default();
    let demand_set = match bounded_demand_set(
        graph,
        exec_order,
        ibp_bounds,
        window.host_deadline,
        &mut counters,
    ) {
        Ok(demand_set) => demand_set,
        Err(decline) => {
            record_decline(&mut counters, decline);
            return counters;
        }
    };
    let (targets, stats) = match select_targets(
        graph,
        exec_order,
        &demand_set,
        ibp_bounds,
        window.host_deadline,
        &mut counters,
    ) {
        Ok(selection) => selection,
        Err(decline) => {
            record_decline(&mut counters, decline);
            return counters;
        }
    };
    if targets.is_empty() {
        return counters;
    }
    let Some(graph_identity_sha256) = graph_identity(graph, exec_order, window.host_deadline)
    else {
        if Instant::now() >= window.host_deadline {
            counters.preselection_timeouts += 1;
        }
        return counters;
    };
    let Some(input_identity_sha256) = input_identity(input, window.host_deadline) else {
        if Instant::now() >= window.host_deadline {
            counters.preselection_timeouts += 1;
        }
        return counters;
    };
    let Some(bounds_identity_sha256) = bounds_identity(&targets, ibp_bounds, window.host_deadline)
    else {
        if Instant::now() >= window.host_deadline {
            counters.preselection_timeouts += 1;
        }
        return counters;
    };
    let plan = GpuResidentPatchesRootPlan {
        graph_identity_sha256,
        input_identity_sha256,
        bounds_identity_sha256,
        targets: Arc::from(targets),
        deadline: window.transaction_deadline,
        max_device_bytes: FUTURE_DEVICE_CAP_BYTES,
    };
    if plan.validate(Instant::now()).is_err() {
        return counters;
    }
    if let Err(decline) = poll_preselection(window.host_deadline) {
        record_decline(&mut counters, decline);
        return counters;
    }

    let Some(engine) = engine else {
        emit_declined("no_existing_engine", &mut counters);
        return counters;
    };
    let Some(backend) = engine.as_gpu_crown_backward() else {
        emit_declined("engine_has_no_gpu_crown_capability", &mut counters);
        return counters;
    };
    if !backend.provides_resident_patches_root_observer() {
        emit_declined("backend_has_no_resident_patches_observer", &mut counters);
        return counters;
    }
    if let Err(decline) = poll_preselection(window.host_deadline) {
        record_decline(&mut counters, decline);
        return counters;
    }
    counters.observer_calls += 1;
    match backend.observe_resident_patches_root_plan(&plan) {
        Ok(observation) if observation_is_valid(observation, &plan) => {
            let deadline_ms = plan
                .deadline
                .saturating_duration_since(Instant::now())
                .as_millis();
            eprintln!(
                "{TELEMETRY_MARKER} stage=plan demanded_targets={} patches_candidates={} \
                 selected_targets={} selected_rows={} omitted_target_cap={} omitted_row_cap={} \
                 candidate_storage_peak={} demand_nodes_scanned={} bound_pairs_scanned={} \
                 candidate_nodes_scanned={} ancestor_nodes_scanned={} \
                 dense_pair_bytes_avoided={} device_cap_bytes={} deadline_ms={} \
                 graph_sha256={} input_sha256={} bounds_sha256={} authority=false \
                 device_allocations=0 cuda_dispatches=0 bounds_published=0 verdict_mutations=0",
                stats.demanded_targets,
                counters.patches_candidates,
                plan.targets.len(),
                plan.total_rows(),
                stats.omitted_target_cap,
                stats.omitted_row_cap,
                counters.candidate_storage_peak,
                counters.demand_nodes_scanned,
                counters.bound_pairs_scanned,
                counters.candidate_nodes_scanned,
                counters.ancestor_nodes_scanned,
                plan.dense_pair_bytes_avoided(),
                plan.max_device_bytes,
                deadline_ms,
                hex(&plan.graph_identity_sha256),
                hex(&plan.input_identity_sha256),
                hex(&plan.bounds_identity_sha256),
            );
            counters.telemetry_lines += 1;
            for target in plan.targets.iter() {
                let shape = target
                    .target_shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                eprintln!(
                    "{TELEMETRY_MARKER} stage=target rank={} node_hex={} shape={} rows={} \
                     conv_input_cols={} dense_pair_bytes={} endpoint_bytes={} authority=false",
                    target.rank,
                    hex(target.node_name.as_bytes()),
                    shape,
                    target.target_rows,
                    target.conv_input_cols,
                    target.dense_pair_bytes,
                    target.bound_endpoint_bytes,
                );
                counters.telemetry_lines += 1;
            }
            eprintln!(
                "{TELEMETRY_MARKER} stage=backend backend_hex={} backend_ready={} \
                 accepted_targets={} accepted_rows={} device_allocations={} \
                 cuda_dispatches={} bounds_published={} verdict_mutations={} authority=false",
                hex(engine.backend_provenance().as_bytes()),
                observation.backend_ready,
                observation.accepted_targets,
                observation.accepted_rows,
                observation.device_allocations,
                observation.cuda_dispatches,
                observation.bound_values_published,
                observation.verdict_mutations,
            );
            counters.telemetry_lines += 1;
        }
        Ok(_) => emit_declined("backend_observation_contract_violation", &mut counters),
        Err(error) => emit_declined(&format!("backend_error:{error}"), &mut counters),
    }
    counters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Conv2dLayer, Layer, ReLULayer};
    use crate::network::core::GraphNode;
    use ndarray::{Array1, ArrayD, IxDyn};
    use ny_core::{GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NyError, Result};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn cifar_geometry_graph_and_bounds() -> (
        GraphNetwork,
        BoundedTensor,
        HashMap<String, BoundedTensor>,
        Vec<String>,
    ) {
        let conv1 = Conv2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[16, 3, 3, 3]), 0.01),
            Some(Array1::zeros(16)),
            (1, 1),
            (0, 0),
            32,
            32,
        )
        .unwrap();
        let conv2 = Conv2dLayer::with_input_shape(
            ArrayD::from_elem(IxDyn(&[16, 16, 1, 1]), 0.01),
            Some(Array1::zeros(16)),
            (1, 1),
            (0, 0),
            30,
            30,
        )
        .unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["conv1".into()],
        ));
        graph.add_node(GraphNode::new(
            "conv2",
            Layer::Conv2d(conv2),
            vec!["relu1".into()],
        ));
        graph.set_output("conv2");

        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[3, 32, 32]), -1.0),
            ArrayD::from_elem(IxDyn(&[3, 32, 32]), 1.0),
        )
        .unwrap();
        let spatial = || {
            BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[16, 30, 30]), -1.0),
                ArrayD::from_elem(IxDyn(&[16, 30, 30]), 1.0),
            )
            .unwrap()
        };
        let bounds = HashMap::from([
            ("conv1".to_string(), spatial()),
            ("relu1".to_string(), spatial()),
            ("conv2".to_string(), spatial()),
        ]);
        let order = graph.exec_order().unwrap().to_vec();
        (graph, input, bounds, order)
    }

    fn many_candidate_graph_and_bounds(
        count: usize,
    ) -> (GraphNetwork, HashMap<String, BoundedTensor>, Vec<String>) {
        let mut graph = GraphNetwork::new();
        let mut bounds = HashMap::new();
        for index in 0..count {
            let name = format!("conv{index:02}");
            let conv = Conv2dLayer::with_input_shape(
                ArrayD::from_elem(IxDyn(&[16, 3, 3, 3]), 0.01),
                Some(Array1::zeros(16)),
                (1, 1),
                (0, 0),
                32,
                32,
            )
            .unwrap();
            graph.add_node(GraphNode::from_input(name.clone(), Layer::Conv2d(conv)));
            bounds.insert(
                name,
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[16, 30, 30]), -1.0),
                    ArrayD::from_elem(IxDyn(&[16, 30, 30]), 1.0),
                )
                .unwrap(),
            );
        }
        graph.set_output(format!("conv{:02}", count - 1));
        let order = graph.exec_order().unwrap().to_vec();
        (graph, bounds, order)
    }

    #[derive(Default)]
    struct Observer {
        calls: AtomicUsize,
        last: Mutex<Option<GpuResidentPatchesRootPlan>>,
    }

    impl GemmEngine for Observer {
        fn backend_provenance(&self) -> &'static str {
            "test-observer"
        }

        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("test observer has no GEMM".into()))
        }

        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }
    }

    impl GpuCrownBackward for Observer {
        fn provides_resident_patches_root_observer(&self) -> bool {
            true
        }

        fn observe_resident_patches_root_plan(
            &self,
            plan: &GpuResidentPatchesRootPlan,
        ) -> Result<GpuResidentPatchesRootObservation> {
            plan.validate(Instant::now())?;
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.last.lock().unwrap() = Some(plan.clone());
            Ok(GpuResidentPatchesRootObservation {
                backend_ready: true,
                accepted_targets: plan.targets.len(),
                accepted_rows: plan.total_rows(),
                ..GpuResidentPatchesRootObservation::default()
            })
        }

        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp(
                "test observer has no CROWN kernel".into(),
            ))
        }
    }

    #[test]
    fn gate_is_exact_literal_one() {
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("01"),
            Some(" 1"),
        ] {
            assert!(!enabled_from_raw(raw));
        }
        assert!(enabled_from_raw(Some("1")));
    }

    #[test]
    fn absent_or_zero_gate_performs_no_backend_work() {
        let (_graph, _input, _bounds, _order) = cifar_geometry_graph_and_bounds();
        let observer = Observer::default();
        for raw in [None, Some("0")] {
            assert!(begin_planning_if_enabled(
                enabled_from_raw(raw),
                Some(Instant::now() + Duration::from_secs(1)),
            )
            .is_none());
        }
        assert_eq!(observer.calls.load(Ordering::SeqCst), 0);
        assert!(observer.last.lock().unwrap().is_none());
    }

    #[test]
    fn armed_plan_batches_two_cifar_patches_targets_with_zero_authority() {
        let (graph, input, bounds, order) = cifar_geometry_graph_and_bounds();
        let observer = Observer::default();
        let now = Instant::now();
        let counters = observe_in_window(
            &graph,
            &input,
            &bounds,
            &order,
            Some(&observer),
            PlanningWindow {
                host_deadline: now + Duration::from_secs(1),
                transaction_deadline: now + Duration::from_secs(2),
            },
        );
        assert_eq!(observer.calls.load(Ordering::SeqCst), 1);
        assert_eq!(counters.observer_calls, 1);
        assert_eq!(counters.preselection_timeouts, 0);
        assert_eq!(counters.preselection_cap_declines, 0);
        assert_eq!(counters.candidate_storage_peak, 2);
        let plan = observer.last.lock().unwrap().clone().unwrap();
        assert_eq!(plan.targets.len(), 2);
        assert_eq!(plan.targets[0].target_rows, 14_400);
        assert_eq!(plan.targets[1].target_rows, 14_400);
        assert_eq!(plan.targets[0].conv_input_cols, 3 * 32 * 32);
        assert!(plan.dense_pair_bytes_avoided() > 700_000_000);
        plan.validate(Instant::now()).unwrap();
    }

    #[test]
    fn expired_preselection_is_silent_and_never_calls_observer() {
        let (graph, input, bounds, order) = cifar_geometry_graph_and_bounds();
        let observer = Observer::default();
        let now = Instant::now();
        let counters = observe_in_window(
            &graph,
            &input,
            &bounds,
            &order,
            Some(&observer),
            PlanningWindow {
                host_deadline: now,
                transaction_deadline: now + Duration::from_secs(1),
            },
        );
        assert_eq!(counters.preselection_timeouts, 1);
        assert_eq!(counters.preselection_cap_declines, 0);
        assert_eq!(counters.observer_calls, 0);
        assert_eq!(counters.telemetry_lines, 0);
        assert_eq!(observer.calls.load(Ordering::SeqCst), 0);
        assert!(observer.last.lock().unwrap().is_none());
    }

    #[test]
    fn candidate_storage_is_hard_bounded_top_k() {
        let count = GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS + 4;
        let (graph, bounds, order) = many_candidate_graph_and_bounds(count);
        let demand = order.iter().map(String::as_str).collect::<HashSet<_>>();
        let mut counters = PlanningCounters::default();
        let (targets, stats) = select_targets(
            &graph,
            &order,
            &demand,
            &bounds,
            Instant::now() + Duration::from_secs(1),
            &mut counters,
        )
        .unwrap();

        assert_eq!(counters.patches_candidates, count);
        assert_eq!(
            counters.candidate_storage_peak,
            GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS
        );
        assert_eq!(targets.len(), GPU_RESIDENT_PATCHES_ROOT_MAX_TARGETS);
        assert_eq!(stats.omitted_target_cap, 4);
        assert!(targets.iter().all(|target| {
            graph.crown_ibp_target_can_start_in_patches(
                target.node_name.as_ref(),
                bounds.get(target.node_name.as_ref()).unwrap(),
            )
        }));
    }

    #[test]
    fn exact_input_and_target_endpoint_bits_change_identities() {
        let (graph, input, mut bounds, order) = cifar_geometry_graph_and_bounds();
        let demand = HashSet::from(["conv1", "conv2"]);
        let mut counters = PlanningCounters::default();
        let deadline = Instant::now() + Duration::from_secs(1);
        let (targets, _) =
            select_targets(&graph, &order, &demand, &bounds, deadline, &mut counters).unwrap();
        let input_a = input_identity(&input, deadline).unwrap();
        let bounds_a = bounds_identity(&targets, &bounds, deadline).unwrap();

        let mut changed_input_lower = input.lower().clone();
        changed_input_lower[[0, 0, 0]] =
            f32::from_bits(changed_input_lower[[0, 0, 0]].to_bits() + 1);
        let changed_input = BoundedTensor::new(changed_input_lower, input.upper().clone()).unwrap();
        let changed_input_id = input_identity(&changed_input, deadline).unwrap();
        assert_ne!(input_a, changed_input_id);

        let changed = bounds.get("conv1").unwrap();
        let changed_lower = changed.lower().clone();
        let mut changed_upper = changed.upper().clone();
        changed_upper[[0, 0, 0]] = f32::from_bits(changed_upper[[0, 0, 0]].to_bits() + 1);
        bounds.insert(
            "conv1".to_string(),
            BoundedTensor::new(changed_lower, changed_upper).unwrap(),
        );
        let bounds_b = bounds_identity(&targets, &bounds, deadline).unwrap();
        assert_ne!(bounds_a, bounds_b);
    }
}
