// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default-dark comprehensive root intermediate tightening on a private CPU pool.
//!
//! Every eligible ReLU pre-activation is computed from one immutable frozen
//! bounds snapshot. The target backwards are independent and execute on a
//! private, at-most-four-thread Rayon pool whose workers hold an explicit
//! CPU-only sound-f64 scope. Results are indexed, validated in canonical graph
//! order, staged against the unchanged live map, and published as one
//! infallible batch. A target refusal, deadline expiry, memory-admission miss,
//! disjoint interval, stale live entry, or malformed result leaves the complete
//! map unchanged.

use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_tensor::{BoundedTensor, L2Constraint};
use rayon::prelude::*;

use crate::network::CrownDispatchPlan;
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

use super::backward_input_relative_bounds_at_node;

const MAX_TARGETS: usize = 10;
const MAX_TARGET_DIM: usize = 16_384;
const MAX_WORKERS: usize = 4;
const MAX_RUNTIME: Duration = Duration::from_mins(2);

/// Conservative coefficient-frontier factor shared with the audited bounded
/// host executor: two 2R×P f64 products (32 bytes/position), four persistent
/// R×P f32 coefficient/error matrices (16), and up to 8 bytes for the stacked
/// source or propagated errors. The remaining 8 bytes are bookkeeping slack.
/// Here `P` is the stronger sum of simultaneously live DAG-frontier columns,
/// not merely the largest single node.
const FRONTIER_BYTES_PER_ELEMENT: usize = 8 * size_of::<f64>();
const PER_ROW_METADATA_BYTES: usize = 32;
/// faer 0.24 aligns owned plain-number matrices to 64 bytes and pads their row
/// capacity to that alignment. These mirror the audited deadline Linear path's
/// allocation geometry; charging only logical weight elements is unsound for a
/// skinny `1×N` matrix whose owned f32 row capacity is sixteen.
const FAER_OWNED_ALIGNMENT_BYTES: usize = 64;
/// Mirrors `crown_single::DEADLINE_AW_FAER_CHUNK_MACS`. The comprehensive
/// planner cannot call that private implementation helper, so it reproduces
/// its checked admission predicate before any target allocation.
const LINEAR_DEADLINE_FAER_CHUNK_MACS: usize = 1 << 24;
/// Mirrors `crown_single::DEADLINE_AW_FAER_MAX_SCRATCH_BYTES`.
const LINEAR_DEADLINE_FAER_MAX_SCRATCH_BYTES: usize = 64 << 20;
/// Convolution dispatch clones the complete layer before shape setup, then the
/// sound CPU path can materialize widened/absolute kernel operands. Charge
/// both directions because they may execute in a Rayon join and own their
/// kernel scratch concurrently.
const CONV_PARAMETER_SCRATCH_BYTES_PER_ELEMENT: usize = 8 * size_of::<f64>();
/// Small layer-owned vectors (biases, BatchNorm affine/error vectors, reshape
/// metadata) are charged at the same deliberately padded element factor.
const VECTOR_PARAMETER_SCRATCH_BYTES_PER_ELEMENT: usize = 4 * size_of::<f64>();

#[derive(Clone, Debug)]
struct TargetPlan {
    index: usize,
    relu_name: String,
    pre_name: String,
    rows: usize,
    max_live_frontier_dim: usize,
    reachable_layer_scratch_bytes: usize,
}

#[derive(Debug)]
struct ComputedTarget {
    index: usize,
    relu_name: String,
    pre_name: String,
    candidate: BoundedTensor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ComprehensiveCpuReport {
    pub(crate) attempted_targets: usize,
    pub(crate) tightened_targets: usize,
    pub(crate) tightened_elements: usize,
}

/// Governed exact-one opt-in through the workspace lever chokepoint. The
/// shipped arm is dark.
pub(crate) fn comprehensive_cpu_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::collection::ROOT_CPU_PARALLEL_INTERM_CROWN)
        .value
        .as_bool()
}

#[inline]
fn past(deadline: Instant) -> bool {
    Instant::now() >= deadline
}

fn l2_exact_eq(left: Option<&L2Constraint>, right: Option<&L2Constraint>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.axis() == right.axis()
                && left.center().shape() == right.center().shape()
                && left.radius().shape() == right.radius().shape()
                && left
                    .center()
                    .iter()
                    .zip(right.center().iter())
                    .all(|(&a, &b)| a.to_bits() == b.to_bits())
                && left
                    .radius()
                    .iter()
                    .zip(right.radius().iter())
                    .all(|(&a, &b)| a.to_bits() == b.to_bits())
        }
        _ => false,
    }
}

fn tensor_exact_eq(left: &BoundedTensor, right: &BoundedTensor) -> bool {
    left.shape() == right.shape()
        && left
            .lower()
            .iter()
            .zip(right.lower().iter())
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
        && left
            .upper()
            .iter()
            .zip(right.upper().iter())
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
        && l2_exact_eq(left.l2_constraint(), right.l2_constraint())
}

fn checked_tensor_bytes(bounds: &BoundedTensor) -> Option<usize> {
    let endpoints = bounds.len().checked_mul(2 * size_of::<f32>())?;
    let l2 = bounds.l2_constraint().map_or(Some(0), |constraint| {
        constraint
            .center()
            .len()
            .checked_add(constraint.radius().len())?
            .checked_mul(size_of::<f32>())
    })?;
    endpoints.checked_add(l2)
}

fn checked_snapshot_bytes(bounds: &HashMap<String, BoundedTensor>) -> Option<usize> {
    bounds.iter().try_fold(0usize, |total, (name, tensor)| {
        total
            .checked_add(name.len())?
            .checked_add(128)? // HashMap bucket/key/Arc allocator overhead.
            .checked_add(checked_tensor_bytes(tensor)?)
    })
}

fn checked_target_peak_bytes(
    rows: usize,
    max_live_frontier_dim: usize,
    reachable_layer_scratch_bytes: usize,
) -> Option<usize> {
    rows.checked_mul(max_live_frontier_dim)?
        .checked_mul(FRONTIER_BYTES_PER_ELEMENT)?
        .checked_add(rows.checked_mul(PER_ROW_METADATA_BYTES)?)
        .and_then(|frontier| frontier.checked_add(reachable_layer_scratch_bytes))
}

fn checked_aggregate_bytes(
    targets: &[TargetPlan],
    bounds: &HashMap<String, BoundedTensor>,
    workers: usize,
) -> Option<usize> {
    if targets.is_empty() || workers == 0 {
        return None;
    }
    let mut peaks = Vec::new();
    peaks.try_reserve_exact(targets.len()).ok()?;
    for target in targets {
        peaks.push(checked_target_peak_bytes(
            target.rows,
            target.max_live_frontier_dim,
            target.reachable_layer_scratch_bytes,
        )?);
    }
    peaks.sort_unstable_by(|left, right| right.cmp(left));
    let concurrent = peaks
        .into_iter()
        .take(workers.min(targets.len()))
        .try_fold(0usize, usize::checked_add)?;

    // Every computed candidate remains live until the batch is complete, and
    // every staged intersection coexists with those candidates immediately
    // before publication. Charge both, plus preserved L2 annotations.
    let retained_results = targets.iter().try_fold(0usize, |total, target| {
        let live = bounds.get(target.pre_name.as_str())?;
        total
            .checked_add(target.rows.checked_mul(2 * size_of::<f32>())?)?
            .checked_add(checked_tensor_bytes(live)?)
    })?;
    concurrent
        .checked_add(checked_snapshot_bytes(bounds)?)?
        .checked_add(retained_results)
}

#[inline]
fn memory_admitted(required_bytes: Option<usize>, budget_bytes: usize) -> bool {
    required_bytes.is_some_and(|required| required <= budget_bytes)
}

fn aggregate_budget_from(
    dense_budget_bytes: usize,
    process_headroom_bytes: Option<u64>,
    workers: usize,
) -> usize {
    let hard_cap = usize::try_from(16_u64 * 1024 * 1024 * 1024).unwrap_or(usize::MAX);
    let worker_budget = dense_budget_bytes.saturating_mul(workers).min(hard_cap);
    process_headroom_bytes.map_or(worker_budget, |headroom| {
        worker_budget.min(usize::try_from(headroom / 2).unwrap_or(usize::MAX))
    })
}

fn admitted_memory_plan(
    targets: &[TargetPlan],
    bounds: &HashMap<String, BoundedTensor>,
    max_workers: usize,
    dense_budget_bytes: usize,
    process_headroom_bytes: Option<u64>,
) -> Option<(usize, usize, usize)> {
    for workers in (1..=max_workers.min(targets.len()).min(MAX_WORKERS)).rev() {
        let Some(required) = checked_aggregate_bytes(targets, bounds, workers) else {
            continue;
        };
        let budget = aggregate_budget_from(dense_budget_bytes, process_headroom_bytes, workers);
        if memory_admitted(Some(required), budget) {
            return Some((workers, required, budget));
        }
    }
    None
}

/// Maximum sum of coefficient-column dimensions that can be live during one
/// structural backward transition. Unlike a maximum single-node dimension,
/// this charges both branches of Add/Sub DAGs and the old/new coefficient
/// fronts that coexist while a layer is dispatched.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TargetResources {
    max_live_frontier_dim: usize,
    reachable_layer_scratch_bytes: usize,
}

fn checked_vector_scratch(elements: usize) -> Option<usize> {
    elements.checked_mul(VECTOR_PARAMETER_SCRATCH_BYTES_PER_ELEMENT)
}

fn checked_faer_padded_rows(rows: usize, element_bytes: usize) -> Option<usize> {
    if element_bytes == 0 || !FAER_OWNED_ALIGNMENT_BYTES.is_multiple_of(element_bytes) {
        return None;
    }
    let row_granularity = FAER_OWNED_ALIGNMENT_BYTES / element_bytes;
    let remainder = rows % row_granularity;
    if remainder == 0 {
        Some(rows)
    } else {
        rows.checked_add(row_granularity - remainder)
    }
}

fn checked_matrix_bytes(rows: usize, columns: usize, element_bytes: usize) -> Option<usize> {
    rows.checked_mul(columns)?.checked_mul(element_bytes)
}

fn checked_faer_matrix_bytes(rows: usize, columns: usize, element_bytes: usize) -> Option<usize> {
    checked_matrix_bytes(
        checked_faer_padded_rows(rows, element_bytes)?,
        columns,
        element_bytes,
    )
}

/// Exact user-owned f64 faer footprint of one deadline A·W chunk: two padded
/// weight images, two padded input images, and the two padded products that
/// coexist when the absolute product finishes.
fn checked_linear_faer_chunk_bytes(
    contraction: usize,
    output_columns: usize,
    chunk_rows: usize,
) -> Option<usize> {
    checked_faer_matrix_bytes(contraction, output_columns, size_of::<f64>())?
        .checked_mul(2)?
        .checked_add(
            checked_faer_matrix_bytes(chunk_rows, contraction, size_of::<f64>())?.checked_mul(2)?,
        )?
        .checked_add(
            checked_faer_matrix_bytes(chunk_rows, output_columns, size_of::<f64>())?
                .checked_mul(2)?,
        )
}

/// Largest row chunk the actual deadline faer path can own for this shape.
/// `None` means that path takes its scalar fallback and owns no padded f64
/// weight/chunk matrices. The contraction axis is never split.
fn checked_linear_faer_chunk_rows(
    contraction: usize,
    output_columns: usize,
    stacked_rows: usize,
) -> Option<usize> {
    let row_macs = contraction.checked_mul(output_columns)?;
    if row_macs == 0
        || row_macs > LINEAR_DEADLINE_FAER_CHUNK_MACS
        || stacked_rows == 0
        || checked_linear_faer_chunk_bytes(contraction, output_columns, 1)?
            > LINEAR_DEADLINE_FAER_MAX_SCRATCH_BYTES
    {
        return None;
    }

    let mac_limited_rows = LINEAR_DEADLINE_FAER_CHUNK_MACS / row_macs;
    let candidate = stacked_rows.min(mac_limited_rows);
    if checked_linear_faer_chunk_bytes(contraction, output_columns, candidate)
        .is_some_and(|bytes| bytes <= LINEAR_DEADLINE_FAER_MAX_SCRATCH_BYTES)
    {
        return Some(candidate);
    }

    // One row is known-good. Find the largest admitted row count; the actual
    // helper uses the same monotone padded-capacity predicate.
    let mut accepted = 1usize;
    let mut rejected = candidate;
    while rejected - accepted > 1 {
        let midpoint = accepted.midpoint(rejected);
        if checked_linear_faer_chunk_bytes(contraction, output_columns, midpoint)
            .is_some_and(|bytes| bytes <= LINEAR_DEADLINE_FAER_MAX_SCRATCH_BYTES)
        {
            accepted = midpoint;
        } else {
            rejected = midpoint;
        }
    }
    Some(accepted)
}

/// Checked transient footprint specific to one reachable Linear layer.
///
/// The general frontier charge covers persistent coefficient carriers. This
/// adds every major allocation whose shape is not captured by logical frontier
/// elements, including faer's 64-byte row padding:
///
/// * the call-wide padded f32 `w_abs`;
/// * the padded f32 `2R×K` stacked input;
/// * the two logical ndarray f64 `2R×P` result temporaries; and
/// * when the faer deadline chunk is admitted, both padded f64 weight images,
///   both padded chunk operands, and both padded chunk products.
fn checked_linear_scratch_bytes(
    linear: &crate::layers::LinearLayer,
    target_rows: usize,
) -> Option<usize> {
    let contraction = linear.weight().nrows();
    let output_columns = linear.weight().ncols();
    let stacked_rows = target_rows.checked_mul(2)?;

    let mut bytes = checked_faer_matrix_bytes(contraction, output_columns, size_of::<f32>())?;
    bytes = bytes.checked_add(checked_faer_matrix_bytes(
        stacked_rows,
        contraction,
        size_of::<f32>(),
    )?)?;
    bytes = bytes.checked_add(
        checked_matrix_bytes(stacked_rows, output_columns, size_of::<f64>())?.checked_mul(2)?,
    )?;
    if let Some(chunk_rows) =
        checked_linear_faer_chunk_rows(contraction, output_columns, stacked_rows)
    {
        bytes = bytes.checked_add(checked_linear_faer_chunk_bytes(
            contraction,
            output_columns,
            chunk_rows,
        )?)?;
    }
    bytes.checked_add(checked_vector_scratch(
        linear.bias().map_or(0, |bias| bias.len()),
    )?)
}

fn checked_conv_scratch(kernel_elements: usize, bias_elements: usize) -> Option<usize> {
    kernel_elements
        .checked_mul(CONV_PARAMETER_SCRATCH_BYTES_PER_ELEMENT)?
        .checked_add(checked_vector_scratch(bias_elements)?)
}

/// Scratch whose size comes from immutable layer parameters rather than the
/// live coefficient frontier. Only layers supported by the comprehensive CPU
/// ResNet route are admitted. A future layer variant must get an audited
/// checked charge here before it can participate; declining is sound and keeps
/// the default-dark optimization from allocating beyond its admission proof.
fn checked_layer_scratch_bytes(layer: &Layer, target_rows: usize) -> Option<usize> {
    match layer {
        Layer::Linear(linear) => checked_linear_scratch_bytes(linear, target_rows),
        Layer::Conv1d(conv) => checked_conv_scratch(
            conv.kernel.len(),
            conv.bias.as_ref().map_or(0, |bias| bias.len()),
        ),
        Layer::ConvTranspose1d(conv) => checked_conv_scratch(
            conv.kernel.len(),
            conv.bias.as_ref().map_or(0, |bias| bias.len()),
        ),
        Layer::Conv2d(conv) => checked_conv_scratch(
            conv.kernel.len(),
            conv.bias.as_ref().map_or(0, |bias| bias.len()),
        ),
        Layer::ConvTranspose2d(conv) => checked_conv_scratch(
            conv.kernel.len(),
            conv.bias.as_ref().map_or(0, |bias| bias.len()),
        ),
        Layer::BatchNorm(batch_norm) => [
            batch_norm.scale.len(),
            batch_norm.bias.len(),
            batch_norm.scale_err.len(),
            batch_norm.bias_err.len(),
        ]
        .into_iter()
        .try_fold(0usize, |total, elements| {
            total.checked_add(checked_vector_scratch(elements)?)
        }),
        Layer::Reshape(reshape) => checked_vector_scratch(reshape.target_shape.len()),
        Layer::ReLU(_)
        | Layer::Add(_)
        | Layer::Sub(_)
        | Layer::AveragePool(_)
        | Layer::MaxPool2d(_)
        | Layer::Flatten(_)
        | Layer::SkipMerge(_) => Some(0),
        _ => None,
    }
}

fn target_resources(
    graph: &GraphNetwork,
    plan: &CrownDispatchPlan,
    target: &str,
    bounds: &HashMap<String, BoundedTensor>,
    input_dim: usize,
    target_rows: usize,
) -> Option<TargetResources> {
    let target_idx = plan.index_of(target)?;
    let mut pending = HashSet::new();
    pending
        .try_reserve(graph.nodes.len().saturating_add(1))
        .ok()?;
    pending.insert(target_idx);
    let mut input_pending = false;
    let mut live_dim = bounds.get(target)?.len();
    let mut max_live_dim = live_dim;
    let mut reachable_layer_scratch_bytes = 0usize;

    for &idx in &plan.reverse_order {
        if !pending.remove(&idx) {
            continue;
        }
        let name = plan.name_of(idx);
        let node_dim = bounds.get(name)?.len();
        let node = graph.nodes.get(name)?;
        reachable_layer_scratch_bytes = reachable_layer_scratch_bytes
            .checked_add(checked_layer_scratch_bytes(&node.layer, target_rows)?)?;

        // The consumed coefficient and every newly produced contribution can
        // coexist inside layer dispatch. Count every contribution here even
        // when another branch already has a pending relation at the same
        // destination: the new image and accumulated destination coexist until
        // their deterministic merge completes.
        let mut transition_dim = live_dim;
        for input_name in &node.inputs {
            if input_name == NETWORK_INPUT {
                transition_dim = transition_dim.checked_add(input_dim)?;
            } else {
                transition_dim = transition_dim.checked_add(bounds.get(input_name)?.len())?;
            }
        }
        max_live_dim = max_live_dim.max(transition_dim);

        live_dim = live_dim.checked_sub(node_dim)?;
        for input_name in &node.inputs {
            if input_name == NETWORK_INPUT {
                if !input_pending {
                    input_pending = true;
                    live_dim = live_dim.checked_add(input_dim)?;
                }
            } else {
                let input_idx = plan.index_of(input_name)?;
                if pending.insert(input_idx) {
                    live_dim = live_dim.checked_add(bounds.get(input_name)?.len())?;
                }
            }
        }
        max_live_dim = max_live_dim.max(live_dim);
    }
    (pending.is_empty() && input_pending && max_live_dim > 0).then_some(TargetResources {
        max_live_frontier_dim: max_live_dim,
        reachable_layer_scratch_bytes,
    })
}

/// Select every eligible target in canonical execution order. Exceeding the
/// hard target cap refuses the entire transaction; the prefix is never used.
fn plan_targets(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
) -> Option<Vec<TargetPlan>> {
    let dispatch_plan = CrownDispatchPlan::build(graph).ok()?;
    let order = graph.exec_order().ok()?;
    let mut seen = HashSet::new();
    seen.try_reserve(MAX_TARGETS.saturating_add(1)).ok()?;
    let mut targets = Vec::new();
    targets.try_reserve_exact(MAX_TARGETS).ok()?;
    for relu_name in order {
        let node = graph.nodes.get(relu_name)?;
        if !matches!(node.layer, Layer::ReLU(_)) {
            continue;
        }
        let pre_name = node.inputs.first()?;
        if pre_name == NETWORK_INPUT || !seen.insert(pre_name.clone()) {
            continue;
        }
        let Some(reference) = bounds.get(pre_name) else {
            continue;
        };
        let rows = reference.len();
        if rows == 0 || rows > MAX_TARGET_DIM {
            continue;
        }
        if targets.len() == MAX_TARGETS {
            return None;
        }
        let resources =
            target_resources(graph, &dispatch_plan, pre_name, bounds, input.len(), rows)?;
        targets.push(TargetPlan {
            index: targets.len(),
            relu_name: relu_name.clone(),
            pre_name: pre_name.clone(),
            rows,
            max_live_frontier_dim: resources.max_live_frontier_dim,
            reachable_layer_scratch_bytes: resources.reachable_layer_scratch_bytes,
        });
    }
    (!targets.is_empty()).then_some(targets)
}

fn build_frozen_snapshot(
    bounds: &HashMap<String, BoundedTensor>,
) -> Option<HashMap<String, Arc<BoundedTensor>>> {
    let mut frozen = HashMap::new();
    frozen.try_reserve(bounds.len()).ok()?;
    for (name, tensor) in bounds {
        frozen.insert(name.clone(), Arc::new(tensor.clone()));
    }
    Some(frozen)
}

/// Run indexed work on an owned CPU-only pool. Installing the f64 guard in the
/// scoped thread wrapper covers nested Rayon joins as well as the outer target
/// closure; the per-task guard keeps faer sequential within each target.
fn run_indexed_pool<T, R, F>(items: &[T], workers: usize, work: F) -> Option<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Sync,
{
    if items.is_empty() || workers == 0 || workers > MAX_WORKERS {
        return None;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers.min(items.len()))
        .thread_name(|index| format!("ny-root-interm-cpu-{index}"))
        .build_scoped(
            |thread| {
                let _cpu_only = crate::sound_f64_gemm::CpuOnlyF64Guard::new();
                thread.run();
            },
            |pool| {
                pool.install(|| {
                    items
                        .par_iter()
                        .enumerate()
                        .map(|(index, item)| {
                            let _task = crate::faer_parallelism::RayonTaskGuard::new();
                            work(index, item)
                        })
                        .collect()
                })
            },
        )
        .ok()
}

fn compute_all_targets(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    frozen: &HashMap<String, Arc<BoundedTensor>>,
    targets: &[TargetPlan],
    workers: usize,
    deadline: Instant,
) -> Option<Vec<ComputedTarget>> {
    let raw = run_indexed_pool(targets, workers, |index, target| {
        if past(deadline) {
            return None;
        }
        let candidate = backward_input_relative_bounds_at_node(
            graph,
            target.pre_name.as_str(),
            frozen,
            input,
            &crate::faer_parallelism::FaerCpuGemmEngine,
            Some(deadline),
            None,
        )
        .and_then(|linear| {
            linear
                .concretize_sound_with_deadline(input, Some(deadline))
                .ok()
        })?;
        if past(deadline)
            || candidate.shape() != frozen.get(target.pre_name.as_str())?.shape()
            || candidate
                .lower()
                .iter()
                .chain(candidate.upper().iter())
                .any(|value| !value.is_finite())
        {
            return None;
        }
        Some(ComputedTarget {
            index,
            relu_name: target.relu_name.clone(),
            pre_name: target.pre_name.clone(),
            candidate,
        })
    })?;
    if past(deadline) || raw.len() != targets.len() {
        return None;
    }
    raw.into_iter().collect()
}

fn tightened_elements(before: &BoundedTensor, after: &BoundedTensor) -> usize {
    before
        .lower()
        .iter()
        .zip(before.upper().iter())
        .zip(after.lower().iter().zip(after.upper().iter()))
        .filter(|((before_l, before_u), (after_l, after_u))| {
            after_l.to_bits() != before_l.to_bits() || after_u.to_bits() != before_u.to_bits()
        })
        .count()
}

fn publish_validated_batch<N>(
    live: &mut HashMap<String, BoundedTensor>,
    frozen: &HashMap<String, Arc<BoundedTensor>>,
    targets: &[TargetPlan],
    computed: Vec<ComputedTarget>,
    deadline: Instant,
    mut now: N,
) -> Option<ComprehensiveCpuReport>
where
    N: FnMut() -> Instant,
{
    if computed.len() != targets.len() || now() >= deadline {
        return None;
    }
    let mut staged = Vec::new();
    staged.try_reserve_exact(targets.len()).ok()?;
    let mut tightened_target_count = 0usize;
    let mut tightened_element_count = 0usize;

    for (expected, result) in targets.iter().zip(computed) {
        if now() >= deadline
            || result.index != expected.index
            || result.relu_name != expected.relu_name
            || result.pre_name != expected.pre_name
        {
            return None;
        }
        let frozen_reference = frozen.get(expected.pre_name.as_str())?;
        let current = live.get(expected.pre_name.as_str())?;
        if !tensor_exact_eq(current, frozen_reference) {
            return None;
        }
        let preserved_l2 = current.l2_constraint().cloned();
        let (mut intersection, disjoint) = current
            .intersection_per_element_with_poll(&result.candidate, || {
                if now() >= deadline {
                    Err(ny_core::NyError::DeadlineExceeded(
                        "comprehensive CPU intermediate publication expired".into(),
                    ))
                } else {
                    Ok(())
                }
            })
            .ok()??;
        if disjoint != 0 {
            return None;
        }
        if let Some(annotation) = preserved_l2 {
            intersection = intersection.with_l2_constraint(annotation);
        }
        if !l2_exact_eq(intersection.l2_constraint(), current.l2_constraint()) {
            return None;
        }
        let changed = tightened_elements(current, &intersection);
        tightened_target_count += usize::from(changed > 0);
        tightened_element_count = tightened_element_count.checked_add(changed)?;
        staged.push((expected.pre_name.clone(), intersection));
    }

    // Revalidate every live source after all fallible staging. The final
    // authority check is intentionally adjacent to the first assignment; no
    // allocation, lookup, clock read, or other fallible work occurs afterward.
    if targets.iter().any(|target| {
        live.get(target.pre_name.as_str())
            .zip(frozen.get(target.pre_name.as_str()))
            .is_none_or(|(current, reference)| !tensor_exact_eq(current, reference))
    }) || now() >= deadline
    {
        return None;
    }
    for (name, replacement) in staged {
        *live
            .get_mut(name.as_str())
            .expect("validated live target must remain present during exclusive commit") =
            replacement;
    }
    Some(ComprehensiveCpuReport {
        attempted_targets: targets.len(),
        tightened_targets: tightened_target_count,
        tightened_elements: tightened_element_count,
    })
}

/// Run the governed all-target CPU transaction. `None` is a clean, sound
/// decline and guarantees that `live` is unchanged.
pub(crate) fn run_comprehensive_cpu_intermediate_tighten(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    live: &mut HashMap<String, BoundedTensor>,
    authority_deadline: Option<Instant>,
) -> Option<ComprehensiveCpuReport> {
    let started = Instant::now();
    let authority_deadline = authority_deadline?;
    let local_cap = started
        .checked_add(MAX_RUNTIME)
        .unwrap_or(authority_deadline);
    let deadline = authority_deadline.min(local_cap);
    if started >= deadline {
        return None;
    }
    let targets = plan_targets(graph, input, live)?;
    let dense_budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    let process_headroom = crate::network::crown_memory::process_memory_headroom_bytes();
    let Some((workers, required, budget)) =
        admitted_memory_plan(&targets, live, MAX_WORKERS, dense_budget, process_headroom)
    else {
        tracing::info!(
            targets = targets.len(),
            max_workers = MAX_WORKERS.min(targets.len()),
            dense_budget_bytes = dense_budget,
            process_headroom_bytes = ?process_headroom,
            "comprehensive CPU intermediate sweep declined checked aggregate memory admission"
        );
        return None;
    };
    if past(deadline) {
        return None;
    }
    let frozen = build_frozen_snapshot(live)?;
    let computed = compute_all_targets(graph, input, &frozen, &targets, workers, deadline)?;
    let report =
        publish_validated_batch(live, &frozen, &targets, computed, deadline, Instant::now)?;
    tracing::info!(
        targets = report.attempted_targets,
        tightened_targets = report.tightened_targets,
        tightened_elements = report.tightened_elements,
        workers,
        required_bytes = required,
        budget_bytes = budget,
        elapsed_s = started.elapsed().as_secs_f64(),
        "comprehensive CPU intermediate sweep committed atomically"
    );
    Some(report)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex};

    use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

    use super::*;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use crate::GraphNode;

    fn bounded(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new_allow_infinite(arr1(lower).into_dyn(), arr1(upper).into_dyn())
            .expect("valid fixture bounds")
    }

    fn fixture() -> (GraphNetwork, HashMap<String, BoundedTensor>, BoundedTensor) {
        let linear0 =
            LinearLayer::new(arr2(&[[1.0, -0.5], [-1.5, 0.25]]), Some(arr1(&[0.1, -0.2])))
                .expect("linear0");
        let linear1 =
            LinearLayer::new(arr2(&[[0.5, -1.0], [1.25, 0.75]]), Some(arr1(&[-0.1, 0.2])))
                .expect("linear1");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear0", Layer::Linear(linear0)));
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["linear0".into()],
        ));
        graph.add_node(GraphNode::new(
            "linear1",
            Layer::Linear(linear1),
            vec!["relu0".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".into()],
        ));
        graph.set_output("relu1");

        // linear0 is stable-positive and non-point. Comprehensive selection
        // must not inherit the sparse route's crossing-only gate.
        let live = HashMap::from([
            ("linear0".into(), bounded(&[0.25, 0.5], &[2.0, 3.0])),
            ("relu0".into(), bounded(&[0.25, 0.5], &[2.0, 3.0])),
            ("linear1".into(), bounded(&[-4.0, -4.0], &[4.0, 4.0])),
            ("relu1".into(), bounded(&[0.0, 0.0], &[4.0, 4.0])),
        ]);
        (graph, live, bounded(&[-1.0, -0.5], &[1.0, 0.75]))
    }

    fn map_signature(map: &HashMap<String, BoundedTensor>) -> Vec<u64> {
        let mut names: Vec<_> = map.keys().collect();
        names.sort_unstable();
        let mut signature = Vec::new();
        for name in names {
            let tensor = &map[name];
            signature.push(name.len() as u64);
            signature.extend(name.bytes().map(u64::from));
            signature.push(tensor.shape().len() as u64);
            signature.extend(tensor.shape().iter().map(|&dim| dim as u64));
            signature.extend(
                tensor
                    .lower()
                    .iter()
                    .map(|value| u64::from(value.to_bits())),
            );
            signature.extend(
                tensor
                    .upper()
                    .iter()
                    .map(|value| u64::from(value.to_bits())),
            );
            match tensor.l2_constraint() {
                None => signature.push(0),
                Some(l2) => {
                    signature.push(1);
                    signature.push(l2.axis() as u64);
                    signature.extend(l2.center().iter().map(|value| u64::from(value.to_bits())));
                    signature.extend(l2.radius().iter().map(|value| u64::from(value.to_bits())));
                }
            }
        }
        signature
    }

    fn target(index: usize, relu: &str, pre: &str, rows: usize) -> TargetPlan {
        TargetPlan {
            index,
            relu_name: relu.into(),
            pre_name: pre.into(),
            rows,
            max_live_frontier_dim: rows,
            reachable_layer_scratch_bytes: 0,
        }
    }

    #[test]
    fn stable_non_point_preactivation_is_eligible() {
        let (graph, live, input) = fixture();
        let targets = plan_targets(&graph, &input, &live).expect("eligible targets");
        assert_eq!(
            targets
                .iter()
                .map(|target| target.pre_name.as_str())
                .collect::<Vec<_>>(),
            ["linear0", "linear1"]
        );
        assert!(live["linear0"]
            .lower()
            .iter()
            .zip(live["linear0"].upper().iter())
            .all(|(&lower, &upper)| lower > 0.0 && lower < upper));
    }

    #[test]
    fn dag_memory_frontier_charges_both_branches_and_transition_sources() {
        let identity =
            || LinearLayer::new(Array2::<f32>::eye(2), Some(Array1::<f32>::zeros(2))).unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("left", Layer::Linear(identity())));
        graph.add_node(GraphNode::from_input("right", Layer::Linear(identity())));
        graph.add_node(GraphNode::new(
            "sum",
            Layer::Add(AddLayer),
            vec!["left".into(), "right".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["sum".into()],
        ));
        graph.set_output("relu");
        let live = HashMap::from([
            ("left".into(), bounded(&[-2.0, -2.0], &[2.0, 2.0])),
            ("right".into(), bounded(&[-2.0, -2.0], &[2.0, 2.0])),
            ("sum".into(), bounded(&[-4.0, -4.0], &[4.0, 4.0])),
            ("relu".into(), bounded(&[0.0, 0.0], &[4.0, 4.0])),
        ]);
        let targets =
            plan_targets(&graph, &bounded(&[-1.0, -1.0], &[1.0, 1.0]), &live).expect("DAG target");
        assert_eq!(targets.len(), 1);
        // Dispatching `sum` can retain its two coefficient columns while
        // materializing both two-column parents: 2 + 2 + 2 = 6.
        assert_eq!(targets[0].max_live_frontier_dim, 6);
    }

    #[test]
    fn scalar_wide_linear_uses_padded_scratch_at_exact_admission_boundary() {
        const WIDE: usize = 16_384;
        const STACKED_ROWS: usize = 2;
        const PADDED_F32_ROWS: usize = FAER_OWNED_ALIGNMENT_BYTES / size_of::<f32>();
        const PADDED_F64_ROWS: usize = FAER_OWNED_ALIGNMENT_BYTES / size_of::<f64>();

        let linear = LinearLayer::new(Array2::<f32>::ones((1, WIDE)), None).unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["linear".into()],
        ));
        graph.set_output("relu");
        let live = HashMap::from([
            ("linear".into(), bounded(&[-16_384.0], &[16_384.0])),
            ("relu".into(), bounded(&[0.0], &[16_384.0])),
        ]);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[WIDE]), -1.0),
            ArrayD::from_elem(IxDyn(&[WIDE]), 1.0),
        )
        .unwrap();

        let targets = plan_targets(&graph, &input, &live).expect("scalar target");
        assert_eq!(targets.len(), 1);
        let target = &targets[0];
        assert_eq!(target.rows, 1);
        assert_eq!(target.max_live_frontier_dim, WIDE + 1);
        assert_eq!(
            checked_linear_faer_chunk_rows(1, WIDE, STACKED_ROWS),
            Some(STACKED_ROWS),
            "the real deadline path admits its padded faer chunk for this shape"
        );

        let padded_f32_w_abs = PADDED_F32_ROWS * WIDE * size_of::<f32>();
        let padded_f32_stacked_input = PADDED_F32_ROWS * size_of::<f32>();
        let ndarray_f64_results = STACKED_ROWS * WIDE * size_of::<f64>() * 2;
        let padded_f64_weights = PADDED_F64_ROWS * WIDE * size_of::<f64>() * 2;
        let padded_f64_chunk_inputs = PADDED_F64_ROWS * size_of::<f64>() * 2;
        let padded_f64_chunk_products = PADDED_F64_ROWS * WIDE * size_of::<f64>() * 2;
        let exact_reachable_scratch = padded_f32_w_abs
            + padded_f32_stacked_input
            + ndarray_f64_results
            + padded_f64_weights
            + padded_f64_chunk_inputs
            + padded_f64_chunk_products;
        assert_eq!(
            target.reachable_layer_scratch_bytes,
            exact_reachable_scratch
        );

        let frontier_only = target.rows * target.max_live_frontier_dim * FRONTIER_BYTES_PER_ELEMENT
            + target.rows * PER_ROW_METADATA_BYTES;
        let charged = checked_target_peak_bytes(
            target.rows,
            target.max_live_frontier_dim,
            target.reachable_layer_scratch_bytes,
        )
        .unwrap();
        assert_eq!(charged, frontier_only + exact_reachable_scratch);

        let aggregate = checked_aggregate_bytes(&targets, &live, 1).unwrap();
        assert_eq!(
            admitted_memory_plan(&targets, &live, 1, aggregate, None),
            Some((1, aggregate, aggregate))
        );
        assert!(admitted_memory_plan(&targets, &live, 1, aggregate - 1, None).is_none());
    }

    #[test]
    fn more_than_ten_targets_refuses_without_selecting_a_prefix() {
        let mut graph = GraphNetwork::new();
        let mut live = HashMap::new();
        let mut prior = None::<String>;
        for index in 0..=MAX_TARGETS {
            let linear_name = format!("linear{index}");
            let relu_name = format!("relu{index}");
            let linear = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap();
            let linear_node = match prior.as_ref() {
                Some(input) => GraphNode::new(
                    linear_name.clone(),
                    Layer::Linear(linear),
                    vec![input.clone()],
                ),
                None => GraphNode::from_input(linear_name.clone(), Layer::Linear(linear)),
            };
            graph.add_node(linear_node);
            graph.add_node(GraphNode::new(
                relu_name.clone(),
                Layer::ReLU(ReLULayer),
                vec![linear_name.clone()],
            ));
            live.insert(linear_name, bounded(&[0.1], &[1.0]));
            live.insert(relu_name.clone(), bounded(&[0.1], &[1.0]));
            prior = Some(relu_name);
        }
        graph.set_output(prior.expect("output"));
        let before = map_signature(&live);
        assert!(plan_targets(&graph, &bounded(&[-1.0], &[1.0]), &live).is_none());
        assert_eq!(map_signature(&live), before);
    }

    #[test]
    fn worker_pool_caps_inflight_and_retains_canonical_order_after_reverse_completion() {
        let items = [0usize, 1, 2, 3];
        let barrier = Barrier::new(MAX_WORKERS);
        let active = AtomicUsize::new(0);
        let max_active = AtomicUsize::new(0);
        let turn = AtomicUsize::new(3);
        let completion = Mutex::new(Vec::new());
        let results = run_indexed_pool(&items, MAX_WORKERS, |index, item| {
            assert_eq!(index, *item);
            assert!(crate::sound_f64_gemm::cpu_only_f64_active());
            let inflight = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(inflight, Ordering::SeqCst);
            barrier.wait();
            while turn.load(Ordering::SeqCst) != *item {
                std::thread::yield_now();
            }
            completion.lock().unwrap().push(*item);
            if *item == 0 {
                turn.store(usize::MAX, Ordering::SeqCst);
            } else {
                turn.fetch_sub(1, Ordering::SeqCst);
            }
            active.fetch_sub(1, Ordering::SeqCst);
            *item
        })
        .expect("private pool");
        assert_eq!(results, items);
        assert_eq!(*completion.lock().unwrap(), [3, 2, 1, 0]);
        assert_eq!(max_active.load(Ordering::SeqCst), MAX_WORKERS);
    }

    #[test]
    fn cpu_only_scope_covers_nested_rayon_work_without_barrier_reentrancy() {
        let results = run_indexed_pool(&[0usize, 1], 2, |_, item| {
            assert!(crate::sound_f64_gemm::cpu_only_f64_active());
            let nested = rayon::join(
                crate::sound_f64_gemm::cpu_only_f64_active,
                crate::sound_f64_gemm::cpu_only_f64_active,
            );
            assert_eq!(nested, (true, true));
            *item
        })
        .expect("private pool");
        assert_eq!(results, [0, 1]);
    }

    #[test]
    fn parallel_targets_match_one_worker_serial_replay_bit_for_bit() {
        let (graph, live, input) = fixture();
        let targets = plan_targets(&graph, &input, &live).expect("targets");
        let frozen = build_frozen_snapshot(&live).expect("snapshot");
        let deadline = Instant::now() + Duration::from_secs(30);
        let serial = compute_all_targets(&graph, &input, &frozen, &targets, 1, deadline)
            .expect("serial computation");
        let parallel = compute_all_targets(&graph, &input, &frozen, &targets, 2, deadline)
            .expect("parallel computation");
        assert_eq!(serial.len(), parallel.len());
        for (serial, parallel) in serial.iter().zip(&parallel) {
            assert_eq!(serial.index, parallel.index);
            assert_eq!(serial.relu_name, parallel.relu_name);
            assert_eq!(serial.pre_name, parallel.pre_name);
            assert!(tensor_exact_eq(&serial.candidate, &parallel.candidate));
        }
    }

    #[test]
    fn cpu_pool_makes_zero_process_global_f64_accessor_attempts() {
        let _counter_lock = crate::sound_f64_gemm::accessor_counter_test_lock();
        crate::sound_f64_gemm::reset_cpu_only_global_accessor_attempts();
        let dim = 256usize;
        let linear = LinearLayer::new(Array2::<f32>::eye(dim), Some(Array1::<f32>::zeros(dim)))
            .expect("linear");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["linear".into()],
        ));
        graph.set_output("relu");
        let live = HashMap::from([
            (
                "linear".into(),
                BoundedTensor::new(
                    ArrayD::from_elem(IxDyn(&[dim]), -2.0),
                    ArrayD::from_elem(IxDyn(&[dim]), 2.0),
                )
                .unwrap(),
            ),
            (
                "relu".into(),
                BoundedTensor::new(
                    ArrayD::zeros(IxDyn(&[dim])),
                    ArrayD::from_elem(IxDyn(&[dim]), 2.0),
                )
                .unwrap(),
            ),
        ]);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[dim]), -1.0),
            ArrayD::from_elem(IxDyn(&[dim]), 1.0),
        )
        .unwrap();
        let targets = plan_targets(&graph, &input, &live).unwrap();
        let frozen = build_frozen_snapshot(&live).unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        let before_all = crate::sound_f64_gemm::global_accessor_attempts();
        assert!(compute_all_targets(&graph, &input, &frozen, &targets, 1, deadline).is_some());
        assert_eq!(
            crate::sound_f64_gemm::cpu_only_global_accessor_attempts(),
            0
        );

        // The same ordinary, unguarded Linear backward retains its historical
        // global admission attempt. This also proves the zero-call assertion
        // above is sensitive to an accidental lookup.
        assert!(backward_input_relative_bounds_at_node(
            &graph,
            "linear",
            &frozen,
            &input,
            &crate::faer_parallelism::FaerCpuGemmEngine,
            Some(deadline),
            None,
        )
        .is_some());
        assert!(crate::sound_f64_gemm::global_accessor_attempts() > before_all);
    }

    fn annotated_two_element_box() -> BoundedTensor {
        let l2 = L2Constraint::new(
            arr1(&[0.0, 0.0]).into_dyn(),
            ArrayD::from_elem(IxDyn(&[]), 3.0),
            0,
            &[2],
        )
        .expect("L2 annotation");
        bounded(&[-2.0, -1.0], &[2.0, 3.0]).with_l2_constraint(l2)
    }

    #[test]
    fn publication_is_atomic_shrink_only_and_preserves_exact_l2() {
        let mut live = HashMap::from([("pre".into(), annotated_two_element_box())]);
        let frozen = build_frozen_snapshot(&live).unwrap();
        let targets = [target(0, "relu", "pre", 2)];
        let computed = vec![ComputedTarget {
            index: 0,
            relu_name: "relu".into(),
            pre_name: "pre".into(),
            candidate: bounded(&[-1.5, -0.5], &[1.5, 2.5]),
        }];
        let original_l2 = live["pre"].l2_constraint().cloned();
        let report = publish_validated_batch(
            &mut live,
            &frozen,
            &targets,
            computed,
            Instant::now() + Duration::from_secs(1),
            Instant::now,
        )
        .expect("atomic publication");
        assert_eq!(report.attempted_targets, 1);
        assert_eq!(report.tightened_targets, 1);
        assert_eq!(report.tightened_elements, 2);
        assert!(l2_exact_eq(
            live["pre"].l2_constraint(),
            original_l2.as_ref()
        ));
    }

    #[test]
    fn disjoint_refusal_and_expired_deadline_leave_map_byte_identical() {
        let targets = [target(0, "relu", "pre", 2)];

        let mut disjoint_live = HashMap::from([("pre".into(), annotated_two_element_box())]);
        let frozen = build_frozen_snapshot(&disjoint_live).unwrap();
        let before = map_signature(&disjoint_live);
        let disjoint = vec![ComputedTarget {
            index: 0,
            relu_name: "relu".into(),
            pre_name: "pre".into(),
            candidate: bounded(&[4.0, 4.0], &[5.0, 5.0]),
        }];
        assert!(publish_validated_batch(
            &mut disjoint_live,
            &frozen,
            &targets,
            disjoint,
            Instant::now() + Duration::from_secs(1),
            Instant::now,
        )
        .is_none());
        assert_eq!(map_signature(&disjoint_live), before);

        let mut expired_live = HashMap::from([("pre".into(), annotated_two_element_box())]);
        let frozen = build_frozen_snapshot(&expired_live).unwrap();
        let before = map_signature(&expired_live);
        let candidate = vec![ComputedTarget {
            index: 0,
            relu_name: "relu".into(),
            pre_name: "pre".into(),
            candidate: bounded(&[-1.5, -0.5], &[1.5, 2.5]),
        }];
        assert!(publish_validated_batch(
            &mut expired_live,
            &frozen,
            &targets,
            candidate,
            Instant::now(),
            Instant::now,
        )
        .is_none());
        assert_eq!(map_signature(&expired_live), before);
    }

    #[test]
    fn late_disjoint_target_refuses_every_earlier_staged_shrink() {
        let mut live = HashMap::from([
            ("a".into(), bounded(&[-2.0], &[2.0])),
            ("b".into(), bounded(&[-3.0], &[3.0])),
        ]);
        let frozen = build_frozen_snapshot(&live).unwrap();
        let targets = [target(0, "ra", "a", 1), target(1, "rb", "b", 1)];
        let computed = vec![
            ComputedTarget {
                index: 0,
                relu_name: "ra".into(),
                pre_name: "a".into(),
                candidate: bounded(&[-1.0], &[1.0]),
            },
            ComputedTarget {
                index: 1,
                relu_name: "rb".into(),
                pre_name: "b".into(),
                candidate: bounded(&[4.0], &[5.0]),
            },
        ];
        let before = map_signature(&live);
        assert!(publish_validated_batch(
            &mut live,
            &frozen,
            &targets,
            computed,
            Instant::now() + Duration::from_secs(1),
            Instant::now,
        )
        .is_none());
        assert_eq!(map_signature(&live), before);
    }

    #[test]
    fn deadline_loss_at_commit_boundary_discards_the_complete_staged_batch() {
        let template = HashMap::from([("pre".into(), annotated_two_element_box())]);
        let targets = [target(0, "relu", "pre", 2)];
        let computed = || {
            vec![ComputedTarget {
                index: 0,
                relu_name: "relu".into(),
                pre_name: "pre".into(),
                candidate: bounded(&[-1.5, -0.5], &[1.5, 2.5]),
            }]
        };
        let deadline = Instant::now() + Duration::from_secs(1);
        let live_time = deadline
            .checked_sub(Duration::from_millis(1))
            .expect("deadline is at least one millisecond after the Instant epoch");

        // Measure this exact shape's polling transcript, then expire only its
        // final authority-adjacent clock read.
        let mut probe = template.clone();
        let frozen = build_frozen_snapshot(&probe).unwrap();
        let probe_calls = Cell::new(0usize);
        assert!(publish_validated_batch(
            &mut probe,
            &frozen,
            &targets,
            computed(),
            deadline,
            || {
                probe_calls.set(probe_calls.get() + 1);
                live_time
            },
        )
        .is_some());
        let final_call = probe_calls.get();
        assert!(final_call > 1);

        let mut live = template;
        let frozen = build_frozen_snapshot(&live).unwrap();
        let before = map_signature(&live);
        let calls = Cell::new(0usize);
        assert!(publish_validated_batch(
            &mut live,
            &frozen,
            &targets,
            computed(),
            deadline,
            || {
                let call = calls.get() + 1;
                calls.set(call);
                if call == final_call {
                    deadline
                } else {
                    live_time
                }
            },
        )
        .is_none());
        assert_eq!(calls.get(), final_call);
        assert_eq!(map_signature(&live), before);
    }

    #[test]
    fn reverse_or_malformed_collection_refuses_whole_publication() {
        let mut live = HashMap::from([
            ("a".into(), bounded(&[-2.0], &[2.0])),
            ("b".into(), bounded(&[-3.0], &[3.0])),
        ]);
        let frozen = build_frozen_snapshot(&live).unwrap();
        let targets = [target(0, "ra", "a", 1), target(1, "rb", "b", 1)];
        let reversed = vec![
            ComputedTarget {
                index: 1,
                relu_name: "rb".into(),
                pre_name: "b".into(),
                candidate: bounded(&[-2.0], &[2.0]),
            },
            ComputedTarget {
                index: 0,
                relu_name: "ra".into(),
                pre_name: "a".into(),
                candidate: bounded(&[-1.0], &[1.0]),
            },
        ];
        let before = map_signature(&live);
        assert!(publish_validated_batch(
            &mut live,
            &frozen,
            &targets,
            reversed,
            Instant::now() + Duration::from_secs(1),
            Instant::now,
        )
        .is_none());
        assert_eq!(map_signature(&live), before);
    }

    #[test]
    fn checked_memory_estimate_refuses_overflow_and_honors_exact_headroom() {
        let bounds = HashMap::from([("pre".into(), bounded(&[-1.0], &[1.0]))]);
        let overflow = [TargetPlan {
            index: 0,
            relu_name: "relu".into(),
            pre_name: "pre".into(),
            rows: usize::MAX,
            max_live_frontier_dim: usize::MAX,
            reachable_layer_scratch_bytes: usize::MAX,
        }];
        assert_eq!(checked_aggregate_bytes(&overflow, &bounds, 1), None);
        assert!(!memory_admitted(None, usize::MAX));

        assert_eq!(aggregate_budget_from(1_000, Some(8_000), 4), 4_000);
        assert_eq!(aggregate_budget_from(1_000, Some(6_000), 4), 3_000);
        assert!(memory_admitted(Some(3_000), 3_000));
        assert!(!memory_admitted(Some(3_001), 3_000));

        let bounds = HashMap::from([
            ("a".into(), bounded(&[-1.0, -1.0], &[1.0, 1.0])),
            ("b".into(), bounded(&[-1.0, -1.0], &[1.0, 1.0])),
        ]);
        let targets = [target(0, "ra", "a", 2), target(1, "rb", "b", 2)];
        let required_one = checked_aggregate_bytes(&targets, &bounds, 1).unwrap();
        let required_two = checked_aggregate_bytes(&targets, &bounds, 2).unwrap();
        assert!(required_two > required_one);
        let one_worker_headroom = u64::try_from(required_one).unwrap() * 2;
        let admitted =
            admitted_memory_plan(&targets, &bounds, 2, usize::MAX, Some(one_worker_headroom))
                .expect("one worker fits exact half-headroom boundary");
        assert_eq!(admitted, (1, required_one, required_one));
        assert!(admitted_memory_plan(
            &targets,
            &bounds,
            2,
            usize::MAX,
            Some(one_worker_headroom - 2),
        )
        .is_none());
    }

    #[test]
    fn aggregate_memory_charges_snapshot_and_staged_l2_exactly_once_each() {
        let plain = HashMap::from([("pre".into(), bounded(&[-2.0, -1.0], &[2.0, 3.0]))]);
        let annotated = HashMap::from([("pre".into(), annotated_two_element_box())]);
        let targets = [target(0, "relu", "pre", 2)];

        let plain_bytes = checked_aggregate_bytes(&targets, &plain, 1).unwrap();
        let annotated_bytes = checked_aggregate_bytes(&targets, &annotated, 1).unwrap();
        let l2 = annotated["pre"].l2_constraint().unwrap();
        let one_l2_bytes = (l2.center().len() + l2.radius().len()) * size_of::<f32>();

        // One clone lives in the immutable snapshot and one moves into the
        // staged replacement; publication creates no third transient clone.
        assert_eq!(annotated_bytes - plain_bytes, 2 * one_l2_bytes);
    }

    #[test]
    fn staged_live_map_drift_refuses_without_overwriting_the_new_authority() {
        let original = bounded(&[-2.0], &[2.0]);
        let mut live = HashMap::from([("pre".into(), original)]);
        let frozen = build_frozen_snapshot(&live).unwrap();
        live.insert("pre".into(), bounded(&[-1.75], &[1.75]));
        let before = map_signature(&live);
        let targets = [target(0, "relu", "pre", 1)];
        let computed = vec![ComputedTarget {
            index: 0,
            relu_name: "relu".into(),
            pre_name: "pre".into(),
            candidate: bounded(&[-1.0], &[1.0]),
        }];
        assert!(publish_validated_batch(
            &mut live,
            &frozen,
            &targets,
            computed,
            Instant::now() + Duration::from_secs(1),
            Instant::now,
        )
        .is_none());
        assert_eq!(map_signature(&live), before);
    }
}
