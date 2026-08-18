// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deadline-safe CPU facade for the default-dark bounded CUDA β-CROWN lane.
//!
//! The shared multi-objective executor accepts a [`GemmEngine`] and uses that
//! engine for both generic propagation work and optional GPU-specific routes.
//! CUDA currently offers a narrower contract: its `2..=8` row β-CROWN entry is
//! call-local and deadline bounded, while its ordinary GEMM surface must not be
//! handed generic finite-deadline work. This module keeps those authorities
//! separate. The shared executor sees only [`DeadlineCpuGemmEngine`]; the
//! constrained backward's existing global bounded-β selector independently
//! observes the already-preinitialized CUDA backend for its one sanctioned
//! call-local transaction.

use std::collections::HashSet;
use std::ffi::OsStr;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{
    checked_dim_product, ConvTranspose2dParams, GemmEngine, GpuCrownBackward, NyError, Result,
    DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
};
use ny_tensor::BoundedTensor;

use crate::beta_crown::domain::NodeBoundsView;
use crate::{GraphNetwork, GraphNode, Layer, NETWORK_INPUT};

pub(super) const MO_CUDA_BOUNDED_SHARED_EXECUTOR_ENV: &str = "NY_MO_CUDA_BOUNDED_SHARED_EXECUTOR";

const CPU_POLL_INTERVAL: usize = 1_024;
/// Hard cap on any one output/scratch allocation made by the local facade.
///
/// This bounds the allocator's non-cooperative portion. Larger generic work is
/// refused instead of borrowing time from the narrow CUDA treatment.
const MAX_CPU_BUFFER_BYTES: usize = 256 * 1024 * 1024;
/// Initialization is split so zero-filling never runs for more than this many
/// scalar stores without another deadline observation.
const CPU_INIT_CHUNK_ELEMENTS: usize = 64 * 1024;
const MAX_ADMISSION_GRAPH_NODES: usize = 512;
const MAX_ADMISSION_GRAPH_EDGES: usize = 1_024;
const MAX_ADMISSION_TENSOR_RANK: usize = 8;
const MAX_ADMISSION_TENSOR_ELEMENTS: usize = 4 * 1024 * 1024;
const MAX_ADMISSION_TOTAL_ELEMENTS: usize = 64 * 1024 * 1024;
const MAX_ADMISSION_PARAMETER_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_ADMISSION_TOTAL_PARAMETER_ELEMENTS: usize = 4 * 1024 * 1024;
/// Includes every graph-owned spelling that may be cloned or hashed by the
/// topology walk and by a subsequently admitted graph clone.
const MAX_ADMISSION_IDENTIFIER_BYTES: usize = 256 * 1024;
const MAX_ADMISSION_SINGLE_IDENTIFIER_BYTES: usize = 1024;
const MAX_ADMISSION_OBJECTIVES: usize = 512;
const MAX_ADMISSION_OBJECTIVE_ELEMENTS: usize = 1024 * 1024;
/// Caps wave-replicated ReLU metadata/bounds before unstable discovery and
/// child construction. The treatment caps its outer wave to the audited K=8
/// backend capacity after admission, but retains each parent and up to two
/// children until publication.
const MAX_ADMISSION_WAVE_RELU_ELEMENTS: usize = 4 * 1024 * 1024;
const MAX_ADMISSION_UNSTABLE_METADATA_BYTES: usize = 256 * 1024 * 1024;
const MAX_ADMISSION_WAVE_COEFFICIENT_ELEMENTS: usize = MAX_CPU_BUFFER_BYTES / size_of::<f32>();
const MAX_RETAINED_DOMAINS_PER_PARENT: usize = 3;
const MAX_BOUNDED_FRONTIER_BYTES: usize = 512 * 1024 * 1024;
/// Aggregate live-product ceiling for one bounded Linear backward. This is
/// separate from the 256 MiB per-allocation cap: the certified implementation
/// stacks lower+upper rows, retains two 2R×P f64 products, and also owns
/// persistent f32 coefficient/error matrices and temporary propagated errors.
const MAX_BOUNDED_LINEAR_TRANSIENT_BYTES: usize = 512 * 1024 * 1024;
const MAX_BOUNDED_FRONTIER_DOMAINS: usize = 4_096;
/// Bound cloned split-history and beta-entry state that is outside GEMM.
/// Parents at this depth are retained as unresolved instead of creating a
/// deeper child whose mandatory state would exceed admission accounting.
pub(super) const MAX_BOUNDED_SHARED_HISTORY_CONSTRAINTS: usize = 128;
/// Conservative combined multiplier for one constraint's history record,
/// beta record, delta spelling, and the separately allocated nested lookup
/// tables/indices (including spare hash buckets and allocator overhead).
const ADMISSION_HISTORY_RECORD_MULTIPLIER: usize = 16;

/// Resolve a typed policy with the treatment's exact environment override.
///
/// An absent value inherits the typed config, literal `1` enables, and every
/// other present byte string disables.
#[inline]
pub(super) fn resolve_gate(typed_enabled: bool, raw_env: Option<&OsStr>) -> bool {
    raw_env.map_or(typed_enabled, |raw| raw == OsStr::new("1"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Admission {
    NotArmed,
    RefusedCallerEngine,
    RefusedPostRootEngine,
    RefusedUnsupportedCallerMode,
    RefusedExpired,
    RefusedUnsupportedGraph,
    RefusedBackendUnavailable,
    RefusedUnsoundBackend,
    RefusedInvalidCapacity(usize),
    Accepted { capacity: usize },
}

impl Admission {
    /// Emit a stable treatment-engagement marker. Dark runs emit nothing.
    pub(super) fn report(self) {
        match self {
            Self::NotArmed => {}
            Self::Accepted { capacity } => {
                eprintln!("[mo-bounded-shared-executor] status=accepted capacity={capacity}");
            }
            Self::RefusedInvalidCapacity(capacity) => {
                eprintln!(
                    "[mo-bounded-shared-executor] status=refused \
                     reason=invalid-capability capacity={capacity}"
                );
            }
            refusal => {
                let reason = match refusal {
                    Self::RefusedCallerEngine => "caller-engine",
                    Self::RefusedPostRootEngine => "post-root-engine",
                    Self::RefusedUnsupportedCallerMode => "unsupported-caller-mode",
                    Self::RefusedExpired => "deadline-expired",
                    Self::RefusedUnsupportedGraph => "unsupported-graph",
                    Self::RefusedBackendUnavailable => "backend-unavailable",
                    Self::RefusedUnsoundBackend => "unsound-backend",
                    Self::NotArmed | Self::RefusedInvalidCapacity(_) | Self::Accepted { .. } => {
                        unreachable!("handled above")
                    }
                };
                eprintln!("[mo-bounded-shared-executor] status=refused reason={reason} capacity=0");
            }
        }
    }

    #[must_use]
    pub(super) const fn accepted_capacity(self) -> Option<usize> {
        match self {
            Self::Accepted { capacity } => Some(capacity),
            _ => None,
        }
    }
}

/// Fail-closed structural and metadata prefilter for the bounded β-CROWN lane.
///
/// The process-global bounded backend supports a clean dense ResNet
/// decomposition, not an arbitrary graph that happens to contain a Conv2d and
/// an Add. Admission therefore mirrors the extractor's topology: a backward
/// walk from the output may contain only audited unary layers or exact
/// two-input Add diamonds whose branches are disjoint unary chains to their
/// topologically latest common ancestor. Every graph node must be on that
/// output ancestry, every root tensor is rank/element capped, residual Add
/// shapes must match exactly (no broadcasting), and parameter buffers are
/// capped.
///
/// This is still only a prefilter. The per-child dynamic extractor remains the
/// final authority and may refuse based on concrete bounds, layout, or alpha
/// state. A false negative loses only the default-dark optimization.
#[inline]
pub(super) fn graph_may_support_bounded_beta<'a>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: impl Into<NodeBoundsView<'a>>,
) -> bool {
    bounded_resnet_metadata_supported(graph, input, node_bounds.into())
        && bounded_resnet_topology_supported(graph)
}

/// Bound every workload-dependent allocation that precedes the facade's first
/// GEMM poll. Root verification is monotone, so the root's still-active
/// objective count is an upper bound on every later union-pruned spec matrix.
/// The admitted executor also caps each outer wave to at most K=8 domains.
pub(super) fn workload_may_support_bounded_beta<'a>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: impl Into<NodeBoundsView<'a>>,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    root_verified: &[bool],
) -> bool {
    let node_bounds = node_bounds.into();
    if objectives.is_empty()
        || objectives.len() > MAX_ADMISSION_OBJECTIVES
        || thresholds.len() != objectives.len()
        || root_verified.len() != objectives.len()
    {
        return false;
    }
    let Some(output_bounds) = node_bounds.get(graph.output_node.as_str()) else {
        return false;
    };
    let output_dim = output_bounds.len();
    if output_dim == 0
        || crate::network::crown_memory::identity_pair_bytes(output_dim)
            .is_none_or(|bytes| bytes > MAX_CPU_BUFFER_BYTES)
        || objectives
            .iter()
            .any(|objective| objective.len() != output_dim)
        || objectives
            .len()
            .checked_mul(output_dim)
            .is_none_or(|elements| elements > MAX_ADMISSION_OBJECTIVE_ELEMENTS)
    {
        return false;
    }

    let active_rows = root_verified.iter().filter(|&&verified| !verified).count();
    if active_rows == 0 {
        return false;
    }
    // A split wave carries only the root-active objective rows, but a domain
    // with no unstable ReLU runs the full-output identity CROWN path. Admit
    // Conv2d scratch for the larger of those two mandatory row counts.
    let max_crown_rows = active_rows.max(output_dim);
    if !bounded_conv_f64_buffers_supported(graph, input, node_bounds, max_crown_rows) {
        return false;
    }
    let retained_domains =
        DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS.checked_mul(MAX_RETAINED_DOMAINS_PER_PARENT);
    let max_identifier_bytes = max_graph_identifier_bytes(graph);
    let retained_history_bytes = size_of::<(String, usize)>()
        .checked_add(max_identifier_bytes)
        .and_then(|entry| entry.checked_mul(ADMISSION_HISTORY_RECORD_MULTIPLIER))
        .and_then(|entry| entry.checked_mul(MAX_BOUNDED_SHARED_HISTORY_CONSTRAINTS))
        .and_then(|per_domain| {
            retained_domains.and_then(|domains| domains.checked_mul(per_domain))
        });
    let max_tensor_elements = node_bounds
        .values()
        .map(|bounds| bounds.len())
        .chain(std::iter::once(input.len()))
        .max()
        .unwrap_or(0);
    if crate::network::crown_memory::dense_pair_bytes(output_dim, max_tensor_elements)
        .is_none_or(|bytes| bytes > MAX_CPU_BUFFER_BYTES)
        || bounded_linear_product_buffer_bytes(max_crown_rows, max_tensor_elements)
            .is_none_or(|bytes| bytes > MAX_CPU_BUFFER_BYTES)
        || bounded_linear_transient_peak_bytes(max_crown_rows, max_tensor_elements)
            .is_none_or(|bytes| bytes > MAX_BOUNDED_LINEAR_TRANSIENT_BYTES)
    {
        return false;
    }
    let coefficient_elements = retained_domains
        .and_then(|domains| domains.checked_mul(active_rows))
        .and_then(|rows| rows.checked_mul(max_tensor_elements));
    if coefficient_elements
        .is_none_or(|elements| elements > MAX_ADMISSION_WAVE_COEFFICIENT_ELEMENTS)
    {
        return false;
    }

    let relu_metadata = graph
        .nodes
        .iter()
        .filter(|(_, node)| matches!(node.layer, Layer::ReLU(_)))
        .try_fold((0usize, 0usize), |(elements, bytes), (name, _)| {
            let neurons = node_bounds.get(name)?.len();
            let elements = elements.checked_add(neurons)?;
            let bytes_per_entry = size_of::<(String, usize)>().checked_add(name.len())?;
            let bytes = bytes.checked_add(neurons.checked_mul(bytes_per_entry)?)?;
            Some((elements, bytes))
        });
    relu_metadata.is_some_and(|(elements, unstable_bytes)| {
        // Unstable vectors are built only for the K popped parents. Children
        // use compact heuristic alpha and do not retain discovery lists.
        let wave_unstable_bytes =
            DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS.checked_mul(unstable_bytes);
        DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
            .checked_mul(elements)
            .is_some_and(|total| total <= MAX_ADMISSION_WAVE_RELU_ELEMENTS)
            && wave_unstable_bytes
                .and_then(|subtotal| {
                    retained_history_bytes.and_then(|history| subtotal.checked_add(history))
                })
                .is_some_and(|total| total <= MAX_ADMISSION_UNSTABLE_METADATA_BYTES)
    }) && bounded_frontier_domain_limit(graph, input, node_bounds, objectives.len()).is_some()
}

fn bounded_conv_f64_buffers_supported(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: NodeBoundsView<'_>,
    crown_rows: usize,
) -> bool {
    let max_elements = crate::layers::convolution::conv2d::DEADLINE_BOUNDED_CONV_HOST_BUFFER_BYTES
        / size_of::<f64>();
    graph.nodes.iter().all(|(name, node)| {
        let Layer::Conv2d(conv) = &node.layer else {
            return true;
        };
        let Ok(input_name) = node.require_unary_input() else {
            return false;
        };
        let Some(input_shape) = shape_for(input_name, input, node_bounds) else {
            return false;
        };
        let Some(output_shape) = node_bounds.get(name).map(|bounds| bounds.shape()) else {
            return false;
        };
        if input_shape.len() != 3 || output_shape.len() != 3 || conv.groups == 0 {
            return false;
        }
        let [in_c, in_h, in_w] = [input_shape[0], input_shape[1], input_shape[2]];
        let [out_c, out_h, out_w] = [output_shape[0], output_shape[1], output_shape[2]];
        let (kh, kw) = conv.kernel_size();
        let Some(total_spatial) = crown_rows
            .checked_mul(out_h)
            .and_then(|value| value.checked_mul(out_w))
        else {
            return false;
        };
        let Some(kernel_cols) = in_c.checked_mul(kh).and_then(|value| value.checked_mul(kw)) else {
            return false;
        };
        [
            total_spatial.checked_mul(out_c),
            out_c.checked_mul(kernel_cols),
            total_spatial.checked_mul(kernel_cols),
            crown_rows
                .checked_mul(in_c)
                .and_then(|value| value.checked_mul(in_h))
                .and_then(|value| value.checked_mul(in_w)),
        ]
        .into_iter()
        .all(|elements| elements.is_some_and(|elements| elements <= max_elements))
    })
}

fn max_graph_identifier_bytes(graph: &GraphNetwork) -> usize {
    graph
        .nodes
        .iter()
        .flat_map(|(name, node)| {
            std::iter::once(name.len()).chain(node.inputs.iter().map(String::len))
        })
        .chain(std::iter::once(graph.output_node.len()))
        .max()
        .unwrap_or(0)
}

/// Derive a hard total-domain limit from this graph's actual retained state.
///
/// Bounded children discard dense lA/per-disjunct/shared-alpha warm starts, but
/// each frontier domain still owns a node-bound map, objective bookkeeping,
/// and up to the audited split-history cap. Reserve four full K=8 waves above
/// the returned limit. The popped parents remain live while a binary split
/// holds as many as 2K child domain shells alongside 2K freshly propagated
/// node-cache maps before publication.
pub(super) fn bounded_frontier_domain_limit<'a>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: impl Into<NodeBoundsView<'a>>,
    objective_count: usize,
) -> Option<usize> {
    let node_bounds = node_bounds.into();
    let tensor_elements = node_bounds
        .values()
        .try_fold(input.len(), |total, bounds| total.checked_add(bounds.len()))?;
    let tensor_bytes = tensor_elements
        .checked_mul(2)?
        .checked_mul(size_of::<f32>())?;
    let map_bytes = graph.nodes.iter().try_fold(0usize, |total, (name, _)| {
        total.checked_add(name.len().checked_add(64)?)
    })?;
    let history_bytes = size_of::<(String, usize)>()
        .checked_add(max_graph_identifier_bytes(graph))?
        .checked_mul(ADMISSION_HISTORY_RECORD_MULTIPLIER)?
        .checked_mul(MAX_BOUNDED_SHARED_HISTORY_CONSTRAINTS)?;
    let objective_slot_bytes = size_of::<(f32, f32)>()
        + size_of::<bool>()
        + size_of::<Option<Arc<crate::batched_domain::CachedLinearBounds>>>();
    let objective_bytes = objective_count.checked_mul(objective_slot_bytes)?;
    let per_domain_bytes = tensor_bytes
        .checked_add(map_bytes)?
        .checked_add(history_bytes)?
        .checked_add(objective_bytes)?
        .checked_add(4 * 1024)?;
    if per_domain_bytes == 0 {
        return None;
    }
    let resident_capacity = MAX_BOUNDED_FRONTIER_BYTES / per_domain_bytes;
    let transient_wave_domains = DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS.checked_mul(4)?;
    if resident_capacity < transient_wave_domains {
        return None;
    }
    let retained_limit = resident_capacity.checked_sub(transient_wave_domains)?;
    (retained_limit >= 2).then_some(retained_limit.min(MAX_BOUNDED_FRONTIER_DOMAINS))
}

fn bounded_tensor_metadata_supported(bounds: &BoundedTensor) -> bool {
    !bounds.shape().is_empty()
        && bounds.shape().len() <= MAX_ADMISSION_TENSOR_RANK
        && !bounds.is_empty()
        && bounds.len() <= MAX_ADMISSION_TENSOR_ELEMENTS
}

/// Compare loader-declared metadata with the concrete runtime tensor shape.
///
/// Some loaders omit a single fixed batch dimension from declared shapes while
/// propagation retains it. Accept exactly that representation difference, in
/// that direction only. This is deliberately stricter than product equality or
/// broadcast compatibility: all layer transitions continue to use and validate
/// the concrete runtime shapes below.
#[inline]
fn declared_runtime_shape_compatible(declared: &[usize], runtime: &[usize]) -> bool {
    declared == runtime || runtime.strip_prefix(&[1]) == Some(declared)
}

fn shape_for<'a>(
    name: &str,
    input: &'a BoundedTensor,
    node_bounds: NodeBoundsView<'a>,
) -> Option<&'a [usize]> {
    if name == NETWORK_INPUT {
        Some(input.shape())
    } else {
        node_bounds.get(name).map(|bounds| bounds.shape())
    }
}

fn audited_unary_layer_supported(
    node: &GraphNode,
    input_shape: &[usize],
    output_shape: &[usize],
) -> bool {
    if node.inputs.len() != 1 {
        return false;
    }
    match &node.layer {
        Layer::Linear(linear) => {
            if linear.weight.len() > MAX_ADMISSION_PARAMETER_ELEMENTS
                || linear
                    .bias
                    .as_ref()
                    .is_some_and(|bias| bias.len() > MAX_ADMISSION_PARAMETER_ELEMENTS)
                || input_shape.is_empty()
                || output_shape.is_empty()
            {
                return false;
            }
            let input_last = input_shape[input_shape.len() - 1];
            let output_last = output_shape[output_shape.len() - 1];
            input_last == linear.in_features()
                && output_last == linear.out_features()
                && input_shape[..input_shape.len() - 1] == output_shape[..output_shape.len() - 1]
        }
        Layer::Conv2d(conv) => {
            if conv.groups != 1
                || conv.dilation != (1, 1)
                || conv.kernel.len() > MAX_ADMISSION_PARAMETER_ELEMENTS
                || conv
                    .bias
                    .as_ref()
                    .is_some_and(|bias| bias.len() > MAX_ADMISSION_PARAMETER_ELEMENTS)
                || input_shape.len() != 3
                || output_shape.len() != 3
                || input_shape[0] != conv.in_channels()
                || output_shape[0] != conv.out_channels()
            {
                return false;
            }
            conv.output_size(input_shape[1], input_shape[2])
                .is_ok_and(|(height, width)| output_shape[1..] == [height, width])
        }
        Layer::ReLU(_) => input_shape == output_shape,
        Layer::Flatten(flatten) => flatten
            .compute_output_shape(input_shape)
            .is_ok_and(|shape| shape == output_shape),
        Layer::Reshape(reshape) => {
            reshape.target_shape.len() <= MAX_ADMISSION_TENSOR_RANK
                && reshape
                    .compute_output_shape(input_shape)
                    .is_ok_and(|shape| shape == output_shape)
        }
        _ => false,
    }
}

fn checked_parameter_sum(total: usize, additional: usize) -> Option<usize> {
    total
        .checked_add(additional)
        .filter(|&sum| sum <= MAX_ADMISSION_TOTAL_PARAMETER_ELEMENTS)
}

fn bounded_linear_product_buffer_bytes(
    crown_rows: usize,
    max_tensor_elements: usize,
) -> Option<usize> {
    crown_rows
        .checked_mul(max_tensor_elements)?
        .checked_mul(2)?
        .checked_mul(size_of::<f64>())
}

fn bounded_linear_transient_peak_bytes(
    crown_rows: usize,
    max_tensor_elements: usize,
) -> Option<usize> {
    // Conservative live-state model for propagate_linear_cpu_with_deadline:
    // two 2R×P f64 products (32·R·P bytes), four persistent R×P f32
    // coefficient/error matrices (16·R·P), and at most either the stacked
    // 2R×K f32 input or two R×P propagated-error matrices (8·R·max(K,P)).
    // Charge 64·R·max(K,P) to cover all 56 bytes plus bookkeeping slack.
    crown_rows
        .checked_mul(max_tensor_elements)?
        .checked_mul(8)?
        .checked_mul(size_of::<f64>())
}

fn checked_parameter_total(total: usize, node: &GraphNode) -> Option<usize> {
    let additional = match &node.layer {
        Layer::Linear(linear) => linear
            .weight
            .len()
            .checked_add(linear.bias.as_ref().map_or(0, |bias| bias.len()))?,
        Layer::Conv2d(conv) => conv
            .kernel
            .len()
            .checked_add(conv.bias.as_ref().map_or(0, |bias| bias.len()))?,
        _ => 0,
    };
    checked_parameter_sum(total, additional)
}

/// Bound all graph-owned identifier metadata before any topology helper clones
/// strings or asks `exec_order` to sort them. Declared-shape entries are also
/// constrained to known graph identifiers so a later admitted graph clone
/// cannot copy an unrelated, unbounded metadata map.
fn bounded_identifier_metadata_supported(graph: &GraphNetwork) -> bool {
    if graph.node_order.len() != graph.nodes.len()
        || graph.declared_shapes.len() > graph.nodes.len().saturating_add(1)
        || graph.output_node.len() > MAX_ADMISSION_SINGLE_IDENTIFIER_BYTES
    {
        return false;
    }

    // Count edges using only O(1) vector lengths before walking any input
    // spelling. This makes the subsequent string scan iteration-bounded.
    let mut edges = 0usize;
    for node in graph.nodes.values() {
        edges = match edges.checked_add(node.inputs.len()) {
            Some(sum) if sum <= MAX_ADMISSION_GRAPH_EDGES => sum,
            _ => return false,
        };
    }

    let mut bytes = graph.output_node.len();
    for (key, node) in &graph.nodes {
        if key.len() > MAX_ADMISSION_SINGLE_IDENTIFIER_BYTES
            || node.name.len() > MAX_ADMISSION_SINGLE_IDENTIFIER_BYTES
        {
            return false;
        }
        bytes = match bytes
            .checked_add(key.len())
            .and_then(|sum| sum.checked_add(node.name.len()))
        {
            Some(sum) if sum <= MAX_ADMISSION_IDENTIFIER_BYTES => sum,
            _ => return false,
        };
        // Charge both spellings before comparing their potentially long bytes.
        if key != &node.name {
            return false;
        }
        for input_name in &node.inputs {
            if input_name.len() > MAX_ADMISSION_SINGLE_IDENTIFIER_BYTES {
                return false;
            }
            bytes = match bytes.checked_add(input_name.len()) {
                Some(sum) if sum <= MAX_ADMISSION_IDENTIFIER_BYTES => sum,
                _ => return false,
            };
        }
    }
    let mut ordered_nodes = HashSet::with_capacity(graph.node_order.len());
    for name in &graph.node_order {
        if name.len() > MAX_ADMISSION_SINGLE_IDENTIFIER_BYTES {
            return false;
        }
        bytes = match bytes.checked_add(name.len()) {
            Some(sum) if sum <= MAX_ADMISSION_IDENTIFIER_BYTES => sum,
            _ => return false,
        };
        if !graph.nodes.contains_key(name) || !ordered_nodes.insert(name.as_str()) {
            return false;
        }
    }
    for (name, shape) in &graph.declared_shapes {
        if name.len() > MAX_ADMISSION_SINGLE_IDENTIFIER_BYTES {
            return false;
        }
        bytes = match bytes.checked_add(name.len()) {
            Some(sum) if sum <= MAX_ADMISSION_IDENTIFIER_BYTES => sum,
            _ => return false,
        };
        if name != NETWORK_INPUT && !graph.nodes.contains_key(name) {
            return false;
        }
        if shape.is_empty() || shape.len() > MAX_ADMISSION_TENSOR_RANK {
            return false;
        }
    }
    true
}

fn bounded_resnet_metadata_supported(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: NodeBoundsView<'_>,
) -> bool {
    if graph.nodes.is_empty()
        || graph.nodes.len() > MAX_ADMISSION_GRAPH_NODES
        || node_bounds.len() != graph.nodes.len()
        || !bounded_tensor_metadata_supported(input)
        || !bounded_identifier_metadata_supported(graph)
    {
        return false;
    }
    if let Some(declared_input) = graph.declared_shape(NETWORK_INPUT) {
        if !declared_runtime_shape_compatible(declared_input, input.shape()) {
            return false;
        }
    }

    let mut total_elements = input.len();
    let mut total_parameter_elements = 0usize;
    let mut edges = 0usize;
    for (name, node) in &graph.nodes {
        edges = match edges.checked_add(node.inputs.len()) {
            Some(edges) if edges <= MAX_ADMISSION_GRAPH_EDGES => edges,
            _ => return false,
        };
        let Some(bounds) = node_bounds.get(name) else {
            return false;
        };
        if !bounded_tensor_metadata_supported(bounds) {
            return false;
        }
        total_elements = match total_elements.checked_add(bounds.len()) {
            Some(total) if total <= MAX_ADMISSION_TOTAL_ELEMENTS => total,
            _ => return false,
        };
        total_parameter_elements = match checked_parameter_total(total_parameter_elements, node) {
            Some(total) => total,
            None => return false,
        };
        if let Some(declared) = graph.declared_shape(name) {
            if !declared_runtime_shape_compatible(declared, bounds.shape()) {
                return false;
            }
        }

        match &node.layer {
            Layer::Add(_) => {
                if node.inputs.len() != 2 {
                    return false;
                }
                let Some(left) = shape_for(&node.inputs[0], input, node_bounds) else {
                    return false;
                };
                let Some(right) = shape_for(&node.inputs[1], input, node_bounds) else {
                    return false;
                };
                if left != right || left != bounds.shape() {
                    return false;
                }
            }
            _ => {
                let Ok(input_name) = node.require_unary_input() else {
                    return false;
                };
                let Some(input_shape) = shape_for(input_name, input, node_bounds) else {
                    return false;
                };
                if !audited_unary_layer_supported(node, input_shape, bounds.shape()) {
                    return false;
                }
            }
        }
    }
    graph.nodes.contains_key(graph.output_node.as_str())
}

fn ancestor_set(graph: &GraphNetwork, start: &str) -> Option<HashSet<String>> {
    let mut ancestors = HashSet::with_capacity(graph.nodes.len().saturating_add(1));
    let mut pending = vec![start.to_string()];
    while let Some(name) = pending.pop() {
        if !ancestors.insert(name.clone()) || name == NETWORK_INPUT {
            continue;
        }
        let node = graph.nodes.get(name.as_str())?;
        pending.extend(node.inputs.iter().cloned());
        if ancestors.len().saturating_add(pending.len())
            > MAX_ADMISSION_GRAPH_NODES.saturating_add(MAX_ADMISSION_GRAPH_EDGES)
        {
            return None;
        }
    }
    Some(ancestors)
}

fn latest_common_ancestor(
    graph: &GraphNetwork,
    execution_order: &[String],
    left: &str,
    right: &str,
) -> Option<String> {
    if left == NETWORK_INPUT || right == NETWORK_INPUT {
        return Some(NETWORK_INPUT.to_string());
    }
    let left_ancestors = ancestor_set(graph, left)?;
    let right_ancestors = ancestor_set(graph, right)?;
    execution_order
        .iter()
        .rev()
        .find(|name| {
            left_ancestors.contains(name.as_str()) && right_ancestors.contains(name.as_str())
        })
        .cloned()
        .or_else(|| {
            (left_ancestors.contains(NETWORK_INPUT) && right_ancestors.contains(NETWORK_INPUT))
                .then(|| NETWORK_INPUT.to_string())
        })
}

fn walk_audited_unary_branch(
    graph: &GraphNetwork,
    start: &str,
    boundary: &str,
    visited: &mut HashSet<String>,
    saw_conv2d: &mut bool,
) -> Option<usize> {
    let mut current = start.to_string();
    let mut steps = 0usize;
    while current != boundary {
        if current == NETWORK_INPUT || steps >= graph.nodes.len() {
            return None;
        }
        if !visited.insert(current.clone()) {
            return None;
        }
        let node = graph.nodes.get(current.as_str())?;
        if node.inputs.len() != 1 || matches!(node.layer, Layer::Add(_)) {
            return None;
        }
        if matches!(node.layer, Layer::Conv2d(_)) {
            *saw_conv2d = true;
        }
        current = node.require_unary_input().ok()?.to_string();
        steps += 1;
    }
    Some(steps)
}

fn bounded_resnet_topology_supported(graph: &GraphNetwork) -> bool {
    let Ok(execution_order) = graph.exec_order() else {
        return false;
    };
    if execution_order.len() != graph.nodes.len() {
        return false;
    }

    let mut current = graph.output_node.clone();
    let mut visited = HashSet::with_capacity(graph.nodes.len());
    let mut saw_add = false;
    let mut saw_conv2d = false;
    let mut steps = 0usize;
    while current != NETWORK_INPUT {
        if steps >= graph.nodes.len() || !visited.insert(current.clone()) {
            return false;
        }
        let Some(node) = graph.nodes.get(current.as_str()) else {
            return false;
        };
        match node.inputs.len() {
            1 if !matches!(node.layer, Layer::Add(_)) => {
                if matches!(node.layer, Layer::Conv2d(_)) {
                    saw_conv2d = true;
                }
                current = node.inputs[0].clone();
            }
            2 if matches!(node.layer, Layer::Add(_)) => {
                let left = node.inputs[0].as_str();
                let right = node.inputs[1].as_str();
                let Some(boundary) = latest_common_ancestor(graph, execution_order, left, right)
                else {
                    return false;
                };
                if boundary == left || boundary == right {
                    let branch = if boundary == left { right } else { left };
                    if walk_audited_unary_branch(
                        graph,
                        branch,
                        &boundary,
                        &mut visited,
                        &mut saw_conv2d,
                    )
                    .is_none_or(|length| length == 0)
                    {
                        return false;
                    }
                } else {
                    for branch in [left, right] {
                        if walk_audited_unary_branch(
                            graph,
                            branch,
                            &boundary,
                            &mut visited,
                            &mut saw_conv2d,
                        )
                        .is_none_or(|length| length == 0)
                        {
                            return false;
                        }
                    }
                }
                saw_add = true;
                current = boundary;
            }
            _ => return false,
        }
        steps += 1;
    }

    saw_add && saw_conv2d && visited.len() == graph.nodes.len()
}

/// Decide whether the CPU facade may activate.
///
/// Prerequisites are ordered so dark, caller-owned, already-routed, and expired
/// calls do not inspect graph structure or observe the process-global backend
/// slot. `backend` must be a get-only accessor in production.
pub(super) fn admit<'backend>(
    armed: bool,
    caller_engine_present: bool,
    post_root_engine_present: bool,
    shared_executor_eligible: bool,
    deadline: Instant,
    mut now: impl FnMut() -> Instant,
    graph_supported: impl FnOnce() -> bool,
    backend: impl FnOnce() -> Option<&'backend dyn GpuCrownBackward>,
) -> Admission {
    if !armed {
        return Admission::NotArmed;
    }
    if caller_engine_present {
        return Admission::RefusedCallerEngine;
    }
    if post_root_engine_present {
        return Admission::RefusedPostRootEngine;
    }
    if !shared_executor_eligible {
        return Admission::RefusedUnsupportedCallerMode;
    }
    if now() >= deadline {
        return Admission::RefusedExpired;
    }
    if !graph_supported() {
        return Admission::RefusedUnsupportedGraph;
    }
    if now() >= deadline {
        return Admission::RefusedExpired;
    }
    let Some(gpu) = backend() else {
        return Admission::RefusedBackendUnavailable;
    };
    if !gpu.provides_sound_gpu_crown() {
        return Admission::RefusedUnsoundBackend;
    }
    let capacity = gpu.deadline_bounded_resnet_sound_beta_max_rows();
    if !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&capacity) {
        return Admission::RefusedInvalidCapacity(capacity);
    }
    if now() >= deadline {
        return Admission::RefusedExpired;
    }
    Admission::Accepted { capacity }
}

/// A local CPU engine whose entire implemented surface polls one immutable BaB
/// deadline. It deliberately exposes no GPU traits and no broad accelerator
/// deadline capability.
pub(super) struct DeadlineCpuGemmEngine<C = fn() -> Instant> {
    deadline: Instant,
    now: C,
}

impl DeadlineCpuGemmEngine<fn() -> Instant> {
    #[must_use]
    pub(super) fn new(deadline: Instant) -> Self {
        Self {
            deadline,
            now: Instant::now,
        }
    }
}

impl<C> DeadlineCpuGemmEngine<C>
where
    C: Fn() -> Instant + Sync + Send,
{
    #[cfg(test)]
    fn with_clock(deadline: Instant, now: C) -> Self {
        Self { deadline, now }
    }

    #[inline]
    fn poll(&self) -> Result<()> {
        if (self.now)() >= self.deadline {
            Err(NyError::DeadlineExceeded(
                "bounded shared-executor CPU GEMM deadline exceeded".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn zeroed_f32(&self, elements: usize, context: &'static str) -> Result<Vec<f32>> {
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| NyError::InvalidSpec(format!("{context}: byte size overflow")))?;
        if bytes > MAX_CPU_BUFFER_BYTES {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: bytes,
                budget_bytes: MAX_CPU_BUFFER_BYTES,
                site: context,
            });
        }
        self.poll()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: bytes,
                budget_bytes: MAX_CPU_BUFFER_BYTES,
                site: context,
            })?;
        while values.len() < elements {
            self.poll()?;
            let next_len = values
                .len()
                .saturating_add(CPU_INIT_CHUNK_ELEMENTS)
                .min(elements);
            values.resize(next_len, 0.0);
        }
        self.poll()?;
        Ok(values)
    }

    fn zeroed_f64(&self, elements: usize, context: &'static str) -> Result<Vec<f64>> {
        let bytes = elements
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| NyError::InvalidSpec(format!("{context}: byte size overflow")))?;
        if bytes > MAX_CPU_BUFFER_BYTES {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: bytes,
                budget_bytes: MAX_CPU_BUFFER_BYTES,
                site: context,
            });
        }
        self.poll()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: bytes,
                budget_bytes: MAX_CPU_BUFFER_BYTES,
                site: context,
            })?;
        while values.len() < elements {
            self.poll()?;
            let next_len = values
                .len()
                .saturating_add(CPU_INIT_CHUNK_ELEMENTS)
                .min(elements);
            values.resize(next_len, 0.0);
        }
        self.poll()?;
        Ok(values)
    }

    #[inline]
    fn poll_work(&self, countdown: &mut usize) -> Result<()> {
        if *countdown == 0 {
            self.poll()?;
            *countdown = CPU_POLL_INTERVAL;
        }
        *countdown -= 1;
        Ok(())
    }
}

impl<C> GemmEngine for DeadlineCpuGemmEngine<C>
where
    C: Fn() -> Instant + Sync + Send,
{
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.poll()?;
        let a_len = checked_dim_product(&[m, k], "bounded shared GEMM f32 lhs")?;
        let b_len = checked_dim_product(&[k, n], "bounded shared GEMM f32 rhs")?;
        let output_len = checked_dim_product(&[m, n], "bounded shared GEMM f32 output")?;
        if a.len() != a_len {
            return Err(NyError::InvalidSpec(format!(
                "bounded shared GEMM f32: a.len()={} != m*k={}*{}={a_len}",
                a.len(),
                m,
                k
            )));
        }
        if b.len() != b_len {
            return Err(NyError::InvalidSpec(format!(
                "bounded shared GEMM f32: b.len()={} != k*n={}*{}={b_len}",
                b.len(),
                k,
                n
            )));
        }

        let mut output = self.zeroed_f32(output_len, "bounded shared GEMM f32 output")?;
        let mut poll_countdown = 0usize;
        for row in 0..m {
            self.poll()?;
            for col in 0..n {
                // Poll independently of the contraction width so `k == 0`
                // cannot turn a huge output row into one unobserved loop.
                self.poll_work(&mut poll_countdown)?;
                let mut sum = 0.0_f32;
                for inner in 0..k {
                    self.poll_work(&mut poll_countdown)?;
                    sum += a[row * k + inner] * b[inner * n + col];
                }
                output[row * n + col] = sum;
            }
        }
        self.poll()?;
        Ok(output)
    }

    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        self.poll()?;
        let a_len = checked_dim_product(&[m, k], "bounded shared GEMM f64 lhs")?;
        let b_len = checked_dim_product(&[k, n], "bounded shared GEMM f64 rhs")?;
        let output_len = checked_dim_product(&[m, n], "bounded shared GEMM f64 output")?;
        if a.len() != a_len {
            return Err(NyError::InvalidSpec(format!(
                "bounded shared GEMM f64: a.len()={} != m*k={}*{}={a_len}",
                a.len(),
                m,
                k
            )));
        }
        if b.len() != b_len {
            return Err(NyError::InvalidSpec(format!(
                "bounded shared GEMM f64: b.len()={} != k*n={}*{}={b_len}",
                b.len(),
                k,
                n
            )));
        }

        let mut output = self.zeroed_f64(output_len, "bounded shared GEMM f64 output")?;
        let mut poll_countdown = 0usize;
        for row in 0..m {
            self.poll()?;
            for col in 0..n {
                self.poll_work(&mut poll_countdown)?;
                let mut sum = 0.0_f64;
                for inner in 0..k {
                    self.poll_work(&mut poll_countdown)?;
                    sum += a[row * k + inner] * b[inner * n + col];
                }
                output[row * n + col] = sum;
            }
        }
        self.poll()?;
        Ok(output)
    }

    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        deadline: Instant,
        _max_dispatch_macs: usize,
    ) -> Result<Vec<f64>> {
        // The facade's immutable BaB deadline must be at least as strict as
        // the requested call-local authority. Returning a typed deadline error
        // early is safer than executing under a later deadline.
        if deadline < self.deadline {
            return Err(NyError::DeadlineExceeded(
                "bounded shared GEMM f64 requested an earlier deadline".into(),
            ));
        }
        self.gemm_f64(m, k, n, a, b)
    }

    fn conv_transpose_2d(
        &self,
        a_reshaped: &[f32],
        weight_col: &[f32],
        params: &ConvTranspose2dParams,
    ) -> Result<Vec<f32>> {
        self.poll()?;
        let s = params.num_specs;
        let oc = params.out_channels;
        let ic = params.in_channels;
        let (oh, ow) = (params.out_h, params.out_w);
        let (ih, iw) = (params.in_h, params.in_w);
        let (kh, kw) = (params.kernel_h, params.kernel_w);
        let (sh, sw) = (params.stride_h, params.stride_w);
        let (ph, pw) = (params.pad_h, params.pad_w);

        let spatial = checked_dim_product(&[oh, ow], "bounded shared conv output spatial")?;
        let total_rows = checked_dim_product(&[s, spatial], "bounded shared conv GEMM rows")?;
        let kernel_cols = checked_dim_product(&[ic, kh, kw], "bounded shared conv kernel columns")?;
        let a_len = checked_dim_product(&[total_rows, oc], "bounded shared conv lhs")?;
        let weight_len = checked_dim_product(&[oc, kernel_cols], "bounded shared conv weights")?;
        if a_reshaped.len() != a_len {
            return Err(NyError::InvalidSpec(format!(
                "bounded shared conv_transpose_2d: a_reshaped.len()={} != \
                 S*OH*OW*OC={a_len}",
                a_reshaped.len()
            )));
        }
        if weight_col.len() != weight_len {
            return Err(NyError::InvalidSpec(format!(
                "bounded shared conv_transpose_2d: weight_col.len()={} != \
                 OC*IC*KH*KW={weight_len}",
                weight_col.len()
            )));
        }

        let flat_input_dim = checked_dim_product(&[ic, ih, iw], "bounded shared conv flat input")?;
        let result_len = checked_dim_product(&[s, flat_input_dim], "bounded shared conv output")?;
        let _max_y_padded = oh
            .saturating_sub(1)
            .checked_mul(sh)
            .and_then(|value| value.checked_add(kh.saturating_sub(1)))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "bounded shared conv_transpose_2d: vertical coordinate overflow".into(),
                )
            })?;
        let _max_x_padded = ow
            .saturating_sub(1)
            .checked_mul(sw)
            .and_then(|value| value.checked_add(kw.saturating_sub(1)))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "bounded shared conv_transpose_2d: horizontal coordinate overflow".into(),
                )
            })?;

        if total_rows == 0 || oc == 0 || kernel_cols == 0 {
            let output = self.zeroed_f32(result_len, "bounded shared conv output")?;
            self.poll()?;
            return Ok(output);
        }

        let gemm_out = self.gemm_f32(total_rows, oc, kernel_cols, a_reshaped, weight_col)?;
        let mut result = self.zeroed_f32(result_len, "bounded shared conv output")?;
        let mut poll_countdown = 0usize;
        for spec in 0..s {
            self.poll()?;
            for gy in 0..oh {
                for gx in 0..ow {
                    let gemm_row = spec * spatial + gy * ow + gx;
                    for ic_idx in 0..ic {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                self.poll_work(&mut poll_countdown)?;
                                let padded_y = gy * sh + ki;
                                let padded_x = gx * sw + kj;
                                if padded_y < ph || padded_x < pw {
                                    continue;
                                }
                                let iy = padded_y - ph;
                                let ix = padded_x - pw;
                                if iy >= ih || ix >= iw {
                                    continue;
                                }
                                let col_idx = ic_idx * kh * kw + ki * kw + kj;
                                let out_idx = ic_idx * ih * iw + iy * iw + ix;
                                result[spec * flat_input_dim + out_idx] +=
                                    gemm_out[gemm_row * kernel_cols + col_idx];
                            }
                        }
                    }
                }
            }
        }
        self.poll()?;
        Ok(result)
    }

    fn poll_crown_backward_deadline(&self) -> Result<()> {
        self.poll()
    }

    fn forbids_unbounded_cpu_fallback(&self) -> bool {
        true
    }

    fn provides_deadline_pollable_host_gemm(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_core::{
        GpuCrownLayer, GpuCrownResult, NaiveCpuGemmEngine, DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
    };
    use ny_tensor::BoundedTensor;

    use super::*;
    use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
    use crate::beta_crown::engine::domain_results::MultiObjectiveGraphDomainResult;
    use crate::beta_crown::engine::graph::domain_batch::{
        GraphDomainBatchExecutionMode, GraphDomainBatchExecutor, GraphDomainBatchPlan,
        MultiObjectiveBatchRequest,
    };
    use crate::beta_crown::{BetaCrownConfig, BetaCrownVerifier};
    use crate::layers::{AddLayer, Conv2dLayer, FlattenLayer, ReLULayer, ReshapeLayer};
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer};

    struct MockBoundedGpu {
        sound: bool,
        capacity: usize,
    }

    impl GpuCrownBackward for MockBoundedGpu {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp("test-only bounded backend".into()))
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.sound
        }

        fn deadline_bounded_resnet_sound_beta_max_rows(&self) -> usize {
            self.capacity
        }
    }

    #[test]
    fn exact_gate_inherits_only_when_absent() {
        assert!(!resolve_gate(false, None));
        assert!(resolve_gate(true, None));
        for typed in [false, true] {
            assert!(resolve_gate(typed, Some(OsStr::new("1"))));
            for raw in ["", "0", "01", "true", " 1", "1 "] {
                assert!(
                    !resolve_gate(typed, Some(OsStr::new(raw))),
                    "present runtime spelling {raw:?} must force the treatment off"
                );
            }
        }
    }

    #[test]
    fn admission_matrix_is_get_only_and_exactly_bounded() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(1);
        let valid = MockBoundedGpu {
            sound: true,
            capacity: 8,
        };

        let lookups = Cell::new(0usize);
        for (armed, caller, routed, shared_eligible, at, expected) in [
            (false, false, false, true, now, Admission::NotArmed),
            (true, true, false, true, now, Admission::RefusedCallerEngine),
            (
                true,
                false,
                true,
                true,
                now,
                Admission::RefusedPostRootEngine,
            ),
            (
                true,
                false,
                false,
                false,
                now,
                Admission::RefusedUnsupportedCallerMode,
            ),
            (
                true,
                false,
                false,
                true,
                deadline,
                Admission::RefusedExpired,
            ),
        ] {
            let result = admit(
                armed,
                caller,
                routed,
                shared_eligible,
                deadline,
                || at,
                || panic!("short-circuited admission must not inspect graph structure"),
                || {
                    lookups.set(lookups.get() + 1);
                    Some(&valid)
                },
            );
            assert_eq!(result, expected);
        }
        assert_eq!(
            lookups.get(),
            0,
            "ineligible calls must not observe the backend slot"
        );

        let structural_checks = Cell::new(0usize);
        let unsupported = admit(
            true,
            false,
            false,
            true,
            deadline,
            || now,
            || {
                structural_checks.set(structural_checks.get() + 1);
                false
            },
            || {
                lookups.set(lookups.get() + 1);
                Some(&valid)
            },
        );
        assert_eq!(unsupported, Admission::RefusedUnsupportedGraph);
        assert_eq!(structural_checks.get(), 1);
        assert_eq!(
            lookups.get(),
            0,
            "an unsupported graph must not observe the backend slot"
        );

        let clock_calls = Cell::new(0usize);
        let crossed_during_structure = admit(
            true,
            false,
            false,
            true,
            deadline,
            || {
                let call = clock_calls.get();
                clock_calls.set(call + 1);
                if call == 0 {
                    now
                } else {
                    deadline
                }
            },
            || true,
            || {
                lookups.set(lookups.get() + 1);
                Some(&valid)
            },
        );
        assert_eq!(crossed_during_structure, Admission::RefusedExpired);
        assert_eq!(clock_calls.get(), 2);
        assert_eq!(
            lookups.get(),
            0,
            "a structural prefilter that consumes the remaining authority must not observe CUDA"
        );

        let clock_calls = Cell::new(0usize);
        let backend_lookups = Cell::new(0usize);
        let crossed_during_capability = admit(
            true,
            false,
            false,
            true,
            deadline,
            || {
                let call = clock_calls.get();
                clock_calls.set(call + 1);
                if call < 2 {
                    now
                } else {
                    deadline
                }
            },
            || true,
            || {
                backend_lookups.set(backend_lookups.get() + 1);
                Some(&valid)
            },
        );
        assert_eq!(crossed_during_capability, Admission::RefusedExpired);
        assert_eq!(clock_calls.get(), 3);
        assert_eq!(backend_lookups.get(), 1);

        let unavailable = admit(true, false, false, true, deadline, || now, || true, || None);
        assert_eq!(unavailable, Admission::RefusedBackendUnavailable);

        for (sound, capacity, expected) in [
            (false, 8, Admission::RefusedUnsoundBackend),
            (true, 0, Admission::RefusedInvalidCapacity(0)),
            (true, 1, Admission::RefusedInvalidCapacity(1)),
            (true, 2, Admission::Accepted { capacity: 2 }),
            (
                true,
                DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
                Admission::Accepted {
                    capacity: DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
                },
            ),
            (
                true,
                DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1,
                Admission::RefusedInvalidCapacity(DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1),
            ),
        ] {
            let candidate = MockBoundedGpu { sound, capacity };
            assert_eq!(
                admit(
                    true,
                    false,
                    false,
                    true,
                    deadline,
                    || now,
                    || true,
                    || Some(&candidate)
                ),
                expected,
                "sound={sound} capacity={capacity}"
            );
        }
    }

    #[test]
    fn non_resnet_graph_refuses_before_backend_observation() {
        let linear = LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32])))
            .expect("single-output linear layer");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.set_output("linear");
        let input = BoundedTensor::concrete(arr1(&[0.0_f32]).into_dyn()).unwrap();
        let node_bounds = HashMap::from([(
            "linear".to_string(),
            Arc::new(BoundedTensor::concrete(arr1(&[0.0_f32]).into_dyn()).unwrap()),
        )]);
        assert!(!graph_may_support_bounded_beta(
            &graph,
            &input,
            &node_bounds
        ));

        let backend_lookups = Cell::new(0usize);
        let now = Instant::now();
        let admission = admit(
            true,
            false,
            false,
            true,
            now + Duration::from_secs(1),
            || now,
            || graph_may_support_bounded_beta(&graph, &input, &node_bounds),
            || {
                backend_lookups.set(backend_lookups.get() + 1);
                None
            },
        );
        assert_eq!(admission, Admission::RefusedUnsupportedGraph);
        assert_eq!(backend_lookups.get(), 0);
    }

    fn zero_bounds(shape: &[usize]) -> Arc<BoundedTensor> {
        Arc::new(
            BoundedTensor::concrete(ArrayD::zeros(IxDyn(shape)))
                .expect("test bounds must be constructible"),
        )
    }

    fn canonical_residual_graph() -> (
        GraphNetwork,
        BoundedTensor,
        HashMap<String, Arc<BoundedTensor>>,
    ) {
        let conv0 = Conv2dLayer::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32),
            None,
            (1, 1),
            (0, 0),
        )
        .unwrap();
        let conv1 = conv0.clone();
        let linear =
            LinearLayer::new(arr2(&[[1.0_f32, 1.0, 1.0, 1.0]]), Some(arr1(&[0.0_f32]))).unwrap();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv0", Layer::Conv2d(conv0)));
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["conv0".into()],
        ));
        graph.add_node(GraphNode::new(
            "conv1",
            Layer::Conv2d(conv1),
            vec!["relu0".into()],
        ));
        graph.add_node(GraphNode::binary(
            "add",
            Layer::Add(AddLayer),
            "conv1",
            "conv0",
        ));
        graph.add_node(GraphNode::new(
            "flatten",
            Layer::Flatten(FlattenLayer::new(0)),
            vec!["add".into()],
        ));
        graph.add_node(GraphNode::new(
            "linear",
            Layer::Linear(linear),
            vec!["flatten".into()],
        ));
        graph.set_output("linear");

        let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 2, 2]))).unwrap();
        let node_bounds = HashMap::from([
            ("conv0".into(), zero_bounds(&[1, 2, 2])),
            ("relu0".into(), zero_bounds(&[1, 2, 2])),
            ("conv1".into(), zero_bounds(&[1, 2, 2])),
            ("add".into(), zero_bounds(&[1, 2, 2])),
            ("flatten".into(), zero_bounds(&[1, 4])),
            ("linear".into(), zero_bounds(&[1, 1])),
        ]);
        (graph, input, node_bounds)
    }

    #[test]
    fn workload_charges_full_output_rows_for_no_unstable_conv_scratch() {
        // Two active objective rows fit this Conv's bounded f64 col buffer, but
        // the mandatory NoUnstable full-output path uses D=100 identity rows
        // and would need roughly 450 MiB. The other D×max and Linear scratch
        // admission checks remain below their 256 MiB ceilings, isolating the
        // Conv col-buffer requirement.
        let stem = Conv2dLayer::new(
            ArrayD::from_elem(IxDyn(&[1, 64, 3, 3]), 1.0_f32),
            None,
            (1, 1),
            (0, 0),
        )
        .unwrap();
        let body = Conv2dLayer::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32),
            None,
            (1, 1),
            (0, 0),
        )
        .unwrap();
        let linear = LinearLayer::new(ndarray::Array2::zeros((100, 1024)), None).unwrap();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("stem", Layer::Conv2d(stem)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["stem".into()],
        ));
        graph.add_node(GraphNode::new(
            "body",
            Layer::Conv2d(body),
            vec!["relu".into()],
        ));
        graph.add_node(GraphNode::binary(
            "add",
            Layer::Add(AddLayer),
            "body",
            "stem",
        ));
        graph.add_node(GraphNode::new(
            "flatten",
            Layer::Flatten(FlattenLayer::new(0)),
            vec!["add".into()],
        ));
        graph.add_node(GraphNode::new(
            "linear",
            Layer::Linear(linear),
            vec!["flatten".into()],
        ));
        graph.set_output("linear");

        let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[64, 34, 34]))).unwrap();
        let node_bounds = HashMap::from([
            ("stem".into(), zero_bounds(&[1, 32, 32])),
            ("relu".into(), zero_bounds(&[1, 32, 32])),
            ("body".into(), zero_bounds(&[1, 32, 32])),
            ("add".into(), zero_bounds(&[1, 32, 32])),
            ("flatten".into(), zero_bounds(&[1, 1024])),
            ("linear".into(), zero_bounds(&[1, 100])),
        ]);

        assert!(bounded_conv_f64_buffers_supported(
            &graph,
            &input,
            (&node_bounds).into(),
            2
        ));
        assert!(!bounded_conv_f64_buffers_supported(
            &graph,
            &input,
            (&node_bounds).into(),
            100
        ));
        assert!(!workload_may_support_bounded_beta(
            &graph,
            &input,
            &node_bounds,
            &[vec![0.0; 100], vec![0.0; 100]],
            &[0.0, 0.0],
            &[false, false],
        ));
    }

    fn projection_residual_graph() -> (
        GraphNetwork,
        BoundedTensor,
        HashMap<String, Arc<BoundedTensor>>,
    ) {
        let conv = Conv2dLayer::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32),
            None,
            (1, 1),
            (0, 0),
        )
        .unwrap();
        let linear =
            LinearLayer::new(arr2(&[[1.0_f32, 1.0, 1.0, 1.0]]), Some(arr1(&[0.0_f32]))).unwrap();

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("stem", Layer::Conv2d(conv.clone())));
        graph.add_node(GraphNode::new(
            "main_relu",
            Layer::ReLU(ReLULayer),
            vec!["stem".into()],
        ));
        graph.add_node(GraphNode::new(
            "main_conv",
            Layer::Conv2d(conv.clone()),
            vec!["main_relu".into()],
        ));
        graph.add_node(GraphNode::new(
            "projection",
            Layer::Conv2d(conv),
            vec!["stem".into()],
        ));
        graph.add_node(GraphNode::binary(
            "add",
            Layer::Add(AddLayer),
            "main_conv",
            "projection",
        ));
        graph.add_node(GraphNode::new(
            "flatten",
            Layer::Flatten(FlattenLayer::new(0)),
            vec!["add".into()],
        ));
        graph.add_node(GraphNode::new(
            "linear",
            Layer::Linear(linear),
            vec!["flatten".into()],
        ));
        graph.set_output("linear");

        let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 2, 2]))).unwrap();
        let node_bounds = HashMap::from([
            ("stem".into(), zero_bounds(&[1, 2, 2])),
            ("main_relu".into(), zero_bounds(&[1, 2, 2])),
            ("main_conv".into(), zero_bounds(&[1, 2, 2])),
            ("projection".into(), zero_bounds(&[1, 2, 2])),
            ("add".into(), zero_bounds(&[1, 2, 2])),
            ("flatten".into(), zero_bounds(&[1, 4])),
            ("linear".into(), zero_bounds(&[1, 1])),
        ]);
        (graph, input, node_bounds)
    }

    fn multi_block_residual_graph() -> (
        GraphNetwork,
        BoundedTensor,
        HashMap<String, Arc<BoundedTensor>>,
    ) {
        let conv = Conv2dLayer::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32),
            None,
            (1, 1),
            (0, 0),
        )
        .unwrap();
        let linear =
            LinearLayer::new(arr2(&[[1.0_f32, 1.0, 1.0, 1.0]]), Some(arr1(&[0.0_f32]))).unwrap();
        let mut graph = GraphNetwork::new();
        let mut node_bounds = HashMap::new();
        graph.add_node(GraphNode::from_input("stem", Layer::Conv2d(conv.clone())));
        node_bounds.insert("stem".into(), zero_bounds(&[1, 2, 2]));

        let mut boundary = "stem".to_string();
        for block in 0..3 {
            let relu0 = format!("block{block}_relu0");
            let conv0 = format!("block{block}_conv0");
            let relu1 = format!("block{block}_relu1");
            let conv1 = format!("block{block}_conv1");
            let add = format!("block{block}_add");
            graph.add_node(GraphNode::new(
                relu0.clone(),
                Layer::ReLU(ReLULayer),
                vec![boundary.clone()],
            ));
            graph.add_node(GraphNode::new(
                conv0.clone(),
                Layer::Conv2d(conv.clone()),
                vec![relu0.clone()],
            ));
            graph.add_node(GraphNode::new(
                relu1.clone(),
                Layer::ReLU(ReLULayer),
                vec![conv0.clone()],
            ));
            graph.add_node(GraphNode::new(
                conv1.clone(),
                Layer::Conv2d(conv.clone()),
                vec![relu1.clone()],
            ));
            graph.add_node(GraphNode::binary(
                add.clone(),
                Layer::Add(AddLayer),
                conv1.clone(),
                boundary,
            ));
            for name in [&relu0, &conv0, &relu1, &conv1, &add] {
                node_bounds.insert(name.clone(), zero_bounds(&[1, 2, 2]));
            }
            boundary = add;
        }
        graph.add_node(GraphNode::new(
            "flatten",
            Layer::Flatten(FlattenLayer::new(0)),
            vec![boundary],
        ));
        graph.add_node(GraphNode::new(
            "linear",
            Layer::Linear(linear),
            vec!["flatten".into()],
        ));
        graph.set_output("linear");
        node_bounds.insert("flatten".into(), zero_bounds(&[1, 4]));
        node_bounds.insert("linear".into(), zero_bounds(&[1, 1]));

        let input = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 2, 2]))).unwrap();
        (graph, input, node_bounds)
    }

    #[test]
    fn declared_runtime_shape_compatibility_is_narrow_and_asymmetric() {
        for (declared, runtime, expected, label) in [
            (&[2, 3][..], &[2, 3][..], true, "exact"),
            (
                &[2, 3][..],
                &[1, 2, 3][..],
                true,
                "one runtime batch singleton",
            ),
            (
                &[1, 2, 3][..],
                &[2, 3][..],
                false,
                "declared-only singleton",
            ),
            (
                &[2, 3][..],
                &[2, 2, 3][..],
                false,
                "non-singleton runtime prefix",
            ),
            (
                &[2, 3][..],
                &[1, 1, 2, 3][..],
                false,
                "two runtime singleton prefixes",
            ),
            (&[2, 3][..], &[1, 6][..], false, "same-product reshape"),
            (&[2, 3][..], &[1, 2, 4][..], false, "product mismatch"),
            (
                &[2, 3][..],
                &[1, 2, 3, 1][..],
                false,
                "trailing rank mismatch",
            ),
        ] {
            assert_eq!(
                declared_runtime_shape_compatible(declared, runtime),
                expected,
                "{label}: declared={declared:?} runtime={runtime:?}"
            );
        }
    }

    #[test]
    fn structural_prefilter_accepts_only_loader_elided_runtime_batch_metadata() {
        // Exercise the concrete Flatten(0) -> Linear -> ReLU -> Linear tail
        // involved in bounded shared-executor admission, while retaining a
        // residual Conv/Add ancestry required by the bounded extractor.
        let (mut graph, input, mut node_bounds) = canonical_residual_graph();
        graph.add_node(GraphNode::new(
            "tail_relu",
            Layer::ReLU(ReLULayer),
            vec!["linear".into()],
        ));
        graph.add_node(GraphNode::new(
            "tail_linear",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap()),
            vec!["tail_relu".into()],
        ));
        graph.set_output("tail_linear");
        node_bounds.insert("tail_relu".into(), zero_bounds(&[1, 1]));
        node_bounds.insert("tail_linear".into(), zero_bounds(&[1, 1]));

        graph.set_declared_shape(NETWORK_INPUT, vec![1, 2, 2]);
        for (name, shape) in [
            ("conv0", vec![1, 2, 2]),
            ("relu0", vec![1, 2, 2]),
            ("conv1", vec![1, 2, 2]),
            ("add", vec![1, 2, 2]),
            ("flatten", vec![1, 4]),
            ("linear", vec![1, 1]),
            ("tail_relu", vec![1, 1]),
            ("tail_linear", vec![1, 1]),
        ] {
            graph.set_declared_shape(name, shape);
        }
        assert!(
            graph_may_support_bounded_beta(&graph, &input, &node_bounds),
            "exact declared/runtime metadata must remain eligible"
        );

        graph.set_declared_shape(NETWORK_INPUT, vec![2, 2]);
        for (name, shape) in [
            ("conv0", vec![2, 2]),
            ("relu0", vec![2, 2]),
            ("conv1", vec![2, 2]),
            ("add", vec![2, 2]),
            ("flatten", vec![4]),
            ("linear", vec![1]),
            ("tail_relu", vec![1]),
            ("tail_linear", vec![1]),
        ] {
            graph.set_declared_shape(name, shape);
        }
        assert!(
            graph_may_support_bounded_beta(&graph, &input, &node_bounds),
            "a single runtime-only leading batch singleton is metadata-compatible"
        );

        graph.set_declared_shape("flatten", vec![2, 2]);
        assert!(
            !graph_may_support_bounded_beta(&graph, &input, &node_bounds),
            "same-product declared metadata must not be accepted as shape equality"
        );
        graph.set_declared_shape("flatten", vec![4]);

        let mut broadcast_bounds = node_bounds.clone();
        broadcast_bounds.insert("add".into(), zero_bounds(&[1, 1, 2, 2]));
        graph.set_declared_shape("add", vec![1, 2, 2]);
        assert!(
            !graph_may_support_bounded_beta(&graph, &input, &broadcast_bounds),
            "metadata compatibility must not relax the Add no-broadcast transition check"
        );
    }

    #[test]
    fn structural_prefilter_accepts_only_complete_shape_checked_residual_ancestry() {
        let (mut graph, input, mut node_bounds) = canonical_residual_graph();
        assert!(graph_may_support_bounded_beta(&graph, &input, &node_bounds));

        let oversized_reshape = GraphNode::new(
            "reshape",
            Layer::Reshape(ReshapeLayer::new(vec![1; MAX_ADMISSION_TENSOR_RANK + 1])),
            vec!["input".into()],
        );
        assert!(
            !audited_unary_layer_supported(&oversized_reshape, &[1], &[1]),
            "layer-owned reshape rank metadata must be capped before shape construction"
        );
        let inconsistent_flatten = GraphNode::new(
            "flatten-check",
            Layer::Flatten(FlattenLayer::new(0)),
            vec!["input".into()],
        );
        assert!(!audited_unary_layer_supported(
            &inconsistent_flatten,
            &[1, 2, 2],
            &[4]
        ));
        assert!(audited_unary_layer_supported(
            &inconsistent_flatten,
            &[1, 2, 2],
            &[1, 4]
        ));

        graph.add_node(GraphNode::from_input(
            "dead",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap()),
        ));
        node_bounds.insert("dead".into(), zero_bounds(&[1]));
        assert!(
            !graph_may_support_bounded_beta(&graph, &input, &node_bounds),
            "even an audited but dead node is outside the exact output decomposition"
        );

        let (graph, input, mut node_bounds) = canonical_residual_graph();
        node_bounds.insert("add".into(), zero_bounds(&[1, 1, 4]));
        assert!(
            !graph_may_support_bounded_beta(&graph, &input, &node_bounds),
            "broadcast or inconsistent residual shapes must refuse"
        );
    }

    #[test]
    fn structural_prefilter_accepts_projection_and_serial_residual_blocks() {
        let (graph, input, node_bounds) = projection_residual_graph();
        assert!(
            graph_may_support_bounded_beta(&graph, &input, &node_bounds),
            "a clean two-branch projection skip is in the extractor contract"
        );

        let (graph, input, node_bounds) = multi_block_residual_graph();
        assert!(
            graph_may_support_bounded_beta(&graph, &input, &node_bounds),
            "serial residual blocks must remain eligible for official ResNet models"
        );
    }

    #[test]
    fn structural_prefilter_caps_all_metadata_authority() {
        let (graph, input, mut node_bounds) = canonical_residual_graph();
        assert!(workload_may_support_bounded_beta(
            &graph,
            &input,
            &node_bounds,
            &[vec![1.0], vec![-1.0]],
            &[0.0, 0.0],
            &[false, false],
        ));
        assert!(
            workload_may_support_bounded_beta(
                &graph,
                &input,
                &node_bounds,
                &[vec![1.0], vec![-1.0]],
                &[0.0, 0.0],
                &[true, false],
            ),
            "a one-row logical tail is padded inside the bounded CUDA transaction"
        );
        assert!(!workload_may_support_bounded_beta(
            &graph,
            &input,
            &node_bounds,
            &[vec![1.0], vec![-1.0]],
            &[0.0, 0.0],
            &[true, true],
        ));
        let too_many_objectives = vec![vec![1.0]; MAX_ADMISSION_OBJECTIVES + 1];
        let too_many_thresholds = vec![0.0; too_many_objectives.len()];
        let too_many_verified = vec![false; too_many_objectives.len()];
        assert!(!workload_may_support_bounded_beta(
            &graph,
            &input,
            &node_bounds,
            &too_many_objectives,
            &too_many_thresholds,
            &too_many_verified,
        ));
        node_bounds.insert("unrelated-extra-root-bound".into(), zero_bounds(&[1]));
        assert!(
            !graph_may_support_bounded_beta(&graph, &input, &node_bounds),
            "every root-bound entry must correspond to exactly one graph node"
        );

        let (mut graph, _, _) = canonical_residual_graph();
        graph.output_node = "x".repeat(MAX_ADMISSION_IDENTIFIER_BYTES + 1);
        assert!(
            !bounded_identifier_metadata_supported(&graph),
            "identifier bytes must be bounded before topology clones them"
        );
        let (mut graph, _, _) = canonical_residual_graph();
        graph
            .nodes
            .get_mut("linear")
            .expect("canonical output node")
            .inputs = vec!["flatten".into(); MAX_ADMISSION_GRAPH_EDGES + 1];
        assert!(
            !bounded_identifier_metadata_supported(&graph),
            "edge cardinality must be capped before iterating input names"
        );
        let (mut graph, _, _) = canonical_residual_graph();
        graph.node_order[1] = graph.node_order[0].clone();
        assert!(
            !bounded_identifier_metadata_supported(&graph),
            "declared node order must contain every node exactly once"
        );

        assert_eq!(
            checked_parameter_sum(MAX_ADMISSION_TOTAL_PARAMETER_ELEMENTS - 1, 1),
            Some(MAX_ADMISSION_TOTAL_PARAMETER_ELEMENTS)
        );
        assert_eq!(
            checked_parameter_sum(MAX_ADMISSION_TOTAL_PARAMETER_ELEMENTS, 1),
            None,
            "aggregate parameters must not evade the per-buffer cap"
        );
        assert_eq!(checked_parameter_sum(usize::MAX, 1), None);
        assert_eq!(
            crate::network::crown_memory::dense_pair_bytes(512, 65_536),
            Some(MAX_CPU_BUFFER_BYTES)
        );
        assert!(
            crate::network::crown_memory::dense_pair_bytes(513, 65_536)
                .is_some_and(|bytes| bytes > MAX_CPU_BUFFER_BYTES),
            "NoUnstable full-output backward must cap D×max-intermediate coefficient pairs"
        );
        assert_eq!(
            bounded_linear_product_buffer_bytes(256, 65_536),
            Some(MAX_CPU_BUFFER_BYTES)
        );
        assert!(
            bounded_linear_product_buffer_bytes(257, 65_536)
                .is_some_and(|bytes| bytes > MAX_CPU_BUFFER_BYTES),
            "bounded Linear must cap each 2R×p f64 product buffer"
        );
        assert_eq!(
            bounded_linear_transient_peak_bytes(128, 65_536),
            Some(MAX_BOUNDED_LINEAR_TRANSIENT_BYTES)
        );
        assert!(
            bounded_linear_transient_peak_bytes(129, 65_536)
                .is_some_and(|bytes| bytes > MAX_BOUNDED_LINEAR_TRANSIENT_BYTES),
            "bounded Linear must cap aggregate live certified-product state"
        );

        let mut oversized_identity_bounds = node_bounds.clone();
        oversized_identity_bounds.insert("linear".into(), zero_bounds(&[8192]));
        assert!(
            !workload_may_support_bounded_beta(
                &graph,
                &input,
                &oversized_identity_bounds,
                &[vec![0.0; 8192], vec![0.0; 8192]],
                &[0.0, 0.0],
                &[false, false],
            ),
            "NoUnstable full-output CROWN must not allocate an unbounded identity pair"
        );
    }

    #[test]
    fn proxy_matches_naive_cpu_for_its_complete_surface() {
        let engine = DeadlineCpuGemmEngine::new(Instant::now() + Duration::from_secs(2));
        let a32 = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b32 = [1.0_f32, -1.0, 2.0, 0.5, 3.0, 4.0];
        assert_eq!(
            engine.gemm_f32(2, 3, 2, &a32, &b32).unwrap(),
            NaiveCpuGemmEngine.gemm_f32(2, 3, 2, &a32, &b32).unwrap()
        );

        let a64: Vec<f64> = a32.iter().map(|&value| f64::from(value)).collect();
        let b64: Vec<f64> = b32.iter().map(|&value| f64::from(value)).collect();
        assert_eq!(
            engine.gemm_f64(2, 3, 2, &a64, &b64).unwrap(),
            NaiveCpuGemmEngine.gemm_f64(2, 3, 2, &a64, &b64).unwrap()
        );
        assert!(engine.forbids_unbounded_cpu_fallback());
        assert!(engine.provides_deadline_pollable_host_gemm());
        assert_eq!(
            engine
                .gemm_f64_with_deadline(
                    2,
                    3,
                    2,
                    &a64,
                    &b64,
                    Instant::now() + Duration::from_secs(3),
                    1,
                )
                .unwrap(),
            NaiveCpuGemmEngine.gemm_f64(2, 3, 2, &a64, &b64).unwrap()
        );

        let params = ConvTranspose2dParams {
            num_specs: 1,
            out_channels: 1,
            in_channels: 1,
            out_h: 2,
            out_w: 2,
            in_h: 3,
            in_w: 3,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let activation = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0_f32, -1.0, 0.5, 2.0];
        assert_eq!(
            engine
                .conv_transpose_2d(&activation, &weight, &params)
                .unwrap(),
            NaiveCpuGemmEngine
                .conv_transpose_2d(&activation, &weight, &params)
                .unwrap()
        );

        let pair = engine
            .gemm_f64_pair_shared_rhs(1, 2, 1, [&[1.0, 2.0], &[3.0, 4.0]], &[5.0, 6.0])
            .unwrap();
        assert_eq!(pair, [vec![17.0], vec![39.0]]);
        let triplet = engine
            .gemm_f64_triplet(1, 1, 1, [&[2.0], &[3.0], &[4.0]], [&[5.0], &[6.0], &[7.0]])
            .unwrap();
        assert_eq!(triplet, [vec![10.0], vec![18.0], vec![28.0]]);
    }

    #[test]
    fn proxy_validates_shapes_and_never_exposes_gpu_authority() {
        let engine = DeadlineCpuGemmEngine::new(Instant::now() + Duration::from_secs(1));
        assert!(matches!(
            engine.gemm_f32(2, 2, 1, &[1.0], &[1.0, 2.0]),
            Err(NyError::InvalidSpec(_))
        ));
        assert!(matches!(
            engine.gemm_f64(usize::MAX, 2, 1, &[], &[]),
            Err(NyError::InvalidSpec(_))
        ));
        let over_cap_elements = MAX_CPU_BUFFER_BYTES / size_of::<f32>() + 1;
        assert!(matches!(
            engine.gemm_f32(over_cap_elements, 0, 1, &[], &[]),
            Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes: MAX_CPU_BUFFER_BYTES,
                ..
            }) if required_bytes > MAX_CPU_BUFFER_BYTES
        ));
        assert!(engine.forbids_unbounded_cpu_fallback());
        assert!(engine.as_gpu_crown_backward().is_none());
        assert!(engine.as_gpu_ibp_forward().is_none());
        assert!(engine.as_gpu_ibp_forward_ext().is_none());
        assert!(engine.as_gpu_dag_ibp_forward_ext().is_none());
        assert!(!engine.supports_deadline_safe_post_root_multi_objective_bab());
    }

    #[test]
    fn proxy_polls_during_contractions_and_immediately_before_publication() {
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let calls = AtomicUsize::new(0);
        let engine = DeadlineCpuGemmEngine::with_clock(deadline, || {
            if calls.fetch_add(1, Ordering::SeqCst) < 6 {
                start
            } else {
                deadline
            }
        });
        assert!(matches!(
            engine.gemm_f32(1, 1, 1, &[2.0], &[3.0]),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(
            calls.load(Ordering::SeqCst) >= 7,
            "the final publication poll must be reached"
        );

        let expired = DeadlineCpuGemmEngine::new(Instant::now());
        assert!(matches!(
            expired.gemm_f32(1, 1, 1, &[1.0], &[1.0]),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(
            expired.gemm_f64(1, 1, 1, &[1.0], &[1.0]),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(
            expired.poll_crown_backward_deadline(),
            Err(NyError::DeadlineExceeded(_))
        ));
    }

    #[test]
    fn empty_contraction_polls_across_a_wide_output_row() {
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let calls = AtomicUsize::new(0);
        let engine = DeadlineCpuGemmEngine::with_clock(deadline, || {
            if calls.fetch_add(1, Ordering::SeqCst) < 7 {
                start
            } else {
                deadline
            }
        });

        assert!(matches!(
            engine.gemm_f32(1, 4_096, 0, &[], &[]),
            Err(NyError::InvalidSpec(_))
        ));
        // Correct zero-contraction shape is m=1, k=0, n=4096.
        assert!(matches!(
            engine.gemm_f32(1, 0, 4_096, &[], &[]),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(
            calls.load(Ordering::SeqCst) >= 8,
            "k=0 must still poll while traversing output columns"
        );
    }

    #[test]
    fn accepted_proxy_selects_and_executes_the_first_shared_wave() {
        let gpu = MockBoundedGpu {
            sound: true,
            capacity: 8,
        };
        let now = Instant::now();
        let deadline = now + Duration::from_secs(2);
        let admission = admit(
            true,
            false,
            false,
            true,
            deadline,
            || now,
            || true,
            || Some(&gpu),
        );
        assert_eq!(admission.accepted_capacity(), Some(8));
        let proxy = DeadlineCpuGemmEngine::new(deadline);
        let engine: &dyn GemmEngine = &proxy;
        let plan = GraphDomainBatchPlan::for_multi_objective(0, 1, 2, true, false);
        assert_eq!(
            plan.execution_mode(),
            GraphDomainBatchExecutionMode::SharedExecutor
        );

        let linear = LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32])))
            .expect("single-output linear layer");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.set_output("linear");
        let input = Arc::new(
            BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn())
                .expect("finite input"),
        );
        let node_bounds = graph.collect_node_bounds(&input).expect("root node bounds");
        let domain = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(0.0, 1.0)],
            input.as_ref(),
            &[0.5],
            false,
        )
        .expect("root multi-objective domain");
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let results = GraphDomainBatchExecutor::execute_multi_objective(
            &verifier,
            MultiObjectiveBatchRequest {
                bab_round: 0,
                graph: &graph,
                domains: &[&domain],
                relu_nodes: &[],
                objectives: &[vec![1.0]],
                thresholds: &[0.5],
                engine,
                cut_pool: None,
                selective_root_alpha_candidate: None,
            },
        );
        assert!(matches!(
            results.as_slice(),
            [MultiObjectiveGraphDomainResult::NoUnstable {
                all_verified: true,
                any_violated: false,
            }]
        ));
        assert!(engine.as_gpu_crown_backward().is_none());
    }
}
