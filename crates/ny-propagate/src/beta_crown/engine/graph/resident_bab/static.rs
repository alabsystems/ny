// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Schedule-independent retained-BaB v1 static payload composition.

use std::collections::HashMap;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;
use std::time::Instant;

use ny_core::{
    gpu_bab_bound_static_payload_identity_v1, reshape_copy_axis_from_sentinel,
    GpuBabBoundF32Tensor, GpuBabBoundF32TensorRole, GpuBabBoundOwnedSlice,
    GpuBabBoundStaticScheduleRequest, GpuBabBoundU32Tensor, GpuBabBoundU32TensorRole, NyError,
    GPU_BAB_BOUND_MAX_ARENA_VALUES, GPU_BAB_BOUND_MAX_OBJECTIVES,
};
use ny_tensor::BoundedTensor;

use crate::beta_crown::bab_cuts::CutFoldScope;
use crate::network::{GraphNetwork, NETWORK_INPUT};
use crate::resident_bab_wire::v1::{
    topology_wire_length_preflight_v1, ResidentBabDecodedTopologyV1, ResidentBabFamilyLengthsV1,
    ResidentBabFrontierBranchV1, ResidentBabLayerBranchV1, ResidentBabLayerKindV1,
    ResidentBabLayerV1, ResidentBabNodeKindV1, ResidentBabNodeV1, ResidentBabSegmentKindV1,
    ResidentBabSegmentV1, ResidentBabTopologyV1, ResidentBabTopologyWireLengthPreflightV1,
    ResidentBabWireRangeV1, RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1, RESIDENT_BAB_MAX_RANK_V1,
    RESIDENT_BAB_MAX_RECORDS_V1, RESIDENT_BAB_NETWORK_INPUT_ID_V1, RESIDENT_BAB_TOPOLOGY_SCHEMA_V1,
};
use crate::Layer;

use super::budget::{
    checked_add, checked_elements, checked_hash_entries, invalid, poll_scaled,
    ResidentBabAdapterHostCapV1, ResidentBabComposeErrorV1, ResidentBabHostBudgetV1,
    RESIDENT_BAB_COMPOSE_POLL_STRIDE,
};

fn unsupported(reason: &'static str) -> ResidentBabComposeErrorV1 {
    ResidentBabComposeErrorV1::Unsupported(reason)
}

/// Exact finalized graph-owner inputs for one schedule-independent static draft.
///
/// `graph` must be the finalized configured graph whose already-warm execution
/// order is being admitted. `input` and `node_bounds` are the finalized root
/// enclosure sources. `initial_output` is specifically
/// `RootObjectiveEvaluation::initial_output`; substituting a domain objective
/// interval or rerunning propagation is forbidden. Objective rows are the full
/// original sign-normalized matrix in stable source order.
///
/// Future live composition must consume or borrow the exact finalized-root
/// handoff while its original `initial_output` remains live. The current
/// integration immediately restores the exact legacy parts and explicitly
/// drops that output without composing a static payload or opening a provider
/// phase.
pub(in crate::beta_crown::engine::graph) struct ResidentBabStaticSourceV1<'a> {
    pub graph: &'a GraphNetwork,
    pub input: &'a BoundedTensor,
    pub node_bounds: &'a HashMap<String, Arc<BoundedTensor>>,
    pub initial_output: &'a BoundedTensor,
    pub sign_normalized_objectives: &'a [Vec<f32>],
}

/// Owned, schedule-independent v1 static payload draft.
///
/// This type intentionally has no dispatch count and no conversion into
/// `GpuBabBoundGraphPlan`. Only a future backend-issued certificate bound to
/// this exact topology/schema/provider identity may supply the actual dispatch
/// schedule. Owning this value therefore confers no provider, phase-open, or
/// raw-execution authority.
#[derive(Debug, PartialEq)]
pub(in crate::beta_crown::engine::graph) struct ResidentBabStaticPayloadV1 {
    graph_scope: CutFoldScope,
    topology_schema_version: u32,
    topology_bytes: GpuBabBoundOwnedSlice<u8>,
    decoded_topology: ResidentBabTopologyV1,
    f32_tensors: Vec<GpuBabBoundF32Tensor>,
    u32_tensors: Vec<GpuBabBoundU32Tensor>,
    static_payload_identity_sha256: [u8; 32],
    adapter_host_peak_bytes: usize,
    adapter_host_retained_bytes_after: usize,
    adapter_host_exclusive_bytes: usize,
}

impl ResidentBabStaticPayloadV1 {
    pub(super) fn graph_scope(&self) -> CutFoldScope {
        self.graph_scope
    }

    pub(super) fn topology_schema_version(&self) -> u32 {
        self.topology_schema_version
    }

    pub(super) fn topology_bytes(&self) -> &GpuBabBoundOwnedSlice<u8> {
        &self.topology_bytes
    }

    pub(super) fn decoded_topology(&self) -> &ResidentBabTopologyV1 {
        &self.decoded_topology
    }

    pub(super) fn f32_tensors(&self) -> &[GpuBabBoundF32Tensor] {
        &self.f32_tensors
    }

    pub(super) fn u32_tensors(&self) -> &[GpuBabBoundU32Tensor] {
        &self.u32_tensors
    }

    pub(super) fn static_payload_identity_sha256(&self) -> &[u8; 32] {
        &self.static_payload_identity_sha256
    }

    pub(super) fn adapter_host_peak_bytes(&self) -> usize {
        self.adapter_host_peak_bytes
    }

    pub(super) fn adapter_host_retained_bytes_after(&self) -> usize {
        self.adapter_host_retained_bytes_after
    }

    pub(super) fn adapter_host_exclusive_bytes(&self) -> usize {
        self.adapter_host_exclusive_bytes
    }

    /// Borrow this exact payload for pre-descriptor backend certification.
    ///
    /// `requested_max_device_bytes` is device-local policy. This helper does
    /// not turn the payload's adapter-host receipt into plan/descriptor/phase
    /// authority; that separate finalized-root custody check remains closed.
    pub(super) fn schedule_request(
        &self,
        deadline: Instant,
        requested_max_device_bytes: usize,
    ) -> ny_core::Result<GpuBabBoundStaticScheduleRequest<'_>> {
        GpuBabBoundStaticScheduleRequest::new(
            self.topology_schema_version,
            self.topology_bytes.as_slice(),
            &self.f32_tensors,
            &self.u32_tensors,
            self.static_payload_identity_sha256,
            deadline,
            requested_max_device_bytes,
        )
    }
}

struct ConfiguredTopologyBuildV1 {
    graph_scope: CutFoldScope,
    topology: ResidentBabTopologyV1,
    adapter_host_peak_bytes: usize,
    adapter_host_retained_bytes_after: usize,
}

fn shape_product(shape: &[usize], label: &str) -> Result<usize, ResidentBabComposeErrorV1> {
    if shape.is_empty() || shape.len() > RESIDENT_BAB_MAX_RANK_V1 {
        return Err(invalid(format!(
            "retained-BaB {label} shape must have bounded nonzero rank and dimensions"
        )));
    }
    if shape.contains(&0) {
        return Err(invalid(format!(
            "retained-BaB {label} shape contains a zero dimension"
        )));
    }
    let values = shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| invalid(format!("retained-BaB {label} shape product overflows")))
    })?;
    if values > GPU_BAB_BOUND_MAX_ARENA_VALUES {
        return Err(invalid(format!(
            "retained-BaB {label} shape exceeds the core arena-value cap"
        )));
    }
    Ok(values)
}

fn eligible_shape_product(
    shape: &[usize],
    label: &str,
) -> Result<usize, ResidentBabComposeErrorV1> {
    if shape.is_empty() {
        return Err(invalid(format!(
            "retained-BaB {label} shape has empty rank or a zero dimension"
        )));
    }
    if shape.len() > RESIDENT_BAB_MAX_RANK_V1 {
        return Err(unsupported("a finalized tensor rank exceeds the v1 cap"));
    }
    if shape.contains(&0) {
        return Err(invalid(format!(
            "retained-BaB {label} shape has a zero dimension"
        )));
    }
    let mut values = 1usize;
    for &dim in shape {
        values = values
            .checked_mul(dim)
            .ok_or_else(|| unsupported("a finalized tensor exceeds the v1 arena cap"))?;
        if values > GPU_BAB_BOUND_MAX_ARENA_VALUES {
            return Err(unsupported("a finalized tensor exceeds the v1 arena cap"));
        }
    }
    Ok(values)
}

fn node_shape<'a>(
    source: &'a ResidentBabStaticSourceV1<'_>,
    name: &str,
    effective_output: &str,
) -> Result<&'a [usize], ResidentBabComposeErrorV1> {
    if name == effective_output {
        if let Some(bounds) = source.node_bounds.get(name) {
            if bounds.shape() != source.initial_output.shape() {
                return Err(invalid(
                    "retained-BaB finalized output shape disagrees with root node bounds",
                ));
            }
        }
        return Ok(source.initial_output.shape());
    }
    source
        .node_bounds
        .get(name)
        .map(|bounds| bounds.shape())
        .ok_or_else(|| {
            invalid(format!(
                "retained-BaB finalized node bounds omit configured node {name}"
            ))
        })
}

fn node_kind(layer: &Layer) -> Result<ResidentBabNodeKindV1, ResidentBabComposeErrorV1> {
    match layer {
        Layer::Linear(_) => Ok(ResidentBabNodeKindV1::Linear),
        Layer::Conv2d(conv) if conv.groups == 1 && conv.dilation == (1, 1) => {
            Ok(ResidentBabNodeKindV1::Conv2d)
        }
        Layer::ReLU(_) => Ok(ResidentBabNodeKindV1::Relu),
        Layer::Flatten(_) => Ok(ResidentBabNodeKindV1::Flatten),
        Layer::Reshape(_) => Ok(ResidentBabNodeKindV1::Reshape),
        Layer::Add(_) => Ok(ResidentBabNodeKindV1::Add),
        Layer::Conv1d(_) => Err(unsupported(
            "Conv1d is not normalized into the exact rank-3 v1 descriptor",
        )),
        _ => Err(unsupported(
            "configured layer kind is outside the v1 whitelist",
        )),
    }
}

fn checked_u32(value: usize, label: &str) -> Result<u32, ResidentBabComposeErrorV1> {
    u32::try_from(value).map_err(|_| invalid(format!("retained-BaB {label} exceeds u32")))
}

fn checked_u64(value: usize, label: &str) -> Result<u64, ResidentBabComposeErrorV1> {
    u64::try_from(value).map_err(|_| invalid(format!("retained-BaB {label} exceeds u64")))
}

fn reserve_hash_map(
    budget: &mut ResidentBabHostBudgetV1,
    map: &mut HashMap<&str, u32>,
    count: usize,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    map.try_reserve(count)
        .map_err(|_| ResidentBabComposeErrorV1::AllocationRefused("topology name index"))?;
    budget.charge_hash_capacity::<&str, u32>(count, map.capacity())?;
    check("resident static topology name-index reserve")?;
    Ok(())
}

fn copy_shape_u64(
    shape: &[usize],
    budget: &mut ResidentBabHostBudgetV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
    label: &'static str,
) -> Result<Vec<u64>, ResidentBabComposeErrorV1> {
    let mut out = Vec::new();
    budget.reserve_vec_full(&mut out, shape.len(), label)?;
    check(label)?;
    for (index, &dim) in shape.iter().enumerate() {
        poll_scaled(check, label, index)?;
        out.push(checked_u64(dim, "shape dimension")?);
    }
    check(label)?;
    Ok(out)
}

fn topology_retained_bytes(
    topology: &ResidentBabTopologyV1,
    resident_bytes_before: usize,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<usize, ResidentBabComposeErrorV1> {
    let mut total = resident_bytes_before;
    checked_add(&mut total, size_of::<ConfiguredTopologyBuildV1>())?;
    checked_add(
        &mut total,
        checked_elements::<u64>(topology.input_shape.capacity())?,
    )?;
    checked_add(
        &mut total,
        checked_elements::<u64>(topology.output_shape.capacity())?,
    )?;
    checked_add(
        &mut total,
        checked_elements::<ResidentBabNodeV1>(topology.nodes.capacity())?,
    )?;
    checked_add(
        &mut total,
        checked_elements::<ResidentBabSegmentV1>(topology.segments.capacity())?,
    )?;
    checked_add(
        &mut total,
        checked_elements::<ResidentBabLayerV1>(topology.layers.capacity())?,
    )?;
    for (index, node) in topology.nodes.iter().enumerate() {
        poll_scaled(check, "resident static retained topology", index)?;
        checked_add(&mut total, node.name.capacity())?;
        checked_add(&mut total, checked_elements::<u32>(node.inputs.capacity())?)?;
        checked_add(
            &mut total,
            checked_elements::<u64>(node.output_shape.capacity())?,
        )?;
    }
    check("resident static retained topology final")?;
    Ok(total)
}

fn topology_nominal_bytes(count: usize) -> Result<usize, ResidentBabComposeErrorV1> {
    let mut total = size_of::<ConfiguredTopologyBuildV1>();
    for charge in [
        size_of::<HashMap<&str, u32>>(),
        checked_hash_entries::<&str, u32>(count)?,
        size_of::<Vec<u8>>(),
        checked_elements::<u8>(count)?,
        size_of::<Vec<u32>>(),
        checked_elements::<u32>(count)?,
        // Reusable residual ancestry marks are allocated only after the
        // selected count is known; their backing capacity is charged then.
        size_of::<Vec<u32>>(),
    ] {
        checked_add(&mut total, charge)?;
    }
    Ok(total)
}

fn initialize_vec<T: Copy>(
    out: &mut Vec<T>,
    count: usize,
    value: T,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
    label: &'static str,
) -> Result<(), ResidentBabComposeErrorV1> {
    for index in 0..count {
        poll_scaled(check, label, index)?;
        out.push(value);
    }
    check(label)?;
    Ok(())
}

fn compact_input_id(
    input: &str,
    name_to_old: &HashMap<&str, u32>,
    remap: &[u32],
) -> Result<u32, ResidentBabComposeErrorV1> {
    if input == NETWORK_INPUT {
        return Ok(RESIDENT_BAB_NETWORK_INPUT_ID_V1);
    }
    let old = *name_to_old
        .get(input)
        .ok_or_else(|| invalid("retained-BaB configured graph has a dangling input"))?;
    let old = usize::try_from(old)
        .map_err(|_| invalid("retained-BaB configured input index exceeds usize"))?;
    let compact = *remap
        .get(old)
        .ok_or_else(|| invalid("retained-BaB configured input remap is out of range"))?;
    if compact == RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        return Err(invalid(
            "retained-BaB output ancestor depends on an omitted configured node",
        ));
    }
    Ok(compact)
}

fn source_shape_for_id<'a>(
    input_shape: &'a [u64],
    nodes: &'a [ResidentBabNodeV1],
    source: u32,
) -> Result<&'a [u64], ResidentBabComposeErrorV1> {
    if source == RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        Ok(input_shape)
    } else {
        nodes
            .get(
                usize::try_from(source)
                    .map_err(|_| invalid("retained-BaB topology source ID exceeds usize"))?,
            )
            .map(|node| node.output_shape.as_slice())
            .ok_or_else(|| invalid("retained-BaB topology source ID is out of range"))
    }
}

/// Admit only the equal-shape Add subset represented by topology v1 while
/// distinguishing a coherent broadcast Add from malformed finalized bounds.
fn validate_selected_add_shape_v1(
    left: &[u64],
    right: &[u64],
    output: &[u64],
) -> Result<(), ResidentBabComposeErrorV1> {
    if left == right && left == output {
        return Ok(());
    }
    let rank = left.len().max(right.len());
    if output.len() != rank {
        return Err(invalid(
            "retained-BaB finalized Add shape disagrees with live broadcasting",
        ));
    }
    for reverse in 0..rank {
        let left_dim = left
            .len()
            .checked_sub(reverse + 1)
            .and_then(|index| left.get(index))
            .copied()
            .unwrap_or(1);
        let right_dim = right
            .len()
            .checked_sub(reverse + 1)
            .and_then(|index| right.get(index))
            .copied()
            .unwrap_or(1);
        if left_dim != right_dim && left_dim != 1 && right_dim != 1 {
            return Err(invalid(
                "retained-BaB finalized Add operands are not broadcast-compatible",
            ));
        }
        let output_index = rank - reverse - 1;
        if output[output_index] != left_dim.max(right_dim) {
            return Err(invalid(
                "retained-BaB finalized Add output disagrees with live broadcasting",
            ));
        }
    }
    Err(unsupported(
        "v1 Add requires exact equal-shape ordered operands",
    ))
}

fn u64_shape_product(shape: &[u64], label: &str) -> Result<u64, ResidentBabComposeErrorV1> {
    shape.iter().try_fold(1u64, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| invalid(format!("retained-BaB {label} shape overflows")))
    })
}

fn shape_matches_usize(decoded: &[u64], live: &[usize]) -> bool {
    decoded.len() == live.len()
        && decoded
            .iter()
            .zip(live)
            .all(|(&wire, &host)| u64::try_from(host) == Ok(wire))
}

fn decoded_input_name(
    topology: &ResidentBabTopologyV1,
    source: u32,
) -> Result<&str, ResidentBabComposeErrorV1> {
    if source == RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        return Ok(NETWORK_INPUT);
    }
    topology
        .nodes
        .get(
            usize::try_from(source)
                .map_err(|_| invalid("retained-BaB decoded input ID exceeds usize"))?,
        )
        .map(|node| node.name.as_str())
        .ok_or_else(|| invalid("retained-BaB decoded input ID is out of range"))
}

fn rebind_kind_v1(layer: &Layer) -> Option<ResidentBabNodeKindV1> {
    match layer {
        Layer::Linear(_) => Some(ResidentBabNodeKindV1::Linear),
        Layer::Conv2d(conv) if conv.groups == 1 && conv.dilation == (1, 1) => {
            Some(ResidentBabNodeKindV1::Conv2d)
        }
        Layer::ReLU(_) => Some(ResidentBabNodeKindV1::Relu),
        Layer::Flatten(_) => Some(ResidentBabNodeKindV1::Flatten),
        Layer::Reshape(_) => Some(ResidentBabNodeKindV1::Reshape),
        Layer::Add(_) => Some(ResidentBabNodeKindV1::Add),
        _ => None,
    }
}

fn rebind_flatten_shape_v1(
    axis: i32,
    source_shape: &[u64],
    output_shape: &[u64],
) -> Result<bool, ResidentBabComposeErrorV1> {
    let rank = i32::try_from(source_shape.len())
        .map_err(|_| invalid("retained-BaB Flatten rank exceeds i32"))?;
    let resolved = if axis < 0 {
        rank.checked_add(axis)
            .ok_or_else(|| invalid("retained-BaB Flatten axis overflows"))?
    } else {
        axis
    };
    if resolved < 0 || resolved > rank || output_shape.len() != 2 {
        return Ok(false);
    }
    let split = usize::try_from(resolved)
        .map_err(|_| invalid("retained-BaB Flatten axis exceeds usize"))?;
    let prefix = u64_shape_product(&source_shape[..split], "Flatten prefix")?;
    let suffix = u64_shape_product(&source_shape[split..], "Flatten suffix")?;
    Ok(output_shape == [prefix, suffix])
}

fn rebind_reshape_shape_v1(
    target: &[i64],
    source_shape: &[u64],
    output_shape: &[u64],
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<bool, ResidentBabComposeErrorV1> {
    if target.is_empty()
        || target.len() > RESIDENT_BAB_MAX_RANK_V1
        || target.len() != output_shape.len()
    {
        return Ok(false);
    }
    let total = u64_shape_product(source_shape, "Reshape source")?;
    let mut infer = None;
    let mut known = 1u64;
    for (index, &dim) in target.iter().enumerate() {
        poll_scaled(check, "resident static Reshape target prepass", index)?;
        let factor = if dim == -1 {
            if infer.replace(index).is_some() {
                return Ok(false);
            }
            continue;
        } else if dim == 0 {
            *source_shape
                .get(index)
                .ok_or_else(|| invalid("retained-BaB Reshape zero dimension exceeds source rank"))?
        } else if let Some(axis) = reshape_copy_axis_from_sentinel(dim) {
            *source_shape
                .get(axis)
                .ok_or_else(|| invalid("retained-BaB Reshape copy-axis exceeds source rank"))?
        } else {
            if dim < 0 {
                return Ok(false);
            }
            u64::try_from(dim).map_err(|_| invalid("retained-BaB Reshape dimension exceeds u64"))?
        };
        if factor == 0 {
            return Ok(false);
        }
        known = known
            .checked_mul(factor)
            .ok_or_else(|| invalid("retained-BaB Reshape known product overflows"))?;
    }
    check("resident static Reshape target prepass final")?;
    if known == 0 || total % known != 0 {
        return Ok(false);
    }
    for (index, (&dim, &decoded)) in target.iter().zip(output_shape).enumerate() {
        poll_scaled(check, "resident static Reshape target comparison", index)?;
        let expected = if dim == -1 {
            total / known
        } else if dim == 0 {
            source_shape[index]
        } else if let Some(axis) = reshape_copy_axis_from_sentinel(dim) {
            source_shape[axis]
        } else {
            u64::try_from(dim).map_err(|_| invalid("retained-BaB Reshape dimension exceeds u64"))?
        };
        if expected != decoded {
            return Ok(false);
        }
    }
    check("resident static Reshape target comparison final")?;
    Ok(u64_shape_product(output_shape, "Reshape output")? == total)
}

fn rebind_decoded_layers_v1(
    topology: &ResidentBabTopologyV1,
    graph: &GraphNetwork,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    let mut parameters = 0u64;
    let mut errors = 0u64;
    let mut activation = 0u64;
    let mut beta = 0u64;
    let mut abs = 0u64;
    for (index, segment) in topology.segments.iter().enumerate() {
        poll_scaled(
            check,
            "resident static decoded frontier-range rebind",
            index,
        )?;
        let width = u64_shape_product(
            source_shape_for_id(
                &topology.input_shape,
                &topology.nodes,
                segment.frontier_node_id,
            )?,
            "decoded frontier",
        )?;
        if segment.frontier_abs
            != (ResidentBabWireRangeV1 {
                start: abs,
                len: width,
            })
        {
            return Err(invalid(
                "retained-BaB decoded frontier range changed from live geometry",
            ));
        }
        abs = abs
            .checked_add(width)
            .ok_or_else(|| invalid("retained-BaB rebound frontier Abs overflows"))?;
    }
    check("resident static decoded frontier-range rebind final")?;

    for (index, layer) in topology.layers.iter().enumerate() {
        poll_scaled(check, "resident static decoded layer rebind", index)?;
        let node = topology
            .nodes
            .get(
                usize::try_from(layer.node_id)
                    .map_err(|_| invalid("retained-BaB decoded layer node ID exceeds usize"))?,
            )
            .ok_or_else(|| invalid("retained-BaB decoded layer node is missing"))?;
        check("resident static decoded layer-name rebind")?;
        let live = graph.node(&node.name).ok_or_else(|| {
            invalid("retained-BaB decoded layer node is absent from the live graph")
        })?;
        let mut expected_geometry = [0u32; 13];
        let (parameter_len, error_len, activation_len, beta_len, abs_len) =
            match (layer.kind, live.layer()) {
                (ResidentBabLayerKindV1::Linear, Layer::Linear(linear)) => {
                    let output = checked_u32(linear.out_features(), "rebound Linear output")?;
                    let input = checked_u32(linear.in_features(), "rebound Linear input")?;
                    let has_bias = u32::from(linear.bias().is_some());
                    expected_geometry[0] = output;
                    expected_geometry[1] = input;
                    expected_geometry[2] = has_bias;
                    let length = u64::from(output)
                        .checked_mul(u64::from(input))
                        .and_then(|value| {
                            value.checked_add(u64::from(output).checked_mul(u64::from(has_bias))?)
                        })
                        .ok_or_else(|| invalid("retained-BaB rebound Linear length overflows"))?;
                    (length, 2, 0, 0, 0)
                }
                (ResidentBabLayerKindV1::Conv2d, Layer::Conv2d(conv)) => {
                    let kernel = conv.kernel.shape();
                    if kernel.len() != 4 {
                        return Err(invalid("retained-BaB live Conv2d kernel rank changed"));
                    }
                    let source_shape = source_shape_for_id(
                        &topology.input_shape,
                        &topology.nodes,
                        node.inputs[0],
                    )?;
                    if source_shape.len() != 3 || node.output_shape.len() != 3 {
                        return Err(invalid("retained-BaB live Conv2d rank changed"));
                    }
                    let fields = [
                        kernel[0],
                        kernel[1],
                        kernel[2],
                        kernel[3],
                        conv.stride.0,
                        conv.stride.1,
                        conv.padding.0,
                        conv.padding.1,
                        usize::try_from(node.output_shape[1]).map_err(|_| {
                            invalid("retained-BaB Conv2d output height exceeds usize")
                        })?,
                        usize::try_from(node.output_shape[2]).map_err(|_| {
                            invalid("retained-BaB Conv2d output width exceeds usize")
                        })?,
                        usize::try_from(source_shape[1]).map_err(|_| {
                            invalid("retained-BaB Conv2d input height exceeds usize")
                        })?,
                        usize::try_from(source_shape[2]).map_err(|_| {
                            invalid("retained-BaB Conv2d input width exceeds usize")
                        })?,
                        usize::from(conv.bias.is_some()),
                    ];
                    for (slot, value) in expected_geometry.iter_mut().zip(fields) {
                        *slot = checked_u32(value, "rebound Conv2d geometry")?;
                    }
                    let weight_len = kernel.iter().try_fold(1u64, |product, &dim| {
                        product
                            .checked_mul(u64::try_from(dim).map_err(|_| {
                                invalid("retained-BaB Conv2d dimension exceeds u64")
                            })?)
                            .ok_or_else(|| invalid("retained-BaB Conv2d weights overflow"))
                    })?;
                    let bias_len = u64::from(expected_geometry[0])
                        .checked_mul(u64::from(expected_geometry[8]))
                        .and_then(|value| value.checked_mul(u64::from(expected_geometry[9])))
                        .and_then(|value| value.checked_mul(u64::from(expected_geometry[12])))
                        .ok_or_else(|| invalid("retained-BaB Conv2d bias expansion overflows"))?;
                    (
                        weight_len.checked_add(bias_len).ok_or_else(|| {
                            invalid("retained-BaB Conv2d parameter length overflows")
                        })?,
                        2,
                        0,
                        0,
                        0,
                    )
                }
                (ResidentBabLayerKindV1::Relu, Layer::ReLU(_)) => {
                    let width = node.output_values;
                    expected_geometry[0] = u32::try_from(width)
                        .map_err(|_| invalid("retained-BaB rebound ReLU width exceeds u32"))?;
                    (
                        0,
                        0,
                        width
                            .checked_mul(6)
                            .and_then(|value| value.checked_add(1))
                            .ok_or_else(|| {
                                invalid("retained-BaB rebound Activation length overflows")
                            })?,
                        width,
                        width,
                    )
                }
                _ => {
                    return Err(invalid(
                        "retained-BaB decoded layer no longer matches its live layer",
                    ));
                }
            };
        if layer.geometry != expected_geometry
            || layer.parameters
                != (ResidentBabWireRangeV1 {
                    start: parameters,
                    len: parameter_len,
                })
            || layer.certified_errors
                != (ResidentBabWireRangeV1 {
                    start: errors,
                    len: error_len,
                })
            || layer.activation
                != (ResidentBabWireRangeV1 {
                    start: activation,
                    len: activation_len,
                })
            || layer.beta
                != (ResidentBabWireRangeV1 {
                    start: beta,
                    len: beta_len,
                })
            || layer.node_abs
                != (ResidentBabWireRangeV1 {
                    start: abs,
                    len: abs_len,
                })
        {
            return Err(invalid(
                "retained-BaB decoded layer geometry/ranges changed from live semantics",
            ));
        }
        parameters = parameters
            .checked_add(parameter_len)
            .ok_or_else(|| invalid("retained-BaB rebound parameter cursor overflows"))?;
        errors = errors
            .checked_add(error_len)
            .ok_or_else(|| invalid("retained-BaB rebound error cursor overflows"))?;
        activation = activation
            .checked_add(activation_len)
            .ok_or_else(|| invalid("retained-BaB rebound Activation cursor overflows"))?;
        beta = beta
            .checked_add(beta_len)
            .ok_or_else(|| invalid("retained-BaB rebound Beta cursor overflows"))?;
        abs = abs
            .checked_add(abs_len)
            .ok_or_else(|| invalid("retained-BaB rebound Abs cursor overflows"))?;
    }
    check("resident static decoded layer rebind final")?;
    if topology.families.parameters != parameters
        || topology.families.certified_errors != errors
        || topology.families.activation != activation
        || topology.families.beta != beta
        || topology.families.abs != abs
    {
        return Err(invalid(
            "retained-BaB decoded family totals changed from live layer geometry",
        ));
    }
    Ok(())
}

/// Rebind an independently decoded wire model to the still-live configured
/// graph without relying on the producer composer. Flatten/Reshape records bind
/// the normalized logical row-major resident layout: their raw ONNX target/axis
/// spelling is intentionally not part of schema v1, while exact source/output
/// shapes and element preservation remain bound.
fn rebind_decoded_topology_v1(
    topology: &ResidentBabTopologyV1,
    source: &ResidentBabStaticSourceV1<'_>,
    expected_scope: CutFoldScope,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    if source.graph.cut_fold_scope() != expected_scope {
        return Err(invalid(
            "retained-BaB configured graph scope changed before static rebind",
        ));
    }
    let exec_order = source
        .graph
        .retained_v1_exec_order_if_cached()
        .ok_or_else(|| invalid("retained-BaB configured execution-order cache disappeared"))?;
    let output_name = if source.graph.output_name().is_empty() {
        exec_order
            .last()
            .map(String::as_str)
            .ok_or_else(|| invalid("retained-BaB configured graph has no effective output"))?
    } else {
        source.graph.output_name()
    };
    let output_index = usize::try_from(topology.output_node_id)
        .map_err(|_| invalid("retained-BaB decoded output ID exceeds usize"))?;
    if topology
        .nodes
        .get(output_index)
        .is_none_or(|node| node.name != output_name)
        || !shape_matches_usize(&topology.input_shape, source.input.shape())
        || !shape_matches_usize(&topology.output_shape, source.initial_output.shape())
    {
        return Err(invalid(
            "retained-BaB decoded topology is not bound to the configured graph I/O",
        ));
    }

    // Bind the compact table to the exact cached-order subsequence without an
    // allocating lookup. Each bounded name comparison is polled separately.
    let mut compact = 0usize;
    for (index, cached_name) in exec_order.iter().enumerate() {
        check("resident static decoded cached-order rebind record")?;
        poll_scaled(check, "resident static decoded cached-order rebind", index)?;
        if topology
            .nodes
            .get(compact)
            .is_some_and(|node| node.name == *cached_name)
        {
            compact += 1;
        }
    }
    check("resident static decoded cached-order rebind final")?;
    if compact != topology.nodes.len() {
        return Err(invalid(
            "retained-BaB decoded nodes are not the configured cached-order subsequence",
        ));
    }

    for (index, node) in topology.nodes.iter().enumerate() {
        check("resident static decoded node-name rebind")?;
        poll_scaled(check, "resident static decoded node rebind", index)?;
        if usize::try_from(node.id) != Ok(index) {
            return Err(invalid("retained-BaB decoded node IDs are not dense"));
        }
        let live = source.graph.node(&node.name).ok_or_else(|| {
            invalid("retained-BaB decoded node is absent from the configured graph")
        })?;
        if live.name() != node.name
            || rebind_kind_v1(live.layer()) != Some(node.kind)
            || live.inputs().len() != node.inputs.len()
        {
            return Err(invalid(
                "retained-BaB decoded node kind/arity changed from the configured graph",
            ));
        }
        for (input_index, (&wire_input, live_input)) in
            node.inputs.iter().zip(live.inputs()).enumerate()
        {
            poll_scaled(
                check,
                "resident static decoded input-edge rebind",
                input_index,
            )?;
            check("resident static decoded input-name rebind")?;
            if decoded_input_name(topology, wire_input)? != live_input {
                return Err(invalid(
                    "retained-BaB decoded input order changed from the configured graph",
                ));
            }
        }
        check("resident static decoded input-edge rebind final")?;
        let live_shape = node_shape(source, &node.name, output_name)?;
        let live_values = shape_product(live_shape, "rebound node output")?;
        if !shape_matches_usize(&node.output_shape, live_shape)
            || node.output_values != u64::try_from(live_values).unwrap_or(u64::MAX)
        {
            return Err(invalid(
                "retained-BaB decoded node shape changed from finalized bounds",
            ));
        }
        let source_shape =
            source_shape_for_id(&topology.input_shape, &topology.nodes, node.inputs[0])?;
        match (node.kind, live.layer()) {
            (ResidentBabNodeKindV1::Linear, Layer::Linear(linear)) => {
                if source_shape
                    != [u64::try_from(linear.in_features())
                        .map_err(|_| invalid("retained-BaB rebound Linear input exceeds u64"))?]
                    || node.output_shape
                        != [u64::try_from(linear.out_features()).map_err(|_| {
                            invalid("retained-BaB rebound Linear output exceeds u64")
                        })?]
                {
                    return Err(invalid(
                        "retained-BaB decoded Linear geometry changed from the live layer",
                    ));
                }
            }
            (ResidentBabNodeKindV1::Conv2d, Layer::Conv2d(conv)) => {
                let kernel = conv.kernel.shape();
                if kernel.len() != 4
                    || source_shape.len() != 3
                    || node.output_shape.len() != 3
                    || source_shape[0] != u64::try_from(kernel[1]).unwrap_or(u64::MAX)
                    || node.output_shape[0] != u64::try_from(kernel[0]).unwrap_or(u64::MAX)
                    || conv
                        .bias
                        .as_ref()
                        .is_some_and(|bias| bias.len() != kernel[0])
                {
                    return Err(invalid(
                        "retained-BaB decoded Conv2d geometry changed from the live layer",
                    ));
                }
                let padded_h = source_shape[1]
                    .checked_add(
                        u64::try_from(conv.padding.0)
                            .ok()
                            .and_then(|pad| pad.checked_mul(2))
                            .ok_or_else(|| invalid("retained-BaB Conv2d padding overflows"))?,
                    )
                    .and_then(|value| value.checked_sub(u64::try_from(kernel[2]).ok()?))
                    .ok_or_else(|| invalid("retained-BaB rebound Conv2d height is invalid"))?;
                let padded_w = source_shape[2]
                    .checked_add(
                        u64::try_from(conv.padding.1)
                            .ok()
                            .and_then(|pad| pad.checked_mul(2))
                            .ok_or_else(|| invalid("retained-BaB Conv2d padding overflows"))?,
                    )
                    .and_then(|value| value.checked_sub(u64::try_from(kernel[3]).ok()?))
                    .ok_or_else(|| invalid("retained-BaB rebound Conv2d width is invalid"))?;
                let stride_h = u64::try_from(conv.stride.0)
                    .map_err(|_| invalid("retained-BaB Conv2d stride exceeds u64"))?;
                let stride_w = u64::try_from(conv.stride.1)
                    .map_err(|_| invalid("retained-BaB Conv2d stride exceeds u64"))?;
                if stride_h == 0
                    || stride_w == 0
                    || node.output_shape[1] != padded_h / stride_h + 1
                    || node.output_shape[2] != padded_w / stride_w + 1
                {
                    return Err(invalid(
                        "retained-BaB decoded Conv2d formula changed from the live layer",
                    ));
                }
            }
            (ResidentBabNodeKindV1::Relu, Layer::ReLU(_)) => {
                if source_shape != node.output_shape
                    || node.relu_preactivation_node_id != Some(node.inputs[0])
                {
                    return Err(invalid(
                        "retained-BaB decoded ReLU association changed from the live layer",
                    ));
                }
            }
            (ResidentBabNodeKindV1::Flatten, Layer::Flatten(flatten)) => {
                if !rebind_flatten_shape_v1(flatten.axis, source_shape, &node.output_shape)? {
                    return Err(invalid(
                        "retained-BaB decoded Flatten shape changed from live semantics",
                    ));
                }
            }
            (ResidentBabNodeKindV1::Reshape, Layer::Reshape(reshape)) => {
                if !rebind_reshape_shape_v1(
                    &reshape.target_shape,
                    source_shape,
                    &node.output_shape,
                    check,
                )? {
                    return Err(invalid(
                        "retained-BaB decoded Reshape shape changed from live semantics",
                    ));
                }
            }
            (ResidentBabNodeKindV1::Add, Layer::Add(_)) => {
                let right_shape =
                    source_shape_for_id(&topology.input_shape, &topology.nodes, node.inputs[1])?;
                if source_shape != node.output_shape || right_shape != node.output_shape {
                    return Err(invalid(
                        "retained-BaB decoded Add is not exact ordered equal-shape addition",
                    ));
                }
            }
            _ => {
                return Err(invalid(
                    "retained-BaB decoded node no longer matches the live layer variant",
                ));
            }
        }
    }
    check("resident static decoded node rebind final")?;
    rebind_decoded_layers_v1(topology, source.graph, check)?;
    Ok(())
}

fn is_structural(kind: ResidentBabNodeKindV1) -> bool {
    matches!(
        kind,
        ResidentBabNodeKindV1::Flatten | ResidentBabNodeKindV1::Reshape
    )
}

fn is_executable(kind: ResidentBabNodeKindV1) -> bool {
    matches!(
        kind,
        ResidentBabNodeKindV1::Linear | ResidentBabNodeKindV1::Conv2d | ResidentBabNodeKindV1::Relu
    )
}

fn unary_parent(
    topology: &ResidentBabTopologyV1,
    node_id: u32,
) -> Result<u32, ResidentBabComposeErrorV1> {
    let node = topology
        .nodes
        .get(
            usize::try_from(node_id)
                .map_err(|_| invalid("retained-BaB topology cursor exceeds usize"))?,
        )
        .ok_or_else(|| invalid("retained-BaB topology cursor is out of range"))?;
    if node.inputs.len() != 1 {
        return Err(invalid(
            "retained-BaB branch contains a non-unary non-Add node",
        ));
    }
    Ok(node.inputs[0])
}

fn layer_kind(kind: ResidentBabNodeKindV1) -> Option<ResidentBabLayerKindV1> {
    match kind {
        ResidentBabNodeKindV1::Linear => Some(ResidentBabLayerKindV1::Linear),
        ResidentBabNodeKindV1::Conv2d => Some(ResidentBabLayerKindV1::Conv2d),
        ResidentBabNodeKindV1::Relu => Some(ResidentBabLayerKindV1::Relu),
        _ => None,
    }
}

fn push_layer_record(
    topology: &mut ResidentBabTopologyV1,
    node_id: u32,
    segment_id: u32,
    branch: ResidentBabLayerBranchV1,
) -> Result<(), ResidentBabComposeErrorV1> {
    let kind = topology
        .nodes
        .get(node_id as usize)
        .and_then(|node| layer_kind(node.kind))
        .ok_or_else(|| invalid("retained-BaB fold attempted to emit a structural node"))?;
    topology.layers.push(ResidentBabLayerV1 {
        ordinal: checked_u32(topology.layers.len(), "layer ordinal")?,
        kind,
        branch,
        segment_id,
        node_id,
        parameters: ResidentBabWireRangeV1::default(),
        certified_errors: ResidentBabWireRangeV1::default(),
        activation: ResidentBabWireRangeV1::default(),
        beta: ResidentBabWireRangeV1::default(),
        node_abs: ResidentBabWireRangeV1::default(),
        geometry: [0; 13],
    });
    Ok(())
}

fn trace_branch_into_layers(
    topology: &mut ResidentBabTopologyV1,
    mut cursor: u32,
    frontier: u32,
    segment_id: u32,
    branch: ResidentBabLayerBranchV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<u32, ResidentBabComposeErrorV1> {
    let first = topology.layers.len();
    let mut steps = 0usize;
    while cursor != frontier {
        poll_scaled(check, "resident static topology branch trace", steps)?;
        let kind = topology
            .nodes
            .get(
                usize::try_from(cursor)
                    .map_err(|_| invalid("retained-BaB branch cursor exceeds usize"))?,
            )
            .map(|node| node.kind)
            .ok_or_else(|| invalid("retained-BaB branch reached network input too early"))?;
        if kind == ResidentBabNodeKindV1::Add {
            return Err(unsupported(
                "retained-BaB residual branch contains a nested Add",
            ));
        }
        if is_executable(kind) {
            push_layer_record(topology, cursor, segment_id, branch)?;
        } else if !is_structural(kind) {
            return Err(invalid("retained-BaB residual branch kind is unsupported"));
        }
        cursor = unary_parent(topology, cursor)?;
        steps = steps
            .checked_add(1)
            .ok_or_else(|| invalid("retained-BaB branch step count overflows"))?;
    }
    check("resident static topology branch trace final")?;
    checked_u32(topology.layers.len() - first, "branch layer count")
}

fn find_residual_frontier(
    topology: &ResidentBabTopologyV1,
    left: u32,
    right: u32,
    marks: &mut [u32],
    epoch: u32,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<u32, ResidentBabComposeErrorV1> {
    let mut cursor = left;
    let mut steps = 0usize;
    let mut left_reached_input = false;
    while cursor != RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        poll_scaled(check, "resident static residual left ancestry", steps)?;
        let slot = marks
            .get_mut(
                usize::try_from(cursor)
                    .map_err(|_| invalid("retained-BaB residual cursor exceeds usize"))?,
            )
            .ok_or_else(|| invalid("retained-BaB residual left cursor is invalid"))?;
        *slot = epoch;
        let cursor_index = usize::try_from(cursor)
            .map_err(|_| invalid("retained-BaB residual cursor exceeds usize"))?;
        let kind = topology.nodes[cursor_index].kind;
        if kind == ResidentBabNodeKindV1::Add {
            // A previous residual merge is a valid shared frontier for the
            // next block, but never an interior branch node. Stop the left
            // ancestry here; the right walk must meet this exact marked Add.
            break;
        }
        cursor = unary_parent(topology, cursor)?;
        steps += 1;
    }
    if cursor == RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        left_reached_input = true;
    }
    check("resident static residual left ancestry final")?;

    cursor = right;
    steps = 0;
    while cursor != RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        poll_scaled(check, "resident static residual right ancestry", steps)?;
        if marks
            .get(
                usize::try_from(cursor)
                    .map_err(|_| invalid("retained-BaB residual cursor exceeds usize"))?,
            )
            .is_some_and(|&mark| mark == epoch)
        {
            check("resident static residual right ancestry final")?;
            return Ok(cursor);
        }
        let kind = topology
            .nodes
            .get(cursor as usize)
            .map(|node| node.kind)
            .ok_or_else(|| invalid("retained-BaB residual right cursor is invalid"))?;
        if kind == ResidentBabNodeKindV1::Add {
            return Err(unsupported(
                "retained-BaB residual branch contains an earlier Add",
            ));
        }
        cursor = unary_parent(topology, cursor)?;
        steps += 1;
    }
    check("resident static residual right ancestry final")?;
    if left_reached_input {
        Ok(RESIDENT_BAB_NETWORK_INPUT_ID_V1)
    } else {
        Err(unsupported(
            "retained-BaB residual branches do not meet at one canonical frontier",
        ))
    }
}

fn configure_layer_ranges(
    topology: &mut ResidentBabTopologyV1,
    graph: &GraphNetwork,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    let mut parameters = 0u64;
    let mut errors = 0u64;
    let mut activation = 0u64;
    let mut beta = 0u64;
    let mut node_abs = 0u64;
    for (index, segment) in topology.segments.iter().enumerate() {
        poll_scaled(check, "resident static frontier range scan", index)?;
        let expected = node_abs;
        if segment.frontier_abs.start != expected {
            return Err(invalid(
                "retained-BaB static segment frontier ranges are not contiguous",
            ));
        }
        node_abs = segment
            .frontier_abs
            .start
            .checked_add(segment.frontier_abs.len)
            .ok_or_else(|| invalid("retained-BaB static frontier Abs range overflows"))?;
    }
    check("resident static frontier range scan final")?;
    let mut relu_count = 0u32;

    let input_shape = topology.input_shape.as_slice();
    let nodes = topology.nodes.as_slice();
    for (index, layer) in topology.layers.iter_mut().enumerate() {
        poll_scaled(check, "resident static layer range composition", index)?;
        let node = nodes
            .get(layer.node_id as usize)
            .ok_or_else(|| invalid("retained-BaB static layer node is missing"))?;
        check("resident static layer-name association")?;
        let graph_node = graph.node(&node.name).ok_or_else(|| {
            invalid("retained-BaB decoded layer is not associated with the configured graph")
        })?;
        if graph_node.name() != node.name || graph_node.inputs().len() != node.inputs.len() {
            return Err(invalid(
                "retained-BaB decoded layer association changed after topology composition",
            ));
        }
        match (layer.kind, graph_node.layer()) {
            (ResidentBabLayerKindV1::Linear, Layer::Linear(linear)) => {
                let source_shape = source_shape_for_id(input_shape, nodes, node.inputs[0])?;
                if source_shape.len() != 1 || node.output_shape.len() != 1 {
                    return Err(unsupported(
                        "v1 Linear requires exact rank-1 source and output",
                    ));
                }
                let input = checked_u32(linear.in_features(), "Linear input width")?;
                let output = checked_u32(linear.out_features(), "Linear output width")?;
                if source_shape != [u64::from(input)] || node.output_shape != [u64::from(output)] {
                    return Err(invalid(
                        "retained-BaB Linear geometry disagrees with finalized graph shapes",
                    ));
                }
                let has_bias = u32::from(linear.bias().is_some());
                let parameter_len = u64::from(output)
                    .checked_mul(u64::from(input))
                    .and_then(|weights| {
                        weights.checked_add(u64::from(output) * u64::from(has_bias))
                    })
                    .ok_or_else(|| invalid("retained-BaB Linear parameter length overflows"))?;
                layer.geometry = [output, input, has_bias, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                layer.parameters = ResidentBabWireRangeV1 {
                    start: parameters,
                    len: parameter_len,
                };
                layer.certified_errors = ResidentBabWireRangeV1 {
                    start: errors,
                    len: 2,
                };
                layer.activation = ResidentBabWireRangeV1 {
                    start: activation,
                    len: 0,
                };
                layer.beta = ResidentBabWireRangeV1 {
                    start: beta,
                    len: 0,
                };
                layer.node_abs = ResidentBabWireRangeV1 {
                    start: node_abs,
                    len: 0,
                };
                parameters = parameters
                    .checked_add(parameter_len)
                    .ok_or_else(|| invalid("retained-BaB parameter family overflows"))?;
                errors = errors
                    .checked_add(2)
                    .ok_or_else(|| invalid("retained-BaB error family overflows"))?;
            }
            (ResidentBabLayerKindV1::Conv2d, Layer::Conv2d(conv)) => {
                if conv.groups != 1 || conv.dilation != (1, 1) || conv.kernel.ndim() != 4 {
                    return Err(invalid(
                        "retained-BaB v1 Conv2d requires dense groups=1, dilation=1 geometry",
                    ));
                }
                let source_shape = source_shape_for_id(input_shape, nodes, node.inputs[0])?;
                if source_shape.len() != 3 || node.output_shape.len() != 3 {
                    return Err(unsupported(
                        "v1 Conv2d requires exact rank-3 source and output",
                    ));
                }
                let kernel_shape = conv.kernel.shape();
                let out_c = checked_u32(kernel_shape[0], "Conv2d output channels")?;
                let in_c = checked_u32(kernel_shape[1], "Conv2d input channels")?;
                let kh = checked_u32(kernel_shape[2], "Conv2d kernel height")?;
                let kw = checked_u32(kernel_shape[3], "Conv2d kernel width")?;
                let sh = checked_u32(conv.stride.0, "Conv2d stride height")?;
                let sw = checked_u32(conv.stride.1, "Conv2d stride width")?;
                let ph = checked_u32(conv.padding.0, "Conv2d padding height")?;
                let pw = checked_u32(conv.padding.1, "Conv2d padding width")?;
                let ih = u32::try_from(source_shape[1])
                    .map_err(|_| invalid("retained-BaB Conv2d input height exceeds u32"))?;
                let iw = u32::try_from(source_shape[2])
                    .map_err(|_| invalid("retained-BaB Conv2d input width exceeds u32"))?;
                let oh = u32::try_from(node.output_shape[1])
                    .map_err(|_| invalid("retained-BaB Conv2d output height exceeds u32"))?;
                let ow = u32::try_from(node.output_shape[2])
                    .map_err(|_| invalid("retained-BaB Conv2d output width exceeds u32"))?;
                if source_shape[0] != u64::from(in_c)
                    || node.output_shape[0] != u64::from(out_c)
                    || sh == 0
                    || sw == 0
                {
                    return Err(invalid(
                        "retained-BaB Conv2d channel/stride geometry is inconsistent",
                    ));
                }
                let expected_oh = u64::from(ih)
                    .checked_add(u64::from(ph) * 2)
                    .and_then(|padded| padded.checked_sub(u64::from(kh)))
                    .map(|span| span / u64::from(sh) + 1)
                    .ok_or_else(|| invalid("retained-BaB Conv2d output height is invalid"))?;
                let expected_ow = u64::from(iw)
                    .checked_add(u64::from(pw) * 2)
                    .and_then(|padded| padded.checked_sub(u64::from(kw)))
                    .map(|span| span / u64::from(sw) + 1)
                    .ok_or_else(|| invalid("retained-BaB Conv2d output width is invalid"))?;
                if expected_oh != u64::from(oh) || expected_ow != u64::from(ow) {
                    return Err(invalid(
                        "retained-BaB Conv2d finalized shape disagrees with its exact formula",
                    ));
                }
                if conv
                    .bias
                    .as_ref()
                    .is_some_and(|bias| bias.len() != out_c as usize)
                {
                    return Err(invalid("retained-BaB Conv2d bias length is invalid"));
                }
                let has_bias = u32::from(conv.bias.is_some());
                let weights =
                    [out_c, in_c, kh, kw]
                        .into_iter()
                        .try_fold(1u64, |product, value| {
                            product.checked_mul(u64::from(value)).ok_or_else(|| {
                                invalid("retained-BaB Conv2d weight length overflows")
                            })
                        })?;
                let expanded_bias = u64::from(out_c)
                    .checked_mul(u64::from(oh))
                    .and_then(|value| value.checked_mul(u64::from(ow)))
                    .and_then(|value| value.checked_mul(u64::from(has_bias)))
                    .ok_or_else(|| invalid("retained-BaB Conv2d expanded bias overflows"))?;
                let parameter_len = weights
                    .checked_add(expanded_bias)
                    .ok_or_else(|| invalid("retained-BaB Conv2d parameters overflow"))?;
                layer.geometry = [
                    out_c, in_c, kh, kw, sh, sw, ph, pw, oh, ow, ih, iw, has_bias,
                ];
                layer.parameters = ResidentBabWireRangeV1 {
                    start: parameters,
                    len: parameter_len,
                };
                layer.certified_errors = ResidentBabWireRangeV1 {
                    start: errors,
                    len: 2,
                };
                layer.activation = ResidentBabWireRangeV1 {
                    start: activation,
                    len: 0,
                };
                layer.beta = ResidentBabWireRangeV1 {
                    start: beta,
                    len: 0,
                };
                layer.node_abs = ResidentBabWireRangeV1 {
                    start: node_abs,
                    len: 0,
                };
                parameters = parameters
                    .checked_add(parameter_len)
                    .ok_or_else(|| invalid("retained-BaB parameter family overflows"))?;
                errors = errors
                    .checked_add(2)
                    .ok_or_else(|| invalid("retained-BaB error family overflows"))?;
            }
            (ResidentBabLayerKindV1::Relu, Layer::ReLU(_)) => {
                let width = node.output_values;
                let activation_len = width
                    .checked_mul(6)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| invalid("retained-BaB Activation family overflows"))?;
                let width_u32 = u32::try_from(width)
                    .map_err(|_| invalid("retained-BaB ReLU width exceeds u32"))?;
                layer.geometry[0] = width_u32;
                layer.parameters = ResidentBabWireRangeV1 {
                    start: parameters,
                    len: 0,
                };
                layer.certified_errors = ResidentBabWireRangeV1 {
                    start: errors,
                    len: 0,
                };
                layer.activation = ResidentBabWireRangeV1 {
                    start: activation,
                    len: activation_len,
                };
                layer.beta = ResidentBabWireRangeV1 {
                    start: beta,
                    len: width,
                };
                layer.node_abs = ResidentBabWireRangeV1 {
                    start: node_abs,
                    len: width,
                };
                activation = activation
                    .checked_add(activation_len)
                    .ok_or_else(|| invalid("retained-BaB Activation family overflows"))?;
                beta = beta
                    .checked_add(width)
                    .ok_or_else(|| invalid("retained-BaB Beta family overflows"))?;
                node_abs = node_abs
                    .checked_add(width)
                    .ok_or_else(|| invalid("retained-BaB Abs family overflows"))?;
                relu_count = relu_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("retained-BaB ReLU count overflows"))?;
            }
            _ => {
                return Err(invalid(
                    "retained-BaB decoded layer no longer matches the configured graph kind",
                ));
            }
        }
    }
    check("resident static layer range composition final")?;
    for (label, value) in [
        ("parameters", parameters),
        ("certified errors", errors),
        ("activation", activation),
        ("beta", beta),
        ("abs", node_abs),
    ] {
        if value > GPU_BAB_BOUND_MAX_ARENA_VALUES as u64 {
            return Err(unsupported(match label {
                "parameters" => "parameter family exceeds the core arena cap",
                "certified errors" => "certified-error family exceeds the core arena cap",
                "activation" => "Activation family exceeds the core arena cap",
                "beta" => "Beta family exceeds the core arena cap",
                _ => "Abs family exceeds the core arena cap",
            }));
        }
    }
    topology.relu_count = relu_count;
    topology.families = ResidentBabFamilyLengthsV1 {
        parameters,
        certified_errors: errors,
        activation,
        beta,
        abs: node_abs,
        box_values: u64_shape_product(&topology.input_shape, "input")?,
        cached_la: 0,
        topology_metadata: 0,
    };
    if parameters == 0 {
        return Err(unsupported(
            "v1 requires a nonempty static parameter family",
        ));
    }
    Ok(())
}

fn compose_configured_topology_v1(
    source: &ResidentBabStaticSourceV1<'_>,
    cap: ResidentBabAdapterHostCapV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<ConfiguredTopologyBuildV1, ResidentBabComposeErrorV1> {
    let minimum_required = cap
        .resident_bytes_before
        .checked_add(size_of::<ConfiguredTopologyBuildV1>())
        .ok_or_else(|| invalid("retained-BaB static topology baseline overflows"))?;
    if cap.limit_bytes == 0 || minimum_required > cap.limit_bytes {
        return Err(ResidentBabComposeErrorV1::Capacity {
            required_bytes: minimum_required,
            limit_bytes: cap.limit_bytes,
        });
    }
    if source.input.has_l2_constraint() || source.initial_output.has_l2_constraint() {
        return Err(unsupported(
            "static bounds carry an unencoded L2 annotation",
        ));
    }
    if !source.input.lower().is_standard_layout()
        || !source.input.upper().is_standard_layout()
        || !source.initial_output.lower().is_standard_layout()
        || !source.initial_output.upper().is_standard_layout()
    {
        return Err(unsupported("static bounds use nonstandard ndarray storage"));
    }
    eligible_shape_product(source.input.shape(), "input")?;
    eligible_shape_product(source.initial_output.shape(), "root output")?;

    let exec_order = source
        .graph
        .retained_v1_exec_order_if_cached()
        .ok_or_else(|| unsupported("configured execution order cache is cold"))?;
    if exec_order.len() > RESIDENT_BAB_MAX_RECORDS_V1 {
        return Err(unsupported(
            "configured graph record count exceeds the v1 cap",
        ));
    }
    if exec_order.is_empty() || exec_order.len() != source.graph.nodes.len() {
        return Err(invalid(
            "retained-BaB cached execution order is empty, oversized, or incomplete",
        ));
    }
    let output_name = if source.graph.output_name().is_empty() {
        exec_order
            .last()
            .map(String::as_str)
            .ok_or_else(|| invalid("retained-BaB configured graph has no effective output"))?
    } else {
        source.graph.output_name()
    };
    if output_name.len() > RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1 {
        return Err(unsupported("configured output name exceeds the v1 cap"));
    }

    let nominal_bytes = topology_nominal_bytes(exec_order.len())?;
    let mut budget = ResidentBabHostBudgetV1::begin(cap, nominal_bytes)?;
    check("resident static topology prospective admission")?;

    let mut name_to_old = HashMap::new();
    reserve_hash_map(&mut budget, &mut name_to_old, exec_order.len(), check)?;
    for (index, name) in exec_order.iter().enumerate() {
        check("resident static topology cached-name binding")?;
        if name.is_empty() {
            return Err(invalid("retained-BaB configured node name is empty"));
        }
        if name.len() > RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1 {
            return Err(unsupported("configured node name exceeds the v1 cap"));
        }
        let node = source.graph.node(name).ok_or_else(|| {
            invalid("retained-BaB cached execution order names a missing configured node")
        })?;
        if node.name() != name
            || name_to_old
                .insert(name.as_str(), checked_u32(index, "cached node ID")?)
                .is_some()
        {
            return Err(invalid(
                "retained-BaB cached execution order has inconsistent or duplicate names",
            ));
        }
    }
    check("resident static topology cached-name binding final")?;

    let output_old = *name_to_old
        .get(output_name)
        .ok_or_else(|| invalid("retained-BaB configured output is absent from execution order"))?;
    let mut selected = Vec::new();
    budget.reserve_vec(&mut selected, exec_order.len(), "topology selected bitmap")?;
    check("resident static topology selected reserve")?;
    initialize_vec(
        &mut selected,
        exec_order.len(),
        0u8,
        check,
        "resident static topology selected initialization",
    )?;
    selected[output_old as usize] = 1;
    for old in (0..exec_order.len()).rev() {
        poll_scaled(check, "resident static topology ancestor marking", old)?;
        if selected[old] == 0 {
            continue;
        }
        check("resident static topology selected-node lookup")?;
        let node = source.graph.node(&exec_order[old]).ok_or_else(|| {
            invalid("retained-BaB selected configured node disappeared during composition")
        })?;
        let kind = node_kind(node.layer())?;
        let expected_inputs = if kind == ResidentBabNodeKindV1::Add {
            2
        } else {
            1
        };
        if node.inputs().len() != expected_inputs {
            return Err(invalid(
                "retained-BaB selected configured node has noncanonical arity",
            ));
        }
        for input in node.inputs() {
            check("resident static topology selected input binding")?;
            if input == NETWORK_INPUT {
                continue;
            }
            if input.is_empty() || input.len() > RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1 {
                if input.is_empty() {
                    return Err(invalid(
                        "retained-BaB selected configured input name is empty",
                    ));
                }
                return Err(unsupported("configured input name exceeds the v1 cap"));
            }
            let parent = *name_to_old
                .get(input.as_str())
                .ok_or_else(|| invalid("retained-BaB configured graph has a dangling input"))?;
            if parent as usize >= old {
                return Err(invalid(
                    "retained-BaB cached execution order is not topological",
                ));
            }
            selected[parent as usize] = 1;
        }
    }
    check("resident static topology ancestor marking final")?;

    let mut remap = Vec::new();
    budget.reserve_vec(&mut remap, exec_order.len(), "topology compact remap")?;
    check("resident static topology compact-remap reserve")?;
    initialize_vec(
        &mut remap,
        exec_order.len(),
        RESIDENT_BAB_NETWORK_INPUT_ID_V1,
        check,
        "resident static topology compact-remap initialization",
    )?;
    let mut selected_count = 0usize;
    for (old, &is_selected) in selected.iter().enumerate() {
        poll_scaled(check, "resident static topology compact-remap fill", old)?;
        if is_selected != 0 {
            remap[old] = checked_u32(selected_count, "compact node ID")?;
            selected_count = selected_count
                .checked_add(1)
                .ok_or_else(|| invalid("retained-BaB compact node count overflows"))?;
        }
    }
    check("resident static topology compact-remap fill final")?;
    if selected_count == 0 || selected_count > RESIDENT_BAB_MAX_RECORDS_V1 {
        return Err(invalid("retained-BaB selected output cone is invalid"));
    }

    let input_shape = copy_shape_u64(
        source.input.shape(),
        &mut budget,
        check,
        "resident static topology input-shape reserve",
    )?;
    let output_shape = copy_shape_u64(
        source.initial_output.shape(),
        &mut budget,
        check,
        "resident static topology output-shape reserve",
    )?;
    let mut nodes = Vec::new();
    budget.reserve_vec_full(&mut nodes, selected_count, "topology node table")?;
    check("resident static topology node-table reserve")?;
    for (old, name) in exec_order.iter().enumerate() {
        poll_scaled(check, "resident static topology node materialization", old)?;
        if selected[old] == 0 {
            continue;
        }
        check("resident static topology node-name materialization")?;
        let graph_node = source.graph.node(name).ok_or_else(|| {
            invalid("retained-BaB selected configured node disappeared during materialization")
        })?;
        let kind = node_kind(graph_node.layer())?;
        let shape = node_shape(source, name, output_name)?;
        let output_values = checked_u64(
            eligible_shape_product(shape, "selected configured node output")?,
            "selected node values",
        )?;
        let mut encoded_name = String::new();
        budget.reserve_string_full(&mut encoded_name, name.len(), "topology node-name storage")?;
        check("resident static topology node-name reserve")?;
        encoded_name.push_str(name);
        let mut inputs = Vec::new();
        budget.reserve_vec_full(
            &mut inputs,
            graph_node.inputs().len(),
            "topology node-input storage",
        )?;
        check("resident static topology node-input reserve")?;
        for input in graph_node.inputs() {
            check("resident static topology node-input materialization")?;
            inputs.push(compact_input_id(input, &name_to_old, &remap)?);
        }
        let output_shape = copy_shape_u64(
            shape,
            &mut budget,
            check,
            "resident static topology node-shape reserve",
        )?;
        if kind == ResidentBabNodeKindV1::Add {
            let left_shape = source_shape_for_id(&input_shape, &nodes, inputs[0])?;
            let right_shape = source_shape_for_id(&input_shape, &nodes, inputs[1])?;
            validate_selected_add_shape_v1(left_shape, right_shape, &output_shape)?;
        }
        let relu_preactivation_node_id = (kind == ResidentBabNodeKindV1::Relu).then_some(inputs[0]);
        nodes.push(ResidentBabNodeV1 {
            id: remap[old],
            name: encoded_name,
            kind,
            inputs,
            relu_preactivation_node_id,
            output_shape,
            output_values,
        });
    }
    check("resident static topology node materialization final")?;

    let mut segments = Vec::new();
    budget.reserve_vec_full(&mut segments, selected_count, "topology segment table")?;
    check("resident static topology segment-table reserve")?;
    let mut layers = Vec::new();
    budget.reserve_vec_full(&mut layers, selected_count, "topology layer table")?;
    check("resident static topology layer-table reserve")?;
    let mut marks = Vec::new();
    budget.reserve_vec_full(&mut marks, selected_count, "topology residual marks")?;
    check("resident static topology residual-mark reserve")?;
    initialize_vec(
        &mut marks,
        selected_count,
        0u32,
        check,
        "resident static topology residual-mark initialization",
    )?;

    let mut topology = ResidentBabTopologyV1 {
        input_shape,
        output_shape,
        output_node_id: remap[output_old as usize],
        nodes,
        segments,
        layers,
        relu_count: 0,
        families: ResidentBabFamilyLengthsV1::default(),
    };
    let mut cursor = topology.output_node_id;
    let mut frontier_abs = 0u64;
    let mut residual_epoch = 0u32;
    while cursor != RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        check("resident static topology segment cursor")?;
        while cursor != RESIDENT_BAB_NETWORK_INPUT_ID_V1
            && is_structural(topology.nodes[cursor as usize].kind)
        {
            cursor = unary_parent(&topology, cursor)?;
            check("resident static topology structural seam")?;
        }
        if cursor == RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
            break;
        }
        if topology.nodes[cursor as usize].kind == ResidentBabNodeKindV1::Add {
            let merge = cursor;
            let merge_index = usize::try_from(merge)
                .map_err(|_| invalid("retained-BaB residual merge ID exceeds usize"))?;
            let merge_inputs = &topology.nodes[merge_index].inputs;
            if merge_inputs.len() != 2 {
                return Err(invalid("retained-BaB residual Add arity is invalid"));
            }
            let inputs = [merge_inputs[0], merge_inputs[1]];
            residual_epoch = residual_epoch
                .checked_add(1)
                .ok_or_else(|| invalid("retained-BaB residual epoch overflows"))?;
            let frontier = find_residual_frontier(
                &topology,
                inputs[0],
                inputs[1],
                &mut marks,
                residual_epoch,
                check,
            )?;
            let segment_id = checked_u32(topology.segments.len(), "segment ID")?;
            let first_layer = checked_u32(topology.layers.len(), "segment first layer")?;
            let left_identity = inputs[0] == frontier;
            let right_identity = inputs[1] == frontier;
            let (kind, main_count, projection_count) = if left_identity != right_identity {
                let main_start = if left_identity { inputs[1] } else { inputs[0] };
                let main_count = trace_branch_into_layers(
                    &mut topology,
                    main_start,
                    frontier,
                    segment_id,
                    ResidentBabLayerBranchV1::Main,
                    check,
                )?;
                if main_count == 0 {
                    return Err(unsupported(
                        "retained-BaB identity residual main branch has no executable layer",
                    ));
                }
                (ResidentBabSegmentKindV1::Residual, main_count, 0)
            } else {
                let main_count = trace_branch_into_layers(
                    &mut topology,
                    inputs[0],
                    frontier,
                    segment_id,
                    ResidentBabLayerBranchV1::Main,
                    check,
                )?;
                let projection_count = trace_branch_into_layers(
                    &mut topology,
                    inputs[1],
                    frontier,
                    segment_id,
                    ResidentBabLayerBranchV1::Projection,
                    check,
                )?;
                if main_count == 0 || projection_count == 0 {
                    return Err(unsupported(
                        "retained-BaB projection residual branches must both execute layers",
                    ));
                }
                (
                    ResidentBabSegmentKindV1::ResidualProjection,
                    main_count,
                    projection_count,
                )
            };
            let frontier_width = u64_shape_product(
                source_shape_for_id(&topology.input_shape, &topology.nodes, frontier)?,
                "residual frontier",
            )?;
            topology.segments.push(ResidentBabSegmentV1 {
                id: segment_id,
                kind,
                first_layer,
                main_layer_count: main_count,
                projection_layer_count: projection_count,
                frontier_node_id: frontier,
                merge_node_id: Some(merge),
                frontier_branch: ResidentBabFrontierBranchV1::SharedResidualInput,
                frontier_abs: ResidentBabWireRangeV1 {
                    start: frontier_abs,
                    len: frontier_width,
                },
            });
            frontier_abs = frontier_abs
                .checked_add(frontier_width)
                .ok_or_else(|| invalid("retained-BaB frontier Abs family overflows"))?;
            cursor = frontier;
            continue;
        }

        let segment_id = checked_u32(topology.segments.len(), "segment ID")?;
        let first_layer = checked_u32(topology.layers.len(), "segment first layer")?;
        let mut main_count = 0u32;
        while cursor != RESIDENT_BAB_NETWORK_INPUT_ID_V1
            && topology.nodes[cursor as usize].kind != ResidentBabNodeKindV1::Add
        {
            check("resident static topology chain trace")?;
            let kind = topology.nodes[cursor as usize].kind;
            if is_executable(kind) {
                push_layer_record(
                    &mut topology,
                    cursor,
                    segment_id,
                    ResidentBabLayerBranchV1::Main,
                )?;
                main_count = main_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("retained-BaB chain layer count overflows"))?;
            } else if !is_structural(kind) {
                return Err(invalid("retained-BaB chain contains an unsupported node"));
            }
            cursor = unary_parent(&topology, cursor)?;
        }
        if main_count == 0 {
            return Err(unsupported(
                "retained-BaB canonical chain contains no executable layer",
            ));
        }
        let frontier_width = u64_shape_product(
            source_shape_for_id(&topology.input_shape, &topology.nodes, cursor)?,
            "chain frontier",
        )?;
        topology.segments.push(ResidentBabSegmentV1 {
            id: segment_id,
            kind: ResidentBabSegmentKindV1::Chain,
            first_layer,
            main_layer_count: main_count,
            projection_layer_count: 0,
            frontier_node_id: cursor,
            merge_node_id: None,
            frontier_branch: ResidentBabFrontierBranchV1::Main,
            frontier_abs: ResidentBabWireRangeV1 {
                start: frontier_abs,
                len: frontier_width,
            },
        });
        frontier_abs = frontier_abs
            .checked_add(frontier_width)
            .ok_or_else(|| invalid("retained-BaB frontier Abs family overflows"))?;
    }
    check("resident static topology segment composition final")?;
    if topology.segments.is_empty() || topology.layers.is_empty() {
        return Err(unsupported(
            "retained-BaB v1 topology has no executable static fold",
        ));
    }
    configure_layer_ranges(&mut topology, source.graph, check)?;

    // Keep the scratch allocations live through the final retained-size scan:
    // the admitted peak is therefore a real simultaneous upper bound, not a
    // sum of mutually exclusive phase receipts.
    let adapter_host_retained_bytes_after =
        topology_retained_bytes(&topology, cap.resident_bytes_before, check)?;
    if adapter_host_retained_bytes_after > budget.peak_bytes() {
        return Err(invalid(
            "retained-BaB static topology retained charge exceeds its admitted peak",
        ));
    }
    drop((name_to_old, selected, remap, marks));
    Ok(ConfiguredTopologyBuildV1 {
        graph_scope: source.graph.cut_fold_scope(),
        topology,
        adapter_host_peak_bytes: budget.peak_bytes(),
        adapter_host_retained_bytes_after,
    })
}

#[derive(Clone, Copy)]
struct StaticTensorLengthsV1 {
    parameters: usize,
    certified_errors: usize,
    input_values: usize,
    output_values: usize,
    objectives: usize,
    objective_values: usize,
}

fn wire_compose_error(error: NyError, allocation_label: &'static str) -> ResidentBabComposeErrorV1 {
    match error {
        NyError::InvalidSpec(message) if message.contains("allocation was refused") => {
            ResidentBabComposeErrorV1::AllocationRefused(allocation_label)
        }
        error => error.into(),
    }
}

fn admit_topology_wire_length_v1(
    preflight: ResidentBabTopologyWireLengthPreflightV1,
) -> Result<usize, ResidentBabComposeErrorV1> {
    match preflight {
        ResidentBabTopologyWireLengthPreflightV1::Encodable { encoded_bytes } => Ok(encoded_bytes),
        ResidentBabTopologyWireLengthPreflightV1::ExceedsV1ByteCap => Err(unsupported(
            "configured output topology exceeds the aggregate v1 wire-byte cap",
        )),
    }
}

fn derive_static_tensor_lengths_v1(
    source: &ResidentBabStaticSourceV1<'_>,
    topology: &ResidentBabTopologyV1,
) -> Result<StaticTensorLengthsV1, ResidentBabComposeErrorV1> {
    let parameters = usize::try_from(topology.families.parameters)
        .map_err(|_| invalid("retained-BaB parameter family exceeds usize"))?;
    let certified_errors = usize::try_from(topology.families.certified_errors)
        .map_err(|_| invalid("retained-BaB error family exceeds usize"))?;
    let input_values = shape_product(source.input.shape(), "static input")?;
    let output_values = shape_product(source.initial_output.shape(), "static root output")?;
    if parameters == 0
        || parameters > GPU_BAB_BOUND_MAX_ARENA_VALUES
        || certified_errors > GPU_BAB_BOUND_MAX_ARENA_VALUES
        || topology.families.box_values != u64::try_from(input_values).unwrap_or(u64::MAX)
        || !shape_matches_usize(&topology.input_shape, source.input.shape())
        || !shape_matches_usize(&topology.output_shape, source.initial_output.shape())
    {
        return Err(invalid(
            "retained-BaB static family lengths disagree with finalized shapes",
        ));
    }
    let objectives = source.sign_normalized_objectives.len();
    if objectives == 0 {
        return Err(invalid("retained-BaB static objective set is empty"));
    }
    if objectives > GPU_BAB_BOUND_MAX_OBJECTIVES {
        return Err(unsupported("objective count exceeds the core cap"));
    }
    let objective_values = objectives
        .checked_mul(output_values)
        .ok_or_else(|| invalid("retained-BaB objective matrix length overflows"))?;
    if objective_values > GPU_BAB_BOUND_MAX_ARENA_VALUES {
        return Err(unsupported(
            "objective matrix exceeds the core arena-value cap",
        ));
    }
    Ok(StaticTensorLengthsV1 {
        parameters,
        certified_errors,
        input_values,
        output_values,
        objectives,
        objective_values,
    })
}

fn validate_static_tensor_sources_v1(
    source: &ResidentBabStaticSourceV1<'_>,
    topology: &ResidentBabTopologyV1,
    lengths: StaticTensorLengthsV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    for (index, row) in source.sign_normalized_objectives.iter().enumerate() {
        poll_scaled(check, "resident static objective-shape scan", index)?;
        if row.len() != lengths.output_values {
            return Err(invalid(
                "retained-BaB sign-normalized objective row has the wrong width",
            ));
        }
    }
    check("resident static objective-shape scan final")?;
    for (index, layer) in topology.layers.iter().enumerate() {
        poll_scaled(check, "resident static parameter-layout scan", index)?;
        let node = topology
            .nodes
            .get(
                usize::try_from(layer.node_id)
                    .map_err(|_| invalid("retained-BaB parameter node ID exceeds usize"))?,
            )
            .ok_or_else(|| invalid("retained-BaB parameter node is missing"))?;
        check("resident static parameter-layout name lookup")?;
        let live = source.graph.node(&node.name).ok_or_else(|| {
            invalid("retained-BaB parameter source is absent from the configured graph")
        })?;
        let standard = match (layer.kind, live.layer()) {
            (ResidentBabLayerKindV1::Linear, Layer::Linear(linear)) => {
                linear.weight().as_slice().is_some()
                    && linear.bias().is_none_or(|bias| bias.as_slice().is_some())
            }
            (ResidentBabLayerKindV1::Conv2d, Layer::Conv2d(conv)) => {
                conv.kernel.as_slice().is_some()
                    && conv
                        .bias
                        .as_ref()
                        .is_none_or(|bias| bias.as_slice().is_some())
            }
            (ResidentBabLayerKindV1::Relu, Layer::ReLU(_)) => true,
            _ => {
                return Err(invalid(
                    "retained-BaB parameter source changed after topology rebind",
                ));
            }
        };
        if !standard {
            return Err(unsupported(
                "static layer parameters use nonstandard ndarray storage",
            ));
        }
    }
    check("resident static parameter-layout scan final")?;
    Ok(())
}

fn static_tensor_nominal_bytes_v1(
    source: &ResidentBabStaticSourceV1<'_>,
    lengths: StaticTensorLengthsV1,
    topology_capacity: usize,
) -> Result<usize, ResidentBabComposeErrorV1> {
    let mut total = size_of::<ResidentBabStaticPayloadV1>();
    checked_add(&mut total, checked_elements::<GpuBabBoundF32Tensor>(8)?)?;
    checked_add(&mut total, checked_elements::<GpuBabBoundU32Tensor>(2)?)?;
    for rank in [
        1usize,
        1,
        1,
        source.input.shape().len(),
        source.input.shape().len(),
        source.initial_output.shape().len(),
        source.initial_output.shape().len(),
        2,
        1,
        1,
    ] {
        checked_add(&mut total, checked_elements::<usize>(rank)?)?;
    }
    let f32_values = lengths
        .parameters
        .checked_add(lengths.certified_errors)
        .and_then(|value| value.checked_add(lengths.input_values.checked_mul(2)?))
        .and_then(|value| value.checked_add(lengths.output_values.checked_mul(2)?))
        .and_then(|value| value.checked_add(lengths.objective_values))
        .ok_or_else(|| invalid("retained-BaB static f32 value count overflows"))?;
    checked_add(&mut total, checked_elements::<f32>(f32_values)?)?;
    checked_add(&mut total, checked_elements::<u32>(lengths.objectives)?)?;
    checked_add(&mut total, topology_capacity)?;
    let owned_headers = GpuBabBoundOwnedSlice::<u8>::fixed_charged_bytes()
        .checked_mul(11)
        .ok_or_else(|| invalid("retained-BaB owned-slice header charge overflows"))?;
    checked_add(&mut total, owned_headers)?;
    Ok(total)
}

fn reserve_shape_v1(
    shape: &[usize],
    budget: &mut ResidentBabHostBudgetV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
    label: &'static str,
) -> Result<Vec<usize>, ResidentBabComposeErrorV1> {
    let mut owned = Vec::new();
    budget.reserve_vec(&mut owned, shape.len(), label)?;
    check(label)?;
    for (index, &dim) in shape.iter().enumerate() {
        poll_scaled(check, label, index)?;
        owned.push(dim);
    }
    check(label)?;
    Ok(owned)
}

fn reserve_f32_values_v1(
    count: usize,
    budget: &mut ResidentBabHostBudgetV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
    label: &'static str,
) -> Result<Vec<f32>, ResidentBabComposeErrorV1> {
    let mut values = Vec::new();
    budget.reserve_vec(&mut values, count, label)?;
    check(label)?;
    Ok(values)
}

fn reserve_u32_values_v1(
    count: usize,
    budget: &mut ResidentBabHostBudgetV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
    label: &'static str,
) -> Result<Vec<u32>, ResidentBabComposeErrorV1> {
    let mut values = Vec::new();
    budget.reserve_vec(&mut values, count, label)?;
    check(label)?;
    Ok(values)
}

fn checked_finite(value: f32, label: &'static str) -> Result<f32, ResidentBabComposeErrorV1> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(format!(
            "retained-BaB {label} contains a nonfinite f32 value"
        )))
    }
}

fn fill_parameter_values_v1(
    topology: &ResidentBabTopologyV1,
    graph: &GraphNetwork,
    values: &mut Vec<f32>,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    for (layer_index, layer) in topology.layers.iter().enumerate() {
        poll_scaled(check, "resident static parameter layer", layer_index)?;
        if layer.parameters.start != u64::try_from(values.len()).unwrap_or(u64::MAX) {
            return Err(invalid(
                "retained-BaB parameter range does not start at the fill cursor",
            ));
        }
        let node_index = usize::try_from(layer.node_id)
            .map_err(|_| invalid("retained-BaB parameter layer node ID exceeds usize"))?;
        let node = topology
            .nodes
            .get(node_index)
            .ok_or_else(|| invalid("retained-BaB parameter layer node is missing"))?;
        check("resident static parameter layer-name lookup")?;
        let live = graph.node(&node.name).ok_or_else(|| {
            invalid("retained-BaB parameter layer is absent from the configured graph")
        })?;
        let start = values.len();
        match (layer.kind, live.layer()) {
            (ResidentBabLayerKindV1::Linear, Layer::Linear(linear)) => {
                for (index, &value) in linear.weight().iter().enumerate() {
                    poll_scaled(check, "resident static Linear parameter copy", index)?;
                    values.push(checked_finite(value, "Linear weights")?);
                }
                if let Some(bias) = linear.bias() {
                    for (index, &value) in bias.iter().enumerate() {
                        poll_scaled(check, "resident static Linear bias copy", index)?;
                        values.push(checked_finite(value, "Linear bias")?);
                    }
                }
                check("resident static Linear parameter copy final")?;
            }
            (ResidentBabLayerKindV1::Conv2d, Layer::Conv2d(conv)) => {
                for (index, &value) in conv.kernel.iter().enumerate() {
                    poll_scaled(check, "resident static Conv2d kernel copy", index)?;
                    values.push(checked_finite(value, "Conv2d kernel")?);
                }
                if let Some(bias) = &conv.bias {
                    let spatial = usize::try_from(layer.geometry[8])
                        .ok()
                        .and_then(|height| {
                            usize::try_from(layer.geometry[9])
                                .ok()
                                .and_then(|width| height.checked_mul(width))
                        })
                        .ok_or_else(|| invalid("retained-BaB expanded Conv2d bias overflows"))?;
                    let mut copied = 0usize;
                    for &value in bias {
                        let value = checked_finite(value, "Conv2d bias")?;
                        for _ in 0..spatial {
                            poll_scaled(
                                check,
                                "resident static Conv2d expanded-bias copy",
                                copied,
                            )?;
                            values.push(value);
                            copied = copied.checked_add(1).ok_or_else(|| {
                                invalid("retained-BaB expanded Conv2d bias cursor overflows")
                            })?;
                        }
                    }
                    check("resident static Conv2d expanded-bias copy final")?;
                }
            }
            (ResidentBabLayerKindV1::Relu, Layer::ReLU(_)) => {}
            _ => {
                return Err(invalid(
                    "retained-BaB parameter source changed after decoded rebind",
                ));
            }
        }
        if u64::try_from(values.len() - start) != Ok(layer.parameters.len) {
            return Err(invalid(
                "retained-BaB copied parameter length disagrees with the wire range",
            ));
        }
    }
    check("resident static parameter layers final")?;
    Ok(())
}

fn fill_certified_errors_v1(
    topology: &ResidentBabTopologyV1,
    values: &mut Vec<f32>,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    for (index, layer) in topology.layers.iter().enumerate() {
        poll_scaled(check, "resident static certified-error fill", index)?;
        if layer.certified_errors.start != u64::try_from(values.len()).unwrap_or(u64::MAX) {
            return Err(invalid(
                "retained-BaB certified-error range does not start at the fill cursor",
            ));
        }
        match layer.kind {
            ResidentBabLayerKindV1::Linear | ResidentBabLayerKindV1::Conv2d => {
                values.push(f32::from_bits(0));
                values.push(f32::from_bits(0));
                if layer.certified_errors.len != 2 {
                    return Err(invalid(
                        "retained-BaB static layer certified-error range is not exact",
                    ));
                }
            }
            ResidentBabLayerKindV1::Relu if layer.certified_errors.len == 0 => {}
            _ => {
                return Err(invalid(
                    "retained-BaB ReLU carries a nonempty certified-error range",
                ));
            }
        }
    }
    check("resident static certified-error fill final")?;
    Ok(())
}

fn fill_bound_pair_v1(
    bounds: &BoundedTensor,
    lower: &mut Vec<f32>,
    upper: &mut Vec<f32>,
    label: &'static str,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    if bounds.lower().len() != bounds.upper().len() {
        return Err(invalid(format!(
            "retained-BaB {label} lower/upper lengths disagree"
        )));
    }
    for (index, (&lo, &hi)) in bounds.lower().iter().zip(bounds.upper()).enumerate() {
        poll_scaled(check, label, index)?;
        let lo = checked_finite(lo, label)?;
        let hi = checked_finite(hi, label)?;
        if lo > hi {
            return Err(invalid(format!(
                "retained-BaB {label} lower bound exceeds upper bound"
            )));
        }
        lower.push(lo);
        upper.push(hi);
    }
    check(label)?;
    Ok(())
}

fn fill_objectives_v1(
    rows: &[Vec<f32>],
    values: &mut Vec<f32>,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    let mut copied = 0usize;
    for row in rows {
        check("resident static objective row copy")?;
        for &value in row {
            poll_scaled(check, "resident static objective value copy", copied)?;
            values.push(checked_finite(value, "objective coefficients")?);
            copied = copied
                .checked_add(1)
                .ok_or_else(|| invalid("retained-BaB objective copy cursor overflows"))?;
        }
    }
    check("resident static objective value copy final")?;
    Ok(())
}

fn push_f32_tensor_v1(
    tensors: &mut Vec<GpuBabBoundF32Tensor>,
    role: GpuBabBoundF32TensorRole,
    shape: Vec<usize>,
    values: Vec<f32>,
) {
    tensors.push(GpuBabBoundF32Tensor {
        role,
        shape,
        values: GpuBabBoundOwnedSlice::new(values),
    });
}

fn push_u32_tensor_v1(
    tensors: &mut Vec<GpuBabBoundU32Tensor>,
    role: GpuBabBoundU32TensorRole,
    shape: Vec<usize>,
    values: Vec<u32>,
) {
    tensors.push(GpuBabBoundU32Tensor {
        role,
        shape,
        values: GpuBabBoundOwnedSlice::new(values),
    });
}

fn decoded_topology_nested_bytes_v1(
    topology: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<usize, ResidentBabComposeErrorV1> {
    topology_retained_bytes(topology, 0, check)?
        .checked_sub(size_of::<ConfiguredTopologyBuildV1>())
        .ok_or_else(|| invalid("retained-BaB decoded topology nested charge underflows"))
}

fn static_payload_retained_bytes_v1(
    payload: &ResidentBabStaticPayloadV1,
    resident_bytes_before: usize,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<usize, ResidentBabComposeErrorV1> {
    let mut total = resident_bytes_before;
    checked_add(&mut total, size_of::<ResidentBabStaticPayloadV1>())?;
    checked_add(
        &mut total,
        payload
            .topology_bytes
            .accountable_bytes()
            .ok_or_else(|| invalid("retained-BaB topology owned charge overflows"))?,
    )?;
    checked_add(
        &mut total,
        decoded_topology_nested_bytes_v1(&payload.decoded_topology, check)?,
    )?;
    checked_add(
        &mut total,
        checked_elements::<GpuBabBoundF32Tensor>(payload.f32_tensors.capacity())?,
    )?;
    checked_add(
        &mut total,
        checked_elements::<GpuBabBoundU32Tensor>(payload.u32_tensors.capacity())?,
    )?;
    for (index, tensor) in payload.f32_tensors.iter().enumerate() {
        poll_scaled(check, "resident static retained f32 tensor scan", index)?;
        checked_add(
            &mut total,
            checked_elements::<usize>(tensor.shape.capacity())?,
        )?;
        checked_add(
            &mut total,
            tensor
                .values
                .accountable_bytes()
                .ok_or_else(|| invalid("retained-BaB f32 owned charge overflows"))?,
        )?;
    }
    check("resident static retained f32 tensor scan final")?;
    for (index, tensor) in payload.u32_tensors.iter().enumerate() {
        poll_scaled(check, "resident static retained u32 tensor scan", index)?;
        checked_add(
            &mut total,
            checked_elements::<usize>(tensor.shape.capacity())?,
        )?;
        checked_add(
            &mut total,
            tensor
                .values
                .accountable_bytes()
                .ok_or_else(|| invalid("retained-BaB u32 owned charge overflows"))?,
        )?;
    }
    check("resident static retained u32 tensor scan final")?;
    Ok(total)
}

fn static_payload_identity_v1(
    topology_bytes: &[u8],
    f32_tensors: &[GpuBabBoundF32Tensor],
    u32_tensors: &[GpuBabBoundU32Tensor],
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<[u8; 32], ResidentBabComposeErrorV1> {
    Ok(gpu_bab_bound_static_payload_identity_v1(
        RESIDENT_BAB_TOPOLOGY_SCHEMA_V1,
        topology_bytes,
        f32_tensors,
        u32_tensors,
        check,
    )?)
}

fn topologies_equal_polled_v1(
    producer: &ResidentBabTopologyV1,
    decoded: &ResidentBabTopologyV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<bool, ResidentBabComposeErrorV1> {
    if producer.output_node_id != decoded.output_node_id
        || producer.relu_count != decoded.relu_count
        || producer.families != decoded.families
        || producer.input_shape != decoded.input_shape
        || producer.output_shape != decoded.output_shape
        || producer.nodes.len() != decoded.nodes.len()
        || producer.segments.len() != decoded.segments.len()
        || producer.layers.len() != decoded.layers.len()
    {
        return Ok(false);
    }
    for (index, (left, right)) in producer.nodes.iter().zip(&decoded.nodes).enumerate() {
        check("resident static topology roundtrip node-name comparison")?;
        poll_scaled(check, "resident static topology roundtrip nodes", index)?;
        if left != right {
            return Ok(false);
        }
    }
    check("resident static topology roundtrip nodes final")?;
    for (index, (left, right)) in producer.segments.iter().zip(&decoded.segments).enumerate() {
        poll_scaled(check, "resident static topology roundtrip segments", index)?;
        if left != right {
            return Ok(false);
        }
    }
    check("resident static topology roundtrip segments final")?;
    for (index, (left, right)) in producer.layers.iter().zip(&decoded.layers).enumerate() {
        poll_scaled(check, "resident static topology roundtrip layers", index)?;
        if left != right {
            return Ok(false);
        }
    }
    check("resident static topology roundtrip layers final")?;
    Ok(true)
}

fn materialize_static_tensors_v1(
    source: &ResidentBabStaticSourceV1<'_>,
    graph_scope: CutFoldScope,
    topology_bytes: Vec<u8>,
    decoded_topology: ResidentBabTopologyV1,
    lengths: StaticTensorLengthsV1,
    mut budget: ResidentBabHostBudgetV1,
    prior_peak_bytes: usize,
    cap: ResidentBabAdapterHostCapV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<ResidentBabStaticPayloadV1, ResidentBabComposeErrorV1> {
    check("resident static tensor prospective admission")?;
    let topology_bytes = GpuBabBoundOwnedSlice::new(topology_bytes);

    let mut f32_tensors = Vec::new();
    budget.reserve_vec(&mut f32_tensors, 8, "static f32 tensor table")?;
    check("resident static f32 tensor-table reserve")?;
    let mut u32_tensors = Vec::new();
    budget.reserve_vec(&mut u32_tensors, 2, "static u32 tensor table")?;
    check("resident static u32 tensor-table reserve")?;

    let mut parameters = reserve_f32_values_v1(
        lengths.parameters,
        &mut budget,
        check,
        "resident static Parameters reserve",
    )?;
    fill_parameter_values_v1(&decoded_topology, source.graph, &mut parameters, check)?;
    if parameters.len() != lengths.parameters {
        return Err(invalid(
            "retained-BaB parameter materialization has the wrong length",
        ));
    }
    push_f32_tensor_v1(
        &mut f32_tensors,
        GpuBabBoundF32TensorRole::Parameters,
        reserve_shape_v1(
            &[lengths.parameters],
            &mut budget,
            check,
            "resident static Parameters shape reserve",
        )?,
        parameters,
    );

    let mut certified_errors = reserve_f32_values_v1(
        lengths.certified_errors,
        &mut budget,
        check,
        "resident static CertifiedErrors reserve",
    )?;
    fill_certified_errors_v1(&decoded_topology, &mut certified_errors, check)?;
    if certified_errors.len() != lengths.certified_errors {
        return Err(invalid(
            "retained-BaB certified-error materialization has the wrong length",
        ));
    }
    push_f32_tensor_v1(
        &mut f32_tensors,
        GpuBabBoundF32TensorRole::CertifiedErrors,
        reserve_shape_v1(
            &[lengths.certified_errors],
            &mut budget,
            check,
            "resident static CertifiedErrors shape reserve",
        )?,
        certified_errors,
    );

    let relaxations =
        reserve_f32_values_v1(0, &mut budget, check, "resident static Relaxations reserve")?;
    push_f32_tensor_v1(
        &mut f32_tensors,
        GpuBabBoundF32TensorRole::Relaxations,
        reserve_shape_v1(
            &[0],
            &mut budget,
            check,
            "resident static Relaxations shape reserve",
        )?,
        relaxations,
    );

    let mut input_lower = reserve_f32_values_v1(
        lengths.input_values,
        &mut budget,
        check,
        "resident static InputLower reserve",
    )?;
    let mut input_upper = reserve_f32_values_v1(
        lengths.input_values,
        &mut budget,
        check,
        "resident static InputUpper reserve",
    )?;
    fill_bound_pair_v1(
        source.input,
        &mut input_lower,
        &mut input_upper,
        "input bounds",
        check,
    )?;
    push_f32_tensor_v1(
        &mut f32_tensors,
        GpuBabBoundF32TensorRole::InputLower,
        reserve_shape_v1(
            source.input.shape(),
            &mut budget,
            check,
            "resident static InputLower shape reserve",
        )?,
        input_lower,
    );
    push_f32_tensor_v1(
        &mut f32_tensors,
        GpuBabBoundF32TensorRole::InputUpper,
        reserve_shape_v1(
            source.input.shape(),
            &mut budget,
            check,
            "resident static InputUpper shape reserve",
        )?,
        input_upper,
    );

    let mut root_lower = reserve_f32_values_v1(
        lengths.output_values,
        &mut budget,
        check,
        "resident static RootLower reserve",
    )?;
    let mut root_upper = reserve_f32_values_v1(
        lengths.output_values,
        &mut budget,
        check,
        "resident static RootUpper reserve",
    )?;
    fill_bound_pair_v1(
        source.initial_output,
        &mut root_lower,
        &mut root_upper,
        "root bounds",
        check,
    )?;
    push_f32_tensor_v1(
        &mut f32_tensors,
        GpuBabBoundF32TensorRole::RootLower,
        reserve_shape_v1(
            source.initial_output.shape(),
            &mut budget,
            check,
            "resident static RootLower shape reserve",
        )?,
        root_lower,
    );
    push_f32_tensor_v1(
        &mut f32_tensors,
        GpuBabBoundF32TensorRole::RootUpper,
        reserve_shape_v1(
            source.initial_output.shape(),
            &mut budget,
            check,
            "resident static RootUpper shape reserve",
        )?,
        root_upper,
    );

    let mut objective_coefficients = reserve_f32_values_v1(
        lengths.objective_values,
        &mut budget,
        check,
        "resident static ObjectiveCoefficients reserve",
    )?;
    fill_objectives_v1(
        source.sign_normalized_objectives,
        &mut objective_coefficients,
        check,
    )?;
    if objective_coefficients.len() != lengths.objective_values {
        return Err(invalid(
            "retained-BaB objective materialization has the wrong length",
        ));
    }
    push_f32_tensor_v1(
        &mut f32_tensors,
        GpuBabBoundF32TensorRole::ObjectiveCoefficients,
        reserve_shape_v1(
            &[lengths.objectives, lengths.output_values],
            &mut budget,
            check,
            "resident static ObjectiveCoefficients shape reserve",
        )?,
        objective_coefficients,
    );

    let mut objective_indices = reserve_u32_values_v1(
        lengths.objectives,
        &mut budget,
        check,
        "resident static ObjectiveIndices reserve",
    )?;
    for index in 0..lengths.objectives {
        poll_scaled(check, "resident static ObjectiveIndices fill", index)?;
        objective_indices.push(
            u32::try_from(index)
                .map_err(|_| invalid("retained-BaB objective ordinal exceeds u32"))?,
        );
    }
    check("resident static ObjectiveIndices fill final")?;
    push_u32_tensor_v1(
        &mut u32_tensors,
        GpuBabBoundU32TensorRole::ObjectiveIndices,
        reserve_shape_v1(
            &[lengths.objectives],
            &mut budget,
            check,
            "resident static ObjectiveIndices shape reserve",
        )?,
        objective_indices,
    );

    let topology_metadata = reserve_u32_values_v1(
        0,
        &mut budget,
        check,
        "resident static TopologyMetadata reserve",
    )?;
    push_u32_tensor_v1(
        &mut u32_tensors,
        GpuBabBoundU32TensorRole::TopologyMetadata,
        reserve_shape_v1(
            &[0],
            &mut budget,
            check,
            "resident static TopologyMetadata shape reserve",
        )?,
        topology_metadata,
    );

    let static_payload_identity_sha256 =
        static_payload_identity_v1(topology_bytes.as_slice(), &f32_tensors, &u32_tensors, check)?;
    let mut payload = ResidentBabStaticPayloadV1 {
        graph_scope,
        topology_schema_version: RESIDENT_BAB_TOPOLOGY_SCHEMA_V1,
        topology_bytes,
        decoded_topology,
        f32_tensors,
        u32_tensors,
        static_payload_identity_sha256,
        adapter_host_peak_bytes: 0,
        adapter_host_retained_bytes_after: 0,
        adapter_host_exclusive_bytes: 0,
    };
    let retained = static_payload_retained_bytes_v1(&payload, cap.resident_bytes_before, check)?;
    if retained != budget.peak_bytes() {
        return Err(invalid(
            "retained-BaB final static charge disagrees with admitted observed capacities",
        ));
    }
    payload.adapter_host_peak_bytes = prior_peak_bytes.max(budget.peak_bytes());
    payload.adapter_host_retained_bytes_after = retained;
    payload.adapter_host_exclusive_bytes = retained
        .checked_sub(cap.resident_bytes_before)
        .ok_or_else(|| invalid("retained-BaB static exclusive charge underflows"))?;
    if payload.adapter_host_retained_bytes_after > payload.adapter_host_peak_bytes
        || payload.adapter_host_peak_bytes > cap.limit_bytes
    {
        return Err(invalid(
            "retained-BaB final static receipt violates retained/peak/cap ordering",
        ));
    }
    Ok(payload)
}

pub(in crate::beta_crown::engine::graph) fn compose_static_payload_v1(
    source: &ResidentBabStaticSourceV1<'_>,
    cap: ResidentBabAdapterHostCapV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<ResidentBabStaticPayloadV1, ResidentBabComposeErrorV1> {
    let producer = compose_configured_topology_v1(source, cap, check)?;
    if producer.adapter_host_retained_bytes_after > producer.adapter_host_peak_bytes
        || producer.adapter_host_peak_bytes > cap.limit_bytes
    {
        return Err(invalid(
            "retained-BaB producer topology receipt violates retained/peak/cap ordering",
        ));
    }
    let expected_topology_bytes = admit_topology_wire_length_v1(
        topology_wire_length_preflight_v1(&producer.topology, check)?,
    )?;
    let mut peak_bytes = producer.adapter_host_peak_bytes;
    let encoded = producer
        .topology
        .encode_with_baseline(
            cap.limit_bytes,
            producer.adapter_host_retained_bytes_after,
            check,
        )
        .map_err(|error| wire_compose_error(error, "topology wire encode"))?;
    peak_bytes = peak_bytes.max(encoded.adapter_host_peak_bytes);
    let encoded_exclusive = encoded
        .adapter_host_retained_bytes
        .checked_sub(producer.adapter_host_retained_bytes_after)
        .ok_or_else(|| invalid("retained-BaB encoder receipt underflows producer baseline"))?;
    let expected_encoded_exclusive = size_of_val(&encoded)
        .checked_add(encoded.bytes.capacity())
        .ok_or_else(|| invalid("retained-BaB encoder exclusive receipt overflows"))?;
    if encoded.bytes.len() != expected_topology_bytes
        || encoded_exclusive != expected_encoded_exclusive
        || encoded.adapter_host_retained_bytes > encoded.adapter_host_peak_bytes
        || encoded.adapter_host_peak_bytes > cap.limit_bytes
    {
        return Err(invalid(
            "retained-BaB encoder receipt violates exact overlap accounting",
        ));
    }
    let decode_baseline = encoded.adapter_host_retained_bytes;
    let decoded =
        ResidentBabTopologyV1::decode(&encoded.bytes, cap.limit_bytes, decode_baseline, check)
            .map_err(|error| wire_compose_error(error, "topology wire decode"))?;
    peak_bytes = peak_bytes.max(decoded.adapter_host_peak_bytes);
    let decoded_nested_bytes = decoded_topology_nested_bytes_v1(decoded.topology(), check)?;
    let decoded_own_receipt = decoded
        .adapter_host_retained_bytes_after
        .checked_sub(decode_baseline)
        .ok_or_else(|| invalid("retained-BaB decoded topology receipt underflows"))?;
    let expected_decoded_own = size_of::<ResidentBabDecodedTopologyV1>()
        .checked_add(decoded_nested_bytes)
        .ok_or_else(|| invalid("retained-BaB decoded topology receipt overflows"))?;
    if decoded_own_receipt != expected_decoded_own
        || decoded.adapter_host_retained_bytes_after > decoded.adapter_host_peak_bytes
        || decoded.adapter_host_peak_bytes > cap.limit_bytes
        || !topologies_equal_polled_v1(&producer.topology, decoded.topology(), check)?
    {
        return Err(invalid(
            "retained-BaB independent topology decode disagrees with the producer model",
        ));
    }

    let graph_scope = producer.graph_scope;
    drop(producer);
    let decoded_topology = decoded.into_topology();
    let topology_bytes = encoded.bytes;
    let lengths = derive_static_tensor_lengths_v1(source, &decoded_topology)?;
    let stage_baseline = cap
        .resident_bytes_before
        .checked_add(decoded_nested_bytes)
        .ok_or_else(|| invalid("retained-BaB final static baseline overflows"))?;
    let nominal = static_tensor_nominal_bytes_v1(source, lengths, topology_bytes.capacity())?;
    let budget = ResidentBabHostBudgetV1::begin(
        ResidentBabAdapterHostCapV1 {
            limit_bytes: cap.limit_bytes,
            resident_bytes_before: stage_baseline,
        },
        nominal,
    )?;
    rebind_decoded_topology_v1(&decoded_topology, source, graph_scope, check)?;
    validate_static_tensor_sources_v1(source, &decoded_topology, lengths, check)?;
    materialize_static_tensors_v1(
        source,
        graph_scope,
        topology_bytes,
        decoded_topology,
        lengths,
        budget,
        peak_bytes,
        cap,
        check,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::mem::size_of;
    use std::sync::Arc;

    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_core::{GpuBabBoundF32TensorRole, GpuBabBoundU32TensorRole, NyError};
    use ny_tensor::BoundedTensor;

    use crate::layers::{AddLayer, Conv2dLayer, ExpLayer, ReshapeLayer};
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

    use super::{
        compose_static_payload_v1, ResidentBabAdapterHostCapV1, ResidentBabComposeErrorV1,
        ResidentBabStaticSourceV1,
    };

    struct ChainFixture {
        graph: GraphNetwork,
        input: BoundedTensor,
        node_bounds: HashMap<String, Arc<BoundedTensor>>,
        initial_output: BoundedTensor,
        objectives: Vec<Vec<f32>>,
    }

    impl ChainFixture {
        fn source(&self) -> ResidentBabStaticSourceV1<'_> {
            ResidentBabStaticSourceV1 {
                graph: &self.graph,
                input: &self.input,
                node_bounds: &self.node_bounds,
                initial_output: &self.initial_output,
                sign_normalized_objectives: &self.objectives,
            }
        }
    }

    fn bounds(lower: &[f32], upper: &[f32]) -> Arc<BoundedTensor> {
        Arc::new(
            BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
                .expect("valid test bounds"),
        )
    }

    fn chain_fixture(warm: bool) -> ChainFixture {
        let linear0 = LinearLayer::new(
            arr2(&[[1.0_f32, -2.0], [3.0, -4.0]]),
            Some(arr1(&[0.5_f32, -0.0])),
        )
        .unwrap();
        let linear1 = LinearLayer::new(arr2(&[[5.0_f32, 6.0], [-7.0, 8.0]]), None).unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear0", Layer::Linear(linear0)));
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["linear0".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear1",
            Layer::Linear(linear1),
            vec!["relu0".to_string()],
        ));
        graph.set_output("linear1");
        if warm {
            graph.exec_order().expect("warm execution order");
        }
        ChainFixture {
            graph,
            input: BoundedTensor::new(
                arr1(&[-1.0_f32, 2.0]).into_dyn(),
                arr1(&[3.0_f32, 4.0]).into_dyn(),
            )
            .unwrap(),
            node_bounds: HashMap::from([
                ("linear0".to_string(), bounds(&[-8.0, -14.0], &[0.0, 2.0])),
                ("relu0".to_string(), bounds(&[0.0, 0.0], &[0.0, 2.0])),
                ("linear1".to_string(), bounds(&[0.0, -1.0], &[12.0, 16.0])),
            ]),
            initial_output: BoundedTensor::new(
                arr1(&[-0.0_f32, -3.5]).into_dyn(),
                arr1(&[12.25_f32, 19.0]).into_dyn(),
            )
            .unwrap(),
            objectives: vec![vec![1.0, -0.0], vec![-2.0, 3.5]],
        }
    }

    fn stacked_residual_fixture() -> ChainFixture {
        let linear = |scale| {
            LinearLayer::new(
                arr2(&[[scale, 0.0_f32], [0.0, scale]]),
                Some(arr1(&[scale, -scale])),
            )
            .unwrap()
        };
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("stem", Layer::Linear(linear(1.0))));
        graph.add_node(GraphNode::new(
            "f1",
            Layer::Linear(linear(2.0)),
            vec!["stem".to_string()],
        ));
        // Identity frontier on input[0]. The main branch is deliberately the
        // second Add input so the composer cannot assume one orientation.
        graph.add_node(GraphNode::binary(
            "merge1",
            Layer::Add(AddLayer),
            "stem",
            "f1",
        ));
        graph.add_node(GraphNode::new(
            "f2",
            Layer::Linear(linear(3.0)),
            vec!["merge1".to_string()],
        ));
        // Identity frontier on input[1] for the adjacent second block.
        graph.add_node(GraphNode::binary(
            "merge2",
            Layer::Add(AddLayer),
            "f2",
            "merge1",
        ));
        graph.set_output("merge2");
        graph.exec_order().expect("warm residual execution order");
        let node_bounds = ["stem", "f1", "merge1", "f2", "merge2"]
            .into_iter()
            .map(|name| (name.to_string(), bounds(&[-4.0, -5.0], &[6.0, 7.0])))
            .collect();
        ChainFixture {
            graph,
            input: BoundedTensor::new(
                arr1(&[-1.0_f32, -2.0]).into_dyn(),
                arr1(&[1.0_f32, 2.0]).into_dyn(),
            )
            .unwrap(),
            node_bounds,
            initial_output: BoundedTensor::new(
                arr1(&[-9.0_f32, -10.0]).into_dyn(),
                arr1(&[11.0_f32, 12.0]).into_dyn(),
            )
            .unwrap(),
            objectives: vec![vec![1.0, -1.0]],
        }
    }

    fn projection_residual_fixture() -> ChainFixture {
        let linear = |scale| {
            LinearLayer::new(
                arr2(&[[scale, 0.0_f32], [0.0, scale]]),
                Some(arr1(&[scale, -scale])),
            )
            .unwrap()
        };
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("stem", Layer::Linear(linear(1.0))));
        graph.add_node(GraphNode::new(
            "main",
            Layer::Linear(linear(2.0)),
            vec!["stem".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "projection",
            Layer::Linear(linear(3.0)),
            vec!["stem".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "merge",
            Layer::Add(AddLayer),
            "main",
            "projection",
        ));
        graph.set_output("merge");
        graph.exec_order().expect("warm projection execution order");
        let node_bounds = ["stem", "main", "projection", "merge"]
            .into_iter()
            .map(|name| (name.to_string(), bounds(&[-4.0, -5.0], &[6.0, 7.0])))
            .collect();
        ChainFixture {
            graph,
            input: BoundedTensor::new(
                arr1(&[-1.0_f32, -2.0]).into_dyn(),
                arr1(&[1.0_f32, 2.0]).into_dyn(),
            )
            .unwrap(),
            node_bounds,
            initial_output: BoundedTensor::new(
                arr1(&[-8.0_f32, -9.0]).into_dyn(),
                arr1(&[10.0_f32, 11.0]).into_dyn(),
            )
            .unwrap(),
            objectives: vec![vec![1.0, -1.0]],
        }
    }

    fn broadcast_add_fixture() -> ChainFixture {
        let main = LinearLayer::new(ndarray::Array2::eye(2), None).unwrap();
        let projection = LinearLayer::new(arr2(&[[1.0_f32, -1.0]]), None).unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("main", Layer::Linear(main)));
        graph.add_node(GraphNode::from_input(
            "projection",
            Layer::Linear(projection),
        ));
        graph.add_node(GraphNode::binary(
            "merge",
            Layer::Add(AddLayer),
            "main",
            "projection",
        ));
        graph.set_output("merge");
        graph.exec_order().expect("warm broadcast execution order");
        ChainFixture {
            graph,
            input: BoundedTensor::new(
                arr1(&[-1.0_f32, -2.0]).into_dyn(),
                arr1(&[1.0_f32, 2.0]).into_dyn(),
            )
            .unwrap(),
            node_bounds: HashMap::from([
                ("main".to_string(), bounds(&[-2.0, -3.0], &[2.0, 3.0])),
                ("projection".to_string(), bounds(&[-4.0], &[4.0])),
                ("merge".to_string(), bounds(&[-6.0, -7.0], &[6.0, 7.0])),
            ]),
            initial_output: BoundedTensor::new(
                arr1(&[-6.0_f32, -7.0]).into_dyn(),
                arr1(&[6.0_f32, 7.0]).into_dyn(),
            )
            .unwrap(),
            objectives: vec![vec![1.0, -1.0]],
        }
    }

    fn conv_fixture() -> ChainFixture {
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3, 3]),
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let conv = Conv2dLayer::new(kernel, Some(arr1(&[-0.25_f32])), (1, 1), (1, 1)).unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.set_output("conv");
        graph.exec_order().expect("warm Conv2d execution order");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 3, 3]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, 3, 3]), 1.0_f32),
        )
        .unwrap();
        let output = Arc::new(
            BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[1, 3, 3]), -10.0_f32),
                ArrayD::from_elem(IxDyn(&[1, 3, 3]), 10.0_f32),
            )
            .unwrap(),
        );
        ChainFixture {
            graph,
            input,
            node_bounds: HashMap::from([("conv".to_string(), Arc::clone(&output))]),
            initial_output: (*output).clone(),
            objectives: vec![vec![1.0; 9]],
        }
    }

    fn reshape_fixture(target_shape: Vec<i64>) -> ChainFixture {
        let linear = LinearLayer::new(ndarray::Array2::eye(6), Some(arr1(&[0.0_f32; 6]))).unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.add_node(GraphNode::new(
            "reshape",
            Layer::Reshape(ReshapeLayer::new(target_shape)),
            vec!["linear".to_string()],
        ));
        graph.set_output("reshape");
        graph.exec_order().expect("warm Reshape execution order");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32; 6]).into_dyn(),
            arr1(&[1.0_f32; 6]).into_dyn(),
        )
        .unwrap();
        let linear_bounds = bounds(&[-2.0; 6], &[2.0; 6]);
        let reshape_bounds = Arc::new(
            BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[2, 3]), -2.0_f32),
                ArrayD::from_elem(IxDyn(&[2, 3]), 2.0_f32),
            )
            .unwrap(),
        );
        ChainFixture {
            graph,
            input,
            node_bounds: HashMap::from([
                ("linear".to_string(), linear_bounds),
                ("reshape".to_string(), Arc::clone(&reshape_bounds)),
            ]),
            initial_output: (*reshape_bounds).clone(),
            objectives: vec![vec![1.0; 6]],
        }
    }

    fn wide_fixture(objective_count: usize) -> ChainFixture {
        let width = super::RESIDENT_BAB_COMPOSE_POLL_STRIDE + 1;
        let linear = LinearLayer::new(
            ndarray::Array2::from_elem((1, width), 0.25_f32),
            Some(arr1(&[-0.0_f32])),
        )
        .unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("wide", Layer::Linear(linear)));
        graph.set_output("wide");
        graph.exec_order().expect("warm wide execution order");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[width]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[width]), 1.0_f32),
        )
        .unwrap();
        ChainFixture {
            graph,
            input,
            node_bounds: HashMap::from([("wide".to_string(), bounds(&[-300.0], &[300.0]))]),
            initial_output: BoundedTensor::new(
                arr1(&[-250.0_f32]).into_dyn(),
                arr1(&[250.0_f32]).into_dyn(),
            )
            .unwrap(),
            objectives: (0..objective_count)
                .map(|index| vec![if index % 2 == 0 { 1.0 } else { -1.0 }])
                .collect(),
        }
    }

    fn compose(fixture: &ChainFixture, baseline: usize) -> super::ResidentBabStaticPayloadV1 {
        let mut check = |_| Ok(());
        compose_static_payload_v1(
            &fixture.source(),
            ResidentBabAdapterHostCapV1 {
                limit_bytes: 16 << 20,
                resident_bytes_before: baseline,
            },
            &mut check,
        )
        .expect("static payload")
    }

    #[test]
    fn static_chain_materializes_exact_closed_roles_bits_and_absolute_accounting() {
        let fixture = chain_fixture(true);
        let baseline = 137usize;
        let payload = compose(&fixture, baseline);
        assert_eq!(payload.topology_schema_version(), 1);
        assert_eq!(payload.graph_scope(), fixture.graph.cut_fold_scope());
        assert!(!payload.topology_bytes().is_empty());
        assert_eq!(payload.decoded_topology().nodes.len(), 3);
        assert_eq!(payload.decoded_topology().layers.len(), 3);
        assert_eq!(payload.decoded_topology().segments.len(), 1);
        let topology = payload.decoded_topology();
        assert_eq!(
            topology.segments[0].frontier_abs,
            super::ResidentBabWireRangeV1 { start: 0, len: 2 }
        );
        assert_eq!(
            topology.layers[0].parameters,
            super::ResidentBabWireRangeV1 { start: 0, len: 4 }
        );
        assert_eq!(
            topology.layers[0].certified_errors,
            super::ResidentBabWireRangeV1 { start: 0, len: 2 }
        );
        assert_eq!(
            topology.layers[0].activation,
            super::ResidentBabWireRangeV1 { start: 0, len: 0 }
        );
        assert_eq!(
            topology.layers[1].parameters,
            super::ResidentBabWireRangeV1 { start: 4, len: 0 }
        );
        assert_eq!(
            topology.layers[1].activation,
            super::ResidentBabWireRangeV1 { start: 0, len: 13 }
        );
        assert_eq!(
            topology.layers[1].beta,
            super::ResidentBabWireRangeV1 { start: 0, len: 2 }
        );
        assert_eq!(
            topology.layers[1].node_abs,
            super::ResidentBabWireRangeV1 { start: 2, len: 2 }
        );
        assert_eq!(
            topology.layers[2].parameters,
            super::ResidentBabWireRangeV1 { start: 4, len: 6 }
        );
        assert_eq!(
            topology.layers[2].certified_errors,
            super::ResidentBabWireRangeV1 { start: 2, len: 2 }
        );
        assert_eq!(
            topology.families,
            super::ResidentBabFamilyLengthsV1 {
                parameters: 10,
                certified_errors: 4,
                activation: 13,
                beta: 2,
                abs: 4,
                box_values: 2,
                cached_la: 0,
                topology_metadata: 0,
            }
        );
        let f32_roles: Vec<_> = payload
            .f32_tensors()
            .iter()
            .map(|tensor| tensor.role)
            .collect();
        assert_eq!(
            f32_roles,
            [
                GpuBabBoundF32TensorRole::Parameters,
                GpuBabBoundF32TensorRole::CertifiedErrors,
                GpuBabBoundF32TensorRole::Relaxations,
                GpuBabBoundF32TensorRole::InputLower,
                GpuBabBoundF32TensorRole::InputUpper,
                GpuBabBoundF32TensorRole::RootLower,
                GpuBabBoundF32TensorRole::RootUpper,
                GpuBabBoundF32TensorRole::ObjectiveCoefficients,
            ]
        );
        let u32_roles: Vec<_> = payload
            .u32_tensors()
            .iter()
            .map(|tensor| tensor.role)
            .collect();
        assert_eq!(
            u32_roles,
            [
                GpuBabBoundU32TensorRole::ObjectiveIndices,
                GpuBabBoundU32TensorRole::TopologyMetadata,
            ]
        );
        let parameters = payload.f32_tensors()[0].values.as_slice();
        assert_eq!(
            parameters.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [5.0_f32, 6.0, -7.0, 8.0, 1.0, -2.0, 3.0, -4.0, 0.5, -0.0,]
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            payload.f32_tensors()[1]
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0; 4]
        );
        assert_eq!(payload.f32_tensors()[2].shape, [0]);
        assert!(payload.f32_tensors()[2].values.is_empty());
        assert_eq!(
            payload.f32_tensors()[5]
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [(-0.0_f32).to_bits(), (-3.5_f32).to_bits()]
        );
        assert_eq!(
            payload.f32_tensors()[7]
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [1.0_f32, -0.0, -2.0, 3.5]
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
        assert_eq!(payload.u32_tensors()[0].values.as_slice(), &[0, 1]);
        assert_eq!(payload.u32_tensors()[1].shape, [0]);
        assert!(payload.u32_tensors()[1].values.is_empty());
        assert!(payload.adapter_host_peak_bytes() >= payload.adapter_host_retained_bytes_after());
        assert_eq!(
            payload.adapter_host_exclusive_bytes(),
            payload.adapter_host_retained_bytes_after() - baseline
        );
        assert_ne!(payload.static_payload_identity_sha256(), &[0; 32]);
    }

    #[test]
    fn landed_static_payload_builds_the_exact_borrowed_core_schedule_request() {
        let fixture = chain_fixture(true);
        let payload = compose(&fixture, 0);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let request = payload.schedule_request(deadline, 1).unwrap();
        assert_eq!(
            request.static_payload_identity_sha256(),
            payload.static_payload_identity_sha256()
        );
        assert_eq!(
            request.topology_schema_version(),
            payload.topology_schema_version()
        );
        assert_eq!(
            request.topology_bytes(),
            payload.topology_bytes().as_slice()
        );
        assert_eq!(request.f32_tensors(), payload.f32_tensors());
        assert_eq!(request.u32_tensors(), payload.u32_tensors());
        assert!(request.logical_static_device_bytes() > 1);
        assert_eq!(request.requested_max_device_bytes(), 1);
        assert_eq!(request.deadline(), deadline);
    }

    #[test]
    fn cold_cache_is_typed_preopen_fallback_but_deadline_and_invalid_are_not() {
        let fixture = chain_fixture(false);
        assert!(fixture.graph.retained_v1_exec_order_if_cached().is_none());
        let mut callbacks = 0_usize;
        let mut check = |_| {
            callbacks += 1;
            Ok(())
        };
        let error = compose_static_payload_v1(
            &fixture.source(),
            ResidentBabAdapterHostCapV1 {
                limit_bytes: 16 << 20,
                resident_bytes_before: 0,
            },
            &mut check,
        )
        .unwrap_err();
        assert!(matches!(error, ResidentBabComposeErrorV1::Unsupported(_)));
        assert!(error.allows_preopen_legacy_fallback());
        assert_eq!(callbacks, 0);
        assert!(fixture.graph.retained_v1_exec_order_if_cached().is_none());

        let deadline = ResidentBabComposeErrorV1::from(NyError::DeadlineExceeded("test".into()));
        assert!(matches!(deadline, ResidentBabComposeErrorV1::Deadline(_)));
        assert!(!deadline.allows_preopen_legacy_fallback());
        let invalid = ResidentBabComposeErrorV1::Invalid(NyError::InvalidSpec("test".into()));
        assert!(!invalid.allows_preopen_legacy_fallback());
        assert!(ResidentBabComposeErrorV1::Capacity {
            required_bytes: 2,
            limit_bytes: 1,
        }
        .allows_preopen_legacy_fallback());
        assert!(
            ResidentBabComposeErrorV1::AllocationRefused("test").allows_preopen_legacy_fallback()
        );
    }

    #[test]
    fn adjacent_identity_residuals_preserve_both_operand_orientations_and_frontiers() {
        let fixture = stacked_residual_fixture();
        let payload = compose(&fixture, 0);
        let topology = payload.decoded_topology();
        assert_eq!(topology.segments.len(), 3);
        assert_eq!(
            topology
                .segments
                .iter()
                .map(|segment| segment.kind)
                .collect::<Vec<_>>(),
            [
                super::ResidentBabSegmentKindV1::Residual,
                super::ResidentBabSegmentKindV1::Residual,
                super::ResidentBabSegmentKindV1::Chain,
            ]
        );
        let node_name = |id: u32| {
            if id == super::RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
                "_input"
            } else {
                topology.nodes[id as usize].name.as_str()
            }
        };
        assert_eq!(
            node_name(topology.segments[0].merge_node_id.unwrap()),
            "merge2"
        );
        assert_eq!(node_name(topology.segments[0].frontier_node_id), "merge1");
        assert_eq!(
            node_name(topology.segments[1].merge_node_id.unwrap()),
            "merge1"
        );
        assert_eq!(node_name(topology.segments[1].frontier_node_id), "stem");
        assert_eq!(node_name(topology.segments[2].frontier_node_id), "_input");
        assert_eq!(
            topology
                .layers
                .iter()
                .map(|layer| node_name(layer.node_id))
                .collect::<Vec<_>>(),
            ["f2", "f1", "stem"]
        );
    }

    #[test]
    fn projection_residual_preserves_main_then_projection_order_and_exact_ranges() {
        let fixture = projection_residual_fixture();
        let payload = compose(&fixture, 0);
        let topology = payload.decoded_topology();
        assert_eq!(topology.segments.len(), 2);
        let projection = &topology.segments[0];
        assert_eq!(
            projection.kind,
            super::ResidentBabSegmentKindV1::ResidualProjection
        );
        assert_eq!(projection.first_layer, 0);
        assert_eq!(projection.main_layer_count, 1);
        assert_eq!(projection.projection_layer_count, 1);
        assert_eq!(
            (projection.frontier_abs.start, projection.frontier_abs.len),
            (0, 2)
        );
        let chain = &topology.segments[1];
        assert_eq!(chain.kind, super::ResidentBabSegmentKindV1::Chain);
        assert_eq!(chain.first_layer, 2);
        assert_eq!(chain.main_layer_count, 1);
        assert_eq!(chain.projection_layer_count, 0);
        assert_eq!((chain.frontier_abs.start, chain.frontier_abs.len), (2, 2));

        let node_name = |id: u32| {
            if id == super::RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
                "_input"
            } else {
                topology.nodes[id as usize].name.as_str()
            }
        };
        assert_eq!(node_name(projection.merge_node_id.unwrap()), "merge");
        assert_eq!(node_name(projection.frontier_node_id), "stem");
        assert_eq!(node_name(chain.frontier_node_id), "_input");
        assert_eq!(
            topology
                .layers
                .iter()
                .map(|layer| (node_name(layer.node_id), layer.branch, layer.segment_id))
                .collect::<Vec<_>>(),
            [
                ("main", super::ResidentBabLayerBranchV1::Main, 0),
                ("projection", super::ResidentBabLayerBranchV1::Projection, 0,),
                ("stem", super::ResidentBabLayerBranchV1::Main, 1),
            ]
        );
        assert_eq!(
            topology
                .layers
                .iter()
                .map(|layer| {
                    (
                        (layer.parameters.start, layer.parameters.len),
                        (layer.certified_errors.start, layer.certified_errors.len),
                        (layer.activation.start, layer.activation.len),
                        (layer.beta.start, layer.beta.len),
                        (layer.node_abs.start, layer.node_abs.len),
                    )
                })
                .collect::<Vec<_>>(),
            [
                ((0, 6), (0, 2), (0, 0), (0, 0), (4, 0)),
                ((6, 6), (2, 2), (0, 0), (0, 0), (4, 0)),
                ((12, 6), (4, 2), (0, 0), (0, 0), (4, 0)),
            ]
        );
        assert_eq!(topology.families.parameters, 18);
        assert_eq!(topology.families.certified_errors, 6);
        assert_eq!(topology.families.activation, 0);
        assert_eq!(topology.families.beta, 0);
        assert_eq!(topology.families.abs, 4);
    }

    #[test]
    fn coherent_broadcast_and_structural_only_branches_are_typed_unsupported() {
        let error = super::admit_topology_wire_length_v1(
            super::ResidentBabTopologyWireLengthPreflightV1::ExceedsV1ByteCap,
        )
        .unwrap_err();
        assert!(matches!(error, ResidentBabComposeErrorV1::Unsupported(_)));
        assert!(error.allows_preopen_legacy_fallback());

        let broadcast = broadcast_add_fixture();
        let mut check = |_| Ok(());
        let error = compose_static_payload_v1(
            &broadcast.source(),
            ResidentBabAdapterHostCapV1 {
                limit_bytes: 16 << 20,
                resident_bytes_before: 0,
            },
            &mut check,
        )
        .unwrap_err();
        assert!(matches!(error, ResidentBabComposeErrorV1::Unsupported(_)));
        assert!(error.allows_preopen_legacy_fallback());

        let mut structural = projection_residual_fixture();
        structural.graph.add_node(GraphNode::new(
            "structural",
            Layer::Reshape(ReshapeLayer::new(vec![2])),
            vec!["stem".to_string()],
        ));
        structural.graph.add_node(GraphNode::binary(
            "structural_merge",
            Layer::Add(AddLayer),
            "main",
            "structural",
        ));
        structural.graph.set_output("structural_merge");
        structural
            .graph
            .exec_order()
            .expect("warm structural-only residual execution order");
        structural
            .node_bounds
            .insert("structural".to_string(), bounds(&[-4.0, -5.0], &[6.0, 7.0]));
        structural.node_bounds.insert(
            "structural_merge".to_string(),
            bounds(&[-8.0, -9.0], &[10.0, 11.0]),
        );
        let mut check = |_| Ok(());
        let error = compose_static_payload_v1(
            &structural.source(),
            ResidentBabAdapterHostCapV1 {
                limit_bytes: 16 << 20,
                resident_bytes_before: 0,
            },
            &mut check,
        )
        .unwrap_err();
        assert!(matches!(error, ResidentBabComposeErrorV1::Unsupported(_)));
        assert!(error.allows_preopen_legacy_fallback());
    }

    #[test]
    fn topology_roundtrip_equality_observes_the_second_record_stride_deadline() {
        let fixture = chain_fixture(true);
        let payload = compose(&fixture, 0);
        let mut producer = payload.decoded_topology().clone();
        let template = producer.nodes[0].clone();
        producer.nodes.clear();
        for index in 0..=super::RESIDENT_BAB_COMPOSE_POLL_STRIDE {
            let mut node = template.clone();
            node.id = index as u32;
            node.name = format!("node_{index}");
            producer.nodes.push(node);
        }
        producer.output_node_id = super::RESIDENT_BAB_COMPOSE_POLL_STRIDE as u32;
        let decoded = producer.clone();
        let mut stride_polls = 0usize;
        let mut check = |label| {
            if label == "resident static topology roundtrip nodes" {
                stride_polls += 1;
                if stride_polls == 2 {
                    return Err(NyError::DeadlineExceeded("roundtrip stride".into()));
                }
            }
            Ok(())
        };
        let error = super::topologies_equal_polled_v1(&producer, &decoded, &mut check).unwrap_err();
        assert!(matches!(error, ResidentBabComposeErrorV1::Deadline(_)));
        assert_eq!(stride_polls, 2);
    }

    #[test]
    fn disconnected_unsupported_node_is_omitted_but_retarget_is_typed_unsupported() {
        let mut fixture = chain_fixture(false);
        fixture.graph.add_node(GraphNode::from_input(
            "dead_exp",
            Layer::Exp(ExpLayer::new()),
        ));
        fixture
            .graph
            .exec_order()
            .expect("warm graph with dead node");
        let payload = compose(&fixture, 0);
        assert!(payload
            .decoded_topology()
            .nodes
            .iter()
            .all(|node| node.name != "dead_exp"));
        assert_eq!(payload.decoded_topology().output_node_id as usize, 2);

        fixture.graph.set_output("dead_exp");
        let mut check = |_| Ok(());
        let error = compose_static_payload_v1(
            &fixture.source(),
            ResidentBabAdapterHostCapV1 {
                limit_bytes: 16 << 20,
                resident_bytes_before: 0,
            },
            &mut check,
        )
        .unwrap_err();
        assert!(matches!(error, ResidentBabComposeErrorV1::Unsupported(_)));
    }

    fn manual_decoded_nested_bytes(topology: &super::ResidentBabTopologyV1) -> usize {
        let mut total = topology.input_shape.capacity() * size_of::<u64>()
            + topology.output_shape.capacity() * size_of::<u64>()
            + topology.nodes.capacity() * size_of::<super::ResidentBabNodeV1>()
            + topology.segments.capacity() * size_of::<super::ResidentBabSegmentV1>()
            + topology.layers.capacity() * size_of::<super::ResidentBabLayerV1>();
        for node in &topology.nodes {
            total += node.name.capacity();
            total += node.inputs.capacity() * size_of::<u32>();
            total += node.output_shape.capacity() * size_of::<u64>();
        }
        total
    }

    fn manual_retained_bytes(
        payload: &super::ResidentBabStaticPayloadV1,
        baseline: usize,
    ) -> usize {
        let mut total = baseline + size_of::<super::ResidentBabStaticPayloadV1>();
        total += payload.topology_bytes().accountable_bytes().unwrap();
        total += manual_decoded_nested_bytes(payload.decoded_topology());
        total += payload.f32_tensors.capacity() * size_of::<ny_core::GpuBabBoundF32Tensor>();
        total += payload.u32_tensors.capacity() * size_of::<ny_core::GpuBabBoundU32Tensor>();
        for tensor in payload.f32_tensors() {
            total += tensor.shape.capacity() * size_of::<usize>();
            total += tensor.values.accountable_bytes().unwrap();
        }
        for tensor in payload.u32_tensors() {
            total += tensor.shape.capacity() * size_of::<usize>();
            total += tensor.values.accountable_bytes().unwrap();
        }
        total
    }

    #[test]
    fn static_host_ledger_counts_eleven_owned_slices_and_shifts_baseline_once() {
        let fixture = chain_fixture(true);
        let zero = compose(&fixture, 0);
        let shifted = compose(&fixture, 911);
        assert_eq!(
            zero.static_payload_identity_sha256(),
            shifted.static_payload_identity_sha256()
        );
        assert_eq!(
            shifted.adapter_host_peak_bytes(),
            zero.adapter_host_peak_bytes() + 911
        );
        assert_eq!(
            shifted.adapter_host_retained_bytes_after(),
            zero.adapter_host_retained_bytes_after() + 911
        );
        assert_eq!(
            shifted.adapter_host_exclusive_bytes(),
            zero.adapter_host_exclusive_bytes()
        );
        assert_eq!(
            zero.adapter_host_retained_bytes_after(),
            manual_retained_bytes(&zero, 0)
        );
        assert_eq!(
            shifted.adapter_host_retained_bytes_after(),
            manual_retained_bytes(&shifted, 911)
        );
        let fixed = ny_core::GpuBabBoundOwnedSlice::<u8>::fixed_charged_bytes();
        let owned_fixed_total = zero.topology_bytes().accountable_bytes().unwrap()
            - zero.topology_bytes().capacity()
            + zero
                .f32_tensors()
                .iter()
                .map(|tensor| {
                    tensor.values.accountable_bytes().unwrap()
                        - tensor.values.capacity() * size_of::<f32>()
                })
                .sum::<usize>()
            + zero
                .u32_tensors()
                .iter()
                .map(|tensor| {
                    tensor.values.accountable_bytes().unwrap()
                        - tensor.values.capacity() * size_of::<u32>()
                })
                .sum::<usize>();
        assert_eq!(owned_fixed_total, 11 * fixed);
        assert_eq!(zero.f32_tensors()[2].values.capacity(), 0);
        assert_eq!(
            zero.f32_tensors()[2].values.accountable_bytes(),
            Some(fixed)
        );
        assert_eq!(zero.u32_tensors()[1].values.capacity(), 0);
        assert_eq!(
            zero.u32_tensors()[1].values.accountable_bytes(),
            Some(fixed)
        );
    }

    #[test]
    fn exact_peak_cap_succeeds_and_peak_minus_one_precedes_scaled_tensor_scans() {
        let fixture = wide_fixture(4096);
        let reference = compose(&fixture, 0);
        assert_eq!(
            reference.adapter_host_peak_bytes(),
            reference.adapter_host_retained_bytes_after(),
            "fixture must make the final owned tensor stage dominate"
        );
        let exact_peak = reference.adapter_host_peak_bytes();
        let mut exact_check = |_| Ok(());
        let exact = compose_static_payload_v1(
            &fixture.source(),
            ResidentBabAdapterHostCapV1 {
                limit_bytes: exact_peak,
                resident_bytes_before: 0,
            },
            &mut exact_check,
        )
        .expect("exact observed peak must be admissible");
        assert_eq!(exact.adapter_host_peak_bytes(), exact_peak);

        let mut scaled_tensor_checks = 0usize;
        let mut check = |label: &'static str| {
            if label.contains("rebind")
                || label == "resident static objective-shape scan"
                || label == "resident static parameter-layout scan"
            {
                scaled_tensor_checks += 1;
            }
            Ok(())
        };
        let error = compose_static_payload_v1(
            &fixture.source(),
            ResidentBabAdapterHostCapV1 {
                limit_bytes: exact_peak - 1,
                resident_bytes_before: 0,
            },
            &mut check,
        )
        .unwrap_err();
        assert!(matches!(error, ResidentBabComposeErrorV1::Capacity { .. }));
        assert_eq!(scaled_tensor_checks, 0);
    }

    #[test]
    fn conv_bias_expansion_is_exact_and_live_geometry_tamper_fails_rebind() {
        let fixture = conv_fixture();
        let payload = compose(&fixture, 0);
        let topology = payload.decoded_topology();
        assert_eq!(topology.layers.len(), 1);
        assert_eq!(
            topology.layers[0].geometry,
            [1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1]
        );
        assert_eq!(
            topology.layers[0].parameters,
            super::ResidentBabWireRangeV1 { start: 0, len: 18 }
        );
        let parameters = payload.f32_tensors()[0].values.as_slice();
        assert_eq!(parameters.len(), 18);
        assert_eq!(
            &parameters[..9],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert!(parameters[9..]
            .iter()
            .all(|value| value.to_bits() == (-0.25_f32).to_bits()));

        let mut tampered = topology.clone();
        tampered.layers[0].geometry[12] = 0;
        let mut check = |_| Ok(());
        let error = super::rebind_decoded_topology_v1(
            &tampered,
            &fixture.source(),
            fixture.graph.cut_fold_scope(),
            &mut check,
        )
        .unwrap_err();
        assert!(matches!(error, ResidentBabComposeErrorV1::Invalid(_)));
    }

    #[test]
    fn equal_product_wrong_live_reshape_target_fails_independent_rebind() {
        let valid = reshape_fixture(vec![2, 3]);
        let payload = compose(&valid, 0);
        assert_eq!(payload.decoded_topology().output_shape, [2, 3]);

        // Same names, edges, finalized bounds, element count, and tensor
        // geometry, but a different live Reshape target. The wire ABI does not
        // serialize raw target spelling; the graph-owner rebind must still
        // reject this source contradiction independently.
        let wrong_live = reshape_fixture(vec![3, 2]);
        let mut check = |_| Ok(());
        let error = super::rebind_decoded_topology_v1(
            payload.decoded_topology(),
            &wrong_live.source(),
            wrong_live.graph.cut_fold_scope(),
            &mut check,
        )
        .unwrap_err();
        assert!(matches!(error, ResidentBabComposeErrorV1::Invalid(_)));
    }

    #[test]
    fn mid_parameter_bound_and_objective_copy_deadlines_stay_typed_nonfallback() {
        let fixture = wide_fixture(super::RESIDENT_BAB_COMPOSE_POLL_STRIDE + 1);
        for label in [
            "resident static Linear parameter copy",
            "input bounds",
            "resident static objective value copy",
        ] {
            let mut hits = 0usize;
            let mut check = |observed: &'static str| {
                if observed == label {
                    hits += 1;
                    if hits == 2 {
                        return Err(NyError::DeadlineExceeded(format!(
                            "injected {label} deadline"
                        )));
                    }
                }
                Ok(())
            };
            let error = compose_static_payload_v1(
                &fixture.source(),
                ResidentBabAdapterHostCapV1 {
                    limit_bytes: 16 << 20,
                    resident_bytes_before: 0,
                },
                &mut check,
            )
            .unwrap_err();
            assert!(matches!(error, ResidentBabComposeErrorV1::Deadline(_)));
            assert!(!error.allows_preopen_legacy_fallback());
            assert_eq!(hits, 2, "{label}");
        }
    }

    fn identity_for_test(
        topology: &[u8],
        f32_tensors: &[ny_core::GpuBabBoundF32Tensor],
        u32_tensors: &[ny_core::GpuBabBoundU32Tensor],
    ) -> [u8; 32] {
        let mut check = |_| Ok(());
        super::static_payload_identity_v1(topology, f32_tensors, u32_tensors, &mut check).unwrap()
    }

    #[test]
    fn static_identity_binds_topology_shape_role_order_and_every_value_bit() {
        let fixture = chain_fixture(true);
        let payload = compose(&fixture, 0);
        let base = *payload.static_payload_identity_sha256();

        let mut topology = payload.topology_bytes().as_slice().to_vec();
        let last = topology.len() - 1;
        topology[last] ^= 1;
        assert_ne!(
            identity_for_test(&topology, payload.f32_tensors(), payload.u32_tensors()),
            base
        );

        let mut shape = payload.f32_tensors().to_vec();
        shape[0].shape[0] += 1;
        assert_ne!(
            identity_for_test(payload.topology_bytes(), &shape, payload.u32_tensors()),
            base
        );

        let mut weight = payload.f32_tensors().to_vec();
        let mut weight_values = weight[0].values.as_slice().to_vec();
        weight_values[0] = f32::from_bits(weight_values[0].to_bits() ^ 1);
        weight[0].values = ny_core::GpuBabBoundOwnedSlice::new(weight_values);
        assert_ne!(
            identity_for_test(payload.topology_bytes(), &weight, payload.u32_tensors()),
            base
        );

        let mut root_zero = payload.f32_tensors().to_vec();
        let mut root_values = root_zero[5].values.as_slice().to_vec();
        assert_eq!(root_values[0].to_bits(), (-0.0_f32).to_bits());
        root_values[0] = 0.0;
        root_zero[5].values = ny_core::GpuBabBoundOwnedSlice::new(root_values);
        assert_ne!(
            identity_for_test(payload.topology_bytes(), &root_zero, payload.u32_tensors()),
            base
        );

        let mut objectives = payload.f32_tensors().to_vec();
        let old = objectives[7].values.as_slice();
        let permuted = vec![old[2], old[3], old[0], old[1]];
        objectives[7].values = ny_core::GpuBabBoundOwnedSlice::new(permuted);
        assert_ne!(
            identity_for_test(payload.topology_bytes(), &objectives, payload.u32_tensors()),
            base
        );

        let mut order = payload.f32_tensors().to_vec();
        order.swap(0, 1);
        assert_ne!(
            identity_for_test(payload.topology_bytes(), &order, payload.u32_tensors()),
            base
        );

        let source = include_str!("static.rs");
        assert!(!source.contains(concat!("dispatches_", "per_subchunk")));
        assert!(!source.contains(concat!("fn into_", "graph_plan")));
        assert!(!source.contains(concat!("GpuBabBoundGraph", "Plan {")));
    }
}
