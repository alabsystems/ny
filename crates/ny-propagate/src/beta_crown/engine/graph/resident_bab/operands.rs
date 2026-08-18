// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::mem::size_of;

use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::state::{AlphaNeuronState, GraphBetaState, GraphDomainAlphaState};
use crate::network::{gpu_relu_affine_cell, GpuReluAffineVariant};
use crate::resident_bab_wire::v1::{
    validate_composition_caps_v1, ResidentBabActivationSectionsV1, ResidentBabActivationVariantV1,
    ResidentBabFrontierBranchV1, ResidentBabLayerKindV1, ResidentBabTopologyV1,
    RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1, RESIDENT_BAB_NETWORK_INPUT_ID_V1,
};
use ny_core::{GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS, GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS};
use rustc_hash::FxHashMap;

use super::budget::{checked_add, checked_elements, invalid, poll_scaled, ResidentBabHostBudgetV1};
use super::history::{compose_history_beta_v1, history_nominal_bytes};
use super::ResidentBabReluSiteV1;
use super::{ResidentBabAdapterHostCapV1, ResidentBabComposeErrorV1};

/// Exact signed source for one canonical true-ReLU row. The composer seals the
/// ordinary or whole-row dual-alpha interpretation before emitting any value.
pub(in crate::beta_crown::engine::graph) struct ResidentBabActivationSourceV1<'a> {
    pub topology_node_id: u32,
    pub preactivation_node_id: u32,
    pub pre_lower: &'a [f32],
    pub pre_upper: &'a [f32],
}

/// Exact signed graph-owner source for one compact segment-frontier Abs row.
///
/// The caller must borrow these endpoints directly from the source
/// `BoundedTensor`; accepting a precomputed opaque Abs row is forbidden.
pub(in crate::beta_crown::engine::graph) struct ResidentBabFrontierSourceV1<'a> {
    pub segment_id: u32,
    pub branch: ResidentBabFrontierBranchV1,
    pub source_node_id: u32,
    pub lower: &'a [f32],
    pub upper: &'a [f32],
}

pub(in crate::beta_crown::engine::graph) struct ResidentBabDomainSourceV1<'a> {
    pub relu_sites: &'a [ResidentBabReluSiteV1],
    pub activations: &'a [ResidentBabActivationSourceV1<'a>],
    pub frontiers: &'a [ResidentBabFrontierSourceV1<'a>],
    pub history: &'a GraphSplitHistory,
    pub beta_state: &'a GraphBetaState,
    /// Exact sparse lower/upper alpha state. Missing unconstrained crossing
    /// neurons use the same scalar heuristic as the legacy graph backward;
    /// no dense alpha array or flattened bounds tensor is allocated.
    pub alpha_state: &'a GraphDomainAlphaState,
    pub box_lower: &'a [f32],
    pub box_upper: &'a [f32],
    /// Every cached-lA objective slot must be absent in v1. A single present
    /// entry refuses the domain; no zero placeholder is serialized.
    pub cached_la_present: bool,
}

fn activation_variant_tag_v1(variant: GpuReluAffineVariant) -> f32 {
    let wire = match variant {
        GpuReluAffineVariant::Ordinary => ResidentBabActivationVariantV1::Ordinary,
        GpuReluAffineVariant::DualAlpha => ResidentBabActivationVariantV1::DualAlpha,
    };
    f32::from_bits(wire.tag_bits())
}

fn decode_activation_variant_tag_v1(
    value: f32,
) -> Result<GpuReluAffineVariant, ResidentBabComposeErrorV1> {
    match ResidentBabActivationVariantV1::from_tag_bits(value.to_bits()) {
        Ok(ResidentBabActivationVariantV1::Ordinary) => Ok(GpuReluAffineVariant::Ordinary),
        Ok(ResidentBabActivationVariantV1::DualAlpha) => Ok(GpuReluAffineVariant::DualAlpha),
        Err(_) => Err(invalid(
            "retained-BaB Activation variant tag has noncanonical f32 bits",
        )),
    }
}

fn effective_alpha_v1(
    optimized: Option<&FxHashMap<usize, AlphaNeuronState>>,
    neuron: usize,
    lower: f32,
    upper: f32,
) -> f32 {
    if lower >= 0.0 {
        return 1.0;
    }
    if upper <= 0.0 {
        return 0.0;
    }
    optimized
        .and_then(|neurons| neurons.get(&neuron))
        .map_or_else(
            || if upper > -lower { 1.0 } else { 0.0 },
            |value| value.alpha(),
        )
}

fn validate_domain_composition_caps_v1(
    topology: &ResidentBabTopologyV1,
    source: &ResidentBabDomainSourceV1<'_>,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<(), ResidentBabComposeErrorV1> {
    let history_word_count = source
        .history
        .constraints
        .len()
        .checked_mul(GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS)
        .ok_or_else(|| invalid("retained-BaB composition history word count overflows usize"))?;
    validate_composition_caps_v1(topology, history_word_count, check)?;

    let max_history_records =
        GPU_BAB_BOUND_MAX_SPLIT_HISTORY_WORDS / GPU_BAB_BOUND_SPLIT_HISTORY_RECORD_WORDS;
    if source.relu_sites.len() > topology.layers.len()
        || source.activations.len() > topology.layers.len()
        || source.frontiers.len() > topology.segments.len()
        || source.history.constraints.len() > max_history_records
        || source.beta_state.entries.len() > max_history_records
    {
        return Err(invalid(
            "retained-BaB composition source counts exceed v1/core bounds",
        ));
    }
    for (index, site) in source.relu_sites.iter().enumerate() {
        poll_scaled(check, "resident composition ReLU-name cap", index)?;
        if site.node_name.is_empty() || site.node_name.len() > RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1 {
            return Err(invalid(
                "retained-BaB composition ReLU-site name exceeds the v1 bound",
            ));
        }
    }
    check("resident composition ReLU-name cap final")?;
    for (index, constraint) in source.history.constraints.iter().enumerate() {
        poll_scaled(check, "resident composition history-name cap", index)?;
        if constraint.node_name.is_empty()
            || constraint.node_name.len() > RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1
        {
            return Err(invalid(
                "retained-BaB composition history name exceeds the v1 bound",
            ));
        }
    }
    check("resident composition history-name cap final")?;
    for (index, entry) in source.beta_state.entries.iter().enumerate() {
        poll_scaled(check, "resident composition beta-name cap", index)?;
        if entry.node_name.is_empty() || entry.node_name.len() > RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1
        {
            return Err(invalid(
                "retained-BaB composition beta-entry name exceeds the v1 bound",
            ));
        }
    }
    check("resident composition beta-name cap final")?;
    Ok(())
}

#[derive(Debug, PartialEq)]
pub(in crate::beta_crown::engine::graph) struct ResidentBabDomainOperandsV1 {
    /// Per true ReLU, in fold order: one exact variant tag, signed pre-lower,
    /// signed pre-upper, and four tag-dependent executed coefficient sections
    /// as specified by [`ResidentBabActivationVariantV1`].
    pub activation: Vec<f32>,
    /// Dense full-width signed beta rows in the same fold order.
    pub beta: Vec<f32>,
    /// All segment frontier rows first, then per-ReLU preactivation Abs rows.
    pub abs: Vec<f32>,
    pub box_lower: Vec<f32>,
    pub box_upper: Vec<f32>,
    pub cached_la: Vec<f32>,
    pub history_words: Vec<u32>,
    /// Simultaneous composition peak, including scratch and allocator capacity
    /// excess. It must never be subtracted as retained state.
    pub adapter_host_peak_bytes: usize,
    /// Caller-owned bytes that remain live with this returned value plus the
    /// caller-supplied `resident_bytes_before` baseline.
    pub adapter_host_retained_bytes_after: usize,
}

fn family_len(value: u64, label: &str) -> Result<usize, ResidentBabComposeErrorV1> {
    usize::try_from(value).map_err(|_| {
        invalid(format!(
            "retained-BaB {label} family length does not fit usize"
        ))
    })
}

fn shape_values(shape: &[u64], label: &str) -> Result<usize, ResidentBabComposeErrorV1> {
    let values = shape.iter().try_fold(1u64, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| invalid(format!("retained-BaB {label} shape overflows")))
    })?;
    usize::try_from(values)
        .map_err(|_| invalid(format!("retained-BaB {label} shape does not fit usize")))
}

fn source_width(
    topology: &ResidentBabTopologyV1,
    source_node_id: u32,
) -> Result<usize, ResidentBabComposeErrorV1> {
    if source_node_id == RESIDENT_BAB_NETWORK_INPUT_ID_V1 {
        return shape_values(&topology.input_shape, "network input");
    }
    let node = topology
        .nodes
        .get(
            usize::try_from(source_node_id)
                .map_err(|_| invalid("retained-BaB source node ID does not fit usize"))?,
        )
        .ok_or_else(|| invalid("retained-BaB source node ID is outside topology"))?;
    shape_values(&node.output_shape, "source node")
}

fn exact_abs(lower: f32, upper: f32) -> Result<f32, ResidentBabComposeErrorV1> {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(invalid(
            "retained-BaB Abs source endpoints must be finite and ordered",
        ));
    }
    Ok(lower.abs().max(upper.abs()))
}

/// Compose all six canonical resident-v2 f32 families plus the exact history.
///
/// This is a graph-owner TCB operation: it consumes signed source endpoints,
/// materializes the exact tagged ordinary or whole-row dual-alpha execution
/// coefficients bit-for-bit, and derives Abs itself. The future provider may
/// validate shape/order/nonnegative association but may not claim independent
/// provenance for compact frontier Abs. A fully validated wire remains the
/// semantic topology prerequisite; the initial cap pass only bounds work on a
/// bare safe Rust topology before that sealed artifact exists.
/// [`ResidentBabDomainOperandsV1`] is not standalone authority: no provider,
/// raw call, or phase open may consume it unless the caller binds it to this
/// exact immutable topology after full wire encoding and independent decode.
/// The later owned static draft will seal those objects together.
///
/// The cap baseline must include all simultaneously live borrowed topology,
/// wire/static artifact, history, beta, alpha, endpoints, and other source
/// state. Returned peak/retained accounting is absolute; callers must not add
/// that baseline again.
pub(in crate::beta_crown::engine::graph) fn compose_domain_operands_v1(
    topology: &ResidentBabTopologyV1,
    source: ResidentBabDomainSourceV1<'_>,
    cap: ResidentBabAdapterHostCapV1,
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
) -> Result<ResidentBabDomainOperandsV1, ResidentBabComposeErrorV1> {
    let declared_relu_count = usize::try_from(topology.relu_count)
        .map_err(|_| invalid("retained-BaB declared ReLU count does not fit usize"))?;
    let declared_box_len = family_len(topology.families.box_values, "box")?;
    if source.cached_la_present
        || topology.families.cached_la != 0
        || !source.history.is_pure_relu_at_zero()
        || !source.history.genbab_split_ids.is_empty()
        || source.history.constraints.len() != source.beta_state.entries.len()
        || source.relu_sites.len() != declared_relu_count
        || source.activations.len() != declared_relu_count
        || source.frontiers.len() != topology.segments.len()
        || source.box_lower.len() != declared_box_len
        || source.box_upper.len() != declared_box_len
    {
        return Err(invalid(
            "retained-BaB v1 mode/cardinality precondition is not satisfied",
        ));
    }
    let _minimum_budget =
        ResidentBabHostBudgetV1::begin(cap, size_of::<ResidentBabDomainOperandsV1>())?;
    validate_domain_composition_caps_v1(topology, &source, check)?;
    let mut relu_count = 0usize;
    for (index, layer) in topology.layers.iter().enumerate() {
        poll_scaled(check, "resident topology ReLU count", index)?;
        if layer.kind == ResidentBabLayerKindV1::Relu {
            relu_count = relu_count
                .checked_add(1)
                .ok_or_else(|| invalid("retained-BaB ReLU count overflows usize"))?;
        }
    }
    check("resident topology ReLU count final")?;
    if relu_count != source.relu_sites.len()
        || relu_count != source.activations.len()
        || topology.segments.len() != source.frontiers.len()
    {
        return Err(invalid(
            "retained-BaB topology/source ReLU or frontier cardinality mismatch",
        ));
    }

    let activation_len = family_len(topology.families.activation, "activation")?;
    let beta_len = family_len(topology.families.beta, "beta")?;
    let abs_len = family_len(topology.families.abs, "Abs")?;
    let box_len = family_len(topology.families.box_values, "box")?;

    // Seal every source-driven arena partition before beginning the host
    // budget or allocating the history indexes. The later fill loops may only
    // push the exact counts reserved from these sealed totals.
    let mut sealed_abs_len = 0usize;
    for (segment_index, (segment, frontier)) in
        topology.segments.iter().zip(source.frontiers).enumerate()
    {
        poll_scaled(
            check,
            "resident prebudget frontier association",
            segment_index,
        )?;
        let width = source_width(topology, segment.frontier_node_id)?;
        let frontier_segment_id = usize::try_from(frontier.segment_id).map_err(|_| {
            invalid("retained-BaB prebudget frontier segment ID does not fit usize")
        })?;
        let range_start = usize::try_from(segment.frontier_abs.start)
            .map_err(|_| invalid("retained-BaB prebudget frontier Abs start does not fit usize"))?;
        let range_len = usize::try_from(segment.frontier_abs.len).map_err(|_| {
            invalid("retained-BaB prebudget frontier Abs length does not fit usize")
        })?;
        if frontier_segment_id != segment_index
            || frontier.branch != segment.frontier_branch
            || frontier.source_node_id != segment.frontier_node_id
            || frontier.lower.len() != width
            || frontier.upper.len() != width
            || range_start != sealed_abs_len
            || range_len != width
        {
            return Err(invalid(format!(
                "retained-BaB prebudget frontier source {segment_index} does not match topology"
            )));
        }
        sealed_abs_len = sealed_abs_len
            .checked_add(width)
            .ok_or_else(|| invalid("retained-BaB prebudget frontier Abs length overflows usize"))?;
    }
    check("resident prebudget frontier association final")?;

    let mut source_relu_index = 0usize;
    let mut sealed_activation_len = 0usize;
    let mut sealed_beta_len = 0usize;
    for (layer_index, layer) in topology.layers.iter().enumerate() {
        poll_scaled(check, "resident prebudget ReLU association", layer_index)?;
        if layer.kind != ResidentBabLayerKindV1::Relu {
            continue;
        }
        let site = source
            .relu_sites
            .get(source_relu_index)
            .ok_or_else(|| invalid("retained-BaB prebudget ReLU-site table ended early"))?;
        let row = source
            .activations
            .get(source_relu_index)
            .ok_or_else(|| invalid("retained-BaB prebudget Activation table ended early"))?;
        let node =
            topology
                .nodes
                .get(usize::try_from(layer.node_id).map_err(|_| {
                    invalid("retained-BaB prebudget ReLU node ID does not fit usize")
                })?)
                .ok_or_else(|| invalid("retained-BaB prebudget ReLU node is outside topology"))?;
        let expected_pre = *node
            .inputs
            .first()
            .ok_or_else(|| invalid("retained-BaB prebudget ReLU node has no input"))?;
        let width = usize::try_from(layer.geometry[0])
            .map_err(|_| invalid("retained-BaB prebudget ReLU width does not fit usize"))?;
        let width_u64 = u64::try_from(width)
            .map_err(|_| invalid("retained-BaB prebudget ReLU width does not fit u64"))?;
        let activation_sections =
            ResidentBabActivationSectionsV1::from_row(layer.activation, width_u64)
                .map_err(|_| invalid("retained-BaB prebudget Activation row is noncanonical"))?;
        let activation_start = usize::try_from(layer.activation.start)
            .map_err(|_| invalid("retained-BaB prebudget Activation start does not fit usize"))?;
        let activation_row_len = usize::try_from(layer.activation.len)
            .map_err(|_| invalid("retained-BaB prebudget Activation length does not fit usize"))?;
        let beta_start = usize::try_from(layer.beta.start)
            .map_err(|_| invalid("retained-BaB prebudget beta start does not fit usize"))?;
        let beta_range_len = usize::try_from(layer.beta.len)
            .map_err(|_| invalid("retained-BaB prebudget beta length does not fit usize"))?;
        let node_abs_start = usize::try_from(layer.node_abs.start)
            .map_err(|_| invalid("retained-BaB prebudget node Abs start does not fit usize"))?;
        let node_abs_len = usize::try_from(layer.node_abs.len)
            .map_err(|_| invalid("retained-BaB prebudget node Abs length does not fit usize"))?;
        check("resident prebudget ReLU name association")?;
        if row.topology_node_id != layer.node_id
            || row.preactivation_node_id != expected_pre
            || site.topology_node_id != layer.node_id
            || site.node_name != node.name
            || site.preactivation_width != width
            || source_width(topology, expected_pre)? != width
            || row.pre_lower.len() != width
            || row.pre_upper.len() != width
            || activation_start != sealed_activation_len
            || usize::try_from(activation_sections.tag_index).ok() != Some(sealed_activation_len)
            || beta_start != sealed_beta_len
            || beta_range_len != width
            || node_abs_start != sealed_abs_len
            || node_abs_len != width
        {
            return Err(invalid(format!(
                "retained-BaB prebudget ReLU source {source_relu_index} does not match topology"
            )));
        }
        sealed_activation_len = sealed_activation_len
            .checked_add(activation_row_len)
            .ok_or_else(|| invalid("retained-BaB prebudget Activation length overflows usize"))?;
        sealed_beta_len = sealed_beta_len
            .checked_add(width)
            .ok_or_else(|| invalid("retained-BaB prebudget beta length overflows usize"))?;
        sealed_abs_len = sealed_abs_len
            .checked_add(width)
            .ok_or_else(|| invalid("retained-BaB prebudget node Abs length overflows usize"))?;
        source_relu_index = source_relu_index
            .checked_add(1)
            .ok_or_else(|| invalid("retained-BaB prebudget ReLU index overflows usize"))?;
    }
    check("resident prebudget ReLU association final")?;
    if source_relu_index != source.relu_sites.len()
        || sealed_activation_len != activation_len
        || sealed_beta_len != beta_len
        || sealed_abs_len != abs_len
    {
        return Err(invalid(
            "retained-BaB prebudget Activation/Beta/Abs partition is incomplete",
        ));
    }
    if source.box_lower.len() != box_len || source.box_upper.len() != box_len {
        return Err(invalid(
            "retained-BaB signed input box length does not match topology",
        ));
    }

    let mut prospective_extra = size_of::<ResidentBabDomainOperandsV1>();
    for bytes in [
        checked_elements::<f32>(activation_len)?,
        checked_elements::<f32>(abs_len)?,
        checked_elements::<f32>(box_len)?,
        checked_elements::<f32>(box_len)?,
        history_nominal_bytes(relu_count, source.history.constraints.len(), beta_len)?,
    ] {
        checked_add(&mut prospective_extra, bytes)?;
    }
    let mut budget = ResidentBabHostBudgetV1::begin(cap, prospective_extra)?;

    let history_beta = compose_history_beta_v1(
        source.relu_sites,
        source.history,
        source.beta_state,
        &mut budget,
        check,
    )?;
    if history_beta.beta.len() != beta_len || history_beta.beta_rows.len() != relu_count {
        return Err(invalid(
            "retained-BaB dense beta length does not match topology",
        ));
    }
    let mut beta_row_index = 0usize;
    for (layer_index, layer) in topology.layers.iter().enumerate() {
        poll_scaled(check, "resident beta-row association", layer_index)?;
        if layer.kind != ResidentBabLayerKindV1::Relu {
            continue;
        }
        let row = history_beta
            .beta_rows
            .get(beta_row_index)
            .ok_or_else(|| invalid("retained-BaB beta-row table ended early"))?;
        if u64::try_from(row.start).ok() != Some(layer.beta.start)
            || u64::try_from(row.len).ok() != Some(layer.beta.len)
        {
            return Err(invalid(format!(
                "retained-BaB beta row {beta_row_index} does not match its canonical layer range"
            )));
        }
        beta_row_index = beta_row_index
            .checked_add(1)
            .ok_or_else(|| invalid("retained-BaB beta-row index overflows usize"))?;
    }
    check("resident beta-row association final")?;
    if beta_row_index != history_beta.beta_rows.len() {
        return Err(invalid("retained-BaB beta-row association is incomplete"));
    }

    let mut activation = Vec::new();
    budget.reserve_vec(&mut activation, activation_len, "activation arena")?;
    check("resident Activation reserve")?;
    let mut abs = Vec::new();
    budget.reserve_vec(&mut abs, abs_len, "Abs arena")?;
    check("resident Abs reserve")?;

    for (segment_index, (segment, frontier)) in
        topology.segments.iter().zip(source.frontiers).enumerate()
    {
        poll_scaled(check, "resident frontier provenance", segment_index)?;
        let width = source_width(topology, segment.frontier_node_id)?;
        let frontier_segment_id = usize::try_from(frontier.segment_id)
            .map_err(|_| invalid("retained-BaB frontier segment ID does not fit usize"))?;
        let range_start = usize::try_from(segment.frontier_abs.start)
            .map_err(|_| invalid("retained-BaB frontier Abs start does not fit usize"))?;
        let range_len = usize::try_from(segment.frontier_abs.len)
            .map_err(|_| invalid("retained-BaB frontier Abs length does not fit usize"))?;
        if frontier_segment_id != segment_index
            || frontier.branch != segment.frontier_branch
            || frontier.source_node_id != segment.frontier_node_id
            || frontier.lower.len() != width
            || frontier.upper.len() != width
            || range_start != abs.len()
            || range_len != width
        {
            return Err(invalid(format!(
                "retained-BaB frontier source {segment_index} does not match topology provenance"
            )));
        }
        for (value_index, (&lower, &upper)) in frontier.lower.iter().zip(frontier.upper).enumerate()
        {
            poll_scaled(check, "resident frontier Abs derivation", value_index)?;
            abs.push(exact_abs(lower, upper)?);
        }
    }
    check("resident frontier Abs derivation final")?;

    let mut relu_index = 0usize;
    for (layer_index, layer) in topology.layers.iter().enumerate() {
        poll_scaled(check, "resident Activation topology", layer_index)?;
        if layer.kind != ResidentBabLayerKindV1::Relu {
            continue;
        }
        let site = source
            .relu_sites
            .get(relu_index)
            .ok_or_else(|| invalid("retained-BaB ReLU-site table ended early"))?;
        let row = source
            .activations
            .get(relu_index)
            .ok_or_else(|| invalid("retained-BaB Activation source table ended early"))?;
        let width = usize::try_from(layer.geometry[0])
            .map_err(|_| invalid("retained-BaB ReLU width does not fit usize"))?;
        let node_id = usize::try_from(layer.node_id)
            .map_err(|_| invalid("retained-BaB ReLU node ID does not fit usize"))?;
        let node = topology
            .nodes
            .get(node_id)
            .ok_or_else(|| invalid("retained-BaB ReLU node ID is outside topology"))?;
        let expected_pre = *node
            .inputs
            .first()
            .ok_or_else(|| invalid("retained-BaB ReLU node has no preactivation input"))?;
        let activation_start = usize::try_from(layer.activation.start)
            .map_err(|_| invalid("retained-BaB Activation start does not fit usize"))?;
        let activation_sections = ResidentBabActivationSectionsV1::from_row(
            layer.activation,
            u64::try_from(width)
                .map_err(|_| invalid("retained-BaB Activation width does not fit u64"))?,
        )
        .map_err(|_| invalid("retained-BaB tagged Activation row is noncanonical"))?;
        let node_abs_start = usize::try_from(layer.node_abs.start)
            .map_err(|_| invalid("retained-BaB node Abs start does not fit usize"))?;
        let node_abs_len = usize::try_from(layer.node_abs.len)
            .map_err(|_| invalid("retained-BaB node Abs length does not fit usize"))?;
        check("resident Activation ReLU name association")?;
        if row.topology_node_id != layer.node_id
            || row.preactivation_node_id != expected_pre
            || site.topology_node_id != layer.node_id
            || site.node_name != node.name
            || site.preactivation_width != width
            || source_width(topology, expected_pre)? != width
            || row.pre_lower.len() != width
            || row.pre_upper.len() != width
            || activation_start != activation.len()
            || usize::try_from(activation_sections.tag_index).ok() != Some(activation.len())
            || node_abs_start != abs.len()
            || node_abs_len != width
        {
            return Err(invalid(format!(
                "retained-BaB ReLU source {relu_index} does not match topology layout"
            )));
        }

        check("resident lower-alpha row lookup")?;
        let lower_alpha_neurons = source.alpha_state.neurons().get(site.node_name.as_str());
        check("resident upper-alpha row lookup")?;
        let upper_alpha_neurons = source
            .alpha_state
            .upper_neurons()
            .get(site.node_name.as_str());
        check("resident alpha row lookup final")?;

        // First scan seals the whole-row variant before any row value is
        // written. This catches a lower/upper mismatch even at the last neuron
        // without reinterpreting an already-filled ordinary prefix.
        let mut variant = GpuReluAffineVariant::Ordinary;
        for neuron in 0..width {
            poll_scaled(check, "resident ReLU semantic validation", neuron)?;
            let lower = row.pre_lower[neuron];
            let upper = row.pre_upper[neuron];
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(invalid(format!(
                    "retained-BaB ReLU source {relu_index}/{neuron} has invalid signed endpoints"
                )));
            }
            // The legacy graph-alpha mask is a property of the current signed
            // endpoints, not of BaB history membership. A constrained neuron
            // can remain crossing after propagation and still executes the
            // sparse-state value (or missing-entry heuristic) through the
            // active-alpha lane.
            let alpha_is_active = lower < 0.0 && upper > 0.0;
            let lower_alpha = effective_alpha_v1(lower_alpha_neurons, neuron, lower, upper);
            let upper_alpha = effective_alpha_v1(upper_alpha_neurons, neuron, lower, upper);
            if !lower_alpha.is_finite()
                || !upper_alpha.is_finite()
                || !(0.0..=1.0).contains(&lower_alpha)
                || !(0.0..=1.0).contains(&upper_alpha)
            {
                return Err(invalid(format!(
                    "retained-BaB ReLU source {relu_index}/{neuron} has an invalid effective alpha"
                )));
            }
            if lower < 0.0 && upper > 0.0 && alpha_is_active && lower_alpha != upper_alpha {
                variant = GpuReluAffineVariant::DualAlpha;
            }
        }
        check("resident ReLU variant seal final")?;
        let tag = activation_variant_tag_v1(variant);
        debug_assert!(matches!(
            decode_activation_variant_tag_v1(tag),
            Ok(decoded) if decoded == variant
        ));
        activation.push(tag);
        for (value_index, &value) in row.pre_lower.iter().enumerate() {
            poll_scaled(check, "resident Activation pre-lower copy", value_index)?;
            activation.push(value);
        }
        for (value_index, &value) in row.pre_upper.iter().enumerate() {
            poll_scaled(check, "resident Activation pre-upper copy", value_index)?;
            activation.push(value);
        }
        for section in 0..4 {
            for (value_index, (&lower, &upper)) in
                row.pre_lower.iter().zip(row.pre_upper).enumerate()
            {
                poll_scaled(
                    check,
                    "resident Activation relaxation derivation",
                    value_index,
                )?;
                let alpha_is_active = lower < 0.0 && upper > 0.0;
                let lower_alpha =
                    effective_alpha_v1(lower_alpha_neurons, value_index, lower, upper);
                let upper_alpha =
                    effective_alpha_v1(upper_alpha_neurons, value_index, lower, upper);
                let cell = gpu_relu_affine_cell(
                    lower,
                    upper,
                    lower_alpha,
                    upper_alpha,
                    alpha_is_active,
                    variant,
                );
                let value = *cell
                    .get(section)
                    .ok_or_else(|| invalid("retained-BaB affine section traversal is invalid"))?;
                if !value.is_finite() {
                    return Err(invalid(format!(
                        "retained-BaB ReLU source {relu_index}/{value_index} produced a nonfinite executed affine value"
                    )));
                }
                activation.push(value);
            }
        }
        for (value_index, (&lower, &upper)) in row.pre_lower.iter().zip(row.pre_upper).enumerate() {
            poll_scaled(check, "resident node Abs derivation", value_index)?;
            abs.push(exact_abs(lower, upper)?);
        }
        relu_index = relu_index
            .checked_add(1)
            .ok_or_else(|| invalid("retained-BaB ReLU index overflows"))?;
    }
    check("resident Activation and node Abs final")?;
    if relu_index != relu_count {
        return Err(invalid(
            "retained-BaB topology ReLU traversal ended inconsistently",
        ));
    }
    if activation.len() != activation_len || abs.len() != abs_len {
        return Err(invalid(
            "retained-BaB composed Activation/Abs lengths do not match topology",
        ));
    }

    if source.box_lower.len() != box_len || source.box_upper.len() != box_len {
        return Err(invalid(
            "retained-BaB signed input box length does not match topology",
        ));
    }
    let mut box_lower = Vec::new();
    let mut box_upper = Vec::new();
    budget.reserve_vec(&mut box_lower, box_len, "box-lower arena")?;
    check("resident box-lower reserve")?;
    budget.reserve_vec(&mut box_upper, box_len, "box-upper arena")?;
    check("resident box-upper reserve")?;
    for (value_index, (&lower, &upper)) in source.box_lower.iter().zip(source.box_upper).enumerate()
    {
        poll_scaled(check, "resident input-box copy", value_index)?;
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(invalid(
                "retained-BaB input-box endpoints must be finite and ordered",
            ));
        }
        box_lower.push(lower);
        box_upper.push(upper);
    }
    check("resident input-box copy final")?;

    let cached_la = Vec::new();
    let mut retained_extra = size_of::<ResidentBabDomainOperandsV1>();
    for bytes in [
        checked_elements::<f32>(activation.capacity())?,
        checked_elements::<f32>(history_beta.beta.capacity())?,
        checked_elements::<f32>(abs.capacity())?,
        checked_elements::<f32>(box_lower.capacity())?,
        checked_elements::<f32>(box_upper.capacity())?,
        checked_elements::<f32>(cached_la.capacity())?,
        checked_elements::<u32>(history_beta.history_words.capacity())?,
    ] {
        checked_add(&mut retained_extra, bytes)?;
    }
    let adapter_host_retained_bytes_after =
        cap.resident_bytes_before
            .checked_add(retained_extra)
            .ok_or_else(|| invalid("retained-BaB retained host bytes overflow usize"))?;
    let adapter_host_peak_bytes = budget.peak_bytes();
    if adapter_host_retained_bytes_after > adapter_host_peak_bytes {
        return Err(invalid(
            "retained-BaB retained host bytes exceed the admitted composition peak",
        ));
    }
    check("resident operand composition final")?;

    Ok(ResidentBabDomainOperandsV1 {
        activation,
        beta: history_beta.beta,
        abs,
        box_lower,
        box_upper,
        cached_la,
        history_words: history_beta.history_words,
        adapter_host_peak_bytes,
        adapter_host_retained_bytes_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
    use crate::beta_crown::state::GraphBetaState;
    use crate::layers::{Layer, ReLULayer};
    use crate::network::{extract_relu_gpu_layer_with_alpha, try_extract_single_gpu_layer};
    use crate::resident_bab_wire::v1::{
        ResidentBabFamilyLengthsV1, ResidentBabLayerBranchV1, ResidentBabLayerV1,
        ResidentBabNodeKindV1, ResidentBabNodeV1, ResidentBabSegmentKindV1, ResidentBabSegmentV1,
        ResidentBabWireRangeV1,
    };
    use ndarray::{ArrayD, IxDyn};
    use ny_core::GpuCrownLayer;
    use ny_tensor::BoundedTensor;

    fn range(start: u64, len: u64) -> ResidentBabWireRangeV1 {
        ResidentBabWireRangeV1 { start, len }
    }

    fn topology() -> ResidentBabTopologyV1 {
        ResidentBabTopologyV1 {
            input_shape: vec![2],
            output_shape: vec![2],
            output_node_id: 2,
            nodes: vec![
                ResidentBabNodeV1 {
                    id: 0,
                    name: "linear_in".to_string(),
                    kind: ResidentBabNodeKindV1::Linear,
                    inputs: vec![RESIDENT_BAB_NETWORK_INPUT_ID_V1],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 1,
                    name: "relu".to_string(),
                    kind: ResidentBabNodeKindV1::Relu,
                    inputs: vec![0],
                    relu_preactivation_node_id: Some(0),
                    output_shape: vec![2],
                    output_values: 2,
                },
                ResidentBabNodeV1 {
                    id: 2,
                    name: "linear_out".to_string(),
                    kind: ResidentBabNodeKindV1::Linear,
                    inputs: vec![1],
                    relu_preactivation_node_id: None,
                    output_shape: vec![2],
                    output_values: 2,
                },
            ],
            segments: vec![ResidentBabSegmentV1 {
                id: 0,
                kind: ResidentBabSegmentKindV1::Chain,
                first_layer: 0,
                main_layer_count: 3,
                projection_layer_count: 0,
                frontier_node_id: RESIDENT_BAB_NETWORK_INPUT_ID_V1,
                merge_node_id: None,
                frontier_branch: ResidentBabFrontierBranchV1::Main,
                frontier_abs: range(0, 2),
            }],
            layers: vec![
                ResidentBabLayerV1 {
                    ordinal: 0,
                    kind: ResidentBabLayerKindV1::Linear,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 2,
                    parameters: range(0, 6),
                    certified_errors: range(0, 2),
                    activation: range(0, 0),
                    beta: range(0, 0),
                    node_abs: range(2, 0),
                    geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 1,
                    kind: ResidentBabLayerKindV1::Relu,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 1,
                    parameters: range(6, 0),
                    certified_errors: range(2, 0),
                    activation: range(0, 13),
                    beta: range(0, 2),
                    node_abs: range(2, 2),
                    geometry: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                ResidentBabLayerV1 {
                    ordinal: 2,
                    kind: ResidentBabLayerKindV1::Linear,
                    branch: ResidentBabLayerBranchV1::Main,
                    segment_id: 0,
                    node_id: 0,
                    parameters: range(6, 6),
                    certified_errors: range(2, 2),
                    activation: range(13, 0),
                    beta: range(2, 0),
                    node_abs: range(4, 0),
                    geometry: [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
            ],
            relu_count: 1,
            families: ResidentBabFamilyLengthsV1 {
                parameters: 12,
                certified_errors: 4,
                activation: 13,
                beta: 2,
                abs: 4,
                box_values: 2,
                cached_la: 0,
                topology_metadata: 0,
            },
        }
    }

    struct Sources {
        sites: Vec<ResidentBabReluSiteV1>,
        history: GraphSplitHistory,
        beta: GraphBetaState,
        alpha: GraphDomainAlphaState,
        pre_lower: Vec<f32>,
        pre_upper: Vec<f32>,
        frontier_lower: Vec<f32>,
        frontier_upper: Vec<f32>,
        box_lower: Vec<f32>,
        box_upper: Vec<f32>,
    }

    fn sources() -> Sources {
        let pre_lower = vec![-2.0, 1.0];
        let pre_upper = vec![3.0, 4.0];
        let mut history = GraphSplitHistory::new();
        history
            .add_constraint(GraphNeuronConstraint::new("relu".to_string(), 0, false, 7.0).unwrap());
        let beta = GraphBetaState::from_history_with_init(&history, 0.5).unwrap();
        Sources {
            sites: vec![ResidentBabReluSiteV1 {
                topology_node_id: 1,
                node_name: "relu".to_string(),
                preactivation_width: 2,
            }],
            history,
            beta,
            alpha: GraphDomainAlphaState::empty(),
            pre_lower,
            pre_upper,
            frontier_lower: vec![-4.0, -1.0],
            frontier_upper: vec![2.0, 5.0],
            box_lower: vec![-1.0, -2.0],
            box_upper: vec![1.0, 2.0],
        }
    }

    fn compose_with_topology(
        topology: &ResidentBabTopologyV1,
        sources: &Sources,
        limit_bytes: usize,
        resident_bytes_before: usize,
        check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
    ) -> Result<ResidentBabDomainOperandsV1, ResidentBabComposeErrorV1> {
        let activations = [ResidentBabActivationSourceV1 {
            topology_node_id: 1,
            preactivation_node_id: 0,
            pre_lower: &sources.pre_lower,
            pre_upper: &sources.pre_upper,
        }];
        let frontiers = [ResidentBabFrontierSourceV1 {
            segment_id: 0,
            branch: ResidentBabFrontierBranchV1::Main,
            source_node_id: RESIDENT_BAB_NETWORK_INPUT_ID_V1,
            lower: &sources.frontier_lower,
            upper: &sources.frontier_upper,
        }];
        compose_domain_operands_v1(
            topology,
            ResidentBabDomainSourceV1 {
                relu_sites: &sources.sites,
                activations: &activations,
                frontiers: &frontiers,
                history: &sources.history,
                beta_state: &sources.beta,
                alpha_state: &sources.alpha,
                box_lower: &sources.box_lower,
                box_upper: &sources.box_upper,
                cached_la_present: false,
            },
            ResidentBabAdapterHostCapV1 {
                limit_bytes,
                resident_bytes_before,
            },
            check,
        )
    }

    fn compose(
        sources: &Sources,
        limit_bytes: usize,
    ) -> Result<ResidentBabDomainOperandsV1, ResidentBabComposeErrorV1> {
        let topology = topology();
        let mut check = |_| Ok(());
        compose_with_topology(&topology, sources, limit_bytes, 64, &mut check)
    }

    #[test]
    fn composes_six_families_with_signed_sources_and_derived_abs() {
        let sources = sources();
        let operands = compose(&sources, 1 << 20).unwrap();
        assert_eq!(operands.activation[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(&operands.activation[1..3], &[-2.0, 1.0]);
        assert_eq!(&operands.activation[3..5], &[3.0, 4.0]);
        assert_eq!(operands.activation.len(), 13);
        assert_eq!(operands.beta.len(), 2);
        assert_eq!(operands.beta[0], -0.5);
        assert_eq!(operands.abs, vec![4.0, 5.0, 3.0, 4.0]);
        assert_eq!(operands.box_lower, vec![-1.0, -2.0]);
        assert_eq!(operands.box_upper, vec![1.0, 2.0]);
        assert!(operands.cached_la.is_empty());
        assert_eq!(operands.history_words.len(), 4);
        assert!(operands.adapter_host_peak_bytes > 64);
        assert!(operands.adapter_host_retained_bytes_after > 64);
        assert!(operands.adapter_host_peak_bytes >= operands.adapter_host_retained_bytes_after);

        let preactivation = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), sources.pre_lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), sources.pre_upper.clone()).unwrap(),
        )
        .unwrap();
        let mut legacy = Vec::new();
        try_extract_single_gpu_layer(&Layer::ReLU(ReLULayer), &preactivation, &mut legacy).unwrap();
        let GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            ..
        } = legacy.pop().unwrap()
        else {
            panic!("ordinary ReLU extraction must produce Activation");
        };
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&operands.activation[5..7]), bits(&lower_slope));
        assert_eq!(bits(&operands.activation[7..9]), bits(&upper_slope));
        assert_eq!(bits(&operands.activation[9..11]), bits(&lower_intercept));
        assert_eq!(bits(&operands.activation[11..13]), bits(&upper_intercept));

        // The first crossing neuron is already constrained in `history`, but
        // its sparse alpha entry is absent. Legacy execution still derives an
        // active endpoint mask and applies the missing-entry heuristic.
        let legacy_alpha = extract_relu_gpu_layer_with_alpha(
            &sources.pre_lower,
            &sources.pre_upper,
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[true, false],
        );
        let GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            ..
        } = legacy_alpha
        else {
            panic!("missing-alpha heuristic must keep the ordinary Activation ABI");
        };
        assert_eq!(bits(&operands.activation[5..7]), bits(&lower_slope));
        assert_eq!(bits(&operands.activation[7..9]), bits(&upper_slope));
        assert_eq!(bits(&operands.activation[9..11]), bits(&lower_intercept));
        assert_eq!(bits(&operands.activation[11..13]), bits(&upper_intercept));
    }

    #[test]
    fn activation_tags_are_exact_and_late_crossing_mismatch_seals_dual_row() {
        assert_eq!(
            decode_activation_variant_tag_v1(0.0).unwrap(),
            GpuReluAffineVariant::Ordinary
        );
        assert_eq!(
            decode_activation_variant_tag_v1(1.0).unwrap(),
            GpuReluAffineVariant::DualAlpha
        );
        assert!(decode_activation_variant_tag_v1(-0.0).is_err());
        assert!(decode_activation_variant_tag_v1(2.0).is_err());

        let mut sources = sources();
        sources.pre_lower = vec![1.0, -1.0];
        sources.pre_upper = vec![2.0, 2.0];
        sources
            .alpha
            .insert("relu".to_string(), 1, AlphaNeuronState::new(0.25));
        sources
            .alpha
            .upper_neurons_mut()
            .get_mut("relu")
            .and_then(|neurons| neurons.get_mut(&1))
            .expect("insert mirrors upper alpha")
            .set_alpha(0.75);
        let operands = compose(&sources, 1 << 20).unwrap();
        assert_eq!(operands.activation[0].to_bits(), 1.0f32.to_bits());
        // Earlier stable-positive cell must use the dual section-2 value 1,
        // not the ordinary lower-intercept value 0.
        assert_eq!(operands.activation[9].to_bits(), 1.0f32.to_bits());

        let legacy = extract_relu_gpu_layer_with_alpha(
            &sources.pre_lower,
            &sources.pre_upper,
            &[1.0, 0.25],
            &[1.0, 0.75],
            &[false, true],
        );
        let GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            ..
        } = legacy
        else {
            panic!("late alpha mismatch must select the whole-row dual descriptor");
        };
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&operands.activation[5..7]), bits(&lower_pos_slope));
        assert_eq!(bits(&operands.activation[7..9]), bits(&cross_slope));
        assert_eq!(bits(&operands.activation[9..11]), bits(&upper_neg_slope));
        assert_eq!(bits(&operands.activation[11..13]), bits(&cross_intercept));
    }

    #[test]
    fn refuses_relaxation_or_frontier_provenance_drift() {
        let mut bad_relaxation = sources();
        bad_relaxation.pre_lower[0] = f32::NAN;
        assert!(matches!(
            compose(&bad_relaxation, 1 << 20),
            Err(ResidentBabComposeErrorV1::Invalid(_))
        ));

        let mut bad_frontier = sources();
        bad_frontier.frontier_lower.pop();
        assert!(matches!(
            compose(&bad_frontier, 1 << 20),
            Err(ResidentBabComposeErrorV1::Invalid(_))
        ));
    }

    #[test]
    fn prebudget_layout_seal_refuses_activation_or_abs_drift_before_any_reserve() {
        let mut malformed = Vec::new();

        let mut activation_row = topology();
        activation_row.layers[1].activation.len -= 1;
        activation_row.families.activation -= 1;
        malformed.push(activation_row);

        let mut activation_partition = topology();
        activation_partition.layers[1].activation.start = 1;
        malformed.push(activation_partition);

        let mut frontier_abs = topology();
        frontier_abs.segments[0].frontier_abs.len -= 1;
        frontier_abs.families.abs -= 1;
        malformed.push(frontier_abs);

        let mut node_abs = topology();
        node_abs.layers[1].node_abs.start += 1;
        malformed.push(node_abs);

        let mut abs_total = topology();
        abs_total.families.abs += 1;
        malformed.push(abs_total);

        let sources = sources();
        for malformed_topology in malformed {
            let mut reserve_callbacks = 0usize;
            let mut check = |label: &'static str| {
                if label.contains("reserve") {
                    reserve_callbacks += 1;
                }
                Ok(())
            };
            assert!(matches!(
                compose_with_topology(&malformed_topology, &sources, 1 << 20, 64, &mut check,),
                Err(ResidentBabComposeErrorV1::Invalid(_))
            ));
            assert_eq!(reserve_callbacks, 0);
        }
    }

    #[test]
    fn cap_validation_rejects_long_names_and_family_overflow_before_reserve() {
        let mut cases = Vec::new();

        let mut site_name = sources();
        site_name.sites[0].node_name = "s".repeat(RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1 + 1);
        cases.push((topology(), site_name));

        let mut history_name = sources();
        history_name.history.constraints[0].node_name =
            "h".repeat(RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1 + 1);
        cases.push((topology(), history_name));

        let mut beta_name = sources();
        beta_name.beta.entries[0].node_name = "b".repeat(RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1 + 1);
        cases.push((topology(), beta_name));

        let mut topology_name = topology();
        topology_name.nodes[1].name = "n".repeat(RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1 + 1);
        cases.push((topology_name, sources()));

        let mut family_overflow = topology();
        family_overflow.families.activation =
            u64::try_from(ny_core::GPU_BAB_BOUND_MAX_ARENA_VALUES).unwrap() + 1;
        cases.push((family_overflow, sources()));

        for (topology, sources) in cases {
            let mut reserve_callbacks = 0usize;
            let mut check = |label: &'static str| {
                if label.contains("reserve") {
                    reserve_callbacks += 1;
                }
                Ok(())
            };
            assert!(matches!(
                compose_with_topology(&topology, &sources, 1 << 20, 64, &mut check),
                Err(ResidentBabComposeErrorV1::Invalid(_))
            ));
            assert_eq!(reserve_callbacks, 0);
        }
    }

    #[test]
    fn accounting_is_absolute_and_alpha_name_maps_are_looked_up_once_per_relu() {
        let mut topology = topology();
        let long_name = "r".repeat(RESIDENT_BAB_MAX_NODE_NAME_BYTES_V1);
        topology.nodes[1].name.clone_from(&long_name);
        let mut sources = sources();
        sources.sites[0].node_name.clone_from(&long_name);
        sources.history.constraints[0]
            .node_name
            .clone_from(&long_name);
        sources.beta.entries[0].node_name.clone_from(&long_name);

        let mut lower_lookups = 0usize;
        let mut upper_lookups = 0usize;
        let mut check = |label| {
            if label == "resident lower-alpha row lookup" {
                lower_lookups += 1;
            } else if label == "resident upper-alpha row lookup" {
                upper_lookups += 1;
            }
            Ok(())
        };
        let baseline = 64usize;
        let first =
            compose_with_topology(&topology, &sources, 1 << 20, baseline, &mut check).unwrap();
        assert_eq!(lower_lookups, 1);
        assert_eq!(upper_lookups, 1);

        let shift = 4_096usize;
        let mut check = |_| Ok(());
        let shifted =
            compose_with_topology(&topology, &sources, 1 << 20, baseline + shift, &mut check)
                .unwrap();
        assert_eq!(
            shifted.adapter_host_peak_bytes,
            first.adapter_host_peak_bytes + shift
        );
        assert_eq!(
            shifted.adapter_host_retained_bytes_after,
            first.adapter_host_retained_bytes_after + shift
        );
    }

    #[test]
    fn prospective_adapter_host_cap_refuses_before_provider_use() {
        let sources = sources();
        let topology = topology();
        let baseline = 64usize;
        let minimum = baseline + size_of::<ResidentBabDomainOperandsV1>();
        let mut callbacks = 0usize;
        let mut check = |_| {
            callbacks += 1;
            Ok(())
        };
        assert!(matches!(
            compose_with_topology(
                &topology,
                &sources,
                minimum - 1,
                baseline,
                &mut check,
            ),
            Err(ResidentBabComposeErrorV1::Capacity {
                limit_bytes,
                ..
            }) if limit_bytes == minimum - 1
        ));
        assert_eq!(callbacks, 0);
    }

    #[test]
    fn refuses_cached_la_or_nonfinite_signed_source() {
        let mut sources = sources();
        sources.box_upper[0] = f32::INFINITY;
        assert!(matches!(
            compose(&sources, 1 << 20),
            Err(ResidentBabComposeErrorV1::Invalid(_))
        ));
    }
}
