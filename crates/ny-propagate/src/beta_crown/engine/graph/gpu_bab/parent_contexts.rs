// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::time::Instant;

use ndarray::Axis;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::PickedDomains;
use crate::beta_crown::engine::graph::domain_batch::{
    DenseSpecBatchRequest, GraphDomainBatchExecutor,
};
use crate::beta_crown::engine::graph::input_split::metrics::DenseSpecReboundTiming;
use crate::beta_crown::engine::graph::input_split::shared::compute_crown_or_ibp_bounds_with_node_bounds;
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::LinearBounds;
use crate::GraphNetwork;

use super::init::{restore_input_split_linear_bounds, InputSplitBootstrap};

pub(super) struct ParentDomainContext {
    pub(super) input_bounds: BoundedTensor,
    pub(super) lower_bound: f32,
    pub(super) upper_bound: f32,
    pub(super) linear_bounds: Option<LinearBounds>,
}

pub(super) struct ParentContextBuildOutcome {
    pub(super) contexts: Vec<ParentDomainContext>,
    pub(super) rebound_timing: DenseSpecReboundTiming,
    pub(super) deferred_count: usize,
    pub(super) batched_count: usize,
    pub(super) override_count: usize,
}

fn materialize_picked_input_bounds(
    picked: &PickedDomains,
    picked_idx: usize,
) -> Result<BoundedTensor> {
    let input_batch = picked.input_lowers.shape().first().copied().unwrap_or(0);
    if picked_idx >= input_batch {
        return Err(NyError::InvalidSpec(format!(
            "GPU BaB input split: picked_idx {} >= input batch size {}",
            picked_idx, input_batch
        )));
    }
    let input_lower = picked
        .input_lowers
        .index_axis(Axis(0), picked_idx)
        .to_owned()
        .into_dyn();
    let input_upper = picked
        .input_uppers
        .index_axis(Axis(0), picked_idx)
        .to_owned()
        .into_dyn();
    BoundedTensor::new(input_lower, input_upper)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_parent_contexts(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    picked: &PickedDomains,
    picked_indices: &[usize],
    bootstrap: &InputSplitBootstrap,
    engine: Option<&dyn GemmEngine>,
) -> Result<ParentContextBuildOutcome> {
    let mut parent_inputs = Vec::with_capacity(picked_indices.len());
    for &picked_idx in picked_indices {
        parent_inputs.push(materialize_picked_input_bounds(picked, picked_idx)?);
    }

    let mut deferred_batched_positions = Vec::new();
    let mut deferred_input_refs = Vec::new();
    let mut deferred_override_positions = Vec::new();
    for (position, (&picked_idx, input_bounds)) in
        picked_indices.iter().zip(parent_inputs.iter()).enumerate()
    {
        let meta = picked.metadata.get(picked_idx).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "GPU BaB input split: metadata missing for picked_idx {}",
                picked_idx
            ))
        })?;
        if meta.needs_bounding() {
            if meta.node_bounds_override().is_some() {
                deferred_override_positions.push(position);
            } else {
                deferred_batched_positions.push(position);
                deferred_input_refs.push(input_bounds);
            }
        }
    }

    let deferred_count = deferred_batched_positions.len() + deferred_override_positions.len();
    let override_count = deferred_override_positions.len();
    let rebound_start = Instant::now();
    let mut batched_timing = None;
    let deferred_bounds = if deferred_input_refs.is_empty() {
        None
    } else {
        let batched_bounds =
            GraphDomainBatchExecutor::execute_dense_specs(DenseSpecBatchRequest {
                graph,
                input_bounds_batch: &deferred_input_refs,
                spec_matrix: &bootstrap.spec_matrix,
                engine,
                alpha_node_bounds: bootstrap.fixed_node_bounds.as_ref(),
                alpha_state: bootstrap.root_alpha_state.as_ref(),
                mul_binary_alphas: bootstrap.mul_binary_alphas.as_ref(),
                deadline: bootstrap.deadline,
                crown_backward_layers: verifier.config.crown_backward_layers,
                ibp_enhancement: verifier.config.input_split_ibp_enhancement,
                stacked_rebound: verifier.config.input_split_stacked_rebound,
            })?;
        batched_timing = Some(batched_bounds.rebound_timing.clone());

        let n = batched_bounds.bounds.len();
        let mut lower_bounds = Vec::with_capacity(n);
        let mut upper_bounds = Vec::with_capacity(n);
        for bounds in &batched_bounds.bounds {
            lower_bounds.push(bounds.lower_scalar());
            upper_bounds.push(bounds.upper_scalar());
        }

        Some((lower_bounds, upper_bounds, batched_bounds.linear_bounds))
    };
    let rebound_timing = match batched_timing {
        Some(timing) => timing.with_total_elapsed(
            deferred_count,
            bootstrap.spec_matrix.nrows(),
            rebound_start.elapsed().as_secs_f64(),
        ),
        None if deferred_count > 0 => DenseSpecReboundTiming::override_only(
            deferred_count,
            bootstrap.spec_matrix.nrows(),
            rebound_start.elapsed().as_secs_f64(),
        ),
        None => DenseSpecReboundTiming::no_deferred_domains(bootstrap.spec_matrix.nrows()),
    };

    let mut deferred_results: Vec<Option<(f32, f32, Option<LinearBounds>)>> =
        vec![None; parent_inputs.len()];
    if let Some((lower_bounds, upper_bounds, linear_bounds)) = deferred_bounds {
        for (deferred_idx, position) in deferred_batched_positions.iter().copied().enumerate() {
            deferred_results[position] = Some((
                lower_bounds[deferred_idx],
                upper_bounds[deferred_idx],
                linear_bounds[deferred_idx].clone(),
            ));
        }
    }

    for position in deferred_override_positions {
        let picked_idx = picked_indices[position];
        let meta = picked.metadata.get(picked_idx).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "GPU BaB input split: metadata missing for picked_idx {}",
                picked_idx
            ))
        })?;
        let (bounds, linear_bounds) = compute_crown_or_ibp_bounds_with_node_bounds(
            graph,
            &parent_inputs[position],
            &bootstrap.spec_matrix,
            engine,
            bootstrap.fixed_node_bounds.as_ref(),
            meta.node_bounds_override(),
            bootstrap.root_alpha_state.as_ref(),
            bootstrap.mul_binary_alphas.as_ref(),
            bootstrap.deadline,
            verifier.config.crown_backward_layers,
            verifier.config.input_split_ibp_enhancement,
        )?;
        deferred_results[position] =
            Some((bounds.lower_scalar(), bounds.upper_scalar(), linear_bounds));
    }

    let mut contexts = Vec::with_capacity(parent_inputs.len());
    for (position, (input_bounds, picked_idx)) in parent_inputs
        .into_iter()
        .zip(picked_indices.iter().copied())
        .enumerate()
    {
        let meta = picked.metadata.get(picked_idx).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "GPU BaB input split: metadata missing for picked_idx {}",
                picked_idx
            ))
        })?;

        if let Some((lower_bound, upper_bound, linear_bounds)) = deferred_results[position].take() {
            contexts.push(ParentDomainContext {
                input_bounds,
                lower_bound,
                upper_bound,
                linear_bounds,
            });
        } else {
            contexts.push(ParentDomainContext {
                input_bounds,
                lower_bound: meta.lower_bound(),
                upper_bound: meta.upper_bound(),
                linear_bounds: restore_input_split_linear_bounds(meta),
            });
        }
    }

    Ok(ParentContextBuildOutcome {
        contexts,
        rebound_timing,
        deferred_count,
        batched_count: deferred_batched_positions.len(),
        override_count,
    })
}
