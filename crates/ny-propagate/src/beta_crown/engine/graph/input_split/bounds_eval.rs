// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared CROWN/IBP bound evaluation helpers for graph input-split.

use std::collections::HashMap;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::network::merge_reference_bound_maps;
use crate::network::SpecCrownRequest;
use crate::GraphNetwork;

pub(crate) fn graph_output_bounds_are_finite(bounds: &BoundedTensor) -> bool {
    bounds
        .lower()
        .iter()
        .chain(bounds.upper().iter())
        .all(|value| value.is_finite())
}

pub(crate) fn graph_ibp_prescreen_error_should_skip(err: &NyError) -> bool {
    match err {
        NyError::ShapeMismatch { .. }
        | NyError::UnsupportedOp(_)
        | NyError::UnsupportedConfiguration(_)
        | NyError::NumericalInstability(_)
        | NyError::DeadlineExceeded(_)
        | NyError::InfeasibleDomain(_) => true,
        NyError::InvalidSpec(message) => message.contains("empty after clamping"),
        NyError::LayerError { source, .. } => graph_ibp_prescreen_error_should_skip(source),
        _ => false,
    }
}

pub(crate) fn graph_crown_error_should_fallback(err: &NyError) -> bool {
    matches!(
        err,
        NyError::UnsupportedOp(_)
            | NyError::UnsupportedConfiguration(_)
            | NyError::NumericalInstability(_)
            | NyError::ShapeMismatch { .. }
            | NyError::DeadlineExceeded(_)
    )
}

pub(crate) fn graph_spec_ibp_fallback(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
) -> Result<(BoundedTensor, Option<LinearBounds>)> {
    let output_node_name = if graph.output_node.is_empty() {
        graph
            .exec_order()?
            .last()
            .cloned()
            .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
    } else {
        graph.output_node.clone()
    };

    let computed_node_bounds;
    let node_bounds = if let Some(root_bounds) = root_node_bounds {
        root_bounds
    } else {
        computed_node_bounds = graph.collect_node_bounds_with_engine(input, engine)?;
        &computed_node_bounds
    };

    let bounds = graph.propagate_crown_with_specs_fallback_ibp(
        input,
        spec_matrix,
        node_bounds,
        &output_node_name,
    )?;
    Ok((bounds, None))
}

pub(crate) fn try_graph_spec_ibp_prescreen_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    root_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    context: &str,
) -> Result<Option<BoundedTensor>> {
    match graph_spec_ibp_fallback(graph, input, spec_matrix, engine, root_node_bounds) {
        Ok((bounds, _)) if graph_output_bounds_are_finite(&bounds) => Ok(Some(bounds)),
        Ok((_bounds, _)) => {
            tracing::debug!(
                "{context}: enhancement-only IBP prescreen produced non-finite bounds; skipping prescreen"
            );
            Ok(None)
        }
        Err(err) if graph_ibp_prescreen_error_should_skip(&err) => {
            tracing::debug!(
                "{context}: enhancement-only IBP prescreen failed; skipping prescreen: {err}"
            );
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

pub(crate) fn graph_spec_ibp_root_screen_with_deadline(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<(BoundedTensor, Option<LinearBounds>)> {
    let node_bounds =
        graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline)?;
    graph_spec_ibp_fallback(graph, input, spec_matrix, engine, Some(&node_bounds))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn graph_spec_crown_with_mul_binary_and_truncation(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    reference_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
) -> Result<(BoundedTensor, Option<LinearBounds>)> {
    let (bounds, linear) = SpecCrownRequest::new(graph, input, spec_matrix, engine)
        .node_bounds_opt(node_bounds)
        .reference_bounds_opt(reference_node_bounds)
        .alpha_state_opt(alpha_state)
        .mul_binary_alphas_opt(mul_binary_alphas)
        .deadline_opt(deadline)
        .truncate_after_opt(crown_backward_layers)
        .run_with_linear()?;
    // #cgan-fwdlin-ref (DARK, `NY_FORWARD_LINEAR_CONV_TRANSPOSE_REF=1`):
    // PER-DOMAIN forward-linear C-margin intersect. This is the input-split
    // per-child evaluation entry, and it wants the input LinearBounds — which
    // the spec-propagation root-candidate fast paths deliberately skip (their
    // early return carries `None` linear). So the certified fwdlin margin
    // never reached per-domain evaluations. Here we run the CPU loop FIRST
    // (keeping the linear map) and then INTERSECT its concrete bounds with the
    // certified forward-linear C-margin composition, which recomputes per
    // subdomain box in ~ms on latent-input generators and tightens with split
    // depth. Sound: both operands are certified enclosures of the same spec
    // values, element-wise intersection keeps the tighter side; any fwdlin
    // refusal keeps the CPU result untouched. Gate-off is byte-identical.
    let bounds = if GraphNetwork::forward_linear_conv_transpose_reference_enabled()
        && graph.has_conv2d_layers()
        && graph.has_conv_transpose2d_layers()
    {
        match graph.forward_linear_spec_margin_bounds(input, spec_matrix, engine, deadline) {
            Ok(fw) if fw.shape() == bounds.shape() => {
                // Probe telemetry (#cgan-fwdlin-ref diagnostics): sampled so a
                // deep BaB run stays readable — first 20 calls, then every
                // 500th. Answers "how close does the per-domain C-margin get
                // with split depth".
                if std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_PROBE")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    use std::sync::atomic::{AtomicUsize, Ordering};
                    static CMARGIN_CALLS: AtomicUsize = AtomicUsize::new(0);
                    let n = CMARGIN_CALLS.fetch_add(1, Ordering::Relaxed);
                    if n < 20 || n.is_multiple_of(500) {
                        let cpu_worst =
                            bounds.lower().iter().copied().fold(f32::INFINITY, f32::min);
                        let fw_worst = fw.lower().iter().copied().fold(f32::INFINITY, f32::min);
                        let in_w = input
                            .lower()
                            .iter()
                            .zip(input.upper().iter())
                            .map(|(&l, &u)| u - l)
                            .fold(0.0_f32, f32::max);
                        eprintln!(
                            "[fwdlin-cmargin] call={n} cpu_worst={cpu_worst:.6} fw_worst={fw_worst:.6} in_w={in_w:.6}"
                        );
                    }
                }
                bounds
                    .intersection_per_element(&fw)
                    .map(|(t, _)| t)
                    .unwrap_or(bounds)
            }
            _ => bounds,
        }
    } else {
        bounds
    };
    Ok((bounds, linear))
}

pub(crate) fn build_input_split_reference_bounds(
    alpha_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    child_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    ibp_bounds: Option<&HashMap<String, BoundedTensor>>,
) -> Result<Option<HashMap<String, BoundedTensor>>> {
    let inherited_reference = merge_reference_bound_maps(alpha_node_bounds, child_node_bounds)?;
    merge_reference_bound_maps(inherited_reference.as_ref(), ibp_bounds)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_plain_crown_or_ibp_bounds_with_node_bounds(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    fixed_node_bounds: Option<&HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    deadline: Option<Instant>,
    crown_backward_layers: Option<usize>,
) -> Result<(BoundedTensor, Option<LinearBounds>)> {
    let crown_result = graph_spec_crown_with_mul_binary_and_truncation(
        graph,
        input_bounds,
        spec_matrix,
        engine,
        fixed_node_bounds,
        None,
        alpha_state,
        mul_binary_alphas,
        deadline,
        crown_backward_layers,
    );
    match crown_result {
        Ok(result) if graph_output_bounds_are_finite(&result.0) => Ok(result),
        Ok((_bounds, _linear)) => {
            tracing::debug!(
                "spec-guided CROWN produced non-finite bounds on sub-domain, falling back to IBP"
            );
            graph_spec_ibp_fallback(graph, input_bounds, spec_matrix, engine, fixed_node_bounds)
        }
        Err(err) if graph_crown_error_should_fallback(&err) => {
            tracing::debug!(
                "spec-guided CROWN failed on sub-domain with {}, falling back to IBP",
                err
            );
            graph_spec_ibp_fallback(graph, input_bounds, spec_matrix, engine, fixed_node_bounds)
        }
        Err(err) => Err(err),
    }
}

#[cfg(test)]
mod tests {
    use super::graph_ibp_prescreen_error_should_skip;
    use ny_core::NyError;

    #[test]
    fn test_graph_ibp_prescreen_error_should_skip_shape_mismatch_4372() {
        assert!(graph_ibp_prescreen_error_should_skip(
            &NyError::shape_mismatch(vec![192], vec![96],)
        ));
    }

    #[test]
    fn test_graph_ibp_prescreen_error_should_skip_nonfinite_bounds_4372() {
        assert!(graph_ibp_prescreen_error_should_skip(
            &NyError::NumericalInstability(
                "BaB domain bounds are non-finite: lower=-inf, upper=inf".to_string()
            )
        ));
    }

    #[test]
    fn test_graph_ibp_prescreen_error_should_skip_empty_clamped_slice_4372() {
        assert!(graph_ibp_prescreen_error_should_skip(
            &NyError::InvalidSpec(
                "Slice range [1:1) empty after clamping to axis 0 size 1".to_string()
            )
        ));
    }

    #[test]
    fn test_graph_ibp_prescreen_error_does_not_skip_unrelated_invalid_spec_4372() {
        assert!(!graph_ibp_prescreen_error_should_skip(
            &NyError::InvalidSpec(
                "batched_ibp_prescreen spec width 2 does not match per-child output size 1"
                    .to_string()
            )
        ));
    }
}
