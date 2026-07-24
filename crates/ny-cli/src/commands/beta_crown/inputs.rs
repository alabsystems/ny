// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::path::PathBuf;
use tracing::info;

use super::BetaCrownModel;

/// Check if the model needs batch dimension squeezed for Conv inputs.
pub(super) fn check_needs_squeeze(model_net: &BetaCrownModel) -> Result<bool> {
    match model_net {
        BetaCrownModel::Sequential(network) => {
            // Check if first layer is Conv, or first is Transpose and second is Conv
            let layers = network.layers();
            let first_is_conv = layers.first().is_some_and(|l| {
                matches!(
                    l,
                    ny_propagate::Layer::Conv2d(_) | ny_propagate::Layer::Conv1d(_)
                )
            });
            let transpose_then_conv = layers.len() >= 2
                && matches!(layers[0], ny_propagate::Layer::Transpose(_))
                && matches!(
                    layers[1],
                    ny_propagate::Layer::Conv2d(_) | ny_propagate::Layer::Conv1d(_)
                );
            Ok(first_is_conv || transpose_then_conv)
        }
        BetaCrownModel::Graph(graph) => {
            let exec_order = graph.exec_order()?;

            // Find nodes that directly take "_input"
            let first_nodes: Vec<_> = exec_order
                .iter()
                .filter_map(|name| {
                    let node = graph.node(name)?;
                    if node.inputs().iter().any(|i| i == "_input") {
                        Some((name.as_str(), node.layer()))
                    } else {
                        None
                    }
                })
                .collect();

            // Check if first layer is Conv, or if first is Transpose and second is Conv
            let squeeze = first_nodes.iter().any(|(name, layer)| {
                if matches!(
                    layer,
                    ny_propagate::Layer::Conv2d(_) | ny_propagate::Layer::Conv1d(_)
                ) {
                    return true;
                }
                // If first layer is Transpose, check if its output feeds a Conv
                if matches!(layer, ny_propagate::Layer::Transpose(_)) {
                    for next_name in exec_order {
                        if let Some(next_node) = graph.node(next_name) {
                            if next_node.inputs().iter().any(|i| i == *name)
                                && matches!(
                                    next_node.layer(),
                                    ny_propagate::Layer::Conv2d(_) | ny_propagate::Layer::Conv1d(_)
                                )
                            {
                                return true;
                            }
                        }
                    }
                }
                false
            });
            Ok(squeeze)
        }
    }
}

/// Create input bounds from VNNLIB property or epsilon-ball.
// Justification: Input bound creation needs property file, model dimensions, shape info,
// perturbation config, and output format — all from different CLI sources.
// Return type is a tagged union of (BoundedTensor, Option<VnnLibSpec>) variants.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(super) fn create_input_bounds(
    property: &Option<PathBuf>,
    preloaded_vnnlib: Option<ny_onnx::vnnlib::VnnLibSpec>,
    input_dim: usize,
    output_dim: usize,
    input_shape_from_model: &[usize],
    needs_squeeze: bool,
    is_graph_model: bool,
    epsilon: f32,
    threshold: f32,
    _json: bool,
) -> Result<(
    BoundedTensor,
    f32,
    Option<ny_onnx::vnnlib::VnnLibSpec>,
    bool,
    bool,
    Option<usize>,
)> {
    use super::constraint_plan::{classify_constraints, extract_constant_params};

    if let Some(prop_path) = property {
        use ny_onnx::vnnlib::load_vnnlib;

        let vnnlib = if let Some(spec) = preloaded_vnnlib {
            spec
        } else {
            load_vnnlib(prop_path)?
        };
        info!(
            "Loaded VNNLIB: {} inputs, {} outputs, {} constraints",
            vnnlib.num_inputs,
            vnnlib.num_outputs,
            vnnlib.output_constraints.len()
        );

        // Create input bounds from VNNLIB using model's expected input shape
        let (lower_bounds, upper_bounds) = vnnlib.split_input_bounds_f32();

        // #dd-zonotope (dark, `NY_DD_ZONOTOPE=1`, default-OFF): publish the
        // EXACT-decimal input box alongside the engine's f32 one.
        //
        // `split_input_bounds_f32` above widens every endpoint outward by one
        // f32 ULP (#2658) — correct for every existing path, and fatal for the
        // certified double-double zonotope: on vggnet16 it turns all 150527
        // declared-POINT pixels into 2-ULP intervals, which that method can
        // neither carry symbolically (150528 generator columns) nor carry as
        // intervals (the `ec` channel is transported by `|W|`, i.e. by IBP,
        // whose measured VGG16 gain is ~1e13). `CertifiedInputBox` reparses the
        // direct input atoms as exact rationals and rounds each endpoint
        // OUTWARD to f64, so a declared point stays a point.
        //
        // Registration is keyed by the byte-exact f32 box and re-checks
        // containment on lookup, so it can only ever be served back to THIS
        // instance. Gate-off skips the re-parse entirely.
        if ny_propagate::dd_zonotope::dd_zonotope_enabled() {
            match ny_onnx::vnnlib::load_vnnlib_with_certified_input_box(prop_path) {
                Ok((_, certified)) if certified.len() == lower_bounds.len() => {
                    let exact = ny_propagate::dd_zonotope::certified_box::ExactBox {
                        center_hi: certified.center_hi().to_vec(),
                        center_lo: certified.center_lo().to_vec(),
                        center_err: certified.center_err().to_vec(),
                        half_width: certified.half_width().to_vec(),
                        lower: certified.lower().to_vec(),
                        upper: certified.upper().to_vec(),
                    };
                    ny_propagate::dd_zonotope::certified_box::register(
                        &lower_bounds,
                        &upper_bounds,
                        exact,
                    );
                }
                Ok(_) => {
                    info!("#dd-zonotope: certified input box length mismatch; pass will refuse");
                }
                Err(e) => {
                    info!(
                        "#dd-zonotope: exact-decimal input box unavailable ({e}); pass will refuse"
                    );
                }
            }
        }

        // Validate dimensions match
        if vnnlib.num_inputs != input_dim {
            anyhow::bail!(
                "VNNLIB specifies {} inputs but model expects {} (shape {:?})",
                vnnlib.num_inputs,
                input_dim,
                input_shape_from_model
            );
        }
        if vnnlib.num_outputs != output_dim {
            anyhow::bail!(
                "VNNLIB specifies {} outputs but model expects {}",
                vnnlib.num_outputs,
                output_dim
            );
        }

        // Use model's actual input shape (may be [1,1,1,5] for ACAS-Xu ONNX)
        // Strip batch dimension of 1 when:
        // 1. Conv2d first layer (needs_squeeze) — Conv2d expects (C,H,W) not (N,C,H,W)
        // 2. Graph (DAG) models — all layer conversions (Slice, Squeeze, Unsqueeze)
        //    adjust axes by -1 assuming the batch dimension is removed. If the input
        //    retains the batch dim, axis 0 targets the wrong dimension.
        let mut effective_shape = input_shape_from_model.to_vec();
        if (needs_squeeze || is_graph_model)
            && effective_shape.len() >= 2
            && effective_shape[0] == 1
        {
            effective_shape.remove(0);
            info!(
                "Squeezed batch dimension for {} model, shape: {:?}",
                if needs_squeeze { "Conv" } else { "graph" },
                effective_shape
            );
        }
        let lower = ArrayD::from_shape_vec(IxDyn(&effective_shape), lower_bounds).map_err(|e| {
            anyhow::anyhow!(
                "Failed to create lower bounds with shape {:?}: {}",
                effective_shape,
                e
            )
        })?;
        let upper = ArrayD::from_shape_vec(IxDyn(&effective_shape), upper_bounds).map_err(|e| {
            anyhow::anyhow!(
                "Failed to create upper bounds with shape {:?}: {}",
                effective_shape,
                e
            )
        })?;
        let input = BoundedTensor::new(lower, upper)?;

        // Classify constraints via shared planning module (#1881)
        let classification = classify_constraints(&vnnlib);
        let has_relational = classification.has_relational;

        // Extract threshold from VNNLIB (for simple GreaterEqConst/LessEqConst properties)
        // VNNLIB specifies UNSAFE region:
        // - Y_i >= c (GreaterEqConst) means unsafe if output >= c, so prove upper < c
        // - Y_i <= c (LessEqConst) means unsafe if output <= c, so prove lower > c
        let (effective_threshold, verify_upper, const_output_idx) = if has_relational {
            // For relational constraints, threshold is 0 (prove difference > 0)
            (0.0f32, false, None)
        } else if let Some(params) = extract_constant_params(&vnnlib) {
            (
                params.threshold,
                params.verify_upper,
                Some(params.output_idx),
            )
        } else {
            (threshold, false, None)
        };

        Ok((
            input,
            effective_threshold,
            Some(vnnlib),
            verify_upper,
            has_relational,
            const_output_idx,
        ))
    } else {
        // Use epsilon-ball around zero with model's expected shape
        // Strip batch dimension for Conv or graph models (same logic as VNNLIB path above)
        let mut effective_shape = input_shape_from_model.to_vec();
        if (needs_squeeze || is_graph_model)
            && effective_shape.len() >= 2
            && effective_shape[0] == 1
        {
            effective_shape.remove(0);
            info!(
                "Squeezed batch dimension for {} model (epsilon-ball), shape: {:?}",
                if needs_squeeze { "Conv" } else { "graph" },
                effective_shape
            );
        }
        let center = ArrayD::from_elem(IxDyn(&effective_shape), 0.0f32);
        let input = BoundedTensor::from_epsilon(center, epsilon)?;
        Ok((input, threshold, None, false, false, None)) // Default: verify lower > threshold, no relational
    }
}
