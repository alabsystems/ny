// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork bound profiling for native format models.

use ny_propagate::layers::Layer;
use ny_propagate::BoundPropagation;
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use super::stats::{difficulty_score, make_unit_variance_input, median};
use super::types::{BoundStatus, LayerProfile, ProfileConfig, ProfileError, ProfileResult};
use crate::analysis_error::validate_analysis_epsilon;

/// Analyze a GraphNetwork's bound profile using the normalized `analyze_*`
/// verb family.
pub fn analyze_profile_graph(
    graph: &ny_propagate::GraphNetwork,
    config: &ProfileConfig,
    input_shape: &[usize],
) -> Result<ProfileResult, ProfileError> {
    profile_bounds_graph(graph, config, input_shape)
}

/// Profile bounds of a GraphNetwork (for native format models).
///
/// This function profiles IBP bounds through a GraphNetwork, tracking bound width
/// at each node. Useful for diagnosing bound explosion in GGUF/SafeTensors models.
pub fn profile_bounds_graph(
    graph: &ny_propagate::GraphNetwork,
    config: &ProfileConfig,
    input_shape: &[usize],
) -> Result<ProfileResult, ProfileError> {
    validate_analysis_epsilon("profile/graph", config.epsilon)?;

    // Create input tensor
    // Use unit-variance input to avoid artificial amplification in LayerNorm/RMSNorm
    let input = if let Some(ref inp) = config.input {
        inp.clone()
    } else {
        make_unit_variance_input(input_shape, config.epsilon)
            .map_err(|e| ProfileError::propagation("profile/graph", e))?
    };

    let initial_width = input.max_width();

    info!(
        "Starting graph bound profile with input shape {:?}, epsilon {}, initial width {}",
        input.shape(),
        config.epsilon,
        initial_width
    );

    // Get topological order for processing
    let exec_order = graph
        .exec_order()
        .map_err(|e| ProfileError::propagation("profile/graph", e))?;

    if exec_order.is_empty() {
        return Err(ProfileError::no_layers("profile/graph"));
    }

    // Track layer-by-layer bounds
    let mut layers = Vec::new();
    let mut max_growth_ratio: f32 = 1.0;
    let mut max_growth_layer: Option<usize> = None;
    let mut overflow_at_layer: Option<usize> = None;

    // Cache bounds for each node
    let mut bounds_cache: std::collections::HashMap<String, BoundedTensor> =
        std::collections::HashMap::new();
    // Nodes whose diagnostic bounds depend on a substituted value after a
    // propagation error. Taint follows graph edges, so an independent branch
    // remains meaningful while every descendant of a failure stays failed.
    let mut propagation_failed_nodes = std::collections::HashSet::<String>::new();

    // Helper to get bounds for an input (either from cache or network input)
    fn get_bounds<'a>(
        input_name: &str,
        network_input: &BoundedTensor,
        cache: &'a std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<std::borrow::Cow<'a, BoundedTensor>, ProfileError> {
        if input_name == "_input" {
            Ok(std::borrow::Cow::Owned(network_input.clone()))
        } else {
            cache
                .get(input_name)
                .map(std::borrow::Cow::Borrowed)
                .ok_or_else(|| {
                    ProfileError::propagation_msg(
                        "profile/graph",
                        format!("Input {} not found in cache", input_name),
                    )
                })
        }
    }

    // Process nodes in topological order
    for (i, node_name) in exec_order.iter().enumerate() {
        let node = graph.node(node_name).ok_or_else(|| {
            ProfileError::propagation_msg("profile/graph", format!("Node not found: {}", node_name))
        })?;

        let layer_type = format!("{:?}", node.layer().layer_type());

        // Get input width (from first input)
        let input_width = if node.inputs().is_empty() {
            initial_width
        } else {
            get_bounds(&node.inputs()[0], &input, &bounds_cache)
                .map(|b| b.max_width())
                .unwrap_or(initial_width)
        };

        // Propagate bounds through this node.
        // Concat MUST be checked before is_binary() because Layer::is_binary()
        // returns true for Concat. Without this ordering, n-ary Concat (3+ inputs)
        // would silently drop inputs beyond the first two. (#2405)
        let mut propagation_failed = node
            .inputs()
            .iter()
            .any(|input_name| propagation_failed_nodes.contains(input_name));
        let output = if let Layer::Concat(concat) = node.layer() {
            // Handle constant_inputs interleaving (same pattern as dispatch.rs). (#2405)
            let owned_inputs: Vec<BoundedTensor> = if let Some(ref ci) = concat.constant_inputs {
                let mut graph_idx = 0;
                ci.iter()
                    .map(|const_opt| {
                        if let Some(constant) = const_opt {
                            Ok(constant.clone())
                        } else {
                            let name = node.inputs().get(graph_idx).ok_or_else(|| {
                                ProfileError::propagation_msg(
                                    "profile/graph",
                                    format!("Concat: ran out of graph inputs at idx {}", graph_idx),
                                )
                            })?;
                            graph_idx += 1;
                            Ok(get_bounds(name, &input, &bounds_cache)?.into_owned())
                        }
                    })
                    .collect::<Result<Vec<_>, ProfileError>>()?
            } else {
                node.inputs()
                    .iter()
                    .map(|name| Ok(get_bounds(name, &input, &bounds_cache)?.into_owned()))
                    .collect::<Result<Vec<_>, ProfileError>>()?
            };
            let input_refs: Vec<&BoundedTensor> = owned_inputs.iter().collect();
            match concat.propagate_ibp_nary(&input_refs) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(ProfileError::propagation("profile/graph", e));
                    }
                    if overflow_at_layer.is_none() {
                        overflow_at_layer = Some(i);
                    }
                    propagation_failed = true;
                    owned_inputs
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| input.clone())
                }
            }
        } else if node.layer().is_binary() {
            if node.inputs().len() < 2 {
                return Err(ProfileError::propagation_msg(
                    "profile/graph",
                    format!("Binary node {} requires 2 inputs", node_name),
                ));
            }
            let input_a = get_bounds(&node.inputs()[0], &input, &bounds_cache)?;
            let input_b = get_bounds(&node.inputs()[1], &input, &bounds_cache)?;
            match node.layer().propagate_ibp_binary(&input_a, &input_b) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(ProfileError::propagation("profile/graph", e));
                    }
                    if overflow_at_layer.is_none() {
                        overflow_at_layer = Some(i);
                    }
                    propagation_failed = true;
                    input_a.into_owned()
                }
            }
        } else {
            if node.inputs().is_empty() {
                return Err(ProfileError::propagation_msg(
                    "profile/graph",
                    format!("Node {} has no inputs", node_name),
                ));
            }
            let node_input = get_bounds(&node.inputs()[0], &input, &bounds_cache)?;
            match node.layer().propagate_ibp(&node_input) {
                Ok(out) => out,
                Err(e) => {
                    debug!("Node {} propagation failed: {}", node_name, e);
                    if !config.continue_after_overflow {
                        return Err(ProfileError::propagation("profile/graph", e));
                    }
                    if overflow_at_layer.is_none() {
                        overflow_at_layer = Some(i);
                    }
                    propagation_failed = true;
                    node_input.into_owned()
                }
            }
        };

        let output_width = output.max_width();
        let widths: Vec<f32> = output.width().iter().cloned().collect();
        let mean_width = widths.iter().sum::<f32>() / widths.len().max(1) as f32;
        let median_width = median(&widths);

        // Calculate growth ratio
        let growth_ratio = if input_width > 0.0 && input_width.is_finite() {
            output_width / input_width
        } else {
            1.0
        };

        // Track max growth
        if growth_ratio > max_growth_ratio && growth_ratio.is_finite() {
            max_growth_ratio = growth_ratio;
            max_growth_layer = Some(i);
        }

        // Calculate cumulative expansion from input
        let cumulative_expansion = if initial_width > 0.0 && initial_width.is_finite() {
            output_width / initial_width
        } else {
            1.0
        };

        // Determine status
        let has_overflow = propagation_failed || !output_width.is_finite();
        let status = if has_overflow {
            if overflow_at_layer.is_none() {
                overflow_at_layer = Some(i);
            }
            BoundStatus::Overflow
        } else {
            BoundStatus::from_width(output_width, config.epsilon)
        };

        layers.push(LayerProfile {
            name: node_name.clone(),
            layer_type,
            input_width,
            output_width,
            mean_output_width: mean_width,
            median_output_width: median_width,
            growth_ratio,
            cumulative_expansion,
            output_shape: output.shape().to_vec(),
            num_elements: output.lower().len(),
            status,
        });

        debug!(
            "Node {}: width {} -> {}, growth {:.2}x",
            node_name, input_width, output_width, growth_ratio
        );

        // Stop if overflow and not continuing
        if has_overflow && !config.continue_after_overflow {
            break;
        }

        if propagation_failed {
            propagation_failed_nodes.insert(node_name.clone());
        }
        bounds_cache.insert(node_name.clone(), output);
    }

    let final_width = layers
        .last()
        .map(|l| l.output_width)
        .unwrap_or(initial_width);
    let total_expansion = if initial_width > 0.0 && initial_width.is_finite() {
        final_width / initial_width
    } else {
        1.0
    };

    let difficulty = difficulty_score(
        total_expansion,
        max_growth_ratio,
        overflow_at_layer.is_some(),
    );

    Ok(ProfileResult {
        layers,
        input_epsilon: config.epsilon,
        initial_width,
        final_width,
        total_expansion,
        max_growth_layer,
        max_growth_ratio,
        overflow_at_layer,
        difficulty_score: difficulty,
    })
}
