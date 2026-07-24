// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Initialization phase for GPU BaB: root bound computation and DomainList setup.
//!
//! Extracted from the first ~250 lines of `verify_graph_gpu_domain_list`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::batched_domain::{CachedLinearBounds, DomainList, DomainListConfig, DomainMetadata};
use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::input_split::mul_binary::maybe_optimize_mul_binary_alphas;
use crate::beta_crown::engine::graph::input_split::root_bounds::collect_input_split_root_node_bounds;
use crate::beta_crown::engine::graph::input_split::shared::compute_crown_or_ibp_bounds;
use crate::beta_crown::engine::graph::shared::init::{
    compute_graph_bab_bootstrap, compute_graph_root_output_bounds,
};
use crate::beta_crown::engine::graph::shared::setup::{build_root_alpha_state, GraphBabSetup};
use crate::beta_crown::NonlinearBranching;
use crate::bounds::LinearBounds;
use crate::network::SpecCrownRequest;
use crate::GraphNetwork;

use super::super::domain_conversion::create_root_processed_domain;

pub(crate) const INPUT_SPLIT_LINEAR_BOUNDS_CACHE_KEY: &str = "__graph_input_split_linear__";

pub(crate) fn cache_input_split_linear_bounds(linear: &LinearBounds) -> CachedLinearBounds {
    let mut linear_map = HashMap::with_capacity(1);
    linear_map.insert(
        INPUT_SPLIT_LINEAR_BOUNDS_CACHE_KEY.to_string(),
        linear.clone(),
    );
    CachedLinearBounds::from_linear_bounds_map(linear_map)
}

pub(crate) fn restore_input_split_linear_bounds(metadata: &DomainMetadata) -> Option<LinearBounds> {
    metadata
        .cached_la()
        .as_ref()?
        .linear_bounds(INPUT_SPLIT_LINEAR_BOUNDS_CACHE_KEY)
}

pub(crate) struct InputSplitBootstrap {
    pub spec_matrix: ndarray::Array2<f32>,
    pub fixed_node_bounds: Option<HashMap<String, BoundedTensor>>,
    pub root_alpha_state: Option<crate::bounds::GraphAlphaState>,
    pub root_linear_bounds: Option<LinearBounds>,
    pub mul_binary_alphas: Option<HashMap<String, ndarray::Array2<f32>>>,
    pub deadline: Option<Instant>,
}

/// Result of the initialization phase.
pub(crate) struct InitResult {
    /// Initial per-node intermediate bounds.
    pub initial_node_bounds: HashMap<String, BoundedTensor>,
    /// Root alpha state from alpha-CROWN (if enabled).
    pub root_alpha_state: Option<crate::bounds::GraphAlphaState>,
    /// Initial output bounds (for early-exit reporting).
    pub initial_output: BoundedTensor,
    /// Root lower bound (spec-guided CROWN output).
    pub root_lower: f32,
    /// Root upper bound (spec-guided CROWN output).
    pub root_upper: f32,
    /// Reusable root context for GPU DomainList input split.
    pub input_split_bootstrap: Option<InputSplitBootstrap>,
}

/// Compute initial bounds using the configured graph bootstrap mode.
///
/// Returns intermediate node bounds, alpha state, and spec-guided output bounds.
///
/// Two deadlines mirror the CPU β-CROWN ReLU-split pattern (#4321/#4413):
///   * `deadline` (full BaB budget) gates the mandatory foundational node-bounds
///     sweep, which must reach every node so conv-heavy DAGs are not choked into a
///     premature "deadline exceeded before node 'Conv_0'";
///   * `iterative_deadline` (capped at `initial_bounds_fraction`) gates the
///     genuinely-iterative root α-CROWN warmup + spec-guided output optimization.
///     When it expires, α-CROWN bails with `DeadlineExceeded`, which the caller
///     turns into a warmup-cap `Unknown` instead of burning the whole budget.
pub(crate) fn compute_initial_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objective: &[f32],
    config: &BetaCrownConfig,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    iterative_deadline: Option<Instant>,
    is_input_split_mode: bool,
) -> Result<InitResult> {
    config.validate()?;

    let spec_matrix = ndarray::Array2::from_shape_vec((1, objective.len()), objective.to_vec())
        .map_err(|e| NyError::InvalidSpec(format!("spec matrix from objective: {}", e)))?;

    // Step 2: Compute spec-guided CROWN output bounds.
    //
    // #1817/#1848: The objective vector defines a linear combination of outputs
    // (e.g., [0,0,0,0,1] for "output 4 < threshold"). Using spec-guided CROWN
    // propagates the objective through the backward pass directly, preserving
    // output correlations and producing much tighter bounds than post-hoc
    // interval arithmetic on raw per-output bounds.
    if is_input_split_mode {
        // Keep the DomainList input-split warmup aligned with the heap path:
        // forward-linear warmup can miss the deadline and still fall back to a
        // conservative spec-bounds bootstrap instead of aborting before search.
        let (initial_node_bounds, root_alpha_state) = collect_input_split_root_node_bounds(
            graph,
            input,
            config,
            engine,
            deadline,
            "GPU BaB input split",
            None,
        )?;
        let fixed_node_bounds = if config.use_alpha_crown || config.use_forward_bounds {
            initial_node_bounds.clone()
        } else {
            None
        };
        let mul_binary_alphas = maybe_optimize_mul_binary_alphas(
            graph,
            input,
            &spec_matrix,
            engine,
            deadline,
            config.crown_backward_layers,
            "GPU BaB input split",
        )?;
        let (root_bounds, root_linear_bounds) = compute_crown_or_ibp_bounds(
            graph,
            input,
            &spec_matrix,
            engine,
            fixed_node_bounds.as_ref(),
            root_alpha_state.as_ref(),
            mul_binary_alphas.as_ref(),
            deadline,
            config.crown_backward_layers,
            config.input_split_ibp_enhancement,
        )?;
        if root_bounds.is_empty() {
            return Err(NyError::InvalidSpec(
                "spec-guided CROWN produced empty output tensor".to_string(),
            ));
        }
        let root_lower = root_bounds.lower()[[0]];
        let root_upper = root_bounds.upper()[[0]];
        if !root_lower.is_finite() || !root_upper.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "GPU BaB init: non-finite root bounds from spec-guided CROWN \
                 (lower={root_lower}, upper={root_upper})"
            )));
        }

        let initial_output = BoundedTensor::new(
            ndarray::arr1(&[root_lower]).into_dyn(),
            ndarray::arr1(&[root_upper]).into_dyn(),
        )?;
        info!(
            "GPU BaB (DomainList): initial bounds [{:.4}, {:.4}], threshold: {:.4}",
            root_lower,
            root_upper,
            0.0 // threshold logged at call site
        );

        return Ok(InitResult {
            initial_node_bounds: initial_node_bounds.unwrap_or_default(),
            root_alpha_state: root_alpha_state.clone(),
            initial_output,
            root_lower,
            root_upper,
            input_split_bootstrap: Some(InputSplitBootstrap {
                spec_matrix,
                fixed_node_bounds,
                root_alpha_state,
                root_linear_bounds,
                mul_binary_alphas,
                deadline,
            }),
        });
    }

    // α-CROWN runs the iterative warmup *inside* the bootstrap, so it must honor
    // the capped `iterative_deadline` (#4413). The non-alpha foundational IBP /
    // CROWN-IBP node-bounds sweep is mandatory and gets the full `deadline` to
    // avoid choking conv-heavy DAGs (#4321).
    let bootstrap_deadline = if config.use_alpha_crown {
        iterative_deadline
    } else {
        deadline
    };
    let bootstrap = compute_graph_bab_bootstrap(graph, input, config, engine, bootstrap_deadline)?;
    let spec_output = SpecCrownRequest::new(graph, input, &spec_matrix, engine)
        .node_bounds(&bootstrap.initial_node_bounds)
        .alpha_state_opt(bootstrap.root_alpha_state.as_ref())
        .deadline_opt(iterative_deadline)
        .truncate_after_opt(config.crown_backward_layers)
        .run()?;
    if spec_output.is_empty() {
        return Err(NyError::InvalidSpec(
            "spec-guided CROWN produced empty output tensor".to_string(),
        ));
    }
    let root_lower = spec_output.lower()[[0]];
    let root_upper = spec_output.upper()[[0]];

    if !root_lower.is_finite() || !root_upper.is_finite() {
        return Err(NyError::NumericalInstability(format!(
            "GPU BaB init: non-finite root bounds from spec-guided CROWN \
             (lower={root_lower}, upper={root_upper})"
        )));
    }

    let initial_output = compute_graph_root_output_bounds(
        graph,
        input,
        config,
        engine,
        &bootstrap,
        iterative_deadline,
    )?;

    info!(
        "GPU BaB (DomainList): initial bounds [{:.4}, {:.4}], threshold: {:.4}",
        root_lower,
        root_upper,
        0.0 // threshold logged at call site
    );

    Ok(InitResult {
        initial_node_bounds: bootstrap.initial_node_bounds,
        root_alpha_state: bootstrap.root_alpha_state,
        initial_output,
        root_lower,
        root_upper,
        input_split_bootstrap: None,
    })
}

/// Build and initialize the DomainList with the root domain.
///
/// Returns the domain list and sorted layer names.
pub(crate) fn create_domain_list(
    init: &InitResult,
    input: &BoundedTensor,
    graph: &GraphNetwork,
    config: &BetaCrownConfig,
    is_input_split_mode: bool,
    setup: &GraphBabSetup,
) -> Result<(DomainList, Vec<String>)> {
    // Sort layer names for deterministic ordering.
    //
    // #3089: derive the returned sorted layer names from the GRAPH nodes, not from
    // `init.initial_node_bounds`. In input-split mode without an alpha/forward-bounds
    // bootstrap, `initial_node_bounds` is empty (collect_input_split_root_node_bounds
    // returns None), but downstream GPU/CPU batch processing in the BaB loop still
    // needs the full sorted graph layer names to reconstruct per-layer intermediates.
    // The two key sets coincide on the non-input-split path, where initial_node_bounds
    // is fully populated.
    let mut layer_names: Vec<String> = graph.nodes.keys().cloned().collect();
    layer_names.sort();

    let (domain_list_layer_names, layer_shapes) = if is_input_split_mode {
        // Input-split domains recompute output bounds from input bounds each
        // iteration and do not carry per-layer intermediates.
        (Vec::new(), HashMap::new())
    } else {
        let mut layer_shapes = HashMap::new();
        for name in &layer_names {
            let bounds = init.initial_node_bounds.get(name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "missing initial bounds for layer '{}' when building DomainList",
                    name
                ))
            })?;
            layer_shapes.insert(name.clone(), bounds.lower().shape().to_vec());
        }
        (layer_names.clone(), layer_shapes)
    };
    let input_shape: Vec<usize> = input.shape().to_vec();

    let dl_config = DomainListConfig {
        // BreadthFirst (FIFO/queue): after periodic sort by bound quality,
        // best-bound domains at the front are processed before newly-split
        // children (appended at back). This preserves sorted ordering between
        // sort intervals, matching CPU BinaryHeap's best-first behavior.
        // DepthFirst (LIFO/stack) processes new children before sorted domains,
        // biasing toward deep, hard-to-verify regions. See #3870.
        traversal: ny_tensor::TreeTraversal::BreadthFirst,
        layer_names: domain_list_layer_names.clone(),
        layer_shapes,
        input_shape,
        initial_capacity: config.batch_size.max(64),
        max_queue_size: config.max_queue_size,
    };

    let mut domain_list = DomainList::new(dl_config)?;

    // Add root domain, with root alpha initialized the same way as the heap paths:
    // optimized alpha-CROWN values when available, otherwise heuristic alpha
    // from graph bounds.
    let mut root_processed = create_root_processed_domain(
        &init.initial_node_bounds,
        input,
        init.root_lower,
        init.root_upper,
        &domain_list_layer_names,
    )?;
    if let Some(meta) = root_processed.metadata.first_mut() {
        let empty_history = crate::beta_crown::branching::GraphSplitHistory::new();
        let domain_alpha = build_root_alpha_state(
            graph,
            input,
            &empty_history,
            &setup.initial_node_bounds_arc,
            init.root_alpha_state.as_ref(),
            config.beta_iterations > 0,
        );
        meta.set_alpha_state(Some(domain_alpha));
        if let Some(bootstrap) = init.input_split_bootstrap.as_ref() {
            if let Some(root_linear_bounds) = bootstrap.root_linear_bounds.as_ref() {
                meta.cached_la = Some(Arc::new(cache_input_split_linear_bounds(
                    root_linear_bounds,
                )));
            }
        }
    }
    domain_list.add(root_processed)?;

    Ok((domain_list, layer_names))
}

/// Setup context: identifies ReLU nodes, builds pre-activation map, creates GenBaB instance.
pub(crate) struct BabSetupContext {
    /// ReLU node names in the graph.
    pub relu_nodes: Vec<String>,
    /// Maps ReLU node name -> pre-activation layer name.
    pub relu_pre_map: HashMap<String, String>,
    /// GenBaB branching instance (if configured).
    pub genbab_instance: Option<NonlinearBranching>,
    /// Splittable nonlinear node names for GenBaB.
    pub nonlinear_nodes: Vec<String>,
}

/// Build the BaB setup context: identify ReLU/nonlinear nodes and branching instances.
pub(crate) fn build_setup_context(
    graph: &GraphNetwork,
    config: &BetaCrownConfig,
    relu_nodes: Vec<String>,
) -> BabSetupContext {
    let relu_pre_map: HashMap<String, String> = relu_nodes
        .iter()
        .filter_map(|relu_name| {
            let node = graph.nodes.get(relu_name)?;
            let pre_name = node.inputs.first()?.clone();
            Some((relu_name.clone(), pre_name))
        })
        .collect();

    let genbab_instance: Option<NonlinearBranching> = match &config.branching_heuristic {
        BranchingHeuristic::GenBaB(genbab_config) => {
            Some(NonlinearBranching::new(genbab_config.clone()))
        }
        _ => None,
    };

    let nonlinear_nodes: Vec<String> = if genbab_instance.is_some() {
        graph
            .nodes
            .iter()
            .filter_map(|(name, node)| {
                // Include elementwise activations (ReLU, GELU, etc.),
                // BilinearCrown, and MulBinary nodes for BaB splitting on bilinear
                // inputs. BilinearCrown (attention Q@K^T) and MulBinary (element-wise
                // x·y, e.g. ml4acopf power flow) are both McCormick-relaxed: splitting
                // an input interval reduces the envelope gap (ux−lx)(uy−ly)/4 that
                // frozen root facets cannot close.
                // Reference: auto_LiRPA BoundMatMul.splittable (linear.py:948).
                if node.layer.is_elementwise_activation()
                    || matches!(
                        node.layer,
                        crate::layers::Layer::BilinearCrown(_)
                            | crate::layers::Layer::MulBinary(_)
                            // #norm-genbab: RmsNorm branchable on internal inv_rms.
                            | crate::layers::Layer::RmsNorm(_)
                    )
                {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    BabSetupContext {
        relu_nodes,
        relu_pre_map,
        genbab_instance,
        nonlinear_nodes,
    }
}

#[cfg(test)]
mod tests;
