// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Model loading and graph-routing helpers for `ny beta-crown`.

use anyhow::Result;
use ny_onnx::{
    load_onnx_with_config, vnnlib::VnnLibSpec, CompoundNodePolicy, GraphNetworkOptions,
    OnnxLoadConfig,
};
use ny_propagate::{BranchingHeuristic, Layer, VggMaxPoolRewriteMode};
use std::path::Path;
use tracing::{info, warn};

use super::branching::{
    resolve_auto_branching, AutoBranchingRequest, ModelStructure, ResolvedAutoBranching,
};
use super::{routing::route_conv_model_to_graph, BetaCrownModel};
use crate::CompleteVerifierArg;

pub(super) struct LoadedModel {
    pub(super) model: BetaCrownModel,
    pub(super) param_count: usize,
    pub(super) input_dim: usize,
    pub(super) output_dim: usize,
    pub(super) input_shape: Vec<usize>,
    pub(super) is_graph: bool,
    pub(super) preloaded_vnnlib: Option<VnnLibSpec>,
    /// True when the loader auto-peeled a terminal Sigmoid (#cgan-sigmoid-peel).
    /// A counterexample's declared Y values then come from the PEELED network
    /// (pre-sigmoid logits) and must be mapped y = sigmoid(z) at emission so
    /// the witness matches the ORIGINAL graph.
    pub(super) sigmoid_peeled: bool,
    /// The model-class-aware `--branching auto` decision, resolved here once the
    /// network's structural signals are known. `None` when `auto` was not
    /// requested (a preset or explicit CLI token owns branching instead).
    pub(super) auto_branching: Option<ResolvedAutoBranching>,
}

/// Load a model from NNet or ONNX and resolve the beta-crown execution path.
// Justification: this helper keeps the independent CLI routing knobs explicit
// until #4246 finishes the handler decomposition; wrapping them again here
// would only mirror the top-level command surface without reducing complexity.
#[allow(clippy::too_many_arguments)]
pub(super) fn load_model(
    model_path: &Path,
    onnx_load_config: &OnnxLoadConfig,
    property: Option<&Path>,
    peel_last_softmax_layer: bool,
    effective_branching: Option<&BranchingHeuristic>,
    use_relu_split: bool,
    use_alpha: bool,
    preset_conv_mode: Option<ny_propagate::ConvMode>,
    enable_cuts: bool,
    complete_verifier: CompleteVerifierArg,
    json: bool,
    // When `Some`, the caller requested `--branching auto`. We resolve it HERE,
    // once the model's structural signals (param_count, conv presence, ReLU node
    // count, DAG-ness) are known, and use the resolved decision for our own
    // DAG/conv routing so the choice is consistent in a single load pass.
    auto_request: Option<AutoBranchingRequest>,
    // Default-off alpha-beta-CROWN VGG MaxPool treatment. The caller resolves
    // the variant from the property's perturbed-input count.
    vgg_maxpool_rewrite: Option<VggMaxPoolRewriteMode>,
) -> Result<(LoadedModel, bool)> {
    let is_nnet = model_path.extension().and_then(|ext| ext.to_str()) == Some("nnet");
    if is_nnet {
        use ny_onnx::nnet::load_nnet;

        let nnet = load_nnet(model_path)?;
        info!(
            "Loaded NNet: {} layers, {} inputs, {} outputs, {} params",
            nnet.num_layers(),
            nnet.input_size(),
            nnet.output_size(),
            nnet.param_count()
        );

        // NNet networks are always fully-connected ReLU nets (sequential, no conv,
        // no DAG). The hidden layers carry the ReLU activations.
        let auto_branching = auto_request.map(|request| {
            let structure = ModelStructure {
                param_count: nnet.param_count(),
                has_conv: false,
                relu_node_count: nnet.num_layers().saturating_sub(1),
                is_dag: false,
            };
            resolve_auto_branching(request, structure, nnet.input_size())
        });

        let network = nnet.to_prop_network()?;
        let resolved_relu_split = auto_branching
            .as_ref()
            .map(|r| r.use_relu_split)
            .unwrap_or(use_relu_split);
        return Ok((
            LoadedModel {
                model: BetaCrownModel::Sequential(Box::new(network)),
                param_count: nnet.param_count(),
                input_dim: nnet.input_size(),
                output_dim: nnet.output_size(),
                input_shape: vec![1, nnet.input_size()],
                is_graph: false,
                preloaded_vnnlib: None,
                sigmoid_peeled: false,
                auto_branching,
            },
            resolved_relu_split,
        ));
    }

    let mut onnx_model = load_onnx_with_config(model_path, onnx_load_config)?;
    let (network_name, layer_count, param_count) = {
        let onnx_network = &onnx_model.network;
        (
            onnx_network.name.clone(),
            onnx_network.layers.len(),
            onnx_network.param_count,
        )
    };
    info!(
        "Loaded network: {} ({} layers, {} params)",
        network_name, layer_count, param_count
    );

    let mut preloaded_vnnlib = None;
    let mut sigmoid_peeled = false;
    if let Some(prop_path) = property {
        use ny_onnx::vnnlib::load_vnnlib;

        // Parse the property HERE (it is reused downstream via
        // `preloaded_vnnlib` — parsed exactly once either way): the routing
        // decision below needs its per-clause-box disjunction shape, and the
        // softmax peel needs its constraints.
        let mut vnnlib = load_vnnlib(prop_path)?;
        if peel_last_softmax_layer {
            let report = ny_onnx::peel_off_last_softmax_layer(&mut onnx_model, &mut vnnlib);
            if report.peeled && !json {
                warn!(
                    "Peeled off terminal {:?} layer using VNN-LIB constraints.",
                    report.layer_type
                );
            }
        } else {
            // #cgan-sigmoid-peel: default-ON auto-peel for the exactly-
            // invertible case (terminal Sigmoid + all-constant thresholds,
            // e.g. the cgan upsample band specs). NY_SIGMOID_PEEL=0 disables.
            let report = ny_onnx::peel_off_terminal_sigmoid_auto(&mut onnx_model, &mut vnnlib);
            sigmoid_peeled = report.peeled;
            if report.peeled && !json {
                warn!("Auto-peeled terminal Sigmoid using VNN-LIB constant thresholds.");
            }
        }
        preloaded_vnnlib = Some(vnnlib);
    }

    // Flattened input element count, needed BEFORE the routing decision so that
    // model-class-aware auto-branching can be resolved here and feed routing in a
    // single load pass.
    let input_shape = onnx_model
        .network
        .inputs
        .first()
        .map(|input| ny_onnx::resolve_dynamic_shape(&input.shape, 1))
        .unwrap_or_else(|| vec![1]);
    let input_dim: usize = input_shape.iter().product();

    let graph_options = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };
    let mut graph = onnx_model.to_graph_network_with_options(graph_options)?;
    if let Some(mode) = vgg_maxpool_rewrite {
        match graph.rewrite_vgg_maxpool2x2(mode) {
            Ok(report) => {
                info!(
                    "VGG MaxPool rewrite ({mode:?}): {} rewritten, {} retained",
                    report.rewritten.len(),
                    report.skipped.len()
                );
                for (name, reason) in &report.skipped {
                    warn!("VGG MaxPool rewrite retained '{name}': {reason}");
                }
                if !json && !report.rewritten.is_empty() {
                    println!(
                        "VGG MaxPool treatment: rewrote {} eligible pool node(s) with {mode:?} primitives",
                        report.rewritten.len()
                    );
                }
            }
            Err(error) => {
                warn!("VGG MaxPool rewrite failed ({error}); continuing with the original graph")
            }
        }
    }
    let needs_graph = graph.node_names().iter().any(|name| {
        graph
            .node(name)
            .is_some_and(|node| node.layer().is_binary())
    });

    // Resolve `--branching auto` HERE, now that the network's structural signals
    // (param_count, conv presence, ReLU-node count) and the DAG flag (`needs_graph`)
    // are all known. The resolved decision overrides the routing inputs below so
    // the conv/DAG routing is consistent in this single pass. When `auto` was not
    // requested, the passed-in `effective_branching` / `use_relu_split` stand.
    // SOUND: the branching choice never changes a verdict.
    let auto_branching = auto_request.map(|request| {
        let structure = ModelStructure::from_network(&onnx_model.network, needs_graph);
        resolve_auto_branching(request, structure, input_dim)
    });
    let effective_heuristic: Option<BranchingHeuristic> = auto_branching
        .as_ref()
        .map(|r| r.heuristic.clone())
        .or_else(|| effective_branching.cloned());
    let use_relu_split = auto_branching
        .as_ref()
        .map(|r| r.use_relu_split)
        .unwrap_or(use_relu_split);

    let is_input_split = effective_heuristic
        .as_ref()
        .is_some_and(|heuristic| matches!(heuristic, BranchingHeuristic::InputSplit));
    if needs_graph {
        info!("Detected non-sequential graph (binary ops); using GraphNetwork path");
        if !is_input_split && !use_relu_split {
            anyhow::bail!(
                "Model is a DAG (e.g., residual/attention). β-CROWN supports DAGs with --branching input (input splitting) or --branching relu (ReLU splitting)"
            );
        }
    }

    let has_conv2d = !needs_graph
        && graph.node_names().iter().any(|name| {
            graph.node(name).is_some_and(|node| {
                matches!(node.layer(), Layer::Conv2d(_) | Layer::ConvTranspose2d(_))
            })
        });
    let use_patches_mode = preset_conv_mode
        .unwrap_or_default()
        .use_patches(enable_cuts);
    let (use_graph_for_conv, routed_relu_split) = route_conv_model_to_graph(
        has_conv2d,
        complete_verifier,
        use_relu_split,
        is_input_split,
        effective_heuristic.is_some(),
        use_alpha,
        use_patches_mode,
    );
    if has_conv2d && use_graph_for_conv {
        if complete_verifier == CompleteVerifierArg::Mip {
            info!("Conv2d detected: using GraphNetwork for BaB (MIP fallback available)");
        } else if !use_patches_mode && !use_relu_split && !is_input_split {
            info!(
                "Conv2d detected: conv_mode requires matrix backward, auto-selecting GraphNetwork ReLU splitting"
            );
        } else if use_relu_split || is_input_split {
            info!("Conv2d detected: using GraphNetwork for conv-mode-compatible BaB");
        } else if effective_heuristic.is_none() && use_alpha {
            info!("Conv2d detected: auto-selecting ReLU splitting for patches-mode alpha-CROWN");
        }
    }

    // Massive per-clause-box disjunction routing (#mono-corner): nn4sys
    // lindex-shaped specs (tens of thousands of clauses, each over its own
    // tiny input sub-box) are decided by the batched box-refinement screen's
    // sound-f64 lanes (zeroth/centered/mono-corner), which exist for Graph
    // models only. A pure-MLP model otherwise routes Sequential and burns
    // the whole budget in one f32 CROWN pass per clause (measured:
    // lindex_60000, 120k clauses at ~0.8ms/pass ≈ 96s against a 40s
    // budget). Route to Graph when the spec has per-clause input boxes on
    // many clauses AND the graph supports the sound f64 forward. Small
    // disjunctions (acasxu prop_6-class, ≤ dozens of clauses over WIDE
    // boxes) keep the Sequential per-clause input-split BaB lane, which is
    // measured-better there. `NY_DISJ_GRAPH_ROUTE=0` restores the old
    // routing. SOUND: this changes which sound lane runs, never how a proof
    // is judged.
    let huge_clause_box_disjunction = !needs_graph
        && !use_graph_for_conv
        && preloaded_vnnlib.as_ref().is_some_and(|v| {
            v.per_clause_input_bounds
                .iter()
                .filter(|b| !b.is_empty())
                .count()
                >= 512
        })
        && graph.supports_ibp_f64_cell()
        && !std::env::var("NY_DISJ_GRAPH_ROUTE").is_ok_and(|v| v == "0");
    if huge_clause_box_disjunction {
        info!(
            "Massive per-clause-box disjunction on an f64-supported net: \
             using GraphNetwork path for the box-refinement screen"
        );
    }

    let use_graph = needs_graph || use_graph_for_conv || huge_clause_box_disjunction;
    let output_dim = onnx_model
        .network
        .outputs
        .first()
        .map(|output| {
            ny_onnx::resolve_dynamic_shape(&output.shape, 1)
                .iter()
                .product::<usize>()
                .max(1)
        })
        .unwrap_or(1);
    let model = if use_graph {
        BetaCrownModel::Graph(Box::new(graph))
    } else {
        BetaCrownModel::Sequential(Box::new(onnx_model.to_propagate_network()?))
    };

    Ok((
        LoadedModel {
            model,
            param_count,
            input_dim,
            output_dim,
            input_shape,
            is_graph: use_graph,
            preloaded_vnnlib,
            sigmoid_peeled,
            auto_branching,
        },
        routed_relu_split,
    ))
}
