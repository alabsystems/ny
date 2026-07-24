// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Model loading for the verify command — NNet, native, and ONNX paths.

use anyhow::Result;
use ny_propagate::{GraphNetwork, PropagationMethod};
use std::path::Path;
use tracing::info;

use super::super::JsonCliError;
use super::options::configure_layernorm;
use crate::{LayerNormModeArg, LayerNormNormModeArg};

/// Holds either a sequential or graph-based verifiable network.
pub(crate) enum VerifiableNetwork {
    Sequential(Box<ny_propagate::Network>),
    Graph(Box<GraphNetwork>),
}

impl VerifiableNetwork {
    pub(crate) fn set_logsoftmax_sound_mode(&mut self, sound: bool) -> usize {
        match self {
            VerifiableNetwork::Sequential(net) => net.set_logsoftmax_sound_mode(sound),
            VerifiableNetwork::Graph(graph) => graph.set_logsoftmax_sound_mode(sound),
        }
    }

    pub(crate) fn set_softmax_sound_mode(&mut self, sound: bool) -> usize {
        match self {
            VerifiableNetwork::Sequential(net) => net.set_softmax_sound_mode(sound),
            VerifiableNetwork::Graph(graph) => graph.set_softmax_sound_mode(sound),
        }
    }

    pub(crate) fn set_causal_softmax_sound_mode(&mut self, sound: bool) -> usize {
        match self {
            VerifiableNetwork::Sequential(net) => net.set_causal_softmax_sound_mode(sound),
            VerifiableNetwork::Graph(graph) => graph.set_causal_softmax_sound_mode(sound),
        }
    }

    pub(crate) fn set_gelu_sound_mode(&mut self, sound: bool) -> usize {
        match self {
            VerifiableNetwork::Sequential(net) => net.set_gelu_sound_mode(sound),
            VerifiableNetwork::Graph(graph) => graph.set_gelu_sound_mode(sound),
        }
    }

    pub(crate) fn set_sin_sound_mode(&mut self, sound: bool) -> usize {
        match self {
            VerifiableNetwork::Sequential(net) => net.set_sin_sound_mode(sound),
            VerifiableNetwork::Graph(graph) => graph.set_sin_sound_mode(sound),
        }
    }

    pub(crate) fn set_cos_sound_mode(&mut self, sound: bool) -> usize {
        match self {
            VerifiableNetwork::Sequential(net) => net.set_cos_sound_mode(sound),
            VerifiableNetwork::Graph(graph) => graph.set_cos_sound_mode(sound),
        }
    }
}

/// Result of loading a model: the network, input shape, and output dimension.
pub(crate) struct LoadedModel {
    pub(crate) network: VerifiableNetwork,
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_dim: usize,
    /// Pre-loaded VNNLIB spec (when softmax peeling modifies the model during load).
    pub(crate) preloaded_vnnlib: Option<ny_onnx::vnnlib::VnnLibSpec>,
}

/// Detect if the model file is NNet format.
pub(crate) fn is_nnet_format(model: &Path) -> bool {
    model.extension().and_then(|e| e.to_str()) == Some("nnet")
}

/// Detect if the model should use native loader.
pub(crate) fn should_use_native(model: &Path, native_flag: bool, is_nnet: bool) -> bool {
    !is_nnet
        && (native_flag || model.is_dir() || {
            let ext = model.extension().and_then(|e| e.to_str()).unwrap_or("");
            matches!(
                ext,
                "pt" | "pth" | "bin" | "safetensors" | "gguf" | "mlmodel" | "mlpackage"
            )
        })
}

/// Load a model from the given path, returning the network, input shape, and output dimension.
///
/// Handles NNet, native, and ONNX formats. Applies LayerNorm configuration for native models.
// Justification: Parameters correspond to CLI flags for model format, LayerNorm settings,
// softmax peeling, and property path. Grouping into a config struct would add indirection
// for a single call site.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_model(
    model: &Path,
    native: bool,
    conservative_layernorm: bool,
    layernorm_mode: LayerNormModeArg,
    layernorm_norm_mode: LayerNormNormModeArg,
    effective_method: PropagationMethod,
    peel_off_last_softmax_layer: bool,
    property: Option<&Path>,
    json: bool,
) -> Result<LoadedModel> {
    let is_nnet = is_nnet_format(model);
    let use_native = should_use_native(model, native, is_nnet);

    let mut preloaded_vnnlib: Option<ny_onnx::vnnlib::VnnLibSpec> = None;

    if peel_off_last_softmax_layer && property.is_none() && !json {
        eprintln!(
            "Warning: --peel-off-last-softmax-layer requires --property (VNN-LIB); flag ignored."
        );
    }

    let (network, input_shape, output_dim) = if is_nnet {
        load_nnet(model)?
    } else if use_native {
        load_native(
            model,
            conservative_layernorm,
            layernorm_mode,
            layernorm_norm_mode,
            effective_method,
            json,
        )?
    } else {
        let (net, shape, dim, vnnlib) =
            load_onnx_model(model, peel_off_last_softmax_layer, property, json)?;
        preloaded_vnnlib = vnnlib;
        (net, shape, dim)
    };

    Ok(LoadedModel {
        network,
        input_shape,
        output_dim,
        preloaded_vnnlib,
    })
}

/// Load an NNet format model (VNN-COMP / ACAS-Xu).
fn load_nnet(model: &Path) -> Result<(VerifiableNetwork, Vec<usize>, usize)> {
    use ny_onnx::nnet::load_nnet;

    let nnet = load_nnet(model)?;
    info!(
        "Loaded NNet: {} layers, {} inputs, {} outputs, {} params",
        nnet.num_layers(),
        nnet.input_size(),
        nnet.output_size(),
        nnet.param_count()
    );

    let prop_net = nnet.to_prop_network()?;
    let input_shape = vec![1, nnet.input_size()];
    let output_dim = nnet.output_size();

    info!(
        "Converted to propagate network: {} layers",
        prop_net.layers().len()
    );
    Ok((
        VerifiableNetwork::Sequential(Box::new(prop_net)),
        input_shape,
        output_dim,
    ))
}

/// Load a native format model (PyTorch, SafeTensors, GGUF, CoreML).
fn load_native(
    model: &Path,
    conservative_layernorm: bool,
    layernorm_mode: LayerNormModeArg,
    layernorm_norm_mode: LayerNormNormModeArg,
    effective_method: PropagationMethod,
    json: bool,
) -> Result<(VerifiableNetwork, Vec<usize>, usize)> {
    use ny_onnx::native::NativeModel;

    let native_model = NativeModel::load(model)?;
    let network = &native_model.network;
    info!(
        "Loaded native model: {} ({:?}, {} params)",
        network.name, native_model.config.architecture, network.param_count
    );

    let mut graph_net = native_model.to_graph_network()?;

    // Apply LayerNorm configuration
    let crown_mode = layernorm_mode.into();
    let norm_mode = layernorm_norm_mode.into();
    configure_layernorm(
        &mut graph_net,
        conservative_layernorm,
        crown_mode,
        norm_mode,
        effective_method,
        json,
    );

    // Get input/output shapes from network spec
    // For verification, strip the batch dimension (first dim if dynamic)
    // Propagation layers expect unbatched input like [channels, length]
    // Dynamic dims (PyTorch -1, TensorFlow 0) default to 16 for profiling
    let input_shape: Vec<usize> = network
        .inputs
        .first()
        .map(|i| {
            let full_shape = ny_onnx::resolve_dynamic_shape(&i.shape, 16);
            // If first dimension is batch (dynamic in spec), skip it
            if i.shape.first().is_some_and(|&d| d <= 0) && full_shape.len() > 2 {
                full_shape[1..].to_vec()
            } else {
                full_shape
            }
        })
        .unwrap_or_else(|| vec![native_model.config.hidden_dim]);

    // SAFETY(#2983): Dynamic dims mapped to 16, so all factors are positive.
    let output_dim = network
        .outputs
        .first()
        .map(|o| {
            ny_onnx::resolve_dynamic_shape(&o.shape, 16)
                .iter()
                .product::<usize>()
                .max(1)
        })
        .unwrap_or(native_model.config.hidden_dim);

    info!(
        "Converted to graph network: {} nodes",
        graph_net.num_nodes()
    );
    Ok((
        VerifiableNetwork::Graph(Box::new(graph_net)),
        input_shape,
        output_dim,
    ))
}

/// Load an ONNX format model.
fn load_onnx_model(
    model: &Path,
    peel_off_last_softmax_layer: bool,
    property: Option<&Path>,
    json: bool,
) -> Result<(
    VerifiableNetwork,
    Vec<usize>,
    usize,
    Option<ny_onnx::vnnlib::VnnLibSpec>,
)> {
    use ny_onnx::vnnlib::load_vnnlib;
    use ny_onnx::{load_onnx_with_config, OnnxLoadConfig};

    // Crash-isolate ORT shape inference (see `cli_shape_infer_backend`).
    let load_config = OnnxLoadConfig::default()
        .with_shape_infer_backend(crate::commands::cli_shape_infer_backend());
    let mut onnx_model = load_onnx_with_config(model, &load_config)?;
    let onnx_network = &onnx_model.network;
    info!(
        "Loaded network: {} ({} layers)",
        onnx_network.name,
        onnx_network.layers.len()
    );

    // Use batch size 1 for dynamic dimensions (VNNLIB properties are single-instance)
    let input_shape: Vec<usize> = onnx_network
        .inputs
        .first()
        .map(|i| ny_onnx::resolve_dynamic_shape(&i.shape, 1))
        .unwrap_or_else(|| vec![100]);

    // SAFETY(#2983): Dynamic dims resolved, so all factors are positive.
    let output_dim = onnx_network
        .outputs
        .first()
        .map(|o| {
            ny_onnx::resolve_dynamic_shape(&o.shape, 1)
                .iter()
                .product::<usize>()
                .max(1)
        })
        .unwrap_or(10);

    let mut preloaded_vnnlib = None;
    if let Some(prop_path) = property {
        let mut vnnlib = load_vnnlib(prop_path)?;
        if peel_off_last_softmax_layer {
            let report = ny_onnx::peel_off_last_softmax_layer(&mut onnx_model, &mut vnnlib);
            if report.peeled && !json {
                eprintln!(
                    "Peeled off terminal {:?} layer using VNN-LIB constraints.",
                    report.layer_type
                );
            }
        } else {
            // #cgan-sigmoid-peel: default-ON auto-peel for the exactly-
            // invertible case (terminal Sigmoid + all-constant thresholds,
            // e.g. the cgan upsample band specs). NY_SIGMOID_PEEL=0 disables.
            let report = ny_onnx::peel_off_terminal_sigmoid_auto(&mut onnx_model, &mut vnnlib);
            if report.peeled && !json {
                eprintln!("Auto-peeled terminal Sigmoid using VNN-LIB constant thresholds.");
            }
        }
        preloaded_vnnlib = Some(vnnlib);
    }

    // Convert ONNX network to graph network (supports DAG structures like ViT)
    let graph_net = onnx_model.to_graph_network()?;

    info!(
        "Converted to graph network: {} nodes",
        graph_net.num_nodes()
    );
    Ok((
        VerifiableNetwork::Graph(Box::new(graph_net)),
        input_shape,
        output_dim,
        preloaded_vnnlib,
    ))
}

/// Apply heuristic and sound mode flags to the loaded network.
pub(crate) fn apply_soundness_modes(
    network: &mut VerifiableNetwork,
    allow_heuristic_logsoftmax: bool,
    allow_heuristic_softmax: bool,
    require_sound: bool,
    json: bool,
) {
    if allow_heuristic_logsoftmax {
        let modified = network.set_logsoftmax_sound_mode(false);
        if modified > 0 && !json {
            eprintln!(
                "LogSoftmax CROWN using heuristic sampling for {} nodes (not provably sound).",
                modified
            );
        }
    }

    if allow_heuristic_softmax {
        let modified_softmax = network.set_softmax_sound_mode(false);
        if modified_softmax > 0 && !json {
            eprintln!(
                "Softmax CROWN using heuristic sampling for {} nodes (not provably sound).",
                modified_softmax
            );
        }

        let modified_causal = network.set_causal_softmax_sound_mode(false);
        if modified_causal > 0 && !json {
            eprintln!(
                "CausalSoftmax CROWN using heuristic sampling for {} nodes (not provably sound).",
                modified_causal
            );
        }
    }

    if require_sound {
        let modified_gelu = network.set_gelu_sound_mode(true);
        if modified_gelu > 0 && !json {
            eprintln!(
                "GELU CROWN using sound relaxation for {} nodes (tanh/erf precomputed tangents).",
                modified_gelu
            );
        }

        let modified_logsoftmax = network.set_logsoftmax_sound_mode(true);
        if modified_logsoftmax > 0 && !json {
            eprintln!(
                "LogSoftmax CROWN using sound constant bounds for {} nodes (sampling disabled).",
                modified_logsoftmax
            );
        }

        let modified_softmax = network.set_softmax_sound_mode(true);
        if modified_softmax > 0 && !json {
            eprintln!(
                "Softmax CROWN using sound constant bounds for {} nodes (sampling disabled).",
                modified_softmax
            );
        }

        let modified_causal = network.set_causal_softmax_sound_mode(true);
        if modified_causal > 0 && !json {
            eprintln!(
                "CausalSoftmax CROWN using sound constant bounds for {} nodes (sampling disabled).",
                modified_causal
            );
        }

        let modified_sin = network.set_sin_sound_mode(true);
        if modified_sin > 0 && !json {
            eprintln!(
                "Sin CROWN using sound constant bounds for {} nodes (heuristic relaxations disabled).",
                modified_sin
            );
        }

        let modified_cos = network.set_cos_sound_mode(true);
        if modified_cos > 0 && !json {
            eprintln!(
                "Cos CROWN using sound constant bounds for {} nodes (heuristic relaxations disabled).",
                modified_cos
            );
        }
    }
}

/// Validate `--layer-by-layer` / `--block-wise` only work with native models.
pub(crate) fn validate_mode_model_compat(
    layer_by_layer: bool,
    use_block_wise: bool,
    use_native: bool,
) -> Result<()> {
    if (layer_by_layer || use_block_wise) && !use_native {
        let message = if layer_by_layer {
            "Layer-by-layer mode is only supported for native models (GraphNetwork). \
            Hint: use --native flag with PyTorch/SafeTensors/GGUF models."
        } else {
            "Block-wise mode is only supported for native models (GraphNetwork). \
            Hint: use --native flag with PyTorch/SafeTensors/GGUF models."
        };
        return Err(JsonCliError::new("unsupported_model_format", message).into());
    }
    Ok(())
}
