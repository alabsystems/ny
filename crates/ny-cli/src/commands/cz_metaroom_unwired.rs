// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gated, unwired bridge from an exact-decimal VNN-LIB box through the first
//! normalized ONNX `Conv2d` in a constrained-zonotope domain.
//!
//! This module is deliberately not called by a command or verdict path.  It
//! closes the type/provenance seam between three independently qualified
//! pieces while retaining a fail-closed boundary:
//!
//! 1. [`ny_onnx::vnnlib::CertifiedInputBox`] supplies outward binary64 input
//!    endpoints plus exact-rational point hints.
//! 2. [`ny_mip::ConstrainedZonotope64::from_certified_bounds`] decomposes the
//!    complete enclosure without trusting those hints to remove width.
//! 3. [`ny_mip::constrained_zonotope_conv2d_unwired`] propagates the normalized
//!    binary32 convolution as exact dyadic binary64 parameters and charges all
//!    contraction error to the output remainder.
//!
//! The ONNX model and propagation graph are both required.  The bridge first
//! requires immutable raw-FLOAT32-initializer provenance, then checks that the
//! direct-input Conv2d name, shape, attributes, kernel bits, and bias bits agree
//! before doing any abstract propagation.  It supports only a single static
//! `[1,C,H,W]` float32 input and a single direct Conv2d consumer; all broader
//! graph surfaces reject with a typed error.

use ndarray::{Array4, Ix1, Ix4};
use ny_core::LayerType;
use ny_mip::{
    constrained_zonotope_conv2d_unwired, transform_relu_projected_constraints_unwired,
    transform_relu_unwired, ConstrainedZonotope64, ConstrainedZonotope64Error,
    ConstrainedZonotopeConv2dError, ConstrainedZonotopeConv2dLimits, ConstrainedZonotopeConv2dPlan,
    ConstrainedZonotopeConv2dSpec, ReluTransformError, ReluTransformLimits,
};
use ny_onnx::vnnlib::CertifiedInputBox;
use ny_onnx::{AttributeValue, DataType, LayerSpec, OnnxModel};
use ny_propagate::{GraphNetwork, Layer, NETWORK_INPUT};

/// Explicit resource firewall for the unwired certified Conv2d stem.
///
/// There is intentionally no `Default`: every experimental caller must price
/// the input representation, model/graph scan, parameter promotion, and the
/// complete outward convolution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CertifiedCzStemLimits {
    /// Maximum number of loaded model layers and graph nodes inspected.
    pub max_graph_nodes: usize,
    /// Maximum total input-edge references in either the model or graph.
    pub max_graph_edges: usize,
    /// Maximum scalar comparisons required to seal model/graph topology.
    /// String bytes, shape/attribute elements, and container entries each
    /// consume this budget before the private topology snapshot is compared.
    pub max_topology_work_items: usize,
    /// Maximum number of flattened input values.
    pub max_input_values: usize,
    /// Maximum number of independent input alpha symbols.
    pub max_input_alpha_dim: usize,
    /// Maximum number of input sparse-generator nonzeros.
    pub max_input_generator_nonzeros: usize,
    /// Maximum stored binary64 scalars in the input center, generators, and
    /// box remainder (the initializer creates no predicate rows).
    pub max_input_stored_f64: usize,
    /// Maximum promoted kernel plus materialized bias elements.
    pub max_parameter_elements: usize,
    /// Complete limits enforced again by the outward Conv2d primitive.
    pub conv: ConstrainedZonotopeConv2dLimits,
}

/// Checked accounting for the certified input-domain construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedCzInputPlan {
    /// Flattened value dimension.
    pub value_dim: usize,
    /// Exact-rational point hints from the VNN-LIB source.
    pub declared_point_count: usize,
    /// Independent alpha symbols retained by the decomposition.
    pub alpha_dim: usize,
    /// Actual nonzero sparse generator coefficients.
    pub generator_nonzeros: usize,
    /// Stored binary64 scalars in center, generators, constraints, RHS, and
    /// box remainder.  The initial box has no constraints or RHS.
    pub stored_f64: usize,
}

/// Checked accounting for a completed direct-input Conv2d bridge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedCzStemPlan {
    /// Static unbatched NCHW input shape `[C,H,W]` proven against the model.
    pub input_shape: [usize; 3],
    /// Name shared by the raw model layer and normalized graph node.
    pub conv_node: String,
    /// Kernel plus materialized bias elements promoted exactly to binary64.
    pub parameter_elements: usize,
    /// Input-domain accounting.
    pub input: CertifiedCzInputPlan,
    /// Outward sparse convolution accounting.
    pub conv: ConstrainedZonotopeConv2dPlan,
}

/// Explicit resource firewall for a certified direct Conv2d followed by its
/// unique normalized ReLU successor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CertifiedCzReluStemLimits {
    /// Complete limits for the certified input and outward convolution.
    pub stem: CertifiedCzStemLimits,
    /// Complete limits for proof-safe DeepZ ReLU propagation.
    pub relu: ReluTransformLimits,
}

/// Checked accounting for a completed direct Conv2d-to-ReLU bridge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedCzReluStemPlan {
    /// Certified input and outward convolution accounting.
    pub stem: CertifiedCzStemPlan,
    /// Name shared by the raw model ReLU and normalized graph ReLU.
    pub relu_node: String,
    /// Alpha symbols entering the ReLU.
    pub input_alpha_dim: usize,
    /// Alpha symbols after adding one DeepZ symbol per unstable coordinate.
    pub output_alpha_dim: usize,
    /// Number of unstable coordinates and newly introduced symbols.
    pub unstable_count: usize,
    /// Actual sparse generator nonzeros after the ReLU transform.
    pub output_generator_nonzeros: usize,
    /// Coordinates carrying a nonzero outward box remainder.
    pub nonzero_remainder_count: usize,
}

/// Fail-closed errors for the unwired bridge.
#[derive(Debug, thiserror::Error)]
pub enum CertifiedCzStemError {
    /// Model/graph topology or parameter provenance is outside the qualified
    /// direct-Conv surface.
    #[error("unsupported certified Conv2d stem: {message}")]
    Unsupported {
        /// Concrete rejected precondition.
        message: String,
    },

    /// A shape observed at two proof boundaries disagreed.
    #[error("shape mismatch for {field}: expected {expected:?}, got {got:?}")]
    Shape {
        /// Boundary being checked.
        field: &'static str,
        /// Required shape.
        expected: Vec<usize>,
        /// Supplied shape.
        got: Vec<usize>,
    },

    /// A dimension or resource calculation overflowed `usize`.
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow {
        /// Failed calculation.
        operation: &'static str,
    },

    /// An explicit caller-selected cap was exceeded.
    #[error("resource limit exceeded for {resource}: required {required}, limit {limit}")]
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Required count.
        required: usize,
        /// Caller-selected maximum.
        limit: usize,
    },

    /// A bounded allocation failed.
    #[error("unable to reserve storage for {resource}")]
    AllocationFailure {
        /// Requested container.
        resource: &'static str,
    },

    /// A graph parameter was NaN or infinite.
    #[error("{field}[{index}] must be finite")]
    NonFiniteParameter {
        /// Kernel or bias.
        field: &'static str,
        /// Row-major flattened index.
        index: usize,
    },

    /// The accepted input-box constructor rejected its inputs.
    #[error(transparent)]
    Domain(#[from] ConstrainedZonotope64Error),

    /// The accepted outward sparse Conv2d rejected or could not enclose the
    /// transform.
    #[error(transparent)]
    Conv(#[from] ConstrainedZonotopeConv2dError),

    /// A static ONNX dimension did not fit this host's `usize`.
    #[error("model input dimension {dimension} cannot be represented as usize")]
    ModelDimension {
        /// Positive dimension that failed conversion.
        dimension: i64,
    },
}

/// Fail-closed errors for the unwired Conv2d-to-ReLU bridge.
#[derive(Debug, thiserror::Error)]
pub enum CertifiedCzReluStemError {
    /// The accepted certified Conv2d stem rejected the model or resources.
    #[error(transparent)]
    Stem(#[from] CertifiedCzStemError),

    /// The accepted proof-safe ReLU transform rejected the domain or limits.
    #[error(transparent)]
    Relu(#[from] ReluTransformError),

    /// The raw model and normalized graph do not expose one matching direct
    /// Conv2d-to-ReLU chain.
    #[error("unsupported certified Conv2d-to-ReLU stem: {message}")]
    Unsupported {
        /// Concrete rejected precondition.
        message: String,
    },

    /// Checked post-transform resource accounting overflowed `usize`.
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow {
        /// Failed calculation.
        operation: &'static str,
    },
}

/// Build a resource-capped constrained zonotope from a certified VNN-LIB box.
///
/// This is proof-safe only under the input contract of `CertifiedInputBox`:
/// its endpoints already enclose the exact source decimals.  Point bits remain
/// decomposition hints and never discard supplied endpoint width.
pub fn certified_input_box_to_cz_unwired(
    input_box: &CertifiedInputBox,
    limits: CertifiedCzStemLimits,
) -> Result<(ConstrainedZonotope64, CertifiedCzInputPlan), CertifiedCzStemError> {
    let value_dim = input_box.len();
    check_limit("input value count", value_dim, limits.max_input_values)?;

    let declared_point_count = input_box
        .declared_point()
        .iter()
        .filter(|&&point| point)
        .count();
    let alpha_dim = value_dim.checked_sub(declared_point_count).ok_or(
        CertifiedCzStemError::ResourceOverflow {
            operation: "input alpha dimension",
        },
    )?;
    check_limit(
        "input alpha dimension",
        alpha_dim,
        limits.max_input_alpha_dim,
    )?;

    // An axis-box initializer emits at most one nonzero per unmarked symbol.
    // Enforce this conservative upper bound before it allocates generator
    // columns; zero-width unmarked dimensions can only reduce the final count.
    check_limit(
        "input generator nonzeros",
        alpha_dim,
        limits.max_input_generator_nonzeros,
    )?;
    let maximum_stored_f64 = value_dim
        .checked_mul(2)
        .and_then(|count| count.checked_add(alpha_dim))
        .ok_or(CertifiedCzStemError::ResourceOverflow {
            operation: "maximum input stored f64 scalars",
        })?;
    check_limit(
        "input stored f64 scalars",
        maximum_stored_f64,
        limits.max_input_stored_f64,
    )?;

    let domain = ConstrainedZonotope64::from_certified_bounds(
        input_box.lower(),
        input_box.upper(),
        input_box.declared_point(),
    )?;
    let generator_nonzeros = domain
        .generators()
        .iter()
        .try_fold(0_usize, |count, generator| {
            count
                .checked_add(generator.nnz())
                .ok_or(CertifiedCzStemError::ResourceOverflow {
                    operation: "actual input generator nonzeros",
                })
        })?;
    check_limit(
        "input generator nonzeros",
        generator_nonzeros,
        limits.max_input_generator_nonzeros,
    )?;
    let stored_f64 = domain
        .center()
        .len()
        .checked_add(generator_nonzeros)
        .and_then(|count| count.checked_add(domain.constraints().len()))
        .and_then(|count| count.checked_add(domain.rhs().len()))
        .and_then(|count| count.checked_add(domain.box_remainder().len()))
        .ok_or(CertifiedCzStemError::ResourceOverflow {
            operation: "actual input stored f64 scalars",
        })?;
    check_limit(
        "input stored f64 scalars",
        stored_f64,
        limits.max_input_stored_f64,
    )?;

    let plan = CertifiedCzInputPlan {
        value_dim,
        declared_point_count,
        alpha_dim: domain.alpha_dim(),
        generator_nonzeros,
        stored_f64,
    };
    Ok((domain, plan))
}

/// Propagate a certified VNN-LIB box through the single direct-input Conv2d
/// shared by a loaded ONNX model and its normalized propagation graph.
///
/// This remains unwired: no CLI command calls it, and its output cannot reach
/// a verdict.  Supplying both representations is intentional.  Raw
/// model-layer metadata establishes static NCHW and parameter provenance;
/// graph equality establishes that the promoted dyadics are the parameters NY
/// would execute.
pub fn propagate_direct_onnx_conv2d_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input_box: &CertifiedInputBox,
    limits: CertifiedCzStemLimits,
) -> Result<(ConstrainedZonotope64, CertifiedCzStemPlan), CertifiedCzStemError> {
    check_limit(
        "model layer count",
        model.network.layers.len(),
        limits.max_graph_nodes,
    )?;
    check_limit(
        "graph node count",
        graph.num_nodes(),
        limits.max_graph_nodes,
    )?;
    check_bounded_sum(
        model.network.layers.iter().map(|layer| layer.inputs.len()),
        "model edge count",
        limits.max_graph_edges,
    )?;
    let mut topology_work_items = 0_usize;
    check_model_topology_work(
        model,
        &mut topology_work_items,
        limits.max_topology_work_items,
    )?;
    check_limit(
        "graph insertion-order count",
        graph.node_names().len(),
        limits.max_graph_nodes,
    )?;
    consume_topology_work(
        &mut topology_work_items,
        graph.node_names().len(),
        limits.max_topology_work_items,
    )?;
    let mut graph_edge_count = 0_usize;
    for name in graph.node_names() {
        consume_topology_work(
            &mut topology_work_items,
            name.len(),
            limits.max_topology_work_items,
        )?;
        let node = graph
            .node(name)
            .ok_or_else(|| CertifiedCzStemError::Unsupported {
                message: format!("graph insertion order references missing node '{name}'"),
            })?;
        graph_edge_count = graph_edge_count.checked_add(node.inputs().len()).ok_or(
            CertifiedCzStemError::ResourceOverflow {
                operation: "graph edge count",
            },
        )?;
        check_limit("graph edge count", graph_edge_count, limits.max_graph_edges)?;
        consume_topology_work(
            &mut topology_work_items,
            node.inputs().len(),
            limits.max_topology_work_items,
        )?;
        for input in node.inputs() {
            consume_topology_work(
                &mut topology_work_items,
                input.len(),
                limits.max_topology_work_items,
            )?;
        }
    }
    require_original_network_topology(model)?;

    let (input_name, input_shape) = static_nchw_input(model)?;
    let input_value_count = checked_product(&input_shape, "model input value count")?;
    if input_box.len() != input_value_count {
        return Err(CertifiedCzStemError::Shape {
            field: "certified input box",
            expected: vec![input_value_count],
            got: vec![input_box.len()],
        });
    }

    let model_conv = direct_model_conv(model, input_name)?;
    let (node_name, graph_conv) = direct_graph_conv(graph)?;
    if node_name != model_conv.name {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "model direct Conv2d '{}' disagrees with graph direct Conv2d '{node_name}'",
                model_conv.name
            ),
        });
    }
    if graph_conv.input_shape != Some((input_shape[1], input_shape[2])) {
        return Err(CertifiedCzStemError::Shape {
            field: "normalized Conv2d input spatial shape",
            expected: vec![input_shape[1], input_shape[2]],
            got: graph_conv
                .input_shape
                .map_or_else(Vec::new, |(height, width)| vec![height, width]),
        });
    }
    let raw = raw_conv_parameters(model, model_conv)?;
    let raw_bias_elements = raw
        .bias
        .as_ref()
        .map_or(raw.kernel.shape()[0], ndarray::ArrayBase::len);
    let raw_parameter_elements = raw.kernel.len().checked_add(raw_bias_elements).ok_or(
        CertifiedCzStemError::ResourceOverflow {
            operation: "raw Conv2d parameter elements",
        },
    )?;
    check_limit(
        "promoted Conv2d parameter elements",
        raw_parameter_elements,
        limits.max_parameter_elements,
    )?;
    validate_raw_conv_shape(&raw)?;
    validate_normalized_graph_conv_shape(graph_conv, input_shape[0])?;
    require_original_float32_parameters(model, &raw)?;
    ensure_graph_conv_matches_raw(graph_conv, &raw)?;
    let (weights, bias, parameter_elements) = promote_parameters(graph_conv, limits)?;
    debug_assert_eq!(parameter_elements, raw_parameter_elements);
    let (input_domain, input_plan) = certified_input_box_to_cz_unwired(input_box, limits)?;

    let spec = ConstrainedZonotopeConv2dSpec {
        stride: [graph_conv.stride.0, graph_conv.stride.1],
        padding: [
            graph_conv.padding.0,
            graph_conv.padding.1,
            graph_conv.padding.0,
            graph_conv.padding.1,
        ],
        dilation: [graph_conv.dilation.0, graph_conv.dilation.1],
        groups: graph_conv.groups,
    };
    let (output, conv_plan) = constrained_zonotope_conv2d_unwired(
        &input_domain,
        input_shape,
        weights.view(),
        &bias,
        spec,
        limits.conv,
    )?;
    let plan = CertifiedCzStemPlan {
        input_shape,
        conv_node: node_name.to_string(),
        parameter_elements,
        input: input_plan,
        conv: conv_plan,
    };
    Ok((output, plan))
}

fn require_original_network_topology(model: &OnnxModel) -> Result<(), CertifiedCzStemError> {
    match model.original_network_topology_matches_current() {
        Some(true) => Ok(()),
        None => Err(CertifiedCzStemError::Unsupported {
            message: "model was not loaded with private finalized-network provenance".to_string(),
        }),
        Some(false) => Err(CertifiedCzStemError::Unsupported {
            message: "public model network no longer matches private finalized-network provenance"
                .to_string(),
        }),
    }
}

fn check_model_topology_work(
    model: &OnnxModel,
    work_items: &mut usize,
    limit: usize,
) -> Result<(), CertifiedCzStemError> {
    let network = &model.network;
    consume_topology_work(work_items, network.name.len(), limit)?;
    consume_topology_work(work_items, network.inputs.len(), limit)?;
    for input in &network.inputs {
        consume_topology_work(work_items, input.name.len(), limit)?;
        consume_topology_work(work_items, input.shape.len(), limit)?;
        consume_topology_work(work_items, 1, limit)?;
    }
    consume_topology_work(work_items, network.outputs.len(), limit)?;
    for output in &network.outputs {
        consume_topology_work(work_items, output.name.len(), limit)?;
        consume_topology_work(work_items, output.shape.len(), limit)?;
        consume_topology_work(work_items, 1, limit)?;
    }
    consume_topology_work(work_items, network.layers.len(), limit)?;
    for layer in &network.layers {
        consume_topology_work(work_items, layer.name.len(), limit)?;
        consume_topology_work(work_items, 1, limit)?;

        consume_topology_work(work_items, layer.inputs.len(), limit)?;
        for input in &layer.inputs {
            consume_topology_work(work_items, input.len(), limit)?;
        }
        consume_topology_work(work_items, layer.outputs.len(), limit)?;
        for output in &layer.outputs {
            consume_topology_work(work_items, output.len(), limit)?;
        }

        // `HashMap::iter` visits raw buckets, so its work is proportional to
        // capacity rather than entry count. Callers can reserve through this
        // public map without inserting; price the bucket scan before it starts.
        consume_topology_work(work_items, layer.attributes.capacity(), limit)?;
        for (name, value) in &layer.attributes {
            consume_topology_work(work_items, name.len(), limit)?;
            consume_topology_work(work_items, 1, limit)?;
            let payload_len = match value {
                AttributeValue::Float(_) | AttributeValue::Int(_) => 0,
                AttributeValue::String(value) => value.len(),
                AttributeValue::Floats(values) => values.len(),
                AttributeValue::Ints(values) => values.len(),
            };
            consume_topology_work(work_items, payload_len, limit)?;
        }

        if let Some(weights) = &layer.weights {
            consume_topology_work(work_items, 1, limit)?;
            consume_topology_work(work_items, weights.name.len(), limit)?;
            consume_topology_work(work_items, weights.shape.len(), limit)?;
            consume_topology_work(work_items, 1, limit)?;
        }
    }
    consume_topology_work(work_items, 1, limit)
}

fn consume_topology_work(
    work_items: &mut usize,
    additional: usize,
    limit: usize,
) -> Result<(), CertifiedCzStemError> {
    *work_items =
        work_items
            .checked_add(additional)
            .ok_or(CertifiedCzStemError::ResourceOverflow {
                operation: "model/graph topology work items",
            })?;
    check_limit("model/graph topology work items", *work_items, limit)
}

/// Propagate a certified VNN-LIB box through the unique direct-input Conv2d
/// and its unique elementwise ReLU successor shared by the loaded model and
/// normalized propagation graph.
///
/// This remains unreachable from commands and verifier verdicts.  The GPU is
/// not involved: both transforms use exact-dyadic reasoning with all binary64
/// representation error charged to the outward box remainder.
///
/// # Errors
///
/// Fails closed when either representation is not a single matching
/// `Conv2d -> ReLU` chain, when a checked resource count overflows, or when
/// either accepted primitive rejects its proof/resource preconditions.
pub fn propagate_direct_onnx_conv2d_relu_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input_box: &CertifiedInputBox,
    limits: CertifiedCzReluStemLimits,
) -> Result<(ConstrainedZonotope64, CertifiedCzReluStemPlan), CertifiedCzReluStemError> {
    let (conv_output, stem) =
        propagate_direct_onnx_conv2d_unwired(model, graph, input_box, limits.stem)?;
    // The stem call enforces graph-node/edge caps before the successor scans.
    let relu_node = matching_direct_relu_successor(model, graph)?;
    let input_alpha_dim = conv_output.alpha_dim();
    let output = transform_relu_unwired(&conv_output, limits.relu)?;
    let output_alpha_dim = output.alpha_dim();
    let unstable_count = output_alpha_dim.checked_sub(input_alpha_dim).ok_or(
        CertifiedCzReluStemError::ResourceOverflow {
            operation: "ReLU unstable symbol count",
        },
    )?;
    let output_generator_nonzeros =
        output
            .generators()
            .iter()
            .try_fold(0_usize, |count, generator| {
                count.checked_add(generator.nnz()).ok_or(
                    CertifiedCzReluStemError::ResourceOverflow {
                        operation: "ReLU output generator nonzeros",
                    },
                )
            })?;
    let nonzero_remainder_count = output
        .box_remainder()
        .iter()
        .filter(|&&value| value != 0.0)
        .count();
    let plan = CertifiedCzReluStemPlan {
        stem,
        relu_node,
        input_alpha_dim,
        output_alpha_dim,
        unstable_count,
        output_generator_nonzeros,
        nonzero_remainder_count,
    };
    Ok((output, plan))
}

/// Propagate the sealed direct Conv2d-to-ReLU stem and opt in to projected
/// `y >= 0` and `y >= x` alpha predicates for every unstable coordinate.
///
/// This is a separate unwired entry point so callers cannot accidentally
/// change the historical predicate-preserving transform.  It retains the
/// same model/graph provenance and topology seals as
/// [`propagate_direct_onnx_conv2d_relu_unwired`].  The projected rows eliminate
/// both input and output box remainders into their right-hand sides and charge
/// coefficient-rounding residuals outward; every witness retained by the base
/// DeepZ transform therefore remains feasible.
///
/// # Errors
///
/// Fails closed under the same conditions as
/// [`propagate_direct_onnx_conv2d_relu_unwired`], including caller-selected
/// predicate row and dense-element caps enforced before output allocation.
pub fn propagate_direct_onnx_conv2d_relu_projected_constraints_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input_box: &CertifiedInputBox,
    limits: CertifiedCzReluStemLimits,
) -> Result<(ConstrainedZonotope64, CertifiedCzReluStemPlan), CertifiedCzReluStemError> {
    let (conv_output, stem) =
        propagate_direct_onnx_conv2d_unwired(model, graph, input_box, limits.stem)?;
    // The stem call enforces graph-node/edge caps before the successor scans.
    let relu_node = matching_direct_relu_successor(model, graph)?;
    let input_alpha_dim = conv_output.alpha_dim();
    let output = transform_relu_projected_constraints_unwired(&conv_output, limits.relu)?;
    let output_alpha_dim = output.alpha_dim();
    let unstable_count = output_alpha_dim.checked_sub(input_alpha_dim).ok_or(
        CertifiedCzReluStemError::ResourceOverflow {
            operation: "projected ReLU unstable symbol count",
        },
    )?;
    let output_generator_nonzeros =
        output
            .generators()
            .iter()
            .try_fold(0_usize, |count, generator| {
                count.checked_add(generator.nnz()).ok_or(
                    CertifiedCzReluStemError::ResourceOverflow {
                        operation: "projected ReLU output generator nonzeros",
                    },
                )
            })?;
    let nonzero_remainder_count = output
        .box_remainder()
        .iter()
        .filter(|&&value| value != 0.0)
        .count();
    let plan = CertifiedCzReluStemPlan {
        stem,
        relu_node,
        input_alpha_dim,
        output_alpha_dim,
        unstable_count,
        output_generator_nonzeros,
        nonzero_remainder_count,
    };
    Ok((output, plan))
}

fn matching_direct_relu_successor(
    model: &OnnxModel,
    graph: &GraphNetwork,
) -> Result<String, CertifiedCzReluStemError> {
    let (input_name, _) = static_nchw_input(model)?;
    let model_conv = direct_model_conv(model, input_name)?;
    if model_conv.outputs.len() != 1 || model_conv.outputs[0].is_empty() {
        return Err(CertifiedCzReluStemError::Unsupported {
            message: format!(
                "model Conv2d '{}' must have one named output, got {:?}",
                model_conv.name, model_conv.outputs
            ),
        });
    }
    let conv_output = &model_conv.outputs[0];
    let mut model_consumers = model
        .network
        .layers
        .iter()
        .filter(|layer| layer.inputs.iter().any(|input| input == conv_output));
    let model_relu =
        model_consumers
            .next()
            .ok_or_else(|| CertifiedCzReluStemError::Unsupported {
                message: format!(
                    "model Conv2d '{}' output '{}' has no direct consumer",
                    model_conv.name, conv_output
                ),
            })?;
    if let Some(other) = model_consumers.next() {
        return Err(CertifiedCzReluStemError::Unsupported {
            message: format!(
                "model Conv2d '{}' output '{}' has multiple consumers ('{}' and '{}')",
                model_conv.name, conv_output, model_relu.name, other.name
            ),
        });
    }
    if model_relu.layer_type != LayerType::ReLU
        || model_relu.inputs.len() != 1
        || model_relu.inputs.first() != Some(conv_output)
        || model_relu.outputs.len() != 1
        || model_relu.outputs[0].is_empty()
        || model_relu.weights.is_some()
        || !model_relu.attributes.is_empty()
    {
        return Err(CertifiedCzReluStemError::Unsupported {
            message: format!(
                "model Conv2d '{}' must feed one plain unary ReLU; got {} '{}' with {} inputs, {} outputs, weights={}, attrs={}",
                model_conv.name,
                model_relu.layer_type,
                model_relu.name,
                model_relu.inputs.len(),
                model_relu.outputs.len(),
                model_relu.weights.is_some(),
                model_relu.attributes.len()
            ),
        });
    }

    let (graph_conv_name, _) = direct_graph_conv(graph)?;
    let mut graph_consumers = graph.node_names().iter().filter_map(|name| {
        graph
            .node(name)
            .filter(|node| node.inputs().iter().any(|input| input == graph_conv_name))
    });
    let graph_relu =
        graph_consumers
            .next()
            .ok_or_else(|| CertifiedCzReluStemError::Unsupported {
                message: format!("graph Conv2d '{graph_conv_name}' has no direct consumer"),
            })?;
    if let Some(other) = graph_consumers.next() {
        return Err(CertifiedCzReluStemError::Unsupported {
            message: format!(
                "graph Conv2d '{graph_conv_name}' has multiple consumers ('{}' and '{}')",
                graph_relu.name(),
                other.name()
            ),
        });
    }
    if graph_relu.inputs() != [graph_conv_name] || !matches!(graph_relu.layer(), Layer::ReLU(_)) {
        return Err(CertifiedCzReluStemError::Unsupported {
            message: format!(
                "graph Conv2d '{graph_conv_name}' must feed one unary ReLU; got {} '{}' with inputs {:?}",
                graph_relu.layer().layer_type(),
                graph_relu.name(),
                graph_relu.inputs()
            ),
        });
    }
    if graph_relu.name() != model_relu.name {
        return Err(CertifiedCzReluStemError::Unsupported {
            message: format!(
                "model ReLU '{}' disagrees with graph ReLU '{}'",
                model_relu.name,
                graph_relu.name()
            ),
        });
    }
    Ok(graph_relu.name().to_string())
}

fn static_nchw_input(model: &OnnxModel) -> Result<(&str, [usize; 3]), CertifiedCzStemError> {
    if model.network.inputs.len() != 1 {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "expected one model input, found {}",
                model.network.inputs.len()
            ),
        });
    }
    let input = &model.network.inputs[0];
    if input.dtype != DataType::Float32 {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "input '{}' has {:?} dtype; only Float32 is qualified",
                input.name, input.dtype
            ),
        });
    }
    if input.shape.len() != 4 {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "input '{}' must have rank 4 [1,C,H,W], got rank {}",
                input.name,
                input.shape.len()
            ),
        });
    }
    if input.shape[0] != 1 || input.shape[1..].contains(&0) {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "input '{}' must have static shape [1,C,H,W], got {:?}",
                input.name, input.shape
            ),
        });
    }
    if input.shape[1..].iter().any(|&dimension| dimension < 0) {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "input '{}' must have positive static shape [1,C,H,W], got {:?}",
                input.name, input.shape
            ),
        });
    }
    let shape = [
        usize::try_from(input.shape[1]).map_err(|_| CertifiedCzStemError::ModelDimension {
            dimension: input.shape[1],
        })?,
        usize::try_from(input.shape[2]).map_err(|_| CertifiedCzStemError::ModelDimension {
            dimension: input.shape[2],
        })?,
        usize::try_from(input.shape[3]).map_err(|_| CertifiedCzStemError::ModelDimension {
            dimension: input.shape[3],
        })?,
    ];
    if shape.contains(&0) {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!("input '{}' contains an empty dimension", input.name),
        });
    }
    Ok((&input.name, shape))
}

fn direct_model_conv<'a>(
    model: &'a OnnxModel,
    input_name: &str,
) -> Result<&'a LayerSpec, CertifiedCzStemError> {
    let mut direct = model
        .network
        .layers
        .iter()
        .filter(|layer| layer.inputs.iter().any(|name| name == input_name));
    let layer = direct
        .next()
        .ok_or_else(|| CertifiedCzStemError::Unsupported {
            message: format!("model input '{input_name}' has no direct consumer"),
        })?;
    if let Some(other) = direct.next() {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "model input '{input_name}' has multiple direct consumers ('{}' and '{}')",
                layer.name, other.name
            ),
        });
    }
    if layer.layer_type != LayerType::Conv2d
        || !(2..=3).contains(&layer.inputs.len())
        || layer.inputs.first().is_none_or(|name| name != input_name)
    {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "model input '{input_name}' must feed one 2/3-input Conv2d, got {} '{}' with {} inputs",
                layer.layer_type,
                layer.name,
                layer.inputs.len()
            ),
        });
    }
    Ok(layer)
}

fn direct_graph_conv(
    graph: &GraphNetwork,
) -> Result<(&str, &ny_propagate::layers::Conv2dLayer), CertifiedCzStemError> {
    let mut direct = graph.node_names().iter().filter_map(|name| {
        graph
            .node(name)
            .filter(|node| node.inputs().iter().any(|input| input == NETWORK_INPUT))
    });
    let node = direct
        .next()
        .ok_or_else(|| CertifiedCzStemError::Unsupported {
            message: "normalized graph has no direct network-input consumer".to_string(),
        })?;
    if let Some(other) = direct.next() {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "normalized graph has multiple direct input consumers ('{}' and '{}')",
                node.name(),
                other.name()
            ),
        });
    }
    if node.inputs() != [NETWORK_INPUT] {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "direct graph node '{}' has inputs {:?}, expected only {NETWORK_INPUT}",
                node.name(),
                node.inputs()
            ),
        });
    }
    let Layer::Conv2d(conv) = node.layer() else {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "direct graph node '{}' is {}, not Conv2d",
                node.name(),
                node.layer().layer_type()
            ),
        });
    };
    Ok((node.name(), conv))
}

struct RawConvParameters<'a> {
    kernel_name: &'a str,
    kernel: ndarray::ArrayView4<'a, f32>,
    bias_name: Option<&'a str>,
    bias: Option<ndarray::ArrayView1<'a, f32>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
}

fn raw_conv_parameters<'a>(
    model: &'a OnnxModel,
    layer: &'a LayerSpec,
) -> Result<RawConvParameters<'a>, CertifiedCzStemError> {
    let kernel_name = &layer.inputs[1];
    let kernel_raw =
        model
            .weights
            .get(kernel_name)
            .ok_or_else(|| CertifiedCzStemError::Unsupported {
                message: format!(
                    "direct Conv2d '{}' kernel '{}' is not a direct float initializer",
                    layer.name, kernel_name
                ),
            })?;
    if kernel_raw.ndim() != 4 {
        return Err(CertifiedCzStemError::Shape {
            field: "raw Conv2d kernel rank",
            expected: vec![4],
            got: vec![kernel_raw.ndim()],
        });
    }
    let kernel = kernel_raw
        .view()
        .into_dimensionality::<Ix4>()
        .map_err(|_| CertifiedCzStemError::Shape {
            field: "raw Conv2d kernel rank",
            expected: vec![4],
            got: vec![kernel_raw.ndim()],
        })?;
    let bias = if let Some(bias_name) = layer.inputs.get(2) {
        let bias_raw =
            model
                .weights
                .get(bias_name)
                .ok_or_else(|| CertifiedCzStemError::Unsupported {
                    message: format!(
                        "direct Conv2d '{}' bias '{}' is not a direct float initializer",
                        layer.name, bias_name
                    ),
                })?;
        if bias_raw.ndim() != 1 {
            return Err(CertifiedCzStemError::Shape {
                field: "raw Conv2d bias rank",
                expected: vec![1],
                got: vec![bias_raw.ndim()],
            });
        }
        Some(bias_raw.view().into_dimensionality::<Ix1>().map_err(|_| {
            CertifiedCzStemError::Shape {
                field: "raw Conv2d bias rank",
                expected: vec![1],
                got: vec![bias_raw.ndim()],
            }
        })?)
    } else {
        None
    };

    if let Some(AttributeValue::String(auto_pad)) = layer.attributes.get("auto_pad") {
        if auto_pad.len() > 16 {
            return Err(CertifiedCzStemError::Unsupported {
                message: format!("direct Conv2d '{}' auto_pad exceeds 16 bytes", layer.name),
            });
        }
        let normalized = auto_pad.trim().to_ascii_uppercase();
        if !normalized.is_empty() && normalized != "NOTSET" {
            return Err(CertifiedCzStemError::Unsupported {
                message: format!(
                    "direct Conv2d '{}' uses unsupported auto_pad={auto_pad}",
                    layer.name
                ),
            });
        }
    } else if layer.attributes.contains_key("auto_pad") {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!("direct Conv2d '{}' has non-string auto_pad", layer.name),
        });
    }

    let stride = parse_positive_pair(layer, "strides", [1, 1])?;
    let dilation = parse_positive_pair(layer, "dilations", [1, 1])?;
    let padding = parse_symmetric_padding(layer)?;
    let groups = parse_groups(layer)?;
    if let Some(kernel_shape) = layer.attributes.get("kernel_shape") {
        let AttributeValue::Ints(kernel_shape) = kernel_shape else {
            return Err(CertifiedCzStemError::Unsupported {
                message: format!(
                    "direct Conv2d '{}' has non-integer kernel_shape",
                    layer.name
                ),
            });
        };
        let actual = [kernel.shape()[2], kernel.shape()[3]];
        if kernel_shape.len() != 2 {
            return Err(CertifiedCzStemError::Unsupported {
                message: format!(
                    "direct Conv2d '{}' has kernel_shape length {}, expected 2",
                    layer.name,
                    kernel_shape.len()
                ),
            });
        }
        let parsed = [
            parse_nonnegative_usize(kernel_shape[0], layer, "kernel_shape")?,
            parse_nonnegative_usize(kernel_shape[1], layer, "kernel_shape")?,
        ];
        if parsed != actual {
            return Err(CertifiedCzStemError::Shape {
                field: "raw Conv2d kernel_shape attribute",
                expected: actual.to_vec(),
                got: parsed.to_vec(),
            });
        }
    }
    Ok(RawConvParameters {
        kernel_name,
        kernel,
        bias_name: layer.inputs.get(2).map(String::as_str),
        bias,
        stride: (stride[0], stride[1]),
        padding,
        dilation: (dilation[0], dilation[1]),
        groups,
    })
}

fn validate_raw_conv_shape(raw: &RawConvParameters<'_>) -> Result<(), CertifiedCzStemError> {
    let output_channels = raw.kernel.shape()[0];
    if let Some(bias) = raw.bias {
        if bias.len() != output_channels {
            return Err(CertifiedCzStemError::Shape {
                field: "raw Conv2d bias",
                expected: vec![output_channels],
                got: vec![bias.len()],
            });
        }
    }
    Ok(())
}

fn validate_normalized_graph_conv_shape(
    graph: &ny_propagate::layers::Conv2dLayer,
    expected_input_channels: usize,
) -> Result<(), CertifiedCzStemError> {
    let shape = graph.kernel.shape();
    if shape.len() != 4 {
        return Err(CertifiedCzStemError::Shape {
            field: "normalized Conv2d kernel rank",
            expected: vec![4],
            got: vec![shape.len()],
        });
    }
    if graph.groups == 0 {
        return Err(CertifiedCzStemError::Unsupported {
            message: "normalized Conv2d has zero groups".to_string(),
        });
    }
    let input_channels =
        shape[1]
            .checked_mul(graph.groups)
            .ok_or(CertifiedCzStemError::ResourceOverflow {
                operation: "normalized Conv2d input channels",
            })?;
    if input_channels != expected_input_channels {
        return Err(CertifiedCzStemError::Shape {
            field: "normalized Conv2d input channels",
            expected: vec![expected_input_channels],
            got: vec![input_channels],
        });
    }
    if !shape[0].is_multiple_of(graph.groups) {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "normalized Conv2d output channels {} are not divisible by groups {}",
                shape[0], graph.groups
            ),
        });
    }
    if let Some(bias) = &graph.bias {
        if bias.len() != shape[0] {
            return Err(CertifiedCzStemError::Shape {
                field: "normalized Conv2d bias",
                expected: vec![shape[0]],
                got: vec![bias.len()],
            });
        }
    }
    Ok(())
}

fn require_original_float32_parameters(
    model: &OnnxModel,
    raw: &RawConvParameters<'_>,
) -> Result<(), CertifiedCzStemError> {
    require_original_float32_initializer(model, raw.kernel_name, "kernel")?;
    if let Some(bias_name) = raw.bias_name {
        require_original_float32_initializer(model, bias_name, "bias")?;
    }
    Ok(())
}

fn require_original_float32_initializer(
    model: &OnnxModel,
    name: &str,
    field: &'static str,
) -> Result<(), CertifiedCzStemError> {
    match model.original_float32_initializer_matches_current(name) {
        Some(true) => Ok(()),
        None => Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "direct Conv2d {field} '{name}' has no private raw ONNX FLOAT initializer provenance"
            ),
        }),
        Some(false) => Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "direct Conv2d {field} '{name}' no longer matches its private raw ONNX FLOAT initializer provenance"
            ),
        }),
    }
}

fn parse_positive_pair(
    layer: &LayerSpec,
    name: &'static str,
    fallback: [usize; 2],
) -> Result<[usize; 2], CertifiedCzStemError> {
    let Some(value) = layer.attributes.get(name) else {
        return Ok(fallback);
    };
    let pair = match value {
        AttributeValue::Int(value) => {
            let value = parse_nonnegative_usize(*value, layer, name)?;
            [value, value]
        }
        AttributeValue::Ints(values) if values.is_empty() => return Ok(fallback),
        AttributeValue::Ints(values) if values.len() == 1 => {
            let value = parse_nonnegative_usize(values[0], layer, name)?;
            [value, value]
        }
        AttributeValue::Ints(values) if values.len() == 2 => [
            parse_nonnegative_usize(values[0], layer, name)?,
            parse_nonnegative_usize(values[1], layer, name)?,
        ],
        AttributeValue::Ints(values) => {
            return Err(CertifiedCzStemError::Unsupported {
                message: format!(
                    "direct Conv2d '{}' has {name} length {}, expected 1 or 2",
                    layer.name,
                    values.len()
                ),
            });
        }
        _ => {
            return Err(CertifiedCzStemError::Unsupported {
                message: format!("direct Conv2d '{}' has invalid {name}", layer.name),
            });
        }
    };
    if pair.contains(&0) {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!("direct Conv2d '{}' has zero {name}", layer.name),
        });
    }
    Ok(pair)
}

fn parse_symmetric_padding(layer: &LayerSpec) -> Result<(usize, usize), CertifiedCzStemError> {
    let Some(value) = layer.attributes.get("pads") else {
        return Ok((0, 0));
    };
    match value {
        AttributeValue::Int(value) => {
            let value = parse_nonnegative_usize(*value, layer, "pads")?;
            Ok((value, value))
        }
        AttributeValue::Ints(values) if values.is_empty() => Ok((0, 0)),
        AttributeValue::Ints(values) if values.len() == 1 => {
            let value = parse_nonnegative_usize(values[0], layer, "pads")?;
            Ok((value, value))
        }
        AttributeValue::Ints(values) if values.len() == 2 => Ok((
            parse_nonnegative_usize(values[0], layer, "pads")?,
            parse_nonnegative_usize(values[1], layer, "pads")?,
        )),
        AttributeValue::Ints(values)
            if values.len() == 4 && values[0] == values[2] && values[1] == values[3] =>
        {
            Ok((
                parse_nonnegative_usize(values[0], layer, "pads")?,
                parse_nonnegative_usize(values[1], layer, "pads")?,
            ))
        }
        AttributeValue::Ints(values) => Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "direct Conv2d '{}' requires symmetric 1/2/4-element pads, got length {}",
                layer.name,
                values.len()
            ),
        }),
        _ => Err(CertifiedCzStemError::Unsupported {
            message: format!("direct Conv2d '{}' has invalid pads", layer.name),
        }),
    }
}

fn parse_groups(layer: &LayerSpec) -> Result<usize, CertifiedCzStemError> {
    let Some(value) = layer.attributes.get("group") else {
        return Ok(1);
    };
    let AttributeValue::Int(value) = value else {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!("direct Conv2d '{}' has invalid group", layer.name),
        });
    };
    if *value <= 0 {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!("direct Conv2d '{}' has nonpositive group", layer.name),
        });
    }
    usize::try_from(*value).map_err(|_| CertifiedCzStemError::ModelDimension { dimension: *value })
}

fn parse_nonnegative_usize(
    value: i64,
    layer: &LayerSpec,
    name: &'static str,
) -> Result<usize, CertifiedCzStemError> {
    if value < 0 {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "direct Conv2d '{}' has negative {name} value {value}",
                layer.name
            ),
        });
    }
    usize::try_from(value).map_err(|_| CertifiedCzStemError::ModelDimension { dimension: value })
}

fn ensure_graph_conv_matches_raw(
    graph: &ny_propagate::layers::Conv2dLayer,
    raw: &RawConvParameters<'_>,
) -> Result<(), CertifiedCzStemError> {
    if graph.kernel.shape() != raw.kernel.shape() {
        return Err(CertifiedCzStemError::Shape {
            field: "model/graph Conv2d kernel",
            expected: raw.kernel.shape().to_vec(),
            got: graph.kernel.shape().to_vec(),
        });
    }
    for (index, (&model_value, &graph_value)) in
        raw.kernel.iter().zip(graph.kernel.iter()).enumerate()
    {
        if model_value.to_bits() != graph_value.to_bits() {
            return Err(CertifiedCzStemError::Unsupported {
                message: format!("model/graph Conv2d kernel differs at element {index}"),
            });
        }
    }
    match (raw.bias, graph.bias.as_ref()) {
        (None, None) => {}
        (Some(model), Some(graph)) if model.len() == graph.len() => {
            for (index, (&model_value, &graph_value)) in model.iter().zip(graph.iter()).enumerate()
            {
                if model_value.to_bits() != graph_value.to_bits() {
                    return Err(CertifiedCzStemError::Unsupported {
                        message: format!("model/graph Conv2d bias differs at element {index}"),
                    });
                }
            }
        }
        (model, graph) => {
            return Err(CertifiedCzStemError::Shape {
                field: "model/graph Conv2d bias",
                expected: model.map_or_else(Vec::new, |bias| vec![bias.len()]),
                got: graph.map_or_else(Vec::new, |bias| vec![bias.len()]),
            });
        }
    }
    if graph.stride != raw.stride
        || graph.padding != raw.padding
        || graph.dilation != raw.dilation
        || graph.groups != raw.groups
    {
        return Err(CertifiedCzStemError::Unsupported {
            message: format!(
                "model/graph Conv2d attributes disagree: raw stride={:?} pad={:?} dilation={:?} groups={}, graph stride={:?} pad={:?} dilation={:?} groups={}",
                raw.stride,
                raw.padding,
                raw.dilation,
                raw.groups,
                graph.stride,
                graph.padding,
                graph.dilation,
                graph.groups
            ),
        });
    }
    Ok(())
}

fn promote_parameters(
    conv: &ny_propagate::layers::Conv2dLayer,
    limits: CertifiedCzStemLimits,
) -> Result<(Array4<f64>, Vec<f64>, usize), CertifiedCzStemError> {
    let kernel = conv
        .kernel
        .view()
        .into_dimensionality::<Ix4>()
        .map_err(|_| CertifiedCzStemError::Shape {
            field: "normalized Conv2d kernel rank",
            expected: vec![4],
            got: vec![conv.kernel.ndim()],
        })?;
    let parameter_elements = kernel.len().checked_add(conv.out_channels()).ok_or(
        CertifiedCzStemError::ResourceOverflow {
            operation: "promoted Conv2d parameter elements",
        },
    )?;
    check_limit(
        "promoted Conv2d parameter elements",
        parameter_elements,
        limits.max_parameter_elements,
    )?;

    let mut values = Vec::new();
    try_reserve(&mut values, kernel.len(), "promoted Conv2d kernel")?;
    for (index, &value) in kernel.iter().enumerate() {
        if !value.is_finite() {
            return Err(CertifiedCzStemError::NonFiniteParameter {
                field: "kernel",
                index,
            });
        }
        values.push(f64::from(value));
    }
    let shape = kernel.raw_dim();
    let weights = Array4::from_shape_vec(shape, values).map_err(|_| {
        CertifiedCzStemError::ResourceOverflow {
            operation: "promoted Conv2d kernel shape",
        }
    })?;

    let mut bias = Vec::new();
    try_reserve(&mut bias, conv.out_channels(), "promoted Conv2d bias")?;
    if let Some(source) = &conv.bias {
        if source.len() != conv.out_channels() {
            return Err(CertifiedCzStemError::Shape {
                field: "normalized Conv2d bias",
                expected: vec![conv.out_channels()],
                got: vec![source.len()],
            });
        }
        for (index, &value) in source.iter().enumerate() {
            if !value.is_finite() {
                return Err(CertifiedCzStemError::NonFiniteParameter {
                    field: "bias",
                    index,
                });
            }
            bias.push(f64::from(value));
        }
    } else {
        bias.resize(conv.out_channels(), 0.0);
    }
    Ok((weights, bias, parameter_elements))
}

fn checked_product(
    dimensions: &[usize],
    operation: &'static str,
) -> Result<usize, CertifiedCzStemError> {
    dimensions.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(CertifiedCzStemError::ResourceOverflow { operation })
    })
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), CertifiedCzStemError> {
    if required > limit {
        return Err(CertifiedCzStemError::ResourceLimit {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}

fn check_bounded_sum(
    values: impl IntoIterator<Item = usize>,
    resource: &'static str,
    limit: usize,
) -> Result<usize, CertifiedCzStemError> {
    let mut total = 0_usize;
    for value in values {
        total = total
            .checked_add(value)
            .ok_or(CertifiedCzStemError::ResourceOverflow {
                operation: resource,
            })?;
        check_limit(resource, total, limit)?;
    }
    Ok(total)
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), CertifiedCzStemError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| CertifiedCzStemError::AllocationFailure { resource })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use ndarray::{Array1, Array2, Array4, ArrayD, Ix2, IxDyn};
    use num_rational::BigRational;
    use ny_core::LayerType;
    use ny_mip::{
        certified_box_affine_unwired, certified_box_conv2d_unwired,
        constrained_zonotope_affine_unwired,
        diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired,
        diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired,
        exact_relu_tail_margin_from_f64_rows, prepare_relu_tail_triangle_dual_unwired,
        transform_relu_projected_constraints_with_auxiliary_bounds_unwired,
        transform_relu_with_auxiliary_bounds_unwired, unconstrained_zonotope_box_unwired,
        CertifiedAuxiliaryBounds64, CertifiedBox64, CertifiedBox64Limits, ConstrainedZonotope64,
        ConstrainedZonotopeAffineLimits, ConstrainedZonotopeTailLpConfig,
        ConstrainedZonotopeTailLpLimits, PreparedReluTailGeometry64, ReluTailBoxCutAdamSchedule,
        ReluTailBoxCutOptimizedResult, ReluTailBoxCutOptimizerConfig,
        ReluTailBoxCutOptimizerLimits, ReluTailBoxCutSelection, ReluTailDualConfig,
        ReluTailDualLimits, ReluTailDualResult, TailLpMarginOutcome,
    };
    #[cfg(feature = "cuda")]
    use ny_mip::{propose_batched_adam_unwired, BatchedAdamConfig, BatchedAdamLimits};
    use ny_onnx::onnx_proto::{
        tensor_shape_proto, AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
        TensorProto, TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
    };
    use ny_onnx::vnnlib::{
        load_vnnlib_with_certified_input_box, parse_vnnlib_with_certified_input_box,
        OutputConstraint, VnnLibSpec,
    };
    use ny_onnx::{
        load_onnx_bytes, load_onnx_bytes_with_config, load_onnx_with_config, Network,
        OnnxLoadConfig, ShapeInferencePolicy, TensorSpec, WeightStore,
    };
    use ny_propagate::layers::Conv2dLayer;
    use ny_propagate::{GraphNetwork, GraphNode};
    use prost::Message;

    use super::*;

    fn limits() -> CertifiedCzStemLimits {
        CertifiedCzStemLimits {
            max_graph_nodes: 16,
            max_graph_edges: 64,
            max_topology_work_items: 4_096,
            max_input_values: 16,
            max_input_alpha_dim: 16,
            max_input_generator_nonzeros: 16,
            max_input_stored_f64: 64,
            max_parameter_elements: 64,
            conv: ConstrainedZonotopeConv2dLimits {
                max_value_count: 64,
                max_alpha_dim: 16,
                max_generator_nonzeros: 64,
                max_weight_elements: 64,
                max_kernel_visits: 1_024,
                max_interval_products: 4_096,
                max_constraint_count: 0,
                max_constraint_elements: 0,
            },
        }
    }

    fn input_box() -> CertifiedInputBox {
        let content = "
            (declare-const X_0 Real)
            (declare-const X_1 Real)
            (declare-const X_2 Real)
            (declare-const X_3 Real)
            (assert (>= X_0 -1.0))
            (assert (<= X_0 1.0))
            (assert (= X_1 0.1))
            (assert (>= X_2 2.0))
            (assert (<= X_2 4.0))
            (assert (= X_3 0.0))
        ";
        parse_vnnlib_with_certified_input_box(content).unwrap().1
    }

    fn tensor_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
        let dimensions = shape
            .iter()
            .copied()
            .map(|dimension| tensor_shape_proto::Dimension {
                value: Some(tensor_shape_proto::dimension::Value::DimValue(dimension)),
            })
            .collect();
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto { dim: dimensions }),
                }),
            }),
        }
    }

    fn float32_tensor(name: &str, shape: &[i64], values: &[f32]) -> TensorProto {
        TensorProto {
            dims: shape.to_vec(),
            data_type: 1,
            name: name.to_string(),
            raw_data: Vec::new(),
            float_data: values.to_vec(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
            double_data: Vec::new(),
            data_location: 0,
        }
    }

    fn onnx_node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            input: inputs.iter().map(ToString::to_string).collect(),
            output: outputs.iter().map(ToString::to_string).collect(),
            name: name.to_string(),
            op_type: op_type.to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }
    }

    fn encode_tiny_model(nodes: Vec<NodeProto>, initializers: Vec<TensorProto>) -> Vec<u8> {
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            producer_name: "ny-cz-stem-test".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            graph: Some(GraphProto {
                node: nodes,
                name: "tiny".to_string(),
                initializer: initializers,
                input: vec![tensor_value_info("input", &[1, 1, 2, 2])],
                output: vec![tensor_value_info("output", &[1, 1, 2, 2])],
                value_info: Vec::new(),
            }),
        }
        .encode_to_vec()
    }

    fn provenance_load_config() -> OnnxLoadConfig {
        OnnxLoadConfig::default()
            .with_shape_inference_policy(ShapeInferencePolicy::Skip)
            .with_raw_float32_initializer_provenance(true)
    }

    fn load_provenance_bytes(name: &str, bytes: &[u8]) -> ny_core::Result<OnnxModel> {
        load_onnx_bytes_with_config(name, bytes, &provenance_load_config())
    }

    fn initializer_tiny_model(weight: f32) -> OnnxModel {
        let bytes = encode_tiny_model(
            vec![onnx_node(
                "conv",
                "Conv",
                &["input", "kernel", "bias"],
                &["output"],
            )],
            vec![
                float32_tensor("kernel", &[1, 1, 1, 1], &[weight]),
                float32_tensor("bias", &[1], &[0.5]),
            ],
        );
        load_provenance_bytes("cz_stem_tiny.onnx", &bytes).unwrap()
    }

    #[test]
    fn default_loader_has_no_provenance_or_revision_tracking() {
        let bytes = encode_tiny_model(
            vec![onnx_node(
                "conv",
                "Conv",
                &["input", "kernel", "bias"],
                &["output"],
            )],
            vec![
                float32_tensor("kernel", &[1, 1, 1, 1], &[2.0]),
                float32_tensor("bias", &[1], &[0.5]),
            ],
        );
        let model = load_onnx_bytes("default_unsealed.onnx", &bytes).unwrap();
        assert_eq!(
            model.original_float32_initializer_matches_current("kernel"),
            None
        );
        assert_eq!(model.original_network_topology_matches_current(), None);
        assert!(model.weights.revision("kernel").is_none());

        let graph = model.to_graph_network().unwrap();
        let error = propagate_direct_onnx_conv2d_unwired(&model, &graph, &input_box(), limits())
            .expect_err("the default loader must not qualify the bridge");
        assert!(
            error
                .to_string()
                .contains("not loaded with private finalized-network provenance"),
            "{error}"
        );
    }

    fn tiny_model_and_graph(weight: f32) -> (OnnxModel, GraphNetwork) {
        let model = initializer_tiny_model(weight);
        let graph = model.to_graph_network().unwrap();
        (model, graph)
    }

    fn synthetic_model_and_graph(weight: f32) -> (OnnxModel, GraphNetwork) {
        let mut weights = WeightStore::new();
        weights.insert(
            "kernel".to_string(),
            Array4::from_elem((1, 1, 1, 1), weight).into_dyn(),
        );
        weights.insert("bias".to_string(), Array1::from_vec(vec![0.5]).into_dyn());
        let layer = LayerSpec {
            name: "conv".to_string(),
            layer_type: LayerType::Conv2d,
            inputs: vec![
                "input".to_string(),
                "kernel".to_string(),
                "bias".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };
        let network = Network {
            name: "tiny".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 1, 2, 2],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![1, 1, 2, 2],
                dtype: DataType::Float32,
            }],
            layers: vec![layer],
            param_count: 2,
        };
        let model = OnnxModel::empty_with_network(network, weights);

        let kernel: ArrayD<f32> = Array4::from_elem((1, 1, 1, 1), weight).into_dyn();
        let conv = Conv2dLayer::with_input_shape(
            kernel,
            Some(Array1::from_vec(vec![0.5])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .unwrap();
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.set_output("conv");
        (model, graph)
    }

    fn constant_kernel_tiny_model(weight: f32) -> OnnxModel {
        let mut constant = onnx_node("kernel_constant", "Constant", &[], &["kernel"]);
        constant.attribute.push(AttributeProto {
            name: "value".to_string(),
            f: 0.0,
            i: 0,
            s: Vec::new(),
            t: Some(float32_tensor("", &[1, 1, 1, 1], &[weight])),
            r#type: 4,
            floats: Vec::new(),
            ints: Vec::new(),
        });
        let bytes = encode_tiny_model(
            vec![
                constant,
                onnx_node("conv", "Conv", &["input", "kernel", "bias"], &["output"]),
            ],
            vec![float32_tensor("bias", &[1], &[0.5])],
        );
        load_provenance_bytes("cz_stem_constant_kernel.onnx", &bytes).unwrap()
    }

    fn folded_kernel_tiny_model(weight: f32) -> OnnxModel {
        let bytes = encode_tiny_model(
            vec![
                onnx_node(
                    "fold_kernel",
                    "Add",
                    &["kernel_lhs", "kernel_rhs"],
                    &["kernel"],
                ),
                onnx_node("conv", "Conv", &["input", "kernel", "bias"], &["output"]),
            ],
            vec![
                float32_tensor("kernel_lhs", &[1, 1, 1, 1], &[weight]),
                float32_tensor("kernel_rhs", &[1, 1, 1, 1], &[0.0]),
                float32_tensor("bias", &[1], &[0.5]),
            ],
        );
        load_provenance_bytes("cz_stem_folded_kernel.onnx", &bytes).unwrap()
    }

    #[test]
    fn public_loader_rejects_initializer_node_output_collisions() {
        let direct_collision = encode_tiny_model(
            vec![onnx_node(
                "replace_kernel",
                "Add",
                &["kernel_lhs", "kernel_rhs"],
                &["kernel"],
            )],
            vec![
                float32_tensor("kernel", &[1], &[2.0]),
                float32_tensor("kernel_lhs", &[1], &[2.0]),
                float32_tensor("kernel_rhs", &[1], &[0.0]),
            ],
        );
        let error = load_provenance_bytes("direct_collision.onnx", &direct_collision)
            .expect_err("a raw initializer cannot also be a graph node output");
        assert!(
            error
                .to_string()
                .contains("initializer 'kernel' collides with output"),
            "{error}"
        );

        // ReduceL2 lowering synthesizes this Pow output only after the first
        // collision check. The post-rewrite check must reject it too.
        let generated_name = "reduce__reduce_l2_square";
        let generated_collision = encode_tiny_model(
            vec![onnx_node("reduce", "ReduceL2", &["input"], &["output"])],
            vec![float32_tensor(generated_name, &[1], &[2.0])],
        );
        let error = load_provenance_bytes("generated_collision.onnx", &generated_collision)
            .expect_err("a generated graph output cannot collide with a raw initializer");
        assert!(
            error
                .to_string()
                .contains("initializer 'reduce__reduce_l2_square' collides with output"),
            "{error}"
        );

        let empty_initializer = encode_tiny_model(
            vec![onnx_node("conv", "Conv", &["input", ""], &["output"])],
            vec![float32_tensor("", &[1, 1, 1, 1], &[2.0])],
        );
        let error = load_provenance_bytes("empty_initializer.onnx", &empty_initializer)
            .expect_err("an empty initializer name is the ONNX omitted-input sentinel");
        assert!(
            error
                .to_string()
                .contains("initializer name cannot be empty"),
            "{error}"
        );
    }

    fn batchnorm_fused_tiny_model(weight: f32) -> OnnxModel {
        let bytes = encode_tiny_model(
            vec![
                onnx_node(
                    "conv",
                    "Conv",
                    &["input", "kernel", "bias"],
                    &["conv_output"],
                ),
                onnx_node(
                    "batchnorm",
                    "BatchNormalization",
                    &["conv_output", "scale", "beta", "mean", "variance"],
                    &["output"],
                ),
            ],
            vec![
                float32_tensor("kernel", &[1, 1, 1, 1], &[weight]),
                float32_tensor("bias", &[1], &[0.5]),
                float32_tensor("scale", &[1], &[2.0]),
                float32_tensor("beta", &[1], &[0.0]),
                float32_tensor("mean", &[1], &[0.0]),
                float32_tensor("variance", &[1], &[1.0]),
            ],
        );
        load_provenance_bytes("cz_stem_fused_batchnorm.onnx", &bytes).unwrap()
    }

    fn one_node_graph(conv: Conv2dLayer) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
        graph.set_output("conv");
        graph
    }

    fn mutable_tiny_graph_conv(weight: f32) -> Conv2dLayer {
        Conv2dLayer::with_input_shape(
            Array4::from_elem((1, 1, 1, 1), weight).into_dyn(),
            Some(Array1::from_vec(vec![0.5])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .unwrap()
    }

    fn tiny_relu_model_and_graph(weight: f32) -> (OnnxModel, GraphNetwork) {
        let bytes = encode_tiny_model(
            vec![
                onnx_node(
                    "conv",
                    "Conv",
                    &["input", "kernel", "bias"],
                    &["conv_output"],
                ),
                onnx_node("relu", "Relu", &["conv_output"], &["output"]),
            ],
            vec![
                float32_tensor("kernel", &[1, 1, 1, 1], &[weight]),
                float32_tensor("bias", &[1], &[0.5]),
            ],
        );
        let model = load_provenance_bytes("cz_relu_stem_tiny.onnx", &bytes).unwrap();
        let graph = model.to_graph_network().unwrap();
        (model, graph)
    }

    fn relu_limits() -> CertifiedCzReluStemLimits {
        CertifiedCzReluStemLimits {
            stem: limits(),
            relu: ReluTransformLimits {
                max_value_dim: 64,
                max_output_alpha_dim: 16,
                max_constraints: 0,
                max_constraint_elements: 0,
                max_generator_nnz: 64,
                max_unstable: 16,
                max_exact_terms: 1_024,
            },
        }
    }

    #[test]
    fn exact_decimal_box_reaches_bit_matched_sparse_conv() {
        let (model, graph) = tiny_model_and_graph(2.0);
        let (output, plan) =
            propagate_direct_onnx_conv2d_unwired(&model, &graph, &input_box(), limits()).unwrap();

        assert_eq!(plan.input_shape, [1, 2, 2]);
        assert_eq!(plan.conv_node, "conv");
        assert_eq!(plan.parameter_elements, 2);
        assert_eq!(plan.input.value_dim, 4);
        assert_eq!(plan.input.declared_point_count, 2);
        assert_eq!(plan.input.alpha_dim, 2);
        assert_eq!(plan.conv.output_shape, [1, 2, 2]);
        assert_eq!(output.value_dim(), 4);
        assert_eq!(output.alpha_dim(), 2);

        let first = output.evaluate_dual(&[1.0, 0.0, 0.0, 0.0], &[]).unwrap();
        assert!(first.lower <= -1.5);
        assert!(first.upper >= 2.5);
        let point = output.evaluate_dual(&[0.0, 1.0, 0.0, 0.0], &[]).unwrap();
        assert!(point.lower <= 0.7);
        assert!(point.upper >= 0.7);
    }

    #[test]
    fn model_graph_parameter_drift_fails_closed() {
        let (model, graph) = tiny_model_and_graph(2.0);
        let (_, different_graph) = tiny_model_and_graph(2.0_f32.next_up());
        let error =
            propagate_direct_onnx_conv2d_unwired(&model, &different_graph, &input_box(), limits())
                .unwrap_err();
        assert!(error.to_string().contains("kernel differs"));

        // Keep the matching fixture live so this regression cannot pass only
        // because both graph variants are rejected for an unrelated reason.
        propagate_direct_onnx_conv2d_unwired(&model, &graph, &input_box(), limits()).unwrap();
    }

    #[test]
    fn synthetic_constant_folded_and_fused_parameters_fail_provenance() {
        let (synthetic, synthetic_graph) = synthetic_model_and_graph(2.0);
        let error = propagate_direct_onnx_conv2d_unwired(
            &synthetic,
            &synthetic_graph,
            &input_box(),
            limits(),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not loaded with private finalized-network provenance"),
            "{error}"
        );

        for (label, model) in [
            ("Constant", constant_kernel_tiny_model(2.0)),
            ("constant fold", folded_kernel_tiny_model(2.0)),
        ] {
            let graph = model.to_graph_network().unwrap();
            let error =
                propagate_direct_onnx_conv2d_unwired(&model, &graph, &input_box(), limits())
                    .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("no private raw ONNX FLOAT initializer provenance"),
                "{label}: {error}"
            );
        }

        let fused = batchnorm_fused_tiny_model(2.0);
        let fused_graph = fused.to_graph_network().unwrap();
        let error =
            propagate_direct_onnx_conv2d_unwired(&fused, &fused_graph, &input_box(), limits())
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no longer matches its private raw ONNX FLOAT initializer provenance"),
            "{error}"
        );
    }

    #[test]
    fn post_load_weight_mutation_fails_even_when_graph_matches() {
        let mut model = initializer_tiny_model(2.0);
        model.weights.insert(
            "kernel".to_string(),
            // Even a bit-identical replacement is not raw provenance: a
            // generic store mutation may represent an identity-valued fusion.
            Array4::from_elem((1, 1, 1, 1), 2.0).into_dyn(),
        );
        let graph = model.to_graph_network().unwrap();
        let error = propagate_direct_onnx_conv2d_unwired(&model, &graph, &input_box(), limits())
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no longer matches its private raw ONNX FLOAT initializer provenance"),
            "{error}"
        );
    }

    #[test]
    fn post_load_network_mutation_fails_even_when_graph_matches() {
        let mut changed_stride = initializer_tiny_model(2.0);
        changed_stride.network.layers[0]
            .attributes
            .insert("strides".to_string(), AttributeValue::Ints(vec![2, 2]));
        let graph = changed_stride.to_graph_network().unwrap();
        let error =
            propagate_direct_onnx_conv2d_unwired(&changed_stride, &graph, &input_box(), limits())
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("network no longer matches private finalized-network provenance"),
            "{error}"
        );

        let mut changed_input = initializer_tiny_model(2.0);
        changed_input.network.inputs[0].shape[2] = 3;
        let graph = changed_input.to_graph_network().unwrap();
        let error =
            propagate_direct_onnx_conv2d_unwired(&changed_input, &graph, &input_box(), limits())
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("network no longer matches private finalized-network provenance"),
            "{error}"
        );
    }

    #[test]
    fn replacing_loaded_store_cannot_forge_initializer_provenance() {
        let mut model = initializer_tiny_model(2.0);
        let graph = model.to_graph_network().unwrap();
        let mut replacement = WeightStore::new();
        replacement.insert(
            "kernel".to_string(),
            Array4::from_elem((1, 1, 1, 1), 2.0).into_dyn(),
        );
        replacement.insert("bias".to_string(), Array1::from_vec(vec![0.5]).into_dyn());
        model.weights = replacement;

        let error = propagate_direct_onnx_conv2d_unwired(&model, &graph, &input_box(), limits())
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no longer matches its private raw ONNX FLOAT initializer provenance"),
            "{error}"
        );

        let mut cloned_store_model = initializer_tiny_model(2.0);
        let cloned_store_graph = cloned_store_model.to_graph_network().unwrap();
        cloned_store_model.weights = cloned_store_model.weights.clone();
        let error = propagate_direct_onnx_conv2d_unwired(
            &cloned_store_model,
            &cloned_store_graph,
            &input_box(),
            limits(),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no longer matches its private raw ONNX FLOAT initializer provenance"),
            "{error}"
        );
    }

    #[test]
    fn actual_raw_bias_length_is_capped_before_comparison() {
        let mut model = initializer_tiny_model(2.0);
        model.weights.insert(
            "bias".to_string(),
            Array1::from_vec(vec![0.5, 0.5]).into_dyn(),
        );
        let mut conv = mutable_tiny_graph_conv(2.0);
        conv.bias = Some(Array1::from_vec(vec![0.5, 0.5]));
        let graph = one_node_graph(conv);

        let mut capped = limits();
        capped.max_parameter_elements = 2;
        let error =
            propagate_direct_onnx_conv2d_unwired(&model, &graph, &input_box(), capped).unwrap_err();
        assert!(
            matches!(
                &error,
                CertifiedCzStemError::ResourceLimit {
                    resource: "promoted Conv2d parameter elements",
                    required: 3,
                    limit: 2,
                }
            ),
            "{error}"
        );

        capped.max_parameter_elements = 3;
        let error =
            propagate_direct_onnx_conv2d_unwired(&model, &graph, &input_box(), capped).unwrap_err();
        assert!(
            matches!(
                &error,
                CertifiedCzStemError::Shape {
                    field: "raw Conv2d bias",
                    ..
                }
            ),
            "{error}"
        );
    }

    #[test]
    fn malformed_normalized_conv_rejects_without_panicking() {
        let model = initializer_tiny_model(2.0);

        let mut rank_zero = mutable_tiny_graph_conv(2.0);
        rank_zero.kernel = ArrayD::from_elem(IxDyn(&[]), 2.0);
        let error = propagate_direct_onnx_conv2d_unwired(
            &model,
            &one_node_graph(rank_zero),
            &input_box(),
            limits(),
        )
        .unwrap_err();
        assert!(
            matches!(
                &error,
                CertifiedCzStemError::Shape {
                    field: "normalized Conv2d kernel rank",
                    ..
                }
            ),
            "{error}"
        );

        let mut high_rank = mutable_tiny_graph_conv(2.0);
        high_rank.kernel = ArrayD::from_elem(IxDyn(&vec![1; 4_096]), 2.0);
        let error = propagate_direct_onnx_conv2d_unwired(
            &model,
            &one_node_graph(high_rank),
            &input_box(),
            limits(),
        )
        .unwrap_err();
        assert!(
            matches!(
                &error,
                CertifiedCzStemError::Shape {
                    field: "normalized Conv2d kernel rank",
                    expected,
                    got,
                } if expected == &[4] && got == &[4_096]
            ),
            "{error}"
        );

        let mut zero_groups = mutable_tiny_graph_conv(2.0);
        zero_groups.groups = 0;
        let error = propagate_direct_onnx_conv2d_unwired(
            &model,
            &one_node_graph(zero_groups),
            &input_box(),
            limits(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("zero groups"), "{error}");

        let mut wrong_bias = mutable_tiny_graph_conv(2.0);
        wrong_bias.bias = Some(Array1::from_vec(vec![0.5, 0.5]));
        let error = propagate_direct_onnx_conv2d_unwired(
            &model,
            &one_node_graph(wrong_bias),
            &input_box(),
            limits(),
        )
        .unwrap_err();
        assert!(
            matches!(
                &error,
                CertifiedCzStemError::Shape {
                    field: "normalized Conv2d bias",
                    ..
                }
            ),
            "{error}"
        );
    }

    #[test]
    fn original_nonfinite_parameter_rejects_with_typed_error() {
        let (model, graph) = tiny_model_and_graph(f32::INFINITY);
        let error = propagate_direct_onnx_conv2d_unwired(&model, &graph, &input_box(), limits())
            .unwrap_err();
        assert!(
            matches!(
                &error,
                CertifiedCzStemError::NonFiniteParameter {
                    field: "kernel",
                    index: 0,
                }
            ),
            "{error}"
        );
    }

    #[test]
    fn malformed_raw_rank_and_oversized_topology_are_bounded() {
        let mut malformed_raw = initializer_tiny_model(2.0);
        let graph = malformed_raw.to_graph_network().unwrap();
        malformed_raw.weights.insert(
            "kernel".to_string(),
            ArrayD::from_elem(IxDyn(&vec![1; 4_096]), 2.0),
        );
        let error =
            propagate_direct_onnx_conv2d_unwired(&malformed_raw, &graph, &input_box(), limits())
                .unwrap_err();
        assert!(
            matches!(
                &error,
                CertifiedCzStemError::Shape {
                    field: "raw Conv2d kernel rank",
                    expected,
                    got,
                } if expected == &[4] && got == &[4_096]
            ),
            "{error}"
        );

        let mut oversized_topology = initializer_tiny_model(2.0);
        let graph = oversized_topology.to_graph_network().unwrap();
        oversized_topology.network.layers[0]
            .attributes
            .insert("strides".to_string(), AttributeValue::Ints(vec![1; 4_097]));
        let error = propagate_direct_onnx_conv2d_unwired(
            &oversized_topology,
            &graph,
            &input_box(),
            limits(),
        )
        .unwrap_err();
        assert!(
            matches!(
                &error,
                CertifiedCzStemError::ResourceLimit {
                    resource: "model/graph topology work items",
                    ..
                }
            ),
            "{error}"
        );

        let mut oversized_capacity = initializer_tiny_model(2.0);
        let graph = oversized_capacity.to_graph_network().unwrap();
        oversized_capacity.network.layers[0]
            .attributes
            .reserve(4_097);
        let error = propagate_direct_onnx_conv2d_unwired(
            &oversized_capacity,
            &graph,
            &input_box(),
            limits(),
        )
        .unwrap_err();
        assert!(
            matches!(
                &error,
                CertifiedCzStemError::ResourceLimit {
                    resource: "model/graph topology work items",
                    ..
                }
            ),
            "{error}"
        );
    }

    #[test]
    fn exact_decimal_box_reaches_matching_relu_successor() {
        let (model, graph) = tiny_relu_model_and_graph(2.0);
        let (output, plan) =
            propagate_direct_onnx_conv2d_relu_unwired(&model, &graph, &input_box(), relu_limits())
                .unwrap();

        assert_eq!(plan.stem.conv_node, "conv");
        assert_eq!(plan.relu_node, "relu");
        assert_eq!(plan.input_alpha_dim, 2);
        assert_eq!(plan.output_alpha_dim, 3);
        assert_eq!(plan.unstable_count, 1);
        assert_eq!(output.value_dim(), 4);
        let first = output.evaluate_dual(&[1.0, 0.0, 0.0, 0.0], &[]).unwrap();
        assert!(first.lower <= 0.0);
        assert!(first.upper >= 2.5);
        let point = output.evaluate_dual(&[0.0, 1.0, 0.0, 0.0], &[]).unwrap();
        assert!(point.lower <= 0.7);
        assert!(point.upper >= 0.7);
    }

    #[test]
    fn projected_relu_stem_is_opt_in_and_exactly_capped() {
        let (model, graph) = tiny_relu_model_and_graph(2.0);
        let (preserved, preserved_plan) =
            propagate_direct_onnx_conv2d_relu_unwired(&model, &graph, &input_box(), relu_limits())
                .unwrap();

        let mut projected_limits = relu_limits();
        projected_limits.relu.max_constraints = 2;
        projected_limits.relu.max_constraint_elements = 6;
        let (projected, projected_plan) =
            propagate_direct_onnx_conv2d_relu_projected_constraints_unwired(
                &model,
                &graph,
                &input_box(),
                projected_limits,
            )
            .unwrap();

        assert_eq!(preserved_plan, projected_plan);
        assert_eq!(preserved.constraint_count(), 0);
        assert_eq!(projected.constraint_count(), 2);
        assert_eq!(projected.rhs().len(), 2);
        assert_eq!(projected.constraints().len(), 6);
        assert_eq!(preserved.center(), projected.center());
        assert_eq!(preserved.generators(), projected.generators());
        assert_eq!(preserved.box_remainder(), projected.box_remainder());

        let mut capped_rows = projected_limits;
        capped_rows.relu.max_constraints = 1;
        let row_error = propagate_direct_onnx_conv2d_relu_projected_constraints_unwired(
            &model,
            &graph,
            &input_box(),
            capped_rows,
        )
        .unwrap_err();
        assert!(
            row_error.to_string().contains("constraint count"),
            "{row_error}"
        );

        let mut capped_elements = projected_limits;
        capped_elements.relu.max_constraint_elements = 5;
        let element_error = propagate_direct_onnx_conv2d_relu_projected_constraints_unwired(
            &model,
            &graph,
            &input_box(),
            capped_elements,
        )
        .unwrap_err();
        assert!(
            element_error.to_string().contains("constraint elements"),
            "{element_error}"
        );
    }

    #[test]
    fn relu_topology_and_resource_drift_fail_closed() {
        let (model, graph) = tiny_model_and_graph(2.0);
        let error =
            propagate_direct_onnx_conv2d_relu_unwired(&model, &graph, &input_box(), relu_limits())
                .unwrap_err();
        assert!(error.to_string().contains("no direct consumer"));

        let (mut changed_model, _) = tiny_relu_model_and_graph(2.0);
        changed_model.network.layers[1].name = "changed_relu".to_string();
        let changed_graph = changed_model.to_graph_network().unwrap();
        let error = propagate_direct_onnx_conv2d_relu_unwired(
            &changed_model,
            &changed_graph,
            &input_box(),
            relu_limits(),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("network no longer matches private finalized-network provenance"),
            "{error}"
        );

        let (model, graph) = tiny_relu_model_and_graph(2.0);
        let (_, wrong_graph) = tiny_relu_model_and_graph(2.0);
        let mut capped = relu_limits();
        capped.relu.max_output_alpha_dim = 2;
        let error =
            propagate_direct_onnx_conv2d_relu_unwired(&model, &wrong_graph, &input_box(), capped)
                .unwrap_err();
        assert!(error.to_string().contains("alpha"), "{error}");

        // Keep the matching graph live so the cap assertion cannot pass only
        // because the topology itself is rejected.
        propagate_direct_onnx_conv2d_relu_unwired(&model, &graph, &input_box(), relu_limits())
            .unwrap();
    }

    #[test]
    fn every_preallocation_cap_fails_before_expansion() {
        let box_ = input_box();
        for (resource, capped) in [
            ("input value count", {
                let mut value = limits();
                value.max_input_values = 3;
                value
            }),
            ("input alpha dimension", {
                let mut value = limits();
                value.max_input_alpha_dim = 1;
                value
            }),
            ("input generator nonzeros", {
                let mut value = limits();
                value.max_input_generator_nonzeros = 1;
                value
            }),
            ("input stored f64 scalars", {
                let mut value = limits();
                value.max_input_stored_f64 = 9;
                value
            }),
        ] {
            let error = certified_input_box_to_cz_unwired(&box_, capped).unwrap_err();
            assert!(error.to_string().contains(resource), "{error}");
        }

        let (model, graph) = tiny_model_and_graph(2.0);
        let mut capped = limits();
        capped.max_graph_edges = 2;
        let error =
            propagate_direct_onnx_conv2d_unwired(&model, &graph, &box_, capped).unwrap_err();
        assert!(error.to_string().contains("model edge count"));

        let mut capped = limits();
        capped.max_topology_work_items = 0;
        let error =
            propagate_direct_onnx_conv2d_unwired(&model, &graph, &box_, capped).unwrap_err();
        assert!(error.to_string().contains("topology work items"));

        let mut capped = limits();
        capped.max_parameter_elements = 1;
        let error =
            propagate_direct_onnx_conv2d_unwired(&model, &graph, &box_, capped).unwrap_err();
        assert!(error.to_string().contains("parameter elements"));
    }

    #[test]
    fn real_metaroom_119_builds_161_symbol_first_conv_stem() {
        let root = std::env::var_os("NY_METAROOM_119_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../benchmarks/vnncomp2025/benchmarks/metaroom_2023")
            });
        let onnx = root.join("onnx/6cnn_ry_39_6_no_custom_OP.onnx");
        let vnnlib = root.join("vnnlib/spec_idx_119_eps_0.00000436.vnnlib");
        if !onnx.exists() || !vnnlib.exists() {
            return;
        }

        let config = OnnxLoadConfig::default().with_raw_float32_initializer_provenance(true);
        let model = load_onnx_with_config(&onnx, &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (_, box_) = load_vnnlib_with_certified_input_box(&vnnlib).unwrap();
        let metaroom_limits = CertifiedCzStemLimits {
            max_graph_nodes: 32,
            max_graph_edges: 128,
            max_topology_work_items: 16_384,
            max_input_values: 5_376,
            max_input_alpha_dim: 161,
            max_input_generator_nonzeros: 161,
            max_input_stored_f64: 10_913,
            max_parameter_elements: 896,
            conv: ConstrainedZonotopeConv2dLimits {
                max_value_count: 57_344,
                max_alpha_dim: 161,
                max_generator_nonzeros: 46_368,
                max_weight_elements: 864,
                max_kernel_visits: 1_548_288,
                max_interval_products: 3_042_336,
                max_constraint_count: 0,
                max_constraint_elements: 0,
            },
        };
        let (output, plan) =
            propagate_direct_onnx_conv2d_unwired(&model, &graph, &box_, metaroom_limits).unwrap();
        assert_eq!(plan.input_shape, [3, 32, 56]);
        assert_eq!(plan.input.value_dim, 5_376);
        assert_eq!(plan.input.declared_point_count, 5_215);
        assert_eq!(plan.input.alpha_dim, 161);
        assert_eq!(plan.input.generator_nonzeros, 161);
        assert_eq!(plan.input.stored_f64, 10_913);
        assert_eq!(plan.parameter_elements, 896);
        assert_eq!(plan.conv.weight_elements, 864);
        assert_eq!(plan.conv.kernel_visits, 1_548_288);
        assert_eq!(plan.conv.output_generator_nonzeros, 43_200);
        assert_eq!(plan.conv.interval_products, 2_873_856);
        assert_eq!(plan.conv.output_shape, [32, 32, 56]);
        assert_eq!(output.value_dim(), 57_344);
        assert_eq!(output.alpha_dim(), 161);
    }

    #[test]
    fn real_metaroom_119_reaches_first_relu_with_bounded_symbol_growth() {
        let root = std::env::var_os("NY_METAROOM_119_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../benchmarks/vnncomp2025/benchmarks/metaroom_2023")
            });
        let onnx = root.join("onnx/6cnn_ry_39_6_no_custom_OP.onnx");
        let vnnlib = root.join("vnnlib/spec_idx_119_eps_0.00000436.vnnlib");
        if !onnx.exists() || !vnnlib.exists() {
            return;
        }

        let config = OnnxLoadConfig::default().with_raw_float32_initializer_provenance(true);
        let model = load_onnx_with_config(&onnx, &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (_, box_) = load_vnnlib_with_certified_input_box(&vnnlib).unwrap();
        let limits = CertifiedCzReluStemLimits {
            stem: CertifiedCzStemLimits {
                max_graph_nodes: 32,
                max_graph_edges: 128,
                max_topology_work_items: 16_384,
                max_input_values: 5_376,
                max_input_alpha_dim: 161,
                max_input_generator_nonzeros: 161,
                max_input_stored_f64: 10_913,
                max_parameter_elements: 896,
                conv: ConstrainedZonotopeConv2dLimits {
                    max_value_count: 57_344,
                    max_alpha_dim: 161,
                    max_generator_nonzeros: 46_368,
                    max_weight_elements: 864,
                    max_kernel_visits: 1_548_288,
                    max_interval_products: 3_042_336,
                    max_constraint_count: 0,
                    max_constraint_elements: 0,
                },
            },
            relu: ReluTransformLimits {
                max_value_dim: 57_344,
                max_output_alpha_dim: 637,
                max_constraints: 0,
                max_constraint_elements: 0,
                max_generator_nnz: 43_676,
                max_unstable: 476,
                max_exact_terms: 145_648,
            },
        };
        let (output, plan) =
            propagate_direct_onnx_conv2d_relu_unwired(&model, &graph, &box_, limits).unwrap();
        eprintln!("real Metaroom Conv1->ReLU1 plan: {plan:#?}");
        assert_eq!(plan.stem.conv.output_shape, [32, 32, 56]);
        assert_eq!(plan.input_alpha_dim, 161);
        assert_eq!(plan.output_alpha_dim, 637);
        assert_eq!(plan.unstable_count, 476);
        assert_eq!(plan.output_generator_nonzeros, 15_348);
        assert_eq!(plan.nonzero_remainder_count, 19_960);
        assert_eq!(output.value_dim(), 57_344);
        assert_eq!(output.alpha_dim(), plan.output_alpha_dim);
        assert_eq!(
            plan.output_alpha_dim,
            plan.input_alpha_dim + plan.unstable_count
        );
        assert!(plan.output_generator_nonzeros <= 43_676);
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct MetaroomReluResources {
        stage: &'static str,
        input_alpha_dim: usize,
        output_alpha_dim: usize,
        unstable: usize,
        generator_nonzeros: usize,
        constraint_count: usize,
        constraint_elements: usize,
        nonzero_remainders: usize,
    }

    impl MetaroomReluResources {
        fn capture(
            stage: &'static str,
            input_alpha_dim: usize,
            output: &ConstrainedZonotope64,
        ) -> Self {
            let output_alpha_dim = output.alpha_dim();
            Self {
                stage,
                input_alpha_dim,
                output_alpha_dim,
                unstable: output_alpha_dim.checked_sub(input_alpha_dim).unwrap(),
                generator_nonzeros: output
                    .generators()
                    .iter()
                    .map(ny_mip::SparseGenerator64::nnz)
                    .sum(),
                constraint_count: output.constraint_count(),
                constraint_elements: output.constraints().len(),
                nonzero_remainders: output
                    .box_remainder()
                    .iter()
                    .filter(|&&value| value != 0.0)
                    .count(),
            }
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct MetaroomAffineResources {
        input_alpha_dim: usize,
        input_generator_nonzeros: usize,
        output_generator_nonzeros: usize,
        constraint_count: usize,
        constraint_elements: usize,
        weight_elements: usize,
        matrix_visits: usize,
        interval_products: usize,
        nonzero_remainders: usize,
    }

    struct QualifiedMetaroomConvReluTrunk {
        model: OnnxModel,
        graph: GraphNetwork,
        vnnlib_spec: VnnLibSpec,
        relu4_output: ConstrainedZonotope64,
        graph_relu4_name: String,
        model_relu4_output: String,
        relu_resources: Vec<MetaroomReluResources>,
        relu4_auxiliary_counterfactual: Option<MetaroomReluResources>,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum FullTrunkPredicateMode {
        Preserve,
        ProjectedReluGeometry,
    }

    impl FullTrunkPredicateMode {
        fn select(self, preserved: usize, projected: usize) -> usize {
            match self {
                Self::Preserve => preserved,
                Self::ProjectedReluGeometry => projected,
            }
        }

        fn propagate_first_relu(
            self,
            model: &OnnxModel,
            graph: &GraphNetwork,
            input_box: &CertifiedInputBox,
            limits: CertifiedCzReluStemLimits,
        ) -> Result<(ConstrainedZonotope64, CertifiedCzReluStemPlan), CertifiedCzReluStemError>
        {
            match self {
                Self::Preserve => {
                    propagate_direct_onnx_conv2d_relu_unwired(model, graph, input_box, limits)
                }
                Self::ProjectedReluGeometry => {
                    propagate_direct_onnx_conv2d_relu_projected_constraints_unwired(
                        model, graph, input_box, limits,
                    )
                }
            }
        }

        fn transform_relu(
            self,
            input: &ConstrainedZonotope64,
            limits: ReluTransformLimits,
        ) -> Result<ConstrainedZonotope64, ReluTransformError> {
            match self {
                Self::Preserve => transform_relu_unwired(input, limits),
                Self::ProjectedReluGeometry => {
                    transform_relu_projected_constraints_unwired(input, limits)
                }
            }
        }

        fn transform_relu_with_auxiliary(
            self,
            input: &ConstrainedZonotope64,
            auxiliary: &CertifiedAuxiliaryBounds64,
            mut limits: ReluTransformLimits,
        ) -> Result<ConstrainedZonotope64, ReluTransformError> {
            let auxiliary_exact_terms = input.value_dim().checked_mul(4).unwrap();
            limits.max_exact_terms = limits
                .max_exact_terms
                .checked_add(auxiliary_exact_terms)
                .unwrap();
            match self {
                Self::Preserve => {
                    transform_relu_with_auxiliary_bounds_unwired(input, auxiliary, limits)
                }
                Self::ProjectedReluGeometry => {
                    transform_relu_projected_constraints_with_auxiliary_bounds_unwired(
                        input, auxiliary, limits,
                    )
                }
            }
        }
    }

    const PROJECTED_RELU_ROWS: [usize; 4] = [952, 2_326, 3_688, 5_944];
    const PROJECTED_RELU_ELEMENTS: [usize; 4] = [606_424, 3_079_624, 7_394_440, 18_622_552];

    fn metaroom_box_limits() -> CertifiedBox64Limits {
        CertifiedBox64Limits {
            max_values: 57_344,
            max_stored_f64: 114_688,
            max_weight_elements: 7_340_032,
            max_work_items: 20_000_000,
            max_scalar_products: 40_000_000,
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum BoxCoordinateClass {
        Inactive,
        Active,
        Unstable,
    }

    fn classify_box_coordinate(lower: f64, upper: f64) -> BoxCoordinateClass {
        debug_assert!(lower <= upper);
        if upper <= 0.0 {
            BoxCoordinateClass::Inactive
        } else if lower >= 0.0 {
            BoxCoordinateClass::Active
        } else {
            BoxCoordinateClass::Unstable
        }
    }

    fn exact_unconstrained_cz_classes(
        cz: &ConstrainedZonotope64,
        outward_hull: &CertifiedBox64,
    ) -> Vec<BoxCoordinateClass> {
        assert_eq!(cz.value_dim(), outward_hull.len());
        // Outward-stable coordinates cannot become unstable under the exact
        // dyadic radius. Recompute only the outward-unstable frontier so this
        // diagnostic agrees exactly with the production ReLU classifier at
        // zero boundaries without normalizing millions of irrelevant terms.
        let mut exact_radii: Vec<Option<BigRational>> = outward_hull
            .lower()
            .iter()
            .zip(outward_hull.upper())
            .enumerate()
            .map(|(coordinate, (&lower, &upper))| {
                (classify_box_coordinate(lower, upper) == BoxCoordinateClass::Unstable)
                    .then(|| BigRational::from_float(cz.box_remainder()[coordinate]).unwrap())
            })
            .collect();
        for generator in cz.generators() {
            for (coordinate, coefficient) in generator.entries() {
                let Some(radius) = &mut exact_radii[coordinate] else {
                    continue;
                };
                let term = BigRational::from_float(coefficient).unwrap();
                *radius += if coefficient.is_sign_negative() {
                    -term
                } else {
                    term
                };
            }
        }
        let zero = BigRational::from_integer(0.into());
        exact_radii
            .into_iter()
            .enumerate()
            .map(|(coordinate, radius)| {
                let Some(radius) = radius else {
                    return classify_box_coordinate(
                        outward_hull.lower()[coordinate],
                        outward_hull.upper()[coordinate],
                    );
                };
                let center = BigRational::from_float(cz.center()[coordinate]).unwrap();
                let upper = &center + &radius;
                if upper <= zero {
                    BoxCoordinateClass::Inactive
                } else if center - radius >= zero {
                    BoxCoordinateClass::Active
                } else {
                    BoxCoordinateClass::Unstable
                }
            })
            .collect()
    }

    #[derive(Clone, Copy, Debug)]
    struct WidthRatioQuantiles {
        minimum: f64,
        p10: f64,
        median: f64,
        p90: f64,
        maximum: f64,
    }

    impl WidthRatioQuantiles {
        fn emit(self) -> String {
            format!(
                "min={:.6},p10={:.6},median={:.6},p90={:.6},max={:.6}",
                self.minimum, self.p10, self.median, self.p90, self.maximum
            )
        }
    }

    fn width_ratio_quantiles(mut ratios: Vec<f64>) -> Option<WidthRatioQuantiles> {
        if ratios.is_empty() {
            return None;
        }
        ratios.sort_by(f64::total_cmp);
        let sample = |numerator: usize, denominator: usize| {
            let index = (ratios.len() - 1) * numerator / denominator;
            ratios[index]
        };
        Some(WidthRatioQuantiles {
            minimum: ratios[0],
            p10: sample(1, 10),
            median: sample(1, 2),
            p90: sample(9, 10),
            maximum: ratios[ratios.len() - 1],
        })
    }

    #[derive(Clone, Debug)]
    struct BoxReluComparison {
        stage: &'static str,
        value_count: usize,
        cz_active: usize,
        cz_inactive: usize,
        cz_unstable: usize,
        box_active: usize,
        box_inactive: usize,
        box_unstable: usize,
        box_only_active: usize,
        box_only_inactive: usize,
        cz_only_stable: usize,
        box_narrower: usize,
        cz_narrower: usize,
        equal_width: usize,
        box_over_cz_width: Option<WidthRatioQuantiles>,
        intersection_over_cz_width: Option<WidthRatioQuantiles>,
        ignored_cz_constraints: usize,
    }

    impl BoxReluComparison {
        fn emit(&self) {
            let box_ratios = self
                .box_over_cz_width
                .map_or_else(|| "none".to_string(), WidthRatioQuantiles::emit);
            let intersection_ratios = self
                .intersection_over_cz_width
                .map_or_else(|| "none".to_string(), WidthRatioQuantiles::emit);
            eprintln!(
                "Metaroom certified Box {stage}: values={values}, CZ(active={cz_active},inactive={cz_inactive},unstable={cz_unstable}), Box(active={box_active},inactive={box_inactive},unstable={box_unstable}), Box-only(active={box_only_active},inactive={box_only_inactive}), CZ-only-stable={cz_only_stable}, width-wins(Box={box_narrower},CZ={cz_narrower},equal={equal_width}), Box/CZ-width[{box_ratios}], intersection/CZ-width[{intersection_ratios}], ignored-CZ-constraints={ignored}",
                stage = self.stage,
                values = self.value_count,
                cz_active = self.cz_active,
                cz_inactive = self.cz_inactive,
                cz_unstable = self.cz_unstable,
                box_active = self.box_active,
                box_inactive = self.box_inactive,
                box_unstable = self.box_unstable,
                box_only_active = self.box_only_active,
                box_only_inactive = self.box_only_inactive,
                cz_only_stable = self.cz_only_stable,
                box_narrower = self.box_narrower,
                cz_narrower = self.cz_narrower,
                equal_width = self.equal_width,
                ignored = self.ignored_cz_constraints,
            );
        }
    }

    #[derive(Default)]
    struct MetaroomBoxDiagnostic {
        current: Option<CertifiedBox64>,
        relus: Vec<BoxReluComparison>,
        auxiliary_preactivations: Vec<CertifiedAuxiliaryBounds64>,
        conv_scalar_products: usize,
        affine_scalar_products: usize,
        terminal: Option<CertifiedBox64>,
    }

    impl MetaroomBoxDiagnostic {
        fn initialize(&mut self, input_box: &CertifiedInputBox) {
            assert!(self.current.is_none());
            assert!(self.auxiliary_preactivations.is_empty());
            self.current = Some(
                CertifiedBox64::from_certified_bounds(
                    input_box.lower(),
                    input_box.upper(),
                    metaroom_box_limits(),
                )
                .unwrap(),
            );
        }

        fn conv_relu(
            &mut self,
            stage: &'static str,
            input_shape: [usize; 3],
            weights: ndarray::ArrayView4<'_, f64>,
            bias: &[f64],
            spec: ConstrainedZonotopeConv2dSpec,
            cz_preactivation: &ConstrainedZonotope64,
        ) {
            let input = self.current.take().unwrap();
            let (preactivation, plan) = certified_box_conv2d_unwired(
                &input,
                input_shape,
                weights,
                bias,
                spec,
                metaroom_box_limits(),
            )
            .unwrap();
            self.conv_scalar_products = self
                .conv_scalar_products
                .checked_add(plan.scalar_products)
                .unwrap();
            self.record_relu(stage, &preactivation, cz_preactivation);
            self.current = Some(preactivation.relu_unwired(metaroom_box_limits()).unwrap());
        }

        fn affine_relu(
            &mut self,
            stage: &'static str,
            weights: ndarray::ArrayView2<'_, f64>,
            bias: &[f64],
            cz_preactivation: &ConstrainedZonotope64,
        ) {
            let input = self.current.take().unwrap();
            let (preactivation, plan) =
                certified_box_affine_unwired(&input, weights, bias, metaroom_box_limits()).unwrap();
            self.affine_scalar_products = self
                .affine_scalar_products
                .checked_add(plan.scalar_products)
                .unwrap();
            self.record_relu(stage, &preactivation, cz_preactivation);
            self.current = Some(preactivation.relu_unwired(metaroom_box_limits()).unwrap());
        }

        fn terminal_affine(&mut self, weights: ndarray::ArrayView2<'_, f64>, bias: &[f64]) {
            let input = self.current.take().unwrap();
            let (terminal, plan) =
                certified_box_affine_unwired(&input, weights, bias, metaroom_box_limits()).unwrap();
            self.affine_scalar_products = self
                .affine_scalar_products
                .checked_add(plan.scalar_products)
                .unwrap();
            self.terminal = Some(terminal);
        }

        fn record_relu(
            &mut self,
            stage: &'static str,
            box_bounds: &CertifiedBox64,
            cz_preactivation: &ConstrainedZonotope64,
        ) {
            let (cz_bounds, hull_plan) =
                unconstrained_zonotope_box_unwired(cz_preactivation, metaroom_box_limits())
                    .unwrap();
            assert_eq!(box_bounds.len(), cz_bounds.len());
            let cz_classes = exact_unconstrained_cz_classes(cz_preactivation, &cz_bounds);

            let mut comparison = BoxReluComparison {
                stage,
                value_count: box_bounds.len(),
                cz_active: 0,
                cz_inactive: 0,
                cz_unstable: 0,
                box_active: 0,
                box_inactive: 0,
                box_unstable: 0,
                box_only_active: 0,
                box_only_inactive: 0,
                cz_only_stable: 0,
                box_narrower: 0,
                cz_narrower: 0,
                equal_width: 0,
                box_over_cz_width: None,
                intersection_over_cz_width: None,
                ignored_cz_constraints: hull_plan.ignored_constraints,
            };
            let mut box_ratios = Vec::with_capacity(box_bounds.len());
            let mut intersection_ratios = Vec::with_capacity(box_bounds.len());
            for coordinate in 0..box_bounds.len() {
                let box_lower = box_bounds.lower()[coordinate];
                let box_upper = box_bounds.upper()[coordinate];
                let cz_lower = cz_bounds.lower()[coordinate];
                let cz_upper = cz_bounds.upper()[coordinate];
                let intersection_lower = box_lower.max(cz_lower);
                let intersection_upper = box_upper.min(cz_upper);
                assert!(
                    intersection_lower <= intersection_upper,
                    "independent sound enclosures are disjoint at {stage}[{coordinate}]: Box=[{box_lower}, {box_upper}], CZ=[{cz_lower}, {cz_upper}]"
                );

                let box_class = classify_box_coordinate(box_lower, box_upper);
                let cz_class = cz_classes[coordinate];
                match box_class {
                    BoxCoordinateClass::Active => comparison.box_active += 1,
                    BoxCoordinateClass::Inactive => comparison.box_inactive += 1,
                    BoxCoordinateClass::Unstable => comparison.box_unstable += 1,
                }
                match cz_class {
                    BoxCoordinateClass::Active => comparison.cz_active += 1,
                    BoxCoordinateClass::Inactive => comparison.cz_inactive += 1,
                    BoxCoordinateClass::Unstable => comparison.cz_unstable += 1,
                }
                if cz_class == BoxCoordinateClass::Unstable {
                    match box_class {
                        BoxCoordinateClass::Active => comparison.box_only_active += 1,
                        BoxCoordinateClass::Inactive => comparison.box_only_inactive += 1,
                        BoxCoordinateClass::Unstable => {}
                    }
                } else if box_class == BoxCoordinateClass::Unstable {
                    comparison.cz_only_stable += 1;
                }

                let box_width = box_upper - box_lower;
                let cz_width = cz_upper - cz_lower;
                match box_width.total_cmp(&cz_width) {
                    std::cmp::Ordering::Less => comparison.box_narrower += 1,
                    std::cmp::Ordering::Greater => comparison.cz_narrower += 1,
                    std::cmp::Ordering::Equal => comparison.equal_width += 1,
                }
                if cz_width > 0.0 && cz_width.is_finite() {
                    let box_ratio = box_width / cz_width;
                    let intersection_ratio = (intersection_upper - intersection_lower) / cz_width;
                    if box_ratio.is_finite() {
                        box_ratios.push(box_ratio);
                    }
                    if intersection_ratio.is_finite() {
                        intersection_ratios.push(intersection_ratio);
                    }
                }
            }
            comparison.box_over_cz_width = width_ratio_quantiles(box_ratios);
            comparison.intersection_over_cz_width = width_ratio_quantiles(intersection_ratios);
            assert_eq!(
                comparison.cz_active + comparison.cz_inactive + comparison.cz_unstable,
                comparison.value_count
            );
            assert_eq!(
                comparison.box_active + comparison.box_inactive + comparison.box_unstable,
                comparison.value_count
            );
            comparison.emit();
            self.auxiliary_preactivations
                .push(CertifiedAuxiliaryBounds64::try_from_certified_box(box_bounds).unwrap());
            self.relus.push(comparison);
        }
    }

    fn assert_trunk_predicate_footprint(
        mode: FullTrunkPredicateMode,
        output: &ConstrainedZonotope64,
        stage: usize,
    ) {
        let expected_rows = mode.select(0, PROJECTED_RELU_ROWS[stage]);
        let expected_elements = mode.select(0, PROJECTED_RELU_ELEMENTS[stage]);
        assert_eq!(output.constraint_count(), expected_rows);
        assert_eq!(output.rhs().len(), expected_rows);
        assert_eq!(output.constraints().len(), expected_elements);
        assert_eq!(
            expected_elements,
            expected_rows.checked_mul(output.alpha_dim()).unwrap()
        );
    }

    fn assert_domain_constraint_storage(output: &ConstrainedZonotope64) {
        assert_eq!(output.rhs().len(), output.constraint_count());
        assert_eq!(output.constraints().nrows(), output.constraint_count());
        assert_eq!(output.constraints().ncols(), output.alpha_dim());
        assert_eq!(
            output.constraints().len(),
            output
                .constraint_count()
                .checked_mul(output.alpha_dim())
                .unwrap()
        );
    }

    #[test]
    #[ignore = "guarded real-model qualification; run explicitly under ny-safe-gpu-run"]
    fn real_metaroom_119_measures_full_conv_relu_trunk_resources() {
        let _ = qualify_real_metaroom_119_full_conv_relu_trunk(FullTrunkPredicateMode::Preserve);
    }

    #[test]
    #[ignore = "guarded projected real-model qualification; run explicitly under ny-safe-gpu-run"]
    fn real_metaroom_119_measures_projected_full_conv_relu_trunk_resources() {
        let _ = qualify_real_metaroom_119_full_conv_relu_trunk(
            FullTrunkPredicateMode::ProjectedReluGeometry,
        );
    }

    fn qualify_real_metaroom_119_full_conv_relu_trunk(
        mode: FullTrunkPredicateMode,
    ) -> Option<QualifiedMetaroomConvReluTrunk> {
        qualify_real_metaroom_119_full_conv_relu_trunk_impl(mode, None, None, false)
    }

    fn qualify_real_metaroom_119_full_conv_relu_trunk_with_box(
        mode: FullTrunkPredicateMode,
        diagnostic: &mut MetaroomBoxDiagnostic,
    ) -> Option<QualifiedMetaroomConvReluTrunk> {
        qualify_real_metaroom_119_full_conv_relu_trunk_impl(mode, Some(diagnostic), None, false)
    }

    fn qualify_real_metaroom_119_full_conv_relu_trunk_with_box_counterfactual(
        mode: FullTrunkPredicateMode,
        diagnostic: &mut MetaroomBoxDiagnostic,
    ) -> Option<QualifiedMetaroomConvReluTrunk> {
        qualify_real_metaroom_119_full_conv_relu_trunk_impl(mode, Some(diagnostic), None, true)
    }

    fn qualify_real_metaroom_119_full_conv_relu_trunk_with_auxiliary_trace(
        mode: FullTrunkPredicateMode,
        auxiliary_trace: &[CertifiedAuxiliaryBounds64],
    ) -> Option<QualifiedMetaroomConvReluTrunk> {
        qualify_real_metaroom_119_full_conv_relu_trunk_impl(
            mode,
            None,
            Some(auxiliary_trace),
            false,
        )
    }

    fn qualify_real_metaroom_119_full_conv_relu_trunk_impl(
        mode: FullTrunkPredicateMode,
        mut box_diagnostic: Option<&mut MetaroomBoxDiagnostic>,
        auxiliary_trace: Option<&[CertifiedAuxiliaryBounds64]>,
        measure_relu4_auxiliary_counterfactual: bool,
    ) -> Option<QualifiedMetaroomConvReluTrunk> {
        if let Some(trace) = auxiliary_trace {
            assert_eq!(trace.len(), 4);
        }
        assert!(!measure_relu4_auxiliary_counterfactual || box_diagnostic.is_some());
        assert!(!(measure_relu4_auxiliary_counterfactual && auxiliary_trace.is_some()));
        let is_hybrid = auxiliary_trace.is_some();
        let mut relu_resources = Vec::with_capacity(4);
        let root = std::env::var_os("NY_METAROOM_119_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../benchmarks/vnncomp2025/benchmarks/metaroom_2023")
            });
        let onnx = root.join("onnx/6cnn_ry_39_6_no_custom_OP.onnx");
        let vnnlib = root.join("vnnlib/spec_idx_119_eps_0.00000436.vnnlib");
        if !onnx.exists() || !vnnlib.exists() {
            return None;
        }

        let config = OnnxLoadConfig::default().with_raw_float32_initializer_provenance(true);
        let model = load_onnx_with_config(&onnx, &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (vnnlib_spec, box_) = load_vnnlib_with_certified_input_box(&vnnlib).unwrap();
        if let Some(diagnostic) = box_diagnostic.as_deref_mut() {
            diagnostic.initialize(&box_);
        }
        let first_limits = CertifiedCzReluStemLimits {
            stem: CertifiedCzStemLimits {
                max_graph_nodes: 32,
                max_graph_edges: 128,
                max_topology_work_items: 16_384,
                max_input_values: 5_376,
                max_input_alpha_dim: 161,
                max_input_generator_nonzeros: 161,
                max_input_stored_f64: 10_913,
                max_parameter_elements: 896,
                conv: ConstrainedZonotopeConv2dLimits {
                    max_value_count: 57_344,
                    max_alpha_dim: 161,
                    max_generator_nonzeros: 46_368,
                    max_weight_elements: 864,
                    max_kernel_visits: 1_548_288,
                    max_interval_products: 3_042_336,
                    max_constraint_count: 0,
                    max_constraint_elements: 0,
                },
            },
            relu: ReluTransformLimits {
                max_value_dim: 57_344,
                max_output_alpha_dim: 637,
                max_constraints: mode.select(0, PROJECTED_RELU_ROWS[0]),
                max_constraint_elements: mode.select(0, PROJECTED_RELU_ELEMENTS[0]),
                max_generator_nnz: 43_676,
                max_unstable: 476,
                max_exact_terms: mode.select(145_648, 191_704),
            },
        };
        let mut conv1_for_auxiliary = None;
        if box_diagnostic.is_some() || is_hybrid {
            // The public stem returns the post-ReLU CZ. Re-run only its bounded
            // Conv1 prefix here to expose the exact same preactivation for the
            // independent Box comparison; this path remains ignored/unwired.
            let (conv1_output, conv1_plan) =
                propagate_direct_onnx_conv2d_unwired(&model, &graph, &box_, first_limits.stem)
                    .unwrap();
            let (_, graph_conv1) = direct_graph_conv(&graph).unwrap();
            let (weights, bias, _) = promote_parameters(graph_conv1, first_limits.stem).unwrap();
            let conv1_spec = ConstrainedZonotopeConv2dSpec {
                stride: [graph_conv1.stride.0, graph_conv1.stride.1],
                padding: [
                    graph_conv1.padding.0,
                    graph_conv1.padding.1,
                    graph_conv1.padding.0,
                    graph_conv1.padding.1,
                ],
                dilation: [graph_conv1.dilation.0, graph_conv1.dilation.1],
                groups: graph_conv1.groups,
            };
            if let Some(diagnostic) = box_diagnostic.as_deref_mut() {
                diagnostic.conv_relu(
                    "ReLU1",
                    conv1_plan.input_shape,
                    weights.view(),
                    &bias,
                    conv1_spec,
                    &conv1_output,
                );
            }
            conv1_for_auxiliary = Some(conv1_output);
        }

        let (baseline_relu1_output, first_plan) = mode
            .propagate_first_relu(&model, &graph, &box_, first_limits)
            .unwrap();
        let relu1_output = match auxiliary_trace {
            Some(trace) => mode
                .transform_relu_with_auxiliary(
                    conv1_for_auxiliary.as_ref().unwrap(),
                    &trace[0],
                    first_limits.relu,
                )
                .unwrap(),
            None => baseline_relu1_output,
        };
        let relu1_resource =
            MetaroomReluResources::capture("ReLU1", first_plan.input_alpha_dim, &relu1_output);
        assert_eq!(first_plan.stem.conv.interval_products, 2_873_856);
        assert_eq!(first_plan.output_generator_nonzeros, 15_348);
        assert_eq!(first_plan.nonzero_remainder_count, 19_960);
        if !is_hybrid {
            assert_eq!(relu1_output.alpha_dim(), 637);
            assert_trunk_predicate_footprint(mode, &relu1_output, 0);
        }
        assert_domain_constraint_storage(&relu1_output);
        relu_resources.push(relu1_resource);

        let graph_conv2_consumers: Vec<_> = graph
            .node_names()
            .iter()
            .filter_map(|name| {
                graph.node(name).filter(|node| {
                    node.inputs()
                        .iter()
                        .any(|input| input == &first_plan.relu_node)
                })
            })
            .collect();
        assert_eq!(graph_conv2_consumers.len(), 1);
        let graph_conv2_node = graph_conv2_consumers[0];
        assert_eq!(graph_conv2_node.inputs(), [first_plan.relu_node.as_str()]);
        let Layer::Conv2d(graph_conv2) = graph_conv2_node.layer() else {
            panic!(
                "first ReLU must feed Conv2d, got {}",
                graph_conv2_node.layer().layer_type()
            );
        };

        let model_relu1 = model
            .network
            .layers
            .iter()
            .find(|layer| layer.name == first_plan.relu_node)
            .unwrap();
        assert_eq!(model_relu1.outputs.len(), 1);
        let model_conv2_consumers: Vec<_> = model
            .network
            .layers
            .iter()
            .filter(|layer| {
                layer
                    .inputs
                    .iter()
                    .any(|input| input == &model_relu1.outputs[0])
            })
            .collect();
        assert_eq!(model_conv2_consumers.len(), 1);
        let model_conv2 = model_conv2_consumers[0];
        assert_eq!(model_conv2.name, graph_conv2_node.name());
        assert_eq!(model_conv2.layer_type, LayerType::Conv2d);
        assert_eq!(model_conv2.inputs.first(), Some(&model_relu1.outputs[0]));

        let raw_conv2 = raw_conv_parameters(&model, model_conv2).unwrap();
        validate_raw_conv_shape(&raw_conv2).unwrap();
        validate_normalized_graph_conv_shape(graph_conv2, 32).unwrap();
        require_original_float32_parameters(&model, &raw_conv2).unwrap();
        ensure_graph_conv_matches_raw(graph_conv2, &raw_conv2).unwrap();
        let raw_parameter_elements =
            raw_conv2.kernel.len() + raw_conv2.bias.as_ref().map_or(0, |bias| bias.len());
        assert_eq!(raw_parameter_elements, 9_248);
        let mut promotion_limits = first_limits.stem;
        promotion_limits.max_parameter_elements = 9_248;
        let (weights, bias, parameter_elements) =
            promote_parameters(graph_conv2, promotion_limits).unwrap();
        assert_eq!(parameter_elements, raw_parameter_elements);

        let input_shape = first_plan.stem.conv.output_shape;
        assert_eq!(input_shape, [32, 32, 56]);
        assert_eq!(graph_conv2.input_shape, Some((32, 56)));
        assert_eq!(graph_conv2.in_channels(), input_shape[0]);
        let conv2_spec = ConstrainedZonotopeConv2dSpec {
            stride: [graph_conv2.stride.0, graph_conv2.stride.1],
            padding: [
                graph_conv2.padding.0,
                graph_conv2.padding.1,
                graph_conv2.padding.0,
                graph_conv2.padding.1,
            ],
            dilation: [graph_conv2.dilation.0, graph_conv2.dilation.1],
            groups: graph_conv2.groups,
        };
        let conv2_limits = ConstrainedZonotopeConv2dLimits {
            max_value_count: 57_344,
            max_alpha_dim: 637,
            max_generator_nonzeros: 250_688,
            max_weight_elements: 9_216,
            max_kernel_visits: 16_515_072,
            max_interval_products: 25_831_328,
            max_constraint_count: mode.select(0, PROJECTED_RELU_ROWS[0]),
            max_constraint_elements: mode.select(0, PROJECTED_RELU_ELEMENTS[0]),
        };
        let (conv2_output, conv2_plan) = constrained_zonotope_conv2d_unwired(
            &relu1_output,
            input_shape,
            weights.view(),
            &bias,
            conv2_spec,
            conv2_limits,
        )
        .unwrap();
        eprintln!("real Metaroom Conv2 plan: {conv2_plan:#?}");
        assert_eq!(conv2_plan.input_shape, [32, 32, 56]);
        assert_eq!(conv2_plan.output_shape, [32, 32, 56]);
        assert_eq!(conv2_plan.weight_shape, [32, 32, 3, 3]);
        assert_eq!(conv2_plan.weight_elements, 9_216);
        assert_eq!(conv2_plan.kernel_visits, 16_515_072);
        if !is_hybrid {
            assert_eq!(conv2_plan.alpha_dim, 637);
            assert_eq!(conv2_plan.input_generator_nonzeros, 15_348);
            assert_eq!(conv2_plan.output_generator_nonzeros, 250_688);
            assert_eq!(conv2_plan.interval_products, 25_831_328);
            assert_trunk_predicate_footprint(mode, &conv2_output, 0);
        }
        assert_domain_constraint_storage(&conv2_output);
        if let Some(diagnostic) = box_diagnostic.as_deref_mut() {
            diagnostic.conv_relu(
                "ReLU2",
                input_shape,
                weights.view(),
                &bias,
                conv2_spec,
                &conv2_output,
            );
        }

        let graph_relu2_consumers: Vec<_> = graph
            .node_names()
            .iter()
            .filter_map(|name| {
                graph.node(name).filter(|node| {
                    node.inputs()
                        .iter()
                        .any(|input| input == graph_conv2_node.name())
                })
            })
            .collect();
        assert_eq!(graph_relu2_consumers.len(), 1);
        let graph_relu2 = graph_relu2_consumers[0];
        assert_eq!(graph_relu2.inputs(), [graph_conv2_node.name()]);
        assert!(matches!(graph_relu2.layer(), Layer::ReLU(_)));

        assert_eq!(model_conv2.outputs.len(), 1);
        let model_relu2_consumers: Vec<_> = model
            .network
            .layers
            .iter()
            .filter(|layer| {
                layer
                    .inputs
                    .iter()
                    .any(|input| input == &model_conv2.outputs[0])
            })
            .collect();
        assert_eq!(model_relu2_consumers.len(), 1);
        let model_relu2 = model_relu2_consumers[0];
        assert_eq!(model_relu2.name, graph_relu2.name());
        assert_eq!(model_relu2.layer_type, LayerType::ReLU);
        assert_eq!(model_relu2.inputs, vec![model_conv2.outputs[0].clone()]);
        assert_eq!(model_relu2.outputs.len(), 1);
        assert!(!model_relu2.outputs[0].is_empty());
        assert!(model_relu2.weights.is_none());
        assert!(model_relu2.attributes.is_empty());

        let relu2_limits = ReluTransformLimits {
            max_value_dim: 57_344,
            max_output_alpha_dim: 1_324,
            max_constraints: mode.select(0, PROJECTED_RELU_ROWS[1]),
            max_constraint_elements: mode.select(0, PROJECTED_RELU_ELEMENTS[1]),
            max_generator_nnz: 251_375,
            max_unstable: 687,
            max_exact_terms: mode.select(561_468, 816_278),
        };
        let relu2_output = match auxiliary_trace {
            Some(trace) => mode
                .transform_relu_with_auxiliary(&conv2_output, &trace[1], relu2_limits)
                .unwrap(),
            None => mode.transform_relu(&conv2_output, relu2_limits).unwrap(),
        };
        let relu2_generator_nonzeros = relu2_output
            .generators()
            .iter()
            .map(ny_mip::SparseGenerator64::nnz)
            .sum::<usize>();
        let relu2_nonzero_remainders = relu2_output
            .box_remainder()
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        let relu2_unstable = relu2_output.alpha_dim() - conv2_output.alpha_dim();
        eprintln!(
            "real Metaroom ReLU2 plan: input_alpha={}, output_alpha={}, unstable={}, output_generator_nonzeros={}, nonzero_remainders={}",
            conv2_output.alpha_dim(),
            relu2_output.alpha_dim(),
            relu2_unstable,
            relu2_generator_nonzeros,
            relu2_nonzero_remainders
        );
        if !is_hybrid {
            assert_eq!(conv2_output.alpha_dim(), 637);
            assert_eq!(relu2_output.alpha_dim(), 1_324);
            assert_eq!(relu2_unstable, 687);
            assert_eq!(relu2_generator_nonzeros, 71_042);
            assert_eq!(relu2_nonzero_remainders, 12_172);
            assert_trunk_predicate_footprint(mode, &relu2_output, 1);
        }
        assert!(relu2_generator_nonzeros <= conv2_plan.output_generator_nonzeros + relu2_unstable);
        assert_domain_constraint_storage(&relu2_output);
        relu_resources.push(MetaroomReluResources::capture(
            "ReLU2",
            conv2_output.alpha_dim(),
            &relu2_output,
        ));

        let graph_conv3_consumers: Vec<_> = graph
            .node_names()
            .iter()
            .filter_map(|name| {
                graph.node(name).filter(|node| {
                    node.inputs()
                        .iter()
                        .any(|input| input == graph_relu2.name())
                })
            })
            .collect();
        assert_eq!(graph_conv3_consumers.len(), 1);
        let graph_conv3_node = graph_conv3_consumers[0];
        assert_eq!(graph_conv3_node.inputs(), [graph_relu2.name()]);
        let Layer::Conv2d(graph_conv3) = graph_conv3_node.layer() else {
            panic!(
                "second ReLU must feed Conv2d, got {}",
                graph_conv3_node.layer().layer_type()
            );
        };

        assert_eq!(model_relu2.outputs.len(), 1);
        let model_conv3_consumers: Vec<_> = model
            .network
            .layers
            .iter()
            .filter(|layer| {
                layer
                    .inputs
                    .iter()
                    .any(|input| input == &model_relu2.outputs[0])
            })
            .collect();
        assert_eq!(model_conv3_consumers.len(), 1);
        let model_conv3 = model_conv3_consumers[0];
        assert_eq!(model_conv3.name, graph_conv3_node.name());
        assert_eq!(model_conv3.layer_type, LayerType::Conv2d);
        assert_eq!(model_conv3.inputs.first(), Some(&model_relu2.outputs[0]));

        let raw_conv3 = raw_conv_parameters(&model, model_conv3).unwrap();
        validate_raw_conv_shape(&raw_conv3).unwrap();
        validate_normalized_graph_conv_shape(graph_conv3, 32).unwrap();
        require_original_float32_parameters(&model, &raw_conv3).unwrap();
        ensure_graph_conv_matches_raw(graph_conv3, &raw_conv3).unwrap();
        let raw_parameter_elements =
            raw_conv3.kernel.len() + raw_conv3.bias.as_ref().map_or(0, |bias| bias.len());
        assert_eq!(raw_parameter_elements, 18_496);
        let mut promotion_limits = first_limits.stem;
        promotion_limits.max_parameter_elements = 18_496;
        let (weights, bias, parameter_elements) =
            promote_parameters(graph_conv3, promotion_limits).unwrap();
        assert_eq!(parameter_elements, raw_parameter_elements);

        let input_shape = conv2_plan.output_shape;
        assert_eq!(input_shape, [32, 32, 56]);
        assert_eq!(graph_conv3.input_shape, Some((32, 56)));
        assert_eq!(graph_conv3.in_channels(), input_shape[0]);
        let conv3_spec = ConstrainedZonotopeConv2dSpec {
            stride: [graph_conv3.stride.0, graph_conv3.stride.1],
            padding: [
                graph_conv3.padding.0,
                graph_conv3.padding.1,
                graph_conv3.padding.0,
                graph_conv3.padding.1,
            ],
            dilation: [graph_conv3.dilation.0, graph_conv3.dilation.1],
            groups: graph_conv3.groups,
        };
        let conv3_limits = ConstrainedZonotopeConv2dLimits {
            max_value_count: 57_344,
            max_alpha_dim: 1_324,
            max_generator_nonzeros: 390_656,
            max_weight_elements: 18_432,
            max_kernel_visits: 8_257_536,
            max_interval_products: 19_777_728,
            max_constraint_count: mode.select(0, PROJECTED_RELU_ROWS[1]),
            max_constraint_elements: mode.select(0, PROJECTED_RELU_ELEMENTS[1]),
        };
        let (conv3_output, conv3_plan) = constrained_zonotope_conv2d_unwired(
            &relu2_output,
            input_shape,
            weights.view(),
            &bias,
            conv3_spec,
            conv3_limits,
        )
        .unwrap();
        eprintln!("real Metaroom Conv3 plan: {conv3_plan:#?}");
        assert_eq!(conv3_plan.input_shape, [32, 32, 56]);
        assert_eq!(conv3_plan.output_shape, [64, 16, 28]);
        assert_eq!(conv3_plan.weight_shape, [64, 32, 3, 3]);
        assert_eq!(conv3_plan.weight_elements, 18_432);
        assert_eq!(conv3_plan.kernel_visits, 8_257_536);
        if !is_hybrid {
            assert_eq!(conv3_plan.alpha_dim, 1_324);
            assert_eq!(conv3_plan.input_generator_nonzeros, 71_042);
            assert_eq!(conv3_plan.output_generator_nonzeros, 390_656);
            assert_eq!(conv3_plan.interval_products, 19_777_728);
            assert_trunk_predicate_footprint(mode, &conv3_output, 1);
        }
        assert_domain_constraint_storage(&conv3_output);
        if let Some(diagnostic) = box_diagnostic.as_deref_mut() {
            diagnostic.conv_relu(
                "ReLU3",
                input_shape,
                weights.view(),
                &bias,
                conv3_spec,
                &conv3_output,
            );
        }

        let graph_relu3_consumers: Vec<_> = graph
            .node_names()
            .iter()
            .filter_map(|name| {
                graph.node(name).filter(|node| {
                    node.inputs()
                        .iter()
                        .any(|input| input == graph_conv3_node.name())
                })
            })
            .collect();
        assert_eq!(graph_relu3_consumers.len(), 1);
        let graph_relu3 = graph_relu3_consumers[0];
        assert_eq!(graph_relu3.inputs(), [graph_conv3_node.name()]);
        assert!(matches!(graph_relu3.layer(), Layer::ReLU(_)));

        assert_eq!(model_conv3.outputs.len(), 1);
        let model_relu3_consumers: Vec<_> = model
            .network
            .layers
            .iter()
            .filter(|layer| {
                layer
                    .inputs
                    .iter()
                    .any(|input| input == &model_conv3.outputs[0])
            })
            .collect();
        assert_eq!(model_relu3_consumers.len(), 1);
        let model_relu3 = model_relu3_consumers[0];
        assert_eq!(model_relu3.name, graph_relu3.name());
        assert_eq!(model_relu3.layer_type, LayerType::ReLU);
        assert_eq!(model_relu3.inputs, vec![model_conv3.outputs[0].clone()]);
        assert_eq!(model_relu3.outputs.len(), 1);
        assert!(!model_relu3.outputs[0].is_empty());
        assert!(model_relu3.weights.is_none());
        assert!(model_relu3.attributes.is_empty());

        let relu3_limits = ReluTransformLimits {
            max_value_dim: 28_672,
            max_output_alpha_dim: 2_005,
            max_constraints: mode.select(0, PROJECTED_RELU_ROWS[2]),
            max_constraint_elements: mode.select(0, PROJECTED_RELU_ELEMENTS[2]),
            max_generator_nnz: 391_337,
            max_unstable: 681,
            max_exact_terms: mode.select(812_708, 1_207_450),
        };
        let relu3_output = match auxiliary_trace {
            Some(trace) => mode
                .transform_relu_with_auxiliary(&conv3_output, &trace[2], relu3_limits)
                .unwrap(),
            None => mode.transform_relu(&conv3_output, relu3_limits).unwrap(),
        };
        let relu3_generator_nonzeros = relu3_output
            .generators()
            .iter()
            .map(ny_mip::SparseGenerator64::nnz)
            .sum::<usize>();
        let relu3_nonzero_remainders = relu3_output
            .box_remainder()
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        let relu3_unstable = relu3_output.alpha_dim() - conv3_output.alpha_dim();
        eprintln!(
            "real Metaroom ReLU3 plan: input_alpha={}, output_alpha={}, unstable={}, output_generator_nonzeros={}, nonzero_remainders={}",
            conv3_output.alpha_dim(),
            relu3_output.alpha_dim(),
            relu3_unstable,
            relu3_generator_nonzeros,
            relu3_nonzero_remainders
        );
        if !is_hybrid {
            assert_eq!(conv3_output.alpha_dim(), 1_324);
            assert_eq!(relu3_output.alpha_dim(), 2_005);
            assert_eq!(relu3_unstable, 681);
            assert_eq!(relu3_generator_nonzeros, 167_447);
            assert_eq!(relu3_nonzero_remainders, 7_761);
            assert_trunk_predicate_footprint(mode, &relu3_output, 2);
        }
        assert!(relu3_generator_nonzeros <= conv3_plan.output_generator_nonzeros + relu3_unstable);
        assert_domain_constraint_storage(&relu3_output);
        relu_resources.push(MetaroomReluResources::capture(
            "ReLU3",
            conv3_output.alpha_dim(),
            &relu3_output,
        ));

        let graph_conv4_consumers: Vec<_> = graph
            .node_names()
            .iter()
            .filter_map(|name| {
                graph.node(name).filter(|node| {
                    node.inputs()
                        .iter()
                        .any(|input| input == graph_relu3.name())
                })
            })
            .collect();
        assert_eq!(graph_conv4_consumers.len(), 1);
        let graph_conv4_node = graph_conv4_consumers[0];
        assert_eq!(graph_conv4_node.inputs(), [graph_relu3.name()]);
        let Layer::Conv2d(graph_conv4) = graph_conv4_node.layer() else {
            panic!(
                "third ReLU must feed Conv2d, got {}",
                graph_conv4_node.layer().layer_type()
            );
        };

        assert_eq!(model_relu3.outputs.len(), 1);
        let model_conv4_consumers: Vec<_> = model
            .network
            .layers
            .iter()
            .filter(|layer| {
                layer
                    .inputs
                    .iter()
                    .any(|input| input == &model_relu3.outputs[0])
            })
            .collect();
        assert_eq!(model_conv4_consumers.len(), 1);
        let model_conv4 = model_conv4_consumers[0];
        assert_eq!(model_conv4.name, graph_conv4_node.name());
        assert_eq!(model_conv4.layer_type, LayerType::Conv2d);
        assert_eq!(model_conv4.inputs.first(), Some(&model_relu3.outputs[0]));

        let raw_conv4 = raw_conv_parameters(&model, model_conv4).unwrap();
        validate_raw_conv_shape(&raw_conv4).unwrap();
        validate_normalized_graph_conv_shape(graph_conv4, 64).unwrap();
        require_original_float32_parameters(&model, &raw_conv4).unwrap();
        ensure_graph_conv_matches_raw(graph_conv4, &raw_conv4).unwrap();
        let raw_parameter_elements =
            raw_conv4.kernel.len() + raw_conv4.bias.as_ref().map_or(0, |bias| bias.len());
        assert_eq!(raw_parameter_elements, 36_928);
        let mut promotion_limits = first_limits.stem;
        promotion_limits.max_parameter_elements = 36_928;
        let (weights, bias, parameter_elements) =
            promote_parameters(graph_conv4, promotion_limits).unwrap();
        assert_eq!(parameter_elements, raw_parameter_elements);

        let input_shape = conv3_plan.output_shape;
        assert_eq!(input_shape, [64, 16, 28]);
        assert_eq!(graph_conv4.input_shape, Some((16, 28)));
        assert_eq!(graph_conv4.in_channels(), input_shape[0]);
        let conv4_spec = ConstrainedZonotopeConv2dSpec {
            stride: [graph_conv4.stride.0, graph_conv4.stride.1],
            padding: [
                graph_conv4.padding.0,
                graph_conv4.padding.1,
                graph_conv4.padding.0,
                graph_conv4.padding.1,
            ],
            dilation: [graph_conv4.dilation.0, graph_conv4.dilation.1],
            groups: graph_conv4.groups,
        };
        let conv4_limits = ConstrainedZonotopeConv2dLimits {
            max_value_count: 28_672,
            max_alpha_dim: 2_005,
            max_generator_nonzeros: 1_739_328,
            max_weight_elements: 36_864,
            max_kernel_visits: 16_515_072,
            max_interval_products: 113_877_632,
            max_constraint_count: mode.select(0, PROJECTED_RELU_ROWS[2]),
            max_constraint_elements: mode.select(0, PROJECTED_RELU_ELEMENTS[2]),
        };
        let (conv4_output, conv4_plan) = constrained_zonotope_conv2d_unwired(
            &relu3_output,
            input_shape,
            weights.view(),
            &bias,
            conv4_spec,
            conv4_limits,
        )
        .unwrap();
        eprintln!("real Metaroom Conv4 plan: {conv4_plan:#?}");
        assert_eq!(conv4_plan.input_shape, [64, 16, 28]);
        assert_eq!(conv4_plan.output_shape, [64, 16, 28]);
        assert_eq!(conv4_plan.weight_shape, [64, 64, 3, 3]);
        assert_eq!(conv4_plan.weight_elements, 36_864);
        assert_eq!(conv4_plan.kernel_visits, 16_515_072);
        if !is_hybrid {
            assert_eq!(conv4_plan.alpha_dim, 2_005);
            assert_eq!(conv4_plan.input_generator_nonzeros, 167_447);
            assert_eq!(conv4_plan.output_generator_nonzeros, 1_739_328);
            assert_eq!(conv4_plan.interval_products, 113_877_632);
            assert_trunk_predicate_footprint(mode, &conv4_output, 2);
        }
        assert_domain_constraint_storage(&conv4_output);
        if let Some(diagnostic) = box_diagnostic.as_deref_mut() {
            diagnostic.conv_relu(
                "ReLU4",
                input_shape,
                weights.view(),
                &bias,
                conv4_spec,
                &conv4_output,
            );
        }
        let total_kernel_visits = first_plan
            .stem
            .conv
            .kernel_visits
            .checked_add(conv2_plan.kernel_visits)
            .and_then(|total| total.checked_add(conv3_plan.kernel_visits))
            .and_then(|total| total.checked_add(conv4_plan.kernel_visits))
            .unwrap();
        let total_interval_products = first_plan
            .stem
            .conv
            .interval_products
            .checked_add(conv2_plan.interval_products)
            .and_then(|total| total.checked_add(conv3_plan.interval_products))
            .and_then(|total| total.checked_add(conv4_plan.interval_products))
            .unwrap();
        assert_eq!(total_kernel_visits, 42_835_968);
        if !is_hybrid {
            assert_eq!(total_interval_products, 162_360_544);
        }

        let graph_relu4_consumers: Vec<_> = graph
            .node_names()
            .iter()
            .filter_map(|name| {
                graph.node(name).filter(|node| {
                    node.inputs()
                        .iter()
                        .any(|input| input == graph_conv4_node.name())
                })
            })
            .collect();
        assert_eq!(graph_relu4_consumers.len(), 1);
        let graph_relu4 = graph_relu4_consumers[0];
        assert_eq!(graph_relu4.inputs(), [graph_conv4_node.name()]);
        assert!(matches!(graph_relu4.layer(), Layer::ReLU(_)));

        assert_eq!(model_conv4.outputs.len(), 1);
        let model_relu4_consumers: Vec<_> = model
            .network
            .layers
            .iter()
            .filter(|layer| {
                layer
                    .inputs
                    .iter()
                    .any(|input| input == &model_conv4.outputs[0])
            })
            .collect();
        assert_eq!(model_relu4_consumers.len(), 1);
        let model_relu4 = model_relu4_consumers[0];
        assert_eq!(model_relu4.name, graph_relu4.name());
        assert_eq!(model_relu4.layer_type, LayerType::ReLU);
        assert_eq!(model_relu4.inputs, vec![model_conv4.outputs[0].clone()]);
        assert_eq!(model_relu4.outputs.len(), 1);
        assert!(!model_relu4.outputs[0].is_empty());
        assert!(model_relu4.weights.is_none());
        assert!(model_relu4.attributes.is_empty());

        let relu4_limits = ReluTransformLimits {
            max_value_dim: 28_672,
            max_output_alpha_dim: 3_133,
            max_constraints: mode.select(0, PROJECTED_RELU_ROWS[3]),
            max_constraint_elements: mode.select(0, PROJECTED_RELU_ELEMENTS[3]),
            max_generator_nnz: 1_740_456,
            max_unstable: 1_128,
            max_exact_terms: mode.select(3_511_840, 5_257_936),
        };
        let relu4_output = match auxiliary_trace {
            Some(trace) => mode
                .transform_relu_with_auxiliary(&conv4_output, &trace[3], relu4_limits)
                .unwrap(),
            None => mode.transform_relu(&conv4_output, relu4_limits).unwrap(),
        };
        let relu4_generator_nonzeros = relu4_output
            .generators()
            .iter()
            .map(ny_mip::SparseGenerator64::nnz)
            .sum::<usize>();
        let relu4_nonzero_remainders = relu4_output
            .box_remainder()
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        let relu4_unstable = relu4_output.alpha_dim() - conv4_output.alpha_dim();
        eprintln!(
            "real Metaroom ReLU4 plan: input_alpha={}, output_alpha={}, unstable={}, output_generator_nonzeros={}, nonzero_remainders={}",
            conv4_output.alpha_dim(),
            relu4_output.alpha_dim(),
            relu4_unstable,
            relu4_generator_nonzeros,
            relu4_nonzero_remainders
        );
        if !is_hybrid {
            assert_eq!(conv4_output.alpha_dim(), 2_005);
            assert_eq!(relu4_output.alpha_dim(), 3_133);
            assert_eq!(relu4_unstable, 1_128);
            assert_eq!(relu4_generator_nonzeros, 522_125);
            assert_eq!(relu4_nonzero_remainders, 1_440);
            assert_trunk_predicate_footprint(mode, &relu4_output, 3);
        }
        assert!(relu4_generator_nonzeros <= conv4_plan.output_generator_nonzeros + relu4_unstable);
        assert_domain_constraint_storage(&relu4_output);
        relu_resources.push(MetaroomReluResources::capture(
            "ReLU4",
            conv4_output.alpha_dim(),
            &relu4_output,
        ));

        let relu4_auxiliary_counterfactual = if measure_relu4_auxiliary_counterfactual {
            let diagnostic = box_diagnostic.as_deref().unwrap();
            assert_eq!(diagnostic.auxiliary_preactivations.len(), 4);
            let counterfactual = mode
                .transform_relu_with_auxiliary(
                    &conv4_output,
                    &diagnostic.auxiliary_preactivations[3],
                    relu4_limits,
                )
                .unwrap();
            Some(MetaroomReluResources::capture(
                "ReLU4",
                conv4_output.alpha_dim(),
                &counterfactual,
            ))
        } else {
            None
        };

        let graph_relu4_name = graph_relu4.name().to_string();
        let model_relu4_output = model_relu4.outputs[0].clone();
        Some(QualifiedMetaroomConvReluTrunk {
            model,
            graph,
            vnnlib_spec,
            relu4_output,
            graph_relu4_name,
            model_relu4_output,
            relu_resources,
            relu4_auxiliary_counterfactual,
        })
    }

    struct QualifiedMetaroomAffineTail {
        first_weights: Array2<f64>,
        first_bias: Vec<f64>,
        second_weights: Array2<f64>,
        second_bias: Vec<f64>,
    }

    fn unique_graph_consumer<'a>(graph: &'a GraphNetwork, producer: &str) -> &'a GraphNode {
        let consumers: Vec<_> = graph
            .node_names()
            .iter()
            .filter_map(|name| {
                graph
                    .node(name)
                    .filter(|node| node.inputs().iter().any(|input| input == producer))
            })
            .collect();
        assert_eq!(
            consumers.len(),
            1,
            "{producer} must have one graph consumer"
        );
        consumers[0]
    }

    fn unique_model_consumer<'a>(model: &'a OnnxModel, tensor: &str) -> &'a LayerSpec {
        let consumers: Vec<_> = model
            .network
            .layers
            .iter()
            .filter(|layer| layer.inputs.iter().any(|input| input == tensor))
            .collect();
        assert_eq!(consumers.len(), 1, "{tensor} must have one model consumer");
        consumers[0]
    }

    fn seal_and_promote_linear(
        model: &OnnxModel,
        model_linear: &LayerSpec,
        graph_linear: &ny_propagate::layers::LinearLayer,
        expected_weight_name: &str,
        expected_bias_name: &str,
        expected_output_count: usize,
        expected_input_count: usize,
        max_parameter_elements: usize,
    ) -> (Array2<f64>, Vec<f64>) {
        assert_eq!(model_linear.layer_type, LayerType::Linear);
        assert!(model_linear.weights.is_none());
        assert_eq!(
            model_linear.attributes,
            HashMap::from([
                ("alpha".to_string(), AttributeValue::Float(1.0)),
                ("beta".to_string(), AttributeValue::Float(1.0)),
                ("transB".to_string(), AttributeValue::Int(1)),
            ])
        );
        assert_eq!(
            model_linear.inputs.get(1).map(String::as_str),
            Some(expected_weight_name)
        );
        assert_eq!(
            model_linear.inputs.get(2).map(String::as_str),
            Some(expected_bias_name)
        );

        assert_eq!(
            model.original_float32_initializer_matches_current(expected_weight_name),
            Some(true)
        );
        assert_eq!(
            model.original_float32_initializer_matches_current(expected_bias_name),
            Some(true)
        );
        let raw_weights = model.weights.get(expected_weight_name).unwrap();
        let raw_bias = model.weights.get(expected_bias_name).unwrap();
        assert_eq!(
            raw_weights.shape(),
            [expected_output_count, expected_input_count]
        );
        assert_eq!(raw_bias.shape(), [expected_output_count]);
        let raw_weights = raw_weights.view().into_dimensionality::<Ix2>().unwrap();
        let raw_bias = raw_bias.view().into_dimensionality::<Ix1>().unwrap();

        assert_eq!(graph_linear.in_features(), expected_input_count);
        assert_eq!(graph_linear.out_features(), expected_output_count);
        assert_eq!(graph_linear.weight.shape(), raw_weights.shape());
        let graph_bias = graph_linear.bias.as_ref().unwrap();
        assert_eq!(graph_bias.shape(), raw_bias.shape());
        for output in 0..expected_output_count {
            for input in 0..expected_input_count {
                assert_eq!(
                    graph_linear.weight[[output, input]].to_bits(),
                    raw_weights[[output, input]].to_bits(),
                    "normalized Gemm weight bit drift at [{output}, {input}]"
                );
            }
            assert_eq!(
                graph_bias[output].to_bits(),
                raw_bias[output].to_bits(),
                "normalized Gemm bias bit drift at [{output}]"
            );
        }

        // The caller-selected cap is checked before either binary64 parameter
        // buffer is reserved. The raw/graph bit scans above borrow existing
        // loader storage and allocate no parameter-sized container.
        let weight_elements = expected_output_count
            .checked_mul(expected_input_count)
            .unwrap();
        let parameter_elements = weight_elements.checked_add(expected_output_count).unwrap();
        assert!(parameter_elements <= max_parameter_elements);

        let mut promoted_weights = Vec::new();
        promoted_weights
            .try_reserve_exact(weight_elements)
            .expect("reserve sealed binary64 Gemm weights");
        for output in 0..expected_output_count {
            for input in 0..expected_input_count {
                let value = raw_weights[[output, input]];
                assert!(value.is_finite());
                promoted_weights.push(f64::from(value));
            }
        }
        let promoted_weights = Array2::from_shape_vec(
            (expected_output_count, expected_input_count),
            promoted_weights,
        )
        .unwrap();

        let mut promoted_bias = Vec::new();
        promoted_bias
            .try_reserve_exact(expected_output_count)
            .expect("reserve sealed binary64 Gemm bias");
        for &value in &raw_bias {
            assert!(value.is_finite());
            promoted_bias.push(f64::from(value));
        }
        assert_eq!(promoted_bias.len(), expected_output_count);
        (promoted_weights, promoted_bias)
    }

    fn qualify_real_metaroom_119_affine_tail_topology(
        model: &OnnxModel,
        graph: &GraphNetwork,
        graph_relu4_name: &str,
        model_relu4_output: &str,
    ) -> QualifiedMetaroomAffineTail {
        assert_eq!(
            model.original_network_topology_matches_current(),
            Some(true)
        );
        assert_eq!(model.opset_imports().get(""), Some(&14));
        assert_eq!(model.network.layers.len(), 12);
        assert_eq!(graph.num_nodes(), 12);

        let expected_layers = [
            ("/0/Conv", LayerType::Conv2d),
            ("/1/Relu", LayerType::ReLU),
            ("/2/Conv", LayerType::Conv2d),
            ("/3/Relu", LayerType::ReLU),
            ("/4/Conv", LayerType::Conv2d),
            ("/5/Relu", LayerType::ReLU),
            ("/6/Conv", LayerType::Conv2d),
            ("/7/Relu", LayerType::ReLU),
            ("/8/Reshape", LayerType::Reshape),
            ("/9/Gemm", LayerType::Linear),
            ("/10/Relu", LayerType::ReLU),
            ("/11/Gemm", LayerType::Linear),
        ];
        for ((model_layer, graph_name), expected) in model
            .network
            .layers
            .iter()
            .zip(graph.node_names())
            .zip(&expected_layers)
        {
            assert_eq!(model_layer.name, expected.0);
            assert_eq!(&model_layer.layer_type, &expected.1);
            assert_eq!(graph_name, expected.0);
        }
        assert_eq!(graph_relu4_name, "/7/Relu");
        assert_eq!(model_relu4_output, "/7/Relu_output_0");

        for (name, expected_shape) in [
            ("/7/Relu_output_0", &[1, 64, 16, 28][..]),
            ("/8/Reshape_output_0", &[1, 28_672][..]),
            ("/9/Gemm_output_0", &[1, 256][..]),
            ("/10/Relu_output_0", &[1, 256][..]),
            ("output", &[1, 20][..]),
        ] {
            assert_eq!(
                model.tensor_shapes().get(name).map(Vec::as_slice),
                Some(expected_shape),
                "sealed tensor shape drift for {name}"
            );
        }
        assert_eq!(model.network.outputs.len(), 1);
        assert_eq!(model.network.outputs[0].name, "output");
        assert_eq!(model.network.outputs[0].shape, [1, 20]);
        assert_eq!(model.network.outputs[0].dtype, DataType::Float32);

        let model_reshape = unique_model_consumer(model, model_relu4_output);
        let graph_reshape = unique_graph_consumer(graph, graph_relu4_name);
        assert_eq!(model_reshape.name, "/8/Reshape");
        assert_eq!(model_reshape.layer_type, LayerType::Reshape);
        assert_eq!(
            model_reshape.inputs,
            [model_relu4_output, "/8/Constant_output_0"]
        );
        assert_eq!(model_reshape.outputs, ["/8/Reshape_output_0"]);
        assert!(model_reshape.weights.is_none());
        assert_eq!(
            model_reshape.attributes,
            HashMap::from([("allowzero".to_string(), AttributeValue::Int(0))])
        );
        let reshape_shape = model.weights.get_integers("/8/Constant_output_0").unwrap();
        assert_eq!(reshape_shape.shape(), [2]);
        assert_eq!(reshape_shape.iter().copied().collect::<Vec<_>>(), [1, -1]);
        assert_eq!(graph_reshape.name(), model_reshape.name);
        assert_eq!(graph_reshape.inputs(), [graph_relu4_name]);
        let Layer::Reshape(graph_reshape_layer) = graph_reshape.layer() else {
            panic!("sealed /8/Reshape must remain a normalized Reshape");
        };
        assert_eq!(graph_reshape_layer.target_shape, [-1]);
        assert_eq!(
            graph_reshape_layer
                .compute_output_shape(&[64, 16, 28])
                .unwrap(),
            [28_672]
        );

        let model_linear1 = unique_model_consumer(model, &model_reshape.outputs[0]);
        let graph_linear1 = unique_graph_consumer(graph, graph_reshape.name());
        assert_eq!(model_linear1.name, "/9/Gemm");
        assert_eq!(
            model_linear1.inputs,
            ["/8/Reshape_output_0", "9.weight", "9.bias"]
        );
        assert_eq!(model_linear1.outputs, ["/9/Gemm_output_0"]);
        assert_eq!(graph_linear1.name(), model_linear1.name);
        assert_eq!(graph_linear1.inputs(), [graph_reshape.name()]);
        let Layer::Linear(graph_linear1_layer) = graph_linear1.layer() else {
            panic!("sealed /9/Gemm must remain a normalized Linear");
        };
        let (first_weights, first_bias) = seal_and_promote_linear(
            model,
            model_linear1,
            graph_linear1_layer,
            "9.weight",
            "9.bias",
            256,
            28_672,
            7_340_288,
        );
        assert_eq!(first_weights.len() + first_bias.len(), 7_340_288);

        let model_relu5 = unique_model_consumer(model, &model_linear1.outputs[0]);
        let graph_relu5 = unique_graph_consumer(graph, graph_linear1.name());
        assert_eq!(model_relu5.name, "/10/Relu");
        assert_eq!(model_relu5.layer_type, LayerType::ReLU);
        assert_eq!(model_relu5.inputs, ["/9/Gemm_output_0"]);
        assert_eq!(model_relu5.outputs, ["/10/Relu_output_0"]);
        assert!(model_relu5.weights.is_none());
        assert!(model_relu5.attributes.is_empty());
        assert_eq!(graph_relu5.name(), model_relu5.name);
        assert_eq!(graph_relu5.inputs(), [graph_linear1.name()]);
        assert!(matches!(graph_relu5.layer(), Layer::ReLU(_)));

        let model_linear2 = unique_model_consumer(model, &model_relu5.outputs[0]);
        let graph_linear2 = unique_graph_consumer(graph, graph_relu5.name());
        assert_eq!(model_linear2.name, "/11/Gemm");
        assert_eq!(
            model_linear2.inputs,
            ["/10/Relu_output_0", "11.weight", "11.bias"]
        );
        assert_eq!(model_linear2.outputs, ["output"]);
        assert_eq!(graph_linear2.name(), model_linear2.name);
        assert_eq!(graph_linear2.inputs(), [graph_relu5.name()]);
        assert_eq!(graph.output_name(), graph_linear2.name());
        let Layer::Linear(graph_linear2_layer) = graph_linear2.layer() else {
            panic!("sealed /11/Gemm must remain a normalized Linear");
        };
        let (second_weights, second_bias) = seal_and_promote_linear(
            model,
            model_linear2,
            graph_linear2_layer,
            "11.weight",
            "11.bias",
            20,
            256,
            5_140,
        );
        assert_eq!(second_weights.len() + second_bias.len(), 5_140);

        QualifiedMetaroomAffineTail {
            first_weights,
            first_bias,
            second_weights,
            second_bias,
        }
    }

    fn measure_projected_metaroom_119_affine1_resources(
        trunk: &QualifiedMetaroomConvReluTrunk,
        tail: &QualifiedMetaroomAffineTail,
    ) -> MetaroomAffineResources {
        // These are the already-qualified projected baseline maxima. Keeping
        // the same hard caps for the hybrid run makes any improvement a direct
        // consequence of the smaller incoming abstract state, not a relaxed
        // allocation or work budget.
        let limits = ConstrainedZonotopeAffineLimits {
            max_input_value_count: 28_672,
            max_output_value_count: 256,
            max_alpha_dim: 3_133,
            max_generator_nonzeros: 800_511,
            max_weight_elements: 7_340_032,
            max_matrix_visits: 7_340_032,
            max_interval_products: 141_372_078,
            max_constraint_count: PROJECTED_RELU_ROWS[3],
            max_constraint_elements: PROJECTED_RELU_ELEMENTS[3],
        };
        let (output, plan) = constrained_zonotope_affine_unwired(
            &trunk.relu4_output,
            tail.first_weights.view(),
            &tail.first_bias,
            limits,
        )
        .unwrap();
        assert_eq!(plan.input_value_count, 28_672);
        assert_eq!(plan.output_value_count, 256);
        assert_eq!(plan.weight_elements, 7_340_032);
        assert_eq!(plan.matrix_visits, 7_340_032);
        assert_eq!(output.value_dim(), 256);
        assert_eq!(output.alpha_dim(), plan.alpha_dim);
        assert_domain_constraint_storage(&output);
        MetaroomAffineResources {
            input_alpha_dim: plan.alpha_dim,
            input_generator_nonzeros: plan.input_generator_nonzeros,
            output_generator_nonzeros: plan.output_generator_nonzeros,
            constraint_count: plan.constraint_count,
            constraint_elements: plan.constraint_elements,
            weight_elements: plan.weight_elements,
            matrix_visits: plan.matrix_visits,
            interval_products: plan.interval_products,
            nonzero_remainders: output
                .box_remainder()
                .iter()
                .filter(|&&value| value != 0.0)
                .count(),
        }
    }

    #[test]
    #[ignore = "guarded real-model topology/parameter qualification; run explicitly under ny-safe-gpu-run"]
    fn real_metaroom_119_seals_affine_tail_topology_and_float32_bits() {
        let root = std::env::var_os("NY_METAROOM_119_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../benchmarks/vnncomp2025/benchmarks/metaroom_2023")
            });
        let onnx = root.join("onnx/6cnn_ry_39_6_no_custom_OP.onnx");
        if !onnx.exists() {
            return;
        }
        let config = OnnxLoadConfig::default().with_raw_float32_initializer_provenance(true);
        let model = load_onnx_with_config(&onnx, &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let tail = qualify_real_metaroom_119_affine_tail_topology(
            &model,
            &graph,
            "/7/Relu",
            "/7/Relu_output_0",
        );
        assert_eq!(tail.first_weights.shape(), [256, 28_672]);
        assert_eq!(tail.second_weights.shape(), [20, 256]);
    }

    #[test]
    #[ignore = "guarded full Conv/ReLU/affine-tail qualification; run explicitly under ny-safe-gpu-run"]
    fn real_metaroom_119_measures_full_affine_relu_output_tail_resources() {
        let _ = qualify_real_metaroom_119_full_affine_relu_output_tail(
            FullTrunkPredicateMode::Preserve,
        );
    }

    struct QualifiedMetaroomOutput {
        domain: ConstrainedZonotope64,
        vnnlib_spec: VnnLibSpec,
    }

    fn qualify_real_metaroom_119_full_affine_relu_output_tail(
        mode: FullTrunkPredicateMode,
    ) -> Option<QualifiedMetaroomOutput> {
        let trunk = qualify_real_metaroom_119_full_conv_relu_trunk(mode)?;
        let tail = qualify_real_metaroom_119_affine_tail_topology(
            &trunk.model,
            &trunk.graph,
            &trunk.graph_relu4_name,
            &trunk.model_relu4_output,
        );

        let first_weight_elements = 256_usize.checked_mul(28_672).unwrap();
        let first_output_generator_ceiling =
            trunk.relu4_output.alpha_dim().checked_mul(256).unwrap();
        let first_interval_product_ceiling = first_weight_elements
            .checked_mul(2)
            .and_then(|count| {
                count.checked_add(
                    trunk
                        .relu4_output
                        .generators()
                        .iter()
                        .map(ny_mip::SparseGenerator64::nnz)
                        .sum::<usize>()
                        .checked_mul(256)?,
                )
            })
            .unwrap();
        let first_output_generator_cap = 800_511;
        let first_interval_product_cap = 141_372_078;
        assert_eq!(first_weight_elements, 7_340_032);
        assert_eq!(first_output_generator_ceiling, 802_048);
        assert_eq!(first_interval_product_ceiling, 148_344_064);
        assert!(first_output_generator_cap <= first_output_generator_ceiling);
        assert!(first_interval_product_cap <= first_interval_product_ceiling);
        let first_limits = ConstrainedZonotopeAffineLimits {
            max_input_value_count: 28_672,
            max_output_value_count: 256,
            max_alpha_dim: 3_133,
            max_generator_nonzeros: first_output_generator_cap,
            max_weight_elements: first_weight_elements,
            max_matrix_visits: first_weight_elements,
            max_interval_products: first_interval_product_cap,
            max_constraint_count: mode.select(0, PROJECTED_RELU_ROWS[3]),
            max_constraint_elements: mode.select(0, PROJECTED_RELU_ELEMENTS[3]),
        };
        let (affine1_output, affine1_plan) = constrained_zonotope_affine_unwired(
            &trunk.relu4_output,
            tail.first_weights.view(),
            &tail.first_bias,
            first_limits,
        )
        .unwrap();
        let affine1_nonzero_remainders = affine1_output
            .box_remainder()
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        eprintln!(
            "real Metaroom affine1 plan: {affine1_plan:#?}; nonzero_remainders={affine1_nonzero_remainders}"
        );
        assert_eq!(affine1_plan.input_value_count, 28_672);
        assert_eq!(affine1_plan.output_value_count, 256);
        assert_eq!(affine1_plan.alpha_dim, 3_133);
        assert_eq!(
            affine1_plan.constraint_count,
            mode.select(0, PROJECTED_RELU_ROWS[3])
        );
        assert_eq!(
            affine1_plan.constraint_elements,
            mode.select(0, PROJECTED_RELU_ELEMENTS[3])
        );
        assert_eq!(affine1_plan.weight_elements, first_weight_elements);
        assert_eq!(affine1_plan.matrix_visits, first_weight_elements);
        assert_eq!(affine1_plan.input_generator_nonzeros, 522_125);
        assert_eq!(affine1_plan.output_generator_nonzeros, 800_511);
        assert_eq!(affine1_plan.interval_products, 141_372_078);
        assert_eq!(affine1_nonzero_remainders, 256);
        assert_eq!(affine1_output.value_dim(), 256);
        assert_eq!(affine1_output.alpha_dim(), 3_133);
        assert_eq!(
            affine1_output.constraint_count(),
            mode.select(0, PROJECTED_RELU_ROWS[3])
        );

        let relu5_unstable_cap = 217;
        let relu5_generator_cap = affine1_plan
            .output_generator_nonzeros
            .checked_add(relu5_unstable_cap)
            .unwrap();
        let relu5_exact_term_cap = 256_usize
            .checked_add(
                affine1_plan
                    .output_generator_nonzeros
                    .checked_mul(2)
                    .unwrap(),
            )
            .and_then(|count| count.checked_add(relu5_unstable_cap.checked_mul(4)?))
            .unwrap();
        assert_eq!(relu5_generator_cap, 800_728);
        assert_eq!(relu5_exact_term_cap, 1_602_146);
        let projected_relu5_constraints = PROJECTED_RELU_ROWS[3]
            .checked_add(relu5_unstable_cap.checked_mul(2).unwrap())
            .unwrap();
        let projected_relu5_constraint_elements =
            projected_relu5_constraints.checked_mul(3_350).unwrap();
        let projected_relu5_exact_term_cap = relu5_exact_term_cap
            .checked_add(affine1_plan.output_generator_nonzeros)
            .and_then(|count| count.checked_add(relu5_unstable_cap.checked_mul(6)?))
            .unwrap();
        assert_eq!(projected_relu5_constraints, 6_378);
        assert_eq!(projected_relu5_constraint_elements, 21_366_300);
        assert_eq!(projected_relu5_exact_term_cap, 2_403_959);
        let relu5_limits = ReluTransformLimits {
            max_value_dim: 256,
            max_output_alpha_dim: 3_350,
            max_constraints: mode.select(0, projected_relu5_constraints),
            max_constraint_elements: mode.select(0, projected_relu5_constraint_elements),
            max_generator_nnz: relu5_generator_cap,
            max_unstable: relu5_unstable_cap,
            max_exact_terms: mode.select(relu5_exact_term_cap, projected_relu5_exact_term_cap),
        };
        let relu5_output = mode.transform_relu(&affine1_output, relu5_limits).unwrap();
        let relu5_generator_nonzeros = relu5_output
            .generators()
            .iter()
            .map(ny_mip::SparseGenerator64::nnz)
            .sum::<usize>();
        let relu5_nonzero_remainders = relu5_output
            .box_remainder()
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        let relu5_unstable = relu5_output.alpha_dim() - affine1_output.alpha_dim();
        eprintln!(
            "real Metaroom ReLU5 plan: input_alpha={}, output_alpha={}, unstable={}, output_generator_nonzeros={}, nonzero_remainders={}",
            affine1_output.alpha_dim(),
            relu5_output.alpha_dim(),
            relu5_unstable,
            relu5_generator_nonzeros,
            relu5_nonzero_remainders
        );
        assert_eq!(relu5_output.value_dim(), 256);
        assert_eq!(
            relu5_output.constraint_count(),
            mode.select(0, projected_relu5_constraints)
        );
        assert_eq!(relu5_unstable, 217);
        assert_eq!(relu5_output.alpha_dim(), 3_350);
        assert_eq!(relu5_generator_nonzeros, 681_902);
        assert_eq!(relu5_nonzero_remainders, 218);

        let second_weight_elements = 20_usize.checked_mul(256).unwrap();
        let second_output_generator_ceiling = relu5_output.alpha_dim().checked_mul(20).unwrap();
        let second_generator_cap = relu5_generator_nonzeros.max(second_output_generator_ceiling);
        let second_interval_product_ceiling = second_weight_elements
            .checked_mul(2)
            .and_then(|count| count.checked_add(relu5_generator_nonzeros.checked_mul(20)?))
            .unwrap();
        let second_interval_product_cap = 13_647_520;
        assert_eq!(second_weight_elements, 5_120);
        assert_eq!(second_output_generator_ceiling, 67_000);
        assert_eq!(second_generator_cap, 681_902);
        assert_eq!(second_interval_product_ceiling, 13_648_280);
        assert!(second_interval_product_cap <= second_interval_product_ceiling);
        let second_limits = ConstrainedZonotopeAffineLimits {
            max_input_value_count: 256,
            max_output_value_count: 20,
            max_alpha_dim: 3_350,
            max_generator_nonzeros: second_generator_cap,
            max_weight_elements: second_weight_elements,
            max_matrix_visits: second_weight_elements,
            max_interval_products: second_interval_product_cap,
            max_constraint_count: mode.select(0, projected_relu5_constraints),
            max_constraint_elements: mode.select(0, projected_relu5_constraint_elements),
        };
        let (output, second_plan) = constrained_zonotope_affine_unwired(
            &relu5_output,
            tail.second_weights.view(),
            &tail.second_bias,
            second_limits,
        )
        .unwrap();
        let output_nonzero_remainders = output
            .box_remainder()
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        eprintln!(
            "real Metaroom affine2 plan: {second_plan:#?}; nonzero_remainders={output_nonzero_remainders}"
        );
        assert_eq!(second_plan.input_value_count, 256);
        assert_eq!(second_plan.output_value_count, 20);
        assert_eq!(second_plan.alpha_dim, relu5_output.alpha_dim());
        assert_eq!(
            second_plan.constraint_count,
            mode.select(0, projected_relu5_constraints)
        );
        assert_eq!(
            second_plan.constraint_elements,
            mode.select(0, projected_relu5_constraint_elements)
        );
        assert_eq!(second_plan.weight_elements, second_weight_elements);
        assert_eq!(second_plan.matrix_visits, second_weight_elements);
        assert_eq!(
            second_plan.input_generator_nonzeros,
            relu5_generator_nonzeros
        );
        assert_eq!(second_plan.output_generator_nonzeros, 66_880);
        assert_eq!(second_plan.interval_products, 13_647_520);
        assert_eq!(output_nonzero_remainders, 20);
        assert_eq!(output.value_dim(), 20);
        assert_eq!(output.alpha_dim(), relu5_output.alpha_dim());
        assert_eq!(
            output.constraint_count(),
            mode.select(0, projected_relu5_constraints)
        );
        Some(QualifiedMetaroomOutput {
            domain: output,
            vnnlib_spec: trunk.vnnlib_spec,
        })
    }

    #[derive(Debug)]
    struct Metaroom119UnsafeContract {
        target_output: usize,
        challengers: Vec<usize>,
        directions: Array2<f64>,
    }

    fn qualify_metaroom_119_unsafe_contract(spec: &VnnLibSpec) -> Metaroom119UnsafeContract {
        assert_eq!(spec.num_outputs, 20);
        assert!(spec.is_disjunction);
        assert_eq!(spec.output_constraint_clauses.len(), 19);

        let mut directions = Array2::zeros((19, 20));
        let mut challengers = Vec::with_capacity(19);
        let mut seen = [false; 20];
        seen[6] = true;
        for (row, clause) in spec.output_constraint_clauses.iter().enumerate() {
            assert_eq!(clause.len(), 1);
            let OutputConstraint::GreaterEq(challenger, target) = &clause[0] else {
                panic!("Metaroom119 clause {row} must be one non-strict challenger >= target");
            };
            assert_eq!(*target, 6);
            assert!(*challenger < spec.num_outputs);
            assert_ne!(*challenger, 6);
            assert!(!seen[*challenger]);
            seen[*challenger] = true;
            challengers.push(*challenger);
            directions[[row, *challenger]] = 1.0;
            directions[[row, *target]] = -1.0;
        }
        assert!(seen.into_iter().all(|present| present));
        assert_eq!(challengers.len(), 19);
        Metaroom119UnsafeContract {
            target_output: 6,
            challengers,
            directions,
        }
    }

    fn metaroom_119_unsafe_clause_directions(spec: &VnnLibSpec) -> Array2<f64> {
        qualify_metaroom_119_unsafe_contract(spec).directions
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "guarded projected full-tail CUDA dual qualification; run explicitly under ny-safe-gpu-run"]
    fn real_metaroom_119_measures_projected_full_output_cuda_dual() {
        let Some(qualified) = qualify_real_metaroom_119_full_affine_relu_output_tail(
            FullTrunkPredicateMode::ProjectedReluGeometry,
        ) else {
            return;
        };
        let directions = metaroom_119_unsafe_clause_directions(&qualified.vnnlib_spec);
        let limits = BatchedAdamLimits {
            max_directions: 19,
            max_iterations: 20,
            max_value_dim: 20,
            max_constraints: 6_378,
            max_alpha_dim: 3_350,
            max_constraint_elements: 21_366_300,
            max_generator_nonzeros: 66_880,
            max_direction_elements: 380,
            max_projection_products: 1_270_720,
            max_multiplier_elements: 242_364,
            max_working_f32_elements: 43_963_034,
            max_gemm_products: 32_476_776_000,
            max_wall_time: Duration::from_mins(1),
        };
        let config = BatchedAdamConfig {
            iterations: 20,
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            wall_time: Duration::from_secs(45),
            limits,
        };
        let engine = ny_cuda::CudaGemmEngine::new().expect("Metaroom119 CUDA dual engine");
        eprintln!("Metaroom119 CUDA dual device: {}", engine.device_name());
        let result =
            propose_batched_adam_unwired(&qualified.domain, directions.view(), config, &engine)
                .unwrap();
        let plan = result
            .plan
            .expect("exact Metaroom119 dual plan must qualify");
        assert_eq!(plan.constraints, 6_378);
        assert_eq!(plan.alpha_dim, 3_350);
        assert_eq!(plan.constraint_elements, 21_366_300);
        assert_eq!(plan.generator_nonzeros, 66_880);
        assert_eq!(plan.working_f32_elements, 43_963_034);
        assert_eq!(plan.gemm_products, 32_476_776_000);
        assert_eq!(result.proposals.len(), 19);

        let mut excluded = 0_usize;
        for (clause, proposal) in result.proposals.iter().enumerate() {
            assert!(proposal.bounds.lower.is_finite());
            assert!(proposal.bounds.upper.is_finite());
            if proposal.bounds.upper < 0.0 {
                excluded += 1;
            }
            eprintln!(
                "Metaroom119 clause {clause}: lower={:.17e}, upper={:.17e}, lower_improved={}, upper_improved={}",
                proposal.bounds.lower,
                proposal.bounds.upper,
                proposal.lower_improved,
                proposal.upper_improved
            );
        }
        eprintln!(
            "Metaroom119 projected CUDA dual: status={:?}, iterations={}, engine_calls={}, excluded_clauses={excluded}/19, all_unsafe_clauses_excluded={}",
            result.status,
            result.iterations_completed,
            result.engine_calls,
            excluded == 19
        );
    }

    #[test]
    #[ignore = "guarded CPU-only Box/CZ full-topology diagnostic; run explicitly under ny-safe-gpu-run"]
    fn real_metaroom_119_quantifies_certified_box_advantage_over_projected_cz_radii() {
        let mut diagnostic = MetaroomBoxDiagnostic::default();
        let Some(trunk) = qualify_real_metaroom_119_full_conv_relu_trunk_with_box(
            FullTrunkPredicateMode::ProjectedReluGeometry,
            &mut diagnostic,
        ) else {
            return;
        };
        assert_eq!(diagnostic.relus.len(), 4);

        let tail = qualify_real_metaroom_119_affine_tail_topology(
            &trunk.model,
            &trunk.graph,
            &trunk.graph_relu4_name,
            &trunk.model_relu4_output,
        );
        let first_limits = ConstrainedZonotopeAffineLimits {
            max_input_value_count: 28_672,
            max_output_value_count: 256,
            max_alpha_dim: 3_133,
            max_generator_nonzeros: 800_511,
            max_weight_elements: 7_340_032,
            max_matrix_visits: 7_340_032,
            max_interval_products: 141_372_078,
            max_constraint_count: PROJECTED_RELU_ROWS[3],
            max_constraint_elements: PROJECTED_RELU_ELEMENTS[3],
        };
        let (affine1_output, affine1_plan) = constrained_zonotope_affine_unwired(
            &trunk.relu4_output,
            tail.first_weights.view(),
            &tail.first_bias,
            first_limits,
        )
        .unwrap();
        assert_eq!(affine1_plan.input_value_count, 28_672);
        assert_eq!(affine1_plan.output_value_count, 256);
        assert_eq!(affine1_plan.constraint_count, PROJECTED_RELU_ROWS[3]);
        assert_eq!(affine1_plan.constraint_elements, PROJECTED_RELU_ELEMENTS[3]);
        diagnostic.affine_relu(
            "ReLU5",
            tail.first_weights.view(),
            &tail.first_bias,
            &affine1_output,
        );
        diagnostic.terminal_affine(tail.second_weights.view(), &tail.second_bias);

        assert_eq!(diagnostic.relus.len(), 5);
        assert_eq!(diagnostic.terminal.as_ref().unwrap().len(), 20);
        let box_only_stable: usize = diagnostic
            .relus
            .iter()
            .map(|report| report.box_only_active + report.box_only_inactive)
            .sum();
        eprintln!(
            "Metaroom certified Box full-topology totals: box_only_stable={box_only_stable}, conv_scalar_products={}, affine_scalar_products={}, terminal_lower={:?}, terminal_upper={:?}",
            diagnostic.conv_scalar_products,
            diagnostic.affine_scalar_products,
            diagnostic.terminal.as_ref().unwrap().lower(),
            diagnostic.terminal.as_ref().unwrap().upper(),
        );

        // Diagnostic only: these proof-safe bounds and counts are deliberately
        // not consumed by a command, score gate, or verifier verdict.
        assert_eq!(
            diagnostic
                .relus
                .iter()
                .map(|report| report.value_count)
                .collect::<Vec<_>>(),
            [57_344, 57_344, 28_672, 28_672, 256]
        );
        assert_eq!(
            diagnostic
                .relus
                .iter()
                .map(|report| report.cz_unstable)
                .collect::<Vec<_>>(),
            [476, 687, 681, 1_128, 217]
        );
    }

    #[test]
    #[ignore = "guarded inductive Box/CZ real-model traversal; run explicitly under ny-safe-gpu-run"]
    fn real_metaroom_119_measures_inductive_box_cz_hybrid_resources() {
        let mut diagnostic = MetaroomBoxDiagnostic::default();
        let Some(baseline) = qualify_real_metaroom_119_full_conv_relu_trunk_with_box_counterfactual(
            FullTrunkPredicateMode::ProjectedReluGeometry,
            &mut diagnostic,
        ) else {
            return;
        };
        assert_eq!(diagnostic.relus.len(), 4);
        assert_eq!(diagnostic.auxiliary_preactivations.len(), 4);
        assert_eq!(baseline.relu_resources.len(), 4);

        let baseline_resources = baseline.relu_resources.clone();
        assert_eq!(
            baseline_resources
                .iter()
                .map(|row| (
                    row.stage,
                    row.input_alpha_dim,
                    row.output_alpha_dim,
                    row.unstable,
                    row.generator_nonzeros,
                    row.constraint_count,
                    row.constraint_elements,
                    row.nonzero_remainders,
                ))
                .collect::<Vec<_>>(),
            [
                ("ReLU1", 161, 637, 476, 15_348, 952, 606_424, 19_960),
                ("ReLU2", 637, 1_324, 687, 71_042, 2_326, 3_079_624, 12_172,),
                ("ReLU3", 1_324, 2_005, 681, 167_447, 3_688, 7_394_440, 7_761,),
                ("ReLU4", 2_005, 3_133, 1_128, 522_125, 5_944, 18_622_552, 1_440,),
            ]
        );

        let independently_intersected_unstable = diagnostic
            .relus
            .iter()
            .map(|report| {
                report
                    .cz_unstable
                    .checked_sub(report.box_only_active + report.box_only_inactive)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(independently_intersected_unstable, [476, 687, 605, 392]);

        let relu4_counterfactual = baseline
            .relu4_auxiliary_counterfactual
            .as_ref()
            .unwrap()
            .clone();
        assert_eq!(relu4_counterfactual.stage, "ReLU4");
        assert_eq!(relu4_counterfactual.input_alpha_dim, 2_005);
        assert_eq!(relu4_counterfactual.output_alpha_dim, 2_397);
        assert_eq!(relu4_counterfactual.unstable, 392);
        assert_eq!(relu4_counterfactual.constraint_count, 4_472);
        assert_eq!(relu4_counterfactual.constraint_elements, 10_719_384);
        assert!(relu4_counterfactual.generator_nonzeros < baseline_resources[3].generator_nonzeros);

        let tail = qualify_real_metaroom_119_affine_tail_topology(
            &baseline.model,
            &baseline.graph,
            &baseline.graph_relu4_name,
            &baseline.model_relu4_output,
        );
        let baseline_affine = measure_projected_metaroom_119_affine1_resources(&baseline, &tail);
        assert_eq!(
            baseline_affine,
            MetaroomAffineResources {
                input_alpha_dim: 3_133,
                input_generator_nonzeros: 522_125,
                output_generator_nonzeros: 800_511,
                constraint_count: 5_944,
                constraint_elements: 18_622_552,
                weight_elements: 7_340_032,
                matrix_visits: 7_340_032,
                interval_products: 141_372_078,
                nonzero_remainders: 256,
            }
        );

        let auxiliary_trace = std::mem::take(&mut diagnostic.auxiliary_preactivations);
        drop(baseline);
        drop(diagnostic);

        let hybrid = qualify_real_metaroom_119_full_conv_relu_trunk_with_auxiliary_trace(
            FullTrunkPredicateMode::ProjectedReluGeometry,
            &auxiliary_trace,
        )
        .unwrap();
        assert_eq!(hybrid.relu_resources.len(), 4);
        let hybrid_affine = measure_projected_metaroom_119_affine1_resources(&hybrid, &tail);
        assert_eq!(
            hybrid
                .relu_resources
                .iter()
                .map(|row| (
                    row.stage,
                    row.input_alpha_dim,
                    row.output_alpha_dim,
                    row.unstable,
                    row.generator_nonzeros,
                    row.constraint_count,
                    row.constraint_elements,
                    row.nonzero_remainders,
                ))
                .collect::<Vec<_>>(),
            [
                ("ReLU1", 161, 637, 476, 15_348, 952, 606_424, 19_960),
                ("ReLU2", 637, 1_324, 687, 71_042, 2_326, 3_079_624, 12_172,),
                ("ReLU3", 1_324, 1_929, 605, 155_320, 3_536, 6_820_944, 7_685,),
                ("ReLU4", 1_929, 2_321, 392, 172_367, 4_320, 10_026_720, 704,),
            ]
        );
        assert_eq!(
            hybrid_affine,
            MetaroomAffineResources {
                input_alpha_dim: 2_321,
                input_generator_nonzeros: 172_367,
                output_generator_nonzeros: 592_384,
                constraint_count: 4_320,
                constraint_elements: 10_026_720,
                weight_elements: 7_340_032,
                matrix_visits: 7_340_032,
                interval_products: 51_646_197,
                nonzero_remainders: 256,
            }
        );

        eprintln!(
            "Metaroom119 inductive Box/CZ resources: stage | baseline input/output alpha | hybrid input/output alpha | baseline/hybrid unstable | baseline/hybrid generator nnz | baseline/hybrid constraints | baseline/hybrid constraint elements"
        );
        for (baseline_row, hybrid_row) in
            baseline_resources.iter().zip(hybrid.relu_resources.iter())
        {
            assert_eq!(hybrid_row.stage, baseline_row.stage);
            assert!(hybrid_row.input_alpha_dim <= baseline_row.input_alpha_dim);
            assert!(hybrid_row.output_alpha_dim <= baseline_row.output_alpha_dim);
            assert!(hybrid_row.unstable <= baseline_row.unstable);
            assert!(hybrid_row.generator_nonzeros <= baseline_row.generator_nonzeros);
            assert!(hybrid_row.constraint_count <= baseline_row.constraint_count);
            assert!(hybrid_row.constraint_elements <= baseline_row.constraint_elements);
            eprintln!(
                "{} | {}/{} | {}/{} | {}/{} | {}/{} | {}/{} | {}/{}",
                baseline_row.stage,
                baseline_row.input_alpha_dim,
                baseline_row.output_alpha_dim,
                hybrid_row.input_alpha_dim,
                hybrid_row.output_alpha_dim,
                baseline_row.unstable,
                hybrid_row.unstable,
                baseline_row.generator_nonzeros,
                hybrid_row.generator_nonzeros,
                baseline_row.constraint_count,
                hybrid_row.constraint_count,
                baseline_row.constraint_elements,
                hybrid_row.constraint_elements,
            );
        }
        assert!(hybrid.relu_resources[3].output_alpha_dim < baseline_resources[3].output_alpha_dim);
        assert!(hybrid.relu_resources[3].unstable < baseline_resources[3].unstable);
        assert!(
            hybrid.relu_resources[3].generator_nonzeros < baseline_resources[3].generator_nonzeros
        );
        assert!(hybrid.relu_resources[3].constraint_count < baseline_resources[3].constraint_count);
        assert!(
            hybrid.relu_resources[3].constraint_elements
                < baseline_resources[3].constraint_elements
        );

        assert!(hybrid_affine.input_alpha_dim < baseline_affine.input_alpha_dim);
        assert!(hybrid_affine.input_generator_nonzeros < baseline_affine.input_generator_nonzeros);
        assert!(
            hybrid_affine.output_generator_nonzeros < baseline_affine.output_generator_nonzeros
        );
        assert!(hybrid_affine.constraint_count < baseline_affine.constraint_count);
        assert!(hybrid_affine.constraint_elements < baseline_affine.constraint_elements);
        assert_eq!(
            hybrid_affine.weight_elements,
            baseline_affine.weight_elements
        );
        assert_eq!(hybrid_affine.matrix_visits, baseline_affine.matrix_visits);
        assert!(hybrid_affine.interval_products < baseline_affine.interval_products);
        eprintln!(
            "Metaroom119 affine1 baseline/hybrid: alpha={}/{}, input_generator_nnz={}/{}, output_generator_nnz={}/{}, constraints={}/{}, constraint_elements={}/{}, matrix_visits={}/{}, interval_products={}/{}, nonzero_remainders={}/{}",
            baseline_affine.input_alpha_dim,
            hybrid_affine.input_alpha_dim,
            baseline_affine.input_generator_nonzeros,
            hybrid_affine.input_generator_nonzeros,
            baseline_affine.output_generator_nonzeros,
            hybrid_affine.output_generator_nonzeros,
            baseline_affine.constraint_count,
            hybrid_affine.constraint_count,
            baseline_affine.constraint_elements,
            hybrid_affine.constraint_elements,
            baseline_affine.matrix_visits,
            hybrid_affine.matrix_visits,
            baseline_affine.interval_products,
            hybrid_affine.interval_products,
            baseline_affine.nonzero_remainders,
            hybrid_affine.nonzero_remainders,
        );
    }

    const METAROOM_HYBRID_TAIL_ENV: &str = "NY_CZ_HYBRID_TAIL_DIAGNOSTIC";
    const METAROOM_M24_SCHEDULE_ENV: &str = "NY_CZ_HYBRID_TAIL_M24_SCHEDULE";
    const METAROOM_TAIL_TARGET: usize = 6;
    const METAROOM_TAIL_SMOKE_CHALLENGER: usize = 14;
    const METAROOM_TAIL_MAX_POSITIVE_COEFFICIENTS: usize = 135;
    const METAROOM_TAIL_I8_SEARCH_WORK: u64 = 14_857_701;
    const METAROOM_TAIL_I20_SEARCH_WORK: u64 = 36_250_233;
    const METAROOM_M24_MAX_BOX_VARIABLES: usize = 512;
    const METAROOM_M24_TOTAL_ITERATIONS: usize = 8;
    const METAROOM_M24_MAX_SEARCH_WORK: u64 = 11_920_810;
    const METAROOM_M24_MEMBER_WALL_CAP: Duration = Duration::from_secs(3);
    const METAROOM_M24_CANDIDATE_WALL_CAP: Duration = Duration::from_secs(1);
    const METAROOM_TAIL_POSITIVE_COEFFICIENTS: [usize; 20] = [
        116, 112, 109, 100, 110, 107, 0, 124, 111, 115, 120, 122, 95, 123, 135, 102, 107, 116, 109,
        109,
    ];

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum MetaroomHybridTailStage {
        Smoke0,
        One8,
        All8,
        Cascade,
    }

    impl MetaroomHybridTailStage {
        fn parse(value: &str) -> Option<Self> {
            match value {
                "smoke0" => Some(Self::Smoke0),
                "one8" => Some(Self::One8),
                "all8" => Some(Self::All8),
                "cascade" => Some(Self::Cascade),
                _ => None,
            }
        }

        fn primary_iterations(self) -> usize {
            match self {
                Self::Smoke0 => 0,
                Self::One8 | Self::All8 | Self::Cascade => 8,
            }
        }

        fn tail_outer_budget(self) -> Duration {
            match self {
                Self::Smoke0 | Self::One8 => Duration::from_secs(30),
                Self::All8 => Duration::from_mins(3),
                Self::Cascade => Duration::from_mins(8),
            }
        }

        fn is_single_challenger(self) -> bool {
            matches!(self, Self::Smoke0 | Self::One8)
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum MetaroomM24Schedule {
        Bounded4x4,
    }

    impl MetaroomM24Schedule {
        fn parse(value: &str) -> Option<Self> {
            match value {
                "bounded4x4" => Some(Self::Bounded4x4),
                _ => None,
            }
        }

        fn tail_outer_budget(self) -> Duration {
            match self {
                Self::Bounded4x4 => Duration::from_mins(8),
            }
        }
    }

    fn requested_metaroom_hybrid_tail_stage() -> Option<MetaroomHybridTailStage> {
        match std::env::var(METAROOM_HYBRID_TAIL_ENV) {
            Ok(value) => Some(MetaroomHybridTailStage::parse(&value).unwrap_or_else(|| {
                panic!(
                    "{METAROOM_HYBRID_TAIL_ENV} must be one of smoke0, one8, all8, or cascade; got {value:?}"
                )
            })),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("{METAROOM_HYBRID_TAIL_ENV} must be valid UTF-8")
            }
        }
    }

    fn requested_metaroom_m24_schedule() -> Option<MetaroomM24Schedule> {
        match std::env::var(METAROOM_M24_SCHEDULE_ENV) {
            Ok(value) => Some(MetaroomM24Schedule::parse(&value).unwrap_or_else(|| {
                panic!("{METAROOM_M24_SCHEDULE_ENV} must be bounded4x4; got {value:?}")
            })),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("{METAROOM_M24_SCHEDULE_ENV} must be valid UTF-8")
            }
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct MetaroomTailCertificateState {
        challenger_output: usize,
        best_dual_lower_bound: Option<f64>,
        ay_strictly_positive: bool,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum MetaroomTailPortfolioWinner {
        BaselineM17,
        AuxiliaryM20,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct MetaroomTailPortfolioDecision {
        baseline_lower_bound: f64,
        auxiliary_lower_bound: Option<f64>,
        retained_lower_bound: f64,
        winner: MetaroomTailPortfolioWinner,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum MetaroomM24PortfolioWinner {
        BaselineM17,
        AuxiliaryM20,
        BoxCutM24,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct MetaroomM24PortfolioDecision {
        m17_lower_bound: f64,
        m20_lower_bound: Option<f64>,
        m24_lower_bound: Option<f64>,
        retained_lower_bound: f64,
        winner: MetaroomM24PortfolioWinner,
    }

    // Keep prepared M17, M20, and optional M24 structurally tied to the same
    // affine1_output borrow. The private domain borrow inside
    // `PreparedReluTailGeometry64` prevents any portfolio member from being
    // paired with a lookalike CZ.
    struct MetaroomPreparedTailDomain<'domain> {
        geometry: PreparedReluTailGeometry64<'domain>,
    }

    impl<'domain> MetaroomPreparedTailDomain<'domain> {
        fn try_new(
            domain: &'domain ConstrainedZonotope64,
        ) -> Result<Self, ny_mip::ReluTailDualError> {
            let geometry = prepare_relu_tail_triangle_dual_unwired(domain)?;
            Ok(Self { geometry })
        }
    }

    // M17 is the mandatory portfolio member. An M17 error is therefore
    // propagated, while an M20 error is returned alongside an M17-only
    // decision so the diagnostic can attribute the failure and continue with
    // the independently replayed baseline certificate. Exact ties stay with
    // M17 to make it impossible for M20 to hide the baseline member.
    fn retain_metaroom_tail_portfolio<E>(
        baseline: Result<f64, E>,
        auxiliary: Result<f64, E>,
    ) -> Result<(MetaroomTailPortfolioDecision, Option<E>), E> {
        let baseline_lower_bound = baseline?;
        assert!(baseline_lower_bound.is_finite());
        match auxiliary {
            Ok(auxiliary_lower_bound) => {
                assert!(auxiliary_lower_bound.is_finite());
                let (retained_lower_bound, winner) = if auxiliary_lower_bound > baseline_lower_bound
                {
                    (
                        auxiliary_lower_bound,
                        MetaroomTailPortfolioWinner::AuxiliaryM20,
                    )
                } else {
                    (
                        baseline_lower_bound,
                        MetaroomTailPortfolioWinner::BaselineM17,
                    )
                };
                Ok((
                    MetaroomTailPortfolioDecision {
                        baseline_lower_bound,
                        auxiliary_lower_bound: Some(auxiliary_lower_bound),
                        retained_lower_bound,
                        winner,
                    },
                    None,
                ))
            }
            Err(error) => Ok((
                MetaroomTailPortfolioDecision {
                    baseline_lower_bound,
                    auxiliary_lower_bound: None,
                    retained_lower_bound: baseline_lower_bound,
                    winner: MetaroomTailPortfolioWinner::BaselineM17,
                },
                Some(error),
            )),
        }
    }

    // M26 mirrors M24's ordered certificate portfolio without trusting the
    // optimizer's approximate objective. M17 remains mandatory; missing M20
    // or M24 members retain the best earlier exact replay, and strict `>`
    // keeps ties ordered M17, then M20, then M24.
    fn retain_metaroom_m24_portfolio<E>(
        m17: Result<f64, E>,
        m20: Result<f64, E>,
        m24: Result<f64, E>,
    ) -> Result<(MetaroomM24PortfolioDecision, [Option<E>; 2]), E> {
        let m17_lower_bound = m17?;
        assert!(m17_lower_bound.is_finite());
        let mut retained_lower_bound = m17_lower_bound;
        let mut winner = MetaroomM24PortfolioWinner::BaselineM17;

        let (m20_lower_bound, m20_error) = match m20 {
            Ok(lower_bound) => {
                assert!(lower_bound.is_finite());
                if lower_bound > retained_lower_bound {
                    retained_lower_bound = lower_bound;
                    winner = MetaroomM24PortfolioWinner::AuxiliaryM20;
                }
                (Some(lower_bound), None)
            }
            Err(error) => (None, Some(error)),
        };
        let (m24_lower_bound, m24_error) = match m24 {
            Ok(lower_bound) => {
                assert!(lower_bound.is_finite());
                if lower_bound > retained_lower_bound {
                    retained_lower_bound = lower_bound;
                    winner = MetaroomM24PortfolioWinner::BoxCutM24;
                }
                (Some(lower_bound), None)
            }
            Err(error) => (None, Some(error)),
        };

        Ok((
            MetaroomM24PortfolioDecision {
                m17_lower_bound,
                m20_lower_bound,
                m24_lower_bound,
                retained_lower_bound,
                winner,
            },
            [m20_error, m24_error],
        ))
    }

    impl MetaroomTailCertificateState {
        fn new(challenger_output: usize) -> Self {
            Self {
                challenger_output,
                best_dual_lower_bound: None,
                ay_strictly_positive: false,
            }
        }

        fn observe_dual(&mut self, lower_bound: f64) {
            assert!(lower_bound.is_finite());
            self.best_dual_lower_bound = Some(
                self.best_dual_lower_bound
                    .map_or(lower_bound, |prior| prior.max(lower_bound)),
            );
        }

        fn strictly_positive(&self) -> bool {
            self.ay_strictly_positive
                || self
                    .best_dual_lower_bound
                    .is_some_and(|lower_bound| lower_bound > 0.0)
        }
    }

    fn unresolved_metaroom_tail_challengers(
        certificates: &[MetaroomTailCertificateState],
    ) -> Vec<usize> {
        certificates
            .iter()
            .filter(|certificate| !certificate.strictly_positive())
            .map(|certificate| certificate.challenger_output)
            .collect()
    }

    fn metaroom_tail_dual_config(iterations: usize, wall_time: Duration) -> ReluTailDualConfig {
        let (max_iterations, max_search_work) = match iterations {
            // Search is disabled before plan construction, but retain a
            // nonzero, well-formed firewall for config-shape diagnostics.
            0 => (1, 596_013),
            8 => (8, METAROOM_TAIL_I8_SEARCH_WORK),
            20 => (20, METAROOM_TAIL_I20_SEARCH_WORK),
            _ => panic!("unqualified Metaroom tail iteration count {iterations}"),
        };
        assert!(!wall_time.is_zero());
        ReluTailDualConfig {
            iterations,
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            wall_time,
            limits: ReluTailDualLimits {
                max_value_dim: 256,
                max_alpha_dim: 2_321,
                max_constraints: 0,
                max_generator_nonzeros: 592_384,
                max_optimizable_slopes: METAROOM_TAIL_MAX_POSITIVE_COEFFICIENTS,
                max_iterations,
                max_search_work,
                max_wall_time: wall_time,
            },
        }
    }

    fn metaroom_m24_optimizer_config(
        schedule: MetaroomM24Schedule,
        wall_time: Duration,
    ) -> ReluTailBoxCutOptimizerConfig {
        assert!(!wall_time.is_zero());
        assert!(wall_time <= METAROOM_M24_CANDIDATE_WALL_CAP);
        let schedules = match schedule {
            MetaroomM24Schedule::Bounded4x4 => [
                ReluTailBoxCutAdamSchedule {
                    iterations: 4,
                    learning_rate: 0.005,
                    decay: 0.98,
                },
                ReluTailBoxCutAdamSchedule {
                    iterations: 4,
                    learning_rate: 0.1,
                    decay: 0.98,
                },
            ],
        };
        ReluTailBoxCutOptimizerConfig {
            schedules,
            multiplier_cap: 16.0,
            wall_time,
            limits: ReluTailBoxCutOptimizerLimits {
                max_value_dim: 256,
                max_box_variables: METAROOM_M24_MAX_BOX_VARIABLES,
                max_total_iterations: METAROOM_M24_TOTAL_ITERATIONS,
                max_restarts: 2,
                max_exact_replays: 2,
                max_generator_nonzeros: 592_384,
                max_search_work: METAROOM_M24_MAX_SEARCH_WORK,
                max_wall_time: METAROOM_M24_CANDIDATE_WALL_CAP,
            },
        }
    }

    fn metaroom_tail_lp_config(
        target_output: usize,
        challenger_count: usize,
        wall_time: Duration,
    ) -> ConstrainedZonotopeTailLpConfig {
        assert!((1..=19).contains(&challenger_count));
        assert!(!wall_time.is_zero());
        ConstrainedZonotopeTailLpConfig {
            target_output,
            wall_time,
            ay_memory_budget_bytes: 1_536_usize * 1_024 * 1_024,
            exact_milp_binary_cap: 256,
            limits: ConstrainedZonotopeTailLpLimits {
                max_value_dim: 256,
                max_alpha_dim: 2_321,
                max_constraints: 0,
                max_constraint_elements: 0,
                max_generator_nonzeros: 592_384,
                max_constraint_nonzeros: 0,
                max_output_dim: 20,
                max_unstable_relus: 256,
                max_model_columns: 3_109 + challenger_count,
                max_model_rows: 788 + challenger_count,
                max_model_nonzeros: 599_060 + 3 * challenger_count,
                max_solves: challenger_count,
            },
        }
    }

    fn remaining_metaroom_tail_budget(deadline: Instant) -> Option<Duration> {
        deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
    }

    // Candidate-family attribution is intentionally isolated in this one
    // reporter. `zero_multiplier_lower_bound` is the final accepted direction
    // replayed without predicate multipliers; it is not the zero-positive-
    // slope candidate recorded separately below. None of these measurements
    // changes cascade authority or challenger selection.
    fn emit_metaroom_tail_dual_observation(
        phase: &str,
        member: &str,
        challenger_output: usize,
        requested_iterations: usize,
        result: &ReluTailDualResult,
        elapsed: Duration,
    ) {
        let candidate_replays = result.zero_predicate_candidate_replays;
        eprintln!(
            "Metaroom119 hybrid tail {phase} {member}: target={METAROOM_TAIL_TARGET}, challenger={challenger_output}, requested_iterations={requested_iterations}, lower_bound={:.17e}, zero_multiplier_lower_bound={:.17e}, zero_positive_slope_lower_bound={:.17e}, upper_endpoint_lower_bound={:?}, canonical_lower_bound={:?}, optimized_lower_bound={:?}, status={:?}, iterations_completed={}, candidates_replayed={}, optimizable_slopes={}, supplied_multipliers_used={}, elapsed={elapsed:?}, plan={:?}",
            result.lower_bound,
            result.zero_multiplier_lower_bound,
            candidate_replays.zero_positive_slope_lower_bound,
            candidate_replays.upper_endpoint_lower_bound,
            candidate_replays.canonical_lower_bound,
            candidate_replays.optimized_lower_bound,
            result.status,
            result.iterations_completed,
            result.candidates_replayed,
            result.optimizable_slopes,
            result.supplied_multipliers_used,
            result.plan,
        );
    }

    fn run_metaroom_tail_dual_pass(
        phase: &str,
        prepared: &MetaroomPreparedTailDomain<'_>,
        auxiliary: &CertifiedAuxiliaryBounds64,
        tail: &QualifiedMetaroomAffineTail,
        target_output: usize,
        challengers: &[usize],
        iterations: usize,
        outer_deadline: Instant,
        certificates: &mut [MetaroomTailCertificateState],
    ) -> bool {
        assert!(!challengers.is_empty());
        assert!(challengers.len() <= 19);
        assert_eq!(auxiliary.value_dim(), prepared.geometry.value_dim());
        let per_margin_wall_cap = match iterations {
            0 => Duration::from_secs(1),
            8 => Duration::from_secs(5),
            20 => Duration::from_secs(10),
            _ => panic!("unqualified Metaroom tail iteration count {iterations}"),
        };
        let target_row = tail.second_weights.row(target_output);
        let target_row = target_row
            .as_slice()
            .expect("sealed Metaroom target row must be contiguous");
        let exact_zero = BigRational::from_integer(0.into());

        for &challenger_output in challengers {
            let Some(remaining) = remaining_metaroom_tail_budget(outer_deadline) else {
                eprintln!(
                    "Metaroom119 hybrid tail {phase}: outer deadline reached before challenger {challenger_output}"
                );
                return false;
            };
            assert_ne!(challenger_output, target_output);
            assert!(challenger_output < tail.second_weights.nrows());
            let challenger_row = tail.second_weights.row(challenger_output);
            let challenger_row = challenger_row
                .as_slice()
                .expect("sealed Metaroom challenger row must be contiguous");
            let margin = exact_relu_tail_margin_from_f64_rows(
                target_row,
                challenger_row,
                tail.second_bias[target_output],
                tail.second_bias[challenger_output],
            )
            .unwrap();
            let positive_coefficients = margin
                .coefficients()
                .iter()
                .filter(|coefficient| *coefficient > &exact_zero)
                .count();
            assert_eq!(
                positive_coefficients,
                METAROOM_TAIL_POSITIVE_COEFFICIENTS[challenger_output]
            );
            assert!(positive_coefficients <= METAROOM_TAIL_MAX_POSITIVE_COEFFICIENTS);

            let member_wall_time = remaining.min(per_margin_wall_cap);
            let baseline_started = Instant::now();
            let baseline = prepared
                .geometry
                .bound_margin_unwired(
                    &margin,
                    None,
                    metaroom_tail_dual_config(iterations, member_wall_time),
                )
                .unwrap();
            emit_metaroom_tail_dual_observation(
                phase,
                "baseline-M17",
                challenger_output,
                iterations,
                &baseline,
                baseline_started.elapsed(),
            );

            // M20 is independently planned, searched, and replayed from the
            // same original CZ. Its auxiliary Box changes only the valid ReLU
            // minorants; it cannot replace M17, and failures retain M17.
            let auxiliary_started = Instant::now();
            let auxiliary_lower_bound =
                if let Some(auxiliary_remaining) = remaining_metaroom_tail_budget(outer_deadline) {
                    let auxiliary_result = prepared
                        .geometry
                        .bound_margin_with_auxiliary_bounds_unwired(
                            auxiliary,
                            &margin,
                            None,
                            metaroom_tail_dual_config(
                                iterations,
                                auxiliary_remaining.min(per_margin_wall_cap),
                            ),
                        );
                    match auxiliary_result {
                        Ok(auxiliary_result) => {
                            emit_metaroom_tail_dual_observation(
                                phase,
                                "auxiliary-M20",
                                challenger_output,
                                iterations,
                                &auxiliary_result,
                                auxiliary_started.elapsed(),
                            );
                            Ok(auxiliary_result.lower_bound)
                        }
                        Err(error) => Err(format!("API error: {error:?}")),
                    }
                } else {
                    Err("outer deadline exhausted before auxiliary-M20".to_string())
                };
            let (decision, auxiliary_error) = retain_metaroom_tail_portfolio(
                Ok::<f64, String>(baseline.lower_bound),
                auxiliary_lower_bound,
            )
            .expect("mandatory M17 result was already independently replayed");
            if let Some(reason) = auxiliary_error {
                eprintln!(
                    "Metaroom119 hybrid tail {phase} auxiliary-M20: target={METAROOM_TAIL_TARGET}, challenger={challenger_output}, requested_iterations={iterations}, skipped_or_failed={reason}, elapsed={:?}; retaining baseline-M17 lower_bound={:.17e}",
                    auxiliary_started.elapsed(),
                    decision.baseline_lower_bound,
                );
            }
            eprintln!(
                "Metaroom119 hybrid tail {phase} portfolio: target={METAROOM_TAIL_TARGET}, challenger={challenger_output}, baseline_lower_bound={:.17e}, auxiliary_lower_bound={:?}, retained_lower_bound={:.17e}, winner={:?}",
                decision.baseline_lower_bound,
                decision.auxiliary_lower_bound,
                decision.retained_lower_bound,
                decision.winner,
            );
            let certificate = certificates
                .iter_mut()
                .find(|certificate| certificate.challenger_output == challenger_output)
                .expect("dual challenger must belong to the sealed property");
            // Observe M17 explicitly before the portfolio maximum so its
            // certificate remains represented even when M20 wins.
            certificate.observe_dual(baseline.lower_bound);
            certificate.observe_dual(decision.retained_lower_bound);

            // A deadline status still carries the independently replayed prior
            // certificate above. The outer deadline only controls whether a
            // subsequent challenger is started; it never invalidates that
            // accepted bound.
            if remaining_metaroom_tail_budget(outer_deadline).is_none() {
                eprintln!(
                    "Metaroom119 hybrid tail {phase}: outer deadline reached after challenger {challenger_output}"
                );
                return false;
            }
        }
        true
    }

    fn metaroom_m24_winner(selection: ReluTailBoxCutSelection) -> MetaroomM24PortfolioWinner {
        match selection {
            ReluTailBoxCutSelection::Original => MetaroomM24PortfolioWinner::BaselineM17,
            ReluTailBoxCutSelection::Auxiliary => MetaroomM24PortfolioWinner::AuxiliaryM20,
            ReluTailBoxCutSelection::BoxCut => MetaroomM24PortfolioWinner::BoxCutM24,
        }
    }

    fn emit_metaroom_m24_observation(
        phase: &str,
        challenger_output: usize,
        requested_iterations: usize,
        result: &ReluTailBoxCutOptimizedResult,
        decision: MetaroomM24PortfolioDecision,
        member_errors: &[Option<&'static str>; 2],
        elapsed: Duration,
    ) {
        let m17 = &result.portfolio.original;
        let m20 = result.portfolio.auxiliary.as_ref();
        eprintln!(
            "Metaroom119 hybrid tail {phase} M26 portfolio: target={METAROOM_TAIL_TARGET}, challenger={challenger_output}, requested_iterations={requested_iterations}, m17_bound={:.17e}, m17_status={:?}, m17_iterations={}, m17_candidates={}, m20_bound={:?}, m20_status={:?}, m20_iterations={:?}, m20_candidates={:?}, m20_failure={:?}, m24_bound={:?}, m24_replay_status={:?}, m24_search_status={:?}, m24_iterations={}, m24_restarts={}, m24_candidates={}, m24_exact_replays={}, m24_failure={:?}, retained_lower_bound={:.17e}, winner={:?}, elapsed={elapsed:?}, m24_plan={:?}",
            decision.m17_lower_bound,
            m17.status,
            m17.iterations_completed,
            m17.candidates_replayed,
            decision.m20_lower_bound,
            m20.map(|member| &member.status),
            m20.map(|member| member.iterations_completed),
            m20.map(|member| member.candidates_replayed),
            member_errors[0],
            decision.m24_lower_bound,
            result.portfolio.status,
            result.search_status,
            result.iterations_completed,
            result.restarts_completed,
            result.candidates_scored,
            result.exact_replays,
            member_errors[1],
            decision.retained_lower_bound,
            decision.winner,
            result.search_plan,
        );
    }

    fn run_metaroom_m24_pass(
        phase: &str,
        prepared: &MetaroomPreparedTailDomain<'_>,
        auxiliary: &CertifiedAuxiliaryBounds64,
        tail: &QualifiedMetaroomAffineTail,
        target_output: usize,
        challengers: &[usize],
        iterations: usize,
        schedule: MetaroomM24Schedule,
        outer_deadline: Instant,
        certificates: &mut [MetaroomTailCertificateState],
    ) -> bool {
        assert_eq!(target_output, METAROOM_TAIL_TARGET);
        assert_eq!(challengers.len(), 19);
        assert_eq!(iterations, 8);
        assert_eq!(auxiliary.value_dim(), prepared.geometry.value_dim());
        let target_row = tail.second_weights.row(target_output);
        let target_row = target_row
            .as_slice()
            .expect("sealed Metaroom target row must be contiguous");
        let exact_zero = BigRational::from_integer(0.into());

        for &challenger_output in challengers {
            let Some(remaining) = remaining_metaroom_tail_budget(outer_deadline) else {
                eprintln!(
                    "Metaroom119 hybrid tail {phase} M26: outer deadline reached before challenger {challenger_output}"
                );
                return false;
            };
            assert_ne!(challenger_output, target_output);
            assert!(challenger_output < tail.second_weights.nrows());
            let challenger_row = tail.second_weights.row(challenger_output);
            let challenger_row = challenger_row
                .as_slice()
                .expect("sealed Metaroom challenger row must be contiguous");
            let margin = exact_relu_tail_margin_from_f64_rows(
                target_row,
                challenger_row,
                tail.second_bias[target_output],
                tail.second_bias[challenger_output],
            )
            .unwrap();
            let positive_coefficients = margin
                .coefficients()
                .iter()
                .filter(|coefficient| *coefficient > &exact_zero)
                .count();
            assert_eq!(
                positive_coefficients,
                METAROOM_TAIL_POSITIVE_COEFFICIENTS[challenger_output]
            );
            assert!(positive_coefficients <= METAROOM_TAIL_MAX_POSITIVE_COEFFICIENTS);

            let started = Instant::now();
            let result = prepared
                .geometry
                .bound_margin_with_optimized_auxiliary_box_cut_unwired(
                    auxiliary,
                    &margin,
                    None,
                    metaroom_tail_dual_config(
                        iterations,
                        remaining.min(METAROOM_M24_MEMBER_WALL_CAP),
                    ),
                    metaroom_m24_optimizer_config(
                        schedule,
                        remaining.min(METAROOM_M24_CANDIDATE_WALL_CAP),
                    ),
                );

            let certificate = certificates
                .iter_mut()
                .find(|certificate| certificate.challenger_output == challenger_output)
                .expect("M24 challenger must belong to the sealed property");
            match result {
                Ok(result) => {
                    let (decision, member_errors) = retain_metaroom_m24_portfolio(
                        Ok::<f64, &'static str>(result.portfolio.original.lower_bound),
                        result
                            .portfolio
                            .auxiliary
                            .as_ref()
                            .map(|member| member.lower_bound)
                            .ok_or("M20 result unavailable"),
                        result
                            .portfolio
                            .box_cut
                            .as_ref()
                            .map(|member| member.lower_bound)
                            .ok_or("M24 exact replay unavailable"),
                    )
                    .expect("M24 API returned its mandatory M17 member");
                    assert_eq!(decision.winner, metaroom_m24_winner(result.selected));
                    assert_eq!(
                        decision.retained_lower_bound.to_bits(),
                        result.lower_bound.to_bits()
                    );
                    assert_eq!(
                        result.portfolio.lower_bound.to_bits(),
                        result.lower_bound.to_bits()
                    );
                    emit_metaroom_m24_observation(
                        phase,
                        challenger_output,
                        iterations,
                        &result,
                        decision,
                        &member_errors,
                        started.elapsed(),
                    );

                    // Record the mandatory exact M17 replay before the strict
                    // portfolio maximum. Optional-member failures and ties can
                    // therefore never erase the prior certified result.
                    certificate.observe_dual(result.portfolio.original.lower_bound);
                    certificate.observe_dual(decision.retained_lower_bound);
                }
                Err(error) => {
                    // The API can fail only before producing mandatory M17.
                    // Leave this challenge's prior certificate untouched and
                    // continue collecting bounded telemetry for later rows.
                    eprintln!(
                        "Metaroom119 hybrid tail {phase} M26 portfolio: target={METAROOM_TAIL_TARGET}, challenger={challenger_output}, requested_iterations={iterations}, m17_bound=None, m17_status=ApiError({error:?}), m17_iterations=0, m17_candidates=0, m20_bound=None, m20_status=None, m20_iterations=None, m20_candidates=None, m20_failure=Some(\"mandatory M17 failed\"), m24_bound=None, m24_replay_status=None, m24_search_status=None, m24_iterations=0, m24_restarts=0, m24_candidates=0, m24_exact_replays=0, m24_failure=Some(\"mandatory M17 failed\"), retained_lower_bound={:?}, winner=PriorCertificate, elapsed={:?}, m24_plan=None",
                        certificate.best_dual_lower_bound,
                        started.elapsed(),
                    );
                }
            }

            if remaining_metaroom_tail_budget(outer_deadline).is_none() {
                eprintln!(
                    "Metaroom119 hybrid tail {phase} M26: outer deadline reached after challenger {challenger_output}"
                );
                return false;
            }
        }
        true
    }

    fn assert_metaroom_preserve_hybrid_resources(trunk: &QualifiedMetaroomConvReluTrunk) {
        assert_eq!(trunk.relu_resources.len(), 4);
        assert_eq!(
            trunk
                .relu_resources
                .iter()
                .map(|row| (
                    row.stage,
                    row.input_alpha_dim,
                    row.output_alpha_dim,
                    row.unstable,
                    row.generator_nonzeros,
                    row.constraint_count,
                    row.constraint_elements,
                    row.nonzero_remainders,
                ))
                .collect::<Vec<_>>(),
            [
                ("ReLU1", 161, 637, 476, 15_348, 0, 0, 19_960),
                ("ReLU2", 637, 1_324, 687, 71_042, 0, 0, 12_172),
                ("ReLU3", 1_324, 1_929, 605, 155_320, 0, 0, 7_685),
                ("ReLU4", 1_929, 2_321, 392, 172_367, 0, 0, 704),
            ]
        );
        assert_eq!(trunk.relu4_output.value_dim(), 28_672);
        assert_eq!(trunk.relu4_output.alpha_dim(), 2_321);
        assert_eq!(trunk.relu4_output.constraint_count(), 0);
        assert!(trunk.relu4_auxiliary_counterfactual.is_none());
    }

    fn propagate_metaroom_preserve_hybrid_affine1(
        relu4_output: &ConstrainedZonotope64,
        tail: &QualifiedMetaroomAffineTail,
    ) -> ConstrainedZonotope64 {
        let limits = ConstrainedZonotopeAffineLimits {
            max_input_value_count: 28_672,
            max_output_value_count: 256,
            max_alpha_dim: 2_321,
            max_generator_nonzeros: 592_384,
            max_weight_elements: 7_340_032,
            max_matrix_visits: 7_340_032,
            max_interval_products: 51_646_197,
            max_constraint_count: 0,
            max_constraint_elements: 0,
        };
        let (output, plan) = constrained_zonotope_affine_unwired(
            relu4_output,
            tail.first_weights.view(),
            &tail.first_bias,
            limits,
        )
        .unwrap();
        assert_eq!(plan.input_value_count, 28_672);
        assert_eq!(plan.output_value_count, 256);
        assert_eq!(plan.alpha_dim, 2_321);
        assert_eq!(plan.input_generator_nonzeros, 172_367);
        assert_eq!(plan.output_generator_nonzeros, 592_384);
        assert_eq!(plan.constraint_count, 0);
        assert_eq!(plan.constraint_elements, 0);
        assert_eq!(plan.weight_elements, 7_340_032);
        assert_eq!(plan.matrix_visits, 7_340_032);
        assert_eq!(plan.interval_products, 51_646_197);
        assert_eq!(output.value_dim(), 256);
        assert_eq!(output.alpha_dim(), 2_321);
        assert_eq!(output.constraint_count(), 0);
        assert_eq!(
            output
                .generators()
                .iter()
                .map(ny_mip::SparseGenerator64::nnz)
                .sum::<usize>(),
            592_384
        );
        assert_eq!(
            output
                .box_remainder()
                .iter()
                .filter(|&&value| value != 0.0)
                .count(),
            256
        );
        output
    }

    #[test]
    fn hybrid_tail_stage_parser_is_exact_and_default_off() {
        assert_eq!(
            MetaroomHybridTailStage::parse("smoke0"),
            Some(MetaroomHybridTailStage::Smoke0)
        );
        assert_eq!(
            MetaroomHybridTailStage::parse("one8"),
            Some(MetaroomHybridTailStage::One8)
        );
        assert_eq!(
            MetaroomHybridTailStage::parse("all8"),
            Some(MetaroomHybridTailStage::All8)
        );
        assert_eq!(
            MetaroomHybridTailStage::parse("cascade"),
            Some(MetaroomHybridTailStage::Cascade)
        );
        assert_eq!(MetaroomHybridTailStage::parse("1"), None);
        assert_eq!(MetaroomHybridTailStage::parse(""), None);
    }

    #[test]
    fn m24_schedule_parser_is_exact_and_default_off() {
        assert_eq!(
            MetaroomM24Schedule::parse("bounded4x4"),
            Some(MetaroomM24Schedule::Bounded4x4)
        );
        assert_eq!(MetaroomM24Schedule::parse("bounded"), None);
        assert_eq!(MetaroomM24Schedule::parse("Bounded4x4"), None);
        assert_eq!(MetaroomM24Schedule::parse(""), None);
    }

    fn synthetic_metaroom_119_spec() -> VnnLibSpec {
        let mut spec = VnnLibSpec::new();
        spec.num_outputs = 20;
        spec.is_disjunction = true;
        spec.output_constraint_clauses = (0..20)
            .filter(|&challenger| challenger != METAROOM_TAIL_TARGET)
            .rev()
            .map(|challenger| {
                vec![OutputConstraint::GreaterEq(
                    challenger,
                    METAROOM_TAIL_TARGET,
                )]
            })
            .collect();
        spec
    }

    #[test]
    fn metaroom_119_contract_seals_target_coverage_and_clause_order() {
        let spec = synthetic_metaroom_119_spec();
        let contract = qualify_metaroom_119_unsafe_contract(&spec);
        assert_eq!(contract.target_output, METAROOM_TAIL_TARGET);
        assert_eq!(
            contract.challengers,
            (0..20)
                .filter(|&challenger| challenger != METAROOM_TAIL_TARGET)
                .rev()
                .collect::<Vec<_>>()
        );
        assert_eq!(contract.directions.dim(), (19, 20));
        for (row, &challenger) in contract.challengers.iter().enumerate() {
            assert_eq!(contract.directions[[row, challenger]], 1.0);
            assert_eq!(contract.directions[[row, METAROOM_TAIL_TARGET]], -1.0);
            assert_eq!(
                contract
                    .directions
                    .row(row)
                    .iter()
                    .filter(|&&coefficient| coefficient != 0.0)
                    .count(),
                2
            );
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed: !seen[*challenger]")]
    fn metaroom_119_contract_rejects_duplicate_challenger() {
        let mut spec = synthetic_metaroom_119_spec();
        spec.output_constraint_clauses[1] = spec.output_constraint_clauses[0].clone();
        let _ = qualify_metaroom_119_unsafe_contract(&spec);
    }

    #[test]
    fn hybrid_tail_resolution_is_strict_and_order_preserving() {
        let mut certificates = [
            MetaroomTailCertificateState::new(14),
            MetaroomTailCertificateState::new(2),
            MetaroomTailCertificateState::new(9),
        ];
        certificates[0].observe_dual(-1.0);
        certificates[0].observe_dual(0.25);
        certificates[1].observe_dual(0.0);
        certificates[2].observe_dual(-0.0);
        assert_eq!(unresolved_metaroom_tail_challengers(&certificates), [2, 9]);
        certificates[2].ay_strictly_positive = true;
        assert_eq!(unresolved_metaroom_tail_challengers(&certificates), [2]);
    }

    #[test]
    fn hybrid_tail_portfolio_retains_strict_max_and_attributes_winner() {
        let (auxiliary_wins, error) =
            retain_metaroom_tail_portfolio::<&'static str>(Ok(-3.0), Ok(0.25)).unwrap();
        assert_eq!(error, None);
        assert_eq!(auxiliary_wins.baseline_lower_bound, -3.0);
        assert_eq!(auxiliary_wins.auxiliary_lower_bound, Some(0.25));
        assert_eq!(auxiliary_wins.retained_lower_bound, 0.25);
        assert_eq!(
            auxiliary_wins.winner,
            MetaroomTailPortfolioWinner::AuxiliaryM20
        );

        let (baseline_wins, error) =
            retain_metaroom_tail_portfolio::<&'static str>(Ok(0.5), Ok(-2.0)).unwrap();
        assert_eq!(error, None);
        assert_eq!(baseline_wins.retained_lower_bound, 0.5);
        assert_eq!(
            baseline_wins.winner,
            MetaroomTailPortfolioWinner::BaselineM17
        );
    }

    #[test]
    fn hybrid_tail_portfolio_tie_and_auxiliary_error_preserve_m17() {
        let (tie, error) =
            retain_metaroom_tail_portfolio::<&'static str>(Ok(0.0), Ok(0.0)).unwrap();
        assert_eq!(error, None);
        assert_eq!(tie.retained_lower_bound, 0.0);
        assert_eq!(tie.winner, MetaroomTailPortfolioWinner::BaselineM17);

        let (fallback, error) =
            retain_metaroom_tail_portfolio(Ok(-1.5), Err("auxiliary rejected")).unwrap();
        assert_eq!(error, Some("auxiliary rejected"));
        assert_eq!(fallback.auxiliary_lower_bound, None);
        assert_eq!(fallback.retained_lower_bound, -1.5);
        assert_eq!(fallback.winner, MetaroomTailPortfolioWinner::BaselineM17);
        assert_eq!(
            retain_metaroom_tail_portfolio(Err("baseline rejected"), Ok(99.0)),
            Err("baseline rejected")
        );
    }

    #[test]
    fn m24_portfolio_strict_ties_retain_m17_then_m20() {
        let (all_tie, errors) =
            retain_metaroom_m24_portfolio::<&'static str>(Ok(0.25), Ok(0.25), Ok(0.25)).unwrap();
        assert_eq!(errors, [None, None]);
        assert_eq!(all_tie.retained_lower_bound, 0.25);
        assert_eq!(all_tie.winner, MetaroomM24PortfolioWinner::BaselineM17);

        let (m20_m24_tie, errors) =
            retain_metaroom_m24_portfolio::<&'static str>(Ok(-1.0), Ok(0.5), Ok(0.5)).unwrap();
        assert_eq!(errors, [None, None]);
        assert_eq!(m20_m24_tie.retained_lower_bound, 0.5);
        assert_eq!(m20_m24_tie.winner, MetaroomM24PortfolioWinner::AuxiliaryM20);

        let (m24_wins, errors) = retain_metaroom_m24_portfolio::<&'static str>(
            Ok(-1.0),
            Ok(0.5),
            Ok(0.500_000_000_000_000_1),
        )
        .unwrap();
        assert_eq!(errors, [None, None]);
        assert_eq!(m24_wins.winner, MetaroomM24PortfolioWinner::BoxCutM24);
    }

    #[test]
    fn m24_portfolio_failures_retain_best_prior_certificate() {
        let (m24_failure, errors) =
            retain_metaroom_m24_portfolio(Ok(-1.0), Ok(0.75), Err("M24 replay failed")).unwrap();
        assert_eq!(errors, [None, Some("M24 replay failed")]);
        assert_eq!(m24_failure.retained_lower_bound, 0.75);
        assert_eq!(m24_failure.winner, MetaroomM24PortfolioWinner::AuxiliaryM20);

        let (both_optional_fail, errors) =
            retain_metaroom_m24_portfolio(Ok(-2.0), Err("M20 failed"), Err("M24 unavailable"))
                .unwrap();
        assert_eq!(errors, [Some("M20 failed"), Some("M24 unavailable")]);
        assert_eq!(both_optional_fail.retained_lower_bound, -2.0);
        assert_eq!(
            both_optional_fail.winner,
            MetaroomM24PortfolioWinner::BaselineM17
        );
        assert_eq!(
            retain_metaroom_m24_portfolio(Err("M17 failed"), Ok(99.0), Ok(100.0),),
            Err("M17 failed")
        );
    }

    #[test]
    fn hybrid_tail_prepared_geometry_serves_both_portfolio_members() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0, 0.0], &[1.0, 2.0], &[false; 2])
                .unwrap();
        let prepared = MetaroomPreparedTailDomain::try_new(&domain).unwrap();
        assert_eq!(prepared.geometry.value_dim(), domain.value_dim());
        assert_eq!(prepared.geometry.coordinate_hull_generator_additions(), 2);

        let declared =
            exact_relu_tail_margin_from_f64_rows(&[1.0, -1.0], &[0.0, 0.0], 0.0, 0.0).unwrap();
        let auxiliary =
            CertifiedAuxiliaryBounds64::try_new(vec![-1.0, 0.0], vec![1.0, 2.0]).unwrap();
        let original = prepared
            .geometry
            .bound_margin_unwired(
                &declared,
                None,
                metaroom_tail_dual_config(0, Duration::from_secs(1)),
            )
            .unwrap();
        let with_auxiliary = prepared
            .geometry
            .bound_margin_with_auxiliary_bounds_unwired(
                &auxiliary,
                &declared,
                None,
                metaroom_tail_dual_config(0, Duration::from_secs(1)),
            )
            .unwrap();
        assert_eq!(
            original.lower_bound.to_bits(),
            with_auxiliary.lower_bound.to_bits()
        );
    }

    #[test]
    fn hybrid_tail_resource_configs_seal_measured_caps() {
        let i8 = metaroom_tail_dual_config(8, Duration::from_secs(5));
        assert_eq!(i8.limits.max_alpha_dim, 2_321);
        assert_eq!(i8.limits.max_constraints, 0);
        assert_eq!(i8.limits.max_generator_nonzeros, 592_384);
        assert_eq!(i8.limits.max_optimizable_slopes, 135);
        assert_eq!(i8.limits.max_search_work, METAROOM_TAIL_I8_SEARCH_WORK);
        let i20 = metaroom_tail_dual_config(20, Duration::from_secs(10));
        assert_eq!(i20.limits.max_search_work, METAROOM_TAIL_I20_SEARCH_WORK);

        let m24 = metaroom_m24_optimizer_config(
            MetaroomM24Schedule::Bounded4x4,
            METAROOM_M24_CANDIDATE_WALL_CAP,
        );
        assert_eq!(
            MetaroomM24Schedule::Bounded4x4.tail_outer_budget(),
            Duration::from_mins(8)
        );
        assert_eq!(METAROOM_M24_MEMBER_WALL_CAP, Duration::from_secs(3));
        assert_eq!(METAROOM_M24_CANDIDATE_WALL_CAP, Duration::from_secs(1));
        assert_eq!(m24.schedules[0].iterations, 4);
        assert_eq!(m24.schedules[0].learning_rate, 0.005);
        assert_eq!(m24.schedules[1].iterations, 4);
        assert_eq!(m24.schedules[1].learning_rate, 0.1);
        assert_eq!(m24.multiplier_cap, 16.0);
        assert_eq!(m24.limits.max_value_dim, 256);
        assert_eq!(m24.limits.max_box_variables, METAROOM_M24_MAX_BOX_VARIABLES);
        assert_eq!(
            m24.limits.max_total_iterations,
            METAROOM_M24_TOTAL_ITERATIONS
        );
        assert_eq!(m24.limits.max_restarts, 2);
        assert_eq!(m24.limits.max_exact_replays, 2);
        assert_eq!(m24.limits.max_generator_nonzeros, 592_384);
        assert_eq!(m24.limits.max_search_work, METAROOM_M24_MAX_SEARCH_WORK);
        assert_eq!(m24.limits.max_wall_time, Duration::from_secs(1));
        // Worst case over all 256 upper and lower endpoints at the measured
        // 2,321-alpha/592,384-nonzero geometry, two starts, and eight updates.
        let score = 256_u64 * 3 + 512 * 2 + 592_384 * 2 + 2_321;
        let restart_startup = 512_u64 * 6 + 256 * 2 + score;
        let per_iteration = 512_u64 * 6 + score;
        assert_eq!(
            256 + restart_startup * 2 + per_iteration * 8,
            METAROOM_M24_MAX_SEARCH_WORK
        );

        let lp = metaroom_tail_lp_config(6, 19, Duration::from_mins(3));
        assert_eq!(lp.limits.max_model_columns, 3_128);
        assert_eq!(lp.limits.max_model_rows, 807);
        assert_eq!(lp.limits.max_model_nonzeros, 599_117);
        assert_eq!(lp.limits.max_solves, 19);
    }

    #[test]
    #[ignore = "default-off guarded M16->(M17+M20[+M24])->M15 real diagnostic; set NY_CZ_HYBRID_TAIL_DIAGNOSTIC=smoke0|one8|all8|cascade and optional NY_CZ_HYBRID_TAIL_M24_SCHEDULE=bounded4x4, then run explicitly under ny-safe-gpu-run"]
    fn real_metaroom_119_diagnoses_inductive_hybrid_tail_cascade() {
        let Some(stage) = requested_metaroom_hybrid_tail_stage() else {
            return;
        };
        let m24_schedule = requested_metaroom_m24_schedule();
        if m24_schedule.is_some() {
            assert_eq!(
                stage,
                MetaroomHybridTailStage::All8,
                "{METAROOM_M24_SCHEDULE_ENV} is qualified only with {METAROOM_HYBRID_TAIL_ENV}=all8"
            );
        }

        // The first traversal is an independent Box trace. Its large CZ is
        // released before the M16 hybrid traversal, while its four certified
        // preactivation enclosures remain as the only hybrid-trunk auxiliary
        // inputs.
        let mut box_diagnostic = MetaroomBoxDiagnostic::default();
        let Some(baseline) = qualify_real_metaroom_119_full_conv_relu_trunk_with_box(
            FullTrunkPredicateMode::Preserve,
            &mut box_diagnostic,
        ) else {
            return;
        };
        assert_eq!(box_diagnostic.relus.len(), 4);
        assert_eq!(box_diagnostic.auxiliary_preactivations.len(), 4);
        let property = qualify_metaroom_119_unsafe_contract(&baseline.vnnlib_spec);
        assert_eq!(property.target_output, METAROOM_TAIL_TARGET);
        assert!(property
            .challengers
            .contains(&METAROOM_TAIL_SMOKE_CHALLENGER));
        let tail = qualify_real_metaroom_119_affine_tail_topology(
            &baseline.model,
            &baseline.graph,
            &baseline.graph_relu4_name,
            &baseline.model_relu4_output,
        );
        let auxiliary_trace = std::mem::take(&mut box_diagnostic.auxiliary_preactivations);
        drop(baseline);

        let hybrid = qualify_real_metaroom_119_full_conv_relu_trunk_with_auxiliary_trace(
            FullTrunkPredicateMode::Preserve,
            &auxiliary_trace,
        )
        .unwrap();
        assert_metaroom_preserve_hybrid_resources(&hybrid);
        let replayed_property = qualify_metaroom_119_unsafe_contract(&hybrid.vnnlib_spec);
        assert_eq!(replayed_property.target_output, property.target_output);
        assert_eq!(replayed_property.challengers, property.challengers);
        assert_eq!(replayed_property.directions, property.directions);
        let QualifiedMetaroomConvReluTrunk {
            model,
            graph,
            vnnlib_spec,
            relu4_output,
            graph_relu4_name,
            model_relu4_output,
            relu_resources,
            relu4_auxiliary_counterfactual,
        } = hybrid;
        drop((
            model,
            graph,
            vnnlib_spec,
            graph_relu4_name,
            model_relu4_output,
            relu_resources,
            relu4_auxiliary_counterfactual,
            auxiliary_trace,
        ));

        let affine1_output = propagate_metaroom_preserve_hybrid_affine1(&relu4_output, &tail);
        drop(relu4_output);

        // Continue the independently propagated Box to the exact affine1
        // preactivation consumed by ReLU5. `affine_relu` records that same
        // CertifiedBox64 through `try_from_certified_box` before applying ReLU;
        // M20 consumes the typed copy, while neither M17 nor affine1_output is
        // altered by it.
        box_diagnostic.affine_relu(
            "ReLU5",
            tail.first_weights.view(),
            &tail.first_bias,
            &affine1_output,
        );
        assert_eq!(box_diagnostic.relus.len(), 5);
        assert_eq!(box_diagnostic.auxiliary_preactivations.len(), 1);
        assert_eq!(
            box_diagnostic
                .relus
                .iter()
                .map(|report| report.stage)
                .collect::<Vec<_>>(),
            ["ReLU1", "ReLU2", "ReLU3", "ReLU4", "ReLU5"]
        );
        assert_eq!(box_diagnostic.relus[4].value_count, 256);
        assert_eq!(box_diagnostic.relus[4].ignored_cz_constraints, 0);
        let relu5_auxiliary = box_diagnostic.auxiliary_preactivations.pop().unwrap();
        assert_eq!(relu5_auxiliary.value_dim(), 256);
        assert_eq!(relu5_auxiliary.value_dim(), affine1_output.value_dim());
        assert_eq!(tail.first_weights.nrows(), relu5_auxiliary.value_dim());
        assert_eq!(tail.first_bias.len(), relu5_auxiliary.value_dim());
        assert_eq!(tail.second_weights.ncols(), relu5_auxiliary.value_dim());
        let box_limits = metaroom_box_limits();
        assert!(relu5_auxiliary.value_dim() <= box_limits.max_values);
        assert!(relu5_auxiliary.value_dim().checked_mul(2).unwrap() <= box_limits.max_stored_f64);
        assert!(relu5_auxiliary
            .lower()
            .iter()
            .chain(relu5_auxiliary.upper())
            .all(|endpoint| endpoint.is_finite()));
        box_diagnostic.terminal_affine(tail.second_weights.view(), &tail.second_bias);
        assert_eq!(box_diagnostic.terminal.as_ref().unwrap().len(), 20);
        eprintln!(
            "Metaroom119 hybrid Box-only ReLU5 diagnostic: comparison={:#?}, terminal_lower={:?}, terminal_upper={:?}",
            box_diagnostic.relus[4],
            box_diagnostic.terminal.as_ref().unwrap().lower(),
            box_diagnostic.terminal.as_ref().unwrap().upper(),
        );
        drop(box_diagnostic);

        let selected_challengers = if stage.is_single_challenger() {
            vec![METAROOM_TAIL_SMOKE_CHALLENGER]
        } else {
            property.challengers.clone()
        };
        let mut certificates = selected_challengers
            .iter()
            .copied()
            .map(MetaroomTailCertificateState::new)
            .collect::<Vec<_>>();
        let tail_started = Instant::now();
        let tail_outer_budget = m24_schedule.map_or_else(
            || stage.tail_outer_budget(),
            MetaroomM24Schedule::tail_outer_budget,
        );
        let outer_deadline = tail_started
            .checked_add(tail_outer_budget)
            .expect("Metaroom hybrid tail outer deadline must be representable");
        let preparation_started = Instant::now();
        let prepared = MetaroomPreparedTailDomain::try_new(&affine1_output).unwrap();
        assert_eq!(prepared.geometry.value_dim(), 256);
        assert_eq!(
            prepared.geometry.coordinate_hull_generator_additions(),
            592_384
        );
        let portfolio_members_sharing_hull = if m24_schedule.is_some() { 3 } else { 2 };
        eprintln!(
            "Metaroom119 hybrid tail prepared-M21-M23: value_dim={}, coordinate_hull_generator_additions_once={}, portfolio_members_sharing_hull={portfolio_members_sharing_hull}, full_property_challengers=19, direct_portfolio_hull_additions=22510592, prepared_portfolio_hull_additions=592384, avoided_hull_additions=21918208, elapsed={:?}",
            prepared.geometry.value_dim(),
            prepared
                .geometry
                .coordinate_hull_generator_additions(),
            preparation_started.elapsed(),
        );
        let primary_phase = match stage {
            MetaroomHybridTailStage::Smoke0 => "smoke0",
            MetaroomHybridTailStage::One8 => "one8",
            MetaroomHybridTailStage::All8 | MetaroomHybridTailStage::Cascade => "all8",
        };
        let primary_complete = if let Some(schedule) = m24_schedule {
            run_metaroom_m24_pass(
                primary_phase,
                &prepared,
                &relu5_auxiliary,
                &tail,
                property.target_output,
                &selected_challengers,
                stage.primary_iterations(),
                schedule,
                outer_deadline,
                &mut certificates,
            )
        } else {
            run_metaroom_tail_dual_pass(
                primary_phase,
                &prepared,
                &relu5_auxiliary,
                &tail,
                property.target_output,
                &selected_challengers,
                stage.primary_iterations(),
                outer_deadline,
                &mut certificates,
            )
        };

        let mut retry_complete = primary_complete;
        if stage == MetaroomHybridTailStage::Cascade && primary_complete {
            let retry = unresolved_metaroom_tail_challengers(&certificates);
            if !retry.is_empty() {
                retry_complete = run_metaroom_tail_dual_pass(
                    "retry20",
                    &prepared,
                    &relu5_auxiliary,
                    &tail,
                    property.target_output,
                    &retry,
                    20,
                    outer_deadline,
                    &mut certificates,
                );
            }
        }

        if stage == MetaroomHybridTailStage::Cascade && retry_complete {
            let ay_challengers = unresolved_metaroom_tail_challengers(&certificates);
            if !ay_challengers.is_empty() {
                if let Some(remaining) = remaining_metaroom_tail_budget(outer_deadline)
                    .filter(|remaining| *remaining >= Duration::from_secs(1))
                {
                    let ay = diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_for_challengers_unwired(
                        &affine1_output,
                        tail.second_weights.view(),
                        &tail.second_bias,
                        &ay_challengers,
                        metaroom_tail_lp_config(
                            property.target_output,
                            ay_challengers.len(),
                            remaining.min(Duration::from_mins(3)),
                        ),
                    )
                    .unwrap();
                    assert_eq!(ay.target_output, property.target_output);
                    assert_eq!(ay.plan.solve_count, ay_challengers.len());
                    assert_eq!(
                        ay.margins
                            .iter()
                            .map(|margin| margin.challenger_output)
                            .collect::<Vec<_>>(),
                        ay_challengers
                    );
                    assert!(ay.plan.model_columns <= 3_109 + ay_challengers.len());
                    assert!(ay.plan.model_rows <= 788 + ay_challengers.len());
                    assert!(ay.plan.model_nonzeros <= 599_060 + 3 * ay_challengers.len());
                    assert!(ay.exact_milp.within_declared_caps);

                    let exact_zero = BigRational::from_integer(0.into());
                    for margin in &ay.margins {
                        let strictly_positive = matches!(
                            &margin.outcome,
                            TailLpMarginOutcome::RigorousLowerBound(bound) if bound > &exact_zero
                        );
                        certificates
                            .iter_mut()
                            .find(|certificate| {
                                certificate.challenger_output == margin.challenger_output
                            })
                            .unwrap()
                            .ay_strictly_positive = strictly_positive;
                    }
                    eprintln!(
                        "Metaroom119 hybrid tail selected AY fallback: requested={ay_challengers:?}, plan={:#?}, all_positive={}, unresolved={}, minimum={:?}, exact_milp={:#?}, elapsed={:?}, margins={:#?}",
                        ay.plan,
                        ay.all_margins_strictly_positive,
                        ay.unresolved_margin_count,
                        ay.minimum_rigorous_lower_bound,
                        ay.exact_milp,
                        ay.elapsed,
                        ay.margins,
                    );
                } else {
                    eprintln!(
                        "Metaroom119 hybrid tail: outer deadline left no AY construction budget; requested={ay_challengers:?}"
                    );
                }
            }
        }

        let unresolved = unresolved_metaroom_tail_challengers(&certificates);
        if m24_schedule.is_some() {
            eprintln!(
                "Metaroom119 hybrid tail final diagnostic: stage={stage:?}, m24_schedule={m24_schedule:?}, selected={}, strictly_positive={}, unresolved={unresolved:?}, outer_elapsed={:?}, certificates={certificates:#?}",
                certificates.len(),
                certificates.len() - unresolved.len(),
                tail_started.elapsed(),
            );
        } else {
            eprintln!(
                "Metaroom119 hybrid tail final diagnostic: stage={stage:?}, selected={}, strictly_positive={}, unresolved={unresolved:?}, outer_elapsed={:?}, certificates={certificates:#?}",
                certificates.len(),
                certificates.len() - unresolved.len(),
                tail_started.elapsed(),
            );
        }
        // Diagnostic only: no value above is connected to a command, verifier
        // verdict, VNN-COMP score, or any default-on path.
    }

    #[test]
    #[ignore = "default-off exact AY LP tail diagnostic; set NY_CZ_TAIL_AY_LP_DIAGNOSTIC=1 and run explicitly under ny-safe-gpu-run"]
    fn real_metaroom_119_diagnoses_target_6_tail_with_exact_ay_lp() {
        if std::env::var("NY_CZ_TAIL_AY_LP_DIAGNOSTIC").as_deref() != Ok("1") {
            return;
        }
        let Some(trunk) =
            qualify_real_metaroom_119_full_conv_relu_trunk(FullTrunkPredicateMode::Preserve)
        else {
            return;
        };
        // Seal the exact VNN-LIB contract before constructing or solving the
        // hard-coded target-6 tail. Any clause-shape, target, coverage, or
        // output-dimension drift must fail this diagnostic closed.
        let unsafe_directions = metaroom_119_unsafe_clause_directions(&trunk.vnnlib_spec);
        assert_eq!(unsafe_directions.dim(), (19, 20));
        let target_output = unsafe_directions
            .row(0)
            .iter()
            .position(|coefficient| *coefficient == -1.0)
            .expect("qualified Metaroom119 direction must subtract the target output");
        let expected_output_dim = unsafe_directions.ncols();
        let expected_margin_count = unsafe_directions.nrows();
        let tail = qualify_real_metaroom_119_affine_tail_topology(
            &trunk.model,
            &trunk.graph,
            &trunk.graph_relu4_name,
            &trunk.model_relu4_output,
        );

        // Reconstruct exactly the qualified post-/9/Gemm CZ.  This first
        // diagnostic intentionally uses the measured preserve-mode trunk:
        // the reusable primitive still lowers arbitrary C alpha <= d rows,
        // while this guarded run stays below the host's memory firewall.
        let first_weight_elements = 256_usize.checked_mul(28_672).unwrap();
        let first_limits = ConstrainedZonotopeAffineLimits {
            max_input_value_count: 28_672,
            max_output_value_count: 256,
            max_alpha_dim: 3_133,
            max_generator_nonzeros: 800_511,
            max_weight_elements: first_weight_elements,
            max_matrix_visits: first_weight_elements,
            max_interval_products: 141_372_078,
            max_constraint_count: 0,
            max_constraint_elements: 0,
        };
        let (affine1_output, affine1_plan) = constrained_zonotope_affine_unwired(
            &trunk.relu4_output,
            tail.first_weights.view(),
            &tail.first_bias,
            first_limits,
        )
        .unwrap();
        assert_eq!(affine1_plan.output_generator_nonzeros, 800_511);
        assert_eq!(affine1_output.value_dim(), 256);
        assert_eq!(affine1_output.alpha_dim(), 3_133);
        assert_eq!(affine1_output.constraint_count(), 0);

        let diagnostic = diagnose_constrained_zonotope_relu_affine_tail_with_ay_lp_unwired(
            &affine1_output,
            tail.second_weights.view(),
            &tail.second_bias,
            ConstrainedZonotopeTailLpConfig {
                target_output,
                wall_time: Duration::from_mins(3),
                ay_memory_budget_bytes: 1_536 * 1_024 * 1_024,
                exact_milp_binary_cap: 256,
                limits: ConstrainedZonotopeTailLpLimits {
                    max_value_dim: 256,
                    max_alpha_dim: 3_133,
                    max_constraints: 0,
                    max_constraint_elements: 0,
                    max_generator_nonzeros: 800_511,
                    max_constraint_nonzeros: 0,
                    max_output_dim: expected_output_dim,
                    max_unstable_relus: 256,
                    max_model_columns: 4_000,
                    max_model_rows: 1_024,
                    max_model_nonzeros: 850_000,
                    max_solves: expected_margin_count,
                },
            },
        )
        .unwrap();

        eprintln!(
            "real Metaroom119 target-6 AY tail LP: plan={:#?}; all_positive={}; minimum_rigorous_bound={:?}; unresolved={}; exact_milp={:#?}; elapsed={:?}; margins={:#?}",
            diagnostic.plan,
            diagnostic.all_margins_strictly_positive,
            diagnostic.minimum_rigorous_lower_bound,
            diagnostic.unresolved_margin_count,
            diagnostic.exact_milp,
            diagnostic.elapsed,
            diagnostic.margins,
        );
        assert_eq!(diagnostic.target_output, target_output);
        assert_eq!(diagnostic.plan.output_dim, expected_output_dim);
        assert_eq!(diagnostic.plan.solve_count, expected_margin_count);
        assert_eq!(diagnostic.margins.len(), expected_margin_count);
        assert_eq!(diagnostic.exact_milp.relu_binary_count, 217);
        assert!(diagnostic.exact_milp.within_declared_caps);
    }
}
