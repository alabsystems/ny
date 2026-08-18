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
#[path = "cz_metaroom_qualification.rs"]
mod tests;
