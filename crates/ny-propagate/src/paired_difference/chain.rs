// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural admission of two complete unary affine/ReLU chains.
//!
//! Parameters are copied from their authored binary32 values by exact widening.
//! In particular, normalization and successive affine layers are NEVER composed:
//! a rounded composed weight would describe a different mathematical network.

use std::collections::HashSet;

use ndarray::ArrayD;

use crate::layers::Layer;
use crate::network::{GraphNetwork, NETWORK_INPUT};
use crate::shape::{broadcast_flat_index_map, broadcast_shapes};

use super::{PairedChain, Stage};

const MAX_INPUTS: usize = 16;
const MAX_NEURONS: usize = 10_000;
const MAX_NODES: usize = 128;
const MAX_COEFFICIENTS: usize = 2_000_000;
const MAX_RANK: usize = 8;

enum SingleStage {
    Affine {
        rows: usize,
        cols: usize,
        weights: Vec<f64>,
        bias: Vec<f64>,
    },
    Relu(usize),
}

fn shape_len(shape: &[usize]) -> Result<usize, String> {
    if shape.len() > MAX_RANK || shape.contains(&0) {
        return Err("empty or excessive-rank tensor shape".into());
    }
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .filter(|&k| k <= MAX_NEURONS)
            .ok_or_else(|| "tensor shape exceeds paired-chain resource cap".into())
    })
}

/// Broadcast a constant using the same layout as the admitted layer. Require
/// that the operand's flat indices are unchanged: expansion of the DATA tensor
/// would need a larger affine stage and is deliberately refused here.
fn constant_values(
    constant: &ArrayD<f32>,
    effective_shape: &[usize],
    input_shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), String> {
    let output_shape = broadcast_shapes(input_shape, effective_shape)
        .ok_or_else(|| "constant is not broadcastable over the chain operand".to_string())?;
    let width = shape_len(input_shape)?;
    if shape_len(&output_shape)? != width
        || shape_len(effective_shape)? != constant.len()
        || broadcast_flat_index_map(&output_shape, input_shape)
            .iter()
            .enumerate()
            .any(|(i, &j)| i != j)
    {
        return Err("constant expands or reorders the chain operand".into());
    }
    let flat: Vec<f64> = constant.iter().map(|&v| f64::from(v)).collect();
    if flat.iter().any(|v| !v.is_finite()) {
        return Err("non-finite constant parameter".into());
    }
    let indices = broadcast_flat_index_map(&output_shape, effective_shape);
    let values = indices
        .into_iter()
        .map(|i| {
            flat.get(i)
                .copied()
                .ok_or_else(|| "invalid constant index".into())
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((values, output_shape))
}

fn diagonal(scale: Vec<f64>, bias: Vec<f64>) -> Result<SingleStage, String> {
    let width = scale.len();
    let count = width
        .checked_mul(width)
        .filter(|&n| n <= MAX_COEFFICIENTS)
        .ok_or_else(|| "normalization matrix exceeds paired-chain cap".to_string())?;
    let mut weights = vec![0.0; count];
    for (i, s) in scale.into_iter().enumerate() {
        weights[i * width + i] = s;
    }
    Ok(SingleStage::Affine {
        rows: width,
        cols: width,
        weights,
        bias,
    })
}

fn extract(graph: &GraphNetwork, input_dim: usize) -> Result<(Vec<SingleStage>, usize), String> {
    if graph.node_names().is_empty() || graph.node_names().len() > MAX_NODES {
        return Err("empty graph or paired-chain depth cap exceeded".into());
    }
    // Walk the OUTPUT ancestry rather than trusting insertion or topological
    // order. Requiring every graph node on this path excludes hidden branches,
    // unused operators, and any synthetic sentinel outside the real chain.
    let mut cursor = graph.output_name();
    let mut names = Vec::new();
    let mut visited = HashSet::new();
    while cursor != NETWORK_INPUT {
        if !visited.insert(cursor.to_string()) {
            return Err("cycle in paired-chain graph".into());
        }
        let node = graph
            .node(cursor)
            .ok_or_else(|| format!("missing chain node `{cursor}`"))?;
        if node.inputs().len() != 1 {
            return Err(format!(
                "chain node `{cursor}` has {} operands",
                node.inputs().len()
            ));
        }
        names.push(cursor.to_string());
        cursor = &node.inputs()[0];
    }
    if visited.len() != graph.node_names().len() {
        return Err("graph contains a branch or an unused node".into());
    }
    names.reverse();
    let mut shape = graph
        .declared_shape(NETWORK_INPUT)
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| vec![input_dim]);
    if shape_len(&shape)? != input_dim {
        return Err("declared chain input shape does not match the property".into());
    }
    let mut width = input_dim;
    let mut neurons = 0usize;
    let mut coefficients = 0usize;
    let mut stages = Vec::new();
    for name in names {
        let node = graph
            .node(&name)
            .ok_or_else(|| "missing chain node".to_string())?;
        let stage = match node.layer() {
            Layer::Linear(linear) => {
                let (rows, cols) = linear.weight().dim();
                if cols != width
                    || shape.last().copied() != Some(cols)
                    || rows == 0
                    || rows > MAX_NEURONS
                {
                    return Err(format!("Linear `{name}` has incompatible dimensions"));
                }
                let count = rows
                    .checked_mul(cols)
                    .filter(|&n| n <= MAX_COEFFICIENTS)
                    .ok_or_else(|| "affine matrix exceeds paired-chain cap".to_string())?;
                let mut weights = Vec::with_capacity(count);
                for i in 0..rows {
                    for j in 0..cols {
                        weights.push(f64::from(linear.weight()[[i, j]]));
                    }
                }
                let bias = match linear.bias() {
                    Some(b) if b.len() == rows => b.iter().map(|&v| f64::from(v)).collect(),
                    Some(_) => return Err("affine bias has incompatible dimensions".into()),
                    None => vec![0.0; rows],
                };
                if weights.iter().chain(&bias).any(|v| !v.is_finite()) {
                    return Err("non-finite affine parameter".into());
                }
                width = rows;
                // Linear acts on the last axis. The preceding axes denote
                // exactly one vector (cols == total width), so preserve their
                // runtime shape instead of inventing a rank from metadata.
                *shape.last_mut().ok_or("Linear input has no data axis")? = rows;
                Some(SingleStage::Affine {
                    rows,
                    cols,
                    weights,
                    bias,
                })
            }
            Layer::ReLU(_) => Some(SingleStage::Relu(width)),
            Layer::AddConstant(c) => {
                let effective = c
                    .effective_broadcast_shape(&shape)
                    .map_err(|e| format!("AddConstant `{name}` layout: {e}"))?;
                let (values, out_shape) = constant_values(c.constant(), &effective, &shape)?;
                shape = out_shape;
                Some(diagonal(vec![1.0; width], values)?)
            }
            Layer::SubConstant(c) => {
                let (mut values, out_shape) =
                    constant_values(c.constant(), c.constant().shape(), &shape)?;
                shape = out_shape;
                let scale = if c.reverse { -1.0 } else { 1.0 };
                if !c.reverse {
                    for value in &mut values {
                        *value = -*value;
                    }
                }
                Some(diagonal(vec![scale; width], values)?)
            }
            Layer::MulConstant(c) => {
                if c.input_shape().is_some_and(|s| s != shape.as_slice()) {
                    return Err(format!(
                        "MulConstant `{name}` has inconsistent input metadata"
                    ));
                }
                let (values, out_shape) =
                    constant_values(c.constant(), c.constant().shape(), &shape)?;
                shape = out_shape;
                Some(diagonal(values, vec![0.0; width])?)
            }
            Layer::Flatten(flatten) => {
                shape = flatten
                    .compute_output_shape(&shape)
                    .map_err(|e| format!("Flatten `{name}`: {e}"))?;
                None
            }
            Layer::Reshape(reshape) => {
                // Guard the layer's target-product computation before calling
                // it. Exotic dynamic copy-axis sentinels are outside this lane.
                let mut known = 1usize;
                if reshape.target_shape.len() > MAX_RANK {
                    return Err("reshape rank exceeds paired-chain cap".into());
                }
                for (axis, &dim) in reshape.target_shape.iter().enumerate() {
                    if dim < -1 {
                        return Err("dynamic reshape is not admitted".into());
                    }
                    if dim >= 0 {
                        let extent = if dim == 0 {
                            *shape
                                .get(axis)
                                .ok_or_else(|| "reshape copy axis exceeds input rank".to_string())?
                        } else {
                            dim as usize
                        };
                        known = known
                            .checked_mul(extent)
                            .filter(|&n| n <= MAX_NEURONS)
                            .ok_or_else(|| "reshape target exceeds paired-chain cap".to_string())?;
                    }
                }
                shape = reshape
                    .compute_output_shape(&shape)
                    .map_err(|e| format!("Reshape `{name}`: {e}"))?;
                None
            }
            _ => {
                return Err(format!(
                    "operator at `{name}` is outside affine/ReLU chain admission"
                ))
            }
        };
        if let Some(declared) = graph.declared_shape(&name) {
            if shape_len(declared)? != width {
                return Err(format!(
                    "declared shape at `{name}` disagrees with chain width"
                ));
            }
            // Metadata is a consistency guard; it NEVER changes the runtime
            // shape used by a later copy-axis Reshape or broadcast. NY strips
            // the leading singleton ONNX batch from declared shapes, while
            // layers such as Flatten keep it in their finite runtime output.
            // Removing that singleton preserves ordered values. All other
            // differences are rejected, except Linear's existing single-
            // vector metadata convention (also ignored for runtime shape).
            if matches!(node.layer(), Layer::Linear(_)) {
                if declared.last().copied() != Some(width)
                    || declared.iter().filter(|&&d| d != 1).count() > 1
                {
                    return Err(format!("Linear `{name}` is not a single vector"));
                }
            } else if declared != shape.as_slice()
                && !(shape.first() == Some(&1) && declared == &shape[1..])
            {
                return Err(format!(
                    "declared shape at `{name}` disagrees with operator semantics"
                ));
            }
        }
        if shape_len(&shape)? != width {
            return Err(format!(
                "shape-changing operator at `{name}` changes ordered values"
            ));
        }
        neurons = neurons
            .checked_add(width)
            .filter(|&n| n <= MAX_NEURONS)
            .ok_or_else(|| "paired-chain total-neuron cap exceeded".to_string())?;
        if let Some(stage) = stage {
            if let SingleStage::Affine { weights, .. } = &stage {
                coefficients = coefficients
                    .checked_add(weights.len())
                    .filter(|&n| n <= MAX_COEFFICIENTS)
                    .ok_or_else(|| "paired-chain total-coefficient cap exceeded".to_string())?;
            }
            stages.push(stage);
        }
    }
    if stages.is_empty() {
        return Err("chain has no admitted stages".into());
    }
    Ok((stages, width))
}

impl PairedChain {
    /// Admit two complete chains without altering either network's parameters.
    /// Admission depends on structure and mathematical operator semantics, never
    /// on a benchmark/category name or observed pointwise agreement.
    pub fn from_graphs(
        f: &GraphNetwork,
        g: &GraphNetwork,
        input_dim: usize,
    ) -> Result<Self, String> {
        if input_dim == 0 || input_dim > MAX_INPUTS {
            return Err("paired-chain input dimension is outside 1..=16".into());
        }
        let (sf, output_dim) = extract(f, input_dim)?;
        let (sg, output_g) = extract(g, input_dim)?;
        if sf.len() != sg.len() || output_dim != output_g {
            return Err("paired chains have different stage or output dimensions".into());
        }
        let stages = sf
            .into_iter()
            .zip(sg)
            .map(|(a, b)| match (a, b) {
                (
                    SingleStage::Affine {
                        rows,
                        cols,
                        weights: wf,
                        bias: bf,
                    },
                    SingleStage::Affine {
                        rows: gr,
                        cols: gc,
                        weights: wg,
                        bias: bg,
                    },
                ) if rows == gr && cols == gc => Ok(Stage::Affine {
                    rows,
                    cols,
                    wf,
                    wg,
                    bf,
                    bg,
                }),
                (SingleStage::Relu(width), SingleStage::Relu(gw)) if width == gw => {
                    Ok(Stage::Relu { width })
                }
                _ => Err("paired chains have incompatible corresponding stages".into()),
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            input_dim,
            output_dim,
            stages,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{AddConstantLayer, LinearLayer, ReLULayer, SubConstantLayer};
    use crate::network::GraphNode;
    use ndarray::{arr1, arr2};

    fn tower(delta: f32, mean: f32) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "normalize",
            Layer::SubConstant(SubConstantLayer::scalar(mean)),
            vec![NETWORK_INPUT.into()],
        ));
        graph.add_node(GraphNode::new(
            "linear",
            Layer::Linear(
                LinearLayer::new(
                    arr2(&[[1.0, -0.5], [0.25, 1.0 + delta]]),
                    Some(arr1(&[0.2, -0.1])),
                )
                .unwrap(),
            ),
            vec!["normalize".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["linear".into()],
        ));
        graph.add_node(GraphNode::new(
            "head",
            Layer::Linear(LinearLayer::new(arr2(&[[0.3, -0.6]]), None).unwrap()),
            vec!["relu".into()],
        ));
        graph.set_output("head");
        graph
    }

    fn forward(chain: &PairedChain, mut x: Vec<f64>, g: bool) -> Vec<f64> {
        for stage in &chain.stages {
            x = match stage {
                Stage::Relu { .. } => x.into_iter().map(|v| v.max(0.0)).collect(),
                Stage::Affine {
                    rows,
                    cols,
                    wf,
                    wg,
                    bf,
                    bg,
                } => {
                    let (w, b) = if g { (wg, bg) } else { (wf, bf) };
                    (0..*rows)
                        .map(|i| b[i] + (0..*cols).map(|j| w[i * cols + j] * x[j]).sum::<f64>())
                        .collect()
                }
            };
        }
        x
    }

    #[test]
    fn nearby_twin_authored_stages_preserve_forward_values() {
        let delta = 0.002_f32;
        let chain = PairedChain::from_graphs(&tower(0.0, 0.1), &tower(delta, 0.12), 2).unwrap();
        assert_eq!(
            chain.stages.len(),
            4,
            "normalization is its OWN exact affine stage"
        );
        for k in -10..=10 {
            let x = [f64::from(k) / 4.0, f64::from(7 - k) / 8.0];
            for (g, mean, perturb) in [(false, 0.1_f32, 0.0_f32), (true, 0.12_f32, delta)] {
                let u = [x[0] - f64::from(mean), x[1] - f64::from(mean)];
                let z0 = (u[0] - 0.5 * u[1] + f64::from(0.2_f32)).max(0.0);
                let z1 = (0.25 * u[0] + f64::from(1.0_f32 + perturb) * u[1] + f64::from(-0.1_f32))
                    .max(0.0);
                let expected = f64::from(0.3_f32) * z0 + f64::from(-0.6_f32) * z1;
                assert!((forward(&chain, x.to_vec(), g)[0] - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn unused_branch_is_refused() {
        let mut f = tower(0.0, 0.0);
        f.add_node(GraphNode::new(
            "unused",
            Layer::AddConstant(AddConstantLayer::new(arr1(&[1.0, 1.0]).into_dyn())),
            vec![NETWORK_INPUT.into()],
        ));
        assert!(PairedChain::from_graphs(&f, &tower(0.0, 0.0), 2)
            .unwrap_err()
            .contains("branch"));
    }

    #[test]
    fn mismatched_width_and_property_dimensions_are_refused() {
        let f = tower(0.0, 0.0);
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "linear",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0, 1.0], [2.0, 2.0]]), None).unwrap()),
            vec![NETWORK_INPUT.into()],
        ));
        g.set_output("linear");
        assert!(PairedChain::from_graphs(&f, &g, 2).is_err());
        assert!(PairedChain::from_graphs(&f, &f, 3).is_err());
    }

    #[test]
    fn reverse_normalization_keeps_its_exact_sign() {
        let mut f = GraphNetwork::new();
        f.add_node(GraphNode::new(
            "norm",
            Layer::SubConstant(SubConstantLayer::new_reverse(
                arr1(&[0.5, -0.25]).into_dyn(),
            )),
            vec![NETWORK_INPUT.into()],
        ));
        f.set_output("norm");
        let g = f.clone();
        let chain = PairedChain::from_graphs(&f, &g, 2).unwrap();
        assert_eq!(forward(&chain, vec![0.25, 0.5], false), vec![0.25, -0.75]);
    }

    #[test]
    fn broadcasting_cannot_expand_the_data_tensor() {
        let mut f = GraphNetwork::new();
        f.set_declared_shape(NETWORK_INPUT, vec![2, 1]);
        f.add_node(GraphNode::new(
            "add",
            Layer::AddConstant(AddConstantLayer::new(arr1(&[1.0, 2.0]).into_dyn())),
            vec![NETWORK_INPUT.into()],
        ));
        f.set_output("add");
        assert!(PairedChain::from_graphs(&f, &f, 2)
            .unwrap_err()
            .contains("expands"));
    }
}
