// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#relu-pair-collapse`: undo a ReLU *split* exactly.
//!
//! The relusplitter construction (VNN-COMP 2026 `relusplitter_2026`) replaces
//! a layer `z = ReLU(W x + b)` by
//!
//! ```text
//! t = A x + a          (Gemm1)
//! h = ReLU(t)          (Relu1)
//! u = B h + c          (Gemm2)
//! z = ReLU(u)          (Relu2, kept)
//! ```
//!
//! where every row `i` of `B` is either
//!
//! * a SPLIT row: exactly `+1` at column `p` and `-1` at column `q`, with
//!   `A[q] == -A[p]` and `a[q] == -a[p]` BIT FOR BIT. Then
//!   `B[i]·h = ReLU(t_p) - ReLU(-t_p) = t_p` exactly, for every real `t_p`; or
//! * a PASS-THROUGH row: exactly `+1` at column `p` and `c[i] == 0`. Then
//!   `ReLU(B[i]·h + c[i]) = ReLU(ReLU(t_p)) = ReLU(t_p)` — which needs the
//!   following ReLU, so pass-through rows are admitted only when `u` feeds
//!   exactly one `Relu`.
//!
//! In both cases `z_i = ReLU(A[p_i] x + a[p_i] + c[i])` as a real-valued
//! function. The rewrite emits exactly that, as two affine layers and NO
//! folding: `t' = A_sel x + a_sel` (rows `A[p_i]`, `a[p_i]` copied bit for
//! bit), then `u = I t' + c`. Nothing is rounded, so the rewritten graph is the
//! SAME real function as the authored one, not an approximation of it. What it
//! removes is the verifier's view of each split pair as two unstable ReLUs,
//! whose chord relaxation α-CROWN cannot tighten.
//!
//! Any deviation from the exact pattern declines the whole rewrite: a nonzero
//! entry other than the single `+1` / `-1`, an entry that is not exactly
//! `±1.0`, a pair whose rows are not exact negations, a pass-through row with a
//! nonzero bias, non-default Gemm attributes, a tensor with another consumer,
//! or an intermediate that is a graph output.

use crate::model::{AttributeValue, DataType, LayerSpec, WeightRef};
use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};
use ny_core::LayerType;
use std::collections::{HashMap, HashSet};

use super::helpers::fused_subgraph_is_closed;

/// What one successful collapse emits: the two replacement layers keyed by the
/// node index they replace (Gemm1 and Gemm2), plus every consumed node.
pub(in crate::loader) struct ReluPairCollapse {
    pub(in crate::loader) starts: Vec<(usize, LayerSpec)>,
    pub(in crate::loader) consumed: HashSet<usize>,
    pub(in crate::loader) split_rows: usize,
    pub(in crate::loader) passthrough_rows: usize,
}

/// Try the collapse with `start_idx` as Gemm1.
pub(in crate::loader) fn try_collapse_relu_pairs(
    nodes: &[onnx_proto::NodeProto],
    start_idx: usize,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    weights: &mut WeightStore,
    graph_output_names: &HashSet<String>,
) -> Option<ReluPairCollapse> {
    let gemm1 = nodes.get(start_idx)?;
    let (x_name, a_mat, a_bias) = read_default_gemm(gemm1, weights)?;
    let t_name = gemm1.output.first()?.clone();
    let relu_idx = single_consumer(consumers_by_input, &t_name)?;
    let relu = nodes.get(relu_idx)?;
    if relu.op_type != "Relu" || relu.input.len() != 1 || relu.output.len() != 1 {
        return None;
    }
    let h_name = relu.output.first()?.clone();
    let gemm2_idx = single_consumer(consumers_by_input, &h_name)?;
    let gemm2 = nodes.get(gemm2_idx)?;
    let (h_in, b_mat, c_bias) = read_default_gemm(gemm2, weights)?;
    if h_in != h_name {
        return None;
    }
    let u_name = gemm2.output.first()?.clone();
    let (hidden, d) = (a_mat.nrows(), a_mat.ncols());
    if b_mat.ncols() != hidden || a_bias.len() != hidden || c_bias.len() != b_mat.nrows() {
        return None;
    }

    // Classify every row of B; any deviation declines the whole rewrite.
    let rows = b_mat.nrows();
    let mut selected = Vec::with_capacity(rows);
    let (mut split_rows, mut passthrough_rows) = (0usize, 0usize);
    for i in 0..rows {
        let (mut plus, mut minus) = (None, None);
        for (j, &v) in b_mat.row(i).iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let slot = if v.to_bits() == 1.0f32.to_bits() {
                &mut plus
            } else if v.to_bits() == (-1.0f32).to_bits() {
                &mut minus
            } else {
                return None;
            };
            if slot.replace(j).is_some() {
                return None;
            }
        }
        let p = plus?;
        match minus {
            Some(q) => {
                let exact_negation = a_mat
                    .row(q)
                    .iter()
                    .zip(a_mat.row(p).iter())
                    .all(|(&aq, &ap)| aq.to_bits() == (-ap).to_bits())
                    && a_bias[q].to_bits() == (-a_bias[p]).to_bits();
                if !exact_negation {
                    return None;
                }
                split_rows += 1;
            }
            None => {
                if c_bias[i] != 0.0 {
                    return None;
                }
                passthrough_rows += 1;
            }
        }
        selected.push(p);
    }
    if split_rows == 0 {
        return None;
    }
    // A pass-through row is exact only UNDER A ReLU: ReLU(ReLU(t) + 0) =
    // ReLU(t), but ReLU(t) + 0 != t. So when any row passes through, the
    // chain's output must feed exactly one Relu (split rows are exact either
    // way: ReLU(t) - ReLU(-t) = t).
    if passthrough_rows > 0 {
        let feeds_one_relu = single_consumer(consumers_by_input, &u_name)
            .and_then(|i| nodes.get(i))
            .is_some_and(|n| n.op_type == "Relu" && n.input.len() == 1);
        if !feeds_one_relu || graph_output_names.contains(u_name.as_str()) {
            return None;
        }
    }

    let consumed_nodes = [start_idx, relu_idx, gemm2_idx];
    if [&t_name, &h_name]
        .iter()
        .any(|name| graph_output_names.contains(name.as_str()))
        || !fused_subgraph_is_closed(
            nodes,
            &consumed_nodes,
            std::slice::from_ref(&u_name),
            consumers_by_input,
            graph_output_names,
        )
    {
        return None;
    }

    let base = if gemm1.name.is_empty() {
        t_name
    } else {
        gemm1.name.clone()
    };
    let names = [
        format!("{base}__relu_pair_collapse_sel_weight"),
        format!("{base}__relu_pair_collapse_sel_bias"),
        format!("{base}__relu_pair_collapse_sel_out"),
        format!("{base}__relu_pair_collapse_id_weight"),
        format!("{base}__relu_pair_collapse_id_bias"),
    ];
    if names
        .iter()
        .any(|n| name_is_taken(n, nodes, weights, graph_output_names))
    {
        return None;
    }
    let [sel_w, sel_b, sel_out, id_w, id_b] = names;

    let sel_weight = Array2::from_shape_fn((rows, d), |(i, k)| a_mat[[selected[i], k]]);
    let sel_bias = Array1::from_shape_fn(rows, |i| a_bias[selected[i]]);
    let id_weight = Array2::from_shape_fn((rows, rows), |(i, k)| if i == k { 1.0 } else { 0.0 });

    let spec1 = linear_spec(
        format!("{base}__relu_pair_collapse_sel"),
        x_name,
        sel_out.clone(),
        &sel_w,
        &sel_b,
        [rows, d],
    );
    let spec2 = linear_spec(
        format!("{base}__relu_pair_collapse_id"),
        sel_out,
        u_name,
        &id_w,
        &id_b,
        [rows, rows],
    );
    weights.insert(sel_w, sel_weight.into_dyn());
    weights.insert(sel_b, sel_bias.into_dyn());
    weights.insert(id_w, id_weight.into_dyn());
    weights.insert(id_b, c_bias.into_dyn());

    Some(ReluPairCollapse {
        starts: vec![(start_idx, spec1), (gemm2_idx, spec2)],
        consumed: consumed_nodes.into_iter().collect(),
        split_rows,
        passthrough_rows,
    })
}

fn linear_spec(
    name: String,
    input: String,
    output: String,
    weight_name: &str,
    bias_name: &str,
    shape: [usize; 2],
) -> LayerSpec {
    let mut attributes = HashMap::new();
    attributes.insert("transB".to_string(), AttributeValue::Int(1));
    LayerSpec {
        name,
        layer_type: LayerType::Linear,
        inputs: vec![input, weight_name.to_string(), bias_name.to_string()],
        outputs: vec![output],
        weights: Some(WeightRef {
            name: weight_name.to_string(),
            shape: shape.to_vec(),
            original_dtype: DataType::Float32,
        }),
        attributes,
    }
}

/// A standard-semantics Gemm `Y = X W' + C` with `alpha == beta == 1`,
/// `transA == 0`, a constant 2-D weight and a constant 1-D bias. Returns the
/// data input, the weight as `[out, in]`, and the bias.
fn read_default_gemm(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<(String, Array2<f32>, Array1<f32>)> {
    if node.op_type != "Gemm" || node.input.len() != 3 || node.output.len() != 1 {
        return None;
    }
    let mut trans_b = false;
    for attr in &node.attribute {
        match attr.name.as_str() {
            "alpha" | "beta" if attr.f_value().to_bits() != 1.0f32.to_bits() => return None,
            "alpha" | "beta" => {}
            "transA" if attr.i_value() != 0 => return None,
            "transA" => {}
            "transB" => trans_b = attr.i_value() != 0,
            _ => return None,
        }
    }
    let x = node.input.first()?.clone();
    if x.is_empty() || weights.get(x.as_str()).is_some() {
        return None;
    }
    let w: &ArrayD<f32> = weights.get(node.input.get(1)?.as_str())?;
    let w = w.clone().into_dimensionality::<Ix2>().ok()?;
    let w = if trans_b { w } else { w.t().to_owned() };
    let c: &ArrayD<f32> = weights.get(node.input.get(2)?.as_str())?;
    let c = c.clone().into_dimensionality::<Ix1>().ok()?;
    if c.len() != w.nrows() || w.iter().chain(c.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    Some((x, w, c))
}

fn single_consumer(consumers_by_input: &HashMap<&str, Vec<usize>>, name: &str) -> Option<usize> {
    let consumers = consumers_by_input.get(name)?;
    (consumers.len() == 1).then_some(consumers[0])
}

fn name_is_taken(
    candidate: &str,
    nodes: &[onnx_proto::NodeProto],
    weights: &WeightStore,
    graph_output_names: &HashSet<String>,
) -> bool {
    weights.contains_key(candidate)
        || graph_output_names.contains(candidate)
        || nodes.iter().any(|node| {
            node.input.iter().any(|name| name == candidate)
                || node.output.iter().any(|name| name == candidate)
        })
}
