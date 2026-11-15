// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

const MAX_RANK: usize = 8;
const MAX_NAME: usize = 4096;

/// The private constructor borrows the exact current graph and boxes. There is
/// no caller-authored descriptor or stale cross-query cache to trust.
pub(super) struct Region<'a> {
    pub(super) sources: [&'a str; 3],
    pub(super) q: &'a BoundedTensor,
    pub(super) k: &'a BoundedTensor,
    pub(super) v: &'a BoundedTensor,
    pub(super) scale: f32,
    pub(super) authored_kt: bool,
    pub(super) output_len: usize,
    sq: usize,
    sk: usize,
    dv: usize,
    slices: usize,
}

fn product(layer: &Layer) -> Option<(bool, Option<f32>)> {
    match layer {
        Layer::BilinearCrown(x) => Some((x.transpose_b(), x.scale())),
        Layer::MatMul(x) => Some((x.transpose_b, x.scale)),
        _ => None,
    }
}

fn bounded_node<'a>(graph: &'a GraphNetwork, name: &str) -> Option<&'a GraphNode> {
    if name.len() > MAX_NAME {
        return None;
    }
    let node = graph.nodes.get(name)?;
    if node.name.len() > MAX_NAME
        || node.inputs.len() > 3
        || node.inputs.iter().any(|s| s.len() > MAX_NAME)
    {
        return None;
    }
    Some(node)
}

fn endpoints<'a>(
    name: &str,
    input: &'a BoundedTensor,
    boxes: &'a HashMap<String, BoundedTensor>,
) -> Option<&'a BoundedTensor> {
    if name.len() > MAX_NAME {
        return None;
    }
    let tensor = if name == NETWORK_INPUT {
        input
    } else {
        boxes.get(name)?
    };
    let shape = tensor.shape();
    if shape.len() < 2
        || shape.len() > MAX_RANK
        || shape.contains(&0)
        || ny_core::checked_shape_product(shape)? != tensor.len()
    {
        return None;
    }
    Some(tensor)
}

pub(super) fn recognize<'a>(
    graph: &'a GraphNetwork,
    pv: &'a GraphNode,
    input: &'a BoundedTensor,
    boxes: &'a HashMap<String, BoundedTensor>,
) -> Result<Option<Region<'a>>> {
    let Some((transpose_v, pv_scale)) = product(&pv.layer) else {
        return Ok(None);
    };
    if transpose_v
        || pv_scale.is_some_and(|s| s.to_bits() != 1.0f32.to_bits())
        || pv.name.len() > MAX_NAME
        || pv.inputs.len() != 2
        || pv.inputs.iter().any(|s| s.len() > MAX_NAME)
    {
        return Ok(None);
    }
    let Some(softmax) = bounded_node(graph, &pv.inputs[0]) else {
        return Ok(None);
    };
    let Layer::Softmax(sm) = &softmax.layer else {
        return Ok(None);
    };
    if !sm.sound || softmax.inputs.len() != 1 {
        return Ok(None);
    }
    let Some(probabilities) = endpoints(&pv.inputs[0], input, boxes) else {
        return Ok(None);
    };
    let rank = probabilities.shape().len();
    let Ok(axis) =
        crate::layers::common::resolve_axis_i32(sm.axis, rank, "attention region Softmax")
    else {
        return Ok(None);
    };
    if axis != rank - 1 {
        return Ok(None);
    }
    let Some(mut qk) = bounded_node(graph, &softmax.inputs[0]) else {
        return Ok(None);
    };
    let mut authored_scale = None;
    if let Layer::MulConstant(mul) = &qk.layer {
        if qk.inputs.len() != 1 || mul.constant.len() != 1 || mul.constant.ndim() > 1 {
            return Ok(None);
        }
        authored_scale = mul.constant.iter().next().copied();
        let Some(pre_scale) = bounded_node(graph, &qk.inputs[0]) else {
            return Ok(None);
        };
        let Some(pre_scale_box) = endpoints(&qk.inputs[0], input, boxes) else {
            return Ok(None);
        };
        if pre_scale_box.shape() != probabilities.shape() {
            return Ok(None);
        }
        qk = pre_scale;
    }
    let Some((transpose_k, qk_scale)) = product(&qk.layer) else {
        return Ok(None);
    };
    if qk.inputs.len() != 2 || (authored_scale.is_some() && qk_scale.is_some()) {
        return Ok(None);
    }
    // A missing scalar in a decomposed MatMul means exactly one, whereas a
    // missing scalar in SelfAttention infers sqrt(d). Always pass Some here.
    let scale = authored_scale.or(qk_scale).unwrap_or(1.0);
    if !scale.is_finite() || scale <= 0.0 {
        return Ok(None);
    }
    let Some(q) = endpoints(&qk.inputs[0], input, boxes) else {
        return Ok(None);
    };
    let Some(k) = endpoints(&qk.inputs[1], input, boxes) else {
        return Ok(None);
    };
    let Some(v) = endpoints(&pv.inputs[1], input, boxes) else {
        return Ok(None);
    };
    if q.shape().len() != rank || k.shape().len() != rank || v.shape().len() != rank {
        return Ok(None);
    }
    let qs = q.shape();
    let ks = k.shape();
    let vs = v.shape();
    if qs[..rank - 2] != ks[..rank - 2]
        || qs[..rank - 2] != vs[..rank - 2]
        || qs[..rank - 2] != probabilities.shape()[..rank - 2]
    {
        return Ok(None);
    }
    let (sq, dk) = (qs[rank - 2], qs[rank - 1]);
    let (sk, kk) = if transpose_k {
        (ks[rank - 2], ks[rank - 1])
    } else {
        (ks[rank - 1], ks[rank - 2])
    };
    if kk != dk
        || vs[rank - 2] != sk
        || probabilities.shape()[rank - 2] != sq
        || probabilities.shape()[rank - 1] != sk
    {
        return Ok(None);
    }
    // When the original MatMul has no transpose flag, its second tensor IS
    // already K^T. Bind an explicit original producer permutation; the checked
    // canonical-cache indexing maps back to this producer's OUTPUT, not to an
    // invented canonical upstream node. The original producer still executes
    // normally for all pending paths, including its authored head permutation.
    if !transpose_k {
        let Some(k_node) = bounded_node(graph, &qk.inputs[1]) else {
            return Ok(None);
        };
        let Layer::Transpose(t) = &k_node.layer else {
            return Ok(None);
        };
        if k_node.inputs.len() != 1 || t.axes.len() > MAX_RANK + 1 {
            return Ok(None);
        }
        let Some(before) = endpoints(&k_node.inputs[0], input, boxes) else {
            return Ok(None);
        };
        if before.shape().len() != rank {
            return Ok(None);
        }
        let dropped = t.axes.len().saturating_sub(rank);
        if t.axes.len() < rank || dropped > 1 {
            return Ok(None);
        }
        let mut seen = [false; MAX_RANK + 1];
        let mut destination = 0;
        for &a in &t.axes {
            if a >= t.axes.len() || seen[a] {
                return Ok(None);
            }
            seen[a] = true;
            if a >= dropped {
                if destination >= rank || ks[destination] != before.shape()[a - dropped] {
                    return Ok(None);
                }
                destination += 1;
            }
        }
        if destination != rank {
            return Ok(None);
        }
    }
    let Some(score_box) = endpoints(&qk.name, input, boxes) else {
        return Ok(None);
    };
    let Some(softmax_input) = endpoints(&softmax.inputs[0], input, boxes) else {
        return Ok(None);
    };
    if score_box.shape() != probabilities.shape() || softmax_input.shape() != probabilities.shape()
    {
        return Ok(None);
    }
    let Some(output) = endpoints(&pv.name, input, boxes) else {
        return Ok(None);
    };
    if output.shape().len() != rank
        || output.shape()[..rank - 2] != qs[..rank - 2]
        || output.shape()[rank - 2] != sq
        || output.shape()[rank - 1] != vs[rank - 1]
    {
        return Ok(None);
    }
    let slices = ny_core::checked_shape_product(&qs[..rank - 2]).unwrap_or(0);
    if slices == 0 {
        return Ok(None);
    }
    Ok(Some(Region {
        sources: [
            qk.inputs[0].as_str(),
            qk.inputs[1].as_str(),
            pv.inputs[1].as_str(),
        ],
        q,
        k,
        v,
        scale,
        authored_kt: !transpose_k,
        output_len: output.len(),
        sq,
        sk,
        dv: vs[rank - 1],
        slices,
    }))
}

impl Region<'_> {
    pub(super) fn request(
        &self,
        rows: usize,
        input: usize,
        incoming_bytes: usize,
    ) -> Result<WorkRequest> {
        let count = self
            .q
            .len()
            .checked_add(self.k.len())
            .and_then(|n| n.checked_add(self.v.len()))
            .and_then(|n| n.checked_add(input))
            .and_then(|n| n.checked_mul(rows))
            .ok_or_else(overflow)?;
        // Four-f32-equivalent buffers per cell cover retained pairs plus
        // coefficient errors/metadata; eight further buffers bound certified
        // f64 merge/downcast overlap. Includes shared bias's network-input row.
        let persistent = count
            .checked_mul(16)
            .and_then(|n| n.checked_add(HEADER_BYTES))
            .ok_or_else(overflow)?;
        let scratch = count
            .checked_mul(64)
            .and_then(|n| incoming_bytes.checked_mul(4).and_then(|a| n.checked_add(a)))
            .and_then(|n| n.checked_add(HEADER_BYTES))
            .ok_or_else(overflow)?;
        // Original K endpoints are borrowed in either layout. Direct authored
        // indexing removes the old 8*K endpoint transpose allocation. Keep all
        // conservative merge/output/center-cache and work terms unchanged.
        let work = u64::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(64))
            .ok_or_else(overflow)?;
        Ok(WorkRequest {
            scratch_bytes: scratch,
            persistent_bytes: persistent,
            work,
            proposals: 0,
        })
    }

    pub(super) fn kernel_scratch(&self, rows: usize) -> Result<usize> {
        let input = self
            .q
            .len()
            .checked_add(self.k.len())
            .and_then(|n| n.checked_add(self.v.len()))
            .ok_or_else(overflow)?;
        input
            .checked_mul(24)
            .and_then(|n| {
                input
                    .checked_mul(rows)
                    .and_then(|r| r.checked_mul(8))
                    .and_then(|r| n.checked_add(r))
            })
            .and_then(|n| self.sk.checked_mul(256).and_then(|x| n.checked_add(x)))
            .and_then(|n| self.dv.checked_mul(8).and_then(|x| n.checked_add(x)))
            .and_then(|n| rows.checked_mul(64).and_then(|x| n.checked_add(x)))
            .and_then(|n| {
                self.slices
                    .checked_mul(self.sq)
                    .and_then(|x| n.checked_add(x))
            })
            .and_then(|n| n.checked_add(32 * 1024))
            .ok_or_else(overflow)
    }
}

fn overflow() -> NyError {
    NyError::UnsupportedOp("attention region admission overflow".into())
}
