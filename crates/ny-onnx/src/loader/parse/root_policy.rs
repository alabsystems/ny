// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded metadata-only admission on the already decoded AUTHORED graph.
//! No tensor payload copy, graph clone, model reload, inference, or environment read.
use crate::onnx_proto::{AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto};
use std::collections::{BTreeMap, BTreeSet};

const MAX_NODES: usize = 128;
const MAX_INITIALIZERS: usize = 512;
const MAX_NAME_BYTES: usize = 64 << 10;
const MAX_DIM: usize = 1 << 20;
const MAX_BYTES: usize = 256 << 20;
#[derive(Clone, Copy, Debug)]
pub(super) struct Quote {
    pub nodes: usize,
    pub extraction_bytes: usize,
    pub bn_channels: usize,
    pub conv_nodes: usize,
}
fn size(s: &[usize]) -> Option<usize> {
    let n = s.iter().try_fold(1usize, |a, &b| a.checked_mul(b))?;
    (n > 0 && n <= MAX_DIM).then_some(n)
}
fn shape(v: &ValueInfoProto) -> Option<Vec<usize>> {
    let t = v.r#type.as_ref()?.tensor_type.as_ref()?;
    if t.elem_type != 1 {
        return None;
    }
    let d = &t.shape.as_ref()?.dim;
    if d.is_empty() || d.len() > 4 {
        return None;
    }
    let out = d
        .iter()
        .map(|x| match x.value.as_ref()? {
            crate::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(v) => {
                usize::try_from(*v).ok().filter(|&v| v > 0)
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    size(&out)?;
    Some(out)
}
// The admitted unary chain independently derives its concrete output shape.
// A symbolic leading output annotation is not a second runtime batch input:
// resolve_batch_dim fixes the original input batch to one, and every admitted
// operator below preserves or explicitly checks that batch. Do not rewrite the
// authored graph or accept an unknown/mismatching nonleading dimension.
fn output_shape_matches(v: &ValueInfoProto, derived: &[usize]) -> bool {
    let Some(t) = v.r#type.as_ref().and_then(|t| t.tensor_type.as_ref()) else {
        return false;
    };
    let Some(shape) = t.shape.as_ref() else {
        return false;
    };
    if t.elem_type != 1
        || !matches!(derived.len(), 2 | 4)
        || derived[0] != 1
        || shape.dim.len() != derived.len()
    {
        return false;
    }
    shape
        .dim
        .iter()
        .zip(derived)
        .enumerate()
        .all(|(i, (dim, &want))| match dim.value.as_ref() {
            Some(crate::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(got)) => {
                usize::try_from(*got).ok() == Some(want)
            }
            Some(crate::onnx_proto::tensor_shape_proto::dimension::Value::DimParam(label)) => {
                i == 0 && !label.is_empty() && label.len() <= 1024
            }
            None => false,
        })
}
fn tensor(t: &TensorProto) -> Option<usize> {
    if t.dims.is_empty()
        || t.dims.len() > 4
        || t.segment.is_some()
        || t.data_location != 0
        || !t.external_data.is_empty()
        || !t.int32_data.is_empty()
        || !t.double_data.is_empty()
        || !t.string_data.is_empty()
        || !t.uint64_data.is_empty()
    {
        return None;
    }
    let n = t.dims.iter().try_fold(1usize, |a, &v| {
        a.checked_mul(usize::try_from(v).ok().filter(|&v| v > 0)?)
    })?;
    if n > MAX_BYTES / 8 {
        return None;
    }
    match t.data_type {
        1 if t.int64_data.is_empty()
            && ((t.raw_data.len() == n.checked_mul(4)? && t.float_data.is_empty())
                || (t.raw_data.is_empty() && t.float_data.len() == n)) =>
        {
            Some(n)
        }
        7 if n <= 4
            && t.dims.len() == 1
            && t.float_data.is_empty()
            && ((t.raw_data.len() == n.checked_mul(8)? && t.int64_data.is_empty())
                || (t.raw_data.is_empty() && t.int64_data.len() == n)) =>
        {
            Some(n)
        }
        _ => None,
    }
}
fn attr<'a>(n: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    n.attribute.iter().find(|a| a.name == name)
}
fn int(n: &NodeProto, name: &str, default: i64) -> Option<i64> {
    match attr(n, name) {
        None => Some(default),
        Some(a) if a.r#type == 2 => Some(a.i_value()),
        _ => None,
    }
}
fn unit(n: &NodeProto, name: &str) -> bool {
    attr(n, name).is_none_or(|a| a.r#type == 1 && a.f_value().to_bits() == 1f32.to_bits())
}
fn ints(n: &NodeProto, name: &str, default: &[i64]) -> Option<Vec<i64>> {
    match attr(n, name) {
        None => Some(default.to_vec()),
        Some(a) if a.r#type == 7 && a.ints.len() <= 4 => Some(a.ints.clone()),
        _ => None,
    }
}
fn float_weight<'a>(m: &BTreeMap<&str, &'a TensorProto>, name: &str) -> Option<&'a TensorProto> {
    let t = *m.get(name)?;
    (t.data_type == 1).then_some(t)
}
fn bias(m: &BTreeMap<&str, &TensorProto>, n: &NodeProto, out: usize) -> Option<bool> {
    if n.input.len() == 2 {
        return Some(false);
    }
    if n.input.len() != 3 {
        return None;
    }
    let t = float_weight(m, &n.input[2])?;
    (t.dims == [out as i64] || t.dims == [1, out as i64]).then_some(true)
}
fn names_ok(g: &GraphProto) -> Option<()> {
    let mut bytes = 0usize;
    for s in g
        .initializer
        .iter()
        .map(|x| x.name.as_str())
        .chain(g.input.iter().chain(&g.output).map(|x| x.name.as_str()))
        .chain(g.node.iter().flat_map(|n| {
            std::iter::once(n.name.as_str())
                .chain(n.input.iter().chain(&n.output).map(String::as_str))
        }))
    {
        if s.len() > 1024 {
            return None;
        }
        bytes = bytes.checked_add(s.len())?;
        if bytes > MAX_NAME_BYTES {
            return None;
        }
    }
    Some(())
}

pub(super) fn quote(g: &GraphProto) -> Option<Quote> {
    if g.node.is_empty()
        || g.node.len() > MAX_NODES
        || g.initializer.len() > MAX_INITIALIZERS
        || g.input.len() != 1
        || g.output.len() != 1
        || !g.sparse_initializer.is_empty()
    {
        return None;
    }
    // Bound entry counts BEFORE the aggregate-name walk: arbitrarily many
    // empty names consume no byte allowance and must not create an unbounded
    // metadata scan. Each predicate below is O(1) per already-bounded node.
    if g.node.iter().any(|n| {
        n.input.is_empty() || n.input.len() > 5 || n.output.len() != 1 || n.attribute.len() > 8
    }) {
        return None;
    }
    names_ok(g)?;
    // convert_node_to_layer uses authored node.name, or output[0] when empty;
    // GraphNetwork subsequently keys nodes by that effective layer name.
    let mut node_keys = BTreeSet::new();
    for n in &g.node {
        let key = if n.name.is_empty() {
            n.output[0].as_str()
        } else {
            n.name.as_str()
        };
        if key.is_empty() || key == ny_propagate::NETWORK_INPUT || !node_keys.insert(key) {
            return None;
        }
    }
    let mut initializers = BTreeMap::new();
    for t in &g.initializer {
        tensor(t)?;
        if t.name.is_empty() || initializers.insert(t.name.as_str(), t).is_some() {
            return None;
        }
    }
    let mut dims = shape(&g.input[0])?;
    if dims[0] != 1
        || !matches!(dims.len(), 2 | 4)
        || initializers.contains_key(g.input[0].name.as_str())
    {
        return None;
    }
    let mut current = g.input[0].name.as_str();
    let mut seen = BTreeSet::new();
    seen.insert(current);
    let mut words = size(&dims)?;
    let mut bn_channels = 0usize;
    let mut relus = 0;
    let mut conv_nodes = 0;
    for n in &g.node {
        if !n.domain.is_empty() && n.domain != "ai.onnx" {
            return None;
        }
        if n.input.is_empty()
            || n.input.len() > 5
            || n.input[0] != current
            || n.output.len() != 1
            || n.output[0].is_empty()
            || initializers.contains_key(n.output[0].as_str())
            || !seen.insert(n.output[0].as_str())
            || n.attribute.len() > 8
        {
            return None;
        }
        let allowed: &[&str] = match n.op_type.as_str() {
            "Conv" => &[
                "auto_pad",
                "dilations",
                "group",
                "kernel_shape",
                "pads",
                "strides",
            ],
            "Gemm" => &["alpha", "beta", "transA", "transB"],
            "MatMul" | "Relu" => &[],
            "BatchNormalization" => &["epsilon", "momentum", "training_mode"],
            "Flatten" => &["axis"],
            "Reshape" => &["allowzero"],
            _ => return None,
        };
        let mut attrs = BTreeSet::new();
        for a in &n.attribute {
            if !allowed.contains(&a.name.as_str())
                || !attrs.insert(a.name.as_str())
                || a.ints.len() > 4
                || a.s.as_ref().is_some_and(|s| s.len() > 16)
            {
                return None;
            }
        }
        let before = size(&dims)?;
        match n.op_type.as_str() {
            "Conv" => {
                conv_nodes += 1;
                if dims.len() != 4
                    || int(n, "group", 1)? != 1
                    || ints(n, "dilations", &[1, 1])? != [1, 1]
                    || attr(n, "auto_pad")
                        .is_some_and(|a| a.r#type != 3 || a.s_value() != b"NOTSET")
                {
                    return None;
                }
                if !(2..=3).contains(&n.input.len()) {
                    return None;
                }
                let w = float_weight(&initializers, &n.input[1])?;
                if w.dims.len() != 4 || w.dims[1] != dims[1] as i64 {
                    return None;
                }
                let c = usize::try_from(w.dims[0]).ok()?;
                let kh = usize::try_from(w.dims[2]).ok()?;
                let kw = usize::try_from(w.dims[3]).ok()?;
                if attr(n, "kernel_shape").is_some()
                    && ints(n, "kernel_shape", &[])? != [kh as i64, kw as i64]
                {
                    return None;
                }
                let p = ints(n, "pads", &[0, 0, 0, 0])?;
                let s = ints(n, "strides", &[1, 1])?;
                if p.len() != 4
                    || s.len() != 2
                    || p.iter().any(|&x| x < 0)
                    || p[0] != p[2]
                    || p[1] != p[3]
                    || s.iter().any(|&x| x <= 0)
                {
                    return None;
                }
                let oh = dims[2]
                    .checked_add(usize::try_from(p[0]).ok()?.checked_mul(2)?)?
                    .checked_sub(kh)?
                    .checked_div(usize::try_from(s[0]).ok()?)?
                    .checked_add(1)?;
                let ow = dims[3]
                    .checked_add(usize::try_from(p[1]).ok()?.checked_mul(2)?)?
                    .checked_sub(kw)?
                    .checked_div(usize::try_from(s[1]).ok()?)?
                    .checked_add(1)?;
                dims = vec![1, c, oh, ow];
                let out = size(&dims)?;
                words = words
                    .checked_add(tensor(w)?)?
                    .checked_add(if bias(&initializers, n, c)? { out } else { 0 })?;
            }
            "Gemm" | "MatMul" => {
                if dims.len() != 2
                    || !(2..=3).contains(&n.input.len())
                    || int(n, "transA", 0)? != 0
                    || !unit(n, "alpha")
                    || !unit(n, "beta")
                {
                    return None;
                }
                if n.op_type == "MatMul" && n.input.len() != 2 {
                    return None;
                }
                let transpose = int(n, "transB", 0)?;
                if transpose != 0 && transpose != 1 {
                    return None;
                }
                let w = float_weight(&initializers, &n.input[1])?;
                if w.dims.len() != 2 {
                    return None;
                }
                let (inner, out) = if transpose == 1 {
                    (w.dims[1], w.dims[0])
                } else {
                    (w.dims[0], w.dims[1])
                };
                if inner != dims[1] as i64 {
                    return None;
                }
                let out = usize::try_from(out).ok()?;
                words = words
                    .checked_add(tensor(w)?)?
                    .checked_add(if bias(&initializers, n, out)? { out } else { 0 })?;
                dims = vec![1, out];
            }
            "BatchNormalization" => {
                if n.input.len() != 5 || int(n, "training_mode", 0)? != 0 {
                    return None;
                }
                for name in ["epsilon", "momentum"] {
                    if let Some(a) = attr(n, name) {
                        if a.r#type != 1 || !a.f_value().is_finite() || a.f_value() < 0.0 {
                            return None;
                        }
                    }
                }
                let c = dims[1];
                bn_channels = bn_channels.checked_add(c)?;
                if bn_channels > 8192 {
                    return None;
                }
                for name in &n.input[1..] {
                    let w = float_weight(&initializers, name)?;
                    if w.dims != [c as i64] {
                        return None;
                    }
                }
                words = words.checked_add(c.checked_mul(c)?)?.checked_add(before)?;
            }
            "Relu" => {
                if n.input.len() != 1 {
                    return None;
                }
                relus += 1;
                words = words.checked_add(before.checked_mul(5)?)?;
            }
            "Flatten" => {
                if n.input.len() != 1 || int(n, "axis", 1)? != 1 {
                    return None;
                }
                dims = vec![1, before];
            }
            "Reshape" => {
                if n.input.len() != 2 || int(n, "allowzero", 0)? != 0 {
                    return None;
                }
                let t = *initializers.get(n.input[1].as_str())?;
                if t.data_type != 7 {
                    return None;
                }
                let raw: Vec<i64> = if t.raw_data.is_empty() {
                    t.int64_data.clone()
                } else {
                    t.raw_data
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|b| i64::from_le_bytes(*b))
                        .collect()
                };
                if !matches!(raw.len(), 2 | 4) {
                    return None;
                }
                let mut next = Vec::with_capacity(raw.len());
                let mut inferred = None;
                for (i, v) in raw.into_iter().enumerate() {
                    next.push(if v == 0 {
                        *dims.get(i)?
                    } else if v == -1 {
                        if inferred.replace(i).is_some() {
                            return None;
                        }
                        1
                    } else {
                        usize::try_from(v).ok().filter(|&v| v > 0)?
                    });
                }
                let known = size(&next)?;
                if let Some(i) = inferred {
                    if before % known != 0 {
                        return None;
                    }
                    next[i] = before / known;
                }
                if next[0] != 1 || size(&next)? != before {
                    return None;
                }
                dims = next;
            }
            _ => return None,
        }
        size(&dims)?;
        if words.checked_mul(8)?.checked_add(128 * 1024)? > MAX_BYTES {
            return None;
        }
        current = &n.output[0];
    }
    if current != g.output[0].name
        || !output_shape_matches(&g.output[0], &dims)
        || size(&dims)? > 8192
        || relus == 0
    {
        return None;
    }
    Some(Quote {
        nodes: g.node.len(),
        extraction_bytes: words.checked_mul(8)?.checked_add(128 * 1024)?,
        bn_channels,
        conv_nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{BatchNormFoldingPolicy as Policy, CustomOpRegistry};
    use crate::onnx_proto::{
        tensor_shape_proto::{dimension::Value, Dimension},
        ModelProto, OperatorSetIdProto, TensorShapeProto, TensorTypeProto, TypeProto,
    };
    fn value(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.into(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&v| Dimension {
                                value: Some(Value::DimValue(v)),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }
    fn weight(name: &str, dims: &[i64], value: f32) -> TensorProto {
        TensorProto {
            name: name.into(),
            dims: dims.to_vec(),
            data_type: 1,
            float_data: vec![value; dims.iter().product::<i64>() as usize],
            ..Default::default()
        }
    }
    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            name: output.into(),
            op_type: op.into(),
            input: inputs.iter().map(|s| (*s).into()).collect(),
            output: vec![output.into()],
            ..Default::default()
        }
    }
    fn fixture() -> GraphProto {
        GraphProto {
            input: vec![value("x", &[1, 2])],
            output: vec![value("y", &[1, 2])],
            initializer: vec![
                weight("w", &[2, 2], 0.5),
                weight("s", &[2], 1.),
                weight("b", &[2], 0.),
                weight("m", &[2], 0.),
                weight("v", &[2], 1.),
            ],
            node: vec![
                node("Gemm", &["x", "w"], "a"),
                node("BatchNormalization", &["a", "s", "b", "m", "v"], "bn"),
                node("Relu", &["bn"], "y"),
            ],
            ..Default::default()
        }
    }
    #[test]
    fn authored_f32_unary_chain_quotes_without_reading_or_copying_weight_values() {
        let mut g = fixture();
        let before = g.initializer[0].float_data.as_ptr();
        let q = quote(&g).unwrap();
        assert_eq!(q.nodes, 3);
        assert_eq!(q.bn_channels, 2);
        assert!(q.extraction_bytes < 256 << 20);
        assert_eq!(before, g.initializer[0].float_data.as_ptr());
        // Value validity is the subsequent normal loader/evaluator's duty;
        // metadata admission alone must not pretend to certify a numeric bound.
        g.initializer[0].float_data[0] = f32::NAN;
        assert!(quote(&g).is_some());
    }
    #[test]
    fn non_float_and_nonunit_gemm_or_training_bn_never_receive_raw_authority() {
        let mut g = fixture();
        g.initializer[0].data_type = 11;
        g.initializer[0].double_data = vec![0.5; 4];
        g.initializer[0].float_data.clear();
        assert!(quote(&g).is_none());
        for (name, kind, f, i) in [
            ("alpha", 1, Some(0.5), None),
            ("beta", 1, Some(2.), None),
            ("transA", 2, None, Some(1)),
        ] {
            let mut g = fixture();
            g.node[0].attribute.push(AttributeProto {
                name: name.into(),
                r#type: kind,
                f,
                i,
                ..Default::default()
            });
            assert!(quote(&g).is_none());
        }
        let mut g = fixture();
        g.node[1].attribute.push(AttributeProto {
            name: "training_mode".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.node[1].attribute.push(AttributeProto {
            name: "epsilon".into(),
            r#type: 1,
            f: Some(-1.),
            ..Default::default()
        });
        assert!(quote(&g).is_none());
    }
    #[test]
    fn unsupported_topology_shape_and_bounded_metadata_refuse() {
        let mut g = fixture();
        g.node[2].op_type = "Sigmoid".into();
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.node[2].input[0] = "a".into();
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.output[0] = value("y", &[1, 3]);
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.input[0] = value("x", &[2, 2]);
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.node[0].name = "a".repeat(1025);
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.initializer.push(g.initializer[0].clone());
        assert!(quote(&g).is_none());
        // Empty strings do not consume names_ok's byte cap; list lengths must
        // refuse before walking them (small adjacent boundary controls).
        let mut g = fixture();
        g.node[0].input = vec![String::new(); 6];
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.node[0].output = vec![String::new(); 2];
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.node[0].attribute = vec![AttributeProto::default(); 9];
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.node[1].name = g.node[0].name.clone();
        assert!(quote(&g).is_none());
        // An unnamed layer uses its output as its key, which can collide with
        // another layer's explicitly authored name despite unique outputs.
        let mut g = fixture();
        g.node[0].name.clear();
        g.node[1].name = g.node[0].output[0].clone();
        assert!(quote(&g).is_none());
        let mut g = fixture();
        g.node[0].name = ny_propagate::NETWORK_INPUT.into();
        assert!(quote(&g).is_none());
        let mut g = fixture();
        for node in &mut g.node {
            node.name.clear();
        }
        assert!(quote(&g).is_some());
    }
    #[test]
    fn diagonal_bn_payload_cap_is_checked_before_any_quadratic_allocation() {
        for (c, allowed) in [(4096usize, true), (8192, false), (8193, false)] {
            let g = GraphProto {
                input: vec![value("x", &[1, c as i64])],
                output: vec![value("y", &[1, c as i64])],
                initializer: [("s", 1.), ("b", 0.), ("m", 0.), ("v", 1.)]
                    .iter()
                    .map(|&(n, v)| weight(n, &[c as i64], v))
                    .collect(),
                node: vec![
                    node("BatchNormalization", &["x", "s", "b", "m", "v"], "bn"),
                    node("Relu", &["bn"], "y"),
                ],
                ..Default::default()
            };
            assert_eq!(quote(&g).is_some(), allowed, "channels={c}");
        }
    }
    #[test]
    fn int64_shape_constants_are_separate_from_authored_float_weights() {
        let mut g = fixture();
        g.initializer.push(TensorProto {
            name: "shape".into(),
            dims: vec![2],
            data_type: 7,
            int64_data: vec![1, 2],
            ..Default::default()
        });
        g.node[2].input[0] = "view".into();
        g.node.insert(2, node("Reshape", &["bn", "shape"], "view"));
        assert!(quote(&g).is_some());
        let t = g.initializer.last_mut().unwrap();
        t.data_type = 1;
        t.int64_data.clear();
        t.float_data = vec![1., 2.];
        assert!(quote(&g).is_none());
    }
    fn parsed(g: GraphProto, policy: Policy) -> crate::loader::RootCutLoadReceipt {
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            graph: Some(g),
            ..Default::default()
        };
        super::super::parse_onnx_model_with_root_policy(
            model,
            &CustomOpRegistry::defaults(),
            Default::default(),
            false,
            policy,
            false,
            false,
            None,
        )
        .unwrap()
        .1
    }
    #[test]
    fn actual_load_receipt_retains_explicit_policy_and_structural_refusal() {
        let mut conv = fixture();
        conv.input[0] = value("x", &[1, 1, 1, 2]);
        conv.output[0] = value("y", &[1, 2, 1, 2]);
        conv.node[0].op_type = "Conv".into();
        conv.initializer[0] = weight("w", &[2, 1, 1, 1], 0.5);
        let raw = parsed(conv, Policy::StructuralCertifiedRoot);
        assert_eq!(raw.batch_norm_folding(), Policy::PreserveRaw);
        assert!(raw.certified_root_source());
        // Pure MLPs ordinarily route Sequential: automatic policy must not
        // change their baseline merely because the extractor can read them.
        let sequential = parsed(fixture(), Policy::StructuralCertifiedRoot);
        assert_eq!(sequential.batch_norm_folding(), Policy::LegacyEnvironment);
        assert!(!sequential.certified_root_source());
        let old = parsed(fixture(), Policy::LegacyEnvironment);
        assert_eq!(old.batch_norm_folding(), Policy::LegacyEnvironment);
        assert!(!old.certified_root_source());
        let explicit = parsed(fixture(), Policy::PreserveRaw);
        assert!(explicit.certified_root_source());
        let mut unsupported = fixture();
        unsupported.node[2].op_type = "Sigmoid".into();
        let refused = parsed(unsupported, Policy::StructuralCertifiedRoot);
        assert_eq!(refused.batch_norm_folding(), Policy::LegacyEnvironment);
        assert!(!refused.certified_root_source());
    }

    // Pinned original CNN7 bf14f136... metadata: names, topology, attributes,
    // shapes, dtypes and inline FLOAT storage sizes. Zero payloads deliberately
    // test admission only, not the original numerical model or its property.
    fn cnn7_authored_metadata() -> GraphProto {
        let mut g = GraphProto {
            input: vec![value("input", &[1, 3, 32, 32])],
            output: vec![value("output", &[1, 10])],
            ..Default::default()
        };
        for (name, dims) in [
            ("layers.16.weight", &[512, 32768][..]),
            ("layers.16.bias", &[512][..]),
            ("layers.17.weight", &[512][..]),
            ("layers.17.bias", &[512][..]),
            ("layers.17.running_mean", &[512][..]),
            ("layers.17.running_var", &[512][..]),
            ("layers.19.weight", &[10, 512][..]),
            ("layers.19.bias", &[10][..]),
            ("onnx::Conv_66", &[64, 3, 3, 3][..]),
            ("onnx::Conv_67", &[64][..]),
            ("onnx::Conv_69", &[64, 64, 3, 3][..]),
            ("onnx::Conv_70", &[64][..]),
            ("onnx::Conv_72", &[128, 64, 3, 3][..]),
            ("onnx::Conv_73", &[128][..]),
            ("onnx::Conv_75", &[128, 128, 3, 3][..]),
            ("onnx::Conv_76", &[128][..]),
            ("onnx::Conv_78", &[128, 128, 3, 3][..]),
            ("onnx::Conv_79", &[128][..]),
        ] {
            let count = dims.iter().product::<i64>() as usize;
            g.initializer.push(TensorProto {
                name: name.into(),
                dims: dims.to_vec(),
                data_type: 1,
                raw_data: vec![0; count * 4],
                ..Default::default()
            });
        }
        let mut n = node(
            "Conv",
            &["input", "onnx::Conv_66", "onnx::Conv_67"],
            "/layers/layers.0/Conv_output_0",
        );
        n.name = "/layers/layers.0/Conv".into();
        n.attribute.push(AttributeProto {
            name: "dilations".into(),
            r#type: 7,
            ints: vec![1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "group".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "kernel_shape".into(),
            r#type: 7,
            ints: vec![3, 3],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "pads".into(),
            r#type: 7,
            ints: vec![1, 1, 1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "strides".into(),
            r#type: 7,
            ints: vec![1, 1],
            ..Default::default()
        });
        g.node.push(n);
        let mut n = node(
            "Relu",
            &["/layers/layers.0/Conv_output_0"],
            "/layers/layers.2/Relu_output_0",
        );
        n.name = "/layers/layers.2/Relu".into();
        g.node.push(n);
        let mut n = node(
            "Conv",
            &[
                "/layers/layers.2/Relu_output_0",
                "onnx::Conv_69",
                "onnx::Conv_70",
            ],
            "/layers/layers.3/Conv_output_0",
        );
        n.name = "/layers/layers.3/Conv".into();
        n.attribute.push(AttributeProto {
            name: "dilations".into(),
            r#type: 7,
            ints: vec![1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "group".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "kernel_shape".into(),
            r#type: 7,
            ints: vec![3, 3],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "pads".into(),
            r#type: 7,
            ints: vec![1, 1, 1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "strides".into(),
            r#type: 7,
            ints: vec![1, 1],
            ..Default::default()
        });
        g.node.push(n);
        let mut n = node(
            "Relu",
            &["/layers/layers.3/Conv_output_0"],
            "/layers/layers.5/Relu_output_0",
        );
        n.name = "/layers/layers.5/Relu".into();
        g.node.push(n);
        let mut n = node(
            "Conv",
            &[
                "/layers/layers.5/Relu_output_0",
                "onnx::Conv_72",
                "onnx::Conv_73",
            ],
            "/layers/layers.6/Conv_output_0",
        );
        n.name = "/layers/layers.6/Conv".into();
        n.attribute.push(AttributeProto {
            name: "dilations".into(),
            r#type: 7,
            ints: vec![1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "group".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "kernel_shape".into(),
            r#type: 7,
            ints: vec![3, 3],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "pads".into(),
            r#type: 7,
            ints: vec![1, 1, 1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "strides".into(),
            r#type: 7,
            ints: vec![2, 2],
            ..Default::default()
        });
        g.node.push(n);
        let mut n = node(
            "Relu",
            &["/layers/layers.6/Conv_output_0"],
            "/layers/layers.8/Relu_output_0",
        );
        n.name = "/layers/layers.8/Relu".into();
        g.node.push(n);
        let mut n = node(
            "Conv",
            &[
                "/layers/layers.8/Relu_output_0",
                "onnx::Conv_75",
                "onnx::Conv_76",
            ],
            "/layers/layers.9/Conv_output_0",
        );
        n.name = "/layers/layers.9/Conv".into();
        n.attribute.push(AttributeProto {
            name: "dilations".into(),
            r#type: 7,
            ints: vec![1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "group".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "kernel_shape".into(),
            r#type: 7,
            ints: vec![3, 3],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "pads".into(),
            r#type: 7,
            ints: vec![1, 1, 1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "strides".into(),
            r#type: 7,
            ints: vec![1, 1],
            ..Default::default()
        });
        g.node.push(n);
        let mut n = node(
            "Relu",
            &["/layers/layers.9/Conv_output_0"],
            "/layers/layers.11/Relu_output_0",
        );
        n.name = "/layers/layers.11/Relu".into();
        g.node.push(n);
        let mut n = node(
            "Conv",
            &[
                "/layers/layers.11/Relu_output_0",
                "onnx::Conv_78",
                "onnx::Conv_79",
            ],
            "/layers/layers.12/Conv_output_0",
        );
        n.name = "/layers/layers.12/Conv".into();
        n.attribute.push(AttributeProto {
            name: "dilations".into(),
            r#type: 7,
            ints: vec![1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "group".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "kernel_shape".into(),
            r#type: 7,
            ints: vec![3, 3],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "pads".into(),
            r#type: 7,
            ints: vec![1, 1, 1, 1],
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "strides".into(),
            r#type: 7,
            ints: vec![1, 1],
            ..Default::default()
        });
        g.node.push(n);
        let mut n = node(
            "Relu",
            &["/layers/layers.12/Conv_output_0"],
            "/layers/layers.14/Relu_output_0",
        );
        n.name = "/layers/layers.14/Relu".into();
        g.node.push(n);
        let mut n = node(
            "Flatten",
            &["/layers/layers.14/Relu_output_0"],
            "/layers/layers.15/Flatten_output_0",
        );
        n.name = "/layers/layers.15/Flatten".into();
        n.attribute.push(AttributeProto {
            name: "axis".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
        g.node.push(n);
        let mut n = node(
            "Gemm",
            &[
                "/layers/layers.15/Flatten_output_0",
                "layers.16.weight",
                "layers.16.bias",
            ],
            "/layers/layers.16/Gemm_output_0",
        );
        n.name = "/layers/layers.16/Gemm".into();
        n.attribute.push(AttributeProto {
            name: "alpha".into(),
            r#type: 1,
            f: Some(f32::from_bits(0x3f800000)),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "beta".into(),
            r#type: 1,
            f: Some(f32::from_bits(0x3f800000)),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "transB".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
        g.node.push(n);
        let mut n = node(
            "BatchNormalization",
            &[
                "/layers/layers.16/Gemm_output_0",
                "layers.17.weight",
                "layers.17.bias",
                "layers.17.running_mean",
                "layers.17.running_var",
            ],
            "/layers/layers.17/BatchNormalization_output_0",
        );
        n.name = "/layers/layers.17/BatchNormalization".into();
        n.attribute.push(AttributeProto {
            name: "epsilon".into(),
            r#type: 1,
            f: Some(f32::from_bits(0x3727c5ac)),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "momentum".into(),
            r#type: 1,
            f: Some(f32::from_bits(0x3f666666)),
            ..Default::default()
        });
        g.node.push(n);
        let mut n = node(
            "Relu",
            &["/layers/layers.17/BatchNormalization_output_0"],
            "/layers/layers.18/Relu_output_0",
        );
        n.name = "/layers/layers.18/Relu".into();
        g.node.push(n);
        let mut n = node(
            "Gemm",
            &[
                "/layers/layers.18/Relu_output_0",
                "layers.19.weight",
                "layers.19.bias",
            ],
            "output",
        );
        n.name = "/layers/layers.19/Gemm".into();
        n.attribute.push(AttributeProto {
            name: "alpha".into(),
            r#type: 1,
            f: Some(f32::from_bits(0x3f800000)),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "beta".into(),
            r#type: 1,
            f: Some(f32::from_bits(0x3f800000)),
            ..Default::default()
        });
        n.attribute.push(AttributeProto {
            name: "transB".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
        g.node.push(n);
        for v in [&mut g.input[0], &mut g.output[0]] {
            v.r#type
                .as_mut()
                .unwrap()
                .tensor_type
                .as_mut()
                .unwrap()
                .shape
                .as_mut()
                .unwrap()
                .dim[0]
                .value = Some(Value::DimParam("batch_size".into()));
        }
        g
    }
    #[test]
    fn cnn7_authored_metadata_accepts_only_derived_single_batch_output() {
        let mut g = cnn7_authored_metadata();
        assert!(quote(&g).is_none()); // Input batch has not yet been resolved.
        assert!(crate::loader::batch_resolve::resolve_batch_dim(&mut g, 1));
        let before = g.initializer[0].raw_data.as_ptr();
        let q = quote(&g).expect("bounded authored CNN7 metadata must pass");
        assert_eq!((q.nodes, q.conv_nodes, q.bn_channels), (15, 5, 512));
        assert_eq!(q.extraction_bytes, 150_808_144);
        assert_eq!(before, g.initializer[0].raw_data.as_ptr());
        assert!(
            matches!(g.output[0].r#type.as_ref().unwrap().tensor_type.as_ref().unwrap()
            .shape.as_ref().unwrap().dim[0].value.as_ref(),
            Some(Value::DimParam(label)) if label == "batch_size")
        );
        // A symbol is allowed only on the leading annotation; every concrete
        // dimension still has to match the independently walked output.
        for output in [
            value("output", &[2, 10]),
            value("output", &[1, 11]),
            value("output", &[1, 1, 1, 10]),
        ] {
            g.output[0] = output;
            assert!(quote(&g).is_none());
        }
        for (axis, label) in [
            (0, String::new()),
            (0, "b".repeat(1025)),
            (1, "classes".to_string()),
        ] {
            g.output[0] = value("output", &[1, 10]);
            g.output[0]
                .r#type
                .as_mut()
                .unwrap()
                .tensor_type
                .as_mut()
                .unwrap()
                .shape
                .as_mut()
                .unwrap()
                .dim[axis]
                .value = Some(Value::DimParam(label));
            assert!(quote(&g).is_none());
        }
        for invalid in [None, Some(Value::DimValue(0)), Some(Value::DimValue(-1))] {
            g.output[0] = value("output", &[1, 10]);
            g.output[0]
                .r#type
                .as_mut()
                .unwrap()
                .tensor_type
                .as_mut()
                .unwrap()
                .shape
                .as_mut()
                .unwrap()
                .dim[0]
                .value = invalid;
            assert!(quote(&g).is_none());
        }
    }
    #[test]
    fn same_load_symbolic_output_batch_retains_strict_source_and_policy_guards() {
        let mut g = fixture();
        g.input[0] = value("x", &[1, 1, 1, 2]);
        g.output[0] = value("y", &[1, 2, 1, 2]);
        g.node[0].op_type = "Conv".into();
        g.initializer[0] = weight("w", &[2, 1, 1, 1], 0.5);
        for v in [&mut g.input[0], &mut g.output[0]] {
            v.r#type
                .as_mut()
                .unwrap()
                .tensor_type
                .as_mut()
                .unwrap()
                .shape
                .as_mut()
                .unwrap()
                .dim[0]
                .value = Some(Value::DimParam("batch_size".into()));
        }
        let old = parsed(g.clone(), Policy::LegacyEnvironment);
        assert_eq!(old.batch_norm_folding(), Policy::LegacyEnvironment);
        assert!(!old.certified_root_source());
        let raw = parsed(g.clone(), Policy::StructuralCertifiedRoot);
        assert_eq!(raw.batch_norm_folding(), Policy::PreserveRaw);
        assert!(raw.certified_root_source());
        let mut unsupported = g;
        unsupported.node[2].op_type = "Sigmoid".into();
        let refused = parsed(unsupported, Policy::StructuralCertifiedRoot);
        assert_eq!(refused.batch_norm_folding(), Policy::LegacyEnvironment);
        assert!(!refused.certified_root_source());
    }
}
