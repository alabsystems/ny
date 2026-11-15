// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original-source Float32 Gemm/Relu chains with shared-scalar head bounds.
//!
//! A model is admitted only after a bounded wire-field audit, complete chain
//! extraction, and comparison to the CPU runtime's exported optimized graph.
//! Only numerically identical Gemm stages and optional Gemm/Relu fusion are
//! accepted. The enclosure covers exact real evaluation and binary32 GEMM
//! evaluation with arbitrary reduction order and fused multiply-adds.
//! Neither loader-rewritten weights nor sampled outputs grant authority.
use super::*;
use crate::onnx_proto::{ModelProto, TensorProto, ValueInfoProto};
use prost::Message;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

const MAX_BYTES: usize = 32 * 1024 * 1024;
const MAX_WIDTH: usize = 512;
const MAX_PARAMETERS: usize = 1_000_000;

struct Poll<'a> {
    end: Instant,
    callback: &'a mut dyn FnMut() -> bool,
}
impl Poll<'_> {
    fn check(&mut self) -> Option<()> {
        (Instant::now() < self.end && (self.callback)()).then_some(())
    }
}

#[derive(Clone, Debug)]
struct Gemm {
    inputs: usize,
    outputs: usize,
    // Canonical output-major original Float32 parameters, widened exactly.
    weights: Vec<f64>,
    bias: Vec<f64>,
}
impl PartialEq for Gemm {
    fn eq(&self, rhs: &Self) -> bool {
        self.inputs == rhs.inputs
            && self.outputs == rhs.outputs
            && self.weights.len() == rhs.weights.len()
            && self.bias.len() == rhs.bias.len()
            && self
                .weights
                .iter()
                .zip(&rhs.weights)
                .chain(self.bias.iter().zip(&rhs.bias))
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}
#[derive(Clone, PartialEq, Debug)]
enum Stage {
    Gemm(Gemm),
    Relu,
}
#[derive(PartialEq, Debug)]
struct Chain {
    input_name: String,
    output_name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    stages: Vec<Stage>,
}

/// A source-bound CPU execution profile. Public constructors also check that
/// the runtime's optimized graph has exactly the same numerical stages.
pub struct PreparedFloat32Chain {
    chain: Chain,
    source_sha256: [u8; 32],
    optimized_sha256: [u8; 32],
}

/// Both real and Float32 network values are enclosed. The Float32 predicate
/// evaluator must still intersect its input domain with the binary32 grid.
pub struct Float32ChainEnclosure {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub shared_scalar_heads: usize,
}

// Audit fields BEFORE prost can discard an unsupported semantic field.
// Repeated packed scalar encodings are allowed; duplicate singular fields,
// groups, unknown operator attributes, external data and FunctionProto refuse.
#[derive(Clone, Copy)]
enum Schema {
    Model,
    Graph,
    Node,
    Tensor,
    Attribute,
    Value,
    Type,
    TensorType,
    Shape,
    Dim,
    Opset,
    Metadata,
}
fn rule(s: Schema, t: u32) -> Option<(u8, bool, Option<Schema>)> {
    use Schema::{
        Attribute, Dim, Graph, Metadata, Model, Node, Opset, Shape, Tensor, TensorType, Type, Value,
    };
    let r = match (s, t) {
        (Model, 1 | 5) => (1, false, None),
        (Model, 2 | 3 | 4 | 6) => (4, false, None),
        (Model, 7) => (4, false, Some(Graph)),
        (Model, 8) => (4, true, Some(Opset)),
        (Model, 14) => (4, true, Some(Metadata)),
        (Graph, 1) => (4, true, Some(Node)),
        // GraphProto.doc_string is nonsemantic and is emitted even when empty
        // by the CPU runtime's optimized-model serializer.
        (Graph, 2 | 10) => (4, false, None),
        (Graph, 5) => (4, true, Some(Tensor)),
        (Graph, 11..=13) => (4, true, Some(Value)),
        (Graph, 16) => (4, true, Some(Metadata)),
        (Node, 1 | 2) => (4, true, None),
        (Node, 3 | 4 | 6 | 7) => (4, false, None),
        (Node, 5) => (4, true, Some(Attribute)),
        (Node, 9) => (4, true, Some(Metadata)),
        (Tensor, 1) => (1 | 4, true, None),
        (Tensor, 2) => (1, false, None),
        (Tensor, 4) => (4 | 32, true, None),
        (Tensor, 8 | 9 | 12) => (4, false, None),
        (Tensor, 16) => (4, true, Some(Metadata)),
        (Attribute, 1 | 4 | 13) => (4, false, None),
        (Attribute, 2) => (32, false, None),
        (Attribute, 3 | 20) => (1, false, None),
        (Value, 1 | 3) => (4, false, None),
        (Value, 2) => (4, false, Some(Type)),
        (Value, 4) => (4, true, Some(Metadata)),
        (Type, 1) => (4, false, Some(TensorType)),
        (TensorType, 1) => (1, false, None),
        (TensorType, 2) => (4, false, Some(Shape)),
        (Shape, 1) => (4, true, Some(Dim)),
        (Dim, 1) => (1, false, None),
        (Dim, 2 | 3) => (4, false, None),
        (Opset, 1) => (4, false, None),
        (Opset, 2) => (1, false, None),
        (Metadata, 1 | 2) => (4, false, None),
        _ => return None,
    };
    Some(r)
}
fn varint(b: &[u8], p: &mut usize) -> Option<u64> {
    let mut v = 0;
    for shift in (0..70).step_by(7) {
        let a = *b.get(*p)?;
        *p += 1;
        if shift == 63 && a > 1 {
            return None;
        }
        v |= u64::from(a & 127) << shift;
        if a < 128 {
            return Some(v);
        }
    }
    None
}
fn audit(b: &[u8], schema: Schema, poll: &mut Poll<'_>, fields: &mut usize) -> Option<()> {
    let mut p = 0;
    let mut seen = HashSet::new();
    while p < b.len() {
        poll.check()?;
        *fields = fields.checked_add(1)?;
        if *fields > 100_000 {
            return None;
        }
        let key = varint(b, &mut p)?;
        let tag = u32::try_from(key >> 3).ok()?;
        let wire = key & 7;
        let (mask, repeated, child) = rule(schema, tag)?;
        if mask & (1 << wire) == 0 || (!repeated && !seen.insert(tag)) {
            return None;
        }
        match wire {
            0 => {
                varint(b, &mut p)?;
            }
            1 | 5 => {
                p = p.checked_add(if wire == 1 { 8 } else { 4 })?;
            }
            2 => {
                let n = usize::try_from(varint(b, &mut p)?).ok()?;
                let end = p.checked_add(n)?;
                let data = b.get(p..end)?;
                if let Some(s) = child {
                    audit(data, s, poll, fields)?;
                }
                p = end;
            }
            _ => return None,
        }
        if p > b.len() {
            return None;
        }
    }
    Some(())
}
fn shape(v: &ValueInfoProto) -> Option<Vec<usize>> {
    let t = v.r#type.as_ref()?.tensor_type.as_ref()?;
    if t.elem_type != 1 || v.name.is_empty() || v.name.len() > 1024 {
        return None;
    }
    let dims = t
        .shape
        .as_ref()?
        .dim
        .iter()
        .map(|d| {
            use crate::onnx_proto::tensor_shape_proto::dimension::Value;
            match d.value.as_ref()? {
                Value::DimValue(n) => usize::try_from(*n).ok().filter(|&n| n > 0),
                _ => None,
            }
        })
        .collect::<Option<Vec<_>>>()?;
    (dims.len() == 2 && dims[0] == 1 && dims[1] <= MAX_WIDTH).then_some(dims)
}
fn tensor(t: &TensorProto, poll: &mut Poll<'_>) -> Option<Vec<f64>> {
    if t.data_type != 1
        || t.dims.is_empty()
        || t.dims.len() > 2
        || t.name.is_empty()
        || t.name.len() > 1024
    {
        return None;
    }
    let n = t.dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(
            usize::try_from(d)
                .ok()
                .filter(|&d| d > 0 && d <= MAX_WIDTH)?,
        )
    })?;
    let values = if !t.raw_data.is_empty() {
        if !t.float_data.is_empty() || t.raw_data.len() != n.checked_mul(4)? {
            return None;
        }
        t.raw_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|s| f32::from_le_bytes(*s))
            .collect::<Vec<_>>()
    } else {
        if t.float_data.len() != n {
            return None;
        }
        t.float_data.clone()
    };
    let mut result = Vec::with_capacity(n);
    for v in values {
        poll.check()?;
        // DAZ for activations is enclosed separately. A subnormal authored
        // coefficient is outside this deliberately small CPU profile.
        if !v.is_finite() || v.is_subnormal() {
            return None;
        }
        result.push(f64::from(v));
    }
    Some(result)
}
fn extract(raw: &[u8], fused: bool, poll: &mut Poll<'_>) -> Option<Chain> {
    poll.check()?;
    if raw.is_empty() || raw.len() > MAX_BYTES {
        return None;
    }
    audit(raw, Schema::Model, poll, &mut 0)?;
    let m = ModelProto::decode(raw).ok()?;
    if !(3..=12).contains(&m.ir_version) {
        return None;
    }
    let mut domains = HashMap::new();
    for o in &m.opset_import {
        poll.check()?;
        if o.domain.len() > 1024
            || o.version <= 0
            || domains.insert(o.domain.as_str(), o.version).is_some()
        {
            return None;
        }
    }
    let version = domains.get("").or_else(|| domains.get("ai.onnx"))?;
    if !(7..=21).contains(version) || domains.get("ai.onnx").is_some_and(|v| v != version) {
        return None;
    }
    let g = m.graph?;
    if g.input.len() != 1
        || g.output.len() != 1
        || g.node.is_empty()
        || g.node.len() > 64
        || g.initializer.len() > 128
    {
        return None;
    }
    let input_shape = shape(&g.input[0])?;
    let output_shape = shape(&g.output[0])?;
    let mut params = HashMap::new();
    let mut count = 0usize;
    let mut vertices = HashSet::new();
    vertices.insert(g.input[0].name.as_str());
    for t in &g.initializer {
        poll.check()?;
        if !vertices.insert(&t.name) {
            return None;
        }
        let values = tensor(t, poll)?;
        count = count.checked_add(values.len())?;
        if count > MAX_PARAMETERS {
            return None;
        }
        params.insert(t.name.as_str(), (t.dims.as_slice(), values));
    }
    let mut used = HashSet::new();
    let mut current = g.input[0].name.as_str();
    let mut width = input_shape[1];
    let mut stages = Vec::new();
    for n in &g.node {
        poll.check()?;
        if n.output.len() != 1
            || n.output[0].is_empty()
            || n.output[0].len() > 1024
            || !vertices.insert(&n.output[0])
            || n.input.first().map(String::as_str) != Some(current)
        {
            return None;
        }
        let standard = n.domain.is_empty() || n.domain == "ai.onnx";
        match n.op_type.as_str() {
            "Relu" if standard => {
                if n.input.len() != 1 || !n.attribute.is_empty() {
                    return None;
                }
                stages.push(Stage::Relu);
            }
            "Gemm" | "FusedGemm" => {
                let is_fused = n.op_type == "FusedGemm";
                if n.input.len() != 3
                    || (!is_fused && !standard)
                    || (is_fused
                        && (!fused
                            || n.domain != "com.microsoft"
                            || domains.get("com.microsoft") != Some(&1)))
                {
                    return None;
                }
                let mut transpose = false;
                let mut activation = false;
                let mut names = HashSet::new();
                for a in &n.attribute {
                    poll.check()?;
                    if !names.insert(a.name.as_str()) || !a.ref_attr_name.is_empty() {
                        return None;
                    }
                    match a.name.as_str() {
                        "alpha" | "beta"
                            if a.r#type == 1
                                && a.i.is_none()
                                && a.s.is_none()
                                && a.f_value() == 1.0 => {}
                        "transA"
                            if a.r#type == 2
                                && a.f.is_none()
                                && a.s.is_none()
                                && a.i_value() == 0 => {}
                        "transB"
                            if a.r#type == 2
                                && a.f.is_none()
                                && a.s.is_none()
                                && matches!(a.i_value(), 0 | 1) =>
                        {
                            transpose = a.i_value() == 1
                        }
                        "activation"
                            if is_fused
                                && a.r#type == 3
                                && a.f.is_none()
                                && a.i.is_none()
                                && a.s_value() == b"Relu" =>
                        {
                            activation = true
                        }
                        _ => return None,
                    }
                }
                if is_fused && !activation {
                    return None;
                }
                let (dims, w) = params.get(n.input[1].as_str())?;
                let (bdims, bias) = params.get(n.input[2].as_str())?;
                if dims.len() != 2 {
                    return None;
                }
                let (outputs, inputs) = if transpose {
                    (dims[0] as usize, dims[1] as usize)
                } else {
                    (dims[1] as usize, dims[0] as usize)
                };
                if inputs != width
                    || !(bdims == &[outputs as i64] || bdims == &[1, outputs as i64])
                    || bias.len() != outputs
                {
                    return None;
                }
                let mut weights = Vec::with_capacity(w.len());
                for j in 0..outputs {
                    for i in 0..inputs {
                        poll.check()?;
                        weights.push(if transpose {
                            w[j * inputs + i]
                        } else {
                            w[i * outputs + j]
                        });
                    }
                }
                stages.push(Stage::Gemm(Gemm {
                    inputs,
                    outputs,
                    weights,
                    bias: bias.clone(),
                }));
                if activation {
                    stages.push(Stage::Relu);
                }
                used.insert(n.input[1].as_str());
                used.insert(n.input[2].as_str());
                width = outputs;
            }
            _ => return None,
        }
        current = &n.output[0];
    }
    if current != g.output[0].name || width != output_shape[1] || used.len() != params.len() {
        return None;
    }
    poll.check()?;
    Some(Chain {
        input_name: g.input[0].name.clone(),
        output_name: g.output[0].name.clone(),
        input_shape,
        output_shape,
        stages,
    })
}

fn up_add(a: f64, b: f64) -> f64 {
    if a == 0.0 && b == 0.0 {
        0.0
    } else {
        (a + b).next_up()
    }
}
fn up_mul(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        (a * b).next_up()
    }
}
fn expand(a: OpenIv, e: f64) -> Option<OpenIv> {
    if !e.is_finite() || e < 0.0 {
        return None;
    }
    OpenIv::closed((a.lo - e).next_down(), (a.hi + e).next_up()).checked()
}
fn dot(x: &[OpenIv], w: &[f64], b: f64, poll: &mut Poll<'_>) -> Option<OpenIv> {
    if x.len() != w.len() {
        return None;
    }
    let mut value = OpenIv::closed(b, b);
    for (&a, &weight) in x.iter().zip(w) {
        poll.check()?;
        if weight != 0.0 {
            value = value.add(a.mul(OpenIv::closed(weight, weight))?)?;
        }
    }
    Some(value)
}
/// If q binary32 operations each obey |fl(z)-z| <= u|z|+eta,
/// their absolute error is <= gamma_q*S + q*eta/(1-q*u).
/// S sums the absolute original terms, so any binary reduction tree and
/// FMA have no longer a rounding path. DAZ adds eta*sum|w| at the inputs;
/// eta=4*min_normal covers gradual underflow and input/result flushing at
/// each operation, including both inputs of a partial-sum addition.
/// All parameters are normal/zero, alpha=beta=1, and the absolute-sum guard
/// also excludes overflow in any supported evaluation order.
fn rounding_error(x: &[OpenIv], w: &[f64], b: f64, poll: &mut Poll<'_>) -> Option<f64> {
    if x.len() != w.len() {
        return None;
    }
    let mut magnitude = b.abs();
    let mut sum_weights = 0.0;
    for (&a, &weight) in x.iter().zip(w) {
        poll.check()?;
        magnitude = up_add(magnitude, up_mul(weight.abs(), a.lo.abs().max(a.hi.abs())));
        sum_weights = up_add(sum_weights, weight.abs());
    }
    let q = (2 * w.len() + 4) as f64;
    let u = f64::from(f32::EPSILON) * 0.5;
    let eta = 4.0 * f64::from(f32::MIN_POSITIVE);
    let denominator = (1.0 - q * u).next_down();
    let gamma = (q * u / denominator).next_up();
    let daz = up_mul(eta, up_add(1.0, sum_weights));
    let error = up_add(
        daz,
        up_add(
            up_mul(gamma, up_add(magnitude, daz)),
            (q * eta / denominator).next_up(),
        ),
    );
    if !error.is_finite() || up_add(magnitude, error) > f64::from(f32::MAX) * 0.25 {
        return None;
    }
    Some(error)
}
fn affine(g: &Gemm, x: &[OpenIv], poll: &mut Poll<'_>) -> Option<Vec<OpenIv>> {
    if x.len() != g.inputs {
        return None;
    }
    (0..g.outputs)
        .map(|j| {
            let w = &g.weights[j * g.inputs..(j + 1) * g.inputs];
            expand(
                dot(x, w, g.bias[j], poll)?,
                rounding_error(x, w, g.bias[j], poll)?,
            )
        })
        .collect()
}
fn relu(x: OpenIv) -> OpenIv {
    let lo = x.lo.max(0.0);
    // A direct graph input or a final activation has no following Gemm in
    // which to charge this flush. Keep zero whenever a positive subnormal
    // activation can be flushed, while also retaining gradual evaluation.
    OpenIv::closed(
        if lo < f64::from(f32::MIN_POSITIVE) {
            0.0
        } else {
            lo
        },
        x.hi.max(0.0),
    )
}

/// For scalar t, the ideal affine-ReLU-affine map is piecewise affine.
/// Its extrema are at the two ends or a first-layer zero. Each zero is
/// enclosed by a directed division; no approximate root is substituted.
/// Errors of the first Gemm cross ReLU through its Lipschitz bound; then
/// the last Gemm's own rounding error is charged using the expanded ranges.
fn shared_head(a: &Gemm, b: &Gemm, t: OpenIv, poll: &mut Poll<'_>) -> Option<Vec<OpenIv>> {
    if a.inputs != 1 || a.outputs > 8 || b.inputs != a.outputs || b.outputs > 16 {
        return None;
    }
    let mut sites = vec![OpenIv::closed(t.lo, t.lo), OpenIv::closed(t.hi, t.hi)];
    let mut errors = Vec::new();
    let mut ranges = Vec::new();
    for i in 0..a.outputs {
        poll.check()?;
        let w = a.weights[i];
        let c = a.bias[i];
        let e = up_add(
            rounding_error(&[t], &[w], c, poll)?,
            f64::from(f32::MIN_POSITIVE),
        );
        errors.push(e);
        ranges.push(relu(expand(dot(&[t], &[w], c, poll)?, e)?));
        if w != 0.0 {
            let r = -c / w;
            if !r.is_finite() {
                return None;
            }
            let lo = r.next_down().max(t.lo);
            let hi = r.next_up().min(t.hi);
            if lo <= hi {
                sites.push(OpenIv::closed(lo, hi));
            }
        }
    }
    let mut result = Vec::new();
    for j in 0..b.outputs {
        poll.check()?;
        let w = &b.weights[j * b.inputs..(j + 1) * b.inputs];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for site in &sites {
            poll.check()?;
            let z = (0..a.outputs)
                .map(|i| dot(&[*site], &[a.weights[i]], a.bias[i], poll).map(relu))
                .collect::<Option<Vec<_>>>()?;
            let v = dot(&z, w, b.bias[j], poll)?;
            lo = lo.min(v.lo);
            hi = hi.max(v.hi);
        }
        let mut error = rounding_error(&ranges, w, b.bias[j], poll)?;
        for (&w, &e) in w.iter().zip(&errors) {
            poll.check()?;
            error = up_add(error, up_mul(w.abs(), e));
        }
        result.push(expand(OpenIv::closed(lo, hi), error)?);
    }
    Some(result)
}

impl PreparedFloat32Chain {
    /// Prepare against the original complete model and original formula. The
    /// CPU runtime export is checked, never trusted as a replacement model.
    #[cfg(feature = "ort")]
    pub fn prepare(
        raw: &[u8],
        formula: &NonLinearFormula,
        end: Instant,
        callback: &mut dyn FnMut() -> bool,
    ) -> Option<Self> {
        if cfg!(all(target_os = "macos", target_arch = "x86_64"))
            || !cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
        {
            return None;
        }
        let mut poll = Poll { end, callback };
        poll.check()?;
        formula.prepare_f32_predicate(end, poll.callback)?;
        let chain = extract(raw, false, &mut poll)?;
        if formula.decls.iter().any(|d| {
            d.shape
                != match d.kind {
                    VarKind::Input => chain.input_shape.clone(),
                    VarKind::Output => chain.output_shape.clone(),
                }
        }) {
            return None;
        }
        let temp = tempfile::tempdir().ok()?;
        let path = temp.path().join("optimized.onnx");
        // Explicit CPU provider, standard accuracy, original source bytes.
        let _session = crate::runtime::session_builder()
            .ok()?
            .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
            .ok()?
            .with_intra_threads(1)
            .ok()?
            .with_inter_threads(1)
            .ok()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)
            .ok()?
            .with_optimized_model_path(&path)
            .ok()?
            .commit_from_memory(raw)
            .ok()?;
        poll.check()?;
        if std::fs::metadata(&path).ok()?.len() > MAX_BYTES as u64 {
            return None;
        }
        let optimized = std::fs::read(&path).ok()?;
        let replay = extract(&optimized, true, &mut poll)?;
        if replay != chain {
            return None;
        }
        let result = Self {
            chain,
            source_sha256: Sha256::digest(raw).into(),
            optimized_sha256: Sha256::digest(&optimized).into(),
        };
        poll.check()?;
        Some(result)
    }

    #[must_use]
    pub fn source_sha256(&self) -> [u8; 32] {
        self.source_sha256
    }
    #[must_use]
    pub fn optimized_sha256(&self) -> [u8; 32] {
        self.optimized_sha256
    }

    /// Enclose both the real input cell and every Float32 cast of that cell.
    /// The owner must cover the full domain, retain the original formula,
    /// and check its absolute deadline again before any verdict publication.
    pub fn enclose(
        &self,
        lo: &[f64],
        hi: &[f64],
        end: Instant,
        callback: &mut dyn FnMut() -> bool,
    ) -> Option<Float32ChainEnclosure> {
        let mut poll = Poll { end, callback };
        poll.check()?;
        if !f32_predicate::environment()
            || lo.len() != self.chain.input_shape[1]
            || hi.len() != lo.len()
        {
            return None;
        }
        let mut values = lo
            .iter()
            .zip(hi)
            .map(|(&l, &u)| {
                let (fl, fu) = (f64::from(l as f32), f64::from(u as f32));
                OpenIv::closed(l.min(fl), u.max(fu)).checked().filter(|_| {
                    l.is_finite() && u.is_finite() && l <= u && fl.is_finite() && fu.is_finite()
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let mut i = 0;
        let mut heads = 0;
        while i < self.chain.stages.len() {
            poll.check()?;
            if let Some([Stage::Gemm(a), Stage::Relu, Stage::Gemm(b)]) =
                self.chain.stages.get(i..i + 3)
            {
                if values.len() == 1 && a.inputs == 1 && a.outputs <= 8 && b.outputs <= 16 {
                    values = shared_head(a, b, values[0], &mut poll)?;
                    i += 3;
                    heads += 1;
                    continue;
                }
            }
            values = match &self.chain.stages[i] {
                Stage::Gemm(g) => affine(g, &values, &mut poll)?,
                Stage::Relu => values.into_iter().map(relu).collect(),
            };
            i += 1;
        }
        let result = Float32ChainEnclosure {
            lower: values.iter().map(|v| v.lo).collect(),
            upper: values.iter().map(|v| v.hi).collect(),
            shared_scalar_heads: heads,
        };
        poll.check()?;
        Some(result)
    }
}

#[path = "f32_chain/refute.rs"]
mod refute;
pub use refute::{Float32ChainAttempt, Float32ChainRefutation};

#[cfg(test)]
#[path = "f32_chain_tests.rs"]
mod tests;
