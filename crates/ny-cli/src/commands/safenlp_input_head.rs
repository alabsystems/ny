// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Narrow original-source adapter. The singleton stays a singleton and never
//! enters TwinSpec extraction or a synthetic disjunctive graph.

use super::vnncomp::VnncompResult;
use ny_onnx::onnx_proto::{ModelProto, TensorProto, ValueInfoProto};
use ny_onnx::vnnlib::{CertifiedRelationalOutputAtom, TensorDeclarationKind};
use ny_propagate::margin_row::{InputHeadOutcome, InputHeadSpec, MarginRowRuntimePolicy};
use prost::Message;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const MAX_MODEL_BYTES: u64 = 1024 * 1024;
const MAX_PROPERTY_BYTES: u64 = 256 * 1024;

// Preserve semantic fields omitted by the shared minimal ONNX decoder. Bytes
// fields use the same length-delimited wire encoding as the original messages.
#[derive(prost::Message)]
struct ModelEnvelope {
    #[prost(bytes = "vec", repeated, tag = "7")]
    graphs: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "25")]
    functions: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "20")]
    training: Vec<Vec<u8>>,
}
#[derive(prost::Message)]
struct GraphEnvelope {
    #[prost(message, repeated, tag = "1")]
    nodes: Vec<NodeEnvelope>,
    #[prost(message, repeated, tag = "11")]
    inputs: Vec<ValueEnvelope>,
    #[prost(message, repeated, tag = "12")]
    outputs: Vec<ValueEnvelope>,
    #[prost(bytes = "vec", repeated, tag = "13")]
    values: Vec<Vec<u8>>,
}
#[derive(prost::Message)]
struct NodeEnvelope {
    #[prost(string, repeated, tag = "8")]
    overload: Vec<String>,
}
#[derive(prost::Message)]
struct ValueEnvelope {
    #[prost(message, optional, tag = "2")]
    kind: Option<TypeEnvelope>,
}
#[derive(prost::Message)]
struct TypeEnvelope {
    #[prost(bytes = "vec", repeated, tag = "1")]
    tensor: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "4")]
    sequence: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "5")]
    map: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "7")]
    opaque: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "8")]
    sparse: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    optional: Vec<Vec<u8>>,
}
fn dense_type(value: &ValueEnvelope) -> bool {
    value.kind.as_ref().is_some_and(|t| {
        t.tensor.len() == 1
            && t.sequence.is_empty()
            && t.map.is_empty()
            && t.opaque.is_empty()
            && t.sparse.is_empty()
            && t.optional.is_empty()
    })
}

fn read_capped(path: &Path, cap: u64) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(cap.checked_add(1)?)
        .read_to_end(&mut bytes)
        .ok()?;
    (bytes.len() as u64 <= cap).then_some(bytes)
}

/// This private layout obligation is minted only by the original typed tensor
/// property or complete scalar VNN-LIB1 namespace. It becomes graph authority
/// only after the entire constant-right five-node chain is authenticated:
/// one rank2 input [B,D] flattens to D original scalars, hence B=1; each actual
/// MatMul/Add/ReLU preserves B and the one [B,2] output contains two scalars.
/// A symbolic axis never supplies its own batch binding or overrides rank.
struct OriginalSingletonLayout {
    inputs: usize,
    outputs: usize,
}

fn dense_dimensions(
    value: &ValueInfoProto,
) -> Option<&[ny_onnx::onnx_proto::tensor_shape_proto::Dimension]> {
    let tensor = value.r#type.as_ref()?.tensor_type.as_ref()?;
    if tensor.elem_type != 1 {
        return None;
    }
    Some(tensor.shape.as_ref()?.dim.as_slice())
}

/// A width hint permits the caller to authenticate the original property
/// BEFORE model specialization. It is not a graph/source certificate.
fn input_feature_width(value: &ValueInfoProto) -> Option<usize> {
    use ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value;
    let [batch, features] = dense_dimensions(value)? else {
        return None;
    };
    match batch.value.as_ref()? {
        Value::DimValue(1) => {}
        Value::DimParam(name) if !name.is_empty() => {}
        _ => return None,
    }
    let Value::DimValue(features) = features.value.as_ref()? else {
        return None;
    };
    let width = usize::try_from(*features).ok()?;
    (1..=128).contains(&width).then_some(width)
}

fn input_width_hint(bytes: &[u8]) -> Option<usize> {
    if bytes.len() > MAX_MODEL_BYTES as usize {
        return None;
    }
    let model = ModelProto::decode(bytes).ok()?;
    let g = model.graph.as_ref()?;
    let [input] = g.input.as_slice() else {
        return None;
    };
    input_feature_width(input)
}

fn singleton_shape_matches(
    value: &ValueInfoProto,
    features: usize,
    layout: Option<&OriginalSingletonLayout>,
) -> bool {
    use ny_onnx::onnx_proto::tensor_shape_proto::dimension::Value;
    let Some([batch, width]) = dense_dimensions(value) else {
        return false;
    };
    if !matches!(width.value.as_ref(), Some(Value::DimValue(n)) if usize::try_from(*n).ok() == Some(features))
    {
        return false;
    }
    match batch.value.as_ref() {
        Some(Value::DimValue(1)) => true,
        Some(Value::DimParam(name)) => !name.is_empty() && layout.is_some(),
        _ => false,
    }
}

/// Additional imports are inert only when unused. The exact chain below
/// requires EVERY actual node's domain to be empty, and the original wire
/// refuses local functions and overloads. Ambiguous maps are never resolved.
fn standard_opset_is_owned(model: &ModelProto) -> bool {
    if model.opset_import.is_empty() || model.opset_import.len() > 16 {
        return false;
    }
    let mut domains = HashSet::new();
    let mut standard = None;
    for import in &model.opset_import {
        // "ai.onnx" aliases the default standard domain. It is not an inert
        // additional domain; retain the existing empty-domain spelling only.
        if import.version <= 0
            || import.domain == "ai.onnx"
            || !domains.insert(import.domain.as_str())
        {
            return false;
        }
        if import.domain.is_empty() {
            standard = Some(import.version);
        }
    }
    standard.is_some_and(|version| (7..=21).contains(&version))
}

fn parameters(t: &TensorProto, dims: &[i64]) -> Option<Vec<f32>> {
    if t.data_type != 1
        || t.data_location != 0
        || t.segment.is_some()
        || !t.external_data.is_empty()
        || t.dims != dims
        || !t.double_data.is_empty()
        || !t.int32_data.is_empty()
        || !t.int64_data.is_empty()
        || !t.string_data.is_empty()
        || !t.uint64_data.is_empty()
    {
        return None;
    }
    let count = dims
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))?;
    let values = if t.raw_data.is_empty() {
        t.float_data.clone()
    } else {
        if !t.float_data.is_empty() || t.raw_data.len() != count.checked_mul(4)? {
            return None;
        }
        t.raw_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect()
    };
    (values.len() == count && values.iter().all(|x| x.is_finite())).then_some(values)
}

/// Exactly five ordinary ONNX nodes, no dynamic weights, broadcast tricks,
/// folding, transposition, extra branch or custom domain. All source storage
/// is FLOAT32; the downstream producer also charges FLOAT32 execution error.
#[cfg(test)]
fn extract(bytes: &[u8]) -> Option<InputHeadSpec> {
    extract_bound(bytes, None)
}

fn extract_bound(bytes: &[u8], layout: Option<&OriginalSingletonLayout>) -> Option<InputHeadSpec> {
    ny_core::require_f64_interval_proof_environment().ok()?;
    if bytes.len() > MAX_MODEL_BYTES as usize {
        return None;
    }
    let model = ModelProto::decode(bytes).ok()?;
    let raw = ModelEnvelope::decode(bytes).ok()?;
    if !(4..=10).contains(&model.ir_version)
        || raw.graphs.len() != 1
        || !raw.functions.is_empty()
        || !raw.training.is_empty()
    {
        return None;
    }
    let envelope = GraphEnvelope::decode(raw.graphs[0].as_slice()).ok()?;
    if envelope.nodes.iter().any(|n| !n.overload.is_empty())
        || envelope
            .inputs
            .iter()
            .chain(&envelope.outputs)
            .any(|v| !dense_type(v))
    {
        return None;
    }
    if !standard_opset_is_owned(&model) {
        return None;
    }
    let g = model.graph.as_ref()?;
    if g.name.is_empty()
        || g.input.len() != 1
        || g.output.len() != 1
        || g.node.len() != 5
        || g.initializer.len() != 4
        || !g.sparse_initializer.is_empty()
    {
        return None;
    }
    if g.input[0].name.is_empty()
        || g.output[0].name.is_empty()
        || g.initializer.iter().any(|t| t.name.is_empty())
    {
        return None;
    }
    let inputs = input_feature_width(&g.input[0])?;
    if layout.is_some_and(|binding| binding.inputs != inputs || binding.outputs != 2)
        || !singleton_shape_matches(&g.input[0], inputs, layout)
        || !singleton_shape_matches(&g.output[0], 2, layout)
    {
        return None;
    }
    let nodes = &g.node;
    for (n, kind) in nodes.iter().zip(["MatMul", "Add", "Relu", "MatMul", "Add"]) {
        if n.op_type != kind
            || !n.domain.is_empty()
            || !n.attribute.is_empty()
            || n.output.len() != 1
            || n.output[0].is_empty()
            || n.input.len() != if kind == "Relu" { 1 } else { 2 }
        {
            return None;
        }
    }
    if nodes[0].input[0] != g.input[0].name
        || nodes[1].input[0] != nodes[0].output[0]
        || nodes[2].input[0] != nodes[1].output[0]
        || nodes[3].input[0] != nodes[2].output[0]
        || nodes[4].input[0] != nodes[3].output[0]
        || nodes[4].output[0] != g.output[0].name
    {
        return None;
    }
    let mut names = HashSet::new();
    if !names.insert(g.input[0].name.as_str()) {
        return None;
    }
    for t in &g.initializer {
        if !names.insert(t.name.as_str()) {
            return None;
        }
    }
    for n in nodes {
        if !names.insert(n.output[0].as_str()) {
            return None;
        }
    }
    let constants = [
        &nodes[0].input[1],
        &nodes[1].input[1],
        &nodes[3].input[1],
        &nodes[4].input[1],
    ];
    if constants
        .iter()
        .enumerate()
        .any(|(j, n)| constants[..j].contains(n))
    {
        return None;
    }
    let tensor = |name: &str| g.initializer.iter().find(|t| t.name == name);
    let first = tensor(constants[0])?;
    if first.dims.len() != 2 || first.dims[0] != inputs as i64 {
        return None;
    }
    let h = usize::try_from(first.dims[1]).ok()?;
    if !(1..=512).contains(&h) {
        return None;
    }
    // Optional annotations must describe this actual static dense chain.
    let mut annotations = HashSet::new();
    for bytes in &envelope.values {
        if !dense_type(&ValueEnvelope::decode(bytes.as_slice()).ok()?) {
            return None;
        }
        let v = ValueInfoProto::decode(bytes.as_slice()).ok()?;
        if !annotations.insert(v.name.clone()) {
            return None;
        }
        let expected = if nodes[..3].iter().any(|n| n.output[0] == v.name) {
            h
        } else if nodes[3..].iter().any(|n| n.output[0] == v.name) {
            2
        } else {
            return None;
        };
        if !singleton_shape_matches(&v, expected, layout) {
            return None;
        }
    }
    InputHeadSpec::new(
        inputs,
        h,
        parameters(first, &[inputs as i64, h as i64])?,
        parameters(tensor(constants[1])?, &[h as i64])?,
        parameters(tensor(constants[2])?, &[h as i64, 2])?,
        parameters(tensor(constants[3])?, &[2])?,
        Sha256::digest(bytes).into(),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OriginalPropertyFormat {
    ScalarRealV1,
    TensorFloat32V2,
}

impl OriginalPropertyFormat {
    fn token(self) -> &'static str {
        match self {
            Self::ScalarRealV1 => "scalar_v1_real",
            Self::TensorFloat32V2 => "tensor_v2_float32",
        }
    }
}

struct Property {
    spec: ny_onnx::vnnlib::VnnLibSpec,
    lower: Vec<f64>,
    upper: Vec<f64>,
    target: usize,
    adversary: usize,
    source_digest: [u8; 32],
    singleton_layout: OriginalSingletonLayout,
    format: OriginalPropertyFormat,
}

fn property(path: &Path, inputs: usize) -> Option<Property> {
    let bytes = read_capped(path, MAX_PROPERTY_BYTES)?;
    property_from_source(std::str::from_utf8(&bytes).ok()?, inputs)
}

fn property_from_source(content: &str, inputs: usize) -> Option<Property> {
    if content.len() as u64 > MAX_PROPERTY_BYTES {
        return None;
    }
    let declarations = ny_onnx::vnnlib::parse_vnnlib_assignment_declarations(content).ok()?;
    if declarations.is_empty() {
        let (spec, box_, atoms) =
            ny_onnx::vnnlib::parse_vnnlib_with_certified_scalar_singleton_property(content, inputs)
                .ok()?;
        return owned_property(
            content,
            inputs,
            spec,
            box_,
            atoms,
            OriginalPropertyFormat::ScalarRealV1,
        );
    }
    let [x, y] = declarations.as_slice() else {
        return None;
    };
    if x.kind != TensorDeclarationKind::Input
        || y.kind != TensorDeclarationKind::Output
        || x.name != "X"
        || y.name != "Y"
        || x.network != y.network
        || x.network.is_none()
        || x.element_type != "float32"
        || y.element_type != "float32"
        || x.shape != [1, inputs]
        || y.shape != [1, 2]
    {
        return None;
    }
    let (spec, box_, atoms) =
        ny_onnx::vnnlib::parse_vnnlib_with_certified_singleton_batch_property(content).ok()?;
    owned_property(
        content,
        inputs,
        spec,
        box_,
        atoms,
        OriginalPropertyFormat::TensorFloat32V2,
    )
}

fn owned_property(
    content: &str,
    inputs: usize,
    spec: ny_onnx::vnnlib::VnnLibSpec,
    box_: ny_onnx::vnnlib::CertifiedInputBox,
    atoms: ny_onnx::vnnlib::CertifiedRelationalOutputConjunction,
    format: OriginalPropertyFormat,
) -> Option<Property> {
    if spec.num_inputs != inputs
        || spec.num_outputs != 2
        || spec.is_disjunction
        || spec.dual_network.is_some()
    {
        return None;
    }
    let (target, adversary) = match atoms.atoms() {
        [CertifiedRelationalOutputAtom::LessEq(t, a)] => (*t, *a),
        [CertifiedRelationalOutputAtom::GreaterEq(a, t)] => (*t, *a),
        _ => return None, // equality remains unsafe; strict/non-relational forms are outside this route
    };
    if target >= 2 || adversary >= 2 || target == adversary {
        return None;
    }
    Some(Property {
        spec,
        lower: box_.lower().to_vec(),
        upper: box_.upper().to_vec(),
        target,
        adversary,
        source_digest: Sha256::digest(content.as_bytes()).into(),
        singleton_layout: OriginalSingletonLayout { inputs, outputs: 2 },
        format,
    })
}

fn unchanged_sources(onnx: &Path, vnnlib: &Path, model: &InputHeadSpec, p: &Property) -> bool {
    let Some(m) = read_capped(onnx, MAX_MODEL_BYTES) else {
        return false;
    };
    let Some(v) = read_capped(vnnlib, MAX_PROPERTY_BYTES) else {
        return false;
    };
    <[u8; 32]>::from(Sha256::digest(&m)) == model.authored_digest()
        && <[u8; 32]>::from(Sha256::digest(&v)) == p.source_digest
}

/// This carrier is minted only after the original owned singleton property
/// and raw model have been authenticated. It never imports a generic bound,
/// graph cover, augmented class, or a caller-authored numerical certificate.
pub(crate) struct BoundInputHeadContinuation {
    source: Arc<ny_propagate::margin_row::InputHeadContinuation>,
    model: InputHeadSpec,
    property: Property,
    onnx: PathBuf,
    vnnlib: PathBuf,
    authority: Instant,
    independent_full_root_mip: Mutex<Option<String>>,
}

impl std::fmt::Debug for BoundInputHeadContinuation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundInputHeadContinuation")
            .field("inputs", &self.model.inputs())
            .field("heads", &self.model.heads())
            .field("target", &self.property.target)
            .field("adversary", &self.property.adversary)
            .finish_non_exhaustive()
    }
}

impl BoundInputHeadContinuation {
    fn sources_live(&self) -> bool {
        Instant::now() < self.authority
            && ny_core::require_f64_interval_proof_environment().is_ok()
            && unchanged_sources(&self.onnx, &self.vnnlib, &self.model, &self.property)
            && Instant::now() < self.authority
    }

    /// Attach only to a second ordinary parse of the SAME original property
    /// and its actual widened f32 evaluation root. Scalar-network sign/tail
    /// authentication is repeated by the private producer on every use.
    pub(crate) fn admitted_source(
        &self,
        spec: &ny_onnx::vnnlib::VnnLibSpec,
        input: &ny_tensor::BoundedTensor,
    ) -> Option<Arc<ny_propagate::margin_row::InputHeadContinuation>> {
        let original = &self.property.spec;
        if spec.num_inputs != original.num_inputs
            || spec.num_outputs != original.num_outputs
            || spec.input_bounds != original.input_bounds
            || spec.declared_input_bounds != original.declared_input_bounds
            || spec.output_constraints != original.output_constraints
            || spec.output_constraint_clauses != original.output_constraint_clauses
            || spec.per_clause_input_bounds != original.per_clause_input_bounds
            || spec.is_disjunction != original.is_disjunction
            || spec.dual_network.is_some()
            || spec.version != original.version
            || !self.sources_live()
        {
            return None;
        }
        let lo = input.lower().as_slice()?;
        let hi = input.upper().as_slice()?;
        (self.source.root_box_matches(lo, hi) && self.sources_live())
            .then(|| Arc::clone(&self.source))
    }

    /// Only the exact complete MIP publication site can supply this opaque
    /// receipt. A successful return, bare status, stale capture or log text
    /// cannot bypass a missing Source sibling.
    #[cfg(feature = "mip")]
    pub(crate) fn note_independent_full_root_mip_publication(
        &self,
        network: &ny_propagate::Network,
        spec: &ny_onnx::vnnlib::VnnLibSpec,
        input: &ny_tensor::BoundedTensor,
        publication: super::beta_crown::MipUnsatPublication,
    ) -> bool {
        if self.admitted_source(spec, input).is_none()
            || !self.source.matches_original_network(network)
            || !self.sources_live()
        {
            return false;
        }
        let Ok(mut receipt) = self.independent_full_root_mip.lock() else {
            return false;
        };
        *receipt = Some(publication.into_captured_json());
        true
    }

    fn independent_mip_capture_matches(&self, captured: Option<&str>) -> bool {
        let Ok(receipt) = self.independent_full_root_mip.lock() else {
            return false;
        };
        match (receipt.as_deref(), captured) {
            (Some(expected), Some(actual)) => expected == actual,
            _ => false,
        }
    }

    fn finish_original(&self) -> Option<VnncompResult> {
        if !self.sources_live() {
            return None;
        }
        // A generic verified leaf, clip, or empty queue cannot complete this
        // cover. Only the private original-input partition authorizes G2.
        let token = self.source.try_finish(self.authority)?;
        if !self.sources_live() {
            return None;
        }
        let (report, fragment) = super::margin_row_bab::margin_row_spec_agreement_owned(
            &self.property.spec,
            self.property.target,
            &[self.property.adversary],
        );
        if !fragment.starts_with("g3=ran ") {
            return None;
        }
        let token = ny_propagate::certify::ratify(token, &report).ok()?;
        let receipt = super::unsat_receipt::UnsatReceipt::from_token(token)?;
        if !self.sources_live() {
            return None;
        }
        eprintln!("[g3] lane=margin_row_input_head_bridge {fragment} receipt=ok");
        // B2 (eng14 landing review): the same rule as `attempt_input_head`'s
        // direct publication. This Instance token comes from the private
        // Source partition, which closes OUTSIDE every margin-row frame, and
        // its only consumer (the sequential input-split lane) records no lane
        // tape, so nothing else streams this class. Claim exactly what was
        // computed — the one original class, as one `opaque`
        // `input_head_bridge` node over the whole region — with the
        // property's clauses record, only after G3 ratification, the receipt
        // and the final source re-check. Write-only, and a no-op under
        // Competition (no bundle is wired, hint capture is off). A refused
        // finish returned above and claims nothing.
        let (classes, _, _) = receipt.published_summary();
        super::proof_bundle::note_clauses(self.property.target, &classes);
        ny_propagate::certify::stream_opaque_claim(classes, "input_head_bridge");
        Some(VnncompResult::Unsat(receipt))
    }

    /// Trusted SAT remains independent. A Source-dependent generic UNSAT
    /// cannot use the legacy BetaCrownVerdict door. Authenticated nonpositive
    /// scalar MAX ranking is independent only if no earlier use set dependency.
    pub(crate) fn settle_result(
        &self,
        result: VnncompResult,
        captured: Option<&str>,
    ) -> VnncompResult {
        if matches!(&result, VnncompResult::Sat { .. }) {
            return result;
        }
        if matches!(&result, VnncompResult::Unsat(_))
            && self.independent_mip_capture_matches(captured)
        {
            let (report, fragment) = super::margin_row_bab::margin_row_spec_agreement_owned(
                &self.property.spec,
                self.property.target,
                &[self.property.adversary],
            );
            if fragment.starts_with("g3=ran ") && report.is_clear() && self.sources_live() {
                eprintln!("[input-head-bridge] independent_whole_root_mip=1 source_partition_required=0 receipt=existing_mip_authority");
                return result;
            }
            return VnncompResult::Unknown;
        }
        if let Some(proved) = self.finish_original() {
            return proved;
        }
        if self.source.dependent_use() && matches!(&result, VnncompResult::Unsat(_)) {
            eprintln!(
                "[input-head-bridge] dependent=1 complete_source_cover=0 publication=refused"
            );
            VnncompResult::Unknown
        } else {
            result
        }
    }
}

#[derive(Default)]
pub(crate) struct InputHeadAttempt {
    pub(crate) verdict: Option<VnncompResult>,
    pub(crate) continuation: Option<Arc<BoundInputHeadContinuation>>,
}

fn evaluation_box(p: &Property) -> Option<(Vec<f64>, Vec<f64>)> {
    // Exactly the conversion used by beta_crown::inputs; its one-ULP wider
    // root is a NEW source box, not an enlargement of a retained certificate.
    let (lo, hi) = p.spec.split_input_bounds_f32();
    if lo.len() != p.lower.len() || hi.len() != p.upper.len() {
        return None;
    }
    let lo: Vec<_> = lo.into_iter().map(f64::from).collect();
    let hi: Vec<_> = hi.into_iter().map(f64::from).collect();
    lo.iter()
        .zip(&hi)
        .zip(p.lower.iter().zip(&p.upper))
        .all(|((&l, &u), (&original_l, &original_u))| {
            l.is_finite() && u.is_finite() && l <= original_l && original_u <= u && l <= u
        })
        .then_some((lo, hi))
}

/// Verdict-only test adapter for synchronous optional input-head work.
/// Every rejected/expired/unsupported attempt falls through under the same
/// original deadline; no detached worker or restarted instance budget exists.
#[cfg(test)]
pub(crate) fn try_input_head(
    onnx: &Path,
    vnnlib: &Path,
    deadline: Option<Instant>,
    policy: &MarginRowRuntimePolicy,
) -> Option<VnncompResult> {
    attempt_input_head(onnx, vnnlib, deadline, policy, false)?.verdict
}

pub(crate) fn try_input_head_with_continuation(
    onnx: &Path,
    vnnlib: &Path,
    deadline: Option<Instant>,
    policy: &MarginRowRuntimePolicy,
) -> InputHeadAttempt {
    attempt_input_head(onnx, vnnlib, deadline, policy, true).unwrap_or_default()
}

fn attempt_input_head(
    onnx: &Path,
    vnnlib: &Path,
    deadline: Option<Instant>,
    policy: &MarginRowRuntimePolicy,
    retain: bool,
) -> Option<InputHeadAttempt> {
    let policy = policy.head_groups?;
    let deadline = deadline?;
    let slice = Instant::now()
        .checked_add(Duration::from_secs_f64(policy.max_seconds()))?
        .min(deadline);
    if Instant::now() >= slice
        || !ny_propagate::margin_row::margin_row_bab_enabled()
        || std::fs::metadata(onnx).ok()?.len() > MAX_MODEL_BYTES
    {
        return None;
    }
    let bytes = read_capped(onnx, MAX_MODEL_BYTES)?;
    ny_core::require_f64_interval_proof_environment().ok()?;
    // The original property, not an unknown model axis, binds batch one.
    // Every MatMul/Add/ReLU and fixed constant shape is then authenticated:
    // [1,n]*[n,h]+[h] -> [1,h] -> [1,h]*[h,2]+[2] -> [1,2].
    let p = property(vnnlib, input_width_hint(&bytes)?)?;
    if Instant::now() >= slice {
        return None;
    }
    let model = extract_bound(&bytes, Some(&p.singleton_layout))?;
    if Instant::now() >= slice {
        return None;
    }
    eprintln!(
        "[input-head-admission] graph=constant_right_float32 inputs={} head={} outputs=2 required=1 batch=1 binding=original_property property_format={} equality=unsafe authority=mathematical_onnx ieee_fp32_extra_enclosed=1 external_backend_qualified=0 phase_cover_max_heads={}",
        model.inputs(),
        model.heads(),
        p.format.token(),
        policy.phase_cover().map_or(0, |cover| cover.max_heads())
    );
    let (outcome, continuation) = if let Some((evaluation_lo, evaluation_hi)) =
        retain.then(|| evaluation_box(&p)).flatten()
    {
        ny_propagate::margin_row::verify_input_head_retained(
            &model,
            &p.lower,
            &p.upper,
            p.target,
            p.adversary,
            policy,
            deadline,
            &evaluation_lo,
            &evaluation_hi,
            slice,
        )
    } else {
        (
            ny_propagate::margin_row::verify_input_head(
                &model,
                &p.lower,
                &p.upper,
                p.target,
                p.adversary,
                policy,
                slice,
            ),
            None,
        )
    };
    let token = match outcome {
        InputHeadOutcome::Unsat(_, token) => token,
        InputHeadOutcome::Unknown(_) => {
            if Instant::now() >= deadline
                || !unchanged_sources(onnx, vnnlib, &model, &p)
                || ny_core::require_f64_interval_proof_environment().is_err()
            {
                return None;
            }
            return Some(InputHeadAttempt {
                verdict: None,
                continuation: continuation.map(|source| {
                    Arc::new(BoundInputHeadContinuation {
                        source,
                        model,
                        property: p,
                        onnx: onnx.to_owned(),
                        vnnlib: vnnlib.to_owned(),
                        authority: deadline,
                        independent_full_root_mip: Mutex::new(None),
                    })
                }),
            });
        }
    };
    if Instant::now() >= deadline || !unchanged_sources(onnx, vnnlib, &model, &p) {
        return None;
    }
    let (report, fragment) =
        super::margin_row_bab::margin_row_spec_agreement_owned(&p.spec, p.target, &[p.adversary]);
    if !fragment.starts_with("g3=ran ") {
        return None;
    }
    let token = ny_propagate::certify::ratify(token, &report).ok()?;
    let receipt = super::unsat_receipt::UnsatReceipt::from_token(token)?;
    if Instant::now() >= deadline
        || ny_core::require_f64_interval_proof_environment().is_err()
        || !unchanged_sources(onnx, vnnlib, &model, &p)
        || Instant::now() >= deadline
    {
        return None;
    }
    eprintln!("[g3] lane=margin_row_input_head {fragment} receipt=ok");
    // B2 (review(eng14)): this route closes its one class OUTSIDE every
    // margin-row frame and returns before the lane proof capture is armed, so
    // nothing else streams it. Claim exactly what it computed — the one
    // original class, as one `opaque` `input_head` node (the checker has no
    // input-head rung) — with the property's clauses record, only now that
    // every check above passed and the receipt exists. Write-only, and a
    // no-op under Competition (no bundle is wired, hint capture is off). The
    // classes are the published token's own, the verdict record's `required`.
    let (classes, _, _) = receipt.published_summary();
    super::proof_bundle::note_clauses(p.target, &classes);
    ny_propagate::certify::stream_opaque_claim(classes, "input_head");
    Some(InputHeadAttempt {
        verdict: Some(VnncompResult::Unsat(receipt)),
        continuation: None,
    })
}

#[cfg(test)]
#[path = "safenlp_input_head_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "safenlp_input_head_bridge_tests.rs"]
mod bridge_tests;

#[cfg(test)]
#[path = "safenlp_input_head_vnnlib1_tests.rs"]
mod vnnlib1_tests;
